// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef MUELU_AGGREGATIONPHASE2BALGORITHM_KOKKOS_DEF_HPP
#define MUELU_AGGREGATIONPHASE2BALGORITHM_KOKKOS_DEF_HPP

#ifdef HAVE_MUELU_KOKKOS_REFACTOR

#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>

#include <Xpetra_Vector.hpp>

#include "MueLu_AggregationPhase2bAlgorithm_kokkos_decl.hpp"

#include "MueLu_Aggregates_kokkos.hpp"
#include "MueLu_Exceptions.hpp"
#include "MueLu_LWGraph_kokkos.hpp"
#include "MueLu_Monitor.hpp"

namespace MueLu {

  // Try to stick unaggregated nodes into a neighboring aggregate if they are
  // not already too big
  template <class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggregationPhase2bAlgorithm_kokkos<LocalOrdinal, GlobalOrdinal, Node>::
  BuildAggregates(const ParameterList& params,
                  const LWGraph_kokkos& graph,
                  Aggregates_kokkos& aggregates,
                  Kokkos::View<unsigned*, typename LWGraph_kokkos::memory_space>& aggStat,
                  LO& numNonAggregatedNodes) const {

    if(params.get<bool>("aggregation: deterministic")) {
      Monitor m(*this, "BuildAggregatesDeterministic");
      BuildAggregatesDeterministic(params, graph, aggregates, aggStat, numNonAggregatedNodes);
    } else {
      Monitor m(*this, "BuildAggregatesRandom");
      BuildAggregatesRandom(params, graph, aggregates, aggStat, numNonAggregatedNodes);
    }

  } // BuildAggregates

  template <class LO, class GO, class Node>
  void AggregationPhase2bAlgorithm_kokkos<LO, GO, Node>::
  BuildAggregatesRandom(const ParameterList& params,
                        const LWGraph_kokkos& graph,
                        Aggregates_kokkos& aggregates,
                        Kokkos::View<unsigned*, typename LWGraph_kokkos::memory_space>& aggStat,
                        LO& numNonAggregatedNodes) const {
    using memory_space    = typename LWGraph_kokkos::memory_space;
    using execution_space = typename LWGraph_kokkos::execution_space;
    using scratch_space   = typename execution_space::scratch_memory_space;
    using ScratchViewType = typename Kokkos::View<LO*, scratch_space,
                                                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

    const LO  numRows = graph.GetNodeNumVertices();
    const int myRank  = graph.GetComm()->getRank();

    auto vertex2AggId  = aggregates.GetVertex2AggId()->template getLocalView<memory_space>();
    auto procWinner    = aggregates.GetProcWinner()  ->template getLocalView<memory_space>();

    // LO numLocalAggregates = aggregates.GetNumAggregates();

    const int defaultConnectWeight = 100;
    const int penaltyConnectWeight = 10;

    // This actually corresponds to the maximum number of entries per row in the matrix.
    const size_t maxNumNeighbors = graph.getNodeMaxNumRowEntries();
    int scratch_size = ScratchViewType::shmem_size( 3*maxNumNeighbors );

    Kokkos::View<int*, memory_space> connectWeight("connectWeight", numRows);
    Kokkos::View<int*, memory_space> aggPenalties("aggPenalties",  numRows);

    Kokkos::deep_copy(connectWeight, defaultConnectWeight);

    // taw: by running the aggregation routine more than once there is a chance that also
    // non-aggregated nodes with a node distance of two are added to existing aggregates.
    // Assuming that the aggregate size is 3 in each direction running the algorithm only twice
    // should be sufficient.
    // lbv: If the prior phase of aggregation where run without specifying an aggregate size,
    // the distance 2 coloring and phase 1 aggregation actually guarantee that only one iteration
    // is needed to reach distance 2 neighbors.
    int maxIters = 2;
    int maxNodesPerAggregate = params.get<int>("aggregation: max agg size");
    if(maxNodesPerAggregate == std::numeric_limits<int>::max()) {maxIters = 1;}
    for (int iter = 0; iter < maxIters; ++iter) {
      // total work = numberOfTeams * teamSize
      typedef typename Kokkos::TeamPolicy<execution_space>::member_type  member_type;
      LO tmpNumNonAggregatedNodes = 0;
      Kokkos::TeamPolicy<execution_space> outerPolicy(numRows, Kokkos::AUTO);
      Kokkos::parallel_reduce("Aggregation Phase 2b: aggregates expansion",
                              outerPolicy.set_scratch_size(0, Kokkos::PerTeam( scratch_size ) ),
                              KOKKOS_LAMBDA (const member_type &teamMember,
                                             LO& lNumNonAggregatedNodes) {

                                // Retrieve the id of the vertex we are currently working on and
                                // allocate view locally so that threads do not trash the weigth
                                // when working on the same aggregate.
                                const int vertexIdx = teamMember.league_rank();
                                int numAggregatedNeighbors = 0;
                                ScratchViewType aggregatedNeighbors(teamMember.team_scratch( 0 ),
                                                                    maxNumNeighbors);
                                ScratchViewType vertex2AggLID(teamMember.team_scratch( 0 ),
                                                                  maxNumNeighbors);
                                ScratchViewType aggWeight(teamMember.team_scratch( 0 ),
                                                              maxNumNeighbors);

                                if (aggStat(vertexIdx) == READY) {

                                  // neighOfINode should become scratch and shared among
                                  // threads in the team...
                                  auto neighOfINode = graph.getNeighborVertices(vertexIdx);

                                  // create a mapping from neighbor "lid" to aggregate "lid"
                                  Kokkos::single( Kokkos::PerTeam( teamMember ), [&] () {
                                      int aggLIDCount = 0;
                                      for (int j = 0; j < neighOfINode.length; ++j) {
                                        LO neigh = neighOfINode(j);
                                        if( graph.isLocalNeighborVertex(neigh) &&
                                            (aggStat(neigh) == AGGREGATED) ) {
                                          aggregatedNeighbors(numAggregatedNeighbors) = j;

                                          bool useNewLID = true;
                                          for(int k = 0; k < numAggregatedNeighbors; ++k) {
                                            LO lowerNeigh = neighOfINode(aggregatedNeighbors(k));
                                            if(vertex2AggId(neigh, 0) == vertex2AggId(lowerNeigh, 0)) {
                                              vertex2AggLID(numAggregatedNeighbors) = vertex2AggLID(k);
                                              useNewLID = false;
                                            }
                                          }
                                          if(useNewLID) {
                                            vertex2AggLID(numAggregatedNeighbors) = aggLIDCount;
                                            ++aggLIDCount;
                                          }

                                          ++numAggregatedNeighbors;
                                        }
                                      }
                                    });

                                  // LBV on Sept 13, 2019: double check that localNeigh
                                  // is not needed here...
                                  for (int j = 0; j < numAggregatedNeighbors; j++) {
                                    // LO localNeigh = aggregatedNeighbors(j);
                                    LO neigh = neighOfINode(j);

                                    aggWeight(vertex2AggLID(j)) = aggWeight(vertex2AggLID(j))
                                      + connectWeight(neigh);
                                  }

                                  int bestScore   = -100000;
                                  int bestAggId   = -1;
                                  int bestConnect = -1;

                                  for (int j = 0; j < numAggregatedNeighbors; j++) {
                                    LO  localNeigh = aggregatedNeighbors(j);
                                    LO  neigh = neighOfINode(localNeigh);
                                    int aggId = vertex2AggId(neigh, 0);
                                    int score = aggWeight(vertex2AggLID(j)) - aggPenalties(aggId);

                                    if (score > bestScore) {
                                      bestAggId   = aggId;
                                      bestScore   = score;
                                      bestConnect = connectWeight(neigh);

                                    } else if (aggId == bestAggId
                                               && connectWeight(neigh) > bestConnect) {
                                      bestConnect = connectWeight(neigh);
                                    }

                                    // Reset the weights for the next loop
                                    // LBV: this looks a little suspicious, it would probably
                                    // need to be taken out of this inner for loop...
                                    aggWeight(vertex2AggLID(j)) = 0;
                                  }

                                  // Do the actual aggregate update with a single thread!
                                  Kokkos::single( Kokkos::PerTeam( teamMember ), [&] () {
                                      if (bestScore >= 0) {
                                        aggStat     (vertexIdx)    = AGGREGATED;
                                        vertex2AggId(vertexIdx, 0) = bestAggId;
                                        procWinner  (vertexIdx, 0) = myRank;

                                        lNumNonAggregatedNodes--;

                                        // This does not protect bestAggId's aggPenalties from being
                                        // fetched by another thread before this update happens, it just
                                        // guarantees that the update is performed correctly...
                                        Kokkos::atomic_add(&aggPenalties(bestAggId), 1);
                                        connectWeight(vertexIdx) = bestConnect - penaltyConnectWeight;
                                      }
                                    });
                                }
                              }, tmpNumNonAggregatedNodes);
      numNonAggregatedNodes += tmpNumNonAggregatedNodes;
    } // loop over k

  } // BuildAggregatesRandom



  template <class LO, class GO, class Node>
  void AggregationPhase2bAlgorithm_kokkos<LO, GO, Node>::
  BuildAggregatesDeterministic(const ParameterList& params,
                               const LWGraph_kokkos& graph,
                               Aggregates_kokkos& aggregates,
                               Kokkos::View<unsigned*, typename LWGraph_kokkos::memory_space>& aggStat,
                               LO& numNonAggregatedNodes) const {
    using memory_space    = typename LWGraph_kokkos::memory_space;
    using execution_space = typename LWGraph_kokkos::execution_space;

    const LO  numRows = graph.GetNodeNumVertices();
    const int myRank  = graph.GetComm()->getRank();

    auto vertex2AggId     = aggregates.GetVertex2AggId()->template getLocalView<memory_space>();
    auto procWinner       = aggregates.GetProcWinner()  ->template getLocalView<memory_space>();
    auto colors           = aggregates.GetGraphColors();
    const LO numColors    = aggregates.GetGraphNumColors();
    LO numLocalAggregates = aggregates.GetNumAggregates();

    const int defaultConnectWeight = 100;
    const int penaltyConnectWeight = 10;

    Kokkos::View<int*, memory_space> aggWeight    ("aggWeight",     numLocalAggregates);
    Kokkos::View<int*, memory_space> connectWeight("connectWeight", numRows);
    Kokkos::View<int*, memory_space> aggPenalties ("aggPenalties",  numRows);

    Kokkos::deep_copy(connectWeight, defaultConnectWeight);

    // We do this cycle twice.
    // I don't know why, but ML does it too
    // taw: by running the aggregation routine more than once there is a chance that also
    // non-aggregated nodes with a node distance of two are added to existing aggregates.
    // Assuming that the aggregate size is 3 in each direction running the algorithm only twice
    // should be sufficient.
    int maxIters = 2;
    int maxNodesPerAggregate = params.get<int>("aggregation: max agg size");
    if(maxNodesPerAggregate == std::numeric_limits<int>::max()) {maxIters = 1;}
    for (int iter = 0; iter < maxIters; ++iter) {
      for(LO color = 1; color <= numColors; color++) {
        Kokkos::deep_copy(aggWeight, 0);

        //the reduce counts how many nodes are aggregated by this phase,
        //which will then be subtracted from numNonAggregatedNodes
        LO numAggregated = 0;
        Kokkos::parallel_reduce("Aggregation Phase 2b: aggregates expansion",
                                Kokkos::RangePolicy<execution_space>(0, numRows),
                                KOKKOS_LAMBDA (const LO i, LO& tmpNumAggregated) {
                                  if (aggStat(i) != READY || colors(i) != color)
                                    return;

                                  auto neighOfINode = graph.getNeighborVertices(i);
                                  for (int j = 0; j < neighOfINode.length; j++) {
                                    LO neigh = neighOfINode(j);

                                    // We don't check (neigh != i), as it is covered by checking
                                    // (aggStat[neigh] == AGGREGATED)
                                    if (graph.isLocalNeighborVertex(neigh) &&
                                        aggStat(neigh) == AGGREGATED)
                                      Kokkos::atomic_add(&aggWeight(vertex2AggId(neigh, 0), 0),
                                                         connectWeight(neigh));
                                  }

                                  int bestScore   = -100000;
                                  int bestAggId   = -1;
                                  int bestConnect = -1;

                                  for (int j = 0; j < neighOfINode.length; j++) {
                                    LO neigh = neighOfINode(j);

                                    if (graph.isLocalNeighborVertex(neigh) &&
                                        aggStat(neigh) == AGGREGATED) {
                                      auto aggId = vertex2AggId(neigh, 0);
                                      int score = aggWeight(aggId) - aggPenalties(aggId);

                                      if (score > bestScore) {
                                        bestAggId   = aggId;
                                        bestScore   = score;
                                        bestConnect = connectWeight(neigh);

                                      } else if (aggId == bestAggId &&
                                                 connectWeight(neigh) > bestConnect) {
                                        bestConnect = connectWeight(neigh);
                                      }
                                    }
                                  }
                                  if (bestScore >= 0) {
                                    aggStat(i, 0)      = AGGREGATED;
                                    vertex2AggId(i, 0) = bestAggId;
                                    procWinner(i, 0)   = myRank;

                                    Kokkos::atomic_add(&aggPenalties(bestAggId), 1);
                                    connectWeight(i) = bestConnect - penaltyConnectWeight;
                                    tmpNumAggregated++;
                                  }
                                }, numAggregated); //parallel_for
        numNonAggregatedNodes -= numAggregated;
      }
    } // loop over k

  } // BuildAggregatesDeterministic
} // end namespace

#endif // HAVE_MUELU_KOKKOS_REFACTOR
#endif // MUELU_AGGREGATIONPHASE2BALGORITHM_KOKKOS_DEF_HPP
