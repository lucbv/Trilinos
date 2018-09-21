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
#ifndef MUELU_BLACKBOXCONNECTIVITY_DECL_HPP
#define MUELU_BLACKBOXCONNECTIVITY_DECL_HPP

#include "MueLu_ConfigDefs.hpp"
#include "MueLu_BaseClass.hpp"
#include "MueLu_Exceptions.hpp"
#include "MueLu_BlackBoxConnectivity_fwd.hpp"

namespace MueLu {

/*!
    @class BlackBoxConnectivity
    @brief Container class for aggregation information.

    @ingroup Aggregation

    Structure holding aggregate information. Right now, nElementData, IsRoot,
    Vertex2AggId, procWinner are populated.  This allows us to look at a node
    and determine the aggregate to which it has been assigned and the id of the
    processor that owns this aggregate. It is not so easy to determine vertices
    within the kth aggregate or the size of the kth aggregate. Thus, it might be
    useful to have a secondary structure which would be a rectangular CrsGraph
    where rows (or vertices) correspond to aggregates and colunmns (or edges)
    correspond to nodes. While not strictly necessary, it might be convenient.
*/

  template <class LocalOrdinal  = int, class GlobalOrdinal = LocalOrdinal, class Node = KokkosClassic::DefaultNode::DefaultNodeType>
  class BlackBoxConnectivity : public BaseClass {
#undef MUELU_BLACKBOXCONNECTIVITY_SHORT
#include "MueLu_UseShortNamesOrdinal.hpp"

  public:

    /*! @brief Standard constructor for BlackBoxConnectivity structure
     *
     * Standard constructor of aggregates takes a Graph object as parameter.
     * Uses the graph.GetImportMap() to initialize the internal vector for mapping nodes to (local) aggregate ids as well as
     * the mapping of node to the owning processor id.
     *
     */
    BlackBoxConnectivity(const LO numLocalElements) : numLocalElements_(numLocalElements) {
      elementsData.resize(numLocalElements_);
    }

    /*! @brief Destructor
     *
     */
    virtual ~BlackBoxConnectivity() { }

    //! @name Overridden from Teuchos::Describable
    //@{

    //! Return a simple one-line description of this object.
    std::string description() const;

    //! Print the object with some verbosity level to an FancyOStream object.
    //using MueLu::Describable::describe; // overloading, not hiding
    void print(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel = verbLevel_default) const;

    void setElement(const LO elementIdx, ArrayView<const LO> connectivity,
                    ArrayView<const LO> dimensions, ArrayView<const LO> isMeshEdge);

    struct elementEntry {
      // Define data to hold
      Array<LO> connectivity_;
      Array<LO> dimensions_;
      Array<LO> isMeshEdge_;

      // Define empty constructor
      elementEntry() {}

      // Define constructor for on the fly object creation outside this class
      elementEntry(ArrayView<const LO> connectivity, ArrayView<const LO> dimensions,
                   ArrayView<const LO> isMeshEdge) {
        // Check that inputs have correct dimensions
        TEUCHOS_TEST_FOR_EXCEPTION(dimensions.size() != 3, Exceptions::RuntimeError, "dimensions needs to be of size 3.");
        TEUCHOS_TEST_FOR_EXCEPTION(isMeshEdge.size() != 6, Exceptions::RuntimeError, "isMeshEdge needs to be of size 6.");

        connectivity = connectivity;
        dimensions_ = dimensions;
        isMeshEdge_ = isMeshEdge;
      }
    };

  private:

    const LO numLocalElements_;

    Array<elementEntry> elementsData;

  };

} //namespace MueLu

#define MUELU_BLACKBOXCONNECTIVITY_SHORT
#endif // MUELU_BLACKBOXCONNECTIVITY_DECL_HPP
