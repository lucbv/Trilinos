/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
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
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOSSPARSE_IMPL_SPMV_STRUCT_DEF_HPP_
#define KOKKOSSPARSE_IMPL_SPMV_STRUCT_DEF_HPP_

#include "Kokkos_InnerProductSpaceTraits.hpp"
#include "KokkosBlas1_scal.hpp"
#include "KokkosSparse_CrsMatrix.hpp"

namespace KokkosSparse {
namespace Impl {

enum {FD, FE};

// This TransposeFunctor is functional, but not necessarily performant.
template<class AMatrix,
         class XVector,
         class YVector,
         int dobeta,
         bool conjugate>
struct SPMV_Struct_Transpose_Functor {
  typedef typename AMatrix::execution_space            execution_space;
  typedef typename AMatrix::non_const_ordinal_type     ordinal_type;
  typedef typename AMatrix::non_const_value_type       value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type            team_member;
  typedef Kokkos::Details::ArithTraits<value_type>     ATV;
  typedef typename YVector::non_const_value_type       coefficient_type;
  typedef typename YVector::non_const_value_type       y_value_type;

  const coefficient_type alpha;
  AMatrix m_A;
  XVector m_x;
  const coefficient_type beta;
  YVector m_y;
  const ordinal_type rows_per_thread;

  SPMV_Struct_Transpose_Functor (const coefficient_type& alpha_,
                                 const AMatrix& m_A_,
                                 const XVector& m_x_,
                                 const coefficient_type& beta_,
                                 const YVector& m_y_,
                                 const ordinal_type rows_per_thread_) :
    alpha (alpha_), m_A (m_A_), m_x (m_x_),
    beta (beta_), m_y (m_y_),
    rows_per_thread (rows_per_thread_)
  {}

  KOKKOS_INLINE_FUNCTION void
  operator() (const team_member& dev) const
  {
    // This should be a thread loop as soon as we can use C++11
    for (ordinal_type loop = 0; loop < rows_per_thread; ++loop) {
      // iRow represents a row of the matrix, so its correct type is
      // ordinal_type.
      const ordinal_type iRow = (static_cast<ordinal_type> (dev.league_rank() * dev.team_size() + dev.team_rank()))
                                * rows_per_thread + loop;
      if (iRow >= m_A.numRows ()) {
        return;
      }

      const auto row = m_A.rowConst (iRow);
      const ordinal_type row_length = row.length;

#ifdef __CUDA_ARCH__
      for (ordinal_type iEntry = static_cast<ordinal_type> (threadIdx.x);
           iEntry < row_length;
           iEntry += static_cast<ordinal_type> (blockDim.x))
#else
      for (ordinal_type iEntry = 0;
           iEntry < row_length;
           iEntry ++)
#endif
      {
        const value_type val = conjugate ?
          ATV::conj (row.value(iEntry)) :
          row.value(iEntry);
        const ordinal_type ind = row.colidx(iEntry);

        Kokkos::atomic_add (&m_y(ind), static_cast<y_value_type> (alpha * val * m_x(iRow)));
      }
    }
  }
};

template<class mat_structure,
         class AMatrix,
         class XVector,
         class YVector,
         int dobeta,
         bool conjugate>
struct SPMV_Struct_Functor {
  typedef typename AMatrix::non_const_size_type              size_type;
  typedef typename AMatrix::non_const_ordinal_type           ordinal_type;
  typedef typename AMatrix::non_const_value_type             value_type;
  typedef typename AMatrix::execution_space                  execution_space;
  typedef typename KokkosSparse::SparseRowViewConst<AMatrix> row_view_const;
  typedef typename Kokkos::TeamPolicy<execution_space>       team_policy;
  typedef typename team_policy::member_type                  team_member;
  typedef Kokkos::Details::ArithTraits<value_type>           ATV;
  typedef typename mat_structure::HostMirror                 mat_structure_h;

  // Tags to perform SPMV on interior and boundaries
  struct interior3ptTag{};    // 1D FD and FE discretization
  struct interior5ptTag{};    // 2D FD discretization
  struct interior9ptTag{};    // 2D FE discretization
  struct interior7ptTag{};    // 3D FD discretization
  struct interior27ptTag{};   // 3D FE discretization
  struct exterior1DTag{};
  struct exterior2DTag{};
  struct exterior3DTag{};

  const mat_structure structure;
  const value_type alpha;
  AMatrix  m_A;
  XVector m_x;
  const value_type beta;
  YVector m_y;
  ordinal_type numInterior, numExterior;

  const int64_t rows_per_team;

  Kokkos::View<ordinal_type[5], execution_space> column_offsets_5pt;
  Kokkos::View<ordinal_type[9], execution_space> column_offsets_9pt;

  SPMV_Struct_Functor (const mat_structure structure_,
                       const value_type alpha_,
                       const AMatrix m_A_,
                       const XVector m_x_,
                       const value_type beta_,
                       const YVector m_y_,
                       const int64_t rows_per_team_) :
    structure(structure_),
    alpha (alpha_), m_A (m_A_), m_x (m_x_),
    beta (beta_), m_y (m_y_),
    rows_per_team (rows_per_team_),
    column_offsets_5pt("column offsets 5pts stencil"),
    column_offsets_9pt("column offsets 9pts stencil")
  {
    static_assert (static_cast<int> (XVector::rank) == 1,
                   "XVector must be a rank 1 View.");
    static_assert (static_cast<int> (YVector::rank) == 1,
                   "YVector must be a rank 1 View.");
  }

  void compute(mat_structure_h structure_h, const int stencil_type, const int64_t worksets,
               const int team_size, const int vector_length) {

    int numDimensions = structure_h.extent(0);

    if(numDimensions == 1) {
      // Treat interior points using structured algorithm
      numInterior = structure_h(0,0) - 2;
      if(numInterior > 0) {
        Kokkos::TeamPolicy<interior3ptTag,
                           execution_space,
                           Kokkos::Schedule<Kokkos::Static> > policy(1,1);
        if(team_size < 0) {
          policy = Kokkos::TeamPolicy<interior3ptTag, execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets,Kokkos::AUTO,vector_length);
        } else {
          policy = Kokkos::TeamPolicy<interior3ptTag, execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets,team_size,vector_length);
        }
        Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: interior", policy, *this);
      }

      // Treat exterior points using unstructured algorithm
      numExterior = structure_h(0,0) - numInterior;
      if(numExterior > 0) {
        Kokkos::RangePolicy<exterior1DTag,
                            execution_space,
                            Kokkos::Schedule<Kokkos::Static> > policy(0, numExterior);
        Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: exterior", policy, *this);
      }

    } else if(numDimensions == 2) {
      // Treat interior points using structured algorithm
      numInterior = (structure_h(0,0) - 2)*(structure_h(1,0) - 2);
      if(numInterior > 0) {
        if(stencil_type == 1) {
          Kokkos::TeamPolicy<interior5ptTag,
                             execution_space,
                             Kokkos::Schedule<Kokkos::Static> > policy(1,1);
          if(team_size < 0) {
            policy = Kokkos::TeamPolicy<interior5ptTag, execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets,Kokkos::AUTO,vector_length);
          } else {
            policy = Kokkos::TeamPolicy<interior5ptTag, execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets,team_size,vector_length);
          }
          Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: interior", policy, *this);
        } else if(stencil_type == 2) {
          Kokkos::TeamPolicy<interior9ptTag,
                             execution_space,
                             Kokkos::Schedule<Kokkos::Dynamic> > policy(1,1);
          if(team_size < 0) {
            policy = Kokkos::TeamPolicy<interior9ptTag, execution_space, Kokkos::Schedule<Kokkos::Dynamic> >(worksets,Kokkos::AUTO,vector_length);
          } else {
            policy = Kokkos::TeamPolicy<interior9ptTag, execution_space, Kokkos::Schedule<Kokkos::Dynamic> >(worksets,team_size,vector_length);
          }
          Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: interior", policy, *this);
        }
      }

      // Treat exterior points using unstructured algorithm
      numExterior = structure_h(0,0)*structure_h(1,0) - numInterior;
      if(numExterior > 0) {
        Kokkos::RangePolicy<exterior2DTag,
                            execution_space,
                            Kokkos::Schedule<Kokkos::Static> > policy(0, numExterior);
        Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: exterior", policy, *this);
      }
    } else if(numDimensions == 3) {
      // Treat interior points using structured algorithm
      numInterior = (structure_h(2,0) - structure_h(2,1) - structure_h(2,2))
        *(structure_h(1,0) - structure_h(1,1) - structure_h(1,2))
        *(structure_h(0,0) - structure_h(0,1) - structure_h(0,2));
      if(numInterior > 0) {
        if(stencil_type == 1) {
          Kokkos::TeamPolicy<interior7ptTag,
                             execution_space,
                             Kokkos::Schedule<Kokkos::Static> > policy(1,1);
          if(team_size < 0) {
            policy = Kokkos::TeamPolicy<interior7ptTag, execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets,Kokkos::AUTO,vector_length);
          } else {
            policy = Kokkos::TeamPolicy<interior7ptTag, execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets,team_size,vector_length);
          }
          Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: interior", policy, *this);
        } else if(stencil_type == 2) {
          Kokkos::TeamPolicy<interior27ptTag,
                             execution_space,
                             Kokkos::Schedule<Kokkos::Dynamic> > policy(1,1);
          if(team_size < 0) {
            policy = Kokkos::TeamPolicy<interior27ptTag, execution_space, Kokkos::Schedule<Kokkos::Dynamic> >(worksets,Kokkos::AUTO,vector_length);
          } else {
            policy = Kokkos::TeamPolicy<interior27ptTag, execution_space, Kokkos::Schedule<Kokkos::Dynamic> >(worksets,team_size,vector_length);
          }
          Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: interior", policy, *this);
        }
      }

      // Treat exterior points using unstructured algorithm
      numExterior = structure_h(0,0)*structure_h(1,0)*structure_h(2,0) - numInterior;
      if(numExterior > 0) {
        Kokkos::RangePolicy<exterior3DTag,
                            execution_space,
                            Kokkos::Schedule<Kokkos::Static> > policy(0, numExterior);
        Kokkos::parallel_for("KokkosSparse::spmv_struct<NoTranspose,Static>: exterior", policy, *this);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const interior3ptTag&, const team_member& dev) const
  {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(dev, 0, rows_per_team), [&] (const ordinal_type& loop) {
        const ordinal_type interiorIdx = static_cast<ordinal_type> ( dev.league_rank() ) * rows_per_team + loop;
        if(interiorIdx >= numInterior) { return; }

        ordinal_type rowIdx;
        rowIdx = interiorIdx + 1;

        const size_type rowOffset = m_A.graph.row_map(rowIdx);
        const value_type* value_ptr = m_A.values.data() + rowOffset;
        m_y(rowIdx) = beta*m_y(rowIdx) + alpha*(*(value_ptr+0)*m_x(rowIdx - 1)
                                                + *(value_ptr+1)*m_x(rowIdx)
                                                + *(value_ptr+2)*m_x(rowIdx + 1));
      });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const interior5ptTag&, const team_member& dev) const
  {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(dev, 0, rows_per_team), [&] (const ordinal_type& loop) {
        const ordinal_type interiorIdx = static_cast<ordinal_type> ( dev.league_rank() ) * rows_per_team + loop;
        if(interiorIdx >= numInterior) { return; }

        ordinal_type i, j, rowIdx;
        j = interiorIdx / (structure(0,0) - 2) + 1;
        i = interiorIdx % (structure(0,0) - 2) + 1;
        rowIdx = j*structure(0,0) + i;

        const size_type rowOffset = m_A.graph.row_map(rowIdx);
        const value_type* value_ptr = m_A.values.data() + rowOffset;
        m_y(rowIdx) = beta*m_y(rowIdx) + alpha*(*(value_ptr+0)*m_x(rowIdx - structure(0,0))
                                                + *(value_ptr+1)*m_x(rowIdx - 1)
                                                + *(value_ptr+2)*m_x(rowIdx)
                                                + *(value_ptr+3)*m_x(rowIdx + 1)
                                                + *(value_ptr+4)*m_x(rowIdx + structure(0,0)) );
      });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const interior9ptTag&, const team_member& dev) const
  {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(dev, 0, rows_per_team), [&] (const ordinal_type& loop) {
        const ordinal_type interiorIdx = static_cast<ordinal_type> ( dev.league_rank() ) * rows_per_team + loop;
        if(interiorIdx >= numInterior) { return; }

        ordinal_type i, j, rowIdx;
        j = interiorIdx / (structure(0,0) - 2) + 1;
        i = interiorIdx % (structure(0,0) - 2) + 1;
        rowIdx = j*structure(0,0) + i;

        const size_type rowOffset = m_A.graph.row_map(rowIdx);
        const value_type* value_ptr = &(m_A.values(rowOffset));
        m_y(rowIdx) = beta*m_y(rowIdx) + alpha*(*(value_ptr+0)*m_x(rowIdx - structure(0,0) - 1)
                                                + *(value_ptr+1)*m_x(rowIdx - structure(0,0))
                                                + *(value_ptr+2)*m_x(rowIdx - structure(0,0) + 1)
                                                + *(value_ptr+3)*m_x(rowIdx - 1)
                                                + *(value_ptr+4)*m_x(rowIdx)
                                                + *(value_ptr+5)*m_x(rowIdx + 1)
                                                + *(value_ptr+6)*m_x(rowIdx + structure(0,0) - 1)
                                                + *(value_ptr+7)*m_x(rowIdx + structure(0,0))
                                                + *(value_ptr+8)*m_x(rowIdx + structure(0,0) + 1) );
      });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const interior7ptTag&, const team_member& dev) const
  {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(dev, 0, rows_per_team), [&] (const ordinal_type& loop) {
        const ordinal_type interiorIdx = static_cast<ordinal_type> ( dev.league_rank() ) * rows_per_team + loop;
        if(interiorIdx >= numInterior) { return; }

        ordinal_type i, j, k, rowIdx, rem;
        k = interiorIdx / ((structure(0,0) - structure(0,1) - structure(0,2))
                           *(structure(1,0) - structure(1,1) - structure(1,2)));
        rem  = interiorIdx % ((structure(0,0) - structure(0,1) - structure(0,2))
                              *(structure(1,0) - structure(1,1) - structure(1,2)));
        j = rem / (structure(0,0) - structure(0,1) - structure(0,2));
        i = rem % (structure(0,0) - structure(0,1) - structure(0,2));
        rowIdx = (k + structure(2,1))*structure(1,0)*structure(0,0)
          + (j + structure(1,1))*structure(0,0) + (i + structure(0,1));

        const size_type rowOffset = m_A.graph.row_map(rowIdx);
        const value_type* value_ptr = &(m_A.values(rowOffset));
        m_y(rowIdx) = beta*m_y(rowIdx) + alpha*(*(value_ptr+0)*m_x(rowIdx - structure(0,0)*structure(1,0))
                                                + *(value_ptr+1)*m_x(rowIdx - structure(0,0))
                                                + *(value_ptr+2)*m_x(rowIdx - 1)
                                                + *(value_ptr+3)*m_x(rowIdx)
                                                + *(value_ptr+4)*m_x(rowIdx + 1)
                                                + *(value_ptr+5)*m_x(rowIdx + structure(0,0))
                                                + *(value_ptr+6)*m_x(rowIdx + structure(0,0)*structure(1,0)));
      });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const interior27ptTag&, const team_member& dev) const
  {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(dev, 0, rows_per_team), [&] (const ordinal_type& loop) {
        const ordinal_type interiorIdx = static_cast<ordinal_type> ( dev.league_rank() ) * rows_per_team + loop;
        if(interiorIdx >= numInterior) { return; }

        ordinal_type i, j, k, rowIdx, rem;
        k = interiorIdx / ((structure(0,0) - structure(0,1) - structure(0,2))
                           *(structure(1,0) - structure(1,1) - structure(1,2)));
        rem  = interiorIdx % ((structure(0,0) - structure(0,1) - structure(0,2))
                              *(structure(1,0) - structure(1,1) - structure(1,2)));
        j = rem / (structure(0,0) - structure(0,1) - structure(0,2));
        i = rem % (structure(0,0) - structure(0,1) - structure(0,2));
        rowIdx = (k + structure(2,1))*structure(1,0)*structure(0,0)
          + (j + structure(1,1))*structure(0,0) + (i + structure(0,1));

        const size_type rowOffset = m_A.graph.row_map(rowIdx);
        const value_type* value_ptr = &(m_A.values(rowOffset));
        m_y(rowIdx) = beta*m_y(rowIdx) + alpha*(*(value_ptr+0)*m_x(rowIdx - structure(0,0)*structure(1,0) - structure(0,0) - 1)
                                                + *(value_ptr+ 1)*m_x(rowIdx - structure(0,0)*structure(1,0) - structure(0,0))
                                                + *(value_ptr+ 2)*m_x(rowIdx - structure(0,0)*structure(1,0) - structure(0,0) + 1)
                                                + *(value_ptr+ 3)*m_x(rowIdx - structure(0,0)*structure(1,0) - 1)
                                                + *(value_ptr+ 4)*m_x(rowIdx - structure(0,0)*structure(1,0))
                                                + *(value_ptr+ 5)*m_x(rowIdx - structure(0,0)*structure(1,0) + 1)
                                                + *(value_ptr+ 6)*m_x(rowIdx - structure(0,0)*structure(1,0) + structure(0,0) - 1)
                                                + *(value_ptr+ 7)*m_x(rowIdx - structure(0,0)*structure(1,0) + structure(0,0))
                                                + *(value_ptr+ 8)*m_x(rowIdx - structure(0,0)*structure(1,0) + structure(0,0) + 1)
                                                + *(value_ptr+ 9)*m_x(rowIdx - structure(0,0) - 1)
                                                + *(value_ptr+10)*m_x(rowIdx - structure(0,0))
                                                + *(value_ptr+11)*m_x(rowIdx - structure(0,0) + 1)
                                                + *(value_ptr+12)*m_x(rowIdx - 1)
                                                + *(value_ptr+13)*m_x(rowIdx)
                                                + *(value_ptr+14)*m_x(rowIdx + 1)
                                                + *(value_ptr+15)*m_x(rowIdx + structure(0,0) - 1)
                                                + *(value_ptr+16)*m_x(rowIdx + structure(0,0))
                                                + *(value_ptr+17)*m_x(rowIdx + structure(0,0) + 1)
                                                + *(value_ptr+18)*m_x(rowIdx + structure(0,0)*structure(1,0) - structure(0,0) - 1)
                                                + *(value_ptr+19)*m_x(rowIdx + structure(0,0)*structure(1,0) - structure(0,0))
                                                + *(value_ptr+20)*m_x(rowIdx + structure(0,0)*structure(1,0) - structure(0,0) + 1)
                                                + *(value_ptr+21)*m_x(rowIdx + structure(0,0)*structure(1,0) - 1)
                                                + *(value_ptr+22)*m_x(rowIdx + structure(0,0)*structure(1,0))
                                                + *(value_ptr+23)*m_x(rowIdx + structure(0,0)*structure(1,0) + 1)
                                                + *(value_ptr+24)*m_x(rowIdx + structure(0,0)*structure(1,0) + structure(0,0) - 1)
                                                + *(value_ptr+25)*m_x(rowIdx + structure(0,0)*structure(1,0) + structure(0,0))
                                                + *(value_ptr+26)*m_x(rowIdx + structure(0,0)*structure(1,0) + structure(0,0) + 1));
      });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const exterior1DTag&, const ordinal_type& exteriorIdx) const
  {
    typedef typename YVector::non_const_value_type y_value_type;
    const ordinal_type topFlag = exteriorIdx / (structure(0,0) + 2*structure(1,0) - 4);
    const ordinal_type bottomFlag = static_cast<const ordinal_type>((exteriorIdx / structure(0,0)) == 0);

    ordinal_type rowIdx = exteriorIdx*(structure(0,0) - 1);

    const size_type rowOffset = m_A.graph.row_map(rowIdx);
    const ordinal_type row_length = static_cast<ordinal_type> (m_A.graph.row_map(rowIdx + 1) - rowOffset);
    const value_type* value_ptr = &(m_A.values(rowOffset));
    const ordinal_type* column_ptr = &(m_A.graph.entries(rowOffset));
    y_value_type sum = 0;
    for(ordinal_type entryIdx = 0; entryIdx < row_length; ++entryIdx) {
      sum += (*(value_ptr + entryIdx))*m_x(*(column_ptr + entryIdx));
    }
    m_y(rowIdx) = beta*m_y(rowIdx) + alpha*sum;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const exterior2DTag&, const ordinal_type& exteriorIdx) const
  {
    typedef typename YVector::non_const_value_type y_value_type;
    const ordinal_type topFlag = exteriorIdx / (structure(0,0) + 2*structure(1,0) - 4);
    const ordinal_type bottomFlag = static_cast<const ordinal_type>((exteriorIdx / structure(0,0)) == 0);

    ordinal_type rowIdx = 0;
    if(bottomFlag == 1) {
      rowIdx = exteriorIdx;
    } else if(topFlag == 1) {
      rowIdx = exteriorIdx - (structure(0,0) + 2*structure(1,0) - 4)
        + structure(0,0)*(structure(1,0) - 1);
    } else {
      ordinal_type edgeIdx = (exteriorIdx - structure(0,0)) / 2;
      ordinal_type edgeFlg = (exteriorIdx - structure(0,0)) % 2;
      rowIdx = (edgeIdx + 1)*structure(0,0) + edgeFlg*(structure(0,0) - 1);
    }

    const size_type rowOffset = m_A.graph.row_map(rowIdx);
    const ordinal_type row_length = static_cast<ordinal_type> (m_A.graph.row_map(rowIdx + 1) - rowOffset);
    const value_type* value_ptr = &(m_A.values(rowOffset));
    const ordinal_type* column_ptr = &(m_A.graph.entries(rowOffset));
    y_value_type sum = 0;
    for(ordinal_type entryIdx = 0; entryIdx < row_length; ++entryIdx) {
      sum += (*(value_ptr + entryIdx))*m_x(*(column_ptr + entryIdx));
    }
    m_y(rowIdx) = beta*m_y(rowIdx) + alpha*sum;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const exterior3DTag&, const ordinal_type& exteriorIdx) const
  {
    typedef typename YVector::non_const_value_type y_value_type;
    const ordinal_type topFlag = static_cast<const ordinal_type>(numExterior - exteriorIdx - 1 < structure(0,0)*structure(1,0));
    const ordinal_type bottomFlag = static_cast<const ordinal_type>(exteriorIdx / (structure(0,0)*structure(1,0)) == 0);

    ordinal_type rowIdx = 0;
    if(bottomFlag == 1) {
      rowIdx = exteriorIdx;
    } else if(topFlag == 1) {
      rowIdx = exteriorIdx - structure(0,0)*structure(1,0)
        - 2*(structure(2,0) - 2)*(structure(1,0) - 1 + structure(0,0) - 1)
        + (structure(2,0) - 1)*structure(0,0)*structure(1,0);
    } else {
      ordinal_type i, j, k, rem;
      k = (exteriorIdx - structure(0,0)*structure(1,0)) / (2*(structure(0,0) - 1 + structure(1,0) - 1));
      rem = (exteriorIdx - structure(0,0)*structure(1,0)) % (2*(structure(0,0) - 1 + structure(1,0) - 1));
      ordinal_type frontFlg = static_cast<ordinal_type>(rem < structure(0,0));
      ordinal_type backFlg = static_cast<ordinal_type>(rem - structure(0,0) - 2*(structure(1,0) - 1) - 1 > 0);
      if(rem < structure(0,0)) {
        rowIdx = (k + 1)*structure(0,0)*structure(1,0) + rem;
      } else if(rem < structure(0,0) + 2*(structure(1,0) - 2)) {
        ordinal_type edgeIdx = (rem - structure(0,0)) / 2;
        ordinal_type edgeFlg = (rem - structure(0,0)) % 2;
        if(edgeFlg == 0) {
          rowIdx = (k + 1)*structure(0,0)*structure(1,0) + (edgeIdx + 1)*structure(0,0);
        } else if(edgeFlg == 1) {
          rowIdx = (k + 1)*structure(0,0)*structure(1,0) + (edgeIdx + 2)*structure(0,0) - 1;
        }
      } else {
        rowIdx = (k + 1)*structure(0,0)*structure(1,0)
          + rem - structure(0,0) - 2*(structure(1,0) - 2) + (structure(1,0) - 1)*structure(0,0);
      }
    }

    const size_type rowOffset = m_A.graph.row_map(rowIdx);
    const ordinal_type row_length = static_cast<ordinal_type> (m_A.graph.row_map(rowIdx + 1) - rowOffset);
    const value_type* value_ptr = &(m_A.values(rowOffset));
    const ordinal_type* column_ptr = &(m_A.graph.entries(rowOffset));
    y_value_type sum = 0;
    for(ordinal_type entryIdx = 0; entryIdx < row_length; ++entryIdx) {
      sum += (*(value_ptr + entryIdx))*m_x(*(column_ptr + entryIdx));
    }
    m_y(rowIdx) = beta*m_y(rowIdx) + alpha*sum;
  }
};

template<class execution_space>
int64_t spmv_struct_launch_parameters(int64_t numInterior, int64_t nnz, int nnz_per_row,
                                      int64_t rows_per_thread, int& team_size, int& vector_length) {
  int64_t rows_per_team;

  if(nnz_per_row < 1) nnz_per_row = 1;

  if(vector_length < 1) {
    vector_length = 1;
    while(vector_length<32 && vector_length*6 < nnz_per_row)
      vector_length*=2;
  }
  vector_length = 1;

  // Determine rows per thread
  if(rows_per_thread < 1) {
    #ifdef KOKKOS_ENABLE_CUDA
    if(std::is_same<Kokkos::Cuda,execution_space>::value)
      rows_per_thread = 1;
    else
    #endif
    {
      if(nnz_per_row < 20 && numInterior*nnz_per_row > 5000000 ) {
        rows_per_thread = 256;
      } else
        rows_per_thread = 64;
    }
  }

  #ifdef KOKKOS_ENABLE_CUDA
  if(team_size < 1) {
    if(std::is_same<Kokkos::Cuda,execution_space>::value)
    { team_size = 256/vector_length; }
    else
    { team_size = 1; }
  }
  #endif

  rows_per_team = rows_per_thread * team_size;

  if(rows_per_team < 0) {
    int64_t nnz_per_team = 4096;
    int64_t conc = execution_space::concurrency();
    while((conc * nnz_per_team * 4 > nnz)&&(nnz_per_team > 256)) nnz_per_team/=2;
    rows_per_team = (nnz_per_team + nnz_per_row - 1)/nnz_per_row;
  }


  return rows_per_team;
}

template<class mat_structure,
         class AMatrix,
         class XVector,
         class YVector,
         int dobeta,
         bool conjugate>
static void
spmv_struct_beta_no_transpose (const int stencil_type,
                               const mat_structure& structure,
                               typename YVector::const_value_type& alpha,
                               const AMatrix& A,
                               const XVector& x,
                               typename YVector::const_value_type& beta,
                               const YVector& y)
{
  typedef typename AMatrix::ordinal_type ordinal_type;
  typedef typename AMatrix::execution_space execution_space;
  if (A.numRows () <= static_cast<ordinal_type> (0)) {
    return;
  }

  typename mat_structure::HostMirror structure_h = Kokkos::create_mirror_view(structure);
  Kokkos::deep_copy(structure_h, structure);

  int team_size = -1;
  int vector_length = -1;
  int nnzPerRow = -1;
  int64_t rows_per_thread = -1;
  int64_t numInteriorPts = 0;

  if(structure_h.extent(0) == 1) {
    numInteriorPts = structure_h(0,0) - structure_h(0,1) - structure_h(0,2);
    nnzPerRow = 3;
  } else if(structure_h.extent(0) == 2) {
    numInteriorPts = (structure_h(1,0) - structure_h(1,1) - structure_h(1,2))
      *(structure_h(0,0) - structure_h(0,1) - structure_h(0,2));
    if(stencil_type == 1) {
      nnzPerRow = 5;
    } else if(stencil_type == 2) {
      nnzPerRow = 9;
    }
  } else if(structure_h.extent(0) == 3) {
    numInteriorPts = (structure_h(2,0) - structure_h(2,1) - structure_h(2,2))
      *(structure_h(1,0) - structure_h(1,1) - structure_h(1,2))
      *(structure_h(0,0) - structure_h(0,1) - structure_h(0,2));
    if(stencil_type == 1) {
      nnzPerRow = 7;
    } else if(stencil_type == 2) {
      nnzPerRow = 27;
    }
  }

  int64_t rows_per_team = spmv_struct_launch_parameters<execution_space>(numInteriorPts,
                                                                         A.nnz(),
                                                                         nnzPerRow,
                                                                         rows_per_thread,
                                                                         team_size,
                                                                         vector_length);
  int64_t worksets = (numInteriorPts + rows_per_team - 1) / rows_per_team;

  // std::cout << "worksets=" << worksets
  //           << ", rows_per_team=" << rows_per_team
  //           << ", team_size=" << team_size
  //           << ",  vector_length=" << vector_length << std::endl;

  SPMV_Struct_Functor<mat_structure,AMatrix,XVector,YVector,dobeta,conjugate> func(structure,
                                                                                   alpha,A,x,beta,y,
                                                                                   rows_per_team);

  func.compute(structure_h, stencil_type, worksets, team_size, vector_length);
}

template<class mat_structure,
         class AMatrix,
         class XVector,
         class YVector,
         int dobeta,
         bool conjugate>
static void
spmv_struct_beta_transpose (const int stencil_type,
                            const mat_structure& structure,
                            typename YVector::const_value_type& alpha,
                            const AMatrix& A,
                            const XVector& x,
                            typename YVector::const_value_type& beta,
                            const YVector& y)
{
  typedef typename AMatrix::ordinal_type ordinal_type;

  if (A.numRows () <= static_cast<ordinal_type> (0)) {
    return;
  }

  // We need to scale y first ("scaling" by zero just means filling
  // with zeros), since the functor works by atomic-adding into y.
  if (dobeta != 1) {
    KokkosBlas::scal (y, beta, y);
  }

  typedef typename AMatrix::size_type size_type;

  // Assuming that no row contains duplicate entries, NNZPerRow
  // cannot be more than the number of columns of the matrix.  Thus,
  // the appropriate type is ordinal_type.
  const ordinal_type NNZPerRow = static_cast<ordinal_type> (A.nnz () / A.numRows ());

  int vector_length = 1;
  while( (static_cast<ordinal_type> (vector_length*2*3) <= NNZPerRow) && (vector_length<32) ) vector_length*=2;

  typedef SPMV_Struct_Transpose_Functor<AMatrix, XVector, YVector, dobeta, conjugate> OpType;

  typename AMatrix::const_ordinal_type nrow = A.numRows();

  OpType op (alpha, A, x, beta, y, RowsPerThread<typename AMatrix::execution_space> (NNZPerRow));

  const int rows_per_thread = RowsPerThread<typename AMatrix::execution_space > (NNZPerRow);
  const int team_size = Kokkos::TeamPolicy<typename AMatrix::execution_space>(rows_per_thread, Kokkos::AUTO, vector_length).team_size_recommended(op, Kokkos::ParallelForTag());
  const int rows_per_team = rows_per_thread * team_size;
  const size_type nteams = (nrow+rows_per_team-1)/rows_per_team;
  Kokkos::parallel_for("KokkosSparse::spmv_struct<Transpose>", Kokkos::TeamPolicy< typename AMatrix::execution_space >
     ( nteams , team_size , vector_length ) , op );

}

template<class AMatrix,
         class mat_structure,
         class XVector,
         class YVector,
         int dobeta>
static void
spmv_struct_beta (const char mode[],
                  const int stencil_type,
                  const mat_structure& structure,
                  typename YVector::const_value_type& alpha,
                  const AMatrix& A,
                  const XVector& x,
                  typename YVector::const_value_type& beta,
                  const YVector& y)
{
  if (mode[0] == NoTranspose[0]) {
    spmv_struct_beta_no_transpose<mat_structure,AMatrix,XVector,YVector,dobeta,false>
      (stencil_type,structure,alpha,A,x,beta,y);
  }
  else if (mode[0] == Conjugate[0]) {
    spmv_struct_beta_no_transpose<mat_structure,AMatrix,XVector,YVector,dobeta,true>
      (stencil_type,structure,alpha,A,x,beta,y);
  }
  else if (mode[0]==Transpose[0]) {
    spmv_struct_beta_transpose<mat_structure,AMatrix,XVector,YVector,dobeta,false>
      (stencil_type,structure,alpha,A,x,beta,y);
  }
  else if(mode[0]==ConjugateTranspose[0]) {
    spmv_struct_beta_transpose<mat_structure,AMatrix,XVector,YVector,dobeta,true>
      (stencil_type,structure,alpha,A,x,beta,y);
  }
  else {
    Kokkos::Impl::throw_runtime_exception("Invalid Transpose Mode for KokkosSparse::spmv_struct()");
  }
}

}
}

#endif // KOKKOSSPARSE_IMPL_SPMV_STRUCT_DEF_HPP_
