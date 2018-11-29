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

#ifndef KOKKOSKERNELS_TEST_STRUCTURE_MATRIX_HPP
#define KOKKOSKERNELS_TEST_STRUCTURE_MATRIX_HPP

#include "KokkosKernels_Utils.hpp"
namespace Test {

  enum {FD, FE};

  template <typename CrsMatrix_t, typename mat_structure>
  CrsMatrix_t generate_structured_matrix1D(const mat_structure& structure) {

    typedef typename CrsMatrix_t::StaticCrsGraphType graph_t;
    typedef typename CrsMatrix_t::row_map_type::non_const_type row_map_view_t;
    typedef typename CrsMatrix_t::index_type::non_const_type   cols_view_t;
    typedef typename CrsMatrix_t::values_type::non_const_type  scalar_view_t;
    typedef typename CrsMatrix_t::non_const_size_type size_type;
    typedef typename CrsMatrix_t::non_const_ordinal_type ordinal_type;

    typename mat_structure::HostMirror structure_h = Kokkos::create_mirror_view(structure);
    Kokkos::deep_copy(structure_h, structure);

    // Extract geometric data
    const ordinal_type nx          = structure_h(0,0);
    const ordinal_type numNodes    = nx;
    const ordinal_type leftBC      = structure_h(0,1);
    const ordinal_type rightBC     = structure_h(0,2);
    const ordinal_type numInterior = (nx - leftBC - rightBC);
    const ordinal_type numCorner   = leftBC + rightBC;
    const ordinal_type interiorStencilLength = 3, cornerStencilLength = 2;
    const size_type numEntries = numInterior*interiorStencilLength + numCorner*cornerStencilLength;

    // Create matrix data
    row_map_view_t rowmap_view ("rowmap_view",  numNodes + 1);
    cols_view_t    columns_view("colsmap_view", numEntries);
    scalar_view_t  values_view ("values_view",  numEntries);

    typename row_map_view_t::HostMirror rowmap_view_h  = Kokkos::create_mirror_view(rowmap_view);
    typename cols_view_t::HostMirror    columns_view_h = Kokkos::create_mirror_view(columns_view);
    typename scalar_view_t::HostMirror  values_view_h  = Kokkos::create_mirror_view(values_view);

    Kokkos::deep_copy(rowmap_view_h,  rowmap_view);
    Kokkos::deep_copy(columns_view_h, columns_view);
    Kokkos::deep_copy(values_view_h,  values_view);

    size_type    rowOffset;
    ordinal_type rowIdx;

    // Loop over the interior points
    for(ordinal_type idx = 0; idx < numInterior; ++idx) {
      rowIdx = idx + leftBC;
      rowOffset = (rowIdx + 1 - leftBC)*interiorStencilLength + leftBC*cornerStencilLength;

      rowmap_view_h(rowIdx + 1) = rowOffset;

      // Fill column indices
      columns_view_h(rowOffset - 3) = rowIdx - 1;
      columns_view_h(rowOffset - 2) = rowIdx;
      columns_view_h(rowOffset - 1) = rowIdx + 1;

      // Fill values
      values_view_h(rowOffset - 3) = -1.0;
      values_view_h(rowOffset - 2) =  2.0;
      values_view_h(rowOffset - 1) = -1.0;
    }

    // LeftBC
    if(leftBC == 1) {
      rowmap_view_h(1) = 2;

      columns_view_h(0) = 0;
      columns_view_h(1) = 1;

      values_view_h(0) =  1.0;
      values_view_h(1) = -1.0;
    }

    // RightBC
    if(leftBC == 1) {
      rowmap_view_h(numNodes) = numEntries;

      columns_view_h(numEntries - 2) = numNodes - 2;
      columns_view_h(numEntries - 1) = numNodes - 1;

      values_view_h(numEntries - 2) = -1.0;
      values_view_h(numEntries - 1) =  1.0;
    }

    Kokkos::deep_copy(rowmap_view,  rowmap_view_h);
    Kokkos::deep_copy(columns_view, columns_view_h);
    Kokkos::deep_copy(values_view,  values_view_h);

    graph_t static_graph (columns_view, rowmap_view);
    std::string name = "CrsMatrixFE";

    return CrsMatrix_t(name, numNodes, values_view, static_graph);

  } // generate_structured_matrix1D

  template <typename CrsMatrix_t, typename mat_structure>
  CrsMatrix_t generate_structured_matrix2D(const std::string stencil,
                                    const mat_structure& structure) {

    typedef typename CrsMatrix_t::StaticCrsGraphType graph_t;
    typedef typename CrsMatrix_t::row_map_type::non_const_type row_map_view_t;
    typedef typename CrsMatrix_t::index_type::non_const_type   cols_view_t;
    typedef typename CrsMatrix_t::values_type::non_const_type  scalar_view_t;
    typedef typename CrsMatrix_t::non_const_size_type size_type;
    typedef typename CrsMatrix_t::non_const_ordinal_type ordinal_type;

    int stencil_type = 0;
    if (stencil == "FD") {
      stencil_type = FD;
    } else if (stencil == "FE") {
      stencil_type = FE;
    } else {
      std::ostringstream os;
      os << "Test::generate_structured_matrix2D only accepts stencil: FD and FEM, you passed: "
         << stencil <<" !" << std::endl;
      Kokkos::Impl::throw_runtime_exception (os.str ());
    }

    typename mat_structure::HostMirror structure_h = Kokkos::create_mirror_view(structure);
    Kokkos::deep_copy(structure_h, structure);

    // Extract geometric data
    const ordinal_type nx          = structure_h(0,0);
    const ordinal_type ny          = structure_h(1,0);
    const ordinal_type numNodes    = ny*nx;
    const ordinal_type leftBC      = structure_h(0,1);
    const ordinal_type rightBC     = structure_h(0,2);
    const ordinal_type bottomBC    = structure_h(1,1);
    const ordinal_type topBC       = structure_h(1,2);
    const ordinal_type numInterior = (nx - leftBC - rightBC)*(ny - bottomBC - topBC);
    const ordinal_type numEdge     = (bottomBC + topBC)*(nx - leftBC - rightBC)
      + (leftBC + rightBC)*(ny - bottomBC - topBC);
    const ordinal_type numCorner   = (bottomBC + topBC)*(leftBC + rightBC);
    ordinal_type interiorStencilLength = 0, edgeStencilLength = 0, cornerStencilLength = 0;

    if(stencil_type == FD) {
      interiorStencilLength = 5;
      edgeStencilLength     = 4;
      cornerStencilLength   = 3;
    } else if(stencil_type == FE) {
      interiorStencilLength = 9;
      edgeStencilLength     = 6;
      cornerStencilLength   = 4;
    }

    const size_type numEntries = numInterior*interiorStencilLength
      + numEdge*edgeStencilLength
      + numCorner*cornerStencilLength;

    // Create matrix data
    row_map_view_t rowmap_view ("rowmap_view",  numNodes + 1);
    cols_view_t    columns_view("colsmap_view", numEntries);
    scalar_view_t  values_view ("values_view",  numEntries);

    typename row_map_view_t::HostMirror rowmap_view_h  = Kokkos::create_mirror_view(rowmap_view);
    typename cols_view_t::HostMirror    columns_view_h = Kokkos::create_mirror_view(columns_view);
    typename scalar_view_t::HostMirror  values_view_h  = Kokkos::create_mirror_view(values_view);

    Kokkos::deep_copy(rowmap_view_h,  rowmap_view);
    Kokkos::deep_copy(columns_view_h, columns_view);
    Kokkos::deep_copy(values_view_h,  values_view);

    // Fill the CrsGraph and the CrsMatrix
    // To start simple we construct 2D 5pt stencil Laplacian.
    // We assume Neumann boundary conditions on the edge of the domain.
    const ordinal_type numEntriesPerGridRow = (nx - leftBC - rightBC)*interiorStencilLength
      + (leftBC + rightBC)*edgeStencilLength;
    const ordinal_type numEntriesBottomRow = (nx - leftBC - rightBC)*edgeStencilLength
      + (leftBC + rightBC)*cornerStencilLength;
    size_type    rowOffset;
    ordinal_type rowIdx;

    // Loop over the interior points
    for(ordinal_type idx = 0; idx < numInterior; ++idx) {
      ordinal_type i, j;

      // Compute row index
      j = idx / (nx - leftBC - rightBC);
      i = idx % (nx - leftBC - rightBC);
      rowIdx = (j + bottomBC)*nx + i + leftBC;

      // Compute rowOffset
      rowOffset = j*numEntriesPerGridRow + bottomBC*numEntriesBottomRow
        + (i + 1)*interiorStencilLength + leftBC*edgeStencilLength;
      rowmap_view_h(rowIdx + 1) = rowOffset;

      if(stencil_type == FD) {
        // Fill column indices
        columns_view_h(rowOffset - 5) = rowIdx - nx;
        columns_view_h(rowOffset - 4) = rowIdx - 1;
        columns_view_h(rowOffset - 3) = rowIdx;
        columns_view_h(rowOffset - 2) = rowIdx + 1;
        columns_view_h(rowOffset - 1) = rowIdx + nx;

        // Fill values
        values_view_h(rowOffset - 5) = -1.0;
        values_view_h(rowOffset - 4) = -1.0;
        values_view_h(rowOffset - 3) =  4.0;
        values_view_h(rowOffset - 2) = -1.0;
        values_view_h(rowOffset - 1) = -1.0;
      } else if(stencil_type == FE) {
        // Fill column indices
        columns_view_h(rowOffset - 9) = rowIdx - nx - 1;
        columns_view_h(rowOffset - 8) = rowIdx - nx;
        columns_view_h(rowOffset - 7) = rowIdx - nx + 1;
        columns_view_h(rowOffset - 6) = rowIdx - 1;
        columns_view_h(rowOffset - 5) = rowIdx;
        columns_view_h(rowOffset - 4) = rowIdx + 1;
        columns_view_h(rowOffset - 3) = rowIdx + nx - 1;
        columns_view_h(rowOffset - 2) = rowIdx + nx;
        columns_view_h(rowOffset - 1) = rowIdx + nx + 1;

        // Fill values
        values_view_h(rowOffset - 9) = -2.0;
        values_view_h(rowOffset - 8) = -2.0;
        values_view_h(rowOffset - 7) = -2.0;
        values_view_h(rowOffset - 6) = -2.0;
        values_view_h(rowOffset - 5) = 16.0;
        values_view_h(rowOffset - 4) = -2.0;
        values_view_h(rowOffset - 3) = -2.0;
        values_view_h(rowOffset - 2) = -2.0;
        values_view_h(rowOffset - 1) = -2.0;
      }
    }

    // Loop over horizontal edge points
    for(ordinal_type idx = 0; idx < nx - leftBC - rightBC; ++idx) {

      if(bottomBC == 1) {
        /***************/
        /* Bottom edge */
        /***************/
        rowIdx    = idx + leftBC;
        rowOffset = (idx + 1)*edgeStencilLength + leftBC*cornerStencilLength;
        rowmap_view_h(rowIdx + 1) = rowOffset;
        if(stencil_type == FD) {
          // Fill column indices
          columns_view_h(rowOffset - 4) = rowIdx - 1;
          columns_view_h(rowOffset - 3) = rowIdx;
          columns_view_h(rowOffset - 2) = rowIdx + 1;
          columns_view_h(rowOffset - 1) = rowIdx + nx;

          // Fill values
          values_view_h(rowOffset - 4) = -1.0;
          values_view_h(rowOffset - 3) =  3.0;
          values_view_h(rowOffset - 2) = -1.0;
          values_view_h(rowOffset - 1) = -1.0;
      } else if(stencil_type == FE) {
          // Fill column indices
          columns_view_h(rowOffset - 6) = rowIdx - 1;
          columns_view_h(rowOffset - 5) = rowIdx;
          columns_view_h(rowOffset - 4) = rowIdx + 1;
          columns_view_h(rowOffset - 3) = rowIdx + nx - 1;
          columns_view_h(rowOffset - 2) = rowIdx + nx;
          columns_view_h(rowOffset - 1) = rowIdx + nx + 1;

          // Fill values
          values_view_h(rowOffset - 6) = -1.0;
          values_view_h(rowOffset - 5) =  8.0;
          values_view_h(rowOffset - 4) = -1.0;
          values_view_h(rowOffset - 3) = -2.0;
          values_view_h(rowOffset - 2) = -2.0;
          values_view_h(rowOffset - 1) = -2.0;
        }
      }

      if(topBC == 1) {
        /************/
        /* Top edge */
        /************/
        rowIdx    = (ny - 1)*nx + idx + leftBC;
        rowOffset = (ny - 1 - bottomBC)*numEntriesPerGridRow + bottomBC*numEntriesBottomRow
          + (idx + 1)*edgeStencilLength + leftBC*cornerStencilLength;
        rowmap_view_h(rowIdx + 1) = rowOffset;
        if(stencil_type == FD) {
          // Fill column indices
          columns_view_h(rowOffset - 4) = rowIdx - nx;
          columns_view_h(rowOffset - 3) = rowIdx - 1;
          columns_view_h(rowOffset - 2) = rowIdx;
          columns_view_h(rowOffset - 1) = rowIdx + 1;

          // Fill values
          values_view_h(rowOffset - 4) = -1.0;
          values_view_h(rowOffset - 3) = -1.0;
          values_view_h(rowOffset - 2) =  3.0;
          values_view_h(rowOffset - 1) = -1.0;
        } else if(stencil_type == FE) {
          // Fill column indices
          columns_view_h(rowOffset - 6) = rowIdx - nx - 1;
          columns_view_h(rowOffset - 5) = rowIdx - nx;
          columns_view_h(rowOffset - 4) = rowIdx - nx + 1;
          columns_view_h(rowOffset - 3) = rowIdx - 1;
          columns_view_h(rowOffset - 2) = rowIdx;
          columns_view_h(rowOffset - 1) = rowIdx + 1;

          // Fill values
          values_view_h(rowOffset - 6) = -2.0;
          values_view_h(rowOffset - 5) = -2.0;
          values_view_h(rowOffset - 4) = -2.0;
          values_view_h(rowOffset - 3) = -1.0;
          values_view_h(rowOffset - 2) =  8.0;
          values_view_h(rowOffset - 1) = -1.0;
        }
      }
    }

    // Loop over vertical edge points
    for(ordinal_type idx = 0; idx < ny - bottomBC - topBC; ++idx) {

      if(leftBC == 1) {
        /*************/
        /* Left edge */
        /*************/
        rowIdx    = (idx + bottomBC)*nx;
        rowOffset = idx*numEntriesPerGridRow + bottomBC*numEntriesBottomRow
          + edgeStencilLength;
        rowmap_view_h(rowIdx + 1) = rowOffset;
        if(stencil_type == FD) {
          // Fill column indices
          columns_view_h(rowOffset - 4) = rowIdx - nx;
          columns_view_h(rowOffset - 3) = rowIdx;
          columns_view_h(rowOffset - 2) = rowIdx + 1;
          columns_view_h(rowOffset - 1) = rowIdx + nx;

          // Fill values
          values_view_h(rowOffset - 4) = -1.0;
          values_view_h(rowOffset - 3) =  3.0;
          values_view_h(rowOffset - 2) = -1.0;
          values_view_h(rowOffset - 1) = -1.0;
        } else if(stencil_type == FE) {
          // Fill column indices
          columns_view_h(rowOffset - 6) = rowIdx - nx;
          columns_view_h(rowOffset - 5) = rowIdx - nx + 1;
          columns_view_h(rowOffset - 4) = rowIdx;
          columns_view_h(rowOffset - 3) = rowIdx + 1;
          columns_view_h(rowOffset - 2) = rowIdx + nx;
          columns_view_h(rowOffset - 1) = rowIdx + nx + 1;

          // Fill values
          values_view_h(rowOffset - 6) = -1.0;
          values_view_h(rowOffset - 5) = -2.0;
          values_view_h(rowOffset - 4) =  8.0;
          values_view_h(rowOffset - 3) = -2.0;
          values_view_h(rowOffset - 2) = -1.0;
          values_view_h(rowOffset - 1) = -2.0;
        }
      }

      if(rightBC == 1) {
        /**************/
        /* Right edge */
        /**************/
        rowIdx    = (idx + bottomBC + 1)*nx - 1;
        rowOffset = (idx + 1)*numEntriesPerGridRow + bottomBC*numEntriesBottomRow;
        rowmap_view_h(rowIdx + 1) = rowOffset;
        if(stencil_type == FD) {
          // Fill column indices
          columns_view_h(rowOffset - 4) = rowIdx - nx;
          columns_view_h(rowOffset - 3) = rowIdx - 1;
          columns_view_h(rowOffset - 2) = rowIdx;
          columns_view_h(rowOffset - 1) = rowIdx + nx;

          // Fill values
          values_view_h(rowOffset - 4) = -1.0;
          values_view_h(rowOffset - 3) = -1.0;
          values_view_h(rowOffset - 2) =  3.0;
          values_view_h(rowOffset - 1) = -1.0;
        } else if(stencil_type == FE) {
          // Fill column indices
          columns_view_h(rowOffset - 6) = rowIdx - nx - 1;
          columns_view_h(rowOffset - 5) = rowIdx - nx;
          columns_view_h(rowOffset - 4) = rowIdx - 1;
          columns_view_h(rowOffset - 3) = rowIdx;
          columns_view_h(rowOffset - 2) = rowIdx + nx - 1;
          columns_view_h(rowOffset - 1) = rowIdx + nx;

          // Fill values
          values_view_h(rowOffset - 6) = -2.0;
          values_view_h(rowOffset - 5) = -1.0;
          values_view_h(rowOffset - 4) = -2.0;
          values_view_h(rowOffset - 3) =  8.0;
          values_view_h(rowOffset - 2) = -2.0;
          values_view_h(rowOffset - 1) = -1.0;
        }
      }
    }

    // Bottom-left corner
    if(bottomBC*leftBC == 1) {
      rowIdx = 0;
      rowOffset = cornerStencilLength;
      rowmap_view_h(rowIdx + 1)     = rowOffset;

      if(stencil_type == FD) {
        columns_view_h(rowOffset - 3) = rowIdx;
        columns_view_h(rowOffset - 2) = rowIdx + 1;
        columns_view_h(rowOffset - 1) = rowIdx + nx;

        values_view_h(rowOffset - 3)  =  2.0;
        values_view_h(rowOffset - 2)  = -1.0;
        values_view_h(rowOffset - 1)  = -1.0;
      } else if(stencil_type == FE) {
        columns_view_h(rowOffset - 4) = rowIdx;
        columns_view_h(rowOffset - 3) = rowIdx + 1;
        columns_view_h(rowOffset - 2) = rowIdx + nx;
        columns_view_h(rowOffset - 1) = rowIdx + nx + 1;

        values_view_h(rowOffset - 4)  =  4.0;
        values_view_h(rowOffset - 3)  = -1.0;
        values_view_h(rowOffset - 2)  = -1.0;
        values_view_h(rowOffset - 1)  = -2.0;
      }
    }

    // Bottom-right corner
    if(bottomBC*rightBC == 1) {
      rowIdx = nx - 1;
      rowOffset = (1 - bottomBC)*numEntriesPerGridRow + bottomBC*numEntriesBottomRow;
      rowmap_view_h(rowIdx + 1)     = rowOffset;

      if(stencil_type == FD) {
        columns_view_h(rowOffset - 3) = rowIdx - 1;
        columns_view_h(rowOffset - 2) = rowIdx;
        columns_view_h(rowOffset - 1) = rowIdx + nx;

        values_view_h(rowOffset - 3)  = -1.0;
        values_view_h(rowOffset - 2)  =  2.0;
        values_view_h(rowOffset - 1)  = -1.0;
      } else if(stencil_type == FE) {
        columns_view_h(rowOffset - 4) = rowIdx - 1;
        columns_view_h(rowOffset - 3) = rowIdx;
        columns_view_h(rowOffset - 2) = rowIdx + nx - 1;
        columns_view_h(rowOffset - 1) = rowIdx + nx;

        values_view_h(rowOffset - 4)  = -1.0;
        values_view_h(rowOffset - 3)  =  4.0;
        values_view_h(rowOffset - 2)  = -2.0;
        values_view_h(rowOffset - 1)  = -1.0;
      }
    }

    // Top-left corner
    if(topBC*leftBC == 1) {
      rowIdx = (ny - 1)*nx;
      rowOffset = (ny - 1 - bottomBC)*numEntriesPerGridRow + bottomBC*numEntriesBottomRow
        + cornerStencilLength;
      rowmap_view_h(rowIdx + 1)     = rowOffset;

      if(stencil_type == FD) {
        columns_view_h(rowOffset - 3) = rowIdx - nx;
        columns_view_h(rowOffset - 2) = rowIdx;
        columns_view_h(rowOffset - 1) = rowIdx + 1;

        values_view_h(rowOffset - 3)  = -1.0;
        values_view_h(rowOffset - 2)  =  2.0;
        values_view_h(rowOffset - 1)  = -1.0;
      } else if(stencil_type == FE) {
        columns_view_h(rowOffset - 4) = rowIdx - nx;
        columns_view_h(rowOffset - 3) = rowIdx - nx + 1;
        columns_view_h(rowOffset - 2) = rowIdx;
        columns_view_h(rowOffset - 1) = rowIdx + 1;

        values_view_h(rowOffset - 4)  = -1.0;
        values_view_h(rowOffset - 3)  = -2.0;
        values_view_h(rowOffset - 2)  =  4.0;
        values_view_h(rowOffset - 1)  = -1.0;
      }
    }

    // Top-right corner
    if(topBC*rightBC == 1) {
      rowIdx = ny*nx - 1;
      rowOffset = numEntries;
      rowmap_view_h(rowIdx + 1)     = rowOffset;

      if(stencil_type == FD) {
        columns_view_h(rowOffset - 3) = rowIdx - nx;
        columns_view_h(rowOffset - 2) = rowIdx - 1;
        columns_view_h(rowOffset - 1) = rowIdx;

        values_view_h(rowOffset - 3)  = -1.0;
        values_view_h(rowOffset - 2)  = -1.0;
        values_view_h(rowOffset - 1)  =  2.0;
      } else if(stencil_type == FE) {
        columns_view_h(rowOffset - 4) = rowIdx - nx - 1;
        columns_view_h(rowOffset - 3) = rowIdx - nx;
        columns_view_h(rowOffset - 2) = rowIdx - 1;
        columns_view_h(rowOffset - 1) = rowIdx;

        values_view_h(rowOffset - 4)  = -2.0;
        values_view_h(rowOffset - 3)  = -1.0;
        values_view_h(rowOffset - 2)  = -1.0;
        values_view_h(rowOffset - 1)  =  4.0;
      }
    }

    Kokkos::deep_copy(rowmap_view,  rowmap_view_h);
    Kokkos::deep_copy(columns_view, columns_view_h);
    Kokkos::deep_copy(values_view,  values_view_h);

    graph_t static_graph (columns_view, rowmap_view);
    std::string name;
    if(stencil_type == FD) {
      name = "CrsMatrixFD";
    } else if(stencil_type == FE) {
      name = "CrsMatrixFE";
    }

    return CrsMatrix_t(name, numNodes, values_view, static_graph);

  } // generate_structured_matrix2D

  template <typename CrsMatrix_t, typename mat_structure>
  CrsMatrix_t generate_structured_matrix3D(const std::string stencil,
                                    const mat_structure& structure) {

    typedef typename CrsMatrix_t::StaticCrsGraphType graph_t;
    typedef typename CrsMatrix_t::row_map_type::non_const_type row_map_view_t;
    typedef typename CrsMatrix_t::index_type::non_const_type   cols_view_t;
    typedef typename CrsMatrix_t::values_type::non_const_type  scalar_view_t;
    typedef typename CrsMatrix_t::non_const_size_type size_type;
    typedef typename CrsMatrix_t::non_const_ordinal_type ordinal_type;

    int stencil_type = 0;
    if (stencil == "FD") {
      stencil_type = FD;
    } else if (stencil == "FE") {
      stencil_type = FE;
    } else {
      std::ostringstream os;
      os << "Test::generate_structured_matrix3D only accepts stencil: FD and FEM, you passed: "
         << stencil <<" !" << std::endl;
      Kokkos::Impl::throw_runtime_exception (os.str ());
    }

    typename mat_structure::HostMirror structure_h = Kokkos::create_mirror_view(structure);
    Kokkos::deep_copy(structure_h, structure);

    // Extract geometric data
    const ordinal_type nx          = structure_h(0,0);
    const ordinal_type ny          = structure_h(1,0);
    const ordinal_type nz          = structure_h(2,0);
    const ordinal_type numNodes    = ny*nx*nz;
    const ordinal_type leftBC      = structure_h(0,1);
    const ordinal_type rightBC     = structure_h(0,2);
    const ordinal_type frontBC     = structure_h(1,1);
    const ordinal_type backBC      = structure_h(1,2);
    const ordinal_type bottomBC    = structure_h(2,1);
    const ordinal_type topBC       = structure_h(2,2);
    const ordinal_type numInterior = (nx - leftBC - rightBC)*(ny - frontBC - backBC)
      *(nz - bottomBC - topBC);
    const ordinal_type numFace     =
      (leftBC + rightBC)*(ny - frontBC - backBC)*(nz - bottomBC - topBC)
      + (frontBC + backBC)*(nx - leftBC - rightBC)*(nz - bottomBC - topBC)
      + (bottomBC + topBC)*(nx - leftBC - rightBC)*(ny - frontBC - backBC);
    const ordinal_type numEdge     =
      (frontBC*bottomBC + frontBC*topBC + backBC*bottomBC + backBC*topBC)*(nx - leftBC - rightBC)
       + (leftBC*bottomBC + leftBC*topBC + rightBC*bottomBC + rightBC*topBC)*(ny - frontBC - backBC)
       + (leftBC*frontBC + leftBC*backBC + rightBC*frontBC + rightBC*backBC)*(nz - bottomBC - topBC);
    const ordinal_type numCorner   = leftBC*frontBC*bottomBC + rightBC*frontBC*bottomBC
      + leftBC*backBC*bottomBC + rightBC*backBC*bottomBC
      + leftBC*frontBC*topBC + rightBC*frontBC*topBC
      + leftBC*backBC*topBC + rightBC*backBC*topBC;
    ordinal_type interiorStencilLength = 0, faceStencilLength = 0, edgeStencilLength = 0, cornerStencilLength = 0;

    if(stencil_type == FD) {
      interiorStencilLength = 7;
      faceStencilLength     = 6;
      edgeStencilLength     = 5;
      cornerStencilLength   = 4;
    } else if(stencil_type == FE) {
      interiorStencilLength = 27;
      faceStencilLength     = 18;
      edgeStencilLength     = 12;
      cornerStencilLength   = 8;
    }

    const size_type numEntries = numInterior*interiorStencilLength
      + numFace*faceStencilLength
      + numEdge*edgeStencilLength
      + numCorner*cornerStencilLength;

    const ordinal_type numXFace = (ny - frontBC - backBC)*(nz - bottomBC - topBC);
    const ordinal_type numYFace = (nx - leftBC - rightBC)*(nz - bottomBC - topBC);
    const ordinal_type numZFace = (nx - leftBC - rightBC)*(ny - frontBC - backBC);

    // Create matrix data
    row_map_view_t rowmap_view ("rowmap_view",  numNodes + 1);
    cols_view_t    columns_view("colsmap_view", numEntries);
    scalar_view_t  values_view ("values_view",  numEntries);

    typename row_map_view_t::HostMirror rowmap_view_h  = Kokkos::create_mirror_view(rowmap_view);
    typename cols_view_t::HostMirror    columns_view_h = Kokkos::create_mirror_view(columns_view);
    typename scalar_view_t::HostMirror  values_view_h  = Kokkos::create_mirror_view(values_view);

    Kokkos::deep_copy(rowmap_view_h,  rowmap_view);
    Kokkos::deep_copy(columns_view_h, columns_view);
    Kokkos::deep_copy(values_view_h,  values_view);

    // Fill the CrsGraph and the CrsMatrix
    // To start simple we construct 2D 5pt stencil Laplacian.
    // We assume Neumann boundary conditions on the edge of the domain.
    const ordinal_type numEntriesPerGridPlane =
      (nx - leftBC - rightBC)*(ny - frontBC - backBC)*interiorStencilLength
      + (backBC + frontBC)*(nx - leftBC - rightBC)*faceStencilLength
      + (leftBC + rightBC)*(ny - frontBC - backBC)*faceStencilLength
      + (leftBC*frontBC + leftBC*backBC + rightBC*frontBC + rightBC*backBC)*edgeStencilLength;
    const ordinal_type numEntriesBottomPlane  =
      (nx - leftBC - rightBC)*(ny - frontBC - backBC)*faceStencilLength
      + (backBC + frontBC)*(nx - leftBC - rightBC)*edgeStencilLength
      + (leftBC + rightBC)*(ny - frontBC - backBC)*edgeStencilLength
      + (leftBC*frontBC + leftBC*backBC + rightBC*frontBC + rightBC*backBC)*cornerStencilLength;
    const ordinal_type numEntriesPerGridRow = (nx - leftBC - rightBC)*interiorStencilLength
      + (leftBC + rightBC)*faceStencilLength;
    const ordinal_type numEntriesFrontRow = (nx - leftBC - rightBC)*faceStencilLength
      + (leftBC + rightBC)*edgeStencilLength;
    const ordinal_type numEntriesBottomFrontRow = (nx - leftBC - rightBC)*edgeStencilLength
      + (leftBC + rightBC)*cornerStencilLength;

    ordinal_type rowIdx;
    size_type    rowOffset;
    ordinal_type i, j, k, rem;
    // Loop over the interior points
    for(ordinal_type idx = 0; idx < numInterior; ++idx) {
      // Compute row index
      k   = idx / ((ny - frontBC - backBC)*(nx - leftBC - rightBC));
      rem = idx % ((ny - frontBC - backBC)*(nx - leftBC - rightBC));
      j   = rem / (nx - leftBC - rightBC);
      i   = rem % (nx - leftBC - rightBC);
      rowIdx = (k + bottomBC)*ny*nx + (j + frontBC)*nx + i + leftBC;

      // Compute rowOffset
      rowOffset = k*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
        + j*numEntriesPerGridRow + frontBC*numEntriesFrontRow
        + (i + 1)*interiorStencilLength + leftBC*faceStencilLength;
      rowmap_view_h(rowIdx + 1) = rowOffset;

      if(stencil_type == FD) {
        // Fill column indices
        columns_view_h(rowOffset - 7) = rowIdx - ny*nx;
        columns_view_h(rowOffset - 6) = rowIdx - nx;
        columns_view_h(rowOffset - 5) = rowIdx - 1;
        columns_view_h(rowOffset - 4) = rowIdx;
        columns_view_h(rowOffset - 3) = rowIdx + 1;
        columns_view_h(rowOffset - 2) = rowIdx + nx;
        columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

        // Fill values
        values_view_h(rowOffset - 7) = -1.0;
        values_view_h(rowOffset - 6) = -1.0;
        values_view_h(rowOffset - 5) = -1.0;
        values_view_h(rowOffset - 4) =  6.0;
        values_view_h(rowOffset - 3) = -1.0;
        values_view_h(rowOffset - 2) = -1.0;
        values_view_h(rowOffset - 1) = -1.0;
      } else if(stencil_type == FE) {
        // Fill column indices
        columns_view_h(rowOffset - 27) = rowIdx - ny*nx - nx - 1;
        columns_view_h(rowOffset - 26) = rowIdx - ny*nx - nx;
        columns_view_h(rowOffset - 25) = rowIdx - ny*nx - nx + 1;
        columns_view_h(rowOffset - 24) = rowIdx - ny*nx - 1;
        columns_view_h(rowOffset - 23) = rowIdx - ny*nx;
        columns_view_h(rowOffset - 22) = rowIdx - ny*nx + 1;
        columns_view_h(rowOffset - 21) = rowIdx - ny*nx + nx - 1;
        columns_view_h(rowOffset - 20) = rowIdx - ny*nx + nx;
        columns_view_h(rowOffset - 19) = rowIdx - ny*nx + nx + 1;
        columns_view_h(rowOffset - 18) = rowIdx - nx - 1;
        columns_view_h(rowOffset - 17) = rowIdx - nx;
        columns_view_h(rowOffset - 16) = rowIdx - nx + 1;
        columns_view_h(rowOffset - 15) = rowIdx - 1;
        columns_view_h(rowOffset - 14) = rowIdx;
        columns_view_h(rowOffset - 13) = rowIdx + 1;
        columns_view_h(rowOffset - 12) = rowIdx + nx - 1;
        columns_view_h(rowOffset - 11) = rowIdx + nx;
        columns_view_h(rowOffset - 10) = rowIdx + nx + 1;
        columns_view_h(rowOffset -  9) = rowIdx + nx*ny - nx - 1;
        columns_view_h(rowOffset -  8) = rowIdx + nx*ny - nx;
        columns_view_h(rowOffset -  7) = rowIdx + nx*ny - nx + 1;
        columns_view_h(rowOffset -  6) = rowIdx + nx*ny - 1;
        columns_view_h(rowOffset -  5) = rowIdx + nx*ny;
        columns_view_h(rowOffset -  4) = rowIdx + nx*ny + 1;
        columns_view_h(rowOffset -  3) = rowIdx + nx*ny + nx - 1;
        columns_view_h(rowOffset -  2) = rowIdx + nx*ny + nx;
        columns_view_h(rowOffset -  1) = rowIdx + nx*ny + nx + 1;

        // Fill values
        values_view_h(rowOffset - 27) = -1.0;
        values_view_h(rowOffset - 26) = -2.0;
        values_view_h(rowOffset - 25) = -1.0;
        values_view_h(rowOffset - 24) = -2.0;
        values_view_h(rowOffset - 23) =  0.0;
        values_view_h(rowOffset - 22) = -2.0;
        values_view_h(rowOffset - 21) = -1.0;
        values_view_h(rowOffset - 20) = -2.0;
        values_view_h(rowOffset - 19) = -1.0;
        values_view_h(rowOffset - 18) = -2.0;
        values_view_h(rowOffset - 17) =  0.0;
        values_view_h(rowOffset - 16) = -2.0;
        values_view_h(rowOffset - 15) =  0.0;
        values_view_h(rowOffset - 14) = 32.0;
        values_view_h(rowOffset - 13) =  0.0;
        values_view_h(rowOffset - 12) =  2.0;
        values_view_h(rowOffset - 11) =  0.0;
        values_view_h(rowOffset - 10) = -2.0;
        values_view_h(rowOffset -  9) = -1.0;
        values_view_h(rowOffset -  8) = -2.0;
        values_view_h(rowOffset -  7) = -1.0;
        values_view_h(rowOffset -  6) = -2.0;
        values_view_h(rowOffset -  5) =  0.0;
        values_view_h(rowOffset -  4) = -2.0;
        values_view_h(rowOffset -  3) = -1.0;
        values_view_h(rowOffset -  2) = -2.0;
        values_view_h(rowOffset -  1) = -1.0;
      }
    }

    // Loop over the x-face points
    for(ordinal_type idx = 0; idx < numXFace; ++idx) {
      /*******************/
      /*   x == 0 face   */
      /*******************/
      if(leftBC == 1) {
        // Compute row index
        k = idx / (ny - frontBC - backBC);
        j = idx % (ny - frontBC - backBC);
        i = 0;
        rowIdx = (k + bottomBC)*ny*nx + (j + frontBC)*nx + i;

        // Compute rowOffset
        rowOffset = k*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
          + j*numEntriesPerGridRow + frontBC*numEntriesFrontRow + faceStencilLength;
        rowmap_view_h(rowIdx + 1) = rowOffset;

        if(stencil_type == FD) {
          // Fill column indices
          columns_view_h(rowOffset - 6) = rowIdx - ny*nx;
          columns_view_h(rowOffset - 5) = rowIdx - nx;
          columns_view_h(rowOffset - 4) = rowIdx;
          columns_view_h(rowOffset - 3) = rowIdx + 1;
          columns_view_h(rowOffset - 2) = rowIdx + nx;
          columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

          // Fill values
          values_view_h(rowOffset - 6) = -1.0;
          values_view_h(rowOffset - 5) = -1.0;
          values_view_h(rowOffset - 4) =  5.0;
          values_view_h(rowOffset - 3) = -1.0;
          values_view_h(rowOffset - 2) = -1.0;
          values_view_h(rowOffset - 1) = -1.0;
        } else if(stencil_type == FE) {
          // Fill column indices
          columns_view_h(rowOffset - 18) = rowIdx - ny*nx - nx;
          columns_view_h(rowOffset - 17) = rowIdx - ny*nx - nx + 1;
          columns_view_h(rowOffset - 16) = rowIdx - ny*nx;
          columns_view_h(rowOffset - 15) = rowIdx - ny*nx + 1;
          columns_view_h(rowOffset - 14) = rowIdx - ny*nx + nx;
          columns_view_h(rowOffset - 13) = rowIdx - ny*nx + nx + 1;
          columns_view_h(rowOffset - 12) = rowIdx - nx;
          columns_view_h(rowOffset - 11) = rowIdx - nx + 1;
          columns_view_h(rowOffset - 10) = rowIdx;
          columns_view_h(rowOffset -  9) = rowIdx + 1;
          columns_view_h(rowOffset -  8) = rowIdx + nx;
          columns_view_h(rowOffset -  7) = rowIdx + nx + 1;
          columns_view_h(rowOffset -  6) = rowIdx + nx*ny - nx;
          columns_view_h(rowOffset -  5) = rowIdx + nx*ny - nx + 1;
          columns_view_h(rowOffset -  4) = rowIdx + nx*ny;
          columns_view_h(rowOffset -  3) = rowIdx + nx*ny + 1;
          columns_view_h(rowOffset -  2) = rowIdx + nx*ny + nx;
          columns_view_h(rowOffset -  1) = rowIdx + nx*ny + nx + 1;

          // Fill values
          values_view_h(rowOffset - 18) = -1.0;
          values_view_h(rowOffset - 17) = -1.0;
          values_view_h(rowOffset - 16) =  0.0;
          values_view_h(rowOffset - 15) = -2.0;
          values_view_h(rowOffset - 14) = -1.0;
          values_view_h(rowOffset - 13) = -1.0;
          values_view_h(rowOffset - 12) =  0.0;
          values_view_h(rowOffset - 11) = -2.0;
          values_view_h(rowOffset - 10) = 16.0;
          values_view_h(rowOffset -  9) =  0.0;
          values_view_h(rowOffset -  8) =  0.0;
          values_view_h(rowOffset -  7) = -2.0;
          values_view_h(rowOffset -  6) = -1.0;
          values_view_h(rowOffset -  5) = -1.0;
          values_view_h(rowOffset -  4) =  0.0;
          values_view_h(rowOffset -  3) = -2.0;
          values_view_h(rowOffset -  2) = -1.0;
          values_view_h(rowOffset -  1) = -1.0;
        }

      }

      /********************/
      /*   x == nx face   */
      /********************/
      if(rightBC == 1) {
        // Compute row index
        k = idx / (ny - frontBC - backBC);
        j = idx % (ny - frontBC - backBC);
        i   = nx - 1;
        rowIdx = (k + bottomBC)*ny*nx + (j + frontBC)*nx + i;

        // Compute rowOffset
        rowOffset = k*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
          + (j + 1)*numEntriesPerGridRow + frontBC*numEntriesFrontRow;
        rowmap_view_h(rowIdx + 1) = rowOffset;

        if(stencil_type == FD) {
          // Fill column indices
          columns_view_h(rowOffset - 6) = rowIdx - ny*nx;
          columns_view_h(rowOffset - 5) = rowIdx - nx;
          columns_view_h(rowOffset - 4) = rowIdx - 1;
          columns_view_h(rowOffset - 3) = rowIdx;
          columns_view_h(rowOffset - 2) = rowIdx + nx;
          columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

          // Fill values
          values_view_h(rowOffset - 6) = -1.0;
          values_view_h(rowOffset - 5) = -1.0;
          values_view_h(rowOffset - 4) = -1.0;
          values_view_h(rowOffset - 3) =  5.0;
          values_view_h(rowOffset - 2) = -1.0;
          values_view_h(rowOffset - 1) = -1.0;
        } else if(stencil_type == FE) {
          // Fill column indices
          columns_view_h(rowOffset - 18) = rowIdx - ny*nx - nx - 1;
          columns_view_h(rowOffset - 17) = rowIdx - ny*nx - nx;
          columns_view_h(rowOffset - 16) = rowIdx - ny*nx - 1;
          columns_view_h(rowOffset - 15) = rowIdx - ny*nx;
          columns_view_h(rowOffset - 14) = rowIdx - ny*nx + nx - 1;
          columns_view_h(rowOffset - 13) = rowIdx - ny*nx + nx;
          columns_view_h(rowOffset - 12) = rowIdx - nx - 1;
          columns_view_h(rowOffset - 11) = rowIdx - nx;
          columns_view_h(rowOffset - 10) = rowIdx - 1;
          columns_view_h(rowOffset -  9) = rowIdx;
          columns_view_h(rowOffset -  8) = rowIdx + nx - 1;
          columns_view_h(rowOffset -  7) = rowIdx + nx;
          columns_view_h(rowOffset -  6) = rowIdx + nx*ny - nx - 1;
          columns_view_h(rowOffset -  5) = rowIdx + nx*ny - nx;
          columns_view_h(rowOffset -  4) = rowIdx + nx*ny - 1;
          columns_view_h(rowOffset -  3) = rowIdx + nx*ny;
          columns_view_h(rowOffset -  2) = rowIdx + nx*ny + nx - 1;
          columns_view_h(rowOffset -  1) = rowIdx + nx*ny + nx;

          // Fill values
          values_view_h(rowOffset - 18) = -1.0;
          values_view_h(rowOffset - 17) = -1.0;
          values_view_h(rowOffset - 16) = -2.0;
          values_view_h(rowOffset - 15) =  0.0;
          values_view_h(rowOffset - 14) = -1.0;
          values_view_h(rowOffset - 13) = -1.0;
          values_view_h(rowOffset - 12) = -2.0;
          values_view_h(rowOffset - 11) =  0.0;
          values_view_h(rowOffset - 10) =  0.0;
          values_view_h(rowOffset -  9) = 16.0;
          values_view_h(rowOffset -  8) = -2.0;
          values_view_h(rowOffset -  7) =  0.0;
          values_view_h(rowOffset -  6) = -1.0;
          values_view_h(rowOffset -  5) = -1.0;
          values_view_h(rowOffset -  4) = -2.0;
          values_view_h(rowOffset -  3) =  0.0;
          values_view_h(rowOffset -  2) = -1.0;
          values_view_h(rowOffset -  1) = -1.0;
        }

      }
    }

    // Loop over the y-face points
    for(ordinal_type idx = 0; idx < numYFace; ++idx) {
      /*******************/
      /*   y == 0 face   */
      /*******************/
      if(frontBC == 1) {
        // Compute row index
        k = idx / (nx - leftBC - rightBC);
        j = 0;
        i = idx % (nx - leftBC - rightBC);
        rowIdx = (k + bottomBC)*ny*nx + i + leftBC;

        // Compute rowOffset
        rowOffset = k*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
          + (i + 1)*faceStencilLength + leftBC*edgeStencilLength;
        rowmap_view_h(rowIdx + 1) = rowOffset;

        if(stencil_type == FD) {
          // Fill column indices
          columns_view_h(rowOffset - 6) = rowIdx - ny*nx;
          columns_view_h(rowOffset - 5) = rowIdx - 1;
          columns_view_h(rowOffset - 4) = rowIdx;
          columns_view_h(rowOffset - 3) = rowIdx + 1;
          columns_view_h(rowOffset - 2) = rowIdx + nx;
          columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

          // Fill values
          values_view_h(rowOffset - 6) = -1.0;
          values_view_h(rowOffset - 5) = -1.0;
          values_view_h(rowOffset - 4) =  5.0;
          values_view_h(rowOffset - 3) = -1.0;
          values_view_h(rowOffset - 2) = -1.0;
          values_view_h(rowOffset - 1) = -1.0;
        } else if(stencil_type == FE) {
          // Fill column indices
          columns_view_h(rowOffset - 18) = rowIdx - ny*nx - 1;
          columns_view_h(rowOffset - 17) = rowIdx - ny*nx;
          columns_view_h(rowOffset - 16) = rowIdx - ny*nx + 1;
          columns_view_h(rowOffset - 15) = rowIdx - ny*nx + nx - 1;
          columns_view_h(rowOffset - 14) = rowIdx - ny*nx + nx;
          columns_view_h(rowOffset - 13) = rowIdx - ny*nx + nx + 1;
          columns_view_h(rowOffset - 12) = rowIdx - 1;
          columns_view_h(rowOffset - 11) = rowIdx;
          columns_view_h(rowOffset - 10) = rowIdx + 1;
          columns_view_h(rowOffset -  9) = rowIdx + nx - 1;
          columns_view_h(rowOffset -  8) = rowIdx + nx;
          columns_view_h(rowOffset -  7) = rowIdx + nx + 1;
          columns_view_h(rowOffset -  6) = rowIdx + nx*ny - 1;
          columns_view_h(rowOffset -  5) = rowIdx + nx*ny;
          columns_view_h(rowOffset -  4) = rowIdx + nx*ny + 1;
          columns_view_h(rowOffset -  3) = rowIdx + nx*ny + nx - 1;
          columns_view_h(rowOffset -  2) = rowIdx + nx*ny + nx;
          columns_view_h(rowOffset -  1) = rowIdx + nx*ny + nx + 1;

          // Fill values
          values_view_h(rowOffset - 18) = -1.0;
          values_view_h(rowOffset - 17) =  0.0;
          values_view_h(rowOffset - 16) = -1.0;
          values_view_h(rowOffset - 15) = -1.0;
          values_view_h(rowOffset - 14) = -2.0;
          values_view_h(rowOffset - 13) = -1.0;
          values_view_h(rowOffset - 12) =  0.0;
          values_view_h(rowOffset - 11) = 16.0;
          values_view_h(rowOffset - 10) =  0.0;
          values_view_h(rowOffset -  9) = -2.0;
          values_view_h(rowOffset -  8) =  0.0;
          values_view_h(rowOffset -  7) = -2.0;
          values_view_h(rowOffset -  6) = -1.0;
          values_view_h(rowOffset -  5) =  0.0;
          values_view_h(rowOffset -  4) = -1.0;
          values_view_h(rowOffset -  3) = -1.0;
          values_view_h(rowOffset -  2) = -2.0;
          values_view_h(rowOffset -  1) = -1.0;
        }

      }

      /********************/
      /*   y == ny face   */
      /********************/
      if(backBC == 1) {
        // Compute row index
        k = idx / (nx - leftBC - rightBC);
        j = ny - 1 - frontBC;
        i = idx % (nx - leftBC - rightBC);
        rowIdx = (k + bottomBC)*ny*nx + (j + frontBC)*nx + i + leftBC;

        // Compute rowOffset
        rowOffset = k*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
          + j*numEntriesPerGridRow + frontBC*numEntriesFrontRow
          + (i + 1)*faceStencilLength + leftBC*edgeStencilLength;
        rowmap_view_h(rowIdx + 1) = rowOffset;

        if(stencil_type == FD) {
          // Fill column indices
          columns_view_h(rowOffset - 6) = rowIdx - ny*nx;
          columns_view_h(rowOffset - 5) = rowIdx - nx;
          columns_view_h(rowOffset - 4) = rowIdx - 1;
          columns_view_h(rowOffset - 3) = rowIdx;
          columns_view_h(rowOffset - 2) = rowIdx + 1;
          columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

          // Fill values
          values_view_h(rowOffset - 6) = -1.0;
          values_view_h(rowOffset - 5) = -1.0;
          values_view_h(rowOffset - 4) = -1.0;
          values_view_h(rowOffset - 3) =  5.0;
          values_view_h(rowOffset - 2) = -1.0;
          values_view_h(rowOffset - 1) = -1.0;
        } else if(stencil_type == FE) {
          // Fill column indices
          columns_view_h(rowOffset - 18) = rowIdx - ny*nx - nx - 1;
          columns_view_h(rowOffset - 17) = rowIdx - ny*nx - nx;
          columns_view_h(rowOffset - 16) = rowIdx - ny*nx - 1;
          columns_view_h(rowOffset - 15) = rowIdx - ny*nx - 1;
          columns_view_h(rowOffset - 14) = rowIdx - ny*nx;
          columns_view_h(rowOffset - 13) = rowIdx - ny*nx + 1;
          columns_view_h(rowOffset - 12) = rowIdx - nx - 1;
          columns_view_h(rowOffset - 11) = rowIdx - nx;
          columns_view_h(rowOffset - 10) = rowIdx - nx + 1;
          columns_view_h(rowOffset -  9) = rowIdx - 1;
          columns_view_h(rowOffset -  8) = rowIdx;
          columns_view_h(rowOffset -  7) = rowIdx + 1;
          columns_view_h(rowOffset -  6) = rowIdx + nx*ny - nx - 1;
          columns_view_h(rowOffset -  5) = rowIdx + nx*ny - nx;
          columns_view_h(rowOffset -  4) = rowIdx + nx*ny - nx + 1;
          columns_view_h(rowOffset -  3) = rowIdx + nx*ny - 1;
          columns_view_h(rowOffset -  2) = rowIdx + nx*ny;
          columns_view_h(rowOffset -  1) = rowIdx + nx*ny + 1;

          // Fill values
          values_view_h(rowOffset - 18) = -1.0;
          values_view_h(rowOffset - 17) = -2.0;
          values_view_h(rowOffset - 16) = -1.0;
          values_view_h(rowOffset - 15) = -1.0;
          values_view_h(rowOffset - 14) =  0.0;
          values_view_h(rowOffset - 13) = -1.0;
          values_view_h(rowOffset - 12) = -2.0;
          values_view_h(rowOffset - 11) =  0.0;
          values_view_h(rowOffset - 10) = -2.0;
          values_view_h(rowOffset -  9) =  0.0;
          values_view_h(rowOffset -  8) = 16.0;
          values_view_h(rowOffset -  7) =  0.0;
          values_view_h(rowOffset -  6) = -1.0;
          values_view_h(rowOffset -  5) = -2.0;
          values_view_h(rowOffset -  4) = -1.0;
          values_view_h(rowOffset -  3) = -1.0;
          values_view_h(rowOffset -  2) =  0.0;
          values_view_h(rowOffset -  1) = -1.0;
        }

      }
    }

    // Loop over the z-face points
    for(ordinal_type idx = 0; idx < numZFace; ++idx) {
      /*******************/
      /*   z == 0 face   */
      /*******************/
      if(bottomBC == 1) {
        // Compute row index
        k = 0;
        j = idx / (nx - leftBC - rightBC);
        i = idx % (nx - leftBC - rightBC);
        rowIdx = (j + frontBC)*nx + i + leftBC;

        // Compute rowOffset
        rowOffset = j*numEntriesFrontRow + frontBC*numEntriesBottomFrontRow
          + (i + 1)*faceStencilLength + leftBC*edgeStencilLength;
        rowmap_view_h(rowIdx + 1) = rowOffset;

        if(stencil_type == FD) {
          // Fill column indices
          columns_view_h(rowOffset - 6) = rowIdx - nx;
          columns_view_h(rowOffset - 5) = rowIdx - 1;
          columns_view_h(rowOffset - 4) = rowIdx;
          columns_view_h(rowOffset - 3) = rowIdx + 1;
          columns_view_h(rowOffset - 2) = rowIdx + nx;
          columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

          // Fill values
          values_view_h(rowOffset - 6) = -1.0;
          values_view_h(rowOffset - 5) = -1.0;
          values_view_h(rowOffset - 4) =  5.0;
          values_view_h(rowOffset - 3) = -1.0;
          values_view_h(rowOffset - 2) = -1.0;
          values_view_h(rowOffset - 1) = -1.0;
        } else if(stencil_type == FE) {
          // Fill column indices
          columns_view_h(rowOffset - 18) = rowIdx - nx - 1;
          columns_view_h(rowOffset - 17) = rowIdx - nx;
          columns_view_h(rowOffset - 16) = rowIdx - nx + 1;
          columns_view_h(rowOffset - 15) = rowIdx - 1;
          columns_view_h(rowOffset - 14) = rowIdx;
          columns_view_h(rowOffset - 13) = rowIdx + 1;
          columns_view_h(rowOffset - 12) = rowIdx + nx - 1;
          columns_view_h(rowOffset - 11) = rowIdx + nx;
          columns_view_h(rowOffset - 10) = rowIdx + nx + 1;
          columns_view_h(rowOffset -  9) = rowIdx + nx*ny - nx - 1;
          columns_view_h(rowOffset -  8) = rowIdx + nx*ny - nx;
          columns_view_h(rowOffset -  7) = rowIdx + nx*ny - nx + 1;
          columns_view_h(rowOffset -  6) = rowIdx + nx*ny - 1;
          columns_view_h(rowOffset -  5) = rowIdx + nx*ny;
          columns_view_h(rowOffset -  4) = rowIdx + nx*ny + 1;
          columns_view_h(rowOffset -  3) = rowIdx + nx*ny + nx - 1;
          columns_view_h(rowOffset -  2) = rowIdx + nx*ny + nx;
          columns_view_h(rowOffset -  1) = rowIdx + nx*ny + nx + 1;

          // Fill values
          values_view_h(rowOffset - 18) = -1.0;
          values_view_h(rowOffset - 17) =  0.0;
          values_view_h(rowOffset - 16) = -1.0;
          values_view_h(rowOffset - 15) =  0.0;
          values_view_h(rowOffset - 14) = 16.0;
          values_view_h(rowOffset - 13) =  0.0;
          values_view_h(rowOffset - 12) = -1.0;
          values_view_h(rowOffset - 11) =  0.0;
          values_view_h(rowOffset - 10) = -1.0;
          values_view_h(rowOffset -  9) = -1.0;
          values_view_h(rowOffset -  8) = -2.0;
          values_view_h(rowOffset -  7) = -1.0;
          values_view_h(rowOffset -  6) = -2.0;
          values_view_h(rowOffset -  5) =  0.0;
          values_view_h(rowOffset -  4) = -2.0;
          values_view_h(rowOffset -  3) = -1.0;
          values_view_h(rowOffset -  2) = -2.0;
          values_view_h(rowOffset -  1) = -1.0;
        }

      }

      /********************/
      /*   z == nz face   */
      /********************/
      if(topBC == 1) {
        // Compute row index
        k = nz - bottomBC - 1;
        j = idx / (nx - leftBC - rightBC);
        i = idx % (nx - leftBC - rightBC);
        rowIdx = (k + bottomBC)*ny*nx + (j + frontBC)*nx + i + leftBC;

        // Compute rowOffset
        rowOffset = k*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
          + j*numEntriesFrontRow + frontBC*numEntriesBottomFrontRow
          + (i + 1)*faceStencilLength + leftBC*edgeStencilLength;
        rowmap_view_h(rowIdx + 1) = rowOffset;

        if(stencil_type == FD) {
          // Fill column indices
          columns_view_h(rowOffset - 6) = rowIdx - ny*nx;
          columns_view_h(rowOffset - 5) = rowIdx - nx;
          columns_view_h(rowOffset - 4) = rowIdx - 1;
          columns_view_h(rowOffset - 3) = rowIdx;
          columns_view_h(rowOffset - 2) = rowIdx + 1;
          columns_view_h(rowOffset - 1) = rowIdx + nx;

          // Fill values
          values_view_h(rowOffset - 6) = -1.0;
          values_view_h(rowOffset - 5) = -1.0;
          values_view_h(rowOffset - 4) = -1.0;
          values_view_h(rowOffset - 3) =  5.0;
          values_view_h(rowOffset - 2) = -1.0;
          values_view_h(rowOffset - 1) = -1.0;
        } else if(stencil_type == FE) {
          // Fill column indices
          columns_view_h(rowOffset - 18) = rowIdx - nx*ny - nx - 1;
          columns_view_h(rowOffset - 17) = rowIdx - nx*ny - nx;
          columns_view_h(rowOffset - 16) = rowIdx - nx*ny - nx + 1;
          columns_view_h(rowOffset - 15) = rowIdx - nx*ny - 1;
          columns_view_h(rowOffset - 14) = rowIdx - nx*ny;
          columns_view_h(rowOffset - 13) = rowIdx - nx*ny + 1;
          columns_view_h(rowOffset - 12) = rowIdx - nx*ny + nx - 1;
          columns_view_h(rowOffset - 11) = rowIdx - nx*ny + nx;
          columns_view_h(rowOffset - 10) = rowIdx - nx*ny + nx + 1;
          columns_view_h(rowOffset -  9) = rowIdx - nx - 1;
          columns_view_h(rowOffset -  8) = rowIdx - nx;
          columns_view_h(rowOffset -  7) = rowIdx - nx + 1;
          columns_view_h(rowOffset -  6) = rowIdx - 1;
          columns_view_h(rowOffset -  5) = rowIdx;
          columns_view_h(rowOffset -  4) = rowIdx + 1;
          columns_view_h(rowOffset -  3) = rowIdx + nx - 1;
          columns_view_h(rowOffset -  2) = rowIdx + nx;
          columns_view_h(rowOffset -  1) = rowIdx + nx + 1;

          // Fill values
          values_view_h(rowOffset - 18) = -1.0;
          values_view_h(rowOffset - 17) = -2.0;
          values_view_h(rowOffset - 16) = -1.0;
          values_view_h(rowOffset - 15) = -2.0;
          values_view_h(rowOffset - 14) =  0.0;
          values_view_h(rowOffset - 13) = -2.0;
          values_view_h(rowOffset - 12) = -1.0;
          values_view_h(rowOffset - 11) = -2.0;
          values_view_h(rowOffset - 10) = -1.0;
          values_view_h(rowOffset -  9) = -1.0;
          values_view_h(rowOffset -  8) =  0.0;
          values_view_h(rowOffset -  7) = -1.0;
          values_view_h(rowOffset -  6) =  0.0;
          values_view_h(rowOffset -  5) = 16.0;
          values_view_h(rowOffset -  4) =  0.0;
          values_view_h(rowOffset -  3) = -1.0;
          values_view_h(rowOffset -  2) =  0.0;
          values_view_h(rowOffset -  1) = -1.0;
        }
      }
    }

    // Edges around the bottom face
    if(bottomBC == 1) {
      if(frontBC == 1) {
        for(ordinal_type idx = 0; idx < nx - leftBC - rightBC; ++idx) {
          // Compute row index
          k = 0;
          j = 0;
          i = idx;
          rowIdx = i + leftBC;

          // Compute rowOffset
          rowOffset = (i + 1)*edgeStencilLength + leftBC*cornerStencilLength;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 5) = rowIdx - 1;
            columns_view_h(rowOffset - 4) = rowIdx;
            columns_view_h(rowOffset - 3) = rowIdx + 1;
            columns_view_h(rowOffset - 2) = rowIdx + nx;
            columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

            // Fill values
            values_view_h(rowOffset - 5) = -1.0;
            values_view_h(rowOffset - 4) =  4.0;
            values_view_h(rowOffset - 3) = -1.0;
            values_view_h(rowOffset - 2) = -1.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset - 12) = rowIdx - 1;
            columns_view_h(rowOffset - 11) = rowIdx;
            columns_view_h(rowOffset - 10) = rowIdx + 1;
            columns_view_h(rowOffset -  9) = rowIdx + nx - 1;
            columns_view_h(rowOffset -  8) = rowIdx + nx;
            columns_view_h(rowOffset -  7) = rowIdx + nx + 1;
            columns_view_h(rowOffset -  6) = rowIdx + nx*ny - 1;
            columns_view_h(rowOffset -  5) = rowIdx + nx*ny;
            columns_view_h(rowOffset -  4) = rowIdx + nx*ny + 1;
            columns_view_h(rowOffset -  3) = rowIdx + nx*ny + nx - 1;
            columns_view_h(rowOffset -  2) = rowIdx + nx*ny + nx;
            columns_view_h(rowOffset -  1) = rowIdx + nx*ny + nx + 1;

            // Fill values
            values_view_h(rowOffset - 12) =  0.0;
            values_view_h(rowOffset - 11) =  8.0;
            values_view_h(rowOffset - 10) =  0.0;
            values_view_h(rowOffset -  9) = -1.0;
            values_view_h(rowOffset -  8) =  0.0;
            values_view_h(rowOffset -  7) = -1.0;
            values_view_h(rowOffset -  6) = -1.0;
            values_view_h(rowOffset -  5) =  0.0;
            values_view_h(rowOffset -  4) = -1.0;
            values_view_h(rowOffset -  3) = -1.0;
            values_view_h(rowOffset -  2) = -2.0;
            values_view_h(rowOffset -  1) = -1.0;
          }
        }
      }

      if(backBC == 1) {
        for(ordinal_type idx = 0; idx < nx - leftBC - rightBC; ++idx) {
          // Compute row index
          k = 0;
          j = ny - frontBC - 1;
          i = idx;
          rowIdx = (j + frontBC)*nx + i + leftBC;

          // Compute rowOffset
          rowOffset = j*numEntriesFrontRow + frontBC*numEntriesBottomFrontRow
            + (i + 1)*edgeStencilLength + leftBC*cornerStencilLength;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 5) = rowIdx - 1;
            columns_view_h(rowOffset - 4) = rowIdx;
            columns_view_h(rowOffset - 3) = rowIdx + 1;
            columns_view_h(rowOffset - 2) = rowIdx + nx;
            columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

            // Fill values
            values_view_h(rowOffset - 5) = -1.0;
            values_view_h(rowOffset - 4) =  4.0;
            values_view_h(rowOffset - 3) = -1.0;
            values_view_h(rowOffset - 2) = -1.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset - 12) = rowIdx - nx - 1;
            columns_view_h(rowOffset - 11) = rowIdx - nx;
            columns_view_h(rowOffset - 10) = rowIdx - nx + 1;
            columns_view_h(rowOffset -  9) = rowIdx - 1;
            columns_view_h(rowOffset -  8) = rowIdx;
            columns_view_h(rowOffset -  7) = rowIdx + 1;
            columns_view_h(rowOffset -  6) = rowIdx + nx*ny - nx - 1;
            columns_view_h(rowOffset -  5) = rowIdx + nx*ny - nx;
            columns_view_h(rowOffset -  4) = rowIdx + nx*ny - nx + 1;
            columns_view_h(rowOffset -  3) = rowIdx + nx*ny - 1;
            columns_view_h(rowOffset -  2) = rowIdx + nx*ny;
            columns_view_h(rowOffset -  1) = rowIdx + nx*ny + 1;

            // Fill values
            values_view_h(rowOffset - 12) = -1.0;
            values_view_h(rowOffset - 11) =  0.0;
            values_view_h(rowOffset - 10) = -1.0;
            values_view_h(rowOffset -  9) =  0.0;
            values_view_h(rowOffset -  8) =  8.0;
            values_view_h(rowOffset -  7) =  0.0;
            values_view_h(rowOffset -  6) = -1.0;
            values_view_h(rowOffset -  5) = -2.0;
            values_view_h(rowOffset -  4) = -1.0;
            values_view_h(rowOffset -  3) = -1.0;
            values_view_h(rowOffset -  2) =  0.0;
            values_view_h(rowOffset -  1) = -1.0;
          }
        }
      }

      if(leftBC == 1) {
        for(ordinal_type idx = 0; idx < ny - frontBC - backBC; ++idx) {
          // Compute row index
          k = 0;
          j = idx;
          i = 0;
          rowIdx = (j + frontBC)*nx;

          // Compute rowOffset
          rowOffset = j*numEntriesFrontRow + frontBC*numEntriesBottomFrontRow + edgeStencilLength;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 5) = rowIdx - nx;
            columns_view_h(rowOffset - 4) = rowIdx;
            columns_view_h(rowOffset - 3) = rowIdx + 1;
            columns_view_h(rowOffset - 2) = rowIdx + nx;
            columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

            // Fill values
            values_view_h(rowOffset - 5) = -1.0;
            values_view_h(rowOffset - 4) =  4.0;
            values_view_h(rowOffset - 3) = -1.0;
            values_view_h(rowOffset - 2) = -1.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset - 12) = rowIdx - nx;
            columns_view_h(rowOffset - 11) = rowIdx - nx + 1;
            columns_view_h(rowOffset - 10) = rowIdx;
            columns_view_h(rowOffset -  9) = rowIdx + 1;
            columns_view_h(rowOffset -  8) = rowIdx + nx;
            columns_view_h(rowOffset -  7) = rowIdx + nx + 1;
            columns_view_h(rowOffset -  6) = rowIdx + nx*ny - nx;
            columns_view_h(rowOffset -  5) = rowIdx + nx*ny - nx + 1;
            columns_view_h(rowOffset -  4) = rowIdx + nx*ny;
            columns_view_h(rowOffset -  3) = rowIdx + nx*ny + 1;
            columns_view_h(rowOffset -  2) = rowIdx + nx*ny + nx;
            columns_view_h(rowOffset -  1) = rowIdx + nx*ny + nx + 1;

            // Fill values
            values_view_h(rowOffset - 12) =  0.0;
            values_view_h(rowOffset - 11) = -1.0;
            values_view_h(rowOffset - 10) =  8.0;
            values_view_h(rowOffset -  9) =  0.0;
            values_view_h(rowOffset -  8) =  0.0;
            values_view_h(rowOffset -  7) = -1.0;
            values_view_h(rowOffset -  6) = -1.0;
            values_view_h(rowOffset -  5) = -1.0;
            values_view_h(rowOffset -  4) =  0.0;
            values_view_h(rowOffset -  3) = -2.0;
            values_view_h(rowOffset -  2) = -1.0;
            values_view_h(rowOffset -  1) = -1.0;
          }
        }
      }

      if(rightBC == 1) {
        for(ordinal_type idx = 0; idx < ny - frontBC - backBC; ++idx) {
          // Compute row index
          k = 0;
          j = idx;
          i = nx - 1;
          rowIdx = (j + frontBC)*nx + i;

          // Compute rowOffset
          rowOffset = (j + 1)*numEntriesFrontRow + frontBC*numEntriesBottomFrontRow;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 5) = rowIdx - nx;
            columns_view_h(rowOffset - 4) = rowIdx - 1;
            columns_view_h(rowOffset - 3) = rowIdx;
            columns_view_h(rowOffset - 2) = rowIdx + nx;
            columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

            // Fill values
            values_view_h(rowOffset - 5) = -1.0;
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) = -4.0;
            values_view_h(rowOffset - 2) = -1.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset - 12) = rowIdx - nx - 1;
            columns_view_h(rowOffset - 11) = rowIdx - nx;
            columns_view_h(rowOffset - 10) = rowIdx - 1;
            columns_view_h(rowOffset -  9) = rowIdx;
            columns_view_h(rowOffset -  8) = rowIdx + nx - 1;
            columns_view_h(rowOffset -  7) = rowIdx + nx;
            columns_view_h(rowOffset -  6) = rowIdx + nx*ny - nx - 1;
            columns_view_h(rowOffset -  5) = rowIdx + nx*ny - nx;
            columns_view_h(rowOffset -  4) = rowIdx + nx*ny - 1;
            columns_view_h(rowOffset -  3) = rowIdx + nx*ny;
            columns_view_h(rowOffset -  2) = rowIdx + nx*ny + nx - 1;
            columns_view_h(rowOffset -  1) = rowIdx + nx*ny + nx;

            // Fill values
            values_view_h(rowOffset - 12) = -1.0;
            values_view_h(rowOffset - 11) = -1.0;
            values_view_h(rowOffset - 10) = -2.0;
            values_view_h(rowOffset -  9) =  0.0;
            values_view_h(rowOffset -  8) = -1.0;
            values_view_h(rowOffset -  7) = -1.0;
            values_view_h(rowOffset -  6) = -1.0;
            values_view_h(rowOffset -  5) =  0.0;
            values_view_h(rowOffset -  4) =  0.0;
            values_view_h(rowOffset -  3) =  8.0;
            values_view_h(rowOffset -  2) = -1.0;
            values_view_h(rowOffset -  1) =  0.0;
          }
        }
      }
    }

    // Edges around the top face
    if(topBC == 1) {
      if(frontBC == 1) {
        for(ordinal_type idx = 0; idx < nx - leftBC - rightBC; ++idx) {
          // Compute row index
          k = nz - bottomBC - 1;
          j = 0;
          i = idx;
          rowIdx = (k + bottomBC)*ny*nx + i + leftBC;

          // Compute rowOffset
          rowOffset = k*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
            + (i + 1)*edgeStencilLength + leftBC*cornerStencilLength;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 5) = rowIdx - ny*nx;
            columns_view_h(rowOffset - 4) = rowIdx - 1;
            columns_view_h(rowOffset - 3) = rowIdx;
            columns_view_h(rowOffset - 2) = rowIdx + 1;
            columns_view_h(rowOffset - 1) = rowIdx + nx;

            // Fill values
            values_view_h(rowOffset - 5) = -1.0;
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) =  4.0;
            values_view_h(rowOffset - 2) = -1.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset - 12) = rowIdx - nx*ny - 1;
            columns_view_h(rowOffset - 11) = rowIdx - nx*ny;
            columns_view_h(rowOffset - 10) = rowIdx - nx*ny + 1;
            columns_view_h(rowOffset -  9) = rowIdx - nx*ny + nx - 1;
            columns_view_h(rowOffset -  8) = rowIdx - nx*ny + nx;
            columns_view_h(rowOffset -  7) = rowIdx - nx*ny + nx + 1;
            columns_view_h(rowOffset -  6) = rowIdx - 1;
            columns_view_h(rowOffset -  5) = rowIdx;
            columns_view_h(rowOffset -  4) = rowIdx + 1;
            columns_view_h(rowOffset -  3) = rowIdx + nx - 1;
            columns_view_h(rowOffset -  2) = rowIdx + nx;
            columns_view_h(rowOffset -  1) = rowIdx + nx + 1;

            // Fill values
            values_view_h(rowOffset - 12) = -1.0;
            values_view_h(rowOffset - 11) =  0.0;
            values_view_h(rowOffset - 10) = -1.0;
            values_view_h(rowOffset -  9) = -1.0;
            values_view_h(rowOffset -  8) = -2.0;
            values_view_h(rowOffset -  7) = -1.0;
            values_view_h(rowOffset -  6) =  0.0;
            values_view_h(rowOffset -  5) =  8.0;
            values_view_h(rowOffset -  4) =  0.0;
            values_view_h(rowOffset -  3) = -1.0;
            values_view_h(rowOffset -  2) =  0.0;
            values_view_h(rowOffset -  1) = -1.0;
          }
        }
      }

      if(backBC == 1) {
        for(ordinal_type idx = 0; idx < nx - leftBC - rightBC; ++idx) {
          // Compute row index
          k = nz - bottomBC - 1;
          j = ny - frontBC - 1;
          i = idx;
          rowIdx = (k + bottomBC)*ny*nx + (j + frontBC)*nx + i + leftBC;

          // Compute rowOffset
          rowOffset = k*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
            + j*numEntriesFrontRow + frontBC*numEntriesBottomFrontRow
            + (i + 1)*edgeStencilLength + leftBC*cornerStencilLength;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 5) = rowIdx - ny*nx;
            columns_view_h(rowOffset - 4) = rowIdx - nx;
            columns_view_h(rowOffset - 3) = rowIdx - 1;
            columns_view_h(rowOffset - 2) = rowIdx;
            columns_view_h(rowOffset - 1) = rowIdx + 1;

            // Fill values
            values_view_h(rowOffset - 5) = -1.0;
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) = -1.0;
            values_view_h(rowOffset - 2) =  4.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset - 12) = rowIdx - nx*ny - nx - 1;
            columns_view_h(rowOffset - 11) = rowIdx - nx*ny - nx;
            columns_view_h(rowOffset - 10) = rowIdx - nx*ny - nx + 1;
            columns_view_h(rowOffset -  9) = rowIdx - nx*ny - 1;
            columns_view_h(rowOffset -  8) = rowIdx - nx*ny;
            columns_view_h(rowOffset -  7) = rowIdx - nx*ny + 1;
            columns_view_h(rowOffset -  6) = rowIdx - nx - 1;
            columns_view_h(rowOffset -  5) = rowIdx - nx;
            columns_view_h(rowOffset -  4) = rowIdx - nx + 1;
            columns_view_h(rowOffset -  3) = rowIdx - 1;
            columns_view_h(rowOffset -  2) = rowIdx;
            columns_view_h(rowOffset -  1) = rowIdx + 1;

            // Fill values
            values_view_h(rowOffset - 12) = -1.0;
            values_view_h(rowOffset - 11) = -2.0;
            values_view_h(rowOffset - 10) = -1.0;
            values_view_h(rowOffset -  9) = -1.0;
            values_view_h(rowOffset -  8) =  0.0;
            values_view_h(rowOffset -  7) = -1.0;
            values_view_h(rowOffset -  6) = -1.0;
            values_view_h(rowOffset -  5) =  0.0;
            values_view_h(rowOffset -  4) = -1.0;
            values_view_h(rowOffset -  3) =  0.0;
            values_view_h(rowOffset -  2) =  8.0;
            values_view_h(rowOffset -  1) =  0.0;
          }
        }
      }

      if(leftBC == 1) {
        for(ordinal_type idx = 0; idx < ny - frontBC - backBC; ++idx) {
          // Compute row index
          k = nz - bottomBC - 1;
          j = idx;
          i = 0;
          rowIdx = (k + bottomBC)*ny*nx + (j + frontBC)*nx;

          // Compute rowOffset
          rowOffset = k*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
            + j*numEntriesFrontRow + frontBC*numEntriesBottomFrontRow
            + edgeStencilLength;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 5) = rowIdx - ny*nx;
            columns_view_h(rowOffset - 4) = rowIdx - 1;
            columns_view_h(rowOffset - 3) = rowIdx;
            columns_view_h(rowOffset - 2) = rowIdx + 1;
            columns_view_h(rowOffset - 1) = rowIdx + nx;

            // Fill values
            values_view_h(rowOffset - 5) = -1.0;
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) =  4.0;
            values_view_h(rowOffset - 2) = -1.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset - 12) = rowIdx - nx*ny - 1;
            columns_view_h(rowOffset - 11) = rowIdx - nx*ny;
            columns_view_h(rowOffset - 10) = rowIdx - nx*ny + 1;
            columns_view_h(rowOffset -  9) = rowIdx - nx*ny + nx - 1;
            columns_view_h(rowOffset -  8) = rowIdx - nx*ny + nx;
            columns_view_h(rowOffset -  7) = rowIdx - nx*ny + nx + 1;
            columns_view_h(rowOffset -  6) = rowIdx - 1;
            columns_view_h(rowOffset -  5) = rowIdx;
            columns_view_h(rowOffset -  4) = rowIdx + 1;
            columns_view_h(rowOffset -  3) = rowIdx + nx - 1;
            columns_view_h(rowOffset -  2) = rowIdx + nx;
            columns_view_h(rowOffset -  1) = rowIdx + nx + 1;

            // Fill values
            values_view_h(rowOffset - 12) = -1.0;
            values_view_h(rowOffset - 11) =  0.0;
            values_view_h(rowOffset - 10) = -1.0;
            values_view_h(rowOffset -  9) = -1.0;
            values_view_h(rowOffset -  8) = -2.0;
            values_view_h(rowOffset -  7) = -1.0;
            values_view_h(rowOffset -  6) =  0.0;
            values_view_h(rowOffset -  5) =  8.0;
            values_view_h(rowOffset -  4) =  0.0;
            values_view_h(rowOffset -  3) = -1.0;
            values_view_h(rowOffset -  2) =  0.0;
            values_view_h(rowOffset -  1) = -1.0;
          }
        }
      }

      if(rightBC == 1) {
        for(ordinal_type idx = 0; idx < ny - frontBC - backBC; ++idx) {
          // Compute row index
          k = nz - bottomBC - 1;
          j = idx;
          i = nx - 1;
          rowIdx = (k + bottomBC)*ny*nx + (j + frontBC)*nx + i;

          // Compute rowOffset
          rowOffset = k*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
            + (j + 1)*numEntriesFrontRow + frontBC*numEntriesBottomFrontRow;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 5) = rowIdx - ny*nx;
            columns_view_h(rowOffset - 4) = rowIdx - nx;
            columns_view_h(rowOffset - 3) = rowIdx - 1;
            columns_view_h(rowOffset - 2) = rowIdx;
            columns_view_h(rowOffset - 1) = rowIdx + nx;

            // Fill values
            values_view_h(rowOffset - 5) = -1.0;
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) = -1.0;
            values_view_h(rowOffset - 2) =  4.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset - 12) = rowIdx - nx*ny - nx - 1;
            columns_view_h(rowOffset - 11) = rowIdx - nx*ny - nx;
            columns_view_h(rowOffset - 10) = rowIdx - nx*ny - 1;
            columns_view_h(rowOffset -  9) = rowIdx - nx*ny;
            columns_view_h(rowOffset -  8) = rowIdx - nx*ny + nx - 1;
            columns_view_h(rowOffset -  7) = rowIdx - nx*ny + nx;
            columns_view_h(rowOffset -  6) = rowIdx - nx - 1;
            columns_view_h(rowOffset -  5) = rowIdx - nx;
            columns_view_h(rowOffset -  4) = rowIdx - 1;
            columns_view_h(rowOffset -  3) = rowIdx;
            columns_view_h(rowOffset -  2) = rowIdx + nx - 1;
            columns_view_h(rowOffset -  1) = rowIdx + nx;

            // Fill values
            values_view_h(rowOffset - 12) = -1.0;
            values_view_h(rowOffset - 11) = -1.0;
            values_view_h(rowOffset - 10) = -2.0;
            values_view_h(rowOffset -  9) =  0.0;
            values_view_h(rowOffset -  8) = -1.0;
            values_view_h(rowOffset -  7) = -1.0;
            values_view_h(rowOffset -  6) = -1.0;
            values_view_h(rowOffset -  5) =  0.0;
            values_view_h(rowOffset -  4) =  0.0;
            values_view_h(rowOffset -  3) =  8.0;
            values_view_h(rowOffset -  2) = -1.0;
            values_view_h(rowOffset -  1) =  0.0;
          }
        }
      }
    }

    // Vertical edges front
    if(frontBC == 1) {
      if(leftBC == 1) {
        for(ordinal_type idx = 0; idx < nz - bottomBC - topBC; ++idx) {
          // Compute row index
          k = idx;
          j = 0;
          i = 0;
          rowIdx = (k + bottomBC)*ny*nx;

          // Compute rowOffset
          rowOffset = k*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane + edgeStencilLength;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 5) = rowIdx - ny*nx;
            columns_view_h(rowOffset - 4) = rowIdx;
            columns_view_h(rowOffset - 3) = rowIdx + 1;
            columns_view_h(rowOffset - 2) = rowIdx + nx;
            columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

            // Fill values
            values_view_h(rowOffset - 5) = -1.0;
            values_view_h(rowOffset - 4) =  4.0;
            values_view_h(rowOffset - 3) = -1.0;
            values_view_h(rowOffset - 2) = -1.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset - 12) = rowIdx - nx*ny;
            columns_view_h(rowOffset - 11) = rowIdx - nx*ny + 1;
            columns_view_h(rowOffset - 10) = rowIdx - nx*ny + nx;
            columns_view_h(rowOffset -  9) = rowIdx - nx*ny + nx + 1;
            columns_view_h(rowOffset -  8) = rowIdx;
            columns_view_h(rowOffset -  7) = rowIdx + 1;
            columns_view_h(rowOffset -  6) = rowIdx + nx;
            columns_view_h(rowOffset -  5) = rowIdx + nx + 1;
            columns_view_h(rowOffset -  4) = rowIdx + ny*nx;
            columns_view_h(rowOffset -  3) = rowIdx + ny*nx + 1;
            columns_view_h(rowOffset -  2) = rowIdx + ny*nx + nx;
            columns_view_h(rowOffset -  1) = rowIdx + ny*nx + nx + 1;

            // Fill values
            values_view_h(rowOffset - 12) =  0.0;
            values_view_h(rowOffset - 11) = -1.0;
            values_view_h(rowOffset - 10) = -1.0;
            values_view_h(rowOffset -  9) = -1.0;
            values_view_h(rowOffset -  8) =  8.0;
            values_view_h(rowOffset -  7) =  0.0;
            values_view_h(rowOffset -  6) =  0.0;
            values_view_h(rowOffset -  5) = -2.0;
            values_view_h(rowOffset -  4) =  0.0;
            values_view_h(rowOffset -  3) = -1.0;
            values_view_h(rowOffset -  2) = -1.0;
            values_view_h(rowOffset -  1) = -1.0;
          }
        }
      }

      if(rightBC == 1) {
        for(ordinal_type idx = 0; idx < nz - bottomBC - topBC; ++idx) {
          // Compute row index
          k = idx;
          j = 0;
          i = nx - leftBC - rightBC;
          rowIdx = (k + bottomBC)*ny*nx + i + leftBC;

          // Compute rowOffset
          rowOffset = k*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
            + i*faceStencilLength + (leftBC + rightBC)*edgeStencilLength;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 5) = rowIdx - ny*nx;
            columns_view_h(rowOffset - 4) = rowIdx - 1;
            columns_view_h(rowOffset - 3) = rowIdx;
            columns_view_h(rowOffset - 2) = rowIdx + nx;
            columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

            // Fill values
            values_view_h(rowOffset - 5) = -1.0;
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) =  4.0;
            values_view_h(rowOffset - 2) = -1.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset - 12) = rowIdx - nx*ny - 1;
            columns_view_h(rowOffset - 11) = rowIdx - nx*ny;
            columns_view_h(rowOffset - 10) = rowIdx - nx*ny + nx - 1;
            columns_view_h(rowOffset -  9) = rowIdx - nx*ny + nx;
            columns_view_h(rowOffset -  8) = rowIdx - 1;
            columns_view_h(rowOffset -  7) = rowIdx;
            columns_view_h(rowOffset -  6) = rowIdx + nx - 1;
            columns_view_h(rowOffset -  5) = rowIdx + nx;
            columns_view_h(rowOffset -  4) = rowIdx + ny*nx - 1;
            columns_view_h(rowOffset -  3) = rowIdx + ny*nx;
            columns_view_h(rowOffset -  2) = rowIdx + ny*nx + nx - 1;
            columns_view_h(rowOffset -  1) = rowIdx + ny*nx + nx;

            // Fill values
            values_view_h(rowOffset - 12) = -1.0;
            values_view_h(rowOffset - 11) =  0.0;
            values_view_h(rowOffset - 10) = -1.0;
            values_view_h(rowOffset -  9) = -1.0;
            values_view_h(rowOffset -  8) =  0.0;
            values_view_h(rowOffset -  7) =  8.0;
            values_view_h(rowOffset -  6) = -2.0;
            values_view_h(rowOffset -  5) =  0.0;
            values_view_h(rowOffset -  4) = -1.0;
            values_view_h(rowOffset -  3) =  0.0;
            values_view_h(rowOffset -  2) = -1.0;
            values_view_h(rowOffset -  1) = -1.0;
          }
        }
      }
    }

    // Vertical edges back
    if(backBC == 1) {
      if(leftBC == 1) {
        for(ordinal_type idx = 0; idx < nz - bottomBC - topBC; ++idx) {
          // Compute row index
          k = idx;
          j = ny - frontBC - backBC;
          i = 0;
          rowIdx = (k + bottomBC)*ny*nx + (j + frontBC)*nx;

          // Compute rowOffset
          rowOffset = k*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
            + j*numEntriesPerGridRow + frontBC*numEntriesFrontRow + edgeStencilLength;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 5) = rowIdx - ny*nx;
            columns_view_h(rowOffset - 4) = rowIdx - nx;
            columns_view_h(rowOffset - 3) = rowIdx;
            columns_view_h(rowOffset - 2) = rowIdx + 1;
            columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

            // Fill values
            values_view_h(rowOffset - 5) = -1.0;
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) =  4.0;
            values_view_h(rowOffset - 2) = -1.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset - 12) = rowIdx - nx*ny - nx;
            columns_view_h(rowOffset - 11) = rowIdx - nx*ny - nx + 1;
            columns_view_h(rowOffset - 10) = rowIdx - nx*ny;
            columns_view_h(rowOffset -  9) = rowIdx - nx*ny + 1;
            columns_view_h(rowOffset -  8) = rowIdx - nx;
            columns_view_h(rowOffset -  7) = rowIdx - nx + 1;
            columns_view_h(rowOffset -  6) = rowIdx;
            columns_view_h(rowOffset -  5) = rowIdx + 1;
            columns_view_h(rowOffset -  4) = rowIdx + ny*nx - nx;
            columns_view_h(rowOffset -  3) = rowIdx + ny*nx - nx + 1;
            columns_view_h(rowOffset -  2) = rowIdx + ny*nx;
            columns_view_h(rowOffset -  1) = rowIdx + ny*nx + 1;

            // Fill values
            values_view_h(rowOffset - 12) = -1.0;
            values_view_h(rowOffset - 11) = -1.0;
            values_view_h(rowOffset - 10) =  0.0;
            values_view_h(rowOffset -  9) = -1.0;
            values_view_h(rowOffset -  8) =  0.0;
            values_view_h(rowOffset -  7) = -2.0;
            values_view_h(rowOffset -  6) =  8.0;
            values_view_h(rowOffset -  5) =  0.0;
            values_view_h(rowOffset -  4) = -1.0;
            values_view_h(rowOffset -  3) = -1.0;
            values_view_h(rowOffset -  2) =  0.0;
            values_view_h(rowOffset -  1) = -1.0;
          }
        }
      }

      if(rightBC == 1) {
        for(ordinal_type idx = 0; idx < nz - bottomBC - topBC; ++idx) {
          // Compute row index
          k = idx;
          j = ny - frontBC - backBC;
          i = nx - leftBC - rightBC;
          rowIdx = (k + bottomBC)*ny*nx + (j + frontBC)*nx + i + leftBC;

          // Compute rowOffset
          rowOffset = k*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
            + j*numEntriesPerGridRow + frontBC*numEntriesFrontRow
            + i*faceStencilLength + (leftBC + rightBC)*edgeStencilLength;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 5) = rowIdx - ny*nx;
            columns_view_h(rowOffset - 4) = rowIdx - nx;
            columns_view_h(rowOffset - 3) = rowIdx - 1;
            columns_view_h(rowOffset - 2) = rowIdx;
            columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

            // Fill values
            values_view_h(rowOffset - 5) = -1.0;
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) = -1.0;
            values_view_h(rowOffset - 2) =  4.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset - 12) = rowIdx - nx*ny - nx - 1;
            columns_view_h(rowOffset - 11) = rowIdx - nx*ny - nx;
            columns_view_h(rowOffset - 10) = rowIdx - nx*ny - 1;
            columns_view_h(rowOffset -  9) = rowIdx - nx*ny;
            columns_view_h(rowOffset -  8) = rowIdx - nx - 1;
            columns_view_h(rowOffset -  7) = rowIdx - nx;
            columns_view_h(rowOffset -  6) = rowIdx - 1;
            columns_view_h(rowOffset -  5) = rowIdx;
            columns_view_h(rowOffset -  4) = rowIdx + ny*nx - nx - 1;
            columns_view_h(rowOffset -  3) = rowIdx + ny*nx - nx;
            columns_view_h(rowOffset -  2) = rowIdx + ny*nx - 1;
            columns_view_h(rowOffset -  1) = rowIdx + ny*nx;

            // Fill values
            values_view_h(rowOffset - 12) = -1.0;
            values_view_h(rowOffset - 11) = -1.0;
            values_view_h(rowOffset - 10) = -1.0;
            values_view_h(rowOffset -  9) =  0.0;
            values_view_h(rowOffset -  8) = -2.0;
            values_view_h(rowOffset -  7) =  0.0;
            values_view_h(rowOffset -  6) =  0.0;
            values_view_h(rowOffset -  5) =  8.0;
            values_view_h(rowOffset -  4) = -1.0;
            values_view_h(rowOffset -  3) = -1.0;
            values_view_h(rowOffset -  2) = -1.0;
            values_view_h(rowOffset -  1) =  0.0;
          }
        }
      }
    }

    // Bottom corners
    if(bottomBC == 1) {
      if(frontBC == 1) {
        if(leftBC == 1) {
          rowIdx = 0;
          rowOffset = cornerStencilLength;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 4) = rowIdx;
            columns_view_h(rowOffset - 3) = rowIdx + 1;
            columns_view_h(rowOffset - 2) = rowIdx + nx;
            columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

            // Fill values
            values_view_h(rowOffset - 4) =  3.0;
            values_view_h(rowOffset - 3) = -1.0;
            values_view_h(rowOffset - 2) = -1.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset -  8) = rowIdx;
            columns_view_h(rowOffset -  7) = rowIdx + 1;
            columns_view_h(rowOffset -  6) = rowIdx + nx;
            columns_view_h(rowOffset -  5) = rowIdx + nx + 1;
            columns_view_h(rowOffset -  4) = rowIdx + ny*nx;
            columns_view_h(rowOffset -  3) = rowIdx + ny*nx + 1;
            columns_view_h(rowOffset -  2) = rowIdx + ny*nx + nx;
            columns_view_h(rowOffset -  1) = rowIdx + ny*nx + nx + 1;

            // Fill values
            values_view_h(rowOffset -  8) =  4.0;
            values_view_h(rowOffset -  7) =  0.0;
            values_view_h(rowOffset -  6) =  0.0;
            values_view_h(rowOffset -  5) = -1.0;
            values_view_h(rowOffset -  4) =  0.0;
            values_view_h(rowOffset -  3) = -1.0;
            values_view_h(rowOffset -  2) = -1.0;
            values_view_h(rowOffset -  1) = -1.0;
          }
        }

        if(rightBC == 1) {
          rowIdx = nx - 1;
          rowOffset = numEntriesBottomFrontRow;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 4) = rowIdx - 1;
            columns_view_h(rowOffset - 3) = rowIdx;
            columns_view_h(rowOffset - 2) = rowIdx + nx;
            columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

            // Fill values
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) =  3.0;
            values_view_h(rowOffset - 2) = -1.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset -  8) = rowIdx - 1;
            columns_view_h(rowOffset -  7) = rowIdx;
            columns_view_h(rowOffset -  6) = rowIdx + nx - 1;
            columns_view_h(rowOffset -  5) = rowIdx + nx;
            columns_view_h(rowOffset -  4) = rowIdx + ny*nx - 1;
            columns_view_h(rowOffset -  3) = rowIdx + ny*nx;
            columns_view_h(rowOffset -  2) = rowIdx + ny*nx + nx - 1;
            columns_view_h(rowOffset -  1) = rowIdx + ny*nx + nx;

            // Fill values
            values_view_h(rowOffset -  8) =  0.0;
            values_view_h(rowOffset -  7) =  4.0;
            values_view_h(rowOffset -  6) = -1.0;
            values_view_h(rowOffset -  5) =  0.0;
            values_view_h(rowOffset -  4) = -1.0;
            values_view_h(rowOffset -  3) =  0.0;
            values_view_h(rowOffset -  2) = -1.0;
            values_view_h(rowOffset -  1) = -1.0;
          }
        }
      }

      if(backBC == 1) {
        if(leftBC == 1) {
          rowIdx = (ny - 1)*nx;
          rowOffset = (ny - frontBC - 1)*numEntriesFrontRow + frontBC*numEntriesBottomFrontRow
            + cornerStencilLength;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 4) = rowIdx - nx;
            columns_view_h(rowOffset - 3) = rowIdx;
            columns_view_h(rowOffset - 2) = rowIdx + 1;
            columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

            // Fill values
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) =  3.0;
            values_view_h(rowOffset - 2) = -1.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset -  8) = rowIdx - nx;
            columns_view_h(rowOffset -  7) = rowIdx - nx + 1;
            columns_view_h(rowOffset -  6) = rowIdx;
            columns_view_h(rowOffset -  5) = rowIdx + 1;
            columns_view_h(rowOffset -  4) = rowIdx + ny*nx - nx;
            columns_view_h(rowOffset -  3) = rowIdx + ny*nx - nx + 1;
            columns_view_h(rowOffset -  2) = rowIdx + ny*nx;
            columns_view_h(rowOffset -  1) = rowIdx + ny*nx + 1;

            // Fill values
            values_view_h(rowOffset -  8) =  0.0;
            values_view_h(rowOffset -  7) = -1.0;
            values_view_h(rowOffset -  6) =  4.0;
            values_view_h(rowOffset -  5) =  0.0;
            values_view_h(rowOffset -  4) = -1.0;
            values_view_h(rowOffset -  3) = -1.0;
            values_view_h(rowOffset -  2) =  0.0;
            values_view_h(rowOffset -  1) = -1.0;
          }
        }

        if(rightBC == 1) {
          rowIdx = ny*nx - 1;
          rowOffset = numEntriesBottomPlane;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 4) = rowIdx - nx;
            columns_view_h(rowOffset - 3) = rowIdx - 1;
            columns_view_h(rowOffset - 2) = rowIdx;
            columns_view_h(rowOffset - 1) = rowIdx + ny*nx;

            // Fill values
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) = -1.0;
            values_view_h(rowOffset - 2) =  4.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset -  8) = rowIdx - nx - 1;
            columns_view_h(rowOffset -  7) = rowIdx - nx;
            columns_view_h(rowOffset -  6) = rowIdx - 1;
            columns_view_h(rowOffset -  5) = rowIdx;
            columns_view_h(rowOffset -  4) = rowIdx + ny*nx - nx - 1;
            columns_view_h(rowOffset -  3) = rowIdx + ny*nx - nx;
            columns_view_h(rowOffset -  2) = rowIdx + ny*nx - 1;
            columns_view_h(rowOffset -  1) = rowIdx + ny*nx;

            // Fill values
            values_view_h(rowOffset -  8) = -1.0;
            values_view_h(rowOffset -  7) =  0.0;
            values_view_h(rowOffset -  6) =  0.0;
            values_view_h(rowOffset -  5) =  4.0;
            values_view_h(rowOffset -  4) = -1.0;
            values_view_h(rowOffset -  3) = -1.0;
            values_view_h(rowOffset -  2) = -1.0;
            values_view_h(rowOffset -  1) =  0.0;
          }
        }
      }
    }

    // Top corners
    if(topBC == 1) {
      if(frontBC == 1) {
        if(leftBC == 1) {
          rowIdx = (nz - 1)*ny*nx;
          rowOffset = (nz - bottomBC - 1)*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
            + cornerStencilLength;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 4) = rowIdx - ny*nx;
            columns_view_h(rowOffset - 3) = rowIdx;
            columns_view_h(rowOffset - 2) = rowIdx + 1;
            columns_view_h(rowOffset - 1) = rowIdx + nx;

            // Fill values
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) =  4.0;
            values_view_h(rowOffset - 2) = -1.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset -  8) = rowIdx - ny*nx;
            columns_view_h(rowOffset -  7) = rowIdx - ny*nx + 1;
            columns_view_h(rowOffset -  6) = rowIdx - ny*nx + nx;
            columns_view_h(rowOffset -  5) = rowIdx - ny*nx + nx + 1;
            columns_view_h(rowOffset -  4) = rowIdx;
            columns_view_h(rowOffset -  3) = rowIdx + 1;
            columns_view_h(rowOffset -  2) = rowIdx + nx;
            columns_view_h(rowOffset -  1) = rowIdx + nx + 1;

            // Fill values
            values_view_h(rowOffset -  8) =  0.0;
            values_view_h(rowOffset -  7) = -1.0;
            values_view_h(rowOffset -  6) = -1.0;
            values_view_h(rowOffset -  5) = -1.0;
            values_view_h(rowOffset -  4) =  4.0;
            values_view_h(rowOffset -  3) =  0.0;
            values_view_h(rowOffset -  2) =  0.0;
            values_view_h(rowOffset -  1) = -1.0;
          }
        }

        if(rightBC == 1) {
          rowIdx = (nz - 1)*ny*nx + nx - 1;
          rowOffset = (nz - bottomBC - 1)*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
            + numEntriesBottomFrontRow;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 4) = rowIdx - ny*nx;
            columns_view_h(rowOffset - 3) = rowIdx - 1;
            columns_view_h(rowOffset - 2) = rowIdx;
            columns_view_h(rowOffset - 1) = rowIdx + nx;

            // Fill values
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) = -1.0;
            values_view_h(rowOffset - 2) =  4.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset -  8) = rowIdx - ny*nx - 1;
            columns_view_h(rowOffset -  7) = rowIdx - ny*nx;
            columns_view_h(rowOffset -  6) = rowIdx - ny*nx + nx - 1;
            columns_view_h(rowOffset -  5) = rowIdx - ny*nx + nx;
            columns_view_h(rowOffset -  4) = rowIdx - 1;
            columns_view_h(rowOffset -  3) = rowIdx;
            columns_view_h(rowOffset -  2) = rowIdx + nx - 1;
            columns_view_h(rowOffset -  1) = rowIdx + nx;

            // Fill values
            values_view_h(rowOffset -  8) = -1.0;
            values_view_h(rowOffset -  7) =  0.0;
            values_view_h(rowOffset -  6) = -1.0;
            values_view_h(rowOffset -  5) = -1.0;
            values_view_h(rowOffset -  4) =  0.0;
            values_view_h(rowOffset -  3) =  4.0;
            values_view_h(rowOffset -  2) = -1.0;
            values_view_h(rowOffset -  1) =  0.0;
          }
        }
      }

      if(backBC == 1) {
        if(leftBC == 1) {
          rowIdx = nz*ny*nx - nx;
          rowOffset = (nz - bottomBC - 1)*numEntriesPerGridPlane + bottomBC*numEntriesBottomPlane
            + (ny - frontBC - 1)*numEntriesFrontRow + frontBC*numEntriesBottomFrontRow
            + cornerStencilLength;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 4) = rowIdx - ny*nx;
            columns_view_h(rowOffset - 3) = rowIdx - nx;
            columns_view_h(rowOffset - 2) = rowIdx;
            columns_view_h(rowOffset - 1) = rowIdx + 1;

            // Fill values
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) = -1.0;
            values_view_h(rowOffset - 2) =  4.0;
            values_view_h(rowOffset - 1) = -1.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset -  8) = rowIdx - ny*nx - nx;
            columns_view_h(rowOffset -  7) = rowIdx - ny*nx - nx + 1;
            columns_view_h(rowOffset -  6) = rowIdx - ny*nx;
            columns_view_h(rowOffset -  5) = rowIdx - ny*nx + 1;
            columns_view_h(rowOffset -  4) = rowIdx - nx;
            columns_view_h(rowOffset -  3) = rowIdx - nx + 1;
            columns_view_h(rowOffset -  2) = rowIdx;
            columns_view_h(rowOffset -  1) = rowIdx + 1;

            // Fill values
            values_view_h(rowOffset -  8) = -1.0;
            values_view_h(rowOffset -  7) = -1.0;
            values_view_h(rowOffset -  6) =  0.0;
            values_view_h(rowOffset -  5) = -1.0;
            values_view_h(rowOffset -  4) =  0.0;
            values_view_h(rowOffset -  3) = -1.0;
            values_view_h(rowOffset -  2) =  4.0;
            values_view_h(rowOffset -  1) =  0.0;
          }
        }

        if(rightBC == 1) {
          rowIdx = nz*ny*nx - 1;
          rowOffset = numEntries;
          rowmap_view_h(rowIdx + 1) = rowOffset;

          if(stencil_type == FD) {
            // Fill column indices
            columns_view_h(rowOffset - 4) = rowIdx - ny*nx;
            columns_view_h(rowOffset - 3) = rowIdx - nx;
            columns_view_h(rowOffset - 2) = rowIdx - 1;
            columns_view_h(rowOffset - 1) = rowIdx;

            // Fill values
            values_view_h(rowOffset - 4) = -1.0;
            values_view_h(rowOffset - 3) = -1.0;
            values_view_h(rowOffset - 2) = -1.0;
            values_view_h(rowOffset - 1) =  4.0;
          } else if(stencil_type == FE) {
            // Fill column indices
            columns_view_h(rowOffset -  8) = rowIdx - ny*nx - nx - 1;
            columns_view_h(rowOffset -  7) = rowIdx - ny*nx - nx;
            columns_view_h(rowOffset -  6) = rowIdx - ny*nx - 1;
            columns_view_h(rowOffset -  5) = rowIdx - ny*nx;
            columns_view_h(rowOffset -  4) = rowIdx - nx - 1;
            columns_view_h(rowOffset -  3) = rowIdx - nx;
            columns_view_h(rowOffset -  2) = rowIdx - 1;
            columns_view_h(rowOffset -  1) = rowIdx;

            // Fill values
            values_view_h(rowOffset -  8) = -1.0;
            values_view_h(rowOffset -  7) = -1.0;
            values_view_h(rowOffset -  6) = -1.0;
            values_view_h(rowOffset -  5) =  0.0;
            values_view_h(rowOffset -  4) = -1.0;
            values_view_h(rowOffset -  3) =  0.0;
            values_view_h(rowOffset -  2) =  0.0;
            values_view_h(rowOffset -  1) =  4.0;
          }
        }
      }
    }

    Kokkos::deep_copy(rowmap_view,  rowmap_view_h);
    Kokkos::deep_copy(columns_view, columns_view_h);
    Kokkos::deep_copy(values_view,  values_view_h);

    graph_t static_graph (columns_view, rowmap_view);
    std::string name;
    if(stencil_type == FD) {
      name = "CrsMatrixFD";
    } else if(stencil_type == FE) {
      name = "CrsMatrixFE";
    }

    return CrsMatrix_t(name, numNodes, values_view, static_graph);

  } // generate_structured_matrix3D

}

#endif // KOKKOSKERNELS_TEST_STRUCTURE_MATRIX_HPP
