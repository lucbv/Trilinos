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
      stencil_type = 1;
    } else if (stencil == "FE") {
      stencil_type = 2;
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

    if(stencil_type == 1) {
      interiorStencilLength = 5;
      edgeStencilLength     = 4;
      cornerStencilLength   = 3;
    } else if(stencil_type == 2) {
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

      if(stencil_type == 1) {
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
      } else if(stencil_type == 2) {
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

    // Loop over edge points
    for(ordinal_type idx = 0; idx < nx - leftBC - rightBC; ++idx) {

      if(bottomBC == 1) {
        /***************/
        /* Bottom edge */
        /***************/
        rowIdx    = idx + leftBC;
        rowOffset = (idx + 1)*edgeStencilLength + leftBC*cornerStencilLength;
        rowmap_view_h(rowIdx + 1) = rowOffset;
        if(stencil_type == 1) {
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
      } else if(stencil_type == 2) {
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
        if(stencil_type == 1) {
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
        } else if(stencil_type == 2) {
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

    for(ordinal_type idx = 0; idx < ny - bottomBC - topBC; ++idx) {

      if(leftBC == 1) {
        /*************/
        /* Left edge */
        /*************/
        rowIdx    = (idx + bottomBC)*nx;
        rowOffset = idx*numEntriesPerGridRow + bottomBC*numEntriesBottomRow
          + edgeStencilLength;
        rowmap_view_h(rowIdx + 1) = rowOffset;
        if(stencil_type == 1) {
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
        } else if(stencil_type == 2) {
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
        if(stencil_type == 1) {
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
        } else if(stencil_type == 2) {
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

      if(stencil_type == 1) {
        columns_view_h(rowOffset - 3) = rowIdx;
        columns_view_h(rowOffset - 2) = rowIdx + 1;
        columns_view_h(rowOffset - 1) = rowIdx + nx;

        values_view_h(rowOffset - 3)  =  2.0;
        values_view_h(rowOffset - 2)  = -1.0;
        values_view_h(rowOffset - 1)  = -1.0;
      } else if(stencil_type == 2) {
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

      if(stencil_type == 1) {
        columns_view_h(rowOffset - 3) = rowIdx - 1;
        columns_view_h(rowOffset - 2) = rowIdx;
        columns_view_h(rowOffset - 1) = rowIdx + nx;

        values_view_h(rowOffset - 3)  = -1.0;
        values_view_h(rowOffset - 2)  =  2.0;
        values_view_h(rowOffset - 1)  = -1.0;
      } else if(stencil_type == 2) {
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

      if(stencil_type == 1) {
        columns_view_h(rowOffset - 3) = rowIdx - nx;
        columns_view_h(rowOffset - 2) = rowIdx;
        columns_view_h(rowOffset - 1) = rowIdx + 1;

        values_view_h(rowOffset - 3)  = -1.0;
        values_view_h(rowOffset - 2)  =  2.0;
        values_view_h(rowOffset - 1)  = -1.0;
      } else if(stencil_type == 2) {
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

      if(stencil_type == 1) {
        columns_view_h(rowOffset - 3) = rowIdx - nx;
        columns_view_h(rowOffset - 2) = rowIdx - 1;
        columns_view_h(rowOffset - 1) = rowIdx;

        values_view_h(rowOffset - 3)  = -1.0;
        values_view_h(rowOffset - 2)  = -1.0;
        values_view_h(rowOffset - 1)  =  2.0;
      } else if(stencil_type == 2) {
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
    if(stencil_type == 1) {
      name = "CrsMatrixFD";
    } else if(stencil_type == 2) {
      name = "CrsMatrixFEM";
    }

    return CrsMatrix_t(name, numNodes, values_view, static_graph);

  }

}

#endif // KOKKOSKERNELS_TEST_STRUCTURE_MATRIX_HPP
