/*
// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER
*/

#include "Tpetra_TestingUtilities.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Experimental_StructuredCrsWrapper.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Details_getNumDiags.hpp"
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


namespace { // (anonymous)

  using Tpetra::TestingUtilities::getDefaultComm;
  using Tpetra::createContigMapWithNode;
  using Tpetra::createNonContigMapWithNode;
  using Teuchos::RCP;
  using Teuchos::ArrayRCP;
  using Teuchos::rcp;
  using Teuchos::outArg;
  using Teuchos::Comm;
  using Teuchos::Array;
  using Teuchos::ArrayView;
  using Teuchos::tuple;
  using Teuchos::NO_TRANS;
  //using Teuchos::TRANS;
  using Teuchos::CONJ_TRANS;
  using std::endl;
  typedef Tpetra::global_size_t GST;

#define STD_TESTS(matrix) \
  { \
    using Teuchos::outArg; \
    RCP<const Comm<int> > STCOMM = matrix.getComm(); \
    ArrayView<const GO> STMYGIDS = matrix.getRowMap()->getNodeElementList(); \
    ArrayView<const LO> loview; \
    ArrayView<const Scalar> sview; \
    size_t STMAX = 0; \
    for (size_t STR=0; STR < matrix.getNodeNumRows(); ++STR) { \
      const size_t numEntries = matrix.getNumEntriesInLocalRow(STR); \
      TEST_EQUALITY( numEntries, matrix.getNumEntriesInGlobalRow( STMYGIDS[STR] ) ); \
      matrix.getLocalRowView(STR,loview,sview); \
      TEST_EQUALITY( static_cast<size_t>(loview.size()), numEntries ); \
      TEST_EQUALITY( static_cast<size_t>( sview.size()), numEntries ); \
      STMAX = std::max( STMAX, numEntries ); \
    } \
    TEST_EQUALITY( matrix.getNodeMaxNumRowEntries(), STMAX ); \
    GST STGMAX; \
    Teuchos::reduceAll<int,GST>( *STCOMM, Teuchos::REDUCE_MAX, STMAX, outArg(STGMAX) ); \
    TEST_EQUALITY( matrix.getGlobalMaxNumRowEntries(), STGMAX ); \
  }

  //
  // UNIT TESTS
  //

  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL( StructuredCrsWrapper, 1D_1rhs, LO, GO, Scalar, Node )
  {
    typedef Tpetra::CrsMatrix<Scalar,LO,GO,Node> MAT;
    typedef Tpetra::Experimental::StructuredCrsWrapper<Scalar,LO,GO,Node> SMAT;
    typedef Teuchos::ScalarTraits<Scalar> ST;
    typedef Tpetra::MultiVector<Scalar,LO,GO,Node> MV;
    typedef typename ST::magnitudeType Mag;
    typedef Teuchos::ScalarTraits<Mag> MT;
    const GST INVALID = Teuchos::OrdinalTraits<GST>::invalid();

    // get a comm
    RCP<const Comm<int> > comm = getDefaultComm();
    int myRank  = comm->getRank();
    int numProc = comm->getSize();
    // create a Map
    const size_t numLocal = 10;
    const size_t numVecs  = 1;
    RCP<const Tpetra::Map<LO,GO,Node> > map =
      createContigMapWithNode<LO,GO,Node>(INVALID,numLocal,comm);
    // create a 1D matrix
    MAT Crs(map,3);
    Scalar one = Teuchos::ScalarTraits<Scalar>::one();

    for(size_t i = 0; i < numLocal; i++) {
      GO row = map->getGlobalElement(i);
      GO cols[3];
      Scalar vals[3];
      int idx=0;
      if(i == 0 && myRank == 0) {
        idx = 2;
        cols[0] = row;
        vals[0] = one;
        cols[1] = row + 1;
        vals[1] = -one;
      } else if(i==numLocal-1 && myRank == numProc-1) {
        idx = 2;
        cols[0] = row - 1;
        vals[0] = -one;
        cols[1] = row;
        vals[1] = one;
      } else {
        idx = 3;
        cols[0] = row - 1;
        vals[0] = -one;
        cols[1] = row;
        vals[1] = 2*one;
        cols[2] = row + 1;
        vals[2] = -one;
      }

      Crs.insertGlobalValues(row,idx,vals,cols);
    }

    Crs.fillComplete();
    RCP<MAT> RCPcrs(&Crs,false);

    // Wrap the matrix for structured spmv
    Teuchos::ParameterList params;
    params.set("dimension",1);
    params.set("stencil type","FD");
    Teuchos::Array<LO> ppd(1),boundary_lo(1),boundary_hi(1);
    ppd[0] = numLocal;
    boundary_lo[0] = (numProc == 1 || myRank == 0) ? 1 : 0;
    boundary_hi[0] = (numProc == 1 || myRank != numProc-1) ? 1 : 0;
    params.set("points per dimension", ppd);
    params.set("low boundary",boundary_lo);
    params.set("high boundary",boundary_hi);

    SMAT structured(RCPcrs,params);

    // Compare output for an input matrix of ones
    MV mvone(map,numVecs,false), mvres1(map,numVecs,false), mvres2(map,numVecs,false);
    mvone.putScalar(one);

    Crs.apply(mvone,mvres1);
    structured.apply(mvone,mvres2);

    Array<Mag> norms1(numVecs),norms2(numVecs);
    mvres1.norm1(norms1());
    mvres2.norm1(norms2());
    if (ST::isOrdinal) {
      TEST_COMPARE_ARRAYS(norms1,norms2);
    } else {
      TEST_COMPARE_FLOATING_ARRAYS(norms1,norms2,MT::zero());
    }
  }






  template<class Scalar, class LO, class GO, class Node>
  bool test_2d_serial(Teuchos::ParameterList & params, int numVecs) {
    typedef Tpetra::CrsMatrix<Scalar,LO,GO,Node> MAT;
    typedef Tpetra::Experimental::StructuredCrsWrapper<Scalar,LO,GO,Node> SMAT;
    typedef Teuchos::ScalarTraits<Scalar> ST;
    typedef Tpetra::MultiVector<Scalar,LO,GO,Node> MV;
    typedef typename ST::magnitudeType Mag;
    typedef Tpetra::Experimental::StructuredCrsWrapper<Scalar,LO,GO,Node> SMAT;
    Scalar one = Teuchos::ScalarTraits<Scalar>::one();

    RCP<const Comm<int> > comm = getDefaultComm();
    const int numRanks = comm->getSize();
    const int myRank   = comm->getRank();

    bool success = true;

    // Get global params
    std::string discretization_stencil =  params.get("stencil type","FD");
    Teuchos::Array<LO> points_per_dim;
    points_per_dim = params.get("points per dimension", points_per_dim);
    Teuchos::Array<LO> boundary_low, boundary_high;
    boundary_low  = params.get("low boundary", boundary_low);
    boundary_high = params.get("high boundary", boundary_high);
    const GST numGblRows = static_cast<GST>(points_per_dim[1]*points_per_dim[0]);

    // Compute local params based on number of ranks
    Array<LO> lclNumPointsPerDim(2);
    if(numRanks == 1) {
      lclNumPointsPerDim[0] = points_per_dim[0];
      lclNumPointsPerDim[1] = points_per_dim[1];
    } else if(numRanks == 4) {
      lclNumPointsPerDim[0] = points_per_dim[0] / 2;
      lclNumPointsPerDim[1] = points_per_dim[1] / 2;
    }
    const size_t numLclRows = static_cast<size_t>(lclNumPointsPerDim[1]*lclNumPointsPerDim[0]);

    RCP<const Tpetra::Map<LO,GO,Node> > rowMap = createContigMapWithNode<LO,GO,Node>(numGblRows,
                                                                                     numLclRows,
                                                                                     comm);

    const GO nodeOffset = myRank*numLclRows;
    int leftBC = 0, rightBC = 0, bottomBC = 0, topBC = 0;
    int leftGhost = 0, rightGhost = 0, bottomGhost = 0, topGhost = 0;
    if(numRanks == 1) {
      leftBC = 1;
      rightBC = 1;
      bottomBC = 1;
      topBC = 1;
    } else if(numRanks == 4) {
      if(myRank == 0) {
        leftBC = 1;
        bottomBC = 1;

        rightGhost = 1;
        topGhost = 1;
      } else if(myRank == 1) {
        rightBC = 1;
        bottomBC = 1;

        leftGhost = 1;
        topGhost = 1;
      } else if(myRank == 2) {
        leftBC = 1;
        topBC = 1;

        rightGhost = 1;
        bottomGhost = 1;
      } else if(myRank == 3) {
        rightBC = 1;
        topBC = 1;

        leftGhost = 1;
        bottomGhost = 1;
      }
    }

    size_t numLclCols = numLclRows
      + static_cast<size_t>(lclNumPointsPerDim[0]) + static_cast<size_t>(lclNumPointsPerDim[1]);
    Array<GO> colMapGIDs(numLclCols);
    Teuchos::ArrayRCP<size_t> numEntriesPerRow(numLclRows);
    LO countNodes = 0;
    if(bottomGhost == 1) {
      for(LO idx = 0; idx < lclNumPointsPerDim[0]; ++idx) {
        colMapGIDs[countNodes] = static_cast<GO>(nodeOffset - numLclRows - lclNumPointsPerDim[0] + idx);
        ++countNodes;
      }
    }
    for(LO j = 0; j < lclNumPointsPerDim[1]; ++j) {
      if(leftGhost == 1) {
        colMapGIDs[countNodes] = static_cast<GO>(nodeOffset - numLclRows + (j + 1)*lclNumPointsPerDim[0] - 1);
        ++countNodes;
      }
      for(LO i = 0; i < lclNumPointsPerDim[0]; ++i) {
        colMapGIDs[countNodes] = static_cast<GO>(nodeOffset + j*lclNumPointsPerDim[0] + i);
        ++countNodes;

        // Fill nnz per row
        numEntriesPerRow[j*lclNumPointsPerDim[0] + i] = 5;
        if((i == 0 && leftBC == 1) || (i == lclNumPointsPerDim[0] - 1 && rightBC == 1)) {
          numEntriesPerRow[j*lclNumPointsPerDim[0] + i] -= 1;
        }
        if((j == 0 && bottomBC == 1) || (j == lclNumPointsPerDim[1] - 1 && topBC == 1)) {
          numEntriesPerRow[j*lclNumPointsPerDim[0] + i] -= 1;
        }
      }
      if(rightGhost == 1) {
        colMapGIDs[countNodes] = static_cast<GO>(nodeOffset + numLclRows + j*lclNumPointsPerDim[0]);
        ++countNodes;
      }
    }
    if(topGhost == 1) {
      for(LO idx = 0; idx < lclNumPointsPerDim[0]; ++idx) {
        colMapGIDs[countNodes] = static_cast<GO>(nodeOffset + 2*numLclRows + idx);
        ++countNodes;
      }
    }

    RCP<const Tpetra::Map<LO,GO,Node> > colMap = createNonContigMapWithNode<LO,GO,Node>(colMapGIDs(),
                                                                                        comm);

    RCP<MAT> Tcrs = rcp(new MAT(rowMap, colMap, numEntriesPerRow));

    LO rowIdx, numRowEntries, numNodesPerRow = lclNumPointsPerDim[0], bottomOffset = 0;
    if(leftGhost == 1)   {++numNodesPerRow;}
    if(rightGhost == 1)  {++numNodesPerRow;}
    if(bottomGhost == 1) {bottomOffset = lclNumPointsPerDim[0];}
    Array<LO> colIndices(5);
    Array<Scalar> values(5);
    for(LO j = 0; j < lclNumPointsPerDim[1]; ++j) {
      for(LO i = 0; i < lclNumPointsPerDim[0]; ++i) {
        rowIdx = j*lclNumPointsPerDim[0] + i;
        numRowEntries = numEntriesPerRow[rowIdx];

        if((i == 0 && leftBC == 1) && (j == 0 && bottomBC == 1)) { // bottom left corner
          colIndices[0] = 0 + bottomOffset;
          colIndices[1] = 1 + bottomOffset;
          colIndices[2] = numNodesPerRow + bottomOffset;

          values[0] = 2.0;
          values[1] = -1.0;
          values[2] = -1.0;
        } else if((i == 0 && leftBC == 1) && (j == lclNumPointsPerDim[1] - 1 && topBC == 1)) { // top left corner
          colIndices[0] = (lclNumPointsPerDim[1] - 2)*numNodesPerRow + bottomOffset;
          colIndices[1] = (lclNumPointsPerDim[1] - 1)*numNodesPerRow + bottomOffset;
          colIndices[2] = (lclNumPointsPerDim[1] - 1)*numNodesPerRow + 1 + bottomOffset;

          values[0] = -1.0;
          values[1] = 2.0;
          values[2] = -1.0;
        } else if((i == lclNumPointsPerDim[0] - 1 && rightBC == 1) && (j == 0 && bottomBC == 1)) { // bottom right corner
          colIndices[0] = numNodesPerRow - 2 + bottomOffset;
          colIndices[1] = numNodesPerRow - 1 + bottomOffset;
          colIndices[2] = 2*numNodesPerRow - 1 + bottomOffset;

          values[0] = -1.0;
          values[1] = 2.0;
          values[2] = -1.0;
        } else if((i == lclNumPointsPerDim[0] - 1 && rightBC == 1) && (j == lclNumPointsPerDim[1] - 1 && topBC == 1)) { // top right corner
          colIndices[0] = (lclNumPointsPerDim[1] - 1)*numNodesPerRow - 1 + bottomOffset;
          colIndices[1] = lclNumPointsPerDim[1]*numNodesPerRow - 2 + bottomOffset;
          colIndices[2] = lclNumPointsPerDim[1]*numNodesPerRow - 1 + bottomOffset;

          values[0] = -1.0;
          values[1] = -1.0;
          values[2] = 2.0;
        } else if(i == 0 && leftBC == 1) { // left edge
          if(j == 0 && bottomOffset > 0) {
            colIndices[0] = (j - 1)*bottomOffset + i + bottomOffset + leftGhost;
          } else {
            colIndices[0] = (j - 1)*numNodesPerRow + i + bottomOffset + leftGhost;
          }
          colIndices[1] = j*numNodesPerRow + i + bottomOffset + leftGhost;
          colIndices[2] = j*numNodesPerRow + i + 1 + bottomOffset + leftGhost;
          colIndices[3] = (j + 1)*numNodesPerRow + i + bottomOffset + leftGhost;

          values[0] = -1.0;
          values[1] = 3.0;
          values[2] = -1.0;
          values[3] = -1.0;
        } else if(i == lclNumPointsPerDim[0] - 1 && rightBC == 1) { // right edge
          colIndices[0] = (j - 1)*numNodesPerRow + i + bottomOffset + leftGhost;
          colIndices[1] = j*numNodesPerRow + i - 1 + bottomOffset + leftGhost;
          colIndices[2] = j*numNodesPerRow + i + bottomOffset + leftGhost;
          if(j == lclNumPointsPerDim[1] - 1 && topGhost) {
            colIndices[3] = (j + 1)*numNodesPerRow + i + bottomOffset;
          } else {
            colIndices[3] = (j + 1)*numNodesPerRow + i + bottomOffset + leftGhost;
          }

          values[0] = -1.0;
          values[1] = -1.0;
          values[2] = 3.0;
          values[3] = -1.0;
        } else if(j == 0 && bottomBC == 1) { // bottom edge
          colIndices[0] = i - 1 + bottomOffset + leftGhost;
          colIndices[1] = i + bottomOffset + leftGhost;
          colIndices[2] = i + 1 + bottomOffset + leftGhost;
          colIndices[3] = numNodesPerRow + i + bottomOffset + leftGhost;

          values[0] = -1.0;
          values[1] = 3.0;
          values[2] = -1.0;
          values[3] = -1.0;
        } else if(j == lclNumPointsPerDim[1] - 1 && topBC == 1) { // top edge
          colIndices[0] = (j - 1)*numNodesPerRow + i + bottomOffset + leftGhost;
          colIndices[1] = j*numNodesPerRow + i - 1 + bottomOffset + leftGhost;
          colIndices[2] = j*numNodesPerRow + i + bottomOffset + leftGhost;
          colIndices[3] = j*numNodesPerRow + i + 1 + bottomOffset + leftGhost;

          values[0] = -1.0;
          values[1] = -1.0;
          values[2] = 3.0;
          values[3] = -1.0;
        } else {
          if(j == 0 && bottomGhost) {
            colIndices[0] = j*numNodesPerRow + i;
          } else {
            colIndices[0] = (j - 1)*numNodesPerRow + i + bottomOffset + leftGhost;
          }
          colIndices[1] = j*numNodesPerRow + i - 1 + bottomOffset + leftGhost;
          colIndices[2] = j*numNodesPerRow + i + bottomOffset + leftGhost;
          colIndices[3] = j*numNodesPerRow + i + 1 + bottomOffset + leftGhost;
          if(j == lclNumPointsPerDim[1] - 1 && topGhost) {
            colIndices[4] = (j + 1)*numNodesPerRow + i + bottomOffset;
          } else {
            colIndices[4] = (j + 1)*numNodesPerRow + i + bottomOffset + leftGhost;
          }

          values[0] = -1.0;
          values[1] = -1.0;
          values[2] = 4.0;
          values[3] = -1.0;
          values[4] = -1.0;
        }

        Tcrs->insertLocalValues(rowIdx, colIndices(0, numRowEntries), values(0, numRowEntries));
      }
    }
    Tcrs->fillComplete();

    // Now do a structured wrap, we need to reset
    // the params here because of the parallel
    // decomposition of the computational domain
    Teuchos::Array<LO> ppd(2), lo(2), hi(2);
    ppd[0] = lclNumPointsPerDim[0]; ppd[1] = lclNumPointsPerDim[1];
    lo[0] = 1; lo[1] = 1;
    hi[0] =1; hi[1] = 1;
    params.set("points per dimension",ppd);
    params.set("low boundary",lo);
    params.set("high boundary",hi);
    SMAT Tstruct(Tcrs, params);

    // Compare output for an input multivector of ones
    MV mvone(rowMap, numVecs, false), mvres1(rowMap, numVecs, false), mvres2(rowMap, numVecs, false);
    mvone.putScalar(one);

    Tcrs->apply(mvone,mvres1);
    Tstruct.apply(mvone,mvres2);

    Array<Mag> norms1(numVecs), norms2(numVecs);
    mvres1.norm1(norms1());
    mvres2.norm1(norms2());
    Scalar diff = 0.0;
    for (int i = 0; i<numVecs; i++)
      diff += std::abs(norms1[i] - norms2[i]);

    if(diff > 10*ST::eps()) success=false;


    return success;
  }




  TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL( StructuredCrsWrapper, 2D_fd_serial_1rhs, LO, GO, Scalar, Node )
  {
    RCP<const Comm<int> > comm = getDefaultComm();
    const int numProc = comm->getSize();
    if((numProc != 1) && (numProc != 4)) {
      TEST_EQUALITY_CONST( 1, 1 );
    } else {
      // Set the global parameters for the problem
      Teuchos::ParameterList params;
      params.set("stencil type","FD");
      params.set("dimension",2);
      Teuchos::Array<LO> ppd(2), lo(2), hi(2);
      ppd[0] = 6; ppd[1] = 10;
      lo[0] = 1; lo[1] = 1;
      hi[0] = 1; hi[1] = 1;

      params.set("points per dimension",ppd);
      params.set("low boundary",lo);
      params.set("high boundary",hi);

      bool rv = test_2d_serial<Scalar,LO,GO,Node>(params,1);
      TEST_EQUALITY_CONST( rv, true );
    }
   }


#define UNIT_TEST_GROUP( SCALAR, LO, GO, NODE ) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT( StructuredCrsWrapper, 1D_1rhs,     LO, GO, SCALAR, NODE ) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT( StructuredCrsWrapper, 2D_fd_serial_1rhs,     LO, GO, SCALAR, NODE )


  TPETRA_ETI_MANGLING_TYPEDEFS()

  TPETRA_INSTANTIATE_SLGN_NO_ORDINAL_SCALAR( UNIT_TEST_GROUP )

} // namespace (anonymous)
