/*
//@HEADER
// ***********************************************************************
//
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//                 Copyright (2009) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// ***********************************************************************
//@HEADER
*/


/*! \file Ifpack2_UnitTestSingleProcessMDF.cpp

\brief Ifpack2 single-process unit tests for the MDF class.
*/

#include "Teuchos_ConfigDefs.hpp"
#include "Ifpack2_ConfigDefs.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <iostream>

#include "Tpetra_Core.hpp"
// #include "Tpetra_MatrixIO.hpp"
// #include "MatrixMarket_Tpetra.hpp"
// #include "TpetraExt_MatrixMatrix.hpp"

#include "Ifpack2_UnitTestHelpers.hpp"
#include "Ifpack2_MDF.hpp"

namespace { // (anonymous)

using Tpetra::global_size_t;
typedef tif_utest::Node Node;

struct IlukImplTypeDetails {
  enum Enum { Serial, KSPILUK };
};

// template<class MatrixType, class VectorType>
// void remove_diags_and_scale(const MatrixType& L, const MatrixType& U,
//                             Teuchos::RCP<MatrixType>& Ln, Teuchos::RCP<MatrixType>& Un, Teuchos::RCP<VectorType>& Dn) {

//   typedef typename MatrixType::local_matrix_device_type local_matrix_type;
//   typedef typename std::remove_const<typename local_matrix_type::size_type>::type    size_type;
//   typedef typename std::remove_const<typename local_matrix_type::ordinal_type>::type ordinal_type;
//   typedef typename std::remove_const<typename local_matrix_type::value_type>::type   value_type;
//   typedef typename local_matrix_type::device_type device_type;
//   typedef typename device_type::execution_space   execution_space;

//   typedef typename Kokkos::View<size_type*, Kokkos::LayoutLeft, device_type> rowmap_type;
//   typedef typename Kokkos::View<ordinal_type*, Kokkos::LayoutLeft, device_type> entries_type;
//   typedef typename Kokkos::View<value_type*, Kokkos::LayoutRight, device_type> values_type;

//   typedef Kokkos::TeamPolicy<execution_space> team_policy;
//   typedef typename Kokkos::TeamPolicy<execution_space>::member_type member_type;

//   auto L_rowmap  = L.getLocalMatrixDevice().graph.row_map;
//   auto L_entries = L.getLocalMatrixDevice().graph.entries;
//   auto L_values  = L.getLocalValuesView();
//   auto U_rowmap  = U.getLocalMatrixDevice().graph.row_map;
//   auto U_entries = U.getLocalMatrixDevice().graph.entries;
//   auto U_values  = U.getLocalValuesView();

//   rowmap_type  Ln_rowmap ("Ln_rowmap",  L_rowmap.extent(0));
//   entries_type Ln_entries("Ln_entries", L_entries.extent(0) - (L_rowmap.extent(0) - 1));
//   values_type  Ln_values ("Ln_values",  L_values.extent(0)  - (L_rowmap.extent(0) - 1));
//   rowmap_type  Un_rowmap ("Un_rowmap",  U_rowmap.extent(0));
//   entries_type Un_entries("Un_entries", U_entries.extent(0) - (U_rowmap.extent(0) - 1));
//   values_type  Un_values ("Un_values",  U_values.extent(0)  - (U_rowmap.extent(0) - 1));

//   values_type  Dn_values ("Dn_values",  U_rowmap.extent(0) - 1);

//   Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, U_rowmap.extent(0)-1),
//                        KOKKOS_LAMBDA(const int& i) {
//                          Ln_rowmap(i+1) = L_rowmap(i+1) - (i+1);
//                          Un_rowmap(i+1) = U_rowmap(i+1) - (i+1);
//                          Dn_values(i)   = 1.0/U_values(U_rowmap(i));
//                        });

//   const team_policy policy( U_rowmap.extent(0)-1, Kokkos::AUTO );

//   Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const member_type& teamMember) {
//       const int rowid = teamMember.league_rank();

//       auto Lentries_src = subview(L_entries,  Kokkos::make_pair(L_rowmap(rowid),  L_rowmap(rowid+1) - 1));
//       auto Lvalues_src  = subview(L_values,   Kokkos::make_pair(L_rowmap(rowid),  L_rowmap(rowid+1) - 1));
//       auto Lentries_dst = subview(Ln_entries, Kokkos::make_pair(Ln_rowmap(rowid), Ln_rowmap(rowid+1)));
//       auto Lvalues_dst  = subview(Ln_values,  Kokkos::make_pair(Ln_rowmap(rowid), Ln_rowmap(rowid+1)));

//       auto Uentries_src = subview(U_entries,  Kokkos::make_pair(U_rowmap(rowid)+1, U_rowmap(rowid+1)));
//       auto Uvalues_src  = subview(U_values,   Kokkos::make_pair(U_rowmap(rowid)+1, U_rowmap(rowid+1)));
//       auto Uentries_dst = subview(Un_entries, Kokkos::make_pair(Un_rowmap(rowid),  Un_rowmap(rowid+1)));
//       auto Uvalues_dst  = subview(Un_values,  Kokkos::make_pair(Un_rowmap(rowid),  Un_rowmap(rowid+1)));

//       Kokkos::Experimental::local_deep_copy(teamMember, Lentries_dst, Lentries_src);
//       Kokkos::Experimental::local_deep_copy(teamMember, Uentries_dst, Uentries_src);
//       Kokkos::Experimental::local_deep_copy(teamMember, Lvalues_dst, Lvalues_src);
//       Kokkos::Experimental::local_deep_copy(teamMember, Uvalues_dst, Uvalues_src);

//       teamMember.team_barrier();

//       Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, Uvalues_dst.extent(0)), [&](const int& i) {
//           Uvalues_dst(i) = Uvalues_dst(i)*Dn_values(rowid);
//         });

//       teamMember.team_barrier();
//     });

//   Ln = Teuchos::rcp (new MatrixType (L.getRowMap(), L.getColMap(),
//                                      Ln_rowmap, Ln_entries, Ln_values));
//   Un = Teuchos::rcp (new MatrixType (U.getRowMap(), U.getColMap(),
//                                      Un_rowmap, Un_entries, Un_values));
//   auto Dn_view = Dn->getLocalViewDevice(Tpetra::Access::OverwriteAll);
//   Kokkos::deep_copy(subview(Dn_view,Kokkos::ALL(), 0),Dn_values);

//   Ln->fillComplete();
//   Un->fillComplete();
// }

// template<typename Scalar, typename LO, typename GO>
// void Ifpack2SingleProcessMDF_test0 (bool& success, Teuchos::FancyOStream& out, const IlukImplTypeDetails::Enum ilukimplType) {
//   using Teuchos::RCP;
//   using std::endl;

//   if (ilukimplType == IlukImplTypeDetails::Serial)
//     out << "Ifpack2::RILUK: Test0 -- Serial" << endl;
//   else
//     out << "Ifpack2::RILUK: Test0 -- Kokkos Kernels SPILUK" << endl;

//   Teuchos::OSTab tab0 (out);

//   global_size_t num_rows_per_proc = 5;
//   const RCP<const Tpetra::Map<LO,GO,Node> > rowmap =
//     tif_utest::create_tpetra_map<LO,GO,Node>(num_rows_per_proc);

//   if (rowmap->getComm()->getSize() > 1) {
//     out << endl << "This test may only be run in serial." << endl;
//     return;
//   }

//   RCP<const Tpetra::CrsMatrix<Scalar,LO,GO,Node> > crsmatrix =
//     tif_utest::create_test_matrix<Scalar,LO,GO,Node> (rowmap);
  
//   //----------------Default trisolver----------------//
//   {
//     Ifpack2::MDF<Tpetra::RowMatrix<Scalar,LO,GO,Node> > prec (crsmatrix);

//     Teuchos::ParameterList params;
//     const int fill_level = 0;
//     params.set("fact: mdf level-of-fill", fill_level);
//     params.set("fact: mdf level-of-overlap", 0);

//     if (ilukimplType == IlukImplTypeDetails::KSPILUK)
//       params.set("fact: type", "KSPILUK");

//     TEST_NOTHROW(prec.setParameters(params));

//     TEST_EQUALITY( prec.getLevelOfFill(), fill_level);

//     prec.initialize();
//     //trivial tests to insist that the preconditioner's domain/range maps are
//     //the same as those of the matrix:
//     const Tpetra::Map<LO,GO,Node>& mtx_dom_map = *crsmatrix->getDomainMap();
//     const Tpetra::Map<LO,GO,Node>& mtx_rng_map = *crsmatrix->getRangeMap();

//     const Tpetra::Map<LO,GO,Node>& prec_dom_map = *prec.getDomainMap();
//     const Tpetra::Map<LO,GO,Node>& prec_rng_map = *prec.getRangeMap();

//     TEST_ASSERT( prec_dom_map.isSameAs(mtx_dom_map) );
//     TEST_ASSERT( prec_rng_map.isSameAs(mtx_rng_map) );

//     prec.compute();

//     Tpetra::MultiVector<Scalar,LO,GO,Node> x(rowmap,2), y(rowmap,2);
//     x.putScalar (Teuchos::ScalarTraits<Scalar>::one ());

//     prec.apply(x, y);

//     Teuchos::ArrayRCP<const Scalar> yview = y.get1dView();

//     //y should be full of 0.5's now.

//     Teuchos::ArrayRCP<Scalar> halfs(num_rows_per_proc*2, 0.5);

//     TEST_COMPARE_FLOATING_ARRAYS(yview, halfs(), Teuchos::ScalarTraits<Scalar>::eps());
//   }
//   //----------------Kokkos Kernels SPTRSV----------------//
//   {
//     Ifpack2::MDF<Tpetra::RowMatrix<Scalar,LO,GO,Node> > prec (crsmatrix);

//     Teuchos::ParameterList params;
//     const int fill_level = 0;
//     params.set("fact: mdf level-of-fill", fill_level);
//     params.set("fact: mdf level-of-overlap", 0);

//     if (ilukimplType == IlukImplTypeDetails::KSPILUK)
//       params.set("fact: type", "KSPILUK");

//     params.set("trisolver: type", "KSPTRSV");

//     TEST_NOTHROW(prec.setParameters(params));

//     TEST_EQUALITY( prec.getLevelOfFill(), fill_level);

//     prec.initialize();
//     //trivial tests to insist that the preconditioner's domain/range maps are
//     //the same as those of the matrix:
//     const Tpetra::Map<LO,GO,Node>& mtx_dom_map = *crsmatrix->getDomainMap();
//     const Tpetra::Map<LO,GO,Node>& mtx_rng_map = *crsmatrix->getRangeMap();

//     const Tpetra::Map<LO,GO,Node>& prec_dom_map = *prec.getDomainMap();
//     const Tpetra::Map<LO,GO,Node>& prec_rng_map = *prec.getRangeMap();

//     TEST_ASSERT( prec_dom_map.isSameAs(mtx_dom_map) );
//     TEST_ASSERT( prec_rng_map.isSameAs(mtx_rng_map) );

//     prec.compute();

//     Tpetra::MultiVector<Scalar,LO,GO,Node> x(rowmap,2), y(rowmap,2);
//     x.putScalar (Teuchos::ScalarTraits<Scalar>::one ());

//     prec.apply(x, y);

//     Teuchos::ArrayRCP<const Scalar> yview = y.get1dView();

//     //y should be full of 0.5's now.

//     Teuchos::ArrayRCP<Scalar> halfs(num_rows_per_proc*2, 0.5);

//     TEST_COMPARE_FLOATING_ARRAYS(yview, halfs(), Teuchos::ScalarTraits<Scalar>::eps());
//   }
// }

// template<typename Scalar, typename LO, typename GO>
// void Ifpack2SingleProcessMDF_test1 (bool& success, Teuchos::FancyOStream& out, const IlukImplTypeDetails::Enum ilukimplType) {
//   using Kokkos::Details::ArithTraits;
//   using Teuchos::RCP;
//   using std::endl;
//   typedef Tpetra::Map<LO, GO, Node> map_type;
//   typedef Tpetra::CrsMatrix<Scalar,LO,GO,Node> crs_matrix_type;
//   typedef Tpetra::MultiVector<Scalar,LO,GO,Node> MV;
//   typedef Tpetra::RowMatrix<Scalar,LO,GO,Node> row_matrix_type;
//   typedef Teuchos::ScalarTraits<Scalar> STS;
//   typedef typename MV::impl_scalar_type val_type;
//   typedef typename Kokkos::Details::ArithTraits<val_type>::mag_type mag_type;
//   typedef typename map_type::device_type device_type;
//   const mag_type oneMag = ArithTraits<mag_type>::one ();
//   const mag_type twoMag = oneMag + oneMag;

//   if (ilukimplType == IlukImplTypeDetails::Serial)
//     out << "Ifpack2::MDF: Test1 -- Serial" << endl;
//   else
//     out << "Ifpack2::MDF: Test1 -- Kokkos Kernels SPILUK" << endl;

//   Teuchos::OSTab tab1 (out);

//   const global_size_t num_rows_per_proc = 5;
//   RCP<const map_type> rowmap =
//     tif_utest::create_tpetra_map<LO, GO, Node> (num_rows_per_proc);

//   // Matrix
//   // [ 2 .1  0  0  0]
//   // [.1  2  0  0  0]
//   // [ 0 .1  2 .1  0]
//   // [ 0  0 .1  2 .1]
//   // [ 0  0  0 .1  2]

//   // Matlab's Factors
//   // L
//   // Diagonal = 1 (implied)
//   // Subdiagonal (approx) = .05, .0501, .0501 .0501

//   // U
//   // Diagonal (approx)      = 2 1.995 1.995 1.995 1.995
//   // Superdiagonal (approx) = .1 .1 .1 .1


//   if (rowmap->getComm ()->getSize () > 1) {
//     out << "This test may only be run in serial "
//       "or with a single MPI process." << endl;
//     return;
//   }

//   out << "Creating matrix" << endl;
//   RCP<const crs_matrix_type> crsmatrix =
//     tif_utest::create_test_matrix2<Scalar,LO,GO,Node>(rowmap);

//   {//CMS
//     auto out = Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cout));
//     *out<<"***** A *****"<<std::endl;
//     crsmatrix->describe(*out,Teuchos::VERB_EXTREME);
//   }

//   //----------------Default trisolver----------------//
//   {
//     out << "Creating preconditioner" << endl;
//     Ifpack2::MDF<row_matrix_type> prec (crsmatrix);

//     out << "Setting preconditioner's parameters" << endl;
//     Teuchos::ParameterList params;
//     params.set ("fact: mdf level-of-fill", 1);
//     params.set ("fact: mdf level-of-overlap", 0);
//     if (ilukimplType == IlukImplTypeDetails::KSPILUK)
//       params.set("fact: type", "KSPILUK");
//     TEST_NOTHROW(prec.setParameters(params));

//     out << "Calling initialize() and compute()" << endl;
//     prec.initialize();
//     prec.compute();

//   {//CMS
//     auto out = Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cout));
//     *out<<"***** Test L *****"<<std::endl;
//     prec.getL().describe(*out,Teuchos::VERB_EXTREME);
//     *out<<"***** Test U *****"<<std::endl;
//     prec.getU().describe(*out,Teuchos::VERB_EXTREME);
//     *out<<"***** Test D *****"<<std::endl;
//     prec.getD().describe(*out,Teuchos::VERB_EXTREME);
//   }


//     out << "Creating test problem" << endl;
//     MV x (rowmap, 2);
//     MV y (rowmap, 2);
//     x.putScalar (STS::one ());

//     out << "Calling crsmatrix->apply(x, y)" << endl;
//     crsmatrix->apply(x,y);

//     out << "Calling prec.apply(y, x)" << endl;
//     //apply the preconditioner to y, putting ~A^-1*y in x
//     //(this should set x back to 1's)
//     prec.apply(y, x);

//     //x should be full of 1's now.
//     out << "Checking result" << endl;

//     // Useful things for comparing results.
//     Kokkos::View<mag_type*, device_type> norms ("norms", x.getNumVectors ());
//     auto norms_h = Kokkos::create_mirror_view (norms);
//     MV diff (x.getMap (), x.getNumVectors ());

//     {
//       // FIXME (mfh 04 Oct 2016) This is the bound that I found here
//       // when I fixed this test.  Not sure if it's sensible.
//       const mag_type bound = twoMag * ArithTraits<val_type>::eps ();

//       diff.putScalar (STS::one ());
//       diff.update (STS::one (), x, -STS::one ());
//       diff.normInf (norms);
//       Kokkos::deep_copy (norms_h, norms);
//       for (LO j = 0; j < static_cast<LO> (norms_h.extent (0)); ++j) {
//         const mag_type absVal = ArithTraits<mag_type>::abs (norms_h(j));
//         TEST_ASSERT( absVal <= bound );
//         if (absVal > bound) {
//           out << "\\| x(:," << j << ") - 1 \\|_{\\infty} = "
//               << absVal << " > " << bound
//               << "; the norm equals " << norms_h(j)
//               << endl;
//         }
//       }
//     }

//     auto test_alpha_beta = [&] (const Scalar& alpha, const Scalar& beta,
//                                 const Teuchos::ETransp& mode) {
//       out << "Testing apply() for alpha = " << alpha
//           << " and beta = " << beta << endl;
//       const Scalar x_magic_number = -0.42;
//       x.putScalar (x_magic_number);
//       crsmatrix->apply (x, y, mode);
//       MV z = Tpetra::createCopy (x);
//       const Scalar z_magic_number = 2.1;
//       z.putScalar (z_magic_number);
//       // z = beta z + alpha inv(A) y
//       //   = beta z + alpha x
//       //   = (beta z_magic_number + alpha x_magic_number) 1s
//       prec.apply(y, z, mode, alpha, beta);

//       MV z_true = Tpetra::createCopy (x);
//       const Scalar z_true_scalar = beta*z_magic_number + alpha*x_magic_number;
//       z_true.putScalar(z_true_scalar);

//       {
//         // FIXME (mfh 04 Oct 2016) This is the bound that I found here
//         // when I fixed this test.  Not sure if it's sensible.
//         const mag_type bound = 10.0 * ArithTraits<val_type>::eps ();

//         diff.putScalar (z_true_scalar);
//         diff.update (STS::one (), z, -STS::one ());
//         diff.normInf (norms);
//         Kokkos::deep_copy (norms_h, norms);

//         for (LO j = 0; j < static_cast<LO> (norms_h.extent (0)); ++j) {
//           const mag_type absVal = ArithTraits<mag_type>::abs (norms_h(j));
//           TEST_ASSERT( absVal <= bound );
//           if (absVal > bound) {
//             out << "\\| x(:," << j << ") - 1 \\|_{\\infty} = "
//                 << absVal << " > " << bound
//                 << "; the norm equals " << norms_h(j)
//                 << endl;
//           }
//         }
//       }
//     };

//     for (const auto mode : {Teuchos::NO_TRANS, Teuchos::TRANS}) {
//       test_alpha_beta(0.0, 0.0, mode);
//       test_alpha_beta(2.0, 0.0, mode);
//       test_alpha_beta(0.0, -1.5, mode);
//       test_alpha_beta(-0.42, 4.2, mode);
//     }
//   }

//   return;//CMS
//   //----------------Kokkos Kernels SPTRSV----------------//
//   {
//     out << "Creating preconditioner" << endl;
//     Ifpack2::MDF<row_matrix_type> prec (crsmatrix);

//     out << "Setting preconditioner's parameters" << endl;
//     Teuchos::ParameterList params;
//     params.set ("fact: mdf level-of-fill", 0);
//     params.set ("fact: mdf level-of-overlap", 0);
//     if (ilukimplType == IlukImplTypeDetails::KSPILUK)
//       params.set("fact: type", "KSPILUK");
//     params.set("trisolver: type", "KSPTRSV");
//     TEST_NOTHROW(prec.setParameters(params));

//     out << "Calling initialize() and compute()" << endl;
//     prec.initialize();
//     prec.compute();

//     out << "Creating test problem" << endl;
//     MV x (rowmap, 2);
//     MV y (rowmap, 2);
//     x.putScalar (STS::one ());

//     out << "Calling crsmatrix->apply(x, y)" << endl;
//     crsmatrix->apply(x,y);

//     out << "Calling prec.apply(y, x)" << endl;
//     //apply the preconditioner to y, putting ~A^-1*y in x
//     //(this should set x back to 1's)
//     prec.apply(y, x);

//     //x should be full of 1's now.
//     out << "Checking result" << endl;

//     // Useful things for comparing results.
//     Kokkos::View<mag_type*, device_type> norms ("norms", x.getNumVectors ());
//     auto norms_h = Kokkos::create_mirror_view (norms);
//     MV diff (x.getMap (), x.getNumVectors ());

//     {
//       // FIXME (mfh 04 Oct 2016) This is the bound that I found here
//       // when I fixed this test.  Not sure if it's sensible.
//       const mag_type bound = twoMag * ArithTraits<val_type>::eps ();

//       diff.putScalar (STS::one ());
//       diff.update (STS::one (), x, -STS::one ());
//       diff.normInf (norms);
//       Kokkos::deep_copy (norms_h, norms);
//       for (LO j = 0; j < static_cast<LO> (norms_h.extent (0)); ++j) {
//         const mag_type absVal = ArithTraits<mag_type>::abs (norms_h(j));
//         TEST_ASSERT( absVal <= bound );
//         if (absVal > bound) {
//           out << "\\| x(:," << j << ") - 1 \\|_{\\infty} = "
//               << absVal << " > " << bound
//               << "; the norm equals " << norms_h(j)
//               << endl;
//         }
//       }
//     }

//     auto test_alpha_beta = [&] (const Scalar& alpha, const Scalar& beta,
//                                 const Teuchos::ETransp& mode) {
//       out << "Testing apply() for alpha = " << alpha
//           << " and beta = " << beta << endl;
//       const Scalar x_magic_number = -0.42;
//       x.putScalar (x_magic_number);
//       crsmatrix->apply (x, y, mode);
//       MV z = Tpetra::createCopy (x);
//       const Scalar z_magic_number = 2.1;
//       z.putScalar (z_magic_number);
//       // z = beta z + alpha inv(A) y
//       //   = beta z + alpha x
//       //   = (beta z_magic_number + alpha x_magic_number) 1s
//       prec.apply(y, z, mode, alpha, beta);

//       MV z_true = Tpetra::createCopy (x);
//       const Scalar z_true_scalar = beta*z_magic_number + alpha*x_magic_number;
//       z_true.putScalar(z_true_scalar);

//       {
//         // FIXME (mfh 04 Oct 2016) This is the bound that I found here
//         // when I fixed this test.  Not sure if it's sensible.
//         const mag_type bound = 10.0 * ArithTraits<val_type>::eps ();

//         diff.putScalar (z_true_scalar);
//         diff.update (STS::one (), z, -STS::one ());
//         diff.normInf (norms);
//         Kokkos::deep_copy (norms_h, norms);

//         for (LO j = 0; j < static_cast<LO> (norms_h.extent (0)); ++j) {
//           const mag_type absVal = ArithTraits<mag_type>::abs (norms_h(j));
//           TEST_ASSERT( absVal <= bound );
//           if (absVal > bound) {
//             out << "\\| x(:," << j << ") - 1 \\|_{\\infty} = "
//                 << absVal << " > " << bound
//                 << "; the norm equals " << norms_h(j)
//                 << endl;
//           }
//         }
//       }
//     };

//     for (const auto mode : {Teuchos::NO_TRANS, Teuchos::TRANS}) {
//       test_alpha_beta(0.0, 0.0, mode);
//       test_alpha_beta(2.0, 0.0, mode);
//       test_alpha_beta(0.0, -1.5, mode);
//       test_alpha_beta(-0.42, 4.2, mode);
//     }
//   }
//   out << "Done with test" << endl;
// }

template <class crs_matrix_type>
struct MDF_discarded_fill_norm {

  using static_crs_graph_type = typename crs_matrix_type::StaticCrsGraphType;
  using col_ind_type          = typename static_crs_graph_type::entries_type::non_const_type;
  using values_type           = typename crs_matrix_type::values_type::non_const_type;
  using size_type             = typename crs_matrix_type::size_type;
  using ordinal_type          = typename crs_matrix_type::ordinal_type;
  using scalar_type           = typename crs_matrix_type::value_type;

  const scalar_type zero = Kokkos::ArithTraits<scalar_type>::zero();

  crs_matrix_type A, At;

  values_type  discarded_fill;
  col_ind_type deficiency;

  MDF_discarded_fill_norm(crs_matrix_type A_, values_type  discarded_fill_,
                          col_ind_type deficiency_)
    : A(A_), At(KokkosKernels::Impl::transpose_matrix<crs_matrix_type>(A_)),
      discarded_fill(discarded_fill_), deficiency(deficiency_) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const ordinal_type rowIdx) const {
    scalar_type discard_norm = zero, diag_val = zero;
    bool entryIsDiscarded = true;
    ordinal_type numFillEntries = 0;
    for(size_type alphaIdx = At.graph.row_map(rowIdx); alphaIdx < At.graph.row_map(rowIdx + 1); ++alphaIdx) {
      ordinal_type fillRowIdx = At.graph.entries(alphaIdx);
      if(fillRowIdx != rowIdx) {
        for(size_type betaIdx = A.graph.row_map(rowIdx); betaIdx < A.graph.row_map(rowIdx + 1); ++betaIdx) {
          ordinal_type fillColIdx = A.graph.entries(betaIdx);
          if(fillColIdx != rowIdx) {
            entryIsDiscarded = true;
            for(size_type entryIdx = A.graph.row_map(fillRowIdx); entryIdx < A.graph.row_map(fillRowIdx + 1); ++entryIdx) {
              if(A.graph.entries(entryIdx) == fillColIdx) {entryIsDiscarded = false;}
            }
            if(entryIsDiscarded) {
              numFillEntries += 1;
              discard_norm += Kokkos::ArithTraits<scalar_type>::abs(At.values(alphaIdx)*A.values(betaIdx))
                *Kokkos::ArithTraits<scalar_type>::abs(At.values(alphaIdx)*A.values(betaIdx));
            }
          }
        }
      } else {
        diag_val = At.values(alphaIdx);
      }
    }

    // TODO add a check on `diag_val == zero`
    discard_norm              = discard_norm / (diag_val*diag_val);
    discarded_fill(rowIdx)    = discard_norm;
    deficiency(rowIdx)        = numFillEntries;
    const ordinal_type degree = ordinal_type(A.graph.row_map(rowIdx + 1) - A.graph.row_map(rowIdx) - 1);
    printf("Row %d has discarded fill of %f, deficiency of %d and degree %d\n", rowIdx, discarded_fill(rowIdx), deficiency(rowIdx), degree);
  }

}; // MDF_discarded_fill_norm

template <class crs_matrix_type>
struct MDF_select_row{

  using values_type  = typename crs_matrix_type::values_type::non_const_type;
  using col_ind_type = typename crs_matrix_type::StaticCrsGraphType::entries_type::non_const_type;
  using row_map_type = typename crs_matrix_type::StaticCrsGraphType::row_map_type;
  using size_type    = typename crs_matrix_type::size_type;
  using ordinal_type = typename crs_matrix_type::ordinal_type;
  using scalar_type  = typename crs_matrix_type::value_type;

  // type used to perform the reduction
  // do not confuse it with scalar_type!
  using value_type = typename crs_matrix_type::ordinal_type;

  values_type  discarded_fill;
  col_ind_type deficiency;
  row_map_type row_map;

  MDF_select_row(values_type  discarded_fill_, col_ind_type deficiency_,
                 row_map_type row_map_)
    : discarded_fill(discarded_fill_), deficiency(deficiency_),
      row_map(row_map_) {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const ordinal_type src, ordinal_type& dst) const {
    const ordinal_type degree_src = row_map(src + 1) - row_map(src) - 1;
    const ordinal_type degree_dst = row_map(dst + 1) - row_map(dst) - 1;

    if(discarded_fill(src) < discarded_fill(dst)) {
      dst = src;
      return;
    }

    if((discarded_fill(src) == discarded_fill(dst))
       && (deficiency(src) < deficiency(dst))) {
      dst = src;
      return;
    }

    if((discarded_fill(src) == discarded_fill(dst))
       && (deficiency(src) == deficiency(dst))
       && (degree_src < degree_dst)) {
      dst = src;
      return;
    }

    if((discarded_fill(src) == discarded_fill(dst))
       && (deficiency(src) == deficiency(dst))
       && (degree_src == degree_dst)
       && (src < dst)) {
      dst = src;
      return;
    }

    return;
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type& dst,
            const volatile value_type& src) const {
    const ordinal_type degree_src = row_map(src + 1) - row_map(src) - 1;
    const ordinal_type degree_dst = row_map(dst + 1) - row_map(dst) - 1;

    if(discarded_fill(src) < discarded_fill(dst)) {
      dst = src;
      return;
    }

    if((discarded_fill(src) == discarded_fill(dst))
       && (deficiency(src) < deficiency(dst))) {
      dst = src;
      return;
    }

    if((discarded_fill(src) == discarded_fill(dst))
       && (deficiency(src) == deficiency(dst))
       && (degree_src < degree_dst)) {
      dst = src;
      return;
    }

    if((discarded_fill(src) == discarded_fill(dst))
       && (deficiency(src) == deficiency(dst))
       && (degree_src == degree_dst)
       && (src < dst)) {
      dst = src;
      return;
    }

    return;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& dst) const {
    dst = Kokkos::ArithTraits<ordinal_type>::zero();
  }

}; // MDF_select_row

template<typename Scalar, typename LO, typename GO>
void Ifpack2SingleProcessMDF_analytical (bool& success, Teuchos::FancyOStream& out,
                                         const IlukImplTypeDetails::Enum ilukimplType) {
  using Teuchos::RCP;
  using crs_matrix_type   = Tpetra::CrsMatrix<Scalar,LO,GO,Node>;
  using local_matrix_type = typename crs_matrix_type::local_matrix_device_type;
  using local_graph_type  = typename local_matrix_type::StaticCrsGraphType;
  using row_map_type      = typename local_graph_type::row_map_type::non_const_type;
  using col_ind_type      = typename local_graph_type::entries_type::non_const_type;
  using values_type       = typename local_matrix_type::values_type::non_const_type;
  using size_type         = typename local_matrix_type::size_type;
  using ordinal_type      = typename local_matrix_type::ordinal_type;
  using value_type        = typename local_matrix_type::value_type;
  using execution_space   = typename local_matrix_type::execution_space;

  Kokkos::initialize();
  {

    const ordinal_type numRows  = 16;
    const ordinal_type numCols  = 16;
    const size_type numNonZeros = 64;
    row_map_type row_map("row map", numRows + 1);
    col_ind_type col_ind("column indices", numNonZeros);
    values_type  values("values", numNonZeros);

    const size_type row_mapRaw[]    = {0, 3, 7, 11, 14, 18, 23, 28, 32, 36, 41, 46, 50, 53, 57, 61, 64};
    const ordinal_type col_indRaw[] = {0, 1, 4,
                                       0, 1, 2, 5,
                                       1, 2, 3, 6,
                                       2, 3, 7,
                                       0, 4, 5, 8,
                                       1, 4, 5, 6, 9,
                                       2, 5, 6, 7, 10,
                                       3, 6, 7, 11,
                                       4, 8, 9, 12,
                                       5, 8, 9, 10, 13,
                                       6, 9, 10, 11, 14,
                                       7, 10, 11, 15,
                                       8, 12, 13,
                                       9, 12, 13, 14,
                                       10, 13, 14, 15,
                                       11, 14, 15};
    const value_type values_Raw[]   = {4, -1, -1,
                                       -1, 4, -1, -1,
                                       -1, 4, -1, -1,
                                       -1, 4, -1,
                                       -1, 4, -1, -1,
                                       -1, -1, 4, -1, -1,
                                       -1, -1, 4, -1, -1,
                                       -1, -1, 4, -1,
                                       -1, 4, -1, -1,
                                       -1, -1, 4, -1, -1,
                                       -1, -1, 4, -1, -1,
                                       -1, -1, 4, -1,
                                       -1, 4, -1,
                                       -1, -1, 4, -1,
                                       -1, -1, 4, -1,
                                       -1, -1, 4};

    typename row_map_type::HostMirror::const_type row_map_host(row_mapRaw, numRows + 1);
    typename col_ind_type::HostMirror::const_type col_ind_host(col_indRaw, numNonZeros);
    typename values_type::HostMirror::const_type  values_host(values_Raw, numNonZeros);

    Kokkos::deep_copy(row_map, row_map_host);
    Kokkos::deep_copy(col_ind, col_ind_host);
    Kokkos::deep_copy(values, values_host);

    local_matrix_type A = local_matrix_type("A", numRows, numCols, numNonZeros, values, row_map, col_ind);

    values_type  discarded_fill("discarded fill", numRows);
    col_ind_type deficiency("deficiency", numRows);
    col_ind_type permutation("row permutation", numRows);

    Kokkos::RangePolicy<ordinal_type, execution_space> myRangePolicy(0, A.numRows());
    MDF_discarded_fill_norm<local_matrix_type> MDF_df_norm(A, discarded_fill, deficiency);
    Kokkos::parallel_for(myRangePolicy, MDF_df_norm);

    ordinal_type selected_row_idx = 0;
    MDF_select_row<local_matrix_type> MDF_row_selector(discarded_fill, deficiency, A.graph.row_map);
    Kokkos::parallel_reduce(myRangePolicy, MDF_row_selector, selected_row_idx);
    printf("Selected row is: %d\n", selected_row_idx);

  } // Scope for Kokkos::initialize/finalize
  Kokkos::finalize();
} // unit test analytical

TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL(Ifpack2SingleProcessMDF, analytical, Scalar, LO, GO)
{
  Ifpack2SingleProcessMDF_analytical<Scalar, LO, GO> (success, out, IlukImplTypeDetails::Serial);
}

//
// Instantiate and run unit tests
//

#define UNIT_TEST_GROUP_SC_LO_GO( SC, LO, GO ) \
  TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( Ifpack2SingleProcessMDF, analytical, SC, LO, GO )

// FIXME (21 Oct 2015) There was a FIXME here a while back about
// matrix-matrix add not getting instantiated for Scalar != double.
// Need to fix that to make this test work for Scalar != double.

#ifdef HAVE_TPETRA_INST_DOUBLE
#  define UNIT_TEST_GROUP_LO_GO( LO, GO ) \
     UNIT_TEST_GROUP_SC_LO_GO( double, LO, GO )
#else // NOT HAVE_TPETRA_INST_DOUBLE
#  define UNIT_TEST_GROUP_LO_GO( LO, GO )
#endif // HAVE_TPETRA_INST_DOUBLE

#include "Ifpack2_ETIHelperMacros.h"

IFPACK2_ETI_MANGLING_TYPEDEFS()

IFPACK2_INSTANTIATE_LG( UNIT_TEST_GROUP_LO_GO )

} // namespace (anonymous)
