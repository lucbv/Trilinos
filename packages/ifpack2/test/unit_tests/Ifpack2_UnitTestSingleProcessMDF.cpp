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
#include "Tpetra_MatrixIO.hpp"
#include "MatrixMarket_Tpetra.hpp"
#include "TpetraExt_MatrixMatrix.hpp"

#include "Ifpack2_UnitTestHelpers.hpp"
#include "Ifpack2_MDF.hpp"

namespace { // (anonymous)

using Tpetra::global_size_t;
typedef tif_utest::Node Node;

struct IlukImplTypeDetails {
  enum Enum { Serial, KSPILUK };
};

template<class MatrixType, class VectorType>
void remove_diags_and_scale(const MatrixType& L, const MatrixType& U,
                            Teuchos::RCP<MatrixType>& Ln, Teuchos::RCP<MatrixType>& Un, Teuchos::RCP<VectorType>& Dn) {

  typedef typename MatrixType::local_matrix_device_type local_matrix_type;
  typedef typename std::remove_const<typename local_matrix_type::size_type>::type    size_type;
  typedef typename std::remove_const<typename local_matrix_type::ordinal_type>::type ordinal_type;
  typedef typename std::remove_const<typename local_matrix_type::value_type>::type   value_type;
  typedef typename local_matrix_type::device_type device_type;
  typedef typename device_type::execution_space   execution_space;

  typedef typename Kokkos::View<size_type*, Kokkos::LayoutLeft, device_type> rowmap_type;
  typedef typename Kokkos::View<ordinal_type*, Kokkos::LayoutLeft, device_type> entries_type;
  typedef typename Kokkos::View<value_type*, Kokkos::LayoutRight, device_type> values_type;

  typedef Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename Kokkos::TeamPolicy<execution_space>::member_type member_type;

  auto L_rowmap  = L.getLocalMatrixDevice().graph.row_map;
  auto L_entries = L.getLocalMatrixDevice().graph.entries;
  auto L_values  = L.getLocalValuesView();
  auto U_rowmap  = U.getLocalMatrixDevice().graph.row_map;
  auto U_entries = U.getLocalMatrixDevice().graph.entries;
  auto U_values  = U.getLocalValuesView();

  rowmap_type  Ln_rowmap ("Ln_rowmap",  L_rowmap.extent(0));
  entries_type Ln_entries("Ln_entries", L_entries.extent(0) - (L_rowmap.extent(0) - 1));
  values_type  Ln_values ("Ln_values",  L_values.extent(0)  - (L_rowmap.extent(0) - 1));
  rowmap_type  Un_rowmap ("Un_rowmap",  U_rowmap.extent(0));
  entries_type Un_entries("Un_entries", U_entries.extent(0) - (U_rowmap.extent(0) - 1));
  values_type  Un_values ("Un_values",  U_values.extent(0)  - (U_rowmap.extent(0) - 1));

  values_type  Dn_values ("Dn_values",  U_rowmap.extent(0) - 1);

  Kokkos::parallel_for( Kokkos::RangePolicy<execution_space>(0, U_rowmap.extent(0)-1), KOKKOS_LAMBDA(const int& i) {
    Ln_rowmap(i+1) = L_rowmap(i+1) - (i+1);
    Un_rowmap(i+1) = U_rowmap(i+1) - (i+1);
    Dn_values(i)   = 1.0/U_values(U_rowmap(i));
  });

  const team_policy policy( U_rowmap.extent(0)-1, Kokkos::AUTO );

  Kokkos::parallel_for( policy, KOKKOS_LAMBDA(const member_type& teamMember) {
    const int rowid = teamMember.league_rank();

    auto Lentries_src = subview(L_entries,  Kokkos::make_pair(L_rowmap(rowid),  L_rowmap(rowid+1) - 1));
    auto Lvalues_src  = subview(L_values,   Kokkos::make_pair(L_rowmap(rowid),  L_rowmap(rowid+1) - 1));
    auto Lentries_dst = subview(Ln_entries, Kokkos::make_pair(Ln_rowmap(rowid), Ln_rowmap(rowid+1)));
    auto Lvalues_dst  = subview(Ln_values,  Kokkos::make_pair(Ln_rowmap(rowid), Ln_rowmap(rowid+1)));

    auto Uentries_src = subview(U_entries,  Kokkos::make_pair(U_rowmap(rowid)+1, U_rowmap(rowid+1)));
    auto Uvalues_src  = subview(U_values,   Kokkos::make_pair(U_rowmap(rowid)+1, U_rowmap(rowid+1)));
    auto Uentries_dst = subview(Un_entries, Kokkos::make_pair(Un_rowmap(rowid),  Un_rowmap(rowid+1)));
    auto Uvalues_dst  = subview(Un_values,  Kokkos::make_pair(Un_rowmap(rowid),  Un_rowmap(rowid+1)));

    Kokkos::Experimental::local_deep_copy(teamMember, Lentries_dst, Lentries_src);
    Kokkos::Experimental::local_deep_copy(teamMember, Uentries_dst, Uentries_src);
    Kokkos::Experimental::local_deep_copy(teamMember, Lvalues_dst, Lvalues_src);
    Kokkos::Experimental::local_deep_copy(teamMember, Uvalues_dst, Uvalues_src);

    teamMember.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, Uvalues_dst.extent(0)), [&](const int& i) {
      Uvalues_dst(i) = Uvalues_dst(i)*Dn_values(rowid);
    });

    teamMember.team_barrier();
  });

  Ln = Teuchos::rcp (new MatrixType (L.getRowMap(), L.getColMap(),
                                     Ln_rowmap, Ln_entries, Ln_values));
  Un = Teuchos::rcp (new MatrixType (U.getRowMap(), U.getColMap(),
                                     Un_rowmap, Un_entries, Un_values));
  auto Dn_view = Dn->getLocalViewDevice(Tpetra::Access::OverwriteAll);
  Kokkos::deep_copy(subview(Dn_view,Kokkos::ALL(), 0),Dn_values);

  Ln->fillComplete();
  Un->fillComplete();
}

template<typename Scalar, typename LO, typename GO>
void Ifpack2SingleProcessMDF_test0 (bool& success, Teuchos::FancyOStream& out, const IlukImplTypeDetails::Enum ilukimplType) {
  using Teuchos::RCP;
  using std::endl;

  if (ilukimplType == IlukImplTypeDetails::Serial)
    out << "Ifpack2::RILUK: Test0 -- Serial" << endl;
  else
    out << "Ifpack2::RILUK: Test0 -- Kokkos Kernels SPILUK" << endl;

  Teuchos::OSTab tab0 (out);

  global_size_t num_rows_per_proc = 5;
  const RCP<const Tpetra::Map<LO,GO,Node> > rowmap =
    tif_utest::create_tpetra_map<LO,GO,Node>(num_rows_per_proc);

  if (rowmap->getComm()->getSize() > 1) {
    out << endl << "This test may only be run in serial." << endl;
    return;
  }

  RCP<const Tpetra::CrsMatrix<Scalar,LO,GO,Node> > crsmatrix =
    tif_utest::create_test_matrix<Scalar,LO,GO,Node> (rowmap);

  //----------------Default trisolver----------------//
  {
    Ifpack2::MDF<Tpetra::RowMatrix<Scalar,LO,GO,Node> > prec (crsmatrix);

    Teuchos::ParameterList params;
    const int fill_level = 0;
    params.set("fact: mdf level-of-fill", fill_level);
    params.set("fact: mdf level-of-overlap", 0);

    if (ilukimplType == IlukImplTypeDetails::KSPILUK)
      params.set("fact: type", "KSPILUK");

    TEST_NOTHROW(prec.setParameters(params));

    TEST_EQUALITY( prec.getLevelOfFill(), fill_level);

    prec.initialize();
    //trivial tests to insist that the preconditioner's domain/range maps are
    //the same as those of the matrix:
    const Tpetra::Map<LO,GO,Node>& mtx_dom_map = *crsmatrix->getDomainMap();
    const Tpetra::Map<LO,GO,Node>& mtx_rng_map = *crsmatrix->getRangeMap();

    const Tpetra::Map<LO,GO,Node>& prec_dom_map = *prec.getDomainMap();
    const Tpetra::Map<LO,GO,Node>& prec_rng_map = *prec.getRangeMap();

    TEST_ASSERT( prec_dom_map.isSameAs(mtx_dom_map) );
    TEST_ASSERT( prec_rng_map.isSameAs(mtx_rng_map) );

    prec.compute();

    Tpetra::MultiVector<Scalar,LO,GO,Node> x(rowmap,2), y(rowmap,2);
    x.putScalar (Teuchos::ScalarTraits<Scalar>::one ());

    prec.apply(x, y);

    Teuchos::ArrayRCP<const Scalar> yview = y.get1dView();

    //y should be full of 0.5's now.

    Teuchos::ArrayRCP<Scalar> halfs(num_rows_per_proc*2, 0.5);

    TEST_COMPARE_FLOATING_ARRAYS(yview, halfs(), Teuchos::ScalarTraits<Scalar>::eps());
  }
  //----------------Kokkos Kernels SPTRSV----------------//
  {
    Ifpack2::MDF<Tpetra::RowMatrix<Scalar,LO,GO,Node> > prec (crsmatrix);

    Teuchos::ParameterList params;
    const int fill_level = 0;
    params.set("fact: mdf level-of-fill", fill_level);
    params.set("fact: mdf level-of-overlap", 0);

    if (ilukimplType == IlukImplTypeDetails::KSPILUK)
      params.set("fact: type", "KSPILUK");

    params.set("trisolver: type", "KSPTRSV");

    TEST_NOTHROW(prec.setParameters(params));

    TEST_EQUALITY( prec.getLevelOfFill(), fill_level);

    prec.initialize();
    //trivial tests to insist that the preconditioner's domain/range maps are
    //the same as those of the matrix:
    const Tpetra::Map<LO,GO,Node>& mtx_dom_map = *crsmatrix->getDomainMap();
    const Tpetra::Map<LO,GO,Node>& mtx_rng_map = *crsmatrix->getRangeMap();

    const Tpetra::Map<LO,GO,Node>& prec_dom_map = *prec.getDomainMap();
    const Tpetra::Map<LO,GO,Node>& prec_rng_map = *prec.getRangeMap();

    TEST_ASSERT( prec_dom_map.isSameAs(mtx_dom_map) );
    TEST_ASSERT( prec_rng_map.isSameAs(mtx_rng_map) );

    prec.compute();

    Tpetra::MultiVector<Scalar,LO,GO,Node> x(rowmap,2), y(rowmap,2);
    x.putScalar (Teuchos::ScalarTraits<Scalar>::one ());

    prec.apply(x, y);

    Teuchos::ArrayRCP<const Scalar> yview = y.get1dView();

    //y should be full of 0.5's now.

    Teuchos::ArrayRCP<Scalar> halfs(num_rows_per_proc*2, 0.5);

    TEST_COMPARE_FLOATING_ARRAYS(yview, halfs(), Teuchos::ScalarTraits<Scalar>::eps());
  }
}

template<typename Scalar, typename LO, typename GO>
void Ifpack2SingleProcessMDF_test1 (bool& success, Teuchos::FancyOStream& out, const IlukImplTypeDetails::Enum ilukimplType) {
  using Kokkos::Details::ArithTraits;
  using Teuchos::RCP;
  using std::endl;
  typedef Tpetra::Map<LO, GO, Node> map_type;
  typedef Tpetra::CrsMatrix<Scalar,LO,GO,Node> crs_matrix_type;
  typedef Tpetra::MultiVector<Scalar,LO,GO,Node> MV;
  typedef Tpetra::RowMatrix<Scalar,LO,GO,Node> row_matrix_type;
  typedef Teuchos::ScalarTraits<Scalar> STS;
  typedef typename MV::impl_scalar_type val_type;
  typedef typename Kokkos::Details::ArithTraits<val_type>::mag_type mag_type;
  typedef typename map_type::device_type device_type;
  const mag_type oneMag = ArithTraits<mag_type>::one ();
  const mag_type twoMag = oneMag + oneMag;

  if (ilukimplType == IlukImplTypeDetails::Serial)
    out << "Ifpack2::MDF: Test1 -- Serial" << endl;
  else
    out << "Ifpack2::MDF: Test1 -- Kokkos Kernels SPILUK" << endl;

  Teuchos::OSTab tab1 (out);

  const global_size_t num_rows_per_proc = 5;
  RCP<const map_type> rowmap =
    tif_utest::create_tpetra_map<LO, GO, Node> (num_rows_per_proc);

  // Matrix
  // [ 2 .1  0  0  0]
  // [.1  2  0  0  0]
  // [ 0 .1  2 .1  0]
  // [ 0  0 .1  2 .1]
  // [ 0  0  0 .1  2]

  // Matlab's Factors
  // L
  // Diagonal = 1 (implied)
  // Subdiagonal (approx) = .05, .0501, .0501 .0501

  // U
  // Diagonal (approx)      = 2 1.995 1.995 1.995 1.995
  // Superdiagonal (approx) = .1 .1 .1 .1


  if (rowmap->getComm ()->getSize () > 1) {
    out << "This test may only be run in serial "
      "or with a single MPI process." << endl;
    return;
  }

  out << "Creating matrix" << endl;
  RCP<const crs_matrix_type> crsmatrix =
    tif_utest::create_test_matrix2<Scalar,LO,GO,Node>(rowmap);

  {//CMS
    auto out = Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cout));
    *out<<"***** A *****"<<std::endl;
    crsmatrix->describe(*out,Teuchos::VERB_EXTREME);
  }

  //----------------Default trisolver----------------//
  {
    out << "Creating preconditioner" << endl;
    Ifpack2::MDF<row_matrix_type> prec (crsmatrix);

    out << "Setting preconditioner's parameters" << endl;
    Teuchos::ParameterList params;
    params.set ("fact: mdf level-of-fill", 1);
    params.set ("fact: mdf level-of-overlap", 0);
    if (ilukimplType == IlukImplTypeDetails::KSPILUK)
      params.set("fact: type", "KSPILUK");
    TEST_NOTHROW(prec.setParameters(params));

    out << "Calling initialize() and compute()" << endl;
    prec.initialize();
    prec.compute();

  {//CMS
    auto out = Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cout));
    *out<<"***** Test L *****"<<std::endl;
    prec.getL().describe(*out,Teuchos::VERB_EXTREME);
    *out<<"***** Test U *****"<<std::endl;
    prec.getU().describe(*out,Teuchos::VERB_EXTREME);
    *out<<"***** Test D *****"<<std::endl;
    prec.getD().describe(*out,Teuchos::VERB_EXTREME);
  }


    out << "Creating test problem" << endl;
    MV x (rowmap, 2);
    MV y (rowmap, 2);
    x.putScalar (STS::one ());

    out << "Calling crsmatrix->apply(x, y)" << endl;
    crsmatrix->apply(x,y);

    out << "Calling prec.apply(y, x)" << endl;
    //apply the preconditioner to y, putting ~A^-1*y in x
    //(this should set x back to 1's)
    prec.apply(y, x);

    //x should be full of 1's now.
    out << "Checking result" << endl;

    // Useful things for comparing results.
    Kokkos::View<mag_type*, device_type> norms ("norms", x.getNumVectors ());
    auto norms_h = Kokkos::create_mirror_view (norms);
    MV diff (x.getMap (), x.getNumVectors ());

    {
      // FIXME (mfh 04 Oct 2016) This is the bound that I found here
      // when I fixed this test.  Not sure if it's sensible.
      const mag_type bound = twoMag * ArithTraits<val_type>::eps ();

      diff.putScalar (STS::one ());
      diff.update (STS::one (), x, -STS::one ());
      diff.normInf (norms);
      Kokkos::deep_copy (norms_h, norms);
      for (LO j = 0; j < static_cast<LO> (norms_h.extent (0)); ++j) {
        const mag_type absVal = ArithTraits<mag_type>::abs (norms_h(j));
        TEST_ASSERT( absVal <= bound );
        if (absVal > bound) {
          out << "\\| x(:," << j << ") - 1 \\|_{\\infty} = "
              << absVal << " > " << bound
              << "; the norm equals " << norms_h(j)
              << endl;
        }
      }
    }

    auto test_alpha_beta = [&] (const Scalar& alpha, const Scalar& beta,
                                const Teuchos::ETransp& mode) {
      out << "Testing apply() for alpha = " << alpha
          << " and beta = " << beta << endl;
      const Scalar x_magic_number = -0.42;
      x.putScalar (x_magic_number);
      crsmatrix->apply (x, y, mode);
      MV z = Tpetra::createCopy (x);
      const Scalar z_magic_number = 2.1;
      z.putScalar (z_magic_number);
      // z = beta z + alpha inv(A) y
      //   = beta z + alpha x
      //   = (beta z_magic_number + alpha x_magic_number) 1s
      prec.apply(y, z, mode, alpha, beta);

      MV z_true = Tpetra::createCopy (x);
      const Scalar z_true_scalar = beta*z_magic_number + alpha*x_magic_number;
      z_true.putScalar(z_true_scalar);

      {
        // FIXME (mfh 04 Oct 2016) This is the bound that I found here
        // when I fixed this test.  Not sure if it's sensible.
        const mag_type bound = 10.0 * ArithTraits<val_type>::eps ();

        diff.putScalar (z_true_scalar);
        diff.update (STS::one (), z, -STS::one ());
        diff.normInf (norms);
        Kokkos::deep_copy (norms_h, norms);

        for (LO j = 0; j < static_cast<LO> (norms_h.extent (0)); ++j) {
          const mag_type absVal = ArithTraits<mag_type>::abs (norms_h(j));
          TEST_ASSERT( absVal <= bound );
          if (absVal > bound) {
            out << "\\| x(:," << j << ") - 1 \\|_{\\infty} = "
                << absVal << " > " << bound
                << "; the norm equals " << norms_h(j)
                << endl;
          }
        }
      }
    };

    for (const auto mode : {Teuchos::NO_TRANS, Teuchos::TRANS}) {
      test_alpha_beta(0.0, 0.0, mode);
      test_alpha_beta(2.0, 0.0, mode);
      test_alpha_beta(0.0, -1.5, mode);
      test_alpha_beta(-0.42, 4.2, mode);
    }
  }

  return;//CMS
  //----------------Kokkos Kernels SPTRSV----------------//
  {
    out << "Creating preconditioner" << endl;
    Ifpack2::MDF<row_matrix_type> prec (crsmatrix);

    out << "Setting preconditioner's parameters" << endl;
    Teuchos::ParameterList params;
    params.set ("fact: mdf level-of-fill", 0);
    params.set ("fact: mdf level-of-overlap", 0);
    if (ilukimplType == IlukImplTypeDetails::KSPILUK)
      params.set("fact: type", "KSPILUK");
    params.set("trisolver: type", "KSPTRSV");
    TEST_NOTHROW(prec.setParameters(params));

    out << "Calling initialize() and compute()" << endl;
    prec.initialize();
    prec.compute();

    out << "Creating test problem" << endl;
    MV x (rowmap, 2);
    MV y (rowmap, 2);
    x.putScalar (STS::one ());

    out << "Calling crsmatrix->apply(x, y)" << endl;
    crsmatrix->apply(x,y);

    out << "Calling prec.apply(y, x)" << endl;
    //apply the preconditioner to y, putting ~A^-1*y in x
    //(this should set x back to 1's)
    prec.apply(y, x);

    //x should be full of 1's now.
    out << "Checking result" << endl;

    // Useful things for comparing results.
    Kokkos::View<mag_type*, device_type> norms ("norms", x.getNumVectors ());
    auto norms_h = Kokkos::create_mirror_view (norms);
    MV diff (x.getMap (), x.getNumVectors ());

    {
      // FIXME (mfh 04 Oct 2016) This is the bound that I found here
      // when I fixed this test.  Not sure if it's sensible.
      const mag_type bound = twoMag * ArithTraits<val_type>::eps ();

      diff.putScalar (STS::one ());
      diff.update (STS::one (), x, -STS::one ());
      diff.normInf (norms);
      Kokkos::deep_copy (norms_h, norms);
      for (LO j = 0; j < static_cast<LO> (norms_h.extent (0)); ++j) {
        const mag_type absVal = ArithTraits<mag_type>::abs (norms_h(j));
        TEST_ASSERT( absVal <= bound );
        if (absVal > bound) {
          out << "\\| x(:," << j << ") - 1 \\|_{\\infty} = "
              << absVal << " > " << bound
              << "; the norm equals " << norms_h(j)
              << endl;
        }
      }
    }

    auto test_alpha_beta = [&] (const Scalar& alpha, const Scalar& beta,
                                const Teuchos::ETransp& mode) {
      out << "Testing apply() for alpha = " << alpha
          << " and beta = " << beta << endl;
      const Scalar x_magic_number = -0.42;
      x.putScalar (x_magic_number);
      crsmatrix->apply (x, y, mode);
      MV z = Tpetra::createCopy (x);
      const Scalar z_magic_number = 2.1;
      z.putScalar (z_magic_number);
      // z = beta z + alpha inv(A) y
      //   = beta z + alpha x
      //   = (beta z_magic_number + alpha x_magic_number) 1s
      prec.apply(y, z, mode, alpha, beta);

      MV z_true = Tpetra::createCopy (x);
      const Scalar z_true_scalar = beta*z_magic_number + alpha*x_magic_number;
      z_true.putScalar(z_true_scalar);

      {
        // FIXME (mfh 04 Oct 2016) This is the bound that I found here
        // when I fixed this test.  Not sure if it's sensible.
        const mag_type bound = 10.0 * ArithTraits<val_type>::eps ();

        diff.putScalar (z_true_scalar);
        diff.update (STS::one (), z, -STS::one ());
        diff.normInf (norms);
        Kokkos::deep_copy (norms_h, norms);

        for (LO j = 0; j < static_cast<LO> (norms_h.extent (0)); ++j) {
          const mag_type absVal = ArithTraits<mag_type>::abs (norms_h(j));
          TEST_ASSERT( absVal <= bound );
          if (absVal > bound) {
            out << "\\| x(:," << j << ") - 1 \\|_{\\infty} = "
                << absVal << " > " << bound
                << "; the norm equals " << norms_h(j)
                << endl;
          }
        }
      }
    };

    for (const auto mode : {Teuchos::NO_TRANS, Teuchos::TRANS}) {
      test_alpha_beta(0.0, 0.0, mode);
      test_alpha_beta(2.0, 0.0, mode);
      test_alpha_beta(0.0, -1.5, mode);
      test_alpha_beta(-0.42, 4.2, mode);
    }
  }
  out << "Done with test" << endl;
}

TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL(Ifpack2SingleProcessMDF, Test0, Scalar, LO, GO)
{
  Ifpack2SingleProcessMDF_test0<Scalar, LO, GO> (success, out, IlukImplTypeDetails::Serial);
}

TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL(Ifpack2SingleProcessMDF, Test1, Scalar, LO, GO)
{
  Ifpack2SingleProcessMDF_test1<Scalar, LO, GO> (success, out, IlukImplTypeDetails::Serial);
}

TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL(Ifpack2SingleProcessMDF, TestGIDConsistency, Scalar, LO, GO)
{
  // Test that MDF(k) throws an exception if the ordering of the GIDs
  // in the row Map is not the same as the ordering of the local GIDs
  // in the column Map.  The MDF(k) setup and algorithm assumes this
  // for the moment.

  // 25April2014 JJH: The local filter appears to fix the column Map in parallel so that it's
  //                  consistently ordered with the row Map.  In otherwords, I can't get this
  //                  test to fail in parallel.  So this check is only necessary in serial.

  using Teuchos::RCP;
  using Teuchos::rcp;
  using std::endl;
  typedef Tpetra::CrsMatrix<Scalar, LO, GO, Node> crs_matrix_type;
  typedef Tpetra::RowMatrix<Scalar, LO, GO, Node> row_matrix_type;
  typedef Tpetra::Map<LO, GO, Node> map_type;
  typedef Tpetra::global_size_t GST;

  out << "Ifpack2::MDF: TestGIDConsistency" << endl;

  const GST INVALID = Teuchos::OrdinalTraits<GST>::invalid ();
  RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm ();

  if (comm->getSize () > 1) {
    out << endl << "This test only runs in serial." << endl;
    return;
  }

  // Create matrix: 5 rows per process, 3 entries per row
  const LO indexBase = 0;
  GST numRowsPerProc = 5;
  RCP<map_type> rowMap =
    rcp (new map_type (INVALID, numRowsPerProc, indexBase, comm));

  // Create a column Map with the same GIDs at the row Map, but in
  // permuted order.  The first entry is the same as the row Map, the
  // remainder are in descending order.
  Teuchos::ArrayView<const GO> rowGIDs = rowMap->getNodeElementList ();
  Teuchos::Array<GO> colElements (rowGIDs.size ());
  colElements[0] = rowGIDs[0];
  for (GO i = 1; i < rowGIDs.size (); ++i) {
    colElements[i] = rowGIDs[rowGIDs.size () - i];
  }

  RCP<const map_type> colMap =
    rcp (new map_type (INVALID, colElements (), indexBase, comm));
  RCP<crs_matrix_type> A = rcp (new crs_matrix_type (rowMap, colMap, 3));

  // Construct a nondiagonal matrix.  It's not tridiagonal because of
  // process boundaries.
  const Scalar one = Teuchos::ScalarTraits<Scalar>::one ();
  const Scalar two = one + one;
  Teuchos::Array<GO> col (3);
  Teuchos::Array<Scalar> val (3);
  size_t numLocalElts = rowMap->getNodeNumElements ();
  for (LO l_row = 0; static_cast<size_t> (l_row) < numLocalElts; ++l_row) {
    const GO g_row = rowMap->getGlobalElement (l_row);
    size_t i = 0;
    col[i] = g_row;
    val[i++] = two;
    if (l_row>0) {
      col[i] = rowMap->getGlobalElement (l_row - 1);
      val[i++] = -one;
    }
    if (static_cast<size_t> (l_row) < numLocalElts - 1) {
      col[i] = rowMap->getGlobalElement (l_row + 1);
      val[i++] = -one;
    }
    A->insertGlobalValues (g_row, col (0, i), val (0, i));
  }
  A->fillComplete ();

  RCP<const crs_matrix_type> constA = A;

  {
    Ifpack2::MDF<row_matrix_type> prec (constA);

    Teuchos::ParameterList params;
    const GO lof = 0;
    params.set ("fact: mdf level-of-fill", lof);
    params.set ("fact: mdf level-of-overlap", 0);

    prec.setParameters (params);
    TEST_THROW( prec.initialize (), std::runtime_error);
  }
  {
    Ifpack2::MDF<row_matrix_type> prec (constA);

    Teuchos::ParameterList params;
    const GO lof = 0;
    params.set ("fact: mdf level-of-fill", lof);
    params.set ("fact: mdf level-of-overlap", 0);
    params.set ("fact: type", "KSPILUK");

    prec.setParameters (params);
    TEST_THROW( prec.initialize (), std::runtime_error);
  }

} // unit test TestGIDConsistency()

//
// Instantiate and run unit tests
//

#define UNIT_TEST_GROUP_SC_LO_GO( SC, LO, GO ) \
  TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( Ifpack2SingleProcessMDF, Test0, SC, LO, GO ) \
  TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( Ifpack2SingleProcessMDF, Test1, SC, LO, GO ) \
  TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( Ifpack2SingleProcessMDF, TestGIDConsistency, SC, LO, GO )

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
