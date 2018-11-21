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

#ifndef TPETRA_EXPERIMENTAL_STRUCTUREDCRSWRAPPER_DEF_HPP
#define TPETRA_EXPERIMENTAL_STRUCTUREDCRSWRAPPER_DEF_HPP

#include "Tpetra_Experimental_StructuredCrsWrapper_decl.hpp"
#include "Tpetra_Details_Profiling.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Import.hpp"

namespace Tpetra {

namespace Experimental {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::StructuredCrsWrapper(const Teuchos::RCP<const crs_matrix_type> &matrix, const Teuchos::RCP<Teuchos::ParameterList>& params):matrix_(matrix) {

}


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > 
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getDomainMap() const {return matrix_->getDomainMap(); }
  

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > 
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getRangeMap() const {return matrix_->getRangeMap();}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > 
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getRowMap() const {return matrix_->getRowMap();}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > 
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getColMap() const {return matrix_->getColMap();}


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Teuchos::Comm<int> > 
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getComm() const {return matrix_->getComm();}


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::apply (const MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &X,
                                                                     MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &Y,
                                                                     Teuchos::ETransp mode,
                                                                     Scalar alpha,
                                                                     Scalar beta) const {
  using Tpetra::Details::ProfilingRegion;  
  const char fnName[] = "Tpetra::StructuredCrsWrapper::apply";
   TEUCHOS_TEST_FOR_EXCEPTION
     (! matrix_->isFillComplete (), std::runtime_error,
       fnName << ": Cannot call apply() until fillComplete() "
       "has been called.");

  if (mode == Teuchos::NO_TRANS) {
    ProfilingRegion regionNonTranspose (fnName);
    this->applyNonTranspose (X, Y, alpha, beta);
  }
  else {
    // Default to the CrsMatrix apply for the Transpose case
    ProfilingRegion regionTranspose (fnName);
    matrix_->apply(X,Y,mode,alpha,beta);
  }
  
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::applyNonTranspose (const MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &X_in,
                                                                                 MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &Y_in,
                                                                                 Scalar alpha,
                                                                                 Scalar beta) const {
  typedef MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> MV;
  // NOTE: This code is cut and paste with only minor modifications from Tpetra::CrsMatrix::applyNonTranspose

    using Tpetra::Details::ProfilingRegion;
    using Teuchos::null;
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::rcp_const_cast;
    using Teuchos::rcpFromRef;
    const Scalar ZERO = Teuchos::ScalarTraits<Scalar>::zero ();
    const Scalar ONE = Teuchos::ScalarTraits<Scalar>::one ();

    // mfh 05 Jun 2014: Special case for alpha == 0.  I added this to
    // fix an Ifpack2 test (RILUKSingleProcessUnitTests), which was
    // failing only for the Kokkos refactor version of Tpetra.  It's a
    // good idea regardless to have the bypass.
    if (alpha == ZERO) {
      if (beta == ZERO) {
        Y_in.putScalar (ZERO);
      } else if (beta != ONE) {
        Y_in.scale (beta);
      }
      return;
    }

    // It's possible that X is a view of Y or vice versa.  We don't
    // allow this (apply() requires that X and Y not alias one
    // another), but it's helpful to detect and work around this case.
    // We don't try to to detect the more subtle cases (e.g., one is a
    // subview of the other, but their initial pointers differ).  We
    // only need to do this if this matrix's Import is trivial;
    // otherwise, we don't actually apply the operator from X into Y.

    RCP<const import_type> importer = matrix_->getGraph ()->getImporter ();
    RCP<const export_type> exporter = matrix_->getGraph ()->getExporter ();

    // If beta == 0, then the output MV will be overwritten; none of
    // its entries should be read.  (Sparse BLAS semantics say that we
    // must ignore any Inf or NaN entries in Y_in, if beta is zero.)
    // This matters if we need to do an Export operation; see below.
    const bool Y_is_overwritten = (beta == ZERO);

    // We treat the case of a replicated MV output specially.
    const bool Y_is_replicated =
      (! Y_in.isDistributed () && this->getComm ()->getSize () != 1);

    // This is part of the special case for replicated MV output.
    // We'll let each process do its thing, but do an all-reduce at
    // the end to sum up the results.  Setting beta=0 on all processes
    // but Proc 0 makes the math work out for the all-reduce.  (This
    // assumes that the replicated data is correctly replicated, so
    // that the data are the same on all processes.)
    if (Y_is_replicated && this->getComm ()->getRank () > 0) {
      beta = ZERO;
    }

    // Temporary MV for Import operation.  After the block of code
    // below, this will be an (Imported if necessary) column Map MV
    // ready to give to localMultiply().
    RCP<const MV> X_colMap;
    if (importer.is_null ()) {
      if (! X_in.isConstantStride ()) {
        // Not all sparse mat-vec kernels can handle an input MV with
        // nonconstant stride correctly, so we have to copy it in that
        // case into a constant stride MV.  To make a constant stride
        // copy of X_in, we force creation of the column (== domain)
        // Map MV (if it hasn't already been created, else fetch the
        // cached copy).  This avoids creating a new MV each time.
        RCP<MV> X_colMapNonConst = matrix_->getColumnMapMultiVector (X_in, true);
        Tpetra::deep_copy (*X_colMapNonConst, X_in);
        X_colMap = rcp_const_cast<const MV> (X_colMapNonConst);
      }
      else {
        // The domain and column Maps are the same, so do the local
        // multiply using the domain Map input MV X_in.
        X_colMap = rcpFromRef (X_in);
      }
    }
    else { // need to Import source (multi)vector
      ProfilingRegion regionImport ("Tpetra::StructuredCrsWrapper::apply: Import");

      // We're doing an Import anyway, which will copy the relevant
      // elements of the domain Map MV X_in into a separate column Map
      // MV.  Thus, we don't have to worry whether X_in is constant
      // stride.
      RCP<MV> X_colMapNonConst = matrix_->getColumnMapMultiVector (X_in);

      // Import from the domain Map MV to the column Map MV.
      X_colMapNonConst->doImport (X_in, *importer, INSERT);
      X_colMap = rcp_const_cast<const MV> (X_colMapNonConst);
    }

    // Temporary MV for doExport (if needed), or for copying a
    // nonconstant stride output MV into a constant stride MV.  This
    // is null if we don't need the temporary MV, that is, if the
    // Export is trivial (null).
    RCP<MV> Y_rowMap = matrix_->getRowMapMultiVector (Y_in);

    // If we have a nontrivial Export object, we must perform an
    // Export.  In that case, the local multiply result will go into
    // the row Map multivector.  We don't have to make a
    // constant-stride version of Y_in in this case, because we had to
    // make a constant stride Y_rowMap MV and do an Export anyway.
    if (! exporter.is_null ()) {
      this->localApply (*X_colMap, *Y_rowMap, Teuchos::NO_TRANS, alpha, ZERO);
      {
        ProfilingRegion regionExport ("Tpetra::StructuredCrsWrapper::apply: Export");

        // If we're overwriting the output MV Y_in completely (beta ==
        // 0), then make sure that it is filled with zeros before we
        // do the Export.  Otherwise, the ADD combine mode will use
        // data in Y_in, which is supposed to be zero.
        if (Y_is_overwritten) {
          Y_in.putScalar (ZERO);
        }
        else {
          // Scale output MV by beta, so that doExport sums in the
          // mat-vec contribution: Y_in = beta*Y_in + alpha*A*X_in.
          Y_in.scale (beta);
        }
        // Do the Export operation.
        Y_in.doExport (*Y_rowMap, *exporter, ADD);
      }
    }
    else { // Don't do an Export: row Map and range Map are the same.
      //
      // If Y_in does not have constant stride, or if the column Map
      // MV aliases Y_in, then we can't let the kernel write directly
      // to Y_in.  Instead, we have to use the cached row (== range)
      // Map MV as temporary storage.
      //
      // FIXME (mfh 05 Jun 2014) This test for aliasing only tests if
      // the user passed in the same MultiVector for both X and Y.  It
      // won't detect whether one MultiVector views the other.  We
      // should also check the MultiVectors' raw data pointers.
      if (! Y_in.isConstantStride () || X_colMap.getRawPtr () == &Y_in) {
        // Force creating the MV if it hasn't been created already.
        // This will reuse a previously created cached MV.
        Y_rowMap = matrix_->getRowMapMultiVector (Y_in, true);

        // If beta == 0, we don't need to copy Y_in into Y_rowMap,
        // since we're overwriting it anyway.
        if (beta != ZERO) {
          Tpetra::deep_copy (*Y_rowMap, Y_in);
        }
        this->localApply (*X_colMap, *Y_rowMap, Teuchos::NO_TRANS, alpha, beta);
        Tpetra::deep_copy (Y_in, *Y_rowMap);
      }
      else {
        this->localApply (*X_colMap, Y_in, Teuchos::NO_TRANS, alpha, beta);
      }
    }

    // If the range Map is a locally replicated Map, sum up
    // contributions from each process.  We set beta = 0 on all
    // processes but Proc 0 initially, so this will handle the scaling
    // factor beta correctly.
    if (Y_is_replicated) {
      ProfilingRegion regionReduce ("Tpetra::StructuredCrsWrapper::apply: Reduce Y");
      Y_in.reduce ();
    }
}


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::localApply (const MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &X,
                                                                     MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &Y,
                                                                     Teuchos::ETransp mode,
                                                                     const Scalar& alpha,
                                                                     const Scalar& beta) const {  
  // NOTE: This code is largely cut and paste from Tpetra::CrsMatrix::localMultiply()
      using Teuchos::NO_TRANS;

      // Should probably fix this later to allow for scalar type mixing
      //      typedef Scalar DomainScalar;
      typedef Scalar RangeScalar;

      // Error out if we get called w/ transpose
      const char fnName[] = "Tpetra::StructuredCrsWrapper::localApply";
      TEUCHOS_TEST_FOR_EXCEPTION
        (mode != NO_TRANS, std::runtime_error,
         fnName << ": Cannot call localApply() in TRANSPOSE mode.");
      
      // Just like Scalar and impl_scalar_type may differ in CrsMatrix,
      // RangeScalar and its corresponding impl_scalar_type may differ in
      // MultiVector.
      typedef typename MultiVector<RangeScalar, LocalOrdinal, GlobalOrdinal,
        Node>::impl_scalar_type range_impl_scalar_type;
#ifdef HAVE_TPETRA_DEBUG
      const char tfecfFuncName[] = "localApply: ";
#endif // HAVE_TPETRA_DEBUG

      const range_impl_scalar_type theAlpha = static_cast<range_impl_scalar_type> (alpha);
      const range_impl_scalar_type theBeta = static_cast<range_impl_scalar_type> (beta);
#if 0
      const bool conjugate = (mode == Teuchos::CONJ_TRANS);
      const bool transpose = (mode != Teuchos::NO_TRANS);
#endif
      auto X_lcl = X.template getLocalView<device_type> ();
      auto Y_lcl = Y.template getLocalView<device_type> ();

#ifdef HAVE_TPETRA_DEBUG
      TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC
        (X.getNumVectors () != Y.getNumVectors (), std::runtime_error,
         "X.getNumVectors() = " << X.getNumVectors () << " != Y.getNumVectors() = "
         << Y.getNumVectors () << ".");
      TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC
        (! transpose && X.getLocalLength () != getColMap ()->getNodeNumElements (),
         std::runtime_error, "NO_TRANS case: X has the wrong number of local rows.  "
         "X.getLocalLength() = " << X.getLocalLength () << " != getColMap()->"
         "getNodeNumElements() = " << getColMap ()->getNodeNumElements () << ".");
      TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC
        (! transpose && Y.getLocalLength () != getRowMap ()->getNodeNumElements (),
         std::runtime_error, "NO_TRANS case: Y has the wrong number of local rows.  "
         "Y.getLocalLength() = " << Y.getLocalLength () << " != getRowMap()->"
         "getNodeNumElements() = " << getRowMap ()->getNodeNumElements () << ".");
      TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC
        (transpose && X.getLocalLength () != getRowMap ()->getNodeNumElements (),
         std::runtime_error, "TRANS or CONJ_TRANS case: X has the wrong number of "
         "local rows.  X.getLocalLength() = " << X.getLocalLength () << " != "
         "getRowMap()->getNodeNumElements() = "
         << getRowMap ()->getNodeNumElements () << ".");
      TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC
        (transpose && Y.getLocalLength () != getColMap ()->getNodeNumElements (),
         std::runtime_error, "TRANS or CONJ_TRANS case: X has the wrong number of "
         "local rows.  Y.getLocalLength() = " << Y.getLocalLength () << " != "
         "getColMap()->getNodeNumElements() = "
         << getColMap ()->getNodeNumElements () << ".");
      TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC
        (! isFillComplete (), std::runtime_error, "The matrix is not fill "
         "complete.  You must call fillComplete() (possibly with domain and range "
         "Map arguments) without an intervening resumeFill() call before you may "
         "call this method.");
      TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC
        (! X.isConstantStride () || ! Y.isConstantStride (), std::runtime_error,
         "X and Y must be constant stride.");
      // If the two pointers are NULL, then they don't alias one
      // another, even though they are equal.
      TEUCHOS_TEST_FOR_EXCEPTION_CLASS_FUNC(
        X_lcl.data () == Y_lcl.data () &&
        X_lcl.data () != NULL,
        std::runtime_error, "X and Y may not alias one another.");
#endif // HAVE_TPETRA_DEBUG

      // Y = alpha*op(M) + beta*Y

#if 1
      KokkosSparse::spmv (KokkosSparse::NoTranspose,
                          theAlpha,
                          matrix_->lclMatrix_,
                          X.template getLocalView<device_type> (),
                          theBeta,
                          Y.template getLocalView<device_type> ());
#else
      KokkosSparse::Experimental::spmv_struct(mode,stencil_type,structure,
                                theAlpha,
                                matrix_->lclMatrix_,
                                X.template getLocalView<device_type> (),
                                theBeta,
                                Y.template getLocalView<device_type> (),
                                RANK_ONE);

#endif
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool 
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::hasTransposeApply() const {return true;}


} // namespace Experimental

} // namespace Tpetra

//
// Explicit instantiation macro
//
// Must be expanded from within the Tpetra namespace!
//
#define TPETRA_EXPERIMENTAL_STRUCTUREDCRSWRAPPER_INSTANT(S,LO,GO,NODE) \
  namespace Experimental { \
    template class StructuredCrsWrapper< S, LO, GO, NODE >; \
  }

#endif // TPETRA_EXPERIMENTAL_STRUCTUREDCRSWRAPPER_DEF_HPP
