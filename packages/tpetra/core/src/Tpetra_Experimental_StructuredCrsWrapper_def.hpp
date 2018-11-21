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
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Import.hpp"

namespace Tpetra {

namespace Experimental {

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::StructuredCrsWrapper(Teuchos::RCP<crs_matrix_type> matrix, const Teuchos::RCP<Teuchos::ParameterList>& params):matrix_(matrix) {

}


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > 
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getDomainMap() const { return matrix_->getDomainMap(); }
  

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Map<LocalOrdinal,GlobalOrdinal,Node> > 
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getRangeMap() const {return matrix_->getRangeMap();}


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::apply (const MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &X,
                                                                     MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node> &Y,
                                                                     Teuchos::ETransp mode,
                                                                     Scalar alpha,
                                                                     Scalar beta) const {

}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool 
StructuredCrsWrapper<Scalar,LocalOrdinal,GlobalOrdinal,Node>::hasTransposeApply() const {return false;}


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
