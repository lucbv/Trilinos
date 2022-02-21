/*@HEADER
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

/// \file Ifpack2_MDF_decl.hpp
/// \brief Declaration of MDF interface

#ifndef IFPACK2_MDF_DECL_HPP
#define IFPACK2_MDF_DECL_HPP

#include "KokkosSparse_spiluk.hpp"

#include "Ifpack2_Preconditioner.hpp"
#include "Ifpack2_Details_CanChangeMatrix.hpp"
#include "Tpetra_CrsMatrix_decl.hpp"
#include "Ifpack2_ScalingType.hpp"
#include "Ifpack2_IlukGraph.hpp"
#include "Ifpack2_LocalSparseTriangularSolver_decl.hpp"

#include <type_traits>

namespace Teuchos {
  class ParameterList; // forward declaration
}

namespace Ifpack2 {

template<class MatrixType>
class MDF:
    virtual public Ifpack2::Preconditioner<typename MatrixType::scalar_type,
                                           typename MatrixType::local_ordinal_type,
                                           typename MatrixType::global_ordinal_type,
                                           typename MatrixType::node_type>,
    virtual public Ifpack2::Details::CanChangeMatrix<Tpetra::RowMatrix<typename MatrixType::scalar_type,
                                                                       typename MatrixType::local_ordinal_type,
                                                                       typename MatrixType::global_ordinal_type,
                                                                       typename MatrixType::node_type> >
{
 public:
  //! The type of the entries of the input MatrixType.
  using scalar_type = typename MatrixType::scalar_type;

  //! The type of local indices in the input MatrixType.
  using local_ordinal_type = typename MatrixType::local_ordinal_type;

  //! The type of global indices in the input MatrixType.
  using global_ordinal_type = typename MatrixType::global_ordinal_type;

  //! The Node type used by the input MatrixType.
  using node_type = typename MatrixType::node_type;

  //! The type of the magnitude (absolute value) of a matrix entry.
  using magnitude_type = typename Teuchos::ScalarTraits<scalar_type>::magnitudeType;

  //! The Kokkos device type of the input MatrixType.
  using device_type = typename node_type::device_type;

  //! The Kokkos execution space of the input MatrixType.
  using execution_space = typename node_type::execution_space;

  //! Tpetra::RowMatrix specialization used by this class.
  using row_matrix_type = Tpetra::RowMatrix<scalar_type,
                                            local_ordinal_type,
                                            global_ordinal_type,
                                            node_type>;


  static_assert(std::is_same<MatrixType, row_matrix_type>::value, "Ifpack2::MDF: The template parameter MatrixType must be a Tpetra::RowMatrix specialization.  Please don't use Tpetra::CrsMatrix (a subclass of Tpetra::RowMatrix) here anymore.");

  //! Tpetra::CrsMatrix specialization used by this class for representing L and U.
  using crs_matrix_type = Tpetra::CrsMatrix<scalar_type,
                                            local_ordinal_type,
                                            global_ordinal_type,
                                            node_type>;

  //! Scalar type stored in Kokkos::Views (CrsMatrix and MultiVector)
  using impl_scalar_type = typename crs_matrix_type::impl_scalar_type;

  template <class NewMatrixType> friend class MDF;

  using global_inds_host_view_type = typename crs_matrix_type::global_inds_host_view_type;
  using local_inds_host_view_type  = typename crs_matrix_type::local_inds_host_view_type;
  using values_host_view_type      = typename crs_matrix_type::values_host_view_type;


  using nonconst_global_inds_host_view_type = typename crs_matrix_type::nonconst_global_inds_host_view_type;
  using nonconst_local_inds_host_view_type  = typename crs_matrix_type::nonconst_local_inds_host_view_type;
  using nonconst_values_host_view_type      = typename crs_matrix_type::nonconst_values_host_view_type;


  //@}
  //! \name Implementation of Kokkos Kernels ILU(k).
  //@{

  using local_matrix_device_type = typename crs_matrix_type::local_matrix_device_type;
  using lno_row_view_t           = typename local_matrix_device_type::StaticCrsGraphType::row_map_type;
  using lno_nonzero_view_t       = typename local_matrix_device_type::StaticCrsGraphType::entries_type;
  using scalar_nonzero_view_t    = typename local_matrix_device_type::values_type;
  using TemporaryMemorySpace     = typename local_matrix_device_type::StaticCrsGraphType::device_type::memory_space;
  using PersistentMemorySpace    = typename local_matrix_device_type::StaticCrsGraphType::device_type::memory_space;
  using HandleExecSpace          = typename local_matrix_device_type::StaticCrsGraphType::device_type::execution_space;
  using kk_handle_type           = typename KokkosKernels::Experimental::KokkosKernelsHandle
    <typename lno_row_view_t::const_value_type,
     typename lno_nonzero_view_t::const_value_type,
     typename scalar_nonzero_view_t::value_type,
     HandleExecSpace,
     TemporaryMemorySpace,
     PersistentMemorySpace>;

  /// \brief Constructor that takes a Tpetra::RowMatrix.
  ///
  /// \param A_in [in] The input matrix.
  MDF (const Teuchos::RCP<const row_matrix_type>& A_in);

  /// \brief Constructor that takes a Tpetra::CrsMatrix.
  ///
  /// This constructor exists to avoid "ambiguous constructor"
  /// warnings.  It does the same thing as the constructor that takes
  /// a Tpetra::RowMatrix.
  ///
  /// \param A_in [in] The input matrix.
  MDF (const Teuchos::RCP<const crs_matrix_type>& A_in);

 private:
  /// \brief Copy constructor: declared private but not defined, so
  ///   that calling it is syntactically forbidden.
  MDF (const MDF<MatrixType> & src);

 public:
  //! Destructor (declared virtual for memory safety).
  virtual ~MDF ();

  /// Set parameters for the incomplete factorization.
  ///
  /// This preconditioner supports the following parameters:
  ///   - "fact: mdf level-of-fill" (int)
  ///   - "fact: absolute threshold" (magnitude_type)
  ///   - "fact: relative threshold" (magnitude_type)
  ///   - "fact: relax value" (magnitude_type)
  ///   - "fact: mdf overalloc" (double)
  void setParameters (const Teuchos::ParameterList& params);

  //! Initialize by computing the symbolic incomplete factorization.
  void initialize ();

  /// \brief Compute the (numeric) incomplete factorization.
  ///
  /// This function computes the MDF(k) factors L and U using the current:
  /// - Ifpack2_IlukGraph specifying the structure of L and U.
  /// - Value for the MDF(k) relaxation parameter.
  /// - Value for the a priori diagonal threshold values.
  ///
  /// initialize() must be called first, before this method may be called.
  void compute ();

  //! Whether initialize() has been called on this object.
  bool isInitialized () const {
    return isInitialized_;
  }
  //! Whether compute() has been called on this object.
  bool isComputed () const {
    return isComputed_;
  }

  //! Number of successful initialize() calls for this object.
  int getNumInitialize () const {
    return numInitialize_;
  }
  //! Number of successful compute() calls for this object.
  int getNumCompute () const {
    return numCompute_;
  }
  //! Number of successful apply() calls for this object.
  int getNumApply () const {
    return numApply_;
  }

  //! Total time in seconds taken by all successful initialize() calls for this object.
  double getInitializeTime () const {
    return initializeTime_;
  }
  //! Total time in seconds taken by all successful compute() calls for this object.
  double getComputeTime () const {
    return computeTime_;
  }
  //! Total time in seconds taken by all successful apply() calls for this object.
  double getApplyTime () const {
    return applyTime_;
  }

  //! Get a rough estimate of cost per iteration
  size_t getNodeSmootherComplexity() const;



  //! \name Implementation of Ifpack2::Details::CanChangeMatrix
  //@{

  /// \brief Change the matrix to be preconditioned.
  ///
  /// \param A [in] The new matrix.
  ///
  /// \post <tt>! isInitialized ()</tt>
  /// \post <tt>! isComputed ()</tt>
  ///
  /// Calling this method resets the preconditioner's state.  After
  /// calling this method with a nonnull input, you must first call
  /// initialize() and compute() (in that order) before you may call
  /// apply().
  ///
  /// You may call this method with a null input.  If A is null, then
  /// you may not call initialize() or compute() until you first call
  /// this method again with a nonnull input.  This method invalidates
  /// any previous factorization whether or not A is null, so calling
  /// setMatrix() with a null input is one way to clear the
  /// preconditioner's state (and free any memory that it may be
  /// using).
  ///
  /// The new matrix A need not necessarily have the same Maps or even
  /// the same communicator as the original matrix.
  virtual void
  setMatrix (const Teuchos::RCP<const row_matrix_type>& A);

  //@}
  //! @name Implementation of Teuchos::Describable interface
  //@{

  //! A one-line description of this object.
  std::string description () const;

  //@}
  //! \name Implementation of Tpetra::Operator
  //@{

  //! Returns the Tpetra::Map object associated with the domain of this operator.
  Teuchos::RCP<const Tpetra::Map<local_ordinal_type,global_ordinal_type,node_type> >
  getDomainMap () const;

  //! Returns the Tpetra::Map object associated with the range of this operator.
  Teuchos::RCP<const Tpetra::Map<local_ordinal_type,global_ordinal_type,node_type> >
  getRangeMap () const;

  /// \brief Apply the (inverse of the) incomplete factorization to X, resulting in Y.
  ///
  /// For an incomplete factorization \f$A \approx LDU\f$, this method
  /// computes the following, depending on the value of \c mode:
  /// <ul>
  /// <li> If mode = Teuchos::NO_TRANS, it computes
  ///      <tt>Y = beta*Y + alpha*(U \ (D \ (L \ X)))</tt> </li>
  /// <li> If mode = Teuchos::TRANS, it computes
  ///      <tt>Y = beta*Y + alpha*(L^T \ (D^T \ (U^T \ X)))</tt> </li>
  /// <li> If mode = Teuchos::CONJ_TRANS, it computes
  ///      <tt>Y = beta*Y + alpha*(L^* \ (D^* \ (U^* \ X)))</tt>,
  ///      where the asterisk indicates the conjugate transpose. </li>
  /// </ul>
  /// If alpha is zero, then the result of applying the operator to a
  /// vector is ignored.  This matters because zero times NaN (not a
  /// number) is NaN, not zero.  Analogously, if beta is zero, then
  /// any values in Y on input are ignored.
  ///
  /// \param X [in] The input multivector.
  ///
  /// \param Y [in/out] The output multivector.
  ///
  /// \param mode [in] If Teuchos::TRANS resp. Teuchos::CONJ_TRANS,
  ///   apply the transpose resp. conjugate transpose of the incomplete
  ///   factorization.  Otherwise, don't apply the tranpose.
  ///
  /// \param alpha [in] Scaling factor for the result of applying the preconditioner.
  ///
  /// \param beta [in] Scaling factor for the initial value of Y.
  void
  apply (const Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type>& X,
         Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type>& Y,
         Teuchos::ETransp mode = Teuchos::NO_TRANS,
         scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one (),
         scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero ()) const;
  //@}

private:
  /// \brief Apply the incomplete factorization (as a product) to X, resulting in Y.
  ///
  /// Given an incomplete factorization is \f$A \approx LDU\f$, this
  /// method computes the following, depending on the value of \c mode:
  ///
  ///   - If mode = Teuchos::NO_TRANS, it computes
  ///     <tt>Y = beta*Y + alpha*(L \ (D \ (U \ X)))</tt>
  ///   - If mode = Teuchos::TRANS, it computes
  ///     <tt>Y = beta*Y + alpha*(U^T \ (D^T \ (L^T \ X)))</tt>
  ///   - If mode = Teuchos::CONJ_TRANS, it computes
  ///     <tt>Y = beta*Y + alpha*(U^* \ (D^* \ (L^* \ X)))</tt>,
  ///     where the asterisk indicates the conjugate transpose.
  ///
  /// \param X [in] The input multivector.
  ///
  /// \param Y [in/out] The output multivector.
  ///
  /// \param mode [in] If Teuchos::TRANS resp. Teuchos::CONJ_TRANS,
  ///   apply the transpose resp. conjugate transpose of the
  ///   incomplete factorization.  Otherwise, don't apply the
  ///   transpose.
  void
  multiply (const Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type>& X,
            Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type>& Y,
            const Teuchos::ETransp mode = Teuchos::NO_TRANS) const;
public:

  //! Get the input matrix.
  Teuchos::RCP<const row_matrix_type> getMatrix () const;

  // Attribute access functions

  //! Get MDF(k) relaxation parameter
  magnitude_type getRelaxValue () const { return RelaxValue_; }

  //! Get absolute threshold value
  magnitude_type getAbsoluteThreshold () const { return Athresh_; }

  //! Get relative threshold value
  magnitude_type getRelativeThreshold () const {return Rthresh_;}

  //! Get level of fill (the "k" in MDF(k)).
  int getLevelOfFill () const { return LevelOfFill_; }

  //! Get overlap mode type
  Tpetra::CombineMode getOverlapMode () {
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, std::logic_error, "Ifpack2::MDF::SetOverlapMode: "
      "RILUK no longer implements overlap on its own.  "
      "Use MDF with AdditiveSchwarz if you want overlap.");
  }

  //! Returns the number of nonzero entries in the global graph.
  Tpetra::global_size_t getGlobalNumEntries () const {
    return getL ().getGlobalNumEntries () + getU ().getGlobalNumEntries ();
  }

  //! Return the Ifpack2::IlukGraph associated with this factored matrix.
  Teuchos::RCP<Ifpack2::IlukGraph<Tpetra::CrsGraph<local_ordinal_type,
                                                   global_ordinal_type,
                                                   node_type>, kk_handle_type> > getGraph () const {
    return Graph_;
  }

  //! Return the L factor of the MDF factorization.
  const crs_matrix_type& getL () const;

  //! Return the diagonal entries of the MDF factorization.
  const Tpetra::Vector<scalar_type,local_ordinal_type,global_ordinal_type,node_type>&
  getD () const;

  //! Return the U factor of the ILU factorization.
  const crs_matrix_type& getU () const;

  //! Return the input matrix A as a Tpetra::CrsMatrix, if possible; else throws.
  Teuchos::RCP<const crs_matrix_type> getCrsMatrix () const;

private:
  using MV  = Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type>;
  using STS = Teuchos::ScalarTraits<scalar_type>;
  using STM = Teuchos::ScalarTraits<magnitude_type>;

  void allocateSolvers ();
  void allocate_L_and_U ();
  static void checkOrderingConsistency (const row_matrix_type& A);
  void initAllValues (const row_matrix_type& A);

  /// \brief Return A, wrapped in a LocalFilter, if necessary.
  ///
  /// "If necessary" means that if A is already a LocalFilter, or if
  /// its communicator only has one process, then we don't need to
  /// wrap it, so we just return A.
  static Teuchos::RCP<const row_matrix_type>
  makeLocalFilter (const Teuchos::RCP<const row_matrix_type>& A);

protected:
  using vec_type = Tpetra::Vector<scalar_type,local_ordinal_type,global_ordinal_type,node_type>;

  //! The (original) input matrix for which to compute MDF(k).
  Teuchos::RCP<const row_matrix_type> A_;

  //! The MDF(k) graph.
  Teuchos::RCP<Ifpack2::IlukGraph<Tpetra::CrsGraph<local_ordinal_type,
                                                   global_ordinal_type,
                                                   node_type>, kk_handle_type> > Graph_;
  /// \brief The matrix whos numbers are used to to compute MDF(k). The graph
  /// may be computed using a crs_matrix_type that initialize() constructs
  /// temporarily.
  Teuchos::RCP<const row_matrix_type> A_local_;
  lno_row_view_t A_local_rowmap_;
  lno_nonzero_view_t A_local_entries_;
  scalar_nonzero_view_t A_local_values_;

  //! The L (lower triangular) factor of MDF(k).
  Teuchos::RCP<crs_matrix_type> L_;
  //! Sparse triangular solver for L
  Teuchos::RCP<LocalSparseTriangularSolver<row_matrix_type> > L_solver_;
  //! The U (upper triangular) factor of MDF(k).
  Teuchos::RCP<crs_matrix_type> U_;
  //! Sparse triangular solver for U
  Teuchos::RCP<LocalSparseTriangularSolver<row_matrix_type> > U_solver_;

  //! The diagonal entries of the MDF(k) factorization.
  Teuchos::RCP<vec_type> D_;

  int LevelOfFill_;
  double Overalloc_;

  bool isAllocated_;
  bool isInitialized_;
  bool isComputed_;

  int numInitialize_;
  int numCompute_;
  mutable int numApply_;

  double initializeTime_;
  double computeTime_;
  mutable double applyTime_;

  magnitude_type RelaxValue_;
  magnitude_type Athresh_;
  magnitude_type Rthresh_;

  //! Optional KokkosKernels implementation.
  bool isKokkosKernelsSpiluk_;
  Teuchos::RCP<kk_handle_type> KernelHandle_;

}; // class MDF

// NOTE (mfh 11 Feb 2015) This used to exist in order to deal with
// different behavior of Tpetra::Crs{Graph,Matrix} for
// KokkosClassic::ThrustGPUNode.  In particular, fillComplete on a
// CrsMatrix used to make the graph go away by default, so we had to
// pass in a parameter to keep a host copy of the graph.  With the new
// (Kokkos refactor) version of Tpetra, this problem has gone away.
namespace detail {
  template<class MatrixType, class NodeType>
  struct setLocalSolveParams{
    static Teuchos::RCP<Teuchos::ParameterList>
    setParams (const Teuchos::RCP<Teuchos::ParameterList>& param) {
      return param;
    }
  };
} // namespace detail

} // namespace Ifpack2

#endif /* IFPACK2_MDF_DECL_HPP */
