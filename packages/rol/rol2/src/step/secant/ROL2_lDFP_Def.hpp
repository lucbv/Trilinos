// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL2) Package
//                 Copyright (2014) Sandia Corporation
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
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

#pragma once 
#ifndef ROL2_LDFP_DEF_HPP
#define ROL2_LDFP_DEF_HPP

namespace ROL2 {


// Apply lDFP Approximate Inverse Hessian
template<class Real>
void lDFP<Real>::applyH(       Vector<Real>& Hv, 
                         const Vector<Real>& v ) const {

  auto& state = Secant<Real>::getState();

  const Real one(1);

  // Apply initial Hessian approximation to v
  applyH0(Hv,v);

  std::vector<Ptr<Vector<Real>>> a(state.current+1),  b(state.current+1);

  Real bv(0), av(0), bs(0), as(0);

  for (int i = 0; i <= state.current; i++) {
    b[i] = Hv.clone();
    b[i]->set(*(state.iterDiff[i]));
    b[i]->scale(1.0/sqrt(state.product[i]));
    bv = b[i]->apply(v);
    Hv.axpy(bv,*b[i]);

    a[i] = Hv.clone();
    applyH0(*a[i],*(state.gradDiff[i]));

    for (int j = 0; j < i; j++) {
      bs = b[j]->apply(*(state.gradDiff[i]));
      a[i]->axpy(bs,*b[j]);
      as = a[j]->apply(*(state.gradDiff[i]));
      a[i]->axpy(-as,*a[j]);
    }

    as = a[i]->apply(*(state.gradDiff[i]));
    a[i]->scale(one/sqrt(as));
    av = a[i]->apply(v);
    Hv.axpy(-av,*a[i]);
  }
} // lDFP<Real>::applyH



// Apply Initial lDFP Approximate Inverse Hessian
template<class Real>
void lDFP<Real>::applyH0(       Vector<Real>& Hv, 
                          const Vector<Real>& v ) const  { 
  auto& state = Secant<Real>::getState();
  Hv.set(v.dual());
  if (useDefaultScaling_) {
    if (state.iter != 0 && state.current != -1) {
      Real ss = state.iterDiff[state.current]->dot(*(state.iterDiff[state.current]));
      Hv.scale(state.product[state.current]/ss);
    }
  }
  else Hv.scale(static_cast<Real>(1)/Bscaling_);

} // lDFP<Real>::applyH0


//// Apply lDFP Approximate Inverse Hessian
//template<class Real>
//void lDFP<Real>::applyH(       Vector<Real>& Hv, 
//                         const Vector<Real> &v ) const {
//  auto& state = Secant<Real>::getState();
//    const Real zero(0);
//
//  Bv.set(v.dual());
//  std::vector<Real> alpha(state.current+1,zero);
//
//  for (int i = state.current; i>=0; i--) {
//    alpha[i]  = state.gradDiff[i]->dot(Bv);
//    alpha[i] /= state.product[i];
//    Bv.axpy(-alpha[i],(state.iterDiff[i])->dual());
//  }
//
//  // Apply initial inverse Hessian approximation to v
//  Ptr<Vector<Real>> tmp = Bv.clone();
//  applyB0(*tmp,Bv.dual());
//  Bv.set(*tmp);
//
//  Real beta(0);
//  for (int i = 0; i <= state.current; i++) {
//    beta  = state.iterDiff[i]->apply(Bv);
//    beta /= state.product[i];
//    Bv.axpy((alpha[i]-beta),*(state.gradDiff[i]));
//  }
//} // lDFP<Real>::applyH


// Apply lDFP Approximate Hessian
template<class Real>
void lDFP<Real>::applyB(       Vector<Real>& Bv, 
                         const Vector<Real>& v ) const {

  auto& state = Secant<Real>::getState();
  const Real zero(0);

  Bv.set(v.dual());
  std::vector<Real> alpha(state.current+1,zero);

  for (int i = state.current; i>=0; i--) {
    alpha[i]  = state.gradDiff[i]->dot(Bv);
    alpha[i] /= state.product[i];
    Bv.axpy(-alpha[i],(state.iterDiff[i])->dual());
  }

  // Apply initial inverse Hessian approximation to v
  Ptr<Vector<Real>> tmp = Bv.clone();
  applyB0(*tmp,Bv.dual());
  Bv.set(*tmp);

  Real beta(0);

  for (int i = 0; i <= state.current; i++) {
    beta  = state.iterDiff[i]->apply(Bv);
    beta /= state.product[i];
    Bv.axpy((alpha[i]-beta),*(state.gradDiff[i]));
  }
} // lDFP<Real>::applyB


// Apply Initial lDFP Approximate Hessian 
template<class Real>
void lDFP<Real>::applyB0(       Vector<Real>& Bv, 
                          const Vector<Real>& v ) const {

  auto& state = Secant<Real>::getState();
  Bv.set(v.dual());
  if (useDefaultScaling_) {
    if (state.iter != 0 && state.current != -1) {
      Real ss = state.iterDiff[state.current]->dot(*(state.iterDiff[state.current]));
      Bv.scale(ss/state.product[state.current]);
    }
  }
  else Bv.scale(Bscaling_);

} // lDFP<Real>::applyB0

} // namespace ROL2

#endif // ROL2_LDFP_DEF_HPP
