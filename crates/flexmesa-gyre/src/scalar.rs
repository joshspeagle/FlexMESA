// SPDX-License-Identifier: GPL-3.0-only
//! The two element types of GYRE's `${T}$` templates: `real(RD)` (`rt`
//! instances) and `complex(RD)` (`ct` instances), with their extended types
//! (`ext_rt`, `ext_ct`) and the LAPACK routines each instance calls.
//!
//! Every operation is the one gfortran 15.2.0 emits for that instance: for
//! `complex(RD)`, mixed real/complex operands widen the real one to
//! `(r, +0)` and keep the full complex operation (see
//! [`crate::fcomplex`]), and `abs` is GYRE `math_m`'s CRMATH `abs`.

use std::fmt::Debug;

use crate::crmath::abs_c;
use crate::ext_complex::{self, ExtC};
use crate::ext_real::{self, ExtR};
use crate::fcomplex::Cplx;
use crate::lapack::{aux, blas, lu, svd, zaux, zblas, zlu, zsvd, LapackError};

/// An element type of GYRE's templated matrices.
pub trait Scalar: Copy + Debug + PartialEq {
    /// The matching extended-exponent type.
    type Ext: Copy + Debug;

    /// `0._RD` in this type.
    fn zero() -> Self;
    /// `1._RD` in this type.
    fn one() -> Self;
    /// `T(r)`: a real constant or real-valued expression in this type.
    fn from_r(r: f64) -> Self;
    /// `a + b`.
    #[must_use]
    fn add(self, b: Self) -> Self;
    /// `a - b`.
    #[must_use]
    fn sub(self, b: Self) -> Self;
    /// `a*b`.
    #[must_use]
    fn mul(self, b: Self) -> Self;
    /// `-a`.
    #[must_use]
    fn neg(self) -> Self;
    /// `r*a` with `r` real.
    fn real_mul(r: f64, a: Self) -> Self;
    /// `a/r` with `r` real.
    #[must_use]
    fn div_r(self, r: f64) -> Self;
    /// GYRE `math_m` `abs` (CRMATH `abs` for complex values).
    fn abs_m(self) -> f64;
    /// `CONJG` (identity for reals).
    #[must_use]
    fn conj(self) -> Self;

    /// `ext_T(a)`.
    fn ext(self) -> Self::Ext;
    /// `ext_T(r)` for a real `r`.
    fn ext_r(r: f64) -> Self::Ext;
    /// `x*y` of extended values.
    fn ext_mul(x: Self::Ext, y: Self::Ext) -> Self::Ext;
    /// `-x`.
    fn ext_neg(x: Self::Ext) -> Self::Ext;
    /// `product(x)`.
    fn ext_product(x: &[Self::Ext]) -> Self::Ext;

    /// `XGETRF(m, n, A, lda, ipiv, info)`.
    ///
    /// # Errors
    ///
    /// LAPACK argument errors.
    fn getrf(
        m: usize,
        n: usize,
        a: &mut [Self],
        lda: usize,
        ipiv: &mut [usize],
    ) -> Result<usize, LapackError>;
    /// `XLASWP_inplace(n, A, lda, k1, k2, ipiv, 1)`.
    fn laswp(
        n: usize,
        a: &mut [Self],
        lda: usize,
        k1: usize,
        k2: usize,
        ipiv: &[usize],
    );
    /// `XTRSM_inplace('L', uplo, 'N', diag, m, n, ONE, A, lda, B, ldb)`.
    ///
    /// # Errors
    ///
    /// LAPACK argument errors.
    #[allow(clippy::too_many_arguments)]
    fn trsm_l(
        uplo: u8,
        diag: u8,
        m: usize,
        n: usize,
        a: &[Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
    ) -> Result<(), LapackError>;
    /// `XGEMM_inplace('N', 'N', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)`.
    ///
    /// # Errors
    ///
    /// LAPACK argument errors.
    #[allow(clippy::too_many_arguments)]
    fn gemm_nn(
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) -> Result<(), LapackError>;
    /// `XGEMV_inplace('N', m, n, alpha, A, lda, x, 1, beta, y, 1)`.
    ///
    /// # Errors
    ///
    /// LAPACK argument errors.
    #[allow(clippy::too_many_arguments)]
    fn gemv_n(
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        beta: Self,
        y: &mut [Self],
    ) -> Result<(), LapackError>;
    /// `sing_decompose(A, sigma, U, V_H)` (`linalg_m.fypp:703-768`): the
    /// right singular vectors `V_H` of a square `n x n` matrix, column-major,
    /// and `INFO`.
    ///
    /// # Errors
    ///
    /// LAPACK errors.
    fn sing_decompose_vh(
        n: usize,
        a: &mut [Self],
    ) -> Result<(Vec<Self>, usize), LapackError>;
}

impl Scalar for f64 {
    type Ext = ExtR;

    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_r(r: f64) -> Self {
        r
    }
    fn add(self, b: Self) -> Self {
        self + b
    }
    fn sub(self, b: Self) -> Self {
        self - b
    }
    fn mul(self, b: Self) -> Self {
        self * b
    }
    fn neg(self) -> Self {
        -self
    }
    fn real_mul(r: f64, a: Self) -> Self {
        r * a
    }
    fn div_r(self, r: f64) -> Self {
        self / r
    }
    fn abs_m(self) -> f64 {
        self.abs()
    }
    fn conj(self) -> Self {
        self
    }
    fn ext(self) -> ExtR {
        ExtR::from_real(self)
    }
    fn ext_r(r: f64) -> ExtR {
        ExtR::from_real(r)
    }
    fn ext_mul(x: ExtR, y: ExtR) -> ExtR {
        x.times(y)
    }
    fn ext_neg(x: ExtR) -> ExtR {
        x.negate()
    }
    fn ext_product(x: &[ExtR]) -> ExtR {
        ext_real::product(x)
    }
    fn getrf(
        m: usize,
        n: usize,
        a: &mut [Self],
        lda: usize,
        ipiv: &mut [usize],
    ) -> Result<usize, LapackError> {
        lu::dgetrf(m, n, a, lda, ipiv)
    }
    fn laswp(
        n: usize,
        a: &mut [Self],
        lda: usize,
        k1: usize,
        k2: usize,
        ipiv: &[usize],
    ) {
        aux::dlaswp(n, a, lda, k1, k2, ipiv, 1);
    }
    fn trsm_l(
        uplo: u8,
        diag: u8,
        m: usize,
        n: usize,
        a: &[Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
    ) -> Result<(), LapackError> {
        blas::dtrsm(b'L', uplo, b'N', diag, m, n, 1.0, a, lda, b, ldb)
    }
    fn gemm_nn(
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) -> Result<(), LapackError> {
        blas::dgemm(b'N', b'N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }
    fn gemv_n(
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        beta: Self,
        y: &mut [Self],
    ) -> Result<(), LapackError> {
        blas::dgemv(b'N', m, n, alpha, a, lda, x, 1, beta, y, 1)
    }
    fn sing_decompose_vh(
        n: usize,
        a: &mut [Self],
    ) -> Result<(Vec<Self>, usize), LapackError> {
        // sing_decompose_r_: XGESVD('A','A',n,n,A,n,sigma,U,n,V_H,n,work,5n,info)
        let mut u = vec![0.0; n * n];
        let mut vt = vec![0.0; n * n];
        let o = svd::dgesvd(n, n, a, &mut u, &mut vt, 5 * n)?;
        Ok((vt, o.info))
    }
}

impl Scalar for Cplx {
    type Ext = ExtC;

    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
    fn one() -> Self {
        Self::new(1.0, 0.0)
    }
    fn from_r(r: f64) -> Self {
        Self::from_real(r)
    }
    fn add(self, b: Self) -> Self {
        self + b
    }
    fn sub(self, b: Self) -> Self {
        self - b
    }
    fn mul(self, b: Self) -> Self {
        self * b
    }
    fn neg(self) -> Self {
        -self
    }
    fn real_mul(r: f64, a: Self) -> Self {
        Self::real_mul(r, a)
    }
    fn div_r(self, r: f64) -> Self {
        self.div_real(r)
    }
    fn abs_m(self) -> f64 {
        abs_c(self)
    }
    fn conj(self) -> Self {
        Self::conj(self)
    }
    fn ext(self) -> ExtC {
        ExtC::from_cplx(self)
    }
    fn ext_r(r: f64) -> ExtC {
        ExtC::from_real(r)
    }
    fn ext_mul(x: ExtC, y: ExtC) -> ExtC {
        x.times(y)
    }
    fn ext_neg(x: ExtC) -> ExtC {
        x.negate()
    }
    fn ext_product(x: &[ExtC]) -> ExtC {
        ext_complex::product(x)
    }
    fn getrf(
        m: usize,
        n: usize,
        a: &mut [Self],
        lda: usize,
        ipiv: &mut [usize],
    ) -> Result<usize, LapackError> {
        zlu::zgetrf(m, n, a, lda, ipiv)
    }
    fn laswp(
        n: usize,
        a: &mut [Self],
        lda: usize,
        k1: usize,
        k2: usize,
        ipiv: &[usize],
    ) {
        zaux::zlaswp(n, a, lda, k1, k2, ipiv, 1);
    }
    fn trsm_l(
        uplo: u8,
        diag: u8,
        m: usize,
        n: usize,
        a: &[Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
    ) -> Result<(), LapackError> {
        zblas::ztrsm(b'L', uplo, b'N', diag, m, n, Self::one(), a, lda, b, ldb)
    }
    fn gemm_nn(
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        beta: Self,
        c: &mut [Self],
        ldc: usize,
    ) -> Result<(), LapackError> {
        zblas::zgemm(b'N', b'N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }
    fn gemv_n(
        m: usize,
        n: usize,
        alpha: Self,
        a: &[Self],
        lda: usize,
        x: &[Self],
        beta: Self,
        y: &mut [Self],
    ) -> Result<(), LapackError> {
        zblas::zgemv(b'N', m, n, alpha, a, lda, x, 1, beta, y, 1)
    }
    fn sing_decompose_vh(
        n: usize,
        a: &mut [Self],
    ) -> Result<(Vec<Self>, usize), LapackError> {
        // sing_decompose_c_: XGESVD('A','A',n,n,A,n,sigma,U,n,V_H,n,work,3n,rwork,info)
        let mut u = vec![Self::zero(); n * n];
        let mut vt = vec![Self::zero(); n * n];
        let o = zsvd::zgesvd(n, n, a, &mut u, &mut vt, 3 * n)?;
        Ok((vt, o.info))
    }
}
