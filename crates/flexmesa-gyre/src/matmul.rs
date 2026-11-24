// SPDX-License-Identifier: GPL-3.0-only
//! Fortran `MATMUL` as gfortran 15.2.0 inlines it at `-O2`.
//!
//! With front-end optimization on (`-O2`), gfortran replaces
//! `C = MATMUL(A, B)` by a loop nest when `m*k*n <= 27000`
//! (`-finline-matmul-limit=30`, the product evaluated in `REAL(4)`), and
//! calls `_gfortran_matmul_r8`/`_c8` otherwise. The inlined nest, read from
//! `-fdump-tree-original`, is
//!
//! ```text
//! C = 0
//! do j = 1, n; do l = 1, k; do i = 1, m
//!    C(i,j) = C(i,j) + A(i,l)*B(l,j)
//! ```
//!
//! so every element accumulates over `l` in increasing order from `+0`.
//! `y = MATMUL(A, x)` is always inlined: `y = 0`, then
//! `do l; do i; y(i) = y(i) + A(i,l)*x(l)`. Complex products use the
//! gfortran expansion of [`crate::fcomplex`].
//!
//! The library path is never taken by the GYRE sizes MESA reaches
//! (`n_e <= 6`, blocks up to 12); requesting it returns an error rather
//! than a different summation order.
//!
//! All matrices are column-major slices (`A(i,l)` at `i + l*m`).

use crate::fcomplex::Cplx;
use crate::GyreNumError;

/// Inline limit test (`(real(m)*real(n))*real(k) <= 2.7e4` in `REAL(4)`).
#[allow(clippy::cast_precision_loss)]
fn check_inline(m: usize, k: usize, n: usize) -> Result<(), GyreNumError> {
    let p = (m as f32 * n as f32) * k as f32;
    if p <= 2.7e4_f32 {
        Ok(())
    } else {
        Err(GyreNumError::MatmulLibraryPath { m, k, n })
    }
}

/// `C = MATMUL(A, B)` for real `A (m x k)`, `B (k x n)`.
///
/// # Errors
///
/// [`GyreNumError::MatmulLibraryPath`] when the sizes exceed the inline
/// limit.
///
/// # Panics
///
/// Panics if the slice lengths do not match the dimensions.
pub fn matmul_rr(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f64>, GyreNumError> {
    assert_eq!(a.len(), m * k, "A has wrong length");
    assert_eq!(b.len(), k * n, "B has wrong length");
    check_inline(m, k, n)?;
    let mut c = vec![0.0_f64; m * n];
    for j in 0..n {
        for l in 0..k {
            let blj = b[l + j * k];
            for i in 0..m {
                c[i + j * m] += a[i + l * m] * blj;
            }
        }
    }
    Ok(c)
}

/// `C = MATMUL(A, B)` for complex `A (m x k)`, `B (k x n)`.
///
/// # Errors
///
/// [`GyreNumError::MatmulLibraryPath`] when the sizes exceed the inline
/// limit.
///
/// # Panics
///
/// Panics if the slice lengths do not match the dimensions.
pub fn matmul_cc(
    a: &[Cplx],
    b: &[Cplx],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<Cplx>, GyreNumError> {
    assert_eq!(a.len(), m * k, "A has wrong length");
    assert_eq!(b.len(), k * n, "B has wrong length");
    check_inline(m, k, n)?;
    let mut c = vec![Cplx::new(0.0, 0.0); m * n];
    for j in 0..n {
        for l in 0..k {
            let blj = b[l + j * k];
            for i in 0..m {
                c[i + j * m] = c[i + j * m] + a[i + l * m] * blj;
            }
        }
    }
    Ok(c)
}

/// `y = MATMUL(A, x)` for real `A (m x k)`.
///
/// # Panics
///
/// Panics if the slice lengths do not match the dimensions.
#[must_use]
pub fn matvec_r(a: &[f64], x: &[f64], m: usize, k: usize) -> Vec<f64> {
    assert_eq!(a.len(), m * k, "A has wrong length");
    assert_eq!(x.len(), k, "x has wrong length");
    let mut y = vec![0.0_f64; m];
    for (l, &xl) in x.iter().enumerate() {
        for i in 0..m {
            y[i] += a[i + l * m] * xl;
        }
    }
    y
}

/// `y = MATMUL(A, x)` for complex `A (m x k)`.
///
/// # Panics
///
/// Panics if the slice lengths do not match the dimensions.
#[must_use]
pub fn matvec_c(a: &[Cplx], x: &[Cplx], m: usize, k: usize) -> Vec<Cplx> {
    assert_eq!(a.len(), m * k, "A has wrong length");
    assert_eq!(x.len(), k, "x has wrong length");
    let mut y = vec![Cplx::new(0.0, 0.0); m];
    for (l, &xl) in x.iter().enumerate() {
        for i in 0..m {
            y[i] = y[i] + a[i + l * m] * xl;
        }
    }
    y
}
