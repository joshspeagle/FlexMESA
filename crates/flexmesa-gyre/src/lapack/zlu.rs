// SPDX-License-Identifier: GPL-3.0-only
//! Reference LAPACK 3.12.0 complex LU routines (`SRC/zgetrf.f`,
//! `zgetrf2.f`), double precision, as the MESA SDK `26.6.1` builds them
//! (`gfortran -O2 -fPIC`). `zgetrf2` is the SDK's CRMATH-patched build:
//! the pivot-size test `ABS(A(1,1)) .GE. SFMIN` (`zgetrf2.f:205`) uses
//! CRMATH `abs_c` ([`crmath::abs_c`], `hypot`) instead of libm `cabs`.

use super::zaux::zlaswp;
use super::zblas::{izamax, zgemm, zscal, ztrsm};
use super::{dlamch, ix, LapackError};
use crate::crmath;
use crate::fcomplex::Cplx;

const ONE: Cplx = Cplx::new(1.0, 0.0);

/// `ILAENV(1, 'ZGETRF', ...)` (`SRC/ilaenv.f:280-286`).
const NB_ZGETRF: usize = 64;

/// `ZGETRF2` (`SRC/zgetrf2.f`): recursive LU factorisation with partial
/// pivoting of the `m x n` complex matrix `A`. `ipiv` (length `min(m, n)`)
/// receives 1-based pivot rows. Returns `INFO` (`> 0`: `U(info, info)` is
/// exactly 0).
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] for `lda < max(1, m)`; propagates BLAS
/// errors (none for valid sizes).
pub fn zgetrf2(
    m: usize,
    n: usize,
    a: &mut [Cplx],
    lda: usize,
    ipiv: &mut [usize],
) -> Result<usize, LapackError> {
    if lda < m.max(1) {
        return Err(LapackError::IllegalArgument {
            routine: "ZGETRF2",
            info: -4,
        });
    }
    let mut info = 0;
    if m == 0 || n == 0 {
        return Ok(0);
    }
    if m == 1 {
        // One row case (zgetrf2.f:171-178)
        ipiv[0] = 1;
        if a[0].is_zero() {
            info = 1;
        }
    } else if n == 1 {
        // One column case (zgetrf2.f:180-215)
        let sfmin = dlamch::SFMIN;
        let i = izamax(m, a, 1);
        ipiv[0] = i;
        if a[i - 1].is_zero() {
            info = 1;
        } else {
            if i != 1 {
                a.swap(0, i - 1);
            }
            // SDK: ABS of a complex value is CRMATH abs_c (hypot)
            if crmath::abs_c(a[0]) >= sfmin {
                let r = Cplx::real_div(1.0, a[0]);
                zscal(m - 1, r, &mut a[1..], 1);
            } else {
                for k in 1..m {
                    a[k] = a[k] / a[0];
                }
            }
        }
    } else {
        // Recursive case (zgetrf2.f:221-264)
        let n1 = m.min(n) / 2;
        let n2 = n - n1;
        let iinfo = zgetrf2(m, n1, a, lda, ipiv)?;
        if info == 0 && iinfo > 0 {
            info = iinfo;
        }
        // [A12; A22] row interchanges
        zlaswp(n2, &mut a[ix(1, n1 + 1, lda)..], lda, 1, n1, ipiv, 1);
        // A12 = L11 \ A12 (L11 copied: it is read only and lies in A)
        let l11 = a[..ix(n1, n1, lda) + 1].to_vec();
        ztrsm(
            b'L',
            b'L',
            b'N',
            b'U',
            n1,
            n2,
            ONE,
            &l11,
            lda,
            &mut a[ix(1, n1 + 1, lda)..],
            lda,
        )?;
        // A22 = A22 - A21*A12 (A21, A12 copied)
        let a21 = a[ix(n1 + 1, 1, lda)..].to_vec();
        let a12 = a[ix(1, n1 + 1, lda)..].to_vec();
        zgemm(
            b'N',
            b'N',
            m - n1,
            n2,
            n1,
            -ONE,
            &a21,
            lda,
            &a12,
            lda,
            ONE,
            &mut a[ix(n1 + 1, n1 + 1, lda)..],
            lda,
        )?;
        // Factor A22
        let iinfo = zgetrf2(
            m - n1,
            n2,
            &mut a[ix(n1 + 1, n1 + 1, lda)..],
            lda,
            &mut ipiv[n1..],
        )?;
        if info == 0 && iinfo > 0 {
            info = iinfo + n1;
        }
        for p in &mut ipiv[n1..m.min(n)] {
            *p += n1;
        }
        // Apply interchanges to A21
        zlaswp(n1, a, lda, n1 + 1, m.min(n), ipiv, 1);
    }
    Ok(info)
}

/// `ZGETRF` (`SRC/zgetrf.f`): LU factorisation of the `m x n` complex `A`
/// (leading dimension `lda`); unblocked (`ZGETRF2`) when
/// `NB = 64 >= min(m, n)`. Returns `INFO`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] for `lda < max(1, m)`;
/// [`LapackError::UnportedPath`] for the blocked path (`min(m, n) > 64`,
/// lines 174-216).
pub fn zgetrf(
    m: usize,
    n: usize,
    a: &mut [Cplx],
    lda: usize,
    ipiv: &mut [usize],
) -> Result<usize, LapackError> {
    if lda < m.max(1) {
        return Err(LapackError::IllegalArgument {
            routine: "ZGETRF",
            info: -4,
        });
    }
    if m == 0 || n == 0 {
        return Ok(0);
    }
    if NB_ZGETRF <= 1 || NB_ZGETRF >= m.min(n) {
        zgetrf2(m, n, a, lda, ipiv)
    } else {
        Err(LapackError::UnportedPath {
            routine: "ZGETRF",
            lines: "174-216",
        })
    }
}
