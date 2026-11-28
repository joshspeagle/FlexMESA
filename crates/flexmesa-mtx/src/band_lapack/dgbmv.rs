// SPDX-License-Identifier: GPL-3.0-or-later
//! `DGBMV`: banded matrix-vector product.

use crate::lapack::{ix, lsame, LapackError};

fn bad(info: i32) -> LapackError {
    LapackError::IllegalArgument {
        routine: "DGBMV",
        info,
    }
}

/// `DGBMV` (`BLAS/SRC/dgbmv.f:186-370`): `y := alpha*A*x + beta*y` or
/// `y := alpha*A**T*x + beta*y` for an `m`-by-`n` band matrix with `kl`
/// sub-diagonals and `ku` super-diagonals stored as `A(i,j)` at
/// `AB(ku+1+i-j,j)`, `lda >= kl+ku+1`.
///
/// `x` and `y` are vectors with positive strides `incx`, `incy` starting at
/// element 1; the negative-stride start points (`dgbmv.f:264-273`) are not
/// reachable with `usize` strides.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] with upstream's `INFO`
/// (`dgbmv.f:225-248`) for a bad `trans`, `lda < kl+ku+1` or a zero stride.
pub fn dgbmv(
    trans: u8,
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
    incy: usize,
) -> Result<(), LapackError> {
    // dgbmv.f:225-248
    if !lsame(trans, b'N') && !lsame(trans, b'T') && !lsame(trans, b'C') {
        return Err(bad(1));
    } else if lda < kl + ku + 1 {
        return Err(bad(8));
    } else if incx == 0 {
        return Err(bad(10));
    } else if incy == 0 {
        return Err(bad(13));
    }
    // dgbmv.f:251-252
    if m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0) {
        return Ok(());
    }
    // dgbmv.f:257-263
    let leny = if lsame(trans, b'N') { m } else { n };
    // First elements (dgbmv.f:264-273, positive strides): KX = KY = 1.
    // Scale y first (dgbmv.f:280-305).
    if beta != 1.0 {
        for i in 0..leny {
            let iy = i * incy;
            y[iy] = if beta == 0.0 { 0.0 } else { beta * y[iy] };
        }
    }
    // dgbmv.f:306
    if alpha == 0.0 {
        return Ok(());
    }
    let kup1 = ku + 1;
    if lsame(trans, b'N') {
        // Form y := alpha*A*x + y (dgbmv.f:308-334).
        let mut jx = 0usize;
        let mut ky = 0usize;
        for j in 1..=n {
            let temp = alpha * x[jx];
            // K = KUP1 - J may be negative: A(K+I,J) is A(i + kup1 - j, j).
            let lo = 1.max(j.saturating_sub(ku));
            let hi = m.min(j + kl);
            let mut iy = ky;
            for i in lo..=hi {
                let row = i + kup1 - j;
                y[iy] += temp * a[ix(row, j, lda)];
                iy += incy;
            }
            jx += incx;
            if j > ku {
                ky += incy;
            }
        }
    } else {
        // Form y := alpha*A**T*x + y (dgbmv.f:335-364).
        let mut jy = 0usize;
        let mut kx = 0usize;
        for j in 1..=n {
            let mut temp = 0.0_f64;
            let lo = 1.max(j.saturating_sub(ku));
            let hi = m.min(j + kl);
            let mut ixx = kx;
            for i in lo..=hi {
                let row = i + kup1 - j;
                temp += a[ix(row, j, lda)] * x[ixx];
                ixx += incx;
            }
            y[jy] += alpha * temp;
            jy += incy;
            if j > ku {
                kx += incx;
            }
        }
    }
    Ok(())
}
