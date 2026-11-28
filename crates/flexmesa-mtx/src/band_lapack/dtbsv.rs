// SPDX-License-Identifier: GPL-3.0-or-later
//! `DTBSV`: solve a banded triangular system.

use crate::lapack::{ix, lsame, LapackError};

fn bad(info: i32) -> LapackError {
    LapackError::IllegalArgument {
        routine: "DTBSV",
        info,
    }
}

/// `DTBSV` (`BLAS/SRC/dtbsv.f:188-400`): `x := inv(A)*x` or
/// `x := inv(A**T)*x` for an `n`-by-`n` triangular band matrix with `k`
/// off-diagonals, stored as `A(i,j)` at `AB(k+1+i-j,j)` (upper) or
/// `AB(1+i-j,j)` (lower), `lda >= k+1`.
///
/// `x` is a vector with positive stride `incx` starting at `X(1)`. The
/// `INCX < 0` start-point branch (`dtbsv.f:259-260`) is not reachable with a
/// `usize` stride.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] with upstream's `INFO` (`dtbsv.f:226-248`)
/// for a bad `uplo`, `trans`, `diag`, `lda < k+1` or `incx == 0`.
#[allow(clippy::many_single_char_names)]
pub fn dtbsv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    x: &mut [f64],
    incx: usize,
) -> Result<(), LapackError> {
    // dtbsv.f:226-248
    if !lsame(uplo, b'U') && !lsame(uplo, b'L') {
        return Err(bad(1));
    } else if !lsame(trans, b'N') && !lsame(trans, b'T') && !lsame(trans, b'C')
    {
        return Err(bad(2));
    } else if !lsame(diag, b'U') && !lsame(diag, b'N') {
        return Err(bad(3));
    } else if lda < k + 1 {
        return Err(bad(7));
    } else if incx == 0 {
        return Err(bad(9));
    }
    // dtbsv.f:252
    if n == 0 {
        return Ok(());
    }
    let nounit = lsame(diag, b'N');
    // 1-based, signed Fortran indices into X (dtbsv.f:261-262).
    let inc = incx as isize;
    let nn = n as isize;
    let kk = k as isize;
    let mut kx: isize = 1;
    let at = |i: isize| (i - 1) as usize;
    let amat = |i: usize, j: usize| a[ix(i, j, lda)];

    if lsame(trans, b'N') {
        // Form x := inv(A)*x (dtbsv.f:268-332)
        if lsame(uplo, b'U') {
            let kplus1 = k + 1;
            if incx == 1 {
                // dtbsv.f:274-284
                for j in (1..=n).rev() {
                    if x[j - 1] != 0.0 {
                        // L = KPLUS1 - J, A(L+I,J)
                        if nounit {
                            x[j - 1] /= amat(kplus1, j);
                        }
                        let temp = x[j - 1];
                        let lo = 1.max(j.saturating_sub(k));
                        for i in (lo..j).rev() {
                            x[i - 1] -= temp * amat(kplus1 + i - j, j);
                        }
                    }
                }
            } else {
                // dtbsv.f:286-302
                kx += (nn - 1) * inc;
                let mut jx = kx;
                for j in (1..=n).rev() {
                    kx -= inc;
                    if x[at(jx)] != 0.0 {
                        let mut ixx = kx;
                        if nounit {
                            x[at(jx)] /= amat(kplus1, j);
                        }
                        let temp = x[at(jx)];
                        let lo = 1.max(j.saturating_sub(k));
                        for i in (lo..j).rev() {
                            x[at(ixx)] -= temp * amat(kplus1 + i - j, j);
                            ixx -= inc;
                        }
                    }
                    jx -= inc;
                }
            }
        } else if incx == 1 {
            // dtbsv.f:304-314
            for j in 1..=n {
                if x[j - 1] != 0.0 {
                    // L = 1 - J, A(L+I,J)
                    if nounit {
                        x[j - 1] /= amat(1, j);
                    }
                    let temp = x[j - 1];
                    for i in j + 1..=n.min(j + k) {
                        x[i - 1] -= temp * amat(1 + i - j, j);
                    }
                }
            }
        } else {
            // dtbsv.f:316-332
            let mut jx = kx;
            for j in 1..=n {
                kx += inc;
                if x[at(jx)] != 0.0 {
                    let mut ixx = kx;
                    if nounit {
                        x[at(jx)] /= amat(1, j);
                    }
                    let temp = x[at(jx)];
                    for i in j + 1..=n.min(j + k) {
                        x[at(ixx)] -= temp * amat(1 + i - j, j);
                        ixx += inc;
                    }
                }
                jx += inc;
            }
        }
    } else if lsame(uplo, b'U') {
        // Form x := inv(A**T)*x, upper (dtbsv.f:337-364)
        let kplus1 = k + 1;
        if incx == 1 {
            // dtbsv.f:339-348
            for j in 1..=n {
                let mut temp = x[j - 1];
                for i in 1.max(j.saturating_sub(k))..j {
                    temp -= amat(kplus1 + i - j, j) * x[i - 1];
                }
                if nounit {
                    temp /= amat(kplus1, j);
                }
                x[j - 1] = temp;
            }
        } else {
            // dtbsv.f:350-363
            let mut jx = kx;
            for j in 1..=n {
                let mut temp = x[at(jx)];
                let mut ixx = kx;
                for i in 1.max(j.saturating_sub(k))..j {
                    temp -= amat(kplus1 + i - j, j) * x[at(ixx)];
                    ixx += inc;
                }
                if nounit {
                    temp /= amat(kplus1, j);
                }
                x[at(jx)] = temp;
                jx += inc;
                if j > k {
                    kx += inc;
                }
            }
        }
    } else if incx == 1 {
        // Form x := inv(A**T)*x, lower (dtbsv.f:366-375)
        for j in (1..=n).rev() {
            let mut temp = x[j - 1];
            for i in (j + 1..=n.min(j + k)).rev() {
                temp -= amat(1 + i - j, j) * x[i - 1];
            }
            if nounit {
                temp /= amat(1, j);
            }
            x[j - 1] = temp;
        }
    } else {
        // dtbsv.f:377-391
        kx += (nn - 1) * inc;
        let mut jx = kx;
        for j in (1..=n).rev() {
            let mut temp = x[at(jx)];
            let mut ixx = kx;
            for i in (j + 1..=n.min(j + k)).rev() {
                temp -= amat(1 + i - j, j) * x[at(ixx)];
                ixx -= inc;
            }
            if nounit {
                temp /= amat(1, j);
            }
            x[at(jx)] = temp;
            jx -= inc;
            if nn - (j as isize) >= kk {
                kx -= inc;
            }
        }
    }
    Ok(())
}
