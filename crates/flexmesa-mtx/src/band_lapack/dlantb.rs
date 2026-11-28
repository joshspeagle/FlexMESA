// SPDX-License-Identifier: GPL-3.0-or-later
//! `DLANTB`: norm of a triangular band matrix.

use crate::lapack::{ix, lsame, LapackError};

/// `DLANTB` (`SRC/dlantb.f:138-358`): the `'M'`, `'1'`/`'O'` and `'I'` norms
/// of an `n`-by-`n` triangular band matrix with `k` off-diagonals, stored as
/// `A(i,j)` at `AB(k+1+i-j,j)` (upper) or `AB(1+i-j,j)` (lower), `ldab >=
/// k+1`. `work` (length at least `n`) is used only for `'I'`.
///
/// # Errors
///
/// - [`LapackError::UnportedPath`] for `'F'`/`'E'` (`dlantb.f:308-351`,
///   which needs `DLASSQ`; not reached by `DGBSVX`).
/// - [`LapackError::IllegalArgument`] (`info = -1`) for any other `norm`.
///   Upstream returns `VALUE` unset there (`dlantb.f:176-353`), the same
///   unset-read pattern as `DLANGB` (Deferred bug #689, which names only
///   `dlangb.f`; `DGBSVX` always passes `'M'`).
pub fn dlantb(
    norm: u8,
    uplo: u8,
    diag: u8,
    n: usize,
    k: usize,
    ab: &[f64],
    ldab: usize,
    work: &mut [f64],
) -> Result<f64, LapackError> {
    let upper = lsame(uplo, b'U');
    let unit = lsame(diag, b'U');
    let mut value;
    if n == 0 {
        // dlantb.f:176-177
        value = 0.0_f64;
    } else if lsame(norm, b'M') {
        // Find max(abs(A(i,j))) (dlantb.f:178-216).
        let upd = |value: &mut f64, sum: f64| {
            if *value < sum || sum.is_nan() {
                *value = sum;
            }
        };
        if unit {
            value = 1.0;
            if upper {
                for j in 1..=n {
                    for i in (k + 2).saturating_sub(j).max(1)..=k {
                        upd(&mut value, ab[ix(i, j, ldab)].abs());
                    }
                }
            } else {
                for j in 1..=n {
                    for i in 2..=(n + 1 - j).min(k + 1) {
                        upd(&mut value, ab[ix(i, j, ldab)].abs());
                    }
                }
            }
        } else {
            value = 0.0;
            if upper {
                for j in 1..=n {
                    for i in (k + 2).saturating_sub(j).max(1)..=k + 1 {
                        upd(&mut value, ab[ix(i, j, ldab)].abs());
                    }
                }
            } else {
                for j in 1..=n {
                    for i in 1..=(n + 1 - j).min(k + 1) {
                        upd(&mut value, ab[ix(i, j, ldab)].abs());
                    }
                }
            }
        }
    } else if lsame(norm, b'O') || norm == b'1' {
        // Find norm1(A) (dlantb.f:217-253).
        value = 0.0;
        for j in 1..=n {
            let mut sum;
            if upper {
                let lo = (k + 2).saturating_sub(j).max(1);
                if unit {
                    sum = 1.0;
                    for i in lo..=k {
                        sum += ab[ix(i, j, ldab)].abs();
                    }
                } else {
                    sum = 0.0;
                    for i in lo..=k + 1 {
                        sum += ab[ix(i, j, ldab)].abs();
                    }
                }
            } else if unit {
                sum = 1.0;
                for i in 2..=(n + 1 - j).min(k + 1) {
                    sum += ab[ix(i, j, ldab)].abs();
                }
            } else {
                sum = 0.0;
                for i in 1..=(n + 1 - j).min(k + 1) {
                    sum += ab[ix(i, j, ldab)].abs();
                }
            }
            if value < sum || sum.is_nan() {
                value = sum;
            }
        }
    } else if lsame(norm, b'I') {
        // Find normI(A) (dlantb.f:254-307).
        value = 0.0;
        let init = if unit { 1.0 } else { 0.0 };
        for w in work.iter_mut().take(n) {
            *w = init;
        }
        for j in 1..=n {
            if upper {
                // L = K + 1 - J, AB( L+I, J )
                let hi = if unit { j - 1 } else { j };
                for i in 1.max(j.saturating_sub(k))..=hi {
                    work[i - 1] += ab[ix(k + 1 + i - j, j, ldab)].abs();
                }
            } else {
                // L = 1 - J
                let lo = if unit { j + 1 } else { j };
                for i in lo..=n.min(j + k) {
                    work[i - 1] += ab[ix(1 + i - j, j, ldab)].abs();
                }
            }
        }
        for &sum in work.iter().take(n) {
            if value < sum || sum.is_nan() {
                value = sum;
            }
        }
    } else if lsame(norm, b'F') || lsame(norm, b'E') {
        // dlantb.f:308-351 calls DLASSQ, which is not ported.
        return Err(LapackError::UnportedPath {
            routine: "DLANTB",
            lines: "308-351",
        });
    } else {
        // Upstream returns VALUE unset (dlantb.f:176-353); pattern of #689.
        return Err(LapackError::IllegalArgument {
            routine: "DLANTB",
            info: -1,
        });
    }
    Ok(value)
}
