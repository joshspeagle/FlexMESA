// SPDX-License-Identifier: GPL-3.0-or-later
//! `DLANGB`: norm of a general band matrix.

use crate::lapack::{ix, lsame, LapackError};

/// `DLANGB` (`SRC/dlangb.f:122-222`): `'M'` max-abs, `'1'`/`'O'` one-norm
/// and `'I'` infinity-norm of an `n`-by-`n` band matrix with `kl`
/// sub-diagonals and `ku` super-diagonals stored as
/// `AB(ku+1+i-j,j) = A(i,j)` (`ldab >= kl+ku+1`).
///
/// `work` (length at least `n`) is used only for `'I'` (`dlangb.f:185-202`).
/// `DISNAN(t)` (`dlangb.f:170,183,201`) is `t.is_nan()`.
///
/// # Errors
///
/// - [`LapackError::UnportedPath`] for `'F'`/`'E'` (`dlangb.f:203-214`,
///   which needs `DLASSQ`; never reached by `DGBCON` or `DGBSVX`).
/// - [`LapackError::IllegalArgument`] (`info = -1`) for any other `norm`.
///   Upstream returns `VALUE` unset there (`dlangb.f:160-217`), an unset
///   read that DECISIONS #43 turns into an explicit error; Deferred bug #689.
pub fn dlangb(
    norm: u8,
    n: usize,
    kl: usize,
    ku: usize,
    ab: &[f64],
    ldab: usize,
    work: &mut [f64],
) -> Result<f64, LapackError> {
    let value;
    if n == 0 {
        // dlangb.f:160-161
        value = 0.0;
    } else if lsame(norm, b'M') {
        // Find max(abs(A(i,j))) (dlangb.f:162-172)
        let mut v = 0.0_f64;
        for j in 1..=n {
            let lo = (ku + 2).saturating_sub(j).max(1);
            let hi = (n + ku + 1 - j).min(kl + ku + 1);
            for i in lo..=hi {
                let temp = ab[ix(i, j, ldab)].abs();
                if v < temp || temp.is_nan() {
                    v = temp;
                }
            }
        }
        value = v;
    } else if lsame(norm, b'O') || norm == b'1' {
        // Find norm1(A) (dlangb.f:173-184)
        let mut v = 0.0_f64;
        for j in 1..=n {
            let mut sum = 0.0_f64;
            let lo = (ku + 2).saturating_sub(j).max(1);
            let hi = (n + ku + 1 - j).min(kl + ku + 1);
            for i in lo..=hi {
                sum += ab[ix(i, j, ldab)].abs();
            }
            if v < sum || sum.is_nan() {
                v = sum;
            }
        }
        value = v;
    } else if lsame(norm, b'I') {
        // Find normI(A) (dlangb.f:185-202)
        for w in work.iter_mut().take(n) {
            *w = 0.0;
        }
        for j in 1..=n {
            let lo = 1.max(j.saturating_sub(ku));
            let hi = n.min(j + kl);
            for i in lo..=hi {
                // K = KU + 1 - J, AB( K+I, J )
                work[i - 1] += ab[ix(ku + 1 + i - j, j, ldab)].abs();
            }
        }
        let mut v = 0.0_f64;
        for &temp in work.iter().take(n) {
            if v < temp || temp.is_nan() {
                v = temp;
            }
        }
        value = v;
    } else if lsame(norm, b'F') || lsame(norm, b'E') {
        // dlangb.f:203-214 calls DLASSQ, which is not ported.
        return Err(LapackError::UnportedPath {
            routine: "DLANGB",
            lines: "203-214",
        });
    } else {
        // Upstream returns VALUE unset here (dlangb.f:160-219): Deferred
        // bug #689. Explicit error instead (DECISIONS #43).
        return Err(LapackError::IllegalArgument {
            routine: "DLANGB",
            info: -1,
        });
    }
    Ok(value)
}
