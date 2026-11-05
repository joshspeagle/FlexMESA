//! Huynh (1993) and Suresh & Huynh (1997) monotonicity-preserving cubic
//! interpolation (`interp_m3`, `interp_m3a`, `interp_m3b`, `interp_m3q`).
//!
//! References:
//! - Huynh, H. T. (1993), "Accurate Monotone Cubic Interpolation", SIAM Journal
//!   on Numerical Analysis, 30, 57-100.
//! - Suresh, A., & Huynh, H. T. (1997), "Accurate Monotonicity-Preserving
//!   Schemes with Runge-Kutta Time Stepping", Journal of Computational Physics,
//!   136, 83-99.

use crate::pm::{fortran_sign1, interp_pm};
use crate::{validate_grid_and_widths, CubicCoeffs, Interp1dError, MpLimiter};

/// Two-argument minmod limiter:
/// $$\mathrm{minmod}(a, b)
///   = \tfrac{1}{2}\,(\mathrm{sign}(1, a) + \mathrm{sign}(1, b))
///     \,\min(|a|, |b|)$$
#[inline]
#[must_use]
pub fn minmod(a: f64, b: f64) -> f64 {
    f64::midpoint(fortran_sign1(a), fortran_sign1(b)) * a.abs().min(b.abs())
}

/// Four-argument minmod limiter applied sequentially
/// (`minmod(minmod(minmod(a, b), c), d)`).
#[inline]
#[must_use]
pub fn minmod4(a: f64, b: f64, c: f64, d: f64) -> f64 {
    minmod(minmod(minmod(a, b), c), d)
}

/// Median of three real numbers `a`, `b`, and `c` computed via
/// $a + \mathrm{minmod}(b - a, c - a)$.
#[inline]
#[must_use]
pub fn median(a: f64, b: f64, c: f64) -> f64 {
    a + minmod(b - a, c - a)
}

/// Constructs a Huynh (1993) monotonicity-preserving cubic interpolant on a
/// strictly monotonic grid `x` (`n >= 2`) with knot values `y` using the
/// specified [`MpLimiter`] (`Average`, `Quartic`, or `SuperBee`).
///
/// For $n \in \{2, 3\}$, delegates to `interp_pm` (`mk_pmlinear` / `mk_pmquad`)
/// matching upstream MESA `m3`.
///
/// # Errors
///
/// Returns [`Interp1dError`] if `x` and `y` have mismatched lengths, fewer than
/// 2 points, non-finite values, or non-monotonic abscissas.
///
/// # Examples
///
/// ```
/// use flexmesa_interp1d::{interp_m3q, interp_value};
///
/// let x = [0.0, 1.0, 2.0, 3.0, 4.0];
/// let y = [0.0, 1.0, 8.0, 27.0, 64.0];
/// let coeffs = interp_m3q(&x, &y).unwrap();
/// let val = interp_value(&x, &coeffs, 2.5).unwrap();
/// assert!((val - 15.625).abs() < 0.2);
/// ```
pub fn interp_m3(
    x: &[f64],
    y: &[f64],
    limiter: MpLimiter,
) -> Result<Vec<CubicCoeffs>, Interp1dError> {
    let h = validate_grid_and_widths(x, y, 2)?;
    let n = x.len();
    if n <= 3 {
        return interp_pm(x, y);
    }

    let mut coeffs = vec![[0.0_f64; 4]; n];
    for i in 0..n {
        coeffs[i][0] = y[i];
    }

    let mut s_mid = vec![0.0_f64; n - 1];
    for i in 0..(n - 1) {
        s_mid[i] = (y[i + 1] - y[i]) / h[i];
    }

    let mut d = vec![0.0_f64; n];
    for i in 1..(n - 1) {
        d[i] = (s_mid[i] - s_mid[i - 1]) / (x[i + 1] - x[i - 1]);
    }
    d[0] = d[1];
    d[n - 1] = d[n - 2];

    let mut d_mid = vec![0.0_f64; n - 1];
    for i in 0..(n - 1) {
        let tmp1 = 4.0 * d[i + 1] - d[i];
        let tmp2 = 4.0 * d[i] - d[i + 1];
        d_mid[i] = minmod4(d[i + 1], d[i], tmp1, tmp2);
    }

    let mut sp_l = vec![0.0_f64; n];
    let mut sp_r = vec![0.0_f64; n];
    for i in 0..(n - 1) {
        let hd_mid = h[i] * d_mid[i];
        sp_l[i + 1] = s_mid[i] + hd_mid;
        sp_r[i] = s_mid[i] - hd_mid;
    }

    let mut s = vec![0.0_f64; n];
    let mut t = vec![0.0_f64; n];
    for i in 1..(n - 1) {
        s[i] = minmod(s_mid[i - 1], s_mid[i]);
        t[i] = minmod(sp_l[i], sp_r[i]);
    }

    match limiter {
        MpLimiter::Average => {
            for i in 1..(n - 1) {
                let bound = (3.0 * s[i].abs()).max(1.5 * t[i].abs());
                let avg = 0.5 * (sp_l[i] + sp_r[i]).abs();
                coeffs[i][1] = fortran_sign1(t[i]) * avg.min(bound);
            }
        }
        MpLimiter::Quartic => {
            let mut e_mid = vec![0.0_f64; n - 1];
            for i in 1..(n - 2) {
                e_mid[i] = (d[i + 1] - d[i]) / (x[i + 2] - x[i - 1]);
            }
            for i in 2..(n - 2) {
                let numer = e_mid[i - 1] * (x[i + 2] - x[i])
                    + e_mid[i] * (x[i] - x[i - 2]);
                let denom = x[i + 2] - x[i - 2];
                coeffs[i][1] =
                    s_mid[i] - h[i] * (d[i] + h[i - 1] * numer / denom);
            }
            for idx in [1, n - 2] {
                let bound = (3.0 * s[idx].abs()).max(1.5 * t[idx].abs());
                let avg = 0.5 * (sp_l[idx] + sp_r[idx]).abs();
                coeffs[idx][1] = fortran_sign1(t[idx]) * avg.min(bound);
            }
            for i in 1..(n - 1) {
                let med = median(coeffs[i][1], sp_l[i], sp_r[i]);
                let tmax = fortran_sign1(t[i])
                    * (3.0 * s[i].abs()).max(1.5 * t[i].abs());
                coeffs[i][1] = minmod(med, tmax);
            }
        }
        MpLimiter::SuperBee => {
            for i in 1..(n - 1) {
                let bound = (3.0 * s[i].abs()).max(1.5 * t[i].abs());
                let max_sp = sp_l[i].abs().max(sp_r[i].abs());
                coeffs[i][1] = fortran_sign1(t[i]) * max_sp.min(bound);
            }
        }
    }

    coeffs[0][1] = minmod(s_mid[0], s_mid[1]);
    coeffs[n - 1][1] = minmod(s_mid[n - 3], s_mid[n - 2]);

    for i in 0..(n - 1) {
        let c1_i = coeffs[i][1];
        let c1_next = coeffs[i + 1][1];
        coeffs[i][2] = (3.0 * s_mid[i] - 2.0 * c1_i - c1_next) / h[i];
        coeffs[i][3] = (c1_i + c1_next - 2.0 * s_mid[i]) / (h[i] * h[i]);
    }
    let hn = h[n - 2];
    let yn_prev = coeffs[n - 2][0];
    let yn = coeffs[n - 1][0];
    let s_prev = coeffs[n - 2][1];
    let s_end = coeffs[n - 1][1];
    coeffs[n - 1][2] =
        (3.0 * yn_prev - 3.0 * yn + (s_prev + 2.0 * s_end) * hn) / (hn * hn);
    coeffs[n - 1][3] =
        (-2.0 * yn_prev + 2.0 * yn - (s_prev + s_end) * hn) / (hn * hn * hn);

    Ok(coeffs)
}

/// Constructs a Huynh (1993) monotonicity-preserving cubic interpolant using
/// the average slope limiter (`MpLimiter::Average`, upstream `interp_m3a`).
///
/// # Errors
///
/// Returns [`Interp1dError`] under the same conditions as [`interp_m3`].
pub fn interp_m3a(
    x: &[f64],
    y: &[f64],
) -> Result<Vec<CubicCoeffs>, Interp1dError> {
    interp_m3(x, y, MpLimiter::Average)
}

/// Constructs a Huynh (1993) monotonicity-preserving cubic interpolant using
/// the superbee slope limiter (`MpLimiter::SuperBee`, upstream `interp_m3b`).
///
/// # Errors
///
/// Returns [`Interp1dError`] under the same conditions as [`interp_m3`].
pub fn interp_m3b(
    x: &[f64],
    y: &[f64],
) -> Result<Vec<CubicCoeffs>, Interp1dError> {
    interp_m3(x, y, MpLimiter::SuperBee)
}

/// Constructs a Huynh (1993) monotonicity-preserving cubic interpolant using
/// the quartic slope limiter (`MpLimiter::Quartic`, upstream `interp_m3q`).
///
/// # Errors
///
/// Returns [`Interp1dError`] under the same conditions as [`interp_m3`].
pub fn interp_m3q(
    x: &[f64],
    y: &[f64],
) -> Result<Vec<CubicCoeffs>, Interp1dError> {
    interp_m3(x, y, MpLimiter::Quartic)
}
