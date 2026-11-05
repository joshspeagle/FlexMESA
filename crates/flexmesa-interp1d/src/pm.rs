//! Steffen (1990) piecewise-monotonic cubic interpolation (`interp_pm`).
//!
//! Reference: Steffen, M. (1990), "A simple method for monotonic interpolation
//! in one dimension", Astronomy & Astrophysics, 239, 443-450.

use crate::{validate_grid_and_widths, CubicCoeffs, Interp1dError};
use flexmesa_autodiff::DualOrder1;
use flexmesa_utils::is_bad;

/// Computes Fortran `dsign(1.0, val)`, returning `+1.0` when `val >= 0.0`
/// (including `-0.0` and `+0.0`) and `-1.0` when `val < 0.0`.
#[inline]
pub(crate) fn fortran_sign1(val: f64) -> f64 {
    if val < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// Constructs a Steffen (1990) piecewise-monotonic cubic interpolant on a
/// strictly monotonic grid `x` (`n >= 2`) with knot values `y`.
///
/// Returns `n` coefficient tuples `[c0, c1, c2, c3]` where on interval
/// $[x_i, x_{i+1}]$ with $\Delta x = x - x_i$:
/// $$p_i(x) = c_{0,i} + \Delta x\,(c_{1,i} + \Delta x\,(c_{2,i} + \Delta x\,c_{3,i}))$$
///
/// # Errors
///
/// Returns [`Interp1dError`] if `x` and `y` have mismatched lengths, fewer than
/// 2 points, non-finite values, or non-monotonic abscissas.
///
/// # Examples
///
/// ```
/// use flexmesa_interp1d::{interp_pm, interp_value};
///
/// let x = [0.0, 1.0, 2.0, 3.0];
/// let y = [0.0, 1.0, 8.0, 27.0];
/// let coeffs = interp_pm(&x, &y).unwrap();
/// let val = interp_value(&x, &coeffs, 1.5).unwrap();
/// assert!(val > 1.0 && val < 8.0);
/// ```
pub fn interp_pm(
    x: &[f64],
    y: &[f64],
) -> Result<Vec<CubicCoeffs>, Interp1dError> {
    mk_pmcub(x, y, false)
}

/// Constructs a Steffen (1990) piecewise-monotonic interpolant computing only
/// the knot values `c0` and limited first derivatives `c1` (with `c2 = c3 = 0`).
///
/// # Errors
///
/// Returns [`Interp1dError`] under the same conditions as [`interp_pm`].
pub fn interp_pm_slopes_only(
    x: &[f64],
    y: &[f64],
) -> Result<Vec<CubicCoeffs>, Interp1dError> {
    mk_pmcub(x, y, true)
}

/// Internal builder for Steffen (1990) piecewise-monotonic splines (`mk_pmcub`,
/// `mk_pmcub4`, `mk_pmquad`, and `mk_pmlinear`).
fn mk_pmcub(
    x: &[f64],
    y: &[f64],
    slope_only: bool,
) -> Result<Vec<CubicCoeffs>, Interp1dError> {
    let h = validate_grid_and_widths(x, y, 2)?;
    let n = x.len();
    let mut coeffs = vec![[0.0_f64; 4]; n];
    for i in 0..n {
        coeffs[i][0] = y[i];
    }

    if n == 2 {
        let s0 = (y[1] - y[0]) / h[0];
        coeffs[0][1] = s0;
        coeffs[1][1] = 0.0;
        return Ok(coeffs);
    }

    let mut s = vec![0.0_f64; n - 1];
    for i in 0..(n - 1) {
        s[i] = (y[i + 1] - y[i]) / h[i];
    }

    for i in 1..(n - 1) {
        let pi = (s[i - 1] * h[i] + s[i] * h[i - 1]) / (h[i - 1] + h[i]);
        let min_s = s[i - 1].abs().min(s[i].abs()).min(0.5 * pi.abs());
        coeffs[i][1] = (fortran_sign1(s[i - 1]) + fortran_sign1(s[i])) * min_s;
    }

    let p0 = s[0] * (1.0 + h[0] / (h[0] + h[1])) - s[1] * h[0] / (h[0] + h[1]);
    coeffs[0][1] = if p0 * s[0] <= 0.0 {
        0.0
    } else if p0.abs() > 2.0 * s[0].abs() {
        2.0 * s[0]
    } else {
        p0
    };

    let h_last = h[n - 2];
    let h_prev = h[n - 3];
    let p_end = s[n - 2] * (1.0 + h_last / (h_last + h_prev))
        - s[n - 3] * h_last / (h_last + h_prev);
    coeffs[n - 1][1] = if p_end * s[n - 2] <= 0.0 {
        0.0
    } else if p_end.abs() > 2.0 * s[n - 2].abs() {
        2.0 * s[n - 2]
    } else {
        p_end
    };

    if slope_only {
        return Ok(coeffs);
    }

    for i in 0..(n - 1) {
        let c1_i = coeffs[i][1];
        let c1_next = coeffs[i + 1][1];
        coeffs[i][2] = (3.0 * s[i] - 2.0 * c1_i - c1_next) / h[i];
        coeffs[i][3] = (c1_i + c1_next - 2.0 * s[i]) / (h[i] * h[i]);
    }
    coeffs[n - 1][2] = 0.0;
    coeffs[n - 1][3] = 0.0;

    Ok(coeffs)
}

/// Computes the three cubic expansion coefficients `[a1, a2, a3]` for Steffen
/// (1990) monotonic cubic interpolation across the central interval
/// `[x[1], x[2]]` of a four-point stencil.
///
/// For $x \in [x_1, x_2]$ and $\Delta x = x - x_1$:
/// $$y(x) = y_1 + \Delta x\,(a_1 + \Delta x\,(a_2 + \Delta x\,a_3))$$
///
/// # Errors
///
/// Returns [`Interp1dError`] if `x` is not strictly monotonic or contains
/// non-finite values.
pub fn interp_4pt_pm(
    x: &[f64; 4],
    y: &[f64; 4],
) -> Result<[f64; 3], Interp1dError> {
    let h = validate_grid_and_widths(x, y, 4)?;
    let (h1, h2, h3) = (h[0], h[1], h[2]);
    let s1 = (y[1] - y[0]) / h1;
    let s2 = (y[2] - y[1]) / h2;
    let s3 = (y[3] - y[2]) / h3;
    let p2 = (s1 * h2 + s2 * h1) / (h1 + h2);
    let p3 = (s2 * h3 + s3 * h2) / (h2 + h3);
    let as2 = s2.abs();
    let ss2 = fortran_sign1(s2);
    let yp2 = (fortran_sign1(s1) + ss2) * s1.abs().min(as2).min(0.5 * p2.abs());
    let yp3 = (ss2 + fortran_sign1(s3)) * as2.min(s3.abs()).min(0.5 * p3.abs());
    let a1 = yp2;
    let a2 = (3.0 * s2 - 2.0 * yp2 - yp3) / h2;
    let a3 = (yp2 + yp3 - 2.0 * s2) / (h2 * h2);
    Ok([a1, a2, a3])
}

/// Constructs a Steffen (1990) piecewise-monotonic cubic interpolant over
/// first-order forward-mode dual numbers [`DualOrder1<N>`].
///
/// # Errors
///
/// Returns [`Interp1dError`] if `x` and `y` have mismatched lengths, fewer than
/// 2 points, non-finite values, or non-monotonic primal abscissas.
pub fn interp_pm_autodiff<const N: usize>(
    x: &[DualOrder1<N>],
    y: &[DualOrder1<N>],
) -> Result<Vec<[DualOrder1<N>; 4]>, Interp1dError> {
    if x.len() != y.len() {
        return Err(Interp1dError::LengthMismatch {
            x_len: x.len(),
            y_len: y.len(),
        });
    }
    let n = x.len();
    if n < 2 {
        return Err(Interp1dError::InsufficientPoints {
            min_points: 2,
            actual: n,
        });
    }
    if x.iter().any(|v| is_bad(v.val)) || y.iter().any(|v| is_bad(v.val)) {
        return Err(Interp1dError::NonFiniteInput);
    }

    let zero = DualOrder1::<N>::constant(0.0);
    let mut h = Vec::with_capacity(n - 1);
    let h0 = x[1] - x[0];
    if h0.val == 0.0 || is_bad(h0.val) {
        return Err(Interp1dError::NonMonotonicGrid { index: 0 });
    }
    let increasing = h0.val > 0.0;
    h.push(h0);
    for i in 1..(n - 1) {
        let hi = x[i + 1] - x[i];
        if hi.val == 0.0 || is_bad(hi.val) || (hi.val > 0.0) != increasing {
            return Err(Interp1dError::NonMonotonicGrid { index: i });
        }
        h.push(hi);
    }

    let mut coeffs = vec![[zero; 4]; n];
    for i in 0..n {
        coeffs[i][0] = y[i];
    }

    if n == 2 {
        coeffs[0][1] = (y[1] - y[0]) / h[0];
        return Ok(coeffs);
    }

    let mut s = vec![zero; n - 1];
    for i in 0..(n - 1) {
        s[i] = (y[i + 1] - y[i]) / h[i];
    }

    for i in 1..(n - 1) {
        let pi = (s[i - 1] * h[i] + s[i] * h[i - 1]) / (h[i - 1] + h[i]);
        let sign_sum = s[i - 1].sign() + s[i].sign();
        let min_s = s[i - 1].abs().min(s[i].abs()).min(0.5 * pi.abs());
        coeffs[i][1] = sign_sum * min_s;
    }

    let p0 = s[0] * (1.0 + h[0] / (h[0] + h[1])) - s[1] * h[0] / (h[0] + h[1]);
    coeffs[0][1] = if (p0 * s[0]).val <= 0.0 {
        zero
    } else if p0.abs().val > (2.0 * s[0].abs()).val {
        2.0 * s[0]
    } else {
        p0
    };

    let h_last = h[n - 2];
    let h_prev = h[n - 3];
    let p_end = s[n - 2] * (1.0 + h_last / (h_last + h_prev))
        - s[n - 3] * h_last / (h_last + h_prev);
    coeffs[n - 1][1] = if (p_end * s[n - 2]).val <= 0.0 {
        zero
    } else if p_end.abs().val > (2.0 * s[n - 2].abs()).val {
        2.0 * s[n - 2]
    } else {
        p_end
    };

    for i in 0..(n - 1) {
        let c1_i = coeffs[i][1];
        let c1_next = coeffs[i + 1][1];
        coeffs[i][2] = (3.0 * s[i] - 2.0 * c1_i - c1_next) / h[i];
        coeffs[i][3] = (c1_i + c1_next - 2.0 * s[i]) / (h[i] * h[i]);
    }

    Ok(coeffs)
}
