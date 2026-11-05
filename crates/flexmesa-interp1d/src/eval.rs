//! Evaluation, differentiation, and definite integration for 1D cubic splines.

use crate::{CubicCoeffs, Interp1dError};
use flexmesa_utils::is_bad;

/// Clamps `xval` to the knot domain and locates the interval index
/// `k in 0..(n - 1)` for a strictly monotonic grid `init_x` (`n >= 2`).
#[inline]
pub(crate) fn locate_clamped_interval(
    init_x: &[f64],
    xval: f64,
    increasing: bool,
) -> (usize, f64) {
    let n = init_x.len();
    let x0 = init_x[0];
    let xn = init_x[n - 1];
    let x_clamped = if increasing {
        xval.clamp(x0, xn)
    } else {
        xval.clamp(xn, x0)
    };
    let idx = if increasing {
        init_x.partition_point(|&xk| xk < x_clamped)
    } else {
        init_x.partition_point(|&xk| xk > x_clamped)
    };
    let k = if idx == 0 {
        0
    } else if idx >= n {
        n - 2
    } else {
        idx - 1
    };
    (k, x_clamped - init_x[k])
}

/// Validates `init_x` and `coeffs` for evaluation.
fn validate_eval_inputs(
    init_x: &[f64],
    coeffs: &[CubicCoeffs],
) -> Result<bool, Interp1dError> {
    if init_x.len() != coeffs.len() {
        return Err(Interp1dError::LengthMismatch {
            x_len: init_x.len(),
            y_len: coeffs.len(),
        });
    }
    let n = init_x.len();
    if n == 0 {
        return Err(Interp1dError::InsufficientPoints {
            min_points: 1,
            actual: 0,
        });
    }
    if init_x.iter().any(|&v| is_bad(v))
        || coeffs.iter().any(|c| c.iter().any(|&v| is_bad(v)))
    {
        return Err(Interp1dError::NonFiniteInput);
    }
    if n == 1 {
        return Ok(true);
    }
    let h0 = init_x[1] - init_x[0];
    if h0 == 0.0 {
        return Err(Interp1dError::NonMonotonicGrid { index: 0 });
    }
    let increasing = h0 > 0.0;
    for i in 1..(n - 1) {
        let hi = init_x[i + 1] - init_x[i];
        if hi == 0.0 || (hi > 0.0) != increasing {
            return Err(Interp1dError::NonMonotonicGrid { index: i });
        }
    }
    Ok(increasing)
}

/// Evaluates a 1D cubic spline at a single abscissa `xval`, clamping
/// out-of-range queries to the domain boundaries.
///
/// # Errors
///
/// Returns [`Interp1dError`] if `init_x` and `coeffs` are invalid or `xval` is
/// non-finite.
pub fn interp_value(
    init_x: &[f64],
    coeffs: &[CubicCoeffs],
    xval: f64,
) -> Result<f64, Interp1dError> {
    let increasing = validate_eval_inputs(init_x, coeffs)?;
    if is_bad(xval) {
        return Err(Interp1dError::NonFiniteInput);
    }
    if init_x.len() == 1 {
        return Ok(coeffs[0][0]);
    }
    let (k, delta) = locate_clamped_interval(init_x, xval, increasing);
    let c = &coeffs[k];
    Ok(c[0] + delta * (c[1] + delta * (c[2] + delta * c[3])))
}

/// Evaluates a 1D cubic spline at a slice of query abscissas `x`, clamping
/// out-of-range queries to the domain boundaries.
///
/// # Errors
///
/// Returns [`Interp1dError`] if `init_x`, `coeffs`, or `x` are invalid.
pub fn interp_values(
    init_x: &[f64],
    coeffs: &[CubicCoeffs],
    x: &[f64],
) -> Result<Vec<f64>, Interp1dError> {
    let increasing = validate_eval_inputs(init_x, coeffs)?;
    if x.iter().any(|&v| is_bad(v)) {
        return Err(Interp1dError::NonFiniteInput);
    }
    if init_x.len() == 1 {
        return Ok(vec![coeffs[0][0]; x.len()]);
    }
    let mut vals = Vec::with_capacity(x.len());
    for &xk in x {
        let (k, delta) = locate_clamped_interval(init_x, xk, increasing);
        let c = &coeffs[k];
        vals.push(c[0] + delta * (c[1] + delta * (c[2] + delta * c[3])));
    }
    Ok(vals)
}

/// Evaluates both the interpolated value $p(x)$ and first derivative $p'(x)$ at
/// a single abscissa `xval`.
///
/// # Errors
///
/// Returns [`Interp1dError`] if `init_x` and `coeffs` are invalid or `xval` is
/// non-finite.
pub fn interp_value_and_slope(
    init_x: &[f64],
    coeffs: &[CubicCoeffs],
    xval: f64,
) -> Result<(f64, f64), Interp1dError> {
    let increasing = validate_eval_inputs(init_x, coeffs)?;
    if is_bad(xval) {
        return Err(Interp1dError::NonFiniteInput);
    }
    if init_x.len() == 1 {
        return Ok((coeffs[0][0], 0.0));
    }
    let (k, delta) = locate_clamped_interval(init_x, xval, increasing);
    let c = &coeffs[k];
    let val = c[0] + delta * (c[1] + delta * (c[2] + delta * c[3]));
    let slope = c[1] + 2.0 * delta * (c[2] + 1.5 * delta * c[3]);
    Ok((val, slope))
}

/// Evaluates both the interpolated values and first derivatives at a slice of
/// query abscissas `x`.
///
/// # Errors
///
/// Returns [`Interp1dError`] if `init_x`, `coeffs`, or `x` are invalid.
pub fn interp_values_and_slopes(
    init_x: &[f64],
    coeffs: &[CubicCoeffs],
    x: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), Interp1dError> {
    let increasing = validate_eval_inputs(init_x, coeffs)?;
    if x.iter().any(|&v| is_bad(v)) {
        return Err(Interp1dError::NonFiniteInput);
    }
    if init_x.len() == 1 {
        return Ok((vec![coeffs[0][0]; x.len()], vec![0.0; x.len()]));
    }
    let mut vals = Vec::with_capacity(x.len());
    let mut slopes = Vec::with_capacity(x.len());
    for &xk in x {
        let (k, delta) = locate_clamped_interval(init_x, xk, increasing);
        let c = &coeffs[k];
        vals.push(c[0] + delta * (c[1] + delta * (c[2] + delta * c[3])));
        slopes.push(c[1] + 2.0 * delta * (c[2] + 1.5 * delta * c[3]));
    }
    Ok((vals, slopes))
}

/// Evaluates the definite integral of the piecewise cubic spline over consecutive
/// sub-intervals $[x_{m-1}, x_m]$ for $m = 1, \dots, N_v - 1$, with
/// `vals[0] = 0.0`.
///
/// # Errors
///
/// Returns [`Interp1dError`] if `x` lies outside `[init_x[0], init_x[n - 1]]`
/// or is not monotonic in the same direction as `init_x`.
pub fn integrate_values(
    init_x: &[f64],
    coeffs: &[CubicCoeffs],
    x: &[f64],
) -> Result<Vec<f64>, Interp1dError> {
    let increasing = validate_eval_inputs(init_x, coeffs)?;
    let nx = init_x.len();
    if nx < 2 {
        return Err(Interp1dError::InsufficientPoints {
            min_points: 2,
            actual: nx,
        });
    }
    let nv = x.len();
    if nv == 0 {
        return Ok(Vec::new());
    }
    if x.iter().any(|&v| is_bad(v)) {
        return Err(Interp1dError::NonFiniteInput);
    }
    if (increasing && (x[0] < init_x[0] || x[nv - 1] > init_x[nx - 1]))
        || (!increasing && (x[0] > init_x[0] || x[nv - 1] < init_x[nx - 1]))
    {
        return Err(Interp1dError::IntegrationDomainError);
    }
    for i in 1..nv {
        if (increasing && x[i] < x[i - 1]) || (!increasing && x[i] > x[i - 1]) {
            return Err(Interp1dError::IntegrationDomainError);
        }
    }

    let poly_antideriv = |k: usize, d: f64| -> f64 {
        let c = &coeffs[k];
        d * (c[0] + d * (0.5 * c[1] + d * (c[2] / 3.0 + 0.25 * d * c[3])))
    };

    let mut vals = vec![0.0_f64; nv];
    let mut k_old = 0usize;
    let mut xkp1_old = init_x[1];
    let mut xk_prev = x[0];

    for val_out in vals.iter_mut().skip(1).zip(x.iter().skip(1)) {
        let (out_slot, &xk_target) = val_out;
        let mut xk_new = xk_target;
        let mut sum = 0.0_f64;

        while (increasing && xk_new > xkp1_old)
            || (!increasing && xk_new < xkp1_old)
        {
            k_old += 1;
            if k_old >= nx - 1 {
                k_old = nx - 2;
                xk_new = xkp1_old;
                break;
            }
            let x0 = init_x[k_old - 1];
            let x1 = xk_prev;
            let x2 = xkp1_old;
            if x1 != x2 {
                let a2 = poly_antideriv(k_old - 1, x2 - x0);
                let a1 = if x1 != x0 {
                    poly_antideriv(k_old - 1, x1 - x0)
                } else {
                    0.0
                };
                sum += a2 - a1;
                xk_prev = x2;
            }
            xkp1_old = init_x[k_old + 1];
        }

        let x0 = init_x[k_old];
        let x1 = xk_prev;
        let x2 = xk_new;
        if x1 != x2 {
            let a2 = poly_antideriv(k_old, x2 - x0);
            let a1 = if x1 != x0 {
                poly_antideriv(k_old, x1 - x0)
            } else {
                0.0
            };
            sum += a2 - a1;
            xk_prev = x2;
        }
        *out_slot = sum;
    }

    Ok(vals)
}

