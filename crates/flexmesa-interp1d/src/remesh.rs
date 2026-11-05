//! Fixed-point stencil remeshing (`interp_3_to_1`, `interp_3_to_2`,
//! `interp_4_to_1`) and forward-mode automatic differentiation (`DualOrder1`)
//! evaluation for 1D cubic splines.

use crate::eval::{interp_values, locate_clamped_interval};
use crate::pm::{interp_pm, interp_pm_autodiff};
use crate::{CubicCoeffs, Interp1dError};
use flexmesa_autodiff::DualOrder1;
use flexmesa_utils::is_bad;

/// Interpolates `v_old` from `x_old` onto `x_new` using a caller-supplied
/// spline builder `builder`.
///
/// # Errors
///
/// Returns [`Interp1dError`] if spline construction or evaluation fails.
pub fn interpolate_vector<F>(
    x_old: &[f64],
    v_old: &[f64],
    x_new: &[f64],
    builder: F,
) -> Result<Vec<f64>, Interp1dError>
where
    F: FnOnce(&[f64], &[f64]) -> Result<Vec<CubicCoeffs>, Interp1dError>,
{
    let coeffs = builder(x_old, v_old)?;
    interp_values(x_old, &coeffs, x_new)
}

/// Interpolates `v_old` from `x_old` onto `x_new` using Steffen (1990)
/// piecewise-monotonic cubic interpolation (`interp_pm`).
///
/// # Errors
///
/// Returns [`Interp1dError`] if spline construction or evaluation fails.
pub fn interpolate_vector_pm(
    x_old: &[f64],
    v_old: &[f64],
    x_new: &[f64],
) -> Result<Vec<f64>, Interp1dError> {
    interpolate_vector(x_old, v_old, x_new, interp_pm)
}

/// Remeshes 3 input points spaced by `(pdqm1, pdq00)` with values
/// `(pfm1, pf00, pfp1)` onto 1 output point at offset `ndqm1` from the first
/// point using piecewise-monotonic quadratic interpolation (`interp_3_to_1`).
///
/// # Errors
///
/// Returns [`Interp1dError`] if the spacing is non-monotonic or non-finite.
pub fn interp_3_to_1(
    pdqm1: f64,
    pdq00: f64,
    ndqm1: f64,
    pfm1: f64,
    pf00: f64,
    pfp1: f64,
) -> Result<f64, Interp1dError> {
    let x_old = [0.0, pdqm1, pdqm1 + pdq00];
    let v_old = [pfm1, pf00, pfp1];
    let x_new = [ndqm1];
    let v_new = interpolate_vector_pm(&x_old, &v_old, &x_new)?;
    Ok(v_new[0])
}

/// Remeshes 3 input points spaced by `(pdqm1, pdq00)` with values
/// `(pfm1, pf00, pfp1)` onto 2 output points at cumulative spacings `ndqm1` and
/// `ndqm1 + ndq00` (`interp_3_to_2`).
///
/// # Errors
///
/// Returns [`Interp1dError`] if the spacing is non-monotonic or non-finite.
pub fn interp_3_to_2(
    pdqm1: f64,
    pdq00: f64,
    ndqm1: f64,
    ndq00: f64,
    pfm1: f64,
    pf00: f64,
    pfp1: f64,
) -> Result<(f64, f64), Interp1dError> {
    let x_old = [0.0, pdqm1, pdqm1 + pdq00];
    let v_old = [pfm1, pf00, pfp1];
    let x_new = [ndqm1, ndqm1 + ndq00];
    let v_new = interpolate_vector_pm(&x_old, &v_old, &x_new)?;
    Ok((v_new[0], v_new[1]))
}

/// Remeshes 4 input points spaced by `(pdqm1, pdq00, pdqp1)` with values
/// `(pfm1, pf00, pfp1, pfp2)` onto 1 output point at offset `ndq00`
/// (`interp_4_to_1`).
///
/// # Errors
///
/// Returns [`Interp1dError`] if the spacing is non-monotonic or non-finite.
#[allow(clippy::too_many_arguments)]
pub fn interp_4_to_1(
    pdqm1: f64,
    pdq00: f64,
    pdqp1: f64,
    ndq00: f64,
    pfm1: f64,
    pf00: f64,
    pfp1: f64,
    pfp2: f64,
) -> Result<f64, Interp1dError> {
    let x_old = [0.0, pdqm1, pdqm1 + pdq00, pdqm1 + pdq00 + pdqp1];
    let v_old = [pfm1, pf00, pfp1, pfp2];
    let x_new = [ndq00];
    let v_new = interpolate_vector_pm(&x_old, &v_old, &x_new)?;
    Ok(v_new[0])
}

/// Evaluates a 1D cubic spline over first-order forward-mode dual numbers
/// [`DualOrder1<N>`].
///
/// # Errors
///
/// Returns [`Interp1dError`] if `init_x`, `coeffs`, or `x` are invalid.
pub fn interp_values_autodiff<const N: usize>(
    init_x: &[DualOrder1<N>],
    coeffs: &[[DualOrder1<N>; 4]],
    x: &[DualOrder1<N>],
) -> Result<Vec<DualOrder1<N>>, Interp1dError> {
    if init_x.len() != coeffs.len() {
        return Err(Interp1dError::LengthMismatch {
            x_len: init_x.len(),
            y_len: coeffs.len(),
        });
    }
    let nx = init_x.len();
    if nx == 0 {
        return Err(Interp1dError::InsufficientPoints {
            min_points: 1,
            actual: 0,
        });
    }
    if x.iter().any(|v| is_bad(v.val)) || init_x.iter().any(|v| is_bad(v.val)) {
        return Err(Interp1dError::NonFiniteInput);
    }
    if nx == 1 {
        return Ok(vec![coeffs[0][0]; x.len()]);
    }
    let primal_x: Vec<f64> = init_x.iter().map(|v| v.val).collect();
    let h0 = primal_x[1] - primal_x[0];
    if h0 == 0.0 {
        return Err(Interp1dError::NonMonotonicGrid { index: 0 });
    }
    let increasing = h0 > 0.0;
    for i in 1..(nx - 1) {
        let hi = primal_x[i + 1] - primal_x[i];
        if hi == 0.0 || (hi > 0.0) != increasing {
            return Err(Interp1dError::NonMonotonicGrid { index: i });
        }
    }

    let mut vals = Vec::with_capacity(x.len());
    for &xk in x {
        let x_clamped = if increasing {
            if xk.val > init_x[nx - 1].val {
                init_x[nx - 1]
            } else if xk.val < init_x[0].val {
                init_x[0]
            } else {
                xk
            }
        } else if xk.val < init_x[nx - 1].val {
            init_x[nx - 1]
        } else if xk.val > init_x[0].val {
            init_x[0]
        } else {
            xk
        };
        let (k, _) =
            locate_clamped_interval(&primal_x, x_clamped.val, increasing);
        let delta = x_clamped - init_x[k];
        let c = &coeffs[k];
        vals.push(c[0] + delta * (c[1] + delta * (c[2] + delta * c[3])));
    }
    Ok(vals)
}

/// Interpolates `v_old` from `x_old` onto `x_new` over [`DualOrder1<N>`] using
/// Steffen (1990) piecewise-monotonic cubic interpolation.
///
/// # Errors
///
/// Returns [`Interp1dError`] if spline construction or evaluation fails.
pub fn interpolate_vector_pm_autodiff<const N: usize>(
    x_old: &[DualOrder1<N>],
    v_old: &[DualOrder1<N>],
    x_new: &[DualOrder1<N>],
) -> Result<Vec<DualOrder1<N>>, Interp1dError> {
    let coeffs = interp_pm_autodiff(x_old, v_old)?;
    interp_values_autodiff(x_old, &coeffs, x_new)
}
