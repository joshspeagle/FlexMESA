//! Two-dimensional piecewise-monotonic cubic interpolation (`interp_mkbipm_db`
//! and `interp_evbipm_db`).
//!
//! Constructs 1D Steffen (1990) piecewise-monotonic cubic splines along `x` at
//! every `y[iy]` slice during setup (`interp_mkbipm_db`). At evaluation time
//! (`interp_evbipm_db`), evaluates the four 1D `x`-splines bracketing `yget`
//! and performs a 4-point local Steffen (1990) interpolation along `y`.
//!
//! Reference: Steffen, M. (1990), "A simple method for monotonic interpolation
//! in one dimension", Astronomy & Astrophysics, 239, 443-450.

use crate::spline1d::splinck_db;
use crate::Interp2dError;
use flexmesa_interp1d::{interp_pm, interp_value};
use flexmesa_utils::is_bad;

/// Constructs 2D piecewise-monotonic cubic spline coefficients along the `x`
/// axis for each `y` slice (`interp_mkbipm_db`).
///
/// Input `z` has row-major shape `(nx, ny)` where
/// `z[ix * ny + iy] = f(x[ix], y[iy])`. Returns a vector of `nx * ny` 4-element
/// `x`-spline coefficient arrays `[c0, c1, c2, c3]` indexed as
/// `coeffs[ix * ny + iy]`.
///
/// # Errors
///
/// Returns [`Interp2dError`] if `nx < 4`, `ny < 4`, `x` or `y` is not strictly
/// ascending, or `z` has mismatched length or non-finite values.
pub fn interp_mkbipm_db(
    x: &[f64],
    y: &[f64],
    z: &[f64],
) -> Result<Vec<[f64; 4]>, Interp2dError> {
    let nx = x.len();
    let ny = y.len();
    if nx < 4 {
        return Err(Interp2dError::InsufficientPoints {
            axis: "x",
            min_points: 4,
            actual: nx,
        });
    }
    if ny < 4 {
        return Err(Interp2dError::InsufficientPoints {
            axis: "y",
            min_points: 4,
            actual: ny,
        });
    }
    if z.len() != nx * ny {
        return Err(Interp2dError::DimensionMismatch {
            expected: nx * ny,
            actual: z.len(),
        });
    }
    if z.iter().any(|&v| is_bad(v)) {
        return Err(Interp2dError::NonFiniteInput);
    }
    splinck_db(x, 1.0e-3, "x")?;
    splinck_db(y, 1.0e-3, "y")?;

    // Fit 1D Steffen (1990) monotonic spline along x for each y-slice.
    let mut coeffs = vec![[0.0_f64; 4]; nx * ny];
    let mut x_vals = vec![0.0_f64; nx];
    for iy in 0..ny {
        for ix in 0..nx {
            x_vals[ix] = z[ix * ny + iy];
        }
        let c_x =
            interp_pm(x, &x_vals).map_err(|_| Interp2dError::NonFiniteInput)?;
        for ix in 0..nx {
            coeffs[ix * ny + iy] = c_x[ix];
        }
    }
    Ok(coeffs)
}

/// Evaluates the 2D piecewise-monotonic cubic interpolant at `(xget, yget)` by
/// evaluating four 1D `x`-splines and performing a 4-point 1D `interp_pm` along
/// `y` (`interp_evbipm_db`).
///
/// # Errors
///
/// Returns [`Interp2dError`] if `(xget, yget)` lies outside the grid or inputs
/// are invalid.
pub fn interp_evbipm_db(
    xget: f64,
    yget: f64,
    x: &[f64],
    y: &[f64],
    coeffs: &[[f64; 4]],
) -> Result<f64, Interp2dError> {
    let nx = x.len();
    let ny = y.len();
    if nx < 4 {
        return Err(Interp2dError::InsufficientPoints {
            axis: "x",
            min_points: 4,
            actual: nx,
        });
    }
    if ny < 4 {
        return Err(Interp2dError::InsufficientPoints {
            axis: "y",
            min_points: 4,
            actual: ny,
        });
    }
    if coeffs.len() != nx * ny {
        return Err(Interp2dError::DimensionMismatch {
            expected: nx * ny,
            actual: coeffs.len(),
        });
    }
    if is_bad(xget) || is_bad(yget) {
        return Err(Interp2dError::NonFiniteInput);
    }
    splinck_db(x, 1.0e-3, "x")?;
    splinck_db(y, 1.0e-3, "y")?;
    if xget < x[0] || xget >= x[nx - 1] {
        return Err(Interp2dError::OutOfRange {
            axis: "x",
            value: xget,
        });
    }
    if yget < y[0] || yget >= y[ny - 1] {
        return Err(Interp2dError::OutOfRange {
            axis: "y",
            value: yget,
        });
    }

    // Locate x-interval ix and y-interval jy via binary search.
    let pos_x = x.partition_point(|&xk| xk <= xget);
    let ix = pos_x.saturating_sub(1).min(nx - 2);

    let pos_y = y.partition_point(|&yk| yk <= yget);
    let jy = pos_y.saturating_sub(1).min(ny - 2);

    // Choose 4-point y-stencil [jlo..jlo + 4] bracketing yget.
    let dx = xget - x[ix];
    let jlo = if jy == 0 {
        0
    } else if jy >= ny - 2 {
        jy - 2
    } else {
        jy - 1
    };

    let mut ys = [0.0_f64; 4];
    let mut ff = [0.0_f64; 4];
    for k in 0..4 {
        let j = jlo + k;
        ys[k] = y[j];
        let c = &coeffs[ix * ny + j];
        ff[k] = c[0] + dx * (c[1] + dx * (c[2] + dx * c[3]));
    }

    let y_coeffs =
        interp_pm(&ys, &ff).map_err(|_| Interp2dError::NonFiniteInput)?;
    interp_value(&ys, &y_coeffs, yget)
        .map_err(|_| Interp2dError::NonFiniteInput)
}
