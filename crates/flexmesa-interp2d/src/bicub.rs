//! Two-dimensional $C^2$ bicubic spline setup (`interp_mkbicub_db`).
//!
//! Constructs tensor-product bicubic spline coefficients
//! `[f, f_xx, f_yy, f_xxyy]` on a rectangular grid `(x, y)` of shape `(nx, ny)`
//! (`nx >= 4`, `ny >= 4`) in three passes:
//! 1. **Pass 1 (`f_xx`)**: Solves 1D cubic splines along `x` at each `y[iy]`
//!    slice using the specified left/right `x` boundary conditions (`ibcxmin`,
//!    `bcxmin`, `ibcxmax`, `bcxmax`).
//! 2. **Pass 2 (homogeneous `f_yy` and `f_xxyy`)**: Solves 1D cubic splines
//!    along `y` at each `x[ix]` slice with homogeneous (`0.0`) boundary targets
//!    to ensure commutable cross-derivative moments.
//! 3. **Pass 3 (inhomogeneous `y` boundary correction)**: When non-zero first-
//!    or second-derivative boundary values are supplied in `bcymin` or
//!    `bcymax`, solves a residual 1D spline on zero knot data along `y` and
//!    propagates its `x`-second derivative into `f_xxyy`, matching upstream
//!    PSPLINE / MESA `do_mkbicub_db`.

pub use crate::bicub_eval::{
    herm2xy_db, interp_evbicub_db, interp_evbicub_mask_db,
};
use crate::spline1d::{mkspline_db, splinck_db};
use crate::{BicubicBc, Interp2dError};
use flexmesa_utils::is_bad;

/// Validates that a boundary-condition slice `bc` has length `expected` when
/// `code` is [`BicubicBc::FirstDeriv`] or [`BicubicBc::SecondDeriv`].
///
/// For homogeneous or divided-difference boundary codes (`Periodic`,
/// `NotAKnot`, `ZeroFirstDeriv`, `ZeroSecondDeriv`, `DividedDiff1..=3`), the
/// boundary slice is unused and may be empty.
#[inline]
fn validate_bc_len(
    code: BicubicBc,
    bc: &[f64],
    expected: usize,
) -> Result<(), Interp2dError> {
    if matches!(code, BicubicBc::FirstDeriv | BicubicBc::SecondDeriv)
        && bc.len() != expected
    {
        return Err(Interp2dError::DimensionMismatch {
            expected,
            actual: bc.len(),
        });
    }
    Ok(())
}

/// Constructs 2D bicubic spline coefficients on rectangular grid `(x, y)` of
/// shape `(nx, ny)` (`nx >= 4`, `ny >= 4`).
///
/// The input `z` slice has row-major shape `(nx, ny)` where
/// `z[ix * ny + iy] = f(x[ix], y[iy])`. On output, returns
/// `(coeffs, ilinx, iliny)` where `coeffs` has length `nx * ny` of
/// `[f, f_xx, f_yy, f_xxyy]` at each grid node `(ix, iy)`.
///
/// # Arguments
///
/// * `x` - Strictly ascending x-grid coordinates of length `nx >= 4`.
/// * `y` - Strictly ascending y-grid coordinates of length `ny >= 4`.
/// * `z` - Row-major function values of length `nx * ny`.
/// * `ibcxmin`, `bcxmin` - Boundary condition and data slice at `x[0]`.
/// * `ibcxmax`, `bcxmax` - Boundary condition and data slice at `x[nx - 1]`.
/// * `ibcymin`, `bcymin` - Boundary condition and data slice at `y[0]`.
/// * `ibcymax`, `bcymax` - Boundary condition and data slice at `y[ny - 1]`.
///
/// # Errors
///
/// Returns [`Interp2dError`] if grid dimensions are `< 4`, non-ascending,
/// mismatched with `z` or boundary condition slices, or contain non-finite
/// values.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn interp_mkbicub_db(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    ibcxmin: BicubicBc,
    bcxmin: &[f64],
    ibcxmax: BicubicBc,
    bcxmax: &[f64],
    ibcymin: BicubicBc,
    bcymin: &[f64],
    ibcymax: BicubicBc,
    bcymax: &[f64],
) -> Result<(Vec<[f64; 4]>, bool, bool), Interp2dError> {
    let (nx, ny) = (x.len(), y.len());
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
    validate_bc_len(ibcxmin, bcxmin, ny)?;
    validate_bc_len(ibcxmax, bcxmax, ny)?;
    validate_bc_len(ibcymin, bcymin, nx)?;
    validate_bc_len(ibcymax, bcymax, nx)?;

    let ilinx = splinck_db(x, 1.0e-3, "x")?;
    let iliny = splinck_db(y, 1.0e-3, "y")?;

    // Check whether inhomogeneous y-boundary conditions require Step 3.
    let mut iflg2 = false;
    if ibcymin != BicubicBc::Periodic {
        if matches!(ibcymin, BicubicBc::FirstDeriv | BicubicBc::SecondDeriv)
            && bcymin.iter().any(|&v| v != 0.0)
        {
            iflg2 = true;
        }
        if matches!(ibcymax, BicubicBc::FirstDeriv | BicubicBc::SecondDeriv)
            && bcymax.iter().any(|&v| v != 0.0)
        {
            iflg2 = true;
        }
    }

    let mut coeffs = vec![[0.0_f64; 4]; nx * ny];
    for i in 0..(nx * ny) {
        coeffs[i][0] = z[i];
    }

    // Step 1: 1D spline along x for each y-slice to compute f_xx.
    let mut x_slice = vec![0.0_f64; nx];
    for iy in 0..ny {
        for ix in 0..nx {
            x_slice[ix] = coeffs[ix * ny + iy][0];
        }
        let zbcmin = if matches!(
            ibcxmin,
            BicubicBc::FirstDeriv | BicubicBc::SecondDeriv
        ) {
            bcxmin[iy]
        } else {
            0.0
        };
        let zbcmax = if matches!(
            ibcxmax,
            BicubicBc::FirstDeriv | BicubicBc::SecondDeriv
        ) {
            bcxmax[iy]
        } else {
            0.0
        };
        let (fxx, _) =
            mkspline_db(x, &x_slice, ibcxmin, zbcmin, ibcxmax, zbcmax)?;
        for ix in 0..nx {
            coeffs[ix * ny + iy][1] = fxx[ix];
        }
    }

    let ibcmin_hom = if iflg2
        && matches!(ibcymin, BicubicBc::FirstDeriv | BicubicBc::SecondDeriv)
    {
        BicubicBc::NotAKnot
    } else {
        ibcymin
    };
    let ibcmax_hom = if iflg2
        && matches!(ibcymax, BicubicBc::FirstDeriv | BicubicBc::SecondDeriv)
    {
        BicubicBc::NotAKnot
    } else {
        ibcymax
    };

    // Step 2: Homogeneous 1D spline along y for each x-slice (f_yy and f_xxyy).
    let mut y_slice = vec![0.0_f64; ny];
    for ix in 0..nx {
        for iy in 0..ny {
            y_slice[iy] = coeffs[ix * ny + iy][0];
        }
        let (fyy, _) =
            mkspline_db(y, &y_slice, ibcmin_hom, 0.0, ibcmax_hom, 0.0)?;
        for iy in 0..ny {
            coeffs[ix * ny + iy][2] = fyy[iy];
        }

        for iy in 0..ny {
            y_slice[iy] = coeffs[ix * ny + iy][1];
        }
        let (fxxyy, _) =
            mkspline_db(y, &y_slice, ibcmin_hom, 0.0, ibcmax_hom, 0.0)?;
        for iy in 0..ny {
            coeffs[ix * ny + iy][3] = fxxyy[iy];
        }
    }

    // Step 3: Correct f_yy and f_xxyy for inhomogeneous y boundary conditions.
    // First compute the discrepancy (zdiff1, zdiff2) between the requested
    // y-boundary derivatives and those of the homogeneous spline from Step 2,
    // solve a correction spline fcorr1 along y on zero knot values, and then
    // spline fcorr1 along x to update f_xxyy.
    if iflg2 {
        let mut fcorr1 = vec![0.0_f64; nx * ny];
        let zeros_y = vec![0.0_f64; ny];
        for ix in 0..nx {
            let zdiff1 = match ibcymin {
                BicubicBc::FirstDeriv => {
                    let hy = y[1] - y[0];
                    let cur = (coeffs[ix * ny + 1][0] - coeffs[ix * ny][0])
                        / hy
                        + hy * (-2.0 * coeffs[ix * ny][2]
                            - coeffs[ix * ny + 1][2])
                            / 6.0;
                    bcymin[ix] - cur
                }
                BicubicBc::SecondDeriv => bcymin[ix] - coeffs[ix * ny][2],
                _ => 0.0,
            };
            let zdiff2 = match ibcymax {
                BicubicBc::FirstDeriv => {
                    let hy = y[ny - 1] - y[ny - 2];
                    let cur = (coeffs[ix * ny + ny - 1][0]
                        - coeffs[ix * ny + ny - 2][0])
                        / hy
                        + hy * (2.0 * coeffs[ix * ny + ny - 1][2]
                            + coeffs[ix * ny + ny - 2][2])
                            / 6.0;
                    bcymax[ix] - cur
                }
                BicubicBc::SecondDeriv => {
                    bcymax[ix] - coeffs[ix * ny + ny - 1][2]
                }
                _ => 0.0,
            };
            let (corr_y, _) =
                mkspline_db(y, &zeros_y, ibcymin, zdiff1, ibcymax, zdiff2)?;
            for iy in 0..ny {
                fcorr1[ix * ny + iy] = corr_y[iy];
            }
        }

        for iy in 0..ny {
            for ix in 0..nx {
                x_slice[ix] = fcorr1[ix * ny + iy];
            }
            let (corr_xy, _) =
                mkspline_db(x, &x_slice, ibcxmin, 0.0, ibcxmax, 0.0)?;
            for ix in 0..nx {
                coeffs[ix * ny + iy][2] += fcorr1[ix * ny + iy];
                coeffs[ix * ny + iy][3] += corr_xy[ix];
            }
        }
    }

    Ok((coeffs, ilinx, iliny))
}
