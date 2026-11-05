//! Two-dimensional bicubic spline setup (`interp_mkbicub_db`).

pub use crate::bicub_eval::{
    herm2xy_db, interp_evbicub_db, interp_evbicub_mask_db,
};
use crate::spline1d::{mkspline_db, splinck_db};
use crate::{BicubicBc, Interp2dError};
use flexmesa_utils::is_bad;

/// Constructs 2D bicubic spline coefficients on rectangular grid `(x, y)` of
/// shape `(nx, ny)` (`nx >= 4`, `ny >= 4`).
///
/// The input `z` slice has row-major shape `(nx, ny)` where
/// `z[ix * ny + iy] = f(x[ix], y[iy])`. On output, returns `(coeffs, ilinx, iliny)`
/// where `coeffs` has length `nx * ny` of `[f, f_xx, f_yy, f_xxyy]` at each
/// grid node `(ix, iy)`.
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
    if matches!(ibcxmin, BicubicBc::FirstDeriv | BicubicBc::SecondDeriv)
        && bcxmin.len() != ny
    {
        return Err(Interp2dError::DimensionMismatch {
            expected: ny,
            actual: bcxmin.len(),
        });
    }
    if matches!(ibcxmax, BicubicBc::FirstDeriv | BicubicBc::SecondDeriv)
        && bcxmax.len() != ny
    {
        return Err(Interp2dError::DimensionMismatch {
            expected: ny,
            actual: bcxmax.len(),
        });
    }
    if matches!(ibcymin, BicubicBc::FirstDeriv | BicubicBc::SecondDeriv)
        && bcymin.len() != nx
    {
        return Err(Interp2dError::DimensionMismatch {
            expected: nx,
            actual: bcymin.len(),
        });
    }
    if matches!(ibcymax, BicubicBc::FirstDeriv | BicubicBc::SecondDeriv)
        && bcymax.len() != nx
    {
        return Err(Interp2dError::DimensionMismatch {
            expected: nx,
            actual: bcymax.len(),
        });
    }

    let ilinx = splinck_db(x, 1.0e-3, "x")?;
    let iliny = splinck_db(y, 1.0e-3, "y")?;

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
