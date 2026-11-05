//! Two-dimensional bicubic spline evaluation (`herm2xy_db`,
//! `interp_evbicub_db`, and `interp_evbicub_mask_db`).
//!
//! Evaluates the 6-component bicubic spline vector
//! `(f, df/dx, df/dy, d2f/dx2, d2f/dy2, d2f/dxdy)` in full `f64` precision from
//! the coefficient table `[f, f_xx, f_yy, f_xxyy]` produced by
//! [`crate::interp_mkbicub_db`].
//!
//! # Precision Contract
//!
//! Upstream MESA's `fvbicub_db` in `interp_2d/private/bicub_db.f` omits
//! `implicit none`, leaving its local coordinate and cubic basis variables
//! (`xp`, `yp`, `cx`, `cy`, `cxd`, `cyd`, `hx2`, `hy2`, `z36th`) implicitly
//! typed as 32-bit `REAL` (`f32`). `FlexMESA` evaluates all basis terms in full
//! `f64` precision so analytical first and second partial derivatives satisfy
//! exact polynomial calculus to $\le 10^{-13}$.
//!
//! # Grid Lookup Complexity
//!
//! When `ilinx` or `iliny` is `true`, cell lookup uses $O(1)$ direct index
//! calculation with adjacent-knot verification; otherwise, it falls back to
//! $O(\log N)$ binary search via `slice::partition_point`.

use crate::{BicubicEval, Interp2dError};
use flexmesa_utils::is_bad;

/// Locates the grid cell `(i, j)` and normalized local coordinates
/// `(xparam, yparam, hx, hxi, hy, hyi)` for query `(xget, yget)` matching
/// upstream `herm2xy_db`.
///
/// Queries slightly outside `[x[0], x[nx - 1]]` or `[y[0], y[ny - 1]]` within
/// relative tolerance $4 \times 10^{-7}$ are clamped to the boundary.
///
/// # Errors
///
/// Returns [`Interp2dError`] if `(xget, yget)` is non-finite or lies outside
/// the grid beyond relative tolerance $4 \times 10^{-7}$.
pub fn herm2xy_db(
    xget: f64,
    yget: f64,
    x: &[f64],
    y: &[f64],
    ilinx: bool,
    iliny: bool,
) -> Result<(usize, usize, f64, f64, f64, f64, f64, f64), Interp2dError> {
    if is_bad(xget) || is_bad(yget) {
        return Err(Interp2dError::NonFiniteInput);
    }
    let nx = x.len();
    let ny = y.len();
    let mut zxget = xget;
    if xget < x[0] || xget > x[nx - 1] {
        let zxtol = 4.0e-7 * x[0].abs().max(x[nx - 1].abs());
        if xget < x[0] - zxtol || xget > x[nx - 1] + zxtol {
            return Err(Interp2dError::OutOfRange {
                axis: "x",
                value: xget,
            });
        }
        zxget = xget.clamp(x[0], x[nx - 1]);
    }
    let mut zyget = yget;
    if yget < y[0] || yget > y[ny - 1] {
        let zytol = 4.0e-7 * y[0].abs().max(y[ny - 1].abs());
        if yget < y[0] - zytol || yget > y[ny - 1] + zytol {
            return Err(Interp2dError::OutOfRange {
                axis: "y",
                value: yget,
            });
        }
        zyget = yget.clamp(y[0], y[ny - 1]);
    }

    let i = locate_axis_cell(x, zxget, ilinx);
    let j = locate_axis_cell(y, zyget, iliny);

    let hx = x[i + 1] - x[i];
    let hxi = 1.0 / hx;
    let xparam = (zxget - x[i]) * hxi;

    let hy = y[j + 1] - y[j];
    let hyi = 1.0 / hy;
    let yparam = (zyget - y[j]) * hyi;

    Ok((i, j, xparam, yparam, hx, hxi, hy, hyi))
}

/// Locates the 0-based cell index along `axis` in $O(1)$ for uniform grids or
/// $O(\log N)$ via binary search for non-uniform grids.
#[inline]
fn locate_axis_cell(axis: &[f64], val: f64, uniform: bool) -> usize {
    let n = axis.len();
    let nm = n - 1;
    if uniform {
        let raw =
            ((nm as f64) * (val - axis[0]) / (axis[nm] - axis[0])) as usize;
        let mut idx = raw.min(nm - 1);
        if val < axis[idx] && idx > 0 {
            idx -= 1;
        } else if val > axis[idx + 1] && idx + 1 < nm {
            idx += 1;
        }
        idx
    } else {
        let pos = axis.partition_point(|&xk| xk < val);
        if pos == 0 {
            0
        } else if pos >= n {
            nm - 1
        } else {
            pos - 1
        }
    }
}

/// Evaluates the 2D bicubic spline value and all first and second partial
/// derivatives at `(xget, yget)` (`interp_evbicub_db` / `fvbicub_db`).
///
/// # Errors
///
/// Returns [`Interp2dError`] if `(xget, yget)` is out of range or `coeffs`
/// dimensions do not match `x.len() * y.len()`.
pub fn interp_evbicub_db(
    xget: f64,
    yget: f64,
    x: &[f64],
    y: &[f64],
    ilinx: bool,
    iliny: bool,
    coeffs: &[[f64; 4]],
) -> Result<BicubicEval, Interp2dError> {
    let nx = x.len();
    let ny = y.len();
    if coeffs.len() != nx * ny {
        return Err(Interp2dError::DimensionMismatch {
            expected: nx * ny,
            actual: coeffs.len(),
        });
    }
    let (i, j, xp, yp, hx, hxi, hy, hyi) =
        herm2xy_db(xget, yget, x, y, ilinx, iliny)?;

    let sixth = 1.0_f64 / 6.0;
    let z36th = sixth * sixth;

    // Cubic Hermite basis polynomials and derivatives along x.
    let xpi = 1.0 - xp;
    let xp2 = xp * xp;
    let xpi2 = xpi * xpi;
    let cx = xp * (xp2 - 1.0);
    let cxi = xpi * (xpi2 - 1.0);
    let cxd = 3.0 * xp2 - 1.0;
    let cxdi = -3.0 * xpi2 + 1.0;
    let hx2 = hx * hx;

    // Cubic Hermite basis polynomials and derivatives along y.
    let ypi = 1.0 - yp;
    let yp2 = yp * yp;
    let ypi2 = ypi * ypi;
    let cy = yp * (yp2 - 1.0);
    let cyi = ypi * (ypi2 - 1.0);
    let cyd = 3.0 * yp2 - 1.0;
    let cydi = -3.0 * ypi2 + 1.0;
    let hy2 = hy * hy;

    let c00 = &coeffs[i * ny + j];
    let c01 = &coeffs[i * ny + j + 1];
    let c10 = &coeffs[(i + 1) * ny + j];
    let c11 = &coeffs[(i + 1) * ny + j + 1];

    // Precompute shared y-basis combinations on left (`0`) and right (`1`)
    // cell boundaries to eliminate redundant arithmetic across the 6 outputs.
    let y0_f = ypi * c00[0] + yp * c01[0];
    let y1_f = ypi * c10[0] + yp * c11[0];
    let y0_fxx = ypi * c00[1] + yp * c01[1];
    let y1_fxx = ypi * c10[1] + yp * c11[1];
    let y0_fyy = ypi * c00[2] + yp * c01[2];
    let y1_fyy = ypi * c10[2] + yp * c11[2];
    let y0_fxxyy = ypi * c00[3] + yp * c01[3];
    let y1_fxxyy = ypi * c10[3] + yp * c11[3];

    let cy0_fyy = cyi * c00[2] + cy * c01[2];
    let cy1_fyy = cyi * c10[2] + cy * c11[2];
    let cy0_fxxyy = cyi * c00[3] + cy * c01[3];
    let cy1_fxxyy = cyi * c10[3] + cy * c11[3];

    let cyd0_fyy = cydi * c00[2] + cyd * c01[2];
    let cyd1_fyy = cydi * c10[2] + cyd * c11[2];
    let cyd0_fxxyy = cydi * c00[3] + cyd * c01[3];
    let cyd1_fxxyy = cydi * c10[3] + cyd * c11[3];

    let df0_y = c01[0] - c00[0];
    let df1_y = c11[0] - c10[0];
    let dfxx0_y = c01[1] - c00[1];
    let dfxx1_y = c11[1] - c10[1];

    let f = xpi * y0_f
        + xp * y1_f
        + sixth * hx2 * (cxi * y0_fxx + cx * y1_fxx)
        + sixth * hy2 * (xpi * cy0_fyy + xp * cy1_fyy)
        + z36th * hx2 * hy2 * (cxi * cy0_fxxyy + cx * cy1_fxxyy);

    let df_dx = hxi * (y1_f - y0_f)
        + sixth * hx * (cxdi * y0_fxx + cxd * y1_fxx)
        + sixth * hxi * hy2 * (cy1_fyy - cy0_fyy)
        + z36th * hx * hy2 * (cxdi * cy0_fxxyy + cxd * cy1_fxxyy);

    let df_dy = hyi * (xpi * df0_y + xp * df1_y)
        + sixth * hx2 * hyi * (cxi * dfxx0_y + cx * dfxx1_y)
        + sixth * hy * (xpi * cyd0_fyy + xp * cyd1_fyy)
        + z36th * hx2 * hy * (cxi * cyd0_fxxyy + cx * cyd1_fxxyy);

    let d2f_dx2 = xpi * y0_fxx
        + xp * y1_fxx
        + sixth * hy2 * (xpi * cy0_fxxyy + xp * cy1_fxxyy);

    let d2f_dy2 = xpi * y0_fyy
        + xp * y1_fyy
        + sixth * hx2 * (cxi * y0_fxxyy + cx * y1_fxxyy);

    let d2f_dxdy = hxi * hyi * (df1_y - df0_y)
        + sixth * hx * hyi * (cxdi * dfxx0_y + cxd * dfxx1_y)
        + sixth * hxi * hy * (cyd1_fyy - cyd0_fyy)
        + z36th * hx * hy * (cxdi * cyd0_fxxyy + cxd * cyd1_fxxyy);

    Ok(BicubicEval {
        f,
        df_dx,
        df_dy,
        d2f_dx2,
        d2f_dy2,
        d2f_dxdy,
    })
}

/// Evaluates `interp_evbicub_db` packing the selected derivatives according to
/// the 6-element selector `ict`
/// (`[f, df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]`) into a 6-element output
/// array `fval` matching upstream Fortran semantics.
///
/// # Errors
///
/// Returns [`Interp2dError`] under the same conditions as
/// [`interp_evbicub_db`].
pub fn interp_evbicub_mask_db(
    xget: f64,
    yget: f64,
    x: &[f64],
    y: &[f64],
    ilinx: bool,
    iliny: bool,
    coeffs: &[[f64; 4]],
    ict: &[i32; 6],
) -> Result<[f64; 6], Interp2dError> {
    let ev = interp_evbicub_db(xget, yget, x, y, ilinx, iliny, coeffs)?;
    let all = [
        ev.f,
        ev.df_dx,
        ev.df_dy,
        ev.d2f_dx2,
        ev.d2f_dy2,
        ev.d2f_dxdy,
    ];
    let mut fval = [0.0_f64; 6];
    let mut iadr = 0usize;
    for k in 0..6 {
        if ict[k] == 1 {
            fval[iadr] = all[k];
            iadr += 1;
        }
    }
    Ok(fval)
}
