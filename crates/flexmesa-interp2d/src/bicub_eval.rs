//! Two-dimensional bicubic spline evaluation (`herm2xy_db`, `interp_evbicub_db`,
//! and `interp_evbicub_mask_db`).

use crate::{BicubicEval, Interp2dError};
use flexmesa_utils::is_bad;

/// Locates the grid cell `(i, j)` and normalized local coordinates
/// `(xparam, yparam, hx, hxi, hy, hyi)` for query `(xget, yget)` matching
/// upstream `herm2xy_db`.
///
/// # Errors
///
/// Returns [`Interp2dError`] if `(xget, yget)` lies outside the grid beyond
/// relative tolerance $4 \times 10^{-7}$.
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

#[inline]
fn locate_axis_cell(axis: &[f64], val: f64, uniform: bool) -> usize {
    let n = axis.len();
    let nm = n - 1;
    if uniform {
        let raw = ((nm as f64) * (val - axis[0]) / (axis[nm] - axis[0])) as usize;
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

    let xpi = 1.0 - xp;
    let xp2 = xp * xp;
    let xpi2 = xpi * xpi;
    let cx = xp * (xp2 - 1.0);
    let cxi = xpi * (xpi2 - 1.0);
    let cxd = 3.0 * xp2 - 1.0;
    let cxdi = -3.0 * xpi2 + 1.0;
    let hx2 = hx * hx;

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

    let f = xpi * (ypi * c00[0] + yp * c01[0])
        + xp * (ypi * c10[0] + yp * c11[0])
        + sixth
            * hx2
            * (cxi * (ypi * c00[1] + yp * c01[1])
                + cx * (ypi * c10[1] + yp * c11[1]))
        + sixth
            * hy2
            * (xpi * (cyi * c00[2] + cy * c01[2])
                + xp * (cyi * c10[2] + cy * c11[2]))
        + z36th
            * hx2
            * hy2
            * (cxi * (cyi * c00[3] + cy * c01[3])
                + cx * (cyi * c10[3] + cy * c11[3]));

    let df_dx = hxi
        * (-(ypi * c00[0] + yp * c01[0]) + (ypi * c10[0] + yp * c11[0]))
        + sixth
            * hx
            * (cxdi * (ypi * c00[1] + yp * c01[1])
                + cxd * (ypi * c10[1] + yp * c11[1]))
        + sixth
            * hxi
            * hy2
            * (-(cyi * c00[2] + cy * c01[2]) + (cyi * c10[2] + cy * c11[2]))
        + z36th
            * hx
            * hy2
            * (cxdi * (cyi * c00[3] + cy * c01[3])
                + cxd * (cyi * c10[3] + cy * c11[3]));

    let df_dy = hyi
        * (xpi * (-c00[0] + c01[0]) + xp * (-c10[0] + c11[0]))
        + sixth
            * hx2
            * hyi
            * (cxi * (-c00[1] + c01[1]) + cx * (-c10[1] + c11[1]))
        + sixth
            * hy
            * (xpi * (cydi * c00[2] + cyd * c01[2])
                + xp * (cydi * c10[2] + cyd * c11[2]))
        + z36th
            * hx2
            * hy
            * (cxi * (cydi * c00[3] + cyd * c01[3])
                + cx * (cydi * c10[3] + cyd * c11[3]));

    let d2f_dx2 = xpi * (ypi * c00[1] + yp * c01[1])
        + xp * (ypi * c10[1] + yp * c11[1])
        + sixth
            * hy2
            * (xpi * (cyi * c00[3] + cy * c01[3])
                + xp * (cyi * c10[3] + cy * c11[3]));

    let d2f_dy2 = xpi * (ypi * c00[2] + yp * c01[2])
        + xp * (ypi * c10[2] + yp * c11[2])
        + sixth
            * hx2
            * (cxi * (ypi * c00[3] + yp * c01[3])
                + cx * (ypi * c10[3] + yp * c11[3]));

    let d2f_dxdy = hxi * hyi * (c00[0] - c01[0] - c10[0] + c11[0])
        + sixth
            * hx
            * hyi
            * (cxdi * (-c00[1] + c01[1]) + cxd * (-c10[1] + c11[1]))
        + sixth
            * hxi
            * hy
            * (-(cydi * c00[2] + cyd * c01[2])
                + (cydi * c10[2] + cyd * c11[2]))
        + z36th
            * hx
            * hy
            * (cxdi * (cydi * c00[3] + cyd * c01[3])
                + cxd * (cydi * c10[3] + cyd * c11[3]));

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
/// the 6-element selector `ict` (`[f, df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]`)
/// into a 6-element output array `fval` matching upstream Fortran semantics.
///
/// # Errors
///
/// Returns [`Interp2dError`] under the same conditions as [`interp_evbicub_db`].
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
