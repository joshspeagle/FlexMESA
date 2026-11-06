//! Pre-validated 2D bicubic spline evaluation kernel (`bicub_fast`).
//!
//! Hoists $O(n_x n_y)$ grid monotonicity (`splinck_db`) and coefficient
//! finiteness checks outside the zone loop so Stage 2 and Stage 3 benchmarks
//! measure pure 16-term Hermite cell evaluation across $N_{\mathrm{zones}}$
//! queries.

/// Evaluates the 16-term Hermite bicubic expansion on a pre-validated uniform
/// rectangular grid `(x, y, coeffs)` without repeating $O(n_x n_y)$ table
/// validation on every query point.
///
/// Returns `[f, df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]`.
#[inline]
pub fn eval_bicub_prevalidated(
    xget: f64,
    yget: f64,
    x: &[f64],
    y: &[f64],
    coeffs: &[[f64; 4]],
) -> [f64; 6] {
    let nx = x.len();
    let ny = y.len();
    let zx = xget.clamp(x[0], x[nx - 1]);
    let zy = yget.clamp(y[0], y[ny - 1]);
    let dx_avg = (x[nx - 1] - x[0]) / ((nx - 1) as f64);
    let dy_avg = (y[ny - 1] - y[0]) / ((ny - 1) as f64);
    let mut i = (((zx - x[0]) / dx_avg) as usize).min(nx - 2);
    if zx < x[i] && i > 0 {
        i -= 1;
    } else if zx > x[i + 1] && i + 1 < nx - 1 {
        i += 1;
    }
    let mut j = (((zy - y[0]) / dy_avg) as usize).min(ny - 2);
    if zy < y[j] && j > 0 {
        j -= 1;
    } else if zy > y[j + 1] && j + 1 < ny - 1 {
        j += 1;
    }
    let hx = x[i + 1] - x[i];
    let hxi = 1.0 / hx;
    let xp = (zx - x[i]) * hxi;
    let xpi = 1.0 - xp;
    let hy = y[j + 1] - y[j];
    let hyi = 1.0 / hy;
    let yp = (zy - y[j]) * hyi;
    let ypi = 1.0 - yp;

    let xp2 = xp * xp;
    let xpi2 = xpi * xpi;
    let cx = xp * (xp2 - 1.0);
    let cxi = xpi * (xpi2 - 1.0);
    let cxd = 3.0 * xp2 - 1.0;
    let cxdi = -3.0 * xpi2 + 1.0;
    let hx2 = hx * hx;

    let yp2 = yp * yp;
    let ypi2 = ypi * ypi;
    let cy = yp * (yp2 - 1.0);
    let cyi = ypi * (ypi2 - 1.0);
    let cyd = 3.0 * yp2 - 1.0;
    let cydi = -3.0 * ypi2 + 1.0;
    let hy2 = hy * hy;

    let sixth = 1.0 / 6.0;
    let z36th = 1.0 / 36.0;
    let c00 = coeffs[i * ny + j];
    let c01 = coeffs[i * ny + j + 1];
    let c10 = coeffs[(i + 1) * ny + j];
    let c11 = coeffs[(i + 1) * ny + j + 1];

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

    [f, df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]
}
