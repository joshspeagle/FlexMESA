// SPDX-License-Identifier: GPL-3.0-or-later
//! 2D bicubic spline evaluation and grid-location helpers for `flexmesa-kap`.
//!
//! Mirrors `kap/private/kap_eval_support.f90` (`locate_log`, `locate_logR`,
//! `locate_logT`, and `do_kap_interpolations`) and `kap/public/kap_def.f90`
//! (`get_output_string`).

use crate::types::KapError;
use flexmesa_const::ONE_SIXTH;
use flexmesa_num::binary_search;

const Z36TH: f64 = 1.0 / 36.0;

/// Format a composition value $X$ or $Z \in [0, 1]$ into MESA's filename token
/// (`get_output_string` in `kap_def.f90`).
///
/// # Errors
/// Returns [`KapError::InvalidComposition`] if `x` is outside `[0.0, 1.0]`.
pub fn get_output_string(x: f64) -> Result<String, KapError> {
    if !(0.0..=1.0).contains(&x) {
        return Err(KapError::InvalidComposition("x outside [0, 1]"));
    }
    let s = format!("{x:8.6}");
    let mut k = 0_usize;
    for (idx, ch) in s.char_indices() {
        if ('1'..='9').contains(&ch) {
            k = k.max(idx + 1);
        }
    }
    Ok(s[..k.max(3)].trim().to_string())
}

/// Locate a coordinate `log_find` in a 1D grid `logs`, clipping to `[log_min_in, log_max_in]`
/// (`locate_log` in `kap_eval_support.f90`).
///
/// Returns `(i, log_find_clipped, log0, log1)` where `i` is 0-based (`0 <= i < num_logs - 1`).
///
/// # Errors
/// Returns [`KapError::OutOfBounds`] if `logs` is empty or search fails.
pub fn locate_log(
    log_min_in: f64,
    log_max_in: f64,
    ili_logs: bool,
    logs: &[f64],
    mut log_find: f64,
    clip_to_boundaries: bool,
) -> Result<(usize, f64, f64, f64), KapError> {
    let num_logs = logs.len();
    if num_logs == 0 {
        return Err(KapError::OutOfBounds("empty coordinate grid"));
    }
    let log_min = log_min_in.max(logs[0]);
    let log_max = log_max_in.min(logs[num_logs - 1]);
    if num_logs == 1 {
        log_find = log_min;
        return Ok((0, log_find, log_find, log_find));
    }
    let mut i: usize;
    if log_find < log_min || log_find > log_max {
        if !clip_to_boundaries {
            return Err(KapError::OutOfBounds(
                "coordinate outside table boundaries",
            ));
        }
        if log_find < log_min {
            i = 0;
            log_find = log_min;
        } else {
            i = num_logs - 2;
            log_find = log_max;
        }
    } else if (log_find - log_max).abs() < 1.0e-7 {
        i = num_logs - 2;
        log_find = log_max;
    } else if ili_logs {
        let dlog = (log_max - log_min) / ((num_logs - 1) as f64);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            i = ((log_find - log_min) / dlog) as usize;
        }
        if i >= num_logs - 1 {
            i = num_logs - 2;
        }
        if logs[i] > log_find && i > 0 {
            i -= 1;
        } else if log_find >= logs[i + 1] && i + 1 < num_logs - 1 {
            i += 1;
        }
    } else if let Some(idx) = binary_search(logs, log_find, None) {
        i = idx;
        if i >= num_logs - 1 {
            return Err(KapError::OutOfBounds("binary_search out of range"));
        }
    } else {
        i = 0;
    }

    if i >= num_logs - 1 || logs[i] > log_find || log_find > logs[i + 1] {
        return Err(KapError::OutOfBounds("locate_log cell check failed"));
    }
    let log0 = logs[i];
    let log1 = logs[i + 1];
    if !log0.is_finite() || !log1.is_finite() {
        return Err(KapError::NumericalFailure(
            "non-finite grid knot in locate_log",
        ));
    }
    Ok((i, log_find, log0, log1))
}

/// Evaluate the PSPLINE bicubic spline cell `(i, j)` and its first partial derivatives
/// (`do_kap_interpolations` in `kap_eval_support.f90`).
///
/// * `fin1` has row-major `(nx, ny)` layout of `[f, f_xx, f_yy, f_xxyy]`:
///   `fin(c, ix, jy) = fin1[ix * ny + jy][c]`.
/// * `i` and `j` are 0-based cell indices (`0 <= i < nx - 1`, `0 <= j < ny - 1`).
///
/// Returns `(fval, df_dx, df_dy)`.
#[must_use]
pub fn do_kap_interpolations(
    fin1: &[[f64; 4]],
    _nx: usize,
    ny: usize,
    i: usize,
    j: usize,
    x0: f64,
    xget: f64,
    x1: f64,
    y0: f64,
    yget: f64,
    y1: f64,
) -> (f64, f64, f64) {
    let idx = |c: usize, ix: usize, jy: usize| -> f64 { fin1[ix * ny + jy][c] };

    let hx = x1 - x0;
    let hxi = 1.0 / hx;
    let hx2 = hx * hx;

    let xp = (xget - x0) * hxi;
    let xpi = 1.0 - xp;
    let xp2 = xp * xp;
    let xpi2 = xpi * xpi;

    let cx = xp * (xp2 - 1.0);
    let cxi = xpi * (xpi2 - 1.0);
    let cxd = 3.0 * xp2 - 1.0;
    let cxdi = -3.0 * xpi2 + 1.0;

    let hy = y1 - y0;
    let hyi = 1.0 / hy;
    let hy2 = hy * hy;

    let yp = (yget - y0) * hyi;
    let ypi = 1.0 - yp;
    let yp2 = yp * yp;
    let ypi2 = ypi * ypi;

    let cy = yp * (yp2 - 1.0);
    let cyi = ypi * (ypi2 - 1.0);
    let cyd = 3.0 * yp2 - 1.0;
    let cydi = -3.0 * ypi2 + 1.0;

    let fval = xpi * (ypi * idx(0, i, j) + yp * idx(0, i, j + 1))
        + xp * (ypi * idx(0, i + 1, j) + yp * idx(0, i + 1, j + 1))
        + ONE_SIXTH
            * hx2
            * (cxi * (ypi * idx(1, i, j) + yp * idx(1, i, j + 1))
                + cx * (ypi * idx(1, i + 1, j) + yp * idx(1, i + 1, j + 1)))
        + ONE_SIXTH
            * hy2
            * (xpi * (cyi * idx(2, i, j) + cy * idx(2, i, j + 1))
                + xp * (cyi * idx(2, i + 1, j) + cy * idx(2, i + 1, j + 1)))
        + Z36TH
            * hx2
            * hy2
            * (cxi * (cyi * idx(3, i, j) + cy * idx(3, i, j + 1))
                + cx * (cyi * idx(3, i + 1, j) + cy * idx(3, i + 1, j + 1)));

    let df_dx = hxi
        * (-(ypi * idx(0, i, j) + yp * idx(0, i, j + 1))
            + (ypi * idx(0, i + 1, j) + yp * idx(0, i + 1, j + 1)))
        + ONE_SIXTH
            * hx
            * (cxdi * (ypi * idx(1, i, j) + yp * idx(1, i, j + 1))
                + cxd * (ypi * idx(1, i + 1, j) + yp * idx(1, i + 1, j + 1)))
        + ONE_SIXTH
            * hxi
            * hy2
            * (-(cyi * idx(2, i, j) + cy * idx(2, i, j + 1))
                + (cyi * idx(2, i + 1, j) + cy * idx(2, i + 1, j + 1)))
        + Z36TH
            * hx
            * hy2
            * (cxdi * (cyi * idx(3, i, j) + cy * idx(3, i, j + 1))
                + cxd * (cyi * idx(3, i + 1, j) + cy * idx(3, i + 1, j + 1)));

    let df_dy = hyi
        * (xpi * (-idx(0, i, j) + idx(0, i, j + 1))
            + xp * (-idx(0, i + 1, j) + idx(0, i + 1, j + 1)))
        + ONE_SIXTH
            * hx2
            * hyi
            * (cxi * (-idx(1, i, j) + idx(1, i, j + 1))
                + cx * (-idx(1, i + 1, j) + idx(1, i + 1, j + 1)))
        + ONE_SIXTH
            * hy
            * (xpi * (cydi * idx(2, i, j) + cyd * idx(2, i, j + 1))
                + xp * (cydi * idx(2, i + 1, j) + cyd * idx(2, i + 1, j + 1)))
        + Z36TH
            * hx2
            * hy
            * (cxi * (cydi * idx(3, i, j) + cyd * idx(3, i, j + 1))
                + cx * (cydi * idx(3, i + 1, j) + cyd * idx(3, i + 1, j + 1)));

    (fval, df_dx, df_dy)
}

fn check_grid_uniform(grid: &[f64]) -> Result<bool, KapError> {
    let n = grid.len();
    if n < 4 || grid.iter().any(|v| !v.is_finite()) {
        return Err(KapError::NumericalFailure(
            "invalid grid coordinates for bicubic spline",
        ));
    }
    let dxavg = (grid[n - 1] - grid[0]) / ((n - 1) as f64);
    let zeps = (1.0e-3 * dxavg).abs();
    let mut uniform = true;
    for i in 1..n {
        let diff = grid[i] - grid[i - 1];
        if diff <= 0.0 {
            return Err(KapError::NumericalFailure(
                "non-ascending grid coordinates for bicubic spline",
            ));
        }
        if (diff - dxavg).abs() > zeps {
            uniform = false;
        }
    }
    Ok(uniform)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum KapSplineBc {
    NotAKnot,
    ZeroFirstDeriv,
}

/// 1D $C^2$ cubic spline matching upstream PSPLINE / MESA's
/// `mkspline_db` -> `cspline_db` -> `V_SPLINE_db` floating-point evaluation
/// order (including `cspline_db`'s `(half, sixth)` scaling and `(s_next - s_i) / h`
/// before multiplying by `6.0`, preserving 0-ULP edge endpoint parity when
/// $x_{N-1} - x_{N-2}$ is not a power of 2; `bicub_db.f:1228-1231, 3047-3051, 3256-3390`).
fn mkspline_1d(x: &[f64], y: &[f64], bc: KapSplineBc) -> Vec<f64> {
    let n = x.len();
    let mut f = vec![[0.0_f64; 4]; n];
    for (node, &val) in f.iter_mut().zip(y.iter()) {
        node[0] = val;
    }

    f[0][3] = x[1] - x[0];
    f[1][2] = (f[1][0] - f[0][0]) / f[0][3];
    for i in 1..(n - 1) {
        f[i][3] = x[i + 1] - x[i];
        f[i][1] = 2.0 * (f[i - 1][3] + f[i][3]);
        f[i + 1][2] = (f[i + 1][0] - f[i][0]) / f[i][3];
        f[i][2] = f[i + 1][2] - f[i][2];
    }

    let (imin, imax) = match bc {
        KapSplineBc::NotAKnot => {
            // Not-a-knot boundary conditions (`imin = 1`, `imax = n - 2` in 0-based).
            f[1][1] = f[0][3] + 2.0 * f[1][3];
            f[1][2] = f[1][2] * f[1][3] / (f[0][3] + f[1][3]);
            f[n - 2][1] = 2.0 * f[n - 3][3] + f[n - 2][3];
            f[n - 2][2] =
                f[n - 2][2] * f[n - 3][3] / (f[n - 2][3] + f[n - 3][3]);
            (1, n - 2)
        }
        KapSplineBc::ZeroFirstDeriv => {
            // Homogeneous zero-slope boundary conditions (`k_bc1 = k_bcn = 3`,
            // `imin = 0`, `imax = n - 1` in 0-based; `bicub_db.f:3256-3275`).
            f[0][1] = 2.0 * f[0][3];
            f[0][2] = (f[1][0] - f[0][0]) / f[0][3] - 0.0;
            f[n - 1][1] = 2.0 * f[n - 2][3];
            f[n - 1][2] = -(f[n - 1][0] - f[n - 2][0]) / f[n - 2][3] + 0.0;
            (0, n - 1)
        }
    };

    for i in (imin + 1)..=imax {
        let t = if bc == KapSplineBc::NotAKnot && i == imax {
            (f[i - 1][3] - f[i][3]) / f[i - 1][1]
        } else {
            f[i - 1][3] / f[i - 1][1]
        };
        let diag_off = if bc == KapSplineBc::NotAKnot && i == imin + 1 {
            f[i - 1][3] - f[i - 2][3]
        } else {
            f[i - 1][3]
        };
        f[i][1] -= t * diag_off;
        f[i][2] -= t * f[i - 1][2];
    }

    f[imax][2] /= f[imax][1];
    for ib in 1..=(imax - imin) {
        let i = imax - ib;
        let upper = if bc == KapSplineBc::NotAKnot && i == imin {
            f[i][3] - f[i - 1][3]
        } else {
            f[i][3]
        };
        f[i][2] = (f[i][2] - upper * f[i + 1][2]) / f[i][1];
    }

    if bc == KapSplineBc::NotAKnot {
        f[0][3] = x[1] - x[0];
        f[n - 2][3] = x[n - 1] - x[n - 2];
        f[0][2] = (f[1][2] * (f[0][3] + f[1][3]) - f[2][2] * f[0][3]) / f[1][3];
        f[n - 1][2] = f[n - 2][2]
            + (f[n - 2][2] - f[n - 3][2]) * f[n - 2][3] / f[n - 3][3];
    }

    for i in 0..(n - 1) {
        let h = f[i][3];
        let s_i = f[i][2];
        let s_next = f[i + 1][2];
        f[i][1] = (f[i + 1][0] - f[i][0]) / h - h * (s_next + 2.0 * s_i);
        f[i][3] = (s_next - s_i) / h;
        f[i][2] = 6.0 * s_i;
        f[i][3] *= 6.0;
    }
    for node in &mut f {
        node[2] *= 0.5;
        node[3] *= ONE_SIXTH;
    }

    let mut fxx = vec![0.0_f64; n];
    for (dst, node) in fxx[..(n - 1)].iter_mut().zip(&f[..(n - 1)]) {
        *dst = 2.0 * node[2];
    }
    fxx[n - 1] = 2.0 * f[n - 2][2] + (x[n - 1] - x[n - 2]) * 6.0 * f[n - 2][3];
    fxx
}

/// Fit a 2D PSPLINE bicubic spline with not-a-knot boundary conditions (`ibc = 0`)
/// from a flat `(num_log_ts, num_log_rs)` row-major array (`raw_vals[j_t * nx + i_r]`).
///
/// Returns `(coeffs, ili_log_rs, ili_log_ts)`.
///
/// # Errors
/// Returns [`KapError::NumericalFailure`] if grid coordinates are invalid or `nx`/`ny < 4`.
pub fn fit_bicubic_not_a_knot(
    log_rs: &[f64],
    log_ts: &[f64],
    raw_vals: &[f64],
) -> Result<(Vec<[f64; 4]>, bool, bool), KapError> {
    fit_bicubic_with_bc(log_rs, log_ts, raw_vals, KapSplineBc::NotAKnot)
}

/// Fit a 2D PSPLINE bicubic spline with homogeneous zero-slope boundary conditions
/// (`ibc = 3`, used by `condint.f90:115-125`) from a flat `(nx, ny)` row-major array
/// (`raw_vals[i_r * ny + j_t]`).
///
/// Returns `(coeffs, ili_log_rs, ili_log_ts)`.
///
/// # Errors
/// Returns [`KapError::NumericalFailure`] if grid coordinates are invalid or `nx`/`ny < 4`.
pub fn fit_bicubic_zero_first_deriv(
    log_rs: &[f64],
    log_ts: &[f64],
    raw_vals: &[f64],
) -> Result<(Vec<[f64; 4]>, bool, bool), KapError> {
    fit_bicubic_with_bc(log_rs, log_ts, raw_vals, KapSplineBc::ZeroFirstDeriv)
}

fn fit_bicubic_with_bc(
    log_rs: &[f64],
    log_ts: &[f64],
    raw_vals: &[f64],
    bc: KapSplineBc,
) -> Result<(Vec<[f64; 4]>, bool, bool), KapError> {
    let nx = log_rs.len();
    let ny = log_ts.len();
    if raw_vals.len() != nx * ny {
        return Err(KapError::NumericalFailure(
            "mismatched raw_vals length in fit_bicubic",
        ));
    }
    let ili_x = check_grid_uniform(log_rs)?;
    let ili_y = check_grid_uniform(log_ts)?;

    let mut coeffs = vec![[0.0_f64; 4]; nx * ny];
    match bc {
        KapSplineBc::NotAKnot => {
            for j_t in 0..ny {
                for i_r in 0..nx {
                    let v = raw_vals[j_t * nx + i_r];
                    coeffs[i_r * ny + j_t][0] =
                        if v.is_finite() { v } else { 0.0 };
                }
            }
        }
        KapSplineBc::ZeroFirstDeriv => {
            for (dst, &v) in coeffs.iter_mut().zip(raw_vals.iter()) {
                dst[0] = if v.is_finite() { v } else { 0.0 };
            }
        }
    }

    let mut x_slice = vec![0.0_f64; nx];
    for j_t in 0..ny {
        for i_r in 0..nx {
            x_slice[i_r] = coeffs[i_r * ny + j_t][0];
        }
        let fxx = mkspline_1d(log_rs, &x_slice, bc);
        for i_r in 0..nx {
            coeffs[i_r * ny + j_t][1] = fxx[i_r];
        }
    }

    let mut y_slice = vec![0.0_f64; ny];
    for i_r in 0..nx {
        for j_t in 0..ny {
            y_slice[j_t] = coeffs[i_r * ny + j_t][0];
        }
        let fyy = mkspline_1d(log_ts, &y_slice, bc);
        for j_t in 0..ny {
            coeffs[i_r * ny + j_t][2] = fyy[j_t];
        }

        for j_t in 0..ny {
            y_slice[j_t] = coeffs[i_r * ny + j_t][1];
        }
        let fxxyy = mkspline_1d(log_ts, &y_slice, bc);
        for j_t in 0..ny {
            coeffs[i_r * ny + j_t][3] = fxxyy[j_t];
        }
    }

    for node in &mut coeffs {
        for v in node {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
    }
    Ok((coeffs, ili_x, ili_y))
}
