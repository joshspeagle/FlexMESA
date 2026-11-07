//! 2D bicubic spline evaluation and grid-location helpers for `flexmesa-kap`.
//!
//! Mirrors `kap/private/kap_eval_support.f90` (`locate_log`, `locate_logR`,
//! `locate_logT`, and `do_kap_interpolations`) and `kap/public/kap_def.f90`
//! (`get_output_string`).

use crate::types::KapError;
use flexmesa_const::ONE_SIXTH;
use flexmesa_interp2d::{interp_mkbicub_db, BicubicBc};
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

/// Fit a 2D PSPLINE bicubic spline with not-a-knot boundary conditions (`ibc = 0`)
/// from a flat `(num_log_ts, num_log_rs)` row-major array (`raw_vals[j_t * nx + i_r]`).
///
/// Returns `(coeffs, ili_log_rs, ili_log_ts)`.
///
/// # Errors
/// Returns [`KapError::NumericalFailure`] if `interp_mkbicub_db` fails.
pub fn fit_bicubic_not_a_knot(
    log_rs: &[f64],
    log_ts: &[f64],
    raw_vals: &[f64],
) -> Result<(Vec<[f64; 4]>, bool, bool), KapError> {
    let nx = log_rs.len();
    let ny = log_ts.len();
    let mut z = vec![0.0_f64; nx * ny];
    for j_t in 0..ny {
        for i_r in 0..nx {
            let v = raw_vals[j_t * nx + i_r];
            z[i_r * ny + j_t] = if v.is_finite() { v } else { 0.0 };
        }
    }
    let (mut coeffs, ili_x, ili_y) = interp_mkbicub_db(
        log_rs,
        log_ts,
        &z,
        BicubicBc::NotAKnot,
        &[],
        BicubicBc::NotAKnot,
        &[],
        BicubicBc::NotAKnot,
        &[],
        BicubicBc::NotAKnot,
        &[],
    )
    .map_err(|_| KapError::NumericalFailure("interp_mkbicub_db failed"))?;
    for node in &mut coeffs {
        for v in node {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
    }
    Ok((coeffs, ili_x, ili_y))
}
