// SPDX-License-Identifier: LGPL-3.0-or-later
//! Derivative-free asteroseismic optimization drivers: Nelder-Mead simplex
//! (`flexmesa_num::nm_simplex`), Powell trust-region `NEWUOA` / `BOBYQA`
//! (`flexmesa_num::{newuoa, bobyqa}`), and Cartesian grid scans
//! (`astero/private/astero_run_support.f90:366-982`).

use flexmesa_num::{bobyqa, newuoa, nm_simplex, NumError, SimplexConfig};
use flexmesa_star::StarError;

/// Minimizes a scalar objective `eval_fn(x)` over bounded parameter space
/// `x_lower <= x <= x_upper` using `flexmesa_num::nm_simplex`
/// (`mod_simplex.f90:30-384`, `astero_run_support.f90:710-808`).
///
/// Returns `(x_best, f_best, num_evaluations)`.
#[allow(clippy::too_many_arguments)]
pub fn nelder_mead_minimize<F>(
    x_first: &[f64],
    x_lower: &[f64],
    x_upper: &[f64],
    max_evals: usize,
    x_atol: f64,
    x_rtol: f64,
    chi2_tol: f64,
    centroid_weight_power: f64,
    eval_fn: F,
) -> (Vec<f64>, f64, usize)
where
    F: FnMut(&[f64]) -> f64,
{
    simplex_minimize_with_options(
        x_first,
        x_lower,
        x_upper,
        max_evals,
        x_atol,
        x_rtol,
        chi2_tol,
        centroid_weight_power,
        true,
        false,
        107_461_921,
        eval_fn,
    )
}

/// Full-featured Nelder-Mead wrapper around `flexmesa_num::nm_simplex`
/// supporting `simplex_enforce_bounds`, `simplex_adaptive_random_search`, and `simplex_seed`.
#[allow(clippy::too_many_arguments)]
pub fn simplex_minimize_with_options<F>(
    x_first: &[f64],
    x_lower: &[f64],
    x_upper: &[f64],
    max_evals: usize,
    x_atol: f64,
    x_rtol: f64,
    chi2_tol: f64,
    centroid_weight_power: f64,
    enforce_bounds: bool,
    adaptive_random_search: bool,
    seed: i32,
    mut eval_fn: F,
) -> (Vec<f64>, f64, usize)
where
    F: FnMut(&[f64]) -> f64,
{
    try_simplex_minimize_with_options(
        x_first,
        x_lower,
        x_upper,
        max_evals,
        x_atol,
        x_rtol,
        chi2_tol,
        centroid_weight_power,
        enforce_bounds,
        adaptive_random_search,
        seed,
        |x| Ok(eval_fn(x)),
    )
    .unwrap_or_else(|e| panic!("simplex_minimize_with_options failed: {e}"))
}

/// Fallible Nelder-Mead minimization propagating objective or optimizer errors.
#[allow(clippy::too_many_arguments)]
pub fn try_simplex_minimize_with_options<F>(
    x_first: &[f64],
    x_lower: &[f64],
    x_upper: &[f64],
    max_evals: usize,
    x_atol: f64,
    x_rtol: f64,
    chi2_tol: f64,
    centroid_weight_power: f64,
    enforce_bounds: bool,
    adaptive_random_search: bool,
    seed: i32,
    eval_fn: F,
) -> Result<(Vec<f64>, f64, usize), StarError>
where
    F: FnMut(&[f64]) -> Result<f64, StarError>,
{
    try_simplex_minimize_with_budget(
        x_first,
        x_lower,
        x_upper,
        max_evals,
        max_evals,
        x_atol,
        x_rtol,
        chi2_tol,
        centroid_weight_power,
        enforce_bounds,
        adaptive_random_search,
        seed,
        eval_fn,
    )
}

/// Fallible Nelder-Mead minimization with explicit `max_iters` and `max_evals`
/// matching `NM_simplex` (`num/private/mod_simplex.f90:30-257`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_simplex_minimize_with_budget<F>(
    x_first: &[f64],
    x_lower: &[f64],
    x_upper: &[f64],
    max_iters: usize,
    max_evals: usize,
    x_atol: f64,
    x_rtol: f64,
    chi2_tol: f64,
    centroid_weight_power: f64,
    enforce_bounds: bool,
    adaptive_random_search: bool,
    seed: i32,
    mut eval_fn: F,
) -> Result<(Vec<f64>, f64, usize), StarError>
where
    F: FnMut(&[f64]) -> Result<f64, StarError>,
{
    let n = x_first.len().min(x_lower.len()).min(x_upper.len());
    if n == 0 || max_evals == 0 {
        return Ok((Vec::new(), 0.0, 0));
    }
    let cfg = SimplexConfig {
        max_iters,
        max_evals,
        x_tol: x_rtol.max(x_atol).max(1.0e-12),
        f_tol: chi2_tol.max(0.0),
        alpha: 1.0,
        beta: 2.0,
        gamma: 0.5,
        delta: 0.5,
        enforce_bounds,
        adaptive_random_search,
        seed,
        centroid_weight_power,
    };
    // Upstream `num/private/mod_simplex.f90:218-236` sets `simplex(:, n+1) = x_first`
    // without clamping when `enforce_bounds == false` (allowing initial trial
    // coordinates outside `[x_lower, x_upper]` while using the span for scale).
    let x_init: Vec<f64> = if enforce_bounds {
        (0..n)
            .map(|i| x_first[i].clamp(x_lower[i], x_upper[i]))
            .collect()
    } else {
        x_first[..n].to_vec()
    };
    let mut best_x = x_init.clone();
    let mut best_f = f64::INFINITY;
    let mut evals = 0usize;
    let mut cb_err: Option<StarError> = None;
    let res =
        nm_simplex(n, &x_lower[..n], &x_upper[..n], &x_init, &cfg, |x_try| {
            let x_eval: Vec<f64> = if enforce_bounds {
                (0..n)
                    .map(|i| x_try[i].clamp(x_lower[i], x_upper[i]))
                    .collect()
            } else {
                x_try[..n].to_vec()
            };
            let f = match eval_fn(&x_eval) {
                Ok(v) => v,
                Err(e) => {
                    cb_err = Some(e);
                    return Err(NumError::ConvergenceFailed(
                        "simplex callback error",
                    ));
                }
            };
            evals += 1;
            if f < best_f {
                best_f = f;
                best_x = x_eval;
            }
            Ok(f)
        });
    if let Some(err) = cb_err {
        return Err(err);
    }
    match res {
        Ok(r) => {
            if r.f_min < best_f {
                best_f = r.f_min;
                best_x = r.x_min;
            }
        }
        Err(NumError::ConvergenceFailed(_)) => {}
        Err(other) => {
            return Err(StarError::ConvergenceFailure(format!(
                "nm_simplex failed: {other}"
            )));
        }
    }
    Ok((best_x, best_f, evals))
}

/// Maps normalized trust-region coordinate $u_i \in [-1, 1]$ to physical
/// parameter $p_i \in [p_{\min,i}, p_{\max,i}]$ with
/// $u_i = 0 \mapsto p_{\mathrm{first},i}$, matching `bobyqa_param` in
/// `astero_run_support.f90:631-638`.
#[inline]
#[must_use]
pub fn map_bobyqa_param(
    u: f64,
    p_first: f64,
    p_lower: f64,
    p_upper: f64,
) -> f64 {
    let u_c = u.clamp(-1.0, 1.0);
    if u_c > 0.0 {
        p_first + u_c * (p_upper - p_first)
    } else {
        p_first + u_c * (p_first - p_lower)
    }
}

/// Inverse of [`map_bobyqa_param`], mapping physical parameter $p_i$ to
/// normalized trust-region coordinate $u_i \in [-1, 1]$.
#[inline]
#[must_use]
pub fn unmap_bobyqa_param(
    p: f64,
    p_first: f64,
    p_lower: f64,
    p_upper: f64,
) -> f64 {
    if p >= p_first {
        let span = (p_upper - p_first).max(1.0e-30);
        ((p - p_first) / span).clamp(-1.0, 1.0)
    } else {
        let span = (p_first - p_lower).max(1.0e-30);
        ((p - p_first) / span).clamp(-1.0, 1.0)
    }
}

/// Minimizes `eval_fn(x)` using `flexmesa_num::newuoa` in normalized unit-cube
/// coordinates (`do_bobyqa_or_newuoa`, `astero_run_support.f90:534-707`).
pub fn newuoa_minimize<F>(
    x_first: &[f64],
    x_lower: &[f64],
    x_upper: &[f64],
    max_evals: usize,
    rhobeg: f64,
    rhoend: f64,
    mut eval_fn: F,
) -> (Vec<f64>, f64, usize)
where
    F: FnMut(&[f64]) -> f64,
{
    try_newuoa_minimize(
        x_first,
        x_lower,
        x_upper,
        max_evals,
        rhobeg,
        rhoend,
        |x| Ok(eval_fn(x)),
    )
    .unwrap_or_else(|e| panic!("newuoa_minimize failed: {e}"))
}

/// Fallible `NEWUOA` minimization propagating objective or optimizer errors.
pub fn try_newuoa_minimize<F>(
    x_first: &[f64],
    x_lower: &[f64],
    x_upper: &[f64],
    max_evals: usize,
    rhobeg: f64,
    rhoend: f64,
    mut eval_fn: F,
) -> Result<(Vec<f64>, f64, usize), StarError>
where
    F: FnMut(&[f64]) -> Result<f64, StarError>,
{
    let n = x_first.len().min(x_lower.len()).min(x_upper.len());
    if n == 0 || max_evals == 0 {
        return Ok((Vec::new(), 0.0, 0));
    }
    let n_eff = n.max(2);
    let npt = 2 * n_eff + 1;
    let mut u = vec![0.0_f64; n_eff];
    let r_beg = rhobeg.max(1.0e-6);
    let r_end = rhoend.clamp(1.0e-12, r_beg);
    let max_evals_eff = if rhoend >= r_beg {
        max_evals.min(npt + 1)
    } else {
        max_evals
    };
    let mut best_x: Vec<f64> = (0..n)
        .map(|i| x_first[i].clamp(x_lower[i], x_upper[i]))
        .collect();
    let mut best_f = f64::INFINITY;
    let mut evals = 0usize;
    let mut cb_err: Option<StarError> = None;

    let res = newuoa(
        n_eff,
        npt,
        &mut u,
        r_beg,
        r_end,
        max_evals_eff,
        1.0e10,
        |u_try| -> Result<f64, NumError> {
            if evals >= max_evals_eff {
                return Ok(best_f.min(1.0e10));
            }
            let out_pen: f64 = u_try[..n]
                .iter()
                .map(|&ui| flexmesa_math::pow2((ui.abs() - 1.0).max(0.0)))
                .sum();
            let phys: Vec<f64> = (0..n)
                .map(|i| {
                    map_bobyqa_param(
                        u_try[i], x_first[i], x_lower[i], x_upper[i],
                    )
                })
                .collect();
            let f = match eval_fn(&phys) {
                Ok(v) => v,
                Err(e) => {
                    cb_err = Some(e);
                    return Err(NumError::ConvergenceFailed(
                        "newuoa callback error",
                    ));
                }
            };
            evals += 1;
            if f < best_f {
                best_f = f;
                best_x = phys;
            }
            let f_pen = f + 1.0e4 * out_pen;
            Ok(if n_eff > n {
                f_pen + 1.0e-6 * flexmesa_math::pow2(u_try[n])
            } else {
                f_pen
            })
        },
    );
    if let Some(err) = cb_err {
        return Err(err);
    }
    match res {
        Ok(_) | Err(NumError::ConvergenceFailed(_)) => {}
        Err(other) => {
            return Err(StarError::ConvergenceFailure(format!(
                "newuoa failed: {other}"
            )));
        }
    }
    Ok((best_x, best_f, evals))
}

/// Minimizes `eval_fn(x)` using `flexmesa_num::bobyqa` on $[-1, 1]^n$ normalized
/// unit-cube coordinates (`do_bobyqa_or_newuoa`, `astero_run_support.f90:534-707`).
pub fn bobyqa_minimize<F>(
    x_first: &[f64],
    x_lower: &[f64],
    x_upper: &[f64],
    max_evals: usize,
    rhobeg: f64,
    rhoend: f64,
    mut eval_fn: F,
) -> (Vec<f64>, f64, usize)
where
    F: FnMut(&[f64]) -> f64,
{
    try_bobyqa_minimize(
        x_first,
        x_lower,
        x_upper,
        max_evals,
        rhobeg,
        rhoend,
        |x| Ok(eval_fn(x)),
    )
    .unwrap_or_else(|e| panic!("bobyqa_minimize failed: {e}"))
}

/// Fallible `BOBYQA` minimization propagating objective or optimizer errors.
pub fn try_bobyqa_minimize<F>(
    x_first: &[f64],
    x_lower: &[f64],
    x_upper: &[f64],
    max_evals: usize,
    rhobeg: f64,
    rhoend: f64,
    mut eval_fn: F,
) -> Result<(Vec<f64>, f64, usize), StarError>
where
    F: FnMut(&[f64]) -> Result<f64, StarError>,
{
    let n = x_first.len().min(x_lower.len()).min(x_upper.len());
    if n == 0 || max_evals == 0 {
        return Ok((Vec::new(), 0.0, 0));
    }
    let n_eff = n.max(2);
    let npt = 2 * n_eff + 1;
    let mut u = vec![0.0_f64; n_eff];
    let xl = vec![-1.0_f64; n_eff];
    let xu = vec![1.0_f64; n_eff];
    let r_beg = rhobeg.clamp(1.0e-6, 0.95);
    let r_end = rhoend.clamp(1.0e-12, r_beg);
    let max_evals_eff = if rhoend >= r_beg {
        max_evals.min(npt + 1)
    } else {
        max_evals
    };
    let mut best_x: Vec<f64> = (0..n)
        .map(|i| x_first[i].clamp(x_lower[i], x_upper[i]))
        .collect();
    let mut best_f = f64::INFINITY;
    let mut evals = 0usize;
    let mut cb_err: Option<StarError> = None;

    let res = bobyqa(
        n_eff,
        npt,
        &mut u,
        &xl,
        &xu,
        r_beg,
        r_end,
        max_evals_eff,
        1.0e10,
        |u_try| -> Result<f64, NumError> {
            if evals >= max_evals_eff {
                return Ok(best_f.min(1.0e10));
            }
            let phys: Vec<f64> = (0..n)
                .map(|i| {
                    map_bobyqa_param(
                        u_try[i], x_first[i], x_lower[i], x_upper[i],
                    )
                })
                .collect();
            let f = match eval_fn(&phys) {
                Ok(v) => v,
                Err(e) => {
                    cb_err = Some(e);
                    return Err(NumError::ConvergenceFailed(
                        "bobyqa callback error",
                    ));
                }
            };
            evals += 1;
            if f < best_f {
                best_f = f;
                best_x = phys;
            }
            Ok(if n_eff > n {
                f + 1.0e-6 * flexmesa_math::pow2(u_try[n])
            } else {
                f
            })
        },
    );
    if let Some(err) = cb_err {
        return Err(err);
    }
    match res {
        Ok(_) | Err(NumError::ConvergenceFailed(_)) => {}
        Err(other) => {
            return Err(StarError::ConvergenceFailure(format!(
                "bobyqa failed: {other}"
            )));
        }
    }
    Ok((best_x, best_f, evals))
}

/// Minimizes `eval_fn(x)` using Powell's `BOBYQA` trust-region quadratic
/// interpolation in normalized `bobyqa_param` coordinates
/// (`do_bobyqa_or_newuoa`, `astero_run_support.f90:534-707`).
pub fn powell_quadratic_minimize<F>(
    x_first: &[f64],
    x_lower: &[f64],
    x_upper: &[f64],
    max_evals: usize,
    rhobeg: f64,
    rhoend: f64,
    eval_fn: F,
) -> (Vec<f64>, f64, usize)
where
    F: FnMut(&[f64]) -> f64,
{
    bobyqa_minimize(
        x_first, x_lower, x_upper, max_evals, rhobeg, rhoend, eval_fn,
    )
}

/// Evaluates `eval_fn` across a Cartesian grid `x_lower..=x_upper` with step
/// sizes `x_delta` (`do_scan_grid`, `astero_run_support.f90:366-531`).
///
/// Returns `(x_best, f_best, all_points, all_values)`.
pub fn scan_cartesian_grid<F>(
    x_lower: &[f64],
    x_upper: &[f64],
    x_delta: &[f64],
    mut eval_fn: F,
) -> (Vec<f64>, f64, Vec<Vec<f64>>, Vec<f64>)
where
    F: FnMut(&[f64]) -> f64,
{
    let n = x_lower.len().min(x_upper.len()).min(x_delta.len());
    if n == 0 {
        return (Vec::new(), 0.0, Vec::new(), Vec::new());
    }
    let mut axes: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let step = x_delta[i].abs().max(1.0e-12);
        let mut vals = Vec::new();
        let mut x = x_lower[i];
        while x <= x_upper[i] + 1.0e-6 * step {
            vals.push(x);
            x += step;
        }
        if vals.is_empty() {
            vals.push(x_lower[i]);
        }
        axes.push(vals);
    }

    let total: usize = axes.iter().map(Vec::len).product();
    let mut all_pts = Vec::with_capacity(total);
    let mut all_vals = Vec::with_capacity(total);
    let mut best_x = Vec::new();
    let mut best_f = f64::INFINITY;

    let mut idx = vec![0usize; n];
    for _ in 0..total {
        let pt: Vec<f64> = (0..n).map(|k| axes[k][idx[k]]).collect();
        let f = eval_fn(&pt);
        if f < best_f {
            best_f = f;
            best_x.clone_from(&pt);
        }
        all_pts.push(pt);
        all_vals.push(f);

        for k in (0..n).rev() {
            idx[k] += 1;
            if idx[k] < axes[k].len() {
                break;
            }
            idx[k] = 0;
        }
    }
    (best_x, best_f, all_pts, all_vals)
}
