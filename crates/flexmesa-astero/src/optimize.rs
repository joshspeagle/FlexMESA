// SPDX-License-Identifier: GPL-3.0-or-later
//! Derivative-free asteroseismic optimization drivers: Nelder-Mead simplex
//! (`num/private/mod_simplex.f90`), Powell trust-region `NEWUOA` / `BOBYQA`
//! (`flexmesa_num::{newuoa, bobyqa}`), and Cartesian grid scans
//! (`astero/private/astero_run_support.f90:366-982`).

use flexmesa_num::{bobyqa, newuoa, r8_uniform_01, NumError};
use flexmesa_star::StarError;

/// Minimizes a scalar objective `eval_fn(x)` over bounded parameter space
/// `x_lower <= x <= x_upper` using MESA's Nelder-Mead simplex algorithm
/// (`num/private/mod_simplex.f90:30-384`,
/// `astero/private/astero_run_support.f90:710-808`).
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

/// Full-featured Nelder-Mead wrapper (`num/private/mod_simplex.f90`)
/// supporting `simplex_enforce_bounds`, `simplex_adaptive_random_search`, and
/// `simplex_seed`.
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
    eval_fn: F,
) -> Result<(Vec<f64>, f64, usize), StarError>
where
    F: FnMut(&[f64]) -> Result<f64, StarError>,
{
    try_simplex_minimize_with_config(
        x_first,
        x_lower,
        x_upper,
        max_iters,
        max_evals,
        x_atol,
        x_rtol,
        chi2_tol,
        centroid_weight_power,
        enforce_bounds,
        adaptive_random_search,
        seed,
        1.0,
        2.0,
        0.5,
        0.5,
        None,
        eval_fn,
    )
}

/// Full Nelder-Mead minimization supporting upstream `simplex_alpha/beta/gamma/delta`
/// and warm-starting from an existing `(simplex, f)` (`restart_simplex_from_file`,
/// `astero/private/astero_run_support.f90:858-980`, `num/private/mod_simplex.f90:30-384`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_simplex_minimize_with_config<F>(
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
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
    initial_simplex: Option<(Vec<Vec<f64>>, Vec<f64>)>,
    mut eval_fn: F,
) -> Result<(Vec<f64>, f64, usize), StarError>
where
    F: FnMut(&[f64]) -> Result<f64, StarError>,
{
    let n = x_first.len().min(x_lower.len()).min(x_upper.len());
    if n == 0 || max_evals == 0 {
        return Ok((Vec::new(), 0.0, 0));
    }
    if let Some((simplex_in, f_in)) = initial_simplex {
        return run_simplex_loop(
            n,
            &x_lower[..n],
            &x_upper[..n],
            simplex_in,
            f_in,
            0,
            max_iters,
            max_evals,
            x_atol,
            x_rtol,
            chi2_tol,
            centroid_weight_power,
            enforce_bounds,
            adaptive_random_search,
            seed,
            alpha,
            beta,
            gamma,
            delta,
            eval_fn,
        );
    }

    // Port `set_initial_simplex` from `num/private/mod_simplex.f90:210-257`:
    // column `n` holds un-clamped `x_first`, and each displaced vertex `j`
    // retries halving toward `x_first` up to 20 times on callback error.
    let mut num_evals = 0usize;
    let mut simplex = vec![vec![0.0_f64; n]; n + 1];
    let mut f = vec![0.0_f64; n + 1];
    simplex[n].copy_from_slice(&x_first[..n]);
    let (v_last, early_last) = eval_simplex_point(
        n,
        &simplex[n],
        &x_lower[..n],
        &x_upper[..n],
        enforce_bounds,
        chi2_tol,
        &mut num_evals,
        &mut eval_fn,
    )?;
    f[n] = v_last;
    if early_last {
        return Ok((simplex[n].clone(), v_last, num_evals));
    }

    for j in 0..n {
        simplex[j].copy_from_slice(&x_first[..n]);
        let span = x_upper[j] - x_lower[j];
        simplex[j][j] = if x_upper[j] - x_first[j] < x_first[j] - x_lower[j] {
            x_first[j] - 0.25 * span
        } else {
            x_first[j] + 0.25 * span
        };
        let mut okay = false;
        let mut last_err: Option<StarError> = None;
        for _ in 0..20 {
            match eval_simplex_point(
                n,
                &simplex[j],
                &x_lower[..n],
                &x_upper[..n],
                enforce_bounds,
                chi2_tol,
                &mut num_evals,
                &mut eval_fn,
            ) {
                Ok((val, early)) => {
                    f[j] = val;
                    okay = true;
                    if early {
                        let best_idx = best_simplex_vertex(j, n, &f);
                        return Ok((
                            simplex[best_idx][..n].to_vec(),
                            f[best_idx],
                            num_evals,
                        ));
                    }
                    break;
                }
                Err(err) => {
                    last_err = Some(err);
                    simplex[j][j] = 0.5 * (simplex[j][j] + x_first[j]);
                    if (simplex[j][j] - x_first[j]).abs()
                        < 1.0e-12 * (1.0 + x_first[j].abs())
                    {
                        break;
                    }
                }
            }
        }
        if !okay {
            return Err(last_err.unwrap_or_else(|| {
                StarError::ConvergenceFailure(
                    "failed to get value for initial simplex point".to_string(),
                )
            }));
        }
    }

    run_simplex_loop(
        n,
        &x_lower[..n],
        &x_upper[..n],
        simplex,
        f,
        num_evals,
        max_iters,
        max_evals,
        x_atol,
        x_rtol,
        chi2_tol,
        centroid_weight_power,
        enforce_bounds,
        adaptive_random_search,
        seed,
        alpha,
        beta,
        gamma,
        delta,
        eval_fn,
    )
}

fn best_simplex_vertex(up_to: usize, n: usize, f: &[f64]) -> usize {
    let mut best = n;
    for idx in 0..=up_to {
        if f[idx] < f[best] {
            best = idx;
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn eval_simplex_point<F>(
    n: usize,
    pt: &[f64],
    x_lower: &[f64],
    x_upper: &[f64],
    enforce_bounds: bool,
    chi2_tol: f64,
    evals: &mut usize,
    eval_fn: &mut F,
) -> Result<(f64, bool), StarError>
where
    F: FnMut(&[f64]) -> Result<f64, StarError>,
{
    if enforce_bounds {
        for j in 0..n {
            if pt[j] < x_lower[j] || pt[j] > x_upper[j] {
                return Ok((1.0e99, false));
            }
        }
    }
    *evals += 1;
    let val = eval_fn(&pt[..n])?;
    Ok((val, val < chi2_tol))
}

#[allow(clippy::too_many_arguments)]
fn run_simplex_loop<F>(
    n: usize,
    x_lower: &[f64],
    x_upper: &[f64],
    mut simplex: Vec<Vec<f64>>,
    mut f: Vec<f64>,
    mut num_evals: usize,
    max_iters: usize,
    max_evals: usize,
    x_atol: f64,
    x_rtol: f64,
    chi2_tol: f64,
    centroid_weight_power: f64,
    enforce_bounds: bool,
    adaptive_random_search: bool,
    mut seed: i32,
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
    mut eval_fn: F,
) -> Result<(Vec<f64>, f64, usize), StarError>
where
    F: FnMut(&[f64]) -> Result<f64, StarError>,
{
    if simplex.len() != n + 1 || f.len() != n + 1 {
        return Err(StarError::InvalidInput(format!(
            "simplex restart requires n+1={} vertices, got {} and {}",
            n + 1,
            simplex.len(),
            f.len()
        )));
    }
    let mut num_iters = 0usize;
    let mut l_best = 0usize;
    for j in 1..=n {
        if f[j] < f[l_best] {
            l_best = j;
        }
    }
    let mut best_x = simplex[l_best][..n].to_vec();
    let mut best_f = f[l_best];
    if best_f < chi2_tol {
        return Ok((best_x, best_f, num_evals));
    }

    macro_rules! eval_and_check {
        ($pt:expr) => {{
            let (val, hit_tol) = eval_simplex_point(
                n,
                $pt,
                x_lower,
                x_upper,
                enforce_bounds,
                chi2_tol,
                &mut num_evals,
                &mut eval_fn,
            )?;
            let valid = !enforce_bounds
                || (0..n).all(|j| $pt[j] >= x_lower[j] && $pt[j] <= x_upper[j]);
            if valid && val < best_f {
                best_f = val;
                best_x = $pt[..n].to_vec();
            }
            if hit_tol {
                return Ok((best_x, best_f, num_evals));
            }
            val
        }};
    }

    loop {
        num_iters += 1;
        let (mut h, mut l) = (0usize, 0usize);
        for j in 1..=n {
            if f[j] > f[h] {
                h = j;
            }
            if f[j] < f[l] {
                l = j;
            }
        }
        let mut s = None;
        for j in 0..=n {
            if j == h {
                continue;
            }
            if s.is_none_or(|si| f[j] > f[si]) {
                s = Some(j);
            }
        }
        let s = s.unwrap_or(l);
        let mut term_val_x = 0.0_f64;
        for j in 0..=n {
            if j == l {
                continue;
            }
            for i in 0..n {
                let denom = x_atol
                    + x_rtol * simplex[j][i].abs().max(simplex[l][i].abs());
                if denom > 0.0 {
                    let t = (simplex[j][i] - simplex[l][i]).abs() / denom;
                    if t > term_val_x {
                        term_val_x = t;
                    }
                }
            }
        }
        if term_val_x <= 1.0 || num_iters > max_iters || num_evals > max_evals {
            if f[l] < best_f {
                best_f = f[l];
                best_x = simplex[l][..n].to_vec();
            }
            break;
        }
        let mut c = vec![0.0_f64; n];
        let mut sum_weight = 0.0_f64;
        for j in 0..=n {
            if j == h {
                continue;
            }
            let w = if centroid_weight_power == 0.0 {
                1.0
            } else {
                let inv = 1.0 / f[j].max(1.0e-30);
                if (centroid_weight_power - 1.0).abs() <= 1.0e-15 {
                    inv
                } else {
                    flexmesa_math::pow(inv, centroid_weight_power)
                }
            };
            for i in 0..n {
                c[i] += simplex[j][i] * w;
            }
            sum_weight += w;
        }
        for ci in &mut c {
            *ci /= sum_weight;
        }
        let x_reflect: Vec<f64> = (0..n)
            .map(|i| c[i] + alpha * (c[i] - simplex[h][i]))
            .collect();
        let f_reflect = eval_and_check!(&x_reflect);
        if f_reflect < f[s] {
            simplex[h].clone_from(&x_reflect);
            f[h] = f_reflect;
            if f_reflect < f[l] {
                let x_expand: Vec<f64> = (0..n)
                    .map(|i| c[i] + beta * (x_reflect[i] - c[i]))
                    .collect();
                let f_expand = eval_and_check!(&x_expand);
                if f_expand < f[l] {
                    simplex[h] = x_expand;
                    f[h] = f_expand;
                }
            }
        } else {
            let x_contract: Vec<f64> = if f_reflect < f[h] {
                (0..n)
                    .map(|i| c[i] + gamma * (x_reflect[i] - c[i]))
                    .collect()
            } else {
                (0..n)
                    .map(|i| c[i] + gamma * (simplex[h][i] - c[i]))
                    .collect()
            };
            let f_contract = eval_and_check!(&x_contract);
            if f_contract < f[h].min(f_reflect) {
                simplex[h] = x_contract;
                f[h] = f_contract;
            } else if adaptive_random_search {
                for _ in 0..100 {
                    if num_evals > max_evals {
                        break;
                    }
                    let x_ars: Vec<f64> = (0..n)
                        .map(|i| {
                            x_lower[i]
                                + r8_uniform_01(&mut seed)
                                    * (x_upper[i] - x_lower[i])
                        })
                        .collect();
                    let f_ars = eval_and_check!(&x_ars);
                    if f_ars <= f[h] {
                        simplex[h] = x_ars;
                        f[h] = f_ars;
                        break;
                    }
                }
            } else {
                for j in 0..=n {
                    if j == l {
                        continue;
                    }
                    for i in 0..n {
                        simplex[j][i] = simplex[l][i]
                            + delta * (simplex[j][i] - simplex[l][i]);
                    }
                    f[j] = eval_and_check!(&simplex[j]);
                }
            }
        }
    }
    Ok((best_x, best_f, num_evals))
}

/// Maps trust-region coordinate $u_i$ to physical parameter $p_i$ with
/// $u_i = 0 \mapsto p_{\mathrm{first},i}$, matching `bobyqa_param` in
/// `astero/private/astero_run_support.f90:631-638` (unclamped so `NEWUOA`
/// can extrapolate while `BOBYQA` enforces $u_i \in [-1, 1]$ via box bounds).
#[inline]
#[must_use]
pub fn map_bobyqa_param(
    u: f64,
    p_first: f64,
    p_lower: f64,
    p_upper: f64,
) -> f64 {
    if u > 0.0 {
        p_first + u * (p_upper - p_first)
    } else {
        p_first + u * (p_first - p_lower)
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
/// coordinates (`do_bobyqa_or_newuoa`,
/// `astero/private/astero_run_support.f90:534-707`).
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
    // Upstream `do_newuoa` (`num/private/mod_newuoa.f:48-77, 663`) and
    // `do_bobyqa_or_newuoa` (`astero/private/astero_run_support.f90:687-695`)
    // pass `rhoend` un-clamped, allowing `newuoa_rhoend = 1.0 > rhobeg = 0.45`
    // (e.g. `astero/fast_newuoa`) to stop once the initial `rho = rhobeg` cycle
    // finishes.
    let r_end = rhoend.max(1.0e-12);
    let max_evals_eff = max_evals;
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
        1.0e6,
        |u_try| -> Result<f64, NumError> {
            if evals >= max_evals_eff {
                return Ok(best_f.min(1.0e6));
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
                        "newuoa callback error",
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
                "newuoa failed: {other}"
            )));
        }
    }
    Ok((best_x, best_f, evals))
}

/// Minimizes `eval_fn(x)` using `flexmesa_num::bobyqa` on $[-1, 1]^n$ normalized
/// unit-cube coordinates (`do_bobyqa_or_newuoa`,
/// `astero/private/astero_run_support.f90:534-707`).
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
    // Upstream `do_BOBYQA` (`num/private/mod_bobyqa.f:58-137, 844`) does not
    // clamp `rhoend` to `rhobeg`.
    let r_end = rhoend.max(1.0e-12);
    let max_evals_eff = max_evals;
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
        1.0e6,
        |u_try| -> Result<f64, NumError> {
            if evals >= max_evals_eff {
                return Ok(best_f.min(1.0e6));
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
/// (`do_bobyqa_or_newuoa`, `astero/private/astero_run_support.f90:534-707`).
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
/// sizes `x_delta` (`do_scan_grid`,
/// `astero/private/astero_run_support.f90:366-531`).
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
