//! Nelder-Mead simplex optimization and Park-Miller `ran1_zah` PRNG (`num`).
//!
//! Ports `num/private/mod_simplex.f90` (`NM_simplex`) and
//! `num/private/mod_random.f90` (`r8_uniform_01`, `i4_uniform`, `perm_uniform`)
//! into Safe Rust in `f64` precision.
//!
//! # Algorithmic Overview
//!
//! - [`r8_uniform_01`], [`i4_uniform`], and [`perm_uniform`] implement the
//!   deterministic Park-Miller (`1988`) multiplicative linear congruential
//!   generator ($a = 16807, m = 2^{31} - 1$) using Schrage's overflow-free
//!   decomposition (`ran1_zah`) used by MESA's simplex restart perturbations.
//! - [`nm_simplex`] evolves an $(n + 1)$-vertex polytope in $\mathbb{R}^n$ via
//!   reflection ($\alpha$), expansion ($\beta$), outside/inside contraction
//!   ($\gamma$), and full shrink ($\delta$) toward the current best vertex.
//!
//! # Convergence Criteria
//!
//! - Iteration terminates when both the vertex diameter relative to scale and
//!   the objective spread across the $(n + 1)$ vertices fall below tolerance,
//!   or when the evaluation/iteration budget is exhausted.

use crate::NumError;

/// Park-Miller (`ran1_zah` in `mod_random.f90`) uniform pseudo-random number
/// generator in $[0, 1)$, updating `seed` in-place.
///
/// Uses multiplier $a = 16807$, modulus $m = 2147483647 = 2^{31} - 1$, and
/// Schrage quotient/remainder $(q = 127773, r = 2836)$.
pub fn r8_uniform_01(seed: &mut i32) -> f64 {
    const IA: i64 = 16_807;
    const IM: i64 = 2_147_483_647;
    const IQ: i64 = 127_773;
    const IR: i64 = 2_836;

    if *seed == 0 {
        *seed = 123_459_876;
    }
    let s = i64::from(*seed);
    let k = s / IQ;
    let mut next = IA * (s - k * IQ) - IR * k;
    if next < 0 {
        next += IM;
    }
    *seed = next as i32;
    (next as f64) * 4.656_612_875e-10
}

/// Returns a uniform pseudo-random integer in `[a, b]` (inclusive) using
/// [`r8_uniform_01`].
pub fn i4_uniform(a: i32, b: i32, seed: &mut i32) -> i32 {
    let lo = a.min(b);
    let hi = a.max(b);
    let u = r8_uniform_01(seed);
    let span = i64::from(hi) - i64::from(lo) + 1;
    let offset = ((u * (span as f64)).floor() as i64).clamp(0, span - 1);
    (i64::from(lo) + offset) as i32
}

/// Generates a random 1-based permutation of `1..=n` using [`i4_uniform`].
#[must_use]
pub fn perm_uniform(n: usize, seed: &mut i32) -> Vec<usize> {
    let mut p: Vec<usize> = (1..=n).collect();
    for i in 0..n {
        let j =
            (i4_uniform(i as i32, (n.saturating_sub(1)) as i32, seed)) as usize;
        p.swap(i, j);
    }
    p
}

/// Configuration parameters for [`nm_simplex`] matching upstream `NM_simplex`.
#[derive(Debug, Clone, Copy)]
pub struct SimplexConfig {
    /// Maximum function evaluations (`evaluation_max`).
    pub max_evals: usize,
    /// Relative simplex diameter convergence tolerance (`x_atol` / `x_rtol`).
    pub x_tol: f64,
    /// Objective value spread convergence tolerance (`y_atol`).
    pub f_tol: f64,
    /// Reflection coefficient $\alpha$ (default `1.0`).
    pub alpha: f64,
    /// Expansion coefficient $\beta$ (default `2.0`).
    pub beta: f64,
    /// Contraction coefficient $\gamma$ (default `0.5`).
    pub gamma: f64,
    /// Shrink coefficient $\delta$ (default `0.5`).
    pub delta: f64,
    /// Whether to enforce box bounds `[x_lower, x_upper]` via penalty.
    pub enforce_bounds: bool,
    /// Whether to use MESA's adaptive random search (`ARS`) on contraction
    /// failure (`mod_simplex.f90`).
    pub adaptive_random_search: bool,
    /// Initial PRNG seed for adaptive random search (`107461921` in MESA).
    pub seed: i32,
    /// Power for inverse-objective centroid weighting (`0.0` = uniform).
    pub centroid_weight_power: f64,
}

impl Default for SimplexConfig {
    fn default() -> Self {
        Self {
            max_evals: 20_000,
            x_tol: 1e-10,
            f_tol: 0.0,
            alpha: 1.0,
            beta: 2.0,
            gamma: 0.5,
            delta: 0.5,
            enforce_bounds: false,
            adaptive_random_search: false,
            seed: 107_461_921,
            centroid_weight_power: 0.0,
        }
    }
}

/// Result returned by [`nm_simplex`].
#[derive(Debug, Clone)]
pub struct SimplexResult {
    /// Best parameter vector $x_{\min}$ of length $n$.
    pub x_min: Vec<f64>,
    /// Minimum objective value $f(x_{\min})$.
    pub f_min: f64,
    /// Total number of objective evaluations performed.
    pub num_evals: usize,
    /// Number of simplex iterations completed.
    pub num_iters: usize,
    /// Number of full simplex shrink steps performed.
    pub num_shrinks: usize,
}

/// Minimizes an $n$-dimensional objective function `fcn` over box-referenced
/// initial scales `[x_lower, x_upper]` starting from `x_first` using the
/// Nelder-Mead (`1965`) downhill simplex method (`NM_simplex` in
/// `num/private/mod_simplex.f90`).
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] if slice lengths do not match `n`, or
/// propagates errors from `fcn`.
pub fn nm_simplex<F>(
    n: usize,
    x_lower: &[f64],
    x_upper: &[f64],
    x_first: &[f64],
    config: &SimplexConfig,
    mut fcn: F,
) -> Result<SimplexResult, NumError>
where
    F: FnMut(&[f64]) -> Result<f64, NumError>,
{
    if n == 0 || x_lower.len() != n || x_upper.len() != n || x_first.len() != n
    {
        return Err(NumError::InvalidParameter(
            "x_lower, x_upper, x_first must have length n >= 1 in nm_simplex",
        ));
    }

    let mut num_evals = 0usize;
    let eval_point =
        |pt: &[f64], evals: &mut usize, f: &mut F| -> Result<f64, NumError> {
            if config.enforce_bounds {
                for j in 0..n {
                    if pt[j] < x_lower[j] || pt[j] > x_upper[j] {
                        return Ok(1e99);
                    }
                }
            }
            *evals += 1;
            f(pt)
        };

    // Initialize n + 1 simplex vertices matching mod_simplex.f90:
    // vertex n (1-based n+1) is x_first; vertices 0..n perturb axis j by
    // +/- 0.25*(x_upper[j] - x_lower[j]) toward the farther bound.
    let n_verts = n + 1;
    let mut simplex: Vec<Vec<f64>> = vec![x_first.to_vec(); n_verts];
    let mut f_vals = vec![0.0_f64; n_verts];
    f_vals[n] = eval_point(&simplex[n], &mut num_evals, &mut fcn)?;

    for j in 0..n {
        let span = x_upper[j] - x_lower[j];
        if span.abs() > 1e-14 {
            if x_upper[j] - x_first[j] < x_first[j] - x_lower[j] {
                simplex[j][j] = x_first[j] - 0.25 * span;
            } else {
                simplex[j][j] = x_first[j] + 0.25 * span;
            }
        } else {
            let step = 0.05 * x_first[j].abs().max(1.0);
            simplex[j][j] += step;
        }
        f_vals[j] = eval_point(&simplex[j], &mut num_evals, &mut fcn)?;
    }

    let mut num_iters = 0usize;
    let mut num_shrinks = 0usize;
    let mut seed = config.seed;
    let mut centroid = vec![0.0; n];
    let mut x_ref = vec![0.0; n];
    let mut x_exp = vec![0.0; n];
    let mut x_con = vec![0.0; n];
    let mut x_ars = vec![0.0; n];

    loop {
        num_iters += 1;
        // Identify h (max f), l (min f), and s (2nd max f) in place
        let mut h = 0usize;
        let mut l = 0usize;
        for j in 1..n_verts {
            if f_vals[j] > f_vals[h] {
                h = j;
            }
            if f_vals[j] < f_vals[l] {
                l = j;
            }
        }
        let mut s = usize::MAX;
        for j in 0..n_verts {
            if j == h {
                continue;
            }
            if s == usize::MAX || f_vals[j] > f_vals[s] {
                s = j;
            }
        }

        // Check domain convergence matching mod_simplex.f90 lines 114-124
        let mut term_val_x = 0.0_f64;
        for j in 0..n_verts {
            if j == l {
                continue;
            }
            for i in 0..n {
                let denom = config.x_tol
                    * (1.0 + simplex[j][i].abs().max(simplex[l][i].abs()));
                let term1 = (simplex[j][i] - simplex[l][i]).abs() / denom;
                if term1 > term_val_x {
                    term_val_x = term1;
                }
            }
        }
        if term_val_x <= 1.0
            || (config.f_tol > 0.0 && f_vals[h] - f_vals[l] <= config.f_tol)
        {
            break;
        }
        if num_iters > config.max_evals || num_evals > config.max_evals {
            break;
        }

        // Compute weighted centroid excluding worst vertex h
        centroid.fill(0.0);
        let mut sum_weight = 0.0_f64;
        for j in 0..n_verts {
            if j == h {
                continue;
            }
            let weight = if config.centroid_weight_power == 0.0 {
                1.0
            } else {
                let w = 1.0 / f_vals[j].abs().max(1e-30);
                if (config.centroid_weight_power - 1.0).abs() <= 1e-15 {
                    w
                } else {
                    w.powf(config.centroid_weight_power)
                }
            };
            for i in 0..n {
                centroid[i] += simplex[j][i] * weight;
            }
            sum_weight += weight;
        }
        for c in &mut centroid {
            *c /= sum_weight;
        }

        // Reflection step across the centroid
        for i in 0..n {
            x_ref[i] =
                centroid[i] + config.alpha * (centroid[i] - simplex[h][i]);
        }
        let f_ref = eval_point(&x_ref, &mut num_evals, &mut fcn)?;

        if f_ref < f_vals[s] {
            simplex[h].copy_from_slice(&x_ref);
            f_vals[h] = f_ref;
            if f_ref < f_vals[l] {
                for i in 0..n {
                    x_exp[i] =
                        centroid[i] + config.beta * (x_ref[i] - centroid[i]);
                }
                let f_exp = eval_point(&x_exp, &mut num_evals, &mut fcn)?;
                // Accept expand whenever f_exp < f(l) (mod_simplex.f90:167)
                if f_exp < f_vals[l] {
                    simplex[h].copy_from_slice(&x_exp);
                    f_vals[h] = f_exp;
                }
            }
        } else {
            let outside = f_ref < f_vals[h];
            for i in 0..n {
                x_con[i] = if outside {
                    centroid[i] + config.gamma * (x_ref[i] - centroid[i])
                } else {
                    centroid[i] + config.gamma * (simplex[h][i] - centroid[i])
                };
            }
            let f_con = eval_point(&x_con, &mut num_evals, &mut fcn)?;
            if f_con < f_vals[h].min(f_ref) {
                simplex[h].copy_from_slice(&x_con);
                f_vals[h] = f_con;
            } else if config.adaptive_random_search {
                for _k in 0..100 {
                    if num_evals > config.max_evals {
                        break;
                    }
                    for i in 0..n {
                        let u = r8_uniform_01(&mut seed);
                        x_ars[i] = x_lower[i] + u * (x_upper[i] - x_lower[i]);
                    }
                    let f_ars = eval_point(&x_ars, &mut num_evals, &mut fcn)?;
                    if f_ars <= f_vals[h] {
                        simplex[h].copy_from_slice(&x_ars);
                        f_vals[h] = f_ars;
                        break;
                    }
                }
            } else {
                num_shrinks += 1;
                let best = simplex[l].clone();
                for j in 0..n_verts {
                    if j == l {
                        continue;
                    }
                    for i in 0..n {
                        simplex[j][i] =
                            best[i] + config.delta * (simplex[j][i] - best[i]);
                    }
                    f_vals[j] =
                        eval_point(&simplex[j], &mut num_evals, &mut fcn)?;
                }
            }
        }
    }

    let mut best_idx = 0;
    for k in 1..n_verts {
        if f_vals[k] < f_vals[best_idx] {
            best_idx = k;
        }
    }
    Ok(SimplexResult {
        x_min: simplex[best_idx].clone(),
        f_min: f_vals[best_idx],
        num_evals,
        num_iters,
        num_shrinks,
    })
}
