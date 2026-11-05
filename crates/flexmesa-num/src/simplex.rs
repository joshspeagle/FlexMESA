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
    const AM: f64 = 1.0 / (IM as f64);

    if *seed == 0 {
        *seed = 123_459_876;
    }
    let s = i64::from(*seed).abs().max(1);
    let k = s / IQ;
    let mut next = IA * (s - k * IQ) - IR * k;
    if next < 0 {
        next += IM;
    }
    *seed = next as i32;
    (AM * (next as f64)).min(1.0 - f64::EPSILON)
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
        let j = (i4_uniform(i as i32, (n.saturating_sub(1)) as i32, seed))
            as usize;
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
}

impl Default for SimplexConfig {
    fn default() -> Self {
        Self {
            max_evals: 2_000,
            x_tol: 1e-10,
            f_tol: 1e-12,
            alpha: 1.0,
            beta: 2.0,
            gamma: 0.5,
            delta: 0.5,
            enforce_bounds: false,
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
    if n == 0
        || x_lower.len() != n
        || x_upper.len() != n
        || x_first.len() != n
    {
        return Err(NumError::InvalidParameter(
            "x_lower, x_upper, x_first must have length n >= 1 in nm_simplex",
        ));
    }

    let mut num_evals = 0usize;
    let eval_point = |pt: &[f64],
                      evals: &mut usize,
                      f: &mut F|
     -> Result<f64, NumError> {
        *evals += 1;
        if config.enforce_bounds {
            for j in 0..n {
                if pt[j] < x_lower[j] || pt[j] > x_upper[j] {
                    return Ok(1e30);
                }
            }
        }
        f(pt)
    };

    // Initialize n + 1 simplex vertices by perturbing each coordinate axis j
    let n_verts = n + 1;
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n_verts);
    let mut f_vals = Vec::with_capacity(n_verts);

    simplex.push(x_first.to_vec());
    f_vals.push(eval_point(x_first, &mut num_evals, &mut fcn)?);

    for j in 0..n {
        let mut v = x_first.to_vec();
        let span = (x_upper[j] - x_lower[j]).abs().max(1.0);
        let step = if v[j].abs() > 1e-6 {
            0.05 * v[j].abs()
        } else {
            0.05 * span
        };
        v[j] += if v[j] + step <= x_upper[j] || !config.enforce_bounds {
            step
        } else {
            -step
        };
        let fv = eval_point(&v, &mut num_evals, &mut fcn)?;
        simplex.push(v);
        f_vals.push(fv);
    }

    let mut num_iters = 0usize;
    let mut num_shrinks = 0usize;
    let mut centroid = vec![0.0; n];
    let mut x_ref = vec![0.0; n];
    let mut x_exp = vec![0.0; n];
    let mut x_con = vec![0.0; n];

    while num_evals < config.max_evals {
        // Sort vertices by objective value ascending (0 = best, n = worst)
        let mut order: Vec<usize> = (0..n_verts).collect();
        order.sort_by(|&a, &b| f_vals[a].total_cmp(&f_vals[b]));
        let sorted_s: Vec<Vec<f64>> =
            order.iter().map(|&k| simplex[k].clone()).collect();
        let sorted_f: Vec<f64> = order.iter().map(|&k| f_vals[k]).collect();
        simplex = sorted_s;
        f_vals = sorted_f;

        // Check convergence in function spread and relative vertex diameter
        let f_spread = (f_vals[n] - f_vals[0]).abs();
        let mut x_diam = 0.0_f64;
        for k in 1..n_verts {
            for j in 0..n {
                let scale = simplex[0][j].abs().max(1.0);
                x_diam =
                    x_diam.max((simplex[k][j] - simplex[0][j]).abs() / scale);
            }
        }
        if f_spread <= config.f_tol && x_diam <= config.x_tol {
            break;
        }
        num_iters += 1;

        // Compute centroid of the n best vertices (excluding worst vertex n)
        centroid.fill(0.0);
        for v in simplex.iter().take(n) {
            for j in 0..n {
                centroid[j] += v[j];
            }
        }
        for c in &mut centroid {
            *c /= n as f64;
        }

        // Reflection step across the centroid
        for j in 0..n {
            x_ref[j] =
                centroid[j] + config.alpha * (centroid[j] - simplex[n][j]);
        }
        let f_ref = eval_point(&x_ref, &mut num_evals, &mut fcn)?;

        if f_ref < f_vals[0] {
            // Expansion step along the reflection ray
            for j in 0..n {
                x_exp[j] =
                    centroid[j] + config.beta * (x_ref[j] - centroid[j]);
            }
            let f_exp = eval_point(&x_exp, &mut num_evals, &mut fcn)?;
            if f_exp < f_ref {
                simplex[n].copy_from_slice(&x_exp);
                f_vals[n] = f_exp;
            } else {
                simplex[n].copy_from_slice(&x_ref);
                f_vals[n] = f_ref;
            }
        } else if f_ref < f_vals[n - 1] {
            simplex[n].copy_from_slice(&x_ref);
            f_vals[n] = f_ref;
        } else {
            // Outside or inside contraction toward the centroid
            let outside = f_ref < f_vals[n];
            for j in 0..n {
                x_con[j] = if outside {
                    centroid[j] + config.gamma * (x_ref[j] - centroid[j])
                } else {
                    centroid[j] + config.gamma * (simplex[n][j] - centroid[j])
                };
            }
            let f_con = eval_point(&x_con, &mut num_evals, &mut fcn)?;
            let accept = if outside {
                f_con <= f_ref
            } else {
                f_con < f_vals[n]
            };
            if accept {
                simplex[n].copy_from_slice(&x_con);
                f_vals[n] = f_con;
            } else {
                // Shrink all non-best vertices toward simplex[0]
                num_shrinks += 1;
                let best = simplex[0].clone();
                for k in 1..n_verts {
                    for j in 0..n {
                        simplex[k][j] =
                            best[j] + config.delta * (simplex[k][j] - best[j]);
                    }
                    f_vals[k] =
                        eval_point(&simplex[k], &mut num_evals, &mut fcn)?;
                }
            }
        }
    }

    // Scan final simplex vertices to return the lowest-objective vertex
    // and package diagnostic evaluation/iteration/shrink counters
    let mut best_idx = 0;
    for k in 1..n_verts {
        if f_vals[k] < f_vals[best_idx] {
            best_idx = k;
        }
    }
    // Package minimum point, objective value, and diagnostic counters
    Ok(SimplexResult {
        x_min: simplex[best_idx].clone(),
        f_min: f_vals[best_idx],
        num_evals,
        num_iters,
        num_shrinks,
    })
}
