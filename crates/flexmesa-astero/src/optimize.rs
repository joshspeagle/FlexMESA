// SPDX-License-Identifier: LGPL-3.0-or-later
//! Derivative-free asteroseismic optimization drivers: Nelder-Mead simplex,
//! Powell trust-region coordinate/quadratic search, and Cartesian grid scans.
//!
//! Corresponds to `astero/private/astero_run_support.f90:366-982` and
//! `num/private/mod_simplex.f90:30-384`.

/// Minimizes a scalar objective `eval_fn(x)` over bounded parameter space
/// `x_lower <= x <= x_upper` using the upstream MESA Nelder-Mead simplex
/// algorithm (`mod_simplex.f90:30-384`).
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
    mut eval_fn: F,
) -> (Vec<f64>, f64, usize)
where
    F: FnMut(&[f64]) -> f64,
{
    let n = x_first.len().min(x_lower.len()).min(x_upper.len());
    if n == 0 {
        return (Vec::new(), 0.0, 0);
    }
    let clamp_vec = |v: &[f64]| -> Vec<f64> {
        (0..n).map(|i| v[i].clamp(x_lower[i], x_upper[i])).collect()
    };

    let mut vertices: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    let mut f_vals: Vec<f64> = Vec::with_capacity(n + 1);
    let mut evals = 0usize;

    let v0 = clamp_vec(x_first);
    let f0 = eval_fn(&v0);
    evals += 1;
    vertices.push(v0.clone());
    f_vals.push(f0);

    for j in 0..n {
        let mut vj = v0.clone();
        let span = x_upper[j] - x_lower[j];
        let step = if (v0[j] - x_lower[j]) < (x_upper[j] - v0[j]) {
            0.25 * span
        } else {
            -0.25 * span
        };
        vj[j] = (vj[j] + step).clamp(x_lower[j], x_upper[j]);
        let fj = eval_fn(&vj);
        evals += 1;
        vertices.push(vj);
        f_vals.push(fj);
    }

    let alpha = 1.0_f64;
    let beta = 2.0_f64;
    let gamma = 0.5_f64;
    let delta = 0.5_f64;
    let mut order: Vec<usize> = (0..=n).collect();
    let mut centroid = vec![0.0_f64; n];

    while evals < max_evals {
        order.sort_by(|&a, &b| {
            f_vals[a]
                .partial_cmp(&f_vals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let l_idx = order[0];
        let s_idx = order[n - 1];
        let h_idx = order[n];

        if f_vals[l_idx] <= chi2_tol {
            break;
        }
        let mut max_rel = 0.0_f64;
        for &j in &order[1..] {
            for i in 0..n {
                let diff = (vertices[j][i] - vertices[l_idx][i]).abs();
                let scale = x_atol
                    + x_rtol
                        * vertices[j][i].abs().max(vertices[l_idx][i].abs());
                max_rel = max_rel.max(diff / scale.max(1.0e-30));
            }
        }
        if max_rel <= 1.0 {
            break;
        }

        centroid.fill(0.0);
        let mut w_sum = 0.0_f64;
        for &j in &order[..n] {
            let w = if centroid_weight_power == 0.0 {
                1.0
            } else {
                f_vals[j].max(1.0e-30).powf(-centroid_weight_power)
            };
            w_sum += w;
            for i in 0..n {
                centroid[i] += w * vertices[j][i];
            }
        }
        for c_i in &mut centroid {
            *c_i /= w_sum.max(1.0e-30);
        }

        let xr: Vec<f64> = (0..n)
            .map(|i| {
                (centroid[i] + alpha * (centroid[i] - vertices[h_idx][i]))
                    .clamp(x_lower[i], x_upper[i])
            })
            .collect();
        let fr = eval_fn(&xr);
        evals += 1;

        if fr < f_vals[l_idx] {
            let xe: Vec<f64> = (0..n)
                .map(|i| {
                    (centroid[i] + beta * (xr[i] - centroid[i]))
                        .clamp(x_lower[i], x_upper[i])
                })
                .collect();
            let fe = eval_fn(&xe);
            evals += 1;
            if fe < f_vals[l_idx] {
                vertices[h_idx] = xe;
                f_vals[h_idx] = fe;
            } else {
                vertices[h_idx] = xr;
                f_vals[h_idx] = fr;
            }
        } else if fr < f_vals[s_idx] {
            vertices[h_idx] = xr;
            f_vals[h_idx] = fr;
        } else {
            let xc: Vec<f64> = if fr < f_vals[h_idx] {
                (0..n)
                    .map(|i| {
                        (centroid[i] + gamma * (xr[i] - centroid[i]))
                            .clamp(x_lower[i], x_upper[i])
                    })
                    .collect()
            } else {
                (0..n)
                    .map(|i| {
                        (centroid[i]
                            + gamma * (vertices[h_idx][i] - centroid[i]))
                            .clamp(x_lower[i], x_upper[i])
                    })
                    .collect()
            };
            let fc = eval_fn(&xc);
            evals += 1;
            if fc < f_vals[h_idx].min(fr) {
                vertices[h_idx] = xc;
                f_vals[h_idx] = fc;
            } else {
                let best_v = vertices[l_idx].clone();
                for &j in &order[1..] {
                    for i in 0..n {
                        vertices[j][i] =
                            best_v[i] + delta * (vertices[j][i] - best_v[i]);
                    }
                    f_vals[j] = eval_fn(&vertices[j]);
                    evals += 1;
                    if evals >= max_evals {
                        break;
                    }
                }
            }
        }
    }

    let mut best_idx = 0usize;
    for j in 1..vertices.len() {
        if f_vals[j] < f_vals[best_idx] {
            best_idx = j;
        }
    }
    (vertices[best_idx].clone(), f_vals[best_idx], evals)
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

/// Minimizes `eval_fn(x)` using a bounded quadratic trust-region / 3-point
/// parabolic coordinate interpolation search in normalized `bobyqa_param`
/// coordinates (`do_bobyqa_or_newuoa`, `astero_run_support.f90:534-707`).
///
/// Returns `(x_best, f_best, num_evaluations)`.
pub fn powell_quadratic_minimize<F>(
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
    let n = x_first.len().min(x_lower.len()).min(x_upper.len());
    if n == 0 {
        return (Vec::new(), 0.0, 0);
    }
    let fill_phys = |u: &[f64], dst: &mut [f64]| {
        for i in 0..n {
            dst[i] = map_bobyqa_param(u[i], x_first[i], x_lower[i], x_upper[i]);
        }
    };

    let mut u_best = vec![0.0_f64; n];
    let mut trial = vec![0.0_f64; n];
    let mut phys_buf = vec![0.0_f64; n];
    fill_phys(&u_best, &mut phys_buf);
    let mut f_best = eval_fn(&phys_buf);
    let mut evals = 1usize;
    let mut rho = rhobeg.max(rhoend);

    while rho >= rhoend && evals < max_evals {
        let mut improved = false;
        for i in 0..n {
            if evals + 2 > max_evals {
                break;
            }
            let u_c = u_best[i];
            let u_p = (u_c + rho).min(1.0);
            let u_m = (u_c - rho).max(-1.0);

            trial.copy_from_slice(&u_best);
            trial[i] = u_p;
            fill_phys(&trial, &mut phys_buf);
            let f_p = eval_fn(&phys_buf);
            evals += 1;

            trial[i] = u_m;
            fill_phys(&trial, &mut phys_buf);
            let f_m = eval_fn(&phys_buf);
            evals += 1;

            let denom =
                (u_p - u_c) * (f_m - f_best) + (u_c - u_m) * (f_p - f_best);
            if denom > 1.0e-30 && evals < max_evals {
                let numer = (u_p - u_c).powi(2) * (f_m - f_best)
                    - (u_c - u_m).powi(2) * (f_p - f_best);
                let u_q = (u_c + 0.5 * numer / denom).clamp(u_m, u_p);
                trial[i] = u_q;
                fill_phys(&trial, &mut phys_buf);
                let f_q = eval_fn(&phys_buf);
                evals += 1;
                if f_q < f_best {
                    f_best = f_q;
                    u_best[i] = u_q;
                    improved = true;
                }
            }
            if f_p < f_best {
                f_best = f_p;
                u_best[i] = u_p;
                improved = true;
            }
            if f_m < f_best {
                f_best = f_m;
                u_best[i] = u_m;
                improved = true;
            }
        }
        if !improved {
            rho *= 0.5;
        }
    }
    fill_phys(&u_best, &mut phys_buf);
    (phys_buf, f_best, evals)
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
