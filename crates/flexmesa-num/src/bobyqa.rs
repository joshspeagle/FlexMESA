// SPDX-License-Identifier: GPL-3.0-or-later
//! Powell (`2009`) bound-constrained derivative-free quadratic interpolation
//! trust-region optimizer (`BOBYQA` in `num/private/mod_bobyqa.f`).
//!
//! Minimizes $f(x)$ subject to box bounds $x_l \le x \le x_u$ using
//! $m = \mathrm{npt} \in [n + 2, (n + 1)(n + 2) / 2]$ interpolation points and
//! Frobenius-norm-minimal Hessian updates.

use crate::newuoa::{
    farthest_point_beyond, fit_min_frobenius_quad, quad_step_change, reduce_rho,
};
use crate::NumError;

/// Result returned by [`bobyqa`].
#[derive(Debug, Clone)]
pub struct BobyqaResult {
    /// Best parameter vector $x_{\min} \in [x_l, x_u]$ of length $n$.
    pub x_min: Vec<f64>,
    /// Minimum objective value $f(x_{\min})$.
    pub f_min: f64,
    /// Total number of function evaluations performed.
    pub num_evals: usize,
}

/// Minimizes a bound-constrained $n$-dimensional objective `calfun` on
/// $[x_l, x_u]$ via Powell's `BOBYQA` quadratic-interpolation trust-region
/// algorithm (`num/private/mod_bobyqa.f`, `num/public/num_bobyqa_proc.dek`).
///
/// Updates `x` in-place to the best feasible point found and returns
/// [`BobyqaResult`].
///
/// # Arguments
///
/// - `n`: Problem dimension ($n \ge 2$).
/// - `npt`: Number of interpolation conditions ($n + 2 \le \mathrm{npt} \le
///   (n + 1)(n + 2) / 2$; typically $2 n + 1$).
/// - `x`: Initial guess of length $n$, updated in-place to $x_{\min}$.
/// - `xl`, `xu`: Lower and upper bound slices of length $n$, satisfying
///   $x_{u,i} - x_{l,i} \ge 2\,\mathrm{rhobeg}$.
/// - `rhobeg`: Initial trust-region radius ($> 0$).
/// - `rhoend`: Final trust-region radius ($> 0$; upstream `do_BOBYQA` in
///   `num/private/mod_bobyqa.f:58-137, 844` permits `rhoend > rhobeg`, stopping
///   once the initial `rho = rhobeg` cycle finishes).
/// - `maxfun`: Maximum number of objective evaluations allowed.
/// - `max_valid_value`: Upper bound on acceptable objective values (`1e99`).
/// - `calfun`: Objective callable `calfun(x) -> Result<f64, NumError>`.
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] if dimensions, `npt`, `rhobeg <= 0`,
/// `rhoend <= 0`, or bound widths $x_{u,i} - x_{l,i} < 2\,\mathrm{rhobeg}$ are
/// invalid.
pub fn bobyqa<F>(
    n: usize,
    npt: usize,
    x: &mut [f64],
    xl: &[f64],
    xu: &[f64],
    rhobeg: f64,
    rhoend: f64,
    maxfun: usize,
    max_valid_value: f64,
    mut calfun: F,
) -> Result<BobyqaResult, NumError>
where
    F: FnMut(&[f64]) -> Result<f64, NumError>,
{
    let max_npt = (n + 1) * (n + 2) / 2;
    if n < 2
        || x.len() < n
        || xl.len() < n
        || xu.len() < n
        || npt < n + 2
        || npt > max_npt
        || rhobeg <= 0.0
        || rhoend <= 0.0
    {
        return Err(NumError::InvalidParameter(
            "require n >= 2, n+2 <= npt <= (n+1)(n+2)/2, rhobeg > 0, and rhoend > 0 in bobyqa",
        ));
    }
    for j in 0..n {
        if xu[j] - xl[j] < 2.0 * rhobeg {
            return Err(NumError::InvalidParameter(
                "require xu[j] - xl[j] >= 2 * rhobeg in bobyqa",
            ));
        }
        // Adjust initial x[j] inside bounds matching
        // `num/private/mod_bobyqa.f:95-128`.
        if x[j] <= xl[j] {
            x[j] = xl[j];
        } else if x[j] - xl[j] < rhobeg {
            x[j] = xl[j] + rhobeg;
        } else if x[j] >= xu[j] {
            x[j] = xu[j];
        } else if xu[j] - x[j] < rhobeg {
            x[j] = xu[j] - rhobeg;
        }
    }

    let mut nf = 0usize;
    let mut eval_f = |pt: &[f64], count: &mut usize| -> Result<f64, NumError> {
        *count += 1;
        let clamped: Vec<f64> =
            (0..n).map(|j| pt[j].clamp(xl[j], xu[j])).collect();
        let val = calfun(&clamped)?;
        if !val.is_finite() {
            return Ok(max_valid_value.min(1e99));
        }
        Ok(val.min(max_valid_value))
    };

    // Build initial box-feasible interpolation set around xbase (PRELIM)
    let xbase = x[..n].to_vec();
    let sl: Vec<f64> = (0..n).map(|j| xl[j] - xbase[j]).collect();
    let su: Vec<f64> = (0..n).map(|j| xu[j] - xbase[j]).collect();
    let mut xpt = vec![vec![0.0_f64; n]; npt];
    let mut fval = vec![0.0_f64; npt];
    let mut kopt = 0usize;

    for k in 0..npt {
        if k >= 1 && k <= n {
            let j = k - 1;
            xpt[k][j] = if su[j] == 0.0 {
                -rhobeg
            } else {
                rhobeg.min(su[j])
            };
        } else if k > n && k <= 2 * n {
            let j = k - n - 1;
            xpt[k][j] = if sl[j] == 0.0 {
                (2.0 * rhobeg).min(su[j])
            } else if su[j] == 0.0 {
                (-2.0 * rhobeg).max(sl[j])
            } else {
                (-rhobeg).max(sl[j])
            };
        } else if k > 2 * n {
            let nfmm = k - n;
            let itemp = (nfmm - 1) / n;
            let mut jpt = k - itemp * n - n;
            let mut ipt = jpt + itemp;
            if ipt > n {
                std::mem::swap(&mut ipt, &mut jpt);
                jpt -= n;
            }
            xpt[k][ipt - 1] = xpt[ipt][ipt - 1];
            xpt[k][jpt - 1] = xpt[jpt][jpt - 1];
        }
        let pt: Vec<f64> = (0..n)
            .map(|j| (xbase[j] + xpt[k][j]).clamp(xl[j], xu[j]))
            .collect();
        fval[k] = eval_f(&pt, &mut nf)?;
        if k == 0 || fval[k] < fval[kopt] {
            kopt = k;
        }
        if nf >= maxfun {
            for j in 0..n {
                x[j] = (xbase[j] + xpt[kopt][j]).clamp(xl[j], xu[j]);
            }
            return Ok(BobyqaResult {
                x_min: x[..n].to_vec(),
                f_min: fval[kopt],
                num_evals: nf,
            });
        }
    }

    let mut h_mat = vec![0.0_f64; n * n];
    let mut rho = rhobeg;
    let mut delta = rho;
    let mut diffa = 0.0_f64;
    let mut diffb = 0.0_f64;
    let mut diffc = 0.0_f64;
    let mut nfsav = nf;

    while nf < maxfun {
        let (grad, h_new) =
            fit_min_frobenius_quad(n, npt, kopt, delta, &xpt, &fval, &h_mat);
        h_mat = h_new;

        let d_lo: Vec<f64> = (0..n).map(|j| sl[j] - xpt[kopt][j]).collect();
        let d_hi: Vec<f64> = (0..n).map(|j| su[j] - xpt[kopt][j]).collect();
        let (d, crvmin) = trsbox_bounded(n, &grad, &h_mat, &d_lo, &d_hi, delta);
        let dnorm = delta.min(d.iter().map(|&v| v * v).sum::<f64>().sqrt());

        // Short trust-region step branch (`num/private/mod_bobyqa.f:266-296, 807-887`).
        if dnorm < 0.5 * rho {
            let mut need_alt = nf <= nfsav + 2;
            if !need_alt {
                let errbig = diffa.max(diffb).max(diffc);
                let frhosq = 0.125 * rho * rho;
                if crvmin > 0.0 && errbig > frhosq * crvmin {
                    need_alt = true;
                } else {
                    let bdtol = errbig / rho;
                    for j in 0..n {
                        let xnew_j = (xpt[kopt][j] + d[j]).clamp(sl[j], su[j]);
                        let mut g_j = grad[j];
                        for i in 0..n {
                            g_j += h_mat[j * n + i] * d[i];
                        }
                        let mut bdtest = bdtol;
                        if (xnew_j - sl[j]).abs() <= 1e-14 {
                            bdtest = g_j;
                        }
                        if (xnew_j - su[j]).abs() <= 1e-14 {
                            bdtest = -g_j;
                        }
                        if bdtest < bdtol {
                            bdtest += 0.5 * h_mat[j * n + j] * rho;
                            if bdtest < bdtol {
                                need_alt = true;
                                break;
                            }
                        }
                    }
                }
            }
            if need_alt {
                if let Some((kfar, max_dsq)) = farthest_point_beyond(
                    n,
                    npt,
                    kopt,
                    flexmesa_math::pow2(10.0 * rho),
                    &xpt,
                ) {
                    if nf < maxfun {
                        let dist = max_dsq.sqrt();
                        delta = (0.1 * delta).min(0.5 * dist);
                        if delta <= 1.5 * rho {
                            delta = rho;
                        }
                        let adelt = (0.1 * dist).min(delta).max(rho);
                        let norm = dist.max(1e-15);
                        let xnew: Vec<f64> = (0..n)
                            .map(|j| {
                                let step = (adelt / norm)
                                    * (xpt[kfar][j] - xpt[kopt][j]);
                                (xpt[kopt][j] + step).clamp(sl[j], su[j])
                            })
                            .collect();
                        let pt: Vec<f64> = (0..n)
                            .map(|j| (xbase[j] + xnew[j]).clamp(xl[j], xu[j]))
                            .collect();
                        let fnew = eval_f(&pt, &mut nf)?;
                        let step_taken: Vec<f64> =
                            (0..n).map(|i| xnew[i] - xpt[kopt][i]).collect();
                        let vquad =
                            quad_step_change(n, &grad, &h_mat, &step_taken);
                        diffc = diffb;
                        diffb = diffa;
                        diffa = (fnew - fval[kopt] - vquad).abs();
                        if adelt > rho {
                            nfsav = nf;
                        }
                        xpt[kfar] = xnew;
                        fval[kfar] = fnew;
                        if fnew < fval[kopt] {
                            kopt = kfar;
                        }
                        continue;
                    }
                    break;
                }
            }
            if rho > rhoend {
                (rho, delta) = reduce_rho(rho, rhoend);
                nfsav = nf;
                continue;
            }
            if nf < maxfun {
                let xnew: Vec<f64> = (0..n)
                    .map(|j| (xpt[kopt][j] + d[j]).clamp(sl[j], su[j]))
                    .collect();
                let pt: Vec<f64> = (0..n)
                    .map(|j| (xbase[j] + xnew[j]).clamp(xl[j], xu[j]))
                    .collect();
                let fnew = eval_f(&pt, &mut nf)?;
                if fnew < fval[kopt] {
                    x[..n].copy_from_slice(&pt);
                    return Ok(BobyqaResult {
                        x_min: pt,
                        f_min: fnew,
                        num_evals: nf,
                    });
                }
            }
            break;
        }

        let xnew: Vec<f64> = (0..n)
            .map(|j| (xpt[kopt][j] + d[j]).clamp(sl[j], su[j]))
            .collect();
        let pt: Vec<f64> = (0..n)
            .map(|j| (xbase[j] + xnew[j]).clamp(xl[j], xu[j]))
            .collect();
        let fnew = eval_f(&pt, &mut nf)?;
        let fopt = fval[kopt];

        let step_taken: Vec<f64> =
            (0..n).map(|i| xnew[i] - xpt[kopt][i]).collect();
        let vquad = quad_step_change(n, &grad, &h_mat, &step_taken);
        diffc = diffb;
        diffb = diffa;
        diffa = (fnew - fopt - vquad).abs();
        if dnorm > rho {
            nfsav = nf;
        }
        if vquad >= 0.0 {
            if fnew < fopt {
                x[..n].copy_from_slice(&pt);
                return Ok(BobyqaResult {
                    x_min: pt,
                    f_min: fnew,
                    num_evals: nf,
                });
            }
            break;
        }
        let ratio = (fnew - fopt) / vquad;

        delta = if ratio <= 0.1 {
            (0.5 * delta).min(dnorm)
        } else if ratio <= 0.7 {
            (0.5 * delta).max(dnorm)
        } else {
            (0.5 * delta).max(2.0 * dnorm)
        };
        if delta <= 1.5 * rho {
            delta = rho;
        }

        let k_ref = if fnew < fopt { usize::MAX } else { kopt };
        let mut knew = 0usize;
        let mut best_score = -1.0_f64;
        for k in 0..npt {
            if k == kopt {
                continue;
            }
            let dsq: f64 = if k_ref == usize::MAX {
                (0..n)
                    .map(|j| flexmesa_math::pow2(xpt[k][j] - xnew[j]))
                    .sum()
            } else {
                (0..n)
                    .map(|j| flexmesa_math::pow2(xpt[k][j] - xpt[kopt][j]))
                    .sum()
            };
            let score = flexmesa_math::pow2(dsq).max(1e-30);
            if score > best_score {
                best_score = score;
                knew = k;
            }
        }
        let dsq_knew: f64 = (0..n)
            .map(|j| flexmesa_math::pow2(xpt[knew][j] - xpt[kopt][j]))
            .sum();
        if fnew < fopt
            || dsq_knew > flexmesa_math::pow2(2.0 * delta)
            || dnorm * dnorm < dsq_knew
        {
            xpt[knew] = xnew;
            fval[knew] = fnew;
            if fnew < fopt {
                kopt = knew;
            }
        }

        // Post-step checks (`num/private/mod_bobyqa.f:802-868`).
        if fnew <= fopt + 0.1 * vquad {
            continue;
        }
        let dist_thresh = flexmesa_math::pow2(2.0 * delta)
            .max(flexmesa_math::pow2(10.0 * rho));
        if let Some((kfar, max_dsq)) =
            farthest_point_beyond(n, npt, kopt, dist_thresh, &xpt)
        {
            if nf < maxfun {
                let dist = max_dsq.sqrt();
                let adelt = (0.1 * dist).min(delta).max(rho);
                let norm = dist.max(1e-15);
                let xnew_m: Vec<f64> = (0..n)
                    .map(|j| {
                        let step =
                            (adelt / norm) * (xpt[kfar][j] - xpt[kopt][j]);
                        (xpt[kopt][j] + step).clamp(sl[j], su[j])
                    })
                    .collect();
                let pt_m: Vec<f64> = (0..n)
                    .map(|j| (xbase[j] + xnew_m[j]).clamp(xl[j], xu[j]))
                    .collect();
                let fnew_m = eval_f(&pt_m, &mut nf)?;
                let step_m: Vec<f64> =
                    (0..n).map(|i| xnew_m[i] - xpt[kopt][i]).collect();
                let vquad_m = quad_step_change(n, &grad, &h_mat, &step_m);
                diffc = diffb;
                diffb = diffa;
                diffa = (fnew_m - fval[kopt] - vquad_m).abs();
                if adelt > rho {
                    nfsav = nf;
                }
                xpt[kfar] = xnew_m;
                fval[kfar] = fnew_m;
                if fnew_m < fval[kopt] {
                    kopt = kfar;
                }
                continue;
            }
            break;
        }
        if ratio > 0.0 || delta.max(dnorm) > rho {
            continue;
        }
        if rho > rhoend {
            (rho, delta) = reduce_rho(rho, rhoend);
            nfsav = nf;
            continue;
        }
        break;
    }

    for j in 0..n {
        x[j] = (xbase[j] + xpt[kopt][j]).clamp(xl[j], xu[j]);
    }
    Ok(BobyqaResult {
        x_min: x[..n].to_vec(),
        f_min: fval[kopt],
        num_evals: nf,
    })
}

/// Solves $\min_{\|d\|_2 \le \Delta, \, l \le d \le u} g^T d + \frac{1}{2} d^T H d$
/// via projected Steihaug-Toint conjugate gradient (`TRSBOX` in
/// `num/private/mod_bobyqa.f:1790-2163`), returning `(step, crvmin)`.
fn trsbox_bounded(
    n: usize,
    grad: &[f64],
    h_mat: &[f64],
    d_lo: &[f64],
    d_hi: &[f64],
    delta: f64,
) -> (Vec<f64>, f64) {
    let mut d = vec![0.0_f64; n];
    let mut active = vec![false; n];
    // `CRVMIN = -1.0D0` initially (`num/private/mod_bobyqa.f:1804, 1846`),
    // returning `-1.0` when every search is restricted by bounds.
    let mut crvmin = -1.0_f64;

    for _restart in 0..n {
        let mut r = grad.to_vec();
        for i in 0..n {
            for j in 0..n {
                r[i] += h_mat[i * n + j] * d[j];
            }
            if (d[i] <= d_lo[i] + 1e-14 && r[i] >= 0.0)
                || (d[i] >= d_hi[i] - 1e-14 && r[i] <= 0.0)
            {
                active[i] = true;
            }
        }
        let mut p: Vec<f64> = (0..n)
            .map(|i| if active[i] { 0.0 } else { -r[i] })
            .collect();
        let mut rr: f64 = p.iter().map(|&v| v * v).sum();
        if rr <= 1e-30 {
            break;
        }

        let mut hit_box = false;
        for _iter in 0..(2 * n) {
            let mut hp = vec![0.0_f64; n];
            for i in 0..n {
                if !active[i] {
                    let mut s = 0.0_f64;
                    for j in 0..n {
                        s += h_mat[i * n + j] * p[j];
                    }
                    hp[i] = s;
                }
            }
            let p_hp: f64 = (0..n).map(|i| p[i] * hp[i]).sum();
            let dd: f64 = d.iter().map(|&v| v * v).sum();
            let dp: f64 = (0..n).map(|i| d[i] * p[i]).sum();
            let pp: f64 = p.iter().map(|&v| v * v).sum();
            if pp <= 1e-30 {
                crvmin = 0.0;
                break;
            }
            let disc = (dp * dp + pp * (delta * delta - dd)).max(0.0);
            let alpha_tr = (-dp + disc.sqrt()) / pp;

            let mut alpha_box = f64::INFINITY;
            let mut box_idx = usize::MAX;
            for i in 0..n {
                if active[i] {
                    continue;
                }
                if p[i] > 1e-15 {
                    let a = (d_hi[i] - d[i]) / p[i];
                    if a < alpha_box {
                        alpha_box = a;
                        box_idx = i;
                    }
                } else if p[i] < -1e-15 {
                    let a = (d_lo[i] - d[i]) / p[i];
                    if a < alpha_box {
                        alpha_box = a;
                        box_idx = i;
                    }
                }
            }
            let max_step = alpha_tr.min(alpha_box);
            let alpha_cg = if p_hp > 1e-30 {
                rr / p_hp
            } else {
                f64::INFINITY
            };

            if alpha_cg >= max_step {
                for i in 0..n {
                    d[i] = (d[i] + max_step * p[i]).clamp(d_lo[i], d_hi[i]);
                }
                if alpha_box <= alpha_tr && box_idx < n {
                    active[box_idx] = true;
                    hit_box = true;
                }
                // `STPLEN == BLEN` -> `90 CRVMIN = ZERO`
                // (`num/private/mod_bobyqa.f:1956-1962`).
                if max_step >= alpha_tr - 1e-15 {
                    crvmin = 0.0;
                }
                break;
            }

            let rayleigh = p_hp / pp;
            // Track minimum unconstrained positive curvature across box
            // restarts (`num/private/mod_bobyqa.f:1927-1930`).
            if rayleigh > 0.0 {
                crvmin = crvmin.min(rayleigh);
                if crvmin == -1.0 {
                    crvmin = rayleigh;
                }
            }
            for i in 0..n {
                d[i] += alpha_cg * p[i];
                r[i] += alpha_cg * hp[i];
            }
            let rr_new: f64 = (0..n)
                .map(|i| if active[i] { 0.0 } else { r[i] * r[i] })
                .sum();
            if rr_new <= 1e-24 {
                break;
            }
            let beta = rr_new / rr;
            for i in 0..n {
                p[i] = if active[i] { 0.0 } else { -r[i] + beta * p[i] };
            }
            rr = rr_new;
        }
        if !hit_box {
            break;
        }
    }
    // `CRVMIN = 0` whenever `d` reaches the trust-region boundary
    // (`num/private/mod_bobyqa.f:1801, 1962`).
    if d.iter().map(|&v| v * v).sum::<f64>() >= delta * delta * (1.0 - 1e-12) {
        crvmin = 0.0;
    }
    (d, crvmin)
}
