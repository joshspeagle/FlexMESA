//! Powell (`2009`) bound-constrained derivative-free quadratic interpolation
//! trust-region optimizer (`BOBYQA` in `num/private/mod_bobyqa.f`).
//!
//! Minimizes $f(x)$ subject to box bounds $x_l \le x \le x_u$ using
//! $m = \mathrm{npt} \in [n + 2, (n + 1)(n + 2) / 2]$ interpolation points and
//! Frobenius-norm-minimal Hessian updates.

use crate::newuoa::fit_min_frobenius_quad;
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
/// algorithm (`num_bobyqa.dek`).
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
/// - `rhoend`: Final trust-region radius ($0 < \mathrm{rhoend} \le
///   \mathrm{rhobeg}$).
/// - `maxfun`: Maximum number of objective evaluations allowed.
/// - `max_valid_value`: Upper bound on acceptable objective values (`1e99`).
/// - `calfun`: Objective callable `calfun(x) -> Result<f64, NumError>`.
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] if dimensions, `npt`, or bound
/// widths $x_{u,i} - x_{l,i} < 2\,\mathrm{rhobeg}$ are invalid.
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
        || rhoend > rhobeg
    {
        return Err(NumError::InvalidParameter(
            "require n >= 2, n+2 <= npt <= (n+1)(n+2)/2, and 0 < rhoend <= rhobeg in bobyqa",
        ));
    }
    for j in 0..n {
        if xu[j] - xl[j] < 2.0 * rhobeg {
            return Err(NumError::InvalidParameter(
                "require xu[j] - xl[j] >= 2 * rhobeg in bobyqa",
            ));
        }
        // Adjust initial x[j] inside bounds matching mod_bobyqa.f:69-82
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
    let mut delta = rhobeg;

    while nf < maxfun {
        let (grad, h_new) =
            fit_min_frobenius_quad(n, npt, kopt, delta, &xpt, &fval, &h_mat);
        h_mat = h_new;

        let d_lo: Vec<f64> = (0..n).map(|j| sl[j] - xpt[kopt][j]).collect();
        let d_hi: Vec<f64> = (0..n).map(|j| su[j] - xpt[kopt][j]).collect();
        let d = trsbox_bounded(n, &grad, &h_mat, &d_lo, &d_hi, delta);
        let dnorm = d.iter().map(|&v| v * v).sum::<f64>().sqrt();

        if dnorm < 0.5 * rho {
            let mut kfar = kopt;
            let mut max_dist_sq = 0.0_f64;
            let dist_thresh = (2.0 * delta).powi(2).max((10.0 * rho).powi(2));
            for k in 0..npt {
                if k == kopt {
                    continue;
                }
                let dsq: f64 =
                    (0..n).map(|j| (xpt[k][j] - xpt[kopt][j]).powi(2)).sum();
                if dsq > max_dist_sq {
                    max_dist_sq = dsq;
                    kfar = k;
                }
            }
            if max_dist_sq > dist_thresh && nf < maxfun {
                let mut xnew = vec![0.0_f64; n];
                let norm = max_dist_sq.sqrt().max(1e-15);
                for j in 0..n {
                    let step = (rho / norm) * (xpt[kfar][j] - xpt[kopt][j]);
                    xnew[j] = (xpt[kopt][j] + step).clamp(sl[j], su[j]);
                }
                let pt: Vec<f64> = (0..n)
                    .map(|j| (xbase[j] + xnew[j]).clamp(xl[j], xu[j]))
                    .collect();
                let fnew = eval_f(&pt, &mut nf)?;
                xpt[kfar] = xnew;
                fval[kfar] = fnew;
                if fnew < fval[kopt] {
                    kopt = kfar;
                }
                continue;
            }
            if rho <= rhoend {
                break;
            }
            delta = 0.5 * rho;
            let ratio = rho / rhoend;
            rho = if ratio <= 16.0 {
                rhoend
            } else if ratio <= 250.0 {
                ratio.sqrt() * rhoend
            } else {
                0.1 * rho
            };
            delta = delta.max(rho);
            continue;
        }

        let xnew: Vec<f64> = (0..n)
            .map(|j| (xpt[kopt][j] + d[j]).clamp(sl[j], su[j]))
            .collect();
        let pt: Vec<f64> = (0..n)
            .map(|j| (xbase[j] + xnew[j]).clamp(xl[j], xu[j]))
            .collect();
        let fnew = eval_f(&pt, &mut nf)?;
        let fopt = fval[kopt];

        let mut g_dot_d = 0.0_f64;
        let mut d_h_d = 0.0_f64;
        for i in 0..n {
            let di = xnew[i] - xpt[kopt][i];
            g_dot_d += grad[i] * di;
            for j in 0..n {
                let dj = xnew[j] - xpt[kopt][j];
                d_h_d += di * h_mat[i * n + j] * dj;
            }
        }
        let pred = -(g_dot_d + 0.5 * d_h_d);
        let ratio = if pred > 1e-30 {
            (fopt - fnew) / pred
        } else if fnew < fopt {
            1.0
        } else {
            0.0
        };

        delta = if ratio <= 0.1 {
            0.5 * dnorm
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
                (0..n).map(|j| (xpt[k][j] - xnew[j]).powi(2)).sum()
            } else {
                (0..n).map(|j| (xpt[k][j] - xpt[kopt][j]).powi(2)).sum()
            };
            let score = dsq.powi(2).max(1e-30);
            if score > best_score {
                best_score = score;
                knew = k;
            }
        }
        let dsq_knew: f64 =
            (0..n).map(|j| (xpt[knew][j] - xpt[kopt][j]).powi(2)).sum();
        if fnew < fopt
            || dsq_knew > (2.0 * delta).powi(2)
            || dnorm * dnorm < dsq_knew
        {
            xpt[knew] = xnew;
            fval[knew] = fnew;
            if fnew < fopt {
                kopt = knew;
            }
        }
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
/// via projected Steihaug-Toint conjugate gradient (`TRSBOX` in `mod_bobyqa.f`).
fn trsbox_bounded(
    n: usize,
    grad: &[f64],
    h_mat: &[f64],
    d_lo: &[f64],
    d_hi: &[f64],
    delta: f64,
) -> Vec<f64> {
    let mut d = vec![0.0_f64; n];
    let mut active = vec![false; n];

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
                break;
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
    d
}
