//! Powell (`2006`) unconstrained derivative-free quadratic interpolation
//! trust-region optimizer (`NEWUOA` in `num/private/mod_newuoa.f`).
//!
//! Minimizes an unconstrained objective $f: \mathbb{R}^n \to \mathbb{R}$ using
//! $m = \mathrm{npt} \in [n + 2, (n + 1)(n + 2) / 2]$ interpolation points by
//! constructing quadratic models whose Hessian updates minimize the Frobenius
//! norm $\|\Delta H\|_F$ subject to interpolation conditions.

use crate::NumError;
use flexmesa_mtx::{dgetrf, dgetrs};

/// Result returned by [`newuoa`].
#[derive(Debug, Clone)]
pub struct NewuoaResult {
    /// Best parameter vector $x_{\min}$ of length $n$.
    pub x_min: Vec<f64>,
    /// Minimum objective value $f(x_{\min})$.
    pub f_min: f64,
    /// Total number of function evaluations performed.
    pub num_evals: usize,
}

/// Minimizes an unconstrained $n$-dimensional objective `calfun` via Powell's
/// `NEWUOA` quadratic-interpolation trust-region algorithm (`num_newuoa.dek`).
///
/// Updates `x` in-place to the best point found and returns [`NewuoaResult`].
///
/// # Arguments
///
/// - `n`: Problem dimension ($n \ge 2$).
/// - `npt`: Number of interpolation conditions ($n + 2 \le \mathrm{npt} \le
///   (n + 1)(n + 2) / 2$; typically $2 n + 1$).
/// - `x`: Initial guess of length $n$, updated in-place to $x_{\min}$.
/// - `rhobeg`: Initial trust-region radius ($> 0$).
/// - `rhoend`: Final trust-region radius ($0 < \mathrm{rhoend} \le
///   \mathrm{rhobeg}$).
/// - `maxfun`: Maximum number of objective evaluations allowed.
/// - `max_valid_value`: Upper bound on acceptable objective values (`1e99`).
/// - `calfun`: Objective callable `calfun(x) -> Result<f64, NumError>`.
///
/// # Errors
///
/// Returns [`NumError::InvalidParameter`] if `n < 2`, `npt` is outside
/// $[n + 2, (n + 1)(n + 2) / 2]$, or `rhobeg < rhoend <= 0`.
pub fn newuoa<F>(
    n: usize,
    npt: usize,
    x: &mut [f64],
    rhobeg: f64,
    rhoend: f64,
    maxfun: usize,
    max_valid_value: f64,
    mut calfun: F,
) -> Result<NewuoaResult, NumError>
where
    F: FnMut(&[f64]) -> Result<f64, NumError>,
{
    let max_npt = (n + 1) * (n + 2) / 2;
    if n < 2
        || x.len() < n
        || npt < n + 2
        || npt > max_npt
        || rhobeg <= 0.0
        || rhoend <= 0.0
        || rhoend > rhobeg
    {
        return Err(NumError::InvalidParameter(
            "require n >= 2, n+2 <= npt <= (n+1)(n+2)/2, and 0 < rhoend <= rhobeg in newuoa",
        ));
    }

    let mut nf = 0usize;
    let mut eval_f = |pt: &[f64], count: &mut usize| -> Result<f64, NumError> {
        *count += 1;
        let val = calfun(pt)?;
        if !val.is_finite() {
            return Ok(max_valid_value.min(1e99));
        }
        Ok(val.min(max_valid_value))
    };

    // Build initial interpolation set around x[0..n] (mod_newuoa.f:158-183)
    let xbase = x[..n].to_vec();
    let mut xpt = vec![vec![0.0_f64; n]; npt];
    let mut fval = vec![0.0_f64; npt];
    let mut kopt = 0usize;

    for k in 0..npt {
        if k >= 1 && k <= n {
            xpt[k][k - 1] = rhobeg;
        } else if k > n && k <= 2 * n {
            xpt[k][k - n - 1] = -rhobeg;
        } else if k > 2 * n {
            let nfmm = k - n;
            let itemp = (nfmm - 1) / n;
            let mut jpt = k - itemp * n - n;
            let mut ipt = jpt + itemp;
            if ipt > n {
                std::mem::swap(&mut ipt, &mut jpt);
                jpt -= n;
            }
            let xipt = if fval[ipt + n] < fval[ipt] {
                -rhobeg
            } else {
                rhobeg
            };
            let xjpt = if fval[jpt + n] < fval[jpt] {
                -rhobeg
            } else {
                rhobeg
            };
            xpt[k][ipt - 1] = xipt;
            xpt[k][jpt - 1] = xjpt;
        }
        let pt: Vec<f64> = (0..n).map(|j| xbase[j] + xpt[k][j]).collect();
        fval[k] = eval_f(&pt, &mut nf)?;
        if k == 0 || fval[k] < fval[kopt] {
            kopt = k;
        }
        if nf >= maxfun {
            x[..n].copy_from_slice(&(0..n).map(|j| xbase[j] + xpt[kopt][j]).collect::<Vec<_>>());
            return Ok(NewuoaResult {
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

        let d = trsbox_unconstrained(n, &grad, &h_mat, delta);
        let dnorm = d.iter().map(|&v| v * v).sum::<f64>().sqrt();

        if dnorm < 0.5 * rho {
            // Check if any interpolation point is far from xopt
            let mut kfar = kopt;
            let mut max_dist_sq = 0.0_f64;
            let dist_thresh = (2.0 * delta).powi(2).max((10.0 * rho).powi(2));
            for k in 0..npt {
                if k == kopt {
                    continue;
                }
                let dsq: f64 = (0..n)
                    .map(|j| (xpt[k][j] - xpt[kopt][j]).powi(2))
                    .sum();
                if dsq > max_dist_sq {
                    max_dist_sq = dsq;
                    kfar = k;
                }
            }
            if max_dist_sq > dist_thresh && nf < maxfun {
                let d_geo = geometry_step_unconstrained(n, npt, kopt, kfar, rho, &xpt);
                let xnew: Vec<f64> =
                    (0..n).map(|j| xpt[kopt][j] + d_geo[j]).collect();
                let pt: Vec<f64> = (0..n).map(|j| xbase[j] + xnew[j]).collect();
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

        let xnew: Vec<f64> = (0..n).map(|j| xpt[kopt][j] + d[j]).collect();
        let pt: Vec<f64> = (0..n).map(|j| xbase[j] + xnew[j]).collect();
        let fnew = eval_f(&pt, &mut nf)?;
        let fopt = fval[kopt];

        // Predicted reduction vquad = - (g^T d + 0.5 d^T H d)
        let mut g_dot_d = 0.0_f64;
        let mut d_h_d = 0.0_f64;
        for i in 0..n {
            g_dot_d += grad[i] * d[i];
            for j in 0..n {
                d_h_d += d[i] * h_mat[i * n + j] * d[j];
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

        // Select point to replace: farthest from the new best point
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
        if fnew < fopt || max_dist_sq_from(&xpt, kopt) > (2.0 * delta).powi(2) {
            xpt[knew] = xnew;
            fval[knew] = fnew;
            if fnew < fopt {
                kopt = knew;
            }
        } else {
            // Replace knew if the new point is closer than knew to xopt
            let dsq_knew: f64 = (0..n)
                .map(|j| (xpt[knew][j] - xpt[kopt][j]).powi(2))
                .sum();
            if dnorm * dnorm < dsq_knew {
                xpt[knew] = xnew;
                fval[knew] = fnew;
            }
        }
    }

    for j in 0..n {
        x[j] = xbase[j] + xpt[kopt][j];
    }
    Ok(NewuoaResult {
        x_min: x[..n].to_vec(),
        f_min: fval[kopt],
        num_evals: nf,
    })
}

fn max_dist_sq_from(xpt: &[Vec<f64>], kopt: usize) -> f64 {
    let n = xpt[0].len();
    xpt.iter()
        .enumerate()
        .filter(|&(k, _)| k != kopt)
        .map(|(_, p)| (0..n).map(|j| (p[j] - xpt[kopt][j]).powi(2)).sum::<f64>())
        .fold(0.0_f64, f64::max)
}

/// Fits the minimum-Frobenius-norm quadratic model around `xpt[kopt]`.
pub(crate) fn fit_min_frobenius_quad(
    n: usize,
    npt: usize,
    kopt: usize,
    scale: f64,
    xpt: &[Vec<f64>],
    fval: &[f64],
    h_prev: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let m = npt - 1;
    let dim = m + n;
    let inv_s = 1.0 / scale.max(1e-12);
    let mut s_rows: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut rhs = vec![0.0_f64; dim];

    for (k, pt) in xpt.iter().enumerate() {
        if k == kopt {
            continue;
        }
        let r = s_rows.len();
        let dx: Vec<f64> = (0..n).map(|j| pt[j] - xpt[kopt][j]).collect();
        let mut quad_prev = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                quad_prev += dx[i] * h_prev[i * n + j] * dx[j];
            }
        }
        rhs[r] = (fval[k] - fval[kopt]) - 0.5 * quad_prev;
        s_rows.push(dx.iter().map(|&v| v * inv_s).collect());
    }

    let mut kkt = vec![0.0_f64; dim * dim];
    for r in 0..m {
        for q in 0..=r {
            let dot: f64 = (0..n).map(|j| s_rows[r][j] * s_rows[q][j]).sum();
            let val = 0.5 * dot * dot + if r == q { 1e-11 } else { 0.0 };
            kkt[r * dim + q] = val;
            kkt[q * dim + r] = val;
        }
        for j in 0..n {
            kkt[r * dim + (m + j)] = s_rows[r][j];
            kkt[(m + j) * dim + r] = s_rows[r][j];
        }
    }
    for j in 0..n {
        kkt[(m + j) * dim + (m + j)] = -1e-12;
    }

    let mut ipiv = vec![0usize; dim];
    if dgetrf(dim, &mut kkt, &mut ipiv).is_ok()
        && dgetrs(dim, &kkt, &ipiv, &mut rhs).is_ok()
    {
        let grad: Vec<f64> = (0..n).map(|j| rhs[m + j] * inv_s).collect();
        let mut h_new = h_prev.to_vec();
        let inv_s2 = inv_s * inv_s;
        for r in 0..m {
            let lam = rhs[r] * inv_s2;
            for i in 0..n {
                for j in 0..n {
                    h_new[i * n + j] += lam * s_rows[r][i] * s_rows[r][j];
                }
            }
        }
        (grad, h_new)
    } else {
        (vec![0.0; n], h_prev.to_vec())
    }
}

/// Solves $\min_{\|d\|_2 \le \Delta} g^T d + \frac{1}{2} d^T H d$ via
/// Steihaug-Toint truncated conjugate gradient (`TRSBOX` unconstrained).
fn trsbox_unconstrained(
    n: usize,
    grad: &[f64],
    h_mat: &[f64],
    delta: f64,
) -> Vec<f64> {
    let mut d = vec![0.0_f64; n];
    let mut r = grad.to_vec();
    let mut p: Vec<f64> = r.iter().map(|&v| -v).collect();
    let mut rr: f64 = r.iter().map(|&v| v * v).sum();
    if rr <= 1e-30 {
        return d;
    }

    for _iter in 0..(2 * n) {
        let mut hp = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = 0.0_f64;
            for j in 0..n {
                s += h_mat[i * n + j] * p[j];
            }
            hp[i] = s;
        }
        let p_hp: f64 = (0..n).map(|i| p[i] * hp[i]).sum();
        let dd: f64 = d.iter().map(|&v| v * v).sum();
        let dp: f64 = (0..n).map(|i| d[i] * p[i]).sum();
        let pp: f64 = p.iter().map(|&v| v * v).sum();
        let disc = (dp * dp + pp * (delta * delta - dd)).max(0.0);
        let tau = (-dp + disc.sqrt()) / pp.max(1e-30);

        if p_hp <= 1e-30 || rr / p_hp >= tau {
            for i in 0..n {
                d[i] += tau * p[i];
            }
            return d;
        }
        let alpha = rr / p_hp;
        for i in 0..n {
            d[i] += alpha * p[i];
            r[i] += alpha * hp[i];
        }
        let rr_new: f64 = r.iter().map(|&v| v * v).sum();
        if rr_new.sqrt() <= 1e-12 * grad.iter().map(|&v| v * v).sum::<f64>().sqrt()
        {
            break;
        }
        let beta = rr_new / rr;
        for i in 0..n {
            p[i] = -r[i] + beta * p[i];
        }
        rr = rr_new;
    }
    d
}

/// Chooses a geometry-improving step of length `rho` orthogonal to the active
/// span when `kfar` is replaced.
fn geometry_step_unconstrained(
    n: usize,
    npt: usize,
    kopt: usize,
    kfar: usize,
    rho: f64,
    xpt: &[Vec<f64>],
) -> Vec<f64> {
    let dir: Vec<f64> = (0..n).map(|j| xpt[kfar][j] - xpt[kopt][j]).collect();
    let norm = dir.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if norm > 1e-15 && npt > 0 {
        dir.iter().map(|&v| (rho / norm) * v).collect()
    } else {
        let mut e = vec![0.0; n];
        e[0] = rho;
        e
    }
}
