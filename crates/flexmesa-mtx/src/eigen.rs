// SPDX-License-Identifier: LGPL-3.0-or-later
//! Real nonsymmetric eigenvalue and right-eigenvector solver (`DGEEV` parity)
//! using Parlett-Reinsch radix-2 balancing (`dgebal`), Householder upper
//! Hessenberg reduction (`dgehrd`), Francis implicit double-shift QR Schur
//! decomposition (`dhseqr`), and quasi-triangular back-substitution (`dtrevc`).
//!
//! Required by the $4N \times 4N$ complex non-adiabatic linear pulsation
//! stability solver (`LINA`, `star/private/rsp_lina.f90`).

use crate::eigen_prep::{balance_matrix, reduce_to_hessenberg};
use crate::eigen_vecs::{
    backtransform_and_normalize_right_vecs, compute_schur_right_eigenvectors,
};
use crate::MtxError;
use flexmesa_utils::is_bad_slice;

/// Eigenvalues and right eigenvectors of a real $N \times N$ nonsymmetric
/// matrix $A$, matching LAPACK `DGEEV` right-eigenvector storage.
#[derive(Debug, Clone)]
pub struct EigenResult {
    /// Real parts of the $N$ eigenvalues $\mathrm{Re}(\lambda_j)$.
    pub wr: Vec<f64>,
    /// Imaginary parts of the $N$ eigenvalues $\mathrm{Im}(\lambda_j)$.
    /// Complex-conjugate pairs appear consecutively with `wi[j] > 0.0` and
    /// `wi[j + 1] = -wi[j]`.
    pub wi: Vec<f64>,
    /// Row-major $N \times N$ right-eigenvector matrix (`vr[i * n + j]`).
    ///
    /// - If `wi[j] == 0.0`, column `j` is the real unit eigenvector $v_j$.
    /// - If `wi[j] > 0.0`, column `j` holds $\mathrm{Re}(v_j)$ and column
    ///   `j + 1` holds $\mathrm{Im}(v_j)$ for $\lambda_j = \mathrm{wr}_j +
    ///   i\,\mathrm{wi}_j$ (and $\bar{v}_j$ corresponds to $\lambda_{j+1}$).
    pub vr: Vec<f64>,
}

/// Computes the Givens rotation `(cs, sn)` that reduces a $2 \times 2$ block to
/// standardized real Schur form (`dlanv2`) and returns
/// `(a_out, b_out, c_out, d_out, cs, sn, wr1, wi1, wr2, wi2)`.
#[inline]
fn standardize_2x2(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let (cs, sn) = if c == 0.0 {
        (1.0, 0.0)
    } else if b == 0.0 {
        (0.0, 1.0)
    } else if (a - d) == 0.0 && (b > 0.0) != (c > 0.0) {
        (1.0, 0.0)
    } else {
        let p = 0.5 * (a - d);
        let bc = b * c;
        let disc = p * p + bc;
        if disc >= 0.0 {
            let z = p + disc.sqrt().copysign(if p >= 0.0 { 1.0 } else { -1.0 });
            let tau = z.hypot(c);
            if tau == 0.0 {
                (1.0, 0.0)
            } else {
                (z / tau, c / tau)
            }
        } else {
            let s = f64::midpoint(b, c);
            if s == 0.0 {
                let h = std::f64::consts::FRAC_1_SQRT_2;
                (h, h)
            } else {
                let r = p.hypot(s);
                let cs_v = f64::midpoint(1.0, s.abs() / r).sqrt();
                let sn_v = -(p / (2.0 * r * cs_v))
                    * (if s < 0.0 { -1.0 } else { 1.0 });
                (cs_v, sn_v)
            }
        }
    };

    // B' = Q^T * B * Q with Q = [[cs, -sn], [sn, cs]]
    let t00 = cs * a + sn * c;
    let t01 = cs * b + sn * d;
    let t10 = -sn * a + cs * c;
    let t11 = -sn * b + cs * d;
    let mut aa = t00 * cs + t01 * sn;
    let bb = -t00 * sn + t01 * cs;
    let mut cc = t10 * cs + t11 * sn;
    let mut dd = -t10 * sn + t11 * cs;

    if cc != 0.0 {
        let p = 0.5 * (aa - dd);
        if p * p + bb * cc >= 0.0 {
            cc = 0.0;
        } else {
            let mean = f64::midpoint(aa, dd);
            aa = mean;
            dd = mean;
            if (bb > 0.0) == (cc > 0.0) {
                cc = 0.0;
            }
        }
    }

    if cc == 0.0 {
        (aa, bb, 0.0, dd, cs, sn, aa, 0.0, dd, 0.0)
    } else {
        let wi = (bb.abs().sqrt()) * (cc.abs().sqrt());
        (aa, bb, cc, dd, cs, sn, aa, wi, dd, -wi)
    }
}

/// Constructs a 2- or 3-element Householder reflector `(tau, u1, u2)` mapping
/// `(x0, x1, x2)` to `(beta, 0, 0)`.
#[inline]
fn reflector_3(x0: f64, x1: f64, x2: f64, len: usize) -> (f64, f64, f64) {
    let norm_tail = if len == 3 { x1.hypot(x2) } else { x1.abs() };
    if norm_tail == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let norm = x0.hypot(norm_tail);
    let beta = if x0 >= 0.0 { -norm } else { norm };
    let tau = (beta - x0) / beta;
    let denom = x0 - beta;
    let u1 = x1 / denom;
    let u2 = if len == 3 { x2 / denom } else { 0.0 };
    (tau, u1, u2)
}

/// Francis implicit double-shift QR iteration (`dhseqr`) reducing upper
/// Hessenberg `h` in-place to real Schur form $T$ (when `want_t` is `true`)
/// and accumulating `z` when provided.
fn schur_qr(
    n: usize,
    ilo: usize,
    ihi: usize,
    h: &mut [f64],
    mut z: Option<&mut [f64]>,
    want_t: bool,
    wr: &mut [f64],
    wi: &mut [f64],
) -> Result<(), MtxError> {
    for i in 0..n {
        if i < ilo || i > ihi {
            wr[i] = h[i * n + i];
            wi[i] = 0.0;
        }
    }
    if ilo > ihi {
        return Ok(());
    }

    let ulp = f64::EPSILON;
    let smlnum = f64::MIN_POSITIVE * ((ihi - ilo + 1).max(1) as f64) / ulp;

    let mut h_norm = 0.0_f64;
    for i in ilo..=ihi {
        for j in i.saturating_sub(1)..=ihi {
            h_norm = h_norm.hypot(h[i * n + j]);
        }
    }

    let max_total_iters = 60 * (ihi - ilo + 1).max(10);
    let mut total_iters = 0usize;
    let mut m = ihi;
    let mut its = 0usize;

    while m >= ilo {
        // Find active unreduced submatrix `l..=m`.
        let mut l = m;
        while l > ilo {
            let d_sum = h[(l - 1) * n + (l - 1)].abs() + h[l * n + l].abs();
            let tst = if d_sum == 0.0 { h_norm } else { d_sum };
            if h[l * n + (l - 1)].abs() <= (ulp * tst).max(smlnum) {
                h[l * n + (l - 1)] = 0.0;
                break;
            }
            l -= 1;
        }

        if l == m {
            wr[m] = h[m * n + m];
            wi[m] = 0.0;
            its = 0;
            if m == ilo {
                break;
            }
            m -= 1;
            continue;
        }

        if l + 1 == m {
            let (aa, bb, cc, dd, cs, sn, wr1, wi1, wr2, wi2) = standardize_2x2(
                h[l * n + l],
                h[l * n + m],
                h[m * n + l],
                h[m * n + m],
            );
            h[l * n + l] = aa;
            h[l * n + m] = bb;
            h[m * n + l] = cc;
            h[m * n + m] = dd;
            if want_t {
                for col in (m + 1)..n {
                    let t1 = h[l * n + col];
                    let t2 = h[m * n + col];
                    h[l * n + col] = cs * t1 + sn * t2;
                    h[m * n + col] = -sn * t1 + cs * t2;
                }
                for row in 0..l {
                    let t1 = h[row * n + l];
                    let t2 = h[row * n + m];
                    h[row * n + l] = cs * t1 + sn * t2;
                    h[row * n + m] = -sn * t1 + cs * t2;
                }
            }
            if let Some(z_mat) = z.as_deref_mut() {
                for row in 0..n {
                    let t1 = z_mat[row * n + l];
                    let t2 = z_mat[row * n + m];
                    z_mat[row * n + l] = cs * t1 + sn * t2;
                    z_mat[row * n + m] = -sn * t1 + cs * t2;
                }
            }
            wr[l] = wr1;
            wi[l] = wi1;
            wr[m] = wr2;
            wi[m] = wi2;
            its = 0;
            if l <= ilo {
                break;
            }
            m = l - 1;
            continue;
        }

        if total_iters >= max_total_iters {
            return Err(MtxError::SingularMatrix(m));
        }
        its += 1;
        total_iters += 1;

        // Compute double-shift pair: either two real shifts `(sr1, sr2)` with
        // `si1 = 0`, or complex conjugates `sr1 +- i*si1` with `sr1 == sr2`.
        let (sr1, si1, sr2) = if its.is_multiple_of(30) {
            let s = h[m * n + (m - 1)].abs() + h[(m - 1) * n + (m - 2)].abs();
            (h[m * n + m] + 0.5 * s, 0.0, h[m * n + m] - 0.5 * s)
        } else if its.is_multiple_of(10) {
            let s = h[m * n + (m - 1)].abs() + h[(m - 1) * n + (m - 2)].abs();
            let a = 0.75 * s + h[m * n + m];
            let d = a;
            let b = -0.4375 * s;
            let c = s;
            let (_, _, _, _, _, _, r1, i1, r2, _) = standardize_2x2(a, b, c, d);
            (r1, i1.abs(), r2)
        } else {
            let (_, _, _, _, _, _, r1, i1, r2, _) = standardize_2x2(
                h[(m - 1) * n + (m - 1)],
                h[(m - 1) * n + m],
                h[m * n + (m - 1)],
                h[m * n + m],
            );
            (r1, i1.abs(), r2)
        };

        // Choose bulge-chase start index `m_start` (`l..=m-2`) as in `dlahqr`.
        let mut m_start = l;
        let mut v0 = 0.0_f64;
        let mut v1 = 0.0_f64;
        let mut v2 = 0.0_f64;
        for k in (l..=(m - 2)).rev() {
            let h00 = h[k * n + k];
            let h11 = h[(k + 1) * n + (k + 1)];
            let s = (h00 - sr2).abs() + si1 + h[(k + 1) * n + k].abs();
            if s == 0.0 {
                v0 = 0.0;
                v1 = 0.0;
                v2 = 0.0;
            } else {
                let h10s = h[(k + 1) * n + k] / s;
                v0 = h10s * h[k * n + (k + 1)]
                    + ((h00 - sr1) * ((h00 - sr2) / s) + si1 * (si1 / s));
                v1 = h10s * (h00 + h11 - sr1 - sr2);
                v2 = h10s * h[(k + 2) * n + (k + 1)];
                let v_sum = v0.abs() + v1.abs() + v2.abs();
                if v_sum > 0.0 {
                    v0 /= v_sum;
                    v1 /= v_sum;
                    v2 /= v_sum;
                }
            }
            if k == l {
                m_start = l;
                break;
            }
            let lhs = h[k * n + (k - 1)].abs() * (v1.abs() + v2.abs());
            let rhs = ulp
                * v0.abs()
                * (h[(k - 1) * n + (k - 1)].abs() + h00.abs() + h11.abs());
            if lhs <= rhs {
                m_start = k;
                break;
            }
        }

        let col_end = if want_t { n } else { m + 1 };
        let row_start = if want_t { 0 } else { l };

        for k in m_start..m {
            let nr = (m - k + 1).min(3);
            if k > m_start {
                v0 = h[k * n + (k - 1)];
                v1 = h[(k + 1) * n + (k - 1)];
                v2 = if nr == 3 {
                    h[(k + 2) * n + (k - 1)]
                } else {
                    0.0
                };
            }
            let (tau, u1, u2) = reflector_3(v0, v1, v2, nr);
            if tau == 0.0 {
                continue;
            }
            if k > m_start {
                let norm = if nr == 3 {
                    v0.hypot(v1.hypot(v2))
                } else {
                    v0.hypot(v1)
                };
                h[k * n + (k - 1)] = if v0 >= 0.0 { -norm } else { norm };
                h[(k + 1) * n + (k - 1)] = 0.0;
                if nr == 3 {
                    h[(k + 2) * n + (k - 1)] = 0.0;
                }
            } else if m_start > l {
                h[k * n + (k - 1)] *= 1.0 - tau;
            }

            if nr == 3 {
                for j in k..col_end {
                    let sum = h[k * n + j]
                        + u1 * h[(k + 1) * n + j]
                        + u2 * h[(k + 2) * n + j];
                    let t = tau * sum;
                    h[k * n + j] -= t;
                    h[(k + 1) * n + j] -= t * u1;
                    h[(k + 2) * n + j] -= t * u2;
                }
                let r_end = (k + 3).min(m);
                for i in row_start..=r_end {
                    let sum = h[i * n + k]
                        + u1 * h[i * n + (k + 1)]
                        + u2 * h[i * n + (k + 2)];
                    let t = tau * sum;
                    h[i * n + k] -= t;
                    h[i * n + (k + 1)] -= t * u1;
                    h[i * n + (k + 2)] -= t * u2;
                }
                if let Some(z_mat) = z.as_deref_mut() {
                    for i in 0..n {
                        let sum = z_mat[i * n + k]
                            + u1 * z_mat[i * n + (k + 1)]
                            + u2 * z_mat[i * n + (k + 2)];
                        let t = tau * sum;
                        z_mat[i * n + k] -= t;
                        z_mat[i * n + (k + 1)] -= t * u1;
                        z_mat[i * n + (k + 2)] -= t * u2;
                    }
                }
            } else {
                for j in k..col_end {
                    let sum = h[k * n + j] + u1 * h[(k + 1) * n + j];
                    let t = tau * sum;
                    h[k * n + j] -= t;
                    h[(k + 1) * n + j] -= t * u1;
                }
                for i in row_start..=m {
                    let sum = h[i * n + k] + u1 * h[i * n + (k + 1)];
                    let t = tau * sum;
                    h[i * n + k] -= t;
                    h[i * n + (k + 1)] -= t * u1;
                }
                if let Some(z_mat) = z.as_deref_mut() {
                    for i in 0..n {
                        let sum =
                            z_mat[i * n + k] + u1 * z_mat[i * n + (k + 1)];
                        let t = tau * sum;
                        z_mat[i * n + k] -= t;
                        z_mat[i * n + (k + 1)] -= t * u1;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Computes all $N$ eigenvalues `(wr, wi)` of a row-major $N \times N$ real
/// nonsymmetric matrix `a` without computing eigenvectors.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] if `n == 0`,
/// [`MtxError::DimensionMismatch`] if `a.len() != n * n`,
/// [`MtxError::NonFiniteInput`] if `a` contains `NaN` or $\pm\infty$, or
/// [`MtxError::SingularMatrix`] if QR iteration fails to converge.
pub fn dgeev_eigenvalues(
    n: usize,
    a: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), MtxError> {
    if n == 0 {
        return Err(MtxError::InvalidParameter("n must be >= 1"));
    }
    if a.len() != n * n {
        return Err(MtxError::DimensionMismatch("a must have length n * n"));
    }
    if is_bad_slice(a) {
        return Err(MtxError::NonFiniteInput("matrix a contains NaN/Inf"));
    }

    let mut work = a.to_vec();
    let (ilo, ihi, _, _) = balance_matrix(n, &mut work);
    reduce_to_hessenberg(n, ilo, ihi, &mut work, None);
    let mut wr = vec![0.0_f64; n];
    let mut wi = vec![0.0_f64; n];
    schur_qr(n, ilo, ihi, &mut work, None, false, &mut wr, &mut wi)?;
    Ok((wr, wi))
}

/// Computes all $N$ eigenvalues and right eigenvectors of a row-major
/// $N \times N$ real nonsymmetric matrix `a` (`DGEEV('N', 'V', ...)` parity).
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] if `n == 0`,
/// [`MtxError::DimensionMismatch`] if `a.len() != n * n`,
/// [`MtxError::NonFiniteInput`] if `a` contains `NaN` or $\pm\infty$, or
/// [`MtxError::SingularMatrix`] if QR iteration fails to converge.
pub fn dgeev_right(n: usize, a: &[f64]) -> Result<EigenResult, MtxError> {
    if n == 0 {
        return Err(MtxError::InvalidParameter("n must be >= 1"));
    }
    if a.len() != n * n {
        return Err(MtxError::DimensionMismatch("a must have length n * n"));
    }
    if is_bad_slice(a) {
        return Err(MtxError::NonFiniteInput("matrix a contains NaN/Inf"));
    }

    let mut t = a.to_vec();
    let (ilo, ihi, scale, perm) = balance_matrix(n, &mut t);
    let mut z = vec![0.0_f64; n * n];
    reduce_to_hessenberg(n, ilo, ihi, &mut t, Some(&mut z));

    let mut wr = vec![0.0_f64; n];
    let mut wi = vec![0.0_f64; n];
    schur_qr(n, ilo, ihi, &mut t, Some(&mut z), true, &mut wr, &mut wi)?;

    let mut t_norm = 0.0_f64;
    for i in 0..n {
        for j in i.saturating_sub(1)..n {
            t_norm = t_norm.max(t[i * n + j].abs());
        }
    }

    let mut vr = vec![0.0_f64; n * n];
    compute_schur_right_eigenvectors(n, &t, &z, &wr, &wi, t_norm, &mut vr);
    backtransform_and_normalize_right_vecs(
        n, ilo, ihi, &scale, &perm, &wi, &mut vr,
    );

    Ok(EigenResult { wr, wi, vr })
}
