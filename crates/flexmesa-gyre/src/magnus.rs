// SPDX-License-Identifier: GPL-3.0-only
//! GYRE eigendecomposition wrappers (`matrix/linalg_m.fypp:111-416`) and
//! Magnus difference-equation builders (`diff/magnus_diff_m.fypp`).

use crate::crmath::{exp_c, sqrt_c};
use crate::ext_complex::ExtC;
use crate::ext_real::ExtR;
use crate::fcomplex::Cplx;
use crate::forum::sort_indices;
use crate::grid::Point;
use crate::lapack::eig::dgeev;
use crate::lapack::zblas::{zcopy, zgemm, zscal};
use crate::lapack::zeig::zgeev;
use crate::GyreError;

const GL2_C_I: [f64; 1] = [0.5];

/// Closed-form eigendecomposition of a `2 x 2` column-major real matrix
/// (`eigen_decompose_2_r_`, `linalg_m.fypp:332-416`).
#[must_use]
pub fn eigen_decompose_2_r(a: &[f64; 4]) -> ([Cplx; 2], [Cplx; 4], [Cplx; 4]) {
    let c = Cplx::from_real;
    let b = -a[0] - a[3];
    let c_det = a[0] * a[3] - a[2] * a[1];
    let s = sqrt_c(Cplx::new(b * b - 4.0 * c_det, 0.0));
    let q = Cplx::real_mul(-0.5, c(b) + Cplx::real_mul(1.0_f64.copysign(b), s));
    let lambda = [q, c(c_det) / q];

    let mut vl = [c(a[1]), c(a[1]), lambda[0] - c(a[0]), lambda[1] - c(a[0])];
    let vr = [c(a[2]), lambda[0] - c(a[0]), c(a[2]), lambda[1] - c(a[0])];
    for i in 0..2 {
        let norm = vl[i] * vr[2 * i] + vl[i + 2] * vr[2 * i + 1];
        vl[i] = vl[i] / norm;
        vl[i + 2] = vl[i + 2] / norm;
    }
    (lambda, vl, vr)
}

/// Closed-form eigendecomposition of a `2 x 2` column-major complex matrix
/// (`eigen_decompose_2_c_`, `linalg_m.fypp:332-416`).
#[must_use]
pub fn eigen_decompose_2_c(a: &[Cplx; 4]) -> ([Cplx; 2], [Cplx; 4], [Cplx; 4]) {
    let b = -a[0] - a[3];
    let c_det = a[0] * a[3] - a[2] * a[1];
    let s = sqrt_c(b.powi2() - Cplx::real_mul(4.0, c_det));
    let sgn = 1.0_f64.copysign((b.conj() * s).re);
    let q = Cplx::real_mul(-0.5, b + Cplx::real_mul(sgn, s));
    let lambda = [q, c_det / q];

    let mut vl = [a[1], a[1], lambda[0] - a[0], lambda[1] - a[0]];
    let vr = [a[2], lambda[0] - a[0], a[2], lambda[1] - a[0]];
    for i in 0..2 {
        let norm = vl[i] * vr[2 * i] + vl[i + 2] * vr[2 * i + 1];
        vl[i] = vl[i] / norm;
        vl[i + 2] = vl[i + 2] / norm;
    }
    (lambda, vl, vr)
}

fn reconstruct_eigenvectors(
    n: usize,
    wr: &[f64],
    wi: &[f64],
    v_r: &[f64],
) -> Result<Vec<Cplx>, GyreError> {
    let mut v = vec![Cplx::new(0.0, 0.0); n * n];
    let mut i = 0;
    while i < n {
        if wi[i] == 0.0 {
            for r in 0..n {
                v[r + n * i] = Cplx::from_real(v_r[r + n * i]);
            }
            i += 1;
        } else if i + 1 < n && wr[i] == wr[i + 1] && wi[i] == -wi[i + 1] {
            for r in 0..n {
                let z = Cplx::new(v_r[r + n * i], v_r[r + n * (i + 1)]);
                v[r + n * i] = z;
                v[r + n * (i + 1)] = z.conj();
            }
            i += 2;
        } else {
            return Err(GyreError::Abort(
                "Complex eigenvalue does not have conjugate pair",
            ));
        }
    }
    Ok(v)
}

fn sort_eigenpairs(
    n: usize,
    lambda: &mut Vec<Cplx>,
    vl: &mut Vec<Cplx>,
    vr: &mut Vec<Cplx>,
) {
    let re: Vec<f64> = lambda.iter().map(|z| z.re).collect();
    let ind = sort_indices(&re, false);
    let mut l_s = vec![Cplx::new(0.0, 0.0); n];
    let mut vl_s = vec![Cplx::new(0.0, 0.0); n * n];
    let mut vr_s = vec![Cplx::new(0.0, 0.0); n * n];
    for (new_i, &idx_1) in ind.iter().enumerate() {
        let old_i = usize::try_from(idx_1 - 1).unwrap_or(0);
        l_s[new_i] = lambda[old_i];
        for c in 0..n {
            vl_s[new_i + n * c] = vl[old_i + n * c];
            vr_s[c + n * new_i] = vr[c + n * old_i];
        }
    }
    *lambda = l_s;
    *vl = vl_s;
    *vr = vr_s;
}

/// Result of [`eigen_decompose_r`] / [`eigen_decompose_c`].
#[derive(Debug, Clone)]
pub struct EigenDecomp {
    /// Eigenvalues `lambda(1..n)`.
    pub lambda: Vec<Cplx>,
    /// Left eigenvectors `V_l(1..n, 1..n)` column-major, normalized to `V_r`.
    pub vl: Vec<Cplx>,
    /// Right eigenvectors `V_r(1..n, 1..n)` column-major.
    pub vr: Vec<Cplx>,
}

/// `eigen_decompose_r_` (`linalg_m.fypp:111-244`). Overwrites `a`.
///
/// # Errors
///
/// [`GyreError::LapackInfo`] if `DGEEV` returns nonzero `info`, or
/// [`GyreError::Abort`] if complex eigenvalues lack a conjugate pair.
pub fn eigen_decompose_r(
    n: usize,
    a: &mut [f64],
    sort: bool,
) -> Result<EigenDecomp, GyreError> {
    if n == 2 {
        let arr: [f64; 4] = [a[0], a[1], a[2], a[3]];
        let (l2, vl2, vr2) = eigen_decompose_2_r(&arr);
        let mut lambda = l2.to_vec();
        let mut vl = vl2.to_vec();
        let mut vr = vr2.to_vec();
        if sort {
            sort_eigenpairs(n, &mut lambda, &mut vl, &mut vr);
        }
        return Ok(EigenDecomp { lambda, vl, vr });
    }

    let mut vl_r = vec![0.0_f64; n * n];
    let mut vr_r = vec![0.0_f64; n * n];
    let out = dgeev(b'V', b'V', n, a, n, &mut vl_r, n, &mut vr_r, n, 4 * n)?;
    if out.info != 0 {
        return Err(GyreError::LapackInfo {
            routine: "XGEEV",
            info: i64::try_from(out.info).unwrap_or(i64::MAX),
        });
    }
    let mut lambda: Vec<Cplx> = out
        .wr
        .iter()
        .zip(out.wi.iter())
        .map(|(&re, &im)| Cplx::new(re, im))
        .collect();
    let vl_cols = reconstruct_eigenvectors(n, &out.wr, &out.wi, &vl_r)?;
    let mut vr = reconstruct_eigenvectors(n, &out.wr, &out.wi, &vr_r)?;
    let mut vl = vec![Cplx::new(0.0, 0.0); n * n];
    for c in 0..n {
        for r in 0..n {
            vl[r + n * c] = vl_cols[c + n * r].conj();
        }
    }
    for i in 0..n {
        let mut norm = Cplx::new(0.0, 0.0);
        for k in 0..n {
            norm = norm + vl[i + n * k] * vr[k + n * i];
        }
        for k in 0..n {
            vl[i + n * k] = vl[i + n * k] / norm;
        }
    }
    if sort {
        sort_eigenpairs(n, &mut lambda, &mut vl, &mut vr);
    }
    Ok(EigenDecomp { lambda, vl, vr })
}

/// `eigen_decompose_c_` (`linalg_m.fypp:248-326`). Overwrites `a`.
///
/// # Errors
///
/// [`GyreError::LapackInfo`] if `ZGEEV` returns nonzero `info`.
pub fn eigen_decompose_c(
    n: usize,
    a: &mut [Cplx],
    sort: bool,
) -> Result<EigenDecomp, GyreError> {
    if n == 2 {
        let arr: [Cplx; 4] = [a[0], a[1], a[2], a[3]];
        let (l2, vl2, vr2) = eigen_decompose_2_c(&arr);
        let mut lambda = l2.to_vec();
        let mut vl = vl2.to_vec();
        let mut vr = vr2.to_vec();
        if sort {
            sort_eigenpairs(n, &mut lambda, &mut vl, &mut vr);
        }
        return Ok(EigenDecomp { lambda, vl, vr });
    }

    let mut vl_raw = vec![Cplx::new(0.0, 0.0); n * n];
    let mut vr = vec![Cplx::new(0.0, 0.0); n * n];
    let out = zgeev(b'V', b'V', n, a, n, &mut vl_raw, n, &mut vr, n, 2 * n)?;
    if out.info != 0 {
        return Err(GyreError::LapackInfo {
            routine: "XGEEV",
            info: i64::try_from(out.info).unwrap_or(i64::MAX),
        });
    }
    let mut lambda = out.w;
    let mut vl = vec![Cplx::new(0.0, 0.0); n * n];
    for c in 0..n {
        for r in 0..n {
            vl[r + n * c] = vl_raw[c + n * r].conj();
        }
    }
    for i in 0..n {
        let mut norm = Cplx::new(0.0, 0.0);
        for k in 0..n {
            norm = norm + vl[i + n * k] * vr[k + n * i];
        }
        for k in 0..n {
            vl[i + n * k] = vl[i + n * k] / norm;
        }
    }
    if sort {
        sort_eigenpairs(n, &mut lambda, &mut vl, &mut vr);
    }
    Ok(EigenDecomp { lambda, vl, vr })
}

/// Stencil setup for `magnus_diff_t` (`magnus_diff_m.fypp:95-141`).
/// Returns `(dx, pt)`.
///
/// # Errors
///
/// [`GyreError::Unported`] for `GL4`/`GL6`, [`GyreError::Abort`] for unknown
/// schemes.
pub fn magnus_stencil(
    pt_a: Point,
    pt_b: Point,
    scheme: &str,
) -> Result<(f64, Vec<Point>), GyreError> {
    let c_i: &[f64] = match scheme {
        "GL2" => &GL2_C_I,
        "GL4" | "GL6" => {
            return Err(GyreError::Unported {
                what: "magnus scheme other than 'GL2'",
                at: "diff/magnus_diff_m.fypp:114-120",
            })
        }
        _ => return Err(GyreError::Abort("invalid scheme")),
    };
    let dx = pt_b.x - pt_a.x;
    let pt = c_i
        .iter()
        .map(|&c| Point {
            s: pt_a.s,
            x: pt_a.x + c * dx,
        })
        .collect();
    Ok((dx, pt))
}

fn magnus_blocks(
    n_e: usize,
    dx: f64,
    eig: &EigenDecomp,
) -> Result<(Vec<Cplx>, Vec<Cplx>, Cplx), GyreError> {
    let lambda = &eig.lambda;
    let v_l = &eig.vl;
    let v_r = &eig.vr;
    let mut v_pos = vec![Cplx::new(0.0, 0.0); n_e * n_e];
    let mut v_neg = vec![Cplx::new(0.0, 0.0); n_e * n_e];
    for i in 0..n_e {
        let s = i * n_e;
        let e = s + n_e;
        zcopy(n_e, &v_r[s..e], 1, &mut v_pos[s..e], 1);
        if lambda[i].re >= 0.0 {
            let arg = (-lambda[i]).mul_real(dx);
            zscal(n_e, exp_c(arg), &mut v_pos[s..e], 1);
        }
    }
    for i in 0..n_e {
        let s = i * n_e;
        let e = s + n_e;
        zcopy(n_e, &v_r[s..e], 1, &mut v_neg[s..e], 1);
        if lambda[i].re < 0.0 {
            let arg = lambda[i].mul_real(dx);
            zscal(n_e, exp_c(arg), &mut v_neg[s..e], 1);
        }
    }
    let mut e_l = vec![Cplx::new(0.0, 0.0); n_e * n_e];
    let mut e_r = vec![Cplx::new(0.0, 0.0); n_e * n_e];
    zgemm(
        b'N',
        b'N',
        n_e,
        n_e,
        n_e,
        Cplx::new(-1.0, 0.0),
        &v_neg,
        n_e,
        v_l,
        n_e,
        Cplx::new(0.0, 0.0),
        &mut e_l,
        n_e,
    )?;
    zgemm(
        b'N',
        b'N',
        n_e,
        n_e,
        n_e,
        Cplx::new(1.0, 0.0),
        &v_pos,
        n_e,
        v_l,
        n_e,
        Cplx::new(0.0, 0.0),
        &mut e_r,
        n_e,
    )?;
    let mut sum = Cplx::new(0.0, 0.0);
    for &lam in lambda {
        if lam.re >= 0.0 {
            sum = sum + lam;
        }
    }
    Ok((e_l, e_r, sum.mul_real(dx)))
}

/// Real `magnus_diff_t%build` (`magnus_diff_m.fypp:145-256`) from `dOmega`.
///
/// # Errors
///
/// Eigendecomposition errors.
pub fn magnus_build_r(
    n_e: usize,
    dx: f64,
    domega: &mut [f64],
) -> Result<(Vec<f64>, Vec<f64>, ExtR), GyreError> {
    let eig = eigen_decompose_r(n_e, domega, false)?;
    let (e_l_c, e_r_c, arg) = magnus_blocks(n_e, dx, &eig)?;
    let e_l = e_l_c.into_iter().map(|z| z.re).collect();
    let e_r = e_r_c.into_iter().map(|z| z.re).collect();
    let scl = ExtR::from_real(arg.re).exp();
    Ok((e_l, e_r, scl))
}

/// Complex `magnus_diff_t%build` (`magnus_diff_m.fypp:145-256`) from `dOmega`.
///
/// # Errors
///
/// Eigendecomposition errors.
pub fn magnus_build_c(
    n_e: usize,
    dx: f64,
    domega: &mut [Cplx],
) -> Result<(Vec<Cplx>, Vec<Cplx>, ExtC), GyreError> {
    let eig = eigen_decompose_c(n_e, domega, false)?;
    let (e_l, e_r, arg) = magnus_blocks(n_e, dx, &eig)?;
    let scl = ExtC::from_cplx(arg).exp();
    Ok((e_l, e_r, scl))
}
