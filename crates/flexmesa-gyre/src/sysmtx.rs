// SPDX-License-Identifier: GPL-3.0-only
//! GYRE's block-bidiagonal system matrix for real problems
//! (`block_sysmtx_rt`, `matrix/block_sysmtx_m.fypp`): row scaling, cyclic
//! structured-Gaussian-elimination factorisation, determinant, and the
//! homogeneous solution by SVD and back-substitution. Its `sing_decompose`
//! is `matrix/linalg_m.fypp:703-733`.

use crate::ext_real::{self, ExtR};
use crate::grid::fmax;
use crate::lapack::aux::dlaswp;
use crate::lapack::blas::{dgemm, dgemv, dtrsm};
use crate::lapack::lu::dgetrf;
use crate::lapack::svd::dgesvd;
use crate::GyreError;

/// `block_sysmtx_rt`.
#[derive(Debug, Clone)]
pub struct BlockSysmtx {
    /// Number of blocks `n`.
    pub n: usize,
    /// Equations per point `n_e`.
    pub n_e: usize,
    /// Inner conditions `n_i`.
    pub n_i: usize,
    /// Outer conditions `n_o`.
    pub n_o: usize,
    /// `E_l(:,:,k)`, column-major `n_e x n_e` each.
    pub e_l: Vec<Vec<f64>>,
    /// `E_r(:,:,k)`.
    pub e_r: Vec<Vec<f64>>,
    /// `B_i(n_i, n_e)`, column-major.
    pub b_i: Vec<f64>,
    /// `B_o(n_o, n_e)`, column-major.
    pub b_o: Vec<f64>,
    /// `scl_i(n_i)`.
    pub scl_i: Vec<f64>,
    /// `scl_o(n_o)`.
    pub scl_o: Vec<f64>,
    /// `scl(n)`.
    pub scl: Vec<ExtR>,
}

fn maxabs(it: impl Iterator<Item = f64>) -> f64 {
    // MAXVAL of a non-empty array: running maximum from the first element.
    let mut m = f64::NEG_INFINITY;
    let mut first = true;
    for v in it {
        let a = v.abs();
        if first {
            m = a;
            first = false;
        } else if a > m {
            m = a;
        }
    }
    m
}

fn info_err(routine: &'static str, info: usize) -> GyreError {
    GyreError::LapackInfo {
        routine,
        info: i64::try_from(info).unwrap_or(i64::MAX),
    }
}

impl BlockSysmtx {
    /// `block_sysmtx_rt(n, n_e, n_i, n_o)` (`block_sysmtx_m.fypp:87-119`).
    #[must_use]
    pub fn new(n: usize, n_e: usize, n_i: usize, n_o: usize) -> Self {
        Self {
            n,
            n_e,
            n_i,
            n_o,
            e_l: vec![vec![0.0; n_e * n_e]; n],
            e_r: vec![vec![0.0; n_e * n_e]; n],
            b_i: vec![0.0; n_i * n_e],
            b_o: vec![0.0; n_o * n_e],
            scl_i: vec![0.0; n_i],
            scl_o: vec![0.0; n_o],
            scl: vec![ExtR::from_real(1.0); n],
        }
    }

    /// `set_B_i(B, scl)`.
    pub fn set_b_i(&mut self, b: &[f64], scl: &[f64]) {
        self.b_i.copy_from_slice(b);
        self.scl_i.copy_from_slice(scl);
    }

    /// `set_B_o(B, scl)`.
    pub fn set_b_o(&mut self, b: &[f64], scl: &[f64]) {
        self.b_o.copy_from_slice(b);
        self.scl_o.copy_from_slice(scl);
    }

    /// `set_E(k, E_l, E_r, scl)` with a 0-based `k`.
    pub fn set_e(&mut self, k: usize, e_l: &[f64], e_r: &[f64], scl: ExtR) {
        self.e_l[k].copy_from_slice(e_l);
        self.e_r[k].copy_from_slice(e_r);
        self.scl[k] = scl;
    }

    /// `scale_rows_()` (`block_sysmtx_m.fypp:314-351`).
    fn scale_rows(&mut self) {
        let (n_e, n_i, n_o) = (self.n_e, self.n_i, self.n_o);
        for i in 0..n_i {
            let scl = maxabs((0..n_e).map(|c| self.b_i[i + c * n_i]));
            for c in 0..n_e {
                self.b_i[i + c * n_i] /= scl;
            }
            self.scl_i[i] *= scl;
        }
        for k in 0..self.n {
            for i in 0..n_e {
                let scl = fmax(
                    maxabs((0..n_e).map(|c| self.e_l[k][i + c * n_e])),
                    maxabs((0..n_e).map(|c| self.e_r[k][i + c * n_e])),
                );
                for c in 0..n_e {
                    self.e_l[k][i + c * n_e] /= scl;
                    self.e_r[k][i + c * n_e] /= scl;
                }
                self.scl[k] = self.scl[k].mul_r(scl);
            }
        }
        for i in 0..n_o {
            let scl = maxabs((0..n_e).map(|c| self.b_o[i + c * n_o]));
            for c in 0..n_e {
                self.b_o[i + c * n_o] /= scl;
            }
            self.scl_o[i] *= scl;
        }
    }

    /// `factor()` (`block_sysmtx_m.fypp:203-310`).
    ///
    /// # Errors
    ///
    /// LAPACK errors.
    pub fn factor(&mut self) -> Result<(), GyreError> {
        self.scale_rows();
        let (n, n_e) = (self.n, self.n_e);
        let ld = 2 * n_e;
        let mut l = 1;
        while l < n {
            let mut k = 1;
            while k <= n - l {
                let (k0, kl) = (k - 1, k - 1 + l);
                let mut m_g = vec![0.0_f64; ld * n_e];
                let mut m_u = vec![0.0_f64; ld * n_e];
                let mut m_e = vec![0.0_f64; ld * n_e];
                for c in 0..n_e {
                    for r in 0..n_e {
                        m_g[r + c * ld] = self.e_l[k0][r + c * n_e];
                        m_g[n_e + r + c * ld] = 0.0;
                        m_u[r + c * ld] = self.e_r[k0][r + c * n_e];
                        m_u[n_e + r + c * ld] = self.e_l[kl][r + c * n_e];
                        m_e[r + c * ld] = 0.0;
                        m_e[n_e + r + c * ld] = self.e_r[kl][r + c * n_e];
                    }
                }
                let mut ipiv = vec![0usize; n_e];
                let info = dgetrf(ld, n_e, &mut m_u, ld, &mut ipiv)?;
                // `@:ASSERT(info >= 0)` always holds for a returned INFO.
                let _ = info;
                for mm in [&mut m_e, &mut m_g] {
                    dlaswp(n_e, mm, ld, 1, n_e, &ipiv, 1);
                    dtrsm(
                        b'L', b'L', b'N', b'U', n_e, n_e, 1.0, &m_u, ld, mm, ld,
                    )?;
                    // XGEMM_inplace('N','N', n_e, n_e, n_e, -1, M_U(n_e+1,1), 2n_e,
                    //               M(1,1), 2n_e, 1, M(n_e+1,1), 2n_e)
                    let b_top: Vec<f64> = mm.clone();
                    dgemm(
                        b'N',
                        b'N',
                        n_e,
                        n_e,
                        n_e,
                        -1.0,
                        &m_u[n_e..],
                        ld,
                        &b_top,
                        ld,
                        1.0,
                        &mut mm[n_e..],
                        ld,
                    )?;
                    dtrsm(
                        b'L', b'U', b'N', b'N', n_e, n_e, 1.0, &m_u, ld, mm, ld,
                    )?;
                }
                for c in 0..n_e {
                    for r in 0..n_e {
                        self.e_l[k0][r + c * n_e] = m_g[n_e + r + c * ld];
                        self.e_r[k0][r + c * n_e] = m_e[n_e + r + c * ld];
                        self.e_l[kl][r + c * n_e] = m_g[r + c * ld];
                        self.e_r[kl][r + c * n_e] = m_e[r + c * ld];
                    }
                }
                // scl(k) = scl(k)*PRODUCT(diagonal(M_U))
                let mut prod = 1.0_f64;
                for i in 0..n_e {
                    prod *= m_u[i + i * ld];
                }
                self.scl[k0] = self.scl[k0].mul_r(prod);
                for (i, &p) in ipiv.iter().enumerate() {
                    if p != i + 1 {
                        self.scl[k0] = self.scl[k0].negate();
                    }
                }
                k += 2 * l;
            }
            l *= 2;
        }
        Ok(())
    }

    /// The `2 n_e x 2 n_e` boundary matrix assembled by `det_` and
    /// `soln_vec_hom_` (`block_sysmtx_m.fypp:371-377`, `:432-438`).
    fn boundary_matrix(&self) -> Vec<f64> {
        let (n_e, n_i, n_o) = (self.n_e, self.n_i, self.n_o);
        let ld = 2 * n_e;
        let mut m = vec![0.0_f64; ld * ld];
        for c in 0..n_e {
            for r in 0..n_i {
                m[r + c * ld] = self.b_i[r + c * n_i];
            }
            for r in 0..n_e {
                m[n_i + r + c * ld] = self.e_l[0][r + c * n_e];
            }
            for r in n_i + n_e..ld {
                m[r + c * ld] = 0.0;
            }
            for r in 0..n_i {
                m[r + (n_e + c) * ld] = 0.0;
            }
            for r in 0..n_e {
                m[n_i + r + (n_e + c) * ld] = self.e_r[0][r + c * n_e];
            }
            for r in 0..n_o {
                m[n_i + n_e + r + (n_e + c) * ld] = self.b_o[r + c * n_o];
            }
        }
        m
    }

    /// `det()` (`block_sysmtx_m.fypp:355-398`).
    ///
    /// # Errors
    ///
    /// LAPACK errors.
    pub fn det(&self) -> Result<ExtR, GyreError> {
        let n_e = self.n_e;
        let ld = 2 * n_e;
        let mut m = self.boundary_matrix();
        let mut ipiv = vec![0usize; ld];
        let _info = dgetrf(ld, ld, &mut m, ld, &mut ipiv)?;
        let mut parts: Vec<ExtR> =
            (0..ld).map(|i| ExtR::from_real(m[i + i * ld])).collect();
        parts.extend(self.scl_i.iter().map(|&s| ExtR::from_real(s)));
        parts.extend(self.scl.iter().copied());
        parts.extend(self.scl_o.iter().map(|&s| ExtR::from_real(s)));
        let mut det = ext_real::product(&parts);
        for (i, &p) in ipiv.iter().enumerate() {
            if p != i + 1 {
                det = det.negate();
            }
        }
        Ok(det)
    }

    /// `soln_vec_hom()` (`block_sysmtx_m.fypp:402-496`) with `USE_SVD`.
    ///
    /// # Errors
    ///
    /// LAPACK errors and a non-zero `XGESVD` return (`ASSERT`).
    pub fn soln_vec_hom(&self) -> Result<Vec<f64>, GyreError> {
        let n_e = self.n_e;
        let ld = 2 * n_e;
        let mut m = self.boundary_matrix();
        // sing_decompose_r_ (linalg_m.fypp:703-733): XGESVD('A','A',n,n,...,5n)
        let mut su = vec![0.0_f64; ld * ld];
        let mut sv_h = vec![0.0_f64; ld * ld];
        let o = dgesvd(ld, ld, &mut m, &mut su, &mut sv_h, 5 * ld)?;
        if o.info != 0 {
            return Err(info_err("XGESVD", o.info));
        }
        // u_bound = sV_H(2*n_e,:)
        let u_bound: Vec<f64> =
            (0..ld).map(|c| sv_h[(ld - 1) + c * ld]).collect();
        self.backsub(&u_bound)
    }

    /// `backsub_(u_bound, u)` (`block_sysmtx_m.fypp:564-625`).
    fn backsub(&self, u_bound: &[f64]) -> Result<Vec<f64>, GyreError> {
        let (n, n_e) = (self.n, self.n_e);
        let mut u = vec![0.0_f64; n_e * (n + 1)];
        u[..n_e].copy_from_slice(&u_bound[..n_e]);
        u[n_e * n..].copy_from_slice(&u_bound[n_e..]);
        let mut l = 1;
        while l < n {
            l *= 2;
        }
        loop {
            l /= 2;
            if l == 0 {
                break Ok(u);
            }
            let mut k = 1;
            while k <= n - l {
                let i_a = (k - 1) * n_e;
                let i_b = i_a + l * n_e;
                let i_c = (i_b + l * n_e).min(n_e * n);
                for v in &mut u[i_b..i_b + n_e] {
                    *v = 0.0;
                }
                let x_a: Vec<f64> = u[i_a..i_a + n_e].to_vec();
                let x_c: Vec<f64> = u[i_c..i_c + n_e].to_vec();
                let (e_l, e_r) = (&self.e_l[k - 1 + l], &self.e_r[k - 1 + l]);
                let y = &mut u[i_b..i_b + n_e];
                // XGEMV_inplace('N', n_e, n_e, -1, E_l(1,1,k+l), n_e, u(i_a), 1, 0, u(i_b), 1)
                dgemv(b'N', n_e, n_e, -1.0, e_l, n_e, &x_a, 1, 0.0, y, 1)?;
                dgemv(b'N', n_e, n_e, -1.0, e_r, n_e, &x_c, 1, 1.0, y, 1)?;
                k += 2 * l;
            }
        }
    }
}
