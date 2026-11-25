// SPDX-License-Identifier: GPL-3.0-only
//! GYRE's block-bidiagonal system matrix (`block_sysmtx_rt` and
//! `block_sysmtx_ct`, `matrix/block_sysmtx_m.fypp`): row scaling, cyclic
//! structured-Gaussian-elimination factorisation, determinant, and the
//! homogeneous solution by SVD and back-substitution. Its `sing_decompose`
//! is `matrix/linalg_m.fypp:703-768`.

use crate::grid::fmax;
use crate::scalar::Scalar;
use crate::GyreError;

/// `block_sysmtx_rt` (`T = f64`) and `block_sysmtx_ct` (`T = Cplx`).
#[derive(Debug, Clone)]
pub struct BlockSysmtx<T: Scalar> {
    /// Number of blocks `n`.
    pub n: usize,
    /// Equations per point `n_e`.
    pub n_e: usize,
    /// Inner conditions `n_i`.
    pub n_i: usize,
    /// Outer conditions `n_o`.
    pub n_o: usize,
    /// `E_l(:,:,k)`, column-major `n_e x n_e` each.
    pub e_l: Vec<Vec<T>>,
    /// `E_r(:,:,k)`.
    pub e_r: Vec<Vec<T>>,
    /// `B_i(n_i, n_e)`, column-major.
    pub b_i: Vec<T>,
    /// `B_o(n_o, n_e)`, column-major.
    pub b_o: Vec<T>,
    /// `scl_i(n_i)`.
    pub scl_i: Vec<T>,
    /// `scl_o(n_o)`.
    pub scl_o: Vec<T>,
    /// `scl(n)`.
    pub scl: Vec<T::Ext>,
}

fn maxabs<T: Scalar>(it: impl Iterator<Item = T>) -> f64 {
    // MAXVAL of a non-empty array: running maximum from the first element.
    let mut m = f64::NEG_INFINITY;
    let mut first = true;
    for v in it {
        let a = v.abs_m();
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

impl<T: Scalar> BlockSysmtx<T> {
    /// `block_sysmtx_rt(n, n_e, n_i, n_o)` (`block_sysmtx_m.fypp:87-119`).
    #[must_use]
    pub fn new(n: usize, n_e: usize, n_i: usize, n_o: usize) -> Self {
        Self {
            n,
            n_e,
            n_i,
            n_o,
            e_l: vec![vec![T::zero(); n_e * n_e]; n],
            e_r: vec![vec![T::zero(); n_e * n_e]; n],
            b_i: vec![T::zero(); n_i * n_e],
            b_o: vec![T::zero(); n_o * n_e],
            scl_i: vec![T::zero(); n_i],
            scl_o: vec![T::zero(); n_o],
            scl: vec![T::ext_r(1.0); n],
        }
    }

    /// `set_B_i(B, scl)`.
    pub fn set_b_i(&mut self, b: &[T], scl: &[T]) {
        self.b_i.copy_from_slice(b);
        self.scl_i.copy_from_slice(scl);
    }

    /// `set_B_o(B, scl)`.
    pub fn set_b_o(&mut self, b: &[T], scl: &[T]) {
        self.b_o.copy_from_slice(b);
        self.scl_o.copy_from_slice(scl);
    }

    /// `set_E(k, E_l, E_r, scl)` with a 0-based `k`.
    pub fn set_e(&mut self, k: usize, e_l: &[T], e_r: &[T], scl: T::Ext) {
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
                self.b_i[i + c * n_i] = self.b_i[i + c * n_i].div_r(scl);
            }
            self.scl_i[i] = self.scl_i[i].mul(T::from_r(scl));
        }
        for k in 0..self.n {
            for i in 0..n_e {
                let scl = fmax(
                    maxabs((0..n_e).map(|c| self.e_l[k][i + c * n_e])),
                    maxabs((0..n_e).map(|c| self.e_r[k][i + c * n_e])),
                );
                for c in 0..n_e {
                    self.e_l[k][i + c * n_e] =
                        self.e_l[k][i + c * n_e].div_r(scl);
                    self.e_r[k][i + c * n_e] =
                        self.e_r[k][i + c * n_e].div_r(scl);
                }
                self.scl[k] = T::ext_mul(self.scl[k], T::ext_r(scl));
            }
        }
        for i in 0..n_o {
            let scl = maxabs((0..n_e).map(|c| self.b_o[i + c * n_o]));
            for c in 0..n_e {
                self.b_o[i + c * n_o] = self.b_o[i + c * n_o].div_r(scl);
            }
            self.scl_o[i] = self.scl_o[i].mul(T::from_r(scl));
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
                let mut m_g = vec![T::zero(); ld * n_e];
                let mut m_u = vec![T::zero(); ld * n_e];
                let mut m_e = vec![T::zero(); ld * n_e];
                for c in 0..n_e {
                    for r in 0..n_e {
                        m_g[r + c * ld] = self.e_l[k0][r + c * n_e];
                        m_g[n_e + r + c * ld] = T::zero();
                        m_u[r + c * ld] = self.e_r[k0][r + c * n_e];
                        m_u[n_e + r + c * ld] = self.e_l[kl][r + c * n_e];
                        m_e[r + c * ld] = T::zero();
                        m_e[n_e + r + c * ld] = self.e_r[kl][r + c * n_e];
                    }
                }
                let mut ipiv = vec![0usize; n_e];
                let info = T::getrf(ld, n_e, &mut m_u, ld, &mut ipiv)?;
                // `@:ASSERT(info >= 0)` always holds for a returned INFO.
                let _ = info;
                for mm in [&mut m_e, &mut m_g] {
                    T::laswp(n_e, mm, ld, 1, n_e, &ipiv);
                    T::trsm_l(b'L', b'U', n_e, n_e, &m_u, ld, mm, ld)?;
                    // XGEMM_inplace('N','N', n_e, n_e, n_e, -1, M_U(n_e+1,1), 2n_e,
                    //               M(1,1), 2n_e, 1, M(n_e+1,1), 2n_e)
                    let b_top: Vec<T> = mm.clone();
                    T::gemm_nn(
                        n_e,
                        n_e,
                        n_e,
                        T::one().neg(),
                        &m_u[n_e..],
                        ld,
                        &b_top,
                        ld,
                        T::one(),
                        &mut mm[n_e..],
                        ld,
                    )?;
                    T::trsm_l(b'U', b'N', n_e, n_e, &m_u, ld, mm, ld)?;
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
                // (PRODUCT starts from 1 and multiplies in index order.)
                let mut prod = T::one();
                for i in 0..n_e {
                    prod = prod.mul(m_u[i + i * ld]);
                }
                self.scl[k0] = T::ext_mul(self.scl[k0], prod.ext());
                for (i, &p) in ipiv.iter().enumerate() {
                    if p != i + 1 {
                        self.scl[k0] = T::ext_neg(self.scl[k0]);
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
    fn boundary_matrix(&self) -> Vec<T> {
        let (n_e, n_i, n_o) = (self.n_e, self.n_i, self.n_o);
        let ld = 2 * n_e;
        let mut m = vec![T::zero(); ld * ld];
        for c in 0..n_e {
            for r in 0..n_i {
                m[r + c * ld] = self.b_i[r + c * n_i];
            }
            for r in 0..n_e {
                m[n_i + r + c * ld] = self.e_l[0][r + c * n_e];
            }
            for r in n_i + n_e..ld {
                m[r + c * ld] = T::zero();
            }
            for r in 0..n_i {
                m[r + (n_e + c) * ld] = T::zero();
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
    pub fn det(&self) -> Result<T::Ext, GyreError> {
        let n_e = self.n_e;
        let ld = 2 * n_e;
        let mut m = self.boundary_matrix();
        let mut ipiv = vec![0usize; ld];
        let _info = T::getrf(ld, ld, &mut m, ld, &mut ipiv)?;
        let mut parts: Vec<T::Ext> =
            (0..ld).map(|i| m[i + i * ld].ext()).collect();
        parts.extend(self.scl_i.iter().map(|&s| s.ext()));
        parts.extend(self.scl.iter().copied());
        parts.extend(self.scl_o.iter().map(|&s| s.ext()));
        let mut det = T::ext_product(&parts);
        for (i, &p) in ipiv.iter().enumerate() {
            if p != i + 1 {
                det = T::ext_neg(det);
            }
        }
        Ok(det)
    }

    /// `soln_vec_hom()` (`block_sysmtx_m.fypp:402-496`) with `USE_SVD`.
    ///
    /// # Errors
    ///
    /// LAPACK errors and a non-zero `XGESVD` return (`ASSERT`).
    pub fn soln_vec_hom(&self) -> Result<Vec<T>, GyreError> {
        let n_e = self.n_e;
        let ld = 2 * n_e;
        let mut m = self.boundary_matrix();
        let (sv_h, info) = T::sing_decompose_vh(ld, &mut m)?;
        if info != 0 {
            return Err(info_err("XGESVD", info));
        }
        // u_bound = sV_H(2*n_e,:) (real) or CONJG(sV_H(2*n_e,:)) (complex)
        let u_bound: Vec<T> =
            (0..ld).map(|c| sv_h[(ld - 1) + c * ld].conj()).collect();
        self.backsub(&u_bound)
    }

    /// `backsub_(u_bound, u)` (`block_sysmtx_m.fypp:564-625`).
    fn backsub(&self, u_bound: &[T]) -> Result<Vec<T>, GyreError> {
        let (n, n_e) = (self.n, self.n_e);
        let mut u = vec![T::zero(); n_e * (n + 1)];
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
                    *v = T::zero();
                }
                let x_a: Vec<T> = u[i_a..i_a + n_e].to_vec();
                let x_c: Vec<T> = u[i_c..i_c + n_e].to_vec();
                let (e_l, e_r) = (&self.e_l[k - 1 + l], &self.e_r[k - 1 + l]);
                let y = &mut u[i_b..i_b + n_e];
                // XGEMV_inplace('N', n_e, n_e, -1, E_l(1,1,k+l), n_e, u(i_a), 1, 0, u(i_b), 1)
                let (one, zero) = (T::one(), T::zero());
                T::gemv_n(n_e, n_e, one.neg(), e_l, n_e, &x_a, zero, y)?;
                T::gemv_n(n_e, n_e, one.neg(), e_r, n_e, &x_c, one, y)?;
                k += 2 * l;
            }
        }
    }
}
