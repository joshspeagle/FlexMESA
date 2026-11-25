// SPDX-License-Identifier: GPL-3.0-or-later
//! 3D/4D Spectral Energy Distribution (SED) interpolation kernels
//! (`colors/private/hermite_interp.f90`, `linear_interp.f90`,
//! `knn_interp.f90`).
//!
//! Implements the three 3D stellar atmosphere SED interpolators supported by
//! upstream MESA `colors`:
//! 1. **Cubic Hermite tensor interpolation (`eval_sed_hermite_generic`)**:
//!    Evaluates the 8-vertex tensor product of the 1D cubic Hermite value
//!    ($h_{00}, h_{01}$) and tangent ($h_{10}, h_{11}$) polynomials using
//!    centered/one-sided finite-difference slopes along $(T_{\mathrm{eff}},
//!    \log g, [\mathrm{M/H}])$. Generic over [`RealScalar`] (`f64` and
//!    `DualOrder1<N>`) for exact forward-mode automatic differentiation.
//! 2. **Log-fallback trilinear interpolation (`eval_sed_trilinear`)**:
//!    Floors the 8 cell corners at $10^{-10}$, blends in $\ln F_\lambda$ space
//!    whenever the linear blend exceeds $10^{-10}$, and falls back to the
//!    nearest grid node if any non-finite value occurs.
//! 3. **4-nearest-neighbor inverse-distance interpolation (`eval_sed_knn`)**:
//!    Normalizes $(T_{\mathrm{eff}}, \log g, [\mathrm{M/H}])$ to $[0, 1]^3$,
//!    locates the 4 globally closest grid nodes, and blends their SEDs with
//!    inverse-distance weights $w_k \propto 1 / \max(d_k, 10^{-10})$.

use crate::sed_cube::SedCube;
use crate::types::{RealScalar, SedInterpMethod};
use crate::utils::{find_interval, find_nearest_point, is_dummy_axis};

/// Cubic Hermite value basis $h_{00}(t) = 2t^3 - 3t^2 + 1$ at the left knot
/// ($h_{00}(0) = 1$, $h_{00}(1) = 0$, $h_{00}'(0) = h_{00}'(1) = 0$).
#[inline]
#[must_use]
pub fn h00<T: RealScalar>(t: T) -> T {
    let t2 = t * t;
    let t3 = t2 * t;
    T::from_f64(2.0) * t3 - T::from_f64(3.0) * t2 + T::one()
}

/// Cubic Hermite tangent basis $h_{10}(t) = t^3 - 2t^2 + t$ at the left knot
/// ($h_{10}(0) = h_{10}(1) = 0$, $h_{10}'(0) = 1$, $h_{10}'(1) = 0$).
#[inline]
#[must_use]
pub fn h10<T: RealScalar>(t: T) -> T {
    let t2 = t * t;
    let t3 = t2 * t;
    t3 - T::from_f64(2.0) * t2 + t
}

/// Cubic Hermite value basis $h_{01}(t) = -2t^3 + 3t^2$ at the right knot
/// ($h_{01}(0) = 0$, $h_{01}(1) = 1$, $h_{01}'(0) = h_{01}'(1) = 0$).
#[inline]
#[must_use]
pub fn h01<T: RealScalar>(t: T) -> T {
    let t2 = t * t;
    let t3 = t2 * t;
    T::from_f64(-2.0) * t3 + T::from_f64(3.0) * t2
}

/// Cubic Hermite tangent basis $h_{11}(t) = t^3 - t^2$ at the right knot
/// ($h_{11}(0) = h_{11}(1) = 0$, $h_{11}'(0) = 0$, $h_{11}'(1) = 1$).
#[inline]
#[must_use]
pub fn h11<T: RealScalar>(t: T) -> T {
    let t2 = t * t;
    let t3 = t2 * t;
    t3 - t2
}

/// Inserts candidate `(dist2, idx)` into the sorted top-4 nearest-neighbor
/// buffers `(best_d2, best_idx)` in ascending order of squared distance.
#[inline]
fn insert_top4(
    best_d2: &mut [f64; 4],
    best_idx: &mut [(usize, usize, usize); 4],
    dist2: f64,
    idx: (usize, usize, usize),
) {
    for k in 0..4 {
        if dist2 < best_d2[k] {
            for s in ((k + 1)..4).rev() {
                best_d2[s] = best_d2[s - 1];
                best_idx[s] = best_idx[s - 1];
            }
            best_d2[k] = dist2;
            best_idx[k] = idx;
            break;
        }
    }
}

impl SedCube {
    /// Dispatches 3D SED interpolation across [`SedInterpMethod`] (`Hermite`,
    /// `Linear`, or `Knn`) and applies geometric dilution $(R / d)^2$.
    ///
    /// # Arguments
    /// * `teff` - Effective temperature $T_{\mathrm{eff}}$ in $\mathrm{K}$.
    /// * `log_g` - Surface gravity $\log_{10}(g / \mathrm{cm\,s^{-2}})$.
    /// * `metallicity` - Metallicity $[\mathrm{M/H}]$ in dex.
    /// * `r` - Stellar photospheric radius $R$ in $\mathrm{cm}$.
    /// * `d` - Observer distance $d$ in $\mathrm{cm}$.
    /// * `method` - Interpolation algorithm selector.
    #[must_use]
    pub fn eval_sed(
        &self,
        teff: f64,
        log_g: f64,
        metallicity: f64,
        r: f64,
        d: f64,
        method: SedInterpMethod,
    ) -> Vec<f64> {
        match method {
            SedInterpMethod::Hermite => {
                self.eval_sed_hermite_generic(teff, log_g, metallicity, r, d)
            }
            SedInterpMethod::Linear => {
                self.eval_sed_trilinear(teff, log_g, metallicity, r, d)
            }
            SedInterpMethod::Knn => {
                self.eval_sed_knn(teff, log_g, metallicity, r, d)
            }
        }
    }

    /// Evaluates the diluted observed SED $F_\lambda(\lambda) =
    /// F_{\mathrm{surf}}(\lambda)\,(R / d)^2$ using 3D cubic Hermite tensor
    /// interpolation over $(T_{\mathrm{eff}}, \log g, [\mathrm{M/H}])$
    /// (`hermite_interp.f90:construct_sed_hermite`).
    ///
    /// Generic over [`RealScalar`] (`f64` and `DualOrder1<N>`), propagating
    /// exact analytical derivatives through both the cubic Hermite basis
    /// weights and the geometric dilution factor $(R / d)^2$.
    ///
    /// # Arguments
    /// * `teff` - Effective temperature $T_{\mathrm{eff}}$ in $\mathrm{K}$.
    /// * `log_g` - Surface gravity $\log_{10}(g / \mathrm{cm\,s^{-2}})$.
    /// * `metallicity` - Metallicity $[\mathrm{M/H}]$ in dex.
    /// * `r` - Stellar photospheric radius $R$ in $\mathrm{cm}$.
    /// * `d` - Observer distance $d$ in $\mathrm{cm}$.
    #[must_use]
    pub fn eval_sed_hermite_generic<T: RealScalar>(
        &self,
        teff: T,
        log_g: T,
        metallicity: T,
        r: T,
        d: T,
    ) -> Vec<T> {
        let mut result = vec![T::zero(); self.wavelengths.len()];
        self.eval_sed_hermite_into(teff, log_g, metallicity, r, d, &mut result);
        result
    }

    /// Evaluates the diluted 3D cubic Hermite SED directly into `result`
    /// without heap allocation (`hermite_interp.f90:hermite_interp_vector`).
    pub fn eval_sed_hermite_into<T: RealScalar>(
        &self,
        teff: T,
        log_g: T,
        metallicity: T,
        r: T,
        d: T,
        result: &mut [T],
    ) {
        let nlam = self.wavelengths.len();
        result[..nlam].fill(T::zero());
        let (nx, ny, nz) = (
            self.teff_grid.len(),
            self.logg_grid.len(),
            self.meta_grid.len(),
        );
        // Match `colors/private/hermite_interp.f90:203-213`: whenever any
        // axis has fewer than 2 points (`nx < 2 || ny < 2 || nz < 2`), `i >= n`
        // triggers nearest-point lookup across all wavelengths.
        if nx < 2 || ny < 2 || nz < 2 {
            let (ni, nj, nk) = find_nearest_point(
                teff.val(),
                log_g.val(),
                metallicity.val(),
                &self.teff_grid,
                &self.logg_grid,
                &self.meta_grid,
            );
            let sed = self.sed_at(ni, nj, nk);
            let scale = (r / d) * (r / d);
            for (dst, &src) in result[..nlam].iter_mut().zip(sed.iter()) {
                *dst = T::from_f64(src) * scale;
            }
            return;
        }
        let (ix, hx, hxd, dx) = Self::axis_hermite_basis(&self.teff_grid, teff);
        let (iy, hy, hyd, dy) =
            Self::axis_hermite_basis(&self.logg_grid, log_g);
        let (iz, hz, hzd, dz) =
            Self::axis_hermite_basis(&self.meta_grid, metallicity);
        let kx_limit = if is_dummy_axis(&self.teff_grid) { 1 } else { 2 };
        let ky_limit = if is_dummy_axis(&self.logg_grid) { 1 } else { 2 };
        let kz_limit = if is_dummy_axis(&self.meta_grid) { 1 } else { 2 };
        let (dx_t, dy_t, dz_t) =
            (T::from_f64(dx), T::from_f64(dy), T::from_f64(dz));

        for kz in 0..kz_limit {
            let (wz, wzd, k) = (hz[kz], hzd[kz], iz + kz);
            let (km, kp, span_z) = Self::diff_stencil(k, nz, &self.meta_grid);
            for ky in 0..ky_limit {
                let (wy, wyd, j) = (hy[ky], hyd[ky], iy + ky);
                let (jm, jp, span_y) =
                    Self::diff_stencil(j, ny, &self.logg_grid);
                for kx in 0..kx_limit {
                    let (wx, wxd, i) = (hx[kx], hxd[kx], ix + kx);
                    let (im, ip, span_x) =
                        Self::diff_stencil(i, nx, &self.teff_grid);

                    let w_val = (wx * wy) * wz;
                    let w_dx = ((wxd * wy) * wz) * dx_t;
                    let w_dy = ((wx * wyd) * wz) * dy_t;
                    let w_dz = ((wx * wy) * wzd) * dz_t;

                    let f_ijk = self.sed_at(i, j, k);
                    let (f_xm, f_xp) =
                        (self.sed_at(im, j, k), self.sed_at(ip, j, k));
                    let (f_ym, f_yp) =
                        (self.sed_at(i, jm, k), self.sed_at(i, jp, k));
                    let (f_zm, f_zp) =
                        (self.sed_at(i, j, km), self.sed_at(i, j, kp));

                    for lam in 0..nlam {
                        let df_dx = if span_x.abs() > 0.0 {
                            (f_xp[lam] - f_xm[lam]) / span_x
                        } else {
                            0.0
                        };
                        let df_dy = if span_y.abs() > 0.0 {
                            (f_yp[lam] - f_ym[lam]) / span_y
                        } else {
                            0.0
                        };
                        let df_dz = if span_z.abs() > 0.0 {
                            (f_zp[lam] - f_zm[lam]) / span_z
                        } else {
                            0.0
                        };
                        result[lam] = result[lam]
                            + w_val * T::from_f64(f_ijk[lam])
                            + w_dx * T::from_f64(df_dx)
                            + w_dy * T::from_f64(df_dy)
                            + w_dz * T::from_f64(df_dz);
                    }
                }
            }
        }

        let scale = (r / d) * (r / d);
        for v in &mut result[..nlam] {
            *v = *v * scale;
        }
    }

    /// Computes `(idx0, [h00, h01], [h10, h11], dx)` along `grid`.
    #[inline]
    fn axis_hermite_basis<T: RealScalar>(
        grid: &[f64],
        val: T,
    ) -> (usize, [T; 2], [T; 2], f64) {
        if grid.len() < 2 || is_dummy_axis(grid) {
            return (0, [T::one(), T::zero()], [T::zero(), T::zero()], 0.0);
        }
        let (idx, t) = find_interval(grid, val);
        let dx = grid[idx + 1] - grid[idx];
        (idx, [h00(t), h01(t)], [h10(t), h11(t)], dx)
    }

    /// Returns `(i_minus, i_plus, grid[i_plus] - grid[i_minus])` for
    /// 1D finite-difference slope evaluation at grid node `i`
    /// (`hermite_interp.f90:compute_derivatives_at_point_4d`).
    #[inline]
    fn diff_stencil(i: usize, n: usize, grid: &[f64]) -> (usize, usize, f64) {
        let (im, ip) = if n <= 1 {
            return (0, 0, 0.0);
        } else if i > 0 && i + 1 < n {
            (i - 1, i + 1)
        } else if i == 0 {
            (0, 1)
        } else {
            (i - 1, i)
        };
        (im, ip, grid[ip] - grid[im])
    }

    /// Evaluates the diluted observed SED using log-fallback 3D trilinear
    /// interpolation (`linear_interp.f90:trilinear_interp_vector`).
    ///
    /// Clamps all 8 cell corner fluxes to `1.0e-10`, evaluates the trilinear
    /// blend in $\ln F_\lambda$ space when positive, and falls back to the
    /// nearest grid node if any non-finite value occurs.
    #[must_use]
    pub fn eval_sed_trilinear(
        &self,
        teff: f64,
        log_g: f64,
        metallicity: f64,
        r: f64,
        d: f64,
    ) -> Vec<f64> {
        const TINY: f64 = 1.0e-10;
        let nlam = self.wavelengths.len();
        let (ix, tx) = find_interval(&self.teff_grid, teff);
        let (iy, ty) = find_interval(&self.logg_grid, log_g);
        let (iz, tz) = find_interval(&self.meta_grid, metallicity);
        let (tx, ty, tz) =
            (tx.clamp(0.0, 1.0), ty.clamp(0.0, 1.0), tz.clamp(0.0, 1.0));

        let ix1 = (ix + 1).min(self.teff_grid.len().saturating_sub(1));
        let iy1 = (iy + 1).min(self.logg_grid.len().saturating_sub(1));
        let iz1 = (iz + 1).min(self.meta_grid.len().saturating_sub(1));
        let rows = [
            self.sed_at(ix, iy, iz),
            self.sed_at(ix, iy, iz1),
            self.sed_at(ix, iy1, iz),
            self.sed_at(ix, iy1, iz1),
            self.sed_at(ix1, iy, iz),
            self.sed_at(ix1, iy, iz1),
            self.sed_at(ix1, iy1, iz),
            self.sed_at(ix1, iy1, iz1),
        ];

        let scale = (r / d) * (r / d);
        let mut interp = vec![0.0_f64; nlam];
        for lam in 0..nlam {
            let corners = rows.map(|row| row[lam].max(TINY));
            let blend = |v: [f64; 8]| {
                let c00 = v[0] * (1.0 - tx) + v[4] * tx;
                let c01 = v[1] * (1.0 - tx) + v[5] * tx;
                let c10 = v[2] * (1.0 - tx) + v[6] * tx;
                let c11 = v[3] * (1.0 - tx) + v[7] * tx;
                let c0 = c00 * (1.0 - ty) + c10 * ty;
                let c1 = c01 * (1.0 - ty) + c11 * ty;
                c0 * (1.0 - tz) + c1 * tz
            };
            let mut val = blend(corners);
            if val > TINY {
                // `colors/private/linear_interp.f90:237-240,249` omits `use math_lib` (#198).
                let log_v = blend(corners.map(flexmesa_math::intrinsic_log));
                if log_v.is_finite() {
                    val = flexmesa_math::intrinsic_exp(log_v);
                }
            }
            if !val.is_finite() || val <= 0.0 {
                let (nx, ny, nz) = find_nearest_point(
                    teff,
                    log_g,
                    metallicity,
                    &self.teff_grid,
                    &self.logg_grid,
                    &self.meta_grid,
                );
                val = self.sed_at(nx, ny, nz)[lam].max(TINY);
            }
            interp[lam] = val * scale;
        }
        interp
    }

    /// Evaluates the diluted observed SED using 4-nearest-neighbor
    /// inverse-distance-weighted interpolation (`knn_interp.f90:63,91-104,183-276`).
    ///
    /// Normalizes $(T_{\mathrm{eff}}, \log g, [\mathrm{M/H}])$ to $[0, 1]^3$
    /// via direct division, locates the 4 globally closest grid nodes, blends
    /// their surface SEDs with weights $w_k \propto 1 / d_k$ (replacing only
    /// exact $d_k = 0$ with $10^{-10}$), and applies $(R / d)^2$ dilution.
    ///
    /// # Arguments
    /// * `teff` - Effective temperature $T_{\mathrm{eff}}$ in $\mathrm{K}$.
    /// * `log_g` - Surface gravity $\log_{10}(g / \mathrm{cm\,s^{-2}})$.
    /// * `metallicity` - Metallicity $[\mathrm{M/H}]$ in dex.
    /// * `r` - Stellar photospheric radius $R$ in $\mathrm{cm}$.
    /// * `d` - Observer distance $d$ in $\mathrm{cm}$.
    #[must_use]
    pub fn eval_sed_knn(
        &self,
        teff: f64,
        log_g: f64,
        metallicity: f64,
        r: f64,
        d: f64,
    ) -> Vec<f64> {
        let norm_axis = |grid: &[f64]| -> (bool, f64, f64) {
            if is_dummy_axis(grid) {
                return (false, 0.0, 0.0);
            }
            let min_v = grid.iter().copied().fold(f64::INFINITY, f64::min);
            let max_v = grid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (true, min_v, max_v - min_v)
        };
        let norm_coord = |active: bool, min_v: f64, span: f64, v: f64| -> f64 {
            if active && span > 0.0 {
                (v - min_v) / span
            } else {
                0.0
            }
        };
        let (use_t, t_min, t_span) = norm_axis(&self.teff_grid);
        let (use_g, g_min, g_span) = norm_axis(&self.logg_grid);
        let (use_m, m_min, m_span) = norm_axis(&self.meta_grid);
        let nt_v = norm_coord(use_t, t_min, t_span, teff);
        let ng_v = norm_coord(use_g, g_min, g_span, log_g);
        let nm_v = norm_coord(use_m, m_min, m_span, metallicity);

        let dg_sq: Vec<f64> = self
            .logg_grid
            .iter()
            .map(|&gv| {
                if use_g {
                    let sg = norm_coord(true, g_min, g_span, gv);
                    flexmesa_math::pow2(sg - ng_v)
                } else {
                    0.0
                }
            })
            .collect();
        let dm_sq: Vec<f64> = self
            .meta_grid
            .iter()
            .map(|&mv| {
                if use_m {
                    let sm = norm_coord(true, m_min, m_span, mv);
                    flexmesa_math::pow2(sm - nm_v)
                } else {
                    0.0
                }
            })
            .collect();

        let mut best_d2 = [f64::MAX; 4];
        let mut best_idx = [(0usize, 0usize, 0usize); 4];
        for (it, &tv) in self.teff_grid.iter().enumerate() {
            let dt = if use_t {
                let st = norm_coord(true, t_min, t_span, tv);
                flexmesa_math::pow2(st - nt_v)
            } else {
                0.0
            };
            for (ig, &dg) in dg_sq.iter().enumerate() {
                let dt_dg = dt + dg;
                for (im, &dm) in dm_sq.iter().enumerate() {
                    let idx = (it, ig, im);
                    insert_top4(&mut best_d2, &mut best_idx, dt_dg + dm, idx);
                }
            }
        }

        let inv_w = best_d2.map(|d2| {
            let dist = d2.sqrt();
            let d_safe = if dist == 0.0 { 1.0e-10 } else { dist };
            1.0 / d_safe
        });
        let w_sum: f64 = inv_w.iter().sum();
        let weights = inv_w.map(|w| w / w_sum);
        let nlam = self.wavelengths.len();
        let mut interp = vec![0.0_f64; nlam];
        for (&(it, ig, im), &wk) in best_idx.iter().zip(weights.iter()) {
            let sed = self.sed_at(it, ig, im);
            for (dst, &src) in interp.iter_mut().zip(sed.iter()) {
                *dst += wk * src;
            }
        }
        let scale = (r / d) * (r / d);
        for v in &mut interp {
            *v *= scale;
        }
        interp
    }
}
