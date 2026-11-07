// SPDX-License-Identifier: LGPL-3.0-or-later
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
    T::from_f64(2.0) * t2 * t - T::from_f64(3.0) * t2 + T::one()
}

/// Cubic Hermite tangent basis $h_{10}(t) = t^3 - 2t^2 + t$ at the left knot
/// ($h_{10}(0) = h_{10}(1) = 0$, $h_{10}'(0) = 1$, $h_{10}'(1) = 0$).
#[inline]
#[must_use]
pub fn h10<T: RealScalar>(t: T) -> T {
    let t2 = t * t;
    t2 * t - T::from_f64(2.0) * t2 + t
}

/// Cubic Hermite value basis $h_{01}(t) = -2t^3 + 3t^2$ at the right knot
/// ($h_{01}(0) = 0$, $h_{01}(1) = 1$, $h_{01}'(0) = h_{01}'(1) = 0$).
#[inline]
#[must_use]
pub fn h01<T: RealScalar>(t: T) -> T {
    let t2 = t * t;
    T::from_f64(3.0) * t2 - T::from_f64(2.0) * t2 * t
}

/// Cubic Hermite tangent basis $h_{11}(t) = t^3 - t^2$ at the right knot
/// ($h_{11}(0) = h_{11}(1) = 0$, $h_{11}'(0) = 0$, $h_{11}'(1) = 1$).
#[inline]
#[must_use]
pub fn h11<T: RealScalar>(t: T) -> T {
    let t2 = t * t;
    t2 * t - t2
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
    /// without heap allocation.
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
        let (ix, hx, gx, ext_x) =
            Self::axis_hermite_stencil(&self.teff_grid, teff);
        let (iy, hy, gy, ext_y) =
            Self::axis_hermite_stencil(&self.logg_grid, log_g);
        let (iz, hz, gz, ext_z) =
            Self::axis_hermite_stencil(&self.meta_grid, metallicity);
        let sgn = [T::from_f64(-1.0), T::from_f64(1.0)];

        for kz in 0..2 {
            let (wz, k, k_ext) = (hz[kz], iz + kz, ext_z[kz]);
            for ky in 0..2 {
                let (wy, j, j_ext) = (hy[ky], iy + ky, ext_y[ky]);
                let wy_wz = wy * wz;
                let yz_tan =
                    sgn[ky] * gy[1 - ky] * wz + sgn[kz] * gz[1 - kz] * wy;
                for kx in 0..2 {
                    let (wx, i, i_ext) = (hx[kx], ix + kx, ext_x[kx]);
                    let w_corner =
                        (wx + sgn[kx] * gx[1 - kx]) * wy_wz + wx * yz_tan;
                    let w_x = sgn[kx] * gx[kx] * wy_wz;
                    let w_y = sgn[ky] * wx * gy[ky] * wz;
                    let w_z = sgn[kz] * wx * wy * gz[kz];

                    let (f_ijk, f_x) =
                        (self.sed_at(i, j, k), self.sed_at(i_ext, j, k));
                    let (f_y, f_z) =
                        (self.sed_at(i, j_ext, k), self.sed_at(i, j, k_ext));

                    for lam in 0..nlam {
                        let c = w_corner * T::from_f64(f_ijk[lam])
                            + w_x * T::from_f64(f_x[lam])
                            + w_y * T::from_f64(f_y[lam])
                            + w_z * T::from_f64(f_z[lam]);
                        result[lam] = result[lam] + c;
                    }
                }
            }
        }

        let scale = (r / d) * (r / d);
        for v in &mut result[..nlam] {
            *v = *v * scale;
        }
    }

    /// Computes `(idx0, [h00, h01], [gx0, gx1], [im0, ip1])` along `grid`.
    #[inline]
    fn axis_hermite_stencil<T: RealScalar>(
        grid: &[f64],
        val: T,
    ) -> (usize, [T; 2], [T; 2], [usize; 2]) {
        let n = grid.len();
        let (idx, t) = find_interval(grid, val);
        let dx = grid[idx + 1] - grid[idx];
        let s0 = Self::diff_stencil(idx, n, grid);
        let s1 = Self::diff_stencil(idx + 1, n, grid);
        (
            idx,
            [h00(t), h01(t)],
            [
                h10(t) * T::from_f64(dx * s0.2),
                h11(t) * T::from_f64(dx * s1.2),
            ],
            [s0.0, s1.1],
        )
    }

    /// Returns `(i_minus, i_plus, 1.0 / (grid[i_plus] - grid[i_minus]))` for
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
        let span = grid[ip] - grid[im];
        (im, ip, if span.abs() > 0.0 { 1.0 / span } else { 0.0 })
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

        let rows = [
            self.sed_at(ix, iy, iz),
            self.sed_at(ix, iy, iz + 1),
            self.sed_at(ix, iy + 1, iz),
            self.sed_at(ix, iy + 1, iz + 1),
            self.sed_at(ix + 1, iy, iz),
            self.sed_at(ix + 1, iy, iz + 1),
            self.sed_at(ix + 1, iy + 1, iz),
            self.sed_at(ix + 1, iy + 1, iz + 1),
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
                let log_v = blend(corners.map(f64::ln));
                if log_v.is_finite() {
                    val = log_v.exp();
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
    /// inverse-distance-weighted interpolation (`knn_interp.f90`).
    ///
    /// Normalizes $(T_{\mathrm{eff}}, \log g, [\mathrm{M/H}])$ to $[0, 1]^3$,
    /// locates the 4 globally closest grid nodes, and blends their SEDs with
    /// weights $w_k \propto 1 / \max(d_k, 10^{-10})$.
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
        let norm_axis = |grid: &[f64]| -> (f64, f64) {
            if is_dummy_axis(grid) {
                return (0.0, 0.0);
            }
            let min_v = grid.iter().copied().fold(f64::INFINITY, f64::min);
            let max_v = grid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let span = max_v - min_v;
            (min_v, if span > 0.0 { 1.0 / span } else { 0.0 })
        };
        let (t_min, t_inv) = norm_axis(&self.teff_grid);
        let (g_min, g_inv) = norm_axis(&self.logg_grid);
        let (m_min, m_inv) = norm_axis(&self.meta_grid);
        let (nt_v, ng_v, nm_v) = (
            (teff - t_min) * t_inv,
            (log_g - g_min) * g_inv,
            (metallicity - m_min) * m_inv,
        );

        let dg_sq: Vec<f64> = self
            .logg_grid
            .iter()
            .map(|&gv| ((gv - g_min) * g_inv - ng_v).powi(2))
            .collect();
        let dm_sq: Vec<f64> = self
            .meta_grid
            .iter()
            .map(|&mv| ((mv - m_min) * m_inv - nm_v).powi(2))
            .collect();

        let mut best_d2 = [f64::INFINITY; 4];
        let mut best_idx = [(0usize, 0usize, 0usize); 4];
        for (it, &tv) in self.teff_grid.iter().enumerate() {
            let dt = ((tv - t_min) * t_inv - nt_v).powi(2);
            for (ig, &dg) in dg_sq.iter().enumerate() {
                let dt_dg = dt + dg;
                for (im, &dm) in dm_sq.iter().enumerate() {
                    let idx = (it, ig, im);
                    insert_top4(&mut best_d2, &mut best_idx, dt_dg + dm, idx);
                }
            }
        }

        let inv_w = best_d2.map(|d2| 1.0 / d2.sqrt().max(1.0e-10));
        let w_sum: f64 = inv_w.iter().sum();
        let scale = (r / d) * (r / d) / w_sum;
        let nlam = self.wavelengths.len();
        let mut interp = vec![0.0_f64; nlam];
        for k in 0..4 {
            let (it, ig, im) = best_idx[k];
            let sed = self.sed_at(it, ig, im);
            let wk = inv_w[k] * scale;
            for lam in 0..nlam {
                interp[lam] += wk * sed[lam];
            }
        }
        interp
    }
}
