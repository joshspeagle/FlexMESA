// SPDX-License-Identifier: LGPL-3.0-or-later
//! 4D Stellar Atmosphere Spectral Energy Distribution (SED) cube representation
//! and built-in Kurucz2003 + Johnson loader
//! (`colors/private/colors_utils.f90`).
//!
//! Stores the atmosphere parameter axes $(T_{\mathrm{eff}}, \log g,
//! [\mathrm{M/H}])$, the canonical wavelength grid $\lambda$ (`1199` points in
//! $[147.2, 1.6 \times 10^6]\,\mathrm{\AA}$), the surface flux spectra
//! $F_\lambda(T_{\mathrm{eff}}, \log g, [\mathrm{M/H}], \lambda)$, and the
//! 7 Johnson filter transmission curves pre-interpolated onto $\lambda$ along
//! with their photon-counting denominator integrals $\int S_\lambda \lambda\,
//! d\lambda$.

use std::sync::OnceLock;

use flexmesa_autodiff::DualOrder1;
use rayon::prelude::*;

use crate::filters_data::{
    JOHNSON_FILTERS, NUM_COLORS_OUT, NUM_JOHNSON_FILTERS,
};
use crate::types::{ColorsError, MagSystem, SedInterpMethod};
use crate::utils::{interpolate_array, simpson_integration_fn};

/// Raw bytes of the built-in `Kurucz2003all` sparse SED cube asset.
const BUILTIN_KURUCZ_BYTES: &[u8] =
    include_bytes!("../data/builtin_kurucz_johnson.bin");

/// Return type of [`SedCube::precompute_johnson_filters`].
///
/// Bundles the 7 resampled filter transmission vectors, their
/// wavelength-weighted counterparts $S_\lambda(\lambda_i)\,\lambda_i$, their
/// photon-counting normalization integrals $\int S_\lambda \lambda\,d\lambda$,
/// and the precomputed 3-point non-uniform Simpson panel weights
/// `[c0, c1, c2]` along the canonical SED wavelength grid.
type PrecomputedFilters = (
    [Vec<f64>; NUM_JOHNSON_FILTERS],
    [Vec<f64>; NUM_JOHNSON_FILTERS],
    [f64; NUM_JOHNSON_FILTERS],
    Vec<[f64; 3]>,
);

/// 4D stellar atmosphere SED lookup cube over
/// $(T_{\mathrm{eff}}, \log g, [\mathrm{M/H}], \lambda)$.
#[derive(Debug, Clone)]
pub struct SedCube {
    /// Effective temperature axis $T_{\mathrm{eff}}$ in $\mathrm{K}$ (`nt`).
    pub teff_grid: Vec<f64>,
    /// Log10 surface gravity axis $\log_{10}(g / \mathrm{cm\,s^{-2}})$ (`ng`).
    pub logg_grid: Vec<f64>,
    /// Metallicity axis $[\mathrm{M/H}]$ in dex (`nm`).
    pub meta_grid: Vec<f64>,
    /// Canonical wavelength grid $\lambda$ in $\mathrm{\AA}$ (`n_lambda`).
    pub wavelengths: Vec<f64>,
    /// Mapping from `(it * ng + ig) * nm + im` to SED slot index in
    /// `sed_table`.
    pub slot_map: Vec<usize>,
    /// Flattened SED flux table of shape `(n_slots, n_lambda)`.
    pub sed_table: Vec<f64>,
    /// Johnson `B, I, J, M, R, U, V` filter transmission curves resampled onto
    /// `wavelengths`, each of length `n_lambda`.
    pub filters_on_sed_grid: [Vec<f64>; NUM_JOHNSON_FILTERS],
    /// Precomputed wavelength-weighted filter curves $S_\lambda(\lambda_i)\,
    /// \lambda_i$ for each Johnson filter, each of length `n_lambda`.
    pub filters_weighted_on_sed_grid: [Vec<f64>; NUM_JOHNSON_FILTERS],
    /// Precomputed photon-counting denominator
    /// $\int S_\lambda(\lambda)\,\lambda\,d\lambda$ for each Johnson filter.
    pub filter_norm_integrals: [f64; NUM_JOHNSON_FILTERS],
    /// Precomputed non-uniform Simpson panel weights `[c0, c1, c2]` for each
    /// consecutive 3-node panel `2 * p ..= 2 * p + 2` along `wavelengths`.
    pub simpson_panels: Vec<[f64; 3]>,
}

impl SedCube {
    /// Parses the built-in `Kurucz2003all` SED cube and pre-interpolates the
    /// 7 Johnson filter transmission curves onto its wavelength grid.
    ///
    /// # Errors
    /// Returns [`ColorsError::BinaryParseError`] if the embedded binary payload
    /// is truncated or malformed.
    pub fn from_builtin() -> Result<Self, ColorsError> {
        let bytes = BUILTIN_KURUCZ_BYTES;
        if bytes.len() < 20 {
            return Err(ColorsError::BinaryParseError("header too short"));
        }
        let mut pos = 0usize;
        let read_u32 = |pos: &mut usize| -> Result<usize, ColorsError> {
            let end = *pos + 4;
            let b: [u8; 4] = bytes
                .get(*pos..end)
                .ok_or(ColorsError::BinaryParseError("unexpected EOF (u32)"))?
                .try_into()
                .map_err(|_| ColorsError::BinaryParseError("u32 slice"))?;
            *pos = end;
            Ok(u32::from_le_bytes(b) as usize)
        };
        let read_f64_vec =
            |pos: &mut usize, count: usize| -> Result<Vec<f64>, ColorsError> {
                let byte_len = count * 8;
                let end = *pos + byte_len;
                let slice = bytes.get(*pos..end).ok_or(
                    ColorsError::BinaryParseError("unexpected EOF (f64 vec)"),
                )?;
                *pos = end;
                Ok(slice
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|&c| f64::from_le_bytes(c))
                    .collect())
            };

        let nt = read_u32(&mut pos)?;
        let ng = read_u32(&mut pos)?;
        let nm = read_u32(&mut pos)?;
        let nlam = read_u32(&mut pos)?;
        let n_nodes = read_u32(&mut pos)?;

        let teff_grid = read_f64_vec(&mut pos, nt)?;
        let logg_grid = read_f64_vec(&mut pos, ng)?;
        let meta_grid = read_f64_vec(&mut pos, nm)?;
        let wavelengths = read_f64_vec(&mut pos, nlam)?;

        let mut sed_table = Vec::with_capacity(n_nodes * nlam);
        let mut node_coords = Vec::with_capacity(n_nodes);
        let mut exact_map = vec![usize::MAX; nt * ng * nm];

        for slot in 0..n_nodes {
            let it = read_u32(&mut pos)?;
            let ig = read_u32(&mut pos)?;
            let im = read_u32(&mut pos)?;
            let sed = read_f64_vec(&mut pos, nlam)?;
            sed_table.extend_from_slice(&sed);
            node_coords.push((it, ig, im));
            exact_map[(it * ng + ig) * nm + im] = slot;
        }

        // Populate `slot_map` for all `(it, ig, im)` cells: exact slot if
        // present in the sparse table, else nearest stored node in index space.
        let mut slot_map = vec![0usize; nt * ng * nm];
        for it in 0..nt {
            for ig in 0..ng {
                for im in 0..nm {
                    let flat = (it * ng + ig) * nm + im;
                    if exact_map[flat] == usize::MAX {
                        let best = node_coords
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, &(nt_i, ng_i, nm_i))| {
                                let dt = it.abs_diff(nt_i);
                                let dg = ig.abs_diff(ng_i);
                                let dm = im.abs_diff(nm_i);
                                dt * 100 + dg * 10 + dm
                            })
                            .map_or(0, |(idx, _)| idx);
                        slot_map[flat] = best;
                    } else {
                        slot_map[flat] = exact_map[flat];
                    }
                }
            }
        }

        let (
            filters_on_sed_grid,
            filters_weighted_on_sed_grid,
            filter_norm_integrals,
            simpson_panels,
        ) = Self::precompute_johnson_filters(&wavelengths)?;

        Ok(Self {
            teff_grid,
            logg_grid,
            meta_grid,
            wavelengths,
            slot_map,
            sed_table,
            filters_on_sed_grid,
            filters_weighted_on_sed_grid,
            filter_norm_integrals,
            simpson_panels,
        })
    }

    /// Constructs a dense [`SedCube`] from explicit coordinate grids and a
    /// flattened row-major flux array of shape `(nt, ng, nm, n_lambda)`.
    ///
    /// # Errors
    /// Returns [`ColorsError::InvalidArrayLength`] if `flux.len()` does not
    /// equal `nt * ng * nm * n_lambda`.
    pub fn from_dense(
        teff_grid: Vec<f64>,
        logg_grid: Vec<f64>,
        meta_grid: Vec<f64>,
        wavelengths: Vec<f64>,
        flux: Vec<f64>,
    ) -> Result<Self, ColorsError> {
        let nt = teff_grid.len();
        let ng = logg_grid.len();
        let nm = meta_grid.len();
        let nlam = wavelengths.len();
        let expected = nt * ng * nm * nlam;
        if flux.len() != expected {
            return Err(ColorsError::InvalidArrayLength {
                context: "SedCube::from_dense",
                got: flux.len(),
                expected,
            });
        }
        let slot_map: Vec<usize> = (0..(nt * ng * nm)).collect();
        let (
            filters_on_sed_grid,
            filters_weighted_on_sed_grid,
            filter_norm_integrals,
            simpson_panels,
        ) = Self::precompute_johnson_filters(&wavelengths)?;
        Ok(Self {
            teff_grid,
            logg_grid,
            meta_grid,
            wavelengths,
            slot_map,
            sed_table: flux,
            filters_on_sed_grid,
            filters_weighted_on_sed_grid,
            filter_norm_integrals,
            simpson_panels,
        })
    }

    /// Precomputes the 7 Johnson filter transmission curves on `wavelengths`,
    /// their $\lambda$-weighted counterparts, their photon-counting
    /// normalization integrals $\int S_\lambda(\lambda)\,\lambda\,d\lambda$,
    /// and the non-uniform Simpson panel weights `[c0, c1, c2]`.
    fn precompute_johnson_filters(
        wavelengths: &[f64],
    ) -> Result<PrecomputedFilters, ColorsError> {
        let mut on_grid: [Vec<f64>; NUM_JOHNSON_FILTERS] =
            core::array::from_fn(|_| Vec::new());
        let mut weighted_on_grid: [Vec<f64>; NUM_JOHNSON_FILTERS] =
            core::array::from_fn(|_| Vec::new());
        let mut norms = [0.0_f64; NUM_JOHNSON_FILTERS];
        for (idx, filt) in JOHNSON_FILTERS.iter().enumerate() {
            let f_interp = interpolate_array(
                filt.wavelengths,
                filt.transmission,
                wavelengths,
            )?;
            // Pre-multiply transmission by wavelength for photon-counting
            // synthetic photometry and precompute the denominator integral.
            let fw: Vec<f64> = f_interp
                .iter()
                .zip(wavelengths.iter())
                .map(|(&tr, &w)| tr * w)
                .collect();
            norms[idx] =
                simpson_integration_fn(wavelengths, fw.len(), |i| fw[i])?;
            on_grid[idx] = f_interp;
            weighted_on_grid[idx] = fw;
        }
        // Precompute non-uniform Simpson panel weights `[c0, c1, c2]` once
        // along the fixed canonical wavelength grid.
        let n = wavelengths.len();
        let mut panels = Vec::with_capacity(n / 2);
        let mut i = 0usize;
        while i + 2 < n {
            let h1 = wavelengths[i + 1] - wavelengths[i];
            let h2 = wavelengths[i + 2] - wavelengths[i + 1];
            let h12 = h1 + h2;
            let w_outer = h12 / 6.0;
            panels.push([
                w_outer * (2.0 * h1 - h2) / h1,
                w_outer * (h12 * h12) / (h1 * h2),
                w_outer * (2.0 * h2 - h1) / h2,
            ]);
            i += 2;
        }
        Ok((on_grid, weighted_on_grid, norms, panels))
    }

    /// Returns the surface flux spectrum $F_\lambda(\lambda)$ of length
    /// `self.wavelengths.len()` at grid indices `(it, ig, im)`.
    #[inline]
    #[must_use]
    pub fn sed_at(&self, it: usize, ig: usize, im: usize) -> &[f64] {
        let ng = self.logg_grid.len();
        let nm = self.meta_grid.len();
        let nlam = self.wavelengths.len();
        let slot = self.slot_map[(it * ng + ig) * nm + im];
        let start = slot * nlam;
        &self.sed_table[start..start + nlam]
    }

    /// Evaluates `eval_colors_generic` with `DualOrder1<4>` seeded on
    /// `(teff, log_g, metallicity, r)` at fixed distance `d`.
    ///
    /// # Errors
    /// Returns [`ColorsError`] if quadrature fails.
    pub fn eval_colors_dual(
        &self,
        teff: f64,
        log_g: f64,
        metallicity: f64,
        r: f64,
        d: f64,
        mag_system: MagSystem,
    ) -> Result<[DualOrder1<4>; NUM_COLORS_OUT], ColorsError> {
        let t_d = DualOrder1::<4>::independent(teff, 0);
        let g_d = DualOrder1::<4>::independent(log_g, 1);
        let m_d = DualOrder1::<4>::independent(metallicity, 2);
        let r_d = DualOrder1::<4>::independent(r, 3);
        let d_d = DualOrder1::<4>::constant(d);
        self.eval_colors_generic(t_d, g_d, m_d, r_d, d_d, mag_system)
    }

    /// Evaluates `eval_colors` across multiple stellar models/zones in parallel
    /// via `rayon`.
    ///
    /// # Errors
    /// Returns [`ColorsError`] if input slice lengths differ or quadrature
    /// fails.
    pub fn eval_colors_zones(
        &self,
        teff: &[f64],
        log_g: &[f64],
        metallicity: &[f64],
        r: &[f64],
        d: f64,
        mag_system: MagSystem,
        method: SedInterpMethod,
    ) -> Result<Vec<[f64; NUM_COLORS_OUT]>, ColorsError> {
        let n = teff.len();
        if log_g.len() != n || metallicity.len() != n || r.len() != n {
            let got = if log_g.len() != n {
                log_g.len()
            } else if metallicity.len() != n {
                metallicity.len()
            } else {
                r.len()
            };
            return Err(ColorsError::InvalidArrayLength {
                context: "eval_colors_zones",
                got,
                expected: n,
            });
        }
        (0..n)
            .into_par_iter()
            .map(|idx| {
                self.eval_colors(
                    teff[idx],
                    log_g[idx],
                    metallicity[idx],
                    r[idx],
                    d,
                    mag_system,
                    method,
                )
            })
            .collect()
    }
}

/// Lazily initialized singleton for the built-in `Kurucz2003all` + `Johnson`
/// [`SedCube`].
///
/// # Panics
/// Panics only if the compile-time embedded `builtin_kurucz_johnson.bin` asset
/// is corrupted.
#[must_use]
pub fn builtin_sed_cube() -> &'static SedCube {
    static CUBE: OnceLock<SedCube> = OnceLock::new();
    CUBE.get_or_init(|| {
        SedCube::from_builtin()
            .expect("built-in Kurucz2003 SED cube must parse")
    })
}
