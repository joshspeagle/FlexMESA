// SPDX-License-Identifier: LGPL-3.0-or-later
//! Bolometric and synthetic filter photometry evaluators
//! (`colors/private/bolometric.f90`, `synthetic.f90`, `colors_history.f90`).
//!
//! Computes:
//! - Bolometric flux $F_{\mathrm{bol}} = \int F_\lambda(\lambda)\,d\lambda$
//!   and magnitude $m_{\mathrm{bol}} = -2.5 \log_{10}(F_{\mathrm{bol}})$.
//! - Photon-counting synthetic filter flux
//!   $F_{\mathrm{syn}} = \int F_\lambda S_\lambda \lambda\,d\lambda /
//!   \int S_\lambda \lambda\,d\lambda$ and magnitude
//!   $m_{\mathrm{band}} = -2.5 \log_{10}(F_{\mathrm{syn}} / F_0)$.
//! - Zero-point fluxes in Vega, AB ($f_\nu = 3631\,\mathrm{Jy}$), and ST
//!   ($f_\lambda = 3.63 \times 10^{-9}$) systems.

use flexmesa_const::CLIGHT;

use crate::filters_data::{
    JOHNSON_FILTERS, NUM_COLORS_OUT, NUM_JOHNSON_FILTERS,
};
use crate::sed_cube::SedCube;
use crate::types::{ColorsError, MagSystem, RealScalar, SedInterpMethod};
use crate::utils::{
    compute_interp_radius, interpolate_array, simpson_integration_fn,
};

/// Clamps negative flux entries at `i < n - 1` or non-positive wavelengths to
/// zero (`bolometric.f90:calculate_bolometric_phot`).
#[inline]
fn cleaned_flux_at<T: RealScalar>(
    wavelengths: &[f64],
    fluxes: &[T],
    i: usize,
) -> T {
    let n = wavelengths.len();
    if i + 1 < n && (wavelengths[i] <= 0.0 || fluxes[i].val() < 0.0) {
        T::zero()
    } else {
        fluxes[i]
    }
}

/// Converts an integrated flux $F > 0$ into a Pogson magnitude
/// $m = -2.5 \log_{10}(F)$, returning `99.0` if $F \le 0$
/// (`bolometric.f90:flux_to_magnitude`).
#[inline]
#[must_use]
pub fn flux_to_magnitude<T: RealScalar>(flux: T) -> T {
    if flux.val() <= 0.0 {
        T::from_f64(99.0)
    } else {
        T::from_f64(-2.5) * flux.log10()
    }
}

/// Integrates bolometric flux $F_{\mathrm{bol}} = \int F_\lambda\,d\lambda$
/// via [`simpson_integration_fn`] and returns `(bol_mag, bol_flux)`
/// (`bolometric.f90:calculate_bolometric_phot`).
///
/// Follows upstream MESA by clamping negative flux entries over
/// `0..(wavelengths.len() - 1)` to zero before integration.
///
/// # Errors
/// Returns [`ColorsError::InvalidArrayLength`] if `wavelengths` and `fluxes`
/// differ in length or have fewer than 2 points.
pub fn calculate_bolometric_phot<T: RealScalar>(
    wavelengths: &[f64],
    fluxes: &[T],
) -> Result<(T, T), ColorsError> {
    let n = wavelengths.len();
    if fluxes.len() != n || n < 2 {
        return Err(ColorsError::InvalidArrayLength {
            context: "calculate_bolometric_phot",
            got: fluxes.len(),
            expected: n.max(2),
        });
    }
    let bol_flux = simpson_integration_fn(wavelengths, n, |i| {
        cleaned_flux_at(wavelengths, fluxes, i)
    })?;
    Ok((flux_to_magnitude(bol_flux), bol_flux))
}

/// Computes the photon-counting synthetic filter flux
/// $F_{\mathrm{syn}} = \int F_{\mathrm{conv}}(\lambda)\,\lambda\,d\lambda /
/// \int S_\lambda(\lambda)\,\lambda\,d\lambda$
/// (`synthetic.f90:calculate_synthetic_flux`).
///
/// # Errors
/// Returns [`ColorsError`] if array lengths are invalid or the integrated
/// filter transmission is non-positive.
pub fn calculate_synthetic_flux<T: RealScalar>(
    wavelengths: &[f64],
    convolved_flux: &[T],
    filter_on_sed_grid: &[f64],
) -> Result<T, ColorsError> {
    let n = wavelengths.len();
    if convolved_flux.len() != n || filter_on_sed_grid.len() != n || n < 2 {
        let got = if convolved_flux.len() == n {
            filter_on_sed_grid.len()
        } else {
            convolved_flux.len()
        };
        return Err(ColorsError::InvalidArrayLength {
            context: "calculate_synthetic_flux",
            got,
            expected: n.max(2),
        });
    }
    let int_flux = simpson_integration_fn(wavelengths, n, |i| {
        convolved_flux[i] * T::from_f64(wavelengths[i])
    })?;
    let int_filt = simpson_integration_fn(wavelengths, n, |i| {
        filter_on_sed_grid[i] * wavelengths[i]
    })?;
    if int_filt <= 0.0 {
        return Err(ColorsError::NonPositiveFlux {
            context: "calculate_synthetic_flux (filter integral)",
            flux: int_filt,
        });
    }
    Ok(int_flux / T::from_f64(int_filt))
}

/// Computes the synthetic filter magnitude
/// $m = -2.5 \log_{10}(F_{\mathrm{syn}} / F_0)$ for a given SED and filter
/// transmission curve (`synthetic.f90:calculate_synthetic`).
///
/// # Errors
/// Returns [`ColorsError`] if resampling fails or if `zero_point_flux` or
/// `synthetic_flux` is non-positive.
pub fn calculate_synthetic<T: RealScalar>(
    wavelengths: &[f64],
    fluxes: &[T],
    filter_wavelengths: &[f64],
    filter_trans: &[f64],
    zero_point_flux: f64,
) -> Result<T, ColorsError> {
    if zero_point_flux <= 0.0 {
        return Err(ColorsError::NonPositiveFlux {
            context: "calculate_synthetic (zero_point_flux)",
            flux: zero_point_flux,
        });
    }
    let n = wavelengths.len();
    if fluxes.len() != n || n < 2 {
        return Err(ColorsError::InvalidArrayLength {
            context: "calculate_synthetic",
            got: fluxes.len(),
            expected: n.max(2),
        });
    }
    let f_on_sed =
        interpolate_array(filter_wavelengths, filter_trans, wavelengths)?;
    let convolved: Vec<T> = (0..n)
        .map(|i| {
            cleaned_flux_at(wavelengths, fluxes, i) * T::from_f64(f_on_sed[i])
        })
        .collect();
    let syn_flux =
        calculate_synthetic_flux(wavelengths, &convolved, &f_on_sed)?;
    if syn_flux.val() <= 0.0 {
        return Err(ColorsError::NonPositiveFlux {
            context: "calculate_synthetic (synthetic_flux)",
            flux: syn_flux.val(),
        });
    }
    Ok(T::from_f64(-2.5) * (syn_flux / T::from_f64(zero_point_flux)).log10())
}

/// Computes the Vega photon-counting zero-point flux for a filter curve
/// (`synthetic.f90:compute_vega_zero_point`).
///
/// # Errors
/// Returns [`ColorsError`] if filter interpolation or quadrature fails.
pub fn compute_vega_zero_point(
    vega_wave: &[f64],
    vega_flux: &[f64],
    filt_wave: &[f64],
    filt_trans: &[f64],
) -> Result<f64, ColorsError> {
    let f_on_vega = interpolate_array(filt_wave, filt_trans, vega_wave)?;
    let conv: Vec<f64> = vega_flux
        .iter()
        .zip(f_on_vega.iter())
        .map(|(&vf, &ft)| vf * ft)
        .collect();
    calculate_synthetic_flux(vega_wave, &conv, &f_on_vega)
}

/// Computes the AB photon-counting zero-point flux ($f_\nu = 3631\,$Jy,
/// $f_\lambda = 3.631 \times 10^{-20} (c \cdot 10^8) / \lambda^2$) on the
/// filter wavelength grid (`synthetic.f90:compute_ab_zero_point`).
///
/// # Errors
/// Returns [`ColorsError`] if `filt_wave` and `filt_trans` have invalid
/// lengths.
pub fn compute_ab_zero_point(
    filt_wave: &[f64],
    filt_trans: &[f64],
) -> Result<f64, ColorsError> {
    let n = filt_wave.len();
    if filt_trans.len() != n || n < 2 {
        return Err(ColorsError::InvalidArrayLength {
            context: "compute_ab_zero_point",
            got: filt_trans.len(),
            expected: n.max(2),
        });
    }
    let conv: Vec<f64> = filt_wave
        .iter()
        .zip(filt_trans.iter())
        .map(|(&w, &tr)| {
            let ab_f = if w > 0.0 {
                3.631e-20 * ((CLIGHT * 1.0e8) / (w * w))
            } else {
                0.0
            };
            ab_f * tr
        })
        .collect();
    calculate_synthetic_flux(filt_wave, &conv, filt_trans)
}

/// Computes the ST photon-counting zero-point flux
/// ($f_\lambda = 3.63 \times 10^{-9}$) on the filter wavelength grid
/// (`synthetic.f90:compute_st_zero_point`).
///
/// # Errors
/// Returns [`ColorsError`] if `filt_wave` and `filt_trans` have invalid
/// lengths.
pub fn compute_st_zero_point(
    filt_wave: &[f64],
    filt_trans: &[f64],
) -> Result<f64, ColorsError> {
    let n = filt_wave.len();
    if filt_trans.len() != n || n < 2 {
        return Err(ColorsError::InvalidArrayLength {
            context: "compute_st_zero_point",
            got: filt_trans.len(),
            expected: n.max(2),
        });
    }
    let conv: Vec<f64> = filt_trans.iter().map(|&tr| 3.63e-9 * tr).collect();
    calculate_synthetic_flux(filt_wave, &conv, filt_trans)
}

impl SedCube {
    /// Evaluates the 10-column `colors` history output vector
    /// `[Mag_bol, Flux_bol, Interp_rad, B, I, J, M, R, U, V]` using cubic
    /// Hermite SED interpolation, generic over [`RealScalar`] (`f64` and
    /// `DualOrder1<N>`) (`colors_history.f90:data_for_colors_history_columns`).
    ///
    /// # Errors
    /// Returns [`ColorsError`] if quadrature fails.
    pub fn eval_colors_generic<T: RealScalar>(
        &self,
        teff: T,
        log_g: T,
        metallicity: T,
        r: T,
        d: T,
        mag_system: MagSystem,
    ) -> Result<[T; NUM_COLORS_OUT], ColorsError> {
        let interp_rad = compute_interp_radius(
            teff.val(),
            log_g.val(),
            metallicity.val(),
            &self.teff_grid,
            &self.logg_grid,
            &self.meta_grid,
        );
        if self.wavelengths.len() == 1199 {
            let mut sed_buf = [T::zero(); 1199];
            self.eval_sed_hermite_into(
                teff,
                log_g,
                metallicity,
                r,
                d,
                &mut sed_buf,
            );
            return self
                .colors_from_sed_generic(&sed_buf, interp_rad, mag_system);
        }
        let fluxes =
            self.eval_sed_hermite_generic(teff, log_g, metallicity, r, d);
        self.colors_from_sed_generic(&fluxes, interp_rad, mag_system)
    }

    /// Evaluates the 10-column `colors` history output vector in `f64`
    /// precision for any [`SedInterpMethod`] (`Hermite`, `Linear`, or `Knn`).
    ///
    /// # Errors
    /// Returns [`ColorsError`] if quadrature fails.
    pub fn eval_colors(
        &self,
        teff: f64,
        log_g: f64,
        metallicity: f64,
        r: f64,
        d: f64,
        mag_system: MagSystem,
        method: SedInterpMethod,
    ) -> Result<[f64; NUM_COLORS_OUT], ColorsError> {
        // Fast-path stack-allocated Hermite SED evaluation when requested.
        if method == SedInterpMethod::Hermite {
            return self.eval_colors_generic(
                teff,
                log_g,
                metallicity,
                r,
                d,
                mag_system,
            );
        }
        let interp_rad = compute_interp_radius(
            teff,
            log_g,
            metallicity,
            &self.teff_grid,
            &self.logg_grid,
            &self.meta_grid,
        );
        let fluxes = self.eval_sed(teff, log_g, metallicity, r, d, method);
        self.colors_from_sed_generic(&fluxes, interp_rad, mag_system)
    }

    /// Computes `[Mag_bol, Flux_bol, Interp_rad, B, I, J, M, R, U, V]` from an
    /// already-diluted SED vector `fluxes` in a single fused Simpson pass.
    fn colors_from_sed_generic<T: RealScalar>(
        &self,
        fluxes: &[T],
        interp_rad: f64,
        mag_system: MagSystem,
    ) -> Result<[T; NUM_COLORS_OUT], ColorsError> {
        let w = &self.wavelengths;
        let n = w.len();
        if fluxes.len() != n || n < 2 {
            return Err(ColorsError::InvalidArrayLength {
                context: "colors_from_sed_generic",
                got: fluxes.len(),
                expected: n.max(2),
            });
        }

        let mut bol_flux = T::zero();
        let mut int_flux = [T::zero(); NUM_JOHNSON_FILTERS];
        // Accumulate bolometric and photon-weighted filter integrals in one
        // fused pass over all precomputed non-uniform Simpson panels.
        for (p, &[c0_f, c1_f, c2_f]) in self.simpson_panels.iter().enumerate() {
            let i = 2 * p;
            let (c0, c1, c2) =
                (T::from_f64(c0_f), T::from_f64(c1_f), T::from_f64(c2_f));
            let f0 = cleaned_flux_at(w, fluxes, i);
            let f1 = cleaned_flux_at(w, fluxes, i + 1);
            let f2 = cleaned_flux_at(w, fluxes, i + 2);
            bol_flux = bol_flux + f0 * c0 + f1 * c1 + f2 * c2;

            for (f_idx, acc) in int_flux.iter_mut().enumerate() {
                let fw = &self.filters_weighted_on_sed_grid[f_idx];
                let y0 = f0 * T::from_f64(fw[i]);
                let y1 = f1 * T::from_f64(fw[i + 1]);
                let y2 = f2 * T::from_f64(fw[i + 2]);
                *acc = *acc + y0 * c0 + y1 * c1 + y2 * c2;
            }
        }
        if n.is_multiple_of(2) {
            let dx = T::from_f64(0.5 * (w[n - 1] - w[n - 2]));
            let f_last = cleaned_flux_at(w, fluxes, n - 1);
            let f_prev = cleaned_flux_at(w, fluxes, n - 2);
            bol_flux = bol_flux + dx * (f_last + f_prev);
            for (f_idx, acc) in int_flux.iter_mut().enumerate() {
                let fw = &self.filters_weighted_on_sed_grid[f_idx];
                let y_last = f_last * T::from_f64(fw[n - 1]);
                let y_prev = f_prev * T::from_f64(fw[n - 2]);
                *acc = *acc + dx * (y_last + y_prev);
            }
        }

        let mut out = [T::zero(); NUM_COLORS_OUT];
        out[0] = flux_to_magnitude(bol_flux);
        out[1] = bol_flux;
        out[2] = T::from_f64(interp_rad);

        for (f_idx, &iflux) in int_flux.iter().enumerate() {
            let zp = JOHNSON_FILTERS[f_idx].zero_point(mag_system);
            let norm = self.filter_norm_integrals[f_idx];
            let syn_flux = iflux / T::from_f64(norm);
            out[3 + f_idx] =
                T::from_f64(-2.5) * (syn_flux / T::from_f64(zp)).log10();
        }
        Ok(out)
    }
}
