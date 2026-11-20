// SPDX-License-Identifier: LGPL-3.0-or-later
//! Thin public entry points delegating convective premixing (`conv_premix.rs`)
//! and predictive mixing (`predictive_mix.rs`) to verbatim MESA `26.4.1`
//! implementations, plus white-dwarf crystallization phase separation
//! (`phase_separation.f90`).

use crate::controls::StarError;
use crate::micro::{do_eos, set_basic_vars};
use crate::model::StarModel;

/// Evaluates the Blouin et al. (2020, 2021) 5th-order polynomial distillation
/// shift $\Delta x_{\mathrm{heavy}}$ between the solid crystal and coexisting
/// liquid phase at crystallization (`phase_separation.f90:279-302`).
#[must_use]
pub fn eval_phase_separation_shift(x_heavy: f64, is_one_mixture: bool) -> f64 {
    let x = x_heavy.clamp(0.0, 1.0);
    let ratio_a = if is_one_mixture {
        16.0 / 20.0
    } else {
        12.0 / 16.0
    };
    let xi = (x * ratio_a) / (1.0 - x * (1.0 - ratio_a)).max(1.0e-12);
    let coeffs: [f64; 5] = if is_one_mixture {
        [0.435920, -1.242958, 1.711583, -1.366634, 0.462089]
    } else {
        [0.642856, -1.954689, 2.487018, -1.789000, 0.613815]
    };
    let mut delta_xi = 0.0;
    let mut xp = xi;
    for &c in &coeffs {
        delta_xi += c * xp;
        xp *= xi;
    }
    let xi_solid = (xi + delta_xi).clamp(0.0, 1.0);
    let x_solid =
        xi_solid / (ratio_a + xi_solid * (1.0 - ratio_a)).max(1.0e-12);
    (x_solid - x).max(0.0)
}

/// Runs verbatim MESA `26.4.1` convective premixing
/// (`crate::conv_premix::do_conv_premix`) and updates `model.aux.eps_pre_mix`
/// from the resulting internal energy change when `do_premix_heating` is set.
///
/// The `_max_iters` parameter is retained for PyO3 API compatibility; verbatim
/// MESA `26.4.1` advances each eligible boundary until stopped by stability,
/// clock budget (`conv_premix_time_factor`), or `avoid_fuel_increase`.
///
/// # Errors
/// Returns [`StarError`] if EOS, opacity, MLT, or zone validation fails.
pub fn do_conv_premix(
    model: &mut StarModel,
    _max_iters: usize,
    avoid_fuel_increase: bool,
) -> Result<usize, StarError> {
    let nz = model.nz;
    if model.aux.eps_pre_mix.len() != nz {
        model.aux.eps_pre_mix.resize(nz, 0.0);
    }
    model.aux.eps_pre_mix.fill(0.0);
    if nz < 2 || model.species == 0 {
        return Ok(0);
    }

    let e_before = model.energy.clone();
    let prev_avoid = model.controls.mix_wind.conv_premix_avoid_increase;
    model.controls.mix_wind.conv_premix_avoid_increase = avoid_fuel_increase;
    let res = crate::conv_premix::do_conv_premix(model);
    model.controls.mix_wind.conv_premix_avoid_increase = prev_avoid;
    let total_advanced = res?;

    if total_advanced > 0 && model.controls.do_premix_heating && model.dt > 0.0
    {
        for k in 0..nz {
            model.aux.eps_pre_mix[k] =
                (e_before[k] - model.energy[k]) / model.dt;
        }
    }
    Ok(total_advanced)
}

/// Runs verbatim MESA `26.4.1` predictive mixing
/// (`crate::predictive_mix::add_predictive_mixing`) using an explicit
/// `superad_thresh` matching the standalone Python helper signature.
///
/// `_ingested_mass_limit_coeff` is retained for Python signature compatibility;
/// configure `PredictiveParamSet::{limit_ingestion, ingestion_factor}` directly
/// on `model.controls.mix_wind.predictive_sets` to enable Spruit (2015) mass
/// ingestion limiting.
///
/// # Errors
/// Returns [`StarError`] if `do_conv_premix` is simultaneously enabled or if
/// trial EOS, opacity, or MLT evaluation fails.
pub fn add_predictive_mixing(
    model: &mut StarModel,
    _ingested_mass_limit_coeff: f64,
    superad_thresh: f64,
) -> Result<usize, StarError> {
    if model.nz < 2 {
        return Ok(0);
    }
    let saved_sets = model.controls.mix_wind.predictive_sets.clone();
    for ps in &mut model.controls.mix_wind.predictive_sets {
        ps.predictive_mix = false;
    }
    if let Some(ps0) = model.controls.mix_wind.predictive_sets.first_mut() {
        ps0.predictive_mix = true;
        ps0.zone_type = String::from("any");
        ps0.zone_loc = String::from("any");
        ps0.bdy_loc = String::from("any");
        ps0.bdy_q_min = 0.0;
        ps0.bdy_q_max = 1.0;
        ps0.superad_thresh = superad_thresh;
        ps0.avoid_reversal = String::new();
        ps0.limit_ingestion = String::new();
    }
    if model.aux.mixing.conv_bdy_loc.is_empty() {
        let zeros = vec![0.0; model.nz];
        crate::mix_info::locate_convection_boundaries_in_place(
            model, &zeros, &zeros, &zeros,
        );
    }
    let res = crate::predictive_mix::add_predictive_mixing(model);
    model.controls.mix_wind.predictive_sets = saved_sets;
    res
}

/// Evaluates all enabled `predictive_mix` rules in `model.controls.mix_wind`
/// via [`crate::predictive_mix::add_predictive_mixing`].
///
/// # Errors
/// Returns [`StarError`] if a predictive parameter set has invalid selectors,
/// conflicts with `do_conv_premix`, or fails trial EOS/opacity/MLT evaluation.
pub fn apply_predictive_mixing_from_controls(
    model: &mut StarModel,
) -> Result<usize, StarError> {
    crate::predictive_mix::add_predictive_mixing(model)
}

/// Executes white-dwarf crystallization phase separation
/// (`phase_separation.f90:do_phase_separation`) across zones reaching plasma
/// coupling $\Gamma_k \ge \Gamma_{\mathrm{cryst}}$ (`~175`).
///
/// # Errors
/// Returns [`StarError`] if refreshing basic/EOS variables fails after
/// redistributing phase-separated abundances.
pub fn do_phase_separation(
    model: &mut StarModel,
    dt: f64,
    gamma_crystallization: f64,
    is_one_mixture: bool,
) -> Result<Vec<f64>, StarError> {
    let nz = model.nz;
    let nc = model.species;
    let mut eps_phase_sep = vec![0.0; nz];
    if nz < 3 || nc < 2 || dt <= 0.0 {
        return Ok(eps_phase_sep);
    }

    let (j_light, j_heavy) = if is_one_mixture {
        let jo = model.iso_names.iter().position(|s| s == "o16").unwrap_or(0);
        let jne = model
            .iso_names
            .iter()
            .position(|s| s == "ne20")
            .unwrap_or((jo + 1).min(nc - 1));
        (jo, jne)
    } else {
        let jc = model.iso_names.iter().position(|s| s == "c12").unwrap_or(0);
        let jo = model
            .iso_names
            .iter()
            .position(|s| s == "o16")
            .unwrap_or((jc + 1).min(nc - 1));
        (jc, jo)
    };
    if j_light == j_heavy {
        return Ok(eps_phase_sep);
    }

    let gam_crit = gamma_crystallization.max(100.0);
    let mut e_before: Option<Vec<f64>> = None;

    for k in (1..nz).rev() {
        if model.gam[k] >= gam_crit && model.gam[k - 1] < gam_crit {
            let k_top = k.saturating_sub(6);
            let avail_heavy: f64 = (k_top..k)
                .map(|kl| {
                    let cap = model.xa[kl * nc + j_heavy]
                        .min(1.0 - model.xa[kl * nc + j_light])
                        .min(0.5);
                    model.dm[kl] * cap.max(0.0)
                })
                .sum();
            let dm_solid = model.dm[k];
            if dm_solid <= 0.0 || avail_heavy <= 0.0 {
                continue;
            }
            let x_heavy_k = model.xa[k * nc + j_heavy];
            let x_light_k = model.xa[k * nc + j_light];
            let x_pair_tot = (x_heavy_k + x_light_k).max(1.0e-12);
            let x_heavy_norm = (x_heavy_k / x_pair_tot).clamp(0.0, 1.0);
            let dx =
                (eval_phase_separation_shift(x_heavy_norm, is_one_mixture)
                    * x_pair_tot)
                    .min(x_light_k)
                    .min(1.0 - x_heavy_k)
                    .min(avail_heavy / dm_solid);
            if dx <= 1.0e-8 {
                continue;
            }
            if e_before.is_none() {
                e_before = Some(model.energy.clone());
            }
            let delta_m_heavy = dm_solid * dx;
            model.xa[k * nc + j_heavy] =
                (model.xa[k * nc + j_heavy] + dx).clamp(0.0, 1.0);
            model.xa[k * nc + j_light] =
                (model.xa[k * nc + j_light] - dx).clamp(0.0, 1.0);

            let frac = delta_m_heavy / avail_heavy;
            for kl in k_top..k {
                let cap = model.xa[kl * nc + j_heavy]
                    .min(1.0 - model.xa[kl * nc + j_light])
                    .clamp(0.0, 0.5);
                let shift = frac * cap;
                model.xa[kl * nc + j_heavy] =
                    (model.xa[kl * nc + j_heavy] - shift).clamp(0.0, 1.0);
                model.xa[kl * nc + j_light] =
                    (model.xa[kl * nc + j_light] + shift).clamp(0.0, 1.0);
            }
        }
    }

    if let Some(e_old) = e_before {
        set_basic_vars(model)?;
        do_eos(model)?;
        for k in 0..nz {
            eps_phase_sep[k] = (e_old[k] - model.energy[k]) / dt;
        }
    }
    Ok(eps_phase_sep)
}
