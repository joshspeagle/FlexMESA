// SPDX-License-Identifier: LGPL-3.0-or-later
//! Thin public entry points delegating convective premixing (`conv_premix.rs`)
//! and predictive mixing (`predictive_mix.rs`) to verbatim MESA `26.4.1`
//! implementations, plus white-dwarf crystallization phase separation
//! (`phase_separation.f90`).

use crate::controls::StarError;
use crate::model::StarModel;

pub use crate::phase_separation::{
    do_phase_separation, eval_phase_separation_shift,
};

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
