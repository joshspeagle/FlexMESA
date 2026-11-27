// SPDX-License-Identifier: GPL-3.0-or-later
//! Element diffusion and white-dwarf phase separation `&controls` matching
//! MESA `26.4.1` `star/defaults/controls.defaults:6883-7570` and
//! `star/defaults/controls_dev.defaults:152-161`.

use crate::controls::StarError;
use flexmesa_schema::ControlValue;

use crate::inlist::SectionControls;

/// Maximum number of element diffusion species classes (`max_num_diffusion_classes`).
pub const MAX_NUM_DIFFUSION_CLASSES: usize = 100;

/// Element diffusion and phase-separation control parameters (`&controls`).
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DiffusionControls {
    /// Terminal output for element diffusion (`show_diffusion_info`).
    pub show_diffusion_info: bool,
    /// Terminal output per diffusion substep (`show_diffusion_substep_info`).
    pub show_diffusion_substep_info: bool,
    /// Terminal timing report for diffusion (`show_diffusion_timing`).
    pub show_diffusion_timing: bool,
    /// Skip element diffusion if timestep `dt < diffusion_dt_limit` (s).
    pub diffusion_dt_limit: f64,
    /// Use Paquette et al. (1986) ion-ion collision integrals instead of Stanton & Murillo (`diffusion_use_paquette`).
    pub diffusion_use_paquette: bool,
    /// Blend to Caplan, Bauer & Freeman (2022) strong-coupling OCP diffusion for $\Gamma > 3$ (`diffusion_use_caplan`).
    pub diffusion_use_caplan: bool,
    /// Use Iben & MacDonald (1985) pure screened Coulomb coefficients (`diffusion_use_iben_macdonald`).
    pub diffusion_use_iben_macdonald: bool,
    /// Solve unmodified Burgers equations in CGS units instead of Thoul et al. (1994) (`diffusion_use_cgs_solver`).
    pub diffusion_use_cgs_solver: bool,
    /// Lower $\eta$ threshold where CGS thermal diffusion is fully on (`cgs_thermal_diffusion_eta_full_on`).
    pub cgs_thermal_diffusion_eta_full_on: f64,
    /// Upper $\eta$ threshold where CGS thermal diffusion is fully off (`cgs_thermal_diffusion_eta_full_off`).
    pub cgs_thermal_diffusion_eta_full_off: f64,
    /// Enable white-dwarf gravitational sedimentation heating (`do_WD_sedimentation_heating`).
    pub do_wd_sedimentation_heating: bool,
    /// Minimum species mass fraction included in WD sedimentation heating (`min_xa_for_WD_sedimentation_heating`).
    pub min_xa_for_wd_sedimentation_heating: f64,
    /// Enable internal-energy change heating from element diffusion (`do_diffusion_heating`).
    pub do_diffusion_heating: bool,
    /// Minimum cumulative fractional mass combined at surface for diffusion (`diffusion_min_dq_at_surface`).
    pub diffusion_min_dq_at_surface: f64,
    /// Combine surface cells cooler than this temperature for diffusion (`diffusion_min_T_at_surface`).
    pub diffusion_min_t_at_surface: f64,
    /// Combine surface cells until cumulative `dq >= ratio * dq(k+1)` (`diffusion_min_dq_ratio_at_surface`).
    pub diffusion_min_dq_ratio_at_surface: f64,
    /// Substep size multiplier relative to diffusion timescale (`diffusion_dt_div_timescale`).
    pub diffusion_dt_div_timescale: f64,
    /// Minimum number of diffusion substeps per stellar step (`diffusion_min_num_substeps`).
    pub diffusion_min_num_substeps: usize,
    /// Maximum Newton iterations per backward-Euler diffusion substep (`diffusion_max_iters_per_substep`).
    pub diffusion_max_iters_per_substep: usize,
    /// Maximum retries per backward-Euler diffusion substep (`diffusion_max_retries_per_substep`).
    pub diffusion_max_retries_per_substep: usize,
    /// Maximum per-cell correction tolerance for backward-Euler convergence (`diffusion_tol_correction_max`).
    pub diffusion_tol_correction_max: f64,
    /// Mean absolute correction tolerance for backward-Euler convergence (`diffusion_tol_correction_norm`).
    pub diffusion_tol_correction_norm: f64,
    /// Hard negative mass-fraction limit triggering diffusion substep retry (`diffusion_min_X_hard_limit`).
    pub diffusion_min_x_hard_limit: f64,
    /// Absolute species-mass conservation tolerance in `fixup` (`diffusion_X_total_atol`).
    pub diffusion_x_total_atol: f64,
    /// Relative species-mass conservation tolerance in `fixup` (`diffusion_X_total_rtol`).
    pub diffusion_x_total_rtol: f64,
    /// Switch face advection to upwind when $|v_{\mathrm{adv}}| \ge$ limit (`diffusion_upwind_abs_v_limit`).
    pub diffusion_upwind_abs_v_limit: f64,
    /// Maximum allowed advection velocity magnitude in $\mathrm{cm\,s^{-1}}$ (`diffusion_v_max`).
    pub diffusion_v_max: f64,
    /// Mixing diffusivity threshold above which element diffusion is skipped in core/envelope CZs (`D_mix_ignore_diffusion`).
    pub d_mix_ignore_diffusion: f64,
    /// Upper $\Gamma$ cutoff where diffusion is completely shut off (`diffusion_gamma_full_off`).
    pub diffusion_gamma_full_off: f64,
    /// Lower $\Gamma$ threshold where diffusion remains fully on (`diffusion_gamma_full_on`).
    pub diffusion_gamma_full_on: f64,
    /// Temperature above which diffusion is fully on (`diffusion_T_full_on`).
    pub diffusion_t_full_on: f64,
    /// Temperature below which diffusion is completely shut off (`diffusion_T_full_off`).
    pub diffusion_t_full_off: f64,
    /// Compute local ionization charges via Paquette/Dupuis continuum depression (`diffusion_calculates_ionization`).
    pub diffusion_calculates_ionization: bool,
    /// Number of forward/backward smoothing passes on typical charges (`diffusion_nsmooth_typical_charge`).
    pub diffusion_nsmooth_typical_charge: usize,
    /// Multiplier for concentration-diffusion `SIG` terms (`diffusion_SIG_factor`).
    pub diffusion_sig_factor: f64,
    /// Multiplier for gravitational/thermal/radiative `GT` advection terms (`diffusion_GT_factor`).
    pub diffusion_gt_factor: f64,
    /// Exterior mass (Msun) inside which artificial surface concentration diffusion is full on (`diffusion_AD_dm_full_on`).
    pub diffusion_ad_dm_full_on: f64,
    /// Exterior mass (Msun) below which artificial surface concentration diffusion is full off (`diffusion_AD_dm_full_off`).
    pub diffusion_ad_dm_full_off: f64,
    /// Boost factor for artificial surface concentration diffusion (`diffusion_AD_boost_factor`).
    pub diffusion_ad_boost_factor: f64,
    /// Exterior mass (Msun) inside which surface velocity limiting is full on (`diffusion_Vlimit_dm_full_on`).
    pub diffusion_vlimit_dm_full_on: f64,
    /// Exterior mass (Msun) below which surface velocity limiting is full off (`diffusion_Vlimit_dm_full_off`).
    pub diffusion_vlimit_dm_full_off: f64,
    /// Surface velocity cap in units of local cell-crossing velocity `dr/dt` (`diffusion_Vlimit`).
    pub diffusion_vlimit: f64,
    /// Minimum temperature for radiative levitation (`diffusion_min_T_for_radaccel`).
    pub diffusion_min_t_for_radaccel: f64,
    /// Maximum temperature for radiative levitation (`diffusion_max_T_for_radaccel`).
    pub diffusion_max_t_for_radaccel: f64,
    /// Minimum nuclear charge $Z$ for radiative levitation (`diffusion_min_Z_for_radaccel`).
    pub diffusion_min_z_for_radaccel: isize,
    /// Maximum nuclear charge $Z$ for radiative levitation (`diffusion_max_Z_for_radaccel`).
    pub diffusion_max_z_for_radaccel: isize,
    /// Include plasma screening in radiative levitation (`diffusion_screening_for_radaccel`).
    pub diffusion_screening_for_radaccel: bool,
    /// Treat every isotope in the nuclear network as its own diffusion class (`diffusion_use_full_net`).
    pub diffusion_use_full_net: bool,
    /// Number of lumped diffusion classes when `diffusion_use_full_net = false` (`diffusion_num_classes`).
    pub diffusion_num_classes: usize,
    /// Representative isotope name per diffusion class (`diffusion_class_representative`, 100 slots).
    pub diffusion_class_representative: Vec<String>,
    /// Upper atomic mass number `A` cutoff per diffusion class (`diffusion_class_A_max`, 100 slots).
    pub diffusion_class_a_max: Vec<f64>,
    /// Fixed typical ionic charge per class when `diffusion_calculates_ionization = false` (`diffusion_class_typical_charge`, 100 slots).
    pub diffusion_class_typical_charge: Vec<f64>,
    /// Per-class multiplicative scaling factor for diffusion coefficients (`diffusion_class_factor`, 100 slots).
    pub diffusion_class_factor: Vec<f64>,
    /// Use stiff Rosenbrock `isolve` integrator instead of backward Euler (`diffusion_use_isolve`).
    pub diffusion_use_isolve: bool,
    /// Relative tolerance for `isolve` (`diffusion_rtol_for_isolve`).
    pub diffusion_rtol_for_isolve: f64,
    /// Absolute tolerance for `isolve` (`diffusion_atol_for_isolve`).
    pub diffusion_atol_for_isolve: f64,
    /// Maximum steps allowed in `isolve` (`diffusion_maxsteps_for_isolve`).
    pub diffusion_maxsteps_for_isolve: usize,
    /// Rosenbrock solver name for `isolve` (`diffusion_isolve_solver`).
    pub diffusion_isolve_solver: String,
    /// Debug dump call number (`diffusion_dump_call_number`).
    pub diffusion_dump_call_number: isize,
    /// Use Ledoux/Brunt-B criterion instead of simple $\mu$ gradient in `mix_outward` (`phase_separation_mixing_use_brunt`).
    pub phase_separation_mixing_use_brunt: bool,
    /// Disable element diffusion below the phase-separation mixing boundary (`phase_separation_no_diffusion`).
    pub phase_separation_no_diffusion: bool,
    /// Custom `other_diffusion` hook flag (`use_other_diffusion`).
    pub use_other_diffusion: bool,
    /// Custom `other_diffusion_factor` hook flag (`use_other_diffusion_factor`).
    pub use_other_diffusion_factor: bool,
    /// Custom `other_diffusion_coefficients` hook flag (`use_other_diffusion_coefficients`).
    pub use_other_diffusion_coefficients: bool,
}

impl Default for DiffusionControls {
    fn default() -> Self {
        Self {
            show_diffusion_info: false,
            show_diffusion_substep_info: false,
            show_diffusion_timing: false,
            diffusion_dt_limit: 3.15e7,
            diffusion_use_paquette: false,
            diffusion_use_caplan: true,
            diffusion_use_iben_macdonald: false,
            diffusion_use_cgs_solver: true,
            cgs_thermal_diffusion_eta_full_on: 0.0,
            cgs_thermal_diffusion_eta_full_off: 2.0,
            do_wd_sedimentation_heating: false,
            min_xa_for_wd_sedimentation_heating: 1.0e-5,
            do_diffusion_heating: true,
            diffusion_min_dq_at_surface: 1.0e-9,
            diffusion_min_t_at_surface: 1.0e4,
            diffusion_min_dq_ratio_at_surface: 10.0,
            diffusion_dt_div_timescale: 1.0,
            diffusion_min_num_substeps: 1,
            diffusion_max_iters_per_substep: 10,
            diffusion_max_retries_per_substep: 10,
            diffusion_tol_correction_max: 1.0e-1,
            diffusion_tol_correction_norm: 1.0e-3,
            diffusion_min_x_hard_limit: -1.0e-3,
            diffusion_x_total_atol: 1.0e-9,
            diffusion_x_total_rtol: 1.0e-6,
            diffusion_upwind_abs_v_limit: 1.0e99,
            diffusion_v_max: 1.0e-3,
            d_mix_ignore_diffusion: 1.0e5,
            diffusion_gamma_full_off: 1.0e99,
            diffusion_gamma_full_on: 1.0e99,
            diffusion_t_full_on: 1.0e3,
            diffusion_t_full_off: 1.0e3,
            diffusion_calculates_ionization: true,
            diffusion_nsmooth_typical_charge: 10,
            diffusion_sig_factor: 1.0,
            diffusion_gt_factor: 1.0,
            diffusion_ad_dm_full_on: -1.0,
            diffusion_ad_dm_full_off: -1.0,
            diffusion_ad_boost_factor: 0.0,
            diffusion_vlimit_dm_full_on: -1.0,
            diffusion_vlimit_dm_full_off: -1.0,
            diffusion_vlimit: 0.0,
            diffusion_min_t_for_radaccel: 0.0,
            diffusion_max_t_for_radaccel: 0.0,
            diffusion_min_z_for_radaccel: 0,
            diffusion_max_z_for_radaccel: 1000,
            diffusion_screening_for_radaccel: true,
            diffusion_use_full_net: false,
            diffusion_num_classes: 5,
            diffusion_class_representative: vec![String::from("h1"), String::from("he3"), String::from("he4"), String::from("o16"), String::from("fe56")],
            diffusion_class_a_max: vec![2.0, 3.0, 4.0, 16.0, 10000.0],
            diffusion_class_typical_charge: vec![1.0, 2.0, 2.0, 8.0, 21.0],
            diffusion_class_factor: vec![1.0; MAX_NUM_DIFFUSION_CLASSES],
            diffusion_use_isolve: false,
            diffusion_rtol_for_isolve: 1.0e-4,
            diffusion_atol_for_isolve: 1.0e-5,
            diffusion_maxsteps_for_isolve: 1000,
            diffusion_isolve_solver: String::from("ros2_solver"),
            diffusion_dump_call_number: -1,
            phase_separation_mixing_use_brunt: true,
            phase_separation_no_diffusion: false,
            use_other_diffusion: false,
            use_other_diffusion_factor: false,
            use_other_diffusion_coefficients: false,
        }
    }
}

fn parse_f64_val(val: &ControlValue, ctx: &str) -> Result<f64, StarError> {
    match val {
        ControlValue::Float(f) if f.is_finite() => Ok(*f),
        ControlValue::Float(f) => Err(StarError::InvalidInput(format!(
            "non-finite value for {ctx}: {f}"
        ))),
        ControlValue::Integer(i) => Ok(*i as f64),
        other => Err(StarError::InvalidInput(format!(
            "expected real number for {ctx}, found {other:?}"
        ))),
    }
}

fn apply_indexed_f64_slots(
    ctrl: &SectionControls,
    key: &str,
    dst: &mut Vec<f64>,
    default_fill: f64,
) -> Result<(), StarError> {
    let low = key.to_ascii_lowercase();
    if let Some(val) = ctrl.scalars.get(&low) {
        let f = parse_f64_val(val, &format!("'&controls::{low}'"))?;
        dst.resize(MAX_NUM_DIFFUSION_CLASSES, default_fill);
        dst.fill(f);
    }
    if let Some(list) = ctrl.indexed.get(&low) {
        for (sub, val) in list {
            let trimmed = sub.trim();
            if trimmed == ":" {
                let f = parse_f64_val(val, &format!("'&controls::{low}(:)'"))?;
                dst.resize(MAX_NUM_DIFFUSION_CLASSES, default_fill);
                dst.fill(f);
                continue;
            }
            let idx = trimmed.parse::<usize>().map_err(|_| {
                StarError::InvalidInput(format!(
                    "invalid subscript '({sub})' for '&controls::{low}'"
                ))
            })?;
            if idx == 0 || idx > MAX_NUM_DIFFUSION_CLASSES {
                return Err(StarError::InvalidInput(format!(
                    "subscript '({idx})' out of range 1..={MAX_NUM_DIFFUSION_CLASSES} for '&controls::{low}'"
                )));
            }
            if idx > dst.len() {
                dst.resize(idx, default_fill);
            }
            dst[idx - 1] =
                parse_f64_val(val, &format!("'&controls::{low}({idx})'"))?;
        }
    }
    Ok(())
}

fn apply_indexed_str_slots(
    ctrl: &SectionControls,
    key: &str,
    dst: &mut Vec<String>,
) -> Result<(), StarError> {
    let low = key.to_ascii_lowercase();
    if let Some(val) = ctrl.scalars.get(&low) {
        match val {
            ControlValue::String(s) => {
                dst.resize(MAX_NUM_DIFFUSION_CLASSES, String::new());
                dst.fill(s.clone());
            }
            other => {
                return Err(StarError::InvalidInput(format!(
                    "expected string for '&controls::{low}', found {other:?}"
                )));
            }
        }
    }
    if let Some(list) = ctrl.indexed.get(&low) {
        for (sub, val) in list {
            let trimmed = sub.trim();
            let s = match val {
                ControlValue::String(s) => s.clone(),
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "expected string for '&controls::{low}({sub})', found {other:?}"
                    )));
                }
            };
            if trimmed == ":" {
                dst.resize(MAX_NUM_DIFFUSION_CLASSES, String::new());
                dst.fill(s);
                continue;
            }
            let idx = trimmed.parse::<usize>().map_err(|_| {
                StarError::InvalidInput(format!(
                    "invalid subscript '({sub})' for '&controls::{low}'"
                ))
            })?;
            if idx == 0 || idx > MAX_NUM_DIFFUSION_CLASSES {
                return Err(StarError::InvalidInput(format!(
                    "subscript '({idx})' out of range 1..={MAX_NUM_DIFFUSION_CLASSES} for '&controls::{low}'"
                )));
            }
            if idx > dst.len() {
                dst.resize(idx, String::new());
            }
            dst[idx - 1] = s;
        }
    }
    Ok(())
}

/// Populates [`DiffusionControls`] from a parsed `&controls` namelist section.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] on type mismatches, out-of-range array
/// subscripts, or unrecognized `diffusion_isolve_solver` option strings.
pub(crate) fn populate_diffusion_controls(
    ctrl: &SectionControls,
    d: &mut DiffusionControls,
) -> Result<(), StarError> {
    ctrl.try_bool_into(
        "controls",
        &mut d.show_diffusion_info,
        "show_diffusion_info",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.show_diffusion_substep_info,
        "show_diffusion_substep_info",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.show_diffusion_timing,
        "show_diffusion_timing",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_dt_limit,
        "diffusion_dt_limit",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.diffusion_use_paquette,
        "diffusion_use_paquette",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.diffusion_use_caplan,
        "diffusion_use_caplan",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.diffusion_use_iben_macdonald,
        "diffusion_use_iben_macdonald",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.diffusion_use_cgs_solver,
        "diffusion_use_cgs_solver",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.cgs_thermal_diffusion_eta_full_on,
        "cgs_thermal_diffusion_eta_full_on",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.cgs_thermal_diffusion_eta_full_off,
        "cgs_thermal_diffusion_eta_full_off",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.do_wd_sedimentation_heating,
        "do_wd_sedimentation_heating",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.min_xa_for_wd_sedimentation_heating,
        "min_xa_for_wd_sedimentation_heating",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.do_diffusion_heating,
        "do_diffusion_heating",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_min_dq_at_surface,
        "diffusion_min_dq_at_surface",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_min_t_at_surface,
        "diffusion_min_t_at_surface",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_min_dq_ratio_at_surface,
        "diffusion_min_dq_ratio_at_surface",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_dt_div_timescale,
        "diffusion_dt_div_timescale",
    )?;
    ctrl.try_usize_into(
        "controls",
        &mut d.diffusion_min_num_substeps,
        "diffusion_min_num_substeps",
        0,
    )?;
    ctrl.try_usize_into(
        "controls",
        &mut d.diffusion_max_iters_per_substep,
        "diffusion_max_iters_per_substep",
        1,
    )?;
    ctrl.try_usize_into(
        "controls",
        &mut d.diffusion_max_retries_per_substep,
        "diffusion_max_retries_per_substep",
        1,
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_tol_correction_max,
        "diffusion_tol_correction_max",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_tol_correction_norm,
        "diffusion_tol_correction_norm",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_min_x_hard_limit,
        "diffusion_min_x_hard_limit",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_x_total_atol,
        "diffusion_x_total_atol",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_x_total_rtol,
        "diffusion_x_total_rtol",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_upwind_abs_v_limit,
        "diffusion_upwind_abs_v_limit",
    )?;
    ctrl.try_f64_into("controls", &mut d.diffusion_v_max, "diffusion_v_max")?;
    ctrl.try_f64_into(
        "controls",
        &mut d.d_mix_ignore_diffusion,
        "d_mix_ignore_diffusion",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_gamma_full_off,
        "diffusion_gamma_full_off",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_gamma_full_on,
        "diffusion_gamma_full_on",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_t_full_on,
        "diffusion_t_full_on",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_t_full_off,
        "diffusion_t_full_off",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.diffusion_calculates_ionization,
        "diffusion_calculates_ionization",
    )?;
    ctrl.try_usize_into(
        "controls",
        &mut d.diffusion_nsmooth_typical_charge,
        "diffusion_nsmooth_typical_charge",
        0,
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_sig_factor,
        "diffusion_sig_factor",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_gt_factor,
        "diffusion_gt_factor",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_ad_dm_full_on,
        "diffusion_ad_dm_full_on",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_ad_dm_full_off,
        "diffusion_ad_dm_full_off",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_ad_boost_factor,
        "diffusion_ad_boost_factor",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_vlimit_dm_full_on,
        "diffusion_vlimit_dm_full_on",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_vlimit_dm_full_off,
        "diffusion_vlimit_dm_full_off",
    )?;
    ctrl.try_f64_into("controls", &mut d.diffusion_vlimit, "diffusion_vlimit")?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_min_t_for_radaccel,
        "diffusion_min_t_for_radaccel",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_max_t_for_radaccel,
        "diffusion_max_t_for_radaccel",
    )?;
    ctrl.try_isize_into(
        "controls",
        &mut d.diffusion_min_z_for_radaccel,
        "diffusion_min_z_for_radaccel",
    )?;
    ctrl.try_isize_into(
        "controls",
        &mut d.diffusion_max_z_for_radaccel,
        "diffusion_max_z_for_radaccel",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.diffusion_screening_for_radaccel,
        "diffusion_screening_for_radaccel",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.diffusion_use_full_net,
        "diffusion_use_full_net",
    )?;
    ctrl.try_usize_into(
        "controls",
        &mut d.diffusion_num_classes,
        "diffusion_num_classes",
        1,
    )?;
    if d.diffusion_num_classes > MAX_NUM_DIFFUSION_CLASSES {
        return Err(StarError::InvalidInput(format!(
            "diffusion_num_classes ({}) exceeds max_num_diffusion_classes ({MAX_NUM_DIFFUSION_CLASSES})",
            d.diffusion_num_classes
        )));
    }
    apply_indexed_str_slots(
        ctrl,
        "diffusion_class_representative",
        &mut d.diffusion_class_representative,
    )?;
    apply_indexed_f64_slots(
        ctrl,
        "diffusion_class_a_max",
        &mut d.diffusion_class_a_max,
        0.0,
    )?;
    apply_indexed_f64_slots(
        ctrl,
        "diffusion_class_typical_charge",
        &mut d.diffusion_class_typical_charge,
        0.0,
    )?;
    apply_indexed_f64_slots(
        ctrl,
        "diffusion_class_factor",
        &mut d.diffusion_class_factor,
        1.0,
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.diffusion_use_isolve,
        "diffusion_use_isolve",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_rtol_for_isolve,
        "diffusion_rtol_for_isolve",
    )?;
    ctrl.try_f64_into(
        "controls",
        &mut d.diffusion_atol_for_isolve,
        "diffusion_atol_for_isolve",
    )?;
    ctrl.try_usize_into(
        "controls",
        &mut d.diffusion_maxsteps_for_isolve,
        "diffusion_maxsteps_for_isolve",
        0,
    )?;
    // `star/private/ctrls_io.f90:1807`, `star/private/diffusion.f90:482`,
    // `num/public/num_solvers_options.dek:7-24`: Fortran `==` pads trailing
    // blanks only and is case-sensitive.
    if let Some(solver_name) =
        ctrl.try_get_str("controls", "diffusion_isolve_solver")?
    {
        match solver_name.trim_end() {
            "ros2_solver" | "rose2_solver" | "ros3p_solver"
            | "ros3pl_solver" | "rodas3_solver" | "rodas4_solver"
            | "rodasp_solver" => {
                solver_name
                    .trim_end()
                    .clone_into(&mut d.diffusion_isolve_solver);
            }
            other => {
                return Err(StarError::InvalidInput(format!(
                    "unrecognized '&controls::diffusion_isolve_solver' = '{other}'"
                )));
            }
        }
    }
    ctrl.try_isize_into(
        "controls",
        &mut d.diffusion_dump_call_number,
        "diffusion_dump_call_number",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.phase_separation_mixing_use_brunt,
        "phase_separation_mixing_use_brunt",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.phase_separation_no_diffusion,
        "phase_separation_no_diffusion",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.use_other_diffusion,
        "use_other_diffusion",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.use_other_diffusion_factor,
        "use_other_diffusion_factor",
    )?;
    ctrl.try_bool_into(
        "controls",
        &mut d.use_other_diffusion_coefficients,
        "use_other_diffusion_coefficients",
    )?;
    Ok(())
}
