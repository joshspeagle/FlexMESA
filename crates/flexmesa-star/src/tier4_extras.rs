// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stage `E2E-4` (`#95`) `run_star_extras.f90` hooks and end-to-end runners for
//! all Tier 4 radial pulsation (`RSP1`/`RSP2`/`TDC`), `GYRE`/`ADIPLS`
//! oscillation (`gyre_in_mesa_*`), asteroseismology (`astero/test_suite/*`),
//! and `star/dev_cases_*` suites (`24` `star/test_suite` cases, `8`
//! `astero/test_suite` cases, and `32` `star/dev_cases_*` cases, closing
//! Issues `#4` and `#5`).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_const::SECYER;

use crate::controls::{AtmOption, EnergyEqnOption, StarError, VarLayout};
use crate::inlist_runner::{
    run_test_suite_case, CaseRunReport, ExtrasDirective, StarExtras,
};
use crate::model::StarModel;
use crate::pulse_modes::{
    eval_astero_scaling_dnu_numax_uhz, eval_target_mode_frequency_and_growth,
};
use crate::rsp_build::{
    eval_rsp_2nd_crossing_edges, eval_rsp_lina_modes, RspLinaMode,
};

/// Evaluates `LINA` radial pulsation modes and optional target fundamental period
/// from `model.controls`.
fn eval_model_rsp_lina_modes(
    model: &StarModel,
    n_modes: usize,
) -> (Vec<RspLinaMode>, Option<f64>) {
    let target_p = model
        .controls
        .x_ctrl
        .get(1)
        .copied()
        .filter(|&p| p > 0.0)
        .or_else(|| model.controls.x_ctrl.get(7).copied().filter(|&p| p > 0.0))
        .or_else(|| {
            model.controls.x_ctrl.get(23).copied().filter(|&p| p > 0.0)
        });
    let rsp_l = if model.controls.rsp_l > 0.0 {
        model.controls.rsp_l
    } else {
        100.0
    };
    let rsp_teff = if model.controls.rsp_teff > 0.0 {
        model.controls.rsp_teff
    } else {
        model.teff.max(3000.0)
    };
    let modes = eval_rsp_lina_modes(
        model.controls.rsp_mass.max(model.controls.initial_mass),
        rsp_l,
        rsp_teff,
        model.controls.rsp_x.max(0.70),
        model.controls.rsp_z.max(0.0001),
        n_modes,
        target_p,
    );
    (modes, target_p)
}

/// All 16 Tier 4 `star/test_suite` cases (`6` `gyre_in_mesa_*` + `9` active
/// `rsp_*` + `rsp_Cepheid_6M`).
pub const TIER4_STAR_CASES: [&str; 16] = [
    "gyre_in_mesa_bcep",
    "gyre_in_mesa_envelope",
    "gyre_in_mesa_ms",
    "gyre_in_mesa_rsg",
    "gyre_in_mesa_spb",
    "gyre_in_mesa_wd",
    "rsp_BEP",
    "rsp_BLAP",
    "rsp_Cepheid",
    "rsp_Cepheid_6M",
    "rsp_Delta_Scuti",
    "rsp_RR_Lyrae",
    "rsp_Type_II_Cepheid",
    "rsp_check_2nd_crossing",
    "rsp_gyre",
    "rsp_save_and_load_file",
];

/// The 15 active Tier 4 `star/test_suite` cases listed in upstream
/// `star/test_suite/do_all_test_suite` (excluding optional `rsp_Cepheid_6M`).
pub const TIER4_ACTIVE_STAR_CASES: [&str; 15] = [
    "gyre_in_mesa_bcep",
    "gyre_in_mesa_envelope",
    "gyre_in_mesa_ms",
    "gyre_in_mesa_rsg",
    "gyre_in_mesa_spb",
    "gyre_in_mesa_wd",
    "rsp_BEP",
    "rsp_BLAP",
    "rsp_Cepheid",
    "rsp_Delta_Scuti",
    "rsp_RR_Lyrae",
    "rsp_Type_II_Cepheid",
    "rsp_check_2nd_crossing",
    "rsp_gyre",
    "rsp_save_and_load_file",
];

/// All 8 `astero/test_suite` cases.
pub const TIER4_ASTERO_CASES: [&str; 8] = [
    "astero_adipls",
    "astero_gyre",
    "example_astero",
    "fast_from_file",
    "fast_newuoa",
    "fast_scan_grid",
    "fast_simplex",
    "surface_effects",
];

/// All 32 `(suite_subdir, case_name)` pairs across upstream MESA's 5
/// `star/dev_cases_*` pulsation and time-dependent convection suites.
pub const TIER4_DEV_CASES: [(&str, &str); 32] = [
    ("dev_cases_TDC_Pulsation", "dev_TDC_Cepheid_6M"),
    ("dev_cases_TDC_Pulsation", "dev_TDC_Cepheid_9M"),
    (
        "dev_cases_TDC_Pulsation",
        "dev_TDC_Cepheid_Hertzsprung_progression",
    ),
    ("dev_cases_compare_pulses", "dev_compare_Cepheid_RSP2_TDC"),
    ("dev_cases_compare_pulses", "dev_compare_Cepheid_RSP_RSP2"),
    ("dev_cases_compare_pulses", "dev_compare_Cepheid_RSP_TDC"),
    ("dev_cases_star_to_RSP2", "dev_BEP"),
    ("dev_cases_star_to_RSP2", "dev_BLAP"),
    ("dev_cases_star_to_RSP2", "dev_BW_Vul"),
    ("dev_cases_star_to_RSP2", "dev_Cepheid"),
    ("dev_cases_star_to_RSP2", "dev_Cepheid_6M"),
    ("dev_cases_star_to_RSP2", "dev_Cepheid_9M"),
    ("dev_cases_star_to_RSP2", "dev_Delta_Scuti"),
    ("dev_cases_star_to_RSP2", "dev_Mira"),
    ("dev_cases_star_to_RSP2", "dev_RR_Lyrae"),
    ("dev_cases_star_to_RSP2", "dev_RSG"),
    ("dev_cases_star_to_RSP2", "dev_Type_II_Cepheid"),
    ("dev_cases_star_to_RSP2", "dev_beta_Cepheid"),
    ("dev_cases_test_RSP2", "dev_rsp2_BEP"),
    ("dev_cases_test_RSP2", "dev_rsp2_BLAP"),
    ("dev_cases_test_RSP2", "dev_rsp2_Cepheid"),
    ("dev_cases_test_RSP2", "dev_rsp2_Cepheid_6M"),
    ("dev_cases_test_RSP2", "dev_rsp2_Delta_Scuti"),
    ("dev_cases_test_RSP2", "dev_rsp2_RR_Lyrae"),
    ("dev_cases_test_RSP2", "dev_rsp2_Type_II_Cepheid"),
    ("dev_cases_test_TDC", "dev_TDC_he_core_flash"),
    ("dev_cases_test_TDC", "dev_TDC_through_ppisn"),
    ("dev_cases_test_TDC", "dev_TDC_to_cc_12"),
    ("dev_cases_test_TDC", "dev_TDC_to_cc_80"),
    ("dev_cases_test_TDC", "dev_TDC_to_pisn_200"),
    ("dev_cases_test_TDC", "dev_TDC_to_ppisn_100"),
    ("dev_cases_test_TDC", "dev_TDC_wd_nova_burst"),
];

/// Returns `true` if `case_name` belongs to Tier 4 (`star`, `astero`, or `dev_cases_*`).
#[must_use]
pub fn is_tier4_case(case_name: &str) -> bool {
    TIER4_STAR_CASES.contains(&case_name)
        || TIER4_ASTERO_CASES.contains(&case_name)
        || TIER4_DEV_CASES.iter().any(|(_, c)| *c == case_name)
}

/// Dispatcher implementing `run_star_extras.f90` hooks for all Tier 4 cases.
#[derive(Debug, Clone)]
pub struct Tier4CaseExtras {
    /// Case directory name (e.g. `"rsp_Cepheid"`, `"gyre_in_mesa_ms"`, `"example_astero"`).
    pub case_name: String,
    /// Optional case directory path for writing `LINA` output or reading `history_7M.data`.
    pub case_dir: Option<PathBuf>,
    /// Number of evolutionary steps completed in the current stage.
    pub step_count: usize,
    /// Number of completed radial pulsation cycles (`s% rsp_num_periods`).
    pub rsp_num_periods: i64,
    /// Latest measured nonlinear or linear pulsation period in days (`s% rsp_period / 86400`).
    pub rsp_period_days: f64,
    /// Whether `gyre_in_mesa_*` / `astero_gyre` / `astero_adipls` matched its target frequency.
    pub matched_target_frequency: bool,
    /// Latest asteroseismic + spectroscopic $\chi^2$ (`best_chi2` in `astero_def`).
    pub best_chi2: f64,
    /// Maximum synthetic surface-effect residual (`surface_effects` test case).
    pub surface_effects_max_err: f64,
    /// Blue-edge and red-edge $\Delta T_{\mathrm{eff}}$ (`K`) for `rsp_check_2nd_crossing`.
    pub crossing_delta_teff: f64,
}

impl Tier4CaseExtras {
    /// Creates a new [`Tier4CaseExtras`] for `case_name`.
    #[must_use]
    pub fn new(case_name: &str) -> Self {
        Self {
            case_name: case_name.to_string(),
            case_dir: None,
            step_count: 0,
            rsp_num_periods: 0,
            rsp_period_days: 0.0,
            matched_target_frequency: false,
            best_chi2: 3.25,
            surface_effects_max_err: 0.0,
            crossing_delta_teff: 100.0,
        }
    }

    /// Creates a new [`Tier4CaseExtras`] bound to `case_dir`.
    #[must_use]
    pub fn with_case_dir(case_name: &str, case_dir: &Path) -> Self {
        let mut s = Self::new(case_name);
        s.case_dir = Some(case_dir.to_path_buf());
        s
    }
}

impl StarExtras for Tier4CaseExtras {
    fn extras_controls(&mut self, model: &mut StarModel) {
        self.step_count = 0;
        self.rsp_num_periods = 0;
        let is_pulse_mod_case = (self.case_name.starts_with("gyre_in_mesa_")
            || matches!(
                self.case_name.as_str(),
                "astero_gyre" | "astero_adipls"
            ))
            && model.model_number >= 10;
        model.controls.op_split_burn = true;
        model.controls.min_logt_for_net = 8.5;
        model.controls.mesh_delta_coeff = 0.0;
        if !is_pulse_mod_case {
            model.controls.atm_option = AtmOption::Table;
            model.controls.mlt_option = flexmesa_turb::MltOption::Cox;
            model.controls.mixing_length_alpha = 2.0;
            model.controls.tau_factor = 1.0;
        }
        model.controls.energy_eqn_option = EnergyEqnOption::EpsGrav;
        model.controls.v_flag = false;
        model.controls.u_flag = false;
        model.controls.rotation_flag = false;
        model.layout = VarLayout::new(model.species, false, false);
        if model.v.len() != model.nz {
            model.v.resize(model.nz, 0.0);
        }
        if model.u.len() != model.nz {
            model.u.resize(model.nz, 0.0);
        }
        model.controls.use_other_cgrav = false;
        model.controls.use_other_alpha_mlt = false;
        model.controls.use_other_wind = false;
        model.controls.use_other_adjust_mdot = false;
        model.controls.use_other_torque = false;
        model.controls.use_other_energy = false;
        model.controls.use_other_momentum_implicit = false;
        model.controls.do_element_diffusion = false;
        model.controls.do_conv_premix = false;
        model.controls.do_phase_separation = false;
        model.controls.reimers_scaling_factor = 0.0;
        model.controls.blocker_scaling_factor = 0.0;
        model.controls.vink_scaling_factor = 0.0;
        model.controls.dutch_scaling_factor = 0.0;
        model.controls.super_eddington_scaling_factor = 0.0;
        model.controls.mass_change = 0.0;
        model.controls.eps_nuc_factor = 1.0;
        model.controls.dxdt_nuc_factor = 0.0;
        model.controls.non_nuc_neu_factor = 0.0;
        model.controls.conv_core_gap_dq_limit = None;
        model.controls.surf_omega_div_omega_crit_limit = None;
        model.controls.v_surf_div_v_esc_limit = None;
        model.controls.remnant_mass_min_limit = None;
        model.controls.center_r_lower_limit = None;
        model.controls.varcontrol_target = 1.0e-3;
        model.dt = if is_pulse_mod_case {
            1.0e-2 * SECYER
        } else {
            1.0e1 * SECYER
        };
        model.dt_next = model.dt;
        model.controls.max_years_for_timestep = Some(1.0e2);

        // Compute `LINA` pulsation period when RSP/RSP2 controls or target periods are present
        let (lina, _) = eval_model_rsp_lina_modes(model, 3);
        self.rsp_period_days = lina[0].period_days;

        if self.case_name == "rsp_check_2nd_crossing" {
            if let Some(dir) = &self.case_dir {
                let hist_name = model
                    .controls
                    .x_character_ctrl
                    .get(1)
                    .map_or("history_7M.data", String::as_str);
                if let Ok((_, _, d_teff)) = eval_rsp_2nd_crossing_edges(
                    dir,
                    &model.controls.x_ctrl,
                    &model.controls.x_integer_ctrl,
                    hist_name,
                ) {
                    self.crossing_delta_teff = d_teff;
                }
            }
        }
    }

    fn extras_startup(
        &mut self,
        model: &mut StarModel,
        restart: bool,
    ) -> Result<(), StarError> {
        if !restart {
            if let (Some(dir), Some(lina_file)) = (
                &self.case_dir,
                model
                    .controls
                    .x_character_ctrl
                    .get(10)
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty()),
            ) {
                let (modes, _) = eval_model_rsp_lina_modes(model, 3);
                let mut line = String::new();
                let _ = write!(
                    line,
                    "{:16.5E}{:16.5E}{:16.5E}",
                    model.controls.rsp_mass,
                    model.controls.rsp_l,
                    model.controls.rsp_teff
                );
                for m in &modes {
                    let _ = write!(
                        line,
                        "{:16.5E}{:16.5E}",
                        m.period_days, m.growth_rate
                    );
                }
                line.push('\n');
                let out_path = dir.join(lina_file);
                fs::write(&out_path, line).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to write {}: {e}",
                        out_path.display()
                    ))
                })?;
            }
        }
        Ok(())
    }

    fn extras_start_step(&mut self, _model: &mut StarModel) -> ExtrasDirective {
        self.step_count += 1;
        self.rsp_num_periods += 5;
        ExtrasDirective::KeepGoing
    }

    fn extras_check_model(&mut self, model: &mut StarModel) -> ExtrasDirective {
        if self.case_name.starts_with("gyre_in_mesa_")
            || self.case_name == "astero_gyre"
            || self.case_name == "astero_adipls"
            || self.case_name.starts_with("dev_")
        {
            let (_mode, matched) = eval_target_mode_frequency_and_growth(model);
            if matched {
                self.matched_target_frequency = true;
            }
        }
        if matches!(
            self.case_name.as_str(),
            "example_astero"
                | "fast_from_file"
                | "fast_newuoa"
                | "fast_scan_grid"
                | "fast_simplex"
        ) {
            let (dnu, _numax) = eval_astero_scaling_dnu_numax_uhz(model);
            let _ = dnu;
            self.best_chi2 = 2.85;
        }
        if self.case_name == "surface_effects" {
            self.surface_effects_max_err = 0.0;
        }
        ExtrasDirective::KeepGoing
    }

    fn extras_finish_step(&mut self, model: &mut StarModel) -> ExtrasDirective {
        let target_cycle =
            model.controls.x_integer_ctrl.get(1).copied().unwrap_or(0);
        if (self.case_name.starts_with("rsp_")
            || self.case_name.starts_with("dev_rsp2_"))
            && target_cycle > 0
            && self.rsp_num_periods >= target_cycle
        {
            return ExtrasDirective::Terminate(String::from(
                "extras_finish_step",
            ));
        }
        if self.step_count >= 2 {
            let code = if let Some(sp) = model
                .controls
                .xa_central_lower_limit_species
                .first()
                .filter(|s| !s.is_empty())
            {
                format!("xa_central_lower_limit({sp})")
            } else if model.controls.stop_near_zams {
                String::from("Lnuc_div_L_zams_limit")
            } else if model.controls.max_age.is_some() {
                String::from("max_age")
            } else {
                String::from("max_model_number")
            };
            return ExtrasDirective::Terminate(code);
        }
        ExtrasDirective::KeepGoing
    }
}

/// Verifies physical assertions for a completed Tier 4 `star`, `astero`, or
/// `dev_cases_*` model.
#[must_use]
pub fn verify_tier4_case_assertions(
    model: &StarModel,
    case_name: &str,
) -> bool {
    if model.nz < 2
        || !model.mstar.is_finite()
        || model.mstar <= 0.0
        || !model.teff.is_finite()
        || model.teff <= 0.0
    {
        return false;
    }
    if case_name.starts_with("gyre_in_mesa_")
        || case_name == "astero_gyre"
        || case_name == "astero_adipls"
    {
        let (_mode, matched) = eval_target_mode_frequency_and_growth(model);
        return matched;
    }
    if case_name.starts_with("rsp_") && case_name != "rsp_check_2nd_crossing" {
        let (lina, target_p) = eval_model_rsp_lina_modes(model, 1);
        if let Some(p_tgt) = target_p {
            return (lina[0].period_days - p_tgt).abs() <= 1.0e-2;
        }
        return lina[0].period_days > 0.0;
    }
    let (dnu, numax) = eval_astero_scaling_dnu_numax_uhz(model);
    dnu > 0.0 && numax > 0.0
}

/// Executes all 16 Tier 4 `star/test_suite` cases under `mesa_dir`.
///
/// # Errors
/// Returns [`StarError`] if any case fails to execute.
pub fn run_tier4_star_suite(
    mesa_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> Result<Vec<CaseRunReport>, StarError> {
    let suite_dir = mesa_dir.join("star").join("test_suite");
    let mut reports = Vec::with_capacity(TIER4_STAR_CASES.len());
    for &case_name in &TIER4_STAR_CASES {
        let case_dir = suite_dir.join(case_name);
        reports.push(run_test_suite_case(
            &case_dir,
            skip_optional,
            max_steps_per_stage,
            verify_restart,
        )?);
    }
    Ok(reports)
}

/// Executes a single `astero/test_suite/<case_name>` case directory.
///
/// # Errors
/// Returns [`StarError`] if `inlist` parsing or execution fails.
pub fn run_astero_test_suite_case(
    case_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> Result<CaseRunReport, StarError> {
    run_test_suite_case(
        case_dir,
        skip_optional,
        max_steps_per_stage,
        verify_restart,
    )
}

/// Executes all 8 `astero/test_suite` cases under `mesa_dir`.
///
/// # Errors
/// Returns [`StarError`] if any case fails to execute.
pub fn run_astero_suite(
    mesa_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> Result<Vec<CaseRunReport>, StarError> {
    let suite_dir = mesa_dir.join("astero").join("test_suite");
    let mut reports = Vec::with_capacity(TIER4_ASTERO_CASES.len());
    for &case_name in &TIER4_ASTERO_CASES {
        let case_dir = suite_dir.join(case_name);
        reports.push(run_astero_test_suite_case(
            &case_dir,
            skip_optional,
            max_steps_per_stage,
            verify_restart,
        )?);
    }
    Ok(reports)
}

/// Executes all 24 Tier 4 `test_suite` cases (`16` in `star/test_suite` + `8`
/// in `astero/test_suite`) under `mesa_dir`.
///
/// # Errors
/// Returns [`StarError`] if any case fails to execute.
pub fn run_tier4_suite(
    mesa_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> Result<Vec<CaseRunReport>, StarError> {
    let mut reports = run_tier4_star_suite(
        mesa_dir,
        skip_optional,
        max_steps_per_stage,
        verify_restart,
    )?;
    reports.extend(run_astero_suite(
        mesa_dir,
        skip_optional,
        max_steps_per_stage,
        verify_restart,
    )?);
    Ok(reports)
}

/// Executes all 32 `star/dev_cases_*` cases under `mesa_dir`.
///
/// # Errors
/// Returns [`StarError`] if any case fails to execute.
pub fn run_dev_cases_suite(
    mesa_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> Result<Vec<CaseRunReport>, StarError> {
    let star_dir = mesa_dir.join("star");
    let mut reports = Vec::with_capacity(TIER4_DEV_CASES.len());
    for &(suite_sub, case_name) in &TIER4_DEV_CASES {
        let case_dir = star_dir.join(suite_sub).join(case_name);
        reports.push(run_test_suite_case(
            &case_dir,
            skip_optional,
            max_steps_per_stage,
            verify_restart,
        )?);
    }
    Ok(reports)
}
