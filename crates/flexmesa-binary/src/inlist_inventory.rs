// SPDX-License-Identifier: LGPL-3.0-or-later
//! Authoritative inventory and validation helpers for `&binary_job`,
//! `&binary_controls`, and headless `&pgbinary` namelist sections
//! (`DECISIONS #25`).

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;

use flexmesa_schema::{
    ControlValidationReport, copy_cleaned_case_dir, is_include_key,
    upstream_section_keys,
};
use flexmesa_star::{
    SectionControls, StarError, supported_star_control_inventory,
};

/// Supported `&binary_job` keys.
pub const BINARY_JOB_SUPPORTED: &[&str] = &[
    "binary_history_columns_file",
    "change_ignore_rlof_flag",
    "change_initial_ignore_rlof_flag",
    "change_initial_m1",
    "change_initial_m2",
    "change_initial_period_eccentricity",
    "change_initial_point_mass_i",
    "change_initial_separation_eccentricity",
    "change_m1",
    "change_m2",
    "change_period_eccentricity",
    "change_point_mass_i",
    "change_separation_eccentricity",
    "evolve_both_stars",
    "inlist_names",
    "new_eccentricity",
    "new_ignore_rlof_flag",
    "new_m1",
    "new_m2",
    "new_period",
    "new_point_mass_i",
    "new_separation",
];

/// Headless `&binary_job` GUI flags (`DECISIONS #25`).
pub const BINARY_JOB_HEADLESS_GUI: &[&str] = &["pgbinary_flag"];

/// `&binary_job` keys whose only upstream use is to gate text written to the
/// terminal (`DECISIONS #26`).
///
/// Upstream use at MESA `26.4.1`:
/// - `show_binary_log_description_at_start`: `binary/private/run_binary_support.f90:313`.
pub const BINARY_JOB_TERMINAL_OUTPUT: &[&str] =
    &["show_binary_log_description_at_start"];

/// `&binary_controls` keys whose only upstream use is to gate text written to
/// the terminal (`DECISIONS #26`).
///
/// Upstream uses at MESA `26.4.1`:
/// - `terminal_interval`: `binary/private/run_binary_support.f90:671,678-679`.
/// - `write_header_frequency`: `binary/private/run_binary_support.f90:678-679`.
pub const BINARY_CONTROLS_TERMINAL_OUTPUT: &[&str] =
    &["terminal_interval", "write_header_frequency"];

/// Supported `&binary_controls` keys.
pub const BINARY_CONTROLS_SUPPORTED: &[&str] = &[
    "accretion_powered_irradiation",
    "accretor_overflow_terminate",
    "anomaly_steps",
    "append_to_star_history",
    "change_factor_fraction",
    "change_factor_increase",
    "circ_type_1",
    "circ_type_2",
    "col_depth_for_eps_extra",
    "cur_mdot_frac",
    "do_enhance_wind_1",
    "do_enhance_wind_2",
    "do_initial_orbit_sync_1",
    "do_initial_orbit_sync_2",
    "do_j_accretion",
    "do_jdot_gr",
    "do_jdot_ls",
    "do_jdot_mb",
    "do_jdot_missing_wind",
    "do_jdot_ml",
    "do_tidal_circ",
    "do_tidal_sync",
    "do_wind_mass_transfer_1",
    "do_wind_mass_transfer_2",
    "dt_reduction_factor_for_j",
    "dt_softening_factor",
    "f_sync_switch_from_rot_defor",
    "f_sync_switch_lim",
    "f_sync_switch_width",
    "fa",
    "fa_hard",
    "fdm",
    "fdm_hard",
    "fe",
    "fe_hard",
    "fe_limit",
    "fj",
    "fj_hard",
    "fm",
    "fm_hard",
    "fm_limit",
    "fr",
    "fr_dt_limit",
    "fr_hard",
    "fr_limit",
    "ftid_1",
    "ftid_2",
    "history_interval",
    "history_name",
    "implicit_lambda",
    "implicit_scheme_tiny_factor",
    "implicit_scheme_tolerance",
    "include_accretor_mb",
    "initial_bh_spin",
    "initial_change_factor",
    "initial_eccentricity",
    "initial_period_in_days",
    "initial_separation_in_rsuns",
    "irrad_flux_at_std_distance",
    "jdot_mb_mass_frac_for_scale",
    "jdot_mb_max_qconv_core",
    "jdot_mb_max_qconv_env",
    "jdot_mb_min_qconv_env",
    "jdot_mb_qlim_for_check_conv_env",
    "jdot_mb_qlim_for_check_rad_core",
    "jdot_mb_scale_for_low_qconv_env",
    "jdot_multiplier",
    "keep_donor_fixed",
    "keep_mb_on",
    "limit_retention_by_mdot_edd",
    "log_directory",
    "m1",
    "m2",
    "magnetic_braking_gamma",
    "mass_transfer_alpha",
    "mass_transfer_beta",
    "mass_transfer_delta",
    "mass_transfer_gamma",
    "max_abs_edot_enhance",
    "max_abs_edot_tidal",
    "max_change_factor",
    "max_eccentricity",
    "max_explicit_abs_mdot",
    "max_f_irr",
    "max_implicit_abs_mdot",
    "max_tries_to_achieve",
    "max_wind_transfer_fraction_1",
    "max_wind_transfer_fraction_2",
    "mdot_limit_donor_switch",
    "mdot_scheme",
    "min_change_factor",
    "min_eccentricity",
    "min_mdot_for_implicit",
    "num_tries_for_increase_change_factor",
    "report_rlo_solver_progress",
    "roche_min_mdot",
    "solver_type",
    "starting_mdot",
    "std_distance_for_irradiation",
    "sync_mode_1",
    "sync_mode_2",
    "sync_type_1",
    "sync_type_2",
    "terminate_if_initial_overflow",
    "terminate_if_l2_overflow",
    "tidal_reduction",
    "time_delta_coeff",
    "tout_b_wind_1",
    "tout_b_wind_2",
    "use_accretor_luminosity_for_irrad",
    "use_eccentricity_enhancement",
    "use_es_opacity_for_mdot_edd",
    "use_radiation_corrected_transfer_rate",
    "use_this_for_mdot_edd",
    "use_this_for_mdot_edd_eta",
    "use_tidal_deformation",
    "varcontrol_case_a",
    "varcontrol_case_b",
    "varcontrol_ms",
    "varcontrol_post_ms",
    "wind_bh_alpha_1",
    "wind_bh_alpha_2",
    "wind_bh_beta_1",
    "wind_bh_beta_2",
];

/// Pre-hashed supported control-key sets for binary namelists.
#[derive(Clone, Debug)]
pub struct BinaryControlInventory {
    /// Supported `&binary_job` keys.
    pub binary_job_supported: HashSet<&'static str>,
    /// Headless GUI `&binary_job` flags (`pgbinary_flag`).
    pub binary_job_headless_gui_flags: HashSet<&'static str>,
    /// `&binary_job` keys that only write to the terminal (`DECISIONS #26`).
    pub binary_job_terminal_output: HashSet<&'static str>,
    /// Supported `&binary_controls` keys.
    pub binary_controls_supported: HashSet<&'static str>,
    /// `&binary_controls` keys that only write to the terminal (`DECISIONS #26`).
    pub binary_controls_terminal_output: HashSet<&'static str>,
}

static BINARY_INVENTORY: OnceLock<BinaryControlInventory> = OnceLock::new();

/// Returns the cached [`BinaryControlInventory`] for binary namelist sections.
#[must_use]
pub fn supported_binary_control_inventory() -> &'static BinaryControlInventory {
    BINARY_INVENTORY.get_or_init(|| BinaryControlInventory {
        binary_job_supported: BINARY_JOB_SUPPORTED.iter().copied().collect(),
        binary_job_headless_gui_flags: BINARY_JOB_HEADLESS_GUI
            .iter()
            .copied()
            .collect(),
        binary_job_terminal_output: BINARY_JOB_TERMINAL_OUTPUT
            .iter()
            .copied()
            .collect(),
        binary_controls_supported: BINARY_CONTROLS_SUPPORTED
            .iter()
            .copied()
            .collect(),
        binary_controls_terminal_output: BINARY_CONTROLS_TERMINAL_OUTPUT
            .iter()
            .copied()
            .collect(),
    })
}

/// Validates parsed binary namelist sections against `BINARY_*_SUPPORTED`
/// and MESA `26.4.1` defaults (including `pgbinary_flag`-gated headless
/// `&pgbinary` validation, matching `binary/private/pgbinary_full.f90:132`).
#[must_use]
pub fn validate_binary_sections<S: ::std::hash::BuildHasher>(
    sections: &HashMap<String, SectionControls, S>,
) -> ControlValidationReport {
    let inv = supported_binary_control_inventory();
    let empty: HashSet<&'static str> = HashSet::new();
    let mut report = ControlValidationReport::default();

    let pgbinary_active = match sections
        .get("binary_job")
        .map(|s| s.try_get_bool("binary_job", "pgbinary_flag"))
    {
        Some(Ok(v)) => v.unwrap_or(false),
        Some(Err(_)) => {
            // A non-boolean `pgbinary_flag` is rejected like any other
            // malformed control instead of being read as `.false.`.
            report.unknown_controls.push((
                String::from("binary_job"),
                String::from("pgbinary_flag"),
            ));
            false
        }
        None => false,
    };

    for (sec, ctrls) in sections {
        let keys = ctrls.all_keys();
        match sec.as_str() {
            "pgbinary" => {
                report.validate_headless_plotting_section_with_active(
                    sec,
                    &keys,
                    pgbinary_active,
                );
            }
            "binary_job" => {
                report.validate_implemented_section_with_terminal_output(
                    sec,
                    &keys,
                    &inv.binary_job_supported,
                    &empty,
                    &inv.binary_job_headless_gui_flags,
                    &inv.binary_job_terminal_output,
                );
            }
            "binary_controls" => {
                report.validate_implemented_section_with_terminal_output(
                    sec,
                    &keys,
                    &inv.binary_controls_supported,
                    &empty,
                    &empty,
                    &inv.binary_controls_terminal_output,
                );
            }
            other => {
                report.validate_other_section(other);
            }
        }
    }
    report
}

/// Copies a binary test case directory `src_case` to `dst_case` and strips
/// unsupported sections/keys so strict `#147` loading passes cleanly.
///
/// # Errors
/// Returns [`StarError::IoError`] if copying or rewriting files fails.
pub fn copy_cleaned_binary_case(
    src_case: &Path,
    dst_case: &Path,
) -> Result<(), StarError> {
    let star_inv = supported_star_control_inventory();
    let bin_inv = supported_binary_control_inventory();
    copy_cleaned_case_dir(
        src_case,
        dst_case,
        &|sec| {
            matches!(
                sec,
                "star_job"
                    | "eos"
                    | "kap"
                    | "controls"
                    | "pgstar"
                    | "binary_job"
                    | "binary_controls"
                    | "pgbinary"
            )
        },
        &|sec, key| match sec {
            "star_job" => {
                is_include_key(sec, key)
                    || star_inv.star_job_supported.contains(key)
                    || star_inv.star_job_headless_gui_flags.contains(key)
                    || star_inv.star_job_terminal_output.contains(key)
            }
            "eos" => {
                is_include_key(sec, key) || star_inv.eos_supported.contains(key)
            }
            "kap" => {
                is_include_key(sec, key) || star_inv.kap_supported.contains(key)
            }
            "controls" => {
                is_include_key(sec, key)
                    || star_inv.controls_supported.contains(key)
                    || star_inv.controls_terminal_output.contains(key)
            }
            "binary_job" => {
                is_include_key(sec, key)
                    || bin_inv.binary_job_supported.contains(key)
                    || bin_inv.binary_job_headless_gui_flags.contains(key)
                    || bin_inv.binary_job_terminal_output.contains(key)
            }
            "binary_controls" => {
                is_include_key(sec, key)
                    || bin_inv.binary_controls_supported.contains(key)
                    || bin_inv.binary_controls_terminal_output.contains(key)
            }
            "pgstar" | "pgbinary" => {
                is_include_key(sec, key)
                    || upstream_section_keys(sec)
                        .is_some_and(|set| set.contains(key))
            }
            _ => false,
        },
    )
    .map_err(|e| StarError::IoError(e.to_string()))
}
