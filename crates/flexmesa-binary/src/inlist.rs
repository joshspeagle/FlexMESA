// SPDX-License-Identifier: LGPL-3.0-or-later
//! Recursive `&binary_job` and `&binary_controls` Fortran namelist loader.
//!
//! Corresponds to `binary/private/binary_job_ctrls_io.f90`,
//! `binary/private/binary_ctrls_io.f90`, `binary/defaults/binary_job.defaults`,
//! and `binary/defaults/binary_controls.defaults`, resolving nested
//! `read_extra_binary_{job,controls}_inlist(1..5)` chains (such as
//! `inlist -> inlist_project -> inlist_extra` in `double_bh`).

use std::path::Path;

use flexmesa_star::{
    InlistBundle, InlistLoadOptions, SectionControls, StarError,
};

pub use crate::inlist_inventory::{
    copy_cleaned_binary_case, supported_binary_control_inventory,
    validate_binary_sections, BinaryControlInventory,
    BINARY_CONTROLS_SUPPORTED, BINARY_JOB_HEADLESS_GUI, BINARY_JOB_SUPPORTED,
};

/// Parsed `&binary_job` and `&binary_controls` settings corresponding to
/// `binary/defaults/binary_job.defaults` and `binary_controls.defaults`.
#[derive(Debug, Clone)]
pub struct BinaryInlistControls {
    // --- `&binary_job` controls ---
    /// Stellar inlist filenames `[inlist1, inlist2]`.
    pub inlist_names: [String; 2],
    /// Custom `binary_history_columns.list` path (`binary_history_columns_file`, `binary_job.defaults:31`).
    pub binary_history_columns_file: String,
    /// Evolve both components simultaneously (`evolve_both_stars`).
    pub evolve_both_stars: bool,
    /// Override initial `ignore_rlof_flag` (`change_initial_ignore_rlof_flag`).
    pub change_initial_ignore_rlof_flag: bool,
    /// Override `ignore_rlof_flag` on load (`change_ignore_rlof_flag`).
    pub change_ignore_rlof_flag: bool,
    /// Target `ignore_rlof_flag` value (`new_ignore_rlof_flag`).
    pub new_ignore_rlof_flag: bool,

    // --- Binary history output controls (`binary_controls.defaults:65-122`) ---
    /// Output directory for `binary_history.data` (`log_directory`, default `"."`).
    pub log_directory: String,
    /// Output filename for binary history (`history_name`, default `"binary_history.data"`).
    pub history_name: String,
    /// Step interval between binary history writes (`history_interval`, default `5`).
    pub history_interval: usize,
    /// Whether to append binary history columns to each component star's `history.data`
    /// (`append_to_star_history`, default `true`).
    pub append_to_star_history: bool,

    // --- Initial orbital & component specifications ---
    /// Initial primary mass $M_1$ in $M_\odot$ (`m1`).
    pub m1_msun: f64,
    /// Initial secondary mass $M_2$ in $M_\odot$ (`m2`).
    pub m2_msun: f64,
    /// Initial orbital period in days (`initial_period_in_days`; `<= 0` uses separation).
    pub initial_period_in_days: f64,
    /// Initial orbital separation in $R_\odot$ (`initial_separation_in_Rsuns`).
    pub initial_separation_in_rsuns: f64,
    /// Initial orbital eccentricity (`initial_eccentricity`).
    pub initial_eccentricity: f64,
    /// Lower clamp on eccentricity (`min_eccentricity`).
    pub min_eccentricity: f64,
    /// Upper clamp on eccentricity (`max_eccentricity`).
    pub max_eccentricity: f64,
    /// Ceiling on $|\dot{e}_{\mathrm{tidal}}|$ in $\mathrm{s}^{-1}$ (`max_abs_edot_tidal`).
    pub max_abs_edot_tidal: f64,
    /// Initial dimensionless Kerr spin parameter (`initial_bh_spin`, clamped to `[0, 1]`).
    pub initial_bh_spin: f64,

    // --- Donor switching & explicit/implicit RLOF solver controls ---
    /// Keep star 1 as donor outside `"contact"` (`keep_donor_fixed`).
    pub keep_donor_fixed: bool,
    /// Maximum $|\dot{M}_{\mathrm{tr}}|$ ($M_\odot\,\mathrm{yr}^{-1}$) allowing switch (`mdot_limit_donor_switch`).
    pub mdot_limit_donor_switch: f64,
    /// Mass-transfer scheme (`"Ritter"`, `"Kolb"`, `"roche_lobe"`, `"contact"`, `"Arras"`).
    pub mdot_scheme: String,
    /// Maximum implicit RLOF tries per step (`max_tries_to_achieve`; `<= 0` selects explicit).
    pub max_tries_to_achieve: i64,
    /// Implicit root-solver strategy (`"cubic"`, `"bisect"`, `"both"`).
    pub solver_type: String,
    /// Implicit RLOF objective tolerance $\xi$ (`implicit_scheme_tolerance`).
    pub implicit_scheme_tolerance: f64,
    /// Relative bracket width exit factor (`implicit_scheme_tiny_factor`).
    pub implicit_scheme_tiny_factor: f64,
    /// Initial step multiplier for unbracketed search (`initial_change_factor`).
    pub initial_change_factor: f64,
    /// Step-to-step change-factor adaptation fraction (`change_factor_fraction`).
    pub change_factor_fraction: f64,
    /// Relaxation weight for `change_factor` updates (`implicit_lambda`).
    pub implicit_lambda: f64,
    /// Upper clamp on `change_factor` (`max_change_factor`).
    pub max_change_factor: f64,
    /// Lower clamp on `change_factor` (`min_change_factor`).
    pub min_change_factor: f64,
    /// Detachment threshold in $M_\odot\,\mathrm{yr}^{-1}$ (`roche_min_mdot`).
    pub roche_min_mdot: f64,
    /// Minimum explicit $|\dot{M}|$ ($M_\odot\,\mathrm{yr}^{-1}$) for implicit RLOF (`min_mdot_for_implicit`).
    pub min_mdot_for_implicit: f64,
    /// Ceiling on implicit $|\dot{M}|$ in $M_\odot\,\mathrm{yr}^{-1}$ (`max_implicit_abs_mdot`).
    pub max_implicit_abs_mdot: f64,
    /// Starting $|\dot{M}|$ guess in $M_\odot\,\mathrm{yr}^{-1}$ (`starting_mdot`).
    pub starting_mdot: f64,
    /// Old-step weight in explicit mass-transfer smoothing (`cur_mdot_frac`, default `0.5`).
    pub cur_mdot_frac: f64,
    /// Ceiling on explicit $|\dot{M}|$ in $M_\odot\,\mathrm{yr}^{-1}$ (`max_explicit_abs_mdot`).
    pub max_explicit_abs_mdot: f64,
    /// Accretor relative overflow termination threshold (`accretor_overflow_terminate`).
    pub accretor_overflow_terminate: f64,
    /// Terminate if overflowing on step 1 (`terminate_if_initial_overflow`).
    pub terminate_if_initial_overflow: bool,
    /// Terminate on Marchant et al. (2016) $L_2$ overflow (`terminate_if_L2_overflow`).
    pub terminate_if_l2_overflow: bool,

    // --- Non-conservative retention, Eddington & wind-transfer controls ---
    /// Fraction lost isotropically from donor (`mass_transfer_alpha`).
    pub mass_transfer_alpha: f64,
    /// Fraction lost isotropically from accretor vicinity (`mass_transfer_beta`).
    pub mass_transfer_beta: f64,
    /// Fraction lost from circumbinary coplanar toroid (`mass_transfer_delta`).
    pub mass_transfer_delta: f64,
    /// Circumbinary toroid radius factor $\gamma$ (`mass_transfer_gamma`).
    pub mass_transfer_gamma: f64,
    /// Cap point-mass retention at $\dot{M}_{\mathrm{Edd}}$ (`limit_retention_by_mdot_edd`).
    pub limit_retention_by_mdot_edd: bool,
    /// Use $0.2(1 + X_{\mathrm{surf}})$ for $\dot{M}_{\mathrm{Edd}}$ (`use_es_opacity_for_mdot_edd`).
    pub use_es_opacity_for_mdot_edd: bool,
    /// Fixed radiative efficiency override if `> 0` (`use_this_for_mdot_edd_eta`).
    pub use_this_for_mdot_edd_eta: f64,
    /// Subtract radiated rest-mass fraction $\eta$ (`use_radiation_corrected_transfer_rate`).
    pub use_radiation_corrected_transfer_rate: bool,
    /// Fixed $\dot{M}_{\mathrm{Edd}}$ override in $M_\odot\,\mathrm{yr}^{-1}$ if `> 0` (`use_this_for_mdot_edd`).
    pub use_this_for_mdot_edd: f64,
    /// Enable Tout & Eggleton tidal wind enhancement on star 1 (`do_enhance_wind_1`).
    pub do_enhance_wind_1: bool,
    /// Enable Tout & Eggleton tidal wind enhancement on star 2 (`do_enhance_wind_2`).
    pub do_enhance_wind_2: bool,
    /// Tout & Eggleton $B_{\mathrm{wind}}$ for star 1 (`tout_B_wind_1`).
    pub tout_b_wind_1: f64,
    /// Tout & Eggleton $B_{\mathrm{wind}}$ for star 2 (`tout_B_wind_2`).
    pub tout_b_wind_2: f64,
    /// Enable Bondi-Hoyle wind transfer from star 1 (`do_wind_mass_transfer_1`).
    pub do_wind_mass_transfer_1: bool,
    /// Enable Bondi-Hoyle wind transfer from star 2 (`do_wind_mass_transfer_2`).
    pub do_wind_mass_transfer_2: bool,
    /// Bondi-Hoyle $\alpha_{\mathrm{BH}}$ for star 1 (`wind_BH_alpha_1`).
    pub wind_bh_alpha_1: f64,
    /// Bondi-Hoyle $\beta_{\mathrm{BH}}$ for star 1 (`wind_BH_beta_1`).
    pub wind_bh_beta_1: f64,
    /// Maximum wind transfer fraction for star 1 (`max_wind_transfer_fraction_1`).
    pub max_wind_transfer_fraction_1: f64,
    /// Bondi-Hoyle $\alpha_{\mathrm{BH}}$ for star 2 (`wind_BH_alpha_2`).
    pub wind_bh_alpha_2: f64,
    /// Bondi-Hoyle $\beta_{\mathrm{BH}}$ for star 2 (`wind_BH_beta_2`).
    pub wind_bh_beta_2: f64,
    /// Maximum wind transfer fraction for star 2 (`max_wind_transfer_fraction_2`).
    pub max_wind_transfer_fraction_2: f64,

    // --- Orbital angular-momentum loss (`do_jdot_*`) & tidal controls ---
    /// Enable gravitational-wave $\dot{J}_{\mathrm{gr}}$ (`do_jdot_gr`).
    pub do_jdot_gr: bool,
    /// Enable mass-loss $\dot{J}_{\mathrm{ml}}$ (`do_jdot_ml`).
    pub do_jdot_ml: bool,
    /// Enable spin-orbit $\dot{J}_{\mathrm{ls}}$ (`do_jdot_ls`).
    pub do_jdot_ls: bool,
    /// Enable missing-wind $\dot{J}_{\mathrm{missing\_wind}}$ (`do_jdot_missing_wind`).
    pub do_jdot_missing_wind: bool,
    /// Enable magnetic-braking $\dot{J}_{\mathrm{mb}}$ (`do_jdot_mb`).
    pub do_jdot_mb: bool,
    /// Include accretor in magnetic braking (`include_accretor_mb`).
    pub include_accretor_mb: bool,
    /// Magnetic-braking exponent $\gamma_{\mathrm{mb}}$ (`magnetic_braking_gamma`).
    pub magnetic_braking_gamma: f64,
    /// Keep magnetic braking active once triggered (`keep_mb_on`).
    pub keep_mb_on: bool,
    /// Minimum convective-envelope mass fraction for MB (`jdot_mb_min_qconv_env`).
    pub jdot_mb_min_qconv_env: f64,
    /// Maximum convective-envelope mass fraction for MB (`jdot_mb_max_qconv_env`).
    pub jdot_mb_max_qconv_env: f64,
    /// Maximum convective-core mass fraction for MB (`jdot_mb_max_qconv_core`).
    pub jdot_mb_max_qconv_core: f64,
    /// Inner $q$ limit for radiative-core check (`jdot_mb_qlim_for_check_rad_core`).
    pub jdot_mb_qlim_for_check_rad_core: f64,
    /// Outer $q$ limit for convective-envelope check (`jdot_mb_qlim_for_check_conv_env`).
    pub jdot_mb_qlim_for_check_conv_env: f64,
    /// Smoothly damp MB at low convective-envelope mass (`jdot_mb_scale_for_low_qconv_env`).
    pub jdot_mb_scale_for_low_qconv_env: bool,
    /// Characteristic low convective-envelope mass fraction (`jdot_mb_mass_frac_for_scale`).
    pub jdot_mb_mass_frac_for_scale: f64,
    /// Multiplier on non-`ls` orbital $\dot{J}$ terms (`jdot_multiplier`).
    pub jdot_multiplier: f64,
    /// Accrete stream/disk specific angular momentum (`do_j_accretion`).
    pub do_j_accretion: bool,
    /// Enable tidal spin synchronization (`do_tidal_sync`).
    pub do_tidal_sync: bool,
    /// Tidal sync prescription for star 1 (`sync_type_1`).
    pub sync_type_1: String,
    /// Tidal sync prescription for star 2 (`sync_type_2`).
    pub sync_type_2: String,
    /// Tidal sync radial mode for star 1 (`sync_mode_1`).
    pub sync_mode_1: String,
    /// Tidal sync radial mode for star 2 (`sync_mode_2`).
    pub sync_mode_2: String,
    /// Tidal strength factor for star 1 (`Ftid_1`).
    pub ftid_1: f64,
    /// Tidal strength factor for star 2 (`Ftid_2`).
    pub ftid_2: f64,
    /// Synchronize star 1 spin to orbit at $t = 0$ (`do_initial_orbit_sync_1`).
    pub do_initial_orbit_sync_1: bool,
    /// Synchronize star 2 spin to orbit at $t = 0$ (`do_initial_orbit_sync_2`).
    pub do_initial_orbit_sync_2: bool,
    /// Convective turnover tidal reduction exponent (`tidal_reduction`).
    pub tidal_reduction: f64,
    /// Enable tidal orbital circularization (`do_tidal_circ`).
    pub do_tidal_circ: bool,
    /// Circularization prescription for star 1 (`circ_type_1`).
    pub circ_type_1: String,
    /// Circularization prescription for star 2 (`circ_type_2`).
    pub circ_type_2: String,
    /// Enable eccentric mass-loss eccentricity enhancement (`use_eccentricity_enhancement`).
    pub use_eccentricity_enhancement: bool,
    /// Upper ceiling on $\dot{e}_{\mathrm{enhance}}$ in $\mathrm{s}^{-1}$ (`max_abs_edot_enhance`).
    pub max_abs_edot_enhance: f64,
    /// Number of true-anomaly quadrature points (`anomaly_steps`, default `500`).
    pub anomaly_steps: usize,
    /// Enable full Roche-lobe tidal distortion tables (`use_tidal_deformation`).
    pub use_tidal_deformation: bool,
    /// Synchronicity sigmoid midpoint (`f_sync_switch_from_rot_defor`, default `0.8`).
    pub f_sync_switch_from_rot_defor: f64,
    /// Synchronicity sigmoid width (`f_sync_switch_width`, default `0.1`).
    pub f_sync_switch_width: f64,
    /// Synchronicity sigmoid cutoff limit (`f_sync_switch_lim`, default `0.01`).
    pub f_sync_switch_lim: f64,

    // --- Binary timestep soft/hard limits & stage `varcontrol` ---
    /// Global binary timestep multiplier (`time_delta_coeff`).
    pub time_delta_coeff: f64,
    /// Relative envelope-mass timestep limit (`fm`).
    pub fm: f64,
    /// Hard relative envelope-mass limit (`fm_hard`).
    pub fm_hard: f64,
    /// Envelope-mass floor for `fm` (`fm_limit`).
    pub fm_limit: f64,
    /// Relative Roche-gap timestep limit (`fr`).
    pub fr: f64,
    /// Hard relative Roche-gap limit (`fr_hard`).
    pub fr_hard: f64,
    /// Gap floor for `fr` (`fr_limit`).
    pub fr_limit: f64,
    /// Minimum `fr` timestep in years (`fr_dt_limit`).
    pub fr_dt_limit: f64,
    /// Relative $J_{\mathrm{orb}}$ timestep limit (`fj`).
    pub fj: f64,
    /// Hard relative $J_{\mathrm{orb}}$ limit (`fj_hard`).
    pub fj_hard: f64,
    /// Relative separation timestep limit (`fa`).
    pub fa: f64,
    /// Hard relative separation limit (`fa_hard`).
    pub fa_hard: f64,
    /// Relative eccentricity timestep limit (`fe`).
    pub fe: f64,
    /// Hard relative eccentricity limit (`fe_hard`).
    pub fe_hard: f64,
    /// Eccentricity floor for `fe` (`fe_limit`).
    pub fe_limit: f64,
    /// Relative component-mass timestep limit (`fdm`).
    pub fdm: f64,
    /// Hard relative component-mass limit (`fdm_hard`).
    pub fdm_hard: f64,
    /// Log-space timestep softening weight (`dt_softening_factor`).
    pub dt_softening_factor: f64,
    /// Timestep reduction factor on $J_{\mathrm{orb}}$ retry (`dt_reduction_factor_for_j`).
    pub dt_reduction_factor_for_j: f64,
    /// Stellar `varcontrol_target` during Case A RLOF (`varcontrol_case_a`).
    pub varcontrol_case_a: f64,
    /// Stellar `varcontrol_target` during Case B RLOF (`varcontrol_case_b`).
    pub varcontrol_case_b: f64,
    /// Stellar `varcontrol_target` on detached MS (`varcontrol_ms`).
    pub varcontrol_ms: f64,
    /// Stellar `varcontrol_target` on detached post-MS (`varcontrol_post_ms`).
    pub varcontrol_post_ms: f64,
}

impl Default for BinaryInlistControls {
    fn default() -> Self {
        Self {
            inlist_names: [String::from("inlist1"), String::from("inlist2")],
            binary_history_columns_file: String::new(),
            evolve_both_stars: false,
            change_initial_ignore_rlof_flag: false,
            change_ignore_rlof_flag: false,
            new_ignore_rlof_flag: false,
            log_directory: String::from("."),
            history_name: String::from("binary_history.data"),
            history_interval: 5,
            append_to_star_history: true,
            m1_msun: 1.0,
            m2_msun: 0.8,
            initial_period_in_days: 0.5,
            initial_separation_in_rsuns: 100.0,
            initial_eccentricity: 0.0,
            min_eccentricity: 0.0,
            max_eccentricity: 0.99,
            max_abs_edot_tidal: 1.0e-6,
            initial_bh_spin: 0.0,
            keep_donor_fixed: true,
            mdot_limit_donor_switch: 1.0e-20,
            mdot_scheme: String::from("Ritter"),
            max_tries_to_achieve: 20,
            solver_type: String::from("both"),
            implicit_scheme_tolerance: 1.0e-3,
            implicit_scheme_tiny_factor: 1.0e-6,
            initial_change_factor: 1.2,
            change_factor_fraction: 0.9,
            implicit_lambda: 0.25,
            max_change_factor: 1.5,
            min_change_factor: 1.05,
            roche_min_mdot: 1.0e-12,
            min_mdot_for_implicit: 1.0e-16,
            max_implicit_abs_mdot: 1.0e99,
            starting_mdot: 1.0e-12,
            cur_mdot_frac: 0.5,
            max_explicit_abs_mdot: 1.0e-7,
            accretor_overflow_terminate: 0.0,
            terminate_if_initial_overflow: true,
            terminate_if_l2_overflow: false,
            mass_transfer_alpha: 0.0,
            mass_transfer_beta: 0.0,
            mass_transfer_delta: 0.0,
            mass_transfer_gamma: 0.0,
            limit_retention_by_mdot_edd: false,
            use_es_opacity_for_mdot_edd: true,
            use_this_for_mdot_edd_eta: -1.0,
            use_radiation_corrected_transfer_rate: false,
            use_this_for_mdot_edd: -1.0,
            do_enhance_wind_1: false,
            do_enhance_wind_2: false,
            tout_b_wind_1: 1.0e4,
            tout_b_wind_2: 1.0e4,
            do_wind_mass_transfer_1: false,
            do_wind_mass_transfer_2: false,
            wind_bh_alpha_1: 1.5,
            wind_bh_beta_1: 0.125,
            max_wind_transfer_fraction_1: 0.5,
            wind_bh_alpha_2: 1.5,
            wind_bh_beta_2: 0.125,
            max_wind_transfer_fraction_2: 0.5,
            do_jdot_gr: true,
            do_jdot_ml: true,
            do_jdot_ls: true,
            do_jdot_missing_wind: false,
            do_jdot_mb: true,
            include_accretor_mb: false,
            magnetic_braking_gamma: 3.0,
            keep_mb_on: false,
            jdot_mb_min_qconv_env: 1.0e-6,
            jdot_mb_max_qconv_env: 0.99,
            jdot_mb_max_qconv_core: 1.0e-2,
            jdot_mb_qlim_for_check_rad_core: 1.0e-3,
            jdot_mb_qlim_for_check_conv_env: 0.999,
            jdot_mb_scale_for_low_qconv_env: true,
            jdot_mb_mass_frac_for_scale: 0.02,
            jdot_multiplier: 1.0,
            do_j_accretion: false,
            do_tidal_sync: false,
            sync_type_1: String::from("Hut_conv"),
            sync_type_2: String::from("Hut_conv"),
            sync_mode_1: String::from("Uniform"),
            sync_mode_2: String::from("Uniform"),
            ftid_1: 1.0,
            ftid_2: 1.0,
            do_initial_orbit_sync_1: false,
            do_initial_orbit_sync_2: false,
            tidal_reduction: 2.0,
            do_tidal_circ: false,
            circ_type_1: String::from("Hut_conv"),
            circ_type_2: String::from("Hut_conv"),
            use_eccentricity_enhancement: false,
            max_abs_edot_enhance: 1.0e-6,
            anomaly_steps: 500,
            use_tidal_deformation: false,
            f_sync_switch_from_rot_defor: 0.8,
            f_sync_switch_width: 0.1,
            f_sync_switch_lim: 0.01,
            time_delta_coeff: 1.0,
            fm: 0.01,
            fm_hard: -1.0,
            fm_limit: 1.0e-3,
            fr: 0.10,
            fr_hard: -1.0,
            fr_limit: 1.0e-2,
            fr_dt_limit: 10.0,
            fj: 0.001,
            fj_hard: 0.01,
            fa: 0.01,
            fa_hard: 0.02,
            fe: 0.01,
            fe_hard: -1.0,
            fe_limit: 1.0e-1,
            fdm: 0.005,
            fdm_hard: 0.01,
            dt_softening_factor: 0.5,
            dt_reduction_factor_for_j: 0.1,
            varcontrol_case_a: -1.0,
            varcontrol_case_b: -1.0,
            varcontrol_ms: -1.0,
            varcontrol_post_ms: -1.0,
        }
    }
}

impl BinaryInlistControls {
    /// Loads `&binary_job` and `&binary_controls` from `case_dir`, starting at
    /// `inlist` when present (recursively resolving `read_extra_binary_*` chains)
    /// and falling back to `inlist_project` only if `inlist` is absent, with strict
    /// default validation (`allow_unsupported_controls = false`).
    pub fn from_case_dir(case_dir: &Path) -> Result<Self, StarError> {
        Self::from_case_dir_with_options(case_dir, InlistLoadOptions::default())
    }

    /// Loads `&binary_job` and `&binary_controls` from `case_dir` with explicit
    /// [`InlistLoadOptions`].
    pub fn from_case_dir_with_options(
        case_dir: &Path,
        options: InlistLoadOptions,
    ) -> Result<Self, StarError> {
        let entry = if case_dir.join("inlist").is_file() {
            "inlist"
        } else {
            "inlist_project"
        };
        Self::from_inlist_file_with_options(case_dir, entry, options)
    }

    /// Loads `&binary_job` and `&binary_controls` from `entry_file` relative to
    /// `case_dir` with strict default validation (`allow_unsupported_controls = false`).
    pub fn from_inlist_file(
        case_dir: &Path,
        entry_file: &str,
    ) -> Result<Self, StarError> {
        Self::from_inlist_file_with_options(
            case_dir,
            entry_file,
            InlistLoadOptions::default(),
        )
    }

    /// Loads `&binary_job` and `&binary_controls` from `entry_file` relative to
    /// `case_dir` with explicit [`InlistLoadOptions`].
    pub fn from_inlist_file_with_options(
        case_dir: &Path,
        entry_file: &str,
        options: InlistLoadOptions,
    ) -> Result<Self, StarError> {
        let sections =
            InlistBundle::load_sections_in_case_dir(case_dir, entry_file)?;
        let report = validate_binary_sections(&sections);
        report
            .into_result(options.allow_unsupported_controls)
            .map_err(|e| StarError::InvalidInput(e.to_string()))?;

        let mut out = Self::default();
        if let Some(bjob) = sections.get("binary_job") {
            out.apply_binary_job(bjob)?;
        }
        if let Some(bctrl) = sections.get("binary_controls") {
            out.apply_binary_controls(bctrl)?;
        }
        out.validate_values()?;
        Ok(out)
    }

    fn validate_values(&mut self) -> Result<(), StarError> {
        if !self.initial_bh_spin.is_finite() {
            return Err(StarError::InvalidInput(format!(
                "invalid '&binary_controls::initial_bh_spin' = {} (must be finite)",
                self.initial_bh_spin
            )));
        }
        // Upstream `binary/private/binary_evolve.f90:144-150` warns and clamps
        // `initial_bh_spin` to `[0, 1]` (exercised by `wind_fed_bhhmxb`).
        self.initial_bh_spin = self.initial_bh_spin.clamp(0.0, 1.0);
        if !self.m1_msun.is_finite() || self.m1_msun <= 0.0 {
            return Err(StarError::InvalidInput(format!(
                "invalid '&binary_controls::m1' = {} (must be > 0)",
                self.m1_msun
            )));
        }
        if !self.m2_msun.is_finite() || self.m2_msun <= 0.0 {
            return Err(StarError::InvalidInput(format!(
                "invalid '&binary_controls::m2' = {} (must be > 0)",
                self.m2_msun
            )));
        }
        if !self.initial_eccentricity.is_finite()
            || self.initial_eccentricity < 0.0
            || self.initial_eccentricity >= 1.0
        {
            return Err(StarError::InvalidInput(format!(
                "invalid '&binary_controls::initial_eccentricity' = {} (must lie in [0, 1))",
                self.initial_eccentricity
            )));
        }
        if !self.dt_reduction_factor_for_j.is_finite()
            || self.dt_reduction_factor_for_j <= 0.0
            || self.dt_reduction_factor_for_j >= 1.0
        {
            return Err(StarError::InvalidInput(format!(
                "invalid '&binary_controls::dt_reduction_factor_for_j' = {} (must lie in (0, 1))",
                self.dt_reduction_factor_for_j
            )));
        }
        if !self.time_delta_coeff.is_finite() || self.time_delta_coeff <= 0.0 {
            return Err(StarError::InvalidInput(format!(
                "invalid '&binary_controls::time_delta_coeff' = {} (must be > 0)",
                self.time_delta_coeff
            )));
        }
        if !matches!(
            self.mdot_scheme.trim().to_ascii_lowercase().as_str(),
            "ritter" | "kolb" | "roche_lobe" | "contact" | "arras"
        ) {
            return Err(StarError::InvalidInput(format!(
                "unrecognized '&binary_controls::mdot_scheme' option '{}'",
                self.mdot_scheme
            )));
        }
        if !matches!(
            self.solver_type.trim().to_ascii_lowercase().as_str(),
            "cubic" | "bisect" | "both"
        ) {
            return Err(StarError::InvalidInput(format!(
                "unrecognized '&binary_controls::solver_type' option '{}'",
                self.solver_type
            )));
        }
        crate::system_jdot::validate_sync_type(&self.sync_type_1)?;
        crate::system_jdot::validate_sync_type(&self.sync_type_2)?;
        crate::system_jdot::validate_sync_mode(&self.sync_mode_1)?;
        crate::system_jdot::validate_sync_mode(&self.sync_mode_2)?;
        crate::system_jdot::validate_circ_type(&self.circ_type_1)?;
        crate::system_jdot::validate_circ_type(&self.circ_type_2)?;
        Ok(())
    }

    fn apply_binary_job(
        &mut self,
        bjob: &SectionControls,
    ) -> Result<(), StarError> {
        let names =
            bjob.try_get_indexed_strings("binary_job", "inlist_names")?;
        for (idx, dst) in self.inlist_names.iter_mut().enumerate() {
            if let Some(n) = names.get(idx).filter(|s| !s.trim().is_empty()) {
                *dst = n.trim().to_string();
            }
        }
        if let Some(v) =
            bjob.try_get_str("binary_job", "binary_history_columns_file")?
        {
            self.binary_history_columns_file = v.trim().to_string();
        }
        for (dst, key) in [
            (&mut self.evolve_both_stars, "evolve_both_stars"),
            (
                &mut self.change_initial_ignore_rlof_flag,
                "change_initial_ignore_rlof_flag",
            ),
            (&mut self.change_ignore_rlof_flag, "change_ignore_rlof_flag"),
            (&mut self.new_ignore_rlof_flag, "new_ignore_rlof_flag"),
        ] {
            bjob.try_bool_into("binary_job", dst, key)?;
        }
        Ok(())
    }

    fn apply_binary_controls(
        &mut self,
        s: &SectionControls,
    ) -> Result<(), StarError> {
        if let Some(v) =
            s.try_get_i64("binary_controls", "max_tries_to_achieve")?
        {
            self.max_tries_to_achieve = v;
        }
        if let Some(v) = s.try_get_i64("binary_controls", "anomaly_steps")? {
            if v < 2 {
                return Err(StarError::InvalidInput(format!(
                    "invalid '&binary_controls::anomaly_steps' = {v} (must be >= 2)"
                )));
            }
            self.anomaly_steps = v as usize;
        }
        s.try_usize_into(
            "binary_controls",
            &mut self.history_interval,
            "history_interval",
            0,
        )?;
        for (dst, key) in [
            (&mut self.log_directory, "log_directory"),
            (&mut self.history_name, "history_name"),
            (&mut self.mdot_scheme, "mdot_scheme"),
            (&mut self.solver_type, "solver_type"),
            (&mut self.sync_type_1, "sync_type_1"),
            (&mut self.sync_type_2, "sync_type_2"),
            (&mut self.sync_mode_1, "sync_mode_1"),
            (&mut self.sync_mode_2, "sync_mode_2"),
            (&mut self.circ_type_1, "circ_type_1"),
            (&mut self.circ_type_2, "circ_type_2"),
        ] {
            if let Some(v) = s.try_get_str("binary_controls", key)? {
                *dst = v.trim().to_string();
            }
        }
        for (dst, key) in [
            (&mut self.append_to_star_history, "append_to_star_history"),
            (&mut self.keep_donor_fixed, "keep_donor_fixed"),
            (
                &mut self.terminate_if_initial_overflow,
                "terminate_if_initial_overflow",
            ),
            (
                &mut self.terminate_if_l2_overflow,
                "terminate_if_l2_overflow",
            ),
            (
                &mut self.limit_retention_by_mdot_edd,
                "limit_retention_by_mdot_edd",
            ),
            (
                &mut self.use_es_opacity_for_mdot_edd,
                "use_es_opacity_for_mdot_edd",
            ),
            (
                &mut self.use_radiation_corrected_transfer_rate,
                "use_radiation_corrected_transfer_rate",
            ),
            (&mut self.do_enhance_wind_1, "do_enhance_wind_1"),
            (&mut self.do_enhance_wind_2, "do_enhance_wind_2"),
            (&mut self.do_wind_mass_transfer_1, "do_wind_mass_transfer_1"),
            (&mut self.do_wind_mass_transfer_2, "do_wind_mass_transfer_2"),
            (&mut self.do_jdot_gr, "do_jdot_gr"),
            (&mut self.do_jdot_ml, "do_jdot_ml"),
            (&mut self.do_jdot_ls, "do_jdot_ls"),
            (&mut self.do_jdot_missing_wind, "do_jdot_missing_wind"),
            (&mut self.do_jdot_mb, "do_jdot_mb"),
            (&mut self.include_accretor_mb, "include_accretor_mb"),
            (&mut self.keep_mb_on, "keep_mb_on"),
            (
                &mut self.jdot_mb_scale_for_low_qconv_env,
                "jdot_mb_scale_for_low_qconv_env",
            ),
            (&mut self.do_j_accretion, "do_j_accretion"),
            (&mut self.do_tidal_sync, "do_tidal_sync"),
            (&mut self.do_initial_orbit_sync_1, "do_initial_orbit_sync_1"),
            (&mut self.do_initial_orbit_sync_2, "do_initial_orbit_sync_2"),
            (&mut self.do_tidal_circ, "do_tidal_circ"),
            (
                &mut self.use_eccentricity_enhancement,
                "use_eccentricity_enhancement",
            ),
            (&mut self.use_tidal_deformation, "use_tidal_deformation"),
        ] {
            s.try_bool_into("binary_controls", dst, key)?;
        }
        for (dst, key) in [
            (&mut self.m1_msun, "m1"),
            (&mut self.m2_msun, "m2"),
            (&mut self.initial_period_in_days, "initial_period_in_days"),
            (
                &mut self.initial_separation_in_rsuns,
                "initial_separation_in_rsuns",
            ),
            (&mut self.initial_eccentricity, "initial_eccentricity"),
            (&mut self.min_eccentricity, "min_eccentricity"),
            (&mut self.max_eccentricity, "max_eccentricity"),
            (&mut self.max_abs_edot_tidal, "max_abs_edot_tidal"),
            (&mut self.max_abs_edot_enhance, "max_abs_edot_enhance"),
            (
                &mut self.f_sync_switch_from_rot_defor,
                "f_sync_switch_from_rot_defor",
            ),
            (&mut self.f_sync_switch_width, "f_sync_switch_width"),
            (&mut self.f_sync_switch_lim, "f_sync_switch_lim"),
            (&mut self.initial_bh_spin, "initial_bh_spin"),
            (&mut self.mdot_limit_donor_switch, "mdot_limit_donor_switch"),
            (
                &mut self.implicit_scheme_tolerance,
                "implicit_scheme_tolerance",
            ),
            (
                &mut self.implicit_scheme_tiny_factor,
                "implicit_scheme_tiny_factor",
            ),
            (&mut self.initial_change_factor, "initial_change_factor"),
            (&mut self.change_factor_fraction, "change_factor_fraction"),
            (&mut self.implicit_lambda, "implicit_lambda"),
            (&mut self.max_change_factor, "max_change_factor"),
            (&mut self.min_change_factor, "min_change_factor"),
            (&mut self.roche_min_mdot, "roche_min_mdot"),
            (&mut self.min_mdot_for_implicit, "min_mdot_for_implicit"),
            (&mut self.max_implicit_abs_mdot, "max_implicit_abs_mdot"),
            (&mut self.starting_mdot, "starting_mdot"),
            (&mut self.cur_mdot_frac, "cur_mdot_frac"),
            (&mut self.max_explicit_abs_mdot, "max_explicit_abs_mdot"),
            (
                &mut self.accretor_overflow_terminate,
                "accretor_overflow_terminate",
            ),
            (&mut self.mass_transfer_alpha, "mass_transfer_alpha"),
            (&mut self.mass_transfer_beta, "mass_transfer_beta"),
            (&mut self.mass_transfer_delta, "mass_transfer_delta"),
            (&mut self.mass_transfer_gamma, "mass_transfer_gamma"),
            (
                &mut self.use_this_for_mdot_edd_eta,
                "use_this_for_mdot_edd_eta",
            ),
            (&mut self.use_this_for_mdot_edd, "use_this_for_mdot_edd"),
            (&mut self.tout_b_wind_1, "tout_b_wind_1"),
            (&mut self.tout_b_wind_2, "tout_b_wind_2"),
            (&mut self.wind_bh_alpha_1, "wind_bh_alpha_1"),
            (&mut self.wind_bh_beta_1, "wind_bh_beta_1"),
            (
                &mut self.max_wind_transfer_fraction_1,
                "max_wind_transfer_fraction_1",
            ),
            (&mut self.wind_bh_alpha_2, "wind_bh_alpha_2"),
            (&mut self.wind_bh_beta_2, "wind_bh_beta_2"),
            (
                &mut self.max_wind_transfer_fraction_2,
                "max_wind_transfer_fraction_2",
            ),
            (&mut self.magnetic_braking_gamma, "magnetic_braking_gamma"),
            (&mut self.jdot_mb_min_qconv_env, "jdot_mb_min_qconv_env"),
            (&mut self.jdot_mb_max_qconv_env, "jdot_mb_max_qconv_env"),
            (&mut self.jdot_mb_max_qconv_core, "jdot_mb_max_qconv_core"),
            (
                &mut self.jdot_mb_qlim_for_check_rad_core,
                "jdot_mb_qlim_for_check_rad_core",
            ),
            (
                &mut self.jdot_mb_qlim_for_check_conv_env,
                "jdot_mb_qlim_for_check_conv_env",
            ),
            (
                &mut self.jdot_mb_mass_frac_for_scale,
                "jdot_mb_mass_frac_for_scale",
            ),
            (&mut self.jdot_multiplier, "jdot_multiplier"),
            (&mut self.ftid_1, "ftid_1"),
            (&mut self.ftid_2, "ftid_2"),
            (&mut self.tidal_reduction, "tidal_reduction"),
            (&mut self.time_delta_coeff, "time_delta_coeff"),
            (&mut self.fm, "fm"),
            (&mut self.fm_hard, "fm_hard"),
            (&mut self.fm_limit, "fm_limit"),
            (&mut self.fr, "fr"),
            (&mut self.fr_hard, "fr_hard"),
            (&mut self.fr_limit, "fr_limit"),
            (&mut self.fr_dt_limit, "fr_dt_limit"),
            (&mut self.fj, "fj"),
            (&mut self.fj_hard, "fj_hard"),
            (&mut self.fa, "fa"),
            (&mut self.fa_hard, "fa_hard"),
            (&mut self.fe, "fe"),
            (&mut self.fe_hard, "fe_hard"),
            (&mut self.fe_limit, "fe_limit"),
            (&mut self.fdm, "fdm"),
            (&mut self.fdm_hard, "fdm_hard"),
            (&mut self.dt_softening_factor, "dt_softening_factor"),
            (
                &mut self.dt_reduction_factor_for_j,
                "dt_reduction_factor_for_j",
            ),
            (&mut self.varcontrol_case_a, "varcontrol_case_a"),
            (&mut self.varcontrol_case_b, "varcontrol_case_b"),
            (&mut self.varcontrol_ms, "varcontrol_ms"),
            (&mut self.varcontrol_post_ms, "varcontrol_post_ms"),
        ] {
            s.try_f64_into("binary_controls", dst, key)?;
        }
        Ok(())
    }
}
