// SPDX-License-Identifier: GPL-3.0-or-later
//! Binary star orbital mechanics, Roche-lobe overflow, angular momentum loss,
//! tidal synchronization/circularization, and coupled binary evolution for
//! `FlexMESA`.
//!
//! Corresponds to the upstream Fortran MESA `binary/` module (`binary_def`,
//! `binary_lib`, `binary_utils`, `binary_mdot`, `binary_disk`, `binary_wind`,
//! `binary_jdot`, `binary_tides`, `binary_edot`, `binary_timestep`, and
//! `binary_evolve`).
//!
//! **Status**: Closed-form Roche-lobe, torque, and mass-transfer kernels are
//! unit-tested; the coupled `BinarySystem` evolution driver, orbital step
//! sequence, implicit mass-transfer search, and `binary/test_suite` runner are
//! **unverified** against real MESA output (pending #143, #154, and #160).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::fn_params_excessive_bools,
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

pub mod binary_history;
pub mod disk;
pub mod e2e;
pub mod edd_wind;
pub mod extras;
pub mod inlist;
pub mod inlist_inventory;
pub mod jdot_tides;
pub mod mdot;
pub mod mdot_implicit;
pub mod orbit;
pub mod roche_deformation;
pub mod system;
pub mod system_init;
pub mod system_jdot;
pub mod system_rlof;
pub mod system_step;
pub mod timestep;

pub use binary_history::{
    BINARY_LOG_COL_WIDTH, BinaryHistoryColumnSpec, BinaryLogsWriter,
    DEFAULT_BINARY_HISTORY_COLUMNS_TEXT, IncrementalBinaryHistoryWriter,
    binary_history_row_as_f64_pairs, default_binary_history_specs,
    evaluate_binary_history_row, format_binary_history_data_line,
    format_binary_history_file_header, parse_binary_history_columns,
    resolve_binary_history_columns,
};
pub use disk::eval_l2_mass_loss_fraction;
pub use e2e::{
    BinaryCaseRunReport, BinaryInlistControls, BinaryLogsOptions,
    run_binary_test_suite_case, run_binary_test_suite_case_with_extras,
    run_binary_test_suite_case_with_logs,
    run_binary_test_suite_case_with_options,
};
pub use edd_wind::{
    PointMassEddRetention, bh_initial_equatorial_mass,
    eval_accreted_material_j, eval_bh_spin, eval_bondi_hoyle_wind_fraction,
    eval_mdot_edd, eval_mdot_edd_with_overrides,
    eval_point_mass_edd_and_retention, eval_tout_wind_enhancement,
    eval_tout_wind_enhancement_orbit_averaged,
};
pub use extras::{BinaryExtras, BinaryExtrasAction, DefaultBinaryExtras};
pub use inlist::ensure_no_unported_binary_driver_controls;
pub use inlist_inventory::{
    BINARY_CONTROLS_SUPPORTED, BINARY_CONTROLS_TERMINAL_OUTPUT,
    BINARY_JOB_HEADLESS_GUI, BINARY_JOB_SUPPORTED, BINARY_JOB_TERMINAL_OUTPUT,
    BinaryControlInventory, copy_cleaned_binary_case,
    supported_binary_control_inventory, validate_binary_sections,
};
pub use jdot_tides::{
    check_jdot_mb_conditions, eval_edot_tidal_hut, eval_jdot_gr, eval_jdot_ls,
    eval_jdot_ls_with_am_removed, eval_jdot_ls_with_winds, eval_jdot_mb,
    eval_jdot_mb_with_scale_mass, eval_jdot_missing_wind, eval_jdot_ml,
    eval_k_div_t_convective, eval_k_div_t_radiative, eval_tsync, hut_f2,
    hut_f3, hut_f4, hut_f5,
};
pub use mdot::{
    ImplicitMdotSolver, ImplicitRlofSolver, ImplicitStepAction,
    eval_arras_mdot_thin, eval_edot_enhancement_isotropic,
    eval_kolb_mdot_eccentric, eval_kolb_mdot_eccentric_photosphere,
    eval_kolb_mdot_thick, eval_kolb_mdot_thick_photosphere,
    eval_ritter_mdot_eccentric, eval_ritter_mdot_thin, kolb_f3,
    update_change_factor_after_step,
};
pub use orbit::{
    build_anomaly_grid, eval_j_orb, eval_l2_overflow_limit,
    eval_rl_relative_gap, eval_rlobe, eval_rlobe_ad, period_from_separation,
    separation_from_j_orb, separation_from_period,
};
pub use roche_deformation::{
    BinaryTidalDeformationHook, ROCHE_NUDGE, RocheDeformationTables,
    RocheTable2D, eval_synchronicity_switch,
};
pub use system::{BinaryCompanion, BinaryStepCounters, BinarySystem};
pub use system_jdot::{
    EdotBreakdown, JdotBreakdown, eval_coupled_edot,
    eval_coupled_edot_breakdown, eval_gated_jdot,
};
pub use system_rlof::{
    BinaryStepSnapshot, CoupledMdotAccounting, MAX_NUM_ACCRETION_SPECIES,
    blend_explicit_mdot, eval_scheme_explicit_mdot,
    eval_scheme_explicit_mdot_with_steps,
    eval_star_intrinsic_wind_cgs_eccentric,
    eval_star_intrinsic_wind_cgs_with_extras, prepare_coupled_mdots,
    prepare_coupled_mdots_with_extras, set_accretion_composition,
    star_skips_hard_limits,
};
pub use timestep::{
    B_NUM_TLIM, B_TLIM_COMP, B_TLIM_DM, B_TLIM_ECC, B_TLIM_ENV, B_TLIM_JORB,
    B_TLIM_ROCHE, B_TLIM_SEP, BinaryHardLimitViolation, BinaryTimestepControls,
    binary_dt_why_str, binary_pick_next_timestep, check_binary_hard_limits,
};

#[cfg(test)]
mod tests {
    use super::*;
    use flexmesa_autodiff::DualOrder1;
    use flexmesa_const::{MSUN, RSUN, SECYER};

    #[test]
    fn test_binary_pick_next_timestep_matches_upstream_golden_output() {
        let mut ctrl = BinaryTimestepControls {
            donor_time_step_yr: 10.0,
            max_timestep_sec: 10.0,
            time_delta_coeff: 1.0,
            dt_softening_factor: 0.0,
            rl_relative_gap: [-0.100, 0.0],
            rl_relative_gap_old: [-0.102, 0.0],
            is_point_mass_companion: true,
            fm: -1.0,
            fr: 1.0e-2,
            fr_limit: 1.0e-2,
            fr_dt_limit: 0.0,
            fj: -1.0,
            fa: -1.0,
            fe: -1.0,
            fdm: -1.0,
            ..BinaryTimestepControls::default()
        };
        let (dt1_sec, reason1) =
            binary_pick_next_timestep(&ctrl).expect("finite limits");
        assert_eq!(reason1, B_TLIM_ROCHE);
        assert!((dt1_sec / SECYER - 5.000_000_000_000_004_4).abs() < 1.0e-13);

        ctrl.max_timestep_sec = 10.0;
        ctrl.time_delta_coeff = 0.5;
        let (dt2_sec, reason2) =
            binary_pick_next_timestep(&ctrl).expect("finite limits");
        assert_eq!(reason2, B_TLIM_ROCHE);
        assert!((dt2_sec / SECYER - 2.500_000_000_000_006_7).abs() < 1.0e-13);
    }

    #[test]
    fn test_eval_l2_mass_loss_fraction_matches_upstream_golden_output() {
        let fl2_1 =
            eval_l2_mass_loss_fraction(20.0, 10.0, 0.001, 2.0, 0.1, 1.3 / 2.4);
        assert!(
            (fl2_1 - 8.970_156_414_543_509_8e-1).abs() < 1.0e-12,
            "fl2_1 = {fl2_1:.16e}"
        );

        let fl2_2 =
            eval_l2_mass_loss_fraction(20.0, 10.0, 0.0001, 2.0, 0.1, 1.3 / 2.4);
        assert!(
            (fl2_2 - 2.102_213_402_041_898_1e-1).abs() < 1.0e-12,
            "fl2_2 = {fl2_2:.16e}"
        );

        let fl2_3 =
            eval_l2_mass_loss_fraction(20.0, 18.0, 0.0, 304.0, 0.1, 0.64);
        assert_eq!(fl2_3, 0.0);
    }

    #[test]
    fn test_eval_rlobe_ad_and_implicit_rlof_solver_match_analytical_oracles() {
        let m1 = 2.0 * MSUN;
        let m2 = 1.2 * MSUN;
        let a = 8.0 * RSUN;
        let ad = eval_rlobe_ad::<3>(
            DualOrder1::independent(m1, 0),
            DualOrder1::independent(m2, 1),
            DualOrder1::independent(a, 2),
        );
        let val = eval_rlobe(m1, m2, a);
        assert!((ad.val - val).abs() / val < 1.0e-15);

        let q = flexmesa_math::pow(m1 / m2, 1.0 / 3.0);
        let denom = 0.6 * q * q + flexmesa_math::log1p(q);
        let drl_dq = a
            * (0.98 * q * denom - 0.49 * q * q * (1.2 * q + 1.0 / (1.0 + q)))
            / (denom * denom);
        let drl_dm1_exact = drl_dq * q / (3.0 * m1);
        let drl_dm2_exact = -drl_dq * q / (3.0 * m2);
        assert!(
            (ad.d1[0] - drl_dm1_exact).abs() / drl_dm1_exact.abs() < 1.0e-15
        );
        assert!(
            (ad.d1[1] - drl_dm2_exact).abs() / drl_dm2_exact.abs() < 1.0e-15
        );
        assert!((ad.d1[2] - val / a).abs() / (val / a) < 1.0e-15);

        let solver = ImplicitRlofSolver::default();
        let target_mdot = -1.0e-7 * MSUN / SECYER;
        let (mdot_sol, iters, obj) =
            solver.solve_roche_lobe_mdot(-2.0e-8 * MSUN / SECYER, 30, |mdot| {
                0.01 * (1.0 - mdot / target_mdot)
            });
        assert!(iters <= 30);
        assert!(obj.abs() <= solver.tolerance);
        assert!((mdot_sol - target_mdot).abs() / target_mdot.abs() < 1.0e-2);
    }

    #[test]
    fn test_e2e_5_binary_case_runner_rejects_raw_unsupported_controls() {
        let tmp_dir = std::env::temp_dir()
            .join(format!("flexmesa_bin_unsup_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).expect("create synthetic bin dir");
        std::fs::write(
            tmp_dir.join("inlist"),
            "&binary_job\n  evolve_both_stars = .false.\n/\n\
             &binary_controls\n  m1 = 10.0\n  m2 = 8.0\n  initial_period_in_days = 5.0\n/\n",
        )
        .expect("write synthetic binary inlist");
        std::fs::write(
            tmp_dir.join("inlist1"),
            "&star_job\n  echo_at_start = 'hi'\n/\n\
             &controls\n  initial_mass = 10.0\n/\n",
        )
        .expect("write synthetic component inlist1");
        let err = run_binary_test_suite_case(&tmp_dir, Some(4))
            .expect_err("unsupported star control must fail strict binary run");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        let msg = err.to_string();
        assert!(
            matches!(err, flexmesa_star::StarError::InvalidInput(_))
                && msg.contains("unsupported control")
                && msg.contains("echo_at_start"),
            "unexpected binary case error: {msg}"
        );

        if let Some(bin_suite) =
            flexmesa_utils::require_mesa_test_input_path("binary/test_suite")
        {
            for (case_name, inlist) in [
                ("double_bh", "inlist1"),
                ("double_bh", "inlist2"),
                ("jdot_gr_check", "inlist1"),
            ] {
                let bundle = flexmesa_star::InlistBundle::from_file_in_case_dir(
                    &bin_suite.join(case_name),
                    inlist,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "raw {case_name}/{inlist} must pass strict validation: {e}"
                    )
                });
                assert_eq!(
                    bundle.validation_report.unsupported_controls.len(),
                    0,
                    "unexpected unsupported controls in raw {case_name}/{inlist}: {:?}",
                    bundle.validation_report.unsupported_controls
                );
            }
        }
    }

    #[test]
    fn test_anomaly_grid_edot_175_186_jdot_mb_and_contact_solver() {
        for n in [8_usize, 9_usize, 16_usize] {
            let (theta, time_co) = build_anomaly_grid(n, 0.4);
            assert_eq!(theta.len(), n);
            assert_eq!(time_co.len(), n);
            assert!((time_co[0]).abs() < 1.0e-15);
            for w in time_co.windows(2) {
                assert!(
                    w[1] > w[0] && w[1] < 1.0,
                    "non-monotonic time_co for n={n}: {time_co:?}"
                );
            }
        }

        // `#175` & `#186`: `do_jdot_gr` contributes zero `edot`, and `edot_tidal`
        // uses `Ftid_1` on both stars (`binary_edot.f90:93, 171-172`).
        let s1 = flexmesa_star::load_zams_model(
            flexmesa_star::StarControls {
                initial_mass: 1.4,
                zbase: 0.02,
                ..flexmesa_star::StarControls::default()
            },
            flexmesa_star::StarJobControls::default(),
        )
        .expect("s1");
        let s2 = flexmesa_star::load_zams_model(
            flexmesa_star::StarControls {
                initial_mass: 1.1,
                zbase: 0.02,
                ..flexmesa_star::StarControls::default()
            },
            flexmesa_star::StarJobControls::default(),
        )
        .expect("s2");
        let mut ctrl = BinaryInlistControls {
            do_jdot_gr: true,
            do_tidal_circ: false,
            circ_type_1: String::from("Hut_rad"),
            circ_type_2: String::from("Hut_rad"),
            ftid_1: 1.0,
            ftid_2: 7.0,
            ..BinaryInlistControls::default()
        };
        let eb_gr_only = eval_coupled_edot_breakdown(
            &ctrl,
            &s1,
            Some(&s2),
            1,
            s1.mstar,
            s2.mstar,
            6.0 * RSUN,
            period_from_separation(s1.mstar, s2.mstar, 6.0 * RSUN),
            0.3,
        )
        .expect("edot gr only");
        assert_eq!(eb_gr_only.edot_total, 0.0);

        ctrl.do_tidal_circ = true;
        let eb_ftid_7 = eval_coupled_edot_breakdown(
            &ctrl,
            &s1,
            Some(&s2),
            1,
            s1.mstar,
            s2.mstar,
            6.0 * RSUN,
            period_from_separation(s1.mstar, s2.mstar, 6.0 * RSUN),
            0.3,
        )
        .expect("edot ftid_2=7");
        ctrl.ftid_2 = 1.0;
        let eb_ftid_1 = eval_coupled_edot_breakdown(
            &ctrl,
            &s1,
            Some(&s2),
            1,
            s1.mstar,
            s2.mstar,
            6.0 * RSUN,
            period_from_separation(s1.mstar, s2.mstar, 6.0 * RSUN),
            0.3,
        )
        .expect("edot ftid_2=1");
        assert_eq!(
            eb_ftid_7.edot_tidal.to_bits(),
            eb_ftid_1.edot_tidal.to_bits()
        );
        assert!(eb_ftid_1.edot_tidal < 0.0);

        // `F3-14`: `keep_mb_on` activates MB even when `was_using_mb == false`,
        // `jdot_ls` is zero on `doing_first_model_of_run`, and orbit-averaged
        // `Tout_enhance_wind` integrates over `anomaly_steps`.
        let mut mb_ctrl = BinaryInlistControls {
            keep_mb_on: true,
            jdot_mb_min_qconv_env: 0.99,
            ..BinaryInlistControls::default()
        };
        assert!(check_jdot_mb_conditions(&s1, &mb_ctrl, false).0);
        mb_ctrl.keep_mb_on = false;
        assert!(!check_jdot_mb_conditions(&s1, &mb_ctrl, true).0);

        assert_eq!(
            eval_jdot_ls_with_am_removed(
                true, 2, false, 1.0e50, 0.9e50, -1.0e18, 1.0e32, -1.0e18, 0.0,
                0.0, 0.0, 0.0, 1.0e14
            ),
            0.0
        );
        let w_enh = eval_tout_wind_enhancement_orbit_averaged(
            -1.0e18,
            RSUN,
            3.0 * RSUN,
            1.0e4,
            0.4,
            500,
        );
        assert!(w_enh < -1.0e18 && w_enh.is_finite());

        let jdot_ls_wind = eval_jdot_ls_with_winds(
            1.0e50, 0.9e50, -1.0e18, 1.0e17, 0.5e50, 0.5e50, 0.0, 0.0, 1.0e14,
        );
        assert!((jdot_ls_wind / 1.0e35).abs() < 1.0e-14);

        let jdot_mw = eval_jdot_missing_wind(
            false, 1.0e18, -2.0e18, 1.0, -0.5e18, 1.0e17,
        );
        assert!((jdot_mw - (-0.5e35)).abs() / 0.5e35 < 1.0e-14);
        let jdot_mw_neg = eval_jdot_missing_wind(
            false, -1.0e18, -2.0e18, 0.5, -0.5e18, 1.0e17,
        );
        assert!((jdot_mw_neg - (-1.0e35)).abs() / 1.0e35 < 1.0e-14);

        let solver = ImplicitRlofSolver::default();
        let target_mdot = -2.0e-7 * MSUN / SECYER;
        let q = 0.8_f64;
        let slope_c = flexmesa_math::pow(q, -0.52);
        let (mdot_c, iters_c, obj_c) =
            solver.solve_contact_mdot(-5.0e-8 * MSUN / SECYER, q, 30, |mdot| {
                let gap_d = 0.02 * (1.0 - 0.5 * mdot / target_mdot);
                let gap_a = 0.01 * slope_c;
                (gap_d, gap_a)
            });
        assert!(iters_c <= 30);
        assert!(obj_c.abs() <= solver.tolerance);
        assert!((mdot_c - target_mdot).abs() / target_mdot.abs() < 1.0e-2);
    }
}
