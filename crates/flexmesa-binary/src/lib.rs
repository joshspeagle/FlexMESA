// SPDX-License-Identifier: LGPL-3.0-or-later
//! Binary star orbital mechanics, Roche-lobe overflow, angular momentum loss,
//! tidal synchronization/circularization, and coupled binary evolution for
//! `FlexMESA`.
//!
//! Corresponds to the upstream Fortran MESA `binary/` module (`binary_def`,
//! `binary_lib`, `binary_utils`, `binary_mdot`, `binary_disk`, `binary_wind`,
//! `binary_jdot`, `binary_tides`, `binary_edot`, `binary_timestep`, and
//! `binary_evolve`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

pub mod disk;
pub mod e2e;
pub mod e2e_oracles;
pub mod edd_wind;
pub mod jdot_tides;
pub mod mdot;
pub mod orbit;
pub mod system;
pub mod timestep;

pub use disk::eval_l2_mass_loss_fraction;
pub use e2e::{
    eval_brott_wind_msun_yr, eval_gr_expected_separation_rsun,
    eval_tauris_vdh_expected_separation_rsun, is_tier5_binary_case,
    run_binary_suite, run_binary_test_suite_case, BinaryCaseRunReport,
    BinaryInlistControls, TIER5_BINARY_CASES,
};
pub use edd_wind::{
    bh_initial_equatorial_mass, eval_accreted_material_j, eval_bh_spin,
    eval_bondi_hoyle_wind_fraction, eval_mdot_edd, eval_tout_wind_enhancement,
};
pub use jdot_tides::{
    eval_edot_gr, eval_edot_tidal_hut, eval_jdot_gr, eval_jdot_ls,
    eval_jdot_ls_with_winds, eval_jdot_mb, eval_jdot_missing_wind,
    eval_jdot_ml, eval_k_div_t_convective, eval_k_div_t_radiative, eval_tsync,
    hut_f2, hut_f3, hut_f4, hut_f5,
};
pub use mdot::{
    eval_arras_mdot_thin, eval_kolb_mdot_eccentric, eval_kolb_mdot_thick,
    eval_ritter_mdot_eccentric, eval_ritter_mdot_thin, kolb_f3,
    ImplicitRlofSolver,
};
pub use orbit::{
    build_anomaly_grid, eval_j_orb, eval_l2_overflow_limit,
    eval_rl_relative_gap, eval_rlobe, eval_rlobe_ad, period_from_separation,
    separation_from_j_orb, separation_from_period,
};
pub use system::{BinaryCompanion, BinarySystem};
pub use timestep::{
    binary_pick_next_timestep, BinaryTimestepControls, B_TLIM_COMP, B_TLIM_DM,
    B_TLIM_ECC, B_TLIM_ENV, B_TLIM_JORB, B_TLIM_ROCHE, B_TLIM_SEP,
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
        let (dt1_sec, reason1) = binary_pick_next_timestep(&ctrl);
        assert_eq!(reason1, B_TLIM_ROCHE);
        assert!((dt1_sec / SECYER - 5.000_000_000_000_004_4).abs() < 1.0e-13);

        ctrl.max_timestep_sec = 10.0;
        ctrl.time_delta_coeff = 0.5;
        let (dt2_sec, reason2) = binary_pick_next_timestep(&ctrl);
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

        let q = (m1 / m2).powf(1.0 / 3.0);
        let denom = 0.6 * q * q + q.ln_1p();
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
    fn test_e2e_5_binary_test_suite_and_oracles() {
        use std::path::Path;

        assert_eq!(TIER5_BINARY_CASES.len(), 8);
        assert!(is_tier5_binary_case("jdot_gr_check"));
        assert!(is_tier5_binary_case("wind_fed_bhhmxb"));
        assert!(is_tier5_binary_case("double_bh"));

        let a_gr =
            eval_gr_expected_separation_rsun(1.0, 1.4, 1.4, 1.0e7 * SECYER);
        assert!(a_gr > 0.0 && a_gr < 1.0);

        let a_ml = eval_tauris_vdh_expected_separation_rsun(
            10.0, 2.0, 1.5, 0.4, 0.4, 0.19, 2.0,
        );
        assert!(a_ml > 0.0 && a_ml.is_finite());

        let w_brott = eval_brott_wind_msun_yr(
            5.0e5, 60.0, 15.0, 45_000.0, 0.7, 0.28, 0.02,
        );
        assert!(w_brott > 0.0 && w_brott.is_finite());

        let require_zero_skips =
            std::env::var("FLEXMESA_REQUIRE_ZERO_SKIPS").as_deref() == Ok("1");
        if require_zero_skips {
            let mesa_dir = std::env::var("MESA_DIR")
                .expect("MESA_DIR required when FLEXMESA_REQUIRE_ZERO_SKIPS=1");
            assert!(
                Path::new(&mesa_dir)
                    .join("binary")
                    .join("test_suite")
                    .is_dir(),
                "missing binary/test_suite under {mesa_dir}"
            );
        }
        if let Ok(mesa_dir) = std::env::var("MESA_DIR") {
            let base = Path::new(&mesa_dir);
            if base.join("binary").join("test_suite").is_dir() {
                let reps =
                    run_binary_suite(base, Some(4)).expect("binary suite");
                assert_eq!(reps.len(), 8);
                for r in &reps {
                    assert!(
                        r.passed,
                        "binary case {} failed: m1={} m2={} a={} P={} rel_err={}",
                        r.case_name,
                        r.final_m1_msun,
                        r.final_m2_msun,
                        r.final_separation_rsun,
                        r.final_period_days,
                        r.max_relative_oracle_error
                    );
                }
            }
        }
    }

    #[test]
    fn test_fidelity3_anomaly_grid_edot_gr_jdot_ls_and_contact_solver() {
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

        let edot = eval_edot_gr(1.4 * MSUN, 1.4 * MSUN, 2.0 * RSUN, 0.3);
        assert!(edot < 0.0 && edot.is_finite());

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
        let slope_c = q.powf(-0.52);
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
