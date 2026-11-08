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
pub mod edd_wind;
pub mod jdot_tides;
pub mod mdot;
pub mod orbit;
pub mod system;
pub mod timestep;

pub use disk::eval_l2_mass_loss_fraction;
pub use edd_wind::{
    bh_initial_equatorial_mass, eval_accreted_material_j, eval_bh_spin,
    eval_bondi_hoyle_wind_fraction, eval_mdot_edd, eval_tout_wind_enhancement,
};
pub use jdot_tides::{
    eval_edot_tidal_hut, eval_jdot_gr, eval_jdot_ls, eval_jdot_mb,
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
}
