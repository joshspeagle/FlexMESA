// SPDX-License-Identifier: GPL-3.0-or-later
//! Asteroseismic scaling relations, surface-effect frequency corrections,
//! separation ratios ($r_{01}, r_{10}, r_{02}$), composite $\chi^2$ evaluation,
//! optimization drivers, and `astero/test_suite` runner for `FlexMESA`.
//!
//! Corresponds to the upstream Fortran MESA `astero/` module (`astero_def`,
//! `astero_lib`, `astero_support`, `astero_run_support`, `extras_support`, and
//! `pgstar_astero_plots`) and `star/private/report.f90` / `star_utils.f90`.
//!
//! **Status**: Closed-form surface corrections, separation ratios, and $\chi^2$
//! kernels are unit-tested; the coupled asteroseismic search driver, in-house
//! pulsation scaffolding, and `astero/test_suite` runner are **unverified**
//! against real MESA output (pending #143, #162, #170, and #171).

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
    clippy::type_complexity,
    clippy::unreadable_literal
)]

pub mod e2e;
pub mod inlist_astero;
pub mod inlist_parser;
pub mod optimize;
pub mod ratios_chi2;
pub mod sample_eval;
pub mod scaling;
pub mod search_driver;
mod search_methods;
pub mod surface_corr;

pub use e2e::{
    run_astero_test_suite_case, run_astero_test_suite_case_with_logs,
    run_astero_test_suite_case_with_options,
    run_astero_test_suite_case_with_case_hooks, TempCaseStaging,
};
pub use inlist_astero::{
    parse_astero_search_controls, parse_astero_search_controls_with_options,
    supported_astero_control_inventory, validate_astero_sections,
    AsteroControlInventory, AsteroParamSet, AsteroSearchControls,
    AsteroSearchParam, SpectroConstraint, ASTERO_SEARCH_CONTROLS_SUPPORTED,
    ASTERO_SEARCH_CONTROLS_UNUSED, MAX_CONSTRAINTS,
};
pub use inlist_parser::{
    copy_cleaned_astero_case, validate_astero_sections_with_pgstar,
};
pub use optimize::{
    bobyqa_minimize, map_bobyqa_param, nelder_mead_minimize, newuoa_minimize,
    powell_quadratic_minimize, scan_cartesian_grid,
    simplex_minimize_with_options, try_bobyqa_minimize, try_newuoa_minimize,
    try_simplex_minimize_with_options, unmap_bobyqa_param,
};
pub use ratios_chi2::{
    eval_astero_chi2, eval_frequency_ratios_r01_r10, eval_frequency_ratios_r02,
    eval_frequency_ratios_r02_with_dnu, eval_frequency_ratios_with_trimming,
    get_max_sequence, interpolate_ratio_r010, interpolate_ratio_r02,
    match_modes_by_degree, AsteroChi2Weights, FrequencyRatiosR010,
};
pub use sample_eval::{eval_model_astero_sample, AsteroSampleResult};
pub use scaling::{
    eval_acoustic_profile, eval_delta_nu_asymptotic,
    eval_delta_nu_radius_scaling, eval_delta_nu_scaling,
    eval_delta_pg_bildsten2012, eval_delta_pg_traditional,
    eval_glitch_acoustic_depths, eval_nu_ac, eval_nu_max,
    eval_radial_delta_nu_ols, DELTA_NU_SUN, NU_MAX_SUN, TEFF_SUN_ASTERO,
};
pub use search_driver::{
    run_single_astero_track, run_single_astero_track_with_logs,
    run_single_astero_track_with_case_hooks, run_star_astero,
    run_star_astero_with_logs, run_star_astero_with_options,
    run_star_astero_with_case_hooks, AsteroCaseHooks,
    AsteroSearchReport,
};
pub use surface_corr::{
    estimate_l0_n_obs, eval_combined_corr, eval_cubic_corr, eval_kjeldsen_corr,
    eval_power_law_corr, eval_power_law_corr_radial_only, eval_sonoi_corr,
    eval_sonoi_corr_radial_only, interpolate_l0_inertia,
};

#[cfg(test)]
mod tests {
    use super::*;
    use flexmesa_math::pow;

    #[test]
    fn test_surface_effect_corrections_recover_upstream_synthetic_targets() {
        // Exact synthetic target recovery test from
        // `astero/test_suite/surface_effects/src/run_star_extras.f90:194-245`
        // and `targets.inc` (`perfect_tol = 1.0e-12`).
        let target_a3 = -1.979_289_417_129_323_8e-18;
        let target_b1 = -5.125_972_793_567_201_1e-5;
        let target_b3 = -1.446_307_489_879_336_1e-18;
        let target_p0 = -6.157_582_281_235_104_2;
        let target_p1 = 4.342_864_640_344_271_4;
        let target_s0 = -4.380_735_022_728_040_1e-3;
        let target_s1 = 7.143_783_592_158_950_3;

        let nu_max = 3090.0_f64;
        let l0_freqs: Vec<f64> =
            (14..=25).map(|n| 135.0 * (f64::from(n) + 1.45)).collect();
        let l0_inertias: Vec<f64> = l0_freqs
            .iter()
            .map(|&f| 1.0e-7 * flexmesa_math::exp((3100.0 - f) / 1000.0))
            .collect();
        let sigmas = vec![0.15_f64; l0_freqs.len()];

        // 1. Ball & Gizon (2014) 1-term cubic
        let obs_cubic: Vec<f64> = l0_freqs
            .iter()
            .zip(&l0_inertias)
            .map(|(&f, &e)| f + target_a3 * (f * f * f) / e)
            .collect();
        let (corr_cubic, a3_fit) =
            eval_cubic_corr(&obs_cubic, &sigmas, &l0_freqs, &l0_inertias, 1.0);
        assert!((a3_fit / target_a3 - 1.0).abs() < 1.0e-12);
        for (c, o) in corr_cubic.iter().zip(&obs_cubic) {
            assert!((c - o).abs() < 1.0e-11);
        }

        // 2. Ball & Gizon (2014) 2-term combined
        let obs_comb: Vec<f64> = l0_freqs
            .iter()
            .zip(&l0_inertias)
            .map(|(&f, &e)| f + (target_b1 / f + target_b3 * (f * f * f)) / e)
            .collect();
        let (corr_comb, b3_fit, b1_fit) = eval_combined_corr(
            &obs_comb,
            &sigmas,
            &l0_freqs,
            &l0_inertias,
            1.0,
        );
        assert!((b3_fit / target_b3 - 1.0).abs() < 1.0e-12);
        assert!((b1_fit / target_b1 - 1.0).abs() < 1.0e-12);
        for (c, o) in corr_comb.iter().zip(&obs_comb) {
            assert!((c - o).abs() < 1.0e-11);
        }

        // 3. Ball & Gizon (2017) power-law
        let obs_pl: Vec<f64> = l0_freqs
            .iter()
            .map(|&f| f + target_p0 * pow(f / nu_max, target_p1))
            .collect();
        let (_, p0_fit, p1_fit) = eval_power_law_corr(
            &obs_pl,
            &sigmas,
            &l0_freqs,
            &l0_inertias,
            &l0_freqs,
            &l0_inertias,
            nu_max,
            1.0,
        );
        assert!((p0_fit / target_p0 - 1.0).abs() < 1.0e-12);
        assert!((p1_fit / target_p1 - 1.0).abs() < 1.0e-12);

        // 4. Sonoi et al. (2015) modified Lorentzian
        let obs_sonoi: Vec<f64> = l0_freqs
            .iter()
            .map(|&f| {
                f + target_s0
                    * nu_max
                    * (1.0 - 1.0 / (1.0 + pow(f / nu_max, target_s1)))
            })
            .collect();
        let (_, s0_fit, s1_fit) = eval_sonoi_corr(
            &obs_sonoi,
            &sigmas,
            &l0_freqs,
            &l0_inertias,
            &l0_freqs,
            &l0_inertias,
            nu_max,
            1.0,
        );
        assert!((s0_fit / target_s0 - 1.0).abs() < 1.0e-12);
        assert!((s1_fit / target_s1 - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_nelder_mead_and_powell_quadratic_optimizers_recover_minimum() {
        let target = [1.02_f64, 0.275_f64];
        let (x_nm, f_nm, _) = nelder_mead_minimize(
            &[1.0, 0.25],
            &[0.8, 0.20],
            &[1.3, 0.35],
            300,
            1.0e-7,
            1.0e-7,
            1.0e-12,
            0.0,
            |x| {
                flexmesa_math::pow2(x[0] - target[0])
                    + 10.0 * flexmesa_math::pow2(x[1] - target[1])
            },
        );
        assert!(f_nm < 1.0e-10);
        assert!((x_nm[0] - target[0]).abs() < 1.0e-5);
        assert!((x_nm[1] - target[1]).abs() < 1.0e-5);

        let (x_pq, f_pq, _) = powell_quadratic_minimize(
            &[1.0, 0.25],
            &[0.8, 0.20],
            &[1.3, 0.35],
            200,
            0.45,
            1.0e-6,
            |x| {
                flexmesa_math::pow2(x[0] - target[0])
                    + 10.0 * flexmesa_math::pow2(x[1] - target[1])
            },
        );
        assert!(f_pq < 1.0e-10);
        assert!((x_pq[0] - target[0]).abs() < 1.0e-5);
        assert!((x_pq[1] - target[1]).abs() < 1.0e-5);
    }

    #[test]
    fn test_fidelity3_radial_only_surface_corr_and_ratio_interpolators() {
        let nu_max = 3090.0_f64;
        let l0_freqs: Vec<f64> =
            (18..=22).map(|n| 135.0 * (f64::from(n) + 1.45)).collect();
        let l0_inertias = vec![1.0e-7_f64; l0_freqs.len()];
        let sigmas = vec![0.10_f64; l0_freqs.len()];

        let target_a = -4.5_f64;
        let target_b = 4.2_f64;
        let l0_obs: Vec<f64> = l0_freqs
            .iter()
            .map(|&f| f + target_a * pow(f / nu_max, target_b))
            .collect();
        let (corr_ro, a_ro, b_ro) = eval_power_law_corr_radial_only(
            &l0_obs,
            &sigmas,
            &l0_freqs,
            &l0_inertias,
            &l0_freqs,
            &l0_inertias,
            nu_max,
            1.0,
        );
        assert!((a_ro - target_a).abs() < 1.0e-10);
        assert!((b_ro - target_b).abs() < 1.0e-10);
        for (c, o) in corr_ro.iter().zip(&l0_obs) {
            assert!((c - o).abs() < 1.0e-9);
        }

        let l1_freqs: Vec<f64> = l0_freqs.iter().map(|&f| f + 62.0).collect();
        let l2_freqs: Vec<f64> =
            l0_freqs[1..].iter().map(|&f| f - 9.0).collect();
        let l2_sigmas = vec![0.10_f64; l2_freqs.len()];
        let (r02, sig_r02) = eval_frequency_ratios_r02(
            &l0_freqs, &sigmas, &l1_freqs, &sigmas, &l2_freqs, &l2_sigmas,
        );
        assert_eq!(r02.len(), l0_freqs.len() - 1);
        for (&r, &s) in r02.iter().zip(&sig_r02) {
            assert!((r - 9.0 / 135.0).abs() < 1.0e-12);
            assert!(s > 0.0);
        }

        let mid_f = 0.5 * (l0_freqs[1] + l0_freqs[2]);
        let r02_interp = interpolate_ratio_r02(mid_f, &l0_freqs, &r02);
        assert!((r02_interp - 9.0 / 135.0).abs() < 1.0e-12);
        let r010_interp = interpolate_ratio_r010(mid_f, &l0_freqs[1..], &r02);
        assert!((r010_interp - 9.0 / 135.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_simplex_budget_preserves_unclamped_first_param_and_itermax() {
        let mut pts: Vec<(f64, f64)> = Vec::new();
        let (x_best, f_best, evals) = optimize::try_simplex_minimize_with_budget(
            &[1.00, 1.70],
            &[1.24, 1.50],
            &[1.32, 1.90],
            2,
            5,
            1.0e-10,
            1.0e-10,
            0.0,
            0.0,
            false,
            false,
            1_074_698_122,
            |x| {
                pts.push((x[0], x[1]));
                // Mirrors upstream 7-eval simplex trajectory where expand
                // fails with 1e99 and inside contraction succeeds.
                let val = if x[0] < 0.97 {
                    search_driver::FAILED_TRACK_CHI2
                } else {
                    flexmesa_math::pow2(x[0] - 0.98)
                        + flexmesa_math::pow2(x[1] - 1.80)
                        + 0.046
                };
                Ok(val)
            },
        )
        .expect("simplex with enforce_bounds=false succeeds");
        assert!((pts[0].0 - 1.00).abs() < 1.0e-15);
        assert!((pts[0].1 - 1.70).abs() < 1.0e-15);
        assert!((pts[1].0 - 1.02).abs() < 1.0e-15);
        assert!((pts[2].1 - 1.80).abs() < 1.0e-15);
        // Iter 1 does reflect + expand (5 evals); Iter 2 runs because
        // `num_evals == 5 <= max_evals` (`mod_simplex.f90:127`), taking 7 evals.
        assert_eq!(evals, 7);
        assert!((x_best[0] - 0.98).abs() < 1.0e-12);
        assert!((x_best[1] - 1.80).abs() < 1.0e-12);
        assert!((f_best - 0.046).abs() < 1.0e-12);
    }

    #[test]
    fn test_astero_pass_logic_accepts_limit_num_and_failed_track_scores_1e99() {
        assert!(e2e::is_astero_search_passed(
            true,
            "limit_num_chi2_too_big",
            true,
            true
        ));
        assert!(e2e::is_astero_search_passed(
            true,
            "chi2_search_limit2",
            true,
            true
        ));
        assert!(e2e::is_astero_search_passed(true, "max_age", true, true));
        assert!(!e2e::is_astero_search_passed(
            true,
            "max_steps_cap",
            true,
            true
        ));
        assert!(!e2e::is_astero_search_passed(
            true,
            "min_timestep_limit",
            true,
            true
        ));
        assert!(!e2e::is_astero_search_passed(
            true,
            "limit_num_chi2_too_big",
            false,
            true
        ));
        assert!(!e2e::is_astero_search_passed(
            false,
            "limit_num_chi2_too_big",
            true,
            true
        ));

        let ctrl = flexmesa_star::StarControls::default();
        let job = flexmesa_star::StarJobControls::default();
        let model = flexmesa_star::load_zams_model(ctrl, job)
            .expect("1 Msun ZAMS for sentinel check");
        let failed = search_driver::failed_sample_result(
            &model,
            AsteroSearchControls::default().first_params(),
            1,
        );
        assert_eq!(failed.chi2, 1.0e99);
        assert_eq!(failed.chi2_seismo, 1.0e99);
        assert_eq!(failed.chi2_spectro, 1.0e99);
        assert!(failed.model_freqs.iter().all(Vec::is_empty));
    }
}
