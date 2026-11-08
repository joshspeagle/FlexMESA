// SPDX-License-Identifier: LGPL-3.0-or-later
//! Asteroseismic scaling relations, surface-effect frequency corrections,
//! separation ratios ($r_{01}, r_{10}, r_{02}$), composite $\chi^2$ evaluation,
//! and optimization drivers for `FlexMESA`.
//!
//! Corresponds to the upstream Fortran MESA `astero/` module (`astero_def`,
//! `astero_lib`, `astero_support`, `astero_run_support`, `extras_support`, and
//! `pgstar_astero_plots`) and `star/private/report.f90` / `star_utils.f90`.

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

pub mod optimize;
pub mod ratios_chi2;
pub mod scaling;
pub mod surface_corr;

pub use optimize::{
    map_bobyqa_param, nelder_mead_minimize, powell_quadratic_minimize,
    scan_cartesian_grid,
};
pub use ratios_chi2::{
    eval_astero_chi2, eval_frequency_ratios_r01_r10, eval_frequency_ratios_r02,
    match_modes_by_degree, AsteroChi2Weights,
};
pub use scaling::{
    eval_acoustic_profile, eval_delta_nu_asymptotic,
    eval_delta_nu_radius_scaling, eval_delta_nu_scaling,
    eval_delta_pg_bildsten2012, eval_delta_pg_traditional,
    eval_glitch_acoustic_depths, eval_nu_ac, eval_nu_max,
    eval_radial_delta_nu_ols, DELTA_NU_SUN, NU_MAX_SUN, TEFF_SUN_ASTERO,
};
pub use surface_corr::{
    eval_combined_corr, eval_cubic_corr, eval_kjeldsen_corr,
    eval_power_law_corr, eval_sonoi_corr, interpolate_l0_inertia,
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
            .map(|&f| 1.0e-7 * ((3100.0 - f) / 1000.0).exp())
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
        assert!((p0_fit / target_p0 - 1.0).abs() < 1.0e-8);
        assert!((p1_fit / target_p1 - 1.0).abs() < 1.0e-8);

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
        assert!((s0_fit / target_s0 - 1.0).abs() < 1.0e-6);
        assert!((s1_fit / target_s1 - 1.0).abs() < 1.0e-6);
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
            |x| (x[0] - target[0]).powi(2) + 10.0 * (x[1] - target[1]).powi(2),
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
            |x| (x[0] - target[0]).powi(2) + 10.0 * (x[1] - target[1]).powi(2),
        );
        assert!(f_pq < 1.0e-10);
        assert!((x_pq[0] - target[0]).abs() < 1.0e-5);
        assert!((x_pq[1] - target[1]).abs() < 1.0e-5);
    }
}
