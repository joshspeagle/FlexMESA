// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests for `flexmesa-astero` across all 8 upstream MESA
//! `astero/test_suite` cases (`astero_adipls`, `astero_gyre`, `example_astero`,
//! `fast_from_file`, `fast_newuoa`, `fast_scan_grid`, `fast_simplex`,
//! `surface_effects`) and offline synthetic staging.

#![forbid(unsafe_code)]
#![allow(clippy::similar_names, clippy::too_many_lines)]

use std::fs;
use std::path::PathBuf;

use flexmesa_astero::{
    copy_cleaned_astero_case, eval_combined_corr, eval_cubic_corr,
    eval_kjeldsen_corr, eval_model_astero_sample, eval_power_law_corr,
    eval_sonoi_corr, interpolate_l0_inertia, parse_astero_search_controls,
    run_astero_suite, run_astero_test_suite_case, TIER4_ASTERO_CASES,
};
use flexmesa_math::pow;
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_runner::DefaultStarExtras;
use flexmesa_star::star_job_init::init_or_update_model_from_bundle;
use flexmesa_star::StarError;

struct TempAsteroDirGuard {
    root: PathBuf,
}

impl TempAsteroDirGuard {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_astero_t4_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create cleaned astero temp root");
        Self { root }
    }
}

impl Drop for TempAsteroDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn astero_suite_dir() -> PathBuf {
    let base = std::env::var("MESA_DIR").map_or_else(
        |_| {
            let home =
                std::env::var("HOME").unwrap_or_else(|_| String::from("."));
            PathBuf::from(home)
                .join(".cache")
                .join("flexmesa")
                .join("mesa_test_inputs")
        },
        PathBuf::from,
    );
    base.join("astero").join("test_suite")
}

#[test]
fn test_offline_synthetic_mock_case_runs_two_steps_with_restart_parity() {
    let tmp_root = std::env::temp_dir()
        .join(format!("flexmesa_astero_mock_{}", std::process::id()));
    let tmp = tmp_root.join("example_astero");
    let _ = fs::remove_dir_all(&tmp_root);
    fs::create_dir_all(&tmp).expect("create temp mock dir");
    fs::write(tmp.join("rn"), "#!/bin/sh\ndo_one inlist final.mod\n")
        .expect("write rn");
    fs::write(
        tmp.join("inlist"),
        "&star_job\n  save_model_when_terminate = .true.\n  save_model_filename = 'final.mod'\n/\n&controls\n  initial_mass = 1.1\n  initial_z = 0.02\n  max_model_number = 2\n/\n",
    )
    .expect("write inlist");
    fs::write(
        tmp.join("inlist_astero_search_controls"),
        "&astero_search_controls\n  chi2_seismo_fraction = 0.666\n/\n",
    )
    .expect("write inlist_astero_search_controls");

    let rep = run_astero_test_suite_case(&tmp, true, Some(2), true)
        .expect("synthetic mock case should succeed");
    let _ = fs::remove_dir_all(&tmp_root);
    assert_eq!(rep.case_name, "example_astero");
    assert_eq!(rep.tier, 4);
    assert!(rep.passed);
    assert!(rep.photo_restart_verified);
    assert!(rep.assertions_passed);
    assert!(rep.total_steps >= 1);
}

#[test]
fn test_surface_effects_case_recovers_synthetic_targets_and_bison_solar_fits() {
    let case_dir = astero_suite_dir().join("surface_effects");
    assert!(
        case_dir.exists(),
        "missing surface_effects directory: {}",
        case_dir.display()
    );
    let raw_err = parse_astero_search_controls(&case_dir)
        .expect_err("raw surface_effects must reject unsupported controls");
    assert!(
        matches!(raw_err, StarError::InvalidInput(_))
            && raw_err.to_string().contains("unsupported control"),
        "unexpected raw surface_effects error: {raw_err}"
    );
    let tmp = TempAsteroDirGuard::new("surface_effects");
    let cleaned = tmp.root.join("surface_effects");
    copy_cleaned_astero_case(&case_dir, &cleaned)
        .expect("clean surface_effects case");
    let staged = cleaned.as_path();

    let ctrl = parse_astero_search_controls(staged)
        .expect("parse cleaned surface_effects controls");
    assert_eq!(ctrl.nl, [23, 21, 20, 17]);
    assert_eq!(ctrl.l0_n_obs.len(), 23);
    assert!(!ctrl.add_atmosphere);
    assert!(!ctrl.keep_surface_point);
    assert!(ctrl.add_center_point);

    let bundle = InlistBundle::from_file_in_case_dir(
        staged,
        "inlist_surface_effects_header",
    )
    .expect("parse surface_effects inlist");
    let mut extras = DefaultStarExtras;
    let model = init_or_update_model_from_bundle(&bundle, None, &mut extras)
        .expect("load s1.mod");

    let sample =
        eval_model_astero_sample(&model, &ctrl, ctrl.first_params(), 1, staged)
            .expect("eval surface_effects sample");
    let (mut obs_f, mut obs_sig, mut mod_f, mut mod_in) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for l in 0..4 {
        obs_f.extend_from_slice(&ctrl.freq_target[l]);
        obs_sig.extend_from_slice(&ctrl.freq_sigma[l]);
        mod_f.extend_from_slice(&sample.model_freqs[l]);
        mod_in.extend_from_slice(&sample.model_inertias[l]);
    }
    assert_eq!(mod_f.len(), 81);

    let l0_f = &sample.model_freqs[0];
    let l0_in = &sample.model_inertias[0];
    let q_nl: Vec<f64> = mod_f
        .iter()
        .zip(&mod_in)
        .map(|(&f, &e)| e / interpolate_l0_inertia(f, l0_f, l0_in))
        .collect();
    let nu_max = ctrl.nu_max;

    let target_a_div_r = -4.322_017_649_527_42_f64;
    let target_a3 = -1.979_289_417_129_323_8e-18_f64;
    let target_b1 = -5.125_972_793_567_201e-5_f64;
    let target_b3 = -1.446_307_489_879_336_1e-18_f64;
    let target_p0 = -6.157_582_281_235_104_f64;
    let target_p1 = 4.342_864_640_344_271_f64;
    let target_s0 = -4.380_735_022_728_04e-3_f64;
    let target_s1 = 7.143_783_592_158_95_f64;

    // 1. Synthetic data recovery (`< 1e-12`)
    let syn_cubic: Vec<f64> = mod_f
        .iter()
        .zip(&mod_in)
        .map(|(&f, &e)| f + target_a3 * (f * f * f) / e)
        .collect();
    let (_, a3_syn) =
        eval_cubic_corr(&syn_cubic, &obs_sig, &mod_f, &mod_in, 1.0);
    assert!(((a3_syn / target_a3) - 1.0).abs() < 1.0e-12);

    let syn_comb: Vec<f64> = mod_f
        .iter()
        .zip(&mod_in)
        .map(|(&f, &e)| f + (target_b1 / f + target_b3 * (f * f * f)) / e)
        .collect();
    let (_, b3_syn, b1_syn) =
        eval_combined_corr(&syn_comb, &obs_sig, &mod_f, &mod_in, 1.0);
    assert!(((b1_syn / target_b1) - 1.0).abs() < 1.0e-12);
    assert!(((b3_syn / target_b3) - 1.0).abs() < 1.0e-12);

    let syn_pl: Vec<f64> = mod_f
        .iter()
        .zip(&q_nl)
        .map(|(&f, &q)| f + target_p0 * pow(f / nu_max, target_p1) / q)
        .collect();
    let (_, p0_syn, p1_syn) = eval_power_law_corr(
        &syn_pl, &obs_sig, &mod_f, &mod_in, l0_f, l0_in, nu_max, 1.0,
    );
    assert!(((p0_syn / target_p0) - 1.0).abs() < 1.0e-12);
    assert!(((p1_syn / target_p1) - 1.0).abs() < 1.0e-12);

    let syn_sonoi: Vec<f64> = mod_f
        .iter()
        .zip(&q_nl)
        .map(|(&f, &q)| {
            f + target_s0
                * nu_max
                * (1.0 - 1.0 / (1.0 + pow(f / nu_max, target_s1)))
                / q
        })
        .collect();
    let (_, s0_syn, s1_syn) = eval_sonoi_corr(
        &syn_sonoi, &obs_sig, &mod_f, &mod_in, l0_f, l0_in, nu_max, 1.0,
    );
    assert!(((s0_syn / target_s0) - 1.0).abs() < 1.0e-12);
    assert!(((s1_syn / target_s1) - 1.0).abs() < 1.0e-12);

    // 2. BiSON solar fits (`solar_tol = 10.0`)
    let (_, a_div_r, r_k) = eval_kjeldsen_corr(
        &ctrl.freq_target[0],
        l0_f,
        l0_in,
        &ctrl.l0_n_obs,
        ctrl.delta_nu,
        &mod_f,
        &mod_in,
        nu_max,
        ctrl.correction_b,
        1.0,
    );
    assert!(
        r_k > 0.95 && r_k < 1.05,
        "unexpected homology factor r={r_k}"
    );
    assert!(
        ((a_div_r / target_a_div_r) - 1.0).abs() < 10.0,
        "a_div_r={a_div_r} vs target={target_a_div_r}"
    );
}

fn assert_uncapped_case_passes(case_name: &str) {
    let suite_dir = astero_suite_dir();
    assert!(
        suite_dir.exists(),
        "missing astero test_suite directory: {}",
        suite_dir.display()
    );
    let raw_case = suite_dir.join(case_name);
    let raw_err = run_astero_test_suite_case(&raw_case, true, Some(2), true)
        .expect_err("raw astero case must fail on unsupported controls");
    assert!(
        matches!(raw_err, StarError::InvalidInput(_))
            && raw_err.to_string().contains("unsupported control"),
        "unexpected raw astero error for {case_name}: {raw_err}"
    );
    let tmp = TempAsteroDirGuard::new(case_name);
    let cleaned = tmp.root.join(case_name);
    copy_cleaned_astero_case(&raw_case, &cleaned).expect("clean astero case");
    let rep = run_astero_test_suite_case(&cleaned, true, None, true)
        .unwrap_or_else(|e| panic!("astero case {case_name} errored: {e}"));
    assert_eq!(rep.case_name, case_name);
    assert!(
        rep.passed,
        "astero case {} failed: assertions_passed={}, restart_verified={}, steps={}, teff={:.1}, logL={:.4}, logR={:.4}, h1c={:.5}, term={}",
        rep.case_name,
        rep.assertions_passed,
        rep.photo_restart_verified,
        rep.total_steps,
        rep.final_teff,
        rep.final_log_l,
        rep.final_log_r,
        rep.final_center_h1,
        rep.termination_code
    );
    assert!(rep.photo_restart_verified);
    assert!(rep.assertions_passed);
    eprintln!(
        "  [uncapped] {} steps={} wall={:.2}s teff={:.1} logL={:.4} center_h1={:.5} {}",
        rep.case_name,
        rep.total_steps,
        rep.total_runtime_sec,
        rep.final_teff,
        rep.final_log_l,
        rep.final_center_h1,
        rep.termination_code
    );
}

#[test]
fn test_astero_adipls_and_gyre_uncapped_pass() {
    for case_name in ["astero_adipls", "astero_gyre", "surface_effects"] {
        assert_uncapped_case_passes(case_name);
    }
}

#[test]
fn test_all_eight_astero_cases_bounded_eight_steps_pass() {
    let suite_dir = astero_suite_dir();
    assert!(
        suite_dir.exists(),
        "missing astero test_suite directory: {}",
        suite_dir.display()
    );
    let raw_err = run_astero_suite(&suite_dir, true, Some(2), true).expect_err(
        "raw $MESA_DIR astero suite must reject unsupported controls",
    );
    assert!(
        matches!(raw_err, StarError::InvalidInput(_))
            && raw_err.to_string().contains("unsupported control"),
        "unexpected raw astero suite error: {raw_err}"
    );
    let tmp = TempAsteroDirGuard::new("all8");
    let cleaned_suite = tmp.root.join("astero").join("test_suite");
    for &case_name in &TIER4_ASTERO_CASES {
        copy_cleaned_astero_case(
            &suite_dir.join(case_name),
            &cleaned_suite.join(case_name),
        )
        .expect("clean astero suite case");
    }
    let reports = run_astero_suite(&cleaned_suite, true, Some(8), true)
        .expect("run all 8 cleaned astero cases bounded to 8 steps");
    assert_eq!(reports.len(), TIER4_ASTERO_CASES.len());
    for (rep, &expected_name) in reports.iter().zip(&TIER4_ASTERO_CASES) {
        assert_eq!(rep.case_name, expected_name);
        assert!(
            rep.passed,
            "bounded astero case {} failed: assertions_passed={}, restart_verified={}, term={}",
            rep.case_name,
            rep.assertions_passed,
            rep.photo_restart_verified,
            rep.termination_code
        );
        assert!(rep.photo_restart_verified);
        assert!(rep.assertions_passed);
    }
}

#[test]
#[ignore = "unverified uncapped astero campaign; tracked in #143 and #162"]
fn test_example_astero_uncapped() {
    assert_uncapped_case_passes("example_astero");
}

#[test]
#[ignore = "unverified uncapped astero campaign; tracked in #143 and #162"]
fn test_fast_from_file_uncapped() {
    assert_uncapped_case_passes("fast_from_file");
}

#[test]
#[ignore = "unverified uncapped astero campaign; tracked in #143 and #162"]
fn test_fast_scan_grid_uncapped() {
    assert_uncapped_case_passes("fast_scan_grid");
}

#[test]
#[ignore = "unverified uncapped astero campaign; tracked in #143 and #162"]
fn test_fast_simplex_uncapped() {
    assert_uncapped_case_passes("fast_simplex");
}

#[test]
#[ignore = "unverified uncapped astero campaign; tracked in #143 and #162"]
fn test_fast_newuoa_uncapped() {
    assert_uncapped_case_passes("fast_newuoa");
}
