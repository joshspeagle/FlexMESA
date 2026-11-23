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
    run_astero_test_suite_case, run_astero_test_suite_case_with_case_hooks,
    AsteroCaseHooks, AsteroSearchControls, MAX_CONSTRAINTS,
};

const ASTERO_CASES: [&str; 8] = [
    "astero_adipls",
    "astero_gyre",
    "example_astero",
    "fast_from_file",
    "fast_newuoa",
    "fast_scan_grid",
    "fast_simplex",
    "surface_effects",
];
use flexmesa_const::{MixingType, RSUN};
use flexmesa_math::pow;
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_runner::DefaultStarExtras;
use flexmesa_star::model::StarModel;
use flexmesa_star::report_utils::surface_avg_x_slot;
use flexmesa_star::star_job_init::init_or_update_model_from_bundle;
use flexmesa_star::{
    star_get_history_output, InlistLoadOptions, StarError, StarLogsOptions,
};

/// Test-side port of the search cases' own `set_param`
/// (`astero/test_suite/fast_newuoa/src/run_star_extras.f90:115-184`, identical
/// in `example_astero:115-184`, `fast_scan_grid:121-190` and
/// `fast_simplex:122-191`; `fast_from_file:122-199` adds `semiconv`,
/// `thermohal` and `diffusion` at `:185-193`). The Python twin used by the
/// harness is `tests/case_extras/astero_extras.py`.
struct CaseSetParam {
    fast_from_file: bool,
}

impl AsteroCaseHooks for CaseSetParam {
    fn set_param(
        &mut self,
        bundle: &mut InlistBundle,
        ctrl: &AsteroSearchControls,
        name: &str,
        val: f64,
    ) -> Result<(), StarError> {
        let (job, c) = (&mut bundle.job, &mut bundle.controls);
        match name {
            "initial_mass" => {
                job.new_mass = val;
                job.relax_mass = true;
                c.initial_mass = val;
            }
            "initial_Y" => {
                job.initial_he3 = ctrl.y_frac_he3 * val;
                job.initial_he4 = val - job.initial_he3;
                job.set_uniform_initial_composition = true;
            }
            "initial_FeH" => {
                let z_div_x = ctrl.z_div_x_solar * flexmesa_math::exp10(val);
                let x = if ctrl.y_depends_on_z {
                    let cc = 1.0 + z_div_x * (1.0 + ctrl.dydz);
                    let x = (1.0 - ctrl.y0) / cc;
                    let y = (ctrl.y0 + z_div_x * (ctrl.dydz + ctrl.y0)) / cc;
                    job.initial_he3 = ctrl.y_frac_he3 * y;
                    job.initial_he4 = y - job.initial_he3;
                    x
                } else {
                    let y = job.initial_he3 / ctrl.y_frac_he3;
                    (1.0 - y) / (1.0 + z_div_x)
                };
                job.initial_h1 = x;
                job.initial_h2 = 0.0;
                job.set_uniform_initial_composition = true;
            }
            "alpha" => c.mixing_length_alpha = val,
            "f_ov" => {
                let ov = &mut c.mix_wind.overshoot_sets[0];
                let on = val > 0.0;
                let any = |b: bool| String::from(if b { "any" } else { "" });
                ov.scheme = String::from(if on { "exponential" } else { "" });
                (ov.zone_type, ov.zone_loc, ov.bdy_loc) =
                    (any(on), any(on), any(on));
                (ov.f, ov.f0) = if on {
                    (val, ctrl.f0_ov_div_f_ov * val)
                } else {
                    (0.0, 0.0)
                };
            }
            "semiconv" if self.fast_from_file => c.alpha_semiconvection = val,
            "thermohal" if self.fast_from_file => c.thermohaline_coeff = val,
            "diffusion" if self.fast_from_file => {
                c.diffusion.diffusion_class_factor.fill(val);
            }
            other => {
                return Err(StarError::InvalidInput(format!(
                    "invalid name in set_param {other}"
                )));
            }
        }
        Ok(())
    }

    /// `set_constraint_value`, identical in the five search cases
    /// (`astero/test_suite/fast_newuoa/src/run_star_extras.f90:58-112`).
    fn set_constraint_value(
        &mut self,
        model: &StarModel,
        ctrl: &AsteroSearchControls,
        name: &str,
    ) -> Result<f64, StarError> {
        let slot = |iso: &str| {
            model
                .iso_names
                .iter()
                .position(|n| n.eq_ignore_ascii_case(iso))
        };
        let x = surface_avg_x_slot(model, slot("h1")).max(1.0e-10);
        let y = surface_avg_x_slot(model, slot("he3"))
            + surface_avg_x_slot(model, slot("he4"));
        let z = (1.0 - x - y).clamp(1.0e-99, 1.0);
        Ok(match name {
            "M_H" => flexmesa_math::log10(z / x / ctrl.z_div_x_solar),
            "surface_Z_div_X" => z / x,
            "surface_He" => y,
            "Rcz" => {
                let conv = MixingType::ConvectiveMixing;
                let found = (0..model.nz - 1).find(|&i| {
                    model.mixing_type[i + 1] != conv
                        && model.mixing_type[i] == conv
                        && model.r[i + 1] > 0.25 * RSUN
                        && model.r[i] < 0.9 * RSUN
                });
                // Upstream leaves `val` unset here: no defined behaviour
                // (DECISIONS #43, deferred #438).
                let Some(i) = found else {
                    return Err(StarError::InvalidInput(String::from(
                        "set_constraint_value('Rcz'): val left unset",
                    )));
                };
                model.r[i] / RSUN
            }
            _ => star_get_history_output(model, name)?,
        })
    }
}

/// Runs a cleaned astero case with its test-side `set_param`.
fn run_case_with_case_hooks(
    case_dir: &std::path::Path,
    case_name: &str,
    max_steps: Option<usize>,
    logs_opts: StarLogsOptions<'_>,
) -> Result<flexmesa_star::CaseRunReport, StarError> {
    let mut hook = CaseSetParam {
        fast_from_file: case_name == "fast_from_file",
    };
    run_astero_test_suite_case_with_case_hooks(
        case_dir,
        true,
        max_steps,
        true,
        logs_opts,
        InlistLoadOptions::default(),
        Some(&mut hook),
    )
}

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
        "&star_job\n  astero_just_call_my_extras_check_model = .true.\n  save_model_when_terminate = .true.\n  save_model_filename = 'final.mod'\n/\n&controls\n  initial_mass = 1.1\n  initial_z = 0.02\n  max_model_number = 2\n/\n&kap\n  zbase = 0.02d0\n/\n",
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
    assert_eq!(rep.tier, 0);
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

    let sample = eval_model_astero_sample(
        &model,
        &ctrl,
        &[f64::NAN; MAX_CONSTRAINTS],
        ctrl.first_params(),
        1,
        staged,
    )
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
    let solar_tol = 10.0_f64;
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
        ((a_div_r / target_a_div_r) - 1.0).abs() < solar_tol,
        "a_div_r={a_div_r} vs target={target_a_div_r}"
    );
    let (_, a3_sol) = eval_cubic_corr(&obs_f, &obs_sig, &mod_f, &mod_in, 1.0);
    assert!(a3_sol <= 0.0 && ((a3_sol / target_a3) - 1.0).abs() < solar_tol);

    let (_, b3_sol, b1_sol) =
        eval_combined_corr(&obs_f, &obs_sig, &mod_f, &mod_in, 1.0);
    assert!(
        b3_sol <= 0.0
            && ((b1_sol / target_b1) - 1.0).abs() < solar_tol
            && ((b3_sol / target_b3) - 1.0).abs() < solar_tol
    );

    let (_, p0_sol, p1_sol) = eval_power_law_corr(
        &obs_f, &obs_sig, &mod_f, &mod_in, l0_f, l0_in, nu_max, 1.0,
    );
    assert!(
        p0_sol <= 0.0
            && ((p0_sol / target_p0) - 1.0).abs() < solar_tol
            && ((p1_sol / target_p1) - 1.0).abs() < solar_tol
    );

    let (_, s0_sol, s1_sol) = eval_sonoi_corr(
        &obs_f, &obs_sig, &mod_f, &mod_in, l0_f, l0_in, nu_max, 1.0,
    );
    assert!(
        s0_sol <= 0.0
            && s1_sol >= 0.0
            && ((s0_sol / target_s0) - 1.0).abs() < solar_tol
            && ((s1_sol / target_s1) - 1.0).abs() < solar_tol
    );
}

fn parse_search_chi2(term_code: &str) -> Option<(usize, f64)> {
    let (_, after_samples) = term_code.split_once("samples=")?;
    let (samples_str, after_chi2) = after_samples.split_once("_best_chi2=")?;
    let (chi2_str, _) = after_chi2.split_once("_term=")?;
    Some((samples_str.parse().ok()?, chi2_str.parse().ok()?))
}

fn copy_dir_recursive(
    src: &std::path::Path,
    dst: &std::path::Path,
) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let target = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

fn assert_uncapped_case_passes(case_name: &str) {
    let suite_dir = astero_suite_dir();
    assert!(
        suite_dir.exists(),
        "missing astero test_suite directory: {}",
        suite_dir.display()
    );
    let raw_case = suite_dir.join(case_name);
    let tmp = TempAsteroDirGuard::new(case_name);
    let staged = tmp.root.join(case_name);
    if case_name == "astero_gyre" {
        copy_dir_recursive(&raw_case, &staged).expect("copy raw astero_gyre");
    } else {
        let raw_err = run_astero_test_suite_case(&raw_case, true, Some(2), true)
            .expect_err("raw astero case must fail on unsupported controls");
        assert!(
            matches!(raw_err, StarError::InvalidInput(_))
                && raw_err.to_string().contains("unsupported control"),
            "unexpected raw astero error for {case_name}: {raw_err}"
        );
        copy_cleaned_astero_case(&raw_case, &staged)
            .expect("clean astero case");
    }
    let logs_opts = StarLogsOptions::default();
    let rep = run_case_with_case_hooks(&staged, case_name, None, logs_opts)
        .unwrap_or_else(|e| panic!("astero case {case_name} errored: {e}"));
    assert_eq!(rep.case_name, case_name);
    if case_name == "example_astero" {
        let (samples, best_chi2) = parse_search_chi2(&rep.termination_code)
            .unwrap_or_else(|| {
                panic!(
                    "unexpected search termination string: {}",
                    rep.termination_code
                )
            });
        assert_eq!(samples, 1);
        assert!(
            best_chi2 > 1.0 && best_chi2 < 15.0,
            "example_astero best_chi2={best_chi2} outside expected (1, 15) oracle"
        );
    } else if case_name == "fast_newuoa" {
        let (samples, _) = parse_search_chi2(&rep.termination_code)
            .expect("fast_newuoa sample count");
        assert!(samples >= 4);
    }
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

// These three cases set `astero_just_call_my_extras_check_model`, so they run
// as plain star runs (`astero/private/astero_run_support.f90:171-174`). Their
// upstream `run_star_extras.f90` (the ADIPLS/GYRE frequency checks and the
// `freqs.dat` output) is not ported, so this covers the star run only.
// One test per case, so the three runs can go in parallel and be placed in
// different CI shards (`scripts/ci/cargo_test_shard.py`).
#[test]
fn test_just_call_my_extras_astero_adipls_runs_as_plain_star_run() {
    assert_uncapped_case_passes("astero_adipls");
}

#[test]
fn test_just_call_my_extras_astero_gyre_runs_as_plain_star_run() {
    assert_uncapped_case_passes("astero_gyre");
}

#[test]
fn test_just_call_my_extras_surface_effects_runs_as_plain_star_run() {
    assert_uncapped_case_passes("surface_effects");
}

#[test]
fn test_all_eight_astero_cases_bounded_eight_steps_pass() {
    let suite_dir = astero_suite_dir();
    assert!(
        suite_dir.exists(),
        "missing astero test_suite directory: {}",
        suite_dir.display()
    );
    let raw_err = run_astero_test_suite_case(
        &suite_dir.join("astero_adipls"),
        true,
        Some(2),
        true,
    )
    .expect_err("raw $MESA_DIR astero case must reject unsupported controls");
    assert!(
        matches!(raw_err, StarError::InvalidInput(_))
            && raw_err.to_string().contains("unsupported control"),
        "unexpected raw astero error: {raw_err}"
    );
    let tmp = TempAsteroDirGuard::new("all8");
    let cleaned_suite = tmp.root.join("astero").join("test_suite");
    for &case_name in &ASTERO_CASES {
        copy_cleaned_astero_case(
            &suite_dir.join(case_name),
            &cleaned_suite.join(case_name),
        )
        .expect("clean astero suite case");
    }
    // A search case without its `set_param` hook fails loudly instead of
    // running with parameters nobody applied.
    let no_hook = run_astero_test_suite_case(
        &cleaned_suite.join("fast_newuoa"),
        true,
        Some(8),
        true,
    )
    .expect_err("search case without set_param must fail");
    assert!(
        no_hook.to_string().contains("set_param hook"),
        "unexpected error without set_param: {no_hook}"
    );
    for &case_name in &ASTERO_CASES {
        let rep = run_case_with_case_hooks(
            &cleaned_suite.join(case_name),
            case_name,
            Some(8),
            StarLogsOptions::default(),
        )
        .unwrap_or_else(|e| {
            panic!("bounded astero case {case_name} errored: {e}")
        });
        assert_eq!(rep.case_name, case_name);
        assert!(rep.photo_restart_verified);
        assert!(rep.assertions_passed);
        if case_name == "surface_effects" {
            assert!(
                rep.passed,
                "surface_effects must pass at 0 steps: {rep:?}"
            );
        } else {
            assert!(
                !rep.passed,
                "capped 8-step astero case {} must not mark passed=true: {rep:?}",
                rep.case_name
            );
            assert!(rep.total_steps >= 1);
        }
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
