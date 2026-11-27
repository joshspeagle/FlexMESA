// SPDX-License-Identifier: GPL-3.0-or-later
//! Zero-`$MESA_DIR` unit and track invariant tests for `astero` search
//! against MESA `26.4.1` (`from_file`, unused `'Rcz'`, `checking_age`, and
//! `extras_after_evolve`).

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use flexmesa_astero::{
    parse_astero_search_controls, run_single_astero_track_with_case_hooks,
    AsteroCaseHooks, AsteroSearchControls,
};
use flexmesa_const::SECYER;
use flexmesa_star::controls::{
    AtmOption, AtmTTauOpacity, AtmTTauRelation, StarControls, StarError,
    StarJobControls,
};
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_runner::StarLogsOptions;
use flexmesa_star::model::StarModel;
use flexmesa_star::star_lib::star_write_model;
use flexmesa_star::{load_zams_model, set_hydro_vars, star_get_history_output};

static TEST_DIR_SEQ: AtomicUsize = AtomicUsize::new(1);

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        let id = TEST_DIR_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "flexmesa_astero_162_w2_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn write(&self, rel: &str, text: &str) {
        let dst = self.path.join(rel);
        if let Some(p) = dst.parent() {
            fs::create_dir_all(p).expect("create parent");
        }
        fs::write(&dst, text).expect("write fixture");
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn hydrated_solar_zams() -> StarModel {
    let controls = StarControls {
        zbase: 0.02,
        calculate_brunt_n2: true,
        atm_option: AtmOption::TTau,
        atm_t_tau_relation: AtmTTauRelation::Eddington,
        atm_t_tau_opacity: AtmTTauOpacity::Fixed,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    set_hydro_vars(&mut model, false, true).expect("set_hydro_vars");
    model
}

fn write_zams_star_case(
    dir: &TempDirGuard,
    max_years_dt: f64,
    extra_star_job: &str,
    extra_controls: &str,
) -> StarModel {
    let model = hydrated_solar_zams();
    star_write_model(&model, &dir.path.join("start.mod"))
        .expect("write start.mod");
    let max_mod = model.model_number + 10;
    let inlist = format!(
        "&star_job
  load_saved_model = .true.
  load_model_filename = 'start.mod'
  pgstar_flag = .false.
{extra_star_job}
/ ! end of star_job namelist

&controls
  max_model_number = {max_mod}
  max_years_for_timestep = {max_years_dt:.6e}
  calculate_brunt_n2 = .false.
  atm_option = 'T_tau'
  atm_T_tau_relation = 'Eddington'
  atm_T_tau_opacity = 'fixed'
  write_profiles_flag = .false.
{extra_controls}
/ ! end of controls namelist

&kap
  Zbase = 0.02d0
/ ! end of kap namelist
"
    );
    dir.write("inlist", &inlist);
    model
}

struct MockSearchHooks;

impl AsteroCaseHooks for MockSearchHooks {
    fn set_param(
        &mut self,
        _bundle: &mut InlistBundle,
        _ctrl: &AsteroSearchControls,
        name: &str,
        _val: f64,
    ) -> Result<(), StarError> {
        Err(StarError::InvalidInput(format!(
            "unexpected set_param '{name}'"
        )))
    }

    fn set_constraint_value(
        &mut self,
        model: &StarModel,
        _ctrl: &AsteroSearchControls,
        name: &str,
    ) -> Result<f64, StarError> {
        match name {
            "Teff" => Ok(model.teff),
            "Rcz" => Err(StarError::InvalidInput(String::from(
                "set_constraint_value('Rcz'): no convective-zone base \
                 between 0.25 and 0.9 Rsun, so upstream leaves val unset \
                 (astero/test_suite/example_astero/src/run_star_extras.f90:96-105)",
            ))),
            other => star_get_history_output(model, other),
        }
    }
}

#[test]
fn test_from_file_skips_non_varied_columns_and_validates_varied_columns() {
    // Replicates `fast_from_file` where `vary_param(5) = .false.` (`f_ov`)
    // keeps `first_param(5) = 0.0` even though `parameters.txt` col 7 = 0.015
    // (`astero/private/astero_run_support.f90:277-318`,
    // `extras_support.f90:974-988`).
    let dir = TempDirGuard::new("from_file_vary");
    dir.write(
        "inlist_astero_search_controls",
        "&astero_search_controls
  search_type = 'from_file'
  filename_for_parameters = 'parameters.txt'
  param_name(1) = 'mass'
  first_param(1) = 1.005d0
  vary_param(1) = .false.
  file_column_for_param(1) = 3
  param_name(5) = 'f_ov'
  first_param(5) = 0.0d0
  vary_param(5) = .false.
  file_column_for_param(5) = 7
  param_name(6) = 'semiconv'
  first_param(6) = 1.0d0
  vary_param(6) = .true.
  file_column_for_param(6) = 8
  param_name(8) = 'diffusion'
  first_param(8) = 1.0d0
  vary_param(8) = .true.
  file_column_for_param(8) = 10
/
",
    );
    dir.write(
        "parameters.txt",
        "sample chi2 mass Y FeH alpha f_ov semiconv thermohal diffusion
1 2 3 4 5 6 7 8 9 10
1 2.0 1.000 0.25 0.1 1.7 0.015 0.000 1.0 1.0
2 3.0 1.000 0.25 0.1 1.7 0.015 0.005 1.0 0.5
",
    );
    let ctrl = parse_astero_search_controls(&dir.path).expect("parse astero");
    let rows = ctrl
        .read_parameters_from_file(&dir.path)
        .expect("read_parameters_from_file");
    assert_eq!(rows.len(), 2);
    assert!((rows[0].initial_mass - 1.005).abs() <= 1.0e-15);
    assert!((rows[0].f_ov - 0.0).abs() <= 1.0e-15);
    assert!((rows[0].my_param1 - 0.0).abs() <= 1.0e-15);
    assert!((rows[0].my_param3 - 1.0).abs() <= 1.0e-15);
    assert!((rows[1].initial_mass - 1.005).abs() <= 1.0e-15);
    assert!((rows[1].f_ov - 0.0).abs() <= 1.0e-15);
    assert!((rows[1].my_param1 - 0.005).abs() <= 1.0e-15);
    assert!((rows[1].my_param3 - 0.5).abs() <= 1.0e-15);

    // Non-varied parameter with non-positive column still keeps `first_param`,
    // whereas a varied parameter with non-positive column errors
    // (`astero_run_support.f90:279-283`).
    let bad_dir = TempDirGuard::new("from_file_bad_col");
    bad_dir.write(
        "inlist_astero_search_controls",
        "&astero_search_controls
  search_type = 'from_file'
  filename_for_parameters = 'parameters.txt'
  param_name(1) = 'mass'
  first_param(1) = 1.0d0
  vary_param(1) = .true.
  file_column_for_param(1) = 0
  param_name(5) = 'f_ov'
  first_param(5) = 0.02d0
  vary_param(5) = .false.
  file_column_for_param(5) = 0
/
",
    );
    bad_dir.write("parameters.txt", "col1\n1.2\n");
    let bad_ctrl =
        parse_astero_search_controls(&bad_dir.path).expect("parse bad ctrl");
    let err = bad_ctrl
        .read_parameters_from_file(&bad_dir.path)
        .expect_err("varied param with col <= 0 must error");
    assert!(matches!(err, StarError::InvalidInput(_)));
}

fn run_constraint_subcase(
    tag: &str,
    astero_inlist: &str,
) -> Result<
    (
        flexmesa_astero::AsteroSampleResult,
        flexmesa_star::inlist_runner::StageRunReport,
        StarModel,
    ),
    StarError,
> {
    let star_job = "  set_initial_age = .true.\n  initial_age = 2.0d7\n  set_initial_dt = .true.\n  years_for_initial_dt = 1.0d3";
    let dir = TempDirGuard::new(tag);
    let model = write_zams_star_case(&dir, 1.0e4, star_job, "");
    dir.write("inlist_astero_search_controls", astero_inlist);
    let bundle =
        InlistBundle::from_file_in_case_dir(&dir.path, "inlist").expect("in");
    let ctrl = parse_astero_search_controls(&dir.path).expect("astero");
    let mut hooks = MockSearchHooks;
    run_single_astero_track_with_case_hooks(
        &bundle,
        &ctrl,
        ctrl.first_params(),
        1,
        Some(model.model_number + 3),
        &dir.path,
        StarLogsOptions::default(),
        Some(&mut hooks),
    )
}

#[test]
fn test_unused_rcz_unset_allowed_on_inactive_slot() {
    // Inactive `'Rcz'` slot (`include_constraint_in_chi2_spectro = .false.` and
    // `sigmas_coeff_for_constraint_limit = 0`) keeps `constraint_value = 0.0`
    // (`astero/private/astero_run_support.f90:165`,
    // `astero/test_suite/example_astero/src/run_star_extras.f90:72,96-105`).
    let (sample, stage, _end) = run_constraint_subcase(
        "rcz_unused_ok",
        "&astero_search_controls
  search_type = 'use_first_values'
  chi2_seismo_fraction = 0.0d0
  min_age_limit = 1.0d6
  lnuc_div_l_limit = 0.0d0
  max_yrs_dt_when_cold = 1.0d4
  constraint_name(1) = 'Teff'
  include_constraint_in_chi2_spectro(1) = .true.
  constraint_target(1) = 5777.0d0
  constraint_sigma(1) = 65.0d0
  constraint_name(6) = 'Rcz'
  include_constraint_in_chi2_spectro(6) = .false.
  sigmas_coeff_for_constraint_limit(6) = 0.0d0
/
",
    )
    .expect("unused unset Rcz slot must succeed");
    assert!(stage.steps_taken >= 1);
    assert!(sample.chi2.is_finite() && sample.chi2 < 1.0e90);
    assert!((sample.constraint_values[5] - 0.0).abs() <= 1.0e-15);
}

#[test]
fn test_unset_rcz_or_unknown_constraint_fails_when_read_or_unrecognized() {
    // Active `'Rcz'` slots (`include` or `sigmas_coeff != 0` with `sigma > 0`)
    // and unknown `star_get_history_output` names (`:108-109`) still abort.
    let cases = [
        (
            "rcz_active_fail",
            "&astero_search_controls
  search_type = 'use_first_values'
  chi2_seismo_fraction = 0.0d0
  min_age_limit = 1.0d6
  lnuc_div_l_limit = 0.0d0
  constraint_name(6) = 'Rcz'
  include_constraint_in_chi2_spectro(6) = .true.
  constraint_target(6) = 0.71d0
  constraint_sigma(6) = 0.01d0
/
",
        ),
        (
            "rcz_limit_fail",
            "&astero_search_controls
  search_type = 'use_first_values'
  chi2_seismo_fraction = 0.0d0
  min_age_limit = 1.0d6
  lnuc_div_l_limit = 0.0d0
  constraint_name(6) = 'Rcz'
  include_constraint_in_chi2_spectro(6) = .false.
  constraint_target(6) = 0.71d0
  constraint_sigma(6) = 0.05d0
  sigmas_coeff_for_constraint_limit(6) = 5.0d0
/
",
        ),
        (
            "unknown_col_fail",
            "&astero_search_controls
  search_type = 'use_first_values'
  chi2_seismo_fraction = 0.0d0
  min_age_limit = 1.0d6
  lnuc_div_l_limit = 0.0d0
  constraint_name(6) = 'bogus_col'
  include_constraint_in_chi2_spectro(6) = .false.
  sigmas_coeff_for_constraint_limit(6) = 0.0d0
/
",
        ),
    ];
    for (tag, inlist_text) in cases {
        let err = run_constraint_subcase(tag, inlist_text)
            .expect_err("read/unknown constraint slot must fail loudly");
        assert!(
            matches!(err, StarError::InvalidInput(_)),
            "case {tag}: {err}"
        );
    }
}

#[test]
fn test_checking_age_hits_target_age_via_check_model_and_finish_step() {
    // Evolves with ZAMS-safe steps (`0.0 -> 1000.0 -> 1500.0` yr) to verify
    // `extras_check_model` + `extras_finish_step` stops cleanly at `age_target`
    // (`astero/private/extras_support.f90:96-102,153,935`).
    let dir = TempDirGuard::new("checking_age_target");
    let star_job = "  set_initial_age = .true.\n  initial_age = 0.0d0\n  set_initial_dt = .true.\n  years_for_initial_dt = 1.0d3";
    let base = write_zams_star_case(&dir, 2.0e3, star_job, "");
    dir.write(
        "inlist_astero_search_controls",
        "&astero_search_controls
  search_type = 'use_first_values'
  chi2_seismo_fraction = 0.0d0
  eval_chi2_at_target_age_only = .true.
  age_target = 1500.0d0
  num_smaller_steps_before_age_target = 50
  dt_for_smaller_steps_before_age_target = 500.0d0
  min_age_limit = 0.0d0
  lnuc_div_l_limit = 0.0d0
  constraint_name(1) = 'Teff'
  include_constraint_in_chi2_spectro(1) = .true.
  constraint_target(1) = 5777.0d0
  constraint_sigma(1) = 65.0d0
/
",
    );
    let bundle =
        InlistBundle::from_file_in_case_dir(&dir.path, "inlist").expect("in");
    let ctrl = parse_astero_search_controls(&dir.path).expect("astero");
    let mut hooks = MockSearchHooks;
    let (sample, stage, end_model) = run_single_astero_track_with_case_hooks(
        &bundle,
        &ctrl,
        ctrl.first_params(),
        1,
        Some(base.model_number + 6),
        &dir.path,
        StarLogsOptions::default(),
        Some(&mut hooks),
    )
    .expect("checking_age track succeeds");
    assert_eq!(stage.steps_taken, 2);
    assert_eq!(stage.termination_code, "age_target");
    assert!((end_model.star_age - 1500.0).abs() <= 1.0e-9);
    assert!((sample.age_yr - 1500.0).abs() <= 1.0e-9);
    assert!((end_model.dt / SECYER - 500.0).abs() <= 1.0e-9);
    assert!(sample.chi2.is_finite() && sample.chi2 < 1.0e90);
}

#[test]
fn test_extras_after_evolve_evaluates_final_model_on_max_age_stop() {
    // When `do_check_limits` hits `max_age` before `extras_check_model`,
    // `extras_after_evolve` (`astero/private/extras_support.f90:1109-1115`)
    // runs `do_astero_extras_check_model` so `best_sample` is still scored.
    let dir = TempDirGuard::new("after_evolve_max_age");
    let star_job = "  set_initial_age = .true.\n  initial_age = 0.0d0\n  set_initial_dt = .true.\n  years_for_initial_dt = 1.0d3";
    let base =
        write_zams_star_case(&dir, 2.0e3, star_job, "  max_age = 1500.0d0");
    dir.write(
        "inlist_astero_search_controls",
        "&astero_search_controls
  search_type = 'use_first_values'
  chi2_seismo_fraction = 0.0d0
  eval_chi2_at_target_age_only = .true.
  age_target = 1500.0d0
  num_smaller_steps_before_age_target = 50
  dt_for_smaller_steps_before_age_target = 500.0d0
  min_age_limit = 0.0d0
  lnuc_div_l_limit = 0.0d0
  constraint_name(1) = 'Teff'
  include_constraint_in_chi2_spectro(1) = .true.
  constraint_target(1) = 5777.0d0
  constraint_sigma(1) = 65.0d0
/
",
    );
    let bundle =
        InlistBundle::from_file_in_case_dir(&dir.path, "inlist").expect("in");
    let ctrl = parse_astero_search_controls(&dir.path).expect("astero");
    let mut hooks = MockSearchHooks;
    let (sample, stage, end_model) = run_single_astero_track_with_case_hooks(
        &bundle,
        &ctrl,
        ctrl.first_params(),
        1,
        Some(base.model_number + 6),
        &dir.path,
        StarLogsOptions::default(),
        Some(&mut hooks),
    )
    .expect("extras_after_evolve scores max_age stop");
    assert_eq!(stage.steps_taken, 2);
    assert_eq!(stage.termination_code, "max_age");
    assert!((end_model.star_age - 1500.0).abs() <= 1.0e-9);
    assert!((sample.age_yr - 1500.0).abs() <= 1.0e-9);
    assert!(sample.chi2.is_finite() && sample.chi2 < 1.0e90);
}

#[test]
fn test_checking_age_rejects_nonpositive_dt_for_smaller_steps() {
    // Matches `astero/private/extras_support.f90:84-89`: checking age with
    // non-positive `dt_for_smaller_steps_before_age_target` aborts loudly.
    let dir = TempDirGuard::new("checking_age_bad_dt_sm");
    let star_job = "  set_initial_age = .true.\n  initial_age = 0.0d0\n  set_initial_dt = .true.\n  years_for_initial_dt = 1.0d3";
    let base = write_zams_star_case(&dir, 2.0e3, star_job, "");
    dir.write(
        "inlist_astero_search_controls",
        "&astero_search_controls
  search_type = 'use_first_values'
  chi2_seismo_fraction = 0.0d0
  eval_chi2_at_target_age_only = .true.
  age_target = 1500.0d0
  num_smaller_steps_before_age_target = 50
  dt_for_smaller_steps_before_age_target = 0.0d0
  min_age_limit = 0.0d0
  lnuc_div_l_limit = 0.0d0
/
",
    );
    let bundle =
        InlistBundle::from_file_in_case_dir(&dir.path, "inlist").expect("in");
    let ctrl = parse_astero_search_controls(&dir.path).expect("astero");
    let mut hooks = MockSearchHooks;
    let err = run_single_astero_track_with_case_hooks(
        &bundle,
        &ctrl,
        ctrl.first_params(),
        1,
        Some(base.model_number + 3),
        &dir.path,
        StarLogsOptions::default(),
        Some(&mut hooks),
    )
    .expect_err("dt_for_smaller_steps_before_age_target <= 0 must abort");
    assert!(matches!(err, StarError::InvalidInput(_)));
}
