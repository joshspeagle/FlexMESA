// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone unit and integration tests for Train-8 Lane B fixes:
//! 1. `&star_job` cumulative energy error controls (`set_cumulative_energy_error`,
//!    `set_cumulative_energy_error_at_step`, and
//!    `set_cumulative_energy_error_each_step_if_age_less_than`, matching
//!    `star/defaults/star_job.defaults:2370-2375` and
//!    `star/job/run_star_support.f90:636-640, 689-692, 2762-2767`).
//! 2. Clean `ExtrasDirective::Terminate("")` (`s% termination_code == 0`)
//!    handling in `inlist_runner` (`star/job/run_star_support.f90:789-792, 828-843`).
//! 3. `AtmTTauOpacity::Varying` surface BC `log10(pgas) * LN10` EOS callback
//!    (`star/private/atm_support.f90:1221-1225`).

#![forbid(unsafe_code)]
#![allow(clippy::float_cmp, clippy::too_many_lines)]

use std::fs;
use std::path::PathBuf;

use flexmesa_star::controls::{AtmOption, AtmTTauOpacity};
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_inventory::STAR_JOB_SUPPORTED;
use flexmesa_star::inlist_runner::{
    run_inlist_stage, run_inlist_stage_with_logs,
    run_test_suite_case_with_extras, DefaultStarExtras, ExtrasDirective,
    StarExtras, StarLogsOptions,
};
use flexmesa_star::mesa_log_reader::read_mesa_log_file;
use flexmesa_star::model_io::write_mod_string;
use flexmesa_star::{
    eval_surf_bc, init_or_update_model_from_bundle, load_zams_model,
    set_hydro_vars, InlistLoadOptions, StarControls, StarError,
    StarJobControls, StarModel,
};

struct TempCaseGuard {
    root: PathBuf,
}

impl TempCaseGuard {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_star_t8_quick_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp case dir");
        Self { root }
    }
}

impl Drop for TempCaseGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// Verifies upstream MESA `26.4.1` defaults, inventory membership, strict
/// parsing, and loud type errors for the cumulative energy error controls
/// (`star/defaults/star_job.defaults:2370-2375`).
#[test]
fn test_cumulative_energy_error_controls_defaults_inventory_and_validation() {
    let def = StarJobControls::default();
    assert!(!def.set_cumulative_energy_error);
    assert_eq!(def.set_cumulative_energy_error_at_step, -1);
    assert_eq!(
        def.set_cumulative_energy_error_each_step_if_age_less_than,
        -1.0e99
    );
    assert!(!def.set_initial_cumulative_energy_error);
    assert_eq!(def.new_cumulative_energy_error, 0.0);
    assert!(def.set_cumulative_energy_error_each_relax);

    for key in [
        "set_cumulative_energy_error",
        "set_cumulative_energy_error_at_step",
        "set_cumulative_energy_error_each_relax",
        "set_cumulative_energy_error_each_step_if_age_less_than",
        "set_initial_cumulative_energy_error",
        "new_cumulative_energy_error",
    ] {
        assert!(
            STAR_JOB_SUPPORTED.contains(&key),
            "missing '{key}' from STAR_JOB_SUPPORTED"
        );
    }

    let tmp = TempCaseGuard::new("cum_err_parse");
    let valid = tmp.root.join("inlist_valid");
    fs::write(
        &valid,
        "&star_job\n\
           set_cumulative_energy_error = .true.\n\
           set_cumulative_energy_error_at_step = 1\n\
           set_cumulative_energy_error_each_step_if_age_less_than = 4d7\n\
           new_cumulative_energy_error = -3.25d36\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
         /\n",
    )
    .expect("write valid inlist");
    let bundle = InlistBundle::from_file(&valid)
        .expect("cumulative energy error controls must pass strict validation");
    assert_eq!(bundle.validation_report.unsupported_controls, Vec::new());
    assert!(bundle.job.set_cumulative_energy_error);
    assert_eq!(bundle.job.set_cumulative_energy_error_at_step, 1);
    assert_eq!(
        bundle
            .job
            .set_cumulative_energy_error_each_step_if_age_less_than,
        4.0e7
    );
    assert_eq!(bundle.job.new_cumulative_energy_error, -3.25e36);

    for (tag, body) in [
        (
            "set_bool_as_int",
            "&star_job\n  set_cumulative_energy_error = 1\n/\n",
        ),
        (
            "at_step_as_str",
            "&star_job\n  set_cumulative_energy_error_at_step = 'one'\n/\n",
        ),
        (
            "age_lt_as_bool",
            "&star_job\n  set_cumulative_energy_error_each_step_if_age_less_than = .true.\n/\n",
        ),
    ] {
        let bad = tmp.root.join(format!("inlist_bad_{tag}"));
        fs::write(&bad, body).expect("write malformed inlist");
        let err = InlistBundle::from_file(&bad)
            .expect_err("malformed cumulative energy error control must fail");
        assert!(
            matches!(err, StarError::InvalidInput(_)),
            "expected InvalidInput for {tag}, got {err:?}"
        );
    }
}

/// Verifies `set_cumulative_energy_error` and startup
/// `set_cumulative_energy_error_at_step == model_number` reset
/// `cumulative_energy_error` during initialization (`run_star_support.f90:2762-2767`).
#[test]
fn test_set_cumulative_energy_error_and_at_step_zero_on_startup() {
    let tmp = TempCaseGuard::new("cum_err_init");
    let mut seed =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("zams seed");
    seed.model_number = 0;
    seed.aux.cumulative_energy_error = 8.5e41;
    let mod_path = tmp.root.join("seed.mod");
    fs::write(&mod_path, write_mod_string(&seed).expect("write seed.mod"))
        .expect("save seed.mod");

    let inlist_flag = tmp.root.join("inlist_set_flag");
    fs::write(
        &inlist_flag,
        "&star_job\n\
           load_saved_model = .true.\n\
           load_model_filename = 'seed.mod'\n\
           set_cumulative_energy_error = .true.\n\
           new_cumulative_energy_error = -1.5d38\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n",
    )
    .expect("write inlist_set_flag");
    let b_flag = InlistBundle::from_file(&inlist_flag).expect("parse flag");
    let mut extras = DefaultStarExtras;
    let m_flag = init_or_update_model_from_bundle(&b_flag, None, &mut extras)
        .expect("init with set_cumulative_energy_error");
    assert_eq!(m_flag.aux.cumulative_energy_error, -1.5e38);
    assert_eq!(m_flag.aux.step_hist.cumulative_energy_error_old, -1.5e38);

    let inlist_step0 = tmp.root.join("inlist_step0");
    fs::write(
        &inlist_step0,
        "&star_job\n\
           load_saved_model = .true.\n\
           load_model_filename = 'seed.mod'\n\
           set_initial_model_number = .true.\n\
           initial_model_number = 0\n\
           set_cumulative_energy_error_at_step = 0\n\
           new_cumulative_energy_error = 4.25d36\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n",
    )
    .expect("write inlist_step0");
    let b_step0 = InlistBundle::from_file(&inlist_step0).expect("parse step0");
    let m_step0 = init_or_update_model_from_bundle(&b_step0, None, &mut extras)
        .expect("init with set_cumulative_energy_error_at_step=0");
    assert_eq!(m_step0.aux.cumulative_energy_error, 4.25e36);
    assert_eq!(m_step0.aux.step_hist.cumulative_energy_error_old, 4.25e36);
}

#[derive(Default)]
struct EnergyObserverExtras {
    start_step_errors: Vec<(usize, f64, f64)>,
    finish_step_errors: Vec<(usize, f64)>,
}

impl StarExtras for EnergyObserverExtras {
    fn extras_start_step(&mut self, m: &mut StarModel) -> ExtrasDirective {
        self.start_step_errors.push((
            m.model_number,
            m.star_age,
            m.aux.cumulative_energy_error,
        ));
        ExtrasDirective::KeepGoing
    }

    fn extras_finish_step(&mut self, m: &mut StarModel) -> ExtrasDirective {
        self.finish_step_errors
            .push((m.model_number, m.aux.cumulative_energy_error));
        ExtrasDirective::KeepGoing
    }
}

/// Verifies `set_cumulative_energy_error_at_step` resets
/// `cumulative_energy_error` in `after_step_loop` before `extras_finish_step`
/// and `history.data` output, then carries forward on subsequent steps
/// (`star/job/run_star_support.f90:689-692`).
#[test]
fn test_set_cumulative_energy_error_at_step_resets_after_target_step() {
    let tmp = TempCaseGuard::new("at_step_evolve");
    let mut seed =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("zams seed");
    seed.model_number = 0;
    seed.star_age = 0.0;
    let mod_path = tmp.root.join("seed.mod");
    fs::write(&mod_path, write_mod_string(&seed).expect("write seed.mod"))
        .expect("save seed.mod");

    fs::write(
        tmp.root.join("history_columns.list"),
        "model_number\nstar_age\ncumulative_energy_error\n",
    )
    .expect("write history_columns.list");

    let inlist = tmp.root.join("inlist_at_step");
    fs::write(
        &inlist,
        "&star_job\n\
           load_saved_model = .true.\n\
           load_model_filename = 'seed.mod'\n\
           set_initial_model_number = .true.\n\
           initial_model_number = 0\n\
           set_initial_cumulative_energy_error = .true.\n\
           set_cumulative_energy_error_at_step = 2\n\
           new_cumulative_energy_error = -2.5d38\n\
           set_initial_dt = .true.\n\
           years_for_initial_dt = 1.0d4\n\
           history_columns_file = 'history_columns.list'\n\
           save_photo_when_terminate = .false.\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           max_model_number = 3\n\
           photo_interval = -1\n\
           history_interval = 1\n\
         /\n",
    )
    .expect("write inlist");

    let bundle = InlistBundle::from_file(&inlist).expect("parse inlist");
    let mut extras = EnergyObserverExtras::default();
    let mut model =
        init_or_update_model_from_bundle(&bundle, None, &mut extras)
            .expect("init model");
    let hist_out = tmp.root.join("history.data");
    let log_opts = StarLogsOptions::from_target(&hist_out, false, false, false);
    let mut writer = log_opts
        .build_writer_with_extras(&model, &tmp.root, None, &extras)
        .expect("build writer")
        .expect("writer present");

    let rep = run_inlist_stage_with_logs(
        &mut model,
        &bundle,
        &mut extras,
        None,
        Some(&mut writer),
    )
    .expect("run 3-step stage");
    assert_eq!(rep.steps_taken, 3);
    assert_eq!(rep.termination_code, "max_model_number");
    assert_eq!(extras.finish_step_errors.len(), 3);
    // Step 1: reset at step 2 has not fired yet, while startup reset set it to
    // -2.5e38 plus step 1 energy error.
    assert_eq!(extras.finish_step_errors[0].0, 1);
    assert_ne!(extras.finish_step_errors[0].1, -2.5e38);
    // Step 2: `after_step_loop` resets cumulative_energy_error before
    // `extras_finish_step` and before `history.data` emission.
    assert_eq!(extras.finish_step_errors[1], (2, -2.5e38));
    // Step 3: accumulates from `-2.5e38`, not reset again.
    assert_eq!(extras.finish_step_errors[2].0, 3);
    assert_ne!(extras.finish_step_errors[2].1, -2.5e38);
    assert!(
        (extras.finish_step_errors[2].1 - (-2.5e38)).abs() < 1.0e45,
        "step 3 should accumulate near -2.5e38, got {}",
        extras.finish_step_errors[2].1
    );

    let tbl = read_mesa_log_file(&hist_out).expect("read history.data");
    let cum = tbl
        .column("cumulative_energy_error")
        .expect("cumulative_energy_error column");
    assert_eq!(cum.len(), 3);
    assert_ne!(cum[0], -2.5e38);
    assert_eq!(cum[1], -2.5e38);
    assert_ne!(cum[2], -2.5e38);
}

/// Verifies `set_cumulative_energy_error_each_step_if_age_less_than` resets
/// `cumulative_energy_error` in `before_step_loop` before `extras_start_step`
/// only while `star_age` is strictly below the threshold
/// (`star/job/run_star_support.f90:636-640`).
#[test]
fn test_set_cumulative_energy_error_each_step_if_age_less_than() {
    let tmp = TempCaseGuard::new("age_lt_evolve");
    let mut seed =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("zams seed");
    seed.model_number = 0;
    seed.star_age = 0.0;
    seed.aux.cumulative_energy_error = 9.0e40;
    let mod_path = tmp.root.join("seed.mod");
    fs::write(&mod_path, write_mod_string(&seed).expect("write seed.mod"))
        .expect("save seed.mod");

    let inlist = tmp.root.join("inlist_age_lt");
    fs::write(
        &inlist,
        "&star_job\n\
           load_saved_model = .true.\n\
           load_model_filename = 'seed.mod'\n\
           set_initial_age = .true.\n\
           initial_age = 0.0d0\n\
           set_initial_model_number = .true.\n\
           initial_model_number = 0\n\
           set_cumulative_energy_error_each_step_if_age_less_than = 50.0d0\n\
           new_cumulative_energy_error = 1.25d37\n\
           set_initial_dt = .true.\n\
           years_for_initial_dt = 100.0d0\n\
           save_photo_when_terminate = .false.\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           max_model_number = 2\n\
           photo_interval = -1\n\
         /\n",
    )
    .expect("write inlist");

    let bundle = InlistBundle::from_file(&inlist).expect("parse inlist");
    let mut extras = EnergyObserverExtras::default();
    let mut model =
        init_or_update_model_from_bundle(&bundle, None, &mut extras)
            .expect("init model");
    // Startup leaves seed's 9.0e40 intact (neither set_initial nor set_cum is true).
    assert_eq!(model.aux.cumulative_energy_error, 9.0e40);

    let rep = run_inlist_stage(&mut model, &bundle, &mut extras, None)
        .expect("run 2-step stage");
    assert_eq!(rep.steps_taken, 2);
    assert_eq!(extras.start_step_errors.len(), 2);
    // Step 1: star_age = 0.0 < 50.0 -> reset to 1.25e37 before extras_start_step.
    assert_eq!(extras.start_step_errors[0].1, 0.0);
    assert_eq!(extras.start_step_errors[0].2, 1.25e37);
    // Step 2: star_age = 100.0 >= 50.0 -> NOT reset before extras_start_step.
    assert_eq!(extras.start_step_errors[1].1, 100.0);
    assert_eq!(
        extras.start_step_errors[1].2,
        extras.finish_step_errors[0].1
    );
}

struct ConfigurableTerminateExtras {
    hook: HookKind,
    reason: String,
}

#[derive(Clone, Copy)]
enum HookKind {
    StartStep,
    CheckModel,
    FinishStep,
}

impl StarExtras for ConfigurableTerminateExtras {
    fn extras_start_step(&mut self, _m: &mut StarModel) -> ExtrasDirective {
        match self.hook {
            HookKind::StartStep => {
                ExtrasDirective::Terminate(self.reason.clone())
            }
            HookKind::CheckModel | HookKind::FinishStep => {
                ExtrasDirective::KeepGoing
            }
        }
    }

    fn extras_check_model(&mut self, _m: &mut StarModel) -> ExtrasDirective {
        match self.hook {
            HookKind::CheckModel => {
                ExtrasDirective::Terminate(self.reason.clone())
            }
            HookKind::StartStep | HookKind::FinishStep => {
                ExtrasDirective::KeepGoing
            }
        }
    }

    fn extras_finish_step(&mut self, _m: &mut StarModel) -> ExtrasDirective {
        match self.hook {
            HookKind::FinishStep => {
                ExtrasDirective::Terminate(self.reason.clone())
            }
            HookKind::StartStep | HookKind::CheckModel => {
                ExtrasDirective::KeepGoing
            }
        }
    }
}

/// Verifies clean `ExtrasDirective::Terminate("")` (`s% termination_code == 0`
/// in MESA, as in `star/test_suite/5M_cepheid_blue_loop/src/run_star_extras.f90:127`) passes
/// `required_termination_code_string`, writes termination `.mod`/`.photo`
/// saves, and marks `CaseRunReport::passed == true`, while explicit
/// `"extras_check_model"` continues enforcing `required_termination_code_string`.
#[test]
fn test_extras_empty_termination_code_passes_and_saves_models() {
    for (tag, hook, expected_steps) in [
        ("start", HookKind::StartStep, 0),
        ("check", HookKind::CheckModel, 1),
        ("finish", HookKind::FinishStep, 1),
    ] {
        let tmp = TempCaseGuard::new(&format!("empty_term_{tag}"));
        fs::write(
            tmp.root.join("rn"),
            "#!/usr/bin/env bash\ndo_one inlist final.mod\n",
        )
        .expect("write rn");
        let inlist_path = tmp.root.join("inlist");
        fs::write(
            &inlist_path,
            "&star_job\n\
               save_model_when_terminate = .true.\n\
               save_model_filename = 'final.mod'\n\
               save_photo_when_terminate = .true.\n\
               save_photo_filename = 'final.photo'\n\
               required_termination_code_string(1) = 'star_species_mass_min_limit'\n\
             /\n\
             &kap\n\
               Zbase = 0.02d0\n\
             /\n\
             &controls\n\
               initial_mass = 1.0d0\n\
               max_model_number = 5\n\
               photo_interval = -1\n\
             /\n",
        )
        .expect("write inlist");

        let mut extras = ConfigurableTerminateExtras {
            hook,
            reason: String::new(),
        };
        let rep = run_test_suite_case_with_extras(
            &tmp.root,
            true,
            None,
            false,
            StarLogsOptions::default(),
            InlistLoadOptions::default(),
            &mut extras,
        )
        .expect("empty extras terminate must run cleanly");
        assert!(
            rep.passed,
            "empty extras termination (s% termination_code == 0) must pass for {tag}: {rep:?}"
        );
        assert_eq!(rep.total_steps, expected_steps);
        assert_eq!(rep.termination_code, "");

        // Also check direct stage writes termination outputs rather than withholding.
        let bundle = InlistBundle::from_file(&inlist_path).expect("parse");
        let mut stage_extras = ConfigurableTerminateExtras {
            hook,
            reason: String::new(),
        };
        let mut model =
            init_or_update_model_from_bundle(&bundle, None, &mut stage_extras)
                .expect("init");
        let srep =
            run_inlist_stage(&mut model, &bundle, &mut stage_extras, None)
                .expect("stage run");
        assert_eq!(srep.termination_code, "");
        assert!(tmp.root.join("final.mod").is_file());
        assert!(tmp.root.join("final.photo").is_file());
    }

    // Explicit `t_extras_check_model` still checks `required_termination_code_string`.
    let tmp_named = TempCaseGuard::new("named_extras_term");
    fs::write(
        tmp_named.root.join("rn"),
        "#!/usr/bin/env bash\ndo_one inlist final.mod\n",
    )
    .expect("write rn");
    fs::write(
        tmp_named.root.join("inlist"),
        "&star_job\n\
           required_termination_code_string(1) = 'extras_check_model'\n\
           save_photo_when_terminate = .false.\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           max_model_number = 5\n\
           photo_interval = -1\n\
         /\n",
    )
    .expect("write matching inlist");
    let mut match_extras = ConfigurableTerminateExtras {
        hook: HookKind::CheckModel,
        reason: String::from("extras_check_model"),
    };
    let match_rep = run_test_suite_case_with_extras(
        &tmp_named.root,
        true,
        None,
        false,
        StarLogsOptions::default(),
        InlistLoadOptions::default(),
        &mut match_extras,
    )
    .expect("matching extras_check_model run");
    assert!(match_rep.passed);
    assert_eq!(match_rep.termination_code, "extras_check_model");

    fs::write(
        tmp_named.root.join("inlist"),
        "&star_job\n\
           required_termination_code_string(1) = 'max_age'\n\
           save_photo_when_terminate = .false.\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           max_model_number = 5\n\
           photo_interval = -1\n\
         /\n",
    )
    .expect("write mismatching inlist");
    let mut mis_extras = ConfigurableTerminateExtras {
        hook: HookKind::CheckModel,
        reason: String::from("extras_check_model"),
    };
    let mis_rep = run_test_suite_case_with_extras(
        &tmp_named.root,
        true,
        None,
        false,
        StarLogsOptions::default(),
        InlistLoadOptions::default(),
        &mut mis_extras,
    )
    .expect("mismatching extras_check_model run");
    assert!(!mis_rep.passed);
    assert_eq!(mis_rep.termination_code, "extras_check_model");
}

/// Regression check confirming `AtmTTauOpacity::Varying` surface BC
/// (`hydro_brunt_surf.rs:557` `log10(pgas) * LN10`) evaluates cleanly.
#[test]
fn test_varying_t_tau_surface_bc_evaluates_with_log10_pgas_eos_proc() {
    let mut ctrl = StarControls {
        initial_mass: 1.0,
        zbase: 0.02,
        atm_option: AtmOption::TTau,
        atm_t_tau_opacity: AtmTTauOpacity::Varying,
        ..StarControls::default()
    };
    ctrl.atm_t_tau_max_steps = 200;
    let mut model = load_zams_model(ctrl, StarJobControls::default())
        .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    eval_surf_bc(&mut model).expect("varying T-tau surface BC");
    assert!(model.surf_pt.ln_p.is_finite());
    assert!(model.surf_pt.ln_t.is_finite());
    assert!(model.teff.is_finite() && model.teff > 0.0);
}
