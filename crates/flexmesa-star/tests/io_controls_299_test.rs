// SPDX-License-Identifier: LGPL-3.0-or-later
//! Issue `#299` tests for photo output scheduling,
//! `required_termination_code_string`, `x_logical_ctrl`, and terminal/timing
//! driver controls (`DECISIONS #26`).

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_star::inlist::{InlistBundle, InlistLoadOptions};
use flexmesa_star::inlist_inventory::copy_cleaned_star_case;
use flexmesa_star::inlist_runner::{
    DefaultStarExtras, ExtrasDirective, StarExtras, StarLogsOptions,
    photo_restart_filename, rel_run_e_err_exceeds_warn_limit, run_inlist_stage,
    run_test_suite_case_with_extras, termination_code_string_okay,
};
use flexmesa_star::model_io::{load_photo_bytes, write_photo_bytes};
use flexmesa_star::rn_parser::{RnStage, parse_rn_stages};
use flexmesa_star::{
    StarError, StarModel, init_or_update_model_from_bundle,
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
            "flexmesa_star_299_{tag}_{}_{id}",
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

fn mesa_root_dir() -> Option<PathBuf> {
    let base = std::env::var("MESA_DIR").map_or_else(
        |_| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home)
                .join(".cache")
                .join("flexmesa")
                .join("mesa_test_inputs")
        },
        PathBuf::from,
    );
    base.join("star").join("test_suite").is_dir().then_some(base)
}

/// Exact naming parity with `get_name_for_restart_file` and
/// `get_string_for_model_number('x', n, num_digits, num)`
/// (`star/private/star_utils.f90:329-351`).
#[test]
fn test_photo_restart_filename_matches_upstream_star_utils() {
    assert_eq!(photo_restart_filename(0, 3), "0");
    assert_eq!(photo_restart_filename(1, 3), "x001");
    assert_eq!(photo_restart_filename(50, 3), "x050");
    assert_eq!(photo_restart_filename(999, 3), "x999");
    assert_eq!(photo_restart_filename(1000, 3), "1000");
    assert_eq!(photo_restart_filename(1050, 3), "x050");
    assert_eq!(photo_restart_filename(2000, 3), "2000");

    assert_eq!(photo_restart_filename(42, 5), "x00042");
    assert_eq!(photo_restart_filename(100_000, 5), "100000");
    assert_eq!(photo_restart_filename(100_007, 5), "x00007");

    assert_eq!(photo_restart_filename(42, 8), "x00000042");
    assert_eq!(photo_restart_filename(7, 1), "x7");
    assert_eq!(photo_restart_filename(10, 1), "10");
}

/// Exact matching semantics for `termination_code_string_okay`
/// (`star/job/run_star_support.f90:828-843`).
#[test]
fn test_termination_code_string_okay_matches_upstream_run_star_support() {
    let empty: Vec<String> = vec![String::new(); 9];
    assert!(termination_code_string_okay(&empty, "max_model_number"));
    assert!(termination_code_string_okay(&empty, ""));

    let mut req = vec![String::new(); 9];
    req[0] = String::from("  max_age  ");
    assert!(
        termination_code_string_okay(&req, ""),
        "empty termination_code matches upstream s% termination_code == 0"
    );
    assert!(termination_code_string_okay(&req, "max_age"));
    assert!(!termination_code_string_okay(&req, "max_model_number"));

    req[3] = String::from("xa_central_lower_limit");
    assert!(termination_code_string_okay(
        &req,
        "xa_central_lower_limit"
    ));
    assert!(termination_code_string_okay(
        &req,
        "xa_central_lower_limit(h1)"
    ));
    assert!(!termination_code_string_okay(
        &req,
        "xa_central_lower_limit_extra"
    ));
    assert!(!termination_code_string_okay(&req, "min_timestep_limit"));
}

/// Periodic `photo_interval`, `photo_digits`, and `photo_directory` output
/// scheduling (`star/private/evolve.f90:2340-2361`).
#[test]
fn test_periodic_photo_interval_digits_and_directory() {
    let tmp = TempCaseGuard::new("photo_periodic");
    let inlist = tmp.root.join("inlist_periodic");
    fs::write(
        &inlist,
        "&star_job\n\
           save_photo_when_terminate = .false.\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           max_model_number = 4\n\
           photo_interval = 2\n\
           photo_digits = 5\n\
           photo_directory = 'photos_custom'\n\
         /\n",
    )
    .expect("write periodic inlist");
    let bundle = InlistBundle::from_file(&inlist).expect("parse periodic");
    assert_eq!(bundle.validation_report.unsupported_controls.len(), 0);
    assert_eq!(bundle.controls.output.photo_interval, 2);
    assert_eq!(bundle.controls.output.photo_digits, 5);
    assert_eq!(bundle.controls.output.photo_directory, "photos_custom");

    let mut extras = DefaultStarExtras;
    let mut model = init_or_update_model_from_bundle(&bundle, None, &mut extras)
        .expect("init periodic model");
    let rep = run_inlist_stage(&mut model, &bundle, &mut extras, None)
        .expect("run periodic stage");
    assert_eq!(rep.steps_taken, 4);

    let custom_dir = tmp.root.join("photos_custom");
    assert!(!custom_dir.join("x00001").exists());
    assert!(custom_dir.join("x00002").is_file());
    assert!(!custom_dir.join("x00003").exists());
    assert!(custom_dir.join("x00004").is_file());

    let p2_model = load_photo_bytes(
        &fs::read(custom_dir.join("x00002")).expect("read x00002"),
    )
    .expect("decode x00002");
    assert_eq!(p2_model.model_number, 2);
    let p4_model = load_photo_bytes(
        &fs::read(custom_dir.join("x00004")).expect("read x00004"),
    )
    .expect("decode x00004");
    assert_eq!(p4_model.model_number, 4);
    assert_eq!(p4_model.nz, model.nz);
}

/// Termination `save_photo_when_terminate` (default vs explicit
/// `save_photo_filename`) and mid-run `save_photo_number`
/// (`star/job/run_star_support.f90:263,789-790,1276-1281`,
/// `star/public/star_lib.f90:492-502`).
#[test]
fn test_termination_and_save_photo_number_outputs() {
    let mut extras = DefaultStarExtras;

    // 1. Default leaf inside photo_directory when save_photo_filename == ''.
    let tmp1 = TempCaseGuard::new("photo_term_default");
    let inlist1 = tmp1.root.join("inlist_term_default");
    fs::write(
        &inlist1,
        "&star_job\n\
           save_photo_when_terminate = .true.\n\
           save_photo_filename = ''\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           max_model_number = 2\n\
           photo_interval = -1\n\
           photo_digits = 3\n\
           photo_directory = 'term_photos'\n\
         /\n",
    )
    .expect("write term default inlist");
    let b1 = InlistBundle::from_file(&inlist1).expect("parse term default");
    let mut m1 = init_or_update_model_from_bundle(&b1, None, &mut extras)
        .expect("init term default model");
    let rep1 = run_inlist_stage(&mut m1, &b1, &mut extras, None)
        .expect("run term default stage");
    assert_eq!(rep1.steps_taken, 2);
    assert!(!tmp1.root.join("term_photos").join("x001").exists());
    let term_photo = tmp1.root.join("term_photos").join("x002");
    assert!(term_photo.is_file());
    let loaded1 =
        load_photo_bytes(&fs::read(&term_photo).expect("read term photo"))
            .expect("decode term photo");
    assert_eq!(loaded1.model_number, 2);

    // 2. Explicit save_photo_filename writes directly inside case_dir
    //    (star/public/star_lib.f90:492-502, make_sdb/inlist_make_sdb:27).
    let tmp2 = TempCaseGuard::new("photo_term_named");
    let inlist2 = tmp2.root.join("inlist_term_named");
    fs::write(
        &inlist2,
        "&star_job\n\
           save_photo_when_terminate = .true.\n\
           save_photo_filename = 'sdb.photo'\n\
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
    .expect("write named photo inlist");
    let b2 = InlistBundle::from_file(&inlist2).expect("parse named photo");
    let mut m2 = init_or_update_model_from_bundle(&b2, None, &mut extras)
        .expect("init named photo model");
    run_inlist_stage(&mut m2, &b2, &mut extras, None)
        .expect("run named photo stage");
    assert!(tmp2.root.join("sdb.photo").is_file());
    assert!(!tmp2.root.join("photos").exists());
    let loaded2 = load_photo_bytes(
        &fs::read(tmp2.root.join("sdb.photo")).expect("read sdb.photo"),
    )
    .expect("decode sdb.photo");
    assert_eq!(loaded2.model_number, 2);

    // 3. Mid-run save_photo_number writes photo at the requested step only.
    let tmp3 = TempCaseGuard::new("photo_save_num");
    let inlist3 = tmp3.root.join("inlist_save_num");
    fs::write(
        &inlist3,
        "&star_job\n\
           save_photo_number = 2\n\
           save_photo_when_terminate = .false.\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           max_model_number = 3\n\
           photo_interval = -1\n\
         /\n",
    )
    .expect("write save_photo_number inlist");
    let b3 = InlistBundle::from_file(&inlist3).expect("parse save_photo_num");
    let mut m3 = init_or_update_model_from_bundle(&b3, None, &mut extras)
        .expect("init save_photo_num model");
    run_inlist_stage(&mut m3, &b3, &mut extras, None)
        .expect("run save_photo_num stage");
    assert!(tmp3.root.join("photos").join("x002").is_file());
    assert!(!tmp3.root.join("photos").join("x003").exists());
}

struct StagedCaseCheckExtras {
    staged_dir: Option<PathBuf>,
    after_evolve_calls: usize,
}

impl StarExtras for StagedCaseCheckExtras {
    fn set_staged_case_dir(&mut self, case_dir: &Path) {
        self.staged_dir = Some(case_dir.to_path_buf());
    }

    fn extras_after_evolve(
        &mut self,
        _model: &mut StarModel,
    ) -> Result<(), StarError> {
        self.after_evolve_calls += 1;
        Ok(())
    }
}

/// Direct-stage `required_termination_code_string` mismatch withholds both
/// termination `.mod` and `.photo` saves, and out-of-bounds subscripts fail
/// loudly (`star/job/run_star_support.f90:789-792,828-843`,
/// `star_data/public/star_data_def.inc:369`).
#[test]
fn test_required_termination_code_string_direct_mismatch_and_bounds() {
    let tmp = TempCaseGuard::new("req_term_direct");
    let inlist_mis = tmp.root.join("inlist_mismatch");
    fs::write(
        &inlist_mis,
        "&star_job\n\
           save_model_when_terminate = .true.\n\
           save_model_filename = 'withheld.mod'\n\
           save_photo_when_terminate = .true.\n\
           save_photo_filename = 'withheld.photo'\n\
           required_termination_code_string(1) = 'xa_central_lower_limit'\n\
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
    .expect("write mismatch inlist");
    let b_mis = InlistBundle::from_file(&inlist_mis).expect("parse mismatch");
    assert_eq!(b_mis.validation_report.unsupported_controls.len(), 0);
    let mut extras = DefaultStarExtras;
    let mut m_mis = init_or_update_model_from_bundle(&b_mis, None, &mut extras)
        .expect("init mismatch model");
    let rep_mis = run_inlist_stage(&mut m_mis, &b_mis, &mut extras, None)
        .expect("run mismatch stage");
    assert_eq!(rep_mis.termination_code, "max_model_number");
    assert!(
        !tmp.root.join("withheld.mod").exists(),
        "mismatch must withhold save_model_when_terminate"
    );
    assert!(
        !tmp.root.join("withheld.photo").exists(),
        "mismatch must withhold save_photo_when_terminate"
    );

    for (tag, bad_sub) in [("sub0", 0), ("sub10", 10)] {
        let p = tmp.root.join(format!("inlist_req_{tag}"));
        fs::write(
            &p,
            format!(
                "&star_job\n  required_termination_code_string({bad_sub}) = 'max_age'\n/\n"
            ),
        )
        .expect("write bad subscript");
        let err = InlistBundle::from_file(&p).expect_err(
            "out-of-bounds required_termination_code_string must fail",
        );
        assert!(matches!(err, StarError::InvalidInput(_)));
    }
}

/// Multi-stage `run_test_suite_case_with_extras` passes when all stages match
/// `required_termination_code_string`
/// (`star/test_suite/test_suite_helpers:22,36-46`).
#[test]
fn test_required_termination_code_string_multistage_pass() {
    let tmp_pass = TempCaseGuard::new("req_term_pass_case");
    fs::write(
        tmp_pass.root.join("rn"),
        "#!/usr/bin/env bash\n\
         do_one inlist_s1 stage1.mod\n\
         do_one inlist_s2 final.mod\n",
    )
    .expect("write pass rn");
    fs::write(
        tmp_pass.root.join("inlist_s1"),
        "&star_job\n\
           save_model_when_terminate = .true.\n\
           save_model_filename = 'stage1.mod'\n\
           save_photo_when_terminate = .true.\n\
           save_photo_filename = 'stage1.photo'\n\
           required_termination_code_string(1) = 'max_model_number'\n\
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
    .expect("write pass inlist_s1");
    fs::write(
        tmp_pass.root.join("inlist_s2"),
        "&star_job\n\
           load_saved_model = .true.\n\
           load_model_filename = 'stage1.mod'\n\
           save_model_when_terminate = .true.\n\
           save_model_filename = 'final.mod'\n\
           required_termination_code_string(2) = 'max_model_number'\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           max_model_number = 3\n\
           photo_interval = -1\n\
         /\n",
    )
    .expect("write pass inlist_s2");
    let mut pass_extras = StagedCaseCheckExtras {
        staged_dir: None,
        after_evolve_calls: 0,
    };
    let pass_rep = run_test_suite_case_with_extras(
        &tmp_pass.root,
        true,
        None,
        false,
        StarLogsOptions::default(),
        InlistLoadOptions::default(),
        &mut pass_extras,
    )
    .expect("run passing multi-stage case");
    assert!(
        pass_rep.passed,
        "matching required code must pass: {pass_rep:?}"
    );
    assert_eq!(pass_rep.stages.len(), 2);
    assert_eq!(pass_rep.termination_code, "max_model_number");
    assert_eq!(pass_extras.after_evolve_calls, 2);
}

/// Multi-stage `run_test_suite_case_with_extras` removes stale stage output,
/// aborts subsequent stages, and fails when `required_termination_code_string`
/// mismatches (`star/test_suite/test_suite_helpers:22,36-46`).
#[test]
fn test_required_termination_code_string_multistage_mismatch() {
    let tmp_fail = TempCaseGuard::new("req_term_fail_case");
    fs::write(
        tmp_fail.root.join("rn"),
        "#!/usr/bin/env bash\n\
         do_one inlist_s1 stage1.mod\n\
         do_one inlist_s2 final.mod\n",
    )
    .expect("write fail rn");
    fs::write(tmp_fail.root.join("stage1.mod"), "stale mod file")
        .expect("write stale stage1.mod");
    fs::write(
        tmp_fail.root.join("inlist_s1"),
        "&star_job\n\
           save_model_when_terminate = .true.\n\
           save_model_filename = 'stage1.mod'\n\
           required_termination_code_string(1) = 'xa_central_lower_limit'\n\
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
    .expect("write fail inlist_s1");
    fs::write(
        tmp_fail.root.join("inlist_s2"),
        "&star_job\n\
           load_saved_model = .true.\n\
           load_model_filename = 'stage1.mod'\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           max_model_number = 3\n\
           photo_interval = -1\n\
         /\n",
    )
    .expect("write fail inlist_s2");
    let mut fail_extras = StagedCaseCheckExtras {
        staged_dir: None,
        after_evolve_calls: 0,
    };
    let fail_rep = run_test_suite_case_with_extras(
        &tmp_fail.root,
        true,
        None,
        false,
        StarLogsOptions::default(),
        InlistLoadOptions::default(),
        &mut fail_extras,
    )
    .expect("run failing multi-stage case");
    assert!(
        !fail_rep.passed,
        "required_termination_code_string mismatch must fail case: {fail_rep:?}"
    );
    assert_eq!(fail_rep.stages.len(), 1);
    assert_eq!(fail_rep.termination_code, "max_model_number");
    assert_eq!(fail_extras.after_evolve_calls, 1);
}

struct LogicalCtrlExtras {
    saw_slot0_true: bool,
    saw_slot36_true: bool,
    saw_slot1_false: bool,
}

impl StarExtras for LogicalCtrlExtras {
    fn extras_check_model(&mut self, model: &mut StarModel) -> ExtrasDirective {
        self.saw_slot0_true = model.controls.x_logical_ctrl[0];
        self.saw_slot36_true = model.controls.x_logical_ctrl[36];
        self.saw_slot1_false = !model.controls.x_logical_ctrl[1];
        if model.controls.x_logical_ctrl[0] {
            ExtrasDirective::Terminate(String::from("x_logical_ctrl_stop"))
        } else {
            ExtrasDirective::KeepGoing
        }
    }
}

/// Verifies `x_logical_ctrl` 1-based -> 0-based indexing (`DECISIONS #28`),
/// strict bounds/type checks, and live runtime propagation into [`StarExtras`]
/// (`star_data/private/star_controls.inc:1411`,
/// `star/defaults/controls.defaults:11915-11923`).
#[test]
fn test_x_logical_ctrl_parsing_validation_and_extras_runtime_access() {
    let tmp = TempCaseGuard::new("x_logical_ctrl");
    let valid = tmp.root.join("inlist_x_logical_ok");
    fs::write(
        &valid,
        "&kap\n\
           Zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           photo_interval = -1\n\
           x_logical_ctrl(1) = .true.\n\
           x_logical_ctrl(37) = .true.\n\
         /\n",
    )
    .expect("write valid x_logical_ctrl");
    let bundle = InlistBundle::from_file(&valid)
        .expect("valid x_logical_ctrl must pass strict validation");
    assert_eq!(bundle.validation_report.unsupported_controls.len(), 0);
    assert!(bundle.controls.x_logical_ctrl[0]);
    assert!(!bundle.controls.x_logical_ctrl[1]);
    assert!(bundle.controls.x_logical_ctrl[36]);

    let mut extras = LogicalCtrlExtras {
        saw_slot0_true: false,
        saw_slot36_true: false,
        saw_slot1_false: false,
    };
    let mut model = init_or_update_model_from_bundle(&bundle, None, &mut extras)
        .expect("init x_logical_ctrl model");
    let rep = run_inlist_stage(&mut model, &bundle, &mut extras, Some(5))
        .expect("run x_logical_ctrl stage");
    assert_eq!(rep.steps_taken, 1);
    assert_eq!(rep.termination_code, "x_logical_ctrl_stop");
    assert!(extras.saw_slot0_true);
    assert!(extras.saw_slot36_true);
    assert!(extras.saw_slot1_false);

    for (tag, body) in [
        ("zero", "&controls\n  x_logical_ctrl(0) = .true.\n/\n"),
        ("high", "&controls\n  x_logical_ctrl(101) = .true.\n/\n"),
        ("scalar", "&controls\n  x_logical_ctrl = .true.\n/\n"),
        ("bad_type", "&controls\n  x_logical_ctrl(1) = 42\n/\n"),
    ] {
        let p = tmp.root.join(format!("inlist_xlog_{tag}"));
        fs::write(&p, body).expect("write bad x_logical_ctrl");
        let err = InlistBundle::from_file(&p)
            .expect_err("malformed x_logical_ctrl must fail");
        assert!(
            matches!(err, StarError::InvalidInput(_)),
            "expected InvalidInput for {tag}, got {err}"
        );
    }
}

/// Verifies `#299` `DECISIONS #26` terminal/timing keys pass strict validation
/// and survive `copy_cleaned_star_case`.
#[test]
fn test_terminal_timing_controls_validation_and_copy_cleaned() {
    let tmp = TempCaseGuard::new("term_cleaned");
    let src_dir = tmp.root.join("src_case");
    let dst_dir = tmp.root.join("dst_case");
    fs::create_dir_all(&src_dir).expect("mkdir src_case");
    let inlist_path = src_dir.join("inlist");
    fs::write(
        &inlist_path,
        "&star_job\n\
           pause_before_terminate = .false.\n\
           first_model_for_timing = 10\n\
           warn_run_star_extras = .false.\n\
           save_photo_when_terminate = .true.\n\
           save_photo_filename = 'end.photo'\n\
           save_photo_number = 5\n\
           required_termination_code_string(1) = 'max_model_number'\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           photo_directory = 'photos_out'\n\
           photo_interval = 25\n\
           photo_digits = 4\n\
           x_logical_ctrl(2) = .true.\n\
           extra_terminal_output_file = 'log1'\n\
           report_solver_progress = .true.\n\
           warning_limit_for_max_residual = 1d99\n\
         /\n",
    )
    .expect("write full 299 inlist");

    let bundle = InlistBundle::from_file(&inlist_path)
        .expect("all #299 keys must pass strict validation");
    assert_eq!(bundle.validation_report.unsupported_controls.len(), 0);
    let recorded: Vec<(&str, &str)> = bundle
        .validation_report
        .terminal_output_keys
        .iter()
        .map(|(s, k)| (s.as_str(), k.as_str()))
        .collect();
    for expected in [
        ("star_job", "pause_before_terminate"),
        ("star_job", "first_model_for_timing"),
        ("star_job", "warn_run_star_extras"),
        ("controls", "extra_terminal_output_file"),
        ("controls", "report_solver_progress"),
        ("controls", "warning_limit_for_max_residual"),
    ] {
        assert!(
            recorded.contains(&expected),
            "missing terminal_output_keys entry {expected:?}: {recorded:?}"
        );
    }

    copy_cleaned_star_case(&src_dir, &dst_dir).expect("copy cleaned star case");
    let cleaned_bundle = InlistBundle::from_file(&dst_dir.join("inlist"))
        .expect("cleaned case must parse strictly");
    assert_eq!(cleaned_bundle.controls.output.photo_directory, "photos_out");
    assert_eq!(cleaned_bundle.controls.output.photo_interval, 25);
    assert_eq!(cleaned_bundle.controls.output.photo_digits, 4);
    assert_eq!(cleaned_bundle.job.save_photo_number, 5);
    assert!(cleaned_bundle.controls.x_logical_ctrl[1]);
    assert!(
        cleaned_bundle
            .validation_report
            .terminal_output_keys
            .contains(&(
                String::from("controls"),
                String::from("extra_terminal_output_file")
            ))
    );
}

/// Malformed value types and out-of-range `photo_digits` fail loudly.
#[test]
fn test_photo_and_terminal_controls_reject_malformed_types() {
    let tmp = TempCaseGuard::new("bad_types");
    for (tag, body) in [
        ("pause_int", "&star_job\n  pause_before_terminate = 123\n/\n"),
        ("timing_str", "&star_job\n  first_model_for_timing = 'abc'\n/\n"),
        ("warn_extras_int", "&star_job\n  warn_run_star_extras = 42\n/\n"),
        ("save_photo_num_str", "&star_job\n  save_photo_number = 'five'\n/\n"),
        (
            "extra_term_bool",
            "&controls\n  extra_terminal_output_file = .true.\n/\n",
        ),
        ("solver_prog_str", "&controls\n  report_solver_progress = 'yes'\n/\n"),
        (
            "warn_resid_bool",
            "&controls\n  warning_limit_for_max_residual = .false.\n/\n",
        ),
        ("photo_digits_zero", "&controls\n  photo_digits = 0\n/\n"),
        ("photo_digits_huge", "&controls\n  photo_digits = 19\n/\n"),
        ("photo_interval_str", "&controls\n  photo_interval = 'fifty'\n/\n"),
        ("photo_dir_int", "&controls\n  photo_directory = 123\n/\n"),
    ] {
        let p = tmp.root.join(format!("inlist_bad_{tag}"));
        fs::write(&p, body).expect("write malformed inlist");
        let err = InlistBundle::from_file(&p)
            .expect_err("malformed control value must fail loudly");
        assert!(
            matches!(err, StarError::InvalidInput(_)),
            "expected InvalidInput for {tag}, got {err}"
        );
    }
}

/// Strict zero-ignored-controls validation on real MESA `26.4.1` `star` and
/// `binary` sample inlists unblocked by `#299` (when `$MESA_DIR` or the
/// local test-inputs cache is present).
#[test]
fn test_zero_ignored_controls_on_affected_mesa_sample_cases() {
    let Some(mesa_root) = mesa_root_dir() else {
        return;
    };

    let star_suite = mesa_root.join("star").join("test_suite");
    for case_name in [
        "1.3M_ms_high_Z",
        "semiconvection",
        "timing",
        "extended_convective_penetration",
        "make_o_ne_wd",
    ] {
        let case_dir = star_suite.join(case_name);
        if !case_dir.is_dir() {
            continue;
        }
        let stages = parse_rn_stages(&case_dir, true).unwrap_or_else(|e| {
            panic!("parse_rn_stages failed for {case_name}: {e}")
        });
        for stage in stages {
            if let RnStage::DoOne { inlist, .. } = stage {
                let bundle =
                    InlistBundle::from_file_in_case_dir(&case_dir, &inlist)
                        .unwrap_or_else(|e| {
                            panic!(
                                "strict validation failed for star/{case_name}/{inlist}: {e}"
                            )
                        });
                assert_eq!(
                    bundle.validation_report.unsupported_controls.len(),
                    0,
                    "star/{case_name}/{inlist} has unsupported controls: {:?}",
                    bundle.validation_report.unsupported_controls
                );
            }
        }
    }

    let binary_suite = mesa_root.join("binary").join("test_suite");
    for (case_name, inlists) in [
        ("evolve_both_stars", &["inlist1", "inlist2"][..]),
        ("jdot_gr_check", &["inlist1"][..]),
        ("jdot_ls_check", &["inlist1"][..]),
        ("jdot_ml_check", &["inlist1"][..]),
        ("star_plus_point_mass", &["inlist1"][..]),
        ("star_plus_point_mass_explicit_mdot", &["inlist1"][..]),
        ("wind_fed_bhhmxb", &["inlist1"][..]),
    ] {
        let case_dir = binary_suite.join(case_name);
        if !case_dir.is_dir() {
            continue;
        }
        for inlist in inlists {
            let bundle = InlistBundle::from_file_in_case_dir(&case_dir, inlist)
                .unwrap_or_else(|e| {
                    panic!(
                        "strict validation failed for binary/{case_name}/{inlist}: {e}"
                    )
                });
            assert_eq!(
                bundle.validation_report.unsupported_controls.len(),
                0,
                "binary/{case_name}/{inlist} has unsupported controls: {:?}",
                bundle.validation_report.unsupported_controls
            );
            assert!(
                bundle.validation_report.terminal_output_keys.contains(&(
                    String::from("controls"),
                    String::from("extra_terminal_output_file")
                )),
                "binary/{case_name}/{inlist} must record extra_terminal_output_file"
            );
        }
    }
}

#[test]
fn test_warn_when_large_rel_run_e_err_parsing_roundtrip_and_trigger() {
    use flexmesa_star::controls::StarControls;

    let tmp = TempCaseGuard::new("warn_rel_e");
    assert!(
        (StarControls::default().output.warn_when_large_rel_run_e_err - 0.1)
            .abs()
            <= 1.0e-15
    );

    let inlist_path = tmp.root.join("inlist_warn");
    fs::write(
        &inlist_path,
        "&star_job\n/\n&kap\n  zbase = 0.02d0\n/\n&controls\n  initial_mass = 1.0d0\n  initial_z = 0.02d0\n  warn_when_large_rel_run_E_err = 0.025d0\n/\n",
    )
    .expect("write inlist_warn");
    let bundle =
        InlistBundle::from_file(&inlist_path).expect("strict parse must pass");
    assert_eq!(bundle.validation_report.unsupported_controls.len(), 0);

    let mut extras = DefaultStarExtras;
    let mut model = init_or_update_model_from_bundle(&bundle, None, &mut extras)
        .expect("init model with warn_when_large_rel_run_e_err");
    assert!(
        (model.controls.output.warn_when_large_rel_run_e_err - 0.025).abs()
            <= 1.0e-15
    );

    // Zero total_energy_end never triggers even with nonzero cumulative error.
    model.aux.step_hist.total_energy_end = 0.0;
    model.aux.cumulative_energy_error = -1.0e40;
    assert_eq!(rel_run_e_err_exceeds_warn_limit(&model), None);

    // Below threshold is suppressed; above threshold triggers with |err / tot|.
    model.aux.step_hist.total_energy_end = -1.0e48;
    model.aux.cumulative_energy_error = 2.0e46; // rel_err = 0.02 <= 0.025
    assert_eq!(rel_run_e_err_exceeds_warn_limit(&model), None);

    model.aux.cumulative_energy_error = -5.0e46; // rel_err = 0.05 > 0.025
    let triggered = rel_run_e_err_exceeds_warn_limit(&model)
        .expect("exceeding threshold must trigger warning");
    assert!((triggered - 0.05).abs() <= 1.0e-15);

    // Round-trip through .photo preserves threshold.
    let bytes = write_photo_bytes(&model);
    let restored = load_photo_bytes(&bytes).expect("load photo");
    assert!(
        (restored.controls.output.warn_when_large_rel_run_e_err - 0.025).abs()
            <= 1.0e-15
    );

    // Non-numeric input fails strictly.
    let bad_path = tmp.root.join("inlist_bad_warn");
    fs::write(
        &bad_path,
        "&controls\n  warn_when_large_rel_run_e_err = 'large'\n/\n",
    )
    .expect("write bad inlist");
    let err = InlistBundle::from_file(&bad_path)
        .expect_err("non-numeric warn_when_large_rel_run_e_err must fail");
    assert!(matches!(err, StarError::InvalidInput(_)));
}
