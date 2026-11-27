// SPDX-License-Identifier: GPL-3.0-or-later
//! Stage boundaries of a multi-stage `./rn` case (`DECISIONS #21`, `#154`).
//!
//! Each `do_one` is a separate MESA process, so a later stage that sets
//! `load_saved_model` starts from the text of the `.mod` file the earlier
//! stage saved (`star/job/run_star_support.f90:do_load1_star`,
//! `star/private/read_model.f90:finish_load_model`) and carries nothing from
//! the earlier stage's memory.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use flexmesa_star::inlist::{InlistBundle, InlistLoadOptions};
use flexmesa_star::inlist_runner::{
    run_inlist_stage, run_test_suite_case_with_extras, DefaultStarExtras,
    StarExtras, StarLogsOptions,
};
use flexmesa_star::model_io::read_mod_str;
use flexmesa_star::star_job_init::init_or_update_model_from_bundle;
use flexmesa_star::{StarError, StarModel};

struct TempCaseGuard {
    root: PathBuf,
}

impl TempCaseGuard {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "flexmesa_star_stage_handoff_{tag}_{}",
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

/// Records, for every stage, the state the star has once the model is loaded
/// and before its first step.
#[derive(Default)]
struct StartupRecorder {
    /// `(generations, dt, dt_old)` per stage.
    seen: Vec<(u8, f64, f64)>,
}

impl StarExtras for StartupRecorder {
    fn extras_startup(
        &mut self,
        m: &mut StarModel,
        _restart: bool,
    ) -> Result<(), StarError> {
        self.seen.push((m.generations, m.dt, m.dt_old));
        Ok(())
    }
}

fn stage_inlist(load: Option<&str>, save: &str, max_model: u32) -> String {
    let load_lines = load.map_or_else(String::new, |f| {
        format!("  load_saved_model = .true.\n  load_model_filename = '{f}'\n")
    });
    format!(
        "&star_job\n{load_lines}  save_model_when_terminate = .true.\n  \
         save_model_filename = '{save}'\n/\n\
         &kap\n  Zbase = 0.02d0\n/\n\
         &controls\n  initial_mass = 1.0d0\n  max_model_number = {max_model}\n  \
         photo_interval = -1\n/\n"
    )
}

fn run_case(
    root: &std::path::Path,
    extras: &mut dyn StarExtras,
) -> Result<flexmesa_star::inlist_runner::CaseRunReport, StarError> {
    run_test_suite_case_with_extras(
        root,
        true,
        None,
        false,
        StarLogsOptions::default(),
        InlistLoadOptions::default(),
        extras,
    )
}

/// A later stage that loads a saved model starts as a freshly loaded model:
/// `generations = 1` and `dt_old = dt` (`read_model.f90`), not the two
/// generations and previous-step `dt_old` of the finished stage.
#[test]
fn test_stage_two_starts_from_saved_mod_not_stage_one_memory() {
    let tmp = TempCaseGuard::new("reload");
    fs::write(
        tmp.root.join("rn"),
        "#!/usr/bin/env bash\n\
         do_one inlist_s1 stage1.mod\n\
         do_one inlist_s2 final.mod\n",
    )
    .expect("write rn");
    fs::write(
        tmp.root.join("inlist_s1"),
        stage_inlist(None, "stage1.mod", 3),
    )
    .expect("write inlist_s1");
    fs::write(
        tmp.root.join("inlist_s2"),
        stage_inlist(Some("stage1.mod"), "final.mod", 4),
    )
    .expect("write inlist_s2");

    let mut rec = StartupRecorder::default();
    let rep = run_case(&tmp.root, &mut rec).expect("two-stage case runs");
    assert_eq!(rep.stages.len(), 2);
    assert_eq!(rec.seen.len(), 2, "one startup per stage");
    let (gens, dt, dt_old) = rec.seen[1];
    assert_eq!(gens, 1, "a loaded model starts with one generation");
    assert_eq!(
        dt_old.to_bits(),
        dt.to_bits(),
        "a loaded model has dt_old = dt"
    );
}

/// A later stage that loads a file the earlier stage did not write is an
/// explicit error: no fallback to the earlier stage's in-memory model.
#[test]
fn test_stage_two_missing_saved_mod_is_an_error() {
    let tmp = TempCaseGuard::new("missing");
    fs::write(
        tmp.root.join("rn"),
        "#!/usr/bin/env bash\n\
         do_one inlist_s1\n\
         do_one inlist_s2 final.mod\n",
    )
    .expect("write rn");
    fs::write(
        tmp.root.join("inlist_s1"),
        // Saves under a name the next stage does not load.
        stage_inlist(None, "other.mod", 2),
    )
    .expect("write inlist_s1");
    fs::write(
        tmp.root.join("inlist_s2"),
        stage_inlist(Some("never_written.mod"), "final.mod", 3),
    )
    .expect("write inlist_s2");

    let err = run_case(&tmp.root, &mut DefaultStarExtras)
        .expect_err("a missing saved model must not fall back to memory");
    assert!(
        matches!(err, StarError::IoError(ref s) if s.contains("never_written.mod")),
        "unexpected error: {err:?}"
    );
}

/// A `do_one` stage that names an `output_mod` unconditionally removes any
/// stale copy beforehand and fails loudly if the stage does not write it
/// (`star/test_suite/test_suite_helpers:22, 32-43`).
#[test]
fn test_do_one_fails_loudly_when_stage_omits_output_mod() {
    let tmp = TempCaseGuard::new("omit_output_mod");
    let expected_mod = tmp.root.join("expected.mod");
    fs::write(&expected_mod, "stale output model\n").expect("write stale mod");
    fs::write(
        tmp.root.join("rn"),
        "#!/usr/bin/env bash\ndo_one inlist_s1 expected.mod\n",
    )
    .expect("write rn");
    fs::write(
        tmp.root.join("inlist_s1"),
        "&star_job\n\
           save_model_when_terminate = .false.\n\
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
    .expect("write inlist_s1");

    let err = run_case(&tmp.root, &mut DefaultStarExtras)
        .expect_err("missing output_mod must fail loudly");
    assert!(
        matches!(
            err,
            StarError::IoError(ref s)
                if s.contains("expected.mod") && s.contains("inlist_s1")
        ),
        "unexpected error: {err:?}"
    );
    assert!(
        !expected_mod.exists(),
        "stale expected.mod must be removed before running stage"
    );
}

/// Termination saves overwrite an existing `save_model_filename` cleanly
/// (`do_saves`, `star/job/run_star_support.f90:1270-1274`).
#[test]
fn test_save_model_when_terminate_overwrites_existing_mod() {
    let tmp = TempCaseGuard::new("overwrite_existing");
    let existing_mod = tmp.root.join("existing.mod");
    let sentinel = "SENTINEL_STALE_MOD_TEXT\n";
    fs::write(&existing_mod, sentinel).expect("write sentinel mod");

    let inlist_path = tmp.root.join("inlist");
    fs::write(&inlist_path, stage_inlist(None, "existing.mod", 2))
        .expect("write inlist");
    let bundle = InlistBundle::from_file(&inlist_path).expect("parse inlist");
    let mut extras = DefaultStarExtras;
    let mut model =
        init_or_update_model_from_bundle(&bundle, None, &mut extras)
            .expect("init model");
    let rep = run_inlist_stage(&mut model, &bundle, &mut extras, None)
        .expect("run stage");
    assert_eq!(rep.termination_code, "max_model_number");

    let written = fs::read_to_string(&existing_mod).expect("read existing.mod");
    assert_ne!(written, sentinel, "existing.mod must be overwritten");
    let reloaded =
        read_mod_str(&written, bundle.controls.clone(), bundle.job.clone())
            .expect("overwritten mod must parse cleanly");
    assert_eq!(reloaded.model_number, model.model_number);
}

/// A step-capped smoke run stops after the capped stage without synthesizing
/// a fake handoff `.mod` or running later stages against missing inputs.
#[test]
fn test_capped_smoke_stops_after_capped_stage_without_synthesizing_mod() {
    let tmp = TempCaseGuard::new("capped_smoke");
    fs::write(
        tmp.root.join("rn"),
        "#!/usr/bin/env bash\n\
         do_one inlist_s1 stage1.mod\n\
         do_one inlist_s2 final.mod\n",
    )
    .expect("write rn");
    fs::write(
        tmp.root.join("inlist_s1"),
        "&star_job\n\
           save_model_when_terminate = .true.\n\
           save_model_filename = 'stage1.mod'\n\
           required_termination_code_string(1) = 'max_model_number'\n\
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
    .expect("write inlist_s1");
    fs::write(
        tmp.root.join("inlist_s2"),
        stage_inlist(Some("stage1.mod"), "final.mod", 6),
    )
    .expect("write inlist_s2");

    let rep = run_test_suite_case_with_extras(
        &tmp.root,
        true,
        Some(1),
        false,
        StarLogsOptions::default(),
        InlistLoadOptions::default(),
        &mut DefaultStarExtras,
    )
    .expect("capped smoke run must stop cleanly without error");
    assert_eq!(rep.stages.len(), 1);
    assert_eq!(rep.termination_code, "max_steps_cap");
    assert!(!rep.passed);
    assert!(
        !tmp.root.join("stage1.mod").exists(),
        "capped stage must not synthesize stage1.mod"
    );
}
