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

use flexmesa_star::inlist::InlistLoadOptions;
use flexmesa_star::inlist_runner::{
    run_test_suite_case_with_extras, DefaultStarExtras, StarExtras,
    StarLogsOptions,
};
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
