// SPDX-License-Identifier: GPL-3.0-or-later
//! Tests for the `&star_job` installation-location key `mesa_dir`
//! (`star/defaults/star_job.defaults:10-18`, `DECISIONS #68`, `#548`).
//!
//! Upstream `mesa_dir` overrides `$MESA_DIR` when non-blank to locate the
//! data tree (`const/public/const_def.f90:186-198`) and, for output only, the
//! default column lists and the `stop_warnings_for_rel_E_err` sentinel.
//! `FlexMESA` stores the value and resolves those resources from its
//! verified MESA `26.4.1` copies.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_inventory::STAR_JOB_SUPPORTED;
use flexmesa_star::{
    load_zams_model, resolve_star_history_columns, StarControls, StarError,
    StarJobControls,
};
use flexmesa_utils::require_mesa_test_input_path;

struct TempCaseGuard {
    root: PathBuf,
}

impl TempCaseGuard {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_star_mesa_dir_548_{tag}_{}_{id}",
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

/// `mesa_dir` defaults to the blank string, is a supported key, is stored
/// verbatim, and a non-string value is rejected loudly.
#[test]
fn test_mesa_dir_default_blank_parses_into_job_controls_and_rejects_bad_type() {
    assert_eq!(StarJobControls::default().mesa_dir, "");
    assert!(STAR_JOB_SUPPORTED.contains(&"mesa_dir"));

    let tmp = TempCaseGuard::new("parse");
    let valid = tmp.root.join("inlist_valid");
    fs::write(
        &valid,
        "&star_job\n  mesa_dir = '../../..'\n/\n\
         &controls\n  initial_mass = 1.0d0\n/\n",
    )
    .expect("write valid inlist");
    let bundle = InlistBundle::from_file(&valid)
        .expect("mesa_dir must pass strict validation");
    assert_eq!(bundle.validation_report.unsupported_controls, Vec::new());
    assert_eq!(bundle.job.mesa_dir, "../../..");

    let bad = tmp.root.join("inlist_bad");
    fs::write(&bad, "&star_job\n  mesa_dir = .true.\n/\n")
        .expect("write malformed inlist");
    let err = InlistBundle::from_file(&bad)
        .expect_err("a logical mesa_dir must fail");
    assert!(
        matches!(err, StarError::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// The upstream `star/starspots` inlist chain (the only `26.4.1` test case
/// that sets `mesa_dir`) validates with no unsupported controls.
#[test]
fn test_starspots_inlist_validates_with_no_unsupported_controls() {
    let Some(header) = require_mesa_test_input_path(
        "star/test_suite/starspots/inlist_starspots_header",
    ) else {
        return;
    };
    let bundle = InlistBundle::from_file(&header)
        .expect("starspots inlist must pass strict validation");
    assert_eq!(bundle.validation_report.unsupported_controls, Vec::new());
    assert_eq!(bundle.job.mesa_dir, "../../..");
}

/// A requested `history_columns_file` that does not exist is a loud error.
/// `mesa_dir` does not open a fallback to `star/defaults/history_columns.list`
/// (upstream `star/private/history_specs.f90:116`) in `FlexMESA`.
#[test]
fn test_missing_history_columns_file_errors_even_with_mesa_dir_set() {
    let tmp = TempCaseGuard::new("missing_columns");
    let ctrl = StarControls {
        initial_mass: 1.0,
        zbase: 0.02,
        initial_z: 0.02,
        ..StarControls::default()
    };
    let job = StarJobControls {
        mesa_dir: String::from("../../.."),
        history_columns_file: String::from("no_such_history_columns.list"),
        ..StarJobControls::default()
    };
    let model = load_zams_model(ctrl, job).expect("load zams");
    let err = resolve_star_history_columns(&model, Some(&tmp.root))
        .expect_err("missing history_columns_file must fail");
    assert!(
        matches!(err, StarError::IoError(_)),
        "expected IoError, got {err:?}"
    );
}
