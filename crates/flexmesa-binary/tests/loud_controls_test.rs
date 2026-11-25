// SPDX-License-Identifier: GPL-3.0-or-later
//! Issue `#147` fail-loudly validation tests for `flexmesa-binary`.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_binary::e2e::run_binary_test_suite_case;
use flexmesa_binary::inlist::{
    BINARY_CONTROLS_SUPPORTED, BINARY_JOB_HEADLESS_GUI, BINARY_JOB_SUPPORTED,
    BinaryInlistControls,
};
use flexmesa_binary::mdot_implicit::ImplicitMdotSolver;
use flexmesa_binary::roche_deformation::RocheDeformationTables;
use flexmesa_binary::system_rlof::eval_scheme_explicit_mdot;
use flexmesa_schema::upstream_section_keys;
use flexmesa_star::{
    InlistLoadOptions, StarControls, StarError, StarJobControls,
    load_zams_model,
};

struct TempBinGuard {
    root: PathBuf,
}

impl TempBinGuard {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_bin_147_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp binary dir");
        Self { root }
    }
}

impl Drop for TempBinGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn test_binary_unsupported_unknown_and_headless_pgbinary_validation() {
    let tmp = TempBinGuard::new("schema");
    let unimpl = tmp.root.join("unimpl");
    fs::create_dir_all(&unimpl).expect("mkdir");
    fs::write(
        unimpl.join("inlist"),
        "&binary_controls\n  m1 = 2.0d0\n  extra_binary_terminal_output_file = 'log'\n/\n",
    )
    .expect("write");
    let err = BinaryInlistControls::from_case_dir(&unimpl)
        .expect_err("unimplemented binary control must fail by default");
    assert!(matches!(err, StarError::InvalidInput(_)));
    assert!(
        err.to_string()
            .contains("extra_binary_terminal_output_file")
    );
    assert!(
        BinaryInlistControls::from_case_dir_with_options(
            &unimpl,
            InlistLoadOptions {
                allow_unsupported_controls: true,
            },
        )
        .is_ok()
    );

    let pg_ok = tmp.root.join("pg_ok");
    fs::create_dir_all(&pg_ok).expect("mkdir");
    fs::write(
        pg_ok.join("inlist"),
        "&binary_job\n  pgbinary_flag = .true.\n/\n\
         &pgbinary\n  pgbinary_interval = 1\n/\n\
         &binary_controls\n  m1 = 1.5d0\n/\n",
    )
    .expect("write");
    assert!(BinaryInlistControls::from_case_dir(&pg_ok).is_ok());

    let pg_bad = tmp.root.join("pg_bad");
    fs::create_dir_all(&pg_bad).expect("mkdir");
    fs::write(
        pg_bad.join("inlist"),
        "&binary_job\n  pgbinary_flag = .true.\n/\n\
         &pgbinary\n  no_such_pgbinary_ctrl = 1\n/\n",
    )
    .expect("write");
    assert!(BinaryInlistControls::from_case_dir(&pg_bad).is_err());

    let pg_inactive = tmp.root.join("pg_inactive");
    fs::create_dir_all(&pg_inactive).expect("mkdir");
    fs::write(
        pg_inactive.join("inlist"),
        "&binary_job\n  pgbinary_flag = .false.\n/\n\
         &pgbinary\n  no_such_pgbinary_ctrl = 1\n  pgbinary_interval = 1\n/\n",
    )
    .expect("write");
    assert!(BinaryInlistControls::from_case_dir(&pg_inactive).is_ok());

    for (sec, list) in [
        ("binary_job", BINARY_JOB_SUPPORTED),
        ("binary_job", BINARY_JOB_HEADLESS_GUI),
        ("binary_controls", BINARY_CONTROLS_SUPPORTED),
    ] {
        let upstream = upstream_section_keys(sec)
            .unwrap_or_else(|| panic!("missing schema keys for &{sec}"));
        for &key in list {
            assert!(
                upstream.contains(key),
                "binary inventory '&{sec}::{key}' missing from MESA 26.4.1 schema"
            );
        }
    }
}

#[test]
fn test_binary_missing_inlist2_and_missing_roche_tables_fail_loudly() {
    let tmp = TempBinGuard::new("missing");
    let case_dir = tmp.root.join("missing_inlist2");
    fs::create_dir_all(&case_dir).expect("mkdir");
    fs::write(
        case_dir.join("inlist"),
        "&binary_job\n  evolve_both_stars = .true.\n  inlist_names(1) = 'inlist1'\n  inlist_names(2) = 'missing_inlist2'\n/\n&binary_controls\n  m1 = 10.0d0\n  m2 = 8.0d0\n/\n",
    )
    .expect("write inlist");
    fs::write(
        case_dir.join("inlist1"),
        "&star_job\n/\n&kap\n  zbase = 0.02d0\n/\n&controls\n  initial_mass = 10.0d0\n  max_model_number = 1\n/\n",
    )
    .expect("write inlist1");

    let err = run_binary_test_suite_case(&case_dir, Some(1))
        .expect_err("missing inlist2 with evolve_both_stars=true must fail");
    assert!(
        err.to_string().contains("missing_inlist2")
            || matches!(
                err,
                StarError::IoError(_) | StarError::InvalidInput(_)
            ),
        "unexpected missing inlist2 error: {err}"
    );

    assert!(
        RocheDeformationTables::from_dir(Path::new("/nonexistent_roche_dir"))
            .is_err()
    );
}

#[test]
fn test_binary_option_strings_and_removed_clamps_fail_loudly() {
    let tmp = TempBinGuard::new("opts");
    for (idx, body) in [
        "&binary_controls\n  mdot_scheme = 'bogus_scheme'\n/\n",
        "&binary_controls\n  solver_type = 'bogus_solver'\n/\n",
        "&binary_controls\n  sync_type_1 = 'bogus_sync'\n/\n",
        "&binary_controls\n  sync_mode_1 = 'bogus_mode'\n/\n",
        "&binary_controls\n  circ_type_1 = 'bogus_circ'\n/\n",
        "&binary_controls\n  anomaly_steps = 1\n/\n",
        "&binary_controls\n  history_interval = -1\n/\n",
        "&binary_controls\n  dt_reduction_factor_for_j = 1.5d0\n/\n",
        "&binary_controls\n  time_delta_coeff = 0.0d0\n/\n",
    ]
    .iter()
    .enumerate()
    {
        let d = tmp.root.join(format!("bad_{idx}"));
        fs::create_dir_all(&d).expect("mkdir");
        fs::write(d.join("inlist"), body).expect("write");
        let err = BinaryInlistControls::from_case_dir(&d).expect_err(
            "invalid binary option or out-of-range control must fail",
        );
        assert!(
            matches!(err, StarError::InvalidInput(_)),
            "expected InvalidInput for {body}: got {err}"
        );
    }

    let bh_clamp_dir = tmp.root.join("bh_clamp");
    fs::create_dir_all(&bh_clamp_dir).expect("mkdir");
    fs::write(
        bh_clamp_dir.join("inlist"),
        "&binary_controls\n  initial_bh_spin = 2.0d0\n/\n",
    )
    .expect("write");
    let clamped = BinaryInlistControls::from_case_dir(&bh_clamp_dir)
        .expect("upstream MESA clamps finite initial_bh_spin to [0, 1]");
    assert!((clamped.initial_bh_spin - 1.0).abs() < 1.0e-15);

    let bad_solver = BinaryInlistControls {
        solver_type: String::from("bogus_solver"),
        ..BinaryInlistControls::default()
    };
    assert!(ImplicitMdotSolver::from_controls(&bad_solver, 1.2).is_err());

    let mut solver = ImplicitMdotSolver::from_controls(
        &BinaryInlistControls::default(),
        1.2,
    )
    .expect("valid default solver");
    // Upstream `binary/private/binary_mdot.f90:157,646,675-681` supports
    // `"Arras"` in implicit RLOF trials alongside `"Ritter"` and `"Kolb"`.
    assert!(solver
        .evaluate_trial("bogus_scheme", 0, 10, 0.0, 0.0, -1.0, -1.0, 0.8)
        .is_err());
    assert!(solver
        .evaluate_trial("Arras", 1, 10, 0.0, 0.0, -1.0, -1.0, 0.8)
        .is_ok());

    let donor = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("zams");
    assert!(
        eval_scheme_explicit_mdot(
            "roche_lobe",
            &donor,
            1.0e33,
            1.0e11,
            3.0e11,
            0.0,
        )
        .is_err()
    );
    assert!(
        eval_scheme_explicit_mdot(
            "bogus_mdot",
            &donor,
            1.0e33,
            1.0e11,
            3.0e11,
            0.0,
        )
        .is_err()
    );
}
