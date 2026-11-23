// SPDX-License-Identifier: LGPL-3.0-or-later
//! Issue `#147` fail-loudly validation tests for `flexmesa-star`.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use flexmesa_schema::upstream_section_keys;
use flexmesa_star::evolve::star_evolve_step;
use flexmesa_star::inlist::{InlistBundle, InlistLoadOptions};
use flexmesa_star::inlist_inventory::{
    CONTROLS_SUPPORTED, CONTROLS_TERMINAL_OUTPUT, CONTROLS_UNUSED,
    EOS_SUPPORTED, KAP_SUPPORTED, STAR_JOB_HEADLESS_GUI, STAR_JOB_SUPPORTED,
    STAR_JOB_TERMINAL_OUTPUT, STAR_JOB_UNUSED,
};
use flexmesa_star::pulse_exporters::export_pulse_by_format;
use flexmesa_star::rn_parser::parse_rn_stages;
use flexmesa_star::{
    StarControls, StarError, StarJobControls, load_zams_model,
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
            "flexmesa_star_147_{tag}_{}_{id}",
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

/// Keys that only gate or format terminal text upstream are accepted and
/// recorded, not rejected (`DECISIONS #26`).
#[test]
fn test_terminal_output_keys_pass_strict_validation_and_are_recorded() {
    let tmp = TempCaseGuard::new("terminal_output");
    let term = tmp.root.join("inlist_terminal_output");
    fs::write(
        &term,
        "&star_job\n  show_log_description_at_start = .false.\n  \
         pause_before_terminate = .false.\n  \
         first_model_for_timing = 5\n  \
         warn_run_star_extras = .false.\n/\n\
         &controls\n  initial_mass = 1.0d0\n  terminal_interval = 10\n  \
         write_header_frequency = 10\n  num_trace_history_values = 1\n  \
         trace_history_value_name(1) = 'rel_E_err'\n  \
         extra_terminal_output_file = 'log1'\n  \
         report_solver_progress = .true.\n  \
         warning_limit_for_max_residual = 1d99\n/\n",
    )
    .expect("write");
    let term_bundle = InlistBundle::from_file(&term)
        .expect("terminal-output keys must pass strict validation");
    let term_report = &term_bundle.validation_report;
    assert_eq!(
        term_report.unsupported_controls.len(),
        0,
        "unexpected unsupported controls: {:?}",
        term_report.unsupported_controls
    );
    let recorded: Vec<&str> = term_report
        .terminal_output_keys
        .iter()
        .map(|(_, k)| k.as_str())
        .collect();
    for key in [
        "show_log_description_at_start",
        "pause_before_terminate",
        "first_model_for_timing",
        "warn_run_star_extras",
        "terminal_interval",
        "write_header_frequency",
        "num_trace_history_values",
        "trace_history_value_name",
        "extra_terminal_output_file",
        "report_solver_progress",
        "warning_limit_for_max_residual",
    ] {
        assert!(recorded.contains(&key), "{key} not recorded: {recorded:?}");
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_unknown_section_key_unused_and_removed_aliases_fail_loudly() {
    let tmp = TempCaseGuard::new("strict");
    let bogus_sec = tmp.root.join("inlist_bogus_sec");
    fs::write(&bogus_sec, "&not_a_section\n  x = 1\n/\n").expect("write");
    let err = InlistBundle::from_file(&bogus_sec)
        .expect_err("unknown section must fail");
    assert!(matches!(err, StarError::InvalidInput(_)));
    assert!(err.to_string().contains("not_a_section"));

    let unknown_key = tmp.root.join("inlist_unknown_key");
    fs::write(&unknown_key, "&controls\n  not_a_mesa_control = 1.0d0\n/\n")
        .expect("write");
    assert!(matches!(
        InlistBundle::from_file(&unknown_key),
        Err(StarError::InvalidInput(_))
    ));
    assert!(
        InlistBundle::from_file_with_options(
            &unknown_key,
            InlistLoadOptions {
                allow_unsupported_controls: true,
            },
        )
        .is_err(),
        "unknown controls must fail even with allow_unsupported_controls=true"
    );

    let unimpl = tmp.root.join("inlist_unimpl");
    fs::write(
        &unimpl,
        "&controls\n  initial_mass = 1.0d0\n  fgong_ivers = 300\n/\n",
    )
    .expect("write");
    let unimpl_err = InlistBundle::from_file(&unimpl)
        .expect_err("unimplemented upstream key must fail by default");
    assert!(unimpl_err.to_string().contains("fgong_ivers"));

    assert!(
        InlistBundle::from_file_with_options(
            &unimpl,
            InlistLoadOptions {
                allow_unsupported_controls: true,
            },
        )
        .is_ok()
    );

    for (tag, body, expected) in [
        (
            "unused_job",
            "&star_job\n  h_he_net = 'h_he.net'\n/\n",
            "h_he_net",
        ),
        (
            "unused_rsp_map",
            "&controls\n  rsp_map_filename = 'map.data'\n/\n",
            "rsp_map_filename",
        ),
        (
            "unused_neg_mass_frac",
            "&controls\n  neg_mass_fraction_hold = 1.0d-4\n/\n",
            "neg_mass_fraction_hold",
        ),
        (
            "unused_bad_max_corr",
            "&controls\n  tol_bad_max_correction = 1.0d-2\n/\n",
            "tol_bad_max_correction",
        ),
        (
            "unused_surface_pt",
            "&controls\n  use_other_surface_pt = .true.\n/\n",
            "use_other_surface_pt",
        ),
    ] {
        let p = tmp.root.join(format!("inlist_{tag}"));
        fs::write(&p, body).expect("write");
        let e = InlistBundle::from_file(&p)
            .expect_err("unused control must fail by default");
        assert!(
            e.to_string().contains(expected),
            "expected '{expected}' in error: {e}"
        );
    }

    for (idx, body) in [
        "&star_job\n  saved_model_name = 'm.mod'\n/\n",
        "&star_job\n  load_saved_model_for_rsp2 = .true.\n/\n",
        "&controls\n  rgb_wind_scheme = 'Reimers'\n/\n",
        "&controls\n  agb_wind_scheme = 'Blocker'\n/\n",
        "&controls\n  reimers_wind_eta = 0.5\n/\n",
        "&controls\n  overshoot_f2(1) = 0.01\n/\n",
        "&controls\n  initial_dump_missing_heaviest = .true.\n/\n",
        "&controls\n  min_logt_for_net = 6.0\n/\n",
    ]
    .iter()
    .enumerate()
    {
        let p = tmp.root.join(format!("inlist_alias_{idx}"));
        fs::write(&p, body).expect("write alias");
        let err = InlistBundle::from_file_with_options(
            &p,
            InlistLoadOptions {
                allow_unsupported_controls: true,
            },
        )
        .expect_err("removed non-upstream alias must always fail");
        assert!(matches!(err, StarError::InvalidInput(_)));
    }
}

#[test]
fn test_headless_pgstar_valid_keys_noop_and_typos_fail_loudly() {
    let tmp = TempCaseGuard::new("pgstar");
    let valid = tmp.root.join("inlist_pgstar_ok");
    fs::write(
        &valid,
        "&star_job\n  pgstar_flag = .true.\n/\n\
         &pgstar\n  pgstar_interval = 1\n  hr_logt_min = 3.5d0\n/\n\
         &controls\n  initial_mass = 1.0d0\n/\n",
    )
    .expect("write valid pgstar");
    let bundle = InlistBundle::from_file(&valid)
        .expect("valid headless &pgstar must succeed in strict mode");
    assert!(
        bundle.validation_report.pgstar_plotting_keys.contains(&(
            String::from("pgstar"),
            String::from("pgstar_interval")
        ))
    );

    let bad = tmp.root.join("inlist_pgstar_bad");
    fs::write(
        &bad,
        "&star_job\n  pgstar_flag = .true.\n/\n\
         &pgstar\n  grid7_file_cnt = 2\n/\n\
         &controls\n  initial_mass = 1.0d0\n/\n",
    )
    .expect("write bad pgstar");
    let err = InlistBundle::from_file(&bad)
        .expect_err("misspelled active &pgstar key must fail loudly");
    assert!(err.to_string().contains("grid7_file_cnt"));

    let inactive_stale = tmp.root.join("inlist_pgstar_inactive");
    fs::write(
        &inactive_stale,
        "&star_job\n  pgstar_flag = .false.\n/\n\
         &pgstar\n  grid7_file_cnt = 2\n  pgstar_interval = 1\n/\n\
         &controls\n  initial_mass = 1.0d0\n/\n",
    )
    .expect("write inactive pgstar");
    let ok_inactive = InlistBundle::from_file(&inactive_stale)
        .expect("inactive pgstar must ignore stale keys");
    assert!(
        ok_inactive
            .validation_report
            .pgstar_plotting_keys
            .contains(&(
                String::from("pgstar"),
                String::from("pgstar_interval")
            ))
    );
}

#[test]
fn test_star_inventory_subset_of_mesa_26_4_1_and_x_ctrl_indexing() {
    for (sec, list) in [
        ("star_job", STAR_JOB_SUPPORTED),
        ("star_job", STAR_JOB_UNUSED),
        ("star_job", STAR_JOB_HEADLESS_GUI),
        ("star_job", STAR_JOB_TERMINAL_OUTPUT),
        ("eos", EOS_SUPPORTED),
        ("kap", KAP_SUPPORTED),
        ("controls", CONTROLS_SUPPORTED),
        ("controls", CONTROLS_TERMINAL_OUTPUT),
        ("controls", CONTROLS_UNUSED),
    ] {
        let upstream = upstream_section_keys(sec)
            .unwrap_or_else(|| panic!("missing schema keys for &{sec}"));
        for &key in list {
            assert!(
                upstream.contains(key),
                "star inventory entry '&{sec}::{key}' missing from MESA 26.4.1 schema"
            );
        }
    }

    let tmp = TempCaseGuard::new("xctrl");
    let x_zero = tmp.root.join("inlist_x_zero");
    fs::write(&x_zero, "&controls\n  x_ctrl(0) = 1.0d0\n/\n").expect("write");
    assert!(InlistBundle::from_file(&x_zero).is_err());

    let x_scalar = tmp.root.join("inlist_x_scalar");
    fs::write(&x_scalar, "&controls\n  x_ctrl = 1.0d0\n/\n").expect("write");
    assert!(InlistBundle::from_file(&x_scalar).is_err());

    let x_one = tmp.root.join("inlist_x_one");
    fs::write(
        &x_one,
        "&controls\n  x_ctrl(1) = 3.5d0\n  x_integer_ctrl(2) = 7\n  x_logical_ctrl(3) = .true.\n/\n",
    )
    .expect("write");
    let bundle = InlistBundle::from_file(&x_one).expect("valid x_ctrl(1)");
    assert!((bundle.controls.x_ctrl[0] - 3.5).abs() < 1.0e-12);
    assert_eq!(bundle.controls.x_integer_ctrl[1], 7);
    assert!(bundle.controls.x_logical_ctrl[2]);

    let ctrl_rti = tmp.root.join("inlist_ctrl_rti");
    fs::write(&ctrl_rti, "&controls\n  rti_flag = .true.\n/\n").expect("write");
    assert!(
        InlistBundle::from_file_with_options(
            &ctrl_rti,
            InlistLoadOptions {
                allow_unsupported_controls: true,
            },
        )
        .is_err()
    );
}

#[test]
fn test_unrecognized_option_strings_and_simple_photosphere_fail_loudly() {
    let tmp = TempCaseGuard::new("options");
    for (idx, body) in [
        "&controls\n  atm_option = 'simple_photosphere'\n/\n",
        "&controls\n  atm_option = 'bogus_atm'\n/\n",
        "&controls\n  mlt_option = 'bogus_mlt'\n/\n",
        "&controls\n  atm_T_tau_relation = 'bogus_ttau'\n/\n",
        "&controls\n  screening_mode = 'bogus_screen'\n/\n",
        "&controls\n  energy_eqn_option = 'bogus_energy'\n/\n",
        "&controls\n  semiconvection_option = 'bogus_semi'\n/\n",
        "&controls\n  thermohaline_option = 'bogus_thermo'\n/\n",
        "&controls\n  kap_file_prefix = 'bogus_kap'\n/\n",
        "&controls\n  kap_co_prefix = 'bogus_co'\n/\n",
        "&controls\n  kap_lowt_prefix = 'bogus_lowt'\n/\n",
        "&controls\n  cool_wind_rgb_scheme = 'bogus_wind'\n/\n",
        "&controls\n  overshoot_scheme(1) = 'bogus_ov'\n/\n",
        "&controls\n  phase_separation_option = 'bogus_phase'\n/\n",
    ]
    .iter()
    .enumerate()
    {
        let p = tmp.root.join(format!("inlist_opt_{idx}"));
        fs::write(&p, body).expect("write bad option");
        let err = InlistBundle::from_file(&p)
            .expect_err("unrecognized option string must fail");
        assert!(
            matches!(err, StarError::InvalidInput(_)),
            "expected InvalidInput for {body}: got {err}"
        );
    }

    let model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("zams");
    assert!(matches!(
        export_pulse_by_format(&model, "bogus_format", true),
        Err(StarError::InvalidInput(_))
    ));
}

#[test]
fn test_missing_files_stages_and_removed_clamps_behave_strictly() {
    let tmp = TempCaseGuard::new("missing");

    let missing_inc = tmp.root.join("inlist_missing_inc");
    fs::write(
        &missing_inc,
        "&controls\n  read_extra_controls_inlist(1) = .true.\n  extra_controls_inlist_name(1) = 'no_such_extra'\n/\n",
    )
    .expect("write missing inc");
    assert!(InlistBundle::from_file(&missing_inc).is_err());

    let skipped_inc = tmp.root.join("inlist_skipped_inc");
    fs::write(
        &skipped_inc,
        "&controls\n  read_extra_controls_inlist(1) = .false.\n  extra_controls_inlist_name(1) = 'no_such_extra'\n/\n",
    )
    .expect("write skipped inc");
    assert!(InlistBundle::from_file(&skipped_inc).is_ok());

    let missing_mod = tmp.root.join("inlist_missing_mod");
    fs::write(
        &missing_mod,
        "&star_job\n  load_saved_model = .true.\n  load_model_filename = 'missing_model.mod'\n/\n&controls\n  initial_mass = 1.0d0\n/\n",
    )
    .expect("write missing mod");
    let bundle = InlistBundle::from_file(&missing_mod).expect("parse ok");
    let mut extras = flexmesa_star::inlist_runner::DefaultStarExtras;
    assert!(
        flexmesa_star::init_or_update_model_from_bundle(
            &bundle,
            None,
            &mut extras,
        )
        .is_err()
    );

    let empty_rn_dir = tmp.root.join("empty_rn_case");
    fs::create_dir_all(&empty_rn_dir).expect("mkdir");
    fs::write(empty_rn_dir.join("rn"), "#!/bin/sh\necho nothing\n")
        .expect("write empty rn");
    assert!(parse_rn_stages(&empty_rn_dir, true).is_err());

    let zero_retry = tmp.root.join("inlist_zero_retry");
    fs::write(
        &zero_retry,
        "&controls\n  retry_limit = 0\n  timestep_factor_for_retries = 0.25d0\n/\n",
    )
    .expect("write zero retry");
    let b0 = InlistBundle::from_file(&zero_retry).expect("retry_limit=0");
    assert_eq!(b0.controls.ts_stop.retry_limit, 0);

    // Upstream applies `retry_limit` and `redo_limit` only when positive
    // and accepts any integer (`star/private/evolve.f90:2161,2214`).
    let neg_retry = tmp.root.join("inlist_neg_retry");
    fs::write(&neg_retry, "&controls\n  retry_limit = -2\n/\n")
        .expect("write negative retry limit");
    let bneg = InlistBundle::from_file(&neg_retry)
        .expect("a non-positive retry_limit means no limit");
    assert_eq!(bneg.controls.ts_stop.retry_limit, 0);

    // Upstream applies `redo_limit` only when positive and accepts negative
    // sentinels (`star/private/evolve.f90:2161`; `binary/test_suite/double_bh/inlist1:47`
    // sets `redo_limit = -1`).
    let neg_redo = tmp.root.join("inlist_neg_redo");
    fs::write(&neg_redo, "&controls\n  redo_limit = -1\n/\n")
        .expect("write negative redo limit");
    let bredo = InlistBundle::from_file(&neg_redo)
        .expect("a non-positive redo_limit means no limit");
    assert_eq!(bredo.controls.ts_stop.redo_limit, 0);

    let bad_retry_fac = tmp.root.join("inlist_bad_retry_fac");
    fs::write(
        &bad_retry_fac,
        "&controls\n  timestep_factor_for_retries = 1.0d0\n/\n",
    )
    .expect("write bad retry fac");
    assert!(InlistBundle::from_file(&bad_retry_fac).is_err());

    let mut model = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("zams");
    model.controls.ts_stop.timestep_factor_for_retries = 1.2;
    assert!(matches!(
        star_evolve_step(&mut model, false, false, false),
        Err(StarError::InvalidInput(_))
    ));
}
