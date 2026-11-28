// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration tests for `StarExtras` extra profile columns and extra
//! history/profile header items (`#442`, MESA `26.4.1` parity).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use flexmesa_const::SECYER;
use flexmesa_star::mesa_history::format_star_history_file_header_with_extras;
use flexmesa_star::mesa_profile::{
    default_star_profile_specs, format_star_profile_text_with_extras,
};
use flexmesa_star::{
    default_star_history_specs, load_zams_model, read_mesa_log_file,
    read_mesa_log_str, run_test_suite_case_with_extras, set_hydro_vars,
    IncrementalHistoryWriter, InlistLoadOptions, MesaHeaderValue,
    OutputControls, StarControls, StarError, StarExtras, StarJobControls,
    StarLogsOptions, StarModel,
};

struct TempTestDir(PathBuf);

impl TempTestDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "flexmesa_star_442_{label}_{}_{n}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp test dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempTestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn load_minimal_zams_model(
    controls: StarControls,
    job: StarJobControls,
) -> StarModel {
    let mut model = load_zams_model(controls, job).expect("load zams model");
    model.dt = 1.0e3 * SECYER;
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.save_start_values().expect("save start values");
    model
}

#[test]
fn test_extra_history_and_profile_headers_and_columns_format_roundtrip() {
    let controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        zbase: 0.02,
        output: OutputControls {
            max_num_profile_zones: 5,
            ..OutputControls::default()
        },
        ..StarControls::default()
    };
    let mut model =
        load_minimal_zams_model(controls, StarJobControls::default());
    model.model_number = 7;

    let hist_hdr = format_star_history_file_header_with_extras(
        &model,
        &default_star_history_specs(),
        &[
            (String::from("mixing_length_alpha"), 1.85),
            (String::from("custom_scale"), 42.0),
        ],
        &["extra_h_col"],
    );
    let mut hist_text = hist_hdr;
    let row_eval = flexmesa_star::evaluate_star_history_row(
        &model,
        &default_star_history_specs(),
    )
    .expect("history row");
    hist_text.push_str(
        &flexmesa_star::format_star_history_data_line_with_extra(
            &row_eval,
            &[(String::from("extra_h_col"), -3.25)],
        ),
    );
    let hist_tbl =
        read_mesa_log_str(&hist_text).expect("parse history with extra header");
    assert_eq!(hist_tbl.header.len(), 13);
    assert_eq!(hist_tbl.header[10].0, "lsun");
    assert_eq!(hist_tbl.header[11].0, "mixing_length_alpha");
    assert_eq!(hist_tbl.header[11].1, MesaHeaderValue::Float(1.85));
    assert_eq!(hist_tbl.header[12].0, "custom_scale");
    assert_eq!(hist_tbl.header[12].1, MesaHeaderValue::Float(42.0));
    assert_eq!(
        hist_tbl.column_names.last().map(String::as_str),
        Some("extra_h_col")
    );
    assert_eq!(hist_tbl.column("extra_h_col"), Some(&[-3.25][..]));

    let zbar_div_abar: Vec<f64> = (0..model.nz)
        .map(|k| model.zbar[k] / model.abar[k])
        .collect();
    let prof_text = format_star_profile_text_with_extras(
        &model,
        &default_star_profile_specs(),
        &[(String::from("mixing_length_alpha"), 1.85)],
        &[(String::from("zbar_div_abar"), zbar_div_abar.clone())],
    )
    .expect("format subsampled profile with extras");
    let prof_tbl =
        read_mesa_log_str(&prof_text).expect("parse profile with extras");
    assert_eq!(prof_tbl.num_rows, 5);
    assert_eq!(prof_tbl.header[48].0, "lsun");
    assert_eq!(prof_tbl.header[49].0, "mixing_length_alpha");
    assert_eq!(prof_tbl.header[49].1, MesaHeaderValue::Float(1.85));
    assert_eq!(
        prof_tbl.column_names.last().map(String::as_str),
        Some("zbar_div_abar")
    );
    let zones = prof_tbl.column("zone").expect("zone column");
    let parsed_vals = prof_tbl
        .column("zbar_div_abar")
        .expect("zbar_div_abar column");
    for (row_idx, (&z, &v)) in zones.iter().zip(parsed_vals.iter()).enumerate()
    {
        let kk = (z as usize) - 1;
        let expected = zbar_div_abar[kk];
        assert!(
            (v - expected).abs() <= 1.0e-13,
            "subsampled row {row_idx} zone {z}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn test_extra_profile_and_header_validation_fails_loudly() {
    let controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    let model = load_minimal_zams_model(controls, StarJobControls::default());
    let bad_len = format_star_profile_text_with_extras(
        &model,
        &default_star_profile_specs(),
        &[],
        &[(String::from("bad_col"), vec![1.0; model.nz - 1])],
    )
    .expect_err("length mismatch must fail loudly");
    assert!(matches!(bad_len, StarError::InvalidInput(_)));

    let bad_prof_col_name = format_star_profile_text_with_extras(
        &model,
        &default_star_profile_specs(),
        &[],
        &[(String::from("   "), vec![1.0; model.nz])],
    )
    .expect_err("blank profile column name must fail loudly");
    assert!(matches!(bad_prof_col_name, StarError::InvalidInput(_)));

    let bad_prof_hdr_name = format_star_profile_text_with_extras(
        &model,
        &default_star_profile_specs(),
        &[(String::new(), 1.0)],
        &[],
    )
    .expect_err("blank profile header name must fail loudly");
    assert!(matches!(bad_prof_hdr_name, StarError::InvalidInput(_)));

    let tmp = TempTestDir::new("bad_hist_hdr");
    let mut hw = IncrementalHistoryWriter::new(
        tmp.path().join("history.data"),
        default_star_history_specs(),
    );
    let ctx = flexmesa_star::StepReportContext::compute(&model);
    let bad_hist_hdr = hw
        .write_step_with_ctx_and_extras(
            &model,
            &ctx,
            &[(String::from("  "), 1.0)],
            &[],
        )
        .expect_err("blank history header name must fail loudly");
    assert!(matches!(bad_hist_hdr, StarError::InvalidInput(_)));
}

struct RecordingProfileHeaderExtras {
    hist_hdr_calls: AtomicUsize,
    prof_hdr_calls: AtomicUsize,
    prof_col_calls: AtomicUsize,
    wrong_length: bool,
}

impl Default for RecordingProfileHeaderExtras {
    fn default() -> Self {
        Self {
            hist_hdr_calls: AtomicUsize::new(0),
            prof_hdr_calls: AtomicUsize::new(0),
            prof_col_calls: AtomicUsize::new(0),
            wrong_length: false,
        }
    }
}

impl StarExtras for RecordingProfileHeaderExtras {
    fn extra_history_columns(&self, m: &StarModel) -> Vec<(String, f64)> {
        vec![(String::from("step_alpha"), m.controls.mixing_length_alpha)]
    }

    fn extra_history_header_items(
        &self,
        m: &StarModel,
    ) -> Result<Vec<(String, f64)>, StarError> {
        self.hist_hdr_calls.fetch_add(1, Ordering::Relaxed);
        Ok(vec![(
            String::from("mixing_length_alpha"),
            m.controls.mixing_length_alpha,
        )])
    }

    fn extra_profile_header_items(
        &self,
        m: &StarModel,
    ) -> Result<Vec<(String, f64)>, StarError> {
        self.prof_hdr_calls.fetch_add(1, Ordering::Relaxed);
        Ok(vec![(
            String::from("mixing_length_alpha"),
            m.controls.mixing_length_alpha,
        )])
    }

    fn extra_profile_columns(
        &self,
        m: &StarModel,
    ) -> Result<Vec<(String, Vec<f64>)>, StarError> {
        self.prof_col_calls.fetch_add(1, Ordering::Relaxed);
        let len = if self.wrong_length {
            m.nz.saturating_sub(1)
        } else {
            m.nz
        };
        let vals = (0..len).map(|k| m.zbar[k] / m.abar[k]).collect();
        Ok(vec![(String::from("zbar_div_abar"), vals)])
    }
}

fn write_minimal_two_step_case(case_dir: &Path) {
    fs::write(case_dir.join("rn"), "#!/bin/bash\ndo_one inlist_test\n")
        .expect("write rn");
    fs::write(
        case_dir.join("inlist_test"),
        "\
&star_job
  profile_starting_model = .true.
  write_profile_when_terminate = .true.
  filename_for_profile_when_terminate = 'final_prof.data'
  steps_to_take_before_terminate = 2
/
&controls
  initial_mass = 1.0
  initial_z = 0.02
  mixing_length_alpha = 1.75
  history_interval = 1
  profile_interval = 1
  max_num_profile_models = 10
  write_profiles_flag = .true.
/
&kap
  zbase = 0.02d0
/
",
    )
    .expect("write inlist_test");
}

#[test]
fn test_run_test_suite_case_wires_profile_columns_and_headers() {
    let case_dir = TempTestDir::new("case");
    let out_dir = TempTestDir::new("out");
    write_minimal_two_step_case(case_dir.path());

    let mut extras = RecordingProfileHeaderExtras::default();
    let rep = run_test_suite_case_with_extras(
        case_dir.path(),
        true,
        Some(2),
        false,
        StarLogsOptions {
            output_root: Some(out_dir.path()),
            explicit_history_file: None,
            history_interval_override: Some(1),
            write_profiles_override: Some(true),
        },
        InlistLoadOptions::default(),
        &mut extras,
    )
    .expect("run two-step case with profile/header extras");
    assert_eq!(rep.total_steps, 2);
    // `extra_history_header_items` runs once when writing the initial header,
    // even across 2 history rows.
    assert_eq!(extras.hist_hdr_calls.load(Ordering::Relaxed), 1);
    // Starting model (build_writer) + step 1 + step 2 (also term_prof_due).
    assert_eq!(extras.prof_hdr_calls.load(Ordering::Relaxed), 3);
    assert_eq!(extras.prof_col_calls.load(Ordering::Relaxed), 3);

    let log_dir = out_dir.path().join("LOGS");
    let hist = read_mesa_log_file(&log_dir.join("history.data"))
        .expect("history.data");
    assert_eq!(hist.num_rows, 2);
    assert_eq!(
        hist.header_value("mixing_length_alpha"),
        Some(&MesaHeaderValue::Float(1.75))
    );
    assert_eq!(hist.column("step_alpha"), Some(&[1.75, 1.75][..]));

    for prof_path in [
        log_dir.join("profile1.data"),
        log_dir.join("profile2.data"),
        log_dir.join("profile3.data"),
        out_dir.path().join("final_prof.data"),
    ] {
        let prof = read_mesa_log_file(&prof_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", prof_path.display()));
        assert_eq!(
            prof.header_value("mixing_length_alpha"),
            Some(&MesaHeaderValue::Float(1.75))
        );
        let z_over_a = prof
            .column("zbar_div_abar")
            .expect("zbar_div_abar profile column");
        assert_eq!(z_over_a.len(), prof.num_rows);
        assert!(z_over_a
            .iter()
            .all(|v| v.is_finite() && *v >= 0.49 && *v <= 1.0));
    }

    // Mismatched profile column length must abort execution loudly.
    let bad_out = TempTestDir::new("bad_out");
    let mut bad_extras = RecordingProfileHeaderExtras {
        wrong_length: true,
        ..RecordingProfileHeaderExtras::default()
    };
    let err = run_test_suite_case_with_extras(
        case_dir.path(),
        true,
        Some(2),
        false,
        StarLogsOptions {
            output_root: Some(bad_out.path()),
            explicit_history_file: None,
            history_interval_override: Some(1),
            write_profiles_override: Some(true),
        },
        InlistLoadOptions::default(),
        &mut bad_extras,
    )
    .expect_err("wrong extra profile column length must fail stage");
    assert!(matches!(err, StarError::InvalidInput(_)));
}
