// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests for MESA `26.4.1` `binary_history.data` round-trip reading,
//! 32-character column formatting, `[donor, accretor] -> [star_1, star_2]` slot
//! mapping, `step_mtransfer_rate` stepping, interval gating, and
//! `append_to_star_history` (`#163`).

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_binary::{
    BINARY_LOG_COL_WIDTH, BinaryLogsWriter, BinarySystem, DefaultBinaryExtras,
    IncrementalBinaryHistoryWriter, default_binary_history_specs,
    evaluate_binary_history_row, parse_binary_history_columns,
};
use flexmesa_const::{MSUN, SECYER};
use flexmesa_star::{
    MesaCellValue, MesaHeaderValue, OutputControls, StarControls, StarError,
    StarJobControls, load_zams_model, read_mesa_log_file, read_mesa_log_str,
    safe_log10, set_hydro_vars,
};

/// Verbatim 8-line prefix (6 header lines + first 2 data rows) from real Fortran
/// MESA `26.4.1` (`binary/test_suite/evolve_both_stars/binary_history.data`).
const REAL_MESA_26_4_1_EVOLVE_BOTH_STARS_BINARY_HISTORY_HEAD: &str = concat!(
    "                               1                                2                                3                                4                                5                                6                                7                                8 \n",
    "                  version_number                 initial_don_mass                 initial_acc_mass              initial_period_days                         compiler                            build                 MESA_SDK_version                             date \n",
    "                       \"r26.4.1\"          1.0000000000000000E+000          8.0000000000000004E-001          5.0000000000000000E-001                       \"gfortran\"                         \"15.2.0\"            \"x86_64-linux-26.6.1\"                       \"20261115\" \n",
    "\n",
    "                               1                                2                                3                                4                                5                                6                                7                                8                                9                               10                               11                               12                               13                               14                               15                               16                               17                               18                               19                               20                               21                               22                               23                               24                               25                               26                               27                               28                               29                               30                               31 \n",
    "                    model_number                              age                      period_days                binary_separation                          v_orb_1                          v_orb_2                             rl_1                             rl_2           rl_relative_overflow_1           rl_relative_overflow_2                      star_1_mass                      star_2_mass                lg_mtransfer_rate                   lg_mstar_dot_1                   lg_mstar_dot_2                 lg_system_mdot_1                 lg_system_mdot_2                   lg_wind_mdot_1                   lg_wind_mdot_2              fixed_xfer_fraction                eff_xfer_fraction                            J_orb                             Jdot                          jdot_mb                          jdot_gr                          jdot_ml                          jdot_ls                jdot_missing_wind                       extra_jdot                      donor_index                 point_mass_index \n",
    "                               5          7.4416000000000000E+005          4.9998763143955111E-001          3.2247902081002859E+000          1.4502665044134557E+002          1.8128331305168197E+002          1.2846914762054187E+000          1.1601902356566174E+000         -3.0749272197402672E-001         -3.8155383404825138E-001          1.0000000000000000E+000          8.0000000000000004E-001         -9.8999999999999986E+001         -9.8999999999999986E+001         -9.8999999999999986E+001         -9.8999999999999986E+001         -9.8999999999999986E+001         -9.8999999999999986E+001         -9.8999999999999986E+001          1.0000000000000000E+000          1.0000000000000000E+000          6.4695965231490442E+051         -2.2716958557536661E+033          0.0000000000000000E+000         -2.2716958557536661E+033          0.0000000000000000E+000          0.0000000000000000E+000          0.0000000000000000E+000          0.0000000000000000E+000                                1                                0 \n",
    "                              10          2.5958682111999998E+006          4.9995685257157724E-001          3.2246578629458686E+000          1.4502962647451452E+002          1.8128703309314318E+002          1.2846387525611425E+000          1.1601426215341528E+000         -3.0715040857389853E-001         -3.8135548225700222E-001          1.0000000000000000E+000          8.0000000000000004E-001         -9.8999999999999986E+001         -9.8999999999999986E+001         -9.8999999999999986E+001         -9.8999999999999986E+001         -9.8999999999999986E+001         -9.8999999999999986E+001         -9.8999999999999986E+001          1.0000000000000000E+000          1.0000000000000000E+000          6.4694637659027416E+051         -2.2719677910928292E+033          0.0000000000000000E+000         -2.2719677910928292E+033          0.0000000000000000E+000          0.0000000000000000E+000          0.0000000000000000E+000          0.0000000000000000E+000                                1                                0 \n",
);

struct TempTestDir(PathBuf);

impl TempTestDir {
    fn new(label: &str) -> Self {
        let dir = std::env::temp_dir()
            .join(format!("flexmesa_bin_163_{label}_{}", std::process::id()));
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

#[test]
fn test_real_mesa_26_4_1_binary_history_roundtrip_and_loud_parser_errors() {
    let tbl = read_mesa_log_str(REAL_MESA_26_4_1_EVOLVE_BOTH_STARS_BINARY_HISTORY_HEAD)
        .expect("parse real MESA 26.4.1 binary_history.data");
    assert_eq!(tbl.header.len(), 8);
    assert_eq!(
        tbl.header_value("version_number"),
        Some(&MesaHeaderValue::Text(String::from("r26.4.1")))
    );
    assert_eq!(
        tbl.header_value("initial_don_mass"),
        Some(&MesaHeaderValue::Float(1.0))
    );
    let specs = default_binary_history_specs();
    let expected_cols: Vec<String> =
        specs.iter().map(|s| s.column_name().to_owned()).collect();
    assert_eq!(tbl.column_names, expected_cols);
    assert_eq!(tbl.num_rows, 2);
    assert_eq!(tbl.column("model_number"), Some(&[5.0, 10.0][..]));
    assert!(
        (tbl.column("period_days").unwrap()[0] - 4.999_876_314_395_511e-1).abs()
            < 1.0e-16
    );

    // Reject unknown, unported, empty, and trailing-garbage `.list` inputs loudly.
    for bad_input in [
        "model_number\nbogus_bin_col\n",
        "model_number\nCE_flag\n",
        "model_number\nextra_edot\n",
        "! comment only\n\n",
        "include '' trailing_garbage\n",
    ] {
        assert!(
            matches!(
                parse_binary_history_columns(bad_input, None),
                Err(StarError::InvalidInput(_))
            ),
            "expected InvalidInput on {bad_input:?}"
        );
    }
}

#[test]
fn test_binary_history_donor_switch_mapping_and_width_32() {
    let c1 = StarControls {
        initial_mass: 2.0,
        initial_z: 0.02,
        output: OutputControls {
            log_directory: String::from("LOGS1"),
            history_interval: 5,
            write_profiles_flag: false,
            ..OutputControls::default()
        },
        ..StarControls::default()
    };
    let mut c2 = c1.clone();
    c2.initial_mass = 1.2;
    c2.output.log_directory = String::from("LOGS2");

    let s1 = load_zams_model(c1.clone(), StarJobControls::default()).expect("s1");
    let s2 = load_zams_model(c2, StarJobControls::default()).expect("s2");

    // Verify colliding component `log_directory` paths fail loudly.
    {
        let mut s2_collide = s2.clone();
        s2_collide.controls.output.log_directory = String::from("LOGS1");
        let collide_sys =
            BinarySystem::new_two_stars(s1.clone(), s2_collide, 2.5, 0.0);
        let tmp_col = TempTestDir::new("collide");
        let err = BinaryLogsWriter::for_run(
            &collide_sys,
            tmp_col.path(),
            tmp_col.path(),
            None,
            None,
            true,
        );
        assert!(matches!(err, Err(StarError::InvalidInput(_))));
    }

    let mut sys = BinarySystem::new_two_stars(s1, s2, 2.5, 0.0);
    sys.controls.history_interval = 1;
    sys.controls.append_to_star_history = true;
    sys.model_number = 1;
    sys.binary_age_yr = 1.0e4;
    sys.step_mtransfer_rate = -1.0e-7 * MSUN / SECYER;
    sys.last_mdot.component_mdot = [
        -1.0e-7 * MSUN / SECYER,
        0.8e-7 * MSUN / SECYER,
    ];

    let specs = default_binary_history_specs();
    assert_eq!(specs.len(), 31);
    let row_d1 = evaluate_binary_history_row(&sys, &specs);
    let get_val = |row: &[(String, MesaCellValue)], name: &str| -> f64 {
        row.iter()
            .find_map(|(k, v)| {
                (k == name).then_some(match *v {
                    MesaCellValue::Float(x) => x,
                    MesaCellValue::Int(i) => i as f64,
                })
            })
            .unwrap_or_else(|| panic!("missing binary column {name}"))
    };
    assert!((get_val(&row_d1, "star_1_mass") - 2.0).abs() < 1.0e-10);
    assert!((get_val(&row_d1, "star_2_mass") - 1.2).abs() < 1.0e-10);
    assert_eq!(get_val(&row_d1, "donor_index") as i64, 1);
    assert!((get_val(&row_d1, "lg_mstar_dot_1") - (-7.0)).abs() < 1.0e-12);
    assert!(
        (get_val(&row_d1, "lg_mstar_dot_2") - safe_log10(0.8e-6 * 0.1)).abs() < 1.0e-12
    );

    // Force donor switch (`donor_i = 2`): star 1 must remain 2.0 Msun and star 2
    // must remain 1.2 Msun, while `donor_slot = 0` now maps to `star_2`!
    sys.rl_relative_gap = [-0.2, 0.05];
    assert!(sys.check_donor_switch(false));
    assert_eq!(sys.donor_i, 2);
    sys.last_mdot.component_mdot = [
        -3.0e-6 * MSUN / SECYER, // donor = star 2
        1.5e-6 * MSUN / SECYER,  // accretor = star 1
    ];
    let row_d2 = evaluate_binary_history_row(&sys, &specs);
    assert_eq!(get_val(&row_d2, "donor_index") as i64, 2);
    assert!((get_val(&row_d2, "star_1_mass") - 2.0).abs() < 1.0e-10);
    assert!((get_val(&row_d2, "star_2_mass") - 1.2).abs() < 1.0e-10);
    assert!(
        (get_val(&row_d2, "lg_mstar_dot_1") - safe_log10(1.5e-6)).abs() < 1.0e-12
    );
    assert!(
        (get_val(&row_d2, "lg_mstar_dot_2") - safe_log10(3.0e-6)).abs() < 1.0e-12
    );
    assert!((get_val(&row_d2, "eff_xfer_fraction") - 0.5).abs() < 1.0e-12);

    // Verify 32-char column width on flushed `binary_history.data`.
    let tmp = TempTestDir::new("donor_switch");
    {
        let mut bw = IncrementalBinaryHistoryWriter::new(
            tmp.path().join("aborted_binary_history.data"),
            specs.clone(),
        );
        bw.write_step(&sys).expect("write step");
    }
    let raw_bin =
        fs::read_to_string(tmp.path().join("aborted_binary_history.data"))
            .expect("read flushed binary_history.data");
    let data_line = raw_bin.lines().nth(6).expect("row 1");
    assert_eq!(data_line.len(), specs.len() * (BINARY_LOG_COL_WIDTH + 1));
}

#[test]
fn test_binary_logs_writer_coupled_stepping_intervals() {
    let c1 = StarControls {
        initial_mass: 2.0,
        initial_z: 0.02,
        output: OutputControls {
            log_directory: String::from("LOGS1"),
            history_interval: 5,
            write_profiles_flag: false,
            ..OutputControls::default()
        },
        ..StarControls::default()
    };

    // Step a close donor + point-mass binary across 5 real `try_step_with_extras`
    // steps (`s1.model_number` starting at 10 vs `sys.model_number` starting at 0,
    // `history_interval = 5`, `history_interval_override = None`, explicit mass transfer).
    let mut donor_pm = load_zams_model(c1, StarJobControls::default()).expect("donor_pm");
    donor_pm.model_number = 10;
    donor_pm.dt = 1.0e2 * SECYER;
    donor_pm.dt_next = donor_pm.dt;
    set_hydro_vars(&mut donor_pm, false, false).expect("hydro");
    donor_pm.save_start_values();

    let mut sys_pm =
        BinarySystem::new_point_mass(donor_pm, 1.4, 0.35, 0.0, false, 0.0);
    sys_pm.controls.history_interval = 5;
    sys_pm.controls.append_to_star_history = true;
    sys_pm.controls.mdot_scheme = String::from("Ritter");
    sys_pm.controls.max_tries_to_achieve = 0; // explicit Ritter step

    let step_tmp = TempTestDir::new("coupled_steps");
    let mut coord = BinaryLogsWriter::for_run(
        &sys_pm,
        step_tmp.path(),
        step_tmp.path(),
        None,
        None,
        true,
    )
    .expect("BinaryLogsWriter");
    let mut extras = DefaultBinaryExtras;
    let mut dt_step = 1.0e2 * SECYER;
    for step in 1..=5_usize {
        let (next_dt, _) = sys_pm
            .try_step_with_extras(dt_step, &mut extras)
            .expect("coupled binary step");
        dt_step = next_dt;
        coord
            .record_step(&sys_pm, step == 1, false)
            .expect("record coupled step");
    }

    // `binary_history.data` (`history_interval = 5`, `is_terminating = false`)
    // writes ONLY step 5, whereas `LOGS1/history.data` writes step 1 (`is_first_step`)
    // and step 5 (`15 % 5 == 0`) with distinct stellar vs binary `model_number`!
    let bin_tbl = read_mesa_log_file(&step_tmp.path().join("binary_history.data"))
        .expect("read binary_history.data");
    assert_eq!(bin_tbl.num_rows, 1);
    assert_eq!(bin_tbl.column("model_number"), Some(&[5.0][..]));
    assert!(sys_pm.step_mtransfer_rate < 0.0);
    let expected_lg_mtr = safe_log10((-sys_pm.step_mtransfer_rate) / (MSUN / SECYER));
    assert!(
        (bin_tbl.column("lg_mtransfer_rate").unwrap()[0] - expected_lg_mtr).abs()
            < 1.0e-12
    );

    let star1_hist =
        read_mesa_log_file(&step_tmp.path().join("LOGS1").join("history.data"))
            .expect("read LOGS1/history.data");
    assert_eq!(star1_hist.column_names.len(), 91);
    assert_eq!(star1_hist.num_rows, 2);
    // First-occurrence `columns["model_number"]` retains stellar `model_number` (`11`, `15`),
    // while positional `column_values[60]` holds the appended binary `model_number` (`1`, `5`).
    assert_eq!(star1_hist.column("model_number"), Some(&[11.0, 15.0][..]));
    assert_eq!(&star1_hist.column_values[60], &[1.0, 5.0]);
    assert!(
        (star1_hist.column_values[72][1] - expected_lg_mtr).abs() < 1.0e-12,
        "appended lg_mtransfer_rate in LOGS1/history.data must match binary_history.data"
    );
}
