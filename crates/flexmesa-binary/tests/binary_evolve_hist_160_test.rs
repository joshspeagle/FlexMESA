// SPDX-License-Identifier: GPL-3.0-or-later
//! Issue `#160` unit and integration tests for `flexmesa-binary` inlist
//! parsing, binary history evaluation/appending, `BinarySystem` job overrides,
//! and `e2e` step-loop lifecycle hooks (without external MESA data files).

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_binary::{
    default_binary_history_specs, eval_rl_relative_gap,
    evaluate_binary_history_row, format_binary_history_data_line,
    parse_binary_history_columns, resolve_binary_history_columns,
    run_binary_test_suite_case_with_extras, BinaryCompanion, BinaryExtras,
    BinaryExtrasAction, BinaryInlistControls, BinaryLogsOptions,
    BinaryLogsWriter, BinarySystem, BINARY_LOG_COL_WIDTH,
    DEFAULT_BINARY_HISTORY_COLUMNS_TEXT,
};
use flexmesa_const::{MSUN, RSUN, SECDAY, SECYER};
use flexmesa_star::{
    eval_photosphere_info, format_mesa_dbl_width, load_zams_model,
    read_mesa_log_file, safe_log10, InlistLoadOptions, MesaCellValue,
    OutputControls, StarControls, StarError, StarJobControls, StarModel,
};

struct TempTestDir {
    root: PathBuf,
}

impl TempTestDir {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir()
            .join(format!("flexmesa_bin160_{tag}_{}_{id}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp test dir");
        Self { root }
    }

    fn path(&self) -> &Path {
        &self.root
    }
}

impl Drop for TempTestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn test_sparse_and_out_of_bounds_inlist_names_and_separation_validation() {
    let tmp = TempTestDir::new("inlist_names");

    let sparse_dir = tmp.path().join("sparse");
    fs::create_dir_all(&sparse_dir).expect("mkdir sparse");
    fs::write(
        sparse_dir.join("inlist"),
        "&binary_job\n  inlist_names(2) = 'custom_star2'\n/\n",
    )
    .expect("write sparse inlist");
    let sparse =
        BinaryInlistControls::from_case_dir(&sparse_dir).expect("parse sparse");
    assert_eq!(sparse.inlist_names[0], "inlist1");
    assert_eq!(sparse.inlist_names[1], "custom_star2");

    let scalar_dir = tmp.path().join("scalar");
    fs::create_dir_all(&scalar_dir).expect("mkdir scalar");
    fs::write(
        scalar_dir.join("inlist"),
        "&binary_job\n  inlist_names = 'scalar_star1'\n/\n",
    )
    .expect("write scalar inlist");
    let scalar =
        BinaryInlistControls::from_case_dir(&scalar_dir).expect("parse scalar");
    assert_eq!(scalar.inlist_names[0], "scalar_star1");
    assert_eq!(scalar.inlist_names[1], "inlist2");

    for (tag, body) in [
        ("oob_3", "&binary_job\n  inlist_names(3) = 'bad'\n/\n"),
        ("oob_0", "&binary_job\n  inlist_names(0) = 'bad'\n/\n"),
        (
            "bad_sep",
            "&binary_controls\n  initial_period_in_days = -1.0d0\n  initial_separation_in_rsuns = 0.0d0\n/\n",
        ),
    ] {
        let dir = tmp.path().join(tag);
        fs::create_dir_all(&dir).expect("mkdir bad");
        fs::write(dir.join("inlist"), body).expect("write bad");
        let err = BinaryInlistControls::from_case_dir(&dir)
            .expect_err("invalid subscript or non-positive orbit must fail");
        assert!(
            matches!(err, StarError::InvalidInput(_)),
            "expected InvalidInput for {tag}, got {err:?}"
        );
    }
}

#[test]
fn test_binary_history_model0_tsync_zero_and_nonrotating_p_rot_div_p_orb() {
    let c1 = StarControls {
        initial_mass: 1.5,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut c2 = c1.clone();
    c2.initial_mass = 1.0;

    let s1 = load_zams_model(c1, StarJobControls::default()).expect("s1");
    let s2 = load_zams_model(c2, StarJobControls::default()).expect("s2");

    let mut sys_two = BinarySystem::new_two_stars(s1.clone(), s2, 3.0, 0.25);
    sys_two.controls.do_tidal_sync = true;
    sys_two.controls.sync_type_1 = String::from("Hut_conv");
    sys_two.controls.sync_type_2 = String::from("Hut_conv");

    let specs = parse_binary_history_columns(
        "lg_t_sync_1\nlg_t_sync_2\nP_rot_div_P_orb_1\nP_rot_div_P_orb_2\n\
         rl_relative_overflow_1\nrl_relative_overflow_2\n",
        None,
    )
    .expect("parse specs");
    let row = evaluate_binary_history_row(&sys_two, &specs);
    let get = |name: &str| -> f64 {
        row.iter()
            .find_map(|(k, v)| match *v {
                MesaCellValue::Float(x) if k == name => Some(x),
                _ => None,
            })
            .unwrap_or_else(|| panic!("missing float column {name}"))
    };

    // Upstream `binary/private/binary_history.f90:575-578` and
    // `binary/private/run_binary_support.f90:449-453`:
    // `t_sync_{1,2}` equals `0.0` at model 0 (`safe_log10(0.0) == -99.0`).
    let zero_log = safe_log10(0.0);
    assert_eq!(get("lg_t_sync_1").to_bits(), zero_log.to_bits());
    assert_eq!(get("lg_t_sync_2").to_bits(), zero_log.to_bits());

    // Upstream `binary/private/binary_history.f90:559-574, 434`: non-rotating
    // evolved stars yield `+Inf` (`2*pi/0/P`), formatted to `-1e99`.
    assert!(get("P_rot_div_P_orb_1").is_infinite());
    assert!(get("P_rot_div_P_orb_2").is_infinite());
    let formatted = format_binary_history_data_line(&row);
    let tokens: Vec<&str> = formatted.split_whitespace().collect();
    let expected_inf =
        format_mesa_dbl_width(f64::INFINITY, BINARY_LOG_COL_WIDTH);
    assert_eq!(tokens[2], expected_inf.trim());
    assert_eq!(tokens[3], expected_inf.trim());
    assert_eq!(
        tokens[2].parse::<f64>().expect("parse inf").to_bits(),
        (-1.0e99_f64).to_bits()
    );
    assert_eq!(
        tokens[3].parse::<f64>().expect("parse inf").to_bits(),
        (-1.0e99_f64).to_bits()
    );

    // Bit-identical two-division `eval_rl_relative_gap`
    // (`binary/private/binary_evolve.f90:367-370`).
    let r1 = eval_photosphere_info(&sys_two.donor).r_rsun * RSUN;
    let r2 = match &sys_two.companion {
        BinaryCompanion::Star(st2) => eval_photosphere_info(st2).r_rsun * RSUN,
        BinaryCompanion::PointMass { .. } => 0.0,
    };
    let rl = sys_two.rl_by_star();
    assert_eq!(
        get("rl_relative_overflow_1").to_bits(),
        eval_rl_relative_gap(r1, rl[0], sys_two.eccentricity).to_bits()
    );
    assert_eq!(
        get("rl_relative_overflow_2").to_bits(),
        eval_rl_relative_gap(r2, rl[1], sys_two.eccentricity).to_bits()
    );

    // Point-mass companion (`point_mass_i = 2`) returns exact `0.0`
    // (`binary/private/binary_history.f90:570`).
    let sys_pm = BinarySystem::new_point_mass(s1, 1.4, 3.0, 0.0, false, 0.0);
    let row_pm = evaluate_binary_history_row(&sys_pm, &specs);
    let p_rot_2_pm = row_pm
        .iter()
        .find_map(|(k, v)| match *v {
            MesaCellValue::Float(x) if k == "P_rot_div_P_orb_2" => Some(x),
            _ => None,
        })
        .expect("P_rot_div_P_orb_2");
    assert_eq!(p_rot_2_pm, 0.0);
}

fn load_test_zams_star(initial_mass: f64) -> StarModel {
    load_zams_model(
        StarControls {
            initial_mass,
            initial_z: 0.02,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("ZAMS model")
}

#[test]
fn test_binary_history_local_include_resolution() {
    let tmp = TempTestDir::new("hist_order");
    let local_dir = tmp.path().join("local");
    let clean_dir = tmp.path().join("clean");
    fs::create_dir_all(&local_dir).expect("mkdir local");
    fs::create_dir_all(&clean_dir).expect("mkdir clean");
    fs::write(
        local_dir.join("binary_history_columns.list"),
        "model_number\neccentricity\n",
    )
    .expect("write local binary_history_columns.list");

    // `binary/private/binary_history_specs.f90:65-78`: both explicit
    // `'binary_history_columns.list'` and empty `''` check the local directory
    // first before falling back to `binary/defaults/`.
    let inc_local = parse_binary_history_columns(
        "include 'binary_history_columns.list'\nage\n",
        Some(&local_dir),
    )
    .expect("include local binary_history_columns.list");
    let local_names: Vec<&str> =
        inc_local.iter().map(|c| c.column_name()).collect();
    assert_eq!(local_names, vec!["model_number", "eccentricity", "age"]);

    let inc_empty_local = parse_binary_history_columns(
        "include ''\nage\n",
        Some(&local_dir),
    )
    .expect(
        "include empty string checks local binary_history_columns.list first",
    );
    let empty_local_names: Vec<&str> =
        inc_empty_local.iter().map(|c| c.column_name()).collect();
    assert_eq!(
        empty_local_names,
        vec!["model_number", "eccentricity", "age"]
    );

    let inc_default =
        parse_binary_history_columns("include ''\n", Some(&clean_dir)).expect(
            "include empty string falls back to default when absent locally",
        );
    let default_specs =
        parse_binary_history_columns(DEFAULT_BINARY_HISTORY_COLUMNS_TEXT, None)
            .expect("embedded default");
    assert_eq!(inc_default, default_specs);
    let inc_named_default = parse_binary_history_columns(
        "include 'binary_history_columns.list'\n",
        Some(&clean_dir),
    )
    .expect("named default list falls back when absent locally");
    assert_eq!(inc_named_default, default_specs);

    let mut s1 = load_test_zams_star(1.5);
    s1.controls.output = OutputControls {
        log_directory: String::from("LOGS1"),
        history_interval: 1,
        write_profiles_flag: false,
        ..OutputControls::default()
    };
    let sys = BinarySystem::new_point_mass(s1, 1.4, 2.0, 0.0, false, 0.0);
    let resolved_local = resolve_binary_history_columns(&sys, Some(&local_dir))
        .expect("resolve local binary_history_columns.list");
    let resolved_local_names: Vec<&str> =
        resolved_local.iter().map(|c| c.column_name()).collect();
    assert_eq!(resolved_local_names, vec!["model_number", "eccentricity"]);
    let resolved_clean = resolve_binary_history_columns(&sys, Some(&clean_dir))
        .expect("resolve fallback default binary_history_columns.list");
    assert_eq!(resolved_clean, default_specs);
}

#[test]
fn test_binary_history_star_column_order_and_append() {
    let tmp = TempTestDir::new("hist_append");
    // Verify `BinaryLogsWriter::record_step_with_extra` emits star extra
    // columns BEFORE binary columns, and appends `extra_bin_cols` AFTER them
    // when `append_to_star_history = true` (`star/private/history.f90:472-480`).
    let mut s1 = load_test_zams_star(1.5);
    s1.controls.output = OutputControls {
        log_directory: String::from("LOGS1"),
        history_interval: 1,
        write_profiles_flag: false,
        ..OutputControls::default()
    };
    let mut sys = BinarySystem::new_point_mass(s1, 1.4, 2.0, 0.0, false, 0.0);
    sys.controls.history_interval = 5;
    sys.controls.append_to_star_history = true;
    sys.model_number = 1;
    sys.donor.model_number = 1;

    let out_dir = tmp.path().join("out");
    let case_dir = tmp.path().join("case");
    fs::create_dir_all(&case_dir).expect("mkdir case");
    let mut writer =
        BinaryLogsWriter::for_run(&sys, &case_dir, &out_dir, None, None, true)
            .expect("BinaryLogsWriter");
    let extra_bin = vec![(String::from("bin_extra"), 42.5)];
    let extra_star = [vec![(String::from("star1_extra"), 7.25)], Vec::new()];
    writer
        .record_step_with_extra(&sys, true, false, &extra_bin, &extra_star)
        .expect("record step 1");

    assert!(
        !out_dir.join("binary_history.data").exists(),
        "step 1 with history_interval=5 must not emit binary_history.data"
    );
    let star_tbl =
        read_mesa_log_file(&out_dir.join("LOGS1").join("history.data"))
            .expect("read LOGS1/history.data");
    let n_bin = default_binary_history_specs().len();
    let idx_star_extra = star_tbl
        .column_names
        .iter()
        .position(|c| c == "star1_extra")
        .expect("star1_extra present");
    let idx_bin_extra = star_tbl
        .column_names
        .iter()
        .position(|c| c == "bin_extra")
        .expect("bin_extra appended when append_to_star_history=true");
    assert_eq!(star_tbl.column_names[idx_star_extra + 1], "model_number");
    assert_eq!(idx_bin_extra, idx_star_extra + 1 + n_bin);
    assert_eq!(star_tbl.column_values[idx_star_extra], vec![7.25]);
    assert_eq!(star_tbl.column_values[idx_bin_extra], vec![42.5]);
}

#[test]
fn test_system_ignore_rlof_and_eccentricity_no_early_clamp() {
    let s1 = load_test_zams_star(1.5);

    // `binary/private/binary_evolve.f90:106-118`: negative
    // `initial_eccentricity` is not rejected at `binarydata_init`, whereas
    // `initial_eccentricity = 1.0` triggers `is_bad(rl_relative_gap)`
    // (`binary/private/binary_evolve.f90:137-138`).
    let neg_ecc_ctrl = BinaryInlistControls {
        m1_msun: 1.5,
        m2_msun: 1.0,
        initial_period_in_days: 3.0,
        initial_eccentricity: -0.05,
        ..BinaryInlistControls::default()
    };
    let sys_neg =
        BinarySystem::from_controls_point_mass(s1.clone(), neg_ecc_ctrl)
            .expect("negative initial_eccentricity allowed in binarydata_init");
    assert!((sys_neg.eccentricity - (-0.05)).abs() < 1.0e-15);

    let bad_ecc_ctrl = BinaryInlistControls {
        m1_msun: 1.5,
        m2_msun: 1.0,
        initial_period_in_days: 3.0,
        initial_eccentricity: 1.0,
        ..BinaryInlistControls::default()
    };
    assert!(
        BinarySystem::from_controls_point_mass(s1.clone(), bad_ecc_ctrl)
            .is_err(),
        "eccentricity = 1.0 must fail via is_bad(rl_relative_gap)"
    );

    let ctrl = BinaryInlistControls {
        m1_msun: 1.5,
        m2_msun: 1.0,
        initial_period_in_days: 3.0,
        initial_eccentricity: 0.7,
        min_eccentricity: 0.1,
        max_eccentricity: 0.5,
        ..BinaryInlistControls::default()
    };
    let mut sys = BinarySystem::from_controls_point_mass(s1, ctrl)
        .expect("from_controls_point_mass");
    assert!((sys.eccentricity - 0.7).abs() < 1.0e-15);

    sys.mtransfer_rate = -1.0e-8 * MSUN / SECYER;
    sys.set_ignore_rlof_flag(true);
    assert!(sys.ignore_rlof_flag);
    assert!(sys.mtransfer_rate < 0.0);

    // `binary/private/binary_utils.f90:105-180`: setters perform no
    // non-upstream range rejection and only guard `point_mass_i` in
    // `set_m1`/`set_m2`.
    sys.set_period_eccentricity(4.0 * SECDAY, 0.8)
        .expect("set_period_eccentricity");
    assert!((sys.eccentricity - 0.8).abs() < 1.0e-15);
    sys.set_separation_eccentricity(8.0 * RSUN, 0.25)
        .expect("set_separation_eccentricity");
    assert!((sys.eccentricity - 0.25).abs() < 1.0e-15);
    sys.set_m2(1.4)
        .expect("set_m2 succeeds when point_mass_i == 2");
    assert!((sys.star_masses_g()[1] / MSUN - 1.4).abs() < 1.0e-12);
    assert!(sys.set_m1(1.2).is_err());
}

#[test]
fn test_system_binary_job_overrides_preserve_start_and_old() {
    let s1 = load_test_zams_star(1.5);
    let ctrl = BinaryInlistControls {
        m1_msun: 1.5,
        m2_msun: 1.0,
        initial_period_in_days: 3.0,
        initial_eccentricity: 0.7,
        ..BinaryInlistControls::default()
    };
    let mut sys = BinarySystem::from_controls_point_mass(s1.clone(), ctrl)
        .expect("from_controls_point_mass");

    // `binary/private/run_binary_support.f90:945,957,995-1012` &
    // `binary/private/binary_evolve.f90:501-570`: job overrides reset
    // `generations = 0` without mutating `*_start`/`*_old`, and `m(2) /= new_m2`
    // compares grams with `Msun` (`#565`).
    sys.generations = 2;
    sys.donor.generations = 2;
    let saved_mstar_start = sys.donor.mstar_start;
    sys.donor.aux.step_hist.teff_old = 12345.0;
    sys.controls.change_m2 = true;
    sys.controls.new_m2 = 1.8;
    sys.apply_binary_job_controls_after(false)
        .expect("apply_binary_job_controls_after");
    assert!(
        (sys.star_masses_g()[1] / MSUN - 1.8).abs() < 1.0e-12,
        "set_m2 must update point-mass m(2) to new_m2 * MSUN"
    );
    assert_eq!(sys.generations, 0);
    assert_eq!(sys.donor.generations, 0);
    assert_eq!(sys.donor.mstar_start, saved_mstar_start);
    assert_eq!(sys.donor.aux.step_hist.teff_old, 12345.0);

    // Even when `m(2) == 1.8 * MSUN`, `m(2)` (grams) != `1.8` (`Msun`), so
    // `apply_binary_job_controls_after` still fires and resets `generations = 0`
    // (`#565`).
    sys.generations = 2;
    sys.donor.generations = 2;
    sys.apply_binary_job_controls_after(false)
        .expect("grams vs Msun comparison triggers set_m2 again");
    assert_eq!(sys.generations, 0);
    assert_eq!(sys.donor.generations, 0);

    // Two-star `change_initial_ignore_rlof_flag` resets `generations = 0` on
    // both stars and binary while preserving stellar `*_start` and `*_old`.
    let s2 = load_test_zams_star(1.2);
    let mut sys_two = BinarySystem::new_two_stars(s1, s2, 3.0, 0.0);
    assert!(sys_two.set_m2(1.2).is_err());
    sys_two.generations = 2;
    sys_two.donor.generations = 2;
    sys_two.donor.aux.step_hist.teff_old = 54321.0;
    let s1_mstar_start = sys_two.donor.mstar_start;
    let s2_mstar_start = match &mut sys_two.companion {
        BinaryCompanion::Star(st2) => {
            st2.generations = 2;
            st2.aux.step_hist.teff_old = 67890.0;
            st2.mstar_start
        }
        BinaryCompanion::PointMass { .. } => unreachable!("expected two stars"),
    };
    sys_two.controls.change_initial_ignore_rlof_flag = true;
    sys_two.controls.new_ignore_rlof_flag = true;
    sys_two
        .apply_binary_job_controls_after(false)
        .expect("two-star ignore_rlof_flag override");
    assert!(sys_two.ignore_rlof_flag);
    assert_eq!(sys_two.generations, 0);
    assert_eq!(sys_two.donor.generations, 0);
    assert_eq!(sys_two.donor.mstar_start, s1_mstar_start);
    assert_eq!(sys_two.donor.aux.step_hist.teff_old, 54321.0);
    match &sys_two.companion {
        BinaryCompanion::Star(st2) => {
            assert_eq!(st2.generations, 0);
            assert_eq!(st2.mstar_start, s2_mstar_start);
            assert_eq!(st2.aux.step_hist.teff_old, 67890.0);
        }
        BinaryCompanion::PointMass { .. } => unreachable!("expected two stars"),
    }
}

struct LifecycleExtras {
    saw_ignore_rlof_at_startup: Option<bool>,
    saw_photo_interval_disabled: Option<bool>,
    after_evolve_called: bool,
}

impl BinaryExtras for LifecycleExtras {
    fn startup(
        &mut self,
        bin: &mut BinarySystem,
        _restart: bool,
    ) -> BinaryExtrasAction {
        self.saw_ignore_rlof_at_startup = Some(bin.ignore_rlof_flag);
        let photo_ok = bin.donor.controls.output.photo_interval == -1
            && !bin.donor.job.save_photo_when_terminate;
        self.saw_photo_interval_disabled = Some(photo_ok);
        BinaryExtrasAction::KeepGoing
    }

    fn after_evolve(
        &mut self,
        _bin: &mut BinarySystem,
    ) -> Result<(), StarError> {
        self.after_evolve_called = true;
        Ok(())
    }

    fn extra_binary_history_columns(
        &self,
        _bin: &BinarySystem,
    ) -> Vec<(String, f64)> {
        vec![(String::from("custom_bin"), 123.5)]
    }
}

fn write_lifecycle_fixture(case_dir: &Path) {
    fs::write(
        case_dir.join("inlist"),
        "&binary_job\n\
           evolve_both_stars = .true.\n\
           inlist_names(2) = 'star2_inlist'\n\
           change_initial_ignore_rlof_flag = .true.\n\
           new_ignore_rlof_flag = .true.\n\
         /\n\
         &binary_controls\n\
           m1 = 1.0d0\n\
           m2 = 0.8d0\n\
           initial_period_in_days = 2.0d0\n\
           history_interval = 5\n\
           append_to_star_history = .true.\n\
         /\n",
    )
    .expect("write binary inlist");
    fs::write(
        case_dir.join("history_columns.list"),
        "model_number\nstar_age\ncumulative_energy_error\n",
    )
    .expect("write star history_columns.list");
    fs::write(
        case_dir.join("inlist1"),
        "&star_job\n\
           set_cumulative_energy_error_at_step = 1\n\
           new_cumulative_energy_error = 1.25d30\n\
         /\n\
         &kap\n\
           zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           log_directory = 'LOGS1'\n\
           history_interval = 1\n\
           write_profiles_flag = .false.\n\
         /\n",
    )
    .expect("write inlist1");
    fs::write(
        case_dir.join("star2_inlist"),
        "&star_job\n\
           set_cumulative_energy_error_each_step_if_age_less_than = 1.0d99\n\
           new_cumulative_energy_error = -1.0d45\n\
         /\n\
         &kap\n\
           zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 0.8d0\n\
           log_directory = 'LOGS2'\n\
           history_interval = 1\n\
           write_profiles_flag = .false.\n\
         /\n",
    )
    .expect("write star2_inlist");
}

#[test]
fn test_e2e_startup_order_energy_resets_and_extra_bin() {
    let tmp = TempTestDir::new("e2e_lifecycle");
    let case_dir = tmp.path().join("case");
    let out_dir = tmp.path().join("out");
    fs::create_dir_all(&case_dir).expect("mkdir case");
    write_lifecycle_fixture(&case_dir);

    let mut extras = LifecycleExtras {
        saw_ignore_rlof_at_startup: None,
        saw_photo_interval_disabled: None,
        after_evolve_called: false,
    };
    let report = run_binary_test_suite_case_with_extras(
        &case_dir,
        Some(1),
        BinaryLogsOptions {
            output_root: Some(&out_dir),
            explicit_binary_history_file: None,
            history_interval_override: None,
            write_star_logs: true,
        },
        InlistLoadOptions::default(),
        &mut extras,
    )
    .expect("1-step coupled lifecycle run");

    assert_eq!(report.total_steps, 1);
    assert_eq!(
        extras.saw_ignore_rlof_at_startup,
        Some(false),
        "extras.startup must run before apply_binary_job_controls_after"
    );
    assert_eq!(extras.saw_photo_interval_disabled, Some(true));
    assert!(extras.after_evolve_called);

    let l1 = read_mesa_log_file(&out_dir.join("LOGS1").join("history.data"))
        .expect("read LOGS1/history.data");
    let cee1 = l1
        .column("cumulative_energy_error")
        .expect("cumulative_energy_error col");
    assert!((cee1[0] - 1.25e30).abs() / 1.25e30 < 1.0e-12);
    let custom1 = l1
        .column("custom_bin")
        .expect("extra_binary_history_columns appended on step 1");
    assert!((custom1[0] - 123.5).abs() < 1.0e-12);

    let l2 = read_mesa_log_file(&out_dir.join("LOGS2").join("history.data"))
        .expect("read LOGS2/history.data");
    let cee2 = l2
        .column("cumulative_energy_error")
        .expect("cumulative_energy_error col in LOGS2");
    assert!(cee2[0] < -1.0e44);
}

#[test]
fn test_e2e_save_model_filename_collision_fails_loudly() {
    // Colliding `save_model_filename` when both stars set
    // `save_model_when_terminate = .true.` must fail loudly
    // (`binary/private/run_binary_support.f90:285-294`).
    let tmp = TempTestDir::new("e2e_save_mod_collide");
    let collide_dir = tmp.path().join("collide_save_mod");
    fs::create_dir_all(&collide_dir).expect("mkdir collide");
    fs::write(
        collide_dir.join("inlist"),
        "&binary_job\n\
           evolve_both_stars = .true.\n\
         /\n\
         &binary_controls\n\
           m1 = 1.0d0\n\
           m2 = 0.8d0\n\
           initial_period_in_days = 2.0d0\n\
         /\n",
    )
    .expect("write collide inlist");
    for name in ["inlist1", "inlist2"] {
        fs::write(
            collide_dir.join(name),
            "&star_job\n\
               save_model_when_terminate = .true.\n\
               save_model_filename = 'final.mod'\n\
             /\n\
             &kap\n\
               zbase = 0.02d0\n\
             /\n\
             &controls\n\
               initial_mass = 1.0d0\n\
             /\n",
        )
        .expect("write colliding star inlist");
    }
    let mut noop_extras = LifecycleExtras {
        saw_ignore_rlof_at_startup: None,
        saw_photo_interval_disabled: None,
        after_evolve_called: false,
    };
    let err = run_binary_test_suite_case_with_extras(
        &collide_dir,
        Some(1),
        BinaryLogsOptions::default(),
        InlistLoadOptions::default(),
        &mut noop_extras,
    )
    .expect_err("colliding save_model_filename must fail loudly");
    assert!(matches!(err, StarError::InvalidInput(_)));
}
