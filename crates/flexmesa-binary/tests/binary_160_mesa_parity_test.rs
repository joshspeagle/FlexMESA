// SPDX-License-Identifier: GPL-3.0-or-later
//! MESA `26.4.1` coupled parity pins for `#160` (`jdot_gr_check`,
//! `star_plus_point_mass`, and `star_plus_point_mass_explicit_mdot`).

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use flexmesa_binary::{
    binary_history_row_as_f64_pairs, default_binary_history_specs, eval_rlobe,
    evaluate_binary_history_row, BinaryInlistControls, BinarySystem,
    DefaultBinaryExtras,
};
use flexmesa_const::SECYER;
use flexmesa_math::log10;
use flexmesa_star::{
    eval_photosphere_info, init_or_update_model_from_bundle, load_zams_model,
    set_hydro_vars, DefaultStarExtras, InlistBundle, OvershootParamSet,
    StarControls, StarJobControls,
};

fn build_jdot_gr_check_system() -> BinarySystem {
    let d_ctrl = StarControls {
        initial_mass: 0.5,
        zbase: 0.02,
        max_age: 1.4e10,
        use_gold2_tolerances: true,
        ..StarControls::default()
    };
    let donor = load_zams_model(d_ctrl, StarJobControls::default())
        .expect("0.5 Msun ZAMS");
    let mut bctrl = BinaryInlistControls {
        m1_msun: 0.5,
        m2_msun: 0.8,
        initial_period_in_days: -1.0,
        initial_separation_in_rsuns: 2.0,
        mdot_scheme: String::from("roche_lobe"),
        max_tries_to_achieve: 20,
        do_jdot_mb: false,
        fj: 0.001,
        ..BinaryInlistControls::default()
    };
    bctrl.use_radiation_corrected_transfer_rate = false;
    BinarySystem::from_controls_point_mass(donor, bctrl)
        .expect("jdot_gr_check BinarySystem")
}

fn build_star_plus_point_mass_system(
    max_tries_to_achieve: i64,
) -> BinarySystem {
    let d_ctrl = StarControls {
        initial_mass: 5.0,
        zbase: 0.02,
        max_model_number: Some(1000),
        max_number_retries: 80,
        use_gold2_tolerances: true,
        varcontrol_target: 1.0e-3,
        use_ledoux_criterion: true,
        alpha_semiconvection: 1.0,
        ..StarControls::default()
    };
    let donor = load_zams_model(d_ctrl, StarJobControls::default())
        .expect("5.0 Msun ZAMS");
    let mut bctrl = BinaryInlistControls {
        m1_msun: 5.0,
        m2_msun: 10.0,
        initial_period_in_days: 1.0,
        fr: 0.05,
        fr_limit: 1.0e-2,
        limit_retention_by_mdot_edd: true,
        max_tries_to_achieve,
        ..BinaryInlistControls::default()
    };
    bctrl.use_radiation_corrected_transfer_rate = false;
    BinarySystem::from_controls_point_mass(donor, bctrl)
        .expect("star_plus_point_mass BinarySystem")
}

fn advance_to_model(sys: &mut BinarySystem, target_model: usize) {
    let mut extras = DefaultBinaryExtras;
    let mut dt = sys.donor.dt_next;
    while sys.model_number < target_model {
        let (next_dt, _) = sys
            .try_step_with_extras(dt, &mut extras)
            .expect("binary step succeeds");
        dt = next_dt;
    }
    assert_eq!(sys.model_number, target_model);
}

fn row_map(sys: &BinarySystem) -> BTreeMap<String, f64> {
    let specs = default_binary_history_specs();
    let row = evaluate_binary_history_row(sys, &specs);
    binary_history_row_as_f64_pairs(&row).into_iter().collect()
}

fn assert_rel_close(actual: f64, expected: f64, rel_tol: f64, label: &str) {
    let denom = expected.abs().max(1.0e-99);
    let rel = (actual - expected).abs() / denom;
    assert!(
        rel <= rel_tol,
        "{label}: actual={actual:.16e}, expected={expected:.16e}, \
         rel={rel:.3e} > tol={rel_tol:.3e}"
    );
}

fn assert_abs_close(actual: f64, expected: f64, abs_tol: f64, label: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= abs_tol,
        "{label}: actual={actual:.16e}, expected={expected:.16e}, \
         diff={diff:.3e} > tol={abs_tol:.3e}"
    );
}

#[test]
fn jdot_gr_check_initial_dt_and_model5_match_mesa_26_4_1() {
    let mut sys = build_jdot_gr_check_system();
    // MESA `26.4.1` `star/private/star_utils.f90:2900-2907` &
    // `jdot_gr_check.stepdump` model-1 CALL (`0x4286F60FA1600000`).
    assert_eq!(sys.donor.dt_next, 1.0e5 * SECYER);
    assert_eq!(sys.donor.dt_next.to_bits(), 0x4286_F60F_A160_0000);

    advance_to_model(&mut sys, 5);
    assert_eq!(sys.donor.nz, 790);

    let cols = row_map(&sys);
    assert_rel_close(cols["age"], 7.4416e5, 1.0e-12, "jdot_gr m5 age");
    assert_rel_close(
        cols["period_days"],
        2.873_362_413_960_604_5e-1,
        1.0e-11,
        "jdot_gr m5 period_days",
    );
    assert_rel_close(
        cols["binary_separation"],
        1.999_919_488_726_777_7,
        1.0e-11,
        "jdot_gr m5 binary_separation",
    );
    assert_rel_close(
        cols["rl_1"],
        6.780_581_886_470_01e-1,
        1.0e-11,
        "jdot_gr m5 rl_1",
    );
    assert_rel_close(
        cols["rl_2"],
        8.403_047_374_065_568e-1,
        1.0e-11,
        "jdot_gr m5 rl_2",
    );
    assert_abs_close(
        cols["rl_relative_overflow_1"],
        -3.430_032_941_570_881_3e-1,
        1.0e-10,
        "jdot_gr m5 rl_relative_overflow_1",
    );
    assert_rel_close(
        cols["J_orb"],
        2.997_555_966_815_422_6e51,
        1.0e-11,
        "jdot_gr m5 J_orb",
    );
    assert_rel_close(
        cols["jdot_gr"],
        -2.569_371_427_414_080_7e33,
        1.0e-10,
        "jdot_gr m5 jdot_gr",
    );

    let phot = eval_photosphere_info(&sys.donor);
    assert_abs_close(
        log10(phot.r_rsun),
        -3.511_698_428_298_629e-1,
        1.0e-8,
        "jdot_gr m5 log_R",
    );
    assert_abs_close(
        log10(phot.l_lsun),
        -1.404_470_030_416_498_3,
        1.0e-8,
        "jdot_gr m5 log_L",
    );
    assert_abs_close(
        log10(sys.donor.teff),
        3.585_793_994_243_055_2,
        1.0e-8,
        "jdot_gr m5 log_Teff",
    );
}

#[test]
#[ignore = "leaves the MESA 26.4.1 pin after #518's upstream Brunt face arithmetic; tracked in #160 (#530)"]
fn star_plus_point_mass_model5_and_model115_match_mesa_26_4_1() {
    let mut sys = build_star_plus_point_mass_system(20);
    advance_to_model(&mut sys, 5);
    let m5 = row_map(&sys);
    assert_rel_close(
        m5["age"],
        1.331_193_876_908_994_6e4,
        1.0e-11,
        "spm m5 age",
    );
    assert_rel_close(
        m5["period_days"],
        9.999_978_516_146_46e-1,
        1.0e-11,
        "spm m5 period_days",
    );
    assert_rel_close(
        m5["J_orb"],
        2.512_836_497_924_668e53,
        1.0e-11,
        "spm m5 J_orb",
    );
    assert_rel_close(
        m5["jdot_gr"],
        -4.283_622_744_690_115e35,
        1.0e-11,
        "spm m5 jdot_gr",
    );

    // Model 115 is 35 steps into active implicit RLOF.
    advance_to_model(&mut sys, 115);
    let m115 = row_map(&sys);
    assert_rel_close(
        m115["age"],
        3.838_849_143_529_478_5e7,
        1.0e-9,
        "spm m115 age",
    );
    assert_rel_close(
        m115["period_days"],
        9.999_347_222_446_818e-1,
        1.0e-9,
        "spm m115 period_days",
    );
    assert_rel_close(
        m115["binary_separation"],
        1.037_805_520_662_648_3e1,
        1.0e-9,
        "spm m115 binary_separation",
    );
    assert_rel_close(
        m115["star_1_mass"],
        4.979_505_257_805_243,
        1.0e-9,
        "spm m115 star_1_mass",
    );
    assert_rel_close(
        m115["star_2_mass"],
        1.002_049_474_219_476e1,
        1.0e-9,
        "spm m115 star_2_mass",
    );
    assert_abs_close(
        m115["rl_relative_overflow_1"],
        1.810_766_337_720_462_4e-3,
        1.0e-9,
        "spm m115 rl_relative_overflow_1",
    );
    assert_abs_close(
        m115["lg_mtransfer_rate"],
        -7.488_485_785_440_727,
        1.0e-9,
        "spm m115 lg_mtransfer_rate",
    );
    assert_rel_close(
        m115["J_orb"],
        2.507_612_624_410_114e53,
        1.0e-9,
        "spm m115 J_orb",
    );
}

#[test]
#[ignore = "leaves the MESA 26.4.1 pin after #518's upstream Brunt face arithmetic; tracked in #160 (#530)"]
fn star_plus_point_mass_explicit_mdot_model5_and_model110_match_mesa_26_4_1() {
    let mut sys = build_star_plus_point_mass_system(0);
    advance_to_model(&mut sys, 5);
    let m5 = row_map(&sys);
    assert_rel_close(
        m5["age"],
        1.331_193_876_908_994_6e4,
        1.0e-11,
        "exp m5 age",
    );
    assert_rel_close(
        m5["period_days"],
        9.999_978_516_146_46e-1,
        1.0e-11,
        "exp m5 period_days",
    );

    // Model 110 is 30 steps into explicit Ritter RLOF with `cur_mdot_frac = 0.5`.
    advance_to_model(&mut sys, 110);
    assert_eq!(sys.donor.nz, 947);
    let m110 = row_map(&sys);
    assert_rel_close(
        m110["age"],
        3.767_193_250_351_380_6e7,
        1.0e-9,
        "exp m110 age",
    );
    assert_rel_close(
        m110["period_days"],
        9.941_087_242_937_153e-1,
        1.0e-9,
        "exp m110 period_days",
    );
    assert_rel_close(
        m110["binary_separation"],
        1.033_770_497_675_283_2e1,
        1.0e-9,
        "exp m110 binary_separation",
    );
    assert_rel_close(
        m110["star_1_mass"],
        4.999_267_169_440_302,
        1.0e-9,
        "exp m110 star_1_mass",
    );
    assert_rel_close(
        m110["star_2_mass"],
        1.000_073_283_055_97e1,
        1.0e-9,
        "exp m110 star_2_mass",
    );
    assert_abs_close(
        m110["rl_relative_overflow_1"],
        9.759_693_607_434_231e-4,
        1.0e-9,
        "exp m110 rl_relative_overflow_1",
    );
    assert_abs_close(
        m110["lg_mtransfer_rate"],
        -8.351_605_960_023_617,
        1.0e-9,
        "exp m110 lg_mtransfer_rate",
    );
    assert_rel_close(
        m110["J_orb"],
        2.507_710_154_412_364_7e53,
        1.0e-9,
        "exp m110 J_orb",
    );
}

fn build_wind_fed_bhhmxb_system() -> BinarySystem {
    let mut d_ctrl = StarControls {
        initial_mass: 20.0,
        zbase: 0.02,
        max_number_retries: 15,
        max_model_number: Some(2000),
        use_gold2_tolerances: true,
        varcontrol_target: 1.0e-3,
        use_ledoux_criterion: true,
        alpha_semiconvection: 1.0,
        hot_wind_full_on_t: 1.0,
        cool_wind_full_on_t: 0.0,
        dutch_scaling_factor: 20.0,
        xa_central_lower_limit_species: vec![String::from("h1")],
        xa_central_lower_limit: vec![1.0e-2],
        ..StarControls::default()
    };
    d_ctrl.mix_wind.hot_wind_scheme = String::from("Dutch");
    d_ctrl.mix_wind.overshoot_sets[0] = OvershootParamSet {
        scheme: String::from("step"),
        zone_type: String::from("burn_H"),
        zone_loc: String::from("core"),
        bdy_loc: String::from("top"),
        f: 0.345,
        f0: 0.01,
        ..OvershootParamSet::mesa_default()
    };
    let mut donor = load_zams_model(d_ctrl, StarJobControls::default())
        .expect("20.0 Msun ZAMS");
    donor.eos_config.use_free_eos = false;
    set_hydro_vars(&mut donor, false, false)
        .expect("refresh initial hydro vars with use_FreeEOS=false");
    let bctrl = BinaryInlistControls {
        m1_msun: 20.0,
        m2_msun: 10.0,
        initial_period_in_days: 1.5,
        initial_bh_spin: 2.0,
        mdot_scheme: String::from("Kolb"),
        use_radiation_corrected_transfer_rate: true,
        fr: 0.05,
        fr_limit: 1.0e-2,
        fj: 0.005,
        limit_retention_by_mdot_edd: true,
        do_wind_mass_transfer_1: true,
        wind_bh_beta_1: 0.125,
        do_jdot_mb: false,
        implicit_scheme_tolerance: 1.0e-8,
        implicit_scheme_tiny_factor: 0.0,
        max_tries_to_achieve: 20,
        ..BinaryInlistControls::default()
    };
    BinarySystem::from_controls_point_mass(donor, bctrl)
        .expect("wind_fed_bhhmxb BinarySystem")
}

fn assert_wind_fed_snapshot(
    row: &BTreeMap<String, f64>,
    tag: &str,
    rel_pins: &[(&str, f64, f64)],
    abs_pins: &[(&str, f64, f64)],
) {
    for &(col, expected, tol) in rel_pins {
        assert_rel_close(row[col], expected, tol, &format!("{tag} {col}"));
    }
    for &(col, expected, tol) in abs_pins {
        assert_abs_close(row[col], expected, tol, &format!("{tag} {col}"));
    }
}

#[test]
#[ignore = "jdot_ml leaves the MESA 26.4.1 pin after the upstream-order energy-equation partials (#536); tracked in #160 (#538)"]
fn wind_fed_bhhmxb_model5_and_model10_match_mesa_26_4_1() {
    let mut sys = build_wind_fed_bhhmxb_system();
    advance_to_model(&mut sys, 5);
    assert_wind_fed_snapshot(
        &row_map(&sys),
        "wind m5",
        &[
            ("age", 4.159_980_865_340_611e2, 1.0e-11),
            ("period_days", 1.500_020_615_489_601_2, 1.0e-11),
            ("binary_separation", 1.713_464_779_579_958_7e1, 1.0e-11),
            ("star_1_mass", 1.999_974_617_055_151_4e1, 1.0e-11),
            ("star_2_mass", 1.000_000_951_427_948_5e1, 1.0e-11),
            ("J_orb", 9.132_219_846_940_914e53, 1.0e-11),
            ("jdot_ml", -2.916_398_041_673_092_6e38, 1.0e-8),
        ],
        &[
            ("lg_mstar_dot_1", -6.237_622_613_229_801, 1.0e-8),
            ("lg_mstar_dot_2", -7.661_647_263_061_589, 1.0e-8),
            ("lg_system_mdot_2", -7.797_106_038_996_278_5, 1.0e-8),
        ],
    );
    assert!(sys.last_mdot.wind_xfer_fraction[0] > 0.0);
    assert!(sys.last_mdot.mdot_edd > 0.0);
    let max_ret = sys.last_mdot.mdot_edd * (1.0 - sys.last_mdot.mdot_edd_eta);
    assert!(sys.last_mdot.retained_accretion_rate <= max_ret * (1.0 + 1.0e-12));

    advance_to_model(&mut sys, 10);
    assert_wind_fed_snapshot(
        &row_map(&sys),
        "wind m10",
        &[
            ("age", 1.451_134_445_218_496e3, 1.0e-11),
            ("period_days", 1.500_064_073_632_323_2, 1.0e-11),
            ("binary_separation", 1.713_488_028_645_593e1, 1.0e-11),
            ("star_1_mass", 1.999_920_856_350_514e1, 1.0e-11),
            ("star_2_mass", 1.000_003_000_580_514e1, 1.0e-11),
            ("J_orb", 9.132_133_739_741_865e53, 1.0e-11),
            ("jdot_gr", -1.676_112_020_056_3e36, 1.0e-11),
            ("jdot_ml", -2.513_567_547_841_301_4e38, 1.0e-8),
        ],
        &[
            ("lg_mstar_dot_1", -6.302_521_845_529_249, 1.0e-8),
            ("lg_mstar_dot_2", -7.718_964_432_561_775, 1.0e-8),
            ("lg_system_mdot_2", -7.854_423_208_496_463, 1.0e-8),
        ],
    );
}

#[test]
fn implicit_star_limit_redo_preserves_solver_progress() {
    let mut d_ctrl = StarControls {
        initial_mass: 1.0,
        zbase: 0.02,
        max_number_retries: 10,
        use_gold2_tolerances: true,
        ..StarControls::default()
    };
    let dt_yr = 1.0e4;
    d_ctrl.ts_stop.when_to_stop_atol = 1.0e-2;
    d_ctrl.ts_stop.when_to_stop_rtol = 0.0;
    d_ctrl.ts_stop.redo_limit = 8;
    let mut donor = load_zams_model(d_ctrl, StarJobControls::default())
        .expect("1.0 Msun ZAMS");
    donor.controls.max_age = donor.star_age + 0.5 * dt_yr;
    let r_rsun = eval_photosphere_info(&donor).r_rsun;
    let a_init_rsun = r_rsun / (eval_rlobe(1.0, 0.8, 1.0) * 1.001);
    let mut bctrl = BinaryInlistControls {
        m1_msun: 1.0,
        m2_msun: 0.8,
        initial_period_in_days: -1.0,
        initial_separation_in_rsuns: a_init_rsun,
        mdot_scheme: String::from("roche_lobe"),
        max_tries_to_achieve: 30,
        implicit_scheme_tolerance: 1.0e-4,
        starting_mdot: 1.0e-10,
        terminate_if_initial_overflow: false,
        ..BinaryInlistControls::default()
    };
    bctrl.use_radiation_corrected_transfer_rate = false;
    let mut sys = BinarySystem::from_controls_point_mass(donor, bctrl)
        .expect("gently overflowing ZAMS BinarySystem");
    assert!(
        sys.rl_relative_gap[0] > 5.0e-4,
        "expected gentle initial overflow"
    );

    let mut extras = DefaultBinaryExtras;
    let (_, _) = sys
        .try_step_with_extras(dt_yr * SECYER, &mut extras)
        .expect("star limit redo inside implicit RLOF should converge");
    assert_eq!(sys.model_number, 1);
    assert_eq!(
        sys.last_termination_reason.as_deref(),
        Some("max_age"),
        "step should hit max_age after star-limit redo"
    );
    assert_eq!(
        sys.counters.num_retries, 0,
        "star-limit redo must stay inside step_loop without outer retry"
    );
    assert!(
        sys.counters.num_implicit_trials > 1,
        "implicit solver must preserve trial progress across star-limit redo"
    );
    assert!(
        sys.mtransfer_rate < 0.0,
        "implicit RLOF mass transfer should converge"
    );
}

fn build_jdot_ls_check_system() -> BinarySystem {
    const INLIST1: &str = "\
&star_job
   change_rotation_flag = .true.
   new_rotation_flag = .true.
   new_omega_div_omega_crit = 0.2d0
   set_initial_omega_div_omega_crit = .true.
   set_initial_dt = .true.
   years_for_initial_dt = 1d0
/
&kap
   Zbase = 0.02d0
/
&controls
   max_number_retries = 8
   max_age = 1d10
   use_gold2_tolerances = .true.
   varcontrol_target = 1d-3
/
";
    let tmp_dir = std::env::temp_dir()
        .join(format!("flexmesa_160_jdot_ls_check_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).expect("create temp inlist dir");
    std::fs::write(tmp_dir.join("inlist1"), INLIST1).expect("write inlist1");
    let mut bundle = InlistBundle::from_file_in_case_dir(&tmp_dir, "inlist1")
        .expect("parse jdot_ls_check inlist1");
    let _ = std::fs::remove_dir_all(&tmp_dir);
    bundle.controls.initial_mass = 1.0;
    let donor =
        init_or_update_model_from_bundle(&bundle, None, &mut DefaultStarExtras)
            .expect("1.0 Msun jdot_ls_check initial star");
    let bctrl = BinaryInlistControls {
        m1_msun: 1.0,
        m2_msun: 0.8,
        initial_period_in_days: 1.0,
        do_tidal_sync: true,
        do_initial_orbit_sync_1: true,
        do_jdot_mb: false,
        ..BinaryInlistControls::default()
    };
    BinarySystem::from_controls_point_mass(donor, bctrl)
        .expect("jdot_ls_check BinarySystem")
}

#[test]
fn jdot_ls_check_model5_and_model100_match_mesa_26_4_1() {
    let mut sys = build_jdot_ls_check_system();
    assert!(sys.donor.aux.binary_other_torque.is_some());

    // Step 1 (`doing_first_model_of_run = true`): records t_sync while leaving
    // `jdot_ls == 0.0` (`binary/private/binary_tides.f90:256-269`).
    advance_to_model(&mut sys, 1);
    let m1 = row_map(&sys);
    assert_eq!(m1["jdot_ls"], 0.0);
    assert!(sys.t_sync[0] > 0.0);

    // MESA 26.4.1 `jdot_ls_check/binary_history.data` at models 5 and 100.
    advance_to_model(&mut sys, 5);
    let m5 = row_map(&sys);
    assert_rel_close(m5["age"], 7.4416, 1.0e-11, "jdot_ls m5 age");
    assert_rel_close(
        m5["period_days"],
        9.999_999_999_612_21e-1,
        1.0e-11,
        "jdot_ls m5 period_days",
    );
    assert_rel_close(
        m5["binary_separation"],
        5.119_119_790_454_294,
        1.0e-11,
        "jdot_ls m5 binary_separation",
    );
    assert_rel_close(
        m5["J_orb"],
        8.151_248_057_093_91e51,
        1.0e-11,
        "jdot_ls m5 J_orb",
    );
    assert_rel_close(
        m5["jdot_gr"],
        -4.507_427_812_009_082e32,
        1.0e-11,
        "jdot_ls m5 jdot_gr",
    );

    advance_to_model(&mut sys, 100);
    let m100 = row_map(&sys);
    assert_rel_close(
        m100["age"],
        4.140_898_676_100_715_4e8,
        1.0e-9,
        "jdot_ls m100 age",
    );
    assert_rel_close(
        m100["period_days"],
        9.978_057_000_049_148e-1,
        1.0e-9,
        "jdot_ls m100 period_days",
    );
    assert_rel_close(
        m100["binary_separation"],
        5.111_628_459_522_707_5,
        1.0e-9,
        "jdot_ls m100 binary_separation",
    );
    assert_rel_close(
        m100["J_orb"],
        8.145_281_596_476_726e51,
        1.0e-9,
        "jdot_ls m100 J_orb",
    );
    assert_rel_close(
        m100["jdot_ls"],
        -1.514_797_922_419_692e31,
        1.0e-9,
        "jdot_ls m100 jdot_ls",
    );
}

#[test]
fn tidal_deformation_attaches_before_initial_orbit_sync_and_premix_runs() {
    let make_star = |mass_msun: f64| {
        let mut ctrl = StarControls {
            initial_mass: mass_msun,
            zbase: 0.02,
            use_gold2_tolerances: true,
            ..StarControls::default()
        };
        ctrl.mix_wind.premix_omega = true;
        let mut s = load_zams_model(ctrl, StarJobControls::default())
            .expect("ZAMS model");
        s.controls.rotation_flag = true;
        s.dt = 2.0 * SECYER;
        s.dt_next = 2.0 * SECYER;
        s
    };
    let s1 = make_star(5.0);
    let s2 = make_star(4.0);
    let bctrl = BinaryInlistControls {
        evolve_both_stars: true,
        m1_msun: 5.0,
        m2_msun: 4.0,
        initial_period_in_days: 1.2,
        use_tidal_deformation: true,
        do_tidal_sync: true,
        do_initial_orbit_sync_1: true,
        do_initial_orbit_sync_2: true,
        sync_type_1: String::from("Hut_rad"),
        sync_type_2: String::from("Hut_rad"),
        sync_mode_1: String::from("Uniform"),
        sync_mode_2: String::from("Uniform"),
        do_jdot_mb: false,
        ..BinaryInlistControls::default()
    };
    let mut sys = BinarySystem::from_controls_two_stars(s1, s2, bctrl)
        .expect("two-star tidal binary");
    assert!(sys.donor.aux.tidal_deformation.is_some());
    assert!(sys.donor.aux.binary_other_torque.is_some());
    assert!(sys
        .star_ref(2)
        .expect("star 2")
        .aux
        .tidal_deformation
        .is_some());

    // Model 1 skips extra_jdot but populates t_sync for both stars, and leaves
    // start-of-step thermal variables (ln_t_start) distinct from post-solve ln_t.
    let j_cons_0 = sys.total_angular_momentum_conserved();
    advance_to_model(&mut sys, 1);
    let m1 = row_map(&sys);
    assert_eq!(m1["jdot_ls"], 0.0);
    assert!(sys.t_sync[0] > 0.0 && sys.t_sync[1] > 0.0);
    assert!(
        sys.donor.ln_t_start[0].to_bits() != sys.donor.ln_t[0].to_bits()
            || sys.donor.ln_t_start[sys.donor.nz / 2].to_bits()
                != sys.donor.ln_t[sys.donor.nz / 2].to_bits(),
        "post-step finalize override must not overwrite ln_t_start"
    );

    // Model 2 runs `sync_spin_orbit_torque` inside premix + post-solve
    // `do_solve_omega_mix` and conserves total angular momentum (J_orb + S_1 + S_2).
    advance_to_model(&mut sys, 2);
    let m2 = row_map(&sys);
    assert!(m2["jdot_ls"].is_finite() && m2["jdot_ls"] != 0.0);
    let j_cons_2 = sys.total_angular_momentum_conserved();
    let rel_cons = (j_cons_2 - j_cons_0).abs() / j_cons_0;
    assert!(
        rel_cons < 1.0e-12,
        "spin-orbit total AM conservation violated: rel={rel_cons:.3e}"
    );
}
