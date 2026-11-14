// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stage `E2E-5` (`#114`) coupled `binary/test_suite` integration tests and
//! unit checks for decoupled Eddington retention vs. radiation-corrected
//! accretion, recursive 3-level binary inlist chaining,
//! `initial_separation_in_rsuns` orbital initialization, Brott et al. (2011)
//! winds, and dynamic `set_point_mass_i` conversion.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use flexmesa_binary::{
    BinaryCompanion, BinaryInlistControls, BinarySystem, TIER5_BINARY_CASES,
    bh_initial_equatorial_mass, eval_brott_wind_msun_yr,
    eval_point_mass_edd_and_retention, run_binary_suite,
    run_binary_test_suite_case,
};
use flexmesa_const::{CLIGHT, MSUN, RSUN, SECYER};
use flexmesa_star::{
    DefaultStarExtras, InlistBundle, StarControls, StarJobControls,
    init_or_update_model_from_bundle, load_zams_model,
};

fn resolve_suite_dir() -> Option<(PathBuf, PathBuf)> {
    let mesa_dir = PathBuf::from(std::env::var("MESA_DIR").ok()?);
    let suite_dir = mesa_dir.join("binary").join("test_suite");
    suite_dir.is_dir().then_some((mesa_dir, suite_dir))
}

#[test]
fn test_decoupled_edd_retention_limit_vs_radiation_correction() {
    let m_ns = 1.4 * MSUN;
    let m_eq = bh_initial_equatorial_mass(m_ns, 0.0);
    let mdot_tr = -1.0e-3 * MSUN / SECYER;

    let capped_no_rad = eval_point_mass_edd_and_retention(
        m_ns, m_eq, 0.70, 0.34, true, -1.0, -1.0, true, false, mdot_tr, 0.0,
        0.0, 0.0, 0.0,
    );
    assert!(capped_no_rad.mdot_edd > 0.0);
    assert!(
        (capped_no_rad.component_mdot_accretor - capped_no_rad.mdot_edd).abs()
            / capped_no_rad.mdot_edd
            < 1.0e-14
    );
    let expected_eject = mdot_tr + capped_no_rad.mdot_edd;
    assert!(
        (capped_no_rad.mdot_system_transfer_accretor - expected_eject).abs()
            / expected_eject.abs()
            < 1.0e-14
    );

    let capped_with_rad = eval_point_mass_edd_and_retention(
        m_ns, m_eq, 0.70, 0.34, true, -1.0, -1.0, true, true, mdot_tr, 0.0,
        0.0, 0.0, 0.0,
    );
    let expected_rad_acc =
        capped_with_rad.mdot_edd * (1.0 - capped_with_rad.mdot_edd_eta);
    assert!(
        (capped_with_rad.component_mdot_accretor - expected_rad_acc).abs()
            / expected_rad_acc
            < 1.0e-14
    );
    let rad_mass_loss =
        -capped_with_rad.accretion_luminosity / (CLIGHT * CLIGHT);
    assert!(
        (capped_with_rad.mdot_system_transfer_accretor
            - (expected_eject + rad_mass_loss))
            .abs()
            / expected_eject.abs()
            < 1.0e-14
    );

    let uncapped_no_rad = eval_point_mass_edd_and_retention(
        m_ns, m_eq, 0.70, 0.34, true, -1.0, -1.0, false, false, mdot_tr, 0.0,
        0.0, 0.0, 0.0,
    );
    assert!(
        (uncapped_no_rad.component_mdot_accretor - (-mdot_tr)).abs()
            / (-mdot_tr)
            < 1.0e-14
    );
}

#[test]
fn test_initial_separation_and_dynamic_point_mass_conversion() {
    let c1 = StarControls {
        initial_mass: 1.0,
        ..StarControls::default()
    };
    let s1 =
        load_zams_model(c1, StarJobControls::default()).expect("1 Msun ZAMS");
    let c2 = StarControls {
        initial_mass: 0.8,
        ..StarControls::default()
    };
    let s2 =
        load_zams_model(c2, StarJobControls::default()).expect("0.8 Msun ZAMS");

    let ctrl = BinaryInlistControls {
        evolve_both_stars: true,
        m1_msun: 1.0,
        m2_msun: 0.8,
        initial_period_in_days: -1.0,
        initial_separation_in_rsuns: 5.5,
        terminate_if_initial_overflow: false,
        ..BinaryInlistControls::default()
    };
    let mut bin = BinarySystem::from_controls_two_stars(s1, s2, ctrl)
        .expect("two-star system");
    assert!(((bin.separation / RSUN) - 5.5).abs() < 1.0e-12);
    assert!(bin.period > 0.0);
    assert_eq!(bin.point_mass_i, 0);

    bin.set_point_mass_i(1);
    assert_eq!(bin.point_mass_i, 1);
    assert_eq!(bin.donor_i, 2);
    assert!((bin.donor.mstar / MSUN - 0.8).abs() < 1.0e-6);
    assert!(matches!(bin.companion, BinaryCompanion::PointMass { .. }));
    assert!((bin.accretor_mass_g() / MSUN - 1.0).abs() < 1.0e-6);
}

#[test]
fn test_brott_2011_wind_regimes_are_positive_and_continuous() {
    let w_hot =
        eval_brott_wind_msun_yr(5.0e5, 60.0, 12.0, 42_000.0, 0.70, 0.299, 0.02);
    let w_bistab =
        eval_brott_wind_msun_yr(5.0e5, 60.0, 35.0, 23_000.0, 0.70, 0.299, 0.02);
    let w_wr =
        eval_brott_wind_msun_yr(6.0e5, 35.0, 3.0, 80_000.0, 0.05, 0.949, 0.02);
    let w_blend =
        eval_brott_wind_msun_yr(5.5e5, 45.0, 8.0, 50_000.0, 0.55, 0.449, 0.02);
    for w in [w_hot, w_bistab, w_wr, w_blend] {
        assert!(w > 0.0 && w.is_finite(), "invalid Brott wind rate {w}");
    }
    assert!(w_wr > w_hot);
}

#[test]
fn test_recursive_binary_inlists_and_all_8_cases_capped_4_steps() {
    let Some((mesa_dir, suite_dir)) = resolve_suite_dir() else {
        return;
    };

    let dbh_ctrl =
        BinaryInlistControls::from_case_dir(&suite_dir.join("double_bh"))
            .expect("double_bh recursive inlists");
    assert!(dbh_ctrl.evolve_both_stars);
    assert!(dbh_ctrl.new_ignore_rlof_flag);
    assert_eq!(dbh_ctrl.mdot_scheme, "contact");
    assert!((dbh_ctrl.m1_msun - 70.0).abs() < 1.0e-12);
    assert!((dbh_ctrl.m2_msun - 65.0).abs() < 1.0e-12);
    assert!((dbh_ctrl.initial_period_in_days - 1.20).abs() < 1.0e-12);

    let gr_ctrl =
        BinaryInlistControls::from_case_dir(&suite_dir.join("jdot_gr_check"))
            .expect("jdot_gr_check inlists");
    assert!(gr_ctrl.initial_period_in_days <= 0.0);
    assert!((gr_ctrl.initial_separation_in_rsuns - 2.0).abs() < 1.0e-12);
    assert!(gr_ctrl.do_jdot_gr && !gr_ctrl.do_jdot_mb && gr_ctrl.do_jdot_ml);

    let reports =
        run_binary_suite(&mesa_dir, Some(4)).expect("capped 4-step suite");
    assert_eq!(reports.len(), TIER5_BINARY_CASES.len());
    for rep in &reports {
        assert!(rep.passed, "capped case {} failed: {rep:?}", rep.case_name);
        assert_eq!(rep.total_steps, 4);
        match rep.case_name.as_str() {
            "jdot_gr_check" => assert!(rep.max_relative_oracle_error < 1.0e-2),
            "jdot_ml_check" => assert!(rep.max_relative_oracle_error < 1.0e-4),
            "jdot_ls_check" => assert!(rep.max_relative_oracle_error < 1.0e-12),
            "wind_fed_bhhmxb" => {
                assert!(rep.max_relative_oracle_error <= 1.0e-15);
            }
            _ => {}
        }
    }
}

#[test]
fn test_coupled_double_bh_35_steps_preserves_star_controls() {
    let Some((_, suite_dir)) = resolve_suite_dir() else {
        return;
    };
    let cdir = suite_dir.join("double_bh");
    let b1 =
        InlistBundle::from_file_in_case_dir(&cdir, "inlist1").expect("inlist1");
    assert!(b1.controls.rotation_flag);
    assert!(b1.controls.use_gravity_rotation_correction);
    assert!(b1.controls.use_gold2_tolerances);
    assert!(b1.controls.use_ledoux_criterion);
    assert!(b1.controls.use_other_wind);

    let mut b_probe = b1;
    b_probe.controls.initial_mass = 10.0;
    b_probe.job.relax_z = false;
    b_probe.job.relax_y = false;
    let seed =
        load_zams_model(b_probe.controls.clone(), b_probe.job.clone())
            .expect("probe ZAMS");
    let mut extras = DefaultStarExtras;
    let s_probe =
        init_or_update_model_from_bundle(&b_probe, Some(seed), &mut extras)
            .expect("init probe");
    let bctrl = BinaryInlistControls::from_case_dir(&cdir).expect("bctrl");
    let bin_probe = BinarySystem::from_controls_two_stars(
        s_probe.clone(),
        s_probe,
        bctrl,
    )
    .expect("two-star probe");
    assert!(bin_probe.donor.controls.use_gravity_rotation_correction);
    assert!(bin_probe.donor.controls.use_gold2_tolerances);
    assert!(bin_probe.donor.controls.use_ledoux_criterion);
    assert!(bin_probe.donor.controls.use_other_wind);

    let rep =
        run_binary_test_suite_case(&cdir, Some(35)).expect("double_bh 35 steps");
    assert!(rep.passed, "double_bh 35-step run failed: {rep:?}");
    assert_eq!(rep.total_steps, 35);
    assert!(rep.final_m1_msun > 69.0 && rep.final_m2_msun > 64.0);
}

#[test]
fn test_uncapped_jdot_ls_check_campaign() {
    let Some((_, suite_dir)) = resolve_suite_dir() else {
        return;
    };
    let rep =
        run_binary_test_suite_case(&suite_dir.join("jdot_ls_check"), None)
            .expect("uncapped jdot_ls_check");
    assert!(rep.passed, "uncapped jdot_ls_check failed: {rep:?}");
    assert_eq!(rep.termination_code, "LsCheckPass");
    assert!(rep.total_steps >= 100);
    assert!(rep.max_relative_oracle_error < 1.0e-12);
}

fn assert_slow_uncapped_case(case_name: &str) -> flexmesa_binary::BinaryCaseRunReport {
    let (_, suite_dir) =
        resolve_suite_dir().expect("MESA_DIR binary/test_suite must exist");
    let rep = run_binary_test_suite_case(&Path::new(&suite_dir).join(case_name), None)
        .unwrap_or_else(|e| panic!("uncapped {case_name} failed: {e}"));
    assert!(rep.passed, "uncapped {case_name} report failed: {rep:?}");
    rep
}

#[test]
#[ignore = "slow uncapped campaign (~121s) verified locally via --ignored"]
fn test_uncapped_jdot_gr_check_campaign() {
    let rep = assert_slow_uncapped_case("jdot_gr_check");
    assert_eq!(rep.termination_code, "GrCheckPass");
    assert!(rep.max_relative_oracle_error < 1.0e-2);
}

#[test]
#[ignore = "slow uncapped campaign (~147s) verified locally via --ignored"]
fn test_uncapped_star_plus_point_mass_explicit_mdot_campaign() {
    let rep = assert_slow_uncapped_case("star_plus_point_mass_explicit_mdot");
    assert_eq!(rep.termination_code, "xa_central_lower_limit(h1)");
    assert!(rep.final_m1_msun < 5.0 && rep.final_m2_msun > 10.0);
}

#[test]
#[ignore = "slow uncapped campaign (~415s) verified locally via --ignored"]
fn test_uncapped_star_plus_point_mass_campaign() {
    let rep = assert_slow_uncapped_case("star_plus_point_mass");
    assert_eq!(rep.termination_code, "xa_central_lower_limit(h1)");
    assert!(rep.final_m1_msun < 5.0 && rep.final_m2_msun > 10.0);
}

#[test]
#[ignore = "slow uncapped campaign (~621s) verified locally via --ignored"]
fn test_uncapped_jdot_ml_check_campaign() {
    let rep = assert_slow_uncapped_case("jdot_ml_check");
    assert_eq!(rep.termination_code, "MlCheckPass");
    assert!(rep.max_relative_oracle_error < 1.0e-4);
}

#[test]
#[ignore = "slow uncapped campaign (~928s) verified locally via --ignored"]
fn test_uncapped_evolve_both_stars_campaign() {
    let rep = assert_slow_uncapped_case("evolve_both_stars");
    assert_eq!(rep.termination_code, "accretor_overflow_terminate");
}

#[test]
#[ignore = "slow uncapped campaign (~1376s) verified locally via --ignored"]
fn test_uncapped_wind_fed_bhhmxb_campaign() {
    let rep = assert_slow_uncapped_case("wind_fed_bhhmxb");
    assert_eq!(rep.termination_code, "WindFedBhHmxbPass");
    assert!(rep.max_relative_oracle_error <= 1.0e-15);
}

#[test]
#[ignore = "slow multi-hundred-step campaign; run explicitly with --ignored"]
fn test_uncapped_double_bh_campaign() {
    let rep = assert_slow_uncapped_case("double_bh");
    assert_eq!(rep.termination_code, "DoubleBhPass");
    assert!(rep.final_m1_msun > 20.0 && rep.final_m2_msun > 20.0);
}
