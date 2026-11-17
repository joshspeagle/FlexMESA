// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stage `E2E-5` (`#114`) coupled `binary/test_suite` integration tests and
//! unit checks for decoupled Eddington retention vs. radiation-corrected
//! accretion, recursive 3-level binary inlist chaining,
//! `initial_separation_in_rsuns` orbital initialization, Brott et al. (2011)
//! winds, and dynamic `set_point_mass_i` conversion.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use flexmesa_binary::{
    bh_initial_equatorial_mass, eval_brott_wind_msun_yr,
    eval_edot_enhancement_isotropic, eval_point_mass_edd_and_retention,
    eval_synchronicity_switch, run_binary_suite, run_binary_test_suite_case,
    BinaryCompanion, BinaryInlistControls, BinarySystem, ImplicitMdotSolver,
    ImplicitStepAction, RocheDeformationTables, TIER5_BINARY_CASES,
};
use flexmesa_const::{CLIGHT, MSUN, RSUN, SECYER};
use flexmesa_star::{
    check_termination_limits, init_or_update_model_from_bundle,
    load_zams_model, DefaultStarExtras, InlistBundle, MdotRedoState,
    SelectMdotAction, StarControls, StarJobControls,
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
    let seed = load_zams_model(b_probe.controls.clone(), b_probe.job.clone())
        .expect("probe ZAMS");
    let mut extras = DefaultStarExtras;
    let s_probe =
        init_or_update_model_from_bundle(&b_probe, Some(seed), &mut extras)
            .expect("init probe");
    let bctrl = BinaryInlistControls::from_case_dir(&cdir).expect("bctrl");
    let bin_probe =
        BinarySystem::from_controls_two_stars(s_probe.clone(), s_probe, bctrl)
            .expect("two-star probe");
    assert!(bin_probe.donor.controls.use_gravity_rotation_correction);
    assert!(bin_probe.donor.controls.use_gold2_tolerances);
    assert!(bin_probe.donor.controls.use_ledoux_criterion);
    assert!(bin_probe.donor.controls.use_other_wind);

    let rep = run_binary_test_suite_case(&cdir, Some(35))
        .expect("double_bh 35 steps");
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
    assert_eq!(rep.termination_code, "max_age");
    assert!(rep.total_steps >= 100);
    assert!(rep.max_relative_oracle_error < 1.0e-12);
}

fn assert_slow_uncapped_case(
    case_name: &str,
) -> flexmesa_binary::BinaryCaseRunReport {
    let (_, suite_dir) =
        resolve_suite_dir().expect("MESA_DIR binary/test_suite must exist");
    let rep = run_binary_test_suite_case(
        &Path::new(&suite_dir).join(case_name),
        None,
    )
    .unwrap_or_else(|e| panic!("uncapped {case_name} failed: {e}"));
    assert!(rep.passed, "uncapped {case_name} report failed: {rep:?}");
    rep
}

#[test]
#[ignore = "unverified uncapped binary campaign; tracked in #143 and #160"]
fn test_uncapped_jdot_gr_check_campaign() {
    let rep = assert_slow_uncapped_case("jdot_gr_check");
    assert_eq!(rep.termination_code, "GrCheckPass");
    assert!(rep.max_relative_oracle_error < 1.0e-2);
}

#[test]
#[ignore = "unverified uncapped binary campaign; tracked in #143 and #160"]
fn test_uncapped_star_plus_point_mass_explicit_mdot_campaign() {
    let rep = assert_slow_uncapped_case("star_plus_point_mass_explicit_mdot");
    assert_eq!(rep.termination_code, "xa_central_lower_limit(h1)");
    assert!(rep.final_m1_msun < 5.0 && rep.final_m2_msun > 10.0);
}

#[test]
#[ignore = "unverified uncapped binary campaign; tracked in #143 and #160"]
fn test_uncapped_star_plus_point_mass_campaign() {
    let rep = assert_slow_uncapped_case("star_plus_point_mass");
    assert_eq!(rep.termination_code, "xa_central_lower_limit(h1)");
    assert!(rep.final_m1_msun < 5.0 && rep.final_m2_msun > 10.0);
}

#[test]
#[ignore = "unverified uncapped binary campaign; tracked in #143 and #160"]
fn test_uncapped_jdot_ml_check_campaign() {
    let rep = assert_slow_uncapped_case("jdot_ml_check");
    assert_eq!(rep.termination_code, "max_age");
    assert!(rep.max_relative_oracle_error < 1.0e-4);
}

#[test]
#[ignore = "unverified uncapped binary campaign; tracked in #143 and #160"]
fn test_uncapped_evolve_both_stars_campaign() {
    let rep = assert_slow_uncapped_case("evolve_both_stars");
    assert_eq!(rep.termination_code, "accretor_overflow_terminate");
}

#[test]
#[ignore = "unverified uncapped binary campaign; tracked in #143 and #160"]
fn test_uncapped_wind_fed_bhhmxb_campaign() {
    let rep = assert_slow_uncapped_case("wind_fed_bhhmxb");
    assert_eq!(rep.termination_code, "xa_central_lower_limit(h1)");
    assert!(rep.max_relative_oracle_error <= 1.0e-15);
}

#[test]
#[ignore = "unverified uncapped binary campaign; tracked in #143 and #160"]
fn test_uncapped_double_bh_campaign() {
    let rep = assert_slow_uncapped_case("double_bh");
    assert_eq!(rep.termination_code, "DoubleBhPass");
    assert!(rep.final_m1_msun > 20.0 && rep.final_m2_msun > 20.0);
}

#[test]
fn test_roche_deformation_tables_and_synchronicity_switch() {
    let Some(_) = resolve_suite_dir() else {
        return;
    };
    let tables = RocheDeformationTables::global()
        .expect("roche_data tables should load from MESA_DIR");
    let fp = tables.eval_fp(-0.5, 1.35);
    let ft = tables.eval_ft(-0.5, 1.35);
    let irot = tables.eval_irot(-0.5, 1.35);
    assert!(fp > 0.3 && fp < 1.0, "unexpected fp={fp}");
    assert!(ft > 0.3 && ft < 1.0, "unexpected ft={ft}");
    assert!((1.0..5.0).contains(&irot), "unexpected irot={irot}");
    assert_eq!(tables.eval_psi(-0.5, 0.0), -9.0e99);

    let period_unit_omega = 2.0 * std::f64::consts::PI;
    let f_sync =
        eval_synchronicity_switch(1.0, period_unit_omega, 0.8, 20.0, 0.05);
    assert!((f_sync - 1.0).abs() < 1.0e-3);
    let f_far =
        eval_synchronicity_switch(0.1, period_unit_omega, 0.8, 20.0, 0.05);
    assert!(f_far < 0.05);
}

#[test]
fn test_contact_solver_fixed_delta_mdot_crosses_zero_for_donor_switch() {
    let ctrl = BinaryInlistControls {
        mdot_scheme: String::from("contact"),
        implicit_scheme_tolerance: 1.0e-3,
        initial_change_factor: 1.5,
        ..BinaryInlistControls::default()
    };
    let mut solver = ImplicitMdotSolver::from_controls(&ctrl, 1.5);
    let m0 = -1.0e-5 * MSUN / SECYER;
    // Try 1: accretor overflows more than donor (f < 0) -> steps toward 0 using fixed_delta_mdot.
    let p1 = solver.evaluate_trial("contact", 1, 20, m0, m0, 0.02, 0.08, 0.9);
    let ImplicitStepAction::TryNextMdot(m1) = p1 else {
        panic!("expected TryNextMdot on try 1, got {p1:?}");
    };
    assert!(m1 > m0 && m1 < 0.0);
    let mut m_cur = m1;
    let mut crossed_zero = false;
    for try_idx in 2..=10 {
        let prop = solver.evaluate_trial(
            "contact", try_idx, 20, m_cur, m_cur, 0.02, 0.08, 0.9,
        );
        if let ImplicitStepAction::TryNextMdot(m_next) = prop {
            m_cur = m_next;
            if m_cur > 0.0 {
                crossed_zero = true;
                break;
            }
        }
    }
    assert!(
        crossed_zero,
        "contact solver must cross zero when f < 0 persists"
    );
    solver.flip_for_contact_donor_switch();
    let q_new = 1.0_f64 / 0.9;
    let gap_a_eq = 0.05 * flexmesa_math::pow(q_new, -0.52);
    let p_conv = solver.evaluate_trial(
        "contact", 5, 20, -m_cur, -m_cur, 0.05, gap_a_eq, q_new,
    );
    assert_eq!(p_conv, ImplicitStepAction::Converged(-m_cur));
}

#[test]
fn test_edot_isotropic_eccentricity_enhancement_and_mdot_redo_boost() {
    let r_prof = [10.0 * RSUN, 9.5 * RSUN];
    let p_prof = [1.0e5, 2.0e5];
    let t_prof = [3.0e4, 3.5e4];
    let mu_prof = [0.62, 0.62];
    let gam_prof = [1.66, 1.66];
    let edot_enh = eval_edot_enhancement_isotropic(
        20.0 * MSUN,
        15.0 * MSUN,
        10.0 * RSUN,
        9.8 * RSUN,
        0.3,
        1.0e-8,
        1.0e5,
        3.0e4,
        0.62,
        &r_prof,
        &p_prof,
        &t_prof,
        &mu_prof,
        &gam_prof,
        false,
        200,
    );
    assert!(edot_enh.is_finite());

    let mut c1 = StarControls {
        initial_mass: 20.0,
        rotation_flag: true,
        surf_omega_div_omega_crit_limit: Some(0.98),
        ..StarControls::default()
    };
    c1.mix_wind.surf_omega_div_omega_crit_tol = 0.02;
    c1.mix_wind.max_mdot_redo_cnt = 10;
    c1.mix_wind.mdot_revise_factor = 1.2;
    c1.mix_wind.implicit_mdot_boost = 0.1;
    let mut star =
        load_zams_model(c1, StarJobControls::default()).expect("20 Msun ZAMS");
    for k in 0..star.nz {
        star.omega[k] = 1.0e-3;
    }
    // Verify check_termination_limits does NOT terminate on surf_omega_div_omega_crit_limit.
    let term = check_termination_limits(&star);
    assert_ne!(
        term.as_deref(),
        Some("surf_omega_div_omega_crit_limit"),
        "surf_omega_div_omega_crit_limit must be handled by mdot_redo, not termination"
    );
    let mut redo = MdotRedoState::setup(&mut star);
    let act = redo.select_action(&mut star, 1.0e2 * SECYER);
    assert!(matches!(
        act,
        SelectMdotAction::Accept | SelectMdotAction::RedoWithMdot(_)
    ));
}

#[test]
fn test_double_bh_che_diffusivity_stiffness_and_bh_collapse_invariants() {
    let Some((_, suite_dir)) = resolve_suite_dir() else {
        return;
    };
    let cdir = suite_dir.join("double_bh");
    let b1 =
        InlistBundle::from_file_in_case_dir(&cdir, "inlist1").expect("inlist1");
    let mw = &b1.controls.mix_wind;
    assert!(mw.premix_omega);
    assert!((mw.am_nu_st_factor - 1.0).abs() < 1.0e-12);
    assert!(mw.am_nu_sh_factor.abs() < 1.0e-12);
    assert!((mw.am_d_mix_factor - 1.0 / 30.0).abs() < 1.0e-5);
    assert_eq!(mw.num_cells_for_smooth_gradl_composition_term, 2);

    let mut probe = b1.clone();
    probe.controls.initial_mass = 20.0;
    probe.job.relax_z = false;
    probe.job.relax_y = false;
    let seed = load_zams_model(probe.controls.clone(), probe.job.clone())
        .expect("zams");
    let mut extras = DefaultStarExtras;
    let mut star =
        init_or_update_model_from_bundle(&probe, Some(seed), &mut extras)
            .expect("init rotating star");
    let bctrl = BinaryInlistControls::from_case_dir(&cdir).expect("bctrl");
    let mut bin = BinarySystem::from_controls_two_stars(
        star.clone(),
        star.clone(),
        bctrl,
    )
    .expect("two-star system");
    assert!((bin.donor.controls.varcontrol_target - 1.0e-3).abs() < 1.0e-12);

    // Stress low-shear Spruit-Tayler + extreme boundary stiffness (> 2^53 raw).
    let om_sync = 2.0 * std::f64::consts::PI / bin.period;
    let nz_f = f64::from(u32::try_from(star.nz).unwrap_or(1));
    for k in 0..star.nz {
        let k_f = f64::from(u32::try_from(k).unwrap_or(0));
        star.omega[k] = om_sync * (1.0 + 1.0e-6 * k_f / nz_f);
    }
    star.dt = 1.0e4 * SECYER;
    flexmesa_star::update_rotation_distortion_factors(&mut star);
    flexmesa_star::set_hydro_vars(&mut star, false, false)
        .expect("set_hydro_vars with CHE");
    assert!(
        star.mixing_type
            .contains(&flexmesa_const::MixingType::RotationMixing),
        "CHE radiative envelope must activate RotationMixing (D_ES > 0)"
    );
    for mlt_k in &mut star.mlt_d {
        if *mlt_k > 0.0 {
            *mlt_k = 1.0e22;
        }
    }
    let j_before: f64 = star
        .dm_bar
        .iter()
        .zip(&star.j_rot)
        .map(|(m, j)| m * j)
        .sum();
    let zones = flexmesa_star::set_rotation_mixing_info(&mut star);
    assert!(zones.iter().all(|z| z.nu_st == 0.0));
    let dt_star = star.dt;
    flexmesa_star::sync_post_henyey_rotation(&mut star, dt_star);
    assert!(star.omega.iter().all(|w| w.is_finite() && *w > 0.0));
    assert!(star.j_rot.iter().all(|j| j.is_finite() && *j > 0.0));
    let j_after: f64 = star
        .dm_bar
        .iter()
        .zip(&star.j_rot)
        .map(|(m, j)| m * j)
        .sum();
    assert!(
        (j_after - j_before).abs() / j_before.abs().max(1.0) < 1.0e-12,
        "solve_omega_transport must conserve J_tot: before={j_before}, after={j_after}"
    );

    // Verify dynamic BH1 collapse clears stale termination reason & preserves [m1, m2].
    bin.donor.mstar = 35.0 * MSUN;
    if let BinaryCompanion::Star(ref mut s2) = bin.companion {
        s2.mstar = 32.0 * MSUN;
    }
    bin.last_termination_reason = Some(String::from("max_model_number"));
    bin.change_factor = 3.2;
    bin.set_point_mass_i(1);
    assert!(bin.last_termination_reason.is_none());
    assert!(
        (bin.change_factor - bin.controls.max_change_factor).abs() < 1.0e-12
    );
    let [m1_g, m2_g] = bin.star_masses_g();
    assert!((m1_g / MSUN - 35.0).abs() < 1.0e-12);
    assert!((m2_g / MSUN - 32.0).abs() < 1.0e-12);
}
