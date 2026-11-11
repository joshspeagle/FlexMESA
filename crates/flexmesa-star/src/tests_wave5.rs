// SPDX-License-Identifier: LGPL-3.0-or-later
//! Unit tests for `flexmesa-star` Wave 5 modules (`diffusion`, `rotation`,
//! `premix`, `riemann`, `rsp2`, and `pulse`).

use super::*;

#[test]
fn test_wave5_1_diffusion_rotation_and_premix_preserve_physical_invariants() {
    // 1. Burgers diffusion settles He4 downward and floats H1 upward
    let a = [1.0, 4.0, 16.0];
    let z = [1.0, 2.0, 8.0];
    let x = [0.70, 0.28, 0.02];
    let res = eval_burgers_velocities(
        &a,
        &z,
        &x,
        10.0,
        1.0e7,
        -1.0e-10,
        2.74e4,
        None,
        DiffusionScheme::StantonMurillo,
    );
    assert_eq!(res.v_gt.len(), 3);
    // Hydrogen floats upward (v_gt[0] > 0), heavier ions sink (< 0)
    assert!(res.v_gt[0] > 0.0);
    assert!(res.v_gt[1] < 0.0);
    assert!(res.v_gt[2] < 0.0);
    // Exact mass flux conservation: sum(X_i * v_gt_i) == 0
    let mass_flux: f64 =
        x.iter().zip(res.v_gt.iter()).map(|(xi, vi)| xi * vi).sum();
    assert!(mass_flux.abs() <= 1.0e-12);

    // 2. Fabry et al. (2022) / Paxton et al. (2019) Roche distortion
    let (fp0, ft0, i0) = eval_rotation_fp_ft(0.0, 0.75, 0.95);
    assert!((fp0 - 1.0).abs() <= 1.0e-15);
    assert!((ft0 - 1.0).abs() <= 1.0e-15);
    assert!((i0 - 2.0 / 3.0).abs() <= 1.0e-15);
    let (fp_fast, ft_fast, i_fast) = eval_rotation_fp_ft(0.6, 0.5, 0.5);
    assert!(fp_fast < 1.0 && fp_fast < ft_fast && ft_fast < 1.0);
    assert!(i_fast > 2.0 / 3.0);

    // 3. Rotational instabilities + Spruit-Tayler dynamo
    let rot_zone = eval_rotational_mixing_zone(
        3.0e10, 5.0e9, 2.5e4, 5.0, 1.5e7, 3.5e8, 0.5, 1.0, 1.0, 0.4, 0.2,
        1.0e-6, 1.0e-7, 1.0e-4, 0.5, 1.3, 1.1, 0.0,
    );
    assert!(rot_zone.d_es > 0.0);
    assert!(
        rot_zone.d_st > 0.0 && rot_zone.nu_st > 0.0 && rot_zone.b_phi > 0.0
    );

    // 4. Blouin et al. (2020, 2021) phase separation shift
    let dx_co = eval_phase_separation_shift(0.5, false);
    let dx_one = eval_phase_separation_shift(0.5, true);
    assert!(dx_co > 0.05 && dx_co < 0.30);
    assert!(dx_one > 0.02 && dx_one < 0.25);
}

#[test]
fn test_wave5_2_riemann_rsp2_and_pulse_exporters_preserve_invariants() {
    // 1. HLLC Riemann exact hydrostatic preservation: u_L = u_R = 0
    let r_face = 5.0e10_f64;
    let m_grav = 1.0e33_f64;
    let cgrav = 6.67430e-8_f64;
    let dm = 1.0e30_f64;
    let g_over_a =
        cgrav * m_grav / (4.0 * std::f64::consts::PI * r_face.powi(4));
    let p_star = 1.0e15_f64;
    let p_l = p_star + 0.5 * dm * g_over_a;
    let p_r = p_star - 0.5 * dm * g_over_a;
    let f = eval_hllc_face(
        p_l, 10.0, 5.0e7, 0.0, dm, p_r, 8.0, 4.5e7, 0.0, dm, r_face, m_grav,
        cgrav,
    );
    assert!(f.u_face.abs() <= 1.0e-14);
    assert!((f.p_face - p_star).abs() / p_star <= 1.0e-14);

    // 2. RSP2 stationary root satisfies S(w) - D(w) - D_r(w) == 0 to < 1e-13
    let cfg = Rsp2Config::default();
    let w_eq = solve_stationary_rsp2_w(
        1.0e5, 1.0e-5, 1.0e8, 3.0e8, 0.40, 1.2, 2.0, 1.0e9, 0.55, &cfg,
    );
    assert!(w_eq > 0.0);
    let z = eval_rsp2_zone(
        w_eq, 1.0e5, 1.0e-5, 1.0e8, 3.0e8, 0.40, 1.2, 2.0, 1.0e9, 0.55,
        5.01e10, 5.0e10, 0.0, 0.0, 1.0e28, 0.0, &cfg,
    );
    let rel_res =
        z.de_t_dt.abs() / z.source_s.abs().max(z.sink_d.abs()).max(1.0);
    assert!(rel_res <= 1.0e-13);

    // 3. GYRE, FGONG, and OSC pulsation exporters on a ZAMS model
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("zams");
    set_hydro_vars(&mut model, false, false).expect("hydro_vars");
    let pts = build_pulse_points(&model, true, false, 1.0e99);
    let n_pts = model.nz + 1;
    assert_eq!(pts.len(), n_pts);
    assert_eq!(pts[0].r, 0.0);
    assert!((pts[0].u_dziem - 3.0).abs() <= 1.0e-14);
    let gyre_str = export_gyre(&model, true);
    assert_eq!(gyre_str.lines().count(), 1 + n_pts);
    let fgong_str = export_fgong(&model, true);
    assert_eq!(fgong_str.lines().count(), 5 + 3 + 8 * n_pts);
    let osc_str = export_osc(&model, true);
    assert_eq!(osc_str.lines().count(), 6 + 3 + 8 * n_pts);
}

#[test]
fn test_e2e_0_inlist_bundle_net_change_termination_and_photo_restart_parity() {
    use std::fs;

    // 1. Dynamic network switching preserves total mass fraction sum(X) == 1
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("zams");
    assert_eq!(model.species, 8);
    change_model_network(&mut model, "co_burn.net", None)
        .expect("switch to co_burn.net");
    assert_eq!(model.species, 9);
    for k in 0..model.nz {
        let sum_x: f64 = model.cell_xa(k).iter().sum();
        assert!((sum_x - 1.0).abs() <= 1.0e-14);
    }

    // 2. Bit-identical `.photo` restart parity over 2 evolutionary steps
    let (c_cont, c_rest) =
        verify_photo_restart_parity(&model, 2).expect("photo restart parity");
    assert_eq!(c_cont, c_rest);

    // 3. Multi-stage `InlistBundle` + `rn` stage execution in an isolated temp dir
    let tmp_dir = std::env::temp_dir()
        .join(format!("flexmesa_e2e0_test_{}", std::process::id()));
    let _ = fs::create_dir_all(&tmp_dir);

    fs::write(
        tmp_dir.join("inlist_common"),
        r"
&controls
   initial_mass = 1.0
   initial_z = 0.02
   mesh_delta_coeff = 0.0
/ ! end of controls namelist
",
    )
    .expect("write inlist_common");

    fs::write(
        tmp_dir.join("inlist_stage1"),
        r"
&star_job
   save_model_when_terminate = .true.
   save_model_filename = 'stage1_out.mod'
/ ! end of star_job namelist

&controls
   read_extra_controls_inlist(1) = .true.
   extra_controls_inlist_name(1) = 'inlist_common'
   max_model_number = 3
/ ! end of controls namelist
",
    )
    .expect("write inlist_stage1");

    fs::write(
        tmp_dir.join("inlist_stage2"),
        r"
&star_job
   load_saved_model = .true.
   load_model_filename = 'stage1_out.mod'
   change_net = .true.
   new_net_name = 'co_burn.net'
/ ! end of star_job namelist

&controls
   read_extra_controls_inlist(1) = .true.
   extra_controls_inlist_name(1) = 'inlist_common'
   max_model_number = 5
/ ! end of controls namelist
",
    )
    .expect("write inlist_stage2");

    fs::write(
        tmp_dir.join("rn"),
        r"#!/usr/bin/env bash
do_one inlist_stage1 stage1_out.mod
do_one inlist_stage2 final.mod
",
    )
    .expect("write rn");

    let rep = run_test_suite_case(&tmp_dir, true, Some(10), true)
        .expect("run 2-stage case");
    assert_eq!(rep.stages.len(), 2);
    assert_eq!(rep.stages[0].termination_code, "max_model_number");
    assert_eq!(rep.stages[1].termination_code, "max_model_number");
    assert!(rep.photo_restart_verified);
    assert!(tmp_dir.join("stage1_out.mod").exists());
    assert!(tmp_dir.join("final.mod").exists());

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_e2e_1_tier1_physics_helpers_and_suite() {
    use std::path::Path;

    assert_eq!(TIER1_CASES.len(), 33);
    assert!(is_tier1_case("make_zams"));
    assert!(is_tier1_case("simplex_solar_calibration"));
    assert!(!is_tier1_case("unknown_case"));

    let (l_eff, alpha) = eval_starspot_effective_luminosity(1.0, 0.34, 0.85);
    assert!(alpha < 1.0 && alpha > 0.8);
    assert!((l_eff * alpha - 1.0).abs() <= 1.0e-14);

    let (q_edd, fac_edd) =
        eval_t_tau_gradr_factor(2.0 / 3.0, AtmTTauRelation::Eddington);
    assert!((q_edd - 2.0 / 3.0).abs() <= 1.0e-14);
    assert!((fac_edd - 1.0).abs() <= 1.0e-14);
    let (q_ks, fac_ks) =
        eval_t_tau_gradr_factor(2.0 / 3.0, AtmTTauRelation::KrishnaSwamy);
    assert!(q_ks > 0.5 && fac_ks > 1.0);
    let (q_hopf, fac_hopf) =
        eval_t_tau_gradr_factor(2.0 / 3.0, AtmTTauRelation::SolarHopf);
    assert!(q_hopf > 0.5 && fac_hopf > 1.0);

    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("zams");
    set_hydro_vars(&mut model, false, false).expect("hydro_vars");
    let cols = eval_model_colors(&model).expect("colors");
    assert_eq!(cols.len(), 10);
    assert!(cols.iter().all(|v| v.is_finite()));

    let mut simplex = [
        ([0.270, 0.018, 1.80], 12.0),
        ([0.275, 0.019, 1.90], 4.5),
        ([0.280, 0.020, 2.00], 8.0),
        ([0.260, 0.015, 1.60], 25.0),
    ];
    let trial = nelder_mead_solar_calibration_step(&mut simplex);
    assert!(trial[0] >= 0.20 && trial[0] <= 0.35);
    assert!(
        eval_solar_calibration_objective(0.0, 0.0, 0.0229, 0.0229).abs()
            <= 1.0e-15
    );

    let require_zero_skips =
        std::env::var("FLEXMESA_REQUIRE_ZERO_SKIPS").as_deref() == Ok("1");
    if require_zero_skips {
        let mesa_dir = std::env::var("MESA_DIR")
            .expect("MESA_DIR required when FLEXMESA_REQUIRE_ZERO_SKIPS=1");
        for sub in [
            "star/test_suite",
            "astero/test_suite",
            "star/dev_cases_star_to_RSP2",
        ] {
            assert!(
                Path::new(&mesa_dir).join(sub).is_dir(),
                "missing {sub} under {mesa_dir}"
            );
        }
    }
    if let Ok(mesa_dir) = std::env::var("MESA_DIR") {
        let base = Path::new(&mesa_dir);
        let case_dir = base.join("star").join("test_suite").join("make_zams");
        if case_dir.is_dir() {
            let rep = run_test_suite_case(&case_dir, true, Some(2), true)
                .expect("make_zams tier 1 run");
            assert!(rep.passed);
            assert!(rep.assertions_passed);
            assert!(rep.photo_restart_verified);
        }
    }
}

#[test]
fn test_e2e_2_tier2_physics_helpers_and_suite() {
    assert_eq!(TIER2_CASES.len(), 17);
    assert!(is_tier2_case("make_brown_dwarf"));
    assert!(is_tier2_case("conductive_flame"));
    assert!(!is_tier2_case("make_zams"));

    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("zams");
    set_hydro_vars(&mut model, false, false).expect("hydro_vars");

    let (eps_irr, r_rjup) =
        eval_planetary_irradiation_heating(&model, 1.0e8, 250.0);
    assert_eq!(eps_irr.len(), model.nz);
    assert!(eps_irr[0] > 0.0 && eps_irr[0] <= 1.0e8 / (4.0 * 250.0));
    assert!(eps_irr[0] >= eps_irr[model.nz - 1]);
    assert!(r_rjup > 1.0);

    apply_planetary_core(&mut model, 3.0024584e-5, 10.0, 0.0);
    assert!(model.m_center > 0.0 && model.r_center > 0.0);

    for k in 0..model.nz {
        model.rho[k] = 1.0e7;
        model.lnd[k] = (1.0e7_f64).ln();
        model.t[k] = 3.0e6;
        model.ln_t[k] = (3.0e6_f64).ln();
        model.zbar[k] = 8.0;
        model.z53bar[k] = 8.0_f64.powf(5.0 / 3.0);
        model.abar[k] = 16.0;
        model.ye[k] = 0.5;
    }
    let (cryst_frac, l_lat) = eval_wd_phase_separation(&mut model, "ONe");
    assert!(cryst_frac > 0.0 && cryst_frac <= 1.0);
    assert!(l_lat > 0.0);

    model.controls.mass_change = 1.0e-9;
    model.controls.accrete_same_as_surface = false;
    model.controls.accretion_h1 = 0.70;
    model.controls.accretion_he4 = 0.28;
    let mdot_net = apply_accretion_composition_and_nova_wind(&mut model);
    assert!((mdot_net - 1.0e-9).abs() < 1.0e-15);

    let (v_flame, q_flame) = eval_conductive_flame_state(&model);
    assert!(v_flame > 0.0 && v_flame.is_finite());
    assert!(q_flame > 0.0 && q_flame <= 1.0);

    let mesa_dir = flexmesa_utils::resolve_mesa_test_inputs_dir();
    let case_dir = mesa_dir.join("star/test_suite/cburn_inward");
    if case_dir.is_dir() {
        let rep = run_test_suite_case(&case_dir, true, Some(2), true)
            .expect("cburn_inward tier 2 run");
        assert!(rep.passed);
    }
}

#[test]
fn test_e2e_4_tier4_pulsation_astero_and_dev_cases() {
    use std::path::Path;

    assert_eq!(TIER4_STAR_CASES.len(), 16);
    assert_eq!(TIER4_ACTIVE_STAR_CASES.len(), 15);
    assert_eq!(TIER4_ASTERO_CASES.len(), 8);
    assert_eq!(TIER4_DEV_CASES.len(), 32);
    assert!(is_tier4_case("gyre_in_mesa_ms"));
    assert!(is_tier4_case("rsp_Cepheid"));
    assert!(is_tier4_case("example_astero"));
    assert!(is_tier4_case("dev_TDC_Cepheid_6M"));

    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("zams");
    set_hydro_vars(&mut model, false, false).expect("hydro_vars");
    let modes = eval_linear_oscillation_modes(&model, 0, &[1, 2, 3]);
    assert_eq!(modes.len(), 3);
    assert!(modes[0].freq_hz > 0.0 && modes[1].freq_hz > modes[0].freq_hz);
    let (dnu, numax) = eval_astero_scaling_dnu_numax_uhz(&model);
    assert!(dnu > 50.0 && numax > 1000.0);

    let lina = eval_rsp_lina_modes(
        4.165,
        1438.8,
        6050.0,
        0.73,
        0.007,
        3,
        Some(3.92305),
    );
    assert_eq!(lina.len(), 3);
    assert!((lina[0].period_days - 3.92305).abs() < 1.0e-12);
    assert!(lina[1].period_days < lina[0].period_days);
    assert!(lina[0].growth_rate > 0.0);

    let require_zero_skips =
        std::env::var("FLEXMESA_REQUIRE_ZERO_SKIPS").as_deref() == Ok("1");
    if require_zero_skips {
        let mesa_dir = std::env::var("MESA_DIR")
            .expect("MESA_DIR required when FLEXMESA_REQUIRE_ZERO_SKIPS=1");
        for sub in [
            "star/test_suite",
            "astero/test_suite",
            "star/dev_cases_star_to_RSP2",
        ] {
            assert!(
                Path::new(&mesa_dir).join(sub).is_dir(),
                "missing {sub} under {mesa_dir}"
            );
        }
    }
    if let Ok(mesa_dir) = std::env::var("MESA_DIR") {
        let base = Path::new(&mesa_dir);
        if base.join("star").join("test_suite").is_dir() {
            let star_reps = run_tier4_star_suite(base, true, Some(2), true)
                .expect("tier4 star");
            assert_eq!(star_reps.len(), 16);
            assert!(star_reps
                .iter()
                .all(|r| r.passed && r.photo_restart_verified));
        }
        if base.join("astero").join("test_suite").is_dir() {
            let astero_reps = run_astero_suite(base, true, Some(2), true)
                .expect("tier4 astero");
            assert_eq!(astero_reps.len(), 8);
            assert!(astero_reps
                .iter()
                .all(|r| r.passed && r.photo_restart_verified));
        }
        if base.join("star").join("dev_cases_star_to_RSP2").is_dir() {
            let dev_reps = run_dev_cases_suite(base, true, Some(2), true)
                .expect("tier4 dev");
            assert_eq!(dev_reps.len(), 32);
            assert!(dev_reps
                .iter()
                .all(|r| r.passed && r.photo_restart_verified));
        }
    }
}

#[test]
fn test_e2e_5_tier5_star_hooks_and_91_active_star_inventory() {
    use std::path::Path;

    assert_eq!(TIER5_STAR_CASES.len(), 4);
    assert_eq!(all_active_star_case_names().len(), 91);
    assert!(is_tier5_star_case("check_redo"));
    assert!(is_tier5_star_case("other_physics_hooks"));
    assert!(is_tier5_star_case("relax_composition_j_entropy"));
    assert!(is_tier5_star_case("timing"));

    let (d_ov1, vc_ov1) = eval_overshoot_dbl_exp(
        1.0e9, 1.0e14, 1.0e5, 0.002, 0.015, 0.005, 1.0e2, 2.0e7,
    );
    let (d_ov2, _) = eval_overshoot_dbl_exp(
        1.0e9, 1.0e14, 1.0e5, 0.002, 0.015, 0.005, 1.0e2, 8.0e7,
    );
    assert!(d_ov1 > d_ov2 && d_ov2 > 0.0 && vc_ov1 > 0.0);

    let mdot_gk = eval_grafener_kudritzki_wind_msun_yr(
        1.0e6 * 3.828e33,
        60.0 * flexmesa_const::M_SUN,
        30.0 * flexmesa_const::R_SUN,
        40_000.0,
        0.7,
        0.28,
        0.02,
        0.1,
    );
    assert!(mdot_gk < 0.0 && mdot_gk.is_finite());

    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("zams");
    set_hydro_vars(&mut model, false, false).expect("hydro_vars");
    let (c1, c2) = verify_step_redo_bit_parity(&model).expect("redo parity");
    assert_eq!(c1, c2);

    let require_zero_skips =
        std::env::var("FLEXMESA_REQUIRE_ZERO_SKIPS").as_deref() == Ok("1");
    if require_zero_skips {
        let mesa_dir = std::env::var("MESA_DIR")
            .expect("MESA_DIR required when FLEXMESA_REQUIRE_ZERO_SKIPS=1");
        for sub in [
            "star/test_suite",
            "astero/test_suite",
            "star/dev_cases_star_to_RSP2",
        ] {
            assert!(
                Path::new(&mesa_dir).join(sub).is_dir(),
                "missing {sub} under {mesa_dir}"
            );
        }
    }
    if let Ok(mesa_dir) = std::env::var("MESA_DIR") {
        let base = Path::new(&mesa_dir);
        if base.join("star").join("test_suite").is_dir() {
            let tier5_reps =
                run_tier5_star_suite(base, Some(2), true).expect("tier5 star");
            assert_eq!(tier5_reps.len(), 4);
            for r in &tier5_reps {
                assert!(
                    r.passed && r.photo_restart_verified,
                    "tier5 case {} failed: passed={}, assertions={}, restart={}",
                    r.case_name,
                    r.passed,
                    r.assertions_passed,
                    r.photo_restart_verified
                );
            }
        }
    }
}
