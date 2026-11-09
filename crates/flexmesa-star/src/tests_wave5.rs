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
    assert_eq!(osc_str.lines().count(), 2 + 3 + 5 * n_pts);
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
