// SPDX-License-Identifier: LGPL-3.0-or-later
//! Unit tests for `flexmesa-star` (Waves 4.1, 4.2, and 4.3).

use super::*;
use flexmesa_const::{M_SUN, MixingType, SECYER};

#[test]
fn test_star_ad_order1_shifts_and_chain_rule_match_analytic_derivatives() {
    let lnd_00 = StarAdOrder1::variable(2.0, I_LND_00);
    let lnt_00 = StarAdOrder1::variable(16.0, I_LNT_00);
    let f_cell = (lnd_00 * 1.5 + lnt_00 * 0.5).exp();
    let expected_val = flexmesa_math::exp(2.0_f64 * 1.5 + 16.0 * 0.5);
    assert!((f_cell.val - expected_val).abs() <= 1.0e-12 * expected_val);
    assert!(
        (f_cell.d1[I_LND_00] - 1.5 * expected_val).abs()
            <= 1.0e-12 * expected_val
    );
    assert!(
        (f_cell.d1[I_LNT_00] - 0.5 * expected_val).abs()
            <= 1.0e-12 * expected_val
    );

    let shifted_m1 = f_cell.shift_m1();
    assert_eq!(shifted_m1.d1[I_LND_M1], f_cell.d1[I_LND_00]);
    assert_eq!(shifted_m1.d1[I_LNT_M1], f_cell.d1[I_LNT_00]);
    assert_eq!(shifted_m1.d1[I_LND_00], 0.0);

    let shifted_p1 = f_cell.shift_p1();
    assert_eq!(shifted_p1.d1[I_LND_P1], f_cell.d1[I_LND_00]);
    assert_eq!(shifted_p1.d1[I_LNT_P1], f_cell.d1[I_LNT_00]);
    assert_eq!(shifted_p1.d1[I_LND_00], 0.0);
}

#[test]
fn test_zams_4msun_star_test_interpolation_and_initial_timestep() {
    let controls = StarControls {
        initial_mass: 4.0,
        initial_z: 0.02,
        ..StarControls::default()
    };
    let job = StarJobControls::default();

    let yrs = yrs_for_init_timestep(4.0);
    assert!((yrs - 3125.0).abs() <= 1.0e-12);

    let model = load_zams_model(controls, job).expect("load 4 Msun ZAMS model");
    assert_eq!(model.nz, 2464);
    assert_eq!(model.species, 8);
    assert!((model.mstar - 4.0 * M_SUN).abs() <= 1.0e-12 * model.mstar);
    assert!((model.dt - 3125.0 * SECYER).abs() <= 1.0e-12 * model.dt);
    assert!((model.q[0] - 1.0).abs() <= 1.0e-15);

    let dq_sum: f64 = model.dq.iter().sum();
    assert!((dq_sum - 1.0).abs() <= 1.0e-14);
    for k in [0, model.nz / 2, model.nz - 1] {
        let xa_sum: f64 = model.cell_xa(k).iter().sum();
        assert!((xa_sum - 1.0).abs() <= 1.0e-14);
    }
}

#[test]
fn test_mod_and_photo_roundtrip_preserve_state() {
    let controls = StarControls {
        initial_mass: 1.0,
        ..StarControls::default()
    };
    let job = StarJobControls::default();
    let model = load_zams_model(controls.clone(), job.clone())
        .expect("load 1 Msun ZAMS");

    let mod_text = write_mod_string(&model).expect("write .mod");
    let model_from_mod = read_mod_str(&mod_text, controls.clone(), job.clone())
        .expect("read .mod");
    assert_eq!(model_from_mod.nz, model.nz);
    assert_eq!(model_from_mod.species, model.species);
    for k in [0, model.nz / 2, model.nz - 1] {
        assert!((model_from_mod.lnd[k] - model.lnd[k]).abs() <= 1.0e-14);
        assert!((model_from_mod.ln_t[k] - model.ln_t[k]).abs() <= 1.0e-14);
        assert!((model_from_mod.ln_r[k] - model.ln_r[k]).abs() <= 1.0e-14);
        assert!(
            (model_from_mod.lum[k] - model.lum[k]).abs()
                <= 1.0e-14 * model.lum[k].abs().max(1.0)
        );
    }

    let photo_bytes = write_photo_bytes(&model);
    let model_from_photo =
        read_photo_bytes(&photo_bytes, controls, job).expect("read .photo");
    assert_eq!(model_from_photo.lnd, model.lnd);
    assert_eq!(model_from_photo.ln_t, model.ln_t);
    assert_eq!(model_from_photo.ln_r, model.ln_r);
    assert_eq!(model_from_photo.lum, model.lum);
    assert_eq!(model_from_photo.dq, model.dq);
    assert_eq!(model_from_photo.xa, model.xa);
}

#[test]
fn test_set_hydro_vars_on_4msun_zams_model_populates_convective_core() {
    let controls = StarControls {
        initial_mass: 4.0,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    let job = StarJobControls::default();
    let mut model = load_zams_model(controls, job).expect("load 4 Msun");

    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    let k_center = model.nz - 1;
    assert!(model.peos[k_center] > 1.0e16);
    assert!(model.t[k_center] > 2.0e7);
    assert!(model.eps_nuc[k_center] > 100.0);
    assert!(model.teff > 1.0e4 && model.teff < 2.5e4);
    assert!(model.ln_peos_surf_ad.val.is_finite());
    assert!(model.ln_t_surf_ad.val.is_finite());

    let n_conv_core = (model.nz - 200..model.nz)
        .filter(|&k| model.mixing_type[k] == MixingType::ConvectiveMixing)
        .count();
    assert!(n_conv_core > 50);
}

#[test]
fn test_build_pre_ms_model_1msun_converges_to_convective_polytrope() {
    let controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    let job = StarJobControls {
        create_pre_ms_model: true,
        pre_ms_d_log10_p: 0.04,
        ..StarJobControls::default()
    };

    let mut model =
        build_pre_ms_model(controls, job).expect("build 1 Msun pre-MS");
    assert!(model.nz > 50);
    set_hydro_vars(&mut model, true, true).expect("set_hydro_vars on pre-MS");
    assert!(model.t[model.nz - 1] > 2.5e5);
    assert!(model.peos[model.nz - 1] > model.peos[0]);
}

#[test]
fn test_henyey_jacobian_blocks_match_central_finite_differences() {
    let controls = StarControls {
        initial_mass: 4.0,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    let job = StarJobControls::default();
    let mut model = load_zams_model(controls, job).expect("load 4 Msun");
    model.dt = 1.0;
    set_hydro_vars(&mut model, false, false).expect("init hydro vars");
    model.save_start_values().expect("save_start_values");

    let eps = 1.0e-7;
    for include_chem in [false, true] {
        let sys = eval_henyey_equations(&mut model, include_chem)
            .expect("eval equations");
        let nvar = sys.nvar;
        let blk_sz = nvar * nvar;
        let check_blk = |blk: &[f64],
                         sys_p: &HenyeySystem,
                         sys_m: &HenyeySystem,
                         k_eq: usize,
                         i_equ: usize,
                         j_var: usize,
                         name: &str| {
            let idx_eq = k_eq * nvar + i_equ;
            let fd = (sys_p.equ[idx_eq] - sys_m.equ[idx_eq]) / (2.0 * eps);
            let ad = blk[k_eq * blk_sz + i_equ * nvar + j_var];
            let tol = 5.0e-5 * ad.abs().max(fd.abs()).max(1.0);
            assert!(
                (ad - fd).abs() <= tol,
                "{name} mismatch (chem={include_chem}) at k={k_eq}, \
                 equ={i_equ}, var={j_var}: ad={ad}, fd={fd}"
            );
        };

        let nvar_hydro = model.layout.nvar_hydro;
        let k_list = if include_chem {
            vec![model.nz - 2]
        } else {
            vec![0, model.nz / 2, model.nz - 1]
        };
        for k_var in k_list {
            for j_var in 0..nvar {
                let scale = sys.x_scale[k_var * nvar + j_var];
                let delta = eps * scale;

                let perturb = |sign: f64| -> HenyeySystem {
                    let mut m = model.clone();
                    if j_var < nvar_hydro {
                        m.xh[k_var * nvar_hydro + j_var] += sign * delta;
                        m.unpack_xh();
                        m.set_m_grav_and_grav();
                    } else {
                        let j_iso = j_var - nvar_hydro;
                        m.xa[k_var * m.species + j_iso] += sign * delta;
                    }
                    eval_henyey_equations(&mut m, include_chem)
                        .expect("perturbed eval")
                };
                let sys_p = perturb(1.0);
                let sys_m = perturb(-1.0);

                let i_start = if j_var < nvar_hydro { 0 } else { nvar_hydro };
                for i_equ in i_start..nvar {
                    check_blk(
                        &sys.dblk, &sys_p, &sys_m, k_var, i_equ, j_var, "dblk",
                    );
                    if k_var + 1 < model.nz {
                        check_blk(
                            &sys.lblk,
                            &sys_p,
                            &sys_m,
                            k_var + 1,
                            i_equ,
                            j_var,
                            "lblk",
                        );
                    }
                    if k_var > 0 {
                        check_blk(
                            &sys.ublk,
                            &sys_p,
                            &sys_m,
                            k_var - 1,
                            i_equ,
                            j_var,
                            "ublk",
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "10-yr step on a coarse (d_log10_P = 0.15) pre-MS model ends singular with the upstream pre-MS builder; setup not run in MESA; tracked in #154"]
fn test_solve_henyey_step_converges_on_perturbed_model() {
    let controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        zbase: 0.02,
        solver_max_tries_before_reject: 25,
        ..StarControls::default()
    };
    let job = StarJobControls {
        create_pre_ms_model: true,
        pre_ms_d_log10_p: 0.15,
        ..StarJobControls::default()
    };
    let mut model = build_pre_ms_model(controls, job).expect("build pre-MS");
    model.dt = 10.0 * SECYER;
    set_hydro_vars(&mut model, false, false).expect("init hydro");
    model.save_start_values().expect("save_start_values");

    let mut model_thomas = model.clone();
    let report_bcyclic =
        solve_henyey_step(&mut model, false, true).expect("solve bcyclic");
    assert!(
        report_bcyclic.converged,
        "expected bcyclic convergence, got {report_bcyclic:?}"
    );
    let (tol_rn, _, _, _) = crate::solver::active_solver_tolerances(
        &model,
        report_bcyclic.iterations,
    );
    assert!(report_bcyclic.residual_norm <= tol_rn);

    let report_thomas = solve_henyey_step(&mut model_thomas, false, false)
        .expect("solve thomas");
    assert!(
        report_thomas.converged,
        "expected thomas convergence, got {report_thomas:?}"
    );
    for k in 0..model.nz {
        let diff_d = (model.lnd[k] - model_thomas.lnd[k]).abs();
        assert!(
            diff_d <= 1.0e-6,
            "bcyclic vs thomas lnd mismatch at zone {k}: diff={diff_d}"
        );
        let diff_t = (model.ln_t[k] - model_thomas.ln_t[k]).abs();
        assert!(
            diff_t <= 1.0e-6,
            "bcyclic vs thomas ln_t mismatch at zone {k}: diff={diff_t}"
        );
    }

    model.save_start_values().expect("save_start_values");
    let report_chem =
        solve_henyey_step(&mut model, true, true).expect("solve coupled");
    assert!(
        report_chem.converged,
        "expected coupled hydro+chem convergence, got {report_chem:?}"
    );
}

#[test]
fn test_convective_boundaries_overshoot_diffusion_and_winds_conserve_mass() {
    let controls = StarControls {
        initial_mass: 4.0,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    let job = StarJobControls::default();
    let mut model = load_zams_model(controls, job).expect("load 4 Msun");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    clean_mixing_glitches(&mut model, 1.0, 0.1);
    let n_bdy = locate_convection_boundaries(&mut model)
        .expect("locate_convection_boundaries");
    assert_eq!(model.aux.mixing.cz_bdy_dq.len(), model.nz);
    assert_eq!(n_bdy, 5);

    let ov_cfg = OvershootConfig::default();
    crate::overshoot::apply_overshoot_configs(&mut model, &[ov_cfg])
        .expect("apply_overshoot_configs");
    let n_ov = model
        .mixing_type
        .iter()
        .filter(|&&m| m == MixingType::OvershootMixing)
        .count();
    assert!(n_ov > 0);

    compute_convection_sigmas(&mut model, 1.0e8, 1.0, 1.0e10).unwrap();
    let dxdt_mix = eval_dxdt_mix(&model);
    for j in 0..model.species {
        let net_mass_rate: f64 = (0..model.nz)
            .map(|k| model.dm[k] * dxdt_mix[k * model.species + j])
            .sum();
        let scale: f64 = (0..model.nz)
            .map(|k| (model.dm[k] * dxdt_mix[k * model.species + j]).abs())
            .sum::<f64>()
            .max(1.0);
        assert!(net_mass_rate.abs() <= 1.0e-12 * scale);
    }

    let mdot =
        set_mdot(&mut model, WindScheme::Dutch, DutchLowTScheme::DeJager, 1.0);
    assert!(mdot < 0.0);
    let delta_m = -1.0e-6 * M_SUN;
    do_adjust_mass(&mut model, delta_m, None, 0.99, 0.95)
        .expect("do_adjust_mass");
    let dq_sum: f64 = model.dq.iter().sum();
    assert!((dq_sum - 1.0).abs() <= 1.0e-14);
}

#[test]
fn test_remesh_timestep_evolve_and_history_preserve_invariants() {
    let controls = StarControls {
        initial_mass: 4.0,
        initial_z: 0.02,
        zbase: 0.02,
        mesh_delta_coeff: 0.5,
        varcontrol_target: 1.0e-3,
        max_model_number: Some(3),
        ..StarControls::default()
    };
    let job = StarJobControls {
        create_pre_ms_model: true,
        pre_ms_d_log10_p: 0.15,
        ..StarJobControls::default()
    };
    let mut model = build_pre_ms_model(controls, job).expect("build pre-MS");
    model.dt = 10.0 * SECYER;
    model.dt_next = model.dt;
    set_hydro_vars(&mut model, false, false).expect("init hydro");

    let mass_h1_before: f64 = (0..model.nz)
        .map(|k| model.dm[k] * model.xa[k * model.species])
        .sum();
    let (n_splits, _n_merges) =
        do_adjust_mesh(&mut model, &MeshFunctionConfig::default())
            .expect("do_adjust_mesh");
    assert!(n_splits > 0);
    let mass_h1_after: f64 = (0..model.nz)
        .map(|k| model.dm[k] * model.xa[k * model.species])
        .sum();
    assert!((mass_h1_after - mass_h1_before).abs() <= 1.0e-12 * mass_h1_before);

    let reports =
        star_evolve(&mut model, 5, false, false, true).expect("star_evolve");
    // A new pre-MS model starts at model_number 0 (`star/private/init.f90:676`,
    // #359), so reaching `max_model_number = 3` takes three steps.
    assert_eq!(reports.len(), 3);
    assert_eq!(model.model_number, 3);
    assert!(model.star_age > 0.0);

    let h_row = extract_history_row(&model);
    assert_eq!(h_row.model_number, 3);
    let h_str = format_history_row(&h_row);
    assert!(h_str.contains('3'));
    let prof_str = format_profile_string(&model);
    assert!(prof_str.contains("photosphere_L"));
}

#[test]
fn test_do1_energy_eqn_time_centers_non_nuc_neu() {
    let controls = StarControls {
        initial_mass: 4.0,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    model.dt = 100.0;
    set_hydro_vars(&mut model, false, false).expect("init hydro");
    model.save_start_values().expect("save_start_values");

    let k = model.nz - 1;
    let x_scale = compute_x_scale(&model, false);
    let mut sys_base =
        HenyeySystem::new(model.nz, model.layout.nvar_hydro, x_scale.clone());
    do1_energy_eqn(&model, k, false, &mut sys_base);

    let delta_neu = 1.0e4_f64;
    let mut model_trial = model.clone();
    model_trial.non_nuc_neu[k] += delta_neu;
    let mut sys_trial =
        HenyeySystem::new(model.nz, model.layout.nvar_hydro, x_scale);
    do1_energy_eqn(&model_trial, k, false, &mut sys_trial);

    let i_equ = model.layout.i_dlne_dt;
    let idx = k * model.layout.nvar_hydro + i_equ;
    let scal = model.dt / model.energy_start[k].max(1.0);
    let expected_diff = -0.5 * delta_neu * scal;
    let actual_diff = sys_trial.equ[idx] - sys_base.equ[idx];
    assert!(
        (actual_diff - expected_diff).abs() <= 1.0e-12 * expected_diff.abs(),
        "expected time-centered 0.5*delta_neu shift {expected_diff}, got {actual_diff}"
    );
}

#[test]
fn test_default_star_controls_zbase_sentinel_and_wind_surface_z_fallback() {
    assert_eq!(StarControls::default().zbase, -1.0);

    let mut m_uninit =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load ZAMS with default controls");
    assert_eq!(m_uninit.kap_config.zbase, -1.0);
    let err = set_hydro_vars(&mut m_uninit, true, true)
        .expect_err("Type 2 opacities must reject default zbase = -1.0");
    assert!(format!("{err}").contains("zbase"));

    // With Type 2 off and kap_config.zbase = -1.0, hot winds use surface 1-X-Y
    // rather than solar Z = 0.02 (`star/private/winds.f90:336-341`).
    let mut m_lowz = load_zams_model(
        StarControls {
            initial_mass: 20.0,
            hot_wind_full_on_t: 1.0e4,
            cool_wind_full_on_t: 9.0e3,
            vink_scaling_factor: 1.0,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("20 Msun ZAMS");
    m_lowz.controls.mix_wind.hot_wind_scheme = String::from("Vink");
    m_lowz.kap_config.use_type2_opacities = false;
    m_lowz.kap_config.use_zbase_for_type1 = false;
    for k in 0..m_lowz.nz {
        let row = m_lowz.cell_xa_mut(k);
        row.fill(0.0);
        row[0] = 0.758; // h1
        row[2] = 0.240; // he4 -> surface Z = 0.002 (0.1 Z_solar)
        row[3] = 0.002; // c12
    }
    m_lowz.finish_init_model().expect("finish_init_model");
    set_hydro_vars(&mut m_lowz, true, true).expect("Type 1 hydro_vars");
    let mut m_solarz = m_lowz.clone();
    m_solarz.kap_config.zbase = 0.02;

    let mdot_lowz =
        crate::winds_driver::do_set_mdot(&mut m_lowz).expect("mdot lowz");
    let mdot_solarz =
        crate::winds_driver::do_set_mdot(&mut m_solarz).expect("mdot solarz");
    assert!(mdot_lowz < 0.0 && mdot_solarz < 0.0);
    assert!(
        mdot_lowz.abs() < 0.5 * mdot_solarz.abs(),
        "surface Z=0.002 Vink wind ({mdot_lowz:e}) must be weaker than Zbase=0.02 ({mdot_solarz:e})"
    );
}

#[test]
fn test_tdc_mlt_vc_old_preserved_across_pre_step_remesh_and_mod_roundtrip() {
    let controls = StarControls {
        initial_mass: 4.0,
        initial_z: 0.02,
        zbase: 0.02,
        mesh_delta_coeff: 2.5,
        okay_to_remesh: true,
        ..StarControls::default()
    };
    let mut model = load_zams_model(controls, StarJobControls::default())
        .expect("4 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("init hydro");
    star_evolve_step(&mut model, false, false, true).expect("step 1");
    assert!(model.aux.have_mlt_vc);

    let vc_after_step1 = model.mlt_vc.clone();
    let xq_after_step1 = model.xq.clone();
    assert!(vc_after_step1.iter().any(|&v| v > 0.0));

    // 1. Active remesh (`ns1 + nm1 > 0`) interpolates face `mlt_vc` onto new `xq`
    // without prematurely stepping TDC before `save_old_values()`.
    model.aux.okay_to_set_mlt_vc = false;
    model.controls.okay_to_remesh = true;
    model.controls.mesh_delta_coeff = 2.5;
    let dt1 = model.dt;
    let (ns1, nm1, _) =
        crate::evolve_support::remesh_star_before_step(&mut model, dt1)
            .expect("active remesh");
    assert!(ns1 + nm1 > 0);
    assert_ne!(model.nz, xq_after_step1.len());
    // 2. Converge mesh adaptation, then verify a 0-split/0-merge pre-step call
    // runs `set_hydro_vars` while `okay_to_set_mlt_vc == false` and leaves
    // `model.mlt_vc` byte-identical.
    for _ in 0..8 {
        let (s, m, _) =
            crate::evolve_support::remesh_star_before_step(&mut model, dt1)
                .expect("converge remesh");
        if s + m == 0 {
            break;
        }
    }
    let vc_settled = model.mlt_vc.clone();
    let (ns0, nm0, _) =
        crate::evolve_support::remesh_star_before_step(&mut model, dt1)
            .expect("no-op remesh on already-adapted mesh");
    assert_eq!((ns0, nm0), (0, 0));
    assert_eq!(
        model.mlt_vc, vc_settled,
        "set_hydro_vars inside remesh_star_before_step must not overwrite mlt_vc"
    );
    model.save_old_values();
    assert_eq!(model.aux.mlt_vc_old, model.mlt_vc);
    assert!(model.aux.mlt_vc_old.iter().any(|&v| v > 0.0));

    // Radiative .mod roundtrip with mlt_vc == 0 still preserves have_mlt_vc = true.
    model.mlt_vc.fill(0.0);
    model.aux.have_mlt_vc = true;
    let mod_txt = write_mod_string(&model).expect("write radiative .mod");
    let reloaded =
        read_mod_str(&mod_txt, model.controls.clone(), model.job.clone())
            .expect("read radiative .mod");
    assert!(reloaded.aux.have_mlt_vc);
}

#[test]
fn test_verify_photo_restart_parity_succeeds_after_max_age_termination() {
    use crate::inlist_runner::verify_photo_restart_parity;

    let controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        zbase: 0.02,
        okay_to_remesh: false,
        ..StarControls::default()
    };
    let mut base = load_zams_model(controls, StarJobControls::default())
        .expect("1 Msun ZAMS");
    set_hydro_vars(&mut base, false, false).expect("init hydro");
    star_evolve_step(&mut base, false, false, true).expect("step 1");
    assert!(base.star_age > 0.0);

    // 1. Model sitting at `max_age` with `dt_next = 0.0` (clipped by
    // `pick_next_timestep`): direct `star_evolve_step` returns `Terminated`,
    // while `verify_photo_restart_parity` clears age caps on its clones.
    let mut at_max_age = base.clone();
    at_max_age.controls.max_age = at_max_age.star_age;
    at_max_age.dt_next = 0.0;
    let mut direct = at_max_age.clone();
    let direct_err = star_evolve_step(&mut direct, false, false, true)
        .expect_err("direct step at max_age must terminate");
    assert!(
        matches!(
            direct_err,
            StarError::Terminated { ref reason, .. } if reason == "max_age"
        ),
        "unexpected direct error at max_age: {direct_err:?}"
    );
    let (c1, c2) = verify_photo_restart_parity(&at_max_age, 1)
        .expect("photo restart parity must succeed at max_age");
    assert_eq!(c1, c2);
    assert_eq!(at_max_age.controls.max_age, at_max_age.star_age);

    // 2. Model sitting at `max_age_in_days` with positive `dt_next` (clipped
    // in `prepare_for_new_step`).
    let mut at_max_days = base.clone();
    at_max_days.controls.ts_stop.max_age_in_days =
        at_max_days.aux.step_hist.time_sec / flexmesa_const::SECDAY;
    let (d1, d2) = verify_photo_restart_parity(&at_max_days, 1)
        .expect("photo restart parity must succeed at max_age_in_days");
    assert_eq!(d1, d2);

    // 3. Model sitting at `max_age_in_seconds`.
    let mut at_max_secs = base;
    at_max_secs.controls.max_age_in_seconds =
        Some(at_max_secs.aux.step_hist.time_sec);
    let (s1, s2) = verify_photo_restart_parity(&at_max_secs, 1)
        .expect("photo restart parity must succeed at max_age_in_seconds");
    assert_eq!(s1, s2);
}
