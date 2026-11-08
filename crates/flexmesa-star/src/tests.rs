// SPDX-License-Identifier: LGPL-3.0-or-later
//! Unit tests for `flexmesa-star` (Waves 4.1, 4.2, and 4.3).

use super::*;
use flexmesa_const::{MixingType, M_SUN, SECYER};

#[test]
fn test_star_ad_order1_shifts_and_chain_rule_match_analytic_derivatives() {
    let lnd_00 = StarAdOrder1::variable(2.0, I_LND_00);
    let lnt_00 = StarAdOrder1::variable(16.0, I_LNT_00);
    let f_cell = (lnd_00 * 1.5 + lnt_00 * 0.5).exp();
    let expected_val = (2.0_f64 * 1.5 + 16.0 * 0.5).exp();
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
    assert!(model.t[model.nz - 1] > 5.0e5);
    assert!(model.peos[model.nz - 1] > model.peos[0]);
}

#[test]
fn test_henyey_jacobian_blocks_match_central_finite_differences() {
    let controls = StarControls {
        initial_mass: 4.0,
        initial_z: 0.02,
        ..StarControls::default()
    };
    let job = StarJobControls::default();
    let mut model = load_zams_model(controls, job).expect("load 4 Msun");
    model.dt = 1.0;
    set_hydro_vars(&mut model, false, false).expect("init hydro vars");
    model.save_start_values();

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
fn test_solve_henyey_step_converges_on_perturbed_model() {
    let controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        max_newton_iters: 25,
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
    model.save_start_values();

    let mut model_thomas = model.clone();
    let report_bcyclic =
        solve_henyey_step(&mut model, false, true).expect("solve bcyclic");
    assert!(
        report_bcyclic.converged,
        "expected bcyclic convergence, got {report_bcyclic:?}"
    );
    assert!(report_bcyclic.residual_norm <= model.controls.tol_residual_norm);

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

    model.save_start_values();
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
        ..StarControls::default()
    };
    let job = StarJobControls::default();
    let mut model = load_zams_model(controls, job).expect("load 4 Msun");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    clean_mixing_glitches(&mut model, 1.0, 0.1);
    let (cz_bdy_dq, boundaries) = locate_convection_boundaries(&model);
    assert_eq!(cz_bdy_dq.len(), model.nz);
    assert_eq!(boundaries.len(), 5);

    let ov_cfg = OvershootConfig::default();
    apply_overshooting(&mut model, &boundaries, &[ov_cfg]);
    let n_ov = model
        .mixing_type
        .iter()
        .filter(|&&m| m == MixingType::OvershootMixing)
        .count();
    assert!(n_ov > 0);

    compute_convection_sigmas(&mut model, 1.0e8, 1.0, 1.0e10);
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
    assert_eq!(reports.len(), 2);
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
    // Hydrogen (lightest ion) floats upward (v_gt[0] > 0), heavier ions sink (v_gt[1], v_gt[2] < 0)
    assert!(res.v_gt[0] > 0.0);
    assert!(res.v_gt[1] < 0.0);
    assert!(res.v_gt[2] < 0.0);
    // Exact mass flux conservation: sum(X_i * v_gt_i) == 0
    let mass_flux: f64 =
        x.iter().zip(res.v_gt.iter()).map(|(xi, vi)| xi * vi).sum();
    assert!(mass_flux.abs() <= 1.0e-12);

    // 2. Fabry et al. (2022) / Paxton et al. (2019) Roche distortion polynomials
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
