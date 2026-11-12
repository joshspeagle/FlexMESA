// SPDX-License-Identifier: LGPL-3.0-or-later
//! Analytical and finite-difference Jacobian verification tests for Track 3
//! Phase `Fidelity-1` (`#112`: `eps_grav.f90`, `hydro_energy.f90`,
//! `hydro_momentum.f90`, `hydro_temperature.f90`, `hydro_chem_eqns.f90`, and
//! `hydro_eqns.f90`).

use flexmesa_const::{CLIGHT, CRAD, FOUR_PI};
use flexmesa_turb::MltOption;

use crate::controls::{EnergyEqnOption, StarControls, StarJobControls};
use crate::eps_grav::{
    eval_eps_grav, eval_eps_grav_lns_cell, eval_eps_grav_std_cell,
};
use crate::hydro_energy::{
    eval_face_area_time_centered_ad, eval_pvsc_cell_ad,
    setup_d_turbulent_energy_dt, setup_dl_dm, setup_sources_and_others,
};
use crate::hydro_eqns::{
    compute_x_scale, do1_chem_eqns, do1_energy_eqn, do1_momentum_eqn,
    do1_radius_eqn, do1_temperature_eqn, eval_henyey_equations, HenyeySystem,
};
use crate::model::StarModel;
use crate::star_ad::{StarAdExt, StarAdOrder1, I_LND_00, I_V_00};
use crate::zams::load_zams_model;

fn build_solar_test_model(v_flag: bool) -> StarModel {
    let mut ctrl = StarControls {
        v_flag,
        ..StarControls::default()
    };
    ctrl.initial_mass = 1.0;
    let mut model = load_zams_model(ctrl, StarJobControls::default())
        .expect("valid 1 Msun ZAMS test model");
    model.dt = 1.0e10;
    for k in 0..model.nz {
        model.lnd[k] += 1.0e-3 * ((k as f64) * 0.1).sin();
        model.ln_t[k] += 0.8e-3 * ((k as f64) * 0.1).cos();
        if v_flag {
            model.v[k] = -1.0e4 * (1.0 - model.q[k] * 0.5);
            model.v_start[k] = -0.8e4 * (1.0 - model.q[k] * 0.5);
        }
    }
    crate::hydro_vars::set_hydro_vars(&mut model, false, false)
        .expect("valid hydro vars");
    model
}

#[test]
fn test_eps_grav_time_centering_pc_blending_and_latent_heat_match_fortran() {
    let mut model = build_solar_test_model(false);
    let k = model.nz / 2;

    // 1. Standard eps_grav vs time-centered eps_grav
    let std_uncentered = eval_eps_grav_std_cell(&model, k);
    model.controls.use_time_centered_eps_grav = true;
    model.aux.cp_start[k] = model.cp[k] * 1.10;
    model.aux.grada_start[k] = model.grada[k] * 0.95;
    model.aux.chi_t_start[k] = model.chi_t[k] * 1.05;
    model.aux.chi_rho_start[k] = model.chi_rho[k] * 0.98;
    let tc = eval_eps_grav_std_cell(&model, k);

    let dlnd_dt = (model.lnd[k] - model.lnd_start[k]) / model.dt;
    let dlnt_dt = (model.ln_t[k] - model.ln_t_start[k]) / model.dt;
    let t_s = model.ln_t_start[k].exp();
    let eg_start_expected = -t_s
        * model.aux.cp_start[k]
        * ((1.0 - model.aux.grada_start[k] * model.aux.chi_t_start[k])
            * dlnt_dt
            - model.aux.grada_start[k] * model.aux.chi_rho_start[k] * dlnd_dt);
    let expected_tc = 0.5 * std_uncentered.val + 0.5 * eg_start_expected;
    assert!(
        (tc.val - expected_tc).abs() / expected_tc.abs().max(1.0) < 1.0e-12
    );

    // Verify finite-difference lnd/lnT derivatives on time-centered eps_grav
    let eps = 1.0e-6;
    let orig_lnd = model.lnd[k];
    model.lnd[k] = orig_lnd + eps;
    model.rho[k] = model.lnd[k].exp();
    crate::micro::do_eos(&mut model).expect("do_eos +eps");
    let tc_p = eval_eps_grav_std_cell(&model, k).val;
    model.lnd[k] = orig_lnd - eps;
    model.rho[k] = model.lnd[k].exp();
    crate::micro::do_eos(&mut model).expect("do_eos -eps");
    let tc_m = eval_eps_grav_std_cell(&model, k).val;
    model.lnd[k] = orig_lnd;
    model.rho[k] = orig_lnd.exp();
    crate::micro::do_eos(&mut model).expect("do_eos restore");
    let fd_dlnd = (tc_p - tc_m) / (2.0 * eps);
    assert!(
        (tc.d1[I_LND_00] - fd_dlnd).abs() / fd_dlnd.abs().max(1.0) < 1.0e-5
    );

    // 2. Latent heat terms
    model.aux.latent_ddlnrho[k] = 2.5e12;
    model.aux.latent_ddlnt[k] = -1.8e12;
    let with_lat = eval_eps_grav_std_cell(&model, k);
    let expected_lat = tc.val
        - (dlnd_dt * model.aux.latent_ddlnrho[k]
            + dlnt_dt * model.aux.latent_ddlnt[k]);
    assert!(
        (with_lat.val - expected_lat).abs() / expected_lat.abs().max(1.0)
            < 1.0e-12
    );

    // 3. PC blending across [gamma_ln_s_eps_grav_full_off, gamma_ln_s_eps_grav_full_on]
    model.gam[k] = 140.0; // midpoint of [120, 160] -> w_gam = 0.5
    model.aux.eos_frac_pc[k] = 0.8; // alfa_lns = 0.4
    model.controls.eps_grav_factor = 0.75;
    model.controls.include_composition_in_eps_grav = true;
    let base = k * model.species;
    model.aux.dln_e_dxa_for_partials[base] = 0.5;
    model.aux.dln_peos_dxa_for_partials[base] = -0.3;
    eval_eps_grav(&mut model);

    let eg_lns = eval_eps_grav_lns_cell(&model, k);
    let eg_std = eval_eps_grav_std_cell(&model, k);
    let blended_expected = eg_lns.val * 0.4 + eg_std.val * 0.6;
    assert!(
        (model.eps_grav_ad[k].val - blended_expected).abs()
            / blended_expected.abs().max(1.0)
            < 1.0e-12
    );
    assert!(model.aux.d_eps_grav_dx[base].abs() > 0.0);
}

#[test]
fn test_hydro_energy_luminosity_time_centering_and_heating_channels() {
    let mut model = build_solar_test_model(true);
    let k = 5;
    model.controls.using_velocity_time_centering = true;
    model.controls.include_l_in_velocity_time_centering = true;
    model.controls.l_theta_for_velocity_time_centering = 0.6;
    model.lum[k] = 1.2 * model.lum_start[k];
    model.lum[k + 1] = 0.9 * model.lum_start[k + 1];

    let dl_dm = setup_dl_dm(&model, k);
    let l00_exp = 0.6 * model.lum[k] + 0.4 * model.lum_start[k];
    let lp1_exp = 0.6 * model.lum[k + 1] + 0.4 * model.lum_start[k + 1];
    let dl_dm_exp = (l00_exp - lp1_exp) / model.dm[k];
    assert!((dl_dm.val - dl_dm_exp).abs() / dl_dm_exp.abs().max(1.0) < 1.0e-12);

    // Populate extra_heat, WD sedimentation, premix, phase separation, RTI, drag, and turbulent work
    model.aux.extra_heat[k] = 1.5e3;
    model.aux.d_extra_heat_dlnd[k] = 2.0e2;
    model.aux.d_extra_heat_dlnt[k] = -3.0e2;
    model.aux.eps_wd_sedimentation[k] = 4.0e2;
    model.controls.do_conv_premix = true;
    model.controls.do_premix_heating = true;
    model.aux.eps_pre_mix[k] = 2.5e2;
    model.controls.do_phase_separation = true;
    model.controls.do_phase_separation_heating = true;
    model.aux.eps_phase_separation[k] = 3.5e2;
    model.aux.irradiation_heat[k] = 1.0e2;
    model.controls.rsp2_flag = true;
    model.aux.eq_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(5.0e2, 50.0, 25.0);
    model.aux.etrb[k] = 2.0e11;
    model.aux.etrb_start[k] = 1.0e11;
    model.controls.rti_flag = true;
    model.controls.dedt_rti_diffusion_factor = 1.5;
    model.aux.sig_rti[k] = 1.0e20;
    model.aux.sig_rti[k + 1] = 1.2e20;
    model.controls.use_drag_energy = true;
    model.controls.drag_coefficient = 0.2;

    let (sources, doing_op_split) = setup_sources_and_others(&model, k);
    assert!(!doing_op_split);
    assert!(sources.val.is_finite() && sources.d1[I_V_00] != 0.0);

    let d_eturb = setup_d_turbulent_energy_dt(&model, k);
    assert!((d_eturb.val - (2.0e11 - 1.0e11) / model.dt).abs() < 1.0e-10);

    // Check TDC turbulent energy time derivative
    model.controls.rsp2_flag = false;
    model.controls.mlt_option = MltOption::Tdc;
    model.controls.tdc_include_eturb_in_energy_equation = true;
    model.mlt_vc_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(1.0e5, 1.0e4, 2.0e4);
    model.mlt_vc_ad[k + 1] =
        StarAdOrder1::from_dlnd_dlnt_00(0.8e5, 0.5e4, 1.0e4);
    model.aux.mlt_vc_old[k] = 0.9e5;
    model.aux.mlt_vc_old[k + 1] = 0.7e5;
    let d_eturb_tdc = setup_d_turbulent_energy_dt(&model, k);
    let expected_tdc =
        0.75 * ((1.0e10 + 0.64e10) - (0.81e10 + 0.49e10)) / model.dt;
    assert!(
        (d_eturb_tdc.val - expected_tdc).abs() / expected_tdc.abs().max(1.0)
            < 1.0e-12
    );
}

#[test]
fn test_hydro_energy_pressure_time_centering_pvsc_pturb_and_chem_partials() {
    let mut model = build_solar_test_model(true);
    let k = 4;
    model.controls.energy_eqn_option = EnergyEqnOption::Dedt;
    model.controls.using_velocity_time_centering = true;
    model.controls.include_p_in_velocity_time_centering = true;
    model.controls.p_theta_for_velocity_time_centering = 0.55;
    model.controls.use_pvsc_art_visc = true;
    model.controls.pvsc_cq = 2.0;
    model.controls.mlt_pturb_factor = 0.6;
    model.aux.mlt_vc_old[k] = 2.0e5;

    // Make cell k strongly compressing so Pvsc > 0
    model.v[k] = -5.0e6;
    model.v_start[k] = -5.0e6;
    model.v[k + 1] = 1.0e6;
    model.v_start[k + 1] = 1.0e6;
    let pvsc = eval_pvsc_cell_ad(&model, k);
    assert!(pvsc.val > 0.0 && pvsc.d1[I_V_00] < 0.0);

    // Time-centered spherical area formula
    let lnr_ad =
        StarAdOrder1::variable(model.ln_r[k], crate::star_ad::I_LNR_00);
    let area_tc =
        eval_face_area_time_centered_ad(lnr_ad, model.ln_r_start[k], true);
    let r = model.ln_r[k].exp();
    let rs = model.ln_r_start[k].exp();
    let expected_area = (FOUR_PI / 3.0) * (r * r + r * rs + rs * rs);
    assert!((area_tc.val - expected_area).abs() / expected_area < 1.0e-12);

    // Populate EOS composition partial derivatives and check Jacobian entries
    let base = k * model.species;
    model.aux.dln_e_dxa_for_partials[base] = 0.85;
    model.aux.dln_peos_dxa_for_partials[base] = -0.42;

    let nvar = model.layout.nvar(true);
    let x_scale = compute_x_scale(&model, true);
    let mut sys = HenyeySystem::new(model.nz, nvar, x_scale.clone());
    do1_energy_eqn(&model, k, true, &mut sys);

    let i_equ = model.layout.i_dlne_dt;
    let j_chem0 = model.layout.chem_var_index(0);
    let dblk_chem = sys.dblk[k * nvar * nvar + i_equ * nvar + j_chem0]
        / x_scale[k * nvar + j_chem0];

    let scal = crate::hydro_energy::set_energy_eqn_scal(&model, k);
    let d_inv_rho_dt =
        ((-model.lnd[k]).exp() - (-model.lnd_start[k]).exp()) / model.dt;
    let expected_d_chem = scal * model.d_epsnuc_dx[base]
        - scal * (model.energy[k] / model.dt) * 0.85
        - scal * 0.55 * model.peos[k] * d_inv_rho_dt * (-0.42);
    assert!(
        (dblk_chem - expected_d_chem).abs()
            / expected_d_chem.abs().max(1.0e-20)
            < 1.0e-12
    );
}

#[test]
fn test_hydro_momentum_and_radius_rotational_fp_drag_pvsc_rti_and_bounds() {
    let mut model = build_solar_test_model(true);
    let k = 4;
    model.controls.rotation_flag = true;
    model.controls.use_gravity_rotation_correction = true;
    model.aux.fp_rot[k] = 0.88;
    model.controls.use_mass_corrections = true;
    model.aux.mass_correction[k] = 1.002;
    model.aux.mass_correction[k - 1] = 1.001;
    model.aux.extra_grav[k] = StarAdOrder1::constant(-1.0e2);
    model.aux.uq_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(25.0, 2.0, -1.0);
    model.controls.drag_coefficient = 0.15;
    model.controls.use_pvsc_art_visc = true;
    model.controls.mlt_pturb_factor = 0.5;
    model.aux.mlt_vc_old[k] = 1.5e5;
    model.aux.mlt_vc_old[k - 1] = 1.2e5;
    model.controls.rti_flag = true;
    model.controls.dudt_rti_diffusion_factor = 1.2;
    model.controls.dlnddt_rti_diffusion_factor = 1.0;
    model.aux.sigmid_rti[k - 1] = 1.0e22;
    model.aux.sigmid_rti[k] = 1.5e22;
    model.aux.eta_rti[k] = 1.0e14;

    let nvar = model.layout.nvar(false);
    let x_scale = compute_x_scale(&model, false);
    let mut sys = HenyeySystem::new(model.nz, nvar, x_scale.clone());
    do1_momentum_eqn(&model, k, &mut sys);
    do1_radius_eqn(&model, k, &mut sys);

    // Finite-difference check on `do1_momentum_eqn` w.r.t. `v[k]`
    let i_dv = model.layout.i_dv_dt;
    let iv = model.layout.i_v.expect("v_flag enabled");
    let analytic_dv =
        sys.dblk[k * nvar * nvar + i_dv * nvar + iv] / x_scale[k * nvar + iv];

    let eps = 1.0e-3;
    let v_orig = model.v[k];
    model.v[k] = v_orig + eps;
    let mut sys_p = HenyeySystem::new(model.nz, nvar, x_scale.clone());
    do1_momentum_eqn(&model, k, &mut sys_p);
    model.v[k] = v_orig - eps;
    let mut sys_m = HenyeySystem::new(model.nz, nvar, x_scale.clone());
    do1_momentum_eqn(&model, k, &mut sys_m);
    model.v[k] = v_orig;
    let fd_dv =
        (sys_p.equ[k * nvar + i_dv] - sys_m.equ[k * nvar + i_dv]) / (2.0 * eps);
    assert!((analytic_dv - fd_dv).abs() / fd_dv.abs().max(1.0e-12) < 1.0e-6);

    // Check `force_zero_v` bound in `do1_radius_eqn`
    model.controls.velocity_q_upper_bound = 0.5; // q[0] = 1.0 > 0.5 -> force_zero_v
    let mut sys_zero = HenyeySystem::new(model.nz, nvar, x_scale);
    do1_radius_eqn(&model, 0, &mut sys_zero);
    let i_dlnr = model.layout.i_dlnr_dt.unwrap();
    let expected_zero = -model.v[0] / model.csound[0].max(1.0);
    assert!((sys_zero.equ[i_dlnr] - expected_zero).abs() < 1.0e-12);
}

#[test]
fn test_hydro_temperature_gradt_dprad_lp_limiter_and_rsp_lsurf_bc() {
    let mut model = build_solar_test_model(false);
    let k = 3;
    let nvar = model.layout.nvar(false);
    let x_scale = compute_x_scale(&model, false);

    // 1. RSP Lsurf BC at k = 0
    model.controls.use_rsp_l_eqn_outer_bc = true;
    let mut sys_rsp = HenyeySystem::new(model.nz, nvar, x_scale.clone());
    do1_temperature_eqn(&model, 0, &mut sys_rsp);
    let r_0 = model.ln_r[0].exp();
    let t_0 = model.ln_t[0].exp();
    let sigma_sb = 0.25 * CLIGHT * CRAD;
    let l_rad = 0.5 * FOUR_PI * sigma_sb * r_0 * r_0 * t_0.powi(4);
    let l_scale = model.lum_start[0].abs().max(flexmesa_const::L_SUN);
    let expected_rsp = (model.lum[0] - l_rad) / l_scale;
    assert!((sys_rsp.equ[model.layout.i_equ_l] - expected_rsp).abs() < 1.0e-12);

    // 2. Direct gradT form at k = 3
    model
        .controls
        .use_gradt_actual_vs_gradt_mlt_for_t_gradient_eqn = true;
    let mut sys_gt = HenyeySystem::new(model.nz, nvar, x_scale.clone());
    do1_temperature_eqn(&model, k, &mut sys_gt);
    assert!(sys_gt.equ[k * nvar + model.layout.i_equ_l].is_finite());

    // 3. dPrad/dm with Levermore-Pomraning flux limiter at k = 3 (including optically
    // thin flux-limited regime where f_lp > 1.0 depends on ln_t[k])
    model
        .controls
        .use_gradt_actual_vs_gradt_mlt_for_t_gradient_eqn = false;
    model.controls.use_dprad_dm_form_of_t_gradient_eqn = true;
    model.controls.use_flux_limiting_with_dprad_dm_form = true;
    model.dm_bar[k] = 1.0e15;
    model.d_opacity_dlnt[k] = 0.0;
    model.gradt_ad[k] = StarAdOrder1::constant(model.gradt[k]);
    model.gradr_ad[k] = StarAdOrder1::constant(model.gradr[k]);
    let mut sys_lp = HenyeySystem::new(model.nz, nvar, x_scale.clone());
    do1_temperature_eqn(&model, k, &mut sys_lp);
    let i_equ = model.layout.i_equ_l;
    let i_lum = model.layout.i_lum;
    let i_lnt = model.layout.i_lnt;
    let analytic_dl = sys_lp.dblk[k * nvar * nvar + i_equ * nvar + i_lum]
        / x_scale[k * nvar + i_lum];
    let analytic_dlnt = sys_lp.dblk[k * nvar * nvar + i_equ * nvar + i_lnt]
        / x_scale[k * nvar + i_lnt];

    let eps = model.lum[k].abs().max(1.0) * 1.0e-6;
    let l_orig = model.lum[k];
    model.lum[k] = l_orig + eps;
    let mut sys_p = HenyeySystem::new(model.nz, nvar, x_scale.clone());
    do1_temperature_eqn(&model, k, &mut sys_p);
    model.lum[k] = l_orig - eps;
    let mut sys_m = HenyeySystem::new(model.nz, nvar, x_scale.clone());
    do1_temperature_eqn(&model, k, &mut sys_m);
    model.lum[k] = l_orig;
    let fd_dl = (sys_p.equ[k * nvar + i_equ] - sys_m.equ[k * nvar + i_equ])
        / (2.0 * eps);
    assert!((analytic_dl - fd_dl).abs() / fd_dl.abs().max(1.0e-30) < 1.0e-7);

    let eps_t = 1.0e-6;
    let lnt_orig = model.ln_t[k];
    model.ln_t[k] = lnt_orig + eps_t;
    let mut sys_tp = HenyeySystem::new(model.nz, nvar, x_scale.clone());
    do1_temperature_eqn(&model, k, &mut sys_tp);
    model.ln_t[k] = lnt_orig - eps_t;
    let mut sys_tm = HenyeySystem::new(model.nz, nvar, x_scale);
    do1_temperature_eqn(&model, k, &mut sys_tm);
    model.ln_t[k] = lnt_orig;
    let fd_dlnt = (sys_tp.equ[k * nvar + i_equ] - sys_tm.equ[k * nvar + i_equ])
        / (2.0 * eps_t);
    assert!(
        (analytic_dlnt - fd_dlnt).abs() / fd_dlnt.abs().max(1.0e-30) < 1.0e-6
    );
}

#[test]
fn test_hydro_chem_eqns_op_split_do_burn_do_mix_and_mix_factor() {
    let mut model = build_solar_test_model(false);
    let k = model.nz - 2;
    let nvar = model.layout.nvar(true);
    let x_scale = compute_x_scale(&model, true);

    // 1. Coupled burn active (`op_split_burn = false`)
    model.controls.op_split_burn = false;
    model.controls.do_burn = true;
    model.controls.do_mix = true;
    model.controls.mix_factor = 2.0;
    model.sig[k] = 1.0e22;
    model.dxdt_nuc[k * model.species] = -1.0e-14;
    model.d_dxdt_nuc_dlnt[k * model.species] = -5.0e-14;

    let mut sys_coupled = HenyeySystem::new(model.nz, nvar, x_scale.clone());
    do1_chem_eqns(&model, k, &mut sys_coupled);
    let i_chem0 = model.layout.chem_var_index(0);
    let d_lnt_coupled =
        sys_coupled.dblk[k * nvar * nvar + i_chem0 * nvar + model.layout.i_lnt];
    assert!(d_lnt_coupled != 0.0);

    // 2. Operator-split burn active (`op_split_burn = true`, `t_start >= op_split_burn_min_t`)
    model.controls.op_split_burn = true;
    model.controls.op_split_burn_min_t = 1.0e5;
    let mut sys_opsplit = HenyeySystem::new(model.nz, nvar, x_scale);
    do1_chem_eqns(&model, k, &mut sys_opsplit);
    let d_lnt_opsplit =
        sys_opsplit.dblk[k * nvar * nvar + i_chem0 * nvar + model.layout.i_lnt];
    assert_eq!(d_lnt_opsplit, 0.0);

    // Verify full system assembly still succeeds cleanly
    let full = eval_henyey_equations(&mut model, true)
        .expect("eval_henyey_equations succeeds");
    assert_eq!(full.equ.len(), model.nz * nvar);
}
