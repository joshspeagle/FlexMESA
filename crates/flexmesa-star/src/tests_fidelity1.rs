// SPDX-License-Identifier: GPL-3.0-or-later
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
        zbase: 0.02,
        ..StarControls::default()
    };
    ctrl.initial_mass = 1.0;
    let mut model = load_zams_model(ctrl, StarJobControls::default())
        .expect("valid 1 Msun ZAMS test model");
    model.dt = 1.0e10;
    for k in 0..model.nz {
        model.lnd[k] += 1.0e-3 * flexmesa_math::sin((k as f64) * 0.1);
        model.ln_t[k] += 0.8e-3 * flexmesa_math::cos((k as f64) * 0.1);
        if v_flag {
            model.v[k] = -1.0e4 * (1.0 - model.q[k] * 0.5);
            model.v_start[k] = -0.8e4 * (1.0 - model.q[k] * 0.5);
        }
    }
    crate::hydro_vars::set_hydro_vars(&mut model, false, false)
        .expect("valid hydro vars");
    crate::evolve_support::record_start_of_step_energy_integrals(&mut model);
    sync_dxh_with_start(&mut model);
    model
}

/// Sets `dxh_lnd` and `dxh_lnT` to `lnd - lnd_start` and `lnT - lnT_start`,
/// standing in for the accumulated Newton corrections that
/// `set_vars_for_solver` stores (`star/private/solver_support.f90:1142,
/// 1328`) when a test moves `lnd` or `lnT` directly.
fn sync_dxh_with_start(model: &mut StarModel) {
    for k in 0..model.nz {
        model.aux.dxh_lnd[k] = model.lnd[k] - model.lnd_start[k];
        model.aux.dxh_lnt[k] = model.ln_t[k] - model.ln_t_start[k];
    }
}

#[test]
fn test_eps_grav_time_centering_pc_blending_and_latent_heat_match_fortran() {
    let mut model = build_solar_test_model(false);
    let k = model.nz / 2;

    // 1. Standard uncentered eps_grav vs default time-centered eps_grav
    model.controls.use_time_centered_eps_grav = false;
    let std_uncentered =
        eval_eps_grav_std_cell(&model, k).expect("eval_eps_grav_std_cell");
    model.controls.use_time_centered_eps_grav = true;
    model.aux.cp_start[k] = model.cp[k] * 1.10;
    model.aux.grada_start[k] = model.grada[k] * 0.95;
    model.aux.chi_t_start[k] = model.chi_t[k] * 1.05;
    model.aux.chi_rho_start[k] = model.chi_rho[k] * 0.98;
    let tc =
        eval_eps_grav_std_cell(&model, k).expect("eval_eps_grav_std_cell tc");

    let dlnd_dt = (model.lnd[k] - model.lnd_start[k]) / model.dt;
    let dlnt_dt = (model.ln_t[k] - model.ln_t_start[k]) / model.dt;
    let t_s = flexmesa_math::exp(model.ln_t_start[k]);
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
    model.rho[k] = flexmesa_math::exp(model.lnd[k]);
    sync_dxh_with_start(&mut model);
    crate::micro::do_eos(&mut model).expect("do_eos +eps");
    let tc_p = eval_eps_grav_std_cell(&model, k)
        .expect("eval_eps_grav_std_cell +eps")
        .val;
    model.lnd[k] = orig_lnd - eps;
    model.rho[k] = flexmesa_math::exp(model.lnd[k]);
    sync_dxh_with_start(&mut model);
    crate::micro::do_eos(&mut model).expect("do_eos -eps");
    let tc_m = eval_eps_grav_std_cell(&model, k)
        .expect("eval_eps_grav_std_cell -eps")
        .val;
    model.lnd[k] = orig_lnd;
    model.rho[k] = flexmesa_math::exp(orig_lnd);
    sync_dxh_with_start(&mut model);
    crate::micro::do_eos(&mut model).expect("do_eos restore");
    let fd_dlnd = (tc_p - tc_m) / (2.0 * eps);
    assert!(
        (tc.d1[I_LND_00] - fd_dlnd).abs() / fd_dlnd.abs().max(1.0) < 1.0e-5
    );

    // 2. Latent heat terms: the current-step term enters the centred
    // `eps_grav` and the start-of-step term enters `eps_grav_start`, so each
    // carries the weight 1/2 (`star/private/eps_grav.f90:165-182`).
    model.aux.latent_ddlnrho[k] = 2.5e12;
    model.aux.latent_ddlnt[k] = -1.8e12;
    model.aux.latent_ddlnrho_start[k] = 1.5e12;
    model.aux.latent_ddlnt_start[k] = -0.6e12;
    let with_lat =
        eval_eps_grav_std_cell(&model, k).expect("eval_eps_grav_std_cell lat");
    let expected_lat = tc.val
        - 0.5
            * (dlnd_dt * model.aux.latent_ddlnrho[k]
                + dlnt_dt * model.aux.latent_ddlnt[k])
        - 0.5
            * (dlnd_dt * model.aux.latent_ddlnrho_start[k]
                + dlnt_dt * model.aux.latent_ddlnt_start[k]);
    assert!(
        (with_lat.val - expected_lat).abs() / expected_lat.abs().max(1.0)
            < 1.0e-12
    );

    // 3. PC blending across [gamma_ln_s_eps_grav_full_off, gamma_ln_s_eps_grav_full_on] = [120, 150]
    model.gam[k] = 135.0; // midpoint of [120, 150] -> w_gam = 0.5
    model.aux.eos_frac_pc[k] = 0.8; // alfa_lns = 0.4
    model.controls.eps_grav_factor = 0.75;
    model.controls.include_composition_in_eps_grav = true;
    let base = k * model.species;
    model.aux.dln_e_dxa_for_partials[base] = 0.5;
    model.aux.dln_peos_dxa_for_partials[base] = -0.3;
    eval_eps_grav(&mut model).expect("eval_eps_grav");

    let eg_lns = eval_eps_grav_lns_cell(&model, k);
    let eg_std =
        eval_eps_grav_std_cell(&model, k).expect("eval_eps_grav_std_cell");
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

    let (sources, _others, doing_op_split) =
        setup_sources_and_others(&model, k);
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

    // Subsonic compression stays below `pvsc_zshock = 0.1`, while supersonic shock triggers Pvsc > 0
    model.v[k] = -1.0e4;
    model.v_start[k] = -1.0e4;
    model.v[k + 1] = 0.0;
    model.v_start[k + 1] = 0.0;
    assert_eq!(eval_pvsc_cell_ad(&model, k).val, 0.0);

    let dr_k = (model.r[k] - model.r[k + 1]).max(1.0);
    let v_shock = -0.25 * model.csound[k] * (model.r[k] / dr_k);
    model.v[k] = v_shock;
    model.v_start[k] = v_shock;
    model.v[k + 1] = 0.0;
    model.v_start[k + 1] = 0.0;
    let pvsc = eval_pvsc_cell_ad(&model, k);
    let dv_exp = (model.v[k + 1] - model.v[k])
        - model.controls.pvsc_zshock * (model.peos[k] / model.rho[k]).sqrt();
    let pvsc_exp = model.controls.pvsc_cq * model.rho[k] * dv_exp * dv_exp;
    assert!((pvsc.val - pvsc_exp).abs() / pvsc_exp < 1.0e-14);
    assert!(pvsc.d1[I_V_00] < 0.0);

    // Time-centered spherical area formula
    let lnr_ad =
        StarAdOrder1::variable(model.ln_r[k], crate::star_ad::I_LNR_00);
    let area_tc =
        eval_face_area_time_centered_ad(lnr_ad, model.ln_r_start[k], true);
    let r = flexmesa_math::exp(model.ln_r[k]);
    let rs = flexmesa_math::exp(model.ln_r_start[k]);
    let expected_area = (FOUR_PI / 3.0) * (r * r + r * rs + rs * rs);
    assert!((area_tc.val - expected_area).abs() / expected_area < 1.0e-12);

    // Populate EOS composition partial derivatives and check spherical `eval_dwork_dm`
    // tridiagonal Jacobian entries (`lblk`, `dblk`, `ublk`) when `use_p_d_1_div_rho... == false`
    let base = k * model.species;
    let base_m1 = (k - 1) * model.species;
    let base_p1 = (k + 1) * model.species;
    model.aux.dln_e_dxa_for_partials[base] = 0.85;
    model.aux.dln_peos_dxa_for_partials[base_m1] = -0.31;
    model.aux.dln_peos_dxa_for_partials[base] = -0.42;
    model.aux.dln_peos_dxa_for_partials[base_p1] = -0.53;

    let nvar = model.layout.nvar(true);
    let x_scale = compute_x_scale(&model, true);
    let mut sys = HenyeySystem::new(model.nz, nvar, x_scale.clone());
    do1_energy_eqn(&model, k, true, &mut sys).expect("energy eqn");

    let i_equ = model.layout.i_dlne_dt;
    let j_chem0 = model.layout.chem_var_index(0);
    let dblk_chem = sys.dblk[k * nvar * nvar + i_equ * nvar + j_chem0]
        / x_scale[k * nvar + j_chem0];
    let lblk_chem = sys.lblk[k * nvar * nvar + i_equ * nvar + j_chem0]
        / x_scale[(k - 1) * nvar + j_chem0];
    let ublk_chem = sys.ublk[k * nvar * nvar + i_equ * nvar + j_chem0]
        / x_scale[(k + 1) * nvar + j_chem0];

    let scal = crate::hydro_energy::set_energy_eqn_scal(&model, k)
        .expect("energy scal");
    // `unpack_res18` (`star/private/hydro_energy.f90:469-527`); the
    // `dlnPeos_dxa` values set above (-0.42, -0.31, -0.53) enter through
    // `d_dwork_dxa*`.
    let (_, dxa) =
        crate::hydro_work::eval_v_flag_dwork_dm_ad(&model, k, false, true);
    let dm = model.dm[k];
    let expected_d00 = scal * model.d_epsnuc_dx[base]
        + -(scal * (model.energy[k] / model.dt) * 0.85)
        + -(scal * dxa.dxa00[0] / dm);
    let expected_dm1 = -(scal * dxa.dxam1[0] / dm);
    let expected_dp1 = -(scal * dxa.dxap1[0] / dm);
    assert!(
        (dblk_chem - expected_d00).abs() / expected_d00.abs().max(1.0e-20)
            < 1.0e-12
    );
    assert!(
        (lblk_chem - expected_dm1).abs() / expected_dm1.abs().max(1.0e-20)
            < 1.0e-12
    );
    assert!(
        (ublk_chem - expected_dp1).abs() / expected_dp1.abs().max(1.0e-20)
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
    // `resid_ad = v00/s% csound_start(k)` (`star/private/hydro_momentum.f90:565-574`).
    let expected_zero = model.v[0] / model.aux.csound_start[0];
    assert!((sys_zero.equ[i_dlnr] - expected_zero).abs() < 1.0e-12);
}

#[test]
fn test_hydro_temperature_gradt_dprad_lp_limiter_and_rsp_lsurf_bc() {
    let mut model = build_solar_test_model(false);
    let k = 3;
    let nvar = model.layout.nvar(false);
    let x_scale = compute_x_scale(&model, false);

    // 1. RSP Lsurf BC at k = 0 (`star/private/hydro_temperature.f90:358,369`)
    model.controls.use_rsp_l_eqn_outer_bc = true;
    let mut sys_rsp = HenyeySystem::new(model.nz, nvar, x_scale.clone());
    do1_temperature_eqn(&model, 0, &mut sys_rsp);
    let r_0 = flexmesa_math::exp(model.ln_r[0]);
    let t_0 = flexmesa_math::exp(model.ln_t[0]);
    let l_rad = model.controls.rsp2.lsurf_factor
        * FOUR_PI
        * r_0
        * r_0
        * CLIGHT
        * CRAD
        * flexmesa_math::pow4(t_0);
    let max_l_start = model
        .lum_start
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let l_scale = if max_l_start > 0.0 && max_l_start.is_finite() {
        max_l_start
    } else {
        flexmesa_const::L_SUN
    };
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

#[test]
fn test_eval_eps_grav_composition_finite_difference_partials_and_d_eps_grav_dx()
{
    use crate::star_ad::I_LNT_00;
    use flexmesa_const::LN10;
    use flexmesa_eos::{eos_dt_get_zones_with_logs, HelmTable, I_LN_E};

    let mut model = build_solar_test_model(false);
    // Use pure HELM so smooth analytic EOS partials support tight FD checks.
    model.eos_config.use_free_eos = false;
    model.eos_config.use_opal_scvh = false;
    model.eos_config.use_skye = false;
    model.eos_config.use_pc = false;
    model.eos_config.use_cms = false;
    crate::micro::set_basic_vars(&mut model).expect("set_basic_vars");
    crate::micro::do_eos(&mut model).expect("do_eos");
    model.save_start_values().expect("save_start_values");
    crate::evolve_support::record_start_of_step_energy_integrals(&mut model);

    let k = model.nz / 2;
    let sp = model.species;
    let base = k * sp;
    let i_h1 = model
        .iso_names
        .iter()
        .position(|n| n == "h1")
        .expect("h1 species");
    let i_he4 = model
        .iso_names
        .iter()
        .position(|n| n == "he4")
        .expect("he4 species");

    // Shift density, temperature, and convert 5% H1 -> He4 (burning heats).
    model.lnd[k] += 0.02;
    model.ln_t[k] += 0.015;
    sync_dxh_with_start(&mut model);
    model.xa[base + i_h1] -= 0.05;
    model.xa[base + i_he4] += 0.05;
    crate::micro::set_basic_vars(&mut model).expect("set_basic_vars pert");
    crate::micro::do_eos(&mut model).expect("do_eos pert");

    // Compare against direct `eos_dt_get` evaluations for both theta = 1.0
    // (uncentered) and theta = 0.5 (time-centered).
    let ht = HelmTable::builtin().expect("builtin HELM table");
    let xa_start_k = model.xa_start[base..base + sp].to_vec();
    let xa_k = model.xa[base..base + sp].to_vec();
    let e_xa_start = flexmesa_math::exp(
        eos_dt_get_zones_with_logs(
            ht,
            &[model.rho[k]],
            &[model.lnd[k] / LN10],
            &[model.t[k]],
            &[model.ln_t[k] / LN10],
            &model.chem_id,
            &xa_start_k,
            &model.eos_config,
        )
        .expect("eos at xa_start")[0]
            .res[I_LN_E],
    );
    let rho_s = flexmesa_math::exp(model.lnd_start[k]);
    let t_s = flexmesa_math::exp(model.ln_t_start[k]);
    let e_dt_start = flexmesa_math::exp(
        eos_dt_get_zones_with_logs(
            ht,
            &[rho_s],
            &[model.lnd_start[k] / LN10],
            &[t_s],
            &[model.ln_t_start[k] / LN10],
            &model.chem_id,
            &xa_k,
            &model.eos_config,
        )
        .expect("eos at DT_start")[0]
            .res[I_LN_E],
    );

    let de_uncentered = model.energy[k] - e_xa_start;
    let de_centered =
        0.5 * de_uncentered + 0.5 * (e_dt_start - model.energy_start[k]);
    // Converting H1 -> He4 lowers internal energy per gram (`de < 0`),
    // releasing positive composition heating (`-de / dt > 0`).
    assert!(de_uncentered < 0.0);
    assert!(de_centered < 0.0);

    for (time_centered, expected_comp) in [
        (false, -de_uncentered / model.dt),
        (true, -de_centered / model.dt),
    ] {
        model.controls.use_time_centered_eps_grav = time_centered;
        model.controls.include_composition_in_eps_grav = false;
        let eg_no_comp = eval_eps_grav_std_cell(&model, k)
            .expect("eval_eps_grav_std_cell without comp");

        model.controls.include_composition_in_eps_grav = true;
        let eg_with_comp = eval_eps_grav_std_cell(&model, k)
            .expect("eval_eps_grav_std_cell with comp");
        let comp_ad = eg_with_comp - eg_no_comp;

        assert!(
            comp_ad.val > 0.0,
            "H->He composition heating must be positive (tc={time_centered})"
        );
        assert!(
            (comp_ad.val - expected_comp).abs() / expected_comp.abs().max(1.0)
                < 1.0e-12,
            "comp val mismatch (tc={time_centered}): got {}, expected {}",
            comp_ad.val,
            expected_comp
        );

        // Central finite-difference verification of d(comp_term)/d(lnd) and
        // d(comp_term)/d(lnT).
        let eps = 1.0e-6;
        let eval_comp_val = |m: &mut StarModel| -> f64 {
            crate::micro::set_basic_vars(m).expect("basic");
            crate::micro::do_eos(m).expect("eos");
            m.controls.include_composition_in_eps_grav = false;
            let a = eval_eps_grav_std_cell(m, k).expect("no comp").val;
            m.controls.include_composition_in_eps_grav = true;
            let b = eval_eps_grav_std_cell(m, k).expect("with comp").val;
            b - a
        };

        let mut mp = model.clone();
        let mut mm = model.clone();
        mp.lnd[k] += eps;
        mm.lnd[k] -= eps;
        let fd_dlnd =
            (eval_comp_val(&mut mp) - eval_comp_val(&mut mm)) / (2.0 * eps);
        assert!(
            (comp_ad.d1[I_LND_00] - fd_dlnd).abs() / fd_dlnd.abs().max(1.0e-12)
                < 1.0e-5,
            "comp d/dlnd mismatch (tc={time_centered}): an={}, fd={}",
            comp_ad.d1[I_LND_00],
            fd_dlnd
        );

        let mut tp = model.clone();
        let mut tm = model.clone();
        tp.ln_t[k] += eps;
        tm.ln_t[k] -= eps;
        let fd_dlnt =
            (eval_comp_val(&mut tp) - eval_comp_val(&mut tm)) / (2.0 * eps);
        assert!(
            (comp_ad.d1[I_LNT_00] - fd_dlnt).abs() / fd_dlnt.abs().max(1.0e-12)
                < 1.0e-5,
            "comp d/dlnT mismatch (tc={time_centered}): an={}, fd={}",
            comp_ad.d1[I_LNT_00],
            fd_dlnt
        );
    }

    // Reverse composition change (He4 -> H1) must produce negative heating.
    model.xa[base + i_h1] = model.xa_start[base + i_h1] + 0.05;
    model.xa[base + i_he4] = model.xa_start[base + i_he4] - 0.05;
    crate::micro::set_basic_vars(&mut model).expect("set_basic_vars rev");
    crate::micro::do_eos(&mut model).expect("do_eos rev");
    model.controls.include_composition_in_eps_grav = false;
    let eg_rev_no = eval_eps_grav_std_cell(&model, k).expect("rev no comp");
    model.controls.include_composition_in_eps_grav = true;
    let eg_rev_yes = eval_eps_grav_std_cell(&model, k).expect("rev with comp");
    assert!(eg_rev_yes.val - eg_rev_no.val < 0.0);

    // Verify exact `d_eps_grav_dx` formula (`eps_grav.f90:302-305`) and that
    // pure `do_lnS_eps_grav` zeroes `d_eps_grav_dx`.
    eval_eps_grav(&mut model).expect("eval_eps_grav std");
    let dxh_lnd = model.lnd[k] - model.lnd_start[k];
    for j in 0..sp {
        let expected = -model.energy[k]
            * model.aux.dln_e_dxa_for_partials[base + j]
            / model.dt
            + (model.peos[k] / model.rho[k])
                * model.aux.dln_peos_dxa_for_partials[base + j]
                * dxh_lnd
                / model.dt;
        assert!(
            (model.aux.d_eps_grav_dx[base + j] - expected).abs()
                <= 1.0e-14 * expected.abs().max(1.0),
            "d_eps_grav_dx[{j}] mismatch: got {}, expected {}",
            model.aux.d_eps_grav_dx[base + j],
            expected
        );
    }

    model.aux.eos_frac_pc[k] = 1.0;
    model.gam[k] = model.controls.gamma_ln_s_eps_grav_full_on + 10.0;
    eval_eps_grav(&mut model).expect("eval_eps_grav lns");
    assert!(
        model.aux.d_eps_grav_dx[base..base + sp]
            .iter()
            .all(|&v| v == 0.0),
        "pure do_lnS_eps_grav must zero d_eps_grav_dx"
    );
}

#[test]
fn test_fix_d_eos_dxa_partials_populates_skye_ideal_and_skips_below_threshold_or_helm(
) {
    use flexmesa_const::LN10;
    use flexmesa_eos::{
        eos_dt_get_zones_with_logs, HelmTable, I_FRAC_IDEAL, I_FRAC_PC,
        I_FRAC_SKYE, I_LN_E, I_LN_PGAS,
    };

    let mut model = build_solar_test_model(false);
    // Force Skye EOS (with HELM fallback outside Skye's domain) and place one
    // interior zone squarely inside Skye's dense plasma domain (`logRho = 5`,
    // `logT = 7.5`) so `eos_frac_Skye == 1.0` and raw `do_eos` yields zero
    // analytic composition partials.
    model.eos_config.use_free_eos = false;
    model.eos_config.use_opal_scvh = false;
    model.eos_config.use_cms = false;
    model.eos_config.use_pc = false;
    model.eos_config.use_skye = true;
    let k = model.nz / 2;
    model.lnd[k] = 5.0 * LN10;
    model.ln_t[k] = 7.5 * LN10;
    crate::micro::set_basic_vars(&mut model).expect("set_basic_vars skye");
    crate::micro::do_eos(&mut model).expect("do_eos skye");
    model.save_start_values().expect("save_start_values skye");
    crate::evolve_support::record_start_of_step_energy_integrals(&mut model);

    let sp = model.species;
    let i_h1 = model
        .iso_names
        .iter()
        .position(|n| n == "h1")
        .expect("h1 species");
    let i_he4 = model
        .iso_names
        .iter()
        .position(|n| n == "he4")
        .expect("he4 species");

    let base = k * sp;
    let frac = model.eos_res[k][I_FRAC_PC]
        + model.eos_res[k][I_FRAC_SKYE]
        + model.eos_res[k][I_FRAC_IDEAL];
    assert!(
        frac > 0.99,
        "expected Skye/PC/ideal fraction > 0.99 at logRho=5, logT=7.5, got {frac}"
    );

    // Sub-threshold change (`5e-5 < 1e-4`) must be skipped.
    model.xa[base + i_h1] -= 5.0e-5;
    model.xa[base + i_he4] += 5.0e-5;
    crate::micro::set_basic_vars(&mut model).expect("basic sub-thresh");
    crate::micro::do_eos(&mut model).expect("eos sub-thresh");
    let raw_dlne = model.aux.dln_e_dxa_for_partials[base + i_h1];
    let raw_dlnp = model.aux.dln_peos_dxa_for_partials[base + i_h1];
    assert_eq!(raw_dlne, 0.0);
    assert_eq!(raw_dlnp, 0.0);
    crate::micro::fix_d_eos_dxa_partials(&mut model)
        .expect("fix_d_eos_dxa_partials sub-thresh");
    assert_eq!(model.aux.dln_e_dxa_for_partials[base + i_h1], 0.0);
    assert_eq!(model.aux.dln_peos_dxa_for_partials[base + i_h1], 0.0);

    // Above-threshold change (`dxa = -0.02` on h1, `+0.02` on he4) must
    // populate exact finite-difference partials (`hydro_eqns.f90:480-485`).
    let dxa_h1 = -0.02;
    let dxa_he4 = 0.02;
    model.xa[base + i_h1] = model.xa_start[base + i_h1] + dxa_h1;
    model.xa[base + i_he4] = model.xa_start[base + i_he4] + dxa_he4;
    crate::micro::set_basic_vars(&mut model).expect("basic active");
    crate::micro::do_eos(&mut model).expect("eos active");
    crate::micro::fix_d_eos_dxa_partials(&mut model)
        .expect("fix_d_eos_dxa_partials active");

    let ht = HelmTable::builtin().expect("builtin HELM");
    let xa_s = model.xa_start[base..base + sp].to_vec();
    let mut xa_s_h1 = xa_s.clone();
    xa_s_h1[i_h1] += dxa_h1;
    let out_s = &eos_dt_get_zones_with_logs(
        ht,
        &[model.rho[k]],
        &[model.lnd[k] / LN10],
        &[model.t[k]],
        &[model.ln_t[k] / LN10],
        &model.chem_id,
        &xa_s,
        &model.eos_config,
    )
    .expect("eos xa_start")[0];
    let out_h1 = &eos_dt_get_zones_with_logs(
        ht,
        &[model.rho[k]],
        &[model.lnd[k] / LN10],
        &[model.t[k]],
        &[model.ln_t[k] / LN10],
        &model.chem_id,
        &xa_s_h1,
        &model.eos_config,
    )
    .expect("eos xa_start_1")[0];
    let exp_dlne_h1 = out_h1.d_dxa[i_h1][1]
        + frac * (out_h1.res[I_LN_E] - out_s.res[I_LN_E]) / dxa_h1;
    let exp_dlnp_h1 = model.pgas[k] * out_h1.d_dxa[i_h1][0] / model.peos[k]
        + frac
            * (model.pgas[k] / model.peos[k])
            * (out_h1.res[I_LN_PGAS] - out_s.res[I_LN_PGAS])
            / dxa_h1;
    assert!(exp_dlne_h1.abs() > 1.0e-3);
    assert!(exp_dlnp_h1.abs() > 1.0e-3);
    assert!(
        (model.aux.dln_e_dxa_for_partials[base + i_h1] - exp_dlne_h1).abs()
            <= 1.0e-12 * exp_dlne_h1.abs()
    );
    assert!(
        (model.aux.dln_peos_dxa_for_partials[base + i_h1] - exp_dlnp_h1).abs()
            <= 1.0e-12 * exp_dlnp_h1.abs()
    );
    // Unchanged species (`|dxa| == 0`) remain untouched.
    for j in 0..sp {
        if j != i_h1 && j != i_he4 {
            assert_eq!(model.aux.dln_e_dxa_for_partials[base + j], 0.0);
        }
    }

    // Pure HELM zone (`frac_without_dxa == 0`) must leave analytic partials
    // bit-for-bit untouched even when `|dxa| >= 1e-4`.
    let mut helm_model = build_solar_test_model(false);
    helm_model.eos_config.use_free_eos = false;
    helm_model.eos_config.use_opal_scvh = false;
    helm_model.eos_config.use_skye = false;
    helm_model.eos_config.use_pc = false;
    helm_model.eos_config.use_cms = false;
    crate::micro::set_basic_vars(&mut helm_model).expect("basic helm");
    crate::micro::do_eos(&mut helm_model).expect("eos helm");
    helm_model.save_start_values().expect("save_start helm");
    crate::evolve_support::record_start_of_step_energy_integrals(
        &mut helm_model,
    );
    helm_model.xa[base + i_h1] -= 0.02;
    helm_model.xa[base + i_he4] += 0.02;
    crate::micro::set_basic_vars(&mut helm_model).expect("basic helm pert");
    crate::micro::do_eos(&mut helm_model).expect("eos helm pert");
    let before_e = helm_model.aux.dln_e_dxa_for_partials.clone();
    let before_p = helm_model.aux.dln_peos_dxa_for_partials.clone();
    crate::micro::fix_d_eos_dxa_partials(&mut helm_model)
        .expect("fix_d_eos_dxa_partials helm");
    assert_eq!(helm_model.aux.dln_e_dxa_for_partials, before_e);
    assert_eq!(helm_model.aux.dln_peos_dxa_for_partials, before_p);
}

#[test]
fn test_eval_face_dm_bar_innermost_zone_uses_symmetric_half_sum() {
    let mut model = build_solar_test_model(false);
    let nz = model.nz;
    assert!(nz >= 4);
    let k_last = nz - 1;
    // Set distinct cell masses at the innermost two zones so the cell-centered
    // `dm_bar[nz - 1] = 0.5 * dm[nz - 2] + dm[nz - 1]` differs strictly from
    // the momentum-face `0.5 * (dm[nz - 2] + dm[nz - 1])` (`hydro_momentum.f90:191-196`).
    model.dm[k_last - 1] = 3.0e30;
    model.dm[k_last] = 1.0e30;
    model.dm_bar[k_last] = 0.5 * model.dm[k_last - 1] + model.dm[k_last];
    model.controls.use_mass_corrections = false;

    let expected_last = 0.5 * (model.dm[k_last - 1] + model.dm[k_last]);
    let got_last = crate::hydro_momentum::eval_face_dm_bar(&model, k_last);
    assert!(
        (got_last - expected_last).abs() <= 1.0e-15 * expected_last,
        "eval_face_dm_bar(nz - 1) must be 0.5 * (dm[nz-2] + dm[nz-1]) = {expected_last}, got {got_last}"
    );
    assert!(
        (got_last - model.dm_bar[k_last]).abs() > 0.1 * expected_last,
        "face dm_bar at k = nz - 1 must differ from cell-centered dm_bar[nz - 1]"
    );

    // Surface face (`k = 0`) and interior face (`k = nz / 2`).
    let got_surf = crate::hydro_momentum::eval_face_dm_bar(&model, 0);
    assert!((got_surf - 0.5 * model.dm[0]).abs() <= 1.0e-15 * model.dm[0]);
    let k_mid = nz / 2;
    let exp_mid = 0.5 * (model.dm[k_mid - 1] + model.dm[k_mid]);
    let got_mid = crate::hydro_momentum::eval_face_dm_bar(&model, k_mid);
    assert!((got_mid - exp_mid).abs() <= 1.0e-15 * exp_mid);

    // With `use_mass_corrections = true`: `0.5 * (mc[k-1]*dm[k-1] + mc[k]*dm[k])`.
    model.controls.use_mass_corrections = true;
    model.aux.mass_correction.resize(nz, 1.0);
    model.aux.mass_correction[k_last - 1] = 1.02;
    model.aux.mass_correction[k_last] = 1.08;
    let exp_mc = 0.5 * (1.02 * model.dm[k_last - 1] + 1.08 * model.dm[k_last]);
    let got_mc = crate::hydro_momentum::eval_face_dm_bar(&model, k_last);
    assert!((got_mc - exp_mc).abs() <= 1.0e-15 * exp_mc);
}
