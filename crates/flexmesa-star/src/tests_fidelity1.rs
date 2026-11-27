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
    // Upstream reads `pow2(w)` and `pow2(w_start)`
    // (`star/private/auto_diff_support.f90:396-400,434-439`).
    model.aux.rsp.w[k] = 2.0e11_f64.sqrt();
    model.aux.rsp.et_rsp[k] = 2.0e11;
    model.aux.rsp.w_start[k] = 1.0e11_f64.sqrt();
    model.controls.rti_flag = true;
    model.controls.dedt_rti_diffusion_factor = 1.5;
    model.aux.sig_rti[k] = 1.0e20;
    model.aux.sig_rti[k + 1] = 1.2e20;
    model.controls.use_drag_energy = true;
    model.controls.drag_coefficient = 0.2;

    let (sources, _others, doing_op_split) =
        setup_sources_and_others(&model, k).expect("sources");
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
    model.aux.okay_to_set_mlt_vc = true;
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

/// Row `0` of the pressure or momentum equation after a full assembly.
fn surface_row_after_assembly(
    model: &mut StarModel,
) -> Result<f64, crate::controls::StarError> {
    let sys = eval_henyey_equations(model, false)?;
    Ok(sys.equ[model.layout.i_dv_dt])
}

#[test]
fn test_fixed_vsurf_outer_bc_row_zero_survives_momentum_outer_bc() {
    // `use_fixed_vsurf_outer_BC` is chosen before `need_P_surf`, so
    // `use_momentum_outer_BC` must not rewrite row 0
    // (`star/private/hydro_eqns.f90:758-770,796`).
    let mut model = build_solar_test_model(true);
    model.controls.use_fixed_vsurf_outer_bc = true;
    model.controls.use_momentum_outer_bc = true;
    model.controls.fixed_vsurf = 0.0;
    let row0 = surface_row_after_assembly(&mut model).expect("assembly");
    let expected = model.v[0] / model.aux.csound_start[0];
    assert!((row0 - expected).abs() <= 1.0e-14 * expected.abs());
}

#[test]
fn test_compression_outer_bc_row_zero_uses_dxh_lnd_and_survives_momentum_bc() {
    // `set_compression_BC` (`hydro_eqns.f90:1084-1099`): the row is
    // `(rho2*dxh_lnd(1) - rho1*dxh_lnd(2))/dt`, and it is not rewritten by
    // the momentum boundary equation.
    let mut model = build_solar_test_model(true);
    model.controls.use_compression_outer_bc = true;
    model.controls.use_momentum_outer_bc = true;
    model.aux.dxh_lnd[0] = 2.0e-3;
    model.aux.dxh_lnd[1] = -1.0e-3;
    let row0 = surface_row_after_assembly(&mut model).expect("assembly");
    let rho1 = flexmesa_math::exp(model.lnd[0]);
    let rho2 = flexmesa_math::exp(model.lnd[1]);
    let expected = (rho2 * 2.0e-3 - rho1 * -1.0e-3) / model.dt;
    assert!((row0 - expected).abs() <= 1.0e-12 * expected.abs());
}

#[test]
fn test_fixed_vsurf_outer_bc_without_velocity_flag_is_an_error() {
    // `star/private/hydro_eqns.f90:1113-1115`.
    let mut model = build_solar_test_model(false);
    model.controls.use_fixed_vsurf_outer_bc = true;
    assert!(matches!(
        surface_row_after_assembly(&mut model),
        Err(crate::controls::StarError::InvalidInput(_))
    ));
}

#[test]
fn test_u_flag_with_v_flag_or_mass_corrections_is_an_error() {
    // `star/private/evolve.f90:328-331` and `hydro_eqns.f90:82-83`.
    let mut model = build_solar_test_model(true);
    model.controls.u_flag = true;
    assert!(surface_row_after_assembly(&mut model).is_err());
    let mut model = build_solar_test_model(true);
    model.controls.v_flag = false;
    model.controls.u_flag = true;
    model.controls.use_mass_corrections = true;
    assert!(surface_row_after_assembly(&mut model).is_err());
}

/// Hydrostatic `eps_grav` form: upstream sums `- dwork_dm + eps_grav` with
/// `dwork_dm` evaluated with `skip_P = .true.`
/// (`star/private/hydro_energy.f90:102-104,166-192`), so work by the
/// non-`Peos` pressures (here `mlt_Pturb`) still enters the residual, and it
/// is exactly zero when none of them is active.
#[test]
fn test_hydrostatic_eps_grav_residual_includes_skip_p_work() {
    let mut model = build_solar_test_model(false);
    let k = 4;
    model.controls.energy_eqn_option = EnergyEqnOption::EpsGrav;
    for j in 0..model.nz {
        model.ln_r_start[j] = model.ln_r[j] - 1.0e-4;
    }
    let nvar = model.layout.nvar(false);
    let i_equ = model.layout.i_dlne_dt;
    let residual = |model: &StarModel| {
        let x_scale = compute_x_scale(model, false);
        let mut sys = HenyeySystem::new(model.nz, nvar, x_scale);
        do1_energy_eqn(model, k, false, &mut sys).expect("energy eqn");
        sys.equ[k * nvar + i_equ]
    };
    let (w0, _, _) =
        crate::hydro_work::eval1_work_v_flag_ad(&model, k, true, false);
    assert_eq!(w0.val, 0.0);
    let base = residual(&model);

    model.controls.mlt_pturb_factor = 0.6;
    model.aux.mlt_vc_old[k] = 2.0e5;
    let with_pturb = residual(&model);
    let (dwork_dm, _) =
        crate::hydro_work::eval_v_flag_dwork_dm_ad(&model, k, true, false);
    let scal = crate::hydro_energy::set_energy_eqn_scal(&model, k).unwrap();
    assert!(dwork_dm.val != 0.0);
    let expected = base - scal * dwork_dm.val;
    assert!(
        (with_pturb - expected).abs()
            <= 1.0e-12 * expected.abs().max(base.abs()),
        "{with_pturb} vs {expected}"
    );
}

/// `Pvsc` in `eval1_work` is averaged with `Pvsc_start` on both sides of the
/// face when `include_P_in_velocity_time_centering` holds and
/// `lnT(k)/ln10 <= max_logT_for_include_P_and_L_in_velocity_time_centering`,
/// with no `using_velocity_time_centering` test
/// (`star/private/hydro_energy.f90:709-728`); `Pvsc_start` takes the first
/// `Pvsc` of the step and keeps it (`star/private/star_utils.f90:3452-3468`).
#[test]
fn test_eval1_work_pvsc_time_centred_with_captured_pvsc_start() {
    use crate::hydro_state::PVSC_START_UNSET;
    let mut model = build_solar_test_model(true);
    let k = 4;
    model.controls.using_velocity_time_centering = false;
    model.controls.include_p_in_velocity_time_centering = true;
    model.controls.use_pvsc_art_visc = true;
    model.controls.pvsc_cq = 2.0;
    model.controls.mlt_pturb_factor = 0.0;
    let dr_k = (model.r[k] - model.r[k + 1]).max(1.0);
    let v_shock = -0.25 * model.csound[k] * (model.r[k] / dr_k);
    model.v[k - 1] = 2.0 * v_shock;
    model.v[k] = v_shock;
    model.v[k + 1] = 0.0;
    let pvsc_l = eval_pvsc_cell_ad(&model, k).val;
    let pvsc_r = eval_pvsc_cell_ad(&model, k - 1).val;
    assert!(pvsc_l > 0.0 && pvsc_r > 0.0);
    let (alfa, beta) = model.get_face_weights(k);
    let av =
        crate::hydro_work::eval1_a_times_v_face_v_flag_ad(&model, k, false).val;
    let work = |model: &StarModel| {
        crate::hydro_work::eval1_work_v_flag_ad(model, k, true, false)
            .0
            .val
    };

    // Before the first `get_Pvsc_ad` of the step, `Pvsc_start = Pvsc`.
    model.aux.pvsc_start = vec![PVSC_START_UNSET; model.nz];
    let expected_unset = av
        * ((0.5 * (pvsc_l + pvsc_l)) * alfa + (0.5 * (pvsc_r + pvsc_r)) * beta);
    assert_eq!(work(&model), expected_unset);

    crate::hydro_energy::capture_pvsc_start(&mut model);
    assert_eq!(model.aux.pvsc_start[k], pvsc_l);
    assert_eq!(model.aux.pvsc_start[k - 1], pvsc_r);
    // Cell `k - 2` is not compressed: `get_Pvsc_ad` returns early with 0.
    assert_eq!(model.aux.pvsc_start[k - 2], 0.0);

    // A captured value is kept for the rest of the step.
    let (start_l, start_r) = (0.25 * pvsc_l, 3.0 * pvsc_r);
    model.aux.pvsc_start[k] = start_l;
    model.aux.pvsc_start[k - 1] = start_r;
    crate::hydro_energy::capture_pvsc_start(&mut model);
    assert_eq!(model.aux.pvsc_start[k], start_l);
    let expected_tc = av
        * ((0.5 * (pvsc_l + start_l)) * alfa
            + (0.5 * (pvsc_r + start_r)) * beta);
    let w_tc = work(&model);
    assert!(
        (w_tc - expected_tc).abs() <= 1.0e-14 * expected_tc.abs(),
        "{w_tc} vs {expected_tc}"
    );

    // Above `max_logT_for_include_P_and_L_in_velocity_time_centering` or
    // without `include_P_in_velocity_time_centering`, `Pvsc` is not centred.
    let expected_plain = av * (pvsc_l * alfa + pvsc_r * beta);
    model
        .controls
        .max_logt_for_include_p_and_l_in_velocity_time_centering =
        model.ln_t[k] / flexmesa_const::LN10 - 1.0;
    let w_hot = work(&model);
    assert!((w_hot - expected_plain).abs() <= 1.0e-14 * expected_plain.abs());
    model
        .controls
        .max_logt_for_include_p_and_l_in_velocity_time_centering = 1.0e99;
    model.controls.include_p_in_velocity_time_centering = false;
    let w_off = work(&model);
    assert!((w_off - expected_plain).abs() <= 1.0e-14 * expected_plain.abs());
}

/// A `v_flag` model with TDC eddy viscosity on (`TDC_alpha_M > 0`,
/// `MLT_option = 'TDC'`), velocity time centering off, and a set
/// `mlt_vc_ad` profile.
fn tdc_viscosity_test_model() -> StarModel {
    let mut model = build_solar_test_model(true);
    model.controls.rsp2_flag = false;
    model.controls.mlt_option = MltOption::Tdc;
    model.controls.tdc_alpha_m = 0.25;
    model.controls.using_velocity_time_centering = false;
    model.controls.steps_before_use_velocity_time_centering = -1;
    for k in 0..model.nz {
        model.mlt_vc_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(
            1.0e5 * (1.0 + 0.1 * (k as f64)),
            2.0e3,
            -1.0e3,
        );
        // Shear the velocity field so `d(v/r)/dr` is not small.
        model.v[k] = -1.0e4 * (1.0 + 0.3 * flexmesa_math::sin(k as f64));
    }
    model
}

/// Hand evaluation of `compute_tdc_Eq_div_w_face` at an interior face `k`
/// for `v_flag` without time centering, written from
/// `star/private/tdc_hydro.f90:234-355` and `star_utils.f90:3785-4018`
/// with plain `f64` arithmetic.
fn hand_eq_div_w_face(model: &StarModel, k: usize) -> f64 {
    let (alfa, beta) = model.get_face_weights(k);
    let v00 = 0.5 * (model.v[k] + model.v[k + 1]);
    let vm1 = 0.5 * (model.v[k] + model.v[k - 1]);
    let r00 = 0.5 * (model.r[k] + model.r[k + 1]);
    let rm1 = 0.5 * (model.r[k] + model.r[k - 1]);
    let d_v_div_r = vm1 / rm1 - v00 / r00;
    let rho = alfa * model.rho[k] + beta * model.rho[k - 1];
    let p = alfa * model.peos[k] + beta * model.peos[k - 1];
    let g = model.cgrav[k];
    let grav = g * model.m_grav[k] / (model.r[k] * model.r[k]);
    let mut hp = p / (grav * rho);
    if model.controls.mix_wind.alt_scale_height_flag {
        let hp2 = (p / g).sqrt() / rho;
        if hp2 < hp {
            hp = hp2;
        }
    }
    let dmbar = 0.5 * (model.dm[k] + model.dm[k - 1]);
    let alfam_alfa =
        model.controls.tdc_alpha_m * model.controls.mixing_length_alpha;
    let f = (16.0 / 3.0) * std::f64::consts::PI * alfam_alfa / dmbar;
    let chi = f * rho * rho * flexmesa_math::pow6(model.r[k]) * d_v_div_r * hp;
    4.0 * std::f64::consts::PI * chi * d_v_div_r / dmbar
}

/// `compute_tdc_Eq_div_w_face` against a hand evaluation of the upstream
/// formulas, its `v_00` and `lnR_00` partials against central finite
/// differences, and the zero at the forced-nonturbulent inner cells
/// (`star/private/tdc_hydro.f90:311-355`).
#[test]
fn test_tdc_eq_div_w_face_value_partials_and_forced_inner_cells() {
    use crate::tdc_hydro::compute_tdc_eq_div_w_face;
    let mut model = tdc_viscosity_test_model();
    let k = model.nz / 2;
    let (eq, eq_out) = compute_tdc_eq_div_w_face(&model, k);
    let expected = hand_eq_div_w_face(&model, k);
    assert!(expected != 0.0);
    assert!(
        (eq.val - expected).abs() <= 1.0e-13 * expected.abs(),
        "{} vs {expected}",
        eq.val
    );
    // `s% Eq_ad(k) = Eq_face*w_00` with `w_00 = mlt_vc_ad(k)/sqrt_2_div_3`.
    let w = model.mlt_vc_ad[k].val / flexmesa_const::SQRT_2_DIV_3;
    assert!((eq_out.val - eq.val * w).abs() <= 1.0e-14 * (eq.val * w).abs());

    let h = 1.0e-6;
    let v0 = model.v[k];
    let dv = h * v0.abs();
    model.v[k] = v0 + dv;
    let up = compute_tdc_eq_div_w_face(&model, k).0.val;
    model.v[k] = v0 - dv;
    let down = compute_tdc_eq_div_w_face(&model, k).0.val;
    model.v[k] = v0;
    let fd = (up - down) / (2.0 * dv);
    assert!(
        (eq.d1[I_V_00] - fd).abs() <= 1.0e-6 * fd.abs(),
        "v_00: ad {} vs fd {fd}",
        eq.d1[I_V_00]
    );
    let r0 = model.r[k];
    model.r[k] = r0 * flexmesa_math::exp(h);
    let up = compute_tdc_eq_div_w_face(&model, k).0.val;
    model.r[k] = r0 * flexmesa_math::exp(-h);
    let down = compute_tdc_eq_div_w_face(&model, k).0.val;
    model.r[k] = r0;
    let fd = (up - down) / (2.0 * h);
    let ad = eq.d1[crate::star_ad::I_LNR_00];
    assert!(
        (ad - fd).abs() <= 1.0e-6 * fd.abs(),
        "lnR_00: ad {ad} vs fd {fd}"
    );

    let nz = model.nz;
    model
        .controls
        .mix_wind
        .tdc_num_innermost_cells_forced_nonturbulent = nz - k;
    assert_eq!(compute_tdc_eq_div_w_face(&model, k).0.val, 0.0);
    model
        .controls
        .mix_wind
        .tdc_num_innermost_cells_forced_nonturbulent = nz - k - 1;
    assert_eq!(compute_tdc_eq_div_w_face(&model, k).0.val, eq.val);
}

/// The energy equation's TDC heating term
/// `0.5*(Eq_div_w(k)*mlt_vc_ad(k) + shift_p1(Eq_div_w(k+1))*shift_p1(mlt_vc_ad(k+1)))/sqrt_2_div_3`
/// enters `sources` only with `TDC_include_eturb_in_energy_equation`
/// (`star/private/hydro_energy.f90:264-277,304-308`).
#[test]
fn test_tdc_eturb_heating_enters_energy_sources() {
    use crate::tdc_hydro::{compute_tdc_eq_div_w_face, tdc_energy_eq_ad};
    let mut model = tdc_viscosity_test_model();
    let k = model.nz / 2;
    let eq_ad = tdc_energy_eq_ad(&model, k);
    let e00 = compute_tdc_eq_div_w_face(&model, k).0.val;
    let ep1 = compute_tdc_eq_div_w_face(&model, k + 1).0.val;
    let expected = 0.5
        * (e00 * model.mlt_vc_ad[k].val + ep1 * model.mlt_vc_ad[k + 1].val)
        * (1.0 / flexmesa_const::SQRT_2_DIV_3);
    // `auto_diff` divides by a real as `powm1(y)*x`
    // (`auto_diff/private/auto_diff_real_star_order1_module.f90:1077-1085`).
    assert_eq!(eq_ad.val, expected);

    model.controls.tdc_include_eturb_in_energy_equation = false;
    let (sources_off, _, _) =
        setup_sources_and_others(&model, k).expect("sources");
    model.controls.tdc_include_eturb_in_energy_equation = true;
    let (sources_on, _, _) =
        setup_sources_and_others(&model, k).expect("sources");
    let diff = sources_on.val - sources_off.val;
    assert!(
        (diff - eq_ad.val).abs()
            <= 1.0e-12 * eq_ad.val.abs().max(sources_on.val.abs()),
        "{diff} vs {}",
        eq_ad.val
    );
}

/// `compute_tdc_Uq_face` (`star/private/tdc_hydro.f90:358-392`) against a
/// hand evaluation of `4*pi*(Chi_cell(k-1) - Chi_cell(k))/(r(k)*dm_bar(k))`
/// with `compute_Chi_cell` (`:150-231`), and its entry in the momentum
/// equation's non-HSE term (`star/private/hydro_momentum.f90:433-438`).
#[test]
fn test_tdc_uq_face_value_and_momentum_term() {
    use crate::tdc_hydro::compute_tdc_uq_face;
    let mut model = tdc_viscosity_test_model();
    let k = model.nz / 2;
    let pi = std::f64::consts::PI;
    let hp_face = |model: &StarModel, j: usize| {
        crate::tdc_hydro::get_scale_height_face(model, j).val
    };
    let chi_cell = |model: &StarModel, j: usize| {
        let hp_cell = 0.5 * (hp_face(model, j) + hp_face(model, j + 1));
        let d_v_div_r =
            model.v[j] / model.r[j] - model.v[j + 1] / model.r[j + 1];
        let w = 0.5 * (model.mlt_vc_ad[j].val + model.mlt_vc_ad[j + 1].val)
            / flexmesa_const::SQRT_2_DIV_3;
        let alfam_alfa =
            model.controls.tdc_alpha_m * model.controls.mixing_length_alpha;
        let f = (16.0 / 3.0) * pi * alfam_alfa / model.dm[j];
        let r6 = 0.5
            * (flexmesa_math::pow6(model.r[j])
                + flexmesa_math::pow6(model.r[j + 1]));
        f * model.rho[j] * model.rho[j] * r6 * d_v_div_r * hp_cell * w
    };
    let expected = 4.0 * pi * (chi_cell(&model, k - 1) - chi_cell(&model, k))
        / (model.r[k] * model.dm_bar[k]);
    let uq = compute_tdc_uq_face(&model, k);
    assert!(expected != 0.0);
    assert!(
        (uq.val - expected).abs() <= 1.0e-12 * expected.abs(),
        "{} vs {expected}",
        uq.val
    );

    let with_uq = crate::hydro_momentum::expected_non_hse_term(&model, k);
    model.controls.tdc_alpha_m = 0.0;
    let without = crate::hydro_momentum::expected_non_hse_term(&model, k);
    assert_eq!(with_uq.val, (without + uq).val);

    // Forced-nonturbulent outer cells give zero.
    model.controls.tdc_alpha_m = 0.25;
    model
        .controls
        .mix_wind
        .tdc_num_outermost_cells_forced_nonturbulent = k + 1;
    assert_eq!(compute_tdc_uq_face(&model, k).val, 0.0);
}

/// `set_viscosity_vars_TDC` (`star/private/tdc_hydro.f90:40-89`) stores
/// `Eq_ad = Eq_div_w*w`, `Eq`, `Chi` and `Uq` for every face, and zeroes
/// them without `v_flag` or `u_flag`.
#[test]
fn test_set_viscosity_vars_tdc_stores_outputs() {
    let mut model = tdc_viscosity_test_model();
    let k = model.nz / 2;
    crate::tdc_hydro::set_viscosity_vars_tdc(&mut model).expect("tdc vars");
    let (_, eq_out) = crate::tdc_hydro::compute_tdc_eq_div_w_face(&model, k);
    assert_eq!(model.aux.eq_ad[k].val, eq_out.val);
    assert_eq!(model.aux.rsp.eq_rsp[k], eq_out.val);
    assert_eq!(
        model.aux.rsp.uq_rsp[k],
        crate::tdc_hydro::compute_tdc_uq_face(&model, k).val
    );
    assert!(model.aux.rsp.chi_rsp[k] != 0.0);
    assert!(model.aux.rsp.hp_face.iter().all(|&h| h > 0.0));
    model.controls.v_flag = false;
    crate::tdc_hydro::set_viscosity_vars_tdc(&mut model).expect("tdc vars");
    assert_eq!(model.aux.eq_ad[k].val, 0.0);
    assert_eq!(model.aux.rsp.uq_rsp[k], 0.0);
}

/// `eval_simple_PdV_work` with `mlt_Pturb_factor > 0` and velocity time
/// centering at an interior cell: `Ptot` is the face-averaged
/// `mlt_Pturb_factor*pow2(mlt_vc_old(k))*get_rho_face(s,k)/3` blended with
/// `mlt_Pturb_factor*pow2(mlt_vc_old(k))*(rho_start(k-1) + rho_start(k))/6`
/// by `P_theta_for_velocity_time_centering` (`star/private/star_utils.f90:3390-3432`),
/// times `Av_face00 - Av_facep1` over `dm` (`star/private/hydro_energy.f90:845-875`).
/// The density partials are checked against central finite differences.
#[test]
fn test_simple_pdv_work_mlt_pturb_time_centred_value_and_density_partials() {
    let mut model = build_solar_test_model(true);
    let k = 4;
    model.controls.using_velocity_time_centering = true;
    model.controls.include_p_in_velocity_time_centering = true;
    model.controls.p_theta_for_velocity_time_centering = 0.55;
    model.controls.mlt_pturb_factor = 0.6;
    model.controls.use_pvsc_art_visc = false;
    model.aux.mlt_vc_old[k] = 2.0e5;
    for j in 0..model.nz {
        model.ln_r_start[j] = model.ln_r[j] - 1.0e-4;
        model.lnd_start[j] = model.lnd[j] + 2.0e-3;
    }
    let eval = |model: &StarModel| {
        crate::hydro_work::eval_v_flag_simple_pdv_dm_ad(model, k, true, true).0
    };
    let dwork_dm = eval(&model);

    let (alfa_f, beta_f) = model.get_face_weights(k);
    let rho_face = flexmesa_math::exp(model.lnd[k]) * alfa_f
        + flexmesa_math::exp(model.lnd[k - 1]) * beta_f;
    let coef = 0.6 * 2.0e5 * 2.0e5;
    let p_theta = 0.55;
    let pturb = coef * rho_face / 3.0;
    let pturb_start = coef
        * (flexmesa_math::exp(model.lnd_start[k - 1])
            + flexmesa_math::exp(model.lnd_start[k]))
        / 6.0;
    let p_tot = p_theta * pturb + (1.0 - p_theta) * pturb_start;
    let av_00 =
        crate::hydro_work::eval1_a_times_v_face_v_flag_ad(&model, k, true).val;
    let av_p1 =
        crate::hydro_work::eval1_a_times_v_face_v_flag_ad(&model, k + 1, true)
            .val;
    let expected = p_tot * (av_00 - av_p1) / model.dm[k];
    assert!(expected != 0.0);
    assert!(
        (dwork_dm.val - expected).abs() <= 1.0e-13 * expected.abs(),
        "{} vs {expected}",
        dwork_dm.val
    );

    let h = 1.0e-6;
    for (cell, idx) in [(k, I_LND_00), (k - 1, crate::star_ad::I_LND_M1)] {
        let base = model.lnd[cell];
        model.lnd[cell] = base + h;
        let up = eval(&model).val;
        model.lnd[cell] = base - h;
        let down = eval(&model).val;
        model.lnd[cell] = base;
        let fd = (up - down) / (2.0 * h);
        let ad = dwork_dm.d1[idx];
        assert!(
            (ad - fd).abs() <= 1.0e-6 * fd.abs(),
            "cell {cell}: ad {ad} vs fd {fd}"
        );
    }
}

/// With `u_flag`, upstream switches the work term to `eval_simple_PdV_work`
/// on the Riemann face velocity when velocity time centering and
/// `use_P_d_1_div_rho_form_of_work_when_time_centering_velocity` hold
/// (`star/private/hydro_energy.f90:170-175`); that path is not ported and
/// the energy equation returns an error.
#[test]
fn test_u_flag_simple_pdv_work_form_is_an_explicit_error() {
    let mut model = build_solar_test_model(true);
    let k = 4;
    model.controls.v_flag = false;
    model.controls.u_flag = true;
    model.controls.using_velocity_time_centering = true;
    model
        .controls
        .use_p_d_1_div_rho_form_of_work_when_time_centering_velocity = true;
    let nvar = model.layout.nvar(false);
    let x_scale = compute_x_scale(&model, false);
    let mut sys = HenyeySystem::new(model.nz, nvar, x_scale);
    let err = do1_energy_eqn(&model, k, false, &mut sys)
        .expect_err("u_flag simple PdV work");
    assert!(err.to_string().contains("hydro_energy.f90:170-175"));
}

/// `compute_tdc_Uq_dm_cell` for `u_flag` (`star/private/tdc_hydro.f90:395-445`):
/// `4*pi*(Chi_div_w_face(k)*w(k) - Chi_div_w_face(k+1)*w(k+1))/r_cell` with
/// `r_cell = 0.5*(r(k) + r(k+1))` and the `u_flag` form of `d(v/r)/dr`,
/// `u(k-1)/r(k-1) - u(k)/r(k)` (`:578-616`), evaluated by hand.
#[test]
fn test_tdc_uq_dm_cell_u_flag_value() {
    let mut model = tdc_viscosity_test_model();
    model.controls.v_flag = false;
    model.controls.u_flag = true;
    for k in 0..model.nz {
        model.u[k] = model.v[k];
    }
    let k = model.nz / 2;
    let pi = std::f64::consts::PI;
    let chi_div_w = |model: &StarModel, j: usize| {
        let (alfa, beta) = model.get_face_weights(j);
        let d_v_div_r =
            model.u[j - 1] / model.r[j - 1] - model.u[j] / model.r[j];
        let rho = alfa * model.rho[j] + beta * model.rho[j - 1];
        let hp = crate::tdc_hydro::get_scale_height_face(model, j).val;
        let dmbar = 0.5 * (model.dm[j] + model.dm[j - 1]);
        let alfam_alfa =
            model.controls.tdc_alpha_m * model.controls.mixing_length_alpha;
        let f = (16.0 / 3.0) * pi * alfam_alfa / dmbar;
        f * rho * rho * flexmesa_math::pow6(model.r[j]) * d_v_div_r * hp
    };
    let w = |j: usize| model.mlt_vc_ad[j].val / flexmesa_const::SQRT_2_DIV_3;
    let r_cell = 0.5 * (model.r[k] + model.r[k + 1]);
    let expected = 4.0
        * pi
        * (chi_div_w(&model, k) * w(k) - chi_div_w(&model, k + 1) * w(k + 1))
        / r_cell;
    let uq = crate::tdc_hydro::compute_tdc_uq_dm_cell(&model, k);
    assert!(expected != 0.0);
    assert!(
        (uq.val - expected).abs() <= 1.0e-12 * expected.abs(),
        "{} vs {expected}",
        uq.val
    );
}
