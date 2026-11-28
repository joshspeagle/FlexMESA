// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone invariant tests (`#45`, `#152`, `#155`)
//! across `hydro_momentum`, `hydro_temperature`, `hydro_brunt_surf`,
//! `riemann_ad`, and `riemann_eqns`.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

use flexmesa_atm::eval_irradiated;
use flexmesa_const::{
    MixingType, CLIGHT, CRAD, FOUR_PI, LN10, L_SUN, M_SUN, R_SUN,
    STANDARD_CGRAV,
};
use flexmesa_eos::I_GAMMA1;
use flexmesa_star::controls::{
    AtmIrradiatedOpacity, AtmOption, StarControls, StarJobControls,
};
use flexmesa_star::hydro_brunt_surf::{do_brunt_b, do_brunt_n2, eval_surf_bc};
use flexmesa_star::hydro_eqns::{compute_x_scale, HenyeySystem};
use flexmesa_star::hydro_momentum::{
    do1_radius_eqn, expected_non_hse_term, get_area_info_opt_time_center,
};
use flexmesa_star::hydro_temperature::{do1_temperature_eqn, set_rsp_lsurf_bc};
use flexmesa_star::hydro_vars::set_hydro_vars;
use flexmesa_star::riemann_ad::{
    calc_ptot_cell_ad, do1_riemann_momentum_eqn, do_uface_and_pface,
};
use flexmesa_star::rsp2::X_ALFAP;
use flexmesa_star::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LND_M1, I_LNR_00, I_LNT_00, I_LNT_M1,
    I_V_00, I_V_M1, I_V_P1, I_W_00, NUM_STAR_AD_VARS,
};
use flexmesa_star::zams::load_zams_model;
use flexmesa_star::StarModel;

const IDEAL_GAS_C: f64 = 1.0e8;

fn assert_close(label: &str, actual: f64, expected: f64, rtol: f64, atol: f64) {
    let diff = (actual - expected).abs();
    let scale = actual.abs().max(expected.abs());
    assert!(
        diff <= atol || (scale > 0.0 && diff / scale <= rtol),
        "{label}: actual={actual:.12e}, expected={expected:.12e}, diff={diff:.3e}"
    );
}

fn new_sys(model: &StarModel) -> HenyeySystem {
    let nvar = model.layout.nvar(false);
    let x_scale = compute_x_scale(model, false);
    HenyeySystem::new(model.nz, nvar, x_scale)
}

fn unscaled_partial(
    sys: &HenyeySystem,
    k_equ: usize,
    i_equ: usize,
    k_var: usize,
    j_var: usize,
) -> f64 {
    let nvar = sys.nvar;
    let idx = k_equ * nvar * nvar + i_equ * nvar + j_var;
    let blk = match (k_equ, k_var) {
        (ke, kv) if ke == kv => sys.dblk[idx],
        (ke, kv) if kv + 1 == ke => sys.lblk[idx],
        (ke, kv) if ke + 1 == kv => sys.ublk[idx],
        _ => return 0.0,
    };
    blk / sys.x_scale[k_var * nvar + j_var]
}

fn sync_ideal_gas_eos(model: &mut StarModel) {
    for k in 0..model.nz {
        model.rho[k] = flexmesa_math::exp(model.lnd[k]);
        model.t[k] = flexmesa_math::exp(model.ln_t[k]);
        model.r[k] = flexmesa_math::exp(model.ln_r[k]);
        let p = IDEAL_GAS_C * model.rho[k] * model.t[k];
        model.pgas[k] = p;
        model.prad[k] = 0.0;
        model.peos[k] = p;
        model.ln_pgas[k] = flexmesa_math::log(p);
        model.ln_peos[k] = model.ln_pgas[k];
        model.gamma1[k] = 5.0 / 3.0;
        model.csound[k] = (model.gamma1[k] * p / model.rho[k]).sqrt();
        model.eos_d_dlnd[k][I_GAMMA1] = 0.0;
        model.eos_d_dlnt[k][I_GAMMA1] = 0.0;
        model.ln_peos_ad[k] =
            StarAdOrder1::from_dlnd_dlnt_00(model.ln_peos[k], 1.0, 1.0);
    }
}

fn build_synthetic_model(controls: StarControls) -> StarModel {
    let nz = 3;
    let mut m =
        StarModel::new(nz, vec![1], StarJobControls::default(), controls);
    m.dt = 10.0;
    m.m_center = 0.0;
    m.r_center = 0.1 * R_SUN;
    m.dq = vec![0.2, 0.3, 0.5];
    m.mstar = M_SUN;
    m.xmstar = M_SUN;
    m.set_qs();
    m.set_xqs();
    m.set_m_and_dm();
    m.cgrav = vec![STANDARD_CGRAV; nz];
    m.lnd = vec![-12.0, -10.5, -9.0];
    m.ln_t = vec![10.0, 11.5, 13.0];
    m.ln_r = vec![
        flexmesa_math::log(1.0 * R_SUN),
        flexmesa_math::log(0.7 * R_SUN),
        flexmesa_math::log(0.4 * R_SUN),
    ];
    m.lum = vec![2.0 * L_SUN, 1.8 * L_SUN, 1.5 * L_SUN];
    m.energy = vec![1.0e13, 1.5e13, 2.0e13];
    m.opacity = vec![0.4, 0.5, 0.6];
    sync_ideal_gas_eos(&mut m);
    m.set_m_grav_and_grav();
    m.save_start_values().expect("save_start_values");
    m.save_old_values();
    m
}

#[test]
fn test_expected_non_hse_term_local_v_flag_and_u_flag_exclusion() {
    let mut ctrl = StarControls {
        v_flag: true,
        velocity_logt_lower_bound: 4.0,
        drag_coefficient: 0.0,
        min_q_for_drag: 0.5,
        ..StarControls::default()
    };
    let mut m = build_synthetic_model(ctrl.clone());
    m.v = vec![3.0e5, 4.0e5, 5.0e5];
    m.v_start = vec![1.0e5, 2.0e5, 3.0e5];
    m.aux.dxh_v = vec![2.0e5, 2.0e5, 2.0e5];
    m.ln_t_old = vec![5.0 * LN10, 5.0 * LN10, 3.0 * LN10];

    // `local_v_flag == true` at zone 1 (`logT_old = 5 >= 4`): `-dxh_v/dt`.
    let res_local_true = expected_non_hse_term(&m, 1);
    let exp_val_true = -(m.v[1] - m.v_start[1]) / m.dt;
    assert_close(
        "local_v_flag true val",
        res_local_true.val,
        exp_val_true,
        1.0e-14,
        0.0,
    );
    assert_close(
        "local_v_flag true dv00",
        res_local_true.d1[I_V_00],
        -1.0 / m.dt,
        1.0e-14,
        0.0,
    );

    // `local_v_flag == false` at zone 2 (`logT_old = 3 < 4`): `-2*v/dt`.
    let res_local_false = expected_non_hse_term(&m, 2);
    let exp_val_false = -2.0 * m.v[2] / m.dt;
    assert_close(
        "local_v_flag false val",
        res_local_false.val,
        exp_val_false,
        1.0e-14,
        0.0,
    );
    assert_close(
        "local_v_flag false dv00",
        res_local_false.d1[I_V_00],
        -2.0 / m.dt,
        1.0e-14,
        0.0,
    );

    // Active drag (`q[1] = 0.8 > min_q_for_drag = 0.5`) and `RSP2 Uq_ad`.
    m.controls.drag_coefficient = 0.25;
    m.controls.rsp2_flag = true;
    let mut uq = StarAdOrder1::constant(120.0);
    uq.d1[I_V_00] = -0.3;
    m.aux.uq_ad[1] = uq;
    let res_drag_uq = expected_non_hse_term(&m, 1);
    let exp_drag_val =
        exp_val_true - m.controls.drag_coefficient * m.v[1] / m.dt + 120.0;
    let exp_drag_dv = -1.0 / m.dt - m.controls.drag_coefficient / m.dt - 0.3;
    assert_close("drag+uq val", res_drag_uq.val, exp_drag_val, 1.0e-14, 0.0);
    assert_close(
        "drag+uq dv00",
        res_drag_uq.d1[I_V_00],
        exp_drag_dv,
        1.0e-14,
        0.0,
    );

    // `u_flag`-only model must NOT add acceleration, drag, or `Uq_ad`.
    ctrl.v_flag = false;
    ctrl.u_flag = true;
    ctrl.drag_coefficient = 0.25;
    ctrl.rsp2_flag = true;
    let mut mu = build_synthetic_model(ctrl);
    mu.u = vec![2.0e5, 3.0e5, 4.0e5];
    mu.u_start = vec![0.0; 3];
    mu.aux.uq_ad[1] = uq;
    let res_u = expected_non_hse_term(&mu, 1);
    assert_close("u_flag excludes accel/drag/uq", res_u.val, 0.0, 0.0, 0.0);
    assert_eq!(res_u.d1, [0.0; NUM_STAR_AD_VARS]);
}

#[test]
fn test_do1_radius_eqn_force_zero_v_checks_u_flag_before_v_flag() {
    let ctrl = StarControls {
        v_flag: true,
        u_flag: true,
        velocity_q_upper_bound: 0.5,
        ..StarControls::default()
    };
    let mut m = build_synthetic_model(ctrl);
    m.u = vec![3.0e5, 0.0, 0.0];
    m.v = vec![7.0e5, 0.0, 0.0];
    m.aux.csound_start = vec![1.0e6; m.nz];

    let mut sys = new_sys(&m);
    do1_radius_eqn(&m, 0, &mut sys).expect("do1_radius_eqn");
    let i_equ = m.layout.i_dlnr_dt.expect("i_dlnr_dt");
    assert_close(
        "force_zero_v checks u_flag before v_flag",
        sys.equ[i_equ],
        m.u[0] / m.aux.csound_start[0],
        1.0e-15,
        0.0,
    );
    let j_vel = m.layout.i_v.or(m.layout.i_u).expect("velocity col");
    assert_close(
        "force_zero_v d_vel",
        unscaled_partial(&sys, 0, i_equ, 0, j_vel),
        1.0 / m.aux.csound_start[0],
        1.0e-15,
        0.0,
    );
}

#[test]
fn test_set_rsp_lsurf_bc_formula_time_centering_and_ad_vs_fd() {
    for use_tc in [false, true] {
        let mut ctrl = StarControls {
            use_rsp_l_eqn_outer_bc: true,
            using_velocity_time_centering: use_tc,
            ..StarControls::default()
        };
        ctrl.rsp2.lsurf_factor = 0.75;
        let mut m = build_synthetic_model(ctrl);
        m.lum_start = vec![1.0e35, 2.5e35, 1.8e35];
        if use_tc {
            m.ln_r_start[0] -= 0.05;
        }

        let mut sys = new_sys(&m);
        set_rsp_lsurf_bc(&m, &mut sys);
        let i_equ = m.layout.i_equ_l;
        let (area, _) = get_area_info_opt_time_center(&m, 0);
        let t0 = flexmesa_math::exp(m.ln_t[0]);
        let l_exp = m.controls.rsp2.lsurf_factor
            * area.val
            * CLIGHT
            * (CRAD * flexmesa_math::pow4(t0));
        let max_l_start = 2.5e35;
        assert_close(
            "set_rsp_lsurf_bc resid",
            sys.equ[i_equ],
            (m.lum[0] - l_exp) / max_l_start,
            1.0e-14,
            0.0,
        );

        let eval_at = |lum_0: f64, lnr_0: f64, lnt_0: f64| {
            let mut mc = m.clone();
            mc.lum[0] = lum_0;
            mc.ln_r[0] = lnr_0;
            mc.r[0] = flexmesa_math::exp(lnr_0);
            mc.ln_t[0] = lnt_0;
            mc.t[0] = flexmesa_math::exp(lnt_0);
            let mut sc = new_sys(&mc);
            set_rsp_lsurf_bc(&mc, &mut sc);
            sc.equ[i_equ]
        };
        let eps_l = 1.0e-3 * m.lum[0];
        let fd_l = (eval_at(m.lum[0] + eps_l, m.ln_r[0], m.ln_t[0])
            - eval_at(m.lum[0] - eps_l, m.ln_r[0], m.ln_t[0]))
            / (2.0 * eps_l);
        let eps = 1.0e-5;
        let fd_lnr = (eval_at(m.lum[0], m.ln_r[0] + eps, m.ln_t[0])
            - eval_at(m.lum[0], m.ln_r[0] - eps, m.ln_t[0]))
            / (2.0 * eps);
        let fd_lnt = (eval_at(m.lum[0], m.ln_r[0], m.ln_t[0] + eps)
            - eval_at(m.lum[0], m.ln_r[0], m.ln_t[0] - eps))
            / (2.0 * eps);

        assert_close(
            "rsp_lsurf d_lum",
            unscaled_partial(&sys, 0, i_equ, 0, m.layout.i_lum),
            fd_l,
            1.0e-9,
            1.0e-45,
        );
        assert_close(
            "rsp_lsurf d_lnr",
            unscaled_partial(&sys, 0, i_equ, 0, m.layout.i_lnr),
            fd_lnr,
            1.0e-9,
            1.0e-12,
        );
        assert_close(
            "rsp_lsurf d_lnt",
            unscaled_partial(&sys, 0, i_equ, 0, m.layout.i_lnt),
            fd_lnt,
            1.0e-9,
            1.0e-12,
        );
    }
}

#[test]
fn test_do1_gradt_eqn_uses_ln_peos_ad_and_ln_t_diffs_and_ad_vs_fd() {
    let ctrl = StarControls {
        use_gradt_actual_vs_gradt_mlt_for_t_gradient_eqn: true,
        using_velocity_time_centering: true,
        include_p_in_velocity_time_centering: true,
        ..StarControls::default()
    };
    let mut m = build_synthetic_model(ctrl);
    m.peos_start = vec![1.0e12, 5.0e13, 2.0e15];
    m.gradt_ad[1] = StarAdOrder1::constant(0.32);

    let mut sys = new_sys(&m);
    do1_temperature_eqn(&m, 1, &mut sys);
    let i_equ = m.layout.i_equ_l;
    let nvar = sys.nvar;

    let dln_p = m.ln_peos[0] - m.ln_peos[1];
    let dln_t = m.ln_t[0] - m.ln_t[1];
    assert_close(
        "do1_gradt_eqn resid",
        sys.equ[nvar + i_equ],
        0.32 * dln_p - dln_t,
        1.0e-14,
        0.0,
    );

    let vars = [
        (0, m.layout.i_lnd),
        (0, m.layout.i_lnt),
        (1, m.layout.i_lnd),
        (1, m.layout.i_lnt),
    ];
    let eps = 1.0e-4;
    for &(k_var, j_var) in &vars {
        let eval_pert = |sgn: f64| {
            let mut mp = m.clone();
            if j_var == mp.layout.i_lnd {
                mp.lnd[k_var] += sgn * eps;
            } else {
                mp.ln_t[k_var] += sgn * eps;
            }
            sync_ideal_gas_eos(&mut mp);
            let mut sp = new_sys(&mp);
            do1_temperature_eqn(&mp, 1, &mut sp);
            sp.equ[nvar + i_equ]
        };
        let fd = (eval_pert(1.0) - eval_pert(-1.0)) / (2.0 * eps);
        let an = unscaled_partial(&sys, 1, i_equ, k_var, j_var);
        assert_close(
            &format!("gradt partial k_var={k_var} j_var={j_var}"),
            an,
            fd,
            1.0e-9,
            1.0e-12,
        );
    }
}

#[test]
fn test_do1_alt_dlnt_dm_eqn_convective_logt_gate_and_flux_limiter() {
    let mut ctrl = StarControls {
        use_dprad_dm_form_of_t_gradient_eqn: true,
        use_flux_limiting_with_dprad_dm_form: true,
        ..StarControls::default()
    };
    ctrl.mix_wind.max_logt_for_mlt = 6.0;
    let mut m = build_synthetic_model(ctrl);
    m.lum = vec![2.0e29, 1.8e29, 1.5e29];
    m.mixing_type[1] = MixingType::ConvectiveMixing;
    m.gradr[1] = 2.0;
    m.gradt[1] = 0.4;
    m.gradr_ad[1] = StarAdOrder1::constant(2.0);
    m.gradt_ad[1] = StarAdOrder1::constant(0.4);
    m.d_opacity_dlnd = vec![0.0; m.nz];
    m.d_opacity_dlnt = vec![0.0; m.nz];

    // `ln_t[1] = 11.5 -> logT = 4.994 <= 6.0`, so radiative luminosity scales
    // by `gradt_ad / gradr_ad = 0.2`.
    let mut sys_conv = new_sys(&m);
    do1_temperature_eqn(&m, 1, &mut sys_conv);
    let i_equ = m.layout.i_equ_l;
    let nvar = sys_conv.nvar;

    let mut m_gated = m.clone();
    m_gated.controls.mix_wind.max_logt_for_mlt = 4.5;
    let mut sys_gated = new_sys(&m_gated);
    do1_temperature_eqn(&m_gated, 1, &mut sys_gated);

    let r1 = flexmesa_math::exp(m.ln_r[1]);
    let area = FOUR_PI * r1 * r1;
    let area2 = area * area;
    let kap_f =
        0.2 * m.opacity[1] / (0.2 + 0.3) + 0.3 * m.opacity[0] / (0.2 + 0.3);
    let t4_00 = flexmesa_math::pow4(flexmesa_math::exp(m.ln_t[1]));
    let t4_m1 = flexmesa_math::pow4(flexmesa_math::exp(m.ln_t[0]));
    let flx_r = (area * (t4_m1 - t4_00).abs() / m.dm_bar[1])
        / (kap_f * 0.5 * (t4_m1 + t4_00));
    let flx_lam = (6.0 + 3.0 * flx_r) / (6.0 + (3.0 + flx_r) * flx_r);
    let dprad_act = (CRAD / 3.0) * (t4_m1 - t4_00);
    let scale = m.energy_start[1] * flexmesa_math::exp(m.lnd_start[1]);
    let dprad_conv =
        -(m.dm_bar[1] * kap_f * (m.lum[1] * 0.2)) / (CLIGHT * area2) / flx_lam;
    let dprad_full =
        -(m.dm_bar[1] * kap_f * m.lum[1]) / (CLIGHT * area2) / flx_lam;

    assert_close(
        "alt_dlnt_dm conv active",
        sys_conv.equ[nvar + i_equ],
        (dprad_conv - dprad_act) / scale,
        1.0e-13,
        0.0,
    );
    assert_close(
        "alt_dlnt_dm max_logt_for_mlt gated",
        sys_gated.equ[nvar + i_equ],
        (dprad_full - dprad_act) / scale,
        1.0e-13,
        0.0,
    );

    // Check AD partials wrt `lnR_00`, `lnT_00`, `lnT_m1`, `L_00`.
    let eps = 2.0e-5;
    for (k_var, j_var, step) in [
        (1, m.layout.i_lnr, eps),
        (1, m.layout.i_lnt, eps),
        (0, m.layout.i_lnt, eps),
        (1, m.layout.i_lum, 1.0e-3 * m.lum[1]),
    ] {
        let eval_pert = |sgn: f64| {
            let mut mp = m.clone();
            if j_var == mp.layout.i_lnr {
                mp.ln_r[k_var] += sgn * step;
                mp.r[k_var] = flexmesa_math::exp(mp.ln_r[k_var]);
            } else if j_var == mp.layout.i_lnt {
                mp.ln_t[k_var] += sgn * step;
                mp.t[k_var] = flexmesa_math::exp(mp.ln_t[k_var]);
            } else {
                mp.lum[k_var] += sgn * step;
            }
            let mut sp = new_sys(&mp);
            do1_temperature_eqn(&mp, 1, &mut sp);
            sp.equ[nvar + i_equ]
        };
        let fd = (eval_pert(1.0) - eval_pert(-1.0)) / (2.0 * step);
        let an = unscaled_partial(&sys_conv, 1, i_equ, k_var, j_var);
        assert_close(
            &format!("alt_dlnt_dm partial k_var={k_var} j_var={j_var}"),
            an,
            fd,
            2.0e-8,
            1.0e-15,
        );
    }
}

#[test]
fn test_calc_ptot_cell_ad_includes_p_theta_pvsc_rsp2_and_mlt_pturb() {
    let mut ctrl = StarControls {
        v_flag: true,
        using_velocity_time_centering: true,
        include_p_in_velocity_time_centering: true,
        p_theta_for_velocity_time_centering: 0.4,
        use_pvsc_art_visc: true,
        pvsc_cq: 4.0,
        pvsc_zshock: 0.1,
        mlt_pturb_factor: 0.5,
        rsp2_flag: true,
        ..StarControls::default()
    };
    ctrl.rsp2.alfap = 0.6;
    let mut m = build_synthetic_model(ctrl);
    m.v = vec![1.0e5, 1.0e6, 3.0e6];
    m.v_start = m.v.clone();
    m.aux.mlt_vc_old = vec![5.0e4, 6.0e4, 7.0e4];
    m.aux.rsp.et_rsp = vec![1.0e9, 2.0e9, 3.0e9];
    m.aux.rsp.ptrb_start = (0..m.nz)
        .map(|k| {
            X_ALFAP * m.controls.rsp2.alfap * m.rho[k] * m.aux.rsp.et_rsp[k]
        })
        .collect();
    // Perturb `lnd`, `ln_t`, and `et_rsp` from start values so time centering
    // and artificial viscosity are strictly non-trivial.
    for k in 0..m.nz {
        m.lnd[k] += 0.05;
        m.ln_t[k] += 0.03;
        m.aux.rsp.et_rsp[k] *= 1.2;
    }
    sync_ideal_gas_eos(&mut m);

    let ptot0 = calc_ptot_cell_ad(&m, 0, false, false);
    let p0_theta = 0.4;
    let expected_peos0 =
        p0_theta * m.peos[0] + (1.0 - p0_theta) * m.peos_start[0];
    let dv0 = (m.v[1] - m.v[0])
        - m.controls.pvsc_zshock * (m.peos[0] / m.rho[0]).sqrt();
    let expected_pvsc0 = m.controls.pvsc_cq * m.rho[0] * dv0 * dv0;
    let ptrb0_now =
        X_ALFAP * m.controls.rsp2.alfap * m.rho[0] * m.aux.rsp.et_rsp[0];
    let expected_ptrb0 =
        p0_theta * ptrb0_now + (1.0 - p0_theta) * m.aux.rsp.ptrb_start[0];
    // At `k == 0`, `mlt_Pturb` is excluded (`k > 0` guard).
    assert_close(
        "calc_ptot_cell_ad k=0 skips mlt_pturb",
        ptot0.val,
        expected_peos0 + expected_pvsc0 + expected_ptrb0,
        1.0e-13,
        0.0,
    );

    let ptot1 = calc_ptot_cell_ad(&m, 1, false, false);
    let ptot1_shift = calc_ptot_cell_ad(&m, 1, true, false);
    let (alfa, beta) = m.get_face_weights(1);
    let rho_face = alfa * m.rho[1] + beta * m.rho[0];
    let mlt_now = m.controls.mlt_pturb_factor
        * flexmesa_math::pow2(m.aux.mlt_vc_old[1])
        * rho_face
        / 3.0;
    let rho_s0 = flexmesa_math::exp(m.lnd_start[0]);
    let rho_s1 = flexmesa_math::exp(m.lnd_start[1]);
    let mlt_start = m.controls.mlt_pturb_factor
        * flexmesa_math::pow2(m.aux.mlt_vc_old[1])
        * (rho_s0 + rho_s1)
        / 6.0;
    let expected_mlt1 = p0_theta * mlt_now + (1.0 - p0_theta) * mlt_start;
    let dv1 = (m.v[2] - m.v[1])
        - m.controls.pvsc_zshock * (m.peos[1] / m.rho[1]).sqrt();
    let expected_pvsc1 = m.controls.pvsc_cq * m.rho[1] * dv1 * dv1;
    let ptrb1_now =
        X_ALFAP * m.controls.rsp2.alfap * m.rho[1] * m.aux.rsp.et_rsp[1];
    let expected_ptrb1 =
        p0_theta * ptrb1_now + (1.0 - p0_theta) * m.aux.rsp.ptrb_start[1];
    let expected_peos1 =
        p0_theta * m.peos[1] + (1.0 - p0_theta) * m.peos_start[1];

    assert_close(
        "calc_ptot_cell_ad k=1 val",
        ptot1.val,
        expected_peos1 + expected_pvsc1 + expected_ptrb1 + expected_mlt1,
        1.0e-13,
        0.0,
    );
    assert_eq!(ptot1_shift, ptot1.shift_m1());

    // Central FD partial checks wrt cell 1 `lnd`, `ln_t`, `v_00`, `v_p1`, `w_00`, `lnd_m1`.
    let eps = 1.0e-5;
    for (slot, perturb) in [
        (
            I_LND_00,
            Box::new(|mc: &mut StarModel, d: f64| mc.lnd[1] += d)
                as Box<dyn Fn(&mut StarModel, f64)>,
        ),
        (
            I_LND_M1,
            Box::new(|mc: &mut StarModel, d: f64| mc.lnd[0] += d),
        ),
        (
            I_LNT_00,
            Box::new(|mc: &mut StarModel, d: f64| mc.ln_t[1] += d),
        ),
        (I_V_00, Box::new(|mc: &mut StarModel, d: f64| mc.v[1] += d)),
        (I_V_P1, Box::new(|mc: &mut StarModel, d: f64| mc.v[2] += d)),
        (
            I_W_00,
            Box::new(|mc: &mut StarModel, d: f64| {
                let w = mc.aux.rsp.et_rsp[1].sqrt() + d;
                mc.aux.rsp.et_rsp[1] = w * w;
            }),
        ),
    ] {
        let h = match slot {
            I_LND_M1 => 1.0e-3,
            I_V_00 | I_V_P1 => 1.0e3,
            I_W_00 => 1.0e1,
            _ => eps,
        };
        let mut mp = m.clone();
        let mut mm = m.clone();
        perturb(&mut mp, h);
        perturb(&mut mm, -h);
        sync_ideal_gas_eos(&mut mp);
        sync_ideal_gas_eos(&mut mm);
        let fd = (calc_ptot_cell_ad(&mp, 1, false, false).val
            - calc_ptot_cell_ad(&mm, 1, false, false).val)
            / (2.0 * h);
        assert_close(
            &format!("calc_ptot_cell_ad partial slot={slot}"),
            ptot1.d1[slot],
            fd,
            5.0e-7,
            1.0e-6,
        );
    }
}

#[test]
fn test_do_uface_and_pface_surface_peos_rsp2_uq_and_per_face_starts() {
    let mut ctrl = StarControls {
        u_flag: true,
        rsp2_flag: true,
        ..StarControls::default()
    };
    ctrl.rsp2.alfap = 0.5;
    let mut m = build_synthetic_model(ctrl);
    m.lnd = vec![-2.0, -1.0, 0.0];
    m.ln_t = vec![14.0, 15.0, 16.0];
    sync_ideal_gas_eos(&mut m);
    m.save_start_values().expect("save_start_values");
    m.save_old_values();
    m.u = vec![1.0e5, 2.0e5, 3.0e5];
    m.aux.rsp.et_rsp = vec![1.0e9, 2.0e9, 3.0e9];
    m.aux.rsp.ptrb_start = (0..m.nz)
        .map(|k| {
            X_ALFAP * m.controls.rsp2.alfap * m.rho[k] * m.aux.rsp.et_rsp[k]
        })
        .collect();
    m.aux.p_face_start = vec![-1.0; m.nz];
    m.aux.u_face_start = vec![0.0; m.nz];

    let mut m_no_uq = m.clone();
    do_uface_and_pface(&mut m_no_uq).expect("do_uface_and_pface no_uq");

    let mut uq = StarAdOrder1::constant(4.5e4);
    uq.d1[I_W_00] = 2.5e2;
    m.aux.uq_ad[1] = uq;
    do_uface_and_pface(&mut m).expect("do_uface_and_pface with uq");

    // Face 0 pressure must be `wrap_Peos_00` (excluding `RSP2 Ptrb`).
    assert_close(
        "p_face_ad[0] == peos[0]",
        m.aux.p_face_ad[0].val,
        m.peos[0],
        1.0e-14,
        0.0,
    );
    assert_close(
        "p_face_ad[0] d_lnd",
        m.aux.p_face_ad[0].d1[I_LND_00],
        m.peos[0],
        1.0e-14,
        0.0,
    );
    // Surface `p_face_start[0]` / `u_face_start[0]` stay untouched (`-1.0` / `0.0`).
    assert_eq!(m.aux.p_face_start[0], -1.0);
    assert_eq!(m.aux.u_face_start[0], 0.0);
    assert_close(
        "p_face_start[1] initialized",
        m.aux.p_face_start[1],
        m.aux.p_face_ad[1].val,
        1.0e-14,
        0.0,
    );
    assert_close(
        "u_face_start[1] initialized",
        m.aux.u_face_start[1],
        m.aux.u_face_ad[1].val,
        1.0e-14,
        0.0,
    );

    // `RSP2 Uq_ad` enters `u_face_ad[1]` value and partials.
    assert_close(
        "u_face_ad[1] adds uq_ad val",
        m.aux.u_face_ad[1].val - m_no_uq.aux.u_face_ad[1].val,
        uq.val,
        1.0e-13,
        0.0,
    );
    assert_close(
        "u_face_ad[1] adds uq_ad d_w00",
        m.aux.u_face_ad[1].d1[I_W_00] - m_no_uq.aux.u_face_ad[1].d1[I_W_00],
        uq.d1[I_W_00],
        1.0e-13,
        0.0,
    );

    // Subsequent call after velocity change preserves start-of-step interior faces.
    let saved_pf1 = m.aux.p_face_start[1];
    let saved_uf1 = m.aux.u_face_start[1];
    m.u[1] += 1.0e4;
    do_uface_and_pface(&mut m).expect("second do_uface_and_pface");
    assert_eq!(m.aux.p_face_start[0], -1.0);
    assert_eq!(m.aux.p_face_start[1], saved_pf1);
    assert_eq!(m.aux.u_face_start[1], saved_uf1);

    // Check internal-face HLLC AD derivatives against central FD.
    m.aux.uq_ad[1] = StarAdOrder1::ZERO;
    m.aux.uq_ad[2] = StarAdOrder1::ZERO;
    do_uface_and_pface(&mut m).expect("clean uface_pface for fd");
    let uf1 = m.aux.u_face_ad[1];
    let pf1 = m.aux.p_face_ad[1];
    let eps = 1.0e-5;
    for (slot, perturb) in [
        (
            I_LND_00,
            Box::new(|mc: &mut StarModel, d: f64| mc.lnd[1] += d)
                as Box<dyn Fn(&mut StarModel, f64)>,
        ),
        (
            I_LND_M1,
            Box::new(|mc: &mut StarModel, d: f64| mc.lnd[0] += d),
        ),
        (
            I_LNT_00,
            Box::new(|mc: &mut StarModel, d: f64| mc.ln_t[1] += d),
        ),
        (
            I_LNT_M1,
            Box::new(|mc: &mut StarModel, d: f64| mc.ln_t[0] += d),
        ),
        (
            I_LNR_00,
            Box::new(|mc: &mut StarModel, d: f64| mc.ln_r[1] += d),
        ),
        (
            I_V_00,
            Box::new(|mc: &mut StarModel, d: f64| mc.u[1] += d * 1.0e5),
        ),
        (
            I_V_M1,
            Box::new(|mc: &mut StarModel, d: f64| mc.u[0] += d * 1.0e5),
        ),
    ] {
        let mut mp = m.clone();
        let mut mm = m.clone();
        perturb(&mut mp, eps);
        perturb(&mut mm, -eps);
        sync_ideal_gas_eos(&mut mp);
        sync_ideal_gas_eos(&mut mm);
        do_uface_and_pface(&mut mp).expect("fd +");
        do_uface_and_pface(&mut mm).expect("fd -");
        let denom = if slot == I_V_00 || slot == I_V_M1 {
            2.0 * eps * 1.0e5
        } else {
            2.0 * eps
        };
        let fd_u = (mp.aux.u_face_ad[1].val - mm.aux.u_face_ad[1].val) / denom;
        let fd_p = (mp.aux.p_face_ad[1].val - mm.aux.p_face_ad[1].val) / denom;
        assert_close(
            &format!("u_face_ad[1] partial slot={slot}"),
            uf1.d1[slot],
            fd_u,
            1.0e-8,
            1.0e-6,
        );
        assert_close(
            &format!("p_face_ad[1] partial slot={slot}"),
            pf1.d1[slot],
            fd_p,
            1.0e-8,
            1.0e-4,
        );
    }
}

#[test]
fn test_do1_riemann_momentum_eqn_uses_m_not_m_grav_and_time_centered_area() {
    let ctrl = StarControls {
        u_flag: true,
        using_velocity_time_centering: true,
        include_p_in_velocity_time_centering: true,
        p_theta_for_velocity_time_centering: 0.5,
        ..StarControls::default()
    };
    let mut m = build_synthetic_model(ctrl);
    m.lnd = vec![-2.0, -1.0, 0.0];
    m.ln_t = vec![14.0, 15.0, 16.0];
    sync_ideal_gas_eos(&mut m);
    m.save_start_values().expect("save_start_values");
    m.save_old_values();
    m.u = vec![1.0e5, 2.0e5, 3.0e5];
    m.save_start_values().expect("save_start_values");
    m.ln_r_start[1] -= 0.02;
    m.ln_r_start[2] -= 0.03;
    // Perturb `lnd`, `ln_r`, `u` away from start values and decouple `m_grav`
    // from `m` to isolate the gravity source term's mass array.
    m.lnd[1] += 0.01;
    m.ln_r[1] += 0.01;
    m.u[1] += 5.0e3;
    sync_ideal_gas_eos(&mut m);
    do_uface_and_pface(&mut m).expect("do_uface_and_pface");

    let mut sys_base = new_sys(&m);
    do1_riemann_momentum_eqn(&m, 1, None, &mut sys_base);

    let mut m_alt = m.clone();
    m_alt.m_grav = m_alt.m.iter().map(|&v| 0.5 * v).collect();
    m_alt.controls.drag_coefficient = 5.0;
    m_alt.aux.extra_grav[1] = StarAdOrder1::constant(999.0);
    let mut sys_alt = new_sys(&m_alt);
    do1_riemann_momentum_eqn(&m_alt, 1, None, &mut sys_alt);

    let i_equ = m.layout.i_dv_dt;
    let nvar = sys_base.nvar;
    // Notice `p_face_ad` stays fixed at start-eval above while `do1_riemann_momentum_eqn`
    // must depend on `m[1], m[2]` (not `m_grav`) and must ignore `drag_coefficient`
    // and `extra_grav`.
    assert_close(
        "riemann_momentum independent of m_grav, drag_coefficient, extra_grav",
        sys_alt.equ[nvar + i_equ],
        sys_base.equ[nvar + i_equ],
        1.0e-14,
        0.0,
    );

    // Check analytical Jacobian entries against central FD with live HLLC updates.
    let eps = 1.0e-6;
    let j_u = m.layout.i_u.expect("i_u");
    for (k_var, j_var, step) in [
        (0, m.layout.i_lnd, eps),
        (1, m.layout.i_lnd, eps),
        (2, m.layout.i_lnd, eps),
        (0, m.layout.i_lnt, eps),
        (1, m.layout.i_lnt, eps),
        (2, m.layout.i_lnt, eps),
        (1, m.layout.i_lnr, eps),
        (2, m.layout.i_lnr, eps),
        (0, j_u, 1.0e-3 * m.u[0]),
        (1, j_u, 1.0e-3 * m.u[1]),
        (2, j_u, 1.0e-3 * m.u[2]),
    ] {
        let eval_pert = |sgn: f64| {
            let mut mp = m.clone();
            if j_var == mp.layout.i_lnd {
                mp.lnd[k_var] += sgn * step;
            } else if j_var == mp.layout.i_lnt {
                mp.ln_t[k_var] += sgn * step;
            } else if j_var == mp.layout.i_lnr {
                mp.ln_r[k_var] += sgn * step;
            } else {
                mp.u[k_var] += sgn * step;
            }
            sync_ideal_gas_eos(&mut mp);
            do_uface_and_pface(&mut mp).expect("fd uface_pface");
            let mut sp = new_sys(&mp);
            do1_riemann_momentum_eqn(&mp, 1, None, &mut sp);
            sp.equ[nvar + i_equ]
        };
        let fd = (eval_pert(1.0) - eval_pert(-1.0)) / (2.0 * step);
        let an = unscaled_partial(&sys_base, 1, i_equ, k_var, j_var);
        assert_close(
            &format!("riemann_momentum partial k_var={k_var} j_var={j_var}"),
            an,
            fd,
            1.0e-8,
            1.0e-10,
        );
    }

    // Verify `m[1]` vs `m_grav[1]` enters `gravity_source` when `P_face` is held fixed.
    let mut m_mass = m.clone();
    m_mass.m[1] *= 1.1;
    let mut sys_mass = new_sys(&m_mass);
    do1_riemann_momentum_eqn(&m_mass, 1, None, &mut sys_mass);
    let (_, ir2_00) = get_area_info_opt_time_center(&m, 1);
    let ie_ke = m.energy_start[1] + 0.5 * flexmesa_math::pow2(m.u_start[1]);
    let scal = m.dt * m.u_start[1].abs().max(m.aux.csound_start[1]) / ie_ke;
    let expected_delta =
        scal * (-m.cgrav[1] * 0.5 * (0.1 * m.m[1]) * ir2_00.val);
    assert_close(
        "riemann_momentum sensitivity wrt m[1]",
        sys_mass.equ[nvar + i_equ] - sys_base.equ[nvar + i_equ],
        expected_delta,
        1.0e-12,
        0.0,
    );
}

#[test]
fn test_brunt_disabled_returns_nan_and_eval_surf_bc_rsp_and_irrad_uses_m() {
    let mut model = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load ZAMS model");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    // 1. Disabled Brunt B / N2 return NaN (`brunt.f90:53-57, 128-132`).
    let mut m_no_b = model.clone();
    m_no_b.controls.calculate_brunt_b = false;
    do_brunt_b(&mut m_no_b).expect("do_brunt_b disabled");
    do_brunt_n2(&mut m_no_b).expect("do_brunt_n2 disabled");
    assert!(m_no_b.brunt_b.iter().all(|v| v.is_nan()));
    assert!(m_no_b
        .aux
        .mixing
        .unsmoothed_brunt_b
        .iter()
        .all(|v| v.is_nan()));
    assert!(m_no_b.brunt_n2.iter().all(|v| v.is_nan()));
    assert!(m_no_b
        .aux
        .brunt_n2_composition_term
        .iter()
        .all(|v| v.is_nan()));

    let mut m_no_n2 = model.clone();
    m_no_n2.controls.calculate_brunt_n2 = false;
    do_brunt_b(&mut m_no_n2).expect("do_brunt_b enabled");
    do_brunt_n2(&mut m_no_n2).expect("do_brunt_n2 disabled");
    assert!(m_no_n2.brunt_b.iter().all(|v| v.is_finite()));
    assert!(m_no_n2.brunt_n2.iter().all(|v| v.is_nan()));
    assert!(m_no_n2
        .aux
        .brunt_n2_composition_term
        .iter()
        .all(|v| v.is_nan()));

    // 2. `eval_surf_bc` with `rsp_flag` sets `T_surf = T(1)` and `P_surf = Peos(1)`
    //    with zero partials and tolerates `opacity[0] == 0.0`.
    let mut m_rsp = model.clone();
    m_rsp.controls.rsp_flag = true;
    m_rsp.opacity[0] = 0.0;
    eval_surf_bc(&mut m_rsp).expect("eval_surf_bc rsp_flag");
    assert_close(
        "rsp_flag ln_t",
        m_rsp.surf_pt.ln_t,
        m_rsp.ln_t[0],
        1.0e-15,
        0.0,
    );
    assert_close(
        "rsp_flag ln_p",
        m_rsp.surf_pt.ln_p,
        m_rsp.ln_peos[0],
        1.0e-15,
        0.0,
    );
    assert_eq!(m_rsp.ln_peos_surf_ad.d1, [0.0; NUM_STAR_AD_VARS]);
    assert_eq!(m_rsp.ln_t_surf_ad.d1, [0.0; NUM_STAR_AD_VARS]);
    assert!(m_rsp.ln_peos_surf_ad.val.is_finite());
    assert!(m_rsp.ln_t_surf_ad.val.is_finite());

    // 3. `eval_irradiated_surf` uses `model.m[0]` rather than `model.m_grav[0]`.
    let mut m_irr = model;
    m_irr.controls.atm_option = AtmOption::IrradiatedGrey;
    m_irr.controls.atm_irradiated_opacity = AtmIrradiatedOpacity::Fixed;
    m_irr.controls.atm_irradiated_t_eq = 1000.0;
    m_irr.controls.atm_irradiated_kap_v = 4.0e-3;
    m_irr.controls.atm_irradiated_kap_v_div_kap_th = 0.0;
    m_irr.controls.atm_irradiated_p_surf = 1.0e6;
    m_irr.m_grav[0] = 0.25 * m_irr.m[0];
    eval_surf_bc(&mut m_irr).expect("eval_surf_bc irradiated_grey");

    let expected_irr = eval_irradiated(
        m_irr.lum[0],
        m_irr.r[0],
        m_irr.m[0],
        m_irr.cgrav[0],
        m_irr.controls.atm_irradiated_t_eq,
        m_irr.controls.atm_irradiated_p_surf,
        m_irr.opacity[0],
        m_irr.controls.atm_irradiated_kap_v,
        m_irr.controls.atm_irradiated_kap_v_div_kap_th,
        |_ln_p, _ln_t| unreachable!("fixed mode skips eos_proc"),
        |_ln_rho, _ln_t, _eos| unreachable!("fixed mode skips kap_proc"),
        m_irr.controls.atm_irradiated_errtol,
        0,
        false,
    )
    .expect("eval_irradiated with m[0]");

    assert_close(
        "irradiated_grey uses m[0]",
        m_irr.ln_t_surf_ad.val,
        expected_irr.ln_t,
        1.0e-14,
        0.0,
    );
    assert_close(
        "irradiated_grey tau_factor uses m[0]",
        m_irr.controls.tau_factor,
        expected_irr.tau / (2.0 / 3.0),
        1.0e-14,
        0.0,
    );
}
