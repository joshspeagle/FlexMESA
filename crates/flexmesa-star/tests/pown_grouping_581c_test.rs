// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact IEEE-754 `pow2`/`pow3`/`powi_f90` multiplication-grouping unit tests
//! for `RSP2`, `RSP1`, and pulsation helpers (`#581` part c).

#![forbid(unsafe_code)]
#![allow(clippy::manual_midpoint, clippy::similar_names)]

use flexmesa_const::{
    CGAS, CLIGHT, CRAD, FOUR_THIRDS_PI, L_SUN, M_SUN, PI, PI4, R_SUN,
    STANDARD_CGRAV,
};
use flexmesa_math::{
    exp, pow2, pow3, pow4, pow6, powi_f90, with_numerics_mode, NumericsMode,
};
use flexmesa_star::controls::{StarControls, StarJobControls};
use flexmesa_star::model::StarModel;
use flexmesa_star::pulse_exporters::get_cafein_data;
use flexmesa_star::pulse_utils::{
    build_pulse_points_with_options, PulseExportOptions,
};
use flexmesa_star::rsp2::{eval_rsp2_zone, Rsp2Config, X_ALFAP};
use flexmesa_star::rsp2_ad::eval_rsp2_ptrb_cell_ad;
use flexmesa_star::rsp2_eqns::eval_rsp2_hp_face_zone;
use flexmesa_star::rsp_config::RspConfig;
use flexmesa_star::rsp_def::RspDerivedConsts;
use flexmesa_star::rsp_grid::RspEnvGrid;
use flexmesa_star::rsp_hyd_turb::calc_source_sink;
use flexmesa_star::rsp_hyd_work::RspHydWork;
use flexmesa_star::rsp_turb_partials::RspTurbPartials;
use flexmesa_star::rsp_zone_micro::{LcFilterMode, RspZoneMicro};
use flexmesa_star::star_ad::{I_LND_00, I_W_00};

#[test]
fn rsp2_eval_zone_chi_turb_uses_pow2_rho() {
    let cfg = Rsp2Config::default();
    let rho = 1.409_e-6_f64;
    let r_outer = 7.13_e10_f64;
    let r_inner = 6.91_e10_f64;
    let u_outer = 1.3_e5_f64;
    let u_inner = -0.7_e5_f64;
    let hp_cell = 2.3_e9_f64;
    let vel_w = 3.7_e5_f64;
    let dm = 1.37_e26_f64;

    let res = eval_rsp2_zone(
        vel_w, 1.2_e4, rho, 1.5_e6, 1.5_e8, 0.4, 1.0, 1.2, hp_cell, 0.5,
        r_outer, r_inner, u_outer, u_inner, dm, 0.0, &cfg,
    );

    let r6_cell = 0.5 * (pow6(r_outer) + pow6(r_inner));
    let du_over_r = u_outer / r_outer - u_inner / r_inner;
    let f_chi = (16.0 / 3.0) * PI * (cfg.alfam * cfg.alfa) / dm;
    let expected = f_chi * pow2(rho) * r6_cell * du_over_r * hp_cell * vel_w;
    let naive =
        ((((f_chi * rho) * rho) * r6_cell) * du_over_r) * hp_cell * vel_w;
    assert_ne!(expected.to_bits(), naive.to_bits());
    assert_eq!(res.chi_turb.to_bits(), expected.to_bits());
}

#[test]
fn rsp2_ad_eval_ptrb_cell_ad_uses_pow2_w() {
    let mut controls = StarControls {
        rsp2_flag: true,
        mixing_length_alpha: 1.5,
        using_velocity_time_centering: false,
        ..StarControls::default()
    };
    controls.rsp2.alfap = 1.0;
    controls.rsp2.num_outermost_cells_forced_nonturbulent = 0;
    controls.rsp2.nz_div_ibotom = 0.0;

    let mut model =
        StarModel::new(4, Vec::new(), StarJobControls::default(), controls);
    let lnd = -11.986_3_f64;
    let et = 1.017_3_e10_f64;
    model.lnd[1] = lnd;
    model.aux.rsp.et_rsp[1] = et;

    let ptrb_ad = eval_rsp2_ptrb_cell_ad(&model, 1, false);
    let rho = exp(lnd);
    let vel_w = et.sqrt();
    let coeff = X_ALFAP * 1.0;
    let expected = rho * pow2(vel_w) * coeff;
    let naive = ((rho * vel_w) * vel_w) * coeff;
    assert_ne!(expected.to_bits(), naive.to_bits());
    assert_eq!(ptrb_ad.val.to_bits(), expected.to_bits());
    assert!(ptrb_ad.d1[I_LND_00] > 0.0 && ptrb_ad.d1[I_W_00] > 0.0);
}

#[test]
fn rsp2_eqns_eval_hp_face_zone_uses_pow2_area() {
    let mut controls = StarControls::default();
    controls.rsp2.assume_hse = false;
    controls.rsp2.use_mass_interp_face_values = false;

    let mut model =
        StarModel::new(4, Vec::new(), StarJobControls::default(), controls);
    let r_face = R_SUN * 0.664_9_f64;
    model.r[1] = r_face;
    model.rho[0] = 1.1_e-5;
    model.rho[1] = 1.5_e-5;
    model.ln_peos[0] = 9.83;
    model.ln_peos[1] = 10.20;
    model.dm_bar[1] = 1.37_e25;

    let hp_actual = eval_rsp2_hp_face_zone(&model, 1);
    let half_d1 = 0.5 * model.rho[1];
    let half_d0 = 0.5 * model.rho[0];
    let rho_face = half_d1 + half_d0;
    let dlnp = model.ln_peos[0] - model.ln_peos[1];
    let area_pow2 = PI4 * pow2(r_face);
    let area_naive = (PI4 * r_face) * r_face;
    let expected = -model.dm_bar[1] / (area_pow2 * rho_face * dlnp);
    let naive = -model.dm_bar[1] / (area_naive * rho_face * dlnp);
    assert_ne!(expected.to_bits(), naive.to_bits());
    assert_eq!(hp_actual.to_bits(), expected.to_bits());
}

#[test]
fn rsp_hyd_turb_calc_source_sink_uses_pow2_vol() {
    let mut controls = StarControls::default();
    controls.rsp.alfa = 1.5;
    controls.rsp.gammar = 1.0;
    controls.rsp.nz_div_ibotom = 0.0;

    let mut model =
        StarModel::new(4, Vec::new(), StarJobControls::default(), controls);
    let dc = RspDerivedConsts::from_config(&model.controls.rsp, 4);
    let w_in = 112_740.6_f64;
    let et_in = w_in * w_in;
    model.aux.rsp.hp_face = vec![2.7_e9, 2.7_e9, 2.3_e9, 2.7_e9];
    model.aux.rsp.vol_rsp = vec![137_049.6; 4];
    model.t = vec![11_332.8; 4];
    model.cp = vec![1.5_e8; 4];
    model.opacity = vec![1.37; 4];
    model.aux.rsp.pii_rsp = vec![1.0; 4];
    model.aux.rsp.qq_rsp = vec![1.0_e-5; 4];
    model.pgas = vec![1.0_e6; 4];
    model.prad = vec![1.0_e2; 4];
    model.aux.rsp.et_rsp = vec![et_in; 4];

    let mut wk = RspHydWork::from_model(&model);
    calc_source_sink(&mut model, &dc, &mut wk, 2);

    let pom_r = (dc.gammar * dc.gammar) / (dc.alfa * dc.alfa) * 4.0 * dc.sig;
    let t_k = model.t[2];
    let vol_k = model.aux.rsp.vol_rsp[2];
    let pom2b = 1.0 / (model.cp[2] * model.opacity[2]);
    let pom3_r = wk.w[2] * wk.w[2];
    let hp0_2 = model.aux.rsp.hp_face[2] * model.aux.rsp.hp_face[2];
    let hpi_2 = model.aux.rsp.hp_face[3] * model.aux.rsp.hp_face[3];
    let pom4_r = 0.5 * (hp0_2 + hpi_2);

    let pom2_pow2 = pow3(t_k) * pow2(vol_k) * pom2b;
    let pom2_naive = ((pow3(t_k) * vol_k) * vol_k) * pom2b;
    let expected = pom_r * pom2_pow2 * pom3_r / pom4_r;
    let naive = pom_r * pom2_naive * pom3_r / pom4_r;
    assert_ne!(expected.to_bits(), naive.to_bits());
    assert_eq!(model.aux.rsp.dampr_rsp[2].to_bits(), expected.to_bits());
}

#[test]
fn rsp_turb_partials_evaluate_uses_pow2_vol_and_t() {
    let nzn = 4;
    let rsp_cfg = RspConfig {
        alfa: 1.5,
        gammar: 1.0,
        nz_div_ibotom: 0.0,
        ..RspConfig::default()
    };
    let dc = RspDerivedConsts::from_config(&rsp_cfg, nzn);
    let mut grid = RspEnvGrid::new(nzn);
    let mut mc = RspZoneMicro::new(nzn);
    let mut partials = RspTurbPartials::new(nzn);

    grid.r[1..=4].copy_from_slice(&[4.0_e10, 5.0_e10, 6.0_e10, 7.0_e10]);
    grid.m[1..=4].copy_from_slice(&[1.0_e32, 1.1_e32, 1.2_e32, 1.3_e32]);
    grid.dm.fill(1.0_e26);
    grid.dm_bar.fill(1.0_e26);
    grid.vol[1..=4].copy_from_slice(&[1.37_e5, 137_031.5, 1.39_e5, 1.40_e5]);
    grid.t.fill(11_350.7);
    grid.et.fill(1.2_e10);
    mc.p[1..=4].copy_from_slice(&[2.0_e6, 1.5_e6, 1.1_e6, 0.8_e6]);
    mc.cps.fill(1.5_e8);
    mc.k.fill(1.37);
    mc.qqs.fill(0.0);

    partials.evaluate(
        &dc,
        1.5,
        0.0,
        0.0,
        1.0_e4,
        &mut grid,
        &mc,
        LcFilterMode::Lina { efl02: 1.0_e4 },
    );

    let pr = (dc.gammar * dc.gammar) / (1.5 * 1.5) * 4.0 * dc.sig;
    let hp1_2 = grid.hp_face[1] * grid.hp_face[1];
    let hp2_2 = grid.hp_face[2] * grid.hp_face[2];
    let pr4 = 0.5 * (hp2_2 + hp1_2);
    let tr2 = pr * grid.et[2] / pr4;
    let cpk = mc.cps[2] * mc.k[2];
    let expected = -(tr2 * pow2(grid.vol[2]) * (3.0 * pow2(grid.t[2])) / cpk);
    let naive = -(((tr2 * grid.vol[2]) * grid.vol[2])
        * ((3.0 * grid.t[2]) * grid.t[2])
        / cpk);
    assert_ne!(expected.to_bits(), naive.to_bits());
    assert_eq!(partials.dc_dt_00[2].to_bits(), expected.to_bits());
}

#[test]
fn pulse_utils_center_point_uses_pow2_rho_and_r_in() {
    let mut model = StarModel::new(
        4,
        Vec::new(),
        StarJobControls::default(),
        StarControls::default(),
    );
    model.r_center = 0.0;
    model.r = vec![5.0_e9, 4.0_e9, 3.0_e9, 2.0_e9];
    model.rmid = vec![4.5_e9, 3.5_e9, 2.71_e9, 1.37_e9];
    model.m = vec![1.0_e30, 7.0_e29, 4.0_e29, 1.0_e29];
    model.m_grav = model.m.clone();
    model.dm = vec![2.5_e29; 4];
    model.dq = vec![0.25; 4];
    model.lum = vec![L_SUN; 4];
    model.peos = vec![1.0_e11, 3.0_e11, 7.0_e11, 1.013_7_e12];
    model.pgas = model.peos.clone();
    model.rho = vec![100.0, 120.0, 150.3, 150.3];
    model.t = vec![1.0_e6; 4];
    model.gamma1 = vec![5.0 / 3.0; 4];
    model.grada = vec![0.4; 4];
    model.chi_t = vec![1.0; 4];
    model.chi_rho = vec![1.0; 4];
    model.cv = vec![1.0_e8; 4];
    model.cp = vec![1.5_e8; 4];
    model.opacity = vec![1.0; 4];
    model.eps_nuc = vec![1.0; 4];
    model.gradt = vec![0.25; 4];
    model.grav = vec![1.0_e4; 4];
    model.brunt_n2 = vec![1.0_e-5; 4];
    model.scale_height = vec![1.0_e8; 4];

    let opts = PulseExportOptions {
        add_center_point: true,
        interpolate_rho: false,
        ..PulseExportOptions::default()
    };
    let pts = build_pulse_points_with_options(&model, &opts);

    let rho_c = pts[0].rho;
    let r_in = model.rmid[3];
    let d2p = -FOUR_THIRDS_PI * STANDARD_CGRAV * pow2(rho_c);
    let delta_p = 0.5 * d2p * pow2(r_in);
    let expected = model.peos[3] - delta_p;
    let d2p_naive = ((-FOUR_THIRDS_PI * STANDARD_CGRAV) * rho_c) * rho_c;
    let delta_p_naive = ((0.5 * d2p_naive) * r_in) * r_in;
    let naive = model.peos[3] - delta_p_naive;
    assert_ne!(expected.to_bits(), naive.to_bits());
    assert_eq!(pts[0].p.to_bits(), expected.to_bits());
}

fn make_cafein_test_model() -> StarModel {
    let controls = StarControls {
        calculate_brunt_n2: true,
        tau_factor: 1.0,
        interpolate_rho_for_pulse_data: true,
        gyre_data_schema: 101,
        ..StarControls::default()
    };
    let m_star = 1.027_4_f64 * M_SUN;
    let r_star = 1.014_6_f64 * R_SUN;
    let mut model =
        StarModel::new(5, Vec::new(), StarJobControls::default(), controls);
    model.r = vec![r_star, 5.171_1_e10, 3.5_e10, 2.0_e10, 1.0_e10];
    model.rmid = vec![0.9 * R_SUN, 4.3_e10, 2.75_e10, 1.6_e10, 0.8_e10];
    model.dq = vec![0.2; 5];
    model.m_grav = vec![
        m_star,
        0.8 * m_star,
        0.5 * m_star,
        0.25 * m_star,
        0.1 * m_star,
    ];
    model.m = model.m_grav.clone();
    model.dm = vec![0.2 * m_star; 5];
    model.rho = vec![0.403, 0.403, 0.5, 0.6, 0.7];
    model.t = vec![5_020.3, 5_020.3, 5_150.0, 5_328.7, 5_328.7];
    model.peos = vec![1.37_e13; 5];
    model.gradt = vec![0.25; 5];
    model.grada = vec![0.4; 5];
    model.cp = vec![1.5_e8; 5];
    model.chi_rho = vec![1.0; 5];
    model.chi_t = vec![1.0; 5];
    model.gamma1 = vec![5.0 / 3.0; 5];
    model.opacity = vec![1.37; 5];
    model.eps_nuc = vec![17.3; 5];
    model.grav = vec![1.13_e4; 5];
    model.brunt_n2 = vec![1.0_e-5; 5];
    model.scale_height = vec![1.0_e9; 5];
    model.lum = vec![L_SUN, 0.9 * L_SUN, 0.7 * L_SUN, 0.4 * L_SUN, 0.1 * L_SUN];
    model.ln_s = vec![20.0; 5];
    model
}

fn eval_center_const(r1: f64, r2: f64, val: f64) -> f64 {
    (val * r2 * r2 - val * r1 * r1) / (r2 * r2 - r1 * r1)
}

#[test]
fn pulse_exporters_cafein_uses_pow2_pow3_and_powi4() {
    with_numerics_mode(NumericsMode::Strict, || {
        let model = make_cafein_test_model();
        let (global, pd) = get_cafein_data(&model, true, true, false)
            .expect("get_cafein_data succeeds");

        let m_star = model.m_grav[0];
        let r_star = model.r[0];
        let g_const = STANDARD_CGRAV;
        let s_units = (pow3(r_star) / (g_const * m_star)).sqrt();
        let s2 = s_units * s_units;

        let erg_exp = g_const * pow2(m_star) / r_star;
        let erg_naive = (g_const * m_star) * m_star / r_star;
        assert_ne!(erg_exp.to_bits(), erg_naive.to_bits());
        assert_eq!(global[10].to_bits(), erg_exp.to_bits());

        let k_exp = g_const * pow2(m_star) / (r_star * CGAS);
        let k_naive = (g_const * m_star) * m_star / (r_star * CGAS);
        assert_ne!(k_exp.to_bits(), k_naive.to_bits());
        assert_eq!(global[11].to_bits(), k_exp.to_bits());

        // `k = 2` face corresponds to `pd[4]` in center-to-surface order.
        let rad2 = 5.171_1_e10_f64;
        let rho2 = 0.403_f64;
        let t2 = 5_020.3_f64;
        let m2 = 0.8 * m_star;
        let v2 = rho2 * 1.13_e4 * rad2 / 1.37_e13;

        let u_exp = PI4 * rho2 * pow3(rad2) / m2;
        let u_naive = ((((PI4 * rho2) * rad2) * rad2) * rad2) / m2;
        assert_ne!(u_exp.to_bits(), u_naive.to_bits());
        assert_eq!(pd[4][7].to_bits(), u_exp.to_bits());

        let l2_exp = ((5.0 / 3.0) * 1.37_e13 / (rho2 * pow2(rad2))) * s2;
        let l2_naive = ((5.0 / 3.0) * 1.37_e13 / ((rho2 * rad2) * rad2)) * s2;
        assert_ne!(l2_exp.to_bits(), l2_naive.to_bits());
        assert_eq!(pd[4][10].to_bits(), l2_exp.to_bits());

        let lr_exp =
            16.0 * PI * rad2 * CRAD * CLIGHT * powi_f90(t2, 4) * 0.25 * v2
                / (3.0 * 1.37 * rho2);
        let lr_ltr =
            16.0 * PI * rad2 * CRAD * CLIGHT * t2 * t2 * t2 * t2 * 0.25 * v2
                / (3.0 * 1.37 * rho2);
        let lr_pow4 = 16.0 * PI * rad2 * CRAD * CLIGHT * pow4(t2) * 0.25 * v2
            / (3.0 * 1.37 * rho2);
        let c3_exp = PI4 * pow3(rad2) * rho2 * 17.3 / lr_exp;
        let c3_ltr = PI4 * pow3(rad2) * rho2 * 17.3 / lr_ltr;
        let c3_pow4 = PI4 * pow3(rad2) * rho2 * 17.3 / lr_pow4;
        assert_ne!(c3_exp.to_bits(), c3_ltr.to_bits());
        assert_ne!(c3_exp.to_bits(), c3_pow4.to_bits());
        assert_eq!(pd[4][22].to_bits(), c3_exp.to_bits());

        // Center point (`pd[0]`) checks `powi_f90(t, 4)` in `c_3` and `c_4`.
        let tc = eval_center_const(0.8_e10, 1.6_e10, 5_328.7);
        let p_c = eval_center_const(0.8_e10, 1.6_e10, 1.37_e13);
        let eps_c = eval_center_const(0.8_e10, 1.6_e10, 17.3);
        let kap_c = eval_center_const(0.8_e10, 1.6_e10, 1.37);
        let cp_c = eval_center_const(0.8_e10, 1.6_e10, 1.5_e8);
        let nab_c = eval_center_const(1.0_e10, 2.0_e10, 0.25);
        let cgrav_c = eval_center_const(1.0_e10, 2.0_e10, g_const);
        let denom_exp =
            16.0 * PI * CRAD * CLIGHT * powi_f90(tc, 4) * nab_c * cgrav_c;
        let denom_ltr =
            16.0 * PI * CRAD * CLIGHT * tc * tc * tc * tc * nab_c * cgrav_c;
        let denom_p4 = 16.0 * PI * CRAD * CLIGHT * pow4(tc) * nab_c * cgrav_c;
        let num_c3 = 9.0 * eps_c * kap_c * p_c;
        let ctr_c3_exp = num_c3 / denom_exp;
        assert_ne!(ctr_c3_exp.to_bits(), (num_c3 / denom_ltr).to_bits());
        assert_ne!(ctr_c3_exp.to_bits(), (num_c3 / denom_p4).to_bits());
        assert_eq!(pd[0][22].to_bits(), ctr_c3_exp.to_bits());

        let num_c4 = 9.0 * tc * cp_c * kap_c * p_c;
        let dyn_scale = (g_const * m_star / pow3(r_star)).sqrt();
        let ctr_c4_exp = (num_c4 / denom_exp) * dyn_scale;
        let ctr_c4_ltr = (num_c4 / denom_ltr) * dyn_scale;
        let ctr_c4_p4 = (num_c4 / denom_p4) * dyn_scale;
        assert_ne!(ctr_c4_exp.to_bits(), ctr_c4_ltr.to_bits());
        assert_ne!(ctr_c4_exp.to_bits(), ctr_c4_p4.to_bits());
        assert_eq!(pd[0][23].to_bits(), ctr_c4_exp.to_bits());
    });
}
