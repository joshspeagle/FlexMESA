// SPDX-License-Identifier: GPL-3.0-or-later
//! Invariant unit tests for `#581` Tier-1 `powN` multiplication grouping in
//! `flexmesa-star` (`hydro_momentum`, `hydro_work`, `riemann_eqns`, and
//! `adjust_mesh_split_merge`).

#![forbid(unsafe_code)]
#![allow(
    clippy::excessive_precision,
    clippy::similar_names,
    clippy::unreadable_literal
)]

use flexmesa_const::FOUR_PI;
use flexmesa_math::pow2;
use flexmesa_star::hydro_momentum::eval_mlt_pturb_cell_ad;
use flexmesa_star::hydro_work::{
    eval1_a_times_v_face_v_flag_ad, eval1_work_v_flag_ad,
};
use flexmesa_star::riemann_eqns::eval_u_flag_total_energy_rates_ad;
use flexmesa_star::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LNR_00, NUM_STAR_AD_VARS,
};
use flexmesa_star::{
    load_zams_model, set_hydro_vars, set_mlt_vars, StarControls,
    StarJobControls,
};

fn solar_zams_controls() -> StarControls {
    StarControls {
        zbase: 0.02,
        ..StarControls::default()
    }
}

#[test]
fn test_eval_mlt_pturb_cell_ad_uses_pow2_and_div_three_grouping() {
    let mut model =
        load_zams_model(solar_zams_controls(), StarJobControls::default())
            .expect("load ZAMS");
    let k = 1;
    let vc_old = 1.23e5_f64;
    let factor = 0.7_f64;
    model.controls.mlt_pturb_factor = factor;
    model.aux.mlt_vc_old = vec![vc_old; model.nz];

    let actual = eval_mlt_pturb_cell_ad(&model, k, false);
    let rho_ad = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
    let expected = (factor * pow2(vc_old) * rho_ad) / 3.0;
    let legacy = rho_ad * ((1.0 / 3.0) * factor * vc_old * vc_old);

    assert_ne!(expected.val.to_bits(), legacy.val.to_bits());
    assert_eq!(actual.val.to_bits(), expected.val.to_bits());
    assert_eq!(
        actual.d1[I_LND_00].to_bits(),
        expected.d1[I_LND_00].to_bits()
    );
}

#[test]
fn test_center_work_and_area_velocity_use_pow2_rcenter_grouping() {
    let mut model =
        load_zams_model(solar_zams_controls(), StarJobControls::default())
            .expect("load ZAMS");
    let nz = model.nz;

    let r_av = 3.14e9_f64;
    let v_av = 1.3e5_f64;
    model.r_center = r_av;
    model.v_center = v_av;
    let av = eval1_a_times_v_face_v_flag_ad(&model, nz, false);
    let expected_av = FOUR_PI * pow2(r_av) * v_av;
    let legacy_av = FOUR_PI * r_av * r_av * v_av;
    assert_ne!(expected_av.to_bits(), legacy_av.to_bits());
    assert_eq!(av.val.to_bits(), expected_av.to_bits());

    let r_w = 5.13e10_f64;
    let p_in = 1.2e15_f64;
    let v_w = 1.3e5_f64;
    model.r_center = r_w;
    model.v_center = v_w;
    model.peos_start = vec![p_in; nz];
    let (work, _, _) = eval1_work_v_flag_ad(&model, nz, false, false);
    let expected_work = FOUR_PI * pow2(r_w) * p_in * v_w;
    let legacy_work = FOUR_PI * r_w * r_w * p_in * v_w;
    assert_ne!(expected_work.to_bits(), legacy_work.to_bits());
    assert_eq!(work.val.to_bits(), expected_work.to_bits());
}

#[test]
fn test_set_mlt_vars_mlt_cdc_ad_uses_pow2_grouping() {
    let mut model =
        load_zams_model(solar_zams_controls(), StarJobControls::default())
            .expect("load ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    set_mlt_vars(&mut model).expect("set_mlt_vars");

    let k = (1..model.nz)
        .find(|&idx| model.mlt_d[idx] > 0.0)
        .expect("at least one convective interior face in 1 Msun ZAMS");
    let (alfa, beta_w) = model.get_face_weights(k);
    let rho_00 =
        StarAdOrder1::from_dlnd_dlnt_00(model.rho[k], model.rho[k], 0.0);
    let rho_m1 = StarAdOrder1::from_dlnd_dlnt_m1(
        model.rho[k - 1],
        model.rho[k - 1],
        0.0,
    );
    let rho_f = rho_00 * alfa + rho_m1 * beta_w;
    let r_k = if model.r[k] > 0.0 {
        let mut r_ad = StarAdOrder1::constant(model.r[k]);
        r_ad.d1[I_LNR_00] = model.r[k];
        r_ad
    } else {
        StarAdOrder1::variable(model.ln_r[k], I_LNR_00).exp()
    };
    let d_f = model.mlt_d_ad[k];
    let area_rho_ad = FOUR_PI * r_k.pow2() * rho_f;
    let expected = d_f * area_rho_ad.pow2();

    assert_eq!(model.mlt_cdc_ad[k].val.to_bits(), expected.val.to_bits());
    for slot in 0..NUM_STAR_AD_VARS {
        assert_eq!(
            model.mlt_cdc_ad[k].d1[slot].to_bits(),
            expected.d1[slot].to_bits(),
            "mlt_cdc_ad d1[{slot}] mismatch at face {k}"
        );
    }

    let mut r_witness = StarAdOrder1::constant(3.14e9_f64);
    r_witness.d1[I_LNR_00] = 3.14e9_f64;
    let rho_witness = StarAdOrder1::from_dlnd_dlnt_00(1.23e-2, 1.23e-2, 0.0);
    let pow2_area = FOUR_PI * r_witness.pow2() * rho_witness;
    let l2r_area = FOUR_PI * r_witness * r_witness * rho_witness;
    assert_ne!(pow2_area.val.to_bits(), l2r_area.val.to_bits());
}

#[test]
fn test_riemann_u_flag_center_work_uses_pow2_rcenter_grouping() {
    let mut model =
        load_zams_model(solar_zams_controls(), StarJobControls::default())
            .expect("load ZAMS");
    model.aux.need_to_setvars = true;
    set_hydro_vars(&mut model, false, false).expect("init set_hydro_vars");
    model.aux.need_to_setvars = false;
    model.save_start_values().expect("save_start_values");

    let nz = model.nz;
    let kc = nz - 1;
    let r_w = 5.13e10_f64;
    let p_in = 1.2e15_f64;
    let v_w = 1.3e5_f64;

    model.controls.u_flag = true;
    model.u.fill(0.0);
    model.u_start.fill(0.0);
    model.cgrav.fill(0.0);
    model.aux.u_face_ad = vec![StarAdOrder1::ZERO; nz];
    model.aux.u_face_start = vec![0.0; nz];
    model.energy_start[kc] = model.ln_e_ad[kc].exp().val;
    model.dm[kc] = 1.0;
    model.r_center = r_w;
    model.v_center = v_w;
    model.peos_start = vec![p_in; nz];

    let rates = eval_u_flag_total_energy_rates_ad(&model, kc, 1.0);
    let expected = -(FOUR_PI * pow2(r_w) * p_in * v_w);
    let legacy = -(FOUR_PI * r_w * r_w * p_in * v_w);
    assert_ne!(expected.to_bits(), legacy.to_bits());
    assert_eq!(rates.val.to_bits(), expected.to_bits());
}
