// SPDX-License-Identifier: LGPL-3.0-or-later
//! Targeted `#516` MESA `26.4.1` `approx21_cr60_plus_co56` H-burning parity tests.
//!
//! Pins exact IEEE-754 hex oracles from MESA `26.4.1` `CALL 1` model-1
//! stepdumps across `star/20M_z2m2_high_rotation` (including zone 650 at
//! `logT = 7.2075` and central zone 945), `star/twin_studies` (zone 520 at
//! `logT = 7.3378` and central zone 630), and
//! `star/20M_pre_ms_to_core_collapse` (zone 467 at `logT = 7.2288`), plus
//! analytical vs finite-difference Jacobian verification at H-burning
//! temperatures (`6.8 < logT < 7.6`).
//!
//! Also verifies the `#516` root-cause identity: before `#464` added the
//! second `a2_he3_he4` term to `q_he4` (`net/private/net_approx21.f90:1113`),
//! omitting `y(he3) * y(he4) * rate(irhe3ag) * mion(he4) * QCONV` shifted
//! zone-650 `eps_nuc` from `0x4005_5a53_abe0_d9f9` to exact
//! `0x4005_6e0a_0d6e_be9e` (`+3.606e-3` rel).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{AVO, CLIGHT, LN10, QCONV};
use flexmesa_eos::{EosConfig, I_ETA};
use flexmesa_math::{exp10, with_numerics_mode, NumericsMode};
use flexmesa_net::approx21_eps::{
    approx21_d_epsneu_dy, approx21_eval_ppii_fraction, global_approx21_qs,
};
use flexmesa_net::{
    eval_net_with_work, net_get, with_thread_local_work, Approx21Indices,
    NetEvalResult, NetGeneralInfo, Quad, NUM_RVS,
};
use flexmesa_star::micro::{do_eos, do_net, resolve_model_net, set_basic_vars};
use flexmesa_star::{StarControls, StarJobControls, StarModel};

const APPROX21_NET: &str = "approx21_cr60_plus_co56.net";

struct ZoneOracle {
    label: &'static str,
    zbase: f64,
    lnd_bits: u64,
    ln_t_bits: u64,
    xa_bits: [u64; 22],
    expected_eps_nuc_bits: u64,
}

fn ulp_distance(a: f64, b: f64) -> u64 {
    a.to_bits().abs_diff(b.to_bits())
}

fn eval_single_zone_do_net(oracle: &ZoneOracle) -> StarModel {
    let g = NetGeneralInfo::from_name(APPROX21_NET)
        .expect("load approx21_cr60_plus_co56.net");
    assert_eq!(g.num_isos, 22);
    let job = StarJobControls {
        net_name: String::from(APPROX21_NET),
        ..Default::default()
    };
    let controls = StarControls {
        initial_z: oracle.zbase,
        zbase: oracle.zbase,
        ..Default::default()
    };
    let mut model = StarModel::new(1, g.chem_id, job, controls);
    model.eos_config = EosConfig::default();
    model.lnd[0] = f64::from_bits(oracle.lnd_bits);
    model.ln_t[0] = f64::from_bits(oracle.ln_t_bits);
    model.ln_r[0] = 25.0;
    model.dq[0] = 1.0;
    model.mstar = 20.0 * 1.988_409_870_698_051e33;
    model.xmstar = model.mstar;
    model.dm[0] = model.mstar;
    for (slot, &bits) in oracle.xa_bits.iter().enumerate() {
        model.xa[slot] = f64::from_bits(bits);
    }

    set_basic_vars(&mut model).expect("set_basic_vars");
    do_eos(&mut model).expect("do_eos");
    let net_info = resolve_model_net(&model).expect("resolve_model_net");
    do_net(&mut model, &net_info, false).expect("do_net");
    model
}

const ORACLES: &[ZoneOracle] = &[
    // star/20M_z2m2_high_rotation model 1 zone 650 (logT = 7.2075)
    ZoneOracle {
        label: "20M_z2m2_high_rotation m1 z650",
        zbase: 0.02,
        lnd_bits: 0xbfb8_6928_c537_341a,
        ln_t_bits: 0x4030_9892_b542_d309,
        xa_bits: [
            0x2b71_4f4a_252c_8053, // neut
            0x3fe7_210d_abfe_3e67, // h1
            0x2b71_4f4a_252c_8053, // prot
            0x3efc_b4e5_559a_eeb6, // he3
            0x3fd0_7949_ed2e_9b39, // he4
            0x3f6b_ed83_9348_00e5, // c12
            0x3f50_6338_aa53_3068, // n14
            0x3f83_012d_bf9b_df6d, // o16
            0x3f60_e590_33ac_7b21, // ne20
            0x3f70_8546_429d_9aad, // mg24
            0x2b71_4f4a_252c_8053, // si28
            0x2b71_4f4a_252c_8053, // s32
            0x2b71_4f4a_252c_8053, // ar36
            0x2b71_4f4a_252c_8053, // ca40
            0x2b71_4f4a_252c_8053, // ti44
            0x2b71_4f4a_252c_8053, // cr48
            0x2b71_4f4a_252c_8053, // cr60
            0x2b71_4f4a_252c_8053, // fe52
            0x2b71_4f4a_252c_8053, // fe54
            0x2b71_4f4a_252c_8053, // fe56
            0x2b71_4f4a_252c_8053, // co56
            0x2b71_4f4a_252c_8053, // ni56
        ],
        expected_eps_nuc_bits: 0x4005_5a53_abe0_d9f9,
    },
    // star/20M_z2m2_high_rotation model 1 zone 945 (core, logT = 7.5657)
    ZoneOracle {
        label: "20M_z2m2_high_rotation m1 z945",
        zbase: 0.02,
        lnd_bits: 0x3ffa_ef03_2328_0eb4,
        ln_t_bits: 0x4031_6bb8_364b_a790,
        xa_bits: [
            0x2b71_4f4a_252c_8053, // neut
            0x3fe7_19f5_5c2b_f076, // h1
            0x2b71_4f4a_252c_8053, // prot
            0x3e9c_3954_63e6_cd63, // he3
            0x3fd0_7f26_79b2_f421, // he4
            0x3f17_0790_8d91_03f7, // c12
            0x3f74_7df8_e766_716b, // n14
            0x3f82_b48e_88f4_500c, // o16
            0x3f60_e590_33ac_7b21, // ne20
            0x3f70_8546_429d_9aad, // mg24
            0x2b71_4f4a_252c_8053, // si28
            0x2b71_4f4a_252c_8053, // s32
            0x2b71_4f4a_252c_8053, // ar36
            0x2b71_4f4a_252c_8053, // ca40
            0x2b71_4f4a_252c_8053, // ti44
            0x2b71_4f4a_252c_8053, // cr48
            0x2b71_4f4a_252c_8053, // cr60
            0x2b71_4f4a_252c_8053, // fe52
            0x2b71_4f4a_252c_8053, // fe54
            0x2b71_4f4a_252c_8053, // fe56
            0x2b71_4f4a_252c_8053, // co56
            0x2b71_4f4a_252c_8053, // ni56
        ],
        expected_eps_nuc_bits: 0x4108_0446_ea2c_dd37,
    },
    // star/twin_studies model 1 zone 520 (logT = 7.3378)
    ZoneOracle {
        label: "twin_studies m1 z520",
        zbase: 0.02,
        lnd_bits: 0x3fed_abb6_a549_e8b7,
        ln_t_bits: 0x4030_e55f_68b6_0645,
        xa_bits: [
            0x2b61_640d_3e29_bfe3, // neut
            0x3fe6_6207_5657_eb83, // h1
            0x2b61_640d_3e29_bfe3, // prot
            0x3ec2_c80f_b4e4_2549, // he3
            0x3fd1_ecf4_b110_73af, // he4
            0x3ef6_1c5a_b354_16e3, // c12
            0x3f74_687c_cade_85d6, // n14
            0x3f83_1427_33df_9a6f, // o16
            0x3f60_f9c3_5efb_5125, // ne20
            0x3f70_9906_5127_feda, // mg24
            0x2b61_640d_3e29_bfe3, // si28
            0x2b61_640d_3e29_bfe3, // s32
            0x2b61_640d_3e29_bfe3, // ar36
            0x2b61_640d_3e29_bfe3, // ca40
            0x2b61_640d_3e29_bfe3, // ti44
            0x2b61_640d_3e29_bfe3, // cr48
            0x2b61_640d_3e29_bfe3, // cr60
            0x2b61_640d_3e29_bfe3, // fe52
            0x2b61_640d_3e29_bfe3, // fe54
            0x2bf1_2d4c_1c37_c8bb, // fe56
            0x2b61_640d_3e29_bfe3, // co56
            0x2b61_640d_3e29_bfe3, // ni56
        ],
        expected_eps_nuc_bits: 0x4033_26bc_024d_bfb8,
    },
    // star/twin_studies model 1 zone 630 (core, logT = 7.5378)
    ZoneOracle {
        label: "twin_studies m1 z630",
        zbase: 0.02,
        lnd_bits: 0x3ffd_3692_851d_903b,
        ln_t_bits: 0x4031_5b38_4c1e_b18e,
        xa_bits: [
            0x2b61_7621_07cf_6ded, // neut
            0x3fe5_c301_b033_399f, // h1
            0x2b61_7621_07cf_6ded, // prot
            0x3e79_79b8_153e_1814, // he3
            0x3fd3_3293_96f3_e6bb, // he4
            0x3f09_6753_8e43_99d9, // c12
            0x3f80_a810_f145_62b7, // n14
            0x3f77_41a4_41be_57c3, // o16
            0x3f60_f98a_73d7_5b3b, // ne20
            0x3f70_98ce_aa60_1a08, // mg24
            0x2b61_7621_07cf_6ded, // si28
            0x2b61_7621_07cf_6ded, // s32
            0x2b61_7621_07cf_6ded, // ar36
            0x2b61_7621_07cf_6ded, // ca40
            0x2b61_7621_07cf_6ded, // ti44
            0x2b61_7621_07cf_6ded, // cr48
            0x2b61_7621_07cf_6ded, // cr60
            0x2b61_7621_07cf_6ded, // fe52
            0x2b61_7621_07cf_6ded, // fe54
            0x2bf1_2d19_6db7_432c, // fe56
            0x2b61_640d_3e29_bfe1, // co56
            0x2b61_640d_3e29_bfe1, // ni56
        ],
        expected_eps_nuc_bits: 0x40f6_cadf_7ddf_47f4,
    },
    // star/20M_pre_ms_to_core_collapse model 2099 zone 467 (H-shell, logT = 7.2288)
    ZoneOracle {
        label: "20M_pre_ms_to_core_collapse m2099 z467",
        zbase: 0.0142,
        lnd_bits: 0xc002_806c_7467_6aae,
        ln_t_bits: 0x4030_a51c_ea57_bf3d,
        xa_bits: [
            0x2b61_8efa_738e_ec70, // neut
            0x3fd0_e380_fa85_2e19, // h1
            0x2b61_8efa_738e_ec70, // prot
            0x3e31_0d7e_7f56_e252, // he3
            0x3fe7_1cb8_420a_ef95, // he4
            0x3f1f_210f_670e_af5c, // c12
            0x3f81_d270_1bbe_45b0, // n14
            0x3f29_7c54_18c4_f375, // o16
            0x3f55_b6bd_f2df_d9a5, // ne20
            0x3f6c_d151_6b9b_c5a0, // mg24
            0x2b61_8efa_738e_f698, // si28
            0x2b61_8efa_738e_ec70, // s32
            0x2b61_8efa_738e_ec70, // ar36
            0x2b61_8efa_738e_ec70, // ca40
            0x2b61_8efa_738e_ec70, // ti44
            0x2b61_8efa_738e_ec70, // cr48
            0x2b61_8efa_738e_ec70, // cr60
            0x2b61_8efa_738e_ec70, // fe52
            0x2b61_8efa_738e_ec70, // fe54
            0x2c17_fa43_29ce_1b4d, // fe56
            0x2b61_acf0_9c41_504e, // co56
            0x0000_0000_0000_0000, // ni56
        ],
        expected_eps_nuc_bits: 0x3f89_2922_e6c4_150c,
    },
];

#[test]
fn test_approx21_516_h_burning_zero_ulp_vs_mesa_stepdumps() {
    with_numerics_mode(NumericsMode::Strict, || {
        for oracle in ORACLES {
            let model = eval_single_zone_do_net(oracle);
            let got = model.eps_nuc[0];
            let exp = f64::from_bits(oracle.expected_eps_nuc_bits);
            let ulp = ulp_distance(got, exp);
            assert_eq!(
                ulp,
                0,
                "{}: eps_nuc got {:.17e} (0x{:016x}), expected {:.17e} (0x{:016x}), ulp={ulp}",
                oracle.label,
                got,
                got.to_bits(),
                exp,
                oracle.expected_eps_nuc_bits
            );
            let cat_sum: f64 = model.aux.eps_nuc_categories[0].iter().sum();
            assert!(
                (cat_sum - got).abs() <= 1.0e-14 * got.abs().max(1.0e-30),
                "{}: category sum mismatch {cat_sum:.17e} vs {got:.17e}",
                oracle.label
            );
        }
    });
}

#[test]
fn test_approx21_516_zone650_missing_a2_he3_he4_reproduces_pre_464_offset() {
    // Before #464 aligned `approx21_dydt.rs` with MESA `net_approx21.f90:1113`,
    // omitting the second `a2_he3_he4 = y(he3) * y(he4) * rate(irhe3ag)` term
    // shifted `eps_nuc` upward by `a2_he3_he4 * mion(he4) * QCONV`.
    with_numerics_mode(NumericsMode::Strict, || {
        let model = eval_single_zone_do_net(&ORACLES[0]);
        let net_info = resolve_model_net(&model).expect("resolve_model_net");
        let cat = NuclideCatalog::builtin_ref();
        let idx = |name: &str| -> usize {
            let cid = cat.get_nuclide_index(name).expect(name);
            net_info.net_iso[cid] - 1
        };
        let ihe3 = idx("he3");
        let ihe4 = idx("he4");
        let ir_he3ag =
            net_info.reaction_index_by_name("r_he3_ag_be7").unwrap() - 1;

        let rf = vec![1.0_f64; net_info.num_reactions];
        let mut dydt = vec![[Quad::ZERO; NUM_RVS]; net_info.num_isos];
        let mut rate_he3ag = 0.0_f64;
        with_thread_local_work(&net_info, |w| {
            let (eps, _, _, _) = eval_net_with_work(
                &net_info,
                w,
                &mut dydt,
                false,
                &model.xa,
                model.t[0],
                model.ln_t[0] / LN10,
                model.rho[0],
                model.lnd[0] / LN10,
                model.abar[0],
                model.zbar[0],
                model.z2bar[0],
                model.ye[0],
                model.eta[0],
                model.eos_d_dlnt[0][I_ETA],
                model.eos_d_dlnd[0][I_ETA],
                &rf,
                model.controls.weak_rate_factor,
                model.controls.screening_mode,
                false,
                false,
                false,
            )
            .expect("eval_net_with_work");
            assert_eq!(eps.to_bits(), ORACLES[0].expected_eps_nuc_bits);
            rate_he3ag = w.rate_screened[ir_he3ag];
        });

        let a_he3 = cat.get_by_id(net_info.chem_id[ihe3]).unwrap();
        let a_he4 = cat.get_by_id(net_info.chem_id[ihe4]).unwrap();
        let z_plus_n_he3 = f64::from(
            u32::try_from(a_he3.z + a_he3.n).expect("he3 A fits u32"),
        );
        let z_plus_n_he4 = f64::from(
            u32::try_from(a_he4.z + a_he4.n).expect("he4 A fits u32"),
        );
        let y_he3 = model.xa[ihe3] / z_plus_n_he3;
        let y_he4 = model.xa[ihe4] / z_plus_n_he4;
        let shifted = model.eps_nuc[0]
            + (y_he3 * y_he4 * rate_he3ag) * a_he4.mass_excess * QCONV;
        assert_eq!(shifted.to_bits(), 0x4005_6e0a_0d6e_be9e);

        // Also check the pre-#464 `a2_he3_he4` offset on twin_studies z520 and
        // 20M_pre_ms_to_core_collapse z467 (`+1.28e-2` and `+7.9e-5` reported in #516).
        for (oracle_idx, min_rel, max_rel) in
            [(2_usize, 1.25e-2, 1.32e-2), (4_usize, 7.5e-5, 8.3e-5)]
        {
            let m2 = eval_single_zone_do_net(&ORACLES[oracle_idx]);
            let net2 = resolve_model_net(&m2).expect("resolve_model_net");
            let mut dydt2 = vec![[Quad::ZERO; NUM_RVS]; net2.num_isos];
            let mut r_he3ag_2 = 0.0_f64;
            with_thread_local_work(&net2, |w| {
                eval_net_with_work(
                    &net2,
                    w,
                    &mut dydt2,
                    false,
                    &m2.xa,
                    m2.t[0],
                    m2.ln_t[0] / LN10,
                    m2.rho[0],
                    m2.lnd[0] / LN10,
                    m2.abar[0],
                    m2.zbar[0],
                    m2.z2bar[0],
                    m2.ye[0],
                    m2.eta[0],
                    m2.eos_d_dlnt[0][I_ETA],
                    m2.eos_d_dlnd[0][I_ETA],
                    &rf,
                    m2.controls.weak_rate_factor,
                    m2.controls.screening_mode,
                    false,
                    false,
                    false,
                )
                .expect("eval_net_with_work");
                r_he3ag_2 = w.rate_screened[ir_he3ag];
            });
            let yh3 = m2.xa[ihe3] / z_plus_n_he3;
            let yh4 = m2.xa[ihe4] / z_plus_n_he4;
            let delta = (yh3 * yh4 * r_he3ag_2) * a_he4.mass_excess * QCONV;
            let rel = delta / m2.eps_nuc[0];
            assert!(
                rel > min_rel && rel < max_rel,
                "{}: expected pre-#464 offset in ({min_rel:.2e}, {max_rel:.2e}), got {rel:.6e}",
                ORACLES[oracle_idx].label
            );
        }
    });
}

fn check_thermo_axis_fd(
    label: &str,
    axis: &str,
    g: &NetGeneralInfo,
    cat: &NuclideCatalog,
    idx21: &Approx21Indices,
    y: &[f64],
    dvar: f64,
    out0: &NetEvalResult,
    out_p: &NetEvalResult,
    out_m: &NetEvalResult,
    analytic_deps_nuc_dvar: f64,
    analytic_dxdt_dvar: &[f64],
    rate_screened_dvar: &[f64],
) {
    let qs = global_approx21_qs();
    let f_ii_p = approx21_eval_ppii_fraction(idx21, y, &out_p.rate_screened);
    let f_ii_m = approx21_eval_ppii_fraction(idx21, y, &out_m.rate_screened);
    let df_ii_dvar = (f_ii_p - f_ii_m) / dvar;
    let d_epsneu_fii_dvar = (qs.qneu_rpp2 - qs.qneu_rpp3)
        * y[idx21.ihe3]
        * y[idx21.ihe4]
        * out0.rate_screened[idx21.irhe3ag]
        * df_ii_dvar
        * QCONV;
    let d_epsneu_rnpg_missing = qs.qneu_rnpg
        * y[idx21.in14]
        * y[idx21.ih1]
        * ((out_p.rate_screened[idx21.irnpg] * out_p.rate_screened[idx21.ifa]
            - out_m.rate_screened[idx21.irnpg]
                * out_m.rate_screened[idx21.ifa])
            / dvar
            - rate_screened_dvar[idx21.irnpg] * rate_screened_dvar[idx21.ifa])
        * QCONV;

    let fd_deps_nuc = (out_p.eps_nuc - out_m.eps_nuc) / dvar;
    let deps_corrected =
        analytic_deps_nuc_dvar - d_epsneu_fii_dvar - d_epsneu_rnpg_missing;
    let rel_nuc =
        (deps_corrected - fd_deps_nuc).abs() / fd_deps_nuc.abs().max(1.0e-30);
    assert!(
        rel_nuc < 1.0e-6,
        "{label} d_eps_nuc/{axis}: corrected rel_err={rel_nuc:.3e}"
    );

    let fd_deps_total = ((out_p.eps_nuc + out_p.eps_neu_total)
        - (out_m.eps_nuc + out_m.eps_neu_total))
        / dvar;
    let deps_total_from_dxdt: f64 = -AVO
        * CLIGHT
        * CLIGHT
        * g.chem_id
            .iter()
            .enumerate()
            .map(|(i, &cid)| {
                let a = f64::from(
                    u32::try_from(cat.get_by_id(cid).unwrap().z_plus_n)
                        .unwrap(),
                );
                (analytic_dxdt_dvar[i] / a) * g.mion[i]
            })
            .sum::<f64>();
    let rel_tot = (deps_total_from_dxdt - fd_deps_total).abs()
        / fd_deps_total.abs().max(1.0e-30);
    assert!(
        rel_tot < 1.0e-6,
        "{label} d_eps_total/{axis}: rel_err={rel_tot:.3e}"
    );

    for (i, &cid) in g.chem_id.iter().enumerate() {
        let name = cat.get_by_id(cid).map_or("?", |n| n.name.as_str());
        let fd_dxdt = (out_p.dxdt[i] - out_m.dxdt[i]) / dvar;
        if fd_dxdt.abs() > 1.0e-26 {
            let rel = (analytic_dxdt_dvar[i] - fd_dxdt).abs() / fd_dxdt.abs();
            assert!(
                rel < 1.0e-6,
                "{label} d(dxdt[{name}])/{axis}: rel={rel:.3e}"
            );
        }
    }
}

#[test]
fn test_approx21_516_h_burning_analytical_jacobian_vs_finite_differences() {
    with_numerics_mode(NumericsMode::Strict, || {
        let g = NetGeneralInfo::from_name(APPROX21_NET).expect("load approx21");
        let cat = NuclideCatalog::builtin_ref();
        let idx21 = Approx21Indices::from_net(&g);
        let rf = vec![1.0_f64; g.num_reactions];

        // Check analytical `(T, rho, X)` derivatives against central FD at both
        // `20M_z2m2_high_rotation` zone 650 (`logT=7.21`) and `twin_studies`
        // zone 520 (`logT=7.34`).
        for oracle in [&ORACLES[0], &ORACLES[2]] {
            let model = eval_single_zone_do_net(oracle);
            let temp = model.t[0];
            let log_t = model.ln_t[0] / LN10;
            let rho = model.rho[0];
            let log_rho = model.lnd[0] / LN10;
            let smode = model.controls.screening_mode;
            let d_eta_dlnt = model.eos_d_dlnt[0][I_ETA];
            let d_eta_dlnd = model.eos_d_dlnd[0][I_ETA];

            let eval_net_pt = |skip_jac: bool,
                               xa: &[f64],
                               t: f64,
                               lt: f64,
                               d: f64,
                               ld: f64| {
                net_get(
                    &g,
                    skip_jac,
                    xa,
                    t,
                    lt,
                    d,
                    ld,
                    model.abar[0],
                    model.zbar[0],
                    model.z2bar[0],
                    model.ye[0],
                    model.eta[0],
                    d_eta_dlnt,
                    d_eta_dlnd,
                    &rf,
                    1.0,
                    smode,
                )
                .expect("net_get")
            };

            let out0 = eval_net_pt(false, &model.xa, temp, log_t, rho, log_rho);
            assert_eq!(
                ulp_distance(
                    out0.eps_nuc,
                    f64::from_bits(oracle.expected_eps_nuc_bits)
                ),
                0,
                "{}: net_get eps_nuc must match MESA 0 ULP",
                oracle.label
            );

            let mut rate_screened_dt0 = Vec::new();
            let mut rate_screened_drho0 = Vec::new();
            let mut y = vec![0.0_f64; g.num_isos];
            with_thread_local_work(&g, |w| {
                eval_net_with_work(
                    &g,
                    w,
                    &mut vec![[Quad::ZERO; NUM_RVS]; g.num_isos],
                    false,
                    &model.xa,
                    temp,
                    log_t,
                    rho,
                    log_rho,
                    model.abar[0],
                    model.zbar[0],
                    model.z2bar[0],
                    model.ye[0],
                    model.eta[0],
                    d_eta_dlnt,
                    d_eta_dlnd,
                    &rf,
                    1.0,
                    smode,
                    false,
                    false,
                    false,
                )
                .expect("eval_net_with_work");
                rate_screened_dt0 = w.rate_screened_dt.clone();
                rate_screened_drho0 = w.rate_screened_drho.clone();
                y.copy_from_slice(&w.y);
            });

            let dlnt = 1.0e-6_f64;
            let ltp = log_t + dlnt / LN10;
            let ltm = log_t - dlnt / LN10;
            let tp = exp10(ltp);
            let tm = exp10(ltm);
            let out_tp = eval_net_pt(true, &model.xa, tp, ltp, rho, log_rho);
            let out_tm = eval_net_pt(true, &model.xa, tm, ltm, rho, log_rho);
            check_thermo_axis_fd(
                oracle.label,
                "dT",
                &g,
                cat,
                &idx21,
                &y,
                tp - tm,
                &out0,
                &out_tp,
                &out_tm,
                out0.d_eps_nuc_dt,
                &out0.d_dxdt_dt,
                &rate_screened_dt0,
            );

            let dlnr = 1.0e-6_f64;
            let lrp = log_rho + dlnr / LN10;
            let lrm = log_rho - dlnr / LN10;
            let rp = exp10(lrp);
            let rm = exp10(lrm);
            let out_rp = eval_net_pt(true, &model.xa, temp, log_t, rp, lrp);
            let out_rm = eval_net_pt(true, &model.xa, temp, log_t, rm, lrm);
            check_thermo_axis_fd(
                oracle.label,
                "dRho",
                &g,
                cat,
                &idx21,
                &y,
                rp - rm,
                &out0,
                &out_rp,
                &out_rm,
                out0.d_eps_nuc_drho,
                &out0.d_dxdt_drho,
                &rate_screened_drho0,
            );

            let mut d_epsneu_dy = vec![0.0_f64; g.num_isos];
            approx21_d_epsneu_dy(
                &idx21,
                &y,
                &out0.rate_screened,
                &mut d_epsneu_dy,
            );
            let mut xa_p = model.xa.clone();
            let mut xa_m = model.xa.clone();
            for j in [
                idx21.ih1, idx21.ihe3, idx21.ihe4, idx21.ic12, idx21.in14,
                idx21.io16,
            ] {
                let h = 1.0e-6_f64 * model.xa[j].max(1.0e-8);
                xa_p[j] = model.xa[j] + h;
                xa_m[j] = model.xa[j] - h;
                let out_xp =
                    eval_net_pt(true, &xa_p, temp, log_t, rho, log_rho);
                let out_xm =
                    eval_net_pt(true, &xa_m, temp, log_t, rho, log_rho);
                xa_p[j] = model.xa[j];
                xa_m[j] = model.xa[j];

                let j_name = cat
                    .get_by_id(g.chem_id[j])
                    .map_or("?", |n| n.name.as_str());
                let fd_eps_tot_dx = ((out_xp.eps_nuc + out_xp.eps_neu_total)
                    - (out_xm.eps_nuc + out_xm.eps_neu_total))
                    / (2.0 * h);
                let aj = f64::from(
                    u32::try_from(
                        cat.get_by_id(g.chem_id[j]).unwrap().z_plus_n,
                    )
                    .unwrap(),
                );
                // Upstream `approx21_dydt` (`net_approx21.f90:1113`) adds a second
                // `a2 = y_he3 * y_he4 * r_he3ag` to `q_he4`, whereas `approx21_dfdy`
                // (`net_approx21.f90:2097,2131`) only includes one copy in
                // `dfdy(ihe4, ihe3)` and `dfdy(ihe4, ihe4)`.
                let extra_a2_dyj = if j == idx21.ihe3 {
                    y[idx21.ihe4] * out0.rate_screened[idx21.irhe3ag]
                } else if j == idx21.ihe4 {
                    y[idx21.ihe3] * out0.rate_screened[idx21.irhe3ag]
                } else {
                    0.0
                };
                let extra_eps_tot_dx =
                    -AVO * CLIGHT * CLIGHT * g.mion[idx21.ihe4] * extra_a2_dyj
                        / aj;
                let analytic_eps_tot_dx = out0.d_eps_nuc_dx[j]
                    + d_epsneu_dy[j] / aj
                    + extra_eps_tot_dx;
                if fd_eps_tot_dx.abs() > 1.0e-20 {
                    let rel = (analytic_eps_tot_dx - fd_eps_tot_dx).abs()
                        / fd_eps_tot_dx.abs();
                    assert!(
                        rel < 1.0e-6,
                        "{} d_epstotal/dX[{j_name}]: rel={rel:.3e}",
                        oracle.label
                    );
                }

                let a_he4 = f64::from(
                    u32::try_from(
                        cat.get_by_id(g.chem_id[idx21.ihe4]).unwrap().z_plus_n,
                    )
                    .unwrap(),
                );
                for (i, &cid_i) in g.chem_id.iter().enumerate() {
                    let i_name =
                        cat.get_by_id(cid_i).map_or("?", |n| n.name.as_str());
                    let fd_jac = (out_xp.dxdt[i] - out_xm.dxdt[i]) / (2.0 * h);
                    if fd_jac.abs() > 1.0e-24 {
                        let extra_dxdt_dx = if i == idx21.ihe4 {
                            extra_a2_dyj * a_he4 / aj
                        } else {
                            0.0
                        };
                        let analytic =
                            out0.d_dxdt_dx[i * g.num_isos + j] + extra_dxdt_dx;
                        let rel = (analytic - fd_jac).abs() / fd_jac.abs();
                        assert!(
                            rel < 1.0e-6,
                            "{} d(dxdt[{i_name}])/dX[{j_name}]: rel={rel:.3e}",
                            oracle.label
                        );
                    }
                }
            }
        }
    });
}
