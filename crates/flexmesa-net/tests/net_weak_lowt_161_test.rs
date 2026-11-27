// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone analytical consistency tests for dynamically added weaklib-only
//! low-temperature reaction placeholders (`also_in_reaclib = .false.`,
//! `coefficients(1) = -99.0` in `rates/private/reaclib_input.f90:279-294`,
//! `rates/private/rates_initialize.f90:403-459`,
//! `rates/private/eval_weak.f90:100-121`, and
//! `net/private/net_derivs_support.f90:131`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::LN2;
use flexmesa_math::{exp, exp10, log10, with_numerics_mode, NumericsMode};
use flexmesa_net::derivs_support::eval_single_weak_blended;
use flexmesa_net::{
    net_get, synthesize_reaction_info, NetGeneralInfo, RATTAB_TEMP_LO,
};
use flexmesa_rates::{eval_weak_reaction_pair, ScreeningMode};

fn cid_u16(cat: &NuclideCatalog, name: &str) -> u16 {
    cat.get_nuclide_index(name)
        .unwrap_or_else(|| panic!("unknown nuclide {name}")) as u16
}

fn net_slot(g: &NetGeneralInfo, cat: &NuclideCatalog, name: &str) -> usize {
    let cid = cat
        .get_nuclide_index(name)
        .unwrap_or_else(|| panic!("unknown nuclide {name}"));
    let slot = g.net_iso[cid];
    assert!(slot > 0, "nuclide {name} missing from {}", g.net_filename);
    slot - 1
}

#[test]
fn test_synthesize_weaklib_placeholder_sets_negative_lowt_rate() {
    let cat = NuclideCatalog::builtin_ref();
    let ineut = cid_u16(cat, "neut");
    let ih1 = cid_u16(cat, "h1");
    let imn54 = cid_u16(cat, "mn54");
    let ife54 = cid_u16(cat, "fe54");
    let ico56 = cid_u16(cat, "co56");
    let ife56 = cid_u16(cat, "fe56");

    with_numerics_mode(NumericsMode::Strict, || {
        let h1_to_neut = synthesize_reaction_info("r_h1_wk_neut", 9001)
            .expect("r_h1_wk_neut");
        assert_eq!(h1_to_neut.weak_isos, [ih1, ineut]);
        assert_eq!(h1_to_neut.op_code, 2);
        assert_eq!(h1_to_neut.weak_lowt_rate.to_bits(), (-1.0_f64).to_bits());

        let neut_to_h1 = synthesize_reaction_info("r_neut_wk-minus_h1", 9002)
            .expect("r_neut_wk-minus_h1");
        assert_eq!(neut_to_h1.weak_isos, [ineut, ih1]);
        assert_eq!(neut_to_h1.op_code, 2);
        assert_eq!(neut_to_h1.weak_lowt_rate.to_bits(), (-1.0_f64).to_bits());

        let mn54_to_fe54 =
            synthesize_reaction_info("r_mn54_wk-minus_fe54", 9003)
                .expect("r_mn54_wk-minus_fe54");
        assert_eq!(mn54_to_fe54.weak_isos, [imn54, ife54]);
        assert_eq!(mn54_to_fe54.op_code, 2);
        assert_eq!(mn54_to_fe54.weak_lowt_rate.to_bits(), (-1.0_f64).to_bits());

        let co56_to_fe56 = synthesize_reaction_info("r_co56_wk_fe56", 9004)
            .expect("r_co56_wk_fe56");
        assert_eq!(co56_to_fe56.weak_isos, [ico56, ife56]);
        assert_eq!(co56_to_fe56.op_code, 0);
        assert!(co56_to_fe56.weak_lowt_rate > 0.0);
    });
}

#[test]
fn test_eval_single_weak_blended_neut_h1_low_t_matches_weaklib() {
    let cat = NuclideCatalog::builtin_ref();
    let ineut = cid_u16(cat, "neut");
    let ih1 = cid_u16(cat, "h1");

    with_numerics_mode(NumericsMode::Strict, || {
        let temp = RATTAB_TEMP_LO;
        let t9 = temp * 1.0e-9;
        let rho = 6.389_164e-12_f64;
        let ye = 0.798_f64;
        let ye_rho = ye * rho;

        let ev_p2n = eval_single_weak_blended(
            "r_h1_wk_neut",
            [ih1, ineut],
            true,
            t9,
            ye_rho,
            temp,
            rho,
            1.5,
            1.0,
            1.5,
            0.0,
            0.0,
            0.0,
            false,
            false,
        );
        let wl_p2n = eval_weak_reaction_pair("h1", "neut", t9, ye_rho, false)
            .expect("h1->neut weaklib");
        let expected_p2n = exp10(-99.999_f64) + exp10(-99.999_f64);
        assert_eq!(ev_p2n.lambda.to_bits(), wl_p2n.lambda.to_bits());
        assert_eq!(ev_p2n.lambda.to_bits(), expected_p2n.to_bits());
        assert!(ev_p2n.lambda < 1.0e-99);
        assert_ne!(ev_p2n.lambda.to_bits(), exp(-99.0_f64).to_bits());

        let ev_n2p = eval_single_weak_blended(
            "r_neut_wk-minus_h1",
            [ineut, ih1],
            true,
            t9,
            ye_rho,
            temp,
            rho,
            1.5,
            1.0,
            1.5,
            0.0,
            0.0,
            0.0,
            false,
            false,
        );
        let wl_n2p = eval_weak_reaction_pair("neut", "h1", t9, ye_rho, false)
            .expect("neut->h1 weaklib");
        let expected_n2p = exp10(-2.959_f64) + exp10(-99.999_f64);
        assert_eq!(ev_n2p.lambda.to_bits(), wl_n2p.lambda.to_bits());
        assert_eq!(ev_n2p.lambda.to_bits(), expected_n2p.to_bits());
        assert_ne!(ev_n2p.lambda.to_bits(), (LN2 / 632.7373_f64).to_bits());
    });
}

#[test]
fn test_eval_single_weak_blended_weaklib_only_in_weak_info_skips_halflife() {
    let cat = NuclideCatalog::builtin_ref();
    let imn54 = cid_u16(cat, "mn54");
    let ife54 = cid_u16(cat, "fe54");

    with_numerics_mode(NumericsMode::Strict, || {
        let t9 = 1.0e-4_f64;
        let ye_rho = 10.0_f64;
        let ev = eval_single_weak_blended(
            "r_mn54_wk-minus_fe54",
            [imn54, ife54],
            true,
            t9,
            ye_rho,
            t9 * 1.0e9,
            20.0,
            54.0,
            25.0,
            625.0,
            0.0,
            0.0,
            0.0,
            false,
            false,
        );
        let wl = eval_weak_reaction_pair("mn54", "fe54", t9, ye_rho, false)
            .expect("mn54->fe54 weaklib");
        assert_eq!(ev.lambda.to_bits(), wl.lambda.to_bits());
        assert_ne!(ev.lambda.to_bits(), (LN2 / 1.828_143e13_f64).to_bits());
    });
}

#[test]
fn test_mesa_80_net_low_t_dxdt_neut_matches_mesa_split_burn_big_net_z1() {
    let g = NetGeneralInfo::from_name("mesa_80.net").expect("mesa_80.net");
    let cat = NuclideCatalog::builtin_ref();
    let idx_neut = net_slot(&g, cat, "neut");
    let idx_h1 = net_slot(&g, cat, "h1");
    let idx_he4 = net_slot(&g, cat, "he4");

    with_numerics_mode(NumericsMode::Strict, || {
        // Thermodynamic and surface-composition state from MESA 26.4.1
        // `star/split_burn_big_net` model 1, zone 1.
        let temp = 7.852_059e3_f64;
        let log_t = log10(temp);
        let rho = 6.389_164e-12_f64;
        let log_rho = log10(rho);

        let x_neut = 1.002_116_094_738_558_9e-99_f64;
        let x_h1 = 5.966_683_176_519_839_1e-1_f64;
        let mut x = vec![0.0_f64; g.num_isos];
        x[idx_neut] = x_neut;
        x[idx_h1] = x_h1;
        x[idx_he4] = 1.0 - x_h1 - x_neut;

        let comp = get_composition_info(cat, &g.chem_id, &x, false)
            .expect("composition_info");
        let rate_factors = vec![1.0_f64; g.num_reactions];

        let res = net_get(
            &g,
            false,
            &x,
            temp,
            log_t,
            rho,
            log_rho,
            comp.abar,
            comp.zbar,
            comp.z2bar,
            comp.ye,
            0.0,
            0.0,
            0.0,
            &rate_factors,
            1.0,
            ScreeningMode::Chugunov,
        )
        .expect("net_get mesa_80");

        let p2n_idx = g
            .reaction_index_by_name("r_h1_wk_neut")
            .expect("r_h1_wk_neut in mesa_80.net")
            - 1;
        let n2p_idx = g
            .reaction_index_by_name("r_neut_wk-minus_h1")
            .expect("r_neut_wk-minus_h1 in mesa_80.net")
            - 1;
        assert!(res.rate_screened[p2n_idx] < 1.0e-99);
        assert_eq!(
            res.rate_screened[n2p_idx].to_bits(),
            (exp10(-2.959_f64) + exp10(-99.999_f64)).to_bits()
        );

        let expected = -x_neut * (exp10(-2.959_f64) + exp10(-99.999_f64));
        let got = res.dxdt[idx_neut];
        assert!(got < 0.0);
        assert!(
            ((got - expected) / expected).abs() < 1.0e-14,
            "got={got}, expected={expected}"
        );
        assert!(got < -1.101_331e-102 && got > -1.101_332e-102);
    });
}
