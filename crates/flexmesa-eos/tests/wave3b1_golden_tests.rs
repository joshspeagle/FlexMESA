// SPDX-License-Identifier: LGPL-3.0-or-later
//! Golden verification tests for Wave 3b.1 4D Tabulated EOS (`OPAL/SCVH`,
//! `FreeEOS`, `CMS`), the full 7-level `eosDT_get` cascade, and `eosPT_get`
//! against upstream Fortran MESA `eos/test/test_output`.

#![allow(
    clippy::approx_constant,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{CRAD, LN10};
use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, eos_dt_get_t, eos_pt_get, eos_pt_get_pgas,
    eos_pt_get_t, eval_cms_eos, eval_free_eos, eval_opal_scvh_eos, CmsConfig,
    CmsTableSet, EosConfig, EosDtTableSet, FreeEosConfig, HelmTable,
    OpalScvhConfig, I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_DE_DRHO, I_DS_DRHO,
    I_DS_DT, I_ETA, I_FRAC_CMS, I_FRAC_FREE_EOS, I_FRAC_HELM, I_FRAC_IDEAL,
    I_FRAC_OPAL_SCVH, I_FRAC_PC, I_FRAC_SKYE, I_GAMMA1, I_GAMMA3, I_GRAD_AD,
    I_LATENT_DDLNRHO, I_LATENT_DDLNT, I_LN_E, I_LN_FREE_E, I_LN_PGAS, I_LN_S,
    I_MU, I_PHASE,
};
use flexmesa_math::exp10;

/// Builds the 7-species composition `(chem_id, xa)` matching `Init_Composition` in `eos_support.f90`.
fn init_composition(
    x_in: f64,
    zinit_in: f64,
    xc_in: f64,
    xo_in: f64,
) -> (Vec<usize>, Vec<f64>) {
    let cat = NuclideCatalog::builtin_ref();
    let names = ["h1", "he4", "c12", "n14", "o16", "ne20", "mg24"];
    let chem_id: Vec<usize> = names
        .iter()
        .map(|n| cat.get_by_name(n).expect("valid isotope").chem_id)
        .collect();

    const ZFRAC_C: f64 = 0.173312;
    const ZFRAC_N: f64 = 0.053177;
    const ZFRAC_O: f64 = 0.482398;
    const ZFRAC_NE: f64 = 0.098675;

    let mut xc = xc_in;
    let mut xo = xo_in;
    let mut y = 1.0 - (x_in + zinit_in + xc + xo);
    if y < 0.0 {
        let frac = (1.0 - x_in - zinit_in) / (xc + xo);
        xc *= frac;
        xo *= frac;
        y = (1.0 - (x_in + zinit_in + xc + xo)).max(0.0);
    }

    let mut xa = vec![0.0; 7];
    xa[0] = x_in;
    xa[1] = y;
    xa[2] = zinit_in * ZFRAC_C + xc;
    xa[3] = zinit_in * ZFRAC_N;
    xa[4] = zinit_in * ZFRAC_O + xo;
    xa[5] = zinit_in * ZFRAC_NE;
    let sum6: f64 = xa[..6].iter().sum();
    xa[6] = 1.0 - sum6;

    (chem_id, xa)
}

fn assert_rel_close(label: &str, actual: f64, expected: f64, rtol: f64) {
    let denom = expected.abs().max(actual.abs()).max(1.0);
    let rel = (actual - expected).abs() / denom;
    assert!(
        rel <= rtol,
        "{label}: actual={actual:.16e}, expected={expected:.16e}, rel_err={rel:.3e} > {rtol:.3e}"
    );
}

#[derive(Debug)]
struct DoitPointGolden {
    t: f64,
    rho: f64,
    p: f64,
    e: f64,
    s: f64,
    mu: f64,
    lnfree_e: f64,
    eta: f64,
    grad_ad: f64,
    chi_rho: f64,
    chi_t: f64,
    gamma1: f64,
    gamma3: f64,
}

fn parse_golden_doit_points(text: &str) -> Vec<DoitPointGolden> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < lines.len() {
        if lines[idx].trim() == "eosDT_get" {
            let parse_val = |line: &str| -> f64 {
                let s = line
                    .split_whitespace()
                    .last()
                    .expect("numeric column")
                    .replace('D', "E");
                s.parse::<f64>().expect("valid float")
            };
            let rho = exp10(parse_val(lines[idx - 4]));
            let lt = parse_val(lines[idx - 3]);
            let t = if lt > 8.0 { 2.0e8 } else { exp10(lt) };
            out.push(DoitPointGolden {
                t,
                rho,
                p: parse_val(lines[idx + 1]),
                e: parse_val(lines[idx + 2]),
                s: parse_val(lines[idx + 3]),
                mu: parse_val(lines[idx + 4]),
                lnfree_e: parse_val(lines[idx + 5]),
                eta: parse_val(lines[idx + 6]),
                grad_ad: parse_val(lines[idx + 7]),
                chi_rho: parse_val(lines[idx + 8]),
                chi_t: parse_val(lines[idx + 9]),
                gamma1: parse_val(lines[idx + 10]),
                gamma3: parse_val(lines[idx + 11]),
            });
            idx += 13;
        } else {
            idx += 1;
        }
    }
    out
}

#[test]
fn test_wave3b1_golden_doit_16_points_match_fortran_mesa() {
    let golden_text =
        include_str!("../../../tests/fixtures/golden/eos/test_output.txt");
    let pts = parse_golden_doit_points(golden_text);
    assert_eq!(pts.len(), 16);

    let compositions = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.75 * (1.0 - 0.03), 0.03),
        (0.70, 0.02),
    ];

    let ht = HelmTable::builtin();
    let cfg = EosConfig::default();

    for (c_idx, &(x, zinit)) in compositions.iter().enumerate() {
        let (chem_id, xa) = init_composition(x, zinit, 0.0, 0.0);
        for p_idx in 0..4 {
            let g = &pts[c_idx * 4 + p_idx];
            let out = eos_dt_get(ht, g.rho, g.t, &chem_id, &xa, &cfg)
                .expect("eos_dt_get must succeed");
            let prad = CRAD * g.t.powi(4) / 3.0;
            let pgas = out.res[I_LN_PGAS].exp();
            let p = pgas + prad;
            let e = out.res[I_LN_E].exp();
            let s = out.res[I_LN_S].exp();

            let rtol = if p_idx < 3 { 1.0e-12 } else { 5.0e-11 };
            let tag = format!("comp={c_idx} pt={p_idx}");
            for (name, actual, expected) in [
                ("P", p, g.p),
                ("E", e, g.e),
                ("S", s, g.s),
                ("mu", out.res[I_MU], g.mu),
                ("lnfree_e", out.res[I_LN_FREE_E], g.lnfree_e),
                ("eta", out.res[I_ETA], g.eta),
                ("grad_ad", out.res[I_GRAD_AD], g.grad_ad),
                ("chiRho", out.res[I_CHI_RHO], g.chi_rho),
                ("chiT", out.res[I_CHI_T], g.chi_t),
                ("gamma1", out.res[I_GAMMA1], g.gamma1),
                ("gamma3", out.res[I_GAMMA3], g.gamma3),
            ] {
                assert_rel_close(
                    &format!("{tag} {name}"),
                    actual,
                    expected,
                    rtol,
                );
            }
        }
    }
}

#[test]
fn test_wave3b1_golden_get_rho_t_inversions_match_fortran_mesa() {
    let ht = HelmTable::builtin();
    let cfg = EosConfig::default();
    let (chem_id, xa) = init_composition(0.70, 0.02, 0.0, 0.0);

    let log_t = 4.8_f64;
    let t = exp10(log_t);
    let rho = 1.0e-7_f64;
    let log_rho = rho.log10();
    let ln_s = (2.968_064_512e9_f64).ln();
    let tol = 1.0e-5;
    let othertol = 1.0e-12;
    let max_iter = 100;

    let log_rho_guess = (rho * 0.5).log10();
    let inv_rho = eos_dt_get_rho(
        ht,
        &chem_id,
        &xa,
        log_t,
        I_LN_S as isize,
        ln_s,
        log_rho_guess,
        tol,
        othertol,
        max_iter,
        &cfg,
    )
    .expect("eos_dt_get_rho must converge");

    assert_rel_close(
        "found logRho",
        inv_rho.log10_result,
        -7.002_142_915_778_787_1,
        1.0e-12,
    );
    assert_rel_close(
        "got logS (rho)",
        inv_rho.eos.res[I_LN_S] / LN10,
        9.472_473_336_269_533_9,
        1.0e-12,
    );

    let log_t_guess = (t * 2.0).log10();
    let inv_t_s = eos_dt_get_t(
        ht,
        &chem_id,
        &xa,
        log_rho,
        I_LN_S as isize,
        ln_s,
        log_t_guess,
        tol,
        othertol,
        max_iter,
        &cfg,
    )
    .expect("eos_dt_get_t given lnS must converge");

    assert_rel_close(
        "found logT (lnS)",
        inv_t_s.log10_result,
        4.801_146_904_621_497_0,
        1.0e-12,
    );
    assert_rel_close(
        "got logS (T)",
        inv_t_s.eos.res[I_LN_S] / LN10,
        9.472_473_336_251_040_3,
        1.0e-12,
    );

    let t_from_log = exp10(log_t);
    let prad = CRAD * t_from_log.powi(4) / 3.0;
    let pgas = inv_t_s.eos.res[I_LN_PGAS].exp();
    let log_p = (prad + pgas).log10();
    let log_t_guess_p = (t_from_log * 2.0).log10();

    let inv_t_p = eos_dt_get_t(
        ht,
        &chem_id,
        &xa,
        log_rho,
        -2,
        log_p,
        log_t_guess_p,
        tol,
        othertol,
        max_iter,
        &cfg,
    )
    .expect("eos_dt_get_t given logPtot must converge");

    assert_rel_close(
        "found logT (logP)",
        inv_t_p.log10_result,
        4.800_965_466_497_445_7,
        1.0e-12,
    );
    let t_res = exp10(inv_t_p.log10_result);
    let got_log_p =
        (CRAD * t_res.powi(4) / 3.0 + inv_t_p.eos.res[I_LN_PGAS].exp()).log10();
    assert_rel_close("got logP", got_log_p, 5.945_434_039_207_868_6, 1.0e-12);
}

#[test]
fn test_wave3b1_golden_eospt_for_ck_matches_fortran_mesa() {
    let ht = HelmTable::builtin();
    let cfg = EosConfig::default();
    let (chem_id, xa) = init_composition(0.70, 0.02, 0.0, 0.0);

    let pt = eos_pt_get(ht, &chem_id, &xa, 4.2, 3.0, &cfg)
        .expect("eos_pt_get must converge");

    assert_rel_close("eosPT rho", pt.rho, 4.513_685_055_911_351_6e-7, 1.0e-12);
    assert_rel_close(
        "eosPT logRho",
        pt.log10_rho,
        -6.345_468_747_263_481_6,
        1.0e-12,
    );
    assert_rel_close(
        "dlnRho_dlnPgas_c_T",
        pt.dln_rho_dln_pgas_const_t,
        9.999_898_753_296_615_5e-1,
        1.0e-12,
    );
    assert_rel_close(
        "dlnRho_dlnT_c_Pgas",
        pt.dln_rho_dln_t_const_pgas,
        -9.997_576_989_833_861_8e-1,
        1.0e-12,
    );

    let t = exp10(3.0);
    let pgas = exp10(4.2);
    let prad = CRAD * t.powi(4) / 3.0;
    let p = pgas + prad;

    let expected_slots = [
        ("eosPT P", p, 1.584_893_444_652_220_0e4),
        (
            "eosPT E",
            pt.eos.res[I_LN_E].exp(),
            8.182_690_709_432_792_7e10,
        ),
        (
            "eosPT S",
            pt.eos.res[I_LN_S].exp(),
            8.970_017_467_992_692_0e8,
        ),
        ("eosPT mu", pt.eos.res[I_MU], 1.900_000_000_000_000_1),
        (
            "eosPT lnfree_e",
            pt.eos.res[I_LN_FREE_E],
            -9.210_340_371_976_181_8,
        ),
        ("eosPT eta", pt.eos.res[I_ETA], -20.0),
        (
            "eosPT grad_ad",
            pt.eos.res[I_GRAD_AD],
            2.930_972_160_695_617_4e-1,
        ),
        (
            "eosPT chiRho",
            pt.eos.res[I_CHI_RHO],
            1.000_009_965_649_430_8,
        ),
        (
            "eosPT chiT",
            pt.eos.res[I_CHI_T],
            9.997_682_987_053_566_6e-1,
        ),
        ("eosPT Cp", pt.eos.res[I_CP], 1.200_817_512_302_268_6e8),
        ("eosPT Cv", pt.eos.res[I_CV], 8.499_353_278_077_815_5e7),
        (
            "eosPT dE_dRho",
            pt.eos.res[I_DE_DRHO],
            1.798_762_881_151_189_1e13,
        ),
        (
            "eosPT dS_dT",
            pt.eos.res[I_DS_DT],
            8.499_353_278_077_815_8e4,
        ),
        (
            "eosPT dS_dRho",
            pt.eos.res[I_DS_DRHO],
            -7.775_653_019_616_923_4e13,
        ),
        (
            "eosPT gamma1",
            pt.eos.res[I_GAMMA1],
            1.407_841_695_996_403_5,
        ),
        (
            "eosPT gamma3",
            pt.eos.res[I_GAMMA3],
            1.412_543_895_449_647_2,
        ),
    ];
    for &(label, actual, expected) in &expected_slots {
        assert_rel_close(label, actual, expected, 1.0e-12);
    }
    for &idx in &[
        I_PHASE,
        I_LATENT_DDLNT,
        I_LATENT_DDLNRHO,
        I_FRAC_HELM,
        I_FRAC_FREE_EOS,
        I_FRAC_PC,
        I_FRAC_SKYE,
        I_FRAC_CMS,
        I_FRAC_IDEAL,
    ] {
        assert_eq!(pt.eos.res[idx], 0.0);
    }
    assert_eq!(pt.eos.res[I_FRAC_OPAL_SCVH], 1.0);
}

#[test]
fn test_wave3b1_components_cms_and_eospt_inversions() {
    let ht = HelmTable::builtin();
    let dt_tables = EosDtTableSet::builtin();
    let cms_tables = CmsTableSet::builtin();
    let (chem_id, xa) = init_composition(0.70, 0.02, 0.0, 0.0);

    let out_opal = eval_opal_scvh_eos(
        dt_tables,
        1.0e-2,
        1.0e5,
        &chem_id,
        &xa,
        &OpalScvhConfig::default(),
    )
    .expect("OPAL/SCVH standalone evaluation");
    assert_eq!(out_opal.res[I_FRAC_OPAL_SCVH], 1.0);

    let out_free = eval_free_eos(
        dt_tables,
        1.0e-2,
        1.0e6,
        &chem_id,
        &xa,
        &FreeEosConfig::default(),
    )
    .expect("FreeEOS standalone evaluation");
    assert_eq!(out_free.res[I_FRAC_FREE_EOS], 1.0);

    let out_cms = eval_cms_eos(
        cms_tables,
        &CmsConfig::default(),
        1.0e-2,
        1.0e5,
        &chem_id,
        &xa,
    )
    .expect("CMS standalone evaluation");
    assert_eq!(out_cms.res[I_FRAC_CMS], 1.0);

    let cfg = EosConfig::default();
    let pt_ref = eos_pt_get(ht, &chem_id, &xa, 15.0, 7.2, &cfg)
        .expect("reference eosPT");
    let target_lne = pt_ref.eos.res[I_LN_E];

    let inv_t = eos_pt_get_t(
        ht,
        &chem_id,
        &xa,
        15.0,
        I_LN_E,
        target_lne,
        7.0,
        1.0e-6,
        LN10 * 1.0e-6,
        100,
        &cfg,
    )
    .expect("eos_pt_get_t must converge");
    assert_rel_close("eosPT_get_T logT", inv_t.log10_result, 7.2, 1.0e-5);

    let inv_pgas = eos_pt_get_pgas(
        ht,
        &chem_id,
        &xa,
        7.2,
        I_LN_E,
        target_lne,
        14.8,
        1.0e-6,
        LN10 * 1.0e-6,
        100,
        &cfg,
    )
    .expect("eos_pt_get_pgas must converge");
    assert_rel_close(
        "eosPT_get_Pgas logPgas",
        inv_pgas.log10_result,
        15.0,
        1.0e-5,
    );
    assert_eq!(pt_ref.eos.d_dxa.len(), 7);
    assert!(eos_dt_get_t(
        ht, &chem_id, &xa, -2.0, -99, 0.0, 5.0, 1.0e-6, 1.0e-6, 10, &cfg
    )
    .is_err());
    assert!(eos_pt_get_t(
        ht, &chem_id, &xa, 15.0, 99, 0.0, 7.0, 1.0e-6, 1.0e-6, 10, &cfg
    )
    .is_err());
}
