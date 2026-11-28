// SPDX-License-Identifier: GPL-3.0-or-later
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

mod common;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{CRAD, LN10};
use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, eos_dt_get_t, eos_pt_get, eos_pt_get_pgas,
    eos_pt_get_t, eval_cms_eos, eval_free_eos, eval_opal_scvh_eos, CmsConfig,
    CmsTableSet, EosConfig, EosDtTableSet, EosEvalOutput, FreeEosConfig,
    HelmTable, OpalScvhConfig, EOS_BASIC_RESULT_NAMES, I_CHI_RHO, I_CHI_T,
    I_CP, I_CV, I_DE_DRHO, I_DS_DRHO, I_DS_DT, I_ETA, I_FRAC_CMS,
    I_FRAC_FREE_EOS, I_FRAC_HELM, I_FRAC_IDEAL, I_FRAC_OPAL_SCVH, I_FRAC_PC,
    I_FRAC_SKYE, I_GAMMA1, I_GAMMA3, I_GRAD_AD, I_LATENT_DDLNRHO,
    I_LATENT_DDLNT, I_LN_E, I_LN_FREE_E, I_LN_PGAS, I_LN_S, I_MU, I_PHASE,
};
use flexmesa_math::{exp10, with_numerics_mode, NumericsMode};

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
    let abs_err = (actual - expected).abs();
    if expected.abs() <= 1.0e-13 {
        assert!(
            abs_err <= 1.0e-14,
            "{label} (near-zero): actual={actual:.16e}, expected={expected:.16e}, abs_err={abs_err:.3e} > 1.0e-14"
        );
    } else {
        let rel = abs_err / expected.abs();
        assert!(
            rel <= rtol,
            "{label}: actual={actual:.16e}, expected={expected:.16e}, rel_err={rel:.3e} > {rtol:.3e}"
        );
    }
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

    let ht = HelmTable::builtin().expect("builtin HELM table");
    let cfg = EosConfig::default();

    for (c_idx, &(x, zinit)) in compositions.iter().enumerate() {
        let (chem_id, xa) = init_composition(x, zinit, 0.0, 0.0);
        for p_idx in 0..4 {
            let g = &pts[c_idx * 4 + p_idx];
            let out = eos_dt_get(ht, g.rho, g.t, &chem_id, &xa, &cfg)
                .expect("eos_dt_get must succeed");
            let prad = CRAD * flexmesa_math::pow4(g.t) / 3.0;
            let pgas = flexmesa_math::exp(out.res[I_LN_PGAS]);
            let p = pgas + prad;
            let e = flexmesa_math::exp(out.res[I_LN_E]);
            let s = flexmesa_math::exp(out.res[I_LN_S]);

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
    let ht = HelmTable::builtin().expect("builtin HELM table");
    let cfg = EosConfig::default();
    let (chem_id, xa) = init_composition(0.70, 0.02, 0.0, 0.0);

    let log_t = 4.8_f64;
    let t = exp10(log_t);
    let rho = 1.0e-7_f64;
    let log_rho = flexmesa_math::log10(rho);
    let ln_s = flexmesa_math::log(2.968_064_512e9_f64);
    let tol = 1.0e-5;
    let othertol = 1.0e-12;
    let max_iter = 100;

    let log_rho_guess = flexmesa_math::log10(rho * 0.5);
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

    let log_t_guess = flexmesa_math::log10(t * 2.0);
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
    let prad = CRAD * flexmesa_math::pow4(t_from_log) / 3.0;
    let pgas = flexmesa_math::exp(inv_t_s.eos.res[I_LN_PGAS]);
    let log_p = flexmesa_math::log10(prad + pgas);
    let log_t_guess_p = flexmesa_math::log10(t_from_log * 2.0);

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

    // Expected values from MESA 26.4.1 `eos/test/test_output:413,415` (#194).
    assert_rel_close(
        "found logT (logP)",
        inv_t_p.log10_result,
        4.800_961_315_559_555_4,
        1.0e-12,
    );
    let t_res = exp10(inv_t_p.log10_result);
    let got_log_p = flexmesa_math::log10(
        CRAD * flexmesa_math::pow4(t_res) / 3.0
            + flexmesa_math::exp(inv_t_p.eos.res[I_LN_PGAS]),
    );
    assert_rel_close("got logP", got_log_p, 5.945_429_253_933_702_4, 1.0e-12);
}

#[test]
fn test_wave3b1_golden_eospt_for_ck_matches_fortran_mesa() {
    let ht = HelmTable::builtin().expect("builtin HELM table");
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
    let prad = CRAD * flexmesa_math::pow4(t) / 3.0;
    let p = pgas + prad;

    let expected_slots = [
        ("eosPT P", p, 1.584_893_444_652_220_0e4),
        (
            "eosPT E",
            flexmesa_math::exp(pt.eos.res[I_LN_E]),
            8.182_690_709_432_792_7e10,
        ),
        (
            "eosPT S",
            flexmesa_math::exp(pt.eos.res[I_LN_S]),
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

/// MESA `26.4.1` outputs of `eosPT_eval::get_Pgas` at the operating point
/// below, written by
/// `benchmarks/fortran_mesa/eos_pgas_ion_oracles/gen_eos_pgas_ion_oracles.f90`.
const EOS_PGAS_ION_ORACLES: &str = include_str!(
    "../../../benchmarks/fortran_mesa/eos_pgas_ion_oracles/mesa_26_4_1_eos_pgas_ion_oracles.tsv"
);

/// Return the MESA bit pattern of `(field, part)` in `case` (`role` is `in` or `out`).
fn mesa_eos_bits(case: &str, role: &str, field: &str, part: &str) -> u64 {
    for line in EOS_PGAS_ION_ORACLES.lines() {
        if line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 5
            && cols[0] == case
            && cols[1] == role
            && cols[2] == field
            && cols[3] == part
        {
            let hex = cols[4].trim_start_matches("0x");
            return u64::from_str_radix(hex, 16).expect("hex column");
        }
    }
    panic!("no MESA eos_pgas_ion oracle row for {case} {role} {field} {part}");
}

/// Return the MESA bit pattern of `field` in `case` (`role` is `in` or `out`).
fn mesa_pgas_bits(case: &str, role: &str, field: &str) -> u64 {
    mesa_eos_bits(case, role, field, "val")
}

fn check_get1_bits(
    case_id: &str,
    rho: f64,
    temp: f64,
    out: &EosEvalOutput,
    diffs: &mut Vec<String>,
) {
    for (fld, val) in [
        ("rho", rho),
        ("logRho", flexmesa_math::log10(rho)),
        ("T", temp),
        ("logT", flexmesa_math::log10(temp)),
    ] {
        let exp = mesa_eos_bits(case_id, "in", fld, "val");
        if val.to_bits() != exp {
            diffs.push(format!(
                "{case_id}.in.{fld}: MESA 0x{exp:016X} vs Flex 0x{:016X} ULP={}",
                val.to_bits(),
                val.to_bits().abs_diff(exp),
            ));
        }
    }
    for idx in 0..16 {
        let fld = format!("res_{idx}");
        let name = EOS_BASIC_RESULT_NAMES[idx];
        for (part, got) in [
            ("val", out.res[idx]),
            ("d_dlnd", out.d_dlnd[idx]),
            ("d_dlnT", out.d_dln_t[idx]),
        ] {
            let exp = mesa_eos_bits(case_id, "out", &fld, part);
            let got_bits = got.to_bits();
            if got_bits != exp {
                diffs.push(format!(
                    "{case_id}.out[{idx}:{name}].{part}: MESA 0x{exp:016X} ({:.17e}) vs Flex 0x{got_bits:016X} ({got:.17e}) ULP={}",
                    f64::from_bits(exp),
                    got_bits.abs_diff(exp),
                ));
            }
        }
    }
}

fn assert_pgas_bits(label: &str, actual: f64, expected_bits: u64) {
    let actual_bits = actual.to_bits();
    assert_eq!(
        actual_bits,
        expected_bits,
        "{label}: MESA {expected:.17e} ({expected_bits:016X}) vs Flex {actual:.17e} ({actual_bits:016X})",
        expected = f64::from_bits(expected_bits),
    );
}

#[test]
fn test_wave3b1_components_cms_and_eospt_inversions() {
    let ht = HelmTable::builtin().expect("builtin HELM table");
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
    common::assert_eos_against_mesa(
        "opal_scvh_solar_1em2_1e5",
        &out_opal,
        1.0e-14,
        1.0e-14,
        &[],
        false,
    );

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
    common::assert_eos_against_mesa(
        "free_eos_solar_1em2_1e6",
        &out_free,
        1.0e-14,
        1.0e-14,
        &[],
        false,
    );

    with_numerics_mode(NumericsMode::Strict, || {
        let opal_strict = eval_opal_scvh_eos(
            dt_tables,
            1.0e-2,
            1.0e5,
            &chem_id,
            &xa,
            &OpalScvhConfig::default(),
        )
        .expect("OPAL/SCVH Strict evaluation");
        let free_strict = eval_free_eos(
            dt_tables,
            1.0e-2,
            1.0e6,
            &chem_id,
            &xa,
            &FreeEosConfig::default(),
        )
        .expect("FreeEOS Strict evaluation");
        let mut diffs = Vec::new();
        check_get1_bits(
            "get1_opal_scvh_solar_1em2_1e5",
            1.0e-2,
            1.0e5,
            &opal_strict,
            &mut diffs,
        );
        check_get1_bits(
            "get1_free_eos_solar_1em2_1e6",
            1.0e-2,
            1.0e6,
            &free_strict,
            &mut diffs,
        );
        assert!(
            diffs.is_empty(),
            "get1_for_eosdt departures ({}):\n{}",
            diffs.len(),
            diffs.join("\n")
        );
    });

    // Verify both gas-only (`norad`, `include_radiation = false`) and default
    // radiation-enabled (`include_radiation = true`, `eos.defaults:163`) `CMS`
    // against compiled MESA `26.4.1` across all 26x3 slots.
    let cms_norad_cfg = CmsConfig {
        include_radiation: false,
        ..CmsConfig::default()
    };
    let out_cms_norad =
        eval_cms_eos(cms_tables, &cms_norad_cfg, 1.0e-2, 1.0e5, &chem_id, &xa)
            .expect("CMS standalone evaluation (include_radiation = false)");
    assert_eq!(out_cms_norad.res[I_FRAC_CMS], 1.0);
    common::assert_eos_against_mesa(
        "cms_solar_1em2_1e5_norad",
        &out_cms_norad,
        1.0e-14,
        2.0e-12,
        &[],
        false,
    );

    let out_cms_rad = eval_cms_eos(
        cms_tables,
        &CmsConfig::default(),
        1.0e-2,
        1.0e5,
        &chem_id,
        &xa,
    )
    .expect("CMS standalone evaluation (include_radiation = true)");
    assert_eq!(out_cms_rad.res[I_FRAC_CMS], 1.0);
    common::assert_eos_against_mesa(
        "cms_solar_1em2_1e5",
        &out_cms_rad,
        1.0e-14,
        2.0e-12,
        &[],
        false,
    );

    let cfg = EosConfig::default();
    let pt_ref = eos_pt_get(ht, &chem_id, &xa, 15.0, 7.2, &cfg)
        .expect("reference eosPT");
    let target_lne = pt_ref.eos.res[I_LN_E];

    let exp_pt_t = common::load_scalar_row::<6>("INV", "solar_pt_get_t_15_7p2");
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
    assert_rel_close(
        "eosPT_get_T logT",
        inv_t.log10_result,
        exp_pt_t[0],
        1.0e-12,
    );
    assert_rel_close(
        "eosPT_get_T log10_rho",
        inv_t.pt.log10_rho,
        exp_pt_t[1],
        1.0e-12,
    );
    assert_rel_close(
        "eosPT_get_T lnE",
        inv_t.pt.eos.res[I_LN_E],
        exp_pt_t[2],
        1.0e-12,
    );
    assert_rel_close(
        "eosPT_get_T dlnRho_dlnPgas",
        inv_t.pt.dln_rho_dln_pgas_const_t,
        exp_pt_t[4],
        1.0e-12,
    );
    assert_rel_close(
        "eosPT_get_T dlnRho_dlnT",
        inv_t.pt.dln_rho_dln_t_const_pgas,
        exp_pt_t[5],
        1.0e-12,
    );

    for (idx, &x) in xa.iter().enumerate() {
        assert_pgas_bits(
            &format!("eosPT_get_Pgas.in.xa_{idx}"),
            x,
            mesa_pgas_bits(
                "eospt_get_pgas_solar_15_7p2",
                "in",
                &format!("xa_{idx}"),
            ),
        );
    }
    assert_pgas_bits(
        "eosPT_get_Pgas.in.logT",
        7.2,
        mesa_pgas_bits("eospt_get_pgas_solar_15_7p2", "in", "logT"),
    );
    assert_pgas_bits(
        "eosPT_get_Pgas.in.target_lne",
        target_lne,
        mesa_pgas_bits("eospt_get_pgas_solar_15_7p2", "in", "target_lne"),
    );
    assert_pgas_bits(
        "eosPT_get_Pgas.in.logPgas_guess",
        14.8,
        mesa_pgas_bits("eospt_get_pgas_solar_15_7p2", "in", "logPgas_guess"),
    );
    assert_pgas_bits(
        "eosPT_get_Pgas.in.logPgas_tol",
        1.0e-6,
        mesa_pgas_bits("eospt_get_pgas_solar_15_7p2", "in", "logPgas_tol"),
    );
    assert_pgas_bits(
        "eosPT_get_Pgas.in.other_tol",
        LN10 * 1.0e-6,
        mesa_pgas_bits("eospt_get_pgas_solar_15_7p2", "in", "other_tol"),
    );

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
    assert_pgas_bits(
        "eosPT_get_Pgas.out.logPgas_result",
        inv_pgas.log10_result,
        mesa_pgas_bits("eospt_get_pgas_solar_15_7p2", "out", "logPgas_result"),
    );
    assert_pgas_bits(
        "eosPT_get_Pgas.out.Rho",
        inv_pgas.pt.rho,
        mesa_pgas_bits("eospt_get_pgas_solar_15_7p2", "out", "Rho"),
    );
    assert_pgas_bits(
        "eosPT_get_Pgas.out.logRho",
        inv_pgas.pt.log10_rho,
        mesa_pgas_bits("eospt_get_pgas_solar_15_7p2", "out", "logRho"),
    );
    assert_pgas_bits(
        "eosPT_get_Pgas.out.dlnRho_dlnPgas_c_T",
        inv_pgas.pt.dln_rho_dln_pgas_const_t,
        mesa_pgas_bits(
            "eospt_get_pgas_solar_15_7p2",
            "out",
            "dlnRho_dlnPgas_c_T",
        ),
    );
    assert_pgas_bits(
        "eosPT_get_Pgas.out.dlnRho_dlnT_c_Pgas",
        inv_pgas.pt.dln_rho_dln_t_const_pgas,
        mesa_pgas_bits(
            "eospt_get_pgas_solar_15_7p2",
            "out",
            "dlnRho_dlnT_c_Pgas",
        ),
    );
    assert_pgas_bits(
        "eosPT_get_Pgas.out.res_lnE",
        inv_pgas.pt.eos.res[I_LN_E],
        mesa_pgas_bits("eospt_get_pgas_solar_15_7p2", "out", "res_lnE"),
    );
    assert_pgas_bits(
        "eosPT_get_Pgas.out.res_lnPgas",
        inv_pgas.pt.eos.res[I_LN_PGAS],
        mesa_pgas_bits("eospt_get_pgas_solar_15_7p2", "out", "res_lnPgas"),
    );
    assert_pgas_bits(
        "eosPT_get_Pgas.out.eos_calls",
        inv_pgas.eos_calls as f64,
        mesa_pgas_bits("eospt_get_pgas_solar_15_7p2", "out", "eos_calls"),
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

#[test]
fn test_cms_default_config_matches_mesa_26_4_1_default_radiation() {
    let cms_tables = CmsTableSet::builtin();
    let (chem_id, xa) = init_composition(0.70, 0.02, 0.0, 0.0);
    let out_cms_def = eval_cms_eos(
        cms_tables,
        &CmsConfig::default(),
        1.0e-2,
        1.0e5,
        &chem_id,
        &xa,
    )
    .expect("CMS standalone evaluation");
    common::assert_eos_against_mesa(
        "cms_solar_1em2_1e5",
        &out_cms_def,
        1.0e-12,
        1.0e-12,
        &[],
        false,
    );
}
