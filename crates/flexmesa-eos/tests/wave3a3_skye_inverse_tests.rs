// SPDX-License-Identifier: GPL-3.0-or-later
//! Upstream Fortran MESA `26.4.1` oracle tests for Wave 3a.3 (`Skye` equation of
//! state, `get_skye_alfa`, `get_skye_alfa_simple`, `eosDT_get` `Skye`/`HELM`
//! blending, and `eosPT_get` / `eosDT_get_T` / `eosDT_get_Rho` /
//! `eos_gamma_DE_get` inverse solvers; `#161`, `T2-15`).
//!
//! Expected `26 x 3` (`EOS`), `ALFA`, and `INV` reference rows are loaded from
//! `benchmarks/fortran_mesa/eos_oracles/mesa_26_4_1_eos_oracles.tsv`.

#![allow(clippy::cast_possible_wrap, clippy::similar_names)]

mod common;

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::LN10;
use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, eos_dt_get_t, eos_gamma_de_get, eos_pt_get,
    eval_skye_eos, get_skye_alfa, get_skye_alfa_simple, EosConfig, HelmTable,
    SkyeConfig, SkyeSolidMixingRule, I_DS_DRHO, I_DS_DT, I_ETA, I_LN_E,
    I_LN_FREE_E, I_LN_PGAS, I_LN_S, I_MU,
};
use flexmesa_math::{with_numerics_mode, NumericsMode};

const EOS_PGAS_ION_ORACLES: &str = include_str!(
    "../../../benchmarks/fortran_mesa/eos_pgas_ion_oracles/mesa_26_4_1_eos_pgas_ion_oracles.tsv"
);

fn mesa_skye_bits(case: &str, role: &str, field: &str) -> u64 {
    for line in EOS_PGAS_ION_ORACLES.lines() {
        if line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 5
            && cols[0] == case
            && cols[1] == role
            && cols[2] == field
            && cols[3] == "val"
        {
            let hex = cols[4].trim_start_matches("0x");
            return u64::from_str_radix(hex, 16).expect("hex column");
        }
    }
    panic!("no MESA Get_Skye_alfa_simple oracle row for {case} {role} {field}");
}

fn check_skye_alfa_simple_bits(case_id: &str, cfg: &SkyeConfig) {
    let mut bad = Vec::new();
    for (fld, val) in [
        ("logRho", 2.35),
        ("logT", 7.3),
        ("logRho_any", cfg.log_rho_min_for_any_skye),
        ("logRho_all", cfg.log_rho_min_for_all_skye),
        ("logT_any", cfg.log_t_min_for_any_skye),
        ("logT_all", cfg.log_t_min_for_all_skye),
    ] {
        let exp = mesa_skye_bits(case_id, "in", fld);
        if val.to_bits() != exp {
            bad.push(format!(
                "{case_id}.in.{fld}: MESA {e:.17e} (0x{exp:016X}) vs Flex {val:.17e} (0x{:016X}) ULP={}",
                val.to_bits(),
                val.to_bits().abs_diff(exp),
                e = f64::from_bits(exp),
            ));
        }
    }
    let (a, dt, dr) = get_skye_alfa_simple(cfg, 2.35, 7.3);
    for (fld, got) in
        [("alfa", a), ("d_alfa_dlogT", dt), ("d_alfa_dlogRho", dr)]
    {
        let exp = mesa_skye_bits(case_id, "out", fld);
        if got.to_bits() != exp {
            bad.push(format!(
                "{case_id}.{fld}: MESA {e:.17e} (0x{exp:016X}) vs Flex {got:.17e} (0x{:016X}) ULP={}",
                got.to_bits(),
                got.to_bits().abs_diff(exp),
                e = f64::from_bits(exp),
            ));
        }
    }
    assert_eq!(
        bad.len(),
        0,
        "Get_Skye_alfa_simple departures:\n{}",
        bad.join("\n")
    );
}

fn make_comp(
    y_in: f64,
    xc_in: f64,
    xo_in: f64,
) -> (Vec<usize>, Vec<f64>, f64, f64) {
    let cat = NuclideCatalog::builtin();
    let chem_id = ["h1", "he4", "c12", "n14", "o16", "ne20", "mg24"]
        .into_iter()
        .map(|n| cat.get_by_name(n).unwrap().chem_id)
        .collect::<Vec<_>>();
    let xa = vec![0.0, y_in, xc_in, 0.0, xo_in, 0.0, 0.0];
    let comp = get_composition_info(&cat, &chem_id, &xa, true).unwrap();
    (chem_id, xa, comp.abar, comp.zbar)
}

fn run_skye_liquid_and_ion_offset_cases() {
    let ht = HelmTable::builtin().expect("builtin HELM table");
    let (chem_id, xa, abar, zbar) = make_comp(0.0, 1.0, 0.0);

    // 1. skye_c12_liq (use_ion_offsets = false)
    let mut cfg = SkyeConfig {
        skye_use_ion_offsets: false,
        skye_solid_mixing_rule: SkyeSolidMixingRule::Ogata,
        ..SkyeConfig::default()
    };
    let out1 = eval_skye_eos(&cfg, ht, 1.0e6, 1.0e8, abar, zbar, &chem_id, &xa)
        .unwrap();
    common::assert_eos_against_mesa(
        "skye_c12_liq",
        &out1,
        1.0e-12,
        1.0e-12,
        &[],
        false,
    );

    // 2. skye_c12_liq_ion (use_ion_offsets = true; MESA 26.4.1 #192)
    cfg.skye_use_ion_offsets = true;
    let out2 = eval_skye_eos(&cfg, ht, 1.0e6, 1.0e8, abar, zbar, &chem_id, &xa)
        .unwrap();
    common::assert_eos_against_mesa(
        "skye_c12_liq_ion",
        &out2,
        1.0e-12,
        1.0e-12,
        &[],
        false,
    );

    // 3. skye_co_liq
    cfg.skye_use_ion_offsets = false;
    let (chem_id_co, xa_co, abar_co, zbar_co) = make_comp(0.0, 0.5, 0.5);
    let out3 = eval_skye_eos(
        &cfg,
        ht,
        1.0e6,
        1.0e7,
        abar_co,
        zbar_co,
        &chem_id_co,
        &xa_co,
    )
    .unwrap();
    common::assert_eos_against_mesa(
        "skye_co_liq",
        &out3,
        1.0e-11,
        1.0e-11,
        &[],
        false,
    );

    // 4. eosdt_skye_helm_blend (Skye + HELM with 4D tables disabled)
    let mut blend_cfg = EosConfig {
        use_skye: true,
        use_pc: false,
        use_free_eos: false,
        use_opal_scvh: false,
        ..EosConfig::default()
    };
    blend_cfg.skye.skye_use_ion_offsets = false;
    let rho_bl = flexmesa_math::exp10(3.45);
    let out_blend =
        eos_dt_get(ht, rho_bl, 1.0e7, &chem_id, &xa, &blend_cfg).unwrap();
    common::assert_eos_against_mesa(
        "eosdt_skye_helm_blend",
        &out_blend,
        1.0e-8,
        1.0e-8,
        &[],
        false,
    );

    // 5. eosdt_skye_helm_blend_ion (default Skye_use_ion_offsets = true)
    blend_cfg.skye.skye_use_ion_offsets = true;
    let out_blend_ion =
        eos_dt_get(ht, rho_bl, 1.0e7, &chem_id, &xa, &blend_cfg).unwrap();
    common::assert_eos_against_mesa(
        "eosdt_skye_helm_blend_ion",
        &out_blend_ion,
        1.0e-8,
        1.0e-8,
        &[],
        false,
    );
}

#[test]
fn test_skye_liquid_and_ion_offset_regimes_match_mesa_26_4_1_oracle() {
    run_skye_liquid_and_ion_offset_cases();
}

#[test]
fn test_skye_solid_and_phase_transition_regimes_match_mesa_26_4_1_oracle() {
    let ht = HelmTable::builtin().expect("builtin HELM table");
    let (chem_id_co, xa_co, abar_co, zbar_co) = make_comp(0.0, 0.5, 0.5);
    let mut cfg = SkyeConfig {
        skye_use_ion_offsets: false,
        skye_solid_mixing_rule: SkyeSolidMixingRule::Ogata,
        ..SkyeConfig::default()
    };

    // 4. skye_co_sol_ogata (T = 2e6 K, rho = 1e6 g/cm^3)
    let out_og = eval_skye_eos(
        &cfg,
        ht,
        1.0e6,
        2.0e6,
        abar_co,
        zbar_co,
        &chem_id_co,
        &xa_co,
    )
    .unwrap();
    common::assert_eos_against_mesa(
        "skye_co_sol_ogata",
        &out_og,
        1.0e-9,
        1.0e-9,
        &[],
        false,
    );

    // 5. skye_co_sol_pc (PC solid mixing rule)
    cfg.skye_solid_mixing_rule = SkyeSolidMixingRule::Pc;
    let out_pc = eval_skye_eos(
        &cfg,
        ht,
        1.0e6,
        2.0e6,
        abar_co,
        zbar_co,
        &chem_id_co,
        &xa_co,
    )
    .unwrap();
    common::assert_eos_against_mesa(
        "skye_co_sol_pc",
        &out_pc,
        1.0e-9,
        1.0e-9,
        &[],
        false,
    );

    // 6. skye_c12_phase (T = 2.65e6 K, rho = 1e6 g/cm^3)
    cfg.skye_solid_mixing_rule = SkyeSolidMixingRule::Ogata;
    let (chem_id_c, xa_c, abar_c, zbar_c) = make_comp(0.0, 1.0, 0.0);
    let out_ph = eval_skye_eos(
        &cfg, ht, 1.0e6, 2.65e6, abar_c, zbar_c, &chem_id_c, &xa_c,
    )
    .unwrap();
    common::assert_eos_against_mesa(
        "skye_c12_phase",
        &out_ph,
        1.0e-9,
        1.0e-9,
        &[],
        false,
    );
}

fn check_skye_gamma_de(abar: f64, zbar: f64, e_val: f64) {
    let exp_gam = common::load_scalar_row::<6>("INV", "gamma_de_c12_1e4");
    let exp_gam_res =
        common::load_scalar_row::<6>("INV", "gamma_de_c12_1e4_res");
    let gam = eos_gamma_de_get(abar, zbar, 1.0e4, e_val, 5.0 / 3.0);
    assert!((gam.t - exp_gam[0]).abs() / exp_gam[0] < 1.0e-13);
    assert!((gam.log10_t - exp_gam[1]).abs() < 1.0e-13);
    let d_gam = (
        gam.dln_t_dln_e_c_rho,
        gam.dln_t_dlnd_c_e,
        gam.dln_pgas_dln_e_c_rho,
        gam.dln_pgas_dlnd_c_e,
    );
    assert_eq!(d_gam, (exp_gam[2], exp_gam[3], exp_gam[4], exp_gam[5]));
    assert!((gam.res[I_MU] - exp_gam_res[0]).abs() < 1.0e-15);
    assert_eq!(gam.res[I_LN_FREE_E], exp_gam_res[1]);
    assert_eq!(gam.res[I_ETA], exp_gam_res[2]);
    assert_eq!(gam.res[I_DS_DT], 0.0);
    assert_eq!(gam.res[I_DS_DRHO], 0.0);
    assert_eq!(gam.d_dlnd[I_LN_S], 0.0);
    assert_eq!(gam.d_dln_t[I_LN_S], 0.0);
}

#[test]
fn test_skye_alfa_blend_and_inverse_solvers_match_mesa_26_4_1_oracle() {
    let ht = HelmTable::builtin().expect("builtin HELM table");

    let exp_in = common::load_scalar_row::<3>("ALFA", "skye_alfa_c12_4p0_7p5");
    let (a_in, dt_in, dr_in) = get_skye_alfa(ht, 4.0, 7.5, 12.0, 6.0);
    assert_eq!((a_in, dt_in, dr_in), (exp_in[0], exp_in[1], exp_in[2]));

    let exp_ed = common::load_scalar_row::<3>("ALFA", "skye_alfa_c12_3p45_7p0");
    let (a_ed, dt_ed, dr_ed) = get_skye_alfa(ht, 3.45, 7.0, 12.0, 6.0);
    assert!((a_ed - exp_ed[0]).abs() < 1.0e-13);
    assert!((dt_ed - exp_ed[1]).abs() < 1.0e-13);
    assert!((dr_ed - exp_ed[2]).abs() < 1.0e-13);

    // Explicit f64 bounds (`2.3d0, 2.4d0, 7.2d0, 7.4d0` in MESA).
    let skye_cfg_f64 = SkyeConfig {
        log_rho_min_for_any_skye: 2.3,
        log_rho_min_for_all_skye: 2.4,
        log_t_min_for_any_skye: 7.2,
        log_t_min_for_all_skye: 7.4,
        ..SkyeConfig::default()
    };
    let exp_sm_f64 =
        common::load_scalar_row::<3>("ALFA", "skye_alfa_simple_f64_2p35_7p3");
    let (a_sm, dt_sm, dr_sm) = get_skye_alfa_simple(&skye_cfg_f64, 2.35, 7.3);
    assert!((a_sm - exp_sm_f64[0]).abs() < 1.0e-13);
    assert!((dt_sm - exp_sm_f64[1]).abs() < 1.0e-13);
    assert!((dr_sm - exp_sm_f64[2]).abs() < 1.0e-13);
    with_numerics_mode(NumericsMode::Strict, || {
        check_skye_alfa_simple_bits(
            "skye_alfa_simple_f64_2p35_7p3",
            &skye_cfg_f64,
        );
    });

    // Inverse solvers on pure HELM (`use_skye = false, use_pc = false`)
    let (chem_id, xa, abar, zbar) = make_comp(0.0, 1.0, 0.0);
    let cfg = EosConfig {
        use_skye: false,
        use_pc: false,
        use_free_eos: false,
        use_opal_scvh: false,
        ..EosConfig::default()
    };
    let fwd = eos_dt_get(ht, 1.0e4, 1.0e8, &chem_id, &xa, &cfg).unwrap();
    let log_pgas = fwd.res[I_LN_PGAS] / LN10;
    let log_e = fwd.res[I_LN_E] / LN10;
    let e_val = flexmesa_math::exp(fwd.res[I_LN_E]);

    let exp_pt = common::load_scalar_row::<6>("INV", "helm_c12_pt_4_8");
    let pt = eos_pt_get(ht, &chem_id, &xa, log_pgas, 8.0, &cfg).unwrap();
    assert!((pt.rho - exp_pt[0]).abs() / exp_pt[0] < 1.0e-8);
    assert!((pt.log10_rho - exp_pt[1]).abs() < 1.0e-8);
    assert!((pt.dln_rho_dln_pgas_const_t - exp_pt[2]).abs() < 1.0e-8);
    assert!((pt.dln_rho_dln_t_const_pgas - exp_pt[3]).abs() < 1.0e-8);

    let exp_inv_t = common::load_scalar_row::<6>("INV", "helm_c12_get_t_4_8");
    let inv_t = eos_dt_get_t(
        ht,
        &chem_id,
        &xa,
        4.0,
        I_LN_E as isize,
        log_e * LN10,
        7.5,
        1.0e-12,
        1.0e-12,
        100,
        &cfg,
    )
    .unwrap();
    assert!((inv_t.log10_result - exp_inv_t[0]).abs() < 1.0e-9);
    assert!((inv_t.eos.res[I_LN_PGAS] - fwd.res[I_LN_PGAS]).abs() < 1.0e-11);
    assert!((inv_t.eos.res[I_LN_PGAS] - exp_inv_t[1]).abs() < 1.0e-8);

    let exp_inv_rho =
        common::load_scalar_row::<6>("INV", "helm_c12_get_rho_4_8");
    let inv_rho = eos_dt_get_rho(
        ht,
        &chem_id,
        &xa,
        8.0,
        I_LN_PGAS as isize,
        log_pgas * LN10,
        3.5,
        1.0e-12,
        1.0e-12,
        100,
        &cfg,
    )
    .unwrap();
    assert!((inv_rho.log10_result - exp_inv_rho[0]).abs() < 1.0e-10);
    assert!((inv_rho.eos.res[I_LN_E] - fwd.res[I_LN_E]).abs() < 1.0e-11);
    assert!((inv_rho.eos.res[I_LN_E] - exp_inv_rho[1]).abs() < 1.0e-8);

    check_skye_gamma_de(abar, zbar, e_val);
}

#[test]
fn test_skye_alfa_simple_default_bounds_match_mesa_26_4_1_unwidened() {
    let skye_cfg = SkyeConfig::default();
    let exp_def =
        common::load_scalar_row::<3>("ALFA", "skye_alfa_simple_2p35_7p3");
    let (a_sm, dt_sm, dr_sm) = get_skye_alfa_simple(&skye_cfg, 2.35, 7.3);
    assert!((a_sm - exp_def[0]).abs() < 1.0e-13);
    assert!((dt_sm - exp_def[1]).abs() < 1.0e-13);
    assert!((dr_sm - exp_def[2]).abs() < 1.0e-13);
    with_numerics_mode(NumericsMode::Strict, || {
        check_skye_alfa_simple_bits("skye_alfa_simple_2p35_7p3", &skye_cfg);
    });
}
