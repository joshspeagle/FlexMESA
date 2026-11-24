// SPDX-License-Identifier: GPL-3.0-or-later
//! Upstream Fortran MESA `26.4.1` oracle tests for Wave 3a.3 solid/quantum-melting
//! `PC` regimes, `get_pc_alfa`, and `eosDT_get` `PC`/`HELM` & `PC`/`FreeEOS`
//! blending (`#161`, `T2-15`).
//!
//! Expected `26 x 3` (`EOS`) and `ALFA` reference rows are loaded from
//! `benchmarks/fortran_mesa/eos_oracles/mesa_26_4_1_eos_oracles.tsv`.

#![allow(clippy::similar_names)]

mod common;

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_eos::{
    eos_dt_get, eval_pc_eos, get_pc_alfa, EosConfig, HelmTable, PcConfig,
    I_FRAC_IDEAL,
};

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

fn run_pc_solid_and_blend_suite() {
    let cfg = PcConfig::default();

    // 5. pc_c12_sol: T = 1e6 K, rho = 1e6 g/cm^3
    let (chem_id, xa, abar, zbar) = make_comp(0.0, 1.0, 0.0);
    let out_sol =
        eval_pc_eos(&cfg, 1.0e6, 1.0e6, abar, zbar, &chem_id, &xa).unwrap();
    common::assert_eos_against_mesa(
        "pc_c12_sol",
        &out_sol,
        1.0e-11,
        5.0e-9,
        &[],
        false,
    );

    // 6. pc_co_sol: 50/50 C12/O16 at T = 1e6 K, rho = 1e6 g/cm^3
    let (chem_id_co, xa_co, abar_co, zbar_co) = make_comp(0.0, 0.5, 0.5);
    let out_co_sol =
        eval_pc_eos(&cfg, 1.0e6, 1.0e6, abar_co, zbar_co, &chem_id_co, &xa_co)
            .unwrap();
    common::assert_eos_against_mesa(
        "pc_co_sol",
        &out_co_sol,
        1.0e-11,
        5.0e-9,
        &[],
        false,
    );

    // 7. pc_he4_qmelt: pure He4 at T = 1e5 K, rho = 1e8 g/cm^3 (quantum melting RS < RS2)
    let (chem_id_he, xa_he, abar_he, zbar_he) = make_comp(1.0, 0.0, 0.0);
    let out_he =
        eval_pc_eos(&cfg, 1.0e8, 1.0e5, abar_he, zbar_he, &chem_id_he, &xa_he)
            .unwrap();
    common::assert_eos_against_mesa(
        "pc_he4_qmelt",
        &out_he,
        1.0e-11,
        5.0e-9,
        &[],
        false,
    );

    // 8. eosdt_pc_helm_blend (PC + HELM with 4D tables disabled)
    let ht = HelmTable::builtin().expect("builtin HELM table");
    let mut eos_cfg = EosConfig {
        use_skye: false,
        use_pc: true,
        use_free_eos: false,
        use_opal_scvh: false,
        ..EosConfig::default()
    };
    eos_cfg.pc.pc_use_gamma_limit_instead_of_t = true;
    let rho = flexmesa_math::exp10(2.9);
    let t = flexmesa_math::exp10(6.5);
    let out_blend = eos_dt_get(ht, rho, t, &chem_id, &xa, &eos_cfg).unwrap();
    common::assert_eos_against_mesa(
        "eosdt_pc_helm_blend",
        &out_blend,
        1.0e-11,
        5.0e-9,
        &[],
        false,
    );

    // 9. eosdt_pc_freeeos_blend (with full data-v1 FreeEOS active, Z = 1.0 blends PC + FreeEOS)
    let mut cfg_free = eos_cfg;
    cfg_free.use_free_eos = true;
    cfg_free.okay_to_convert_ierr_to_skip = false;
    let out_free = eos_dt_get(ht, rho, t, &chem_id, &xa, &cfg_free).unwrap();
    common::assert_eos_against_mesa(
        "eosdt_pc_freeeos_blend",
        &out_free,
        1.0e-11,
        5.0e-9,
        &[],
        false,
    );
}

#[test]
fn test_pc_solid_and_quantum_melting_regimes_match_mesa_26_4_1_oracle() {
    run_pc_solid_and_blend_suite();
}

#[test]
fn test_pc_alfa_and_helm_off_table_skip_match_mesa_26_4_1_oracle() {
    let mut pc_cfg = PcConfig::default();

    let exp1 = common::load_scalar_row::<3>("ALFA", "pc_alfa_gamma_c12_2p9_6p5");
    let (a1, dt1, dr1) = get_pc_alfa(&pc_cfg, 2.9, 6.5, 12.0, 6.0);
    assert!((a1 - exp1[0]).abs() < 1.0e-14);
    assert!((dt1 - exp1[1]).abs() < 1.0e-14);
    assert!((dr1 - exp1[2]).abs() < 1.0e-13);

    let exp2 =
        common::load_scalar_row::<3>("ALFA", "pc_alfa_gamma_c12_3p5_8p35");
    let (a2, dt2, dr2) = get_pc_alfa(&pc_cfg, 3.5, 8.35, 12.0, 6.0);
    assert!((a2 - exp2[0]).abs() < 1.0e-14);
    assert!((dt2 - exp2[1]).abs() < 1.0e-14);
    assert!((dr2 - exp2[2]).abs() < 1.0e-14);

    pc_cfg.pc_use_gamma_limit_instead_of_t = false;
    let exp3 =
        common::load_scalar_row::<3>("ALFA", "pc_alfa_tlim_c12_3p3_7p65");
    let (a3, dt3, dr3) = get_pc_alfa(&pc_cfg, 3.3, 7.65, 12.0, 6.0);
    assert!((a3 - exp3[0]).abs() < 1.0e-14);
    assert!((dt3 - exp3[1]).abs() < 1.0e-13);
    assert!((dr3 - exp3[2]).abs() < 1.0e-14);

    // Out-of-range HELM point (`eos/private/helm.f90:192-212`, `eosdt_eval.f90:1511-1514`)
    let ht = HelmTable::builtin().expect("builtin HELM table");
    let (chem_id, xa, _, _) = make_comp(0.0, 1.0, 0.0);
    let cfg_helm_only = EosConfig {
        use_pc: false,
        use_skye: false,
        use_free_eos: false,
        use_opal_scvh: false,
        use_cms: false,
        okay_to_convert_ierr_to_skip: true,
        ..EosConfig::default()
    };
    let out_bnd = eos_dt_get(ht, 1.0e-5, 900.0, &chem_id, &xa, &cfg_helm_only)
        .expect("off-table HELM error is converted to a skip by default");
    assert!((out_bnd.res[I_FRAC_IDEAL] - 1.0).abs() < 1.0e-14);
    let cfg_no_skip = EosConfig {
        okay_to_convert_ierr_to_skip: false,
        ..cfg_helm_only
    };
    assert!(
        eos_dt_get(ht, 1.0e-5, 900.0, &chem_id, &xa, &cfg_no_skip).is_err(),
        "without the conversion the HELM error is returned"
    );
}

#[test]
fn test_pc_solid_and_blend_regimes_match_mesa_26_4_1_unwidened() {
    run_pc_solid_and_blend_suite();
}
