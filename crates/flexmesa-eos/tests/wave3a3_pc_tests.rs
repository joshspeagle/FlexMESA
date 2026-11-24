// SPDX-License-Identifier: GPL-3.0-or-later
//! Upstream Fortran MESA `26.4.1` oracle tests for Wave 3a.3 (`PC` liquid and
//! phase-transition regimes; `#161`, `T2-15`).
//!
//! Expected `26 x 3` (`res, d_dlnd, d_dlnT`) reference rows are loaded from
//! `benchmarks/fortran_mesa/eos_oracles/mesa_26_4_1_eos_oracles.tsv`.

#![allow(clippy::similar_names)]

mod common;

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_eos::{eval_pc_eos, PcConfig};

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

fn run_pc_liquid_and_transition_suite() {
    let cfg = PcConfig::default();

    // 1. pc_c12_liq: pure C12 at T = 1e8 K, rho = 1e6 g/cm^3
    let (chem_id, xa, abar, zbar) = make_comp(0.0, 1.0, 0.0);
    let out =
        eval_pc_eos(&cfg, 1.0e6, 1.0e8, abar, zbar, &chem_id, &xa).unwrap();
    common::assert_eos_against_mesa(
        "pc_c12_liq",
        &out,
        1.0e-11,
        5.0e-9,
        &[],
        false,
    );

    // 2. pc_co_liq: 50/50 C12/O16 mixture at T = 1e7 K, rho = 1e6 g/cm^3
    let (chem_id_co, xa_co, abar_co, zbar_co) = make_comp(0.0, 0.5, 0.5);
    let out_co =
        eval_pc_eos(&cfg, 1.0e6, 1.0e7, abar_co, zbar_co, &chem_id_co, &xa_co)
            .unwrap();
    common::assert_eos_against_mesa(
        "pc_co_liq",
        &out_co,
        1.0e-11,
        5.0e-9,
        &[],
        false,
    );

    // 3. pc_c12_trans_eosfi: T = 3.1e6 K, rho = 1e6 g/cm^3 (EOSFI8 blend window)
    let out_tr1 =
        eval_pc_eos(&cfg, 1.0e6, 3.1e6, abar, zbar, &chem_id, &xa).unwrap();
    common::assert_eos_against_mesa(
        "pc_c12_trans_eosfi",
        &out_tr1,
        1.0e-11,
        5.0e-9,
        &[],
        false,
    );

    // 4. pc_c12_trans_phase: T = 2.05e6 K, rho = 1e6 g/cm^3 (170 < GAMImean < 180)
    let out_tr2 =
        eval_pc_eos(&cfg, 1.0e6, 2.05e6, abar, zbar, &chem_id, &xa).unwrap();
    common::assert_eos_against_mesa(
        "pc_c12_trans_phase",
        &out_tr2,
        1.0e-11,
        5.0e-9,
        &[],
        false,
    );
}

#[test]
fn test_pc_liquid_and_transition_regimes_match_mesa_26_4_1_oracle() {
    run_pc_liquid_and_transition_suite();
}

#[test]
fn test_pc_liquid_and_transition_regimes_match_mesa_26_4_1_unwidened() {
    run_pc_liquid_and_transition_suite();
}
