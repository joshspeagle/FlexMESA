// SPDX-License-Identifier: LGPL-3.0-or-later
//! Upstream Fortran MESA `26.4.1` oracle tests for Wave 3a.2 (`HELM`
//! degenerate, relativistic pair-production, crystallized WD, and Hermite basis
//! regimes; `#161`, `T2-15`).
//!
//! Expected `16 x 5` reference rows and `IERR` codes are loaded from
//! `benchmarks/fortran_mesa/eos_oracles/mesa_26_4_1_eos_oracles.tsv`.

#![allow(clippy::similar_names)]

mod common;

use flexmesa_eos::{
    eval_helm_eos, eval_helm_eos_zones, helmeos2, CubicBasis1D, HelmConfig,
    HelmTable, QuinticBasis1D, EOS_BASIC_RESULT_NAMES, I_ETA, I_LN_PGAS,
};

fn assert_helm_case(
    case_id: &str,
    temp: f64,
    den: f64,
    abar: f64,
    zbar: f64,
) {
    let expected = common::load_indexed_rows::<16, 5>("HELM", case_id);
    let (out, d_dabar, d_dzbar) =
        eval_helm_eos(temp, den, abar, zbar).expect(case_id);
    for (idx, exp) in expected.iter().enumerate() {
        let name = EOS_BASIC_RESULT_NAMES[idx];
        let got = [
            out.res[idx],
            out.d_dlnd[idx],
            out.d_dln_t[idx],
            d_dabar[idx],
            d_dzbar[idx],
        ];
        for (col, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
            let scale = e.abs().max(1.0);
            let rel_err = (g - e).abs() / scale;
            assert!(
                rel_err < 1.0e-14,
                "{case_id} {name} col {col}: got {g:.16e}, mesa_26_4_1 {e:.16e} \
                 (rel_err = {rel_err:.3e})"
            );
        }
    }
}

#[test]
fn test_helm_eos_degenerate_and_pair_regimes_match_mesa_26_4_1_oracle() {
    // 1. test_components (degenerate, plasg > 1): logT = 6.0, logRho = 3.5
    assert_helm_case(
        "test_components",
        1.0e6,
        flexmesa_math::exp10(3.5),
        2.990_964_711_670_598,
        1.674_940_238_535_534_8,
    );

    // 2. sample_eos (relativistic e-e+ pairs): T = 1e9, Rho = 1e4, C12
    assert_helm_case("sample_eos_c12", 1.0e9, 1.0e4, 12.0, 6.0);

    // 3. Cold crystallized WD core where sgas < 1e-20 returns Err(1) in MESA
    let (exp_ierr, _) = common::load_ierr_row("crystallized_wd_1e6_6p5");
    let res_err = eval_helm_eos(
        1.0e6,
        flexmesa_math::exp10(6.5),
        13.714_285_714_285_715,
        6.857_142_857_142_858,
    );
    assert_eq!(res_err, Err(exp_ierr));
}

#[test]
fn test_helm_basis_hermite_endpoint_conditions_and_zones() {
    let q0 = QuinticBasis1D::eval(0.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    let q1 = QuinticBasis1D::eval(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    assert!((q0.w0[0] - 1.0).abs() < 1.0e-15);
    assert!(q0.w0[3].abs() < 1.0e-15);
    assert!(q1.w0[0].abs() < 1.0e-15);
    assert!((q1.w0[3] - 1.0).abs() < 1.0e-15);

    let c0 = CubicBasis1D::eval(0.0, 1.0, 1.0, 1.0, 1.0);
    let c1 = CubicBasis1D::eval(1.0, 1.0, 1.0, 1.0, 1.0);
    assert!((c0.w0[0] - 1.0).abs() < 1.0e-15);
    assert!((c1.w0[2] - 1.0).abs() < 1.0e-15);

    let pure_he_oracle =
        common::load_indexed_rows::<16, 5>("HELM", "pure_he_2e8_100");
    let pure_h_oracle =
        common::load_indexed_rows::<16, 5>("HELM", "pure_h_2e8_100");
    let pair_oracle =
        common::load_indexed_rows::<16, 5>("HELM", "sample_eos_c12");

    let table = HelmTable::builtin().expect("builtin HELM table");
    let (f_ele, eta_ele, xne_fer) = table.eval_electron_positron_order3(
        2.0e8,
        flexmesa_math::log10(2.0e8_f64),
        100.0,
        4.0,
        2.0,
    );
    assert!(f_ele.val.is_finite() && f_ele.d3val1.is_finite());
    assert!((eta_ele.val - pure_he_oracle[I_ETA][0]).abs() < 1.0e-14);
    assert!(xne_fer.val > 0.0);

    let batch = eval_helm_eos_zones(
        &[2.0e8, 2.0e8, 1.0e9],
        &[100.0, 100.0, 1.0e4],
        &[4.0, 1.0, 12.0],
        &[2.0, 1.0, 6.0],
    );
    assert_eq!(batch.len(), 3);
    assert!(
        (batch[0].res[I_LN_PGAS] - pure_he_oracle[I_LN_PGAS][0]).abs()
            < 1.0e-14
    );
    assert!(
        (batch[1].res[I_LN_PGAS] - pure_h_oracle[I_LN_PGAS][0]).abs() < 1.0e-14
    );
    assert!(
        (batch[2].res[I_LN_PGAS] - pair_oracle[I_LN_PGAS][0]).abs() < 1.0e-14
    );

    // Off-table temperature returns Err(1) matching `eos/private/helm.f90:194`
    let (exp_low_t_ierr, _) = common::load_ierr_row("helmeos2_low_t_100_100");
    let cfg = HelmConfig::default();
    let res_low_t = helmeos2(table, 100.0, 2.0, 100.0, 2.0, 4.0, 2.0, &cfg);
    assert_eq!(res_low_t, Err(exp_low_t_ierr));
}
