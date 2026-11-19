// SPDX-License-Identifier: LGPL-3.0-or-later
//! Upstream Fortran MESA `26.4.1` oracle tests for Wave 3a.2 (`HELM` equation
//! of state across pure He, pure H, and solar mixtures; `#161`, `T2-15`).
//!
//! Expected `16 x 5` (`res, d_dlnd, d_dlnT, d_dabar, d_dzbar`) reference rows
//! are loaded from `benchmarks/fortran_mesa/eos_oracles/mesa_26_4_1_eos_oracles.tsv`.

#![allow(clippy::similar_names)]

mod common;

use flexmesa_eos::{eval_helm_eos, EOS_BASIC_RESULT_NAMES};

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
fn test_helm_eos_pure_helium_matches_mesa_26_4_1_oracle() {
    assert_helm_case("pure_he_2e8_100", 2.0e8, 100.0, 4.0, 2.0);
}

#[test]
fn test_helm_eos_pure_hydrogen_and_solar_match_mesa_26_4_1_oracle() {
    assert_helm_case("pure_h_2e8_100", 2.0e8, 100.0, 1.0, 1.0);
    assert_helm_case(
        "solar_2e8_100",
        2.0e8,
        100.0,
        1.296_635_341_025_522_2,
        1.102_140_039_871_693_8,
    );
}
