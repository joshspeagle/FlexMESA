// SPDX-License-Identifier: LGPL-3.0-or-later
//! Upstream Fortran MESA `26.4.1` oracle tests for Wave 3a.2 (`HELM` equation
//! of state across pure He, pure H, and solar mixtures; `#161`, `T2-15`).
//!
//! Expected `16 x 5` (`res, d_dlnd, d_dlnT, d_dabar, d_dzbar`) reference rows
//! are loaded from `benchmarks/fortran_mesa/eos_oracles/mesa_26_4_1_eos_oracles.tsv`.

#![allow(clippy::similar_names)]

mod common;

#[test]
fn test_helm_eos_pure_helium_matches_mesa_26_4_1_oracle() {
    common::assert_helm_against_mesa("pure_he_2e8_100", 2.0e8, 100.0, 4.0, 2.0, 1.0e-13);
}

#[test]
fn test_helm_eos_pure_hydrogen_and_solar_match_mesa_26_4_1_oracle() {
    common::assert_helm_against_mesa("pure_h_2e8_100", 2.0e8, 100.0, 1.0, 1.0, 1.0e-13);
    common::assert_helm_against_mesa(
        "solar_2e8_100",
        2.0e8,
        100.0,
        1.296_635_341_025_522_2,
        1.102_140_039_871_693_8,
        1.0e-13,
    );
}
