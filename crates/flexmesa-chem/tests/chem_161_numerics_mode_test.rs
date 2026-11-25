// SPDX-License-Identifier: GPL-3.0-or-later
//! Numerics-mode invariance tests for `NuclideCatalog` / `NuclideEntry` load-time
//! `z53` calculation (`chem/private/chem_isos_io.f90:133`).
//!
//! Runs in a dedicated `*numerics_mode_test.rs` process per issue #210 routing
//! governance.

#![forbid(unsafe_code)]
#![allow(
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::unreadable_literal
)]

use flexmesa_chem::{NuclideCatalog, NuclideEntry, DEL_MN, DEL_MP, NPART};
use flexmesa_const::FIVE_THIRDS;
use flexmesa_math::{pow_strict, with_numerics_mode, NumericsMode};

#[test]
fn nuclide_load_z53_is_invariant_under_fast_numerics_mode() {
    let pfcn = [1.0_f64; NPART];
    let strict_entry = with_numerics_mode(NumericsMode::Strict, || {
        NuclideEntry::from_raw(
            1,
            "fe56".to_string(),
            26,
            30,
            0.0,
            -60.605352,
            0,
            pfcn,
            DEL_MP,
            DEL_MN,
        )
    });
    let fast_entry = with_numerics_mode(NumericsMode::Fast, || {
        NuclideEntry::from_raw(
            1,
            "fe56".to_string(),
            26,
            30,
            0.0,
            -60.605352,
            0,
            pfcn,
            DEL_MP,
            DEL_MN,
        )
    });

    assert_eq!(
        strict_entry.z53.to_bits(),
        pow_strict(26.0_f64, FIVE_THIRDS).to_bits()
    );
    assert_eq!(fast_entry.z53.to_bits(), strict_entry.z53.to_bits());
    assert_eq!(fast_entry.w.to_bits(), strict_entry.w.to_bits());
    assert_eq!(
        fast_entry.binding_energy.to_bits(),
        strict_entry.binding_energy.to_bits()
    );

    let mini_isotopes = "\
header
neut        1.0086649    0    1   0.5   8.071317100
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
h1          1.0078250    1    0   0.5   7.288970500
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
c12        12.0000000    6    6   0.0   0.000000000
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
o16        15.9949146    8    8   0.0  -4.737001410
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
fe56       55.9349375   26   30   0.0 -60.605352000
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
";
    let strict_cat = with_numerics_mode(NumericsMode::Strict, || {
        NuclideCatalog::from_isotopes_str(mini_isotopes).expect("strict parse")
    });
    let fast_cat = with_numerics_mode(NumericsMode::Fast, || {
        NuclideCatalog::from_isotopes_str(mini_isotopes).expect("fast parse")
    });
    let builtin_cat = with_numerics_mode(NumericsMode::Fast, || {
        NuclideCatalog::builtin_ref()
    });

    for nm in ["neut", "h1", "c12", "o16", "fe56"] {
        let s = strict_cat.get_by_name(nm).expect("strict iso");
        let f = fast_cat.get_by_name(nm).expect("fast iso");
        let b = builtin_cat.get_by_name(nm).expect("builtin iso");
        assert_eq!(f.z53.to_bits(), s.z53.to_bits(), "z53 Fast vs Strict {nm}");
        assert_eq!(
            f.z53.to_bits(),
            b.z53.to_bits(),
            "z53 Fast vs builtin {nm}"
        );
    }
}
