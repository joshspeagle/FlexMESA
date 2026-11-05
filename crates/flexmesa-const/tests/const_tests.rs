// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests verifying `flexmesa-const` against upstream MESA golden
//! output (`const/test/test_output`).
//!
//! Each test validates CODATA 2018 and IAU 2015 CGS constants and mixing
//! enumerations to IEEE-754 double-precision tolerance (`rtol <= 1e-15`).

#![allow(clippy::approx_constant, clippy::excessive_precision)]

use flexmesa_const as c;
use flexmesa_testing::assert_values_match;

/// Verify all 23 derived quantities emitted by upstream `test_const.f90`
/// against `const/test/test_output` to relative tolerance `1e-15`.
#[test]
fn test_const_golden_parity_matches_upstream_mesa() {
    // Assemble the 23 constants in the exact order printed by test_const.f90.
    let actual = [
        c::PI,
        c::LN10,
        c::BOLTZ_SIGMA,
        c::BOLTZ_SIGMA * 4.0,
        c::BOLTZ_SIGMA * 4.0 / c::CLIGHT,
        c::BOLTZ_SIGMA / c::CLIGHT,
        c::CRAD,
        c::SECYER,
        c::MSUN,
        c::RSUN,
        c::LSUN,
        c::LY,
        c::M_EARTH,
        c::AU,
        c::AMU,
        c::MN,
        c::MP,
        c::ME,
        c::PLANCK_H,
        c::QE,
        c::AVO,
        c::CLIGHT,
        c::KERG,
    ];
    // Reference values parsed from upstream const/test/test_output.
    let expected = [
        3.141_592_653_589_793_1,
        2.302_585_092_994_045_5,
        5.670_374_419_184_425_9e-5,
        2.268_149_767_673_770_4e-4,
        7.565_733_250_279_999_3e-15,
        1.891_433_312_569_999_8e-15,
        7.565_733_250_279_999_3e-15,
        3.155_76e7,
        1.988_409_870_698_050_4e33,
        6.957e10,
        3.828_000_000_000_000_3e33,
        9.460_730_472_580_8e17,
        5.972_167_867_791_379_0e27,
        1.495_978_707e13,
        1.660_539_067_173_846_6e-24,
        1.674_927_498_040_000_1e-24,
        1.672_621_923_69e-24,
        9.109_383_701_5e-28,
        6.626_070_149_999_999_9e-27,
        4.803_204_712_570_263_4e-10,
        6.022_140_759_999_999_9e23,
        2.997_924_58e10,
        1.380_649e-16,
    ];
    assert_values_match(&actual, &expected, 1e-15, 1e-45, "const_test_output")
        .expect("all 23 constants match upstream MESA");
}

/// Verify integer round-trip conversion for all 12 `MixingType` codes (`-1..=10`)
/// and confirm out-of-range codes return `None`.
#[test]
fn test_mixing_type_roundtrip_codes() {
    for code in -1..=10 {
        let kind = c::MixingType::from_code(code).expect("valid mixing code");
        assert_eq!(kind.code(), code);
    }
    assert!(c::MixingType::from_code(-2).is_none());
    assert!(c::MixingType::from_code(11).is_none());
}
