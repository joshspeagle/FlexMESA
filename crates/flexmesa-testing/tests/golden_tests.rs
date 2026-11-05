// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration and property-based tests for `flexmesa-testing`.
//!
//! Verifies Fortran float parsing (`D`, `E`, and bare 3-digit exponents),
//! structured [`GoldenRecord`] extraction across MESA `test_output` formats,
//! dense table extraction (`extract_golden_table`), numerical parity checks
//! (`assert_values_match` and `assert_golden_records_match`), and `proptest`
//! round-trip invariants.

use flexmesa_testing::{
    assert_golden_records_match, assert_values_match, extract_golden_table,
    parse_fortran_float, parse_golden_text,
};
use proptest::prelude::*;

const SAMPLE_MIXED: &str = "\
 CRLIBM
   Function Value (x=   0.1000000000000000E+00 )
        log  -0.2302585092994045E+01
 test2 T
  x val exact  1  0.0000000000000000D+00  0.0000000000000000D+00
  test2D_db  2  3  2.5132741228718345D+00  -2.4581494853809029D-02
";

const SAMPLE_TABLE: &str = "\
  6.0000000000000000E+00  1.0000000000000000E+00
  6.0000000000000000E+00  3.5000000000000000E+00
";

/// Verify parsing of standard `E`, Fortran `D`, and bare 3-digit exponents.
#[test]
fn test_parse_fortran_float_handles_d_e_and_bare_exponents() {
    let pi = parse_fortran_float("3.1415926535897931E+00").unwrap();
    assert!((pi - std::f64::consts::PI).abs() < 1e-15);

    let d_val = parse_fortran_float("-8.2630874248383526D-04").unwrap();
    assert!((d_val - (-8.263_087_424_838_353e-4)).abs() < 1e-19);

    let bare = parse_fortran_float("1.2500000000000000-102").unwrap();
    assert!((bare - 1.25e-102).abs() < 1e-117);

    assert!(parse_fortran_float("NaN").is_err());
    assert!(parse_fortran_float("1.0e500").is_err());
    assert!(parse_fortran_float("").is_err());
}

/// Verify structured record parsing, table extraction, and parity assertions.
#[test]
fn test_parse_and_compare_golden_records() {
    let expected = parse_golden_text(SAMPLE_MIXED).unwrap();
    assert_eq!(expected.len(), 3);
    assert_eq!(expected[0].label, "log");
    assert_eq!(expected[1].indices, vec![1]);
    assert_eq!(expected[2].indices, vec![2, 3]);

    let table_records = parse_golden_text(SAMPLE_TABLE).unwrap();
    let table = extract_golden_table(&table_records).unwrap();
    assert_eq!(table.len(), 2);
    assert_eq!(table[1], vec![6.0, 3.5]);

    let mut perturbed = expected.clone();
    for rec in &mut perturbed {
        for val in &mut rec.values {
            *val *= 1.0 + 1e-14;
        }
    }
    let summary =
        assert_golden_records_match(&perturbed, &expected, 1e-12, 1e-15)
            .unwrap();
    assert_eq!(summary.num_records, 3);
    assert_eq!(summary.num_values, 5);
    assert!(summary.max_rel_error <= 1e-12);

    perturbed[0].values[0] += 1e-5;
    assert!(
        assert_golden_records_match(&perturbed, &expected, 1e-12, 1e-15)
            .is_err()
    );
}

proptest! {
    /// Property test: arbitrary Fortran `D` floats round-trip within `1e-13`.
    #[test]
    fn prop_fortran_d_exponent_roundtrips(
        mantissa in -9.999_999_999_f64..9.999_999_999_f64,
        exponent in -200_i32..200_i32,
    ) {
        let formatted = format!("{mantissa:.14}D{exponent:+03}");
        let expected: f64 =
            format!("{mantissa:.14}e{exponent:+03}").parse().unwrap();
        let parsed = parse_fortran_float(&formatted).unwrap();
        let (_, rel_err) = assert_values_match(
            &[parsed],
            &[expected],
            1e-13,
            1e-300,
            "prop",
        )
        .unwrap();
        prop_assert!(rel_err <= 1e-13);
    }
}
