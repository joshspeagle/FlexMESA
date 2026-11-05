// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration and property-based tests for `flexmesa-testing`.
//!
//! Verifies:
//! 1. Fortran float parsing (`D`, `E`, and bare 3-digit exponents) and
//!    rejection of non-finite or malformed tokens (`NaN`, `Inf`, `1.0e500`).
//! 2. Structured [`GoldenRecord`] extraction across MESA `test_output`
//!    formats (section headers, separator lines, scalar rows, indexed rows,
//!    and dense tables).
//! 3. Dense 2D table extraction (`extract_golden_table`).
//! 4. Elementwise and record-sequence numerical parity checks
//!    (`assert_values_match` and `assert_golden_records_match`) at
//!    `rtol = 1e-12, atol = 1e-15`.
//! 5. Property-based round-trip invariants via `proptest`.

use flexmesa_testing::{
    assert_golden_records_match, assert_values_match, extract_golden_table,
    is_fortran_float_token, parse_fortran_float, parse_golden_text,
};
use proptest::prelude::*;

/// Representative multi-section MESA `test_output` fixture with `E` and `D`
/// scientific notation, separator lines, and 1D/2D integer indices.
const SAMPLE_MIXED: &str = "\
 ----------------------------------------------------------------
 CRLIBM
 ----------------------------------------------------------------
   Function Value (x=   0.1000000000000000E+00 )
        log  -0.2302585092994045E+01
 test2 T
  x val exact  1  0.0000000000000000D+00  0.0000000000000000D+00
  test2D_db  2  3  2.5132741228718345D+00  -2.4581494853809029D-02
";

/// Representative unlabeled dense numeric table fixture (`neu` pattern).
const SAMPLE_TABLE: &str = "\
  6.0000000000000000E+00  1.0000000000000000E+00
  6.0000000000000000E+00  3.5000000000000000E+00
";

/// Verify parsing of standard `E`, Fortran `D`, and bare 3-digit exponents,
/// and confirm that integer ranges (`1-100`) and non-finite strings fail.
#[test]
fn test_parse_fortran_float_handles_d_e_and_bare_exponents() {
    let pi = parse_fortran_float("3.1415926535897931E+00").unwrap();
    assert!((pi - std::f64::consts::PI).abs() < 1e-15);

    let d_val = parse_fortran_float("-8.2630874248383526D-04").unwrap();
    assert!((d_val - (-8.263_087_424_838_353e-4)).abs() < 1e-19);

    let bare = parse_fortran_float("1.2500000000000000-102").unwrap();
    assert!((bare - 1.25e-102).abs() < 1e-117);

    assert!(!is_fortran_float_token("1-100"));
    assert!(parse_fortran_float("NaN").is_err());
    assert!(parse_fortran_float("1.0e500").is_err());
    assert!(parse_fortran_float("").is_err());
    assert!(parse_golden_text("  pi  1.0D+500\n").is_err());
    assert!(parse_golden_text("  pi  NaN\n").is_err());
}

/// Verify structured record parsing, table extraction, and parity assertions,
/// including rejection of `NaN` tolerances and `1e-5` value perturbations.
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

    assert!(
        assert_values_match(&[1.0], &[2.0], f64::NAN, 1e-15, "nan").is_err()
    );

    perturbed[0].values[0] += 1e-5;
    assert!(
        assert_golden_records_match(&perturbed, &expected, 1e-12, 1e-15)
            .is_err()
    );
}

proptest! {
    /// Property test: arbitrary Fortran `D` floats round-trip within `1e-13`
    /// across mantissas in `[-9.999999999, 9.999999999]` and exponents in
    /// `[-200, 200]`.
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
