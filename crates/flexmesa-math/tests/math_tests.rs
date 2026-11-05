// SPDX-License-Identifier: LGPL-3.0-or-later
//! Golden parity and property-based tests for `flexmesa-math`.
//!
//! Verifies all 33 mathematical functions in upstream MESA's
//! `math/test/test_output.INTRINSIC` at `x = 0.1`, `0.2`, and `0.4` to
//! relative/absolute error `<= 1e-15`, along with domain-safety guards,
//! Euclidean `hypot`, and randomized `proptest` algebraic identities.

#![allow(
    clippy::approx_constant,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_math::{
    SAFE_LOG_BAD_RETURN, acos, acosh, acospi, asin, asinh, asinpi, atan, atanh,
    atanpi, cos, cosh, cospi, exp, exp10, expm1, heaviside, hypot, log, log10,
    log1p, log2, pow, pow2, pow3, pow4, pow5, pow6, pow7, pow8, powi, powm1,
    safe_log, safe_log10, safe_sqrt, sgn, sin, sinh, sinpi, str_to_double,
    str_to_vector, tan, tanh, tanpi,
};
use flexmesa_testing::assert_values_match;
use proptest::prelude::*;

/// Standard evaluation grid used in upstream MESA's `math/test`.
const XS: [f64; 3] = [0.1, 0.2, 0.4];

/// Relative and absolute tolerance for golden intrinsic comparisons.
const TOL: f64 = 1.0e-15;

/// Assert elementwise agreement within `rtol` and `atol`.
fn assert_allclose(actual: &[f64], expected: &[f64], rtol: f64, atol: f64) {
    assert_values_match(actual, expected, rtol, atol, "math_tests")
        .expect("math_tests parity mismatch");
}

/// Evaluate a scalar function at the three standard grid points `XS`.
fn eval3(f: fn(f64) -> f64) -> [f64; 3] {
    [f(XS[0]), f(XS[1]), f(XS[2])]
}

/// Verify logarithms (`log`, `log10`, `log1p`, `log2`, `safe_log`,
/// `safe_log10`) and exponentials (`exp`, `exp10`, `expm1`) against
/// `test_output.INTRINSIC`.
#[test]
fn test_math_intrinsic_golden_logs_and_exps_match_mesa_reference() {
    // Natural and base-10/base-2 logarithms at XS = [0.1, 0.2, 0.4].
    assert_allclose(
        &eval3(log),
        &[
            -2.302_585_092_994_045_5,
            -1.609_437_912_434_100_3,
            -9.162_907_318_741_551_1e-1,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(log10),
        &[
            -9.999_999_999_999_998_9e-1,
            -6.989_700_043_360_187_5e-1,
            -3.979_400_086_720_376_1e-1,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(log1p),
        &[
            9.531_017_980_432_486_9e-2,
            1.823_215_567_939_546_0e-1,
            3.364_722_366_212_129_5e-1,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(log2),
        &[
            -3.321_928_094_887_362_2,
            -2.321_928_094_887_362_2,
            -1.321_928_094_887_362_2,
        ],
        TOL,
        TOL,
    );
    // Domain-safe logarithms agree with exact logarithms on positive inputs.
    assert_allclose(&eval3(safe_log), &eval3(log), TOL, TOL);
    assert_allclose(&eval3(safe_log10), &eval3(log10), TOL, TOL);
    // Natural, base-10, and shifted exponentials at XS = [0.1, 0.2, 0.4].
    assert_allclose(
        &eval3(exp),
        &[
            1.105_170_918_075_647_7,
            1.221_402_758_160_169_9,
            1.491_824_697_641_270_3,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(exp10),
        &[
            1.258_925_411_794_167_3,
            1.584_893_192_461_113_6,
            2.511_886_431_509_580_1,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(expm1),
        &[
            1.051_709_180_756_477_1e-1,
            2.214_027_581_601_698_5e-1,
            4.918_246_976_412_703_5e-1,
        ],
        TOL,
        TOL,
    );
}

/// Verify integer and floating-point powers (`pow`, `powi`, `pow2`..`pow8`,
/// `powm1`, and `hypot`) against `test_output.INTRINSIC`.
#[test]
fn test_math_intrinsic_golden_powers_match_mesa_reference() {
    let p2 = [1.0e-2, 4.0e-2, 1.6e-1];

    // Verify powi, pow, and pow2..pow8 on XS = [0.1, 0.2, 0.4].
    assert_allclose(
        &[powi(XS[0], 2), powi(XS[1], 2), powi(XS[2], 2)],
        &p2,
        TOL,
        TOL,
    );
    assert_allclose(
        &[pow(XS[0], 2.0), pow(XS[1], 2.0), pow(XS[2], 2.0)],
        &p2,
        TOL,
        TOL,
    );
    assert_allclose(&eval3(pow2), &p2, TOL, TOL);
    assert_allclose(&eval3(pow3), &[1.0e-3, 8.0e-3, 6.4e-2], TOL, TOL);
    assert_allclose(&eval3(pow4), &[1.0e-4, 1.6e-3, 2.56e-2], TOL, TOL);
    assert_allclose(&eval3(pow5), &[1.0e-5, 3.2e-4, 1.024e-2], TOL, TOL);
    assert_allclose(&eval3(pow6), &[1.0e-6, 6.4e-5, 4.096e-3], TOL, TOL);
    assert_allclose(&eval3(pow7), &[1.0e-7, 1.28e-5, 1.6384e-3], TOL, TOL);
    assert_allclose(&eval3(pow8), &[1.0e-8, 2.56e-6, 6.5536e-4], TOL, TOL);
    assert_allclose(&eval3(powm1), &[10.0, 5.0, 2.5], TOL, TOL);
    assert_allclose(&[hypot(3.0, 4.0)], &[5.0], TOL, TOL);
}

/// Verify radian, half-turn, inverse trigonometric, and hyperbolic functions
/// against `test_output.INTRINSIC`.
#[test]
fn test_math_intrinsic_golden_trig_and_hyperbolic_match_mesa_reference() {
    // Radian and half-turn direct trigonometric functions.
    assert_allclose(
        &eval3(cos),
        &[
            9.950_041_652_780_258_2e-1,
            9.800_665_778_412_416_3e-1,
            9.210_609_940_028_851_0e-1,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(sin),
        &[
            9.983_341_664_682_815_5e-2,
            1.986_693_307_950_612_2e-1,
            3.894_183_423_086_505_2e-1,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(tan),
        &[
            1.003_346_720_854_505_5e-1,
            2.027_100_355_086_724_8e-1,
            4.227_932_187_381_617_8e-1,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(cospi),
        &[
            9.510_565_162_951_535_3e-1,
            8.090_169_943_749_474_5e-1,
            3.090_169_943_749_474_5e-1,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(sinpi),
        &[
            3.090_169_943_749_474_0e-1,
            5.877_852_522_924_731_4e-1,
            9.510_565_162_951_535_3e-1,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(tanpi),
        &[
            3.249_196_962_329_062_9e-1,
            7.265_425_280_053_608_9e-1,
            3.077_683_537_175_253_1,
        ],
        TOL,
        TOL,
    );
    // Inverse trigonometric functions in radians and half-turns.
    assert_allclose(
        &eval3(acos),
        &[
            1.470_628_905_633_336_8,
            1.369_438_406_004_566_0,
            1.159_279_480_727_408_5,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(asin),
        &[
            1.001_674_211_615_598_0e-1,
            2.013_579_207_903_308_0e-1,
            4.115_168_460_674_880_1e-1,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(atan),
        &[
            9.966_865_249_116_203_8e-2,
            1.973_955_598_498_807_8e-1,
            3.805_063_771_123_649_0e-1,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(acospi),
        &[0.4681157195707401, 0.4359057831510251, 0.3690101195655454],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(asinpi),
        &[0.3188428042925993e-1, 0.6409421684897494e-1, 0.1309898804344546],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(atanpi),
        &[0.3172551743055357e-1, 0.6283295818900120e-1, 0.1211189415908434],
        TOL,
        TOL,
    );
    // Hyperbolic functions at XS = [0.1, 0.2, 0.4].
    assert_allclose(
        &eval3(cosh),
        &[
            1.005_004_168_055_803_5,
            1.020_066_755_619_076_0,
            1.081_072_371_838_454_9,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(sinh),
        &[
            1.001_667_500_198_440_3e-1,
            2.013_360_025_410_940_2e-1,
            4.107_523_258_028_155_1e-1,
        ],
        TOL,
        TOL,
    );
    assert_allclose(
        &eval3(tanh),
        &[
            9.966_799_462_495_581_7e-2,
            1.973_753_202_249_040_1e-1,
            3.799_489_622_552_248_9e-1,
        ],
        TOL,
        TOL,
    );
}

/// Verify domain-safe clamping (`safe_sqrt`, `safe_log`, `safe_log10`), sign
/// and step helpers (`sgn`, `heaviside`), and Fortran number parsers.
#[test]
fn test_safe_guards_and_io_helpers_handle_edge_cases() {
    // Domain-safe clamping on negative, zero, and non-finite inputs.
    assert_eq!(safe_sqrt(-10.0), 0.0);
    assert!(safe_sqrt(f64::NAN).is_nan());
    assert_eq!(safe_log(f64::NAN), SAFE_LOG_BAD_RETURN);
    assert_eq!(safe_log(f64::INFINITY), SAFE_LOG_BAD_RETURN);
    assert_eq!(safe_log10(f64::NAN), SAFE_LOG_BAD_RETURN);
    assert_eq!(safe_log10(0.0), SAFE_LOG_BAD_RETURN);

    // Three-valued signum and symmetric Heaviside step functions.
    assert_eq!(sgn(-4.2), -1.0);
    assert_eq!(sgn(0.0), 0.0);
    assert_eq!(sgn(4.2), 1.0);
    assert!(sgn(f64::NAN).is_nan());
    assert_eq!(heaviside(-0.1), 0.0);
    assert_eq!(heaviside(0.0), 0.5);
    assert_eq!(heaviside(0.1), 1.0);
    assert!(heaviside(f64::NAN).is_nan());

    // Fortran D/d exponent string parsing.
    assert_eq!(str_to_double("1.25D-2").unwrap(), 0.0125);
    assert_eq!(
        str_to_vector(" 1.0d0  -2.5D1  3.0e2 ").unwrap(),
        vec![1.0, -25.0, 300.0]
    );
}

proptest! {
    /// Verify Pythagorean trigonometric and hyperbolic identities across
    /// `[-10, 10]`.
    #[test]
    fn test_trig_and_hyperbolic_identities_hold_across_domain(
        x in -10.0_f64..10.0_f64
    ) {
        let s2c2 = pow2(sin(x)) + pow2(cos(x));
        prop_assert!((s2c2 - 1.0).abs() < 1.0e-14);

        let spi2cpi2 = pow2(sinpi(x)) + pow2(cospi(x));
        prop_assert!((spi2cpi2 - 1.0).abs() < 1.0e-13);

        let x_small = x * 0.25;
        let hyp = pow2(cosh(x_small)) - pow2(sinh(x_small));
        prop_assert!((hyp - 1.0).abs() < 1.0e-13);
        prop_assert!((asinh(sinh(x_small)) - x_small).abs() < 1.0e-14);
        let x_pos = x_small.abs() + 0.1;
        prop_assert!((acosh(cosh(x_pos)) - x_pos).abs() < 1.0e-13);
    }

    /// Verify inverse hyperbolic, half-turn complementary angle, and log/exp
    /// round-trip identities.
    #[test]
    fn test_log_exp_roundtrip_and_inverse_trig_hold(
        u in -0.95_f64..0.95_f64
    ) {
        prop_assert!((atanh(tanh(u)) - u).abs() < 1.0e-14);
        prop_assert!((asinpi(u) + acospi(u) - 0.5).abs() < 1.0e-15);
        let pos = u.exp();
        prop_assert!((exp10(safe_log10(pos)) - pos).abs() / pos < 1.0e-14);
    }
}
