// SPDX-License-Identifier: GPL-3.0-or-later
//! Tests of [`flexmesa_math::Quad`] against exact rational arithmetic.
//!
//! `tests/data/binary128_vectors.tsv` is produced by
//! `tests/data/gen_binary128_vectors.py` with Python `fractions.Fraction`
//! and one final rounding per operation. It does not come from `FlexMESA` or
//! from MESA.

use flexmesa_math::Quad;
use proptest::prelude::*;

const VECTORS: &str = include_str!("data/binary128_vectors.tsv");

fn parse_quad(s: &str) -> Quad {
    Quad::from_bits(u128::from_str_radix(s, 16).expect("binary128 hex"))
}

fn check_quad(line: &str, got: Quad, want: &str) -> bool {
    let ok = if want == "nan" {
        got.is_nan()
    } else {
        got.to_bits() == parse_quad(want).to_bits()
    };
    if !ok {
        eprintln!("{line}\n  got {:032x}", got.to_bits());
    }
    ok
}

#[test]
fn test_quad_matches_exact_rational_binary128_vectors() {
    let (mut n, mut bad) = (0_usize, 0_usize);
    let mut per_op = std::collections::BTreeMap::<&str, usize>::new();
    for line in VECTORS.lines().filter(|l| !l.starts_with('#')) {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 4, "malformed vector line: {line}");
        let (op, a, b, want) = (cols[0], cols[1], cols[2], cols[3]);
        n += 1;
        *per_op.entry(op).or_default() += 1;
        let ok = match op {
            "add" => check_quad(line, parse_quad(a) + parse_quad(b), want),
            "sub" => check_quad(line, parse_quad(a) - parse_quad(b), want),
            "mul" => check_quad(line, parse_quad(a) * parse_quad(b), want),
            "div" => check_quad(line, parse_quad(a) / parse_quad(b), want),
            "powi" => check_quad(
                line,
                parse_quad(a).powi(b.parse().expect("exponent")),
                want,
            ),
            "from_f64" => {
                let x = f64::from_bits(u64::from_str_radix(a, 16).unwrap());
                check_quad(line, Quad::from_f64(x), want)
            }
            "to_f64" => {
                let got = parse_quad(a).to_f64();
                let ok = if want == "nan" {
                    got.is_nan()
                } else {
                    got.to_bits() == u64::from_str_radix(want, 16).unwrap()
                };
                if !ok {
                    eprintln!("{line}\n  got {:016x}", got.to_bits());
                }
                ok
            }
            other => panic!("unknown op {other}"),
        };
        bad += usize::from(!ok);
    }
    assert_eq!(bad, 0, "{bad} of {n} binary128 vectors disagree");
    // The file must not be silently truncated or emptied.
    assert_eq!(n, 5055);
    for op in ["add", "sub", "mul", "div", "powi", "from_f64", "to_f64"] {
        assert!(per_op.get(op).copied().unwrap_or(0) >= 100, "{op}");
    }
}

#[test]
fn test_quad_comparisons_follow_ieee_754() {
    use std::cmp::Ordering::{Equal, Greater, Less};
    let (one, two) = (Quad::from(1_i32), Quad::from(2.0_f64));
    assert_eq!(one.partial_cmp(&two), Some(Less));
    assert_eq!(two.partial_cmp(&one), Some(Greater));
    assert_eq!(one.partial_cmp(&Quad::ONE), Some(Equal));
    assert_eq!((-two).partial_cmp(&-one), Some(Less));
    assert_eq!((-one).partial_cmp(&one), Some(Less));
    assert_eq!(Quad::ZERO.partial_cmp(&-Quad::ZERO), Some(Equal));
    assert_eq!(Quad::ZERO, -Quad::ZERO);
    assert_eq!(Quad::INFINITY.partial_cmp(&Quad::from(f64::MAX)), Some(Greater));
    assert_eq!((-Quad::INFINITY).partial_cmp(&Quad::from(f64::MIN)), Some(Less));
    let nan = Quad::NAN;
    assert_eq!(nan.partial_cmp(&nan), None);
    assert_eq!(nan.partial_cmp(&one), None);
    assert_eq!(one.partial_cmp(&nan), None);
    assert_ne!(nan, nan);
    assert!(nan.is_nan() && !nan.is_finite() && !nan.is_infinite());
    assert!(Quad::INFINITY.is_infinite() && one.is_finite());
    assert_eq!((-two).abs(), two);
    assert!((-two).is_sign_negative());
    // A subnormal binary128 value is smaller than every positive `f64`.
    let tiny = Quad::from_bits(1);
    assert_eq!(tiny.partial_cmp(&Quad::ZERO), Some(Greater));
    assert_eq!(tiny.partial_cmp(&Quad::from(f64::from_bits(1))), Some(Less));
}

#[test]
fn test_quad_integer_conversions_are_exact() {
    for i in [0_i64, 1, -1, 2, 3, -7, 1 << 53, (1 << 62) + 1, i64::MAX, i64::MIN]
    {
        // Every `i64` is exactly representable in 113 bits. Its high and low
        // 32-bit halves are exact in `f64`, so their binary128 sum must equal
        // the direct conversion.
        let hi = f64::from(i32::try_from(i >> 32).unwrap()) * 4_294_967_296.0;
        let lo = f64::from(u32::try_from(i & 0xffff_ffff).unwrap());
        assert_eq!(Quad::from(i), Quad::from(hi) + Quad::from(lo), "i = {i}");
    }
    assert_eq!(Quad::from(i32::MIN).to_f64(), f64::from(i32::MIN));
}

fn finite_f64() -> impl Strategy<Value = f64> {
    any::<u64>()
        .prop_map(f64::from_bits)
        .prop_filter("finite", |x| x.is_finite())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// `f64 -> binary128 -> f64` is the identity.
    #[test]
    fn prop_quad_f64_round_trip(x in finite_f64()) {
        prop_assert_eq!(Quad::from(x).to_f64().to_bits(), x.to_bits());
    }

    /// The product of two `f64` values has at most 106 significant bits, so
    /// it is exact in binary128 (away from the binary128 subnormal range,
    /// which no `f64` product reaches) and rounding it once to `f64` must
    /// reproduce the IEEE-754 `f64` product.
    #[test]
    fn prop_quad_product_of_f64_rounds_to_f64_product(
        a in finite_f64(), b in finite_f64()
    ) {
        let q = Quad::from(a) * Quad::from(b);
        prop_assert_eq!(q.to_f64().to_bits(), (a * b).to_bits());
    }

    /// The same holds for a sum when it is exact in binary128, that is when
    /// the operands' exponents differ by less than 60.
    #[test]
    fn prop_quad_sum_of_f64_rounds_to_f64_sum(
        a in finite_f64(), shift in -59_i32..60, m in any::<u64>(),
        negate in any::<bool>()
    ) {
        let ea = ((a.to_bits() >> 52) & 0x7ff) as i32;
        let eb = u64::try_from((ea + shift).clamp(1, 0x7fe)).unwrap();
        let b = f64::from_bits(
            (u64::from(negate) << 63) | (eb << 52) | (m & ((1 << 52) - 1)),
        );
        prop_assume!(ea >= 1);
        let q = Quad::from(a) + Quad::from(b);
        prop_assert_eq!(q.to_f64().to_bits(), (a + b).to_bits());
        let d = Quad::from(a) - Quad::from(b);
        prop_assert_eq!(d.to_f64().to_bits(), (a - b).to_bits());
    }

    /// Algebraic identities that hold exactly under correct rounding.
    #[test]
    fn prop_quad_exact_identities(a in any::<u128>(), b in any::<u128>()) {
        let (x, y) = (Quad::from_bits(a), Quad::from_bits(b));
        prop_assume!(x.is_finite() && y.is_finite());
        prop_assert_eq!((x + y).to_bits(), (y + x).to_bits());
        prop_assert_eq!((x * y).to_bits(), (y * x).to_bits());
        prop_assert_eq!((x - y).to_bits(), (x + (-y)).to_bits());
        prop_assert_eq!((x * Quad::ONE).to_bits(), x.to_bits());
        prop_assert_eq!((x / Quad::ONE).to_bits(), x.to_bits());
        prop_assert_eq!(x.powi(2).to_bits(), (x * x).to_bits());
        prop_assert_eq!(x.powi(3).to_bits(), (x * (x * x)).to_bits());
    }
}

#[test]
fn test_quad_mixed_f64_operands_are_promoted_exactly() {
    let q = Quad::from(3.5_f64) / Quad::from(7.0_f64);
    for x in [0.1_f64, -2.5e300, 4.9e-324, 1.0, 0.0] {
        let p = Quad::from(x);
        assert_eq!((q + x).to_bits(), (q + p).to_bits());
        assert_eq!((q - x).to_bits(), (q - p).to_bits());
        assert_eq!((q * x).to_bits(), (q * p).to_bits());
        let mut r = q;
        r += x;
        r -= x;
        r *= x;
        assert_eq!(r.to_bits(), (((q + p) - p) * p).to_bits());
        if x != 0.0 {
            assert_eq!((q / x).to_bits(), (q / p).to_bits());
            let mut d = q;
            d /= x;
            assert_eq!(d.to_bits(), (q / p).to_bits());
        }
    }
}
