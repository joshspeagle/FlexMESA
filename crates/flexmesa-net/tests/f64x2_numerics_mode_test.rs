// SPDX-License-Identifier: LGPL-3.0-or-later
//! `F64x2::two_prod` near `f64::MAX` in both numerics modes.
//!
//! This test switches the process-wide numerics mode, so it lives in its own
//! test binary. In a shared binary the switch races with other tests that
//! read the mode (issue #210).

use flexmesa_math::{with_numerics_mode, NumericsMode};
use flexmesa_net::F64x2;

#[test]
fn test_two_prod_near_f64_max_avoids_veltkamp_intermediate_overflow() {
    for mode in [NumericsMode::Strict, NumericsMode::Fast] {
        with_numerics_mode(mode, || {
            // Case 1: `f64::MAX * 0.5` is exact and finite (`hi == f64::MAX * 0.5, lo == 0.0`).
            let half_max = F64x2::from_f64(f64::MAX) * 0.5;
            assert_eq!(half_max.hi, f64::MAX * 0.5);
            assert_eq!(half_max.lo, 0.0);

            // Case 2: Both operands below `SPLIT_HI_THRESH = 2^996`, yet `ah * bh`
            // overflows `2^1024` unless `two_prod` rescales near `f64::MAX`.
            let u = f64::from_bits(0x5fef_ffff_e000_0000);
            let v = f64::from_bits(0x5fe0_0000_0000_0000);
            let prod = F64x2::from_f64(u) * v;
            assert!(prod.hi.is_finite(), "mode={mode:?} hi={}", prod.hi);
            assert!(prod.lo.is_finite(), "mode={mode:?} lo={}", prod.lo);
            assert_eq!(prod.hi, u * v);
            assert_eq!(prod.lo, 0.0);

            // Case 3: Non-zero low compensation near `2^1022` matches scaled product.
            let a =
                (1.0 + f64::EPSILON) * f64::from_bits(0x5fd0_0000_0000_0000);
            let b = (1.0 + 2.0 * f64::EPSILON)
                * f64::from_bits(0x5ff0_0000_0000_0000);
            let ab = F64x2::from_f64(a) * b;
            let scale_down = f64::from_bits(0x3cb0_0000_0000_0000); // 2^-52
            let scale_up = f64::from_bits(0x4330_0000_0000_0000); // 2^52
            let ref_ab = F64x2::from_f64(a * scale_down) * b;
            assert_eq!(ab.hi, ref_ab.hi * scale_up);
            assert_eq!(ab.lo, ref_ab.lo * scale_up);
        });
    }
}
