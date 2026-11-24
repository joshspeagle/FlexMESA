// SPDX-License-Identifier: GPL-3.0-or-later
//! Bit-exact IEEE-754 `f64` regression tests for `eval_charge_of_fe56_in_he4`
//! against MESA 26.4.1 (`ionization/test/src/test_ionization.f90`).

use flexmesa_ionization::eval_charge_of_fe56_in_he4;
use flexmesa_math::{with_numerics_mode, NumericsMode};

#[test]
fn test_fe56_he4_strict_bit_exact_hex() {
    let cases: [(f64, f64, u64); 3] = [
        (30.5, 7.9, 0x4039_e538_134a_ebcf),
        (26.0, 7.0, 0x4033_196d_5873_171a),
        (24.5, 6.1, 0x4026_76b5_5a05_ee15),
    ];

    with_numerics_mode(NumericsMode::Strict, || {
        let mut mismatches = Vec::new();
        for (log_ne, log_t, expected_bits) in cases {
            let got = eval_charge_of_fe56_in_he4(log_ne, log_t)
                .expect("eval_charge_of_fe56_in_he4 should succeed");
            let got_bits = got.to_bits();
            if got_bits != expected_bits {
                mismatches.push(format!(
                    "at ({log_ne}, {log_t}): got 0x{got_bits:016x} ({got:.17e}), \
                     expected 0x{expected_bits:016x} ({:.17e}), ulp={}",
                    f64::from_bits(expected_bits),
                    got_bits.abs_diff(expected_bits),
                ));
            }
        }
        assert!(
            mismatches.is_empty(),
            "Fe56-in-He4 strict hex mismatches:\n{}",
            mismatches.join("\n")
        );
    });
}
