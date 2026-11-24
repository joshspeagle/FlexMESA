// SPDX-License-Identifier: GPL-3.0-or-later
//! Bit-exact IEEE-754 `f64` regression tests for `eval_colors` against MESA
//! 26.4.1 (`colors/test/src/test_colors.f90`).

use flexmesa_colors::{
    builtin_sed_cube, MagSystem, SedInterpMethod, COLOR_COLUMN_NAMES,
    D_10PC_CM, NUM_COLORS_OUT,
};
use flexmesa_const::RSUN;
use flexmesa_math::{with_numerics_mode, NumericsMode};

#[test]
fn test_solar_eval_colors_strict_bit_exact_hex() {
    let expected: [u64; NUM_COLORS_OUT] = [
        0x4030_3bd9_4058_0275, // Mag_bol
        0x3e95_8a80_b289_f0bc, // Flux_bol
        0x3f88_9b5f_2f5a_6e1e, // Interp_rad
        0x4015_bb38_2edc_bf58, // B
        0x4010_18f2_ad65_6265, // I
        0x400d_3c59_0bb4_538a, // J
        0x400a_3c38_2e3e_8e42, // M
        0x4011_5437_5640_16fe, // R
        0x4016_231d_a7c6_9e08, // U
        0x4013_1ba0_f14c_7a97, // V
    ];

    let cube = builtin_sed_cube();
    with_numerics_mode(NumericsMode::Strict, || {
        let got = cube
            .eval_colors(
                5778.0,
                4.44,
                0.0,
                RSUN,
                D_10PC_CM,
                MagSystem::Vega,
                SedInterpMethod::Hermite,
            )
            .expect("eval_colors should succeed");

        let mut mismatches = Vec::new();
        for (idx, (&val, &exp_bits)) in
            got.iter().zip(expected.iter()).enumerate()
        {
            let got_bits = val.to_bits();
            if got_bits != exp_bits {
                mismatches.push(format!(
                    "{}: got 0x{got_bits:016x} ({val:.17e}), \
                     expected 0x{exp_bits:016x} ({:.17e}), ulp={}",
                    COLOR_COLUMN_NAMES[idx],
                    f64::from_bits(exp_bits),
                    got_bits.abs_diff(exp_bits),
                ));
            }
        }
        assert!(
            mismatches.is_empty(),
            "solar eval_colors strict hex mismatches:\n{}",
            mismatches.join("\n")
        );
    });
}
