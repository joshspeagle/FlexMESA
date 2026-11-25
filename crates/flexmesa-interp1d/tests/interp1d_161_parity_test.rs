// SPDX-License-Identifier: GPL-3.0-or-later
#![forbid(unsafe_code)]

//! Standalone MESA `26.4.1` `interp_1d` boundary-clamping parity tests (#161).
//!
//! Verifies that [`interp_value_and_slope`] and [`interp_values_and_slopes`]
//! match upstream `interp_1d/private/interp_1d_misc.f90:280-314`
//! (`do_interp_values_and_slopes`), clamping out-of-range queries to the
//! boundary knot while preserving the boundary cubic's first derivative.

use flexmesa_interp1d::{
    interp_m3q, interp_pm, interp_value_and_slope, interp_values_and_slopes,
    CubicCoeffs,
};

const X_ASC: [f64; 6] = [0.0, 0.4, 1.1, 2.3, 3.7, 5.0];
const X_DEC: [f64; 6] = [5.0, 3.7, 2.3, 1.1, 0.4, 0.0];
const Q_ASC: [f64; 5] = [-1.5, 0.0, 1.75, 5.0, 7.25];
const Q_DEC: [f64; 5] = [7.25, 5.0, 1.75, 0.0, -1.5];

// Pinned MESA 26.4.1 knot values for `sin(0.7 * x_asc(i)) + 0.1 * x_asc(i)**2`.
const Y_ASC_BITS: [u64; 6] = [
    0x0000_0000_0000_0000,
    0x3FD2_B5F4_7758_8F28,
    0x3FEA_25F8_CCCA_0972,
    0x3FF8_73A3_03A8_D70A,
    0x3FFE_49E8_DD16_CB94,
    0x4001_3198_9028_B3BC,
];

const PM_ASC_EXPECTED_BITS: [(u64, u64); 5] = [
    (0x0000_0000_0000_0000, 0x3FE7_2B73_BBE3_C12E),
    (0x0000_0000_0000_0000, 0x3FE7_2B73_BBE3_C12E),
    (0x3FF3_D972_E586_9C3C, 0x3FE3_0C8C_4ED7_F653),
    (0x4001_3198_9028_B3BC, 0x3FC5_4EE1_1BED_5397),
    (0x4001_3198_9028_B3BC, 0x3FC5_4EE1_1BED_5397),
];

const PM_DEC_EXPECTED_BITS: [(u64, u64); 5] = [
    (0x4001_3198_9028_B3BC, 0x3FC5_4EE1_1BED_5396),
    (0x4001_3198_9028_B3BC, 0x3FC5_4EE1_1BED_5396),
    (0x3FF3_D972_E586_9C3C, 0x3FE3_0C8C_4ED7_F653),
    (0xBC90_0000_0000_0000, 0x3FE7_2B73_BBE3_C12F),
    (0xBC90_0000_0000_0000, 0x3FE7_2B73_BBE3_C12F),
];

const M3Q_ASC_EXPECTED_BITS: [(u64, u64); 5] = [
    (0x0000_0000_0000_0000, 0x3FE7_6371_952E_B2F2),
    (0x0000_0000_0000_0000, 0x3FE7_6371_952E_B2F2),
    (0x3FF3_ED09_3B40_9BE2, 0x3FE2_D0ED_DE66_8479),
    (0x4001_3198_9028_B3BC, 0x3FC9_391F_2790_0E21),
    (0x4001_3198_9028_B3BC, 0x3FC9_391F_2790_0E21),
];

fn y_asc_values() -> Vec<f64> {
    Y_ASC_BITS.iter().copied().map(f64::from_bits).collect()
}

fn assert_bits_eq(label: &str, idx: usize, actual: f64, expected_bits: u64) {
    assert_eq!(
        actual.to_bits(),
        expected_bits,
        "{label}[{idx}] bit mismatch: got {:#018X} ({actual:?}), \
         expected {expected_bits:#018X}",
        actual.to_bits(),
    );
}

#[test]
fn test_interp_pm_and_m3q_ascending_exterior_clamping_zero_ulp_mesa_26_4_1() {
    let y_asc = y_asc_values();

    let pm_coeffs = interp_pm(&X_ASC, &y_asc).expect("interp_pm ascending");
    let (pm_vals, pm_slopes) =
        interp_values_and_slopes(&X_ASC, &pm_coeffs, &Q_ASC)
            .expect("interp_values_and_slopes pm ascending");
    for (idx, &(exp_val, exp_slope)) in PM_ASC_EXPECTED_BITS.iter().enumerate()
    {
        assert_bits_eq("pm_asc_val", idx, pm_vals[idx], exp_val);
        assert_bits_eq("pm_asc_slope", idx, pm_slopes[idx], exp_slope);
    }

    let (left_val, left_slope) =
        interp_value_and_slope(&X_ASC, &pm_coeffs, -1.5)
            .expect("scalar left exterior");
    assert_bits_eq(
        "pm_scalar_left_val",
        0,
        left_val,
        PM_ASC_EXPECTED_BITS[0].0,
    );
    assert_bits_eq(
        "pm_scalar_left_slope",
        0,
        left_slope,
        PM_ASC_EXPECTED_BITS[0].1,
    );

    let (right_val, right_slope) =
        interp_value_and_slope(&X_ASC, &pm_coeffs, 7.25)
            .expect("scalar right exterior");
    assert_bits_eq(
        "pm_scalar_right_val",
        0,
        right_val,
        PM_ASC_EXPECTED_BITS[4].0,
    );
    assert_bits_eq(
        "pm_scalar_right_slope",
        0,
        right_slope,
        PM_ASC_EXPECTED_BITS[4].1,
    );

    let m3q_coeffs = interp_m3q(&X_ASC, &y_asc).expect("interp_m3q ascending");
    let (m3q_vals, m3q_slopes) =
        interp_values_and_slopes(&X_ASC, &m3q_coeffs, &Q_ASC)
            .expect("interp_values_and_slopes m3q ascending");
    for (idx, &(exp_val, exp_slope)) in M3Q_ASC_EXPECTED_BITS.iter().enumerate()
    {
        assert_bits_eq("m3q_asc_val", idx, m3q_vals[idx], exp_val);
        assert_bits_eq("m3q_asc_slope", idx, m3q_slopes[idx], exp_slope);
    }
}

#[test]
fn test_interp_pm_descending_exterior_clamping_zero_ulp_mesa_26_4_1() {
    let y_dec: Vec<f64> = y_asc_values().into_iter().rev().collect();
    let coeffs = interp_pm(&X_DEC, &y_dec).expect("interp_pm descending");
    let (vals, slopes) = interp_values_and_slopes(&X_DEC, &coeffs, &Q_DEC)
        .expect("interp_values_and_slopes pm descending");
    for (idx, &(exp_val, exp_slope)) in PM_DEC_EXPECTED_BITS.iter().enumerate()
    {
        assert_bits_eq("pm_dec_val", idx, vals[idx], exp_val);
        assert_bits_eq("pm_dec_slope", idx, slopes[idx], exp_slope);
    }

    for (idx, &q) in Q_DEC.iter().enumerate() {
        let (v, s) = interp_value_and_slope(&X_DEC, &coeffs, q)
            .expect("scalar descending query");
        assert_bits_eq(
            "pm_dec_scalar_val",
            idx,
            v,
            PM_DEC_EXPECTED_BITS[idx].0,
        );
        assert_bits_eq(
            "pm_dec_scalar_slope",
            idx,
            s,
            PM_DEC_EXPECTED_BITS[idx].1,
        );
    }
}

#[test]
fn test_interp_nx1_returns_constant_value_and_zero_slope_mesa_26_4_1() {
    let x = [2.5_f64];
    let coeffs: [CubicCoeffs; 1] = [[9.25, 1.0, 2.0, 3.0]];
    let queries = [1.0_f64, 2.5, 4.0];
    let expected_val_bits = 0x4022_8000_0000_0000_u64;
    let expected_slope_bits = 0x0000_0000_0000_0000_u64;

    let (vals, slopes) = interp_values_and_slopes(&x, &coeffs, &queries)
        .expect("batch nx=1 interpolation");
    for (idx, (&v, &s)) in vals.iter().zip(slopes.iter()).enumerate() {
        assert_bits_eq("nx1_batch_val", idx, v, expected_val_bits);
        assert_bits_eq("nx1_batch_slope", idx, s, expected_slope_bits);
    }

    for (idx, &q) in queries.iter().enumerate() {
        let (v, s) = interp_value_and_slope(&x, &coeffs, q)
            .expect("scalar nx=1 interpolation");
        assert_bits_eq("nx1_scalar_val", idx, v, expected_val_bits);
        assert_bits_eq("nx1_scalar_slope", idx, s, expected_slope_bits);
    }
}
