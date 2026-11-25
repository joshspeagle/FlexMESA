// SPDX-License-Identifier: GPL-3.0-or-later
#![forbid(unsafe_code)]

//! Standalone MESA `26.4.1` `num` ODE solver parity tests (#161).
//!
//! Verifies that [`dopri5`], [`dop853`], [`rosenbrock_dense`], and
//! [`rosenbrock_dae_diag`] match upstream `num/private/mod_dopri5.f`,
//! `num/private/mod_dop853.f`, and `num/private/mod_rosenbrock.f` across
//! automatic initial step sizing (`h0 = 0`), trial-stage callback error step
//! halving, step-size underflow exhaustion, tiny-span `last` step
//! termination, and Rosenbrock LU singularity exhaustion.

use flexmesa_num::{
    dop853, dopri5, rosenbrock_dae_diag, rosenbrock_dense, NumError,
    RosenbrockVariant,
};

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
fn test_dopri5_h0_zero_ascending_and_descending_zero_ulp_mesa_26_4_1() {
    let osc = |_x: f64, y: &[f64]| Ok(vec![y[1], -y[0]]);

    let asc = dopri5(
        0.0,
        2.0,
        &[1.0, 0.0],
        0.0,
        0.0,
        1.0e-8,
        1.0e-8,
        100_000,
        osc,
    )
    .expect("dopri5 ascending h0=0");
    assert_eq!(asc.nfcn, 134);
    assert_eq!(asc.nstep, 22);
    assert_eq!(asc.naccpt, 22);
    assert_eq!(asc.nrejct, 0);
    assert_bits_eq("dopri5_asc", 0, asc.y[0], 0xBFDA_A226_55E2_A742);
    assert_bits_eq("dopri5_asc", 1, asc.y[1], 0xBFED_18F6_E812_6747);

    let y0_dec = [
        f64::from_bits(0xBFDA_A226_5753_7204),
        f64::from_bits(0xBFED_18F6_EAD1_B445),
    ];
    let dec = dopri5(2.0, 0.0, &y0_dec, 0.0, 0.0, 1.0e-8, 1.0e-8, 100_000, osc)
        .expect("dopri5 descending h0=0");
    assert_eq!(dec.nfcn, 134);
    assert_eq!(dec.nstep, 22);
    assert_eq!(dec.naccpt, 22);
    assert_eq!(dec.nrejct, 0);
    assert_bits_eq("dopri5_dec", 0, dec.y[0], 0x3FEF_FFFF_FD38_0A24);
    assert_bits_eq("dopri5_dec", 1, dec.y[1], 0x3E0F_19FA_3000_0000);
}

#[test]
fn test_dopri5_tiny_interval_uses_last_step_flag_zero_ulp_mesa_26_4_1() {
    let tiny = dopri5(
        0.0,
        5.0e-16,
        &[0.0],
        5.0e-16,
        0.0,
        1.0e-8,
        1.0e-8,
        100_000,
        |_x, y| Ok(vec![1.0e16 * (1.0 + 0.1 * y[0])]),
    )
    .expect("dopri5 tiny interval");
    assert_eq!(tiny.nstep, 11);
    assert_eq!(tiny.naccpt, 8);
    assert_eq!(tiny.nrejct, 0);
    assert_bits_eq("dopri5_tiny", 0, tiny.y[0], 0x4019_F2E7_E375_78CC);
}

#[test]
fn test_dopri5_trial_stage_error_recovery_and_underflow_mesa_26_4_1() {
    let barrier = dopri5(
        0.0,
        2.0,
        &[0.5],
        1.0,
        1.0,
        1.0e-8,
        1.0e-8,
        100_000,
        |_x, y| {
            if y[0] > 1.02 {
                Err(NumError::NonFiniteValue("barrier"))
            } else {
                Ok(vec![4.0 * y[0] * (1.0 - y[0])])
            }
        },
    )
    .expect("dopri5 barrier recovery");
    assert_eq!(barrier.nstep, 36);
    assert_eq!(barrier.naccpt, 33);
    assert_eq!(barrier.nrejct, 1);
    assert_bits_eq("dopri5_barrier", 0, barrier.y[0], 0x3FEF_FD40_B713_B407);

    let underflow = dopri5(
        0.0,
        1.0,
        &[0.0],
        0.8,
        0.8,
        1.0e-6,
        1.0e-6,
        100_000,
        |x, y| {
            if y[0] > 0.2 && x < 0.5 {
                Err(NumError::NonFiniteValue("barrier"))
            } else {
                Ok(vec![2.0 * (1.0 + y[0] * y[0])])
            }
        },
    );
    assert_eq!(
        underflow.unwrap_err(),
        NumError::ConvergenceFailed("step size underflow in dopri5"),
    );
}

#[test]
fn test_dop853_h0_zero_barrier_recovery_and_underflow_mesa_26_4_1() {
    let osc = |_x: f64, y: &[f64]| Ok(vec![y[1], -y[0]]);
    let asc = dop853(
        0.0,
        2.0,
        &[1.0, 0.0],
        0.0,
        0.0,
        1.0e-10,
        1.0e-12,
        100_000,
        osc,
    )
    .expect("dop853 ascending h0=0");
    assert_eq!(asc.nfcn, 98);
    assert_eq!(asc.nstep, 8);
    assert_eq!(asc.naccpt, 8);
    assert_eq!(asc.nrejct, 0);
    assert_bits_eq("dop853_asc", 0, asc.y[0], 0xBFDA_A226_574F_8EBC);
    assert_bits_eq("dop853_asc", 1, asc.y[1], 0xBFED_18F6_EAD2_55E8);

    let barrier = dop853(
        0.0,
        2.0,
        &[0.5],
        1.0,
        1.0,
        1.0e-8,
        1.0e-8,
        100_000,
        |_x, y| {
            if y[0] > 1.02 {
                Err(NumError::NonFiniteValue("barrier"))
            } else {
                Ok(vec![4.0 * y[0] * (1.0 - y[0])])
            }
        },
    )
    .expect("dop853 barrier recovery");
    assert_eq!(barrier.nstep, 14);
    assert_eq!(barrier.naccpt, 10);
    assert_eq!(barrier.nrejct, 2);
    assert_bits_eq("dop853_barrier", 0, barrier.y[0], 0x3FEF_FD40_B829_3461);

    let underflow = dop853(
        0.0,
        1.0,
        &[0.0],
        0.8,
        0.8,
        1.0e-6,
        1.0e-6,
        100_000,
        |x, y| {
            if y[0] > 0.2 && x < 0.5 {
                Err(NumError::NonFiniteValue("barrier"))
            } else {
                Ok(vec![2.0 * (1.0 + y[0] * y[0])])
            }
        },
    );
    assert!(matches!(underflow, Err(NumError::ConvergenceFailed(_))));
}

#[test]
fn test_rosenbrock_trial_stage_error_halving_and_singularity_exhaustion() {
    let analytical = 1.0 / (1.0 + flexmesa_math::exp(-8.0));

    for (variant, tol) in [
        (RosenbrockVariant::Ros2, 1.0e-5),
        (RosenbrockVariant::Rodas4, 1.0e-7),
    ] {
        let res = rosenbrock_dense(
            variant,
            0.0,
            2.0,
            &[0.5],
            1.0,
            1.0,
            1.0e-8,
            1.0e-8,
            100_000,
            |_t, y| {
                if y[0] > 1.02 {
                    Err(NumError::NonFiniteValue("barrier"))
                } else {
                    Ok(vec![4.0 * y[0] * (1.0 - y[0])])
                }
            },
            |_t, y| Ok(vec![4.0 * (1.0 - 2.0 * y[0])]),
        )
        .expect("rosenbrock recovers from out-of-domain trial stage");
        assert!(
            (res.y[0] - analytical).abs() < tol,
            "variant {variant:?} y(2) = {}, analytical = {analytical}",
            res.y[0],
        );
    }

    // Persistent stage-evaluation error whenever trial y leaves y0 = 0.5 halves h
    // until hitting the step-reduction limit inside `'retry`.
    let underflow = rosenbrock_dense(
        RosenbrockVariant::Rodas4,
        0.0,
        1.0,
        &[0.5],
        0.5,
        1.0,
        1.0e-8,
        1.0e-8,
        100_000,
        |_t, y| {
            if (y[0] - 0.5).abs() > 0.0 {
                Err(NumError::NonFiniteValue("barrier"))
            } else {
                Ok(vec![1.0])
            }
        },
        |_t, _y| Ok(vec![0.0]),
    );
    assert!(matches!(underflow, Err(NumError::ConvergenceFailed(_))));

    // Zero diagonal mass + zero Jacobian produces E = 0 at every halved h,
    // exhausting the five-attempt LU singularity retry limit (nsing >= 5).
    let singular = rosenbrock_dae_diag(
        RosenbrockVariant::Rodas4,
        &[0.0],
        0.0,
        1.0,
        &[0.5],
        0.5,
        1.0,
        1.0e-8,
        1.0e-8,
        100,
        |_t, _y| Ok(vec![1.0]),
        |_t, _y| Ok(vec![0.0]),
    );
    assert!(matches!(singular, Err(NumError::LinearSolveFailed(_))));
}
