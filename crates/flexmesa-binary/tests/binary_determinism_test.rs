// SPDX-License-Identifier: LGPL-3.0-or-later
//! Strict-mode multi-thread and repeated-run bit-identity tests for the
//! coupled binary driver (`#150`).

#![forbid(unsafe_code)]

use flexmesa_binary::{
    BinaryInlistControls, BinarySystem, DefaultBinaryExtras,
};
use flexmesa_const::SECYER;
use flexmesa_math::{with_numerics_mode, NumericsMode};
use flexmesa_star::{load_zams_model, StarControls, StarJobControls};

fn run_point_mass_rlof_in_pool(n_threads: usize) -> Vec<u8> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .expect("build rayon thread pool");
    pool.install(|| {
        with_numerics_mode(NumericsMode::Strict, || {
            let donor_ctrl = StarControls {
                initial_mass: 1.5,
                ..StarControls::default()
            };
            let mut donor =
                load_zams_model(donor_ctrl, StarJobControls::default())
                    .expect("1.5 Msun donor ZAMS");
            donor.dt = 1.0e3 * SECYER;
            donor.dt_next = 1.0e3 * SECYER;

            let mut bctrl = BinaryInlistControls {
                m1_msun: 1.5,
                m2_msun: 1.0,
                initial_period_in_days: 0.45,
                mdot_scheme: String::from("Ritter"),
                max_tries_to_achieve: 20,
                ..BinaryInlistControls::default()
            };
            bctrl.use_radiation_corrected_transfer_rate = false;

            let mut sys = BinarySystem::from_controls_point_mass(donor, bctrl)
                .expect("point-mass RLOF binary");
            let mut extras = DefaultBinaryExtras;
            let mut dt = 1.0e3 * SECYER;
            for step in 1..=3 {
                if step == 2 {
                    sys.controls.fr_hard = 1.0e-5;
                    dt = dt.max(1.0e4 * SECYER);
                }
                let (next_dt, _) = sys
                    .try_step_with_extras(dt, &mut extras)
                    .unwrap_or_else(|e| panic!("step {step} failed: {e}"));
                dt = next_dt;
            }
            sys.to_checkpoint_bytes().expect("checkpoint bytes")
        })
    })
}

fn run_two_star_coupled_in_pool(n_threads: usize) -> Vec<u8> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .expect("build rayon thread pool");
    pool.install(|| {
        with_numerics_mode(NumericsMode::Strict, || {
            let donor = load_zams_model(
                StarControls {
                    initial_mass: 1.4,
                    ..StarControls::default()
                },
                StarJobControls::default(),
            )
            .expect("1.4 Msun donor ZAMS");
            let accretor = load_zams_model(
                StarControls {
                    initial_mass: 1.1,
                    ..StarControls::default()
                },
                StarJobControls::default(),
            )
            .expect("1.1 Msun accretor ZAMS");

            let mut bctrl = BinaryInlistControls {
                evolve_both_stars: true,
                m1_msun: 1.4,
                m2_msun: 1.1,
                initial_period_in_days: 0.55,
                mdot_scheme: String::from("Ritter"),
                ..BinaryInlistControls::default()
            };
            bctrl.use_radiation_corrected_transfer_rate = false;

            let mut sys =
                BinarySystem::from_controls_two_stars(donor, accretor, bctrl)
                    .expect("two-star binary");
            let mut extras = DefaultBinaryExtras;
            let mut dt = 5.0e2 * SECYER;
            for step in 1..=2 {
                let (next_dt, _) = sys
                    .try_step_with_extras(dt, &mut extras)
                    .unwrap_or_else(|e| panic!("two-star step {step}: {e}"));
                dt = next_dt;
            }
            sys.to_checkpoint_bytes()
                .expect("two-star checkpoint bytes")
        })
    })
}

#[test]
fn test_point_mass_rlof_binary_is_bit_identical_across_thread_counts() {
    let ref_bytes = run_point_mass_rlof_in_pool(1);
    for (label, run_bytes) in [
        ("2 threads", run_point_mass_rlof_in_pool(2)),
        ("4 threads (run 1)", run_point_mass_rlof_in_pool(4)),
        ("4 threads (run 2)", run_point_mass_rlof_in_pool(4)),
    ] {
        assert_eq!(
            run_bytes, ref_bytes,
            "point-mass RLOF binary diverged on {label}"
        );
    }
}

#[test]
fn test_two_star_coupled_binary_is_bit_identical_across_thread_counts() {
    let ref_bytes = run_two_star_coupled_in_pool(1);
    for (label, run_bytes) in [
        ("2 threads", run_two_star_coupled_in_pool(2)),
        ("4 threads (run 1)", run_two_star_coupled_in_pool(4)),
        ("4 threads (run 2)", run_two_star_coupled_in_pool(4)),
    ] {
        assert_eq!(
            run_bytes, ref_bytes,
            "two-star coupled binary diverged on {label}"
        );
    }
}
