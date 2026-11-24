// SPDX-License-Identifier: GPL-3.0-or-later
//! Strict-mode multi-thread and repeated-run bit-identity tests for the
//! coupled binary driver (`#150`).

#![forbid(unsafe_code)]

mod common;

use flexmesa_binary::{
    BinaryInlistControls, BinarySystem, DefaultBinaryExtras,
};
use flexmesa_const::SECYER;
use flexmesa_math::{NumericsMode, with_numerics_mode};
use flexmesa_star::{StarControls, StarJobControls, load_zams_model};

fn run_point_mass_rlof_in_pool(n_threads: usize) -> Vec<u8> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .expect("build rayon thread pool");
    pool.install(|| {
        with_numerics_mode(NumericsMode::Strict, || {
            let mut sys = common::build_rlof_binary();
            common::advance_binary_steps_one_to_three(&mut sys);
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
                    zbase: 0.02,
                    ..StarControls::default()
                },
                StarJobControls::default(),
            )
            .expect("1.4 Msun donor ZAMS");
            let accretor = load_zams_model(
                StarControls {
                    initial_mass: 1.1,
                    zbase: 0.02,
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
                terminate_if_initial_overflow: false,
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
