// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests verifying that binary and stellar retry/trial/cut
//! counters survive implicit-RLOF rollbacks and hard-limit outer retries
//! (`#150`, F3-7).

#![forbid(unsafe_code)]

use flexmesa_binary::{BinaryInlistControls, BinarySystem};
use flexmesa_const::SECYER;
use flexmesa_star::{load_zams_model, StarControls, StarJobControls};

#[test]
fn test_binary_implicit_rlof_and_hard_limit_counters_survive_rollbacks() {
    let d_ctrl = StarControls {
        initial_mass: 1.5,
        ..StarControls::default()
    };
    let mut donor = load_zams_model(d_ctrl, StarJobControls::default())
        .expect("1.5 Msun ZAMS");
    // Tiny initial period puts the donor into Roche-lobe overflow with
    // implicit trials and outer hard-limit retries.
    let mut bctrl = BinaryInlistControls {
        m1_msun: 1.5,
        m2_msun: 1.0,
        initial_period_in_days: 0.45,
        mdot_scheme: String::from("Ritter"),
        max_tries_to_achieve: 20,
        ..BinaryInlistControls::default()
    };
    bctrl.use_radiation_corrected_transfer_rate = false;
    donor.dt_next = 1.0e3 * SECYER;
    donor.dt = 1.0e3 * SECYER;

    let mut sys =
        BinarySystem::from_controls_point_mass(donor, bctrl).expect("binary");
    let (dt1, _) = sys
        .try_step_with_extras(
            1.0e3 * SECYER,
            &mut flexmesa_binary::DefaultBinaryExtras,
        )
        .expect("step 1");
    assert_eq!(sys.model_number, 1);

    // Step 2: tighten `fr_hard` (`exceeds_hard_limits` activates for `model_number > 0`)
    // so an outer rollback retry fires and preserves all counters.
    sys.controls.fr_hard = 1.0e-5;
    let _ = sys
        .try_step_with_extras(
            dt1.max(1.0e4 * SECYER),
            &mut flexmesa_binary::DefaultBinaryExtras,
        )
        .expect("binary step 2 succeeds after hard-limit rollback");

    assert!(
        sys.counters.num_implicit_trials >= 2,
        "expected >= 2 implicit trials, got {}",
        sys.counters.num_implicit_trials
    );
    assert!(
        sys.counters.num_hard_limit_retries >= 1,
        "expected >= 1 hard-limit retry, got {}",
        sys.counters.num_hard_limit_retries
    );
    assert!(
        sys.counters.num_retries >= 1,
        "expected >= 1 outer retry preserved across rollback, got {}",
        sys.counters.num_retries
    );
    assert!(
        sys.donor.total_num_solver_calls >= sys.counters.num_implicit_trials,
        "donor solver calls ({}) must survive trial rollbacks (>= {})",
        sys.donor.total_num_solver_calls,
        sys.counters.num_implicit_trials
    );
}
