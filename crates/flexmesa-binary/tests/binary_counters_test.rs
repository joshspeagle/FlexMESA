// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests verifying that binary and stellar retry/trial/cut
//! counters survive implicit-RLOF rollbacks and hard-limit outer retries
//! (`#150`, F3-7).

#![forbid(unsafe_code)]

mod common;

use flexmesa_binary::DefaultBinaryExtras;
use flexmesa_const::SECYER;
use flexmesa_star::StarError;

#[test]
fn test_binary_implicit_rlof_and_hard_limit_counters_survive_rollbacks() {
    let mut sys = common::build_rlof_binary();
    let (dt1, _) = sys
        .try_step_with_extras(1.0e3 * SECYER, &mut DefaultBinaryExtras)
        .expect("step 1");
    assert_eq!(sys.model_number, 1);
    let calls_after_step1 = sys.donor.total_num_solver_calls;
    assert!(
        calls_after_step1 >= 2,
        "expected step 1 trial rollbacks to preserve >= 2 donor solver calls, got {calls_after_step1}"
    );

    // Step 2: tighten `fr_hard` (`exceeds_hard_limits` activates for `model_number > 0`)
    // so an outer rollback retry fires and preserves all counters.
    sys.controls.fr_hard = 1.0e-5;
    let _ = sys
        .try_step_with_extras(
            dt1.max(1.0e4 * SECYER),
            &mut DefaultBinaryExtras,
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
        sys.donor.total_num_solver_calls
            >= calls_after_step1 + 1 + sys.counters.num_hard_limit_retries,
        "donor solver calls ({}) must survive hard-limit rollbacks (>= {})",
        sys.donor.total_num_solver_calls,
        calls_after_step1 + 1 + sys.counters.num_hard_limit_retries
    );
}

#[test]
fn test_binary_non_retriable_invalid_input_fails_immediately_with_zero_retries() {
    let mut sys = common::build_rlof_binary();
    let (dt1, _) = sys
        .try_step_with_extras(1.0e3 * SECYER, &mut DefaultBinaryExtras)
        .expect("step 1");
    sys.donor.controls.do_phase_separation = true;
    sys.donor.controls.phase_separation_option = String::from("bogus_phase_sep");
    let retries_before = sys.counters.num_retries;

    let err = sys
        .try_step_with_extras(dt1, &mut DefaultBinaryExtras)
        .expect_err("non-retriable donor InvalidInput must fail loudly");
    assert!(
        matches!(err, StarError::InvalidInput(_)),
        "expected StarError::InvalidInput, got {err:?}"
    );
    assert_eq!(
        sys.counters.num_retries, retries_before,
        "non-retriable error must not increment binary outer retries"
    );
    assert_eq!(sys.model_number, 1, "failed step must restore snapshot");
}
