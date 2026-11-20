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

#[test]
fn test_binary_star_limit_redo_halves_dt_without_burning_retries() {
    // Verify `#159`: target-hitting `LimitCheckOutcome::Redo` from the donor's
    // `do_check_limits` cascade (`when_to_stop_atol`/`rtol` on `max_age`) is
    // propagated in BOTH explicit and implicit binary modes, halving `dt` to
    // land within tolerance without consuming outer retries, and honoring
    // `redo_limit`.
    for &(label, max_tries) in &[("explicit", 0_i64), ("implicit", 20_i64)] {
        let mut sys = common::build_rlof_binary();
        sys.controls.max_tries_to_achieve = max_tries;
        let _ = sys
            .try_step_with_extras(1.0e3 * SECYER, &mut DefaultBinaryExtras)
            .expect("step 1");
        let age_after_1 = sys.donor.star_age;
        let retries_before = sys.counters.num_retries;

        // Place `max_age` halfway through a 1000 yr trial step with tight atol
        // so the initial 1000 yr step overshoots (`Redo { dt_new = 500 yr }`)
        // and the halved redo lands squarely on `max_age` and terminates.
        sys.donor.controls.max_age = age_after_1 + 500.0;
        sys.donor.controls.ts_stop.when_to_stop_atol = 1.0;
        sys.donor.controls.ts_stop.when_to_stop_rtol = 0.0;
        sys.donor.controls.ts_stop.redo_limit = 5;

        let _ = sys
            .try_step_with_extras(1.0e3 * SECYER, &mut DefaultBinaryExtras)
            .unwrap_or_else(|e| panic!("{label}: redo step failed: {e:?}"));
        assert_eq!(
            sys.last_termination_reason.as_deref(),
            Some("max_age"),
            "{label}: expected max_age termination after target-hitting redo"
        );
        assert!(
            (sys.donor.star_age - (age_after_1 + 500.0)).abs() <= 1.0,
            "{label}: expected age within 1 yr of max_age, got {} vs {}",
            sys.donor.star_age,
            age_after_1 + 500.0
        );
        assert_eq!(
            sys.counters.num_retries, retries_before,
            "{label}: star limit-hitting redo must not consume binary outer retries"
        );
    }

    // Verify `redo_limit = 1` terminates with `StarError::Terminated { reason: "redo_limit" }`
    // when > 1 halving is needed (`target = age + 100 yr` from `dt = 1000 yr`).
    let mut sys_rl = common::build_rlof_binary();
    let _ = sys_rl
        .try_step_with_extras(1.0e3 * SECYER, &mut DefaultBinaryExtras)
        .expect("step 1");
    let mod1 = sys_rl.model_number;
    sys_rl.donor.controls.max_age = sys_rl.donor.star_age + 100.0;
    sys_rl.donor.controls.ts_stop.when_to_stop_atol = 1.0;
    sys_rl.donor.controls.ts_stop.when_to_stop_rtol = 0.0;
    sys_rl.donor.controls.ts_stop.redo_limit = 1;
    let err = sys_rl
        .try_step_with_extras(1.0e3 * SECYER, &mut DefaultBinaryExtras)
        .expect_err("expected redo_limit termination");
    assert!(
        matches!(err, StarError::Terminated { ref reason, .. } if reason == "redo_limit"),
        "expected StarError::Terminated(redo_limit), got {err:?}"
    );
    assert_eq!(
        sys_rl.model_number, mod1,
        "redo_limit termination must restore pre-step model_number"
    );
}

#[test]
fn test_binary_star_limit_fatal_error_propagates_with_zero_retries() {
    // Verify `#159` + `#264`: fatal error from `do_check_limits`
    // (`peak_burn_vconv_div_cs > 0.75 * limit`) fails loudly in binary runs
    // with zero outer retries instead of being dropped to `None`.
    let mut sys = common::build_rlof_binary();
    let (dt1, _) = sys
        .try_step_with_extras(1.0e3 * SECYER, &mut DefaultBinaryExtras)
        .expect("step 1");
    let retries_before = sys.counters.num_retries;
    sys.donor.controls.ts_stop.peak_burn_vconv_div_cs_limit = 1.0e-12;
    let err = sys
        .try_step_with_extras(dt1, &mut DefaultBinaryExtras)
        .expect_err("peak_burn_vconv_div_cs_limit abort must fail loudly");
    assert!(
        matches!(err, StarError::InvalidInput(_)),
        "expected StarError::InvalidInput from #264 abort, got {err:?}"
    );
    assert_eq!(
        sys.counters.num_retries, retries_before,
        "fatal limit error must not increment binary outer retries"
    );
    assert_eq!(sys.model_number, 1, "failed step must restore snapshot");
}
