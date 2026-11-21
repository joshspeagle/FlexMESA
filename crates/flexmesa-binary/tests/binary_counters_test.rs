// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests verifying that binary and stellar retry/trial/cut
//! counters survive implicit-RLOF rollbacks and hard-limit outer retries
//! (`#150`, F3-7).

#![forbid(unsafe_code)]

mod common;

use flexmesa_binary::DefaultBinaryExtras;
use flexmesa_const::SECYER;
use flexmesa_star::StarError;

/// MESA `26.4.1` on the `(2.0, 1.5, 0.5045 d)` fixture
/// (`benchmarks/fortran_mesa/binary_rlof_fixture/results.json`, case
/// `replacement_2p0_1p5_0p5045d`): model 1 completes at `1000 yr` with no
/// retries; with `fr_hard = 1e-5` and `dt >= 1e4 yr` on model 2 it retries
/// 59 times (52 `fr_hard`, 7 solver) and stops on `min_timestep_limit` at a
/// reported `dt` of `5.474370956548569e-07 s`.
#[test]
fn test_binary_fr_hard_retries_match_mesa_26_4_1_and_counters_survive() {
    let mut sys = common::build_rlof_binary();
    let (dt1, _) = sys
        .try_step_with_extras(1.0e3 * SECYER, &mut DefaultBinaryExtras)
        .expect("step 1");
    assert_eq!(sys.model_number, 1);
    assert_eq!(sys.binary_age_yr, 1000.0, "MESA step1_age_yr = 1000.0");
    assert_eq!(sys.counters.num_retries, 0, "MESA step1_retries_logged = 0");
    let calls_after_step1 = sys.donor.total_num_solver_calls;
    assert!(
        calls_after_step1 >= 2,
        "expected step 1 trial rollbacks to preserve >= 2 donor solver calls, got {calls_after_step1}"
    );

    // Upstream remeshes once per step and every retry restarts from the
    // remeshed model (`star/private/evolve.f90:1882-1895`, `:401`), so the
    // Roche-gap change that `fr_hard` tests
    // (`binary/private/binary_timestep.f90:192-197`) includes the remesh
    // change in radius at every retry and no cut in `dt` removes it.
    sys.controls.fr_hard = 1.0e-5;
    let dt2 = dt1.max(1.0e4 * SECYER);
    let err = sys
        .try_step_with_extras(dt2, &mut DefaultBinaryExtras)
        .expect_err("MESA 26.4.1 stops on min_timestep_limit in model 2");
    let msg = err.to_string();
    assert!(msg.contains("min_timestep_limit"), "unexpected error: {msg}");
    assert_eq!(sys.model_number, 1, "failed step must restore the model");

    // 59 halvings of the 1e4 yr attempt give the timestep MESA reports.
    let dt_final = dt2 * flexmesa_math::powi(0.5, 59);
    let mesa_min_reported_dt_sec = 5.474_370_956_548_569e-7;
    assert!(
        ((dt_final - mesa_min_reported_dt_sec) / mesa_min_reported_dt_sec).abs()
            < 1.0e-12,
        "59 halvings give {dt_final:e} s, MESA reports {mesa_min_reported_dt_sec:e} s"
    );
    assert!(
        msg.contains("retry 59"),
        "MESA retries 59 times (52 fr_hard + 7 solver); got: {msg}"
    );

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
        sys.donor.total_num_solver_calls > calls_after_step1,
        "donor solver calls ({}) must survive hard-limit rollbacks (> {calls_after_step1})",
        sys.donor.total_num_solver_calls
    );
}

/// MESA `26.4.1` on the original `(1.5, 1.0, 0.45 d)` fixture (case
/// `original_1p5_1p0_0p45d`): the `1e3 yr` attempt at model 1 fails and is
/// cut to `125 yr` with one logged retry.
///
/// Model 2 under `fr_hard = 1e-5` is not asserted here. MESA never completes
/// it (15 `fr_hard` retries and 24 solver retries before its
/// `max_number_retries = 40`); `FlexMESA` also never completes it, but with
/// 31 `fr_hard` retries and no solver retries, so the solver behaviour on
/// that model still differs from MESA.
#[test]
fn test_binary_original_fixture_matches_mesa_26_4_1_first_model() {
    let mut sys = common::build_rlof_binary_original();
    let _ = sys
        .try_step_with_extras(1.0e3 * SECYER, &mut DefaultBinaryExtras)
        .expect("step 1");
    assert_eq!(sys.model_number, 1);
    assert_eq!(sys.binary_age_yr, 125.0, "MESA step1_age_yr = 125.0");
    assert_eq!(sys.counters.num_retries, 1, "MESA step1_retries_logged = 1");
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
