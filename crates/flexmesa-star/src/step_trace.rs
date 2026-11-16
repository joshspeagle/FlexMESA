// SPDX-License-Identifier: LGPL-3.0-or-later
//! Opt-in per-attempt step telemetry for diagnosing retry cascades and
//! timestep collapses in evolutionary campaigns (`#143`).
//!
//! Telemetry is disabled unless the environment variable
//! `FLEXMESA_STEP_TRACE` is set. Its value is the first 1-based `step_number`
//! at which lines are emitted (`FLEXMESA_STEP_TRACE=780` starts tracing at
//! step 780; an empty or non-numeric value traces from step 0). Lines are
//! written to standard error with the prefix `[trace]` and report, for every
//! solver call, the Newton iteration count, residual and correction norms, and
//! the zone and equation holding the maximum residual, and for every step
//! attempt the trial timestep and retry reason.

use std::sync::OnceLock;

static TRACE_FROM: OnceLock<Option<usize>> = OnceLock::new();

/// Returns `true` when step telemetry is enabled for 1-based `step_number`.
#[must_use]
pub fn enabled(step_number: usize) -> bool {
    TRACE_FROM
        .get_or_init(|| {
            std::env::var("FLEXMESA_STEP_TRACE")
                .ok()
                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
        })
        .is_some_and(|from| step_number >= from)
}

/// Locates the largest absolute entry of a residual vector laid out as
/// `equ[k * nvar + i]`, returning `(zone_k, equation_i, value)`.
#[must_use]
pub fn argmax_abs(equ: &[f64], nvar: usize) -> (usize, usize, f64) {
    let mut best = (0_usize, 0_usize, 0.0_f64);
    if nvar == 0 {
        return best;
    }
    for (idx, &v) in equ.iter().enumerate() {
        if v.abs() > best.2.abs() || v.is_nan() {
            best = (idx / nvar, idx % nvar, v);
        }
    }
    best
}
