// SPDX-License-Identifier: LGPL-3.0-or-later
//! 1D bisection and damped Newton-Raphson searches for Time-Dependent
//! Convection (`turb/private/tdc.f90` and `turb/private/tdc_support.f90`).
//!
//! Operates strictly in the 1D logarithmic coordinate $Z = \ln|Y|$ using
//! [`AutoDiffReal1VarOrder1`] (`DualOrder1<1>`) with $\partial Z / \partial Z =
//! 1$:
//! - [`q_bisection_search`]: Narrows a sign-changing bracket for
//!   $Q(Y(Z)) = 0$ to width $< 1.0$.
//! - [`dq_dz_bisection_search`]: Locates a stationary point $dQ/dZ = 0$ on the
//!   negative-$Y$ branch where $A_f(Z) > 0$.
//! - [`af_bisection_search`]: Locates the largest $Z$ on the negative-$Y$
//!   branch before $A_f(Z)$ vanishes.
//! - [`bracket_plus_newton_search`]: Combines optional seeded bracketing,
//!   bisection narrowing, and damped Newton-Raphson refinement with bound
//!   clamping.
//! - [`try_seeded_positive_y_bracket`]: Narrows the search interval to a
//!   $\pm 3$ window in $Z$ around $\ln(Y_{\mathrm{face,guess}})$ when a valid
//!   positive seed is provided and brackets the root.
//!
//! # Numerical Guarantees
//! - All inner evaluations use IEEE-754 double precision (`f64`).
//! - Bisection tolerances (`1.0` for `q_bisection_search`, `1e-4` for
//!   `dq_dz_bisection_search` and `af_bisection_search`) and Newton tolerances
//!   (`CORR_TOL = 1e-13`, `RES_TOL = 1e-8`) match upstream MESA (`tdc.f90` and
//!   `tdc_support.f90`) identically.
//! - Every Newton step clamps out-of-bracket iterates to the midpoint between
//!   the current iterate $Z$ and the violated bound, then performs up to 5
//!   line-search halvings until $|Q(Z_{\mathrm{new}})| < |Q(Z)|$.

use flexmesa_autodiff::AutoDiffReal1VarOrder1;
use flexmesa_math::{exp, log};
use flexmesa_utils::is_bad;

use crate::tdc_support::{compute_q, set_y, TdcInfo};
use crate::{TurbError, TurbResult};

/// Bisection search narrowing a bracket $[Z_{\mathrm{lb}}, Z_{\mathrm{ub}}]$
/// where $Q(Y(Z))$ changes sign to width $< 1.0$ (`Q_bisection_search` in
/// `tdc_support.f90`).
///
/// # Errors
/// Returns [`TurbError::TdcConvergenceFailure`] if $Q(Z_{\mathrm{lb}})$ and
/// $Q(Z_{\mathrm{ub}})$ have the same sign.
pub fn q_bisection_search(
    info_p: &TdcInfo<f64>,
    y_is_positive: bool,
    lower_bound_z: &mut f64,
    upper_bound_z: &mut f64,
) -> TurbResult<f64> {
    const BRACKET_TOL: f64 = 1.0;
    const MAX_ITER: usize = 30;

    let (mut q_lb, _) = compute_q(info_p, set_y(y_is_positive, *lower_bound_z));
    let (mut q_ub, _) = compute_q(info_p, set_y(y_is_positive, *upper_bound_z));

    if is_bad(q_lb) || is_bad(q_ub) || q_lb * q_ub > 0.0 {
        return Err(TurbError::TdcConvergenceFailure(
            "Initial Z window does not bracket Q(Z) = 0",
        ));
    }

    let mut z = 0.5 * (*upper_bound_z + *lower_bound_z);
    for _ in 0..MAX_ITER {
        z = 0.5 * (*upper_bound_z + *lower_bound_z);
        let (q, _) = compute_q(info_p, set_y(y_is_positive, z));

        if q * q_ub > 0.0 {
            *upper_bound_z = z;
            q_ub = q;
        } else if q * q_lb > 0.0 {
            *lower_bound_z = z;
            q_lb = q;
        }

        if *upper_bound_z - *lower_bound_z < BRACKET_TOL {
            return Ok(0.5 * (*upper_bound_z + *lower_bound_z));
        }
    }
    Ok(z)
}

/// Bisection search for a stationary point $dQ/dZ = 0$ on $Y < 0$ where
/// $A_f > 0$ (`dQdZ_bisection_search` in `tdc_support.f90`).
///
/// Evaluates $dQ/dZ$ using 1D forward-mode automatic differentiation
/// ([`AutoDiffReal1VarOrder1::independent`]). Returns `None` if $A_f = 0$ at
/// either endpoint or if $dQ/dZ$ does not change sign across the interval.
#[must_use]
pub fn dq_dz_bisection_search(
    info: &TdcInfo<AutoDiffReal1VarOrder1>,
    lower_bound_z_in: f64,
    upper_bound_z_in: f64,
) -> Option<f64> {
    const BRACKET_TOL: f64 = 1.0e-4;
    const MAX_ITER: usize = 50;

    let mut lb = lower_bound_z_in;
    let mut ub = upper_bound_z_in;

    let y_lb = set_y(false, AutoDiffReal1VarOrder1::independent(lb, 0));
    let (q_lb, af_lb) = compute_q(info, y_lb);
    if af_lb.val == 0.0 {
        return None;
    }
    let mut dq_lb = q_lb.d1[0];

    let y_ub = set_y(false, AutoDiffReal1VarOrder1::independent(ub, 0));
    let (q_ub, af_ub) = compute_q(info, y_ub);
    if af_ub.val == 0.0 {
        return None;
    }
    let mut dq_ub = q_ub.d1[0];

    if dq_lb * dq_ub > 0.0 {
        return None;
    }

    let mut z = 0.5 * (ub + lb);
    for _ in 0..MAX_ITER {
        z = 0.5 * (ub + lb);
        let y = set_y(false, AutoDiffReal1VarOrder1::independent(z, 0));
        let (q, _) = compute_q(info, y);
        let dq = q.d1[0];

        if dq * dq_ub > 0.0 {
            ub = z;
            dq_ub = dq;
        } else if dq * dq_lb > 0.0 {
            lb = z;
            dq_lb = dq;
        }

        if ub - lb < BRACKET_TOL {
            return Some(0.5 * (ub + lb));
        }
    }
    Some(z)
}

/// Bisection search for the largest $Z \in [Z_{\mathrm{lb}}, Z_{\mathrm{ub}}]$
/// on $Y < 0$ such that $A_f(Z) > 0$ (`Af_bisection_search` in
/// `tdc_support.f90`).
///
/// Narrows $[Z_{\mathrm{lb}}, Z_{\mathrm{ub}}]$ to width $< 10^{-4}$ over at
/// most 50 iterations and returns the lower bound where $A_f(Z) > 0$.
///
/// # Errors
/// Returns [`TurbError::TdcConvergenceFailure`] if the bounds do not straddle
/// the $A_f = 0$ transition.
pub fn af_bisection_search(
    info_p: &TdcInfo<f64>,
    lower_bound_z_in: f64,
    upper_bound_z_in: f64,
) -> TurbResult<f64> {
    const BRACKET_TOL: f64 = 1.0e-4;
    const MAX_ITER: usize = 50;

    let mut lb = lower_bound_z_in;
    let mut ub = upper_bound_z_in;

    let (_, af_ub) = compute_q(info_p, set_y(false, ub));
    if is_bad(af_ub) || af_ub > 0.0 {
        return Err(TurbError::TdcConvergenceFailure(
            "Af_bisection_search: Af(Zub) > 0",
        ));
    }

    let (_, af_lb) = compute_q(info_p, set_y(false, lb));
    if is_bad(af_lb) || af_lb <= 0.0 {
        return Err(TurbError::TdcConvergenceFailure(
            "Af_bisection_search: Af(Zlb) == 0",
        ));
    }

    for _ in 0..MAX_ITER {
        let z = 0.5 * (ub + lb);
        let (_, af) = compute_q(info_p, set_y(false, z));
        if af > 0.0 {
            lb = z;
        } else {
            ub = z;
        }
        if ub - lb < BRACKET_TOL {
            return Ok(lb);
        }
    }
    Ok(lb)
}

/// Bisection + damped Newton search for $Q(Y(Z)) = 0$ on a monotone interval
/// (`bracket_plus_Newton_search` in `tdc.f90`).
///
/// First attempts to narrow $[Z_{\mathrm{lb}}, Z_{\mathrm{ub}}]$ using
/// [`try_seeded_positive_y_bracket`], then bisects the bracket to width $< 1.0$
/// via [`q_bisection_search`], and finally refines $Z$ with damped
/// Newton-Raphson steps until $|Q(Z)| / \text{scale} \le 10^{-8}$ or
/// $|\Delta Z| < 10^{-13}$.
///
/// # Errors
/// Returns [`TurbError::TdcConvergenceFailure`] if bracketing or Newton
/// refinement fails to converge within 200 iterations.
pub fn bracket_plus_newton_search(
    info_p: &TdcInfo<f64>,
    info_z: &TdcInfo<AutoDiffReal1VarOrder1>,
    scale: f64,
    y_is_positive: bool,
    zlb: f64,
    zub: f64,
    y_face_guess: f64,
) -> TurbResult<(f64, usize)> {
    const CORR_TOL: f64 = 1.0e-13;
    const RES_TOL: f64 = 1.0e-8;
    const MAX_ITER: usize = 200;
    const MAX_LINE_ITER: usize = 5;

    let mut lb = zlb;
    let mut ub = zub;
    try_seeded_positive_y_bracket(
        info_p,
        y_is_positive,
        zlb,
        zub,
        &mut lb,
        &mut ub,
        y_face_guess,
    );

    let mut z = q_bisection_search(info_p, y_is_positive, &mut lb, &mut ub)?;
    let mut converged = false;
    let mut have_derivatives = false;
    let mut iter_taken = 0_usize;

    for iter in 1..=MAX_ITER {
        iter_taken = iter;
        let y = set_y(y_is_positive, AutoDiffReal1VarOrder1::independent(z, 0));
        let (q, _) = compute_q(info_z, y);
        if is_bad(q.val) {
            return Err(TurbError::TdcConvergenceFailure("NaN/Inf in Q(Z)"));
        }

        if q.val.abs() / scale <= RES_TOL && have_derivatives {
            converged = true;
            break;
        }

        let dq_dz = q.d1[0];
        if (q.val > 0.0 && dq_dz < 0.0) || (q.val <= 0.0 && dq_dz >= 0.0) {
            lb = z;
        } else {
            ub = z;
        }

        if is_bad(dq_dz) || dq_dz.abs() < 1.0e-99 {
            return Err(TurbError::TdcConvergenceFailure("Singular dQ/dZ"));
        }

        let mut correction = -q.val / dq_dz;
        let mut z_new = z;
        for _ in 1..=MAX_LINE_ITER {
            if correction.abs() < CORR_TOL && have_derivatives {
                converged = true;
                break;
            }
            z_new = z + correction;
            have_derivatives = true;
            if z_new > ub {
                z_new = 0.5 * (z + ub);
            } else if z_new < lb {
                z_new = 0.5 * (z + lb);
            }
            let (qc, _) = compute_q(info_p, set_y(y_is_positive, z_new));
            if is_bad(qc) || qc.abs() >= q.val.abs() {
                correction *= 0.5;
            } else {
                break;
            }
        }
        z = z_new;
        if converged {
            break;
        }
    }

    if !converged {
        return Err(TurbError::TdcConvergenceFailure(
            "Newton search exceeded max_iter",
        ));
    }

    let y_final = if y_is_positive { exp(z) } else { -exp(z) };
    Ok((y_final, iter_taken))
}

/// Narrows $[Z_{\mathrm{lb}}, Z_{\mathrm{ub}}]$ to a $\pm 3$ window around
/// $\ln(Y_{\mathrm{face,guess}})$ when valid (`try_seeded_positive_Y_bracket`
/// in `tdc.f90`).
///
/// If $Y > 0$, `y_face_guess > 0`, and the $\pm 3$ window around
/// $Z_{\mathrm{seed}} = \ln(Y_{\mathrm{face,guess}})$ lies within
/// $[Z_{\mathrm{lb}}, Z_{\mathrm{ub}}]$ with a sign change in $Q$, updates
/// `lower_bound_z` and `upper_bound_z` in place.
pub fn try_seeded_positive_y_bracket(
    info_p: &TdcInfo<f64>,
    y_is_positive: bool,
    zlb: f64,
    zub: f64,
    lower_bound_z: &mut f64,
    upper_bound_z: &mut f64,
    y_face_guess: f64,
) {
    const HALF_WIDTH: f64 = 3.0;
    if !y_is_positive || is_bad(y_face_guess) || y_face_guess <= 0.0 {
        return;
    }
    let z_seed = log(y_face_guess);
    if is_bad(z_seed) || z_seed < zlb || z_seed > zub {
        return;
    }
    let seed_lb = zlb.max(z_seed - HALF_WIDTH);
    let seed_ub = zub.min(z_seed + HALF_WIDTH);
    let (q_lb, _) = compute_q(info_p, set_y(y_is_positive, seed_lb));
    let (q_ub, _) = compute_q(info_p, set_y(y_is_positive, seed_ub));
    if !is_bad(q_lb) && !is_bad(q_ub) && q_lb * q_ub <= 0.0 {
        *lower_bound_z = seed_lb;
        *upper_bound_z = seed_ub;
    }
}
