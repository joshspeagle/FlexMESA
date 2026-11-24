// SPDX-License-Identifier: GPL-3.0-or-later
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
//! - [`bracket_plus_newton_search`]: Bisection narrowing followed by Newton
//!   refinement with bound clamping, on upstream's nested dual number so the
//!   root carries its structure partials.
//!
//! # Numerical Guarantees
//! - Bisection tolerances (`1.0` for `q_bisection_search`, `1e-4` for
//!   `dq_dz_bisection_search` and `af_bisection_search`) and Newton tolerances
//!   (`CORR_TOL = 1e-13`, `RES_TOL = 1e-8`) match upstream MESA (`tdc.f90` and
//!   `tdc_support.f90`) identically.
//! - Every Newton step clamps out-of-bracket iterates to the midpoint between
//!   the current iterate $Z$ and the violated bound, then performs up to 5
//!   line-search halvings until $|Q(Z_{\mathrm{new}})| < |Q(Z)|$.

use flexmesa_autodiff::AutoDiffReal1VarOrder1;
use flexmesa_utils::is_bad;

use crate::tdc_dual::TdcLift;
use crate::tdc_support::{compute_q, set_y, TdcInfo};
use crate::{TurbError, TurbResult, TurbScalar};

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

/// Bisection to a bracket narrower than one in $Z$, then Newton iterations
/// that refine the root and give it its structure partials
/// (`bracket_plus_Newton_search` in `turb/private/tdc.f90:219-380`).
///
/// The iterate $Z$ is an [`AutoDiffRealTdc`](crate::tdc_dual::AutoDiffRealTdc)
/// number with `d1val1 = 1`. Each update `Z + (-Q / dQdZ)` carries the
/// structure partials of the correction, so the partials of the returned
/// $Y$ are those of the last update, evaluated at the iterates upstream
/// evaluates them at. They equal the implicit-function-theorem partials only
/// in the limit of a fully converged iteration. The loop cannot leave before
/// one update has been applied (`have_derivatives`).
///
/// # Errors
/// Returns [`TurbError::TdcConvergenceFailure`] if the bracket does not
/// contain a sign change, `dQ/dZ` is not finite or is below `1e-99` in size, or
/// the iteration does not converge in 200 steps (upstream `ierr = 1`).
pub fn bracket_plus_newton_search<T: TdcLift>(
    info_p: &TdcInfo<f64>,
    info: &TdcInfo<T::Tdc>,
    scale: f64,
    y_is_positive: bool,
    zlb: f64,
    zub: f64,
) -> TurbResult<(T, usize)> {
    const CORRECTION_TOLERANCE: f64 = 1.0e-13;
    const RESIDUAL_TOLERANCE: f64 = 1.0e-8;
    const MAX_ITER: usize = 200;
    const MAX_LINE_SEARCH_ITER: usize = 5;

    let lift = |v: f64| <T::Tdc as TurbScalar>::from_f64(v);

    // Bisect to a narrow interval around the root (`tdc.f90:262-268`).
    let (mut lb_val, mut ub_val) = (zlb, zub);
    let z_start =
        q_bisection_search(info_p, y_is_positive, &mut lb_val, &mut ub_val)?;
    let mut lower_bound_z = lift(lb_val);
    let mut upper_bound_z = lift(ub_val);

    // `Z%d1val1 = 1d0` (`tdc.f90:272`).
    let mut z = T::with_d1val1(lift(z_start), 1.0);
    let mut z_new = z;
    let mut y = set_y(y_is_positive, z);
    let mut converged = false;
    let mut have_derivatives = false;
    let mut iter_taken = MAX_ITER + 1;

    for iter in 1..=MAX_ITER {
        y = set_y(y_is_positive, z);
        let (q, _) = compute_q(info, y);

        if q.val().abs() / scale <= RESIDUAL_TOLERANCE && have_derivatives {
            converged = true;
            iter_taken = iter;
            break;
        }

        // `Q(Y)` is monotonic, so each iterate tightens one bound
        // (`tdc.f90:296-305`).
        let dq_dz = T::differentiate_1(q);
        if q.val() > 0.0 && dq_dz.val() < 0.0 {
            lower_bound_z = z;
        } else if (q.val() > 0.0 && dq_dz.val() > 0.0)
            || (q.val() < 0.0 && dq_dz.val() < 0.0)
        {
            upper_bound_z = z;
        } else {
            lower_bound_z = z;
        }

        if is_bad(dq_dz.val()) || dq_dz.val().abs() < 1.0e-99 {
            return Err(TurbError::TdcConvergenceFailure("Singular dQ/dZ"));
        }

        let mut correction = -q / dq_dz;

        // Line search against steps that are too large (`tdc.f90:315-337`).
        for _ in 1..=MAX_LINE_SEARCH_ITER {
            if correction.val().abs() < CORRECTION_TOLERANCE && have_derivatives
            {
                converged = true;
                break;
            }

            z_new = z + correction;
            have_derivatives = true;

            // A step past a bound went too far in the right direction; take
            // the midpoint to that bound instead.
            if z_new.val() > upper_bound_z.val() {
                z_new = (z + upper_bound_z) / lift(2.0);
            } else if z_new.val() < lower_bound_z.val() {
                z_new = (z + lower_bound_z) / lift(2.0);
            }

            y = set_y(y_is_positive, z_new);
            let (qc, _) = compute_q(info, y);
            if qc.val().abs() < q.val().abs() {
                break;
            }
            correction = lift(0.5) * correction;
        }

        // `Z_new%d1val1 = 1d0; Z = Z_new` (`tdc.f90:341-342`).
        z_new = T::with_d1val1(z_new, 1.0);
        z = z_new;
        y = set_y(y_is_positive, z);
        if converged {
            iter_taken = iter;
            break;
        }
    }

    if !converged {
        return Err(TurbError::TdcConvergenceFailure(
            "Newton search exceeded max_iter",
        ));
    }

    Ok((T::unconvert(y), iter_taken))
}
