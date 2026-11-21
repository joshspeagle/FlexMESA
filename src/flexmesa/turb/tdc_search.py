# SPDX-License-Identifier: LGPL-3.0-or-later
"""1D bisection and damped Newton-Raphson searches for TDC (`tdc.f90`).

Operates in the 1D logarithmic coordinate $Z = \\ln|Y|$:
1. `q_bisection_search`: Narrows a sign-changing bracket for $Q(Y(Z)) = 0$ to
   width $< 1.0$.
2. `dq_dz_bisection_search`: Locates a stationary point $dQ/dZ = 0$ on the
   negative-$Y$ branch where $A_f(Z) > 0$.
3. `af_bisection_search`: Locates the largest $Z$ on the negative-$Y$ branch
   before $A_f(Z)$ vanishes.
4. `bracket_plus_newton_search`: Bisection narrowing from `[zlb, zub]` followed
   by damped Newton-Raphson refinement on `AutoDiffRealTdc`, carrying structure
   partials through the Newton updates as in `turb/private/tdc.f90:219-380`.
"""

from __future__ import annotations

from flexmesa import auto_diff as _ad
from flexmesa import utils as _utils
from flexmesa.turb import tdc_dual as _tdc_dual
from flexmesa.turb import tdc_support as _support
from flexmesa.turb import types as _types


def _z_indep(z: float) -> _ad.DualOrder1:
    """Create a 1-variable independent `DualOrder1` at `z` ($dZ/dZ = 1$)."""
    return _ad.DualOrder1.independent(z, index=0, num_vars=1)


def q_bisection_search(
    info_p: _support.TdcInfo[float],
    y_is_positive: bool,
    lower_bound_z: float,
    upper_bound_z: float,
) -> tuple[float, float, float]:
    """Narrow a bracket $[Z_{\\mathrm{lb}}, Z_{\\mathrm{ub}}]$ where $Q(Z) = 0$.

    Args:
        info_p: Primal `float` TDC state (`TdcInfo[float]`).
        y_is_positive: Sign flag (`True` for $Y > 0$, `False` for $Y < 0$).
        lower_bound_z: Initial lower bound $Z_{\\mathrm{lb}}$.
        upper_bound_z: Initial upper bound $Z_{\\mathrm{ub}}$.

    Returns:
        Three-tuple `(z_mid, lb_out, ub_out)` with narrowed bounds.

    Raises:
        RuntimeError: If $Q(Z_{\\mathrm{lb}})$ and $Q(Z_{\\mathrm{ub}})$ have
            the same sign.
    """
    bracket_tol = 1.0
    max_iter = 30
    lb = float(lower_bound_z)
    ub = float(upper_bound_z)

    q_lb, _ = _support.compute_q(info_p, _support.set_y(y_is_positive, lb))
    q_ub, _ = _support.compute_q(info_p, _support.set_y(y_is_positive, ub))
    if _utils.is_bad(q_lb) or _utils.is_bad(q_ub) or q_lb * q_ub > 0.0:
        msg = "Initial Z window does not bracket Q(Z) = 0"
        raise RuntimeError(msg)

    z = 0.5 * (ub + lb)
    for _ in range(max_iter):
        z = 0.5 * (ub + lb)
        qv, _ = _support.compute_q(info_p, _support.set_y(y_is_positive, z))
        if qv * q_ub > 0.0:
            ub = z
            q_ub = qv
        elif qv * q_lb > 0.0:
            lb = z
            q_lb = qv
        if ub - lb < bracket_tol:
            return 0.5 * (ub + lb), lb, ub
    return z, lb, ub


def dq_dz_bisection_search(
    info_z: _support.TdcInfo[_ad.DualOrder1],
    lower_bound_z_in: float,
    upper_bound_z_in: float,
) -> float | None:
    """Locate a stationary point $dQ/dZ = 0$ on $Y < 0$ where $A_f > 0$.

    Args:
        info_z: 1D `DualOrder1(num_vars=1)` TDC state (`TdcInfo`).
        lower_bound_z_in: Lower bound $Z_{\\mathrm{lb}}$.
        upper_bound_z_in: Upper bound $Z_{\\mathrm{ub}}$.

    Returns:
        Stationary coordinate $Z_1$ if a bracketed stationary point exists with
        $A_f > 0$, or `None` otherwise.
    """
    bracket_tol = 1.0e-4
    max_iter = 50
    lb = float(lower_bound_z_in)
    ub = float(upper_bound_z_in)

    q_lb, af_lb = _support.compute_q(
        info_z, _support.set_y(False, _z_indep(lb))
    )
    if _types.val_of(af_lb) == 0.0:
        return None
    dq_lb = float(q_lb.d1[0])

    q_ub, af_ub = _support.compute_q(
        info_z, _support.set_y(False, _z_indep(ub))
    )
    if _types.val_of(af_ub) == 0.0:
        return None
    dq_ub = float(q_ub.d1[0])

    if dq_lb * dq_ub > 0.0:
        return None

    z = 0.5 * (ub + lb)
    for _ in range(max_iter):
        z = 0.5 * (ub + lb)
        q_d, _ = _support.compute_q(info_z, _support.set_y(False, _z_indep(z)))
        dq = float(q_d.d1[0])
        if dq * dq_ub > 0.0:
            ub = z
            dq_ub = dq
        elif dq * dq_lb > 0.0:
            lb = z
            dq_lb = dq
        if ub - lb < bracket_tol:
            return 0.5 * (ub + lb)
    return z


def af_bisection_search(
    info_p: _support.TdcInfo[float],
    lower_bound_z_in: float,
    upper_bound_z_in: float,
) -> float:
    """Locate the largest $Z$ on $Y < 0$ such that $A_f(Z) > 0$.

    Args:
        info_p: Primal `float` TDC state (`TdcInfo[float]`).
        lower_bound_z_in: Lower bound $Z_{\\mathrm{lb}}$ where $A_f > 0$.
        upper_bound_z_in: Upper bound $Z_{\\mathrm{ub}}$ where $A_f = 0$.

    Returns:
        Largest coordinate $Z_0$ where $A_f(Z_0) > 0$ to tolerance $10^{-4}$.

    Raises:
        RuntimeError: If the bounds do not straddle the $A_f = 0$ transition.
    """
    bracket_tol = 1.0e-4
    max_iter = 50
    lb = float(lower_bound_z_in)
    ub = float(upper_bound_z_in)

    _, af_ub = _support.compute_q(info_p, _support.set_y(False, ub))
    if _utils.is_bad(af_ub) or af_ub > 0.0:
        msg = "af_bisection_search: Af(Zub) > 0"
        raise RuntimeError(msg)

    _, af_lb = _support.compute_q(info_p, _support.set_y(False, lb))
    if _utils.is_bad(af_lb) or af_lb <= 0.0:
        msg = "af_bisection_search: Af(Zlb) == 0"
        raise RuntimeError(msg)

    for _ in range(max_iter):
        z = 0.5 * (ub + lb)
        _, af = _support.compute_q(info_p, _support.set_y(False, z))
        if af > 0.0:
            lb = z
        else:
            ub = z
        if ub - lb < bracket_tol:
            return lb
    return lb


def bracket_plus_newton_search(
    info_p: _support.TdcInfo[float],
    info_t: _support.TdcInfo[_tdc_dual.AutoDiffRealTdc],
    scale: float,
    y_is_positive: bool,
    zlb: float,
    zub: float,
) -> tuple[_tdc_dual.AutoDiffRealTdc, int]:
    """Solve $Q(Y(Z)) = 0$ via bisection narrowing and damped Newton refinement.

    Runs the Newton loop on `AutoDiffRealTdc` (`auto_diff_real_tdc`), matching
    `bracket_plus_Newton_search` in `turb/private/tdc.f90:230-379` and
    `crates/flexmesa-turb/src/tdc_search.rs:203-306`. The returned $Y$ carries
    the structure partials produced by the last Newton update.

    Args:
        info_p: Primal `float` TDC state (`TdcInfo[float]`).
        info_t: Nested dual TDC state (`TdcInfo[AutoDiffRealTdc]`).
        scale: Luminosity scale for relative residual convergence.
        y_is_positive: Sign flag (`True` for $Y > 0$, `False` for $Y < 0$).
        zlb: Lower bound $Z_{\\mathrm{lb}}$.
        zub: Upper bound $Z_{\\mathrm{ub}}$.

    Returns:
        Two-tuple `(y_final, iter_taken)` of converged `AutoDiffRealTdc` root
        $Y^*$ and iteration count.

    Raises:
        RuntimeError: If bracketing or Newton refinement fails to converge.
    """
    corr_tol = 1.0e-13
    res_tol = 1.0e-8
    max_iter = 200
    max_line_iter = 5

    num_vars = info_t.lum.num_vars

    def _lift(v: float) -> _tdc_dual.AutoDiffRealTdc:
        return _tdc_dual.AutoDiffRealTdc.constant(v, num_vars)

    z_start, lb_val, ub_val = q_bisection_search(
        info_p, y_is_positive, zlb, zub
    )
    lower_bound_z = _lift(lb_val)
    upper_bound_z = _lift(ub_val)
    z = _lift(z_start).with_d1val1(1.0)
    z_new = z
    y = _support.set_y(y_is_positive, z)
    converged = False
    have_derivatives = False
    iter_taken = max_iter + 1

    for it in range(1, max_iter + 1):
        y = _support.set_y(y_is_positive, z)
        q, _ = _support.compute_q(info_t, y)
        if abs(q.val) / scale <= res_tol and have_derivatives:
            converged = True
            iter_taken = it
            break

        dq_dz = q.differentiate_1()
        if q.val > 0.0 and dq_dz.val < 0.0:
            lower_bound_z = z
        elif (q.val > 0.0 and dq_dz.val > 0.0) or (
            q.val < 0.0 and dq_dz.val < 0.0
        ):
            upper_bound_z = z
        else:
            lower_bound_z = z

        if _utils.is_bad(dq_dz.val) or abs(dq_dz.val) < 1.0e-99:
            msg = "Singular dQ/dZ in Newton search"
            raise RuntimeError(msg)

        correction = -q / dq_dz

        for _ in range(max_line_iter):
            if abs(correction.val) < corr_tol and have_derivatives:
                converged = True
                break
            z_new = z + correction
            have_derivatives = True
            if z_new.val > upper_bound_z.val:
                z_new = (z + upper_bound_z) / _lift(2.0)
            elif z_new.val < lower_bound_z.val:
                z_new = (z + lower_bound_z) / _lift(2.0)
            y = _support.set_y(y_is_positive, z_new)
            qc, _ = _support.compute_q(info_t, y)
            if abs(qc.val) < abs(q.val):
                break
            correction = _lift(0.5) * correction

        z_new = z_new.with_d1val1(1.0)
        z = z_new
        y = _support.set_y(y_is_positive, z)
        if converged:
            iter_taken = it
            break

    if not converged:
        msg = "TDC Newton search exceeded max_iter"
        raise RuntimeError(msg)

    return y, iter_taken
