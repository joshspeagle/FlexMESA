# SPDX-License-Identifier: LGPL-3.0-or-later
"""1D bisection and damped Newton-Raphson searches for TDC (`tdc.f90`).

Operates strictly in the 1D logarithmic coordinate $Z = \\ln|Y|$ using
1-variable forward-mode dual numbers (`DualOrder1(nvars=1)`):
1. `q_bisection_search`: Narrows a sign-changing bracket for $Q(Y(Z)) = 0$ to
   width $< 1.0$.
2. `dq_dz_bisection_search`: Locates a stationary point $dQ/dZ = 0$ on the
   negative-$Y$ branch where $A_f(Z) > 0$.
3. `af_bisection_search`: Locates the largest $Z$ on the negative-$Y$ branch
   before $A_f(Z)$ vanishes.
4. `bracket_plus_newton_search`: Combines optional seeded bracketing,
   bisection narrowing, and damped Newton-Raphson refinement.
"""

from __future__ import annotations

import math

from flexmesa import auto_diff as _ad
from flexmesa import utils as _utils
from flexmesa.turb import tdc_support as _support
from flexmesa.turb import types as _types


def _z_const(z: float) -> _ad.DualOrder1:
    """Create a 1-variable constant `DualOrder1` at `z`."""
    return _ad.DualOrder1.constant(z, num_vars=1)


def _z_indep(z: float) -> _ad.DualOrder1:
    """Create a 1-variable independent `DualOrder1` at `z` ($dZ/dZ = 1$)."""
    return _ad.DualOrder1.independent(z, index=0, num_vars=1)


def q_bisection_search(
    info_p: _support.TdcInfo,
    y_is_positive: bool,
    lower_bound_z: float,
    upper_bound_z: float,
) -> tuple[float, float, float]:
    """Narrow a bracket $[Z_{\\mathrm{lb}}, Z_{\\mathrm{ub}}]$ where $Q(Z) = 0$.

    Args:
        info_p: Primal `float` TDC state (`TdcInfo`).
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

    q_lb_d, _ = _support.compute_q(info_p, _support.set_y(y_is_positive, lb))
    q_ub_d, _ = _support.compute_q(info_p, _support.set_y(y_is_positive, ub))
    q_lb = _types.val_of(q_lb_d)
    q_ub = _types.val_of(q_ub_d)
    if _utils.is_bad(q_lb) or _utils.is_bad(q_ub) or q_lb * q_ub > 0.0:
        msg = "Initial Z window does not bracket Q(Z) = 0"
        raise RuntimeError(msg)

    z = 0.5 * (ub + lb)
    for _ in range(max_iter):
        z = 0.5 * (ub + lb)
        q_d, _ = _support.compute_q(info_p, _support.set_y(y_is_positive, z))
        qv = _types.val_of(q_d)
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
    info_z: _support.TdcInfo,
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
    if _types.val_of(af_lb) == 0.0 or not isinstance(q_lb, _ad.DualOrder1):
        return None
    dq_lb = float(q_lb.d1[0])

    q_ub, af_ub = _support.compute_q(
        info_z, _support.set_y(False, _z_indep(ub))
    )
    if _types.val_of(af_ub) == 0.0 or not isinstance(q_ub, _ad.DualOrder1):
        return None
    dq_ub = float(q_ub.d1[0])

    if dq_lb * dq_ub > 0.0:
        return None

    z = 0.5 * (ub + lb)
    for _ in range(max_iter):
        z = 0.5 * (ub + lb)
        q_d, _ = _support.compute_q(info_z, _support.set_y(False, _z_indep(z)))
        assert isinstance(q_d, _ad.DualOrder1)
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
    info_p: _support.TdcInfo,
    lower_bound_z_in: float,
    upper_bound_z_in: float,
) -> float:
    """Locate the largest $Z$ on $Y < 0$ such that $A_f(Z) > 0$.

    Args:
        info_p: Primal `float` TDC state (`TdcInfo`).
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
    if _utils.is_bad(_types.val_of(af_ub)) or _types.val_of(af_ub) > 0.0:
        msg = "af_bisection_search: Af(Zub) > 0"
        raise RuntimeError(msg)

    _, af_lb = _support.compute_q(info_p, _support.set_y(False, lb))
    if _utils.is_bad(_types.val_of(af_lb)) or _types.val_of(af_lb) <= 0.0:
        msg = "af_bisection_search: Af(Zlb) == 0"
        raise RuntimeError(msg)

    for _ in range(max_iter):
        z = 0.5 * (ub + lb)
        _, af = _support.compute_q(info_p, _support.set_y(False, z))
        if _types.val_of(af) > 0.0:
            lb = z
        else:
            ub = z
        if ub - lb < bracket_tol:
            return lb
    return lb


def try_seeded_positive_y_bracket(
    info_p: _support.TdcInfo,
    y_is_positive: bool,
    zlb: float,
    zub: float,
    y_face_guess: float,
) -> tuple[float, float]:
    """Narrow $[Z_{\\mathrm{lb}}, Z_{\\mathrm{ub}}]$ around a positive seed.

    Args:
        info_p: Primal `float` TDC state (`TdcInfo`).
        y_is_positive: Sign flag (`True` for $Y > 0$).
        zlb: Global lower bound $Z_{\\mathrm{lb}}$.
        zub: Global upper bound $Z_{\\mathrm{ub}}$.
        y_face_guess: Positive seed $Y_{\\mathrm{face,guess}}$ ($\\le 0$
            disables).

    Returns:
        Two-tuple `(lb, ub)` of narrowed or original $Z$ bounds.
    """
    half_width = 3.0
    if not y_is_positive or _utils.is_bad(y_face_guess) or y_face_guess <= 0.0:
        return zlb, zub
    z_seed = math.log(y_face_guess)
    if _utils.is_bad(z_seed) or z_seed < zlb or z_seed > zub:
        return zlb, zub
    seed_lb = max(zlb, z_seed - half_width)
    seed_ub = min(zub, z_seed + half_width)
    q_lb, _ = _support.compute_q(info_p, _support.set_y(y_is_positive, seed_lb))
    q_ub, _ = _support.compute_q(info_p, _support.set_y(y_is_positive, seed_ub))
    q_lb_v = _types.val_of(q_lb)
    q_ub_v = _types.val_of(q_ub)
    if (
        not _utils.is_bad(q_lb_v)
        and not _utils.is_bad(q_ub_v)
        and q_lb_v * q_ub_v <= 0.0
    ):
        return seed_lb, seed_ub
    return zlb, zub


def bracket_plus_newton_search(
    info_p: _support.TdcInfo,
    info_z: _support.TdcInfo,
    scale: float,
    y_is_positive: bool,
    zlb: float,
    zub: float,
    y_face_guess: float,
) -> tuple[float, int]:
    """Solve $Q(Y(Z)) = 0$ via bisection narrowing and damped Newton refinement.

    Args:
        info_p: Primal `float` TDC state (`TdcInfo`).
        info_z: 1D `DualOrder1(num_vars=1)` TDC state (`TdcInfo`).
        scale: Luminosity scale for relative residual convergence.
        y_is_positive: Sign flag (`True` for $Y > 0$, `False` for $Y < 0$).
        zlb: Lower bound $Z_{\\mathrm{lb}}$.
        zub: Upper bound $Z_{\\mathrm{ub}}$.
        y_face_guess: Optional positive superadiabaticity seed.

    Returns:
        Two-tuple `(y_final, iter_taken)` of converged $Y^*$ and iteration
        count.

    Raises:
        RuntimeError: If bracketing or Newton refinement fails to converge.
    """
    corr_tol = 1.0e-13
    res_tol = 1.0e-8
    max_iter = 200
    max_line_iter = 5

    lb, ub = try_seeded_positive_y_bracket(
        info_p, y_is_positive, zlb, zub, y_face_guess
    )
    z, lb, ub = q_bisection_search(info_p, y_is_positive, lb, ub)
    converged = False
    have_derivatives = False
    iter_taken = 0

    for it in range(1, max_iter + 1):
        iter_taken = it
        q_d, _ = _support.compute_q(
            info_z, _support.set_y(y_is_positive, _z_indep(z))
        )
        assert isinstance(q_d, _ad.DualOrder1)
        qv = float(q_d.val)
        if _utils.is_bad(qv):
            msg = "NaN/Inf encountered in Q(Z)"
            raise RuntimeError(msg)
        if abs(qv) / scale <= res_tol and have_derivatives:
            converged = True
            break

        dq_dz = float(q_d.d1[0])
        if (qv > 0.0 and dq_dz < 0.0) or (qv <= 0.0 and dq_dz >= 0.0):
            lb = z
        else:
            ub = z

        if _utils.is_bad(dq_dz) or abs(dq_dz) < 1.0e-99:
            msg = "Singular dQ/dZ in Newton search"
            raise RuntimeError(msg)

        correction = -qv / dq_dz
        z_new = z
        for _ in range(max_line_iter):
            if abs(correction) < corr_tol and have_derivatives:
                converged = True
                break
            z_new = z + correction
            have_derivatives = True
            if z_new > ub:
                z_new = 0.5 * (z + ub)
            elif z_new < lb:
                z_new = 0.5 * (z + lb)
            qc_d, _ = _support.compute_q(
                info_p, _support.set_y(y_is_positive, z_new)
            )
            qcv = _types.val_of(qc_d)
            if _utils.is_bad(qcv) or abs(qcv) >= abs(qv):
                correction *= 0.5
            else:
                break
        z = z_new
        if converged:
            break

    if not converged:
        msg = "TDC Newton search exceeded max_iter"
        raise RuntimeError(msg)

    y_final = math.exp(z) if y_is_positive else -math.exp(z)
    return y_final, iter_taken
