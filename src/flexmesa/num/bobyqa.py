# SPDX-License-Identifier: LGPL-3.0-or-later
"""Powell (`2009`) bound-constrained quadratic optimizer (``BOBYQA``).

Ports MESA's ``num/private/mod_bobyqa.f`` box-constrained derivative-free
trust-region optimizer into ``float64`` NumPy.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flexmesa.num import newuoa as _newuoa


@dataclasses.dataclass(frozen=True)
class BobyqaResult:
    """Result returned by :func:`bobyqa`.

    Attributes:
        x_min: Best feasible parameter vector ``(n,)`` in ``float64``.
        f_min: Minimum objective value ``f(x_min)``.
        num_evals: Total number of objective evaluations performed.
    """

    x_min: NDArray[np.float64]
    f_min: float
    num_evals: int


def bobyqa(
    calfun: Callable[[NDArray[np.float64]], float],
    x0: ArrayLike,
    xl: ArrayLike,
    xu: ArrayLike,
    npt: int | None = None,
    rhobeg: float = 1e-1,
    rhoend: float = 1e-6,
    maxfun: int = 5000,
    max_valid_value: float = 1e99,
) -> BobyqaResult:
    """Minimizes a box-constrained objective via Powell's ``BOBYQA`` algorithm.

    Args:
        calfun: Objective callable ``f(x) -> float`` accepting a 1D ``(n,)``
            ``float64`` array.
        x0: Initial parameter vector of shape ``(n,)`` with ``n >= 2``.
        xl: Lower bound array of shape ``(n,)``.
        xu: Upper bound array of shape ``(n,)`` with ``xu - xl >= 2 * rhobeg``.
        npt: Number of interpolation conditions in
            ``[n + 2, (n + 1)(n + 2) // 2]`` (defaults to ``2 * n + 1``).
        rhobeg: Initial trust-region radius (``> 0``).
        rhoend: Final trust-region radius (``0 < rhoend <= rhobeg``).
        maxfun: Maximum number of objective evaluations allowed.
        max_valid_value: Upper bound on acceptable objective values.

    Returns:
        :class:`BobyqaResult` with best feasible point ``x_min``, value
        ``f_min``, and evaluation count ``num_evals``.

    Raises:
        ValueError: If dimensions, ``npt``, or box widths ``xu - xl < 2*rhobeg``
            are invalid.
    """
    x = np.asarray(x0, dtype=np.float64).ravel().copy()
    lo = np.asarray(xl, dtype=np.float64).ravel()
    hi = np.asarray(xu, dtype=np.float64).ravel()
    n = int(x.size)
    if npt is None:
        npt = 2 * n + 1
    max_npt = (n + 1) * (n + 2) // 2
    if (
        n < 2
        or lo.size < n
        or hi.size < n
        or npt < n + 2
        or npt > max_npt
        or rhobeg <= 0.0
        or rhoend <= 0.0
        or rhoend > rhobeg
        or np.any(hi[:n] - lo[:n] < 2.0 * rhobeg)
    ):
        raise ValueError(
            "require n >= 2, xu - xl >= 2*rhobeg, and valid npt/rho bounds"
        )

    for j in range(n):
        if x[j] <= lo[j]:
            x[j] = lo[j]
        elif x[j] - lo[j] < rhobeg:
            x[j] = lo[j] + rhobeg
        elif x[j] >= hi[j]:
            x[j] = hi[j]
        elif hi[j] - x[j] < rhobeg:
            x[j] = hi[j] - rhobeg

    nf = 0

    def eval_f(pt: NDArray[np.float64]) -> float:
        nonlocal nf
        nf += 1
        clamped = np.clip(pt, lo[:n], hi[:n])
        val = float(calfun(clamped))
        if not math.isfinite(val):
            return min(max_valid_value, 1e99)
        return min(val, max_valid_value)

    xbase = x.copy()
    sl = lo[:n] - xbase
    su = hi[:n] - xbase
    xpt = np.zeros((npt, n), dtype=np.float64)
    fval = np.zeros(npt, dtype=np.float64)
    kopt = 0

    for k in range(npt):
        if 1 <= k <= n:
            j = k - 1
            xpt[k, j] = -rhobeg if su[j] == 0.0 else min(rhobeg, float(su[j]))
        elif n < k <= 2 * n:
            j = k - n - 1
            if sl[j] == 0.0:
                xpt[k, j] = min(2.0 * rhobeg, float(su[j]))
            elif su[j] == 0.0:
                xpt[k, j] = max(-2.0 * rhobeg, float(sl[j]))
            else:
                xpt[k, j] = max(-rhobeg, float(sl[j]))
        elif k > 2 * n:
            nfmm = k - n
            itemp = (nfmm - 1) // n
            jpt = k - itemp * n - n
            ipt = jpt + itemp
            if ipt > n:
                ipt, jpt = jpt, ipt - n
            xpt[k, ipt - 1] = xpt[ipt, ipt - 1]
            xpt[k, jpt - 1] = xpt[jpt, jpt - 1]
        fval[k] = eval_f(xbase + xpt[k])
        if k == 0 or fval[k] < fval[kopt]:
            kopt = k
        if nf >= maxfun:
            return BobyqaResult(
                x_min=np.clip(xbase + xpt[kopt], lo[:n], hi[:n]),
                f_min=float(fval[kopt]),
                num_evals=nf,
            )

    h_mat = np.zeros((n, n), dtype=np.float64)
    rho = float(rhobeg)
    delta = float(rhobeg)

    while nf < maxfun:
        grad, h_mat = _newuoa.fit_min_frobenius_quad(
            n, npt, kopt, delta, xpt, fval, h_mat
        )
        d_lo = sl - xpt[kopt]
        d_hi = su - xpt[kopt]
        d = _trsbox_bounded(n, grad, h_mat, d_lo, d_hi, delta)
        dnorm = float(np.linalg.norm(d))

        if dnorm < 0.5 * rho:
            diffs = xpt - xpt[kopt]
            dsq = np.sum(diffs * diffs, axis=1)
            dsq[kopt] = -1.0
            kfar = int(np.argmax(dsq))
            max_dist_sq = float(dsq[kfar])
            dist_thresh = max((2.0 * delta) ** 2, (10.0 * rho) ** 2)
            if max_dist_sq > dist_thresh and nf < maxfun:
                norm_dir = max(math.sqrt(max_dist_sq), 1e-15)
                step = (rho / norm_dir) * (xpt[kfar] - xpt[kopt])
                xnew = np.clip(xpt[kopt] + step, sl, su)
                fnew = eval_f(xbase + xnew)
                xpt[kfar] = xnew
                fval[kfar] = fnew
                if fnew < fval[kopt]:
                    kopt = kfar
                continue
            if rho <= rhoend:
                break
            delta = 0.5 * rho
            ratio = rho / rhoend
            if ratio <= 16.0:
                rho = rhoend
            elif ratio <= 250.0:
                rho = math.sqrt(ratio) * rhoend
            else:
                rho = 0.1 * rho
            delta = max(delta, rho)
            continue

        xnew = np.clip(xpt[kopt] + d, sl, su)
        fnew = eval_f(xbase + xnew)
        fopt = float(fval[kopt])
        d_eff = xnew - xpt[kopt]
        pred = -float(np.dot(grad, d_eff) + 0.5 * np.dot(d_eff, h_mat @ d_eff))
        if pred > 1e-30:
            ratio = (fopt - fnew) / pred
        else:
            ratio = 1.0 if fnew < fopt else 0.0

        if ratio <= 0.1:
            delta = 0.5 * dnorm
        elif ratio <= 0.7:
            delta = max(0.5 * delta, dnorm)
        else:
            delta = max(0.5 * delta, 2.0 * dnorm)
        if delta <= 1.5 * rho:
            delta = rho

        ref_pt = xnew if fnew < fopt else xpt[kopt]
        dsq = np.sum((xpt - ref_pt) ** 2, axis=1)
        dsq[kopt] = -1.0
        knew = int(np.argmax(dsq))
        dsq_knew = float(np.sum((xpt[knew] - xpt[kopt]) ** 2))
        if (
            fnew < fopt
            or dsq_knew > (2.0 * delta) ** 2
            or dnorm * dnorm < dsq_knew
        ):
            xpt[knew] = xnew
            fval[knew] = fnew
            if fnew < fopt:
                kopt = knew

    return BobyqaResult(
        x_min=np.clip(xbase + xpt[kopt], lo[:n], hi[:n]),
        f_min=float(fval[kopt]),
        num_evals=nf,
    )


def _trsbox_bounded(
    n: int,
    grad: NDArray[np.float64],
    h_mat: NDArray[np.float64],
    d_lo: NDArray[np.float64],
    d_hi: NDArray[np.float64],
    delta: float,
) -> NDArray[np.float64]:
    """Projected Steihaug-Toint CG solver on ``||d||_2 <= delta``."""
    d = np.zeros(n, dtype=np.float64)
    active = np.zeros(n, dtype=bool)

    for _ in range(n):
        r = grad + h_mat @ d
        for i in range(n):
            if (d[i] <= d_lo[i] + 1e-14 and r[i] >= 0.0) or (
                d[i] >= d_hi[i] - 1e-14 and r[i] <= 0.0
            ):
                active[i] = True
        p = np.where(active, 0.0, -r)
        rr = float(np.dot(p, p))
        if rr <= 1e-30:
            break

        hit_box = False
        for _ in range(2 * n):
            hp = np.where(active, 0.0, h_mat @ p)
            p_hp = float(np.dot(p, hp))
            dd = float(np.dot(d, d))
            dp = float(np.dot(d, p))
            pp = float(np.dot(p, p))
            if pp <= 1e-30:
                break
            disc = max(dp * dp + pp * (delta * delta - dd), 0.0)
            alpha_tr = (-dp + math.sqrt(disc)) / pp

            alpha_box = math.inf
            box_idx = -1
            for i in range(n):
                if active[i]:
                    continue
                if p[i] > 1e-15:
                    a = float((d_hi[i] - d[i]) / p[i])
                    if a < alpha_box:
                        alpha_box = a
                        box_idx = i
                elif p[i] < -1e-15:
                    a = float((d_lo[i] - d[i]) / p[i])
                    if a < alpha_box:
                        alpha_box = a
                        box_idx = i

            max_step = min(alpha_tr, alpha_box)
            alpha_cg = rr / p_hp if p_hp > 1e-30 else math.inf
            if alpha_cg >= max_step:
                d = np.clip(d + max_step * p, d_lo, d_hi)
                if alpha_box <= alpha_tr and box_idx >= 0:
                    active[box_idx] = True
                    hit_box = True
                break

            d = d + alpha_cg * p
            r = r + alpha_cg * hp
            r_free = np.where(active, 0.0, r)
            rr_new = float(np.dot(r_free, r_free))
            if rr_new <= 1e-24:
                break
            p = np.where(active, 0.0, -r + (rr_new / rr) * p)
            rr = rr_new
        if not hit_box:
            break
    return d
