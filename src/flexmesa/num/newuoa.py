# SPDX-License-Identifier: LGPL-3.0-or-later
"""Powell (`2006`) unconstrained quadratic-interpolation optimizer (``NEWUOA``).

Ports MESA's ``num/private/mod_newuoa.f`` unconstrained derivative-free
trust-region optimizer into ``float64`` NumPy.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclasses.dataclass(frozen=True)
class NewuoaResult:
    """Result returned by :func:`newuoa`.

    Attributes:
        x_min: Best parameter vector ``(n,)`` in ``float64``.
        f_min: Minimum objective value ``f(x_min)``.
        num_evals: Total number of objective evaluations performed.
    """

    x_min: NDArray[np.float64]
    f_min: float
    num_evals: int


def newuoa(
    calfun: Callable[[NDArray[np.float64]], float],
    x0: ArrayLike,
    npt: int | None = None,
    rhobeg: float = 1e-1,
    rhoend: float = 1e-6,
    maxfun: int = 5000,
    max_valid_value: float = 1e99,
) -> NewuoaResult:
    """Minimizes an unconstrained objective via Powell's ``NEWUOA`` algorithm.

    Args:
        calfun: Objective callable ``f(x) -> float`` accepting a 1D ``(n,)``
            ``float64`` array.
        x0: Initial parameter vector of shape ``(n,)`` with ``n >= 2``.
        npt: Number of interpolation conditions in
            ``[n + 2, (n + 1)(n + 2) // 2]`` (defaults to ``2 * n + 1``).
        rhobeg: Initial trust-region radius (``> 0``).
        rhoend: Final trust-region radius (``0 < rhoend <= rhobeg``).
        maxfun: Maximum number of objective evaluations allowed.
        max_valid_value: Upper bound on acceptable objective values.

    Returns:
        :class:`NewuoaResult` with best point ``x_min``, value ``f_min``, and
        evaluation count ``num_evals``.

    Raises:
        ValueError: If ``n < 2``, ``npt`` is out of bounds, or ``rhobeg`` /
            ``rhoend`` are non-positive.
    """
    x_init = np.asarray(x0, dtype=np.float64).ravel().copy()
    n = int(x_init.size)
    if npt is None:
        npt = 2 * n + 1
    max_npt = (n + 1) * (n + 2) // 2
    if (
        n < 2
        or npt < n + 2
        or npt > max_npt
        or rhobeg <= 0.0
        or rhoend <= 0.0
        or rhoend > rhobeg
    ):
        raise ValueError(
            "require n >= 2, valid npt, and 0 < rhoend <= rhobeg in newuoa"
        )

    nf = 0

    def eval_f(pt: NDArray[np.float64]) -> float:
        nonlocal nf
        nf += 1
        val = float(calfun(pt))
        if not math.isfinite(val):
            return min(max_valid_value, 1e99)
        return min(val, max_valid_value)

    xbase = x_init.copy()
    xpt = np.zeros((npt, n), dtype=np.float64)
    fval = np.zeros(npt, dtype=np.float64)
    kopt = 0

    for k in range(npt):
        if 1 <= k <= n:
            xpt[k, k - 1] = rhobeg
        elif n < k <= 2 * n:
            xpt[k, k - n - 1] = -rhobeg
        elif k > 2 * n:
            nfmm = k - n
            itemp = (nfmm - 1) // n
            jpt = k - itemp * n - n
            ipt = jpt + itemp
            if ipt > n:
                ipt, jpt = jpt, ipt - n
            xipt = -rhobeg if fval[ipt + n] < fval[ipt] else rhobeg
            xjpt = -rhobeg if fval[jpt + n] < fval[jpt] else rhobeg
            xpt[k, ipt - 1] = xipt
            xpt[k, jpt - 1] = xjpt
        fval[k] = eval_f(xbase + xpt[k])
        if k == 0 or fval[k] < fval[kopt]:
            kopt = k
        if nf >= maxfun:
            return NewuoaResult(
                x_min=xbase + xpt[kopt],
                f_min=float(fval[kopt]),
                num_evals=nf,
            )

    h_mat = np.zeros((n, n), dtype=np.float64)
    rho = float(rhobeg)
    delta = float(rhobeg)

    while nf < maxfun:
        grad, h_mat = fit_min_frobenius_quad(
            n, npt, kopt, delta, xpt, fval, h_mat
        )
        d = _trsbox_unconstrained(n, grad, h_mat, delta)
        dnorm = float(np.linalg.norm(d))

        if dnorm < 0.5 * rho:
            diffs = xpt - xpt[kopt]
            dsq = np.sum(diffs * diffs, axis=1)
            dsq[kopt] = -1.0
            kfar = int(np.argmax(dsq))
            max_dist_sq = float(dsq[kfar])
            dist_thresh = max((2.0 * delta) ** 2, (10.0 * rho) ** 2)
            if max_dist_sq > dist_thresh and nf < maxfun:
                dir_vec = xpt[kfar] - xpt[kopt]
                norm_dir = max(float(np.linalg.norm(dir_vec)), 1e-15)
                xnew = xpt[kopt] + (rho / norm_dir) * dir_vec
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

        xnew = xpt[kopt] + d
        fnew = eval_f(xbase + xnew)
        fopt = float(fval[kopt])
        pred = -float(np.dot(grad, d) + 0.5 * np.dot(d, h_mat @ d))
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

    return NewuoaResult(
        x_min=xbase + xpt[kopt],
        f_min=float(fval[kopt]),
        num_evals=nf,
    )


def fit_min_frobenius_quad(
    n: int,
    npt: int,
    kopt: int,
    scale: float,
    xpt: NDArray[np.float64],
    fval: NDArray[np.float64],
    h_prev: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fits the minimum-Frobenius-norm quadratic model around ``xpt[kopt]``."""
    m = npt - 1
    dim = m + n
    inv_s = 1.0 / max(scale, 1e-12)
    mask = np.arange(npt) != kopt
    dx = xpt[mask] - xpt[kopt]
    s_rows = dx * inv_s
    quad_prev = np.einsum("ri,ij,rj->r", dx, h_prev, dx)
    rhs = np.zeros(dim, dtype=np.float64)
    rhs[:m] = (fval[mask] - fval[kopt]) - 0.5 * quad_prev

    kkt = np.zeros((dim, dim), dtype=np.float64)
    dots = s_rows @ s_rows.T
    kkt[:m, :m] = 0.5 * (dots * dots) + 1e-11 * np.eye(m, dtype=np.float64)
    kkt[:m, m:] = s_rows
    kkt[m:, :m] = s_rows.T
    kkt[m:, m:] = -1e-12 * np.eye(n, dtype=np.float64)

    try:
        sol = np.linalg.solve(kkt, rhs)
        grad = sol[m:] * inv_s
        lam = sol[:m] * (inv_s * inv_s)
        h_new = h_prev + np.einsum("r,ri,rj->ij", lam, s_rows, s_rows)
        return grad, h_new
    except np.linalg.LinAlgError:
        return np.zeros(n, dtype=np.float64), h_prev.copy()


def _trsbox_unconstrained(
    n: int,
    grad: NDArray[np.float64],
    h_mat: NDArray[np.float64],
    delta: float,
) -> NDArray[np.float64]:
    """Steihaug-Toint truncated CG solver on ``||d||_2 <= delta``."""
    d = np.zeros(n, dtype=np.float64)
    r = grad.copy()
    p = -r
    rr = float(np.dot(r, r))
    if rr <= 1e-30:
        return d
    grad_norm = math.sqrt(rr)

    for _ in range(2 * n):
        hp = h_mat @ p
        p_hp = float(np.dot(p, hp))
        dd = float(np.dot(d, d))
        dp = float(np.dot(d, p))
        pp = float(np.dot(p, p))
        disc = max(dp * dp + pp * (delta * delta - dd), 0.0)
        tau = (-dp + math.sqrt(disc)) / max(pp, 1e-30)

        if p_hp <= 1e-30 or rr / p_hp >= tau:
            return d + tau * p
        alpha = rr / p_hp
        d = d + alpha * p
        r = r + alpha * hp
        rr_new = float(np.dot(r, r))
        if math.sqrt(rr_new) <= 1e-12 * grad_norm:
            break
        p = -r + (rr_new / rr) * p
        rr = rr_new
    return d
