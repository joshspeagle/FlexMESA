# SPDX-License-Identifier: LGPL-3.0-or-later
"""Nelder-Mead simplex optimization and Park-Miller ``ran1_zah`` PRNG (``num``).

Ports ``num/private/mod_simplex.f90`` and ``num/private/mod_random.f90`` into
``float64`` Python/NumPy.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def r8_uniform_01(seed: int) -> tuple[float, int]:
    """Evaluates the Park-Miller ``ran1_zah`` uniform PRNG in $[0, 1)$.

    Args:
        seed: 32-bit integer state ($\\ne 0$; ``0`` is mapped to ``123459876``).

    Returns:
        Tuple ``(uniform_value, next_seed)``.
    """
    ia = 16_807
    im = 2_147_483_647
    iq = 127_773
    ir = 2_836
    s = 123_459_876 if seed == 0 else int(seed)
    k = int(s / iq)
    nxt = ia * (s - k * iq) - ir * k
    if nxt < 0:
        nxt += im
    u = float(nxt) * 4.656_612_875e-10
    return u, int(nxt)


def i4_uniform(a: int, b: int, seed: int) -> tuple[int, int]:
    """Draws a uniform integer in ``[min(a, b), max(a, b)]`` via ``ran1_zah``.

    Args:
        a: First integer bound.
        b: Second integer bound.
        seed: 32-bit integer PRNG state.

    Returns:
        Tuple ``(random_int, next_seed)``.
    """
    lo = min(int(a), int(b))
    hi = max(int(a), int(b))
    u, next_seed = r8_uniform_01(seed)
    span = hi - lo + 1
    offset = min(span - 1, max(0, int(u * span)))
    return lo + offset, next_seed


def perm_uniform(n: int, seed: int) -> tuple[npt.NDArray[np.int64], int]:
    """Generates a random 1-based permutation of ``1..=n`` via ``i4_uniform``.

    Args:
        n: Permutation length ($\\ge 1$).
        seed: 32-bit integer PRNG state.

    Returns:
        Tuple ``(perm_array, next_seed)`` with ``perm_array`` of shape ``(n,)``.
    """
    perm = np.arange(1, n + 1, dtype=np.int64)
    cur_seed = int(seed)
    for i in range(n):
        j, cur_seed = i4_uniform(i, max(0, n - 1), cur_seed)
        perm[i], perm[j] = perm[j], perm[i]
    return perm, cur_seed


@dataclasses.dataclass(frozen=True)
class SimplexConfig:
    """Configuration parameters for :func:`nm_simplex`.

    Attributes:
        max_evals: Maximum objective evaluations allowed.
        x_tol: Relative simplex vertex diameter tolerance.
        f_tol: Objective function spread convergence tolerance.
        alpha: Reflection coefficient (default ``1.0``).
        beta: Expansion coefficient (default ``2.0``).
        gamma: Contraction coefficient (default ``0.5``).
        delta: Shrink coefficient (default ``0.5``).
        enforce_bounds: Penalize points outside ``[x_lower, x_upper]``.
        adaptive_random_search: Use MESA's adaptive random search on
            contraction failure.
        seed: Initial PRNG seed for adaptive random search.
        centroid_weight_power: Power for inverse-objective centroid weighting.
    """

    max_evals: int = 20_000
    x_tol: float = 1e-10
    f_tol: float = 0.0
    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 0.5
    delta: float = 0.5
    enforce_bounds: bool = False
    adaptive_random_search: bool = False
    seed: int = 107_461_921
    centroid_weight_power: float = 0.0


@dataclasses.dataclass(frozen=True)
class SimplexResult:
    """Result returned by :func:`nm_simplex`.

    Attributes:
        x_min: Best parameter vector of shape ``(n,)`` in ``float64``.
        f_min: Minimum objective value $f(x_{\\min})$.
        num_evals: Total objective evaluations performed.
        num_iters: Number of Nelder-Mead iterations completed.
        num_shrinks: Number of full simplex shrink steps performed.
    """

    x_min: npt.NDArray[np.float64]
    f_min: float
    num_evals: int
    num_iters: int
    num_shrinks: int


def nm_simplex(
    x_lower: npt.ArrayLike,
    x_upper: npt.ArrayLike,
    x_first: npt.ArrayLike,
    fcn: Callable[[npt.NDArray[np.float64]], float],
    config: SimplexConfig | None = None,
) -> SimplexResult:
    """Minimizes an $n$-dimensional objective via Nelder-Mead simplex search.

    Args:
        x_lower: Lower box scale array of shape ``(n,)``.
        x_upper: Upper box scale array of shape ``(n,)``.
        x_first: Initial vertex array of shape ``(n,)``.
        fcn: Objective callable ``fcn(x)`` taking a ``(n,)`` ``float64`` array.
        config: Optional :class:`SimplexConfig` settings.

    Returns:
        :class:`SimplexResult` containing the minimizer and diagnostic counts.

    Raises:
        ValueError: If input arrays are empty or have mismatched lengths.
    """
    cfg = SimplexConfig() if config is None else config
    lo = np.asarray(x_lower, dtype=np.float64).ravel()
    hi = np.asarray(x_upper, dtype=np.float64).ravel()
    x0 = np.asarray(x_first, dtype=np.float64).ravel()
    n = int(x0.size)
    if n == 0 or lo.size != n or hi.size != n:
        raise ValueError(
            "x_lower, x_upper, x_first must have equal length >= 1"
        )

    num_evals = 0

    def _eval(pt: npt.NDArray[np.float64]) -> float:
        nonlocal num_evals
        if cfg.enforce_bounds and bool(np.any((pt < lo) | (pt > hi))):
            return 1e99
        num_evals += 1
        return float(fcn(pt))

    n_verts = n + 1
    simplex = np.tile(x0, (n_verts, 1)).astype(np.float64)
    f_vals = np.zeros(n_verts, dtype=np.float64)
    f_vals[n] = _eval(simplex[n])

    for j in range(n):
        span = float(hi[j] - lo[j])
        if abs(span) > 1e-14:
            if float(hi[j] - x0[j]) < float(x0[j] - lo[j]):
                simplex[j, j] = x0[j] - 0.25 * span
            else:
                simplex[j, j] = x0[j] + 0.25 * span
        else:
            simplex[j, j] += 0.05 * max(1.0, abs(float(x0[j])))
        f_vals[j] = _eval(simplex[j])

    num_iters = 0
    num_shrinks = 0
    seed = int(cfg.seed)

    while True:
        num_iters += 1
        h = 0
        lo_idx = 0
        for j in range(1, n_verts):
            if f_vals[j] > f_vals[h]:
                h = j
            if f_vals[j] < f_vals[lo_idx]:
                lo_idx = j
        s = -1
        for j in range(n_verts):
            if j == h:
                continue
            if s < 0 or f_vals[j] > f_vals[s]:
                s = j

        term_val_x = 0.0
        for j in range(n_verts):
            if j == lo_idx:
                continue
            denom = cfg.x_tol * (
                1.0 + np.maximum(np.abs(simplex[j]), np.abs(simplex[lo_idx]))
            )
            term1 = float(np.max(np.abs(simplex[j] - simplex[lo_idx]) / denom))
            if term1 > term_val_x:
                term_val_x = term1
        if term_val_x <= 1.0 or (
            cfg.f_tol > 0.0 and float(f_vals[h] - f_vals[lo_idx]) <= cfg.f_tol
        ):
            break
        if num_iters > cfg.max_evals or num_evals > cfg.max_evals:
            break

        if cfg.centroid_weight_power == 0.0:
            mask = np.ones(n_verts, dtype=bool)
            mask[h] = False
            centroid = np.mean(simplex[mask], axis=0)
        else:
            centroid = np.zeros(n, dtype=np.float64)
            sum_weight = 0.0
            for j in range(n_verts):
                if j == h:
                    continue
                w = (1.0 / max(1e-30, abs(float(f_vals[j])))) ** float(
                    cfg.centroid_weight_power
                )
                centroid += simplex[j] * w
                sum_weight += w
            centroid /= sum_weight

        x_ref = centroid + cfg.alpha * (centroid - simplex[h])
        f_ref = _eval(x_ref)

        if f_ref < f_vals[s]:
            simplex[h] = x_ref
            f_vals[h] = f_ref
            if f_ref < f_vals[lo_idx]:
                x_exp = centroid + cfg.beta * (x_ref - centroid)
                f_exp = _eval(x_exp)
                if f_exp < f_vals[lo_idx]:
                    simplex[h] = x_exp
                    f_vals[h] = f_exp
        else:
            outside = f_ref < f_vals[h]
            if outside:
                x_con = centroid + cfg.gamma * (x_ref - centroid)
            else:
                x_con = centroid + cfg.gamma * (simplex[h] - centroid)
            f_con = _eval(x_con)
            if f_con < min(float(f_vals[h]), f_ref):
                simplex[h] = x_con
                f_vals[h] = f_con
            elif cfg.adaptive_random_search:
                for _k in range(100):
                    if num_evals > cfg.max_evals:
                        break
                    x_ars = np.empty(n, dtype=np.float64)
                    for i in range(n):
                        u, seed = r8_uniform_01(seed)
                        x_ars[i] = lo[i] + u * (hi[i] - lo[i])
                    f_ars = _eval(x_ars)
                    if f_ars <= f_vals[h]:
                        simplex[h] = x_ars
                        f_vals[h] = f_ars
                        break
            else:
                num_shrinks += 1
                best = simplex[lo_idx].copy()
                for j in range(n_verts):
                    if j == lo_idx:
                        continue
                    simplex[j] = best + cfg.delta * (simplex[j] - best)
                    f_vals[j] = _eval(simplex[j])

    best_idx = int(np.argmin(f_vals))
    return SimplexResult(
        x_min=simplex[best_idx].copy(),
        f_min=float(f_vals[best_idx]),
        num_evals=num_evals,
        num_iters=num_iters,
        num_shrinks=num_shrinks,
    )
