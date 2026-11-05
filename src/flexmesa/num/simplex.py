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
    s = 123_459_876 if seed == 0 else max(1, abs(int(seed)))
    k = s // iq
    nxt = ia * (s - k * iq) - ir * k
    if nxt < 0:
        nxt += im
    u = min(1.0 - float(np.finfo(np.float64).eps), float(nxt) / float(im))
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
    """

    max_evals: int = 2_000
    x_tol: float = 1e-10
    f_tol: float = 1e-12
    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 0.5
    delta: float = 0.5
    enforce_bounds: bool = False


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
        num_evals += 1
        if cfg.enforce_bounds and bool(np.any((pt < lo) | (pt > hi))):
            return 1e30
        return float(fcn(pt))

    n_verts = n + 1
    simplex = np.zeros((n_verts, n), dtype=np.float64)
    f_vals = np.zeros(n_verts, dtype=np.float64)
    simplex[0] = x0
    f_vals[0] = _eval(x0)

    for j in range(n):
        v = x0.copy()
        span = max(1.0, abs(float(hi[j] - lo[j])))
        vj_abs = abs(float(v[j]))
        step = 0.05 * vj_abs if vj_abs > 1e-6 else 0.05 * span
        if v[j] + step <= hi[j] or not cfg.enforce_bounds:
            v[j] += step
        else:
            v[j] -= step
        simplex[j + 1] = v
        f_vals[j + 1] = _eval(v)

    num_iters = 0
    num_shrinks = 0

    while num_evals < cfg.max_evals:
        order = np.argsort(f_vals, kind="mergesort")
        simplex = simplex[order]
        f_vals = f_vals[order]

        f_spread = abs(float(f_vals[n] - f_vals[0]))
        scales = np.maximum(1.0, np.abs(simplex[0]))
        x_diam = float(np.max(np.abs(simplex[1:] - simplex[0]) / scales))
        if f_spread <= cfg.f_tol and x_diam <= cfg.x_tol:
            break
        num_iters += 1

        centroid = np.mean(simplex[:n], axis=0)
        x_ref = centroid + cfg.alpha * (centroid - simplex[n])
        f_ref = _eval(x_ref)

        if f_ref < f_vals[0]:
            x_exp = centroid + cfg.beta * (x_ref - centroid)
            f_exp = _eval(x_exp)
            if f_exp < f_ref:
                simplex[n] = x_exp
                f_vals[n] = f_exp
            else:
                simplex[n] = x_ref
                f_vals[n] = f_ref
        elif f_ref < f_vals[n - 1]:
            simplex[n] = x_ref
            f_vals[n] = f_ref
        else:
            outside = f_ref < f_vals[n]
            if outside:
                x_con = centroid + cfg.gamma * (x_ref - centroid)
            else:
                x_con = centroid + cfg.gamma * (simplex[n] - centroid)
            f_con = _eval(x_con)
            if (outside and f_con <= f_ref) or (
                not outside and f_con < f_vals[n]
            ):
                simplex[n] = x_con
                f_vals[n] = f_con
            else:
                num_shrinks += 1
                best = simplex[0].copy()
                for k in range(1, n_verts):
                    simplex[k] = best + cfg.delta * (simplex[k] - best)
                    f_vals[k] = _eval(simplex[k])

    best_idx = int(np.argmin(f_vals))
    return SimplexResult(
        x_min=simplex[best_idx].copy(),
        f_min=float(f_vals[best_idx]),
        num_evals=num_evals,
        num_iters=num_iters,
        num_shrinks=num_shrinks,
    )
