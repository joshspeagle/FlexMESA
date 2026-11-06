# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fixed-point stencil remeshing and ``DualOrder1`` 1D spline evaluation.

Provides hydrostatic Lagrangian remeshing stencils (`interp_3_to_1`,
`interp_3_to_2`, `interp_4_to_1`), convenience vector interpolation wrappers
(`interpolate_vector`, `interpolate_vector_pm`), and first-order forward-mode
`DualOrder1` spline evaluation (`interp_values_autodiff`,
`interpolate_vector_pm_autodiff`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np

from flexmesa import auto_diff
from flexmesa.interp_1d import eval as _eval
from flexmesa.interp_1d import pm


def interpolate_vector(
    old_x: np.ndarray | jnp.ndarray,
    old_y: np.ndarray | jnp.ndarray,
    new_x: np.ndarray | jnp.ndarray,
    builder: Callable[
        [np.ndarray | jnp.ndarray, np.ndarray | jnp.ndarray],
        np.ndarray | jnp.ndarray,
    ] = pm.interp_pm,
) -> np.ndarray | jnp.ndarray:
    """Interpolates ``(old_x, old_y)`` onto ``new_x`` using ``builder``.

    Args:
        old_x: Source knot coordinates of shape ``(N,)`` (``N >= 2``), strictly
            monotonic, in physical or dimensionless units.
        old_y: Source knot values of shape ``(N,)`` in physical or
            dimensionless units.
        new_x: Target query coordinates of shape ``(M,)`` in physical or
            dimensionless units.
        builder: Spline builder callable returning an ``(N, 4)`` coefficient
            array (default :func:`flexmesa.interp_1d.pm.interp_pm`).

    Returns:
        ``float64`` array of shape ``(M,)`` containing interpolated values
        at ``new_x``.

    Raises:
        ValueError: If ``old_x``, ``old_y``, or ``new_x`` fail validation.
    """
    coeffs = builder(old_x, old_y)
    return _eval.interp_values(old_x, coeffs, new_x)


def interpolate_vector_pm(
    old_x: np.ndarray | jnp.ndarray,
    old_y: np.ndarray | jnp.ndarray,
    new_x: np.ndarray | jnp.ndarray,
) -> np.ndarray | jnp.ndarray:
    """Interpolates ``(old_x, old_y)`` onto ``new_x`` using ``interp_pm``.

    Args:
        old_x: Source knot coordinates of shape ``(N,)`` (``N >= 2``), strictly
            monotonic, in physical or dimensionless units.
        old_y: Source knot values of shape ``(N,)`` in physical or
            dimensionless units.
        new_x: Target query coordinates of shape ``(M,)`` in physical or
            dimensionless units.

    Returns:
        ``float64`` array of shape ``(M,)`` containing Steffen (1990)
        interpolated values at ``new_x``.

    Raises:
        ValueError: If ``old_x``, ``old_y``, or ``new_x`` fail validation.
    """
    return interpolate_vector(old_x, old_y, new_x, builder=pm.interp_pm)


def _is_finite_dual(pt: auto_diff.DualOrder1) -> bool:
    """Returns ``True`` if both ``pt.val`` and ``pt.d1`` are finite."""
    return bool(np.isfinite(pt.val) and np.all(np.isfinite(pt.d1)))


def _is_finite_dual_row(
    row: tuple[
        auto_diff.DualOrder1,
        auto_diff.DualOrder1,
        auto_diff.DualOrder1,
        auto_diff.DualOrder1,
    ],
) -> bool:
    """Returns ``True`` if all 4 ``DualOrder1`` entries in ``row`` are valid."""
    return all(_is_finite_dual(c) for c in row)


def interp_values_autodiff(
    init_x: Sequence[auto_diff.DualOrder1],
    coeffs: Sequence[
        tuple[
            auto_diff.DualOrder1,
            auto_diff.DualOrder1,
            auto_diff.DualOrder1,
            auto_diff.DualOrder1,
        ]
    ],
    x: Sequence[auto_diff.DualOrder1],
) -> list[auto_diff.DualOrder1]:
    """Evaluates a 1D cubic spline over ``DualOrder1`` queries (`O(M log N)`).

    Args:
        init_x: Sequence of ``N >= 1`` strictly monotonic ``DualOrder1`` knots
            in physical or dimensionless units.
        coeffs: Sequence of ``N`` 4-tuples ``(c0, c1, c2, c3)`` of
            ``DualOrder1`` spline coefficients.
        x: Sequence of ``M`` ``DualOrder1`` query abscissas in physical or
            dimensionless units.

    Returns:
        List of ``M`` interpolated ``DualOrder1`` values.

    Raises:
        ValueError: If ``init_x`` and ``coeffs`` have mismatched or zero
            length, ``init_x`` is not strictly monotonic, or any input contains
            non-finite values or derivatives.
    """
    nx = len(init_x)
    if nx != len(coeffs) or nx == 0:
        raise ValueError("init_x and coeffs must have matching length >= 1.")
    if (
        any(not _is_finite_dual(pt) for pt in init_x)
        or any(not _is_finite_dual(pt) for pt in x)
        or any(not _is_finite_dual_row(row) for row in coeffs)
    ):
        raise ValueError(
            "init_x, coeffs, and x must contain finite DualOrder1 values."
        )
    if nx == 1:
        return [coeffs[0][0] for _ in x]

    x_primal = np.array([pt.val for pt in init_x], dtype=np.float64)
    pm.validate_1d_grid(x_primal, x_primal, min_points=2)
    increasing = bool(x_primal[-1] > x_primal[0])
    grid_asc = x_primal if increasing else x_primal[::-1]
    lo_knot, hi_knot = (
        (init_x[0], init_x[-1]) if increasing else (init_x[-1], init_x[0])
    )
    lo_val, hi_val = lo_knot.val, hi_knot.val
    out: list[auto_diff.DualOrder1] = []

    for xq in x:
        if xq.val < lo_val:
            x_clamped = lo_knot
        elif xq.val > hi_val:
            x_clamped = hi_knot
        else:
            x_clamped = xq

        pos = int(np.searchsorted(grid_asc, x_clamped.val, side="right")) - 1
        idx_asc = max(0, min(pos, nx - 2))
        idx = idx_asc if increasing else (nx - 2) - idx_asc
        dx = x_clamped - init_x[idx]
        c0, c1, c2, c3 = coeffs[idx]
        out.append(c0 + dx * (c1 + dx * (c2 + dx * c3)))
    return out


def interpolate_vector_pm_autodiff(
    old_x: Sequence[auto_diff.DualOrder1],
    old_y: Sequence[auto_diff.DualOrder1],
    new_x: Sequence[auto_diff.DualOrder1],
) -> list[auto_diff.DualOrder1]:
    """Interpolates ``DualOrder1`` vectors via Steffen (1990) ``interp_pm``.

    Args:
        old_x: Sequence of ``N >= 2`` strictly monotonic ``DualOrder1`` knot
            coordinates in physical or dimensionless units.
        old_y: Sequence of ``N`` ``DualOrder1`` knot values in physical or
            dimensionless units.
        new_x: Sequence of ``M`` ``DualOrder1`` target query coordinates.

    Returns:
        List of ``M`` interpolated ``DualOrder1`` values at ``new_x``.

    Raises:
        ValueError: If spline construction or evaluation fails validation.
    """
    coeffs = pm.interp_pm_autodiff(old_x, old_y)
    return interp_values_autodiff(old_x, coeffs, new_x)


def interp_3_to_1(
    pdqm1: float,
    pdq00: float,
    ndqm1: float,
    pfm1: float,
    pf00: float,
    pfp1: float,
) -> float:
    """Remeshes 3 points spaced by ``(pdqm1, pdq00)`` to offset ``ndqm1``.

    Args:
        pdqm1: First source cell spacing in Lagrangian coordinate units.
        pdq00: Second source cell spacing in Lagrangian coordinate units.
        ndqm1: Target offset from the first knot in Lagrangian coordinate
            units.
        pfm1: Value at the first knot in physical or dimensionless units.
        pf00: Value at the second knot in physical or dimensionless units.
        pfp1: Value at the third knot in physical or dimensionless units.

    Returns:
        Interpolated scalar ``float64`` value at offset ``ndqm1``.

    Raises:
        ValueError: If source spacings are non-monotonic or inputs are
            non-finite.
    """
    x_old = np.array([0.0, pdqm1, pdqm1 + pdq00], dtype=np.float64)
    v_old = np.array([pfm1, pf00, pfp1], dtype=np.float64)
    x_new = np.array([ndqm1], dtype=np.float64)
    return float(interpolate_vector_pm(x_old, v_old, x_new)[0])


def interp_3_to_2(
    pdqm1: float,
    pdq00: float,
    ndqm1: float,
    ndq00: float,
    pfm1: float,
    pf00: float,
    pfp1: float,
) -> tuple[float, float]:
    """Remeshes 3 points spaced by ``(pdqm1, pdq00)`` onto 2 target offsets.

    Args:
        pdqm1: First source cell spacing in Lagrangian coordinate units.
        pdq00: Second source cell spacing in Lagrangian coordinate units.
        ndqm1: First target cell spacing from the origin.
        ndq00: Second target cell spacing (evaluated at ``ndqm1 + ndq00``).
        pfm1: Value at the first knot in physical or dimensionless units.
        pf00: Value at the second knot in physical or dimensionless units.
        pfp1: Value at the third knot in physical or dimensionless units.

    Returns:
        Tuple ``(nf00, nfp1)`` of interpolated ``float64`` values at ``ndqm1``
        and ``ndqm1 + ndq00``.

    Raises:
        ValueError: If source spacings are non-monotonic or inputs are
            non-finite.
    """
    x_old = np.array([0.0, pdqm1, pdqm1 + pdq00], dtype=np.float64)
    v_old = np.array([pfm1, pf00, pfp1], dtype=np.float64)
    x_new = np.array([ndqm1, ndqm1 + ndq00], dtype=np.float64)
    res = interpolate_vector_pm(x_old, v_old, x_new)
    return float(res[0]), float(res[1])


def interp_4_to_1(
    pdqm1: float,
    pdq00: float,
    pdqp1: float,
    ndq00: float,
    pfm1: float,
    pf00: float,
    pfp1: float,
    pfp2: float,
) -> float:
    """Remeshes 4 points spaced by ``(pdqm1, pdq00, pdqp1)`` to ``ndq00``.

    Args:
        pdqm1: First source cell spacing in Lagrangian coordinate units.
        pdq00: Second source cell spacing in Lagrangian coordinate units.
        pdqp1: Third source cell spacing in Lagrangian coordinate units.
        ndq00: Target offset from the origin in Lagrangian coordinate units.
        pfm1: Value at the first knot in physical or dimensionless units.
        pf00: Value at the second knot in physical or dimensionless units.
        pfp1: Value at the third knot in physical or dimensionless units.
        pfp2: Value at the fourth knot in physical or dimensionless units.

    Returns:
        Interpolated scalar ``float64`` value at offset ``ndq00``.

    Raises:
        ValueError: If source spacings are non-monotonic or inputs are
            non-finite.
    """
    x_old = np.array(
        [0.0, pdqm1, pdqm1 + pdq00, pdqm1 + pdq00 + pdqp1],
        dtype=np.float64,
    )
    v_old = np.array([pfm1, pf00, pfp1, pfp2], dtype=np.float64)
    x_new = np.array([ndq00], dtype=np.float64)
    return float(interpolate_vector_pm(x_old, v_old, x_new)[0])
