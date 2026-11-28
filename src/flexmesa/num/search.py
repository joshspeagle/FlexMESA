# SPDX-License-Identifier: GPL-3.0-or-later
"""Compensated summation, binary search, and quadratic interpolation (``num``).

Ports utility solvers from upstream MESA ``num/public/num_lib.f90`` into
``float64`` NumPy/Python, including Knuth ``two_sum``, Neumaier compensated
summation, monotonic ``binary_search``, permutation index sorting
``qsort_index``, ``find0``, ``find0_quadratic``, and ``find_max_quadratic``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def two_sum(a: float, b: float) -> tuple[float, float]:
    """Computes the exact Knuth 2Sum transformation ``(sum, error)``.

    Args:
        a: First scalar summand (dimensionless).
        b: Second scalar summand (dimensionless).

    Returns:
        Tuple ``(s, e)`` such that ``s + e == a + b`` in exact real arithmetic.
    """
    s = float(a + b)
    v = float(s - a)
    e = float((a - (s - v)) + (b - v))
    return s, e


def neumaier_sum(values: npt.ArrayLike) -> float:
    """Evaluates the Neumaier compensated sum of a 1D array in ``float64``.

    Args:
        values: 1D sequence of shape ``(N,)`` (dimensionless).

    Returns:
        Compensated scalar sum in ``float64``.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    total = 0.0
    comp = 0.0
    for x_val in arr:
        x = float(x_val)
        t = total + x
        if abs(total) >= abs(x):
            comp += (total - t) + x
        else:
            comp += (x - t) + total
        total = t
    return float(total + comp)


def binary_search(
    vec: npt.ArrayLike,
    val: float,
    hint: int | None = None,
) -> int | None:
    """Locates the 0-based interval index ``k`` in a monotonic array ``vec``.

    For increasing ``vec``, returns ``None`` if ``val < vec[0]``, ``n - 1`` if
    ``val >= vec[n - 1]``, or ``k`` such that ``vec[k] <= val < vec[k + 1]``.
    For decreasing ``vec``, returns ``None`` if ``val > vec[0]``, ``n - 1`` if
    ``val <= vec[n - 1]``, or ``k`` such that ``vec[k] >= val > vec[k + 1]``.

    Args:
        vec: 1D monotonic knot vector of shape ``(n,)``.
        val: Query value (same physical units as ``vec``).
        hint: Optional 0-based interval guess probed in $O(1)$ time before
            falling back to bisection.

    Returns:
        0-based interval index ``k`` or ``None`` if out of bounds at the start.
    """
    arr = np.asarray(vec, dtype=np.float64).ravel()
    n = int(arr.size)
    if n == 0:
        return None
    if n == 1:
        return 0
    increasing = bool(arr[n - 1] >= arr[0])
    if increasing:
        if val < float(arr[0]):
            return None
        if val >= float(arr[n - 1]):
            return n - 1
        if hint is not None:
            for cand in (max(0, hint - 1), hint, hint + 1):
                if 0 <= cand < n - 1 and arr[cand] <= val < arr[cand + 1]:
                    return int(cand)
        lo = 0
        hi = n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if val >= float(arr[mid]):
                lo = mid
            else:
                hi = mid
        return lo

    if val > float(arr[0]):
        return None
    if val <= float(arr[n - 1]):
        return n - 1
    if hint is not None:
        for cand in (max(0, hint - 1), hint, hint + 1):
            if 0 <= cand < n - 1 and arr[cand] >= val > arr[cand + 1]:
                return int(cand)
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if val <= float(arr[mid]):
            lo = mid
        else:
            hi = mid
    return lo


def binary_search_1based(vec: npt.ArrayLike, val: float, hint: int = 0) -> int:
    """1-based wrapper around :func:`binary_search` matching upstream Fortran.

    Args:
        vec: 1D monotonic knot vector of shape ``(n,)``.
        val: Query value (same physical units as ``vec``).
        hint: 1-based interval hint (``0`` indicates no hint).

    Returns:
        1-based index in ``0..=n``.
    """
    hint_0 = (hint - 1) if hint >= 1 else None
    res = binary_search(vec, val, hint_0)
    return 0 if res is None else (res + 1)


def qsort_index(vals: npt.ArrayLike) -> npt.NDArray[np.intp]:
    """Returns 0-based permutation indices sorting ``vals`` ascending.

    Args:
        vals: 1D array of shape ``(n,)`` (dimensionless).

    Returns:
        Integer index array ``idx`` of shape ``(n,)`` such that ``vals[idx]``
        is non-decreasing.
    """
    arr = np.asarray(vals, dtype=np.float64).ravel()
    return np.argsort(arr, kind="mergesort")


def linear_interp(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x: float,
) -> float:
    """Evaluates linear interpolation between ``(x1, y1)`` and ``(x2, y2)``.

    Args:
        x1: First abscissa.
        y1: First ordinate.
        x2: Second abscissa.
        y2: Second ordinate.
        x: Target evaluation abscissa.

    Returns:
        Interpolated ordinate ``y(x)`` in ``float64``.
    """
    if x2 == x1:
        return 0.5 * (y1 + y2)
    return float(y1 + (y2 - y1) * (x - x1) / (x2 - x1))


def find0(x1: float, y1: float, x2: float, y2: float) -> float:
    """Finds the linear secant root between ``(x1, y1)`` and ``(x2, y2)``.

    Args:
        x1: First abscissa.
        y1: Function value at ``x1``.
        x2: Second abscissa.
        y2: Function value at ``x2`` (must have opposite sign to ``y1``).

    Returns:
        Secant zero abscissa ``x0`` in ``[min(x1, x2), max(x1, x2)]``.

    Raises:
        ValueError: If ``y1`` and ``y2`` have the same strict sign.
    """
    if not (
        np.isfinite(x1)
        and np.isfinite(y1)
        and np.isfinite(x2)
        and np.isfinite(y2)
    ):
        raise ValueError("non-finite input in find0")
    if y1 == 0.0:
        return float(x1)
    if y2 == 0.0:
        return float(x2)
    if (y1 > 0.0 and y2 > 0.0) or (y1 < 0.0 and y2 < 0.0) or y1 == y2:
        raise ValueError("y1 and y2 must bracket zero in find0")
    return float(x1 - y1 * (x2 - x1) / (y2 - y1))


def _quadratic_divided_diffs(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
) -> tuple[float, float, float]:
    """Computes Newton divided-difference coefficients ``(dx12, b, c)``."""
    if not (np.isfinite(y1) and np.isfinite(y2) and np.isfinite(y3)):
        raise ValueError("non-finite ordinate in quadratic interpolation")
    if not ((x1 < x2 < x3) or (x1 > x2 > x3)):
        raise ValueError("x1, x2, x3 must be strictly monotonic")
    dx12 = x2 - x1
    b = (y2 - y1) / dx12
    c = ((y3 - y2) / (x3 - x2) - b) / (x3 - x1)
    return dx12, b, c


def find0_quadratic(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
) -> tuple[float, float]:
    """Finds the root of the quadratic through three monotonic points.

    Args:
        x1: First abscissa (arbitrary units).
        y1: Function value at ``x1`` (arbitrary units).
        x2: Interior abscissa strictly between ``x1`` and ``x3``.
        y2: Function value at ``x2``.
        x3: Third abscissa.
        y3: Function value at ``x3`` (opposite sign to ``y1``).

    Returns:
        Tuple ``(root_x, poly_y)`` at the root inside ``[x1, x3]``.

    Raises:
        ValueError: If abscissas are not strictly monotonic or ``y1, y3`` do
            not bracket zero.
    """
    dx12, b, c = _quadratic_divided_diffs(x1, y1, x2, y2, x3, y3)
    if y1 == 0.0:
        return float(x1), 0.0
    if y2 == 0.0:
        return float(x2), 0.0
    if y3 == 0.0:
        return float(x3), 0.0
    if (y1 > 0.0 and y3 > 0.0) or (y1 < 0.0 and y3 < 0.0):
        raise ValueError("y1 and y3 must bracket zero in find0_quadratic")
    a_q = c
    b_q = b - c * dx12
    c_q = y1
    if abs(a_q) < 1e-15 * abs(b_q):
        return find0(x1, y1, x3, y3), 0.0
    disc = b_q * b_q - 4.0 * a_q * c_q
    if disc < 0.0:
        raise ValueError("negative discriminant in find0_quadratic")
    s = float(np.sqrt(disc))
    xmin = min(x1, x3)
    xmax = max(x1, x3)
    for sign in (1.0, -1.0):
        cand = x1 + (-b_q + sign * s) / (2.0 * a_q)
        if xmin - 1e-12 <= cand <= xmax + 1e-12:
            clamped = float(np.clip(cand, xmin, xmax))
            y_cand = float(
                y1 + b * (clamped - x1) + c * (clamped - x1) * (clamped - x2)
            )
            return clamped, y_cand
    raise ValueError("no quadratic root found inside [x1, x3]")


def find_max_quadratic(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
) -> tuple[float, float, int]:
    """Finds the extremum ``(xmax, ymax, ierr)`` of a 3-point quadratic.

    Mirrors ``find_max_quadratic`` in ``num/public/num_lib.f90:558-578`` in
    upstream's exact expression order and error tests: computes ``dx1``,
    ``dx2``, ``a``, ``b``, ``c``, ``dxmax``, ``xmax``, and ``ymax`` without
    a concavity or finiteness guard, then tests ``ymax < y2``,
    ``y2 < max(y1, y2)`` (deferred upstream bug ``#671``), and strict
    monotonicity of ``(x1, x2, x3)``, returning ``ierr = -1`` when any check
    fails while still returning ``(xmax, ymax)``.

    Args:
        x1: First abscissa (arbitrary units).
        y1: Function value at ``x1`` (arbitrary units).
        x2: Interior abscissa between ``x1`` and ``x3``.
        y2: Function value at ``x2`` (arbitrary units).
        x3: Third abscissa.
        y3: Function value at ``x3``.

    Returns:
        Tuple ``(xmax, ymax, ierr)`` where ``ierr`` is ``0`` on success or
        ``-1`` if any upstream validation check fails.
    """
    ierr = 0
    x1_f = np.float64(x1)
    y1_f = np.float64(y1)
    x2_f = np.float64(x2)
    y2_f = np.float64(y2)
    x3_f = np.float64(x3)
    y3_f = np.float64(y3)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dx1 = x2_f - x1_f
        dx2 = x3_f - x2_f
        # f(dx) = a + b dx + c dx^2; dx = x - x1
        # (`num/public/num_lib.f90:568-574`)
        a = y1_f
        b = (
            (y2_f - y1_f) / dx1
            + (y2_f - y1_f) / (dx1 + dx2)
            + (dx1 / dx2) * (y2_f - y3_f) / (dx1 + dx2)
        )
        c = (dx2 * y1_f - dx1 * y2_f - dx2 * y2_f + dx1 * y3_f) / (
            dx1 * dx2 * (dx1 + dx2)
        )
        dxmax = -(b / (np.float64(2.0) * c))
        xmax = float(dxmax + x1_f)
        ymax = float(a + dxmax * (b + dxmax * c))
    x1_val = float(x1_f)
    x2_val = float(x2_f)
    x3_val = float(x3_f)
    y2_val = float(y2_f)
    if ymax < y2_val:
        ierr = -1
    # Deferred upstream bug #671: `num/public/num_lib.f90:576` tests
    # `y2 < max(y1, y2)` rather than `y2 < max(y1, y3)`. Reproduce verbatim
    # until v0.1.0.
    if y2_val < float(np.fmax(y1_f, y2_f)):
        ierr = -1
    if not (
        (x1_val < x2_val and x2_val < x3_val)
        or (x1_val > x2_val and x2_val > x3_val)
    ):
        ierr = -1
    return xmax, ymax, ierr


def two_piece_linear_coeffs(
    x: float,
    x0: float,
    x1: float,
    x2: float,
) -> tuple[float, float, float]:
    """Solves for two-piece linear weights ``(a0, a1, a2)`` on ``(x0, x1, x2)``.

    Mirrors ``two_piece_linear_coeffs`` in MESA ``num/public/num_lib.f90``
    (lines 580-613) in upstream's order of operations: the weight of the far
    node on the segment holding ``x`` is computed first and clipped to
    ``[0, 1]``, and the near node takes one minus that weight. A query
    outside ``[x0, x2]`` takes the nearest endpoint's coefficients.

    Args:
        x: Query point (arbitrary units).
        x0: First grid abscissa (arbitrary units).
        x1: Middle grid abscissa (arbitrary units).
        x2: Third grid abscissa (arbitrary units).

    Returns:
        Tuple ``(a0, a1, a2)`` such that
        $f(x) = a_0 f(x_0) + a_1 f(x_1) + a_2 f(x_2)$.

    Raises:
        ValueError: If ``(x0, x1, x2)`` is not strictly monotonic, or if
            ``x`` is NaN (upstream leaves the coefficients unset).
    """
    if x0 < x1 < x2:
        if x <= x0:
            return 1.0, 0.0, 0.0
        if x >= x2:
            return 0.0, 0.0, 1.0
        if x <= x1:
            a1 = min(1.0, max(0.0, (x - x0) / (x1 - x0)))
            return 1.0 - a1, a1, 0.0
        if x < x2:
            a2 = min(1.0, max(0.0, (x - x1) / (x2 - x1)))
            return 0.0, 1.0 - a2, a2
    elif x0 > x1 > x2:
        if x >= x0:
            return 1.0, 0.0, 0.0
        if x <= x2:
            return 0.0, 0.0, 1.0
        if x >= x1:
            a1 = min(1.0, max(0.0, (x - x0) / (x1 - x0)))
            return 1.0 - a1, a1, 0.0
        if x > x2:
            a2 = min(1.0, max(0.0, (x - x1) / (x2 - x1)))
            return 0.0, 1.0 - a2, a2
    else:
        raise ValueError("(x0, x1, x2) not strictly monotonic")
    raise ValueError("x is NaN; upstream leaves the coefficients unset")
