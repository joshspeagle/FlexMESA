# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compensated summation, binary search, and quadratic interpolation (``num``).

Ports utility solvers from upstream MESA ``num/public/num_lib.f90`` into
``float64`` NumPy/Python, including Knuth ``two_sum``, Neumaier compensated
summation, monotonic ``binary_search``, permutation index sorting
``qsort_index``, ``find0``, ``find0_quadratic``, and ``find_max_quadratic``
(with the upstream concavity check bug fixed).
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
    if y1 == 0.0:
        return float(x1)
    if y2 == 0.0:
        return float(x2)
    if (y1 > 0.0 and y2 > 0.0) or (y1 < 0.0 and y2 < 0.0) or y1 == y2:
        raise ValueError("y1 and y2 must bracket zero in find0")
    return float(x1 - y1 * (x2 - x1) / (y2 - y1))


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
        x1: First abscissa.
        y1: Function value at ``x1``.
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
    if not ((x1 < x2 < x3) or (x1 > x2 > x3)):
        raise ValueError("x1, x2, x3 must be strictly monotonic")
    if y1 == 0.0:
        return float(x1), 0.0
    if y2 == 0.0:
        return float(x2), 0.0
    if y3 == 0.0:
        return float(x3), 0.0
    if (y1 > 0.0 and y3 > 0.0) or (y1 < 0.0 and y3 < 0.0):
        raise ValueError("y1 and y3 must bracket zero in find0_quadratic")
    dx12 = x2 - x1
    dx23 = x3 - x2
    dx13 = x3 - x1
    b = (y2 - y1) / dx12
    c = ((y3 - y2) / dx23 - b) / dx13
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
) -> tuple[float, float]:
    """Finds the maximum ``(xmax, ymax)`` of a concave quadratic interpolant.

    Fixes the upstream ``num_lib.f90`` line 581 typo (``y2 < max(y1, y2)``) by
    enforcing ``y2 >= max(y1, y3)`` and strict concavity ``c < 0``.

    Args:
        x1: First abscissa.
        y1: Function value at ``x1``.
        x2: Interior abscissa strictly between ``x1`` and ``x3``.
        y2: Function value at ``x2`` (must satisfy ``y2 >= max(y1, y3)``).
        x3: Third abscissa.
        y3: Function value at ``x3``.

    Returns:
        Tuple ``(xmax, ymax)`` at the stationary maximum of the parabola.

    Raises:
        ValueError: If abscissas are not monotonic, ``y2 < max(y1, y3)``, or
            the quadratic is not strictly concave.
    """
    if not ((x1 < x2 < x3) or (x1 > x2 > x3)):
        raise ValueError("x1, x2, x3 must be strictly monotonic")
    if y2 < max(y1, y3):
        raise ValueError("interior value y2 must be >= max(y1, y3)")
    dx1 = x2 - x1
    dx2 = x3 - x2
    dx12 = x3 - x1
    b = (y2 - y1) / dx1
    c = ((y3 - y2) / dx2 - b) / dx12
    if c >= 0.0:
        raise ValueError("quadratic is not strictly concave (c >= 0)")
    dxmax = 0.5 * (dx1 - b / c)
    xmax = float(x1 + dxmax)
    ymax = float(y1 + dxmax * b + dxmax * (xmax - x2) * c)
    return xmax, ymax
