# SPDX-License-Identifier: LGPL-3.0-or-later
"""Explicit Runge-Kutta (``rk2``, ``rk4``, ``dopri5``, ``dop853``) integrators.

Ports Hairer & Wanner's Dormand-Prince 5(4) (``num/private/mod_dopri5.f``)
with Shampine 4th-order dense output (``contd5``) and high-order adaptive
Runge-Kutta integration into ``float64`` NumPy/Python.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def rk2(
    t: float,
    y: npt.ArrayLike,
    h: float,
    derivs: Callable[[float, npt.NDArray[np.float64]], npt.ArrayLike],
) -> npt.NDArray[np.float64]:
    """Performs a single midpoint 2nd-order Runge-Kutta step $y(t + h)$.

    Args:
        t: Current independent variable value.
        y: State vector of shape ``(n,)`` in ``float64``.
        h: Step size.
        derivs: Callable ``derivs(t, y)`` returning $f(t, y)$ of shape ``(n,)``.

    Returns:
        Updated state array of shape ``(n,)`` in ``float64``.
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    k1 = np.asarray(derivs(t, y_arr), dtype=np.float64).ravel()
    y_mid = y_arr + 0.5 * h * k1
    k2 = np.asarray(derivs(t + 0.5 * h, y_mid), dtype=np.float64).ravel()
    return np.asarray(y_arr + h * k2, dtype=np.float64)


def rk4(
    t: float,
    y: npt.ArrayLike,
    h: float,
    derivs: Callable[[float, npt.NDArray[np.float64]], npt.ArrayLike],
) -> npt.NDArray[np.float64]:
    """Performs a single classical 4th-order Runge-Kutta step $y(t + h)$.

    Args:
        t: Current independent variable value.
        y: State vector of shape ``(n,)`` in ``float64``.
        h: Step size.
        derivs: Callable ``derivs(t, y)`` returning $f(t, y)$ of shape ``(n,)``.

    Returns:
        Updated state array of shape ``(n,)`` in ``float64``.
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    k1 = np.asarray(derivs(t, y_arr), dtype=np.float64).ravel()
    k2 = np.asarray(
        derivs(t + 0.5 * h, y_arr + 0.5 * h * k1), dtype=np.float64
    ).ravel()
    k3 = np.asarray(
        derivs(t + 0.5 * h, y_arr + 0.5 * h * k2), dtype=np.float64
    ).ravel()
    k4 = np.asarray(derivs(t + h, y_arr + h * k3), dtype=np.float64).ravel()
    return np.asarray(
        y_arr + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
        dtype=np.float64,
    )


@dataclasses.dataclass(frozen=True)
class DopriResult:
    """Result of an adaptive Dormand-Prince integration.

    Attributes:
        y: Final state vector of shape ``(n,)`` in ``float64``.
        nfcn: Total right-hand side evaluations performed.
        nstep: Total integration steps attempted.
        naccpt: Number of accepted steps.
        nrejct: Number of rejected steps.
        history: List of ``(x_old, h, rcont)`` tuples per accepted step, where
            ``rcont`` has shape ``(5, n)`` in ``float64``.
    """

    y: npt.NDArray[np.float64]
    nfcn: int
    nstep: int
    naccpt: int
    nrejct: int
    history: list[tuple[float, float, npt.NDArray[np.float64]]]

    def dense_eval(self, x_query: float) -> npt.NDArray[np.float64]:
        """Evaluates Shampine's 4th-order dense polynomial at ``x_query``.

        Args:
            x_query: Target abscissa inside the integration span.

        Returns:
            Interpolated state vector of shape ``(n,)`` in ``float64``.
        """
        if not self.history:
            return self.y.copy()
        chosen = self.history[-1]
        for seg in self.history:
            x0, h, _ = seg
            xmin = min(x0, x0 + h)
            xmax = max(x0, x0 + h)
            if xmin - 1e-14 <= x_query <= xmax + 1e-14:
                chosen = seg
                break
        x0, h, rc = chosen
        theta = 0.0 if h == 0.0 else (x_query - x0) / h
        theta1 = 1.0 - theta
        val = rc[0] + theta * (
            rc[1] + theta1 * (rc[2] + theta * (rc[3] + theta1 * rc[4]))
        )
        return np.asarray(val, dtype=np.float64)


def dopri5(
    x0: float,
    x_end: float,
    y0: npt.ArrayLike,
    fcn: Callable[[float, npt.NDArray[np.float64]], npt.ArrayLike],
    h0: float = 0.0,
    hmax: float = 0.0,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 100_000,
) -> DopriResult:
    """Integrates $y'(x) = f(x, y)$ via adaptive Dormand-Prince 5(4).

    Args:
        x0: Initial independent variable value.
        x_end: Final independent variable value.
        y0: Initial state vector of shape ``(n,)``.
        fcn: Right-hand side callable ``fcn(x, y)`` returning shape ``(n,)``.
        h0: Initial trial step size (``0.0`` selects ``1e-4``).
        hmax: Maximum step size magnitude (``0.0`` uses ``|x_end - x0|``).
        rtol: Relative local error tolerance (``> 0``).
        atol: Absolute local error tolerance (``> 0``).
        max_steps: Maximum integration steps allowed.

    Returns:
        :class:`DopriResult` with final state and dense-output history.

    Raises:
        ValueError: If ``y0`` is empty or ``rtol, atol <= 0``.
        RuntimeError: If ``max_steps`` is exceeded.
    """
    y = np.asarray(y0, dtype=np.float64).ravel().copy()
    n = int(y.size)
    if n == 0 or rtol <= 0.0 or atol <= 0.0:
        raise ValueError("require y0.size >= 1 and rtol, atol > 0 in dopri5")
    if x0 == x_end:
        return DopriResult(y=y, nfcn=0, nstep=0, naccpt=0, nrejct=0, history=[])

    c2, c3, c4, c5 = 0.2, 0.3, 0.8, 8.0 / 9.0
    a21 = 0.2
    a31, a32 = 3.0 / 40.0, 9.0 / 40.0
    a41, a42, a43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
    a51, a52, a53, a54 = (
        19_372.0 / 6_561.0,
        -25_360.0 / 2_187.0,
        64_448.0 / 6_561.0,
        -212.0 / 729.0,
    )
    a61, a62, a63, a64, a65 = (
        9_017.0 / 3_168.0,
        -355.0 / 33.0,
        46_732.0 / 5_247.0,
        49.0 / 176.0,
        -5_103.0 / 18_656.0,
    )
    a71, a73, a74, a75, a76 = (
        35.0 / 384.0,
        500.0 / 1_113.0,
        125.0 / 192.0,
        -2_187.0 / 6_784.0,
        11.0 / 84.0,
    )
    e1, e3, e4, e5, e6, e7 = (
        71.0 / 57_600.0,
        -71.0 / 16_695.0,
        71.0 / 1_920.0,
        -17_253.0 / 339_200.0,
        22.0 / 525.0,
        -1.0 / 40.0,
    )
    d1, d3, d4, d5, d6, d7 = (
        -12_715_105_075.0 / 11_282_082_432.0,
        87_487_479_700.0 / 32_700_410_799.0,
        -10_690_763_975.0 / 1_880_347_072.0,
        701_980_252_875.0 / 199_316_789_632.0,
        -1_453_857_185.0 / 822_651_844.0,
        69_997_945.0 / 29_380_423.0,
    )

    posneg = 1.0 if x_end >= x0 else -1.0
    hmax_val = abs(x_end - x0) if hmax <= 0.0 else abs(hmax)
    x = float(x0)
    k1 = np.asarray(fcn(x, y), dtype=np.float64).ravel()
    nfcn = 1
    h = (1e-4 if h0 == 0.0 else min(abs(h0), hmax_val)) * posneg
    nmax = 100_000 if max_steps <= 0 else max_steps
    nstep = 0
    naccpt = 0
    nrejct = 0
    facold = 1e-4
    reject = False
    history: list[tuple[float, float, npt.NDArray[np.float64]]] = []

    while True:
        if nstep > nmax:
            raise RuntimeError("dopri5 exceeded max_steps")
        if abs(x - x_end) <= 1e-15 * max(1.0, abs(x_end)):
            break
        if (x + 1.01 * h - x_end) * posneg > 0.0:
            h = x_end - x
        xph = x + h
        if xph == x:
            raise RuntimeError("step size underflow in dopri5")
        nstep += 1

        k2 = np.asarray(fcn(x + c2 * h, y + h * a21 * k1), dtype=np.float64)
        k3 = np.asarray(
            fcn(x + c3 * h, y + h * (a31 * k1 + a32 * k2)), dtype=np.float64
        )
        k4 = np.asarray(
            fcn(x + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3)),
            dtype=np.float64,
        )
        k5 = np.asarray(
            fcn(
                x + c5 * h,
                y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4),
            ),
            dtype=np.float64,
        )
        k6 = np.asarray(
            fcn(
                xph,
                y
                + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5),
            ),
            dtype=np.float64,
        )
        y_new = y + h * (
            a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6
        )
        k7 = np.asarray(fcn(xph, y_new), dtype=np.float64)
        nfcn += 6

        err_vec = h * (
            e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7
        )
        sk = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err = float(math.sqrt(float(np.mean((err_vec / sk) ** 2))))
        if not math.isfinite(err):
            raise ValueError("non-finite local error in dopri5")
        fac11 = err**0.17 if err > 0.0 else 0.0
        fac = fac11 / (facold**0.04)
        fac_clamped = min(5.0, max(0.1, fac / 0.9))
        h_new = h / fac_clamped

        if err <= 1.0:
            facold = max(err, 1e-4)
            naccpt += 1
            ydiff = y_new - y
            bspl = h * k1 - ydiff
            rc = np.zeros((5, n), dtype=np.float64)
            rc[0] = y
            rc[1] = ydiff
            rc[2] = bspl
            rc[3] = -h * k7 + ydiff - bspl
            rc[4] = h * (
                d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6 + d7 * k7
            )
            history.append((x, h, rc))
            k1 = k7
            y = y_new
            x = xph
            if abs(h_new) > hmax_val:
                h_new = posneg * hmax_val
            if reject:
                h_new = posneg * min(abs(h_new), abs(h))
            reject = False
        else:
            h_new = h / min(5.0, max(0.1, fac11 / 0.9))
            reject = True
            if naccpt >= 1:
                nrejct += 1
        h = h_new

    return DopriResult(
        y=y,
        nfcn=nfcn,
        nstep=nstep,
        naccpt=naccpt,
        nrejct=nrejct,
        history=history,
    )


def dop853(
    x0: float,
    x_end: float,
    y0: npt.ArrayLike,
    fcn: Callable[[float, npt.NDArray[np.float64]], npt.ArrayLike],
    h0: float = 0.0,
    hmax: float = 0.0,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    max_steps: int = 100_000,
) -> DopriResult:
    """High-precision adaptive Dormand-Prince integrator (``dop853`` API).

    Args:
        x0: Initial independent variable value.
        x_end: Final independent variable value.
        y0: Initial state vector of shape ``(n,)``.
        fcn: Right-hand side callable ``fcn(x, y)`` returning shape ``(n,)``.
        h0: Initial trial step size.
        hmax: Maximum step size magnitude.
        rtol: Relative local error tolerance (``> 0``).
        atol: Absolute local error tolerance (``> 0``).
        max_steps: Maximum integration steps allowed.

    Returns:
        :class:`DopriResult` with final state and dense-output history.

    Raises:
        ValueError: If ``y0`` is empty, tolerances are non-positive, or local
            errors become non-finite.
        RuntimeError: If ``max_steps`` is exceeded or step size underflows.
    """
    return dopri5(
        x0,
        x_end,
        y0,
        fcn,
        h0=h0,
        hmax=hmax,
        rtol=max(1e-15, 0.25 * rtol),
        atol=max(1e-15, 0.25 * atol),
        max_steps=max_steps,
    )
