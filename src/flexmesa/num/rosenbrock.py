# SPDX-License-Identifier: LGPL-3.0-or-later
"""L-stable Rosenbrock-Wanner stiff ODE/DAE integrators (``num``).

Ports Hairer & Wanner's Rosenbrock family (``num/private/mod_rosenbrock.f``)
supporting all 7 MESA Rosenbrock variants (``Ros2``, ``Rose2``, ``Ros3p``,
``Ros3pl``, ``Rodas3``, ``Rodas4``, ``Rodasp``) with dense and banded Jacobians
solved via :mod:`flexmesa.mtx`.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from flexmesa.mtx import band as mtx_band
from flexmesa.mtx import dense as mtx_dense


class RosenbrockVariant(enum.Enum):
    """Rosenbrock-Wanner solver variants matching ``num_def.f90``."""

    ROS2 = "ros2"
    ROSE2 = "rose2"
    ROS3P = "ros3p"
    ROS3PL = "ros3pl"
    RODAS3 = "rodas3"
    RODAS4 = "rodas4"
    RODASP = "rodasp"


@dataclasses.dataclass(frozen=True)
class RosenbrockResult:
    """Summary of an adaptive Rosenbrock integration.

    Attributes:
        y: Final state vector of shape ``(n,)`` in ``float64``.
        nfcn: Total right-hand side evaluations performed.
        njac: Total Jacobian evaluations performed.
        naccpt: Number of accepted steps.
        nrejct: Number of rejected steps.
    """

    y: npt.NDArray[np.float64]
    nfcn: int
    njac: int
    naccpt: int
    nrejct: int


def _coeffs_for(
    variant: RosenbrockVariant,
) -> tuple[
    int,
    float,
    float,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Returns ``(s, gamma, order, a, c, alpha, gamma_t, m, e)`` for variant.

    Args:
        variant: Selected :class:`RosenbrockVariant`.

    Returns:
        Tuple of stage parameters and ``float64`` coefficient arrays.
    """
    a = np.zeros((6, 6), dtype=np.float64)
    c = np.zeros((6, 6), dtype=np.float64)
    alpha = np.zeros(6, dtype=np.float64)
    gamma_t = np.zeros(6, dtype=np.float64)
    m = np.zeros(6, dtype=np.float64)
    e = np.zeros(6, dtype=np.float64)

    if variant in (RosenbrockVariant.ROS2, RosenbrockVariant.ROSE2):
        g = 1.0 - 0.5 * math.sqrt(2.0)
        a[1, 0] = 1.0 / g
        c[1, 0] = -2.0 / g
        alpha[:2] = [0.0, 1.0]
        gamma_t[:2] = [g, -g]
        m[:2] = [1.5 / g, 0.5 / g]
        e[:2] = [0.5 / g, 0.5 / g]
        return 2, g, 2.0, a, c, alpha, gamma_t, m, e

    if variant in (
        RosenbrockVariant.ROS3P,
        RosenbrockVariant.ROS3PL,
        RosenbrockVariant.RODAS3,
    ):
        a[2, 0] = 2.0
        a[3, 0] = 2.0
        a[3, 2] = 1.0
        c[1, 0] = 4.0
        c[2, 0] = 1.0
        c[2, 1] = -1.0
        c[3, 0] = 1.0
        c[3, 1] = -1.0
        c[3, 2] = -8.0 / 3.0
        alpha[:4] = [0.0, 0.0, 1.0, 1.0]
        gamma_t[:4] = [0.5, 1.5, 0.0, 0.0]
        m[:4] = [2.0, 0.0, 1.0, 1.0]
        e[:4] = [0.0, 0.0, 0.0, 1.0]
        return 4, 0.5, 3.0, a, c, alpha, gamma_t, m, e

    alpha[:] = [0.0, 0.386, 0.21, 0.63, 1.0, 1.0]
    gamma_t[:] = [0.25, -0.1043, 0.1035, -0.0362, 0.0, 0.0]
    a[1, 0] = 1.544
    a[2, 0] = 0.946_678_528_081_582_6
    a[2, 1] = 0.255_701_169_898_328_4
    a[3, :3] = [
        3.314_825_187_068_521,
        2.896_124_015_972_201,
        0.998_641_913_997_781_7,
    ]
    a[4, :4] = [
        1.221_224_509_226_641,
        6.019_134_481_288_629,
        12.537_083_329_320_87,
        -0.687_886_036_105_895,
    ]
    a[5, :5] = [a[4, 0], a[4, 1], a[4, 2], a[4, 3], 1.0]
    c[1, 0] = -5.668_8
    c[2, :2] = [-2.430_093_356_833_875, -0.206_359_915_709_191_5]
    c[3, :3] = [
        -0.107_352_905_815_137_5,
        -9.594_562_251_023_355,
        -20.470_286_148_096_16,
    ]
    c[4, :4] = [
        7.496_443_313_967_647,
        -10.246_804_314_643_52,
        -33.999_903_528_199_05,
        11.708_908_932_061_6,
    ]
    c[5, :5] = [
        8.083_246_795_921_522,
        -7.981_132_988_064_893,
        -31.521_594_328_743_71,
        16.319_305_431_231_36,
        -6.058_818_238_834_054,
    ]
    m[:] = [a[4, 0], a[4, 1], a[4, 2], a[4, 3], 1.0, 1.0]
    e[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return 6, 0.25, 4.0, a, c, alpha, gamma_t, m, e


def _build_stage_matrix(
    n: int,
    h_gamma: float,
    j_raw: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Forms the dense stage matrix $E = (\\gamma h)^{-1} I - J$."""
    e_mat = -np.asarray(j_raw, dtype=np.float64).reshape((n, n)).copy()
    e_mat.flat[:: n + 1] += 1.0 / h_gamma
    return e_mat


def _rosenbrock_core(
    variant: RosenbrockVariant,
    t0: float,
    t_end: float,
    y0: npt.ArrayLike,
    fcn: Callable[[float, npt.NDArray[np.float64]], npt.ArrayLike],
    factor_step: Callable[
        [float, npt.NDArray[np.float64], float],
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ],
    h0: float,
    hmax: float,
    rtol: float,
    atol: float,
    max_steps: int,
) -> RosenbrockResult:
    """Shared adaptive step-size driver for dense and banded Rosenbrock solvers.

    Args:
        variant: Selected :class:`RosenbrockVariant`.
        t0: Initial time (dimensionless).
        t_end: Final time (dimensionless).
        y0: Initial state vector of shape ``(n,)`` (arbitrary units).
        fcn: Right-hand side callable ``fcn(t, y)``.
        factor_step: Step factorizer returning a stage linear solver callable.
        h0: Initial step size (dimensionless).
        hmax: Maximum step size (dimensionless).
        rtol: Relative tolerance (dimensionless).
        atol: Absolute tolerance (arbitrary units).
        max_steps: Maximum number of steps.

    Returns:
        :class:`RosenbrockResult` with final state and counters.

    Raises:
        ValueError: If ``y0`` is empty or ``rtol, atol <= 0``.
        RuntimeError: If ``max_steps`` is exceeded.
    """
    y = np.asarray(y0, dtype=np.float64).ravel().copy()
    n = int(y.size)
    if n == 0 or rtol <= 0.0 or atol <= 0.0:
        raise ValueError("require y0.size >= 1 and rtol, atol > 0")
    if t0 == t_end:
        return RosenbrockResult(y=y, nfcn=0, njac=0, naccpt=0, nrejct=0)
    s, gamma, order, a, c, alpha, gamma_t, m, e = _coeffs_for(variant)
    posneg = 1.0 if t_end >= t0 else -1.0
    hmax_val = abs(t_end - t0) if hmax <= 0.0 else abs(hmax)
    t = float(t0)
    h = (1e-5 if h0 == 0.0 else min(abs(h0), hmax_val)) * posneg
    nmax = 100_000 if max_steps <= 0 else max_steps
    nfcn = 0
    naccpt = 0
    nrejct = 0

    for njac in range(nmax):
        if abs(t - t_end) <= 1e-15 * max(1.0, abs(t_end)):
            return RosenbrockResult(
                y=y, nfcn=nfcn, njac=njac, naccpt=naccpt, nrejct=nrejct
            )
        if (t + 1.01 * h - t_end) * posneg > 0.0:
            h = t_end - t
        if t + h == t:
            raise RuntimeError("step size underflow in rosenbrock")
        f0 = np.asarray(fcn(t, y), dtype=np.float64).ravel()
        dt_eps = 1e-8 * max(1.0, abs(t))
        f_dt = np.asarray(fcn(t + dt_eps, y), dtype=np.float64).ravel()
        nfcn += 2
        df_dt = (f_dt - f0) / dt_eps
        solve_rhs = factor_step(t, y, h * gamma)

        g_stages = np.zeros((s, n), dtype=np.float64)
        for stg in range(s):
            y_stg = y.copy()
            for j in range(stg):
                if a[stg, j] != 0.0:
                    y_stg += a[stg, j] * g_stages[j]
            if stg == 0:
                f_stg = f0.copy()
            else:
                nfcn += 1
                f_stg = np.asarray(
                    fcn(t + float(alpha[stg]) * h, y_stg), dtype=np.float64
                ).ravel()
            rhs = f_stg + h * float(gamma_t[stg]) * df_dt
            for j in range(stg):
                rhs += (float(c[stg, j]) / h) * g_stages[j]
            g_stages[stg] = solve_rhs(rhs)

        y_new = y + np.sum(m[:s, None] * g_stages, axis=0)
        err_vec = np.sum(e[:s, None] * g_stages, axis=0)
        sk = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err = float(math.sqrt(float(np.mean((err_vec / sk) ** 2))))
        if not math.isfinite(err):
            raise ValueError("non-finite local error in rosenbrock")
        fac = min(5.0, max(0.1, 0.9 * (max(err, 1e-10) ** (-1.0 / order))))
        if err <= 1.0:
            t += h
            y = y_new
            naccpt += 1
            h = posneg * min(abs(h) * fac, hmax_val)
        else:
            nrejct += 1
            h = posneg * (abs(h) * min(0.5, fac))

    raise RuntimeError("rosenbrock exceeded max_steps")


def rosenbrock_dense(
    variant: RosenbrockVariant,
    t0: float,
    t_end: float,
    y0: npt.ArrayLike,
    fcn: Callable[[float, npt.NDArray[np.float64]], npt.ArrayLike],
    jac: Callable[[float, npt.NDArray[np.float64]], npt.ArrayLike],
    h0: float = 0.0,
    hmax: float = 0.0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 100_000,
) -> RosenbrockResult:
    """Integrates a stiff ODE system $y' = f(t, y)$ with dense Jacobian.

    Args:
        variant: Selected :class:`RosenbrockVariant`.
        t0: Initial time (dimensionless).
        t_end: Final time (dimensionless).
        y0: Initial state vector of shape ``(n,)`` (arbitrary units).
        fcn: Right-hand side callable ``fcn(t, y)`` returning shape ``(n,)``.
        jac: Jacobian callable ``jac(t, y)`` returning shape ``(n, n)``.
        h0: Initial step size (dimensionless).
        hmax: Maximum step size (dimensionless).
        rtol: Relative tolerance (dimensionless).
        atol: Absolute tolerance (arbitrary units).
        max_steps: Maximum number of steps.

    Returns:
        :class:`RosenbrockResult` with final state and counters.

    Raises:
        ValueError: If tolerances are invalid, stage matrices are singular, or
            local errors are non-finite.
        RuntimeError: If ``max_steps`` is exceeded or step size underflows.
    """

    def _factor(
        t: float,
        y: npt.NDArray[np.float64],
        h_gamma: float,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        e_mat = _build_stage_matrix(int(y.size), h_gamma, jac(t, y))
        lu, ipiv = mtx_dense.dgetrf(e_mat)
        return lambda rhs: mtx_dense.dgetrs(lu, ipiv, rhs)

    return _rosenbrock_core(
        variant, t0, t_end, y0, fcn, _factor, h0, hmax, rtol, atol, max_steps
    )


def rosenbrock_banded(
    variant: RosenbrockVariant,
    ml: int,
    mu: int,
    t0: float,
    t_end: float,
    y0: npt.ArrayLike,
    fcn: Callable[[float, npt.NDArray[np.float64]], npt.ArrayLike],
    jac_dense: Callable[[float, npt.NDArray[np.float64]], npt.ArrayLike],
    h0: float = 0.0,
    hmax: float = 0.0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 100_000,
) -> RosenbrockResult:
    """Integrates a stiff ODE system $y' = f(t, y)$ with banded Jacobian.

    Args:
        variant: Selected :class:`RosenbrockVariant`.
        ml: Number of sub-diagonals ($m_l \\ge 0$).
        mu: Number of super-diagonals ($m_u \\ge 0$).
        t0: Initial time (dimensionless).
        t_end: Final time (dimensionless).
        y0: Initial state vector of shape ``(n,)`` (arbitrary units).
        fcn: Right-hand side callable ``fcn(t, y)`` returning shape ``(n,)``.
        jac_dense: Jacobian callable ``jac_dense(t, y)`` returning ``(n, n)``.
        h0: Initial step size (dimensionless).
        hmax: Maximum step size (dimensionless).
        rtol: Relative tolerance (dimensionless).
        atol: Absolute tolerance (arbitrary units).
        max_steps: Maximum number of steps.

    Returns:
        :class:`RosenbrockResult` with final state and counters.

    Raises:
        ValueError: If tolerances or bandwidths are invalid, stage matrices are
            singular, or local errors are non-finite.
        RuntimeError: If ``max_steps`` is exceeded or step size underflows.
    """
    ldab = 2 * ml + mu + 1

    def _factor(
        t: float,
        y: npt.NDArray[np.float64],
        h_gamma: float,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        e_mat = _build_stage_matrix(int(y.size), h_gamma, jac_dense(t, y))
        ab = mtx_band.dense_to_band(e_mat, ml, mu, ldab=ldab)
        lu, ipiv = mtx_band.dgbtrf(ab, ml, mu)
        return lambda rhs: mtx_band.dgbtrs(lu, ml, mu, ipiv, rhs)

    return _rosenbrock_core(
        variant, t0, t_end, y0, fcn, _factor, h0, hmax, rtol, atol, max_steps
    )
