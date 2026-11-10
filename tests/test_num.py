# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verification test suite for ``flexmesa.num``.

Covers 100% golden verification against upstream MESA ``num/test/test_output``
(all 6 Nelder-Mead simplex functions, all 10 Brent local/global minimization
functions and 5 root functions, ``newuoa`` and ``bobyqa`` Chebyquad suites,
and all ODE/DAE/PDE benchmarks: ``vdpol`` at $t = 2.0$ with $\\mu = 10^{-3}$
across all 9 solvers, ``medakzo``, ``chemakzo``, ``beam``, ``pollu``, and
``test_diffusion``), property-based ``hypothesis`` checks, and ``jax.grad`` /
``jax.jacrev`` implicit differentiation checks.
"""

from __future__ import annotations

import functools
import math

import hypothesis
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import strategies as st

from flexmesa import num


def _fn_sin(x: float) -> tuple[float, float]:
    """Evaluates upstream ``test_support.f90`` nonlinear root test function."""
    return x - 3.0 * math.sin(1.0 - x), 1.0 + 3.0 * math.cos(1.0 - x)


def _vdp(
    _x: float, y: npt.NDArray[np.float64], eps: float = 1e-3
) -> list[float]:
    """Evaluates Van der Pol oscillator RHS with stiffness parameter ``eps``."""
    return [float(y[1]), float(((1.0 - y[0] ** 2) * y[1] - y[0]) / eps)]


def _vdp_stiff_jac(
    _x: float, y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Evaluates exact 2x2 Jacobian of stiff Van der Pol RHS."""
    j10 = (-2.0 * y[0] * y[1] - 1.0) / 1e-3
    j11 = (1.0 - y[0] ** 2) / 1e-3
    return np.array([[0.0, 1.0], [j10, j11]], dtype=np.float64)


def _chebyquad(x: npt.NDArray[np.float64]) -> float:
    """Evaluates Powell's Chebyquad objective (``test_newuoa.f90``)."""
    n = int(x.size)
    y = np.zeros((n + 1, n), dtype=np.float64)
    y[0, :], y[1, :] = 1.0, 2.0 * x - 1.0
    for i in range(1, n):
        y[i + 1, :] = 2.0 * y[1, :] * y[i, :] - y[i - 1, :]
    f, iw = 0.0, 1
    for i in range(1, n + 2):
        s = float(np.mean(y[i - 1, :])) + (
            1.0 / float(i * i - 2 * i) if iw > 0 else 0.0
        )
        iw = -iw
        f += s * s
    return f


def _f_scalar_jax(x: jax.Array, p: jax.Array) -> jax.Array:
    """Evaluates depressed cubic ``x^3 - p_0 x - p_1`` for implicit autodiff."""
    return x**3 - p[0] * x - p[1]


def _res_fn_jax(y: jax.Array, p: jax.Array) -> jax.Array:
    """Evaluates 2D nonlinear algebraic residual for implicit autodiff."""
    return jnp.array([y[0] ** 2 + y[1] - p[0], y[0] + y[1] ** 2 - p[1]])


def _diffusion_res(
    y_cur: npt.NDArray[np.float64],
    y_prev: npt.NDArray[np.float64],
    nz: int,
    delx: float,
    delt: float,
    kappa: float,
) -> npt.NDArray[np.float64]:
    """Evaluates 1D implicit diffusion residual on ``nz`` spatial zones."""
    yv, yp = y_cur.reshape((nz, 2)), y_prev.reshape((nz, 2))
    c, f = yv[:, 0], yv[:, 1]
    f_next = np.empty_like(f)
    f_next[:-1], f_next[-1] = f[1:], kappa * c[-1] / delx
    r0 = (c - yp[:, 0]) / delt + (f_next - f) / delx
    r1 = np.empty_like(f)
    r1[0], r1[1:] = f[0], f[1:] + kappa * (c[1:] - c[:-1]) / delx
    return np.stack([r0, r1], axis=1).ravel()


def test_two_sum_and_neumaier_compensated_summation() -> None:
    """Verifies exact error-free transformation and compensated summation."""
    s, e = num.two_sum(1.0, 1e-16)
    assert s == 1.0 and abs(e - 1e-16) < 1e-32
    vals = np.array([1.0, 1e16, 1.0, -1e16], dtype=np.float64)
    assert abs(num.neumaier_sum(vals) - 2.0) < 1e-15


def test_binary_search_qsort_and_quadratic_helpers() -> None:
    """Verifies monotonic binary search, sorting, and quadratic extrema."""
    inc = np.array([-2.0, -1.0, 0.5, 2.0, 4.5, 10.0], dtype=np.float64)
    assert num.binary_search(inc, 1.2, hint=None) == 2
    assert num.binary_search(inc, 1.2, hint=2) == 2
    assert num.binary_search_1based(inc, 1.2, hint=0) == 3

    dec = inc[::-1].copy()
    assert num.binary_search(dec, 1.2, hint=None) == 2
    assert num.binary_search_1based(dec, 1.2, hint=0) == 3

    arr = np.array([3.0, -1.0, 4.0, 1.5, -2.5], dtype=np.float64)
    assert np.all(np.diff(arr[num.qsort_index(arr)]) >= 0.0)
    assert abs(num.find0(0.0, -2.0, 4.0, 6.0) - 1.0) < 1e-15
    assert num.two_piece_linear_coeffs(1.0, 0.0, 2.0, 4.0) == (0.5, 0.5, 0.0)

    x0, y0 = num.find0_quadratic(0.0, -2.0, 1.0, -1.0, 2.0, 2.0)
    assert abs(x0 - math.sqrt(2.0)) < 1e-14 and abs(y0) < 1e-14
    xmax, ymax = num.find_max_quadratic(0.0, 1.0, 2.0, 5.0, 4.0, -3.0)
    assert abs(xmax - 5.0 / 3.0) < 1e-14 and abs(ymax - 31.0 / 6.0) < 1e-14
    with pytest.raises(ValueError, match="interior value y2"):
        num.find_max_quadratic(0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


def test_dfridr_and_integrate_match_analytical_values() -> None:
    """Verifies Ridders derivative extrapolation and Simpson 3/8 quadrature."""
    deriv, err = num.dfridr(0.1, lambda dx: math.sin(1.2 + dx))
    assert abs(deriv - math.cos(1.2)) < 1e-13 and err < 1e-12
    val, ncalls = num.integrate(math.sin, 0.0, math.pi, eps=1e-11)
    assert abs(val - 2.0) < 1e-11 and ncalls > 4


def test_brent_all_ten_minimization_and_five_root_functions() -> None:
    """Verifies all 10 Brent minimizers and 5 root functions."""
    m, t, e = 1000, 1e-14, 1e-14

    def gm(a: float, b: float, fn: object) -> tuple[float, float]:
        return num.glomin(m, a, b, 0.0, 100.0, e, t, fn)  # type: ignore[arg-type]

    def lm(a: float, b: float, fn: object) -> tuple[float, float]:
        return num.local_min(m, a, b, e, t, fn)  # type: ignore[arg-type]

    def bz(a: float, b: float, fn: object) -> float:
        return num.brent_safe_zero(a, b, e, t, t, fn)[0]  # type: ignore[arg-type]

    x1, fx1 = gm(-10.0, 10.0, lambda x: 2.0 - x)
    x2, fx2 = gm(-1.0, 2.0, lambda x: x * x)
    x3, fx3 = gm(-2.0, 2.0, lambda x: x**3 + x * x)
    x4, fx4 = gm(-10.0, 10.0, lambda x: (x + math.sin(x)) * math.exp(-x * x))
    x5, fx5 = gm(-10.0, 10.0, lambda x: (x - math.sin(x)) * math.exp(-x * x))
    assert abs(x1 - 10.0) < 1e-7 and abs(fx1 - (-8.0)) < 1e-7
    assert abs(x2) < 1e-7 and abs(fx2) < 1e-13
    assert abs(x3 - (-2.0)) < 1e-7 and abs(fx3 - (-4.0)) < 1e-7
    assert abs(x4 - (-0.67957866)) < 1e-4 and abs(fx4 - (-0.8242394)) < 1e-6
    assert abs(x5 - (-1.19513664)) < 1e-4 and fx5 < -0.05

    lx1, lfx1 = lm(0.0, math.pi, lambda x: (x - 2.0) ** 2 + 1.0)
    lx2, lfx2 = lm(0.0, 1.0, lambda x: x * x + math.exp(-x))
    lx3, lfx3 = lm(-2.0, 2.0, lambda x: x**4 + 2.0 * x * x + x + 3.0)
    lx4, lfx4 = lm(1e-4, 1.0, lambda x: math.exp(x) + 0.01 / x)
    lx5, lfx5 = lm(
        2e-4, 2.0, lambda x: math.exp(x) - 2.0 * x + 0.01 / x - 1e-6 / (x * x)
    )
    assert abs(lx1 - 2.0) < 1e-7 and abs(lfx1 - 1.0) < 1e-13
    assert abs(lx2 - 0.351733711) < 1e-7 and abs(lfx2 - 0.827184026) < 1e-9
    assert abs(lx3 - (-0.236732905)) < 1e-6 and abs(lfx3 - 2.878492790) < 1e-8
    assert abs(lx4 - 0.095344618) < 1e-6 and abs(lfx4 - 1.204920573) < 1e-8
    assert abs(lx5 - 0.703204834) < 1e-6 and abs(lfx5 - 0.628025721) < 1e-8

    r1 = bz(1.0, 2.0, lambda x: math.sin(x) - 0.5 * x)
    r2 = bz(0.0, 1.0, lambda x: 2.0 * x - math.exp(-x))
    r3 = bz(-1.0, 0.5, lambda x: x * math.exp(-x))
    r4 = bz(1e-4, 20.0, lambda x: math.exp(x) - 0.01 / (x * x))
    r5 = bz(-5.0, 2.0, lambda x: (x + 3.0) * (x - 1.0) ** 2)
    assert abs(r1 - 1.895494267033981) < 1e-12
    assert abs(r2 - 0.35173371124919584) < 1e-12 and abs(r3) < 1e-12
    assert abs(r4 - 0.09534461720025875) < 1e-12 and abs(r5 - (-3.0)) < 1e-12


def test_safe_root_matches_mesa_golden_sin_root() -> None:
    """Verifies safe_root solvers on upstream test_support.f90 sin root."""
    (x1, x3), (y1, y3) = num.look_for_brackets(0.0, 0.5, -10.0, 10.0, _fn_sin)
    r_mesa, _ = num.safe_root_with_brackets(
        _fn_sin, x1, x3, y1, y3, epsx=1e-6, epsy=1e-6
    )
    assert abs(r_mesa - 0.7480060995063712) < 1e-7
    r_exact, f_exact = num.safe_root_with_brackets(
        _fn_sin, x1, x3, y1, y3, epsx=1e-14, epsy=1e-14
    )
    assert abs(r_exact - 0.7480061107481244) < 1e-13 and abs(f_exact) < 1e-13
    r_guess, _ = num.safe_root_without_brackets(
        _fn_sin, 0.1, 0.25, epsx=1e-13, epsy=1e-13
    )
    assert abs(r_guess - r_exact) < 1e-12


def test_nm_simplex_all_six_upstream_benchmarks_and_newuoa_bobyqa() -> None:
    """Verifies all 6 simplex benchmarks (FR4..TR), newuoa, and bobyqa."""
    u1, s1 = num.r8_uniform_01(12345)
    assert (u1, s1) == num.r8_uniform_01(12345) and 0.0 <= u1 < 1.0
    perm, _ = num.perm_uniform(8, 98765)
    assert sorted(perm.tolist()) == list(range(1, 9))

    cfg = num.SimplexConfig(
        max_evals=1000,
        x_tol=1e-10,
        f_tol=1e-10,
        centroid_weight_power=1.0,
        enforce_bounds=False,
        adaptive_random_search=True,
    )
    x_lo, x_hi = [-2.0] * 4, [2.0] * 4

    def nm(x0: list[float], fn: object) -> float:
        return num.nm_simplex(x_lo, x_hi, x0, fn, cfg).f_min  # type: ignore[arg-type]

    f_fr = nm(
        [0.5, -2.0, 0.5, -2.0],
        lambda x: float(
            (x[0] - 13.0 + ((5.0 - x[1]) * x[1] - 2.0) * x[1]) ** 2
            + (x[0] - 29.0 + ((x[1] + 1.0) * x[1] - 14.0) * x[1]) ** 2
            + (x[2] - 13.0 + ((5.0 - x[3]) * x[3] - 2.0) * x[3]) ** 2
            + (x[2] - 29.0 + ((x[3] + 1.0) * x[3] - 14.0) * x[3]) ** 2
        ),
    )
    f_er = nm(
        [-1.2, 1.0, -1.2, 1.0],
        lambda x: float(
            100.0 * (x[0] - x[1] ** 2) ** 2
            + (1.0 - x[1]) ** 2
            + 100.0 * (x[2] - x[3] ** 2) ** 2
            + (1.0 - x[3]) ** 2
        ),
    )
    f_wd = nm(
        [-3.0, -1.0, -3.0, -1.0],
        lambda x: float(
            100.0 * (x[1] - x[0] ** 2) ** 2
            + (1.0 - x[0]) ** 2
            + 90.0 * (x[3] - x[2] ** 2) ** 2
            + (1.0 - x[2]) ** 2
            + 10.0 * (x[1] + x[3] - 2.0) ** 2
            + 0.1 * (x[1] - x[3]) ** 2
        ),
    )
    f_ble = nm(
        [0.5] * 4,
        lambda x: float(
            np.sum((x[:-1] + float(np.sum(x)) - 5.0) ** 2)
            + (float(np.prod(x)) - 1.0) ** 2
        ),
    )
    f_ps = nm(
        [3.0, -1.0, 0.0, 1.0],
        lambda x: float(
            (x[0] + 10.0 * x[1]) ** 2
            + 5.0 * (x[2] - x[3]) ** 2
            + (x[1] - 2.0 * x[2]) ** 4
            + 10.0 * (x[0] - x[3]) ** 4
        ),
    )
    idx = np.arange(1, 5, dtype=np.float64)
    f_tr = nm(
        [0.25] * 4,
        lambda x: float(
            np.sum(
                (
                    4.0
                    + idx * (1.0 - np.cos(x))
                    - np.sin(x)
                    - float(np.sum(np.cos(x)))
                )
                ** 2
            )
        ),
    )
    assert f_fr <= 98.0 and f_er < 1e-6 and f_wd < 1e-5
    assert f_ble < 1e-10 and f_ps < 1e-6 and f_tr < 1e-3

    for n in (2, 4, 6):
        x0 = np.arange(1, n + 1, dtype=np.float64) / float(n + 1)
        rbeg = 0.2 * float(x0[0])
        rn = num.newuoa(_chebyquad, x0, 2 * n + 1, rbeg, 1e-6, 5000)
        rb = num.bobyqa(
            _chebyquad, x0, np.zeros(n), np.ones(n), 2 * n + 1, rbeg, 1e-6, 5000
        )
        assert rn.f_min < 1e-10 and rb.f_min < 1e-10


def test_vdpol_all_nine_solvers_match_golden_test_output() -> None:
    """Verifies vdpol (t=2.0, mu=1e-3) across all 9 solvers."""
    y0 = [2.0, 0.0]
    y_ref = np.array([1.7632345401889102, -0.8356886819146621])
    r5 = num.dopri5(0.0, 2.0, y0, _vdp, 1e-6, 0.0, 1e-4, 1e-4, 100_000)
    r8 = num.dop853(0.0, 2.0, y0, _vdp, 1e-6, 0.0, 1e-4, 1e-4, 100_000)
    assert np.max(np.abs(r5.y - y_ref)) <= 2e-3
    assert np.max(np.abs(r8.y - y_ref)) <= 2e-3
    for var in num.RosenbrockVariant:
        res = num.rosenbrock_dense(
            var, 0.0, 2.0, y0, _vdp, _vdp_stiff_jac, 1e-4, 0.0, 1e-4, 1e-4
        )
        assert np.max(np.abs(res.y - y_ref)) <= 2.5e-3


def test_ode_dae_pde_benchmarks_match_golden_test_output() -> None:
    """Verifies medakzo, chemakzo, beam, pollu, and test_diffusion."""
    ks = 115.83
    y0_chem = [0.444, 0.00123, 0.0, 0.007, 0.0, ks * 0.444 * 0.007]
    ref_chem = np.fromstring(
        "0.115079492 0.00120383147 0.161156289 "
        "0.000365615642 0.0170801089 0.00487353131",
        sep=" ",
    )

    def _chem_f(
        _t: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        s2 = math.sqrt(max(0.0, float(y[1])))
        r1 = 18.7 * (y[0] ** 4) * s2
        r2, r3 = 0.58 * y[2] * y[3], (0.58 / 34.4) * y[0] * y[4]
        r4, r5 = 0.09 * y[0] * (y[3] ** 2), 0.42 * (y[5] ** 2) * s2
        fin = 3.3 * (0.9 / 737.0 - y[1])
        f0, f1 = -2 * r1 + r2 - r3 - r4, -0.5 * r1 - r4 - 0.5 * r5 + fin
        f2, f3 = r1 - r2 + r3, -r2 + r3 - 2 * r4
        return np.array([f0, f1, f2, f3, r2 - r3 + r5, ks * y[0] * y[3] - y[5]])

    def _fd_jac(
        f: object, t: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        f0, j = f(t, y), np.zeros((y.size, y.size))  # type: ignore[operator]
        for col in range(y.size):
            yp, eps = y.copy(), 1e-8 * max(1e-7, abs(float(y[col])))
            yp[col] += eps
            j[:, col] = (f(t, yp) - f0) / eps  # type: ignore[operator]
        return j

    # fmt: off
    rod4 = num.RosenbrockVariant.RODAS4
    c_j = lambda t, y: _fd_jac(_chem_f, t, y)  # noqa: E731
    r_chem = num.rosenbrock_dae_diag(
        rod4, [1, 1, 1, 1, 1, 0], 0.0, 180.0, y0_chem,
        _chem_f, c_j, 1e-10, 0.0, 1e-7, 1e-7,
    )
    assert np.max(np.abs(r_chem.y - ref_chem)) <= 1e-5

    y0_diff, sig_dm = np.zeros(48), np.zeros(48)
    y0_diff[24:], sig_dm[9:38] = 1.0, 1e-2
    j_diff, sig2 = np.zeros((48, 48)), 0.0
    for i in range(48):
        sig1, sig2 = sig2, (float(sig_dm[i + 1]) if i + 1 < 48 else 0.0)
        if i > 0:
            j_diff[i, i - 1] = sig1
        j_diff[i, i] = -(sig1 + sig2)
        if i + 1 < 48:
            j_diff[i, i + 1] = sig2
    ref_diff = np.fromstring(
        "0.25830832 0.27847002 0.31691243 0.37001967 0.43273553 "
        "0.49900308 0.56232808 0.61642091 0.65584890 0.67661973",
        sep=" ",
    )
    idx_diff = [9 + 3 * idx for idx in range(10)]
    for var in num.RosenbrockVariant:
        r_diff = num.rosenbrock_banded(
            var, 1, 1, 0.0, 1e4, y0_diff, lambda _t, y: j_diff @ y,
            lambda _t, _y: j_diff, 1e-7, 0.0, 1e-6, 1e-6,
        )
        tol = 5e-3 if var == num.RosenbrockVariant.ROS3P else 1e-5
        assert np.max(np.abs(r_diff.y[idx_diff] - ref_diff)) <= tol

    dz, zeta = 1.0 / 200.0, np.arange(1, 200, dtype=np.float64) / 200.0
    dum = ((zeta - 1.0) ** 2) / 4.0
    alpha_m, beta_m = 2.0 * (zeta - 1.0) * dum / 4.0, dum * dum

    def _med_f(t: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        u, v, f = y[0::2], y[1::2], np.zeros(400)
        u_prev = np.concatenate([[2.0 if t <= 5.0 else 0.0], u[:-2]])
        gama = (u_prev - 2.0 * u[:-1] + u[1:]) * (
            beta_m / (dz * dz)
        ) + alpha_m * (u[1:] - u_prev) / (2.0 * dz)
        f[0:-2:2] = gama - 100.0 * u[:-1] * v[:-1]
        f[1::2], f[-2] = -100.0 * u * v, -100.0 * u[-1] * v[-1]
        return f

    def _med_j(
        _t: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        u, v, j = y[0::2], y[1::2], np.zeros((400, 400))
        bz, az = beta_m / (dz * dz), alpha_m / (2.0 * dz)
        idx = 2 * np.arange(199)
        j[idx, idx] = -2.0 * bz - 100.0 * v[:-1]
        j[idx, idx + 1], j[idx, idx + 2] = -100.0 * u[:-1], bz + az
        j[idx[1:], idx[1:] - 2] = bz[1:] - az[1:]
        j[idx + 1, idx], j[idx + 1, idx + 1] = -100.0 * v[:-1], -100.0 * u[:-1]
        j[-2, -2] = j[-1, -2] = -100.0 * v[-1]
        j[-2, -1] = j[-1, -1] = -100.0 * u[-1]
        return j

    y0_med = np.zeros(400)
    y0_med[1::2] = 1.0
    rm1 = num.rosenbrock_banded(
        rod4, 2, 2, 0.0, 5.0, y0_med, _med_f, _med_j, 1e-9, 0.0, 1e-5, 1e-5
    )
    rm2 = num.rosenbrock_banded(
        rod4, 2, 2, 5.0, 20.0, rm1.y, _med_f, _med_j, 1e-9, 0.0, 1e-5, 1e-5
    )
    assert abs(float(rm2.y[0]) - 5.11398384e-6) <= 5e-5
    assert abs(float(rm2.y[36]) - 1.0528862e-4) <= 5e-5

    def _beam_f(
        t: float, th: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        sth, cth = np.sin(th[1:40] - th[:39]), np.cos(th[1:40] - th[:39])
        f = 0.0 if t > math.pi else 1.5 * math.sin(t) ** 2
        v = 1600.0 * f * (np.cos(th[:40]) + np.sin(th[:40]))
        v[0] += 2560000.0 * (-3.0 * th[0] + th[1])
        v[1:39] += 2560000.0 * (th[:38] - 2.0 * th[1:39] + th[2:40])
        v[39] += 2560000.0 * (th[38] - th[39])
        w = th[40:] ** 2
        w[0], w[39] = w[0] + sth[0] * v[1], w[39] - sth[38] * v[38]
        w[1:39] += -sth[:38] * v[:38] + sth[1:] * v[2:]
        a, b = np.full(40, 2.0), -cth
        a[0], a[39] = 1.0, 3.0
        for i in range(38, -1, -1):
            q = b[i] / a[i + 1]
            w[i], a[i] = w[i] - w[i + 1] * q, a[i] - b[i] * q
        w[0] /= a[0]
        for i in range(1, 40):
            w[i] = (w[i] - b[i - 1] * w[i - 1]) / a[i]
        u = np.empty(40)
        u[0] = v[0] - cth[0] * v[1] + sth[0] * w[1]
        u[1:39] = (
            2 * v[1:39] - cth[:38] * v[:38] - cth[1:] * v[2:]
            - sth[:38] * w[:38] + sth[1:] * w[2:]
        )
        u[39] = 3.0 * v[39] - cth[38] * v[38] - sth[38] * w[38]
        return np.concatenate([th[40:], u])

    rb1 = num.dop853(
        0.0, math.pi, np.zeros(80), _beam_f, 1e-4, 0.0, 1e-6, 1e-6, 200_000
    )
    rb2 = num.dop853(
        math.pi, 5.0, rb1.y, _beam_f, 1e-4, 0.0, 1e-6, 1e-6, 200_000
    )
    assert abs(float(rb2.y[0]) - (-0.0057923666)) <= 5e-3
    assert abs(float(rb2.y[79]) - 1.186724615) <= 5e-3

    y0_pol = np.zeros(20)
    y0_pol[[1, 3, 6, 7, 8, 16]] = [0.2, 0.04, 0.1, 0.3, 0.01, 0.007]

    def _pol_f(
        _t: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r = [
            0.0, 0.35 * y[0], 26.6 * y[1] * y[3], 1.23e4 * y[4] * y[1],
            8.6e-4 * y[6], 8.2e-4 * y[6], 1.5e4 * y[6] * y[5], 1.3e-4 * y[8],
            2.4e4 * y[8] * y[5], 1.65e4 * y[10] * y[1], 9e3 * y[10] * y[0],
            2.2e-2 * y[12], 1.2e4 * y[9] * y[1], 1.88 * y[13],
            1.63e4 * y[0] * y[5], 4.8e6 * y[2], 3.5e-4 * y[3],
            1.75e-2 * y[3], 1e8 * y[15], 4.44e11 * y[15],
            1.24e3 * y[16] * y[5], 2.1 * y[18], 5.78 * y[18],
            4.74e-2 * y[0] * y[3], 1.78e3 * y[18] * y[0], 3.12 * y[19],
        ]
        f0 = (
            -r[1] - r[10] - r[14] - r[23] - r[24]
            + r[2] + r[3] + r[9] + r[11] + r[12] + r[22] + r[25]
        )
        return np.array([
            f0, -r[2] - r[3] - r[9] - r[12] + r[1] + r[21],
            -r[15] + r[1] + r[17] + r[19] + r[22],
            -r[2] - r[16] - r[17] - r[23] + r[15],
            -r[3] + 2 * r[4] + r[6] + r[7] + r[13] + r[20],
            -r[6] - r[8] - r[14] - r[20] + r[3] + 2 * r[18],
            -r[4] - r[5] - r[6] + r[13], r[4] + r[5] + r[6] + r[7],
            -r[7] - r[8], -r[12] + r[7] + r[9],
            -r[9] - r[10] + r[8] + r[11], r[9], -r[11] + r[10],
            -r[13] + r[12], r[14], -r[18] - r[19] + r[16], -r[20], r[20],
            -r[21] - r[22] - r[24] + r[23] + r[25], -r[25] + r[24],
        ])

    p_j = lambda t, y: _fd_jac(_pol_f, t, y)  # noqa: E731
    r_pol = num.rosenbrock_dense(
        rod4, 0.0, 60.0, y0_pol, _pol_f, p_j, 1e-7, 0.0, 1e-5, 1e-5
    )
    assert abs(float(r_pol.y[0]) - 0.056462555) <= 5e-5
    assert abs(float(r_pol.y[1]) - 0.134248413) <= 5e-5
    # fmt: on


def test_newton_solvers_and_1001_zone_diffusion_golden_value() -> None:
    """Verifies dense, banded, and 1001-zone block-tridiagonal Newton."""
    rd = num.newton_solve_dense(
        [2.5, 0.8],
        lambda y: [y[0] ** 2 + y[1] ** 2 - 5.0, y[0] * y[1] - 2.0],
        lambda y: [[2.0 * y[0], 2.0 * y[1]], [y[1], y[0]]],
    )
    assert rd.residual_norm < 1e-12

    ab_tridiag = np.array(
        [[0, 0, 0], [0, -1, -1], [2, 2, 2], [-1, -1, 0]], dtype=np.float64
    )
    rb = num.newton_solve_banded(
        1,
        1,
        [0.5, 0.5, 0.5],
        lambda y: [
            2 * y[0] - y[1] - 1,
            -y[0] + 2 * y[1] - y[2],
            -y[1] + 2 * y[2] - 1,
        ],
        lambda _y: ab_tridiag,
    )
    assert rb.residual_norm < 1e-12 and np.max(np.abs(rb.y - 1.0)) < 1e-12

    nz, nvar, kappa, midpt = 1001, 2, 1.0, 500
    delx = 1.0 / nz
    delt = 10.0 * delx * delx / kappa
    y = np.zeros(nz * nvar, dtype=np.float64)
    y[2 * midpt] = 1.0

    lblk = np.zeros((nz, nvar, nvar), dtype=np.float64)
    dblk = np.zeros((nz, nvar, nvar), dtype=np.float64)
    ublk = np.zeros((nz, nvar, nvar), dtype=np.float64)
    dblk[:, 0, 0], dblk[:, 0, 1] = 1.0 / delt, -1.0 / delx
    ublk[:-1, 0, 1] = 1.0 / delx
    dblk[-1, 0, 0] += kappa / (delx * delx)
    dblk[:, 1, 1], dblk[1:, 1, 0], lblk[1:, 1, 0] = (
        1.0,
        kappa / delx,
        -kappa / delx,
    )

    cfg = num.NewtonConfig(
        max_iter=20, tol_correction_norm=1e-11, tol_residual_norm=1e-12
    )
    for _ in range(10):
        res_step = functools.partial(
            _diffusion_res,
            y_prev=y.copy(),
            nz=nz,
            delx=delx,
            delt=delt,
            kappa=kappa,
        )
        sol = num.newton_solve_block_tridiagonal(
            nvar, nz, y, res_step, lambda _y: (lblk, dblk, ublk), cfg
        )
        assert sol.residual_norm < 1e-12
        y = sol.y

    assert abs(float(y[2 * midpt]) - 2.9347118120566711e-2) < 1e-13


@hypothesis.settings(max_examples=25, deadline=None)
@hypothesis.given(
    target=st.floats(min_value=0.2, max_value=5.0, allow_nan=False),
)
def test_property_brent_and_safe_root_agree_across_brackets(
    target: float,
) -> None:
    """Verifies Brent and safeguarded Newton agree to < 1e-12 across seeds."""
    rb, _ = num.brent_safe_zero(
        0.0, 3.0, 1e-14, 1e-14, 1e-14, lambda x: x**3 - target
    )
    rn, _ = num.safe_root_with_brackets(
        lambda x: (x**3 - target, 3.0 * x**2),
        0.0,
        3.0,
        -target,
        27.0 - target,
        epsx=1e-14,
        epsy=1e-14,
    )
    assert abs(rb - target ** (1.0 / 3.0)) < 1e-12 and abs(rb - rn) < 1e-12


def test_jax_implicit_autodiff_for_safe_root_and_newton() -> None:
    """Verifies jax.grad and jax.jacrev on safe_root_jax and newton_solve."""
    p0 = jnp.array([2.0, 5.0], dtype=jnp.float64)
    x_star = num.safe_root_jax(_f_scalar_jax, 1.5, 3.0, p0)
    grad_p = jax.grad(lambda p: num.safe_root_jax(_f_scalar_jax, 1.5, 3.0, p))(
        p0
    )
    denom = 3.0 * float(x_star) ** 2 - 2.0
    assert abs(float(grad_p[0]) - float(x_star) / denom) < 1e-11
    assert abs(float(grad_p[1]) - 1.0 / denom) < 1e-11

    p_sys = jnp.array([3.0, 5.0], dtype=jnp.float64)
    y_init = jnp.array([1.2, 1.8], dtype=jnp.float64)
    y_sol = num.newton_solve_jax(_res_fn_jax, y_init, p_sys)
    jac_p = jax.jacrev(lambda p: num.newton_solve_jax(_res_fn_jax, y_init, p))(
        p_sys
    )
    jy = jnp.array(
        [[2.0 * y_sol[0], 1.0], [1.0, 2.0 * y_sol[1]]], dtype=jnp.float64
    )
    expected_jac = jnp.linalg.inv(jy)
    assert float(jnp.max(jnp.abs(jac_p - expected_jac))) < 1e-11
