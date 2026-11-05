# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verification test suite for ``flexmesa.num``.

Covers golden verification against upstream MESA ``num/test/src`` cases,
property-based ``hypothesis`` checks across random brackets and monotonic
grids, and ``jax.grad`` / ``jax.jacrev`` implicit differentiation checks on
:func:`flexmesa.num.safe_root_jax` and :func:`flexmesa.num.newton_solve_jax`.
"""

from __future__ import annotations

import math

import hypothesis
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import strategies as st

from flexmesa import num


def test_two_sum_and_neumaier_compensated_summation() -> None:
    """Verifies exact error-free transformation and compensated summation."""
    s, e = num.two_sum(1.0, 1e-16)
    assert s == 1.0
    assert abs(e - 1e-16) < 1e-32

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
    idx = num.qsort_index(arr)
    assert np.all(np.diff(arr[idx]) >= 0.0)

    assert abs(num.find0(0.0, -2.0, 4.0, 6.0) - 1.0) < 1e-15
    x0, y0 = num.find0_quadratic(0.0, -2.0, 1.0, -1.0, 2.0, 2.0)
    assert abs(x0 - math.sqrt(2.0)) < 1e-14
    assert abs(y0) < 1e-14

    xmax, ymax = num.find_max_quadratic(0.0, 1.0, 2.0, 5.0, 4.0, -3.0)
    assert abs(xmax - 5.0 / 3.0) < 1e-14
    assert abs(ymax - 31.0 / 6.0) < 1e-14

    with pytest.raises(ValueError, match="interior value y2"):
        num.find_max_quadratic(0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


def test_dfridr_and_integrate_match_analytical_values() -> None:
    """Verifies Ridders derivative extrapolation and Simpson 3/8 quadrature."""
    deriv, err = num.dfridr(0.1, lambda dx: math.sin(1.2 + dx))
    assert abs(deriv - math.cos(1.2)) < 1e-13
    assert err < 1e-12

    val, ncalls = num.integrate(math.sin, 0.0, math.pi, eps=1e-11)
    assert abs(val - 2.0) < 1e-11
    assert ncalls > 4


def test_brent_root_and_minima_match_mesa_test_brent() -> None:
    """Verifies Brent zero-finder and local/global minimizers."""
    root, f_root = num.brent_safe_zero(
        2.0,
        3.0,
        1e-14,
        1e-14,
        1e-14,
        lambda x: (x + 3.0) * (x - 1.0) ** 2 - 10.0,
    )
    assert abs((root + 3.0) * (root - 1.0) ** 2 - 10.0) < 1e-12
    assert abs(f_root) < 1e-12

    xmin, fmin = num.brent_local_min(
        200, 0.0, 3.0, 1e-12, 1e-12, lambda x: (x - 2.0) ** 2 + 1.0
    )
    assert abs(xmin - 2.0) < 1e-7
    assert abs(fmin - 1.0) < 1e-13

    xg, fg = num.brent_global_min(
        500,
        -2.0,
        2.0,
        0.0,
        10.0,
        1e-12,
        1e-12,
        lambda x: (x - 0.5) ** 4 - 0.25 * (x - 0.5) ** 2 + 2.0,
    )
    assert abs(abs(xg - 0.5) - math.sqrt(0.125)) < 1e-5
    assert abs(fg - (2.0 - 0.015625)) < 1e-9


def test_safe_root_matches_mesa_golden_sin_root() -> None:
    """Verifies safe_root solvers on upstream test_support.f90 sin root."""

    def fn_sin(x: float) -> tuple[float, float]:
        return x - 3.0 * math.sin(1.0 - x), 1.0 + 3.0 * math.cos(1.0 - x)

    (x1, x3), (y1, y3) = num.look_for_brackets(0.0, 0.5, -10.0, 10.0, fn_sin)
    r_mesa, _ = num.safe_root_with_brackets(
        fn_sin, x1, x3, y1, y3, epsx=1e-6, epsy=1e-6
    )
    assert abs(r_mesa - 0.7480060995063712) < 1e-7

    r_exact, f_exact = num.safe_root_with_brackets(
        fn_sin, x1, x3, y1, y3, epsx=1e-14, epsy=1e-14
    )
    assert abs(r_exact - 0.7480061107481244) < 1e-13
    assert abs(f_exact) < 1e-13

    r_guess, _ = num.safe_root_without_brackets(
        fn_sin, 0.1, 0.25, epsx=1e-13, epsy=1e-13
    )
    assert abs(r_guess - r_exact) < 1e-12


def test_nelder_mead_simplex_and_ran1_zah_prng() -> None:
    """Verifies Nelder-Mead simplex on 2D Rosenbrock and ran1_zah PRNG."""
    u1, s1 = num.r8_uniform_01(12345)
    u2, s2 = num.r8_uniform_01(12345)
    assert u1 == u2 and s1 == s2
    assert 0.0 <= u1 < 1.0

    perm, _ = num.perm_uniform(8, 98765)
    assert sorted(perm.tolist()) == list(range(1, 9))

    res = num.nm_simplex(
        [-5.0, -5.0],
        [5.0, 5.0],
        [-1.2, 1.0],
        lambda x: float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2),
        num.SimplexConfig(max_evals=2000, x_tol=1e-11, f_tol=1e-14),
    )
    assert abs(float(res.x_min[0]) - 1.0) < 1e-4
    assert abs(float(res.x_min[1]) - 1.0) < 1e-4
    assert res.f_min < 1e-8


def test_explicit_rk_and_dopri_solve_van_der_pol() -> None:
    """Verifies rk2, rk4, dopri5, and dop853 on linear and Van der Pol ODEs."""
    y_rk4 = num.rk4(0.0, [1.0, 0.0], 0.05, lambda _t, y: [y[1], -y[0]])
    assert abs(float(y_rk4[0]) - math.cos(0.05)) < 1e-8

    y_rk2 = num.rk2(0.0, [1.0, 0.0], 0.01, lambda _t, y: [y[1], -y[0]])
    assert abs(float(y_rk2[0]) - math.cos(0.01)) < 1e-5

    def vdp(_x: float, y: npt.NDArray[np.float64]) -> list[float]:
        return [float(y[1]), float(((1.0 - y[0] ** 2) * y[1] - y[0]) / 1e-1)]

    res5 = num.dopri5(0.0, 1.5, [2.0, 0.0], vdp, rtol=1e-9, atol=1e-9)
    res8 = num.dop853(0.0, 1.5, [2.0, 0.0], vdp, rtol=1e-10, atol=1e-10)
    assert np.max(np.abs(res5.y - res8.y)) < 1e-6

    mid = res5.dense_eval(0.75)
    res_mid = num.dopri5(0.0, 0.75, [2.0, 0.0], vdp, rtol=1e-9, atol=1e-9)
    assert np.max(np.abs(mid - res_mid.y)) < 1e-5


def test_rosenbrock_all_seven_variants_and_banded_vdp() -> None:
    """Verifies all 7 Rosenbrock variants on stiff Van der Pol ODE."""
    eps = 1e-3

    def fcn(_x: float, y: npt.NDArray[np.float64]) -> list[float]:
        return [float(y[1]), float(((1.0 - y[0] ** 2) * y[1] - y[0]) / eps)]

    def jac_dense(
        _x: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return np.array(
            [
                [0.0, 1.0],
                [(-2.0 * y[0] * y[1] - 1.0) / eps, (1.0 - y[0] ** 2) / eps],
            ],
            dtype=np.float64,
        )

    ref = num.rosenbrock_dense(
        num.RosenbrockVariant.RODAS4,
        0.0,
        0.5,
        [2.0, 0.0],
        fcn,
        jac_dense,
        rtol=1e-9,
        atol=1e-9,
    )
    for var in num.RosenbrockVariant:
        out = num.rosenbrock_dense(
            var, 0.0, 0.5, [2.0, 0.0], fcn, jac_dense, rtol=1e-6, atol=1e-6
        )
        assert np.max(np.abs(out.y - ref.y)) < 2e-3

    def fcn3(_x: float, y: npt.NDArray[np.float64]) -> list[float]:
        return [
            float(y[1]),
            float(((1.0 - y[0] ** 2) * y[1] - y[0]) / eps),
            float(-y[2]),
        ]

    def jac3(
        _x: float, y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        j10 = (-2.0 * y[0] * y[1] - 1.0) / eps
        j11 = (1.0 - y[0] ** 2) / eps
        return np.array(
            [[0.0, 1.0, 0.0], [j10, j11, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )

    dense3 = num.rosenbrock_dense(
        num.RosenbrockVariant.RODAS4,
        0.0,
        0.5,
        [2.0, 0.0, 1.0],
        fcn3,
        jac3,
        rtol=1e-9,
        atol=1e-9,
    )
    banded = num.rosenbrock_banded(
        num.RosenbrockVariant.RODAS4,
        1,
        1,
        0.0,
        0.5,
        [2.0, 0.0, 1.0],
        fcn3,
        jac3,
        rtol=1e-9,
        atol=1e-9,
    )
    assert np.max(np.abs(banded.y - dense3.y)) < 1e-14
    assert np.max(np.abs(banded.y[:2] - ref.y)) < 1e-9


def test_newton_solvers_and_1001_zone_diffusion_golden_value() -> None:
    """Verifies dense, banded, and 1001-zone block-tridiagonal Newton."""
    rd = num.newton_solve_dense(
        [2.5, 0.8],
        lambda y: [y[0] ** 2 + y[1] ** 2 - 5.0, y[0] * y[1] - 2.0],
        lambda y: [[2.0 * y[0], 2.0 * y[1]], [y[1], y[0]]],
    )
    assert rd.residual_norm < 1e-12

    rb = num.newton_solve_banded(
        1,
        1,
        [0.5, 0.5, 0.5],
        lambda y: [
            2.0 * y[0] - y[1] - 1.0,
            -y[0] + 2.0 * y[1] - y[2],
            -y[1] + 2.0 * y[2] - 1.0,
        ],
        lambda _y: np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, -1.0, -1.0],
                [2.0, 2.0, 2.0],
                [-1.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    assert rb.residual_norm < 1e-12
    assert np.max(np.abs(rb.y - 1.0)) < 1e-12

    nz = 1001
    nvar = 2
    kappa = 1.0
    delx = 1.0 / nz
    delt = 10.0 * delx * delx / kappa
    midpt = 500
    y = np.zeros(nz * nvar, dtype=np.float64)
    y[2 * midpt] = 1.0

    lblk = np.zeros((nz, nvar, nvar), dtype=np.float64)
    dblk = np.zeros((nz, nvar, nvar), dtype=np.float64)
    ublk = np.zeros((nz, nvar, nvar), dtype=np.float64)
    for k in range(nz):
        dblk[k, 0, 0] = 1.0 / delt
        dblk[k, 0, 1] = -1.0 / delx
        if k < nz - 1:
            ublk[k, 0, 1] = 1.0 / delx
        else:
            dblk[k, 0, 0] += kappa / (delx * delx)
        if k > 0:
            dblk[k, 1, 1] = 1.0
            dblk[k, 1, 0] = kappa / delx
            lblk[k, 1, 0] = -kappa / delx
        else:
            dblk[k, 1, 1] = 1.0

    cfg = num.NewtonConfig(
        max_iter=20, tol_correction_norm=1e-11, tol_residual_norm=1e-12
    )
    for _ in range(10):
        y_old = y.copy()

        def eval_res(
            y_cur: npt.NDArray[np.float64],
            y_prev: npt.NDArray[np.float64] = y_old,
        ) -> npt.NDArray[np.float64]:
            yv = y_cur.reshape((nz, 2))
            yp = y_prev.reshape((nz, 2))
            c = yv[:, 0]
            f = yv[:, 1]
            f_next = np.empty_like(f)
            f_next[:-1] = f[1:]
            f_next[-1] = kappa * c[-1] / delx
            r0 = (c - yp[:, 0]) / delt + (f_next - f) / delx
            r1 = np.empty_like(f)
            r1[0] = f[0]
            r1[1:] = f[1:] + kappa * (c[1:] - c[:-1]) / delx
            return np.stack([r0, r1], axis=1).ravel()

        sol = num.newton_solve_block_tridiagonal(
            nvar, nz, y, eval_res, lambda _y: (lblk, dblk, ublk), cfg
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
    assert abs(rb - target ** (1.0 / 3.0)) < 1e-12
    assert abs(rb - rn) < 1e-12


def test_jax_implicit_autodiff_for_safe_root_and_newton() -> None:
    """Verifies jax.grad and jax.jacrev on safe_root_jax and newton_solve."""

    def f_scalar(x: jax.Array, p: jax.Array) -> jax.Array:
        return x**3 - p[0] * x - p[1]

    p0 = jnp.array([2.0, 5.0], dtype=jnp.float64)
    x_star = num.safe_root_jax(f_scalar, 1.5, 3.0, p0)
    grad_p = jax.grad(lambda p: num.safe_root_jax(f_scalar, 1.5, 3.0, p))(p0)
    denom = 3.0 * float(x_star) ** 2 - 2.0
    assert abs(float(grad_p[0]) - float(x_star) / denom) < 1e-11
    assert abs(float(grad_p[1]) - 1.0 / denom) < 1e-11

    def res_fn(y: jax.Array, p: jax.Array) -> jax.Array:
        return jnp.array([y[0] ** 2 + y[1] - p[0], y[0] + y[1] ** 2 - p[1]])

    p_sys = jnp.array([3.0, 5.0], dtype=jnp.float64)
    y_init = jnp.array([1.2, 1.8], dtype=jnp.float64)
    y_sol = num.newton_solve_jax(res_fn, y_init, p_sys)
    jac_p = jax.jacrev(lambda p: num.newton_solve_jax(res_fn, y_init, p))(p_sys)
    jy = jnp.array(
        [[2.0 * y_sol[0], 1.0], [1.0, 2.0 * y_sol[1]]], dtype=jnp.float64
    )
    expected_jac = jnp.linalg.inv(jy)
    assert float(jnp.max(jnp.abs(jac_p - expected_jac))) < 1e-11
