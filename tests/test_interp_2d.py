# SPDX-License-Identifier: LGPL-3.0-or-later
"""Golden, autodiff, and property tests for `flexmesa.interp_2d`.

Validates FlexMESA 2D bicubic splines (`interp_mkbicub_db`,
`interp_evbicub_db`, `interp_evbicub_mask_db`, `interp_mkspline_db`), 2D
Steffen (1990) piecewise-monotonic splines (`interp_mkbipm_db`,
`interp_evbipm_db`), and Akima (1996, ACM Algorithm 760) bivariate surface
fitting (`interp_rgbi3p_db`, `interp_rgsf3p_db`, `rgpd3p_db`) against upstream
MESA test outputs (`test2D_db`, `test_akima_db`) and `jax.grad`/`jax.hessian`.
"""

from __future__ import annotations

import math

import hypothesis
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st

from flexmesa import interp_2d


def _mesa_2d_test_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds the 31x35 2D test grid from upstream MESA `test2D_db`.

    Evaluates $z(x, y) = \\exp(-r / r_0) \\sin(x) \\cos(y)$ with
    $r = \\sqrt{x^2 + y^2}$ and $r_0 = \\pi / 2$ on uniform grids over
    $[-\\pi, \\pi]$.
    """
    x = np.linspace(-math.pi, math.pi, 31, dtype=np.float64)
    y = np.linspace(-math.pi, math.pi, 35, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(xx**2 + yy**2)
    r0 = 0.5 * math.pi
    z = np.exp(-r / r0) * np.sin(xx) * np.cos(yy)
    return x, y, z


def test_upstream_mesa_interp2d_bicub_and_bipm_match_golden() -> None:
    """Verifies 2D bicubic and piecewise-monotonic interpolation and autodiff.

    Checks `interp_mkspline_db`, `interp_mkbicub_db`, `interp_evbicub_db`,
    `interp_evbicub_mask_db`, `interp_mkbipm_db`, and `interp_evbipm_db`
    against all 6 upstream MESA `test2D_db` golden points and `jax.grad` /
    `jax.hessian`.
    """
    x, y, z = _mesa_2d_test_grid()

    # Verify 1D C^2 spline helper `interp_mkspline_db` reproduces quadratic f''.
    fxx_quad, uniform_x = interp_2d.interp_mkspline_db(x, 3.0 * x**2 - 2.0 * x)
    assert uniform_x
    np.testing.assert_allclose(
        np.asarray(fxx_quad), 6.0, rtol=1e-13, atol=1e-13
    )

    coeffs, ilinx, iliny = interp_2d.interp_mkbicub_db(x, y, z)
    assert ilinx and iliny

    # Check interior bicubic spline coefficients [f, f_xx, f_yy, f_xxyy].
    c_mid = np.asarray(coeffs[14, 16])
    expected_mid = np.array(
        [
            -1.7107869940376103e-1,
            1.2297098132743365,
            3.2112489619292410e-1,
            4.7603778557260390e-1,
        ]
    )
    np.testing.assert_allclose(c_mid, expected_mid, rtol=2e-12, atol=1e-13)

    x_min, x_max = 0.1, 0.8 * math.pi
    y_min, y_max = 0.2, 0.6 * math.pi
    dx = x_max - x_min
    dy = 0.5 * (y_max - y_min)

    # Pinned (i, j, z, dz/dx, dz/dy) points from upstream test2D_db(.true.).
    expected_bicub = (
        (
            0,
            0,
            8.412838387532941e-2,
            8.223244982667942e-1,
            -5.830990551255731e-2,
        ),
        (
            1,
            0,
            1.1571939326516818e-1,
            -2.327225412452813e-1,
            -2.9310550186889182e-2,
        ),
        (
            0,
            1,
            2.58331372196636e-2,
            2.5594985035557577e-1,
            -6.062774870752535e-2,
        ),
        (
            1,
            1,
            5.2413685332749614e-2,
            -1.0296823150811425e-1,
            -1.0258873034645082e-1,
        ),
        (
            0,
            2,
            -9.275811236254504e-3,
            -9.214212823541054e-2,
            -2.265013757433625e-2,
        ),
        (
            1,
            2,
            -2.458149485380903e-2,
            4.635499485797099e-2,
            -6.625610809426089e-2,
        ),
    )

    # Evaluate each of the 6 upstream test2D_db(.true.) sample points and
    # verify both full BicubicEval and masked interp_evbicub_mask_db outputs.
    # Note that upstream MESA's fvbicub_db omits `implicit none` and evaluates
    # its local basis polynomials in 32-bit float; our full float64 basis
    # agrees with upstream within single-precision truncation (~5e-7) while
    # matching exact polynomial calculus and jax.grad/jax.hessian to ~1e-13.
    for i, j, exp_z, exp_dzdx, exp_dzdy in expected_bicub:
        xq = x_min + float(i) * dx
        yq = y_min + float(j) * dy
        ev = interp_2d.interp_evbicub_db(xq, yq, x, y, ilinx, iliny, coeffs)
        np.testing.assert_allclose(float(ev.f), exp_z, rtol=1e-7, atol=5e-9)
        np.testing.assert_allclose(
            float(ev.df_dx), exp_dzdx, rtol=1e-5, atol=5e-7
        )
        np.testing.assert_allclose(
            float(ev.df_dy), exp_dzdy, rtol=1e-5, atol=5e-7
        )
        masked = np.asarray(
            interp_2d.interp_evbicub_mask_db(
                xq, yq, x, y, ilinx, iliny, coeffs, [1, 1, 1, 0, 0, 0]
            )
        )
        np.testing.assert_allclose(
            masked[:3],
            [float(ev.f), float(ev.df_dx), float(ev.df_dy)],
            rtol=1e-15,
            atol=1e-15,
        )
        np.testing.assert_allclose(masked[3:], 0.0, atol=1e-15)

    def f_scalar(xy: jnp.ndarray) -> jnp.ndarray:
        return interp_2d.interp_evbicub_db(
            xy[0], xy[1], x, y, ilinx, iliny, coeffs
        ).f

    # Verify analytic first and second partials against jax.grad / jax.hessian.
    x0, y0 = 0.1, 0.2
    ev0 = interp_2d.interp_evbicub_db(x0, y0, x, y, ilinx, iliny, coeffs)
    xy0 = jnp.array([x0, y0], dtype=jnp.float64)
    grad_xy = np.asarray(jax.grad(f_scalar)(xy0))
    hess_xy = np.asarray(jax.hessian(f_scalar)(xy0))
    np.testing.assert_allclose(
        grad_xy, [float(ev0.df_dx), float(ev0.df_dy)], rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(
        hess_xy[0, 0], float(ev0.d2f_dx2), rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(
        hess_xy[1, 1], float(ev0.d2f_dy2), rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(
        hess_xy[0, 1], float(ev0.d2f_dxdy), rtol=1e-13, atol=1e-14
    )

    # Verify 2D Steffen (1990) piecewise-monotonic interpolation on all 6 pts.
    # Each query evaluates four 1D x-splines on a 4-point y stencil followed by
    # a 1D Steffen cubic interpolation along y, matching test2D_db(.false.) to
    # full double precision (<= 1e-12).
    coeffs_pm = interp_2d.interp_mkbipm_db(x, y, z)
    np.testing.assert_allclose(
        np.asarray(coeffs_pm[14, 16]),
        [
            -1.7107869940376103e-1,
            7.131561165935149e-1,
            9.9011407468666e-1,
            -2.3637232381685926,
        ],
        rtol=1e-12,
    )
    expected_bipm = (
        (0, 0, 8.351309743282355e-2),
        (1, 0, 1.1572304354801646e-1),
        (0, 1, 2.580157764222351e-2),
        (1, 1, 5.2402920852187326e-2),
        (0, 2, -9.271793796082475e-3),
        (1, 2, -2.457567139911116e-2),
    )
    for i, j, exp_z in expected_bipm:
        xq = x_min + float(i) * dx
        yq = y_min + float(j) * dy
        val_pm = float(interp_2d.interp_evbipm_db(xq, yq, x, y, coeffs_pm))
        np.testing.assert_allclose(val_pm, exp_z, rtol=1e-12, atol=1e-13)


def test_upstream_mesa_interp2d_akima_matches_golden_table() -> None:
    """Verifies Akima (1996) `interp_rgbi3p_db` and `interp_rgsf3p_db`.

    Compares surface and point interpolation against the published Akima (1996,
    ACM Algorithm 760) 9x11 table in upstream `test_akima_db`.
    """
    xd = np.arange(9, dtype=np.float64)
    yd = np.arange(11, dtype=np.float64)
    # Published 9x11 topographic elevation table from Akima (1996, Table I).
    zd_col_major = np.array(
        [
            [0.0] * 9,
            [0.0] * 9,
            [0.0] * 9,
            [3.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [7.4, 4.8, 1.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [12.0, 8.0, 5.3, 2.9, 0.6, 0.0, 0.0, 0.0, 0.0],
            [16.8, 14.4, 8.1, 6.9, 6.2, 0.6, 0.1, 0.0, 0.0],
            [21.8, 20.5, 12.8, 17.6, 5.8, 7.6, 0.8, 0.6, 0.6],
            [22.4, 22.5, 14.6, 22.5, 4.7, 7.2, 1.8, 2.1, 2.1],
            [37.2, 40.0, 27.0, 41.3, 14.1, 24.5, 17.3, 20.2, 20.8],
            [58.2, 61.5, 47.9, 62.3, 34.6, 45.5, 38.2, 41.2, 41.7],
        ],
        dtype=np.float64,
    )
    zd = zd_col_major.T.copy()
    xi = -0.5 + 0.5 * np.arange(19, dtype=np.float64)
    yi = -0.5 + 0.5 * np.arange(23, dtype=np.float64)
    z_sf = np.asarray(interp_2d.interp_rgsf3p_db(xd, yd, zd, xi, yi))
    assert abs(float(z_sf[0, 0]) - (-0.847)) < 5e-4
    assert abs(float(z_sf[0, 1]) - 0.0) < 5e-4
    assert abs(float(z_sf[0, 22]) - 58.284) < 5e-4
    assert abs(float(z_sf[18, 22]) - 48.268) < 5e-4

    # Point evaluation `interp_rgbi3p_db` must match surface grid evaluation.
    z_bi = np.asarray(
        interp_2d.interp_rgbi3p_db(
            xd, yd, zd, np.array([xi[0], xi[18]]), np.array([yi[22], yi[22]])
        )
    )
    np.testing.assert_allclose(
        z_bi, [z_sf[0, 22], z_sf[18, 22]], rtol=1e-14, atol=1e-14
    )


@hypothesis.settings(max_examples=25, deadline=None)
@hypothesis.given(
    a00=st.floats(-5.0, 5.0),
    a10=st.floats(-3.0, 3.0),
    a01=st.floats(-3.0, 3.0),
    a20=st.floats(-2.0, 2.0),
    a11=st.floats(-2.0, 2.0),
    a02=st.floats(-2.0, 2.0),
    xq=st.floats(-1.5, 1.5),
    yq=st.floats(-1.5, 1.5),
)
def test_prop_interp2d_bicubic_reproduces_biquadratic_and_derivatives(
    a00: float,
    a10: float,
    a01: float,
    a20: float,
    a11: float,
    a02: float,
    xq: float,
    yq: float,
) -> None:
    """Property test: not-a-knot bicubic spline reproduces biquadratic surfaces.

    Verifies that `interp_mkbicub_db` and `interp_evbicub_db` reproduce any
    biquadratic polynomial $P(x, y) = a_{00} + a_{10} x + a_{01} y + a_{20} x^2
    + a_{11} x y + a_{02} y^2$ and its first and second partial derivatives to
    machine precision.
    """
    x = np.linspace(-2.0, 2.0, 7, dtype=np.float64)
    y = np.linspace(-2.0, 2.0, 8, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    z = a00 + a10 * xx + a01 * yy + a20 * xx**2 + a11 * xx * yy + a02 * yy**2
    coeffs, ilinx, iliny = interp_2d.interp_mkbicub_db(x, y, z)
    ev = interp_2d.interp_evbicub_db(xq, yq, x, y, ilinx, iliny, coeffs)

    f_exact = (
        a00 + a10 * xq + a01 * yq + a20 * xq**2 + a11 * xq * yq + a02 * yq**2
    )
    df_dx_exact = a10 + 2.0 * a20 * xq + a11 * yq
    df_dy_exact = a01 + a11 * xq + 2.0 * a02 * yq

    np.testing.assert_allclose(float(ev.f), f_exact, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        float(ev.df_dx), df_dx_exact, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        float(ev.df_dy), df_dy_exact, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        float(ev.d2f_dx2), 2.0 * a20, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        float(ev.d2f_dy2), 2.0 * a02, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(float(ev.d2f_dxdy), a11, rtol=1e-11, atol=1e-11)


def test_renka790_cshep2_cubic_and_scattered_2d_parity() -> None:
    """Verify ACM TOMS 790 (`cshep2`/`qshep2`) cubic and scattered 2D parity."""
    # Exact cubic reproduction from test_renka790_db.f (N=20, NC=10, NW=16)

    def fq(xx: float, yy: float) -> float:
        return ((xx + 2.0 * yy) / 3.0) ** 3

    def fx(xx: float, yy: float) -> float:
        return ((xx + 2.0 * yy) / 3.0) ** 2

    def fy(xx: float, yy: float) -> float:
        return 2.0 * ((xx + 2.0 * yy) / 3.0) ** 2

    def fxx(xx: float, yy: float) -> float:
        return 2.0 * (xx + 2.0 * yy) / 9.0

    def fxy(xx: float, yy: float) -> float:
        return 4.0 * (xx + 2.0 * yy) / 9.0

    def fyy(xx: float, yy: float) -> float:
        return 8.0 * (xx + 2.0 * yy) / 9.0

    xs_c, ys_c, fs_c = [], [], []
    for j in range(5):
        yj = j / 4.0
        for i in range(4):
            xi = i / 3.0
            xs_c.append(xi)
            ys_c.append(yj)
            fs_c.append(fq(xi, yj))

    cshep = interp_2d.interp_cshep2_db(
        np.array(xs_c), np.array(ys_c), np.array(fs_c), nc=10, nw=16, nr=3
    )
    qshep = interp_2d.qshep2(
        np.array(xs_c), np.array(ys_c), np.array(fs_c), nc=10, nw=16, nr=3
    )

    for py in np.linspace(0.0, 1.0, 11):
        for px in np.linspace(0.0, 1.0, 11):
            c_val = interp_2d.interp_cs2val_db(cshep, float(px), float(py))
            q_val = interp_2d.qs2val(qshep, float(px), float(py))
            cg, cx, cy = interp_2d.interp_cs2grd_db(cshep, float(px), float(py))
            qg, qx, qy = interp_2d.qs2grd(qshep, float(px), float(py))
            ch, chx, chy, chxx, chxy, chyy = interp_2d.cs2hes_db(
                cshep, float(px), float(py)
            )
            assert abs(c_val - fq(px, py)) < 1e-13
            assert abs(q_val - c_val) < 1e-15
            assert abs(cg - c_val) < 1e-14 and abs(qg - q_val) < 1e-14
            assert abs(cx - fx(px, py)) < 1e-12 and abs(qx - cx) < 1e-15
            assert abs(cy - fy(px, py)) < 1e-12 and abs(qy - cy) < 1e-15
            assert abs(ch - c_val) < 1e-14
            assert abs(chx - cx) < 1e-14 and abs(chy - cy) < 1e-14
            assert abs(chxx - fxx(px, py)) < 1e-11
            assert abs(chxy - fxy(px, py)) < 1e-11
            assert abs(chyy - fyy(px, py)) < 1e-11

    # Franke's 33-node set (KS=2, N=33, NC=17, NW=30, NR=6)
    x2 = np.array(
        [
            0.05,
            0.00,
            0.00,
            0.00,
            0.10,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.50,
            0.50,
            0.55,
            0.60,
            0.60,
            0.60,
            0.65,
            0.70,
            0.70,
            0.70,
            0.75,
            0.75,
            0.75,
            0.80,
            0.80,
            0.85,
            0.90,
            0.90,
            0.95,
            1.00,
            1.00,
            1.00,
        ],
        dtype=np.float64,
    )
    y2 = np.array(
        [
            0.45,
            0.50,
            1.00,
            0.00,
            0.15,
            0.75,
            0.30,
            0.10,
            0.20,
            0.35,
            0.85,
            0.00,
            1.00,
            0.95,
            0.25,
            0.65,
            0.85,
            0.70,
            0.20,
            0.65,
            0.90,
            0.10,
            0.35,
            0.85,
            0.40,
            0.65,
            0.25,
            0.35,
            0.80,
            0.90,
            0.00,
            0.50,
            1.00,
        ],
        dtype=np.float64,
    )

    def tstfn1(
        xx: np.ndarray | float, yy: np.ndarray | float
    ) -> np.ndarray | float:
        return (
            0.75
            * np.exp(-((9.0 * xx - 2.0) ** 2 + (9.0 * yy - 2.0) ** 2) / 4.0)
            + 0.75
            * np.exp(-((9.0 * xx + 1.0) ** 2) / 49.0 - (9.0 * yy + 1.0) / 10.0)
            + 0.5
            * np.exp(-((9.0 * xx - 7.0) ** 2 + (9.0 * yy - 3.0) ** 2) / 4.0)
            - 0.2 * np.exp(-((9.0 * xx - 4.0) ** 2) - (9.0 * yy - 7.0) ** 2)
        )

    w2 = np.asarray(tstfn1(x2, y2), dtype=np.float64)
    franke = interp_2d.interp_cshep2_db(x2, y2, w2, nc=17, nw=30, nr=6)
    p_grid = np.linspace(0.0, 1.0, 33)
    errs = [
        abs(
            interp_2d.interp_cs2val_db(franke, float(px), float(py))
            - float(tstfn1(px, py))
        )
        for px in p_grid
        for py in p_grid
    ]
    assert abs(max(errs) - 0.1985167087686014) < 1e-12
    assert abs(float(np.mean(errs)) - 0.03217025005461399) < 1e-12

    # Scattered 2D interpolation on MESA's 31x35 test2D_db grid at 6 golden pts
    xpts, ypts, z_grid = _mesa_2d_test_grid()
    xx, yy = np.meshgrid(xpts, ypts, indexing="ij")
    scat = interp_2d.interp_cshep2_db(
        xx.ravel(), yy.ravel(), z_grid.ravel(), nc=17, nw=32, nr=10
    )
    x_min, x_max = 0.1, 0.8 * math.pi
    y_min, y_max = 0.2, 0.6 * math.pi
    dx = x_max - x_min
    dy = 0.5 * (y_max - y_min)
    golden_pts = (
        (0, 0, 0.08443031571220527, 0.8426978068643065, -0.05312016510832069),
        (1, 0, 0.11571953715487195, -0.23272372576109734, -0.0292880351263485),
        (0, 1, 0.02583107216746556, 0.25668785540470757, -0.06076899376517829),
        (1, 1, 0.05241360426423525, -0.10275593626780094, -0.10257100263948463),
        (
            0,
            2,
            -0.009280594758942013,
            -0.09261745586884203,
            -0.02266957111867004,
        ),
        (
            1,
            2,
            -0.024582273200498193,
            0.04625995292101696,
            -0.06629310142916707,
        ),
    )
    for i, j, exp_z, exp_dzdx, exp_dzdy in golden_pts:
        qx = x_min + float(i) * dx
        qy = y_min + float(j) * dy
        z_scat, dz_dx, dz_dy = interp_2d.interp_cs2grd_db(scat, qx, qy)
        r = math.hypot(qx, qy)
        z_exact = math.exp(-r / (0.5 * math.pi)) * math.sin(qx) * math.cos(qy)
        assert abs(z_scat - exp_z) < 1e-12
        assert abs(dz_dx - exp_dzdx) < 1e-12
        assert abs(dz_dy - exp_dzdy) < 1e-12
        assert abs(z_scat - z_exact) < 5e-4
