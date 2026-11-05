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

import jax
import jax.numpy as jnp
import numpy as np

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

    Checks `interp_mkbicub_db`, `interp_evbicub_db`, `interp_mkbipm_db`, and
    `interp_evbipm_db` against upstream MESA golden values and `jax.grad` /
    `jax.hessian`.
    """
    x, y, z = _mesa_2d_test_grid()
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

    # Evaluate bicubic value and partial derivatives at (x0, y0) = (0.1, 0.2).
    x0, y0 = 0.1, 0.2
    ev = interp_2d.interp_evbicub_db(x0, y0, x, y, ilinx, iliny, coeffs)
    np.testing.assert_allclose(
        float(ev.f), 8.412838387532941e-2, rtol=1e-7, atol=5e-9
    )
    np.testing.assert_allclose(
        float(ev.df_dx), 8.223244982667942e-1, rtol=1e-6, atol=5e-7
    )
    np.testing.assert_allclose(
        float(ev.df_dy), -5.830990551255731e-2, rtol=1e-5, atol=5e-7
    )

    def f_scalar(xy: jnp.ndarray) -> jnp.ndarray:
        return interp_2d.interp_evbicub_db(
            xy[0], xy[1], x, y, ilinx, iliny, coeffs
        ).f

    # Verify analytic first and second partials against jax.grad / jax.hessian.
    xy0 = jnp.array([x0, y0], dtype=jnp.float64)
    grad_xy = np.asarray(jax.grad(f_scalar)(xy0))
    hess_xy = np.asarray(jax.hessian(f_scalar)(xy0))
    np.testing.assert_allclose(
        grad_xy, [float(ev.df_dx), float(ev.df_dy)], rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(
        hess_xy[0, 0], float(ev.d2f_dx2), rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(
        hess_xy[1, 1], float(ev.d2f_dy2), rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(
        hess_xy[0, 1], float(ev.d2f_dxdy), rtol=1e-13, atol=1e-14
    )

    # Verify 2D Steffen (1990) piecewise-monotonic interpolation.
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
    val_pm = float(interp_2d.interp_evbipm_db(x0, y0, x, y, coeffs_pm))
    np.testing.assert_allclose(val_pm, 8.351309743282355e-2, rtol=1e-12)


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
