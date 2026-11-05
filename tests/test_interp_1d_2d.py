# SPDX-License-Identifier: LGPL-3.0-or-later
"""Golden, autodiff, and property tests for `interp_1d` and `interp_2d`."""

from __future__ import annotations

import math

import hypothesis
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st

from flexmesa import auto_diff, interp_1d, interp_2d

# fmt: off
_TEST_U: tuple[float, ...] = (
    0.3691382918814334, 0.3695899829574669, 0.3699633032242006,
    0.3709860746618878, 0.3724175838188959, 0.3742482756438662,
    0.3764768199547272, 0.3790910321656262, 0.3820871425096383,
    0.3854500131142883, 0.3891727128876379, 0.3932377816410259,
    0.3976355585696444, 0.4023461172079230, 0.4073570798427897,
    0.4126461990878889, 0.4181985492965749, 0.4239897218739848,
    0.4300024071530404, 0.4362102270543894, 0.4425936978527515,
    0.4491247203356548, 0.4557819260544076, 0.4625358769752868,
    0.4693638888826394, 0.4762362700345437, 0.4831315999154387,
    0.4900124769496845, 0.4968509249135122, 0.5036231656397859,
    0.5102978108992456, 0.5168503329341820, 0.5232493058916751,
    0.5294708374154181, 0.5354843215442489, 0.5412671682413143,
    0.5467901937111218, 0.5520326792853251, 0.5569674077513700,
    0.5615760727117703, 0.5658339305000317, 0.5697255981615711,
    0.5732292848999179, 0.5763330504836860, 0.5790185013283228,
    0.5812774415045446, 0.5830949206727917, 0.5844671751637984,
    0.5853828024764719, 0.5858432670435806, 0.5858432670435451,
    0.5853828024764413, 0.5844671751637710, 0.5830949206727681,
    0.5812774415045240, 0.5790185013283052, 0.5763330504836706,
    0.5732292848999045, 0.5697255981615594, 0.5658339305000216,
    0.5615760727117616, 0.5569674077513626, 0.5520326792853185,
    0.5467901937111161, 0.5412671682413093, 0.5354843215442447,
    0.5294708374154145, 0.5232493058916718, 0.5168503329341793,
    0.5102978108992431, 0.5036231656397836, 0.4968509249135103,
    0.4900124769496828, 0.4831315999154373, 0.4762362700345427,
    0.4693638888826384, 0.4625358769752859, 0.4557819260544067,
    0.4491247203356541, 0.4425936978527509, 0.4362102270543888,
    0.4300024071530398, 0.4239897218739841, 0.4181985492965744,
    0.4126461990878886, 0.4073570798427895, 0.4023461172079228,
    0.3976355585696443, 0.3932377816410258, 0.3891727128876379,
    0.3854500131142883, 0.3820871425096385, 0.3790910321656264,
    0.3764768199547278, 0.3742482756438669, 0.3724175838188968,
    0.3709860746618888, 0.3699633032242018, 0.3695899829574566,
    0.3691382918814347,
)
# fmt: on


def test_upstream_mesa_interp1d_test1_m3q_matches_golden_output() -> None:
    """Verifies Huynh (1993) quartic MP cubic against upstream `test1`."""
    dx = 0.594059405940594
    u_arr = np.asarray(_TEST_U, dtype=np.float64)
    for sign in (1.0, -1.0):
        init_xs = sign * dx * np.arange(100, dtype=np.float64)
        coeffs = np.asarray(interp_1d.interp_m3q(init_xs, u_arr))
        np.testing.assert_allclose(
            coeffs[:3, 1],
            sign
            * np.array(
                [
                    6.2842244900173001e-4,
                    6.6140349791541427e-4,
                    1.1067133492894063e-3,
                ]
            ),
            rtol=1e-12,
            atol=1e-14,
        )
        xs = sign * np.array([10.0, 10.4, 10.8, 11.0], dtype=np.float64)
        vals = np.asarray(interp_1d.interp_values(init_xs, coeffs, xs))
        expected = np.array(
            [
                4.2300860237857557e-1,
                4.2701010015389296e-1,
                4.3110602983550195e-1,
                4.3318662265017888e-1,
            ]
        )
        np.testing.assert_allclose(vals, expected, rtol=1e-12, atol=1e-14)


def test_upstream_mesa_interp1d_test2_sin_and_integral_match_golden() -> None:
    """Verifies `interp_m3q` and `integrate_values` on `test2(true/false)`."""
    n = 43
    dx = 1.0 / float(n - 1)
    for sign in (1.0, -1.0):
        init_xs = sign * 4.0 * dx * np.arange(n, dtype=np.float64)
        u = np.sin(init_xs)
        coeffs = interp_1d.interp_m3q(init_xs, u)
        xs = sign * np.array(
            [
                0.0,
                math.pi / 4.0,
                math.pi / 3.0,
                math.pi / 2.0,
                2.0 * math.pi / 3.0,
                3.0 * math.pi / 4.0,
                19.0 * math.pi / 20.0,
            ],
            dtype=np.float64,
        )
        vals = np.asarray(interp_1d.interp_values(init_xs, coeffs, xs))
        expected_vals = sign * np.array(
            [
                0.0,
                7.0710667730056076e-1,
                8.6602540429696606e-1,
                9.9999978275888857e-1,
                8.6602540242370141e-1,
                7.0710667151355655e-1,
                1.5643445870792264e-1,
            ]
        )
        np.testing.assert_allclose(vals, expected_vals, rtol=1e-12, atol=1e-14)

        integrals = np.asarray(interp_1d.integrate_values(init_xs, coeffs, xs))
        expected_integrals = np.array(
            [
                0.0,
                2.9288520516627042e-1,
                2.0710675623584285e-1,
                4.9999994192722058e-1,
                4.9999994167149031e-1,
                2.0710675639571818e-1,
                2.8058152844437595e-1,
            ]
        )
        np.testing.assert_allclose(
            integrals, expected_integrals, rtol=1e-12, atol=1e-14
        )


def test_upstream_mesa_interp1d_test3_4pt_and_remesh_match_golden() -> None:
    """Verifies Steffen (1990) `test3`, `interp_4pt_pm`, and `interp_3_to_2`."""
    dx = 0.594059405940594
    u_arr = np.asarray(_TEST_U, dtype=np.float64)
    for sign in (1.0, -1.0):
        init_xs = sign * dx * np.arange(100, dtype=np.float64)
        coeffs = np.asarray(interp_1d.interp_pm(init_xs, u_arr))
        slopes_only = np.asarray(
            interp_1d.interp_pm_slopes_only(init_xs, u_arr)
        )
        np.testing.assert_allclose(slopes_only[:, 1], coeffs[:, 1], rtol=1e-15)
        np.testing.assert_allclose(slopes_only[:, 2:], 0.0, atol=1e-15)

        xs = sign * np.array([10.0, 10.4, 10.8], dtype=np.float64)
        vals = np.asarray(interp_1d.interp_values(init_xs, coeffs, xs))
        expected = np.array(
            [
                4.2300894327454436e-1,
                4.2701013523153308e-1,
                4.3110567262196459e-1,
            ]
        )
        np.testing.assert_allclose(vals, expected, rtol=1e-12, atol=1e-14)

    x_full = np.array(
        [
            math.pi / 4.0,
            math.pi / 3.0,
            math.pi / 2.0,
            2.0 * math.pi / 3.0,
            3.0 * math.pi / 4.0,
            19.0 * math.pi / 20.0,
        ]
    )
    y_full = np.sin(x_full)
    x4 = x_full[1:5]
    y4 = y_full[1:5]
    x_mid = 0.5 * (x4[1] + x4[2])
    a = interp_1d.interp_4pt_pm(x4, y4)
    dx_q = x_mid - x4[1]
    val_4pt = float(y4[1] + dx_q * (a[0] + dx_q * (a[1] + dx_q * a[2])))
    np.testing.assert_allclose(val_4pt, 9.6508141383418289e-1, rtol=1e-14)

    nf1, nf2 = interp_1d.interp_3_to_2(
        2.0114947208182310e-3,
        2.0097307373083619e-3,
        0.5 * 2.0114947208182310e-3,
        0.5 * 2.0114947208182310e-3,
        2.3370198042006560e30,
        1.8984458478374221e30,
        1.4579738233866260e30,
    )
    nf_3to1 = interp_1d.interp_3_to_1(
        2.0114947208182310e-3,
        2.0097307373083619e-3,
        0.5 * 2.0114947208182310e-3,
        2.3370198042006560e30,
        1.8984458478374221e30,
        1.4579738233866260e30,
    )
    np.testing.assert_allclose(nf_3to1, nf1, rtol=1e-15)
    assert math.isfinite(nf2)


def test_interp1d_jax_grad_and_dual_order1_match_analytic_derivatives() -> None:
    """Verifies `jax.grad` and `DualOrder1` partials against spline calculus."""
    x_grid = jnp.linspace(0.0, 3.0, 16, dtype=jnp.float64)
    y_grid = jnp.sin(x_grid) + 0.2 * x_grid**2
    coeffs = interp_1d.interp_pm(x_grid, y_grid)

    for xq in (0.35, 1.42, 2.68):
        val, slope = interp_1d.interp_value_and_slope(x_grid, coeffs, xq)
        grad_x = float(
            jax.grad(lambda z: interp_1d.interp_value(x_grid, coeffs, z))(
                jnp.float64(xq)
            )
        )
        np.testing.assert_allclose(grad_x, float(slope), rtol=1e-13, atol=1e-14)

        x_ad = [
            auto_diff.AutoDiffReal2VarOrder1.constant(float(xk))
            for xk in x_grid
        ]
        y_ad = [
            auto_diff.AutoDiffReal2VarOrder1.constant(float(yk))
            for yk in y_grid
        ]
        q_ad = [auto_diff.AutoDiffReal2VarOrder1.independent(xq, 0)]
        out_ad = interp_1d.interpolate_vector_pm_autodiff(x_ad, y_ad, q_ad)[0]
        np.testing.assert_allclose(out_ad.val, float(val), rtol=1e-14)
        np.testing.assert_allclose(out_ad.d1[0], float(slope), rtol=1e-13)


def _mesa_2d_test_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-math.pi, math.pi, 31, dtype=np.float64)
    y = np.linspace(-math.pi, math.pi, 35, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(xx**2 + yy**2)
    r0 = 0.5 * math.pi
    z = np.exp(-r / r0) * np.sin(xx) * np.cos(yy)
    return x, y, z


def test_upstream_mesa_interp2d_bicub_bipm_and_akima_match_golden() -> None:
    """Verifies 2D bicubic, piecewise-monotonic, and Akima surface fitting."""
    x, y, z = _mesa_2d_test_grid()
    coeffs, ilinx, iliny = interp_2d.interp_mkbicub_db(x, y, z)
    assert ilinx and iliny

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

    # Verify JAX first and second derivatives match `interp_evbicub_db`.
    def f_scalar(xy: jnp.ndarray) -> jnp.ndarray:
        return interp_2d.interp_evbicub_db(
            xy[0], xy[1], x, y, ilinx, iliny, coeffs
        ).f

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

    # Verify `interp_mkbipm_db` and `interp_evbipm_db` against upstream golden.
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

    # Verify Akima (1996) `interp_rgbi3p_db` and `interp_rgsf3p_db` against
    # upstream `test_akima_db`.
    xd = np.arange(9, dtype=np.float64)
    yd = np.arange(11, dtype=np.float64)
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

    z_bi = np.asarray(
        interp_2d.interp_rgbi3p_db(
            xd, yd, zd, np.array([xi[0], xi[18]]), np.array([yi[22], yi[22]])
        )
    )
    np.testing.assert_allclose(
        z_bi, [z_sf[0, 22], z_sf[18, 22]], rtol=1e-14, atol=1e-14
    )


@hypothesis.given(
    slope=st.floats(min_value=-5.0, max_value=5.0),
    intercept=st.floats(min_value=-10.0, max_value=10.0),
)
@hypothesis.settings(max_examples=25, deadline=None)
def test_prop_interp1d_reproduces_linear_functions_exactly(
    slope: float,
    intercept: float,
) -> None:
    """Property test: all 1D spline builders reproduce linear functions."""
    x = np.array([0.0, 0.7, 1.9, 3.2, 5.0, 8.5], dtype=np.float64)
    y = slope * x + intercept
    xq = np.linspace(0.0, 8.5, 15, dtype=np.float64)
    expected = slope * xq + intercept
    for builder in (
        interp_1d.interp_pm,
        interp_1d.interp_m3a,
        interp_1d.interp_m3b,
        interp_1d.interp_m3q,
    ):
        coeffs = builder(x, y)
        vals = np.asarray(interp_1d.interp_values(x, coeffs, xq))
        np.testing.assert_allclose(vals, expected, rtol=1e-12, atol=1e-12)
