# SPDX-License-Identifier: LGPL-3.0-or-later
"""Golden, autodiff, and property tests for `flexmesa.interp_1d`.

Validates FlexMESA 1D piecewise-monotonic cubic spline builders (`interp_pm`,
`interp_m3a`, `interp_m3b`, `interp_m3q`, `interp_4pt_pm`), evaluators
(`interp_values`, `interp_value_and_slope`), definite integrator
(`integrate_values`), and Lagrangian remeshing stencils (`interp_3_to_1`,
`interp_3_to_2`, `interp_4_to_1`) against upstream MESA test outputs (`test1`,
`test2`, `test3`), `jax.grad`, and `DualOrder1` forward-mode autodiff.
"""

from __future__ import annotations

import math

import hypothesis
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st

from flexmesa import auto_diff, interp_1d

# Upstream MESA 100-point test vector `u` (`test_interp_1d.f90`).
# Represents a smooth non-monotonic bell-shaped profile used by `test1`
# (Huynh 1993 `interp_m3q`) and `test3` (Steffen 1990 `interp_pm`).
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
    """Verifies Huynh (1993) quartic MP cubic against upstream `test1`.

    Checks both strictly increasing (`sign = +1`) and strictly decreasing
    (`sign = -1`) knot grids against the pinned upstream MESA `test1`
    coefficients (`coeffs[:3, 1]`) and evaluated values on the 100-point test
    profile `_TEST_U`. Under grid reflection $x \\to -x$, odd-order derivatives
    flip sign while interpolated function values remain invariant.
    """
    dx = 0.594059405940594
    u_arr = np.asarray(_TEST_U, dtype=np.float64)
    for sign in (1.0, -1.0):
        init_xs = sign * dx * np.arange(100, dtype=np.float64)
        coeffs = np.asarray(interp_1d.interp_m3q(init_xs, u_arr))
        # Compare first three knot slopes f'(x_0), f'(x_1), f'(x_2).
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
        # Compare interpolated values at the four upstream test abscissas.
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
    """Verifies `interp_m3q` and `integrate_values` on `test2(true/false)`.

    Confirms that definite integration over adjacent intervals reproduces the
    exact symmetry of ``sin(x)`` on both ascending and descending grids. Note
    that upstream MESA's `do_integrate_values` had a decreasing-grid bug
    (`if (x1 > x0)` instead of `if (x1 /= x0)`); FlexMESA fixes this so both
    grid orientations yield identical positive segment integrals.
    """
    n = 43
    dx = 1.0 / float(n - 1)
    for sign in (1.0, -1.0):
        init_xs = sign * 4.0 * dx * np.arange(n, dtype=np.float64)
        coeffs = interp_1d.interp_m3q(init_xs, np.sin(init_xs))
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

        # Definite integral of sin(sign * x) over sign * [xa, xb] is invariant.
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
    """Verifies Steffen (1990) `test3`, `interp_4pt_pm`, and `interp_3_to_2`.

    Checks `interp_pm`, `interp_pm_slopes_only`, `interp_4pt_pm`,
    `interp_3_to_2`, and `interp_3_to_1` against upstream MESA golden values,
    confirming that `interp_pm_slopes_only` matches the first-derivative column
    of `interp_pm` to machine precision and zeroes higher-order columns.
    """
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

    # Check 4-point local Steffen interpolation around midpoint of [x1, x2].
    x4 = np.array(
        [
            math.pi / 3.0,
            math.pi / 2.0,
            2.0 * math.pi / 3.0,
            3.0 * math.pi / 4.0,
        ]
    )
    y4 = np.sin(x4)
    a = interp_1d.interp_4pt_pm(x4, y4)
    dx_q = 0.5 * (x4[2] - x4[1])
    val_4pt = float(y4[1] + dx_q * (a[0] + dx_q * (a[1] + dx_q * a[2])))
    np.testing.assert_allclose(val_4pt, 9.6508141383418289e-1, rtol=1e-14)

    # Check 3-to-2 and 3-to-1 Lagrangian mesh remeshing helpers.
    nf1, nf2 = interp_1d.interp_3_to_2(
        2.0114947208182310e-3,
        2.0097307373083619e-3,
        1.5784933404892154e-3,
        1.3270582904786616e-3,
        1.8984458478374221e30,
        1.4579738233866260e30,
        1.1136297079131214e30,
    )
    np.testing.assert_allclose(nf1, 1.5446941380706298e30, rtol=1e-14)
    np.testing.assert_allclose(nf2, 1.2929705208875774e30, rtol=1e-14)

    nf_3to1 = interp_1d.interp_3_to_1(
        2.0114947208182310e-3,
        2.0097307373083619e-3,
        1.5784933404892154e-3,
        1.8984458478374221e30,
        1.4579738233866260e30,
        1.1136297079131214e30,
    )
    np.testing.assert_allclose(nf_3to1, 1.5446941380706298e30, rtol=1e-14)


def test_interp1d_jax_grad_and_dual_order1_match_analytic_derivatives() -> None:
    """Verifies `jax.grad` and `DualOrder1` partials against spline calculus.

    Evaluates derivatives at interior query points (`0.35`, `1.42`, `2.68`) and
    confirms exact parity between:
    1. Analytic spline first derivatives `interp_value_and_slope`,
    2. Reverse-mode automatic differentiation `jax.grad`, and
    3. Forward-mode dual-number propagation `interpolate_vector_pm_autodiff`.
    """
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

        # Propagate forward-mode dual numbers with respect to query abscissa xq.
        x_ad = [
            auto_diff.AutoDiffReal2VarOrder1.constant(float(xk))
            for xk in x_grid
        ]
        y_ad = [
            auto_diff.AutoDiffReal2VarOrder1.constant(float(yk))
            for yk in y_grid
        ]
        # Seed independent variable 0 at xq and verify primal and derivative.
        q_ad = [auto_diff.AutoDiffReal2VarOrder1.independent(xq, 0)]
        out_ad = interp_1d.interpolate_vector_pm_autodiff(x_ad, y_ad, q_ad)[0]
        np.testing.assert_allclose(out_ad.val, float(val), rtol=1e-14)
        np.testing.assert_allclose(out_ad.d1[0], float(slope), rtol=1e-13)


@hypothesis.given(
    slope=st.floats(min_value=-5.0, max_value=5.0),
    intercept=st.floats(min_value=-10.0, max_value=10.0),
)
@hypothesis.settings(max_examples=25, deadline=None)
def test_prop_interp1d_reproduces_linear_functions_exactly(
    slope: float,
    intercept: float,
) -> None:
    """Property test: all 1D spline builders reproduce linear functions.

    For any linear profile $y(x) = m x + b$ on a strictly ascending non-uniform
    knot grid, Steffen (1990) and Huynh (1993) slope limiters preserve the
    constant secant slope $m$ and zero out quadratic and cubic curvature terms.
    """
    x = np.array([0.0, 0.7, 1.9, 3.2, 5.0, 8.5], dtype=np.float64)
    y = slope * x + intercept
    xq = np.linspace(0.0, 8.5, 15, dtype=np.float64)
    expected = slope * xq + intercept
    # Evaluate each of the four 1D cubic spline builders on the linear data.
    # Both Steffen (1990) and Huynh (1993) limiters must preserve exact
    # linearity across all interior and boundary intervals.
    for builder in (
        interp_1d.interp_pm,
        interp_1d.interp_m3a,
        interp_1d.interp_m3b,
        interp_1d.interp_m3q,
    ):
        coeffs = builder(x, y)
        vals = np.asarray(interp_1d.interp_values(x, coeffs, xq))
        np.testing.assert_allclose(vals, expected, rtol=1e-12, atol=1e-12)
