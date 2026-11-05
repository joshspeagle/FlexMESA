# SPDX-License-Identifier: LGPL-3.0-or-later
"""Golden parity, JAX autodiff, and property tests for `math` and `auto_diff`.

Verifies all 33 functions in upstream MESA's `math/test/test_output.INTRINSIC`
to `<= 1e-15`, `test_auto_diff.f90` cases across `DualOrder1` and
`AutoDiffReal2VarOrder2` to `<= 1e-14`, `jax.test_util.check_grads` and
`jax.hessian` parity, and `hypothesis` property checks.
"""

from __future__ import annotations

from collections.abc import Sequence

import hypothesis
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st
from jax import test_util

from flexmesa import auto_diff as ad
from flexmesa import constants
from flexmesa import math as fmath
from flexmesa.testing import golden

# Standard evaluation grid used in upstream MESA's math/test.
XS: jax.Array = jnp.asarray([0.1, 0.2, 0.4], dtype=jnp.float64)

# Relative/absolute tolerances for math intrinsics and auto_diff comparisons.
TOL_MATH: float = 1.0e-15
TOL_AD: float = 1.0e-14


def _assert_close(
    actual: jax.Array | np.ndarray | Sequence[float | jax.Array],
    expected: jax.Array | np.ndarray | Sequence[float | jax.Array],
    tol: float = TOL_MATH,
) -> None:
    """Assert elementwise parity within `rtol = tol` and `atol = tol`."""
    golden.assert_values_match(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=tol,
        atol=tol,
    )


def _smooth_combo(val: jax.Array) -> jax.Array:
    """Evaluate a smooth combination of intrinsics for `check_grads`."""
    return (
        fmath.sinpi(val)
        + fmath.exp10(val)
        + fmath.safe_log10(val)
        + fmath.atanh(val)
    )


def _scalar_fn(xy: jax.Array) -> jax.Array:
    """Evaluate a 2-variable smooth test function for Hessian verification."""
    u, v = xy[0], xy[1]
    return (
        fmath.exp(u * v)
        + fmath.pow3(u) / fmath.sin(v)
        + fmath.sinpi(u) * fmath.log1p(v)
    )


def _fn_jax(xy: jax.Array) -> jax.Array:
    """Evaluate a 2-variable inverse/hyperbolic test function for Hypothesis."""
    u, v = xy[0], xy[1]
    return fmath.atanh(u) * fmath.acospi(v) + fmath.exp10(u) / fmath.cosh(v)


def test_math_intrinsic_golden_logs_and_exps_match_mesa() -> None:
    """Verify logarithms and exponentials against `test_output.INTRINSIC`."""
    # Natural, base-10, shifted, and base-2 logarithms on XS = [0.1, 0.2, 0.4].
    _assert_close(
        fmath.log(XS),
        [-2.3025850929940455, -1.6094379124341003, -0.9162907318741551],
    )
    _assert_close(
        fmath.log10(XS),
        [-1.0, -0.6989700043360187, -0.3979400086720376],
    )
    _assert_close(
        fmath.log1p(XS),
        [0.09531017980432487, 0.1823215567939546, 0.33647223662121295],
    )
    _assert_close(
        fmath.log2(XS),
        [-3.321928094887362, -2.321928094887362, -1.3219280948873622],
    )
    # Domain-safe logarithms and natural/base-10 exponentials.
    _assert_close(fmath.safe_log(XS), fmath.log(XS))
    _assert_close(fmath.safe_log10(XS), fmath.log10(XS))
    _assert_close(
        fmath.exp(XS),
        [1.1051709180756477, 1.2214027581601699, 1.4918246976412703],
    )
    _assert_close(
        fmath.exp10(XS),
        [1.2589254117941673, 1.5848931924611136, 2.51188643150958],
    )
    _assert_close(
        fmath.expm1(XS),
        [0.10517091807564771, 0.22140275816016985, 0.49182469764127035],
    )


def test_math_intrinsic_golden_powers_match_mesa() -> None:
    """Verify `pow`, `powi`, `pow2`..`pow8`, `powm1`, and `hypot`."""
    p2 = [1.0e-2, 4.0e-2, 1.6e-1]
    _assert_close(fmath.powi(XS, 2), p2)
    _assert_close(fmath.pow(XS, 2.0), p2)
    _assert_close(fmath.pow2(XS), p2)
    _assert_close(fmath.pow3(XS), [1.0e-3, 8.0e-3, 6.4e-2])
    _assert_close(fmath.pow4(XS), [1.0e-4, 1.6e-3, 2.56e-2])
    _assert_close(fmath.pow5(XS), [1.0e-5, 3.2e-4, 1.024e-2])
    _assert_close(fmath.pow6(XS), [1.0e-6, 6.4e-5, 4.096e-3])
    _assert_close(fmath.pow7(XS), [1.0e-7, 1.28e-5, 1.6384e-3])
    _assert_close(fmath.pow8(XS), [1.0e-8, 2.56e-6, 6.5536e-4])
    _assert_close(fmath.powm1(XS), [10.0, 5.0, 2.5])
    _assert_close([float(fmath.hypot(3.0, 4.0))], [5.0])


def test_math_intrinsic_golden_direct_trig_matches_mesa() -> None:
    """Verify radian and half-turn trig against `test_output.INTRINSIC`."""
    _assert_close(
        fmath.cos(XS),
        [0.9950041652780258, 0.9800665778412416, 0.9210609940028851],
    )
    _assert_close(
        fmath.sin(XS),
        [0.09983341664682815, 0.19866933079506122, 0.3894183423086505],
    )
    _assert_close(
        fmath.tan(XS),
        [0.10033467208545055, 0.20271003550867248, 0.4227932187381618],
    )
    _assert_close(
        fmath.cospi(XS),
        [0.9510565162951535, 0.8090169943749475, 0.30901699437494745],
    )
    _assert_close(
        fmath.sinpi(XS),
        [0.3090169943749474, 0.5877852522924731, 0.9510565162951535],
    )
    _assert_close(
        fmath.tanpi(XS),
        [0.3249196962329063, 0.7265425280053609, 3.077683537175253],
    )


def test_math_intrinsic_golden_inverse_and_hyperbolic_match_mesa() -> None:
    """Verify inverse trig and hyperbolic functions against `test_output`."""
    # Inverse trigonometric functions in radians and half-turns.
    _assert_close(
        fmath.acos(XS),
        [1.4706289056333368, 1.369438406004566, 1.1592794807274085],
    )
    _assert_close(
        fmath.asin(XS),
        [0.1001674211615598, 0.2013579207903308, 0.411516846067488],
    )
    _assert_close(
        fmath.atan(XS),
        [0.09966865249116204, 0.19739555984988078, 0.3805063771123649],
    )
    _assert_close(
        fmath.acospi(XS),
        [0.4681157195707401, 0.4359057831510251, 0.3690101195655454],
    )
    _assert_close(
        fmath.asinpi(XS),
        [0.03188428042925993, 0.06409421684897494, 0.1309898804344546],
    )
    _assert_close(
        fmath.atanpi(XS),
        [0.03172551743055357, 0.0628329581890012, 0.1211189415908434],
    )
    # Direct hyperbolic functions cosh, sinh, tanh.
    _assert_close(
        fmath.cosh(XS),
        [1.0050041680558035, 1.020066755619076, 1.081072371838455],
    )
    _assert_close(
        fmath.sinh(XS),
        [0.10016675001984403, 0.20133600254109402, 0.4107523258028155],
    )
    _assert_close(
        fmath.tanh(XS),
        [0.09966799462495582, 0.197375320224904, 0.3799489622552249],
    )


def test_math_safe_guards_and_jax_check_grads_avoid_nan_contamination() -> None:
    """Verify double-`where` guards under `jax.grad` and `check_grads`."""
    assert float(fmath.safe_sqrt(-4.0)) == 0.0
    assert np.isnan(float(fmath.safe_sqrt(np.nan)))
    assert float(fmath.safe_log(np.nan)) == fmath.SAFE_LOG_BAD_RETURN
    assert float(fmath.safe_log10(0.0)) == fmath.SAFE_LOG_BAD_RETURN

    grad_sqrt = jax.grad(lambda z: jnp.sum(fmath.safe_sqrt(z)))
    g_vals = grad_sqrt(jnp.asarray([-2.0, 0.0, 4.0], dtype=jnp.float64))
    _assert_close(g_vals, [0.0, 0.0, 0.25])

    grad_log = jax.grad(lambda z: jnp.sum(fmath.safe_log(z)))
    g_log = grad_log(jnp.asarray([-5.0, np.nan, 2.0], dtype=jnp.float64))
    _assert_close(g_log, [0.0, 0.0, 0.5])

    test_util.check_grads(
        _smooth_combo, (jnp.asarray(0.35, dtype=jnp.float64),), 2
    )
    assert fmath.str_to_double("1.5D-2") == 0.015
    _assert_close(fmath.str_to_vector("1.0D0 2.0d1 -3.5"), [1.0, 20.0, -3.5])


def test_auto_diff_order1_binary_and_unary_match_mesa_test_suite() -> None:
    """Verify `DualOrder1` operations against `test_auto_diff.f90`."""
    x = ad.AutoDiffReal2VarOrder1(1.0, [1.0, 0.0])
    y = ad.AutoDiffReal2VarOrder1(2.0, [1.0, 0.0])
    z = x + y
    assert isinstance(z, ad.AutoDiffReal2VarOrder1)
    _assert_close([float(z.val), *z.d1.tolist()], [3.0, 2.0, 0.0], TOL_AD)
    x = ad.AutoDiffReal2VarOrder1(3.0, [1.0, 0.0])
    y = ad.AutoDiffReal2VarOrder1(2.0, [0.0, 1.0])
    z_mul = x * y
    _assert_close(
        [float(z_mul.val), *z_mul.d1.tolist()], [6.0, 2.0, 3.0], TOL_AD
    )
    z_div = x / y
    _assert_close(
        [float(z_div.val), *z_div.d1.tolist()], [1.5, 0.5, -0.75], TOL_AD
    )
    x_pow = ad.AutoDiffReal2VarOrder1(2.0, [1.0, 0.0])
    y_pow = ad.AutoDiffReal2VarOrder1(2.0, [0.0, 1.0])
    z_pow = x_pow.pow(y_pow)
    _assert_close(
        [float(z_pow.val), *z_pow.d1.tolist()],
        [4.0, 4.0, 4.0 * float(fmath.log(2.0))],
        TOL_AD,
    )
    x_dim = ad.AutoDiffReal2VarOrder1(2.0, [2.0, 0.0])
    y_dim = ad.AutoDiffReal2VarOrder1(1.0, [0.0, 1.0])
    z_dim = x_dim.dim(y_dim)
    _assert_close(
        [float(z_dim.val), *z_dim.d1.tolist()], [1.0, 2.0, -1.0], TOL_AD
    )
    x_pi = ad.AutoDiffReal2VarOrder1(constants.PI, [1.0, 0.0])
    z_tan = x_pi.tan()
    _assert_close(
        [float(z_tan.val), *z_tan.d1.tolist()],
        [
            float(fmath.tan(constants.PI)),
            float(1.0 / fmath.pow2(fmath.cos(constants.PI))),
            0.0,
        ],
        TOL_AD,
    )


def test_auto_diff_order2_matches_jax_hessian_and_differentiate() -> None:
    """Verify `AutoDiffReal2VarOrder2` against `jax.grad` and `jax.hessian`."""
    xv, yv = 1.2, 0.7
    xh = ad.AutoDiffReal2VarOrder2.var1(xv)
    yh = ad.AutoDiffReal2VarOrder2.var2(yv)
    zh = (xh * yh).exp() + xh.pow3() / yh.sin() + xh.sinpi() * yh.log1p()

    pt = jnp.asarray([xv, yv], dtype=jnp.float64)
    val_jax = _scalar_fn(pt)
    grad_jax = jax.grad(_scalar_fn)(pt)
    hess_jax = jax.hessian(_scalar_fn)(pt)

    _assert_close(
        [zh.val, zh.d1val1, zh.d1val2, zh.d2val1, zh.d1val1_d1val2, zh.d2val2],
        [
            val_jax,
            grad_jax[0],
            grad_jax[1],
            hess_jax[0, 0],
            hess_jax[0, 1],
            hess_jax[1, 1],
        ],
        TOL_AD,
    )
    d1 = zh.differentiate_1()
    _assert_close(
        [d1.val, d1.d1val1, d1.d1val2],
        [grad_jax[0], hess_jax[0, 0], hess_jax[0, 1]],
        TOL_AD,
    )


@hypothesis.settings(max_examples=25, deadline=None)
@hypothesis.given(
    xv=st.floats(min_value=0.2, max_value=0.8),
    yv=st.floats(min_value=0.2, max_value=0.8),
)
def test_property_hyperdual_matches_jax_jacfwd_and_hessian(
    xv: float, yv: float
) -> None:
    """Property check that `AutoDiffReal2VarOrder2` matches JAX 2nd partials."""
    xh = ad.AutoDiffReal2VarOrder2.var1(xv)
    yh = ad.AutoDiffReal2VarOrder2.var2(yv)
    zh = xh.atanh() * yh.acospi() + xh.exp10() / yh.cosh()

    pt = jnp.asarray([xv, yv], dtype=jnp.float64)
    g = jax.jacfwd(_fn_jax)(pt)
    h = jax.hessian(_fn_jax)(pt)
    _assert_close(
        [zh.val, zh.d1val1, zh.d1val2, zh.d2val1, zh.d1val1_d1val2, zh.d2val2],
        [_fn_jax(pt), g[0], g[1], h[0, 0], h[0, 1], h[1, 1]],
        1.0e-12,
    )
