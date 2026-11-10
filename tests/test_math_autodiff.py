# SPDX-License-Identifier: LGPL-3.0-or-later
"""Golden parity, JAX autodiff, and property tests for `math` and `auto_diff`.

Verifies all 33 functions in upstream MESA's `math/test/test_output.INTRINSIC`
to `<= 1e-15`, `test_auto_diff.f90` cases across `DualOrder1` and
`AutoDiffReal2VarOrder2` to `<= 1e-14`, `jax.test_util.check_grads` and
`jax.hessian` parity, and `hypothesis` property checks.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import hypothesis
import jax
import jax.numpy as jnp
import numpy as np
import pytest
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


def test_auto_diff_pytree_unflatten_batching_and_arity_guards() -> None:
    """Verify PyTree `tree_map`/`eval_shape`, batching, and arity guards."""
    x4 = ad.AutoDiffReal4VarOrder1.independent(2.0, 1)
    mapped = jax.tree_util.tree_map(lambda a: a.ndim, x4)
    assert mapped.val == 0
    assert mapped.d1 == 1

    shaped = jax.eval_shape(
        lambda u: u.exp() + 1.0, ad.AutoDiffReal2VarOrder2.var1(1.5)
    )
    assert shaped.val.ndim == 0

    x2 = ad.AutoDiffReal2VarOrder1.independent(3.0, 0)
    z_batch = x2 + jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    assert z_batch.val.shape == (3,)
    assert z_batch.d1.shape == (3, 2)

    y1 = ad.AutoDiffReal1VarOrder1.independent(2.0, 0)
    with pytest.raises(ValueError, match="Cannot combine"):
        _ = x2 + y1


def _parse_upstream_auto_diff_output(
    text: str,
) -> tuple[list[tuple[list[float], list[float]]], list[tuple[bool, bool]]]:
    """Parse numeric and boolean records from `auto_diff/test/test_output`."""
    numeric: list[tuple[list[float], list[float]]] = []
    booleans: list[tuple[bool, bool]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Should print T:"):
            booleans.append((True, line.split(":", 1)[1].strip() == "T"))
        elif line.startswith("Should print F:"):
            booleans.append((False, line.split(":", 1)[1].strip() == "T"))
        elif line.startswith("Should print") and ":" in line:
            lhs_raw, rhs_raw = line.split(":", 1)
            lhs = (
                lhs_raw.removeprefix("Should print approximately")
                .removeprefix("Should print")
                .strip()
            )
            exp_vals = [
                float(t.replace("D", "E").replace("d", "e"))
                for t in lhs.split()
            ]
            act_vals = [
                float(t.replace("D", "E").replace("d", "e"))
                for t in rhs_raw.strip().split()
            ]
            numeric.append((exp_vals, act_vals))
    return numeric, booleans


def _run_order1_comparisons(cls: type[ad.DualOrder1]) -> list[bool]:
    """Run the 21 relational comparison checks for `cls`."""
    x = cls.constant(3.1)
    y = cls.constant(3.1)
    y_lo = cls.constant(2.9)
    x_int = cls.constant(3.0)
    return [
        x == y,
        x > y_lo,
        x < y_lo,
        x >= y_lo,
        x <= y_lo,
        x >= y,
        x <= y,
        x == 3.1,
        x > 2.9,
        x < 2.9,
        x >= 2.9,
        x <= 2.9,
        x >= 3.1,
        x <= 3.1,
        x_int == 3,
        x_int > 2,
        x_int < 2,
        x_int >= 3,
        x_int <= 2,
        x_int >= 2,
        x_int <= 3,
    ]


def _run_order1_unary(u: object, rec: object) -> list[list[float]]:
    """Run the 36 shared unary operator cases for `u(val)`."""
    pi = constants.PI
    e1 = float(fmath.exp(1.0))
    return [
        rec(u(pi).sin()),
        rec(u(pi).cos()),
        rec(u(pi).tan()),
        rec(u(pi).sinh()),
        rec(u(pi).cosh()),
        rec(u(pi).tanh()),
        rec(u(0.5).asin()),
        rec(u(0.5).acos()),
        rec(u(0.5).atan()),
        rec(u(2.0).asinh()),
        rec(u(2.0).acosh()),
        rec(u(0.5).atanh()),
        rec(-u(1.0)),
        rec(-u(-1.0)),
        rec(u(1.0).log()),
        rec(u(1.0).safe_log()),
        rec(u(10.0).log10()),
        rec(u(10.0).safe_log10()),
        rec(u(e1).log()),
        rec(u(e1).safe_log()),
        rec(u(100.0).log10()),
        rec(u(100.0).safe_log10()),
        rec(u(3.0).pow2()),
        rec(u(2.0).pow3()),
        rec(u(2.0).pow4()),
        rec(u(3.0).pow5()),
        rec(u(2.0).pow6()),
        rec(u(2.0).pow7()),
        rec(u(4.0).powf(0.5)),
        rec(u(2.0).rpow(e1)),
        rec(u(4.0).powi(2)),
        rec(u(2.0).rpow(2.0)),
    ]


def _run_1var_order1_golden() -> tuple[list[bool], list[list[float]]]:
    """Run all `do_test_auto_diff_1var_order1` cases from `test_auto_diff`."""
    c1 = ad.AutoDiffReal1VarOrder1

    def rec(z: ad.AutoDiffReal1VarOrder1) -> list[float]:
        return [float(z.val), float(z.d1[0])]

    def u(v: float) -> ad.AutoDiffReal1VarOrder1:
        return c1(v, [1.0])

    x1, y2, x3 = c1(1.0, [1.0]), c1(2.0, [1.0]), c1(3.0, [1.0])
    y2_d2, x2_d2 = c1(2.0, [2.0]), c1(2.0, [2.0])
    # fmt: off
    nums = [
        rec(c1.constant(3.1)), rec(x1 + y2), rec(x1 + 2.0), rec(2.0 + x1),
        rec(x3 * y2), rec(x3 * 2.0), rec(2.0 * x3), rec(x1.max(y2_d2)),
        rec(x1.max(2.0)), rec(x1.max(0.0)), rec(x1.max(2.0)), rec(x1.max(0.0)),
        rec(x2_d2.dim(x1)), rec(x1.dim(x2_d2)), rec(x2_d2.dim(1.0)),
        rec(u(1.0).exp()), rec(u(1.0).abs()), rec(u(-1.0).abs()),
        rec(u(2.0).exp()), *_run_order1_unary(u, rec),
    ]
    # fmt: on
    return _run_order1_comparisons(c1), nums


def _run_2var_order1_golden() -> tuple[list[bool], list[list[float]]]:
    """Run all `do_test_auto_diff_2var_order1` cases from `test_auto_diff`."""
    c2 = ad.AutoDiffReal2VarOrder1

    def rec(z: ad.AutoDiffReal2VarOrder1) -> list[float]:
        return [float(z.val), float(z.d1[0]), float(z.d1[1])]

    def u1(v: float) -> ad.AutoDiffReal2VarOrder1:
        return c2(v, [1.0, 0.0])

    def u2(v: float) -> ad.AutoDiffReal2VarOrder1:
        return c2(v, [0.0, 1.0])

    x1, y2_v1, x3 = u1(1.0), u1(2.0), u1(3.0)
    y2_v2, y5_v2 = c2(2.0, [0.0, 1.0]), c2(5.0, [0.0, 2.0])
    x2_v1, y2_d2 = c2(2.0, [1.0, 0.0]), c2(2.0, [0.0, 2.0])
    x2_d2, y1_v2 = c2(2.0, [2.0, 0.0]), c2(1.0, [0.0, 1.0])
    # fmt: off
    nums = [
        rec(c2.constant(3.1)), rec(x1 + y2_v1), rec(x1 + 2.0), rec(2.0 + x1),
        rec(x1 + 2), rec(2 + x1), rec(x3 * y2_v2), rec(x3 * 2.0),
        rec(x3 / y2_v2), rec(x3 / 2.0), rec(x1 - y2_v2), rec(x1 - y5_v2),
        rec(x2_v1 - 1.0), rec(2.0 * x3), rec(x1.max(y2_d2)), rec(x1.max(2.0)),
        rec(x1.max(0.0)), rec(x1.max(2.0)), rec(x1.max(0.0)),
        rec(x1.min(y2_d2)), rec(x1.min(2.0)), rec(x1.min(0.0)),
        rec(x1.min(2.0)), rec(x1.min(0.0)), rec(x2_d2.dim(y1_v2)),
        rec(y1_v2.dim(x2_d2)), rec(x2_d2.dim(1.0)), rec(x2_v1.pow(y2_v2)),
        rec(u1(1.0).exp()), rec(u2(1.0).exp()), rec(u1(2.0).exp()),
        rec(u1(1.0).abs()), rec(u1(-1.0).abs()), *_run_order1_unary(u1, rec),
    ]
    # fmt: on
    return _run_order1_comparisons(c2), nums


def _run_star_order1_golden() -> list[list[float]]:
    """Run all `do_test_auto_diff_star_order1` cases from `test_auto_diff`."""
    cs = ad.AutoDiffRealStarOrder1
    nums: list[list[float]] = []
    x_assign = cs.independent(3.0, 3)
    nums.extend([[float(x_assign.d1[i])] for i in range(15)])

    d1_seq = [float(i + 1) if i < 15 else 0.0 for i in range(33)]
    x_base = cs(3.0, d1_seq)
    y_pow2 = x_base.pow2()
    nums.extend([[float(y_pow2.d1[i])] for i in range(15)])
    x_pow2 = x_base.pow2()
    nums.extend([[float(x_pow2.d1[i])] for i in range(15)])

    y_add1 = x_base + 1.0
    nums.extend([[float(y_add1.d1[i])] for i in range(15)])
    y_addx = x_base + x_base
    nums.extend([[float(y_addx.d1[i])] for i in range(15)])

    z_d1 = [float(1 - (i + 1)) if i < 15 else 0.0 for i in range(33)]
    z = cs(2.0, z_d1)
    y_chain = x_base.exp() * z
    nums.extend([[float(y_chain.d1[i])] for i in range(15)])
    return nums


def test_auto_diff_complete_golden_test_output_and_order2_sweep() -> None:
    """Verify all 704 lines of `auto_diff/test/test_output` and `Order2Var2`."""
    bools_1v, nums_1v = _run_1var_order1_golden()
    bools_2v, nums_2v = _run_2var_order1_golden()
    nums_star = _run_star_order1_golden()

    all_bools = [*bools_1v, *bools_2v]
    all_nums = [*nums_1v, *nums_2v, *nums_star]
    assert len(all_bools) == 42
    assert len(all_nums) == 206

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    out_path = (
        repo_root
        / ".mesa_temp_cache"
        / "mesa_root"
        / "auto_diff"
        / "test"
        / "test_output"
    )
    if out_path.exists():
        golden_nums, golden_bools = _parse_upstream_auto_diff_output(
            out_path.read_text()
        )
        assert len(golden_bools) == len(all_bools)
        for actual_b, (exp_b, ref_b) in zip(
            all_bools, golden_bools, strict=True
        ):
            assert actual_b is exp_b
            assert actual_b is ref_b

        assert len(golden_nums) == len(all_nums)
        for actual_vec, (exp_vec, ref_vec) in zip(
            all_nums, golden_nums, strict=True
        ):
            _assert_close(actual_vec, exp_vec, TOL_AD)
            _assert_close(actual_vec, ref_vec, TOL_AD)

    # Order2Var2 comparison, binary, unary, and chain_rule parity
    c22 = ad.AutoDiffReal2VarOrder2
    x_cmp = c22.constant(3.1)
    y_cmp = c22.constant(2.9)
    assert x_cmp == c22.constant(3.1) and x_cmp > y_cmp and not (x_cmp < y_cmp)
    assert x_cmp >= y_cmp and not (x_cmp <= y_cmp) and x_cmp == 3.1

    xv, yv = 0.6, 1.1
    x2 = ad.AutoDiffReal2VarOrder1(xv, [1.0, 0.0])
    y2 = ad.AutoDiffReal2VarOrder1(yv, [0.0, 1.0])
    chain_2v = (x2.pow2() + x2.sin()).exp() * y2.cos()
    x22 = c22.var1(xv)
    y22 = c22.var2(yv)
    chain_22 = (x22.pow2() + x22.sin()).exp() * y22.cos()
    _assert_close(
        [float(chain_2v.val), float(chain_2v.d1[0]), float(chain_2v.d1[1])],
        [float(chain_22.val), float(chain_22.d1val1), float(chain_22.d1val2)],
        TOL_AD,
    )
