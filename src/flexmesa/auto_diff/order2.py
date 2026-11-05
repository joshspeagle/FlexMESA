# SPDX-License-Identifier: LGPL-3.0-or-later
"""Two-variable second-order hyperdual numbers (`AutoDiffReal2VarOrder2`).

Ports MESA's `auto_diff_real_2var_order2` (`auto_diff_real_2nd_order`) as a
registered JAX PyTree propagating exact first and second partial derivatives
`(val, d1val1, d1val2, d2val1, d1val1_d1val2, d2val2)` in `float64`.
"""

from __future__ import annotations

from typing import Self

import jax
import jax.numpy as jnp

from flexmesa import math as fmath
from flexmesa.auto_diff import base, kernels
from flexmesa.math import intrinsics


def _as_bcast_f64(
    arr: intrinsics.ArrayLike, shape: tuple[int, ...]
) -> jax.Array:
    """Convert `arr` to `float64` and broadcast to `shape` if needed."""
    a = intrinsics.as_f64(arr)
    return a if a.shape == shape else jnp.broadcast_to(a, shape)


@jax.tree_util.register_pytree_node_class
class AutoDiffReal2VarOrder2(base.AutoDiffBase):
    """Two-variable second-order forward-mode hyperdual number.

    Attributes:
        val: Primal value $f(x_1, x_2)$ of shape `(*batch,)`.
        d1val1: First partial derivative $\\partial f / \\partial x_1$.
        d1val2: First partial derivative $\\partial f / \\partial x_2$.
        d2val1: Second partial derivative $\\partial^2 f / \\partial x_1^2$.
        d1val1_d1val2: Mixed partial $\\partial_{12} f$.
        d2val2: Second partial derivative $\\partial^2 f / \\partial x_2^2$.

    Examples:
        >>> from flexmesa.auto_diff import AutoDiffReal2VarOrder2
        >>> x = AutoDiffReal2VarOrder2.var1(2.0)
        >>> y = AutoDiffReal2VarOrder2.var2(3.0)
        >>> z = x * y + x.pow2()
        >>> (float(z.val), float(z.d1val1), float(z.d2val1))
        (10.0, 7.0, 2.0)
    """

    __slots__ = (
        "d1val1",
        "d1val1_d1val2",
        "d1val2",
        "d2val1",
        "d2val2",
        "val",
    )

    def __init__(
        self,
        val: intrinsics.ArrayLike,
        d1val1: intrinsics.ArrayLike = 0.0,
        d1val2: intrinsics.ArrayLike = 0.0,
        d2val1: intrinsics.ArrayLike = 0.0,
        d1val1_d1val2: intrinsics.ArrayLike = 0.0,
        d2val2: intrinsics.ArrayLike = 0.0,
    ) -> None:
        """Initialize all six `float64` hyperdual components."""
        v = intrinsics.as_f64(val)
        sh = v.shape
        self.val: jax.Array = v
        self.d1val1: jax.Array = _as_bcast_f64(d1val1, sh)
        self.d1val2: jax.Array = _as_bcast_f64(d1val2, sh)
        self.d2val1: jax.Array = _as_bcast_f64(d2val1, sh)
        self.d1val1_d1val2: jax.Array = _as_bcast_f64(d1val1_d1val2, sh)
        self.d2val2: jax.Array = _as_bcast_f64(d2val2, sh)

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        None,
    ]:
        """Flatten hyperdual components for JAX PyTree transformations."""
        return (
            self.val,
            self.d1val1,
            self.d1val2,
            self.d2val1,
            self.d1val1_d1val2,
            self.d2val2,
        ), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
    ) -> Self:
        """Reconstruct `AutoDiffReal2VarOrder2` from flattened PyTree leaves."""
        del aux_data
        obj = object.__new__(cls)
        (
            obj.val,
            obj.d1val1,
            obj.d1val2,
            obj.d2val1,
            obj.d1val1_d1val2,
            obj.d2val2,
        ) = children
        return obj

    @classmethod
    def constant(cls, val: intrinsics.ArrayLike) -> Self:
        """Construct a constant hyperdual with zero derivatives."""
        return cls(val)

    @classmethod
    def var1(cls, val: intrinsics.ArrayLike) -> Self:
        """Seed the first independent variable $x_1$ (`d1val1 = 1.0`)."""
        return cls(val, d1val1=1.0)

    @classmethod
    def var2(cls, val: intrinsics.ArrayLike) -> Self:
        """Seed the second independent variable $x_2$ (`d1val2 = 1.0`)."""
        return cls(val, d1val2=1.0)

    def make_unary_operator(
        self,
        z_val: intrinsics.ArrayLike,
        z_d1x: intrinsics.ArrayLike,
        z_d2x: intrinsics.ArrayLike,
    ) -> Self:
        """Apply MESA's second-order unary chain rule."""
        zv = intrinsics.as_f64(z_val)
        zd1 = intrinsics.as_f64(z_d1x)
        zd2 = intrinsics.as_f64(z_d2x)
        return type(self)(
            zv,
            self.d1val1 * zd1,
            self.d1val2 * zd1,
            self.d2val1 * zd1 + zd2 * fmath.pow2(self.d1val1),
            self.d1val1 * self.d1val2 * zd2 + self.d1val1_d1val2 * zd1,
            self.d2val2 * zd1 + zd2 * fmath.pow2(self.d1val2),
        )

    def make_binary_operator(
        self,
        other: AutoDiffReal2VarOrder2,
        z_val: intrinsics.ArrayLike,
        z_d1x: intrinsics.ArrayLike,
        z_d1y: intrinsics.ArrayLike,
        z_d2x: intrinsics.ArrayLike,
        z_d1x_d1y: intrinsics.ArrayLike,
        z_d2y: intrinsics.ArrayLike,
    ) -> Self:
        """Apply MESA's second-order binary chain rule."""
        d1x = intrinsics.as_f64(z_d1x)
        d1y = intrinsics.as_f64(z_d1y)
        d2x = intrinsics.as_f64(z_d2x)
        d1xy = intrinsics.as_f64(z_d1x_d1y)
        d2y = intrinsics.as_f64(z_d2y)
        q0 = self.d1val1 * d1xy
        q1 = self.d1val2 * d1xy
        return type(self)(
            z_val,
            self.d1val1 * d1x + other.d1val1 * d1y,
            self.d1val2 * d1x + other.d1val2 * d1y,
            (
                2.0 * q0 * other.d1val1
                + self.d2val1 * d1x
                + other.d2val1 * d1y
                + d2x * fmath.pow2(self.d1val1)
                + d2y * fmath.pow2(other.d1val1)
            ),
            (
                q0 * other.d1val2
                + q1 * other.d1val1
                + self.d1val1 * self.d1val2 * d2x
                + self.d1val1_d1val2 * d1x
                + other.d1val1 * other.d1val2 * d2y
                + other.d1val1_d1val2 * d1y
            ),
            (
                2.0 * q1 * other.d1val2
                + self.d2val2 * d1x
                + other.d2val2 * d1y
                + d2x * fmath.pow2(self.d1val2)
                + d2y * fmath.pow2(other.d1val2)
            ),
        )

    def apply_unary(self, kernel: kernels.UnaryKernel) -> Self:
        """Propagate a unary kernel `(f, f', f'')` through `self`."""
        zv, zd1, zd2 = kernel(self.val)
        return self.make_unary_operator(zv, zd1, zd2)

    def differentiate_1(self) -> Self:
        """Extract $\\partial / \\partial x_1$ with zero second partials."""
        return type(self)(
            self.d1val1, self.d2val1, self.d1val1_d1val2, 0.0, 0.0, 0.0
        )

    def differentiate_2(self) -> Self:
        """Extract $\\partial / \\partial x_2$ with zero second partials."""
        return type(self)(
            self.d1val2, self.d1val1_d1val2, self.d2val2, 0.0, 0.0, 0.0
        )

    def pow(
        self, other: AutoDiffReal2VarOrder2 | intrinsics.ArrayLike
    ) -> Self:
        """Raise `self` to a hyperdual or scalar exponent `other` ($x^{y}$)."""
        if isinstance(other, AutoDiffReal2VarOrder2):
            zv, d1x, d1y, d2x, d1x_d1y, d2y = kernels.binary_pow(
                self.val, other.val
            )
            return self.make_binary_operator(
                other, zv, d1x, d1y, d2x, d1x_d1y, d2y
            )
        zv, d1x, d2x = kernels.unary_pow_scalar(self.val, other)
        return self.make_unary_operator(zv, d1x, d2x)

    def max(
        self, other: AutoDiffReal2VarOrder2 | intrinsics.ArrayLike
    ) -> Self:
        """Compute $\\max(x, y)$ using MESA's Heaviside subgradient."""
        if isinstance(other, AutoDiffReal2VarOrder2):
            q0 = fmath.heaviside(self.val - other.val)
            q1 = fmath.heaviside(other.val - self.val)
            return self.make_binary_operator(
                other, jnp.maximum(self.val, other.val), q0, q1, 0.0, 0.0, 0.0
            )
        c = intrinsics.as_f64(other)
        return self.make_unary_operator(
            jnp.maximum(self.val, c), fmath.heaviside(self.val - c), 0.0
        )

    def min(
        self, other: AutoDiffReal2VarOrder2 | intrinsics.ArrayLike
    ) -> Self:
        """Compute $\\min(x, y)$ using MESA's Heaviside subgradient."""
        if isinstance(other, AutoDiffReal2VarOrder2):
            q0 = fmath.heaviside(other.val - self.val)
            q1 = fmath.heaviside(self.val - other.val)
            return self.make_binary_operator(
                other, jnp.minimum(self.val, other.val), q0, q1, 0.0, 0.0, 0.0
            )
        c = intrinsics.as_f64(other)
        return self.make_unary_operator(
            jnp.minimum(self.val, c), fmath.heaviside(c - self.val), 0.0
        )

    def dim(
        self, other: AutoDiffReal2VarOrder2 | intrinsics.ArrayLike
    ) -> Self:
        """Compute $\\operatorname{dim}(x, y) = \\max(x - y, 0)$."""
        return (self - other).max(0.0)

    def __neg__(self) -> Self:
        """Negate all six hyperdual components directly without squaring."""
        return type(self)(
            -self.val,
            -self.d1val1,
            -self.d1val2,
            -self.d2val1,
            -self.d1val1_d1val2,
            -self.d2val2,
        )

    def __add__(
        self, other: AutoDiffReal2VarOrder2 | intrinsics.ArrayLike
    ) -> Self:
        """Add a hyperdual or scalar operand."""
        if isinstance(other, AutoDiffReal2VarOrder2):
            return type(self)(
                self.val + other.val,
                self.d1val1 + other.d1val1,
                self.d1val2 + other.d1val2,
                self.d2val1 + other.d2val1,
                self.d1val1_d1val2 + other.d1val1_d1val2,
                self.d2val2 + other.d2val2,
            )
        c = intrinsics.as_f64(other)
        return type(self)(
            self.val + c,
            self.d1val1,
            self.d1val2,
            self.d2val1,
            self.d1val1_d1val2,
            self.d2val2,
        )

    def __radd__(self, other: intrinsics.ArrayLike) -> Self:
        """Add a scalar left operand to `self`."""
        return self.__add__(other)

    def __sub__(
        self, other: AutoDiffReal2VarOrder2 | intrinsics.ArrayLike
    ) -> Self:
        """Subtract a hyperdual or scalar operand from `self`."""
        if isinstance(other, AutoDiffReal2VarOrder2):
            return type(self)(
                self.val - other.val,
                self.d1val1 - other.d1val1,
                self.d1val2 - other.d1val2,
                self.d2val1 - other.d2val1,
                self.d1val1_d1val2 - other.d1val1_d1val2,
                self.d2val2 - other.d2val2,
            )
        c = intrinsics.as_f64(other)
        return type(self)(
            self.val - c,
            self.d1val1,
            self.d1val2,
            self.d2val1,
            self.d1val1_d1val2,
            self.d2val2,
        )

    def __rsub__(self, other: intrinsics.ArrayLike) -> Self:
        """Subtract `self` from a scalar left operand."""
        return (-self) + other

    def __mul__(
        self, other: AutoDiffReal2VarOrder2 | intrinsics.ArrayLike
    ) -> Self:
        """Multiply `self` by a hyperdual or scalar operand."""
        if isinstance(other, AutoDiffReal2VarOrder2):
            return self.make_binary_operator(
                other, self.val * other.val, other.val, self.val, 0.0, 1.0, 0.0
            )
        c = intrinsics.as_f64(other)
        return type(self)(
            self.val * c,
            self.d1val1 * c,
            self.d1val2 * c,
            self.d2val1 * c,
            self.d1val1_d1val2 * c,
            self.d2val2 * c,
        )

    def __rmul__(self, other: intrinsics.ArrayLike) -> Self:
        """Multiply a scalar left operand by `self`."""
        return self.__mul__(other)

    def __truediv__(
        self, other: AutoDiffReal2VarOrder2 | intrinsics.ArrayLike
    ) -> Self:
        """Divide `self` by a hyperdual or scalar operand."""
        if isinstance(other, AutoDiffReal2VarOrder2):
            inv_y = 1.0 / other.val
            inv_y2 = inv_y * inv_y
            return self.make_binary_operator(
                other,
                self.val * inv_y,
                inv_y,
                -self.val * inv_y2,
                0.0,
                -inv_y2,
                2.0 * self.val * inv_y2 * inv_y,
            )
        return self.__mul__(1.0 / intrinsics.as_f64(other))

    def __rtruediv__(self, other: intrinsics.ArrayLike) -> Self:
        """Divide a scalar left operand by `self` ($c / y = c y^{-1}$)."""
        return self.powm1().__mul__(intrinsics.as_f64(other))

    def __pow__(
        self, other: AutoDiffReal2VarOrder2 | intrinsics.ArrayLike
    ) -> Self:
        """Raise `self` to power `other` ($x^{y}$)."""
        return self.pow(other)


AutoDiffReal2ndOrder = AutoDiffReal2VarOrder2
