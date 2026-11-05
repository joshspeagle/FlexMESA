# SPDX-License-Identifier: LGPL-3.0-or-later
"""First-order forward-mode dual numbers (`DualOrder1`) as JAX PyTrees.

Provides `DualOrder1` and MESA-compatible factory helpers for 1, 2, 4, and 33
independent variables (`AutoDiffReal1VarOrder1`, `AutoDiffReal2VarOrder1`,
`AutoDiffReal4VarOrder1`, `AutoDiffRealStarOrder1`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import math as fmath
from flexmesa.auto_diff import base, kernels
from flexmesa.math import intrinsics

STAR_ORDER1_VARS: int = 33


@jax.tree_util.register_pytree_node_class
class DualOrder1(base.AutoDiffBase):
    """First-order forward-mode dual number with `N` partial derivatives.

    Attributes:
        val: Primal value `float64` JAX array of shape `(*batch,)`.
        d1: First-order partial derivatives `float64` JAX array of shape
            `(*batch, N)`.

    Examples:
        >>> from flexmesa.auto_diff import AutoDiffReal2VarOrder1
        >>> x = AutoDiffReal2VarOrder1.independent(3.0, 0)
        >>> y = AutoDiffReal2VarOrder1.independent(2.0, 1)
        >>> z = x * y + x.pow2()
        >>> float(z.val), [float(v) for v in z.d1]
        (15.0, [8.0, 3.0])
    """

    __slots__ = ("d1", "val")

    def __init__(
        self, val: intrinsics.ArrayLike, d1: intrinsics.ArrayLike
    ) -> None:
        """Initialize `DualOrder1` with `float64` `val` and `d1` arrays."""
        self.val: jax.Array = jnp.asarray(val, dtype=jnp.float64)
        self.d1: jax.Array = jnp.asarray(d1, dtype=jnp.float64)

    def tree_flatten(
        self,
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        """Flatten `DualOrder1` into `(val, d1)` leaves for JAX PyTrees."""
        return (self.val, self.d1), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[jax.Array, jax.Array],
    ) -> DualOrder1:
        """Reconstruct `DualOrder1` from flattened `(val, d1)` leaves."""
        del aux_data
        return cls(children[0], children[1])

    @classmethod
    def constant(
        cls, val: intrinsics.ArrayLike, num_vars: int = 1
    ) -> DualOrder1:
        """Construct a constant `DualOrder1` with `num_vars` zero partials."""
        v = jnp.asarray(val, dtype=jnp.float64)
        d1 = jnp.zeros((*v.shape, num_vars), dtype=jnp.float64)
        return cls(v, d1)

    @classmethod
    def independent(
        cls, val: intrinsics.ArrayLike, index: int, num_vars: int = 1
    ) -> DualOrder1:
        """Seed the `index`-th independent variable ($\\partial_i x_i = 1$)."""
        if index < 0 or index >= num_vars:
            msg = f"index {index} out of bounds for num_vars={num_vars}"
            raise IndexError(msg)
        v = jnp.asarray(val, dtype=jnp.float64)
        basis = jnp.zeros((num_vars,), dtype=jnp.float64).at[index].set(1.0)
        d1 = jnp.broadcast_to(basis, (*v.shape, num_vars))
        return cls(v, d1)

    def make_unary_operator(
        self, z_val: intrinsics.ArrayLike, z_d1x: intrinsics.ArrayLike
    ) -> DualOrder1:
        """Apply the first-order unary chain rule $z = f(x)$."""
        zv = jnp.asarray(z_val, dtype=jnp.float64)
        zd1 = jnp.asarray(z_d1x, dtype=jnp.float64)
        return DualOrder1(zv, self.d1 * zd1[..., None])

    def make_binary_operator(
        self,
        other: DualOrder1,
        z_val: intrinsics.ArrayLike,
        z_d1x: intrinsics.ArrayLike,
        z_d1y: intrinsics.ArrayLike,
    ) -> DualOrder1:
        """Apply the first-order binary chain rule $z = g(x, y)$."""
        zv = jnp.asarray(z_val, dtype=jnp.float64)
        zd1x = jnp.asarray(z_d1x, dtype=jnp.float64)
        zd1y = jnp.asarray(z_d1y, dtype=jnp.float64)
        return DualOrder1(
            zv, self.d1 * zd1x[..., None] + other.d1 * zd1y[..., None]
        )

    def apply_unary(self, kernel: kernels.UnaryKernel) -> DualOrder1:
        """Propagate a unary kernel `(f, f', f'')` through `self`."""
        zv, zd1, _ = kernel(self.val)
        return self.make_unary_operator(zv, zd1)

    def pow(self, other: DualOrder1 | intrinsics.ArrayLike) -> DualOrder1:
        """Raise `self` to a dual or scalar exponent `other` ($x^{y}$)."""
        if isinstance(other, DualOrder1):
            zv, d1x, d1y, _, _, _ = kernels.binary_pow(self.val, other.val)
            return self.make_binary_operator(other, zv, d1x, d1y)
        zv, d1x, _ = kernels.unary_pow_scalar(self.val, other)
        return self.make_unary_operator(zv, d1x)

    def max(self, other: DualOrder1 | intrinsics.ArrayLike) -> DualOrder1:
        """Compute $\\max(x, y)$ using MESA's Heaviside subgradient."""
        if isinstance(other, DualOrder1):
            q0 = fmath.heaviside(self.val - other.val)
            q1 = fmath.heaviside(other.val - self.val)
            return self.make_binary_operator(
                other, jnp.maximum(self.val, other.val), q0, q1
            )
        c = jnp.asarray(other, dtype=jnp.float64)
        return self.make_unary_operator(
            jnp.maximum(self.val, c), fmath.heaviside(self.val - c)
        )

    def min(self, other: DualOrder1 | intrinsics.ArrayLike) -> DualOrder1:
        """Compute $\\min(x, y)$ using MESA's Heaviside subgradient."""
        if isinstance(other, DualOrder1):
            q0 = fmath.heaviside(other.val - self.val)
            q1 = fmath.heaviside(self.val - other.val)
            return self.make_binary_operator(
                other, jnp.minimum(self.val, other.val), q0, q1
            )
        c = jnp.asarray(other, dtype=jnp.float64)
        return self.make_unary_operator(
            jnp.minimum(self.val, c), fmath.heaviside(c - self.val)
        )

    def dim(self, other: DualOrder1 | intrinsics.ArrayLike) -> DualOrder1:
        """Compute $\\operatorname{dim}(x, y) = \\max(x - y, 0)$."""
        if isinstance(other, DualOrder1):
            diff = self.val - other.val
            w = 0.5 + 0.5 * fmath.sgn(diff)
            val = 0.5 * diff + 0.5 * jnp.abs(diff)
            return self.make_binary_operator(other, val, w, -w)
        c = jnp.asarray(other, dtype=jnp.float64)
        diff = self.val - c
        w = 0.5 + 0.5 * fmath.sgn(diff)
        val = 0.5 * diff + 0.5 * jnp.abs(diff)
        return self.make_unary_operator(val, w)

    def __neg__(self) -> DualOrder1:
        return DualOrder1(-self.val, -self.d1)

    def __add__(self, other: DualOrder1 | intrinsics.ArrayLike) -> DualOrder1:
        if isinstance(other, DualOrder1):
            return DualOrder1(self.val + other.val, self.d1 + other.d1)
        return DualOrder1(
            self.val + jnp.asarray(other, dtype=jnp.float64), self.d1
        )

    def __radd__(self, other: intrinsics.ArrayLike) -> DualOrder1:
        return self.__add__(other)

    def __sub__(self, other: DualOrder1 | intrinsics.ArrayLike) -> DualOrder1:
        if isinstance(other, DualOrder1):
            return DualOrder1(self.val - other.val, self.d1 - other.d1)
        return DualOrder1(
            self.val - jnp.asarray(other, dtype=jnp.float64), self.d1
        )

    def __rsub__(self, other: intrinsics.ArrayLike) -> DualOrder1:
        return DualOrder1(
            jnp.asarray(other, dtype=jnp.float64) - self.val, -self.d1
        )

    def __mul__(self, other: DualOrder1 | intrinsics.ArrayLike) -> DualOrder1:
        if isinstance(other, DualOrder1):
            return self.make_binary_operator(
                other, self.val * other.val, other.val, self.val
            )
        c = jnp.asarray(other, dtype=jnp.float64)
        return self.make_unary_operator(self.val * c, c)

    def __rmul__(self, other: intrinsics.ArrayLike) -> DualOrder1:
        return self.__mul__(other)

    def __truediv__(
        self, other: DualOrder1 | intrinsics.ArrayLike
    ) -> DualOrder1:
        if isinstance(other, DualOrder1):
            inv_y = 1.0 / other.val
            return self.make_binary_operator(
                other,
                self.val * inv_y,
                inv_y,
                -self.val * inv_y * inv_y,
            )
        c = jnp.asarray(other, dtype=jnp.float64)
        return self.make_unary_operator(self.val / c, 1.0 / c)

    def __rtruediv__(self, other: intrinsics.ArrayLike) -> DualOrder1:
        c = jnp.asarray(other, dtype=jnp.float64)
        return self.powm1() * c

    def __pow__(self, other: DualOrder1 | intrinsics.ArrayLike) -> DualOrder1:
        return self.pow(other)


class _FixedVarOrder1Factory:
    """Factory bound to a fixed number of independent variables `num_vars`."""

    def __init__(self, num_vars: int) -> None:
        self.num_vars = num_vars

    def constant(self, val: intrinsics.ArrayLike) -> DualOrder1:
        """Construct a constant dual number with `self.num_vars` partials."""
        return DualOrder1.constant(val, num_vars=self.num_vars)

    def independent(self, val: intrinsics.ArrayLike, index: int) -> DualOrder1:
        """Construct the `index`-th independent variable for `self.num_vars`."""
        return DualOrder1.independent(val, index, num_vars=self.num_vars)

    def __call__(
        self, val: intrinsics.ArrayLike, d1: intrinsics.ArrayLike
    ) -> DualOrder1:
        """Construct a `DualOrder1` from explicit `val` and `d1`."""
        return DualOrder1(val, d1)


AutoDiffReal1VarOrder1 = _FixedVarOrder1Factory(1)
AutoDiffReal2VarOrder1 = _FixedVarOrder1Factory(2)
AutoDiffReal4VarOrder1 = _FixedVarOrder1Factory(4)
AutoDiffReal4Coeff = AutoDiffReal4VarOrder1
AutoDiffRealStarOrder1 = _FixedVarOrder1Factory(STAR_ORDER1_VARS)
