# SPDX-License-Identifier: LGPL-3.0-or-later
"""First-order forward-mode dual numbers (`DualOrder1`) as JAX PyTrees.

Provides `DualOrder1` and MESA-compatible fixed-arity subclasses for 1, 2, 4,
and 33 independent variables (`AutoDiffReal1VarOrder1`,
`AutoDiffReal2VarOrder1`, `AutoDiffReal4VarOrder1`,
`AutoDiffRealStarOrder1`).
"""

from __future__ import annotations

from typing import ClassVar, Self

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
    NUM_VARS: ClassVar[int] = 0

    def __init__(
        self, val: intrinsics.ArrayLike, d1: intrinsics.ArrayLike
    ) -> None:
        """Initialize `DualOrder1` with `float64` `val` and `d1` arrays."""
        self.val: jax.Array = intrinsics.as_f64(val)
        d1_arr = intrinsics.as_f64(d1)
        if d1_arr.ndim == 0:
            msg = "DualOrder1 d1 must have at least 1 dimension"
            raise ValueError(msg)
        if self.NUM_VARS > 0 and d1_arr.shape[-1:] != (self.NUM_VARS,):
            msg = (
                f"{type(self).__name__} requires trailing dimension "
                f"{self.NUM_VARS}, got shape {d1_arr.shape}"
            )
            raise ValueError(msg)
        target_shape = (*self.val.shape, d1_arr.shape[-1])
        self.d1: jax.Array = (
            d1_arr
            if d1_arr.shape == target_shape
            else jnp.broadcast_to(d1_arr, target_shape)
        )

    def _check_compatible_dual(self, other: DualOrder1) -> None:
        """Ensure two `DualOrder1` operands share variable arity `N`."""
        if self.d1.shape[-1:] != other.d1.shape[-1:]:
            msg = (
                f"Cannot combine {type(self).__name__} "
                f"(N={self.d1.shape[-1]}) with {type(other).__name__} "
                f"(N={other.d1.shape[-1]})"
            )
            raise ValueError(msg)

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
    ) -> Self:
        """Reconstruct `DualOrder1` from flattened `(val, d1)` leaves."""
        del aux_data
        obj = object.__new__(cls)
        obj.val, obj.d1 = children
        return obj

    @classmethod
    def constant(
        cls, val: intrinsics.ArrayLike, num_vars: int | None = None
    ) -> Self:
        """Construct a constant `DualOrder1` with `num_vars` zero partials."""
        n = num_vars if num_vars is not None else (cls.NUM_VARS or 1)
        v = intrinsics.as_f64(val)
        d1 = jnp.zeros((*v.shape, n), dtype=jnp.float64)
        return cls(v, d1)

    @classmethod
    def independent(
        cls,
        val: intrinsics.ArrayLike,
        index: int,
        num_vars: int | None = None,
    ) -> Self:
        """Seed the `index`-th independent variable ($\\partial_i x_i = 1$)."""
        n = num_vars if num_vars is not None else (cls.NUM_VARS or 1)
        if index < 0 or index >= n:
            msg = f"index {index} out of bounds for num_vars={n}"
            raise IndexError(msg)
        v = intrinsics.as_f64(val)
        basis = jnp.zeros((n,), dtype=jnp.float64).at[index].set(1.0)
        d1 = jnp.broadcast_to(basis, (*v.shape, n))
        return cls(v, d1)

    def make_unary_operator(
        self, z_val: intrinsics.ArrayLike, z_d1x: intrinsics.ArrayLike
    ) -> Self:
        """Apply the first-order unary chain rule $z = f(x)$."""
        zv = intrinsics.as_f64(z_val)
        zd1 = intrinsics.as_f64(z_d1x)
        return type(self)(zv, self.d1 * zd1[..., None])

    def make_binary_operator(
        self,
        other: DualOrder1,
        z_val: intrinsics.ArrayLike,
        z_d1x: intrinsics.ArrayLike,
        z_d1y: intrinsics.ArrayLike,
    ) -> Self:
        """Apply the first-order binary chain rule $z = g(x, y)$."""
        self._check_compatible_dual(other)
        zv = intrinsics.as_f64(z_val)
        zd1x = intrinsics.as_f64(z_d1x)
        zd1y = intrinsics.as_f64(z_d1y)
        return type(self)(
            zv, self.d1 * zd1x[..., None] + other.d1 * zd1y[..., None]
        )

    def apply_unary(self, kernel: kernels.UnaryKernel) -> Self:
        """Propagate a unary kernel `(f, f', f'')` through `self`."""
        zv, zd1, _ = kernel(self.val)
        return self.make_unary_operator(zv, zd1)

    def pow(self, other: DualOrder1 | intrinsics.ArrayLike) -> Self:
        """Raise `self` to a dual or scalar exponent `other` ($x^{y}$)."""
        if isinstance(other, DualOrder1):
            zv, d1x, d1y, _, _, _ = kernels.binary_pow(self.val, other.val)
            return self.make_binary_operator(other, zv, d1x, d1y)
        zv, d1x, _ = kernels.unary_pow_scalar(self.val, other)
        return self.make_unary_operator(zv, d1x)

    def max(self, other: DualOrder1 | intrinsics.ArrayLike) -> Self:
        """Compute $\\max(x, y)$ using MESA's Heaviside subgradient."""
        if isinstance(other, DualOrder1):
            q0 = fmath.heaviside(self.val - other.val)
            q1 = fmath.heaviside(other.val - self.val)
            return self.make_binary_operator(
                other, jnp.maximum(self.val, other.val), q0, q1
            )
        c = intrinsics.as_f64(other)
        return self.make_unary_operator(
            jnp.maximum(self.val, c), fmath.heaviside(self.val - c)
        )

    def min(self, other: DualOrder1 | intrinsics.ArrayLike) -> Self:
        """Compute $\\min(x, y)$ using MESA's Heaviside subgradient."""
        if isinstance(other, DualOrder1):
            q0 = fmath.heaviside(other.val - self.val)
            q1 = fmath.heaviside(self.val - other.val)
            return self.make_binary_operator(
                other, jnp.minimum(self.val, other.val), q0, q1
            )
        c = intrinsics.as_f64(other)
        return self.make_unary_operator(
            jnp.minimum(self.val, c), fmath.heaviside(c - self.val)
        )

    def dim(self, other: DualOrder1 | intrinsics.ArrayLike) -> Self:
        """Compute $\\operatorname{dim}(x, y) = \\max(x - y, 0)$."""
        return (self - other).max(0.0)

    def __neg__(self) -> Self:
        return type(self)(-self.val, -self.d1)

    def __add__(self, other: DualOrder1 | intrinsics.ArrayLike) -> Self:
        if isinstance(other, DualOrder1):
            self._check_compatible_dual(other)
            return type(self)(self.val + other.val, self.d1 + other.d1)
        return type(self)(self.val + intrinsics.as_f64(other), self.d1)

    def __radd__(self, other: intrinsics.ArrayLike) -> Self:
        return self.__add__(other)

    def __sub__(self, other: DualOrder1 | intrinsics.ArrayLike) -> Self:
        if isinstance(other, DualOrder1):
            self._check_compatible_dual(other)
            return type(self)(self.val - other.val, self.d1 - other.d1)
        return type(self)(self.val - intrinsics.as_f64(other), self.d1)

    def __rsub__(self, other: intrinsics.ArrayLike) -> Self:
        return type(self)(intrinsics.as_f64(other) - self.val, -self.d1)

    def __mul__(self, other: DualOrder1 | intrinsics.ArrayLike) -> Self:
        if isinstance(other, DualOrder1):
            return self.make_binary_operator(
                other, self.val * other.val, other.val, self.val
            )
        c = intrinsics.as_f64(other)
        return self.make_unary_operator(self.val * c, c)

    def __rmul__(self, other: intrinsics.ArrayLike) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: DualOrder1 | intrinsics.ArrayLike) -> Self:
        if isinstance(other, DualOrder1):
            inv_y = 1.0 / other.val
            return self.make_binary_operator(
                other,
                self.val * inv_y,
                inv_y,
                -self.val * inv_y * inv_y,
            )
        c = intrinsics.as_f64(other)
        return self.make_unary_operator(self.val / c, 1.0 / c)

    def __rtruediv__(self, other: intrinsics.ArrayLike) -> Self:
        c = intrinsics.as_f64(other)
        return self.powm1() * c

    def __pow__(self, other: DualOrder1 | intrinsics.ArrayLike) -> Self:
        return self.pow(other)


@jax.tree_util.register_pytree_node_class
class AutoDiffReal1VarOrder1(DualOrder1):
    """One-variable first-order dual number (`auto_diff_real_1var_order1`)."""

    __slots__ = ()
    NUM_VARS: ClassVar[int] = 1


@jax.tree_util.register_pytree_node_class
class AutoDiffReal2VarOrder1(DualOrder1):
    """Two-variable first-order dual number (`auto_diff_real_2var_order1`)."""

    __slots__ = ()
    NUM_VARS: ClassVar[int] = 2


@jax.tree_util.register_pytree_node_class
class AutoDiffReal4VarOrder1(DualOrder1):
    """Four-variable first-order dual number (`auto_diff_real_4var_order1`)."""

    __slots__ = ()
    NUM_VARS: ClassVar[int] = 4


@jax.tree_util.register_pytree_node_class
class AutoDiffRealStarOrder1(DualOrder1):
    """Stellar 33-variable first-order dual (`auto_diff_real_star_order1`)."""

    __slots__ = ()
    NUM_VARS: ClassVar[int] = STAR_ORDER1_VARS


AutoDiffReal4Coeff = AutoDiffReal4VarOrder1
