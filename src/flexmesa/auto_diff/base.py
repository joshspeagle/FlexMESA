# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared typed elementary method mixin (`AutoDiffBase`) for `auto_diff`.

Provides all 32 unary elementary functions (`sin`, `cos`, `exp`, `log`,
`safe_log`, `safe_sqrt`, etc.), fixed integer powers (`pow2` through `pow8`),
scalar exponentiation (`powf`, `powi`, `rpow`), and value-based relational
comparisons shared by `DualOrder1` and `AutoDiffReal2VarOrder2`.
"""

from __future__ import annotations

from typing import Self

import jax
import jax.numpy as jnp

from flexmesa.auto_diff import kernels
from flexmesa.math import intrinsics


class AutoDiffBase:
    """Mixin providing typed elementary methods and value comparisons."""

    val: jax.Array

    def apply_unary(self, kernel: kernels.UnaryKernel) -> Self:
        """Propagate a unary kernel `(f, f', f'')` through `self`."""
        raise NotImplementedError

    def sign(self) -> Self:
        """Compute $\\operatorname{sgn}(x)$ with zero derivative."""
        return self.apply_unary(kernels.unary_sign)

    def abs(self) -> Self:
        """Compute $|x|$."""
        return self.apply_unary(kernels.unary_abs)

    def sqrt(self) -> Self:
        """Compute $\\sqrt{x}$."""
        return self.apply_unary(kernels.unary_sqrt)

    def safe_sqrt(self) -> Self:
        """Compute $\\sqrt{\\max(x, 0)}$."""
        return self.apply_unary(kernels.unary_safe_sqrt)

    def exp(self) -> Self:
        """Compute $\\exp(x)$."""
        return self.apply_unary(kernels.unary_exp)

    def expm1(self) -> Self:
        """Compute $e^{x} - 1$."""
        return self.apply_unary(kernels.unary_expm1)

    def exp10(self) -> Self:
        """Compute $10^{x}$."""
        return self.apply_unary(kernels.unary_exp10)

    def powm1(self) -> Self:
        """Compute $x^{-1}$."""
        return self.apply_unary(kernels.unary_powm1)

    def log(self) -> Self:
        """Compute $\\ln(x)$."""
        return self.apply_unary(kernels.unary_log)

    def safe_log(self) -> Self:
        """Compute $\\operatorname{safe\\_log}(x)$."""
        return self.apply_unary(kernels.unary_safe_log)

    def log1p(self) -> Self:
        """Compute $\\ln(1 + x)$."""
        return self.apply_unary(kernels.unary_log1p)

    def log10(self) -> Self:
        """Compute $\\log_{10}(x)$."""
        return self.apply_unary(kernels.unary_log10)

    def safe_log10(self) -> Self:
        """Compute $\\operatorname{safe\\_log10}(x)$."""
        return self.apply_unary(kernels.unary_safe_log10)

    def log2(self) -> Self:
        """Compute $\\log_{2}(x)$."""
        return self.apply_unary(kernels.unary_log2)

    def sin(self) -> Self:
        """Compute $\\sin(x)$."""
        return self.apply_unary(kernels.unary_sin)

    def cos(self) -> Self:
        """Compute $\\cos(x)$."""
        return self.apply_unary(kernels.unary_cos)

    def tan(self) -> Self:
        """Compute $\\tan(x)$."""
        return self.apply_unary(kernels.unary_tan)

    def sinpi(self) -> Self:
        """Compute $\\sin(\\pi x)$."""
        return self.apply_unary(kernels.unary_sinpi)

    def cospi(self) -> Self:
        """Compute $\\cos(\\pi x)$."""
        return self.apply_unary(kernels.unary_cospi)

    def tanpi(self) -> Self:
        """Compute $\\tan(\\pi x)$."""
        return self.apply_unary(kernels.unary_tanpi)

    def sinh(self) -> Self:
        """Compute $\\sinh(x)$."""
        return self.apply_unary(kernels.unary_sinh)

    def cosh(self) -> Self:
        """Compute $\\cosh(x)$."""
        return self.apply_unary(kernels.unary_cosh)

    def tanh(self) -> Self:
        """Compute $\\tanh(x)$."""
        return self.apply_unary(kernels.unary_tanh)

    def asin(self) -> Self:
        """Compute $\\arcsin(x)$."""
        return self.apply_unary(kernels.unary_asin)

    def acos(self) -> Self:
        """Compute $\\arccos(x)$."""
        return self.apply_unary(kernels.unary_acos)

    def atan(self) -> Self:
        """Compute $\\arctan(x)$."""
        return self.apply_unary(kernels.unary_atan)

    def asinpi(self) -> Self:
        """Compute $\\arcsin(x) / \\pi$."""
        return self.apply_unary(kernels.unary_asinpi)

    def acospi(self) -> Self:
        """Compute $\\arccos(x) / \\pi$."""
        return self.apply_unary(kernels.unary_acospi)

    def atanpi(self) -> Self:
        """Compute $\\arctan(x) / \\pi$."""
        return self.apply_unary(kernels.unary_atanpi)

    def asinh(self) -> Self:
        """Compute $\\operatorname{arsinh}(x)$."""
        return self.apply_unary(kernels.unary_asinh)

    def acosh(self) -> Self:
        """Compute $\\operatorname{arcosh}(x)$."""
        return self.apply_unary(kernels.unary_acosh)

    def atanh(self) -> Self:
        """Compute $\\operatorname{artanh}(x)$."""
        return self.apply_unary(kernels.unary_atanh)

    def pow2(self) -> Self:
        """Compute $x^{2}$."""
        return self.apply_unary(lambda x: kernels.unary_pown(x, 2))

    def pow3(self) -> Self:
        """Compute $x^{3}$."""
        return self.apply_unary(lambda x: kernels.unary_pown(x, 3))

    def pow4(self) -> Self:
        """Compute $x^{4}$."""
        return self.apply_unary(lambda x: kernels.unary_pown(x, 4))

    def pow5(self) -> Self:
        """Compute $x^{5}$."""
        return self.apply_unary(lambda x: kernels.unary_pown(x, 5))

    def pow6(self) -> Self:
        """Compute $x^{6}$."""
        return self.apply_unary(lambda x: kernels.unary_pown(x, 6))

    def pow7(self) -> Self:
        """Compute $x^{7}$."""
        return self.apply_unary(lambda x: kernels.unary_pown(x, 7))

    def pow8(self) -> Self:
        """Compute $x^{8}$."""
        return self.apply_unary(lambda x: kernels.unary_pown(x, 8))

    def powf(self, exponent: intrinsics.ArrayLike) -> Self:
        """Raise `self` to a constant floating-point `exponent` ($x^{p}$)."""
        return self.apply_unary(lambda x: kernels.unary_pow_scalar(x, exponent))

    def powi(self, exponent: int) -> Self:
        """Raise `self` to a constant integer `exponent` ($x^{n}$)."""
        return self.powf(float(exponent))

    def rpow(self, base: intrinsics.ArrayLike) -> Self:
        """Raise positive scalar `base` to exponent `self` ($b^{x}$)."""
        return self.apply_unary(lambda x: kernels.unary_rpow_scalar(base, x))

    def __abs__(self) -> Self:
        return self.abs()

    def __rpow__(self, other: intrinsics.ArrayLike) -> Self:
        return self.rpow(other)

    def __eq__(self, other: object) -> bool:
        val_other = other.val if isinstance(other, AutoDiffBase) else other
        return bool(jnp.all(self.val == val_other))

    def __lt__(self, other: AutoDiffBase | float | int) -> bool:
        val_other = other.val if isinstance(other, AutoDiffBase) else other
        return bool(jnp.all(self.val < val_other))

    def __le__(self, other: AutoDiffBase | float | int) -> bool:
        val_other = other.val if isinstance(other, AutoDiffBase) else other
        return bool(jnp.all(self.val <= val_other))

    def __gt__(self, other: AutoDiffBase | float | int) -> bool:
        val_other = other.val if isinstance(other, AutoDiffBase) else other
        return bool(jnp.all(self.val > val_other))

    def __ge__(self, other: AutoDiffBase | float | int) -> bool:
        val_other = other.val if isinstance(other, AutoDiffBase) else other
        return bool(jnp.all(self.val >= val_other))
