# SPDX-License-Identifier: GPL-3.0-or-later
"""Nested dual number of the TDC Newton solver (`auto_diff_real_tdc`).

Mirrors `crates/flexmesa-turb/src/tdc_dual.rs` and upstream
`auto_diff/private/auto_diff_real_tdc_module.f90`. Each instance holds a primal
value `val`, its derivative with respect to the Newton variable $Z = \\ln|Y|$
(`d1val1`), `N` structure partials (`d1_array`), and the `N` mixed second
partials (`d1val1_d1_array`). Iterating `bracket_plus_newton_search` on this
type carries structure partials through every Newton update, matching
`turb/private/tdc.f90:253-345`.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from flexmesa import auto_diff as _ad

type _FloatArray = npt.NDArray[np.float64]


def _zeros(n: int) -> _FloatArray:
    """Return a 1D `float64` zero vector of length `n`."""
    return np.zeros((n,), dtype=np.float64)


class AutoDiffRealTdc:
    """Value, Newton derivative, `N` structure partials, and `N` mixed partials.

    Attributes:
        val: Primal scalar value (`val`).
        d1val1: Derivative with respect to the Newton variable $Z$ (`d1val1`).
        d1_array: First-order partials with respect to structure variables
            (`d1Array`), 1D `float64` array of shape `(N,)`.
        d1val1_d1_array: Mixed second partials with respect to $Z$ and the
            structure variables (`d1val1_d1Array`), 1D `float64` array of
            shape `(N,)`.
    """

    __slots__ = ("d1_array", "d1val1", "d1val1_d1_array", "val")

    def __init__(
        self,
        val: float,
        d1val1: float,
        d1_array: _FloatArray,
        d1val1_d1_array: _FloatArray,
    ) -> None:
        """Initialize an `AutoDiffRealTdc` number.

        Args:
            val: Primal scalar value.
            d1val1: Derivative with respect to the Newton variable $Z$.
            d1_array: Structure partials of shape `(N,)`.
            d1val1_d1_array: Mixed partials of shape `(N,)`.
        """
        self.val: float = float(val)
        self.d1val1: float = float(d1val1)
        self.d1_array: _FloatArray = np.asarray(d1_array, dtype=np.float64)
        self.d1val1_d1_array: _FloatArray = np.asarray(
            d1val1_d1_array, dtype=np.float64
        )

    @property
    def num_vars(self) -> int:
        """Return the number of tracked structure variables `N`."""
        return int(self.d1_array.shape[0])

    @classmethod
    def constant(cls, val: float, num_vars: int) -> AutoDiffRealTdc:
        """Construct a constant with `num_vars` zero structure partials.

        Args:
            val: Scalar value.
            num_vars: Number of structure variables `N`.

        Returns:
            Constant `AutoDiffRealTdc` with all derivatives zeroed.
        """
        z = _zeros(num_vars)
        return cls(float(val), 0.0, z, z.copy())

    @classmethod
    def convert(
        cls, x: float | _ad.DualOrder1, num_vars: int
    ) -> AutoDiffRealTdc:
        """Lift a scalar or `DualOrder1` into `AutoDiffRealTdc` (`convert`).

        Copies `val` and `d1_array` and zeroes `d1val1` and `d1val1_d1_array`
        (`turb/private/tdc_support.f90:424-430`).

        Args:
            x: Plain `float` or `DualOrder1` input.
            num_vars: Number of structure variables `N`.

        Returns:
            Lifted `AutoDiffRealTdc` instance.

        Raises:
            ValueError: If `x` is a `DualOrder1` whose arity differs from
                `num_vars`.
        """
        if isinstance(x, _ad.DualOrder1):
            d1 = np.asarray(x.d1, dtype=np.float64)
            if d1.shape != (num_vars,):
                msg = (
                    f"DualOrder1 arity {d1.shape} does not match "
                    f"expected ({num_vars},)"
                )
                raise ValueError(msg)
            return cls(float(x.val), 0.0, d1, _zeros(num_vars))
        return cls.constant(float(x), num_vars)

    def unconvert(self, *, is_dual: bool) -> float | _ad.DualOrder1:
        """Project back to `DualOrder1` or plain `float` (`unconvert`).

        Drops `d1val1` and `d1val1_d1_array` while preserving `val` and
        `d1_array` (`turb/private/tdc_support.f90:443-447`).

        Args:
            is_dual: Return a `DualOrder1` when `True`, or a `float` when
                `False`.

        Returns:
            `DualOrder1` carrying `(val, d1_array)` or scalar `float`.
        """
        if is_dual:
            return _ad.DualOrder1(self.val, self.d1_array)
        return self.val

    def with_val(self, new_val: float) -> AutoDiffRealTdc:
        """Return a copy with `val` replaced and all derivatives preserved."""
        return AutoDiffRealTdc(
            float(new_val),
            self.d1val1,
            self.d1_array,
            self.d1val1_d1_array,
        )

    def with_d1val1(self, d1val1: float) -> AutoDiffRealTdc:
        """Return a copy with `d1val1` set (`Z%d1val1 = 1d0`, `tdc.f90:261`)."""
        return AutoDiffRealTdc(
            self.val,
            float(d1val1),
            self.d1_array,
            self.d1val1_d1_array,
        )

    def differentiate_1(self) -> AutoDiffRealTdc:
        """Differentiate with respect to $Z$ (`differentiate_1`)."""
        z = _zeros(self.num_vars)
        return AutoDiffRealTdc(
            self.d1val1,
            0.0,
            self.d1val1_d1_array,
            z,
        )

    def _coerce(self, other: float | int | AutoDiffRealTdc) -> AutoDiffRealTdc:
        if isinstance(other, AutoDiffRealTdc):
            return other
        return AutoDiffRealTdc.constant(float(other), self.num_vars)

    def __add__(self, other: float | int | AutoDiffRealTdc) -> AutoDiffRealTdc:
        y = self._coerce(other)
        return AutoDiffRealTdc(
            self.val + y.val,
            self.d1val1 + y.d1val1,
            self.d1_array + y.d1_array,
            self.d1val1_d1_array + y.d1val1_d1_array,
        )

    def __radd__(self, other: float | int) -> AutoDiffRealTdc:
        return AutoDiffRealTdc.constant(float(other), self.num_vars) + self

    def __sub__(self, other: float | int | AutoDiffRealTdc) -> AutoDiffRealTdc:
        y = self._coerce(other)
        return AutoDiffRealTdc(
            self.val - y.val,
            self.d1val1 - y.d1val1,
            self.d1_array - y.d1_array,
            self.d1val1_d1_array - y.d1val1_d1_array,
        )

    def __rsub__(self, other: float | int) -> AutoDiffRealTdc:
        return AutoDiffRealTdc.constant(float(other), self.num_vars) - self

    def __mul__(self, other: float | int | AutoDiffRealTdc) -> AutoDiffRealTdc:
        x = self
        y = self._coerce(other)
        return AutoDiffRealTdc(
            x.val * y.val,
            x.d1val1 * y.val + x.val * y.d1val1,
            x.d1_array * y.val + x.val * y.d1_array,
            (
                x.d1_array * y.d1val1
                + x.d1val1 * y.d1_array
                + x.d1val1_d1_array * y.val
                + x.val * y.d1val1_d1_array
            ),
        )

    def __rmul__(self, other: float | int) -> AutoDiffRealTdc:
        return AutoDiffRealTdc.constant(float(other), self.num_vars) * self

    def __truediv__(
        self, other: float | int | AutoDiffRealTdc
    ) -> AutoDiffRealTdc:
        x = self
        y = self._coerce(other)
        q0 = y.val * y.val
        q1 = 1.0 / q0
        q2 = x.val * y.d1val1
        inv_y3 = 1.0 / (q0 * y.val)
        return AutoDiffRealTdc(
            x.val * (1.0 / y.val),
            q1 * (-q2 + x.d1val1 * y.val),
            q1 * (x.d1_array * y.val - x.val * y.d1_array),
            (
                2.0 * q2 * y.d1_array
                + q0 * x.d1val1_d1_array
                - y.val
                * (
                    x.d1_array * y.d1val1
                    + x.d1val1 * y.d1_array
                    + x.val * y.d1val1_d1_array
                )
            )
            * inv_y3,
        )

    def __rtruediv__(self, other: float | int) -> AutoDiffRealTdc:
        return AutoDiffRealTdc.constant(float(other), self.num_vars) / self

    def __neg__(self) -> AutoDiffRealTdc:
        return AutoDiffRealTdc(
            -self.val,
            -self.d1val1,
            -self.d1_array,
            -self.d1val1_d1_array,
        )

    def abs(self) -> AutoDiffRealTdc:
        """Evaluate $|x|$ (`abs_self`)."""
        if self.val < 0.0:
            q0 = -1.0
        elif self.val == 0.0:
            q0 = 0.0
        else:
            q0 = 1.0
        return AutoDiffRealTdc(
            abs(self.val),
            q0 * self.d1val1,
            q0 * self.d1_array,
            q0 * self.d1val1_d1_array,
        )

    def sqrt(self) -> AutoDiffRealTdc:
        """Evaluate $\\sqrt{x}$ (`sqrt_self`)."""
        q0 = math.sqrt(self.val)
        q1 = 0.5 * (1.0 / q0)
        inv = 1.0 / (q0 * q0 * q0)
        return AutoDiffRealTdc(
            q0,
            q1 * self.d1val1,
            q1 * self.d1_array,
            0.25
            * (
                2.0 * self.d1val1_d1_array * self.val
                - self.d1_array * self.d1val1
            )
            * inv,
        )

    def exp(self) -> AutoDiffRealTdc:
        """Evaluate $\\exp(x)$ (`exp_self`)."""
        q0 = math.exp(self.val)
        return AutoDiffRealTdc(
            q0,
            q0 * self.d1val1,
            q0 * self.d1_array,
            q0 * (self.d1_array * self.d1val1 + self.d1val1_d1_array),
        )

    def log(self) -> AutoDiffRealTdc:
        """Evaluate $\\ln(x)$ (`log_self`)."""
        q0 = 1.0 / self.val
        inv = 1.0 / (self.val * self.val)
        return AutoDiffRealTdc(
            math.log(self.val),
            q0 * self.d1val1,
            q0 * self.d1_array,
            (-self.d1_array * self.d1val1 + self.d1val1_d1_array * self.val)
            * inv,
        )

    def powf(self, y: float) -> AutoDiffRealTdc:
        """Evaluate $x^y$ for a scalar exponent `y` (`pow_self_real`)."""
        q0 = y * math.pow(self.val, y - 1.0)
        tail = math.pow(self.val, -2.0 + y)
        q1 = self.d1_array * self.d1val1
        return AutoDiffRealTdc(
            math.pow(self.val, y),
            q0 * self.d1val1,
            q0 * self.d1_array,
            y * (q1 * y - q1 + self.d1val1_d1_array * self.val) * tail,
        )

    def pow2(self) -> AutoDiffRealTdc:
        """Evaluate $x^2$ (`pow2_self`)."""
        q0 = 2.0 * self.val
        return AutoDiffRealTdc(
            self.val * self.val,
            q0 * self.d1val1,
            q0 * self.d1_array,
            2.0 * self.d1_array * self.d1val1 + q0 * self.d1val1_d1_array,
        )

    def pow3(self) -> AutoDiffRealTdc:
        """Evaluate $x^3$ (`pow3_self`)."""
        v2 = self.val * self.val
        q0 = 3.0 * v2
        return AutoDiffRealTdc(
            v2 * self.val,
            q0 * self.d1val1,
            q0 * self.d1_array,
            3.0
            * self.val
            * (
                2.0 * self.d1_array * self.d1val1
                + self.d1val1_d1_array * self.val
            ),
        )

    def pow4(self) -> AutoDiffRealTdc:
        """Evaluate $x^4$ (`pow4_self`)."""
        sq = self.val * self.val
        q0 = 4.0 * (sq * self.val)
        return AutoDiffRealTdc(
            sq * sq,
            q0 * self.d1val1,
            q0 * self.d1_array,
            4.0
            * (
                3.0 * self.d1_array * self.d1val1
                + self.d1val1_d1_array * self.val
            )
            * sq,
        )

    def tanh(self) -> AutoDiffRealTdc:
        """Evaluate $\\tanh(x)$ (`tanh_self`)."""
        q0 = math.tanh(self.val)
        q1 = q0 * q0 - 1.0
        ch = math.cosh(self.val)
        inv = 1.0 / (ch * ch)
        return AutoDiffRealTdc(
            q0,
            -q1 * self.d1val1,
            -q1 * self.d1_array,
            -(2.0 * q0 * self.d1_array * self.d1val1 - self.d1val1_d1_array)
            * inv,
        )

    def tan(self) -> AutoDiffRealTdc:
        """Evaluate $\\tan(x)$ (`tan_self`)."""
        q0 = math.tan(self.val)
        q1 = q0 * q0 + 1.0
        c = math.cos(self.val)
        inv = 1.0 / (c * c)
        return AutoDiffRealTdc(
            q0,
            q1 * self.d1val1,
            q1 * self.d1_array,
            (2.0 * q0 * self.d1_array * self.d1val1 + self.d1val1_d1_array)
            * inv,
        )

    def atan(self) -> AutoDiffRealTdc:
        """Evaluate $\\arctan(x)$ (`atan_self`)."""
        q0 = self.val * self.val + 1.0
        q1 = 1.0 / q0
        inv = 1.0 / (q0 * q0)
        return AutoDiffRealTdc(
            math.atan(self.val),
            q1 * self.d1val1,
            q1 * self.d1_array,
            (
                -2.0 * self.d1_array * self.d1val1 * self.val
                + q0 * self.d1val1_d1_array
            )
            * inv,
        )
