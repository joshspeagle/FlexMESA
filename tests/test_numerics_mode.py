# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for `NumericsMode` (`strict` vs `fast`) Python/PyO3 dispatch."""

from __future__ import annotations

import pytest

import flexmesa
from flexmesa import _rust
from flexmesa import math as fmath


def test_numerics_mode_switching_context_and_validation() -> None:
    """Verify default `strict` mode, context restoration, and error guards."""
    assert flexmesa.get_numerics_mode() == "strict"
    assert fmath.get_numerics_mode() == "strict"

    with flexmesa.numerics_mode_context("fast") as active:
        assert active == "fast"
        assert flexmesa.get_numerics_mode() == "fast"
        assert fmath.get_numerics_mode() == "fast"
    assert flexmesa.get_numerics_mode() == "strict"

    # Exception inside context still restores previous mode.
    with (
        pytest.raises(RuntimeError, match="boom"),
        flexmesa.numerics_mode_context("fast"),
    ):
        assert flexmesa.get_numerics_mode() == "fast"
        raise RuntimeError("boom")
    assert flexmesa.get_numerics_mode() == "strict"

    with pytest.raises(ValueError, match="invalid numerics mode"):
        flexmesa.set_numerics_mode("invalid")


def test_strict_vs_fast_rust_transcendental_and_small_arg_dispatch() -> None:
    """Verify `_rust` transcendentals switch by mode and keep small args."""
    # For x = 0.73, CRMATH `exp10` (`exp_rd(0.73 * log_rz(10.0))`) and platform
    # `(0.73 * LN10).exp()` differ by 1 ULP on IEEE-754 x86_64/aarch64.
    with flexmesa.numerics_mode_context("strict"):
        strict_exp10 = _rust.exp10(0.73)
        strict_pow = _rust.pow(0.0, 0.0)
    with flexmesa.numerics_mode_context("fast"):
        fast_exp10 = _rust.exp10(0.73)
        fast_pow = _rust.pow(0.0, 0.0)

    # `pow_r_(0.0, 0.0) == 0.0` in Strict (`math_lib_crmath.f90:229`), whereas
    # IEEE-754 `f64::powf(0.0, 0.0) == 1.0` in Fast mode.
    assert strict_pow == 0.0
    assert fast_pow == 1.0
    assert abs(strict_exp10 - fast_exp10) < 1.0e-14

    # Small-argument `log1p` and `expm1` avoid catastrophic cancellation.
    tiny = 1.0e-15
    assert abs(float(fmath.log1p(tiny)) - tiny) < 1.0e-29
    assert abs(float(fmath.expm1(tiny)) - tiny) < 1.0e-29

    # Dispatch overhead timing helper returns positive nanosecond measurements.
    t_s_exp, t_f_exp, t_s_fma, t_f_fma = _rust.measure_dispatch_overhead_ns(
        10_000
    )
    assert t_s_exp > 0.0 and t_f_exp > 0.0
    assert t_s_fma > 0.0 and t_f_fma > 0.0
