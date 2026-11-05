# SPDX-License-Identifier: LGPL-3.0-or-later
"""One-dimensional piecewise-monotonic cubic interpolation for FlexMESA.

This subpackage ports upstream MESA ``interp_1d`` into ``float64`` NumPy/JAX
with differentiable evaluation and ``DualOrder1`` forward-mode support:

- **Steffen (1990) piecewise-monotonic cubic** (`interp_pm`,
  `interp_pm_slopes_only`, `interp_4pt_pm`, `interp_pm_autodiff`): guarantees
  strict monotonicity on every subinterval without introducing spurious
  extrema.
- **Huynh (1993) monotonicity-preserving cubics** (`interp_m3a`, `interp_m3b`,
  `interp_m3q`, `interp_m3`): preserves local extrema and third-order accuracy
  at smooth peaks via median and minmod slope limiters (`MpLimiter`).
- **Evaluation, integration, and remeshing** (`interp_value`,
  `interp_value_and_slope`, `interp_values`, `interp_values_and_slopes`,
  `integrate_values`, `interpolate_vector`, `interpolate_vector_pm`,
  `interp_3_to_1`, `interp_3_to_2`, `interp_4_to_1`).
"""

from __future__ import annotations

from flexmesa.interp_1d import eval as _eval
from flexmesa.interp_1d import mp as _mp
from flexmesa.interp_1d import pm as _pm

MP_WORK_SIZE = _pm.MP_WORK_SIZE
MpLimiter = _mp.MpLimiter
PM_WORK_SIZE = _pm.PM_WORK_SIZE
SMACHEPS = _pm.SMACHEPS
integrate_values = _eval.integrate_values
interp_3_to_1 = _eval.interp_3_to_1
interp_3_to_2 = _eval.interp_3_to_2
interp_4_to_1 = _eval.interp_4_to_1
interp_4pt_pm = _pm.interp_4pt_pm
interp_m3 = _mp.interp_m3
interp_m3a = _mp.interp_m3a
interp_m3b = _mp.interp_m3b
interp_m3q = _mp.interp_m3q
interp_pm = _pm.interp_pm
interp_pm_autodiff = _pm.interp_pm_autodiff
interp_pm_slopes_only = _pm.interp_pm_slopes_only
interp_value = _eval.interp_value
interp_value_and_slope = _eval.interp_value_and_slope
interp_values = _eval.interp_values
interp_values_and_slopes = _eval.interp_values_and_slopes
interp_values_autodiff = _eval.interp_values_autodiff
interpolate_vector = _eval.interpolate_vector
interpolate_vector_pm = _eval.interpolate_vector_pm
interpolate_vector_pm_autodiff = _eval.interpolate_vector_pm_autodiff
median = _mp.median
minmod = _mp.minmod
minmod4 = _mp.minmod4

__all__ = [
    "MP_WORK_SIZE",
    "PM_WORK_SIZE",
    "SMACHEPS",
    "MpLimiter",
    "integrate_values",
    "interp_3_to_1",
    "interp_3_to_2",
    "interp_4_to_1",
    "interp_4pt_pm",
    "interp_m3",
    "interp_m3a",
    "interp_m3b",
    "interp_m3q",
    "interp_pm",
    "interp_pm_autodiff",
    "interp_pm_slopes_only",
    "interp_value",
    "interp_value_and_slope",
    "interp_values",
    "interp_values_and_slopes",
    "interp_values_autodiff",
    "interpolate_vector",
    "interpolate_vector_pm",
    "interpolate_vector_pm_autodiff",
    "median",
    "minmod",
    "minmod4",
]
