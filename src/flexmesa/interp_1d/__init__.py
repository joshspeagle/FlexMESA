# SPDX-License-Identifier: LGPL-3.0-or-later
"""One-dimensional piecewise-monotonic cubic interpolation for FlexMESA.

This subpackage ports upstream MESA ``interp_1d`` (`interp_pm`, `interp_m3a`,
`interp_m3b`, `interp_m3q`, `interp_4pt_pm`, `interp_value`,
`interp_value_and_slope`, `interp_values`, `interp_values_and_slopes`,
`integrate_values`, `interpolate_vector`, `interpolate_vector_pm`,
`interp_3_to_1`, `interp_3_to_2`, and `interp_4_to_1`) into ``float64``
NumPy/JAX with differentiable evaluation and ``DualOrder1`` support.
"""

from __future__ import annotations

from flexmesa.interp_1d import eval as _eval
from flexmesa.interp_1d import mp as _mp
from flexmesa.interp_1d import pm as _pm

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
interpolate_vector = _eval.interpolate_vector
interpolate_vector_pm = _eval.interpolate_vector_pm
interpolate_vector_pm_autodiff = _eval.interpolate_vector_pm_autodiff

__all__ = [
    "SMACHEPS",
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
    "interpolate_vector",
    "interpolate_vector_pm",
    "interpolate_vector_pm_autodiff",
]
