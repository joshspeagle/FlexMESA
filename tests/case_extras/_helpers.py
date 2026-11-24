# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared IEEE-754 arithmetic and profile-column helpers for `tests.case_extras`
(`#231`).
"""

from __future__ import annotations

import math
from typing import Any


def fdiv(num: float, den: float) -> float:
    """Divide in IEEE-754 `float64` without raising `ZeroDivisionError`.

    Args:
        num: Dividend (`float64`).
        den: Divisor (`float64`).

    Returns:
        `num / den`, `nan` for `0.0 / 0.0` or `nan` inputs, or signed `inf`
        when `den == 0.0` and `num != 0.0`, matching compiled Fortran `real(dp)`
        division.
    """
    try:
        return num / den
    except ZeroDivisionError:
        if num == 0.0 or math.isnan(num):
            return math.nan
        return math.copysign(math.inf, num) * math.copysign(1.0, den)


def zbar_div_abar_profile_column(
    state: dict[str, Any],
) -> list[tuple[str, list[float]]]:
    """Compute the `zbar_div_abar` profile column `(nz,)` (dimensionless).

    Reproduces `data_for_extra_profile_columns` in:
    - `star/test_suite/wd_acc_small_dm/src/run_star_extras.f90:198-214`
    - `star/test_suite/magnetic_braking/src/run_star_extras.f90:251-267`
    - `star/test_suite/wd_aic/src/run_star_extras.f90:214-230`
    - `star/test_suite/hb_2M/src/run_star_extras.f90:202-218`

    Args:
        state: Star snapshot mapping containing `nz`, `zbar` `(nz,)`, and
            `abar` `(nz,)`.

    Returns:
        Single-element list `[("zbar_div_abar", vals)]` with `vals` of shape
        `(nz,)`.
    """
    nz = int(state["nz"])
    zbar = state["zbar"]
    abar = state["abar"]
    vals = [fdiv(float(zbar[k]), float(abar[k])) for k in range(nz)]
    return [("zbar_div_abar", vals)]
