# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream non-`RSP` `star/test_suite/*/src/run_star_extras.f90` dispatch
(`#148`, `#231`).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from tests.case_extras import (
    star_burn_flame,
    star_mixing_init,
    star_wd_hooks,
    star_winds_hooks,
)

AdjustNetStarExtras = star_burn_flame.AdjustNetStarExtras
CarbonKhStarExtras = star_burn_flame.CarbonKhStarExtras
CburnInwardStarExtras = star_burn_flame.CburnInwardStarExtras
CepheidBlueLoopStarExtras = star_burn_flame.CepheidBlueLoopStarExtras
ConductiveFlameStarExtras = star_burn_flame.ConductiveFlameStarExtras
ExtendedConvectivePenetrationStarExtras = (
    star_mixing_init.ExtendedConvectivePenetrationStarExtras
)
MagneticBrakingStarExtras = star_winds_hooks.MagneticBrakingStarExtras
MakeCoWdStarExtras = star_wd_hooks.MakeCoWdStarExtras
MakeEnvStarExtras = star_mixing_init.MakeEnvStarExtras
MakeONeWdStarExtras = star_wd_hooks.MakeONeWdStarExtras
OtherPhysicsHooksStarExtras = star_winds_hooks.OtherPhysicsHooksStarExtras
PisnStarExtras = star_burn_flame.PisnStarExtras
TzoStarExtras = star_mixing_init.TzoStarExtras
WdAccSmallDmStarExtras = star_wd_hooks.WdAccSmallDmStarExtras
WdAicStarExtras = star_wd_hooks.WdAicStarExtras


class CheckRedoStarExtras:
    """Step-redo hook logic from
    `star/test_suite/check_redo/src/run_star_extras.f90:60-128`.
    """

    def __init__(self) -> None:
        self._done_redo = False
        self._doing_restart = False

    def extras_startup(self, _state: dict[str, Any], restart: bool) -> None:
        """Reset one-shot redo state at startup or restore on restart."""
        self._done_redo = bool(restart)
        self._doing_restart = bool(restart)

    def extras_check_model(self, state: dict[str, Any]) -> str:
        """Trigger a step redo once at `model_number >= 10`."""
        if not self._done_redo and not self._doing_restart:
            if int(state["model_number"]) >= 10:
                self._done_redo = True
                return "redo"
        return "keep_going"

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Clear one-shot redo guard after the step finishes."""
        self._done_redo = False
        return "keep_going"


_STAR_DISPATCH_RULES: tuple[tuple[re.Pattern[str], Callable[[], Any]], ...] = (
    (re.compile(r"extras_check_model\s*=\s*redo\b"), CheckRedoStarExtras),
    (
        re.compile(r"s%\s*other_kap_get\s*=>\s*my_kap_get\b"),
        OtherPhysicsHooksStarExtras,
    ),
    (
        re.compile(r"terminate as flame reached half way"),
        CburnInwardStarExtras,
    ),
    (
        re.compile(
            r"s%\s*other_overshooting_scheme\s*=>\s*"
            r"extended_convective_penetration\b"
        ),
        ExtendedConvectivePenetrationStarExtras,
    ),
    (
        re.compile(r"crossed blue edge to start 3rd crossing"),
        CepheidBlueLoopStarExtras,
    ),
    (
        re.compile(r"check_initial_accretion_ok\s*\("),
        WdAccSmallDmStarExtras,
    ),
    (
        re.compile(
            r"s%\s*other_build_initial_model\s*=>\s*build_isothermal_ball\b"
        ),
        ConductiveFlameStarExtras,
    ),
    (re.compile(r"rel_e_eos_err_run_blend"), CarbonKhStarExtras),
    (
        re.compile(r"s%\s*other_torque\s*=>\s*magnetic_braking\b"),
        MagneticBrakingStarExtras,
    ),
    (
        re.compile(r"a=20 electron captures have started a thermal runaway"),
        WdAicStarExtras,
    ),
    (
        re.compile(r"call\s+create_env\s*\(\s*id\s*,\s*s\s*,\s*ierr\s*\)"),
        MakeEnvStarExtras,
    ),
    (
        re.compile(
            r"safe_log10\s*\(\s*s%\s*power_he_burn\s*\)\s*<\s*s%\s*x_ctrl\(3\)"
        ),
        MakeONeWdStarExtras,
    ),
    (
        re.compile(r"testhub_extras_names\(1\)\s*=\s*['\"]num_isos['\"]"),
        AdjustNetStarExtras,
    ),
    (
        re.compile(
            r"carbon has ignited in the interior and will produce an one wd"
        ),
        MakeCoWdStarExtras,
    ),
    (
        re.compile(r"pass:\s*have reached requested ns mass"),
        TzoStarExtras,
    ),
    (re.compile(r"gamma1_cntr_pulse_start"), PisnStarExtras),
)


def match_star_extras(norm_star_text: str) -> Any:
    """Match non-RSP upstream star extras strictly by Fortran source content."""
    for pattern, factory in _STAR_DISPATCH_RULES:
        if pattern.search(norm_star_text):
            return factory()
    return None
