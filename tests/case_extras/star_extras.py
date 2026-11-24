# SPDX-License-Identifier: GPL-3.0-or-later
"""Upstream non-`RSP` `star/test_suite/*/src/run_star_extras.f90` dispatch
(`#148`, `#231`).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from tests.case_extras import (
    star_agb_hb,
    star_atm_pulse,
    star_burn_flame,
    star_ccsn_iip,
    star_core_collapse,
    star_mixing_init,
    star_ns_wd_simplex,
    star_post_checks,
    star_ppisn,
    star_relax_hydro,
    star_rot_mix_checks,
    star_wd_hooks,
    star_winds_hooks,
)

AccretedMaterialJStarExtras = star_relax_hydro.AccretedMaterialJStarExtras
AdjustNetStarExtras = star_burn_flame.AdjustNetStarExtras
C13PocketStarExtras = star_agb_hb.C13PocketStarExtras
CarbonKhStarExtras = star_burn_flame.CarbonKhStarExtras
CburnInwardStarExtras = star_burn_flame.CburnInwardStarExtras
CcsnIIpStarExtras = star_ccsn_iip.CcsnIIpStarExtras
CepheidBlueLoopStarExtras = star_burn_flame.CepheidBlueLoopStarExtras
CheckPulseAtmStarExtras = star_atm_pulse.CheckPulseAtmStarExtras
ConductiveFlameStarExtras = star_burn_flame.ConductiveFlameStarExtras
ConserveAngularMomentumStarExtras = (
    star_relax_hydro.ConserveAngularMomentumStarExtras
)
ConvCoreCpmStarExtras = star_post_checks.ConvCoreCpmStarExtras
DiffusionSmoothnessStarExtras = (
    star_rot_mix_checks.DiffusionSmoothnessStarExtras
)
Dynamo15MStarExtras = star_rot_mix_checks.Dynamo15MStarExtras
ExtendedConvectivePenetrationStarExtras = (
    star_mixing_init.ExtendedConvectivePenetrationStarExtras
)
GyreInMesaStarExtras = star_atm_pulse.GyreInMesaStarExtras
Hb2MStarExtras = star_agb_hb.Hb2MStarExtras
HighRotation20MZ2m2StarExtras = (
    star_rot_mix_checks.HighRotation20MZ2m2StarExtras
)
HighRotDarkeningStarExtras = star_post_checks.HighRotDarkeningStarExtras
HotCoolWindStarExtras = star_post_checks.HotCoolWindStarExtras
HseRiemannStarExtras = star_relax_hydro.HseRiemannStarExtras
LowZStarExtras = star_agb_hb.LowZStarExtras
MagneticBrakingStarExtras = star_winds_hooks.MagneticBrakingStarExtras
MakeCoWdStarExtras = star_wd_hooks.MakeCoWdStarExtras
MakeEnvStarExtras = star_mixing_init.MakeEnvStarExtras
MakeONeWdStarExtras = star_wd_hooks.MakeONeWdStarExtras
MakePreCcsn13bvnStarExtras = star_core_collapse.MakePreCcsn13bvnStarExtras
MakeSdbStarExtras = star_relax_hydro.MakeSdbStarExtras
MsOpMonoStarExtras = star_atm_pulse.MsOpMonoStarExtras
NeutronStarEnvStarExtras = star_ns_wd_simplex.NeutronStarEnvStarExtras
OtherPhysicsHooksStarExtras = star_winds_hooks.OtherPhysicsHooksStarExtras
PisnStarExtras = star_burn_flame.PisnStarExtras
PpisnStarExtras = star_ppisn.PpisnStarExtras
PreMs1MToWdStarExtras = star_ns_wd_simplex.PreMs1MToWdStarExtras
PreMs12MCoreCollapseStarExtras = (
    star_core_collapse.PreMs12MCoreCollapseStarExtras
)
PreMs20MCoreCollapseStarExtras = (
    star_core_collapse.PreMs20MCoreCollapseStarExtras
)
RadiativeLevitationStarExtras = (
    star_rot_mix_checks.RadiativeLevitationStarExtras
)
RCrBStarExtras = star_agb_hb.RCrBStarExtras
RelaxCompositionJEntropyStarExtras = (
    star_relax_hydro.RelaxCompositionJEntropyStarExtras
)
SemiconvectionStarExtras = star_post_checks.SemiconvectionStarExtras
SimplexSolarCalibrationStarExtras = (
    star_ns_wd_simplex.SimplexSolarCalibrationStarExtras
)
SplitBurnBigNetStarExtras = star_core_collapse.SplitBurnBigNetStarExtras
Thermohaline1MStarExtras = star_rot_mix_checks.Thermohaline1MStarExtras
TTauGradrStarExtras = star_atm_pulse.TTauGradrStarExtras
TzoStarExtras = star_mixing_init.TzoStarExtras
WdAccSmallDmStarExtras = star_wd_hooks.WdAccSmallDmStarExtras
WdAicStarExtras = star_wd_hooks.WdAicStarExtras
WdCool06MStarExtras = star_ns_wd_simplex.WdCool06MStarExtras
WdDiffusionStarExtras = star_post_checks.WdDiffusionStarExtras
WdNovaBurstStarExtras = star_ns_wd_simplex.WdNovaBurstStarExtras


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
    (
        re.compile(r"gyre_in_mesa_extras_finish_step\s*\("),
        GyreInMesaStarExtras,
    ),
    (
        re.compile(r"relaxed model does not match input"),
        RelaxCompositionJEntropyStarExtras,
    ),
    (
        re.compile(
            r"t_face\s*=\s*exp\s*\(\s*\(\s*s%\s*dq\(k-1\)\s*\*\s*s%\s*lnt\(k\)"
        ),
        TTauGradrStarExtras,
    ),
    (
        re.compile(r"mass\s*>\s*6\.73d0\s*\.and\.\s*mass\s*<\s*6\.77d0"),
        HotCoolWindStarExtras,
    ),
    (
        re.compile(r"s%\s*other_adjust_mdot\s*=>\s*accretor_adjust_mdot\b"),
        AccretedMaterialJStarExtras,
    ),
    (
        re.compile(r"test passed:\s*mass_conv_core within specified range"),
        ConvCoreCpmStarExtras,
    ),
    (
        re.compile(r"passed test for electric field in the core"),
        WdDiffusionStarExtras,
    ),
    (re.compile(r"third pulse with 3dup has occurred"), C13PocketStarExtras),
    (
        re.compile(r"call\s+star_export_pulse_data\s*\(.*?['\"]fgong['\"]"),
        CheckPulseAtmStarExtras,
    ),
    (
        re.compile(r"w_div_wcrit_roche\(1\)\s*is:"),
        HighRotDarkeningStarExtras,
    ),
    (re.compile(r"mixing type at 0\.135 msun"), SemiconvectionStarExtras),
    (re.compile(r"mass_conv_core_y050"), Hb2MStarExtras),
    (
        re.compile(
            r"s%\s*how_many_extra_profile_header_items\s*=>\s*"
            r"how_many_extra_profile_header_items.*?names\(1\)\s*=\s*"
            r"['\"]mixing_length_alpha['\"]"
        ),
        LowZStarExtras,
    ),
    (
        re.compile(r"envelope mass outside tolerance,\s*retrying"),
        MakeSdbStarExtras,
    ),
    (
        re.compile(r"successfully used op_mono opacities"),
        MsOpMonoStarExtras,
    ),
    (re.compile(r"file\s*=\s*['\"]inlist_nzhi['\"]"), RCrBStarExtras),
    (
        re.compile(r"failed to keep constant total angular momentum"),
        ConserveAngularMomentumStarExtras,
    ),
    (
        re.compile(
            r"stop time should be 1d5 seconds.*?shock has reached target "
            r"location"
        ),
        HseRiemannStarExtras,
    ),
    (
        re.compile(r"subroutine\s+low_density_wind_routine\b"),
        CcsnIIpStarExtras,
    ),
    (
        re.compile(r"z_div_z_solar\s*=\s*s%\s*kap_rq%\s*zbase\s*/\s*0\.0142d0"),
        PpisnStarExtras,
    ),
    (
        re.compile(r"include\s+['\"]xtra_coeff_os/xtra_coeff_os\.inc['\"]"),
        PreMs1MToWdStarExtras,
    ),
    (
        re.compile(
            r"s%\s*other_mesh_fcn_data\s*=>\s*gradr_grada_mesh_fcn_data\b"
        ),
        SimplexSolarCalibrationStarExtras,
    ),
    (
        re.compile(r"names\(1\)\s*=\s*['\"]log_total_drag_energy['\"]"),
        MakePreCcsn13bvnStarExtras,
    ),
    (
        re.compile(
            r"testhub_extras_names\(1\)\s*=\s*['\"]total_burn_solver_steps['\"]"
        ),
        SplitBurnBigNetStarExtras,
    ),
    (
        re.compile(
            r"select\s+case\s*\(\s*s%\s*x_integer_ctrl\(5\)\s*\)\s*"
            r"case\s*\(\s*6\s*\)\s*testhub_extras_names\(1\)\s*=\s*"
            r"['\"]fe_core_mass['\"]"
        ),
        PreMs12MCoreCollapseStarExtras,
    ),
    (
        re.compile(
            r"select\s+case\s*\(\s*s%\s*x_integer_ctrl\(5\)\s*\)\s*"
            r"case\s*\(\s*7\s*\)\s*testhub_extras_names\(1\)\s*=\s*"
            r"['\"]fe_core_mass['\"]"
        ),
        PreMs20MCoreCollapseStarExtras,
    ),
    (
        re.compile(r"avg\s+from\s+3\.0\s+to\s+3\.4\s+msun"),
        Dynamo15MStarExtras,
    ),
    (
        re.compile(r"avg\s+near\s+0\.245\s+msun"),
        Thermohaline1MStarExtras,
    ),
    (
        re.compile(r"brunt_n2_composition_term\s+is\s+smooth\s+enough"),
        DiffusionSmoothnessStarExtras,
    ),
    (
        re.compile(r"found\s+expected\s+effects\s+of\s+radiative\s+levitation"),
        RadiativeLevitationStarExtras,
    ),
    (
        re.compile(r"include\s+['\"]other_cgrav\.inc['\"]"),
        NeutronStarEnvStarExtras,
    ),
    (
        re.compile(r"names\(2\)\s*=\s*['\"]eospc_fraction['\"]"),
        WdCool06MStarExtras,
    ),
    (
        re.compile(r"have\s+finished\s+burst"),
        WdNovaBurstStarExtras,
    ),
    (
        re.compile(
            r"^module\s+run_star_extras\b"
            r"(?:(?!other_alpha_mlt|other_photo_read|other_cgrav|other_torque|"
            r"xtra_coeff_os|eospc_fraction|h1_mass_added|electron\s+captures).)*"
            r"names\(1\)\s*=\s*['\"]zbar_div_abar['\"]"
        ),
        HighRotation20MZ2m2StarExtras,
    ),
)


def match_star_extras(norm_star_text: str) -> Any:
    """Match non-RSP upstream star extras strictly by Fortran source content."""
    make_env = star_mixing_init.match_make_env_extras(norm_star_text)
    if make_env is not None:
        return make_env
    for pattern, factory in _STAR_DISPATCH_RULES:
        if pattern.search(norm_star_text):
            return factory()
    return None
