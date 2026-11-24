# SPDX-License-Identifier: GPL-3.0-or-later
"""Upstream wind-mass, convective-premixing, white-dwarf electric-field,
gravity-darkening, and semiconvection `StarExtras` ports (`#231`).

Covers:
- `star/test_suite/hot_cool_wind/src/run_star_extras.f90:38-165`
- `star/test_suite/conv_core_cpm/src/run_star_extras.f90:36-197`
- `star/test_suite/wd_diffusion/src/run_star_extras.f90:38-179`
- `star/test_suite/high_rot_darkening/src/run_star_extras.f90:44-170`
- `star/test_suite/semiconvection/src/run_star_extras.f90:36-205`
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from flexmesa import constants
from tests.case_extras import _helpers


class HotCoolWindStarExtras:
    """Post-evolution luminosity and remnant-mass check from
    `star/test_suite/hot_cool_wind/src/run_star_extras.f90:38-165`.
    """

    def __init__(self) -> None:
        self.within_tolerance: bool | None = None

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:94-102`)."""
        return "keep_going"

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:156-164`)."""
        return "keep_going"

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Check `log_surface_luminosity >= log_L_upper_limit` and
        `6.73 < Mass < 6.77` (`:69-90`).
        """
        mass = float(state["mstar"]) / constants.MSUN
        log_l = float(state["log_surface_luminosity"])
        log_l_lim = float(state["log_l_upper_limit"])
        self.within_tolerance = bool(
            log_l >= log_l_lim and mass > 6.73 and mass < 6.77
        )


class ConvCoreCpmStarExtras:
    """Convective-premixing final `mass_conv_core` range check from
    `star/test_suite/conv_core_cpm/src/run_star_extras.f90:36-197`.
    """

    def __init__(self) -> None:
        self.mass_conv_core: float | None = None
        self.within_tolerance: bool | None = None

    def extras_startup(self, _state: dict[str, Any], _restart: bool) -> None:
        """Run startup (`extras_photo_read` is a no-op at `:57-62, 71-80`)."""

    def extras_start_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:114-122`)."""
        return "keep_going"

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:126-134`)."""
        return "keep_going"

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:188-196`)."""
        return "keep_going"

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Verify `x_ctrl(1) <= mass_conv_core <= x_ctrl(2)` (`:83-111`)."""
        x_ctrl = state["x_ctrl"]
        min_mcc = float(x_ctrl[0])
        max_mcc = float(x_ctrl[1])
        mcc = float(state["mass_conv_core"])
        self.mass_conv_core = mcc
        self.within_tolerance = not (mcc < min_mcc or mcc > max_mcc)


class WdDiffusionStarExtras:
    r"""White-dwarf cooling `Teff` guard and core $eE / (m_u g)$ check from
    `star/test_suite/wd_diffusion/src/run_star_extras.f90:38-179`.
    """

    def __init__(self) -> None:
        self.core_eemg: float | None = None
        self.within_tolerance: bool | None = None

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:108-116`)."""
        return "keep_going"

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:170-178`)."""
        return "keep_going"

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        r"""Require `Teff <= Teff_lower_limit` (`ierr = -1`) and evaluate core
        $eE / (m_u g)$ at $m < 0.5\,M_\odot$ (`:69-104`).
        """
        if float(state["teff"]) > float(state["teff_lower_limit"]):
            raise RuntimeError("failed to reach target for Teff")
        nz = int(state["nz"])
        m = state["m"]
        k0 = -1
        for k in range(nz):
            if float(m[k]) / constants.MSUN < 0.5:
                k0 = k
                break
        if k0 < 0:
            raise IndexError("mass profile does not fall below 0.5 Msun")
        eemg = _helpers.fdiv(
            constants.QE * float(state["e_field"][k0]),
            constants.AMU * float(state["g_field_element_diffusion"][k0]),
        )
        self.core_eemg = eemg
        self.within_tolerance = bool(eemg > 1.95 and eemg < 2.05)


class HighRotDarkeningStarExtras:
    """Rapid-rotation `w_div_w_crit_avg_surf` and surface Roche ratio step hook
    from `star/test_suite/high_rot_darkening/src/run_star_extras.f90:44-170`.
    """

    def __init__(self) -> None:
        self.last_w_div_w_crit_avg_surf: float | None = None
        self.last_w_div_w_crit_roche_surf: float | None = None

    def extras_start_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:82-92`)."""
        return "keep_going"

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:94-102`)."""
        return "keep_going"

    def extras_finish_step(self, state: dict[str, Any]) -> str:
        """Read surface rotation ratios after pre-MS creation (`:143-158`)."""
        if bool(state["create_pre_main_sequence_model"]):
            return "keep_going"
        self.last_w_div_w_crit_avg_surf = float(state["w_div_w_crit_avg_surf"])
        self.last_w_div_w_crit_roche_surf = float(
            state["w_div_w_crit_roche"][0]
        )
        return "keep_going"


def _dq_avg_slice(
    vec: Sequence[float], dq: Sequence[float], k_lo: int, k_hi: int
) -> float:
    """Evaluate `dot_product(v(k_lo:k_hi), dq(k_lo:k_hi)) / sum(dq(k_lo:k_hi))`
    for inclusive 0-based bounds `k_lo..k_hi`
    (`star/test_suite/semiconvection/src/run_star_extras.f90:112-115`).
    """
    num = 0.0
    den = 0.0
    for k in range(k_lo, k_hi + 1):
        dq_k = float(dq[k])
        num += float(vec[k]) * dq_k
        den += dq_k
    return _helpers.fdiv(num, den)


class SemiconvectionStarExtras:
    """Post-evolution mixing-zone and `dq`-averaged `(logT, logRho)` checks from
    `star/test_suite/semiconvection/src/run_star_extras.f90:36-205`.
    """

    def __init__(self) -> None:
        self.checks_okay: bool | None = None

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:134-142`)."""
        return "keep_going"

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:196-204`)."""
        return "keep_going"

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Locate `0.125, 0.135, 0.145 Msun` cells and run all 5 checks
        (`:67-130`).
        """
        nz = int(state["nz"])
        m = state["m"]
        msun = constants.MSUN
        k1 = -1
        k2 = -1
        k3 = -1
        for k in range(nz - 1, -1, -1):
            m_k = float(m[k])
            if m_k > 0.125 * msun and k1 < 0:
                k1 = k
            if m_k > 0.135 * msun and k2 < 0:
                k2 = k
            if m_k > 0.145 * msun and k3 < 0:
                k3 = k
        if min(k1, k2, k3) < 0:
            raise IndexError("mass profile does not span 0.125..0.145 Msun")
        mix = state["mixing_type"]
        dq = state["dq"]
        log_t_avg = _dq_avg_slice(state["lnt"], dq, k3, k2) / constants.LN10
        log_rho_avg = _dq_avg_slice(state["lnd"], dq, k3, k2) / constants.LN10
        self.checks_okay = all(
            (
                int(mix[k1]) == int(constants.MixingType.CONVECTIVE_MIXING),
                int(mix[k2]) == int(constants.MixingType.SEMICONVECTIVE_MIXING),
                int(mix[k3]) == int(constants.MixingType.NO_MIXING),
                7.15 <= log_t_avg <= 7.31,
                1.75 <= log_rho_avg <= 1.80,
            )
        )
