# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream Cepheid-crossing, carbon-flame, EOS-residual, PISN, and network
`StarExtras` ports (`#231`).

Covers:
- `star/test_suite/5M_cepheid_blue_loop/src/run_star_extras.f90`
- `star/test_suite/cburn_inward/src/run_star_extras.f90`
- `star/test_suite/conductive_flame/src/run_star_extras.f90`
- `star/test_suite/carbon_kh/src/run_star_extras.f90`
- `star/test_suite/pisn/src/run_star_extras.f90`
- `star/test_suite/adjust_net/src/run_star_extras.f90`
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from flexmesa import _rust, constants
from tests.case_extras import _helpers

_DBL_HUGE = sys.float_info.max


class CepheidBlueLoopStarExtras:
    """Instability-strip 3rd-crossing detector from
    `star/test_suite/5M_cepheid_blue_loop/src/run_star_extras.f90:38-318`.
    """

    def __init__(self) -> None:
        self._have_crossed_red_edge = False
        self._have_crossed_blue_edge = False

    def extras_startup(self, _state: dict[str, Any], restart: bool) -> None:
        """Reset crossing flags or fail loudly on photo restart (`:57-73`)."""
        if restart:
            raise NotImplementedError(
                "5M_cepheid_blue_loop: extras_startup(restart=True) unpacks "
                "s% extra_iwork (:71, 233-318), which requires photo "
                "extra_iwork persistence in flexmesa-star (#150)"
            )
        self._have_crossed_red_edge = False
        self._have_crossed_blue_edge = False

    def extras_check_model(self, state: dict[str, Any]) -> str:
        """Terminate on entering the 3rd blue-edge crossing (`:89-152`)."""
        x_ctrl = state["x_ctrl"]
        log_t1, log_l1 = float(x_ctrl[1]), float(x_ctrl[2])
        log_t2, log_l2 = float(x_ctrl[3]), float(x_ctrl[4])
        log_t3, log_l3 = float(x_ctrl[5]), float(x_ctrl[6])
        log_t4, log_l4 = float(x_ctrl[7]), float(x_ctrl[8])
        log_t = _rust.log10(float(state["teff"]))
        log_l = _rust.log10(float(state["surface_l_lsun"]))
        _ = int(state["model_number"])

        if log_l3 - log_l4 > 0.0:
            red_log_t = log_t4 + (log_l - log_l4) * (log_t3 - log_t4) / (
                log_l3 - log_l4
            )
        else:
            red_log_t = -1.0e99

        if abs(log_l1 - log_l2) > 1.0e-10:
            blue_log_t = log_t2 + (log_l - log_l2) * (log_t1 - log_t2) / (
                log_l1 - log_l2
            )
        else:
            # Exact upstream else-branch constant at :138 (`1d-99`).
            blue_log_t = 1.0e-99

        if log_t > red_log_t and not self._have_crossed_red_edge:
            self._have_crossed_red_edge = True
        if log_t > blue_log_t and not self._have_crossed_blue_edge:
            self._have_crossed_blue_edge = True
        if log_t < blue_log_t and self._have_crossed_blue_edge:
            return "terminate"
        return "keep_going"

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Keep going after step acceptance (`:206-215`)."""
        return "keep_going"


def has_ignited(state: dict[str, Any], k: int) -> bool:
    """Test carbon-flame ignition at 0-based cell `k`
    (`cburn_inward/src/run_star_extras.f90:229-255`).
    """
    co_core_mass = float(state["co_core_mass"])
    if co_core_mass <= 0.0:
        return False
    if float(state["m"][k]) / constants.MSUN >= co_core_mass:
        return False
    if float(state["net_nuclear_energy"][k]) < 0.0:
        return False
    ne_abun = float(state["xa_ne20"][k])
    na_abun = float(state["xa_na23"][k])
    mg_abun = float(state["xa_mg24"][k])
    he_abun = float(state["xa_he4"][k])
    return bool(ne_abun > na_abun and na_abun > mg_abun and he_abun < 1.0e-8)


class CburnInwardStarExtras:
    """Inward carbon-flame tracker from
    `star/test_suite/cburn_inward/src/run_star_extras.f90:39-288`.
    """

    def __init__(self) -> None:
        self._ign_mass = -1.0
        self._ign_density = -1.0
        self._ign_co_core_mass = -1.0
        self._flame_mass = -1.0
        self.testhub_extras: dict[str, float] = {}

    def extras_startup(self, state: dict[str, Any], restart: bool) -> None:
        """Reset ignition trackers or require `other_photo_read` (`:62-77`,
        `:257-271`).
        """
        if not restart:
            self._ign_mass = -1.0
            self._ign_density = -1.0
            self._ign_co_core_mass = -1.0
            self._flame_mass = -1.0
            return
        if int(state["x_integer_ctrl"][0]) == 2:
            raise NotImplementedError(
                "cburn_inward: extras_startup(restart=True) with "
                "x_integer_ctrl(1)==2 requires other_photo_read in "
                "flexmesa-star (#150)"
            )

    def extras_finish_step(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        """Track inward flame mass and stop at half `ign_mass` (`:181-226`)."""
        if int(state["x_integer_ctrl"][0]) == 2:
            nz = int(state["nz"])
            flame_cell = -1
            for k in range(nz - 1, -1, -1):
                if has_ignited(state, k):
                    flame_cell = k
                    break
            if flame_cell >= 0:
                m_flame = float(state["m"][flame_cell])
                if self._ign_mass < 0.0:
                    self._ign_mass = m_flame
                    self._ign_density = float(state["rho"][flame_cell])
                    self._ign_co_core_mass = float(state["co_core_mass"])
                self._flame_mass = m_flame
                if self._ign_mass > 0.0 and m_flame < 0.5 * self._ign_mass:
                    return ("terminate", "Terminate as flame reached half way")
        return "keep_going"

    def extra_history_columns(
        self, _state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Return the 4 flame-ignition history columns (`:122-150`)."""
        return [
            ("ign_mass", self._ign_mass / constants.MSUN),
            ("log(ign_den)", _rust.safe_log10(self._ign_density)),
            ("ign_co_core_mass", self._ign_co_core_mass),
            ("flame_mass", self._flame_mass / constants.MSUN),
        ]

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Populate `testhub_extras` in part 2 (`:80-107`)."""
        if int(state["x_integer_ctrl"][0]) == 2:
            self.testhub_extras = {
                "ign_mass": self._ign_mass / constants.MSUN,
                "ign_log_density": _rust.safe_log10(self._ign_density),
                "ign_co_core_mass": self._ign_co_core_mass,
            }


def flame_properties(
    eps_nuc: Sequence[float], r_arr: Sequence[float]
) -> tuple[float, float]:
    """Compute `(position, width)` in $\\mathrm{cm}$
    (`conductive_flame/src/run_star_extras.f90:260-287`).
    """
    nz = len(eps_nuc)
    kmax = max(range(nz), key=lambda idx: float(eps_nuc[idx]))
    eps_max = float(eps_nuc[kmax])
    thresh = 0.1 * eps_max
    rtop = 0.0
    for k in range(kmax, -1, -1):
        if float(eps_nuc[k]) < thresh:
            rtop = float(r_arr[k])
            break
    rbot = 0.0
    for k in range(kmax, nz):
        if float(eps_nuc[k]) < thresh:
            rbot = float(r_arr[k])
            break
    return float(r_arr[kmax]), max(0.0, rtop - rbot)


class ConductiveFlameStarExtras:
    """Conductive deflagration-wave tracker from
    `star/test_suite/conductive_flame/src/run_star_extras.f90:39-542`.
    """

    def __init__(self) -> None:
        self.constant_ln_p = 0.0
        self.constant_ln_t = 0.0
        self._flame_r0 = -1.0
        self._flame_t0 = -1.0
        self.testhub_extras: dict[str, float] = {}
        self.within_tolerance: bool | None = None

    def extras_controls(self, _state: dict[str, Any]) -> None:
        """Fail loudly on unexposed surface-PT and header hooks (`:201-238`)."""
        raise NotImplementedError(
            "conductive_flame: extras_controls registers other_surface_PT "
            "(:211, #161), build_isothermal_ball (:212, #154), and "
            "extra_history_header_items (:228-229, #231), which flexmesa-star "
            "StarExtras does not expose"
        )

    def other_build_initial_model(self, _ctx: Any) -> Any:
        """Fail loudly naming in-place `StarModel` allocation and EOS access
        (`:39-149`).
        """
        raise NotImplementedError(
            "conductive_flame: build_isothermal_ball (:39-149) requires "
            "mutable StarModel cell allocation and eosDT_get access in "
            "flexmesa-star (#154)"
        )

    def constant_surface_pt(self) -> dict[str, float]:
        """Return fixed surface `(lnT, lnP)` and zero partials (`:152-182`)."""
        return {
            "lnT_surf": self.constant_ln_t,
            "dlnT_dL": 0.0,
            "dlnT_dlnR": 0.0,
            "dlnT_dlnM": 0.0,
            "dlnT_dlnkap": 0.0,
            "lnP_surf": self.constant_ln_p,
            "dlnP_dL": 0.0,
            "dlnP_dlnR": 0.0,
            "dlnP_dlnM": 0.0,
            "dlnP_dlnkap": 0.0,
        }

    def extras_startup(self, _state: dict[str, Any], restart: bool) -> None:
        """Reset flame anchors or require `other_photo_read` (`:244-257`,
        `:185-191`).
        """
        if restart:
            raise NotImplementedError(
                "conductive_flame: extras_startup(restart=True) requires "
                "other_photo_read in flexmesa-star (#150)"
            )
        self._flame_r0 = -1.0
        self._flame_t0 = -1.0

    def extra_history_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Return `r_flame` and `l_flame` history columns (`:313-347`)."""
        nz = int(state["nz"])
        pos, width = flame_properties(state["eps_nuc"][:nz], state["r"][:nz])
        return [("r_flame", pos), ("l_flame", width)]

    def extra_history_header_items(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Return `x_ctrl(1)..x_ctrl(5)` history header items (`:389-425`)."""
        x_ctrl = state["x_ctrl"]
        return [(f"x_ctrl({i + 1})", float(x_ctrl[i])) for i in range(5)]

    def extras_finish_step(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        """Record 30% anchor and stop at 50% domain radius (`:459-496`)."""
        nz = int(state["nz"])
        r_arr = state["r"][:nz]
        pos, _ = flame_properties(state["eps_nuc"][:nz], r_arr)
        r_surf = float(r_arr[0])
        if self._flame_r0 < 0.0 and pos > 0.3 * r_surf:
            self._flame_r0 = pos
            self._flame_t0 = float(state["star_age"])
        if pos > 0.5 * r_surf:
            return ("terminate", "flame reached halfway point")
        return "keep_going"

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Evaluate flame speed and 10% target tolerances (`:499-542`)."""
        nz = int(state["nz"])
        pos, width = flame_properties(state["eps_nuc"][:nz], state["r"][:nz])
        dt_sec = (float(state["star_age"]) - self._flame_t0) * constants.SECYER
        speed = _helpers.fdiv(pos - self._flame_r0, dt_sec)
        self.testhub_extras = {"flame_speed": speed, "flame_width": width}
        x_ctrl = state["x_ctrl"]
        exp_speed = float(x_ctrl[8])
        exp_width = float(x_ctrl[9])
        self.within_tolerance = not (
            abs(speed - exp_speed) > 0.1 * exp_speed
            or abs(width - exp_width) > 0.1 * exp_width
        )


def _cell_eos_energy_diff(state: dict[str, Any], k: int) -> tuple[float, float]:
    """Compute `(dedt_minus_rse, rse_term)` in $\\mathrm{erg\\,g^{-1}}$
    (`carbon_kh/src/run_star_extras.f90:215-220, 246-249`).
    """
    de = float(state["energy"][k]) - float(state["energy_start"][k])
    term_t = (
        0.5
        * (
            float(state["t_start"][k]) * float(state["cv_start"][k])
            + float(state["t"][k]) * float(state["cv"][k])
        )
        * float(state["dxh_lnt"][k])
    )
    term_rho = (
        0.5
        * (
            float(state["rho_start"][k]) * float(state["de_drho_start"][k])
            + float(state["rho"][k]) * float(state["de_drho"][k])
        )
        * float(state["dxh_lnd"][k])
    )
    return (de - term_t) - term_rho, -term_t - term_rho


def in_eos_blend(state: dict[str, Any], k: int) -> bool:
    """Test whether 0-based cell `k` lies in an EOS blend
    (`carbon_kh/src/run_star_extras.f90:262-274`).
    """
    fracs = (
        float(state["eos_frac_opal_scvh"][k]),
        float(state["eos_frac_helm"][k]),
        float(state["eos_frac_skye"][k]),
        float(state["eos_frac_pc"][k]),
        float(state["eos_frac_cms"][k]),
        float(state["eos_frac_free_eos"][k]),
    )
    return any(0.0 < val < 1.0 for val in fracs)


class CarbonKhStarExtras:
    """Kelvin-Helmholtz EOS-blend energy residual tracker from
    `star/test_suite/carbon_kh/src/run_star_extras.f90:40-275`.
    """

    def __init__(self) -> None:
        self._e_res_err_step = 0.0
        self._e_res_err_run = 0.0
        self._e_eos_err_step = 0.0
        self._e_eos_err_run = 0.0
        self._e_eos_err_step_blend = 0.0
        self._e_eos_err_run_blend = 0.0

    def extras_startup(self, _state: dict[str, Any], restart: bool) -> None:
        """Zero accumulators or require `other_photo_read` (`:82-102`)."""
        if restart:
            raise NotImplementedError(
                "carbon_kh: extras_startup(restart=True) requires "
                "other_photo_read in flexmesa-star (#150)"
            )
        self._e_res_err_step = 0.0
        self._e_res_err_run = 0.0
        self._e_eos_err_step = 0.0
        self._e_eos_err_run = 0.0
        self._e_eos_err_step_blend = 0.0
        self._e_eos_err_run_blend = 0.0

    def extras_start_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`extras_start_step = 0`, `:117-126`)."""
        return "keep_going"

    def extras_finish_step(self, state: dict[str, Any]) -> str:
        """Accumulate step and cumulative EOS energy errors (`:227-259`)."""
        nz = int(state["nz"])
        ergs_error = state["ergs_error"]
        dm = state["dm"]
        e_res = 0.0
        e_eos = 0.0
        e_eos_blend = 0.0
        for k in range(nz):
            e_res += float(ergs_error[k])
            diff_k, _ = _cell_eos_energy_diff(state, k)
            cell_err = float(dm[k]) * diff_k
            e_eos += cell_err
            if in_eos_blend(state, k):
                e_eos_blend += cell_err
        self._e_res_err_step = e_res
        self._e_eos_err_step = e_eos
        self._e_eos_err_step_blend = e_eos_blend
        self._e_res_err_run += e_res
        self._e_eos_err_run += e_eos
        self._e_eos_err_run_blend += e_eos_blend
        return "keep_going"

    def extra_history_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Return the 6 relative EOS/residual history columns (`:141-181`)."""
        e_end = float(state["total_energy_end"])
        return [
            ("rel_e_res_err_step", _helpers.fdiv(self._e_res_err_step, e_end)),
            ("rel_e_res_err_run", _helpers.fdiv(self._e_res_err_run, e_end)),
            ("rel_e_eos_err_step", _helpers.fdiv(self._e_eos_err_step, e_end)),
            ("rel_e_eos_err_run", _helpers.fdiv(self._e_eos_err_run, e_end)),
            (
                "rel_e_eos_err_step_blend",
                _helpers.fdiv(self._e_eos_err_step_blend, e_end),
            ),
            (
                "rel_e_eos_err_run_blend",
                _helpers.fdiv(self._e_eos_err_run_blend, e_end),
            ),
        ]

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Return the 4 EOS-fraction and energy profile columns (`:184-223`)."""
        nz = int(state["nz"])
        dt = float(state["dt"])
        helm = [float(state["eos_frac_helm"][k]) for k in range(nz)]
        free_eos = [float(state["eos_frac_free_eos"][k]) for k in range(nz)]
        eps_eos: list[float] = []
        rse_grav: list[float] = []
        for k in range(nz):
            diff_k, rse_k = _cell_eos_energy_diff(state, k)
            eps_eos.append(diff_k * dt)
            rse_grav.append(rse_k * dt)
        return [
            ("eos_frac_HELM", helm),
            ("eos_frac_FreeEOS", free_eos),
            ("eps_eos", eps_eos),
            ("rse_eps_grav", rse_grav),
        ]


def gamma1_integral(state: dict[str, Any]) -> float:
    """Compute pressure-weighted $\\Gamma_1 - 4/3$ integral
    (`pisn/src/run_star_extras.f90:161-177`).
    """
    nz = int(state["nz"])
    peos = state["peos"]
    dm = state["dm"]
    rho = state["rho"]
    gamma1 = state["gamma1"]
    integral_norm = 0.0
    gamma1_int = 0.0
    four_thirds = 4.0 / 3.0
    for k in range(nz):
        weight = float(peos[k]) * float(dm[k]) / float(rho[k])
        integral_norm += weight
        gamma1_int += (float(gamma1[k]) - four_thirds) * weight
    # Exact upstream denominator guard `max(1d-99, integral_norm)` (:175).
    return gamma1_int / max(1.0e-99, integral_norm)


class PisnStarExtras:
    """Pair-instability pressure-weighted $\\langle\\Gamma_1 - 4/3\\rangle$ and
    unbinding hooks from `star/test_suite/pisn/src/run_star_extras.f90:30-272`.
    """

    def __init__(self) -> None:
        self._gamma1_cntr_pulse_start = _DBL_HUGE
        self.testhub_extras: dict[str, float] = {}

    def extras_controls(self, state: dict[str, Any]) -> None:
        """Require `other_remove_surface` when enabled (`:40-62`)."""
        if bool(state["use_other_remove_surface"]):
            raise NotImplementedError(
                "pisn: extras_controls registers other_remove_surface (:56), "
                "which flexmesa-star StarExtras does not expose (#153)"
            )

    def other_remove_surface(self, state: dict[str, Any]) -> None:
        """Guard `remove_ejecta_one_cell_per_step` surface cut (`:65-77`)."""
        if float(state["star_ejecta_mass"]) > 0.1 * constants.MSUN:
            raise NotImplementedError(
                "pisn: remove_ejecta_one_cell_per_step (:65-77) calls "
                "star_remove_surface_at_cell_k, which flexmesa-star does not "
                "implement (#153)"
            )

    def extras_startup(self, state: dict[str, Any], restart: bool) -> None:
        """Initialize `gamma1_cntr_pulse_start` or require `#150` in part 7
        (`:80-94`, `:241-255`).
        """
        if not restart:
            self._gamma1_cntr_pulse_start = _DBL_HUGE
            return
        if int(state["x_integer_ctrl"][0]) == 7:
            raise NotImplementedError(
                "pisn: extras_startup(restart=True) with x_integer_ctrl(1)==7 "
                "requires other_photo_read in flexmesa-star (#150)"
            )

    def extra_history_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Return `integral_gamma1` history column (`:132-158`)."""
        return [("integral_gamma1", gamma1_integral(state))]

    def extras_start_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:207-210`)."""
        return "keep_going"

    def extras_finish_step(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        """Stop when `total_energy > 0` and capture pulse onset (`:213-238`)."""
        result: str | tuple[str, str] = "keep_going"
        if float(state["total_energy"]) > 0.0:
            result = ("terminate", "Star is unbound")
        if int(state["x_integer_ctrl"][0]) == 7:
            if (
                self._gamma1_cntr_pulse_start > 1.0e50
                and gamma1_integral(state) < 0.0
            ):
                nz = int(state["nz"])
                self._gamma1_cntr_pulse_start = float(
                    state["gamma1"][nz - 1]
                ) - (4.0 / 3.0)
        return result

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Record `gamma1_cntr_pulse_start` in part 7 (`:97-117`)."""
        if int(state["x_integer_ctrl"][0]) == 7:
            self.testhub_extras["gamma1_cntr_pulse_start"] = (
                self._gamma1_cntr_pulse_start
            )


class AdjustNetStarExtras:
    """Post-evolution network species recorder from
    `star/test_suite/adjust_net/src/run_star_extras.f90:38-172`.
    """

    def __init__(self) -> None:
        self.testhub_extras: dict[str, float] = {}

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Record `num_isos = s% species` when `x_integer_ctrl(1) == 2`
        (`:69-92`).
        """
        if int(state["x_integer_ctrl"][0]) == 2:
            self.testhub_extras["num_isos"] = float(int(state["species"]))
