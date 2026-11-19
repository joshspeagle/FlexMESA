# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream `binary/test_suite/*/src/run_binary_extras.f90` and coupled
`run_star_extras.f90` implementations (`#148`).
"""

from __future__ import annotations

import math
import re
from typing import Any

from flexmesa import _rust, constants

_BROTT_WIND_RE = re.compile(r"s%\s*other_wind\s*=>\s*brott_wind\b")
_DOUBLE_BH_FINISH_RE = re.compile(r"terminate due to overflow of l2 at zams")
_JDOT_GR_CHECK_RE = re.compile(
    r'names\(1\)\s*=\s*"expected_separation".*?64d0/5d0\s*\*\s*pow3\(standard_cgrav\)'
)
_JDOT_ML_CHECK_RE = re.compile(
    r'names\(1\)\s*=\s*"expected_separation".*?initial_separation_in_rsuns'
)
_WIND_FED_RE = re.compile(r"lx_tested_super_edd")


class DoubleBhBinaryExtras:
    """Brott et al. (2011) massive-star wind and evolutionary phase-transition
    logic from `binary/test_suite/double_bh/src/run_binary_extras.f90:161-265`
    and `binary/test_suite/double_bh/src/run_star_extras.f90:55-280`.
    """

    def other_wind_cgs(
        self,
        star_state: dict[str, Any],
        _star_idx: int,
        ignore_rlof_flag: bool,
    ) -> float | None:
        """Evaluate Brott et al. (2011) massive-star wind in CGS."""
        if ignore_rlof_flag:
            return 0.0
        l_lsun = max(float(star_state["surface_l_lsun"]), 1.0e-10)
        m_msun = max(float(star_state["star_mass"]), 1.0e-5)
        l1 = max(l_lsun * constants.LSUN, 1.0)
        t_k = float(star_state["teff"])
        if t_k <= 0.0:
            r0 = max(float(star_state["surface_r_rsun"]) * constants.RSUN, 1.0)
            t_k = _rust.pow(
                l1
                / (constants.PI * r0 * r0 * constants.CRAD * constants.CLIGHT),
                0.25,
            )
        t_k = max(t_k, 100.0)
        t2 = t_k * t_k
        t4 = t2 * t2
        r1 = max(
            _rust.safe_sqrt(
                l1 / (constants.PI * constants.CRAD * constants.CLIGHT * t4)
            ),
            1.0,
        )
        r_rsun = max(r1 / constants.RSUN, 1.0e-5)
        x_surf = float(star_state.get("surface_h1", 0.0))
        y_surf = float(star_state.get("surface_he4", 0.0))
        z_base = float(star_state.get("zbase", 0.0))
        if z_base <= 0.0:
            z_base = min(max(1.0 - x_surf - y_surf, 1.0e-6), 0.1)
        z_div_z_solar = z_base / 0.017

        log_z_ratio = _rust.safe_log10(z_div_z_solar)
        teff_jump = 1.0e3 * (61.2 + 2.59 * (-13.636 + 0.889 * log_z_ratio))
        if t_k > 27_500.0:
            alfa_vink = 1.0
        elif t_k < 22_500.0:
            alfa_vink = 0.0
        else:
            dt = 100.0
            if t_k > teff_jump + dt:
                alfa_vink = 1.0
            elif t_k < teff_jump - dt:
                alfa_vink = 0.0
            else:
                alfa_vink = (t_k - (teff_jump - dt)) / (2.0 * dt)

        z_scale_013 = _rust.pow(z_div_z_solar, 0.13)
        w1 = 0.0
        if alfa_vink > 0.0:
            vinf_div_vesc = 2.6 * z_scale_013
            log_t_40k = _rust.safe_log10(t_k / 4.0e4)
            log_mdot = (
                -6.697
                + 2.194 * _rust.safe_log10(l_lsun / 1.0e5)
                - 1.313 * _rust.safe_log10(m_msun / 30.0)
                - 1.226 * _rust.safe_log10(vinf_div_vesc / 2.0)
                + 0.933 * log_t_40k
                - 10.92 * log_t_40k * log_t_40k
                + 0.85 * log_z_ratio
            )
            w1 = _rust.exp10(log_mdot)

        w2 = 0.0
        if alfa_vink < 1.0:
            vinf_div_vesc = 1.3 * z_scale_013
            log_mdot = (
                -6.688
                + 2.210 * _rust.safe_log10(l_lsun / 1.0e5)
                - 1.339 * _rust.safe_log10(m_msun / 30.0)
                - 1.601 * _rust.safe_log10(vinf_div_vesc / 2.0)
                + 1.07 * _rust.safe_log10(t_k / 2.0e4)
                + 0.85 * log_z_ratio
            )
            w2 = _rust.exp10(log_mdot)

        vink_wind = alfa_vink * w1 + (1.0 - alfa_vink) * w2
        log_l_lsun = _rust.safe_log10(l_lsun)
        nieu_wind = _rust.exp10(
            -14.02
            + 1.24 * log_l_lsun
            + 0.16 * _rust.safe_log10(m_msun)
            + 0.81 * _rust.safe_log10(r_rsun)
            + 0.85 * log_z_ratio
        )
        hamann_wind = (
            _rust.exp10(
                -11.95 + 1.5 * log_l_lsun - 2.85 * x_surf + 0.85 * log_z_ratio
            )
        ) / 10.0

        if t_k < teff_jump:
            w_msun_yr = max(vink_wind, nieu_wind)
        else:
            if x_surf > 0.7:
                alfa_wr = 1.0
            elif x_surf > 0.4:
                alfa_wr = (x_surf - 0.4) / 0.3
            else:
                alfa_wr = 0.0
            w_msun_yr = alfa_wr * vink_wind + (1.0 - alfa_wr) * hamann_wind

        return float(
            -(max(w_msun_yr, 0.0) * (constants.MSUN / constants.SECYER))
        )

    def finish_step(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str] | dict[str, Any]:
        """Engage RLOF at thermal equilibrium and stop on He exhaustion."""
        updates: dict[str, Any] = {}
        ignore_rlof = bool(state.get("ignore_rlof_flag", False))
        s1_log_lnuc = float(state.get("s1_log_lnuc", -99.0))
        s1_log_l = float(state.get("s1_log_l", 0.0))
        s1_age = float(state.get("s1_star_age", 0.0))
        thermal_diff = abs(s1_log_lnuc - s1_log_l)

        if ignore_rlof and thermal_diff < 0.005 and s1_age > 100.0:
            rl_gaps = state.get("rl_relative_gap", [0.0, 0.0])
            gap1 = float(rl_gaps[0])
            if gap1 > 0.320819224:
                return ("terminate", "overflow_l2_at_zams")
            if gap1 > 0.0:
                updates["max_implicit_abs_mdot"] = 1.0e-3
                updates["delta_mdot_hard_limit"] = -1.0
            updates["ignore_rlof_flag"] = False
            updates["terminate_if_l2_overflow"] = True
        elif ignore_rlof and (thermal_diff > 0.005 or s1_age < 100.0):
            init_p = float(state.get("initial_period_sec", state["period"]))
            updates["period_eccentricity"] = (init_p, 0.0)

        if "s1_center_h1" in state:
            s1_h1 = float(state["s1_center_h1"])
            s1_he4 = float(state["s1_center_he4"])
            s1_she4 = float(state["s1_surface_he4"])
            if s1_h1 > 1.0e-3 and (s1_he4 - s1_she4) > 0.2:
                return ("terminate", "primary_not_homogeneous")
        else:
            s1_he4 = 1.0

        if "s2_center_h1" in state:
            s2_h1 = float(state["s2_center_h1"])
            s2_he4 = float(state["s2_center_he4"])
            s2_she4 = float(state["s2_surface_he4"])
            if s2_h1 > 1.0e-3 and (s2_he4 - s2_she4) > 0.2:
                return ("terminate", "secondary_not_homogeneous")

        pm_i = int(state.get("point_mass_i", 0))
        if pm_i == 0:
            if s1_he4 < 1.0e-5:
                updates["point_mass_i"] = 1
                pm_i = 1
            elif float(state.get("s2_center_he4", 1.0)) < 1.0e-5:
                updates["point_mass_i"] = 2
                pm_i = 2

        if pm_i == 1 and float(state.get("s2_center_he4", 1.0)) < 1.0e-5:
            return ("terminate", "helium_depletion")
        if pm_i == 2 and s1_he4 < 1.0e-5:
            return ("terminate", "helium_depletion")

        if updates:
            return {"action": "keep_going", "updates": updates}
        return "keep_going"


class JdotGrCheckBinaryExtras:
    """Peters (1964) gravitational-wave inspiral analytical check
    (`binary/test_suite/jdot_gr_check/src/run_binary_extras.f90:89-168`).
    """

    def _expected_sep_rsun(self, state: dict[str, Any]) -> float:
        a0 = float(state["initial_separation_cm"])
        init_masses = state["initial_mass_msun"]
        m1 = float(init_masses[0])
        m2 = float(init_masses[1])
        age_sec = (
            float(state.get("s1_star_age", state.get("binary_age_yr", 0.0)))
            * constants.SECYER
        )
        g = constants.STANDARD_CGRAV
        ms = constants.MSUN
        c = constants.CLIGHT
        c2 = c * c
        beta = (
            (64.0 / 5.0)
            * (g * g * g)
            * m1
            * m2
            * (m1 + m2)
            * (ms * ms * ms)
            / (c2 * c2 * c)
        )
        a02 = a0 * a0
        a_exp_4 = a02 * a02 - 4.0 * beta * age_sec
        return (
            _rust.pow(a_exp_4, 0.25) if a_exp_4 > 0.0 else 0.0
        ) / constants.RSUN

    def check_model(self, state: dict[str, Any]) -> str | tuple[str, str]:
        """Stop cleanly if RLO begins or analytical GW separation drifts."""
        if int(state.get("model_number", 0)) == 0:
            return "keep_going"
        if float(state.get("mtransfer_rate", 0.0)) != 0.0:
            return ("terminate", "beginning_of_rlof")
        exp_rsun = self._expected_sep_rsun(state)
        if exp_rsun > 0.0:
            cur_rsun = float(state["separation"]) / constants.RSUN
            rel_err = abs((exp_rsun - cur_rsun) / exp_rsun)
            if rel_err > 1.0e-2:
                return ("terminate", "large_separation_error")
        return "keep_going"

    def extra_binary_history_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Emit analytical `expected_separation` ($R_\\odot$) column."""
        return [("expected_separation", self._expected_sep_rsun(state))]


class JdotMlCheckBinaryExtras:
    """Tauris & van den Heuvel (2006) orbital separation check
    (`binary/test_suite/jdot_ml_check/src/run_binary_extras.f90:91-166`).
    """

    def _expected_sep_rsun(self, state: dict[str, Any]) -> float:
        a0_rsun = float(state["initial_separation_cm"]) / constants.RSUN
        init_masses = state["initial_mass_msun"]
        m1_0 = float(init_masses[0])
        m2_0 = float(init_masses[1])
        q0 = m1_0 / m2_0
        donor_m = float(state["donor_mass_msun"])
        accretor_m = float(state["accretor_mass_msun"])
        if int(state.get("donor_i", 1)) == 2:
            m1, m2 = accretor_m, donor_m
        else:
            m1, m2 = donor_m, accretor_m
        q = m1 / m2
        alpha = float(state["mass_transfer_alpha"])
        beta = float(state["mass_transfer_beta"])
        delta = float(state["mass_transfer_delta"])
        gamma = float(state["mass_transfer_gamma"])

        eps = 1.0 - alpha - beta - delta
        one_m_eps = max(1.0 - eps, 1.0e-15)
        term1 = _rust.pow(q / q0, 2.0 * (alpha + gamma * delta - 1.0))
        term2 = _rust.pow(
            (q + 1.0) / (q0 + 1.0), (-alpha - beta + delta) / one_m_eps
        )
        if abs(eps) > 1.0e-12:
            exp3 = 3.0 + 2.0 * (
                alpha * eps * eps + beta + gamma * delta * one_m_eps * one_m_eps
            ) / (eps * one_m_eps)
            term3 = _rust.pow((eps * q + 1.0) / (eps * q0 + 1.0), exp3)
        else:
            term3 = _rust.crmath_exp(2.0 * (beta + gamma * delta) * (q - q0))
        return float(a0_rsun * term1 * term2 * term3)

    def check_model(self, state: dict[str, Any]) -> str | tuple[str, str]:
        """Stop if numerical separation departs from analytical curve."""
        exp_rsun = self._expected_sep_rsun(state)
        if exp_rsun > 0.0 and not math.isnan(exp_rsun):
            cur_rsun = float(state["separation"]) / constants.RSUN
            if abs((exp_rsun - cur_rsun) / exp_rsun) > 1.0e-4:
                return ("terminate", "large_separation_error")
        return "keep_going"

    def extra_binary_history_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Emit analytical `expected_separation` ($R_\\odot$) column."""
        return [("expected_separation", self._expected_sep_rsun(state))]


def match_binary_extras(norm_bin_text: str, norm_star_text: str) -> Any:
    """Match binary extras strictly from normalized Fortran source content."""
    if _DOUBLE_BH_FINISH_RE.search(norm_bin_text) or _BROTT_WIND_RE.search(
        norm_star_text
    ):
        return DoubleBhBinaryExtras()
    if _JDOT_GR_CHECK_RE.search(norm_bin_text):
        return JdotGrCheckBinaryExtras()
    if _JDOT_ML_CHECK_RE.search(norm_bin_text):
        return JdotMlCheckBinaryExtras()
    return None
