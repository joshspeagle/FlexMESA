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


def _fdiv(num: float, den: float) -> float:
    """Divide as Fortran does: IEEE-754, no exception on a zero divisor."""
    try:
        return num / den
    except ZeroDivisionError:
        if num == 0.0 or math.isnan(num):
            return math.nan
        return math.copysign(math.inf, num) * math.copysign(1.0, den)


class DoubleBhBinaryExtras:
    """Brott et al. (2011) massive-star wind and evolutionary phase-transition
    logic from `binary/test_suite/double_bh/src/run_binary_extras.f90:161-259`
    and `binary/test_suite/double_bh/src/run_star_extras.f90:55-206`.

    Every quantity upstream reads is indexed directly from the snapshot, so a
    missing one raises `KeyError` instead of being substituted.
    """

    def other_wind_cgs(
        self,
        star_state: dict[str, Any],
        _star_idx: int,
        ignore_rlof_flag: bool,
        l_surf: float,
        m_surf: float,
        r_surf: float,
        t_surf: float,
    ) -> float:
        """Evaluate `brott_wind`.

        Port of
        `binary/test_suite/double_bh/src/run_star_extras.f90:55-206`.

        Args:
            star_state: Star snapshot; `surface_h1` is `s% xa(h1,1)` and
                `zbase` is `s% kap_rq% Zbase`.
            _star_idx: Component index (unused, as upstream).
            ignore_rlof_flag: `b% ignore_rlof_flag`.
            l_surf: `Lsurf` argument of `other_wind` (erg s^-1).
            m_surf: `Msurf` argument (g).
            r_surf: `Rsurf` argument (cm).
            t_surf: `Tsurf` argument (K).

        Returns:
            Signed wind rate in g s^-1 (`-w*Msun/secyer`, where `w >= 0` is
            upstream's result in Msun yr^-1).
        """
        lsun = constants.LSUN
        msun = constants.MSUN
        # :78-81
        l1 = float(l_surf)
        m1 = float(m_surf)
        r1 = float(r_surf)
        t1 = float(t_surf)
        # :85 (`Ys`, :86, is read upstream but never used)
        xs = float(star_state["surface_h1"])
        # :88, :90
        z_div_z_solar = float(star_state["zbase"]) / 0.017
        log_z = _rust.log10(z_div_z_solar)
        teff_jump = 1.0e3 * (61.2 + 2.59 * (-13.636 + 0.889 * log_z))

        # :95-99
        if ignore_rlof_flag:
            return 0.0

        # eval_Vink_wind, :126-179
        if t1 > 27500.0:
            alfa = 1.0
        elif t1 < 22500.0:
            alfa = 0.0
        else:
            dt = 100.0
            if t1 > teff_jump + dt:
                alfa = 1.0
            elif t1 < teff_jump - dt:
                alfa = 0.0
            else:
                alfa = (t1 - (teff_jump - dt)) / (2 * dt)
        if alfa > 0:
            vinf_div_vesc = 2.6
            vinf_div_vesc = vinf_div_vesc * _rust.pow(z_div_z_solar, 0.13)
            log_t = _rust.log10(t1 / 4.0e4)
            log_mdot = (
                -6.697
                + 2.194 * _rust.log10(l1 / lsun / 1.0e5)
                - 1.313 * _rust.log10(m1 / msun / 30)
                - 1.226 * _rust.log10(vinf_div_vesc / 2.0)
                + 0.933 * log_t
                - 10.92 * (log_t * log_t)
                + 0.85 * log_z
            )
            w1 = _rust.exp10(log_mdot)
        else:
            w1 = 0.0
        if alfa < 1:
            vinf_div_vesc = 1.3
            vinf_div_vesc = vinf_div_vesc * _rust.pow(z_div_z_solar, 0.13)
            log_mdot = (
                -6.688
                + 2.210 * _rust.log10(l1 / lsun / 1.0e5)
                - 1.339 * _rust.log10(m1 / msun / 30)
                - 1.601 * _rust.log10(vinf_div_vesc / 2.0)
                + 1.07 * _rust.log10(t1 / 2.0e4)
                + 0.85 * log_z
            )
            w2 = _rust.exp10(log_mdot)
        else:
            w2 = 0.0
        vink_wind = alfa * w1 + (1 - alfa) * w2

        # eval_Nieuwenhuijzen_wind, :181-192
        nieu_wind = _rust.exp10(
            -14.02
            + 1.24 * _rust.log10(l1 / lsun)
            + 0.16 * _rust.log10(m1 / msun)
            + 0.81 * _rust.log10(r1 / constants.RSUN)
            + 0.85 * log_z
        )
        # eval_Hamann_wind, :194-204, then "use 1/10 hamann", :106
        hamann_wind = _rust.exp10(
            -11.95 + 1.5 * _rust.log10(l1 / lsun) - 2.85 * xs + 0.85 * log_z
        )
        hamann_wind = hamann_wind / 10.0

        # :108-120
        if t1 < teff_jump:
            w = max(vink_wind, nieu_wind)
        else:
            alfa = 0.0
            if xs > 0.7:
                alfa = 1.0
            elif xs > 0.4 and xs < 0.7:
                alfa = (xs - 0.4) / 0.3
            w = alfa * vink_wind + (1.0 - alfa) * hamann_wind

        # Msun/yr to g/s as `star/private/winds.f90:189` (`wind*Msun/secyer`);
        # the adapter contract is a signed rate (mass loss negative).
        return -(w * msun / constants.SECYER)

    def finish_step(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str] | dict[str, Any]:
        """Port of `extras_binary_finish_step`
        (`binary/test_suite/double_bh/src/run_binary_extras.f90:161-259`).
        """
        updates: dict[str, Any] = {}
        ignore_rlof = bool(state["ignore_rlof_flag"])
        if ignore_rlof:
            # :171-173 and :196-198:
            # abs(log10(abs(b% s1% L_nuc_burn_total * Lsun / b% s1% L(1))))
            thermal = abs(
                _rust.log10(
                    abs(
                        _fdiv(
                            float(state["s1_l_nuc_burn_total"])
                            * constants.LSUN,
                            float(state["s1_l_surf"]),
                        )
                    )
                )
            )
            s1_age = float(state["s1_star_age"])
            if thermal < 0.005 and s1_age > 1.0e2:
                gap1 = float(state["rl_relative_gap"][0])
                if gap1 > 0.320819224:
                    # :177-180
                    return (
                        "terminate",
                        "Terminate due to overflow of L2 at ZAMS",
                    )
                if gap1 > 0.0:
                    # :181-187
                    updates["max_implicit_abs_mdot"] = 1.0e-3
                    updates["delta_mdot_hard_limit"] = -1.0
                # :191-195
                updates["ignore_rlof_flag"] = False
                updates["terminate_if_l2_overflow"] = True
            elif thermal > 0.005 or s1_age < 1.0e2:
                # :199-204
                updates["period_eccentricity"] = (
                    float(state["initial_period_in_days"]) * constants.SECDAY,
                    0.0,
                )

        # :207-221. Upstream reads `b% s1` and `b% s2` unconditionally; a
        # star the snapshot does not carry raises `KeyError`.
        if float(state["s1_center_h1"]) > 1.0e-3:
            if (
                float(state["s1_center_he4"]) - float(state["s1_surface_he4"])
                > 0.2
            ):
                return (
                    "terminate",
                    "Terminate due to primary not evolving homogeneously",
                )
        if float(state["s2_center_h1"]) > 1.0e-3:
            if (
                float(state["s2_center_he4"]) - float(state["s2_surface_he4"])
                > 0.2
            ):
                return (
                    "terminate",
                    "Terminate due to secondary not evolving homogeneously",
                )

        # :225-240
        pm_i = int(state["point_mass_i"])
        if pm_i == 0:
            if (
                float(state["s1_center_he4"]) < 1.0e-5
                or float(state["s2_center_he4"]) < 1.0e-5
            ):
                # Upstream calls `binary_set_point_mass_i` and then sets
                # `b% eq_initial_bh_mass = b% m(i)` (:229, :235). FlexMESA's
                # binary system has no `eq_initial_bh_mass`, so the second
                # assignment cannot be made.
                raise NotImplementedError(
                    "double_bh: extras_binary_finish_step sets "
                    "b% eq_initial_bh_mass (upstream lines 229 and 235), "
                    "which the binary system does not carry"
                )
        # :242-255
        if pm_i == 1:
            if float(state["s2_center_he4"]) < 1.0e-5:
                return ("terminate", "Terminate due to helium depletion")
        elif pm_i == 2:
            if float(state["s1_center_he4"]) < 1.0e-5:
                return ("terminate", "Terminate due to helium depletion")

        if updates:
            return {"action": "keep_going", "updates": updates}
        return "keep_going"


class JdotGrCheckBinaryExtras:
    """Peters (1964) gravitational-wave inspiral check
    (`binary/test_suite/jdot_gr_check/src/run_binary_extras.f90:99-169`).

    Upstream computes the separation error while writing the history columns,
    stores it in `b% xtra(ix_separation_error)` and tests the stored value in
    `extras_binary_check_model`, so the test sees the value from the most
    recent history write. `_separation_error` is that stored value; it starts
    at zero as `b% xtra` does (`binary/private/binary_evolve.f90:176`).
    """

    def __init__(self) -> None:
        self._separation_error = 0.0

    def extra_binary_history_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Port of `data_for_extra_binary_history_columns` (:99-120)."""
        g = constants.STANDARD_CGRAV
        msun = constants.MSUN
        c = constants.CLIGHT
        rsun = constants.RSUN
        m1 = float(state["m1"])
        m2 = float(state["m2"])
        # :116; pow3 and pow5 are repeated products
        # (`math/private/math_pown.f90:89,109`).
        beta = (
            64.0
            / 5.0
            * (g * g * g)
            * m1
            * m2
            * (m1 + m2)
            * (msun * msun * msun)
            / (c * c * c * c * c)
        )
        # :117-118
        a0 = float(state["initial_separation_in_rsuns"]) * rsun
        expected = (
            _rust.pow(
                a0 * a0 * a0 * a0
                - 4 * beta * float(state["s1_star_age"]) * constants.SECYER,
                0.25,
            )
            / rsun
        )
        # :119
        self._separation_error = _fdiv(
            expected - float(state["separation"]) / rsun, expected
        )
        return [("expected_separation", expected)]

    def check_model(self, state: dict[str, Any]) -> str | tuple[str, str]:
        """Port of `extras_binary_check_model` (:151-169)."""
        if float(state["s1_mstar_dot"]) != 0:
            return ("terminate", "terminate due to beginning of RLO")
        if abs(self._separation_error) > 1.0e-2:
            return (
                "terminate",
                "Terminate because of large error in orbital separation",
            )
        return "keep_going"


class JdotMlCheckBinaryExtras:
    """Tauris & van den Heuvel orbital separation check
    (`binary/test_suite/jdot_ml_check/src/run_binary_extras.f90:99-178`).

    The separation error is stored by the history-columns routine and tested
    in `extras_binary_check_model`, as in `JdotGrCheckBinaryExtras`.
    """

    def __init__(self) -> None:
        self._separation_error = 0.0

    def extra_binary_history_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Port of `data_for_extra_binary_history_columns` (:99-132)."""
        # :117-124
        a0 = float(state["initial_separation_in_rsuns"])
        q = _fdiv(float(state["m"][0]), float(state["m"][1]))
        q0 = _fdiv(float(state["m1"]), float(state["m2"]))
        alpha = float(state["mass_transfer_alpha"])
        beta = float(state["mass_transfer_beta"])
        delta = float(state["mass_transfer_delta"])
        gamma = float(state["mass_transfer_gamma"])
        epsilon = 1 - alpha - beta - delta
        # :125-129 (`**2` is a product in compiled Fortran)
        expected = (
            a0
            * _rust.pow(_fdiv(q, q0), 2 * (alpha + gamma * delta - 1))
            * _rust.pow(
                _fdiv(q + 1, q0 + 1),
                _fdiv(-alpha - beta + delta, 1 - epsilon),
            )
            * _rust.pow(
                _fdiv(epsilon * q + 1, epsilon * q0 + 1),
                3
                + _fdiv(
                    2
                    * (
                        alpha * (epsilon * epsilon)
                        + beta
                        + gamma * delta * ((1 - epsilon) * (1 - epsilon))
                    ),
                    epsilon * (1 - epsilon),
                ),
            )
        )
        # :130
        self._separation_error = _fdiv(
            expected - float(state["separation"]) / constants.RSUN, expected
        )
        return [("expected_separation", expected)]

    def check_model(self, _state: dict[str, Any]) -> str | tuple[str, str]:
        """Port of `extras_binary_check_model` (:163-178)."""
        if abs(self._separation_error) > 1.0e-4:
            return (
                "terminate",
                "Terminate because of large error in orbital separation",
            )
        return "keep_going"


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
