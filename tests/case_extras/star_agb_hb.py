# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Upstream AGB $^{13}\mathrm{C}$-pocket, horizontal-branch, low-$Z$ header,
and R CrB envelope-cut `StarExtras` ports (`#231`).

Covers:
- `star/test_suite/c13_pocket/src/run_star_extras.f90:30-583`
- `star/test_suite/hb_2M/src/run_star_extras.f90:30-240`
- `star/test_suite/low_z/src/run_star_extras.f90:38-204`
- `star/test_suite/R_CrB_star/src/run_star_extras.f90:37-182`
"""

from __future__ import annotations

import pathlib
from typing import Any

from flexmesa import constants
from tests.case_extras import _helpers


class C13PocketStarExtras:
    r"""Multi-stage MS/CHeB, thermal-pulse 3DUP, and $^{13}\mathrm{C}$-pocket
    tracker from `star/test_suite/c13_pocket/src/run_star_extras.f90:30-583`.
    """

    def __init__(self) -> None:
        self.ms_t0 = 0.0
        self.cheb_t0 = 0.0
        self.ms_t1 = 0.0
        self.cheb_t1 = 0.0
        self.m_1dup = 0.0
        self.mcore_tacheb = 0.0
        self.mcore_1tp = 0.0
        self.age_1tp = 0.0
        self.mcore_at_tp = 0.0
        self.age_at_tp = 0.0
        self.mcore_min_after_tp = 0.0
        self.mcore_2tp_with_3dup = 0.0
        self.age_2tp_with_3dup = 0.0
        self.tp_count = 0
        self.tp_with_3dup = 0
        self.in_lhe_peak = False
        self.initial_surface_c12 = 0.0
        self.pg_abundance_xrange: tuple[float, float] | None = None
        self.testhub_extras: dict[str, float] = {}
        self.within_tolerance: bool | None = None

    def extras_startup(self, state: dict[str, Any], restart: bool) -> None:
        """Initialize stage trackers or require `other_photo_read` on restart
        (`:44-65, 125-163`).
        """
        part = int(state["x_integer_ctrl"][0])
        if restart:
            if part in (1, 2, 4):
                raise NotImplementedError(
                    f"c13_pocket: extras_startup(restart=True) in part {part} "
                    "requires other_photo_read (:44-65) in flexmesa-star (#231)"
                )
            return
        if part == 1:
            self.ms_t0 = 0.0
            self.ms_t1 = 0.0
            self.cheb_t0 = 0.0
            self.cheb_t1 = 0.0
            self.m_1dup = 1.0e99
            self.mcore_tacheb = 0.0
        elif part == 2:
            self.mcore_1tp = 0.0
            self.age_1tp = 0.0
            self.mcore_at_tp = 0.0
            self.mcore_min_after_tp = 0.0
            self.tp_count = 0
            self.mcore_2tp_with_3dup = 0.0
            self.age_2tp_with_3dup = 0.0
            self.tp_with_3dup = 0
            self.in_lhe_peak = False
        elif part == 4:
            self.tp_count = 0
            self.in_lhe_peak = False
            self.initial_surface_c12 = float(state["surface_c12"])

    def extras_start_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:166-174`)."""
        return "keep_going"

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:178-189`)."""
        return "keep_going"

    def _finish_part1(self, state: dict[str, Any]) -> None:
        """Track MS/CHeB lifetimes and first dredge-up (`:335-374`)."""
        age = float(state["star_age"])
        l_surf = float(state["surface_l_lsun"])
        if self.ms_t0 == 0.0:
            if float(state["power_h_burn"]) > 0.99 * l_surf:
                self.ms_t0 = age
        elif self.ms_t1 == 0.0 and float(state["center_h1"]) < 1.0e-4:
            self.ms_t1 = age

        if self.ms_t1 != 0.0 and self.cheb_t0 == 0.0:
            if float(state["power_he_burn"]) > 0.01 * l_surf:
                self.cheb_t0 = age
        elif self.cheb_t1 == 0.0 and float(state["center_he4"]) < 1.0e-4:
            self.cheb_t1 = age
            self.mcore_tacheb = float(state["he_core_mass"])

        if self.ms_t1 > 0.0 and self.cheb_t0 == 0.0:
            conv_bot = float(state["conv_mx1_bot"])
            if float(state["conv_mx1_top"]) - conv_bot > 0.1:
                self.m_1dup = min(
                    conv_bot * float(state["star_mass"]), self.m_1dup
                )

    def _finish_part2(self, state: dict[str, Any]) -> str | tuple[str, str]:
        """Track thermal pulses and stop after 3rd 3DUP pulse (`:375-421`)."""
        he_core = float(state["he_core_mass"])
        age = float(state["star_age"])
        p_he = float(state["power_he_burn"])
        if not self.in_lhe_peak:
            if p_he > 1.0e4:
                self.in_lhe_peak = True
                self.tp_count += 1
                self.mcore_at_tp = he_core
                self.age_at_tp = age
                self.mcore_min_after_tp = self.mcore_at_tp
                if self.tp_count == 1:
                    self.mcore_1tp = he_core
                    self.age_1tp = age
                if (
                    self.tp_with_3dup > 0
                    and (self.tp_count - self.tp_with_3dup) == 1
                ):
                    self.mcore_2tp_with_3dup = he_core
                    self.age_2tp_with_3dup = age
        elif _helpers.fdiv(float(state["power_h_burn"]), p_he) > 10.0:
            self.in_lhe_peak = False

        if self.tp_count > 0:
            self.mcore_min_after_tp = min(self.mcore_min_after_tp, he_core)
        if (
            self.tp_with_3dup == 0
            and (self.mcore_min_after_tp - self.mcore_at_tp) < -1.0e-4
        ):
            self.tp_with_3dup = self.tp_count

        if (
            not self.in_lhe_peak
            and self.tp_with_3dup > 0
            and (self.tp_count - self.tp_with_3dup) == 2
        ):
            return ("terminate", "third pulse with 3DUP has occurred")
        return "keep_going"

    def _finish_part4(self, state: dict[str, Any]) -> str | tuple[str, str]:
        """Stop after one TP cycle and update `pgstar` `c13` window
        (`:424-461`).
        """
        result: str | tuple[str, str] = "keep_going"
        p_he = float(state["power_he_burn"])
        if not self.in_lhe_peak:
            if p_he > 1.0e4:
                self.in_lhe_peak = True
                self.tp_count += 1
                self.mcore_min_after_tp = self.mcore_at_tp
        elif _helpers.fdiv(float(state["power_h_burn"]), p_he) > 10.0:
            self.in_lhe_peak = False

        if not self.in_lhe_peak and self.tp_count == 1:
            result = ("terminate", "one thermal pulse cycle complete")

        nz = int(state["nz"])
        xa_c13 = state["xa_c13"]
        k_max = max(range(nz), key=lambda idx: float(xa_c13[idx]))
        if float(xa_c13[k_max]) > 0.01:
            m_msun = float(state["m"][k_max]) / constants.MSUN
            self.pg_abundance_xrange = (m_msun - 0.0001, m_msun + 0.0001)
        else:
            he_core = float(state["he_core_mass"])
            self.pg_abundance_xrange = (he_core - 0.0125, he_core + 0.0025)
        return result

    def extras_finish_step(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        """Dispatch per-stage step-end tracking (`:317-463`)."""
        part = int(state["x_integer_ctrl"][0])
        if part == 1:
            self._finish_part1(state)
            return "keep_going"
        if part == 2:
            return self._finish_part2(state)
        if part == 4:
            return self._finish_part4(state)
        return "keep_going"

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        r"""Populate `testhub_extras` and check part-4 $^{13}\mathrm{C}$ pocket
        tolerances (`:466-582`).
        """
        part = int(state["x_integer_ctrl"][0])
        if part == 1:
            self.testhub_extras = {
                "MS_lifetime": (self.ms_t1 - self.ms_t0) / 1.0e9,
                "m_1DUP": self.m_1dup,
                "mcore_TACHeB": self.mcore_tacheb,
            }
        elif part == 2:
            self.testhub_extras = {
                "mcore_1TP": self.mcore_1tp,
                "age_1TP": self.age_1tp / 1.0e9,
                "mcore_2TP_with_3DUP": self.mcore_2tp_with_3dup,
                "interpulse_time": (self.age_at_tp - self.age_2tp_with_3dup)
                / 1.0e3,
                "delta_mcore_TP": (self.mcore_at_tp - self.mcore_2tp_with_3dup)
                / 1.0e-3,
                "mass_DUP": (self.mcore_at_tp - self.mcore_min_after_tp)
                / 1.0e-3,
            }
        elif part == 4:
            self._after_evolve_part4(state)

    def _after_evolve_part4(self, state: dict[str, Any]) -> None:
        r"""Evaluate $^{13}\mathrm{C}$ pocket metrics and tolerances
        (`:515-576`).
        """
        nz = int(state["nz"])
        xa_c13 = state["xa_c13"]
        q = state["q"]
        dq = state["dq"]
        k_max = max(range(nz), key=lambda idx: float(xa_c13[idx]))
        max_c13 = float(xa_c13[k_max])
        mass_max_c13 = (
            float(state["mstar"])
            * (float(q[k_max]) + float(dq[k_max]) / 2.0)
            / constants.MSUN
        )
        pocket_dq = 0.0
        for k in range(nz):
            if float(xa_c13[k]) > 5.0e-3:
                pocket_dq += float(dq[k])
        pocket_mass_c13 = pocket_dq * float(state["star_mass"])
        delta_surface_c12 = (
            float(state["surface_c12"]) - self.initial_surface_c12
        )
        self.testhub_extras = {
            "max_c13": max_c13,
            "mass_max_c13": mass_max_c13,
            "pocket_mass_c13": pocket_mass_c13,
            "delta_surface_c12": delta_surface_c12,
        }
        x_ctrl = state["x_ctrl"]
        self.within_tolerance = not (
            abs(max_c13 - float(x_ctrl[0])) > 1.0e-2
            or abs(mass_max_c13 - float(x_ctrl[1])) > 1.0e-2
            or abs(pocket_mass_c13 - float(x_ctrl[2])) > 1.0e-5
            or abs(delta_surface_c12 - float(x_ctrl[3])) > 6.0e-4
        )


class Hb2MStarExtras:
    r"""Core-He-burning convective core mass at $Y_c = 0.5$ tracker from
    `star/test_suite/hb_2M/src/run_star_extras.f90:30-240`.
    """

    def __init__(self) -> None:
        self._mass_conv_core_y050 = -1.0
        self.testhub_extras: dict[str, float] = {}
        self.within_tolerance: bool | None = None

    def extras_startup(self, state: dict[str, Any], restart: bool) -> None:
        """Initialize `mass_conv_core_y050` or require `other_photo_read` in
        part 3 (`:39-53, 94-109`).
        """
        if not restart:
            self._mass_conv_core_y050 = -1.0
            return
        if int(state["x_integer_ctrl"][0]) == 3:
            raise NotImplementedError(
                "hb_2M: extras_startup(restart=True) with x_integer_ctrl(1)==3 "
                "requires other_photo_read (:39-53) in flexmesa-star (#231)"
            )

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:154-162`)."""
        return "keep_going"

    def extra_history_columns(
        self, _state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Return `mass_conv_core_y050` history column (`:165-187`)."""
        return [("mass_conv_core_y050", self._mass_conv_core_y050)]

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Return `zbar_div_abar` profile column (`:202-218`)."""
        return _helpers.zbar_div_abar_profile_column(state)

    def extras_finish_step(self, state: dict[str, Any]) -> str:
        """Record `mass_conv_core` on first drop below `center_he4 = 0.5` in
        part 3 (`:222-239`).
        """
        if int(state["x_integer_ctrl"][0]) == 3:
            if (
                float(state["center_he4"]) < 0.5
                and self._mass_conv_core_y050 < 0.0
            ):
                self._mass_conv_core_y050 = float(state["mass_conv_core"])
        return "keep_going"

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Record `testhub_extras` and check `x_ctrl(1:2)` bounds in part 3
        (`:112-150`).
        """
        if int(state["x_integer_ctrl"][0]) == 3:
            self.testhub_extras = {
                "mass_conv_core_y050": self._mass_conv_core_y050
            }
            x_ctrl = state["x_ctrl"]
            min_m = float(x_ctrl[0])
            max_m = float(x_ctrl[1])
            self.within_tolerance = not (
                self._mass_conv_core_y050 < min_m
                or self._mass_conv_core_y050 > max_m
            )


class LowZStarExtras:
    """Extra history and profile `mixing_length_alpha` header items from
    `star/test_suite/low_z/src/run_star_extras.f90:38-204`.
    """

    def extras_controls(self, _state: dict[str, Any]) -> None:
        """Require `extra_history_header_items` and `extra_profile_header_items`
        (`:38-60`).
        """
        raise NotImplementedError(
            "low_z: extras_controls registers extra_history_header_items "
            "(:55-56) and extra_profile_header_items (:57-58), which "
            "flexmesa-star StarExtras does not expose (#231)"
        )

    def extra_history_header_items(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Return `mixing_length_alpha` history header item (`:109-129`)."""
        return [("mixing_length_alpha", float(state["mixing_length_alpha"]))]

    def extra_profile_header_items(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Return `mixing_length_alpha` profile header item (`:132-152`)."""
        return [("mixing_length_alpha", float(state["mixing_length_alpha"]))]

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:88-96`)."""
        return "keep_going"

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:195-203`)."""
        return "keep_going"


class RCrBStarExtras:
    """Post-evolution `T > 1e8 K` cell locator writing `inlist_nzhi` from
    `star/test_suite/R_CrB_star/src/run_star_extras.f90:37-182`.
    """

    def __init__(self) -> None:
        self.saved_nzhi: int | None = None

    def extras_start_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:99-107`)."""
        return "keep_going"

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:111-119`)."""
        return "keep_going"

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:173-181`)."""
        return "keep_going"

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Write `&star_job set_nzhi = k /` to `inlist_nzhi` when
        `x_logical_ctrl(1)` is true (`:68-96`).
        """
        if not bool(state["x_logical_ctrl"][0]):
            return
        nz = int(state["nz"])
        t_arr = state["t"]
        # Matches Fortran `do k = 1, s% nz; if (s% T(k) > 1d8) exit; end do`,
        # which leaves `k == nz + 1` when no cell exceeds `1d8 K` (:84-86).
        k_1based = nz + 1
        for k in range(nz):
            if float(t_arr[k]) > 1.0e8:
                k_1based = k + 1
                break
        self.saved_nzhi = k_1based
        out_path = pathlib.Path(str(state["case_dir"])) / "inlist_nzhi"
        out_path.write_text(
            f" &star_job\n  set_nzhi = {k_1based:4d}\n /\n",
            encoding="utf-8",
        )
