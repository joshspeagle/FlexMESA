# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream white-dwarf accretion, AIC electron-capture, and CO/ONe WD
`StarExtras` ports (`#231`).

Covers:
- `star/test_suite/wd_acc_small_dm/src/run_star_extras.f90`
- `star/test_suite/wd_aic/src/run_star_extras.f90`
- `star/test_suite/make_o_ne_wd/src/run_star_extras.f90`
- `star/test_suite/make_co_wd/src/run_star_extras.f90`
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from flexmesa import _rust, constants
from tests.case_extras import _helpers


class WdAccSmallDmStarExtras:
    """Small-`dm` hydrogen accretion mass-buildup and profile checks from
    `star/test_suite/wd_acc_small_dm/src/run_star_extras.f90:39-322`.
    """

    def __init__(self) -> None:
        self._initial_h1_mass = 0.0
        self._h1_acc_abund = 0.0
        self._h1_init_surf_abund = 0.0
        self.all_ok = True

    def extras_startup(self, state: dict[str, Any], restart: bool) -> None:
        """Record initial `H1` mass and surface abundance (`:62-67, 76-109`)."""
        self._h1_acc_abund = float(state["accretion_h1"])
        if restart:
            raise NotImplementedError(
                "wd_acc_small_dm: extras_startup(restart=True) requires "
                "other_photo_read in flexmesa-star (#150)"
            )
        nz = int(state["nz"])
        dm = state["dm"]
        xa_h1 = state["xa_h1"]
        total_h1 = 0.0
        for k in range(nz):
            total_h1 += float(xa_h1[k]) * float(dm[k])
        self._initial_h1_mass = total_h1
        self._h1_init_surf_abund = float(xa_h1[0])

    def _compute_h1_mass_added(
        self, dm: Sequence[float], xa_h1: Sequence[float]
    ) -> float:
        """Compute `h1_mass_added` in $M_\\odot$ (`:173-179, 245-250`)."""
        h1_mass = 0.0
        for k in range(len(dm)):
            h1_mass += float(dm[k]) * float(xa_h1[k])
        h1_mass /= constants.MSUN
        return h1_mass - self._initial_h1_mass / constants.MSUN

    def extra_history_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Return `h1_mass_added` in $M_\\odot$ (`:148-183`)."""
        nz = int(state["nz"])
        added = self._compute_h1_mass_added(
            state["dm"][:nz], state["xa_h1"][:nz]
        )
        return [("h1_mass_added", added)]

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Return `zbar_div_abar` profile column (`:186-214`)."""
        return _helpers.zbar_div_abar_profile_column(state)

    def check_initial_accretion_ok(
        self, dm: Sequence[float], xa_h1: Sequence[float], accreted_mass: float
    ) -> bool:
        """Verify `h1_mass_added` matches `accreted_mass * h1_acc_abund` within
        0.5% (`:238-271`).
        """
        added = self._compute_h1_mass_added(dm, xa_h1)
        expected = accreted_mass * self._h1_acc_abund
        if _helpers.fdiv(abs(added - expected), expected) > 0.005:
            raise RuntimeError("check_initial_accretion_ok")
        return True

    def extras_finish_step(self, state: dict[str, Any]) -> str:
        """Check accreted `H1` mass at end of each step (`:218-235`)."""
        nz = int(state["nz"])
        accreted_mass = float(state["star_age"]) * float(state["star_mdot"])
        ok = self.check_initial_accretion_ok(
            state["dm"][:nz], state["xa_h1"][:nz], accreted_mass
        )
        self.all_ok = self.all_ok and ok
        return "keep_going"

    def check_accretion_profile_ok(
        self, dm: Sequence[float], xa_h1: Sequence[float], accreted_mass: float
    ) -> bool:
        """Verify accreted and underlying `H1` abundance layers (`:273-321`)."""
        accreted_mass_g = accreted_mass * constants.MSUN
        prof_all_ok = True
        k = 0
        xm = 0.0
        while xm < 0.8 * accreted_mass_g:
            if abs(float(xa_h1[k]) - self._h1_acc_abund) > 0.005:
                prof_all_ok = False
            xm += float(dm[k])
            k += 1
        while xm < 1.2 * accreted_mass_g:
            xm += float(dm[k])
            k += 1
        while xm < 2.0 * accreted_mass_g:
            if abs(float(xa_h1[k]) - self._h1_init_surf_abund) > 0.001:
                prof_all_ok = False
            xm += float(dm[k])
            k += 1
        return prof_all_ok

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Run end-of-evolution abundance profile verification (`:112-133`)."""
        nz = int(state["nz"])
        accreted_mass = float(state["star_age"]) * float(state["star_mdot"])
        ok = self.check_accretion_profile_ok(
            state["dm"][:nz], state["xa_h1"][:nz], accreted_mass
        )
        self.all_ok = self.all_ok and ok


def center_avg_x(
    dq_arr: Sequence[float],
    xa_iso: Sequence[float],
    center_avg_value_dq: float,
) -> float:
    """Compute central `dq`-weighted isotope average
    (`wd_aic/src/run_star_extras.f90:38-57`).
    """
    sum_x = 0.0
    sum_dq = 0.0
    for k in range(len(dq_arr) - 1, -1, -1):
        dq = float(dq_arr[k])
        dx = float(xa_iso[k]) * dq
        if sum_dq + dq >= center_avg_value_dq:
            sum_x += dx * (center_avg_value_dq - sum_dq) / dq
            sum_dq = center_avg_value_dq
            break
        sum_x += dx
        sum_dq += dq
    return _helpers.fdiv(sum_x, sum_dq)


class WdAicStarExtras:
    """Accretion-induced collapse $A=24$ and $A=20$ electron-capture checks from
    `star/test_suite/wd_aic/src/run_star_extras.f90:38-243`.
    """

    def extras_check_model(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        """Evaluate $A=24$ and $A=20$ capture thresholds (`:104-176`)."""
        if int(state["x_integer_ctrl"][0]) <= 0:
            return "keep_going"
        nz = int(state["nz"])
        dq_arr = state["dq"][:nz]
        c_dq = float(state["center_avg_value_dq"])
        center_mg24 = center_avg_x(dq_arr, state["xa_mg24"][:nz], c_dq)
        center_na24 = center_avg_x(dq_arr, state["xa_na24"][:nz], c_dq)
        center_ne20 = center_avg_x(dq_arr, state["xa_ne20"][:nz], c_dq)
        log_rho_c = float(state["log_center_density"])
        log_t_c = float(state["log_center_temperature"])

        if log_rho_c >= 9.7 and (center_mg24 + center_na24) > 0.01:
            return (
                "terminate",
                "FAIL: A=24 electron captures should have occurred by now",
            )
        if log_t_c > 8.8:
            if (0.45 - center_ne20) < 0.02:
                return (
                    "terminate",
                    "FAIL: A=20 electron captures should have occurred by now",
                )
            return (
                "terminate",
                "PASS: A=20 electron captures have started a thermal runaway "
                "in the core",
            )
        if log_rho_c >= 10.0:
            return (
                "terminate",
                "FAIL: electron captures should have occurred by now",
            )
        return "keep_going"

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Return `zbar_div_abar` profile column (`:202-230`)."""
        return _helpers.zbar_div_abar_profile_column(state)


class MakeONeWdStarExtras:
    """Part-2 ONe white-dwarf cooling/burning stop condition from
    `star/test_suite/make_o_ne_wd/src/run_star_extras.f90:36-181`.
    """

    def extras_finish_step(self, state: dict[str, Any]) -> str:
        """Terminate when `L/Lsun`, `Teff`, and `safe_log10(power_he_burn)` pass
        `x_ctrl(1:3)` (`:160-180`).
        """
        if int(state["x_integer_ctrl"][0]) == 2:
            x_ctrl = state["x_ctrl"]
            if (
                float(state["surface_l_lsun"]) < float(x_ctrl[0])
                and float(state["teff"]) > float(x_ctrl[1])
                and _rust.safe_log10(float(state["power_he_burn"]))
                < float(x_ctrl[2])
            ):
                return "terminate"
        return "keep_going"


class MakeCoWdStarExtras:
    """Carbon-ignition check, post-TP diffusion toggle, and envelope cut from
    `star/test_suite/make_co_wd/src/run_star_extras.f90:37-225`.
    """

    def __init__(self) -> None:
        self._lxtra1 = False
        self._lxtra2 = False

    def extras_startup(self, state: dict[str, Any], restart: bool) -> None:
        """Set `lxtra(1:2)` and guard part-3 diffusion toggle (`:56-78`)."""
        self._lxtra1 = False
        self._lxtra2 = bool(restart)
        if int(state["x_integer_ctrl"][0]) == 3:
            _ = float(state["power_he_burn"]) > 1.0e3
            raise NotImplementedError(
                "make_co_wd: extras_startup mutates s% do_element_diffusion "
                "(:70-76), which flexmesa-star StarExtras updates do not "
                "expose (#158)"
            )

    def extras_check_model(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        """Check `L_by_category(icc) > 1d4` and part-3 diffusion toggle
        (`:94-128`).
        """
        result: str | tuple[str, str] = "keep_going"
        if float(state["l_by_category_icc"]) > 1.0e4:
            result = (
                "terminate",
                "Carbon has ignited in the interior and will produce an ONe "
                "WD.",
            )
        if int(state["x_integer_ctrl"][0]) == 3:
            _ = float(state["power_he_burn"]) > 1.0e3
            raise NotImplementedError(
                "make_co_wd: extras_check_model mutates "
                "s% do_element_diffusion (:120-126), which flexmesa-star "
                "StarExtras updates do not expose (#158)"
            )
        return result

    def extras_finish_step(self, state: dict[str, Any]) -> str:
        """Scan part-2 envelope cut and require `#153` when triggered
        (`:181-224`).
        """
        if int(state["x_integer_ctrl"][0]) == 2:
            h_env_limit = float(state["x_ctrl"][0])
            if not (self._lxtra1 or self._lxtra2):
                nz = int(state["nz"])
                x_h1 = state["x_h1"]
                m_arr = state["m"]
                h1_bdy = float(state["he_core_boundary_h1_fraction"])
                kbot = 0
                bottom_mass = 0.0
                for k in range(nz - 1, -1, -1):
                    if float(x_h1[k]) > h1_bdy:
                        bottom_mass = float(m_arr[k])
                        kbot = k
                        break
                for k in range(kbot, -1, -1):
                    if (
                        float(m_arr[k])
                        > bottom_mass + h_env_limit * constants.MSUN
                    ):
                        raise NotImplementedError(
                            "make_co_wd: extras_finish_step calls "
                            f"star_remove_surface_at_cell_k at 1-based cell "
                            f"{k + 1} (:207), which flexmesa-star does not "
                            "implement (#153)"
                        )
        return "keep_going"
