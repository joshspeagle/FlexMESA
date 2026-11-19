# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Upstream massive-star `alpha_mlt_routine`, core-collapse infall, split-burn,
and pre-CCSN envelope-stripping `StarExtras` ports (`#231`).

Covers:
- `star/test_suite/12M_pre_ms_to_core_collapse/src/run_star_extras.f90:44-221`
- `star/test_suite/20M_pre_ms_to_core_collapse/src/run_star_extras.f90:44-221`
- `star/test_suite/zams_to_cc_80/src/run_star_extras.f90:44-221`
- `star/test_suite/split_burn_big_net/src/run_star_extras.f90:47-217`
- `star/test_suite/make_pre_ccsn_13bvn/src/run_star_extras.f90:37-280`
"""

from __future__ import annotations

from typing import Any

from flexmesa import _rust
from tests.case_extras import _helpers


def compute_alpha_mlt_vector(state: dict[str, Any]) -> list[float] | None:
    """Evaluate per-cell `s% alpha_mlt(1:nz)` from `x_ctrl(21:23)` and `xa_h1`
    (`star/test_suite/12M_pre_ms_to_core_collapse/src/run_star_extras.f90:63-93`).

    Args:
        state: Star snapshot mapping with `x_ctrl` `(>=23,)`, `net_iso_h1`,
            `nz`, and `xa_h1` `(nz,)` mass fractions (dimensionless).

    Returns:
        Per-cell `alpha_mlt` list of length `nz` (dimensionless), or `None`
        when `alpha_H <= 0`, `alpha_other <= 0`, or `h1 <= 0`.
    """
    x_ctrl = state["x_ctrl"]
    alpha_h = float(x_ctrl[20])
    alpha_other = float(x_ctrl[21])
    h_limit = float(x_ctrl[22])
    h1 = int(state["net_iso_h1"])
    if alpha_h <= 0.0 or alpha_other <= 0.0 or h1 <= 0:
        return None
    nz = int(state["nz"])
    xa_h1 = state["xa_h1"]
    return [
        alpha_h if float(xa_h1[k]) >= h_limit else alpha_other
        for k in range(nz)
    ]


def require_per_cell_alpha_mlt(
    state: dict[str, Any], case_label: str, line_span: str
) -> float | None:
    """Check `alpha_mlt_routine` guards and fail on per-cell assignment
    (`#155`, `#231`).

    Args:
        state: Star snapshot mapping with `x_ctrl` `(>=23,)` and `net_iso_h1`.
        case_label: Upstream test-suite case label for diagnostics.
        line_span: Upstream Fortran line span string (`:<lo>-<hi>`).

    Returns:
        `None` when `alpha_H <= 0`, `alpha_other <= 0`, or `net_iso_h1 <= 0`.

    Raises:
        NotImplementedError: When `alpha_mlt_routine` is active, because
            `StarExtras.other_alpha_mlt` accepts only a scalar `alpha` rather
            than a per-cell `(nz,)` vector (`#155`, `#231`).
    """
    x_ctrl = state["x_ctrl"]
    alpha_h = float(x_ctrl[20])
    alpha_other = float(x_ctrl[21])
    _ = float(x_ctrl[22])
    h1 = int(state["net_iso_h1"])
    if alpha_h <= 0.0 or alpha_other <= 0.0 or h1 <= 0:
        return None
    raise NotImplementedError(
        f"{case_label}: alpha_mlt_routine ({line_span}) assigns per-cell "
        "s% alpha_mlt(1:nz), whereas flexmesa-star StarExtras.other_alpha_mlt "
        "accepts only a scalar mixing_length_alpha (#155, #231)"
    )


class _CoreCollapseFeInfallBase:
    """Shared `alpha_mlt_routine`, `zbar_div_abar` profile column, and iron-core
    infall post-check for core-collapse test cases (`#231`).
    """

    _CASE_LABEL: str = ""
    _CC_PART: int = 0

    def __init__(self) -> None:
        self.testhub_extras: dict[str, float] = {}
        self.within_tolerance: bool | None = None

    def other_alpha_mlt(self, state: dict[str, Any]) -> float | None:
        """Guard per-cell `alpha_mlt_routine` (`:63-93`)."""
        return require_per_cell_alpha_mlt(state, self._CASE_LABEL, ":63-93")

    def extras_startup(self, state: dict[str, Any], _restart: bool) -> None:
        """Read `x_logical_ctrl(37)` (`:96-108`)."""
        _ = bool(state["x_logical_ctrl"][36])

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:145-153`)."""
        return "keep_going"

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Return `zbar_div_abar` `(nz,)` profile column (`:179-207`)."""
        return _helpers.zbar_div_abar_profile_column(state)

    def extras_finish_step(self, state: dict[str, Any]) -> str:
        """Read `x_logical_ctrl(37)` and return `keep_going` (`:210-221`)."""
        _ = bool(state["x_logical_ctrl"][36])
        return "keep_going"

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        r"""Check `fe_core_mass >= 1 M_\odot` and `fe_core_infall` at final
        stage (`:111-141`).
        """
        _ = int(state["nz"])
        if int(state["x_integer_ctrl"][4]) == self._CC_PART:
            fe_core_mass = float(state["fe_core_mass"])
            self.testhub_extras = {"fe_core_mass": fe_core_mass}
            if fe_core_mass < 1.0:
                self.within_tolerance = False
            else:
                self.within_tolerance = float(state["fe_core_infall"]) > float(
                    state["fe_core_infall_limit"]
                )
        _ = bool(state["x_logical_ctrl"][36])


class PreMs12MCoreCollapseStarExtras(_CoreCollapseFeInfallBase):
    """Part-6 iron-core infall check and `alpha_mlt_routine` from
    `star/test_suite/12M_pre_ms_to_core_collapse/src/run_star_extras.f90:44-221`.
    """

    _CASE_LABEL = "12M_pre_ms_to_core_collapse"
    _CC_PART = 6


class PreMs20MCoreCollapseStarExtras(_CoreCollapseFeInfallBase):
    """Part-7 iron-core infall check and `alpha_mlt_routine` shared by
    `star/test_suite/20M_pre_ms_to_core_collapse/src/run_star_extras.f90:44-221`
    and `star/test_suite/zams_to_cc_80/src/run_star_extras.f90:44-221`.
    """

    _CASE_LABEL = "20M_pre_ms_to_core_collapse"
    _CC_PART = 7


class SplitBurnBigNetStarExtras:
    """Split-burn solver iteration sum and `alpha_mlt_routine` from
    `star/test_suite/split_burn_big_net/src/run_star_extras.f90:47-217`.
    """

    def __init__(self) -> None:
        self.testhub_extras: dict[str, float] = {}

    def other_alpha_mlt(self, state: dict[str, Any]) -> float | None:
        """Guard per-cell `alpha_mlt_routine` (`:66-96`)."""
        return require_per_cell_alpha_mlt(state, "split_burn_big_net", ":66-96")

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:141-149`)."""
        return "keep_going"

    def extras_finish_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:206-217`)."""
        return "keep_going"

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Return `zbar_div_abar` `(nz,)` profile column (`:175-203`)."""
        return _helpers.zbar_div_abar_profile_column(state)

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Record `total_burn_solver_steps = sum(burn_num_iters(1:nz))` in
        part 1 (`:112-137`).
        """
        if int(state["x_integer_ctrl"][0]) == 1:
            nz = int(state["nz"])
            burn_iters = state["burn_num_iters"]
            total = 0.0
            for k in range(nz):
                total += float(burn_iters[k])
            self.testhub_extras = {"total_burn_solver_steps": total}


class MakePreCcsn13bvnStarExtras:
    """Drag-energy history/profile diagnostics, `alpha_mlt_routine`, and
    envelope-removal `mass_change` schedule from
    `star/test_suite/make_pre_ccsn_13bvn/src/run_star_extras.f90:37-280`.
    """

    def other_alpha_mlt(self, state: dict[str, Any]) -> float | None:
        """Guard per-cell `alpha_mlt_routine` (`:58-88`)."""
        return require_per_cell_alpha_mlt(
            state, "make_pre_ccsn_13bvn", ":58-88"
        )

    def extras_startup(self, _state: dict[str, Any], _restart: bool) -> None:
        """No-op (`:92-101`; `:45-54` leaves `s% other_photo_read` unassigned,
        so `:249-267` is unreachable on both fresh starts and restarts).
        """
        return None

    def extras_start_step(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:212-223`)."""
        return "keep_going"

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:117-125`)."""
        return "keep_going"

    def extra_history_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        r"""Evaluate `log_total_drag_energy` ($\log_{10}(\mathrm{erg\,s^{-1}})$)
        (`:128-164`).
        """
        val = 0.0
        if bool(state["v_flag"]) and float(state["drag_coefficient"]) > 0.0:
            nz = int(state["nz"])
            q_arr = state["q"]
            min_q = float(state["min_q_for_drag"])
            fdotv = state["fdotv_drag_energy"]
            dm_arr = state["dm"]
            for k in range(nz):
                if float(q_arr[k]) >= min_q:
                    val = float(fdotv[k]) * float(dm_arr[k]) + val
            val = _rust.log10(max(1.0e-40, abs(val)))
        return [("log_total_drag_energy", val)]

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Return `zbar_div_abar`, `dvdt_drag`, and `FdotV_drag_energy` `(nz,)`
        columns (`:167-209`).
        """
        nz = int(state["nz"])
        zbar = state["zbar"]
        abar = state["abar"]
        z_div_a = [
            _helpers.fdiv(float(zbar[k]), float(abar[k])) for k in range(nz)
        ]
        if bool(state["v_flag"]) and float(state["drag_coefficient"]) > 0.0:
            dvdt = [float(state["dvdt_drag"][k]) for k in range(nz)]
            fdotv = [float(state["fdotv_drag_energy"][k]) for k in range(nz)]
        else:
            dvdt = [0.0] * nz
            fdotv = [0.0] * nz
        return [
            ("zbar_div_abar", z_div_a),
            ("dvdt_drag", dvdt),
            ("FdotV_drag_energy", fdotv),
        ]

    def compute_target_mass_change(self, state: dict[str, Any]) -> float | None:
        r"""Evaluate target `s% mass_change` in $M_\odot\,\mathrm{yr^{-1}}$
        (`:237-246`), or `None` when upstream leaves `s% mass_change` unchanged.
        """
        env_mass_check = float(state["star_mass"]) - float(
            state["he_core_mass"]
        )
        if bool(state["x_logical_ctrl"][0]):
            if env_mass_check >= 1.0e-1:
                return -1.0e-3
            if env_mass_check <= 1.0e-1 and env_mass_check >= float(
                state["star_species_mass_min_limit"]
            ):
                return -1.0e-4
            return None
        return 0.0

    def extras_finish_step(self, state: dict[str, Any]) -> str:
        """Evaluate `mass_change` schedule and require `#159` when mutated
        (`:227-247`).
        """
        target = self.compute_target_mass_change(state)
        if target is None:
            return "keep_going"
        raise NotImplementedError(
            "make_pre_ccsn_13bvn: extras_finish_step sets s% mass_change = "
            f"{target!r} (:238-246), which flexmesa-star StarExtras updates "
            "do not expose (#159, #231)"
        )
