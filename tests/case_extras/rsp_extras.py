# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream `star/test_suite/rsp_*` `run_star_extras.f90` implementations
(`#148`).
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

import numpy as np

from flexmesa import _rust, constants, interp_1d

_GYRE_HOOK_RE = re.compile(
    r"other_rsp_linear_analysis\s*=>\s*rsp_set_gyre_linear_analysis"
)
_CROSSING_HOOK_RE = re.compile(
    r"other_rsp_build_model\s*=>\s*rsp_check_2nd_crossing"
)
_RSP_FINISH_RE = re.compile(
    r"s%\s*rsp_num_periods\s*<\s*s%\s*x_integer_ctrl\(1\)"
)
_EIGHT_HIST_COLS_RE = re.compile(r"how_many_extra_history_columns\s*=\s*8\b")


class RspPeriodLimitExtras:
    """Upstream period-limit termination and extra history columns for RSP
    cases (`star/test_suite/rsp_BEP/src/run_star_extras.f90:57-184`,
    `star/test_suite/rsp_BLAP/src/run_star_extras.f90:56-183`,
    `star/test_suite/rsp_Cepheid/src/run_star_extras.f90:57-184`,
    `star/test_suite/rsp_Delta_Scuti/src/run_star_extras.f90:56-184`,
    `star/test_suite/rsp_Type_II_Cepheid/src/run_star_extras.f90:56-184`,
    `star/test_suite/rsp_RR_Lyrae/src/run_star_extras.f90:56-172`, and
    `star/test_suite/rsp_save_and_load_file/src/run_star_extras.f90:57-173`).
    """

    def __init__(self, *, emit_eight_history_columns: bool) -> None:
        self._emit_eight_history_columns = emit_eight_history_columns
        self._need_lina_file = False

    def extras_startup(self, state: dict[str, Any], restart: bool) -> None:
        """Record whether an initial `LINA` summary file should be emitted."""
        x_char = state.get("x_character_ctrl", [])
        out_name = x_char[9].strip() if len(x_char) >= 10 else ""
        self._need_lina_file = (not restart) and bool(out_name)

    def extras_start_step(self, state: dict[str, Any]) -> str:
        """Write `x_character_ctrl(10)` `LINA` summary on the first step."""
        if not self._need_lina_file:
            return "keep_going"
        x_char = state.get("x_character_ctrl", [])
        out_name = x_char[9].strip() if len(x_char) >= 10 else ""
        case_dir = str(state.get("case_dir", "")).strip()
        if out_name and case_dir:
            out_path = pathlib.Path(case_dir) / out_name
            nmodes = int(state.get("rsp_nmodes", 0))
            periods = state.get("rsp_lina_periods", [])
            growths = state.get("rsp_lina_growth_rates", [])
            vals = [
                f"{float(state.get('rsp_mass', 0.0)):16.5e}",
                f"{float(state.get('rsp_l', 0.0)):16.5e}",
                f"{float(state.get('rsp_teff', 0.0)):16.5e}",
            ]
            for idx in range(min(nmodes, len(periods), len(growths))):
                vals.append(f"{float(periods[idx]):16.5e}")
                vals.append(f"{float(growths[idx]):16.5e}")
            out_path.write_text("".join(vals) + "\n", encoding="utf-8")
        self._need_lina_file = False
        return "keep_going"

    def extras_finish_step(self, state: dict[str, Any]) -> str:
        """Terminate once `rsp_num_periods >= x_integer_ctrl(1)`."""
        x_int = state.get("x_integer_ctrl", [])
        target_periods = int(x_int[0]) if x_int else 0
        if target_periods <= 0:
            return "keep_going"
        if int(state.get("rsp_num_periods", 0)) < target_periods:
            return "keep_going"
        return "terminate"

    def extra_history_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Return upstream 8 `RSP` history columns when enabled."""
        if not self._emit_eight_history_columns:
            return []
        return [
            ("num_periods", float(state.get("rsp_num_periods", 0))),
            ("period", float(state.get("rsp_period", 0.0)) / constants.SECDAY),
            ("growth", float(state.get("rsp_grekm_avg_abs", 0.0))),
            ("max_v_div_cs", 0.0),
            ("delta_R", float(state.get("rsp_delta_r", 0.0))),
            ("delta_Teff", 0.0),
            ("delta_logL", float(state.get("rsp_delta_mag", 0.0)) / 2.5),
            ("delta_Mag", 0.0),
        ]


class RspGyreExtras(RspPeriodLimitExtras):
    """Upstream `GYRE` starting-velocity initialization for `RSP`
    (`star/test_suite/rsp_gyre/src/run_star_extras.f90:64-264`).
    """

    def other_rsp_linear_analysis(
        self,
        state: dict[str, Any],
        gyre_str: str,
        case_dir: str,
        _log_dir: str | None,
    ) -> dict[str, Any]:
        """Solve radial `GYRE` modes and set initial `RSP` velocity kick."""
        nz = int(state["nz"])
        staged = pathlib.Path(case_dir or str(state.get("case_dir", ".")))
        (staged / "gyre.data").write_text(gyre_str, encoding="utf-8")

        modes = _rust.star_solve_gyre_mesa_modes_full(
            gyre_str, l=0, omega_min=0.5, omega_max=12.0, n_freq=400
        )
        modes_sorted = sorted(modes, key=lambda row: row[3])[:3]
        if not modes_sorted:
            raise ValueError("GYRE returned no radial modes for RSP init")

        x_ctrl = state.get("x_ctrl", [])
        amix1 = float(x_ctrl[3]) if len(x_ctrl) > 3 else 0.0
        amix2 = float(x_ctrl[4]) if len(x_ctrl) > 4 else 0.0
        if amix1 + amix2 > 1.0:
            raise ValueError("AMIX DO NOT ADD UP RIGHT")
        velkm = float(x_ctrl[5]) if len(x_ctrl) > 5 else 0.0
        amix_f = 1.0 - (amix1 + amix2)

        n_inner = nz - 1
        xi_modes: list[list[float]] = []
        periods_sec: list[float] = []
        for mode in modes_sorted:
            freq_uhz = float(mode[4])
            periods_sec.append(1.0e6 / freq_uhz if freq_uhz > 0.0 else 0.0)
            xi_r = [float(val) for val in mode[6]]
            if len(xi_r) >= n_inner:
                xi_modes.append(xi_r[:n_inner])
            else:
                xi_modes.append(xi_r + [0.0] * (n_inner - len(xi_r)))

        while len(xi_modes) < 3:
            xi_modes.append([0.0] * n_inner)
            periods_sec.append(periods_sec[-1] if periods_sec else 0.0)

        v_surf = (
            amix_f * xi_modes[0][n_inner - 1]
            + amix1 * xi_modes[1][n_inner - 1]
            + amix2 * xi_modes[2][n_inner - 1]
        )
        if v_surf == 0.0:
            raise ValueError("degenerate zero surface GYRE eigenfunction")

        v_star = [0.0] * nz
        for i in range(n_inner):
            k = (n_inner - 1) - i + 1
            if 1 <= k < nz:
                v_star[k] = (
                    1.0e5
                    * velkm
                    / v_surf
                    * (
                        amix_f * xi_modes[0][i]
                        + amix1 * xi_modes[1][i]
                        + amix2 * xi_modes[2][i]
                    )
                )
        if nz >= 2:
            v_star[0] = v_star[1]

        mode_idx = int(state.get("mode_for_setting_periodlin", 0))
        period_sec = periods_sec[min(mode_idx, len(periods_sec) - 1)]
        return {
            "handled": True,
            "v": v_star,
            "v_center": 0.0,
            "rsp_period": period_sec,
        }


class RspCheck2ndCrossingExtras(RspPeriodLimitExtras):
    """Upstream 2nd-crossing Cepheid instability-strip scanner
    (`star/test_suite/rsp_check_2nd_crossing/src/run_star_extras.f90:58-453`).
    """

    def other_rsp_build_model(self, ctx: Any) -> Any:
        """Scan `LINA` growth rates along `x_character_ctrl(1)` and write
        instability-strip edges to `x_character_ctrl(2)`.
        """
        case_dir = pathlib.Path(ctx.case_dir)
        x_ctrl = list(ctx.x_ctrl)
        x_int = list(ctx.x_integer_ctrl)
        x_char = list(ctx.x_character_ctrl)

        delta_teff = float(x_ctrl[0])
        log_t3, log_l3 = float(x_ctrl[5]), float(x_ctrl[6])
        log_t4, log_l4 = float(x_ctrl[7]), float(x_ctrl[8])

        skip_cols = int(x_int[0])
        col_mod = int(x_int[1]) - 1
        col_age = int(x_int[2]) - 1
        col_log_t = int(x_int[3]) - 1
        col_log_l = int(x_int[4]) - 1
        n_cols = max(col_mod, col_age, col_log_t, col_log_l) + 1

        in_path = case_dir / x_char[0].strip()
        rows: list[tuple[int, float, float, float]] = []
        raw_lines = in_path.read_text(encoding="utf-8").splitlines()
        for raw_line in raw_lines[skip_cols:]:
            toks = raw_line.split()
            if len(toks) < n_cols:
                continue
            rows.append(
                (
                    int(float(toks[col_mod])),
                    float(toks[col_age]),
                    _rust.exp10(float(toks[col_log_t])),
                    _rust.exp10(float(toks[col_log_l])),
                )
            )

        def _red_log_t(log_l: float) -> float:
            return log_t4 + (log_l - log_l4) * (log_t3 - log_t4) / (
                log_l3 - log_l4
            )

        finished_1st = False
        in_2nd = False
        have_first = False
        just_failed = False
        num_beyond_blue = 0
        prev_teff = 1.0e99
        sampled: list[tuple[int, float, float, float, float, float]] = []

        for modnum, age, teff, lum in rows:
            log_l = _rust.safe_log10(lum)
            if not finished_1st:
                finished_1st = teff < _rust.exp10(_red_log_t(log_l))
                prev_teff = teff
                continue
            if teff < prev_teff:
                prev_teff = teff
                continue
            min_t = _rust.exp10(_red_log_t(log_l))
            if teff < min_t - 4.0 * delta_teff:
                prev_teff = teff
                continue
            if have_first and teff - delta_teff < prev_teff:
                continue
            try:
                periods_days, growths = ctx.eval_lina(teff, lum, 1)
            except (RuntimeError, ValueError):
                if num_beyond_blue > 0 or just_failed:
                    break
                just_failed = True
                continue
            just_failed = False
            p_days = float(periods_days[0])
            eta = float(growths[0])
            if (not have_first) and eta > 0.0:
                raise ValueError("failed to find red edge")
            have_first = True
            if eta > 0.0:
                in_2nd = True
            if eta <= 0.0 and in_2nd:
                num_beyond_blue += 1
            sampled.append((modnum, p_days, eta, teff, lum, age))
            if num_beyond_blue == 2 or len(sampled) >= 200:
                break
            prev_teff = teff

        n_pts = len(sampled)
        if n_pts < 3 or sampled[0][2] >= 0.0 or num_beyond_blue < 1:
            raise ValueError(
                f"insufficient strip coverage (n={n_pts}, "
                f"beyond_blue={num_beyond_blue})"
            )

        growth = np.array([row[2] for row in sampled], dtype=np.float64)
        temp = np.array([row[3] for row in sampled], dtype=np.float64)
        lum_arr = np.array([row[4] for row in sampled], dtype=np.float64)
        zero_q = np.array([0.0], dtype=np.float64)

        teff_red: float | None = None
        if growth[1] < 0.0:
            for i in range(3, n_pts):
                if (
                    growth[i] > 0.0
                    and growth[i - 1] > 0.0
                    and growth[i - 2] < 0.0
                    and growth[i - 3] < 0.0
                ):
                    teff_red = float(
                        interp_1d.interpolate_vector_pm(
                            growth[i - 3 : i + 1], temp[i - 3 : i + 1], zero_q
                        )[0]
                    )
                    break
        elif growth[2] > 0.0:
            teff_red = float(
                interp_1d.interpolate_vector_pm(growth[:3], temp[:3], zero_q)[0]
            )
        if teff_red is None:
            raise ValueError("failed to locate Teff_red_edge")

        teff_blue: float | None = None
        if num_beyond_blue == 2 and n_pts >= 4:
            for i in range(n_pts - 4, -1, -1):
                if (
                    growth[i] > 0.0
                    and growth[i + 1] > 0.0
                    and growth[i + 2] < 0.0
                    and growth[i + 3] < 0.0
                ):
                    teff_blue = float(
                        interp_1d.interpolate_vector_pm(
                            growth[i : i + 4], temp[i : i + 4], zero_q
                        )[0]
                    )
                    break
        elif (
            num_beyond_blue == 1
            and n_pts > 3
            and growth[n_pts - 1] < 0.0
            and growth[n_pts - 2] > 0.0
            and growth[n_pts - 3] > 0.0
        ):
            teff_blue = float(
                interp_1d.interpolate_vector_pm(
                    growth[n_pts - 3 : n_pts], temp[n_pts - 3 : n_pts], zero_q
                )[0]
            )
        if teff_blue is None:
            raise ValueError("failed to locate Teff_blue_edge")

        offsets: list[tuple[int, float, float]] = []
        offset = 0.0
        while True:
            x_val = max(teff_blue - offset, teff_red)
            y_val = float(
                interp_1d.interpolate_vector_pm(
                    temp, lum_arr, np.array([x_val], dtype=np.float64)
                )[0]
            )
            offsets.append((int(teff_blue - x_val), x_val, y_val))
            if x_val == teff_red:
                break
            offset += 100.0

        out_name = x_char[1].strip() if len(x_char) > 1 else ""
        if out_name:
            out_lines = [
                f"{'RSP_mass':>24}{float(ctx.rsp_mass):26.16E}",
                f"{'RSP_X':>24}{float(ctx.rsp_x):26.16E}",
                f"{'RSP_Z':>24}{float(ctx.rsp_z):26.16E}",
                f"{'RSP_alfam':>24}{float(ctx.rsp_alfam):26.16E}",
                f"{'RSP_alfap':>24}{float(ctx.rsp_alfap):26.16E}",
                f"{'RSP_alfat':>24}{float(ctx.rsp_alfat):26.16E}",
                f"{'RSP_gammar':>24}{float(ctx.rsp_gammar):26.16E}",
                "",
                "".join(
                    f"{col:>20}"
                    for col in (
                        "model_number",
                        "period(d)",
                        "growth",
                        "Teff",
                        "L",
                        "star_age",
                    )
                ),
            ]
            for modnum, p_days, eta, teff, lum, age in sampled:
                out_lines.append(
                    f"{modnum:20d}{p_days:20.10E}{eta:20.10E}"
                    f"{teff:20.10E}{lum:20.10E}{age:20.10E}"
                )
            out_lines.extend(
                [
                    "",
                    f"{'Teff_red_edge':>20}{teff_red:20.10f}",
                    f"{'Teff_blue_edge':>20}{teff_blue:20.10f}",
                    "",
                    f"{'offset':>10}{'Teff':>20}{'L':>20}",
                ]
            )
            for off_k, t_k, l_val in offsets:
                out_lines.append(f"{off_k:10d}{t_k:20.10f}{l_val:20.10f}")
            out_path = case_dir / out_name
            out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        return ("terminate", "other_rsp_build_model")


def match_rsp_extras(norm_star_text: str) -> RspPeriodLimitExtras | None:
    """Match an `RSP` extras implementation strictly from Fortran text."""
    eight_cols = bool(_EIGHT_HIST_COLS_RE.search(norm_star_text))
    if _CROSSING_HOOK_RE.search(norm_star_text):
        return RspCheck2ndCrossingExtras(emit_eight_history_columns=eight_cols)
    if _GYRE_HOOK_RE.search(norm_star_text):
        return RspGyreExtras(emit_eight_history_columns=eight_cols)
    if _RSP_FINISH_RE.search(norm_star_text):
        return RspPeriodLimitExtras(emit_eight_history_columns=eight_cols)
    return None
