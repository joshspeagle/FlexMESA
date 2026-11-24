# SPDX-License-Identifier: GPL-3.0-or-later
"""Upstream `star/test_suite/rsp_*` `run_star_extras.f90` implementations
(`#148`).
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

import numpy as np

from flexmesa import _rust, constants, interp_1d

_FORTRAN_REAL_RE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([EeDd][+-]?\d+)?")
_FORTRAN_NAME_RE = re.compile(r"[A-Za-z_]\w*")

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
_LINA_FILE_RE = re.compile(
    r"need_to_write_lina_data\s*=\s*len_trim\(s%\s*x_character_ctrl\(10\)\)"
)


def _list_directed_real(tok: str) -> float | None:
    """Parse one token of a Fortran list-directed `read` into a real.

    Returns:
        The value, or `None` when the token cannot be a real, which upstream
        sees as `iostat /= 0`.

    Raises:
        NotImplementedError: If the token uses list-directed syntax this
            reader does not reproduce (separators, repeat counts, quotes,
            `nan`/`inf`, exponents without a letter).
    """
    if _FORTRAN_REAL_RE.fullmatch(tok):
        return float(tok.replace("D", "E").replace("d", "e"))
    if _FORTRAN_NAME_RE.fullmatch(tok) and tok.lower() not in (
        "nan",
        "inf",
        "infinity",
    ):
        return None
    raise NotImplementedError(
        f"rsp_check_2nd_crossing: list-directed token {tok!r} is not handled"
    )


def _fmt_d(val: float, width: int, digits: int) -> str:
    """Format as Fortran `1pd<width>.<digits>` (`D` exponent letter)."""
    return f"{val:{width}.{digits}E}".replace("E", "D")


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

    def __init__(
        self, *, emit_eight_history_columns: bool, write_lina_file: bool
    ) -> None:
        self._emit_eight_history_columns = emit_eight_history_columns
        # Only the sources that carry `need_to_write_LINA_data` write the
        # `x_character_ctrl(10)` file, for example
        # `star/test_suite/rsp_Cepheid/src/run_star_extras.f90:66-70,81-89`;
        # `rsp_gyre`, `rsp_check_2nd_crossing` and `twin_studies` do not.
        self._write_lina_file = write_lina_file
        self._need_lina_file = False

    def extras_startup(self, state: dict[str, Any], restart: bool) -> None:
        """Record whether an initial `LINA` summary file should be emitted."""
        if not self._write_lina_file:
            return
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
    """`rsp_gyre` extras (`star/test_suite/rsp_gyre/src/run_star_extras.f90`).

    Only the period-limit `extras_finish_step` inherited from the base class
    (`star/test_suite/rsp_gyre/src/run_star_extras.inc:16-47`) is ported.
    Upstream's `rsp_set_gyre_linear_analysis`
    (`star/test_suite/rsp_gyre/src/run_star_extras.f90:64-264`) calls GYRE.
    """

    def other_rsp_linear_analysis(
        self,
        _state: dict[str, Any],
        _gyre_str: str,
        _case_dir: str,
        _log_dir: str | None,
    ) -> dict[str, Any]:
        """Fail loudly: the upstream routine needs GYRE."""
        raise NotImplementedError(
            "rsp_gyre: upstream run_star_extras.f90 calls GYRE, which is "
            "not ported (#170)"
        )


class RspCheck2ndCrossingExtras(RspPeriodLimitExtras):
    """Upstream 2nd-crossing Cepheid instability-strip scanner
    (`star/test_suite/rsp_check_2nd_crossing/src/run_star_extras.f90:58-453`).
    """

    def other_rsp_build_model(self, ctx: Any) -> Any:
        """Scan `LINA` growth rates along `x_character_ctrl(1)` and write
        instability-strip edges to `x_character_ctrl(2)`.
        """
        case_dir = pathlib.Path(ctx.case_dir)
        x_int = list(ctx.x_integer_ctrl)
        x_char = list(ctx.x_character_ctrl)

        skip_cols = int(x_int[0])
        col_mod = int(x_int[1]) - 1
        col_age = int(x_int[2]) - 1
        col_log_t = int(x_int[3]) - 1
        col_log_l = int(x_int[4]) - 1
        n_cols = max(col_mod, col_age, col_log_t, col_log_l) + 1

        in_path = case_dir / x_char[0].strip()
        rows: list[tuple[int, float, float, float]] = []
        raw_lines = in_path.read_text(encoding="utf-8").splitlines()
        # :122-131. A list-directed `read` takes its
        # values from as many records as it needs, ignores the rest of the
        # last one, and the loop stops at the first read that fails (a
        # token that is not a real, or end of file).
        line_iter = iter(raw_lines[skip_cols:])
        reading = True
        while reading:
            vals: list[float] = []
            while reading and len(vals) < n_cols:
                raw_line = next(line_iter, None)
                if raw_line is None:
                    reading = False
                    break
                for tok in raw_line.split():
                    if len(vals) == n_cols:
                        break
                    val = _list_directed_real(tok)
                    if val is None:
                        reading = False
                        break
                    vals.append(val)
            if not reading:
                break
            if len(rows) >= 9000:
                # :125 (`max_cnt`)
                raise ValueError("need to increase max_cnt")
            rows.append(
                (
                    int(vals[col_mod]),
                    vals[col_age],
                    _rust.exp10(vals[col_log_t]),
                    _rust.exp10(vals[col_log_l]),
                )
            )

        # :135-155. Upstream opens the output file before the scan and
        # writes as it goes, so the file holds everything written up to
        # whichever exit is taken; `out_lines` is flushed in `finally`.
        out_name = x_char[1].strip() if len(x_char) > 1 else ""
        out_lines = [
            f"{'RSP_mass':>55}{_fmt_d(float(ctx.rsp_mass), 26, 16)}",
            f"{'RSP_X':>55}{_fmt_d(float(ctx.rsp_x), 26, 16)}",
            f"{'RSP_Z':>55}{_fmt_d(float(ctx.rsp_z), 26, 16)}",
            f"{'RSP_alfam':>55}{_fmt_d(float(ctx.rsp_alfam), 26, 16)}",
            f"{'RSP_alfap':>55}{_fmt_d(float(ctx.rsp_alfap), 26, 16)}",
            f"{'RSP_alfat':>55}{_fmt_d(float(ctx.rsp_alfat), 26, 16)}",
            f"{'RSP_gammar':>55}{_fmt_d(float(ctx.rsp_gammar), 26, 16)}",
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
        try:
            return self._scan_and_write(ctx, rows, out_lines)
        finally:
            if out_name:
                (case_dir / out_name).write_text(
                    "\n".join(out_lines) + "\n", encoding="utf-8"
                )

    def _scan_and_write(
        self,
        ctx: Any,
        rows: list[tuple[int, float, float, float]],
        out_lines: list[str],
    ) -> Any:
        """Scan the track and append the report to `out_lines`.

        Port of
        `star/test_suite/rsp_check_2nd_crossing/src/run_star_extras.f90:157-357`.
        """
        x_ctrl = list(ctx.x_ctrl)
        delta_teff = float(x_ctrl[0])
        log_t3, log_l3 = float(x_ctrl[5]), float(x_ctrl[6])
        log_t4, log_l4 = float(x_ctrl[7]), float(x_ctrl[8])

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
            # :169
            log_l = _rust.log10(lum)
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
            # :221-222
            out_lines.append(
                f"{modnum:20d}"
                + "".join(
                    _fmt_d(val, 20, 10) for val in (p_days, eta, teff, lum, age)
                )
            )
            if num_beyond_blue == 2 or len(sampled) == 200:
                break
            prev_teff = teff

        n_pts = len(sampled)
        # :227-233 (`n == 0`)
        if n_pts == 0 or sampled[0][2] >= 0.0 or num_beyond_blue < 1:
            raise ValueError(
                f"insufficient strip coverage (n={n_pts}, "
                f"beyond_blue={num_beyond_blue})"
            )

        growth = np.array([row[2] for row in sampled], dtype=np.float64)
        temp = np.array([row[3] for row in sampled], dtype=np.float64)
        lum_arr = np.array([row[4] for row in sampled], dtype=np.float64)
        zero_q = np.array([0.0], dtype=np.float64)

        # :235-276
        teff_red = -1.0
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
                    out_lines.append("")
                    out_lines.append(f"{'Teff_red_edge':>20}{teff_red:20.10f}")
                    break
        elif growth[2] > 0.0:
            teff_red = float(
                interp_1d.interpolate_vector_pm(growth[:3], temp[:3], zero_q)[0]
            )
            out_lines.append("")
            out_lines.append(f"{'Teff_red_edge':>20}{teff_red:20.10f}")
        if teff_red < 0.0:
            raise ValueError("failed to find red edge")

        # :278-316
        teff_blue = -1.0
        if num_beyond_blue == 2:
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
                    out_lines.append(
                        f"{'Teff_blue_edge':>20}{teff_blue:20.10f}"
                    )
                    break
        elif num_beyond_blue == 1 and n_pts > 3:
            if (
                growth[n_pts - 1] < 0.0
                and growth[n_pts - 2] > 0.0
                and growth[n_pts - 3] > 0.0
            ):
                teff_blue = float(
                    interp_1d.interpolate_vector_pm(
                        growth[n_pts - 3 : n_pts],
                        temp[n_pts - 3 : n_pts],
                        zero_q,
                    )[0]
                )
                out_lines.append(f"{'Teff_blue_edge':>20}{teff_blue:20.10f}")
        out_lines.append("")
        if teff_blue < 0.0:
            # :319-322: upstream prints 'failed to find blue edge' and
            # returns with ierr = 0, so the run is neither stopped nor
            # failed by the hook. Returning None reports the same to the
            # adapter (no model, no termination).
            return None

        # :334-348
        out_lines.append(f"{'offset':>10}{'Teff':>20}{'L':>20}")
        offset = 0.0
        while True:
            x_val = teff_blue - offset
            if x_val < teff_red:
                x_val = teff_red
            y_val = float(
                interp_1d.interpolate_vector_pm(
                    temp, lum_arr, np.array([x_val], dtype=np.float64)
                )[0]
            )
            out_lines.append(
                f"{int(teff_blue - x_val):10d}{x_val:20.10f}{y_val:20.10f}"
            )
            if x_val == teff_red:
                break
            offset += 100.0
        out_lines.append("")
        return "terminate"


def match_rsp_extras(norm_star_text: str) -> RspPeriodLimitExtras | None:
    """Match an `RSP` extras implementation strictly from Fortran text."""
    eight_cols = bool(_EIGHT_HIST_COLS_RE.search(norm_star_text))
    lina_file = bool(_LINA_FILE_RE.search(norm_star_text))
    if _CROSSING_HOOK_RE.search(norm_star_text):
        return RspCheck2ndCrossingExtras(
            emit_eight_history_columns=eight_cols, write_lina_file=lina_file
        )
    if _GYRE_HOOK_RE.search(norm_star_text):
        return RspGyreExtras(
            emit_eight_history_columns=eight_cols, write_lina_file=lina_file
        )
    if _RSP_FINISH_RE.search(norm_star_text):
        return RspPeriodLimitExtras(
            emit_eight_history_columns=eight_cols, write_lina_file=lina_file
        )
    return None
