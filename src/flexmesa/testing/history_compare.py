# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream MESA trajectory loader and ``float64`` history track comparator.

Loads raw 6-line MESA ``LOGS/history.data`` or ``binary_history.data`` files as
well as thinned tab-separated reference histories, scrubs restart backups,
interpolates onto a shared monotonic grid uniting uniform samples and interior
track nodes, evaluates final-state and along-track tolerances with strict
finiteness and sentinel checks, and identifies the first step at which a
trajectory exceeds tolerance.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import shlex
from typing import Any

import numpy as np

from flexmesa.testing import case_manifest, golden

COLUMN_ALIASES: dict[str, str] = {
    "log_cntr_T": "log_center_T",
    "log_cntr_Rho": "log_center_Rho",
}
SKIP_ALONG_TRACK: frozenset[str] = frozenset(
    {"model_number", "num_zones", "num_retries", "num_iters", "log_dt", "stage"}
)
SENTINEL_ABS_THRESHOLD: float = 0.99e99


@dataclasses.dataclass(frozen=True)
class HistoryTable:
    """Columnar ``float64`` representation of a stellar or binary history file.

    Attributes:
        columns: Mapping from canonical column name to a 1D ``float64`` array
            of shape ``(n_models,)`` in MESA output units (ages in ``yr``,
            masses in ``M_sun``, logarithmic quantities in ``dex``).
        num_rows: Number of stored evolutionary rows ``n_models``.
        provenance_comment: Leading ``# ...`` provenance note when loaded from
            a thinned TSV file, or empty string for raw 6-line MESA logs.
        mesa_version_number: ``version_number`` token from a 6-line MESA header
            (such as ``"r26.4.1"``), or empty string for TSV files.
    """

    columns: dict[str, np.ndarray]
    num_rows: int
    provenance_comment: str = ""
    mesa_version_number: str = ""


@dataclasses.dataclass(frozen=True)
class TrackDeviation:
    """Summary of differences along the common independent-variable interval.

    Attributes:
        column: History column name.
        max_abs_diff: Maximum absolute difference ``max |run - ref|`` on the
            interior common grid.
        median_abs_diff: Median absolute difference on the interior common grid.
        x_at_max: Value of the independent coordinate where ``max_abs_diff``
            occurs (in ``yr`` for ``star_age``/``age`` or dimensionless step).
        tolerance: Configured track tolerance if constrained, else ``None``.
        within_tolerance: True when ``tolerance is None`` or all compared
            samples are finite, non-sentinel, and ``max_abs_diff <= tolerance``.
    """

    column: str
    max_abs_diff: float
    median_abs_diff: float
    x_at_max: float
    tolerance: float | None
    within_tolerance: bool


@dataclasses.dataclass(frozen=True)
class HistoryComparison:
    """Quantitative comparison between a reference history and a run history.

    Attributes:
        independent_var: Coordinate column name used for alignment.
        ref_rows: Number of rows in the reference table.
        run_rows: Number of rows in the candidate run table.
        overlap_min: Lower bound of the shared coordinate range.
        overlap_max: Upper bound of the shared coordinate range.
        final_diffs: Mapping from column name (plus ``rel_<xcol>``) to
            ``run[-1] - ref[-1]`` (or signed relative error).
        within_final_tolerances: True when every configured final tolerance
            holds on finite non-sentinel values.
        failed_final_columns: Tuple of column names exceeding final tolerance.
        track_deviations: Tuple of ``TrackDeviation`` records across shared
            continuous columns.
        within_track_tolerances: True when every configured track tolerance
            holds across adequate coordinate-span coverage.
        failed_track_columns: Tuple of column names exceeding track tolerance.
        first_diverging_step: Earliest ``model_number`` (or 1-based row index)
            in ``run`` at which any tracked column exceeds its track tolerance,
            or ``None`` if no column diverges.
        first_diverging_x: Coordinate value at ``first_diverging_step``, or
            ``None``.
    """

    independent_var: str
    ref_rows: int
    run_rows: int
    overlap_min: float
    overlap_max: float
    final_diffs: dict[str, float]
    within_final_tolerances: bool
    failed_final_columns: tuple[str, ...]
    track_deviations: tuple[TrackDeviation, ...]
    within_track_tolerances: bool
    failed_track_columns: tuple[str, ...]
    first_diverging_step: int | None
    first_diverging_x: float | None


def _parse_float_fast(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        return golden.parse_fortran_float(token)


def _extract_mesa_version_number(
    header_names_line: str,
    header_vals_line: str,
) -> str:
    hdr_names = header_names_line.split()
    if "version_number" not in hdr_names:
        return ""
    v_idx = hdr_names.index("version_number")
    try:
        hdr_vals = shlex.split(header_vals_line)
    except ValueError:
        hdr_vals = [
            tok.strip('"').strip("'") for tok in header_vals_line.split()
        ]
    if v_idx < len(hdr_vals):
        return hdr_vals[v_idx].strip('"').strip("'")
    return ""


def load_history_file(path: pathlib.Path | str) -> HistoryTable:
    """Load a MESA 6-line ``history.data`` file or a thinned TSV history file.

    Filters multi-stage TSV logs to the final ``stage`` and scrubs superseded
    ``model_number`` backup rows (where an earlier row has ``model_number >=``
    a subsequent restarted row).

    Args:
        path: Path to a raw MESA 6-line ASCII log or ``#``-prefixed TSV file.

    Returns:
        ``HistoryTable`` containing 1D ``float64`` arrays of shape
        ``(n_models,)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file contains no header or zero data rows.
    """
    fpath = pathlib.Path(path)
    with fpath.open(encoding="utf-8") as fh:
        raw_lines = fh.read().splitlines()

    if not raw_lines:
        raise ValueError(f"{fpath}: empty history file")

    provenance = ""
    mesa_ver = ""
    if raw_lines[0].startswith("#") or "\t" in raw_lines[0]:
        comments = [ln[1:].strip() for ln in raw_lines if ln.startswith("#")]
        provenance = " ".join(comments).strip()
        body = [ln for ln in raw_lines if ln.strip() and not ln.startswith("#")]
        if len(body) < 2:
            raise ValueError(f"{fpath}: TSV history has fewer than 2 lines")
        names = [n.strip() for n in body[0].split("\t")]
        rows = [
            [_parse_float_fast(tok) for tok in ln.split("\t")]
            for ln in body[1:]
        ]
    else:
        if len(raw_lines) < 7:
            raise ValueError(
                f"{fpath}: MESA log has fewer than 7 lines ({len(raw_lines)})"
            )
        mesa_ver = _extract_mesa_version_number(raw_lines[1], raw_lines[2])
        names = raw_lines[5].split()
        rows = [
            [_parse_float_fast(tok) for tok in ln.split()]
            for ln in raw_lines[6:]
            if ln.strip()
        ]

    if not rows:
        raise ValueError(f"{fpath}: history file contains zero data rows")

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError(
            f"{fpath}: column count mismatch ({matrix.shape} vs {len(names)})"
        )

    canon_names = [COLUMN_ALIASES.get(n, n) for n in names]
    if "stage" in canon_names:
        s_idx = canon_names.index("stage")
        s_col = matrix[:, s_idx]
        matrix = matrix[s_col == np.max(s_col)]

    if "model_number" in canon_names and matrix.shape[0] > 1:
        m_idx = canon_names.index("model_number")
        m_col = matrix[:, m_idx]
        if np.all(np.isfinite(m_col)):
            suffix_min = np.minimum.accumulate(m_col[::-1])[::-1]
            keep_rows = np.ones(matrix.shape[0], dtype=bool)
            keep_rows[:-1] = m_col[:-1] < suffix_min[1:]
            matrix = matrix[keep_rows]

    cols: dict[str, np.ndarray] = {}
    for idx, canon in enumerate(canon_names):
        if canon not in cols:
            cols[canon] = np.ascontiguousarray(matrix[:, idx], dtype=np.float64)

    num_rows = int(matrix.shape[0])
    return HistoryTable(
        columns=cols,
        num_rows=num_rows,
        provenance_comment=provenance,
        mesa_version_number=mesa_ver,
    )


def _collapse_nondecreasing(
    x_arr: np.ndarray,
    cols_subset: dict[str, np.ndarray],
    label: str,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    if not np.all(np.isfinite(x_arr)) or np.any(
        np.abs(x_arr) >= SENTINEL_ABS_THRESHOLD
    ):
        raise ValueError(
            f"{label} independent coordinate contains NaN/sentinel"
        )
    if x_arr.size > 1 and np.any(np.diff(x_arr) < 0.0):
        raise ValueError(f"{label} independent coordinate is non-monotonic")

    if x_arr.size <= 1 or bool(np.all(np.diff(x_arr) > 0.0)):
        exact_bounds = {k: (v, v) for k, v in cols_subset.items()}
        return x_arr, cols_subset, exact_bounds

    uniq_x, inv_idx = np.unique(x_arr, return_inverse=True)
    n_u = uniq_x.size
    last_vals: dict[str, np.ndarray] = {}
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    last_pos = np.zeros(n_u, dtype=np.intp)
    last_pos[inv_idx] = np.arange(x_arr.size, dtype=np.intp)
    for k, arr in cols_subset.items():
        last_vals[k] = arr[last_pos]
        mins = np.full(n_u, np.inf, dtype=np.float64)
        maxs = np.full(n_u, -np.inf, dtype=np.float64)
        np.minimum.at(mins, inv_idx, arr)
        np.maximum.at(maxs, inv_idx, arr)
        bounds[k] = (mins, maxs)
    return uniq_x, last_vals, bounds


def _is_bad_value(val: float) -> bool:
    return not math.isfinite(val) or abs(val) >= SENTINEL_ABS_THRESHOLD


def compare_histories(
    ref: HistoryTable,
    run: HistoryTable,
    spec: case_manifest.CaseManifestEntry,
    num_grid: int = 400,
) -> HistoryComparison:
    """Compare a ``FlexMESA`` history table against an upstream MESA reference.

    Args:
        ref: Reference ``HistoryTable`` with arrays of shape ``(n_ref,)``.
        run: Candidate ``HistoryTable`` with arrays of shape ``(n_run,)``.
        spec: ``CaseManifestEntry`` providing ``independent_var`` and
            tolerance dictionaries.
        num_grid: Number of uniform interior grid samples combined with interior
            sample coordinates (default ``400``).

    Returns:
        Populated ``HistoryComparison`` with final differences, track
        deviations, tolerance outcomes, and ``first_diverging_step``.

    Raises:
        KeyError: If ``spec.independent_var`` is absent from ``ref`` or ``run``.
        ValueError: If coordinate arrays contain ``NaN``, ``Inf``, ``-1e99``
            sentinels, or non-monotonic reversals.
    """
    xcol = spec.independent_var
    if xcol not in ref.columns:
        raise KeyError(f"reference history lacks independent_var {xcol!r}")
    if xcol not in run.columns:
        raise KeyError(f"run history lacks independent_var {xcol!r}")

    shared_cols = [
        c for c in run.columns if c in ref.columns and c not in ("stage",)
    ]
    interp_names = [
        c for c in shared_cols if c not in SKIP_ALONG_TRACK and c != xcol
    ]

    ref_x, ref_cols, ref_bounds = _collapse_nondecreasing(
        ref.columns[xcol],
        {c: ref.columns[c] for c in interp_names},
        "reference",
    )
    run_x, run_cols, _ = _collapse_nondecreasing(
        run.columns[xcol],
        {c: run.columns[c] for c in interp_names},
        "run",
    )

    final_diffs: dict[str, float] = {}
    ref_x_end = float(ref.columns[xcol][-1])
    run_x_end = float(run.columns[xcol][-1])
    final_diffs[xcol] = run_x_end - ref_x_end
    if abs(ref_x_end) > 0.0:
        final_diffs[f"rel_{xcol}"] = (run_x_end - ref_x_end) / abs(ref_x_end)
    else:
        final_diffs[f"rel_{xcol}"] = 0.0 if run_x_end == 0.0 else float("inf")

    for col in shared_cols:
        if col in ("model_number", xcol):
            continue
        final_diffs[col] = float(run.columns[col][-1] - ref.columns[col][-1])

    failed_final: list[str] = []
    for col, tol in spec.final_tolerances.items():
        if col not in final_diffs:
            failed_final.append(col)
            continue
        raw_col = xcol if col == f"rel_{xcol}" else col
        run_end = (
            float(run.columns[raw_col][-1]) if raw_col in run.columns else 0.0
        )
        diff_val = final_diffs[col]
        if (
            not math.isfinite(diff_val)
            or _is_bad_value(run_end)
            or abs(diff_val) > tol
        ):
            failed_final.append(col)

    lo = float(max(ref_x[0], run_x[0]))
    hi = float(min(ref_x[-1], run_x[-1]))
    ref_span = float(ref_x[-1] - ref_x[0])
    span_tol = spec.final_tolerances.get(f"rel_{xcol}", 0.05)
    span_ok = (
        (hi - lo) >= max(0.0, 1.0 - 2.0 * span_tol) * ref_span
        if ref_span > 0.0
        else abs(float(run_x[-1] - ref_x[0])) <= span_tol
    )

    deviations: list[TrackDeviation] = []
    failed_track: list[str] = []
    has_overlap = hi > lo and ref_x.size >= 2 and run_x.size >= 2

    if has_overlap:
        lin_grid = np.linspace(lo, hi, num_grid, dtype=np.float64)[1:-1]
        ref_in = ref_x[(ref_x > lo) & (ref_x < hi)]
        run_in = run_x[(run_x > lo) & (run_x < hi)]
        grid = np.unique(np.concatenate([lin_grid, ref_in, run_in]))
        for col in interp_names:
            ref_interp = np.interp(grid, ref_x, ref_cols[col])
            run_interp = np.interp(grid, run_x, run_cols[col])
            abs_d = np.abs(run_interp - ref_interp)
            has_bad = bool(
                not np.all(np.isfinite(abs_d))
                or np.any(np.abs(run_cols[col]) >= SENTINEL_ABS_THRESHOLD)
            )
            if has_bad:
                max_d = float("inf")
                med_d = float("nan")
                x_max = float(grid[0]) if grid.size > 0 else lo
            else:
                idx_max = int(np.argmax(abs_d))
                max_d = float(abs_d[idx_max])
                med_d = float(np.median(abs_d))
                x_max = float(grid[idx_max])
            tol_opt = spec.track_tolerances.get(col)
            ok = tol_opt is None or (not has_bad and max_d <= tol_opt)
            if not ok:
                failed_track.append(col)
            deviations.append(
                TrackDeviation(
                    column=col,
                    max_abs_diff=max_d,
                    median_abs_diff=med_d,
                    x_at_max=x_max,
                    tolerance=tol_opt,
                    within_tolerance=ok,
                )
            )
        for col in spec.track_tolerances:
            if col not in run_cols and col not in failed_track:
                failed_track.append(col)
    elif ref_span == 0.0 and span_ok:
        for col, tol in spec.track_tolerances.items():
            if col not in run.columns or col not in ref.columns:
                failed_track.append(col)
                continue
            r_val = float(run.columns[col][-1])
            d_val = abs(r_val - float(ref.columns[col][-1]))
            if not math.isfinite(d_val) or _is_bad_value(r_val) or d_val > tol:
                failed_track.append(col)
    else:
        failed_track.extend(spec.track_tolerances.keys())

    if not span_ok:
        for col in spec.track_tolerances:
            if col not in failed_track:
                failed_track.append(col)

    first_step: int | None = None
    first_x: float | None = None
    if has_overlap and spec.track_tolerances:
        raw_x = run.columns[xcol]
        in_range = (raw_x >= lo) & (raw_x <= hi)
        exceed_mask = np.zeros(run.num_rows, dtype=bool)
        for col, tol in spec.track_tolerances.items():
            if col not in ref_bounds or col not in run.columns:
                exceed_mask |= in_range
                continue
            r_arr = run.columns[col]
            lo_b, hi_b = ref_bounds[col]
            ref_lo = np.interp(raw_x, ref_x, lo_b)
            ref_hi = np.interp(raw_x, ref_x, hi_b)
            dist = np.maximum(0.0, np.maximum(ref_lo - r_arr, r_arr - ref_hi))
            bad_pt = (
                ~np.isfinite(dist)
                | (np.abs(r_arr) >= SENTINEL_ABS_THRESHOLD)
                | (dist > tol)
            )
            exceed_mask |= in_range & bad_pt
        hit_indices = np.flatnonzero(exceed_mask)
        if hit_indices.size > 0:
            idx0 = int(hit_indices[0])
            first_x = float(raw_x[idx0])
            if "model_number" in run.columns:
                first_step = round(float(run.columns["model_number"][idx0]))
            else:
                first_step = idx0 + 1

    return HistoryComparison(
        independent_var=xcol,
        ref_rows=ref.num_rows,
        run_rows=run.num_rows,
        overlap_min=lo,
        overlap_max=hi,
        final_diffs=final_diffs,
        within_final_tolerances=len(failed_final) == 0,
        failed_final_columns=tuple(failed_final),
        track_deviations=tuple(deviations),
        within_track_tolerances=(len(failed_track) == 0 and span_ok),
        failed_track_columns=tuple(failed_track),
        first_diverging_step=first_step,
        first_diverging_x=first_x,
    )


def _json_safe_float(val: float | None) -> float | None:
    if val is None:
        return None
    if not math.isfinite(val):
        return 1.0e99 if val > 0.0 else -1.0e99
    return val


def comparison_to_dict(comp: HistoryComparison) -> dict[str, Any]:
    """Serialize a ``HistoryComparison`` into a strict JSON-safe dictionary.

    Args:
        comp: ``HistoryComparison`` instance.

    Returns:
        Dictionary free of ``NaN``/``Inf`` tokens suitable for ``json.dumps``.
    """
    return {
        "independent_var": comp.independent_var,
        "ref_rows": comp.ref_rows,
        "run_rows": comp.run_rows,
        "overlap_min": _json_safe_float(comp.overlap_min),
        "overlap_max": _json_safe_float(comp.overlap_max),
        "final_diffs": {
            k: _json_safe_float(v) for k, v in comp.final_diffs.items()
        },
        "within_final_tolerances": comp.within_final_tolerances,
        "failed_final_columns": list(comp.failed_final_columns),
        "track_deviations": [
            {
                "column": d.column,
                "max_abs_diff": _json_safe_float(d.max_abs_diff),
                "median_abs_diff": _json_safe_float(d.median_abs_diff),
                "x_at_max": _json_safe_float(d.x_at_max),
                "tolerance": _json_safe_float(d.tolerance),
                "within_tolerance": d.within_tolerance,
            }
            for d in comp.track_deviations
        ],
        "within_track_tolerances": comp.within_track_tolerances,
        "failed_track_columns": list(comp.failed_track_columns),
        "first_diverging_step": comp.first_diverging_step,
        "first_diverging_x": _json_safe_float(comp.first_diverging_x),
    }


def _restore_json_float(val: Any) -> float:
    fval = float(val)
    if abs(fval) >= SENTINEL_ABS_THRESHOLD:
        return float("inf") if fval > 0.0 else float("-inf")
    return fval


def _restore_json_opt_float(val: Any) -> float | None:
    if val is None:
        return None
    return _restore_json_float(val)


def comparison_from_dict(raw: dict[str, Any]) -> HistoryComparison:
    """Deserialize a ``HistoryComparison`` produced by ``comparison_to_dict``.

    Args:
        raw: JSON dictionary previously emitted by ``comparison_to_dict``.

    Returns:
        Reconstructed ``HistoryComparison`` with sentinel values restored to
        signed infinities.
    """
    devs = tuple(
        TrackDeviation(
            column=str(d["column"]),
            max_abs_diff=_restore_json_float(d["max_abs_diff"]),
            median_abs_diff=_restore_json_float(d["median_abs_diff"]),
            x_at_max=_restore_json_float(d["x_at_max"]),
            tolerance=_restore_json_opt_float(d.get("tolerance")),
            within_tolerance=bool(d["within_tolerance"]),
        )
        for d in raw.get("track_deviations", ())
    )
    f_step_raw = raw.get("first_diverging_step")
    return HistoryComparison(
        independent_var=str(raw["independent_var"]),
        ref_rows=int(raw["ref_rows"]),
        run_rows=int(raw["run_rows"]),
        overlap_min=_restore_json_float(raw["overlap_min"]),
        overlap_max=_restore_json_float(raw["overlap_max"]),
        final_diffs={
            str(k): _restore_json_float(v)
            for k, v in dict(raw.get("final_diffs", {})).items()
        },
        within_final_tolerances=bool(raw["within_final_tolerances"]),
        failed_final_columns=tuple(
            str(c) for c in raw.get("failed_final_columns", ())
        ),
        track_deviations=devs,
        within_track_tolerances=bool(raw["within_track_tolerances"]),
        failed_track_columns=tuple(
            str(c) for c in raw.get("failed_track_columns", ())
        ),
        first_diverging_step=int(f_step_raw)
        if f_step_raw is not None
        else None,
        first_diverging_x=_restore_json_opt_float(raw.get("first_diverging_x")),
    )
