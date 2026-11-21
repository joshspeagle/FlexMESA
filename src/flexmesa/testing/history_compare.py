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
from typing import Any

import numpy as np

from flexmesa import _rust
from flexmesa.testing import case_manifest, golden

COLUMN_ALIASES: dict[str, str] = {
    "log_cntr_T": "log_center_T",
    "log_cntr_Rho": "log_center_Rho",
}
SKIP_ALONG_TRACK: frozenset[str] = frozenset(
    {"model_number", "num_retries", "num_iters", "log_dt", "stage"}
)
DISCRETE_TRACK_COLUMNS: frozenset[str] = frozenset({"num_zones"})
SENTINEL_ABS_THRESHOLD: float = 0.99e99


@dataclasses.dataclass(frozen=True)
class HistoryRunStats:
    """Step, retry, and Newton iteration statistics extracted from a history.

    Attributes:
        steps: Evolutionary steps spanned by the final stage, or ``None`` if
            ``model_number`` is absent.
        retries: Cumulative ``num_retries`` at the final logged step, or
            ``None`` if ``num_retries`` is absent.
        iters_sum: Sum of per-step ``num_iters`` across all logged rows, or
            ``None`` if ``num_iters`` is absent. Equals cumulative Newton
            iterations across the stage when ``iters_rows == steps``.
        mean_iters: Mean per-step ``num_iters`` across logged rows, or
            ``None`` if ``num_iters`` is absent.
        iters_rows: Number of logged rows averaged in ``mean_iters`` and
            summed in ``iters_sum`` (``0`` when ``num_iters`` is absent).
    """

    steps: int | None = None
    retries: int | None = None
    iters_sum: int | None = None
    mean_iters: float | None = None
    iters_rows: int = 0


def extract_history_run_stats(
    tbl: HistoryTable,
    suite: str,
) -> HistoryRunStats:
    """Extract final-stage step, retry, and iteration counts from ``tbl``.

    Args:
        tbl: Loaded ``HistoryTable`` for a single evolutionary stage.
        suite: Upstream suite name (``"star"``, ``"binary"``, or ``"astero"``).

    Returns:
        ``HistoryRunStats`` populated from ``model_number``, ``num_retries``,
        and ``num_iters`` where present.
    """
    steps: int | None = None
    if "model_number" in tbl.columns and tbl.num_rows > 0:
        m_arr = tbl.columns["model_number"]
        if math.isfinite(float(m_arr[0])) and math.isfinite(float(m_arr[-1])):
            m0 = round(float(m_arr[0]))
            m1 = round(float(m_arr[-1]))
            if suite == "binary":
                steps = m1
            elif tbl.num_rows == 1:
                steps = 1
            elif m0 <= 1:
                steps = m1
            elif (
                tbl.num_rows >= 3
                and round(float(m_arr[1])) - m0 == m0
                and round(float(m_arr[2])) - round(float(m_arr[1])) == m0
            ):
                # Run started at model_number = 0 with history_interval == m0.
                steps = m1
            else:
                steps = m1 - m0 + 1

    retries: int | None = None
    if "num_retries" in tbl.columns and tbl.num_rows > 0:
        r_end = float(tbl.columns["num_retries"][-1])
        if math.isfinite(r_end) and abs(r_end) < SENTINEL_ABS_THRESHOLD:
            retries = round(r_end)

    iters_sum: int | None = None
    mean_iters: float | None = None
    iters_rows = 0
    if "num_iters" in tbl.columns and tbl.num_rows > 0:
        i_arr = tbl.columns["num_iters"]
        valid = i_arr[
            np.isfinite(i_arr) & (np.abs(i_arr) < SENTINEL_ABS_THRESHOLD)
        ]
        if valid.size > 0:
            iters_sum = round(float(np.sum(valid)))
            mean_iters = round(float(np.mean(valid)), 2)
            iters_rows = int(valid.size)

    return HistoryRunStats(
        steps=steps,
        retries=retries,
        iters_sum=iters_sum,
        mean_iters=mean_iters,
        iters_rows=iters_rows,
    )


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
            samples are finite, non-sentinel, and within the configured
            absolute (``<col>``) or relative (``rel_<col>``) tolerance.
        max_rel_diff: Maximum relative difference ``max |run - ref| / |ref|``
            on the interior common grid when ``|ref| > 0`` everywhere on the
            grid, else ``None``.
    """

    column: str
    max_abs_diff: float
    median_abs_diff: float
    x_at_max: float
    tolerance: float | None
    within_tolerance: bool
    max_rel_diff: float | None = None


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
    ref_span: float = 0.0


def _parse_float_fast(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        return golden.parse_fortran_float(token)


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
    if not fpath.is_file():
        raise FileNotFoundError(f"missing history file: {fpath}")
    with fpath.open(encoding="utf-8") as fh:
        first_line = fh.readline()
    if not first_line:
        raise ValueError(f"{fpath}: empty history file")

    provenance = ""
    mesa_ver = ""
    if first_line.startswith("#") or "\t" in first_line:
        raw_lines = fpath.read_text(encoding="utf-8").splitlines()
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
        if not rows:
            raise ValueError(f"{fpath}: history file contains zero data rows")
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ValueError(
                f"{fpath}: column mismatch ({matrix.shape} vs {len(names)})"
            )
    else:
        parsed = _rust.mesa_read_log_file(str(fpath))
        if int(parsed["num_rows"]) == 0:
            raise ValueError(f"{fpath}: history file contains zero data rows")
        mesa_ver = str(parsed["header"].get("version_number", "")).strip()
        names = [str(n) for n in parsed["column_names"]]
        col_vecs = [
            np.asarray(arr, dtype=np.float64) for arr in parsed["column_values"]
        ]
        matrix = np.column_stack(col_vecs)

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


def _resolve_track_tolerance(
    col: str,
    track_tolerances: dict[str, float],
) -> tuple[float | None, bool, str]:
    """Resolve absolute ``<col>`` or relative ``rel_<col>`` track tolerance."""
    if col in track_tolerances:
        return track_tolerances[col], False, col
    rel_key = f"rel_{col}"
    if rel_key in track_tolerances:
        return track_tolerances[rel_key], True, rel_key
    return None, False, col


def _compare_scalar_track_tol(
    tol_key: str,
    tol: float,
    run_cols: dict[str, np.ndarray],
    ref_cols: dict[str, np.ndarray],
) -> bool:
    """Evaluate a single-point (``ref_span == 0``) track tolerance."""
    is_rel = tol_key.startswith("rel_")
    raw_col = tol_key[4:] if is_rel else tol_key
    if raw_col not in run_cols or raw_col not in ref_cols:
        return False
    r_val = float(run_cols[raw_col][-1])
    ref_val = float(ref_cols[raw_col][-1])
    if _is_bad_value(r_val) or _is_bad_value(ref_val):
        return False
    diff = abs(r_val - ref_val)
    if is_rel:
        if abs(ref_val) > 0.0:
            diff /= abs(ref_val)
        else:
            diff = 0.0 if r_val == 0.0 else float("inf")
    return math.isfinite(diff) and diff <= tol


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
    if "num_zones" in shared_cols:
        ref_nz_end = float(ref.columns["num_zones"][-1])
        run_nz_end = float(run.columns["num_zones"][-1])
        if abs(ref_nz_end) > 0.0:
            final_diffs["rel_num_zones"] = (run_nz_end - ref_nz_end) / abs(
                ref_nz_end
            )
        else:
            final_diffs["rel_num_zones"] = (
                0.0 if run_nz_end == 0.0 else float("inf")
            )

    failed_final: list[str] = []
    for col, tol in spec.final_tolerances.items():
        if col not in final_diffs:
            failed_final.append(col)
            continue
        raw_col = (
            xcol
            if col == f"rel_{xcol}"
            else (col[4:] if col.startswith("rel_") else col)
        )
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
        ref_overlap_mask = (ref_x >= lo) & (ref_x <= hi)
        ref_node_grid = ref_x[ref_overlap_mask]
        for col in interp_names:
            if col in DISCRETE_TRACK_COLUMNS and ref_node_grid.size >= 2:
                col_grid = ref_node_grid
                ref_interp = ref_cols[col][ref_overlap_mask]
                run_interp = np.interp(col_grid, run_x, run_cols[col])
            else:
                col_grid = grid
                ref_interp = np.interp(col_grid, ref_x, ref_cols[col])
                run_interp = np.interp(col_grid, run_x, run_cols[col])
            abs_d = np.abs(run_interp - ref_interp)
            has_bad = bool(
                not np.all(np.isfinite(abs_d))
                or np.any(np.abs(run_cols[col]) >= SENTINEL_ABS_THRESHOLD)
            )
            pos_ref = bool(
                col_grid.size > 0
                and np.all(np.isfinite(ref_interp))
                and np.all(np.abs(ref_interp) > 0.0)
            )
            max_rel: float | None = None
            if has_bad:
                max_d = float("inf")
                med_d = float("nan")
                x_max = float(col_grid[0]) if col_grid.size > 0 else lo
                if pos_ref:
                    max_rel = float("inf")
            else:
                idx_max = int(np.argmax(abs_d))
                max_d = float(abs_d[idx_max])
                med_d = float(np.median(abs_d))
                x_max = float(col_grid[idx_max])
                if pos_ref:
                    max_rel = float(np.max(abs_d / np.abs(ref_interp)))
            tol_opt, is_rel, tol_key = _resolve_track_tolerance(
                col, spec.track_tolerances
            )
            if tol_opt is None:
                ok = True
            elif is_rel:
                ok = (
                    not has_bad
                    and max_rel is not None
                    and math.isfinite(max_rel)
                    and max_rel <= tol_opt
                )
            else:
                ok = not has_bad and max_d <= tol_opt
            if not ok:
                failed_track.append(tol_key)
            deviations.append(
                TrackDeviation(
                    column=col,
                    max_abs_diff=max_d,
                    median_abs_diff=med_d,
                    x_at_max=x_max,
                    tolerance=tol_opt,
                    within_tolerance=ok,
                    max_rel_diff=max_rel,
                )
            )
        for tol_key in spec.track_tolerances:
            raw_col = tol_key[4:] if tol_key.startswith("rel_") else tol_key
            if raw_col not in run_cols and tol_key not in failed_track:
                failed_track.append(tol_key)
    elif ref_span == 0.0 and span_ok:
        for tol_key, tol in spec.track_tolerances.items():
            if not _compare_scalar_track_tol(
                tol_key, tol, run.columns, ref.columns
            ):
                failed_track.append(tol_key)
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
        for tol_key, tol in spec.track_tolerances.items():
            is_rel = tol_key.startswith("rel_")
            raw_col = tol_key[4:] if is_rel else tol_key
            if raw_col not in ref_bounds or raw_col not in run.columns:
                exceed_mask |= in_range
                continue
            if raw_col in DISCRETE_TRACK_COLUMNS and ref_node_grid.size >= 2:
                ref_lo_d = ref_bounds[raw_col][0][ref_overlap_mask]
                ref_hi_d = ref_bounds[raw_col][1][ref_overlap_mask]
                run_at_ref = np.interp(ref_node_grid, run_x, run_cols[raw_col])
                dist_d = np.maximum(
                    0.0,
                    np.maximum(ref_lo_d - run_at_ref, run_at_ref - ref_hi_d),
                )
                if is_rel:
                    scale_d = np.minimum(np.abs(ref_lo_d), np.abs(ref_hi_d))
                    metric_d = np.where(
                        scale_d > 0.0,
                        dist_d / np.where(scale_d > 0.0, scale_d, 1.0),
                        np.where(dist_d > 0.0, np.inf, 0.0),
                    )
                else:
                    metric_d = dist_d
                bad_ref = (
                    ~np.isfinite(metric_d)
                    | (np.abs(run_at_ref) >= SENTINEL_ABS_THRESHOLD)
                    | (metric_d > tol)
                )
                for bad_x in ref_node_grid[bad_ref]:
                    pos = int(np.searchsorted(raw_x, bad_x, side="left"))
                    if pos < run.num_rows:
                        exceed_mask[pos] = True
                continue
            r_arr = run.columns[raw_col]
            lo_b, hi_b = ref_bounds[raw_col]
            ref_lo = np.interp(raw_x, ref_x, lo_b)
            ref_hi = np.interp(raw_x, ref_x, hi_b)
            dist = np.maximum(0.0, np.maximum(ref_lo - r_arr, r_arr - ref_hi))
            if is_rel:
                scale = np.minimum(np.abs(ref_lo), np.abs(ref_hi))
                metric = np.where(
                    scale > 0.0,
                    dist / np.where(scale > 0.0, scale, 1.0),
                    np.where(dist > 0.0, np.inf, 0.0),
                )
            else:
                metric = dist
            bad_pt = (
                ~np.isfinite(metric)
                | (np.abs(r_arr) >= SENTINEL_ABS_THRESHOLD)
                | (metric > tol)
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
        ref_span=ref_span,
    )


def re_evaluate_comparison(
    comp: HistoryComparison,
    spec: case_manifest.CaseManifestEntry,
) -> HistoryComparison:
    """Re-evaluate ``comp`` against ``spec``'s current tolerances."""
    failed_final: list[str] = []
    for col, tol in spec.final_tolerances.items():
        if col not in comp.final_diffs:
            failed_final.append(col)
            continue
        diff_val = comp.final_diffs[col]
        if (
            not math.isfinite(diff_val)
            or abs(diff_val) >= SENTINEL_ABS_THRESHOLD
            or abs(diff_val) > tol
        ):
            failed_final.append(col)

    has_overlap = (
        comp.overlap_max > comp.overlap_min
        and comp.ref_rows >= 2
        and comp.run_rows >= 2
    )
    xcol = spec.independent_var
    span_tol = spec.final_tolerances.get(f"rel_{xcol}", 0.05)
    if comp.ref_span > 0.0:
        span_ok = (comp.overlap_max - comp.overlap_min) >= max(
            0.0, 1.0 - 2.0 * span_tol
        ) * comp.ref_span
    else:
        x_diff = abs(comp.final_diffs.get(xcol, 0.0))
        span_ok = math.isfinite(x_diff) and x_diff <= span_tol

    new_devs: list[TrackDeviation] = []
    failed_track: list[str] = []
    seen_cols: set[str] = set()
    for d in comp.track_deviations:
        seen_cols.add(d.column)
        tol_opt, is_rel, tol_key = _resolve_track_tolerance(
            d.column, spec.track_tolerances
        )
        bad_d = (
            not math.isfinite(d.max_abs_diff)
            or abs(d.max_abs_diff) >= SENTINEL_ABS_THRESHOLD
        )
        if tol_opt is None:
            ok = True
        elif is_rel:
            ok = (
                not bad_d
                and d.max_rel_diff is not None
                and math.isfinite(d.max_rel_diff)
                and d.max_rel_diff <= tol_opt
            )
        else:
            ok = not bad_d and d.max_abs_diff <= tol_opt
        if not ok:
            failed_track.append(tol_key)
        new_devs.append(
            TrackDeviation(
                column=d.column,
                max_abs_diff=d.max_abs_diff,
                median_abs_diff=d.median_abs_diff,
                x_at_max=d.x_at_max,
                tolerance=tol_opt,
                within_tolerance=ok,
                max_rel_diff=d.max_rel_diff,
            )
        )
    if has_overlap:
        for tol_key in spec.track_tolerances:
            raw_col = tol_key[4:] if tol_key.startswith("rel_") else tol_key
            if raw_col not in seen_cols and tol_key not in failed_track:
                failed_track.append(tol_key)
    elif comp.ref_span == 0.0 and span_ok:
        for tol_key, tol in spec.track_tolerances.items():
            d_val = comp.final_diffs.get(tol_key, float("inf"))
            if (
                not math.isfinite(d_val)
                or abs(d_val) >= SENTINEL_ABS_THRESHOLD
                or abs(d_val) > tol
            ):
                failed_track.append(tol_key)
    else:
        for col in spec.track_tolerances:
            if col not in failed_track:
                failed_track.append(col)
    if not span_ok:
        for col in spec.track_tolerances:
            if col not in failed_track:
                failed_track.append(col)

    return dataclasses.replace(
        comp,
        within_final_tolerances=(len(failed_final) == 0),
        failed_final_columns=tuple(failed_final),
        track_deviations=tuple(new_devs),
        within_track_tolerances=(len(failed_track) == 0 and span_ok),
        failed_track_columns=tuple(failed_track),
    )


def _json_safe_float(val: float | None) -> float | str | None:
    if val is None:
        return None
    if math.isnan(val):
        return "nan"
    if math.isinf(val):
        return "inf" if val > 0.0 else "-inf"
    return val


def _track_dev_to_dict(d: TrackDeviation) -> dict[str, Any]:
    out: dict[str, Any] = {
        "column": d.column,
        "max_abs_diff": _json_safe_float(d.max_abs_diff),
        "median_abs_diff": _json_safe_float(d.median_abs_diff),
        "x_at_max": _json_safe_float(d.x_at_max),
        "tolerance": _json_safe_float(d.tolerance),
        "within_tolerance": d.within_tolerance,
    }
    if d.max_rel_diff is not None:
        out["max_rel_diff"] = _json_safe_float(d.max_rel_diff)
    return out


def comparison_to_dict(comp: HistoryComparison) -> dict[str, Any]:
    """Serialize a ``HistoryComparison`` into a strict JSON-safe dictionary."""
    return {
        "independent_var": comp.independent_var,
        "ref_rows": comp.ref_rows,
        "run_rows": comp.run_rows,
        "overlap_min": _json_safe_float(comp.overlap_min),
        "overlap_max": _json_safe_float(comp.overlap_max),
        "ref_span": _json_safe_float(comp.ref_span),
        "final_diffs": {
            k: _json_safe_float(v) for k, v in comp.final_diffs.items()
        },
        "within_final_tolerances": comp.within_final_tolerances,
        "failed_final_columns": list(comp.failed_final_columns),
        "track_deviations": [
            _track_dev_to_dict(d) for d in comp.track_deviations
        ],
        "within_track_tolerances": comp.within_track_tolerances,
        "failed_track_columns": list(comp.failed_track_columns),
        "first_diverging_step": comp.first_diverging_step,
        "first_diverging_x": _json_safe_float(comp.first_diverging_x),
    }


def _restore_json_float(val: Any) -> float:
    if isinstance(val, str):
        low = val.strip().lower()
        if low == "nan":
            return float("nan")
        if low in ("inf", "+inf", "infinity"):
            return float("inf")
        if low in ("-inf", "-infinity"):
            return float("-inf")
    fval = float(val)
    if abs(fval) >= SENTINEL_ABS_THRESHOLD:
        return float("inf") if fval > 0.0 else float("-inf")
    return fval


def _restore_json_opt_float(val: Any) -> float | None:
    if val is None:
        return None
    return _restore_json_float(val)


def comparison_from_dict(raw: dict[str, Any]) -> HistoryComparison:
    """Deserialize a ``HistoryComparison`` from ``comparison_to_dict``."""
    devs = tuple(
        TrackDeviation(
            column=str(d["column"]),
            max_abs_diff=_restore_json_float(d["max_abs_diff"]),
            median_abs_diff=_restore_json_float(d["median_abs_diff"]),
            x_at_max=_restore_json_float(d["x_at_max"]),
            tolerance=_restore_json_opt_float(d.get("tolerance")),
            within_tolerance=bool(d["within_tolerance"]),
            max_rel_diff=_restore_json_opt_float(d.get("max_rel_diff")),
        )
        for d in raw.get("track_deviations", ())
    )
    f_step_raw = raw.get("first_diverging_step")
    ov_min = _restore_json_float(raw["overlap_min"])
    ov_max = _restore_json_float(raw["overlap_max"])
    if "ref_span" in raw and raw["ref_span"] is not None:
        r_span = _restore_json_float(raw["ref_span"])
    else:
        r_span = (
            max(0.0, ov_max - ov_min) if math.isfinite(ov_max - ov_min) else 0.0
        )
    return HistoryComparison(
        independent_var=str(raw["independent_var"]),
        ref_rows=int(raw["ref_rows"]),
        run_rows=int(raw["run_rows"]),
        overlap_min=ov_min,
        overlap_max=ov_max,
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
        ref_span=r_span,
    )
