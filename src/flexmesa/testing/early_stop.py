# SPDX-License-Identifier: GPL-3.0-or-later
"""Stop a harness case early when its partial history has left tolerance.

The history harness judges a finished run by comparing its history with the
stored MESA reference along the independent coordinate (``DECISIONS #27``,
``compare_histories``). A case whose partial history already violates an
along-track tolerance cannot pass, so a fast loop may stop it instead of
waiting for upstream's termination condition (``DECISIONS #68``).

The along-track check of the final comparison evaluates, on a grid that
contains every interior row of the run, the distance between the run and the
reference interpolated to the same coordinate. :func:`find_early_divergence`
evaluates the same distance with the same tolerance and bad-value rules at a
subset of those points: the interior rows of the run that are already
complete (every row except the last collapsed one, whose value can still
change). For a sparse reference or a discrete column, which the final
comparison evaluates only at the reference rows, the points are the reference
rows that those complete run rows span, and only once at least two of them
lie in the run. A point that fails is therefore also checked, and fails, in
the final comparison of any longer run that contains the same rows, so the
check can never stop a case that the final comparison would
pass. Final-value tolerances are only defined at termination and are never
used here.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import tempfile
from collections.abc import Callable

import numpy as np

from flexmesa import _rust
from flexmesa.data import fetch
from flexmesa.testing import (
    _case_verdict,
    case_manifest,
    history_compare,
)

__all__ = [
    "EarlyDivergence",
    "EarlyStopMonitor",
    "RunMonitor",
    "StopRequest",
    "build_monitor",
    "find_early_divergence",
    "read_partial_history",
    "single_stage_star_case",
]

_SENTINEL = history_compare.SENTINEL_ABS_THRESHOLD


@dataclasses.dataclass(frozen=True)
class EarlyDivergence:
    """The first along-track point at which a partial run is out of tolerance.

    Attributes:
        column: Tolerance key that failed (``<col>`` or ``rel_<col>``).
        x_name: Independent coordinate name (such as ``star_age``).
        x: Coordinate value of the failing point (``yr`` for ages).
        model: ``model_number`` of the first run row at or after ``x``, or
            ``None`` when the run has no ``model_number`` column.
        run_value: Run value of the column at the point.
        ref_value: Reference value of the column interpolated to the point.
        tolerance: The configured tolerance that was exceeded.
    """

    column: str
    x_name: str
    x: float
    model: int | None
    run_value: float
    ref_value: float
    tolerance: float

    def message(self) -> str:
        """Return the one-line report of the early stop."""
        where = f"{self.x_name} = {self.x:.6e}"
        if self.model is not None:
            where += f" (model {self.model})"
        return (
            f"early stop: {self.column} left tolerance {self.tolerance:g} at "
            f"{where}; run {self.run_value:.6e}, reference "
            f"{self.ref_value:.6e}"
        )


def _bad_run(values: np.ndarray) -> np.ndarray:
    return ~np.isfinite(values) | (np.abs(values) >= _SENTINEL)


def _exceeds(
    run_vals: np.ndarray,
    ref_vals: np.ndarray,
    tol: float,
    is_rel: bool,
) -> np.ndarray:
    """Return the points at which ``run`` is out of tolerance of ``ref``.

    A relative tolerance is exceeded only where the reference is non-zero
    (a zero reference makes the final comparison fail for the whole column,
    which this check conservatively does not anticipate).
    """
    abs_d = np.abs(run_vals - ref_vals)
    bad = _bad_run(run_vals) | ~np.isfinite(abs_d)
    with np.errstate(divide="ignore", invalid="ignore"):
        if is_rel:
            nonzero = np.abs(ref_vals) > 0.0
            over = nonzero & (
                abs_d / np.where(nonzero, np.abs(ref_vals), 1.0) > tol
            )
        else:
            over = abs_d > tol
    return np.asarray(bad | over, dtype=bool)


def find_early_divergence(
    ref: history_compare.HistoryTable,
    run: history_compare.HistoryTable,
    spec: case_manifest.CaseManifestEntry,
) -> EarlyDivergence | None:
    """Return the first point of a partial run that is out of tolerance.

    Args:
        ref: Reference table (augmented as for the final comparison), arrays
            of shape ``(n_ref,)``.
        run: Partial run table, arrays of shape ``(n_run,)``. Its last row
            may be an incomplete step; it is never used to decide.
        spec: Case manifest entry giving ``independent_var`` and
            ``track_tolerances``.

    Returns:
        The earliest failing point over all tracked columns, or ``None``
        when no complete point of the partial run is out of tolerance, or
        when the comparison cannot be decided yet (too few rows, a missing
        coordinate, or a reference of zero span).
    """
    xcol = spec.independent_var
    if (
        not spec.track_tolerances
        or xcol not in ref.columns
        or xcol not in run.columns
    ):
        return None
    names = {
        (key[4:] if key.startswith("rel_") else key)
        for key in spec.track_tolerances
    }
    cols = sorted(
        c
        for c in names
        if c in ref.columns
        and c in run.columns
        and c != xcol
        and c not in history_compare.SKIP_ALONG_TRACK
    )
    try:
        ref_x, ref_cols, _ = history_compare._collapse_nondecreasing(
            ref.columns[xcol], {c: ref.columns[c] for c in cols}, "reference"
        )
        run_x, run_cols, _ = history_compare._collapse_nondecreasing(
            run.columns[xcol], {c: run.columns[c] for c in cols}, "run"
        )
    except ValueError:
        return None
    if ref_x.size < 2 or run_x.size < 3 or ref_x[-1] <= ref_x[0]:
        return None
    lo = float(max(ref_x[0], run_x[0]))
    # The last collapsed row can still change; everything before it is final.
    done = run_x.size - 1
    x_done = run_x[:done]
    interior = (x_done > lo) & (x_done < float(ref_x[-1]))
    ref_sparse = history_compare._reference_skips_models(ref)
    node_mask = (ref_x >= lo) & (ref_x <= float(x_done[-1]))
    nodes = ref_x[node_mask]
    best: tuple[float, str, float, float, float] | None = None
    for key, tol in spec.track_tolerances.items():
        is_rel = key.startswith("rel_")
        col = key[4:] if is_rel else key
        if col not in run_cols:
            continue
        if ref_sparse or col in history_compare.DISCRETE_TRACK_COLUMNS:
            # The final comparison evaluates these columns only at the
            # reference rows. With fewer than two such rows in the partial
            # run it cannot yet tell which grid it will use, so it waits.
            if nodes.size < 2:
                continue
            pts = nodes
            ref_vals = ref_cols[col][node_mask]
            run_vals = np.interp(pts, run_x, run_cols[col])
        else:
            pts = x_done[interior]
            if pts.size == 0:
                continue
            ref_vals = np.interp(pts, ref_x, ref_cols[col])
            run_vals = np.interp(pts, run_x, run_cols[col])
        hits = np.flatnonzero(_exceeds(run_vals, ref_vals, tol, is_rel))
        if hits.size == 0:
            continue
        idx = int(hits[0])
        cand = (
            float(pts[idx]),
            key,
            float(run_vals[idx]),
            float(ref_vals[idx]),
            float(tol),
        )
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        return None
    x_hit = best[0]
    model: int | None = None
    if "model_number" in run.columns:
        row = int(np.searchsorted(run.columns[xcol], x_hit, side="left"))
        row = min(row, run.num_rows - 1)
        model = round(float(run.columns["model_number"][row]))
    return EarlyDivergence(
        column=best[1],
        x_name=xcol,
        x=x_hit,
        model=model,
        run_value=best[2],
        ref_value=best[3],
        tolerance=best[4],
    )


def read_partial_history(
    path: pathlib.Path,
) -> history_compare.HistoryTable | None:
    """Load the complete lines of a history file that is still being written.

    Args:
        path: Path of the history file of a running case.

    Returns:
        The table of the complete rows, or ``None`` when the file cannot be
        read or has no complete data row yet.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    cut = raw.rfind(b"\n")
    if cut < 0:
        return None
    with tempfile.TemporaryDirectory(prefix="flexmesa_partial_") as tmp:
        copy = pathlib.Path(tmp) / path.name
        copy.write_bytes(raw[: cut + 1])
        try:
            return history_compare.load_history_file(copy)
        except (OSError, ValueError):
            return None


@dataclasses.dataclass(frozen=True)
class StopRequest:
    """A request from a monitor to stop a running case.

    Attributes:
        status: Run status to report: ``"timeout"`` for a step bound (an
            inconclusive budget cut-off, ``DECISIONS #59``) or
            ``"diverged"`` for an early divergence stop.
        message: Report line for the case table.
    """

    status: str
    message: str


#: Signature of the polling hook of a running case: given the history file
#: written so far, return the reason to stop the case, or ``None``.
RunMonitor = Callable[[pathlib.Path], StopRequest | None]


@dataclasses.dataclass
class EarlyStopMonitor:
    """Polling hook that applies the step bound and the divergence check.

    Attributes:
        spec: Manifest entry of the case.
        ref: Augmented reference table, or ``None`` when the divergence
            check does not apply to this case.
        step_limit: Stop once the run's last ``model_number`` reaches this
            value, or ``None`` for no step bound.
        step_message: Report line used when the step bound stops the case.
    """

    spec: case_manifest.CaseManifestEntry
    ref: history_compare.HistoryTable | None
    step_limit: int | None = None
    step_message: str = ""

    def __call__(self, path: pathlib.Path) -> StopRequest | None:
        """Check the partial history at ``path`` (see class docstring)."""
        if self.ref is None and self.step_limit is None:
            return None
        run = read_partial_history(path)
        if run is None:
            return None
        if self.ref is not None:
            hit = find_early_divergence(self.ref, run, self.spec)
            if hit is not None:
                return StopRequest("diverged", hit.message())
        if self.step_limit is not None and "model_number" in run.columns:
            last = float(run.columns["model_number"][-1])
            if math.isfinite(last) and last >= self.step_limit:
                return StopRequest("timeout", self.step_message)
        return None


def single_stage_star_case(case_dir: pathlib.Path) -> bool:
    """Return whether ``case_dir`` runs exactly one stage under the harness.

    A case with several stages writes a history for each; the history that
    the final comparison sees is the last one, so a partial history of an
    earlier stage says nothing about the verdict.

    Args:
        case_dir: Upstream ``star`` test-suite case directory.

    Returns:
        True when the ``rn`` script runs exactly one ``do_one`` stage with
        the optional inlists skipped, as the harness runs it (file copies
        between stages do not write a history).
    """
    try:
        stages = _rust.star_parse_rn_stages(str(case_dir), True)
    except (ValueError, RuntimeError, OSError):
        return False
    return sum(1 for stage in stages if stage[0] == "do_one") == 1


def build_monitor(
    spec: case_manifest.CaseManifestEntry,
    case_dir: pathlib.Path,
    *,
    early_stop: bool,
    mesa_steps: int | None,
    step_factor: float | None,
    root_dir: pathlib.Path | None = None,
) -> EarlyStopMonitor | None:
    """Build the polling hook of one case, or ``None`` when none applies.

    The divergence check is enabled only for single-stage ``star`` cases with
    a MESA ``26.4.1`` history reference and no per-track reference; every
    other case runs without it.

    Args:
        spec: Manifest entry of the case.
        case_dir: Upstream case directory.
        early_stop: Enable the divergence check.
        mesa_steps: Step count of the MESA reference run, for the step bound.
        step_factor: Stop at ``step_factor * mesa_steps`` steps, or ``None``.
        root_dir: Manifest directory against which reference paths resolve.

    Returns:
        The monitor, or ``None`` when neither check applies to the case.
    """
    ref: history_compare.HistoryTable | None = None
    if (
        early_stop
        and spec.suite == "star"
        and spec.reference_status == "mesa_26_4_1"
        and spec.reference_path
        and not spec.track_reference_dir
        and single_stage_star_case(case_dir)
    ):
        base = case_manifest.repo_root() if root_dir is None else root_dir
        ref_file = _case_verdict.resolve_reference_history_path(
            spec.reference_path, root_dir=base
        )
        ref = history_compare.load_history_file(ref_file)
        ref = _case_verdict._augment_ref_columns(
            ref, spec, fetch.load_upstream_reference().version_number, base
        )
    limit: int | None = None
    message = ""
    if step_factor is not None and mesa_steps:
        limit = max(1, math.ceil(step_factor * mesa_steps))
        message = (
            f"stopped at the step bound: model {limit} reached "
            f"({step_factor:g} x {mesa_steps} MESA steps)"
        )
    if ref is None and limit is None:
        return None
    return EarlyStopMonitor(spec, ref, limit, message)
