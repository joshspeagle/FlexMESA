# SPDX-License-Identifier: GPL-3.0-or-later
"""Governed verdict evaluation and reference-path resolution for ``#143``.

Evaluates trajectory runs against ``case_manifest.toml`` and
``known_gaps.toml``, enforcing ``r26.4.1`` header provenance, pulsation
scaffolding tags (``#170``, ``#171``), and library-extras tracking (``#148``).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re

import numpy as np

from flexmesa.data import fetch
from flexmesa.testing import case_manifest, history_compare

MASS_LOSS_AUGMENT_COLUMNS: tuple[str, ...] = (
    "log_abs_mdot",
    "star_mdot",
    "lg_wind_mdot_1",
    "lg_wind_mdot_2",
)
AUGMENT_REFERENCE_COLUMNS: tuple[str, ...] = (
    MASS_LOSS_AUGMENT_COLUMNS + history_compare.ENERGY_ERROR_COLUMNS
)


@dataclasses.dataclass(frozen=True)
class TraceRunOutcome:
    """Execution telemetry from a ``star``, ``binary``, or ``astero`` run.

    Attributes:
        suite: Suite name (``"star"``, ``"binary"``, or ``"astero"``).
        case: Case directory name.
        status: Execution status (``"ok"``, ``"error"``, ``"timeout"``,
            ``"oom_killed"``, or ``"diverged"`` for a case stopped by
            ``--early-stop-on-divergence``, ``DECISIONS #68``).
        termination_code: Observed termination code (or empty string on crash).
        steps: Total evolutionary steps completed before exit.
        retries: Total step retries recorded.
        wall_time_sec: Elapsed wall-clock time in seconds (``s``).
        peak_rss_gb: Peak resident memory of the worker in gigabytes (``GB``).
        history_path: Path to the emitted ``history.data`` or
            ``binary_history.data`` file if present on disk, else ``None``.
        error_message: Diagnostic error string when ``status != "ok"``.
        controls_ignored: True when the run used the ``#147`` opt-out and
            continued past unsupported controls. Such a run is diagnostic
            only and can never produce a ``pass`` verdict.
        n_controls_ignored: Number of unsupported controls and sections the
            strict run named before stopping (``0`` when unknown).
        strict_rejection: Message of the strict run that stopped on
            unsupported controls, when this outcome is its diagnostic rerun.
        iters: Cumulative Newton solver iterations across logged final-stage
            steps, or ``None`` when ``num_iters`` is unavailable.
        mean_iters: Mean Newton solver iterations per logged final-stage step,
            or ``None`` when ``num_iters`` is unavailable.
        track_history_paths: Per-track ``astero`` search histories
            (``sample_NNNN_history.data``) ordered by track number, the last
            one possibly cut short when the run did not finish (``#430``).
    """

    suite: str
    case: str
    status: str
    termination_code: str
    steps: int
    retries: int
    wall_time_sec: float
    peak_rss_gb: float
    history_path: pathlib.Path | None
    error_message: str = ""
    controls_ignored: bool = False
    n_controls_ignored: int = 0
    strict_rejection: str = ""
    iters: int | None = None
    mean_iters: float | None = None
    track_history_paths: tuple[pathlib.Path, ...] = ()


_UNSUPPORTED_COUNT_RE = re.compile(
    r"unsupported control\(s\) \((\d+)(?: total)?"
)
_UNSUPPORTED_SECTIONS_RE = re.compile(
    r"unsupported upstream namelist section\(s\): ([^;()]+)"
)


def is_unsupported_controls_rejection(message: str) -> bool:
    """Return whether ``message`` is a strict ``#147`` control rejection.

    Only rejections that the opt-out can continue past qualify. Unknown or
    misspelled controls and sections fail with or without the opt-out.

    Args:
        message: Error text of a failed run.

    Returns:
        True when the run stopped only on unsupported upstream controls or
        namelist sections.
    """
    if "unknown" in message:
        return False
    return bool(
        _UNSUPPORTED_COUNT_RE.search(message)
        or _UNSUPPORTED_SECTIONS_RE.search(message)
    )


def count_unsupported_controls(message: str) -> int:
    """Count the unsupported controls and sections named in ``message``.

    Args:
        message: Error text of a strict ``#147`` control rejection.

    Returns:
        Number of unsupported controls plus unsupported namelist sections in
        the stage inlist that stopped the run (later stages are not scanned).
    """
    total = 0
    m_ctrl = _UNSUPPORTED_COUNT_RE.search(message)
    if m_ctrl is not None:
        total += int(m_ctrl.group(1))
    m_sec = _UNSUPPORTED_SECTIONS_RE.search(message)
    if m_sec is not None:
        total += sum(1 for tok in m_sec.group(1).split(",") if tok.strip())
    return total


@dataclasses.dataclass(frozen=True)
class CaseHarnessRecord:
    """Manifest, execution, and trajectory-comparison record for one case."""

    suite: str
    case: str
    reference_status: str
    reference_path: str
    required_termination: str
    verdict: str
    executed: bool
    run_status: str
    observed_termination: str
    termination_matches: bool
    steps: int
    retries: int
    wall_time_sec: float
    peak_rss_gb: float
    known_gap_issue: str
    known_gap_reason: str
    comparison: history_compare.HistoryComparison | None
    error_message: str = ""
    is_undeclared_failure: bool = False
    is_stale_known_gap: bool = False
    controls_ignored: bool = False
    n_controls_ignored: int = 0
    strict_rejection: str = ""
    component_comparisons: dict[str, history_compare.HistoryComparison] = (
        dataclasses.field(default_factory=dict)
    )
    has_component_error: bool = False
    iters: int | None = None
    mean_iters: float | None = None
    mesa_steps: int | None = None
    mesa_retries: int | None = None
    mesa_iters: int | None = None
    mesa_mean_iters: float | None = None
    mesa_iters_rows: int = 0

    @property
    def is_within_tolerances(self) -> bool:
        """Return whether primary and component comparisons meet tolerances."""
        return (
            self.comparison is not None
            and self.comparison.within_final_tolerances
            and self.comparison.within_track_tolerances
            and not self.has_component_error
            and all(
                comp.within_final_tolerances and comp.within_track_tolerances
                for comp in self.component_comparisons.values()
            )
        )

    @property
    def run_mode(self) -> str:
        """Label of how the reported run was made.

        Returns:
            ``"-"`` when the case was not run, ``"strict"`` for a run that
            passed strict control validation, and ``"diagnostic (N controls
            ignored)"`` for a rerun that continued past unsupported controls.
        """
        if not self.executed:
            return "-"
        if not self.controls_ignored:
            return "strict"
        if self.n_controls_ignored > 0:
            return f"diagnostic ({self.n_controls_ignored} controls ignored)"
        return "diagnostic (controls ignored)"


def resolve_reference_history_path(
    reference_path: str,
    root_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Resolve ``reference_path`` against ``root_dir``, repo root, or cache."""
    if not reference_path:
        raise FileNotFoundError("empty reference_path")
    candidates: list[pathlib.Path] = []
    if root_dir is not None:
        candidates.append(root_dir / reference_path)
    candidates.append(case_manifest.repo_root() / reference_path)
    candidates.append(fetch.resolve_mesa_histories_dir() / reference_path)
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"missing reference history {reference_path!r} "
        "(run 'flexmesa data fetch --include-test-inputs')"
    )


def resolve_binary_component_reference_paths(
    case: str,
    root_dir: pathlib.Path | None = None,
) -> dict[str, pathlib.Path]:
    """Locate stored ``LOGS1``/``LOGS2`` component reference tracks."""
    repo = case_manifest.repo_root()
    if root_dir is not None and root_dir.resolve() != repo.resolve():
        # The fetched bundle holds the binary references: keep it as the
        # fallback for an overridden root, as `resolve_reference_history_path`
        # does (#409).
        search_roots = [root_dir, fetch.resolve_mesa_histories_dir()]
    else:
        search_roots = [repo, fetch.resolve_mesa_histories_dir()]

    found: dict[str, pathlib.Path] = {}
    for comp_name in ("LOGS1", "LOGS2"):
        for sroot in search_roots:
            for prefix in ("runs", "thinned"):
                cand = (
                    sroot
                    / prefix
                    / "binary"
                    / case
                    / comp_name
                    / "history.data"
                )
                if cand.is_file():
                    found[comp_name] = cand
                    break
            if comp_name in found:
                break
    return found


def resolve_full_reference_history_path(
    suite: str,
    case: str,
    root_dir: pathlib.Path | None = None,
) -> pathlib.Path | None:
    """Locate the un-thinned MESA ``26.4.1`` history file in ``runs/``.

    Only consults the host ``mesa_histories`` cache when ``root_dir`` is
    ``None`` or matches the repository root so hermetic tests under temporary
    directories remain isolated.

    Args:
        suite: Suite name (``"star"``, ``"binary"``, or ``"astero"``).
        case: Upstream test-case directory name.
        root_dir: Optional root directory override.

    Returns:
        Path to the full MESA ``history.data`` or ``binary_history.data`` file
        if available, else ``None``.
    """
    repo = case_manifest.repo_root()
    if root_dir is not None and root_dir.resolve() != repo.resolve():
        # The fetched bundle holds the binary references: keep it as the
        # fallback for an overridden root, as `resolve_reference_history_path`
        # does (#409).
        search_roots = [root_dir, fetch.resolve_mesa_histories_dir()]
    else:
        search_roots = [repo, fetch.resolve_mesa_histories_dir()]

    for sroot in search_roots:
        case_root = sroot / "runs" / suite / case
        if not case_root.is_dir():
            continue
        if suite == "binary":
            cand = case_root / "binary_history.data"
            if cand.is_file():
                return cand
            continue
        result_json = case_root / "case_result.json"
        if result_json.is_file():
            try:
                raw = json.loads(result_json.read_text(encoding="utf-8"))
                hfiles = raw.get("history_files", [])
                if isinstance(hfiles, list) and hfiles:
                    cand = case_root / str(hfiles[-1])
                    if cand.is_file():
                        return cand
            except (OSError, ValueError):
                pass
        cand = case_root / "LOGS" / "history.data"
        if cand.is_file():
            return cand
    return None


TRACK_REFERENCE_RE = re.compile(r"^track_(\d{4})\.data$")
TRACK_PREFIX_SUFFIX = "_prefix"


def track_key(index: int, *, prefix: bool = False) -> str:
    """Return the comparison key of search track ``index`` (``#430``).

    Args:
        index: 1-based track number in the order the search evolved them.
        prefix: True when the ``FlexMESA`` track was cut short and is
            compared with the matching prefix of the MESA track.

    Returns:
        ``"track_NNNN"``, with ``"_prefix"`` appended for a prefix comparison.
    """
    return f"track_{index:04d}" + (TRACK_PREFIX_SUFFIX if prefix else "")


def numbered_track_files(
    directory: pathlib.Path,
    pattern: re.Pattern[str],
) -> tuple[pathlib.Path, ...]:
    """List the files of ``directory`` whose names number tracks ``1..N``.

    Args:
        directory: Directory to scan.
        pattern: Regular expression whose first group is the track number.

    Returns:
        Matching files ordered by track number.

    Raises:
        ValueError: If the track numbers are not exactly ``1..N``.
    """
    found: dict[int, pathlib.Path] = {}
    for path in directory.iterdir():
        match = pattern.match(path.name)
        if match is not None and path.is_file():
            found[int(match.group(1))] = path
    numbers = sorted(found)
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(
            f"{directory.name}: track files are not numbered 1..N: {numbers}"
        )
    return tuple(found[n] for n in numbers)


def resolve_track_reference_paths(
    track_reference_dir: str,
    root_dir: pathlib.Path | None = None,
) -> tuple[pathlib.Path, ...]:
    """Resolve the per-track MESA references of an ``astero`` search case.

    The references ship in the ``mesa-astero-tracks-v1`` data bundle
    (``flexmesa data fetch --include-test-inputs``, ``#430``). A missing
    bundle is an error: the caller fails the case loudly instead of skipping
    the comparison.

    Args:
        track_reference_dir: Manifest ``track_reference_dir`` value, relative
            to the bundle root (``astero/<case>``).
        root_dir: Optional root directory tried before the bundle when it is
            not the repository root (hermetic tests).

    Returns:
        ``track_0001.data`` ... ``track_NNNN.data`` in track order.

    Raises:
        FileNotFoundError: If no candidate directory holds track files.
        ValueError: If the track numbers are not exactly ``1..N``.
    """
    candidates: list[pathlib.Path] = []
    repo = case_manifest.repo_root()
    if root_dir is not None and root_dir.resolve() != repo.resolve():
        candidates.append(root_dir / track_reference_dir)
    candidates.append(
        fetch.resolve_mesa_astero_tracks_dir() / track_reference_dir
    )
    for cand in candidates:
        if cand.is_dir():
            paths = numbered_track_files(cand, TRACK_REFERENCE_RE)
            if paths:
                return paths
    raise FileNotFoundError(
        f"missing per-track references {track_reference_dir!r} "
        "(run 'flexmesa data fetch --include-test-inputs')"
    )


def _history_prefix(
    tbl: history_compare.HistoryTable,
    xcol: str,
    x_end: float,
) -> history_compare.HistoryTable:
    """Keep the rows of ``tbl`` with ``xcol <= x_end`` (at least one row)."""
    keep = np.asarray(tbl.columns[xcol] <= x_end)
    if not keep.any():
        keep = np.zeros(tbl.num_rows, dtype=bool)
        keep[0] = True
    return dataclasses.replace(
        tbl,
        columns={k: v[keep] for k, v in tbl.columns.items()},
        num_rows=int(keep.sum()),
    )


def compare_search_tracks(
    spec: case_manifest.CaseManifestEntry,
    ref_paths: tuple[pathlib.Path, ...],
    run_paths: tuple[pathlib.Path, ...],
    *,
    run_complete: bool,
    expected_version: str,
) -> tuple[dict[str, history_compare.HistoryComparison], list[str]]:
    """Compare ``FlexMESA`` search track ``i`` with MESA search track ``i``.

    MESA's ``astero`` search evolves one track per objective evaluation and
    rewrites ``LOGS/history.data`` each time, so a history comparison is like
    for like only between tracks with the same number (``#430``). A finished
    run is compared track by track and must have run as many tracks as MESA.
    In an unfinished run (cut by a step cap, time or memory budget, or an
    error) every track before the last is complete and the last one is
    compared with the matching prefix of the MESA track, ending at the last
    ``independent_var`` value the run reached.

    Args:
        spec: Manifest entry supplying ``independent_var`` and tolerances.
        ref_paths: MESA per-track references in track order.
        run_paths: ``FlexMESA`` per-track histories in track order.
        run_complete: True when the run finished uncapped with status
            ``"ok"``, so its last track is complete.
        expected_version: MESA ``version_number`` every reference must carry.

    Returns:
        ``(comparisons, errors)``: comparisons keyed by :func:`track_key`,
        and messages for track-count mismatches and unreadable complete
        tracks.

    Raises:
        ValueError: If a reference lacks the expected MESA version header.
        KeyError: If a reference lacks ``spec.independent_var``.
    """
    comps: dict[str, history_compare.HistoryComparison] = {}
    errors: list[str] = []
    n_ref, n_run = len(ref_paths), len(run_paths)
    if n_run > n_ref or (run_complete and n_run != n_ref):
        errors.append(f"FlexMESA ran {n_run} search tracks, MESA {n_ref}")
    xcol = spec.independent_var
    for idx in range(1, min(n_ref, n_run) + 1):
        is_prefix = not run_complete and idx == n_run
        ref_tbl = history_compare.load_history_file(ref_paths[idx - 1])
        if (
            not ref_tbl.provenance_comment
            and ref_tbl.mesa_version_number != expected_version
        ):
            raise ValueError(
                f"{spec.case} track {idx}: expected MESA version_number "
                f"{expected_version!r}, got {ref_tbl.mesa_version_number!r}"
            )
        if xcol not in ref_tbl.columns:
            raise KeyError(
                f"{spec.case} track {idx}: missing independent_var {xcol!r}"
            )
        try:
            run_tbl = history_compare.load_history_file(run_paths[idx - 1])
            if is_prefix:
                if run_tbl.num_rows == 0 or xcol not in run_tbl.columns:
                    continue
                ref_tbl = _history_prefix(
                    ref_tbl, xcol, float(run_tbl.columns[xcol][-1])
                )
            comps[track_key(idx, prefix=is_prefix)] = (
                history_compare.compare_histories(ref_tbl, run_tbl, spec)
            )
        except (OSError, ValueError, KeyError) as err:
            if is_prefix:
                # A track cut off while its first rows were being written
                # holds nothing to compare yet.
                continue
            err_s = _sanitize_error_message(str(err))
            errors.append(f"track {idx} compare error: {err_s}")
    return comps, errors


def _augment_ref_columns(
    ref_tbl: history_compare.HistoryTable,
    spec: case_manifest.CaseManifestEntry,
    expected_ver: str,
    root_dir: pathlib.Path | None,
) -> history_compare.HistoryTable:
    """Merge mass-loss and energy columns from cached MESA ``26.4.1`` runs."""
    if spec.reference_status != "mesa_26_4_1":
        return ref_tbl
    if all(col in ref_tbl.columns for col in AUGMENT_REFERENCE_COLUMNS):
        return ref_tbl
    full_path = resolve_full_reference_history_path(
        spec.suite, spec.case, root_dir=root_dir
    )
    if full_path is None:
        return ref_tbl
    try:
        full_tbl = history_compare.load_history_file(full_path)
    except (OSError, ValueError):
        return ref_tbl
    if (
        not full_tbl.provenance_comment
        and full_tbl.mesa_version_number != expected_ver
    ):
        return ref_tbl
    if full_tbl.num_rows != ref_tbl.num_rows:
        return ref_tbl
    if (
        "model_number" in ref_tbl.columns
        and "model_number" in full_tbl.columns
        and not np.array_equal(
            ref_tbl.columns["model_number"], full_tbl.columns["model_number"]
        )
    ):
        return ref_tbl
    extra = {
        col: full_tbl.columns[col]
        for col in AUGMENT_REFERENCE_COLUMNS
        if col in full_tbl.columns and col not in ref_tbl.columns
    }
    if not extra:
        return ref_tbl
    merged_cols = dict(ref_tbl.columns)
    merged_cols.update(extra)
    return dataclasses.replace(ref_tbl, columns=merged_cols)


def extract_reference_run_stats(
    spec: case_manifest.CaseManifestEntry,
    *,
    root_dir: pathlib.Path | None = None,
    ref_tbl: history_compare.HistoryTable | None = None,
) -> history_compare.HistoryRunStats:
    """Extract MESA ``26.4.1`` step, retry, and iteration counts for ``spec``.

    For ``binary`` cases whose ``binary_history.data`` omits ``num_retries``
    or ``num_iters``, supplements missing fields from ``LOGS1/history.data``
    when available.
    """
    if spec.reference_status == "no_reference_yet" or not spec.reference_path:
        return history_compare.HistoryRunStats()
    tbl = ref_tbl
    if tbl is None:
        try:
            ref_file = resolve_reference_history_path(
                spec.reference_path, root_dir=root_dir
            )
            tbl = history_compare.load_history_file(ref_file)
        except (FileNotFoundError, OSError, ValueError):
            return history_compare.HistoryRunStats()

    stats = history_compare.extract_history_run_stats(tbl, spec.suite)
    if spec.suite == "binary" and (
        stats.retries is None or stats.iters_sum is None
    ):
        comp_refs = resolve_binary_component_reference_paths(
            spec.case, root_dir=root_dir
        )
        logs1 = comp_refs.get("LOGS1")
        if logs1 is not None:
            try:
                c_tbl = history_compare.load_history_file(logs1)
                c_stats = history_compare.extract_history_run_stats(
                    c_tbl, "star"
                )
                stats = history_compare.HistoryRunStats(
                    steps=stats.steps,
                    retries=(
                        stats.retries
                        if stats.retries is not None
                        else c_stats.retries
                    ),
                    iters_sum=(
                        stats.iters_sum
                        if stats.iters_sum is not None
                        else c_stats.iters_sum
                    ),
                    mean_iters=(
                        stats.mean_iters
                        if stats.mean_iters is not None
                        else c_stats.mean_iters
                    ),
                    iters_rows=(
                        stats.iters_rows
                        if stats.iters_sum is not None
                        else c_stats.iters_rows
                    ),
                )
            except (OSError, ValueError):
                pass
    return stats


def _sanitize_error_message(msg: str) -> str:
    if not msg:
        return ""
    cleaned = msg
    for raw_dir, token in (
        (str(fetch.resolve_mesa_test_inputs_dir()), "$MESA_DIR"),
        (str(fetch.resolve_mesa_histories_dir()), "$FLEXMESA_HISTORIES_DIR"),
        (str(fetch.resolve_data_dir()), "$FLEXMESA_DATA_DIR"),
        (str(case_manifest.repo_root()), "<repo>"),
        (str(pathlib.Path.home()), "~"),
    ):
        if raw_dir:
            cleaned = cleaned.replace(raw_dir, token)
    cleaned = re.sub(r"/(?:tmp|var/folders)/[^\s:'\")\]]+", "<tmpdir>", cleaned)
    cleaned = re.sub(r"/(?:home|Users|mnt)/[^\s:'\")\]]+", "<path>", cleaned)
    return cleaned


def evaluate_case_record(
    spec: case_manifest.CaseManifestEntry,
    known_gaps: dict[tuple[str, str], case_manifest.KnownGapSpec],
    outcome: TraceRunOutcome | None,
    *,
    max_steps: int | None = None,
    root_dir: pathlib.Path | None = None,
    precomputed_comparison: history_compare.HistoryComparison | None = None,
    precomputed_component_comparisons: (
        dict[str, history_compare.HistoryComparison] | None
    ) = None,
    precomputed_has_component_error: bool = False,
    precomputed_mesa_stats: history_compare.HistoryRunStats | None = None,
) -> CaseHarnessRecord:
    """Combine manifest metadata, run outcome, and comparison into a verdict.

    An ``astero`` search case with a manifest ``track_reference_dir`` is
    compared track by track (:func:`compare_search_tracks`, ``#430``): the
    per-track comparisons go to ``component_comparisons`` and the first
    track's comparison is the record's primary ``comparison``.
    """
    gap = known_gaps.get(spec.key)
    gap_issue = gap.issue if gap is not None else ""
    gap_reason = gap.reason if gap is not None else ""
    base = case_manifest.repo_root() if root_dir is None else root_dir
    mesa_stats = precomputed_mesa_stats

    if outcome is None:
        if mesa_stats is None:
            mesa_stats = extract_reference_run_stats(spec, root_dir=base)
        if spec.reference_status == "no_reference_yet":
            verdict = "no reference yet"
        elif spec.reference_status == "interim_0b2e6bb":
            verdict = "interim reference (not run)"
        else:
            verdict = "not run"
        return CaseHarnessRecord(
            suite=spec.suite,
            case=spec.case,
            reference_status=spec.reference_status,
            reference_path=spec.reference_path,
            required_termination=spec.required_termination,
            verdict=verdict,
            executed=False,
            run_status="not_run",
            observed_termination="",
            termination_matches=False,
            steps=0,
            retries=0,
            wall_time_sec=0.0,
            peak_rss_gb=0.0,
            known_gap_issue=gap_issue,
            known_gap_reason=gap_reason,
            comparison=None,
            mesa_steps=mesa_stats.steps,
            mesa_retries=mesa_stats.retries,
            mesa_iters=mesa_stats.iters_sum,
            mesa_mean_iters=mesa_stats.mean_iters,
            mesa_iters_rows=mesa_stats.iters_rows,
        )

    term_ok = case_manifest.termination_matches(
        spec.required_termination, outcome.termination_code
    )
    comp: history_compare.HistoryComparison | None = (
        history_compare.re_evaluate_comparison(precomputed_comparison, spec)
        if precomputed_comparison is not None
        else None
    )
    comp_err = _sanitize_error_message(outcome.error_message)
    run_iters = outcome.iters
    run_mean_iters = outcome.mean_iters

    if (
        comp is None
        and spec.reference_status != "no_reference_yet"
        and spec.reference_path
    ):
        ref_file = resolve_reference_history_path(
            spec.reference_path, root_dir=base
        )
        ref_tbl = history_compare.load_history_file(ref_file)
        expected_ver = fetch.load_upstream_reference().version_number
        if (
            spec.reference_status == "mesa_26_4_1"
            and not ref_tbl.provenance_comment
            and ref_tbl.mesa_version_number != expected_ver
        ):
            raise ValueError(
                f"{spec.reference_path}: expected MESA version_number "
                f"{expected_ver!r}, got {ref_tbl.mesa_version_number!r}"
            )
        if spec.independent_var not in ref_tbl.columns:
            raise KeyError(
                f"{spec.reference_path}: missing independent_var "
                f"{spec.independent_var!r}"
            )
        if mesa_stats is None:
            mesa_stats = extract_reference_run_stats(
                spec, root_dir=base, ref_tbl=ref_tbl
            )
        ref_tbl = _augment_ref_columns(ref_tbl, spec, expected_ver, base)
        # An astero search case with per-track references is compared track
        # by track below: its LOGS/history.data holds a different track than
        # the reference at reference_path (#430).
        if outcome.history_path is not None and not spec.track_reference_dir:
            try:
                run_tbl = history_compare.load_history_file(
                    outcome.history_path
                )
                if run_iters is None:
                    r_stats = history_compare.extract_history_run_stats(
                        run_tbl, spec.suite
                    )
                    run_iters = r_stats.iters_sum
                    run_mean_iters = r_stats.mean_iters
                comp = history_compare.compare_histories(ref_tbl, run_tbl, spec)
            except (OSError, ValueError, KeyError) as err:
                err_s = _sanitize_error_message(str(err))
                comp_err = f"{comp_err}; compare error: {err_s}".strip("; ")
    elif mesa_stats is None:
        mesa_stats = extract_reference_run_stats(spec, root_dir=base)
    if (
        run_iters is None
        and outcome.history_path is not None
        and outcome.history_path.is_file()
    ):
        try:
            r_tbl = history_compare.load_history_file(outcome.history_path)
            r_stats = history_compare.extract_history_run_stats(
                r_tbl, spec.suite
            )
            run_iters = r_stats.iters_sum
            run_mean_iters = r_stats.mean_iters
        except (OSError, ValueError):
            pass

    comp_map: dict[str, history_compare.HistoryComparison] = {}
    has_comp_err = precomputed_has_component_error
    if precomputed_component_comparisons is not None:
        track_mode = bool(spec.track_reference_dir)
        comp_map = {
            cname: history_compare.re_evaluate_comparison(
                ccomp, spec if track_mode else spec.component_spec(cname)
            )
            for cname, ccomp in precomputed_component_comparisons.items()
        }
    elif (
        spec.track_reference_dir
        and spec.reference_status == "mesa_26_4_1"
        and precomputed_comparison is None
        and (outcome.track_history_paths or outcome.history_path is not None)
    ):
        comp_map, track_errors = compare_search_tracks(
            spec,
            resolve_track_reference_paths(
                spec.track_reference_dir, root_dir=base
            ),
            outcome.track_history_paths,
            run_complete=outcome.status == "ok" and max_steps is None,
            expected_version=fetch.load_upstream_reference().version_number,
        )
        if track_errors:
            comp_err = "; ".join([comp_err, *track_errors]).strip("; ")
            has_comp_err = True
        comp = comp_map.get(track_key(1)) or comp_map.get(
            track_key(1, prefix=True)
        )
    elif (
        spec.suite == "binary"
        and spec.reference_status != "no_reference_yet"
        and spec.reference_path
        and outcome.history_path is not None
    ):
        expected_ver_c = fetch.load_upstream_reference().version_number
        comp_refs = resolve_binary_component_reference_paths(
            spec.case, root_dir=base
        )
        for cname, ref_cpath in comp_refs.items():
            cspec = spec.component_spec(cname, str(ref_cpath))
            ref_ctbl = history_compare.load_history_file(ref_cpath)
            if (
                spec.reference_status == "mesa_26_4_1"
                and not ref_ctbl.provenance_comment
                and ref_ctbl.mesa_version_number != expected_ver_c
            ):
                raise ValueError(
                    f"{spec.case}/{cname}: expected MESA version_number "
                    f"{expected_ver_c!r}, got {ref_ctbl.mesa_version_number!r}"
                )
            if cspec.independent_var not in ref_ctbl.columns:
                raise KeyError(
                    f"{spec.case}/{cname}: missing independent_var "
                    f"{cspec.independent_var!r}"
                )
            run_cpath = outcome.history_path.parent / cname / "history.data"
            if run_cpath.is_file():
                try:
                    run_ctbl = history_compare.load_history_file(run_cpath)
                    comp_map[cname] = history_compare.compare_histories(
                        ref_ctbl, run_ctbl, cspec
                    )
                except (OSError, ValueError, KeyError) as err:
                    err_s = _sanitize_error_message(str(err))
                    comp_err = (
                        f"{comp_err}; {cname} compare error: {err_s}".strip(
                            "; "
                        )
                    )
                    has_comp_err = True
            elif outcome.history_path.name == "binary_history.data":
                comp_err = f"{comp_err}; missing {cname}/history.data".strip(
                    "; "
                )
                has_comp_err = True

    # A run that continued past unsupported controls (#147 opt-out) did not
    # do what the inlist asks. Its numbers are diagnostic and it can never
    # satisfy the tolerance condition of any verdict.
    tol_ok = (
        comp is not None
        and comp.within_final_tolerances
        and comp.within_track_tolerances
        and not has_comp_err
        and all(
            c.within_final_tolerances and c.within_track_tolerances
            for c in comp_map.values()
        )
        and not outcome.controls_ignored
    )
    smoke_note = " (controls ignored)" if outcome.controls_ignored else ""
    is_undeclared = False
    is_stale_gap = False

    if max_steps is not None:
        if outcome.status == "ok":
            verdict = f"smoke: {outcome.steps} steps{smoke_note}"
        else:
            verdict = (
                f"smoke: {outcome.status} ({outcome.steps} steps){smoke_note}"
            )
    elif spec.reference_status == "no_reference_yet":
        verdict = "no reference yet"
    elif spec.reference_status == "interim_0b2e6bb":
        if outcome.status in ("timeout", "oom_killed"):
            verdict = f"interim reference ({outcome.status})"
        elif outcome.status != "ok" or comp is None:
            verdict = "interim reference (run failed)"
        elif not term_ok:
            verdict = "interim reference (wrong termination)"
        elif tol_ok:
            verdict = "interim reference (within tol)"
        else:
            verdict = "interim reference (outside tol)"
        if verdict == "interim reference (within tol)":
            is_stale_gap = gap is not None
        else:
            is_undeclared = gap is None
    else:
        if outcome.status == "ok" and term_ok and tol_ok:
            if spec.uses_pulsation_scaffolding:
                verdict = f"scaffolding ({spec.scaffolding_issue})"
            elif spec.uses_library_extras or spec.extras_status == "unported":
                # A case whose upstream extras logic is not executed did not
                # run what upstream runs, so it cannot count as passing.
                if gap is not None:
                    verdict = f"declared gap ({gap.issue})"
                else:
                    verdict = "fail (undeclared)"
                    is_undeclared = True
            else:
                verdict = "pass"
                is_stale_gap = gap is not None
        elif gap is not None:
            verdict = f"declared gap ({gap.issue})"
        elif outcome.status in ("timeout", "oom_killed"):
            # A run cut off by its time or memory budget is inconclusive: it
            # never counts as passing, and the baseline gate checks it as a
            # prefix of the committed row (DECISIONS #59).
            verdict = f"inconclusive ({outcome.status})"
        elif outcome.status == "diverged":
            # Stopped by --early-stop-on-divergence: a tracked column was
            # already out of its along-track tolerance, so the case cannot
            # pass (DECISIONS #68). The column and age are in the message.
            verdict = "fail (early stop)"
            is_undeclared = True
        else:
            verdict = "fail (undeclared)"
            is_undeclared = True

    return CaseHarnessRecord(
        suite=spec.suite,
        case=spec.case,
        reference_status=spec.reference_status,
        reference_path=spec.reference_path,
        required_termination=spec.required_termination,
        verdict=verdict,
        executed=True,
        run_status=outcome.status,
        observed_termination=outcome.termination_code,
        termination_matches=term_ok,
        steps=outcome.steps,
        retries=outcome.retries,
        wall_time_sec=outcome.wall_time_sec,
        peak_rss_gb=outcome.peak_rss_gb,
        known_gap_issue=gap_issue,
        known_gap_reason=gap_reason,
        comparison=comp,
        error_message=comp_err,
        is_undeclared_failure=is_undeclared,
        is_stale_known_gap=is_stale_gap,
        controls_ignored=outcome.controls_ignored,
        n_controls_ignored=outcome.n_controls_ignored,
        strict_rejection=_sanitize_error_message(outcome.strict_rejection),
        component_comparisons=comp_map,
        has_component_error=has_comp_err,
        iters=run_iters,
        mean_iters=run_mean_iters,
        mesa_steps=mesa_stats.steps if mesa_stats is not None else None,
        mesa_retries=mesa_stats.retries if mesa_stats is not None else None,
        mesa_iters=mesa_stats.iters_sum if mesa_stats is not None else None,
        mesa_mean_iters=(
            mesa_stats.mean_iters if mesa_stats is not None else None
        ),
        mesa_iters_rows=(
            mesa_stats.iters_rows if mesa_stats is not None else 0
        ),
    )
