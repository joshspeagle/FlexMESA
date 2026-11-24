# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-track comparison of ``astero`` search cases with MESA (``#430``).

MESA's ``astero`` search rewrites ``LOGS/history.data`` for every track it
evolves, so its stored history is the last track's. These tests build
synthetic MESA-format tracks whose ``log_Teff`` differs from track to track
by more than the track tolerance, so comparing two different tracks fails
and comparing a track with itself passes.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from flexmesa.data import fetch
from flexmesa.testing import (
    _case_verdict,
    case_manifest,
    case_report,
    harness_runner,
    history_compare,
    known_gaps_gate,
)

N_ROWS = 10
COLUMNS = ("model_number", "star_age", "log_L", "log_Teff", "log_g")


def _write_track(
    path: pathlib.Path,
    track: int,
    *,
    n_rows: int = N_ROWS,
    teff_kick: tuple[int, float] | None = None,
) -> pathlib.Path:
    """Write a MESA ``26.4.1``-format history for synthetic track ``track``.

    Args:
        path: Destination file.
        track: Track number; ``log_Teff`` is offset by ``0.01 * track`` dex.
        n_rows: Number of models written (a prefix of the full track).
        teff_kick: Optional ``(model_number, dex)`` added to ``log_Teff`` from
            that model on.

    Returns:
        ``path``.
    """
    head_names = ("version_number", "compiler", "build", "MESA_SDK_version")
    head_vals = ('"r26.4.1"', '"gfortran"', '"15.2.0"', '"x86_64-linux-26.6.1"')
    lines = [
        " ".join(f"{i:>24d}" for i in range(1, len(head_names) + 1)),
        " ".join(f"{n:>24s}" for n in head_names),
        " ".join(f"{v:>24s}" for v in head_vals),
        "",
        " ".join(f"{i:>24d}" for i in range(1, len(COLUMNS) + 1)),
        " ".join(f"{n:>24s}" for n in COLUMNS),
    ]
    for model in range(1, n_rows + 1):
        teff = 3.6 + 0.001 * model + 0.01 * track
        if teff_kick is not None and model >= teff_kick[0]:
            teff += teff_kick[1]
        row = (
            f"{model:>24d}",
            f"{1.0e3 * model**1.5:>24.16e}",
            f"{0.1 * model:>24.16e}",
            f"{teff:>24.16e}",
            f"{4.4:>24.16e}",
        )
        lines.append(" ".join(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _spec(
    track_reference_dir: str = "refs",
) -> case_manifest.CaseManifestEntry:
    return case_manifest.CaseManifestEntry(
        suite="astero",
        case="synthetic_search",
        reference_status="mesa_26_4_1",
        reference_path="refs/track_0003.data",
        required_termination="xtra1",
        independent_var="star_age",
        final_tolerances={
            "rel_star_age": 0.001,
            "log_L": 0.001,
            "log_Teff": 0.0005,
            "log_g": 0.001,
        },
        track_tolerances={"log_L": 0.005, "log_Teff": 0.002, "log_g": 0.002},
        track_reference_dir=track_reference_dir,
    )


def _mesa_refs(root: pathlib.Path, n_tracks: int = 3) -> None:
    for track in range(1, n_tracks + 1):
        _write_track(root / "refs" / f"track_{track:04d}.data", track)


def _run_tracks(
    root: pathlib.Path,
    n_tracks: int,
    *,
    last_rows: int = N_ROWS,
    kick_track: int | None = None,
) -> tuple[pathlib.Path, ...]:
    out = root / "run" / "outputs"
    paths = []
    for track in range(1, n_tracks + 1):
        paths.append(
            _write_track(
                out / f"sample_{track:04d}_history.data",
                track,
                n_rows=last_rows if track == n_tracks else N_ROWS,
                teff_kick=(4, 0.05) if track == kick_track else None,
            )
        )
    # FlexMESA copies one track to LOGS/history.data as well.
    hist = root / "run" / "LOGS" / "history.data"
    _write_track(hist, 1)
    return tuple(paths)


def _outcome(
    root: pathlib.Path,
    tracks: tuple[pathlib.Path, ...],
    status: str = "ok",
) -> harness_runner.TraceRunOutcome:
    return harness_runner.TraceRunOutcome(
        suite="astero",
        case="synthetic_search",
        status=status,
        termination_code="xtra1" if status == "ok" else status,
        steps=N_ROWS * len(tracks),
        retries=0,
        wall_time_sec=1.0,
        peak_rss_gb=0.1,
        history_path=root / "run" / "LOGS" / "history.data",
        track_history_paths=tracks,
    )


def _evaluate(
    root: pathlib.Path,
    outcome: harness_runner.TraceRunOutcome,
    spec: case_manifest.CaseManifestEntry | None = None,
    max_steps: int | None = None,
) -> harness_runner.CaseHarnessRecord:
    return harness_runner.evaluate_case_record(
        spec if spec is not None else _spec(),
        {},
        outcome,
        max_steps=max_steps,
        root_dir=root,
    )


def test_finished_run_with_matching_tracks_passes(
    tmp_path: pathlib.Path,
) -> None:
    """Every FlexMESA track equal to the same-numbered MESA track passes."""
    _mesa_refs(tmp_path)
    rec = _evaluate(tmp_path, _outcome(tmp_path, _run_tracks(tmp_path, 3)))
    assert sorted(rec.component_comparisons) == [
        "track_0001",
        "track_0002",
        "track_0003",
    ]
    assert rec.comparison is rec.component_comparisons["track_0001"]
    assert all(
        c.first_diverging_step is None
        for c in rec.component_comparisons.values()
    )
    assert not rec.has_component_error
    assert rec.verdict == "pass"


def test_whole_history_comparison_pairs_different_tracks(
    tmp_path: pathlib.Path,
) -> None:
    """Without per-track references track 1 is judged against MESA's last."""
    _mesa_refs(tmp_path)
    outcome = _outcome(tmp_path, _run_tracks(tmp_path, 3))
    rec = _evaluate(tmp_path, outcome, spec=_spec(track_reference_dir=""))
    assert rec.comparison is not None
    assert rec.comparison.first_diverging_step == 1
    assert rec.verdict == "fail (undeclared)"


def test_finished_run_with_fewer_tracks_than_mesa_fails(
    tmp_path: pathlib.Path,
) -> None:
    """A finished search that evolved 2 tracks where MESA evolved 3 fails."""
    _mesa_refs(tmp_path)
    rec = _evaluate(tmp_path, _outcome(tmp_path, _run_tracks(tmp_path, 2)))
    assert sorted(rec.component_comparisons) == ["track_0001", "track_0002"]
    assert rec.has_component_error
    assert "FlexMESA ran 2 search tracks, MESA 3" in rec.error_message
    assert rec.verdict == "fail (undeclared)"
    assert rec.is_undeclared_failure


def test_finished_run_with_more_tracks_than_mesa_fails(
    tmp_path: pathlib.Path,
) -> None:
    """Extra FlexMESA tracks have no MESA counterpart and fail the case."""
    _mesa_refs(tmp_path, n_tracks=2)
    spec = dataclasses.replace(_spec(), reference_path="refs/track_0002.data")
    outcome = _outcome(tmp_path, _run_tracks(tmp_path, 3))
    rec = _evaluate(tmp_path, outcome, spec=spec)
    assert sorted(rec.component_comparisons) == ["track_0001", "track_0002"]
    assert "FlexMESA ran 3 search tracks, MESA 2" in rec.error_message
    assert rec.verdict == "fail (undeclared)"


def test_budget_cut_run_compares_partial_last_track_as_prefix(
    tmp_path: pathlib.Path,
) -> None:
    """The cut last track is compared with the MESA prefix it reached."""
    _mesa_refs(tmp_path)
    tracks = _run_tracks(tmp_path, 2, last_rows=4)
    rec = _evaluate(tmp_path, _outcome(tmp_path, tracks, status="timeout"))
    assert sorted(rec.component_comparisons) == [
        "track_0001",
        "track_0002_prefix",
    ]
    prefix = rec.component_comparisons["track_0002_prefix"]
    assert (prefix.run_rows, prefix.ref_rows) == (4, 4)
    assert prefix.first_diverging_step is None
    assert prefix.within_final_tolerances and prefix.within_track_tolerances
    # An unfinished run is not short of tracks, and never passes.
    assert not rec.has_component_error
    assert rec.verdict == "inconclusive (timeout)"


def test_step_capped_run_compares_its_single_track_as_prefix(
    tmp_path: pathlib.Path,
) -> None:
    """A smoke run's only track is a prefix of MESA's first track."""
    _mesa_refs(tmp_path)
    tracks = _run_tracks(tmp_path, 1, last_rows=3)
    rec = _evaluate(tmp_path, _outcome(tmp_path, tracks), max_steps=3)
    assert sorted(rec.component_comparisons) == ["track_0001_prefix"]
    assert rec.comparison is rec.component_comparisons["track_0001_prefix"]
    assert rec.comparison.first_diverging_step is None
    assert not rec.has_component_error
    assert rec.verdict == "smoke: 10 steps"


def test_divergence_is_reported_on_the_track_where_it_happens(
    tmp_path: pathlib.Path,
) -> None:
    """A kick in track 2 from model 4 on shows as track 2's first divergence."""
    _mesa_refs(tmp_path)
    tracks = _run_tracks(tmp_path, 3, kick_track=2)
    rec = _evaluate(tmp_path, _outcome(tmp_path, tracks))
    steps = {
        k: c.first_diverging_step for k, c in rec.component_comparisons.items()
    }
    assert steps == {"track_0001": None, "track_0002": 4, "track_0003": None}
    assert not rec.is_within_tolerances
    assert rec.verdict == "fail (undeclared)"


def test_stored_track_rows_rebuild_the_same_verdict(
    tmp_path: pathlib.Path,
) -> None:
    """Records rebuilt from stored JSON rows keep the per-track verdict."""
    _mesa_refs(tmp_path)
    spec = _spec()
    manifest = case_manifest.HarnessManifest(
        mesa_release="26.4.1", zenodo_record="19722306", cases=(spec,)
    )
    for n_run, verdict in ((3, "pass"), (2, "fail (undeclared)")):
        tracks = _run_tracks(tmp_path / str(n_run), n_run)
        rec = _evaluate(tmp_path, _outcome(tmp_path / str(n_run), tracks))
        row = case_report.record_to_dict(rec)
        (rebuilt,) = case_report.records_from_stored_rows(
            manifest, {}, {spec.key: row}, tmp_path
        )
        assert rebuilt.verdict == verdict
        assert sorted(rebuilt.component_comparisons) == sorted(
            rec.component_comparisons
        )
        assert rebuilt.has_component_error == rec.has_component_error


def test_case_table_lists_each_track(tmp_path: pathlib.Path) -> None:
    """The case table renders one row per compared track."""
    _mesa_refs(tmp_path)
    tracks = _run_tracks(tmp_path, 2, last_rows=4, kick_track=1)
    rec = _evaluate(tmp_path, _outcome(tmp_path, tracks, status="timeout"))
    manifest = case_manifest.HarnessManifest(
        mesa_release="26.4.1", zenodo_record="19722306", cases=(_spec(),)
    )
    text = case_report.render_case_table_markdown(manifest, [rec])
    assert "## Astero Search Track Comparisons (`#430`)" in text
    assert "| `astero` | `synthetic_search` | 1 | full | 10 (10) | 4 |" in text
    assert "| `astero` | `synthetic_search` | 2 | prefix | 4 (4) | - |" in text


def test_track_references_must_be_numbered_from_one(
    tmp_path: pathlib.Path,
) -> None:
    """A missing reference track number is an error, not a shorter search."""
    _write_track(tmp_path / "refs" / "track_0001.data", 1)
    _write_track(tmp_path / "refs" / "track_0003.data", 3)
    with pytest.raises(ValueError, match="not numbered"):
        _case_verdict.resolve_track_reference_paths("refs", root_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        _case_verdict.resolve_track_reference_paths("absent", root_dir=tmp_path)


def test_locate_track_histories_orders_tracks_and_clears_stale_ones(
    tmp_path: pathlib.Path,
) -> None:
    """Run tracks are found in track order and removed before a rerun."""
    out = tmp_path / "run"
    for track in (2, 1, 10, 3, 4, 5, 6, 7, 8, 9):
        _write_track(out / "outputs" / f"sample_{track:04d}_history.data", 1)
    (out / "outputs" / "sample_0001.data").write_text("summary\n")
    found = harness_runner.locate_track_histories(out, True)
    assert [p.name[:11] for p in found] == [
        f"sample_{n:04d}" for n in range(1, 11)
    ]
    assert harness_runner.locate_track_histories(out / "LOGS", True) == ()
    harness_runner._clear_stale_history_outputs(out, True)
    assert harness_runner.locate_track_histories(out, True) == ()
    assert (out / "outputs" / "sample_0001.data").is_file()


def test_locate_track_histories_rejects_a_gap(tmp_path: pathlib.Path) -> None:
    """A run whose track numbers skip one is reported, not compared."""
    out = tmp_path / "run" / "outputs"
    _write_track(out / "sample_0001_history.data", 1)
    _write_track(out / "sample_0003_history.data", 3)
    with pytest.raises(ValueError, match="not numbered"):
        harness_runner.locate_track_histories(tmp_path / "run", True)


def test_manifest_accepts_track_references_only_for_astero(
    tmp_path: pathlib.Path,
) -> None:
    """``track_reference_dir`` is an ``astero`` ``mesa_26_4_1`` field."""
    body = (
        '[metadata]\nmesa_release = "26.4.1"\nzenodo_record = "19722306"\n\n'
        '[[cases]]\nsuite = "{suite}"\ncase = "c"\n'
        'reference_status = "mesa_26_4_1"\nreference_path = "r.data"\n'
        'required_termination = "xtra1"\ntrack_reference_dir = "refs/c"\n'
    )
    path = tmp_path / "case_manifest.toml"
    path.write_text(body.format(suite="astero"), encoding="utf-8")
    (entry,) = case_manifest.load_case_manifest(path).cases
    assert entry.track_reference_dir == "refs/c"
    path.write_text(body.format(suite="star"), encoding="utf-8")
    with pytest.raises(ValueError, match="track_reference_dir"):
        case_manifest.load_case_manifest(path)


def test_prefix_reference_keeps_rows_up_to_the_run_end() -> None:
    """The reference prefix ends at the last coordinate the run reached."""
    tbl = history_compare.HistoryTable(
        columns={
            "star_age": np.array([1.0, 2.0, 3.0]),
            "log_L": np.array([0.1, 0.2, 0.3]),
        },
        num_rows=3,
    )
    cut = _case_verdict._history_prefix(tbl, "star_age", 2.5)
    assert cut.num_rows == 2
    assert list(cut.columns["log_L"]) == [0.1, 0.2]
    early = _case_verdict._history_prefix(tbl, "star_age", 0.5)
    assert early.num_rows == 1


#: Tracks MESA 26.4.1 evolved per case: run1_star calls counted by the tap
#: (``history_tap/tracks.tsv``); fast_simplex has 7 tracks for 6 samples.
MESA_TRACK_COUNTS = {
    "example_astero": 1,
    "fast_from_file": 2,
    "fast_newuoa": 8,
    "fast_scan_grid": 4,
    "fast_simplex": 7,
}


def test_manifest_track_references_resolve_from_the_bundle() -> None:
    """Every search case's track directory holds MESA 26.4.1 tracks 1..N."""
    sentinel = fetch.resolve_mesa_astero_tracks_dir() / "PROVENANCE.json"
    if not known_gaps_gate.require_upstream_file(
        sentinel, label="mesa-astero-tracks-v1 bundle"
    ):
        return
    entries = {
        e.case: e
        for e in case_manifest.load_case_manifest().cases
        if e.track_reference_dir
    }
    assert sorted(entries) == sorted(MESA_TRACK_COUNTS)
    for case, entry in entries.items():
        paths = _case_verdict.resolve_track_reference_paths(
            entry.track_reference_dir
        )
        assert len(paths) == MESA_TRACK_COUNTS[case], case
        for path in paths:
            tbl = history_compare.load_history_file(path)
            assert tbl.mesa_version_number == "r26.4.1", path.name
            assert entry.independent_var in tbl.columns
