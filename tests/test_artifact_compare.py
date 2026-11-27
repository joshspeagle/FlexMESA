# SPDX-License-Identifier: GPL-3.0-or-later
"""Artifact comparison for zero-step cases (`#606`).

The reference is a real `MESA` `26.4.1` file committed under
`benchmarks/history_harness/reference_artifacts/`; the "runs" below are that
file with one edit each, so every expected outcome follows from the edit.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

from flexmesa.testing import (
    artifact_compare,
    case_manifest,
    case_report,
    harness_runner,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
KEY = ("star", "rsp_check_2nd_crossing")


@pytest.fixture(scope="module")
def spec() -> artifact_compare.ArtifactSpec:
    """Return the manifest's artifact declaration for the case."""
    entry = case_manifest.load_case_manifest().by_key()[KEY]
    assert entry.artifact is not None
    return entry.artifact


@pytest.fixture(scope="module")
def ref_text(spec: artifact_compare.ArtifactSpec) -> str:
    """Return the committed `MESA` file."""
    return (REPO_ROOT / spec.reference).read_text(encoding="utf-8")


def test_identical_artifact_is_within_tolerance_in_every_field(
    spec: artifact_compare.ArtifactSpec, ref_text: str
) -> None:
    """The reference compared with itself agrees in all fields."""
    cmp = artifact_compare.compare_artifact(spec, ref_text, ref_text)
    assert cmp.ok and cmp.n_out == 0
    # header values, model rows x 6, 2 edges, 11 table rows x 3
    scan = artifact_compare.parse_instability_strip(ref_text)
    assert cmp.n_fields == (
        len(scan.header) + 6 * len(scan.models) + 2 + 3 * len(scan.table)
    )
    assert scan.edges["Teff_red_edge"] == pytest.approx(5046.6949595302)
    assert scan.table[-1][0] == 951


def test_tolerance_is_one_unit_in_the_last_printed_digit(
    spec: artifact_compare.ArtifactSpec, ref_text: str
) -> None:
    """Tolerance is one unit of the last digit; 5 units fail, half passes."""
    run = ref_text.replace("-6.3966920632D-02", "-6.3966920637D-02", 1)
    assert run != ref_text
    strict = artifact_compare.compare_artifact(spec, ref_text, run)
    assert strict.first_failure is not None
    assert strict.first_failure.excess == pytest.approx(5.0, rel=1e-6)
    tols = dict(spec.tolerances, model_growth={"sig_digits": 10.0})
    loose = dataclasses.replace(spec, tolerances=tols)
    assert artifact_compare.compare_artifact(loose, ref_text, run).ok


def test_one_growth_rate_out_of_tolerance_names_that_field_first(
    spec: artifact_compare.ArtifactSpec, ref_text: str
) -> None:
    """A change in the ninth digit of one growth rate is the first failure."""
    run = ref_text.replace("-6.3966920632D-02", "-6.3966921632D-02", 1)
    cmp = artifact_compare.compare_artifact(spec, ref_text, run)
    assert cmp.status == "differs" and cmp.n_out == 1
    assert cmp.first_failure is not None
    assert cmp.first_failure.label == "model_growth[0]"
    assert cmp.largest is not None and cmp.largest.label == "model_growth[0]"
    assert "model_growth[0]" in cmp.message


def test_edge_beyond_print_precision_fails_with_edge_label(
    spec: artifact_compare.ArtifactSpec, ref_text: str
) -> None:
    """A red edge off by 1e-9 K (tolerance 1e-10 K) is reported."""
    run = ref_text.replace("5046.6949595302", "5046.6949595312", 1)
    cmp = artifact_compare.compare_artifact(spec, ref_text, run)
    assert cmp.first_failure is not None
    assert cmp.first_failure.label == "edge[Teff_red_edge]"


def test_missing_artifact_is_reported_as_missing(
    spec: artifact_compare.ArtifactSpec, ref_text: str
) -> None:
    """A run that wrote no file does not agree."""
    cmp = artifact_compare.compare_artifact(spec, ref_text, None)
    assert cmp.status == "missing" and not cmp.ok


def test_truncated_artifact_counts_missing_rows_as_out_of_tolerance(
    spec: artifact_compare.ArtifactSpec, ref_text: str
) -> None:
    """A file cut off after the model rows lacks the edges and the table."""
    run = ref_text.split("\n       Teff_red_edge")[0] + "\n"
    cmp = artifact_compare.compare_artifact(spec, ref_text, run)
    assert cmp.status == "differs"
    assert cmp.first_failure is not None
    assert cmp.first_failure.label == "edge[Teff_red_edge]"
    assert "table rows 0 vs 11" in cmp.message


def test_unparsable_artifact_is_reported(
    spec: artifact_compare.ArtifactSpec, ref_text: str
) -> None:
    """A line the format does not allow is an explicit failure."""
    cmp = artifact_compare.compare_artifact(
        spec, ref_text, ref_text + "x y z w\n"
    )
    assert cmp.status == "unparsable"


def test_stdout_blocks_compare_at_five_significant_digits(
    spec: artifact_compare.ArtifactSpec, ref_text: str
) -> None:
    """Per-model terminal lines agree at their print precision only."""
    stdout = (REPO_ROOT / spec.stdout_reference).read_text(encoding="utf-8")
    assert len(artifact_compare.parse_stdout_models(stdout)) == 16
    ok = artifact_compare.compare_artifact(
        spec, ref_text, ref_text, reference_stdout=stdout, run_stdout=stdout
    )
    assert ok.ok and ok.stdout_checked
    bad = artifact_compare.compare_artifact(
        spec,
        ref_text,
        ref_text,
        reference_stdout=stdout,
        run_stdout=stdout.replace("0.24733E+02", "0.24735E+02", 1),
    )
    assert bad.first_failure is not None
    assert bad.first_failure.label == "stdout_period[0]"


def test_manifest_rejects_incomplete_tolerance_table() -> None:
    """A tolerance group the parser needs but the table omits is an error."""
    raw = {
        "file": "a.data",
        "reference": "r.data",
        "parser": "rsp_instability_strip",
        "tolerances": {"header": {"abs": 0.0}},
    }
    with pytest.raises(ValueError, match="absent"):
        artifact_compare.parse_spec(raw, "star/x")


def test_comparison_round_trips_through_json_dict(
    spec: artifact_compare.ArtifactSpec, ref_text: str
) -> None:
    """Stored rows rebuild the same comparison (shard merge re-evaluates)."""
    run = ref_text.replace("5046.6949595302", "5046.6949595312", 1)
    cmp = artifact_compare.compare_artifact(spec, ref_text, run)
    back = artifact_compare.comparison_from_dict(
        artifact_compare.comparison_to_dict(cmp)
    )
    assert back == cmp


def _outcome(
    *, status: str = "ok", term: str = "other_rsp_build_model", steps: int = 0
) -> harness_runner.TraceRunOutcome:
    return harness_runner.TraceRunOutcome(
        suite="star",
        case="rsp_check_2nd_crossing",
        status=status,
        termination_code=term,
        steps=steps,
        retries=0,
        wall_time_sec=1.0,
        peak_rss_gb=0.1,
        history_path=None,
    )


def _judged_entry() -> case_manifest.CaseManifestEntry:
    entry = case_manifest.load_case_manifest().by_key()[KEY]
    assert entry.artifact is not None
    return dataclasses.replace(
        entry,
        reference_status="mesa_26_4_1_artifact",
        reference_path=entry.artifact.reference,
    )


def test_verdict_is_no_reference_yet_while_status_is_unchanged(
    ref_text: str, spec: artifact_compare.ArtifactSpec
) -> None:
    """Declaring the table does not judge a `no_reference_yet` entry."""
    entry = dataclasses.replace(
        case_manifest.load_case_manifest().by_key()[KEY],
        reference_status="no_reference_yet",
        reference_path="",
    )
    cmp = artifact_compare.compare_artifact(spec, ref_text, ref_text)
    rec = harness_runner.evaluate_case_record(
        entry, {}, _outcome(), artifact_comparison=cmp
    )
    assert rec.verdict == "no reference yet"
    assert rec.artifact_comparison == cmp


def test_live_manifest_uses_mesa_26_4_1_artifact_with_declared_gap(
    ref_text: str, spec: artifact_compare.ArtifactSpec
) -> None:
    """Live manifest judges `rsp_check_2nd_crossing` under gap `#162`."""
    manifest = case_manifest.load_case_manifest()
    gaps = case_manifest.load_known_gaps()
    entry = manifest.by_key()[KEY]
    assert entry.reference_status == "mesa_26_4_1_artifact"
    assert entry.artifact is not None
    assert entry.reference_path == entry.artifact.reference
    assert gaps[KEY].issue == "#162"
    assert "#606" in gaps[KEY].reason

    miss_cmp = artifact_compare.compare_artifact(spec, ref_text, None)
    miss_rec = harness_runner.evaluate_case_record(
        entry, gaps, _outcome(), artifact_comparison=miss_cmp
    )
    assert miss_rec.verdict == "declared gap (#162)"
    assert not miss_rec.is_undeclared_failure
    assert not miss_rec.is_stale_known_gap

    pass_cmp = artifact_compare.compare_artifact(spec, ref_text, ref_text)
    pass_rec = harness_runner.evaluate_case_record(
        entry, gaps, _outcome(), artifact_comparison=pass_cmp
    )
    assert pass_rec.verdict == "pass"
    assert pass_rec.is_stale_known_gap

    _, _, records = case_report.build_manifest_inventory_records()
    summary = case_report.summarize_records(manifest, records, gaps)
    assert not summary["undeclared_failures"]
    stale = [str(item) for item in summary["stale_known_gaps"]]
    assert not any("rsp_check_2nd_crossing" in item for item in stale)
    row_rec = next(r for r in records if (r.suite, r.case) == KEY)
    assert row_rec.verdict == "declared gap (#162)"


def test_verdict_passes_when_artifact_steps_and_termination_agree(
    ref_text: str, spec: artifact_compare.ArtifactSpec
) -> None:
    """Agreement in every field, 0 steps and the hook exit give `pass`."""
    cmp = artifact_compare.compare_artifact(spec, ref_text, ref_text)
    rec = harness_runner.evaluate_case_record(
        _judged_entry(), {}, _outcome(), artifact_comparison=cmp
    )
    assert rec.verdict == "pass"
    assert not rec.is_undeclared_failure


def test_verdict_fails_with_first_field_when_one_field_is_out(
    ref_text: str, spec: artifact_compare.ArtifactSpec
) -> None:
    """One field out of tolerance is an undeclared failure naming it."""
    run = ref_text.replace("5046.6949595302", "5046.6949595312", 1)
    cmp = artifact_compare.compare_artifact(spec, ref_text, run)
    rec = harness_runner.evaluate_case_record(
        _judged_entry(), {}, _outcome(), artifact_comparison=cmp
    )
    assert rec.verdict == "fail (undeclared)" and rec.is_undeclared_failure
    assert "edge[Teff_red_edge]" in rec.error_message


def test_verdict_fails_when_artifact_is_missing(
    ref_text: str, spec: artifact_compare.ArtifactSpec
) -> None:
    """No file written means no pass."""
    cmp = artifact_compare.compare_artifact(spec, ref_text, None)
    rec = harness_runner.evaluate_case_record(
        _judged_entry(), {}, _outcome(), artifact_comparison=cmp
    )
    assert rec.verdict == "fail (undeclared)"
    assert "not written" in rec.error_message


def test_verdict_fails_on_wrong_termination_even_with_matching_file(
    ref_text: str, spec: artifact_compare.ArtifactSpec
) -> None:
    """The run must end as the manifest requires."""
    cmp = artifact_compare.compare_artifact(spec, ref_text, ref_text)
    rec = harness_runner.evaluate_case_record(
        _judged_entry(),
        {},
        _outcome(term="max_model_number"),
        artifact_comparison=cmp,
    )
    assert rec.verdict == "fail (undeclared)"
    assert not rec.termination_matches


def test_verdict_fails_when_the_run_takes_steps_and_says_so(
    ref_text: str, spec: artifact_compare.ArtifactSpec
) -> None:
    """Both sides take 0 steps; a run with steps is reported explicitly."""
    cmp = artifact_compare.compare_artifact(spec, ref_text, ref_text)
    rec = harness_runner.evaluate_case_record(
        _judged_entry(), {}, _outcome(steps=3), artifact_comparison=cmp
    )
    assert rec.verdict == "fail (undeclared)"
    assert "run took 3 steps, MESA 0" in rec.error_message


def test_artifact_comparison_survives_the_stored_row_round_trip(
    ref_text: str, spec: artifact_compare.ArtifactSpec
) -> None:
    """`record_to_dict` keeps the comparison that shard merge re-judges."""
    run = ref_text.replace("5046.6949595302", "5046.6949595312", 1)
    cmp = artifact_compare.compare_artifact(spec, ref_text, run)
    rec = harness_runner.evaluate_case_record(
        _judged_entry(), {}, _outcome(), artifact_comparison=cmp
    )
    row = case_report.record_to_dict(rec)
    assert row["artifact_comparison"]["first_failure"]["label"] == (
        "edge[Teff_red_edge]"
    )


def test_committed_reference_matches_its_provenance_checksums() -> None:
    """The committed copy of MESA's file is the one the bundle lists."""
    import hashlib
    import tomllib

    ref_dir = (
        REPO_ROOT
        / "benchmarks/history_harness/reference_artifacts/star"
        / "rsp_check_2nd_crossing"
    )
    with (ref_dir / "PROVENANCE.toml").open("rb") as fh:
        prov = tomllib.load(fh)
    for name in ("7B_2nd_crossing.data", "models_stdout.txt"):
        digest = hashlib.sha256((ref_dir / name).read_bytes()).hexdigest()
        assert digest == prov["files"][name]
    assert prov["source"]["mesa_release"] == "26.4.1"
