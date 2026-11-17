# SPDX-License-Identifier: LGPL-3.0-or-later
"""Non-synthetic-figure and generated-benchmark-page regression gate (`#142`).

Verifies that:
1. The pre-audit synthetic MESA I-VI paper figure scripts and assets remain
   deleted (`scripts/mesa_paper_figures/`, `docs/assets/figures/mesa_papers/`).
2. `scripts/build_docs_figures.py` contains no synthetic paper-atlas hook.
3. `docs/benchmarks/mesa_paper_figures.md` reflects the honest `#173` roadmap
   with zero `"Verified Value"` tables, `ledger.csv` references, or links to
   deleted `mesa_papers/` PNGs.
4. Generated documentation pages (`docs/benchmarks/known-gaps.md` and
   `docs/benchmarks/coverage-map.md`) stay synchronized with repository data.
5. Deterministic edge cases (non-zero verified LOC column sums, mutated gap
   counts, empty within-tolerance lists, `--check` exit codes, and zero
   dependence on `dashboard.json`) hold on synthetic repositories.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Protocol, cast

from flexmesa.testing import history_compare

ROOT = pathlib.Path(__file__).resolve().parents[1]


class _KnownGapsRenderer(Protocol):
    def render_known_gaps_markdown(self, repo_root: pathlib.Path) -> str: ...

    def main(self, argv: list[str] | None = None) -> int: ...


class _CoverageMapRenderer(Protocol):
    def render_coverage_map_markdown(self, repo_root: pathlib.Path) -> str: ...

    def main(self, argv: list[str] | None = None) -> int: ...


def _load_renderer_module(rel_path: str, mod_name: str) -> object:
    docs_scripts_dir = str(ROOT / "scripts" / "docs")
    if docs_scripts_dir not in sys.path:
        sys.path.insert(0, docs_scripts_dir)
    script_path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_harness_tree(
    root: pathlib.Path,
    *,
    cases_toml: str,
    gaps_toml: str,
    baseline_cases: list[dict[str, object]],
) -> None:
    hh_dir = root / "benchmarks" / "history_harness"
    hh_dir.mkdir(parents=True, exist_ok=True)
    header = (
        '[metadata]\nmesa_release = "26.4.1"\nzenodo_record = "19722306"\n'
        '[defaults.star]\nindependent_var = "star_age"\n'
        'uses_library_extras = true\nlibrary_extras_issue = "#148"\n'
        "[defaults.star.final_tolerances]\nlog_L = 0.001\n"
        "[defaults.star.track_tolerances]\nlog_L = 0.005\n"
        '[defaults.binary]\nindependent_var = "age"\n'
        'uses_library_extras = true\nlibrary_extras_issue = "#148"\n'
        "[defaults.binary.final_tolerances]\nrel_age = 0.001\n"
        "[defaults.binary.track_tolerances]\nstar_1_mass = 0.002\n"
        '[defaults.astero]\nindependent_var = "star_age"\n'
        'uses_library_extras = true\nlibrary_extras_issue = "#148"\n'
        "[defaults.astero.final_tolerances]\nlog_L = 0.001\n"
        "[defaults.astero.track_tolerances]\nlog_L = 0.005\n\n"
    )
    (hh_dir / "case_manifest.toml").write_text(
        header + cases_toml, encoding="utf-8"
    )
    (hh_dir / "known_gaps.toml").write_text(gaps_toml, encoding="utf-8")
    (hh_dir / "baseline_run_26_4_1.json").write_text(
        json.dumps({"summary": {}, "cases": baseline_cases}),
        encoding="utf-8",
    )


def _write_minimal_coverage_tree(root: pathlib.Path) -> None:
    _write_minimal_harness_tree(
        root,
        cases_toml=(
            '[[cases]]\nsuite = "star"\ncase = "c1"\n'
            'reference_status = "mesa_26_4_1"\nreference_path = "ref.data"\n'
            'required_termination = "max_model_number"\n'
        ),
        gaps_toml=(
            '[[gaps]]\nsuite = "star"\ncase = "c1"\nissue = "#152"\n'
            'reason = "synthetic gap"\n'
        ),
        baseline_cases=[],
    )
    dash_dir = root / "benchmarks" / "dashboard"
    dash_dir.mkdir(parents=True, exist_ok=True)
    # Deliberately omit dashboard.json to guarantee zero dependency on it.
    (dash_dir / "port_manifest.toml").write_text(
        "schema_version = 1\n[metadata]\nschema_version = 1\n"
        'upstream_version = "r26.4.1"\nupstream_commit = "abcdef1234567"\n'
        'archive_md5 = "0123456789abcdef0123456789abcdef"\n\n'
        '[[files]]\nupstream_path = "star/private/a.f90"\nmodule = "star"\n'
        'upstream_lines = 100\nupstream_sha256 = "' + "0" * 64 + '"\n'
        'status = "absent"\n\n'
        '[[files]]\nupstream_path = "star/private/b.f90"\nmodule = "star"\n'
        'upstream_lines = 50\nupstream_sha256 = "' + "1" * 64 + '"\n'
        'status = "stub"\n\n'
        '[[files]]\nupstream_path = "star/private/c.f90"\nmodule = "star"\n'
        'upstream_lines = 40\nupstream_sha256 = "' + "2" * 64 + '"\n'
        'status = "partial"\n\n'
        '[[files]]\nupstream_path = "star/private/d.f90"\nmodule = "star"\n'
        'upstream_lines = 30\nupstream_sha256 = "' + "3" * 64 + '"\n'
        'status = "ported_unverified"\n'
        'flexmesa_files = ["crates/flexmesa-star/src/d.rs"]\n'
        "[files.routine_map]\n"
        'sub_d = "crates/flexmesa-star/src/d.rs::sub_d"\n\n'
        '[[files]]\nupstream_path = "star/private/e.f90"\nmodule = "star"\n'
        'upstream_lines = 20\nupstream_sha256 = "' + "4" * 64 + '"\n'
        'status = "kernel_reference_vectors"\n'
        'flexmesa_files = ["crates/flexmesa-star/src/e.rs"]\n'
        'evidence_artifacts = ["art.txt"]\nevidence_tests = ["test_e.py"]\n'
        "[files.routine_map]\n"
        'sub_e = "crates/flexmesa-star/src/e.rs::sub_e"\n\n'
        '[[files]]\nupstream_path = "const/public/c.f90"\nmodule = "const"\n'
        'upstream_lines = 60\nupstream_sha256 = "' + "5" * 64 + '"\n'
        'status = "case_level_evidence"\n'
        'flexmesa_files = ["crates/flexmesa-const/src/lib.rs"]\n'
        'evidence_artifacts = ["art_c.txt"]\nevidence_tests = ["test_c.py"]\n'
        'harness_cases = ["star/c1"]\n'
        '[files.routine_map]\nsub_c = "crates/flexmesa-const/src/lib.rs::c"\n',
        encoding="utf-8",
    )

    audit_dir = root / "notes" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "defaults_diff_26_4_1.md").write_text(
        "## Summary by Namelist\n"
        "| star &controls | 100 | 40 | 35 | 5 | 0 | 60 |\n"
        "| star &star_job | 50 | 10 | 8 | 2 | 0 | 40 |\n",
        encoding="utf-8",
    )
    (audit_dir / "coverage-map.md").write_text(
        "## 8. Totals\n"
        "| status | upstream LOC | share of all 100 | "
        "share excluding plotting (100) |\n"
        "| :-- | --: | --: | --: |\n"
        "| PORTED | 20 | 20% | 20% |\n"
        "| PARTIAL | 30 | 30% | 30% |\n"
        "| STUB | 10 | 10% | 10% |\n"
        "| ABSENT | 40 | 40% | 40 = 40% |\n",
        encoding="utf-8",
    )

    vec_dir = root / "benchmarks" / "fortran_mesa" / "math_hex_vectors"
    vec_dir.mkdir(parents=True, exist_ok=True)
    (vec_dir / "mesa_26_4_1_math_vectors.txt").write_text(
        "# header\n[unary:safe_sqrt]\n0000 0000\n0001 0001\n0002 0002\n",
        encoding="utf-8",
    )

    hex_dir = root / "tests" / "fixtures" / "golden" / "module_hex"
    hex_dir.mkdir(parents=True, exist_ok=True)
    (hex_dir / "summary.json").write_text(
        json.dumps(
            {
                "total_exact_values_captured": 10,
                "modules": [
                    {"module": "math", "exact_values_captured": 5},
                    {"module": "const", "exact_values_captured": 5},
                ],
            }
        ),
        encoding="utf-8",
    )
    wf_dir = root / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "ci.yml").write_text("name: CI\n", encoding="utf-8")


def test_synthetic_mesa_paper_figures_and_ledgers_removed() -> None:
    """Removed synthetic paper scripts and assets stay absent."""
    assert not (ROOT / "scripts" / "mesa_paper_figures").exists()
    assert not (ROOT / "docs" / "assets" / "figures" / "mesa_papers").exists()

    build_figs = (ROOT / "scripts" / "build_docs_figures.py").read_text(
        encoding="utf-8"
    )
    assert "mesa_paper_figures" not in build_figs
    assert "generate_all_mesa_paper_figures" not in build_figs


def test_mesa_paper_figures_page_is_non_synthetic_roadmap() -> None:
    """`mesa_paper_figures.md` describes the `#173` roadmap cleanly."""
    page = ROOT / "docs" / "benchmarks" / "mesa_paper_figures.md"
    assert page.is_file()
    text = page.read_text(encoding="utf-8")
    assert "#173" in text
    assert "\u2014" not in text
    for forbidden in (
        "Verified Value",
        "ledger.csv",
        "assets/figures/mesa_papers/",
        "reproduce_mesa_",
    ):
        assert forbidden not in text


def test_generated_known_gaps_and_coverage_map_pages_are_fresh() -> None:
    """Generated `known-gaps.md` and `coverage-map.md` match repo data."""
    gaps_mod = cast(
        _KnownGapsRenderer,
        _load_renderer_module(
            "scripts/docs/render_known_gaps_page.py",
            "render_known_gaps_page",
        ),
    )
    cov_mod = cast(
        _CoverageMapRenderer,
        _load_renderer_module(
            "scripts/docs/render_coverage_map_page.py",
            "render_coverage_map_page",
        ),
    )

    gaps_path = ROOT / "docs" / "benchmarks" / "known-gaps.md"
    cov_path = ROOT / "docs" / "benchmarks" / "coverage-map.md"
    assert gaps_path.is_file()
    assert cov_path.is_file()

    expected_gaps = gaps_mod.render_known_gaps_markdown(ROOT)
    expected_cov = cov_mod.render_coverage_map_markdown(ROOT)
    assert gaps_path.read_text(encoding="utf-8") == expected_gaps
    assert cov_path.read_text(encoding="utf-8") == expected_cov
    assert gaps_mod.main(["--repo-root", str(ROOT), "--check"]) == 0
    assert cov_mod.main(["--repo-root", str(ROOT), "--check"]) == 0


def test_coverage_map_column_sums_and_no_dashboard_json(
    tmp_path: pathlib.Path,
) -> None:
    """Verify `Absent+Stub+Partial+Ported(Unverified)+Verified == LOC`."""
    _write_minimal_coverage_tree(tmp_path)
    cov_mod = cast(
        _CoverageMapRenderer,
        _load_renderer_module(
            "scripts/docs/render_coverage_map_page.py",
            "render_coverage_map_page",
        ),
    )
    rendered = cov_mod.render_coverage_map_markdown(tmp_path)
    assert "Rust LOC" not in rendered
    assert "Python/JAX LOC" not in rendered
    assert "3 vectors" in rendered
    assert "18,172" not in rendered
    assert "474,062" not in rendered

    checked_totals = 0
    for line in rendered.splitlines():
        if line.startswith(
            ("| `star` |", "| `const` |", "| **Core", "| **All")
        ):
            cols = [
                c.replace("*", "").replace(",", "").strip()
                for c in line.strip("|").split("|")
            ]
            up_loc = int(cols[1])
            absent = int(cols[2])
            stub = int(cols[3])
            partial = int(cols[4])
            ported_unverified = int(cols[5])
            verified = int(cols[6])
            assert (
                absent + stub + partial + ported_unverified + verified == up_loc
            )
            checked_totals += 1
    assert checked_totals == 4


def test_known_gaps_mutated_counts_and_within_tol_guard(
    tmp_path: pathlib.Path,
) -> None:
    """Verify dynamic snippet counts and `within_tol_non_pass` formatting."""
    gaps_mod = cast(
        _KnownGapsRenderer,
        _load_renderer_module(
            "scripts/docs/render_known_gaps_page.py",
            "render_known_gaps_page",
        ),
    )
    _write_minimal_harness_tree(
        tmp_path,
        cases_toml=(
            '[[cases]]\nsuite = "star"\ncase = "c1"\n'
            'reference_status = "mesa_26_4_1"\nreference_path = "c1.data"\n'
            'required_termination = "max_model_number"\n\n'
            '[[cases]]\nsuite = "star"\ncase = "c2"\n'
            'reference_status = "no_reference_yet"\nreference_path = ""\n'
            'required_termination = "max_age"\n'
        ),
        gaps_toml=(
            '[[gaps]]\nsuite = "star"\ncase = "c1"\nissue = "#152"\n'
            'reason = "synthetic fail"\n'
        ),
        baseline_cases=[],
    )
    md_empty = gaps_mod.render_known_gaps_markdown(tmp_path)
    assert "assert len(manifest.cases) == 2" in md_empty
    assert "assert len(gaps) == 1" in md_empty
    assert "102" not in md_empty
    assert (
        "- **Within Numeric Tolerance While Governed as Gaps**: `0` cases."
        in md_empty
    )

    fail_comp = history_compare.HistoryComparison(
        independent_var="star_age",
        ref_rows=10,
        run_rows=10,
        overlap_min=0.0,
        overlap_max=10.0,
        final_diffs={"log_L": 1.0},
        within_final_tolerances=False,
        failed_final_columns=("log_L",),
        track_deviations=(
            history_compare.TrackDeviation(
                column="log_L",
                max_abs_diff=1.0,
                median_abs_diff=0.5,
                x_at_max=5.0,
                tolerance=0.005,
                within_tolerance=False,
            ),
        ),
        within_track_tolerances=False,
        failed_track_columns=("log_L",),
        first_diverging_step=2,
        first_diverging_x=2.0,
        ref_span=10.0,
    )
    comp = history_compare.HistoryComparison(
        independent_var="star_age",
        ref_rows=10,
        run_rows=10,
        overlap_min=0.0,
        overlap_max=10.0,
        final_diffs={"log_L": 0.0},
        within_final_tolerances=True,
        failed_final_columns=(),
        track_deviations=(
            history_compare.TrackDeviation(
                column="log_L",
                max_abs_diff=0.0,
                median_abs_diff=0.0,
                x_at_max=5.0,
                tolerance=0.005,
                within_tolerance=True,
            ),
        ),
        within_track_tolerances=True,
        failed_track_columns=(),
        first_diverging_step=None,
        first_diverging_x=None,
        ref_span=10.0,
    )
    _write_minimal_harness_tree(
        tmp_path,
        cases_toml=(
            '[[cases]]\nsuite = "star"\ncase = "c1"\n'
            'reference_status = "mesa_26_4_1"\nreference_path = "c1.data"\n'
            'required_termination = "max_model_number"\n\n'
            '[[cases]]\nsuite = "star"\ncase = "c3"\n'
            'reference_status = "mesa_26_4_1"\nreference_path = "c3.data"\n'
            'required_termination = "extras_finish_step"\n'
        ),
        gaps_toml=(
            '[[gaps]]\nsuite = "star"\ncase = "c1"\nissue = "#152"\n'
            'reason = "synthetic fail"\n\n'
            '[[gaps]]\nsuite = "star"\ncase = "c3"\nissue = "#148"\n'
            'reason = "extras active"\n'
        ),
        baseline_cases=[
            {
                "suite": "star",
                "case": "c1",
                "executed": True,
                "run_status": "error",
                "observed_termination": "timeout",
                "steps": 5,
                "retries": 0,
                "comparison": history_compare.comparison_to_dict(fail_comp),
            },
            {
                "suite": "star",
                "case": "c3",
                "executed": True,
                "run_status": "ok",
                "observed_termination": "extras_finish_step",
                "steps": 10,
                "retries": 0,
                "comparison": history_compare.comparison_to_dict(comp),
            },
        ],
    )
    md_wtol = gaps_mod.render_known_gaps_markdown(tmp_path)
    assert "assert len(manifest.cases) == 2" in md_wtol
    assert "assert len(gaps) == 2" in md_wtol
    assert "`1` cases (`star/c3` ([#148]" in md_wtol


def test_renderer_cli_check_exit_codes(tmp_path: pathlib.Path) -> None:
    """Verify `--check` returns `1` when missing/drifted and `0` when fresh."""
    _write_minimal_coverage_tree(tmp_path)
    gaps_mod = cast(
        _KnownGapsRenderer,
        _load_renderer_module(
            "scripts/docs/render_known_gaps_page.py",
            "render_known_gaps_page",
        ),
    )
    cov_mod = cast(
        _CoverageMapRenderer,
        _load_renderer_module(
            "scripts/docs/render_coverage_map_page.py",
            "render_coverage_map_page",
        ),
    )
    for mod, fname in (
        (gaps_mod, "known-gaps.md"),
        (cov_mod, "coverage-map.md"),
    ):
        out = tmp_path / "docs" / "benchmarks" / fname
        assert mod.main(["--repo-root", str(tmp_path), "--check"]) == 1
        assert mod.main(["--repo-root", str(tmp_path)]) == 0
        assert mod.main(["--repo-root", str(tmp_path), "--check"]) == 0
        out.write_text("stale\n", encoding="utf-8")
        assert mod.main(["--repo-root", str(tmp_path), "--check"]) == 1
