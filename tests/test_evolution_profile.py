# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for `#168` stellar evolution profiling artifacts and rendered docs."""

from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import sys
from typing import Any, cast

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PROF_DIR = _REPO_ROOT / "benchmarks" / "evolution_profile" / "26.4.1"
_AUDIT_MD = _REPO_ROOT / "notes" / "audit" / "evolution_profile_26_4_1.md"
_DOCS_MD = _REPO_ROOT / "docs" / "benchmarks" / "evolution-profiling.md"
_EXPECTED_CASES: tuple[str, ...] = (
    "1.3M_ms_high_Z",
    "low_z",
    "high_z",
    "make_metals",
)


def _load_renderer() -> Any:
    path = (
        _REPO_ROOT / "scripts" / "audit" / "render_evolution_profile_report.py"
    )
    spec = importlib.util.spec_from_file_location(
        "render_evolution_profile_report", path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_evolution_profile_report"] = mod
    spec.loader.exec_module(mod)
    return mod


def _read_json(name: str) -> dict[str, Any]:
    return cast(
        dict[str, Any], json.loads((_PROF_DIR / name).read_text("utf-8"))
    )


def test_provenance_declares_hardware_toolchains_and_four_cases() -> None:
    """`provenance.json` records host CPU, compilers, commits, and 4 cases."""
    prov = _read_json("provenance.json")
    assert prov["issue"] == "#168"
    assert prov["ci_continuous_verification"] is False
    assert tuple(prov["selected_cases"]) == _EXPECTED_CASES

    hw = cast(dict[str, Any], prov["hardware"])
    tc = cast(dict[str, Any], prov["toolchains"])
    assert str(hw["cpu_model"]).strip() != ""
    assert int(hw["physical_cores"]) >= 1
    assert float(hw["total_ram_gib"]) > 0.0
    assert tc["mesa_release"] == "r26.4.1"
    assert tc["mesa_commit"] == "c8c2026"
    assert "Fortran" in str(tc["gfortran"])
    assert "rustc" in str(tc["rustc"])
    assert int(tc["omp_num_threads"]) == int(tc["rayon_num_threads"])


def test_profile_artifacts_and_reports_contain_zero_forbidden_tokens() -> None:
    """Committed `#168` JSON, CSV, and markdown files contain zero bad paths."""
    forbidden: tuple[str, ...] = (
        "/" + "home/",
        "/" + "mnt/",
        "/" + "tmp/",
        "." + "venvs",
        "file" + "://",
        "-" + "desktop",
        "\u2014",
    )
    checked = [*sorted(_PROF_DIR.iterdir()), _AUDIT_MD, _DOCS_MD]
    assert len(checked) >= 12
    for path in checked:
        assert path.is_file(), f"Missing #168 artifact: {path}"
        text = path.read_text("utf-8")
        rel = path.relative_to(_REPO_ROOT)
        for tok in forbidden:
            assert tok not in text, f"Forbidden token {tok!r} in {rel}"


def test_phase_breakdown_shares_sum_to_hundred_percent_and_count_calls() -> (
    None
):
    """Per-step subsystem shares sum to 100% and record microphysics calls."""
    strict = _read_json("flexmesa_cases_strict.json")
    cases = cast(list[dict[str, Any]], strict["cases"])
    assert tuple(str(c["case"]) for c in cases) == _EXPECTED_CASES

    share_keys = (
        "eos_share_pct",
        "kap_share_pct",
        "net_and_jacobian_share_pct",
        "turbulence_share_pct",
        "equation_assembly_share_pct",
        "block_solve_share_pct",
        "remesh_share_pct",
        "bookkeeping_and_copies_share_pct",
    )
    for c in cases:
        assert int(c["steps_taken"]) > 100
        assert float(c["evolve_wall_sec"]) > 0.0
        ph = cast(dict[str, Any], c["phases"])
        total_pct = sum(float(ph[k]) for k in share_keys)
        assert abs(total_pct - 100.0) < 0.05
        assert float(ph["eos_evals_per_accepted_step"]) > 0.0
        assert float(ph["kap_evals_per_accepted_step"]) > 0.0
        assert float(ph["net_jac_evals_per_accepted_step"]) > 0.0
        assert float(ph["mlt_evals_per_accepted_step"]) > 0.0
        assert float(ph["linear_solves_per_accepted_step"]) >= 1.0


def test_mesa_26_4_1_and_strict_vs_fast_records_align_across_cases() -> None:
    """Real MESA `26.4.1` and `strict` vs `fast` artifacts cover all 4 cases."""
    mesa = cast(
        list[dict[str, Any]], _read_json("mesa_26_4_1_cases.json")["cases"]
    )
    svf = cast(
        list[dict[str, Any]],
        _read_json("strict_vs_fast_trajectories.json")["cases"],
    )
    assert tuple(str(m["case"]) for m in mesa) == _EXPECTED_CASES
    assert tuple(str(r["case"]) for r in svf) == _EXPECTED_CASES

    for m in mesa:
        assert m["mesa_version_number"] == "r26.4.1"
        assert int(m["steps_taken"]) > 100
        assert float(m["mesa_internal_evolve_sec"]) > 0.0
        assert float(m["evolve_wall_us_per_zone_step"]) > 0.0

    for r in svf:
        assert r["comparison"] == "fast_vs_strict_only"
        assert float(r["strict_evolve_wall_sec"]) > 0.0
        assert float(r["fast_evolve_wall_sec"]) > 0.0
        td = cast(dict[str, Any], r["trajectory_diff"])
        assert abs(float(td["rel_star_age_diff"])) < 0.05

    with (_PROF_DIR / "mesa_26_4_1_cases.csv").open(encoding="utf-8") as fh:
        lines = fh.readlines()
    assert lines[0].startswith("# MESA version: r26.4.1")
    rows = list(csv.DictReader(ln for ln in lines if not ln.startswith("#")))
    assert len(rows) == 4


def test_waste_removal_batching_and_accelerator_estimates_validate() -> None:
    """`#150` git A/B, `N`-star worker pools, and `#166` estimates validate."""
    waste = _read_json("waste_removal_150.json")
    ab = cast(dict[str, Any], waste["git_ab_1_3M_ms_high_Z"])
    assert float(ab["pre_150"]["median_wall_sec"]) > 0.0
    assert float(ab["post_150"]["median_wall_sec"]) > 0.0
    assert len(cast(list[Any], waste["head_component_breakdown"])) == 4

    batch = _read_json("batch_scaling_rust.json")
    pts_2t = cast(list[dict[str, Any]], batch["pool_2_threads"]["points"])
    pts_8t = cast(list[dict[str, Any]], batch["pool_8_threads"]["points"])
    assert [int(p["n_stars"]) for p in pts_2t] == [1, 2, 4, 8, 16]
    assert [int(p["n_stars"]) for p in pts_8t] == [1, 2, 4, 8, 16, 32]
    assert float(pts_8t[-1]["speedup_vs_n1"]) > 2.0

    accel = _read_json("accelerator_grid_estimate.json")
    assert 0.0 < float(accel["stateless_microphysics_fraction"]) < 1.0
    assert float(accel["amdahl_infinite_gpu_micro_speedup_ceiling"]) > 1.0
    assert len(cast(list[Any], accel["grid_projections"])) == 5


def test_rendered_evolution_profile_reports_are_up_to_date() -> None:
    """Tracked audit and docs markdown match `render_reports` byte-for-byte."""
    mod = _load_renderer()
    expected_audit, expected_docs = mod.render_reports(_PROF_DIR)
    assert _AUDIT_MD.read_text("utf-8") == expected_audit
    assert _DOCS_MD.read_text("utf-8") == expected_docs
