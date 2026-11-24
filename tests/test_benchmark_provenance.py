# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for raw MESA `26.4.1` benchmark captures and rendered docs."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Any, cast

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_RAW_DIR = _REPO_ROOT / "benchmarks" / "raw" / "26.4.1"
_BENCH_FIG_SCRIPTS: tuple[str, ...] = (
    "build_docs_figures.py",
    "build_docs_figures_wave2.py",
    "build_docs_figures_wave3a.py",
    "build_docs_figures_wave3b.py",
    "build_docs_figures_wave4.py",
    "build_docs_figures_wave5.py",
)
_RENDERED_DOC_PATHS: tuple[pathlib.Path, ...] = (
    _REPO_ROOT / "docs" / "benchmarks" / "kernel-scorecards.md",
    _REPO_ROOT / "docs" / "benchmarks" / "overview.md",
    _REPO_ROOT / "docs" / "design" / "wave-1-benchmark-report.md",
    _REPO_ROOT / "docs" / "design" / "wave-2-benchmark-report.md",
    _REPO_ROOT / "docs" / "design" / "wave-3a-benchmark-report.md",
    _REPO_ROOT / "docs" / "design" / "wave-3b-benchmark-report.md",
    _REPO_ROOT / "docs" / "design" / "wave-4-benchmark-report.md",
    _REPO_ROOT / "docs" / "design" / "wave-5-benchmark-report.md",
)


def _load_script(mod_name: str, script_rel: str) -> Any:
    path = _REPO_ROOT / script_rel
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_provenance_schema_and_comparator_taxonomy_match_workloads() -> None:
    """`provenance.json` declares honest flags and a 28-workload split."""
    raw_mod = _load_script("raw_benchmarks", "scripts/docs/raw_benchmarks.py")
    prov = json.loads((_RAW_DIR / "provenance.json").read_text("utf-8"))

    assert prov["ci_continuous_verification"] is False
    hw = cast(dict[str, object], prov["hardware"])
    tc = cast(dict[str, object], prov["toolchains"])
    assert str(hw["cpu_model"]).strip() != ""
    assert str(hw["gpu_model"]).strip() != ""
    assert tc["mesa_release"] == "r26.4.1"

    comp = cast(dict[str, object], prov["comparators"])
    groups = (
        ("mesa_26_4_1_all_stages", 21),
        ("mesa_26_4_1_stages_0_1_reference_fortran_stages_2_3", 3),
        ("hybrid_mesa_helpers_and_reference_fortran", 3),
        ("standalone_reference_fortran_non_mesa", 1),
    )
    seen: list[str] = []
    for key, expected_len in groups:
        names = [str(x) for x in cast(list[object], comp[key])]
        assert len(names) == expected_len
        seen.extend(names)

    assert len(seen) == 28
    assert set(seen) == set(raw_mod.WORKLOAD_TO_WAVE.keys())
    assert comp["total_workloads"] == 28
    assert comp["total_records_per_suite"] == 84


def test_raw_benchmarks_load_complete_84_record_suites() -> None:
    """`load_raw_benchmarks` yields aligned 84-point suites and `bcyclic`."""
    raw_mod = _load_script("raw_benchmarks", "scripts/docs/raw_benchmarks.py")
    suite = raw_mod.load_raw_benchmarks(repo_root=_REPO_ROOT)

    assert len(suite.fortran) == 84
    assert len(suite.rust) == 84
    assert len(suite.python) == 84
    assert set(suite.fortran.keys()) == set(suite.rust.keys())
    assert set(suite.fortran.keys()) == set(suite.python.keys())
    assert set(suite.fortran_bcyclic.keys()) == {128, 1024, 4096}


def test_raw_captures_and_rendered_docs_contain_zero_bad_tokens() -> None:
    """Committed raw files and benchmark docs avoid leaked paths and junk."""
    forbidden_common: tuple[str, ...] = (
        "/" + "home/",
        "/" + "tmp/",
        "/" + "mnt/",
        "." + "venvs",
        "file" + "://",
        "\u2014",
    )
    overflow_token = "*" * 7
    checked_paths = [*sorted(_RAW_DIR.iterdir()), *_RENDERED_DOC_PATHS]
    for path in checked_paths:
        assert path.is_file(), f"Missing provenance artifact: {path}"
        text = path.read_text("utf-8")
        rel = path.relative_to(_REPO_ROOT)
        for token in forbidden_common:
            assert token not in text, f"Forbidden token {token!r} in {rel}"
        if path.suffix in {".txt", ".json"}:
            assert overflow_token not in text, (
                f"Fortran field overflow {overflow_token!r} in {rel}"
            )


def test_waves_1_to_3b_checksums_agree_across_all_57_operating_points() -> None:
    """Waves 1-3b Fortran, Rust, and Python/JAX checksums agree cleanly."""
    raw_mod = _load_script("raw_benchmarks", "scripts/docs/raw_benchmarks.py")
    suite = raw_mod.load_raw_benchmarks(repo_root=_REPO_ROOT)
    w1_to_3b = {"wave1", "wave2", "wave3a", "wave3b"}

    checked = 0
    for key, f_rec in suite.fortran.items():
        if f_rec.wave not in w1_to_3b:
            continue
        r_rec = suite.rust[key]
        p_rec = suite.python[key]
        denom = abs(f_rec.checksum)
        rel_rf = abs(r_rec.checksum - f_rec.checksum) / denom
        rel_pf = abs(p_rec.checksum - f_rec.checksum) / denom
        assert rel_rf <= 1.0e-11, f"Rust vs Fortran mismatch at {key}: {rel_rf}"
        assert rel_pf <= 1.0e-8, f"JAX vs Fortran mismatch at {key}: {rel_pf}"
        checked += 1

    assert checked == 57


def test_rendered_benchmark_pages_match_raw_captures_cleanly() -> None:
    """`render_benchmark_pages.py --check` confirms zero documentation drift."""
    render_mod = _load_script(
        "render_benchmark_pages", "scripts/docs/render_benchmark_pages.py"
    )
    assert render_mod.main(["--repo-root", str(_REPO_ROOT), "--check"]) == 0


def test_figure_builders_import_raw_benchmarks_without_legacy_sha() -> None:
    """Benchmark plotting scripts read `raw_benchmarks` with no `0b2e6bb`."""
    legacy_sha = "0b2e" + "6bb"
    for name in _BENCH_FIG_SCRIPTS:
        text = (_REPO_ROOT / "scripts" / name).read_text("utf-8")
        assert "raw_benchmarks" in text, f"{name} must import raw_benchmarks"
        assert legacy_sha not in text, f"{name} still references legacy sha"


def test_eos_oracles_driver_and_table_record_mesa_26_4_1_provenance() -> None:
    """`eos_oracles` driver and TSV record MESA `26.4.1` provenance cleanly."""
    oracles_dir = _REPO_ROOT / "benchmarks" / "fortran_mesa" / "eos_oracles"
    required_files = (
        "Makefile",
        "README.md",
        "gen_eos_oracles.f90",
        "mesa_26_4_1_eos_oracles.tsv",
    )
    forbidden_tokens: tuple[str, ...] = (
        "/" + "home/",
        "/" + "tmp/",
        "/" + "mnt/",
        "." + "venvs",
        "file" + "://",
        "\u2014",
        "*" * 7,
    )
    for rel_name in required_files:
        path = oracles_dir / rel_name
        assert path.is_file(), f"Missing EOS oracle file: {path}"
        content = path.read_text("utf-8")
        for token in forbidden_tokens:
            assert token not in content, (
                f"Forbidden token {token!r} in {path.relative_to(_REPO_ROOT)}"
            )

    tsv_text = (oracles_dir / "mesa_26_4_1_eos_oracles.tsv").read_text("utf-8")
    for expected_header in (
        "# FlexMESA upstream Fortran MESA 26.4.1 EOS module oracle table",
        "# mesa_version_number=r26.4.1",
        "# mesa_git_tag=26.4.1",
        "# mesa_git_commit=c8c202662922a22b3efefa824473ad615c623afd",
        "# mesasdk_version=x86_64-linux-26.6.1",
        "# math_backend=CRMATH",
        "# compiler=GCC version 15.2.0",
        "# fflags=-march=x86-64",
    ):
        assert expected_header in tsv_text, f"Missing header: {expected_header}"

    prefixes = {
        line.split("\t", 1)[0]
        for line in tsv_text.splitlines()
        if line and not line.startswith("#")
    }
    assert prefixes == {"DFERMI", "ALFA", "HELM", "EOS", "INV", "IERR"}


def test_new_mesa_26_4_1_oracles_record_provenance() -> None:
    """`turb_tdc`, `eps_mdot`, and `net` oracle suites record provenance."""
    forbidden_tokens: tuple[str, ...] = (
        "/" + "home/",
        "/" + "tmp/",
        "/" + "mnt/",
        "." + "venvs",
        "file" + "://",
        "\u2014",
        "*" * 7,
    )
    required_headers: tuple[str, ...] = (
        "# mesa_version: r26.4.1",
        "# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd",
        "# mesasdk_version: x86_64-linux-26.6.1",
        "# math_backend: CRMATH",
        "# compiler: GCC version",
        "# fflags: -march=x86-64",
    )
    specs: tuple[tuple[str, str, str], ...] = (
        (
            "turb_tdc_oracles",
            "gen_turb_tdc_oracles.f90",
            "mesa_26_4_1_turb_tdc_oracles.tsv",
        ),
        (
            "eps_mdot_oracles",
            "gen_eps_mdot_oracles.f90",
            "mesa_26_4_1_eps_mdot_oracles.tsv",
        ),
        (
            "net_oracles",
            "gen_net_oracles.f90",
            "mesa_26_4_1_net_oracles.tsv",
        ),
    )
    for folder, driver, tsv_name in specs:
        oracles_dir = _REPO_ROOT / "benchmarks" / "fortran_mesa" / folder
        for rel_name in ("Makefile", "README.md", driver, tsv_name):
            path = oracles_dir / rel_name
            assert path.is_file(), f"Missing oracle file: {path}"
            content = path.read_text("utf-8")
            for token in forbidden_tokens:
                assert token not in content, (
                    f"Forbidden token {token!r} in "
                    f"{path.relative_to(_REPO_ROOT)}"
                )
        tsv_text = (oracles_dir / tsv_name).read_text("utf-8")
        for expected_header in required_headers:
            assert expected_header in tsv_text, (
                f"Missing {expected_header!r} in {folder}/{tsv_name}"
            )
