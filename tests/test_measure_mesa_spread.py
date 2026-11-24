# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for `scripts/mesa_reference/measure_mesa_spread.py`."""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import stat
import sys
from typing import Any

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT_PATH = (
    _REPO_ROOT / "scripts" / "mesa_reference" / "measure_mesa_spread.py"
)


def _load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "measure_mesa_spread", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_history(
    path: pathlib.Path, rows: list[tuple[int, float, float]]
) -> None:
    lines = [
        " 1 2 3",
        " version_number compiler build_date",
        ' "r26.4.1" "gfortran" "2026"',
        "",
        " 1 2 3",
        " model_number star_age log_L",
    ]
    for mod_num, age, log_l in rows:
        lines.append(f" {mod_num:10d} {age:22.16e} {log_l:22.16e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_and_compare_history_tables_ulp_math(
    tmp_path: pathlib.Path,
) -> None:
    """Measure exact IEEE-754 ULPs, NaN inf spread, and row mismatches."""
    mod = _load_script_module()
    base_p = tmp_path / "base.data"
    alt_p = tmp_path / "alt.data"
    identical_p = tmp_path / "same.data"

    _write_history(base_p, [(1, 1.0, 0.5), (2, 2.0, 1.25)])
    _write_history(identical_p, [(1, 1.0, 0.5), (2, 2.0, 1.25)])
    one_ulp_up = 1.25 + 2.220446049250313e-16
    _write_history(alt_p, [(1, 1.0, 0.5), (2, 2.0, one_ulp_up)])

    same_res = mod.compare_history_tables(
        "star", "synthetic", base_p, identical_p
    )
    assert same_res.files_byte_identical is True
    assert same_res.max_ulp == 0
    assert same_res.zero_ulp_cells == 6

    diff_res = mod.compare_history_tables("star", "synthetic", base_p, alt_p)
    assert diff_res.files_byte_identical is False
    assert diff_res.max_ulp == 1
    assert diff_res.zero_ulp_cells == 5
    assert diff_res.total_cells_compared == 6
    assert diff_res.top_columns_by_ulp[0].column == "log_L"
    assert diff_res.top_columns_by_ulp[0].max_ulp == 1

    # NaN vs finite cell must yield infinite abs/rel spread, and mismatched
    # row counts must appear cleanly in rendered Markdown.
    nan_p = tmp_path / "nan.data"
    _write_history(
        nan_p, [(1, 1.0, 0.5), (2, 2.0, float("nan")), (3, 3.0, 1.5)]
    )
    nan_res = mod.compare_history_tables("star", "nan_case", base_p, nan_p)
    assert math.isinf(nan_res.max_abs_diff)
    assert math.isinf(nan_res.max_rel_diff)
    assert nan_res.rows_baseline == 2
    assert nan_res.rows_variant == 3
    md = mod.render_spread_markdown(
        mod.build_spread_report(
            mod.CompilerConfigSpec("base", "-O2"),
            mod.CompilerConfigSpec("var", "-O0"),
            [nan_res],
            dry_run=False,
        )
    )
    assert "base=2, var=3" in md
    assert "`1` row-count mismatch" in md


def test_dry_run_includes_touch_before_install_and_compare_cli(
    tmp_path: pathlib.Path,
) -> None:
    """Verify --dry-run runs ./touch before ./install and CLI schema."""
    mod = _load_script_module()
    dry_json = tmp_path / "dry.json"
    dry_md = tmp_path / "dry.md"
    rc = mod.main(
        [
            "--dry-run",
            "--mesa-dir",
            str(tmp_path / "mesa"),
            "--mesasdk-root",
            str(tmp_path / "sdk"),
            "--json-out",
            str(dry_json),
            "--markdown-out",
            str(dry_md),
        ]
    )
    assert rc == 0
    payload = json.loads(dry_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["dry_run"] is True
    assert payload["summary"]["cases_compared"] == 0
    assert any(
        "./touch && ./install" in cmd for cmd in payload["planned_commands"]
    )
    assert "Planned Commands" in dry_md.read_text(encoding="utf-8")

    b_p = tmp_path / "b.data"
    v_p = tmp_path / "v.data"
    _write_history(b_p, [(1, 10.0, 0.0)])
    _write_history(v_p, [(1, 10.0, 0.0)])
    cmp_json = tmp_path / "cmp.json"
    rc2 = mod.main(
        [
            "--compare-histories",
            "star/mock",
            str(b_p),
            str(v_p),
            "--json-out",
            str(cmp_json),
        ]
    )
    assert rc2 == 0
    cmp_data = json.loads(cmp_json.read_text(encoding="utf-8"))
    assert cmp_data["summary"]["byte_identical_cases"] == 1
    assert cmp_data["mesa_version"] == "26.4.1"
    assert cmp_data["mesa_version_raw"] == "r26.4.1"
    assert (
        cmp_data["cases"][0]["baseline_header"]["version_number"] == "r26.4.1"
    )
    assert cmp_data["cases"][0]["baseline_header"]["compiler"] == "gfortran"


def test_patch_makefile_header_updates_top_level_fcopt_and_fails_loudly(
    tmp_path: pathlib.Path,
) -> None:
    """Verify _patch_makefile_header ignores FPE block and fails on no-op."""
    mod = _load_script_module()
    header = tmp_path / "makefile_header"
    sample = (
        "USE_PGSTAR = YES\n"
        "FCopt = -O${OPT_LEVEL} -ftree-vectorize\n"
        "ifeq ($(MESA_FPE_CHECKS_ON),1)\n"
        "   FCopt = -O0 -ftree-vectorize\n"
        "endif\n"
    )
    header.write_text(sample, encoding="utf-8")

    prev = mod._patch_makefile_header(
        header, "-O0 -fno-tree-vectorize -ffp-contract=off"
    )
    assert prev == "-O${OPT_LEVEL} -ftree-vectorize"
    patched = header.read_text(encoding="utf-8")
    assert "USE_PGSTAR = NO\n" in patched
    assert "FCopt = -O0 -fno-tree-vectorize -ffp-contract=off\n" in patched
    assert "   FCopt = -O0 -ftree-vectorize\n" in patched

    with pytest.raises(ValueError, match="unchanged"):
        mod._patch_makefile_header(
            header, "-O0 -fno-tree-vectorize -ffp-contract=off"
        )

    no_fcopt = tmp_path / "no_fcopt"
    no_fcopt.write_text("USE_PGSTAR = YES\n", encoding="utf-8")
    with pytest.raises(ValueError, match="FCopt ="):
        mod._patch_makefile_header(no_fcopt, "-O0")

    no_pgstar = tmp_path / "no_pgstar"
    no_pgstar.write_text("FCopt = -O2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="USE_PGSTAR ="):
        mod._patch_makefile_header(no_pgstar, "-O0")


def test_probe_mesa_version_and_toolchain_provenance_rendering(
    tmp_path: pathlib.Path,
) -> None:
    """Verify MESA version and SDK/gfortran probes serialize cleanly."""
    mod = _load_script_module()
    mesa_dir = tmp_path / "mesa"
    with pytest.raises(FileNotFoundError):
        mod.probe_mesa_version(mesa_dir)

    ver_file = mesa_dir / "data" / "version_number"
    ver_file.parent.mkdir(parents=True)
    ver_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty MESA version"):
        mod.probe_mesa_version(mesa_dir)

    ver_file.write_text("r26.4.1\n", encoding="utf-8")
    assert mod.probe_mesa_version(mesa_dir) == ("26.4.1", "r26.4.1")

    sdk_bin = tmp_path / "sdk" / "bin"
    sdk_bin.mkdir(parents=True)
    sdk_ver = sdk_bin / "mesasdk_version"
    sdk_ver.write_text(
        "#!/bin/sh\necho x86_64-linux-26.6.1\n", encoding="utf-8"
    )
    sdk_ver.chmod(sdk_ver.stat().st_mode | stat.S_IXUSR)

    gf = sdk_bin / "gfortran"
    gf.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-dumpmachine" ]; then\n'
        "  echo x86_64-pc-linux-gnu\n"
        "else\n"
        '  echo "GNU Fortran (GCC) 15.2.0"\n'
        '  echo "Copyright (C) 2025 Free Software Foundation"\n'
        "fi\n",
        encoding="utf-8",
    )
    gf.chmod(gf.stat().st_mode | stat.S_IXUSR)

    tc = mod.probe_toolchain(tmp_path / "sdk", mesa_dir, "crmath")
    assert tc.mesasdk_version == "x86_64-linux-26.6.1"
    assert tc.gfortran_version == "GNU Fortran (GCC) 15.2.0"
    assert tc.target_triple == "x86_64-pc-linux-gnu"
    assert tc.math_slot == "crmath"

    report = mod.build_spread_report(
        mod.CompilerConfigSpec(
            "baseline_o2",
            "-O2 -ftree-vectorize",
            makefile_header_fcopt="-O${OPT_LEVEL} -ftree-vectorize",
        ),
        mod.CompilerConfigSpec(
            "variant_alt",
            "-O0 -fno-tree-vectorize -ffp-contract=off",
            makefile_header_fcopt="-O0 -fno-tree-vectorize -ffp-contract=off",
        ),
        [],
        dry_run=False,
        mesa_version="26.4.1",
        mesa_version_raw="r26.4.1",
        toolchain=tc,
    )
    assert report["mesa_version"] == "26.4.1"
    assert report["toolchain"]["gfortran_version"] == "GNU Fortran (GCC) 15.2.0"
    assert (
        report["baseline_config"]["makefile_header_fcopt"]
        == "-O${OPT_LEVEL} -ftree-vectorize"
    )
    md = mod.render_spread_markdown(report)
    assert "# MESA 26.4.1 Compiler-Variant Numerical Spread" in md
    assert "x86_64-linux-26.6.1" in md
    assert "GNU Fortran (GCC) 15.2.0" in md
