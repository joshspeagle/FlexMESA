# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for `scripts/mesa_reference/measure_mesa_spread.py`."""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
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


def test_makefile_patch_refuses_noop_and_missing_fcopt(
    tmp_path: pathlib.Path,
) -> None:
    """A variant build that cannot differ from the baseline is refused."""
    mod = _load_script_module()
    header = tmp_path / "makefile_header"
    header.write_text(
        "USE_PGSTAR = YES\nFCopt = -O2 -ftree-vectorize\n", encoding="utf-8"
    )
    assert mod.read_makefile_fcopt(header) == "-O2 -ftree-vectorize"
    with pytest.raises(RuntimeError, match="equals the baseline FCopt"):
        mod._patch_makefile_header(header, "-O2 -ftree-vectorize")
    mod._patch_makefile_header(header, "-O0 -fno-tree-vectorize")
    text = header.read_text(encoding="utf-8")
    assert "FCopt = -O0 -fno-tree-vectorize" in text
    assert "USE_PGSTAR = NO" in text

    no_fcopt = tmp_path / "makefile_header_bad"
    no_fcopt.write_text("USE_PGSTAR = YES\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no 'FCopt =' line"):
        mod._patch_makefile_header(no_fcopt, "-O0")
