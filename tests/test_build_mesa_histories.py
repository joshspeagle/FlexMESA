# SPDX-License-Identifier: GPL-3.0-or-later
"""Builder logic for reference histories when MESA wrote several files."""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
COLS = ["model_number", "num_zones", "star_age", "log_Teff"]


def _load_builder() -> ModuleType:
    path = REPO_ROOT / "scripts" / "data" / "build_mesa_histories.py"
    spec = importlib.util.spec_from_file_location("_build_mesa_histories", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load_builder()


def _write_history(
    path: pathlib.Path,
    models: list[int],
    columns: list[str] = COLS,
) -> None:
    head = [
        " ".join(f"{i:>10d}" for i in range(1, 4)),
        " ".join(f"{n:>10s}" for n in ("version_number", "compiler", "build")),
        " ".join(f"{n:>10s}" for n in ('"r26.4.1"', '"gfortran"', '"15.2.0"')),
        "",
        " ".join(f"{i:>10d}" for i in range(1, len(columns) + 1)),
        " ".join(f"{n:>10s}" for n in columns),
    ]
    rows = [
        " ".join(f"{v:>10s}" for v in (str(m), "150", f"{m * 0.5}", "3.8"))
        for m in models
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([*head, *rows]) + "\n", encoding="utf-8")


def _models(blob: bytes) -> list[int]:
    return [int(ln.split()[0]) for ln in blob.decode().splitlines()[6:]]


def test_map_history_beside_history_is_a_second_source_in_order(
    tmp_path: pathlib.Path,
) -> None:
    _write_history(tmp_path / "LOGS" / "history.data", [1, 10])
    _write_history(tmp_path / "LOGS" / "map_history.data", [20, 30])
    got = build.history_source_files(tmp_path, "LOGS/history.data", "star")
    assert [p.name for p in got] == ["history.data", "map_history.data"]


def test_run_without_map_history_has_one_source_and_unchanged_output(
    tmp_path: pathlib.Path,
) -> None:
    _write_history(tmp_path / "LOGS" / "history.data", [1, 10, 20])
    paths = build.history_source_files(tmp_path, "LOGS/history.data", "star")
    assert len(paths) == 1
    single = build.thin_history_preserving_mesa_header(paths[0], "star")
    assert single == build.thin_history_preserving_mesa_header(paths, "star")
    assert _models(single) == [1, 10, 20]


def test_two_files_concatenate_in_model_order_under_one_header(
    tmp_path: pathlib.Path,
) -> None:
    _write_history(tmp_path / "a" / "history.data", [1, 10, 20])
    _write_history(tmp_path / "a" / "map_history.data", [30, 40])
    blob = build.thin_history_preserving_mesa_header(
        build.history_source_files(tmp_path / "a", "history.data", "star"),
        "star",
    )
    assert _models(blob) == [1, 10, 20, 30, 40]
    assert blob.decode().splitlines()[2].count('"r26.4.1"') == 1


def test_non_star_suite_ignores_a_map_history_file(
    tmp_path: pathlib.Path,
) -> None:
    _write_history(tmp_path / "history.data", [1])
    _write_history(tmp_path / "map_history.data", [2])
    got = build.history_source_files(tmp_path, "history.data", "binary")
    assert len(got) == 1


def test_differing_header_columns_raise_instead_of_merging(
    tmp_path: pathlib.Path,
) -> None:
    _write_history(tmp_path / "history.data", [1, 10])
    _write_history(
        tmp_path / "map_history.data", [20], columns=["model_number", "extra"]
    )
    with pytest.raises(ValueError, match="header differs"):
        build.thin_history_preserving_mesa_header(
            build.history_source_files(tmp_path, "history.data", "star"),
            "star",
        )


def test_overlapping_model_numbers_across_files_raise(
    tmp_path: pathlib.Path,
) -> None:
    _write_history(tmp_path / "history.data", [1, 10, 20])
    _write_history(tmp_path / "map_history.data", [20, 30])
    with pytest.raises(ValueError, match="does not follow"):
        build.thin_history_preserving_mesa_header(
            build.history_source_files(tmp_path, "history.data", "star"),
            "star",
        )
