# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for closing a step-dump window at the end of its stage (#608)."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
STEPDUMPS = REPO / "benchmarks" / "fortran_mesa" / "stepdumps"


def _load(name: str, path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rs = _load("run_stepdumps", STEPDUMPS / "run_stepdumps.py")


@pytest.mark.parametrize(
    ("records", "stage", "expected"),
    [
        (["RUN"], 2, "not_started"),
        (["RUN", "CALL 1 5 x"], 1, "running_open"),
        (["RUN", "CALL 1 5 x", "DONE"], 1, "running_closed"),
        (["RUN", "CALL 1 5 x", "RUN"], 1, "ended_open"),
        (["RUN", "CALL 1 5 x", "DONE", "RUN"], 1, "running_closed"),
        (["RUN", "RUN"], 1, "ended_unopened"),
        (["RUN"], 1, "running_closed"),
        (["RUN", "RUN", "CALL 1 9 x"], 2, "running_open"),
    ],
)
def test_stage_window_state_classifies_the_selected_section(
    records: list[str], stage: int, expected: str
) -> None:
    assert rs.stage_window_state(records, stage) == expected


def test_run_appends_done_and_stops_when_the_next_stage_starts(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "x.raw"
    script = tmp_path / "run_case.sh"
    # Stage 1 writes a window and ends; stage 2 starts and would run long.
    script.write_text(
        f'printf "RUN\\nCALL 1 5 x\\nRUN\\n" >> {raw}\nsleep 120\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(rs, "RUN_CASE", script)
    monkeypatch.setattr(rs, "POLL_SEC", 0.1)
    status = rs._run("star/x", {}, 60, raw, 1)
    assert "DONE appended" in status
    assert raw.read_text().splitlines() == ["RUN", "CALL 1 5 x", "RUN", "DONE"]


def test_run_appends_done_when_the_last_stage_exits_with_window_open(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "x.raw"
    script = tmp_path / "run_case.sh"
    script.write_text(
        f'printf "RUN\\nCALL 1 5 x\\n" >> {raw}\n', encoding="utf-8"
    )
    monkeypatch.setattr(rs, "RUN_CASE", script)
    monkeypatch.setattr(rs, "POLL_SEC", 0.1)
    status = rs._run("star/x", {}, 60, raw, 1)
    assert "DONE appended" in status
    assert raw.read_text().splitlines()[-1] == "DONE"


def test_run_does_not_close_a_window_that_never_opened(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "x.raw"
    script = tmp_path / "run_case.sh"
    script.write_text(f'printf "RUN\\n" >> {raw}\n', encoding="utf-8")
    monkeypatch.setattr(rs, "RUN_CASE", script)
    monkeypatch.setattr(rs, "POLL_SEC", 0.1)
    rs._run("star/x", {}, 60, raw, 1)
    assert "DONE" not in raw.read_text().splitlines()
