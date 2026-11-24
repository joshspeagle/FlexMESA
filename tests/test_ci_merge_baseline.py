# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for `scripts/ci/merge_baseline_candidate.py` (`DECISIONS #51`)."""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "ci"
    / "merge_baseline_candidate.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("merge_baseline", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(case: str, status: str, term: str, steps: int) -> dict[str, object]:
    return {
        "suite": "star",
        "case": case,
        "run_status": status,
        "observed_termination": term,
        "steps": steps,
    }


def test_merge_keeps_a_finished_row_over_a_budget_timeout() -> None:
    """A finished committed row survives a candidate that ran out of time."""
    mod = _load()
    baseline = {
        "summary": {},
        "cases": [
            _row("finished", "ok", "max_age", 100),
            _row("errored", "error", "", 0),
            _row("slow", "timeout", "timeout", 50),
            _row("same", "ok", "max_age", 7),
        ],
    }
    candidate = {
        "summary": {},
        "cases": [
            _row("finished", "timeout", "timeout", 60),
            _row("errored", "timeout", "timeout", 12),
            _row("slow", "timeout", "timeout", 40),
            _row("same", "ok", "max_age", 7),
        ],
    }
    merged, kept, replaced = mod.merge(candidate, baseline)
    steps = {row["case"]: row["steps"] for row in merged["cases"]}
    assert steps == {"finished": 100, "errored": 12, "slow": 40, "same": 7}
    assert kept == ["star/finished"]
    assert replaced == ["star/errored", "star/slow"]


def test_merge_rejects_files_with_different_cases() -> None:
    """The two files must list the same cases."""
    mod = _load()
    baseline = {"summary": {}, "cases": [_row("a", "ok", "max_age", 1)]}
    candidate = {"summary": {}, "cases": [_row("b", "ok", "max_age", 1)]}
    with pytest.raises(ValueError, match="different cases"):
        mod.merge(candidate, baseline)
