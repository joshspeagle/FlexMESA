# SPDX-License-Identifier: GPL-3.0-or-later
"""Manifest terminations agree with MESA 26.4.1's own output (`#604`)."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType

from flexmesa.testing import case_manifest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "mesa_terminations_under_test",
        REPO_ROOT / "scripts" / "audit" / "mesa_terminations.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_manifest_terminations_match_mesa_ledger_and_exceptions() -> None:
    """Every case equals MESA's final code or has a verified exception."""
    mod = _load()
    failures, _ = mod.compare(REPO_ROOT)
    assert failures == []
    status, errors = mod.reproducible(REPO_ROOT)
    assert errors == []
    assert status in (None, True)


def test_ledger_covers_every_manifest_case() -> None:
    """The ledger has one row per manifest case."""
    mod = _load()
    runs = mod._ledger_runs(REPO_ROOT)
    cases = {
        (e.suite, e.case) for e in case_manifest.load_case_manifest().cases
    }
    assert set(runs) == cases


def test_split_stages_assigns_codes_to_the_stage_they_end() -> None:
    """A star code precedes its runtime line; a binary code follows it."""
    mod = _load()
    star = (
        "termination code: max_model_number\n"
        "  runtime (minutes), retries, steps   0.01   0   50\n"
        "  runtime (minutes), retries, steps   0.90   0   7216\n"
    )
    assert mod.split_stages(star)[:2] == ([50, 7216], ["max_model_number", ""])
    binary = (
        "  runtime (minutes), retries, steps   10.01   0   1295\n"
        "termination code: custom message\n"
    )
    assert mod.split_stages(binary)[:2] == ([1295], ["custom message"])


def _run(**over: object) -> dict[str, object]:
    row: dict[str, object] = {
        "run_status": "ok",
        "stage_codes": [""],
        "evidence": ["msg"],
        "source_checked": True,
    }
    row.update(over)
    return row


def _exc(**over: str) -> dict[str, str]:
    row = {
        "manifest": "extras_finish_step",
        "kind": "hook_without_code",
        "source": "synthetic-source:3",
        "source_text": "terminate",
        "message": "msg",
    }
    row.update(over)
    return row


def test_exception_without_verifiable_evidence_fails() -> None:
    """Missing message, unchecked source, or a stale manifest value fail."""
    mod = _load()
    ok = mod._check_exception("s/c", "extras_finish_step", _run(), _exc(), "")
    assert ok == []
    assert mod._check_exception(
        "s/c", "extras_finish_step", _run(evidence=[]), _exc(), ""
    )
    assert mod._check_exception(
        "s/c", "extras_finish_step", _run(source_checked=False), _exc(), ""
    )
    assert mod._check_exception("s/c", "max_age", _run(), _exc(), "")
    assert mod._check_exception(
        "s/c", "extras_finish_step", _run(), _exc(kind="bogus"), ""
    )
    # A hook exception is wrong when MESA did print a code.
    assert mod._check_exception(
        "s/c", "extras_finish_step", _run(), _exc(), "max_age"
    )
    # no_mesa_run is wrong when the MESA run completed.
    assert mod._check_exception(
        "s/c",
        "max_age",
        _run(),
        _exc(kind="no_mesa_run", manifest="max_age"),
        "",
    )
