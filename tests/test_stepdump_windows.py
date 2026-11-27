# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the mid-run step-dump window manifest and bundle builder (#608)."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType

REPO = pathlib.Path(__file__).resolve().parents[1]
STEPDUMPS = REPO / "benchmarks" / "fortran_mesa" / "stepdumps"


def _load(name: str, path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


regen = _load("regenerate_windows", STEPDUMPS / "regenerate_windows.py")
build = _load(
    "build_mesa_stepdump_windows",
    REPO / "scripts" / "data" / "build_mesa_stepdump_windows.py",
)


def test_manifest_windows_have_unique_ids_and_known_cases() -> None:
    windows = regen.load_windows()
    ids = [regen.window_id(w) for w in windows]
    assert len(ids) == len(set(ids)) > 0
    for w in windows:
        suite, _, case = w["case"].partition("/")
        assert suite in ("star", "binary", "astero") and case
        assert w["stage"] >= 1 and w["occurrence"] >= 1
        assert w["first_model"] >= 1 and w["steps"] >= 1


def test_manifest_holds_no_machine_paths() -> None:
    text = (STEPDUMPS / "windows.toml").read_text(encoding="utf-8")
    for needle in ("/home/", "/tmp/", "/mnt/", "/Users/"):
        assert needle not in text


def test_window_key_of_a_dump_before_occurrence_and_stage_lines() -> None:
    head = {
        "case": "star/ns_h",
        "first_model": "90",
        "steps_per_star": "12",
        "stage": "2 of 2 RUN sections",
    }
    assert build.window_key(head) == ("star/ns_h", 2, 1, 90, 12)
    head = {"case": "star/x", "first_model": "5", "steps_per_star": "3"}
    assert build.window_key(head) == ("star/x", 1, 1, 5, 3)
    assert build.window_key({"case": "star/x", "steps_per_star": "3"}) is None


def test_normalized_drops_unknown_records_and_sets_window_lines() -> None:
    text = "\n".join(
        [
            "# case: star/x",
            "# first_model: 5",
            "# stage: 1 of 1 RUN sections",
            "# produced by: somewhere",
            "RUN",
            "NAMES 1 2 1 1 3 lnd lnT",
            "XTRA 1 5 1 0000000000000000",
            "CALL 1 5 0000000000000000 0000000000000000 0000000000000000 1 1",
            "XTRA2 1 5 1 0000000000000000",
            "DONE",
        ]
    )
    out = build.normalized(text, stage=2, occurrence=3).decode()
    lines = out.splitlines()
    assert lines[:5] == [
        "# case: star/x",
        "# first_model: 5",
        "# occurrence: 3",
        "# stage: 2",
        "# produced by: somewhere",
    ]
    assert [x.split()[0] for x in lines[5:]] == [
        "RUN",
        "NAMES",
        "CALL",
        "DONE",
    ]
