# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of the input validation behind the ``case-run`` workflow."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
KNOWN = {("star", "1.3M_ms_high_Z"), ("star", "ns_h"), ("binary", "x")}


def _load() -> types.ModuleType:
    path = REPO_ROOT / "scripts" / "ci" / "case_run_inputs.py"
    spec = importlib.util.spec_from_file_location("case_run_inputs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cri = _load()


def test_parse_cases_valid_list_keeps_order_and_drops_duplicates() -> None:
    got = cri.parse_cases("star/ns_h  binary/x star/ns_h", KNOWN)
    assert got == [("star", "ns_h"), ("binary", "x")]


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "ns_h",
        "star/unknown",
        "star/ns_h;id",
        "star/$(id)",
        "../star/ns_h",
        "star/ns_h/extra",
    ],
)
def test_parse_cases_rejects_malformed_or_unknown_tokens(raw: str) -> None:
    with pytest.raises(cri.InputError):
        cri.parse_cases(raw, KNOWN)


def test_parse_cases_rejects_more_than_sixteen_cases() -> None:
    known = {("star", f"c{i}") for i in range(17)}
    raw = " ".join(f"star/c{i}" for i in range(17))
    with pytest.raises(cri.InputError, match="at most 16"):
        cri.parse_cases(raw, known)
    assert len(cri.parse_cases(" ".join(raw.split()[:16]), known)) == 16


def _env(**over: str) -> dict[str, str]:
    env = {
        "CASES": "star/1.3M_ms_high_Z",
        "MODE": "harness",
        "WINDOW": "",
        "TIMEOUT_SEC": "3000",
        "MAX_RSS_GB": "5.5",
    }
    env.update(over)
    return env


def test_plan_builds_matrix_and_timeout_with_45_minute_headroom() -> None:
    out = cri.plan(_env(CASES="star/1.3M_ms_high_Z"), REPO_ROOT)
    assert json.loads(out["matrix"]) == {
        "include": [
            {
                "suite": "star",
                "case": "1.3M_ms_high_Z",
                "id": "star-1.3M_ms_high_Z",
            }
        ]
    }
    assert out["job_minutes"] == "95"


@pytest.mark.parametrize(
    "over",
    [
        {"MODE": "bogus"},
        {"TIMEOUT_SEC": "0"},
        {"TIMEOUT_SEC": "abc"},
        {"TIMEOUT_SEC": "99999"},
        {"MAX_RSS_GB": "nan"},
        {"MAX_RSS_GB": "64"},
        {"WINDOW": "w1"},
        {"MODE": "compare", "WINDOW": "w1;rm"},
    ],
)
def test_plan_rejects_invalid_inputs(over: dict[str, str]) -> None:
    with pytest.raises(cri.InputError):
        cri.plan(_env(**over), REPO_ROOT)


def test_resolve_dump_finds_first_window_dump(tmp_path: pathlib.Path) -> None:
    dump = tmp_path / "mesa_stepdumps" / "star" / "ns_h.stepdump"
    dump.parent.mkdir(parents=True)
    dump.write_text("#steps_per_star: 3\n")
    got = cri.resolve_dump(tmp_path, "star", "ns_h", "", KNOWN)
    assert got == dump


def test_resolve_dump_reports_missing_dump_and_unknown_case(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(cri.InputError, match="no MESA step dump"):
        cri.resolve_dump(tmp_path, "star", "ns_h", "", KNOWN)
    with pytest.raises(cri.InputError, match="manifest"):
        cri.resolve_dump(tmp_path, "star", "nope", "", KNOWN)


def test_window_trace_args_read_stage_and_occurrence_from_the_id() -> None:
    assert cri.window_trace_args("ns_h_s2_k1_m90_n12") == ["--stage", "2"]
    assert cri.window_trace_args("wd_stable_h_burn_s1_k2_m70_n12") == [
        "--occurrence",
        "2",
    ]
    assert cri.window_trace_args("make_planets_s3_k2_m9_n3") == [
        "--stage",
        "3",
        "--occurrence",
        "2",
    ]


def test_window_trace_args_default_window_selects_nothing() -> None:
    assert cri.window_trace_args("") == []
    assert cri.window_trace_args("x_s1_k1_m1_n3") == []


def test_window_trace_args_reject_an_id_without_window_fields() -> None:
    with pytest.raises(cri.InputError):
        cri.window_trace_args("ns_h_window")
