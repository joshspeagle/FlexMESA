# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the W1 window-snapshot overlay patch and planner (#684)."""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
from types import ModuleType

REPO = pathlib.Path(__file__).resolve().parents[1]
STEPDUMPS = REPO / "benchmarks" / "fortran_mesa" / "stepdumps"
PLANNER = REPO / "scripts" / "windows" / "plan_windows.py"


def _load(name: str, path: pathlib.Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sp = _load("snapshot_patch", STEPDUMPS / "snapshot_patch.py")
pw = _load("plan_windows", PLANNER)


def _synthetic_run_star_support() -> str:
    edits: list[tuple[str, str, str]] = sp.EDITS
    end_anchor: str = sp.END_ANCHOR
    return "".join(anchor for anchor, _, _ in edits) + end_anchor


def test_snapshot_patch_is_output_only_and_idempotent_guarded() -> None:
    src = _synthetic_run_star_support()
    out = sp.patch_text(src)
    assert out.count("call flexmesa_snap_hook(id, 1, ierr)") == 1
    assert out.count("call flexmesa_snap_hook(id, 2, ierr)") == 1
    assert out.count("call flexmesa_snap_hook(id, 3, ierr)") == 1
    assert out.count("call star_write_model(id, trim(snap_path), ierr)") == 1
    added = [x for x in out.splitlines() if x not in src.splitlines()]
    assign = re.compile(r"^\s*s%\s*\w+(\([^)]*\))?\s*=[^=]")
    assert not [x for x in added if assign.match(x)]
    try:
        sp.patch_text(out)
    except ValueError as exc:
        assert "already patched" in str(exc)
    else:
        raise AssertionError("expected ValueError on double-patch")


def test_short_stage_and_seeded_points_follow_window_rules() -> None:
    short = pw.candidate_points_for_stage("T_tau_gradr", 1, 1, 50)
    assert len(short) == 1
    assert short[0].kind == "short_stage"
    assert short[0].start_model == 1
    assert short[0].seed == pw.stage_seed("T_tau_gradr", 1)

    pts_a = pw.candidate_points_for_stage("accreted_material_j", 1, 1, 452)
    pts_b = pw.candidate_points_for_stage("accreted_material_j", 1, 1, 452)
    assert pts_a == pts_b
    assert [p.kind for p in pts_a] == [
        "start",
        "frac_25",
        "frac_50",
        "frac_75",
        "tail_95",
    ]
    assert pts_a[0].start_model == 1
    assert 90 <= pts_a[1].start_model <= 136
    assert 203 <= pts_a[2].start_model <= 249
    assert 316 <= pts_a[3].start_model <= 362
    assert pts_a[4].start_model == round(0.95 * 452)


def test_compute_window_w_clamps_and_deduplicates() -> None:
    steps = [
        pw.StepRecord(
            stage=1,
            star_id=1,
            model_number=m,
            dt_wall=1.0,
            nz=100,
            nz_old=100,
            generations=2,
            prev_flag=True,
        )
        for m in range(1, 501)
    ]
    w, m_steps, mean_s = pw.compute_window_w(steps, 10)
    assert w == 60
    assert m_steps == 60
    assert abs(mean_s - 1.0) < 1e-9

    fast = [
        pw.StepRecord(1, 1, m, 0.05, 100, 100, 2, True) for m in range(1, 501)
    ]
    w_fast, m_fast, _ = pw.compute_window_w(fast, 1)
    assert w_fast == 300
    assert m_fast == 300

    slow = [
        pw.StepRecord(1, 1, m, 10.0, 100, 100, 2, True) for m in range(1, 100)
    ]
    w_slow, m_slow, _ = pw.compute_window_w(slow, 85)
    assert w_slow == 20
    assert m_slow == 14

    cands = [
        (pw.CandidatePoint(1, "start", 1, 1, True), 60),
        (pw.CandidatePoint(1, "frac_25", 40, 1, True), 60),
        (pw.CandidatePoint(1, "frac_50", 61, 1, True), 60),
    ]
    kept = pw.deduplicate_stage_candidates(cands)
    assert [c.start_model for c, _ in kept] == [1, 61]


def test_parse_mod_header_extracts_prev_model_and_rule43_fields(
    tmp_path: pathlib.Path,
) -> None:
    mod = tmp_path / "sample.mod"
    # bit 2 (4) + bit 4 (16) -> prev_model=True, rotation_flag=True
    mod.write_text(
        "\n".join(
            [
                "! comment",
                "            20 -- model for mesa/star",
                "",
                "   version_number   'r26.4.1'",
                "           M/Msun     1.0000000000000000D+00",
                "     model_number                         42",
                "         star_age     1.2340000000000000D+08",
                "        initial_z     2.0000000000000000D-02",
                "         n_shells                        500",
                "         net_name   'basic.net'",
                "          species                          8",
                "",
                "1 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    hdr = pw.parse_mod_header(mod)
    assert hdr.file_type == 20
    assert hdr.prev_model is True
    assert hdr.rotation_flag is True
    assert hdr.v_flag is False
    assert hdr.net_name == "basic.net"
    assert abs(hdr.mass_msun - 1.0) < 1e-15
    assert abs(hdr.initial_z - 0.02) < 1e-15
    assert hdr.model_number == 42
    assert len(hdr.sha256) == 64
