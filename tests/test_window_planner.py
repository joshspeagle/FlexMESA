# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the W1 window-snapshot overlay patch and planner (#684)."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys
from types import ModuleType

import pytest

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


def test_prune_case_snapshots_keeps_dedup_and_avoids_prefix_collisions(
    tmp_path: pathlib.Path,
) -> None:
    clock_dir = tmp_path / "clocks"
    snap_dir = tmp_path / "snaps"
    clock_dir.mkdir()
    snap_dir.mkdir()

    # 120 fast steps (0.1s each -> W=300): point 1 keeps all others inside W,
    # so only model 1 is retained after Rule-2 deduplication.
    lines = [
        "# case: star/make_zams",
        *(f"STEP\t1\t1\t{m}\t0.1\t100\t100\t2\tT" for m in range(1, 121)),
        "STAGE_END\t1\t1\t120\tmax_model_number\tok",
    ]
    clock_path = clock_dir / "make_zams.clock.tsv"
    clock_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cands = pw.candidate_points_for_stage("make_zams", 1, 1, 120)
    assert len(cands) == 5
    for cand in cands:
        (snap_dir / f"make_zams_1_{cand.start_model}.mod").write_text("m")
    sibling = snap_dir / "make_zams_low_mass_1_1.mod"
    sibling.write_text("sibling")

    kept, deleted = pw.prune_case_snapshots("make_zams", clock_path, snap_dir)
    assert kept == ["make_zams_1_1.mod"]
    assert len(deleted) == 4
    assert (snap_dir / "make_zams_1_1.mod").is_file()
    assert sibling.is_file()

    # Missing kept file raises before deleting any extra snapshot.
    (snap_dir / "make_zams_1_1.mod").unlink()
    extra = snap_dir / "make_zams_1_99.mod"
    extra.write_text("extra")
    try:
        pw.prune_case_snapshots("make_zams", clock_path, snap_dir)
    except FileNotFoundError:
        assert extra.is_file()
    else:
        raise AssertionError("expected FileNotFoundError when kept mod missing")


def test_empty_stage_steps_recorded_with_v9_reason(
    tmp_path: pathlib.Path,
) -> None:
    stage_steps_file = tmp_path / "stage_steps.json"
    stage_steps_file.write_text(
        json.dumps(
            {
                "cases": {
                    "star/1.4M_ms_op_mono": {"stage_steps": []},
                    "star/short_ok": {"stage_steps": [30]},
                }
            }
        ),
        encoding="utf-8",
    )
    clock_dir = tmp_path / "clocks"
    snap_dir = tmp_path / "snaps"
    clock_dir.mkdir()
    snap_dir.mkdir()
    (clock_dir / "short_ok.clock.tsv").write_text(
        "\n".join(
            [
                "# case: star/short_ok",
                "# mesa_version: r26.4.1",
                "# compiler: gfortran 15.2.0",
                "# mesa_sdk_version: x86_64-linux-26.6.1",
                *(
                    f"STEP\t1\t1\t{m}\t0.2\t100\t100\t2\tT"
                    for m in range(1, 31)
                ),
                "SNAP\t1\t1\tT\tshort_ok_1_1.mod",
                "STAGE_END\t1\t1\t30\tmax_model_number\tok",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (snap_dir / "short_ok_1_1.mod").write_text(
        "\n".join(
            [
                "! synthetic",
                "            4 -- model for mesa/star",
                "",
                "   version_number   'r26.4.1'",
                "           M/Msun     1.0000000000000000D+00",
                "     model_number                          1",
                "         star_age     0.0000000000000000D+00",
                "        initial_z     2.0000000000000000D-02",
                "         n_shells                        100",
                "         net_name   'basic.net'",
                "          species                          8",
                "",
            ]
        ),
        encoding="utf-8",
    )

    wins, skipped, hdr, not_built = pw.build_windows_from_clocks(
        stage_steps_file, None, clock_dir, snap_dir
    )
    assert len(wins) == 1
    assert not_built == []
    assert skipped == [
        ("1.4M_ms_op_mono", "window not built (no MESA stage steps)")
    ]
    out_toml = tmp_path / "windows_e2e.toml"
    out_toml.write_text(
        pw.render_windows_toml(wins, skipped, hdr), encoding="utf-8"
    )
    n_wins, n_skip = pw.check_committed_toml(out_toml)
    assert (n_wins, n_skip) == (1, 1)


def test_twin_studies_override_replaces_three_stage_row_with_two_stars() -> (
    None
):
    ovr = pw.STAGE_STEPS_OVERRIDES["twin_studies"]
    row = {
        "stage_steps": [80, 80, 80],
        "out_txt_sha256": ovr.out_txt_sha256,
    }
    assert pw.case_stage_steps("twin_studies", row) == [80, 80]
    assert pw.case_stage_steps("other", {"stage_steps": [5, 9]}) == [5, 9]
    stale = {"stage_steps": [80, 80, 80], "out_txt_sha256": "0" * 64}
    try:
        pw.case_stage_steps("twin_studies", stale)
    except SystemExit as exc:
        assert "review the override" in str(exc)
    else:
        raise AssertionError("expected SystemExit on a stale out.txt hash")
    text = pw.render_windows_toml([], [], {}, {"twin_studies": ovr})
    parsed = pw.tomllib.loads(text)
    (entry,) = parsed["stage_steps_overrides"]
    assert entry["stage_steps"] == [80, 80]
    assert entry["source_lines"].count("terminated evolution: forced_stop") == 2


def test_rule43_net_mismatch_records_window_not_built_instead_of_aborting(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_steps_file = tmp_path / "stage_steps.json"
    stage_steps_file.write_text(
        json.dumps({"cases": {"star/netcase": {"stage_steps": [30]}}}),
        encoding="utf-8",
    )
    clock_dir = tmp_path / "clocks"
    snap_dir = tmp_path / "snaps"
    clock_dir.mkdir()
    snap_dir.mkdir()
    (clock_dir / "netcase.clock.tsv").write_text(
        "\n".join(
            ["# case: star/netcase"]
            + [
                f"STEP\t1\t1\t{m}\t1.000000000\t100\t100\t2\tT"
                for m in range(1, 31)
            ]
            + ["STAGE_END\t1\t1\t30\tmax_model_number\tnormal"]
        )
        + "\n",
        encoding="utf-8",
    )
    (snap_dir / "netcase_1_1.mod").write_text(
        "\n".join(
            [
                "            4 -- model for mesa/star",
                "",
                "   version_number   'r26.4.1'",
                "           M/Msun     1.0000000000000000D+00",
                "     model_number                          1",
                "         star_age     0.0000000000000000D+00",
                "        initial_z     2.0000000000000000D-02",
                "         n_shells                        100",
                "         net_name   'approx21.net'",
                "          species                         21",
                "",
            ]
        ),
        encoding="utf-8",
    )
    original = pw.plan_case_stages

    def _expect_co_burn(
        case: str, steps: list[int], mesa_dir: pathlib.Path | None
    ) -> list[object]:
        return [
            pw.dataclasses.replace(st, expected_net_name="co_burn.net")
            for st in original(case, steps, mesa_dir)
        ]

    monkeypatch.setattr(pw, "plan_case_stages", _expect_co_burn)
    wins, skipped, hdr, not_built = pw.build_windows_from_clocks(
        stage_steps_file, None, clock_dir, snap_dir
    )
    assert wins == [] and skipped == []
    (nb,) = not_built
    assert nb.window_id == "netcase_s1_m1"
    assert nb.reason == (
        "window not built (Rule 4.3: net_name 'approx21.net' != 'co_burn.net')"
    )
    text = pw.render_windows_toml(wins, skipped, hdr, None, not_built)
    parsed = pw.tomllib.loads(text)
    assert parsed["metadata"]["not_built_window_count"] == 1
    assert parsed["windows_not_built"][0]["snapshot_file"] == "netcase_1_1.mod"
