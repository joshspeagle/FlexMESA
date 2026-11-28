# SPDX-License-Identifier: GPL-3.0-or-later
"""PyO3 integration tests for extra profile columns and header items."""

from __future__ import annotations

import pathlib
import tomllib
from typing import Any

import pytest

from flexmesa.testing import harness_runner
from tests import case_extras
from tests.case_extras import star_extras


def _parse_6line_header_and_cols(
    path: pathlib.Path,
) -> tuple[dict[str, str], list[str], list[list[str]]]:
    """Parse MESA 6-line header tokens, column names, and string rows."""
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 7, f"expected at least 7 lines in {path}"
    hdr_keys = lines[1].split()
    hdr_vals = lines[2].split()
    assert len(hdr_keys) == len(hdr_vals)
    col_names = lines[5].split()
    rows = [ln.split() for ln in lines[6:] if ln.strip()]
    return dict(zip(hdr_keys, hdr_vals, strict=True)), col_names, rows


def _write_minimal_case(case_dir: pathlib.Path) -> None:
    """Write a fast 2-step 1 Msun ZAMS case emitting start/end profiles."""
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "rn").write_text(
        "#!/bin/bash\ndo_one inlist_test\n",
        encoding="utf-8",
    )
    (case_dir / "inlist_test").write_text(
        "\n".join(
            [
                "&star_job",
                "  profile_starting_model = .true.",
                "  write_profile_when_terminate = .true.",
                "  filename_for_profile_when_terminate = 'final_prof.data'",
                "  steps_to_take_before_terminate = 2",
                "/",
                "&controls",
                "  initial_mass = 1.0",
                "  initial_z = 0.02",
                "  mixing_length_alpha = 1.85",
                "  history_interval = 1",
                "  profile_interval = 1",
                "  max_num_profile_models = 10",
                "  write_profiles_flag = .true.",
                "/",
                "&kap",
                "  zbase = 0.02d0",
                "/",
                "",
            ]
        ),
        encoding="utf-8",
    )


class _RecordingExtras:
    """Python extras implementing all extra log column and header hooks."""

    def __init__(self) -> None:
        self.hist_hdr_calls = 0
        self.prof_hdr_calls = 0
        self.prof_col_calls = 0

    def extra_history_header_items(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        self.hist_hdr_calls += 1
        return [("alpha_hdr", float(state["mixing_length_alpha"]))]

    def extra_history_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        return [("model_plus_alpha", float(state["model_number"]) + 1.85)]

    def extra_profile_header_items(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        self.prof_hdr_calls += 1
        return [("prof_alpha_hdr", float(state["mixing_length_alpha"]))]

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        self.prof_col_calls += 1
        nz = int(state["nz"])
        zbar = state["zbar"]
        abar = state["abar"]
        return [
            (
                "zbar_div_abar",
                [float(zbar[k]) / float(abar[k]) for k in range(nz)],
            )
        ]


def test_py_adapter_wires_all_extra_profile_and_header_hooks(
    tmp_path: pathlib.Path,
) -> None:
    """Verify history headers, profile headers, and profile columns via PyO3."""
    case_dir = tmp_path / "case"
    out_dir = tmp_path / "out"
    _write_minimal_case(case_dir)
    extras = _RecordingExtras()

    rep = harness_runner.run_trace_inprocess(
        "star",
        case_dir,
        out_dir,
        max_steps=2,
        respect_intervals=True,
        write_profiles=True,
        is_logs_dir=True,
        extras=extras,
    )
    assert rep["total_steps"] == 2
    assert extras.hist_hdr_calls == 1
    assert extras.prof_hdr_calls == 3
    assert extras.prof_col_calls == 3

    hist_hdr, hist_cols, hist_rows = _parse_6line_header_and_cols(
        out_dir / "LOGS" / "history.data"
    )
    assert float(hist_hdr["alpha_hdr"]) == pytest.approx(1.85, rel=1.0e-13)
    assert hist_cols[-1] == "model_plus_alpha"
    assert [float(r[-1]) for r in hist_rows] == pytest.approx(
        [2.85, 3.85], rel=1.0e-13
    )

    for prof_path in (
        out_dir / "LOGS" / "profile1.data",
        out_dir / "LOGS" / "profile2.data",
        out_dir / "LOGS" / "profile3.data",
        out_dir / "final_prof.data",
    ):
        hdr, cols, rows = _parse_6line_header_and_cols(prof_path)
        assert float(hdr["prof_alpha_hdr"]) == pytest.approx(1.85, rel=1.0e-13)
        assert cols[-1] == "zbar_div_abar"
        assert len(rows) > 100
        surf_val = float(rows[0][-1])
        assert 0.5 < surf_val <= 1.0


@pytest.mark.parametrize(
    "bad_kind",
    ["wrong_len", "empty_col_name", "empty_hdr_name", "raised"],
)
def test_py_adapter_profile_and_header_errors_propagate(
    tmp_path: pathlib.Path, bad_kind: str
) -> None:
    """Invalid extra columns/headers or Python exceptions fail loudly."""
    case_dir = tmp_path / "bad_case"
    _write_minimal_case(case_dir)

    class _BadExtras:
        def extra_history_header_items(
            self, _s: dict[str, Any]
        ) -> list[tuple[str, float]]:
            if bad_kind == "empty_hdr_name":
                return [("   ", 1.0)]
            return []

        def extra_profile_columns(
            self, s: dict[str, Any]
        ) -> list[tuple[str, list[float]]]:
            if bad_kind == "wrong_len":
                return [("bad_len", [1.0] * (int(s["nz"]) - 1))]
            if bad_kind == "empty_col_name":
                return [("", [1.0] * int(s["nz"]))]
            if bad_kind == "raised":
                raise RuntimeError("boom in extra_profile_columns")
            return []

    with pytest.raises(ValueError):
        harness_runner.run_trace_inprocess(
            "star",
            case_dir,
            tmp_path / f"out_{bad_kind}",
            max_steps=1,
            write_profiles=True,
            is_logs_dir=True,
            extras=_BadExtras(),
        )


def test_newly_graduated_profile_header_cases_status_and_attachment() -> None:
    """7 profile/header cases attach cleanly; wd_aic stays detached."""
    graduated = (
        ("20M_z2m2_high_rotation", star_extras.HighRotation20MZ2m2StarExtras),
        ("T_tau_gradr", star_extras.TTauGradrStarExtras),
        ("hb_2M", star_extras.Hb2MStarExtras),
        ("low_z", star_extras.LowZStarExtras),
        ("wd_acc_small_dm", star_extras.WdAccSmallDmStarExtras),
        ("wd_cool_0.6M", star_extras.WdCool06MStarExtras),
        ("wd_nova_burst", star_extras.WdNovaBurstStarExtras),
    )
    ledger_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "benchmarks/history_harness/extras_status.toml"
    )
    ledger = tomllib.loads(ledger_path.read_text(encoding="utf-8"))
    by_case = {(c["suite"], c["case"]): c for c in ledger["cases"]}

    for case, expected_cls in graduated:
        cdir = harness_runner.resolve_case_dir("star", case)
        attached = case_extras.resolve_case_extras("star", cdir)
        assert isinstance(attached, expected_cls)
        assert case_extras.classify_case_extras("star", cdir) == "ported"
        row = by_case[("star", case)]
        assert row["status"] == "ported"
        assert "issue" not in row

    aic_dir = harness_runner.resolve_case_dir("star", "wd_aic")
    assert isinstance(
        case_extras.match_case_extras("star", aic_dir),
        star_extras.WdAicStarExtras,
    )
    assert case_extras.resolve_case_extras("star", aic_dir) is None
    assert case_extras.classify_case_extras("star", aic_dir) == "unported"
    assert by_case[("star", "wd_aic")]["status"] == "unported"
    assert by_case[("star", "wd_aic")]["issue"] == "#231"


def test_live_case_extras_emit_profile_columns_and_header_items(
    tmp_path: pathlib.Path,
) -> None:
    """Exercise low_z, wd_cool_0.6M, T_tau_gradr, and zbar/abar handlers."""
    lz_dir = harness_runner.resolve_case_dir("star", "low_z")
    lz_out = tmp_path / "out_low_z"
    harness_runner.run_trace_inprocess(
        "star",
        lz_dir,
        lz_out,
        max_steps=2,
        write_profiles=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    lz_h_hdr, _, _ = _parse_6line_header_and_cols(
        lz_out / "LOGS" / "history.data"
    )
    lz_p_hdr, _, _ = _parse_6line_header_and_cols(
        lz_out / "LOGS" / "profile1.data"
    )
    assert float(lz_h_hdr["mixing_length_alpha"]) == pytest.approx(1.6)
    assert float(lz_p_hdr["mixing_length_alpha"]) == pytest.approx(1.6)

    wdc_dir = harness_runner.resolve_case_dir("star", "wd_cool_0.6M")
    wdc_out = tmp_path / "out_wdc"
    harness_runner.run_trace_inprocess(
        "star",
        wdc_dir,
        wdc_out,
        max_steps=2,
        write_profiles=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    _, wdc_cols, wdc_rows = _parse_6line_header_and_cols(
        wdc_out / "LOGS" / "profile1.data"
    )
    assert wdc_cols[-2:] == ["zbar_div_abar", "eosPC_fraction"]
    assert len(wdc_rows) > 10

    tt_dir = harness_runner.resolve_case_dir("star", "T_tau_gradr")
    tt_out = tmp_path / "out_ttau"
    harness_runner.run_trace_inprocess(
        "star",
        tt_dir,
        tt_out,
        max_steps=11,
        write_profiles=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    _, tt_cols, tt_rows = _parse_6line_header_and_cols(
        tt_out / "LOGS" / "profile1.data"
    )
    assert tt_cols[-1] == "T_check"
    assert float(tt_rows[0][-1]) > 1000.0

    wa_dir = harness_runner.resolve_case_dir("star", "wd_acc_small_dm")
    wa_out = tmp_path / "out_wd_acc"
    harness_runner.run_trace_inprocess(
        "star",
        wa_dir,
        wa_out,
        max_steps=2,
        write_profiles=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    _, wa_cols, wa_rows = _parse_6line_header_and_cols(
        wa_out / "LOGS" / "profile1.data"
    )
    assert wa_cols[-1] == "zbar_div_abar"
    assert 0.5 < float(wa_rows[0][-1]) <= 1.0

    # Handlers whose upstream `profile_columns.list` asks for unrelated
    # unported columns still run `extra_profile_columns` cleanly via PyO3.
    synth_case = tmp_path / "synth_case"
    _write_minimal_case(synth_case)
    for label, handler in (
        ("hb2m", star_extras.Hb2MStarExtras()),
        ("rot20", star_extras.HighRotation20MZ2m2StarExtras()),
        ("nova", star_extras.WdNovaBurstStarExtras()),
    ):
        sout = tmp_path / f"out_{label}"
        harness_runner.run_trace_inprocess(
            "star",
            synth_case,
            sout,
            max_steps=1,
            write_profiles=True,
            extras=handler,
        )
        _, scols, srows = _parse_6line_header_and_cols(
            sout / "LOGS" / "profile1.data"
        )
        assert scols[-1] == "zbar_div_abar"
        assert 0.5 < float(srows[0][-1]) <= 1.0
