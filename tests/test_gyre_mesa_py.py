# SPDX-License-Identifier: GPL-3.0-or-later
"""Live PyO3 tests for `_rust.GyreMesa` and `GyreInMesaStarExtras` (#170)."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from flexmesa import _rust
from flexmesa.testing import harness_runner
from tests.case_extras import star_extras


def test_gyre_mesa_error_paths(tmp_path: pathlib.Path) -> None:
    """`GyreMesa` raises `RuntimeError` on bad inputs or calls pre-model."""
    with pytest.raises(RuntimeError, match="failed to read GYRE input file"):
        _rust.GyreMesa(str(tmp_path / "missing_gyre.in"))

    bad_in = tmp_path / "bad_gyre.in"
    bad_in.write_text("not_a_namelist", encoding="utf-8")
    with pytest.raises(RuntimeError, match="GYRE:"):
        _rust.GyreMesa(str(bad_in))

    ms_dir = harness_runner.resolve_case_dir("star", "gyre_in_mesa_ms")
    gyre = _rust.GyreMesa(str(ms_dir / "gyre.in"))
    gyre.set_mesa_constants("mesa")
    with pytest.raises(RuntimeError, match="GYRE:"):
        gyre.get_modes(0)


def test_gyre_mesa_wd_l1_adiabatic_inverse_and_modes(
    tmp_path: pathlib.Path,
) -> None:
    """`GyreMesa` solves `l=1` `INVERSE` adiabatic g-modes on a WD handle."""
    wd_dir = harness_runner.resolve_case_dir("star", "gyre_in_mesa_wd")
    captured: dict[str, Any] = {}

    class _HandleCaptureProbe:
        wants_star_model = True

        def extras_after_evolve(self, state: dict[str, Any]) -> None:
            captured["handle"] = state["star_model"]

    _rust.trace_star_case(
        str(wd_dir),
        str(tmp_path / "wd_cap"),
        max_steps_per_stage=1,
        skip_optional=True,
        allow_unsupported_controls=False,
        extras=_HandleCaptureProbe(),
    )
    handle = captured["handle"]

    gyre = _rust.GyreMesa(str(wd_dir / "gyre.in"))
    gyre.set_mesa_constants("mesa")
    gyre.set_model(handle, False, False, False)
    modes = gyre.get_modes(1)

    assert len(modes) >= 1
    for n_pg, (freq_re, freq_im) in modes:
        assert n_pg < 0
        assert freq_re > 0.0
        assert freq_im == 0.0


def test_gyre_in_mesa_ms_live_handler_and_profile_due_flag(
    tmp_path: pathlib.Path,
) -> None:
    """`GyreInMesaStarExtras` steps cleanly and reads accurate snapshot flags.

    Verifies the `#170` fix for `steps=0`: `build_star_snapshot` always
    exposes `need_to_save_profiles_now` (`False` outside post-solve checks,
    True on step 1 finish with `profile_interval=100`, `False` on step 2),
    and `GyreInMesaStarExtras` solves `l=0` nonadiabatic modes on step 2.
    """
    ms_dir = harness_runner.resolve_case_dir("star", "gyre_in_mesa_ms")
    observed_flags: list[tuple[str, int, bool]] = []

    class _RecordingGyreExtras(star_extras.GyreInMesaStarExtras):
        def extras_startup(self, state: dict[str, Any], restart: bool) -> None:
            observed_flags.append(
                (
                    "extras_startup",
                    int(state["model_number"]),
                    bool(state["need_to_save_profiles_now"]),
                )
            )
            super().extras_startup(state, restart)

        def extras_start_step(self, state: dict[str, Any]) -> str:
            observed_flags.append(
                (
                    "extras_start_step",
                    int(state["model_number"]),
                    bool(state["need_to_save_profiles_now"]),
                )
            )
            return "keep_going"

        def extras_check_model(self, state: dict[str, Any]) -> str:
            observed_flags.append(
                (
                    "extras_check_model",
                    int(state["model_number"]),
                    bool(state["need_to_save_profiles_now"]),
                )
            )
            return "keep_going"

        def extras_finish_step(self, state: dict[str, Any]) -> str:
            observed_flags.append(
                (
                    "extras_finish_step",
                    int(state["model_number"]),
                    bool(state["need_to_save_profiles_now"]),
                )
            )
            return super().extras_finish_step(state)

        def extras_after_evolve(self, state: dict[str, Any]) -> None:
            observed_flags.append(
                (
                    "extras_after_evolve",
                    int(state["model_number"]),
                    bool(state["need_to_save_profiles_now"]),
                )
            )
            super().extras_after_evolve(state)

    handler = _RecordingGyreExtras()
    rep = _rust.trace_star_case(
        str(ms_dir),
        str(tmp_path / "ms_out"),
        max_steps_per_stage=2,
        skip_optional=True,
        allow_unsupported_controls=False,
        extras=handler,
    )

    assert int(rep["total_steps"]) == 2
    assert observed_flags == [
        ("extras_startup", 0, False),
        ("extras_start_step", 0, False),
        ("extras_check_model", 1, True),
        ("extras_finish_step", 1, True),
        ("extras_start_step", 1, False),
        ("extras_check_model", 2, False),
        ("extras_finish_step", 2, False),
        ("extras_after_evolve", 2, False),
    ]
    assert handler.gyre is None
    assert len(handler.calls) == 1
    call_model, modes = handler.calls[0]
    assert call_model == 2
    assert len(modes) >= 1
    expected_written = min(len(modes), 2)
    assert handler.num_results == expected_written
    assert len(handler.ixtra1) == expected_written
    assert len(handler.xtra1) == expected_written
    assert len(handler.xtra2) == expected_written
    assert len(handler.xtra3) == expected_written
    for freq_re, freq_im in (m[1] for m in modes):
        assert freq_re > 0.0
        assert freq_im != 0.0

    with pytest.raises(NotImplementedError, match="eigenfunction"):
        handler._process_mode(
            {"x_ctrl": [2.0e-4, 1.0e99], "x_logical_ctrl": [False, True]},
            2,
            1,
            2.0e-4,
            -1.0e-9,
        )
