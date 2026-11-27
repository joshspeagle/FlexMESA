# SPDX-License-Identifier: GPL-3.0-or-later
"""PyO3 integration tests for `other_overshooting_scheme` and
`other_adjust_mlt_gradt_fraction` (`#449`).
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from flexmesa.testing import harness_runner
from tests import case_extras


def _write_other_overshoot_case(case_dir: pathlib.Path) -> pathlib.Path:
    """Create a minimal 2.5 Msun convective-core `overshoot_scheme='other'`
    test case under `case_dir`.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "inlist_other_overshoot").write_text(
        "&star_job\n"
        "  load_saved_model = .false.\n"
        "  create_pre_main_sequence_model = .false.\n"
        "  save_model_when_terminate = .false.\n"
        "/\n"
        "&kap\n"
        "  Zbase = 0.02\n"
        "/\n"
        "&controls\n"
        "  initial_mass = 2.5\n"
        "  initial_z = 0.02\n"
        "  overshoot_scheme(1) = 'other'\n"
        "  overshoot_zone_type(1) = 'any'\n"
        "  overshoot_zone_loc(1) = 'core'\n"
        "  overshoot_bdy_loc(1) = 'top'\n"
        "  overshoot_f(1) = 0.055\n"
        "  overshoot_f0(1) = 0.005\n"
        "  overshoot_Delta0(1) = 1.0\n"
        "  overshoot_D_min = 1d-2\n"
        "  max_model_number = 5\n"
        "/\n",
        encoding="utf-8",
    )
    (case_dir / "rn").write_text(
        "#!/bin/bash\ndo_one inlist_other_overshoot\n",
        encoding="utf-8",
    )
    return case_dir


def test_python_other_overshoot_without_hook_or_returning_none_fails(
    tmp_path: pathlib.Path,
) -> None:
    """Default `None` or unimplemented `other_overshooting_scheme` fails."""
    case_dir = _write_other_overshoot_case(tmp_path / "case_missing")
    with pytest.raises(ValueError, match="overshoot_scheme"):
        harness_runner.run_trace_inprocess(
            "star",
            case_dir,
            tmp_path / "out_none",
            max_steps=2,
            numerics_mode="strict",
            extras=None,
        )

    class _NoneHookExtras:
        def other_overshooting_scheme(
            self, _state: dict[str, Any], _i_bdy: int, _j_param: int
        ) -> None:
            return None

    with pytest.raises(ValueError, match="null_other_overshooting_scheme"):
        harness_runner.run_trace_inprocess(
            "star",
            case_dir,
            tmp_path / "out_ret_none",
            max_steps=2,
            numerics_mode="strict",
            extras=_NoneHookExtras(),
        )


def test_python_other_overshoot_and_gradt_hooks_run_live(
    tmp_path: pathlib.Path,
) -> None:
    """Python `other_overshooting_scheme` + `other_adjust_mlt_gradt_fraction`
    run cleanly through `PyStarExtrasAdapter` even without `extras_controls`.
    """
    case_dir = _write_other_overshoot_case(tmp_path / "case_live")

    class _StepAndBlendExtras:
        def __init__(self) -> None:
            self.overshoot_calls = 0
            self.gradt_calls = 0
            self.saw_snapshot_fields = False

        def other_adjust_mlt_gradt_fraction(
            self, state: dict[str, Any]
        ) -> list[float]:
            self.gradt_calls += 1
            nz = int(state["nz"])
            assert len(state["ln_peos"]) == nz
            assert len(state["cp"]) == nz
            assert float(state["xmstar"]) > 0.0
            assert "min_d_mix" in state
            self.saw_snapshot_fields = True
            return [
                0.3 if float(state["d_mix"][k]) > 0.0 else -1.0
                for k in range(nz)
            ]

        def other_overshooting_scheme(
            self, state: dict[str, Any], i_bdy: int, j_param: int
        ) -> tuple[int, int, list[float], list[float]]:
            self.overshoot_calls += 1
            nz = int(state["nz"])
            f = float(state["overshoot_f"][j_param])
            f0 = float(state["overshoot_f0"][j_param])
            d0 = float(state["overshoot_d0"][j_param])
            delta0 = float(state["overshoot_delta0"][j_param])
            hp_cb = float(state["conv_bdy_hp"][i_bdy])
            k_ob, r_ob, d_ob, vc_ob = state["eval_over_bdy_params"](i_bdy, f0)
            assert int(state["over_bdy_k"][i_bdy]) == int(k_ob)
            d_min = float(state["overshoot_d_min"])
            r_arr = state["r"]
            d_arr = [0.0] * nz
            vc_arr = [0.0] * nz
            k_b = 0
            for k in range(int(k_ob), -1, -1):
                dr = float(r_arr[k]) - float(r_ob)
                if dr <= f * hp_cb and float(d_ob) >= d_min:
                    d_arr[k] = d0 + delta0 * float(d_ob)
                    vc_arr[k] = float(vc_ob)
                else:
                    k_b = k
                    break
            return int(k_ob), k_b, d_arr, vc_arr

    hooks = _StepAndBlendExtras()
    rep = harness_runner.run_trace_inprocess(
        "star",
        case_dir,
        tmp_path / "out_live",
        max_steps=2,
        numerics_mode="strict",
        extras=hooks,
    )
    assert rep["total_steps"] == 2
    assert hooks.overshoot_calls > 0
    assert hooks.gradt_calls > 0
    assert hooks.saw_snapshot_fields is True


def test_python_other_overshoot_invalid_return_raises(
    tmp_path: pathlib.Path,
) -> None:
    """Negative index bounds or wrong vector lengths raise `ValueError`."""
    case_dir = _write_other_overshoot_case(tmp_path / "case_bad")

    class _BadBoundsExtras:
        def other_overshooting_scheme(
            self, state: dict[str, Any], _i_bdy: int, _j_param: int
        ) -> tuple[int, int, list[float], list[float]]:
            nz = int(state["nz"])
            return -1, 0, [0.0] * nz, [0.0] * nz

    with pytest.raises(ValueError, match="negative index bounds"):
        harness_runner.run_trace_inprocess(
            "star",
            case_dir,
            tmp_path / "out_bad_bounds",
            max_steps=1,
            numerics_mode="strict",
            extras=_BadBoundsExtras(),
        )


def test_extended_convective_penetration_resolves_and_steps(
    tmp_path: pathlib.Path,
) -> None:
    """`star/extended_convective_penetration` resolves to ported extras and
    evolves live steps in strict mode.
    """
    case_dir = harness_runner.resolve_case_dir(
        "star", "extended_convective_penetration"
    )
    attached = case_extras.resolve_case_extras("star", case_dir)
    ecp_cls = case_extras.star_extras.ExtendedConvectivePenetrationStarExtras
    assert isinstance(attached, ecp_cls)
    rep = harness_runner.run_trace_inprocess(
        "star",
        case_dir,
        tmp_path / "out_ecp_2steps",
        max_steps=2,
        numerics_mode="strict",
        extras=attached,
    )
    assert rep["total_steps"] == 2
    assert attached.checks_okay is not None
