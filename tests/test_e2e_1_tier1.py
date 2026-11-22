# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stage E2E-1 (#92) Tier 1 hydrostatic, solar, mixing, and diffusion
`test_suite` execution and ZAMS/restart verification suite (`#148`).
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile

import numpy as np
import pytest

from flexmesa import star
from flexmesa.star import e2e
from flexmesa.testing import known_gaps_gate

from . import cleaned_case_gaps

_REPRESENTATIVE_TIER1_CASES = (
    "1.3M_ms_high_Z",
    "semiconvection",
    "check_pulse_atm",
    "custom_colors",
    "extended_convective_penetration",
    "make_env",
    "custom_rates",
    "carbon_kh",
)
_STRICT_VALID_TIER1_CASES = frozenset(
    {
        "1.3M_ms_high_Z",
        "semiconvection",
        "extended_convective_penetration",
        "make_env",
    }
)


def test_pulse_atm_export_preserves_invariants() -> None:
    """Verify pulsation atmosphere exports (`GYRE`, `FGONG`) on a 1 Msun ZAMS
    model preserve radial zone dimensions (`check_pulse_atm`).
    """
    model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    gyre_txt = model.to_gyre_str(add_center_point=True)
    fgong_txt = model.to_fgong_str(add_center_point=True)
    assert len(gyre_txt.splitlines()) == model.nz + 2
    assert len(fgong_txt.splitlines()) > model.nz


def test_convective_premix_predictive_mix_and_diffusion_hooks() -> None:
    """Verify convective premixing, predictive mixing, and Burgers atomic
    diffusion preserve species mass conservation.
    """
    model = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    n_cpm = model.do_conv_premix(max_iters=3, avoid_fuel_increase=False)
    assert n_cpm >= 0
    n_pred = model.add_predictive_mixing(
        ingested_mass_limit_coeff=1.0e-2,
        superad_thresh=1.0e-4,
    )
    assert n_pred >= 0
    v_gt, eps_sed = model.do_element_diffusion(dt=1.0e10)
    assert v_gt.shape == (model.nz, model.species)
    assert eps_sed.shape == (model.nz,)
    xa = model.xa
    row_sums = np.sum(xa, axis=1)
    assert np.allclose(row_sums, 1.0, rtol=1e-12, atol=1e-12)
    assert np.all((model.xh_mass >= 0.0) & (model.xh_mass <= 1.0))
    assert np.all((model.yhe_mass >= 0.0) & (model.yhe_mass <= 1.0))
    assert np.any(model.d_mix > 0.0)


@pytest.mark.parametrize(
    "case_name,inlist_body",
    [
        (
            "starspots",
            "&kap\n"
            "  zbase = 0.02\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  x_ctrl(1) = 0.34\n"
            "  x_ctrl(2) = 0.85\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "T_tau_gradr",
            "&kap\n"
            "  zbase = 0.02\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  atm_option = 'T_tau'\n"
            "  atm_T_tau_relation = 'Krishna_Swamy'\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "check_pulse_atm",
            "&kap\n"
            "  zbase = 0.02\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "custom_colors",
            "&kap\n"
            "  zbase = 0.02\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "adjust_net",
            "&star_job\n"
            "  change_net = .true.\n"
            "  new_net_name = 'co_burn.net'\n"
            "/\n"
            "&kap\n"
            "  zbase = 0.02\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 2.0\n"
            "  initial_z = 0.02\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "semiconvection",
            "&kap\n"
            "  zbase = 0.02\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 4.0\n"
            "  initial_z = 0.02\n"
            "  use_Ledoux_criterion = .true.\n"
            "  alpha_semiconvection = 0.04\n"
            "  thermohaline_coeff = 2.0\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "simplex_solar_calibration",
            "&kap\n"
            "  zbase = 0.02\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  mixing_length_alpha = 1.90\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
    ],
)
def test_synthetic_tier1_case_runners_pass_assertions_and_restart(
    case_name: str,
    inlist_body: str,
) -> None:
    """Verify synthetic Tier 1 directories reach `max_model_number = 2`
    without in-library case overrides and verify `.photo` restart parity.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = pathlib.Path(tmp) / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "inlist").write_text(inlist_body, encoding="utf-8")
        (case_dir / "rn").write_text(
            "#!/usr/bin/env bash\ndo_one inlist final.mod\n",
            encoding="utf-8",
        )
        rep = e2e.run_test_suite_case(
            case_dir,
            skip_optional=True,
            max_steps_per_stage=3,
            verify_restart=True,
        )
        assert rep["case_name"] == case_name
        assert rep["tier"] == 0
        assert rep["passed"] is True
        assert rep["assertions_passed"] is True
        assert rep["photo_restart_passed"] is True
        assert rep["photo_restart_verified"] is True
        assert rep["total_steps"] >= 1
        assert rep["total_newton_iters"] >= 1
        assert rep["wall_time_ms"] > 0.0
        assert np.isfinite(rep["final_teff"]) and rep["final_teff"] > 0.0
        assert np.isfinite(rep["final_log_l"])
        assert np.isfinite(rep["final_log_r"])
        assert np.isfinite(rep["final_log_tc"])
        assert np.isfinite(rep["final_log_rhoc"])
        assert 0.0 <= rep["final_center_h1"] <= 1.0
        assert 0.0 <= rep["final_center_he4"] <= 1.0


def test_full_stages_skip_optional_false_multi_stage_tier1_execution() -> None:
    """Verify `skip_optional=False` (`--full-stages`) executes all `do_one`
    stages sequentially to physical `max_model_number` termination.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = pathlib.Path(tmp) / "make_zams"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "rn").write_text(
            "#!/usr/bin/env bash\n"
            'if [ -n "$MESA_SKIP_OPTIONAL" ]; then\n'
            "  cp standard_zams.mod zams.mod\n"
            "else\n"
            "  do_one inlist_stage1_header zams.mod\n"
            "fi\n"
            "do_one inlist_stage2_header final.mod\n",
            encoding="utf-8",
        )
        (case_dir / "inlist_stage1_header").write_text(
            "&star_job\n"
            "  save_model_when_terminate = .true.\n"
            "  save_model_filename = 'zams.mod'\n"
            "/\n"
            "&kap\n"
            "  zbase = 0.02\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  max_model_number = 2\n"
            "/\n",
            encoding="utf-8",
        )
        (case_dir / "inlist_stage2_header").write_text(
            "&star_job\n"
            "  load_saved_model = .true.\n"
            "  load_model_filename = 'zams.mod'\n"
            "/\n"
            "&kap\n"
            "  zbase = 0.02\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  max_model_number = 4\n"
            "/\n",
            encoding="utf-8",
        )
        rep = e2e.run_test_suite_case(
            case_dir,
            skip_optional=False,
            max_steps_per_stage=4,
            verify_restart=True,
        )
        assert rep["case_name"] == "make_zams"
        assert rep["tier"] == 0
        assert rep["passed"] is True
        assert rep["assertions_passed"] is True
        assert rep["photo_restart_passed"] is True
        assert rep["photo_restart_verified"] is True
        assert len(rep["stages"]) == 2
        assert rep["total_steps"] >= 2


def test_unrelaxed_non_solar_zams_z_and_y_fail_loudly(
    tmp_path: pathlib.Path,
) -> None:
    """Verify requesting non-solar Z or Y from built-in ZAMS without relax
    directives raises `ValueError` instead of substituting Z=0.02 (#148).
    """
    z_dir = tmp_path / "bad_z"
    z_dir.mkdir(parents=True, exist_ok=True)
    (z_dir / "rn").write_text(
        "#!/usr/bin/env bash\ndo_one inlist final.mod\n", encoding="utf-8"
    )
    (z_dir / "inlist").write_text(
        "&controls\n  initial_mass = 1.0\n  initial_z = 0.001\n/\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"initial_z = 0\.02"):
        e2e.run_test_suite_case(z_dir, max_steps_per_stage=1)

    y_dir = tmp_path / "bad_y"
    y_dir.mkdir(parents=True, exist_ok=True)
    (y_dir / "rn").write_text(
        "#!/usr/bin/env bash\ndo_one inlist final.mod\n", encoding="utf-8"
    )
    (y_dir / "inlist").write_text(
        "&controls\n  initial_mass = 1.0\n  initial_z = 0.02\n"
        "  initial_y = 0.24\n/\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"initial_y = 0\.28"):
        e2e.run_test_suite_case(y_dir, max_steps_per_stage=1)


@pytest.mark.parametrize("case_name", _REPRESENTATIVE_TIER1_CASES)
def test_upstream_tier1_test_suite_case_and_restart_reproducibility(
    case_name: str,
    tmp_path: pathlib.Path,
) -> None:
    """Verify unmodified Tier 1 cases fail strictly under `#147` and cleaned
    copies respect declared stops or capped 2-step execution without in-library
    case overrides.
    """
    base = known_gaps_gate.require_mesa_test_inputs_dir()
    if base is None:
        return
    case_dir = base / "star" / "test_suite" / case_name
    assert case_dir.is_dir(), f"Upstream case directory not found: {case_dir}"

    if case_name in _STRICT_VALID_TIER1_CASES:
        raw_copy = tmp_path / "raw" / case_name
        shutil.copytree(case_dir, raw_copy)
        gap = cleaned_case_gaps.DECLARED_STAR_STOPS.get(case_name)
        if gap is not None:
            with pytest.raises(ValueError, match=gap.message_pattern):
                e2e.run_test_suite_case(
                    raw_copy,
                    skip_optional=True,
                    max_steps_per_stage=2,
                    verify_restart=True,
                )
            return
        rep = e2e.run_test_suite_case(
            raw_copy,
            skip_optional=True,
            max_steps_per_stage=2,
            verify_restart=True,
        )
        assert rep["case_name"] == case_name
        assert rep["tier"] == 0
        assert rep["termination_code"] == "max_steps_cap"
        assert rep["passed"] is False
        assert rep["assertions_passed"] is True
        assert rep["photo_restart_verified"] is True
        assert rep["total_steps"] >= 1
        assert rep["total_newton_iters"] >= 1
        return

    with pytest.raises(
        ValueError,
        match=r"(unsupported|unknown|unrecognized).*control",
    ):
        e2e.run_test_suite_case(
            case_dir,
            skip_optional=True,
            max_steps_per_stage=2,
            verify_restart=True,
        )

    clean_dir = tmp_path / case_name
    e2e.copy_cleaned_case(case_dir, clean_dir, suite="star")
    gap = cleaned_case_gaps.DECLARED_STAR_STOPS.get(case_name)
    if gap is not None:
        with pytest.raises(ValueError, match=gap.message_pattern):
            e2e.run_test_suite_case(
                clean_dir,
                skip_optional=True,
                max_steps_per_stage=2,
                verify_restart=True,
            )
        return
    rep = e2e.run_test_suite_case(
        clean_dir,
        skip_optional=True,
        max_steps_per_stage=2,
        verify_restart=True,
    )
    assert rep["case_name"] == case_name
    assert rep["tier"] == 0
    assert rep["termination_code"] == "max_steps_cap"
    assert rep["passed"] is False
    assert rep["assertions_passed"] is True
    assert rep["photo_restart_verified"] is True
    assert rep["total_steps"] >= 1
    assert rep["total_newton_iters"] >= 1
    assert np.isfinite(rep["final_teff"]) and rep["final_teff"] > 0.0
    assert np.isfinite(rep["final_log_l"])
    assert np.isfinite(rep["final_log_r"])
    assert np.isfinite(rep["final_log_tc"])
    assert np.isfinite(rep["final_log_rhoc"])
