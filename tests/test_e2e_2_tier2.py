# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stage E2E-2 (#93) Tier 2 substellar, planetary, white dwarf, and accretion
`test_suite` execution and loud-error checks (`#148`).
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

from flexmesa.star import e2e
from flexmesa.testing import known_gaps_gate

from . import cleaned_case_gaps

_REPRESENTATIVE_TIER2_CASES = (
    "irradiated_planet",
    "wd_aic",
    "make_co_wd",
)


def test_accretion_composition_and_super_eddington_nova_wind() -> None:
    """Verify accretion composition and super-Eddington nova wind execution on
    a synthetic white dwarf accretion configuration (`wd_nova_burst`).
    """
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = pathlib.Path(tmp) / "wd_nova_burst"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "inlist").write_text(
            "&star_job\n"
            "  relax_initial_mass_to_remove_h_env = .true.\n"
            "  new_mass = 0.60\n"
            "/\n"
            "&kap\n"
            "  zbase = 0.02\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 0.60\n"
            "  initial_z = 0.02\n"
            "  mass_change = 1.0d-9\n"
            "  accrete_same_as_surface = .false.\n"
            "  accretion_h1 = 0.70\n"
            "  accretion_he4 = 0.28\n"
            "  accretion_zfracs = 4\n"
            "  super_eddington_scaling_factor = 1.0\n"
            "  max_model_number = 2\n"
            "/\n",
            encoding="utf-8",
        )
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
        assert rep["case_name"] == "wd_nova_burst"
        assert rep["tier"] == 0
        assert rep["passed"] is True
        assert rep["assertions_passed"] is True
        assert rep["photo_restart_passed"] is True
        assert rep["photo_restart_verified"] is True
        assert rep["final_log_tc"] > 6.0


@pytest.mark.parametrize(
    "case_name,inlist_body",
    [
        (
            "wd_cool_0.6M",
            "&star_job\n"
            "  relax_initial_mass_to_remove_h_env = .true.\n"
            "  new_mass = 0.60\n"
            "/\n"
            "&kap\n"
            "  zbase = 0.02\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 0.60\n"
            "  initial_z = 0.02\n"
            "  do_phase_separation = .true.\n"
            "  phase_separation_option = 'CO'\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "make_o_ne_wd",
            "&star_job\n"
            "  relax_initial_mass_to_remove_h_env = .true.\n"
            "  new_mass = 1.10\n"
            "/\n"
            "&kap\n"
            "  zbase = 0.02\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 1.10\n"
            "  initial_z = 0.02\n"
            "  do_phase_separation = .true.\n"
            "  phase_separation_option = 'ONe'\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
    ],
)
def test_synthetic_tier2_case_runners_pass_assertions_and_restart(
    case_name: str,
    inlist_body: str,
) -> None:
    """Verify synthetic Tier 2 directories run to `max_model_number = 2` and
    reproduce `.photo` restarts bit-identically without library-side extras.
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


def test_unimplemented_create_initial_model_and_missing_hook_fail_loudly(
    tmp_path: pathlib.Path,
) -> None:
    """Verify unimplemented `create_initial_model` and missing
    `other_build_initial_model` hooks fail loudly without ZAMS fallback
    (`#148`).
    """
    init_dir = tmp_path / "create_init"
    init_dir.mkdir(parents=True, exist_ok=True)
    (init_dir / "rn").write_text(
        "#!/usr/bin/env bash\ndo_one inlist final.mod\n", encoding="utf-8"
    )
    (init_dir / "inlist").write_text(
        "&star_job\n  create_initial_model = .true.\n/\n"
        "&controls\n  initial_z = 0.02\n  max_model_number = 2\n/\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match=(
            r"create_initial_model = \.true\. is not yet implemented \(#154\)"
        ),
    ):
        e2e.run_test_suite_case(init_dir, max_steps_per_stage=2)

    hook_dir = tmp_path / "missing_build_hook"
    hook_dir.mkdir(parents=True, exist_ok=True)
    (hook_dir / "rn").write_text(
        "#!/usr/bin/env bash\ndo_one inlist final.mod\n", encoding="utf-8"
    )
    (hook_dir / "inlist").write_text(
        "&star_job\n  use_other_build_initial_model = .true.\n/\n"
        "&controls\n  initial_z = 0.02\n  max_model_number = 2\n/\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"other_build_initial_model"):
        e2e.run_test_suite_case(hook_dir, max_steps_per_stage=2)


@pytest.mark.parametrize("case_name", _REPRESENTATIVE_TIER2_CASES)
def test_upstream_tier2_test_suite_case_and_restart_reproducibility(
    case_name: str,
    tmp_path: pathlib.Path,
) -> None:
    """Verify unmodified Tier 2 cases fail strictly under `#147` and cleaned
    copies follow declared stops or capped 2-step execution without in-library
    case overrides.
    """
    base = known_gaps_gate.require_mesa_test_inputs_dir()
    if base is None:
        return
    case_dir = base / "star" / "test_suite" / case_name
    assert case_dir.is_dir(), f"Upstream case directory not found: {case_dir}"

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
        cleaned_case_gaps.assert_declared_stop(
            gap,
            lambda: e2e.run_test_suite_case(
                clean_dir,
                skip_optional=True,
                max_steps_per_stage=2,
                verify_restart=True,
            ),
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
    if case_name == "make_co_wd":
        # Cleaned make_co_wd without extras or entropy relaxation (#154, #157)
        # terminates cleanly at min_timestep_limit under #152.
        assert rep["termination_code"] == "min_timestep_limit"
        assert rep["total_steps"] >= 2
        return
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
