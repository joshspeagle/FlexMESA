# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stage E2E-3 (#94) verification test suite for massive-star and compact
`star/test_suite` execution without in-library case overrides (`#148`).
"""

from __future__ import annotations

import pathlib
import shutil
from typing import Any

import pytest

from flexmesa import star
from flexmesa.testing import known_gaps_gate

from . import cleaned_case_gaps

_REPRESENTATIVE_TIER3_CASES = (
    "high_mass",
    "20M_z2m2_high_rotation",
    "tzo",
)
_STRICT_VALID_TIER3_CASES = frozenset({"high_mass"})


@pytest.mark.parametrize(
    "case_name",
    [
        "high_mass",
        "hot_cool_wind",
        "high_rot_darkening",
        "15M_dynamo",
        "conserve_angular_momentum",
        "magnetic_braking",
        "12M_pre_ms_to_core_collapse",
        "ccsn_IIp",
        "pisn",
        "hse_riemann",
        "ns_h",
        "tzo",
        "ppisn",
        "twin_studies",
    ],
)
def test_synthetic_tier3_case_execution_and_photo_restart(
    tmp_path: pathlib.Path,
    case_name: str,
) -> None:
    """Verify synthetic offline Tier 3 case execution to `max_model_number = 2`
    and bit-identical `.photo` restart without library-side extras.
    """
    case_dir = tmp_path / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    if case_name == "twin_studies":
        (case_dir / "rn").write_text(
            "#!/usr/bin/env bash\ndo_one\n",
            encoding="utf-8",
        )
        (case_dir / "inlist_multi_stars_job").write_text(
            "&multi_stars_job\n"
            "  num_stars = 2\n"
            "  inlist_names(1) = 'inlist_star1'\n"
            "  inlist_names(2) = 'inlist_star2'\n"
            "/\n",
            encoding="utf-8",
        )
        for sname in ("inlist_star1", "inlist_star2"):
            (case_dir / sname).write_text(
                "&star_job\n/\n&eos\n/\n&kap\n  zbase = 0.02\n/\n&controls\n"
                "  initial_mass = 20.0\n  initial_z = 0.02\n"
                "  max_model_number = 2\n/\n",
                encoding="utf-8",
            )
    else:
        (case_dir / "rn").write_text(
            "#!/usr/bin/env bash\ndo_one inlist final.mod\n",
            encoding="utf-8",
        )
        extra_ctrl = "  x_integer_ctrl(1) = 4\n" if case_name == "tzo" else ""
        (case_dir / "inlist").write_text(
            "&star_job\n"
            "  save_model_when_terminate = .true.\n"
            "  save_model_filename = 'final.mod'\n"
            "/\n&eos\n/\n&kap\n  zbase = 0.02\n/\n&controls\n"
            "  initial_mass = 15.0\n"
            "  initial_z = 0.02\n"
            "  max_model_number = 2\n"
            f"{extra_ctrl}"
            "/\n",
            encoding="utf-8",
        )
    rep = star.run_test_suite_case(
        case_dir,
        skip_optional=True,
        max_steps_per_stage=3,
        verify_restart=True,
    )
    assert rep["case_name"] == case_name
    assert rep["tier"] == 0
    assert rep["passed"] is True
    assert rep["assertions_passed"] is True
    assert rep["photo_restart_verified"] is True


@pytest.mark.parametrize("case_name", _REPRESENTATIVE_TIER3_CASES)
def test_upstream_mesa_tier3_test_suite_case_with_restart_reproducibility(
    case_name: str,
    tmp_path: pathlib.Path,
) -> None:
    """Reject raw Tier 3 cases on unsupported controls; run cleaned copies."""
    base = known_gaps_gate.require_mesa_test_inputs_dir()
    if base is None:
        return
    case_dir = base / "star" / "test_suite" / case_name
    assert case_dir.is_dir(), f"Upstream case directory not found: {case_dir}"

    if case_name in _STRICT_VALID_TIER3_CASES:
        raw_copy = tmp_path / "raw" / case_name
        shutil.copytree(case_dir, raw_copy)
        rep = star.run_test_suite_case(
            raw_copy,
            skip_optional=True,
            max_steps_per_stage=2,
            verify_restart=True,
        )
        assert rep["case_name"] == case_name
        assert rep["tier"] == 0
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
        star.run_test_suite_case(
            case_dir,
            skip_optional=True,
            max_steps_per_stage=2,
            verify_restart=True,
        )

    clean_dir = tmp_path / case_name
    star.copy_cleaned_case(case_dir, clean_dir, suite="star")

    def run() -> dict[str, Any]:
        return star.run_test_suite_case(
            clean_dir,
            skip_optional=True,
            max_steps_per_stage=2,
            verify_restart=True,
        )

    gap = cleaned_case_gaps.DECLARED_STAR_STOPS.get(case_name)
    if gap is not None:
        cleaned_case_gaps.assert_declared_stop(gap, run)
        return
    rep = run()
    assert rep["case_name"] == case_name
    assert rep["tier"] == 0
    assert rep["termination_code"] == "max_steps_cap"
    assert rep["passed"] is False
    assert rep["assertions_passed"] is True
    assert rep["photo_restart_verified"] is True
    assert rep["total_steps"] >= 1
    assert rep["final_teff"] > 0.0
