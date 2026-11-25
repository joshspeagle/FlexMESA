# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage E2E-2 (#93) Tier 2 substellar, planetary, white dwarf, and accretion
`test_suite` execution and loud-error checks (`#148`).
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
from typing import Any

import numpy as np
import pytest

from flexmesa.star import e2e
from flexmesa.testing import known_gaps_gate

from . import cleaned_case_gaps

_REPRESENTATIVE_TIER2_CASES = ("wd_aic",)


class _NovaWindObserver:
    """Read-only step observer recording mass, mdot, and surface H1."""

    def __init__(self) -> None:
        self.steps: list[tuple[int, float, float, float]] = []

    def extras_finish_step(self, state: dict[str, Any]) -> str:
        """Record per-step mass, accretion/wind rate, and surface H1."""
        self.steps.append(
            (
                int(state["model_number"]),
                float(state["star_mass"]),
                float(state["star_mdot"]),
                float(state["surface_h1"]),
            )
        )
        return "keep_going"


def _upstream_one_wd_model() -> pathlib.Path | None:
    """Resolve the upstream MESA `1.1M_lgTc_7.7.mod` ONe white dwarf."""
    base = known_gaps_gate.require_mesa_test_inputs_dir()
    if base is None:
        return None
    wd_mod = (
        base / "star" / "test_suite" / "wd_nova_burst" / "1.1M_lgTc_7.7.mod"
    )
    assert wd_mod.is_file(), f"Upstream ONe WD model not found: {wd_mod}"
    return wd_mod


def test_accretion_composition_and_super_eddington_nova_wind() -> None:
    """Verify accretion composition and super-Eddington nova wind execution on
    an evolved ONe white dwarf (`1.1M_lgTc_7.7.mod`, `wd_nova_burst`).
    """
    wd_mod = _upstream_one_wd_model()
    if wd_mod is None:
        return

    with tempfile.TemporaryDirectory() as tmp:
        case_dir = pathlib.Path(tmp) / "wd_nova_burst"
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wd_mod, case_dir / wd_mod.name)
        (case_dir / "inlist_accrete").write_text(
            "&star_job\n"
            "  load_saved_model = .true.\n"
            "  load_model_filename = '1.1M_lgTc_7.7.mod'\n"
            "  save_model_when_terminate = .true.\n"
            "  save_model_filename = 'accreted.mod'\n"
            "  set_initial_age = .true.\n"
            "  initial_age = 0\n"
            "  set_initial_model_number = .true.\n"
            "  initial_model_number = 0\n"
            "  set_initial_dt = .true.\n"
            "  years_for_initial_dt = 1.0d-3\n"
            "/\n"
            "&kap\n"
            "  zbase = 0.02\n"
            "  use_Type2_opacities = .true.\n"
            "/\n"
            "&controls\n"
            "  mass_change = 1.0d-9\n"
            "  accrete_same_as_surface = .false.\n"
            "  accretion_h1 = 0.70\n"
            "  accretion_he3 = 0.292910d-4\n"
            "  accretion_he4 = 0.279971\n"
            "  accretion_zfracs = 4\n"
            "  super_eddington_scaling_factor = 1.0\n"
            "  max_model_number = 2\n"
            "/\n",
            encoding="utf-8",
        )
        (case_dir / "inlist_wind").write_text(
            "&star_job\n"
            "  load_saved_model = .true.\n"
            "  load_model_filename = 'accreted.mod'\n"
            "/\n"
            "&kap\n"
            "  zbase = 0.02\n"
            "  use_Type2_opacities = .true.\n"
            "/\n"
            "&controls\n"
            "  mass_change = 1.0d-9\n"
            "  accrete_same_as_surface = .false.\n"
            "  accretion_h1 = 0.70\n"
            "  accretion_he3 = 0.292910d-4\n"
            "  accretion_he4 = 0.279971\n"
            "  accretion_zfracs = 4\n"
            "  super_eddington_scaling_factor = 1.0\n"
            "  super_eddington_wind_Ledd_factor = 1.0d-6\n"
            "  max_model_number = 4\n"
            "/\n",
            encoding="utf-8",
        )
        (case_dir / "rn").write_text(
            "#!/usr/bin/env bash\n"
            "do_one inlist_accrete accreted.mod\n"
            "do_one inlist_wind final.mod\n",
            encoding="utf-8",
        )
        obs = _NovaWindObserver()
        rep = e2e.run_test_suite_case(
            case_dir,
            skip_optional=True,
            max_steps_per_stage=3,
            verify_restart=True,
            extras=obs,
        )
        assert rep["case_name"] == "wd_nova_burst"
        assert rep["tier"] == 0
        assert rep["passed"] is True
        assert rep["assertions_passed"] is True
        assert rep["photo_restart_passed"] is True
        assert rep["photo_restart_verified"] is True
        assert rep["final_log_tc"] > 6.0
        assert len(rep["stages"]) == 2
        assert len(obs.steps) == 4
        # Stage 1 accretes H-rich Lodders 03 composition onto the pure-He WD.
        assert obs.steps[0][2] == pytest.approx(1.0e-9, rel=1.0e-12)
        assert obs.steps[1][1] > obs.steps[0][1] > 1.115
        assert obs.steps[1][3] > obs.steps[0][3] > 1.0e-5
        # Stage 2 triggers super-Eddington wind mass loss over accretion.
        assert obs.steps[2][2] < 0.0
        assert obs.steps[3][2] < 0.0
        assert obs.steps[3][1] < obs.steps[2][1] < obs.steps[1][1]


_MAKE_O_NE_WD_SYNTHETIC_INLIST = (
    "&star_job\n"
    "  load_saved_model = .true.\n"
    "  load_model_filename = '1.1M_lgTc_7.7.mod'\n"
    "  set_initial_model_number = .true.\n"
    "  initial_model_number = 0\n"
    "/\n"
    "&kap\n"
    "  zbase = 0.02\n"
    "  use_Type2_opacities = .true.\n"
    "/\n"
    "&controls\n"
    "  do_phase_separation = .true.\n"
    "  phase_separation_option = 'ONe'\n"
    "  max_model_number = 2\n"
    "/\n"
)


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
    ],
)
def test_synthetic_tier2_case_runners_pass_assertions_and_restart(
    case_name: str,
    inlist_body: str,
) -> None:
    """Verify synthetic Tier 2 directories run to `max_model_number = 2` and
    reproduce `.photo` restarts bit-identically without library-side extras.
    """
    _check_synthetic_tier2_case(case_name, inlist_body)


def test_synthetic_tier2_make_o_ne_wd_runner_and_restart() -> None:
    """Verify the ONe white dwarf `make_o_ne_wd` setup from
    `1.1M_lgTc_7.7.mod` passes physical assertions and `.photo` restart parity.
    """
    wd_mod = _upstream_one_wd_model()
    if wd_mod is None:
        return
    rep = _check_synthetic_tier2_case(
        "make_o_ne_wd",
        _MAKE_O_NE_WD_SYNTHETIC_INLIST,
        mod_src=wd_mod,
    )
    assert rep["final_log_tc"] > 7.0
    assert rep["final_log_rhoc"] > 7.5


def _check_synthetic_tier2_case(
    case_name: str,
    inlist_body: str,
    *,
    mod_src: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Runs one synthetic Tier 2 directory (see the tests above)."""
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = pathlib.Path(tmp) / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        if mod_src is not None:
            shutil.copy2(mod_src, case_dir / mod_src.name)
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
        return rep


def test_unimplemented_create_initial_model_and_missing_hook_fail_loudly(
    tmp_path: pathlib.Path,
) -> None:
    """Verify `create_initial_model` without a mass and a missing
    `other_build_initial_model` hook fail loudly without ZAMS fallback
    (`#148`, `#154`).

    Upstream stops when `mass_in_gm_for_create_initial_model <= 0`
    (`star/private/create_initial_model.f90:104-110`).
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
        match=r"must set mass_in_gm_for_create_initial_model",
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
    _check_tier2_case(case_name, tmp_path)


def test_upstream_irradiated_planet_runs_unmodified_with_relaxes_ported() -> (
    None
):
    """Verify unmodified irradiated_planet reads every control strictly
    (`relax_initial_tau_factor` and `relax_initial_irradiation` are ported,
    `#436`) and runs a capped smoke run with a bit-identical photo restart.
    """
    base = known_gaps_gate.require_mesa_test_inputs_dir()
    if base is None:
        return
    case_dir = base / "star" / "test_suite" / "irradiated_planet"
    assert case_dir.is_dir(), f"Upstream case directory not found: {case_dir}"
    rep = e2e.run_test_suite_case(
        case_dir,
        skip_optional=True,
        max_steps_per_stage=2,
        verify_restart=True,
    )
    assert rep["case_name"] == "irradiated_planet"
    assert rep["termination_code"] == "max_steps_cap"
    assert rep["passed"] is False
    assert rep["assertions_passed"] is True
    assert rep["photo_restart_verified"] is True
    assert rep["total_steps"] >= 1
    assert np.isfinite(rep["final_teff"]) and rep["final_teff"] > 0.0


def test_upstream_tier2_make_co_wd_runs_unmodified_with_relax_to_star_cut(
    tmp_path: pathlib.Path,
) -> None:
    """Verify unmodified make_co_wd is accepted strictly now that
    `remove_surface_by_relax_to_star_cut` (`do_relax_to_star_cut`,
    star/private/remove_shells.f90:1156-1404) is ported (`#436`), and runs
    its first stages. The envelope cut is reached through
    `star_remove_surface_at_cell_k` in the case's `run_star_extras`
    (`#153`), which these two-step stages do not reach.
    """
    _check_tier2_case("make_co_wd", tmp_path, strict_rejects=False)


def _check_tier2_case(
    case_name: str, tmp_path: pathlib.Path, strict_rejects: bool = True
) -> None:
    """Runs one Tier 2 case unmodified, then cleaned (see the tests above).

    With `strict_rejects=False` the unmodified case must be accepted, and it
    is run directly in place of the cleaned copy.
    """
    base = known_gaps_gate.require_mesa_test_inputs_dir()
    if base is None:
        return
    case_dir = base / "star" / "test_suite" / case_name
    assert case_dir.is_dir(), f"Upstream case directory not found: {case_dir}"

    if strict_rejects:
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
    else:
        clean_dir = tmp_path / case_name
        shutil.copytree(case_dir, clean_dir)

    def run() -> dict[str, Any]:
        return e2e.run_test_suite_case(
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
    assert rep["total_newton_iters"] >= 1
    assert np.isfinite(rep["final_teff"]) and rep["final_teff"] > 0.0
    assert np.isfinite(rep["final_log_l"])
    assert np.isfinite(rep["final_log_r"])
    assert np.isfinite(rep["final_log_tc"])
    assert np.isfinite(rep["final_log_rhoc"])
