# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for Stage E2E-0 (`inlist` recursive bundle parser, `./rn`
stage resolver, `.mod`/`.photo` self-roundtrip reproducibility, and multi-stage
`test_suite` execution driver).
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from flexmesa import star
from flexmesa.testing import known_gaps_gate


def test_parse_inlist_bundle_recursively_resolves_extra_inlists() -> None:
    """Verify `parse_inlist_bundle` merges `read_extra_*_inlist(1..5)`."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "inlist_shared").write_text(
            "&star_job\n"
            "  change_net = .true.\n"
            "  new_net_name = 'co_burn.net'\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 2.5\n"
            "  initial_z = 0.014\n"
            "/\n",
            encoding="utf-8",
        )
        (root / "inlist").write_text(
            "&star_job\n"
            "  read_extra_star_job_inlist(1) = .true.\n"
            "  extra_star_job_inlist_name(1) = 'inlist_shared'\n"
            "  save_model_filename = 'after_stage1.mod'\n"
            "/\n"
            "&controls\n"
            "  read_extra_controls_inlist1 = .true.\n"
            "  extra_controls_inlist1_name = 'inlist_shared'\n"
            "  max_model_number = 4\n"
            "/\n",
            encoding="utf-8",
        )
        bundle = star.parse_inlist_bundle(str(root / "inlist"))
        assert bundle["change_net"] is True
        assert bundle["new_net_name"] == "co_burn.net"
        assert bundle["initial_mass"] == 2.5
        assert bundle["initial_z"] == 0.014
        assert bundle["max_model_number"] == 4
        assert bundle["save_model_filename"] == "after_stage1.mod"


def test_parse_rn_stages_honors_mesa_skip_optional_branches() -> None:
    """Verify `parse_rn_stages` switches between fast and full `rn` branches."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "rn").write_text(
            "#!/usr/bin/env bash\n"
            'if [ -n "$MESA_SKIP_OPTIONAL" ]; then\n'
            "   cp standard_zams.mod zams.mod\n"
            "else\n"
            "   do_one inlist_to_zams_header zams.mod\n"
            "fi\n"
            "do_one inlist_main_header final.mod\n",
            encoding="utf-8",
        )
        fast_stages = star.parse_rn_stages(str(root), skip_optional=True)
        assert fast_stages == [
            ("cp", "standard_zams.mod", "zams.mod"),
            ("do_one", "inlist_main_header", "final.mod"),
        ]
        full_stages = star.parse_rn_stages(str(root), skip_optional=False)
        assert full_stages == [
            ("do_one", "inlist_to_zams_header", "zams.mod"),
            ("do_one", "inlist_main_header", "final.mod"),
        ]


def test_star_model_verify_photo_restart_is_bit_identical() -> None:
    """Verify `.photo` checkpoint restart produces identical state hashes."""
    model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    c_cont, c_rest = model.verify_photo_restart(n_steps=2)
    assert c_cont == c_rest
    assert c_cont != 0


def test_run_test_suite_case_executes_two_stage_pipeline() -> None:
    """Verify `run_test_suite_case` executes a 2-stage ZAMS-to-saved-model
    pipeline.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "inlist_stage1").write_text(
            "&star_job\n"
            "  save_model_when_terminate = .true.\n"
            "  save_model_filename = 'after_stage1.mod'\n"
            "/\n"
            "&kap\n"
            "  zbase = 0.02\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  mesh_delta_coeff = 0.0\n"
            "  max_model_number = 3\n"
            "/\n",
            encoding="utf-8",
        )
        (root / "inlist_stage2").write_text(
            "&star_job\n"
            "  load_saved_model = .true.\n"
            "  load_model_filename = 'after_stage1.mod'\n"
            "  change_net = .true.\n"
            "  new_net_name = 'co_burn.net'\n"
            "/\n"
            "&kap\n"
            "  zbase = 0.02\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  mesh_delta_coeff = 0.0\n"
            "  max_model_number = 5\n"
            "/\n",
            encoding="utf-8",
        )
        (root / "rn").write_text(
            "#!/usr/bin/env bash\n"
            "do_one inlist_stage1 after_stage1.mod\n"
            "do_one inlist_stage2 final.mod\n",
            encoding="utf-8",
        )
        rep = star.run_test_suite_case(
            str(root),
            skip_optional=True,
            max_steps_per_stage=10,
            verify_restart=True,
        )
        assert rep["photo_restart_verified"] is True
        stages = rep["stages"]
        assert isinstance(stages, list)
        assert len(stages) == 2
        assert stages[0]["termination_code"] == "max_model_number"
        assert stages[1]["termination_code"] == "max_model_number"
        assert (root / "after_stage1.mod").exists()
        assert (root / "final.mod").exists()


def test_optional_upstream_mesa_dir_inlists_and_mods_parse() -> None:
    """Verify all upstream `rn` scripts and `.mod` files parse cleanly and
    unmodified upstream `inlist*` files reject unsupported controls loudly under
    default strict validation.
    """
    base = known_gaps_gate.require_mesa_test_inputs_dir()
    if base is None:
        return
    # Models whose network file is not shipped with FlexMESA now fail to load
    # with the missing file named instead of falling back to another network
    # (declared gaps for custom_rates #161 and wd_aic #152).
    declared_unloadable_mods = {
        "star/test_suite/custom_rates/before_flash.mod": "nco.net",
        "star/test_suite/wd_aic/wd_aic.mod": "wd_aic.net",
    }
    unloadable_seen: set[str] = set()
    # `test_memory` is a standalone memory-leak harness whose `rn` runs
    # `./test_memory` and has no `do_one`; it is not one of the 106 cases.
    non_runnable_dirs = {"star/test_suite/test_memory"}
    non_runnable_seen: set[str] = set()
    for suite in ("star/test_suite", "binary/test_suite", "astero/test_suite"):
        suite_dir = base / suite
        assert suite_dir.is_dir(), f"Missing upstream suite dir: {suite_dir}"
        for case_dir in sorted(suite_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            rel_case = case_dir.relative_to(base).as_posix()
            try:
                stages = star.parse_rn_stages(str(case_dir), skip_optional=True)
            except ValueError as exc:
                assert rel_case in non_runnable_dirs, f"{rel_case}: {exc}"
                assert "no runnable stages" in str(exc)
                non_runnable_seen.add(rel_case)
                continue
            assert len(stages) >= 1
            for inlist_file in sorted(case_dir.glob("inlist*")):
                if inlist_file.is_file():
                    try:
                        bundle = star.parse_inlist_bundle(str(inlist_file))
                        assert "entry_inlist" in bundle
                    except ValueError as exc:
                        msg = str(exc).lower()
                        assert (
                            "unsupported" in msg
                            or "unknown namelist section" in msg
                            or "unknown control" in msg
                            or "unknown scalar control" in msg
                            or "missing" in msg
                        ), f"unexpected parse failure on {inlist_file}: {exc}"
            for mod_file in sorted(case_dir.glob("*.mod")):
                if mod_file.is_file():
                    mod_text = mod_file.read_text(encoding="utf-8")
                    rel = mod_file.relative_to(base).as_posix()
                    if rel in declared_unloadable_mods:
                        with pytest.raises(
                            ValueError,
                            match=declared_unloadable_mods[rel],
                        ):
                            star.StarModel.from_mod_str(mod_text)
                        unloadable_seen.add(rel)
                        continue
                    loaded = star.StarModel.from_mod_str(mod_text)
                    assert loaded.nz >= 2
    assert unloadable_seen == set(declared_unloadable_mods)
    assert non_runnable_seen == non_runnable_dirs
