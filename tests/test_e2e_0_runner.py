# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration tests for Stage E2E-0 (`inlist` recursive bundle parser, `./rn`
stage resolver, `.mod` / `.photo` bit-identical restart parity, and multi-stage
`test_suite` execution driver).
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import pytest

from flexmesa import star


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
            "  required_termination_code_string = 'max_model_number'\n"
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
        assert bundle["required_termination_code_string"] == "max_model_number"


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
    """When `MESA_DIR` is set in the environment, verify all upstream `inlist`
    and `rn` scripts across `star/test_suite`, `binary/test_suite`, and
    `astero/test_suite` parse without error.
    """
    mesa_dir = os.environ.get("MESA_DIR", "")
    if not mesa_dir:
        pytest.skip("MESA_DIR not set; skipping upstream MESA parse audit")
    base = pathlib.Path(mesa_dir)
    assert base.is_dir(), f"MESA_DIR is not a directory: {base}"
    for suite in ("star/test_suite", "binary/test_suite", "astero/test_suite"):
        suite_dir = base / suite
        assert suite_dir.is_dir(), f"Missing upstream suite dir: {suite_dir}"
        for case_dir in sorted(suite_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            stages = star.parse_rn_stages(str(case_dir), skip_optional=True)
            assert len(stages) >= 1
            for inlist_file in sorted(case_dir.glob("inlist*")):
                if inlist_file.is_file():
                    bundle = star.parse_inlist_bundle(str(inlist_file))
                    assert "entry_inlist" in bundle
            for mod_file in sorted(case_dir.glob("*.mod")):
                if mod_file.is_file():
                    mod_text = mod_file.read_text(encoding="utf-8")
                    loaded = star.StarModel.from_mod_str(mod_text)
                    assert loaded.nz >= 2
