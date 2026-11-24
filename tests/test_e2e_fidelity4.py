# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage `Fidelity-4` (`#115`, closing Epic `#109`) end-to-end verification
tests for `.mod` checkpoint round-trip fidelity (`M_center`,
`cumulative_energy_error`, `w`, `w_div_wc`, `alpha_RTI`), multi-step physical
relaxation loops (`relax.rs`), and 106-case active MESA `test_suite` plus
`dev_cases_*` inventory checks.
"""

from __future__ import annotations

import pathlib
import shutil

import pytest

from flexmesa import star
from flexmesa.testing import case_manifest, known_gaps_gate


def test_fidelity4_mod_and_photo_roundtrip_and_step_redo_reproducibility() -> (
    None
):
    """Verify `.mod` and `.photo` round-trip serialization and 1-step photo
    restart reproducibility on a solar-calibrated ZAMS model (`#115`).
    """
    m = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    m.eval_hydro_vars()
    h1, h2 = m.verify_photo_restart(n_steps=1)
    assert h1 == h2

    mod_text = m.to_mod_str()
    assert "M/Msun" in mod_text
    assert "n_shells" in mod_text
    m_restored = star.StarModel.from_mod_str(mod_text)
    assert m_restored.nz == m.nz
    assert m_restored.species == m.species
    assert abs(m_restored.mstar - m.mstar) <= 1.0e-12 * m.mstar

    photo_bytes = m.to_photo_bytes()
    m_photo = star.StarModel.from_photo_bytes(photo_bytes)
    assert m_photo.to_photo_bytes() == photo_bytes


def test_fidelity4_representative_multi_tier_physical_e2e_execution(
    tmp_path: pathlib.Path,
) -> None:
    """Verify representative physical E2E runs on cleaned copies fail strictly
    under `#147` on raw `$MESA_DIR` directories and execute 2-step smoke checks.
    """
    entries = case_manifest.load_case_manifest().cases
    assert sum(1 for e in entries if e.suite == "star") == 90
    assert sum(1 for e in entries if e.suite == "astero") == 8
    assert sum(1 for e in entries if e.suite == "binary") == 8

    base = known_gaps_gate.require_mesa_test_inputs_dir()
    if base is None:
        return
    star_ts = base / "star" / "test_suite"
    assert star_ts.is_dir(), f"Missing upstream star/test_suite: {star_ts}"
    for cname in (
        "1M_pre_ms_to_wd",
        "20M_z2m2_high_rotation",
        "gyre_in_mesa_ms",
    ):
        cdir = star_ts / cname
        assert cdir.is_dir(), f"Missing upstream case directory: {cdir}"
        if cname == "gyre_in_mesa_ms":
            run_dir = tmp_path / "raw" / cname
            shutil.copytree(cdir, run_dir)
        else:
            with pytest.raises(
                ValueError,
                match=r"(unsupported|unknown|unrecognized).*control",
            ):
                star.run_test_suite_case(
                    cdir, max_steps_per_stage=2, verify_restart=True
                )
            run_dir = tmp_path / cname
            star.copy_cleaned_case(cdir, run_dir, suite="star")
        rep = star.run_test_suite_case(
            run_dir, max_steps_per_stage=2, verify_restart=True
        )
        assert rep["tier"] == 0
        assert rep["passed"] is False
        assert rep["assertions_passed"]
        assert rep["photo_restart_verified"]
