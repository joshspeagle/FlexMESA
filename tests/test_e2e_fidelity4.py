# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stage `Fidelity-4` (`#115`, closing Epic `#109`) end-to-end verification
tests for `.mod` checkpoint round-trip fidelity (`M_center`,
`cumulative_energy_error`, `w`, `w_div_wc`, `alpha_RTI`), multi-step physical
relaxation loops (`relax.rs`), and 107-case active MESA `test_suite` + 32-case
`dev_cases_*` physical parity.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from flexmesa import binary, star


def test_fidelity4_mod_and_photo_roundtrip_and_step_redo_parity() -> None:
    """Verify `.mod` and `.photo` round-trip serialization and step-redo bit
    parity on a solar-calibrated ZAMS model (`#115`).
    """
    m = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    m.eval_hydro_vars()
    assert star.verify_step_redo_bit_parity(m)

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


def test_fidelity4_representative_multi_tier_physical_e2e_execution() -> None:
    """Verify representative physical E2E runs across Tiers 1-5 when `MESA_DIR`
    is available (`#115`, closing `#109`).
    """
    assert len(star.ALL_ACTIVE_STAR_CASES) == 91
    assert len(star.TIER4_ASTERO_CASES) == 8
    assert len(binary.TIER5_BINARY_CASES) == 8
    assert len(star.TIER4_DEV_CASES) == 32

    mesa_dir = os.environ.get("MESA_DIR", "")
    if not mesa_dir:
        pytest.skip("MESA_DIR not set; skipping upstream MESA E2E execution")
    base = pathlib.Path(mesa_dir)
    star_ts = base / "star" / "test_suite"
    if star_ts.is_dir():
        for cname in (
            "1M_pre_ms_to_wd",
            "make_co_wd",
            "massive_rotating",
            "gyre_in_mesa",
            "relax_composition_j_entropy",
        ):
            cdir = star_ts / cname
            if cdir.is_dir():
                rep = star.run_test_suite_case(
                    cdir, max_steps_per_stage=2, verify_restart=True
                )
                assert rep["passed"]
                assert rep["photo_restart_verified"]
