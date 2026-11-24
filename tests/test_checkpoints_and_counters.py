# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration tests for generated checkpoints and retry-proof counters."""

from __future__ import annotations

import numpy as np
import pytest

from flexmesa import binary, star
from flexmesa import constants as const


def test_star_counters_and_photo_bytes_roundtrip_bit_exact() -> None:
    """Verify StarModel counters and `to_photo_bytes` / `from_photo_bytes`."""
    m1 = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    m1.set_evolve_controls(mesh_delta_coeff=0.9, varcontrol_target=2.0e-4)
    rep1 = m1.evolve_step(do_remesh=True, op_split_burn=False, use_bcyclic=True)
    assert rep1.converged
    assert m1.total_num_solver_calls >= 1
    assert m1.micro_cache_hits >= 1
    assert m1.micro_cache_misses >= 1
    assert m1.num_retries >= 0
    assert m1.num_redos >= 0

    photo1 = m1.to_photo_bytes()
    m2 = star.StarModel.from_photo_bytes(photo1)
    assert m2.to_photo_bytes() == photo1
    assert np.array_equal(m2.lnd, m1.lnd)
    assert np.array_equal(m2.ln_t, m1.ln_t)
    assert m2.total_num_solver_calls == m1.total_num_solver_calls
    assert m2.num_retries == m1.num_retries
    assert m2.num_redos == m1.num_redos
    assert m2.micro_cache_hits == 0
    assert m2.micro_cache_misses == 0

    rep_cont = m1.evolve_step(
        do_remesh=True, op_split_burn=False, use_bcyclic=True
    )
    rep_rest = m2.evolve_step(
        do_remesh=True, op_split_burn=False, use_bcyclic=True
    )
    assert rep_cont.converged and rep_rest.converged
    assert m2.to_photo_bytes() == m1.to_photo_bytes()
    assert np.array_equal(m2.lnd, m1.lnd)
    assert np.array_equal(m2.ln_t, m1.ln_t)

    with pytest.raises(ValueError):
        star.StarModel.from_photo_bytes(photo1[:-1])
    with pytest.raises(ValueError):
        star.StarModel.from_photo_bytes(photo1 + b"\x00")


def test_binary_counters_and_checkpoint_roundtrip_bit_exact() -> None:
    """Verify BinarySystem counters and checkpoint byte round-tripping."""
    donor = star.StarModel.from_zams(initial_mass=1.5, initial_z=0.02)
    sys1 = binary.BinarySystem.from_point_mass(
        donor,
        companion_mass_msun=1.0,
        initial_period_days=0.45,
        eccentricity=0.0,
        is_black_hole=False,
    )
    dt_next, _ = sys1.step(1.0e3 * const.SECYER)
    assert sys1.model_number == 1
    assert sys1.num_implicit_trials >= 1
    assert sys1.num_retries >= 0
    assert sys1.num_dt_cuts >= 0
    assert sys1.num_hard_limit_retries >= 0
    calls_step1 = sys1.donor.total_num_solver_calls
    assert calls_step1 >= 1

    chk1 = sys1.to_photo_bytes()
    sys2 = binary.BinarySystem.from_photo_bytes(chk1)
    assert sys2.to_photo_bytes() == chk1
    assert sys2.model_number == sys1.model_number
    assert sys2.num_implicit_trials == sys1.num_implicit_trials
    assert sys2.num_retries == sys1.num_retries
    assert sys2.num_dt_cuts == sys1.num_dt_cuts
    assert sys2.num_hard_limit_retries == sys1.num_hard_limit_retries
    assert np.array_equal(sys2.donor.lnd, sys1.donor.lnd)

    dt_cont, _ = sys1.step(dt_next)
    dt_rest, _ = sys2.step(dt_next)
    assert dt_rest == dt_cont
    assert sys2.binary_age_yr == sys1.binary_age_yr
    assert (
        sys2.donor.total_num_solver_calls == sys1.donor.total_num_solver_calls
    )
    assert sys1.donor.total_num_solver_calls > calls_step1
    assert sys2.to_photo_bytes() == sys1.to_photo_bytes()

    with pytest.raises(ValueError):
        binary.BinarySystem.from_photo_bytes(chk1[:-1])
    with pytest.raises(ValueError):
        binary.BinarySystem.from_photo_bytes(chk1 + b"\xa5")
