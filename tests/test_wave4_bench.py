# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the Wave 4 (`star_data` & `star`) benchmark suite and ZAMS parity.

Verifies that:
1. All 5 Wave 4 benchmark kernels (`bench_star_micro_vars`,
   `bench_star_hydro_eqns`, `bench_star_newton_step`, `bench_star_remesh`, and
   `bench_star_evolve_step`) and `run_all_wave4_benchmarks` execute cleanly
   across `rust` and `jax` pathways and return positive latencies and finite
   numerical parity digests.
2. `StarModel.from_zams(m, 0.02)` initializes cleanly across all built-in ZAMS
   catalog masses (`0.51, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0 Msun`).
3. Evolving a `4.0 Msun, Z = 0.02` ZAMS model to `star_age = 1.0e5 yr` matches
   upstream Fortran MESA `star/test/LOGS/history.data` (`lg_Tcntr`, `lg_Dcntr`,
   `lg_L`, `lg_R`, `lg_Teff`, `H_cntr`) to within `<= 1.0e-3` dex.
"""

from __future__ import annotations

import numpy as np

from flexmesa import bench, star


def test_wave4_bench_kernels_return_positive_throughput() -> None:
    """All 5 Wave 4 benchmark kernels return valid timing and parity records."""
    records = bench.run_all_wave4_benchmarks(
        zone_counts=(8,), n_warmup=1, n_repeats=3
    )
    assert len(records) == 5
    workloads = {r.workload for r in records}
    assert workloads == {
        "star_micro_vars",
        "star_hydro_eqns",
        "star_newton_step",
        "star_remesh",
        "star_evolve_step",
    }
    for rec in records:
        assert rec.n_zones == 8
        assert rec.stage0.median_us > 0.0
        assert rec.stage1.median_us > 0.0
        assert rec.stage2.median_us > 0.0
        assert rec.stage3.median_us > 0.0
        assert rec.max_rel_err < 1.0e-11
        assert np.isfinite(rec.checksum)


def test_zams_catalog_masses_initialize_without_error() -> None:
    """All built-in ZAMS catalog masses initialize with finite microphysics."""
    for mass in (0.51, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0):
        model = star.StarModel.from_zams(initial_mass=mass, initial_z=0.02)
        assert model.nz > 100
        assert np.all(np.isfinite(model.peos))
        assert np.all(np.isfinite(model.gradt))
        assert np.all(model.chi_t > 0.0)


def test_four_msun_zams_evolve_matches_upstream_fortran_mesa_star_test() -> (
    None
):
    """Evolving 4 Msun ZAMS to 1e5 yr matches Fortran MESA within 1e-3 dex."""
    model = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    reports = model.evolve(
        max_steps=20,
        do_remesh=False,
        op_split_burn=False,
        use_bcyclic=True,
        max_age=1.0e5,
    )
    assert len(reports) == 11
    assert all(r.converged for r in reports)
    row = model.history_row()
    assert abs(row.star_age - 1.0e5) < 1.0e-6
    assert abs(row.log_center_t - 7.4124) < 1.0e-3
    assert abs(row.log_center_rho - 1.4368) < 1.0e-3
    assert abs(row.log_l - 2.3746) < 1.0e-3
    assert abs(row.log_r - 0.3703) < 1.0e-3
    assert abs(row.log_teff - 4.1698) < 1.0e-3
    assert abs(row.center_h1 - 0.696725) < 5.0e-5
