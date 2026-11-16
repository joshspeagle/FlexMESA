# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the Wave 4 (`star_data` & `star`) benchmark suite.

Checks that:
1. All 5 Wave 4 benchmark kernels (`bench_star_micro_vars`,
   `bench_star_hydro_eqns`, `bench_star_newton_step`, `bench_star_remesh`, and
   `bench_star_evolve_step`) and `run_all_wave4_benchmarks` execute across
   `rust` and `jax` pathways and match self-generated FlexMESA regression
   snapshots (internal regression checks, not upstream MESA outputs; tracked in
   #146).
2. `StarModel.from_zams(m, 0.02)` initializes across all built-in ZAMS catalog
   masses (`0.51, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0 Msun`).
3. Evolving a `4.0 Msun, Z = 0.02` ZAMS model to `star_age = 1.0e5 yr` is
   compared at every logged step (`model_number` 1, 5, 10, 11) against the
   committed MESA `26.4.1` `star/test/LOGS/history.data` fixture (`#141`,
   `G1-4`).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from flexmesa import bench, star

# Self-generated FlexMESA regression snapshots at N=8 (not upstream MESA
# outputs; `star_hydro_eqns`, `star_newton_step`, and `star_evolve_step`
# shifted under #141 when `neu` #193 and `num` #194 were aligned to MESA
# 26.4.1; tracked in #146).
_WAVE4_REGRESSION_SNAPSHOTS_N8: dict[str, float] = {
    "star_micro_vars": 2.497297961588147e2,
    "star_hydro_eqns": 4.253179678462525e6,
    "star_newton_step": 1.451336644709231e2,
    "star_remesh": 4.203476959958853e2,
    "star_evolve_step": 1.005942875453512e2,
}

_STAR_TEST_HISTORY = (
    pathlib.Path(__file__).resolve().parent
    / "fixtures"
    / "golden"
    / "star_test"
    / "history.data"
)


def _load_star_test_history_26_4_1() -> dict[int, dict[str, float]]:
    """Parse committed MESA `26.4.1` `star/test/LOGS/history.data` rows."""
    from flexmesa.testing import history_compare

    tbl = history_compare.load_history_file(_STAR_TEST_HISTORY)
    assert tbl.mesa_version_number == "r26.4.1"
    rows: dict[int, dict[str, float]] = {}
    for idx in range(tbl.num_rows):
        row = {name: float(arr[idx]) for name, arr in tbl.columns.items()}
        rows[round(row["model_number"])] = row
    return rows


@pytest.fixture(scope="module")
def evolved_four_msun_zams_by_step() -> dict[int, star.HistoryRow]:
    """Run 11 steps of `StarModel.from_zams(4.0, 0.02)` to `1.0e5 yr` once."""
    model = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    logged: dict[int, star.HistoryRow] = {}
    for step_idx in range(1, 11):
        rep = model.evolve_step(
            do_remesh=False, op_split_burn=False, use_bcyclic=True
        )
        assert rep.converged
        logged[step_idx] = model.history_row()
    # Final step 11 caps dt to land at max_age = 1.0e5 yr, matching star/test.
    final_reps = model.evolve(
        max_steps=5,
        do_remesh=False,
        op_split_burn=False,
        use_bcyclic=True,
        max_age=1.0e5,
    )
    assert len(final_reps) == 1
    assert final_reps[0].converged
    logged[11] = model.history_row()
    return logged


def test_wave4_bench_kernels_return_positive_throughput() -> None:
    """All 5 Wave 4 benchmark kernels match FlexMESA regression snapshots."""
    records = bench.run_all_wave4_benchmarks(
        zone_counts=(8,), n_warmup=1, n_repeats=3
    )
    assert len(records) == 5
    workloads = {r.workload for r in records}
    assert workloads == set(_WAVE4_REGRESSION_SNAPSHOTS_N8)
    for rec in records:
        assert rec.n_zones == 8
        assert rec.stage0.median_us > 0.0
        assert rec.stage1.median_us > 0.0
        assert rec.stage2.median_us > 0.0
        assert rec.stage3.median_us > 0.0
        assert rec.max_rel_err < 1.0e-10
        assert np.isfinite(rec.checksum)
        snap = _WAVE4_REGRESSION_SNAPSHOTS_N8[rec.workload]
        rel_err = abs(rec.checksum - snap) / abs(snap)
        assert rel_err <= 1.0e-12, (
            f"{rec.workload} (N=8) regression snapshot {rec.checksum:.15e} "
            f"!= {snap:.15e} (rel {rel_err:.3e})"
        )


def test_zams_catalog_masses_initialize_without_error() -> None:
    """All built-in ZAMS catalog masses initialize with finite microphysics."""
    for mass in (0.51, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0):
        model = star.StarModel.from_zams(initial_mass=mass, initial_z=0.02)
        assert model.nz > 100
        assert np.all(np.isfinite(model.peos))
        assert np.all(np.isfinite(model.gradt))
        assert np.all(model.chi_t > 0.0)


def test_four_msun_zams_evolve_matches_upstream_fortran_mesa_star_test(
    evolved_four_msun_zams_by_step: dict[int, star.HistoryRow],
) -> None:
    """Compare 4 Msun ZAMS evolution against MESA `26.4.1` `star/test` history.

    Note (`G1-4`): Over `1.0e5 yr` on the ZAMS, `center_h1` decreases by
    `2.54e-4` (tested to `5.0e-6` at every logged step), whereas `log_L`
    evolves by `3.89e-4` dex and `FlexMESA` (`do_remesh=False`) differs from
    MESA `26.4.1` by `9.19e-4` dex at step 11 (`#152`, `#153`).
    """
    ref_rows = _load_star_test_history_26_4_1()
    assert sorted(ref_rows) == [1, 5, 10, 11]
    logged = evolved_four_msun_zams_by_step

    # Step 1 precedes MESA's step-2 remesh (both models have 2,464 zones).
    step1 = logged[1]
    ref1 = ref_rows[1]
    assert abs(step1.log_center_rho - ref1["log_center_Rho"]) < 1.0e-4

    for mod_num, ref in ref_rows.items():
        row = logged[mod_num]
        assert abs(row.star_age - ref["star_age"]) < 1.0e-6
        assert abs(row.log_center_t - ref["log_center_T"]) < 1.0e-3
        assert abs(row.log_l - ref["log_L"]) < 1.0e-3
        assert abs(row.log_r - ref["log_R"]) < 1.0e-3
        assert abs(row.log_teff - ref["log_Teff"]) < 1.0e-3
        assert abs(row.center_h1 - ref["center_h1"]) < 5.0e-6


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MESA 26.4.1 star/test remeshes 2464 -> 992 zones at step 2; "
        "log_cntr_Rho differs by 1.11e-3 > 1.0e-3 at step 11 (#153)"
    ),
)
def test_four_msun_zams_evolve_log_cntr_rho_matches_mesa_26_4_1(
    evolved_four_msun_zams_by_step: dict[int, star.HistoryRow],
) -> None:
    """Assert `< 1e-3` dex on `log_cntr_Rho` across all `star/test` steps."""
    ref_rows = _load_star_test_history_26_4_1()
    for mod_num, ref in ref_rows.items():
        assert (
            abs(
                evolved_four_msun_zams_by_step[mod_num].log_center_rho
                - ref["log_center_Rho"]
            )
            < 1.0e-3
        )
