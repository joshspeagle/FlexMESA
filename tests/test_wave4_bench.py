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
# 26.4.1, all five shifted under #149 when `&controls`, microphysics, and
# mesh-function defaults were aligned to MESA 26.4.1, `star_micro_vars`
# (2.497139571258105e2 -> 2.500111893102388e2), `star_hydro_eqns`
# (4.253179678855851e6 -> 4.243492123572694e6), and `star_newton_step`
# (1.451475459976951e2 -> 6.383737350497265) shifted under #155 when
# `remove_mixing_singletons` (`mix_info.f90:1026-1072`) switched the isolated
# `k = 2` convective singleton on the coarse `N = 8` grid to radiative, and
# `star_hydro_eqns`, `star_newton_step`, and `star_evolve_step` shifted under
# #152 when `set_xscale_info` and Newton step/equilibration semantics were
# aligned to MESA 26.4.1 (`DECISIONS #42`), and `star_micro_vars`
# (2.500111893102388e2 -> 2.499330102887365e2), `star_hydro_eqns`
# (4.243485684423328e6 -> 4.262783875690212e6) and `star_newton_step`
# (6.383773435545090 -> 2.210834731068863e5) shifted under #291 when the
# face radiation pressure in `do1_mlt_2` was changed to `crad*T_face^4/3`
# (`star_utils.f90:3838-3843`). Under #291 `star_evolve_step` also moved from
# the 8-zone grid (1.005928065548472e2) to `MIN_EVOLVE_STEP_ZONES = 64` zones
# (8.895997453722753e2), because the 8-zone fixture no longer converges at
# the model's timestep; tracked in #146). Under #340 (`eval_face_dm_bar` at
# `k = nz - 1` matching `star/private/hydro_momentum.f90:191-196` and
# `dt_limit_ratio_old = 0.0` matching `star/private/init.f90:509, 696`),
# `star_hydro_eqns` (4.262783875690212e6 -> 4.262780484077989e6),
# `star_newton_step` (2.210834731068863e5 -> 2.263803560298164e5), and
# `star_evolve_step` (8.895997453722753e2 -> 8.895028893306655e2) shifted. Under
# #342 (ZAMS `interp_q` matching
# `star/private/star_utils.f90:729-752, 1232-1265`, 8-decimal ZAMS index
# masses matching `star/private/init_model.f90:210, 254, 444-446`, and initial
# ZAMS `model_number = 0` matching `star/private/init.f90:367`),
# `star_micro_vars`
# (2.499330102887365e2 -> 2.253780782165544e2), `star_hydro_eqns`
# (4.262780484077989e6 -> 1.742907215943596e6), `star_newton_step`
# (2.263803560298164e5 -> 8.653548156900340e5), `star_remesh`
# (5.513777703025969e2 -> 5.455218449741383e2), and `star_evolve_step`
# (8.895028893306655e2 -> 8.843428194210857e2) shifted.
_WAVE4_REGRESSION_SNAPSHOTS_N8: dict[str, float] = {
    "star_micro_vars": 2.253780782165544e2,
    "star_hydro_eqns": 1.742907215943596e6,
    "star_newton_step": 8.653548156900340e5,
    "star_remesh": 5.455218449741383e2,
    "star_evolve_step": 8.843428194210857e2,
}

# Bound on the cross-stage (Rust against JAX) relative difference.
_WAVE4_CROSS_STAGE_BOUND = 1.0e-10

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
        if rec.workload == "star_evolve_step":
            assert rec.n_zones == bench.kernels_wave4.MIN_EVOLVE_STEP_ZONES
        else:
            assert rec.n_zones == 8
        assert rec.stage0.median_us > 0.0
        assert rec.stage1.median_us > 0.0
        assert rec.stage2.median_us > 0.0
        assert rec.stage3.median_us > 0.0
        if rec.workload not in {"star_micro_vars", "star_hydro_eqns"}:
            # Checked on their own below (#295).
            assert rec.max_rel_err < _WAVE4_CROSS_STAGE_BOUND
        assert np.isfinite(rec.checksum)
        snap = _WAVE4_REGRESSION_SNAPSHOTS_N8[rec.workload]
        rel_err = abs(rec.checksum - snap) / abs(snap)
        assert rel_err <= 1.0e-12, (
            f"{rec.workload} (N=8) regression snapshot {rec.checksum:.15e} "
            f"!= {snap:.15e} (rel {rel_err:.3e})"
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#295: Rust do1_mlt_2 takes the face radiation pressure from the "
        "face temperature (MESA 26.4.1 star_utils.f90:3838-3843, #291); "
        "the frozen JAX face-MLT twin still interpolates the cell values "
        "(measured max_rel_err 8.9e-3)."
    ),
)
def test_wave4_star_micro_vars_rust_matches_frozen_jax() -> None:
    """`star_micro_vars` Rust and JAX stages agree to the cross-stage bound."""
    rec = bench.bench_star_micro_vars(8, n_warmup=1, n_repeats=3)
    assert rec.max_rel_err < _WAVE4_CROSS_STAGE_BOUND


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#295: Rust eval_face_dm_bar at k = nz - 1 uses 0.5*(dm[k-1] + dm[k]) "
        "(MESA 26.4.1 star/private/hydro_momentum.f90:191-196, #340); "
        "frozen JAX eval_hydro_residuals_jax still uses dm_bar[nz - 1] "
        "(measured max_rel_err 9.73e-3)."
    ),
)
def test_wave4_star_hydro_eqns_rust_matches_frozen_jax() -> None:
    """`star_hydro_eqns` Rust and JAX stages agree to the cross-stage bound."""
    rec = bench.bench_star_hydro_eqns(8, n_warmup=1, n_repeats=3)
    assert rec.max_rel_err < _WAVE4_CROSS_STAGE_BOUND


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
