#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Render `#168` stellar evolution profiling reports from raw JSON artifacts.

Generates two deterministic markdown documents from
`benchmarks/evolution_profile/26.4.1/*.json`:
1. `notes/audit/evolution_profile_26_4_1.md` (audit record for `#168`).
2. `docs/benchmarks/evolution-profiling.md` (MkDocs benchmark page with an
   executable Python verification snippet).

Pass `--check` to verify that both tracked markdown files match the freshly
rendered output byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, cast

try:
    import _mesa_profile_helpers as helpers
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import _mesa_profile_helpers as helpers

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_IN_DIR = _REPO_ROOT / "benchmarks" / "evolution_profile" / "26.4.1"
_DEFAULT_AUDIT_OUT = (
    _REPO_ROOT / "notes" / "audit" / "evolution_profile_26_4_1.md"
)
_DEFAULT_DOCS_OUT = (
    _REPO_ROOT / "docs" / "benchmarks" / "evolution-profiling.md"
)

_CASE_PHYSICS_NOTES: dict[str, str] = {
    "1.3M_ms_high_Z": (
        "`basic.net` (8 species), Cox MLT, fixed `gs98` opacity, "
        "CNO main sequence (`Z = 0.04`) to `log_L_upper_limit`"
    ),
    "low_z": (
        "`basic.net` (8 species), Henyey MLT, Type-2 CO opacity, "
        "Eddington `T(tau)` atmosphere (`Z = 1e-4`) to central H depletion"
    ),
    "high_z": (
        "`o18_and_ne22.net` (10 species), Type-2 CO opacity, exponential "
        "overshooting, Reimers+Vink winds, core H and He depletion"
    ),
    "make_metals": (
        "`basic.net` (8 species), Population III (`Z = 0`) pre-MS contraction "
        "through core H and He depletion"
    ),
}


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _fmt_pct(val: float, digits: int = 2, signed: bool = False) -> str:
    fmt = f"{'+' if signed else ''}.{digits}f"
    return f"{format(val, fmt)}%"


def _build_shared_sections(
    in_dir: pathlib.Path, *, for_docs: bool
) -> list[str]:
    prov = _read_json(in_dir / "provenance.json")
    strict_cases = cast(
        list[dict[str, Any]],
        _read_json(in_dir / "flexmesa_cases_strict.json")["cases"],
    )
    mesa_cases = cast(
        list[dict[str, Any]],
        _read_json(in_dir / "mesa_26_4_1_cases.json")["cases"],
    )
    svf_cases = cast(
        list[dict[str, Any]],
        _read_json(in_dir / "strict_vs_fast_trajectories.json")["cases"],
    )
    waste = _read_json(in_dir / "waste_removal_150.json")
    batch = _read_json(in_dir / "batch_scaling_rust.json")
    accel = _read_json(in_dir / "accelerator_grid_estimate.json")

    strict_by_case = {str(c["case"]): c for c in strict_cases}
    mesa_by_case = {str(c["case"]): c for c in mesa_cases}
    svf_by_case = {str(c["case"]): c for c in svf_cases}

    hw = cast(dict[str, Any], prov["hardware"])
    tc = cast(dict[str, Any], prov["toolchains"])
    qh = cast(dict[str, Any], prov["quiet_host"])
    main_short = str(prov["flexmesa_origin_main_commit"])[:7]
    branch_short = str(prov["flexmesa_branch_commit"])[:7]

    lines: list[str] = []
    if for_docs:
        lines.append(
            "# Stellar Evolution Profiling & Optimization Tradeoffs (`#168`)"
        )
    else:
        lines.append(
            "# Audit A6 (`#168`): Real Evolution Profile, Waste Removal, "
            "`strict` vs `fast`, and Batching"
        )
    lines.append("")
    lines.append(
        "This report is rendered deterministically by "
        "`scripts/audit/render_evolution_profile_report.py` from raw JSON and "
        "CSV captures in `benchmarks/evolution_profile/26.4.1/`. Every timing "
        "and trajectory comparison below was recorded on a single quiet host "
        "against compiled Fortran MESA `r26.4.1` (`commit c8c2026`, Zenodo "
        "`19722306`) at matched thread count."
    )
    lines.append("")

    # Provenance block
    lines.append("## 1. Measurement Provenance & Selected Cases")
    lines.append("")
    lines.append(
        f"- **Capture Timestamp (UTC)**: `{prov['date_utc']}` "
        f"(1-minute host load average `{qh['load_avg_1m_start']:.2f}` "
        f"at start; `{qh['load_avg_1m_end']:.2f}` at completion immediately "
        f"after the `8`-thread `N = 32` worker-pool scaling burst)"
    )
    lines.append(
        f"- **Host Hardware**: `{hw['cpu_model']}` "
        f"(`{hw['physical_cores']}` physical / `{hw['logical_cores']}` logical "
        f"cores, `{hw['total_ram_gib']:.2f} GiB` RAM, `{hw['os']}`, GPU "
        f"`{hw['gpu_model']}`)"
    )
    lines.append(
        f"- **Upstream MESA Toolchain**: `{tc['mesa_release']}` "
        f"(`commit {tc['mesa_commit']}`, Zenodo `{tc['mesa_zenodo_record']}`), "
        f"MESA SDK `{tc['mesa_sdk']}`, `{tc['gfortran']}`, `CRMATH`"
    )
    lines.append(
        f"- **FlexMESA Toolchain**: `{tc['rustc']}`, `{tc['python']}`, "
        f"`JAX {tc['jax']}`, `origin/main` base `{main_short}` (driver commit "
        f"`{branch_short}`)"
    )
    lines.append(
        f"- **Matched Thread Caps**: "
        f"`OMP_NUM_THREADS={tc['omp_num_threads']}`, "
        f"`RAYON_NUM_THREADS={tc['rayon_num_threads']}`, "
        f"`OPENBLAS_NUM_THREADS={tc['omp_num_threads']}`"
    )
    lines.append("")
    lines.append(
        "Four `star/test_suite` cases that run to upstream's physical stopping "
        "condition on `origin/main` with stock `run_star_extras.f90` hooks "
        "were selected to span fixed and Type-2 CO opacities, 8-species and "
        "10-species nuclear networks, Cox and Henyey MLT, stellar winds, "
        "convective overshooting, and zero-metallicity pre-MS-through-core-He "
        "evolution (`star/test_suite/semiconvection` currently stops at step "
        "197 on `origin/main` awaiting the `#149` Type-2 `a09_co` defaults "
        "alignment):"
    )
    lines.append("")
    for cname in prov["selected_cases"]:
        lines.append(f"- **`{cname}`**: {_CASE_PHYSICS_NOTES[cname]}.")
    lines.append("")
    lines.append(
        "| Case | Termination (`strict` / MESA) | Steps (`strict` / MESA) | "
        "Zones (`init -> final`) | Final Age (`strict` / MESA, yr) | "
        "`d_age / age` | Max Track `|d log L|` | Max Track `|d log Teff|` | "
        "Max Track `|d log rho_c|` |"
    )
    lines.append(
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for cname in prov["selected_cases"]:
        sc = strict_by_case[cname]
        mc = mesa_by_case[cname]
        cmp = cast(dict[str, Any], mc["strict_vs_mesa_comparison"])
        cols = cast(dict[str, Any], cmp["columns"])
        rel_age_pct = 100.0 * float(cmp["rel_star_age_diff"])
        lines.append(
            f"| `{cname}` | `{sc['termination_code']}` / "
            f"`{mc['termination_code']}` | "
            f"`{sc['steps_taken']}` / `{mc['steps_taken']}` | "
            f"`{sc['initial_zones']}->{sc['final_zones']}` | "
            f"`{sc['final_star_age_yr']:.4e}` / "
            f"`{mc['final_star_age_yr']:.4e}` | "
            f"`{_fmt_pct(rel_age_pct, 2, signed=True)}` | "
            f"`{cols['log_L']['track_max_abs_diff']:.4f}` | "
            f"`{cols['log_Teff']['track_max_abs_diff']:.4f}` | "
            f"`{cols['log_cntr_Rho']['track_max_abs_diff']:.4f}` |"
        )
    lines.append("")

    # Section 2: Per-step phase breakdown
    lines.append("## 2. Per-Step Phase Breakdown & Microphysics Call Counts")
    lines.append("")
    lines.append(
        "Unperturbed wall times (`repeats = 3` median) are measured without "
        "instrumentation hooks inside `flexmesa-star`. A second pass "
        "(`profile_evolution case --with-phase-profile`) times read-only step "
        "probes cloned before each step while verifying a `0-ULP` match on the "
        "serialized `.photo` state checksum so phase attribution never "
        "perturbs the canonical evolution trajectory."
    )
    lines.append("")
    lines.append("### 2.1 Share of Step Wall Time by Subsystem (`%`)")
    lines.append("")
    lines.append(
        "| Case | EOS | Opacity | Net + Jacobian | Turbulence | "
        "Eqn Assembly | Block Solve | Remesh | Bookkeeping & Copies |"
    )
    lines.append(
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for cname in prov["selected_cases"]:
        ph = cast(dict[str, Any], strict_by_case[cname]["phases"])
        lines.append(
            f"| `{cname}` | "
            f"`{ph['eos_share_pct']:.1f}%` | "
            f"`{ph['kap_share_pct']:.1f}%` | "
            f"`{ph['net_and_jacobian_share_pct']:.1f}%` | "
            f"`{ph['turbulence_share_pct']:.1f}%` | "
            f"`{ph['equation_assembly_share_pct']:.1f}%` | "
            f"`{ph['block_solve_share_pct']:.1f}%` | "
            f"`{ph['remesh_share_pct']:.1f}%` | "
            f"`{ph['bookkeeping_and_copies_share_pct']:.1f}%` |"
        )
    mean_sh = cast(dict[str, float], accel["mean_evolution_phase_shares_pct"])
    lines.append(
        f"| **4-case mean** | "
        f"**`{mean_sh['eos_share_pct']:.1f}%`** | "
        f"**`{mean_sh['kap_share_pct']:.1f}%`** | "
        f"**`{mean_sh['net_and_jacobian_share_pct']:.1f}%`** | "
        f"**`{mean_sh['turbulence_share_pct']:.1f}%`** | "
        f"**`{mean_sh['equation_assembly_share_pct']:.1f}%`** | "
        f"**`{mean_sh['block_solve_share_pct']:.1f}%`** | "
        f"**`{mean_sh['remesh_share_pct']:.1f}%`** | "
        f"**`{mean_sh['bookkeeping_and_copies_share_pct']:.1f}%`** |"
    )
    lines.append("")

    lines.append("### 2.2 Microphysics & Solver Evaluations per Accepted Step")
    lines.append("")
    lines.append(
        "| Case | Mean Zones | Newton Iters / Step | "
        "EOS/Kap/Net Sweeps / Step | Micro Zone Evals / Step | "
        "MLT Sweeps / Step | Block Solves / Step | "
        "`MicrophysicsCache` Hits / Step |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    hit_rates: list[float] = []
    for cname in prov["selected_cases"]:
        sc = strict_by_case[cname]
        ph = cast(dict[str, Any], sc["phases"])
        sweeps = float(ph["eos_evals_per_accepted_step"])
        mlt_sweeps = float(ph["mlt_evals_per_accepted_step"])
        hits_step = float(sc["micro_cache_hits_per_step"])
        zone_evals = sweeps * float(sc["mean_zones"])
        hit_rates.append(100.0 * hits_step / max(mlt_sweeps, 1.0e-12))
        lines.append(
            f"| `{cname}` | "
            f"`{sc['mean_zones']:.0f}` | "
            f"`{sc['newton_iters_per_step']:.2f}` | "
            f"`{sweeps:.2f}` | "
            f"`{zone_evals:.0f}` | "
            f"`{mlt_sweeps:.2f}` | "
            f"`{ph['linear_solves_per_accepted_step']:.2f}` | "
            f"`{hits_step:.2f}` |"
        )
    lines.append("")
    micro_tot_pct = 100.0 * float(accel["stateless_microphysics_fraction"])
    solver_tot_pct = 100.0 * float(accel["stateful_solver_remesh_fraction"])
    lines.append(
        f"Across the four evolutions, local zone microphysics (`EOS`, "
        f"`Opacity`, `Net + Jacobian`, and `Turbulence`) accounts for "
        f"`{micro_tot_pct:.1f}%` of per-step wall time on average, while "
        f"coupled Henyey equation assembly, block-tridiagonal linear solves, "
        f"mesh adjustment, and step bookkeeping account for the remaining "
        f"`{solver_tot_pct:.1f}%`. Notably, `MicrophysicsCache` (`#150`) hits "
        f"on `{min(hit_rates):.1f}%` to `{max(hit_rates):.1f}%` of full-star "
        f"`set_hydro_vars` sweeps (skipping `EOS`, `Opacity`, `Neu`, and "
        f"`Net + Jacobian`), whereas `Turbulence` (`MLT`/`TDC` plus "
        f"finite-difference face perturbations) executes on every sweep "
        f"because convective transport depends on neighbor-zone face "
        f"gradients, making `Turbulence` the largest single subsystem "
        f"(`{mean_sh['turbulence_share_pct']:.1f}%` mean share)."
    )
    lines.append("")

    # Section 3: Real MESA 26.4.1 vs FlexMESA strict
    lines.append(
        "## 3. Real Fortran MESA `26.4.1` vs FlexMESA (`strict`) at Matched "
        "Thread Count (`2` Threads)"
    )
    lines.append("")
    lines.append(
        "Each upstream case was compiled cleanly against `$MESA_DIR` "
        "(`r26.4.1`) and run via `./rn` with `OMP_NUM_THREADS=2` and "
        "`MESA_SKIP_OPTIONAL=t`. Both MESA's internal `runtime (minutes)` "
        "evolve timer (excluding one-time table startup) and total `./rn` "
        "wall time are reported alongside FlexMESA's post-warmup "
        "`evolve_wall_sec`:"
    )
    lines.append("")
    lines.append(
        "| Case | Engine (`2t`) | Steps | Retries | Mean Zones | "
        "Evolve Wall (`s`) | Wall / Step (`ms`) | Wall / Zone-Step (`us`) | "
        "Zone-Step Speedup (`MESA / FlexMESA`) |"
    )
    lines.append(
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    zs_speedups: list[float] = []
    wall_speedups: list[float] = []
    for cname in prov["selected_cases"]:
        sc = strict_by_case[cname]
        mc = mesa_by_case[cname]
        f_uzs = float(sc["wall_us_per_zone_step"])
        m_uzs = float(mc["evolve_wall_us_per_zone_step"])
        zs_sp = m_uzs / max(f_uzs, 1.0e-12)
        w_sp = float(mc["mesa_internal_evolve_sec"]) / max(
            float(sc["evolve_wall_sec"]), 1.0e-12
        )
        zs_speedups.append(zs_sp)
        wall_speedups.append(w_sp)
        lines.append(
            f"| `{cname}` | FlexMESA `strict` | "
            f"`{sc['steps_taken']}` | `{sc['total_retries']}` | "
            f"`{sc['mean_zones']:.0f}` | `{sc['evolve_wall_sec']:.3f}` | "
            f"`{sc['wall_ms_per_step']:.2f}` | `{f_uzs:.2f}` | "
            f"`{zs_sp:.2f}x` |"
        )
        lines.append(
            f"| `{cname}` | MESA `26.4.1` (evolve / `./rn`) | "
            f"`{mc['steps_taken']}` | `{mc['total_retries']}` | "
            f"`{mc['mean_zones']:.0f}` | "
            f"`{mc['mesa_internal_evolve_sec']:.2f}` "
            f"(`{mc['rn_wall_sec']:.2f}`) | "
            f"`{mc['evolve_wall_ms_per_step']:.2f}` | `{m_uzs:.2f}` | "
            f"`1.00x` |"
        )
    lines.append("")
    lines.append(
        f"Per zone-step (`wall_us_per_zone_step`), FlexMESA `strict` runs at "
        f"`{min(zs_speedups):.2f}x` to `{max(zs_speedups):.2f}x` the speed of "
        f"compiled Fortran MESA `26.4.1` (`{min(wall_speedups):.2f}x` to "
        f"`{max(wall_speedups):.2f}x` on end-to-end evolution wall clock). "
        f"Differences between per-zone-step speedup and total-run speedup stem "
        f"directly from timestep (`dt`) controller trajectories (`#149`), "
        f"because `1.3M_ms_high_Z` and `make_metals` currently take more "
        f"timesteps than upstream MESA while `low_z` and `high_z` take fewer."
    )
    lines.append("")

    # Section 4: #150 waste removal
    lines.append("## 4. What `#150` Waste Removal Bought")
    lines.append("")
    ab = cast(dict[str, Any], waste["git_ab_1_3M_ms_high_Z"])
    pre150 = cast(dict[str, Any], ab["pre_150"])
    post150 = cast(dict[str, Any], ab["post_150"])
    lines.append(
        "Issue `#150` removed two recurring sources of per-step waste in "
        "`flexmesa-star`: (1) redundant microphysics evaluations across "
        "unchanged `(lnRho, lnT, X)` zone states via `MicrophysicsCache`, and "
        "(2) unconditional full `StarModel::clone()` heap allocations prior to "
        "every timestep via the pre-sized `PrimaryStateBuffer` retry snapshot."
    )
    lines.append("")
    lines.append("### 4.1 Direct Pre-`#150` vs Post-`#150` Commit Benchmark")
    lines.append("")
    lines.append(
        "| Case | Build Commit | Steps | Median Wall (`s`) | "
        "Sample Range (`s`) | Wall / Step (`ms`) | Wall Saved (`%`) | "
        "Speedup (`pre / post`) |"
    )
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| `{ab['case']}` | Pre-`#150` (`{pre150['short_commit']}`) | "
        f"`{pre150['steps_taken']}` | `{pre150['median_wall_sec']:.4f}` | "
        f"`{pre150['range_wall_sec']:.4f}` | "
        f"`{pre150['wall_ms_per_step']:.2f}` | "
        f"`0.00%` | `1.00x` |"
    )
    lines.append(
        f"| `{ab['case']}` | Post-`#150` (`{post150['short_commit']}`) | "
        f"`{post150['steps_taken']}` | `{post150['median_wall_sec']:.4f}` | "
        f"`{post150['range_wall_sec']:.4f}` | "
        f"`{post150['wall_ms_per_step']:.2f}` | "
        f"`{ab['wall_time_saved_pct']:.2f}%` | "
        f"`{ab['wall_speedup_pre_over_post']:.2f}x` |"
    )
    lines.append("")

    lines.append("### 4.2 `HEAD` Component Breakdown Across All Four Cases")
    lines.append("")
    lines.append(
        "| Case | `MicrophysicsCache` Hits / Step | "
        "Est. Micro Wall Saved (`s`) | `StarModel` Heap (`KiB`) | "
        "`PrimaryStateBuffer` (`KiB`) | Snapshot Footprint Ratio | "
        "Pre-`#150` Clone Cost (`s`) vs Lazy Buffer (`s`) |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for wc in cast(list[dict[str, Any]], waste["head_component_breakdown"]):
        sm_kib = float(wc["star_model_heap_bytes"]) / 1024.0
        pb_kib = float(wc["primary_state_buffer_heap_bytes"]) / 1024.0
        lines.append(
            f"| `{wc['case']}` | "
            f"`{wc['micro_cache_hits_per_step']:.2f}` | "
            f"`{wc['saved_micro_eval_sec']:.4f}` | "
            f"`{sm_kib:.1f}` | `{pb_kib:.1f}` | "
            f"`{wc['snapshot_byte_reduction_factor']:.2f}x` | "
            f"`{wc['pre150_full_clone_every_step_sec']:.4f}` vs "
            f"`{wc['post150_lazy_primary_buf_sec']:.5f}` |"
        )
    lines.append("")

    # Section 5: strict vs fast (#165)
    lines.append(
        "## 5. What `fast` Math Buys Over `strict` (`#165`) & Trajectory Drift"
    )
    lines.append("")
    lines.append(
        "Per `DECISIONS #23` and `#165`, `NumericsMode::Fast` (platform `std` "
        "`f64` intrinsics and hardware `mul_add`) is compared **exclusively** "
        "against `NumericsMode::Strict` (`crlibm 1.0beta4` without FMA "
        "contraction) and is never compared against MESA:"
    )
    lines.append("")
    lines.append(
        "| Case | Steps (`strict` / `fast`) | Retries (`s` / `f`) | "
        "Wall (`strict` / `fast`, `s`) | `us / zone-step` (`s` / `f`) | "
        "Zone-Step Speedup (`s / f`) | `d_age / age` | "
        "Max Track `|d log L|` | Max Track `|d log Teff|` | "
        "Max Track `|d log rho_c|` |"
    )
    lines.append(
        "| :--- | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |"
    )
    svf_zs_sp: list[float] = []
    svf_w_sp: list[float] = []
    for cname in prov["selected_cases"]:
        r = svf_by_case[cname]
        td = cast(dict[str, Any], r["trajectory_diff"])
        tcols = cast(dict[str, Any], td["columns"])
        zs_sp = float(r["zone_step_speedup_strict_over_fast"])
        w_sp = float(r["wall_speedup_strict_over_fast"])
        svf_zs_sp.append(zs_sp)
        svf_w_sp.append(w_sp)
        lines.append(
            f"| `{cname}` | "
            f"`{r['strict_steps']}` / `{r['fast_steps']}` | "
            f"`{r['strict_retries']}` / `{r['fast_retries']}` | "
            f"`{r['strict_evolve_wall_sec']:.3f}` / "
            f"`{r['fast_evolve_wall_sec']:.3f}` | "
            f"`{r['strict_us_per_zone_step']:.2f}` / "
            f"`{r['fast_us_per_zone_step']:.2f}` | "
            f"`{zs_sp:.2f}x` | "
            f"`{float(td['rel_star_age_diff']):+.2e}` | "
            f"`{tcols['log_L']['track_max_abs_diff']:.2e}` | "
            f"`{tcols['log_Teff']['track_max_abs_diff']:.2e}` | "
            f"`{tcols['log_cntr_Rho']['track_max_abs_diff']:.2e}` |"
        )
    lines.append("")
    wall_red_lo = 100.0 * (1.0 - 1.0 / max(min(svf_w_sp), 1.0e-12))
    wall_red_hi = 100.0 * (1.0 - 1.0 / max(max(svf_w_sp), 1.0e-12))
    lines.append(
        f"Switching from `strict` (`crlibm`) to `fast` (`std` + hardware FMA) "
        f"yields `{min(svf_zs_sp):.2f}x` to `{max(svf_zs_sp):.2f}x` per "
        f"zone-step (`{min(svf_w_sp):.2f}x` to `{max(svf_w_sp):.2f}x` wall "
        f"speedup, reducing evolution runtime by `{wall_red_lo:.1f}%` to "
        f"`{wall_red_hi:.1f}%`) while keeping relative lifetime differences "
        f"within `|d_age / age| <= 6.8e-5`. The measured speedup reflects "
        f"`crlibm` transcendental calls (`exp`, `ln`, `log10`, `pow`) in "
        f"reaction screening, nuclear rates, and MLT/opacity log conversions "
        f"operating alongside FMA-insensitive spline lookups and "
        f"block-tridiagonal linear solves."
    )
    lines.append("")

    # Section 6: N-star worker pool batching + GPU grid estimate (#166)
    lines.append(
        "## 6. `N`-Star Worker-Pool Batching vs Stateless Accelerator "
        "Estimate (`#166` Input)"
    )
    lines.append("")
    lines.append(
        "### 6.1 Safe Rust `rayon::ThreadPool` Multi-Star Batch Scaling"
    )
    lines.append("")
    lines.append(
        "`profile_evolution batch` evolves `N` independent `low_z` models "
        "(`60` steps per track, heterogeneous `mixing_length_alpha` spanning "
        "`[1.98, 2.02]` so Newton iteration counts vary realistically across "
        "stars) on dedicated `2`-thread and `8`-thread `rayon` worker pools "
        "sharing immutable `OnceLock` EOS and opacity tables:"
    )
    lines.append("")
    lines.append(
        "| Pool Threads | Batch `N` Stars | Wall (`s`) | Total Steps | "
        "Throughput (`stars / min`) | `zone-steps / s` | "
        "Speedup vs `N = 1` | Peak RSS (`MiB`) |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for pool_key in ("pool_2_threads", "pool_8_threads"):
        p_obj = cast(dict[str, Any], batch[pool_key])
        for pt in cast(list[dict[str, Any]], p_obj["points"]):
            rss_mib = float(pt["peak_rss_kib"]) / 1024.0
            lines.append(
                f"| `{pt['pool_threads']}` | `{pt['n_stars']}` | "
                f"`{pt['wall_sec']:.3f}` | `{pt['total_steps']}` | "
                f"`{pt['stars_per_min']:.1f}` | "
                f"`{pt['zone_steps_per_sec']:.0f}` | "
                f"`{pt['speedup_vs_n1']:.2f}x` | `{rss_mib:.1f}` |"
            )
    pts_8t = cast(list[dict[str, Any]], batch["pool_8_threads"]["points"])
    best_8t = max(pts_8t, key=lambda p: float(p["stars_per_min"]))
    rss_32_mib = float(pts_8t[-1]["peak_rss_kib"]) / 1024.0
    gain_vs_2t = float(
        accel["rust_8t_worker_pool_throughput_gain_vs_single_star_2t"]
    )
    lines.append("")
    lines.append(
        f"At `N = 1`, an `8`-thread pool is slightly slower than a `2`-thread "
        f"pool because intra-star `rayon` loops partition `~785` zones into "
        f"sub-`100`-zone chunks. Once `N >= 8` independent tracks run "
        f"concurrently, `rayon` work-stealing assigns whole stars to threads "
        f"without per-zone synchronization, reaching "
        f"`{best_8t['stars_per_min']:.1f}` `stars / min` "
        f"(`{best_8t['zone_steps_per_sec']:.0f}` `zone-steps / s`, "
        f"`{best_8t['speedup_vs_n1']:.2f}x` vs `8t` `N = 1` and "
        f"`{gain_vs_2t:.2f}x` vs `2t` `N = 1`), while `32` concurrent models "
        f"hold peak RSS at `{rss_32_mib:.1f} MiB` via shared `OnceLock` EOS "
        f"and opacity tables."
    )
    lines.append("")

    lines.append(
        "### 6.2 Kernel Anchors (`benchmarks/raw/26.4.1/`) & Grid-Scale "
        "Accelerator Model"
    )
    lines.append("")
    lines.append(
        "Single-kernel timings from `benchmarks/raw/26.4.1/` anchor what an "
        "accelerator (`NVIDIA GeForce RTX 5080`, `jax.vmap` `float64`) "
        "achieves relative to `4`-thread Safe Rust CPU per zone:"
    )
    lines.append("")
    lines.append(
        "| Kernel (`benchmarks/raw/26.4.1/`) | Ref Zones | "
        "Safe Rust `4t` (`us / zone`) | JAX `vmap` (`us / zone`) | "
        "JAX / Rust `4t` Speedup | Execution Notes |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | :--- |")
    kernel_notes = {
        "eos_helm": (
            "GPU `@jax.jit` poly eval on pre-fetched cells "
            "(public API does `N1-25` host lookup)"
        ),
        "eos_skye": (
            "GPU `@jax.jit` 3rd-order AD on pre-fetched cells "
            "(public API does `N1-25` host lookup)"
        ),
        "eos_4d": (
            "GPU `@jax.jit` bicubic eval on pre-fetched cells "
            "(public API does `N1-25` host lookup)"
        ),
        "kap_fixed": (
            "`pure_callback` into `_rust.kap_get_results_zones_rust` (`N1-25`)"
        ),
        "rates_screen": "Pure `@jax.jit` analytical screening kernel",
        "net_eval_host_cb": "`pure_callback` into `_rust.net_eval` (`N1-24`)",
        "turb_mlt": "Pure `@jax.jit` local cubic/quartic MLT solver",
        "turb_tdc": "Pure `@jax.jit` unrolled Newton/dual TDC vs Rust IFT",
        "colors_sed": "Pure `@jax.jit` 3D SED + filter integration",
        "mtx_tridiag": (
            "`O(N log N)` GPU `vmap` PCR vs 4-system amortized "
            "`O(N)` CPU block-Thomas"
        ),
    }
    anchors = cast(
        dict[str, dict[str, float]], accel["measured_kernel_anchors"]
    )
    for kname, krec in anchors.items():
        lines.append(
            f"| `{kname}` | `{int(krec['n_zones_ref'])}` | "
            f"`{krec['rust_4t_us_per_zone']:.4f}` | "
            f"`{krec['jax_gpu_vmap_us_per_zone']:.4f}` | "
            f"`{krec['gpu_vs_rust_4t_speedup']:.2f}x` | "
            f"{kernel_notes[kname]} |"
        )
    lines.append("")
    amdahl_ceil = float(accel["amdahl_infinite_gpu_micro_speedup_ceiling"])
    lines.append(
        f"Combining the measured `{micro_tot_pct:.1f}%` stateless microphysics "
        f"share with Amdahl's law fixes an upper bound of `{amdahl_ceil:.2f}x` "
        f"for any hybrid CPU-Henyey + GPU-microphysics step, even before PCIe "
        f"transfers (`24 GB/s`), XLA launch latency (`80 us` per Newton "
        f"iteration), and heterogeneous adaptive-timestep lockstep padding "
        f"(`0.72` utilization). By contrast, standalone stateless grid "
        f"evaluations (`SED` photometry, screening, analytical EOS, MLT "
        f"surrogates) avoid the Henyey solver loop entirely:"
    )
    lines.append("")
    lines.append(
        "| Grid Stars `N` | Total Zones | `8t` Rust Pool Step (`s`) | "
        "Hybrid GPU-Micro Step (`s`) | Hybrid / `8t` Rust Speedup | "
        "Stateless-Only `8t` Rust (`s`) | Stateless-Only GPU (`s`) | "
        "Stateless GPU Speedup |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for gp in cast(list[dict[str, Any]], accel["grid_projections"]):
        lines.append(
            f"| `{gp['n_stars']}` | `{gp['total_grid_zones']}` | "
            f"`{gp['rust_8t_worker_pool_step_sec']:.4f}` | "
            f"`{gp['hybrid_gpu_micro_step_sec']:.4f}` | "
            f"`{gp['hybrid_gpu_vs_rust_8t_ratio']:.2f}x` | "
            f"`{gp['stateless_batch_only_rust_8t_sec']:.4f}` | "
            f"`{gp['stateless_batch_only_gpu_sec']:.5f}` | "
            f"`{gp['stateless_batch_only_gpu_speedup_vs_rust_8t']:.2f}x` |"
        )
    lines.append("")
    lines.append("### 6.3 Concrete Recommendations for `#166`")
    lines.append("")
    lines.append(
        "1. **Population & Grid Stellar Evolution (`N >= 8` tracks)**: Run "
        "independent `StarModel` trajectories across a Safe Rust `rayon` "
        "worker pool rather than padding heterogeneous meshes (`nz`) and "
        "adaptive timesteps (`dt`) into lockstep GPU batches. Shared "
        "`OnceLock` physics tables keep memory overhead small across dozens "
        "of concurrent stars, and each worker runs asynchronously with zero "
        "host-device synchronization."
    )
    lines.append(
        "2. **Differentiable Stellar Structure & Evolution (Post-`#143`)**: "
        "Keep forward evolution in Safe Rust and expose Implicit Function "
        "Theorem (`@jax.custom_vjp` / `@jax.custom_jvp`) step wrappers solved "
        'on CPU (`bcyclic` via `mtx` `solver_mode="rust_ffi"` or '
        "`fast_block_thomas_solve`) plus `#150` `.photo` binomial "
        "checkpointing. At `N = 4096`, GPU `PCR` (`7.45 ms` single-profile, "
        "`2.80 ms` `vmap` amortized) is slower than CPU solvers (`1.38 ms` "
        "1-thread `PyO3` `bcyclic`, `1.29 ms` 1-thread Rust block-Thomas, "
        "`0.32 ms` 4-system amortized Rust block-Thomas) because `PCR` "
        "performs `O(N log2 N)` dense block multiplications in `float64`."
    )
    lines.append(
        "3. **Where Native Pure-XLA JAX Kernels Pay (`#166`)**: Prioritize "
        "pure `@jax.jit` / `jax.vmap` kernels on stateless, fixed-shape batch "
        "workloads that run outside the Henyey iteration loop, specifically "
        "bolometric and filter photometry (`flexmesa.colors`), analytical "
        "screening and neutrino emission (`flexmesa.rates`, `flexmesa.neu`), "
        "local `set_mlt_jax` / `T(tau)` boundary evaluations (`flexmesa.turb`, "
        "`flexmesa.atm`), and batch inference likelihoods over `>= 10^4` "
        "evaluation points. Porting stateful 4D `eos` / `kap` bicubic tables "
        "or stiff `net` Rosenbrock solvers to pure GPU solely to accelerate "
        "1D Henyey steps is bounded by the `<= "
        f"{amdahl_ceil:.2f}x` Amdahl ceiling."
    )
    lines.append("")

    # Section 7: Plain Disclosure (What Was Not Verified)
    lines.append("## 7. What Was Not Verified")
    lines.append("")
    lines.append(
        "- **Case breadth**: Measurements cover four `star/test_suite` cases "
        "(`1.3M_ms_high_Z`, `low_z`, `high_z`, `make_metals`) on a single "
        "`x86_64` workstation; `semiconvection` (blocked at step 197 on "
        "`origin/main` pending `#149`), late burning with `approx21.net`, "
        "hydrodynamic `rsp2` pulsations, `binary` evolutions, and `astero` "
        "simplex runs were not profiled in this campaign."
    )
    lines.append(
        "- **`aarch64` & multi-socket scaling**: `strict` vs `fast` ratios and "
        "`rayon` worker-pool curves were measured on one `8`-core / "
        "`16`-thread `x86_64` CPU under WSL2; Apple Silicon (`aarch64`) and "
        "high-core-count server NUMA scaling were not measured."
    )
    lines.append(
        "- **`#150` git A/B scope**: Direct pre-`#150` (`af9747e`) vs "
        "post-`#150` (`3f7fa734`) binary timing was run on `1.3M_ms_high_Z`; "
        "the remaining three cases report `HEAD` component measurements "
        "(`MicrophysicsCache` hit rates and `PrimaryStateBuffer` vs "
        "`StarModel::clone` timings)."
    )
    lines.append(
        "- **Grid-scale GPU evolution**: Because `src/flexmesa/star/` JAX "
        "helpers are frozen (`JAX_ENGINE_STATUS = "
        '"frozen-unverified-until-#143"`), the `N = 1..1024` hybrid GPU '
        "numbers in Section 6.2 come from an analytical Amdahl + PCIe + "
        "lockstep model anchored to `benchmarks/raw/26.4.1/` single-kernel "
        "measurements rather than an executed GPU Henyey evolution engine."
    )

    if for_docs:
        lines.append("")
        lines.append("## 8. Inspecting the Raw Artifacts from Python")
        lines.append("")
        lines.append("```python")
        lines.append("import csv")
        lines.append("import json")
        lines.append("import pathlib")
        lines.append("import flexmesa")
        lines.append("")
        lines.append(
            "repo_root = pathlib.Path(flexmesa.__file__).resolve().parents[2]"
        )
        lines.append(
            "prof_dir = repo_root / 'benchmarks' / 'evolution_profile' / "
            "'26.4.1'"
        )
        lines.append(
            "prov = json.loads("
            "(prof_dir / 'provenance.json').read_text(encoding='utf-8')"
            ")"
        )
        lines.append(
            "with (prof_dir / 'phase_breakdown.csv').open("
            "encoding='utf-8'"
            ") as fh:"
        )
        lines.append(
            "    rows = list("
            "csv.DictReader(ln for ln in fh if not ln.startswith('#'))"
            ")"
        )
        lines.append("assert len(rows) == 4")
        lines.append("for row in rows:")
        lines.append("    total_pct = sum(")
        lines.append("        float(row[col])")
        lines.append("        for col in (")
        lines.append("            'eos_share_pct',")
        lines.append("            'kap_share_pct',")
        lines.append("            'net_and_jacobian_share_pct',")
        lines.append("            'turbulence_share_pct',")
        lines.append("            'equation_assembly_share_pct',")
        lines.append("            'block_solve_share_pct',")
        lines.append("            'remesh_share_pct',")
        lines.append("            'bookkeeping_and_copies_share_pct',")
        lines.append("        )")
        lines.append("    )")
        lines.append("    assert abs(total_pct - 100.0) < 0.05")
        lines.append("mesa_rel = prov['toolchains']['mesa_release']")
        lines.append(
            'print(f"Profiled {len(rows)} cases against MESA {mesa_rel}")'
        )
        lines.append("```")

    return lines


def render_reports(in_dir: pathlib.Path = _DEFAULT_IN_DIR) -> tuple[str, str]:
    """Return `(audit_markdown, docs_markdown)` rendered from ``in_dir``."""
    audit_md = "\n".join(_build_shared_sections(in_dir, for_docs=False)) + "\n"
    docs_md = "\n".join(_build_shared_sections(in_dir, for_docs=True)) + "\n"
    helpers.assert_clean_public_text(audit_md, "evolution_profile_26_4_1.md")
    helpers.assert_clean_public_text(docs_md, "evolution-profiling.md")
    return audit_md, docs_md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=pathlib.Path, default=_DEFAULT_IN_DIR)
    parser.add_argument(
        "--audit-out", type=pathlib.Path, default=_DEFAULT_AUDIT_OUT
    )
    parser.add_argument(
        "--docs-out", type=pathlib.Path, default=_DEFAULT_DOCS_OUT
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    audit_md, docs_md = render_reports(args.in_dir)
    if args.check:
        if not args.audit_out.is_file() or not args.docs_out.is_file():
            sys.stderr.write("Missing rendered evolution profile markdown.\n")
            return 1
        cur_audit = args.audit_out.read_text(encoding="utf-8")
        cur_docs = args.docs_out.read_text(encoding="utf-8")
        if cur_audit != audit_md or cur_docs != docs_md:
            sys.stderr.write(
                "Rendered evolution profile markdown is out of date; re-run "
                "scripts/audit/render_evolution_profile_report.py.\n"
            )
            return 1
        return 0

    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(audit_md, encoding="utf-8")
    args.docs_out.write_text(docs_md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
