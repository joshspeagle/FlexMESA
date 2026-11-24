#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
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
    "semiconvection": (
        "`pp_and_cno_extras.net` (25 species), Henyey MLT, Type-2 `a09_co` "
        "opacity, Ledoux criterion with Langer (1985) semiconvection and "
        "predictive mixing (`1.5 Msun`, `Z = 0.02`) to `center_h1 <= 0.38`"
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
    tp = cast(dict[str, Any], prov["timing_protocol"])
    bf = cast(dict[str, Any], prov["build_flags"])
    main_short = str(prov["flexmesa_origin_main_commit"])[:8]
    branch_short = str(prov["flexmesa_branch_commit"])[:8]
    repeats = int(tp["matched_repeats_per_engine"])

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
        "CSV captures in `benchmarks/evolution_profile/26.4.1/`. Section 3 "
        "compares with compiled Fortran MESA `r26.4.1` (`commit c8c2026`, "
        "Zenodo `19722306`) under a matched timing protocol; Sections 2, 4, "
        "5, and 6 compare FlexMESA with itself."
    )
    lines.append("")
    n_diag = sum(
        1 for c in strict_cases if bool(c["allow_unsupported_controls"])
    )
    mode_summary = ", ".join(
        f"`{c['case']}`: `{c['run_mode']}`" for c in strict_cases
    )
    step_matched_names = [
        str(cname)
        for cname in prov["selected_cases"]
        if int(strict_by_case[cname]["steps_taken"])
        == int(mesa_by_case[cname]["steps_taken"])
    ]
    if len(step_matched_names) == 2:
        step_matched_txt = (
            f"`{step_matched_names[0]}` and `{step_matched_names[1]}`"
        )
    elif step_matched_names:
        step_matched_txt = ", ".join(f"`{c}`" for c in step_matched_names)
    else:
        step_matched_txt = "none"
    lines.append(
        f"**Every FlexMESA run in this report is a diagnostic run.** "
        f"`{n_diag}` of the `{len(strict_cases)}` `strict` records carry "
        f"`allow_unsupported_controls = true` ({mode_summary}), so the runs "
        f"continued past upstream controls FlexMESA does not implement and "
        f"did not execute every control requested by the inlists "
        f"(`DECISIONS #26`). All four cases are declared known gaps in "
        f"`benchmarks/history_harness/known_gaps.toml` and none counts as "
        f"passing under `#143`. Although {step_matched_txt} match MESA's step "
        f"counts identically at `{main_short}`, track differences in Section 1 "
        f"remain non-zero; these timings profile FlexMESA at `{main_short}` "
        f"rather than a verified `#143` parity run."
    )
    lines.append("")

    # Provenance block
    lines.append("## 1. Measurement Provenance & Selected Cases")
    lines.append("")
    lines.append(
        f"- **Capture Timestamp (UTC)**: `{prov['date_utc']}` "
        f"(1-minute host load average `{qh['load_avg_1m_start']:.2f}` "
        f"at start; `{qh['load_avg_1m_end']:.2f}` at completion immediately "
        f"after the `8`-thread `N = 32` worker-pool burst; serialized with "
        f"an exclusive host timing lock)"
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
        f"MESA SDK `{tc['mesa_sdk']}`, `{tc['gfortran']}`, `CRMATH`, "
        f"flags `{bf['mesa_fortran_flags']}`"
    )
    lines.append(
        f"- **FlexMESA Toolchain**: `{tc['rustc']}`, `{tc['python']}`, "
        f"`JAX {tc['jax']}`, `origin/main` `{main_short}` (driver commit "
        f"`{branch_short}`), profile `{bf['flexmesa_cargo_profile']}`"
    )
    lines.append(
        f"- **Matched Timing Protocol**: `{repeats}` repeats per case on "
        f"both MESA `26.4.1` and FlexMESA (`strict` and `fast`), reporting "
        f"median and `[min, max]` spread, with normal history output written "
        f"on every timed repeat and matched thread caps "
        f"`OMP_NUM_THREADS={tc['omp_num_threads']}`, "
        f"`RAYON_NUM_THREADS={tc['rayon_num_threads']}`, "
        f"`OPENBLAS_NUM_THREADS={tc['omp_num_threads']}`, "
        f"`MKL_NUM_THREADS={tc['omp_num_threads']}`"
    )
    lines.append("")
    lines.append(
        f"Four `star/test_suite` cases whose diagnostic runs reach upstream's "
        f"termination code on `origin/main` (`{main_short}`), with stock "
        f"`run_star_extras.f90` evolutionary loops, were selected to span "
        f"fixed and Type-2 CO opacities, 8-species and 25-species nuclear "
        f"networks, Cox and Henyey MLT, Ledoux semiconvection with predictive "
        f"mixing, and zero-metallicity pre-MS-through-core-He evolution "
        f"(including the two exact step-matching cases {step_matched_txt}):"
    )
    lines.append("")
    for cname in prov["selected_cases"]:
        lines.append(f"- **`{cname}`**: {_CASE_PHYSICS_NOTES[cname]}.")
    lines.append("")
    lines.append(
        "| Case | FlexMESA Run Mode | Termination (`strict` / MESA) | "
        "Steps (`strict` / MESA) | Zones (`init -> final`) | "
        "Final Age (`strict` / MESA, yr) | `d_age / age` | "
        "Max Track `|d log L|` | Max Track `|d log Teff|` | "
        "Max Track `|d log rho_c|` |"
    )
    lines.append(
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: |"
    )
    for cname in prov["selected_cases"]:
        sc = strict_by_case[cname]
        mc = mesa_by_case[cname]
        cmp = cast(dict[str, Any], mc["strict_vs_mesa_comparison"])
        cols = cast(dict[str, Any], cmp["columns"])
        rel_age_pct = 100.0 * float(cmp["rel_star_age_diff"])
        lines.append(
            f"| `{cname}` | `{sc['run_mode']}` | "
            f"`{sc['termination_code']}` / `{mc['termination_code']}` | "
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
        f"Unperturbed wall times (`repeats = {repeats}` median, with full "
        f"`history.data` written on every repeat) are measured without "
        f"instrumentation hooks inside `flexmesa-star`. A second pass "
        f"(`profile_evolution case --with-phase-profile`) times read-only step "
        f"probes cloned before each step while verifying a `0-ULP` match on "
        f"the serialized `.photo` state checksum so phase attribution never "
        f"perturbs the canonical evolution trajectory."
    )
    lines.append("")
    lines.append(
        "The shares below are a rescaled bottom-up model, not a direct "
        "instrumented wall-clock measurement. Each kernel is timed once per "
        "step on a copy of the pre-step model and multiplied by the call "
        "counts recorded during the real step. The sum of those products "
        "exceeds the measured evolve wall time in every case, and the shares "
        "are then rescaled proportionally to sum to `100%`:"
    )
    lines.append("")
    lines.append(
        "| Case | Run Mode | Measured Evolve Wall (`s`) | "
        "Sum of Kernel Estimates (`s`) | Overshoot |"
    )
    lines.append("| :--- | :--- | ---: | ---: | ---: |")
    for cname in prov["selected_cases"]:
        sc = strict_by_case[cname]
        ph = cast(dict[str, Any], sc["phases"])
        ref_sec = float(ph["reference_evolve_wall_sec"])
        bottom_up = float(ph["bottom_up_total_sec"])
        over_pct = 100.0 * (bottom_up / ref_sec - 1.0)
        lines.append(
            f"| `{cname}` | `{sc['run_mode']}` | `{ref_sec:.2f}` | "
            f"`{bottom_up:.2f}` | `{_fmt_pct(over_pct, 1, signed=True)}` |"
        )
    lines.append("")
    lines.append(
        "The turbulence bucket also includes the Brunt, surface boundary, and "
        "`eps_grav` probes, which the real step does not call on every sweep. "
        "Figures derived from these shares later in the report (the stateless "
        "microphysics share and the Amdahl ceiling in Section 6) inherit this "
        "uncertainty."
    )
    lines.append("")
    lines.append(
        "### 2.1 Modelled Share of Step Wall Time by Subsystem (`%`, "
        "rescaled to `100%`)"
    )
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
        f"In the rescaled bottom-up model across these four evolutions, local "
        f"zone microphysics (`EOS`, `Opacity`, `Net + Jacobian`, and "
        f"`Turbulence`) accounts for `{micro_tot_pct:.1f}%` of per-step wall "
        f"time on average, while coupled Henyey equation assembly, "
        f"block-tridiagonal linear solves, mesh adjustment, and step "
        f"bookkeeping account for `{solver_tot_pct:.1f}%`. Directly counted "
        f"`MicrophysicsCache` (`#150`) hits occur on `{min(hit_rates):.1f}%` "
        f"to `{max(hit_rates):.1f}%` of full-star `set_hydro_vars` sweeps "
        f"(skipping `EOS`, `Opacity`, `Neu`, and `Net + Jacobian`), whereas "
        f"`Turbulence` (`MLT`/`TDC` plus face perturbations) executes on every "
        f"sweep because convective transport depends on neighbor-zone face "
        f"gradients (`{mean_sh['turbulence_share_pct']:.1f}%` modelled mean "
        f"share)."
    )
    lines.append("")

    # Section 3: Matched-protocol real MESA 26.4.1 vs FlexMESA strict
    lines.append(
        "## 3. Matched-Protocol Comparison: Real Fortran MESA `26.4.1` vs "
        f"FlexMESA (`strict`) at `{tc['omp_num_threads']}` Threads"
    )
    lines.append("")
    lines.append(
        f"Each upstream case was compiled once against `$MESA_DIR` (`r26.4.1`) "
        f"and timed over `{repeats}` clean `./rn` repeats with "
        f"`OMP_NUM_THREADS={tc['omp_num_threads']}` and `MESA_SKIP_OPTIONAL=t` "
        f"under the same exclusive host lock as FlexMESA (`{repeats}` repeats "
        f"at `RAYON_NUM_THREADS={tc['rayon_num_threads']}`, writing full "
        f"`history.data` and `final.mod` on every repeat). Exact cumulative "
        f"Newton solver iterations (including retries) are read from MESA's "
        f"`testhub.yml` (`s% total_num_solver_iterations`) and FlexMESA's "
        f"`model.total_num_solver_iterations`. For MESA, both internal "
        f"`runtime (minutes)` (excluding table startup, quantized to `0.6 s`) "
        f"and full `./rn` wall time are recorded:"
    )
    lines.append("")
    lines.append(
        "| Case | Engine & Run Mode (`2t`) | Steps | Retries | "
        "Newton Iters | Mean Zones | Evolve Wall Median `[min, max]` (`s`) | "
        "`ms / step` | `us / zone-step` | `ms / Newton-iter` | "
        "`MESA / FlexMESA` (`us/zs` ; `ms/iter`) |"
    )
    lines.append(
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |"
    )
    matched_zs_ratios: list[float] = []
    matched_nit_ratios: list[float] = []
    divergent_zs_ratios: list[float] = []
    divergent_nit_ratios: list[float] = []
    for cname in prov["selected_cases"]:
        sc = strict_by_case[cname]
        mc = mesa_by_case[cname]
        f_uzs = float(sc["wall_us_per_zone_step"])
        m_uzs = float(mc["evolve_wall_us_per_zone_step"])
        f_mnit = float(sc["wall_ms_per_newton_iter"])
        m_mnit = float(mc["evolve_wall_ms_per_newton_iter"])
        zs_sp = m_uzs / max(f_uzs, 1.0e-12)
        nit_sp = m_mnit / max(f_mnit, 1.0e-12)
        if int(sc["steps_taken"]) == int(mc["steps_taken"]):
            matched_zs_ratios.append(zs_sp)
            matched_nit_ratios.append(nit_sp)
        else:
            divergent_zs_ratios.append(zs_sp)
            divergent_nit_ratios.append(nit_sp)
        lines.append(
            f"| `{cname}` | FlexMESA `strict`, `{sc['run_mode']}` | "
            f"`{sc['steps_taken']}` | `{sc['total_retries']}` | "
            f"`{sc['total_newton_iterations']}` | `{sc['mean_zones']:.0f}` | "
            f"`{sc['evolve_wall_sec']:.3f}` "
            f"`[{sc['evolve_wall_sec_min']:.3f}, "
            f"{sc['evolve_wall_sec_max']:.3f}]` | "
            f"`{sc['wall_ms_per_step']:.2f}` | `{f_uzs:.2f}` | "
            f"`{f_mnit:.2f}` | `{zs_sp:.2f}x` ; `{nit_sp:.2f}x` |"
        )
        lines.append(
            f"| `{cname}` | MESA `26.4.1` (evolve; `./rn` in note) | "
            f"`{mc['steps_taken']}` | `{mc['total_retries']}` | "
            f"`{mc['total_newton_iterations']}` | `{mc['mean_zones']:.0f}` | "
            f"`{mc['mesa_internal_evolve_sec']:.2f}` "
            f"`[{mc['mesa_internal_evolve_sec_min']:.2f}, "
            f"{mc['mesa_internal_evolve_sec_max']:.2f}]` | "
            f"`{mc['evolve_wall_ms_per_step']:.2f}` | `{m_uzs:.2f}` | "
            f"`{m_mnit:.2f}` | `1.00x` ; `1.00x` |"
        )
    lines.append("")
    rn_notes = "; ".join(
        f"`{c}`: `{mesa_by_case[c]['rn_wall_sec']:.2f} s` "
        f"`[{mesa_by_case[c]['rn_wall_sec_min']:.2f}, "
        f"{mesa_by_case[c]['rn_wall_sec_max']:.2f}]` "
        f"(`{mesa_by_case[c]['rn_wall_ms_per_newton_iter']:.2f} ms/iter`)"
        for c in prov["selected_cases"]
    )
    lines.append(
        f"*MESA full `./rn` wall medians (`{repeats}` repeats): {rn_notes}.*"
    )
    lines.append("")
    if matched_zs_ratios:
        lines.append(
            f"On the two step-matched cases ({step_matched_txt}), wall time "
            f"per zone-step for the FlexMESA `strict` diagnostic runs is "
            f"`{min(matched_zs_ratios):.2f}x` to "
            f"`{max(matched_zs_ratios):.2f}x` smaller than MESA `26.4.1` "
            f"(`{min(matched_nit_ratios):.2f}x` to "
            f"`{max(matched_nit_ratios):.2f}x` per Newton iteration). On the "
            f"two cases whose step counts still diverge from MESA, the "
            f"per-zone-step ratios are `{min(divergent_zs_ratios):.2f}x` to "
            f"`{max(divergent_zs_ratios):.2f}x` "
            f"(`{min(divergent_nit_ratios):.2f}x` to "
            f"`{max(divergent_nit_ratios):.2f}x` per Newton iteration). "
            f"Normalizing by Newton iterations isolates per-iteration cost "
            f"from step-count and retry differences, but even the step-matched "
            f"runs remain diagnostic (`allow_unsupported_controls = true`) "
            f"with non-zero track differences (Section 1) and different "
            f"default build optimization levels (`gfortran -O2` vs Cargo "
            f"`opt-level=3`, `lto=fat`)."
        )
    else:
        lines.append(
            "All four cases take different step counts and tracks from MESA "
            "(Section 1), and every FlexMESA run is diagnostic "
            "(`allow_unsupported_controls = true`)."
        )
    lines.append("")

    # Section 4: #150 waste removal
    lines.append("## 4. What `#150` Waste Removal Bought")
    lines.append("")
    ab = cast(dict[str, Any], waste["git_ab_1_3M_ms_high_Z"])
    pre150 = cast(dict[str, Any], ab["pre_150"])
    post150 = cast(dict[str, Any], ab["post_150"])
    comp_commit = str(
        waste.get(
            "head_component_measured_commit",
            prov["flexmesa_origin_main_commit"],
        )
    )[:8]
    lines.append(
        "Issue `#150` removed two recurring sources of per-step waste in "
        "`flexmesa-star`: (1) redundant microphysics evaluations across "
        "unchanged `(lnRho, lnT, X)` zone states via `MicrophysicsCache`, and "
        "(2) unconditional full `StarModel::clone()` heap allocations prior to "
        "every timestep via the pre-sized `PrimaryStateBuffer` retry snapshot."
    )
    lines.append("")
    lines.append("### 4.1 Historical Pre-`#150` vs Post-`#150` Git A/B Record")
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
    lines.append(
        f"This git A/B record covers one case (`{ab['case']}`, "
        f"`{pre150['repeats']}` repeats per build) measured directly across "
        f"commits `{pre150['short_commit']}` and `{post150['short_commit']}`. "
        f"The spread of the pre-`#150` samples "
        f"(`{pre150['range_wall_sec']:.2f} s`) is about as large as the "
        f"measured saving (`{ab['wall_time_saved_sec']:.2f} s`), so the "
        f"wall-clock saving is not well resolved. The two historical commits "
        f"also differ by more than the two items described above."
    )
    lines.append("")
    lines.append(
        f"### 4.2 `{comp_commit}` Cache & Memory Measurements vs Probe "
        f"Estimates"
    )
    lines.append("")
    lines.append(
        f"In the table below, the cache counters (`Hits / Step`, `Hit Rate`) "
        f"and heap footprint columns (`StarModel` vs `PrimaryStateBuffer` "
        f"bytes and ratio) are **direct measurements on `{comp_commit}`**. "
        f"The last two columns (`Est. Micro Wall Saved` and `Est. Clone vs "
        f"Lazy Buffer`) are **un-rescaled probe estimates**, not A/B "
        f"measurements: `Est. Micro Wall Saved` is cache hits multiplied by "
        f"the difference between a cold and a warm kernel probe from Section "
        f"2, and the clone column times `StarModel::clone()` at "
        f"`{comp_commit}` rather than on pre-`#150` code."
    )
    lines.append("")
    lines.append(
        "| Case | Run Mode | `MicrophysicsCache` Hits / Step (Hit Rate) | "
        "`StarModel` Heap (`KiB`) | `PrimaryStateBuffer` (`KiB`) | "
        "Heap Reduction | Est. Micro Wall Saved (`s`, probe) | "
        "Est. Clone vs Lazy Buffer (`s`, probe) |"
    )
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for wc in cast(list[dict[str, Any]], waste["head_component_breakdown"]):
        sm_kib = float(wc["star_model_heap_bytes"]) / 1024.0
        pb_kib = float(wc["primary_state_buffer_heap_bytes"]) / 1024.0
        hit_pct = float(wc["micro_cache_hit_rate_pct"])
        lines.append(
            f"| `{wc['case']}` | `{wc['run_mode']}` | "
            f"`{wc['micro_cache_hits_per_step']:.2f}` (`{hit_pct:.1f}%`) | "
            f"`{sm_kib:.1f}` | `{pb_kib:.1f}` | "
            f"`{wc['snapshot_byte_reduction_factor']:.2f}x` | "
            f"`{wc['saved_micro_eval_sec']:.4f}` | "
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
        f"Per `DECISIONS #23` and `#165`, `NumericsMode::Fast` (platform `std` "
        f"`f64` intrinsics and hardware `mul_add`) is compared **exclusively** "
        f"against `NumericsMode::Strict` (`crlibm 1.0beta4` without FMA "
        f"contraction) over `{repeats}` matched repeats per mode and is never "
        f"compared against MESA:"
    )
    lines.append("")
    lines.append(
        "| Case | Run Mode | Steps (`s` / `f`) | Newton Iters (`s` / `f`) | "
        "Wall Median `[min, max]` (`s` ; `f`, `s`) | "
        "`us / zone-step` (`s` / `f`) | `ms / Newton-iter` (`s` / `f`) | "
        "Speedup (`us/zs` ; `ms/iter`) | `d_age / age` | "
        "Max Track `|d log L|` | Max Track `|d log Teff|` | "
        "Max Track `|d log rho_c|` |"
    )
    lines.append(
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |"
    )
    svf_zs_sp: list[float] = []
    svf_nit_sp: list[float] = []
    svf_w_sp: list[float] = []
    svf_abs_age_diffs: list[float] = []
    for cname in prov["selected_cases"]:
        r = svf_by_case[cname]
        td = cast(dict[str, Any], r["trajectory_diff"])
        tcols = cast(dict[str, Any], td["columns"])
        zs_sp = float(r["zone_step_speedup_strict_over_fast"])
        nit_sp = float(r["newton_iter_speedup_strict_over_fast"])
        w_sp = float(r["wall_speedup_strict_over_fast"])
        age_d = float(td["rel_star_age_diff"])
        svf_zs_sp.append(zs_sp)
        svf_nit_sp.append(nit_sp)
        svf_w_sp.append(w_sp)
        svf_abs_age_diffs.append(abs(age_d))
        lines.append(
            f"| `{cname}` | `{r['run_mode']}` | "
            f"`{r['strict_steps']}` / `{r['fast_steps']}` | "
            f"`{r['strict_total_newton_iterations']}` / "
            f"`{r['fast_total_newton_iterations']}` | "
            f"`{r['strict_evolve_wall_sec']:.3f}` "
            f"`[{r['strict_evolve_wall_sec_min']:.3f}, "
            f"{r['strict_evolve_wall_sec_max']:.3f}]` ; "
            f"`{r['fast_evolve_wall_sec']:.3f}` "
            f"`[{r['fast_evolve_wall_sec_min']:.3f}, "
            f"{r['fast_evolve_wall_sec_max']:.3f}]` | "
            f"`{r['strict_us_per_zone_step']:.2f}` / "
            f"`{r['fast_us_per_zone_step']:.2f}` | "
            f"`{r['strict_ms_per_newton_iter']:.2f}` / "
            f"`{r['fast_ms_per_newton_iter']:.2f}` | "
            f"`{zs_sp:.2f}x` ; `{nit_sp:.2f}x` | "
            f"`{age_d:+.2e}` | "
            f"`{tcols['log_L']['track_max_abs_diff']:.2e}` | "
            f"`{tcols['log_Teff']['track_max_abs_diff']:.2e}` | "
            f"`{tcols['log_cntr_Rho']['track_max_abs_diff']:.2e}` |"
        )
    lines.append("")
    wall_red_lo = 100.0 * (1.0 - 1.0 / max(min(svf_w_sp), 1.0e-12))
    wall_red_hi = 100.0 * (1.0 - 1.0 / max(max(svf_w_sp), 1.0e-12))
    max_age_diff = max(svf_abs_age_diffs)
    lines.append(
        f"Switching from `strict` (`crlibm`) to `fast` (`std` + hardware FMA) "
        f"on `{main_short}` yields `{min(svf_zs_sp):.2f}x` to "
        f"`{max(svf_zs_sp):.2f}x` per zone-step (`{min(svf_nit_sp):.2f}x` to "
        f"`{max(svf_nit_sp):.2f}x` per Newton iteration, and "
        f"`{min(svf_w_sp):.2f}x` to `{max(svf_w_sp):.2f}x` wall speedup, "
        f"reducing evolution runtime by `{wall_red_lo:.1f}%` to "
        f"`{wall_red_hi:.1f}%`) while keeping relative lifetime differences "
        f"within `|d_age / age| <= {max_age_diff:.2e}`. The measured speedup "
        f"reflects `crlibm` transcendental calls (`exp`, `ln`, `log10`, `pow`) "
        f"in reaction screening, nuclear rates, and MLT/opacity log "
        f"conversions operating alongside FMA-insensitive spline lookups and "
        f"block-tridiagonal linear solves."
    )
    lines.append("")

    # Section 6: N-star worker pool batching + GPU grid estimate (#166)
    lines.append(
        "## 6. `N`-Star Worker-Pool Batching (`smoke: 60 steps`) vs Stateless "
        "Accelerator Model (`#166` Input)"
    )
    lines.append("")
    lines.append(
        "### 6.1 Safe Rust `rayon::ThreadPool` Multi-Star Batch Scaling "
        "(`smoke: 60 steps`)"
    )
    lines.append("")
    b2 = cast(dict[str, Any], batch["pool_2_threads"])
    b8 = cast(dict[str, Any], batch["pool_8_threads"])
    b_cap = int(b2["max_steps_cap"])
    b_zones = int(b2["template_zones"])
    b_mode = str(b2["run_mode"])
    lines.append(
        f"This experiment is a **`smoke: {b_cap} steps`** scaling benchmark "
        f"(capped at `{b_cap}` steps per star rather than uncapped evolution "
        f"to physical termination). `profile_evolution batch` evolves `N` "
        f"independent `low_z` diagnostic models (`{b_mode}`, `{b_zones}` "
        f"initial zones, heterogeneous `mixing_length_alpha` spanning "
        f"`[1.98, 2.02]` so Newton iteration counts vary across stars) on "
        f"dedicated `2`-thread and `8`-thread `rayon` worker pools sharing "
        f"immutable `OnceLock` EOS and opacity tables:"
    )
    lines.append("")
    lines.append(
        "| Pool Threads | Batch `N` Stars (`smoke: 60 steps`) | Wall (`s`) | "
        "Total Steps | Throughput (`stars / min`) | `zone-steps / s` | "
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
    pts_2t = cast(list[dict[str, Any]], b2["points"])
    pts_8t = cast(list[dict[str, Any]], b8["points"])
    best_8t = max(pts_8t, key=lambda p: float(p["stars_per_min"]))
    rss_32_mib = float(pts_8t[-1]["peak_rss_kib"]) / 1024.0
    gain_vs_2t = float(
        accel["rust_8t_worker_pool_throughput_gain_vs_single_star_2t"]
    )
    n1_2t_wall = float(pts_2t[0]["wall_sec"])
    n1_8t_wall = float(pts_8t[0]["wall_sec"])
    n1_cmp = (
        "slower than"
        if n1_8t_wall > n1_2t_wall
        else "faster than"
        if n1_8t_wall < n1_2t_wall
        else "equal to"
    )
    lines.append("")
    lines.append(
        f"In this `smoke: {b_cap} steps` scaling benchmark, at `N = 1` an "
        f"`8`-thread pool (`{n1_8t_wall:.3f} s`) is {n1_cmp} a `2`-thread pool "
        f"(`{n1_2t_wall:.3f} s`) because intra-star `rayon` loops partition "
        f"`~{b_zones}` zones into sub-`100`-zone chunks. Once `N >= 8` "
        f"independent tracks run concurrently, `rayon` work-stealing assigns "
        f"whole stars to threads without per-zone synchronization, reaching "
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
    assump = cast(dict[str, Any], accel["model_assumptions"])
    dev_sp = float(assump["hypothetical_device_micro_speedup_vs_2t_rust"])
    pcie_bw = float(assump["pcie_effective_bandwidth_gb_s"])
    xla_us = float(assump["xla_dispatch_latency_us_per_iter"])
    lock_eff = float(assump["heterogeneous_grid_lockstep_efficiency"])
    amdahl_ceil = float(accel["amdahl_infinite_gpu_micro_speedup_ceiling"])
    lines.append(
        f"Combining the modelled `{micro_tot_pct:.1f}%` stateless microphysics "
        f"share (Section 2) with Amdahl's law fixes an upper bound of "
        f"`{amdahl_ceil:.2f}x` for any hybrid CPU-Henyey + GPU-microphysics "
        f"step, even before finite device speedup and host-device overheads. "
        f"The analytical grid table below explicitly assumes a hypothetical "
        f"**`{dev_sp:.1f}x` device speedup** for pure-GPU stateless "
        f"microphysics over `2`-thread Safe Rust (assuming `N1-24`/`N1-25` "
        f"host callbacks were removed), together with PCIe transfers "
        f"(`{pcie_bw:.0f} GB/s`), XLA dispatch latency (`{xla_us:.0f} us` per "
        f"Newton iteration), and heterogeneous adaptive-timestep lockstep "
        f"efficiency (`{lock_eff:.2f}`). By contrast, standalone stateless "
        f"grid evaluations (`SED` photometry, screening, analytical EOS, MLT "
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
    lines.append(
        f"*Model footnote: Both GPU columns assume a hypothetical "
        f"`{dev_sp:.1f}x` device kernel speedup over `2`-thread Safe Rust "
        f"(`run_evolution_profile_26_4_1.py`, `accelerator_grid_estimate.json` "
        f"`hypothetical_device_micro_speedup_vs_2t_rust`), "
        f"`{pcie_bw:.0f} GB/s` effective PCIe bandwidth, `{xla_us:.0f} us` "
        f"per Newton iteration, and `{lock_eff:.2f}` lockstep utilization "
        f"for `N > 1`, anchored to the Section 2 rescaled subsystem shares "
        f"and the Section 6.1 `smoke: {b_cap} steps` `8`-thread Rust pool "
        f"throughput.*"
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
        f"{amdahl_ceil:.2f}x` modelled Amdahl ceiling."
    )
    lines.append("")

    # Section 7: Plain Disclosure (What Was Not Verified)
    lines.append("## 7. What Was Not Verified")
    lines.append("")
    lines.append(
        "- **Agreement with MESA (`#143`)**: None of the four cases passes "
        "the upstream-history harness. Every FlexMESA run here is a diagnostic "
        "run (`allow_unsupported_controls = true`, exact counts in Section 1), "
        "and track differences remain non-zero even on the step-matched cases. "
        "No timing in this report represents a verified `#143` parity run."
    )
    lines.append(
        "- **Remaining I/O, timer, and build-flag differences**: Both engines "
        f"ran `{repeats}` timed repeats with history output on every repeat at "
        f"`{tc['omp_num_threads']}` threads, but MESA's stock `./rn` also "
        "writes `LOGS/profile*.data`, `photos/`, and terminal summaries "
        "whereas `profile_evolution` writes `history.data` on every step and "
        "`final.mod` (radial profiles are omitted because `semiconvection`'s "
        "`profile_columns.list` requests unported `grad_temperature`). MESA's "
        "internal evolve timer rounds to `0.01 min` (`0.6 s`), MESA "
        "zone-steps are estimated via trapezoidal integration of `num_zones` "
        "in `history.data`, and compiler defaults differ (`gfortran -O2` vs "
        "Cargo `opt-level=3, lto=fat`)."
    )
    lines.append(
        "- **Subsystem shares**: The per-subsystem shares are a rescaled "
        "bottom-up model (Section 2), not instrumented wall timings, and the "
        "stateless share and Amdahl ceiling in Section 6 inherit that model "
        "overshoot."
    )
    lines.append(
        f"- **Engine version**: FlexMESA measurements were taken at "
        f"`origin/main` `{main_short}` and have not been repeated on later "
        f"commits."
    )
    lines.append(
        f"- **Batching step cap**: Section 6.1 is a `smoke: {b_cap} steps` "
        f"benchmark on `low_z` (`{b_mode}`), not an uncapped evolution to "
        f"physical termination."
    )
    lines.append(
        "- **Case breadth**: Measurements cover four `star/test_suite` cases "
        f"({', '.join(f'`{c}`' for c in prov['selected_cases'])}) on a single "
        "`x86_64` workstation; late burning with `approx21.net`, hydrodynamic "
        "`rsp2` pulsations, `binary` evolutions, and `astero` simplex runs "
        "were not profiled in this campaign."
    )
    lines.append(
        "- **`aarch64` & multi-socket scaling**: `strict` vs `fast` ratios and "
        "`rayon` worker-pool curves were measured on one `8`-core / "
        "`16`-thread `x86_64` CPU under WSL2; Apple Silicon (`aarch64`) and "
        "high-core-count server NUMA scaling were not measured."
    )
    lines.append(
        f"- **`#150` git A/B scope**: Historical pre-`#150` "
        f"(`{pre150['short_commit']}`) vs post-`#150` "
        f"(`{post150['short_commit']}`) binary timing covers one case "
        f"(`{ab['case']}`) with sample spread comparable to the saving; "
        f"Section 4.2 separates direct `{comp_commit}` cache/memory "
        f"measurements from un-rescaled probe estimates."
    )
    lines.append(
        f"- **Grid-scale GPU evolution**: Because `src/flexmesa/star/` JAX "
        f"helpers are frozen (`JAX_ENGINE_STATUS = "
        f'"frozen-unverified-until-#143"`), the `N = 1..1024` hybrid GPU '
        f"numbers in Section 6.2 come from an analytical model assuming a "
        f"`{dev_sp:.1f}x` device speedup, `{pcie_bw:.0f} GB/s` PCIe, "
        f"`{xla_us:.0f} us` XLA launch overhead, and `{lock_eff:.2f}` "
        f"lockstep efficiency rather than an executed GPU Henyey engine."
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
