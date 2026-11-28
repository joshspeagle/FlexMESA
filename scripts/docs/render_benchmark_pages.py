# SPDX-License-Identifier: GPL-3.0-or-later
"""Render MESA `26.4.1` benchmark scorecards and wave reports (`#146`).

Builds the eight public benchmark Markdown pages deterministically from
committed raw captures under `benchmarks/raw/26.4.1/` using
`scripts/docs/raw_benchmarks.py`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import sys
import tomllib
from collections.abc import Callable
from typing import cast

try:
    import _common
    import raw_benchmarks
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import _common
    import raw_benchmarks

DEFAULT_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_REL = "scripts/docs/render_benchmark_pages.py"
GENERATED_COMMENT = _common.generated_comment(SCRIPT_REL, "#146")

_COMPARATOR_LABELS: dict[str, str] = {
    "mesa_26_4_1_all_stages": "MESA `r26.4.1` (all stages)",
    "mesa_26_4_1_stages_0_1_reference_fortran_stages_2_3": (
        "MESA `r26.4.1` (0-1) / Ref. Fortran (2-3)"
    ),
    "hybrid_mesa_helpers_and_reference_fortran": "Ref. Fortran (hybrid)",
    "standalone_reference_fortran_non_mesa": "Ref. Fortran (non-MESA)",
}


def _fmt_lat(timing: raw_benchmarks.StageTiming) -> str:
    return f"`{timing.median_us:.2f} +/- {timing.iqr_us:.2f}`"


def _fmt_med(timing: raw_benchmarks.StageTiming) -> str:
    return f"`{timing.median_us:.2f}`"


def _fmt_ms(timing: raw_benchmarks.StageTiming) -> str:
    return f"`{timing.compile_us / 1.0e3:.1f}`"


def _fmt_ratio(numerator: float, denominator: float) -> str:
    if denominator <= 0.0:
        return "`n/a`"
    return f"`{numerator / denominator:.2f}x`"


def _rel_diff(lhs: float, rhs: float) -> float:
    denom = abs(rhs)
    if denom == 0.0:
        return abs(lhs - rhs)
    return abs(lhs - rhs) / denom


#: Largest relative rounding error of a checksum printed with 9 significant
#: digits. Differences between a printed Fortran or Rust checksum and a
#: full-precision one cannot be resolved below this.
PRINT_ROUNDING_9_DIGITS = 5.0e-9


def _max_full_vs_printed(
    suite: raw_benchmarks.RawBenchmarkSuite, waves: tuple[str, ...]
) -> float:
    """Largest relative difference, over ``waves``, between the
    full-precision Python checksum and the 9-digit printed Fortran one."""
    worst = 0.0
    for key, f_rec in suite.fortran.items():
        p_rec = suite.python.get(key)
        if f_rec.wave in waves and p_rec is not None:
            worst = max(worst, _rel_diff(p_rec.checksum, f_rec.checksum))
    return worst


def _printed_checksum_range(
    suite: raw_benchmarks.RawBenchmarkSuite, wave: str
) -> tuple[float, float]:
    """Smallest and largest relative difference, over every workload and
    size of ``wave``, between the printed Rust and Fortran checksums."""
    diffs = [
        _rel_diff(suite.rust[key].checksum, f_rec.checksum)
        for key, f_rec in suite.fortran.items()
        if f_rec.wave == wave and key in suite.rust
    ]
    return min(diffs), max(diffs)


_WAVES_1_TO_3B: tuple[str, ...] = ("wave1", "wave2", "wave3a", "wave3b")


def _fmt_factor_or_pct(rel: float) -> str:
    """Format a relative difference as a percentage, or as a factor when
    it exceeds 100%."""
    if rel > 1.0:
        return f"a factor of `{1.0 + rel:.1f}`"
    return f"`{100.0 * rel:.2f}%`"


def _provenance_bullets(
    suite: raw_benchmarks.RawBenchmarkSuite,
) -> list[str]:
    prov = suite.provenance
    hw = cast(dict[str, object], prov["hardware"])
    tc = cast(dict[str, object], prov["toolchains"])
    comp = cast(dict[str, object], prov["comparators"])
    m_all = len(cast(list[object], comp["mesa_26_4_1_all_stages"]))
    m_mix = len(
        cast(
            list[object],
            comp["mesa_26_4_1_stages_0_1_reference_fortran_stages_2_3"],
        )
    )
    m_hyb = len(
        cast(
            list[object],
            comp["hybrid_mesa_helpers_and_reference_fortran"],
        )
    )
    m_non = len(
        cast(list[object], comp["standalone_reference_fortran_non_mesa"])
    )
    base_sha = str(prov["flexmesa_base_commit"])[:7]
    rerun_utc = str(prov["fortran_wave3b_wave4_clean_rerun_utc"])
    ci_flag = str(prov["ci_continuous_verification"]).lower()
    return [
        (
            "- **Raw archive**: `benchmarks/raw/26.4.1/` (`provenance.json`, "
            "`fortran_wave*.txt`, `rust_waves_1_to_5.txt`, "
            "`python_waves_1_to_5.json`)."
        ),
        (
            f"- **Capture timestamp (UTC)**: `{prov['date_utc']}` (with clean "
            "`fortran_wave3b.txt` and `fortran_wave4.txt` rerun from a "
            f"relative `MESA_DIR` layout at `{rerun_utc}` on base commit "
            f"`{base_sha}`)."
        ),
        (
            f"- **Continuous verification**: `ci_continuous_verification = "
            f"{ci_flag}` (CI builds and tests Safe Rust, PyO3, and JAX "
            "without a MESA installation, so the Fortran comparators run "
            "offline on a workstation)."
        ),
        (
            f"- **Host hardware**: `{hw['cpu_model']}` ({hw['logical_cores']} "
            f"logical cores, `OMP_NUM_THREADS={tc['omp_num_threads']}`, "
            f"`RAYON_NUM_THREADS={tc['rayon_max_threads']}`), "
            f"`{hw['gpu_model']}`, `{hw['os']}`."
        ),
        (
            f"- **Toolchains**: MESA `{tc['mesa_release']}`, SDK "
            f"`{tc['mesa_sdk']}`, `{tc['gfortran']}` "
            f"(`{tc['fortran_fflags']}`), `{tc['rustc']}`, `{tc['python']}`, "
            f"JAX `{tc['jax']}` (`{tc['jax_backend']}`, `jax_enable_x64=True`)."
        ),
        (
            f"- **Workload taxonomy (`{comp['total_workloads']}` workloads, "
            f"`{comp['total_records_per_suite']}` `(workload, N_zones)` "
            f"points)**: `{m_all}` full MESA `r26.4.1` workloads, `{m_mix}` "
            "mixed MESA `0-1` / Ref. Fortran `2-3` workloads (`auto_diff`, "
            f"`mtx_tridiag`, `chem_comp`), `{m_hyb}` hybrid Reference Fortran "
            "workloads (`w5_diff_rot`, `w5_binary_rlof`, `w5_astero_seism`), "
            f"and `{m_non}` standalone non-MESA Reference Fortran workload "
            "(`w5_riemann_rsp2`)."
        ),
    ]


def _render_scorecard_table(
    suite: raw_benchmarks.RawBenchmarkSuite,
    rows: list[tuple[str, str, int]],
    *,
    fortran_col_prefix: str,
    pyo3_mode: str,
) -> list[str]:
    lines = [
        (
            f"| Workload | Comparator | {fortran_col_prefix} 1t Stage 2 "
            f"($\\mu\\mathrm{{s}}$) | {fortran_col_prefix} 4t Stage 3 "
            "($\\mu\\mathrm{s}$) | Safe Rust 1t Stage 2 "
            "($\\mu\\mathrm{s}$) | Safe Rust 4t Stage 3 "
            "($\\mu\\mathrm{s}$) | PyO3 "
            f"{pyo3_mode} ($\\mu\\mathrm{{s}}$) | JAX `@jit` Stage 2 "
            "($\\mu\\mathrm{s}$) | JAX `vmap` Stage 3 "
            f"($\\mu\\mathrm{{s}}$) | {fortran_col_prefix} Checksum | "
            "Safe Rust Checksum | Rel. Diff |"
        ),
        (
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    for wl, desc, nz in rows:
        f_rec = suite.get("fortran", wl, nz)
        r_rec = suite.get("rust", wl, nz)
        p_rec = suite.get("python", wl, nz)
        if pyo3_mode == "Best":
            pyo3_val = min(p_rec.stage0.median_us, p_rec.stage1.median_us)
        elif pyo3_mode == "1t Stage 0":
            pyo3_val = p_rec.stage0.median_us
        else:
            pyo3_val = p_rec.stage1.median_us
        rel = _rel_diff(r_rec.checksum, f_rec.checksum)
        comp_lbl = _COMPARATOR_LABELS[suite.comparator_kind[wl]]
        lines.append(
            f"| `{wl}` ({desc}) | {comp_lbl} | {_fmt_med(f_rec.stage2)} | "
            f"{_fmt_med(f_rec.stage3)} | {_fmt_med(r_rec.stage2)} | "
            f"{_fmt_med(r_rec.stage3)} | `{pyo3_val:.2f}` | "
            f"{_fmt_med(p_rec.stage2)} | {_fmt_med(p_rec.stage3)} | "
            f"`{f_rec.checksum:.8e}` | `{r_rec.checksum:.8e}` | "
            f"`{rel:.2e}` |"
        )
    return lines


def _render_kernel_scorecards(
    suite: raw_benchmarks.RawBenchmarkSuite,
    _repo_root: pathlib.Path,
) -> str:
    bc = suite.fortran_bcyclic[4096]
    f_w4_rem = suite.get("fortran", "star_remesh", 1024)
    r_w4_rem = suite.get("rust", "star_remesh", 1024)
    p_w4_rem = suite.get("python", "star_remesh", 1024)
    f_w4_newt = suite.get("fortran", "star_newton_step", 1024)
    r_w4_newt = suite.get("rust", "star_newton_step", 1024)
    p_w4_newt = suite.get("python", "star_newton_step", 1024)
    f_w4_hyd = suite.get("fortran", "star_hydro_eqns", 1024)
    r_w4_hyd = suite.get("rust", "star_hydro_eqns", 1024)
    f_w4_mic = suite.get("fortran", "star_micro_vars", 1024)
    r_w4_mic = suite.get("rust", "star_micro_vars", 1024)
    f_w4_evo = suite.get("fortran", "star_evolve_step", 1024)
    r_w4_evo = suite.get("rust", "star_evolve_step", 1024)
    f_w5_dr = suite.get("fortran", "w5_diff_rot", 1024)
    r_w5_dr = suite.get("rust", "w5_diff_rot", 1024)
    f_w5_rl = suite.get("fortran", "w5_binary_rlof", 1024)
    r_w5_rl = suite.get("rust", "w5_binary_rlof", 1024)
    f_w5_as = suite.get("fortran", "w5_astero_seism", 1024)
    r_w5_as = suite.get("rust", "w5_astero_seism", 1024)
    f_i2d = suite.get("fortran", "interp_2d", 4096)
    r_i2d = suite.get("rust", "interp_2d", 4096)
    f_tri = suite.get("fortran", "mtx_tridiag", 4096)
    r_tri = suite.get("rust", "mtx_tridiag", 4096)
    f_fe56 = suite.get("fortran", "ion_fe56", 4096)
    r_fe56 = suite.get("rust", "ion_fe56", 4096)
    p_fe56 = suite.get("python", "ion_fe56", 4096)
    raw_url = (
        "https://github.com/joshspeagle/FlexMESA/tree/main/"
        "benchmarks/raw/26.4.1"
    )
    rel_mic = _rel_diff(r_w4_mic.checksum, f_w4_mic.checksum)
    rel_hyd = _rel_diff(r_w4_hyd.checksum, f_w4_hyd.checksum)
    rel_newt = _rel_diff(r_w4_newt.checksum, f_w4_newt.checksum)
    rel_evo = _rel_diff(r_w4_evo.checksum, f_w4_evo.checksum)
    rel_dr = _rel_diff(r_w5_dr.checksum, f_w5_dr.checksum)

    lines: list[str] = [
        GENERATED_COMMENT,
        "",
        "# Kernel Performance Scorecards (MESA `r26.4.1`)",
        "",
        (
            "These four-stage scorecards report the measurements in "
            "the committed MESA `r26.4.1` raw captures in "
            f"[`benchmarks/raw/26.4.1/`]({raw_url}). All Rust numbers were "
            "recorded in `NumericsMode::Strict` (`crlibm` elementary "
            "functions, no FMA contraction); see "
            "`notes/audit/fast_vs_strict.md` for `fast` vs. `strict` "
            "scaling (`#165`)."
        ),
        "",
        "## Provenance & Stage Semantics",
        "",
        *_provenance_bullets(suite),
        "",
        "Each workload reports medians in microseconds ($\\mu\\mathrm{s}$):",
        "",
        (
            "1. **Stage 0**: 1-thread baseline (`gfortran -O2`, Safe Rust "
            "`--release` strict mode, and 1-thread `flexmesa._rust` PyO3)."
        ),
        (
            "2. **Stage 1**: 1-thread contiguous Structure-of-Arrays (SoA) "
            "loops in Fortran and Rust, and 4-thread `rayon` `flexmesa._rust` "
            "PyO3 in Python."
        ),
        (
            "3. **Stage 2**: 1-thread hoisted/fused loops in Fortran and Rust "
            "(using Reference Fortran loops on `auto_diff`, `mtx_tridiag`, "
            "and `chem_comp`), and warm single-profile `@jax.jit` (`float64` "
            "GPU)."
        ),
        (
            "4. **Stage 3**: 4-thread OpenMP (`OMP_NUM_THREADS=4`) in Fortran, "
            "4-thread `rayon` in Safe Rust, and amortized per-profile batched "
            "`@jax.jit(jax.vmap)` ($B = 4$ profiles) in JAX."
        ),
        "",
        "---",
        "",
        (
            "## 1. Foundational Numerics, Splines, Neutrinos & Convection "
            "($N_{\\mathrm{zones}} = 4096$)"
        ),
        "",
        (
            "Full $N_{\\mathrm{zones}} \\in \\{128, 1024, 4096\\}$ tables "
            "live in the [Wave 1 Benchmark Report]"
            "(../design/wave-1-benchmark-report.md). Three Wave 1 caveats "
            "apply at $N_{\\mathrm{zones}} = 4096$:"
        ),
        "",
        (
            "- **Mixed comparators**: `auto_diff` (`eval_autodiff_fused`) "
            "and `mtx_tridiag` (`fast_block_thomas_solve`) Stages 2-3 time "
            "Reference Fortran loops rather than MESA library routines; "
            "MESA's single-matrix `bcyclic_dble_decsolblk` 4096-zone "
            f"timing is `{bc.t1_median_us:.2f} us` (1t) and "
            f"`{bc.t4_median_us:.2f} us` (4t OpenMP)."
        ),
        (
            "- **Workloads where Fortran is faster**: At 4 threads, Fortran "
            "MESA outperforms Safe Rust on `interp_2d` "
            f"(`{f_i2d.stage3.median_us:.2f} us` vs. "
            f"`{r_i2d.stage3.median_us:.2f} us`), and Reference Fortran batch "
            "Thomas outperforms Safe Rust on `mtx_tridiag` "
            f"(`{f_tri.stage3.median_us:.2f} us` vs. "
            f"`{r_tri.stage3.median_us:.2f} us`)."
        ),
        (
            "- **Checksum precision**: Fortran and Rust print 9 significant "
            "digits, and every Wave 1 printed checksum is identical. The "
            "full `float64` Python checksums differ from the printed Fortran "
            "values by at most "
            f"`{_max_full_vs_printed(suite, ('wave1',)):.2e}`, "
            f"within the `{PRINT_ROUNDING_9_DIGITS:.0e}` rounding of a 9-digit "
            "print, so these captures cannot show a difference below that."
        ),
        "",
        *_render_scorecard_table(
            suite,
            [
                ("auto_diff", "hyperdual", 4096),
                ("interp_2d", "bicubic", 4096),
                ("neu", "Itoh 5-channel", 4096),
                ("mtx_tridiag", "$5 \\times 5$ block", 4096),
                ("turb_mlt", "Cox MLT", 4096),
                ("turb_tdc", "Kuhfuss TDC", 4096),
            ],
            fortran_col_prefix="Fortran",
            pyo3_mode="Best",
        ),
        "",
        "---",
        "",
        (
            "## 2. Atomic Composition, Ionization & Synthetic Photometry "
            "($N_{\\mathrm{zones}} = 4096$)"
        ),
        "",
        (
            "Full $N_{\\mathrm{zones}} \\in \\{128, 1024, 4096\\}$ tables "
            "live in the [Wave 2 Benchmark Report]"
            "(../design/wave-2-benchmark-report.md). Stage 2/3 `chem_comp` "
            "Fortran timings run the hoisted Reference Fortran loop "
            "(`eval_comp_hoisted`). At 4 threads on `ion_fe56`, Fortran MESA "
            f"(`{f_fe56.stage3.median_us:.2f} us`) is faster than Safe Rust "
            f"Stage 3 (`{r_fe56.stage3.median_us:.2f} us`) and PyO3 Stage 1 "
            f"(`{p_fe56.stage1.median_us:.2f} us`)."
        ),
        "",
        *_render_scorecard_table(
            suite,
            [
                ("chem_comp", "$\\bar{A}, \\bar{Z}, Y_e$", 4096),
                ("ion_fe56", "$\\mathrm{Fe}^{56}$ in $\\mathrm{He}^4$", 4096),
                ("ion_4d", "4D partial ion.", 4096),
                ("colors_sed", "Kurucz + $UBVRI$", 4096),
            ],
            fortran_col_prefix="Fortran",
            pyo3_mode="4t Stage 1",
        ),
        "",
        "---",
        "",
        (
            "## 3. Equations of State, Opacity, Networks & Atmospheres "
            "($N_{\\mathrm{zones}} = 4096$)"
        ),
        "",
        (
            "All nine Wave 3a and Wave 3b workloads invoke compiled MESA "
            "`r26.4.1` subroutines in every Fortran stage and match printed "
            "checksums across Fortran, Safe Rust, and Python/JAX (`0.00e+00` "
            "at 9 significant digits; at most "
            f"`{_max_full_vs_printed(suite, ('wave3a', 'wave3b')):.2e}` "
            "between full `f64` JSON and the 9-digit printed values, within "
            "print rounding). Full per-size tables live in the "
            "[Wave 3a Report](../design/wave-3a-benchmark-report.md) and "
            "[Wave 3b Report](../design/wave-3b-benchmark-report.md). Note "
            "that cold XLA compilation for `eos_skye` on GPU took `60.7 s` "
            "(Stage 3 at `N_zones = 4096`) prior to warm execution."
        ),
        "",
        *_render_scorecard_table(
            suite,
            [
                ("eos_ideal", "neutral + rad", 4096),
                ("eos_helm", "Timmes & Swesty", 4096),
                ("eos_pc", "Potekhin-Chabrier", 4096),
                ("eos_skye", "3rd-order AD", 4096),
                ("eos_4d", "`OPAL`/`FreeEOS`", 4096),
                ("kap_fixed", "rad + cond", 4096),
                ("rates_screen", "REACLIB", 4096),
                ("net_eval", "`basic.net` + $J$", 4096),
                ("atm_table", "2D bicubic", 4096),
            ],
            fortran_col_prefix="MESA",
            pyo3_mode="4t Stage 1",
        ),
        "",
        "---",
        "",
        (
            "## 4. 1D Stellar Structure, Henyey Solver & Evolution "
            "($N_{\\mathrm{zones}} = 1024$)"
        ),
        "",
        (
            "Full tables and `star/test` comparisons live in the "
            "[Wave 4 Benchmark Report](../design/wave-4-benchmark-report.md). "
            "All five Wave 4 workloads run post-`51e1adc` (`need_to_setvars = "
            ".true.`), but **do not yet match MESA `26.4.1` numerically or "
            "execute equal work scope**:"
        ),
        "",
        (
            "- **Workloads where Fortran is faster**: At 4 threads, MESA "
            "`r26.4.1` outperforms Safe Rust on `star_hydro_eqns` "
            f"(`{f_w4_hyd.stage3.median_us:.2f} us` vs. "
            f"`{r_w4_hyd.stage3.median_us:.2f} us`) and `star_newton_step` "
            f"(`{f_w4_newt.stage3.median_us:.2f} us` vs. "
            f"`{r_w4_newt.stage3.median_us:.2f} us`). The `star_remesh` "
            f"timings (`{f_w4_rem.stage3.median_us:.2f} us` Fortran vs. "
            f"`{r_w4_rem.stage3.median_us:.2f} us` Rust 4t / "
            f"`{r_w4_rem.stage2.median_us:.2f} us` 1t Stage 2) are not a "
            "like-for-like comparison: the Fortran stage times only "
            "`set_mesh_function_data`, the Rust stage the whole "
            "`do_adjust_mesh`."
        ),
        (
            "- **PyO3 `star_newton_step` Stage 0 mismatch**: Python Stage 0 "
            f"PyO3 (`{p_w4_newt.stage0.median_us:.2f} us`) times only the "
            "`bcyclic` linear solve without calling `eval_equations`."
        ),
        (
            "- **Checksum discrepancies (`1024` zones)**: `star_micro_vars` "
            f"differs by `{rel_mic:.2e}`, `star_hydro_eqns` by "
            f"`{rel_hyd:.2e}`, `star_newton_step` by `{rel_newt:.2%}`, "
            f"`star_evolve_step` by `{rel_evo:.2e}` (comparing Fortran "
            "`dt = 100 s` in `do_struct_burn_mix` against Rust `base.dt`, "
            "and Fortran's `8`-zone pre-flight check logs "
            "`hydro_solver_step returned ierr -1`), and `star_remesh` shows "
            "a **three-way mismatch** across Fortran `set_mesh_function_data` "
            f"(`{f_w4_rem.checksum:.8e}`), Rust `do_adjust_mesh` "
            f"(`{r_w4_rem.checksum:.8e}`), and Python `eval_mesh_gval_jax` "
            f"(`{p_w4_rem.checksum:.8e}`)."
        ),
        "",
        *_render_scorecard_table(
            suite,
            [
                ("star_micro_vars", "EOS/kap/net/MLT", 1024),
                ("star_hydro_eqns", "residuals + $J$", 1024),
                ("star_newton_step", "Henyey solve", 1024),
                ("star_remesh", "3-way mismatch", 1024),
                ("star_evolve_step", "unequal `dt`", 1024),
            ],
            fortran_col_prefix="MESA",
            pyo3_mode="1t Stage 0",
        ),
        "",
        "---",
        "",
        (
            "## 5. Wave 5 Reference Fortran vs. `FlexMESA` "
            "($N_{\\mathrm{zones}} = 1024$)"
        ),
        "",
        (
            "Full tables live in the [Wave 5 Benchmark Report]"
            "(../design/wave-5-benchmark-report.md). Wave 5 compares "
            "`FlexMESA` against `benchmarks/fortran_mesa/"
            "bench_kernels_wave5.f90` (**Reference Fortran**, not upstream "
            "MESA evolution drivers): `w5_diff_rot`, `w5_binary_rlof`, and "
            "`w5_astero_seism` mix MESA utility subroutines with local "
            "Fortran formula loops (`hybrid`), while `w5_riemann_rsp2` calls "
            "zero MESA subroutines (`non-MESA`)."
        ),
        "",
        (
            "- **Workloads where Reference Fortran is faster**: At 4 threads, "
            "Reference Fortran is faster than Safe Rust on `w5_diff_rot` "
            f"(`{f_w5_dr.stage3.median_us:.2f} us` vs. "
            f"`{r_w5_dr.stage3.median_us:.2f} us`), `w5_binary_rlof` "
            f"(`{f_w5_rl.stage3.median_us:.2f} us` vs. "
            f"`{r_w5_rl.stage3.median_us:.2f} us`), and `w5_astero_seism` "
            f"(`{f_w5_as.stage3.median_us:.2f} us` vs. "
            f"`{r_w5_as.stage3.median_us:.2f} us`)."
        ),
        (
            "- **Spruit-Tayler checksum divergence**: `w5_diff_rot` "
            f"diverges by `{rel_dr:.2%}` between Reference Fortran "
            f"(`{f_w5_dr.checksum:.8e}`) and post-`642da54` Safe Rust/JAX "
            f"(`{r_w5_dr.checksum:.8e}`), tracked in `#158`."
        ),
        "",
        *_render_scorecard_table(
            suite,
            [
                ("w5_diff_rot", "Spruit divergence", 1024),
                ("w5_riemann_rsp2", "HLLC + `RSP2`", 1024),
                ("w5_binary_rlof", "RLOF + Kerr + tides", 1024),
                ("w5_astero_seism", "BG14 + $r_{01}/r_{10}$", 1024),
            ],
            fortran_col_prefix="Ref. Fortran",
            pyo3_mode="4t Stage 1",
        ),
        "",
    ]
    return "\n".join(lines)


def _baseline_counts(repo_root: pathlib.Path) -> tuple[int, int, int]:
    cases_toml = tomllib.loads(
        (
            repo_root / "benchmarks" / "history_harness" / "case_manifest.toml"
        ).read_text("utf-8")
    )
    gaps_toml = tomllib.loads(
        (
            repo_root / "benchmarks" / "history_harness" / "known_gaps.toml"
        ).read_text("utf-8")
    )
    base_json = json.loads(
        (
            repo_root
            / "benchmarks"
            / "history_harness"
            / "baseline_run_26_4_1.json"
        ).read_text("utf-8")
    )
    total_cases = len(cast(list[object], cases_toml["cases"]))
    declared_gaps = len(cast(list[object], gaps_toml["gaps"]))
    passing = sum(
        1
        for row in cast(list[dict[str, object]], base_json.get("cases", []))
        if row.get("verdict") == "pass"
    )
    return passing, total_cases, declared_gaps


def _render_overview(
    suite: raw_benchmarks.RawBenchmarkSuite,
    repo_root: pathlib.Path,
) -> str:
    passing, total_cases, declared_gaps = _baseline_counts(repo_root)
    awaiting = total_cases - passing - declared_gaps

    rep_rows: list[tuple[str, str, int, str]] = [
        (
            "Turbulent Convection",
            "turb_tdc",
            4096,
            "MESA `set_TDC`; scalar `f64` root + IFT (`DualOrder1<6>`)",
        ),
        (
            "Mixing Length Theory",
            "turb_mlt",
            4096,
            "MESA `set_MLT`; closed-form cubic root + IFT tangents",
        ),
        (
            "Synthetic Photometry",
            "colors_sed",
            1024,
            "MESA `colors`; 32-gather stencil + precomputed Simpson weights",
        ),
        (
            "Model Atmospheres",
            "atm_table",
            4096,
            "MESA `atm_eval_table`; pre-validated bicubic spline lookup",
        ),
        (
            "4D Tabulated EOS",
            "eos_4d",
            4096,
            "MESA `eosDT_get`; sorted `patch_lut` + stack 16-channel blend",
        ),
        (
            "Analytical `Skye` EOS",
            "eos_skye",
            4096,
            "MESA `eosDT_get_component`; `OnceLock<NuclideCatalog>` + AD",
        ),
        (
            "Nuclear Networks",
            "net_eval",
            4096,
            "MESA `net_get`; `real(qp)` sums + thread-local buffers",
        ),
        (
            "2D Bicubic Spline",
            "interp_2d",
            4096,
            (
                "MESA `interp_evbicub_db`; Fortran 4t faster; printed "
                "checksums identical"
            ),
        ),
        (
            "1D Hydro Equations",
            "star_hydro_eqns",
            1024,
            "MESA `eval_equ`; Fortran 4t faster; `1.60e-5` checksum diff",
        ),
        (
            "1D Stellar Evolution",
            "star_evolve_step",
            1024,
            "Unequal `dt = 100 s` vs `base.dt`; checksum diff `7.48e-6`",
        ),
        (
            "Riemann Hydro & `RSP2`",
            "w5_riemann_rsp2",
            1024,
            "Non-MESA Reference Fortran (`bench_kernels_wave5.f90`)",
        ),
    ]

    rep_table = [
        (
            "| Physics Domain | Workload ($N_{\\mathrm{zones}}$) | "
            "Comparator | Comparator 4t ($\\mu\\mathrm{s}$) | Safe Rust 4t "
            "($\\mu\\mathrm{s}$) | JAX `vmap` Batch/4 ($\\mu\\mathrm{s}$) | "
            "Scope & Result Notes |"
        ),
        "| :--- | :--- | :--- | ---: | ---: | ---: | :--- |",
    ]
    for domain, wl, nz, note in rep_rows:
        f = suite.get("fortran", wl, nz)
        r = suite.get("rust", wl, nz)
        p = suite.get("python", wl, nz)
        comp_lbl = _COMPARATOR_LABELS[suite.comparator_kind[wl]]
        rust_ratio = f.stage3.median_us / r.stage3.median_us
        jax_ratio = f.stage3.median_us / p.stage3.median_us
        rep_table.append(
            f"| **{domain}** | `{wl}` (`{nz}`) | {comp_lbl} | "
            f"`{f.stage3.median_us:.2f}` | "
            f"`{r.stage3.median_us:.2f}` (`{rust_ratio:.2f}x`) | "
            f"`{p.stage3.median_us:.2f}` (`{jax_ratio:.2f}x`) | {note} |"
        )

    lines: list[str] = [
        GENERATED_COMMENT,
        "",
        "# Verification & Benchmarks Overview",
        "",
        (
            "`FlexMESA` compares its numerical kernels, microphysics "
            "evaluators, linear solvers, and stellar evolution drivers "
            "against upstream Fortran MESA (release `26.4.1`, commit "
            "`c8c2026`, Zenodo record `19722306`) across four execution paths:"
        ),
        "",
        (
            "1. **Fortran Comparator (`gfortran -O2 -fopenmp`)**: Compiled "
            "against MESA `r26.4.1` static archives in "
            "`benchmarks/fortran_mesa/`. Across the `28` benchmark workloads, "
            "`21` call MESA subroutines in all stages, `3` call MESA in "
            "Stages 0-1 and Reference Fortran in Stages 2-3 (`auto_diff`, "
            "`mtx_tridiag`, `chem_comp`), `3` combine MESA utilities with "
            "local loops (`w5_diff_rot`, `w5_binary_rlof`, "
            "`w5_astero_seism`), and `1` is standalone non-MESA Reference "
            "Fortran (`w5_riemann_rsp2`)."
        ),
        (
            "2. **Safe Rust (`crates/flexmesa-*`)**: Canonical implementation "
            "in `NumericsMode::Strict` (`crlibm`, no FMA contraction, "
            "4-thread `rayon` pool)."
        ),
        (
            "3. **Python / PyO3 (`flexmesa._rust`)**: Zero-copy C-contiguous "
            "`numpy.float64` bindings releasing the Python GIL."
        ),
        (
            "4. **Python / JAX (`@jax.jit` & `jax.vmap` in `float64`)**: "
            "Derived XLA GPU kernels validated against Safe Rust `strict` "
            "mode (`#166`, `DECISIONS #17`, `#23`)."
        ),
        "",
        (
            f"The acceptance pass rate against MESA `26.4.1` references "
            f"currently stands at **{passing} / {total_cases} passing cases** "
            f"(`{declared_gaps}` declared gaps, `{awaiting}` awaiting "
            "reference runs; see [Declared Known Gaps](known-gaps.md) and "
            "[Port Coverage Map](coverage-map.md))."
        ),
        "",
        "---",
        "",
        "## Provenance & Representative MESA `r26.4.1` Timings",
        "",
        *_provenance_bullets(suite),
        "",
        (
            "- **Checksum accounting**: Across all `57` `(workload, "
            "N_zones)` operating points in Waves 1-3b, Safe Rust and Fortran "
            "match across all 9 printed significant digits. The full "
            "`float64` Python checksums differ from the printed Fortran "
            "values by at most "
            f"`{_max_full_vs_printed(suite, _WAVES_1_TO_3B):.2e}`, "
            f"within the `{PRINT_ROUNDING_9_DIGITS:.0e}` rounding of a 9-digit "
            "print. Wave 4 diverges on all five workloads: across all sizes "
            "the Rust and Fortran checksums differ by between "
            f"`{_printed_checksum_range(suite, 'wave4')[0]:.2e}` and "
            f"{_fmt_factor_or_pct(_printed_checksum_range(suite, 'wave4')[1])} "
            "(the `star_remesh` stages also time different work). Wave 5 "
            "`w5_diff_rot` diverges by `99.17%` (`#158`)."
        ),
        (
            "- **Cases where Fortran is faster**: At 4 threads, Fortran is "
            "faster than Safe Rust on `interp_2d`, `mtx_tridiag` (batch "
            "Thomas), `ion_fe56`, `star_hydro_eqns`, `star_newton_step`, "
            "`w5_diff_rot`, `w5_binary_rlof`, and `w5_astero_seism`. "
            "`star_remesh` is not listed either way: its Fortran stage times "
            "only `set_mesh_function_data`, while the Rust stage times the "
            "whole `do_adjust_mesh`."
        ),
        "",
        *rep_table,
        "",
        "---",
        "",
        (
            "## Visual Benchmark Gallery (MESA `r26.4.1`, "
            "`benchmarks/raw/26.4.1/`)"
        ),
        "",
        "### 1. Foundational Numerics, Splines, Neutrinos & Convection",
        "",
        (
            "![MESA 26.4.1 Wave 1 four-way runtime staircase for foundational "
            "numerics, splines, neutrinos, and convection]"
            "(../assets/figures/wave1_benchmark_staircase.png)"
        ),
        "",
        (
            "### 2. Composition Moments, Partial Ionization & Kurucz SED "
            "Photometry"
        ),
        "",
        (
            "![MESA 26.4.1 Wave 2 four-way runtime staircase for composition "
            "moments, Fe56 ionization, and Kurucz SED synthesis]"
            "(../assets/figures/wave2_benchmark_staircase.png)"
        ),
        "",
        (
            "### 3. Analytical & Single-Table Equations of State (`Ideal`, "
            "`HELM`, `PC`, `Skye`)"
        ),
        "",
        (
            "![MESA 26.4.1 Wave 3a four-way runtime staircase for Ideal, HELM, "
            "PC, and Skye equations of state]"
            "(../assets/figures/wave3a_benchmark_staircase.png)"
        ),
        "",
        (
            "### 4. 4D Tabulated EOS, Opacities, Reaction Networks & Model "
            "Atmospheres"
        ),
        "",
        (
            "![MESA 26.4.1 Wave 3b four-way runtime staircase for 4D EOS, "
            "opacity, reaction rates, networks, and atmospheres]"
            "(../assets/figures/wave3b_benchmark_staircase.png)"
        ),
        "",
        "### 5. 1D Stellar Structure, Henyey Solver & Coupled Evolution Step",
        "",
        (
            "![MESA 26.4.1 Wave 4 four-way runtime staircase for 1D stellar "
            "structure, Henyey Newton step, remeshing, and coupled evolution]"
            "(../assets/figures/wave4_benchmark_staircase.png)"
        ),
        "",
        (
            "### 6. Reference Fortran Comparator vs. FlexMESA (`w5_*`, "
            "Hybrid & Non-MESA)"
        ),
        "",
        (
            "![Wave 5 four-way runtime comparison against Reference Fortran "
            "hybrid and non-MESA comparators]"
            "(../assets/figures/wave5_benchmark_speedup.png)"
        ),
        "",
        "---",
        "",
        "## Reproducing the Benchmarks Locally",
        "",
        (
            "Run the standalone Rust benchmark binary, Python/JAX suite, or "
            "(when a local MESA `26.4.1` build is installed) the Fortran "
            "comparator drivers:"
        ),
        "",
        "```bash",
        (
            "# 1. Safe Rust Waves 1-5 CLI table (`--quick` runs N_zones=8 "
            "smoke pass)"
        ),
        (
            "cargo run --release -p flexmesa-bench --bin "
            "run_rust_benchmarks -- --quick"
        ),
        "",
        "# 2. Fortran comparator against a local MESA 26.4.1 checkout",
        "make -C benchmarks/fortran_mesa MESA_DIR=/path/to/mesa run",
        "```",
        "",
        "```python",
        "from flexmesa import bench",
        "",
        "# Quick 128-zone Wave 1 PyO3 / JAX timing pass",
        "records = bench.run_all_wave1_benchmarks(",
        "    zone_counts=(128,), n_warmup=1, n_repeats=2",
        ")",
        "for rec in records[:3]:",
        "    print(",
        '        f"{rec.workload:12s} n_zones={rec.n_zones:4d} "',
        '        f"stage0={rec.stage0.median_us:8.2f} us "',
        '        f"checksum={rec.checksum:.8e}"',
        "    )",
        "```",
        "",
        (
            "For all 28 benchmark workload tables, see "
            "**[Kernel Performance Scorecards](kernel-scorecards.md)**. For "
            "tolerance rules and documented discrepancies, see "
            "**[Tolerance & Discrepancy Ledger]"
            "(tolerance-and-discrepancy-ledger.md)**."
        ),
        "",
    ]
    return "\n".join(lines)


def _checksum_table_rows(
    suite: raw_benchmarks.RawBenchmarkSuite,
    workloads: tuple[str, ...],
    zone_counts: tuple[int, ...],
) -> list[str]:
    lines = [
        (
            "| Workload | $N_{\\mathrm{zones}}$ | Comparator Checksum | "
            "Safe Rust Checksum | Python/JAX Checksum | `|Rust - F90|` Rel. | "
            "`|JAX - F90|` Rel. | Rust `max_rel_err` | JAX `max_rel_err` |"
        ),
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for nz in zone_counts:
        for wl in workloads:
            f = suite.get("fortran", wl, nz)
            r = suite.get("rust", wl, nz)
            p = suite.get("python", wl, nz)
            rf = _rel_diff(r.checksum, f.checksum)
            pf = _rel_diff(p.checksum, f.checksum)
            lines.append(
                f"| `{wl}` | {nz} | `{f.checksum:.8e}` | `{r.checksum:.8e}` | "
                f"`{p.checksum:.8e}` | `{rf:.3e}` | `{pf:.3e}` | "
                f"`{r.max_rel_err:.3e}` | `{p.max_rel_err:.3e}` |"
            )
    return lines


def _four_stage_table(
    suite: raw_benchmarks.RawBenchmarkSuite,
    backend: str,
    workloads: tuple[str, ...],
    zone_counts: tuple[int, ...],
    *,
    stage3_label: str,
) -> list[str]:
    lines = [
        (
            f"| Workload | $N_{{\\mathrm{{zones}}}}$ | Stage 0 "
            f"($\\mu\\mathrm{{s}}$) | Stage 1 ($\\mu\\mathrm{{s}}$) | "
            f"Stage 2 ($\\mu\\mathrm{{s}}$) | {stage3_label} "
            "($\\mu\\mathrm{s}$) | Stage 0 / Stage 2 | "
            "Stage 0 / Stage 3 |"
        ),
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for nz in zone_counts:
        for wl in workloads:
            rec = suite.get(backend, wl, nz)
            lines.append(
                f"| `{wl}` | {nz} | {_fmt_lat(rec.stage0)} | "
                f"{_fmt_lat(rec.stage1)} | {_fmt_lat(rec.stage2)} | "
                f"{_fmt_lat(rec.stage3)} | "
                f"{_fmt_ratio(rec.stage0.median_us, rec.stage2.median_us)} | "
                f"{_fmt_ratio(rec.stage0.median_us, rec.stage3.median_us)} |"
            )
    return lines


def _python_jax_table(
    suite: raw_benchmarks.RawBenchmarkSuite,
    workloads: tuple[str, ...],
    zone_counts: tuple[int, ...],
) -> list[str]:
    lines = [
        (
            "| Workload | $N_{\\mathrm{zones}}$ | Stage 0 PyO3 1t "
            "($\\mu\\mathrm{s}$) | Stage 1 PyO3 4t "
            "($\\mu\\mathrm{s}$) | Stage 2 JAX `@jit` "
            "($\\mu\\mathrm{s}$) | Stage 2 Compile ($\\mathrm{ms}$) | "
            "Stage 3 JAX `vmap` ($\\mu\\mathrm{s}$) | Stage 3 Compile "
            "($\\mathrm{ms}$) |"
        ),
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for nz in zone_counts:
        for wl in workloads:
            rec = suite.get("python", wl, nz)
            lines.append(
                f"| `{wl}` | {nz} | {_fmt_lat(rec.stage0)} | "
                f"{_fmt_lat(rec.stage1)} | {_fmt_lat(rec.stage2)} | "
                f"{_fmt_ms(rec.stage2)} | {_fmt_lat(rec.stage3)} | "
                f"{_fmt_ms(rec.stage3)} |"
            )
    return lines


def _render_wave1_report(
    suite: raw_benchmarks.RawBenchmarkSuite,
    _repo_root: pathlib.Path,
) -> str:
    workloads = (
        "auto_diff",
        "interp_2d",
        "neu",
        "mtx_tridiag",
        "turb_mlt",
        "turb_tdc",
    )
    nzs = (128, 1024, 4096)
    bc128 = suite.fortran_bcyclic[128]
    bc1024 = suite.fortran_bcyclic[1024]
    bc4096 = suite.fortran_bcyclic[4096]
    f_mlt = suite.get("fortran", "turb_mlt", 4096)
    r_mlt = suite.get("rust", "turb_mlt", 4096)
    f_tdc = suite.get("fortran", "turb_tdc", 4096)
    r_tdc = suite.get("rust", "turb_tdc", 4096)
    p_tdc = suite.get("python", "turb_tdc", 4096)
    f_i2d = suite.get("fortran", "interp_2d", 4096)
    r_i2d = suite.get("rust", "interp_2d", 4096)
    f_tri = suite.get("fortran", "mtx_tridiag", 4096)
    r_tri = suite.get("rust", "mtx_tridiag", 4096)

    lines: list[str] = [
        GENERATED_COMMENT,
        "",
        "# Wave 1 Four-Way Performance Benchmark Report (MESA `r26.4.1`)",
        "",
        (
            '!!! info "Raw Capture Provenance (`benchmarks/raw/26.4.1/`, '
            '`#146`)"\n'
            "    Measured offline against MESA `r26.4.1` "
            "(`fortran_wave1.txt`, `rust_waves_1_to_5.txt`, "
            "`python_waves_1_to_5.json`). Stages 2 and 3 of `auto_diff` "
            "(`eval_autodiff_fused`) and `mtx_tridiag` "
            "(`fast_block_thomas_solve`) execute Reference Fortran loops "
            "rather than MESA subroutines; single-matrix MESA "
            "`bcyclic_dble_decsolblk` 4096-zone timings are "
            f"`{bc4096.t1_median_us:.2f} us` (1t) and "
            f"`{bc4096.t4_median_us:.2f} us` (4t OpenMP)."
        ),
        "",
        "## 1. Environment & Reproduction",
        "",
        *_provenance_bullets(suite),
        "",
        "```bash",
        "make -C benchmarks/fortran_mesa MESA_DIR=/path/to/mesa run_wave1",
        (
            "cargo run --release -p flexmesa-bench --bin "
            "run_rust_benchmarks -- --quick"
        ),
        "```",
        "",
        (
            "## 2. Checksum Consistency "
            "($N_{\\mathrm{zones}} \\in \\{128, 1024, 4096\\}$)"
        ),
        "",
        (
            "All 18 Wave 1 `(workload, N_zones)` pairs agree across all 9 "
            "printed significant digits. The full `float64` Python checksums "
            "differ from the 9-digit printed Fortran values by at most "
            f"`{_max_full_vs_printed(suite, ('wave1',)):.2e}`, within the "
            f"`{PRINT_ROUNDING_9_DIGITS:.0e}` rounding of a 9-digit print, so "
            "these captures cannot show a difference below that."
        ),
        "",
        *_checksum_table_rows(suite, workloads, nzs),
        "",
        (
            "![Wave 1 four-way runtime staircase]"
            "(../assets/figures/wave1_benchmark_staircase.png)"
        ),
        "",
        "## 3. Fortran MESA `r26.4.1` & Reference Comparator Results",
        "",
        *_four_stage_table(
            suite,
            "fortran",
            workloads,
            nzs,
            stage3_label="Stage 3 (4t OpenMP)",
        ),
        "",
        "Single-matrix MESA `bcyclic_dble_decsolblk` OpenMP scaling:",
        "",
        (
            f"- `N_zones = 128`: `{bc128.t1_median_us:.2f} us` (1t) -> "
            f"`{bc128.t4_median_us:.2f} us` (4t OpenMP, "
            f"{_fmt_ratio(bc128.t1_median_us, bc128.t4_median_us)})."
        ),
        (
            f"- `N_zones = 1024`: `{bc1024.t1_median_us:.2f} us` (1t) -> "
            f"`{bc1024.t4_median_us:.2f} us` (4t OpenMP, "
            f"{_fmt_ratio(bc1024.t1_median_us, bc1024.t4_median_us)})."
        ),
        (
            f"- `N_zones = 4096`: `{bc4096.t1_median_us:.2f} us` (1t) -> "
            f"`{bc4096.t4_median_us:.2f} us` (4t OpenMP, "
            f"{_fmt_ratio(bc4096.t1_median_us, bc4096.t4_median_us)})."
        ),
        "",
        "## 4. Safe Rust (`strict`, `--release`) Results",
        "",
        *_four_stage_table(
            suite,
            "rust",
            workloads,
            nzs,
            stage3_label="Stage 3 (4t `rayon`)",
        ),
        "",
        "## 5. Python / PyO3 & JAX `float64` Results",
        "",
        *_python_jax_table(suite, workloads, nzs),
        "",
        "## 6. Key Wave 1 Findings ($N_{\\mathrm{zones}} = 4096$)",
        "",
        (
            "1. **IFT + `DualOrder1<6>` on `turb_mlt` and `turb_tdc`**: At 4 "
            "threads, Safe Rust runs `turb_mlt` in "
            f"`{r_mlt.stage3.median_us:.2f} us` vs. "
            f"`{f_mlt.stage3.median_us:.2f} us` in MESA `set_MLT` "
            f"({_fmt_ratio(f_mlt.stage3.median_us, r_mlt.stage3.median_us)}), "
            f"and `turb_tdc` in `{r_tdc.stage3.median_us:.2f} us` vs. "
            f"`{f_tdc.stage3.median_us:.2f} us` in MESA `set_TDC` "
            f"({_fmt_ratio(f_tdc.stage3.median_us, r_tdc.stage3.median_us)}); "
            f"batched GPU `@jax.jit(jax.vmap)` evaluates `turb_tdc` in "
            f"`{p_tdc.stage3.median_us:.2f} us` per profile "
            f"({_fmt_ratio(f_tdc.stage3.median_us, p_tdc.stage3.median_us)})."
        ),
        (
            "2. **Workloads where Fortran is faster**: At 4 threads, Fortran "
            "MESA is faster than Safe Rust on `interp_2d` "
            f"(`{f_i2d.stage3.median_us:.2f} us` vs. "
            f"`{r_i2d.stage3.median_us:.2f} us`, "
            f"{_fmt_ratio(f_i2d.stage3.median_us, r_i2d.stage3.median_us)}), "
            "and Reference Fortran batch Thomas (`fast_block_thomas_solve`) "
            f"is faster on `mtx_tridiag` (`{f_tri.stage3.median_us:.2f} us` "
            f"vs. `{r_tri.stage3.median_us:.2f} us`, "
            f"{_fmt_ratio(f_tri.stage3.median_us, r_tri.stage3.median_us)})."
        ),
        "",
    ]
    return "\n".join(lines)


def _render_wave2_report(
    suite: raw_benchmarks.RawBenchmarkSuite,
    _repo_root: pathlib.Path,
) -> str:
    workloads = ("chem_comp", "ion_fe56", "ion_4d", "colors_sed")
    nzs = (128, 1024, 4096)
    f_fe = suite.get("fortran", "ion_fe56", 4096)
    r_fe = suite.get("rust", "ion_fe56", 4096)
    p_fe = suite.get("python", "ion_fe56", 4096)
    f_col = suite.get("fortran", "colors_sed", 4096)
    r_col = suite.get("rust", "colors_sed", 4096)
    p_col_1024 = suite.get("python", "colors_sed", 1024)
    f_col_1024 = suite.get("fortran", "colors_sed", 1024)

    lines: list[str] = [
        GENERATED_COMMENT,
        "",
        "# Wave 2 Four-Way Performance Benchmark Report (MESA `r26.4.1`)",
        "",
        (
            '!!! info "Raw Capture Provenance (`benchmarks/raw/26.4.1/`, '
            '`#146`)"\n'
            "    Measured offline against MESA `r26.4.1` (`fortran_wave2.txt`, "
            "`rust_waves_1_to_5.txt`, `python_waves_1_to_5.json`). Stage 2/3 "
            "Fortran numbers for `chem_comp` execute `eval_comp_hoisted` in "
            "`bench_kernels_wave2.f90` rather than MESA "
            "`basic_composition_info`."
        ),
        "",
        "## 1. Environment & Checksum Consistency",
        "",
        *_provenance_bullets(suite),
        "",
        (
            "All 12 `(workload, N_zones)` pairs agree across all 9 printed "
            "significant digits. The full `f64` Python checksums differ from "
            "the printed Fortran values by at most "
            f"`{_max_full_vs_printed(suite, ('wave2',)):.2e}`, within the "
            "rounding of a 9-digit print."
        ),
        "",
        *_checksum_table_rows(suite, workloads, nzs),
        "",
        (
            "![Wave 2 four-way runtime staircase]"
            "(../assets/figures/wave2_benchmark_staircase.png)"
        ),
        "",
        (
            "## 2. Four-Way Runtime Tables "
            "($N_{\\mathrm{zones}} \\in \\{128, 1024, 4096\\}$)"
        ),
        "",
        (
            "### 2.1 Fortran Comparator "
            "(`benchmarks/fortran_mesa/run_wave2_bench`)"
        ),
        "",
        *_four_stage_table(
            suite,
            "fortran",
            workloads,
            nzs,
            stage3_label="Stage 3 (4t OpenMP)",
        ),
        "",
        "### 2.2 Safe Rust (`strict`, `--release`)",
        "",
        *_four_stage_table(
            suite,
            "rust",
            workloads,
            nzs,
            stage3_label="Stage 3 (4t `rayon`)",
        ),
        "",
        "### 2.3 Python / PyO3 & JAX `float64`",
        "",
        *_python_jax_table(suite, workloads, nzs),
        "",
        "## 3. Key Wave 2 Findings",
        "",
        (
            "1. **`colors_sed` 32-gather stencil & Simpson weights**: At 4096 "
            "zones, 4-thread Safe Rust evaluates `colors_sed` in "
            f"`{r_col.stage3.median_us:.2f} us` vs. "
            f"`{f_col.stage3.median_us:.2f} us` in MESA "
            f"({_fmt_ratio(f_col.stage3.median_us, r_col.stage3.median_us)}). "
            "At 1024 zones, `@jax.jit(jax.vmap)` reaches "
            f"`{p_col_1024.stage3.median_us:.2f} us` per profile vs. "
            f"`{f_col_1024.stage3.median_us:.2f} us` in MESA ("
            + _fmt_ratio(
                f_col_1024.stage3.median_us, p_col_1024.stage3.median_us
            )
            + ")."
        ),
        (
            "2. **`ion_fe56` comparison**: At 4096 zones and 4 threads, MESA "
            f"`eval_charge_of_Fe56_in_He4` (`{f_fe.stage3.median_us:.2f} us`) "
            "is faster than Safe Rust Stage 3 "
            f"(`{r_fe.stage3.median_us:.2f} us`) and PyO3 Stage 1 "
            f"(`{p_fe.stage1.median_us:.2f} us`)."
        ),
        "",
    ]
    return "\n".join(lines)


def _render_wave3a_report(
    suite: raw_benchmarks.RawBenchmarkSuite,
    _repo_root: pathlib.Path,
) -> str:
    workloads = ("eos_ideal", "eos_helm", "eos_pc", "eos_skye")
    nzs = (128, 1024, 4096)
    f_skye = suite.get("fortran", "eos_skye", 4096)
    r_skye = suite.get("rust", "eos_skye", 4096)
    p_skye = suite.get("python", "eos_skye", 4096)
    f_helm = suite.get("fortran", "eos_helm", 4096)
    r_helm = suite.get("rust", "eos_helm", 4096)

    lines: list[str] = [
        GENERATED_COMMENT,
        "",
        "# Wave 3a Four-Way Performance Benchmark Report (MESA `r26.4.1`)",
        "",
        (
            '!!! info "Raw Capture Provenance (`benchmarks/raw/26.4.1/`, '
            '`#146`)"\n'
            "    Measured offline against MESA `r26.4.1` "
            "(`fortran_wave3a.txt`, `rust_waves_1_to_5.txt`, "
            "`python_waves_1_to_5.json`). All four workloads call compiled "
            "MESA `r26.4.1` `eosDT_get_component` routines in every Fortran "
            "stage."
        ),
        "",
        "## 1. Environment & Checksum Consistency",
        "",
        *_provenance_bullets(suite),
        "",
        (
            "All 12 `(workload, N_zones)` pairs agree across all 9 printed "
            "significant digits (at most "
            f"`{_max_full_vs_printed(suite, ('wave3a',)):.2e}` between JSON "
            "and 9-digit printed sums, within print rounding)."
        ),
        "",
        *_checksum_table_rows(suite, workloads, nzs),
        "",
        (
            "![Wave 3a four-way runtime staircase]"
            "(../assets/figures/wave3a_benchmark_staircase.png)"
        ),
        "",
        (
            "## 2. Four-Way Runtime Tables "
            "($N_{\\mathrm{zones}} \\in \\{128, 1024, 4096\\}$)"
        ),
        "",
        (
            "### 2.1 Fortran MESA `r26.4.1` "
            "(`benchmarks/fortran_mesa/run_wave3a_bench`)"
        ),
        "",
        *_four_stage_table(
            suite,
            "fortran",
            workloads,
            nzs,
            stage3_label="Stage 3 (4t OpenMP)",
        ),
        "",
        "### 2.2 Safe Rust (`strict`, `--release`)",
        "",
        *_four_stage_table(
            suite,
            "rust",
            workloads,
            nzs,
            stage3_label="Stage 3 (4t `rayon`)",
        ),
        "",
        "### 2.3 Python / PyO3 & JAX `float64`",
        "",
        *_python_jax_table(suite, workloads, nzs),
        "",
        "## 3. Key Wave 3a Findings ($N_{\\mathrm{zones}} = 4096$)",
        "",
        (
            "1. **`eos_helm` & analytical EOS throughput**: At 4096 zones and "
            "4 threads, Safe Rust evaluates `eos_helm` in "
            f"`{r_helm.stage3.median_us:.2f} us` vs. "
            f"`{f_helm.stage3.median_us:.2f} us` in MESA "
            f"({_fmt_ratio(f_helm.stage3.median_us, r_helm.stage3.median_us)})."
        ),
        (
            "2. **`eos_skye` warm speed vs. XLA compile cost**: Warm GPU "
            "`@jax.jit(jax.vmap)` evaluates `eos_skye` in "
            f"`{p_skye.stage3.median_us:.2f} us` per profile vs. "
            f"`{r_skye.stage3.median_us:.2f} us` in 4t Safe Rust and "
            f"`{f_skye.stage3.median_us:.2f} us` in 4t MESA ("
            f"{_fmt_ratio(f_skye.stage3.median_us, p_skye.stage3.median_us)}"
            "), at the cost of "
            f"`{p_skye.stage3.compile_us / 1.0e3:.1f} ms` first-call XLA "
            "compilation."
        ),
        "",
    ]
    return "\n".join(lines)


def _render_wave3b_report(
    suite: raw_benchmarks.RawBenchmarkSuite,
    _repo_root: pathlib.Path,
) -> str:
    workloads = ("eos_4d", "kap_fixed", "rates_screen", "net_eval", "atm_table")
    nzs = (128, 1024, 4096)
    f_eos4d = suite.get("fortran", "eos_4d", 4096)
    r_eos4d = suite.get("rust", "eos_4d", 4096)
    p_eos4d = suite.get("python", "eos_4d", 4096)
    f_atm = suite.get("fortran", "atm_table", 4096)
    r_atm = suite.get("rust", "atm_table", 4096)
    f_net = suite.get("fortran", "net_eval", 4096)
    r_net = suite.get("rust", "net_eval", 4096)

    lines: list[str] = [
        GENERATED_COMMENT,
        "",
        "# Wave 3b Four-Way Performance Benchmark Report (MESA `r26.4.1`)",
        "",
        (
            '!!! info "Raw Capture Provenance (`benchmarks/raw/26.4.1/`, '
            '`#146`)"\n'
            "    Measured offline against MESA `r26.4.1` (`fortran_wave3b.txt` "
            "regenerated cleanly from a relative `MESA_DIR` working directory "
            "without Fortran format-width overflow, `rust_waves_1_to_5.txt`, "
            "`python_waves_1_to_5.json`). All five workloads call MESA "
            "`r26.4.1` routines in every Fortran stage."
        ),
        "",
        "## 1. Environment & Checksum Consistency",
        "",
        *_provenance_bullets(suite),
        "",
        (
            "All 15 `(workload, N_zones)` pairs agree across all 9 printed "
            "significant digits across MESA `r26.4.1`, Safe Rust `strict`, "
            "and Python/JAX `float64`."
        ),
        "",
        *_checksum_table_rows(suite, workloads, nzs),
        "",
        (
            "![Wave 3b four-way runtime staircase]"
            "(../assets/figures/wave3b_benchmark_staircase.png)"
        ),
        "",
        (
            "## 2. Four-Way Runtime Tables "
            "($N_{\\mathrm{zones}} \\in \\{128, 1024, 4096\\}$)"
        ),
        "",
        (
            "### 2.1 Fortran MESA `r26.4.1` "
            "(`benchmarks/fortran_mesa/run_wave3b_bench`)"
        ),
        "",
        *_four_stage_table(
            suite,
            "fortran",
            workloads,
            nzs,
            stage3_label="Stage 3 (4t OpenMP)",
        ),
        "",
        "### 2.2 Safe Rust (`strict`, `--release`)",
        "",
        *_four_stage_table(
            suite,
            "rust",
            workloads,
            nzs,
            stage3_label="Stage 3 (4t `rayon`)",
        ),
        "",
        "### 2.3 Python / PyO3 & JAX `float64`",
        "",
        *_python_jax_table(suite, workloads, nzs),
        "",
        "## 3. Key Wave 3b Findings ($N_{\\mathrm{zones}} = 4096$)",
        "",
        (
            "1. **4D tabulated EOS (`eos_4d`)**: At 4096 zones, 4-thread Safe "
            f"Rust runs in `{r_eos4d.stage3.median_us:.2f} us` vs. "
            f"`{f_eos4d.stage3.median_us:.2f} us` in MESA ("
            + _fmt_ratio(f_eos4d.stage3.median_us, r_eos4d.stage3.median_us)
            + "), and GPU `@jax.jit(jax.vmap)` reaches "
            f"`{p_eos4d.stage3.median_us:.2f} us` per profile ("
            + _fmt_ratio(f_eos4d.stage3.median_us, p_eos4d.stage3.median_us)
            + ")."
        ),
        (
            "2. **Networks and atmospheres**: At 4096 zones and 4 threads, "
            "Safe Rust evaluates `net_eval` in "
            f"`{r_net.stage3.median_us:.2f} us` vs. "
            f"`{f_net.stage3.median_us:.2f} us` in MESA ("
            + _fmt_ratio(f_net.stage3.median_us, r_net.stage3.median_us)
            + f"), and `atm_table` in `{r_atm.stage3.median_us:.2f} us` vs. "
            f"`{f_atm.stage3.median_us:.2f} us` in MESA ("
            + _fmt_ratio(f_atm.stage3.median_us, r_atm.stage3.median_us)
            + ")."
        ),
        "",
    ]
    return "\n".join(lines)


def _render_wave4_report(
    suite: raw_benchmarks.RawBenchmarkSuite,
    _repo_root: pathlib.Path,
) -> str:
    workloads = (
        "star_micro_vars",
        "star_hydro_eqns",
        "star_newton_step",
        "star_remesh",
        "star_evolve_step",
    )
    nzs = (128, 512, 1024)
    f_hyd = suite.get("fortran", "star_hydro_eqns", 1024)
    r_hyd = suite.get("rust", "star_hydro_eqns", 1024)
    f_newt = suite.get("fortran", "star_newton_step", 1024)
    r_newt = suite.get("rust", "star_newton_step", 1024)
    p_newt = suite.get("python", "star_newton_step", 1024)
    f_rem = suite.get("fortran", "star_remesh", 1024)
    r_rem = suite.get("rust", "star_remesh", 1024)
    p_rem = suite.get("python", "star_remesh", 1024)
    f_mic = suite.get("fortran", "star_micro_vars", 1024)
    r_mic = suite.get("rust", "star_micro_vars", 1024)
    f_evo = suite.get("fortran", "star_evolve_step", 1024)
    r_evo = suite.get("rust", "star_evolve_step", 1024)
    rel_mic = _rel_diff(r_mic.checksum, f_mic.checksum)
    rel_hyd = _rel_diff(r_hyd.checksum, f_hyd.checksum)
    rel_newt = _rel_diff(r_newt.checksum, f_newt.checksum)
    rel_evo = _rel_diff(r_evo.checksum, f_evo.checksum)

    lines: list[str] = [
        GENERATED_COMMENT,
        "",
        ("# Wave 4 Four-Way Benchmark & `star/test` Report (MESA `r26.4.1`)"),
        "",
        (
            '!!! warning "Known Wave 4 Parity & Work-Scope Discrepancies '
            '(`#146`)"\n'
            "    1. Raw MESA `r26.4.1` timings below were captured cleanly "
            "post-`51e1adc` (`need_to_setvars = .true.`, `fortran_wave4.txt`), "
            "but **all five workloads disagree in checksum** between MESA "
            "and `FlexMESA`.\n"
            "    2. `star_remesh` exhibits a three-way mismatch across "
            "Fortran, Rust, and Python/JAX; `star_newton_step` Stage 0 PyO3 "
            "times only the linear solve; `star_evolve_step` compares "
            "`dt = 100 s` in MESA `do_struct_burn_mix` against Rust "
            "`base.dt`, and MESA's 8-zone pre-flight run logs "
            "`hydro_solver_step returned ierr -1`."
        ),
        "",
        (
            "Wave 4 evaluates the 1D stellar structure engine across five "
            "workloads (`N_zones` in `{128, 512, 1024}`) and the `star/test` "
            "($4.0\\,M_\\odot, Z = 0.02$) trajectory."
        ),
        "",
        (
            "![Wave 4 four-way runtime staircase]"
            "(../assets/figures/wave4_benchmark_staircase.png)"
        ),
        "",
        "---",
        "",
        (
            "## 1. `star/test` Trajectory Comparison (`4.0 Msun, Z = 0.02` "
            "ZAMS to `1.00e5 yr`)"
        ),
        "",
        (
            "### 1.1 Historical Audit Finding on Pre-Audit Local Reference "
            "(`0b2e6bb`)"
        ),
        "",
        (
            "Evolving `StarModel.from_zams(4.0, 0.02)` to `max_age = "
            "1.00e5 yr` (11 steps, `do_remesh=False, op_split_burn=False`) "
            "in pre-audit docs quoted an uncommitted `0b2e6bb` run (`log_L "
            "= 2.374600` vs. `2.374853` in MESA `26.4.1`, leaving pre-audit "
            "`FlexMESA` `2.373933` `9.20e-4` dex away) with `1e-3` dex "
            "bounds wider than MESA's 11-step signal (`3.89e-4` dex on "
            "`log_L`, `1.59e-4` dex on `log_R`, `1.79e-5` dex on `log_Teff`):"
        ),
        "",
        (
            "| Observable | Pre-Audit Uncommitted Ref. | `FlexMESA` "
            "(`0b2e6bb`) | Pre-Audit Quoted Dev. | Pre-Audit Bound |"
        ),
        "| :--- | ---: | ---: | ---: | ---: |",
        (
            "| `log_center_T` | `7.412400` | `7.412298` | "
            "`1.02e-04` dex | `<= 1.0e-03` dex |"
        ),
        (
            "| `log_center_Rho` | `1.436800` | `1.436338` | "
            "`4.62e-04` dex | `<= 1.0e-03` dex |"
        ),
        (
            "| `log_L` | `2.374600` | `2.373933` | "
            "`6.67e-04` dex | `<= 1.0e-03` dex |"
        ),
        (
            "| `log_R` | `0.370300` | `0.370340` | "
            "`4.00e-05` dex | `<= 1.0e-03` dex |"
        ),
        (
            "| `log_Teff` | `4.169800` | `4.169640` | "
            "`1.60e-04` dex | `<= 1.0e-03` dex |"
        ),
        (
            "| `center_h1` | `0.696725` | `0.696732` | "
            "`7.00e-06` | `<= 5.0e-05` |"
        ),
        "",
        (
            "### 1.2 Committed MESA `26.4.1` `star/test` Reference & Measured "
            "Divergence"
        ),
        "",
        (
            "Under `#141` and `#143`, the real MESA `26.4.1` `star/test/LOGS/"
            "history.data` fixture (`tests/fixtures/golden/star_test/"
            "history.data`) is committed in git (`model_number` 1 at "
            "`3125 yr` -> `model_number` 11 at `1.00e5 yr`):"
        ),
        "",
        (
            "- **Committed MESA `26.4.1` Step-11 values**: "
            "`log_L = 2.374853`, `log_Teff = 4.169854`, "
            "`log_R = 0.370371`, `log_cntr_T = 7.412483`, "
            "`log_cntr_Rho = 1.437448`, `center_h1 = 0.6967304`."
        ),
        (
            "- **Regression status (`tests/test_wave4_bench.py`)**: MESA "
            "`26.4.1` remeshes `2464 -> 992` zones at step 2; `FlexMESA` runs "
            "with `do_remesh=False` currently diverge by `1.11e-3` dex (`> "
            "1.0e-3`) on `log_cntr_Rho` at step 11 (`xfail(strict=True)` "
            "tracked under `#153`)."
        ),
        "",
        (
            "### 1.3 MESA `r26.4.1` vs. `FlexMESA` Standalone Checksum "
            "Discrepancies"
        ),
        "",
        (
            "Provenance summary (`benchmarks/raw/26.4.1/`): every `1024`-zone "
            "checksum mismatch is measured directly below (`star_micro_vars` "
            f"`{rel_mic:.2e}`, `star_hydro_eqns` `{rel_hyd:.2e}`, "
            f"`star_newton_step` `{rel_newt:.2%}`, `star_evolve_step` "
            f"`{rel_evo:.2e}` with unequal `dt`, and `star_remesh` 3-way "
            f"mismatch across Fortran `{f_rem.checksum:.8e}`, Rust "
            f"`{r_rem.checksum:.8e}`, and Python `{p_rem.checksum:.8e}`)."
        ),
        "",
        *_checksum_table_rows(suite, workloads, nzs),
        "",
        "---",
        "",
        (
            "## 2. Four-Way Runtime Tables "
            "($N_{\\mathrm{zones}} \\in \\{128, 512, 1024\\}$)"
        ),
        "",
        "### 2.1 Environment",
        "",
        *_provenance_bullets(suite),
        "",
        (
            "### 2.2 Fortran MESA `r26.4.1` "
            "(`benchmarks/fortran_mesa/run_wave4_bench`)"
        ),
        "",
        *_four_stage_table(
            suite,
            "fortran",
            workloads,
            nzs,
            stage3_label="Stage 3 (4t OpenMP)",
        ),
        "",
        "### 2.3 Safe Rust (`strict`, `--release`)",
        "",
        *_four_stage_table(
            suite,
            "rust",
            workloads,
            nzs,
            stage3_label="Stage 3 (4t `rayon`)",
        ),
        "",
        "### 2.4 Python / PyO3 & JAX `float64`",
        "",
        *_python_jax_table(suite, workloads, nzs),
        "",
        "## 3. Key Wave 4 Disclosures ($N_{\\mathrm{zones}} = 1024$)",
        "",
        (
            "1. **Workloads where MESA Fortran is faster**: At 4 threads, "
            "MESA `r26.4.1` outperforms Safe Rust on `star_hydro_eqns` "
            f"(`{f_hyd.stage3.median_us:.2f} us` vs. "
            f"`{r_hyd.stage3.median_us:.2f} us`) and `star_newton_step` "
            f"(`{f_newt.stage3.median_us:.2f} us` vs. "
            f"`{r_newt.stage3.median_us:.2f} us`). The `star_remesh` "
            f"timings (`{f_rem.stage3.median_us:.2f} us` Fortran vs. "
            f"`{r_rem.stage3.median_us:.2f} us` Rust) are not a "
            "like-for-like comparison: the Fortran stage times only "
            "`set_mesh_function_data`, the Rust stage the whole "
            "`do_adjust_mesh`."
        ),
        (
            "2. **`star_newton_step` Stage 0 PyO3 "
            f"(`{p_newt.stage0.median_us:.2f} us`)**: Benchmarks only the Rust "
            "`bcyclic` linear solve without `eval_equations`, whereas Fortran "
            "and Rust full-step stages include equation residual and Jacobian "
            "setup."
        ),
        (
            "3. **`star_evolve_step` unequal timestep**: Fortran sets "
            "`s% dt = 100 s` (`N=8` pre-flight check logs `ierr -1`) while "
            "Rust steps with `base.dt`."
        ),
        "",
    ]
    return "\n".join(lines)


def _render_wave5_report(
    suite: raw_benchmarks.RawBenchmarkSuite,
    _repo_root: pathlib.Path,
) -> str:
    workloads = (
        "w5_diff_rot",
        "w5_riemann_rsp2",
        "w5_binary_rlof",
        "w5_astero_seism",
    )
    nzs = (128, 512, 1024)
    f_dr = suite.get("fortran", "w5_diff_rot", 1024)
    r_dr = suite.get("rust", "w5_diff_rot", 1024)
    f_riem = suite.get("fortran", "w5_riemann_rsp2", 1024)
    r_riem = suite.get("rust", "w5_riemann_rsp2", 1024)
    f_rlof = suite.get("fortran", "w5_binary_rlof", 1024)
    r_rlof = suite.get("rust", "w5_binary_rlof", 1024)
    f_ast = suite.get("fortran", "w5_astero_seism", 1024)
    r_ast = suite.get("rust", "w5_astero_seism", 1024)
    rel_dr = _rel_diff(r_dr.checksum, f_dr.checksum)

    lines: list[str] = [
        GENERATED_COMMENT,
        "",
        "# Wave 5 Four-Way Benchmark Report (`benchmarks/raw/26.4.1/`)",
        "",
        (
            '!!! warning "Reference Fortran Comparator Scope '
            '(Hybrid / Non-MESA, `#146`)"\n'
            "    Wave 5 Fortran numbers come from `benchmarks/fortran_mesa/"
            "bench_kernels_wave5.f90` (**Reference Fortran**, not upstream "
            "MESA evolution drivers): `w5_diff_rot`, `w5_binary_rlof`, and "
            "`w5_astero_seism` mix selected MESA helper routines with local "
            "Fortran formula loops (`hybrid`), while `w5_riemann_rsp2` calls "
            "zero MESA routines (`non-MESA`). On `w5_diff_rot`, Reference "
            f"Fortran (`{f_dr.checksum:.8e}`) diverges by `{rel_dr:.2%}` from "
            f"post-`642da54` Safe Rust/JAX (`{r_dr.checksum:.8e}`), tracked "
            "in `#158`."
        ),
        "",
        (
            "![Wave 5 four-way runtime comparison and zone-count scaling]"
            "(../assets/figures/wave5_benchmark_speedup.png)"
        ),
        "",
        "## 1. Environment & Comparator Classification",
        "",
        *_provenance_bullets(suite),
        "",
        (
            "- `w5_diff_rot` (**hybrid**): Calls MESA `hydro_rotation::a` and "
            "`log_term_power` alongside local Caplan OCP, rotational "
            "instability, and Spruit-Tayler formulas (`#158`)."
        ),
        (
            "- `w5_riemann_rsp2` (**non-MESA**): Standalone spherical HLLC + "
            "`RSP2` loop."
        ),
        (
            "- `w5_binary_rlof` (**hybrid**): Calls MESA "
            "`binary_utils::eval_rlobe` and `binary_tides::f3` / `f4` with "
            "standalone Ritter/Kerr/Jdot formulas."
        ),
        (
            "- `w5_astero_seism` (**hybrid**): Calls MESA "
            "`astero_get_cubic_all_freq_corr`, "
            "`astero_get_combined_all_freq_corr`, and "
            "`astero_support::get_frequency_ratios` with standalone scaling "
            "and acoustic integrals."
        ),
        "",
        "## 2. Checksum Comparison Across Backends",
        "",
        *_checksum_table_rows(suite, workloads, nzs),
        "",
        (
            "*(Note: Across the $s \\approx 0.435$ zero crossing of "
            "`w5_riemann_rsp2`, vector comparison uses the profile norm; see "
            "the [Tolerance & Discrepancy Ledger]"
            "(../benchmarks/tolerance-and-discrepancy-ledger.md"
            "#25-profile-norm-scaling-across-zero-crossings-in-"
            "w5_riemann_rsp2).)*"
        ),
        "",
        (
            "## 3. Four-Way Runtime Tables "
            "($N_{\\mathrm{zones}} \\in \\{128, 512, 1024\\}$)"
        ),
        "",
        "### 3.1 Reference Fortran (`benchmarks/fortran_mesa/run_wave5_bench`)",
        "",
        *_four_stage_table(
            suite,
            "fortran",
            workloads,
            nzs,
            stage3_label="Stage 3 (4t OpenMP)",
        ),
        "",
        "### 3.2 Safe Rust (`strict`, `--release`)",
        "",
        *_four_stage_table(
            suite,
            "rust",
            workloads,
            nzs,
            stage3_label="Stage 3 (4t `rayon`)",
        ),
        "",
        "### 3.3 Python / PyO3 & JAX `float64`",
        "",
        *_python_jax_table(suite, workloads, nzs),
        "",
        "## 4. Key Wave 5 Takeaways ($N_{\\mathrm{zones}} = 1024$)",
        "",
        (
            "1. **Workloads where Reference Fortran is faster**: At 4 "
            "threads, Reference Fortran outperforms Safe Rust on `w5_diff_rot` "
            f"(`{f_dr.stage3.median_us:.2f} us` vs. "
            f"`{r_dr.stage3.median_us:.2f} us`), `w5_binary_rlof` "
            f"(`{f_rlof.stage3.median_us:.2f} us` vs. "
            f"`{r_rlof.stage3.median_us:.2f} us`), and `w5_astero_seism` "
            f"(`{f_ast.stage3.median_us:.2f} us` vs. "
            f"`{r_ast.stage3.median_us:.2f} us`)."
        ),
        (
            "2. **`w5_riemann_rsp2` (non-MESA comparator)**: Safe Rust "
            f"executes in `{r_riem.stage3.median_us:.2f} us` (4t) vs. "
            f"`{f_riem.stage3.median_us:.2f} us` in the standalone OpenMP "
            "loop ("
            f"{_fmt_ratio(f_riem.stage3.median_us, r_riem.stage3.median_us)}"
            "), because the Fortran harness runs two consecutive "
            "`!$omp parallel do` barriers per iteration."
        ),
        "",
    ]
    return "\n".join(lines)


_PAGE_SPECS: tuple[
    tuple[
        pathlib.Path,
        Callable[[raw_benchmarks.RawBenchmarkSuite, pathlib.Path], str],
    ],
    ...,
] = (
    (
        pathlib.Path("docs") / "benchmarks" / "kernel-scorecards.md",
        _render_kernel_scorecards,
    ),
    (
        pathlib.Path("docs") / "benchmarks" / "overview.md",
        _render_overview,
    ),
    (
        pathlib.Path("docs") / "design" / "wave-1-benchmark-report.md",
        _render_wave1_report,
    ),
    (
        pathlib.Path("docs") / "design" / "wave-2-benchmark-report.md",
        _render_wave2_report,
    ),
    (
        pathlib.Path("docs") / "design" / "wave-3a-benchmark-report.md",
        _render_wave3a_report,
    ),
    (
        pathlib.Path("docs") / "design" / "wave-3b-benchmark-report.md",
        _render_wave3b_report,
    ),
    (
        pathlib.Path("docs") / "design" / "wave-4-benchmark-report.md",
        _render_wave4_report,
    ),
    (
        pathlib.Path("docs") / "design" / "wave-5-benchmark-report.md",
        _render_wave5_report,
    ),
)


def render_all_pages(repo_root: pathlib.Path) -> dict[pathlib.Path, str]:
    """Render all eight benchmark scorecards and wave report Markdown files.

    Args:
        repo_root: Root directory of the `FlexMESA` repository.

    Returns:
        Mapping from repository-relative Markdown path to rendered content.
    """
    suite = raw_benchmarks.load_raw_benchmarks(repo_root=repo_root)
    return {
        rel_path: renderer(suite, repo_root)
        for rel_path, renderer in _PAGE_SPECS
    }


def main(argv: list[str] | None = None) -> int:
    """Run CLI to render or verify benchmark Markdown documentation.

    Args:
        argv: Optional argument list overriding `sys.argv[1:]`.

    Returns:
        Exit code (`0` on success, `1` when `--check` detects drift).
    """
    parser = argparse.ArgumentParser(
        description="Render or check MESA 26.4.1 benchmark Markdown pages."
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=DEFAULT_REPO_ROOT,
        help="Path to FlexMESA repository root.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that generated Markdown files match without rewriting.",
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    rendered = render_all_pages(repo_root)

    if args.check:
        stale = False
        for rel_path, expected in rendered.items():
            out_path = repo_root / rel_path
            if not out_path.is_file():
                sys.stderr.write(f"Missing generated file: {out_path}\n")
                stale = True
                continue
            actual = out_path.read_text("utf-8")
            if actual != expected:
                diff = "".join(
                    difflib.unified_diff(
                        actual.splitlines(keepends=True),
                        expected.splitlines(keepends=True),
                        fromfile=str(rel_path),
                        tofile="expected",
                    )
                )
                sys.stderr.write(
                    f"{rel_path.as_posix()} is out of date. Run "
                    "`python scripts/docs/render_benchmark_pages.py`.\n"
                    f"{diff[:2000]}\n"
                )
                stale = True
        if stale:
            return 1
        print(
            f"Benchmark Markdown pages are up to date ({len(rendered)} files)."
        )
        return 0

    for rel_path, content in rendered.items():
        out_path = repo_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
