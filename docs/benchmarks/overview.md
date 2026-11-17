# Verification & Benchmarks Overview

`FlexMESA` compares its numerical kernels, microphysics table evaluators, linear solvers, and stellar evolution drivers against upstream Fortran MESA (release `26.4.1`, commit `c8c2026`, Zenodo record `19722306`) across four execution paths:

1. **Upstream Fortran MESA (`gfortran -O2 -fopenmp`)**: Native Fortran 2008 static archives (`libconst.a` through `libstar.a`, `libbinary.a`, and `libastero.a`) targeted by `benchmarks/fortran_mesa/`. The tables below reproduce pre-audit `0b2e6bb` local benchmark snapshots with no committed raw JSON logs, and several workloads (`auto_diff`, `mtx_tridiag` Stage 2/3, `chem_comp` Stage 2/3, and all Wave 5 cases) time hand-written Fortran loops rather than MESA library routines (`#146`). Re-baselining to MESA `26.4.1` is tracked in [#141](https://github.com/joshspeagle/FlexMESA/issues/141) and [#146](https://github.com/joshspeagle/FlexMESA/issues/146).
2. **Safe Rust (`crates/flexmesa-*`)**: Canonical evolution and microphysics workspace compiled with `--release`, using monomorphized `DualOrder1<N>` automatic differentiation, Implicit Function Theorem root derivatives, Hirshman et al. (2010) `bcyclic` odd-even block cyclic reduction, and a persistent 4-thread `rayon` pool (`strict` mode using `crlibm 1.0beta4` by default; `#165`).
3. **Python / PyO3 (`flexmesa._rust`)**: Zero-copy C-contiguous `numpy.float64` bindings releasing the Python GIL (`py.allow_threads`).
4. **Python / JAX (`@jax.jit` & `jax.vmap` in `float64`)**: Derived XLA-compiled flat-array kernels (`#166`) in IEEE-754 double precision (`jax_enable_x64=True`), validated against Safe Rust `"strict"` mode (`DECISIONS #17`, `#23`).

End-to-end evolutionary parity against MESA `26.4.1` currently stands at **0 / 106 passing cases** (`102` declared gaps, `4` awaiting reference runs; see [Declared Known Gaps](known-gaps.md) and [Port Coverage Map](coverage-map.md)).

---

## Pre-Audit Standalone Kernel Timings (Pending `#146` Re-Run)

In the pre-audit `0b2e6bb` local standalone benchmark runs below (no raw JSON output committed; hardware provenance and MESA `26.4.1` re-execution tracked in [#146](https://github.com/joshspeagle/FlexMESA/issues/146); `fast` vs. `strict` Rust differences documented in `notes/audit/fast_vs_strict.md` under [#165](https://github.com/joshspeagle/FlexMESA/issues/165)), Safe Rust, Python (`PyO3`), and JAX (`float64`) recorded the following representative kernel latencies across $N_{\mathrm{zones}} \in \{128, 512, 1024, 4096\}$:

- **Checksum & Work Scope Caveat**: Across all 28 pre-audit workloads, at most 17 agreed with MESA within `1e-12`: `interp_2d` and `ion_fe56` differ by `7.2e-9` and `4.83e-9` (`fvbicub_db` single-precision locals vs. `f64`), all five Wave 4 checksums diverge by `7.7e-5` to `169x` (and Fortran Wave 4 timings predate the `51e1adc` `need_to_setvars` fix), `star_evolve_step` compares unequal timesteps (`dt = 1e2 s` in Fortran vs. `base.dt` in Rust with omitted/stubbed sub-stages), and Wave 5 comparators are hand-written Fortran loops rather than MESA routines (with `w5_diff_rot` diverging by `120x`).

| Physics Domain | Representative Workload ($N_{\mathrm{zones}}$) | Pre-Audit Fortran Comparator 4t ($\mu\mathrm{s}$) | Safe Rust 4t `rayon` ($\mu\mathrm{s}$) | JAX `@jit(vmap)` Batch/4 ($\mu\mathrm{s}$) | Comparator Scope & Algorithmic Mechanism |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Turbulent Convection** | `turb_tdc` (`4096`) | `58176.83` (MESA `set_TDC`) | `1075.24` (`54.1x`) | `398.61` (`146.0x`) | Scalar `f64` root-find + post-convergence IFT (`DualOrder1<6>`) vs. MESA per-zone 70-slot AD loop |
| **Mixing Length Theory** | `turb_mlt` (`4096`) | `1784.93` (MESA `set_MLT`) | `142.14` (`12.6x`) | `317.73` (`5.6x`) | Closed-form scalar cubic root + IFT tangents vs. MESA `set_MLT` zone loop |
| **Synthetic Photometry** | `colors_sed` (`1024`) | `28570.68` (MESA `colors`) | `1638.59` (`17.4x`) | `744.64` (`38.4x`) | 32-gather interior stencil deduplication + precomputed Simpson filter weights |
| **Model Atmospheres** | `atm_table` (`4096`) | `650.79` (MESA `atm_eval_table`) | `62.66` (`10.4x`) | `167.47` (`3.9x`) | Pre-validated bicubic spline evaluation skipping per-call grid checks |
| **4D Tabulated EOS** | `eos_4d` (`4096`) | `1451.55` (MESA `eosDT_get`) | `994.56` (`1.46x`) | `143.95` (`10.1x`) | Lock-free `patch_lut` binary search + stack 16-channel $X$-plane interpolation |
| **Analytical / `Skye` EOS** | `eos_skye` (`4096`) | `7452.20` (MESA `eosDT_get_component`) | `6842.21` (`1.09x`) | `1421.04` (`5.24x`) | Process-wide `OnceLock<NuclideCatalog>`, stack buffers, and 3rd-order Taylor jets |
| **Nuclear Networks** | `net_eval` (`4096`) | `10185.83` (MESA `net_get`) | `3738.18` (`2.72x`) | `3709.81` (`2.75x`) | Dual-lane `F64x2` derivatives + thread-local work buffers |
| **1D Stellar Evolution** | `star_evolve_step` (`1024`) | `163495.99` (unequal `dt` / pre-`51e1adc`) | `25951.86` (differs `7.7e-5`) | `74358.93` | Fused 3-cell stencil + `bcyclic`; **not equal work** (`dt = 1e2 s` vs. `base.dt`, omitted physics) |
| **Riemann Hydro & `RSP2`** | `w5_riemann_rsp2` (`1024`) | `46.46` (**hand-written Fortran loop, non-MESA**) | `7.94` | `89.90` | Compares against agent-written Fortran loop (`bench_kernels_wave5.f90`, `#146`) |

---

## Visual Benchmark Gallery Across Domains (Pre-Audit `0b2e6bb` Data)

### 1. Foundational Numerics, Splines, Neutrinos & Convection

![Four-way pre-audit runtime staircase for foundational numerics, splines, neutrinos, and turbulent convection](../assets/figures/wave1_benchmark_staircase.png)

### 2. Composition Moments, Partial Ionization & Kurucz SED Photometry

![Four-way pre-audit runtime staircase for composition moments, Fe56 ionization, and Kurucz SED synthesis](../assets/figures/wave2_benchmark_staircase.png)

### 3. Analytical & Single-Table Equations of State (`Ideal`, `HELM`, `PC`, `Skye`)

![Four-way pre-audit runtime staircase for Ideal, HELM, PC, and Skye equations of state](../assets/figures/wave3a_benchmark_staircase.png)

### 4. 4D Tabulated EOS, Opacities, Reaction Networks & Model Atmospheres

![Four-way pre-audit runtime staircase for 4D EOS, opacity, reaction rates, networks, and atmospheres](../assets/figures/wave3b_benchmark_staircase.png)

### 5. 1D Stellar Structure, Henyey Solver & Coupled Evolution Step

![Four-way pre-audit runtime staircase for 1D stellar structure, Henyey Newton step, remeshing, and coupled evolution](../assets/figures/wave4_benchmark_staircase.png)

### 6. Standalone Hand-Written Fortran Comparator vs. FlexMESA (`w5_*`, Non-MESA)

![Four-way pre-audit runtime comparison against standalone hand-written Fortran loops for diffusion, rotation, HLLC/RSP2, binary RLOF, and asteroseismology](../assets/figures/wave5_benchmark_speedup.png)

---

## Reproducing the Benchmarks Locally

You can run the Rust/Python benchmark suite locally from Python via `flexmesa.bench` or execute `cargo test --release -p flexmesa-bench` (note that `benchmarks/fortran_mesa/Makefile` currently targets the pre-rebaseline `lib/` layout and awaits update to MESA `26.4.1` `build/<module>` paths under `#146`):

```python
from flexmesa import bench

# Run a quick 128-zone microphysics timing and snapshot pass (1 warmup, 2 timed reps)
records = bench.run_all_wave1_benchmarks(
    zone_counts=(128,), n_warmup=1, n_repeats=2
)
for rec in records[:3]:
    print(
        f"{rec.workload:12s} n_zones={rec.n_zones:4d} "
        f"stage0={rec.stage0.median_us:8.2f} us checksum={rec.checksum:.8e}"
    )
```

For full per-stage pre-audit latency tables and comparator disclosures, see **[Kernel Performance Scorecards](kernel-scorecards.md)**. For the three-level numerical fidelity policy and documented discrepancies, see **[Tolerance & Discrepancy Ledger](tolerance-and-discrepancy-ledger.md)**.
