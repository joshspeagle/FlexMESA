# Verification & Benchmarks Overview

Every numerical kernel, microphysics table evaluator, linear solver, and stellar evolution driver in `FlexMESA` is verified head-to-head against compiled upstream Fortran MESA (`r24.08.1`, `gfortran -O2 -fopenmp` static archives `lib*.a`) across four execution engines:

1. **Upstream Fortran MESA (`gfortran -O2 -fopenmp`)**: Native Fortran 2008 static archives (`libconst.a` through `libstar.a`, `libbinary.a`, and `libastero.a`) linked directly into the `benchmarks/fortran_mesa/` harness.
2. **Safe Rust (`crates/flexmesa-*`)**: `#![forbid(unsafe_code)]` Cargo workspace crates compiled with `--release`, using monomorphized `DualOrder1<N>` automatic differentiation, Implicit Function Theorem root derivatives, Hirshman et al. (2010) `bcyclic` odd-even block cyclic reduction, and a persistent 4-thread `rayon` worker pool.
3. **Python / PyO3 (`flexmesa._rust`)**: Zero-copy C-contiguous `numpy.float64` bindings releasing the Python GIL (`py.allow_threads`) and dispatching across a lock-free 4-thread adaptive spin-then-park worker team.
4. **Python / JAX (`@jax.jit` & `jax.vmap` in `float64`)**: XLA-compiled flat-array kernels in strict IEEE-754 double precision (`jax_enable_x64=True`), combining recursive `@jax.custom_jvp` spline basis rules, $\lceil \log_2 N_{\mathrm{zones}} \rceil$ vectorized Parallel Cyclic Reduction (`solve_block_tridiag_pcr`), and `@jax.custom_vjp` adjoint linear solvers.

---

## Executive Summary of Cross-Engine Speedups & Parity

Across all 28 benchmarked workloads spanning $N_{\mathrm{zones}} \in \{128, 512, 1024, 4096\}$, Safe Rust, Python (`PyO3`), and JAX (`float64`) match upstream Fortran MESA's 9-digit reduction checksums to $\le 10^{-12}$ (and match one another in IEEE-754 `f64` to $\le 10^{-14}$), while delivering substantial throughput gains across both CPU and GPU execution:

| Physics Domain | Representative Workload ($N_{\mathrm{zones}}$) | Fortran MESA 4t OpenMP ($\mu\mathrm{s}$) | Safe Rust 4t `rayon` ($\mu\mathrm{s}$) | JAX `@jit(vmap)` Batch/4 ($\mu\mathrm{s}$) | Primary Algorithmic & Systems Speedup Driver |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Turbulent Convection** | `turb_tdc` (`4096`) | `58176.83` | `1075.24` (`54.1x`) | `398.61` (`146.0x`) | Scalar `f64` Newton root-finding + Implicit Function Theorem (`DualOrder1<6>`) replacing 70-component nested AD in every iteration |
| **Mixing Length Theory** | `turb_mlt` (`4096`) | `1784.93` | `142.14` (`12.6x`) | `317.73` (`5.6x`) | Exact Cardano/Newton scalar root + post-convergence IFT tangents instead of 33-variable `star` AD propagation |
| **Synthetic Photometry** | `colors_sed` (`1024`) | `28570.68` | `1638.59` (`17.4x`) | `744.64` (`38.4x`) | 32-gather interior corner deduplication in 3D Hermite SED synthesis + precomputed Simpson filter weights |
| **Model Atmospheres** | `atm_table` (`4096`) | `650.79` | `62.66` (`10.4x`) | `167.47` (`3.9x`) | Pre-validated bicubic spline evaluation (`interp_evbicub_prevalidated_db`) skipping redundant $O(N_x N_y)$ grid checks |
| **4D Tabulated EOS** | `eos_4d` (`4096`) | `1451.55` | `994.56` (`1.46x`) | `143.95` (`10.1x`) | Lock-free `patch_lut` binary search + stack-only 16-channel $X$-plane interpolation + pre-blended GPU bicubic gather |
| **Analytical / `Skye` EOS** | `eos_skye` (`4096`) | `7452.20` | `6842.21` (`1.09x`) | `1421.04` (`5.24x`) | Process-wide `OnceLock<NuclideCatalog>`, stack species buffers, and single-pass direct `(T, rho)` 3rd-order JAX autodiff |
| **Nuclear Networks** | `net_eval` (`4096`) | `10185.83` | `3738.18` (`2.72x`) | `3709.81` (`2.75x`) | 2-lane SIMD dual numbers (`F64x2` carrying $\ln T$ and $\ln\rho$ partials simultaneously) + thread-local work-array reuse |
| **1D Stellar Evolution** | `star_evolve_step` (`1024`) | `163495.99` | `25951.86` (`6.30x`) | `74358.93` (`2.20x`) | Fused 3-cell stencil assembly + Hirshman `bcyclic` 4-thread block cyclic reduction + guarded MLT evaluation |
| **Riemann Hydro & `RSP2`** | `w5_riemann_rsp2` (`1024`) | `46.46` | `7.94` (`5.85x`) | `89.90` (`0.52x`) | Lock-free persistent 4-thread spin-barrier worker team eliminating consecutive OpenMP loop barriers |

---

## Visual Benchmark Gallery Across Domains

### 1. Foundational Numerics, Splines, Neutrinos & Convection

![Four-way runtime staircase for foundational numerics, splines, neutrinos, and turbulent convection](../assets/figures/wave1_benchmark_staircase.png)

### 2. Composition Moments, Partial Ionization & Kurucz SED Photometry

![Four-way runtime staircase for composition moments, Fe56 ionization, and Kurucz SED synthesis](../assets/figures/wave2_benchmark_staircase.png)

### 3. Analytical & Single-Table Equations of State (`Ideal`, `HELM`, `PC`, `Skye`)

![Four-way runtime staircase for Ideal, HELM, PC, and Skye equations of state](../assets/figures/wave3a_benchmark_staircase.png)

### 4. 4D Tabulated EOS, Opacities, Reaction Networks & Model Atmospheres

![Four-way runtime staircase for 4D EOS, opacity, reaction rates, networks, and atmospheres](../assets/figures/wave3b_benchmark_staircase.png)

### 5. 1D Stellar Structure, Henyey Solver & Coupled Evolution Step

![Four-way runtime staircase for 1D stellar structure, Henyey Newton step, remeshing, and coupled evolution](../assets/figures/wave4_benchmark_staircase.png)

### 6. Atomic Diffusion, Shellular Rotation, HLLC Hydrodynamics, Binaries & Asteroseismology

![Four-way runtime comparison and zone-count scaling for diffusion, rotation, HLLC/RSP2, binary RLOF, and asteroseismology](../assets/figures/wave5_benchmark_speedup.png)

---

## Reproducing the Benchmarks Locally

You can run any domain benchmark suite from Python via `flexmesa.bench`, or execute the standalone Safe Rust (`cargo test --release -p flexmesa-bench`) and Fortran MESA (`make -C benchmarks/fortran_mesa run`) harnesses:

```python
from flexmesa import bench

# Run a quick 128-zone microphysics timing and checksum pass (1 warmup, 2 timed reps)
records = bench.run_all_wave1_benchmarks(
    zone_counts=(128,), n_warmup=1, n_repeats=2
)
for rec in records[:3]:
    print(
        f"{rec.workload:12s} n_zones={rec.n_zones:4d} "
        f"stage0={rec.stage0.median_us:8.2f} us checksum={rec.checksum:.8e}"
    )
```

For complete kernel-by-kernel latency tables across all four optimization stages, see **[Kernel Performance Scorecards](kernel-scorecards.md)**. For the three-tier numerical verification hierarchy and documented upstream Fortran MESA discrepancies, see **[Tolerance & Discrepancy Ledger](tolerance-and-discrepancy-ledger.md)**.
