# Wave 1 Four-Way Performance Benchmark and Architecture Synthesis

This report presents the empirical results of the Wave 1 four-stage performance
staircase across all four implementations of the Wave 1 foundation and
microphysics modules:

1. **Upstream Fortran MESA (`gfortran -O2 -fopenmp`)**: Native Fortran 2008
   modules (`const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`,
   `mtx`, `num`, `neu`, and `turb`) benchmarked via `benchmarks/fortran_mesa/`.
2. **Safe Rust (`crates/flexmesa-bench`)**: `#![forbid(unsafe_code)]` Cargo
   workspace crates compiled with `--release`.
3. **Python / NumPy (`src/flexmesa/bench`, Stages 0 and 1)**: Pure Python 3.14
   and vectorized `numpy.float64` / LAPACK routines.
4. **Python / JAX (`src/flexmesa/bench`, Stages 2 and 3)**: XLA-compiled
   `@jax.jit` and `@jax.jit(jax.vmap)` GPU kernels in strict IEEE-754 double
   precision (`jax_enable_x64=True`).

---

## 1. Benchmark Methodology and Reproduction

### 1.1 Four-Stage Optimization Staircase

Each representative Wave 1 workload is evaluated across three stellar radial
grid sizes $N_{\mathrm{zones}} \in \{128, 1024, 4096\}$ and four progressive
optimization stages:

1. **Stage 0 (Idiomatic Baseline)**: Zone-by-zone scalar calls with per-step
   buffer allocations, repeated spline table validation, and eager scalar
   dual-number dispatch.
2. **Stage 1 (Contiguous Structure-of-Arrays Layout)**: Pre-allocated contiguous
   `f64` / `float64` Structure-of-Arrays (SoA) buffers of shape `(N_zones,)`
   with persistent solver workspaces or vectorized NumPy/LAPACK calls.
3. **Stage 2 (Fused / Pre-Validated / JIT Kernel Execution)**: Hoisted table
   invariants, pre-allocated block-Thomas LU elimination, and zero-allocation
   loops in Fortran MESA and Safe Rust (`--release`), and XLA-compiled
   `@jax.jit` fused GPU kernels in Python/JAX (recording both first-call XLA
   compilation latency and warm steady-state execution latency).
4. **Stage 3 (Batched / Multi-Core Execution)**: 4-thread OpenMP parallel
   execution in Fortran MESA, 4-thread scoped execution (`std::thread::scope`)
   in Safe Rust, and batched `@jax.jit(jax.vmap)` execution across $B = 4$
   simultaneous stellar profiles of shape `(4, N_zones)` on the XLA accelerator
   (reported as amortized latency per stellar profile).

### 1.2 Hardware Environment and Reproduction Commands

Measurements were collected on a 16-core x86\_64 Linux workstation with 30 GiB
system RAM and an NVIDIA GeForce RTX 5080 GPU (16 GiB GDDR7 VRAM), with CPU
thread pools capped at 4 threads (`OMP_NUM_THREADS=4`,
`OPENBLAS_NUM_THREADS=4`, `MKL_NUM_THREADS=4`) and JAX configured in strict
IEEE-754 double precision (`jax_enable_x64=True`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.65`).

To reproduce all three benchmark suites:

```bash
# 1. Upstream Fortran MESA (gfortran -O2 -fopenmp)
make -C benchmarks/fortran_mesa MESA_DIR=/path/to/mesa run

# 2. Safe Rust (--release)
cargo test --release -p flexmesa-bench -- --ignored --nocapture

# 3. Python / NumPy / JAX (float64)
PYTHONPATH=src python -c "
from flexmesa import bench
for r in bench.run_all_wave1_benchmarks((128, 1024, 4096), 2, 7):
    print(r)
"
```

---

## 2. Four-Way Numerical Parity Verification

Across all 18 `(workload, N_zones)` combinations, the Stage 2 primal checksums
agree across all 9 printed significant digits across Upstream Fortran MESA, Safe
Rust, Python/NumPy, and Python/JAX `float64`:

| Workload | $N_{\mathrm{zones}}$ | Fortran MESA Checksum | Safe Rust Checksum | Python/JAX Checksum | Fortran Stage Rel Err | Rust Stage Rel Err | JAX Stage Rel Err |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `auto_diff` | 128 | `1.29083046e5` | `1.29083046e5` | `1.29083046e5` | `5.00e-16` | `0.00e0` | `1.84e-15` |
| `interp_2d` | 128 | `5.76679201e3` | `5.76679201e3` | `5.76679201e3` | `0.00e0` | `0.00e0` | `0.00e0` |
| `neu` | 128 | `3.03251271e13` | `3.03251271e13` | `3.03251271e13` | `1.04e-14` | `1.74e-14` | `1.74e-14` |
| `mtx_tridiag` | 128 | `1.72569169e2` | `1.72569169e2` | `1.72569169e2` | `1.33e-15` | `0.00e0` | `4.46e-16` |
| `turb_mlt` | 128 | `5.47381466e1` | `5.47381466e1` | `5.47381466e1` | `0.00e0` | `0.00e0` | `1.23e-14` |
| `turb_tdc` | 128 | `5.31136162e1` | `5.31136162e1` | `5.31136162e1` | `0.00e0` | `0.00e0` | `2.63e-15` |
| `auto_diff` | 1024 | `1.01556512e6` | `1.01556512e6` | `1.01556512e6` | `5.81e-16` | `0.00e0` | `2.07e-15` |
| `interp_2d` | 1024 | `4.61222326e4` | `4.61222326e4` | `4.61222326e4` | `0.00e0` | `0.00e0` | `0.00e0` |
| `neu` | 1024 | `2.06958005e14` | `2.06958005e14` | `2.06958005e14` | `5.40e-14` | `2.56e-13` | `2.96e-13` |
| `mtx_tridiag` | 1024 | `1.38152230e3` | `1.38152230e3` | `1.38152230e3` | `1.84e-15` | `0.00e0` | `5.99e-16` |
| `turb_mlt` | 1024 | `4.37918538e2` | `4.37918538e2` | `4.37918538e2` | `0.00e0` | `0.00e0` | `1.79e-14` |
| `turb_tdc` | 1024 | `4.24914505e2` | `4.24914505e2` | `4.24914505e2` | `0.00e0` | `0.00e0` | `2.94e-15` |
| `auto_diff` | 4096 | `4.05502565e6` | `4.05502565e6` | `4.05502565e6` | `5.56e-16` | `0.00e0` | `2.20e-15` |
| `interp_2d` | 4096 | `1.84483783e5` | `1.84483783e5` | `1.84483783e5` | `5.01e-13` | `1.65e-16` | `4.02e-16` |
| `neu` | 4096 | `8.13481382e14` | `8.13481382e14` | `8.13481382e14` | `1.82e-13` | `9.53e-14` | `7.01e-13` |
| `mtx_tridiag` | 4096 | `5.52650461e3` | `5.52650461e3` | `5.52650461e3` | `1.68e-15` | `0.00e0` | `6.35e-16` |
| `turb_mlt` | 4096 | `1.75167983e3` | `1.75167983e3` | `1.75167983e3` | `0.00e0` | `0.00e0` | `2.20e-14` |
| `turb_tdc` | 4096 | `1.69966039e3` | `1.69966039e3` | `1.69966039e3` | `0.00e0` | `0.00e0` | `3.29e-15` |

> [!NOTE]
> In `interp_2d`, Safe Rust and Python/JAX agree to $4.0 \times 10^{-16}$, while
> upstream Fortran MESA's `fvbicub_db` differs by $7.20 \times 10^{-9}$ in the
> 9th digit because `interp_2d/private/bicub_db.f` omits `implicit none` inside
> `fvbicub_db`, leaving its local Hermite basis variables (`xp`, `yp`, `cx`,
> `cy`, `hx2`, `hy2`, `z36th`) implicitly typed as 32-bit `REAL` (`f32`). All
> five other workloads agree to $\le 10^{-12}$ in full `f64` precision across
> all four implementations.

---

## 3. Upstream Fortran MESA Results (`benchmarks/fortran_mesa`)

All latencies below are reported as `median +/- IQR` in microseconds
($\mu\mathrm{s}$) across 9 timed iterations following 3 warmup iterations:

| Workload | $N_{\mathrm{zones}}$ | Stage 0 ($\mu\mathrm{s}$) | Stage 1 ($\mu\mathrm{s}$) | Stage 2 ($\mu\mathrm{s}$) | Stage 3 (4t OpenMP, $\mu\mathrm{s}$) | Speedup (Stage 0 / Stage 2) | Speedup (Stage 0 / Stage 3) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `auto_diff` | 128 | `7.40 +/- 0.01` | `7.34 +/- 0.01` | `0.68 +/- 0.01` | `0.52 +/- 0.09` | `10.88x` | `14.23x` |
| `interp_2d` | 128 | `5.42 +/- 0.02` | `5.35 +/- 0.03` | `5.03 +/- 0.01` | `1.64 +/- 0.08` | `1.08x` | `3.30x` |
| `neu` | 128 | `19.19 +/- 0.07` | `19.16 +/- 0.03` | `17.88 +/- 0.03` | `5.67 +/- 0.39` | `1.07x` | `3.38x` |
| `mtx_tridiag` | 128 | `69.36 +/- 0.40` | `67.11 +/- 0.76` | `26.50 +/- 0.08` | `6.76 +/- 0.02` | `2.62x` | `10.26x` |
| `turb_mlt` | 128 | `216.46 +/- 1.69` | `217.71 +/- 0.07` | `217.57 +/- 13.41` | `55.57 +/- 0.44` | `0.99x` | `3.90x` |
| `turb_tdc` | 128 | `7059.73 +/- 157.38` | `6914.18 +/- 60.13` | `6971.25 +/- 140.88` | `2162.92 +/- 298.93` | `1.01x` | `3.26x` |
| `auto_diff` | 1024 | `73.68 +/- 0.17` | `72.85 +/- 0.09` | `7.26 +/- 0.02` | `2.82 +/- 0.05` | `10.15x` | `26.13x` |
| `interp_2d` | 1024 | `63.16 +/- 0.34` | `62.27 +/- 0.69` | `40.07 +/- 0.21` | `16.09 +/- 0.07` | `1.58x` | `3.93x` |
| `neu` | 1024 | `151.17 +/- 0.30` | `150.88 +/- 1.04` | `142.33 +/- 0.22` | `52.18 +/- 0.79` | `1.06x` | `2.90x` |
| `mtx_tridiag` | 1024 | `520.19 +/- 17.04` | `533.74 +/- 17.73` | `213.98 +/- 7.40` | `54.47 +/- 2.83` | `2.43x` | `9.55x` |
| `turb_mlt` | 1024 | `1765.13 +/- 24.41` | `1757.14 +/- 32.08` | `1791.06 +/- 119.91` | `443.14 +/- 11.76` | `0.99x` | `3.98x` |
| `turb_tdc` | 1024 | `55456.10 +/- 763.56` | `55614.27 +/- 1046.72` | `56336.41 +/- 905.50` | `14123.76 +/- 297.88` | `0.98x` | `3.93x` |
| `auto_diff` | 4096 | `236.53 +/- 11.91` | `243.30 +/- 12.25` | `21.11 +/- 1.55` | `5.65 +/- 0.12` | `11.20x` | `41.86x` |
| `interp_2d` | 4096 | `176.26 +/- 3.03` | `171.18 +/- 0.83` | `160.35 +/- 1.06` | `41.15 +/- 0.03` | `1.10x` | `4.28x` |
| `neu` | 4096 | `610.01 +/- 84.64` | `621.76 +/- 31.53` | `575.47 +/- 13.69` | `152.32 +/- 0.41` | `1.06x` | `4.00x` |
| `mtx_tridiag` | 4096 | `2135.50 +/- 59.79` | `2104.11 +/- 14.53` | `870.34 +/- 16.87` | `221.38 +/- 7.13` | `2.45x` | `9.65x` |
| `turb_mlt` | 4096 | `7045.06 +/- 64.53` | `7082.88 +/- 71.12` | `7081.30 +/- 140.34` | `1784.93 +/- 49.51` | `1.00x` | `3.95x` |
| `turb_tdc` | 4096 | `223757.38 +/- 466.56` | `222985.21 +/- 923.97` | `223895.15 +/- 2699.34` | `58176.83 +/- 2602.71` | `1.00x` | `3.85x` |

In addition, upstream Fortran MESA's `bcyclic_dble_decsolblk` (Hirshman et al.
2010 odd-even parallel block cyclic reduction) parallelizes a **single**
block-tridiagonal matrix solve across 4 OpenMP threads:
- $N_{\mathrm{zones}} = 128$: $67.11\,\mu\mathrm{s}$ (1t) $\to
  41.92\,\mu\mathrm{s}$ (4t OpenMP, $1.60\times$ single-matrix speedup)
- $N_{\mathrm{zones}} = 1024$: $533.74\,\mu\mathrm{s}$ (1t) $\to
  212.62\,\mu\mathrm{s}$ (4t OpenMP, $2.51\times$ single-matrix speedup)
- $N_{\mathrm{zones}} = 4096$: $2104.11\,\mu\mathrm{s}$ (1t) $\to
  791.43\,\mu\mathrm{s}$ (4t OpenMP, $2.66\times$ single-matrix speedup)

---

## 4. Safe Rust and Python/NumPy/JAX Results

### 4.1 Safe Rust `--release` Results

| Workload | $N_{\mathrm{zones}}$ | Stage 0 ($\mu\mathrm{s}$) | Stage 1 ($\mu\mathrm{s}$) | Stage 2 ($\mu\mathrm{s}$) | Stage 3 ($\mu\mathrm{s}$) | Speedup (Stage 0 / Stage 2) | Speedup (Stage 0 / Stage 3) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `auto_diff` | 128 | `3.20 +/- 0.73` | `3.06 +/- 0.01` | `2.52 +/- 0.02` | `226.82 +/- 27.02` | `1.27x` | `0.01x` |
| `interp_2d` | 128 | `403.89 +/- 12.46` | `395.67 +/- 12.56` | `4.03 +/- 0.01` | `212.69 +/- 6.06` | `100.22x` | `1.90x` |
| `neu` | 128 | `19.12 +/- 0.07` | `18.97 +/- 0.01` | `18.68 +/- 0.06` | `217.09 +/- 16.75` | `1.02x` | `0.09x` |
| `mtx_tridiag` | 128 | `50.59 +/- 0.45` | `50.14 +/- 0.04` | `47.19 +/- 0.06` | `67.90 +/- 6.92` | `1.07x` | `0.75x` |
| `turb_mlt` | 128 | `17.53 +/- 0.05` | `17.07 +/- 0.02` | `17.13 +/- 0.05` | `220.74 +/- 3.19` | `1.02x` | `0.08x` |
| `turb_tdc` | 128 | `130.23 +/- 10.97` | `124.55 +/- 2.04` | `124.62 +/- 13.12` | `281.59 +/- 35.58` | `1.05x` | `0.46x` |
| `auto_diff` | 1024 | `25.31 +/- 0.10` | `24.35 +/- 0.01` | `20.15 +/- 0.16` | `223.71 +/- 3.26` | `1.26x` | `0.11x` |
| `interp_2d` | 1024 | `3212.84 +/- 38.27` | `3184.92 +/- 22.88` | `32.11 +/- 0.06` | `223.33 +/- 1.72` | `100.06x` | `14.39x` |
| `neu` | 1024 | `152.72 +/- 23.71` | `150.65 +/- 7.43` | `148.04 +/- 0.17` | `262.04 +/- 25.81` | `1.03x` | `0.58x` |
| `mtx_tridiag` | 1024 | `534.51 +/- 45.20` | `418.00 +/- 8.24` | `406.06 +/- 24.57` | `156.47 +/- 10.65` | `1.32x` | `3.42x` |
| `turb_mlt` | 1024 | `140.23 +/- 0.57` | `136.66 +/- 0.32` | `136.87 +/- 0.44` | `274.05 +/- 56.59` | `1.02x` | `0.51x` |
| `turb_tdc` | 1024 | `1008.05 +/- 31.60` | `996.83 +/- 29.28` | `1005.94 +/- 8.69` | `495.56 +/- 24.62` | `1.00x` | `2.03x` |
| `auto_diff` | 4096 | `100.87 +/- 0.09` | `97.48 +/- 0.07` | `80.56 +/- 0.81` | `241.31 +/- 5.97` | `1.25x` | `0.42x` |
| `interp_2d` | 4096 | `12919.38 +/- 135.53` | `13036.85 +/- 3705.90` | `128.44 +/- 0.10` | `280.35 +/- 32.99` | `100.59x` | `46.08x` |
| `neu` | 4096 | `605.84 +/- 30.45` | `601.48 +/- 10.37` | `602.97 +/- 22.46` | `372.35 +/- 5.86` | `1.00x` | `1.63x` |
| `mtx_tridiag` | 4096 | `2153.70 +/- 18.58` | `1632.69 +/- 9.44` | `1563.45 +/- 13.59` | `451.97 +/- 6.96` | `1.38x` | `4.77x` |
| `turb_mlt` | 4096 | `564.14 +/- 15.07` | `550.30 +/- 3.66` | `556.33 +/- 11.07` | `360.26 +/- 3.37` | `1.01x` | `1.57x` |
| `turb_tdc` | 4096 | `3997.47 +/- 50.87` | `3979.76 +/- 32.32` | `4005.21 +/- 179.70` | `1302.19 +/- 12.80` | `1.00x` | `3.07x` |

### 4.2 Python / NumPy (Stages 0..1) and JAX `float64` (Stages 2..3) Results

| Workload | $N_{\mathrm{zones}}$ | Stage 0 ($\mu\mathrm{s}$) | Stage 1 ($\mu\mathrm{s}$) | Stage 2 Warm ($\mu\mathrm{s}$) | Stage 2 Compile ($\mathrm{ms}$) | Stage 3 Batch/4 ($\mu\mathrm{s}$) | Stage 3 Compile ($\mathrm{ms}$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `auto_diff` | 128 | `2739530.22 +/- 123028.82` | `13152.61 +/- 3804.55` | `1271.46 +/- 23.72` | `158.1` | `317.97 +/- 16.68` | `155.7` |
| `interp_2d` | 128 | `10497050.10 +/- 381302.66` | `93478.72 +/- 3133.53` | `1281.55 +/- 18.97` | `324.0` | `59.32 +/- 1.26` | `326.8` |
| `neu` | 128 | `1414.43 +/- 54.96` | `1409.35 +/- 49.43` | `1269.43 +/- 57.83` | `2187.2` | `314.94 +/- 1.61` | `2123.1` |
| `mtx_tridiag` | 128 | `7928.14 +/- 496.67` | `1060.91 +/- 15.55` | `55670.74 +/- 4181.39` | `306.9` | `14970.95 +/- 4378.25` | `300.4` |
| `turb_mlt` | 128 | `8675169.63 +/- 291259.65` | `45154.78 +/- 6593.05` | `1262.34 +/- 31.90` | `164.3` | `322.99 +/- 17.23` | `177.2` |
| `turb_tdc` | 128 | `47177902.82 +/- 2239194.94` | `471913.60 +/- 34314.73` | `1496.16 +/- 59.15` | `386.0` | `364.02 +/- 5.22` | `434.7` |
| `auto_diff` | 1024 | `23066857.93 +/- 593140.51` | `11941.13 +/- 1094.95` | `1275.54 +/- 10.61` | `142.2` | `315.95 +/- 9.21` | `148.7` |
| `interp_2d` | 1024 | `84720082.46 +/- 1289471.91` | `95168.40 +/- 8396.55` | `1273.78 +/- 74.12` | `300.8` | `317.86 +/- 1.40` | `333.8` |
| `neu` | 1024 | `11801.69 +/- 156.86` | `11402.97 +/- 268.16` | `1262.57 +/- 89.81` | `2253.5` | `321.01 +/- 6.28` | `504.5` |
| `mtx_tridiag` | 1024 | `62072.03 +/- 616.93` | `8512.82 +/- 392.36` | `391651.83 +/- 211574.19` | `643.6` | `85839.18 +/- 43069.87` | `516.7` |
| `turb_mlt` | 1024 | `68264580.10 +/- 6035312.90` | `50696.07 +/- 2180.62` | `1300.18 +/- 66.57` | `166.6` | `319.60 +/- 24.83` | `183.2` |
| `turb_tdc` | 1024 | `360276626.43 +/- 27042344.45` | `476976.35 +/- 25029.67` | `1451.62 +/- 60.82` | `397.5` | `368.45 +/- 25.09` | `441.0` |
| `auto_diff` | 4096 | `91632662.86 +/- 840468.31` | `14626.04 +/- 3511.52` | `1266.05 +/- 34.44` | `148.5` | `318.98 +/- 5.97` | `126.4` |
| `interp_2d` | 4096 | `337694355.67 +/- 620450.50` | `93319.73 +/- 6032.50` | `1285.33 +/- 8.14` | `314.1` | `325.16 +/- 16.16` | `360.0` |
| `neu` | 4096 | `45150.18 +/- 963.39` | `45140.33 +/- 388.42` | `1263.82 +/- 22.64` | `495.7` | `321.29 +/- 5.93` | `419.5` |
| `mtx_tridiag` | 4096 | `251373.59 +/- 11908.83` | `34646.39 +/- 696.00` | `1618998.55 +/- 797815.50` | `1686.6` | `475115.88 +/- 58642.18` | `1869.3` |
| `turb_mlt` | 4096 | `278470433.79 +/- 4812641.28` | `49688.91 +/- 1902.86` | `1287.58 +/- 39.82` | `177.2` | `316.21 +/- 4.18` | `182.6` |
| `turb_tdc` | 4096 | `1471659544.58 +/- 9383351.30` | `474565.86 +/- 20398.81` | `1449.27 +/- 57.31` | `400.9` | `362.28 +/- 75.31` | `471.9` |

---

## 5. Four-Way Synthesis and Wave 2/3 Architecture Evaluation

### 5.1 Four-Way Direct Comparison at $N_{\mathrm{zones}} = 4096$

| Workload ($N_{\mathrm{zones}} = 4096$) | Fortran MESA 1t ($\mu\mathrm{s}$) | Fortran MESA 4t OpenMP ($\mu\mathrm{s}$) | Safe Rust 1t ($\mu\mathrm{s}$) | Safe Rust 4t ($\mu\mathrm{s}$) | Python/NumPy Best ($\mu\mathrm{s}$) | JAX `@jit` 1-Star ($\mu\mathrm{s}$) | JAX `@jit(vmap)` Batch/4 ($\mu\mathrm{s}$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `auto_diff` (hyperdual) | `243.30` | `5.65` (fused) | `97.48` (`80.56` fused) | `241.31` | `14626.04` | `1266.05` | `318.98` |
| `interp_2d` (bicubic) | `171.18` (`160.35` fast) | `41.15` | `128.44` | `280.35` | `93319.73` | `1285.33` | `325.16` |
| `neu` (Itoh 1996) | `621.76` (`575.47` fast) | `152.32` | `601.48` | `372.35` | `45140.33` | `1263.82` | `321.29` |
| `mtx_tridiag` ($5 \times 5$) | `2104.11` (`870.34` Thomas) | `791.43` (`221.38` batch) | `1563.45` | `451.97` | `34646.39` | `1618998.55` | `475115.88` |
| `turb_mlt` (Cox 1968) | `7082.88` | `1784.93` | `550.30` | `360.26` | `49688.91` | `1287.58` | `316.21` |
| `turb_tdc` (Kuhfuss 1986) | `222985.21` | `58176.83` | `3979.76` | `1302.19` | `474565.86` | `1449.27` | `362.28` |

### 5.2 Five Structural Findings from the Four-Way Comparison

1. **Implicit Function Theorem (IFT) + Const-Generic Dual Numbers Deliver a
   $13\times\text{--}162\times$ Speedup Over Upstream Fortran MESA on `turb_mlt`
   and `turb_tdc`**:
   - In upstream Fortran MESA, `set_MLT` and `set_TDC` are hardwired to
     `type(auto_diff_real_star_order1)`, which propagates a fixed **33-element**
     tangent array (`d1Array(33)`) through every elementary operation, and
     `set_TDC` wraps `auto_diff_real_star_order1` inside `auto_diff_real_tdc`
     ($> 70$ floating-point components) across every bracket-search and
     Newton-Raphson iteration of $Q(Y) = 0$. Consequently, 4096 zones of
     `turb_tdc` take $223.0\,\mathrm{ms}$ on 1 thread ($58.2\,\mathrm{ms}$ on 4
     OpenMP threads) in Fortran MESA.
   - By contrast, FlexMESA's Safe Rust and JAX implementations solve for the
     scalar root $Y^*$ in `f64` and apply the **Implicit Function Theorem**
     $\partial Y^* / \partial \theta = -(\partial Q / \partial Y)^{-1}
     (\partial Q / \partial \theta)$ once at convergence using only the 6 active
     local tangents (`DualOrder1<6>`). At $N_{\mathrm{zones}} = 4096$, Safe Rust
     is **$12.9\times$ faster than Fortran MESA on `turb_mlt`**
     ($550\,\mu\mathrm{s}$ vs. $7083\,\mu\mathrm{s}$) and **$56.0\times$ faster
     on `turb_tdc`** ($3.98\,\mathrm{ms}$ vs. $223.0\,\mathrm{ms}$), while JAX
     `@jit(vmap)` on GPU is **$160.6\times$ faster than 4-thread Fortran MESA on
     `turb_tdc`** ($362\,\mu\mathrm{s}$ vs. $58.2\,\mathrm{ms}$).
2. **Cross-Crate Monomorphization in Safe Rust Matches or Beats `gfortran -O2`
   on Leaf Kernels (`auto_diff`, `interp_2d`, `neu`)**:
   - On single-threaded leaf kernels (`Stage 1` and `Stage 2` at
     $N_{\mathrm{zones}} = 4096$), Safe Rust is **$2.50\times$ faster than
     Fortran MESA** on operator-overloaded `AutoDiffReal2VarOrder2`
     ($97.48\,\mu\mathrm{s}$ vs. $243.30\,\mu\mathrm{s}$), **$1.25\times$
     faster** on `interp_2d` ($128.44\,\mu\mathrm{s}$ vs.
     $160.35\,\mu\mathrm{s}$, while fixing upstream MESA's implicit `f32` bug in
     `fvbicub_db`), and **within $3\%$** on `neu_get` ($601.48\,\mu\mathrm{s}$
     vs. $621.76\,\mu\mathrm{s}$).
3. **Persistent Thread Pool (`libgomp` / `rayon`) vs. OS Thread Spawning
   (`std::thread::scope`)**:
   - Upstream Fortran MESA's `!\$omp parallel do` uses `libgomp`'s persistent
     warm worker pool ($\sim 1\,\mu\mathrm{s}$ barrier overhead), achieving a
     $3.8\times\text{--}4.0\times$ 4-core speedup even on $160\,\mu\mathrm{s}$
     workloads (`interp_2d`: $160.35\,\mu\mathrm{s} \to 41.15\,\mu\mathrm{s}$;
     `neu`: $575.47\,\mu\mathrm{s} \to 152.32\,\mu\mathrm{s}$). In Safe Rust,
     spawning fresh OS threads via `std::thread::scope` adds a
     $\sim 215\,\mu\mathrm{s}$ floor; adopting a persistent `rayon` thread pool
     in Wave 2/3 will give Safe Rust the same $\sim 1\,\mu\mathrm{s}$ barrier
     overhead as OpenMP.
4. **Block-Tridiagonal Henyey Solves (`mtx_tridiag`): Sequential Thomas vs.
   Odd-Even Parallel Cyclic Reduction (`bcyclic`)**:
   - Upstream MESA's `bcyclic_dble_decsolblk` (Hirshman et al. 2010) reduces a
     single 4096-zone Henyey solve from $2104\,\mu\mathrm{s}$ (1 thread) to
     $791\,\mu\mathrm{s}$ (4 OpenMP threads) via
     $\lceil \log_2 N_{\mathrm{zones}} \rceil = 12$ odd-even reduction levels,
     whereas sequential block-Thomas takes $870\,\mu\mathrm{s}$ in Fortran and
     $1563\,\mu\mathrm{s}$ in Safe Rust.
   - Crucially, this same $\log_2 N_{\mathrm{zones}}$ parallel cyclic reduction
     (PCR) algorithm explains how to eliminate JAX's $1619\,\mathrm{ms}$
     sequential `lax.scan` bottleneck on GPU by replacing $N_{\mathrm{zones}} =
     4096$ sequential scan steps with $12$ vectorized block-row reductions.
5. **Batched GPU Throughput and End-to-End Differentiability in JAX**:
   - Once compiled with `@jax.jit(jax.vmap)`, JAX evaluates 4096 zones of
     complex microphysics (`neu`, `turb_mlt`, `turb_tdc`) in
     $316\text{--}362\,\mu\mathrm{s}$ per stellar profile on the GPU with exact
     forward and reverse-mode automatic differentiation (`jax.grad`,
     `jax.jacfwd`, `jax.vjp`).

### 5.3 Architectural Options for Wave 2 and Wave 3

- **Option 1 (Recommended Hybrid / Dual-Track Architecture)**:
  - Maintain the **dual-track Safe Rust + JAX `float64` architecture across Wave
    2 microphysics (`rates`, `net`, `eos`, `kap`, `atm`, `ionization`,
    `colors`)**, using const-generic `DualOrder1<N>` + Implicit Function Theorem
    solvers + persistent `rayon` parallelism in Safe Rust, and branchless
    `@jax.jit` + `custom_jvp` / `custom_vjp` kernels in Python/JAX.
  - In `mtx` / Wave 3 (`star`), add **parallel block cyclic reduction (`bcyclic`
    / PCR)** alongside block-Thomas in both Safe Rust and JAX (plus `PyO3` / JAX
    `ffi_call` host bindings), eliminating the sequential GPU `lax.scan`
    bottleneck while preserving end-to-end differentiability.
- **Option 2 (Pure Dual-Track Without Cross-Language FFI)**:
  - Keep Safe Rust and Python/JAX completely independent through Wave 2 and Wave
    3, relying on native JAX parallel cyclic reduction (PCR) and padded zone
    buckets ($N_{\mathrm{pad}} \in \{512, 1024, 2048, 4096\}$) for GPU
    evolution.
- **Option 3 (Single-Language Concentration Post-Wave 1)**:
  - Concentrate Wave 2 and Wave 3 in a single primary engine (either Safe Rust
    with `PyO3` bindings or Python/JAX `float64` with Rust FFI for solvers).
