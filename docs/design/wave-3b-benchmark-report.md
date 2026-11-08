# Wave 3b.5 Four-Way Performance Benchmark and Consolidation Report

This report documents the empirical results of the Wave 3b.5 (`#61`) four-stage
performance staircase and cross-language parity verification across all four
implementations of the Wave 3b stellar microphysics modules (`eos_4d`,
`kap_fixed`, `rates_screen`, `net_eval`, and `atm_table`):

1. **Upstream Fortran MESA (`gfortran -O2 -fopenmp`)**: Native Fortran 2008
   modules (`libeos.a`, `libkap.a`, `librates.a`, `libnet.a`, `libatm.a`,
   `libchem.a`, `libinterp_2d.a`, `libinterp_1d.a`, `libauto_diff.a`) benchmarked
   via `benchmarks/fortran_mesa/run_wave3b_bench`.
2. **Safe Rust (`crates/flexmesa-bench`)**: `#![forbid(unsafe_code)]` Cargo
   workspace crates (`flexmesa-eos`, `flexmesa-kap`, `flexmesa-rates`,
   `flexmesa-net`, `flexmesa-atm`) compiled with `--release` and persistent
   4-thread `rayon` parallelism.
3. **Python / PyO3 (`src/flexmesa/bench`, Stages 0 and 1)**: Zero-copy C-order
   `numpy.float64` bindings (`flexmesa._rust`) dispatching to 1-thread (Stage 0)
   and 4-thread adaptive spin-then-park (Stage 1) Safe Rust batch kernels.
4. **Python / JAX (`src/flexmesa/bench`, Stages 2 and 3)**: XLA-compiled
   `@jax.jit` and `@jax.jit(jax.vmap)` kernels in strict IEEE-754 double
   precision (`jax_enable_x64=True`).

---

## 1. Cross-Language Numerical Parity Gate

Across all five Wave 3b workloads (`eos_4d`, `kap_fixed`, `rates_screen`,
`net_eval`, and `atm_table`) and all radial grid sizes $N_{\mathrm{zones}} \in
\{8, 128, 1024, 4096\}$, the reduction checksums match across Upstream Fortran
MESA, Safe Rust, Python (`flexmesa._rust`), and JAX (`float64`):

| Workload | Physical Reduction | $N_{\mathrm{zones}} = 8$ Checksum | $N_{\mathrm{zones}} = 128$ Checksum | $N_{\mathrm{zones}} = 1024$ Checksum | $N_{\mathrm{zones}} = 4096$ Checksum | Max Relative Error vs. Fortran |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `eos_4d` | $\sum_k \ln P_{\mathrm{gas}, k}$ | `2.03736456e+02` | `3.25978346e+03` | `2.60782677e+04` | `1.04313071e+05` | `< 4.65e-15` |
| `kap_fixed` | $\sum_k \kappa_k$ | `2.90919576e+00` | `4.65455295e+01` | `3.72363584e+02` | `1.48945406e+03` | `< 1.00e-14` |
| `rates_screen` | $\sum_k \log_{10}(\lambda_k S_k)$ | `-1.13918891e+02` | `-1.81971803e+03` | `-1.45565240e+04` | `-5.82255771e+04` | `< 1.05e-13` |
| `net_eval` | $\sum_k \log_{10}\epsilon_{\mathrm{nuc}, k}$ | `9.61964985e+01` | `1.53917124e+03` | `1.23133811e+04` | `4.92535290e+04` | `< 1.00e-14` |
| `atm_table` | $\sum_k \ln P_{\mathrm{surf}, k}$ | `8.98541118e+01` | `1.43822020e+03` | `1.15059988e+04` | `4.60240965e+04` | `< 8.45e-13` |

![Four-way Wave 3b.5 runtime comparison and model atmosphere boundary pressure profiles](../assets/figures/wave3b_benchmark_staircase.png)

---

## 2. Four-Way Runtime Tables ($N_{\mathrm{zones}} \in \{128, 1024, 4096\}$)

### 2.1 Upstream Fortran MESA (`gfortran -O2 -fopenmp`)

| Workload | $N_{\mathrm{zones}}$ | Stage 0 ($\mu\mathrm{s}$) | Stage 1 ($\mu\mathrm{s}$) | Stage 2 ($\mu\mathrm{s}$) | Stage 3 (4t OpenMP, $\mu\mathrm{s}$) | Speedup (Stage 0 / Stage 2) | Speedup (Stage 0 / Stage 3) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eos_4d` | 128 | `139.11 +/- 8.58` | `136.84 +/- 0.27` | `124.25 +/- 0.25` | `31.94 +/- 0.39` | `1.12x` | `4.36x` |
| `kap_fixed` | 128 | `67.37 +/- 0.39` | `67.44 +/- 0.33` | `71.73 +/- 0.56` | `19.04 +/- 0.07` | `0.94x` | `3.54x` |
| `rates_screen` | 128 | `42.93 +/- 1.20` | `28.11 +/- 0.04` | `19.81 +/- 0.04` | `5.38 +/- 0.09` | `2.17x` | `7.98x` |
| `net_eval` | 128 | `908.59 +/- 40.41` | `905.93 +/- 24.96` | `910.89 +/- 26.76` | `227.10 +/- 1.73` | `1.00x` | `4.00x` |
| `atm_table` | 128 | `40.50 +/- 0.18` | `40.45 +/- 0.17` | `40.42 +/- 0.26` | `10.97 +/- 0.62` | `1.00x` | `3.69x` |
| `eos_4d` | 1024 | `1148.30 +/- 34.54` | `1121.00 +/- 28.96` | `1011.23 +/- 7.32` | `252.09 +/- 11.71` | `1.14x` | `4.56x` |
| `kap_fixed` | 1024 | `559.21 +/- 18.74` | `550.81 +/- 10.40` | `451.12 +/- 23.09` | `111.27 +/- 10.51` | `1.24x` | `5.03x` |
| `rates_screen` | 1024 | `225.30 +/- 15.35` | `344.01 +/- 21.23` | `158.06 +/- 0.34` | `40.85 +/- 9.76` | `1.43x` | `5.52x` |
| `net_eval` | 1024 | `7447.43 +/- 239.82` | `7299.62 +/- 111.39` | `7417.57 +/- 155.18` | `1878.05 +/- 97.43` | `1.00x` | `3.97x` |
| `atm_table` | 1024 | `588.08 +/- 11.78` | `566.46 +/- 44.96` | `567.00 +/- 157.34` | `175.32 +/- 14.27` | `1.04x` | `3.35x` |
| `eos_4d` | 4096 | `4603.91 +/- 165.08` | `4644.33 +/- 152.84` | `4068.05 +/- 38.62` | `1451.55 +/- 552.69` | `1.13x` | `3.17x` |
| `kap_fixed` | 4096 | `2192.46 +/- 36.85` | `2199.66 +/- 58.13` | `1789.49 +/- 27.08` | `452.32 +/- 5.25` | `1.23x` | `4.85x` |
| `rates_screen` | 4096 | `910.66 +/- 14.62` | `913.83 +/- 22.61` | `636.49 +/- 10.21` | `271.28 +/- 91.20` | `1.43x` | `3.36x` |
| `net_eval` | 4096 | `28976.79 +/- 179.55` | `29405.56 +/- 748.28` | `29591.90 +/- 4526.77` | `10185.83 +/- 1614.09` | `0.98x` | `2.84x` |
| `atm_table` | 4096 | `2295.47 +/- 35.32` | `2251.61 +/- 25.55` | `2259.16 +/- 30.03` | `650.79 +/- 23.91` | `1.02x` | `3.53x` |

### 2.2 Safe Rust `--release` Results (`crates/flexmesa-bench`)

| Workload | $N_{\mathrm{zones}}$ | Stage 0 ($\mu\mathrm{s}$) | Stage 1 ($\mu\mathrm{s}$) | Stage 2 ($\mu\mathrm{s}$) | Stage 3 (4t `rayon`, $\mu\mathrm{s}$) | Speedup vs. 1t Fortran | Speedup vs. 4t Fortran |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eos_4d` | 128 | `148.39 +/- 0.32` | `147.02 +/- 12.32` | `120.94 +/- 0.15` | `61.11 +/- 1.71` | `1.03x` | `0.52x` |
| `kap_fixed` | 128 | `56.55 +/- 0.23` | `56.18 +/- 0.19` | `32.36 +/- 0.12` | `8.25 +/- 0.07` | `2.22x` | `2.31x` |
| `rates_screen` | 128 | `14.17 +/- 0.01` | `13.29 +/- 0.03` | `11.80 +/- 0.11` | `3.18 +/- 0.08` | `1.68x` | `1.69x` |
| `net_eval` | 128 | `627.15 +/- 32.13` | `617.36 +/- 39.22` | `429.15 +/- 13.07` | `128.02 +/- 22.70` | `2.12x` | `1.77x` |
| `atm_table` | 128 | `8.40 +/- 0.02` | `8.30 +/- 2.37` | `7.53 +/- 0.02` | `4.09 +/- 0.07` | `5.37x` | `2.68x` |
| `eos_4d` | 1024 | `1655.79 +/- 465.69` | `1199.21 +/- 18.81` | `982.48 +/- 9.99` | `243.64 +/- 0.52` | `1.03x` | `1.03x` |
| `kap_fixed` | 1024 | `461.39 +/- 13.09` | `458.80 +/- 10.28` | `258.40 +/- 6.24` | `65.15 +/- 0.05` | `1.75x` | `1.71x` |
| `rates_screen` | 1024 | `113.05 +/- 0.15` | `105.52 +/- 0.25` | `93.34 +/- 0.08` | `24.05 +/- 0.04` | `1.69x` | `1.70x` |
| `net_eval` | 1024 | `5018.69 +/- 81.79` | `4939.75 +/- 43.37` | `3554.85 +/- 64.89` | `924.97 +/- 20.50` | `2.09x` | `2.03x` |
| `atm_table` | 1024 | `66.10 +/- 0.35` | `62.30 +/- 0.49` | `59.86 +/- 0.10` | `31.20 +/- 0.04` | `9.47x` | `5.62x` |
| `eos_4d` | 4096 | `4892.84 +/- 146.21` | `4793.33 +/- 55.42` | `3927.63 +/- 80.07` | `994.56 +/- 31.64` | `1.04x` | `1.46x` |
| `kap_fixed` | 4096 | `1834.23 +/- 205.72` | `1894.74 +/- 69.70` | `1056.85 +/- 11.01` | `266.31 +/- 13.50` | `1.69x` | `1.70x` |
| `rates_screen` | 4096 | `462.55 +/- 9.98` | `433.06 +/- 11.43` | `370.44 +/- 10.16` | `95.25 +/- 0.11` | `1.72x` | `2.85x` |
| `net_eval` | 4096 | `20136.35 +/- 420.27` | `20345.87 +/- 715.09` | `14496.41 +/- 346.48` | `3738.18 +/- 104.26` | `2.04x` | `2.72x` |
| `atm_table` | 4096 | `273.61 +/- 11.00` | `254.15 +/- 13.01` | `249.26 +/- 10.43` | `62.66 +/- 0.19` | `9.06x` | `10.39x` |

### 2.3 Python / PyO3 (`flexmesa._rust`) and JAX `float64` Results

| Workload | $N_{\mathrm{zones}}$ | Stage 0 PyO3 1t ($\mu\mathrm{s}$) | Stage 1 PyO3 4t ($\mu\mathrm{s}$) | Stage 2 JAX `@jit` ($\mu\mathrm{s}$) | Stage 2 Compile ($\mathrm{ms}$) | Stage 3 JAX `@jit(vmap)` ($\mu\mathrm{s}$) | Stage 3 Compile ($\mathrm{ms}$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eos_4d` | 128 | `125.78 +/- 2.66` | `33.23 +/- 1.45` | `72.61 +/- 30.80` | `50.8` | `21.45 +/- 4.19` | `61.1` |
| `kap_fixed` | 128 | `35.30 +/- 0.26` | `19.28 +/- 0.39` | `61.27 +/- 35.85` | `9.2` | `22.32 +/- 5.50` | `7.3` |
| `rates_screen` | 128 | `20.02 +/- 2.05` | `6.42 +/- 0.25` | `86.80 +/- 1.67` | `96.3` | `29.65 +/- 0.61` | `84.1` |
| `net_eval` | 128 | `452.46 +/- 34.05` | `117.13 +/- 7.03` | `230.18 +/- 88.92` | `7.1` | `155.03 +/- 23.74` | `6.5` |
| `atm_table` | 128 | `9.72 +/- 0.19` | `3.99 +/- 0.28` | `90.69 +/- 2.67` | `174.9` | `29.64 +/- 0.33` | `167.8` |
| `eos_4d` | 1024 | `985.26 +/- 34.30` | `262.54 +/- 18.45` | `143.39 +/- 22.38` | `43.2` | `103.78 +/- 53.77` | `52.7` |
| `kap_fixed` | 1024 | `276.80 +/- 18.34` | `100.11 +/- 4.06` | `155.83 +/- 87.95` | `6.6` | `110.21 +/- 11.13` | `6.0` |
| `rates_screen` | 1024 | `152.47 +/- 2.68` | `66.06 +/- 0.48` | `170.44 +/- 19.45` | `80.6` | `114.80 +/- 9.88` | `86.7` |
| `net_eval` | 1024 | `3658.94 +/- 226.94` | `922.78 +/- 49.21` | `1087.57 +/- 54.83` | `8.1` | `958.61 +/- 74.96` | `10.2` |
| `atm_table` | 1024 | `72.19 +/- 2.31` | `21.03 +/- 0.21` | `155.64 +/- 1.22` | `151.6` | `90.29 +/- 3.39` | `179.6` |
| `eos_4d` | 4096 | `4061.90 +/- 308.84` | `1070.91 +/- 416.88` | `199.64 +/- 94.77` | `44.3` | `143.95 +/- 107.92` | `53.2` |
| `kap_fixed` | 4096 | `1125.31 +/- 24.58` | `300.27 +/- 60.32` | `430.56 +/- 18.17` | `7.1` | `401.20 +/- 143.76` | `8.3` |
| `rates_screen` | 4096 | `628.68 +/- 27.61` | `167.71 +/- 76.48` | `412.61 +/- 4.02` | `76.9` | `226.50 +/- 127.13` | `80.1` |
| `net_eval` | 4096 | `14479.96 +/- 477.25` | `3792.84 +/- 1990.71` | `3808.93 +/- 15.25` | `10.7` | `3709.81 +/- 113.37` | `22.1` |
| `atm_table` | 4096 | `284.13 +/- 3.52` | `81.50 +/- 15.01` | `334.22 +/- 28.94` | `194.9` | `167.47 +/- 4.71` | `156.0` |

---

## 3. Four Holistic Wave 3b.5 Cross-Module Optimizations

1. **Zero-Hash `patch_lut`, Lock-Free `get_table_fast` & Stack-Only `eval_xtable_16` (`flexmesa-eos`)**:
   - `EosDtXzTable::get_vertex` previously performed a `SipHash` `HashMap` lookup at each of the 4 cell vertices across 3-4 $X$-interpolation planes (`12-16` hash lookups per zone), while `EosDtTableSet::get_table` acquired a `RwLock` and incremented an atomic `Arc` reference count (`Arc::clone`) on every table access. Adding a sorted `patch_lut: Vec<(u32, EosDtVertex)>` binary search, a lock-free `get_table_fast -> Option<&EosDtXzTable>` fast path for built-in tables, and a stack-allocated 16-channel `eval_xtable_16` reduced 4096-zone `eos_4d` latency from $14.37\,\mathrm{ms}$ to $3.93\,\mathrm{ms}$ (1-thread) and $0.99\,\mathrm{ms}$ (4-thread `rayon`), outperforming upstream Fortran MESA (`4.07 ms` / `1.45 ms`) by **`1.46x`**, while the pre-blended JAX `@jax.jit(vmap)` kernel evaluates 4096 zones in **$143.95\,\mu\mathrm{s}$** (**`10.1x` faster** than 4-thread OpenMP Fortran MESA).
2. **Thread-Local Workspace Reuse & 2-Lane SIMD Dual Numbers (`F64x2`) in `flexmesa-net`**:
   - Upstream Fortran MESA allocates and zeroes per-zone rate and derivative work arrays inside `net_get`. By introducing `F64x2` (carrying $(\ln T, \ln\rho)$ partials simultaneously in a single 16-byte register pair) and `with_thread_local_work` (`NetWorkArrays` cached in thread-local storage across zones), 4096-zone `basic.net` RHS + analytical Jacobian evaluation runs in $14.50\,\mathrm{ms}$ (1-thread) and $3.74\,\mathrm{ms}$ (4-thread `rayon`), achieving **`2.04x` (1t)** and **`2.72x` (4t)** speedups over upstream Fortran MESA (`29.59 ms` / `10.19 ms`).
3. **Pre-Validated Bicubic Atmosphere Splines (`flexmesa-atm`)**:
   - Calling `interp_evbicub_prevalidated_db` directly on the lazily initialized `AtmTables::builtin()` bicubic coefficient arrays eliminates redundant grid monotonicity checks on every surface boundary evaluation, accelerating 4096-zone `atm_table` to $249.26\,\mu\mathrm{s}$ (1-thread) and $62.66\,\mu\mathrm{s}$ (4-thread `rayon`), or **`9.06x` (1t)** and **`10.39x` (4t)** faster than upstream Fortran MESA (`2259.16 us` / `650.79 us`).
4. **Hoisted Composition Moments & Pre-Fetched REACLIB Handles in Batch PyO3 Kernels (`flexmesa-py`)**:
   - Hoisting `get_composition_info` and `RatesCatalog::builtin().reaclib_terms(handle)` outside the parallel `eval_batch_2d` zone closures in `eos_eval_opal_scvh_zones`, `kap_get_results_zones_rust`, and `rates_eval_reaclib_screen_zones` eliminated per-zone string/catalog lookups across Python/PyO3 and JAX `pure_callback` execution.
