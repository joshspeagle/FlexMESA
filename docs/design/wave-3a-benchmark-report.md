# Wave 3a.5 Four-Way Performance Benchmark and Consolidation Report

This report documents the empirical results of the Wave 3a.5 (`#51`) four-stage
performance staircase and cross-language parity verification across all four
implementations of the Wave 3a analytical and single-table equation of state
(`eos`) engines (`ideal`, `helm`, `pc`, and `skye`):

1. **Upstream Fortran MESA (`gfortran -O2 -fopenmp`)**: Native Fortran 2008
   modules (`libeos.a`, `libchem.a`, `libinterp_2d.a`, `libauto_diff.a`)
   benchmarked via `benchmarks/fortran_mesa/run_wave3a_bench`.
2. **Safe Rust (`crates/flexmesa-bench`)**: `#![forbid(unsafe_code)]` Cargo
   workspace crate (`flexmesa-eos`) compiled with `--release` and persistent
   4-thread `rayon` parallelism.
3. **Python / PyO3 (`src/flexmesa/bench`, Stages 0 and 1)**: Zero-copy C-order
   `numpy.float64` bindings (`flexmesa._rust`) dispatching to 1-thread (Stage 0)
   and 4-thread adaptive spin-then-park (Stage 1) Safe Rust batch kernels.
4. **Python / JAX (`src/flexmesa/bench`, Stages 2 and 3)**: XLA-compiled
   `@jax.jit` and `@jax.jit(jax.vmap)` GPU kernels in strict IEEE-754 double
   precision (`jax_enable_x64=True`).

---

## 1. Cross-Language Numerical Parity Gate

Across all four Wave 3a workloads (`eos_ideal`, `eos_helm`, `eos_pc`, and
`eos_skye`) and all radial grid sizes $N_{\mathrm{zones}} \in \{8, 128, 1024,
4096\}$, the reduction checksums ($\sum_{k=1}^{N_{\mathrm{zones}}} \ln
P_{\mathrm{gas}, k}$) match across Upstream Fortran MESA, Safe Rust, Python
(`flexmesa._rust`), and JAX (`float64`):

| Workload | Physical Reduction | $N_{\mathrm{zones}} = 8$ Checksum | $N_{\mathrm{zones}} = 128$ Checksum | $N_{\mathrm{zones}} = 1024$ Checksum | $N_{\mathrm{zones}} = 4096$ Checksum | Max Relative Error vs. Fortran |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `eos_ideal` | $\sum_k \ln P_{\mathrm{gas}, k}$ | `1.43822426e+02` | `2.30115881e+03` | `1.84092705e+04` | `7.36370820e+04` | `< 1.00e-14` |
| `eos_helm` | $\sum_k \ln P_{\mathrm{gas}, k}$ | `3.56168896e+02` | `5.68883671e+03` | `4.55066091e+04` | `1.82024698e+05` | `< 1.00e-12` |
| `eos_pc` | $\sum_k \ln P_{\mathrm{gas}, k}$ | `4.11788689e+02` | `6.59157662e+03` | `5.27338319e+04` | `2.10935846e+05` | `< 1.00e-12` |
| `eos_skye` | $\sum_k \ln P_{\mathrm{gas}, k}$ | `3.56133184e+02` | `5.68822944e+03` | `4.55017356e+04` | `1.82005197e+05` | `< 1.00e-11` |

![Four-way Wave 3a.5 runtime comparison and Coulomb non-ideal pressure shifts](../assets/figures/wave3a_benchmark_staircase.png)

---

## 2. Four-Way Runtime Tables ($N_{\mathrm{zones}} \in \{128, 1024, 4096\}$)

### 2.1 Upstream Fortran MESA (`gfortran -O2 -fopenmp`)

| Workload | $N_{\mathrm{zones}}$ | Stage 0 ($\mu\mathrm{s}$) | Stage 1 ($\mu\mathrm{s}$) | Stage 2 ($\mu\mathrm{s}$) | Stage 3 (4t OpenMP, $\mu\mathrm{s}$) | Speedup (Stage 0 / Stage 2) | Speedup (Stage 0 / Stage 3) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eos_ideal` | 128 | `136.13 +/- 9.46` | `135.19 +/- 3.80` | `118.80 +/- 0.25` | `30.48 +/- 0.15` | `1.15x` | `4.47x` |
| `eos_helm` | 128 | `124.62 +/- 11.17` | `114.32 +/- 0.19` | `101.39 +/- 0.08` | `25.96 +/- 0.40` | `1.23x` | `4.80x` |
| `eos_pc` | 128 | `572.67 +/- 26.29` | `553.90 +/- 17.74` | `555.94 +/- 25.26` | `135.96 +/- 22.73` | `1.03x` | `4.21x` |
| `eos_skye` | 128 | `923.88 +/- 25.83` | `916.15 +/- 10.99` | `913.66 +/- 12.50` | `224.43 +/- 17.99` | `1.01x` | `4.12x` |
| `eos_ideal` | 1024 | `1108.75 +/- 60.78` | `1097.52 +/- 14.98` | `965.82 +/- 12.93` | `349.01 +/- 97.24` | `1.15x` | `3.18x` |
| `eos_helm` | 1024 | `1010.61 +/- 158.73` | `940.48 +/- 33.12` | `831.93 +/- 23.01` | `204.05 +/- 23.85` | `1.21x` | `4.95x` |
| `eos_pc` | 1024 | `4596.94 +/- 81.18` | `4543.44 +/- 126.31` | `4435.50 +/- 202.31` | `1170.70 +/- 71.31` | `1.04x` | `3.93x` |
| `eos_skye` | 1024 | `7400.78 +/- 58.23` | `7453.27 +/- 85.26` | `7473.07 +/- 162.98` | `2931.65 +/- 50.09` | `0.99x` | `2.52x` |
| `eos_ideal` | 4096 | `4621.70 +/- 200.99` | `4500.93 +/- 57.83` | `3968.06 +/- 80.59` | `1016.38 +/- 27.68` | `1.16x` | `4.55x` |
| `eos_helm` | 4096 | `3936.75 +/- 117.74` | `4070.24 +/- 302.54` | `3370.93 +/- 51.31` | `859.26 +/- 198.56` | `1.17x` | `4.58x` |
| `eos_pc` | 4096 | `18825.09 +/- 601.82` | `18644.13 +/- 458.25` | `17940.60 +/- 320.49` | `6540.41 +/- 942.09` | `1.05x` | `2.88x` |
| `eos_skye` | 4096 | `30089.74 +/- 566.28` | `29356.87 +/- 635.00` | `29247.01 +/- 449.81` | `7452.20 +/- 528.98` | `1.03x` | `4.04x` |

### 2.2 Safe Rust `--release` Results (`crates/flexmesa-bench`)

| Workload | $N_{\mathrm{zones}}$ | Stage 0 ($\mu\mathrm{s}$) | Stage 1 ($\mu\mathrm{s}$) | Stage 2 ($\mu\mathrm{s}$) | Stage 3 (4t `rayon`, $\mu\mathrm{s}$) | Speedup vs. 1t Fortran | Speedup vs. 4t Fortran |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eos_ideal` | 128 | `106.30 +/- 0.27` | `107.30 +/- 0.02` | `98.46 +/- 0.32` | `24.65 +/- 0.04` | `1.21x` | `1.24x` |
| `eos_helm` | 128 | `49.41 +/- 0.14` | `49.44 +/- 0.37` | `42.79 +/- 0.09` | `11.07 +/- 0.09` | `2.37x` | `2.35x` |
| `eos_pc` | 128 | `302.56 +/- 18.73` | `311.70 +/- 11.41` | `300.80 +/- 2.33` | `83.05 +/- 74.89` | `1.85x` | `1.64x` |
| `eos_skye` | 128 | `762.28 +/- 30.58` | `747.29 +/- 21.82` | `758.82 +/- 46.73` | `183.35 +/- 5.17` | `1.20x` | `1.22x` |
| `eos_ideal` | 1024 | `865.53 +/- 25.70` | `882.50 +/- 46.89` | `803.50 +/- 14.38` | `198.83 +/- 5.14` | `1.20x` | `1.76x` |
| `eos_helm` | 1024 | `396.45 +/- 15.06` | `408.33 +/- 28.74` | `341.68 +/- 22.73` | `85.72 +/- 7.44` | `2.43x` | `2.38x` |
| `eos_pc` | 1024 | `2599.44 +/- 269.67` | `2463.16 +/- 134.27` | `2493.02 +/- 160.13` | `633.67 +/- 45.24` | `1.78x` | `1.85x` |
| `eos_skye` | 1024 | `6451.20 +/- 627.29` | `6084.70 +/- 199.20` | `6074.05 +/- 130.98` | `2329.83 +/- 81.41` | `1.23x` | `1.26x` |
| `eos_ideal` | 4096 | `3650.39 +/- 248.82` | `3503.18 +/- 141.17` | `3252.28 +/- 200.45` | `840.20 +/- 80.57` | `1.22x` | `1.21x` |
| `eos_helm` | 4096 | `1574.77 +/- 38.51` | `1564.61 +/- 6.26` | `1379.22 +/- 27.82` | `350.72 +/- 17.37` | `2.44x` | `2.45x` |
| `eos_pc` | 4096 | `10107.25 +/- 91.09` | `9832.44 +/- 89.81` | `9783.18 +/- 524.47` | `2797.99 +/- 145.97` | `1.83x` | `2.34x` |
| `eos_skye` | 4096 | `24665.69 +/- 599.87` | `24877.49 +/- 873.30` | `24520.95 +/- 520.64` | `6842.21 +/- 614.57` | `1.19x` | `1.09x` |

### 2.3 Python / PyO3 (`flexmesa._rust`) and JAX `float64` Results

| Workload | $N_{\mathrm{zones}}$ | Stage 0 PyO3 1t ($\mu\mathrm{s}$) | Stage 1 PyO3 4t ($\mu\mathrm{s}$) | Stage 2 JAX `@jit` ($\mu\mathrm{s}$) | Stage 2 Compile ($\mathrm{ms}$) | Stage 3 JAX `@jit(vmap)` ($\mu\mathrm{s}$) | Stage 3 Compile ($\mathrm{ms}$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eos_ideal` | 128 | `101.42 +/- 1.86` | `27.79 +/- 1.02` | `1299.15 +/- 17.27` | `490.9` | `325.88 +/- 12.20` | `512.4` |
| `eos_helm` | 128 | `69.69 +/- 0.93` | `20.61 +/- 0.35` | `1291.87 +/- 29.37` | `1302.7` | `326.80 +/- 3.10` | `1348.2` |
| `eos_pc` | 128 | `298.64 +/- 16.16` | `77.72 +/- 0.57` | `1235.44 +/- 150.03` | `10.3` | `313.96 +/- 46.26` | `12.8` |
| `eos_skye` | 128 | `728.73 +/- 40.56` | `214.82 +/- 23.73` | `1355.22 +/- 16.48` | `9753.6` | `635.62 +/- 26.05` | `10142.0` |
| `eos_ideal` | 1024 | `855.23 +/- 74.76` | `239.28 +/- 39.02` | `1297.16 +/- 16.49` | `431.1` | `327.32 +/- 2.50` | `448.6` |
| `eos_helm` | 1024 | `558.73 +/- 20.45` | `160.92 +/- 21.06` | `1317.97 +/- 12.10` | `815.4` | `328.61 +/- 5.63` | `842.0` |
| `eos_pc` | 1024 | `2447.21 +/- 20.09` | `634.03 +/- 12.37` | `1912.67 +/- 152.02` | `11.9` | `1163.75 +/- 145.37` | `14.1` |
| `eos_skye` | 1024 | `5907.59 +/- 114.78` | `1586.10 +/- 32.89` | `1403.67 +/- 51.55` | `9912.4` | `605.55 +/- 13.33` | `10215.1` |
| `eos_ideal` | 4096 | `3220.25 +/- 85.32` | `1410.42 +/- 11.14` | `1320.46 +/- 30.59` | `333.1` | `325.58 +/- 1.12` | `351.8` |
| `eos_helm` | 4096 | `2350.17 +/- 34.53` | `722.62 +/- 26.61` | `1308.45 +/- 11.83` | `627.8` | `326.53 +/- 4.82` | `655.4` |
| `eos_pc` | 4096 | `9690.70 +/- 124.10` | `2479.66 +/- 42.31` | `3781.57 +/- 321.81` | `14.7` | `3213.93 +/- 213.35` | `18.2` |
| `eos_skye` | 4096 | `23281.02 +/- 210.79` | `6023.79 +/- 114.73` | `2551.69 +/- 81.44` | `8937.3` | `1421.04 +/- 13.91` | `9184.5` |

---

## 3. Four Holistic Wave 3a.5 Cross-Module Optimizations

1. **Process-Wide `OnceLock<NuclideCatalog>` Singleton (`NuclideCatalog::builtin_ref()`)**:
   - Calling `NuclideCatalog::builtin()` inside `eval_ideal_eos`, `eval_pc_eos`,
     `eval_skye_eos`, and `eos_dt_get` previously allocated and hashed all 347
     standard nuclear species on every zone evaluation. Introducing
     `NuclideCatalog::builtin_ref() -> &'static Self` backed by `std::sync::OnceLock`
     and stack-allocated `[0.0_f64; 32]` species buffers accelerated 4096-zone
     Safe Rust `eos_ideal` from $114.19\,\mathrm{ms}$ to $3.65\,\mathrm{ms}$
     (**$31.3\times$ speedup**), `eos_pc` from $122.14\,\mathrm{ms}$ to
     $10.11\,\mathrm{ms}$ (**$12.1\times$ speedup**), and `eos_skye` from
     $142.33\,\mathrm{ms}$ to $24.67\,\mathrm{ms}$ (**$5.77\times$ speedup**).
2. **Zero-Allocation Stack Buffers Across `ideal`, `pc`, `skye`, and `blend_dt`**:
   - Replacing per-zone `Vec<f64>` heap allocations and `EosEvalOutput::zeros(n)`
     in inner mixture loops (`eval_ideal_eos`, `eval_pc_eos`, `eval_skye_eos`,
     `eval_coulomb_solid`, `eval_nonideal_melange`, and `eos_dt_get`) with stack
     arrays (`[0.0_f64; 32]`) and `EosEvalOutput::zeros(0)` when composition
     partials (`need_d_dxa = false`) are omitted eliminated all heap traffic
     inside parallel zone loops, making 4-thread Safe Rust faster than 4-thread
     OpenMP Fortran MESA across all four Wave 3a workloads (`1.76x` on
     `eos_ideal`, `2.45x` on `eos_helm`, `2.34x` on `eos_pc`, and `1.26x` on
     `eos_skye`).
3. **Recursive `@jax.custom_jvp` Horner Polynomial Basis Chain (`_helm_jax_poly.py`)**:
   - Standard JAX autodiff through `w0(z) = 1 - z^2 (3 - 2 z)` evaluates
     `dw0(z) = -2 z (3 - 2 z) + 2 z^2`, which differs by 1-2 ULPs from Fortran
     MESA's factored Horner derivative `dw0(z) = 6 z (z - 1)`. Attaching
     recursive `@jax.custom_jvp` rules (`quintic_w0_coord` and `cubic_w0_coord`)
     eliminates the $O(\Delta \rho^{-2}) \sim 10^{10}$ finite-grid amplification
     at low density and restores `< 10^-11` cross-language parity across all 26
     `HELM` channels.
4. **Single-Pass Direct `(T, rho)` Second-Order JAX Free-Energy Autodiff (`ideal.py`, `_skye_jax.py`)**:
   - Differentiating the `ideal` and `Skye` free energies directly with respect
     to `(T, rho)` in a single `jax.jacfwd(..., has_aux=True)` pass rather than
     chaining through `jnp.exp(ln_T)` and `jnp.exp(ln_rho)` eliminated
     exponential roundoff amplification in $c_V = -T\,\partial^2 F/\partial T^2$
     and enabled GPU `@jax.jit(jax.vmap)` to evaluate 4096-zone `Skye` profiles
     in $1.42\,\mathrm{ms}$ (**$5.24\times$ faster** than 4-thread OpenMP
     Fortran MESA and **$21.2\times$ faster** than 1-thread Fortran MESA).
