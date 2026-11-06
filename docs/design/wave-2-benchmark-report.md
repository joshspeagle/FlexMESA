# Wave 2.5 Four-Way Performance Benchmark and Consolidation Report

This report documents the empirical results of the Wave 2.5 (`#41`) four-stage
performance staircase and cross-language parity verification across all four
implementations of the Wave 2 atomic, ionization, and photometric microphysics
modules (`chem`, `ionization`, and `colors`):

1. **Upstream Fortran MESA (`gfortran -O2 -fopenmp`)**: Native Fortran 2008
   modules (`libchem.a`, `libionization.a`, and `libcolors.a`) benchmarked via
   `benchmarks/fortran_mesa/run_wave2_bench`.
2. **Safe Rust (`crates/flexmesa-bench`)**: `#![forbid(unsafe_code)]` Cargo
   workspace crates (`flexmesa-chem`, `flexmesa-ionization`, `flexmesa-colors`)
   compiled with `--release` and persistent 4-thread `rayon` parallelism.
3. **Python / PyO3 (`src/flexmesa/bench`, Stages 0 and 1)**: Zero-copy C-order
   `numpy.float64` bindings (`flexmesa._rust`) dispatching to 1-thread (Stage 0)
   and 4-thread adaptive spin-then-park (Stage 1) Safe Rust batch kernels.
4. **Python / JAX (`src/flexmesa/bench`, Stages 2 and 3)**: XLA-compiled
   `@jax.jit` and `@jax.jit(jax.vmap)` GPU kernels in strict IEEE-754 double
   precision (`jax_enable_x64=True`).

---

## 1. Cross-Language Numerical Parity Gate

Across all four Wave 2 workloads (`chem_comp`, `ion_fe56`, `ion_4d`, and
`colors_sed`) and all radial grid sizes $N_{\mathrm{zones}} \in \{8, 128, 1024,
4096\}$, the reduction checksums match across Upstream Fortran MESA, Safe Rust,
Python (`flexmesa._rust`), and JAX (`float64`):

| Workload | Physical Reduction | $N_{\mathrm{zones}} = 8$ Checksum | $N_{\mathrm{zones}} = 128$ Checksum | $N_{\mathrm{zones}} = 1024$ Checksum | $N_{\mathrm{zones}} = 4096$ Checksum | Max Relative Error vs. Fortran |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `chem_comp` | $\sum_k \bar{A}_k$ | `1.68700149e+01` | `2.63419264e+02` | `2.10476347e+03` | `8.41795382e+03` | `0.00e+00` |
| `ion_fe56` | $\sum_k Z_{\mathrm{eff}, k}(\mathrm{Fe}^{56}\text{ in }\mathrm{He}^4)$ | `1.68370939e+02` | `2.73675912e+03` | `2.19124406e+04` | `8.76575973e+04` | `4.83e-09` |
| `ion_4d` | $\sum_k \log_{10} P_{\mathrm{gas}, k}$ | `6.11884615e+01` | `9.79041962e+02` | `7.83234656e+03` | `3.13293908e+04` | `< 1.00e-12` |
| `colors_sed` | $\sum_k M_{V, k}$ | `3.60869173e+01` | `5.76955755e+02` | `4.61546861e+03` | `1.84617990e+04` | `< 1.00e-12` |

> [!NOTE]
> In `ion_fe56`, Safe Rust, Python, and JAX agree to $3.70 \times 10^{-14}$ in
> IEEE-754 `f64`, while upstream Fortran MESA's `eval_charge_of_Fe56_in_He4`
> calls `fvbicub_db` (`interp_2d/private/bicub_db.f`), where omitted `implicit
> none` leaves local Hermite basis variables (`xp`, `yp`, `cx`, `cy`, `hx2`,
> `hy2`, `z36th`) implicitly typed as 32-bit `REAL` (`f32`), producing a
> $4.83 \times 10^{-9}$ single-precision floor. All other Wave 2 workloads agree
> to $\le 10^{-12}$ across all four engines.

![Four-way Wave 2.5 runtime comparison and Fe56 in He4 effective charge profiles](../assets/figures/wave2_benchmark_staircase.png)

---

## 2. Four-Way Runtime Tables ($N_{\mathrm{zones}} \in \{128, 1024, 4096\}$)

### 2.1 Upstream Fortran MESA (`gfortran -O2 -fopenmp`)

| Workload | $N_{\mathrm{zones}}$ | Stage 0 ($\mu\mathrm{s}$) | Stage 1 ($\mu\mathrm{s}$) | Stage 2 ($\mu\mathrm{s}$) | Stage 3 (4t OpenMP, $\mu\mathrm{s}$) | Speedup (Stage 0 / Stage 2) | Speedup (Stage 0 / Stage 3) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chem_comp` | 128 | `6.29 +/- 0.02` | `6.20 +/- 0.01` | `1.71 +/- 0.00` | `0.75 +/- 0.33` | `3.68x` | `8.39x` |
| `ion_fe56` | 128 | `2.91 +/- 0.01` | `2.89 +/- 0.02` | `4.15 +/- 0.01` | `1.37 +/- 0.20` | `0.70x` | `2.12x` |
| `ion_4d` | 128 | `148.24 +/- 5.91` | `147.15 +/- 7.93` | `143.87 +/- 0.46` | `37.12 +/- 10.84` | `1.03x` | `3.99x` |
| `colors_sed` | 128 | `14012.13 +/- 443.92` | `14493.28 +/- 295.71` | `13819.15 +/- 511.88` | `6020.47 +/- 147.47` | `1.01x` | `2.33x` |
| `chem_comp` | 1024 | `50.35 +/- 0.09` | `49.55 +/- 0.04` | `13.58 +/- 0.02` | `3.78 +/- 0.11` | `3.71x` | `13.32x` |
| `ion_fe56` | 1024 | `22.46 +/- 0.02` | `22.37 +/- 0.06` | `32.16 +/- 0.01` | `8.65 +/- 0.04` | `0.70x` | `2.60x` |
| `ion_4d` | 1024 | `1192.50 +/- 31.19` | `1184.61 +/- 16.12` | `1183.28 +/- 34.50` | `290.53 +/- 11.00` | `1.01x` | `4.10x` |
| `colors_sed` | 1024 | `112893.43 +/- 3262.08` | `112570.48 +/- 1626.12` | `112498.37 +/- 1591.43` | `28570.68 +/- 6211.56` | `1.00x` | `3.95x` |
| `chem_comp` | 4096 | `203.07 +/- 8.26` | `300.77 +/- 41.27` | `54.21 +/- 14.49` | `13.92 +/- 0.03` | `3.75x` | `14.59x` |
| `ion_fe56` | 4096 | `86.90 +/- 0.20` | `86.89 +/- 0.03` | `126.31 +/- 0.15` | `33.17 +/- 0.06` | `0.69x` | `2.62x` |
| `ion_4d` | 4096 | `4744.66 +/- 14.10` | `4781.55 +/- 75.10` | `4668.26 +/- 16.45` | `1200.22 +/- 62.54` | `1.02x` | `3.95x` |
| `colors_sed` | 4096 | `447327.33 +/- 7981.71` | `444560.94 +/- 2144.19` | `440654.32 +/- 2296.58` | `119326.99 +/- 6228.75` | `1.02x` | `3.75x` |

### 2.2 Safe Rust `--release` Results (`crates/flexmesa-bench`)

| Workload | $N_{\mathrm{zones}}$ | Stage 0 ($\mu\mathrm{s}$) | Stage 1 ($\mu\mathrm{s}$) | Stage 2 ($\mu\mathrm{s}$) | Stage 3 (4t `rayon`, $\mu\mathrm{s}$) | Speedup (Stage 0 / Stage 2) | Speedup (Stage 0 / Stage 3) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chem_comp` | 128 | `16.09 +/- 0.03` | `15.92 +/- 0.04` | `1.84 +/- 0.02` | `0.65 +/- 0.04` | `8.74x` | `24.75x` |
| `ion_fe56` | 128 | `4.87 +/- 0.01` | `4.64 +/- 0.04` | `4.42 +/- 0.01` | `1.22 +/- 0.05` | `1.10x` | `3.99x` |
| `ion_4d` | 128 | `111.63 +/- 0.96` | `112.27 +/- 4.69` | `110.62 +/- 0.29` | `27.83 +/- 0.06` | `1.01x` | `4.01x` |
| `colors_sed` | 128 | `3710.65 +/- 37.08` | `3733.51 +/- 91.74` | `817.75 +/- 15.96` | `402.51 +/- 6.36` | `4.54x` | `9.22x` |
| `chem_comp` | 1024 | `130.12 +/- 0.11` | `127.62 +/- 0.17` | `14.89 +/- 0.09` | `3.82 +/- 0.07` | `8.74x` | `34.06x` |
| `ion_fe56` | 1024 | `38.83 +/- 38.69` | `36.98 +/- 0.16` | `35.19 +/- 0.34` | `17.80 +/- 0.16` | `1.10x` | `2.18x` |
| `ion_4d` | 1024 | `907.16 +/- 17.67` | `906.68 +/- 8.29` | `888.18 +/- 15.79` | `230.60 +/- 5.19` | `1.02x` | `3.93x` |
| `colors_sed` | 1024 | `30556.56 +/- 347.28` | `30362.44 +/- 370.89` | `6566.28 +/- 204.25` | `1638.59 +/- 16.42` | `4.65x` | `18.65x` |
| `chem_comp` | 4096 | `516.48 +/- 13.73` | `508.93 +/- 10.69` | `58.72 +/- 0.30` | `29.57 +/- 0.10` | `8.80x` | `17.47x` |
| `ion_fe56` | 4096 | `154.80 +/- 0.87` | `147.52 +/- 1.13` | `141.13 +/- 1.59` | `70.94 +/- 0.39` | `1.10x` | `2.18x` |
| `ion_4d` | 4096 | `3620.51 +/- 162.86` | `3662.73 +/- 97.58` | `3605.28 +/- 145.34` | `1794.73 +/- 17.82` | `1.00x` | `2.02x` |
| `colors_sed` | 4096 | `121142.58 +/- 1413.12` | `120785.82 +/- 705.42` | `26539.29 +/- 587.83` | `6715.11 +/- 529.77` | `4.56x` | `18.04x` |

### 2.3 Python / PyO3 (`flexmesa._rust`) and JAX `float64` Results

| Workload | $N_{\mathrm{zones}}$ | Stage 0 PyO3 1t ($\mu\mathrm{s}$) | Stage 1 PyO3 4t ($\mu\mathrm{s}$) | Stage 2 JAX `@jit` ($\mu\mathrm{s}$) | Stage 2 Compile ($\mathrm{ms}$) | Stage 3 JAX `@jit(vmap)` ($\mu\mathrm{s}$) | Stage 3 Compile ($\mathrm{ms}$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chem_comp` | 128 | `2.42 +/- 0.16` | `2.79 +/- 0.25` | `227.11 +/- 92.29` | `65.5` | `53.05 +/- 6.43` | `62.6` |
| `ion_fe56` | 128 | `5.43 +/- 0.03` | `2.81 +/- 0.19` | `1445.48 +/- 56.84` | `212.3` | `358.76 +/- 9.22` | `229.7` |
| `ion_4d` | 128 | `142.52 +/- 5.44` | `35.47 +/- 0.26` | `242.73 +/- 71.86` | `463.1` | `318.67 +/- 4.06` | `436.4` |
| `colors_sed` | 128 | `778.17 +/- 27.55` | `198.56 +/- 15.10` | `1510.22 +/- 52.62` | `1290.7` | `399.26 +/- 30.01` | `1464.8` |
| `chem_comp` | 1024 | `13.99 +/- 0.11` | `10.51 +/- 0.25` | `206.17 +/- 62.69` | `70.0` | `45.96 +/- 268.76` | `68.0` |
| `ion_fe56` | 1024 | `39.37 +/- 0.12` | `12.76 +/- 0.37` | `1445.59 +/- 32.91` | `208.5` | `359.66 +/- 225.33` | `230.1` |
| `ion_4d` | 1024 | `1096.38 +/- 21.67` | `293.93 +/- 16.32` | `1273.34 +/- 14.33` | `491.5` | `318.62 +/- 4.87` | `473.4` |
| `colors_sed` | 1024 | `6241.35 +/- 25.49` | `1570.04 +/- 23.18` | `1540.85 +/- 58.09` | `1677.6` | `744.64 +/- 81.75` | `1316.4` |
| `chem_comp` | 4096 | `54.75 +/- 0.48` | `23.78 +/- 0.50` | `253.15 +/- 36.92` | `76.1` | `60.37 +/- 0.33` | `82.7` |
| `ion_fe56` | 4096 | `155.62 +/- 4.82` | `46.85 +/- 15.06` | `1424.94 +/- 14.76` | `225.1` | `365.54 +/- 19.22` | `252.8` |
| `ion_4d` | 4096 | `4499.18 +/- 139.73` | `1185.94 +/- 12.38` | `1287.67 +/- 10.34` | `453.3` | `2217.73 +/- 19.18` | `731.3` |
| `colors_sed` | 4096 | `25195.23 +/- 439.91` | `6431.54 +/- 234.53` | `2720.93 +/- 118.37` | `1283.1` | `5498.57 +/- 72.09` | `1416.7` |

---

## 3. Four Holistic Wave 2.5 Cross-Module Optimizations

1. **Pre-Validated Bicubic Spline Fast Path (`interp_evbicub_prevalidated_db`)**:
   - Calling `interp_evbicub_db` on immutable pre-constructed spline tables
     previously repeated $O(N_x \times N_y)$ grid monotonicity (`splinck_db`)
     and coefficient finiteness checks on every zone evaluation. Adding
     `herm2xy_prevalidated_db` and `interp_evbicub_prevalidated_db` in
     `flexmesa-interp2d` and wiring them into `Fe56He4Table::eval_full` cut
     1024-zone `ion_fe56` latency from $2508.13\,\mu\mathrm{s}$ to
     $35.19\,\mu\mathrm{s}$ on 1 thread and $17.80\,\mu\mathrm{s}$ on 4 threads
     (**$71.3\times$ single-thread speedup**).
2. **Adaptive Spin-Then-Park 4-Thread Worker Team in `flexmesa-py`**:
   - Invoking `rayon::ThreadPool::install()` from Python's main thread enters
     `rayon-core`'s cold `LockLatch` condition-variable wakeup path ($\sim
     180\,\mu\mathrm{s}$ per call). Replacing external `rayon` dispatch in
     `flexmesa-py` with a 100% Safe Rust (`#![deny(unsafe_code)]`) 3-worker
     adaptive spin-then-park team where the calling thread executes Chunk 0
     directly while the 3 workers spin on atomic phase flags before parking
     reduced `PyO3` 4-thread parallel dispatch overhead by **$16.4\times$**
     ($215.93\,\mu\mathrm{s} \to 13.16\,\mu\mathrm{s}$), accelerating Python
     `chem_comp` ($N=1024$, 4t) to $10.51\,\mu\mathrm{s}$ and `ion_fe56`
     ($N=128$, 4t) to $2.81\,\mu\mathrm{s}$.
3. **Precomputed Non-Uniform Simpson Panel Weights & Stack Buffer (`flexmesa-colors`)**:
   - Precomputing the 599 non-uniform quadratic Simpson panel weights
     `[c0, c1, c2]` and wavelength-weighted filter vectors on the 1199-point
     Kurucz grid and fusing the bolometric + 7 Johnson filter integrals over a
     stack `[T; 1199]` buffer eliminated all per-zone heap allocations and
     filter curve resamplings, accelerating 1024-zone Safe Rust `colors_sed` to
     $6566.28\,\mu\mathrm{s}$ on 1 thread (**$17.2\times$ faster** than 1-thread
     Fortran MESA) and $1638.59\,\mu\mathrm{s}$ on 4 threads (**$17.4\times$
     faster** than 4-thread OpenMP Fortran MESA and **$68.9\times$ faster** than
     Fortran Stage 0).
4. **32-Gather Interior Corner Deduplication in 3D Cubic Hermite SED Synthesis**:
   - Factoring the 8 interior cell corners out of the three finite-difference
     tangent axes into a single combined weight `w_corner = w_val + w_dx + w_dy
     + w_dz` in both Safe Rust and JAX reduced 1199-wavelength memory gathers per
     model from 56 to 32 (**43% bandwidth reduction**), enabling JAX
     `@jax.jit(jax.vmap)` to synthesize 1024 Kurucz SEDs and 7-band Johnson
     plus bolometric photometry in $744.64\,\mu\mathrm{s}$ (**$151.6\times$
     faster** than Fortran Stage 0 and **$38.4\times$ faster** than 4-thread
     Fortran OpenMP).
