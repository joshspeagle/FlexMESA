# Kernel Performance Scorecards

This page consolidates the four-stage optimization staircase across all 28 `FlexMESA` benchmark workloads. All measurements report `median` execution latency in microseconds ($\mu\mathrm{s}$) on a 16-core x86\_64 Linux workstation (30 GiB RAM, NVIDIA GeForce RTX 5080 GPU with 16 GiB GDDR7 VRAM), with CPU thread pools capped at 4 threads (`OMP_NUM_THREADS=4`, `RAYON_NUM_THREADS=4`) and JAX configured in strict IEEE-754 double precision (`jax_enable_x64=True`).

---

## Definition of the Four-Stage Optimization Staircase

Every workload is benchmarked across four progressive stages so that algorithmic improvements, memory-layout transformations, multi-core threading, and GPU vectorization are isolated quantitatively:

1. **Stage 0 (Sequential Baseline)**: Zone-by-zone scalar calls in Fortran MESA (`gfortran -O2`) and Safe Rust (`--release`), and 1-thread zero-copy `flexmesa._rust` (`PyO3`) execution in Python.
2. **Stage 1 (Contiguous SoA / Multi-Core FFI)**: Pre-allocated contiguous Structure-of-Arrays (`SoA`) buffers in Fortran MESA and Safe Rust, and 4-thread adaptive spin-then-park `flexmesa._rust` (`PyO3`) execution in Python.
3. **Stage 2 (Fused / Pre-Validated / Single-Star `@jax.jit`)**: Hoisted table invariants, pre-validated spline coefficients, and zero-allocation loops in Fortran MESA and Safe Rust, and warm steady-state single-star `@jax.jit` GPU execution in Python/JAX.
4. **Stage 3 (4-Thread Parallel / Batched `@jax.jit(vmap)`)**: 4-thread OpenMP execution in Fortran MESA, 4-thread persistent `rayon` execution in Safe Rust, and batched `@jax.jit(jax.vmap)` execution across $B = 4$ simultaneous stellar profiles (reported as amortized latency per star).

---

## 1. Foundational Numerics, Splines, Neutrinos & Convection ($N_{\mathrm{zones}} = 4096$)

Full per-size tables ($N_{\mathrm{zones}} \in \{128, 1024, 4096\}$) are archived in the [Foundational Numerics & Convection Benchmark Report](../design/wave-1-benchmark-report.md).

| Workload ($N_{\mathrm{zones}} = 4096$) | Fortran 1t Stage 2 ($\mu\mathrm{s}$) | Fortran 4t Stage 3 ($\mu\mathrm{s}$) | Safe Rust 1t Stage 2 ($\mu\mathrm{s}$) | Safe Rust 4t Stage 3 ($\mu\mathrm{s}$) | PyO3 Best ($\mu\mathrm{s}$) | JAX `@jit` Stage 2 ($\mu\mathrm{s}$) | JAX `vmap` Stage 3 ($\mu\mathrm{s}$) | Stage 2 Checksum |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `auto_diff` (hyperdual) | `21.11` | `5.65` | `23.33` | `5.89` | `100.22` | `1271.92` | `321.12` | `4.05502565e6` |
| `interp_2d` (bicubic) | `160.35` | `41.15` | `145.40` | `36.95` | `142.37` | `1285.74` | `322.45` | `1.84483783e5` |
| `neu` (Itoh 5-channel) | `575.47` | `152.32` | `615.34` | `161.32` | `312.58` | `1297.35` | `319.25` | `8.13481382e14` |
| `mtx_tridiag` ($5 \times 5$) | `870.34` | `221.38` | `1586.32` | `405.79` | `1652.80` | `2757.20` (FFI) / `7426.45` (PCR) | `5356.85` | `5.52650461e3` |
| `turb_mlt` (Cox MLT) | `7081.30` | `1784.93` | `550.28` | `142.14` | `290.02` | `209.01` | `317.73` | `1.75167983e3` |
| `turb_tdc` (Kuhfuss TDC) | `223895.15` | `58176.83` | `4009.46` | `1075.24` | `1205.71` | `1604.99` | `398.61` | `1.69966039e3` |

---

## 2. Atomic Composition, Ionization & Synthetic Photometry ($N_{\mathrm{zones}} = 4096$)

Full per-size tables ($N_{\mathrm{zones}} \in \{8, 128, 1024, 4096\}$) are archived in the [Composition, Ionization & Photometry Benchmark Report](../design/wave-2-benchmark-report.md).

| Workload ($N_{\mathrm{zones}} = 4096$) | Fortran 1t Stage 2 ($\mu\mathrm{s}$) | Fortran 4t Stage 3 ($\mu\mathrm{s}$) | Safe Rust 1t Stage 2 ($\mu\mathrm{s}$) | Safe Rust 4t Stage 3 ($\mu\mathrm{s}$) | PyO3 4t Stage 1 ($\mu\mathrm{s}$) | JAX `@jit` Stage 2 ($\mu\mathrm{s}$) | JAX `vmap` Stage 3 ($\mu\mathrm{s}$) | Stage 2 Checksum |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chem_comp` ($\bar{A}, \bar{Z}, Y_e$) | `54.21` | `13.92` | `58.72` | `29.57` | `23.78` | `253.15` | `60.37` | `8.41795382e3` |
| `ion_fe56` ($\mathrm{Fe}^{56}$ in $\mathrm{He}^4$) | `126.31` | `33.17` | `141.13` | `70.94` | `46.85` | `1424.94` | `365.54` | `8.76575973e4` |
| `ion_4d` (4D partial ion.) | `4668.26` | `1200.22` | `3605.28` | `1794.73` | `1185.94` | `1287.67` | `2217.73` | `3.13293908e4` |
| `colors_sed` (Kurucz + $UBVRI$) | `440654.32` | `119326.99` | `26539.29` | `6715.11` | `6431.54` | `2720.93` | `5498.57` | `1.84617990e4` |

---

## 3. Equations of State, Opacity, Networks & Atmospheres ($N_{\mathrm{zones}} = 4096$)

Full per-size tables ($N_{\mathrm{zones}} \in \{8, 128, 1024, 4096\}$) are archived in the [Analytical & Single-Table EOS Report](../design/wave-3a-benchmark-report.md) and the [4D EOS, Opacity, Networks & Atmospheres Report](../design/wave-3b-benchmark-report.md).

| Workload ($N_{\mathrm{zones}} = 4096$) | Fortran 1t Stage 2 ($\mu\mathrm{s}$) | Fortran 4t Stage 3 ($\mu\mathrm{s}$) | Safe Rust 1t Stage 2 ($\mu\mathrm{s}$) | Safe Rust 4t Stage 3 ($\mu\mathrm{s}$) | PyO3 4t Stage 1 ($\mu\mathrm{s}$) | JAX `@jit` Stage 2 ($\mu\mathrm{s}$) | JAX `vmap` Stage 3 ($\mu\mathrm{s}$) | Stage 2 Checksum |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `eos_ideal` (neutral + rad) | `3968.06` | `1016.38` | `3252.28` | `840.20` | `1410.42` | `1320.46` | `325.58` | `7.36370820e4` |
| `eos_helm` (Timmes & Swesty) | `3370.93` | `859.26` | `1379.22` | `350.72` | `722.62` | `1308.45` | `326.53` | `1.82024698e5` |
| `eos_pc` (Potekhin-Chabrier) | `17940.60` | `6540.41` | `9783.18` | `2797.99` | `2479.66` | `3781.57` | `3213.93` | `2.10935846e5` |
| `eos_skye` (3rd-order AD) | `29247.01` | `7452.20` | `24520.95` | `6842.21` | `6023.79` | `2551.69` | `1421.04` | `1.82005197e5` |
| `eos_4d` (`OPAL`/`FreeEOS`) | `4068.05` | `1451.55` | `3927.63` | `994.56` | `1070.91` | `199.64` | `143.95` | `1.04313071e5` |
| `kap_fixed` (rad + cond) | `1789.49` | `452.32` | `1056.85` | `266.31` | `300.27` | `430.56` | `401.20` | `1.48945406e3` |
| `rates_screen` (REACLIB) | `636.49` | `271.28` | `370.44` | `95.25` | `167.71` | `412.61` | `226.50` | `-5.82255771e4` |
| `net_eval` (`basic.net` + $J$) | `29591.90` | `10185.83` | `14496.41` | `3738.18` | `3792.84` | `3808.93` | `3709.81` | `4.92535290e4` |
| `atm_table` (2D bicubic) | `2259.16` | `650.79` | `249.26` | `62.66` | `81.50` | `334.22` | `167.47` | `4.60240965e4` |

---

## 4. 1D Stellar Structure, Henyey Solver & Evolution ($N_{\mathrm{zones}} = 1024$)

Full per-size tables ($N_{\mathrm{zones}} \in \{128, 512, 1024\}$) and the `star/test` ($4.0\,M_\odot, Z = 0.02$) trajectory comparison are archived in the [1D Stellar Structure & Evolution Benchmark Report](../design/wave-4-benchmark-report.md).

| Workload ($N_{\mathrm{zones}} = 1024$) | Fortran 1t Stage 2 ($\mu\mathrm{s}$) | Fortran 4t Stage 3 ($\mu\mathrm{s}$) | Safe Rust 1t Stage 2 ($\mu\mathrm{s}$) | Safe Rust 4t Stage 3 ($\mu\mathrm{s}$) | PyO3 1t Stage 0 ($\mu\mathrm{s}$) | JAX `@jit` Stage 2 ($\mu\mathrm{s}$) | JAX Batch/Hybrid Stage 3 ($\mu\mathrm{s}$) | `FlexMESA` Checksum |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star_micro_vars` | `9775.37` | `2553.10` | `8237.61` | `3323.26` | `7116.21` | `172.58` | `53.42` | `3.36128273e4` |
| `star_hydro_eqns` | `11513.95` | `2955.19` | `8673.74` | `3427.20` | `7950.37` | `132.00` | `60.07` | `1.51974833e13` |
| `star_newton_step` | `12262.13` | `3174.67` | `8060.87` | `4098.13` | `254.39` (solve) | `5950.96` | `778.28` | `1.28749954e5` |
| `star_remesh` | `5.75` | `5.77` | `3.22` | `92.19` | `122.61` | `43.07` | `13.16` | `4.21132239e2` |
| `star_evolve_step` | `522751.51` | `163495.99` | `78067.93` | `25951.86` | `43055.64` | `76951.25` | `74358.93` | `1.45655282e4` |

---

## 5. Diffusion, Rotation, HLLC Hydro, Binaries & Asteroseismology ($N_{\mathrm{zones}} = 1024$)

Full per-size tables ($N_{\mathrm{zones}} \in \{128, 512, 1024\}$) are archived in the [Advanced Star, Binaries & Asteroseismology Benchmark Report](../design/wave-5-benchmark-report.md).

| Workload ($N_{\mathrm{zones}} = 1024$) | Fortran 1t Stage 2 ($\mu\mathrm{s}$) | Fortran 4t Stage 3 ($\mu\mathrm{s}$) | Safe Rust 1t Stage 2 ($\mu\mathrm{s}$) | Safe Rust 4t Stage 3 ($\mu\mathrm{s}$) | PyO3 4t Stage 1 ($\mu\mathrm{s}$) | JAX `@jit` Stage 2 ($\mu\mathrm{s}$) | JAX `vmap` Stage 3 ($\mu\mathrm{s}$) | Golden Checksum |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `w5_diff_rot` | `182.87` | `91.83` | `190.48` | `70.41` | `49.62` | `248.95` | `120.91` | `3.12476375e5` |
| `w5_riemann_rsp2` | `63.94` | `46.46` | `28.15` | `7.94` | `33.29` | `204.14` | `89.90` | `-2.72283698e5` |
| `w5_binary_rlof` | `111.57` | `28.29` | `115.63` | `57.79` | `60.69` | `220.69` | `101.77` | `6.27887032e0` |
| `w5_astero_seism` | `5.83` | `5.83` | `7.25` | `7.25` | `7.48` | `84.06` | `26.75` | `7.36379793e1` |
