# Wave 4.5 Four-Way Benchmark & `star/test` Pre-Audit Report

!!! warning "Pre-Audit Benchmark Record (`0b2e6bb`, Predates `51e1adc` & MESA `26.4.1`)"
    1. Section 2 Fortran timings predate commit `51e1adc` (`need_to_setvars = .true.`), so Stage 2/3 Fortran numbers omit full microphysics setup (`#146`), and `star_evolve_step` compares unequal work (`dt = 1e2 s` in Fortran vs. `base.dt` in Rust without matched iteration counts).
    2. Fortran and `FlexMESA` Stage 2 checksums disagree by `1.2e-4` to `169x` (tabulated below).
    3. The pre-audit `star/test` table below used unrecorded local `0b2e6bb` values (`2.374600`) and tolerances looser than the 11-step MESA `26.4.1` evolution signal (`3.89e-4` dex on `log_L`); see Section 1.2 for the committed MESA `26.4.1` reference values.

Wave 4.5 evaluates the 1D stellar structure engine (`crates/flexmesa-star` and `flexmesa.star`) across five benchmark workloads (`star_micro_vars`, `star_hydro_eqns`, `star_newton_step`, `star_remesh`, `star_evolve_step`) and the `star/test` ($4.0\,M_\odot, Z = 0.02$) trajectory.

![Wave 4.5 pre-audit benchmark comparison and coupled evolutionary step scaling](../assets/figures/wave4_benchmark_staircase.png)

---

## 1. `star/test` Trajectory Comparison (`4.0 Msun, Z = 0.02` ZAMS to `1.00e5 yr`)

### 1.1 Pre-Audit Local Record (`0b2e6bb`, Uncommitted Reference)

Evolving `StarModel.from_zams(4.0, 0.02)` to `max_age = 1.00e5 yr` (11 steps, `do_remesh=False, op_split_burn=False`) in pre-audit runs yielded the table below. The quoted reference column was an uncommitted local `0b2e6bb` run (`log_L = 2.374600` here vs. `2.374853` in an audit `0b2e6bb` run, which puts the pre-audit `FlexMESA` `2.373933` `9.20e-4` dex away), and over 11 steps MESA's own `log_L` changes by only `3.89e-4` dex, `log_R` by `1.59e-4` dex, and `log_Teff` by `1.79e-5` dex, so a `1e-3` dex gate is wider than the evolutionary signal on those channels:

| Observable | Pre-Audit Quoted Value | `FlexMESA` (Pre-Audit `0b2e6bb`) | Quoted Abs. Dev. | Pre-Audit Loose Bound | Note |
| :--- | ---: | ---: | ---: | ---: | :---: |
| $\log_{10}(T_c/\mathrm{K})$ (`log_center_T`) | `7.412400` | `7.412298` | `1.02e-04` dex | `<= 1.0e-03` dex | Ref. uncommitted |
| $\log_{10}(\rho_c/\mathrm{g\,cm^{-3}})$ (`log_center_Rho`) | `1.436800` | `1.436338` | `4.62e-04` dex | `<= 1.0e-03` dex | Ref. uncommitted |
| $\log_{10}(L/L_\odot)$ (`log_L`) | `2.374600` | `2.373933` | `6.67e-04` dex | `<= 1.0e-03` dex | Signal `3.89e-4` |
| $\log_{10}(R/R_\odot)$ (`log_R`) | `0.370300` | `0.370340` | `4.00e-05` dex | `<= 1.0e-03` dex | Signal `1.59e-4` |
| $\log_{10}(T_{\mathrm{eff}}/\mathrm{K})$ (`log_Teff`) | `4.169800` | `4.169640` | `1.60e-04` dex | `<= 1.0e-03` dex | Signal `1.79e-5` |
| Central ${}^1\mathrm{H}$ mass fraction (`center_h1`) | `0.696725` | `0.696732` | `7.00e-06` | `<= 5.0e-05` | Signal `2.54e-4` |

### 1.2 Committed MESA `26.4.1` `star/test` Reference & Measured Divergence

Under `#141` and `#143`, the real MESA `26.4.1` `star/test/LOGS/history.data` fixture (`tests/fixtures/golden/star_test/history.data`) is committed in git (`model_number` 1 at `3125 yr` -> `model_number` 11 at `1.00e5 yr`):

- **Committed MESA `26.4.1` Step-11 Values**: `log_L = 2.374853`, `log_Teff = 4.169854`, `log_R = 0.370371`, `log_cntr_T = 7.412483`, `log_cntr_Rho = 1.437448`, `center_h1 = 0.6967304`.
- **Current Regression Status (`tests/test_wave4_bench.py`)**: MESA `26.4.1` `star/test` remeshes `2464 -> 992` zones at step 2; `FlexMESA` runs with `do_remesh=False` currently diverge by `1.11e-3` dex (`> 1.0e-3`) on `log_cntr_Rho` at step 11 (`xfail(strict=True)` tracked under `#153`).

### 1.3 Fortran vs. Safe Rust Checksum Discrepancies ($N_{\mathrm{zones}} = 128$, Pre-Audit)

| Workload ($N_{\mathrm{zones}} = 128$) | Fortran Checksum (§2) | Safe Rust Checksum (§3) | Relative / Factor Difference | Cause |
| :--- | ---: | ---: | ---: | :--- |
| `star_micro_vars` | `4.18189385e+03` | `4.18240352e+03` | `1.22e-4` | EOS/opacity or blend difference (`#146`) |
| `star_hydro_eqns` | `7.64217121e+06` | `7.64073344e+06` | `1.88e-4` | Structure residual mismatch (`#146`) |
| `star_newton_step` | `8.89749394e+04` | `1.11807589e+06` | `12.6x` factor | Different linear solve digest / state (`#146`) |
| `star_remesh` | `7.13415890e+04` | `4.21492690e+02` | `169x` factor | Fortran sums `set_mesh_function_data`; Rust sums normalized `g_k` |
| `star_evolve_step` | `1.80069305e+03` | `1.80055528e+03` | `7.65e-5` | Unequal `dt` (`1e2 s` vs. `base.dt`) and sub-stage differences |

---

## 2. Upstream Fortran MESA Static Archive Benchmark (`benchmarks/fortran_mesa/run_wave4_bench`)

Compiled with `gfortran -O2 -fopenmp` against upstream Fortran MESA `libstar.a`, `libstar_data.a`, `libatm.a`, `libnet.a`, `librates.a`, `libkap.a`, `libeos.a`, `libcolors.a`, `libionization.a`, `libturb.a`, `libneu.a`, `libinterp_2d.a`, `libnum.a`, `libmtx.a`, `libinterp_1d.a`, `libauto_diff.a`, `libchem.a`, `libmath.a`, `libutils.a`, `libconst.a`, and `libforum.a` (`OMP_NUM_THREADS=4` for Stage 3):

| Workload | $N_{\mathrm{zones}}$ | Stage 0 ($\mu\mathrm{s}$) | Stage 1 ($\mu\mathrm{s}$) | Stage 2 ($\mu\mathrm{s}$) | Stage 3 4T OpenMP ($\mu\mathrm{s}$) | Max Rel Err | Checksum |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star_micro_vars` | 128 | `3706.77 +/- 96.92` | `3689.29 +/- 24.69` | `1188.80 +/- 22.40` | `347.49 +/- 67.87` | `0.000e+00` | `4.18189385e+03` |
| `star_hydro_eqns` | 128 | `1441.27 +/- 34.80` | `1446.35 +/- 22.73` | `1434.95 +/- 326.62` | `369.29 +/- 17.60` | `0.000e+00` | `7.64217121e+06` |
| `star_newton_step` | 128 | `1546.25 +/- 79.52` | `1551.45 +/- 38.75` | `1551.50 +/- 25.23` | `461.61 +/- 81.74` | `0.000e+00` | `8.89749394e+04` |
| `star_remesh` | 128 | `0.89 +/- 0.03` | `0.97 +/- 0.01` | `0.86 +/- 0.00` | `0.86 +/- 0.01` | `0.000e+00` | `7.13415890e+04` |
| `star_evolve_step` | 128 | `58851.46 +/- 2599.62` | `57862.12 +/- 1517.79` | `58728.83 +/- 1348.73` | `16032.42 +/- 425.53` | `0.000e+00` | `1.80069305e+03` |
| `star_micro_vars` | 512 | `15003.67 +/- 672.34` | `15332.68 +/- 499.05` | `4944.77 +/- 205.87` | `1561.26 +/- 38.04` | `0.000e+00` | `1.67904925e+04` |
| `star_hydro_eqns` | 512 | `5837.85 +/- 84.34` | `5786.23 +/- 87.10` | `5795.04 +/- 62.30` | `1463.12 +/- 10.10` | `0.000e+00` | `3.22884522e+07` |
| `star_newton_step` | 512 | `6310.70 +/- 321.12` | `6079.07 +/- 48.84` | `6034.93 +/- 34.55` | `1616.79 +/- 33.71` | `0.000e+00` | `7.68358489e+04` |
| `star_remesh` | 512 | `3.03 +/- 0.02` | `3.30 +/- 0.01` | `2.99 +/- 0.02` | `2.99 +/- 0.24` | `0.000e+00` | `2.86562751e+05` |
| `star_evolve_step` | 512 | `210876.66 +/- 1807.81` | `208301.76 +/- 2128.70` | `209262.16 +/- 766.00` | `55481.71 +/- 430.74` | `0.000e+00` | `7.26765886e+03` |
| `star_micro_vars` | 1024 | `30477.49 +/- 653.53` | `30119.99 +/- 1180.72` | `9775.37 +/- 337.41` | `2553.10 +/- 57.87` | `0.000e+00` | `3.36107957e+04` |
| `star_hydro_eqns` | 1024 | `11657.26 +/- 429.36` | `11585.97 +/- 480.19` | `11513.95 +/- 172.91` | `2955.19 +/- 24.52` | `0.000e+00` | `6.71116062e+07` |
| `star_newton_step` | 1024 | `12319.50 +/- 123.16` | `12148.14 +/- 136.46` | `12262.13 +/- 56.90` | `3174.67 +/- 15.17` | `0.000e+00` | `1.15383021e+05` |
| `star_remesh` | 1024 | `5.78 +/- 0.02` | `6.37 +/- 0.01` | `5.75 +/- 0.01` | `5.77 +/- 0.01` | `0.000e+00` | `5.73660892e+05` |
| `star_evolve_step` | 1024 | `463034.84 +/- 18420.10` | `514402.36 +/- 21310.40` | `522751.51 +/- 19840.20` | `163495.99 +/- 6880.83` | `0.000e+00` | `1.45656370e+04` |

---

## 3. Safe Rust Benchmark Suite (`crates/flexmesa-bench`)

| Workload | $N_{\mathrm{zones}}$ | Stage 0 ($\mu\mathrm{s}$) | Stage 1 ($\mu\mathrm{s}$) | Stage 2 ($\mu\mathrm{s}$) | Stage 3 4T `rayon` ($\mu\mathrm{s}$) | Max Rel Err | Checksum |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star_micro_vars` | 128 | `5156.01 +/- 786.94` | `5583.06 +/- 856.05` | `5574.57 +/- 905.13` | `1491.33 +/- 602.12` | `0.000e+00` | `4.18240352e+03` |
| `star_hydro_eqns` | 128 | `6301.13 +/- 1660.66` | `5914.89 +/- 629.79` | `5706.41 +/- 1344.98` | `909.76 +/- 1023.97` | `0.000e+00` | `7.64073344e+06` |
| `star_newton_step` | 128 | `6697.04 +/- 1114.55` | `5650.81 +/- 1640.12` | `5948.65 +/- 895.27` | `2943.74 +/- 2604.57` | `2.520e-13` | `1.11807589e+06` |
| `star_remesh` | 128 | `7846.75 +/- 2361.81` | `1.62 +/- 0.06` | `0.45 +/- 0.02` | `86.02 +/- 1.08` | `0.000e+00` | `4.21492690e+02` |
| `star_evolve_step` | 128 | `29257.65 +/- 3499.80` | `47450.01 +/- 1629.98` | `47152.88 +/- 4864.87` | `7739.38 +/- 1404.54` | `4.833e-12` | `1.80055528e+03` |
| `star_micro_vars` | 512 | `6883.43 +/- 824.85` | `7044.78 +/- 2707.61` | `6874.07 +/- 2143.63` | `2102.89 +/- 569.78` | `0.000e+00` | `1.67915615e+04` |
| `star_hydro_eqns` | 512 | `8065.69 +/- 1848.23` | `7375.02 +/- 1149.54` | `7353.01 +/- 1087.30` | `2136.41 +/- 377.27` | `0.000e+00` | `3.22885198e+07` |
| `star_newton_step` | 512 | `6934.45 +/- 1149.48` | `7201.52 +/- 1663.58` | `7694.95 +/- 1394.01` | `2555.69 +/- 2032.31` | `6.843e-12` | `8.54959626e+04` |
| `star_remesh` | 512 | `11266.78 +/- 1633.11` | `5.21 +/- 0.05` | `1.63 +/- 0.00` | `82.72 +/- 14.35` | `0.000e+00` | `4.21135939e+02` |
| `star_evolve_step` | 512 | `39065.24 +/- 6542.38` | `72023.86 +/- 8439.10` | `66742.64 +/- 7790.05` | `15093.50 +/- 1923.77` | `5.872e-12` | `7.26757409e+03` |
| `star_micro_vars` | 1024 | `8921.67 +/- 2405.58` | `7599.99 +/- 1575.56` | `8237.61 +/- 2012.36` | `3323.26 +/- 315.30` | `0.000e+00` | `3.36128273e+04` |
| `star_hydro_eqns` | 1024 | `8607.34 +/- 1907.87` | `8665.29 +/- 2050.08` | `8673.74 +/- 1639.94` | `3427.20 +/- 248.24` | `0.000e+00` | `6.71126771e+07` |
| `star_newton_step` | 1024 | `7863.99 +/- 1661.81` | `7901.26 +/- 1555.32` | `8060.87 +/- 1861.55` | `4098.13 +/- 310.78` | `1.192e-10` | `1.28749954e+05` |
| `star_remesh` | 1024 | `15972.32 +/- 2443.17` | `9.46 +/- 0.89` | `3.22 +/- 0.03` | `92.19 +/- 4.74` | `0.000e+00` | `4.21132239e+02` |
| `star_evolve_step` | 1024 | `45640.69 +/- 6516.90` | `73415.70 +/- 10719.02` | `78067.93 +/- 8556.48` | `25951.86 +/- 2290.49` | `2.377e-11` | `1.45655282e+04` |

---

## 4. Python / PyO3 (`flexmesa._rust`) & JAX (`float64`) Benchmark Suite (`src/flexmesa/bench/kernels_wave4.py`)

| Workload | $N_{\mathrm{zones}}$ | Stage 0 PyO3 1T ($\mu\mathrm{s}$) | Stage 1 PyO3 4T ($\mu\mathrm{s}$) | Stage 2 JAX JIT ($\mu\mathrm{s}$) [compile $\mu\mathrm{s}$] | Stage 3 JAX Batch / Hybrid ($\mu\mathrm{s}$) | Max Rel Err | Checksum |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star_micro_vars` | 128 | `5776.50 +/- 1648.42` | `6139.78 +/- 1491.87` | `82.57 +/- 33.24 [139755]` | `32.90 +/- 24.49` | `0.000e+00` | `4.18240352e+03` |
| `star_hydro_eqns` | 128 | `5625.23 +/- 756.64` | `5528.98 +/- 724.46` | `58.75 +/- 1.20 [101787]` | `37.25 +/- 2.71` | `0.000e+00` | `7.64073344e+06` |
| `star_newton_step` | 128 | `32.21 +/- 0.63` | `937.14 +/- 2702.14` | `960.55 +/- 110.18 [239463]` | `473.28 +/- 2138.69` | `5.881e-13` | `1.11807597e+06` |
| `star_remesh` | 128 | `120.15 +/- 34.74` | `3.44 +/- 1.25` | `3.20 +/- 0.18 [4]` | `13.51 +/- 2.49` | `0.000e+00` | `4.21492690e+02` |
| `star_evolve_step` | 128 | `28902.36 +/- 2994.22` | `48571.24 +/- 6161.90` | `46080.58 +/- 5358.73 [0]` | `47218.96 +/- 6429.20` | `1.403e-12` | `1.80055528e+03` |
| `star_micro_vars` | 512 | `5302.72 +/- 1315.47` | `6313.51 +/- 1743.65` | `173.23 +/- 52.36 [123110]` | `43.56 +/- 2.22` | `0.000e+00` | `1.67915615e+04` |
| `star_hydro_eqns` | 512 | `6152.77 +/- 1902.99` | `6033.38 +/- 1423.70` | `91.62 +/- 1.36 [95865]` | `45.11 +/- 17.30` | `0.000e+00` | `3.22885198e+07` |
| `star_newton_step` | 512 | `126.28 +/- 2.08` | `282.04 +/- 150.31` | `2884.35 +/- 45.54 [201960]` | `646.51 +/- 44.54` | `4.896e-11` | `8.54959627e+04` |
| `star_remesh` | 512 | `119.51 +/- 4.67` | `46.22 +/- 15.02` | `42.38 +/- 1.60 [45]` | `11.05 +/- 3.72` | `0.000e+00` | `4.21135939e+02` |
| `star_evolve_step` | 512 | `38811.58 +/- 5312.92` | `61388.97 +/- 7468.76` | `59854.79 +/- 4514.98 [0]` | `69133.42 +/- 5076.00` | `6.968e-12` | `7.26757409e+03` |
| `star_micro_vars` | 1024 | `7116.21 +/- 1630.83` | `8228.40 +/- 3837.88` | `172.58 +/- 82.65 [120700]` | `53.42 +/- 11.89` | `0.000e+00` | `3.36128273e+04` |
| `star_hydro_eqns` | 1024 | `7950.37 +/- 2296.38` | `6810.54 +/- 1625.63` | `132.00 +/- 71.03 [99615]` | `60.07 +/- 33.83` | `0.000e+00` | `6.71126771e+07` |
| `star_newton_step` | 1024 | `254.39 +/- 45.50` | `640.39 +/- 467.58` | `5950.96 +/- 176.59 [224299]` | `778.28 +/- 344.97` | `1.618e-10` | `1.28749954e+05` |
| `star_remesh` | 1024 | `122.61 +/- 24.01` | `52.19 +/- 30.39` | `43.07 +/- 5.97 [56]` | `13.16 +/- 1.31` | `0.000e+00` | `4.21132239e+02` |
| `star_evolve_step` | 1024 | `43055.64 +/- 4298.30` | `68839.63 +/- 8082.55` | `76951.25 +/- 12024.65 [0]` | `74358.93 +/- 6195.38` | `3.287e-11` | `1.45655282e+04` |

---

## 5. Wave 4.5 Cross-Module Refinements

1. **$C^\infty$-Smooth `HELM` Free-Energy Vertex Evaluation & Superseded Trajectory Patches (`crates/flexmesa-eos`)**:
   - Replaced the discontinuous low-degeneracy branch in `HelmTable::analytical_vertex` with a smooth analytical Fermi-Dirac expansion across all $\eta_{\mathrm{ele}}$; pre-audit binary trajectory patches (`crates/flexmesa-eos/data/helm_patches.bin` and `eos_tables_4d_patches.bin`) were later removed under `DECISIONS #13` when full release-hosted `data-v1` tables replaced local repository patches.
2. **Hydrostatic Scale-Height Floor & Guarded MLT Evaluation (`crates/flexmesa-star/src/hydro_vars.rs`)**:
   - Added upstream MESA's `alt_scale_height` cap $H_P = \min(P / (\rho g), \sqrt{P / (G \rho^2)})$ so central faces ($r \to 0$) maintain finite mixing-length scale heights, and bypassed redundant MLT cubic solves when $\alpha_{\mathrm{MLT}} \le 0$ or $\nabla_{\mathrm{rad}} \le \nabla_{\mathrm{ad}}$.
3. **5-Point Boxcar-Smoothed `varcontrol` & `max_age` Clamping (`crates/flexmesa-star/src/timestep.rs`, `evolve.rs`)**:
   - Matched upstream MESA's `eval_varcontrol` 5-point sliding-window smoothing across zones $k - 2 \dots k + 2$ (`varcontrol_Need_to_smooth = 5`), 1st-order initial controller fallback on step 1 (`dt_prev <= 0`), and exact final-step clamping to `max_age`.
