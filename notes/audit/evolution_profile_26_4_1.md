# Audit A6 (`#168`): Real Evolution Profile, Waste Removal, `strict` vs `fast`, and Batching

This report is rendered deterministically by `scripts/audit/render_evolution_profile_report.py` from raw JSON and CSV captures in `benchmarks/evolution_profile/26.4.1/`. Every timing and trajectory comparison below was recorded on a single quiet host against compiled Fortran MESA `r26.4.1` (`commit c8c2026`, Zenodo `19722306`) at matched thread count.

## 1. Measurement Provenance & Selected Cases

- **Capture Timestamp (UTC)**: `2026-11-19T07:30:22Z` (1-minute host load average `1.88` at start; `5.20` at completion immediately after the `8`-thread `N = 32` worker-pool scaling burst)
- **Host Hardware**: `AMD Ryzen 7 9800X3D 8-Core Processor` (`8` physical / `16` logical cores, `30.15 GiB` RAM, `Linux x86_64`, GPU `NVIDIA GeForce RTX 5080`)
- **Upstream MESA Toolchain**: `r26.4.1` (`commit c8c2026`, Zenodo `19722306`), MESA SDK `x86_64-linux-26.6.1`, `GNU Fortran (GCC) 15.2.0`, `CRMATH`
- **FlexMESA Toolchain**: `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `Python 3.14.0`, `JAX 0.11.0`, `origin/main` base `1131259` (driver commit `d63af21`)
- **Matched Thread Caps**: `OMP_NUM_THREADS=2`, `RAYON_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`

Four `star/test_suite` cases that run to upstream's physical stopping condition on `origin/main` with stock `run_star_extras.f90` hooks were selected to span fixed and Type-2 CO opacities, 8-species and 10-species nuclear networks, Cox and Henyey MLT, stellar winds, convective overshooting, and zero-metallicity pre-MS-through-core-He evolution (`star/test_suite/semiconvection` currently stops at step 197 on `origin/main` awaiting the `#149` Type-2 `a09_co` defaults alignment):

- **`1.3M_ms_high_Z`**: `basic.net` (8 species), Cox MLT, fixed `gs98` opacity, CNO main sequence (`Z = 0.04`) to `log_L_upper_limit`.
- **`low_z`**: `basic.net` (8 species), Henyey MLT, Type-2 CO opacity, Eddington `T(tau)` atmosphere (`Z = 1e-4`) to central H depletion.
- **`high_z`**: `o18_and_ne22.net` (10 species), Type-2 CO opacity, exponential overshooting, Reimers+Vink winds, core H and He depletion.
- **`make_metals`**: `basic.net` (8 species), Population III (`Z = 0`) pre-MS contraction through core H and He depletion.

| Case | Termination (`strict` / MESA) | Steps (`strict` / MESA) | Zones (`init -> final`) | Final Age (`strict` / MESA, yr) | `d_age / age` | Max Track `|d log L|` | Max Track `|d log Teff|` | Max Track `|d log rho_c|` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | `log_L_upper_limit` / `log_L_upper_limit` | `346` / `219` | `807->1175` | `4.6800e+09` / `4.6360e+09` | `+0.95%` | `0.1174` | `0.0137` | `0.5911` |
| `low_z` | `xa_central_lower_limit(h1)` / `xa_central_lower_limit` | `201` / `574` | `773->844` | `1.1810e+10` / `1.1822e+10` | `-0.10%` | `0.0896` | `0.0136` | `0.1608` |
| `high_z` | `xa_central_lower_limit(he4)` / `xa_central_lower_limit` | `328` / `445` | `875->1577` | `3.5236e+07` / `3.5302e+07` | `-0.19%` | `0.1785` | `0.7211` | `1.8064` |
| `make_metals` | `xa_central_lower_limit(he4)` / `xa_central_lower_limit` | `1112` / `511` | `1440->1975` | `2.2133e+08` / `2.2392e+08` | `-1.16%` | `0.1496` | `0.4110` | `3.1405` |

## 2. Per-Step Phase Breakdown & Microphysics Call Counts

Unperturbed wall times (`repeats = 3` median) are measured without instrumentation hooks inside `flexmesa-star`. A second pass (`profile_evolution case --with-phase-profile`) times read-only step probes cloned before each step while verifying a `0-ULP` match on the serialized `.photo` state checksum so phase attribution never perturbs the canonical evolution trajectory.

### 2.1 Share of Step Wall Time by Subsystem (`%`)

| Case | EOS | Opacity | Net + Jacobian | Turbulence | Eqn Assembly | Block Solve | Remesh | Bookkeeping & Copies |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | `12.3%` | `2.2%` | `12.8%` | `43.9%` | `5.3%` | `11.8%` | `11.0%` | `0.7%` |
| `low_z` | `11.7%` | `3.9%` | `20.4%` | `25.8%` | `9.8%` | `23.0%` | `4.8%` | `0.7%` |
| `high_z` | `14.3%` | `1.4%` | `8.1%` | `54.8%` | `3.6%` | `8.0%` | `9.3%` | `0.4%` |
| `make_metals` | `12.1%` | `3.2%` | `19.6%` | `31.2%` | `9.8%` | `14.5%` | `8.2%` | `1.3%` |
| **4-case mean** | **`12.6%`** | **`2.7%`** | **`15.2%`** | **`38.9%`** | **`7.1%`** | **`14.3%`** | **`8.3%`** | **`0.8%`** |

### 2.2 Microphysics & Solver Evaluations per Accepted Step

| Case | Mean Zones | Newton Iters / Step | EOS/Kap/Net Sweeps / Step | Micro Zone Evals / Step | MLT Sweeps / Step | Block Solves / Step | `MicrophysicsCache` Hits / Step |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | `888` | `6.65` | `8.84` | `7857` | `13.40` | `6.65` | `4.56` |
| `low_z` | `786` | `9.00` | `10.05` | `7898` | `15.06` | `9.00` | `5.00` |
| `high_z` | `1123` | `7.98` | `13.30` | `14939` | `16.61` | `7.98` | `3.31` |
| `make_metals` | `1488` | `4.50` | `6.52` | `9695` | `10.97` | `4.50` | `4.46` |

Across the four evolutions, local zone microphysics (`EOS`, `Opacity`, `Net + Jacobian`, and `Turbulence`) accounts for `69.4%` of per-step wall time on average, while coupled Henyey equation assembly, block-tridiagonal linear solves, mesh adjustment, and step bookkeeping account for the remaining `30.6%`. Notably, `MicrophysicsCache` (`#150`) hits on `19.9%` to `40.6%` of full-star `set_hydro_vars` sweeps (skipping `EOS`, `Opacity`, `Neu`, and `Net + Jacobian`), whereas `Turbulence` (`MLT`/`TDC` plus finite-difference face perturbations) executes on every sweep because convective transport depends on neighbor-zone face gradients, making `Turbulence` the largest single subsystem (`38.9%` mean share).

## 3. Real Fortran MESA `26.4.1` vs FlexMESA (`strict`) at Matched Thread Count (`2` Threads)

Each upstream case was compiled cleanly against `$MESA_DIR` (`r26.4.1`) and run via `./rn` with `OMP_NUM_THREADS=2` and `MESA_SKIP_OPTIONAL=t`. Both MESA's internal `runtime (minutes)` evolve timer (excluding one-time table startup) and total `./rn` wall time are reported alongside FlexMESA's post-warmup `evolve_wall_sec`:

| Case | Engine (`2t`) | Steps | Retries | Mean Zones | Evolve Wall (`s`) | Wall / Step (`ms`) | Wall / Zone-Step (`us`) | Zone-Step Speedup (`MESA / FlexMESA`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | FlexMESA `strict` | `346` | `0` | `888` | `54.136` | `156.46` | `176.12` | `4.54x` |
| `1.3M_ms_high_Z` | MESA `26.4.1` (evolve / `./rn`) | `219` | `0` | `956` | `167.40` (`169.46`) | `764.38` | `799.48` | `1.00x` |
| `low_z` | FlexMESA `strict` | `201` | `0` | `786` | `21.143` | `105.19` | `133.92` | `1.93x` |
| `low_z` | MESA `26.4.1` (evolve / `./rn`) | `574` | `0` | `778` | `115.20` (`116.39`) | `200.70` | `258.01` | `1.00x` |
| `high_z` | FlexMESA `strict` | `328` | `5` | `1123` | `136.698` | `416.76` | `371.08` | `2.31x` |
| `high_z` | MESA `26.4.1` (evolve / `./rn`) | `445` | `0` | `1280` | `487.80` (`488.93`) | `1096.18` | `856.38` | `1.00x` |
| `make_metals` | FlexMESA `strict` | `1112` | `4` | `1488` | `131.981` | `118.69` | `79.76` | `5.79x` |
| `make_metals` | MESA `26.4.1` (evolve / `./rn`) | `511` | `4` | `1274` | `300.60` (`348.25`) | `588.26` | `461.67` | `1.00x` |

Per zone-step (`wall_us_per_zone_step`), FlexMESA `strict` runs at `1.93x` to `5.79x` the speed of compiled Fortran MESA `26.4.1` (`2.28x` to `5.45x` on end-to-end evolution wall clock). Differences between per-zone-step speedup and total-run speedup stem directly from timestep (`dt`) controller trajectories (`#149`), because `1.3M_ms_high_Z` and `make_metals` currently take more timesteps than upstream MESA while `low_z` and `high_z` take fewer.

## 4. What `#150` Waste Removal Bought

Issue `#150` removed two recurring sources of per-step waste in `flexmesa-star`: (1) redundant microphysics evaluations across unchanged `(lnRho, lnT, X)` zone states via `MicrophysicsCache`, and (2) unconditional full `StarModel::clone()` heap allocations prior to every timestep via the pre-sized `PrimaryStateBuffer` retry snapshot.

### 4.1 Direct Pre-`#150` vs Post-`#150` Commit Benchmark

| Case | Build Commit | Steps | Median Wall (`s`) | Sample Range (`s`) | Wall / Step (`ms`) | Wall Saved (`%`) | Speedup (`pre / post`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | Pre-`#150` (`af9747e1`) | `346` | `53.8424` | `4.9466` | `155.61` | `0.00%` | `1.00x` |
| `1.3M_ms_high_Z` | Post-`#150` (`3f7fa734`) | `346` | `48.6631` | `0.4205` | `140.64` | `9.62%` | `1.11x` |

### 4.2 `HEAD` Component Breakdown Across All Four Cases

| Case | `MicrophysicsCache` Hits / Step | Est. Micro Wall Saved (`s`) | `StarModel` Heap (`KiB`) | `PrimaryStateBuffer` (`KiB`) | Snapshot Footprint Ratio | Pre-`#150` Clone Cost (`s`) vs Lazy Buffer (`s`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | `4.56` | `9.7663` | `7878.5` | `1945.0` | `4.05x` | `0.2195` vs `0.09239` |
| `low_z` | `5.00` | `4.5635` | `5693.7` | `1397.1` | `4.08x` | `0.1143` vs `0.02325` |
| `high_z` | `3.31` | `10.8924` | `11221.9` | `2709.0` | `4.14x` | `0.4497` vs `0.16348` |
| `make_metals` | `4.46` | `35.6551` | `13159.0` | `3269.3` | `4.03x` | `1.6446` vs `0.35645` |

## 5. What `fast` Math Buys Over `strict` (`#165`) & Trajectory Drift

Per `DECISIONS #23` and `#165`, `NumericsMode::Fast` (platform `std` `f64` intrinsics and hardware `mul_add`) is compared **exclusively** against `NumericsMode::Strict` (`crlibm 1.0beta4` without FMA contraction) and is never compared against MESA:

| Case | Steps (`strict` / `fast`) | Retries (`s` / `f`) | Wall (`strict` / `fast`, `s`) | `us / zone-step` (`s` / `f`) | Zone-Step Speedup (`s / f`) | `d_age / age` | Max Track `|d log L|` | Max Track `|d log Teff|` | Max Track `|d log rho_c|` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | `346` / `346` | `0` / `0` | `54.136` / `39.316` | `176.12` / `127.90` | `1.38x` | `+3.09e-10` | `6.85e-09` | `3.43e-09` | `9.14e-10` |
| `low_z` | `201` / `201` | `0` / `0` | `21.143` / `16.716` | `133.92` / `105.87` | `1.26x` | `-4.48e-11` | `3.29e-11` | `1.63e-10` | `1.26e-09` |
| `high_z` | `328` / `328` | `5` / `5` | `136.698` / `92.005` | `371.08` / `249.75` | `1.49x` | `-2.96e-05` | `4.56e-03` | `4.82e-04` | `2.34e-02` |
| `make_metals` | `1112` / `1083` | `4` / `4` | `131.981` / `101.862` | `79.76` / `62.85` | `1.27x` | `-6.79e-05` | `8.17e-03` | `7.11e-03` | `4.55e-02` |

Switching from `strict` (`crlibm`) to `fast` (`std` + hardware FMA) yields `1.26x` to `1.49x` per zone-step (`1.26x` to `1.49x` wall speedup, reducing evolution runtime by `20.9%` to `32.7%`) while keeping relative lifetime differences within `|d_age / age| <= 6.8e-5`. The measured speedup reflects `crlibm` transcendental calls (`exp`, `ln`, `log10`, `pow`) in reaction screening, nuclear rates, and MLT/opacity log conversions operating alongside FMA-insensitive spline lookups and block-tridiagonal linear solves.

## 6. `N`-Star Worker-Pool Batching vs Stateless Accelerator Estimate (`#166` Input)

### 6.1 Safe Rust `rayon::ThreadPool` Multi-Star Batch Scaling

`profile_evolution batch` evolves `N` independent `low_z` models (`60` steps per track, heterogeneous `mixing_length_alpha` spanning `[1.98, 2.02]` so Newton iteration counts vary realistically across stars) on dedicated `2`-thread and `8`-thread `rayon` worker pools sharing immutable `OnceLock` EOS and opacity tables:

| Pool Threads | Batch `N` Stars | Wall (`s`) | Total Steps | Throughput (`stars / min`) | `zone-steps / s` | Speedup vs `N = 1` | Peak RSS (`MiB`) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2` | `1` | `3.929` | `60` | `15.3` | `11361` | `1.00x` | `2171.1` |
| `2` | `2` | `5.400` | `120` | `22.2` | `16531` | `1.46x` | `2171.1` |
| `2` | `4` | `10.817` | `240` | `22.2` | `16506` | `1.45x` | `2171.1` |
| `2` | `8` | `21.669` | `480` | `22.2` | `16480` | `1.45x` | `2171.1` |
| `2` | `16` | `42.763` | `960` | `22.4` | `16702` | `1.47x` | `2188.3` |
| `8` | `1` | `5.046` | `60` | `11.9` | `8846` | `1.00x` | `2740.2` |
| `8` | `2` | `3.653` | `120` | `32.8` | `24437` | `2.76x` | `2740.2` |
| `8` | `4` | `3.903` | `240` | `61.5` | `45751` | `5.17x` | `2740.2` |
| `8` | `8` | `6.485` | `480` | `74.0` | `55071` | `6.23x` | `2740.2` |
| `8` | `16` | `15.155` | `960` | `63.3` | `47127` | `5.33x` | `2740.2` |
| `8` | `32` | `33.598` | `1920` | `57.1` | `42516` | `4.81x` | `2740.2` |

At `N = 1`, an `8`-thread pool is slightly slower than a `2`-thread pool because intra-star `rayon` loops partition `~785` zones into sub-`100`-zone chunks. Once `N >= 8` independent tracks run concurrently, `rayon` work-stealing assigns whole stars to threads without per-zone synchronization, reaching `74.0` `stars / min` (`55071` `zone-steps / s`, `6.23x` vs `8t` `N = 1` and `4.85x` vs `2t` `N = 1`), while `32` concurrent models hold peak RSS at `2740.2 MiB` via shared `OnceLock` EOS and opacity tables.

### 6.2 Kernel Anchors (`benchmarks/raw/26.4.1/`) & Grid-Scale Accelerator Model

Single-kernel timings from `benchmarks/raw/26.4.1/` anchor what an accelerator (`NVIDIA GeForce RTX 5080`, `jax.vmap` `float64`) achieves relative to `4`-thread Safe Rust CPU per zone:

| Kernel (`benchmarks/raw/26.4.1/`) | Ref Zones | Safe Rust `4t` (`us / zone`) | JAX `vmap` (`us / zone`) | JAX / Rust `4t` Speedup | Execution Notes |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `eos_helm` | `4096` | `0.1348` | `0.0828` | `1.63x` | Host cell-lookup callback (`N1-24`); CPU-bound |
| `eos_skye` | `4096` | `2.3256` | `1.0260` | `2.27x` | Host spline callback (`N1-24`); CPU-bound |
| `eos_4d` | `4096` | `0.2050` | `0.0815` | `2.52x` | Host 4D bicubic callback (`N1-24`); CPU-bound |
| `kap_fixed` | `4096` | `0.1894` | `0.2717` | `0.70x` | Device-cached table interpolation after load |
| `rates_screen` | `4096` | `0.1494` | `0.0780` | `1.92x` | Pure `@jax.jit` analytical screening kernel |
| `net_eval_host_cb` | `4096` | `1.3212` | `1.2720` | `1.04x` | `pure_callback` into `_rust.net_eval` (`N1-25`) |
| `turb_mlt` | `4096` | `0.0735` | `0.0167` | `4.40x` | Pure `@jax.jit` local cubic/quartic MLT solver |
| `turb_tdc` | `4096` | `0.6266` | `0.0956` | `6.55x` | Iterative root + dual overhead vs Rust IFT |
| `colors_sed` | `1024` | `1.5820` | `0.6733` | `2.35x` | Pure `@jax.jit` 3D SED + filter integration |
| `mtx_tridiag` | `4096` | `0.0790` | `0.6825` | `0.12x` | `O(N log N)` GPU PCR vs `O(N)` CPU `bcyclic` |

Combining the measured `69.4%` stateless microphysics share with Amdahl's law fixes an upper bound of `3.27x` for any hybrid CPU-Henyey + GPU-microphysics step, even before PCIe transfers (`24 GB/s`), XLA launch latency (`80 us` per Newton iteration), and heterogeneous adaptive-timestep lockstep padding (`0.72` utilization). By contrast, standalone stateless grid evaluations (`SED` photometry, screening, analytical EOS, MLT surrogates) avoid the Henyey solver loop entirely:

| Grid Stars `N` | Total Zones | `8t` Rust Pool Step (`s`) | Hybrid GPU-Micro Step (`s`) | Hybrid / `8t` Rust Speedup | Stateless-Only `8t` Rust (`s`) | Stateless-Only GPU (`s`) | Stateless GPU Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `888` | `0.0323` | `0.0288` | `1.12x` | `0.0224` | `0.01819` | `1.23x` |
| `8` | `7107` | `0.2582` | `0.2834` | `0.91x` | `0.1792` | `0.14492` | `1.24x` |
| `64` | `56858` | `2.0659` | `2.2611` | `0.91x` | `1.4339` | `1.15881` | `1.24x` |
| `256` | `227430` | `8.2636` | `9.0417` | `0.91x` | `5.7358` | `4.63499` | `1.24x` |
| `1024` | `909720` | `33.0543` | `36.1641` | `0.91x` | `22.9432` | `18.53973` | `1.24x` |

### 6.3 Concrete Recommendations for `#166`

1. **Population & Grid Stellar Evolution (`N >= 8` tracks)**: Run independent `StarModel` trajectories across a Safe Rust `rayon` worker pool rather than padding heterogeneous meshes (`nz`) and adaptive timesteps (`dt`) into lockstep GPU batches. Shared `OnceLock` physics tables keep memory overhead small across dozens of concurrent stars, and each worker runs asynchronously with zero host-device synchronization.
2. **Differentiable Stellar Structure & Evolution (Post-`#143`)**: Keep forward evolution in Safe Rust and expose Implicit Function Theorem (`@jax.custom_vjp` / `@jax.custom_jvp`) step wrappers solved with CPU `bcyclic` (`mtx` `solver_mode="rust_ffi"`) plus `#150` `.photo` binomial checkpointing. Measured GPU `PCR` at `N = 4096` is slower than `4`-thread CPU `bcyclic` because `PCR` performs `O(N log2 N)` dense block multiplications in `float64`.
3. **Where Native Pure-XLA JAX Kernels Pay (`#166`)**: Prioritize pure `@jax.jit` / `jax.vmap` kernels on stateless, fixed-shape batch workloads that run outside the Henyey iteration loop, specifically bolometric and filter photometry (`flexmesa.colors`), analytical screening and neutrino emission (`flexmesa.rates`, `flexmesa.neu`), local `set_mlt_jax` / `T(tau)` boundary evaluations (`flexmesa.turb`, `flexmesa.atm`), and batch inference likelihoods over `>= 10^4` evaluation points. Porting stateful 4D `eos` / `kap` bicubic tables or stiff `net` Rosenbrock solvers to pure GPU solely to accelerate 1D Henyey steps is bounded by the `<= 3.27x` Amdahl ceiling.

## 7. What Was Not Verified

- **Case breadth**: Measurements cover four `star/test_suite` cases (`1.3M_ms_high_Z`, `low_z`, `high_z`, `make_metals`) on a single `x86_64` workstation; `semiconvection` (blocked at step 197 on `origin/main` pending `#149`), late burning with `approx21.net`, hydrodynamic `rsp2` pulsations, `binary` evolutions, and `astero` simplex runs were not profiled in this campaign.
- **`aarch64` & multi-socket scaling**: `strict` vs `fast` ratios and `rayon` worker-pool curves were measured on one `8`-core / `16`-thread `x86_64` CPU under WSL2; Apple Silicon (`aarch64`) and high-core-count server NUMA scaling were not measured.
- **`#150` git A/B scope**: Direct pre-`#150` (`af9747e`) vs post-`#150` (`3f7fa734`) binary timing was run on `1.3M_ms_high_Z`; the remaining three cases report `HEAD` component measurements (`MicrophysicsCache` hit rates and `PrimaryStateBuffer` vs `StarModel::clone` timings).
- **Grid-scale GPU evolution**: Because `src/flexmesa/star/` JAX helpers are frozen (`JAX_ENGINE_STATUS = "frozen-unverified-until-#143"`), the `N = 1..1024` hybrid GPU numbers in Section 6.2 come from an analytical Amdahl + PCIe + lockstep model anchored to `benchmarks/raw/26.4.1/` single-kernel measurements rather than an executed GPU Henyey evolution engine.
