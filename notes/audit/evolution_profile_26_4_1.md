# Audit A6 (`#168`): Real Evolution Profile, Waste Removal, `strict` vs `fast`, and Batching

This report is rendered deterministically by `scripts/audit/render_evolution_profile_report.py` from raw JSON and CSV captures in `benchmarks/evolution_profile/26.4.1/`. Section 3 compares with compiled Fortran MESA `r26.4.1` (`commit c8c2026`, Zenodo `19722306`) under a matched timing protocol; Sections 2, 4, 5, and 6 compare FlexMESA with itself.

**Every FlexMESA run in this report is a diagnostic run.** `4` of the `4` `strict` records carry `allow_unsupported_controls = true` (`1.3M_ms_high_Z`: `diagnostic (2 controls ignored)`, `semiconvection`: `diagnostic (2 controls ignored)`, `low_z`: `diagnostic (3 controls ignored)`, `make_metals`: `diagnostic (2 controls ignored)`), so the runs continued past upstream controls FlexMESA does not implement and did not execute every control requested by the inlists (`DECISIONS #26`). All four cases are declared known gaps in `benchmarks/history_harness/known_gaps.toml` and none counts as passing under `#143`. Although `1.3M_ms_high_Z` and `semiconvection` match MESA's step counts identically at `e0e987fc`, track differences in Section 1 remain non-zero; these timings profile FlexMESA at `e0e987fc` rather than a passing `#143` acceptance run.

## 1. Measurement Provenance & Selected Cases

- **Capture Timestamp (UTC)**: `2026-11-20T20:23:14Z` (1-minute host load average `2.80` at start; `6.32` at completion immediately after the `8`-thread `N = 32` worker-pool burst; serialized with an exclusive host timing lock)
- **Host Hardware**: `AMD Ryzen 7 9800X3D 8-Core Processor` (`8` physical / `16` logical cores, `30.15 GiB` RAM, `Linux x86_64`, GPU `NVIDIA GeForce RTX 5080`)
- **Upstream MESA Toolchain**: `r26.4.1` (`commit c8c2026`, Zenodo `19722306`), MESA SDK `x86_64-linux-26.6.1`, `GNU Fortran (GCC) 15.2.0`, `CRMATH`, flags `-O2 -ftree-vectorize -fopenmp -fprotect-parens -fno-sign-zero -finit-real=snan`
- **FlexMESA Toolchain**: `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `Python 3.14.0`, `JAX 0.11.0`, `origin/main` `e0e987fc` (driver commit `e0e987fc`), profile `release (opt-level=3, lto=fat, codegen-units=1)`
- **Matched Timing Protocol**: `3` repeats per case on both MESA `26.4.1` and FlexMESA (`strict` and `fast`), reporting median and `[min, max]` spread, with normal history output written on every timed repeat and matched thread caps `OMP_NUM_THREADS=2`, `RAYON_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `MKL_NUM_THREADS=2`

Four `star/test_suite` cases whose diagnostic runs reach upstream's termination code on `origin/main` (`e0e987fc`), with stock `run_star_extras.f90` evolutionary loops, were selected to span fixed and Type-2 CO opacities, 8-species and 25-species nuclear networks, Cox and Henyey MLT, Ledoux semiconvection with predictive mixing, and zero-metallicity pre-MS-through-core-He evolution (including the two exact step-matching cases `1.3M_ms_high_Z` and `semiconvection`):

- **`1.3M_ms_high_Z`**: `basic.net` (8 species), Cox MLT, fixed `gs98` opacity, CNO main sequence (`Z = 0.04`) to `log_L_upper_limit`.
- **`semiconvection`**: `pp_and_cno_extras.net` (25 species), Henyey MLT, Type-2 `a09_co` opacity, Ledoux criterion with Langer (1985) semiconvection and predictive mixing (`1.5 Msun`, `Z = 0.02`) to `center_h1 <= 0.38`.
- **`low_z`**: `basic.net` (8 species), Henyey MLT, Type-2 CO opacity, Eddington `T(tau)` atmosphere (`Z = 1e-4`) to central H depletion.
- **`make_metals`**: `basic.net` (8 species), Population III (`Z = 0`) pre-MS contraction through core H and He depletion.

| Case | FlexMESA Run Mode | Termination (`strict` / MESA) | Steps (`strict` / MESA) | Zones (`init -> final`) | Final Age (`strict` / MESA, yr) | `d_age / age` | Max Track `|d log L|` | Max Track `|d log Teff|` | Max Track `|d log rho_c|` |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | `diagnostic (2 controls ignored)` | `log_L_upper_limit` / `log_L_upper_limit` | `219` / `219` | `807->1220` | `4.6356e+09` / `4.6360e+09` | `-0.01%` | `0.0010` | `0.0003` | `0.0056` |
| `semiconvection` | `diagnostic (2 controls ignored)` | `xa_central_lower_limit` / `xa_central_lower_limit` | `293` / `293` | `804->1667` | `1.3650e+09` / `1.3650e+09` | `+0.00%` | `0.0199` | `0.0023` | `0.0258` |
| `low_z` | `diagnostic (3 controls ignored)` | `xa_central_lower_limit` / `xa_central_lower_limit` | `181` / `574` | `773->886` | `1.1823e+10` / `1.1822e+10` | `+0.01%` | `0.0889` | `0.0131` | `0.1708` |
| `make_metals` | `diagnostic (2 controls ignored)` | `xa_central_lower_limit` / `xa_central_lower_limit` | `477` / `511` | `754->2050` | `2.2390e+08` / `2.2392e+08` | `-0.01%` | `0.1536` | `0.4055` | `3.1060` |

## 2. Per-Step Phase Breakdown & Microphysics Call Counts

Unperturbed wall times (`repeats = 3` median, with full `history.data` written on every repeat) are measured without instrumentation hooks inside `flexmesa-star`. A second pass (`profile_evolution case --with-phase-profile`) times read-only step probes cloned before each step while verifying a `0-ULP` match on the serialized `.photo` state checksum so phase attribution never perturbs the canonical evolution trajectory.

The shares below are a rescaled bottom-up model, not a direct instrumented wall-clock measurement. Each kernel is timed once per step on a copy of the pre-step model and multiplied by the call counts recorded during the real step. The sum of those products exceeds the measured evolve wall time in every case, and the shares are then rescaled proportionally to sum to `100%`:

| Case | Run Mode | Measured Evolve Wall (`s`) | Sum of Kernel Estimates (`s`) | Overshoot |
| :--- | :--- | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | `diagnostic (2 controls ignored)` | `59.61` | `79.33` | `+33.1%` |
| `semiconvection` | `diagnostic (2 controls ignored)` | `169.87` | `181.13` | `+6.6%` |
| `low_z` | `diagnostic (3 controls ignored)` | `26.76` | `30.26` | `+13.1%` |
| `make_metals` | `diagnostic (2 controls ignored)` | `103.89` | `150.37` | `+44.7%` |

The turbulence bucket also includes the Brunt, surface boundary, and `eps_grav` probes, which the real step does not call on every sweep. Figures derived from these shares later in the report (the stateless microphysics share and the Amdahl ceiling in Section 6) inherit this uncertainty.

### 2.1 Modelled Share of Step Wall Time by Subsystem (`%`, rescaled to `100%`)

| Case | EOS | Opacity | Net + Jacobian | Turbulence | Eqn Assembly | Block Solve | Remesh | Bookkeeping & Copies |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | `10.2%` | `1.8%` | `16.1%` | `52.9%` | `3.1%` | `10.2%` | `5.1%` | `0.5%` |
| `semiconvection` | `4.3%` | `1.1%` | `24.5%` | `19.9%` | `10.0%` | `36.0%` | `3.4%` | `0.7%` |
| `low_z` | `10.0%` | `3.3%` | `27.6%` | `30.8%` | `6.2%` | `17.2%` | `4.1%` | `0.8%` |
| `make_metals` | `9.2%` | `2.0%` | `20.3%` | `46.1%` | `5.3%` | `10.0%` | `6.4%` | `0.7%` |
| **4-case mean** | **`8.4%`** | **`2.1%`** | **`22.1%`** | **`37.4%`** | **`6.2%`** | **`18.4%`** | **`4.8%`** | **`0.7%`** |

### 2.2 Microphysics & Solver Evaluations per Accepted Step

| Case | Mean Zones | Newton Iters / Step | EOS/Kap/Net Sweeps / Step | Micro Zone Evals / Step | MLT Sweeps / Step | Block Solves / Step | `MicrophysicsCache` Hits / Step |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | `963` | `9.63` | `12.23` | `11778` | `19.15` | `9.63` | `6.92` |
| `semiconvection` | `1629` | `6.80` | `8.19` | `13345` | `15.13` | `6.80` | `6.94` |
| `low_z` | `802` | `9.25` | `11.56` | `9272` | `18.54` | `9.25` | `6.98` |
| `make_metals` | `1375` | `7.04` | `9.26` | `12734` | `15.74` | `7.04` | `6.48` |

In the rescaled bottom-up model across these four evolutions, local zone microphysics (`EOS`, `Opacity`, `Net + Jacobian`, and `Turbulence`) accounts for `70.0%` of per-step wall time on average, while coupled Henyey equation assembly, block-tridiagonal linear solves, mesh adjustment, and step bookkeeping account for `30.0%`. Directly counted `MicrophysicsCache` (`#150`) hits occur on `36.1%` to `45.8%` of full-star `set_hydro_vars` sweeps (skipping `EOS`, `Opacity`, `Neu`, and `Net + Jacobian`), whereas `Turbulence` (`MLT`/`TDC` plus face perturbations) executes on every sweep because convective transport depends on neighbor-zone face gradients (`37.4%` modelled mean share).

## 3. Matched-Protocol Comparison: Real Fortran MESA `26.4.1` vs FlexMESA (`strict`) at `2` Threads

Each upstream case was compiled once against `$MESA_DIR` (`r26.4.1`) and timed over `3` clean `./rn` repeats with `OMP_NUM_THREADS=2` and `MESA_SKIP_OPTIONAL=t` under the same exclusive host lock as FlexMESA (`3` repeats at `RAYON_NUM_THREADS=2`, writing full `history.data` and `final.mod` on every repeat). Exact cumulative Newton solver iterations (including retries) are read from MESA's `testhub.yml` (`s% total_num_solver_iterations`) and FlexMESA's `model.total_num_solver_iterations`. For MESA, both internal `runtime (minutes)` (excluding table startup, quantized to `0.6 s`) and full `./rn` wall time are recorded:

| Case | Engine & Run Mode (`2t`) | Steps | Retries | Newton Iters | Mean Zones | Evolve Wall Median `[min, max]` (`s`) | `ms / step` | `us / zone-step` | `ms / Newton-iter` | `MESA / FlexMESA` (`us/zs` ; `ms/iter`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | FlexMESA `strict`, `diagnostic (2 controls ignored)` | `219` | `0` | `2108` | `963` | `59.607` `[59.586, 59.671]` | `272.18` | `282.68` | `28.28` | `2.75x` ; `2.74x` |
| `1.3M_ms_high_Z` | MESA `26.4.1` (evolve; `./rn` in note) | `219` | `0` | `2100` | `956` | `162.60` `[162.60, 165.60]` | `742.47` | `776.55` | `77.43` | `1.00x` ; `1.00x` |
| `semiconvection` | FlexMESA `strict`, `diagnostic (2 controls ignored)` | `293` | `0` | `1993` | `1629` | `169.868` `[169.512, 169.898]` | `579.76` | `355.99` | `85.23` | `2.18x` ; `2.05x` |
| `semiconvection` | MESA `26.4.1` (evolve; `./rn` in note) | `293` | `0` | `2094` | `1611` | `366.00` `[364.80, 379.80]` | `1249.15` | `775.31` | `174.79` | `1.00x` ; `1.00x` |
| `low_z` | FlexMESA `strict`, `diagnostic (3 controls ignored)` | `181` | `0` | `1675` | `802` | `26.760` `[26.537, 26.793]` | `147.84` | `184.29` | `15.98` | `1.32x` ; `3.99x` |
| `low_z` | MESA `26.4.1` (evolve; `./rn` in note) | `574` | `0` | `1705` | `778` | `108.60` `[108.00, 108.60]` | `189.20` | `243.23` | `63.70` | `1.00x` ; `1.00x` |
| `make_metals` | FlexMESA `strict`, `diagnostic (2 controls ignored)` | `477` | `5` | `3356` | `1375` | `103.891` `[103.560, 104.102]` | `217.80` | `158.45` | `30.96` | `2.79x` ; `2.25x` |
| `make_metals` | MESA `26.4.1` (evolve; `./rn` in note) | `511` | `4` | `4122` | `1274` | `287.40` `[286.80, 291.60]` | `562.43` | `441.39` | `69.72` | `1.00x` ; `1.00x` |

*MESA full `./rn` wall medians (`3` repeats): `1.3M_ms_high_Z`: `163.75 s` `[163.60, 167.84]` (`77.98 ms/iter`); `semiconvection`: `367.46 s` `[366.06, 381.98]` (`175.48 ms/iter`); `low_z`: `109.31 s` `[108.90, 109.80]` (`64.11 ms/iter`); `make_metals`: `333.58 s` `[332.97, 338.77]` (`80.93 ms/iter`).*

On the two step-matched cases (`1.3M_ms_high_Z` and `semiconvection`), wall time per zone-step for the FlexMESA `strict` diagnostic runs is `2.18x` to `2.75x` smaller than MESA `26.4.1` (`2.05x` to `2.74x` per Newton iteration). On the two cases whose step counts still diverge from MESA, the per-zone-step ratios are `1.32x` to `2.79x` (`2.25x` to `3.99x` per Newton iteration). Normalizing by Newton iterations isolates per-iteration cost from step-count and retry differences, but even the step-matched runs remain diagnostic (`allow_unsupported_controls = true`) with non-zero track differences (Section 1) and different default build optimization levels (`gfortran -O2` vs Cargo `opt-level=3`, `lto=fat`).

## 4. What `#150` Waste Removal Bought

Issue `#150` removed two recurring sources of per-step waste in `flexmesa-star`: (1) redundant microphysics evaluations across unchanged `(lnRho, lnT, X)` zone states via `MicrophysicsCache`, and (2) unconditional full `StarModel::clone()` heap allocations prior to every timestep via the pre-sized `PrimaryStateBuffer` retry snapshot.

### 4.1 Historical Pre-`#150` vs Post-`#150` Git A/B Record

| Case | Build Commit | Steps | Median Wall (`s`) | Sample Range (`s`) | Wall / Step (`ms`) | Wall Saved (`%`) | Speedup (`pre / post`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | Pre-`#150` (`af9747e1`) | `346` | `53.8424` | `4.9466` | `155.61` | `0.00%` | `1.00x` |
| `1.3M_ms_high_Z` | Post-`#150` (`3f7fa734`) | `346` | `48.6631` | `0.4205` | `140.64` | `9.62%` | `1.11x` |

This git A/B record covers one case (`1.3M_ms_high_Z`, `3` repeats per build) measured directly across commits `af9747e1` and `3f7fa734`. The spread of the pre-`#150` samples (`4.95 s`) is about as large as the measured saving (`5.18 s`), so the wall-clock saving is not well resolved. The two historical commits also differ by more than the two items described above.

### 4.2 `e0e987fc` Cache & Memory Measurements vs Probe Estimates

In the table below, the cache counters (`Hits / Step`, `Hit Rate`) and heap footprint columns (`StarModel` vs `PrimaryStateBuffer` bytes and ratio) are **direct measurements on `e0e987fc`**. The last two columns (`Est. Micro Wall Saved` and `Est. Clone vs Lazy Buffer`) are **un-rescaled probe estimates**, not A/B measurements: `Est. Micro Wall Saved` is cache hits multiplied by the difference between a cold and a warm kernel probe from Section 2, and the clone column times `StarModel::clone()` at `e0e987fc` rather than on pre-`#150` code.

| Case | Run Mode | `MicrophysicsCache` Hits / Step (Hit Rate) | `StarModel` Heap (`KiB`) | `PrimaryStateBuffer` (`KiB`) | Heap Reduction | Est. Micro Wall Saved (`s`, probe) | Est. Clone vs Lazy Buffer (`s`, probe) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | `diagnostic (2 controls ignored)` | `6.92` (`36.1%`) | `8587.9` | `2069.7` | `4.15x` | `12.3826` | `0.0597` vs `0.01222` |
| `semiconvection` | `diagnostic (2 controls ignored)` | `6.94` (`45.8%`) | `21209.8` | `3713.7` | `5.71x` | `45.0361` | `0.6462` vs `0.13509` |
| `low_z` | `diagnostic (3 controls ignored)` | `6.98` (`37.6%`) | `6270.4` | `1503.1` | `4.17x` | `7.2390` | `0.0462` vs `0.01709` |
| `make_metals` | `diagnostic (2 controls ignored)` | `6.48` (`41.2%`) | `14346.9` | `3477.6` | `4.13x` | `31.8082` | `0.2619` vs `0.10305` |

## 5. What `fast` Math Buys Over `strict` (`#165`) & Trajectory Drift

Per `DECISIONS #23` and `#165`, `NumericsMode::Fast` (platform `std` `f64` intrinsics and hardware `mul_add`) is compared **exclusively** against `NumericsMode::Strict` (`crlibm 1.0beta4` without FMA contraction) over `3` matched repeats per mode and is never compared against MESA:

| Case | Run Mode | Steps (`s` / `f`) | Newton Iters (`s` / `f`) | Wall Median `[min, max]` (`s` ; `f`, `s`) | `us / zone-step` (`s` / `f`) | `ms / Newton-iter` (`s` / `f`) | Speedup (`us/zs` ; `ms/iter`) | `d_age / age` | Max Track `|d log L|` | Max Track `|d log Teff|` | Max Track `|d log rho_c|` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.3M_ms_high_Z` | `diagnostic (2 controls ignored)` | `219` / `219` | `2108` / `2100` | `59.607` `[59.586, 59.671]` ; `44.372` `[44.258, 45.316]` | `282.68` / `210.43` | `28.28` / `21.13` | `1.34x` ; `1.34x` | `+2.00e-12` | `2.93e-11` | `3.59e-12` | `1.76e-10` |
| `semiconvection` | `diagnostic (2 controls ignored)` | `293` / `293` | `1993` / `2024` | `169.868` `[169.512, 169.898]` ; `149.301` `[148.726, 149.418]` | `355.99` / `312.89` | `85.23` / `73.77` | `1.14x` ; `1.16x` | `+0.00e+00` | `3.39e-10` | `1.56e-10` | `1.13e-09` |
| `low_z` | `diagnostic (3 controls ignored)` | `181` / `181` | `1675` / `1676` | `26.760` `[26.537, 26.793]` ; `20.709` `[20.684, 21.164]` | `184.29` / `142.62` | `15.98` / `12.36` | `1.29x` ; `1.29x` | `+9.68e-13` | `2.10e-11` | `1.62e-10` | `9.52e-12` |
| `make_metals` | `diagnostic (2 controls ignored)` | `477` / `416` | `3356` / `3117` | `103.891` `[103.560, 104.102]` ; `66.852` `[64.906, 67.027]` | `158.45` / `117.21` | `30.96` / `21.45` | `1.35x` ; `1.44x` | `+5.72e-05` | `9.90e-04` | `8.00e-03` | `4.60e-02` |

Switching from `strict` (`crlibm`) to `fast` (`std` + hardware FMA) on `e0e987fc` yields `1.14x` to `1.35x` per zone-step (`1.16x` to `1.44x` per Newton iteration, and `1.14x` to `1.55x` wall speedup, reducing evolution runtime by `12.1%` to `35.7%`) while keeping relative lifetime differences within `|d_age / age| <= 5.72e-05`. The measured speedup reflects `crlibm` transcendental calls (`exp`, `ln`, `log10`, `pow`) in reaction screening, nuclear rates, and MLT/opacity log conversions operating alongside FMA-insensitive spline lookups and block-tridiagonal linear solves.

## 6. `N`-Star Worker-Pool Batching (`smoke: 60 steps`) vs Stateless Accelerator Model (`#166` Input)

### 6.1 Safe Rust `rayon::ThreadPool` Multi-Star Batch Scaling (`smoke: 60 steps`)

This experiment is a **`smoke: 60 steps`** scaling benchmark (capped at `60` steps per star rather than uncapped evolution to physical termination). `profile_evolution batch` evolves `N` independent `low_z` diagnostic models (`diagnostic (3 controls ignored)`, `773` initial zones, heterogeneous `mixing_length_alpha` spanning `[1.98, 2.02]` so Newton iteration counts vary across stars) on dedicated `2`-thread and `8`-thread `rayon` worker pools sharing immutable `OnceLock` EOS and opacity tables:

| Pool Threads | Batch `N` Stars (`smoke: 60 steps`) | Wall (`s`) | Total Steps | Throughput (`stars / min`) | `zone-steps / s` | Speedup vs `N = 1` | Peak RSS (`MiB`) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2` | `1` | `6.917` | `60` | `8.7` | `6625` | `1.00x` | `2122.9` |
| `2` | `2` | `8.682` | `120` | `13.8` | `10556` | `1.59x` | `2122.9` |
| `2` | `4` | `17.298` | `240` | `13.9` | `10596` | `1.60x` | `2122.9` |
| `2` | `8` | `34.728` | `480` | `13.8` | `10556` | `1.59x` | `2180.2` |
| `2` | `16` | `69.651` | `960` | `13.8` | `10526` | `1.59x` | `2319.0` |
| `8` | `1` | `8.098` | `60` | `7.4` | `5659` | `1.00x` | `3008.7` |
| `8` | `2` | `5.557` | `120` | `21.6` | `16493` | `2.91x` | `3008.7` |
| `8` | `4` | `6.097` | `240` | `39.4` | `30062` | `5.31x` | `3008.7` |
| `8` | `8` | `10.247` | `480` | `46.8` | `35775` | `6.32x` | `3008.7` |
| `8` | `16` | `20.401` | `960` | `47.1` | `35938` | `6.35x` | `3008.7` |
| `8` | `32` | `42.973` | `1920` | `44.7` | `34123` | `6.03x` | `3008.7` |

In this `smoke: 60 steps` scaling benchmark, at `N = 1` an `8`-thread pool (`8.098 s`) is slower than a `2`-thread pool (`6.917 s`) because intra-star `rayon` loops partition `~773` zones into sub-`100`-zone chunks. Once `N >= 8` independent tracks run concurrently, `rayon` work-stealing assigns whole stars to threads without per-zone synchronization, reaching `47.1` `stars / min` (`35938` `zone-steps / s`, `6.35x` vs `8t` `N = 1` and `5.42x` vs `2t` `N = 1`), while `32` concurrent models hold peak RSS at `3008.7 MiB` via shared `OnceLock` EOS and opacity tables.

### 6.2 Kernel Anchors (`benchmarks/raw/26.4.1/`) & Grid-Scale Accelerator Model

Single-kernel timings from `benchmarks/raw/26.4.1/` anchor what an accelerator (`NVIDIA GeForce RTX 5080`, `jax.vmap` `float64`) achieves relative to `4`-thread Safe Rust CPU per zone:

| Kernel (`benchmarks/raw/26.4.1/`) | Ref Zones | Safe Rust `4t` (`us / zone`) | JAX `vmap` (`us / zone`) | JAX / Rust `4t` Speedup | Execution Notes |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `eos_helm` | `4096` | `0.1348` | `0.0828` | `1.63x` | GPU `@jax.jit` poly eval on pre-fetched cells (public API does `N1-25` host lookup) |
| `eos_skye` | `4096` | `2.3256` | `1.0260` | `2.27x` | GPU `@jax.jit` 3rd-order AD on pre-fetched cells (public API does `N1-25` host lookup) |
| `eos_4d` | `4096` | `0.2050` | `0.0815` | `2.52x` | GPU `@jax.jit` bicubic eval on pre-fetched cells (public API does `N1-25` host lookup) |
| `kap_fixed` | `4096` | `0.1894` | `0.2717` | `0.70x` | `pure_callback` into `_rust.kap_get_results_zones_rust` (`N1-25`) |
| `rates_screen` | `4096` | `0.1494` | `0.0780` | `1.92x` | Pure `@jax.jit` analytical screening kernel |
| `net_eval_host_cb` | `4096` | `1.3212` | `1.2720` | `1.04x` | `pure_callback` into `_rust.net_eval` (`N1-24`) |
| `turb_mlt` | `4096` | `0.0735` | `0.0167` | `4.40x` | Pure `@jax.jit` local cubic/quartic MLT solver |
| `turb_tdc` | `4096` | `0.6266` | `0.0956` | `6.55x` | Pure `@jax.jit` unrolled Newton/dual TDC vs Rust IFT |
| `colors_sed` | `1024` | `1.5820` | `0.6733` | `2.35x` | Pure `@jax.jit` 3D SED + filter integration |
| `mtx_tridiag` | `4096` | `0.0790` | `0.6825` | `0.12x` | `O(N log N)` GPU `vmap` PCR vs 4-system amortized `O(N)` CPU block-Thomas |

Combining the modelled `70.0%` stateless microphysics share (Section 2) with Amdahl's law fixes an upper bound of `3.34x` for any hybrid CPU-Henyey + GPU-microphysics step, even before finite device speedup and host-device overheads. The analytical grid table below explicitly assumes a hypothetical **`6.0x` device speedup** for pure-GPU stateless microphysics over `2`-thread Safe Rust (assuming `N1-24`/`N1-25` host callbacks were removed), together with PCIe transfers (`24 GB/s`), XLA dispatch latency (`80 us` per Newton iteration), and heterogeneous adaptive-timestep lockstep efficiency (`0.72`). By contrast, standalone stateless grid evaluations (`SED` photometry, screening, analytical EOS, MLT surrogates) avoid the Henyey solver loop entirely:

| Grid Stars `N` | Total Zones | `8t` Rust Pool Step (`s`) | Hybrid GPU-Micro Step (`s`) | Hybrid / `8t` Rust Speedup | Stateless-Only `8t` Rust (`s`) | Stateless-Only GPU (`s`) | Stateless GPU Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `963` | `0.0502` | `0.0480` | `1.05x` | `0.0351` | `0.03185` | `1.10x` |
| `8` | `7703` | `0.4014` | `0.4781` | `0.84x` | `0.2811` | `0.25427` | `1.11x` |
| `64` | `61622` | `3.2112` | `3.8165` | `0.84x` | `2.2488` | `2.03357` | `1.11x` |
| `256` | `246489` | `12.8447` | `15.2623` | `0.84x` | `8.9953` | `8.13404` | `1.11x` |
| `1024` | `985958` | `51.3789` | `61.0458` | `0.84x` | `35.9812` | `32.53593` | `1.11x` |

*Model footnote: Both GPU columns assume a hypothetical `6.0x` device kernel speedup over `2`-thread Safe Rust (`run_evolution_profile_26_4_1.py`, `accelerator_grid_estimate.json` `hypothetical_device_micro_speedup_vs_2t_rust`), `24 GB/s` effective PCIe bandwidth, `80 us` per Newton iteration, and `0.72` lockstep utilization for `N > 1`, anchored to the Section 2 rescaled subsystem shares and the Section 6.1 `smoke: 60 steps` `8`-thread Rust pool throughput.*

### 6.3 Concrete Recommendations for `#166`

1. **Population & Grid Stellar Evolution (`N >= 8` tracks)**: Run independent `StarModel` trajectories across a Safe Rust `rayon` worker pool rather than padding heterogeneous meshes (`nz`) and adaptive timesteps (`dt`) into lockstep GPU batches. Shared `OnceLock` physics tables keep memory overhead small across dozens of concurrent stars, and each worker runs asynchronously with zero host-device synchronization.
2. **Differentiable Stellar Structure & Evolution (Post-`#143`)**: Keep forward evolution in Safe Rust and expose Implicit Function Theorem (`@jax.custom_vjp` / `@jax.custom_jvp`) step wrappers solved on CPU (`bcyclic` via `mtx` `solver_mode="rust_ffi"` or `fast_block_thomas_solve`) plus `#150` `.photo` binomial checkpointing. At `N = 4096`, GPU `PCR` (`7.45 ms` single-profile, `2.80 ms` `vmap` amortized) is slower than CPU solvers (`1.38 ms` 1-thread `PyO3` `bcyclic`, `1.29 ms` 1-thread Rust block-Thomas, `0.32 ms` 4-system amortized Rust block-Thomas) because `PCR` performs `O(N log2 N)` dense block multiplications in `float64`.
3. **Where Native Pure-XLA JAX Kernels Pay (`#166`)**: Prioritize pure `@jax.jit` / `jax.vmap` kernels on stateless, fixed-shape batch workloads that run outside the Henyey iteration loop, specifically bolometric and filter photometry (`flexmesa.colors`), analytical screening and neutrino emission (`flexmesa.rates`, `flexmesa.neu`), local `set_mlt_jax` / `T(tau)` boundary evaluations (`flexmesa.turb`, `flexmesa.atm`), and batch inference likelihoods over `>= 10^4` evaluation points. Porting stateful 4D `eos` / `kap` bicubic tables or stiff `net` Rosenbrock solvers to pure GPU solely to accelerate 1D Henyey steps is bounded by the `<= 3.34x` modelled Amdahl ceiling.

## 7. What Was Not Verified

- **Agreement with MESA (`#143`)**: None of the four cases passes the upstream-history harness. Every FlexMESA run here is a diagnostic run (`allow_unsupported_controls = true`, exact counts in Section 1), and track differences remain non-zero even on the step-matched cases. No timing in this report represents a passing `#143` acceptance run.
- **Remaining I/O, timer, and build-flag differences**: Both engines ran `3` timed repeats with history output on every repeat at `2` threads, but MESA's stock `./rn` also writes `LOGS/profile*.data`, `photos/`, and terminal summaries whereas `profile_evolution` writes `history.data` on every step and `final.mod` (radial profiles are omitted because `semiconvection`'s `profile_columns.list` requests unported `grad_temperature`). MESA's internal evolve timer rounds to `0.01 min` (`0.6 s`), MESA zone-steps are estimated via trapezoidal integration of `num_zones` in `history.data`, and compiler defaults differ (`gfortran -O2` vs Cargo `opt-level=3, lto=fat`).
- **Subsystem shares**: The per-subsystem shares are a rescaled bottom-up model (Section 2), not instrumented wall timings, and the stateless share and Amdahl ceiling in Section 6 inherit that model overshoot.
- **Engine version**: FlexMESA measurements were taken at `origin/main` `e0e987fc` and have not been repeated on later commits.
- **Batching step cap**: Section 6.1 is a `smoke: 60 steps` benchmark on `low_z` (`diagnostic (3 controls ignored)`), not an uncapped evolution to physical termination.
- **Case breadth**: Measurements cover four `star/test_suite` cases (`1.3M_ms_high_Z`, `semiconvection`, `low_z`, `make_metals`) on a single `x86_64` workstation; late burning with `approx21.net`, hydrodynamic `rsp2` pulsations, `binary` evolutions, and `astero` simplex runs were not profiled in this campaign.
- **`aarch64` & multi-socket scaling**: `strict` vs `fast` ratios and `rayon` worker-pool curves were measured on one `8`-core / `16`-thread `x86_64` CPU under WSL2; Apple Silicon (`aarch64`) and high-core-count server NUMA scaling were not measured.
- **`#150` git A/B scope**: Historical pre-`#150` (`af9747e1`) vs post-`#150` (`3f7fa734`) binary timing covers one case (`1.3M_ms_high_Z`) with sample spread comparable to the saving; Section 4.2 separates direct `e0e987fc` cache/memory measurements from un-rescaled probe estimates.
- **Grid-scale GPU evolution**: Because `src/flexmesa/star/` JAX helpers are frozen (`JAX_ENGINE_STATUS = "frozen-unverified-until-#143"`), the `N = 1..1024` hybrid GPU numbers in Section 6.2 come from an analytical model assuming a `6.0x` device speedup, `24 GB/s` PCIe, `80 us` XLA launch overhead, and `0.72` lockstep efficiency rather than an executed GPU Henyey engine.
