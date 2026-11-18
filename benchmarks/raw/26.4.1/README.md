# Raw Benchmark Captures (MESA `r26.4.1`)

This directory archives the raw Fortran, Safe Rust (`strict`), PyO3, and
JAX `float64` GPU outputs measured sequentially on an idle workstation across
all 28 workloads and 84 `(workload, N_zones)` operating points.

`ci_continuous_verification` is `false`: CI builds and tests Python, PyO3,
JAX, and Safe Rust without a local MESA checkout, so the upstream MESA
`r26.4.1` Fortran executables run only offline on a workstation with MESA and
the MESA SDK installed.

## Environment Provenance

Exact toolchain and hardware metadata live in `provenance.json`:

- **Capture timestamp (UTC):** `2026-11-17T22:46:37Z` (with `fortran_wave3b.txt`
  and `fortran_wave4.txt` regenerated cleanly from a relative `MESA_DIR`
  working directory without Fortran format-width overflow at `2026-11-17T23:25:34Z`)
- **FlexMESA base commit:** `b5b6abd6245181511ca08ad3236eadb1ba08436b`
- **Host hardware:** AMD Ryzen 7 9800X3D (8-core / 16 logical threads) and
  NVIDIA GeForce RTX 5080 on Linux `x86_64`
- **Upstream Fortran comparator:** MESA release `r26.4.1`, MESA SDK
  `x86_64-linux-26.6.1`, `GNU Fortran (GCC) 15.2.0` (`-O2 -fopenmp`)
- **FlexMESA backends:** `rustc 1.100.0 (0ef1bc386 2026-11-09)`, Python `3.14.0`,
  `JAX 0.11.0` (`gpu` backend), capped at 4 OpenMP/Rayon threads during
  parallel stages (`OMP_NUM_THREADS=4`)

## Stage Semantics

Each suite reports four stages (`Stage 0` through `Stage 3`):

- **Fortran (`fortran_wave*.txt`):** Stage 0 = 1-thread baseline, Stage 1 =
  1-thread SoA loop, Stage 2 = 1-thread fused/hoisted loop, Stage 3 = 4-thread
  OpenMP (`OMP_NUM_THREADS=4`).
- **Safe Rust (`rust_waves_1_to_5.txt`):** Stage 0 = 1-thread scalar baseline,
  Stage 1 = 1-thread contiguous SoA slice loop, Stage 2 = 1-thread
  fused/hoisted slice loop, Stage 3 = 4-thread `rayon` pool.
- **Python (`python_waves_1_to_5.json`):** Stage 0 = `PyO3` 1-thread
  (`flexmesa._rust`), Stage 1 = `PyO3` 4-thread `rayon`, Stage 2 =
  `JAX @jax.jit` single profile (`GPU`), Stage 3 = `JAX @jax.jit(jax.vmap)`
  batched (`GPU`, amortized per profile).

## Comparator Taxonomy

Every workload in `provenance.json` belongs to one of four comparator classes:

1. **MESA `26.4.1` across all stages (`21` workloads):** `interp_2d`, `neu`,
   `turb_mlt`, `turb_tdc`, `ion_fe56`, `ion_4d`, `colors_sed`, `eos_ideal`,
   `eos_helm`, `eos_pc`, `eos_skye`, `eos_4d`, `kap_fixed`, `rates_screen`,
   `net_eval`, `atm_table`, `star_micro_vars`, `star_hydro_eqns`,
   `star_newton_step`, `star_remesh`, and `star_evolve_step` call compiled
   upstream MESA `r26.4.1` subroutines in every stage.
2. **MESA `26.4.1` Stages 0-1, Reference Fortran Stages 2-3 (`3` workloads):**
   `auto_diff` (Stages 0-1 MESA `auto_diff_real_2var_order2`; Stages 2-3
   analytical fused Fortran), `mtx_tridiag` (Stages 0-1 MESA
   `bcyclic_dble_decsolblk`, plus standalone 1t/4t `Single-Matrix BCYCLIC`
   timings at the foot of `fortran_wave1.txt`; Stages 2-3 local
   `fast_block_thomas_solve`), and `chem_comp` (Stages 0-1 MESA
   `basic_composition_info`; Stages 2-3 hoisted Fortran moment loop).
3. **Reference Fortran (hybrid MESA helpers, `3` Wave 5 workloads):**
   `w5_diff_rot` calls MESA `hydro_rotation::a` and `log_term_power` while
   computing Caplan OCP, rotational instabilities, and Spruit-Tayler dynamo
   locally in `bench_kernels_wave5.f90` (and diverges by `99.17%` from the
   post-`642da54` Rust/JAX snapshot); `w5_binary_rlof` calls MESA
   `binary_utils::eval_rlobe` and `binary_tides::f3`/`f4` with standalone
   Ritter/Kerr/Jdot formulas; `w5_astero_seism` calls MESA
   `astero_get_cubic_all_freq_corr`, `astero_get_combined_all_freq_corr`, and
   `astero_support::get_frequency_ratios` with standalone acoustic integrals
   and scaling relations.
4. **Reference Fortran (non-MESA, `1` Wave 5 workload):** `w5_riemann_rsp2` calls
   zero upstream MESA subroutines.

## Notes on Specific Workloads

- **Wave 4 (`fortran_wave4.txt`):** `bench_star_evolve_step` sets
  `s% dt = 1.0e2_dp` and invokes `do_struct_burn_mix` on an `8`-zone test model
  during pre-flight Stage checks (`hydro_solver_step returned ierr -1`), and
  MESA logs `WARNING: Failed to find colors namelist in file: mesa-26.4.1/star/test/inlist`
  during ZAMS model initialization. Checksums match upstream execution at
  `128`, `512`, and `1024` zones without solver errors.
