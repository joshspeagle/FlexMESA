# Open Architectural Questions (`QUESTIONS.md`)

All foundational architectural questions for Wave 0 and Wave 1 have been resolved and recorded in [`notes/DECISIONS.md`](./DECISIONS.md):

1. **Target Language & Backend Architecture (`DECISIONS #1`)**: Wave 1 Dual-Track (`crates/` Safe Rust + `src/flexmesa/` Python/JAX `float64`) covering all 8 foundational numerical modules (`const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num`) plus stateless Tier 3/5 microphysics modules `neu` and `turb`.
2. **Legacy & External Bundled Subsystem Exclusions (`DECISIONS #2`)**: Exclude `pgstar`/`pgbinary` (`#3`), external `stella`/`adipls`/`gyre` source ports (`#4`), and legacy `RSP1` (`#5`), replacing `pgstar` with Python/Matplotlib diagnostics, exporting `GYRE`/`FGONG` structures at the pulsation boundary, and porting unified `TDC`/`RSP2`.
3. **Physics Data Tables & 7-Component EOS Staging (`DECISIONS #3`)**: Convert MESA tables in Wave 0 into 64-byte-aligned `float64` `.safetensors` files (`mmap`-ready in `~/.cache/flexmesa/data/`), distributed as modular `.safetensors.zst` bundles on GitHub Release Assets with a committed `SHA-256` manifest, and stage `eos` into Wave 3a (`Ideal + HELM + PC + Skye`) and Wave 3b (`FreeEOS + OPAL/SCVH + CMS`).
4. **Repository License & Portable Configuration (`DECISIONS #4`)**: License `FlexMESA` under `LGPL-3.0-or-later` and keep build/test configuration portable across platforms.

## Active Decision Gate ([Issue #26](https://github.com/joshspeagle/FlexMESA/issues/26))

- **Post-Wave 1 Backend Selection for Wave 2 and Wave 3 (`Fortran MESA` vs. `Safe Rust` vs. `Python/NumPy` vs. `Python/JAX`)**:
  - Empirical 4-way benchmark and parity synthesis across all 10 Wave 1 modules is complete in [`docs/design/wave-1-benchmark-report.md`](../docs/design/wave-1-benchmark-report.md) and posted to [Issue #26](https://github.com/joshspeagle/FlexMESA/issues/26).
  - **Option 1 (Recommended Hybrid / Dual-Track Architecture)**: Maintain dual-track Safe Rust (`crates/`, with const-generic `DualOrder1<N>`, Implicit Function Theorem solvers, and persistent `rayon` thread pools) + Python/JAX `float64` (`src/flexmesa/`, with branchless `@jax.jit` + `custom_jvp`/`custom_vjp` kernels) across Wave 2 microphysics, and add parallel block cyclic reduction (`bcyclic` / PCR) plus `PyO3` / JAX `ffi_call` solver bindings for block-tridiagonal Henyey solves (`mtx`) and Wave 3 adaptive-mesh stellar evolution (`star`).
  - **Option 2 (Pure Dual-Track Without Cross-Language FFI)**: Keep Safe Rust and Python/JAX completely independent through Wave 2 and Wave 3, using native JAX parallel cyclic reduction (PCR) and padded zone buckets ($N_{\mathrm{pad}} \in \{512, 1024, 2048, 4096\}$) on GPU.
  - **Option 3 (Single-Language Concentration Post-Wave 1)**: Concentrate Wave 2 and Wave 3 in a single primary engine (either Safe Rust with `PyO3` bindings or Python/JAX `float64` with Rust FFI for linear/stiff solvers).
