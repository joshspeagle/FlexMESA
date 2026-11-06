# Milestone Log (`LOG.md`)

## 2026-11-04: MESA Codebase Reconnaissance & Repository Setup (#1, PR #2)

- Completed a full architectural audit across all 2,357 files (`404,831` code LOC) of the upstream [MESA (`MESAHub/mesa`)](https://github.com/MESAHub/mesa) repository:
  - Mapped the 12-tier topological module dependency DAG (`const` through `star`, `binary`, `astero`).
  - Audited MESA's SymPy-based `auto_diff` code generator (8 forward-AD types, `14,188` lines), textual `.inc` and `.dek` metaprogramming (`2,480` includes across `175` files), and the 6,008-field `star_info` state container (`3,673` raw Fortran pointers in `star`).
  - Audited all 16 foundational numerical and microphysics packages (`const`, `math`, `interp_1d`, `interp_2d`, `mtx`, `num`, `chem`, `neu`, `turb`, `eos`, `kap`, `rates`, `net`, `ionization`, `atm`, `colors`) and their two-stage `.bin` spline cache infrastructure.
  - Cataloged the two-tier verification oracle comprising 21 standalone module unit tests (`58` files, `20,618` LOC, `18,885` lines of 16-digit golden `test_output` baselines) and 107 active integration test cases (`91 star`, `8 binary`, `8 astero`).
- Recorded `DECISIONS #1` through `DECISIONS #5` in [`notes/DECISIONS.md`](./DECISIONS.md):
  - Dual-Track (`Safe Rust` + `Python/JAX`) evaluation for Wave 1 plus stateless microphysics pilot (`neu` and `turb`).
  - Opened `deferred` tracking issues `#3` (`pgstar`/`pgbinary`), `#4` (`stella`/`adipls`/`gyre`), and `#5` (`RSP1`).
  - Selected `.safetensors` (`mmap`) + `.safetensors.zst` on GitHub Release Assets for physics data tables and split `eos` into Wave 3a (`Ideal + HELM + PC + Skye`) and Wave 3b (`FreeEOS + OPAL/SCVH + CMS`).
  - Licensed `FlexMESA` under `LGPL-3.0-or-later` and merged initial repository scaffolding in PR `#2`.
- Opened Wave 0 tracking milestone `#6` and sub-issues `#7` (`Wave 0A`), `#8` (`Wave 0B`), and `#9` (`Wave 0C`).

## 2026-11-05: Wave 0 & Wave 1 Dual-Track Completion and Four-Way Benchmark Synthesis (#6–#26)

- Completed **Wave 0** (`#6`–`#9`, PRs `#10`–`#12`): Index and Memory-Order Contract (`docs/design/wave-0-spec.md`), `test_output` golden baseline parser (`flexmesa-verify` / `flexmesa.verify`), `.defaults` and `.list` schema parser (`flexmesa-schema` / `flexmesa.schema`), and 64-byte-aligned `.safetensors` data converter and `mmap` loader (`flexmesa-data` / `flexmesa.data`).
- Completed **Wave 1** (`#13`–`#19`, PRs `#20`–`#25`): Dual-track Safe Rust (`crates/`) and Python/JAX `float64` (`src/flexmesa/`) implementations of all 10 Wave 1 modules (`const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num`, `neu`, and `turb`), verified against MESA's standalone `test_output` suites and `proptest`/`hypothesis` fuzzing.
- Built the native Fortran 2008 benchmark harness (`benchmarks/fortran_mesa/`) against upstream Fortran MESA (`gfortran -O2 -fopenmp`) and synthesized the 4-way benchmark and parity comparison (`Fortran MESA` vs. `Safe Rust` vs. `Python/NumPy` vs. `Python/JAX`) in [`docs/design/wave-1-benchmark-report.md`](../docs/design/wave-1-benchmark-report.md) and [Issue #26](https://github.com/joshspeagle/FlexMESA/issues/26).
