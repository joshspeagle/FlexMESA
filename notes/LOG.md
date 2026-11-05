# Milestone Log (`LOG.md`)

## 2026-11-04: MESA Codebase Reconnaissance & Repository Setup (#1, PR #2)

- Completed a full architectural audit across all 2,357 files (`404,831` code LOC) of the upstream [MESA (`MESAHub/mesa`)](https://github.com/MESAHub/mesa) repository:
  - Mapped the 12-tier topological module dependency DAG (`const` through `star`, `binary`, `astero`).
  - Audited MESA's SymPy-based `auto_diff` code generator (8 forward-AD types, `14,188` lines), textual `.inc` and `.dek` metaprogramming (`2,480` includes across `175` files), and the 6,008-field `star_info` state container (`3,673` raw Fortran pointers in `star`).
  - Audited all 16 foundational numerical and microphysics packages (`const`, `math`, `interp_1d`, `interp_2d`, `mtx`, `num`, `chem`, `neu`, `turb`, `eos`, `kap`, `rates`, `net`, `ionization`, `atm`, `colors`) and their two-stage `.bin` spline cache infrastructure.
  - Cataloged the two-tier verification oracle: 21 standalone module unit tests (`58` files, `20,618` LOC, `18,885` lines of 16-digit golden `test_output` baselines) and 107 active integration test cases (`91 star`, `8 binary`, `8 astero`).
- Recorded `DECISIONS #1` through `DECISIONS #5` in [`notes/DECISIONS.md`](./DECISIONS.md):
  - Dual-Track (`Safe Rust` + `Python/JAX`) evaluation for Wave 1 plus stateless microphysics pilot (`neu` and `turb`).
  - Opened `deferred` tracking issues `#3` (`pgstar`/`pgbinary`), `#4` (`stella`/`adipls`/`gyre`), and `#5` (`RSP1`).
  - Selected `.safetensors` (`mmap`) + `.safetensors.zst` on GitHub Release Assets for physics data tables and split `eos` into Wave 3a (`Ideal + HELM + PC + Skye`) and Wave 3b (`FreeEOS + OPAL/SCVH + CMS`).
  - Licensed `FlexMESA` under `LGPL-3.0-or-later` and merged initial repository scaffolding in PR `#2`.
- Opened Wave 0 tracking milestone `#6` and sub-issues `#7` (`Wave 0A`), `#8` (`Wave 0B`), and `#9` (`Wave 0C`).
