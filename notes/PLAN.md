# FlexMESA Implementation Plan & Wave Roadmap

## Current Status

- **Active Phase**: Pre-Wave 0 Architectural Walkthrough & Repository Setup ([Issue #1](https://github.com/joshspeagle/FlexMESA/issues/1)).
- **Completed Decisions**: `DECISIONS #1` (Dual-Track Safe Rust + Python/JAX evaluation across Wave 1 + stateless microphysics pilot).
- **Pending Decisions (`notes/QUESTIONS.md`)**:
  - Decision 2: Legacy and external subsystem scope exclusions (`PGSTAR`, `STELLA`, `ADIPLS`, legacy `RSP1`).
  - Decision 3: Physics data tables (`2.18 GiB` LFS) and 7-component EOS cascade strategy.
  - Decision 4: Stage 0 toolchain setup on `speagle-desktop` (`gfortran`/`MESASDK`, `ndiff`, `rustup`/`rustc`) and repository license.

---

## 6-Wave Migration Overview

| Wave | Target MESA Modules | Source LOC | Execution Track | Verification Oracle & Exit Gate |
| :--- | :--- | ---: | :--- | :--- |
| **Wave 0: Infrastructure, Toolchain & Schema Codegen** | `make`, `linters`, `defaults`, `data` caches | ~5,000 LOC (replaces ~30k LOC boilerplate) | Shared (Rust + Python) | Stage 0 golden baselines captured; `.defaults` schema generator and `.bin` to `.safetensors`/HDF5 converter verified. |
| **Wave 1: Foundational Numerics + Stateless Microphysics Pilot** | `const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num` + pilot `neu`, `turb` | ~56,000 LOC | **Dual-Track** (`crates/` Safe Rust + `src/flexmesa/` JAX) | Match all 10 standalone `<pkg>/test/test_output` golden suites (`<= 1e-12`), `check_grads` / `proptest` / `hypothesis`, 4-stage performance staircase, and **Dual-Track Architecture Review** with Josh. |
| **Wave 2: Remaining Atomic & Post-Processing Microphysics** | `chem`, `ionization`, `colors` | ~9,800 LOC | Post-Wave 1 Selected Track | Match `chem` (`2,635` lines), `ionization` (`43` lines), and `colors` (`234` lines) golden `test_output` baselines. |
| **Wave 3: Tabulated & Stiff Microphysics** | `eos`, `kap`, `rates`, `net`, `atm` | ~69,300 LOC | Post-Wave 1 Selected Track | Match `eos`, `kap`, `rates`, `net`, and `atm` golden `test_output` baselines, including `f64x2`/`f128` NSE cancellation checks and 1-zone stiff burn benchmarks. |
| **Wave 4: Core 1D `star` Hydro/Henyey Engine** | `star_data` + core `star` (`hydro_*`, `star_solver`, `star_bcyclic`, `mix_info`, `adjust_mesh`, `timestep`, `evolve`) | ~55,000 LOC | Post-Wave 1 Selected Track | Pass `star/test` smoke test + foundational `star/test_suite` benchmarks (`1M_pre_ms_to_wd`, `7M_prems_to_AGB`, `15M_dynamo`, `wd_cool_0.6M`, `make_zams`) and photo-restart checksum parity. |
| **Wave 5: Advanced `star` Physics, `binary` & `astero`** | `diffusion`, `rotation`, `conv_premix`, `predictive_mix`, `riemann`, `rsp2`, `binary`, `astero` | ~55,000 LOC | Post-Wave 1 Selected Track | Scale across the active `star/test_suite`, `binary/test_suite`, and `astero/test_suite` benchmarks. |
| **Final Audit** | Full Workspace | All | All | Execute `github-flow` Section 4 gate: full test suite, `ruff`/`mypy`/`check_code_style.py`/`clippy`, `/code-simplify`, and `/code-review`. |
