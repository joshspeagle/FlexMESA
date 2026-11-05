# FlexMESA Implementation Plan & Wave Roadmap

## Current Status

- **Active Phase**: **Wave 0 (`Infrastructure, Schema Parser & .safetensors Data Pipeline`)** ([Issue #6](https://github.com/joshspeagle/FlexMESA/issues/6)), executing sub-issues [#7 (`Wave 0A`)](https://github.com/joshspeagle/FlexMESA/issues/7), [#8 (`Wave 0B`)](https://github.com/joshspeagle/FlexMESA/issues/8), and [#9 (`Wave 0C`)](https://github.com/joshspeagle/FlexMESA/issues/9).
- **Completed Architectural Decisions (`notes/DECISIONS.md`)**:
  - `DECISIONS #1`: Dual-Track (`Safe Rust` + `Python/JAX`) evaluation across Wave 1 (`const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num`) + stateless microphysics pilot (`neu`, `turb`).
  - `DECISIONS #2`: Legacy and external subsystem scope exclusions (`pgstar`/`pgbinary` `#3`, `stella`/`adipls`/`gyre` `#4`, and legacy `RSP1` `#5`).
  - `DECISIONS #3`: `.safetensors` runtime table format (`mmap`), `.safetensors.zst` hosting on GitHub Release Assets, and two-subwave EOS staging (`Wave 3a` analytic/Helmholtz core $\to$ `Wave 3b` 4D tabulated grids).
  - `DECISIONS #4`: `LGPL-3.0-or-later` license and portable project configuration (`pyproject.toml`, `Cargo.toml`).
  - `DECISIONS #5`: Issue-first wave decomposition (`phase` $\to$ `enhancement` sub-issues) and sequential pre-merge verification, simplification, and code review gate.

---

## 6-Wave Migration Overview

| Wave | Target MESA Modules | Source LOC | Execution Track | Verification Oracle & Exit Gate |
| :--- | :--- | ---: | :--- | :--- |
| **Wave 0: Infrastructure, Schema Parser & Table Pipeline** ([#6](https://github.com/joshspeagle/FlexMESA/issues/6)) | `make`, `linters`, `defaults`, `data` caches | ~5,000 LOC (replaces ~30k LOC boilerplate) | Shared (Rust + Python) | Wave 0 specification and Index/Memory-Order Contract (#7); `<pkg>/test/test_output` golden baseline harness (#7); `.defaults` and `columns.list` schema parser (#8); `.safetensors` table converter and `mmap` loader (#9). |
| **Wave 1: Foundational Numerics + Stateless Microphysics Pilot** | `const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num` + pilot `neu`, `turb` | ~56,000 LOC | **Dual-Track** (`crates/` Safe Rust + `src/flexmesa/` JAX) | Match all 10 standalone `<pkg>/test/test_output` golden suites (`<= 1e-12`), `check_grads` / `proptest` / `hypothesis`, 4-stage performance staircase, and **Post-Wave 1 Dual-Track Architecture Review**. |
| **Wave 2: Remaining Atomic & Post-Processing Microphysics** | `chem`, `ionization`, `colors` | ~9,800 LOC | Post-Wave 1 Selected Track | Match `chem` (`2,635` lines), `ionization` (`43` lines), and `colors` (`234` lines) golden `test_output` baselines. |
| **Wave 3a: Analytical & Single-Table EOS Core** | `eos` (`Ideal`, `HELM`, `PC`, `Skye`, $C^2$ quintic blending) | ~11,500 LOC | Post-Wave 1 Selected Track | Match ionized interior `eos/test` cases (`<= 1e-12`), verify 3rd-order forward-AD Maxwell thermodynamic consistency (`dP/dT|_rho == -dS/dV|_T` to `1e-14`). |
| **Wave 3b: Tabulated 4D EOS, Opacity, Rates, Net & Atmosphere** | `eos` (`FreeEOS`, `OPAL/SCVH`, `CMS`), `kap`, `rates`, `net`, `atm` | ~57,800 LOC | Post-Wave 1 Selected Track | Match full `eos`, `kap`, `rates`, `net`, and `atm` golden `test_output` baselines, including `f64x2`/`f128` NSE cancellation checks and 1-zone stiff burn benchmarks. |
| **Wave 4: Core 1D `star` Hydro/Henyey Engine** | `star_data` + core `star` (`hydro_*`, `star_solver`, `star_bcyclic`, `mix_info`, `adjust_mesh`, `timestep`, `evolve`) | ~55,000 LOC | Post-Wave 1 Selected Track | Pass `star/test` smoke test + foundational `star/test_suite` benchmarks (`1M_pre_ms_to_wd`, `7M_prems_to_AGB`, `15M_dynamo`, `wd_cool_0.6M`, `make_zams`) and photo-restart checksum parity. |
| **Wave 5: Advanced `star` Physics, `binary`, `astero` & Plotting** | `diffusion`, `rotation`, `conv_premix`, `predictive_mix`, `riemann`, `rsp2`, `pulse_*`, `binary`, `astero`, `plotting` | ~55,000 LOC | Post-Wave 1 Selected Track | Scale across the active `star/test_suite`, `binary/test_suite`, and `astero/test_suite` benchmarks. |
| **Final Audit** | Full Workspace | All | All | Full Python and Rust test suites, static analysis (`ruff`, `mypy --strict`, `cargo clippy`), code simplification pass, and multi-angle code review. |
