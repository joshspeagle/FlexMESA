# Changelog

All notable changes to `FlexMESA` are documented in this file.

## [Unreleased]

### Post-Audit Re-Baselining & Governance (`#140`-`#174`)
- Pinned upstream reference exclusively to **MESA release `26.4.1`** (`c8c2026`, Zenodo `19722306`), replacing pre-audit `0b2e6bb` bundles (`mesa-test-inputs-v2`, `mesa-histories-v1`, `mesa-module-hex-v1`) (`#141`, `DECISIONS #22`).
- Replaced step-capped smoke pass assertions with uncapped trajectory comparison against stored MESA `26.4.1` histories (`flexmesa trace`, `flexmesa harness`, `notes/audit/case_table.md`, baseline `0 / 106` passing) and governed `102` declared known gaps in `benchmarks/history_harness/known_gaps.toml` (`#143`, `#144`, `DECISIONS #19`-`#20`).
- Implemented script-generated progress dashboard and line-level port manifest (`benchmarks/dashboard/port_manifest.toml`, `flexmesa dashboard`) alongside `notes/audit/defaults_diff_26_4_1.md` and `docs/benchmarks/{known-gaps,coverage-map}.md` (`#174`, `DECISIONS #24`).
- Separated canonical Safe Rust `"strict"` numerics (`crlibm 1.0beta4` matching `CRMATH`) from `"fast"` hardware intrinsics and derived `float64` JAX layers (`#165`, `#166`, `DECISIONS #23`).
- Implemented upstream MESA `26.4.1` `LOGS/history.data`, `LOGS/profileN.data`, `LOGS/profiles.index`, and `binary_history.data` log formatting and reporting (`#163`, `DECISIONS #18`).
- Removed synthetic MESA I-VI figure generators and landmark tables, corrected pre-audit documentation/scorecard claims, and added MESA `26.4.1` `file:line` citation and non-synthetic figure regression checks (`#142`, `DECISIONS #17`).

### Initial Pre-Audit Scaffolding (Wave 0-1)
- Initial repository scaffolding, `CLAUDE.md` contributor instructions, GitHub issue and pull request templates, `pyproject.toml`, and `Cargo.toml` workspace configuration (#1).
- Initial pre-audit MESA codebase census, dependency DAG, and migration specifications in `docs/design/` (archived as pre-audit historical records under `notes/SCOPE.md` and `DECISIONS #17`) (#1).
- Wave 0 specification (`docs/design/wave-0-spec.md`) and upstream `test_output` reference parser and numerical comparison utilities (`flexmesa.testing.golden`, `crates/flexmesa-testing`) (#7).
- Declarative MESA `<pkg>/defaults/*.defaults` and `*_columns.list` schema parser and legacy Fortran `&namelist` (`inlist`) parser (`flexmesa.schema`, `crates/flexmesa-schema`) (#8).
- Physics table `SHA-256` manifest (`src/flexmesa/data/manifest.toml`), memory-mapped `.safetensors` (`float64`/`int64`) loader, and ASCII-to-`.safetensors` converters (`flexmesa.data`, `crates/flexmesa-data`) (#9).
- Wave 1 specification (`docs/design/wave-1-spec.md`), CODATA/IAU CGS constants (`flexmesa.constants`, `crates/flexmesa-const`), finiteness guards and stream tokenizers (`flexmesa.utils`, `crates/flexmesa-utils`) (#14).
- IEEE-754 `float64` math wrappers (`flexmesa.math`, `crates/flexmesa-math`) and forward-mode dual/hyper-dual numbers (`flexmesa.auto_diff`, `crates/flexmesa-autodiff`) (#15).
- 1D Steffen/Huynh spline interpolation (`flexmesa.interp_1d`, `crates/flexmesa-interp1d`) and 2D bicubic/Akima surface interpolation (`flexmesa.interp_2d`, `crates/flexmesa-interp2d`) (#16).
- Dense, banded, and block-tridiagonal linear solvers (`flexmesa.mtx`, `crates/flexmesa-mtx`), root-finders, simplex optimization, and Runge-Kutta/Rosenbrock ODE integrators (`flexmesa.num`, `crates/flexmesa-num`) (#17).
