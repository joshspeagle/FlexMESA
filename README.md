# FlexMESA

**FlexMESA** ports [MESA (Modules for Experiments in Stellar Astrophysics)](https://github.com/MESAHub/mesa) into a canonical **Safe Rust** engine (`crates/`, enforcing `#![forbid(unsafe_code)]` across all crates except the sole vendored `crlibm` FFI wrapper `flexmesa-crlibm-sys`, exposed to Python via `PyO3` as `flexmesa._rust`) with a Python front end and a derived `float64` **JAX (`@jax.jit` + `jax.vmap`)** layer (`src/flexmesa/`). The long-term goal for `v0.1.0` (tracked against the live baseline counters in the dashboard below) is a faithful, evidence-verified port of **MESA release `26.4.1`** (git tag `26.4.1`, Zenodo record `19722306`), including the bundled codes required by its test suites (`notes/SCOPE.md`).

## Architecture & Scope Principles

1. **Canonical Safe Rust Engine (`strict` and `fast` Modes)**: Physics and evolution algorithms live in Safe Rust (`crates/`), with a default hardware-independent `strict` numerics mode (`crlibm 1.0beta4` matching `CRMATH`, no FMA) judged directly against MESA `26.4.1` and a `fast` mode validated against `strict` (`#165`, `DECISIONS #23`).
2. **Python Front End and Derived `float64` JAX Layer**: Python drives configuration, diagnostics, and plotting (`flexmesa.plotting`), while JAX (`src/flexmesa/`) provides forward and reverse automatic differentiation (`jax.grad`, `jax.jacfwd`, `@jax.custom_vjp`) and batched execution (`jax.vmap`) validated against Rust `strict` mode (`#166`).
3. **Evidence-Gated Parity Verification**: Every status table, end-to-end trajectory comparison (`notes/audit/case_table.md`), and dashboard counter below is generated mechanically from the repository, `benchmarks/dashboard/port_manifest.toml`, `benchmarks/history_harness/`, coverage reports, and the GitHub API (`#143`, `#145`, `#174`).

## Development Progress Dashboard

<!-- BEGIN GENERATED DASHBOARD -->
Generated mechanically by `flexmesa dashboard render` from [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml), [`benchmarks/history_harness/`](https://github.com/joshspeagle/FlexMESA/tree/main/benchmarks/history_harness), `cargo llvm-cov`, `pytest --cov`, and the GitHub API ([#174](https://github.com/joshspeagle/FlexMESA/issues/174)). Reference: **MESA release `26.4.1`** (commit `c8c2026`, Zenodo `19722306`).

### Headline Status

| Metric | Value | Source |
| :--- | ---: | :--- |
| `star` + `star_data` + `binary` + `astero` ported (unverified) | 20,107 / 144,952 (13.9%; `star`+`binary`+`astero`: 14.4%) | [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml) |
| Upstream lines verified (all 26 modules) | 0 / 497,703 (0.0%) | [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml) |
| End-to-end MESA `26.4.1` cases passing (`90`/`8`/`8`) | 0 / 106 (102 declared gaps, 4 no reference yet, 0 undeclared failures) | [`case_table.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/case_table.md) |
| Strict mode bit-identical across machines (`#165`) | gated in CI by scripts/check_strict_determinism.py (x86_64 vs aarch64, 1 vs 4 threads; 474,062 math vectors + uncapped star/1.3M_ms_high_Z + binary/jdot_gr_check run with unsupported controls ignored; a bit-identity check, not a comparison with MESA) | [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |
| Strict-mode kernels bit-identical to MESA `26.4.1` (`#165`) | math: 403,258/403,258 CRMATH 0 ULP (70,804 libm <= 1 ULP); 3,856/3,966 probed module hex 0 ULP (4 complete, 13 non-math probed) | [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |
| Test line coverage (Rust `llvm-cov` / Python `pytest --cov`) | 74.0% (66,808/90,287) / 87.9% (18,970/21,585) | [`coverage_summary.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/coverage_summary.json) |
| Road-to-`v0.1.0` packages ([#140](https://github.com/joshspeagle/FlexMESA/issues/140)) & open PRs | 12 closed, 22 open; 5 open PR(s) | [`github_snapshot.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/github_snapshot.json) |
| Post-parity deferred backlog ([#169](https://github.com/joshspeagle/FlexMESA/issues/169)) | 18 open (17 upstream bugs, 1 enhancement, 5 `parity-pending`) | [`github_snapshot.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/github_snapshot.json) |

### 1. Port Coverage by Module (MESA `26.4.1`)

Progress is measured by **upstream MESA `26.4.1` source lines by status**, never by lines of Rust or Python written. No file starts as verified; raising a status requires a pull request that supplies the corresponding routine counterparts or generated MESA `26.4.1` evidence.

| Module | Upstream LOC | Absent | Stub | Partial | Ported (unverified) | Kernel ref. vectors | Case-level evidence | Rust LOC | Python/JAX LOC | % Ported | % Verified |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star` | 120,509 | 19,941 | 17,567 | 65,013 | 17,988 | 0 | 0 | 55,571 | 3,869 | 14.9% | 0.0% |
| `star_data` | 4,870 | 34 | 0 | 4,836 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `binary` | 11,638 | 2,947 | 0 | 6,572 | 2,119 | 0 | 0 | 9,681 | 1,244 | 18.2% | 0.0% |
| `astero` | 7,935 | 2,185 | 0 | 5,750 | 0 | 0 | 0 | 5,762 | 1,288 | 0.0% | 0.0% |
| `eos` | 28,666 | 7,656 | 0 | 21,010 | 0 | 0 | 0 | 14,202 | 9,673 | 0.0% | 0.0% |
| `kap` | 22,430 | 13,652 | 0 | 8,778 | 0 | 0 | 0 | 4,862 | 3,248 | 0.0% | 0.0% |
| `net` | 13,452 | 0 | 0 | 13,452 | 0 | 0 | 0 | 10,347 | 728 | 0.0% | 0.0% |
| `rates` | 17,217 | 192 | 0 | 17,025 | 0 | 0 | 0 | 3,181 | 1,476 | 0.0% | 0.0% |
| `neu` | 2,184 | 352 | 0 | 0 | 1,832 | 0 | 0 | 1,916 | 2,648 | 83.9% | 0.0% |
| `atm` | 6,853 | 2,468 | 0 | 4,385 | 0 | 0 | 0 | 2,096 | 791 | 0.0% | 0.0% |
| `turb` | 2,033 | 0 | 0 | 681 | 1,352 | 0 | 0 | 2,268 | 2,368 | 66.5% | 0.0% |
| `chem` | 4,195 | 373 | 0 | 3,822 | 0 | 0 | 0 | 2,338 | 930 | 0.0% | 0.0% |
| `ionization` | 3,859 | 2,226 | 0 | 1,633 | 0 | 0 | 0 | 3,123 | 1,033 | 0.0% | 0.0% |
| `colors` | 3,823 | 428 | 0 | 3,395 | 0 | 0 | 0 | 2,453 | 1,500 | 0.0% | 0.0% |
| `num` | 20,045 | 845 | 0 | 19,200 | 0 | 0 | 0 | 4,279 | 3,587 | 0.0% | 0.0% |
| `mtx` | 5,802 | 430 | 0 | 5,372 | 0 | 0 | 0 | 4,173 | 1,768 | 0.0% | 0.0% |
| `interp_1d` | 4,969 | 1,624 | 0 | 3,345 | 0 | 0 | 0 | 1,205 | 1,311 | 0.0% | 0.0% |
| `interp_2d` | 13,565 | 6,661 | 0 | 6,784 | 120 | 0 | 0 | 2,687 | 2,288 | 0.9% | 0.0% |
| `auto_diff` | 14,353 | 1,715 | 0 | 12,638 | 0 | 0 | 0 | 2,799 | 1,234 | 0.0% | 0.0% |
| `math` | 936 | 368 | 0 | 568 | 0 | 0 | 0 | 1,992 | 577 | 0.0% | 0.0% |
| `const` | 256 | 0 | 0 | 0 | 256 | 0 | 0 | 368 | 329 | 100.0% | 0.0% |
| `utils` | 3,633 | 1,555 | 0 | 2,078 | 0 | 0 | 0 | 4,846 | 2,584 | 0.0% | 0.0% |
| `gyre` | 55,708 | 55,708 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `adipls` | 30,923 | 30,923 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `stella` | 62,117 | 62,117 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `plotting` | 35,732 | 35,732 | 0 | 0 | 0 | 0 | 0 | 0 | 2,897 | 0.0% | 0.0% |
| **Subtotal: star + star_data + binary + astero** | 144,952 | 25,107 | 17,567 | 82,171 | 20,107 | 0 | 0 | 71,014 | 6,401 | 13.9% | 0.0% |
| **Total (all 26 modules)** | 497,703 | 250,132 | 17,567 | 206,337 | 23,667 | 0 | 0 | 140,149 | 47,371 | 4.8% | 0.0% |

*Notes*: `star_data` state structs reside in `crates/flexmesa-star`. Bridge, testing, and benchmark infrastructure account for an additional `16,922` Rust lines (`flexmesa-py`, `flexmesa-testing`, `flexmesa-bench`) and `13,609` Python lines (`flexmesa.testing`, `flexmesa.bench`). For plotting ([#173](https://github.com/joshspeagle/FlexMESA/issues/173)), `flexmesa.plotting` currently provides `10` `plot_*` builders against `46` upstream `pgstar`/`pgbinary` panel modules.

### 2. Verification & Numerics Status

| Check | Status | Notes |
| :--- | :--- | :--- |
| Strict mode bit-identical across machines & thread counts (Level 1) | `gated in CI by scripts/check_strict_determinism.py (x86_64 vs aarch64, 1 vs 4 threads; 474,062 math vectors + uncapped star/1.3M_ms_high_Z + binary/jdot_gr_check run with unsupported controls ignored; a bit-identity check, not a comparison with MESA)` | Verified by [#165](https://github.com/joshspeagle/FlexMESA/issues/165) CI jobs (`x86_64` vs `aarch64`) |
| Strict-mode kernels bit-identical to MESA `26.4.1` hex outputs (Level 2) | `math: 403,258/403,258 CRMATH 0 ULP (70,804 libm <= 1 ULP); 3,856/3,966 probed module hex 0 ULP (4 complete, 13 non-math probed)` | `348,321` exact IEEE-754 values captured across 24 in-scope module test suites (`mesa-module-hex-v1`); measured in [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |

### 3. End-to-End Cases (`benchmarks/history_harness`)

| Suite | `26.4.1` Cases | Stored `26.4.1` Refs | No Ref Yet | Strict-Validated | Diagnostic Runs | Reach Upstream Term | Within Numeric Tol | Verified Pass | Declared Gaps | Undeclared Failures |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star` | 90 | 86 | 4 | 0 | 90 | 39 | 5 | 0 | 86 | 0 |
| `binary` | 8 | 8 | 0 | 0 | 8 | 4 | 0 | 0 | 8 | 0 |
| `astero` | 8 | 8 | 0 | 0 | 8 | 3 | 1 | 0 | 8 | 0 |
| **Total** | **106** | **102** | **4** | **0** | **106** | **46** | **6** | **0** | **102** | **0** |

- **Full per-case table**: [`notes/audit/case_table.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/case_table.md). `6` case(s) within numeric tolerance (`rsp_BEP`, `rsp_BLAP`, `rsp_Cepheid`, `rsp_RR_Lyrae`, `rsp_save_and_load_file`, `surface_effects`) remain governed as declared gaps under [#161](https://github.com/joshspeagle/FlexMESA/issues/161), [#162](https://github.com/joshspeagle/FlexMESA/issues/162).
- **Declared gaps by owning package**: [#152](https://github.com/joshspeagle/FlexMESA/issues/152) (23), [#161](https://github.com/joshspeagle/FlexMESA/issues/161) (16), [#154](https://github.com/joshspeagle/FlexMESA/issues/154) (12), [#162](https://github.com/joshspeagle/FlexMESA/issues/162) (9), [#157](https://github.com/joshspeagle/FlexMESA/issues/157) (8), [#155](https://github.com/joshspeagle/FlexMESA/issues/155) (7), [#160](https://github.com/joshspeagle/FlexMESA/issues/160) (7), [#170](https://github.com/joshspeagle/FlexMESA/issues/170) (7), [#156](https://github.com/joshspeagle/FlexMESA/issues/156) (4), [#158](https://github.com/joshspeagle/FlexMESA/issues/158) (4), [#159](https://github.com/joshspeagle/FlexMESA/issues/159) (3), [#151](https://github.com/joshspeagle/FlexMESA/issues/151) (1), [#171](https://github.com/joshspeagle/FlexMESA/issues/171) (1).
- **Trajectory comparison figures ([#173](https://github.com/joshspeagle/FlexMESA/issues/173))**: `0` committed.

### 4. Controls & Defaults Coverage (`#147`, `#149`)

| Namelist | Upstream `26.4.1` Scalars | Matched in `FlexMESA` | Defaults Agree | Value Mismatch | Active Limit `None` | Unimplemented |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `star &controls` | 1,444 | 611 | 611 | 0 | 0 | 833 |
| `star &star_job` | 455 | 88 | 88 | 0 | 0 | 367 |
| `binary &binary_controls + &binary_job` | 209 | 120 | 120 | 0 | 0 | 89 |
| `astero &astero_search_controls` | 150 | 73 | 73 | 0 | 0 | 77 |
| `eos &eos` | 102 | 54 | 54 | 0 | 0 | 48 |
| `kap &kap` | 36 | 16 | 16 | 0 | 0 | 20 |
| `star &pgstar` | 2,900 | 0 | 0 | 0 | 0 | 2,900 |
| `binary &pgbinary` | 1,329 | 0 | 0 | 0 | 0 | 1,329 |

Full scalar diff: [`notes/audit/defaults_diff_26_4_1.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/defaults_diff_26_4_1.md) (loud errors on unknown controls tracked in [#147](https://github.com/joshspeagle/FlexMESA/issues/147); defaults alignment tracked in [#149](https://github.com/joshspeagle/FlexMESA/issues/149)).

### 5. Test & Code Health

- **Rust workspace line coverage (`cargo llvm-cov`)**: **74.0%** (`66,808` / `90,287` lines across `26` crates).
- **Python package statement coverage (`pytest --cov`)**: **87.9%** (`18,970` / `21,585` statements across `27` subpackages).
- **Ignored / skipped tests**: `13` Rust `#[ignore]` ([#143](https://github.com/joshspeagle/FlexMESA/issues/143), [#160](https://github.com/joshspeagle/FlexMESA/issues/160), [#161](https://github.com/joshspeagle/FlexMESA/issues/161), [#162](https://github.com/joshspeagle/FlexMESA/issues/162)) and `0` Python `xfail`/`skip` (none; `0` unlinked).
- **Static & documentation gates**: `cargo clippy --all-targets -- -D warnings`, `RUSTDOCFLAGS='-D warnings' cargo doc`, `ruff check`, `ruff format --check`, `mypy --strict`, `mkdocs build --strict`, and MESA `26.4.1` `file:line` citation resolution enforced in CI.

### 6. Work Tracking (`#140` & `#169`)

| `#140` Phase | Closed | Open | Open Issues |
| :--- | ---: | ---: | :--- |
| Phase 0 | 6 | 0 | none |
| Phase A | 2 | 2 | [#144](https://github.com/joshspeagle/FlexMESA/issues/144), [#168](https://github.com/joshspeagle/FlexMESA/issues/168) |
| Phase B | 2 | 3 | [#148](https://github.com/joshspeagle/FlexMESA/issues/148), [#149](https://github.com/joshspeagle/FlexMESA/issues/149), [#151](https://github.com/joshspeagle/FlexMESA/issues/151) |
| Phase C | 2 | 13 | [#152](https://github.com/joshspeagle/FlexMESA/issues/152), [#153](https://github.com/joshspeagle/FlexMESA/issues/153), [#154](https://github.com/joshspeagle/FlexMESA/issues/154), [#155](https://github.com/joshspeagle/FlexMESA/issues/155), [#156](https://github.com/joshspeagle/FlexMESA/issues/156), [#157](https://github.com/joshspeagle/FlexMESA/issues/157), [#158](https://github.com/joshspeagle/FlexMESA/issues/158), [#159](https://github.com/joshspeagle/FlexMESA/issues/159), [#160](https://github.com/joshspeagle/FlexMESA/issues/160), [#161](https://github.com/joshspeagle/FlexMESA/issues/161), [#162](https://github.com/joshspeagle/FlexMESA/issues/162), [#164](https://github.com/joshspeagle/FlexMESA/issues/164), [#231](https://github.com/joshspeagle/FlexMESA/issues/231) |
| Phase D | 0 | 3 | [#170](https://github.com/joshspeagle/FlexMESA/issues/170), [#171](https://github.com/joshspeagle/FlexMESA/issues/171), [#172](https://github.com/joshspeagle/FlexMESA/issues/172) |
| Phase E | 1 | 1 | [#173](https://github.com/joshspeagle/FlexMESA/issues/173) |
| Deferred scope | 0 | 0 | none |

- **Open pull requests (5)**: [#228](https://github.com/joshspeagle/FlexMESA/pull/228) (draft), [#230](https://github.com/joshspeagle/FlexMESA/pull/230), [#232](https://github.com/joshspeagle/FlexMESA/pull/232), [#233](https://github.com/joshspeagle/FlexMESA/pull/233) (draft), [#234](https://github.com/joshspeagle/FlexMESA/pull/234) (draft).
- **Post-parity deferred backlog ([#169](https://github.com/joshspeagle/FlexMESA/issues/169))**: `18` open issues (`17` `upstream-bug`, `1` `enhancement`; `5` marked `parity-pending` for upstream restoration before `v0.1.0`).

### 7. Headline History Across Merges

| Date (UTC) | Commit | Event | Engine % Ported | All % Verified | Passing Cases | Rust Cov | Python Cov | `#140` Closed/Total | `#169` Backlog |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2026-11-18` | `3b76009` | main merge | 13.9% | 0.0% | 0/106 | 70.8% | 87.7% | 11/33 | 18 |
| `2026-11-18` | `9fb0df9` | main merge | 13.9% | 0.0% | 0/106 | 70.8% | 87.7% | 12/33 | 18 |
| `2026-11-19` | `93de419` | main merge | 13.9% | 0.0% | 0/106 | 73.6% | 87.9% | 12/33 | 18 |
| `2026-11-19` | `b1e4aa2` | main merge | 13.9% | 0.0% | 0/106 | 73.6% | 87.9% | 12/33 | 18 |
| `2026-11-19` | `1131259` | main merge | 13.9% | 0.0% | 0/106 | 74.0% | 87.9% | 12/34 | 18 |
<!-- END GENERATED DASHBOARD -->

## Repository Layout

| Directory | Contents |
| :--- | :--- |
| [`crates/`](crates/) | Canonical Safe Rust Cargo workspace crates (`#![forbid(unsafe_code)]` across all crates except `flexmesa-crlibm-sys`) and `crates/flexmesa-py` (`PyO3` extension) |
| [`src/flexmesa/`](src/flexmesa/) | Python front end, zero-copy `PyO3` bindings (`flexmesa._rust`), and derived `float64` JAX layer |
| [`tests/`](tests/) | Unit, doctest, hybrid bridge, autodiff, and `hypothesis` property test suites |
| [`benchmarks/`](benchmarks/) | Upstream trajectory harness (`benchmarks/history_harness/`), port manifest and dashboard data (`benchmarks/dashboard/`), and Fortran comparators (`benchmarks/fortran_mesa/`) |
| [`docs/`](docs/) | MkDocs Material documentation website (`mkdocs.yml`) |
| [`scripts/`](scripts/) | Reproducible MESA `26.4.1` reference builders, port-manifest builder, and figure scripts |
| [`notes/`](notes/) | Scope and the `v0.1.0` definition ([`SCOPE.md`](notes/SCOPE.md)), decisions in force ([`DECISIONS.md`](notes/DECISIONS.md)), audit registers and generated case table ([`audit/`](notes/audit/)), and unmaintained pre-audit documents ([`archive/`](notes/archive/)) |

## Development Quickstart

```bash
# Build and install the Safe Rust PyO3 extension (flexmesa._rust) and Python package
maturin develop --release -m crates/flexmesa-py/Cargo.toml
pip install -e ".[test,docs]" --no-build-isolation

# Fetch and verify MESA 26.4.1 physics and test-input bundles
flexmesa data fetch --include-test-inputs
flexmesa data verify --include-test-inputs

# Verify the generated case table, port manifest, and progress dashboard
flexmesa harness table --check
flexmesa dashboard check

# Run Python, PyO3, and JAX unit/doctest suites plus static checks
pytest
ruff check src tests
ruff format --check --line-length 80 src tests
mypy --strict src tests

# Run Safe Rust workspace tests and clippy checks
cargo test --workspace --exclude flexmesa-py --release
cargo clippy --workspace --all-targets -- -D warnings

# Build the MkDocs Material documentation website
python scripts/build_docs_figures.py
mkdocs build --strict
```

## License & Upstream Attribution

`FlexMESA` is distributed under the **GNU Lesser General Public License v3.0 or later (`LGPL-3.0-or-later`)**; see [`LICENSE`](LICENSE) for full terms. Physical equations, numerical algorithms, reference test cases, and converted physics tables derive from [MESA (Modules for Experiments in Stellar Astrophysics)](https://github.com/MESAHub/mesa) by Bill Paxton, Frank Timmes, Josiah Schwab, Evan Bauer, Adam Jermyn, and the MESA Team.
