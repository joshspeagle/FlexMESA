# Live Progress & Verification Dashboard

<!-- BEGIN GENERATED DASHBOARD -->
Generated mechanically by `flexmesa dashboard render` from [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml), [`benchmarks/history_harness/`](https://github.com/joshspeagle/FlexMESA/tree/main/benchmarks/history_harness), `cargo llvm-cov`, `pytest --cov`, and the GitHub API ([#174](https://github.com/joshspeagle/FlexMESA/issues/174)). Reference: **MESA release `26.4.1`** (commit `c8c2026`, Zenodo `19722306`).

### Headline Status

| Metric | Value | Source |
| :--- | ---: | :--- |
| `star` + `star_data` + `binary` + `astero` ported (unverified) | 20,107 / 144,952 (13.9%; `star`+`binary`+`astero`: 14.4%) | [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml) |
| Upstream lines verified (all 26 modules) | 0 / 497,703 (0.0%) | [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml) |
| End-to-end MESA `26.4.1` cases passing (`90`/`8`/`8`) | 0 / 106 (102 declared gaps, 4 no reference yet, 0 undeclared failures) | [`case_table.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/case_table.md) |
| Strict mode bit-identical across machines (`#165`) | gated in CI by scripts/check_strict_determinism.py (x86_64 vs aarch64, 1 vs 4 threads; 474,062 math vectors + uncapped star/1.3M_ms_high_Z + binary/jdot_gr_check run with unsupported controls ignored; a bit-identity check, not a comparison with MESA) | [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |
| Strict-mode kernels bit-identical to MESA `26.4.1` (`#165`) | math: 403,258/403,258 CRMATH 0 ULP (70,804 libm <= 1 ULP); 3,394/3,504 probed module hex 0 ULP (2 complete, 12 non-math probed) | [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |
| Test line coverage (Rust `llvm-cov` / Python `pytest --cov`) | 71.1% (66,194/93,038) / 89.6% (19,321/21,565) | [`coverage_summary.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/coverage_summary.json) |
| Road-to-`v0.1.0` packages ([#140](https://github.com/joshspeagle/FlexMESA/issues/140)) & open PRs | 9 closed, 23 open; 1 open PR(s) | [`github_snapshot.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/github_snapshot.json) |
| Post-parity deferred backlog ([#169](https://github.com/joshspeagle/FlexMESA/issues/169)) | 17 open (16 upstream bugs, 1 enhancement, 5 `parity-pending`) | [`github_snapshot.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/github_snapshot.json) |

### 1. Port Coverage by Module (MESA `26.4.1`)

Progress is measured by **upstream MESA `26.4.1` source lines by status**, never by lines of Rust or Python written. No file starts as verified; raising a status requires a pull request that supplies the corresponding routine counterparts or generated MESA `26.4.1` evidence.

| Module | Upstream LOC | Absent | Stub | Partial | Ported (unverified) | Kernel ref. vectors | Case-level evidence | Rust LOC | Python/JAX LOC | % Ported | % Verified |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star` | 120,509 | 19,336 | 18,172 | 65,013 | 17,988 | 0 | 0 | 57,764 | 4,723 | 14.9% | 0.0% |
| `star_data` | 4,870 | 34 | 0 | 4,836 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `binary` | 11,638 | 2,947 | 0 | 6,572 | 2,119 | 0 | 0 | 9,935 | 1,242 | 18.2% | 0.0% |
| `astero` | 7,935 | 2,185 | 0 | 5,750 | 0 | 0 | 0 | 6,219 | 1,206 | 0.0% | 0.0% |
| `eos` | 28,666 | 7,656 | 0 | 21,010 | 0 | 0 | 0 | 13,253 | 9,726 | 0.0% | 0.0% |
| `kap` | 22,430 | 13,652 | 0 | 8,778 | 0 | 0 | 0 | 4,468 | 3,078 | 0.0% | 0.0% |
| `net` | 13,452 | 0 | 0 | 13,452 | 0 | 0 | 0 | 10,256 | 728 | 0.0% | 0.0% |
| `rates` | 17,217 | 192 | 0 | 17,025 | 0 | 0 | 0 | 3,179 | 1,474 | 0.0% | 0.0% |
| `neu` | 2,184 | 352 | 0 | 0 | 1,832 | 0 | 0 | 1,916 | 2,600 | 83.9% | 0.0% |
| `atm` | 6,853 | 2,468 | 0 | 4,385 | 0 | 0 | 0 | 2,096 | 791 | 0.0% | 0.0% |
| `turb` | 2,033 | 0 | 0 | 681 | 1,352 | 0 | 0 | 2,241 | 2,368 | 66.5% | 0.0% |
| `chem` | 4,195 | 373 | 0 | 3,822 | 0 | 0 | 0 | 2,338 | 898 | 0.0% | 0.0% |
| `ionization` | 3,859 | 2,226 | 0 | 1,633 | 0 | 0 | 0 | 2,846 | 982 | 0.0% | 0.0% |
| `colors` | 3,823 | 428 | 0 | 3,395 | 0 | 0 | 0 | 2,453 | 1,499 | 0.0% | 0.0% |
| `num` | 20,045 | 845 | 0 | 19,200 | 0 | 0 | 0 | 4,279 | 3,587 | 0.0% | 0.0% |
| `mtx` | 5,802 | 430 | 0 | 5,372 | 0 | 0 | 0 | 4,173 | 1,768 | 0.0% | 0.0% |
| `interp_1d` | 4,969 | 1,624 | 0 | 3,345 | 0 | 0 | 0 | 1,194 | 1,297 | 0.0% | 0.0% |
| `interp_2d` | 13,565 | 6,661 | 0 | 6,784 | 120 | 0 | 0 | 2,687 | 2,274 | 0.9% | 0.0% |
| `auto_diff` | 14,353 | 1,715 | 0 | 12,638 | 0 | 0 | 0 | 2,747 | 1,234 | 0.0% | 0.0% |
| `math` | 936 | 368 | 0 | 568 | 0 | 0 | 0 | 1,992 | 577 | 0.0% | 0.0% |
| `const` | 256 | 0 | 0 | 0 | 256 | 0 | 0 | 359 | 329 | 100.0% | 0.0% |
| `utils` | 3,633 | 1,555 | 0 | 2,078 | 0 | 0 | 0 | 3,540 | 2,353 | 0.0% | 0.0% |
| `gyre` | 55,708 | 55,708 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `adipls` | 30,923 | 30,923 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `stella` | 62,117 | 62,117 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `plotting` | 35,732 | 35,732 | 0 | 0 | 0 | 0 | 0 | 0 | 2,897 | 0.0% | 0.0% |
| **Subtotal: star + star_data + binary + astero** | 144,952 | 24,502 | 18,172 | 82,171 | 20,107 | 0 | 0 | 73,918 | 7,171 | 13.9% | 0.0% |
| **Total (all 26 modules)** | 497,703 | 249,527 | 18,172 | 206,337 | 23,667 | 0 | 0 | 139,935 | 47,631 | 4.8% | 0.0% |

*Notes*: `star_data` state structs reside in `crates/flexmesa-star`. Bridge, testing, and benchmark infrastructure account for an additional `15,625` Rust lines (`flexmesa-py`, `flexmesa-testing`, `flexmesa-bench`) and `13,438` Python lines (`flexmesa.testing`, `flexmesa.bench`). For plotting ([#173](https://github.com/joshspeagle/FlexMESA/issues/173)), `flexmesa.plotting` currently provides `10` `plot_*` builders against `46` upstream `pgstar`/`pgbinary` panel modules.

### 2. Verification & Numerics Status

| Check | Status | Notes |
| :--- | :--- | :--- |
| Strict mode bit-identical across machines & thread counts (Level 1) | `gated in CI by scripts/check_strict_determinism.py (x86_64 vs aarch64, 1 vs 4 threads; 474,062 math vectors + uncapped star/1.3M_ms_high_Z + binary/jdot_gr_check run with unsupported controls ignored; a bit-identity check, not a comparison with MESA)` | Verified by [#165](https://github.com/joshspeagle/FlexMESA/issues/165) CI jobs (`x86_64` vs `aarch64`) |
| Strict-mode kernels bit-identical to MESA `26.4.1` hex outputs (Level 2) | `math: 403,258/403,258 CRMATH 0 ULP (70,804 libm <= 1 ULP); 3,394/3,504 probed module hex 0 ULP (2 complete, 12 non-math probed)` | `348,321` exact IEEE-754 values captured across 24 in-scope module test suites (`mesa-module-hex-v1`); measured in [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |

| Module | Upstream `26.4.1` Exact Values (`test_output` / hex) | Strict Bit-Identical Compared (`#165`) |
| :--- | ---: | :--- |
| `const` | 23 | `23/23 (0 ULP)` |
| `utils` | 0 | `0` |
| `math` | 102 | `403,258 CRMATH (0 ULP) + 70,804 libm (<= 1 ULP)` |
| `mtx` | 22 | `7/22 (0 ULP partial)` |
| `auto_diff` | 774 | `0` |
| `num` | 570 | `0` |
| `interp_1d` | 234 | `22/234 (0 ULP partial)` |
| `interp_2d` | 104 | `4/104 (0 ULP partial)` |
| `chem` | 1,102 | `286/1,102 (0 ULP partial)` |
| `colors` | 234 | `4/10 0 ULP (max 13 ULP, 234 total)` |
| `eos` | 391 | `0` |
| `kap` | 54 | `13/21 0 ULP (max 1,459,407,697 ULP, 21 total)` |
| `rates` | 3,685 | `5/9 0 ULP (max 6 ULP, 3,685 total)` |
| `neu` | 340,000 | `2,935/3,000 0 ULP (max 68 ULP, 300,000 total)` |
| `net` | 140 | `0` |
| `ionization` | 41 | `28/31 0 ULP (max 10,652,452 ULP, 31 total)` |
| `atm` | 90 | `0` |
| `turb` | 136 | `38/62 0 ULP (max 6 ULP, 71 total)` |
| `star_data` | 0 | `0` |
| `gyre` | 136 | `0` |
| `adipls` | 92 | `0` |
| `star` | 386 | `0` |
| `binary` | 5 | `5/5 (0 ULP)` |
| `astero` | 0 | `0` |

### 3. End-to-End Cases (`benchmarks/history_harness`)

| Suite | `26.4.1` Cases | Stored `26.4.1` Refs | No Ref Yet | Strict-Validated | Diagnostic Runs | Reach Upstream Term | Within Numeric Tol | Verified Pass | Declared Gaps | Undeclared Failures |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star` | 90 | 86 | 4 | 0 | 90 | 35 | 5 | 0 | 86 | 0 |
| `binary` | 8 | 8 | 0 | 0 | 8 | 2 | 0 | 0 | 8 | 0 |
| `astero` | 8 | 8 | 0 | 0 | 8 | 4 | 0 | 0 | 8 | 0 |
| **Total** | **106** | **102** | **4** | **0** | **106** | **41** | **5** | **0** | **102** | **0** |

- **Full per-case table**: [`notes/audit/case_table.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/case_table.md). `5` case(s) within numeric tolerance (`rsp_BEP`, `rsp_BLAP`, `rsp_Cepheid`, `rsp_RR_Lyrae`, `rsp_save_and_load_file`) remain governed as declared gaps under [#148](https://github.com/joshspeagle/FlexMESA/issues/148).
- **Declared gaps by owning package**: [#152](https://github.com/joshspeagle/FlexMESA/issues/152) (18), [#148](https://github.com/joshspeagle/FlexMESA/issues/148) (17), [#154](https://github.com/joshspeagle/FlexMESA/issues/154) (11), [#161](https://github.com/joshspeagle/FlexMESA/issues/161) (9), [#162](https://github.com/joshspeagle/FlexMESA/issues/162) (9), [#157](https://github.com/joshspeagle/FlexMESA/issues/157) (7), [#160](https://github.com/joshspeagle/FlexMESA/issues/160) (7), [#170](https://github.com/joshspeagle/FlexMESA/issues/170) (7), [#155](https://github.com/joshspeagle/FlexMESA/issues/155) (6), [#158](https://github.com/joshspeagle/FlexMESA/issues/158) (4), [#156](https://github.com/joshspeagle/FlexMESA/issues/156) (2), [#159](https://github.com/joshspeagle/FlexMESA/issues/159) (2), [#150](https://github.com/joshspeagle/FlexMESA/issues/150) (1), [#151](https://github.com/joshspeagle/FlexMESA/issues/151) (1), [#171](https://github.com/joshspeagle/FlexMESA/issues/171) (1).
- **Trajectory comparison figures ([#173](https://github.com/joshspeagle/FlexMESA/issues/173))**: `0` committed.

### 4. Controls & Defaults Coverage (`#147`, `#149`)

| Namelist | Upstream `26.4.1` Scalars | Matched in `FlexMESA` | Defaults Agree | Value Mismatch | Active Limit `None` | Unimplemented |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `star &controls` | 1,396 | 582 | 535 | 31 | 16 | 814 |
| `star &star_job` | 431 | 79 | 62 | 11 | 6 | 352 |
| `binary &binary_controls + &binary_job` | 209 | 120 | 117 | 3 | 0 | 89 |
| `astero &astero_search_controls` | 150 | 73 | 70 | 3 | 0 | 77 |
| `eos &eos` | 102 | 54 | 42 | 12 | 0 | 48 |
| `kap &kap` | 36 | 16 | 14 | 2 | 0 | 20 |
| `star &pgstar` | 2,900 | 0 | 0 | 0 | 0 | 2,900 |
| `binary &pgbinary` | 1,329 | 0 | 0 | 0 | 0 | 1,329 |

Full scalar diff: [`notes/audit/defaults_diff_26_4_1.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/defaults_diff_26_4_1.md) (loud errors on unknown controls tracked in [#147](https://github.com/joshspeagle/FlexMESA/issues/147); defaults alignment tracked in [#149](https://github.com/joshspeagle/FlexMESA/issues/149)).

### 5. Test & Code Health

- **Rust workspace line coverage (`cargo llvm-cov`)**: **71.1%** (`66,194` / `93,038` lines across `26` crates).
- **Python package statement coverage (`pytest --cov`)**: **89.6%** (`19,321` / `21,565` statements across `27` subpackages).
- **Ignored / skipped tests**: `13` Rust `#[ignore]` ([#143](https://github.com/joshspeagle/FlexMESA/issues/143), [#160](https://github.com/joshspeagle/FlexMESA/issues/160), [#161](https://github.com/joshspeagle/FlexMESA/issues/161), [#162](https://github.com/joshspeagle/FlexMESA/issues/162)) and `1` Python `xfail`/`skip` ([#153](https://github.com/joshspeagle/FlexMESA/issues/153); `0` unlinked).
- **Static & documentation gates**: `cargo clippy --all-targets -- -D warnings`, `RUSTDOCFLAGS='-D warnings' cargo doc`, `ruff check`, `ruff format --check`, `mypy --strict`, `mkdocs build --strict`, and MESA `26.4.1` `file:line` citation resolution enforced in CI.

#### Rust Line Coverage by Crate (`cargo llvm-cov`)

| Crate | Executable Lines | Covered Lines | Line Coverage |
| :--- | ---: | ---: | ---: |
| `flexmesa-astero` | 4,432 | 1,437 | 32.4% |
| `flexmesa-atm` | 1,332 | 996 | 74.8% |
| `flexmesa-autodiff` | 1,283 | 461 | 35.9% |
| `flexmesa-bench` | 2,408 | 2,334 | 96.9% |
| `flexmesa-binary` | 6,734 | 3,980 | 59.1% |
| `flexmesa-chem` | 908 | 632 | 69.6% |
| `flexmesa-colors` | 1,297 | 888 | 68.5% |
| `flexmesa-const` | 19 | 16 | 84.2% |
| `flexmesa-crlibm-sys` | 78 | 36 | 46.2% |
| `flexmesa-data` | 986 | 675 | 68.5% |
| `flexmesa-eos` | 8,893 | 7,178 | 80.7% |
| `flexmesa-interp1d` | 683 | 579 | 84.8% |
| `flexmesa-interp2d` | 1,711 | 1,318 | 77.0% |
| `flexmesa-ionization` | 1,001 | 674 | 67.3% |
| `flexmesa-kap` | 3,141 | 1,978 | 63.0% |
| `flexmesa-math` | 775 | 589 | 76.0% |
| `flexmesa-mtx` | 2,783 | 2,443 | 87.8% |
| `flexmesa-net` | 7,601 | 6,575 | 86.5% |
| `flexmesa-neu` | 994 | 962 | 96.8% |
| `flexmesa-num` | 2,860 | 2,468 | 86.3% |
| `flexmesa-rates` | 2,109 | 1,844 | 87.4% |
| `flexmesa-schema` | 808 | 666 | 82.4% |
| `flexmesa-star` | 38,531 | 26,156 | 67.9% |
| `flexmesa-testing` | 253 | 197 | 77.9% |
| `flexmesa-turb` | 1,093 | 899 | 82.3% |
| `flexmesa-utils` | 325 | 213 | 65.5% |

#### Python Statement Coverage by Subpackage (`pytest --cov`)

| Subpackage | Statements | Covered | Coverage |
| :--- | ---: | ---: | ---: |
| `flexmesa` | 101 | 91 | 90.1% |
| `flexmesa.astero` | 271 | 268 | 98.9% |
| `flexmesa.atm` | 180 | 173 | 96.1% |
| `flexmesa.auto_diff` | 563 | 482 | 85.6% |
| `flexmesa.bench` | 1,370 | 1,365 | 99.6% |
| `flexmesa.binary` | 321 | 309 | 96.3% |
| `flexmesa.chem` | 240 | 231 | 96.2% |
| `flexmesa.colors` | 445 | 367 | 82.5% |
| `flexmesa.constants` | 107 | 107 | 100.0% |
| `flexmesa.data` | 541 | 428 | 79.1% |
| `flexmesa.eos` | 3,478 | 3,282 | 94.4% |
| `flexmesa.interp_1d` | 432 | 380 | 88.0% |
| `flexmesa.interp_2d` | 1,055 | 865 | 82.0% |
| `flexmesa.ionization` | 312 | 300 | 96.2% |
| `flexmesa.kap` | 1,308 | 1,269 | 97.0% |
| `flexmesa.math` | 178 | 178 | 100.0% |
| `flexmesa.mtx` | 610 | 512 | 83.9% |
| `flexmesa.net` | 148 | 148 | 100.0% |
| `flexmesa.neu` | 1,066 | 947 | 88.8% |
| `flexmesa.num` | 1,596 | 1,394 | 87.3% |
| `flexmesa.plotting` | 797 | 751 | 94.2% |
| `flexmesa.rates` | 496 | 484 | 97.6% |
| `flexmesa.schema` | 137 | 112 | 81.8% |
| `flexmesa.star` | 1,438 | 1,334 | 92.8% |
| `flexmesa.testing` | 3,377 | 2,661 | 78.8% |
| `flexmesa.turb` | 912 | 803 | 88.0% |
| `flexmesa.utils` | 86 | 80 | 93.0% |

### 6. Work Tracking (`#140` & `#169`)

| `#140` Phase | Closed | Open | Open Issues |
| :--- | ---: | ---: | :--- |
| Phase 0 | 6 | 0 | none |
| Phase A | 2 | 2 | [#144](https://github.com/joshspeagle/FlexMESA/issues/144), [#168](https://github.com/joshspeagle/FlexMESA/issues/168) |
| Phase B | 1 | 4 | [#148](https://github.com/joshspeagle/FlexMESA/issues/148), [#149](https://github.com/joshspeagle/FlexMESA/issues/149), [#150](https://github.com/joshspeagle/FlexMESA/issues/150), [#151](https://github.com/joshspeagle/FlexMESA/issues/151) |
| Phase C | 0 | 13 | [#152](https://github.com/joshspeagle/FlexMESA/issues/152), [#153](https://github.com/joshspeagle/FlexMESA/issues/153), [#154](https://github.com/joshspeagle/FlexMESA/issues/154), [#155](https://github.com/joshspeagle/FlexMESA/issues/155), [#156](https://github.com/joshspeagle/FlexMESA/issues/156), [#157](https://github.com/joshspeagle/FlexMESA/issues/157), [#158](https://github.com/joshspeagle/FlexMESA/issues/158), [#159](https://github.com/joshspeagle/FlexMESA/issues/159), [#160](https://github.com/joshspeagle/FlexMESA/issues/160), [#161](https://github.com/joshspeagle/FlexMESA/issues/161), [#162](https://github.com/joshspeagle/FlexMESA/issues/162), [#164](https://github.com/joshspeagle/FlexMESA/issues/164), [#166](https://github.com/joshspeagle/FlexMESA/issues/166) |
| Phase D | 0 | 3 | [#170](https://github.com/joshspeagle/FlexMESA/issues/170), [#171](https://github.com/joshspeagle/FlexMESA/issues/171), [#172](https://github.com/joshspeagle/FlexMESA/issues/172) |
| Phase E | 1 | 1 | [#173](https://github.com/joshspeagle/FlexMESA/issues/173) |
| Deferred scope | 0 | 0 | none |

- **Open pull requests (1)**: [#215](https://github.com/joshspeagle/FlexMESA/pull/215).
- **Post-parity deferred backlog ([#169](https://github.com/joshspeagle/FlexMESA/issues/169))**: `17` open issues (`16` `upstream-bug`, `1` `enhancement`; `5` marked `parity-pending` for upstream restoration before `v0.1.0`).

### 7. Headline History Across Merges

| Date (UTC) | Commit | Event | Engine % Ported | All % Verified | Passing Cases | Rust Cov | Python Cov | `#140` Closed/Total | `#169` Backlog |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2026-11-16` | `f707f99` | PR #199 (#141, #143, #145, #163) | 13.9% | 0.0% | 0/106 | 74.9% | 87.8% | 4/32 | 17 |
| `2026-11-16` | `9c77162` | PR #195 (#165) | 13.9% | 0.0% | 0/106 | 71.6% | 87.7% | 5/32 | 17 |
| `2026-11-16` | `8e1b8b4` | PR #201 (#161) | 13.9% | 0.0% | 0/106 | 71.6% | 87.7% | 5/32 | 17 |
| `2026-11-17` | `f005f2e` | main merge | 13.9% | 0.0% | 0/106 | 72.0% | 87.8% | 6/32 | 17 |
| `2026-11-17` | `39e6407` | main merge | 13.9% | 0.0% | 0/106 | 72.0% | 87.8% | 7/32 | 17 |
| `2026-11-17` | `bdd7b91` | main merge | 13.9% | 0.0% | 0/106 | 71.1% | 87.6% | 8/32 | 17 |
| `2026-11-17` | `69e6f68` | main merge | 13.9% | 0.0% | 0/106 | 71.2% | 87.9% | 7/32 | 17 |
| `2026-11-17` | `c34f5e7` | main merge | 13.9% | 0.0% | 0/106 | 71.2% | 87.9% | 7/32 | 17 |
| `2026-11-17` | `0b78ce7` | main merge | 13.9% | 0.0% | 0/106 | 71.2% | 87.9% | 8/32 | 17 |
| `2026-11-17` | `6df1dcf` | main merge | 13.9% | 0.0% | 0/106 | 71.2% | 87.7% | 8/32 | 17 |
| `2026-11-17` | `0fec8d2` | main merge | 13.9% | 0.0% | 0/106 | 71.2% | 87.7% | 7/32 | 17 |
| `2026-11-18` | `28a3e51` | main merge | 13.9% | 0.0% | 0/106 | 71.2% | 87.7% | 7/32 | 17 |
| `2026-11-18` | `663ea6f` | main merge | 13.9% | 0.0% | 0/106 | 71.2% | 87.7% | 7/32 | 17 |
| `2026-11-18` | `5549331` | main merge | 13.9% | 0.0% | 0/106 | 71.1% | 87.7% | 9/32 | 17 |
| `2026-11-18` | `dd2352d` | main merge | 13.9% | 0.0% | 0/106 | 71.1% | 87.7% | 9/32 | 17 |
| `2026-11-18` | `d565c82` | main merge | 13.9% | 0.0% | 0/106 | 71.1% | 87.7% | 9/32 | 17 |
| `2026-11-18` | `3153e32` | main merge | 13.9% | 0.0% | 0/106 | 71.1% | 89.6% | 9/32 | 17 |
<!-- END GENERATED DASHBOARD -->
