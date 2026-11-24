# Live Progress & Verification Dashboard

<!-- BEGIN GENERATED DASHBOARD -->
Generated mechanically by `flexmesa dashboard render` from [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml), [`benchmarks/history_harness/`](https://github.com/joshspeagle/FlexMESA/tree/main/benchmarks/history_harness), `cargo llvm-cov`, `pytest --cov`, and the GitHub API ([#174](https://github.com/joshspeagle/FlexMESA/issues/174)). Reference: **MESA release `26.4.1`** (commit `c8c2026`, Zenodo `19722306`).

### Headline Status

| Metric | Value | Source |
| :--- | ---: | :--- |
| `star` + `star_data` + `binary` + `astero` ported (unverified) | 44,469 / 144,952 (30.7%; `star`+`binary`+`astero`: 31.7%) | [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml) |
| Upstream lines verified (all 26 modules) | 5,640 / 497,703 (1.1%) | [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml) |
| End-to-end MESA `26.4.1` cases passing (`90`/`8`/`8`) | 27 / 106 (70 declared gaps, 4 no reference yet, 0 undeclared failures) | [`case_table.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/case_table.md) |
| Strict mode bit-identical across machines (`#165`) | gated in CI by scripts/check_strict_determinism.py (x86_64 vs aarch64, 1 vs 4 threads; 474,062 math vectors + uncapped star/1.3M_ms_high_Z + binary/jdot_gr_check run with unsupported controls ignored; a bit-identity check, not a comparison with MESA) | [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |
| Strict-mode kernels bit-identical to MESA `26.4.1` (`#165`) | math: 403,258/403,258 CRMATH 0 ULP (70,804 libm <= 1 ULP); 304,760/304,760 probed module hex 0 ULP (10 complete, 15 non-math probed) | [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |
| Test line coverage (Rust `llvm-cov` / Python `pytest --cov`) | 76.3% (97,612/127,870) / 87.7% (23,577/26,872) | [`coverage_summary.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/coverage_summary.json) |
| Road-to-`v0.1.0` packages ([#140](https://github.com/joshspeagle/FlexMESA/issues/140)) & open PRs | 27 closed, 23 open; 10 open PR(s) | [`github_snapshot.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/github_snapshot.json) |
| Post-parity deferred backlog ([#169](https://github.com/joshspeagle/FlexMESA/issues/169)) | 69 open (63 upstream bugs, 6 enhancements, 4 `parity-pending`) | [`github_snapshot.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/github_snapshot.json) |

### 1. Port Coverage by Module (MESA `26.4.1`)

Progress is measured by **upstream MESA `26.4.1` source lines by status**, never by lines of Rust or Python written. No file starts as verified; raising a status requires a pull request that supplies the corresponding routine counterparts or generated MESA `26.4.1` evidence.

| Module | Upstream LOC | Absent | Stub | Partial | Ported (unverified) | Kernel ref. vectors | Case-level evidence | Rust LOC | Python/JAX LOC | % Ported | % Verified |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star` | 120,509 | 17,759 | 6,378 | 54,084 | 36,648 | 5,640 | 0 | 97,770 | 3,893 | 35.1% | 4.7% |
| `star_data` | 4,870 | 34 | 0 | 4,836 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `binary` | 11,638 | 2,885 | 0 | 6,572 | 2,181 | 0 | 0 | 12,661 | 1,244 | 18.7% | 0.0% |
| `astero` | 7,935 | 2,185 | 0 | 5,750 | 0 | 0 | 0 | 6,333 | 1,295 | 0.0% | 0.0% |
| `eos` | 28,666 | 7,656 | 0 | 21,010 | 0 | 0 | 0 | 15,780 | 9,911 | 0.0% | 0.0% |
| `kap` | 22,430 | 13,652 | 0 | 8,778 | 0 | 0 | 0 | 7,147 | 3,359 | 0.0% | 0.0% |
| `net` | 13,452 | 0 | 0 | 13,452 | 0 | 0 | 0 | 10,880 | 728 | 0.0% | 0.0% |
| `rates` | 17,217 | 192 | 0 | 17,025 | 0 | 0 | 0 | 4,137 | 1,478 | 0.0% | 0.0% |
| `neu` | 2,184 | 352 | 0 | 0 | 1,832 | 0 | 0 | 1,948 | 2,673 | 83.9% | 0.0% |
| `atm` | 6,853 | 2,468 | 0 | 4,385 | 0 | 0 | 0 | 2,710 | 984 | 0.0% | 0.0% |
| `turb` | 2,033 | 0 | 0 | 681 | 1,352 | 0 | 0 | 2,848 | 3,116 | 66.5% | 0.0% |
| `chem` | 4,195 | 373 | 0 | 3,822 | 0 | 0 | 0 | 2,361 | 930 | 0.0% | 0.0% |
| `ionization` | 3,859 | 2,226 | 0 | 1,633 | 0 | 0 | 0 | 4,535 | 1,033 | 0.0% | 0.0% |
| `colors` | 3,823 | 428 | 0 | 3,395 | 0 | 0 | 0 | 3,922 | 1,500 | 0.0% | 0.0% |
| `num` | 20,045 | 845 | 0 | 19,200 | 0 | 0 | 0 | 4,647 | 3,587 | 0.0% | 0.0% |
| `mtx` | 5,802 | 430 | 0 | 5,372 | 0 | 0 | 0 | 4,179 | 1,775 | 0.0% | 0.0% |
| `interp_1d` | 4,969 | 1,624 | 0 | 3,345 | 0 | 0 | 0 | 1,205 | 1,311 | 0.0% | 0.0% |
| `interp_2d` | 13,565 | 6,661 | 0 | 6,784 | 120 | 0 | 0 | 2,687 | 2,288 | 0.9% | 0.0% |
| `auto_diff` | 14,353 | 1,715 | 0 | 12,638 | 0 | 0 | 0 | 5,834 | 1,234 | 0.0% | 0.0% |
| `math` | 936 | 368 | 0 | 568 | 0 | 0 | 0 | 2,629 | 577 | 0.0% | 0.0% |
| `const` | 256 | 0 | 0 | 0 | 256 | 0 | 0 | 368 | 329 | 100.0% | 0.0% |
| `utils` | 3,633 | 1,555 | 0 | 2,078 | 0 | 0 | 0 | 5,090 | 2,676 | 0.0% | 0.0% |
| `gyre` | 55,708 | 55,708 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `adipls` | 30,923 | 30,923 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `stella` | 62,117 | 62,117 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `plotting` | 35,732 | 35,732 | 0 | 0 | 0 | 0 | 0 | 0 | 12,095 | 0.0% | 0.0% |
| **Subtotal: star + star_data + binary + astero** | 144,952 | 22,863 | 6,378 | 71,242 | 38,829 | 5,640 | 0 | 116,764 | 6,432 | 30.7% | 3.9% |
| **Total (all 26 modules)** | 497,703 | 247,888 | 6,378 | 195,408 | 42,389 | 5,640 | 0 | 199,671 | 58,016 | 9.7% | 1.1% |

*Notes*: `star_data` state structs reside in `crates/flexmesa-star`. Bridge, testing, and benchmark infrastructure account for an additional `20,182` Rust lines (`flexmesa-py`, `flexmesa-testing`, `flexmesa-bench`) and `17,848` Python lines (`flexmesa.testing`, `flexmesa.bench`). For plotting ([#173](https://github.com/joshspeagle/FlexMESA/issues/173)), `flexmesa.plotting` currently provides `34` `plot_*` builders against `46` upstream `pgstar`/`pgbinary` panel modules.

### 2. Verification & Numerics Status

| Check | Status | Notes |
| :--- | :--- | :--- |
| Strict mode bit-identical across machines & thread counts (Level 1) | `gated in CI by scripts/check_strict_determinism.py (x86_64 vs aarch64, 1 vs 4 threads; 474,062 math vectors + uncapped star/1.3M_ms_high_Z + binary/jdot_gr_check run with unsupported controls ignored; a bit-identity check, not a comparison with MESA)` | Verified by [#165](https://github.com/joshspeagle/FlexMESA/issues/165) CI jobs (`x86_64` vs `aarch64`) |
| Strict-mode kernels bit-identical to MESA `26.4.1` hex outputs (Level 2) | `math: 403,258/403,258 CRMATH 0 ULP (70,804 libm <= 1 ULP); 304,760/304,760 probed module hex 0 ULP (10 complete, 15 non-math probed)` | `348,321` exact IEEE-754 values captured across 24 in-scope module test suites (`mesa-module-hex-v1`); measured in [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |

| Module | Upstream `26.4.1` Exact Values (`test_output` / hex) | Strict Bit-Identical Compared (`#165`) |
| :--- | ---: | :--- |
| `const` | 23 | `23/23 (0 ULP)` |
| `utils` | 0 | `0` |
| `math` | 102 | `403,258 CRMATH (0 ULP) + 70,804 libm (<= 1 ULP)` |
| `mtx` | 22 | `7/22 (0 ULP partial)` |
| `auto_diff` | 774 | `387/387 (0 ULP)` |
| `num` | 570 | `0` |
| `interp_1d` | 234 | `22/234 (0 ULP partial)` |
| `interp_2d` | 104 | `4/104 (0 ULP partial)` |
| `chem` | 1,102 | `286/1,102 (0 ULP partial)` |
| `colors` | 234 | `10/234 (0 ULP partial)` |
| `eos` | 391 | `0` |
| `kap` | 54 | `21/21 (0 ULP)` |
| `rates` | 3,685 | `3,639/3,639 (0 ULP)` |
| `neu` | 340,000 | `300,000/300,000 (0 ULP)` |
| `net` | 140 | `116/124 (0 ULP partial)` |
| `ionization` | 41 | `31/31 (0 ULP)` |
| `atm` | 90 | `39/39 (0 ULP)` |
| `turb` | 136 | `71/71 (0 ULP)` |
| `star_data` | 0 | `0` |
| `gyre` | 136 | `0` |
| `adipls` | 92 | `0` |
| `star` | 386 | `0` |
| `binary` | 5 | `5/5 (0 ULP)` |
| `astero` | 0 | `0` |

### 3. End-to-End Cases (`benchmarks/history_harness`)

| Suite | `26.4.1` Cases | Stored `26.4.1` Refs | No Ref Yet | Strict-Validated | Diagnostic Runs | Reach Upstream Term | Within Numeric Tol | Verified Pass | Declared Gaps | Undeclared Failures |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star` | 90 | 86 | 4 | 56 | 34 | 54 | 30 | 23 | 58 | 0 |
| `binary` | 8 | 8 | 0 | 7 | 1 | 6 | 3 | 3 | 5 | 0 |
| `astero` | 8 | 8 | 0 | 1 | 7 | 3 | 3 | 1 | 7 | 0 |
| **Total** | **106** | **102** | **4** | **64** | **42** | **63** | **36** | **27** | **70** | **0** |

- **Full per-case table**: [`notes/audit/case_table.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/case_table.md). `4` case(s) within numeric tolerance (`carbon_kh`, `rsp_save_and_load_file`, `astero_adipls`, `surface_effects`) remain governed as declared gaps under [#161](https://github.com/joshspeagle/FlexMESA/issues/161), [#162](https://github.com/joshspeagle/FlexMESA/issues/162), [#171](https://github.com/joshspeagle/FlexMESA/issues/171).
- **Declared gaps by owning package**: [#161](https://github.com/joshspeagle/FlexMESA/issues/161) (23), [#154](https://github.com/joshspeagle/FlexMESA/issues/154) (13), [#162](https://github.com/joshspeagle/FlexMESA/issues/162) (9), [#157](https://github.com/joshspeagle/FlexMESA/issues/157) (8), [#170](https://github.com/joshspeagle/FlexMESA/issues/170) (6), [#158](https://github.com/joshspeagle/FlexMESA/issues/158) (5), [#160](https://github.com/joshspeagle/FlexMESA/issues/160) (4), [#156](https://github.com/joshspeagle/FlexMESA/issues/156) (2), [#231](https://github.com/joshspeagle/FlexMESA/issues/231) (2), [#152](https://github.com/joshspeagle/FlexMESA/issues/152) (1), [#153](https://github.com/joshspeagle/FlexMESA/issues/153) (1), [#171](https://github.com/joshspeagle/FlexMESA/issues/171) (1).
- **Trajectory comparison figures ([#173](https://github.com/joshspeagle/FlexMESA/issues/173))**: `0` committed.

### 4. Controls & Defaults Coverage (`#147`, `#149`)

| Namelist | Upstream `26.4.1` Scalars | Matched in `FlexMESA` | Defaults Agree | Value Mismatch | Active Limit `None` | Non-Scalar Rust | Unimplemented |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star &controls` | 1,444 | 1,131 | 1,131 | 0 | 0 | 17 | 296 |
| `star &star_job` | 455 | 164 | 164 | 0 | 0 | 1 | 290 |
| `binary &binary_controls + &binary_job` | 209 | 145 | 145 | 0 | 0 | 1 | 63 |
| `astero &astero_search_controls` | 150 | 75 | 75 | 0 | 0 | 1 | 74 |
| `eos &eos` | 102 | 60 | 60 | 0 | 0 | 0 | 42 |
| `kap &kap` | 36 | 21 | 21 | 0 | 0 | 0 | 15 |
| `star &pgstar` | 2,900 | 0 | 0 | 0 | 0 | 0 | 2,900 |
| `binary &pgbinary` | 1,329 | 0 | 0 | 0 | 0 | 0 | 1,329 |

Full scalar diff: [`notes/audit/defaults_diff_26_4_1.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/defaults_diff_26_4_1.md) (loud errors on unknown controls tracked in [#147](https://github.com/joshspeagle/FlexMESA/issues/147); defaults alignment tracked in [#149](https://github.com/joshspeagle/FlexMESA/issues/149)).

### 5. Test & Code Health

- **Rust workspace line coverage (`cargo llvm-cov`)**: **76.3%** (`97,612` / `127,870` lines across `26` crates).
- **Python package statement coverage (`pytest --cov`)**: **87.7%** (`23,577` / `26,872` statements across `27` subpackages).
- **Ignored / skipped tests**: `13` Rust `#[ignore]` ([#143](https://github.com/joshspeagle/FlexMESA/issues/143), [#154](https://github.com/joshspeagle/FlexMESA/issues/154), [#160](https://github.com/joshspeagle/FlexMESA/issues/160), [#162](https://github.com/joshspeagle/FlexMESA/issues/162)) and `24` Python `xfail`/`skip` ([#43](https://github.com/joshspeagle/FlexMESA/issues/43), [#154](https://github.com/joshspeagle/FlexMESA/issues/154), [#156](https://github.com/joshspeagle/FlexMESA/issues/156), [#161](https://github.com/joshspeagle/FlexMESA/issues/161), [#291](https://github.com/joshspeagle/FlexMESA/issues/291), [#295](https://github.com/joshspeagle/FlexMESA/issues/295), [#340](https://github.com/joshspeagle/FlexMESA/issues/340), [#349](https://github.com/joshspeagle/FlexMESA/issues/349), [#358](https://github.com/joshspeagle/FlexMESA/issues/358), [#402](https://github.com/joshspeagle/FlexMESA/issues/402), [#413](https://github.com/joshspeagle/FlexMESA/issues/413), [#420](https://github.com/joshspeagle/FlexMESA/issues/420), [#433](https://github.com/joshspeagle/FlexMESA/issues/433), [#445](https://github.com/joshspeagle/FlexMESA/issues/445); `0` unlinked).
- **Static & documentation gates**: `cargo clippy --all-targets -- -D warnings`, `RUSTDOCFLAGS='-D warnings' cargo doc`, `ruff check`, `ruff format --check`, `mypy --strict`, `mkdocs build --strict`, and MESA `26.4.1` `file:line` citation resolution enforced in CI.

#### Rust Line Coverage by Crate (`cargo llvm-cov`)

| Crate | Executable Lines | Covered Lines | Line Coverage |
| :--- | ---: | ---: | ---: |
| `flexmesa-astero` | 4,222 | 1,642 | 38.9% |
| `flexmesa-atm` | 1,792 | 1,569 | 87.6% |
| `flexmesa-autodiff` | 3,599 | 1,651 | 45.9% |
| `flexmesa-bench` | 3,529 | 2,367 | 67.1% |
| `flexmesa-binary` | 8,334 | 5,521 | 66.2% |
| `flexmesa-chem` | 911 | 671 | 73.7% |
| `flexmesa-colors` | 1,308 | 903 | 69.0% |
| `flexmesa-const` | 19 | 19 | 100.0% |
| `flexmesa-crlibm-sys` | 78 | 78 | 100.0% |
| `flexmesa-data` | 948 | 666 | 70.3% |
| `flexmesa-eos` | 10,678 | 8,815 | 82.6% |
| `flexmesa-interp1d` | 668 | 565 | 84.6% |
| `flexmesa-interp2d` | 1,691 | 1,303 | 77.1% |
| `flexmesa-ionization` | 2,072 | 1,788 | 86.3% |
| `flexmesa-kap` | 4,876 | 3,735 | 76.6% |
| `flexmesa-math` | 1,070 | 922 | 86.2% |
| `flexmesa-mtx` | 2,763 | 2,434 | 88.1% |
| `flexmesa-net` | 8,032 | 7,224 | 89.9% |
| `flexmesa-neu` | 1,020 | 989 | 97.0% |
| `flexmesa-num` | 2,991 | 2,603 | 87.0% |
| `flexmesa-rates` | 2,752 | 2,480 | 90.1% |
| `flexmesa-schema` | 790 | 660 | 83.5% |
| `flexmesa-star` | 60,740 | 46,656 | 76.8% |
| `flexmesa-testing` | 249 | 194 | 77.9% |
| `flexmesa-turb` | 1,391 | 1,210 | 87.0% |
| `flexmesa-utils` | 1,347 | 947 | 70.3% |

#### Python Statement Coverage by Subpackage (`pytest --cov`)

| Subpackage | Statements | Covered | Coverage |
| :--- | ---: | ---: | ---: |
| `flexmesa` | 152 | 128 | 84.2% |
| `flexmesa.astero` | 271 | 268 | 98.9% |
| `flexmesa.atm` | 198 | 194 | 98.0% |
| `flexmesa.auto_diff` | 563 | 482 | 85.6% |
| `flexmesa.bench` | 1,388 | 1,383 | 99.6% |
| `flexmesa.binary` | 331 | 322 | 97.3% |
| `flexmesa.chem` | 246 | 234 | 95.1% |
| `flexmesa.colors` | 445 | 367 | 82.5% |
| `flexmesa.constants` | 107 | 107 | 100.0% |
| `flexmesa.data` | 695 | 570 | 82.0% |
| `flexmesa.eos` | 3,514 | 3,338 | 95.0% |
| `flexmesa.interp_1d` | 429 | 385 | 89.7% |
| `flexmesa.interp_2d` | 1,053 | 863 | 82.0% |
| `flexmesa.ionization` | 334 | 312 | 93.4% |
| `flexmesa.kap` | 1,469 | 1,409 | 95.9% |
| `flexmesa.math` | 178 | 178 | 100.0% |
| `flexmesa.mtx` | 610 | 517 | 84.8% |
| `flexmesa.net` | 148 | 148 | 100.0% |
| `flexmesa.neu` | 1,059 | 524 | 49.5% |
| `flexmesa.num` | 1,596 | 1,401 | 87.8% |
| `flexmesa.plotting` | 4,235 | 3,709 | 87.6% |
| `flexmesa.rates` | 496 | 484 | 97.6% |
| `flexmesa.schema` | 137 | 112 | 81.8% |
| `flexmesa.star` | 1,303 | 1,261 | 96.8% |
| `flexmesa.testing` | 4,780 | 3,867 | 80.9% |
| `flexmesa.turb` | 1,049 | 934 | 89.0% |
| `flexmesa.utils` | 86 | 80 | 93.0% |

### 6. Work Tracking (`#140` & `#169`)

| `#140` Phase | Closed | Open | Open Issues |
| :--- | ---: | ---: | :--- |
| Phase 0 | 6 | 0 | none |
| Phase A | 4 | 0 | none |
| Phase B | 5 | 0 | none |
| Phase C | 12 | 18 | [#152](https://github.com/joshspeagle/FlexMESA/issues/152), [#153](https://github.com/joshspeagle/FlexMESA/issues/153), [#154](https://github.com/joshspeagle/FlexMESA/issues/154), [#155](https://github.com/joshspeagle/FlexMESA/issues/155), [#156](https://github.com/joshspeagle/FlexMESA/issues/156), [#157](https://github.com/joshspeagle/FlexMESA/issues/157), [#158](https://github.com/joshspeagle/FlexMESA/issues/158), [#159](https://github.com/joshspeagle/FlexMESA/issues/159), [#160](https://github.com/joshspeagle/FlexMESA/issues/160), [#161](https://github.com/joshspeagle/FlexMESA/issues/161), [#162](https://github.com/joshspeagle/FlexMESA/issues/162), [#231](https://github.com/joshspeagle/FlexMESA/issues/231), [#295](https://github.com/joshspeagle/FlexMESA/issues/295), [#299](https://github.com/joshspeagle/FlexMESA/issues/299), [#365](https://github.com/joshspeagle/FlexMESA/issues/365), [#436](https://github.com/joshspeagle/FlexMESA/issues/436), [#445](https://github.com/joshspeagle/FlexMESA/issues/445), [#453](https://github.com/joshspeagle/FlexMESA/issues/453) |
| Phase D | 0 | 3 | [#170](https://github.com/joshspeagle/FlexMESA/issues/170), [#171](https://github.com/joshspeagle/FlexMESA/issues/171), [#172](https://github.com/joshspeagle/FlexMESA/issues/172) |
| Phase E | 1 | 1 | [#173](https://github.com/joshspeagle/FlexMESA/issues/173) |
| Deferred scope | 0 | 1 | [#477](https://github.com/joshspeagle/FlexMESA/issues/477) |

- **Open pull requests (10)**: [#507](https://github.com/joshspeagle/FlexMESA/pull/507) (draft), [#508](https://github.com/joshspeagle/FlexMESA/pull/508) (draft), [#509](https://github.com/joshspeagle/FlexMESA/pull/509) (draft), [#512](https://github.com/joshspeagle/FlexMESA/pull/512) (draft), [#515](https://github.com/joshspeagle/FlexMESA/pull/515) (draft), [#517](https://github.com/joshspeagle/FlexMESA/pull/517) (draft), [#518](https://github.com/joshspeagle/FlexMESA/pull/518) (draft), [#520](https://github.com/joshspeagle/FlexMESA/pull/520) (draft), [#523](https://github.com/joshspeagle/FlexMESA/pull/523) (draft), [#527](https://github.com/joshspeagle/FlexMESA/pull/527) (draft).
- **Post-parity deferred backlog ([#169](https://github.com/joshspeagle/FlexMESA/issues/169))**: `69` open issues (`63` `upstream-bug`, `6` `enhancement`; `4` marked `parity-pending` for upstream restoration before `v0.1.0`).

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
| `2026-11-18` | `916f075` | main merge | 13.9% | 0.0% | 0/106 | 71.1% | 89.6% | 9/32 | 17 |
| `2026-11-18` | `33dff54` | main merge | 13.9% | 0.0% | 0/106 | 71.1% | 87.6% | 10/32 | 18 |
| `2026-11-18` | `2c061b7` | main merge | 13.9% | 0.0% | 0/106 | 71.1% | 87.6% | 10/32 | 18 |
| `2026-11-18` | `995df81` | main merge | 13.9% | 0.0% | 0/106 | 71.1% | 87.6% | 10/33 | 18 |
| `2026-11-18` | `81b23b3` | main merge | 13.9% | 0.0% | 0/106 | 71.1% | 87.6% | 10/33 | 18 |
| `2026-11-18` | `3f7fa73` | main merge | 13.9% | 0.0% | 0/106 | 70.8% | 87.7% | 11/33 | 18 |
| `2026-11-18` | `3b76009` | main merge | 13.9% | 0.0% | 0/106 | 70.8% | 87.7% | 11/33 | 18 |
| `2026-11-18` | `9fb0df9` | main merge | 13.9% | 0.0% | 0/106 | 70.8% | 87.7% | 12/33 | 18 |
| `2026-11-19` | `93de419` | main merge | 13.9% | 0.0% | 0/106 | 73.6% | 87.9% | 12/33 | 18 |
| `2026-11-19` | `b1e4aa2` | main merge | 13.9% | 0.0% | 0/106 | 73.6% | 87.9% | 12/33 | 18 |
| `2026-11-19` | `1131259` | main merge | 13.9% | 0.0% | 0/106 | 74.0% | 87.9% | 12/34 | 18 |
| `2026-11-19` | `e9d1948` | main merge | 13.9% | 0.0% | 0/106 | 73.0% | 88.2% | 13/34 | 22 |
| `2026-11-19` | `986a6aa` | main merge | 13.9% | 0.0% | 0/106 | 73.1% | 88.2% | 13/34 | 28 |
| `2026-11-19` | `86313c2` | main merge | 13.9% | 0.0% | 0/106 | 73.4% | 88.2% | 13/34 | 32 |
| `2026-11-19` | `226cc16` | main merge | 18.8% | 0.0% | 0/106 | 73.3% | 88.1% | 13/34 | 35 |
| `2026-11-20` | `8b3e430` | main merge | 22.3% | 0.0% | 0/106 | 74.0% | 87.8% | 14/34 | 40 |
| `2026-11-20` | `e038ee0` | main merge | 26.1% | 1.0% | 0/106 | 74.8% | 87.8% | 15/34 | 40 |
| `2026-11-20` | `9ca6e68` | main merge | 26.1% | 1.0% | 0/106 | 74.8% | 87.8% | 15/34 | 40 |
| `2026-11-20` | `486c9d6` | main merge | 30.2% | 1.0% | 0/106 | 74.8% | 87.8% | 15/34 | 40 |
| `2026-11-20` | `e96914a` | main merge | 30.2% | 1.0% | 0/106 | 74.7% | 87.8% | 15/34 | 40 |
| `2026-11-20` | `aaa29e9` | main merge | 30.2% | 1.0% | 0/106 | 74.7% | 87.8% | 15/34 | 42 |
| `2026-11-20` | `e8042fa` | main merge | 30.2% | 1.0% | 0/106 | 74.8% | 87.8% | 15/34 | 42 |
| `2026-11-20` | `70963ef` | main merge | 30.2% | 1.0% | 0/106 | 74.9% | 87.8% | 15/34 | 42 |
| `2026-11-20` | `32028cd` | main merge | 30.2% | 1.0% | 0/106 | 75.0% | 87.8% | 15/34 | 42 |
| `2026-11-20` | `e0e987f` | main merge | 30.2% | 1.0% | 0/106 | 75.3% | 87.8% | 15/35 | 42 |
| `2026-11-20` | `ff8132b` | main merge | 30.2% | 1.0% | 0/106 | 75.3% | 87.8% | 16/36 | 42 |
| `2026-11-21` | `4a4b688` | main merge | 30.2% | 1.0% | 0/106 | 75.3% | 87.5% | 16/36 | 42 |
| `2026-11-21` | `e3ec226` | main merge | 30.2% | 1.0% | 0/106 | 75.3% | 87.5% | 16/36 | 42 |
| `2026-11-21` | `c921137` | main merge | 30.2% | 1.0% | 0/106 | 75.3% | 87.5% | 16/36 | 42 |
| `2026-11-21` | `0afe156` | main merge | 30.2% | 1.0% | 0/106 | 75.3% | 87.5% | 16/36 | 42 |
| `2026-11-21` | `ca49f4c` | main merge | 30.2% | 1.0% | 0/106 | 75.3% | 87.5% | 16/36 | 43 |
| `2026-11-21` | `7d8f040` | main merge | 30.2% | 1.0% | 0/106 | 75.3% | 87.6% | 16/36 | 43 |
| `2026-11-21` | `b1a7813` | main merge | 30.2% | 1.0% | 0/106 | 75.3% | 87.6% | 16/36 | 43 |
| `2026-11-21` | `da20d9c` | main merge | 30.2% | 1.0% | 0/106 | 75.3% | 87.6% | 16/36 | 43 |
| `2026-11-21` | `f171cca` | main merge | 30.2% | 1.0% | 0/106 | 75.3% | 87.6% | 16/36 | 44 |
| `2026-11-21` | `55452c9` | main merge | 30.2% | 1.0% | 0/106 | 75.3% | 87.6% | 16/36 | 44 |
| `2026-11-21` | `3195d37` | main merge | 30.2% | 1.0% | 0/106 | 75.4% | 87.6% | 16/36 | 44 |
| `2026-11-21` | `5f955f5` | main merge | 30.2% | 1.0% | 0/106 | 75.4% | 87.6% | 16/36 | 44 |
| `2026-11-21` | `d9d6453` | main merge | 30.2% | 1.0% | 0/106 | 75.4% | 87.6% | 16/36 | 44 |
| `2026-11-21` | `fbbec5a` | main merge | 30.2% | 1.0% | 0/106 | 75.5% | 87.6% | 16/36 | 44 |
| `2026-11-21` | `0363593` | main merge | 30.2% | 1.0% | 0/106 | 75.5% | 87.6% | 16/36 | 44 |
| `2026-11-21` | `ba2c12c` | main merge | 30.2% | 1.0% | 0/106 | 75.5% | 87.6% | 16/36 | 44 |
| `2026-11-21` | `7bae492` | main merge | 30.2% | 1.0% | 0/106 | 75.5% | 87.6% | 16/36 | 44 |
| `2026-11-21` | `d1fe407` | main merge | 30.2% | 1.0% | 0/106 | 75.5% | 87.6% | 16/36 | 45 |
| `2026-11-21` | `b5d4ba2` | main merge | 30.2% | 1.0% | 0/106 | 75.5% | 87.6% | 16/36 | 45 |
| `2026-11-21` | `a9009d6` | main merge | 30.2% | 1.0% | 0/106 | 75.5% | 87.6% | 16/36 | 45 |
| `2026-11-21` | `7defc98` | main merge | 30.2% | 1.0% | 0/106 | 75.5% | 87.6% | 16/36 | 45 |
| `2026-11-22` | `c3f7d00` | main merge | 30.2% | 1.0% | 0/106 | 75.5% | 87.6% | 16/36 | 45 |
| `2026-11-22` | `8c82649` | main merge | 30.2% | 1.0% | 0/106 | 75.5% | 87.6% | 16/36 | 45 |
| `2026-11-22` | `d348390` | main merge | 30.2% | 1.0% | 0/106 | 75.8% | 87.7% | 16/36 | 45 |
| `2026-11-22` | `9b2d1f7` | main merge | 30.2% | 1.0% | 0/106 | 75.8% | 87.7% | 16/36 | 45 |
| `2026-11-22` | `5b937c9` | main merge | 30.2% | 1.0% | 0/106 | 75.8% | 87.7% | 16/36 | 45 |
| `2026-11-22` | `a3ddf16` | main merge | 30.2% | 1.0% | 0/106 | 75.8% | 87.7% | 16/36 | 45 |
| `2026-11-22` | `d97b204` | main merge | 30.7% | 1.0% | 0/106 | 75.8% | 87.6% | 16/36 | 45 |
| `2026-11-22` | `f0b9899` | main merge | 30.7% | 1.0% | 0/106 | 75.8% | 87.6% | 16/36 | 45 |
| `2026-11-22` | `17ce9c2` | main merge | 30.7% | 1.0% | 0/106 | 75.8% | 87.6% | 16/36 | 45 |
| `2026-11-22` | `2dff765` | main merge | 30.7% | 1.1% | 0/106 | 75.8% | 87.6% | 17/36 | 45 |
| `2026-11-22` | `92994ac` | main merge | 30.7% | 1.1% | 0/106 | 75.8% | 87.6% | 17/36 | 45 |
| `2026-11-22` | `c6ab776` | main merge | 30.7% | 1.1% | 0/106 | 76.2% | 87.7% | 17/37 | 46 |
| `2026-11-22` | `eef96a8` | main merge | 30.7% | 1.1% | 3/106 | 76.2% | 87.7% | 18/37 | 46 |
| `2026-11-22` | `ced60f0` | main merge | 30.7% | 1.1% | 3/106 | 76.2% | 87.7% | 17/37 | 46 |
| `2026-11-23` | `1c1830c` | main merge | 30.7% | 1.1% | 3/106 | 76.2% | 87.7% | 17/37 | 46 |
| `2026-11-23` | `5f94fbe` | main merge | 30.7% | 1.1% | 3/106 | 76.2% | 87.7% | 17/37 | 46 |
| `2026-11-23` | `1fa1165` | main merge | 30.7% | 1.1% | 3/106 | 76.2% | 87.7% | 17/37 | 46 |
| `2026-11-23` | `0d22aa3` | main merge | 30.7% | 1.1% | 3/106 | 76.2% | 87.7% | 19/40 | 46 |
| `2026-11-23` | `54c803d` | main merge | 30.7% | 1.1% | 3/106 | 76.2% | 87.7% | 19/40 | 46 |
| `2026-11-23` | `b9e57eb` | main merge | 30.7% | 1.1% | 4/106 | 76.2% | 87.7% | 19/40 | 47 |
| `2026-11-23` | `550a9f9` | main merge | 30.7% | 1.1% | 4/106 | 76.2% | 87.7% | 19/41 | 51 |
| `2026-11-23` | `565a212` | main merge | 30.7% | 1.1% | 4/106 | 76.2% | 87.7% | 19/41 | 51 |
| `2026-11-23` | `72eea1e` | main merge | 30.7% | 1.1% | 4/106 | 76.2% | 87.7% | 19/41 | 51 |
| `2026-11-23` | `125b0ab` | main merge | 30.7% | 1.1% | 4/106 | 76.2% | 87.7% | 20/42 | 52 |
| `2026-11-23` | `f418498` | main merge | 30.7% | 1.1% | 4/106 | 76.2% | 87.7% | 20/42 | 52 |
| `2026-11-23` | `d0c7711` | main merge | 30.7% | 1.1% | 4/106 | 76.2% | 87.7% | 21/42 | 52 |
| `2026-11-23` | `54e1a65` | main merge | 30.7% | 1.1% | 4/106 | 76.2% | 87.7% | 22/43 | 52 |
| `2026-11-23` | `5799a16` | main merge | 30.7% | 1.1% | 4/106 | 76.2% | 87.7% | 23/45 | 52 |
| `2026-11-23` | `d9fcaf8` | main merge | 30.7% | 1.1% | 4/106 | 76.2% | 87.7% | 23/45 | 52 |
| `2026-11-23` | `21b7a74` | main merge | 30.7% | 1.1% | 4/106 | 76.2% | 87.7% | 26/46 | 54 |
| `2026-11-23` | `51c7bf7` | main merge | 30.7% | 1.1% | 4/106 | 76.2% | 87.7% | 26/48 | 54 |
| `2026-11-23` | `b7d27df` | main merge | 30.7% | 1.1% | 4/106 | 76.3% | 87.7% | 26/49 | 55 |
| `2026-11-23` | `851777e` | main merge | 30.7% | 1.1% | 5/106 | 76.3% | 87.7% | 26/49 | 55 |
| `2026-11-23` | `d72d62b` | main merge | 30.7% | 1.1% | 5/106 | 76.3% | 87.7% | 26/49 | 55 |
| `2026-11-24` | `b61193e` | main merge | 30.7% | 1.1% | 5/106 | 76.3% | 87.7% | 26/50 | 60 |
| `2026-11-24` | `f903a65` | main merge | 30.7% | 1.1% | 5/106 | 76.3% | 87.7% | 26/50 | 62 |
| `2026-11-24` | `e734399` | main merge | 30.7% | 1.1% | 5/106 | 76.3% | 87.7% | 26/50 | 62 |
| `2026-11-24` | `d88e910` | main merge | 30.7% | 1.1% | 5/106 | 76.3% | 87.7% | 26/50 | 62 |
| `2026-11-24` | `f89cdbb` | main merge | 30.7% | 1.1% | 5/106 | 76.3% | 87.7% | 26/50 | 62 |
| `2026-11-24` | `06ce88c` | main merge | 30.7% | 1.1% | 25/106 | 76.3% | 87.7% | 26/50 | 63 |
| `2026-11-24` | `50b1b83` | main merge | 30.7% | 1.1% | 25/106 | 76.3% | 87.7% | 26/50 | 66 |
| `2026-11-24` | `89fede1` | main merge | 30.7% | 1.1% | 25/106 | 76.3% | 87.7% | 26/50 | 66 |
| `2026-11-24` | `b343040` | main merge | 30.7% | 1.1% | 27/106 | 76.3% | 87.7% | 27/50 | 69 |
<!-- END GENERATED DASHBOARD -->
