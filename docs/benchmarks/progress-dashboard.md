# Live Progress & Verification Dashboard

<!-- BEGIN GENERATED DASHBOARD -->
Generated mechanically by `flexmesa dashboard render` from [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml), [`benchmarks/history_harness/`](https://github.com/joshspeagle/FlexMESA/tree/main/benchmarks/history_harness), `cargo llvm-cov`, `pytest --cov`, and the GitHub API ([#174](https://github.com/joshspeagle/FlexMESA/issues/174)). Reference: **MESA release `26.4.1`** (commit `c8c2026`, Zenodo `19722306`).

### Headline Status

| Metric | Value | Source |
| :--- | ---: | :--- |
| `star` + `star_data` + `binary` + `astero` ported (unverified) | 43,800 / 144,952 (30.2%; `star`+`binary`+`astero`: 31.3%) | [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml) |
| Upstream lines verified (all 26 modules) | 4,971 / 497,703 (1.0%) | [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml) |
| End-to-end MESA `26.4.1` cases passing (`90`/`8`/`8`) | 0 / 106 (102 declared gaps, 4 no reference yet, 0 undeclared failures) | [`case_table.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/case_table.md) |
| Strict mode bit-identical across machines (`#165`) | gated in CI by scripts/check_strict_determinism.py (x86_64 vs aarch64, 1 vs 4 threads; 474,062 math vectors + uncapped star/1.3M_ms_high_Z + binary/jdot_gr_check run with unsupported controls ignored; a bit-identity check, not a comparison with MESA) | [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |
| Strict-mode kernels bit-identical to MESA `26.4.1` (`#165`) | math: 403,258/403,258 CRMATH 0 ULP (70,804 libm <= 1 ULP); 304,567/304,635 probed module hex 0 ULP (5 complete, 14 non-math probed) | [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |
| Test line coverage (Rust `llvm-cov` / Python `pytest --cov`) | 75.0% (86,265/114,956) / 87.8% (22,448/25,569) | [`coverage_summary.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/coverage_summary.json) |
| Road-to-`v0.1.0` packages ([#140](https://github.com/joshspeagle/FlexMESA/issues/140)) & open PRs | 15 closed, 19 open; 0 open PR(s) | [`github_snapshot.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/github_snapshot.json) |
| Post-parity deferred backlog ([#169](https://github.com/joshspeagle/FlexMESA/issues/169)) | 42 open (41 upstream bugs, 1 enhancement, 3 `parity-pending`) | [`github_snapshot.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/github_snapshot.json) |

### 1. Port Coverage by Module (MESA `26.4.1`)

Progress is measured by **upstream MESA `26.4.1` source lines by status**, never by lines of Rust or Python written. No file starts as verified; raising a status requires a pull request that supplies the corresponding routine counterparts or generated MESA `26.4.1` evidence.

| Module | Upstream LOC | Absent | Stub | Partial | Ported (unverified) | Kernel ref. vectors | Case-level evidence | Rust LOC | Python/JAX LOC | % Ported | % Verified |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star` | 120,509 | 17,759 | 6,378 | 54,753 | 36,648 | 4,971 | 0 | 83,224 | 3,869 | 34.5% | 4.1% |
| `star_data` | 4,870 | 34 | 0 | 4,836 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `binary` | 11,638 | 2,885 | 0 | 6,572 | 2,181 | 0 | 0 | 11,965 | 1,244 | 18.7% | 0.0% |
| `astero` | 7,935 | 2,185 | 0 | 5,750 | 0 | 0 | 0 | 5,764 | 1,288 | 0.0% | 0.0% |
| `eos` | 28,666 | 7,656 | 0 | 21,010 | 0 | 0 | 0 | 14,334 | 9,867 | 0.0% | 0.0% |
| `kap` | 22,430 | 13,652 | 0 | 8,778 | 0 | 0 | 0 | 4,990 | 3,359 | 0.0% | 0.0% |
| `net` | 13,452 | 0 | 0 | 13,452 | 0 | 0 | 0 | 10,386 | 728 | 0.0% | 0.0% |
| `rates` | 17,217 | 192 | 0 | 17,025 | 0 | 0 | 0 | 3,469 | 1,478 | 0.0% | 0.0% |
| `neu` | 2,184 | 352 | 0 | 0 | 1,832 | 0 | 0 | 1,948 | 2,673 | 83.9% | 0.0% |
| `atm` | 6,853 | 2,468 | 0 | 4,385 | 0 | 0 | 0 | 2,476 | 984 | 0.0% | 0.0% |
| `turb` | 2,033 | 0 | 0 | 681 | 1,352 | 0 | 0 | 2,268 | 2,368 | 66.5% | 0.0% |
| `chem` | 4,195 | 373 | 0 | 3,822 | 0 | 0 | 0 | 2,338 | 930 | 0.0% | 0.0% |
| `ionization` | 3,859 | 2,226 | 0 | 1,633 | 0 | 0 | 0 | 4,152 | 1,033 | 0.0% | 0.0% |
| `colors` | 3,823 | 428 | 0 | 3,395 | 0 | 0 | 0 | 2,453 | 1,500 | 0.0% | 0.0% |
| `num` | 20,045 | 845 | 0 | 19,200 | 0 | 0 | 0 | 4,300 | 3,587 | 0.0% | 0.0% |
| `mtx` | 5,802 | 430 | 0 | 5,372 | 0 | 0 | 0 | 4,173 | 1,768 | 0.0% | 0.0% |
| `interp_1d` | 4,969 | 1,624 | 0 | 3,345 | 0 | 0 | 0 | 1,205 | 1,311 | 0.0% | 0.0% |
| `interp_2d` | 13,565 | 6,661 | 0 | 6,784 | 120 | 0 | 0 | 2,687 | 2,288 | 0.9% | 0.0% |
| `auto_diff` | 14,353 | 1,715 | 0 | 12,638 | 0 | 0 | 0 | 2,799 | 1,234 | 0.0% | 0.0% |
| `math` | 936 | 368 | 0 | 568 | 0 | 0 | 0 | 2,011 | 577 | 0.0% | 0.0% |
| `const` | 256 | 0 | 0 | 0 | 256 | 0 | 0 | 368 | 329 | 100.0% | 0.0% |
| `utils` | 3,633 | 1,555 | 0 | 2,078 | 0 | 0 | 0 | 4,846 | 2,659 | 0.0% | 0.0% |
| `gyre` | 55,708 | 55,708 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `adipls` | 30,923 | 30,923 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `stella` | 62,117 | 62,117 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `plotting` | 35,732 | 35,732 | 0 | 0 | 0 | 0 | 0 | 0 | 12,095 | 0.0% | 0.0% |
| **Subtotal: star + star_data + binary + astero** | 144,952 | 22,863 | 6,378 | 71,911 | 38,829 | 4,971 | 0 | 100,953 | 6,401 | 30.2% | 3.4% |
| **Total (all 26 modules)** | 497,703 | 247,888 | 6,378 | 196,077 | 42,389 | 4,971 | 0 | 172,156 | 57,169 | 9.5% | 1.0% |

*Notes*: `star_data` state structs reside in `crates/flexmesa-star`. Bridge, testing, and benchmark infrastructure account for an additional `19,294` Rust lines (`flexmesa-py`, `flexmesa-testing`, `flexmesa-bench`) and `14,207` Python lines (`flexmesa.testing`, `flexmesa.bench`). For plotting ([#173](https://github.com/joshspeagle/FlexMESA/issues/173)), `flexmesa.plotting` currently provides `34` `plot_*` builders against `46` upstream `pgstar`/`pgbinary` panel modules.

### 2. Verification & Numerics Status

| Check | Status | Notes |
| :--- | :--- | :--- |
| Strict mode bit-identical across machines & thread counts (Level 1) | `gated in CI by scripts/check_strict_determinism.py (x86_64 vs aarch64, 1 vs 4 threads; 474,062 math vectors + uncapped star/1.3M_ms_high_Z + binary/jdot_gr_check run with unsupported controls ignored; a bit-identity check, not a comparison with MESA)` | Verified by [#165](https://github.com/joshspeagle/FlexMESA/issues/165) CI jobs (`x86_64` vs `aarch64`) |
| Strict-mode kernels bit-identical to MESA `26.4.1` hex outputs (Level 2) | `math: 403,258/403,258 CRMATH 0 ULP (70,804 libm <= 1 ULP); 304,567/304,635 probed module hex 0 ULP (5 complete, 14 non-math probed)` | `348,321` exact IEEE-754 values captured across 24 in-scope module test suites (`mesa-module-hex-v1`); measured in [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |

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
| `colors` | 234 | `4/10 0 ULP (max 13 ULP, 234 total)` |
| `eos` | 391 | `0` |
| `kap` | 54 | `13/21 0 ULP (max 1,459,407,697 ULP, 21 total)` |
| `rates` | 3,685 | `3,636/3,639 0 ULP (max 29 ULP, 3,639 total)` |
| `neu` | 340,000 | `300,000/300,000 (0 ULP)` |
| `net` | 140 | `0` |
| `ionization` | 41 | `28/31 0 ULP (max 10,652,452 ULP, 31 total)` |
| `atm` | 90 | `15/39 0 ULP (max 7,863,114 ULP, 39 total)` |
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
| `star` | 90 | 86 | 4 | 0 | 90 | 37 | 5 | 0 | 86 | 0 |
| `binary` | 8 | 8 | 0 | 0 | 8 | 6 | 2 | 0 | 8 | 0 |
| `astero` | 8 | 8 | 0 | 0 | 8 | 3 | 1 | 0 | 8 | 0 |
| **Total** | **106** | **102** | **4** | **0** | **106** | **46** | **8** | **0** | **102** | **0** |

- **Full per-case table**: [`notes/audit/case_table.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/case_table.md). `8` case(s) within numeric tolerance (`rsp_BEP`, `rsp_BLAP`, `rsp_Cepheid`, `rsp_RR_Lyrae`, `rsp_save_and_load_file`, `jdot_gr_check`, `jdot_ml_check`, `surface_effects`) remain governed as declared gaps under [#160](https://github.com/joshspeagle/FlexMESA/issues/160), [#161](https://github.com/joshspeagle/FlexMESA/issues/161), [#162](https://github.com/joshspeagle/FlexMESA/issues/162).
- **Declared gaps by owning package**: [#161](https://github.com/joshspeagle/FlexMESA/issues/161) (27), [#159](https://github.com/joshspeagle/FlexMESA/issues/159) (20), [#154](https://github.com/joshspeagle/FlexMESA/issues/154) (13), [#157](https://github.com/joshspeagle/FlexMESA/issues/157) (9), [#162](https://github.com/joshspeagle/FlexMESA/issues/162) (9), [#160](https://github.com/joshspeagle/FlexMESA/issues/160) (7), [#170](https://github.com/joshspeagle/FlexMESA/issues/170) (7), [#158](https://github.com/joshspeagle/FlexMESA/issues/158) (6), [#231](https://github.com/joshspeagle/FlexMESA/issues/231) (2), [#151](https://github.com/joshspeagle/FlexMESA/issues/151) (1), [#171](https://github.com/joshspeagle/FlexMESA/issues/171) (1).
- **Trajectory comparison figures ([#173](https://github.com/joshspeagle/FlexMESA/issues/173))**: `0` committed.

### 4. Controls & Defaults Coverage (`#147`, `#149`)

| Namelist | Upstream `26.4.1` Scalars | Matched in `FlexMESA` | Defaults Agree | Value Mismatch | Active Limit `None` | Non-Scalar Rust | Unimplemented |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star &controls` | 1,444 | 1,094 | 1,094 | 0 | 0 | 17 | 333 |
| `star &star_job` | 455 | 93 | 93 | 0 | 0 | 1 | 361 |
| `binary &binary_controls + &binary_job` | 209 | 145 | 145 | 0 | 0 | 1 | 63 |
| `astero &astero_search_controls` | 150 | 75 | 75 | 0 | 0 | 1 | 74 |
| `eos &eos` | 102 | 60 | 60 | 0 | 0 | 0 | 42 |
| `kap &kap` | 36 | 21 | 21 | 0 | 0 | 0 | 15 |
| `star &pgstar` | 2,900 | 0 | 0 | 0 | 0 | 0 | 2,900 |
| `binary &pgbinary` | 1,329 | 0 | 0 | 0 | 0 | 0 | 1,329 |

Full scalar diff: [`notes/audit/defaults_diff_26_4_1.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/defaults_diff_26_4_1.md) (loud errors on unknown controls tracked in [#147](https://github.com/joshspeagle/FlexMESA/issues/147); defaults alignment tracked in [#149](https://github.com/joshspeagle/FlexMESA/issues/149)).

### 5. Test & Code Health

- **Rust workspace line coverage (`cargo llvm-cov`)**: **75.0%** (`86,265` / `114,956` lines across `26` crates).
- **Python package statement coverage (`pytest --cov`)**: **87.8%** (`22,448` / `25,569` statements across `27` subpackages).
- **Ignored / skipped tests**: `16` Rust `#[ignore]` ([#143](https://github.com/joshspeagle/FlexMESA/issues/143), [#160](https://github.com/joshspeagle/FlexMESA/issues/160), [#161](https://github.com/joshspeagle/FlexMESA/issues/161), [#162](https://github.com/joshspeagle/FlexMESA/issues/162)) and `2` Python `xfail`/`skip` ([#161](https://github.com/joshspeagle/FlexMESA/issues/161); `0` unlinked).
- **Static & documentation gates**: `cargo clippy --all-targets -- -D warnings`, `RUSTDOCFLAGS='-D warnings' cargo doc`, `ruff check`, `ruff format --check`, `mypy --strict`, `mkdocs build --strict`, and MESA `26.4.1` `file:line` citation resolution enforced in CI.

#### Rust Line Coverage by Crate (`cargo llvm-cov`)

| Crate | Executable Lines | Covered Lines | Line Coverage |
| :--- | ---: | ---: | ---: |
| `flexmesa-astero` | 4,047 | 1,457 | 36.0% |
| `flexmesa-atm` | 1,603 | 1,332 | 83.1% |
| `flexmesa-autodiff` | 1,322 | 458 | 34.6% |
| `flexmesa-bench` | 3,569 | 2,419 | 67.8% |
| `flexmesa-binary` | 7,933 | 5,217 | 65.8% |
| `flexmesa-chem` | 908 | 633 | 69.7% |
| `flexmesa-colors` | 1,297 | 888 | 68.5% |
| `flexmesa-const` | 19 | 0 | 0.0% |
| `flexmesa-crlibm-sys` | 78 | 36 | 46.2% |
| `flexmesa-data` | 986 | 675 | 68.5% |
| `flexmesa-eos` | 9,797 | 7,836 | 80.0% |
| `flexmesa-interp1d` | 683 | 579 | 84.8% |
| `flexmesa-interp2d` | 1,711 | 1,321 | 77.2% |
| `flexmesa-ionization` | 1,983 | 1,680 | 84.7% |
| `flexmesa-kap` | 3,494 | 2,809 | 80.4% |
| `flexmesa-math` | 781 | 627 | 80.3% |
| `flexmesa-mtx` | 2,783 | 2,460 | 88.4% |
| `flexmesa-net` | 7,690 | 6,722 | 87.4% |
| `flexmesa-neu` | 1,022 | 983 | 96.2% |
| `flexmesa-num` | 2,877 | 2,468 | 85.8% |
| `flexmesa-rates` | 2,292 | 2,098 | 91.5% |
| `flexmesa-schema` | 808 | 669 | 82.8% |
| `flexmesa-star` | 54,727 | 41,009 | 74.9% |
| `flexmesa-testing` | 253 | 197 | 77.9% |
| `flexmesa-turb` | 1,093 | 900 | 82.3% |
| `flexmesa-utils` | 1,200 | 792 | 66.0% |

#### Python Statement Coverage by Subpackage (`pytest --cov`)

| Subpackage | Statements | Covered | Coverage |
| :--- | ---: | ---: | ---: |
| `flexmesa` | 138 | 124 | 89.9% |
| `flexmesa.astero` | 271 | 268 | 98.9% |
| `flexmesa.atm` | 198 | 194 | 98.0% |
| `flexmesa.auto_diff` | 563 | 482 | 85.6% |
| `flexmesa.bench` | 1,380 | 1,375 | 99.6% |
| `flexmesa.binary` | 331 | 322 | 97.3% |
| `flexmesa.chem` | 246 | 234 | 95.1% |
| `flexmesa.colors` | 445 | 367 | 82.5% |
| `flexmesa.constants` | 107 | 107 | 100.0% |
| `flexmesa.data` | 695 | 570 | 82.0% |
| `flexmesa.eos` | 3,510 | 3,345 | 95.3% |
| `flexmesa.interp_1d` | 429 | 385 | 89.7% |
| `flexmesa.interp_2d` | 1,053 | 863 | 82.0% |
| `flexmesa.ionization` | 334 | 312 | 93.4% |
| `flexmesa.kap` | 1,469 | 1,409 | 95.9% |
| `flexmesa.math` | 178 | 178 | 100.0% |
| `flexmesa.mtx` | 610 | 512 | 83.9% |
| `flexmesa.net` | 148 | 148 | 100.0% |
| `flexmesa.neu` | 1,059 | 524 | 49.5% |
| `flexmesa.num` | 1,596 | 1,401 | 87.8% |
| `flexmesa.plotting` | 4,235 | 3,708 | 87.6% |
| `flexmesa.rates` | 496 | 484 | 97.6% |
| `flexmesa.schema` | 137 | 112 | 81.8% |
| `flexmesa.star` | 1,303 | 1,265 | 97.1% |
| `flexmesa.testing` | 3,640 | 2,878 | 79.1% |
| `flexmesa.turb` | 912 | 801 | 87.8% |
| `flexmesa.utils` | 86 | 80 | 93.0% |

### 6. Work Tracking (`#140` & `#169`)

| `#140` Phase | Closed | Open | Open Issues |
| :--- | ---: | ---: | :--- |
| Phase 0 | 6 | 0 | none |
| Phase A | 2 | 2 | [#144](https://github.com/joshspeagle/FlexMESA/issues/144), [#168](https://github.com/joshspeagle/FlexMESA/issues/168) |
| Phase B | 4 | 1 | [#151](https://github.com/joshspeagle/FlexMESA/issues/151) |
| Phase C | 3 | 12 | [#152](https://github.com/joshspeagle/FlexMESA/issues/152), [#153](https://github.com/joshspeagle/FlexMESA/issues/153), [#154](https://github.com/joshspeagle/FlexMESA/issues/154), [#155](https://github.com/joshspeagle/FlexMESA/issues/155), [#156](https://github.com/joshspeagle/FlexMESA/issues/156), [#157](https://github.com/joshspeagle/FlexMESA/issues/157), [#158](https://github.com/joshspeagle/FlexMESA/issues/158), [#159](https://github.com/joshspeagle/FlexMESA/issues/159), [#160](https://github.com/joshspeagle/FlexMESA/issues/160), [#161](https://github.com/joshspeagle/FlexMESA/issues/161), [#162](https://github.com/joshspeagle/FlexMESA/issues/162), [#231](https://github.com/joshspeagle/FlexMESA/issues/231) |
| Phase D | 0 | 3 | [#170](https://github.com/joshspeagle/FlexMESA/issues/170), [#171](https://github.com/joshspeagle/FlexMESA/issues/171), [#172](https://github.com/joshspeagle/FlexMESA/issues/172) |
| Phase E | 1 | 1 | [#173](https://github.com/joshspeagle/FlexMESA/issues/173) |
| Deferred scope | 0 | 0 | none |

- **Open pull requests (0)**: none.
- **Post-parity deferred backlog ([#169](https://github.com/joshspeagle/FlexMESA/issues/169))**: `42` open issues (`41` `upstream-bug`, `1` `enhancement`; `3` marked `parity-pending` for upstream restoration before `v0.1.0`).

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
<!-- END GENERATED DASHBOARD -->
