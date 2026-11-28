# Live Progress & Verification Dashboard

<!-- BEGIN GENERATED DASHBOARD -->
Generated mechanically by `flexmesa dashboard render` from [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml), [`benchmarks/history_harness/`](https://github.com/joshspeagle/FlexMESA/tree/main/benchmarks/history_harness), `cargo llvm-cov`, `pytest --cov`, and the GitHub API ([#174](https://github.com/joshspeagle/FlexMESA/issues/174)). Reference: **MESA release `26.4.1`** (commit `c8c2026`, Zenodo `19722306`).

### Headline Status

| Metric | Value | Source |
| :--- | ---: | :--- |
| `star` + `star_data` + `binary` + `astero` ported (unverified) | 45,127 / 144,952 (31.1%; `star`+`binary`+`astero`: 32.2%) | [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml) |
| Upstream lines verified (all 26 modules) | 5,640 / 497,703 (1.1%) | [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml) |
| End-to-end MESA `26.4.1` cases passing (`90`/`8`/`8`) | 63 / 106 (36 declared gaps, 3 no reference yet, 0 undeclared failures) | [`case_table.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/case_table.md) |
| Strict mode bit-identical across machines (`#165`) | gated in CI by scripts/check_strict_determinism.py (x86_64 vs aarch64, 1 vs 4 threads; 474,062 math vectors + uncapped star/1.3M_ms_high_Z + binary/jdot_gr_check run with unsupported controls ignored; a bit-identity check, not a comparison with MESA) | [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |
| Strict-mode kernels bit-identical to MESA `26.4.1` (`#165`) | math: 403,258/403,258 CRMATH 0 ULP (70,804 libm <= 1 ULP); 305,966/305,967 probed module hex 0 ULP (12 complete, 16 non-math probed) | [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |
| Test line coverage (Rust `llvm-cov` / Python `pytest --cov`) | 77.0% (110,497/143,539) / 87.8% (23,723/27,031) | [`coverage_summary.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/coverage_summary.json) |
| Road-to-`v0.1.0` packages ([#140](https://github.com/joshspeagle/FlexMESA/issues/140)) & open PRs | 32 closed, 24 open; 3 open PR(s) | [`github_snapshot.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/github_snapshot.json) |
| Post-parity deferred backlog ([#169](https://github.com/joshspeagle/FlexMESA/issues/169)) | 103 open (97 upstream bugs, 6 enhancements, 15 `parity-pending`) | [`github_snapshot.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/github_snapshot.json) |

### 1. Port Coverage by Module (MESA `26.4.1`)

Progress is measured by **upstream MESA `26.4.1` source lines by status**, never by lines of Rust or Python written. No file starts as verified; raising a status requires a pull request that supplies the corresponding routine counterparts or generated MESA `26.4.1` evidence. A raise from absent to partial may instead rest on a sampled independent review of the file's routine map, recorded in `notes/audit/routine_map_reviews/` (`DECISIONS #79`).

| Module | Upstream LOC | Absent | Stub | Partial | Ported (unverified) | Kernel ref. vectors | Case-level evidence | Rust LOC | Python/JAX LOC | % Ported | % Verified |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star` | 120,509 | 17,101 | 6,378 | 54,084 | 37,306 | 5,640 | 0 | 116,115 | 3,985 | 35.6% | 4.7% |
| `star_data` | 4,870 | 34 | 0 | 4,836 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `binary` | 11,638 | 2,885 | 0 | 6,572 | 2,181 | 0 | 0 | 14,868 | 1,244 | 18.7% | 0.0% |
| `astero` | 7,935 | 2,185 | 0 | 5,750 | 0 | 0 | 0 | 12,415 | 1,295 | 0.0% | 0.0% |
| `eos` | 28,666 | 7,656 | 0 | 21,010 | 0 | 0 | 0 | 16,040 | 9,911 | 0.0% | 0.0% |
| `kap` | 22,430 | 13,652 | 0 | 8,778 | 0 | 0 | 0 | 7,328 | 3,359 | 0.0% | 0.0% |
| `net` | 13,452 | 0 | 0 | 13,452 | 0 | 0 | 0 | 12,924 | 728 | 0.0% | 0.0% |
| `rates` | 17,217 | 192 | 0 | 17,025 | 0 | 0 | 0 | 4,851 | 1,478 | 0.0% | 0.0% |
| `neu` | 2,184 | 352 | 0 | 0 | 1,832 | 0 | 0 | 1,972 | 2,673 | 83.9% | 0.0% |
| `atm` | 6,853 | 2,468 | 0 | 4,385 | 0 | 0 | 0 | 2,876 | 984 | 0.0% | 0.0% |
| `turb` | 2,033 | 0 | 0 | 681 | 1,352 | 0 | 0 | 3,236 | 3,116 | 66.5% | 0.0% |
| `chem` | 4,195 | 373 | 0 | 3,822 | 0 | 0 | 0 | 2,627 | 930 | 0.0% | 0.0% |
| `ionization` | 3,859 | 2,226 | 0 | 1,633 | 0 | 0 | 0 | 4,600 | 1,033 | 0.0% | 0.0% |
| `colors` | 3,823 | 428 | 0 | 3,395 | 0 | 0 | 0 | 4,318 | 1,521 | 0.0% | 0.0% |
| `num` | 20,045 | 845 | 0 | 19,200 | 0 | 0 | 0 | 5,552 | 3,744 | 0.0% | 0.0% |
| `mtx` | 5,802 | 430 | 0 | 5,372 | 0 | 0 | 0 | 9,549 | 1,851 | 0.0% | 0.0% |
| `interp_1d` | 4,969 | 1,624 | 0 | 3,345 | 0 | 0 | 0 | 1,181 | 1,311 | 0.0% | 0.0% |
| `interp_2d` | 13,565 | 6,661 | 0 | 6,784 | 120 | 0 | 0 | 2,697 | 2,288 | 0.9% | 0.0% |
| `auto_diff` | 14,353 | 1,715 | 0 | 12,638 | 0 | 0 | 0 | 5,834 | 1,234 | 0.0% | 0.0% |
| `math` | 936 | 368 | 0 | 568 | 0 | 0 | 0 | 2,745 | 577 | 0.0% | 0.0% |
| `const` | 256 | 0 | 0 | 0 | 256 | 0 | 0 | 368 | 329 | 100.0% | 0.0% |
| `utils` | 3,633 | 1,555 | 0 | 2,078 | 0 | 0 | 0 | 5,343 | 2,789 | 0.0% | 0.0% |
| `gyre` | 55,708 | 29,427 | 0 | 26,281 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `adipls` | 30,923 | 20,829 | 0 | 10,094 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `stella` | 62,117 | 62,117 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `plotting` | 35,732 | 35,732 | 0 | 0 | 0 | 0 | 0 | 0 | 12,095 | 0.0% | 0.0% |
| **Subtotal: star + star_data + binary + astero** | 144,952 | 22,205 | 6,378 | 71,242 | 39,487 | 5,640 | 0 | 143,398 | 6,524 | 31.1% | 3.9% |
| **Total (all 26 modules)** | 497,703 | 210,855 | 6,378 | 231,783 | 43,047 | 5,640 | 0 | 237,439 | 58,475 | 9.8% | 1.1% |

*Notes*: `star_data` state structs reside in `crates/flexmesa-star`. Bridge, testing, and benchmark infrastructure account for an additional `23,307` Rust lines (`flexmesa-py`, `flexmesa-testing`, `flexmesa-bench`) and `26,504` Python lines (`flexmesa.testing`, `flexmesa.bench`). For plotting ([#173](https://github.com/joshspeagle/FlexMESA/issues/173)), `flexmesa.plotting` currently provides `34` `plot_*` builders against `46` upstream `pgstar`/`pgbinary` panel modules.

### 2. Verification & Numerics Status

| Check | Status | Notes |
| :--- | :--- | :--- |
| Strict mode bit-identical across machines & thread counts (Level 1) | `gated in CI by scripts/check_strict_determinism.py (x86_64 vs aarch64, 1 vs 4 threads; 474,062 math vectors + uncapped star/1.3M_ms_high_Z + binary/jdot_gr_check run with unsupported controls ignored; a bit-identity check, not a comparison with MESA)` | Verified by [#165](https://github.com/joshspeagle/FlexMESA/issues/165) CI jobs (`x86_64` vs `aarch64`) |
| Strict-mode kernels bit-identical to MESA `26.4.1` hex outputs (Level 2) | `math: 403,258/403,258 CRMATH 0 ULP (70,804 libm <= 1 ULP); 305,966/305,967 probed module hex 0 ULP (12 complete, 16 non-math probed)` | `348,321` exact IEEE-754 values captured across 24 in-scope module test suites (`mesa-module-hex-v1`); measured in [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |

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
| `chem` | 1,102 | `1,102/1,102 (0 ULP)` |
| `colors` | 234 | `9/10 0 ULP (max 1 ULP, 234 total)` |
| `eos` | 391 | `391/391 (0 ULP)` |
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
| `star` | 90 | 87 | 3 | 78 | 12 | 75 | 59 | 56 | 30 | 0 |
| `binary` | 8 | 8 | 0 | 8 | 0 | 8 | 5 | 4 | 4 | 0 |
| `astero` | 8 | 8 | 0 | 8 | 0 | 7 | 7 | 3 | 2 | 0 |
| **Total** | **106** | **103** | **3** | **94** | **12** | **90** | **71** | **63** | **36** | **0** |

- **Full per-case table**: [`notes/audit/case_table.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/case_table.md). `4` case(s) within numeric tolerance (`make_planets`, `ns_h`, `wind_fed_bhhmxb`, `fast_newuoa`) remain governed as declared gaps under [#162](https://github.com/joshspeagle/FlexMESA/issues/162), [#587](https://github.com/joshspeagle/FlexMESA/issues/587).
- **Declared gaps by owning package**: [#161](https://github.com/joshspeagle/FlexMESA/issues/161) (13), [#162](https://github.com/joshspeagle/FlexMESA/issues/162) (8), [#154](https://github.com/joshspeagle/FlexMESA/issues/154) (5), [#157](https://github.com/joshspeagle/FlexMESA/issues/157) (5), [#158](https://github.com/joshspeagle/FlexMESA/issues/158) (3), [#587](https://github.com/joshspeagle/FlexMESA/issues/587) (3), [#160](https://github.com/joshspeagle/FlexMESA/issues/160) (2), [#231](https://github.com/joshspeagle/FlexMESA/issues/231) (1).
- **Trajectory comparison figures ([#173](https://github.com/joshspeagle/FlexMESA/issues/173))**: `0` committed.

### 4. Controls & Defaults Coverage (`#147`, `#149`)

| Namelist | Upstream `26.4.1` Scalars | Matched in `FlexMESA` | Defaults Agree | Value Mismatch | Active Limit `None` | Non-Scalar Rust | Unimplemented |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star &controls` | 1,444 | 1,190 | 1,190 | 0 | 0 | 17 | 237 |
| `star &star_job` | 455 | 210 | 210 | 0 | 0 | 1 | 244 |
| `binary &binary_controls + &binary_job` | 209 | 152 | 152 | 0 | 0 | 1 | 56 |
| `astero &astero_search_controls` | 150 | 137 | 137 | 0 | 0 | 1 | 12 |
| `eos &eos` | 102 | 60 | 60 | 0 | 0 | 0 | 42 |
| `kap &kap` | 36 | 21 | 21 | 0 | 0 | 0 | 15 |
| `star &pgstar` | 2,900 | 0 | 0 | 0 | 0 | 0 | 2,900 |
| `binary &pgbinary` | 1,329 | 0 | 0 | 0 | 0 | 0 | 1,329 |

Full scalar diff: [`notes/audit/defaults_diff_26_4_1.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/defaults_diff_26_4_1.md) (loud errors on unknown controls tracked in [#147](https://github.com/joshspeagle/FlexMESA/issues/147); defaults alignment tracked in [#149](https://github.com/joshspeagle/FlexMESA/issues/149)).

### 5. Test & Code Health

- **Rust workspace line coverage (`cargo llvm-cov`)**: **77.0%** (`110,497` / `143,539` lines across `28` crates).
- **Python package statement coverage (`pytest --cov`)**: **87.8%** (`23,723` / `27,031` statements across `27` subpackages).
- **Ignored / skipped tests**: `15` Rust `#[ignore]` ([#143](https://github.com/joshspeagle/FlexMESA/issues/143), [#154](https://github.com/joshspeagle/FlexMESA/issues/154), [#160](https://github.com/joshspeagle/FlexMESA/issues/160), [#162](https://github.com/joshspeagle/FlexMESA/issues/162), [#518](https://github.com/joshspeagle/FlexMESA/issues/518), [#530](https://github.com/joshspeagle/FlexMESA/issues/530)) and `22` Python `xfail`/`skip` ([#43](https://github.com/joshspeagle/FlexMESA/issues/43), [#154](https://github.com/joshspeagle/FlexMESA/issues/154), [#156](https://github.com/joshspeagle/FlexMESA/issues/156), [#161](https://github.com/joshspeagle/FlexMESA/issues/161), [#291](https://github.com/joshspeagle/FlexMESA/issues/291), [#295](https://github.com/joshspeagle/FlexMESA/issues/295), [#340](https://github.com/joshspeagle/FlexMESA/issues/340), [#349](https://github.com/joshspeagle/FlexMESA/issues/349), [#358](https://github.com/joshspeagle/FlexMESA/issues/358), [#402](https://github.com/joshspeagle/FlexMESA/issues/402), [#413](https://github.com/joshspeagle/FlexMESA/issues/413), [#420](https://github.com/joshspeagle/FlexMESA/issues/420), [#433](https://github.com/joshspeagle/FlexMESA/issues/433); `0` unlinked).
- **Static & documentation gates**: `cargo clippy --all-targets -- -D warnings`, `RUSTDOCFLAGS='-D warnings' cargo doc`, `ruff check`, `ruff format --check`, `mypy --strict`, `mkdocs build --strict`, and MESA `26.4.1` `file:line` citation resolution enforced in CI.
- **Test classification & listed routine oracles (`W4`, `#563`)**: `133` oracle files, `200` invariant files, `2` regression files (out of `335` test files totaling `2,022` test functions); `2` tracked mislabeled files; `64` / `1623` mapped MESA routines (`3.9%`) have a listed oracle test.

#### Test Classification & Routines with a Listed Oracle Test (W4)

| Module | Oracle Files | Invariant Files | Regression Files | Tracked Mislabeled | Routines with Listed Oracle Test |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `star` | `39` | `84` | `1` | `2` | `20/924` (`2.2%`) |
| `binary` | `5` | `15` | `0` | `0` | `0/116` (`0.0%`) |
| `astero` | `1` | `9` | `0` | `0` | `0/93` (`0.0%`) |
| `eos` | `15` | `6` | `0` | `0` | `30/108` (`27.8%`) |
| `kap` | `5` | `2` | `0` | `0` | `-` |
| `net` | `7` | `5` | `0` | `0` | `0/12` (`0.0%`) |
| `rates` | `5` | `1` | `0` | `0` | `-` |
| `neu` | `3` | `1` | `0` | `0` | `4/12` (`33.3%`) |
| `atm` | `4` | `1` | `0` | `0` | `-` |
| `turb` | `4` | `3` | `0` | `0` | `6/25` (`24.0%`) |
| `chem` | `3` | `1` | `0` | `0` | `-` |
| `ionization` | `2` | `2` | `0` | `0` | `-` |
| `colors` | `3` | `0` | `0` | `0` | `-` |
| `num` | `5` | `1` | `0` | `0` | `-` |
| `mtx` | `5` | `2` | `0` | `0` | `-` |
| `interp_1d` | `4` | `0` | `0` | `0` | `-` |
| `interp_2d` | `4` | `0` | `0` | `0` | `2/2` (`100.0%`) |
| `auto_diff` | `1` | `2` | `0` | `0` | `-` |
| `math` | `3` | `4` | `0` | `0` | `-` |
| `const` | `2` | `0` | `0` | `0` | `2/2` (`100.0%`) |
| `utils` | `0` | `7` | `0` | `0` | `-` |
| `gyre` | `8` | `3` | `0` | `0` | `0/283` (`0.0%`) |
| `adipls` | `3` | `4` | `0` | `0` | `0/46` (`0.0%`) |
| `plotting` | `0` | `3` | `0` | `0` | `-` |
| `tooling` | `2` | `44` | `1` | `0` | `-` |

#### Rust Line Coverage by Crate (`cargo llvm-cov`)

| Crate | Executable Lines | Covered Lines | Line Coverage |
| :--- | ---: | ---: | ---: |
| `flexmesa-adipls` | 6,282 | 4,993 | 79.5% |
| `flexmesa-astero` | 4,304 | 1,664 | 38.7% |
| `flexmesa-atm` | 1,792 | 1,573 | 87.8% |
| `flexmesa-autodiff` | 3,599 | 1,651 | 45.9% |
| `flexmesa-bench` | 3,529 | 2,367 | 67.1% |
| `flexmesa-binary` | 8,468 | 5,696 | 67.3% |
| `flexmesa-chem` | 911 | 671 | 73.7% |
| `flexmesa-colors` | 2,279 | 1,671 | 73.3% |
| `flexmesa-const` | 19 | 19 | 100.0% |
| `flexmesa-crlibm-sys` | 78 | 78 | 100.0% |
| `flexmesa-data` | 948 | 666 | 70.3% |
| `flexmesa-eos` | 10,678 | 8,815 | 82.6% |
| `flexmesa-gyre` | 5,939 | 5,128 | 86.3% |
| `flexmesa-interp1d` | 668 | 565 | 84.6% |
| `flexmesa-interp2d` | 1,691 | 1,303 | 77.1% |
| `flexmesa-ionization` | 2,072 | 1,788 | 86.3% |
| `flexmesa-kap` | 4,998 | 3,864 | 77.3% |
| `flexmesa-math` | 1,070 | 926 | 86.5% |
| `flexmesa-mtx` | 2,763 | 2,434 | 88.1% |
| `flexmesa-net` | 8,085 | 7,284 | 90.1% |
| `flexmesa-neu` | 1,020 | 989 | 97.0% |
| `flexmesa-num` | 2,991 | 2,603 | 87.0% |
| `flexmesa-rates` | 2,752 | 2,480 | 90.1% |
| `flexmesa-schema` | 790 | 660 | 83.5% |
| `flexmesa-star` | 62,817 | 48,239 | 76.8% |
| `flexmesa-testing` | 249 | 194 | 77.9% |
| `flexmesa-turb` | 1,400 | 1,225 | 87.5% |
| `flexmesa-utils` | 1,347 | 951 | 70.6% |

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
| `flexmesa.data` | 698 | 573 | 82.1% |
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
| `flexmesa.star` | 1,305 | 1,262 | 96.7% |
| `flexmesa.testing` | 4,934 | 4,009 | 81.3% |
| `flexmesa.turb` | 1,049 | 934 | 89.0% |
| `flexmesa.utils` | 86 | 80 | 93.0% |

### 6. Work Tracking (`#140` & `#169`)

| `#140` Phase | Closed | Open | Open Issues |
| :--- | ---: | ---: | :--- |
| Phase 0 | 6 | 0 | none |
| Phase A | 4 | 0 | none |
| Phase B | 5 | 0 | none |
| Phase C | 17 | 19 | [#152](https://github.com/joshspeagle/FlexMESA/issues/152), [#153](https://github.com/joshspeagle/FlexMESA/issues/153), [#154](https://github.com/joshspeagle/FlexMESA/issues/154), [#155](https://github.com/joshspeagle/FlexMESA/issues/155), [#156](https://github.com/joshspeagle/FlexMESA/issues/156), [#157](https://github.com/joshspeagle/FlexMESA/issues/157), [#158](https://github.com/joshspeagle/FlexMESA/issues/158), [#159](https://github.com/joshspeagle/FlexMESA/issues/159), [#160](https://github.com/joshspeagle/FlexMESA/issues/160), [#161](https://github.com/joshspeagle/FlexMESA/issues/161), [#162](https://github.com/joshspeagle/FlexMESA/issues/162), [#231](https://github.com/joshspeagle/FlexMESA/issues/231), [#295](https://github.com/joshspeagle/FlexMESA/issues/295), [#299](https://github.com/joshspeagle/FlexMESA/issues/299), [#365](https://github.com/joshspeagle/FlexMESA/issues/365), [#436](https://github.com/joshspeagle/FlexMESA/issues/436), [#539](https://github.com/joshspeagle/FlexMESA/issues/539), [#541](https://github.com/joshspeagle/FlexMESA/issues/541), [#587](https://github.com/joshspeagle/FlexMESA/issues/587) |
| Phase D | 0 | 3 | [#170](https://github.com/joshspeagle/FlexMESA/issues/170), [#171](https://github.com/joshspeagle/FlexMESA/issues/171), [#172](https://github.com/joshspeagle/FlexMESA/issues/172) |
| Phase E | 1 | 1 | [#173](https://github.com/joshspeagle/FlexMESA/issues/173) |
| Deferred scope | 0 | 1 | [#477](https://github.com/joshspeagle/FlexMESA/issues/477) |

- **Open pull requests (3)**: [#747](https://github.com/joshspeagle/FlexMESA/pull/747), [#753](https://github.com/joshspeagle/FlexMESA/pull/753), [#757](https://github.com/joshspeagle/FlexMESA/pull/757).
- **Post-parity deferred backlog ([#169](https://github.com/joshspeagle/FlexMESA/issues/169))**: `103` open issues (`97` `upstream-bug`, `6` `enhancement`; `15` marked `parity-pending` for upstream restoration before `v0.1.0`).

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
| `2026-11-24` | `1958352` | main merge | 30.7% | 1.1% | 27/106 | 76.3% | 87.7% | 27/50 | 69 |
| `2026-11-24` | `05d9893` | main merge | 30.7% | 1.1% | 32/106 | 76.3% | 87.7% | 28/50 | 69 |
| `2026-11-25` | `83dbf2d` | main merge | 30.7% | 1.1% | 35/106 | 77.0% | 87.8% | 29/50 | 71 |
| `2026-11-25` | `4fd18fc` | main merge | 30.7% | 1.1% | 39/106 | 77.0% | 87.8% | 31/54 | 71 |
| `2026-11-26` | `ef0d47c` | main merge | 30.7% | 1.1% | 43/106 | 77.0% | 87.8% | 32/55 | 71 |
| `2026-11-26` | `6705948` | main merge | 30.7% | 1.1% | 43/106 | 77.0% | 87.8% | 32/55 | 71 |
| `2026-11-26` | `2493662` | main merge | 30.7% | 1.1% | 43/106 | 77.0% | 87.8% | 32/55 | 71 |
| `2026-11-26` | `1ed0b7d` | main merge | 30.7% | 1.1% | 43/106 | 77.0% | 87.8% | 32/55 | 72 |
| `2026-11-27` | `b0a3dc2` | main merge | 30.7% | 1.1% | 44/106 | 77.0% | 87.8% | 32/56 | 75 |
| `2026-11-27` | `8cc7650` | main merge | 30.7% | 1.1% | 44/106 | 77.0% | 87.8% | 32/56 | 75 |
| `2026-11-27` | `72660d6` | main merge | 30.7% | 1.1% | 46/106 | 77.0% | 87.8% | 32/56 | 75 |
| `2026-11-27` | `2456762` | main merge | 30.7% | 1.1% | 46/106 | 77.0% | 87.8% | 32/56 | 75 |
| `2026-11-27` | `8857ff2` | main merge | 30.7% | 1.1% | 46/106 | 77.0% | 87.8% | 32/56 | 75 |
| `2026-11-27` | `1c4e546` | main merge | 30.7% | 1.1% | 46/106 | 77.0% | 87.8% | 32/56 | 75 |
| `2026-11-27` | `964c717` | main merge | 30.7% | 1.1% | 57/106 | 77.0% | 87.8% | 32/56 | 75 |
| `2026-11-27` | `6197a69` | main merge | 30.7% | 1.1% | 62/106 | 77.0% | 87.8% | 32/56 | 79 |
| `2026-11-28` | `dc390490` | main merge | 31.1% | 1.1% | 63/106 | 77.0% | 87.8% | 32/56 | 89 |
| `2026-11-28` | `b56fcb64` | main merge | 31.1% | 1.1% | 63/106 | 77.0% | 87.8% | 32/56 | 89 |
| `2026-11-28` | `da9ecf19` | main merge | 31.1% | 1.0% | 63/106 | 77.0% | 87.8% | 32/56 | 89 |
| `2026-11-28` | `784acc96` | main merge | 31.1% | 1.0% | 63/106 | 77.0% | 87.8% | 32/56 | 99 |
| `2026-11-28` | `6f35d6a1` | main merge | 31.1% | 1.1% | 63/106 | 77.0% | 87.8% | 32/56 | 99 |
| `2026-11-28` | `7901ff6f` | main merge | 31.1% | 1.1% | 63/106 | 77.0% | 87.8% | 32/56 | 100 |
| `2026-11-28` | `48385686` | main merge | 31.1% | 1.1% | 63/106 | 77.0% | 87.8% | 32/56 | 103 |
| `2026-11-28` | `74c240af` | main merge | 31.1% | 1.1% | 63/106 | 77.0% | 87.8% | 32/56 | 103 |
<!-- END GENERATED DASHBOARD -->
