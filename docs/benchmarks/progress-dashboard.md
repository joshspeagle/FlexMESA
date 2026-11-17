# Live Progress & Verification Dashboard

<!-- BEGIN GENERATED DASHBOARD -->
Generated mechanically by `flexmesa dashboard render` from [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml), [`benchmarks/history_harness/`](https://github.com/joshspeagle/FlexMESA/tree/main/benchmarks/history_harness), `cargo llvm-cov`, `pytest --cov`, and the GitHub API ([#174](https://github.com/joshspeagle/FlexMESA/issues/174)). Reference: **MESA release `26.4.1`** (git tag `26.4.1`, commit `c8c2026`, Zenodo `19722306`).

### Headline Status

| Metric | Value | Source |
| :--- | ---: | :--- |
| `star` + `star_data` + `binary` + `astero` ported (unverified) | 20,107 / 144,952 (13.9%; `star`+`binary`+`astero`: 14.4%) | [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml) |
| Upstream lines verified (all 26 modules) | 0 / 497,703 (0.0%) | [`port_manifest.toml`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/port_manifest.toml) |
| End-to-end MESA `26.4.1` cases passing (`90`/`8`/`8`) | 0 / 106 (102 declared gaps, 4 no reference yet, 0 undeclared failures) | [`case_table.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/case_table.md) |
| Strict mode bit-identical across machines (`#165`) | not yet measured | [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |
| Strict-mode kernels bit-identical to MESA `26.4.1` (`#165`) | not yet measured | [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |
| Test line coverage (Rust `llvm-cov` / Python `pytest --cov`) | 78.8% (67,773/85,967) / 87.1% (17,146/19,680) | [`coverage_summary.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/coverage_summary.json) |
| Road-to-`v0.1.0` packages ([#140](https://github.com/joshspeagle/FlexMESA/issues/140)) & open PRs | 4 closed, 28 open; 1 open PR(s) | [`github_snapshot.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/github_snapshot.json) |
| Post-parity deferred backlog ([#169](https://github.com/joshspeagle/FlexMESA/issues/169)) | 20 total (19 upstream bugs, 1 enhancement, 5 `parity-pending`) | [`github_snapshot.json`](https://github.com/joshspeagle/FlexMESA/blob/main/benchmarks/dashboard/github_snapshot.json) |

### 1. Port Coverage by Module (MESA `26.4.1`)

Progress is measured by **upstream MESA `26.4.1` source lines by status**, never by lines of Rust or Python written. No file starts as verified; raising a status requires a pull request that supplies the corresponding routine counterparts or generated MESA `26.4.1` evidence.

| Module | Upstream LOC | Absent | Stub | Partial | Ported (unverified) | Kernel ref. vectors | Case-level evidence | Rust LOC | Python/JAX LOC | % Ported | % Verified |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star` | 120,509 | 19,336 | 18,172 | 65,013 | 17,988 | 0 | 0 | 53,356 | 4,649 | 14.9% | 0.0% |
| `star_data` | 4,870 | 34 | 0 | 4,836 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `binary` | 11,638 | 2,947 | 0 | 6,572 | 2,119 | 0 | 0 | 8,426 | 1,234 | 18.2% | 0.0% |
| `astero` | 7,935 | 2,185 | 0 | 5,750 | 0 | 0 | 0 | 4,479 | 1,206 | 0.0% | 0.0% |
| `eos` | 28,666 | 7,656 | 0 | 21,010 | 0 | 0 | 0 | 13,180 | 9,725 | 0.0% | 0.0% |
| `kap` | 22,430 | 13,652 | 0 | 8,778 | 0 | 0 | 0 | 4,193 | 3,078 | 0.0% | 0.0% |
| `net` | 13,452 | 0 | 0 | 13,452 | 0 | 0 | 0 | 10,057 | 728 | 0.0% | 0.0% |
| `rates` | 17,217 | 192 | 0 | 17,025 | 0 | 0 | 0 | 3,102 | 1,474 | 0.0% | 0.0% |
| `neu` | 2,184 | 352 | 0 | 0 | 1,832 | 0 | 0 | 1,916 | 2,558 | 83.9% | 0.0% |
| `atm` | 6,853 | 2,468 | 0 | 4,385 | 0 | 0 | 0 | 2,091 | 791 | 0.0% | 0.0% |
| `turb` | 2,033 | 0 | 0 | 681 | 1,352 | 0 | 0 | 2,238 | 2,368 | 66.5% | 0.0% |
| `chem` | 4,195 | 373 | 0 | 3,822 | 0 | 0 | 0 | 2,364 | 898 | 0.0% | 0.0% |
| `ionization` | 3,859 | 2,226 | 0 | 1,633 | 0 | 0 | 0 | 2,837 | 982 | 0.0% | 0.0% |
| `colors` | 3,823 | 428 | 0 | 3,395 | 0 | 0 | 0 | 2,406 | 1,499 | 0.0% | 0.0% |
| `num` | 20,045 | 845 | 0 | 19,200 | 0 | 0 | 0 | 4,260 | 3,587 | 0.0% | 0.0% |
| `mtx` | 5,802 | 430 | 0 | 5,372 | 0 | 0 | 0 | 3,943 | 1,768 | 0.0% | 0.0% |
| `interp_1d` | 4,969 | 1,624 | 0 | 3,345 | 0 | 0 | 0 | 1,194 | 1,297 | 0.0% | 0.0% |
| `interp_2d` | 13,565 | 6,661 | 0 | 6,784 | 120 | 0 | 0 | 2,682 | 2,274 | 0.9% | 0.0% |
| `auto_diff` | 14,353 | 1,715 | 0 | 135 | 12,503 | 0 | 0 | 2,746 | 1,234 | 87.1% | 0.0% |
| `math` | 936 | 368 | 0 | 568 | 0 | 0 | 0 | 468 | 567 | 0.0% | 0.0% |
| `const` | 256 | 0 | 0 | 0 | 256 | 0 | 0 | 359 | 329 | 100.0% | 0.0% |
| `utils` | 3,633 | 1,555 | 0 | 2,078 | 0 | 0 | 0 | 2,931 | 2,353 | 0.0% | 0.0% |
| `gyre` | 55,708 | 55,708 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `adipls` | 30,923 | 30,923 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `stella` | 62,117 | 62,117 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0% | 0.0% |
| `plotting` | 35,732 | 35,732 | 0 | 0 | 0 | 0 | 0 | 0 | 759 | 0.0% | 0.0% |
| **Subtotal: star + star_data + binary + astero** | 144,952 | 24,502 | 18,172 | 82,171 | 20,107 | 0 | 0 | 66,261 | 7,089 | 13.9% | 0.0% |
| **Total (all 26 modules)** | 497,703 | 249,527 | 18,172 | 193,834 | 36,170 | 0 | 0 | 129,228 | 45,358 | 7.3% | 0.0% |

*Notes*: `star_data` state structs reside in `crates/flexmesa-star`. Bridge, testing, and benchmark infrastructure account for an additional `14,916` Rust lines (`flexmesa-py`, `flexmesa-testing`, `flexmesa-bench`) and `9,993` Python lines (`flexmesa.testing`, `flexmesa.bench`). For plotting ([#173](https://github.com/joshspeagle/FlexMESA/issues/173)), `flexmesa.plotting` currently provides `7` `plot_*` builders against `46` upstream `pgstar`/`pgbinary` panel modules.

### 2. Verification & Numerics Status

| Check | Status | Notes |
| :--- | :--- | :--- |
| Strict mode bit-identical across machines & thread counts (Level 1) | `not yet measured` | Awaiting [#165](https://github.com/joshspeagle/FlexMESA/issues/165) CI jobs (`x86_64` vs `aarch64`) |
| Strict-mode kernels bit-identical to MESA `26.4.1` hex outputs (Level 2) | `not yet measured` | `348,321` exact IEEE-754 values captured across 23 in-scope module test suites (`mesa-module-hex-v1`); strict comparison lands in [#165](https://github.com/joshspeagle/FlexMESA/issues/165) |

| Module | Upstream `26.4.1` Exact Values (`test_output` / hex) | Strict Bit-Identical Compared (`#165`) |
| :--- | ---: | :--- |
| `const` | 23 | `not yet measured` |
| `utils` | 0 | `not yet measured` |
| `math` | 102 | `not yet measured` |
| `mtx` | 22 | `not yet measured` |
| `auto_diff` | 774 | `not yet measured` |
| `num` | 570 | `not yet measured` |
| `interp_1d` | 234 | `not yet measured` |
| `interp_2d` | 104 | `not yet measured` |
| `chem` | 1,102 | `not yet measured` |
| `colors` | 234 | `not yet measured` |
| `eos` | 391 | `not yet measured` |
| `kap` | 54 | `not yet measured` |
| `rates` | 3,685 | `not yet measured` |
| `neu` | 340,000 | `not yet measured` |
| `net` | 140 | `not yet measured` |
| `ionization` | 41 | `not yet measured` |
| `atm` | 90 | `not yet measured` |
| `turb` | 136 | `not yet measured` |
| `star_data` | 0 | `not yet measured` |
| `gyre` | 136 | `not yet measured` |
| `adipls` | 92 | `not yet measured` |
| `star` | 386 | `not yet measured` |
| `binary` | 5 | `not yet measured` |
| `astero` | 0 | `not yet measured` |

### 3. End-to-End Cases (`benchmarks/history_harness`)

| Suite | `26.4.1` Cases | Stored `26.4.1` Refs | No Ref Yet | Reach Upstream Term | Within Numeric Tol | Verified Pass | Declared Gaps | Undeclared Failures |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `star` | 90 | 86 | 4 | 38 | 5 | 0 | 86 | 0 |
| `binary` | 8 | 8 | 0 | 5 | 0 | 0 | 8 | 0 |
| `astero` | 8 | 8 | 0 | 6 | 0 | 0 | 8 | 0 |
| **Total** | **106** | **102** | **4** | **49** | **5** | **0** | **102** | **0** |

- **Full per-case table**: [`notes/audit/case_table.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/case_table.md). The `5` cases currently within numeric tolerance (`rsp_BEP`, `rsp_BLAP`, `rsp_Cepheid`, `rsp_RR_Lyrae`, `rsp_save_and_load_file`) still invoke library-side `Tier4CaseExtras` and are governed as declared gaps under [#148](https://github.com/joshspeagle/FlexMESA/issues/148).
- **Declared gaps by owning package**: [#152](https://github.com/joshspeagle/FlexMESA/issues/152) (18), [#148](https://github.com/joshspeagle/FlexMESA/issues/148) (17), [#154](https://github.com/joshspeagle/FlexMESA/issues/154) (11), [#161](https://github.com/joshspeagle/FlexMESA/issues/161) (9), [#162](https://github.com/joshspeagle/FlexMESA/issues/162) (9), [#157](https://github.com/joshspeagle/FlexMESA/issues/157) (7), [#160](https://github.com/joshspeagle/FlexMESA/issues/160) (7), [#170](https://github.com/joshspeagle/FlexMESA/issues/170) (7), [#155](https://github.com/joshspeagle/FlexMESA/issues/155) (6), [#158](https://github.com/joshspeagle/FlexMESA/issues/158) (4), [#156](https://github.com/joshspeagle/FlexMESA/issues/156) (2), [#159](https://github.com/joshspeagle/FlexMESA/issues/159) (2), [#150](https://github.com/joshspeagle/FlexMESA/issues/150) (1), [#151](https://github.com/joshspeagle/FlexMESA/issues/151) (1), [#171](https://github.com/joshspeagle/FlexMESA/issues/171) (1).
- **Trajectory comparison figures ([#173](https://github.com/joshspeagle/FlexMESA/issues/173))**: `0` committed.

### 4. Controls & Defaults Coverage (`#147`, `#149`)

| Namelist | Upstream `26.4.1` Scalars | Matched in `FlexMESA` | Defaults Agree | Value Mismatch | Active Limit `None` | Unimplemented |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `star &controls` | 1,396 | 582 | 534 | 32 | 16 | 814 |
| `star &star_job` | 431 | 80 | 63 | 11 | 6 | 351 |
| `binary &binary_controls + &binary_job` | 209 | 120 | 117 | 3 | 0 | 89 |
| `astero &astero_search_controls` | 150 | 73 | 70 | 3 | 0 | 77 |
| `eos &eos` | 102 | 54 | 42 | 12 | 0 | 48 |
| `kap &kap` | 36 | 16 | 14 | 2 | 0 | 20 |
| `star &pgstar` | 2,900 | 0 | 0 | 0 | 0 | 2,900 |
| `binary &pgbinary` | 1,329 | 0 | 0 | 0 | 0 | 1,329 |

Full scalar diff: [`notes/audit/defaults_diff_26_4_1.md`](https://github.com/joshspeagle/FlexMESA/blob/main/notes/audit/defaults_diff_26_4_1.md) (loud errors on unknown controls tracked in [#147](https://github.com/joshspeagle/FlexMESA/issues/147); defaults alignment tracked in [#149](https://github.com/joshspeagle/FlexMESA/issues/149)).

### 5. Test & Code Health

- **Rust workspace line coverage (`cargo llvm-cov`)**: **78.8%** (`67,773` / `85,967` lines across `25` crates).
- **Python package statement coverage (`pytest --cov`)**: **87.1%** (`17,146` / `19,680` statements across `27` subpackages).
- **Ignored / skipped tests**: `19` Rust `#[ignore]` (#143, #146, #160, #161, #162, #165) and `1` Python `xfail`/`skip` (`0` unlinked).
- **Static & documentation gates**: `cargo clippy --all-targets -- -D warnings`, `RUSTDOCFLAGS='-D warnings' cargo doc`, `ruff check`, `ruff format --check`, `mypy --strict`, `mkdocs build --strict`, and MESA `26.4.1` `file:line` citation resolution enforced in CI.

#### Rust Line Coverage by Crate (`cargo llvm-cov`)

| Crate | Executable Lines | Covered Lines | Line Coverage |
| :--- | ---: | ---: | ---: |
| `flexmesa-astero` | 3,130 | 1,424 | 45.5% |
| `flexmesa-atm` | 1,326 | 958 | 72.2% |
| `flexmesa-autodiff` | 1,283 | 476 | 37.1% |
| `flexmesa-bench` | 2,342 | 2,328 | 99.4% |
| `flexmesa-binary` | 5,570 | 4,164 | 74.8% |
| `flexmesa-chem` | 937 | 661 | 70.5% |
| `flexmesa-colors` | 1,263 | 852 | 67.5% |
| `flexmesa-const` | 19 | 0 | 0.0% |
| `flexmesa-data` | 1,091 | 762 | 69.8% |
| `flexmesa-eos` | 8,851 | 6,840 | 77.3% |
| `flexmesa-interp1d` | 683 | 579 | 84.8% |
| `flexmesa-interp2d` | 1,705 | 1,315 | 77.1% |
| `flexmesa-ionization` | 996 | 633 | 63.6% |
| `flexmesa-kap` | 2,868 | 2,027 | 70.7% |
| `flexmesa-math` | 159 | 99 | 62.3% |
| `flexmesa-mtx` | 2,651 | 2,337 | 88.2% |
| `flexmesa-net` | 7,412 | 6,630 | 89.4% |
| `flexmesa-neu` | 994 | 962 | 96.8% |
| `flexmesa-num` | 2,856 | 2,425 | 84.9% |
| `flexmesa-rates` | 2,051 | 1,809 | 88.2% |
| `flexmesa-schema` | 467 | 367 | 78.6% |
| `flexmesa-star` | 35,778 | 28,890 | 80.7% |
| `flexmesa-testing` | 253 | 197 | 77.9% |
| `flexmesa-turb` | 1,093 | 914 | 83.6% |
| `flexmesa-utils` | 189 | 124 | 65.6% |

#### Python Statement Coverage by Subpackage (`pytest --cov`)

| Subpackage | Statements | Covered | Coverage |
| :--- | ---: | ---: | ---: |
| `flexmesa` | 68 | 64 | 94.1% |
| `flexmesa.astero` | 271 | 268 | 98.9% |
| `flexmesa.atm` | 180 | 173 | 96.1% |
| `flexmesa.auto_diff` | 563 | 482 | 85.6% |
| `flexmesa.bench` | 1,370 | 1,365 | 99.6% |
| `flexmesa.binary` | 321 | 308 | 96.0% |
| `flexmesa.chem` | 240 | 231 | 96.2% |
| `flexmesa.colors` | 445 | 367 | 82.5% |
| `flexmesa.constants` | 107 | 107 | 100.0% |
| `flexmesa.data` | 541 | 422 | 78.0% |
| `flexmesa.eos` | 3,477 | 3,277 | 94.2% |
| `flexmesa.interp_1d` | 432 | 380 | 88.0% |
| `flexmesa.interp_2d` | 1,055 | 865 | 82.0% |
| `flexmesa.ionization` | 312 | 300 | 96.2% |
| `flexmesa.kap` | 1,308 | 1,269 | 97.0% |
| `flexmesa.math` | 173 | 173 | 100.0% |
| `flexmesa.mtx` | 610 | 512 | 83.9% |
| `flexmesa.net` | 148 | 148 | 100.0% |
| `flexmesa.neu` | 1,049 | 932 | 88.8% |
| `flexmesa.num` | 1,596 | 1,394 | 87.3% |
| `flexmesa.plotting` | 182 | 171 | 94.0% |
| `flexmesa.rates` | 496 | 484 | 97.6% |
| `flexmesa.schema` | 137 | 112 | 81.8% |
| `flexmesa.star` | 1,435 | 1,331 | 92.8% |
| `flexmesa.testing` | 2,166 | 1,128 | 52.1% |
| `flexmesa.turb` | 912 | 803 | 88.0% |
| `flexmesa.utils` | 86 | 80 | 93.0% |

### 6. Work Tracking (`#140` & `#169`)

| `#140` Phase | Closed | Open | Open Issues |
| :--- | ---: | ---: | :--- |
| Phase 0 | 4 | 2 | [#165](https://github.com/joshspeagle/FlexMESA/issues/165), [#174](https://github.com/joshspeagle/FlexMESA/issues/174) |
| Phase A | 0 | 4 | [#142](https://github.com/joshspeagle/FlexMESA/issues/142), [#144](https://github.com/joshspeagle/FlexMESA/issues/144), [#146](https://github.com/joshspeagle/FlexMESA/issues/146), [#168](https://github.com/joshspeagle/FlexMESA/issues/168) |
| Phase B | 0 | 5 | [#147](https://github.com/joshspeagle/FlexMESA/issues/147), [#148](https://github.com/joshspeagle/FlexMESA/issues/148), [#149](https://github.com/joshspeagle/FlexMESA/issues/149), [#150](https://github.com/joshspeagle/FlexMESA/issues/150), [#151](https://github.com/joshspeagle/FlexMESA/issues/151) |
| Phase C | 0 | 13 | [#152](https://github.com/joshspeagle/FlexMESA/issues/152), [#153](https://github.com/joshspeagle/FlexMESA/issues/153), [#154](https://github.com/joshspeagle/FlexMESA/issues/154), [#155](https://github.com/joshspeagle/FlexMESA/issues/155), [#156](https://github.com/joshspeagle/FlexMESA/issues/156), [#157](https://github.com/joshspeagle/FlexMESA/issues/157), [#158](https://github.com/joshspeagle/FlexMESA/issues/158), [#159](https://github.com/joshspeagle/FlexMESA/issues/159), [#160](https://github.com/joshspeagle/FlexMESA/issues/160), [#161](https://github.com/joshspeagle/FlexMESA/issues/161), [#162](https://github.com/joshspeagle/FlexMESA/issues/162), [#164](https://github.com/joshspeagle/FlexMESA/issues/164), [#166](https://github.com/joshspeagle/FlexMESA/issues/166) |
| Phase D | 0 | 3 | [#170](https://github.com/joshspeagle/FlexMESA/issues/170), [#171](https://github.com/joshspeagle/FlexMESA/issues/171), [#172](https://github.com/joshspeagle/FlexMESA/issues/172) |
| Phase E | 0 | 2 | [#173](https://github.com/joshspeagle/FlexMESA/issues/173), [#174](https://github.com/joshspeagle/FlexMESA/issues/174) |
| Deferred scope | 0 | 0 | none |

- **Open pull requests (1)**: [#195](https://github.com/joshspeagle/FlexMESA/pull/195) (draft).
- **Post-parity deferred backlog ([#169](https://github.com/joshspeagle/FlexMESA/issues/169))**: `20` issues (`19` `upstream-bug`, `1` `enhancement`; `5` marked `parity-pending` for upstream restoration before `v0.1.0`).

### 7. Headline History Across Merges

| Date (UTC) | Commit | Event | Engine % Ported | All % Verified | Passing Cases | Rust Cov | Python Cov | `#140` Closed/Total | `#169` Backlog |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2026-11-16` | `f707f99` | Audit baseline (#141, #174) | 13.9% | 0.0% | 0/106 | 78.8% | 87.1% | 4/32 | 20 |
<!-- END GENERATED DASHBOARD -->
