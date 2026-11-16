# T1: Python test suite provenance audit (FlexMESA PR #138 head 91eda0b)

Status: COMPLETE (read-only audit; nothing executed).


## 0. How tests are run (CI / config), verified by reading

- `pyproject.toml` `[tool.pytest.ini_options]`: `addopts = "--doctest-modules --strict-markers -m 'not slow'"`, `testpaths = ["src","tests"]`.
- `.github/workflows/ci.yml` env: `FLEXMESA_REQUIRE_ZERO_SKIPS=1`, `JAX_PLATFORMS=cpu`, `MESA_DIR=$HOME/.cache/flexmesa/mesa_test_inputs`, `FLEXMESA_DATA_DIR=$HOME/.cache/flexmesa/data` (both fetched from a GitHub release bundle by `src/flexmesa/data/fetch.py fetch --include-test-inputs`).
- 4 jobs: `rust` (cargo test, excludes flexmesa-py), `python-jax` (`pytest src tests --ignore=tests/test_docs_snippets.py --ignore=tests/test_mesa_paper_figures.py --ignore-glob="tests/test_e2e_*.py"`), `e2e` (`pytest tests/test_e2e_*.py -v`, 40 min timeout), `docs` (ruff, mypy, mkdocs, the two ignored files). Net effect: every test file runs in exactly one job; nothing is permanently ignored.
- `scripts/verify_ci.sh` mirrors the same four invocations.
- `FLEXMESA_REQUIRE_ZERO_SKIPS` (tests/conftest.py:52-83): `pytest_runtest_logreport` appends any `report.skipped` node id; `pytest_sessionfinish` sets exit status to TESTS_FAILED if any. It catches `pytest.skip` (and xfail). It does NOT catch (i) deselection via `-m 'not slow'`, (ii) tests that return early / loop over an empty list, (iii) parametrised cases that are never generated.

## 1. tests/fixtures provenance (byte comparison done with `cmp`)

| Fixture | Added in | Compared to upstream 0b2e6bb | Result |
|---|---|---|---|
| golden/eos/test_output.txt (474 lines) | 3a5ed30 | eos/test/test_output | byte-identical |
| golden/kap/test_output.txt (78) | e1acfc4 | kap/test/test_output | byte-identical |
| golden/rates/test_output.txt (3191) | d09cd11 | rates/test/test_output | byte-identical |
| golden/net/test_output.txt (214) | 3e23d05 | net/test/test_output | byte-identical |
| golden/atm/test_output.txt (101) | 3e23d05 | atm/test/test_output | byte-identical |
| golden/gyre/{test_output,model.dat} | c947885 | gyre/test/ | byte-identical |
| golden/adipls/{obs.test.txt,Gemma.fgong} | c947885 | adipls/test/ | byte-identical |
| golden/stella/mesa.{abn,hyd,lbol,vel_feII} | c947885 | stella/test/ | byte-identical |
| golden/stella/FeII_5169_eta.dat | c947885 | stella/res/ | byte-identical |
| golden/rsp/list | 47991c0 | star/test_suite/rsp_check_2nd_crossing/list | byte-identical |
| golden/rsp/rsp_*_cycle0.mod (7 files, header `version_number 'a108d91'`) | d4b7a1e | star/test_suite/rsp_*/rsp_*_cycle0.mod | upstream checkout holds 3-line Git-LFS stubs so no direct compare; the 4 I checked (cepheid, cepheid_6M, BEP, RR_Lyrae) are byte-identical to the copies in the release bundle `the test-input bundle`. Plausibly the real LFS objects; not independently verifiable here. |
| golden/rsp/rsp_*_init_r24.mod, rsp_*_step1_r24.mod (14 files, header `version_number '0b2e6bb'`) | d4b7a1e | not present anywhere in upstream tree | NOT upstream-shipped files. The version stamp says they were written by a MESA build at 0b2e6bb (i.e. somebody compiled and ran upstream and saved models), but there is no script, log, or recipe in the repo that regenerates them (grep for `_r24.mod` hits only the Rust tests that consume them). Provenance = unverifiable, used only by Rust tests (`crates/flexmesa-star/tests/rsp_*`), not by any Python test. |

Which Python tests read the fixtures: only `tests/unit/test_rates_wave3b3.py` (rates test_output.txt) and `tests/test_star_wave5_2.py` (gyre/adipls/stella). `tests/test_data.py` mentions "fixtures" only for synthetic tmp files. The eos/kap/net/atm fixture copies are consumed by Rust tests (other agent).


## 2. Findings (first batch: E2E, golden, bench)

### T1-1: All 107 "upstream test_suite" E2E passes are 2-step smoke runs judged by FlexMESA-authored sanity bounds; none compares to any upstream output
- Area: tests/test_e2e_{1,2,3,4,5}_tier*.py, tests/test_e2e_fidelity4.py, src/flexmesa/star/e2e.py
- Severity: unverified claim
- FlexMESA: tests/test_e2e_1_tier1.py:313-322 (`max_steps_per_stage=2 ... assert rep["passed"] is True; rep["assertions_passed"]; rep["photo_restart_passed"]`); same pattern tests/test_e2e_2_tier2.py:289-299, tests/test_e2e_3_tier3.py:208-218, tests/test_e2e_4_tier4.py:229-240 (`run_tier4_suite(root, max_steps_per_stage=2, ...)`, `run_dev_cases_suite(..., max_steps_per_stage=2)`), tests/test_e2e_5_tier5.py:110-119 (`max_steps_per_stage=2`, binary `max_steps_cap=4`), tests/test_e2e_fidelity4.py:64-68.
- Upstream: star/test_suite/<case>/inlist* stop conditions (hundreds to thousands of steps) and src/run_star_extras.f90 checks; e.g. star/test_suite/rsp_Cepheid/README.rst:13 "After 10 periods ... checks ... fundamental period is within 1% of the expected 3.92305 day period".
- Confidence: verified-by-reading-flexmesa-only (Python + the Rust flag computation); upstream side inferred from one README
- What: The Python wrapper is a pass-through (`src/flexmesa/star/e2e.py:66-91` -> `_rust.star_run_test_suite_case`, crates/flexmesa-py/src/star_e2e_py.rs:192). In Rust (crates/flexmesa-star/src/inlist_runner.rs:522-540) `assertions_passed = twin_match && verify_tierN_case_assertions(model, case_name)` and `passed = assertions_passed && (!verify_restart || photo_restart_verified)`. `verify_tier1_case_assertions` (crates/flexmesa-star/src/tier1_extras.rs:214-290) is a table of loose inequalities keyed by case NAME, e.g. `"make_zams" => hr.center_h1 > 0.1 && hr.log_center_t > 5.0 && hr.star_mass > 0.08`, `"7M_prems_to_AGB" => hr.star_mass > 1.0 && hr.log_center_t > 5.0`, after finiteness and sum(X)=1 +-1e-6. With `skip_optional=True` most cases start from an upstream-shipped `.mod` checkpoint, so after 2 steps these bounds are essentially properties of the loaded model. No test in tests/ calls any runner with `max_steps_per_stage=None` (grep: every call site passes 2, 3, 4 or 10). `photo_restart_passed` is FlexMESA-vs-FlexMESA (see T1-2). A green E2E suite therefore establishes: inlists/rn scripts parse, the model loads, 2 Newton steps converge, state is finite, and a self-restart is deterministic. It does not establish that any case reaches upstream's stopping condition or reproduces any upstream number. The test names/docstrings ("reproduction_and_restart_parity", "passes physical target assertions, and achieves 0 ULP `.photo` restart parity") and the "107/107 green" claim overstate this.
- Proposed fix: rename to "smoke"; add a slow tier that runs each case to its upstream termination condition and compares final history columns against real MESA output (L).

### T1-2: "0 ULP photo restart parity" is self-consistency of FlexMESA's own serializer, not parity with an upstream photo
- Area: tests/test_e2e_0_runner.py, tests/test_e2e_*; crates/flexmesa-star/src/inlist_runner.rs
- Severity: unverified claim
- FlexMESA: tests/test_e2e_0_runner.py:80-85 (`c_cont, c_rest = model.verify_photo_restart(n_steps=2); assert c_cont == c_rest`); inlist_runner.rs:492-495 (`let (c1, c2) = verify_photo_restart_parity(&model, 1)?; photo_restart_verified = c1 == c2;`) guarded by `if verify_restart && !photo_restart_verified` so it is evaluated for one step, on the first stage that yields it only.
- Confidence: verified-by-reading-flexmesa-only
- What: the check hashes FlexMESA state after continuing 1 step vs after serialise->deserialise->1 step. It proves FlexMESA's `.photo` round-trips all state that its own stepper reads. It says nothing about upstream. In multi-stage cases only the first stage is checked.
- Proposed fix: wording (S).

### T1-3: tests/test_golden.py does not compare anything against a real upstream test_output
- Area: tests/test_golden.py, src/flexmesa/testing/golden.py
- Severity: unverified claim
- FlexMESA: tests/test_golden.py:27-65 (inline `_SAMPLE_*_OUTPUT` strings), :157-182 (perturb the parsed sample by 1e-14 and 1e-6 and check the comparator accepts/rejects).
- Confidence: verified-by-reading-flexmesa-only
- What: all 5 tests exercise the parser/comparator on four hand-typed snippets (4 + 9 + 8 + ~6 numbers; the numbers themselves are real upstream digits, 15/15 high-precision literals found in upstream outputs). No FlexMESA physics is evaluated. The "golden" comparisons live in the per-module tests, which mostly use transcribed constants (see table).
- Proposed fix: none needed beyond naming; add a generic "every float in <pkg>/test/test_output is reproduced" harness (M-L).

### T1-4: Python "GYRE/ADIPLS/STELLA golden parity" test asserts only counts and line lengths
- Area: tests/test_star_wave5_2.py
- Severity: unverified claim
- FlexMESA: tests/test_star_wave5_2.py:499-534 (`assert len(modes_gyre) == 34`, `assert len(modes_adipls) == 46`, `assert len(lbol) == 30; assert vel`, `len(hyd_out.splitlines()) == 401`).
- Upstream: gyre/test/test_output (70 lines, 136 floats), adipls/test/obs.test.txt, stella/test/mesa.lbol are committed byte-identical under tests/fixtures/golden but this Python test never opens `gyre/test_output`, `obs.test.txt`, `mesa.lbol` or `mesa.vel_feII`.
- Confidence: verified-by-reading-flexmesa-only (fixtures verified identical to upstream by cmp)
- What: the test reads the upstream INPUT files (model.dat, Gemma.fgong, mesa.hyd/abn), runs FlexMESA's own BVP solvers, and checks the number of modes found (34, 46: self-pinned counts). No frequency is compared with upstream's. A Rust test (crates/flexmesa-star/src/tests_phase3_pulse_stella.rs) may do so; that is the Rust auditor's scope.
- Proposed fix: compare each mode frequency to gyre/test/test_output and obs.test.txt (S-M).

### T1-5: Benchmark "reference checksums" are FlexMESA-defined observables on synthetic profiles; 4 of them have been re-pinned 9 times, and one "Fortran MESA" checksum no longer matches the Fortran harness by a factor 120
- Area: tests/test_bench.py, tests/test_wave4_bench.py, tests/test_wave5_bench.py, benchmarks/fortran_mesa/
- Severity: unverified claim
- FlexMESA: tests/test_wave5_bench.py:19-20 docstring "the Python/JAX benchmark checksums match the pinned upstream Fortran MESA (`bench_kernels_wave5.f90`) checksums"; :50-55 `"w5_diff_rot": 2.551374595308515e3`; docs/design/wave-5-benchmark-report.md:30 `| w5_diff_rot | 128 | 3.12459668E+05† | 2.60059307e3 |`; benchmarks/fortran_mesa/bench_kernels_wave5.f90:69-100 (`s = (real(k, dp) - 0.5_dp) / real(nz, dp); r = rsun * (0.08_dp + 0.88_dp * s) ...`).
- Upstream: n/a. `bench_kernels_wave5.f90` is not an upstream file; it is a FlexMESA-authored Fortran re-implementation of the same synthetic workload that `use`s a handful of upstream helpers (`hydro_rotation: a`, `binary_utils: eval_rlobe`, `binary_tides: f3,f4`, `astero_lib`), with the rest of the "physics" (DSI/SSI/GSF/ST formulae, Caplan fit) hand-written in the harness itself.
- Confidence: verified-by-reading-flexmesa-only
- What: (i) No checksum in the three bench test files appears in any upstream file (0/28 literals traced). They are sums of FlexMESA kernel outputs over analytic zone profiles, pinned from FlexMESA's own run; the tests can only detect change, not error. The Fortran harness is not compiled or run in CI (ci.yml has no gfortran step), so "Fortran parity" is a claim recorded in docs, not a gate. (ii) History of re-pins (all in tests, value old -> new):
  - d5a1172 test_wave4_bench.py `star_evolve_step` 9.987143083869501e1 -> 9.931845982528495e1, plus an accepted ALTERNATIVE `_EXPECTED_WAVE4_EVOLVE_ALT_N8 = 9.987143083869501e1` (test accepted either value; "without cache" HELM fallback).
  - bd33084 `star_evolve_step` back to 9.987143083869501e1, ALT removed.
  - 7b2f388 `star_evolve_step` -> 9.987143083921163e1.
  - 6a78100 `star_hydro_eqns` 4.660073228335436e6 -> 4.253224624211422e6 (-8.7%); `star_evolve_step` -> 9.987143125142524e1.
  - 642da54 `star_newton_step` 1.33327716396219e2 -> 1.454270997871555e2 (+9.1%); `star_evolve_step` -> 1.005913169168671e2 (+0.7%); test_wave5_bench.py `w5_diff_rot` N=8 3.082380424983001e5 -> 2.551374595308515e3 and N=128 3.12459668e5 -> 2.60059307e3 (factor ~120).
  - d4b7a1e `w5_riemann_rsp2` -2.543418211343195e5 -> -2.543418211346364e5.
  (iii) In 642da54 the Rust/JAX `w5_diff_rot` kernel changed but `bench_kernels_wave5.f90` did not (git show --stat: no benchmarks/ file touched), so the pinned value is no longer what the Fortran harness prints; the docs table admits this in a dagger footnote while the section header above it still says "Upstream Fortran MESA, Safe Rust, Python/PyO3, and JAX match to 9 out of 9 printed significant digits" and the test docstring still says the checksums match Fortran. The commit message of 642da54 does not mention re-pinning test expectations at all (it lists "sync ... two-regime Spruit-Tayler dynamo across Python/JAX and Wave 4/5 benchmarks").
- Proposed fix: stop describing these as Fortran/upstream parity; if a Fortran cross-check is wanted, call the real upstream routine (e.g. `am_mixing` set_ST) from the harness and gate it in CI (M).

### T1-6: test_wave4_bench "matches upstream Fortran MESA star/test/LOGS/history.data" cites a file that does not exist upstream; 4-digit targets are of unverifiable origin and the tolerance is ~the size of the evolution
- Area: tests/test_wave4_bench.py
- Severity: unverified claim
- FlexMESA: tests/test_wave4_bench.py:66-87 (`model.evolve(max_steps=20, do_remesh=False, op_split_burn=False, use_bcyclic=True, max_age=1.0e5)`, `assert len(reports) == 11`, `abs(row.log_center_t - 7.4124) < 1.0e-3`, ... `abs(row.center_h1 - 0.696725) < 5.0e-5`).
- Upstream: star/test/test_output is one line, `stop because star_age >= max_age`; there is no star/test/LOGS/ in the repo at 0b2e6bb. star/test/inlist: `initial_mass = 4`, `max_age = 1d5`.
- Confidence: verified-by-reading-both-sides (absence of the file); origin of the numbers inferred
- What: the six targets (7.4124, 1.4368, 2.3746, 0.3703, 4.1698, 0.696725) are not in any upstream file; they presumably came from a terminal summary of a local MESA build but nothing in the repo records it. docs/design/wave-4-benchmark-report.md:15 shows FlexMESA at 7.412298 vs 7.412400, i.e. a 1e-4 dex difference after only 1e5 yr from ZAMS, with remeshing and op-split burning switched off (upstream star/test does not switch them off). This is the ONLY test in the Python suite that compares an evolved stellar model against (purported) upstream numbers. It runs 11 steps; `len(reports) == 11` is self-pinned.
- Proposed fix: commit the upstream history.data produced by a documented build and compare at full precision per step (M).

### T1-7: RSP LINA expected periods are self-pinned, and earlier the "expected" period was an INPUT to the function under test
- Area: tests/test_e2e_4_tier4.py
- Severity: unverified claim
- FlexMESA: tests/test_e2e_4_tier4.py:73-80 (`abs(modes[0]["period_days"] - 3.978697530821434) < 1.0e-5` and five more 16-digit constants).
- Upstream: star/test_suite/rsp_Cepheid/inlist_rsp_Cepheid:84 `x_ctrl(1) = 3.92305d0 ! expected period (in days)` (nonlinear period); none of the six 16-digit literals occurs anywhere in upstream.
- Confidence: verified-by-reading-both-sides
- What: before d4b7a1e the test called `star.eval_rsp_lina_modes(..., expected_f_period_days=3.9927)` and asserted `abs(period - 3.9927) < 1e-3`, i.e. the answer was passed in; the "unforced" variant was only required to lie in `2.5 < P < 6.0`. d4b7a1e removed the forcing argument and pinned FlexMESA's own 16-digit output. The same commit changed the 2nd-crossing edge assertions from `abs(tb - 6109.4) < 25` to `4800 < tr < 5600; tb > tr + 300`. The Rust test crates/flexmesa-star/tests/rsp_crossing_test.rs compares to r24 LINA values (Rust auditor's scope); the Python test is change-detection only.
- Proposed fix: point the Python test at the same upstream-derived values the Rust test uses (S).

## 3. Deliverable 1: test-provenance table (one row per file)

Method: every file read (assert lines + helpers); every float literal with >= 12 significant digits in each test file was looked up (first 12 significant digits, exponent-insensitive) in a corpus built from upstream `*/test/test_output*`, `*/test/*` data, and `*/{public,private,test/src}/*.f90` at 0b2e6bb ("traced N/M" below). All `*/test/test_output*` files in the bundle `the test-input bundle` are byte-identical to upstream (cmp, 26 files incl. neu 8.85 MB, mtx `*.data`, utils `token.txt`). Classes: (a) real upstream output, (b) analytic identity, (c) Rust vs JAX, (d) self-pinned / unknown origin, (e) smoke. "fns" = `def test_` count (parametrised items in brackets). Step caps apply only to star-evolution tests. No file uses `@pytest.mark.slow`, `skipif`, or `xfail`.

| File | fns | Compared against (class, approx share) | Tolerances | Step cap | Without data / MESA_DIR | Pass establishes upstream parity? |
|---|---|---|---|---|---|---|
| test_constants_utils.py | 7 | (a) 23/23 values of const/test/test_output transcribed (traced 16/16); utils dict/tokenizer vs utils/test/test_output + token.txt read live; (b) 2 fns | rtol 1e-15 | - | live-file part silently not executed if file absent (`if upstream_output.exists()`), transcribed part still runs | yes: const (complete), utils test output (complete) |
| test_math_autodiff.py | 10 | (a) math test_output.INTRINSIC transcribed (traced 63/63; upstream has 102 floats) and auto_diff/test/test_output parsed live (206 numeric vectors + 42 booleans, all lines); (b)/(c) jax hessian, check_grads ~4 fns | 1e-14 (TOL_MATH/TOL_AD) | - | live auto_diff compare skipped silently if file absent | yes for auto_diff (complete) and math intrinsics (most); only INTRINSIC variant, not CRLIBM/CRMATH |
| test_interp_1d.py | 5 | (a) 3 fns vs interp_1d/test/test_output transcribed (traced 25 in output + 58 in upstream test source = 83/83); (b) 2 fns | rtol 1e-12..1e-14 | - | self-contained | partially: selected rows of a 270-float output |
| test_interp_2d.py | 4 | (a) bicubic/bipm/akima rows (traced 31/51); (d) 20 literals not in upstream: cshep2 `golden_pts` (6x3 values @1e-12) and Franke max/mean error (`0.1985167087686014`, `0.03217025005461399` @1e-12) are FlexMESA outputs; akima table compared at 5e-4 to 3-decimal table; (b) biquadratic property | 1e-12 typical; 1e-5..1e-7 on bicubic derivs; 5e-4 akima | - | self-contained | partially (bicubic/bipm yes; renka790/cshep2 no: self-pinned) |
| test_mtx.py | 8 | (a) mtx/test/test_output square1/2 solutions (traced 7/7) + live `test_case.data` (12x9) and `block_tri.data` (25x979) solved and compared with upstream's stored solution; (b)/(c) ~5 fns (reconstruct rhs, PCR vs Thomas vs bcyclic) | 1e-12..1e-13 | - | live part silently skipped if files absent | yes for dense/block-tridiagonal solves on upstream's own matrices |
| test_num.py | 11 | (a) partial: vdpol `reference` (num/test/test_output:247) and a few solver end values (traced 4/8); (b) most root/min tests vs analytic optima; (d) 4 16-digit roots not in upstream | ODE/DAE vs reference: 2e-3, 2.5e-3, 1e-5, 5e-5, 5e-3; roots 1e-7..1e-13 | - | self-contained | partially/weakly. Docstring "100% golden verification" is false: upstream output has 570 floats; test checks solver results to 3 digits against the reference solution, not against upstream's printed solver output |
| test_neu.py | 10 | (a) all 10,000 x 30 values of neu/test/test_output read live, Rust and JAX; 145/146 transcribed literals traced; (c)/(b) flags additivity, FD, cutoff ~6 fns; (e) solar profile fn only checks finiteness (and silently `return`s if `test_1.0.data` missing, test_neu.py:311) | rel 1e-10 (loss), 2e-10 (JAX sources) | - | falls back to ~20 transcribed rows if file absent (no skip) | yes: neu (complete, 1e-10) |
| test_turb.py | 3 | (a) turb/test/test_output MLT scaling, TDC-vs-Cox, 31-row TDC dt sweep (traced 72 in output + 14 in source = 86/86) | rel 1e-12 | - | self-contained | yes: turb standalone test output (most of 136 floats) |
| test_turb_autodiff.py | 2 | (c) JAX vs Rust; (b) FD gradients | 1e-12 | - | self-contained | no (cross-backend) |
| test_chem.py | 8 | (a) chem/test/test_output o16, AG89, Lodders03 286 rows (traced 10/10 hi-prec; the 286 percentages have few digits); (b) abar/zbar closed forms; (c) Rust vs JAX 2 fns | 1e-15; Lodders pct atol 5e-7; mass_correction atol 1e-6 | - | needs data bundle (fails, not skips) | partially (upstream output has 1102 floats; selected sections) |
| test_colors.py | 6 [+14 params] | (a) 14 cases x ~10 magnitudes + SED samples from colors/test/test_output (traced 149/149; upstream 242 floats); (c) 4 fns Rust vs JAX; check_grads | rtol 1e-12 | - | needs data | yes for the 3 golden groups |
| test_ionization.py | 7 | (a) ionization/test/test_output: 28-value eval_ionization @1e-14 and fe56_in_he4 @atol 1e-7 (traced 28 + 2 = 30/30; upstream 41 floats); (c) 3 fns; check_grads 2 fns | 1e-14; 1e-7 | - | needs data | yes (most of the file) |
| test_eos_wave3a1.py | 6 | (c) dominant; (a) blend constants (1 traced) ; `dfermi` "gfortran oracle" 2.1896786848098353 NOT in upstream (d) | 1e-10..1e-15 | - | self-contained | no (cross-backend) |
| test_eos_wave3a2.py | 4 [+6 params] | (c) HELM Rust vs JAX on "golden states" (states are inputs only); JAX jacfwd vs analytic | `_rel_diff` < 1e-10 (res), 2e-6 (derivs), 5e-8 | - | needs data | no. "golden" in the name refers to input (rho,T) points only |
| test_eos_wave3a3.py | 6 | (c) PC/Skye Rust vs JAX; (b) inverse solver round-trip (recover rho_true) | 1e-6 (Skye off-grid!) .. 1e-12 | - | needs data | no. Fn named `..._match_fortran_goldens` contains only Rust-vs-JAX asserts (test_eos_wave3a3.py:30-76) |
| test_eos_wave3b1.py | 4 | (a) exactly 3 scalars from eos/test/test_output (two total pressures @1e-12, one log10_rho @1e-12; traced 3/3; upstream has 391 floats); (c) rest | 1e-10..1e-13; FD 1e-5 | - | needs data | marginally (3 of 391 values) |
| test_golden.py | 5 | parser self-test on inline snippets (digits are real upstream, 15/15) | 1e-12/1e-14 | - | self-contained | no (tests the comparator, not physics): T1-3 |
| test_hybrid_bridge.py | 6 | (c) Rust vs JAX; (b) libm identities | 1e-12..1e-15 | - | self-contained | no |
| test_schema.py | 5 | parser on inline sample .defaults/.list/inlist | exact | - | self-contained | no (infra) |
| test_data.py | 6 | synthetic tmp safetensors/tar round-trips | exact | - | self-contained | no (infra) |
| test_smoke.py | 1 | x64 enabled, version string | - | - | - | no |
| test_docs_snippets.py | 4 | docs links / snippets execute | - | - | - | no |
| test_mesa_paper_figures.py | 4 | (e)/(d) reads `landmarks.csv` ledgers written by FlexMESA's own `scripts/build_docs_figures.py` and checks loose windows (e.g. `120 <= solar_dnu_uhz <= 170`, `1.20 <= fe_core_mass <= 1.85`) | windows of 10-50% | - | needs generated figures | no. "Reproduces MESA paper figures" = own outputs inside wide physical windows |
| test_bench.py | 5 | (d) 19 N=8 checksums pinned from FlexMESA (traced 0/19); (c) stage0..3 parity | chk 1e-12/1e-13; stage 1e-12, 2e-9 (wave3a), 5e-12 (wave3b) | n/a | needs data | no: T1-5 |
| test_wave4_bench.py | 3 | (d) 5 checksums (0/5 traced, re-pinned 7x); (e) ZAMS init finite; purported (a) 4 Msun evolve vs 4-digit numbers of unrecorded origin | chk 1e-12; evolve 1e-3 dex, H 5e-5 | max_steps=20 (11 taken), remesh and op-split OFF | needs data | no / very weakly: T1-6 |
| test_wave5_bench.py | 1 | (d) 8 checksums (0/4 traced, w5_diff_rot re-pinned x120 away from the Fortran harness) ; (c) | 1e-12 (N=8), 1e-8 (N=128); stage 1e-12..5e-11 | n/a | - | no: T1-5 |
| test_star_wave4_1.py | 4 | (e)/(b) ZAMS load: nz==2464, sums to 1, `1e4 < teff < 2.5e4`; .mod/.photo self round-trip; (c) JAX vs Rust grid/MLT (gradt only rows 200+, rtol 1e-6; grav 1e-7) | 1e-12..1e-14; 1e-6/1e-7 | none | needs data | no |
| test_star_wave4_2.py | 4 | (c) residuals Rust vs JAX 1e-13; PCR vs FFI linear solve 1e-7; Newton converges `residual_norm <= 1e-6` | as listed | 1 Newton solve | needs data | no |
| test_star_wave4_3.py | 4 | (c) + conservation invariants (species mass, J) + check_grads | 1e-11..1e-14 | 1 step | needs data | no |
| test_star_wave4_4.py | 3 | (c) + invariants; 5-step 4 Msun evolve checked with windows (`4.10 < log_teff < 4.25`, `0.65 < center_h1 <= 0.70`), energy conserved to 2e-3 across remesh | windows | 5 steps | needs data | no (smoke) |
| test_star_wave5_1.py | 4 | (c) rotation/diffusion Rust vs JAX; invariants | 1e-12..1e-14 | none | needs data | no |
| test_star_wave5_2.py | 4 | (c) HLLC/RSP2 Rust vs JAX; check_grads; pulse exporters: line counts; "golden parity" fn: counts only (T1-4); X+Y+Z sum loosened 1e-14 -> 1e-3 (c947885) | 1e-13..1e-15; 1e-3 | none | fixtures in repo | no |
| test_binary_wave5_3.py | 6 | (a) all 5 values of binary/test/test_output (traced 4/4, 1 is 0.0) @1e-15/1e-12; (b) Eggleton/Kepler closed forms; (c); FD grads; (e) coupled system: `model_number == 1` | as listed | 1 step | self-contained | yes for the 5-number upstream binary unit output; nothing on evolution |
| test_astero_plotting_wave5_4.py | 4 | (b) round-trip: inject known surface-correction coefficient into synthetic frequencies, recover it (36/37 literals not in upstream; docstring cites `targets.inc`, values not found in upstream astero/); `dnu == 134.91` (the solar constant itself); (c); optimiser finds known minimum; plots render | 1e-12..1e-14; 1e-6/1e-7 | - | self-contained | no |
| test_phase5a_review.py | 3 | (b) baryon conservation; inlist parse; (c) JAX Jacobian vs Rust analytic partials | 1e-13..1e-15 | - | needs data | no |
| test_e2e_0_runner.py | 5 | parser on synthetic inlists/rn; self photo restart (T1-2); 2-stage synthetic pipeline (termination_code == "max_model_number" at 3/5 models); live: all upstream inlists/rn/.mod under 3 suites parse, `nz >= 2` | exact | 10 | last fn `pytest.skip` if MESA_DIR unset (fails in CI zero-skip mode) | no numerics; yes that upstream inlists/.mod files are parseable |
| test_e2e_1_tier1.py | 9 [7 synthetic + 33 upstream params] | (b) starspot / T-tau closed forms 3 fns; (e) hooks run + sums to 1; synthetic cases in tmp dirs NAMED after upstream cases; upstream: `passed`/`assertions_passed`/`photo_restart_passed` + finiteness | closed forms 1e-14 | 2 (upstream), 3 (synthetic) | upstream fn skips without MESA_DIR | no: T1-1 |
| test_e2e_2_tier2.py | 7 [+7 synthetic + 17 upstream] | (e) monotonic/positive checks on helper kernels; same pass flags | - | 2 | skip without MESA_DIR | no: T1-1 |
| test_e2e_3_tier3.py | 7 [+14 synthetic + 22 upstream] | (e) inequalities (`mdot < 0`, `m_fe > 1.0`, `dg1 == dg1` i.e. not-NaN); pass flags | - | 2 | skip without MESA_DIR | no: T1-1 |
| test_e2e_4_tier4.py | 6 [+7 synthetic] | inventory counts; (e) mode ordering; (d) 6 self-pinned LINA numbers (T1-7); pass flags for 24 tier-4 + 32 dev cases in ONE test function | LINA 1e-5 abs | 2 | skip without MESA_DIR | no |
| test_e2e_5_tier5.py | 4 | inventory 91+8+8=107 (matches upstream `do1_test_source` active counts: star 91, binary 8, astero 8: verified); (e) inequalities; pass flags; binary `max_relative_oracle_error` vs FlexMESA's own analytic oracles (1e-2, 1e-4, 1e-12, 1e-14) | as listed | 2 (star), 4 (binary) | skip without MESA_DIR | no |
| test_e2e_fidelity4.py | 2 | self .mod/.photo round-trip, step-redo determinism; 5 "representative" upstream cases `passed` | exact | 2 | skip without MESA_DIR | no |
| unit/test_kap_wave3b2.py | 7 [+6 params] | (a) kap/test/test_output all 3 blocks, both transcribed (traced 33/33) and parsed live (6 outputs x 3 blocks x 2 backends); (c) 3 fns | 1e-11/1e-12 (gs98, CO); 1e-6..2e-7 (AESOPUS block) | - | live fn `pytest.skip` if file absent | yes: kap (complete, but AESOPUS only to 1e-6) |
| unit/test_kap_wave3b2_jax.py | 2 | (c) jax.grad vs analytic; kapCN `log10 == -3.361` (traced) | 1e-12..1e-14 | - | needs data | no (cross-backend) |
| unit/test_rates_wave3b3.py | 5 | (a) rates/test/test_output read from committed byte-identical fixture: screening (3 modes), weak/ecapture scalars (traced 22/24; 2 untraced are x/11 fractions), all 9x307 raw rates (values <= 1e-290 masked: only the nonzero subset is checked, zeros are never asserted to be zero); (c) reaclib JAX vs Rust | 1e-11..1e-14 | - | fixture in repo | yes: rates sections 1-7 (largest genuine golden in the Python suite) |
| unit/test_net_atm_wave3b4.py | 6 | (a) atm table/uniform/irradiated ~7 values (8-17 digits; traced 6 in output + 3 in source) @1e-6..1e-13; net: eps_nuc 1 value @1e-8, burn end abundances 6 values @1e-5..1e-9 (other Rosenbrock variants @1e-3); (c)/(b) FD and custom_vjp for the rest | as listed | - | needs data | partially: ~14 of 91 (atm) + 140 (net) upstream floats |
| conftest.py | 0 | zero-skip plugin, JAX cache clearing | - | - | - | - |
| fixtures/ | - | see section 1 | - | - | - | - |
| src/flexmesa doctests | 57 `>>>` lines in 10 modules (schema, auto_diff, constants, math, data.manifest, testing.golden, utils) | (b) trivial examples | - | - | - | no |

Roll-up: of 46 entries, upstream numeric parity is genuinely established (class a, majority of the upstream unit output, tight tolerance) for const, utils, auto_diff, math intrinsics, mtx, neu, turb, colors, ionization, kap, rates, binary (5 numbers); partially for interp_1d/2d, chem, num (loose), atm/net, eos (3 scalars). It is NOT established by any Python test for: anything in `star` (structure, solver, mixing, winds, remesh, timestep, rotation, diffusion, RSP, pulsation exporters), binary evolution, astero, or any test_suite case. Those are covered only by Rust-vs-JAX agreement, invariants, self-pinned checksums and 2-step smoke runs.

## 4. Findings (second batch)

### T1-8: The `slow` marker is declared but never used; there is no long-running test anywhere, so "do slow tests run in CI" is moot
- Area: pyproject.toml, tests/
- Severity: unverified claim
- FlexMESA: pyproject.toml `addopts = "--doctest-modules --strict-markers -m 'not slow'"`, `markers = ["slow: marks full-grid or multi-step stellar evolution tests (> 5 s)"]`; `grep -rn "mark.slow" tests src` returns nothing. CLAUDE.md layout table says "`slow` marks full-grid or multi-step evolutionary tests".
- Confidence: verified-by-reading-flexmesa-only
- What: no test is marked slow, none is deselected, and no CI job overrides `-m`. The longest stellar evolution in the Python suite is 11 steps (test_wave4_bench.py:69-77); every test_suite case is capped at 2 steps (4 for binary). The whole e2e job (139 cases + extras) is budgeted at 40 minutes on a GitHub runner. There is no tier of tests, in CI or out, that evolves a model to an upstream stopping condition.
- Proposed fix: add real slow tests and a scheduled CI job that runs them (L).

### T1-9: Zero-skip gate and vacuous-pass paths
- Area: tests/conftest.py and the "live upstream file" tests
- Severity: maintainability
- FlexMESA: tests/conftest.py:52-66; tests/test_neu.py:198-216 (`if path.exists(): ... else: data = np.asarray(GOLDEN_ROWS)`), tests/test_neu.py:311-312 (`if not path.exists(): return`), tests/test_mtx.py:431,443,482, tests/test_math_autodiff.py:549, tests/test_constants_utils.py:288-291 (`if upstream_output.exists():`), tests/unit/test_kap_wave3b2.py:184-185 (`pytest.skip`).
- Confidence: verified-by-reading-flexmesa-only
- What: In CI (`MESA_DIR` set, `FLEXMESA_REQUIRE_ZERO_SKIPS=1`) the guards `if os.environ.get("MESA_DIR"): assert path.is_file()` make the live comparisons mandatory, and any `pytest.skip` fails the session. That part is sound, and commit bd33084 genuinely tightened it (it replaced several silent `return`s and `pytest.skip("Upstream case directory not found")` with hard asserts). Outside CI (a developer running plain `pytest` with no MESA_DIR and no cache) 5 tests silently drop their upstream comparison and still report PASSED, and the E2E upstream functions skip. Limits of the gate itself: it counts only `report.skipped`; it cannot see a test that loops over an empty list or returns early; the hooks live in tests/conftest.py so doctest items collected from `src/` are outside its scope (inferred from pytest's conftest scoping, not executed). In the Rust runner a stage whose inlist cannot be resolved is silently `continue`d (crates/flexmesa-star/src/inlist_runner.rs:465-468) and a `cp` stage with a missing source file is silently ignored (:442-453); the Python tests assert `total_steps >= 1`, which one surviving stage satisfies.
- Proposed fix: make missing-stage a hard error; assert `len(rep["stages"])` equals the number of `do_one` lines (S).

### T1-10: Tolerances loosened after the project adopted "never loosen a tolerance" (DECISIONS #9, commit 3860533, 2026-11-08), with no stated reason in the commit message or ledger
- Area: tests/ history
- Severity: unverified claim
- FlexMESA: commits listed below (found with a per-commit diff miner pairing -/+ lines that differ only in numbers; all 82 commits scanned).
- Confidence: verified-by-reading-flexmesa-only (git history)
- What (old -> new, file, commit, stated reason):
  - 97265a5 (11-07, before #9): tests/test_eos_wave3a3.py Skye and eos_dt Rust-vs-JAX `res` `< 1.0e-14 -> < 1.0e-11`. Message: "ensure XLA CPU forward-autodiff parity for ideal and Skye free energy" (no mention of the 1000x relaxation).
  - 89adf36 (11-08): tests/test_eos_wave3a2.py HELM composition-derivative autodiff `< 1.0e-6 -> < 1.5e-2`; reverted to `< 1.0e-6` by 3860533 the same day together with DECISIONS #9. This is the one case handled per the stated governance.
  - e9f8f2c (11-10): tests/test_eos_wave3b1.py Rust-vs-JAX `rtol 1e-14 -> 1e-13` (res) and `1e-14 -> 1e-11` (d_dlnd, d_dlnT). No reason given. Same commit TIGHTENED chem Lodders `atol 1e-6 -> 5e-7` and kap `1e-12 -> 1e-13`, and extended six kap expected constants from 16 to 17 digits (same values).
  - 51e1adc (11-10): tests/test_wave4_bench.py stage parity `max_rel_err < 1.0e-11 -> < 1.0e-10`. No reason given.
  - c947885 (11-10): tests/test_star_wave5_2.py pulse-export `X + Y + Z == 1` check `rtol 1.0e-14 -> 1.0e-3` (two places, lines 448-450 and 477-479), i.e. eleven orders of magnitude, when the comparison was switched to face-interpolated points. No reason given; not in the tolerance ledger (grep for `test_star_wave5_2` / `1.0e-3` finds nothing relevant in docs/benchmarks/tolerance-and-discrepancy-ledger.md).
  - 47991c0 (11-10, "Fix clean-CI HELM patches..."): tests/test_eos_wave3a2.py twelve asserts `1.0e-6 -> 2.0e-6`, `1.0e-8 -> 2.0e-8`, `1.0e-11 -> 2.0e-11`, and one `d_dlnd` zones parity `< 1.0e-9 -> < 2.0e-6` (2000x); tests/test_eos_wave3b1.py `rtol 1.0e-12 -> 2.0e-12` (two places). No reason in message.
  - 36e8dc1 (11-10, "Fix CI EOS test runtime..."): tests/test_eos_wave3a2.py `2.0e-8 -> 5.0e-8` (two); tests/test_eos_wave3a3.py `< 1.0e-12 -> < 1.0e-11` (two). No reason given.
  - d4b7a1e (11-12): tests/test_e2e_4_tier4.py expected LINA period `3.9927 @1e-3 -> 3.978697530821434 @1e-5` (expected value changed by 0.35%; see T1-7), 2nd-crossing edges `abs(tb-6109.4)<25, abs(tr-6009.4)<25 -> 4800<tr<5600, tb>tr+300`.
  - Checksum re-pins: see T1-5 (d5a1172, bd33084, 7b2f388, 6a78100, d4b7a1e, 642da54).
  All of the tolerance loosenings are on Rust-vs-JAX (class c) or self-consistency checks, none on a class (a) upstream comparison; I found no commit that relaxed a tolerance on a real upstream golden value. Pattern: every relaxation coincides with a "fix CI" commit, i.e. the tolerance was moved to where the two backends happened to agree on the CI runner.
- Proposed fix: record each in the ledger with the measured discrepancy and cause, or restore (S each).

### T1-11: EOS Python tests are almost entirely Rust-vs-JAX; "golden"/"fortran" in names does not mean an upstream comparison; the metric floors the denominator at 1
- Area: tests/test_eos_wave3a1.py, test_eos_wave3a2.py, test_eos_wave3a3.py, test_eos_wave3b1.py
- Severity: unverified claim
- FlexMESA: tests/test_eos_wave3a3.py:30-76 (`test_get_pc_and_skye_alfa_rust_and_jax_match_fortran_goldens`: every assert is `abs(float(a_r) - float(a_j)) < 1.0e-12`); tests/test_eos_wave3a2.py:23-26 (`denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)`); tests/test_eos_wave3b1.py:40-41 (the only upstream numbers: `1.3329474231512012e12`, `5.4349014621238190e10`) and :111 (`-6.3454687472634816`); tests/test_eos_wave3a1.py:79 (`2.1896786848098353` labelled "gfortran oracle", not present in any upstream file).
- Upstream: eos/test/test_output has 474 lines / 391 floats.
- Confidence: verified-by-reading-both-sides (literal tracing)
- What: across 20 EOS test functions the Python suite checks 3 upstream scalars. Everything else demonstrates that two FlexMESA implementations agree, which cannot detect an error common to both (both were written by the same agents from the same reading of the Fortran). `_rel_diff` is an absolute difference for any quantity with magnitude below 1 (e.g. composition derivatives, phase, fractions), so "< 2e-6" on those is not a relative statement. The byte-identical fixture tests/fixtures/golden/eos/test_output.txt is consumed only by the Rust test crates/flexmesa-eos/tests/wave3b1_golden_tests.rs (Rust auditor to confirm coverage there).
- Proposed fix: parse eos/test/test_output in Python too and assert every value for both backends (M).

### T1-12: test_num.py's "100% golden verification" is 3-digit agreement with reference solutions
- Area: tests/test_num.py
- Severity: unverified claim
- FlexMESA: tests/test_num.py:4 ("Covers 100% golden verification against upstream MESA ``num/test/test_output``"); :296-305 (`y_ref = [1.7632345401889102, -0.8356886819146621]; assert np.max(np.abs(r5.y - y_ref)) <= 2e-3`; Rosenbrock variants `<= 2.5e-3`); :347 (`<= 1e-5`), :409-410 (`<= 5e-5`), :447-448 (`<= 5e-3`), :487-488 (`<= 5e-5`).
- Upstream: num/test/test_output:247 `reference 1 1.7632345401889102D+00` (760 lines, 570 floats in the file).
- Confidence: verified-by-reading-both-sides
- What: the tests compare FlexMESA solver end states against the reference solution with tolerances of 1e-3..5e-3, which is what any correct ODE solver at rtol 1e-4 achieves. Upstream's own printed solver results (which would test that FlexMESA's dopri5/rodas/etc. take the same steps) are not compared. Roots/minima are compared to analytic optima at 1e-4..1e-13 (class b). So a pass shows the solvers work, not that they reproduce upstream's numerical behaviour (step counts, error control).
- Proposed fix: compare to the per-solver lines of num/test/test_output at their printed precision, including nstep/nfcn counters (M).

### T1-13: Smaller provenance issues
- Area: various
- Severity: unverified claim (minor)
- FlexMESA: (i) tests/test_interp_2d.py:470-471 and :486-512, cshep2/Franke expected values to 1e-12 are not in upstream interp_2d/test/test_output (20/51 literals untraced), yet the function is `test_renka790_cshep2_cubic_and_scattered_2d_parity` in a file whose header says upstream golden. (ii) tests/unit/test_rates_wave3b3.py:271-272 `mask = expected > 1.0e-290; assert_allclose(got[mask], expected[mask])`: entries where upstream prints 0 (1,050 of 2,763 by my count) are never checked to be zero in FlexMESA. (iii) tests/test_mesa_paper_figures.py:80-131 judges "MESA paper figure reproduction" by FlexMESA-computed scalars falling inside wide windows; no digitised paper or MESA data is involved. (iv) tests/test_astero_plotting_wave5_4.py:36-37 says it verifies "against `targets.inc`" / "upstream MESA synthetic targets"; the frequencies used are not found under upstream astero/ and the test is an inject-and-recover round trip. (v) tests/test_e2e_1_tier1.py:130-229: synthetic one-inlist cases are placed in temp dirs named `starspots`, `T_tau_gradr`, ... so the runner applies the tier-1 extras and name-keyed assertions; these pass for a generic 1 Msun ZAMS with `max_model_number = 2`.
- Confidence: verified-by-reading-flexmesa-only except (i), (iv) traced against upstream
- Proposed fix: relabel; assert zeros (S).

### T1-14: astero E2E uses a weaker assertion branch precisely when the step cap is set (which is always, in tests)
- Area: crates/flexmesa-astero/src/e2e.rs (entry point reached from tests/test_e2e_4_tier4.py via star.run_tier4_suite)
- Severity: unverified claim
- FlexMESA: crates/flexmesa-astero/src/e2e.rs:189-196 `let assertions_passed = if max_steps_per_stage.is_none() { h1_c <= 0.0015 && modes.iter().any(|m| m.n_p == 4 && ((m.freq_uhz - target_uhz) / target_uhz).abs() < 0.03) } else { !modes.is_empty() && modes.iter().all(|m| m.freq_uhz.is_finite() && m.freq_uhz > 0.0) };`
- Confidence: verified-by-reading-flexmesa-only
- What: the "real" target check (a p-mode within 3% of the target frequency at core-H exhaustion) only runs when uncapped. Every test passes `max_steps_per_stage=2`, so `assertions_passed` for this astero case means "at least one finite positive frequency was found". Other tier assertion tables should be checked for the same pattern by the Rust auditor.
- Proposed fix: report `assertions_passed = None/"not evaluated"` under a cap instead of True (S).

## 5. Matches (things that are what they claim)

- Inventory 91 star + 8 binary + 8 astero = 107 equals the active `do_one` lines in upstream `star|binary|astero/test_suite/do1_test_source` (counted: 91, 8, 8). tests/test_e2e_5_tier5.py:33-47.
- All 26 `*/test/test_output*` (+ mtx `.data`, utils `token.txt`, neu `test_1.0.data`) in the release bundle are byte-identical to upstream 0b2e6bb; the 16 non-RSP fixture files under tests/fixtures/golden are byte-identical to upstream.
- const: tests/test_constants_utils.py:70-105 covers all 23 lines of const/test/test_output at rtol 1e-15.
- neu: tests/test_neu.py:193-285 covers all 10,000 rows x 30 output columns of neu/test/test_output at 1e-10, both backends, mandatory in CI.
- auto_diff: tests/test_math_autodiff.py:530-566 covers all numeric and boolean records of auto_diff/test/test_output at 1e-14.
- mtx: tests/test_mtx.py:404-537 solves upstream's 12x9 and 25x979 block-tridiagonal systems and matches upstream's stored solutions at 1e-13 (Thomas and bcyclic).
- turb: tests/test_turb.py covers the MLT, TDC-vs-Cox and 31-row TDC sweep of turb/test/test_output at 1e-12 (86/86 literals traced).
- kap: tests/unit/test_kap_wave3b2.py:181-237 covers all 3 blocks of kap/test/test_output live for both backends.
- rates: tests/unit/test_rates_wave3b3.py covers screening, weak, ecapture, 9x307 raw rates (nonzero subset) of rates/test/test_output at 1e-11..1e-13.
- binary: tests/test_binary_wave5_3.py:35-116 covers all 5 numbers of binary/test/test_output.
- colors (149/149 literals), ionization (30/30), chem (10/10 high-precision), interp_1d (83/83) transcribed constants are genuine upstream digits.
- Literal tracing overall: summing the per-file counts, 761 >=12-digit literals in tests/: 567 occur in upstream test outputs, 79 only in upstream sources (inputs), 115 nowhere upstream. The 115 are concentrated in test_bench/test_wave4_bench/test_wave5_bench (28), test_astero (36), test_e2e_4_tier4 (20, incl. synthetic history rows), test_interp_2d (20).
- Zero-skip enforcement works as described for `pytest.skip`, and bd33084 converted silent early returns to hard asserts under MESA_DIR.
- No commit relaxed a tolerance on a class (a) upstream comparison.

## 6. Leads (not verified)

- `tests/fixtures/golden/rsp/*_r24.mod` (14 files stamped `version_number '0b2e6bb'`): presumably produced by a local upstream build; no generating script or log in the repo. Rust auditor should check what the Rust tests assert against them and whether the same build could have been FlexMESA's own writer (the header format alone cannot distinguish).
- 32 dev cases in FlexMESA vs 33 sub-directories under upstream `star/dev_cases_*` (3+3+13+7+7); one is not in the inventory. Not identified which.
- Whether `verify_tier2..5_case_assertions` are as loose as tier 1 (only tier 1 read); and whether other runners switch to weaker assertions under a step cap as astero does (T1-14).
- The Fortran harness `benchmarks/fortran_mesa/` is never built in CI; whether the documented Fortran checksums for waves 1-4 were actually produced by it was not checked. For wave 5 the doc itself shows a mismatch for `w5_diff_rot`.
- `crates/flexmesa-bench/tests/bench_tests.rs` was re-pinned in the same commit 642da54 (6 lines changed): Rust-side twin of T1-5.
- Whether doctest items under `src/` are covered by the zero-skip hooks (conftest scoping; not executed here).
- `flexmesa` is not importable in this audit environment, so nothing was executed; all statements are from reading code, git history and byte/literal comparison.
