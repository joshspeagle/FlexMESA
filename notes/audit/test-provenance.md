# Test provenance

What each test file compares against, and whether a pass establishes parity with upstream MESA (`main` at commit `0b2e6bb`). Paths are relative to the repository root at PR #138 head `91eda0b`. Detailed findings behind the verdicts are in [`tests-python.md`](tests-python.md) (`T1`) and [`tests-rust.md`](tests-rust.md) (`T2`).

Expectation classes: (a) real upstream MESA output, (b) analytic identity or finite difference, (c) another FlexMESA backend (Rust versus JAX), (d) FlexMESA's own earlier output, self-pinned constants or numbers of unrecorded origin, (e) smoke only (runs, finite, positive).

## Summary

- Module-level tests of the small standalone modules are real comparisons against upstream `test_output` files that are byte-identical to upstream. Coverage of each upstream file ranges from about 8% (`num`) to 100% (`const`, `turb`, `neu`, `auto_diff`, `mtx`).
- No test of `star` evolution, of any of the 91 star `test_suite` cases, or of the benchmark checksums compares against upstream output. All star and astero E2E tests in CI run 2 to 8 steps. Binary criteria are faithful to upstream only when run uncapped, and 7 of the 8 uncapped binary runs are `#[ignore]`.
- 18 Rust tests are `#[ignore]` and CI never passes `--ignored`. The `slow` pytest marker is declared and used by no test.

## Python tests (`tests/`)

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

## Rust tests

Source classes: (a) real upstream output, (b) analytic identity / finite difference, (c) other FlexMESA backend, (d) self-pinned / unknown origin, (e) smoke. "Missing data": F = fails, S = silently skips or shrinks, n/a = needs no external data.

| File | #tests | #ignore | Source | Tolerances | Step cap | Missing data | Pass => upstream parity? |
|---|---|---|---|---|---|---|---|
| astero/tests/astero_suite.rs | 9 | 5 (uncapped example_astero, fast_from_file, fast_scan_grid, fast_simplex, fast_newuoa: "slow uncapped campaign (~70..780s) verified locally") | (a) upstream criteria for surface_effects/gyre/adipls; (e) for 8-step run | 1e-12 synthetic; 3% freq; solar_tol 10 | 8 (bounded test), 2 (mock), none (adipls, gyre) | F | yes for 3 cases vs upstream's own criteria; no for 5 |
| astero/src/lib.rs (inline) | 3 | 0 | (a)+(b) | 1e-12..1e-5 | n/a | n/a | kernel-level |
| atm/tests/atm_golden_test.rs | 4 | 0 | (a) 38/45 + (b) FD | 1e-8..1e-14; FD 1e-6 | n/a | uses embedded/loaded tables | yes (kernel) |
| autodiff/tests/autodiff_tests.rs | 5 | 0 | (b) | 1e-14..1e-4 | n/a | n/a | n/a |
| autodiff/tests/golden_suite_tests.rs | 2 | 0 | (a) full file | 1e-14 | n/a | S (analytic subset) unless MESA_DIR/zero-skips | yes |
| autodiff/tests/order3_tests.rs | 3 | 0 | (b) | 1e-14..1e-15 | n/a | n/a | n/a |
| bench/tests/bench_tests.rs | 12 | 6 ("benchmark table generator for --release runs") | (d) self-pinned checksums + (c) stage cross-check | 1e-12 (1e-10 wave 4) | n/a | n/a | no (T2-12) |
| binary/tests/tier5_coupled_suite.rs | 17 | 7 (6 "slow uncapped campaign (~121..1376s)", 1 "tracked in #139") | (a) upstream oracles/tolerances; (b) | 1e-2, 1e-4, 1e-12, 1e-15 | 4 (all 8), 35 (double_bh), none (jdot_ls_check) | S (`return`) | partially (T2-9) |
| binary/tests/phase5c_lifecycle_extras.rs | 5 | 0 | (b)/(e) invariants | 1e-5..1e-14 | few steps | n/a | no |
| binary/src/lib.rs (inline) | 5 | 0 | (a) 4 literals + (b) | 1e-12..1e-15 | 2-4 | S | kernel-level |
| chem/src/lib.rs (inline) | 6 | 0 | (a) reads real file | 5e-7 / 1e-14 | n/a | S (literals only) unless MESA_DIR | yes |
| colors/src/lib.rs (inline) | 4 | 0 | (a) 37/161 | 1e-12 | n/a | n/a | yes (subset) |
| const/tests/const_tests.rs | 2 | 0 | (a) | 1e-15 | n/a | n/a | yes |
| data/tests/data_tests.rs | 4 | 0 | (b) SHA vectors, round trips | exact/1e-12 | n/a | n/a | n/a |
| eos/tests/eos_dxa_partials_test.rs | 2 | 0 | (b) FD | 5e-6..2e-5 | n/a | n/a | no |
| eos/tests/wave3a1_golden_tests.rs | 5 | 0 | (a) 5 blend lines + (b) | 1e-14..1e-15 | n/a | n/a | partly |
| eos/tests/wave3a2_helm_tests.rs | 2 | 0 | (d) unknown-origin "Fortran oracle" (227 literals, 7 upstream) | 1e-14 | n/a | n/a | unverifiable |
| eos/tests/wave3a2_helm_regimes_tests.rs | 2 | 0 | (d) (158 literals, 1 upstream) | 1e-14..1e-15 | n/a | n/a | unverifiable |
| eos/tests/wave3a3_pc_tests.rs | 1 | 0 | (d) (179, 4) | 1e-11 values, 5e-9 partials | n/a | n/a | unverifiable |
| eos/tests/wave3a3_pc_solid_tests.rs | 2 | 0 | (d) (186, 4) | 1e-11..1e-14 | n/a | n/a | unverifiable |
| eos/tests/wave3a3_skye_inverse_tests.rs | 3 | 0 | (d) (312, 5) | 1e-8..1e-13 | n/a | n/a | unverifiable |
| eos/tests/wave3b1_full_tables_tests.rs | 2 | 0 | (e) table shape | exact | n/a | depends on data dir | no |
| eos/tests/wave3b1_golden_tests.rs | 4 | 0 | (a) embedded identical copy | 1e-12 / 5e-11; inversions 1e-5 | n/a | n/a | yes |
| interp1d/tests/interp1d_tests.rs | 7 | 0 | (a) 30/99 + (b) proptest | 1e-12 | n/a | n/a | yes (subset) |
| interp2d/tests/interp2d_tests.rs | 5 | 0 | (a) 31/65 + (b) | 1e-10..1e-15 | n/a | n/a | yes (subset) |
| ionization/src/lib.rs (inline) | 4 | 0 | (a) 31/35 | tight | n/a | n/a | yes |
| kap/tests/kap_golden_test.rs | 6 | 0 | (a) 12 values + (b) FD + (e) table counts | 1e-11..1e-12; FD 1e-6 | n/a | S (full-table test shrinks, :339) | yes (3 points) |
| math/tests/math_tests.rs | 6 | 0 | (a) + (b) | 1e-13..1e-15 | n/a | n/a | yes |
| mtx/tests/block_tests.rs | 5 | 0 | (a) run-time read + (b) | 1e-9..1e-14 | n/a | S | yes |
| mtx/tests/eigen_test.rs | 2 | 0 | (b) | 1e-12..1e-14 | n/a | n/a | n/a |
| mtx/tests/mtx_tests.rs | 6 | 0 | (a) 9/13 | 1e-13..1e-14 | n/a | n/a | yes |
| net/tests/net_golden_test.rs | 5 | 0 | (a) 56/82 | 1e-5..1e-12 | n/a | n/a | yes |
| net/tests/net_jacobian_test.rs | 2 | 0 | (b) FD | 1e-4..1e-7 | n/a | n/a | no |
| net/src/burn_const_density.rs (inline) | 1 | 0 | (b)/(e) | - | n/a | n/a | no |
| neu/tests/neu_tests.rs | 6 | 0 | (a) all 10,000 rows | 1e-10 | n/a | S (6 rows) unless MESA_DIR | yes |
| num/tests/num_tests.rs | 14 | 0 | (a) few + (b) | 1e-6..1e-15 | n/a | n/a | partly |
| num/tests/ode_benchmarks_tests.rs | 5 | 0 | (a) 23 endpoint values + published reference solutions | 1e-4..1e-7 | n/a | n/a | partly |
| num/tests/ode_newton_tests.rs | 6 | 0 | (a) 5 + (b) | 1e-3..1e-13 | n/a | n/a | partly |
| rates/tests/rates_golden_test.rs | 5 | 0 | (a) embedded identical copy | 1e-11..1e-13 (5e-6 weak table) | n/a | n/a | yes |
| rates/src/lib.rs (inline) | 1 | 0 | (e) | - | n/a | n/a | no |
| schema/tests/schema_tests.rs | 3 | 0 | (e) parser | exact | n/a | n/a | n/a |
| star/src/tests.rs | 10 | 0 | (b) FD Jacobian, invariants, round trip | 1e-6..1e-14 | 1-few | n/a | no |
| star/src/tests_fidelity1.rs | 6 | 0 | (b) hand-computed formulas "match_fortran" (formula re-derivation, not Fortran output) | 1e-12 | n/a | n/a | no (formula-level) |
| star/src/tests_fidelity2.rs | 5 | 0 | (b) | 1e-6..1e-12 | n/a | n/a | no |
| star/src/tests_fidelity4.rs | 4 | 0 | (b)/(e) round trip, invariants | 1e-2..1e-12 | few | n/a | no |
| star/src/tests_phase3_pulse_stella.rs | 4 | 0 | (a) gyre/adipls/stella upstream fixtures | 5e-4 / 5e-2 / 1.6e-3 / 1.25e-2 / 0.35 dex | n/a | F (committed fixtures) | yes at stated (loose) tolerance |
| star/src/tests_wave5.rs | 7 | 0 | (e) E2E smoke via tier assertions; (b) helpers | generic inequalities | 2 per stage (10 in one synthetic) | S unless MESA_DIR (tier 2: S always) | no (T2-2..T2-8) |
| star/src/tier3_extras.rs (inline) | 1 | 0 | (e) | inequalities | 2 | S unless MESA_DIR | no |
| star/src/zams.rs (inline) | 1 | 0 | (e) catalog load | - | n/a | - | no |
| star/tests/phase5a_hydro_solver_test.rs | 4 | 0 | (b) AD Jacobian vs central FD | 1e-3..1e-9 | n/a | n/a | no |
| star/tests/phase5a_solver_norms_and_burn_test.rs | 4 | 0 | (b)/(e) convergence, photo round trip | 1e-3..1e-12 | 1 | n/a | no |
| star/tests/phase5b_fidelity2_test.rs | 9 | 0 | (b) analytic wind laws, conservation | 1e-6..1e-15 | few | n/a | no |
| star/tests/rsp2_test.rs | 5 | 0 | (b) hand oracles + 1 upstream cycle0 fixture | 1e-4..1e-15 | n/a | F | partly |
| star/tests/rsp_build_lina_test.rs | 1 | 0 | (a*) 7 `_init_r24.mod` | 1e-8..1e-13 | n/a | F | yes if fixtures are genuine upstream output |
| star/tests/rsp_hyd_step_test.rs | 3 | 0 | (a*) 7 `_step1_r24.mod` | 1e-6..1e-7 | 1 step | F | yes for one HYD step, same caveat |
| star/tests/rsp_crossing_test.rs | 2 | 0 | (a) upstream `list` (v11461, 3e-3/5e-4) + 13 hard-coded "r24" LINA values (1e-6/5e-6; origin not in repo) | as stated; edges +-5 K; 7M case window 4300-5600 K | n/a | S (7M case) unless zero-skips | partly |
| star/tests/rsp_io_eos_test.rs | 5 | 0 | (a) cycle0 fixtures round trip + (b) | 1e-10..1e-15 | n/a | F | I/O only |
| testing/tests/golden_tests.rs | 3 | 0 | (b) | 1e-12..1e-15 | n/a | n/a | n/a |
| turb/tests/turb_tests.rs | 3 | 0 | (a) 72/72 | 1e-12 (2e-11) | n/a | n/a | yes |
| turb/tests/turb_autodiff_and_mixing_tests.rs | 4 | 0 | (b) FD | 1e-4..1e-13 | n/a | n/a | no |
| utils/tests/utils_tests.rs | 5 | 0 | (a) run-time read | exact | n/a | S | yes |

E2E machinery in library code (not test files): star tiers 1-5 see T2-1..T2-8; binary e2e T2-9; astero e2e T2-10.

## Module-level golden `test_output` tests (Rust)

First fact (verified with `cmp`): every `<module>/test/test_output` in the bundle `the test-input bundle` is byte-identical to `<module>/test/test_output` for num, net, chem, auto_diff, ionization, kap, rates, turb, atm, interp_2d, interp_1d, eos, neu, const, mtx, utils, colors, math (.CRMATH). The committed copies `tests/fixtures/golden/{eos,kap,net,rates,atm}/test_output.txt`, `gyre/test_output`, `gyre/model.dat`, `adipls/{Gemma.fgong,obs.test.txt}`, `stella/mesa.{abn,hyd,lbol}` and the 7 `rsp/*_cycle0.mod` + `rsp/list` are byte-identical to upstream files. So the references are real.

"Coverage" below = upstream numbers (>= 6 sig. digits) in the module's test_output that the Rust test actually compares (counted by a script matching test-source literals to upstream values at 1e-9, or by reading the parse loop for file-reading tests).

| Crate | How the reference is obtained | Coverage of upstream test_output | Tolerance | Verdict |
|---|---|---|---|---|
| const | literals transcribed into const_tests.rs | 21/21 | rtol 1e-15 | real, complete |
| math | literals from test_output.INTRINSIC | 85/98 | 1e-13..1e-15 | real, near-complete |
| utils | reads upstream files at run time if present | token/output files | exact string | real (subset fallback offline) |
| auto_diff | golden_suite_tests.rs parses the real file at run time | 209 numeric + 42 boolean records (all) | 1e-14 | real, complete (analytic subset offline) |
| interp_1d | literals | 30/99 | rtol 1e-12 | real, ~30% |
| interp_2d | literals | 31/65 | 1e-10..1e-12 (5e-4 for upstream's 32-bit rows) | real, ~50% |
| mtx | literals (mtx_tests 9, block_tests 4) + run-time read of test_output/test_case.data | 13/13 | 1e-13..1e-14 | real, complete |
| num | literals | ~27/341 (root finders, simp38, 23 ODE endpoint values) | 1e-6..1e-14; ODE endpoints 1e-4..1e-7 | real, ~8% (solver statistics lines not compared) |
| chem | lib.rs inline test parses the real file | 286 Lodders rows @5e-7 (print precision) + 51 do_tests blocks @1e-14 | as stated | real, near-complete |
| colors | literals in lib.rs | 37/161 | 1e-12 (one 5e-3) | real, ~23% |
| ionization | literals in lib.rs | 31/35 | per-test | real, near-complete |
| eos | wave3b1: embedded identical copy parsed; wave3a1: 4-5 literals; wave3a2/3a3: unknown-origin tables (T2-15) | 176 + ~26 of 220 | 1e-12 / 5e-11 | real for wave3b1; unverifiable for wave3a2/3a3 |
| kap | literals | 12 outputs of 39 numbers (3 of upstream's 3 tests: log10kap, 2 partials, kap) | 1e-11..1e-12 (AESOPUS kap 2e-7) | real; dkap_dlnd/dlnT not compared |
| rates | `include_str!` of identical copy | 9 screening + 282x3 weak rows @5e-6 + 9x307 raw rates @1e-11 + ecapture/pycno | 1e-11..1e-13 | real, near-complete |
| net | literals | 56/82 | eps_nuc 1e-8, d(eps) 1e-5..1e-8, one-zone burn 1e-5..1e-9 | real, ~70% |
| neu | reads real file at run time | 10,000 rows x 34 cols (all) | 1e-10 (6 inlined rows 1e-12) | real, complete |
| atm | literals | 38/45 | 1e-8 (print precision) .. 1e-14 | real, ~85% |
| turb | literals | 72/72 | 1e-12 (2e-11 for 2 rows) | real, complete |
| star (gyre/adipls/stella) | reads identical upstream fixtures | GYRE 34 modes; ADIPLS 46 modes | GYRE freq 5e-4, E_norm 5e-2; ADIPLS freq 1.6e-3 (l=0) / 1.25e-2, inertia 0.35 dex | real reference, loose tolerance |
| star (RSP) | `tests/fixtures/golden/rsp/*_{init,step1}_r24.mod` (14 files, not in upstream tree; header `version_number '0b2e6bb'`, consistent with output of a real upstream build) | 7 models: STAH build + LINA, 1 HYD step | 1e-6..1e-8 on profiles; LINA period 1e-6, growth 5e-6 | probably real compiled-upstream output; generator not committed |
| binary (lib.rs) | 4 literals from binary/test/test_output | 4/4 | 1e-12..1e-13 | real, complete |
| astero | upstream surface_effects targets | 8 targets | 1e-12 synthetic | real |

Answer: these tests do use real upstream numbers (as transcribed literals, embedded identical copies, or run-time reads), at tight tolerances, and I found no loosening in their history. Coverage ranges from 8% (num) to 100%. They validate individual kernels at upstream's hand-picked test points; they say nothing about the coupled star solver. The one exception in provenance is eos wave3a2/3a3 (T2-15) and the bench checksums (T2-12).
