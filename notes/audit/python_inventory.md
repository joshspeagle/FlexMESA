# Python & JAX Port Inventory (generated)

<!-- GENERATED VIA scripts/dashboard/build_python_inventory.py (P1.5). Do not edit by hand. -->

Scope: `src/flexmesa/<pkg>/` (excluding `testing/` and `plotting/`) and externally-reachable top-level public Rust items across library crates.

## Classification & Staleness Rules

* **Kinds**: `PyO3 wrapper` (calls `_rust.*`), `JAX engine twin (frozen)` (`#166` frozen star/binary/astero modules), `JAX kernel` (non-frozen `@jax.jit` / `jnp` / `custom_vjp` kernels), and `pure-python mirror` (pure Python/NumPy implementations).
* **States**: `PyO3 wrapper` items delegate directly to Rust at runtime and are always `mirror-current`; non-PyO3 mirrors/JAX kernels are `mirror-stale` when the latest `git blame` committer timestamp over all lines of the paired Rust implementation span is newer than the latest over all lines of the Python span and the two latest commit SHAs differ (a timestamp heuristic, not a behavioural comparison), otherwise `mirror-current`. Unpaired Python items are `python-only`; externally-reachable public Rust items with neither a Python counterpart nor a PyO3 binding in `crates/flexmesa-py` are `rust-only`.

## Per-Subpackage Summary

Headline totals: **702** public Python items (**235** pure-python mirror, **212** PyO3 wrapper, **200** JAX kernel, **55** frozen JAX twin; **499** `mirror-current`, **172** `mirror-stale`, **31** `python-only`), plus **1501** `rust-only` public items and **22** `#295` JAX `xfail` tests.

| Subpackage | Python | `pure-python mirror` | `PyO3 wrapper` | `JAX kernel` | `JAX engine twin (frozen)` | `mirror-current` | `mirror-stale` | `python-only` | `rust-only` | `#295` xfails |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adipls` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 127 | 0 |
| `astero` | 30 | 0 | 18 | 0 | 12 | 24 | 6 | 0 | 44 | 0 |
| `atm` | 20 | 2 | 15 | 3 | 0 | 18 | 2 | 0 | 9 | 0 |
| `auto_diff` | 43 | 4 | 0 | 39 | 0 | 35 | 7 | 1 | 38 | 0 |
| `bench` | 49 | 15 | 24 | 10 | 0 | 42 | 5 | 2 | 26 | 7 |
| `binary` | 24 | 0 | 12 | 0 | 12 | 14 | 10 | 0 | 74 | 0 |
| `chem` | 29 | 2 | 24 | 3 | 0 | 29 | 0 | 0 | 11 | 0 |
| `colors` | 34 | 1 | 14 | 19 | 0 | 25 | 9 | 0 | 15 | 0 |
| `constants` | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `data` | 38 | 32 | 6 | 0 | 0 | 16 | 3 | 19 | 3 | 0 |
| `eos` | 37 | 0 | 23 | 14 | 0 | 30 | 4 | 3 | 145 | 3 |
| `gyre` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 106 | 0 |
| `interp_1d` | 25 | 10 | 5 | 10 | 0 | 23 | 2 | 0 | 1 | 0 |
| `interp_2d` | 19 | 6 | 4 | 9 | 0 | 14 | 4 | 1 | 6 | 0 |
| `ionization` | 13 | 1 | 7 | 5 | 0 | 11 | 2 | 0 | 3 | 2 |
| `kap` | 38 | 1 | 9 | 28 | 0 | 23 | 12 | 3 | 30 | 2 |
| `math` | 44 | 1 | 0 | 43 | 0 | 9 | 34 | 1 | 15 | 0 |
| `mtx` | 34 | 27 | 3 | 4 | 0 | 21 | 12 | 1 | 70 | 0 |
| `net` | 12 | 5 | 6 | 1 | 0 | 10 | 2 | 0 | 34 | 0 |
| `neu` | 17 | 15 | 1 | 1 | 0 | 16 | 1 | 0 | 1 | 0 |
| `num` | 50 | 48 | 0 | 2 | 0 | 32 | 18 | 0 | 6 | 0 |
| `rates` | 25 | 3 | 16 | 6 | 0 | 19 | 6 | 0 | 19 | 0 |
| `replay` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 29 | 0 |
| `schema` | 16 | 12 | 4 | 0 | 0 | 16 | 0 | 0 | 9 | 0 |
| `star` | 46 | 0 | 15 | 0 | 31 | 26 | 20 | 0 | 659 | 8 |
| `turb` | 47 | 44 | 1 | 2 | 0 | 35 | 12 | 0 | 2 | 0 |
| `utils` | 11 | 5 | 5 | 1 | 0 | 10 | 1 | 0 | 19 | 0 |
| **TOTAL** | **702** | **235** | **212** | **200** | **55** | **499** | **172** | **31** | **1501** | **22** |

## Oldest `mirror-stale` Rows

| Python path:line | Symbol | Rust path:line | Kind | Python last | Rust last |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `src/flexmesa/data/loader.py:136` | `save_safetensors_bundle` | `crates/flexmesa-data/src/safetensors.rs:354` | `pure-python mirror` | `2026-11-05T04:54Z` (`8ec9a4b2`) | `2026-11-27T01:00Z` (`b0a3dc24`) |
| `src/flexmesa/data/loader.py:72` | `encode_safetensors_bytes` | `crates/flexmesa-data/src/safetensors.rs:200` | `pure-python mirror` | `2026-11-05T04:54Z` (`8ec9a4b2`) | `2026-11-27T01:00Z` (`b0a3dc24`) |
| `src/flexmesa/auto_diff/kernels.py:108` | `unary_log10` | `crates/flexmesa-autodiff/src/kernels.rs:133` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-05T15:13Z` (`175bbf4b`) |
| `src/flexmesa/auto_diff/kernels.py:124` | `unary_log2` | `crates/flexmesa-autodiff/src/kernels.rs:155` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-05T15:13Z` (`175bbf4b`) |
| `src/flexmesa/auto_diff/kernels.py:292` | `unary_rpow_scalar` | `crates/flexmesa-autodiff/src/kernels.rs:381` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-05T15:25Z` (`2f9e39eb`) |
| `src/flexmesa/auto_diff/kernels.py:303` | `binary_pow` | `crates/flexmesa-autodiff/src/kernels.rs:392` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-05T15:25Z` (`2f9e39eb`) |
| `src/flexmesa/auto_diff/kernels.py:281` | `unary_pow_scalar` | `crates/flexmesa-autodiff/src/kernels.rs:370` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-06T20:05Z` (`380dbd62`) |
| `src/flexmesa/math/intrinsics.py:46` | `safe_sqrt` | `crates/flexmesa-math/src/lib.rs:100` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-16T14:51Z` (`3924b8ff`) |
| `src/flexmesa/math/intrinsics.py:127` | `log10` | `crates/flexmesa-math/src/lib.rs:168` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-17T01:19Z` (`2a594542`) |
| `src/flexmesa/math/intrinsics.py:163` | `exp` | `crates/flexmesa-math/src/lib.rs:204` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-17T01:19Z` (`2a594542`) |
| `src/flexmesa/math/intrinsics.py:187` | `exp10` | `crates/flexmesa-math/src/lib.rs:229` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-17T01:19Z` (`2a594542`) |
| `src/flexmesa/math/intrinsics.py:213` | `pow` | `crates/flexmesa-math/src/lib.rs:308` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-17T01:19Z` (`2a594542`) |
| `src/flexmesa/math/trig.py:103` | `atanh` | `crates/flexmesa-math/src/trig.rs:333` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-17T01:19Z` (`2a594542`) |
| `src/flexmesa/math/trig.py:108` | `sgn` | `crates/flexmesa-math/src/io.rs:12` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-17T01:19Z` (`2a594542`) |
| `src/flexmesa/math/trig.py:123` | `heaviside` | `crates/flexmesa-math/src/io.rs:28` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-17T01:19Z` (`2a594542`) |
| `src/flexmesa/math/trig.py:138` | `str_to_double` | `crates/flexmesa-math/src/io.rs:46` | `pure-python mirror` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-17T01:19Z` (`2a594542`) |
| `src/flexmesa/math/trig.py:155` | `str_to_vector` | `crates/flexmesa-math/src/io.rs:109` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-17T01:19Z` (`2a594542`) |
| `src/flexmesa/math/trig.py:18` | `cos` | `crates/flexmesa-math/src/trig.rs:35` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-17T01:19Z` (`2a594542`) |
| `src/flexmesa/math/trig.py:23` | `sin` | `crates/flexmesa-math/src/trig.rs:47` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-17T01:19Z` (`2a594542`) |
| `src/flexmesa/math/trig.py:28` | `tan` | `crates/flexmesa-math/src/trig.rs:73` | `JAX kernel` | `2026-11-05T14:52Z` (`3663d87f`) | `2026-11-17T01:19Z` (`2a594542`) |

## `#295` JAX `xfail` Tests (22)

| Test path:line | Test | Subpackage | Target symbol(s) | Reason |
| :--- | :--- | :--- | :--- | :--- |
| `tests/test_bench.py:115` | `test_wave1_interp_2d_stages_agree` | `bench` | `bench_interp2d` | #295: Rust interp_evbicub_db in strict mode now reproduces MESA 26.4.1 fvbicub_db implicit-REAL (f32) basis scalars (interp_2d/private/bicub_db.f:24,1371-1698; #672, #680), whereas the frozen JAX twin keeps pure f64 basis scalars; measured cross-stage max_rel_err is 2.980e-8 (N=4) and 2.980e-8 (N=8) against 1.0e-12. |
| `tests/test_bench.py:174` | `test_wave2_ion_fe56_stages_agree` | `bench` | `eval_charge_of_fe56_in_he4`, `bench_ion_fe56` | #295: Rust Fe56He4Table uses MESA 26.4.1 fvbicub_db implicit f32 basis scalars (interp_2d/private/bicub_db.f:1371-1575; #413), whereas the frozen JAX twin keeps f64 basis scalars; measured cross-stage max_rel_err is 3.250e-8 against 1.0e-12. |
| `tests/test_bench.py:208` | `test_wave3a_eos_pc_stages_agree` | `bench` | `eval_pc_eos`, `eval_pc_eos_jax`, `bench_eos_pc` | #295: Rust PC uses the MESA 26.4.1 EXCOR7 bicubic spline table (eos/private/pc_eos.f90:1895, create_EXCOR7_table.f90:136 THETA > 0.005; #161), whereas the frozen JAX twin keeps analytical EXCOR7 (THETA > 0.04); measured cross-stage max_rel_err is 3.716e-9 against 2.0e-9. |
| `tests/test_bench.py:241` | `test_wave3b_atm_table_stages_agree` | `bench` | `eval_table_atm`, `eval_table_jax`, `bench_atm_table` | #295: Rust table_atm uses MESA 26.4.1 interp_2d/private/bicub_db.f implicit f32 fvbicub_db basis scalars in strict mode (#420, #161), whereas the frozen JAX twin keeps pure f64 basis scalars per DECISIONS #43; measured cross-stage max_rel_err is 6.897e-8 (N=4) and 4.705e-6 (N=8) against 5.0e-12. |
| `tests/test_wave4_bench.py:163` | `test_wave4_star_micro_vars_rust_matches_frozen_jax` | `bench` | `eval_face_mlt_and_brunt_jax`, `bench_star_micro_vars` | #295: Rust do1_mlt_2 takes the face radiation pressure from the face temperature (MESA 26.4.1 star_utils.f90:3838-3843, #291); the frozen JAX face-MLT twin still interpolates the cell values (measured max_rel_err 8.9e-3). |
| `tests/test_wave4_bench.py:178` | `test_wave4_star_hydro_eqns_rust_matches_frozen_jax` | `bench` | `eval_hydro_residuals_jax`, `bench_star_hydro_eqns` | #295: Rust eval_face_dm_bar at k = nz - 1 uses 0.5*(dm[k-1] + dm[k]) (MESA 26.4.1 star/private/hydro_momentum.f90:191-196, #340); frozen JAX eval_hydro_residuals_jax still uses dm_bar[nz - 1] (measured max_rel_err 9.73e-3). |
| `tests/test_wave5_bench.py:93` | `test_wave5_diff_rot_rust_matches_frozen_jax` | `bench` | `eval_caplan_ocp_dstar_jax`, `bench_w5_diff_rot` | #295: Rust eval_caplan_ocp_dstar follows MESA 26.4.1 get_Dstar_OCP (diffusion_support.f90:1716-1743, #156); the frozen JAX twin eval_caplan_ocp_dstar_jax keeps its earlier fit (max_rel_err about 5.4e-5). |
| `tests/test_eos_wave3a3.py:496` | `test_eval_pc_eos_and_zones_match_across_rust_and_jax` | `eos` | `eval_pc_eos_zones`, `eval_pc_eos` | #295: Rust PC uses the MESA 26.4.1 EXCOR7 bicubic spline table (pc_eos.f90:1895, create_EXCOR7_table.f90 THETA > 0.005; #161), whereas the frozen JAX twin keeps analytical EXCOR7 (THETA > 0.04), differing by up to 2.17e-6 on res and 4.35e-5 on derivatives. |
| `tests/test_eos_wave3a3.py:1013` | `test_pc_multispecies_buckets_rust_matches_frozen_jax` | `eos` | `eval_pc_eos_zones`, `eval_pc_eos` | #295: Rust PC uses the MESA 26.4.1 EXCOR7 bicubic spline table (pc_eos.f90:1895, create_EXCOR7_table.f90 THETA > 0.005; #161), whereas the frozen JAX twin keeps analytical EXCOR7 (THETA > 0.04), differing by up to 3.72e-6 on res and 9.93e-8 on derivatives across the 9- and 35-species PC mixtures. |
| `tests/test_eos_wave3b1.py:816` | `test_helm_ideal_edge_skip_blend_rust_and_jax_agree` | `eos` | `eos_dt_get`, `eos_dt_get_jax` | #295: AutoDiffReal2VarOrder3 evaluates MESA 26.4.1's auto_diff expressions (#349), which moves the cancellation-dominated de_dd of the Rust ideal-gas EOS by a few ULPs of its 5.7e10 terms; the JAX twin keeps its own rounding. |
| `tests/test_ionization.py:98` | `test_fe56_in_he4_golden_rust_vs_jax` | `ionization` | `eval_charge_of_fe56_in_he4` | #295: Rust Fe56He4Table uses MESA 26.4.1 fvbicub_db implicit f32 basis scalars (interp_2d/private/bicub_db.f:1371-1575; #413), whereas the frozen JAX twin keeps f64 basis scalars; measured max abs diff is 4.680e-9 against rtol=1e-15, atol=1e-15. |
| `tests/test_ionization.py:171` | `test_fe56_in_he4_zones_rust_vs_jax_parity` | `ionization` | `eval_charge_of_fe56_in_he4_zones` | #295: Rust Fe56He4Table uses MESA 26.4.1 fvbicub_db implicit f32 basis scalars (interp_2d/private/bicub_db.f:1371-1575; #413), whereas the frozen JAX twin keeps f64 basis scalars; measured max_rel_err is 5.969e-8 on value and 4.434e-5 on derivatives. |
| `tests/unit/test_kap_wave3b2.py:335` | `test_rust_strict_vs_jax_backend_consistency_across_all_regimes` | `kap` | `kap_get_results`, `kap_get_results_jax` | #295: Rust kapCN and AESOPUS evaluate MESA 26.4.1 interp_2d/private/bicub_db.f implicit f32 fvbicub_db basis scalars in strict mode (#433, #161), whereas the frozen JAX twin keeps f64 basis scalars per DECISIONS #43; measured max rel diff is 2.103e-7 against rtol=1e-12. |
| `tests/unit/test_kap_wave3b2_jax.py:112` | `test_kapcn_knot_and_multi_zone_batch_consistency` | `kap` | `kap_get_kapcn`, `kap_get_zones`, `kap_get`, `kap_get_results` | #295: Rust AESOPUS evaluates MESA 26.4.1 interp_2d/private/bicub_db.f implicit f32 fvbicub_db basis scalars in strict mode (#433, #161), whereas the frozen JAX twin keeps f64 basis scalars per DECISIONS #43; measured max rel diff is 1.713e-8 against rtol=1e-12, atol=1e-12. |
| `tests/test_hybrid_bridge.py:231` | `test_interp2d_rust_strict_and_jax_consistency_and_autodiff` | `star` | - | #295: Rust interp_evbicub_db in strict mode now reproduces MESA 26.4.1 fvbicub_db implicit-REAL (f32) basis scalars (interp_2d/private/bicub_db.f:24,1371-1698; #672, #680), whereas the frozen JAX twin keeps pure f64 basis scalars; measured diffs are f rel=1.800e-8 (abs=5.557e-9), df_dx rel=5.789e-10 (abs=4.896e-10), df_dy rel=2.034e-8 (abs=3.031e-9) against rtol=1e-12. |
| `tests/test_interp_2d.py:198` | `test_interp2d_bicub_rust_strict_vs_jax_autodiff` | `star` | - | #295: Rust interp_evbicub_db in strict mode now reproduces MESA 26.4.1 fvbicub_db implicit-REAL (f32) basis scalars (interp_2d/private/bicub_db.f:24,1371-1698; #672, #680), whereas jax.grad/jax.hessian differentiate the frozen pure-f64 JAX twin; measured diffs at (0.1, 0.2) are df_dx rel=2.112e-8 (abs=1.736e-8), df_dy rel=7.129e-9 (abs=4.157e-10), d2f_dx2 rel=3.211e-8 (abs=1.826e-8), d2f_dy2 rel=3.313e-8 (abs=4.680e-9), d2f_dxdy rel=1.844e-8 (abs=1.001e-8) against rtol=1e-13, atol=1e-14. |
| `tests/test_star_wave4_1.py:102` | `test_star_jax_face_gradr_and_gradt_match_rust` | `star` | `eval_face_mlt_and_brunt_jax`, `eval_face_mlt_jax` | #295: Rust computes the face radiation pressure from the face temperature (#291); the frozen JAX twin still interpolates it from the two cell values. |
| `tests/test_star_wave4_2.py:46` | `test_eval_equations_rust_jax_residual_consistency` | `star` | `eval_hydro_vars`, `eval_equations`, `eval_hydro_residuals_jax` | #295: Rust eval_face_dm_bar at k = nz - 1 uses 0.5*(dm[k-1] + dm[k]) (MESA 26.4.1 star/private/hydro_momentum.f90:191-196, #340); frozen JAX eval_hydro_residuals_jax still uses dm_bar[nz - 1], so zone nz - 1 momentum residual differs by 1.12e-5 against rtol=1e-13, atol=1e-13. |
| `tests/test_star_wave4_4.py:40` | `test_adaptive_remeshing_and_mesh_gval_jax_consistency` | `star` | `eval_mesh_gval_jax` | #295: the Rust pre-MS model is now upstream's build_pre_ms_model (#154); at its 4 Msun inputs the frozen JAX eval_mesh_gval_jax (max/abs, not smooth) fails check_grads: JVP 322.85 against numerical 227.43 at rtol=1e-5. |
| `tests/test_star_wave5_1.py:239` | `test_caplan_ocp_dstar_jax_matches_rust` | `star` | `eval_caplan_ocp_dstar_jax` | #295: Rust eval_caplan_ocp_dstar follows MESA 26.4.1 get_Dstar_OCP (#156); the frozen JAX twin keeps its earlier expression. |
| `tests/test_star_wave5_1.py:264` | `test_burgers_velocities_jax_match_rust` | `star` | `eval_burgers_velocities_jax` | #295: Rust eval_burgers_velocities follows MESA 26.4.1 element diffusion (#156); the frozen JAX twin keeps its earlier expressions. |
| `tests/test_star_wave5_1.py:305` | `test_phase_separation_shift_jax_matches_rust` | `star` | `eval_phase_separation_shift_jax` | #295: Rust eval_phase_separation_shift follows the MESA 26.4.1 Blouin fits (#156); the frozen JAX twin keeps its earlier expression. |

## Python & JAX Items (702)

| Python path:line | Rust path:line | Kind | State | `#295` xfail |
| :--- | :--- | :--- | :--- | :--- |
| `src/flexmesa/astero/drivers.py:16` (`match_modes_by_degree`) | `crates/flexmesa-astero/src/ratios_chi2.rs:12` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/drivers.py:41` (`eval_ratios_r02`) | `crates/flexmesa-astero/src/ratios_chi2.rs:487` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/drivers.py:85` (`eval_astero_chi2`) | `crates/flexmesa-astero/src/ratios_chi2.rs:626` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/drivers.py:263` (`nelder_mead_minimize`) | `crates/flexmesa-astero/src/optimize.rs:87` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/drivers.py:304` (`powell_quadratic_minimize`) | `crates/flexmesa-astero/src/optimize.rs:674` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/drivers.py:339` (`scan_cartesian_grid`) | `crates/flexmesa-astero/src/optimize.rs:789` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/drivers.py:364` (`run_astero_test_suite_case`) | `crates/flexmesa-astero/src/e2e.rs:177` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/jax_kernels.py:23` (`eval_delta_nu_asymptotic_jax`) | `crates/flexmesa-astero/src/scaling.rs:94` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/astero/jax_kernels.py:41` (`eval_delta_nu_scaling_jax`) | `crates/flexmesa-astero/src/scaling.rs:108` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/astero/jax_kernels.py:61` (`eval_delta_nu_radius_scaling_jax`) | `crates/flexmesa-astero/src/scaling.rs:121` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/astero/jax_kernels.py:79` (`eval_nu_max_jax`) | `crates/flexmesa-astero/src/scaling.rs:134` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/astero/jax_kernels.py:99` (`eval_nu_ac_jax`) | `crates/flexmesa-astero/src/scaling.rs:147` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/astero/jax_kernels.py:124` (`eval_acoustic_profile_jax`) | `crates/flexmesa-astero/src/scaling.rs:34` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/astero/jax_kernels.py:165` (`eval_delta_pg_traditional_jax`) | `crates/flexmesa-astero/src/scaling.rs:167` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/astero/jax_kernels.py:194` (`eval_cubic_corr_jax`) | `crates/flexmesa-astero/src/surface_corr.rs:246` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/astero/jax_kernels.py:226` (`eval_combined_corr_jax`) | `crates/flexmesa-astero/src/surface_corr.rs:284` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/astero/jax_kernels.py:265` (`eval_ratios_r01_r10_jax`) | `crates/flexmesa-astero/src/ratios_chi2.rs:236` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/astero/jax_kernels.py:322` (`eval_ratios_r02_jax`) | `crates/flexmesa-astero/src/ratios_chi2.rs:487` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/astero/jax_kernels.py:371` (`eval_astero_chi2_jax`) | `crates/flexmesa-astero/src/ratios_chi2.rs:626` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/astero/scaling_corr.py:20` (`eval_acoustic_profile`) | `crates/flexmesa-astero/src/scaling.rs:34` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/scaling_corr.py:67` (`eval_delta_nu_asymptotic`) | `crates/flexmesa-astero/src/scaling.rs:94` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/scaling_corr.py:86` (`eval_delta_nu_scaling`) | `crates/flexmesa-astero/src/scaling.rs:108` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/scaling_corr.py:111` (`eval_delta_nu_radius_scaling`) | `crates/flexmesa-astero/src/scaling.rs:121` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/scaling_corr.py:134` (`eval_nu_max`) | `crates/flexmesa-astero/src/scaling.rs:134` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/scaling_corr.py:157` (`eval_nu_ac`) | `crates/flexmesa-astero/src/scaling.rs:147` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/scaling_corr.py:194` (`eval_delta_pg`) | `crates/flexmesa-astero/src/scaling.rs:167` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/scaling_corr.py:238` (`eval_glitch_acoustic_depths`) | `crates/flexmesa-astero/src/scaling.rs:246` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/scaling_corr.py:272` (`eval_radial_delta_nu_ols`) | `crates/flexmesa-astero/src/scaling.rs:292` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/scaling_corr.py:291` (`eval_surface_correction`) | `crates/flexmesa-astero/src/surface_corr.rs:141` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/astero/scaling_corr.py:392` (`eval_ratios_r01_r10`) | `crates/flexmesa-astero/src/ratios_chi2.rs:236` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:29` (`AtmEvalOutput`) | `crates/flexmesa-atm/src/types.rs:254` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:65` (`AtmIrradiatedOutput`) | `crates/flexmesa-atm/src/types.rs:281` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:89` (`eval_t_tau`) | `crates/flexmesa-atm/src/t_tau_relations.rs:41` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:103` (`eval_t_tau_dq_dtau`) | `crates/flexmesa-atm/src/t_tau_relations.rs:92` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:116` (`get_t_tau_base`) | `crates/flexmesa-atm/src/t_tau_relations.rs:21` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:128` (`get_t_tau_id`) | `crates/flexmesa-atm/src/types.rs:193` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:141` (`get_table_id`) | `crates/flexmesa-atm/src/types.rs:212` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:155` (`eval_e2`) | `crates/flexmesa-atm/src/utils.rs:84` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:168` (`eval_paczynski_gradr`) | `crates/flexmesa-atm/src/utils.rs:48` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:210` (`eval_table`) | `crates/flexmesa-atm/src/table.rs:404` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:248` (`get_table_alfa_beta`) | `crates/flexmesa-atm/src/table.rs:513` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:281` (`get_table_base`) | `crates/flexmesa-atm/src/table.rs:607` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:293` (`eval_t_tau_uniform`) | `crates/flexmesa-atm/src/t_tau_uniform.rs:124` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:359` (`eval_t_tau_varying`) | `crates/flexmesa-atm/src/t_tau_varying.rs:150` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:417` (`eval_t_tau_atm`) | `crates/flexmesa-atm/src/lib.rs:92` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_core.py:494` (`eval_table_atm`) | `crates/flexmesa-atm/src/lib.rs:211` | `PyO3 wrapper` | `mirror-current` | `tests/test_bench.py:241::test_wave3b_atm_table_stages_agree` |
| `src/flexmesa/atm/_core.py:575` (`eval_irradiated`) | `crates/flexmesa-atm/src/irradiated.rs:124` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/atm/_jax.py:21` (`eval_t_tau_jax`) | `crates/flexmesa-atm/src/t_tau_relations.rs:41` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/atm/_jax.py:65` (`eval_paczynski_gradr_jax`) | `crates/flexmesa-atm/src/utils.rs:48` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/atm/_jax.py:235` (`eval_table_jax`) | `crates/flexmesa-atm/src/table.rs:404` | `JAX kernel` | `mirror-stale` | `tests/test_bench.py:241::test_wave3b_atm_table_stages_agree` |
| `src/flexmesa/auto_diff/base.py:21` (`AutoDiffBase`) | - | `JAX kernel` | `python-only` | - |
| `src/flexmesa/auto_diff/kernels.py:27` (`unary_sign`) | `crates/flexmesa-autodiff/src/kernels.rs:22` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:33` (`unary_abs`) | `crates/flexmesa-autodiff/src/kernels.rs:30` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:38` (`unary_sqrt`) | `crates/flexmesa-autodiff/src/kernels.rs:37` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:44` (`unary_safe_sqrt`) | `crates/flexmesa-autodiff/src/kernels.rs:51` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:60` (`unary_exp`) | `crates/flexmesa-autodiff/src/kernels.rs:64` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:66` (`unary_expm1`) | `crates/flexmesa-autodiff/src/kernels.rs:72` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/auto_diff/kernels.py:72` (`unary_exp10`) | `crates/flexmesa-autodiff/src/kernels.rs:81` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:79` (`unary_powm1`) | `crates/flexmesa-autodiff/src/kernels.rs:91` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:86` (`unary_log`) | `crates/flexmesa-autodiff/src/kernels.rs:100` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:92` (`unary_safe_log`) | `crates/flexmesa-autodiff/src/kernels.rs:112` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:102` (`unary_log1p`) | `crates/flexmesa-autodiff/src/kernels.rs:125` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:108` (`unary_log10`) | `crates/flexmesa-autodiff/src/kernels.rs:133` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/auto_diff/kernels.py:114` (`unary_safe_log10`) | `crates/flexmesa-autodiff/src/kernels.rs:142` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:124` (`unary_log2`) | `crates/flexmesa-autodiff/src/kernels.rs:155` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/auto_diff/kernels.py:130` (`unary_sin`) | `crates/flexmesa-autodiff/src/kernels.rs:164` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:137` (`unary_cos`) | `crates/flexmesa-autodiff/src/kernels.rs:173` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:144` (`unary_tan`) | `crates/flexmesa-autodiff/src/kernels.rs:183` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:151` (`unary_sinpi`) | `crates/flexmesa-autodiff/src/kernels.rs:192` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:158` (`unary_cospi`) | `crates/flexmesa-autodiff/src/kernels.rs:201` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:165` (`unary_tanpi`) | `crates/flexmesa-autodiff/src/kernels.rs:210` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:172` (`unary_sinh`) | `crates/flexmesa-autodiff/src/kernels.rs:219` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:179` (`unary_cosh`) | `crates/flexmesa-autodiff/src/kernels.rs:228` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:186` (`unary_tanh`) | `crates/flexmesa-autodiff/src/kernels.rs:237` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:193` (`unary_asin`) | `crates/flexmesa-autodiff/src/kernels.rs:246` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:200` (`unary_acos`) | `crates/flexmesa-autodiff/src/kernels.rs:255` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:207` (`unary_atan`) | `crates/flexmesa-autodiff/src/kernels.rs:264` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:213` (`unary_asinpi`) | `crates/flexmesa-autodiff/src/kernels.rs:273` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:219` (`unary_acospi`) | `crates/flexmesa-autodiff/src/kernels.rs:283` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:225` (`unary_atanpi`) | `crates/flexmesa-autodiff/src/kernels.rs:293` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:231` (`unary_asinh`) | `crates/flexmesa-autodiff/src/kernels.rs:303` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:238` (`unary_acosh`) | `crates/flexmesa-autodiff/src/kernels.rs:313` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:245` (`unary_atanh`) | `crates/flexmesa-autodiff/src/kernels.rs:323` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/kernels.py:251` (`unary_pown`) | `crates/flexmesa-autodiff/src/kernels.rs:334` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/auto_diff/kernels.py:281` (`unary_pow_scalar`) | `crates/flexmesa-autodiff/src/kernels.rs:370` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/auto_diff/kernels.py:292` (`unary_rpow_scalar`) | `crates/flexmesa-autodiff/src/kernels.rs:381` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/auto_diff/kernels.py:303` (`binary_pow`) | `crates/flexmesa-autodiff/src/kernels.rs:392` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/auto_diff/order1.py:25` (`DualOrder1`) | `crates/flexmesa-autodiff/src/order1.rs:32` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/auto_diff/order1.py:242` (`AutoDiffReal1VarOrder1`) | `crates/flexmesa-autodiff/src/lib.rs:45` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/auto_diff/order1.py:250` (`AutoDiffReal2VarOrder1`) | `crates/flexmesa-autodiff/src/lib.rs:48` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/auto_diff/order1.py:258` (`AutoDiffReal4VarOrder1`) | `crates/flexmesa-autodiff/src/lib.rs:51` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/auto_diff/order1.py:266` (`AutoDiffRealStarOrder1`) | `crates/flexmesa-autodiff/src/lib.rs:57` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/auto_diff/order2.py:30` (`AutoDiffReal2VarOrder2`) | `crates/flexmesa-autodiff/src/order2.rs:35` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/bench/bicub_fast.py:16` (`locate_uniform_cell_jax`) | `crates/flexmesa-interp2d/src/bicub_eval.rs:116` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/bench/bicub_fast.py:43` (`eval_bicub_point_jax`) | `crates/flexmesa-bench/src/bicub_fast.rs:28` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/bench/harness.py:27` (`LatencyStats`) | `crates/flexmesa-bench/src/lib.rs:178` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/bench/harness.py:44` (`BenchRecord`) | `crates/flexmesa-bench/src/lib.rs:189` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/bench/harness.py:69` (`measure_us`) | `crates/flexmesa-bench/src/lib.rs:220` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/bench/harness.py:123` (`measure_and_capture`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/bench/harness.py:157` (`measure_stage3_batch`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/bench/harness.py:191` (`max_rel_diff`) | `crates/flexmesa-bench/src/lib.rs:248` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/bench/harness.py:223` (`max_err3`) | `crates/flexmesa-bench/src/lib.rs:283` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/bench/harness.py:238` (`ThermoGrid`) | `crates/flexmesa-bench/src/lib.rs:294` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/bench/harness.py:293` (`BicubTable`) | `crates/flexmesa-bench/src/lib.rs:354` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/bench/harness.py:351` (`make_block_tridiag_fixture`) | `crates/flexmesa-bench/src/tridiag_fast.rs:19` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_micro.py:67` (`bench_autodiff`) | `crates/flexmesa-bench/src/kernels_micro.rs:175` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_micro.py:134` (`bench_interp2d`) | `crates/flexmesa-bench/src/bicub_fast.rs:28` | `PyO3 wrapper` | `mirror-current` | `tests/test_bench.py:115::test_wave1_interp_2d_stages_agree` |
| `src/flexmesa/bench/kernels_micro.py:267` (`bench_neu`) | `crates/flexmesa-neu/src/lib.rs:274` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_solver.py:164` (`bench_turb_mlt`) | `crates/flexmesa-autodiff/src/order1.rs:32` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_solver.py:190` (`bench_turb_tdc`) | `crates/flexmesa-autodiff/src/order1.rs:32` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_solver.py:216` (`run_all_wave1_benchmarks`) | `crates/flexmesa-bench/src/lib.rs:406` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave2.py:51` (`make_wave2_composition_profile`) | `crates/flexmesa-bench/src/kernels_wave2.rs:50` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave2.py:128` (`bench_chem_comp`) | `crates/flexmesa-chem/src/composition.rs:102` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave2.py:183` (`bench_ion_fe56`) | `crates/flexmesa-ionization/src/fe56_he4.rs:40` | `PyO3 wrapper` | `mirror-current` | `tests/test_bench.py:174::test_wave2_ion_fe56_stages_agree` |
| `src/flexmesa/bench/kernels_wave2.py:225` (`bench_ion_4d`) | `crates/flexmesa-ionization/src/ion_table_load.rs:69` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave2.py:286` (`bench_colors_sed`) | `crates/flexmesa-colors/src/sed_cube.rs:560` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave2.py:338` (`run_all_wave2_benchmarks`) | `crates/flexmesa-bench/src/kernels_wave2.rs:318` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave3a.py:77` (`bench_eos_ideal`) | `crates/flexmesa-eos/src/ideal.rs:27` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave3a.py:154` (`bench_eos_helm`) | `crates/flexmesa-eos/src/helm/convert.rs:150` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave3a.py:214` (`bench_eos_pc`) | `crates/flexmesa-eos/src/pc/mod.rs:33` | `PyO3 wrapper` | `mirror-current` | `tests/test_bench.py:208::test_wave3a_eos_pc_stages_agree` |
| `src/flexmesa/bench/kernels_wave3a.py:348` (`bench_eos_skye`) | `crates/flexmesa-eos/src/helm/table.rs:100` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave3a.py:452` (`run_all_wave3a_benchmarks`) | `crates/flexmesa-bench/src/kernels_wave3a.rs:323` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave3b.py:138` (`bench_eos_4d`) | `crates/flexmesa-eos/src/tables_4d/spline.rs:35` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave3b.py:209` (`bench_kap_fixed`) | `crates/flexmesa-kap/src/eval.rs:28` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave3b.py:299` (`bench_rates_screen`) | `crates/flexmesa-rates/src/loader.rs:454` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave3b.py:356` (`bench_net_eval`) | `crates/flexmesa-chem/src/nuclides.rs:255` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave3b.py:501` (`bench_atm_table`) | `crates/flexmesa-atm/src/table.rs:63` | `PyO3 wrapper` | `mirror-current` | `tests/test_bench.py:241::test_wave3b_atm_table_stages_agree` |
| `src/flexmesa/bench/kernels_wave3b.py:568` (`run_all_wave3b_benchmarks`) | `crates/flexmesa-bench/src/kernels_wave3b.rs:450` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave4.py:122` (`make_wave4_zams_model`) | `crates/flexmesa-bench/src/kernels_wave4.rs:36` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave4.py:183` (`bench_star_micro_vars`) | `crates/flexmesa-bench/src/kernels_wave4.rs:133` | `JAX kernel` | `mirror-current` | `tests/test_wave4_bench.py:163::test_wave4_star_micro_vars_rust_matches_frozen_jax` |
| `src/flexmesa/bench/kernels_wave4.py:286` (`bench_star_hydro_eqns`) | `crates/flexmesa-bench/src/kernels_wave4.rs:195` | `JAX kernel` | `mirror-current` | `tests/test_wave4_bench.py:178::test_wave4_star_hydro_eqns_rust_matches_frozen_jax` |
| `src/flexmesa/bench/kernels_wave4.py:399` (`bench_star_newton_step`) | `crates/flexmesa-mtx/src/block_tridiag.rs:472` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave4.py:460` (`bench_star_remesh`) | `crates/flexmesa-bench/src/kernels_wave4.rs:347` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave4.py:506` (`bench_star_evolve_step`) | `crates/flexmesa-bench/src/kernels_wave4.rs:407` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave4.py:589` (`run_all_wave4_benchmarks`) | `crates/flexmesa-bench/src/kernels_wave4.rs:482` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave5.py:402` (`bench_w5_diff_rot`) | `crates/flexmesa-bench/src/kernels_wave5.rs:54` | `PyO3 wrapper` | `mirror-current` | `tests/test_wave5_bench.py:93::test_wave5_diff_rot_rust_matches_frozen_jax` |
| `src/flexmesa/bench/kernels_wave5.py:418` (`bench_w5_riemann_rsp2`) | `crates/flexmesa-bench/src/kernels_wave5.rs:137` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave5.py:434` (`bench_w5_binary_rlof`) | `crates/flexmesa-bench/src/kernels_wave5.rs:293` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave5.py:450` (`bench_w5_astero_seism`) | `crates/flexmesa-bench/src/kernels_wave5.rs:342` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/bench/kernels_wave5.py:491` (`run_all_wave5_benchmarks`) | `crates/flexmesa-bench/src/kernels_wave5.rs:528` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/bench/tdc_fast.py:76` (`eval_tdc_point_jax`) | `crates/flexmesa-bench/src/kernels_solver.rs:210` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/bench/tridiag_fast.py:32` (`bench_mtx_block_tridiag`) | `crates/flexmesa-mtx/src/block_tridiag.rs:472` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/binary/jax_jdot_tides.py:18` (`eval_bh_spin_and_edd_jax`) | `crates/flexmesa-binary/src/edd_wind.rs:150` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/binary/jax_jdot_tides.py:77` (`eval_jdot_gr_jax`) | `crates/flexmesa-binary/src/jdot_tides.rs:23` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/binary/jax_jdot_tides.py:113` (`eval_jdot_mb_jax`) | `crates/flexmesa-binary/src/jdot_tides.rs:140` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/binary/jax_jdot_tides.py:158` (`eval_edot_tidal_hut_jax`) | `crates/flexmesa-binary/src/jdot_tides.rs:414` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/binary/jax_orbit_mdot.py:17` (`eval_rlobe_jax`) | `crates/flexmesa-binary/src/orbit.rs:21` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/binary/jax_orbit_mdot.py:41` (`eval_rl_relative_gap_jax`) | `crates/flexmesa-binary/src/orbit.rs:46` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/binary/jax_orbit_mdot.py:64` (`separation_from_period_jax`) | `crates/flexmesa-binary/src/orbit.rs:59` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/binary/jax_orbit_mdot.py:87` (`period_from_separation_jax`) | `crates/flexmesa-binary/src/orbit.rs:72` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/binary/jax_orbit_mdot.py:110` (`eval_j_orb_jax`) | `crates/flexmesa-binary/src/orbit.rs:85` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/binary/jax_orbit_mdot.py:136` (`separation_from_j_orb_jax`) | `crates/flexmesa-binary/src/orbit.rs:100` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/binary/jax_orbit_mdot.py:167` (`eval_ritter_mdot_thin_jax`) | `crates/flexmesa-binary/src/mdot.rs:28` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/binary/jax_orbit_mdot.py:252` (`solve_roche_lobe_mdot_jax`) | `crates/flexmesa-binary/src/mdot_implicit.rs:175` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/binary/orbit_mdot.py:17` (`eval_rlobe`) | `crates/flexmesa-binary/src/orbit.rs:21` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/binary/orbit_mdot.py:41` (`eval_rl_relative_gap`) | `crates/flexmesa-binary/src/orbit.rs:46` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/binary/orbit_mdot.py:70` (`separation_from_period`) | `crates/flexmesa-binary/src/orbit.rs:59` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/binary/orbit_mdot.py:98` (`period_from_separation`) | `crates/flexmesa-binary/src/orbit.rs:72` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/binary/orbit_mdot.py:127` (`eval_j_orb`) | `crates/flexmesa-binary/src/orbit.rs:85` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/binary/orbit_mdot.py:160` (`separation_from_j_orb`) | `crates/flexmesa-binary/src/orbit.rs:100` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/binary/orbit_mdot.py:191` (`eval_ritter_mdot_thin`) | `crates/flexmesa-binary/src/mdot.rs:28` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/binary/orbit_mdot.py:260` (`eval_kolb_mdot_thick`) | `crates/flexmesa-binary/src/mdot.rs:192` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/binary/orbit_mdot.py:297` (`eval_l2_mass_loss_fraction`) | `crates/flexmesa-binary/src/disk.rs:140` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/binary/orbit_mdot.py:328` (`pick_next_timestep_rl`) | `crates/flexmesa-binary/src/timestep.rs:78` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/binary/system.py:13` (`BinarySystem`) | `crates/flexmesa-binary/src/system.rs:121` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/binary/system.py:189` (`run_binary_test_suite_case`) | `crates/flexmesa-binary/src/e2e.rs:151` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:33` (`NuclideInfo`) | `crates/flexmesa-chem/src/nuclides.rs:181` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:66` (`chem_get_element_id`) | `crates/flexmesa-chem/src/elements.rs:106` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:86` (`element_name`) | `crates/flexmesa-chem/src/elements.rs:119` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:103` (`element_long_name`) | `crates/flexmesa-chem/src/elements.rs:128` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:120` (`element_main_iso_name`) | `crates/flexmesa-chem/src/elements.rs:136` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:137` (`category_id`) | `crates/flexmesa-chem/src/elements.rs:144` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:149` (`category_name`) | `crates/flexmesa-chem/src/elements.rs:156` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:166` (`get_partition_fcn_indx`) | `crates/flexmesa-chem/src/elements.rs:169` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:178` (`generate_nuclide_names`) | `crates/flexmesa-chem/src/elements.rs:225` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:192` (`generate_long_nuclide_names`) | `crates/flexmesa-chem/src/elements.rs:232` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:208` (`get_nuclide_index`) | `crates/flexmesa-chem/src/nuclides.rs:255` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:236` (`get_nuclide_info`) | `crates/flexmesa-chem/src/nuclides.rs:255` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:251` (`ag89_zsol`) | `crates/flexmesa-chem/src/solar_ag89.rs:14` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:260` (`ag89_yesol`) | `crates/flexmesa-chem/src/solar_ag89.rs:29` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:269` (`chem_xsol`) | `crates/flexmesa-chem/src/solar_ag89.rs:57` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:281` (`chem_get_solar`) | `crates/flexmesa-chem/src/solar_ag89.rs:42` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:293` (`lodders03_element_atom_percent`) | `crates/flexmesa-chem/src/solar_ag89.rs:64` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:305` (`ag89_namsol`) | `crates/flexmesa-chem/src/solar_ag89.rs:42` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:314` (`get_stable_mass_frac`) | `crates/flexmesa-chem/src/solar_ag89.rs:78` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:335` (`solar_element_zfrac`) | `crates/flexmesa-chem/src/solar_scales.rs:132` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:351` (`solar_cno_fraction`) | `crates/flexmesa-chem/src/solar_scales.rs:183` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:366` (`chem_m_div_h`) | `crates/flexmesa-chem/src/solar_scales.rs:195` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/catalog.py:383` (`reaction_qtotal`) | `crates/flexmesa-chem/src/nuclides.rs:255` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/composition.py:39` (`CompositionInfo`) | `crates/flexmesa-chem/src/composition.rs:20` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/chem/composition.py:187` (`get_composition_info_jax`) | `crates/flexmesa-chem/src/composition.rs:281` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/chem/composition.py:223` (`get_composition_info`) | `crates/flexmesa-chem/src/composition.rs:281` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/chem/composition.py:334` (`basic_composition_info`) | `crates/flexmesa-chem/src/composition.rs:281` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/chem/composition.py:358` (`composition_info`) | `crates/flexmesa-chem/src/composition.rs:281` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/chem/composition.py:382` (`get_composition_info_zones`) | `crates/flexmesa-chem/src/composition.rs:102` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/colors/photometry.py:98` (`eval_colors_jax`) | `crates/flexmesa-colors/src/photometry.rs:295` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/colors/photometry.py:137` (`eval_colors`) | `crates/flexmesa-colors/src/sed_cube.rs:560` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/colors/photometry.py:208` (`eval_colors_dual`) | `crates/flexmesa-colors/src/sed_cube.rs:560` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/colors/photometry.py:295` (`eval_colors_zones`) | `crates/flexmesa-colors/src/sed_cube.rs:560` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/colors/sed.py:26` (`h00`) | `crates/flexmesa-colors/src/interp.rs:31` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/colors/sed.py:31` (`h10`) | `crates/flexmesa-colors/src/interp.rs:41` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/colors/sed.py:36` (`h01`) | `crates/flexmesa-colors/src/interp.rs:51` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/colors/sed.py:41` (`h11`) | `crates/flexmesa-colors/src/interp.rs:61` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/colors/sed.py:101` (`compute_interp_radius_jax`) | `crates/flexmesa-colors/src/utils.rs:283` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/colors/sed.py:208` (`eval_sed_hermite_jax`) | `crates/flexmesa-colors/src/interp.rs:139` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/colors/sed.py:237` (`eval_sed_linear_jax`) | `crates/flexmesa-colors/src/interp.rs:295` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/colors/sed.py:288` (`eval_sed_knn_jax`) | `crates/flexmesa-colors/src/interp.rs:377` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/colors/sed.py:360` (`eval_sed`) | `crates/flexmesa-colors/src/sed_cube.rs:560` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/colors/synthetic.py:24` (`flux_to_magnitude`) | `crates/flexmesa-colors/src/photometry.rs:47` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/colors/synthetic.py:40` (`clean_sed_fluxes_jax`) | `crates/flexmesa-colors/src/photometry.rs:29` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/colors/synthetic.py:55` (`calculate_bolometric_phot_jax`) | `crates/flexmesa-colors/src/photometry.rs:65` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/colors/synthetic.py:77` (`calculate_bolometric_phot`) | `crates/flexmesa-colors/src/photometry.rs:65` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/colors/synthetic.py:109` (`calculate_synthetic_flux_jax`) | `crates/flexmesa-colors/src/photometry.rs:91` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/colors/synthetic.py:135` (`calculate_synthetic_jax`) | `crates/flexmesa-colors/src/photometry.rs:131` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/colors/synthetic.py:166` (`calculate_synthetic`) | `crates/flexmesa-colors/src/photometry.rs:131` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/colors/synthetic.py:216` (`compute_vega_zero_point`) | `crates/flexmesa-colors/src/photometry.rs:175` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/colors/synthetic.py:256` (`compute_ab_zero_point`) | `crates/flexmesa-colors/src/photometry.rs:197` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/colors/synthetic.py:286` (`compute_st_zero_point`) | `crates/flexmesa-colors/src/photometry.rs:231` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/colors/utils.py:30` (`ColorsBuiltinGrid`) | `crates/flexmesa-colors/src/sed_cube.rs:54` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/colors/utils.py:65` (`get_builtin_colors_grid`) | `crates/flexmesa-colors/src/sed_cube.rs:560` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/colors/utils.py:100` (`remove_dat`) | `crates/flexmesa-colors/src/utils.rs:18` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/colors/utils.py:115` (`dilute_flux_jax`) | `crates/flexmesa-colors/src/utils.rs:29` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/colors/utils.py:138` (`dilute_flux`) | `crates/flexmesa-colors/src/utils.rs:29` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/colors/utils.py:168` (`trapezoidal_integration_jax`) | `crates/flexmesa-colors/src/utils.rs:40` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/colors/utils.py:184` (`trapezoidal_integration`) | `crates/flexmesa-colors/src/utils.rs:40` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/colors/utils.py:215` (`simpson_integration_jax`) | `crates/flexmesa-colors/src/utils.rs:126` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/colors/utils.py:259` (`simpson_integration`) | `crates/flexmesa-colors/src/utils.rs:126` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/colors/utils.py:290` (`interpolate_array_jax`) | `crates/flexmesa-colors/src/utils.rs:182` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/colors/utils.py:311` (`interpolate_array`) | `crates/flexmesa-colors/src/utils.rs:182` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/constants/__init__.py:296` (`MixingType`) | `crates/flexmesa-const/src/lib.rs:315` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/data/converters.py:21` (`parse_lodders03_text`) | `crates/flexmesa-data/src/converters.rs:132` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/data/converters.py:47` (`parse_isotopes_text`) | `crates/flexmesa-data/src/converters.rs:192` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/data/converters.py:77` (`parse_helm_table_text`) | `crates/flexmesa-data/src/converters.rs:328` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/data/converters.py:136` (`convert_lodders03_file`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/converters.py:156` (`convert_isotopes_file`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/converters.py:176` (`convert_helm_table_file`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:29` (`UpstreamReference`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:57` (`ReleaseBundle`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:78` (`EmbeddedIonizationAsset`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:118` (`is_runtime_cache_disabled`) | `crates/flexmesa-utils/src/lib.rs:61` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/data/fetch.py:161` (`resolve_cache_dir`) | `crates/flexmesa-utils/src/lib.rs:106` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/data/fetch.py:175` (`resolve_data_dir`) | `crates/flexmesa-utils/src/lib.rs:32` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/data/fetch.py:215` (`resolve_mesa_test_inputs_dir`) | `crates/flexmesa-utils/src/lib.rs:154` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/data/fetch.py:225` (`resolve_mesa_histories_dir`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:235` (`resolve_mesa_module_hex_dir`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:250` (`load_upstream_reference`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:278` (`load_bundles_manifest`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:318` (`load_embedded_ionization_asset`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:360` (`resolve_mesa_astero_tracks_dir`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:391` (`select_bundles`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:533` (`clean_runtime_cache`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:622` (`is_bundle_ready`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:637` (`verify_bundles`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:838` (`fetch_bundles`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:961` (`build_parser`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/fetch.py:1079` (`main`) | - | `pure-python mirror` | `python-only` | - |
| `src/flexmesa/data/loader.py:28` (`LoadedTableBundle`) | `crates/flexmesa-data/src/safetensors.rs:124` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/data/loader.py:72` (`encode_safetensors_bytes`) | `crates/flexmesa-data/src/safetensors.rs:200` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/data/loader.py:136` (`save_safetensors_bundle`) | `crates/flexmesa-data/src/safetensors.rs:354` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/data/loader.py:171` (`load_safetensors_bundle`) | `crates/flexmesa-data/src/safetensors.rs:393` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/data/manifest.py:23` (`TableArtifact`) | `crates/flexmesa-data/src/lib.rs:33` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/data/manifest.py:46` (`is_valid_sha256_hex`) | `crates/flexmesa-utils/src/sha256.rs:220` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/data/manifest.py:67` (`compute_sha256_bytes`) | `crates/flexmesa-utils/src/sha256.rs:227` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/data/manifest.py:86` (`compute_sha256_file`) | `crates/flexmesa-data/src/sha256.rs:18` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/data/manifest.py:104` (`verify_file_sha256`) | `crates/flexmesa-data/src/sha256.rs:35` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/data/manifest.py:179` (`parse_manifest_text`) | `crates/flexmesa-data/src/lib.rs:237` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/data/manifest.py:216` (`parse_manifest_file`) | `crates/flexmesa-data/src/lib.rs:326` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/data/manifest.py:228` (`load_default_manifest`) | `crates/flexmesa-data/src/lib.rs:340` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/eos/core.py:95` (`EosEvalOutput`) | `crates/flexmesa-eos/src/types.rs:114` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/eos/core.py:125` (`quintic_blend_jax`) | `crates/flexmesa-eos/src/blend.rs:168` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/eos/core.py:138` (`quintic_blend`) | `crates/flexmesa-eos/src/blend.rs:168` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/core.py:158` (`blend_eos_results`) | `crates/flexmesa-eos/src/blend.rs:189` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/core.py:237` (`is_contained`) | `crates/flexmesa-eos/src/blend.rs:99` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/core.py:282` (`min_distance_to_polygon_jax`) | `crates/flexmesa-eos/src/blend.rs:143` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/eos/core.py:299` (`min_distance_to_polygon`) | `crates/flexmesa-eos/src/blend.rs:143` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/core.py:335` (`compute_ion_offset_from_za`) | `crates/flexmesa-eos/src/ion_offset.rs:60` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/core.py:374` (`nuclide_arrays_jax`) | - | `JAX kernel` | `python-only` | - |
| `src/flexmesa/eos/core.py:393` (`pack_dt_jvp`) | - | `JAX kernel` | `python-only` | - |
| `src/flexmesa/eos/core.py:423` (`compute_ion_offset`) | `crates/flexmesa-eos/src/ion_offset.rs:93` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/fermi.py:232` (`dfermi_jax`) | `crates/flexmesa-eos/src/gauss_fermi.rs:225` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/eos/fermi.py:255` (`dfermi`) | `crates/flexmesa-eos/src/gauss_fermi.rs:225` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/helm.py:153` (`eval_helm_eos`) | `crates/flexmesa-eos/src/helm/convert.rs:150` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/helm.py:198` (`eval_helm_eos_zones`) | `crates/flexmesa-eos/src/helm/convert.rs:150` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/ideal.py:20` (`eval_ideal_eos`) | `crates/flexmesa-eos/src/ideal.rs:147` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/ideal.py:55` (`eval_ideal_eos_from_weights`) | `crates/flexmesa-eos/src/ideal.rs:27` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/ideal.py:111` (`eval_ideal_eos_zones`) | `crates/flexmesa-eos/src/ideal.rs:27` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/ideal.py:221` (`pack_free_energy_thermo_jax`) | `crates/flexmesa-eos/src/free_energy.rs:175` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/eos/pc.py:114` (`eval_pc_eos_jax`) | `crates/flexmesa-eos/src/pc/mod.rs:271` | `JAX kernel` | `mirror-current` | `tests/test_bench.py:208::test_wave3a_eos_pc_stages_agree` |
| `src/flexmesa/eos/pc.py:163` (`get_pc_alfa_jax`) | `crates/flexmesa-eos/src/pc/mod.rs:80` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/eos/pc.py:272` (`get_pc_alfa`) | `crates/flexmesa-eos/src/pc/mod.rs:80` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/pc.py:337` (`eval_pc_eos`) | `crates/flexmesa-eos/src/pc/mod.rs:271` | `PyO3 wrapper` | `mirror-current` | `tests/test_bench.py:208::test_wave3a_eos_pc_stages_agree`, `tests/test_eos_wave3a3.py:1013::test_pc_multispecies_buckets_rust_matches_frozen_jax`, `tests/test_eos_wave3a3.py:496::test_eval_pc_eos_and_zones_match_across_rust_and_jax` |
| `src/flexmesa/eos/pc.py:404` (`eval_pc_eos_zones`) | `crates/flexmesa-eos/src/pc/mod.rs:33` | `PyO3 wrapper` | `mirror-current` | `tests/test_eos_wave3a3.py:1013::test_pc_multispecies_buckets_rust_matches_frozen_jax`, `tests/test_eos_wave3a3.py:496::test_eval_pc_eos_and_zones_match_across_rust_and_jax` |
| `src/flexmesa/eos/skye.py:56` (`get_skye_alfa_jax`) | `crates/flexmesa-eos/src/skye/mod.rs:79` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/eos/skye.py:136` (`get_skye_alfa_simple_jax`) | `crates/flexmesa-eos/src/skye/mod.rs:132` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/eos/skye.py:171` (`get_skye_alfa`) | `crates/flexmesa-eos/src/skye/mod.rs:79` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/skye.py:205` (`get_skye_alfa_simple`) | `crates/flexmesa-eos/src/skye/mod.rs:132` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/skye.py:230` (`eval_host_dt_jvp`) | - | `JAX kernel` | `python-only` | - |
| `src/flexmesa/eos/skye.py:326` (`eval_skye_eos_jax`) | `crates/flexmesa-eos/src/skye/mod.rs:319` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/eos/skye.py:378` (`eval_skye_eos`) | `crates/flexmesa-eos/src/skye/mod.rs:319` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/skye.py:435` (`eval_skye_eos_zones`) | `crates/flexmesa-eos/src/helm/table.rs:100` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/solver.py:44` (`eos_dt_get_jax`) | `crates/flexmesa-eos/src/blend_dt.rs:545` | `JAX kernel` | `mirror-stale` | `tests/test_eos_wave3b1.py:816::test_helm_ideal_edge_skip_blend_rust_and_jax_agree` |
| `src/flexmesa/eos/solver.py:101` (`eos_dt_get`) | `crates/flexmesa-eos/src/blend_dt.rs:545` | `PyO3 wrapper` | `mirror-current` | `tests/test_eos_wave3b1.py:816::test_helm_ideal_edge_skip_blend_rust_and_jax_agree` |
| `src/flexmesa/eos/solver.py:238` (`eval_opal_scvh_eos`) | `crates/flexmesa-eos/src/tables_4d/mod.rs:464` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/solver.py:260` (`eval_free_eos`) | `crates/flexmesa-eos/src/tables_4d/mod.rs:487` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/eos/solver.py:282` (`eval_cms_eos`) | `crates/flexmesa-eos/src/cms/mod.rs:338` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/interp_1d/eval.py:102` (`interp_values_and_slopes`) | `crates/flexmesa-interp1d/src/eval.rs:177` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/interp_1d/eval.py:137` (`interp_values`) | `crates/flexmesa-interp1d/src/eval.rs:120` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/interp_1d/eval.py:166` (`interp_value_and_slope`) | `crates/flexmesa-interp1d/src/eval.rs:148` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/interp_1d/eval.py:199` (`interp_value`) | `crates/flexmesa-interp1d/src/eval.rs:97` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/interp_1d/eval.py:237` (`integrate_values`) | `crates/flexmesa-interp1d/src/eval.rs:257` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/interp_1d/mp.py:23` (`MpLimiter`) | `crates/flexmesa-interp1d/src/lib.rs:69` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/interp_1d/mp.py:36` (`minmod`) | `crates/flexmesa-interp1d/src/mp.rs:20` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/interp_1d/mp.py:49` (`minmod4`) | `crates/flexmesa-interp1d/src/mp.rs:28` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/interp_1d/mp.py:64` (`median`) | `crates/flexmesa-interp1d/src/mp.rs:36` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/interp_1d/mp.py:95` (`interp_m3`) | `crates/flexmesa-interp1d/src/lib.rs:63` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/interp_1d/mp.py:218` (`interp_m3a`) | `crates/flexmesa-interp1d/src/mp.rs:182` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/interp_1d/mp.py:242` (`interp_m3b`) | `crates/flexmesa-interp1d/src/mp.rs:195` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/interp_1d/mp.py:266` (`interp_m3q`) | `crates/flexmesa-interp1d/src/mp.rs:208` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/interp_1d/pm.py:23` (`validate_1d_grid`) | `crates/flexmesa-interp1d/src/lib.rs:145` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/interp_1d/pm.py:129` (`interp_pm`) | `crates/flexmesa-interp1d/src/pm.rs:57` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/interp_1d/pm.py:163` (`interp_pm_slopes_only`) | `crates/flexmesa-interp1d/src/pm.rs:70` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/interp_1d/pm.py:184` (`interp_4pt_pm`) | `crates/flexmesa-interp1d/src/pm.rs:144` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/interp_1d/pm.py:233` (`interp_pm_autodiff`) | `crates/flexmesa-interp1d/src/pm.rs:174` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/interp_1d/remesh.py:23` (`interpolate_vector`) | `crates/flexmesa-interp1d/src/remesh.rs:16` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/interp_1d/remesh.py:55` (`interpolate_vector_pm`) | `crates/flexmesa-interp1d/src/remesh.rs:35` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/interp_1d/remesh.py:97` (`interp_values_autodiff`) | `crates/flexmesa-interp1d/src/remesh.rs:127` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/interp_1d/remesh.py:168` (`interpolate_vector_pm_autodiff`) | `crates/flexmesa-interp1d/src/remesh.rs:194` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/interp_1d/remesh.py:192` (`interp_3_to_1`) | `crates/flexmesa-interp1d/src/remesh.rs:50` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/interp_1d/remesh.py:224` (`interp_3_to_2`) | `crates/flexmesa-interp1d/src/remesh.rs:72` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/interp_1d/remesh.py:259` (`interp_4_to_1`) | `crates/flexmesa-interp1d/src/remesh.rs:96` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/interp_2d/akima.py:152` (`interp_rgbi3p_db`) | `crates/flexmesa-interp2d/src/akima.rs:174` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/interp_2d/akima.py:196` (`interp_rgsf3p_db`) | `crates/flexmesa-interp2d/src/akima.rs:212` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/interp_2d/akima_pd.py:189` (`rgpd3p_db`) | `crates/flexmesa-interp2d/src/akima_pd.rs:174` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/interp_2d/bicub.py:60` (`interp_mkbicub_db`) | `crates/flexmesa-interp2d/src/bicub.rs:76` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/interp_2d/bicub_eval.py:22` (`BicubicEval`) | `crates/flexmesa-interp2d/src/lib.rs:102` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/interp_2d/bicub_eval.py:60` (`herm2xy_db`) | `crates/flexmesa-interp2d/src/bicub_eval.rs:47` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/interp_2d/bicub_eval.py:146` (`interp_evbicub_db`) | `crates/flexmesa-interp2d/src/bicub_eval.rs:179` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/interp_2d/bicub_eval.py:330` (`interp_evbicub_mask_db`) | `crates/flexmesa-interp2d/src/bicub_eval.rs:500` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/interp_2d/bipm.py:13` (`interp_mkbipm_db`) | `crates/flexmesa-interp2d/src/bipm.rs:29` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/interp_2d/bipm.py:62` (`interp_evbipm_db`) | `crates/flexmesa-interp2d/src/bipm.rs:86` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/interp_2d/renka790.py:192` (`Cshep2Interpolant`) | `crates/flexmesa-interp2d/src/renka790.rs:30` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/interp_2d/renka790.py:520` (`interp_cshep2_db`) | `crates/flexmesa-interp2d/src/renka790.rs:439` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/interp_2d/renka790.py:532` (`interp_cs2val_db`) | `crates/flexmesa-interp2d/src/renka790.rs:452` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/interp_2d/renka790.py:537` (`interp_cs2grd_db`) | `crates/flexmesa-interp2d/src/renka790.rs:458` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/interp_2d/renka790.py:544` (`cs2hes_db`) | `crates/flexmesa-interp2d/src/renka790.rs:308` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/interp_2d/spline1d.py:17` (`BicubicBc`) | `crates/flexmesa-interp2d/src/lib.rs:55` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/interp_2d/spline1d.py:43` (`splinck_db`) | `crates/flexmesa-interp2d/src/spline1d.rs:38` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/interp_2d/spline1d.py:77` (`reshape_grid_2d`) | - | `JAX kernel` | `python-only` | - |
| `src/flexmesa/interp_2d/spline1d.py:313` (`interp_mkspline_db`) | `crates/flexmesa-interp2d/src/spline1d.rs:76` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/ionization/fe56_he4.py:140` (`eval_charge_of_fe56_in_he4_jax`) | `crates/flexmesa-ionization/src/fe56_he4.rs:271` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/ionization/fe56_he4.py:166` (`eval_charge_of_fe56_in_he4`) | `crates/flexmesa-ionization/src/fe56_he4.rs:40` | `PyO3 wrapper` | `mirror-current` | `tests/test_bench.py:174::test_wave2_ion_fe56_stages_agree`, `tests/test_ionization.py:98::test_fe56_in_he4_golden_rust_vs_jax` |
| `src/flexmesa/ionization/fe56_he4.py:210` (`eval_charge_of_fe56_in_he4_zones`) | `crates/flexmesa-ionization/src/fe56_he4.rs:40` | `PyO3 wrapper` | `mirror-current` | `tests/test_ionization.py:171::test_fe56_in_he4_zones_rust_vs_jax_parity` |
| `src/flexmesa/ionization/tables.py:36` (`IonizationResult`) | `crates/flexmesa-ionization/src/ion_table.rs:336` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/ionization/tables.py:247` (`eval_ionization_jax`) | `crates/flexmesa-ionization/src/ion_table.rs:336` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/ionization/tables.py:310` (`eval_ionization`) | `crates/flexmesa-ionization/src/ion_table.rs:336` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/ionization/tables.py:367` (`eval_ionization_zones`) | `crates/flexmesa-ionization/src/ion_table_load.rs:69` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/ionization/typical_charge.py:43` (`ionization_potential`) | `crates/flexmesa-ionization/src/typical_charge.rs:25` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/ionization/typical_charge.py:66` (`chi_info_jax`) | `crates/flexmesa-ionization/src/typical_charge.rs:43` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/ionization/typical_charge.py:105` (`chi_effective_jax`) | `crates/flexmesa-ionization/src/typical_charge.rs:70` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/ionization/typical_charge.py:138` (`chi_info`) | `crates/flexmesa-ionization/src/typical_charge.rs:43` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/ionization/typical_charge.py:189` (`chi_effective`) | `crates/flexmesa-ionization/src/typical_charge.rs:70` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/ionization/typical_charge.py:233` (`eval_typical_charge`) | `crates/flexmesa-ionization/src/typical_charge.rs:135` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/kap/core.py:43` (`KapEvalOutput`) | `crates/flexmesa-kap/src/types.rs:254` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/kap/core.py:87` (`kap_get_results`) | `crates/flexmesa-kap/src/eval.rs:28` | `PyO3 wrapper` | `mirror-current` | `tests/unit/test_kap_wave3b2.py:335::test_rust_strict_vs_jax_backend_consistency_across_all_regimes`, `tests/unit/test_kap_wave3b2_jax.py:112::test_kapcn_knot_and_multi_zone_batch_consistency` |
| `src/flexmesa/kap/core.py:246` (`kap_get`) | `crates/flexmesa-kap/src/eval.rs:200` | `PyO3 wrapper` | `mirror-current` | `tests/unit/test_kap_wave3b2_jax.py:112::test_kapcn_knot_and_multi_zone_batch_consistency` |
| `src/flexmesa/kap/core.py:299` (`kap_get_zones`) | `crates/flexmesa-kap/src/eval.rs:28` | `PyO3 wrapper` | `mirror-current` | `tests/unit/test_kap_wave3b2_jax.py:112::test_kapcn_knot_and_multi_zone_batch_consistency` |
| `src/flexmesa/kap/jax_co_lowt.py:323` (`eval_co_kap_jax`) | `crates/flexmesa-kap/src/co.rs:23` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/kap/jax_co_lowt.py:531` (`eval_aesopus_jax`) | `crates/flexmesa-kap/src/low_t.rs:21` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/kap/jax_co_lowt.py:583` (`eval_kapcn_jax`) | `crates/flexmesa-kap/src/low_t.rs:255` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/kap/jax_interp.py:41` (`do_kap_interpolations_jax`) | `crates/flexmesa-kap/src/spline.rs:120` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/kap/jax_interp.py:120` (`interp_evbicub_jax`) | `crates/flexmesa-interp2d/src/bicub_eval.rs:378` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/kap/jax_interp.py:158` (`interp_pm_1d_jax`) | `crates/flexmesa-interp1d/src/pm.rs:57` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/kap/jax_interp.py:190` (`interp_pm_4pt_with_partials_jax`) | `crates/flexmesa-interp1d/src/pm.rs:144` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/kap/jax_interp.py:308` (`eval_fixed_metal_jax`) | `crates/flexmesa-kap/src/fixed.rs:20` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/kap/jax_kernels.py:45` (`compton_opacity_jax`) | `crates/flexmesa-kap/src/compton.rs:29` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/kap/jax_kernels.py:106` (`combine_rad_with_conduction_jax`) | `crates/flexmesa-kap/src/eval.rs:172` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/kap/jax_kernels.py:128` (`electron_conduction_potekhin_jax`) | `crates/flexmesa-kap/src/conduction.rs:168` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/kap/jax_kernels.py:205` (`electron_conduction_jax`) | `crates/flexmesa-kap/src/conduction.rs:340` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/kap/jax_kernels.py:368` (`kap_get_results_jax`) | `crates/flexmesa-kap/src/eval.rs:28` | `JAX kernel` | `mirror-stale` | `tests/unit/test_kap_wave3b2.py:335::test_rust_strict_vs_jax_backend_consistency_across_all_regimes` |
| `src/flexmesa/kap/jax_kernels.py:634` (`make_differentiable_kap`) | - | `JAX kernel` | `python-only` | - |
| `src/flexmesa/kap/submodels.py:20` (`kap_get_elect_cond_opacity`) | `crates/flexmesa-kap/src/conduction.rs:340` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/kap/submodels.py:55` (`kap_get_compton_opacity`) | `crates/flexmesa-kap/src/compton.rs:29` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/kap/submodels.py:113` (`kap_get_fixed_metal`) | `crates/flexmesa-kap/src/fixed.rs:20` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/kap/submodels.py:164` (`kap_get_co`) | `crates/flexmesa-kap/src/co.rs:23` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/kap/submodels.py:224` (`kap_get_aesopus`) | `crates/flexmesa-kap/src/low_t.rs:21` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/kap/submodels.py:269` (`kap_get_kapcn`) | `crates/flexmesa-kap/src/eval.rs:200` | `PyO3 wrapper` | `mirror-current` | `tests/unit/test_kap_wave3b2_jax.py:112::test_kapcn_knot_and_multi_zone_batch_consistency` |
| `src/flexmesa/kap/tables_co_lowt_jax.py:23` (`get_co_meta_jax`) | `crates/flexmesa-kap/src/loader.rs:230` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/kap/tables_co_lowt_jax.py:134` (`get_co_z_coeffs_jax`) | `crates/flexmesa-kap/src/loader.rs:214` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/kap/tables_co_lowt_jax.py:172` (`get_co_all_z_coeffs_jax`) | `crates/flexmesa-kap/src/loader.rs:230` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/kap/tables_co_lowt_jax.py:184` (`get_aesopus_meta_jax`) | `crates/flexmesa-kap/src/loader.rs:292` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/kap/tables_co_lowt_jax.py:190` (`get_aesopus_z_coeffs_jax`) | `crates/flexmesa-kap/src/loader.rs:278` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/kap/tables_co_lowt_jax.py:221` (`get_aesopus_all_z_coeffs_jax`) | `crates/flexmesa-kap/src/loader.rs:292` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/kap/tables_co_lowt_jax.py:235` (`get_kapcn_jax`) | `crates/flexmesa-kap/src/low_t.rs:255` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/kap/tables_jax.py:49` (`splinck`) | `crates/flexmesa-kap/src/spline.rs:218` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/kap/tables_jax.py:118` (`fit_bicubic_batch`) | `crates/flexmesa-kap/src/spline.rs:353` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/kap/tables_jax.py:174` (`reset_index_cache`) | - | `JAX kernel` | `python-only` | - |
| `src/flexmesa/kap/tables_jax.py:193` (`ensure_indexed`) | - | `JAX kernel` | `python-only` | - |
| `src/flexmesa/kap/tables_jax.py:500` (`get_kap_blend_bounds_jax`) | `crates/flexmesa-kap/src/loader.rs:430` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/kap/tables_jax.py:528` (`get_fixed_set_jax`) | `crates/flexmesa-kap/src/loader.rs:123` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/kap/tables_jax.py:618` (`get_potekhin_jax`) | `crates/flexmesa-kap/src/conduction.rs:40` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/intrinsics.py:27` (`as_f64`) | - | `JAX kernel` | `python-only` | - |
| `src/flexmesa/math/intrinsics.py:46` (`safe_sqrt`) | `crates/flexmesa-math/src/lib.rs:100` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/intrinsics.py:72` (`safe_log`) | `crates/flexmesa-math/src/lib.rs:108` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/intrinsics.py:94` (`safe_log10`) | `crates/flexmesa-math/src/lib.rs:132` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/math/intrinsics.py:115` (`log`) | `crates/flexmesa-math/src/lib.rs:156` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/intrinsics.py:127` (`log10`) | `crates/flexmesa-math/src/lib.rs:168` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/intrinsics.py:139` (`log1p`) | `crates/flexmesa-math/src/lib.rs:180` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/intrinsics.py:151` (`log2`) | `crates/flexmesa-math/src/lib.rs:192` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/intrinsics.py:163` (`exp`) | `crates/flexmesa-math/src/lib.rs:204` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/intrinsics.py:175` (`expm1`) | `crates/flexmesa-math/src/lib.rs:216` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/intrinsics.py:187` (`exp10`) | `crates/flexmesa-math/src/lib.rs:229` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/intrinsics.py:208` (`hypot`) | `crates/flexmesa-math/src/lib.rs:260` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/intrinsics.py:213` (`pow`) | `crates/flexmesa-math/src/lib.rs:308` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/intrinsics.py:226` (`powi`) | `crates/flexmesa-math/src/lib.rs:354` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/intrinsics.py:231` (`powm1`) | `crates/flexmesa-math/src/lib.rs:391` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/math/intrinsics.py:243` (`pow2`) | `crates/flexmesa-math/src/lib.rs:398` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/math/intrinsics.py:249` (`pow3`) | `crates/flexmesa-math/src/lib.rs:405` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/math/intrinsics.py:255` (`pow4`) | `crates/flexmesa-math/src/lib.rs:412` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/math/intrinsics.py:261` (`pow5`) | `crates/flexmesa-math/src/lib.rs:419` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/math/intrinsics.py:267` (`pow6`) | `crates/flexmesa-math/src/lib.rs:426` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/math/intrinsics.py:273` (`pow7`) | `crates/flexmesa-math/src/lib.rs:433` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/math/intrinsics.py:279` (`pow8`) | `crates/flexmesa-math/src/lib.rs:440` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/math/trig.py:18` (`cos`) | `crates/flexmesa-math/src/trig.rs:35` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:23` (`sin`) | `crates/flexmesa-math/src/trig.rs:47` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:28` (`tan`) | `crates/flexmesa-math/src/trig.rs:73` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:33` (`cospi`) | `crates/flexmesa-math/src/trig.rs:85` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:38` (`sinpi`) | `crates/flexmesa-math/src/trig.rs:97` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:43` (`tanpi`) | `crates/flexmesa-math/src/trig.rs:109` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:48` (`acos`) | `crates/flexmesa-math/src/trig.rs:121` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:53` (`asin`) | `crates/flexmesa-math/src/trig.rs:133` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:58` (`atan`) | `crates/flexmesa-math/src/trig.rs:145` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:63` (`acospi`) | `crates/flexmesa-math/src/trig.rs:230` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:68` (`asinpi`) | `crates/flexmesa-math/src/trig.rs:242` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:73` (`atanpi`) | `crates/flexmesa-math/src/trig.rs:254` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:78` (`cosh`) | `crates/flexmesa-math/src/trig.rs:266` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:83` (`sinh`) | `crates/flexmesa-math/src/trig.rs:278` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:88` (`tanh`) | `crates/flexmesa-math/src/trig.rs:295` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:93` (`acosh`) | `crates/flexmesa-math/src/trig.rs:307` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:98` (`asinh`) | `crates/flexmesa-math/src/trig.rs:321` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:103` (`atanh`) | `crates/flexmesa-math/src/trig.rs:333` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:108` (`sgn`) | `crates/flexmesa-math/src/io.rs:12` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:123` (`heaviside`) | `crates/flexmesa-math/src/io.rs:28` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:138` (`str_to_double`) | `crates/flexmesa-math/src/io.rs:46` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/math/trig.py:155` (`str_to_vector`) | `crates/flexmesa-math/src/io.rs:109` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/mtx/band.py:20` (`dense_to_band`) | `crates/flexmesa-mtx/src/band.rs:37` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/band.py:59` (`band_to_dense`) | `crates/flexmesa-mtx/src/band.rs:77` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/band.py:93` (`band_multiply_xa`) | `crates/flexmesa-mtx/src/band.rs:117` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/band.py:135` (`dgbtrf`) | `crates/flexmesa-mtx/src/band.rs:167` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/mtx/band.py:205` (`dgbtrs`) | `crates/flexmesa-mtx/src/band.rs:412` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/mtx/band.py:263` (`solve_banded`) | `crates/flexmesa-mtx/src/band.rs:430` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/band.py:289` (`rcond_banded`) | `crates/flexmesa-mtx/src/band.rs:460` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/mtx/block_tridiag.py:75` (`block_dble_mv`) | `crates/flexmesa-mtx/src/block_tridiag.rs:240` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/mtx/block_tridiag.py:109` (`block_multiply_xa`) | `crates/flexmesa-mtx/src/block_tridiag.rs:289` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/mtx/block_tridiag.py:163` (`block_thomas_factor`) | `crates/flexmesa-mtx/src/block_tridiag.rs:383` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/mtx/block_tridiag.py:202` (`block_thomas_solve`) | `crates/flexmesa-mtx/src/block_tridiag.rs:427` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/mtx/block_tridiag.py:244` (`solve_block_tridiagonal`) | `crates/flexmesa-mtx/src/bcyclic.rs:634` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/mtx/block_tridiag.py:340` (`solve_block_tridiagonal_jax`) | `crates/flexmesa-mtx/src/block_tridiag.rs:472` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/mtx/dense.py:31` (`dgetrf`) | `crates/flexmesa-mtx/src/dense.rs:35` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/mtx/dense.py:67` (`dgetrs`) | `crates/flexmesa-mtx/src/dense.rs:198` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/dense.py:142` (`solve_dense`) | `crates/flexmesa-mtx/src/dense.rs:283` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/dense.py:164` (`multiply_xa`) | `crates/flexmesa-mtx/src/dense_ops.rs:22` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/dense.py:196` (`lu_factored_square_mv`) | `crates/flexmesa-mtx/src/dense_ops.rs:71` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/mtx/dense.py:221` (`lu_factored_square_mm`) | `crates/flexmesa-mtx/src/dense_ops.rs:122` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/mtx/dense.py:255` (`rcond_dense`) | `crates/flexmesa-mtx/src/dense.rs:402` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/mtx/dense.py:295` (`solve_dense_jax`) | `crates/flexmesa-mtx/src/dense.rs:283` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/mtx/pcr.py:36` (`validate_jax_blocks`) | - | `JAX kernel` | `python-only` | - |
| `src/flexmesa/mtx/pcr.py:77` (`solve_block_tridiag_pcr`) | `crates/flexmesa-mtx/src/block_tridiag.rs:472` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/mtx/pcr.py:194` (`solve_block_tridiag_rust_ffi`) | `crates/flexmesa-mtx/src/bcyclic.rs:634` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/mtx/sparse.py:16` (`SparseFormat`) | `crates/flexmesa-mtx/src/lib.rs:107` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/sparse.py:92` (`dense_to_row_sparse`) | `crates/flexmesa-mtx/src/sparse.rs:130` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/sparse.py:117` (`dense_to_row_0_based`) | `crates/flexmesa-mtx/src/sparse.rs:168` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/sparse.py:142` (`row_sparse_to_dense`) | `crates/flexmesa-mtx/src/sparse.rs:206` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/sparse.py:167` (`row_sparse_0_based_to_dense`) | `crates/flexmesa-mtx/src/sparse.rs:230` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/sparse.py:204` (`dense_to_col_sparse`) | `crates/flexmesa-mtx/src/sparse.rs:255` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/sparse.py:228` (`dense_to_col_0_based`) | `crates/flexmesa-mtx/src/sparse.rs:293` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/sparse.py:252` (`column_sparse_to_dense`) | `crates/flexmesa-mtx/src/sparse.rs:363` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/sparse.py:277` (`col_sparse_0_based_to_dense`) | `crates/flexmesa-mtx/src/sparse.rs:380` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/mtx/sparse.py:363` (`find_loc_in_sparse`) | `crates/flexmesa-mtx/src/sparse_band.rs:358` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/net/_core.py:17` (`NetInfo`) | `crates/flexmesa-net/src/types.rs:163` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/net/_core.py:38` (`NetEvalOutput`) | `crates/flexmesa-net/src/types.rs:253` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/net/_core.py:80` (`Burn1ZoneOutput`) | `crates/flexmesa-net/src/burn.rs:67` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/net/_core.py:109` (`BurnConstDensityOutput`) | `crates/flexmesa-net/src/burn_const_density.rs:56` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/net/_core.py:140` (`BurnConstPOutput`) | `crates/flexmesa-net/src/burn_const_p.rs:65` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/net/_core.py:170` (`list_networks`) | `crates/flexmesa-net/src/parser.rs:22` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/net/_core.py:175` (`get_net_info`) | `crates/flexmesa-net/src/lib.rs:108` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/net/_core.py:195` (`eval_net`) | `crates/flexmesa-net/src/burn.rs:171` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/net/_core.py:284` (`burn_1_zone`) | `crates/flexmesa-net/src/burn.rs:171` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/net/_core.py:368` (`burn_1_zone_const_density`) | `crates/flexmesa-net/src/burn_const_density.rs:88` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/net/_core.py:442` (`burn_1_zone_const_p`) | `crates/flexmesa-net/src/burn_const_p.rs:96` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/net/_jax.py:19` (`eval_net_jax`) | `crates/flexmesa-net/src/burn.rs:171` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/neu/brem.py:56` (`brem_neu_weak_degen`) | `crates/flexmesa-neu/src/brem.rs:43` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/neu/brem.py:209` (`brem_neu_liquid_metal`) | `crates/flexmesa-neu/src/brem.rs:207` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/neu/brem.py:339` (`brem_neu`) | `crates/flexmesa-neu/src/brem.rs:350` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/neu/driver.py:34` (`NeuChannel`) | `crates/flexmesa-neu/src/lib.rs:88` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/neu/driver.py:57` (`NeuFlags`) | `crates/flexmesa-neu/src/lib.rs:110` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/neu/driver.py:107` (`NeuResult`) | `crates/flexmesa-neu/src/lib.rs:231` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/neu/driver.py:312` (`neu_get`) | `crates/flexmesa-neu/src/lib.rs:274` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/neu/driver.py:398` (`neu_pair_plas_jax`) | `crates/flexmesa-neu/src/pair_plas.rs:48` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/neu/inputs.py:79` (`NeuRate`) | `crates/flexmesa-neu/src/lib.rs:170` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/neu/inputs.py:150` (`NeuInputs`) | `crates/flexmesa-neu/src/inputs.rs:70` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/neu/pair_plas.py:26` (`pair_neu`) | `crates/flexmesa-neu/src/pair_plas.rs:48` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/neu/pair_plas.py:146` (`plas_neu`) | `crates/flexmesa-neu/src/pair_plas.rs:175` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/neu/phot.py:33` (`PhotBandCoeffs`) | `crates/flexmesa-neu/src/phot.rs:41` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/neu/phot.py:157` (`phot_neu`) | `crates/flexmesa-neu/src/phot.rs:161` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/neu/reco.py:72` (`ifermi12`) | `crates/flexmesa-neu/src/reco.rs:48` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/neu/reco.py:96` (`zfermim12`) | `crates/flexmesa-neu/src/reco.rs:107` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/neu/reco.py:177` (`reco_neu`) | `crates/flexmesa-neu/src/reco.rs:196` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/bobyqa.py:21` (`BobyqaResult`) | `crates/flexmesa-num/src/bobyqa.rs:13` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/bobyqa.py:35` (`bobyqa`) | `crates/flexmesa-num/src/bobyqa.rs:48` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/brent.py:17` (`brent_safe_zero`) | `crates/flexmesa-num/src/brent.rs:42` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/brent.py:137` (`eval_brent_safe_zero`) | `crates/flexmesa-num/src/brent.rs:184` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/brent.py:239` (`brent_local_min`) | `crates/flexmesa-num/src/brent.rs:281` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/brent.py:348` (`local_min`) | `crates/flexmesa-num/src/brent.rs:403` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/brent.py:372` (`brent_global_min`) | `crates/flexmesa-num/src/brent.rs:439` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/brent.py:537` (`glomin`) | `crates/flexmesa-num/src/brent.rs:625` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/newton.py:29` (`NewtonConfig`) | `crates/flexmesa-num/src/newton.rs:25` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/newton.py:48` (`NewtonResult`) | `crates/flexmesa-num/src/newton.rs:52` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/newton.py:168` (`newton_solve_dense`) | `crates/flexmesa-num/src/newton.rs:184` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/newton.py:202` (`newton_solve_banded`) | `crates/flexmesa-num/src/newton.rs:209` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/newton.py:239` (`newton_solve_block_tridiagonal`) | `crates/flexmesa-num/src/newton.rs:239` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/newton.py:284` (`newton_solve_jax`) | `crates/flexmesa-num/src/newton.rs:184` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/num/newuoa.py:19` (`NewuoaResult`) | `crates/flexmesa-num/src/newuoa.rs:14` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/newuoa.py:33` (`newuoa`) | `crates/flexmesa-num/src/newuoa.rs:45` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/newuoa.py:202` (`fit_min_frobenius_quad`) | `crates/flexmesa-num/src/newuoa.rs:293` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/quadrature.py:16` (`dfridr`) | `crates/flexmesa-num/src/quadrature.rs:24` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/quadrature.py:63` (`integrate`) | `crates/flexmesa-num/src/quadrature.rs:82` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/rk.py:19` (`rk2`) | `crates/flexmesa-num/src/rk.rs:61` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/rk.py:43` (`rk4`) | `crates/flexmesa-num/src/rk.rs:86` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/rk.py:76` (`DopriResult`) | `crates/flexmesa-num/src/rk_tableau.rs:71` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/rk.py:124` (`dopri5`) | `crates/flexmesa-num/src/rk.rs:127` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/rk.py:306` (`dop853`) | `crates/flexmesa-num/src/rk.rs:422` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/root.py:25` (`look_for_brackets`) | `crates/flexmesa-num/src/root.rs:42` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/root.py:97` (`safe_root_with_brackets`) | `crates/flexmesa-num/src/root.rs:114` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/root.py:138` (`safe_root_with_guess`) | `crates/flexmesa-num/src/root.rs:162` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/root.py:232` (`safe_root_without_brackets`) | `crates/flexmesa-num/src/root.rs:239` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/root.py:277` (`safe_root_jax`) | `crates/flexmesa-num/src/root.rs:283` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/num/rosenbrock.py:24` (`RosenbrockVariant`) | `crates/flexmesa-num/src/rosenbrock_coeffs.rs:19` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/rosenbrock.py:37` (`RosenbrockResult`) | `crates/flexmesa-num/src/rosenbrock.rs:74` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/rosenbrock.py:409` (`rosenbrock_dense`) | `crates/flexmesa-num/src/rosenbrock.rs:96` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/rosenbrock.py:460` (`rosenbrock_banded`) | `crates/flexmesa-num/src/rosenbrock.rs:273` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/rosenbrock.py:517` (`rosenbrock_dae_diag`) | `crates/flexmesa-num/src/rosenbrock.rs:211` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/search.py:16` (`two_sum`) | `crates/flexmesa-num/src/search.rs:15` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/search.py:32` (`neumaier_sum`) | `crates/flexmesa-num/src/search.rs:27` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/search.py:55` (`binary_search`) | `crates/flexmesa-num/src/search.rs:52` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/search.py:121` (`binary_search_1based`) | `crates/flexmesa-num/src/search.rs:122` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/search.py:137` (`qsort_index`) | `crates/flexmesa-num/src/search.rs:130` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/search.py:151` (`linear_interp`) | `crates/flexmesa-num/src/search.rs:138` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/search.py:175` (`find0`) | `crates/flexmesa-num/src/search.rs:152` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/search.py:225` (`find0_quadratic`) | `crates/flexmesa-num/src/search.rs:201` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/search.py:281` (`find_max_quadratic`) | `crates/flexmesa-num/src/search.rs:277` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/search.py:354` (`two_piece_linear_coeffs`) | `crates/flexmesa-num/src/search.rs:329` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/simplex.py:17` (`r8_uniform_01`) | `crates/flexmesa-num/src/simplex.rs:38` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/simplex.py:39` (`i4_uniform`) | `crates/flexmesa-num/src/simplex.rs:67` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/simplex.py:58` (`perm_uniform`) | `crates/flexmesa-num/src/simplex.rs:81` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/simplex.py:77` (`SimplexConfig`) | `crates/flexmesa-num/src/simplex.rs:93` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/num/simplex.py:109` (`SimplexResult`) | `crates/flexmesa-num/src/simplex.rs:142` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/num/simplex.py:127` (`nm_simplex`) | `crates/flexmesa-num/src/simplex.rs:164` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/rates/core.py:29` (`ReactionInfo`) | `crates/flexmesa-rates/src/types.rs:132` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/rates/core.py:62` (`reaction_id_by_name`) | `crates/flexmesa-rates/src/loader.rs:454` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/core.py:74` (`reaction_name_by_id`) | `crates/flexmesa-rates/src/loader.rs:454` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/core.py:86` (`reaction_info`) | `crates/flexmesa-rates/src/loader.rs:454` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/core.py:123` (`eval_reaclib_handle`) | `crates/flexmesa-rates/src/loader.rs:454` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/core.py:174` (`set_raw_rate`) | `crates/flexmesa-rates/src/raw_rates.rs:376` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/core.py:195` (`set_raw_rates`) | `crates/flexmesa-rates/src/raw_rates.rs:391` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/core.py:216` (`set_raw_rates_zones`) | `crates/flexmesa-rates/src/raw_rates.rs:391` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/core.py:242` (`eval_fl_epsnuc_3alf`) | `crates/flexmesa-rates/src/raw_rates.rs:588` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/core.py:282` (`eval_rate_table`) | `crates/flexmesa-rates/src/loader.rs:454` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/jax_kernels.py:60` (`eval_reaclib_jax`) | `crates/flexmesa-rates/src/raw_rates.rs:427` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/rates/jax_kernels.py:111` (`eval_reaclib_terms_jax`) | `crates/flexmesa-rates/src/tfactors.rs:311` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/rates/jax_kernels.py:213` (`eval_fl_epsnuc_3alf_jax`) | `crates/flexmesa-rates/src/raw_rates.rs:588` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/rates/jax_screening.py:17` (`screen_salpeter_jax`) | `crates/flexmesa-rates/src/screen.rs:52` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/rates/jax_screening.py:65` (`screen_extended_jax`) | `crates/flexmesa-rates/src/screen.rs:80` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/rates/jax_screening.py:245` (`screen_chugunov_jax`) | `crates/flexmesa-rates/src/screen_chugunov.rs:17` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/rates/screening.py:25` (`ScreenEval`) | `crates/flexmesa-rates/src/types.rs:203` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/rates/screening.py:53` (`screen_pair`) | `crates/flexmesa-rates/src/screen.rs:16` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/screening.py:141` (`screen_pair_zones`) | `crates/flexmesa-rates/src/screen.rs:16` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/weak.py:19` (`WeakRateEval`) | `crates/flexmesa-rates/src/types.rs:225` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/rates/weak.py:66` (`eval_coulomb`) | `crates/flexmesa-rates/src/ecapture.rs:32` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/weak.py:109` (`eval_weak_pair`) | `crates/flexmesa-rates/src/types.rs:225` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/weak.py:135` (`eval_ecapture_pair`) | `crates/flexmesa-rates/src/ecapture.rs:574` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/weak.py:180` (`weak_info_by_names`) | `crates/flexmesa-rates/src/loader.rs:454` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/rates/weak.py:193` (`eval_weak_table_index`) | `crates/flexmesa-rates/src/loader.rs:454` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/schema/defaults.py:27` (`ControlType`) | `crates/flexmesa-schema/src/lib.rs:34` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/schema/defaults.py:37` (`ControlDefault`) | `crates/flexmesa-schema/src/lib.rs:73` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/schema/defaults.py:67` (`ColumnEntry`) | `crates/flexmesa-schema/src/lib.rs:94` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/schema/defaults.py:84` (`find_unquoted_char`) | `crates/flexmesa-schema/src/lib.rs:185` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/schema/defaults.py:110` (`strip_fortran_comment`) | `crates/flexmesa-schema/src/lib.rs:214` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/schema/defaults.py:125` (`parse_control_value`) | `crates/flexmesa-schema/src/lib.rs:236` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/schema/defaults.py:153` (`parse_assignment_clause`) | `crates/flexmesa-schema/src/lib.rs:306` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/schema/defaults.py:178` (`parse_defaults_text`) | `crates/flexmesa-schema/src/lib.rs:413` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/schema/defaults.py:232` (`parse_defaults_file`) | `crates/flexmesa-schema/src/lib.rs:471` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/schema/defaults.py:245` (`parse_columns_text`) | `crates/flexmesa-schema/src/lib.rs:479` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/schema/defaults.py:277` (`parse_columns_file`) | `crates/flexmesa-schema/src/lib.rs:509` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/schema/namelist.py:22` (`NamelistAssignment`) | `crates/flexmesa-schema/src/namelist.rs:20` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/schema/namelist.py:42` (`NamelistBlock`) | `crates/flexmesa-schema/src/namelist.rs:33` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/schema/namelist.py:55` (`parse_namelist_text`) | `crates/flexmesa-schema/src/namelist.rs:138` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/schema/namelist.py:106` (`parse_namelist_file`) | `crates/flexmesa-schema/src/namelist.rs:208` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/schema/namelist.py:179` (`validate_namelist_block`) | `crates/flexmesa-schema/src/namelist.rs:231` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/star/e2e.py:16` (`copy_cleaned_case`) | `crates/flexmesa-star/src/inlist_inventory.rs:1892` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/star/e2e.py:27` (`parse_inlist_bundle`) | `crates/flexmesa-star/src/inlist.rs:359` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/star/e2e.py:48` (`parse_rn_stages`) | `crates/flexmesa-star/src/rn_parser.rs:189` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/star/e2e.py:64` (`run_test_suite_case`) | `crates/flexmesa-star/src/inlist.rs:359` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/star/e2e.py:100` (`eval_linear_oscillation_modes`) | `crates/flexmesa-star/src/pulse_modes.rs:328` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/star/e2e.py:123` (`eval_target_mode_frequency_and_growth`) | `crates/flexmesa-star/src/pulse_modes.rs:476` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/star/e2e.py:141` (`eval_astero_scaling_dnu_numax_uhz`) | `crates/flexmesa-star/src/pulse_modes.rs:548` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/star/e2e.py:157` (`eval_rsp_lina_modes`) | `crates/flexmesa-star/src/rsp_build.rs:206` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/star/e2e.py:191` (`build_rsp_envelope_model`) | `crates/flexmesa-star/src/rsp_build.rs:333` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/star/jax_diffusion_rot.py:40` (`eval_rotation_fp_ft_jax`) | `crates/flexmesa-star/src/rotation.rs:62` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/jax_diffusion_rot.py:77` (`eval_omega_crit_jax`) | `crates/flexmesa-star/src/rotation.rs:87` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/jax_diffusion_rot.py:115` (`eval_rotational_mixing_zone_jax`) | `crates/flexmesa-star/src/rotation.rs:252` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/jax_diffusion_rot.py:359` (`eval_caplan_ocp_dstar_jax`) | `crates/flexmesa-star/src/diffusion.rs:2246` | `JAX engine twin (frozen)` | `mirror-stale` | `tests/test_star_wave5_1.py:239::test_caplan_ocp_dstar_jax_matches_rust`, `tests/test_wave5_bench.py:93::test_wave5_diff_rot_rust_matches_frozen_jax` |
| `src/flexmesa/star/jax_diffusion_rot.py:384` (`eval_burgers_velocities_jax`) | `crates/flexmesa-star/src/diffusion.rs:2326` | `JAX engine twin (frozen)` | `mirror-stale` | `tests/test_star_wave5_1.py:264::test_burgers_velocities_jax_match_rust` |
| `src/flexmesa/star/jax_diffusion_rot.py:524` (`eval_phase_separation_shift_jax`) | `crates/flexmesa-star/src/phase_separation.rs:85` | `JAX engine twin (frozen)` | `mirror-stale` | `tests/test_star_wave5_1.py:305::test_phase_separation_shift_jax_matches_rust` |
| `src/flexmesa/star/jax_evolve.py:19` (`eval_mesh_gval_jax`) | `crates/flexmesa-star/src/mesh_adjust.rs:333` | `JAX engine twin (frozen)` | `mirror-stale` | `tests/test_star_wave4_4.py:40::test_adaptive_remeshing_and_mesh_gval_jax_consistency` |
| `src/flexmesa/star/jax_evolve.py:60` (`eval_varcontrol_jax`) | `crates/flexmesa-star/src/timestep.rs:252` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/jax_evolve.py:105` (`eval_h211_timestep_jax`) | `crates/flexmesa-star/src/timestep.rs:306` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/jax_hydro_pulse.py:41` (`eval_hllc_face_jax`) | `crates/flexmesa-star/src/riemann.rs:84` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/star/jax_hydro_pulse.py:101` (`eval_hllc_surface_face_jax`) | `crates/flexmesa-star/src/riemann.rs:148` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/star/jax_hydro_pulse.py:148` (`eval_riemann_momentum_residual_jax`) | `crates/flexmesa-star/src/riemann.rs:201` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/star/jax_hydro_pulse.py:188` (`eval_dziembowski_variables_jax`) | `crates/flexmesa-star/src/pulse_utils.rs:352` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/jax_hydro_pulse.py:228` (`eval_sobolev_feii_5169_tau_jax`) | `crates/flexmesa-star/src/stella.rs:202` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/jax_hydro_pulse.py:261` (`solve_gyre_mesa_modes`) | `crates/flexmesa-star/src/pulse_modes.rs:204` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/star/jax_hydro_pulse.py:294` (`solve_adipls_fgong_modes`) | `crates/flexmesa-star/src/pulse_modes.rs:228` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/star/jax_hydro_pulse.py:324` (`run_stella_from_hyd_abn`) | `crates/flexmesa-star/src/stella.rs:532` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/star/jax_kernels.py:25` (`compute_staggered_grid_jax`) | `crates/flexmesa-star/src/mesh.rs:259` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/star/jax_kernels.py:87` (`face_weights_jax`) | `crates/flexmesa-star/src/mesh.rs:290` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/star/jax_kernels.py:108` (`interp_cell_to_face_jax`) | `crates/flexmesa-star/src/mesh.rs:329` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/jax_kernels.py:127` (`eval_optical_depth_jax`) | `crates/flexmesa-star/src/micro.rs:1816` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/star/jax_kernels.py:268` (`eval_face_mlt_and_brunt_jax`) | `crates/flexmesa-star/src/hydro_vars.rs:45` | `JAX engine twin (frozen)` | `mirror-stale` | `tests/test_star_wave4_1.py:102::test_star_jax_face_gradr_and_gradt_match_rust`, `tests/test_wave4_bench.py:163::test_wave4_star_micro_vars_rust_matches_frozen_jax` |
| `src/flexmesa/star/jax_kernels.py:347` (`eval_hydro_residuals_jax`) | `crates/flexmesa-star/src/hydro_eqns.rs:973` | `JAX engine twin (frozen)` | `mirror-current` | `tests/test_star_wave4_2.py:46::test_eval_equations_rust_jax_residual_consistency`, `tests/test_wave4_bench.py:178::test_wave4_star_hydro_eqns_rust_matches_frozen_jax` |
| `src/flexmesa/star/jax_kernels.py:536` (`solve_henyey_linear_jax`) | `crates/flexmesa-star/src/solver.rs:534` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/jax_rsp.py:37` (`rsp2_flux_limiter_jax`) | `crates/flexmesa-star/src/rsp2.rs:138` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/star/jax_rsp.py:64` (`compute_rsp2_uq_face_jax`) | `crates/flexmesa-star/src/rsp2.rs:161` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/star/jax_rsp.py:126` (`eval_rsp2_zone_jax`) | `crates/flexmesa-star/src/rsp2.rs:208` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/jax_rsp.py:226` (`eval_rsp2_face_jax`) | `crates/flexmesa-star/src/rsp2.rs:287` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/jax_rsp.py:317` (`solve_stationary_rsp2_w_jax`) | `crates/flexmesa-star/src/rsp2.rs:361` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/star/jax_rsp.py:374` (`eval_rsp2_turbulent_eqn_residual_jax`) | `crates/flexmesa-star/src/rsp_eqns_cell.rs:228` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/jax_rsp.py:424` (`eval_rsp_hyd_fluxes_jax`) | `crates/flexmesa-star/src/rsp_hyd_flux.rs:67` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/jax_winds_mix.py:25` (`eval_convection_sigmas_jax`) | `crates/flexmesa-star/src/mix_info.rs:1351` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/jax_winds_mix.py:70` (`eval_dxdt_mix_jax`) | `crates/flexmesa-star/src/mix_info.rs:1541` | `JAX engine twin (frozen)` | `mirror-current` | - |
| `src/flexmesa/star/jax_winds_mix.py:100` (`eval_winds_jax`) | `crates/flexmesa-star/src/winds_driver.rs:203` | `JAX engine twin (frozen)` | `mirror-stale` | - |
| `src/flexmesa/star/model.py:30` (`StarModel`) | `crates/flexmesa-star/src/model.rs:62` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/star/model.py:696` (`yrs_for_init_timestep`) | `crates/flexmesa-star/src/zams.rs:395` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/star/model.py:701` (`interp_q`) | `crates/flexmesa-star/src/zams.rs:484` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/turb/mlt.py:27` (`set_mlt`) | `crates/flexmesa-turb/src/mlt.rs:148` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/turb/mlt.py:247` (`set_mlt_jax`) | `crates/flexmesa-turb/src/mlt.rs:148` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/turb/semiconvection.py:25` (`SemiconvectionParams`) | `crates/flexmesa-turb/src/semiconvection.rs:55` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/semiconvection.py:47` (`SemiconvectionInputs`) | `crates/flexmesa-turb/src/semiconvection.rs:73` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/semiconvection.py:79` (`set_semiconvection`) | `crates/flexmesa-turb/src/semiconvection.rs:107` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/turb/semiconvection.py:168` (`set_semiconvection_jax`) | `crates/flexmesa-turb/src/semiconvection.rs:107` | `JAX kernel` | `mirror-stale` | - |
| `src/flexmesa/turb/tdc.py:27` (`TdcParams`) | `crates/flexmesa-turb/src/tdc.rs:34` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/tdc.py:69` (`TdcInputs`) | `crates/flexmesa-turb/src/tdc.rs:71` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/tdc.py:111` (`TdcOutput`) | `crates/flexmesa-turb/src/tdc.rs:112` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/tdc.py:123` (`set_tdc`) | `crates/flexmesa-turb/src/tdc.rs:131` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/turb/tdc_dual.py:30` (`AutoDiffRealTdc`) | `crates/flexmesa-turb/src/tdc_dual.rs:34` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/tdc_search.py:30` (`q_bisection_search`) | `crates/flexmesa-turb/src/tdc_search.rs:41` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/turb/tdc_search.py:77` (`dq_dz_bisection_search`) | `crates/flexmesa-turb/src/tdc_search.rs:97` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/turb/tdc_search.py:131` (`af_bisection_search`) | `crates/flexmesa-turb/src/tdc_search.rs:167` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/tdc_search.py:176` (`bracket_plus_newton_search`) | `crates/flexmesa-turb/src/tdc_search.rs:223` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/turb/tdc_support.py:29` (`TdcInfo`) | `crates/flexmesa-turb/src/tdc_support.rs:34` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/tdc_support.py:222` (`set_y`) | `crates/flexmesa-turb/src/tdc_support.rs:211` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/tdc_support.py:243` (`safe_tanh`) | `crates/flexmesa-turb/src/tdc_support.rs:223` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/tdc_support.py:268` (`flux_limiter_function`) | `crates/flexmesa-turb/src/tdc_support.rs:236` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/turb/tdc_support.py:298` (`eval_xis`) | `crates/flexmesa-turb/src/tdc_support.rs:255` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/turb/tdc_support.py:361` (`eval_af`) | `crates/flexmesa-turb/src/tdc_support.rs:319` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/turb/tdc_support.py:419` (`compute_q`) | `crates/flexmesa-turb/src/tdc_support.rs:363` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/turb/thermohaline.py:25` (`ThermohalineParams`) | `crates/flexmesa-turb/src/thermohaline.rs:107` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/thermohaline.py:49` (`ThermohalineInputs`) | `crates/flexmesa-turb/src/thermohaline.rs:122` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/thermohaline.py:72` (`get_d_thermohaline`) | `crates/flexmesa-turb/src/thermohaline.rs:143` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/thermohaline.py:136` (`set_thermohaline`) | `crates/flexmesa-turb/src/thermohaline.rs:194` | `pure-python mirror` | `mirror-stale` | - |
| `src/flexmesa/turb/types.py:37` (`MltOption`) | `crates/flexmesa-turb/src/mlt.rs:36` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:73` (`SemiconvectionOption`) | `crates/flexmesa-turb/src/semiconvection.rs:31` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:102` (`ThermohalineOption`) | `crates/flexmesa-turb/src/thermohaline.rs:44` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:134` (`DrivingSpecies`) | `crates/flexmesa-turb/src/thermohaline.rs:71` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:162` (`TurbOutput`) | `crates/flexmesa-turb/src/lib.rs:120` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:184` (`MltParams`) | `crates/flexmesa-turb/src/mlt.rs:69` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:211` (`MltInputs`) | `crates/flexmesa-turb/src/mlt.rs:100` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:243` (`val_of`) | `crates/flexmesa-turb/src/lib.rs:164` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:274` (`const_like`) | `crates/flexmesa-turb/src/lib.rs:160` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:305` (`with_val_like`) | `crates/flexmesa-turb/src/lib.rs:170` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:334` (`s_abs`) | `crates/flexmesa-turb/src/lib.rs:189` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:361` (`s_sqrt`) | `crates/flexmesa-turb/src/lib.rs:193` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:388` (`s_exp`) | `crates/flexmesa-turb/src/lib.rs:197` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:415` (`s_log`) | `crates/flexmesa-turb/src/lib.rs:201` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:444` (`s_powf`) | `crates/flexmesa-turb/src/lib.rs:205` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:471` (`s_pow2`) | `crates/flexmesa-turb/src/lib.rs:209` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:498` (`s_pow3`) | `crates/flexmesa-turb/src/lib.rs:213` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:525` (`s_pow4`) | `crates/flexmesa-turb/src/lib.rs:217` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:553` (`s_tanh`) | `crates/flexmesa-turb/src/lib.rs:221` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:580` (`s_tan`) | `crates/flexmesa-turb/src/lib.rs:225` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/turb/types.py:607` (`s_atan`) | `crates/flexmesa-turb/src/lib.rs:229` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/utils/__init__.py:22` (`is_bad`) | `crates/flexmesa-utils/src/lib.rs:241` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/utils/__init__.py:43` (`is_bad_array`) | `crates/flexmesa-utils/src/lib.rs:247` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/utils/__init__.py:57` (`replace_bad_nums`) | `crates/flexmesa-utils/src/lib.rs:264` | `JAX kernel` | `mirror-current` | - |
| `src/flexmesa/utils/__init__.py:79` (`remove_underbars`) | `crates/flexmesa-utils/src/lib.rs:273` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/utils/__init__.py:93` (`split_line`) | `crates/flexmesa-utils/src/lib.rs:292` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/utils/__init__.py:114` (`IntegerDict`) | `crates/flexmesa-utils/src/lib.rs:351` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/utils/__init__.py:185` (`IntegerIdict`) | `crates/flexmesa-utils/src/lib.rs:432` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/utils/__init__.py:255` (`TokenKind`) | `crates/flexmesa-utils/src/lib.rs:492` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/utils/__init__.py:267` (`Token`) | `crates/flexmesa-utils/src/lib.rs:492` | `pure-python mirror` | `mirror-current` | - |
| `src/flexmesa/utils/__init__.py:282` (`tokenize`) | `crates/flexmesa-utils/src/lib.rs:696` | `PyO3 wrapper` | `mirror-current` | - |
| `src/flexmesa/utils/__init__.py:302` (`Tokenizer`) | `crates/flexmesa-utils/src/lib.rs:569` | `pure-python mirror` | `mirror-stale` | - |

## Public Rust Items With No Python Access (1501)

| Crate | Rust path:line | Item kind | Name | State |
| :--- | :--- | :--- | :--- | :--- |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/controls.rs:17` | `struct` | `CadrParam` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/controls.rs:90` | `struct` | `CadiParam` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/controls.rs:147` | `struct` | `CoutptKernelControls` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/controls.rs:180` | `struct` | `AdiplsControlState` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/controls.rs:224` | `struct` | `SetAdiplsControlsArgs` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/controls.rs:258` | `fn` | `set_adipls_controls` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/controls.rs:301` | `fn` | `sig_fac_adipls_support` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/controls.rs:312` | `fn` | `sig_fac_astero_support` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/controls.rs:321` | `fn` | `sig_from_nu_uhz` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/controls.rs:332` | `fn` | `dfsig_from_delta_nu` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/controls.rs:340` | `struct` | `AdiplsCallerFlags` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/controls.rs:354` | `struct` | `AsteroDefPulseFlags` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/controls.rs:374` | `struct` | `ResolvedPulseExportFlags` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/controls.rs:396` | `fn` | `resolve_store_model_flags` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/driver.rs:28` | `struct` | `AdiplsCallResult` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/driver.rs:56` | `struct` | `AdiplsSolver` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/error.rs:9` | `enum` | `AdiplsError` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:26` | `fn` | `zero` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:34` | `fn` | `store` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:44` | `fn` | `notwin` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:58` | `fn` | `notwni` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:73` | `fn` | `leq` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:214` | `fn` | `leqdet` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:226` | `fn` | `leqsd` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:253` | `fn` | `lininh4` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:453` | `fn` | `aramax` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:504` | `fn` | `derive` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:520` | `fn` | `deriv2` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:765` | `struct` | `DerivkCache` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:777` | `fn` | `derivk` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:868` | `fn` | `vinta` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:969` | `struct` | `VintkCache` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:985` | `fn` | `vintk` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:1072` | `struct` | `LirState` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:1087` | `fn` | `lir` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/gensr.rs:1103` | `fn` | `lir1` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/model.rs:41` | `struct` | `FgongModel` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/model.rs:137` | `struct` | `AmdlAa` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/model.rs:294` | `struct` | `FgongAmdlOutput` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/model.rs:310` | `struct` | `StoreAmdlOutput` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/model.rs:335` | `fn` | `fgong_amdl` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/model.rs:532` | `fn` | `store_amdl` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/output.rs:14` | `struct` | `ObsSt` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/output.rs:88` | `struct` | `Csummm` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/output.rs:207` | `struct` | `ModeRecord` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/output.rs:229` | `struct` | `AdiModeStore` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/output.rs:290` | `fn` | `spcout_adi` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/output.rs:318` | `fn` | `spcout_adi_store` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:20` | `struct` | `AdiFileUnit` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:30` | `struct` | `CntrdSections` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:82` | `struct` | `AdiplsDsnControls` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:112` | `struct` | `AdiplsModControls` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:149` | `struct` | `AdiplsOscControls` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:224` | `struct` | `AdiplsRotControls` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:251` | `struct` | `AdiplsCstControls` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:268` | `struct` | `AdiplsIntControls` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:347` | `struct` | `AdiplsOutControls` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:431` | `struct` | `AdiplsDgnControls` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:470` | `struct` | `AdiplsInput` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:498` | `struct` | `RedistrbInput` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:610` | `fn` | `parse_adipls_pruned_file` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:625` | `fn` | `parse_redistrb_pruned_file` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:641` | `fn` | `parse_adipls_pruned_in` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/parser.rs:1022` | `fn` | `parse_redistrb_pruned_in` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:40` | `struct` | `Varcon` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:63` | `struct` | `VarfrqCache` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:98` | `struct` | `KinerOutput` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:113` | `fn` | `kiner` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:135` | `fn` | `kiner_with_observer` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:248` | `fn` | `varfrq` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:283` | `fn` | `varfrq_with_observer` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:637` | `struct` | `CrotSplit` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:647` | `struct` | `ObsStState` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:675` | `fn` | `setobs_st` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:760` | `struct` | `Ssummm` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:770` | `struct` | `SsmModelRecord` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:782` | `struct` | `SetssmOutput` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:794` | `struct` | `SetssmState` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:812` | `fn` | `setssm` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:867` | `struct` | `SigoutState` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:990` | `struct` | `DefaultSigoutHandler` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/postproc.rs:1042` | `fn` | `sigout` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:37` | `fn` | `sp_radial_eps` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:44` | `struct` | `AdiplsSetupState` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:58` | `fn` | `setups_adi` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:70` | `struct` | `Cprcns` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:96` | `struct` | `RhsDat` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:135` | `struct` | `RotDat` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:149` | `struct` | `BcsDat` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:174` | `struct` | `Cincnt` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:267` | `struct` | `Csexpc` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:313` | `struct` | `Cnrkbc` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:330` | `struct` | `XlmultCache` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:350` | `struct` | `ResAdimodOutput` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:380` | `fn` | `res_adimod` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:602` | `fn` | `setbcs` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:1170` | `fn` | `adirhs` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:1359` | `fn` | `geninh` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:1406` | `fn` | `danorm` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:1430` | `fn` | `xlmult` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/rhs.rs:1508` | `fn` | `sclasl` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:35` | `enum` | `AdiplsTapEvent` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:173` | `trait` | `AdiplsTapObserver` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:180` | `struct` | `NoopTapObserver` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:189` | `struct` | `TapRecorder` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:211` | `struct` | `AdiStatusLogEntry` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:237` | `struct` | `SignwPgState` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:245` | `struct` | `SigoutArgs` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:298` | `trait` | `SigoutHandler` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:311` | `struct` | `NoopSigoutHandler` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:325` | `struct` | `SolverContext` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:368` | `fn` | `shtint` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:576` | `fn` | `sigint` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:658` | `fn` | `mchcf1` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:709` | `fn` | `mchcff` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:764` | `fn` | `mchsol` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:845` | `fn` | `stevft` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:916` | `fn` | `rsnfit` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:990` | `fn` | `takata` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:1087` | `fn` | `order` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:1225` | `fn` | `signew` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:1274` | `fn` | `signw_pg` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:1310` | `fn` | `sigitr` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:1462` | `fn` | `sigitc` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:1578` | `fn` | `setlog` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:1620` | `fn` | `sigsol` | `rust-only` |
| `flexmesa-adipls` | `crates/flexmesa-adipls/src/solve.rs:1823` | `fn` | `sigscn` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/e2e.rs:27` | `struct` | `TempCaseStaging` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/e2e.rs:113` | `fn` | `run_astero_test_suite_case` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/e2e.rs:132` | `fn` | `run_astero_test_suite_case_with_logs` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/e2e.rs:151` | `fn` | `run_astero_test_suite_case_with_options` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/inlist_astero.rs:22` | `struct` | `AsteroSearchParam` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/inlist_astero.rs:56` | `struct` | `SpectroConstraint` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/inlist_astero.rs:69` | `struct` | `AsteroParamSet` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/inlist_parser.rs:186` | `struct` | `AsteroControlInventory` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/inlist_parser.rs:197` | `fn` | `supported_astero_control_inventory` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/inlist_parser.rs:208` | `fn` | `validate_astero_sections_with_pgstar` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/inlist_parser.rs:253` | `fn` | `validate_astero_sections` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/optimize.rs:18` | `fn` | `nelder_mead_minimize` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/optimize.rs:51` | `fn` | `simplex_minimize_with_options` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/optimize.rs:500` | `fn` | `map_bobyqa_param` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/optimize.rs:518` | `fn` | `unmap_bobyqa_param` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/optimize.rs:535` | `fn` | `newuoa_minimize` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/optimize.rs:560` | `fn` | `try_newuoa_minimize` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/optimize.rs:649` | `fn` | `bobyqa_minimize` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/optimize.rs:768` | `fn` | `powell_quadratic_minimize` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/ratios_chi2.rs:67` | `fn` | `get_max_sequence` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/ratios_chi2.rs:98` | `struct` | `FrequencyRatiosR010` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/ratios_chi2.rs:131` | `fn` | `eval_frequency_ratios_with_trimming` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/ratios_chi2.rs:303` | `struct` | `FrequencyRatiosR02Obs` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/ratios_chi2.rs:319` | `fn` | `eval_r02_obs_with_indices` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/ratios_chi2.rs:395` | `fn` | `eval_r02_model_with_indices` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/ratios_chi2.rs:429` | `fn` | `eval_frequency_ratios_r02_with_dnu` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/ratios_chi2.rs:506` | `fn` | `interpolate_ratio_r010` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/ratios_chi2.rs:536` | `fn` | `interpolate_ratio_r02` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/sample_eval.rs:36` | `struct` | `AsteroSampleResult` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/sample_eval.rs:207` | `fn` | `eval_spectro_and_delta_nu_chi2` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/sample_eval.rs:239` | `fn` | `eval_model_astero_sample` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/search_driver.rs:43` | `struct` | `AsteroSearchReport` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/search_driver.rs:875` | `fn` | `run_single_astero_track` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/search_driver.rs:953` | `fn` | `run_single_astero_track_with_logs` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/search_driver.rs:981` | `fn` | `run_single_astero_track_with_case_hooks` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/search_driver.rs:1109` | `fn` | `run_star_astero` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/search_driver.rs:1126` | `fn` | `run_star_astero_with_logs` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/search_driver.rs:1144` | `fn` | `run_star_astero_with_options` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/search_driver.rs:1169` | `fn` | `run_star_astero_with_case_hooks` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/surface_corr.rs:54` | `fn` | `estimate_l0_n_obs_with_numax` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/surface_corr.rs:94` | `fn` | `estimate_l0_n_obs` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/surface_corr.rs:107` | `fn` | `eval_surf_coefs` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/surface_corr.rs:493` | `fn` | `eval_power_law_corr_radial_only` | `rust-only` |
| `flexmesa-astero` | `crates/flexmesa-astero/src/surface_corr.rs:559` | `fn` | `eval_sonoi_corr_radial_only` | `rust-only` |
| `flexmesa-atm` | `crates/flexmesa-atm/src/t_tau_uniform.rs:310` | `fn` | `build_t_tau_uniform` | `rust-only` |
| `flexmesa-atm` | `crates/flexmesa-atm/src/t_tau_varying.rs:289` | `fn` | `build_t_tau_varying` | `rust-only` |
| `flexmesa-atm` | `crates/flexmesa-atm/src/table.rs:31` | `struct` | `AtmTableInfo` | `rust-only` |
| `flexmesa-atm` | `crates/flexmesa-atm/src/table.rs:226` | `fn` | `get_table_values` | `rust-only` |
| `flexmesa-atm` | `crates/flexmesa-atm/src/types.rs:148` | `enum` | `AtmTTauOpacity` | `rust-only` |
| `flexmesa-atm` | `crates/flexmesa-atm/src/types.rs:300` | `struct` | `AtmStructure` | `rust-only` |
| `flexmesa-atm` | `crates/flexmesa-atm/src/utils.rs:13` | `fn` | `atm_black_body_t` | `rust-only` |
| `flexmesa-atm` | `crates/flexmesa-atm/src/utils.rs:25` | `fn` | `atm_l` | `rust-only` |
| `flexmesa-atm` | `crates/flexmesa-atm/src/utils.rs:31` | `fn` | `eval_teff_g` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:18` | `fn` | `unary3_sign` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:26` | `fn` | `unary3_abs` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:33` | `fn` | `unary3_sqrt` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:44` | `fn` | `unary3_safe_sqrt` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:57` | `fn` | `unary3_exp` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:65` | `fn` | `unary3_expm1` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:73` | `fn` | `unary3_exp10` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:84` | `fn` | `unary3_powm1` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:93` | `fn` | `unary3_log` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:102` | `fn` | `unary3_safe_log` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:116` | `fn` | `unary3_log1p` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:125` | `fn` | `unary3_log10` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:135` | `fn` | `unary3_safe_log10` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:150` | `fn` | `unary3_log2` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:161` | `fn` | `unary3_sin` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:170` | `fn` | `unary3_cos` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:179` | `fn` | `unary3_tan` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:189` | `fn` | `unary3_sinpi` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:198` | `fn` | `unary3_cospi` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:207` | `fn` | `unary3_tanpi` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:218` | `fn` | `unary3_sinh` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:227` | `fn` | `unary3_cosh` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:236` | `fn` | `unary3_tanh` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:247` | `fn` | `unary3_asin` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:258` | `fn` | `unary3_acos` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:266` | `fn` | `unary3_atan` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:277` | `fn` | `unary3_asinpi` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:286` | `fn` | `unary3_acospi` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:295` | `fn` | `unary3_atanpi` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:304` | `fn` | `unary3_asinh` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:315` | `fn` | `unary3_acosh` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:326` | `fn` | `unary3_atanh` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:337` | `fn` | `unary3_pown` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:363` | `fn` | `unary3_pow_scalar` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/kernels3.rs:372` | `fn` | `unary3_rpow_scalar` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/lib.rs:54` | `type` | `AutoDiffReal4Coeff` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/lib.rs:60` | `type` | `AutoDiffReal2ndOrder` | `rust-only` |
| `flexmesa-autodiff` | `crates/flexmesa-autodiff/src/order3.rs:21` | `struct` | `AutoDiffReal2VarOrder3` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_micro.rs:200` | `fn` | `bench_interp2d` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_micro.rs:253` | `fn` | `bench_neu` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_solver.rs:55` | `fn` | `bench_mtx_block_tridiag` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_solver.rs:270` | `fn` | `bench_turb_mlt` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_solver.rs:297` | `fn` | `bench_turb_tdc` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave2.rs:80` | `fn` | `bench_chem_comp` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave2.rs:136` | `fn` | `bench_ion_fe56` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave2.rs:165` | `fn` | `bench_ion_4d` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave2.rs:219` | `fn` | `bench_colors_sed` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave3a.rs:93` | `fn` | `bench_eos_ideal` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave3a.rs:159` | `fn` | `bench_eos_helm` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave3a.rs:217` | `fn` | `bench_eos_pc` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave3a.rs:267` | `fn` | `bench_eos_skye` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave3b.rs:79` | `fn` | `bench_eos_4d` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave3b.rs:151` | `fn` | `bench_kap_fixed` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave3b.rs:213` | `fn` | `bench_rates_screen` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave3b.rs:280` | `fn` | `bench_net_eval` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave3b.rs:389` | `fn` | `bench_atm_table` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave4.rs:264` | `fn` | `bench_star_newton_step` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave5.rs:443` | `fn` | `bench_w5_diff_rot` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave5.rs:457` | `fn` | `bench_w5_riemann_rsp2` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave5.rs:472` | `fn` | `bench_w5_binary_rlof` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/kernels_wave5.rs:486` | `fn` | `bench_w5_astero_seism` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/lib.rs:121` | `fn` | `measure_stage3_chunks_us` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/lib.rs:266` | `fn` | `nan_max` | `rust-only` |
| `flexmesa-bench` | `crates/flexmesa-bench/src/tridiag_fast.rs:67` | `fn` | `fast_block_thomas_solve` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/binary_history.rs:140` | `struct` | `BinaryHistoryColumnSpec` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/binary_history.rs:155` | `fn` | `default_binary_history_specs` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/binary_history.rs:167` | `fn` | `parse_binary_history_columns` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/binary_history.rs:184` | `fn` | `resolve_binary_history_columns` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/binary_history.rs:348` | `fn` | `evaluate_binary_history_row` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/binary_history.rs:635` | `fn` | `binary_history_row_as_f64_pairs` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/binary_history.rs:653` | `fn` | `format_binary_history_file_header` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/binary_history.rs:709` | `fn` | `format_binary_history_data_line` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/binary_history.rs:726` | `struct` | `IncrementalBinaryHistoryWriter` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/binary_history.rs:855` | `struct` | `BinaryLogsWriter` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/e2e.rs:210` | `fn` | `run_binary_test_suite_case` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/e2e.rs:224` | `fn` | `run_binary_test_suite_case_with_logs` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/e2e.rs:272` | `fn` | `run_binary_test_suite_case_with_options` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/edd_wind.rs:20` | `struct` | `PointMassEddRetention` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/edd_wind.rs:105` | `fn` | `eval_mdot_edd_with_overrides` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/edd_wind.rs:237` | `fn` | `eval_tout_wind_enhancement_orbit_averaged` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/inlist.rs:26` | `struct` | `BinaryInlistControls` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/inlist.rs:528` | `fn` | `ensure_no_unported_binary_driver_controls` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/inlist_inventory.rs:194` | `struct` | `BinaryControlInventory` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/inlist_inventory.rs:211` | `fn` | `supported_binary_control_inventory` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/inlist_inventory.rs:237` | `fn` | `validate_binary_sections` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/jdot_tides.rs:81` | `fn` | `check_jdot_mb_conditions` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/jdot_tides.rs:165` | `fn` | `eval_jdot_mb_with_scale_mass` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/jdot_tides.rs:224` | `fn` | `eval_jdot_ls_with_winds` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/jdot_tides.rs:248` | `fn` | `eval_jdot_ls_with_am_removed` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/jdot_tides.rs:288` | `fn` | `eval_jdot_missing_wind` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/mdot.rs:119` | `fn` | `eval_ritter_mdot_eccentric` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/mdot.rs:177` | `fn` | `kolb_f3` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/mdot.rs:244` | `fn` | `eval_kolb_mdot_thick_photosphere` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/mdot.rs:317` | `fn` | `eval_kolb_mdot_eccentric` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/mdot.rs:369` | `fn` | `eval_kolb_mdot_eccentric_photosphere` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/mdot.rs:462` | `fn` | `eval_edot_enhancement_isotropic` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/mdot_implicit.rs:87` | `enum` | `ImplicitSolverMode` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/mdot_implicit.rs:277` | `enum` | `ImplicitStepAction` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/mdot_implicit.rs:292` | `struct` | `RloIterReportContext` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/mdot_implicit.rs:306` | `struct` | `ImplicitMdotSolver` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/mdot_implicit.rs:869` | `fn` | `update_change_factor_after_step` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/orbit.rs:28` | `fn` | `eval_rlobe_ad` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/roche_deformation.rs:30` | `struct` | `RocheTable2D` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/roche_deformation.rs:244` | `struct` | `RocheDeformationTables` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/roche_deformation.rs:377` | `fn` | `eval_synchronicity_switch` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/roche_deformation.rs:414` | `struct` | `BinaryTidalDeformationHook` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system.rs:79` | `struct` | `BinaryStepCounters` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_jdot.rs:42` | `struct` | `JdotBreakdown` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_jdot.rs:62` | `fn` | `eval_star_spin_j` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_jdot.rs:105` | `fn` | `eval_star_r_gyr_sq` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_jdot.rs:149` | `fn` | `eval_convective_envelope_mass_and_size` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_jdot.rs:163` | `fn` | `apply_initial_orbit_sync` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_jdot.rs:185` | `fn` | `restore_smooth_omega_after_remesh` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_jdot.rs:217` | `fn` | `conserve_spin_after_henyey` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_jdot.rs:341` | `fn` | `apply_tidal_spin_sync` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_jdot.rs:774` | `fn` | `eval_gated_jdot` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_jdot.rs:1036` | `struct` | `EdotBreakdown` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_jdot.rs:1050` | `fn` | `eval_coupled_edot` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_jdot.rs:1078` | `fn` | `eval_coupled_edot_breakdown` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:42` | `struct` | `CoupledMdotAccounting` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:74` | `struct` | `BinaryStepSnapshot` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:145` | `struct` | `BinaryStepScalars` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:374` | `fn` | `eval_star_intrinsic_wind_cgs` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:398` | `fn` | `eval_star_intrinsic_wind_cgs_with_extras` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:450` | `fn` | `eval_star_intrinsic_wind_cgs_eccentric` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:495` | `fn` | `set_accretion_composition` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:537` | `fn` | `prepare_coupled_mdots` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:569` | `fn` | `prepare_coupled_mdots_with_extras` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:813` | `fn` | `eval_scheme_explicit_mdot` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:836` | `fn` | `eval_scheme_explicit_mdot_with_steps` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:925` | `fn` | `blend_explicit_mdot` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:945` | `fn` | `star_skips_hard_limits` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:959` | `fn` | `star_exceeds_delta_mdot_hard_limit` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:1212` | `fn` | `evolve_single_star_attempt` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/system_rlof.rs:1234` | `fn` | `evolve_single_star_without_term_rewind` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/timestep.rs:31` | `fn` | `binary_dt_why_str` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/timestep.rs:47` | `enum` | `BinaryHardLimitViolation` | `rust-only` |
| `flexmesa-binary` | `crates/flexmesa-binary/src/timestep.rs:335` | `fn` | `check_binary_hard_limits` | `rust-only` |
| `flexmesa-chem` | `crates/flexmesa-chem/src/composition.rs:57` | `struct` | `CompositionMoments` | `rust-only` |
| `flexmesa-chem` | `crates/flexmesa-chem/src/composition.rs:235` | `fn` | `get_composition_info_from_props` | `rust-only` |
| `flexmesa-chem` | `crates/flexmesa-chem/src/composition.rs:308` | `fn` | `get_composition_info_zones` | `rust-only` |
| `flexmesa-chem` | `crates/flexmesa-chem/src/lib.rs:95` | `enum` | `ChemError` | `rust-only` |
| `flexmesa-chem` | `crates/flexmesa-chem/src/lib.rs:132` | `trait` | `ChemScalar` | `rust-only` |
| `flexmesa-chem` | `crates/flexmesa-chem/src/nuclides.rs:654` | `struct` | `NuclideSetEntry` | `rust-only` |
| `flexmesa-chem` | `crates/flexmesa-chem/src/nuclides.rs:665` | `fn` | `generate_nuclide_set` | `rust-only` |
| `flexmesa-chem` | `crates/flexmesa-chem/src/nuclides.rs:682` | `fn` | `get_nuclide_index_in_set` | `rust-only` |
| `flexmesa-chem` | `crates/flexmesa-chem/src/nuclides.rs:705` | `fn` | `builtin_nuclide_rows` | `rust-only` |
| `flexmesa-chem` | `crates/flexmesa-chem/src/solar_ag89_data.rs:40` | `type` | `Ag89Row` | `rust-only` |
| `flexmesa-chem` | `crates/flexmesa-chem/src/solar_scales.rs:112` | `fn` | `element_atomic_weights` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/filters_data.rs:55` | `struct` | `FilterCurve` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/loader.rs:40` | `struct` | `LoadedFilter` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/loader.rs:72` | `struct` | `ColorsEvalOutput` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/loader.rs:86` | `struct` | `ColorsContext` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/loader.rs:149` | `struct` | `ColorsLoadConfig` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/sed_cube.rs:571` | `fn` | `builtin_sed_cube_arc` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/types.rs:13` | `trait` | `RealScalar` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/types.rs:122` | `enum` | `ColorsError` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/utils.rs:74` | `fn` | `simpson_integration_fn` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/utils.rs:139` | `fn` | `linear_interpolate` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/utils.rs:203` | `fn` | `is_dummy_axis` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/utils.rs:213` | `fn` | `find_interval` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/utils.rs:247` | `fn` | `find_bracket_index` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/utils.rs:264` | `fn` | `find_nearest_point` | `rust-only` |
| `flexmesa-colors` | `crates/flexmesa-colors/src/utils.rs:323` | `fn` | `compute_interp_radius_lookup` | `rust-only` |
| `flexmesa-data` | `crates/flexmesa-data/src/converters.rs:98` | `fn` | `convert_ascii_table_file` | `rust-only` |
| `flexmesa-data` | `crates/flexmesa-data/src/lib.rs:50` | `enum` | `DataError` | `rust-only` |
| `flexmesa-data` | `crates/flexmesa-data/src/safetensors.rs:304` | `fn` | `decode_safetensors_bytes` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/blend.rs:31` | `fn` | `quadrant` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/blend.rs:80` | `fn` | `winding_number` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/blend.rs:89` | `fn` | `is_contained` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/blend.rs:111` | `fn` | `min_distance_from_point_to_line_segment` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/blend_dt.rs:153` | `fn` | `get_helm_alfa` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/blend_dt.rs:187` | `fn` | `combine_two_eos_components` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/blend_dt.rs:328` | `fn` | `eos_dt_get_from_weights_with_tables` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/blend_dt.rs:507` | `fn` | `eos_dt_get_from_weights` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/blend_dt.rs:589` | `fn` | `eos_dt_get_zones_with_logs` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/blend_dt.rs:651` | `fn` | `eos_dt_get_zones` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/cms/config.rs:164` | `fn` | `get_cms_alfa` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/cms/mod.rs:50` | `fn` | `eval_cms_fixed_x` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/cms/mod.rs:139` | `fn` | `include_cms_radiation` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/cms/mod.rs:215` | `fn` | `eval_cms_eos_from_weights` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/cms/table.rs:31` | `struct` | `CmsXTable` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/eos_pt.rs:21` | `struct` | `EosPtOutput` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/eos_pt.rs:38` | `struct` | `EosPtInverseOutput` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/free_energy.rs:25` | `fn` | `sifac_constant` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/free_energy.rs:35` | `fn` | `compute_f_rad` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/free_energy.rs:54` | `fn` | `compute_f_ideal_ion` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/free_energy.rs:78` | `fn` | `compute_xne` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/free_energy.rs:94` | `fn` | `thermodynamics_from_free_energy` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/free_energy.rs:111` | `struct` | `DerivedThermodynamics` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/free_energy.rs:138` | `fn` | `compute_derived_quantities` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/gamma_de.rs:16` | `struct` | `EosGammaDeOutput` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/gauss_fermi.rs:275` | `fn` | `dfermi_order1` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/components.rs:19` | `struct` | `HelmThermoVar` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/components.rs:104` | `fn` | `eval_radiation` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/components.rs:170` | `fn` | `eval_ideal_ions` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/convert.rs:37` | `fn` | `do_convert_helm_results` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/convert.rs:181` | `fn` | `eval_helm_eos_zones` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/coulomb.rs:37` | `fn` | `eval_helm_coulomb` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/elec_pos.rs:20` | `fn` | `eval_electron_positron` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/eval.rs:32` | `struct` | `HelmConfig` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/eval.rs:80` | `fn` | `helmeos2` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:23` | `fn` | `psi0` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:30` | `fn` | `dpsi0` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:37` | `fn` | `ddpsi0` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:44` | `fn` | `dddpsi0` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:54` | `fn` | `psi1` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:61` | `fn` | `dpsi1` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:68` | `fn` | `ddpsi1` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:75` | `fn` | `dddpsi1` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:85` | `fn` | `psi2` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:92` | `fn` | `dpsi2` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:100` | `fn` | `ddpsi2` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:107` | `fn` | `dddpsi2` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:120` | `fn` | `h5` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:164` | `fn` | `xpsi0` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:171` | `fn` | `xdpsi0` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:178` | `fn` | `xddpsi0` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:185` | `fn` | `xdddpsi0` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:192` | `fn` | `xpsi1` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:199` | `fn` | `xdpsi1` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:206` | `fn` | `xddpsi1` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:213` | `fn` | `xdddpsi1` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:226` | `fn` | `h3` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:249` | `struct` | `QuinticBasis1D` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/polynomials.rs:315` | `struct` | `CubicBasis1D` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/table.rs:42` | `struct` | `HelmVertex` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/table.rs:55` | `struct` | `HelmCell` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/table_io.rs:75` | `fn` | `load_or_build_helm_vertices_in` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/table_io.rs:226` | `fn` | `decode_helm_full_bytes` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/table_io.rs:265` | `fn` | `encode_helm_full_bytes` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/table_io.rs:282` | `fn` | `read_helm_full_bin` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/table_io.rs:290` | `fn` | `write_helm_full_bin_atomic` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/table_io.rs:313` | `fn` | `read_mesa_helm_cache_bin` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/helm/table_io.rs:360` | `fn` | `parse_helm_ascii_reader` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/inverse.rs:30` | `struct` | `EosInverseOutput` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/blin.rs:13` | `struct` | `BlinSeries` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/blin.rs:30` | `struct` | `Blin9Output` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/blin.rs:64` | `fn` | `blin9` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/blin.rs:103` | `fn` | `blin9a` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/blin.rs:215` | `fn` | `blin9b` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/blin_c.rs:14` | `fn` | `blin9c` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/coulomb_ion.rs:13` | `struct` | `PcIonContrib` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/coulomb_ion.rs:36` | `fn` | `fition9` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/coulomb_ion.rs:87` | `fn` | `hlfit12` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/coulomb_ion.rs:291` | `fn` | `fharm12` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/coulomb_ion.rs:321` | `fn` | `anharm8` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/coulomb_screen.rs:16` | `fn` | `fscr_liq8` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/coulomb_screen.rs:182` | `fn` | `fscr_sol8` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/coulomb_screen.rs:363` | `fn` | `cormix` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/elect.rs:16` | `struct` | `Elect11Output` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/elect.rs:52` | `fn` | `elect11` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/elect.rs:91` | `fn` | `elect11a` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/elect.rs:166` | `fn` | `elect11b` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/elect.rs:264` | `fn` | `sommerf` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/eosfi.rs:14` | `fn` | `eosfi8` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/excor.rs:23` | `fn` | `excor7` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/excor.rs:61` | `fn` | `excor7_analytic` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/excor_table.rs:59` | `struct` | `Excor7Interp` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/excor_table.rs:86` | `struct` | `Excor7Table` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/excor_table.rs:107` | `fn` | `excor7_table_point` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/excor_table.rs:641` | `fn` | `get_excor7` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/ferinv.rs:12` | `fn` | `fermi10` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/ferinv.rs:40` | `fn` | `ferinv7` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/ferinv.rs:283` | `fn` | `chemfit` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/melange.rs:18` | `struct` | `Melange9Output` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/melange.rs:53` | `fn` | `melange9` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/pc/mod.rs:321` | `fn` | `eval_pc_eos_zones` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb.rs:24` | `struct` | `SkyeNonidealOutput` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb.rs:38` | `fn` | `ocp_free_energy` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb.rs:56` | `fn` | `extrapolate_free_energy` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb.rs:99` | `fn` | `decide_phase` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb.rs:188` | `fn` | `nonideal_corrections` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb_liquid.rs:18` | `fn` | `classical_ocp_liquid_free_energy` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb_liquid.rs:43` | `fn` | `quantum_ocp_liquid_free_energy_correction` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb_liquid.rs:58` | `fn` | `ocp_liquid_screening_free_energy_correction` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb_liquid.rs:106` | `fn` | `liquid_mixing_rule_correction` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb_liquid.rs:149` | `fn` | `excor7_order3` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb_solid.rs:39` | `fn` | `ocp_solid_anharmonic_free_energy` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb_solid.rs:56` | `fn` | `ocp_solid_harmonic_free_energy` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb_solid.rs:137` | `fn` | `ocp_solid_screening_free_energy_correction` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb_solid.rs:193` | `fn` | `delta_g_ogata93` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb_solid.rs:207` | `fn` | `delta_g_pc13` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/coulomb_solid.rs:218` | `fn` | `solid_mixing_rule_correction` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/skye/mod.rs:377` | `fn` | `eval_skye_eos_zones` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/blend_alfa.rs:186` | `fn` | `get_free_eos_alfa` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/blend_alfa.rs:287` | `fn` | `get_opal_scvh_alfa_and_partials` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/mod.rs:72` | `fn` | `get1_eosdt_for_x` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/mod.rs:397` | `fn` | `eval_eosdt_table_from_weights` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/slice_cache.rs:203` | `fn` | `load_or_build_dt_slice` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/slice_cache.rs:248` | `fn` | `load_or_build_dt_slice_in` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/slice_cache.rs:388` | `fn` | `load_or_build_cms_slice` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/slice_cache.rs:408` | `fn` | `load_or_build_cms_slice_in` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/slice_cache.rs:522` | `fn` | `decode_dt_slice_bytes` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/slice_cache.rs:667` | `fn` | `read_dt_slice_bin` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/slice_cache.rs:677` | `fn` | `encode_dt_slice_bytes` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/slice_cache.rs:701` | `fn` | `write_dt_slice_bin_atomic` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/slice_cache.rs:710` | `fn` | `read_mesa_eosdt_cache_bin` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/slice_cache.rs:814` | `fn` | `decode_cms_slice_bytes` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/slice_cache.rs:891` | `fn` | `read_cms_slice_bin` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/slice_cache.rs:901` | `fn` | `encode_cms_slice_bytes` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/slice_cache.rs:924` | `fn` | `write_cms_slice_bin_atomic` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/spline.rs:44` | `struct` | `EosDtVertex` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/spline.rs:59` | `struct` | `EosDtXzTable` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/spline.rs:172` | `fn` | `do_eos_interpolations` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/spline.rs:269` | `fn` | `eval_xtable_16` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/spline.rs:332` | `fn` | `eval_xtable` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/table_set.rs:94` | `struct` | `DtXzInfo` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/table_set.rs:151` | `fn` | `format_xz_str` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/table_set.rs:163` | `fn` | `eosdt_table_stem` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/tables_4d/table_set.rs:184` | `fn` | `eosdt_relative_data_path` | `rust-only` |
| `flexmesa-eos` | `crates/flexmesa-eos/src/types.rs:155` | `enum` | `EosError` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ad.rs:39` | `struct` | `AdTrans` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ad.rs:304` | `struct` | `AdEqns` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ad.rs:461` | `struct` | `AdBound` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ad.rs:694` | `struct` | `AdMatch` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ad.rs:771` | `struct` | `AdDiff` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ad.rs:878` | `struct` | `AdBvp` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ad.rs:1020` | `fn` | `ad_wave_hom` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/constants.rs:11` | `struct` | `GyreConstants` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/context.rs:16` | `struct` | `Context` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/crmath.rs:24` | `fn` | `hypot` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/crmath.rs:39` | `fn` | `atan2` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/crmath.rs:56` | `fn` | `abs_c` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/crmath.rs:63` | `fn` | `sqrt_c` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/crmath.rs:92` | `fn` | `exp_c` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/crmath.rs:106` | `fn` | `log_c` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ext_complex.rs:23` | `struct` | `ExtC` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ext_complex.rs:32` | `fn` | `split` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ext_complex.rs:295` | `fn` | `product` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ext_real.rs:26` | `fn` | `fraction` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ext_real.rs:32` | `fn` | `exponent` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ext_real.rs:38` | `fn` | `frexp` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ext_real.rs:61` | `struct` | `ExtR` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ext_real.rs:336` | `fn` | `product` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/ext_real.rs:352` | `fn` | `split` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/fcomplex.rs:32` | `struct` | `Cplx` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/forum.rs:20` | `fn` | `sort_indices` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/forum.rs:128` | `fn` | `unique_indices` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/forum.rs:156` | `fn` | `locate` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/grid.rs:10` | `struct` | `Point` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/grid.rs:20` | `struct` | `Grid` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/grid.rs:210` | `fn` | `fmax` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/grid.rs:221` | `fn` | `fmin` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/grid_refine.rs:21` | `fn` | `refine_grid` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/integ.rs:15` | `fn` | `integrate` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/integ.rs:39` | `fn` | `integral` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/lib.rs:93` | `enum` | `GyreNumError` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/magnus.rs:21` | `fn` | `eigen_decompose_2_r` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/magnus.rs:42` | `fn` | `eigen_decompose_2_c` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/magnus.rs:116` | `struct` | `EigenDecomp` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/magnus.rs:131` | `fn` | `eigen_decompose_r` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/magnus.rs:191` | `fn` | `eigen_decompose_c` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/magnus.rs:246` | `fn` | `magnus_stencil` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/magnus.rs:346` | `fn` | `magnus_build_r` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/magnus.rs:364` | `fn` | `magnus_build_c` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/math_m.rs:9` | `struct` | `MathConstants` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/math_m.rs:49` | `fn` | `pow_rr` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/math_m.rs:61` | `fn` | `pow_rc` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/matmul.rs:51` | `fn` | `matmul_rr` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/matmul.rs:83` | `fn` | `matmul_cc` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/matmul.rs:111` | `fn` | `matvec_r` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/matmul.rs:129` | `fn` | `matvec_c` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/mode.rs:25` | `struct` | `Wave` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/mode.rs:551` | `fn` | `rad_wave_hom` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/model.rs:99` | `struct` | `CubicInterp` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/model.rs:185` | `fn` | `deriv_mono` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/model.rs:211` | `fn` | `snap_points` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/model.rs:230` | `struct` | `EvolModel` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/nad.rs:41` | `struct` | `NadTrans` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/nad.rs:113` | `struct` | `NadEqns` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/nad.rs:438` | `struct` | `NadBound` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/nad.rs:683` | `struct` | `NadMatch` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/nad.rs:773` | `struct` | `NadDiff` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/nad.rs:902` | `struct` | `NadBvp` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/nad.rs:1036` | `fn` | `nad_wave_hom` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/namelist.rs:18` | `enum` | `NmlValue` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/namelist.rs:27` | `struct` | `NmlGroup` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/namelist.rs:36` | `struct` | `NmlError` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/namelist.rs:56` | `fn` | `parse_groups` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/par.rs:20` | `struct` | `ModelPar` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/par.rs:140` | `struct` | `ModePar` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/par.rs:181` | `struct` | `OscPar` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/par.rs:330` | `struct` | `RotPar` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/par.rs:385` | `struct` | `NumPar` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/par.rs:457` | `struct` | `GridPar` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/par.rs:541` | `struct` | `ScanPar` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/par.rs:615` | `struct` | `GyrePars` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/par.rs:691` | `trait` | `Tagged` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/par.rs:744` | `fn` | `select_par_v` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/par.rs:769` | `fn` | `select_par_1` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/powi.rs:21` | `fn` | `powi_var` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/powi.rs:33` | `fn` | `powi_const` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/prox.rs:27` | `fn` | `eval_discrim_c` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/prox.rs:226` | `fn` | `prox_search` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/rad.rs:30` | `struct` | `RadTrans` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/rad.rs:188` | `struct` | `RadEqns` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/rad.rs:282` | `fn` | `atmos_chi_r` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/rad.rs:311` | `struct` | `RadBound` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/rad.rs:466` | `enum` | `DiffKind` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/rad.rs:480` | `fn` | `diff_kind` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/rad.rs:502` | `fn` | `colloc_scheme` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/rad.rs:516` | `fn` | `colloc_stencil` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/rad.rs:656` | `struct` | `RadMatch` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/rad.rs:724` | `struct` | `RadDiff` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/rad.rs:828` | `struct` | `RadBvp` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/rad.rs:851` | `trait` | `GyreTap` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/rad.rs:933` | `struct` | `NoTap` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/scalar.rs:20` | `trait` | `Scalar` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/search.rs:30` | `fn` | `build_scan` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/search.rs:112` | `fn` | `check_scan` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/search.rs:151` | `trait` | `RealBvp` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/search.rs:252` | `fn` | `eval_discrim` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/search.rs:266` | `enum` | `Stat` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/search.rs:509` | `fn` | `solve_root_r` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/search.rs:567` | `fn` | `find_turn` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/search.rs:634` | `fn` | `bracket_search` | `rust-only` |
| `flexmesa-gyre` | `crates/flexmesa-gyre/src/sysmtx.rs:14` | `struct` | `BlockSysmtx` | `rust-only` |
| `flexmesa-interp1d` | `crates/flexmesa-interp1d/src/mp.rs:63` | `fn` | `interp_m3` | `rust-only` |
| `flexmesa-interp2d` | `crates/flexmesa-interp2d/src/akima.rs:64` | `fn` | `eval_akima_point` | `rust-only` |
| `flexmesa-interp2d` | `crates/flexmesa-interp2d/src/lib.rs:119` | `enum` | `Interp2dError` | `rust-only` |
| `flexmesa-interp2d` | `crates/flexmesa-interp2d/src/renka790.rs:48` | `type` | `Qshep2Interpolant` | `rust-only` |
| `flexmesa-interp2d` | `crates/flexmesa-interp2d/src/renka790.rs:470` | `fn` | `qshep2` | `rust-only` |
| `flexmesa-interp2d` | `crates/flexmesa-interp2d/src/renka790.rs:483` | `fn` | `qs2val` | `rust-only` |
| `flexmesa-interp2d` | `crates/flexmesa-interp2d/src/renka790.rs:489` | `fn` | `qs2grd` | `rust-only` |
| `flexmesa-ionization` | `crates/flexmesa-ionization/src/lib.rs:100` | `enum` | `IonError` | `rust-only` |
| `flexmesa-ionization` | `crates/flexmesa-ionization/src/lib.rs:140` | `trait` | `IonScalar` | `rust-only` |
| `flexmesa-ionization` | `crates/flexmesa-ionization/src/typical_charge.rs:100` | `fn` | `get_typical_charge` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/aesopus_h5.rs:35` | `enum` | `AesopusRef` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/aesopus_h5.rs:82` | `fn` | `aesopus_source_exists` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/aesopus_h5.rs:107` | `fn` | `resolve_aesopus_data` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/aesopus_h5.rs:141` | `fn` | `load_aesopus_hdf5_file` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/aesopus_h5.rs:162` | `fn` | `load_aesopus_hdf5_bytes` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/blend.rs:35` | `fn` | `get_kap_results_blend_t` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/blend.rs:128` | `fn` | `get_kap_high_t_results` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/blend.rs:238` | `fn` | `get_kap_low_t_results` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/conduction.rs:22` | `struct` | `PotekhinTable` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/conduction.rs:242` | `fn` | `do_electron_conduction_blouin` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/eval.rs:260` | `fn` | `kap_get_zones` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/loader.rs:19` | `fn` | `get_dx_lookup` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/loader.rs:61` | `struct` | `KapXTable` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/loader.rs:115` | `struct` | `KapZTable` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/loader.rs:131` | `struct` | `KapCoSubTable` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/loader.rs:180` | `struct` | `KapCoXTable` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/loader.rs:238` | `struct` | `AesopusSlice` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/loader.rs:314` | `struct` | `KapCnZSet` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/loader.rs:364` | `struct` | `KapCnData` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/loader.rs:495` | `fn` | `kap_tables` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/loader_ascii.rs:61` | `fn` | `find_kap_data_dir` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/loader_ascii.rs:82` | `fn` | `find_kap_bundle_dir` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/loader_ascii.rs:91` | `fn` | `kap_source_relpaths` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/loader_ascii.rs:225` | `fn` | `load_or_build_kap_tables_in` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/loader_ascii.rs:722` | `fn` | `build_full_kap_tables_bytes` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/low_t.rs:43` | `fn` | `aesopus_get_with_data` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/spline.rs:19` | `fn` | `get_output_string` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/spline.rs:40` | `fn` | `locate_log` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/spline.rs:369` | `fn` | `fit_bicubic_zero_first_deriv` | `rust-only` |
| `flexmesa-kap` | `crates/flexmesa-kap/src/types.rs:269` | `enum` | `KapError` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/io.rs:116` | `fn` | `double_to_str` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/lib.rs:88` | `fn` | `fma` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/lib.rs:148` | `fn` | `log_strict` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/lib.rs:284` | `fn` | `pow_strict` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/lib.rs:322` | `fn` | `powi_strict` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/lib.rs:450` | `fn` | `intrinsic_log` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/lib.rs:464` | `fn` | `intrinsic_log10` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/lib.rs:478` | `fn` | `intrinsic_exp` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/lib.rs:493` | `fn` | `intrinsic_pow` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/mode.rs:83` | `struct` | `ParseNumericsModeError` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/pre_z.rs:15` | `struct` | `PrecompInt` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/pre_z.rs:151` | `fn` | `pre_z` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/quad.rs:69` | `struct` | `Quad` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/trig.rs:58` | `fn` | `sin_cos` | `rust-only` |
| `flexmesa-math` | `crates/flexmesa-math/src/trig.rs:165` | `fn` | `atan2` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band.rs:306` | `fn` | `dgbtrs_trans` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/dgbcon.rs:34` | `fn` | `dgbcon` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/dgbmv.rs:26` | `fn` | `dgbmv` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/dgbrfs.rs:15` | `struct` | `Refinement` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/dgbrfs.rs:23` | `type` | `SolveFn` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/dgbrfs.rs:44` | `fn` | `dgbrfs` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/dgbsvx.rs:28` | `struct` | `DgbsvxSolution` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/dgbsvx.rs:67` | `fn` | `dgbsvx_fact_n` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/dgbtrf.rs:23` | `fn` | `dgbtrf_block_size` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/dgbtrf.rs:35` | `fn` | `dgbtrf_uses_blocked_code` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/dgbtrf.rs:78` | `fn` | `dgbtrf_blocked` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/dlangb.rs:21` | `fn` | `dlangb` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/dlantb.rs:19` | `fn` | `dlantb` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/dlatbs.rs:38` | `fn` | `dlatbs` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/drscl.rs:13` | `fn` | `drscl` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/band_lapack/dtbsv.rs:27` | `fn` | `dtbsv` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/bcyclic.rs:276` | `struct` | `BcyclicFactorization` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/bcyclic.rs:329` | `fn` | `bcyclic_factor` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/bcyclic.rs:496` | `fn` | `bcyclic_solve` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/block_io.rs:28` | `type` | `BlockTridiagData` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/block_io.rs:44` | `fn` | `parse_block_tridiagonal_text` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/block_io.rs:278` | `fn` | `write_block_tridiagonal_text` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/block_tridiag.rs:333` | `fn` | `lu_factored_block_dble_mv` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/block_tridiag.rs:501` | `fn` | `clip_block` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/block_tridiag.rs:533` | `fn` | `clip_blocks` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/dense.rs:108` | `fn` | `dgetrs_trans` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/dense.rs:214` | `fn` | `dgetrs_multi` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/dense.rs:307` | `fn` | `solve_dense_multi` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/dense.rs:443` | `fn` | `solve_dense_dual` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/dense_ops.rs:48` | `fn` | `multiply_xa_plus_c` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/eigen.rs:20` | `struct` | `EigenResult` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/eigen.rs:436` | `fn` | `dgeev_eigenvalues` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/eigen.rs:468` | `fn` | `dgeev_right` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/lapack/mod.rs:62` | `enum` | `LapackError` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/lapack/mod.rs:99` | `fn` | `ix` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/lapack/mod.rs:106` | `fn` | `lsame` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/lapack/mod.rs:129` | `fn` | `ilaenv_nb` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/lapack/mod.rs:145` | `fn` | `ilaenv_mnthr` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/lib.rs:92` | `enum` | `DecsolOption` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/lib.rs:124` | `enum` | `MatrixType` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/lib.rs:139` | `fn` | `decsol_option` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/lib.rs:153` | `fn` | `decsol_option_str` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/lib.rs:164` | `fn` | `is_block_tridiagonal_decsol` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/lib.rs:175` | `enum` | `MtxError` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:35` | `enum` | `Equilibration` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:48` | `struct` | `EquilibrationFactors` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:68` | `struct` | `PreconditionedSolve` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:139` | `fn` | `compute_band_preconditioner` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:216` | `fn` | `compute_block_preconditioner` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:276` | `fn` | `compute_block_left_preconditioner` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:353` | `fn` | `dgbsvx_banded_from_bands` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:451` | `fn` | `dgbsvx_banded` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:546` | `fn` | `dgbsvx_block_tridiagonal` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:654` | `fn` | `dgeequ` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:716` | `fn` | `dlaqge` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:776` | `fn` | `dgbequ` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:854` | `fn` | `dlaqgb` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:969` | `fn` | `solve_dense_equilibrated` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/precond.rs:1008` | `fn` | `solve_banded_equilibrated` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/sparse.rs:149` | `fn` | `dense_to_row_sparse_with_diag` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/sparse.rs:187` | `fn` | `dense_to_row_with_diag_0_based` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/sparse.rs:274` | `fn` | `dense_to_col_sparse_with_diag` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/sparse.rs:312` | `fn` | `dense_to_col_with_diag_0_based` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/sparse.rs:332` | `fn` | `dense_to_sparse_with_diag` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/sparse_band.rs:147` | `fn` | `band_to_column_sparse` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/sparse_band.rs:168` | `fn` | `band_to_col_sparse_with_diag` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/sparse_band.rs:189` | `fn` | `column_sparse_to_band` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/sparse_band.rs:210` | `fn` | `band_to_row_sparse` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/sparse_band.rs:231` | `fn` | `band_to_row_sparse_with_diag` | `rust-only` |
| `flexmesa-mtx` | `crates/flexmesa-mtx/src/sparse_band.rs:252` | `fn` | `row_sparse_to_band` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/approx21.rs:14` | `struct` | `Approx21Indices` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/approx21.rs:325` | `fn` | `approx21_pa_pg_fractions` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/approx21.rs:386` | `fn` | `approx21_weak_rates` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/approx21_dfdy.rs:12` | `fn` | `approx21_dfdy` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/approx21_dydt.rs:212` | `fn` | `approx21_dydt` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/approx21_eps.rs:43` | `fn` | `approx21_eval_ppii_fraction` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/approx21_eps.rs:61` | `fn` | `approx21_eps_info` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/approx21_eps.rs:466` | `fn` | `approx21_d_epsneu_dy` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/approx21_eval.rs:27` | `fn` | `approx21_dfdt_dfdrho` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/approx21_eval.rs:331` | `fn` | `eval_net_approx21_procs` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/approx21_special.rs:16` | `struct` | `Approx21ReactionQs` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/approx21_special.rs:97` | `fn` | `global_approx21_qs` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/approx21_special.rs:222` | `fn` | `approx21_special_reactions` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/derivs.rs:81` | `fn` | `eval_net` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/derivs.rs:319` | `fn` | `get_derivs` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/derivs_support.rs:34` | `fn` | `fl_epsnuc_3alf_with_derivs` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/derivs_support.rs:56` | `enum` | `NetWorkOutputs` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/derivs_support.rs:89` | `struct` | `NetWorkState` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/derivs_support.rs:293` | `fn` | `eval_single_weak_blended` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/derivs_support.rs:472` | `fn` | `get_weaklib_rates` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/derivs_support.rs:528` | `fn` | `do_in_out` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/derivs_support.rs:575` | `fn` | `do_in_out_neu` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/derivs_support.rs:703` | `fn` | `update_special_rates` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/initialize.rs:34` | `fn` | `global_logttab` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/initialize.rs:44` | `fn` | `global_ttab` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/lib.rs:159` | `fn` | `net_get_symbolic_d_dxdt_dx` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/parser.rs:53` | `struct` | `ParsedNetDef` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/reaction_synth.rs:13` | `fn` | `parse_reaction_handle` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/reaction_synth.rs:196` | `fn` | `synthesize_reaction_info` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/screen.rs:18` | `fn` | `init_screening_tables` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/screen.rs:107` | `fn` | `screen_net` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/types.rs:106` | `enum` | `ReaclibKind` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/types.rs:132` | `enum` | `ReactionKind` | `rust-only` |
| `flexmesa-net` | `crates/flexmesa-net/src/types.rs:145` | `struct` | `ScreenPairParams` | `rust-only` |
| `flexmesa-neu` | `crates/flexmesa-neu/src/lib.rs:59` | `enum` | `NeuError` | `rust-only` |
| `flexmesa-num` | `crates/flexmesa-num/src/accurate_sum.rs:23` | `struct` | `AccurateReal` | `rust-only` |
| `flexmesa-num` | `crates/flexmesa-num/src/accurate_sum.rs:33` | `fn` | `neumaier_sum_qp` | `rust-only` |
| `flexmesa-num` | `crates/flexmesa-num/src/lib.rs:103` | `enum` | `NumError` | `rust-only` |
| `flexmesa-num` | `crates/flexmesa-num/src/root.rs:26` | `type` | `BracketPair` | `rust-only` |
| `flexmesa-num` | `crates/flexmesa-num/src/rosenbrock.rs:129` | `fn` | `rosenbrock_dense_autonomous` | `rust-only` |
| `flexmesa-num` | `crates/flexmesa-num/src/rosenbrock.rs:308` | `fn` | `rosenbrock_banded_autonomous` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/ecapture.rs:325` | `fn` | `eval_mui_coulomb_with_derivs` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/ecapture.rs:345` | `fn` | `eval_vs_coulomb_with_derivs` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/ecapture.rs:600` | `fn` | `eval_ecapture_pair_with_tables` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/ecapture.rs:645` | `fn` | `eval_ecapture_pair_with_derivs` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/ecapture.rs:683` | `fn` | `eval_ecapnuc_rate` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/loader.rs:152` | `struct` | `EcaptureTables` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/loader.rs:400` | `struct` | `WeakTable` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/loader.rs:427` | `struct` | `SuzukiTable` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/ratelib.rs:57` | `fn` | `eval_analytic_rate` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/raw_rates.rs:145` | `struct` | `RateTable1D` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/raw_rates.rs:453` | `fn` | `eval_reaction_raw_with_overrides` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/raw_rates.rs:501` | `fn` | `eval_reaction_raw` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/raw_rates.rs:559` | `fn` | `eval_interp_rate_table` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/raw_rates.rs:603` | `fn` | `eval_fl_epsnuc_3alf_with_derivs` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/tfactors.rs:440` | `fn` | `eval_reaclib_inverse_ratio` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/types.rs:180` | `struct` | `ReaclibTerm` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/types.rs:250` | `enum` | `RatesError` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/weak.rs:29` | `fn` | `bilinear_interp_2d` | `rust-only` |
| `flexmesa-rates` | `crates/flexmesa-rates/src/weak.rs:158` | `fn` | `eval_suzuki_one` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/argdiff.rs:27` | `type` | `Key` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/argdiff.rs:131` | `struct` | `RecordDiff` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/argdiff.rs:341` | `struct` | `FieldStats` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/argdiff.rs:354` | `struct` | `FamilyMatch` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/argdiff.rs:370` | `struct` | `ArgDiffSummary` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/argdiff.rs:421` | `fn` | `compare_captures` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/compare.rs:12` | `fn` | `ulp_distance` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/compare.rs:34` | `struct` | `Mismatch` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/compare.rs:68` | `struct` | `CallDiff` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/compare.rs:125` | `struct` | `FieldTally` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/compare.rs:138` | `struct` | `FamilySummary` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/eos.rs:250` | `fn` | `eos_config_from_config` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/eos.rs:495` | `fn` | `element_name` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/eos.rs:539` | `fn` | `replay_eos` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/lib.rs:52` | `struct` | `ReplaySummary` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/lib.rs:77` | `fn` | `replay` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/record.rs:37` | `type` | `MesaAd` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/record.rs:41` | `struct` | `FormatError` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/record.rs:53` | `enum` | `FieldType` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/record.rs:84` | `enum` | `Value` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/record.rs:103` | `struct` | `Schema` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/record.rs:150` | `struct` | `Record` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/record.rs:165` | `fn` | `encode_real` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/record.rs:172` | `fn` | `encode_ad` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/record.rs:185` | `fn` | `encode_vec` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/record.rs:369` | `struct` | `CaptureReader` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/turb.rs:52` | `enum` | `Replayed` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/turb.rs:69` | `fn` | `replay_tdc` | `rust-only` |
| `flexmesa-replay` | `crates/flexmesa-replay/src/turb.rs:144` | `fn` | `replay_mlt` | `rust-only` |
| `flexmesa-schema` | `crates/flexmesa-schema/src/lib.rs:107` | `enum` | `SchemaError` | `rust-only` |
| `flexmesa-schema` | `crates/flexmesa-schema/src/validation.rs:48` | `fn` | `upstream_section_keys` | `rust-only` |
| `flexmesa-schema` | `crates/flexmesa-schema/src/validation.rs:58` | `fn` | `is_headless_plotting_section` | `rust-only` |
| `flexmesa-schema` | `crates/flexmesa-schema/src/validation.rs:68` | `fn` | `is_known_upstream_only_section` | `rust-only` |
| `flexmesa-schema` | `crates/flexmesa-schema/src/validation.rs:79` | `fn` | `section_include_prefix` | `rust-only` |
| `flexmesa-schema` | `crates/flexmesa-schema/src/validation.rs:95` | `fn` | `is_include_key` | `rust-only` |
| `flexmesa-schema` | `crates/flexmesa-schema/src/validation.rs:120` | `struct` | `ControlValidationReport` | `rust-only` |
| `flexmesa-schema` | `crates/flexmesa-schema/src/validation.rs:391` | `fn` | `format_fortran_value` | `rust-only` |
| `flexmesa-schema` | `crates/flexmesa-schema/src/validation.rs:408` | `fn` | `copy_cleaned_case_dir` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/accretion_xa.rs:24` | `fn` | `get_xa_for_accretion` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/accretion_xa.rs:75` | `fn` | `build_accreted_xa` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mass.rs:30` | `fn` | `compute_delta_m` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mass.rs:111` | `fn` | `do_adjust_mass_with_dt` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mass.rs:161` | `fn` | `do_adjust_mass_step` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_split_merge.rs:19` | `fn` | `minmod3` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_split_merge.rs:39` | `fn` | `get1_grad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_split_merge.rs:1080` | `fn` | `preview_split_merge_amr` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_split_merge.rs:1118` | `fn` | `remesh_split_merge` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_support.rs:42` | `struct` | `GvalInfo` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_support.rs:93` | `fn` | `smooth_gvals` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_support.rs:252` | `fn` | `set_delta_gval_max_with_controls` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_support.rs:366` | `fn` | `get_gval_info_with_controls` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_support.rs:396` | `fn` | `get_gval_info` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_support.rs:405` | `fn` | `blend_coef` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_support.rs:416` | `fn` | `single_peak` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_support.rs:444` | `fn` | `increasing_inward` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_support.rs:461` | `fn` | `decreasing_outward` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_support.rs:479` | `fn` | `increasing_outward` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_mesh_support.rs:500` | `fn` | `check_validity` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/adjust_net.rs:79` | `fn` | `check_adjust_net` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/cell_energy.rs:183` | `fn` | `cell_specific_pe` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/cell_energy.rs:193` | `fn` | `cell_specific_ke` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/cell_energy.rs:207` | `fn` | `cell_specific_rotational_energy` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/cell_energy.rs:242` | `fn` | `cell_specific_total_energy` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/cell_energy.rs:267` | `struct` | `DkeDpeDt` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/cell_energy.rs:298` | `fn` | `get_dke_dt_dpe_dt` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/controls.rs:898` | `struct` | `ColorsControls` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/controls_diffusion.rs:17` | `struct` | `DiffusionControls` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/controls_mix_wind.rs:26` | `struct` | `OvershootParamSet` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/controls_mix_wind.rs:63` | `struct` | `PredictiveParamSet` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/controls_mix_wind.rs:106` | `struct` | `MixWindControls` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/controls_output.rs:8` | `struct` | `OutputControls` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/controls_timestep_stop.rs:14` | `struct` | `TimestepStopControls` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/core_boundaries.rs:26` | `struct` | `CoreBoundaryInfo` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/core_boundaries.rs:51` | `struct` | `CoreMasses` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/core_boundaries.rs:87` | `struct` | `PowerCategories` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/core_boundaries.rs:144` | `fn` | `k_for_q_1based` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/core_boundaries.rs:194` | `fn` | `get_info_at_q` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/core_boundaries.rs:540` | `fn` | `set_power_info` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/core_boundaries.rs:553` | `fn` | `snapshot_step_history_old` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/create_initial_model.rs:320` | `fn` | `build_initial_model` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/create_initial_model.rs:345` | `fn` | `build_initial_model_with_configs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:51` | `struct` | `BurgersZoneResult` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:67` | `enum` | `IsolveSolver` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:342` | `struct` | `SolveDiffusionInput` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:493` | `struct` | `SolveDiffusionOutput` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:524` | `struct` | `SolveDiffusionStats` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:547` | `fn` | `do_solve_diffusion` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:1694` | `fn` | `get_timescale` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:1785` | `fn` | `get_dx_dt` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:1824` | `fn` | `get1_dx_dt` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:1904` | `fn` | `get_eqn_matrix_entries` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:1960` | `fn` | `get1_eqn_matrix_entries` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:2119` | `fn` | `positivity` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:2155` | `fn` | `set_diffusion_classes` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:2187` | `fn` | `adjust_timestep` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:2236` | `fn` | `eval_effective_kt` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion.rs:2255` | `fn` | `eval_coulomb_collision_factors` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_procs.rs:15` | `struct` | `FixupDiagnostics` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_procs.rs:46` | `fn` | `fixup` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_procs.rs:150` | `fn` | `fix_single_point_extremes` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_procs.rs:263` | `fn` | `fix_negative_masses` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_procs.rs:357` | `fn` | `fix_species_conservation` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_procs.rs:426` | `fn` | `smooth_x` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_procs.rs:566` | `fn` | `get1_dx_dm` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_procs.rs:743` | `fn` | `redistribute_mass` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_procs.rs:966` | `fn` | `get_limit_coeffs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_procs.rs:1029` | `fn` | `setup_struct_info` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_procs.rs:1174` | `fn` | `set_new_xa` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:33` | `struct` | `MatrixCoeffsInput` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:128` | `struct` | `MatrixCoeffsOutput` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:191` | `fn` | `get_matrix_coeffs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:444` | `fn` | `get1_cxzn_face` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:500` | `fn` | `get1_coeffs_face` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:646` | `fn` | `get1_burgers_coeffs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:796` | `fn` | `get1_gradient_coeffs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:1030` | `fn` | `get1_diffusion_velocities` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:1184` | `fn` | `get1_flow_coeffs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:1216` | `fn` | `do1_solve_thoul_hu` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:1437` | `fn` | `solve_burgers_cgs_no_thermal` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:1552` | `fn` | `solve_burgers_cgs_with_thermal` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:1746` | `fn` | `get_sm_coeffs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:1855` | `fn` | `lam_sm` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:1911` | `fn` | `get_cbf_coeffs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:1959` | `fn` | `get_dstar_ocp` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:1990` | `fn` | `kappa_cbf` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/diffusion_support.rs:2031` | `fn` | `bitsafe_erf_fit` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/element_diffusion.rs:53` | `fn` | `do_element_diffusion_step` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/element_diffusion.rs:631` | `fn` | `finish_element_diffusion` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/eps_grav.rs:228` | `fn` | `eval_eps_grav_std_cell` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/eps_grav.rs:243` | `fn` | `eval_eps_grav_lns_cell` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/eps_grav.rs:356` | `fn` | `eval_eps_grav` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/eps_mdot.rs:25` | `fn` | `eval_cell_specific_total_energy` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/eps_mdot.rs:33` | `fn` | `eval_specific_total_energy_profile` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/eps_mdot.rs:44` | `fn` | `find_mass_flux` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/eps_mdot.rs:56` | `struct` | `AdjustMassRecord` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/eps_mdot.rs:107` | `fn` | `total_energy_profile_before_adjust_mass` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/eps_mdot.rs:118` | `fn` | `eval_total_energy_profile` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/eps_mdot.rs:374` | `struct` | `EpsMdotSurface` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/eps_mdot.rs:397` | `fn` | `calculate_eps_mdot` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/evolve.rs:58` | `struct` | `StarStepPart1State` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/evolve.rs:74` | `struct` | `StarStepConverged` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/evolve.rs:90` | `enum` | `StarStepAttemptOutcome` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/evolve.rs:112` | `enum` | `StarPickTimestepOutcome` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/evolve.rs:130` | `fn` | `restore_from_old` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/evolve.rs:268` | `fn` | `star_evolve_step_with_extras` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/evolve.rs:305` | `fn` | `star_evolve_step_part1` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/evolve.rs:628` | `fn` | `star_evolve_step_part2` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/evolve.rs:859` | `fn` | `star_pick_next_timestep_after_step` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/evolve.rs:913` | `fn` | `star_finalize_terminated_step` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/extend_net.rs:71` | `fn` | `extend_net` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/history.rs:16` | `struct` | `HistoryRow` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/history.rs:149` | `fn` | `format_history_header` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_brunt_surf.rs:37` | `fn` | `do_brunt_b` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_brunt_surf.rs:182` | `fn` | `do_brunt_n2` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_brunt_surf.rs:303` | `fn` | `eval_surf_bc` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_energy.rs:47` | `fn` | `eval_pvsc_cell_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_energy.rs:120` | `fn` | `eval_cell_pressure_for_work_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_energy.rs:161` | `fn` | `setup_dl_dm` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_energy.rs:250` | `fn` | `setup_sources_and_others` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_energy.rs:395` | `fn` | `setup_d_turbulent_energy_dt` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_energy.rs:445` | `fn` | `eval_face_area_time_centered_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_energy.rs:476` | `fn` | `set_energy_eqn_scal` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_energy.rs:504` | `fn` | `eval_surface_and_center_work` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_energy.rs:547` | `fn` | `do1_energy_eqn` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_eqns.rs:35` | `struct` | `HenyeySystem` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_eqns.rs:168` | `fn` | `compute_x_scale_into` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_eqns.rs:242` | `fn` | `compute_x_scale` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_eqns.rs:256` | `fn` | `do1_density_eqn` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_eqns.rs:290` | `struct` | `SurfBcAd` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_eqns.rs:306` | `fn` | `get_pt_bc_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_eqns.rs:546` | `fn` | `do_surf_bc_eqns` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_eqns.rs:631` | `fn` | `do1_chem_eqns` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_eqns.rs:858` | `fn` | `assemble_henyey_system` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_eqns.rs:934` | `fn` | `eval_henyey_equations_into` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:48` | `fn` | `eval_p_theta` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:68` | `fn` | `eval_mlt_pturb_cell_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:93` | `fn` | `eval_time_centered_hse_geom_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:119` | `fn` | `has_surf_momentum_bc` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:129` | `fn` | `eval_surf_momentum_bc_pressure_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:158` | `fn` | `wrap_peos_00` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:168` | `fn` | `wrap_peos_m1` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:178` | `fn` | `wrap_r_00` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:189` | `fn` | `get_area_info_opt_time_center` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:216` | `fn` | `expected_hse_grav_and_area` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:235` | `fn` | `eval_cell_peos_time_centered_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:255` | `fn` | `eval_face_dm_bar` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:277` | `fn` | `expected_hse_grav_term` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:345` | `fn` | `wrap_d_00` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:353` | `fn` | `wrap_d_m1` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:366` | `fn` | `expected_non_hse_term` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:429` | `fn` | `calc_ptot_ad_tw` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:451` | `fn` | `do1_momentum_eqn` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_momentum.rs:565` | `fn` | `do1_radius_eqn` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:22` | `fn` | `rpsi_from_re_factor` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:30` | `fn` | `re_from_rpsi_factor` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:38` | `fn` | `roche_a` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:46` | `fn` | `roche_b` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:54` | `fn` | `roche_c` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:62` | `fn` | `roche_sigmoid` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:74` | `fn` | `w_div_w_roche_omega` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:153` | `fn` | `w_div_w_roche_jrot` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:242` | `fn` | `eval_i_rot_dual` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:274` | `fn` | `eval_i_rot` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:288` | `fn` | `eval_fp_ft_single_dual` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:306` | `fn` | `eval_fp_ft_single` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:315` | `fn` | `eval_fp_ft` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:329` | `fn` | `set1_i_rot` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:349` | `fn` | `update1_i_rot_from_xh` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:357` | `fn` | `set_i_rot` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:391` | `fn` | `set_i_rot_from_omega_and_j_rot` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:411` | `fn` | `set_j_rot` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:424` | `fn` | `set_omega` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:439` | `fn` | `use_xh_to_update_i_rot` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:471` | `fn` | `use_xh_to_update_i_rot_and_j_rot` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:524` | `fn` | `set_rotation_info` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:549` | `fn` | `get1_am_sig` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:600` | `fn` | `get_rotation_sigmas` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rotation.rs:638` | `fn` | `total_angular_momentum` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rti.rs:34` | `fn` | `set_dpdr_drhodr_info` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rti.rs:175` | `fn` | `set_rti_mixing_info` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rti.rs:278` | `fn` | `augment_mixing_with_rti` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rti.rs:301` | `fn` | `eval_alpha_rti_sources` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_rti.rs:374` | `fn` | `do1_dalpha_rti_dt_eqn` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_state.rs:17` | `struct` | `StepHistoryScalars` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_state.rs:268` | `struct` | `TidalShellDeformation` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_state.rs:288` | `trait` | `TidalDeformationHook` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_state.rs:302` | `struct` | `TidalDeformationHandle` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_state.rs:315` | `trait` | `BinaryOtherTorqueHook` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_state.rs:329` | `struct` | `BinaryOtherTorqueHandle` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_state.rs:528` | `struct` | `HydroAuxState` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_temperature.rs:41` | `fn` | `interp_p_point_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_temperature.rs:54` | `fn` | `set_rsp_lsurf_bc` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_temperature.rs:172` | `fn` | `do1_temperature_eqn` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_vars.rs:112` | `fn` | `update_sig_from_d_mix` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_vars.rs:1282` | `fn` | `check_for_redo_mlt` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_work.rs:24` | `fn` | `eval_cell_dpe_dt_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_work.rs:42` | `fn` | `eval_dke_dt_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_work.rs:52` | `fn` | `eval_v_flag_dke_dt_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_work.rs:64` | `fn` | `eval1_a_times_v_face_v_flag_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_work.rs:123` | `fn` | `eval1_work_v_flag_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_work.rs:287` | `struct` | `DworkDxa` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_work.rs:301` | `fn` | `eval_v_flag_dwork_dm_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/hydro_work.rs:399` | `fn` | `eval_v_flag_simple_pdv_dm_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/inlist.rs:31` | `struct` | `SectionControls` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/inlist_inventory.rs:18` | `struct` | `StarControlInventory` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/inlist_inventory.rs:53` | `fn` | `supported_star_control_inventory` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/inlist_runner.rs:376` | `fn` | `photo_restart_filename` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/inlist_runner.rs:405` | `fn` | `termination_code_string_okay` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/inlist_runner.rs:435` | `fn` | `rel_run_e_err_exceeds_warn_limit` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/inlist_runner.rs:453` | `fn` | `virial_thm_rel_err_exceeds_warn_limit` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/inlist_runner.rs:571` | `fn` | `write_pulse_data_to_case_dir` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/inlist_runner.rs:606` | `struct` | `StageRunReport` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/inlist_runner.rs:827` | `fn` | `run_inlist_stage` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/inlist_runner.rs:842` | `fn` | `run_inlist_stage_with_logs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/inlist_runner.rs:1299` | `fn` | `run_test_suite_case` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/inlist_runner.rs:1322` | `fn` | `run_test_suite_case_with_logs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/inlist_runner.rs:1344` | `fn` | `run_test_suite_case_with_options` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/job_controls.rs:90` | `enum` | `AtmIrradiatedOpacity` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/job_controls.rs:100` | `struct` | `StepFailureInfo` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/job_controls.rs:229` | `enum` | `EnergyEqnOption` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/job_controls.rs:242` | `struct` | `VarLayout` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mass_utils.rs:28` | `fn` | `one_based` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mass_utils.rs:37` | `fn` | `zero_slot` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mass_utils.rs:43` | `fn` | `reconstruct_m` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mass_utils.rs:53` | `fn` | `reconstruct_xm` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mass_utils.rs:64` | `fn` | `accurate_mass_difference` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mass_utils.rs:90` | `struct` | `CompressedIntersect` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mass_utils.rs:112` | `fn` | `make_compressed_intersect` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mass_utils.rs:219` | `fn` | `find_i_ranges` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mass_utils.rs:237` | `struct` | `PassFractionRow` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mass_utils.rs:256` | `struct` | `PassFraction` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mass_utils.rs:269` | `fn` | `prepare_pass_fraction` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mass_utils.rs:322` | `fn` | `compute_pass_fraction` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mass_utils.rs:338` | `fn` | `find_j_ranges` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mass_utils.rs:394` | `fn` | `integrate_conserved` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mdot_redo.rs:13` | `enum` | `SelectMdotAction` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mdot_redo.rs:27` | `fn` | `eval_surf_avg_w_div_w_crit` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mdot_redo.rs:64` | `struct` | `MdotRedoState` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:104` | `enum` | `MesaCellValue` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:125` | `enum` | `HistoryIsoQuery` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:153` | `enum` | `HistoryColumnSpec` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:770` | `fn` | `parse_star_history_columns` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:1073` | `fn` | `load_star_colors_context` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:1103` | `fn` | `resolve_star_history_columns` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:1158` | `fn` | `default_star_history_specs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:1171` | `fn` | `default_star_history_column_names` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:3101` | `fn` | `star_get_history_output` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:3123` | `fn` | `evaluate_star_history_row` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:3138` | `fn` | `evaluate_star_history_row_with_ctx` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:3444` | `fn` | `format_star_history_file_header` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:3454` | `fn` | `format_star_history_file_header_with_extra` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:3471` | `fn` | `format_star_history_file_header_with_extras` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:3522` | `fn` | `format_star_history_data_line` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:3531` | `fn` | `format_star_history_data_line_with_extra` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_history.rs:3558` | `struct` | `IncrementalHistoryWriter` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_log_reader.rs:27` | `struct` | `MesaLogTable` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_log_reader.rs:132` | `fn` | `read_mesa_log_str` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_log_reader.rs:237` | `fn` | `read_profiles_index_str` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_log_reader.rs:287` | `fn` | `read_profiles_index_file` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:61` | `enum` | `ProfileColumnSpec` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:332` | `fn` | `parse_star_profile_columns` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:476` | `fn` | `resolve_star_profile_columns` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:511` | `fn` | `default_star_profile_specs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:1444` | `fn` | `format_star_profile_text` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:1466` | `fn` | `format_star_profile_text_with_extras` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:1719` | `struct` | `ProfilesIndexEntry` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:1731` | `struct` | `ProfilesIndex` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:1934` | `struct` | `StarStepLogExtras` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:1953` | `struct` | `StarLogsWriter` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:2428` | `enum` | `RspMapColumn` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:2469` | `fn` | `parse_rsp_map_columns` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:2507` | `fn` | `resolve_rsp_map_column_names` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:2556` | `fn` | `format_rsp_map_header` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesa_profile.rs:2574` | `fn` | `format_rsp_map_rows` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_adjust.rs:52` | `struct` | `PlannedMeshSlot` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_adjust.rs:277` | `fn` | `eval_effective_mesh_delta_coeff` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_adjust.rs:380` | `fn` | `try_plan_standard_mesh` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_adjust.rs:529` | `fn` | `plan_mesh` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_functions.rs:52` | `struct` | `MeshFunctionData` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_functions.rs:81` | `fn` | `get_net_iso_0based` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_functions.rs:108` | `fn` | `num_mesh_functions_with_controls` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_functions.rs:178` | `fn` | `num_mesh_functions` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_functions.rs:264` | `fn` | `set_mesh_function_data_with_controls` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_functions.rs:527` | `fn` | `set_mesh_function_data` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_plan.rs:17` | `struct` | `MeshPlanOutput` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_plan.rs:874` | `fn` | `do_mesh_plan` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_remap.rs:29` | `enum` | `CellType` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_remap.rs:43` | `struct` | `StandardMeshPlan` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_remap.rs:183` | `fn` | `classify_new_cells` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_remap.rs:372` | `fn` | `total_angular_momentum` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_remap.rs:390` | `fn` | `get1_lpp_zone` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_remap.rs:443` | `fn` | `compute_monotone_xa_slopes` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_remap.rs:514` | `fn` | `get_xq_integral_col` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_remap.rs:598` | `fn` | `get_old_value_integral` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_remap.rs:648` | `fn` | `solve_cell_lnt_for_energy` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_remap.rs:756` | `fn` | `do_interp_cell_val` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_remap.rs:1877` | `fn` | `remap_onto_dq` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mesh_remap.rs:1926` | `fn` | `do_prune_mesh_surface` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro.rs:150` | `fn` | `global_basic_net` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro.rs:176` | `fn` | `get_or_load_net_with_fe56ec` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro.rs:311` | `fn` | `resolve_model_net` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro.rs:516` | `fn` | `change_model_network` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro.rs:532` | `fn` | `change_model_network_with_adjust` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro.rs:698` | `fn` | `set_basic_vars` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro.rs:1082` | `fn` | `do_eos` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro.rs:1247` | `fn` | `fix_d_eos_dxa_partials` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro.rs:1400` | `fn` | `do_neu` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro.rs:1498` | `fn` | `do_kap` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro.rs:1554` | `fn` | `check_tau_with_report_ierr` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro_cache.rs:683` | `struct` | `MicrophysicsCache` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro_net.rs:27` | `fn` | `build_rate_factors` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro_net.rs:201` | `fn` | `high_temp_rate_warning_zone` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/micro_net.rs:236` | `fn` | `do_net` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mix_info.rs:26` | `fn` | `face_density` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mix_info.rs:43` | `fn` | `cdc_factor` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mix_info.rs:60` | `fn` | `set_mixing_info` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mix_info.rs:95` | `fn` | `set_mixing_info_with_skip` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mix_info.rs:318` | `fn` | `clean_mixing_glitches_from_controls` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mix_info.rs:671` | `fn` | `do_mix_envelope` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mix_info.rs:724` | `fn` | `set_cz_boundary_info` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mix_info.rs:781` | `fn` | `set_cz_bdy_mass` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mix_info.rs:1083` | `fn` | `apply_min_d_mix_floor` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mix_info.rs:1576` | `fn` | `eval_lrad_div_ledd` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mix_info.rs:1653` | `fn` | `set_gradt_excess_alpha` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mix_info.rs:1729` | `fn` | `adjust_gradt_excess` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mix_info.rs:1767` | `fn` | `adjust_gradt_fraction` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mixing_state.rs:21` | `struct` | `MixingAuxState` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mixing_state.rs:463` | `fn` | `weighted_smoothing` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mixing_state.rs:517` | `fn` | `threshold_smoothing` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mixing_state.rs:567` | `fn` | `set_grads` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/mixing_state.rs:668` | `fn` | `driving_species_for_cid` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/model.rs:36` | `struct` | `SurfPt` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/model_io.rs:275` | `fn` | `read_mod_str` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/omega_mix.rs:35` | `enum` | `OmegaMixOutcome` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/omega_mix.rs:70` | `fn` | `do_solve_omega_mix` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/omega_mix.rs:97` | `fn` | `do_solve_omega_mix_with_extras` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/overshoot.rs:89` | `fn` | `eval_conv_bdy_k` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/overshoot.rs:126` | `fn` | `eval_conv_bdy_r` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/overshoot.rs:680` | `fn` | `add_overshooting` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/paquette_coeffs.rs:21` | `fn` | `paquette_coefficients` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/phase_separation.rs:32` | `fn` | `blouin_delta_xo` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/phase_separation.rs:58` | `fn` | `blouin_delta_xne` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/phase_separation.rs:113` | `fn` | `do_phase_separation_step` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/phase_separation.rs:590` | `fn` | `apply_phase_sep_mixing_types` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/photo_io.rs:18` | `fn` | `compute_photo_checksum` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/photo_io.rs:69` | `fn` | `write_photo_bytes` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/photo_io.rs:78` | `fn` | `photo_numerics_mode` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/photo_io.rs:114` | `fn` | `read_photo_bytes` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pre_ms.rs:771` | `fn` | `build_pre_ms_model_with_configs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/premix.rs:111` | `fn` | `apply_predictive_mixing_from_controls` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/primary_state.rs:361` | `struct` | `PrimaryStateBuffer` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse.rs:28` | `fn` | `build_pulse_atmosphere_points` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse.rs:110` | `fn` | `export_pulse_data_text` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse.rs:163` | `fn` | `default_fgong_header` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_bvp.rs:29` | `struct` | `BvpPoint` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_bvp.rs:51` | `struct` | `BvpModel` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_bvp.rs:62` | `struct` | `BvpEigenmode` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_bvp.rs:82` | `fn` | `steffen_slopes` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_bvp.rs:398` | `fn` | `shoot_determinant` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_bvp_recon.rs:20` | `fn` | `reconstruct_and_analyze_mode` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_bvp_recon.rs:302` | `fn` | `classify_mode_orders` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_bvp_recon.rs:361` | `fn` | `parse_fortran_float` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_bvp_recon.rs:527` | `fn` | `parse_fgong_model` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_exporters.rs:43` | `fn` | `fortran_e_no_1p` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_exporters.rs:90` | `struct` | `GsmData` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_exporters.rs:156` | `fn` | `try_export_gsm_data` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_exporters.rs:226` | `fn` | `export_gsm_data` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_exporters.rs:303` | `fn` | `get_saio_data_strict` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_exporters.rs:339` | `fn` | `write_saio_data` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_exporters.rs:366` | `fn` | `export_saio_strict` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_exporters.rs:616` | `fn` | `get_cafein_data` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_exporters.rs:925` | `fn` | `write_cafein_data` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_exporters.rs:981` | `fn` | `get_gr1d_data_strict` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_exporters.rs:1020` | `fn` | `export_gr1d_strict` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_exporters.rs:1132` | `fn` | `export_pulse_with_controls` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_fgong.rs:44` | `struct` | `PulseDataInput` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_fgong.rs:100` | `struct` | `PulseAtmInput` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_fgong.rs:176` | `struct` | `PulseIsoIndices` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_fgong.rs:791` | `fn` | `get_fgong_data` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_gyre_data.rs:27` | `enum` | `GyreDataError` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_gyre_data.rs:55` | `struct` | `GyreDataInput` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_gyre_data.rs:391` | `fn` | `get_gyre_data` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_modes.rs:25` | `fn` | `scan_bounds_from_targets` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_modes.rs:63` | `fn` | `build_bvp_model_from_pulse_points` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_modes.rs:254` | `fn` | `eval_acoustic_large_separation_hz` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_modes.rs:284` | `fn` | `eval_buoyancy_period_spacing_s` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_modes.rs:307` | `fn` | `eval_kappa_mechanism_growth` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_osc.rs:29` | `fn` | `get_osc_data` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_utils.rs:23` | `struct` | `PulseExportOptions` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_utils.rs:50` | `struct` | `PulsePoint` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_utils.rs:115` | `fn` | `eval_center_d2` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_utils.rs:126` | `fn` | `eval_center_rho` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_utils.rs:144` | `fn` | `set_segment_indices` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_utils.rs:195` | `fn` | `eval_face` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_utils.rs:256` | `fn` | `eval_face_a_ast` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_utils.rs:483` | `fn` | `build_pulse_points_with_options` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_write.rs:15` | `enum` | `PulseWriteError` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_write.rs:40` | `fn` | `fortran_1pe` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_write.rs:115` | `fn` | `write_gyre_data` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_write.rs:162` | `fn` | `write_fgong_data` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/pulse_write.rs:217` | `fn` | `write_osc_data` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax.rs:29` | `fn` | `set_uniform_initial_composition` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax.rs:67` | `fn` | `apply_uniform_rotation` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax.rs:131` | `fn` | `excise_core_by_temperature` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:81` | `fn` | `eval_current_y_and_z` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:133` | `fn` | `eval_current_y` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:150` | `fn` | `eval_current_z` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:169` | `fn` | `eval_current_abundance` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:210` | `fn` | `set_abundance_ratio` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:263` | `fn` | `set_y` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:323` | `fn` | `set_z` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:431` | `struct` | `CompositionProfile` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:455` | `fn` | `parse_composition_profile_str` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:567` | `struct` | `RelaxCompositionInterpolants` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:781` | `fn` | `cell_center_xq` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:806` | `struct` | `ScalarProfile` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:819` | `enum` | `EntropyProfileSource` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:906` | `fn` | `parse_scalar_profile_str` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:952` | `fn` | `parse_angular_momentum_profile_str` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:963` | `fn` | `parse_entropy_profile_str` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:976` | `fn` | `parse_entropy_profile_with_eos_str` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_comp.rs:1150` | `struct` | `RelaxScalarInterpolant` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_entropy.rs:33` | `struct` | `RelaxEntropyTarget` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/relax_entropy.rs:130` | `fn` | `parse_entropy_profile_str` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/remove_center.rs:30` | `fn` | `do_remove_center` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/remove_center.rs:126` | `fn` | `do_remove_inner_fraction_q` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/remove_center.rs:161` | `fn` | `do_remove_center_by_entropy` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/remove_surface_cut.rs:39` | `fn` | `remove_surface_at_cell_k` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/remove_surface_cut.rs:97` | `fn` | `do_relax_to_star_cut` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:49` | `fn` | `write_mesa_dbl_width` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:79` | `fn` | `format_mesa_dbl_width` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:88` | `fn` | `format_mesa_dbl` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:94` | `fn` | `write_mesa_int_width` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:101` | `fn` | `format_mesa_int_width` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:108` | `fn` | `format_mesa_int` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:114` | `fn` | `write_mesa_txt_width` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:121` | `fn` | `format_mesa_txt_width` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:128` | `fn` | `format_mesa_txt` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:136` | `fn` | `format_mesa_header_str_width` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:144` | `fn` | `format_mesa_header_str` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:150` | `fn` | `push_mesa_6line_header` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:183` | `fn` | `parse_quoted_include_arg` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:205` | `fn` | `center_value_by` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:254` | `fn` | `iso_slot_for_chem_id` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:265` | `fn` | `resolve_iso_slot` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:285` | `fn` | `center_avg_x_slot` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:341` | `fn` | `star_avg_x_slot` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:387` | `fn` | `csound_face_at` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:402` | `fn` | `equatorial_and_polar_r_cm` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:420` | `fn` | `get_ledd` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:441` | `fn` | `get_lrad_div_ledd` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:460` | `fn` | `eval_upstream_omega_crit` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:491` | `fn` | `eval_conv_time_scale_zone` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:536` | `fn` | `eval_conv_time_scales` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:578` | `fn` | `eval_ledd_erg_s` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:610` | `struct` | `PhotosphereInfo` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:770` | `struct` | `SurfAvgRotationInfo` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:803` | `fn` | `star_surface_omega_crit` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:948` | `fn` | `eval_cz_bdy_dq` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:990` | `struct` | `ConvAndMixSummary` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:1181` | `fn` | `eval_epsnuc_zones` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:1397` | `fn` | `eval_history_mixing_regions` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:1498` | `fn` | `eval_history_burning_regions` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:1557` | `fn` | `eval_astero_report_scalars` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:1659` | `struct` | `StepReportContext` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:1809` | `struct` | `ShockInfo` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:1839` | `fn` | `eval_shock_info` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:2195` | `fn` | `gravity_darkening_table` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:2213` | `fn` | `gravity_darkening_teff_coeff` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:2221` | `fn` | `gravity_darkening_l_coeff` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/report_utils.rs:2229` | `fn` | `eval_surface_w_div_w_kep` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/riemann.rs:27` | `struct` | `HllcFaceState` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/riemann_ad.rs:36` | `fn` | `calc_ptot_cell_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/riemann_ad.rs:105` | `fn` | `do_uface_and_pface` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/riemann_eqns.rs:20` | `fn` | `do1_riemann_momentum_eqn` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/riemann_eqns.rs:241` | `fn` | `eval_u_flag_total_energy_rates_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rotation.rs:17` | `struct` | `RotationalMixZone` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rotation.rs:106` | `fn` | `eval_thermal_diffusivity` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rotation.rs:121` | `fn` | `eval_magnetic_diffusivity` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rotation.rs:348` | `fn` | `update_rotation_distortion_factors` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rotation.rs:408` | `fn` | `conserve_shell_j_rot_and_distortion` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rotation.rs:426` | `fn` | `augment_d_mix_with_rotation` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rotation.rs:437` | `fn` | `sync_post_henyey_rotation` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rotation_mix_info.rs:27` | `struct` | `RotMixArrays` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rotation_mix_info.rs:97` | `struct` | `StStartInfo` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rotation_mix_info.rs:118` | `fn` | `store_st_start_info` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rotation_mix_info.rs:138` | `fn` | `store_prev_mesh_st_info` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp2.rs:93` | `struct` | `Rsp2ZoneResult` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp2.rs:118` | `struct` | `Rsp2FaceResult` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp2_ad.rs:92` | `fn` | `eval_rsp2_ptrb_cell_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp2_ad.rs:129` | `fn` | `eval_rsp2_hp_face_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp2_ad.rs:509` | `fn` | `update_rsp2_vars_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp2_ad.rs:515` | `fn` | `eval_and_save_rsp2_eqns` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp2_eqns.rs:20` | `struct` | `Rsp2ProfileResiduals` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp2_eqns.rs:83` | `fn` | `eval_rsp2_hp_face_zone` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp2_eqns.rs:109` | `fn` | `eval_rsp2_profile_equations` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp2_eqns.rs:218` | `fn` | `rsp2_adjust_vars_before_call_solver` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp2_remesh.rs:19` | `fn` | `seed_rsp2_w_from_mlt_vc_if_needed` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp2_remesh.rs:56` | `fn` | `remesh_for_rsp2` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_build.rs:25` | `struct` | `RspLinaMode` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_build.rs:258` | `fn` | `eval_model_rsp_lina_modes` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_config.rs:12` | `struct` | `RspConfig` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_config.rs:318` | `struct` | `Rsp2Controls` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_def.rs:24` | `struct` | `RspDerivedConsts` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_def.rs:224` | `struct` | `RspState` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_driver.rs:242` | `fn` | `rsp_setup_model` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_driver.rs:381` | `fn` | `rsp_total_energy_integrals` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_driver.rs:394` | `fn` | `rsp_one_step` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_driver.rs:408` | `fn` | `rsp_one_step_with_extras` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_driver.rs:672` | `fn` | `rsp_after_step_map` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eos_inv.rs:29` | `fn` | `eval1_gamma_pt_get_rho` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eos_inv.rs:35` | `fn` | `eval1_mesa_rho_given_pt` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eos_inv.rs:77` | `fn` | `eval1_mesa_t_given_dp` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eos_inv.rs:120` | `fn` | `eval1_mesa_eos_degas_and_kap` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eos_inv.rs:152` | `fn` | `eval1_mesa_eos_de_and_kap` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eos_inv.rs:184` | `fn` | `set_rho_for_new_pgas` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eos_inv.rs:212` | `fn` | `set_t_for_new_egas` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eos_inv.rs:241` | `fn` | `set_t_for_new_pgas` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eos_inv.rs:269` | `fn` | `set_t_for_new_energy` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eos_inv.rs:306` | `fn` | `get_surf_p_t_kap` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eos_kap.rs:65` | `struct` | `RspEosKapOut` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eos_kap.rs:223` | `fn` | `compute_standard_metals_xa` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eos_kap.rs:446` | `fn` | `eval1_mesa_eos_and_kap` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eos_kap.rs:530` | `fn` | `eval_mesa_eos_and_kap` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eqns.rs:59` | `struct` | `RspHydBand` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eqns.rs:101` | `fn` | `eval_eqns` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_eqns.rs:158` | `fn` | `solve_for_corrections` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_grid.rs:21` | `struct` | `MesaRanmar` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_grid.rs:210` | `struct` | `RspEnvGrid` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_grid.rs:277` | `struct` | `RspTotalEosKap` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_grid.rs:311` | `fn` | `mesa_eos_kap` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_grid.rs:341` | `fn` | `eop` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_grid.rs:382` | `fn` | `cleanup_for_lina` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_grid.rs:430` | `fn` | `copy_results` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_grid.rs:507` | `fn` | `finish_build_rsp_model` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_grid.rs:561` | `fn` | `copy_from_xh_to_rsp` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_grid.rs:584` | `fn` | `set_star_vars_rsp` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_grid.rs:640` | `fn` | `finish_after_build_model` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_hyd.rs:19` | `fn` | `init_hyd` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_hyd.rs:39` | `fn` | `hyd` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_hyd.rs:127` | `fn` | `get_psurf` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_hyd.rs:156` | `fn` | `calculate_energies` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_hyd.rs:219` | `fn` | `rsp_set_teff` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_hyd_turb.rs:14` | `fn` | `calc_chi` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_hyd_turb.rs:92` | `fn` | `calc_eq` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_hyd_turb.rs:151` | `fn` | `calc_source_sink` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_hyd_vars.rs:18` | `fn` | `eval_vars` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_hyd_vars.rs:55` | `fn` | `do1_specific_volume` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_hyd_vars.rs:83` | `fn` | `do1_eos_and_kap` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_hyd_work.rs:12` | `struct` | `RspHydWork` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_hyd_work.rs:245` | `fn` | `zero_boundaries` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_lina.rs:21` | `struct` | `RspLinaOutput` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_lina.rs:38` | `fn` | `do_lina` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_relax_env.rs:23` | `fn` | `relax_env` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_stah.rs:15` | `fn` | `znvar` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_stah.rs:172` | `fn` | `stah` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_turb_partials.rs:18` | `struct` | `RspTurbPartials` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_zone_micro.rs:12` | `struct` | `RspZoneMicro` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_zone_micro.rs:107` | `enum` | `LcFilterMode` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_zone_micro.rs:124` | `struct` | `HpFacePartials` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/rsp_zone_micro.rs:139` | `fn` | `eval_hp_face_partials` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver.rs:58` | `fn` | `compute_corr_coeff` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_norms.rs:13` | `struct` | `FlailingTracker` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_norms.rs:79` | `fn` | `compute_vector_norms_with_argmax` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_norms.rs:105` | `fn` | `compute_vector_norms` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_norms.rs:119` | `fn` | `compute_residual_norms_with_argmax` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_norms.rs:213` | `fn` | `compute_residual_norms` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_norms.rs:225` | `fn` | `compute_correction_norms` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_norms.rs:245` | `fn` | `compute_model_correction_norms_with_argmax` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_norms.rs:258` | `fn` | `compute_model_correction_norms_in_solver` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_norms.rs:449` | `fn` | `compute_model_correction_norms` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_norms.rs:466` | `fn` | `active_correction_tolerances` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_norms.rs:501` | `fn` | `active_gold_level` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_norms.rs:553` | `fn` | `active_max_tries` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_norms.rs:573` | `fn` | `active_residual_tolerances_with_gold` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_norms.rs:669` | `fn` | `active_solver_tolerances` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_support.rs:19` | `fn` | `current_min_xa_hard_limit` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_support.rs:42` | `fn` | `current_sum_xa_hard_limit` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_support.rs:95` | `fn` | `apply_bdomain` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_support.rs:207` | `fn` | `force_another_iteration` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_support.rs:242` | `fn` | `set_vars_for_solver` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_support.rs:587` | `fn` | `check_after_converge` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/solver_support.rs:625` | `fn` | `rti_check_after_converge` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_ad.rs:77` | `type` | `StarAdOrder1` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_ad.rs:81` | `trait` | `StarAdExt` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_job_init.rs:413` | `fn` | `init_or_update_model_from_bundle` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_job_init.rs:555` | `fn` | `init_model_from_zams_seed` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:36` | `fn` | `star_new_model_for_build` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:70` | `fn` | `star_normalize_dqs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:75` | `fn` | `star_set_qs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:80` | `fn` | `star_set_m_and_dm` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:85` | `fn` | `star_set_dm_bar` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:95` | `fn` | `change_to_xa_for_accretion` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:114` | `fn` | `star_basic_composition_info` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:145` | `fn` | `star_get_atm_pt` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:174` | `fn` | `star_get_surf_pt` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:202` | `fn` | `star_solve_eos_given_pgas_t` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:241` | `fn` | `star_solve_eos_given_pgas_t_auto` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:285` | `fn` | `star_do_eos_for_cell` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:312` | `fn` | `star_do_kap_for_cell` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:325` | `fn` | `star_set_mlt_vars` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_lib.rs:338` | `fn` | `star_write_model` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_utils.rs:18` | `struct` | `NonPositiveRho` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_utils.rs:52` | `fn` | `use_xh_to_set_rho_to_dm_div_dv` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/star_utils.rs:97` | `fn` | `set_rho_to_dm_div_dv_for_new_step` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/stella.rs:50` | `struct` | `StellaHydZone` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/stella.rs:67` | `struct` | `StellaHydModel` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/stella.rs:78` | `struct` | `StellaAbnZone` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/stella.rs:88` | `struct` | `StellaVelFeIIRow` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/stella.rs:503` | `fn` | `parse_stella_lbol` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/step_trace.rs:22` | `fn` | `enabled` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/step_trace.rs:34` | `fn` | `emit_raw_json_line` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/step_trace.rs:77` | `struct` | `JsonLine` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/step_trace.rs:142` | `fn` | `argmax_abs` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/stepdump.rs:183` | `fn` | `begin_stage` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/stepdump.rs:194` | `fn` | `stop_after_window` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/stepdump.rs:203` | `fn` | `selected_stage_ended` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/stepdump.rs:212` | `fn` | `active` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/stepdump.rs:217` | `fn` | `with_star_id` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/stepdump.rs:229` | `fn` | `record` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/stepdump.rs:292` | `fn` | `push_hex` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/stepdump.rs:298` | `fn` | `push_line` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/struct_burn_mix.rs:71` | `fn` | `op_split_burn_zones` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/tdc_hydro.rs:283` | `fn` | `get_tdc_alfa_beta_face_weights` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/tdc_hydro.rs:489` | `fn` | `compute_tdc_eq_div_w_face` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/tdc_hydro.rs:510` | `fn` | `compute_tdc_uq_face` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/tdc_hydro.rs:531` | `fn` | `compute_tdc_uq_dm_cell` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/tdc_hydro.rs:557` | `fn` | `tdc_energy_eq_ad` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/termination.rs:63` | `enum` | `LimitCheckOutcome` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/termination.rs:92` | `fn` | `compute_core_masses_msun` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/termination.rs:116` | `fn` | `get_ejecta_mass` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/termination.rs:165` | `fn` | `get_remnant_mass` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/termination.rs:175` | `fn` | `get_shock_mass_msun` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/termination.rs:253` | `fn` | `eval_min_gamma1` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/termination.rs:497` | `fn` | `eval_phase_of_evolution` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/termination.rs:508` | `fn` | `set_phase_of_evolution` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/termination.rs:1867` | `fn` | `do_check_limits` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/termination.rs:1896` | `fn` | `check_termination_limits` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/termination.rs:1909` | `fn` | `check_termination_limits_with_history` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/timestep.rs:24` | `enum` | `TimestepAction` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/timestep.rs:36` | `enum` | `TimestepLimitReason` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/timestep.rs:187` | `struct` | `TimestepReport` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/timestep.rs:206` | `fn` | `filter_limit` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/timestep.rs:287` | `fn` | `h211_limtr` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/timestep_limits.rs:35` | `fn` | `timestep_controller` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/winds.rs:106` | `fn` | `eval_reimers_wind` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/winds.rs:122` | `fn` | `eval_blocker_wind` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/winds.rs:142` | `fn` | `eval_vink_wind` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/winds.rs:213` | `fn` | `eval_bjorklund_wind` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/winds.rs:246` | `fn` | `eval_nieuwenhuijzen_wind` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/winds.rs:266` | `fn` | `eval_de_jager_wind` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/winds.rs:280` | `fn` | `eval_van_loon_wind` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/winds.rs:294` | `fn` | `eval_nugis_lamers_wind` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/winds.rs:314` | `fn` | `eval_dutch_wind` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/zams.rs:26` | `struct` | `ZamsEntry` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/zams.rs:54` | `struct` | `ZamsCatalog` | `rust-only` |
| `flexmesa-star` | `crates/flexmesa-star/src/zams.rs:531` | `fn` | `load_zams_model_from_catalog` | `rust-only` |
| `flexmesa-turb` | `crates/flexmesa-turb/src/lib.rs:112` | `type` | `TurbResult` | `rust-only` |
| `flexmesa-turb` | `crates/flexmesa-turb/src/tdc_dual.rs:69` | `trait` | `TdcLift` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/bin_codec.rs:19` | `enum` | `BinCodecError` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/bin_codec.rs:75` | `fn` | `to_bytes` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/bin_codec.rs:90` | `fn` | `from_bytes` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/bin_codec.rs:112` | `fn` | `encode_framed_bytes` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/bin_codec.rs:132` | `fn` | `decode_framed_header` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/cache_envelope.rs:38` | `fn` | `write_verified_cache_atomic` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/cache_envelope.rs:207` | `fn` | `read_verified_cache` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/cache_envelope.rs:277` | `fn` | `open_verified_cache_streaming` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/cache_envelope.rs:360` | `fn` | `open_preverified_cache_streaming` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/lib.rs:181` | `fn` | `is_zero_skips_opt_out` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/lib.rs:192` | `fn` | `require_zero_skips_enabled` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/lib.rs:204` | `fn` | `require_mesa_test_input_path` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/lib.rs:512` | `struct` | `Records` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/sha256.rs:94` | `struct` | `Sha256Hasher` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/sha256.rs:238` | `fn` | `compute_sha256_reader` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/sha256.rs:255` | `fn` | `compute_sha256_raw` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/sha256.rs:262` | `fn` | `decode_sha256_hex` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/sha256.rs:279` | `fn` | `encode_sha256_hex` | `rust-only` |
| `flexmesa-utils` | `crates/flexmesa-utils/src/sha256.rs:291` | `fn` | `compute_sha256_file` | `rust-only` |
