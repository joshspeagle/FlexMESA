# Binary RLOF Test Fixture Reference Check (MESA 26.4.1)

This directory stores the inputs, reproduction script, and measured log summary
for the synthetic point-mass binary RLOF fixture used by `flexmesa-binary`
integration tests (`crates/flexmesa-binary/tests/common/mod.rs`).

## Reference Build

All numbers in `results.json` come from real Fortran MESA executed with
`OMP_NUM_THREADS=4` and recorded from `binary_history.data`'s own header:

- **MESA version**: `r26.4.1` (release `26.4.1`)
- **Compiler**: `gfortran 15.2.0`
- **MESA SDK**: `x86_64-linux-26.6.1`

## Why `build_rlof_binary()` Uses `(2.0 Msun, 1.5 Msun, P = 0.5045 d)`

The original fixture used an unrelaxed `1.5 Msun` ZAMS donor (`Z = 0.02`,
`m2 = 1.0 Msun`, `P = 0.45 d`, `mdot_scheme = 'Ritter'`,
`max_tries_to_achieve = 20`, `terminate_if_initial_overflow = .false.`,
`use_radiation_corrected_transfer_rate = .false.`, initial `dt = 1e3 yr`),
followed on step 2 by an artificial `fr_hard = 1e-5` hard limit and
`dt >= 1e4 yr` to force an outer rollback retry in `binary_counters_test`.

| Case | Engine | Step 1 `age` (yr) | Step 1 retries | Step 2 `hydro_mtx` bound retries | Step 2 `give up in solver` retries | Step 2 `fr_hard` retries | Outcome |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `(1.5, 1.0, 0.45 d)` | MESA `26.4.1` (`max_number_retries = 40`) | `125.0` | `3` (`1` bound + step-1 cuts) | `6` | `17` (`23` solver retries total) | `15` | `max_number_retries` (`min dt = 18.37 s`) |
| `(2.0, 1.5, 0.5045 d)` | MESA `26.4.1` | `1000.0` | `0` (`2` iters, `11` RLOF tries) | `0` | `7` (transient during `fr_hard` cuts) | `52` | `min_timestep_limit` (`5.47e-7 s`; remesh `dR/R = 4.44e-4 > 1e-5`) |
| `(1.5, 1.0, 0.45 d)` | FlexMESA probe (`152-159-audit-code-fixes`) | `125.0` (`3.945e9 s`) | `1` (`6` solver calls) | n/a | `16` total retries by step 2 (`42` calls) | n/a | `dt` collapses to `9.184 s` (step 2) and `1.794e-2 s` (step 3); `checkpoint_child_process_restart_test` times out at `120 s` |

On the unrelaxed `1.5 Msun` ZAMS setup, MESA `26.4.1` fails step 1's initial
`1e3 yr` attempt (`logT > hydro_mtx_max_allowed_logT` at zone `2318`), cuts `dt`
to `125 yr`, and then enters a persistent Newton rejection cascade on step 2
(`max_residual` rising to `132` against `1e-4`, terminating at
`max_number_retries = 40` with `23` solver retries). Switching the synthetic
fixture to `(2.0 Msun, 1.5 Msun, P = 0.5045 d)` gives a donor that converges
cleanly on step 1 in MESA `26.4.1` with zero retries and `2` solver iterations
while exercising implicit RLOF, rollback counters, thread determinism, and
child-process checkpoint restarts in `flexmesa-binary`.

## Reproducing

With `MESA_DIR` pointing to a built MESA `26.4.1` tree and `MESASDK_ROOT`
initialized (`source "$MESASDK_ROOT/bin/mesasdk_init.sh"`):

```bash
python benchmarks/fortran_mesa/binary_rlof_fixture/run_fixture_check.py \
  --mesa-dir "$MESA_DIR" \
  --out benchmarks/fortran_mesa/binary_rlof_fixture/results.json
```

The model-2 counts above were corrected on 2026-11-22 from the instrumented
run below. The earlier `7` bound and `24` solver retries also counted the one
bound retry of model 1, and the first bound failure is at zone `2318`.

## Instrumented Reference Outputs (`instrumented/`)

`instrumented/run_instrumented.py` builds a private overlay of a compiled MESA
`26.4.1` tree, adds `write` statements (and nothing else) to
`star/private/star_solver.f90` and `star/private/init_model.f90`, and reruns the
original `(1.5, 1.0, 0.45 d)` case. The instrumented run reproduces the
counts above (`44` donor solver calls in model 1, `82` in total). Each table
records the MESA version, compiler and SDK from `binary_history.data`, and
stores values as IEEE-754 binary64 bit patterns in hexadecimal:

| File | Contents |
| :--- | :--- |
| `zams_model0_26_4_1.tsv` | `lnd lnT lnR L` per zone right after ZAMS interpolation (`get_zams_model`) |
| `first_call_state_26_4_1.tsv` | `lnd lnT lnR L` and mass fractions per zone at the first solver call of model 1, after `prepare_for_new_step` reset `rho = dm/dV` (`star/private/evolve.f90:1854-1857`) |
| `solver_calls_26_4_1.tsv` | model, `dt`, initial `residual_norm` and `max_residual` of each solver call |
| `line_search_model1_26_4_1.tsv` | Newton and line-search records of model 1 (`min_corr_coeff`, `correction_factor`, `slope`, `f`, trial `alam`, `coeff`, `f_target`) |

`crates/flexmesa-binary/tests/binary_rlof_mesa_state_test.rs` compares the
first two tables with `FlexMESA` bit for bit. To regenerate them:

```bash
python benchmarks/fortran_mesa/binary_rlof_fixture/instrumented/run_instrumented.py \
  --mesa-dir "$MESA_DIR" --mesasdk-root "$MESASDK_ROOT" --scratch <empty directory>
```

