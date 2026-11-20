# `star/test_suite/hb_2M` `use_Ledoux_criterion` 2x2 Experiment (`#155`)

This directory records a reproducible 2x2 experiment comparing upstream MESA `26.4.1` and FlexMESA (at commit `aaa29e9`, `strict` numerics mode) on `star/test_suite/hb_2M` with `use_Ledoux_criterion = .true.` (unmodified upstream `inlist_hb_2M`) and `use_Ledoux_criterion = .false.` (single-line change in a scratch copy of `inlist_hb_2M`).

Both FlexMESA rows are **`diagnostic (2 controls ignored)`** runs produced with `--allow-unsupported-controls` and do not count as passing (`DECISIONS #26`).

## Files

- `run_ledoux_experiment.py`: portable runner and summarizer for all four cells (`run` and `summarize` subcommands).
- `results.json`: machine-readable 2x2 summary with MESA `26.4.1` version, compiler, SDK, and host provenance.
- `results.csv`: flat tabular export of the four cells in `results.json`.

## 2x2 Summary

All four runs started from upstream's stored `standard_2.0_ZACHeB.mod` (`MESA_SKIP_OPTIONAL=t`, `initial_age = 0`, `initial_model_number = 0`), ran with `--timeout-sec 5400` and `OMP_NUM_THREADS=4`, and reached upstream's central helium depletion stop `xa_central_lower_limit` (`center_he4 < 1e-4`) without timing out.

| Engine | `use_Ledoux_criterion` | Status / Classification | Termination | Steps (Retries) | Final `star_age` (Myr) | He-Burning Lifetime (`Yc = 1e-4`, Myr) | `M_cc` at `Yc < 0.9` (`Msun`) | `M_cc` at `Yc < 0.5` (`Msun`) | `M_cc` at `Yc < 0.1` (`Msun`) | Max `M_cc` (`Msun`) | Wall Time (s) |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MESA `26.4.1` | `.true.` (upstream) | `ok` (MESA `26.4.1` run) | `xa_central_lower_limit` | 190 (1) | 134.6928 | 134.6729 | 0.105900 | 0.146816 | 0.156971 | 0.266800 | 609.4 |
| MESA `26.4.1` | `.false.` (modified) | `ok` (MESA `26.4.1` run) | `xa_central_lower_limit` | 191 (1) | 135.1100 | 135.0764 | 0.105900 | 0.141661 | 0.156380 | 0.267788 | 567.5 |
| FlexMESA `aaa29e9` (`strict`) | `.true.` (upstream) | `ok`, `diagnostic (2 controls ignored)` | `xa_central_lower_limit` | 190 (1) | 134.9314 | 134.9224 | 0.105901 | 0.201755 | 0.160919 | 0.267220 | 379.9 |
| FlexMESA `aaa29e9` (`strict`) | `.false.` (modified) | `ok`, `diagnostic (2 controls ignored)` | `xa_central_lower_limit` | 188 (1) | 133.9580 | 133.9136 | 0.105901 | 0.201755 | 0.155759 | 0.265849 | 354.9 |

Metric definitions (full-precision values are in `results.json` and `results.csv`):

- **He-burning lifetime**: `star_age` linearly interpolated in `center_he4` between the two consecutive history rows bracketing `1.0e-4` (`xa_central_lower_limit(1) = 0.0001` for `he4` in `inlist_hb_2M`). Because `inlist_hb_2M` sets `initial_age = 0` at ZACHeB, `star_age` is the elapsed core helium-burning age.
- **`M_cc` milestones**: `mass_conv_core` at the first history row where `center_he4 < 0.9`, `< 0.5`, and `< 0.1`, matching upstream's `run_star_extras.f90` convention for `mass_conv_core_y050` (first step with `center_he4 < 0.5`). Both MESA runs also printed `[TestHub] Convective core mass at Yc = 0.5 (Msun)` matching `results.json` (`0.147` for `.true.`, `0.142` for `.false.`, both inside upstream's `[0.14, 0.25]` `Msun` TestHub bounds).
- **Ignored controls in both FlexMESA cells (count = 2)**: `&controls::photo_interval`, `&star_job::required_termination_code_string`.

## Measured Differences

- **MESA `26.4.1` (`.false.` minus `.true.`)**: final `star_age` increases by `+417,232.8 yr` (`+0.310%`), helium-burning lifetime increases by `+403,440.6 yr` (`+0.300%`), step count moves from `190` to `191`, `mass_conv_core` at `Yc < 0.5` moves from `0.146816` to `0.141661 Msun`, and maximum `mass_conv_core` increases by `+0.000988 Msun`.
- **FlexMESA `aaa29e9` (`.false.` minus `.true.`, diagnostic)**: final `star_age` decreases by `-973,419.9 yr` (`-0.721%`), helium-burning lifetime decreases by `-1,008,748.7 yr` (`-0.748%`), step count moves from `190` to `188`, `mass_conv_core` at `Yc < 0.1` moves from `0.160919` to `0.155759 Msun`, and maximum `mass_conv_core` decreases by `-0.001372 Msun`.
- **FlexMESA minus MESA (`use_Ledoux_criterion = .true.`)**: final `star_age` differs by `+238,651.4 yr` (`+0.177%`), helium-burning lifetime by `+249,479.1 yr` (`+0.185%`), and maximum `mass_conv_core` by `+0.000420 Msun`. At `Yc < 0.5`, `mass_conv_core` jumps between ~`0.14` and ~`0.20 Msun` from step to step under the test suite's `max_years_for_timestep = 1d6` setting in both engines, so single-step sampling at `Yc < 0.5` lands on `0.146816 Msun` in MESA (step 108) and `0.201755 Msun` in FlexMESA (step 107).
- **FlexMESA minus MESA (`use_Ledoux_criterion = .false.`)**: final `star_age` differs by `-1,152,001.3 yr` (`-0.853%`), helium-burning lifetime by `-1,162,710.2 yr` (`-0.861%`), and maximum `mass_conv_core` by `-0.001939 Msun`.

## Provenance

- **MESA Release**: `26.4.1` (`data/version_number`: `r26.4.1`, git tag `26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`, Zenodo `19722306`).
- **MESA SDK and Compiler**: `x86_64-linux-26.6.1`, `GNU Fortran (GCC) 15.2.0`, `CRMATH` math backend (also recorded per MESA cell from `LOGS/history.data` header fields `version_number`, `compiler`, `build`, `MESA_SDK_version`, `math_backend`, and `date`).
- **FlexMESA Build**: commit `aaa29e902af62fbd3f5756240569ebc097c0bfaa` (`aaa29e9`), release `flexmesa-py` (`strict` numerics mode).
- **Host**: `Ubuntu 24.04.3 LTS (Linux 6.6.87.2-microsoft-standard-WSL2 x86_64)`, `AMD Ryzen 7 9800X3D 8-Core Processor` (16 logical threads), Python `3.14.0`, `OMP_NUM_THREADS=4`.
- **Execution Guards**: one heavy run at a time via `flock`, `--timeout-sec 5400`, `JAX_PLATFORMS=cpu` on FlexMESA trace subprocesses.

## What Was Not Verified

- Both FlexMESA cells ran with `--allow-unsupported-controls` and ignored 2 upstream controls (`&controls::photo_interval`, `&star_job::required_termination_code_string`), so neither FlexMESA row is a passing run.
- `star/test_suite/hb_2M/src/run_star_extras.f90` is unported in FlexMESA (`#231`), so the upstream TestHub `mass_conv_core_y050` bounds check ran only in MESA.
- Optional early stages (`inlist_to_TAMS` and `inlist_to_ZACHeB`) were skipped (`MESA_SKIP_OPTIONAL=t`) so all four runs started from upstream's stored `standard_2.0_ZACHeB.mod`.
- Full multi-column history-harness tolerances, radial `profileN.data` comparisons, and timestep-convergence runs (`max_years_for_timestep < 1d6`) were not evaluated here.

## Reproduction Recipe

Set `MESA_DIR` to a compiled MESA `26.4.1` tree, `MESASDK_ROOT` to MESA SDK `x86_64-linux-26.6.1`, `WORK_DIR` to an empty scratch directory, and `LOCK_FILE` to a lock path so heavy runs stay serialized. Ensure the target FlexMESA build is importable on `PYTHONPATH`.

```bash
export MESA_DIR=/path/to/mesa-26.04.1
export MESASDK_ROOT=/path/to/mesasdk-26.6.1
export WORK_DIR=/path/to/scratch/ledoux_hb_2m
export LOCK_FILE="$WORK_DIR/campaign.lock"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=4

mkdir -p "$WORK_DIR"
SCRIPT=benchmarks/fortran_mesa/ledoux_hb_2m_experiment/run_ledoux_experiment.py

for cell in "mesa true" "mesa false" "flexmesa true" "flexmesa false"; do
    set -- $cell
    flock "$LOCK_FILE" python "$SCRIPT" run \
        --engine "$1" --ledoux "$2" \
        --work-root "$WORK_DIR" --timeout-sec 5400
done

python "$SCRIPT" summarize \
    --work-root "$WORK_DIR" \
    --out-dir benchmarks/fortran_mesa/ledoux_hb_2m_experiment
```
