# MESA 26.4.1 GYRE Pulsation-Data Captures (`#170`, Package G3b)

This directory checks FlexMESA's port of `star_get_pulse_data(id, 'GYRE', ...)` (`get_gyre_data`, `star/private/pulse_gyre.f90:38-416`; `crates/flexmesa-star/src/pulse_gyre_data.rs`) against MESA itself. It captures the star state that `get_gyre_data` reads at every GYRE call of the `astero_gyre` test case, and compares the port's output arrays with the arrays MESA passed to GYRE.

- **MESA release:** `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`, Zenodo record `19722306`).
- **Toolchain:** MESA SDK `x86_64-linux-26.6.1`.
- **Case:** `astero/test_suite/astero_gyre`, with its inlists unmodified.

## Contents

| File | Role |
| :--- | :--- |
| `pulse_state_tap.f90`, `pulse_state_wrap.c` | A link-time wrapper of `gyre_mesa_m::set_model` (`-Wl,--wrap`). At each call it writes the `star_info` fields `get_gyre_data` reads, the pulse-data controls, and the `global_data`/`point_data` arrays to `pulse_state/gyre_eval_NNNNNN.tsv`. MESA's libraries are not rebuilt. |
| `run_capture.sh` | Copies the case to a scratch directory, builds it with the tap and runs it. |
| `data/` | The captures of GYRE calls 1 and 10 (model numbers 50 and 465), without the `point_data` record, which the test reads from the `mesa_histories` bundle instead. |

## Recipe

```bash
export MESASDK_ROOT=/path/to/mesasdk MESA_DIR=/path/to/mesa-26.04.1
WORK=/path/to/scratch bash benchmarks/fortran_mesa/gyre_pulse_data/run_capture.sh
cargo test --release -p flexmesa-star --test pulse_gyre_data_g3b
```

## Results

The capture run reproduces the reference run. In all 10 GYRE calls, the captured `point_data` and `global_data` are bit-identical to the reference run's `pulse_tap/models/*.gyre` files in the `mesa_histories` bundle.

Given MESA's captured star state, the port reproduces `global_data` and every point of `point_data` bit for bit:

- 10 of 10 calls (15,309 points) on the full capture set;
- the 2 committed calls (3,081 points) in the test.

## FlexMESA's Own `astero_gyre` Run (Package G3c)

`compare_astero_gyre_run.py` runs FlexMESA's `astero_gyre` with its ported extras (`tests/case_extras/astero_extras.py`, `AsteroGyreExtras`). At each GYRE call it compares two things with the reference run's `pulse_tap` captures:

- the modes that `gyre_call_back` stores: `n_pg`, frequency, inertia and growth rate;
- the GYRE input that FlexMESA's own model gives.

```bash
python benchmarks/fortran_mesa/gyre_pulse_data/compare_astero_gyre_run.py \
    --case-dir "$MESA_DIR/astero/test_suite/astero_gyre" \
    --histories /path/to/mesa_histories --work /path/to/scratch \
    --out benchmarks/fortran_mesa/gyre_pulse_data/results/astero_gyre_flexmesa_vs_mesa.tsv
```

The committed result is in `results/astero_gyre_flexmesa_vs_mesa.tsv`. Through model 350:

- FlexMESA's model has the same number of GYRE points as MESA's;
- `r`, `P`, `T` and `rho` agree to 1e-12 relative;
- every mode's frequency agrees to 2e-12 relative.

After the meshes diverge near model 400 (point counts differ by 1 or 2), frequencies agree to 1e-4. None are bit-identical, because the star states are not: the first-divergence report (`flexmesa trace --compare`) finds the first difference at model 1, `eps_nuc` by 1 ULP in one zone. The final check passes, as in MESA: FlexMESA finds the `n_pg = 4` mode at 379.07 µHz, MESA at 379.05 µHz, and the target is 375 µHz within 3%.

## Live-Run Comparison for Other Cases (`#612`)

`compare_pulse_live_run.py` generalises the comparison above to any `star` or `astero` case whose extras make oscillation calls. It runs the case through the in-process trace entry point (strict mode, the manifest's extras factory) with a recording subclass of the case's own extras handler, which records and changes nothing. Each call is matched with the reference call at the same model number, and each mode on `(l, m, n_pg)`. Per quantity it reports the values compared, the bit-identical count and the maximum relative difference, and it lists calls or modes present on one side only. Quantities the extras do not expose are reported as not compared (`GyreInMesaStarExtras` and `RspGyreExtras` expose frequency and growth rate only, not inertia).

```bash
python benchmarks/fortran_mesa/gyre_pulse_data/compare_pulse_live_run.py \
    --suite star --case gyre_in_mesa_rsg \
    --histories /path/to/mesa_histories --work /path/to/scratch \
    --out benchmarks/fortran_mesa/gyre_pulse_data/results/star_gyre_in_mesa_rsg_flexmesa_vs_mesa.tsv
```

References are MESA `26.4.1` runs in the `mesa_histories` bundle (gfortran 15.2.0, SDK `x86_64-linux-26.6.1`). Result files, one per case, are in `results/`; their headers carry the command and the engine commit.

| Case | Calls | Modes (freq) | Frequency bit-identical, max rel. | Growth rate bit-identical, max rel. | Inertia | One-sided |
| :--- | ---: | ---: | :--- | :--- | :--- | :--- |
| `star/gyre_in_mesa_rsg` | 10 | 50 | 50, 0 | 50, 0 | not exposed | none |
| `star/gyre_in_mesa_spb` | 51 | 511 | 511, 0 | 511, 0 | not exposed | none |
| `star/gyre_in_mesa_bcep` | 54 | 192 | 192, 0 | 192, 0 | not exposed | none |
| `star/gyre_in_mesa_ms` | 61 | 244 | 244, 0 | 244, 0 | not exposed | none |
| `star/gyre_in_mesa_envelope` | 5 | 18 | 0, 9.6e-13 | 0, 3.0e-11 | not exposed | 1 mode (`n_pg = 4` at model 939, FlexMESA only) |
| `star/rsp_gyre` | 1 | 4 | 0, 3.6e-14 | 0, 1.2e-11 | not exposed | none |
| `astero/astero_adipls` | 10 | 92 | 92, 0 | 92, 0 | 92 of 92, 0 | none |

Observations from the files:

- In `gyre_in_mesa_envelope` no frequency is bit-identical, with differences from 8e-15 to 9.6e-13. At model 939 FlexMESA returns an `n_pg = 4` mode that MESA's call does not.
- In `rsp_gyre` GYRE is called once (model 1, at the model build), and the run takes 6215 steps and ends at its `extras_finish_step` termination.
- Whether the star states are bit-identical at each call was not measured here. The comparison covers GYRE's or ADIPLS's output only.

### Code Path of Each Case's Oscillation Calls

- **`gyre_in_mesa_*` (five star cases).** `GyreInMesaStarExtras` (`tests/case_extras/star_atm_pulse.py:184,217,229`) calls `_rust.GyreMesa(...)`, `set_model` and `get_modes`. These go to `crates/flexmesa-py/src/gyre_mesa_py.rs:115,146`, which call `flexmesa_star::pulse_gyre_data::star_get_pulse_data_gyre` (`gyre_mesa_py.rs:123`, a port of `get_gyre_data`, not a solver) and `flexmesa_gyre::mesa::GyreMesa::set_model`/`get_modes` (`crates/flexmesa-gyre/src/mesa.rs:95,126`). `flexmesa-gyre` does not depend on `flexmesa-star` (`crates/flexmesa-gyre/Cargo.toml`), so none of `pulse_bvp`, `pulse_bvp_recon`, `pulse_modes`, `scan_star_bvp_eigenmodes` or `eval_linear_oscillation_modes` is reachable from it. There is no fallback: an unsupported option returns `GyreError::Unported`, which reaches Python as `RuntimeError`.
- **`rsp_gyre`.** `RspGyreExtras.other_rsp_linear_analysis` (`tests/case_extras/rsp_extras.py:169-174`) uses the same `GyreMesa` calls (`get_modes_full`, `gyre_mesa_py.rs:159`). The engine reaches it through `crates/flexmesa-py/src/extras_py.rs:2846`; with `use_other_rsp_linear_analysis` set and no hook result, the driver returns an error (`crates/flexmesa-star/src/rsp_driver.rs:466`), not a fallback to another solver. The path is the same as above.
- **`astero_adipls`.** `AsteroAdiplsExtras` (`tests/case_extras/astero_extras.py:481,495`) calls `_rust.AsteroAdipls` and `get_one_el_info` (`crates/flexmesa-py/src/astero_adipls_py.rs:64`), then `AsteroAdipls::do_adipls_get_one_el_info` (`crates/flexmesa-astero/src/adipls_support.rs:381`), `run_adipls_one_el` (`:416`) and `flexmesa_adipls` `run_call_with_hooks` (`:531`). `flexmesa-adipls` depends only on `flexmesa-const` and `flexmesa-math`. The case sets `astero_just_call_my_extras_check_model`, so the astero search driver is not run (`crates/flexmesa-astero/src/e2e.rs:245`).
- **Where the in-house solver is still reachable.** `scan_star_bvp_eigenmodes` is called only from the silent fallback in `crates/flexmesa-astero/src/sample_eval.rs:1093-1136`, which belongs to the astero search path (`search_driver.rs`) and not to the five cases above. `eval_linear_oscillation_modes` is exposed as `star_eval_linear_oscillation_modes` (`crates/flexmesa-py/src/star_e2e_py.rs:411-417`) and used by `tests/test_e2e_4_tier4.py`. `solve_bvp_eigenmodes` is exposed as `star_solve_gyre_mesa_modes_full` (`crates/flexmesa-py/src/extras_py.rs:408`), which no case extras or test calls. None of these is on the path of a case listed in the table.
