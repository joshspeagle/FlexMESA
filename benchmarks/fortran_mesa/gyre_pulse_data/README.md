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
