# Pulse-Export State Tap for MESA 26.4.1 (`#491`)

This recipe records MESA's own star state at each `export_pulse_data` call,
together with the GYRE, FGONG and OSC text MESA writes from that state, so
that FlexMESA's exporters can be compared with MESA byte for byte on
identical input. That separates exporter fidelity from trajectory drift.

## Files

| File | Role |
| :--- | :--- |
| `pulse_dump_tap.f90` | Module `pulse_dump_tap`. Before each export it writes every `star_info` field that `get_gyre_data`, `get_fgong_data` and `get_osc_data` read (`star/private/pulse_gyre.f90`, `pulse_fgong.f90`, `pulse_osc.f90`) to `pulse_dump/call_NNNN.dump`, with reals as IEEE-754 hexadecimal (`z16.16`). The first 12 calls are kept. |
| `pulse_dump_wrap.c` | Linker wrapper for `__pulse_MOD_export_pulse_data` (`-Wl,--wrap`). It runs the dump, calls the real routine, copies the file MESA wrote to `pulse_dump/call_NNNN.out`, and calls the real routine once more with `data_format = 'OSC'` into `pulse_dump/call_NNNN.osc`, because no test case writes OSC. The export flags are saved in `call_NNNN.flags`. It changes no MESA source and no arithmetic. |

## Reproduce

```bash
source scripts/mesa_reference/env_mesa.sh
gfortran -O2 -ffree-form -ffree-line-length-none -fopenmp -I"$MESA_DIR/include" \
    -c benchmarks/fortran_mesa/pulse_export_tap/pulse_dump_tap.f90 -o <build>/pulse_dump_tap.o
gcc -O2 -c benchmarks/fortran_mesa/pulse_export_tap/pulse_dump_wrap.c -o <build>/pulse_dump_wrap.o
OUT=<scratch> PULSE_TAP_OBJS="<build>/pulse_dump_tap.o <build>/pulse_dump_wrap.o" \
    bash scripts/mesa_reference/run_mesa_case_26_4_1.sh astero fast_from_file
FLEXMESA_PULSE_TAP_DIR=<scratch>/fast_from_file/pulse_dump \
    cargo test --release -p flexmesa-star --test pulse_export_tap_491_test
```

`astero/fast_from_file` writes `best.fgong` (FGONG, `fgong_ivers = 1300`)
and `best.gyre` (GYRE, `gyre_data_schema = 101`) with
`add_center_point = .true.`, `keep_surface_point = .false.` and
`add_atmosphere = .false.` each time the best model of a track improves.
With the tap linked in, the run's `LOGS/history.data` rows are identical to
those of the `mesa-histories-v1` run of the same case.
