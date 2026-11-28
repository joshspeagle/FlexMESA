# MESA 26.4.1 `turb::set_semiconvection` and `turb::set_thermohaline` Reference Oracles (`#155`, `DECISIONS #80`)

This directory holds the standalone Fortran generator (`gen_turb_mix_oracles.f90`), its build recipe (`Makefile`) and the committed IEEE-754 reference table (`mesa_26_4_1_turb_mix_oracles.tsv`). The table was written by compiled MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`) with MESA SDK `x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH`).

## Regenerating

```bash
make -C benchmarks/fortran_mesa/turb_mix_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean
```

## Coverage

The operating points are those of `crates/flexmesa-star/tests/turb_155_bd_parity_test.rs`. Each differentiable input is an `auto_diff_real_star_order1` seeded in `d1Array(1)` only, and the table records the value and slot-1 partial of `gradT`, `Y_face`, `conv_vel` and `D`, plus `mixing_type` and `ierr`:

- `set_semiconvection` (through `semiconvection::calc_semiconvection`) with `'Langer_85 mixing; gradT = gradr'`, with `'Langer_85'`, and with `alpha_semiconvection = 0` and `gradr == gradL`;
- `set_thermohaline` with `'Kippenhahn'`, `'Traxler_Garaud_Stellmach_11'` and `'Brown_Garaud_Stellmach_13'`, driven by `he3`.
