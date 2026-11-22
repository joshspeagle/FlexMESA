# MESA 26.4.1 `turb::set_TDC` Value and `d1Array` Partial Reference Oracles (`#328`, `#321`, `DECISIONS #50`)

This directory contains the standalone Fortran generator (`gen_turb_tdc_oracles.f90`), build recipe (`Makefile`), and committed IEEE-754 double-precision reference table (`mesa_26_4_1_turb_tdc_oracles.tsv`) captured directly against compiled MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`) with `MESA SDK x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH`).

## Regenerating the Oracles

With `MESASDK_ROOT` pointing to the MESA SDK and `MESA_DIR` pointing to a built MESA `26.4.1` tree:

```bash
make -C benchmarks/fortran_mesa/turb_tdc_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean
```

## Coverage

- **`check_tdc`** (`j = 0..30`, 31 rows from `turb/test/src/test_turb.f90:168-235`): Negative-$Y$ decaying convection time-step sweep $\Delta t = 500 \times 1.02^j\,\mathrm{s}$ crossing the damped-convection to radiative-equilibrium transition.
- **`compare_tdc_cox`** (1 row from `turb/test/src/test_turb.f90:82-166`): Long-timestep ($\Delta t = 10^{40}\,\mathrm{s}$) positive-$Y$ approach to steady-state Cox MLT.
- For all 32 operating points, all 16 differentiable inputs to `set_TDC` (`chiT`, `chiRho`, `gradr`, `r`, `P`, `T`, `rho`, `dV`, `Cp`, `opacity`, `scale_height`, `gradL`, `grada`, `Eq_div_w`, `grav`, `energy`) are seeded as independent variables in slots `1..16` of `auto_diff_real_star_order1%d1Array`, and exact IEEE-754 bits are recorded for all scalar parameters, inputs, `tdc_num_iters`, `mixing_type`, `Y_face%val`, `Y_face%d1Array(1:16)`, `conv_vel%val`, `conv_vel%d1Array(1:16)`, `gradT%val`, and `D%val`.
- In upstream `turb/private/tdc.f90:55, 64, 113, 137, 177, 203`, `get_TDC_solution` declares `intent(out) :: tdc_num_iters` and leaves it unassigned on radiative-equilibrium `radY` early exits (`check_tdc_j16` through `check_tdc_j30`), so those 15 rows retain the caller's previous `tdc_num_iters = 6` value from `check_tdc_j15` (`mixing_type = 0`, `conv_vel = 0`, `Y_face = radY`).
- The TSV header records `mesa_version`, `mesa_git_commit`, `mesasdk_version`, `math_backend`, `compiler`, and `fflags`.
