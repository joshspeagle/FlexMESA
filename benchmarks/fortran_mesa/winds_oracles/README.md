# MESA 26.4.1 Winds `set_mdot` Reference Oracles (`#159`)

This directory contains the standalone Fortran generator (`gen_winds_oracles.f90`), build recipe (`Makefile`), and committed IEEE-754 double-precision reference table (`mesa_26_4_1_winds_oracles.tsv`) captured directly against compiled MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`) with `MESA SDK x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH`). It pins `crates/flexmesa-star/tests/winds_159_parity_test.rs`.

## Regenerating the Oracles

With `MESASDK_ROOT` pointing to the MESA SDK and `MESA_DIR` pointing to a built MESA `26.4.1` tree:

```bash
make -C benchmarks/fortran_mesa/winds_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean
```

## Method

`winds::set_mdot` is the only public entry point of the `winds` module (`do_set_mdot`, `eval_super_eddington_wind`, `eval_rlo_wind`, and the wind-scheme evaluators are private or contained). Each case allocates a `star_info` through the `star_lib` setup path, overwrites exactly the fields `do_set_mdot` reads (photospheric `L`, `M`, `T`, surface and central `h1`/`he4`, `Zbase`, `dt`, `mass_change`, hot/cool switch temperatures, wind schemes and scaling factors, super-Eddington, RLO, `min_wind`/`max_wind`, central-temperature taper, and envelope-mass limits), sets `s% L_phot = L_phot / Lsun` and calls `set_mdot(s, s% L_phot * Lsun, M_phot, T_phot, ierr)` matching `star/private/evolve.f90:427` (`#679`), and records the outputs.

- **Inputs** (`role = in`, `in_str`): every value set above, as hex and decimal.
- **Outputs** (`role = out`): `mstar_dot` (the final rate, the value the Rust test compares), `explicit_mstar_dot`, `super_eddington_wind_mdot` (the output of `eval_super_eddington_wind`), `doing_rlo_wind` (1 when `eval_rlo_wind` returned a nonzero rate), and `ierr`.
- **Single schemes** (`vink_*`, `bjorklund_*`, `dutch_*`, `direct_*`): the scheme is selected as the hot wind with both switch temperatures below `T`, so `mstar_dot = -scheme(L, M, R, T, ...)*scaling*Msun/secyr`.
- **`h_env_limit_zero`**: a three-zone model whose `he_core_mass` and `co_core_mass` come from MESA's own `report::get_core_info`.

## Not Captured Directly

- The `xfer_ratio` value of `eval_rlo_wind` is internal to `do_set_mdot`; it is observable only through `rlo_xfer_ratio` (`mass_change` scaled by `xfer_ratio`) and `doing_rlo_wind = 0`.
- `eval_nugis_lamers_wind` has no stand-alone case; it is reached through `dutch_wr_nl`.
