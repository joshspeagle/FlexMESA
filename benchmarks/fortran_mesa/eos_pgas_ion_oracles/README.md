# MESA 26.4.1 `eos` Exact Kernel Oracles (`#161`, `DECISIONS #80`)

This directory holds the standalone Fortran generator (`gen_eos_pgas_ion_oracles.f90`), its build recipe (`Makefile`) and the committed IEEE-754 reference table (`mesa_26_4_1_eos_pgas_ion_oracles.tsv`). The table was written by compiled MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`) with MESA SDK `x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH`).

## Regenerating

```bash
make -C benchmarks/fortran_mesa/eos_pgas_ion_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean
```

## Coverage

- `ion_offset::compute_ion_offset` (`crates/flexmesa-eos/tests/wave3a1_golden_tests.rs`):
  - `ion_offset_pure_h`: 1-species pure `h1` (`xa = [1.0]`).
  - `ion_offset_solar_3sp`: 3-species mixture `h1`, `he4`, `c12` (`xa = [0.70, 0.28, 0.02]`).
- `ideal::get_ideal_for_eosdt` / `ideal::get_ideal_eos_results` (`crates/flexmesa-eos/tests/wave3a1_golden_tests.rs`):
  - `ideal_solar_1e5_1em2`: 3-species mixture `h1`, `he4`, `c12` (`xa = [0.70, 0.28, 0.02]`) at `T = 1e5 K`, `Rho = 1e-2 g cm^-3` (`26 x 3` `val`, `d_dlnd`, `d_dlnT`).
- `eosPT_eval::get_Pgas` (`crates/flexmesa-eos/tests/wave3b1_golden_tests.rs`):
  - `eospt_get_pgas_solar_15_7p2`: 7-species solar mixture (`X = 0.70`, `Z = 0.02`) at `logT = 7.2`, solving for `logPgas` matching `lnE` from `eosPT_get` at `(logPgas=15.0, logT=7.2)` starting from `logPgas_guess = 14.8` with `logPgas_tol = 1e-6`, `other_tol = ln10 * 1e-6`, `max_iter = 100`.
