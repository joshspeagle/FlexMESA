# MESA 26.4.1 Surface Atmosphere (`atm`) Reference Oracles (`#161`)

This directory contains the standalone Fortran generator (`gen_atm_oracles.f90`), build recipe (`Makefile`), and committed IEEE-754 double-precision reference table (`mesa_26_4_1_atm_oracles.tsv`) captured directly against compiled MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`) with `MESA SDK x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH 1.2` / `crlibm 1.0beta4`).

## Regenerating the Oracles

With `MESASDK_ROOT` pointing to the MESA SDK and `MESA_DIR` pointing to a built MESA `26.4.1` tree:

```bash
make -C benchmarks/fortran_mesa/atm_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean
```

## Coverage

Unlike upstream `atm/test/src/test_atm_support.f90` (which sets `SKIP_PARTIALS = .TRUE.` and checks only scalar `log T` and `log P`), this generator evaluates **`skip_partials = .false.`** across every surface atmosphere boundary formulation in `atm` and `star/private/atm_support.f90` (`F2-23`):

- **`t_tau_rel`**: `eval_T_tau`, `eval_T_tau_dq_dtau`, `get_T_tau_base`, `eval_E2(x, E2, dE2_dx)`, and `eval_Paczynski_gradr` (`tau < 2/3` and `tau >= 2/3`).
- **`t_tau_atm` (`F2-23`)**: `'fixed'`, `'iterated'` (including the `star/private/atm_support.f90:312-322` implicit `(ln P, ln T)` opacity partial fixup), and `'varying'` (`dopri5` ODE integration) across `Pextra_factor in {0.0, 1.0, 1.5}` and all four $T(\tau)$ relations (`Eddington`, `solar_Hopf`, `Krishna_Swamy`, `Trampedach_solar`), emitting all 11 outputs `(lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap, lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap, kap)`.
- **`table` & `table_blend` (`F2-23`)**: `atm_eval_table`, `atm_get_table_alfa_beta`, `atm_get_table_base`, and off-table $T(\tau)$ blending (`alfa = 0`, `0 < alfa < 1`, and `alfa = 1` with `tau_base` switching per `star/private/atm_support.f90:430-689`).
- **`irradiated`**: Guillot (2010) grey irradiated atmospheres with analytical partials (`gamma = 0` and `gamma > 0`).
