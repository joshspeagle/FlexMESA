# MESA 26.4.1 Nuclear Reaction Rates (`rates`) Reference Oracles (`#161`)

This directory contains the standalone Fortran generator (`gen_rates_oracles.f90`), build recipe (`Makefile`), and committed IEEE-754 double-precision reference table (`mesa_26_4_1_rates_oracles.tsv`) captured directly against compiled MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`) with `MESA SDK x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH 1.2` / `crlibm 1.0beta4`).

## Regenerating the Oracles

With `MESASDK_ROOT` pointing to the MESA SDK and `MESA_DIR` pointing to a built MESA `26.4.1` tree:

```bash
make -C benchmarks/fortran_mesa/rates_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean
```

## Coverage

- **`screen`**: Salpeter (1954), extended `fxt_screen5` (weak, `xlgfac >= 0.77` crossover, strong, and neutral-reactant regimes), Chugunov et al. (2007), and `no_screening` with exact `(scor, dscor_dT, dscor_dRho)`.
- **`coulomb`**: All four `which_mui_coulomb` models (`None`, `DGC1973`, `I1993`, `PCR2009`) and all three `which_vs_coulomb` models (`None`, `ThomasFermi`, `Itoh2002`) with `(mu1, mu2, dmu_eV, vs)`.
- **`reaclib`**: Direct `reaclib_eval` forward and detailed-balance reverse evaluations `(lambda, dlambda_dlnT, rlambda, drlambda_dlnT)` across `r_h2_pg_he3`, `r_c12_ag_o16`, `r_he4_he4_he4_to_c12`, `r_fe54_pg_co55`, and `r_p31_pa_si28`, spanning `T_9 < 0.1`, interior `0.1 <= T_9 <= 10.0`, and `T_9 > 10.0` partition-function extrapolation (`indx = npart - 1`).
- **`raw_rate` & `pycno`**: Analytic `ratelib` rates (`rpp_to_he3`, `rc12ap_to_o16`, `r_c12_to_he4_he4_he4`, `r1212`, `r1616`, `ro16gp_aux`) and Fushiki & Lamb (1987) `eval_FL_epsnuc_3alf` across pycnonuclear and strong-screening regimes.
- **`weak`**: Full 9-component `(lambda, dlambda_dlnT, dlambda_dlnRho, Q, dQ_dlnT, dQ_dlnRho, Qneu, dQneu_dlnT, dQneu_dlnRho)` outputs across `weaklib`, `Suzuki2016` interior points, and out-of-bounds `Suzuki2016 -> weaklib` fallback.
