# MESA 26.4.1 `do_neu_for_cell` oracle vectors (`#161`)

`gen_micro_neu_oracles.f90` links the compiled MESA release `26.4.1` libraries and calls the real `neu::do_neu_for_cell` (`star/private/neu.f90`) on a one-zone `star_info`, at the five mid-run probe points of `crates/flexmesa-star/tests/micro_161_parity_test.rs` (`HIGHZ_M340`, `WD_M370`, `WD_M668`, `NS_SKYE_HE4`, `NS_SKYE_FE56`).

- **Inputs:** `lnd` and `lnT` from the test's decimal log10 values times `ln10`; `rho = exp(lnd)` and `T = exp(lnT)` through the MESA math library; `abar` and `zbar` from `chem_lib::basic_composition_info` on the raw `xa` (`star/private/micro.f90:321`); `non_nuc_neu_factor = 1`, `use_other_neu = .false.`. Every input is recorded as hex in the TSV.
- **Outputs:** `non_nuc_neu`, `d_nonnucneu_dlnd`, `d_nonnucneu_dlnT`, and the five per-process channels (`plas`, `brem`, `phot`, `pair`, `reco`). The test asserts `non_nuc_neu` and the zone inputs.
- **Not covered:** the `Tmin_neu` gate lives in `do1_neu_kap` (`star/private/micro.f90`), not in `do_neu_for_cell`, so it is not exercised here.

Regenerate with

```bash
make -C benchmarks/fortran_mesa/micro_neu_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean
```

The TSV header records the MESA version, commit, SDK, math backend, compiler and flags.
