# One-zone MESA 26.4.1 `net::net_get` capture for `approx21_cr60_plus_co56.net` (`#516`)

This directory holds the standalone Fortran generator (`gen_net_516.f90`), its build recipe (`Makefile`) and the committed reference table (`mesa_26_4_1_net_approx21_zone650.tsv`). The table was captured from compiled MESA release `26.4.1` (commit `c8c202662922a22b3efefa824473ad615c623afd`), built with MESA SDK `x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH`). The TSV header records the MESA version, commit, SDK, math backend, compiler and flags.

## State and controls

The generator evaluates `net_get` once, at a hydrogen-burning state copied bit for bit from the MESA `26.4.1` step dump for `star/20M_z2m2_high_rotation`: model 1, zone 650, $\log_{10} T \approx 7.208$, $\log_{10}\rho \approx -0.041$, $X \approx 0.723$. In this regime the CN cycle is switching on and the pp-chain He3 branches still matter.

The call reproduces the star's `do1_net` (`star/private/net.f90:184-256`) as follows:

- $T = \exp(\ln T)$ and $\rho = \exp(\ln \rho)$ (`star/private/hydro_vars.f90:299,305`), with `log10_T = lnT/ln10` and `log10_rho = lnd/ln10`.
- `abar`, `zbar`, `z2bar` and `ye` come from `basic_composition_info`. The mass fractions are passed unnormalised, as the star passes `s% xa`.
- `eta` and its partials come from `eosDT_get` with the default `&eos` controls. The case's `&eos` namelist is empty.
- The star's default controls are used: `screening_mode = 'chugunov'`, unit rate, weak-rate and neutrino-Q factors, and `fe56ec_fake_factor = 1d-7` above $T = 3\times10^9\,\mathrm{K}$ (`star/defaults/controls.defaults:6779, 6835, 6845, 8245-8246`).

The captured `eps_nuc`, `0x40055A53ABE0D9F9`, equals the `eps_nuc` that MESA's own star run records in its step dump at this zone. The generator therefore reproduces the star's network call exactly.

## Quantities

All values are stored as exact IEEE-754 bits plus a decimal rendering:

- `INPUT`: the thermodynamic and composition inputs.
- `SCALAR`: `eps_nuc`, `d_eps_nuc_dRho`, `d_eps_nuc_dT` and `eps_neu_total`.
- `ISO`: `dxdt` and its $\rho$, $T$ and composition partials.
- `JAC`: the dense `d_dxdt_dx` matrix.
- `CAT`: the `chem_def` category subtotals.
- `RATE`: `rate_raw` and `rate_screened` for every reaction.

## Regenerating

The generator links against the installed MESA libraries read-only. No MESA source is modified.

```bash
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
make -C benchmarks/fortran_mesa/net_oracles_516 MESA_DIR=/path/to/mesa-26.4.1 \
    MESASDK_ROOT="$MESASDK_ROOT" generate clean
```

If linking fails on `-lz`, add the directory holding `libz.so` to `LIBRARY_PATH`.
