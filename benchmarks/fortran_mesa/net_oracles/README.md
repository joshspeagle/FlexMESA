# MESA 26.4.1 Nuclear Reaction Network (`net::net_get`) Reference Oracles (`#328`, `#161`)

This directory contains the standalone Fortran generator (`gen_net_oracles.f90`), build recipe (`Makefile`), and committed IEEE-754 double-precision reference table (`mesa_26_4_1_net_oracles.tsv`) captured directly against compiled MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`) with `MESA SDK x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH`).

## Regenerating the Oracles

With `MESASDK_ROOT` pointing to the MESA SDK and `MESA_DIR` pointing to a built MESA `26.4.1` tree:

```bash
make -C benchmarks/fortran_mesa/net_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean
```

## Coverage

- **Six networks evaluated with `net::net_get` (`extended_screening`, `use_suzuki_weak_rates = .false.`)**:
  - `basic.net`, `o18_and_ne22.net`, `pp_extras.net`, `cno_extras.net` at $\log_{10} T = 7.833$, $\log_{10}\rho = 2.0$, $\eta = 0$.
  - `approx21.net` at $\log_{10} T = 9.6$, $\log_{10}\rho = 6.0$, $\eta = 0$.
  - `approx21_cr60_plus_co56.net` at $\log_{10} T = 4.6233007922659333$, $\log_{10}\rho = -10.746410107891649$, $\eta = -22.590260158215202$.
- **Quantities recorded in exact IEEE-754 bits**: all thermodynamic and isotopic inputs (`INPUT`), nuclear energy generation scalars (`SCALAR`: `eps_nuc`, `d_eps_nuc_dRho`, `d_eps_nuc_dT`, `eps_neu_total`), per-isotope time derivatives and partials (`ISO`: `dxdt`, `d_dxdt_dRho`, `d_dxdt_dT`, `d_eps_nuc_dx`), full dense Jacobian matrix `d_dxdt_dx(i, j)` (`JAC`), all 24 `chem_def` category subtotals (`CAT`), and unscreened/screened reaction rates (`RATE`: `rate_raw`, `rate_screened`).
- The TSV header records `mesa_version`, `mesa_git_commit`, `mesasdk_version`, `math_backend`, `compiler`, and `fflags`.
