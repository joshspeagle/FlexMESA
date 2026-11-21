# MESA 26.4.1 `eps_mdot` and Surface Rate Reference Oracles (`#328`, `#318`, PR `#319`)

This directory contains the standalone Fortran generator (`gen_eps_mdot_oracles.f90`), build recipe (`Makefile`), and committed IEEE-754 double-precision reference table (`mesa_26_4_1_eps_mdot_oracles.tsv`) captured directly against compiled MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`) with `MESA SDK x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH`).

## Regenerating the Oracles

With `MESASDK_ROOT` pointing to the MESA SDK and `MESA_DIR` pointing to a built MESA `26.4.1` tree:

```bash
make -C benchmarks/fortran_mesa/eps_mdot_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean
```

## Coverage

- **`accreting_1m_zams`**: A `1.0 Msun`, `Z = 0.02` ZAMS model initialized via `star_load_zams` and stepped through `adjust_mass::do_adjust_mass`, `hydro_vars::set_vars`, and `eps_mdot::calculate_eps_mdot` (`star/private/evolve.f90:580-597`) with accretion rate $\dot{M} = +3 \times 10^{-3}\,M_\odot\,\mathrm{yr}^{-1}$ over $\Delta t = 10^7\,\mathrm{s}$.
- **`mass_losing_1m_zams`**: An independent `1.0 Msun`, `Z = 0.02` ZAMS model stepped through the same sequence with mass-loss rate $\dot{M} = -3 \times 10^{-3}\,M_\odot\,\mathrm{yr}^{-1}$ over $\Delta t = 10^7\,\mathrm{s}$, exercising both nonzero `mdot_acoustic_surface` and nonzero `mdot_adiabatic_surface` thermal-leakage surface terms.
- For both models, exact IEEE-754 bits are recorded for all scalar `AdjustMassRecord` metadata, per-zone inputs after `do_adjust_mass` and `set_vars`, per-zone `total_energy_profile_after_adjust_mass(k)` and `eps_mdot(k)`, and both surface rates.
- The TSV header records `mesa_version`, `mesa_git_commit`, `mesasdk_version`, `math_backend`, `compiler`, and `fflags`.
