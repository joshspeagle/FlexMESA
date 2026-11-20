# MESA 26.4.1 Thermal Neutrino (`neu`) Reference Oracles (`#161`)

This directory contains the standalone Fortran generator (`gen_neu_oracles.f90`), build recipe (`Makefile`), and committed IEEE-754 double-precision reference table (`mesa_26_4_1_neu_oracles.tsv`) captured directly against compiled MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`) with `MESA SDK x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH 1.2` / `crlibm 1.0beta4`).

## Regenerating the Oracles

With `MESASDK_ROOT` pointing to the MESA SDK and `MESA_DIR` pointing to a built MESA `26.4.1` tree:

```bash
make -C benchmarks/fortran_mesa/neu_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean
```

## Coverage

- **36 thermodynamic, composition, cutoff, and flag-isolation states (`1,080` exact `f64` outputs)**:
  - Total loss vector `loss(1:5)` (`erg g^-1 s^-1` and partial derivatives with respect to `T`, `Rho`, `abar`, `zbar`) and per-channel matrix `sources(1:5, 1:5)` across Itoh et al. (1996) `pair`, `plas` (including upstream MESA `26.4.1` Weinberg prefactor `0.93153`, `#193`), `phot`, `brem` (liquid, crystal-blend `170 < Gamma < 180`, and pure-crystal `Gamma >= 180` branches), and `reco`.
  - Smooth low-temperature `cospi`/`sinpi` cutoff window (`log10_T <= log10_Tlim`, ramp interior `log10_Tlim < log10_T < log10_Tlim + 0.4`, and unattenuated `log10_T >= log10_Tlim + 0.4` for both `log10_Tlim = 7.5` and `log10_Tlim = 6.8`), `T_9 <= 1` and `T_9 > 1` polynomial branches, and single-channel `flags` isolation.
- Each output is recorded both as big-endian 16-hex-digit IEEE-754 raw bits (`0x<16-hex>`) and round-trip decimal (`ES25.17E3`).
