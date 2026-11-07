# Upstream Fortran MESA Benchmark Suite (`Wave 1`, `Wave 2` & `Wave 3a`)

This directory contains the native Fortran 2008 benchmark harness comparing
upstream Fortran MESA (`gfortran -O2 -fopenmp`) directly against the Safe Rust
(`crates/flexmesa-bench`) and Python/NumPy/JAX (`src/flexmesa/bench`) Wave 1,
Wave 2, and Wave 3a implementations across identical thermodynamic profiles,
spline grids, block-tridiagonal Henyey matrices, convective transport states,
nuclear abundance profiles, ionization grids, stellar atmosphere SED cubes, and
analytical/single-table equations of state.

## Workloads and Four-Stage Ladder

Every workload is evaluated across radial shell counts
$N_{\mathrm{zones}} \in \{128, 1024, 4096\}$ and verified at
$N_{\mathrm{zones}} \in \{8, 32\}$ against the pinned cross-language reference
checksums:

### Wave 1 Workloads (`run_wave1_bench`)

1. `auto_diff`: Two-variable second-order forward-mode hyperdual evaluation
   (`auto_diff_real_2var_order2`) vs. fused analytical evaluation and 4-thread
   OpenMP execution.
2. `interp_2d`: 2D bicubic spline construction (`interp_mkbicub_db`) and
   evaluation (`interp_evbicub_db` and direct `fvbicub_db`) on a $64 \times 64$
   $(\log_{10}\rho, \log_{10}T)$ grid.
3. `neu`: Itoh et al. (1996) pair and plasmon thermal neutrino loss rates
   (`neu_get`) with full $(\epsilon_\nu, \partial\epsilon_\nu/\partial T,
   \partial\epsilon_\nu/\partial\rho, \partial\epsilon_\nu/\partial\bar{A},
   \partial\epsilon_\nu/\partial\bar{Z})$ sensitivities.
4. `mtx_tridiag`: $5 \times 5$ block-tridiagonal Henyey linear solver comparing
   MESA's parallel odd-even block cyclic reduction (`bcyclic_dble_decsolblk` on
   1 and 4 OpenMP threads) and pre-allocated block-Thomas factorization/solve.
5. `turb_mlt`: Cox (1968) Mixing-Length Theory (`set_MLT`) using MESA's
   33-component operator-overloaded `auto_diff_real_star_order1` type across 1
   and 4 OpenMP threads.
6. `turb_tdc`: Kuhfuss (1986) 1D Time-Dependent Convection (`set_TDC`) using
   `auto_diff_real_star_order1` across 1 and 4 OpenMP threads.

### Wave 2 Workloads (`run_wave2_bench`)

1. `chem_comp`: 8-isotope (`h1`, `he3`, `he4`, `c12`, `n14`, `o16`, `ne20`,
   `mg24`) composition moment reduction (`basic_composition_info` vs. hoisted
   species properties and 4-thread OpenMP).
2. `ion_fe56`: Potekhin ${}^{56}\mathrm{Fe}$-in-${}^4\mathrm{He}$ effective
   charge interpolation on the $105 \times 30$ Thomas-Fermi-Dirac-Hubbard grid
   (`eval_charge_of_Fe56_in_He4` and direct `interp_evbicub_db` with partials).
3. `ion_4d`: 28-channel 4D $(Z, X, \log_{10}\rho, \log_{10}T)$
   trilinear-over-bicubic mean ionic charge evaluation (`eval_ionization`).
4. `colors_sed`: 3D `Kurucz2003all` cubic Hermite SED synthesis (`1199`
   wavelengths) plus bolometric integration and 7-band `Generic/Johnson` (`B, I,
   J, M, R, U, V`) Vega synthetic photometry.

### Wave 3a Workloads (`run_wave3a_bench`)

1. `eos_ideal`: Analytical multi-species neutral ideal gas plus blackbody
   radiation equation of state (`get_ideal_for_eosdt` / `eval_ideal_eos`) with
   full 26-slot thermodynamic state and $(\ln\rho, \ln T)$ sensitivities.
2. `eos_helm`: Timmes & Swesty (2000) Helmholtz electron-positron-ion-radiation
   equation of state (`get_helm_eos_results`) on the $541 \times 201$
   biquintic free-energy and bicubic pressure-ionization/Coulomb grid.
3. `eos_pc`: Potekhin & Chabrier (2010) non-ideal Coulomb plasma and bcc
   crystallization equation of state (`Get_PC_Results`) with analytical
   Potekhin-Chabrier electron fits (`chemfit` / `elect11`) and linear-mixing
   rule across an 8-isotope mixture.
4. `eos_skye`: Jermyn et al. (2021) `Skye` free-energy automatic-differentiation
   equation of state (`Get_Skye_Results`) with ground-state NIST ionization
   offsets, biquintic/bicubic `HELM` electron-positron interpolation, non-ideal
   liquid/solid mixing, quantum anharmonic vibrational free energy, and smoothed
   latent heat across crystallization.

## Building and Running

Prerequisite: a compiled MESA checkout with `libconst.a`, `libutils.a`,
`libmath.a`, `libauto_diff.a`, `libinterp_1d.a`, `libinterp_2d.a`, `libmtx.a`,
`libnum.a`, `libchem.a`, `libneu.a`, `libturb.a`, `libionization.a`,
`libcolors.a`, and `libeos.a` installed under `$(MESA_DIR)/lib` and `.mod` files
under `$(MESA_DIR)/include`.

```bash
make -C benchmarks/fortran_mesa MESA_DIR=/path/to/mesa run
make -C benchmarks/fortran_mesa MESA_DIR=/path/to/mesa run_wave2
make -C benchmarks/fortran_mesa MESA_DIR=/path/to/mesa run_wave3a
```
