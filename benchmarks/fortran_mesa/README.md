# Upstream Fortran MESA Benchmark Suite (`Wave 1` through `Wave 5`)

This directory contains the native Fortran 2008 benchmark harness comparing
upstream Fortran MESA (`gfortran -O2 -fopenmp`) and reference Fortran kernels
directly against the Safe Rust (`crates/flexmesa-bench`) and Python/NumPy/JAX
(`src/flexmesa/bench`) Wave 1 to Wave 5 implementations across identical
thermodynamic inputs, spline grids, Henyey matrices, microphysics tables,
stellar structure profiles, and asteroseismic grids.

21 of the 28 workloads call compiled upstream MESA `r26.4.1` subroutines in
every stage, 3 Wave 1-2 workloads (`auto_diff`, `mtx_tridiag`, `chem_comp`)
call upstream MESA in Stages 0-1 and local Reference Fortran loops in Stages
2-3 (with `fortran_wave1.txt` also reporting 1-thread and 4-thread MESA
`bcyclic_dble_decsolblk` timings separately), 3 Wave 5 workloads
(`w5_diff_rot`, `w5_binary_rlof`, `w5_astero_seism`) mix compiled MESA
helper calls (`hydro_rotation`, `binary_utils`/`binary_tides`, `astero_lib`/
`astero_support`) with standalone Reference Fortran workload loops, and 1
workload (`w5_riemann_rsp2`) calls zero MESA subroutines and is labeled
`Reference Fortran (non-MESA)`.

## Workloads and Four-Stage Ladder

Every workload runs across production zone counts ($N_{\mathrm{zones}} \in
\{128, 1024, 4096\}$ in Waves 1 to 3b; $\{128, 512, 1024\}$ in Waves 4 and 5)
after checking internal Stage 1..3 agreement against Stage 0 (and comparing
$N_{\mathrm{zones}} = 8$ checksums in Waves 1 to 3b against recorded FlexMESA
regression snapshots):

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

### Wave 3b Workloads (`run_wave3b_bench`)

1. `eos_4d`: 4D OPAL/SCVH tabulated equation of state (`get_opal_for_eosdt`).
2. `kap_fixed`: Type 1 fixed-metal radiative, conductive, and Compton opacity.
3. `rates_screen`: REACLIB triple-alpha/alpha-capture rate (`r_c12_ag_o16`)
   with extended `screen5` nuclear screening.
4. `net_eval`: 8-isotope `basic.net` nuclear reaction network RHS and Jacobian.
5. `atm_table`: `ATM_TABLE_PHOTOSPHERE` atmosphere surface boundary table.

### Wave 4 Workloads (`run_wave4_bench`)

1. `star_micro_vars`: Per-zone EOS, opacity, neutrino, network, and MLT/Brunt
   microphysics sweep over a initialized MESA star structure (`set_vars`).
2. `star_hydro_eqns`: Coupled 1D Henyey structure residual and $4 \times 4$ block
   Jacobian assembly (`set_hydro_vars`).
3. `star_newton_step`: Single linearize-and-solve block-tridiagonal Henyey
   iteration using MESA's cyclic solver.
4. `star_remesh`: Adaptive mesh spacing function evaluation.
5. `star_evolve_step`: Coupled structure, burn, and mix step
   (`do_struct_burn_mix`) on the initialized star handle.

### Wave 5 Workloads (`run_wave5_bench`)

1. `w5_diff_rot`: Roche distortion (`hydro_rotation::a`) and Spruit-Tayler
   dynamo transport (`log_term_power`).
2. `w5_riemann_rsp2`: Reference Fortran 2008 Lagrangian HLLC / RSP2 / Dziembowski
   pulsation comparator (`eval_riemann_rsp2_workload`, non-MESA).
3. `w5_binary_rlof`: Eggleton Roche lobe (`binary_utils::eval_rlobe`), RLOF
   transfer, and tidal synchronization functions (`binary_tides::f3` / `f4`).
4. `w5_astero_seism`: Acoustic scaling, cubic/combined surface corrections
   (`astero_get_cubic_all_freq_corr` / `astero_get_combined_all_freq_corr`), and
   frequency separation ratios (`astero_support::get_frequency_ratios`).

## Building and Running

Prerequisite: a compiled MESA checkout and initialized MESA SDK. The Makefile
auto-detects both classic `./install` layouts (`$(MESA_DIR)/lib/lib*.a`,
`$(MESA_DIR)/include/*.mod`, and `$(MESA_DIR)/<module>/make/*.mod`) and CMake
build layouts (`$(MESA_DIR)/build/<module>/{lib,include,modules}`), and links
MESA SDK `crmath`, `lapack95`, `lapack`, `blas`, `hdf5`, and `pgplot` helpers
when `MESASDK_ROOT` is set.

```bash
make -C benchmarks/fortran_mesa MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk all
make -C benchmarks/fortran_mesa MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk run
make -C benchmarks/fortran_mesa MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk run_wave2
make -C benchmarks/fortran_mesa MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk run_wave3a
make -C benchmarks/fortran_mesa MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk run_wave3b
make -C benchmarks/fortran_mesa MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk run_wave4
make -C benchmarks/fortran_mesa MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk run_wave5
```

Raw outputs from MESA `r26.4.1` and `mesasdk x86_64-linux-26.6.1` are archived
under `benchmarks/raw/26.4.1/`.

## Module-Level Oracle Generators

Standalone Fortran 2008 reference generators compiled against upstream MESA
`r26.4.1` and the MESA SDK (`CRMATH` math backend) live in subdirectories under
`benchmarks/fortran_mesa/`:

- `math_hex_vectors/`: IEEE-754 bit-exact hex reference vectors for
  `flexmesa-math` (`make -C benchmarks/fortran_mesa/math_hex_vectors MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk run`).
- `eos_oracles/`: Component and inverse-solver reference tables for
  `flexmesa-eos` and `flexmesa.eos` (`DFERMI`, `ALFA`, `HELM`, `ideal`, `PC`,
  `Skye`, `OPAL/SCVH`, `FreeEOS`, `CMS`, and `(P, T)` / `(rho, T)` inversions;
  `make -C benchmarks/fortran_mesa/eos_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk run`).
- `atm_oracles/`: Surface atmosphere and `F2-23` outer boundary condition
  reference tables (`t_tau_rel`, `e2`, `paczynski`, `'fixed'`/`'iterated'`/`'varying'`
  `t_tau_atm`, `table`, `table_ab`, `table_blend`, and `irradiated` with
  `skip_partials = .false.`;
  `make -C benchmarks/fortran_mesa/atm_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean`).
- `neu_oracles/`: Itoh et al. (1996) 5-channel thermal neutrino loss and
  analytical partial-derivative reference tables (`1,080` IEEE-754 hex + decimal
  rows across `36` thermodynamic/flag states;
  `make -C benchmarks/fortran_mesa/neu_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean`).
- `rates_oracles/`: Nuclear reaction rate, `screen5` / `Chugunov` screening,
  Coulomb chemical-potential shift, REACLIB detailed-balance, pycnonuclear
  $3\alpha$, and `weaklib` / `Suzuki2016` reference tables (`185` rows;
  `make -C benchmarks/fortran_mesa/rates_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean`).
