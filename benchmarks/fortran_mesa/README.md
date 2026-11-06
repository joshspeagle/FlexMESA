# Upstream Fortran MESA Wave 1 Benchmark Suite

This directory contains the native Fortran 2008 benchmark harness comparing
upstream Fortran MESA (`gfortran -O2 -fopenmp`) directly against the Safe Rust
(`crates/flexmesa-bench`) and Python/NumPy/JAX (`src/flexmesa/bench`) Wave 1
implementations across identical thermodynamic profiles, spline grids,
block-tridiagonal Henyey matrices, and convective transport states.

## Workloads and Four-Stage Ladder

Every workload is evaluated across radial shell counts
$N_{\mathrm{zones}} \in \{128, 1024, 4096\}$ and verified at
$N_{\mathrm{zones}} \in \{8, 32\}$ against the pinned cross-language reference
checksums:

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

## Building and Running

Prerequisite: a compiled MESA checkout with `libconst.a`, `libutils.a`,
`libmath.a`, `libauto_diff.a`, `libinterp_1d.a`, `libinterp_2d.a`, `libmtx.a`,
`libnum.a`, `libchem.a`, `libneu.a`, and `libturb.a` installed under
`$(MESA_DIR)/lib` and `.mod` files under `$(MESA_DIR)/include`.

```bash
make -C benchmarks/fortran_mesa MESA_DIR=/path/to/mesa run
```
