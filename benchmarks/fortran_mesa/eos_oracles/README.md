# MESA 26.4.1 EOS Module Oracle Generator (`#161`, `T2-15`, `T1-11`)

This directory contains the standalone Fortran 2008 oracle generator (`gen_eos_oracles.f90`), build recipe (`Makefile`), and committed reference output table (`mesa_26_4_1_eos_oracles.tsv`) linked directly against compiled MESA release `26.4.1` (`r26.4.1`, git tag `26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`, MESA SDK `x86_64-linux-26.6.1`, `gfortran 15.2.0`, `CRMATH`).

It provides full-provenance upstream reference values replacing all unrecorded inline literals across `crates/flexmesa-eos/tests/` (`T2-15`) and expanding `tests/test_eos_*.py` (`T1-11`).

## Regenerating the Reference Table

With `MESASDK_ROOT` initialized (`source "$MESASDK_ROOT/bin/mesasdk_init.sh"`) and `MESA_DIR` pointing to a built MESA `26.4.1` checkout:

```bash
make -C benchmarks/fortran_mesa/eos_oracles generate
```

## Covered Upstream Subroutines and Record Schemas

Every record in `mesa_26_4_1_eos_oracles.tsv` prints 18-significant-digit IEEE-754 round-trip decimals (`es25.17e3`) under MESA's strict SSE2 floating-point flags (`-march=x86-64 -mtune=generic -mfpmath=sse -O2 -ffp-contract=off -fprotect-parens -fno-sign-zero -fno-range-check`):

- **`DFERMI`** (`eos_lib::eos_fermi_dirac_integral` / `eos/private/gauss_fermi.f90`): 4 `test_dirac_integrals` orders (`dk = -0.5, 0.5, 1.5, 2.5` from `eos/test/src/test_eos_support.f90:660-675`) plus a 96-point `(dk, eta, theta)` grid (`(fd, fdeta, fdtheta)`).
- **`ALFA`** (`eospc_eval::Get_PC_alfa`, `skye::Get_Skye_alfa`, `skye::Get_Skye_alfa_simple`): 15 `(alfa, d_alfa_dlogT, d_alfa_dlogRho)` blend-weight points across `Gamma_e`-limit and `T`-limit `PC` windows, polygonal `Skye` boundaries, and rectangular `Skye` boundaries (both MESA's `eos.defaults` `real(4)` bounds and explicit `real(8)` bounds).
- **`HELM`** (`eos_lib::eos_get_helm_results` + `eos_lib::eos_convert_helm_results`): 17 thermodynamic states across degenerate, ideal, solar, relativistic electron-positron pair, and full-domain regimes (`16` channels x `5` columns: `val, d_dlnd, d_dlnT, d_dabar, d_dzbar`), plus **`IERR`** off-table and crystallized white-dwarf error-code checks (`helmeos2_eval`).
- **`EOS`** (`eos_lib::eosDT_get_component` and `eos_lib::eosDT_get`): 22 full `26`-channel x `3`-column (`res, d_dlnd, d_dlnT`) evaluations covering standalone `Ideal`, `PC` (liquid, `EOSFI8` transition, crystallization phase transition, solid `Ogata`/linear mixing, quantum melting), `Skye` (with and without NIST ground-state ionization offsets, `Ogata` and `PC` solid mixing, phase transition), `OPAL/SCVH`, `FreeEOS`, `CMS` (with and without blackbody radiation), and `PC+HELM`, `PC+FreeEOS`, and `Skye+HELM` blends.
- **`INV`** (`eos_lib::eosPT_get`, `eosDT_get_T`, `eosDT_get_Rho`, `eosPT_get_T`, `eos_gamma_DE_get`): inverse `(P_gas, T)`, `(\rho, E)`, `(\rho, P_gas)`, and ideal $\gamma$-law solver checks.
