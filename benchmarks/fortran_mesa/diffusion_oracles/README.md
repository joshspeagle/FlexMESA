# MESA 26.4.1 Element Diffusion and Phase Separation Reference Oracles (`#156`)

This directory contains the standalone Fortran generator (`gen_diffusion_oracles.f90`), build recipe (`Makefile`), and committed IEEE-754 double-precision reference table (`mesa_26_4_1_diffusion_oracles.tsv`) captured directly against compiled MESA release `26.4.1` (`r26.4.1`, commit `c8c202662922a22b3efefa824473ad615c623afd`) with `MESA SDK x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH 1.2` / `crlibm 1.0beta4`).

## Regenerating the Oracles

With `MESASDK_ROOT` pointing to the MESA SDK and `MESA_DIR` pointing to a built MESA `26.4.1` tree:

```bash
make -C benchmarks/fortran_mesa/diffusion_oracles MESA_DIR=/path/to/mesa MESASDK_ROOT=/path/to/mesasdk generate clean
```

## Coverage

- **Section 1 (`paquette`)**: `paquette_coeffs::paquette_coefficients` across four representative thermodynamic states (`psi` spanning low-coupling below-table clamp, negative-`psi` solar interior, moderate-`psi` helium/CO shell, and high-`psi` asymptotic extrapolation above `10.0`), outputting exact IEEE-754 bits for `Ddiff(i,j)`, `Kdiff(i,j)`, `Zdiff(i,j)`, `Zdiff1(i,j)`, `Zdiff2(i,j)`, and `Ath` for ion-ion and ion-electron pairs.
- **Section 2 (`matrix_coeffs`)**: `diffusion_support::get_matrix_coeffs` on a 5-zone, 4-class (`H1`, `He4`, `C12`, `O16`) test grid across all 8 transport and Burgers solver regimes:
  1. `pure_coulomb_thoul` (`pure_Coulomb = .true.`, Thoul/Hu solver)
  2. `paquette_thoul` (Paquette et al. 1986, Thoul/Hu solver)
  3. `sm_thoul` (Stanton & Murillo 2016, Thoul/Hu solver)
  4. `sm_cgs_low_eta` (Stanton & Murillo + CGS solver with thermal diffusion, `eta = 2`)
  5. `sm_cgs_blend_eta` (Stanton & Murillo + CGS solver in the `10 < eta < 40` thermal/non-thermal blend, `eta = 25`)
  6. `sm_cgs_high_eta` (Stanton & Murillo + CGS solver without thermal diffusion, `eta = 60`)
  7. `caplan_blend_cgs` (Caplan, Bauer & Freeman 2022 OCP blend `4 < Gamma < 10`, `Gamma = 6`)
  8. `caplan_pure_cgs` (pure Caplan OCP `Gamma = 50` with relativistic screening `rho = 2.0e6`)
- **Section 3 (`solve_diff`)**: Full `diffusion::do_solve_diffusion` evolution on a 12-zone, 4-class (`H1`, `He4`, `C12`, `Fe56`) solar-like profile for `BE_solve` (`diffusion_use_isolve = .false.`) and all 7 Rosenbrock `solve_with_isolve` solvers (`ros2_solver`, `rose2_solver`, `ros3p_solver`, `ros3pl_solver`, `rodas3_solver`, `rodas4_solver`, `rodasp_solver`), plus a 12-zone degenerate CO+He white-dwarf `BE_solve` (`diffusion_use_cgs_solver = .true.`) profile.
- **Section 4 (`phase_sep`)**: Live MESA `26.4.1` `phase_separation::do_phase_separation` on an 8-zone CO white dwarf (`blouin_co`, `phase_separation_mixing_use_brunt = .false.`), an 8-zone ONe white dwarf (`blouin_one`, `phase_separation_mixing_use_brunt = .false.`), and the same CO white dwarf with Ledoux/Brunt-B stopping enabled (`blouin_co_brunt`, `phase_separation_mixing_use_brunt = .true.`), using the Skye EOS (`phase > 0.9` crystallization front, Blouin 2021a/b phase shifts, constant-`Pgas` EOS updates, `mix_outward`, `eps_phase_separation`, `crystal_core_boundary_mass`, `phase_sep_mixing_mass`, `lnPgas_final`, `gradr_final`, and `grada_face_final`).
- **Section 5 (`kernels`)**: Direct evaluations of private MESA `26.4.1` kernels (`get_Dstar_OCP` with `bitsafe_erf_fit` from `star/private/diffusion_support.f90`, and `blouin_delta_xo` / `blouin_delta_xne` from `star/private/phase_separation.f90`). Because `libstar.a` does not export these private procedures, the `Makefile` extracts their exact source text verbatim from the MESA `26.4.1` tree into a temporary build-directory include file and compiles them inside `gen_diffusion_private_kernels` with the same `FFLAGS` and `CRMATH` math backend.
- The TSV header records `mesa_version`, `mesa_git_commit`, `mesasdk_version`, `math_backend`, and `compiler` (`iso_fortran_env::compiler_version()`). Each floating-point output is recorded both as big-endian 16-hex-digit IEEE-754 raw bits (`0x<16-hex>`) and round-trip decimal (`ES25.17E3`).
