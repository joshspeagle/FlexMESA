# Open Architectural Questions (`QUESTIONS.md`)

This file tracks open design questions for our step-by-step walkthrough with Josh, each with a recommended default option.

---

## Question 1 (Supplement to `DECISIONS #1`): Vertical-Slice Microphysics in Wave 1 Dual-Track Pilot

- **Context**: Wave 1 covers MESA's foundational numerical packages (`const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num`). Adding two self-contained, zero-table microphysics modules from Tier 3 and Tier 5, specifically **`neu`** (`1,690` LOC, Itoh et al. 5-channel thermal neutrinos with `10,000` golden test lines) and **`turb`** (`1,349` LOC, Cardano Mixing Length Theory + implicit Newton-Raphson Time-Dependent Convection `TDC` using the 33-variable `star` stencil AD type), exercises the full end-to-end physics + stencil-autodiff pattern without waiting for the 2.18 GiB data tables in Wave 3.
- **Recommended Option**: Include `neu` and `turb` at the end of the Wave 1 Dual-Track pilot so our Rust vs. JAX comparison covers constants/math, 1D/2D splines, block-tridiagonal/stiff solvers, and real 33-variable stellar microphysics.

---

## Question 2: Legacy & External Bundled Subsystem Scope Exclusions (~98,000 LOC)

- **Context**: Nearly 25% of `/tmp/mesa-audit` consists of non-core or external legacy Fortran 77 subsystems:
  1. `star/private/pgstar_*.f90` & `binary/private/pgbinary_*.f90` (`~26,000` LOC of 1980s PGPLOT X11/PNG wrappers).
  2. `stella/` (`36,778` LOC of standalone F77 supernova radiation-hydrodynamics with `1,938` `COMMON` blocks and `1,405` `GOTO` statements, not built in `make/subdirs.mk`).
  3. `adipls/` (`17,831` LOC of F77 adiabatic pulsation code with `357` `COMMON` blocks) and `gyre/` (`58,414` lines in `gyre-9.1.1.tar.gz`).
  4. `star/private/rsp_*.f90` (`10,371` lines of legacy Smolec F77-derived 1D nonlinear pulsation code, which MESA is actively superseding with `turb/tdc.f90` + `hydro_rsp2.f90`).
- **Recommended Option**: Exclude `pgstar`/`pgbinary`, `stella`, `adipls`, and legacy `rsp_*` (`RSP1`) from the migration scope; replace `pgstar` with clean Python/Matplotlib plotting diagnostics, retain modern `TDC`/`RSP2` for pulsations, and export standard `GYRE`/`FGONG` structures (`3,885` LOC in `pulse_*.f90`) for oscillation coupling.

---

## Question 3: Physics Data Tables (`2.18 GiB` LFS) & 7-Component EOS Cascade Strategy

- **Context**: MESA ships `2.18 GiB` of compressed ASCII/HDF5 tables (`1.3 GB` `eosFreeEOS`, `238 MB` `helm_table`, `153 MB` `eosDT`, `186 MB` `kap`, `98 MB` `rates`, `73 MB` `colors`, `14 MB` `atm`), which MESA parses on first run and caches as unformatted Fortran `.bin` spline coefficient files. Meanwhile, `eos` blends 7 components (`CMS > Skye > PC > FreeEOS > OPAL/SCVH > HELM > Ideal`).
- **Recommended Option**: Build a Wave 0 converter that exports MESA's pre-fitted spline coefficient tables into portable, memory-mappable `.safetensors` / HDF5 tensors stored outside Git (on native local storage `/tmp` or a dedicated data directory), and in Wave 3 implement the self-contained analytical/single-table EOS core (`Ideal + HELM + PC + Skye`) first before wiring the 4D tabulated `FreeEOS`/`OPAL`/`CMS` spline tensors.

---

## Question 4: Stage 0 Toolchain Setup on `speagle-desktop` & Repository License

- **Context**: `speagle-desktop` currently has `gcc`/`g++` and `~/.venvs/default` (`Python 3.14`, `jax 0.11.0`, `numpy 2.5.2`, `scipy 1.18.0`, `sympy 1.14.0`, `pytest 9.1.1`, `hypothesis 6.170.14`, `h5py 3.16.0`), while `rustup`/`rustc`/`cargo`, `gfortran`/`MESASDK`, and `ndiff` are not yet installed. Additionally, MESA is licensed under `LGPL-3.0-or-later`.
- **Recommended Option**:
  1. Install `rustup` (stable Rust toolchain in `~/.cargo`) and `gfortran`/MESASDK + `ndiff` so we can both compile Rust crates and execute MESA's Stage 0 Fortran unit tests for live wall-time/memory benchmarking alongside the 18,885 lines of committed `test_output` golden files.
  2. License `FlexMESA` under `LGPL-3.0-or-later` (or `GPL-3.0-or-later`) to maintain full license compatibility with MESA's derived physics equations and data tables.
