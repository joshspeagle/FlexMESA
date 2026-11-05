# Open Architectural Questions (`QUESTIONS.md`)

This file tracks open design questions for our step-by-step walkthrough with Josh, each with a recommended default option.

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
