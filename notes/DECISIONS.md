# Architectural & Engineering Decisions (`DECISIONS.md`)

Every choice that another developer could reasonably make differently and that affects behavior, public API shape, numerical tolerances, array memory order, dependencies, or workflow is recorded here with its rationale and rejected alternatives.

---

## DECISIONS #1: Dual-Track (`Safe Rust` + `Python/JAX`) Pilot in Wave 1 (Including `neu` and `turb`)

- **Date**: 2026-11-04
- **Decision**: Proceed on a dual track (`crates/` in Safe Rust and `src/flexmesa/` in Python/JAX `float64`) for Wave 1 covering all 8 foundational numerical packages (`const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num`) plus the Tier 3 and Tier 5 stateless microphysics packages **`neu`** (`1,690` LOC, Itoh et al. 5-channel thermal neutrinos with `10,000` golden test lines) and **`turb`** (`1,349` LOC, closed-form Cardano Mixing Length Theory + implicit Newton-Raphson Time-Dependent Convection `TDC` using the 33-variable `star` stencil automatic differentiation type). Evaluate both implementations head-to-head on numerical parity against MESA's 16-digit `test_output` baselines, code organization, readability, documentation quality, and 4-stage performance benchmarks before locking in the long-term single or hybrid backend architecture for Waves 2 through 5.
- **Why**: MESA couples (1) embarrassingly zone-parallel microphysics, spline interpolation, and 33-variable stencil automatic differentiation (where JAX's `vmap`, `jacfwd`, `jet`, and `custom_root` eliminate 14,188 lines of generated `auto_diff` boilerplate and enable GPU acceleration and end-to-end differentiability), and (2) stateful adaptive mesh refinement (`nz` changing every step), iterative convective premixing, and small-block tridiagonal cyclic reduction (`bcyclic`), where Safe Rust offers zero-warmup CPU execution, strict IEEE-754 bit-for-bit parity, and zero-allocation memory safety. Running Wave 1 (plus stateless microphysics) on both tracks provides direct empirical benchmarks and side-by-side code readability comparisons before committing the 133,590-LOC `star` engine.
- **Alternatives Rejected**:
  - *Committing exclusively to Pure Rust upfront*: Would forfeit empirical measurement of how much code reduction and GPU/autodiff leverage JAX provides on MESA's forward-AD microphysics (`turb`, `skye`, `hydro_eqns`).
  - *Committing exclusively to Pure JAX upfront*: Would forfeit testing strict IEEE-754 bit-for-bit parity (`crlibm` / disabled FMA) and measuring XLA's compilation and sequential/cyclic block-tridiagonal overhead against native Rust CPU kernels.

---

## DECISIONS #2: Legacy & External Bundled Subsystem Scope Exclusions (~98,000 LOC)

- **Date**: 2026-11-04
- **Decision**:
  1. **Exclude `pgstar` and `pgbinary` (`~26,000` LOC + `2,957` struct fields on `s% pg`; tracked in `#3`)**: Replace Fortran PGPLOT X11/PNG plotting with a modern Python/Matplotlib diagnostic plotting module (`flexmesa.plotting`) operating on history and profile structures.
  2. **Exclude `stella` (`36,778` LOC), `adipls` (`17,831` LOC), and bundled `gyre` (`58,414` lines) source ports (tracked in `#4`)**: Decouple at the pulsation export boundary by porting MESA's `GYRE`, `FGONG`, and `OSC` structure exporters (`pulse_*.f90`, `3,885` LOC) so external oscillation solvers or a future native differentiable pulsation solver can consume `FlexMESA` models directly.
  3. **Exclude legacy `RSP1` (`star/private/rsp_*.f90`, `10,371` lines; tracked in `#5`) and port only modern `TDC` + `RSP2` (`turb/private/tdc.f90` + `star/private/hydro_rsp2.f90`, `1,699` lines)**: Solve time-dependent turbulent convection and radial stellar pulsations inside the unified coupled Henyey/Newton solver rather than maintaining a second standalone F77-derived pulsation engine.
- **Why**: Removes ~98,000 LOC of legacy Fortran 77 (`COMMON` blocks, `GOTO` statements, `-fno-automatic` implicit `SAVE`) and 1980s X11 plotting boilerplate, cuts nearly half the fields on `star_info`, and focuses the codebase on modern, unified 1D stellar and binary evolution physics while tracking deferred capabilities explicitly in GitHub issues.
- **Alternatives Rejected**:
  - *Porting `pgstar` or `adipls`/`stella` line-by-line*: Would add ~80,000 LOC of non-core UI and legacy F77 code that does not contribute to the core 1D Henyey evolution engine.
  - *Porting both `RSP1` and `TDC`/`RSP2`*: Duplicates nonlinear radial pulsation physics that MESA itself is consolidating into `TDC` and `RSP2`.

---

## DECISIONS #3: Physics Data Table Format (`.safetensors`), GitHub Releases Hosting, & Wave 3a/3b EOS Staging

- **Date**: 2026-11-04
- **Decision**:
  1. **Runtime Format (`.safetensors`)**: Convert MESA's ASCII and HDF5 physics data tables and pre-fitted spline coefficient arrays in Wave 0 into contiguous, 64-byte-aligned `float64` `.safetensors` files stored outside Git in `~/.cache/flexmesa/data/` (or `$FLEXMESA_DATA_DIR`). Both Rust (`safetensors` + `memmap2`) and Python/JAX (`safetensors.numpy`) memory-map (`mmap`) these files directly from the OS page cache with sub-millisecond startup latency and zero C-library dependencies.
  2. **Distribution & Hosting (`GitHub Release Assets` + SHA-256 Manifest + Local `$MESA_DIR` Builder)**: Keep the Git repository free of Git LFS binary blobs. Commit a lightweight `SHA-256` manifest (`src/flexmesa/data/manifest.toml`) and distribute modular `zstd`-compressed `.safetensors.zst` bundles via **GitHub Release Assets**, fetched automatically on first use or via `flexmesa data fetch`, alongside a local conversion command (`flexmesa data build --from-mesa $MESA_DIR`) for offline builds from an upstream MESA checkout.
  3. **Two-Subwave EOS Staging (`Wave 3a` $\to$ `Wave 3b`)**: Stage the 7-component EOS migration into **Wave 3a** (`Ideal + HELM + PC + Skye` + $C^2$ quintic smootherstep blending, requiring only the single `helm_table.safetensors` grid and 3rd-order forward automatic differentiation) followed by **Wave 3b** (`FreeEOS + OPAL/SCVH + CMS` 4D $(Z, X, \log T, \log Q)$ bicubic spline tensor evaluators).
- **Why**: MESA's 6 primary `.xz` table archives (`1.91 GiB` compressed) expand to `10.3 GiB` of ASCII text (`1.43 GiB` for `helm_table.dat` alone, which shrinks to `454 MB` in raw binary `float64`). Storing `2.18 GiB` in Git LFS would immediately exceed standard free LFS quotas, whereas GitHub Release Assets provide free storage and bandwidth up to `2.0 GiB` per file. Using `.safetensors` over HDF5 eliminates `libhdf5-dev` C linker and thread-mutex bottlenecks on the Rust track while enabling zero-copy shared OS page-cache `mmap` across parallel worker processes.
- **Alternatives Rejected**:
  - *Storing data tables in Git LFS*: Exceeds GitHub's free LFS quota on a single clone and forces every contributor and CI job to download `2.2 GiB` upfront.
  - *Using compressed HDF5 (`.h5`) as the primary runtime format*: Introduces native `libhdf5` C build/ABI dependencies and global API mutex locks in Rust, and prevents zero-copy OS `mmap` when chunk compression is enabled.
  - *Blocking all of `eos` on the 4D `FreeEOS`/`OPAL`/`CMS` tables*: Delays testing ionized stellar interior physics (`HELM + Skye`) that only need `helm_table`.

---

## DECISIONS #4: `LGPL-3.0-or-later` License & Portable Tooling Configuration

- **Date**: 2026-11-04
- **Decision**:
  1. **Repository License (`LGPL-3.0-or-later`)**: License `FlexMESA` under the **GNU Lesser General Public License v3.0 or later (`LGPL-3.0-or-later`)** matching MESA's root `LICENSE`, with explicit upstream attribution to the MESA Team and module authors.
  2. **Portable Project Configuration (`pyproject.toml` and `Cargo.toml`)**: Keep `pyproject.toml` and `Cargo.toml` free of machine-specific paths or local directory overrides so the repository builds and tests cleanly on any Linux, macOS, or Windows environment.
- **Why**: Adopting `LGPL-3.0-or-later` ensures full license compatibility with MESA's derived physics equations, test oracles, and converted data tables while permitting library linking by downstream users. Keeping build and test configuration files free of hardcoded local paths ensures reproducible builds across contributors and CI runners.
- **Alternatives Rejected**:
  - *Choosing a non-copyleft license (MIT/BSD/Apache-2.0)*: Incompatible with translating and redistributing `LGPL-3.0-or-later` and `GPL-3.0-or-later` algorithms and derived tables from MESA.
  - *Hardcoding local cache directories in `pyproject.toml` or `.cargo/config.toml`*: Leaks developer-specific filesystem assumptions into public configuration files instead of using standard environment variables (`RUFF_CACHE_DIR`, `MYPY_CACHE_DIR`, `CARGO_TARGET_DIR`) when needed locally.

---

## DECISIONS #5: Issue-First Wave Decomposition & Pre-Merge Verification Gate

- **Date**: 2026-11-04
- **Decision**: Structure each migration wave (`Waves 0 through 5`) as a parent milestone issue (`phase` label) decomposed into focused module or sub-wave issues (`enhancement` label), each implemented on its own `<issue>-<slug>` branch and pull request (`<= 400` LOC per file). Every pull request must pass unit tests, doctests, property tests (`hypothesis` and `proptest`), static analyzers (`ruff`, `mypy --strict`, `cargo clippy`), a code simplification pass, and a multi-angle code review before merging into `main`.
- **Why**: Breaking large migration waves into self-contained module pull requests keeps diffs auditable, isolates numerical regressions to a single MESA package, and ensures every merged commit maintains full test and golden parity coverage.
- **Alternatives Rejected**:
  - *Landing an entire migration wave in a single monolithic pull request*: Creates multi-thousand-line diffs that are difficult to review and debug.
