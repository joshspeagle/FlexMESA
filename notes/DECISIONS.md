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

---

## DECISIONS #6: Integrated Three-Tier Hybrid Architecture (`Safe Rust Core via PyO3 + Native @jax.jit Array Kernels`)

- **Date**: 2026-11-06
- **Decision**: Partition `FlexMESA` across three architectural tiers unified behind a single Python API (`backend="auto" | "rust" | "jax"`):
  1. **Tier A (`utils`, `schema`, `data`, `testing`, `auto_diff`)**: Maintain a single canonical implementation in Safe Rust (`crates/` with `#![forbid(unsafe_code)]`, persistent `rayon` worker pool, and zero-copy `.safetensors` memory mapping), exposed to Python via a compiled `PyO3` extension module (`flexmesa._rust`). Prune duplicated pure-Python infrastructure parsers/loaders and scalar Python operator-overloaded dual-number loops.
  2. **Tier B (`const`, `math`, `interp_1d`, `interp_2d`, `neu`, `turb`, `chem`, `ionization`, `colors`, `eos`, `kap`, `rates`, `atm`)**: Implement each physics module as (a) a monomorphized Safe Rust CPU kernel (`f64` and `DualOrder1<N>` with Implicit Function Theorem root differentiation and persistent `rayon` zone parallelism) and (b) a concise, branchless `@jax.jit` flat-array kernel (`jnp.where`, `lax.fori_loop`, `jax.custom_jvp`). Under `backend="auto"` (default), `numpy.ndarray` and float inputs dispatch zero-copy to `flexmesa._rust`, while `jax.Array` and traced inputs dispatch to `@jax.jit`.
  3. **Tier C (`mtx`, `num`, `net`, `star_solver`)**: Provide both sequential block-Thomas and parallel `bcyclic` cyclic reduction in Safe Rust (`crates/flexmesa-mtx`), and provide both native JAX parallel cyclic reduction (`solver_mode="pcr"`) and `flexmesa._rust` FFI solver bindings with analytical `jax.custom_vjp` rules (`solver_mode="rust_ffi"`).
- **Why**: Wave 1 benchmarks against upstream Fortran MESA demonstrated that (1) pure-Python/NumPy loops and Python operator-overloaded dual-number classes are $90\times\text{--}10^6\times$ slower than Safe Rust and are completely superseded by zero-copy `PyO3` dispatch on CPU, while (2) native `@jax.jit` flat-array kernels with Implicit Function Theorem `custom_jvp` rules evaluate 4096 zones of complex microphysics (`neu`, `turb_mlt`, `turb_tdc`) in $316\text{--}362\,\mu\mathrm{s}$ per star on GPU ($160.6\times$ faster than 4-thread Fortran MESA on `turb_tdc`) with exact forward and reverse automatic differentiation.
- **Alternatives Rejected**:
  - *Maintaining separate pure-Python/NumPy fallback implementations alongside Safe Rust and JAX*: Adds thousands of lines of slow, duplicated maintenance burden when `flexmesa._rust` already provides sub-millisecond CPU execution on `numpy.ndarray` inputs.
  - *Writing Wave 2+ physics solely in Safe Rust and calling them from JAX via host callbacks*: Forces PCIe host-device memory transfers on every microphysics call, preventing XLA kernel fusion and GPU `vmap` scaling.

---

## DECISIONS #7: Block-Tridiagonal & Stiff Solver Strategy (`Odd-Even bcyclic + Native JAX PCR + Rust FFI custom_vjp`)

- **Date**: 2026-11-06
- **Decision**:
  1. In Safe Rust (`crates/flexmesa-mtx`), implement Hirshman et al. (2010) odd-even parallel block cyclic reduction (`solve_block_tridiag_bcyclic`) alongside sequential block-Thomas (`solve_block_tridiagonal`), parallelized over a persistent `rayon` thread pool.
  2. In Python/JAX (`src/flexmesa/mtx` and `src/flexmesa/num`), support both (a) **native JAX Parallel Cyclic Reduction (`solver_mode="pcr"`)**, which reduces a block-tridiagonal system in $\lceil \log_2 N_{\mathrm{zones}} \rceil$ vectorized steps (`12` steps at $N_{\mathrm{zones}} = 4096$) entirely on GPU without sequential `lax.scan` overhead, and (b) **Safe Rust FFI solver dispatch (`solver_mode="rust_ffi"`)** wrapped with an analytical Implicit Function Theorem `jax.custom_vjp` rule solving $A^\top \lambda = \bar{x}$ and $\bar{A} = -\lambda x^\top$.
- **Why**: Sequential `lax.scan` block-Thomas on GPU requires $N_{\mathrm{zones}} = 4096$ sequential kernel launches/steps ($1619\,\mathrm{ms}$), whereas $\lceil \log_2 N_{\mathrm{zones}} \rceil$ parallel cyclic reduction reduces dependency depth by $340\times$ on GPU, and the `flexmesa._rust` FFI binding with `custom_vjp` provides sub-millisecond exact solves and adjoints on CPU or hybrid workflows.
- **Alternatives Rejected**:
  - *Sequential `lax.scan` block-Thomas as the sole JAX solver*: Creates a $>1.6\,\mathrm{s}$ bottleneck per Henyey iteration on GPU.
  - *Host FFI callback only (no native JAX PCR)*: Prevents pure on-GPU `jax.vmap` batch evolution of stellar grids.

---

## DECISIONS #8: Standing `Wave X.5` Consolidation, Optimization, Upstream Fortran MESA Gate & Public MkDocs Material Website

- **Date**: 2026-11-06
- **Decision**:
  1. **Explicit `Wave X.5` Stage at the End of Every Wave (`Wave 1.5`, `Wave 2.5`, `Wave 3a.5`, `Wave 3b.5`, `Wave 4.5`, `Wave 5.5`)**: After completing the initial module PRs of any Wave `X`, execute a dedicated **Phase `X.5`** milestone before advancing to Wave `X+1`. Each `Wave X.5` stage (a) compiles and links the corresponding upstream Fortran MESA static libraries in `benchmarks/fortran_mesa/` to run head-to-head 16-digit parity and 4-stage timing benchmarks, (b) conducts a holistic cross-module code and performance review to implement algorithmic and FFI optimizations, (c) loops through both the `FlexMESA` test suite and the compare-to-original-Fortran-MESA benchmark gate after every batch of changes (advancing to the next wave **only** once all Fortran MESA parity and performance gates pass and are posted to the GitHub issue), and (d) syncs the public documentation website.
  2. **Public Documentation Website Stack (`MkDocs Material`)**: Build and maintain a public-facing documentation website using **MkDocs Material** (`mkdocs.yml`, `docs/`, `.github/workflows/docs.yml` for GitHub Pages) with `mkdocstrings-python`, KaTeX math rendering (`pymdownx.arithmatex`), Mermaid architecture diagrams, side-by-side Rust/Python/JAX code tabs, module-by-module physical and numerical explainers, embedded publication-styled diagnostic figures (`scripts/build_docs_figures.py`), live 4-way benchmark scorecards, and synced Python/JAX + Rust (`cargo doc`) API references.
- **Why**: Individual module PR checks verify local correctness, but cross-module bottlenecks, multi-core scaling floors, and upstream Fortran MESA comparisons only emerge when the full wave is benchmarked holistically against compiled Fortran MESA. Building the public MkDocs Material website incrementally during each `Wave X.5` gate ensures the public documentation, equations, diagnostic plots, and API reference remain strictly synchronized with the verified codebase.
- **Alternatives Rejected**:
  - *Relying solely on per-PR unit tests without a post-wave Fortran MESA benchmark and optimization stage*: Misses systemic performance opportunities (such as persistent `rayon` pools, `PyO3` consolidation, and `bcyclic` cyclic reduction) and risks drifting from upstream Fortran MESA performance.
  - *Deferring public documentation to the end of Wave 5*: Leaves the project without accessible public explainers, architecture guides, and visual diagnostics during active development.
