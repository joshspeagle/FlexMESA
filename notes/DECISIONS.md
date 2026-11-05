# Architectural & Engineering Decisions (`DECISIONS.md`)

Every choice that another developer could reasonably make differently and that affects behavior, public API shape, numerical tolerances, array memory order, dependencies, or workflow is recorded here with its rationale and rejected alternatives.

---

## DECISIONS #1: Dual-Track (`Safe Rust` + `Python/JAX`) Pilot in Wave 1 (Including `neu` and `turb`)

- **Date**: 2026-11-04
- **Decision**: Proceed on a dual track (`crates/` in Safe Rust and `src/flexmesa/` in Python/JAX `float64`) for Wave 1 covering all 8 foundational numerical packages (`const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num`) plus two self-contained, zero-table microphysics packages from Tier 3 and Tier 5: **`neu`** (`1,690` LOC, Itoh et al. 5-channel thermal neutrinos with `10,000` golden test lines) and **`turb`** (`1,349` LOC, closed-form Cardano Mixing Length Theory + implicit Newton-Raphson Time-Dependent Convection `TDC` using the 33-variable `star` stencil AD type). Evaluate both implementations head-to-head on numerical parity against MESA's 16-digit `test_output` baselines, code organization, readability, documentation quality, and 4-stage performance staircase before locking in the long-term single or hybrid backend architecture for Waves 2 through 5.
- **Why**: MESA straddles two distinct computational regimes: (1) embarrassingly zone-parallel microphysics, spline interpolation, and 33-variable stencil automatic differentiation (where JAX's `vmap`, `jacfwd`, `jet`, and `custom_root` eliminate 14,188 lines of generated `auto_diff` boilerplate and enable GPU acceleration and end-to-end differentiability), and (2) stateful adaptive mesh refinement (`nz` changing every step), iterative convective premixing, and small-block tridiagonal cyclic reduction (`bcyclic`), where Safe Rust offers zero-warmup CPU execution, strict IEEE-754 bit-for-bit parity, and zero-allocation memory safety. Running Wave 1 (plus stateless microphysics) on both tracks provides direct empirical benchmarks and side-by-side code readability comparisons before committing the 133,590-LOC `star` engine.
- **Alternatives Rejected**:
  - *Committing exclusively to Pure Rust upfront*: Would forfeit empirical measurement of how much code reduction and GPU/autodiff leverage JAX provides on MESA's forward-AD microphysics (`turb`, `skye`, `hydro_eqns`).
  - *Committing exclusively to Pure JAX upfront*: Would forfeit testing strict IEEE-754 bit-for-bit parity (`crlibm` / disabled FMA) and measuring XLA's compilation and sequential/cyclic block-tridiagonal overhead against native Rust CPU kernels.

---

## DECISIONS #2: Native `/tmp` Cache Directories for `pytest`, `ruff`, and `mypy`

- **Date**: 2026-11-04
- **Decision**: Configure `cache_dir = "/tmp/flexmesa-pytest-cache"`, `cache-dir = "/tmp/flexmesa-ruff-cache"`, and `cache_dir = "/tmp/flexmesa-mypy-cache"` in `pyproject.toml`.
- **Why**: Under WSL2 with the repository located on `/mnt/d/Dropbox/GitHub/FlexMESA` (DrvFs / 9P bridge), atomic temporary file renames into `.pytest_cache` and `.ruff_cache` race with the Windows Dropbox sync daemon and fail with `Permission denied (os error 13)`. Staging tool caches on native Linux `/tmp` eliminates 9P rename failures, speeds up incremental checks, and prevents cache churn inside Dropbox (`AGENTS.md` §2.3).
- **Alternatives Rejected**:
  - *Disabling tool caches (`--no-cache`, `-p no:cacheprovider`)*: Slows down incremental `mypy` and `ruff` runs as the module count grows.
  - *Storing `.pytest_cache` and `.ruff_cache` in the repository root on `/mnt/d/Dropbox`*: Triggers intermittent `os error 13` rename failures during Dropbox indexing.

---

## DECISIONS #3: Legacy & External Bundled Subsystem Scope Exclusions (~98,000 LOC)

- **Date**: 2026-11-04
- **Decision**:
  1. **Exclude `pgstar` and `pgbinary` (`~26,000` LOC + `2,957` struct fields on `s% pg`; tracked in `#3`)**: Replace Fortran PGPLOT X11/PNG plotting with a modern Python/Matplotlib diagnostic plotting module (`plot-style` compliant) operating on history and profile structures.
  2. **Exclude `stella` (`36,778` LOC), `adipls` (`17,831` LOC), and bundled `gyre` (`58,414` lines) source ports (tracked in `#4`)**: Decouple at the pulsation export boundary by porting MESA's `GYRE`/`FGONG`/`OSC` structure exporters (`pulse_*.f90`, `3,885` LOC) so external oscillation solvers or a future native differentiable pulsation solver can consume `FlexMESA` models directly.
  3. **Exclude legacy `RSP1` (`star/private/rsp_*.f90`, `10,371` lines; tracked in `#5`) and port only modern `TDC` + `RSP2` (`turb/private/tdc.f90` + `star/private/hydro_rsp2.f90`, `1,699` lines)**: Solve time-dependent turbulent convection and radial stellar pulsations inside the unified coupled Henyey/Newton solver rather than maintaining a second standalone F77-derived pulsation engine.
- **Why**: Removes ~98,000 LOC of legacy Fortran 77 (`COMMON` blocks, `GOTO` statements, `-fno-automatic` implicit `SAVE`) and 1980s X11 plotting boilerplate, cuts nearly half the fields on `star_info`, and focuses the codebase on modern, unified 1D stellar and binary evolution physics while tracking deferred capabilities explicitly in GitHub issues.
- **Alternatives Rejected**:
  - *Porting `pgstar` or `adipls`/`stella` line-by-line*: Would add ~80,000 LOC of non-core UI and legacy F77 code that does not contribute to the core 1D Henyey evolution engine.
  - *Porting both `RSP1` and `TDC`/`RSP2`*: Duplicates nonlinear radial pulsation physics that MESA itself is consolidating into `TDC`/`RSP2`.
