# Architectural & Engineering Decisions (`DECISIONS.md`)

Every choice that another developer could reasonably make differently and that affects behavior, public API shape, numerical tolerances, array memory order, dependencies, or workflow is recorded here with its rationale and rejected alternatives.

---

## DECISIONS #1: Dual-Track (`Safe Rust` + `Python/JAX`) Pilot in Wave 1

- **Date**: 2026-11-04
- **Decision**: Proceed on a dual track (`crates/` in Safe Rust and `src/flexmesa/` in Python/JAX `float64`) for Wave 1 (foundational numerics: `const`, `utils`, `math`, `auto_diff`, `interp_1d`, `interp_2d`, `mtx`, `num`, paired with a vertical slice of stateless microphysics such as `neu` and `turb` if needed to evaluate stencil autodiff and astrophysical kernels). Evaluate both implementations head-to-head on numerical parity against MESA's 16-digit `test_output` baselines, code organization, readability, documentation quality, and 4-stage performance staircase before locking in the long-term single or hybrid backend architecture for Waves 2 through 5.
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
