# FlexMESA

**FlexMESA** is a modern, modular, and verifiable re-engineering of [MESA (Modules for Experiments in Stellar Astrophysics)](https://github.com/MESAHub/mesa) built on an **Integrated Three-Tier Hybrid Architecture** combining **Safe Rust (`#![forbid(unsafe_code)]`)**, zero-copy **Python (`PyO3` / `flexmesa._rust`)**, and **JAX (`float64` `@jax.jit` + `jax.vmap`)**.

## Project Goals

1. **Modularity and Readability (`<= 400` LOC per file)**: Decompose MESA's monolithic Fortran modules, textual `.inc` and `.dek` includes, and 6,008-field `star_info` state container into small, self-contained, pure functional kernels with explicit state structs and comprehensive documentation (`>= 20%` doc ratio with physical units and array shapes).
2. **Integrated Three-Tier Hybrid Architecture**:
   - **Tier A (`utils`, `schema`, `data`, `testing`, `auto_diff`)**: Canonical implementation in Safe Rust (`crates/`) with zero-copy `.safetensors` memory-mapping (`memmap2`) and a persistent `rayon` worker pool, exposed to Python via `flexmesa._rust`.
   - **Tier B (`const`, `math`, `interp_1d`, `interp_2d`, `neu`, `turb`, plus Waves 2–3 microphysics)**: Monomorphized Safe Rust CPU kernels (`f64` + `DualOrder1<N>`) paired with branchless `@jax.jit` flat-array GPU kernels behind a unified `backend="auto" | "rust" | "jax"` Python dispatcher.
   - **Tier C (`mtx`, `num`, `net`, `star_solver`)**: Sequential block-Thomas and parallel Hirshman et al. (2010) `bcyclic` odd-even block cyclic reduction in Safe Rust, paired with $\lceil \log_2 N_{\mathrm{zones}} \rceil$ vectorized Parallel Cyclic Reduction (`solver_mode="pcr"`) and analytical Implicit Function Theorem `@jax.custom_vjp` FFI bindings (`solver_mode="rust_ffi"`) in JAX.
3. **Two-Tier Numerical Verification and Mandatory Post-Wave `Phase X.5` Gates**: Validate every migrated module against MESA's 21 standalone module unit-test suites (`18,885` lines of 16-digit golden outputs), exact forward and reverse automatic differentiation checks (`jax.test_util.check_grads`), property-based fuzzing (`hypothesis` and `proptest`), and end-of-wave 4-way parity and performance benchmarks against compiled upstream Fortran MESA static archives (`benchmarks/fortran_mesa/`).

## Repository Layout

| Directory | Contents |
| :--- | :--- |
| [`src/flexmesa/`](src/flexmesa/) | Unified Python, zero-copy `PyO3` (`flexmesa._rust`), and JAX (`float64`) package |
| [`crates/`](crates/) | Safe Rust (`#![forbid(unsafe_code)]`) Cargo workspace crates + `crates/flexmesa-py` (`PyO3` extension) |
| [`tests/`](tests/) | Unit, doctest, hybrid bridge, autodiff, and `hypothesis` property test suites |
| [`benchmarks/`](benchmarks/) | Upstream Fortran MESA (`benchmarks/fortran_mesa/`) and 4-way benchmark harnesses |
| [`docs/`](docs/) | Public MkDocs Material documentation website (`mkdocs.yml`), design specifications (`docs/design/`), module physics & numerics explainers (`docs/modules/`), and publication-styled figures (`docs/assets/figures/`) |
| [`scripts/`](scripts/) | Reproducible figure generation (`scripts/build_docs_figures.py`) |
| [`notes/`](notes/) | Active roadmap ([`PLAN.md`](notes/PLAN.md)), architectural decisions ([`DECISIONS.md`](notes/DECISIONS.md)), open questions ([`QUESTIONS.md`](notes/QUESTIONS.md)), and milestone log ([`LOG.md`](notes/LOG.md)) |

## Development Quickstart

```bash
# Build and install the Safe Rust PyO3 extension (flexmesa._rust) and Python package
maturin develop --release -m crates/flexmesa-py/Cargo.toml
pip install -e . --no-build-isolation

# Run Python, PyO3 hybrid bridge, and JAX unit/doctest suites
pytest

# Run static analysis, formatting, and strict type-checking
ruff check .
ruff format --check --line-length 80 .
mypy --strict src tests

# Run Safe Rust workspace tests and clippy checks
cargo test --workspace --exclude flexmesa-py
cargo clippy --workspace --exclude flexmesa-py -- -D warnings

# Build the public MkDocs Material documentation website
python scripts/build_docs_figures.py
mkdocs build --strict
```

## License & Upstream Attribution

`FlexMESA` is distributed under the **GNU Lesser General Public License v3.0 or later (`LGPL-3.0-or-later`)**; see [`LICENSE`](LICENSE) for full terms. Physical equations, numerical algorithms, reference test cases, and converted physics tables derive from [MESA (Modules for Experiments in Stellar Astrophysics)](https://github.com/MESAHub/mesa) by Bill Paxton, Frank Timmes, Josiah Schwab, Evan Bauer, Adam Jermyn, and the MESA Team.
