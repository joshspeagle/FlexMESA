# FlexMESA

**FlexMESA** is a modular re-engineering of [MESA (Modules for Experiments in Stellar Astrophysics)](https://github.com/MESAHub/mesa) combining a canonical **Safe Rust** engine (`crates/`, enforcing `#![forbid(unsafe_code)]` across all crates except the sole vendored `crlibm` FFI wrapper `flexmesa-crlibm-sys`), zero-copy **Python (`PyO3` / `flexmesa._rust`)** bindings, and a derived **JAX (`float64` `@jax.jit` + `jax.vmap`)** layer for automatic differentiation and batching.

## Project Goals

1. **Modularity and Readability (`<= 400` LOC per file)**: Decompose MESA's monolithic Fortran modules, textual `.inc` and `.dek` includes, and 6,008-field `star_info` state container into small, self-contained, pure functional kernels with explicit state structs and comprehensive documentation (`>= 20%` doc ratio with physical units and array shapes).
2. **Integrated Three-Tier Hybrid Architecture**:
   - **Tier A (`utils`, `schema`, `data`, `testing`, `auto_diff`)**: Canonical implementation in Safe Rust (`crates/`) with zero-copy `.safetensors` memory-mapping (`memmap2`) and a persistent `rayon` worker pool, exposed to Python via `flexmesa._rust`.
   - **Tier B (`const`, `math`, `interp_1d`, `interp_2d`, `neu`, `turb`, plus Waves 2–3 microphysics)**: Monomorphized Safe Rust CPU kernels (`f64` + `DualOrder1<N>`) paired with branchless `@jax.jit` flat-array GPU kernels behind a unified `backend="auto" | "rust" | "jax"` Python dispatcher.
   - **Tier C (`mtx`, `num`, `net`, `star_solver`)**: Sequential block-Thomas and parallel Hirshman et al. (2010) `bcyclic` odd-even block cyclic reduction in Safe Rust, paired with $\lceil \log_2 N_{\mathrm{zones}} \rceil$ vectorized Parallel Cyclic Reduction (`solver_mode="pcr"`) and analytical Implicit Function Theorem `@jax.custom_vjp` FFI bindings (`solver_mode="rust_ffi"`) in JAX.
3. **Single-Source Strict/Fast Numerics & Evidence-Gated Verification (`#143`, `#165`, `#166`)**: Build hardware-independent `strict` mode (default; vendored `crlibm 1.0beta4` matching MESA `26.4.1` `CRMATH` rounding and operation order, no FMA) and `fast` mode (platform `f64` intrinsics and `mul_add`, validated against `strict`) from one Rust physics source. Only `strict` mode is compared against MESA `26.4.1` across three levels: (1) cross-machine and cross-thread bit-identity on `x86_64` and `aarch64` CI runners, (2) bit-identical kernel outputs against exact hexadecimal MESA vectors where operation order matches, and (3) full-trajectory agreement within `#143` history tolerances (`notes/audit/case_table.md`).

## Repository Layout

| Directory | Contents |
| :--- | :--- |
| [`src/flexmesa/`](src/flexmesa/) | Unified Python front end, zero-copy `PyO3` (`flexmesa._rust`), and derived JAX (`float64`) package |
| [`crates/`](crates/) | Canonical Safe Rust Cargo workspace crates (`#![forbid(unsafe_code)]` across all crates except `flexmesa-crlibm-sys`) + `crates/flexmesa-py` (`PyO3` extension) |
| [`tests/`](tests/) | Unit, doctest, hybrid bridge, autodiff, and `hypothesis` property test suites |
| [`benchmarks/`](benchmarks/) | Upstream Fortran MESA (`benchmarks/fortran_mesa/`) and 4-way benchmark harnesses |
| [`docs/`](docs/) | Public MkDocs Material documentation website (`mkdocs.yml`), design specifications (`docs/design/`), module physics & numerics explainers (`docs/modules/`), and publication-styled figures (`docs/assets/figures/`) |
| [`scripts/`](scripts/) | Reproducible figure generation (`scripts/build_docs_figures.py`) |
| [`notes/`](notes/) | Scope and the `v0.1.0` definition ([`SCOPE.md`](notes/SCOPE.md)), decisions in force ([`DECISIONS.md`](notes/DECISIONS.md)), the audit and the generated case table ([`audit/`](notes/audit/)), and unmaintained pre-audit documents ([`archive/`](notes/archive/)) |

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
