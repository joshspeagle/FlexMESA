# FlexMESA

**FlexMESA** ports [MESA (Modules for Experiments in Stellar Astrophysics)](https://github.com/MESAHub/mesa) into a canonical **Safe Rust** engine (`crates/`, enforcing `#![forbid(unsafe_code)]` across all crates except the sole vendored `crlibm` FFI wrapper `flexmesa-crlibm-sys`, exposed to Python via `PyO3` as `flexmesa._rust`) with a Python front end and a derived `float64` **JAX (`@jax.jit` + `jax.vmap`)** layer (`src/flexmesa/`). The target for `v0.1.0` is a complete, verified port of **MESA release `26.4.1`** (git tag `26.4.1`, Zenodo record `19722306`), including the bundled codes required by its test suites (`notes/SCOPE.md`).

## Architecture & Scope Principles

1. **Canonical Safe Rust Engine (`strict` and `fast` Modes)**: Physics and evolution algorithms live in Safe Rust (`crates/`), with a default hardware-independent `strict` numerics mode (`crlibm 1.0beta4` matching `CRMATH`, no FMA) judged directly against MESA `26.4.1` and a `fast` mode validated against `strict` (`#165`, `DECISIONS #23`).
2. **Python Front End and Derived `float64` JAX Layer**: Python drives configuration, diagnostics, and plotting (`flexmesa.plotting`), while JAX (`src/flexmesa/`) provides forward and reverse automatic differentiation (`jax.grad`, `jax.jacfwd`, `@jax.custom_vjp`) and batched execution (`jax.vmap`) validated against Rust `strict` mode (`#166`).
3. **Evidence-Gated Parity Verification**: Every status table, end-to-end trajectory comparison (`notes/audit/case_table.md`), and dashboard counter below is generated mechanically from the repository, `benchmarks/dashboard/port_manifest.toml`, `benchmarks/history_harness/`, coverage reports, and the GitHub API (`#143`, `#145`, `#174`).

## Development Progress Dashboard

<!-- BEGIN GENERATED DASHBOARD -->
<!-- END GENERATED DASHBOARD -->

## Repository Layout

| Directory | Contents |
| :--- | :--- |
| [`crates/`](crates/) | Canonical Safe Rust Cargo workspace crates (`#![forbid(unsafe_code)]` across all crates except `flexmesa-crlibm-sys`) and `crates/flexmesa-py` (`PyO3` extension) |
| [`src/flexmesa/`](src/flexmesa/) | Python front end, zero-copy `PyO3` bindings (`flexmesa._rust`), and derived `float64` JAX layer |
| [`tests/`](tests/) | Unit, doctest, hybrid bridge, autodiff, and `hypothesis` property test suites |
| [`benchmarks/`](benchmarks/) | Upstream trajectory harness (`benchmarks/history_harness/`), port manifest and dashboard data (`benchmarks/dashboard/`), and Fortran MESA comparators (`benchmarks/fortran_mesa/`) |
| [`docs/`](docs/) | MkDocs Material documentation website (`mkdocs.yml`) |
| [`scripts/`](scripts/) | Reproducible MESA `26.4.1` reference builders, port-manifest builder, and figure scripts |
| [`notes/`](notes/) | Scope and the `v0.1.0` definition ([`SCOPE.md`](notes/SCOPE.md)), decisions in force ([`DECISIONS.md`](notes/DECISIONS.md)), audit registers and generated case table ([`audit/`](notes/audit/)), and unmaintained pre-audit documents ([`archive/`](notes/archive/)) |

## Development Quickstart

```bash
# Build and install the Safe Rust PyO3 extension (flexmesa._rust) and Python package
maturin develop --release -m crates/flexmesa-py/Cargo.toml
pip install -e ".[test,docs]" --no-build-isolation

# Fetch and verify MESA 26.4.1 physics and test-input bundles
flexmesa data fetch --include-test-inputs
flexmesa data verify --include-test-inputs

# Verify the generated case table, port manifest, and progress dashboard
flexmesa harness table --check
flexmesa dashboard check

# Run Python, PyO3, and JAX unit/doctest suites plus static checks
pytest
ruff check src tests
ruff format --check --line-length 80 src tests
mypy --strict src tests

# Run Safe Rust workspace tests and clippy checks
cargo test --workspace --exclude flexmesa-py --release
cargo clippy --workspace --all-targets -- -D warnings

# Build the MkDocs Material documentation website
python scripts/build_docs_figures.py
mkdocs build --strict
```

## License & Upstream Attribution

`FlexMESA` is distributed under the **GNU Lesser General Public License v3.0 or later (`LGPL-3.0-or-later`)**; see [`LICENSE`](LICENSE) for full terms. Physical equations, numerical algorithms, reference test cases, and converted physics tables derive from [MESA (Modules for Experiments in Stellar Astrophysics)](https://github.com/MESAHub/mesa) by Bill Paxton, Frank Timmes, Josiah Schwab, Evan Bauer, Adam Jermyn, and the MESA Team.
