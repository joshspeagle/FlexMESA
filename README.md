# FlexMESA

**FlexMESA** is a modern, modular, and verifiable re-engineering of [MESA (Modules for Experiments in Stellar Astrophysics)](https://github.com/MESAHub/mesa) designed for scientific readability, strict type and memory safety, automatic differentiation, and hardware-accelerated throughput.

## Project Goals

1. **Modularity and Readability (`<= 400` LOC per file)**: Decompose MESA's monolithic Fortran modules, textual `.inc` and `.dek` includes, and 6,008-field `star_info` state container into small, self-contained, pure functional kernels with explicit state structs and comprehensive documentation (`>= 20%` doc ratio with physical units and array shapes).
2. **Dual-Track Architectural Evaluation (Wave 1 Pilot)**: Implement and benchmark MESA's foundational numerical and stateless microphysics modules across both **Safe Rust** (`#![forbid(unsafe_code)]`, `crates/`) and **Python / JAX (`float64`)** (`src/flexmesa/`) to evaluate numerical parity, readability, ergonomics, and compiler/JIT performance before selecting the long-term backend architecture for the 1D stellar evolution engine.
3. **Two-Tier Numerical Verification**: Validate every migrated module against MESA's 21 standalone module unit-test suites (`18,885` lines of 16-digit golden outputs), exact forward and reverse automatic differentiation checks (`jax.test_util.check_grads`), property-based fuzzing (`hypothesis` and `proptest`), and MESA's 107 integration test cases.

## Repository Layout

| Directory | Contents |
| :--- | :--- |
| [`src/flexmesa/`](src/flexmesa/) | Python and JAX (`float64`) implementation of MESA modules |
| [`crates/`](crates/) | Safe Rust (`#![forbid(unsafe_code)]`) Cargo workspace crates |
| [`tests/`](tests/) | Unit, doctest, autodiff, and `hypothesis` property test suites |
| [`benchmarks/`](benchmarks/) | Performance benchmark harnesses comparing Rust, Python/NumPy, and JAX |
| [`docs/design/`](docs/design/) | MESA codebase census, 12-tier dependency DAG, and migration architecture ([`scoping-and-architecture.md`](docs/design/scoping-and-architecture.md), [`wave-0-spec.md`](docs/design/wave-0-spec.md)) |
| [`notes/`](notes/) | Active roadmap ([`PLAN.md`](notes/PLAN.md)), architectural decisions ([`DECISIONS.md`](notes/DECISIONS.md)), open questions ([`QUESTIONS.md`](notes/QUESTIONS.md)), and milestone log ([`LOG.md`](notes/LOG.md)) |

## Development Quickstart

```bash
# Install in editable mode
pip install -e .

# Run fast unit and doctest suite
pytest

# Run static analysis, formatting, and strict type-checking
ruff check .
ruff format --check --line-length 80 .
mypy --strict src tests

# Run Safe Rust workspace tests and clippy checks
cargo test --workspace --all-targets
cargo test --workspace --doc
cargo clippy --workspace --all-targets -- -D warnings
```

## License & Upstream Attribution

`FlexMESA` is distributed under the **GNU Lesser General Public License v3.0 or later (`LGPL-3.0-or-later`)**; see [`LICENSE`](LICENSE) for full terms. Physical equations, numerical algorithms, reference test cases, and converted physics tables derive from [MESA (Modules for Experiments in Stellar Astrophysics)](https://github.com/MESAHub/mesa) by Bill Paxton, Frank Timmes, Josiah Schwab, Evan Bauer, Adam Jermyn, and the MESA Team.
