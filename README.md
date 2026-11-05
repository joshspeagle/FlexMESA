# FlexMESA

**FlexMESA** is a modern, modular, and verifiable re-engineering of [MESA (Modules for Experiments in Stellar Astrophysics)](https://github.com/MESAHub/mesa) designed for scientific readability, strict type and memory safety, automatic differentiation, and hardware-accelerated throughput.

## Project Goals

1. **Modularity and Readability (`<= 400` LOC per file)**: Decompose MESA's monolithic Fortran modules, textual `.inc`/`.dek` includes, and 6,008-field `star_info` state container into small, self-contained, pure functional kernels with explicit state structs and comprehensive documentation (`>= 20%` doc ratio with physical units and array shapes).
2. **Dual-Track Architectural Evaluation (Wave 1 Pilot)**: Implement and benchmark MESA's foundational numerical and stateless microphysics modules across both **Safe Rust** (`#![forbid(unsafe_code)]`, `crates/`) and **Python / JAX (`float64`)** (`src/flexmesa/`) to evaluate numerical parity, readability, ergonomics, and 4-stage compiler/JIT performance before selecting the long-term backend architecture for the 1D stellar evolution engine.
3. **Two-Tier Numerical Verification**: Validate every migrated module against MESA's 21 standalone module unit-test suites (`18,885` lines of 16-digit golden outputs), exact forward/reverse automatic differentiation checks (`check_grads`), property-based fuzzing (`hypothesis` and `proptest`), and MESA's 107 integration test cases.

## Repository Layout

| Directory | Contents |
| :--- | :--- |
| [`src/flexmesa/`](src/flexmesa/) | Python and JAX (`float64`) implementation of MESA modules |
| [`crates/`](crates/) | Safe Rust (`#![forbid(unsafe_code)]`) Cargo workspace crates |
| [`tests/`](tests/) | Unit, doctest, autodiff, and `hypothesis` property test suites |
| [`benchmarks/`](benchmarks/) | 4-stage performance staircase harnesses and `code-migrate` telemetry |
| [`docs/design/`](docs/design/) | Stage 0/1 MESA codebase census, 12-tier dependency DAG, and migration architecture ([`scoping-and-architecture.md`](docs/design/scoping-and-architecture.md)) |
| [`notes/`](notes/) | Active roadmap ([`PLAN.md`](notes/PLAN.md)), architectural decisions ([`DECISIONS.md`](notes/DECISIONS.md)), open questions ([`QUESTIONS.md`](notes/QUESTIONS.md)), and session logs ([`LOG.md`](notes/LOG.md)) |

## Development Quickstart

Using the shared `~/.venvs/default` Python environment:

```bash
# Run fast unit and doctest suite
PYTHONPATH=src /home/joshspeagle/.venvs/default/bin/pytest

# Run static analysis, formatting, type-checking, and Google style checks
~/.local/bin/ruff check .
~/.local/bin/ruff format --check --line-length 80 .
~/.local/bin/mypy --strict src tests
/home/joshspeagle/.venvs/default/bin/python /mnt/d/Dropbox/.agents/skills/code-style/scripts/check_code_style.py src tests
```
