# Changelog

All notable changes to `FlexMESA` are documented in this file.

## [Unreleased]

### Added
- Initial repository scaffolding, `CLAUDE.md` contributor instructions, GitHub issue and pull request templates, `pyproject.toml`, and `Cargo.toml` workspace configuration (#1).
- MESA codebase census, 12-tier dependency DAG, and 6-wave migration architecture in `docs/design/scoping-and-architecture.md` and `notes/` project tracking documents (#1).
- Wave 0 technical specification (`docs/design/wave-0-spec.md`) and MESA `test_output` golden baseline parser and numerical comparison harness in Python (`flexmesa.testing.golden`) and Safe Rust (`crates/flexmesa-testing`) (#7).
- Declarative MESA `<pkg>/defaults/*.defaults` and `*_columns.list` schema parser and legacy Fortran `&namelist` (`inlist`) validator in Python (`flexmesa.schema`) and Safe Rust (`crates/flexmesa-schema`) (#8).
- Physics table `SHA-256` manifest (`src/flexmesa/data/manifest.toml`), memory-mapped `.safetensors` (`float64`/`int64`) loader, and ASCII-to-`.safetensors` converters for `lodders03.data`, `isotopes.data`, and `helm_table.dat` in Python (`flexmesa.data`) and Safe Rust (`crates/flexmesa-data`) (#9).
- Wave 1 technical specification (`docs/design/wave-1-spec.md`), CODATA 2018 and IAU 2015 physical and astronomical CGS constants (`flexmesa.constants`, `crates/flexmesa-const`), and core numerical finiteness guards, dictionaries, and reaction-network stream tokenizers (`flexmesa.utils`, `crates/flexmesa-utils`) (#14).
- IEEE-754 `float64` mathematical intrinsics and domain-safe logarithms/roots (`flexmesa.math`, `crates/flexmesa-math`) and forward-mode first-order (`DualOrder1`) and second-order hyper-dual (`AutoDiffReal2VarOrder2`) automatic differentiation (`flexmesa.auto_diff`, `crates/flexmesa-autodiff`) (#15).
- One-dimensional Steffen (1990) piecewise-monotonic and Huynh (1993) monotonicity-preserving cubic spline interpolation, definite integration, stencil remeshing, and `DualOrder1` autodiff support (`flexmesa.interp_1d`, `crates/flexmesa-interp1d`) and two-dimensional $C^2$ bicubic, piecewise-monotonic, and Akima (1996) surface interpolation (`flexmesa.interp_2d`, `crates/flexmesa-interp2d`) (#16).
- Dense, banded, and coupled block-tridiagonal linear algebra solvers, CSR/CSC sparse matrix conversions, and implicit-function-theorem autodiff (`flexmesa.mtx`, `crates/flexmesa-mtx`), alongside compensated summation, 1D Brent/Newton root-finding and minimization, Nelder-Mead simplex optimization, explicit Runge-Kutta (`rk2`, `rk4`, `dopri5`, `dop853`), stiff Rosenbrock/RODAS ODE integrators, and line-search damped Newton-Raphson solvers (`flexmesa.num`, `crates/flexmesa-num`) (#17).
