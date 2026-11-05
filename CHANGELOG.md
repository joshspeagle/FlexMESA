# Changelog

All notable changes to `FlexMESA` are documented in this file.

## [Unreleased]

### Added
- Initial repository scaffolding, `CLAUDE.md` contributor instructions, GitHub issue and pull request templates, `pyproject.toml`, and `Cargo.toml` workspace configuration (#1).
- MESA codebase census, 12-tier dependency DAG, and 6-wave migration architecture in `docs/design/scoping-and-architecture.md` and `notes/` project tracking documents (#1).
- Wave 0 technical specification (`docs/design/wave-0-spec.md`) and MESA `test_output` golden baseline parser and numerical comparison harness in Python (`flexmesa.testing.golden`) and Safe Rust (`crates/flexmesa-testing`) (#7).
- Declarative MESA `<pkg>/defaults/*.defaults` and `*_columns.list` schema parser and legacy Fortran `&namelist` (`inlist`) validator in Python (`flexmesa.schema`) and Safe Rust (`crates/flexmesa-schema`) (#8).
