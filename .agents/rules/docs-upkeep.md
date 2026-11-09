---
paths:
  - "crates/**/*.rs"
  - "src/flexmesa/**/*.py"
  - "src/flexmesa/_rust.pyi"
  - "docs/**/*.md"
  - "mkdocs.yml"
---

# Mandatory Documentation Upkeep Rule (`docs-upkeep`)

Whenever you add, rename, or modify any public Rust item (`crates/flexmesa-*`), Python/PyO3/JAX module (`src/flexmesa/*` or `src/flexmesa/_rust.pyi`), namelist control, benchmark kernel, or numerical tolerance, you **must** update the corresponding public documentation pages in `docs/` in the **same pull request** and verify that `pytest tests/test_docs_snippets.py` and `mkdocs build --strict` pass with zero warnings.

---

## 1. Code-to-Documentation Mapping Table

Every Cargo crate and Python subpackage maps to specific concept explainers, runnable tutorials, user guides, domain API reference pages, and benchmark/reference ledgers:

| Rust Crates (`crates/`) | Python Subpackages (`src/flexmesa/`) | Background & Physics (`docs/concepts/`) | Tutorials (`docs/tutorials/`) & Guides (`docs/guides/`) | API Reference (`docs/api/`) | Benchmarks & Reference (`docs/benchmarks/`, `docs/reference/`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `flexmesa-py` (dispatch) | `flexmesa`, `flexmesa._dispatch` | `concepts/architecture.md` | `tutorials/06-differentiable-jax-workflows.md`, `guides/backends-and-performance.md` | `api/index.md` | `reference/faq.md` |
| `flexmesa-schema`, `flexmesa-data`, `flexmesa-testing` | `schema`, `data`, `testing` | `concepts/architecture.md` | `guides/inlists-and-models.md` | `api/io-plotting-bench.md` | `benchmarks/tolerance-and-discrepancy-ledger.md`, `reference/faq.md` |
| `flexmesa-const`, `flexmesa-math`, `flexmesa-utils`, `flexmesa-autodiff` | `constants`, `math`, `utils`, `auto_diff` | `concepts/architecture.md` | `tutorials/06-differentiable-jax-workflows.md`, `guides/backends-and-performance.md` | `api/numerics.md` | `benchmarks/overview.md`, `benchmarks/kernel-scorecards.md`, `reference/glossary-and-units.md` |
| `flexmesa-interp1d`, `flexmesa-interp2d`, `flexmesa-mtx`, `flexmesa-num` | `interp_1d`, `interp_2d`, `mtx`, `num` | `concepts/architecture.md` | `tutorials/06-differentiable-jax-workflows.md`, `guides/backends-and-performance.md` | `api/numerics.md` | `benchmarks/overview.md`, `benchmarks/kernel-scorecards.md`, `benchmarks/tolerance-and-discrepancy-ledger.md` |
| `flexmesa-turb` | `turb` | `concepts/convection-mixing-rotation.md` | `tutorials/01-single-star-evolution.md`, `tutorials/06-differentiable-jax-workflows.md` | `api/numerics.md` | `benchmarks/overview.md`, `benchmarks/kernel-scorecards.md`, `reference/references.md` |
| `flexmesa-chem`, `flexmesa-ionization`, `flexmesa-neu`, `flexmesa-rates`, `flexmesa-net` | `chem`, `ionization`, `neu`, `rates`, `net` | `concepts/nuclear-and-neutrinos.md` | `tutorials/03-white-dwarfs-and-diffusion.md`, `tutorials/06-differentiable-jax-workflows.md` | `api/nuclear.md` | `benchmarks/overview.md`, `benchmarks/kernel-scorecards.md`, `reference/references.md` |
| `flexmesa-eos` | `eos` | `concepts/eos.md` | `tutorials/03-white-dwarfs-and-diffusion.md`, `tutorials/06-differentiable-jax-workflows.md` | `api/eos.md` | `benchmarks/overview.md`, `benchmarks/kernel-scorecards.md`, `benchmarks/tolerance-and-discrepancy-ledger.md` |
| `flexmesa-kap`, `flexmesa-atm`, `flexmesa-colors` | `kap`, `atm`, `colors` | `concepts/opacity-and-atmospheres.md` | `tutorials/01-single-star-evolution.md`, `guides/diagnostic-plotting.md` | `api/opacity-atm-colors.md` | `benchmarks/overview.md`, `benchmarks/kernel-scorecards.md`, `reference/references.md` |
| `flexmesa-star-data`, `flexmesa-star` | `star` | `concepts/stellar-structure-and-hydro.md`, `concepts/convection-mixing-rotation.md`, `concepts/asteroseismology.md` | `tutorials/01-single-star-evolution.md`, `tutorials/02-massive-stars-and-hydro.md`, `tutorials/03-white-dwarfs-and-diffusion.md`, `guides/inlists-and-models.md`, `guides/custom-physics-hooks.md` | `api/star.md` | `benchmarks/overview.md`, `benchmarks/kernel-scorecards.md`, `reference/glossary-and-units.md` |
| `flexmesa-binary` | `binary` | `concepts/binaries.md` | `tutorials/04-interacting-binaries.md`, `guides/custom-physics-hooks.md` | `api/binary.md` | `benchmarks/overview.md`, `benchmarks/kernel-scorecards.md`, `reference/references.md` |
| `flexmesa-astero` | `astero` | `concepts/asteroseismology.md` | `tutorials/05-asteroseismology-and-pulsations.md`, `tutorials/06-differentiable-jax-workflows.md` | `api/astero.md` | `benchmarks/overview.md`, `benchmarks/kernel-scorecards.md`, `reference/references.md` |
| `flexmesa-bench` | `plotting`, `bench` | `concepts/architecture.md` | `guides/diagnostic-plotting.md`, `guides/backends-and-performance.md` | `api/io-plotting-bench.md` | `benchmarks/overview.md`, `benchmarks/kernel-scorecards.md` |

---

## 2. Mandatory Pre-Merge Documentation Upkeep Checklist

Before opening or merging any pull request that touches code or documentation:

1. **Synchronize Concept, Tutorial, Guide & API Pages**:
   - Any pull request adding, renaming, or modifying a public class, function, method, control parameter, return tuple, array shape, CGS/solar unit, or physical formula in `crates/flexmesa-*` or `src/flexmesa/*` must update the mapped `docs/concepts/*.md`, `docs/tutorials/*.md`, `docs/guides/*.md`, `docs/api/*.md`, and `docs/reference/glossary-and-units.md` pages in the **same pull request**.
2. **Register New Python Submodules in `docs/api/*.md`**:
   - Because `src/flexmesa/<pkg>/__init__.py` uses Google-style attribute re-exports (`Symbol = submod.Symbol`), `mkdocstrings` (`griffe`) renders classes and functions from explicit submodule directives (`::: flexmesa.<pkg>.<submod>`). Any pull request adding a new public submodule `src/flexmesa/<pkg>/<submod>.py` must add `::: flexmesa.<pkg>.<submod>` to its mapped `docs/api/*.md` page.
3. **Record Numerical Tolerances & Upstream MESA Discrepancies**:
   - Any pull request documenting a numerical tolerance ruling or an upstream Fortran MESA convention discrepancy must update both `notes/DECISIONS.md` and the public ledger in `docs/benchmarks/tolerance-and-discrepancy-ledger.md` (without exposing internal `DECISIONS #N` shorthand in user-facing pages).
4. **Preserve De-Waved User Framing & Public-Facing Prose Hygiene**:
   - User-facing documentation (`docs/index.md`, `docs/getting-started/`, `docs/concepts/`, `docs/tutorials/`, `docs/guides/`, `docs/api/`, `docs/benchmarks/`, `docs/reference/`) must organize content by astrophysical and numerical domain rather than internal porting waves (`Wave 1` through `Wave 5`), and must contain zero prose em-dashes and zero machine-specific filesystem paths.
5. **Pass the Automated Snippet & Strict MkDocs Gates**:
   - Run and pass both the automated documentation test suite and the strict MkDocs site build before merging:
     ```bash
     pytest tests/test_docs_snippets.py -v
     mkdocs build --strict
     ```
