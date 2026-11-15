# CLAUDE.md

`FlexMESA` is a modern, modular, and verifiable re-engineering of [MESA (Modules for Experiments in Stellar Astrophysics)](https://github.com/MESAHub/mesa) into a **Three-Tier Hybrid Architecture** combining a canonical Safe Rust engine (`crates/`, exposed to Python via `PyO3` as `flexmesa._rust`) with differentiable Python/JAX (`float64`) array kernels (`src/flexmesa/`).

## Start here

- Read [`notes/PLAN.md`](notes/PLAN.md) (current position and roadmap), the latest entries in [`notes/LOG.md`](notes/LOG.md), and any open architectural questions in [`notes/QUESTIONS.md`](notes/QUESTIONS.md).
- Every settled architectural, numerical, or governance choice is recorded in [`notes/DECISIONS.md`](notes/DECISIONS.md).
- The MESA codebase census, 12-tier dependency graph, and migration blueprint live in [`docs/design/scoping-and-architecture.md`](docs/design/scoping-and-architecture.md) and specifications in [`docs/design/`](docs/design/).
- Whenever modifying public Rust crates (`crates/`), Python/JAX modules (`src/flexmesa/`), tests (`tests/`), or documentation (`docs/`, `mkdocs.yml`), follow the standing repository rules in [`.claude/rules/docs-upkeep.md`](.claude/rules/docs-upkeep.md) and [`.claude/rules/e2e-debugging-and-parity.md`](.claude/rules/e2e-debugging-and-parity.md) (mirrored in `.agents/rules/`).

## Layout

| Path | Role |
| :--- | :--- |
| `src/flexmesa/` | Unified Python and JAX (`float64`) package (`backend="auto" \| "rust" \| "jax"`) mirroring MESA's module tiers |
| `crates/` | Safe Rust (`#![forbid(unsafe_code)]`) Cargo workspace crates + `flexmesa-py` (`PyO3` extension bridge) |
| `tests/` | Python `pytest` and `hypothesis` test suite (`slow` marks full-grid or multi-step evolutionary tests; `test_docs_snippets.py` verifies all documentation snippets, links, and API directives) |
| `benchmarks/` | Cross-language 4-stage benchmark harnesses, including compiled upstream Fortran MESA (`benchmarks/fortran_mesa/`) |
| `docs/` | 6-tab public MkDocs Material documentation website (`mkdocs.yml`: `Getting Started`, `Background & Physics`, `Tutorials`, `User Guides`, `Verification & Benchmarks`, and `Reference`) |
| `.claude/rules/` | Path-scoped repository rules (`docs-upkeep.md`), mirrored in `.agents/rules/` |
| `notes/` | Active roadmap (`PLAN.md`), decision log (`DECISIONS.md`), milestone log (`LOG.md`), and open rulings (`QUESTIONS.md`) |

## Three-Tier Hybrid Architecture Contract (`DECISIONS #6` & `#7`)

1. **Tier A (`utils`, `schema`, `data`, `testing`, `auto_diff`)**:
   - Canonical implementation lives in Safe Rust (`crates/`) and is exposed zero-copy to Python via `flexmesa._rust` (`PyO3`). Never duplicate infrastructure parsers, `.safetensors` table loaders, or scalar Python dual-number operator-overloaded loops in pure Python.
2. **Tier B (`const`, `math`, `interp_1d`, `interp_2d`, `neu`, `turb`, `chem`, `ionization`, `colors`, `eos`, `kap`, `rates`, `atm`, `star` microphysics/diffusion/rotation/riemann/rsp2, `binary`, `astero`)**:
   - **Safe Rust (`crates/`)**: Monomorphized `f64` and `DualOrder1<N>` kernels with Implicit Function Theorem root derivatives and persistent `rayon` parallel zone evaluation.
   - **Python / JAX (`src/flexmesa/`)**: Branchless `@jax.jit` flat-array kernels (`jnp.where`, `lax.fori_loop`, `jax.custom_jvp`).
   - **Unified Dispatch (`backend="auto" | "rust" | "jax"`)**: `"auto"` routes `numpy.ndarray` and float inputs zero-copy to `flexmesa._rust` and `jax.Array` / traced inputs to `@jax.jit`.
3. **Tier C (`mtx`, `num`, `net`, `star_solver`, `binary` implicit RLOF, `astero` optimization drivers)**:
   - Safe Rust provides sequential block-Thomas, Hirshman `bcyclic` odd-even cyclic reduction, and stiff Rosenbrock/Newton solvers.
   - JAX provides both native $\lceil \log_2 N_{\mathrm{zones}} \rceil$ parallel cyclic reduction (`solver_mode="pcr"`) for on-GPU `vmap` and `flexmesa._rust` FFI bindings with analytical `jax.custom_vjp` rules (`solver_mode="rust_ffi"`).

## Commands

- **Fetch & Verify Release-Hosted `data-v1` Physics & Upstream MESA Test Bundles**:
  ```bash
  flexmesa data fetch --include-test-inputs
  flexmesa data verify --include-test-inputs
  flexmesa data status
  ```
- **Clean-Room Local Pre-PR Verification Gate (Mirrors All 4 GitHub Actions CI Jobs)**:
  ```bash
  bash scripts/verify_ci.sh
  ```
- **Python Unit & Doctest Suite**:
  ```bash
  pytest
  ```
- **Python Linting, Formatting & Type Checking**:
  ```bash
  ruff check .
  ruff format --check --line-length 80 .
  mypy --strict src tests
  ```
- **Rust Workspace Tests & Clippy Gate**:
  ```bash
  cargo test --workspace --all-targets
  cargo test --workspace --doc
  cargo clippy --workspace --all-targets -- -D warnings
  ```
- **Public Documentation & Executable Snippet Gate**:
  ```bash
  pytest tests/test_docs_snippets.py -v
  mkdocs build --strict
  ```

## Governance, Per-PR Gate & Post-Wave `X.5` Gate (`DECISIONS #5`, `#8`, `#9` & `#10`)

- **Lead**: Josh Speagle (`joshspeagle`). Priorities are numerical correctness and parity first, API clarity and documentation second, and throughput third. Never trade correctness for speed.
- **Decisions (`notes/DECISIONS.md`)**: Record every choice affecting behavior, public API shape, array index conventions, numerical tolerances, defaults, or workflow in `notes/DECISIONS.md` with rationale and rejected alternatives. Never choose silently.
- **Git, Issues & Per-PR Gate**:
  1. Open a GitHub issue, branch `<issue>-<slug>` from `main`, and open a draft pull request (`Closes #<issue>`).
  2. Keep all repository files, issues, pull requests, and comments strictly public-facing with zero machine-specific paths, local file URIs, hostnames, or internal environment references.
  3. Preserve native `git config` author/committer settings, include a `Co-Authored-By:` trailer when assisted by an AI agent, and end agent-authored GitHub issue bodies, PR descriptions, and comments with `---` and `_Generated by [Antigravity](https://antigravity.google)_`.
  4. Write a descriptive multi-line commit body on every `git commit` stating (a) what changed module by module, (b) the physical or numerical rationale and upstream MESA files (`<module>/private/<file>.f90`), and (c) the verification tests and linters executed. Never create bare subject-only commits.
  5. Never loosen an existing test tolerance to silence a failure (`DECISIONS #9`). When a test uncovers an upstream Fortran MESA convention discrepancy or bug, preserve the strict `< 1e-10` / `< 1e-6` oracle on all unaffected channels, verify `< 1e-13` cross-backend parity on the upstream formula, and document the exact upstream file, line number, and mathematical mechanism in `notes/DECISIONS.md`, `docs/benchmarks/tolerance-and-discrepancy-ledger.md`, and the test docstring.
  6. Maintain a non-blocking background CI monitor across both `main` (`gh run list --branch main`) and active pull-request branches (`gh pr checks`) after every push and merge so local work continues while GitHub Actions runs online; whenever any CI job fails, inspect the failing log (`gh run view <id> --log-failed`) and fix the root cause early rather than ignoring broken checks.
  7. Before merging any PR, pass (a) Python/Rust tests and static analysis locally (`bash scripts/verify_ci.sh`), (b) a code simplification pass, (c) a multi-lens code review pass, (d) documentation upkeep per `.claude/rules/docs-upkeep.md` (`pytest tests/test_docs_snippets.py -v` and `mkdocs build --strict`), and (e) all live GitHub Actions CI checks (`gh pr checks`) on the PR branch, and verify post-merge `main` CI (`gh run list --branch main`) passes cleanly.
- **Mandatory Post-Wave `X.5` Consolidation, Optimization, Upstream Fortran MESA & Documentation Gate (`Wave 1.5`, `Wave 2.5`, `Wave 3a.5`, `Wave 3b.5`, `Wave 4.5`, `Wave 5.5`)**:
  - At the conclusion of every wave, before starting the next wave, execute an explicit **Phase `X.5`** milestone that:
    1. Compiles and links the actual upstream Fortran MESA modules in `benchmarks/fortran_mesa/`, verifying 16-digit golden `test_output` parity, multi-zone stage checksum parity, and 4-stage timing benchmarks (`Fortran MESA` vs. `Safe Rust` vs. `Python PyO3` vs. `Python JAX`).
    2. Performs a holistic cross-module code and performance review across the entire wave, implementing algorithmic and FFI optimizations to eliminate regressions and maximize speedups.
    3. Re-runs the full `FlexMESA` test suite and the compare-to-original-Fortran-MESA benchmark suite after every batch of `X.5` changes, advancing to the next wave **only** when all Fortran MESA parity and performance gates pass.
    4. Updates the public-facing MkDocs Material documentation website (`docs/`) with module explainers, governing equations, diagnostic plots, updated 4-way benchmark scorecards, and synced API references.

## End-to-End (`test_suite`) Porting, Checkpointing & Debugging Protocol (`DECISIONS #17`)

Full operational checklist in [`.claude/rules/e2e-debugging-and-parity.md`](.claude/rules/e2e-debugging-and-parity.md):

1. **Checkpoint-First Isolation for Long Campaigns (`> 60 s`)**: Never re-run a multi-minute evolutionary test from step 1 more than once to debug a mid-run failure or timestep stall. After the first failure, serialize a `.photo` / `.mod` (or `BinarySystem` snapshot) checkpoint `2` to `5` steps before the failing transition and reproduce the exact step window in a `< 2 s` single-step harness with per-zone Newton residual, `why_Tlim` retry-reason, mixing, and timestep telemetry.
2. **Upstream Step-Lifecycle Audit Before Local Solver Edits**: Never add ad-hoc residual-scale floors, damping clamps, or `dt`-dependent heuristics inside `hydro_eqns`, `solver`, or `bcyclic` to mask a severe timestep collapse (such as falling below the `f64` convective-core quantization ceiling $\Delta t_{\mathrm{crit}} \approx (\sigma/\Delta m)\,\Delta X_{\mathrm{ULP}}/\mathrm{tol}$). Audit the full upstream Fortran MESA step call graph (`evolve.f90` $\to$ `do_adjust_mass` / `do_conv_premix` / `premix_omega` $\to$ `struct_burn_mix.f90` `op_split_burn` & `get_convection_sigmas` $\to$ `save_start_values` $\to$ `star_solver.f90` / `solver_support.f90` $\to$ post-step $\Omega$ transport & outer binary retries) to fix the upstream physical or bookkeeping cause of the first retry.
3. **Three-Fix Stop & Diagnostic Issue Checkpoint**: If three targeted fixes on the `< 2 s` checkpoint harness do not resolve a failure or expose cross-module coupling, stop iterating, push the last working baseline to the active branch / draft PR, and record a structured root-cause and dependency breakdown in a GitHub issue before attempting further changes.
4. **Multi-Step Henyey Relaxation & `Extras` Trait Separation**: Implement `relax_initial_*` (`relax_initial_Z`, `relax_initial_Y`, `relax_initial_mass`, `relax_M_center`, `relax_tau_factor`) via genuine multi-step Henyey equilibration (`doing_relax = true`, frozen nuclear composition, `Cox` MLT) rather than single-pass algebraic array overwrites, and isolate `src/run_star_extras.f90` and `src/run_binary_extras.f90` case logic behind `StarExtras` and `BinaryExtras` traits rather than case-name branches inside library physics or step drivers.

## Scientific Code, Verification & Prose Standards

- **Modular File Hygiene (`<= 400` LOC)**: Target `<= 400` lines of code per file (hard ceiling `600` LOC) and `>= 20%` documentation ratio.
- **Precision & Determinism**: Enforce `float64` (`jnp.float64`, `np.float64`, `f64`) across all thermodynamic, spline, linear-algebra, and PDE state calculations, and double-double (`f64x2`) or `f128` accumulation where nuclear statistical equilibrium (NSE) cancellation occurs. Use explicit `np.random.default_rng(seed)` or `jax.random.key(seed)` for any stochastic test generation.
- **Documentation & Prose Style**: Pair physical and statistical intuition with exact technical terminology across all docstrings, design documents, and website pages. Document array shapes and physical CGS/solar units in every Google-style docstring (`Args:`, `Returns:`, `Raises:`). Avoid em-dashes, mannered metaphors, and mid-sentence setup-and-reveal colons.
