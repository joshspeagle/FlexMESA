# CLAUDE.md

`FlexMESA` ports [MESA (Modules for Experiments in Stellar Astrophysics)](https://github.com/MESAHub/mesa) into a canonical Safe Rust engine (`crates/`, exposed to Python via `PyO3` as `flexmesa._rust`) with a Python front end and a derived `float64` JAX layer (`src/flexmesa/`). The target for `v0.1.0` is a complete, verified port of **MESA release `26.4.1`** (git tag `26.4.1`, Zenodo record `19722306`), including the bundled codes required by its test suites.

## Start here

1. **[Issue #140](https://github.com/joshspeagle/FlexMESA/issues/140)**: the **only** active work tracker (decisions, phases, parallel sets, collision groups, and open work packages). Take work solely from unclaimed sub-issues of `#140`; the earlier epics and tier issues (`#89`, `#91`–`#96`, `#109`, `#110`, `#114`–`#121`, `#137`, `#139`) were closed on 2026-11-16 as superseded by `#140`.
2. [`notes/SCOPE.md`](notes/SCOPE.md): written scope and the `v0.1.0` definition (`90 star`, `8 binary`, and `8 astero` cases in MESA `26.4.1` running unmodified inlists to upstream's termination and meeting strict-mode history-harness tolerances; exact module goldens; cross-machine bit-identical strict mode; zero undeclared gaps; zero departures from upstream).
3. [`notes/audit/README.md`](notes/audit/README.md), [`notes/audit/TRACKER.md`](notes/audit/TRACKER.md), [`notes/audit/measurements.md`](notes/audit/measurements.md), and [`notes/audit/coverage-map.md`](notes/audit/coverage-map.md): whole-repository audit findings (`T1-*`, `T2-*`, `C1-*`, `F1-*` to `F4-*`, `N1-*`, `G1-*`) and baseline measurements against real Fortran MESA.
4. [`notes/DECISIONS.md`](notes/DECISIONS.md): settled architectural, numerical, and governance decisions.
5. Standing path-scoped rules in [`.claude/rules/verification-and-governance.md`](.claude/rules/verification-and-governance.md) and [`.claude/rules/docs-upkeep.md`](.claude/rules/docs-upkeep.md) (mirrored byte-for-byte in `.agents/rules/`).

*(Note: Pre-audit entries in `notes/PLAN.md`, `notes/LOG.md`, and `docs/benchmarks/` contain unsupported completion and parity claims scheduled for correction in `#142` and `#146`. Do not edit `notes/PLAN.md` or `notes/LOG.md` to record success; progress is reported only by the generated dashboard in `#174`.)*

## Layout

| Path | Role |
| :--- | :--- |
| `crates/` | Canonical Safe Rust (`#![forbid(unsafe_code)]`) Cargo workspace crates and `flexmesa-py` (`PyO3` extension bridge) |
| `src/flexmesa/` | Python front end and derived `float64` JAX layer (`star`, `binary`, and `astero` JAX engine twins are frozen under `#166`) |
| `tests/` | Python `pytest` and `hypothesis` suite (`test_docs_snippets.py` checks documentation links, API directives, snippets, and rule mirrors) |
| `benchmarks/` | Benchmark harnesses, including Fortran MESA comparators in `benchmarks/fortran_mesa/` (`#146`) |
| `docs/` | MkDocs Material documentation website (`mkdocs.yml`) |
| `.claude/rules/` | Standing repository rules (`verification-and-governance.md`, `docs-upkeep.md`), mirrored in `.agents/rules/` |
| `notes/` | Written scope (`SCOPE.md`), audit registers (`audit/`), decisions (`DECISIONS.md`), and historical logs |

## Architecture, Reference & Numerics Contract (`notes/SCOPE.md`, `#141`, `#165`, `#166`)

1. **Upstream Reference (`#141`)**:
   - MESA release `26.4.1` is the single reference. Until `#141` lands, bundled test inputs come from MESA `main` at commit `0b2e6bb`; any comparison against `0b2e6bb` histories in `notes/audit/upstream_histories/` must be labelled `"interim reference"` and is never used for package acceptance.
2. **Parity First, Zero Option Flags & Removal of Existing Departures (`#145`, `#169`)**:
   - Until `v0.1.0`, `FlexMESA` reproduces MESA `26.4.1` including its bugs, with no algorithm option flags for alternatives. The only run-level switch is the numerics mode (`strict` or `fast`, `#165`), which changes arithmetic and nothing else.
   - When an upstream bug or clear improvement is identified, open a `[Deferred]` issue labelled `deferred` under `#169` citing the upstream `file:line`, mechanism, and proposed change; do not implement it before `v0.1.0`.
   - **Existing departures are removed, not kept**: code already in the repository that improves on or "fixes" upstream (`#175`–`#178`) must be reverted back to upstream behaviour, with a `#169` backlog issue recording what was removed.
   - If an upstream behaviour cannot be reproduced exactly (for example, dependence on uninitialized memory or compiler-specific evaluation), stop and ask the project lead rather than choosing a substitute behaviour.
3. **One Rust Source, Two Numerics Modes & Three Fidelity Levels (`#165`)**:
   - **Strict mode (default)**: correctly rounded elementary functions via `flexmesa-math`, no fused multiply-add (`mul_add`), and upstream's exact order of operations (no algebraic rearrangement of formulas on the strict path). Strict mode is the **only** mode judged against MESA.
   - **Fast mode**: platform math, FMA, and vectorization permitted, built from the same Rust physics source and validated against strict mode within stated tolerances.
   - **Three numerical fidelity levels**:
     1. *Level 1*: Strict mode is bit-identical across machines and thread counts (hard requirement, tested in CI on two runner types).
     2. *Level 2*: Strict-mode kernels are bit-identical to MESA wherever operations can be made the same in the same order (hard requirement per kernel, tested against exact hexadecimal MESA outputs).
     3. *Level 3*: Whole trajectories match MESA within the history-harness tolerances (`#143`), tracking the first step at which histories differ toward zero.
   - Physics crates must never call platform transcendental methods (`f64::exp`, `ln`, `log10`, `powf`, `powi`, ...) or `mul_add` directly once `#165` lands.
4. **Canonical Rust, Python Front End, Derived JAX (`#166`, superseding `DECISIONS #6` & `#7`)**:
   - Rust (`crates/`) is canonical. JAX (`src/flexmesa/`) is a derived layer for automatic differentiation and batching, validated against Rust strict mode and **never** treated as an upstream parity oracle (`DECISIONS #9` updated).
   - The JAX `star`, `binary`, and `astero` engine code is **frozen** until the Rust engine passes the upstream-history harness (`#143`, `#166`); do not edit frozen JAX engine files.
5. **Scope, Test-Driven Port Order, Bundled Codes & Plotting (`notes/SCOPE.md`, `#147`, `#170`–`#173`)**:
   - Port order is derived from what the next test cases need using the per-case unsupported-feature report (`#147`), never from what is easiest to port.
   - Everything MESA ships is in scope: `GYRE` (`#170`) and `ADIPLS` (`#171`) are ported when test cases need them and are required for `v0.1.0` (the existing in-house pulsation solver is scaffolding and no case depending on it counts as passing). `STELLA` (`#172`) and the 32 `star/dev_cases_*` development cases are confirmed for after `v0.1.0`. Python plotting replacing `pgstar` and `pgbinary` is built incrementally alongside the port (`#173`).

## Commands

- **Fetch & Verify Data and Test-Input Bundles**:
  ```bash
  flexmesa data fetch --include-test-inputs
  flexmesa data verify --include-test-inputs
  flexmesa data status
  ```
- **Local Verification Mirror of GitHub Actions CI**:
  ```bash
  bash scripts/verify_ci.sh
  ```
- **Python Unit Suite, Linting, Formatting & Strict Type Checking**:
  ```bash
  pytest
  ruff check src tests
  ruff format --check --line-length 80 src tests
  mypy --strict src tests
  ```
- **Rust Workspace Tests, Docs & Clippy Gate**:
  ```bash
  cargo test --workspace --exclude flexmesa-py --release
  cargo clippy --workspace --all-targets -- -D warnings
  RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --exclude flexmesa-py
  ```
- **Documentation & Executable Snippet Gate**:
  ```bash
  pytest tests/test_docs_snippets.py -v
  mkdocs build --strict
  ```

## How to Pick Up an Audit Package (`#140`)

1. **Read Before Acting**: Read `#140`, `notes/SCOPE.md`, `notes/audit/{README,TRACKER,measurements,coverage-map}.md`, the target sub-issue, and every audit finding (`T1-*`, `C1-*`, `F1-*`, ...) it cites. Because most register entries came from source reading, verify each cited finding against both `FlexMESA` and upstream MESA `26.4.1` before changing code; if an audit finding does not hold, document why in a comment on the issue.
2. **One Issue, One Branch, One Pull Request**: Claim an unclaimed sub-issue of `#140` in an issue comment, create branch `<issue>-<slug>` from `main`, open an early draft pull request, and post progress comments on the issue so interrupted work survives. Never run packages in the same collision group concurrently (`#147`, `#150`, `#152` on the `star` step driver and solver; `#148`, `#154` on model initialization).
3. **Translate Upstream Faithfully**: Port upstream's algorithm in upstream's order of operations; do not substitute a simpler stand-in routine under the same name, and do not rearrange arithmetic on the strict path. Cite upstream `<module>/private/<file>.f90:<line>` for every behavioural change.
4. **Fail Loudly on Unsupported Inputs (`#147`)**: Unknown controls, unhandled option strings, missing data tables, and unreadable model files must return explicit errors, never silent defaults or synthesized fallbacks.
5. **Bit-Identical Refactors**: Pure structural refactors are allowed only when strict-mode output is bit-identical before and after, verified with the harness.
6. **Handoff Status**: No agent declares its own package complete. Mark finished packages `"ready for verification"`; completion is decided by a scheduled real-MESA run or an independent review session.

## Verification, Expected Values, Pass Criteria & Reporting Rules (`#145`, `DECISIONS #9` & `#17`)

- **Lead**: Josh Speagle (`joshspeagle`). Correctness and exact upstream fidelity come first, API clarity and documentation second, and throughput third. Performance is never a merge gate ahead of verified correctness.
- **Definition of Passing (`"green"`) for an Upstream Case (`#143`)**:
  - An upstream `star`, `binary`, or `astero` test case counts as passing **only** when:
    1. Upstream's inlists run unmodified, with only the logic present in upstream's own `run_star_extras.f90` or `run_binary_extras.f90`.
    2. Evolution reaches upstream's own physical or extras termination condition uncapped.
    3. Named history quantities match a stored real-MESA `26.4.1` history within the tolerances declared in the case manifest.
  - Any run with a step cap or without a real-MESA `26.4.1` history comparison must be reported strictly as `"smoke: N steps"`. A case whose `26.4.1` reference history is not yet available is reported as `"no reference yet"`, never as passing or failing.
  - Every trajectory-changing pull request must include the script-generated before-and-after case table and trajectory comparison figure (`#143`, `#173`).
- **Forbidden Pass-Manufacturing Practices**:
  - Never make a test or case pass by loosening a tolerance (`DECISIONS #9`), changing an expected value, capping step counts, adding `#[ignore]` or `xfail`, clamping physical state, adding convergence fallbacks, or branching on a test-case name or mass range.
  - Library crates (`crates/flexmesa-*` outside dedicated test-suite harnesses/extras) must contain **zero** test-case names (`"double_bh"`, `"hb_2M"`, ...) or case-specific parameter overrides (`#148`).
- **No Undeclared Failures (Replacing `"All Green Before Merge"`, `#144`)**:
  - Removing fabricated behaviour or placeholder physics is expected to make unfinished cases fail; a failing case due to unported physics is an accurate measurement.
  - Record known failures in the declared known-gaps manifest with an open issue link (`#144`). CI gates on **zero undeclared failures** (and flags declared gaps that begin passing). Every `#[ignore]` or `xfail(strict=True)` in the repository must carry an open `#<issue>` link.
- **Expected Values & Regression Snapshots (`#146`)**:
  - Any commit that modifies a pinned numeric literal in a test, fixture, or manifest must state in the commit body: (a) the old value, (b) the new value, and (c) the independent source of the new value.
  - Self-generated `FlexMESA` outputs used for internal regression or cross-backend consistency checks must be named `"regression snapshots"`, stored separately from upstream MESA references, and never labelled `"golden"`, `"reference"`, `"parity"`, or `"Fortran"`.
- **Generated Scorecards & Evidence-Gated Status (`#174`)**:
  - All status tables, case scorecards, and dashboard metrics in `README.md` and `docs/` are generated by script from test artifacts and the port manifest, never typed by hand.
  - Do not edit `notes/PLAN.md` or `notes/LOG.md` to claim completion. Any milestone marked complete requires evidence generated outside the authoring session (a CI artifact or a verified real-MESA comparison).
  - In the port manifest (`#174`), a file's status may be raised only in a pull request that supplies the required evidence; lowering a status is always permitted.
- **Restricted Vocabulary & Plain Non-Verification Disclosure**:
  - Never write `"verified"`, `"parity"`, `"golden"`, `"green"`, or `"complete"` about anything not measured against real MESA `26.4.1` in that pull request. Every pull request must state plainly what was **not** verified.
  - Keep unconfirmed numerical or physical hypotheses out of rule files until demonstrated against a real-MESA run.
- **Decisions, Commits & GitHub Hygiene**:
  1. Record every non-obvious choice affecting behaviour, public API shape, array indexing, tolerances, defaults, or workflow in `notes/DECISIONS.md` with rationale and rejected alternatives.
  2. Write descriptive multi-line commit bodies stating (a) what changed module by module, (b) the upstream MESA `file:line` reference (plus old/new values and source for any literal change), and (c) the verification commands executed.
  3. Keep all committed files, commits, issues, pull requests, and comments strictly public-facing with zero machine-specific paths, local file URIs, local hostnames, or private environment names.
  4. Preserve native `git config` author/committer settings, include a `Co-Authored-By:` trailer for the active agent, and end agent-authored GitHub issue bodies, PR descriptions, and comments with `---` and an italicized attribution footer naming the agent that performed the work (`_Generated by [Claude Code](https://claude.ai/code)_`, `_Generated by [Antigravity](https://antigravity.google)_`, or `_Generated by [OpenAI Codex](https://openai.com/codex)_`).
  5. Tracking-only pull requests (`notes/`, rule files, issue trackers) may merge immediately to `main` without waiting for code CI once local hygiene checks pass; code pull requests must pass local checks, independent review, and GitHub Actions CI (`gh api .../runs`) on the PR branch and `main`.

## Code Structure, Precision & Documentation Standards

- **Upstream Fidelity Outranks File-Length Targets (`N1-30`)**: Target `<= 400` LOC (`600` LOC ceiling) as an **advisory** guideline for new infrastructure modules, with an explicit exception whenever mirroring an upstream Fortran file or subroutine one-to-one. Never split a cohesive algorithm mid-concept or pack unformatted lines (`#[rustfmt::skip]`) to stay under a line count.
- **Precision & Determinism**: Use `float64` (`f64`, `np.float64`, `jnp.float64`) throughout, and explicit `np.random.default_rng(seed)` or `jax.random.key(seed)` for stochastic tests.
- **Documentation & Prose Style**: Pair physical and statistical intuition with exact technical terminology across docstrings and documentation. Document array shapes and CGS/solar units in Google-style docstrings (`Args:`, `Returns:`, `Raises:`). Avoid prose em-dashes, mannered metaphors, and mid-sentence setup-and-reveal colons.
