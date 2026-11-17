# CLAUDE.md

`FlexMESA` ports [MESA (Modules for Experiments in Stellar Astrophysics)](https://github.com/MESAHub/mesa) into a canonical Safe Rust engine (`crates/`, with `#![forbid(unsafe_code)]` across all crates except the sole `crlibm` FFI wrapper crate `flexmesa-crlibm-sys`, exposed to Python via `PyO3` as `flexmesa._rust`) with a Python front end and a derived `float64` JAX layer (`src/flexmesa/`). The target for `v0.1.0` is a complete, verified port of **MESA release `26.4.1`** (git tag `26.4.1`, Zenodo record `19722306`), including the bundled codes required by its test suites.

## Start here

1. **[Issue #140](https://github.com/joshspeagle/FlexMESA/issues/140)**: the **only** active work tracker (decisions, phases, parallel sets, collision groups, and open work packages). Take work solely from unclaimed sub-issues of `#140`; the earlier epics and tier issues (`#89`, `#91`–`#96`, `#109`, `#110`, `#114`–`#121`, `#137`, `#139`) were closed on 2026-11-16 as superseded by `#140`.
2. [`notes/SCOPE.md`](notes/SCOPE.md): written scope and the `v0.1.0` definition (`90 star`, `8 binary`, and `8 astero` cases in MESA `26.4.1` running unmodified inlists to upstream's termination and meeting strict-mode history-harness tolerances; exact module goldens; cross-machine bit-identical strict mode; zero undeclared gaps; zero departures from upstream).
3. [`notes/audit/README.md`](notes/audit/README.md), [`notes/audit/TRACKER.md`](notes/audit/TRACKER.md), [`notes/audit/measurements.md`](notes/audit/measurements.md), and [`notes/audit/coverage-map.md`](notes/audit/coverage-map.md): whole-repository audit findings (`T1-*`, `T2-*`, `C1-*`, `F1-*` to `F4-*`, `N1-*`, `G1-*`) and baseline measurements against real Fortran MESA.
4. [`notes/DECISIONS.md`](notes/DECISIONS.md): the decisions in force, with an index of every numbered decision; superseded entries are in `notes/archive/DECISIONS-archive.md`.
5. Standing path-scoped rules in [`.claude/rules/verification-and-governance.md`](.claude/rules/verification-and-governance.md) and [`.claude/rules/docs-upkeep.md`](.claude/rules/docs-upkeep.md) (mirrored byte-for-byte in `.agents/rules/`).

*(Note: The pre-audit roadmap, milestone log and question list are archived unmaintained in `notes/archive/` (`PLAN.md`, `LOG.md`, `QUESTIONS.md`), together with superseded decisions (`DECISIONS-archive.md`). They and `docs/benchmarks/` contain unsupported completion and parity claims (`#142`, `#146`). Do not edit archived files and do not create new hand-written status files; progress is reported only by the generated case table (`notes/audit/case_table.md`) and dashboard (`#174`).)*

## Layout

| Path | Role |
| :--- | :--- |
| `crates/` | Canonical Safe Rust Cargo workspace crates (`#![forbid(unsafe_code)]` across all crates except `flexmesa-crlibm-sys`, the sole `crlibm` FFI wrapper crate; `#165`) and `flexmesa-py` (`PyO3` extension bridge) |
| `src/flexmesa/` | Python front end and derived `float64` JAX layer (`star`, `binary`, and `astero` JAX engine twins are frozen under `#166`) |
| `tests/` | Python `pytest` and `hypothesis` suite (`test_docs_snippets.py` checks documentation links, API directives, snippets, and rule mirrors) |
| `benchmarks/` | Benchmark harnesses, including Fortran MESA comparators in `benchmarks/fortran_mesa/` (`#146`) |
| `docs/` | MkDocs Material documentation website (`mkdocs.yml`) |
| `.claude/rules/` | Standing repository rules (`verification-and-governance.md`, `docs-upkeep.md`), mirrored in `.agents/rules/` |
| `notes/` | Written scope (`SCOPE.md`), decisions in force (`DECISIONS.md`), audit registers and generated case table (`audit/`), and unmaintained pre-audit documents (`archive/`); see `notes/README.md` |

## Architecture, Reference & Numerics Contract (`notes/SCOPE.md`, `#141`, `#165`, `#166`)

1. **Upstream Reference (`#141`, `DECISIONS #22`)**:
   - MESA release `26.4.1` (git tag `26.4.1`, commit `c8c2026`, Zenodo record `19722306`) is the single reference, recorded in `src/flexmesa/data/bundles.toml`. Bundled test inputs (`mesa-test-inputs-v2.tar.zst`), exact IEEE-754 hexadecimal module test outputs (`mesa-module-hex-v1.tar.zst`), and stored reference histories (`mesa-histories-v1.tar.zst` and `benchmarks/history_harness/reference_histories/`) all come from MESA `26.4.1` built with MESA SDK `x86_64-linux-26.6.1`.
2. **Parity First, Zero Option Flags & Removal of Existing Departures (`#145`, `#169`)**:
   - Until `v0.1.0`, `FlexMESA` reproduces MESA `26.4.1` including its bugs, with no algorithm option flags for alternatives. The only run-level switch is the numerics mode (`strict` or `fast`, `#165`), which changes arithmetic and nothing else.
   - When an upstream bug or clear improvement is identified, open an issue under `#169` using the "Deferred" issue template, citing the upstream `file:line` at tag `26.4.1`, mechanism, effect, and proposed change; do not implement it before `v0.1.0`. Label it `deferred` plus exactly one kind label and use the matching title prefix: `upstream-bug` with `[Deferred bug]` (upstream contradicts its own documentation, comments, units, or evident intent) or `enhancement` with `[Deferred enhancement]` (upstream is self-consistent and a different choice would be better). Add `parity-pending` while `FlexMESA` code still departs from upstream at that point, naming the package that restores upstream behaviour, and `needs-confirmation` until the upstream lines have been read at the tag. Add the issue's row to the register in `#169` in the same session, and cite the deferred issue number in a comment at the reproduced line. In-scope port work scheduled after `v0.1.0` (`#172`) is labelled `deferred-scope` and stays under `#140`.
   - **Existing departures are removed, not kept**: code already in the repository that improves on or "fixes" upstream (`#175`–`#178`) must be reverted back to upstream behaviour, with a `#169` backlog issue recording what was removed.
   - If an upstream behaviour cannot be reproduced exactly (for example, dependence on uninitialized memory or compiler-specific evaluation), stop and ask the project lead rather than choosing a substitute behaviour.
3. **One Rust Source, Two Numerics Modes & Three Fidelity Levels (`#165`, `DECISIONS #23`)**:
   - **Strict mode (default)**: correctly rounded elementary functions via `flexmesa-math` backed by vendored `crlibm 1.0beta4` (`CRMATH 1.2`) in `flexmesa-crlibm-sys` (the **sole** crate containing `unsafe`, compiled with `#![deny(unsafe_op_in_unsafe_fn)]` and `// SAFETY:` comments on every `extern` call, while every other crate enforces `#![forbid(unsafe_code)]`), no fused multiply-add (`mul_add`), and upstream's exact order of operations (no algebraic rearrangement of formulas on the strict path). Strict mode is the **only** mode judged against MESA.
   - **Fast mode**: platform math, FMA, and vectorization permitted, built from the same Rust physics source (`AtomicU8` run-level mode selector) and validated against strict mode within stated tolerances.
   - **Three numerical fidelity levels**:
     1. *Level 1*: Strict mode is bit-identical across machines and thread counts (hard requirement, tested in CI on `ubuntu-latest` `x86_64` and `macos-latest` `aarch64` runners).
     2. *Level 2*: Strict-mode kernels are bit-identical to MESA wherever operations can be made the same in the same order (hard requirement per kernel, tested against exact hexadecimal MESA outputs).
     3. *Level 3*: Whole trajectories match MESA within the history-harness tolerances (`#143`), tracking the first step at which histories differ toward zero.
   - Physics crates must never call platform transcendental methods (`f64::exp`, `ln`, `log10`, `powf`, `powi`, ...) or `mul_add` directly (enforced via workspace `clippy.toml` `disallowed-methods` and `crates/flexmesa-math/tests/routing_governance.rs`).
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
  ruff check src tests scripts/docs scripts/check_strict_determinism.py scripts/mesa_reference/measure_mesa_spread.py
  ruff format --check --line-length 80 src tests scripts/docs scripts/check_strict_determinism.py scripts/mesa_reference/measure_mesa_spread.py
  mypy --strict src tests scripts/docs scripts/check_strict_determinism.py scripts/mesa_reference/measure_mesa_spread.py
  ```
- **Known-Gaps, Module-Hex & Strict-Determinism Gates (`#144`, `#165`)**:
  ```bash
  python -m flexmesa.testing.known_gaps_gate --fail-on-imperative-skips
  python -m flexmesa.testing.module_hex_gate --check-summary
  python scripts/check_strict_determinism.py
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
- **Incremental Case Tracer, Upstream-History Harness & Progress Dashboard (`#143`, `#163`, `#165`, `#174`)**:
  ```bash
  flexmesa trace star <case> --out <out_dir> [--numerics-mode strict|fast --timeout-sec 600 --max-rss-gb 6.0]
  flexmesa trace binary <case> --out <out_dir> [--numerics-mode strict|fast --timeout-sec 600 --max-rss-gb 6.0]
  flexmesa trace astero <case> --out <out_dir> [--numerics-mode strict|fast --timeout-sec 600 --max-rss-gb 6.0]
  flexmesa harness table --check
  flexmesa harness run --preset pr-ci --timeout-sec 480 --max-rss-gb 6.0 --markdown-out case_table.md --json-out case_table.json --gate [--numerics-mode strict]
  flexmesa harness run --preset all --numerics-mode strict --jobs 2 --timeout-sec 300 --max-rss-gb 5.5 --markdown-out case_table.md --json-out case_table.json --gate
  flexmesa dashboard verify-manifest
  flexmesa dashboard render
  flexmesa dashboard check
  ```
  Run at most one full FlexMESA or MESA campaign at a time with `--jobs 2 --max-rss-gb 5.5 --timeout-sec 300` (scheduled campaign budget); running FlexMESA and MESA campaigns concurrently exhausts 15 GiB hosts.

## How to Pick Up an Audit Package (`#140`)

1. **Read Before Acting**: Read `#140`, `notes/SCOPE.md`, `notes/audit/{README,TRACKER,measurements,coverage-map}.md`, the target sub-issue, and every audit finding (`T1-*`, `C1-*`, `F1-*`, ...) it cites. Because most register entries came from source reading, verify each cited finding against both `FlexMESA` and upstream MESA `26.4.1` before changing code; if an audit finding does not hold, document why in a comment on the issue.
2. **One Issue, One Branch, One Pull Request**: Claim an unclaimed sub-issue of `#140` in an issue comment, create branch `<issue>-<slug>` from `main`, open an early draft pull request, and post progress comments on the issue so interrupted work survives. Never run packages in the same collision group concurrently (`#147`, `#150`, `#152` on the `star` step driver and solver; `#148`, `#154` on model initialization).
3. **Translate Upstream Faithfully**: Port upstream's algorithm in upstream's order of operations; do not substitute a simpler stand-in routine under the same name, and do not rearrange arithmetic on the strict path. Cite upstream `<module>/private/<file>.f90:<line>` for every behavioural change.
4. **Fail Loudly on Unsupported Inputs (`#147`)**: Unknown controls, unhandled option strings, missing data tables, and unreadable model files must return explicit errors, never silent defaults or synthesized fallbacks.
5. **Bit-Identical Refactors**: Pure structural refactors are allowed only when strict-mode output is bit-identical before and after, verified with the harness.
6. **Closing a Package on Evidence (`DECISIONS #21`)**: A package is closed on evidence, not on a reviewer's sign-off. The authoring agent closes its own package issue once all of the following hold:
   - The pull request is merged.
   - A closing comment on the issue lists every acceptance criterion with a link to its evidence: a CI run, a script-generated artifact committed in the repository (case table, comparison report, reference vectors), or a `file:line`. A statement by the author without such a link is not evidence.
   - Every criterion that concerns agreement with MESA is evidenced by a script-generated comparison against real MESA `26.4.1` output, where the reference file's own header records the MESA version string, compiler and SDK. The agent that has the local MESA `26.4.1` build runs that comparison itself and commits the reference outputs with the recipe that produced them.
   - Any criterion not met is named and moved to a specific open issue in the closing comment.

   If a criterion cannot be evidenced, the package stays open with a comment saying which one and what is needed. The independent review audits closed packages afterwards and reopens any whose evidence does not hold.

## Verification, Expected Values, Pass Criteria & Reporting Rules (`#145`, `DECISIONS #9` & `#17`)

- **Lead**: Josh Speagle (`joshspeagle`). Correctness and exact upstream fidelity come first, API clarity and documentation second, and throughput third. Performance is never a merge gate ahead of verified correctness.
- **Definition of Passing (`"green"`) for an Upstream Case (`#143`)**:
  - An upstream `star`, `binary`, or `astero` test case counts as passing **only** when:
    1. Upstream's inlists run unmodified, with only the logic present in upstream's own `run_star_extras.f90` or `run_binary_extras.f90`.
    2. Evolution reaches upstream's own physical or extras termination condition uncapped.
    3. Named history quantities match a stored real-MESA `26.4.1` history within the tolerances declared in the case manifest.
  - Any run with a step cap or without a real-MESA `26.4.1` history comparison must be reported strictly as `"smoke: N steps"`. A case whose `26.4.1` reference history is not yet available is reported as `"no reference yet"`, never as passing or failing.
  - A run that continued past unsupported controls (`--allow-unsupported-controls`, `#147`) did not do what the inlist asks. The harness reports it as `"diagnostic (N controls ignored)"` and it never counts as passing (`DECISIONS #26`).
  - Every trajectory-changing pull request must include the script-generated before-and-after case table and trajectory comparison figure (`#143`, `#173`).
- **Forbidden Pass-Manufacturing Practices**:
  - Never make a test or case pass by loosening a tolerance (`DECISIONS #9`), changing an expected value, capping step counts, adding `#[ignore]` or `xfail`, clamping physical state, adding convergence fallbacks, or branching on a test-case name or mass range.
  - Library crates (`crates/flexmesa-*`) must contain **zero** test-case names (`"double_bh"`, `"hb_2M"`, ...) or case-specific parameter overrides, with no exception for files named as harnesses or extras: the existing `tier1_extras.rs` to `tier5_extras.rs` and the case-name branches in `star_job_init.rs` are removed under `#148`. Per-case hooks live in test-side code and port only what upstream's own `run_star_extras.f90` or `run_binary_extras.f90` does.
- **Declared Known-Gaps Merge Gate (Replacing `"All Green Before Merge"`, `#144`)**:
  - Removing fabricated behaviour or placeholder physics is expected to make unfinished cases fail; a failing case due to unported physics is an accurate measurement.
  - Record all unfinished history cases (`[[gaps]]`) and code-level test suppressors (`[[test_gaps]]`) in the single declared ledger `benchmarks/history_harness/known_gaps.toml` with an open `#<issue>` link (`#144`). CI gates on **zero undeclared failures**, **zero stale gaps** (declared gaps that start passing fail the gate), **zero vacuous skips** (`FLEXMESA_REQUIRE_ZERO_SKIPS=1` defaults active in every environment so missing inputs fail tests rather than passing), and Level-2 `module_hex` non-regression (`benchmarks/module_hex/known_module_gaps.toml`). Every `#[ignore]` or `xfail(strict=True)` in the repository must carry an open `#<issue>` link and appear bijectively in `benchmarks/history_harness/known_gaps.toml`.
- **Expected Values & Regression Snapshots (`#146`)**:
  - Any commit that modifies a pinned numeric literal in a test, fixture, or manifest must state in the commit body: (a) the old value, (b) the new value, and (c) the independent source of the new value.
  - Self-generated `FlexMESA` outputs used for internal regression or cross-backend consistency checks must be named `"regression snapshots"`, stored separately from upstream MESA references, and never labelled `"golden"`, `"reference"`, `"parity"`, or `"Fortran"`.
- **Generated Scorecards & Evidence-Gated Status (`#174`)**:
  - All status tables, case scorecards, and dashboard metrics in `README.md` and `docs/` are generated by script from test artifacts and the port manifest, never typed by hand.
  - Do not write completion claims into `notes/` (the pre-audit `PLAN.md` and `LOG.md` are archived in `notes/archive/` and are not edited). Any milestone marked complete requires evidence generated outside the authoring session (a CI artifact or a verified real-MESA comparison).
  - In the port manifest (`#174`), a file's status may be raised only in a pull request that supplies the required evidence; lowering a status is always permitted.
- **Restricted Vocabulary & Plain Non-Verification Disclosure**:
  - Never write `"verified"`, `"parity"`, `"golden"`, `"green"`, or `"complete"` about anything not measured against real MESA `26.4.1` in that pull request. Every pull request must state plainly what was **not** verified.
  - Keep unconfirmed numerical or physical hypotheses out of rule files until demonstrated against a real-MESA run.
- **Decisions, Commits & GitHub Hygiene**:
  1. Record every non-obvious choice affecting behaviour, public API shape, array indexing, tolerances, defaults, or workflow in `notes/DECISIONS.md` with rationale and rejected alternatives. Keep each entry to the decision, the reason and the alternatives (about 20 lines; file lists and verification commands belong in the commit body), and when an entry is superseded move it unchanged to `notes/archive/DECISIONS-archive.md` and update the index in the same pull request.
  2. Write descriptive multi-line commit bodies stating (a) what changed module by module, (b) the upstream MESA `file:line` reference (plus old/new values and source for any literal change), and (c) the verification commands executed.
  3. Keep all committed files, commits, issues, pull requests, and comments strictly public-facing with zero machine-specific paths, local file URIs, local hostnames, or private environment names.
  4. Preserve native `git config` author/committer settings, include a `Co-Authored-By:` trailer for the active agent, and end agent-authored GitHub issue bodies, PR descriptions, and comments with `---` and an italicized attribution footer naming the agent that performed the work (`_Generated by [Claude Code](https://claude.ai/code)_`, `_Generated by [Antigravity](https://antigravity.google)_`, or `_Generated by [OpenAI Codex](https://openai.com/codex)_`).
  5. Tracking-only pull requests (`notes/`, rule files, issue trackers) may merge immediately to `main` without waiting for code CI once local hygiene checks pass; code pull requests must pass local checks, independent review, and GitHub Actions CI (`gh api .../runs`) on the PR branch and `main` under the **declared-known-gaps merge gate** (`benchmarks/history_harness/known_gaps.toml`, zero undeclared failures, zero stale gaps, and zero vacuous skips).

## Code Structure, Precision & Documentation Standards

- **Upstream Fidelity Outranks File-Length Targets (`N1-30`)**: Target `<= 400` LOC (`600` LOC ceiling) as an **advisory** guideline for new infrastructure modules, with an explicit exception whenever mirroring an upstream Fortran file or subroutine one-to-one. Never split a cohesive algorithm mid-concept or pack unformatted lines (`#[rustfmt::skip]`) to stay under a line count.
- **Precision & Determinism**: Use `float64` (`f64`, `np.float64`, `jnp.float64`) throughout, and explicit `np.random.default_rng(seed)` or `jax.random.key(seed)` for stochastic tests.
- **Documentation & Prose Style**: Pair physical and statistical intuition with exact technical terminology across docstrings and documentation. Document array shapes and CGS/solar units in Google-style docstrings (`Args:`, `Returns:`, `Raises:`). Avoid prose em-dashes, mannered metaphors, and mid-sentence setup-and-reveal colons.
