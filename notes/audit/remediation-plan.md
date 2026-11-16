# Remediation plan

> **Reference note (`#141`):** Line numbers, commit diffs, and pre-rebaseline measurements in this historical audit register refer to MESA `main` at commit `0b2e6bb` (2026-11-04) and `FlexMESA` commit `91eda0b`. Active repository inputs, module goldens, `case_table.md`, and `defaults_diff_26_4_1.md` target **MESA release `26.4.1`** (tag `26.4.1`, commit `c8c2026`, Zenodo `19722306`).

Ordered list of work packages that follow from the audit. Each package is
sized to be one issue and one pull request. Finding identifiers refer to the
detailed registers in this directory (`T1` Python tests, `T2` Rust tests, `C1`
coverage and non-upstream behaviour, `F1` star core, `F2` microphysics, `F3`
binary and astero, `F4` mixing, mass change and rotation, `G1` documentation
and governance, `N1` numerics, Python and structure, `M` measurements in
[`measurements.md`](measurements.md)).

The ordering principle is that nothing else can be trusted until the project
can tell whether a change moved it towards or away from upstream. Packages 1
to 4 build that ability and correct the public record. Packages 5 to 9 remove
the code that hides failures. Only then does porting effort (packages 10
onwards) produce verifiable progress.

Sizes: S is under a day, M is a few days, L is a week or more.

The packages are tracked as GitHub issues under #140. [`TRACKER.md`](TRACKER.md)
maps each package to its issue and lists dependencies and the decisions that
need the project lead. The issue list refines this plan in three ways: it adds
a package that moves the upstream reference to MESA release 26.4.1 (#141), it
splits benchmark provenance out of package 1 (#146), and it splits the Phase C
items into 15 packages (#152 to #166). Where the two differ, the issues are
current.

## Phase A. Make the truth visible

### 1. Correct the upstream label and the public claims (S)

- Replace every `r24.08.1` label with `MESA main at commit 0b2e6bb
  (2026-11-04)` and record the commit in one machine-readable place
  (`src/flexmesa/data/bundles.toml`). (G1-6)
- Reword the scorecards in `notes/PLAN.md`, `notes/LOG.md`,
  `docs/benchmarks/tolerance-and-discrepancy-ledger.md` and
  `docs/benchmarks/overview.md` so that "107 / 107", "Verified", "0 ULP" and
  "100% parity" are replaced by what was measured: number of steps run,
  whether an upstream quantity was compared. (T1-1, T2-1, T2-7, T2-8, G1 claims
  register)
- Remove or relabel the "MESA paper figure" atlas, which plots typed arrays,
  closed forms and seeded random numbers. (G1-10)
- Fix the Wave 4 and Wave 5 benchmark reports: relabel hand-written
  comparators as "reference Fortran", remove the "9 of 9 digits" headline that
  the table below it contradicts. (G1-1 to G1-5)

Why first: it costs little, and every later decision by a reader of the
repository depends on it.

### 2. Upstream-history E2E harness (M)

- Land the per-step tracers (`star_case_trace`, `binary_case_trace`) and the
  comparison script from `notes/audit/tools/`.
- Store thinned upstream `history.data` files for each case under version
  control with their provenance (MESA commit, SDK version, run flags). This
  audit contributes the first set in `notes/audit/upstream_histories/`.
- Define the pass criterion per case in a manifest: upstream termination code
  reached, final values within stated tolerances, and maximum deviation along
  the track within stated tolerances. Every intentional departure gets a line
  in the manifest with its rationale.
- Report results as a generated table (steps, retries, termination, largest
  deviation per quantity). Never hand-type a scorecard.

Why second: it turns each of the 107 cases into a measurement. The cases in
[`measurements.md`](measurements.md) show the spread this reveals.

### 3. CI that runs real campaigns (M)

- Nightly or weekly job: build MESA at the pinned commit in a container (the
  recipe is in [`README.md`](README.md), "Ground truth"), regenerate or reuse
  stored histories, run the uncapped FlexMESA cases, publish the generated
  table as an artifact.
- Per-PR job: a small fixed subset of short uncapped cases.
- Replace "all green before merge" by "no undeclared red": allow declared
  known gaps with an issue link, fail on undeclared failures and on a declared
  gap that starts passing. (G1-7, G1 R3)
- Build the Fortran benchmark harness in CI or stop quoting it. The Makefile
  needs the `build/<module>/{lib,include,modules}` layout. (G1-5)

### 4. Governance changes (S)

- Govern expected values and pass criteria, not only tolerances: any change to
  a pinned number states old value, new value and the independent source. A
  self-generated number is a "regression snapshot" and is never called golden,
  parity or Fortran. (G1-7, G1-8, T1-5, T2-12)
- Define "green" for an upstream case (unmodified inlists, upstream
  termination, quantities compared to a stored upstream history). (G1 R1)
- Forbid test-case names in library crates. (C1-5, C1-8 to C1-12)
- Make the file-size ceiling advisory where a file mirrors an upstream routine.
  Keep hypotheses out of rule files. (G1-11, N1-30)

## Phase B. Remove what hides failures

### 5. Fail loudly on unsupported input (S to M)

- Unknown or unsupported controls and option strings become errors (or at
  minimum a printed warning list per run). (C1-13, F2-1, F2-10, N1-28)
- Missing data becomes an error, not a silent fallback to another EOS or a
  zero rate. (F2-5, F2-16, C1-6)
- Stop discarding `Result`s from physics calls. (N1-21, F1-7)
- Remove silent clamps on user controls (`retry_limit`,
  `timestep_factor_for_retries`, `min_timestep_limit`). (F1-3)

Expect cases to start failing. Each new failure is a finding that was already
there.

### 6. Move test-case code out of the library and delete the fabrications (M)

- Delete `tier1_extras.rs` to `tier5_extras.rs` behaviour that overrides
  inlist controls, forces termination strings, overwrites state so stop
  conditions fire, or sets `min_logt_for_net`. Port each upstream
  `run_star_extras.f90` as test-side code where a case has one.
  (C1-8 to C1-12, T2-3)
- Remove case-name and mass-range branches from `star_job_init.rs`, the
  rescaled-ZAMS initial models, and the rescaled gravitational constant and
  opacity. (C1-1, C1-2, C1-5)
- Remove the `atm_option = 'table'` proxy boundary condition. (C1-15, F2-22)
- Remove the invented `min_logt_for_net` control. (F2-13)

After this package most tier 2 to 5 cases will not run. That is the correct
status for them today.

### 7. Align defaults with upstream `*.defaults` (S)

About 30 scalar defaults differ, including `MLT_option`, `varcontrol_target`,
`screening_mode`, Type 2 opacity flags, `min_D_mix`, `fe56ec_fake_factor` and
three binary controls. Add a test that parses upstream `*.defaults` from the
test-input bundle and compares every control FlexMESA knows. (C1-14, F1-32,
F2-8, F2-14, F2-15, F2-17, F3-5)

### 8. Error type, telemetry and checkpoints (M)

- Structured error carrying model number, zone, equation or variable and
  value; keep the per-attempt failure reason. (F1-9, N1-19)
- Counters that survive retries and binary rollbacks. (F3-7)
- Land the opt-in step telemetry from the tooling branch, rebased.
- Photo files that serialise controls, and an on-disk checkpoint for
  `BinarySystem`. (N1-20)
- Replace whole-model clones in the retry and binary paths by old-generation
  buffers of primary variables. (F3-7, N1-17)

### 9. Energy accounting (M)

Implement upstream's `total_energy_end - (total_energy_old + sources and
sinks)` audit, so that the energy error measures conservation, and enable the
upstream limits that depend on it. (F1-1, F1-28)

## Phase C. Port what is missing, judged by the harness

Order within this phase should follow what the harness shows to matter most
for the cases the project cares about. A reasonable default order:

10. Newton iteration control (`star_solver.f90` `do_solver_work`,
    `adjust_correction`, tolerance schedule, `max_tries`) and upstream retry
    semantics. (F1-2 to F1-5, F1-12 to F1-17, C1-17) M.
11. Mesh planner (`mesh_plan.f90`, `mesh_functions.f90`). (F1-33, F1-34,
    C1-18) M.
12. Relaxation machinery (`relax.f90` `do_internal_evolve` and the `relax_*`
    family), `create_initial_model`, saved-model loading for all masses.
    (C1-3, C1-4, F3-9) L.
13. Mixing: predictive mixing, convective premixing, overshoot details.
    The measured factor of two in the `hb_2M` core helium burning lifetime
    points here. (C1-16, F4, M) L.
14. Element diffusion (`diffusion*.f90`, Paquette coefficients) and phase
    separation with upstream's phase diagrams. (C1-16) L.
15. TDC state handling and the remaining turbulence wiring. (F1-4, F1-20,
    F2-18) M.
16. Rotation as part of the solve, tidal torques as `extra_jdot`. (F4, F3-6) L.
17. Binary step structure (part1, `adjust_mdots`, part2), upstream's orbital
    integrator, `check_implicit_rlo` logic. (F3-1 to F3-4, F3-8) M.
18. Microphysics gaps: `&eos` and `&kap` controls, OP monochromatic
    opacities, rate library coverage, weak rates. (F2, C1 map section 7) L.
19. Pulsation: link GYRE or validate the in-house solver against GYRE at the
    precision the astero chi-squared needs; replace the simplified NEWUOA and
    BOBYQA. (F3-15 to F3-17, N1-1) L.
20. Output: `history.data` and `profile` writers with upstream column
    definitions, which also simplifies the harness. (C1 map) M.

Decide explicitly, and record in `notes/DECISIONS.md`, whether bit-level parity
with Fortran MESA is a goal. As built it is not attainable, because the math
kernels use platform `libm` and fused multiply-add where upstream uses
correctly rounded CRMATH. (N1-6, N1-7)
