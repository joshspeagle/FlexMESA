# FlexMESA whole-repository audit

> **Reference note (`#141`):** This audit folder is a frozen historical record from 2026-11-15 to 2026-11-16 against MESA `main` at commit `0b2e6bb` (2026-11-04) and `FlexMESA` commit `91eda0b`, including the nine `notes/audit/upstream_histories/*.tsv` files kept as the interim `0b2e6bb` set referenced by [`measurements.md`](measurements.md) and the reproduction recipe below. Active repository inputs, module goldens, [`case_table.md`](case_table.md), [`defaults_diff_26_4_1.md`](defaults_diff_26_4_1.md), `benchmarks/history_harness/reference_histories/*.data`, and the `data-v1` release bundles target **MESA release `26.4.1`** (tag `26.4.1`, commit `c8c2026`, Zenodo `19722306`, built via `scripts/mesa_reference/` and `scripts/data/`).

Audit dates: 2026-11-15 to 2026-11-16. This directory is the findings
register. It records how far the repository is from a complete, verified port
of MESA, with the evidence for each statement, and a plan for closing the gap.

## What was audited, and against what

- **FlexMESA tree**: PR #138 head `91eda0b` (a superset of `main` at
  `8f6195b`; `main` lacks the coupled binary and astero drivers). All
  `file:line` references in this directory are to `91eda0b` unless stated.
- **Upstream reference**: MESA `main` at commit
  `0b2e6bb07cfbbb0aafd0634fedd2256ce159f90a` (2026-11-04). This is what the
  repository's test-input bundle contains. The repository labelled it
  `r24.08.1`; that label was wrong (finding G1-6) and is corrected on this
  branch.
- **Runnable upstream**: real Fortran MESA at `0b2e6bb` was built with MESA SDK
  26.6.1. All 196 Git LFS data files were filled from public sources and
  verified against the SHA-256 in their pointers. All 23 module self-checks
  passed. Statements labelled "verified by running" were measured against
  this build.

## How to read the confidence labels

- **verified by running**: measured in this audit by executing FlexMESA, real
  MESA, or both.
- **verified by reading both sides**: FlexMESA and upstream source were both
  opened and compared at the cited lines.
- **verified by reading FlexMESA only**: the FlexMESA behaviour was read; the
  upstream side rests on routine names, defaults files or absence by search.
- **inferred**: a reasoned conclusion that was not checked directly.

Most findings come from nine narrowly scoped read-only reviews. The lead
auditor re-checked a sample of the consequential ones directly in the code or
by running; those are marked "re-checked" in the index below. Findings that
were not re-checked carry the confidence their register gives them. Items
without evidence are listed under "Leads" in each register and are not counted
as findings.

## Summary: what changes what one should believe

1. **The end-to-end "107 / 107" result is a set of 2-step smoke runs judged
   by criteria that compare nothing with upstream.** Every star and astero
   E2E test runs 2 to 8 steps per stage; `passed` ignores the termination code,
   so stopping at the step cap is a normal pass. Of the 91 star cases, none
   has a pass criterion that applies an upstream number faithfully; 51 use a
   generic sanity inequality, 24 use an upstream number whose value is
   manufactured or whose tolerance is widened, and 16 cannot fail. "0 ULP
   restart parity" compares FlexMESA with itself. (T1-1, T1-2, T2-1, T2-7,
   T2-8; re-checked by reading and by running capped cases, which report
   `passed=true` with `term=max_steps_cap`.)

2. **For tiers 2 to 5 the library fabricates the outcome.** Code selected by
   the name of the case directory overrides the inlist (burning switched off
   below 2e8 to 3e8 K through a control that does not exist upstream,
   remeshing, winds and hydrodynamics off), builds the initial model by
   rescaling a 1 Msun ZAMS model, rescales the gravitational constant and the
   opacity so that this model balances, overwrites state so that stop
   conditions fire, and emits upstream's success strings after two steps.
   (C1-1, C1-2, C1-5, C1-8 to C1-12, C1-15, T2-3; re-checked.)

3. **Where the real code path runs, the physics is closer to MESA than the
   tests could have shown, and the gaps are specific.** Run to completion
   against real MESA, simple main-sequence cases end within about 1% in age
   and about 1e-3 dex in L, Teff and R. A core helium burning case with
   predictive mixing (`hb_2M`) ends with a helium burning lifetime of 71 Myr
   against 136 Myr in MESA, from the same starting model. A brown dwarf case
   that starts from a fabricated model (`make_brown_dwarf`) ends as a
   different object, with a radius 17 times smaller than MESA's. A rotating
   case (`magnetic_braking`) fails to converge. All of these are reported as
   passed. See [`measurements.md`](measurements.md).

4. **About one sixth of upstream `star`, `binary` and `astero` is
   substantially ported.** By upstream line count, excluding plotting: 16%
   ported, 49% partial, 17% stub, 17% absent. FlexMESA reads 43% of
   `&controls`, 25% of `&star_job`, 5 of 102 `&eos` and 5 of 36 `&kap`
   controls. Unknown controls and unknown option strings are silently
   ignored or mapped to a default. About 30 defaults differ from upstream,
   including `MLT_option`, `varcontrol_target`, `screening_mode` and the
   Type 2 opacity flags. ([`coverage-map.md`](coverage-map.md); defaults
   re-checked.)

5. **The parts that decide which model is evolved and how it is controlled
   are FlexMESA designs, not ports.** The Newton iteration control, the mesh
   planner, the retry policy, every `relax_*` operation, the binary step
   sequence and orbital integrator, element diffusion, convective premixing,
   predictive mixing and phase separation use different algorithms from
   upstream. Step counts, zone counts and retry histories therefore cannot
   match upstream even where the equations do. (F1, C1-16 to C1-18, F3-1 to
   F3-4.)

6. **The module-level tests of the small standalone modules are real.** The
   `test_output` files are byte-identical to upstream and tolerances are
   tight. The Wave 1 Fortran benchmark checksums were reproduced exactly
   against real MESA in this audit. The equation residual kernels, the
   microphysics kernels that were compared term by term, the wind formulae and
   the binary closed-form kernels match upstream. This is the solid part of
   the repository.

7. **Several published artefacts are not what their labels say.** The Wave 5
   "Upstream Fortran MESA" column comes from a hand-written
   re-implementation; pinned "reference" checksums are FlexMESA's own output
   and were re-pinned when the code changed; the "MESA paper figure" atlas
   plots typed arrays, closed forms and seeded random numbers; the reported
   energy conservation error is a sum of solver residuals and is hard-wired
   to zero for one energy-equation form. (G1-1 to G1-5, G1-10, T1-5, T2-12,
   F1-1; paper-figure and energy-error code re-checked.)

8. **The repository rules reward this.** They require every case to be green
   before merge and govern tolerances, but they do not govern expected
   values, pass criteria, step caps or `#[ignore]`. The "never loosen a
   tolerance" rule was broken without comment two days after adoption.
   Scorecards were recorded as complete before the corresponding physics
   existed. (G1-7 to G1-9, T1-10, T2-11.)

## What holds up

- `const`, `neu`, `auto_diff`, `mtx`, `turb` module tests cover all or most of
  the upstream `test_output` at 1e-10 to 1e-15; `kap`, `rates`, `colors`,
  `ionization`, `math`, `interp_1d` cover substantial parts. (T1, T2)
- Wave 1 Fortran checksums: 18 of 18 reproduced against real MESA at
  `0b2e6bb` (verified by running). The Wave 1 to 4 Fortran harnesses call real
  MESA routines.
- Default-path structure equations: residual forms, scalings and face and
  cell indexing of the density, momentum, energy, temperature-gradient and
  chemistry equations match upstream, with no off-by-one found. (F1)
- EOS blend cascade and region boundaries, opacity blends and electron
  conduction, Chugunov screening, the MLT cubic for all five variants,
  thermohaline and semiconvection kernels. (F2)
- Wind formulae and coefficients, `revise_q_and_dq`, `eps_mdot`, exponential
  overshoot profile, core-mass definitions. (F4)
- Binary Roche geometry, angular momentum loss terms, Ritter and Kolb mass
  transfer rates, timestep limit formulae, and the per-case
  `run_binary_extras` checks. (F3)
- Single-star trajectories were bit-identical across 1, 3 and 4 threads and
  across repeated runs (verified by running, one basic-net case). No parallel
  floating-point reduction or hash-map iteration feeding numerics was found.
  (N1-15, N1-16)
- Unfavourable benchmark timings are reported, not hidden. README and
  CHANGELOG are restrained; the strong claims live in `notes/PLAN.md`,
  `notes/LOG.md` (both since moved to `notes/archive/`), `docs/benchmarks/` and commit messages. (G1)

## Index of findings

The detailed registers hold one entry per finding with area, severity,
evidence (`file:line` on both sides), confidence and a proposed fix with a
rough size. The table below indexes the findings that matter most. Severity
abbreviations: WP wrong physics, WN wrong numerics, UC unverified claim, M
maintainability.

| Finding | Sev. | One line | Confidence |
| :-- | :-- | :-- | :-- |
| T2-1, T1-1 | UC | Star E2E `passed` ignores the termination code; all tests cap at 2 steps | reading; re-checked by running |
| T2-7 | UC | 0 of 91 star cases judged by an upstream criterion | reading both sides |
| T2-3, C1-8 | UC, WP | Tier 3 hooks write the state that is then asserted, and emit upstream success strings after 2 steps | reading; re-checked |
| T2-8, T1-2 | UC | "0 ULP photo restart parity" is an in-process FlexMESA round trip over 1 step | reading both sides |
| C1-1, C1-5 | WP | Rescaled 1 Msun ZAMS used as initial model for masses outside 0.51 to 15.8 Msun and for tier 3, 4, 5 cases by directory name | re-checked |
| C1-2 | WP | Per-zone gravitational constant and `opacity_factor` rescaled so the rescaled model balances | re-checked |
| C1-9, F2-13 | WP | Invented `min_logt_for_net` (default 6.3; 8.3 or 8.5 in tier extras) disables burning below that temperature | re-checked |
| C1-15, F2-22 | WP | `atm_option = 'table'` without a table name back-fits the surface boundary condition to the model's own surface | reading FlexMESA |
| C1-3, C1-4 | WP | `relax.f90` (4215 lines) is not ported; relaxations are algebraic rescalings | reading both sides |
| C1-16 | WP | Convective premixing, predictive mixing, phase separation and element diffusion are placeholder algorithms | reading both sides (phase diagram), FlexMESA only (rest) |
| M-2, F4-2 | WP | `hb_2M` core helium burning lifetime 71 Myr against 136 Myr in MESA | verified by running; cause inferred |
| M-6 | WP | `make_brown_dwarf` reaches the upstream stop condition as a different object (log g 8.2 against 4.4) and is reported as passed | verified by running |
| M-8 | WN | `magnetic_braking` fails to converge when run uncapped; three other tier 3 cases did not finish in 20 minutes | verified by running |
| F4-1 | WP | Composition-gradient smoothing uses a different kernel with no threshold, on the default path | reading both sides |
| F1-12 to F1-17, C1-17 | WN | Newton solver damping, iteration budget, norms and acceptance rule differ from `star_solver.f90` | reading both sides |
| F1-33, F1-34, C1-18 | WN | Mesh planner is a different algorithm from `mesh_plan.f90` | reading both sides |
| F1-2, F1-3, F1-5 | WN | Retry discards the remeshed grid, applies an invented 0.2 factor, clamps user controls, does not restore centre state | reading both sides; 0.2 rule re-checked |
| F1-4, F1-20, F2-18 | WP | TDC "old" convective velocity overwritten several times per step; TDC errors fall back silently to MLT | reading both sides |
| F1-1, F1-28 | WN, UC | Energy conservation error is a sum of solver residuals, zero for the `eps_grav` form | re-checked |
| C1-14, F1-32, F2-8, F2-14, F2-17, F3-5 | WP | About 30 defaults differ from upstream `*.defaults` | re-checked for 5 |
| C1-13, F2-1, N1-28 | WP, M | Unknown controls ignored; unknown option strings mapped to a default | reading FlexMESA |
| F2-9 | WP | C/O reference fractions for two Type 2 opacity table families do not match the table headers | reading both sides |
| F2-15 | WP | `fe56ec_fake_factor` 1000 times upstream's star default for approx21; rate factors hard-coded | reading both sides |
| F4-14 | WP | Super-Eddington wind a factor 2 too small and uses the wrong Eddington luminosity | re-checked |
| F4-20 to F4-23 | WP | Rotation: invented moment-of-inertia fit, different `w` definition, floors on fp and ft, Spruit-Tayler inputs | reading both sides |
| F4-24, F4-26, F3-6 | WP | Total angular momentum restored by rescaling omega in every cell; tides applied as post-step relaxation | reading both sides |
| F3-1 to F3-4 | WN | Binary step sequence, orbital integrator (8-substep midpoint against upstream's one Euler step) and implicit mass-transfer search differ from upstream | reading both sides; integrator re-checked |
| F3-9 | WP | Composition relaxation for binary components is not `do_relax_Z` or `do_relax_Y`; followed by invented settle phases | reading both sides |
| F3-7, N1-17 | M | Whole-model deep copies for every retry and binary trial; counters erased by rollback | reading FlexMESA |
| F3-15 to F3-17, N1-1 | WN | Astero mode frequencies from an in-house solver never checked against GYRE; NEWUOA and BOBYQA are short stand-ins | reading both sides |
| N1-6, N1-7 | UC | Math kernels use platform `libm` and fused multiply-add; bit-level parity with Fortran MESA is not attainable as built | reading both sides |
| N1-23 | UC | No JAX stellar evolution step exists; the star "Rust versus JAX" test compares one residual formula written twice | reading FlexMESA |
| N1-10 to N1-12 | M, WN | Data bundle converters not in the repository; runtime caches written into the data directory and preferred over verified data | reading FlexMESA; cache growth observed by running |
| N1-19 to N1-21, F1-9 | M | Errors carry no model, zone, equation or value; about 25 discarded physics `Result`s; photos do not serialise controls | reading FlexMESA |
| G1-1, G1-2 | UC | Wave 5 "Upstream Fortran MESA" is a re-implementation; headline contradicted by its own table | reading |
| G1-3 to G1-5 | UC | Wave 4 checksums disagree by up to a factor of 169 while described as parity; no raw benchmark output committed | reading |
| T1-5, T2-12 | UC | Pinned "reference" checksums are FlexMESA output, re-pinned up to 7 times | reading, git history |
| T1-10, T2-11, G1-8 | UC | Tolerances loosened after the "never loosen" rule, without comment | git history; two re-checked by the governance review |
| G1-6 | UC | `r24.08.1` label wrong in 23 lines; reference is 1,141 commits later | verified; fixed on this branch |
| G1-9 | UC | Scorecards recorded complete before the physics existed | git history |
| G1-10 | UC | "MESA paper figure" atlas plots typed arrays and random numbers | re-checked |
| G1-7 | M | Rules require all green and leave expected values, criteria, caps and ignores ungoverned | reading |

Registers:

| File | Scope | Findings |
| :-- | :-- | :-- |
| [`tests-python.md`](tests-python.md) | `tests/`, CI configuration, fixtures | T1-1 to T1-14 |
| [`tests-rust.md`](tests-rust.md) | Rust tests and the E2E pass machinery | T2-1 to T2-15 |
| [`test-provenance.md`](test-provenance.md) | One row per test file | table |
| [`coverage-map.md`](coverage-map.md) | Status of every upstream `star`, `binary`, `astero` source file; controls coverage; defaults | C1-13, C1-14, map |
| [`non-upstream-behaviour.md`](non-upstream-behaviour.md) | Case-name branches, heuristics, clamps, placeholder algorithms | C1-1 to C1-22 |
| [`star-core.md`](star-core.md) | Step driver, solver, structure equations, timestep, mesh planner, defaults | F1-1 to F1-34 |
| [`star-mixing-mass-rotation.md`](star-mixing-mass-rotation.md) | Mixing coefficients, mass change, winds, rotation, remap, stop conditions | F4-1 to F4-30 |
| [`microphysics.md`](microphysics.md) | EOS, opacity, nets and rates, turbulence, atmospheres, neutrinos | F2-1 to F2-24 |
| [`binary-astero.md`](binary-astero.md) | Binary driver and kernels, astero, claims in issues #137 and #139 | F3-1 to F3-20 |
| [`numerics-python-structure.md`](numerics-python-structure.md) | Numerics crates, data pipeline, Python and JAX layer, determinism, error design | N1-1 to N1-30 |
| [`docs-governance.md`](docs-governance.md) | Claims register, benchmarks, citations, rules, CI | G1-1 to G1-11 |
| [`measurements.md`](measurements.md) | Side-by-side runs against real MESA | M-1 onwards |
| [`remediation-plan.md`](remediation-plan.md) | Ordered work packages | |
| [`TRACKER.md`](TRACKER.md) | Mapping from packages to GitHub issues (#140 and its sub-issues), order, decisions needed | |

## The claims in issues #137 and #139

Checked in [`binary-astero.md`](binary-astero.md), Part 2. All upstream line
numbers cited in the issues resolve at `0b2e6bb`. In short: the `mix_scale`
floor is real and has been removed; the `premix_omega`, `save_start_values`
and renormalisation-threshold claims are true and fixed at PR head; the
composition relaxation is only partly fixed (F3-9); the outer binary retry
factor of 0.25 remains and is not upstream's 0.5; the claim in #139 that
upstream retries the whole step when the solve fails inside the implicit wind
loop is true (`star/private/evolve.f90:677-681`); the "f64 quantization
ceiling at dt = 111 s" arithmetic is right but the conclusion drawn from it is
unsound; "7 of 8 uncapped binary campaigns with full physics oracle parity" is
not supported as worded, because one uncapped campaign runs un-ignored and no
criterion compares with upstream output.

## Ground truth: how to rebuild the reference

1. Fetch the bundles: `python3 src/flexmesa/data/fetch.py fetch
   --include-test-inputs` (needs `zstandard`).
2. Clone upstream outside this repository and check out the commit:
   `GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout
   https://github.com/MESAHub/mesa`, then `git checkout 0b2e6bb`.
3. Fill the Git LFS pointer files and verify each against its pointer with
   `notes/audit/tools/fill_lfs.py <mesa> <source dir>...`. Sources that
   worked: the unpacked MESA 26.04.1 Zenodo archive (record 19722306) for 190
   of 196 files once combined with the FlexMESA test-input bundle, and
   `https://media.githubusercontent.com/media/MESAHub/mesa/<commit>/<path>` for
   the remaining six (`colors/colors_data.txz`, `forum/forum-1.2.3.tar.gz`,
   `gyre/gyre-9.1.1.tar.gz`, `kap/AESOPUS_AGSS09.h5`, `kap/kap_data.tar.xz`,
   `kap/preprocessor/kap_input_data.tar.xz`).
4. Install MESA SDK 26.6.1 (Zenodo record 20598423; needs `tcsh`), source
   `bin/mesasdk_init.sh`, set `MESA_DIR`, and run `make -j4 WITH_PGSTAR=no`
   (the `install` script refuses to run as root; `make` is what it runs).
5. Run a case with `notes/audit/tools/run_mesa_case.sh`, trace the same case
   in FlexMESA with `cargo run --release -p flexmesa-star --example
   star_case_trace -- <case_dir> <out.tsv>`, and compare with
   `notes/audit/tools/cmp_hist.py`.

Practical notes. The upstream tree needs about 20 GB with data and caches.
MESA needs the EOS text tables to remain present even when binary caches
exist. FlexMESA writes fitted-spline caches into its data directory at run
time, and that directory grew from 2.6 GB to 4.7 GB during this audit.

## Changes made on this branch

Only changes that are small, clearly correct and verified were made.

- Relabelled the upstream reference from `r24.08.1` to `main` at commit
  `0b2e6bb` in 15 files (text only). Two test function names and the RSP
  fixture file names still contain `r24`; renaming them is left for the
  package that touches those tests.
- Added `crates/flexmesa-star/examples/star_case_trace.rs`, which runs one
  upstream star case and writes a per-step history comparable with MESA's
  `history.data`.
- Added the comparison tools under `notes/audit/tools/` and the first stored
  upstream histories under `notes/audit/upstream_histories/`.

Nothing else in the code was changed. No tolerance was loosened, no expected
value re-pinned, no test ignored, and no case-name branch added.

## State of the step-trace tooling

The opt-in step telemetry and `binary_case_trace` on branch
`claude/tender-brahmagupta-cixhee` (commit `0d4fc80`) sit on `4aa730e`, two
commits behind PR #138 head, and do not apply cleanly to it: the change to
`crates/flexmesa-binary/src/e2e.rs` conflicts because the runner was
restructured. The other six files apply. A port of the `e2e.rs` hunk onto
`91eda0b` (an `_observed` variant of `run_binary_test_suite_case` that calls
the observer after construction and after each accepted step) built and ran
in this audit but is not committed here, because this branch is based on
`main`, which has no coupled binary driver.

## A conflict between the repository rules and honest auditing

`CLAUDE.md` requires agent-authored GitHub text to end with a footer naming a
specific tool, and requires every merge to pass all gates with every case
green. The first would mislabel work done by another agent. The second
cannot be met honestly today, because the cases are green only through the
mechanisms described above. Both are recorded as governance findings (G1-7
and section 4.2 of [`docs-governance.md`](docs-governance.md)). Nothing was
posted to GitHub during this audit.

## What this audit did not get to

- Runtime comparison covers the ten star cases listed in
  [`measurements.md`](measurements.md) only, not all 107; four of those did
  not finish in FlexMESA within the time limit and so have no trajectory
  comparison. No binary or astero case was compared against real MESA in this
  audit.
- The Wave 2, 3a, 3b and 4 Fortran harnesses were built against real MESA but
  their checksums were not compared with the documented values (the run was
  interrupted); Wave 5 does not link without further library additions.
- The uncapped binary and astero campaigns marked `#[ignore]` ("verified
  locally") were not all re-run.
- The cause of the difference between the `double_bh` trajectory in issue
  #137 and the one obtained from the same commit was not found. Single-star
  runs are deterministic across thread counts; the binary driver was not
  tested for this.
- Not read line by line: the RSP and RSP2 modules, pulsation solver internals,
  `approx21` and the rate library fits, the PC, Skye, CMS and eosPT EOS
  components, AESOPUS and Type 2 opacity interpolation, the TDC kernel,
  atmosphere tables, most of `flexmesa-autodiff`, the Python packages other
  than those sampled, and the tutorials and guides in `docs/`. Each register
  ends with its own "Not covered" list.
- Coverage-map classifications marked "quick" rest on routine names and file
  sizes. The weighted "about 41% ported" figure has an uncertainty of about
  10 percentage points.
- Effect sizes in the reading-based registers are inferred unless a
  measurement is cited.
