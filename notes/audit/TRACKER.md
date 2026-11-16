# Work tracker

The work that follows from the audit is tracked in GitHub issues. The parent
issue is [#140](https://github.com/joshspeagle/FlexMESA/issues/140); each
package below is a sub-issue of it, sized for one pull request (larger ones
say where they can be split). This file mirrors the issue list so that the
mapping from findings to packages is under version control. The issues are
the live record; update this file when packages are added, split or closed.

Finding identifiers refer to the registers in this directory. `M-n` refers to
[`measurements.md`](measurements.md). Sizes: S under a day, M a few days, L a
week or more.

## Packages

| Pkg | Issue | Title | Main findings | Depends on | Size |
| :-- | :-- | :-- | :-- | :-- | :-- |
| A0 | #141 | Re-baseline the upstream reference to MESA release 26.4.1 | G1-6, N1-10, T2-15, G1-4 | none | M |
| A1 | #142 | Correct public claims, scorecards and synthetic figures | T1-1, T2-1, T2-7, T2-8, G1-10, G1 claims register | none | S to M |
| A2 | #143 | Upstream-history end-to-end harness and stored reference histories | T1-1, T2-1, T2-7, M-1 to M-10 | A0 for final histories | M |
| A3 | #144 | CI that runs real campaigns, and a declared-known-gaps mechanism | T1-8, T1-9, T2-14, G1-7 | A2 | M |
| A4 | #145 | Governance: expected values, pass criteria, the meaning of "green" | G1-7 to G1-9, G1-11, T1-5, T1-10, T2-11, T2-12, N1-30 | none | S |
| A5 | #146 | Benchmark honesty and provenance | G1-1 to G1-5, T1-5, T2-12 | A0 | M |
| B1 | #147 | Fail loudly on unsupported controls, missing data and discarded errors | C1-6, C1-13, F1-3, F1-7, F2-1, F2-5, F2-6, F2-10, F2-12, F2-16, F2-18, N1-14, N1-21, N1-22, N1-28 | none | S to M |
| B2 | #148 | Move test-case code out of the library and delete fabricated behaviour | C1-1, C1-2, C1-4, C1-5, C1-8 to C1-12, C1-15, F2-13, F2-22, T2-2 to T2-6, M-6 | A2, B1 | M |
| B3 | #149 | Defaults parity with upstream `*.defaults` | C1-14, F1-32, F2-8, F2-9, F2-14, F2-15, F2-17, F3-5, F4-4, F4-30 | A0, A2 | S |
| B4 | #150 | Error type, telemetry, counters, checkpoints, removal of whole-model clones | F1-9, F1-18, F1-26, F1-31, F3-7, N1-15, N1-17 to N1-20, T2-8, M-8 | none | M |
| B5 | #151 | Energy accounting as upstream does it | F1-1, F1-28 | A2 | M |
| C1 | #152 | Newton iteration control and retry semantics | F1-2, F1-3, F1-5, F1-8, F1-11 to F1-17, C1-17 | A2, B4 | M to L |
| C2 | #153 | Mesh planner | F1-33, F1-34, C1-18, F4-26, F4-27 | A2 | M |
| C3 | #154 | Relaxation machinery and initial-model construction | C1-1, C1-3, F1-24, F3-9, F3-10, M-6 | A2, B2, C1 | L |
| C4 | #155 | Convective boundaries and mixing | C1-16, F4-1 to F4-13, F2-19, M-2 | A2, B2 | L |
| C5 | #156 | Element diffusion and phase separation | C1-11, C1-16, M-5 | A2, B2 | L |
| C6 | #157 | Time-dependent convection state handling and turbulence wiring | F1-4, F1-20, F1-21, F2-17, F2-18, F4-12 | A2, B3 | M |
| C7 | #158 | Rotation, angular momentum transport and rotational mixing | F4-20 to F4-26, F3-6, M-7 to M-10 | A2, B2, C1 | L |
| C8 | #159 | Winds, mass change and stop conditions | F4-14 to F4-19, F4-28 to F4-30, F1-10, F3-13 | A2 | M |
| C9 | #160 | Binary step structure, orbital integrator and implicit mass transfer | F3-1 to F3-8, F3-11 to F3-14 | A2, B4, C1, C3 | M to L |
| C10 | #161 | Microphysics control coverage and gaps | F2-1 to F2-3, F2-7, F2-11, F2-14, F2-15, F2-23, F2-24, T1-11, T2-15 | A0, B1 | L |
| C11 | #162 | Pulsation, asteroseismology and optimizers | F3-15 to F3-20, N1-1 to N1-5, T2-5, T2-10 | A0, A2 | L |
| C12 | #163 | History and profile output with upstream column definitions | F4-28, coverage map | none | M |
| C13 | #164 | Data pipeline provenance and runtime caches | N1-10 to N1-14, F2-4 | A0 | M |
| C14 | #165 | Numerics policy: strict and fast modes from one Rust source | N1-6, N1-7, N1-8 | A0 for exact reference outputs | M |
| C15 | #166 | JAX contract: derived, differentiable and batched; validated against Rust strict mode | N1-23 to N1-27, T1-11 | C14 | M |
| A6 | #168 | Profile a real evolution; measure fast math, batching and waste removal | N1-9, N1-17, N1-18, G1-5 | none; B4 and C14 for later parts | S to M |
| D1 | #170 | Port GYRE | F3-15, F3-16 | A0, A2, C14, C1 to C3 | L |
| D2 | #171 | Port ADIPLS | F3-15 | A0, A2, C14, C11 | L |
| D3 | #172 | Port STELLA, after all test-suite cases pass | coverage map | v0.1.0 criteria | L |
| E1 | #173 | Plotting as we go: pgstar and pgbinary replacement | G1-10, coverage map | C12, A2 | M, then incremental |
| E2 | #174 | Generated progress dashboard in README and docs | G1-7, G1-9, coverage map | none; A2 for the case table | M |

## Suggested order

Phase 0 is serial, one pull request at a time, before any parallel work:

1. PR #138 landed as unverified code with its claims removed (instructions
   on the pull request).
2. A4 (#145): the rules. From this point the new rules apply to every agent.
3. A2 (#143) with a small per-pull-request CI job, and C12 (#163).
4. A0 (#141): the 26.4.1 reference, exact reference outputs, reference
   histories.
5. C14 (#165): the math layer, the routing lint and the cross-machine check.
6. E2 (#174): the dashboard, so progress is visible from the start.

After Phase 0, in parallel where files do not collide:

- A1, A3, A5, A6, B1, B4, C13, C15, E1. B1 and B4 both touch the star
  driver; do them one after the other.
- Then B2 and B3, each followed by a regenerated case table. After B2 most
  tier 2 to 5 cases will no longer run, which is the correct status.
- Then Phase C: C1 and C2 first, because they change every trajectory; then
  C3 and C4; then the rest in the order the generated case table and the
  per-case unsupported-feature report show to matter.
- D1 and D2 after the core engine; D3 after every test-suite case passes.

Collision groups (never run concurrently): B1, B4 and C1 (star step driver
and solver); B2 and C3 (model initialisation).

## Decisions made by the project lead (2026-11-16)

| Topic | Decision | Recorded in |
| :-- | :-- | :-- |
| Upstream reference | MESA release 26.4.1 | #141 |
| Numerics | Strict (hardware-independent, default, judged against MESA) and fast modes from one Rust source; three fidelity levels | #165 |
| Architecture | Rust canonical; Python front end; JAX derived, for derivatives and batching, validated against Rust strict; JAX engine twins frozen | #166 |
| Departures from upstream | None before v0.1.0, and no option flags. Bugs and enhancements are deferred issues under #169. Existing departures are removed | #145, #169, #175 to #178 |
| Bundled codes | GYRE, ADIPLS and STELLA are in scope and are ported; STELLA last | #170, #171, #172 |
| Plotting | Built as the port proceeds | #173 |
| Progress reporting | Generated dashboard in README and docs | #174 |
| v0.1.0 | Defined in [`../SCOPE.md`](../SCOPE.md) | #145 |
| PR #138 | Merge the code, not the claims | PR #138 |
| Earlier issues | Closed as superseded; #140 is the only tracker | below |

Still open: whether the 32 upstream development cases are part of a later
milestone; what to do with behaviour that cannot be reproduced exactly (asked
case by case).

## Post-parity backlog

Parent issue #169. Sub-issues so far: #175 (higher-order orbit integrator
and gravitational-wave eccentricity term), #176 (upstream bug in the
HELM-edge blend derivatives), #177 (upstream bugs in `check_dX` and
`sizequ`), #178 (RSP2 "intentional fixes" to verify and remove).

## Data and release assets to plan

- `mesa-test-inputs-v2`, built from the 26.4.1 release archive with a
  per-file hash manifest (A0).
- `mesa-histories-v1`: real-MESA histories for every case of the pinned
  release, with provenance (A0, A2).
- Regenerated upstream fixtures with committed recipes: RSP models, the
  `star/test` history, EOS oracle tables (A0, C10).
- Rebuilt physics data bundles with committed converters (C13).
- A container image or recipe for MESA at the pinned release (A3).

## How earlier issues map onto the packages

All of these were closed on 2026-11-16 as superseded by #140. Each carries a
comment explaining how the audit changed its status.

| Earlier issue | Carried by |
| :-- | :-- |
| #89 end-to-end epic, #91 to #96 tiers | A2, A3, B1, B2, then Phase C; acceptance is the generated case table |
| #109 fidelity epic | Phase C |
| #114 Fidelity-3 | B3, C6, C9, C10, C11 |
| #115 Fidelity-4 | B1, B2, C3 |
| #137, #139 `double_bh` | B2, B4, C1, C3, C7, C8, C9 |
| #110 paper figures, #116 to #121 | A1, then E1 once the cases behind each figure pass |
| #3 plotting | E1 (#173) |

Still open from before the audit: #90 (documentation), which depends on A1,
A5, A0 and C15.
