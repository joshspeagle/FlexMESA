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
| C14 | #165 | Math kernels and the bit-level parity decision | N1-6, N1-7 | none | M |
| C15 | #166 | Scope and honesty of the Python/JAX layer | N1-23 to N1-27, T1-11 | none | M |

## Suggested order

1. In parallel, first: A0, A2, A4, A1, C12. None touches the physics code
   paths.
2. Then A3, A5, B1, B4, C13, C14, C15. B1 and B4 both touch the star driver;
   do them one after the other.
3. Then B2 and B3, each followed by a regenerated case table. After B2 most
   tier 2 to 5 cases will no longer run, which is the correct status.
4. Then Phase C: C1 and C2 first, because they change every trajectory; then
   C3 and C4; then the rest in the order the generated case table shows to
   matter.

## Decisions needed from the project lead

| Decision | Package | Options |
| :-- | :-- | :-- |
| Upstream reference | A0 | Release 26.4.1 (proposed) |
| Parity target | C14 | Bit-level at kernel level through a correctly rounded math layer, or stated tolerances everywhere |
| Scope of the JAX layer while the Rust engine is unverified | C15 | Freeze `star`, `binary`, `astero` twins; keep microphysics twins |
| Orbital integrator | C9 | Match upstream's first-order step by default, offer the higher-order one as an option |
| Mode frequencies | C11 | Link GYRE and ADIPLS, or validate the in-house solver against GYRE |
| Treatment of upstream bugs found in passing | C1, C10 | Match upstream by default and document; never diverge silently |

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

| Earlier issue | Status after the audit | Now depends on |
| :-- | :-- | :-- |
| #89 end-to-end epic, #91 to #96 tiers | Open; re-scoped so that a case counts only when judged against a stored MESA history | A2, A3, B1, B2, then Phase C |
| #109 fidelity epic | Open as umbrella; Fidelity-1 confirmed for the default path, Fidelity-2 not confirmed | Phase C |
| #114 Fidelity-3 | Open until PR #138 is resolved | B3, C6, C9, C10, C11 |
| #115 Fidelity-4 | Open; the shortcuts it set out to remove are still present | B1, B2, C3 |
| #137, #139 `double_bh` | Open; claims checked in `binary-astero.md` Part 2 | B2, B4, C1, C3, C7, C8, C9 |
| #110 paper figures, #116 to #121 | Blocked; current atlas is synthetic | A1, A2, then the relevant Phase C packages |
| #90 documentation | Open | A1, A5, A0, C15 |
| #3 plotting | Deferred, unchanged | C12 provides the output files plotting would read |
