# Scope

What FlexMESA ports, in what order, and what the first release requires.
Decided by the project lead on 2026-11-16. Rules that go with this file are in
issue #145; the work is tracked in issue #140 and mirrored in
[`audit/TRACKER.md`](audit/TRACKER.md).

## Principles

1. **Reference.** The reference is MESA release 26.4.1 (git tag `26.4.1`,
   commit `c8c2026`, Zenodo record 19722306). All bundled test inputs, module
   golden outputs, exact hexadecimal module test outputs, and stored reference
   histories come from MESA 26.4.1 (#141, `DECISIONS #22`).
2. **Everything MESA ships is in scope.** That includes the third-party codes
   MESA bundles (GYRE, ADIPLS, STELLA) and a replacement for its plotting.
   The goal is a single package that can reproduce every test case in the
   MESA test suites with no other code at run time.
3. **Parity first, no option flags.** Until v0.1.0, FlexMESA reproduces the
   reference exactly, including its bugs. An upstream bug, or a clear
   improvement over upstream, is recorded as a deferred issue under the
   post-parity backlog (#169) and is not implemented. The only run-level
   switch is the numerics mode (#165).
4. **Port what the next test cases need.** Order is derived from the test
   suites, using the per-case report of unsupported controls and features
   (#147), and not from what is easiest to port.
5. **Numerics.** One source of physics in Rust with two modes. Strict mode
   is the default, is bit-identical on all hardware, and is the only mode
   judged against MESA. Fast mode uses platform arithmetic and is validated
   against strict mode (#165).
6. **Languages.** Rust is canonical. Python is the front end. JAX is a
   derived layer for derivatives and batching, validated against Rust strict
   mode, and never an oracle. JAX twins of the `star`, `binary` and `astero`
   engines are frozen until the Rust engine passes the history harness
   (#166).

## v0.1.0

Every case in the MESA 26.4.1 test suites (90 star, 8 binary, 8 astero) runs
its unmodified inlists to upstream's termination condition, taking exactly as
many steps in each stage as MESA's own run (DECISIONS #70), and meets its
history-harness tolerances in strict mode; module golden tests pass exactly;
strict mode is bit-identical across machines; there are no undeclared gaps
and no departures from upstream.

## Modules

Upstream line counts are Fortran source and include files, excluding test
programs and test suites, counted on MESA 26.4.1 (#141). "Status today" is from
the audit ([`audit/coverage-map.md`](audit/coverage-map.md)); it will be
replaced by the generated dashboard (#174).

### Core, required for v0.1.0

| Upstream module | Lines | Status today | Packages |
| :-- | --: | :-- | :-- |
| `star` (evolution engine, excluding plotting) | about 120,500 | about 16% ported, 48% partial, 21% stub, 15% absent | #147 to #159, #161, #163 |
| `star_data` | 7,300 | replaced by Rust state structs; controls coverage 43% of `&controls`, 25% of `&star_job` | #147, #149 |
| `binary` (excluding plotting) | about 11,600 | closed-form kernels match; step structure does not | #160 |
| `astero` | 8,300 | chi-squared and surface corrections match; searches and mode sources do not | #162 |
| `eos` | 28,700 | kernels compared so far match; 5 of 102 controls read; several components not yet compared | #161 |
| `kap` | 22,400 | blends and conduction match; 5 of 36 controls read; OP monochromatic opacities absent | #161 |
| `net` | 13,500 | driver and screening match; `approx21` and general-net paths not yet compared | #161 |
| `rates` | 17,200 | table interpolation matches; 29 of 183 rate routines present | #161 |
| `neu` | 2,200 | driver matches; fits not yet compared; golden test covers all 10,000 rows | #161 |
| `atm` | 6,900 | T(tau) relations match; tables and other paths not yet compared | #161 |
| `turb` | 2,000 | MLT, semiconvection and thermohaline match; TDC kernel not yet compared; state handling wrong in `star` | #157 |
| `chem`, `ionization`, `colors` | 11,900 | golden tests cover substantial parts; data tables not compared | #161 |
| `num` | 20,000 | several integrators and optimizers are simplified stand-ins; golden coverage about 8% | #162, #161 |
| `mtx` | 5,800 | block solvers algebraically faithful, not in upstream operation order | #165 |
| `interp_1d`, `interp_2d` | 18,500 | golden tests cover selected rows | #161 |
| `auto_diff` | 14,300 | golden test covers all records | none open |
| `math` | 900 | ports the non-default math file; physics crates bypass it | #165 |
| `const`, `utils` | 3,700 | values and goldens match | none open |

### Bundled third-party codes

| Code | Lines | Needed by | When | Package |
| :-- | --: | :-- | :-- | :-- |
| GYRE 8.1 | about 55,700 | `gyre_in_mesa_*` (6 cases), `rsp_gyre`, astero cases using `gyre` | after the core engine; required for v0.1.0 | #170 |
| ADIPLS | about 30,900 | astero cases using `adipls` | after the core engine; required for v0.1.0 | #171 |
| STELLA | about 62,000 | no test case runs it; `ccsn_IIp` writes its input files | after every test-suite case passes; not required for v0.1.0 | #172 |

Until GYRE and ADIPLS are ported, real GYRE and ADIPLS (which run inside the
reference MESA build) supply reference frequencies to the test harness. They
never ship in the package. The existing in-house pulsation solver is
scaffolding and no case that depends on it counts as passing.

### Plotting

| Upstream | Lines | Approach | Package |
| :-- | --: | :-- | :-- |
| `pgstar`, `pgbinary` and their controls | about 35,300 | Not translated line by line. Replaced by Python plotting that reads history and profile output, built incrementally as the port proceeds, with upstream's panel definitions as the specification | #173 |

Plotting is built as we go. It is required for inspecting behaviour during
development, and comparison figures are part of every trajectory-changing
pull request once #143 and #163 exist.

### Not in v0.1.0

| Item | Status |
| :-- | :-- |
| STELLA port | In scope, after v0.1.0 unless a test case proves to need it (#172) |
| The 32 upstream development cases (`star/dev_cases_*`) | Confirmed for after v0.1.0 (#145) |
| Upstream bugs and enhancements | Deferred, recorded under #169 |
| JAX twins of the `star`, `binary` and `astero` engines | Frozen until the Rust engine passes the harness (#166) |
| Native JAX kernels beyond the existing microphysics | Decided on profiling data (#168) |

## Changing this file

Scope changes are decisions for the project lead. Record them in
`notes/DECISIONS.md` and update this file and the tracker in the same pull
request.
