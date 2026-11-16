# Measurements: FlexMESA against real MESA, case by case

All statements in this file are **verified by running**.

## Set-up

- **Real MESA**: Fortran MESA `main` at commit `0b2e6bb` (the upstream of the
  repository's test-input bundle), built with MESA SDK 26.6.1, all 23 module
  self-checks passing. Cases were run from `star/test_suite/<case>` with
  `MESA_SKIP_OPTIONAL=t`, so optional early stages are replaced by the saved
  models shipped with the case, and the final inlist stage is evolved.
- **FlexMESA**: PR #138 head `91eda0b`, release build, run with the
  `star_case_trace` example with no step cap and with `skip_optional`, which
  takes the same branch of each case's `rn` script.
- Both codes therefore start the final stage from the same upstream saved
  model wherever FlexMESA actually loads it (see the notes per case).
- Machine: 4 cores, 15 GB. The two codes ran at the same time and shared the
  cores, so wall times are indicative only and must not be read as a
  performance comparison. FlexMESA runs had a 20 minute limit.
- Comparison: final values, and the largest difference along the track after
  interpolating both histories onto a common grid in `star_age`
  (`tools/cmp_hist.py`). Thinned real-MESA histories are stored in
  `upstream_histories/`.

## Results

| ID | Case | Tier | Real MESA | FlexMESA | Final-state and track differences (FlexMESA minus MESA) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| M-1 | `1.3M_ms_high_Z` | 1 | `log_L_upper_limit`; 219 steps, 0 retries | same termination; 346 steps, 0 retries; reported `passed=true` | age +0.9% (4.680 against 4.636 Gyr); log L +0.0009; log Teff -0.0002; log R +0.0009; largest log L difference on the track 0.12 dex near core-hydrogen exhaustion; final zones 1178 against 1220 |
| M-2 | `hb_2M` | 1 | `xa_central_lower_limit`; 195 steps, 2 retries | same termination; 204 steps, 0 retries; reported `passed=true` | **age -47%** (71.4 against 135.6 Myr of core helium burning); log L -0.155; log R -0.102; log Tc -0.186; final zones 2412 against 3000 |
| M-3 | `semiconvection` | 1 | `xa_central_lower_limit`; 293 steps, 0 retries | same termination; 345 steps, 0 retries; reported `passed=true` | age +0.6%; log L -0.0047; log Teff -0.0010; log R -0.0005; log Tc +0.0016; largest log L difference on the track 0.03 dex; final zones 1637 against 1662 |
| M-4 | `1M_thermohaline` | 1 | `he_core_mass_limit`; 2512 steps, 0 retries, 27 min | not finished after 20 min (killed) | not compared |
| M-5 | `wd_cool_0.6M` | 2 | `log_center_temp_lower_limit`; 210 steps, 1 retry | same termination; 403 steps; reported `passed=true` | **cooling age -9.5%** (8.84 against 9.77 Gyr); log L +0.011; log Teff +0.003; zone count fixed at 731 for the whole run against 731 to 1410 in MESA; no nuclear burning (`log_LH` at its floor against -10.2) |
| M-6 | `make_brown_dwarf` | 2 | `max_age`; 237 steps, 0 retries, 1 min | `max_age`; 847 steps, 16 min; reported `passed=true` | **a different object**: log R -2.21 against -0.99; log g 8.19 against 4.41; log Teff 3.39 against 2.31; log L -5.9 against -7.8; central density 4e5 against 51 g/cm3; final age overshoots `max_age` by 0.9%; final zones 3817 against 557 |
| M-7 | `hot_cool_wind` | 3 | `log_L_upper_limit`; 561 steps, 25 retries, 15 min | not finished after 20 min (killed) | not compared |
| M-8 | `magnetic_braking` | 3 | `xa_central_lower_limit`; 221 steps, 0 retries, 5 min | **fails**: `star_evolve_step failed after 24/100 retries (dt=1.000e-6 s, dt_floor=1.000e-6 s)` after 19 min | not compared |
| M-9 | `conserve_angular_momentum` | 3 | `he_core_mass_limit`; 369 steps, 2 retries, 7 min | not finished after 20 min (killed) | not compared |
| M-10 | `15M_dynamo` | 3 | first stage completed; run still in progress when this file was written | not finished after 20 min (killed) | not compared |
| M-11 | `hse_riemann` | 3 | not run before this file was written | `max_age_in_seconds`; 2001 steps, 0 retries, 4 min; reported `passed=true` | not compared; the tier 3 extras switch `u_flag` off for this case (C1-9), so the Riemann hydrodynamics it is named for is not exercised |
| M-12 | `rsp_Cepheid` | 4 | not run before this file was written | `extras_finish_step`; 7266 steps, 0 retries, 0.6 min; reported `passed=true` | not compared |

Not completed when this file was written: `16M_predictive_mix` in both codes,
and the real-MESA runs of `15M_dynamo` (in progress), `hse_riemann` and
`rsp_Cepheid`. No binary or astero case was compared in this audit beyond the
`double_bh` comparison that preceded it.

## What the measurements show

**M-1, M-3. The real code path is close to MESA on simple main-sequence
evolution.** Ages agree within 1% and surface quantities within about 1e-3 to
5e-3 dex at the end. Step counts differ by 20 to 60%, which is expected given
that the Newton iteration control, the retry policy, the mesh planner and
several defaults are not upstream's (F1-12 to F1-17, F1-33, C1-14). The
largest differences along the track occur around core-hydrogen exhaustion,
where a small age offset shifts a rapid change.

**M-2. Core helium burning with predictive mixing is wrong by a factor of
two in lifetime.** Both codes start from the same upstream model (age 439 yr,
central helium 0.9795 and 0.9796, the same luminosity). MESA's convective
core mass grows from 0.15 to 0.27 Msun and central helium lasts 136 Myr.
FlexMESA's central helium falls below 0.03 after 42 Myr and is exhausted at
71 Myr. The most plausible cause is that the predictive-mixing surrogate never
grows the core (F4-2, C1-16); this attribution is inferred, not proven. The
case is reported as passed.

**M-5. A white dwarf that starts from the real upstream model cools at
roughly the right rate, with the tier 2 overrides visible.** The zone count
never changes and there is no burning, both forced by `tier2_extras.rs`
(C1-9). The cooling age is 9.5% short.

**M-6. A case that starts from a fabricated model is a different object.**
`make_brown_dwarf` asks for a 0.01 Msun model built with
`create_initial_model`. FlexMESA builds it by rescaling a 1 Msun ZAMS model
with a rescaled gravitational constant and opacity (C1-1, C1-2). The run
reaches the upstream termination condition and is reported as passed, with a
radius 17 times smaller and a surface gravity nearly four orders of magnitude
larger than MESA's.

**M-7 to M-10. Tier 3 cases do not complete when run uncapped.** One fails to
converge with the timestep collapsing to the floor; three did not finish in
20 minutes where MESA needs 5 to 15 minutes on the same shared machine. The
error message of the failure carries no model number, zone or equation
(N1-19). All four are reported as passed by the 2-step tests.

**Every completed FlexMESA run reported `passed=true`**, including M-2 and
M-6. A capped run also reports `passed=true` with termination
`max_steps_cap`. This is the measured form of findings T2-1 and T2-7.

## Other measurements

- **Determinism.** `1.3M_ms_high_Z`, first 80 steps, with 1, 3 and 4 worker
  threads and one repeat at 4 threads: the four history files are
  byte-identical. The binary driver was not tested.
- **`main` against PR #138 head.** The same 80 steps on `main` (`8f6195b`)
  differ from PR head from the first step, as expected from the star-crate
  changes in the PR.
- **Wave 1 Fortran benchmark checksums.** The repository's
  `benchmarks/fortran_mesa` Wave 1 harness, built against real MESA at
  `0b2e6bb`, reproduces all 18 "Fortran MESA Checksum" values in
  `docs/design/wave-1-benchmark-report.md` exactly. Building needed the
  library and include paths adjusted to MESA's current `build/<module>`
  layout and the SDK's `crmath`, `lapack` and `hdf5` link flags. The Wave 2
  to 4 harnesses also built once private module directories were added to the
  include path; their checksums were not compared. Wave 5 did not link.
- **Memory.** FlexMESA single-star runs peaked at 2.2 to 3.1 GB resident.
- **Data directory growth.** FlexMESA's data directory grew from 2.6 GB to
  4.7 GB during these runs through fitted-spline caches written at run time
  (N1-11).
- **Release distance.** `0b2e6bb` is 265 commits past its common ancestor
  with release tag `26.4.1`; the release branch has 98 commits of its own.
  Between the two, 216 physics source files differ, `controls.defaults`
  differs by 343 lines, 4 module `test_output` files differ, and the release
  has 90 star test cases against 91.

## Reproducing

See "Ground truth" in [`README.md`](README.md). In short, with `MESA_DIR`,
`MESASDK_ROOT` and `OUT` set:

```
notes/audit/tools/run_mesa_case.sh star hb_2M
cargo run --release -p flexmesa-star --example star_case_trace -- \
    "$MESA_TEST_INPUTS/star/test_suite/hb_2M" hb_2M.tsv
python notes/audit/tools/cmp_hist.py \
    notes/audit/upstream_histories/hb_2M.tsv hb_2M.tsv
```

where `MESA_TEST_INPUTS` is the unpacked `mesa-test-inputs-v1` bundle.
