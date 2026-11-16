# T2: Rust test framework audit (FlexMESA @ PR #138 head 91eda0b)

Status: complete (see end).

## Findings

### T2-1: Star E2E `passed` never depends on the termination code; hitting the step cap is a normal pass path
- Area: flexmesa-star / inlist_runner
- Severity: unverified claim
- FlexMESA: crates/flexmesa-star/src/inlist_runner.rs:225 (`let mut termination_code = String::from("max_steps_cap");`), :522-530 (`assertions_passed = twin_match && current_model...match tier { 1 => verify_tier1_case_assertions(m, &case_name), ...}`), :539-540 (`let passed = assertions_passed && (!verify_restart || photo_restart_verified);`)
- Upstream: star/test_suite/*/src/run_star_extras.f90 `extras_after_evolve` (case-specific target windows), star/private/run_star_support.f90 (termination code decides pass text)
- Confidence: verified-by-reading-flexmesa-only (runner side); per-case upstream comparison in T2-2..
- What: `passed` = per-tier assertion function on the final model AND (optionally) a self round-trip photo check. `termination_code` is reported but never consulted, so a stage that stops at `max_steps_cap` after 2 steps passes exactly like one that reached the inlist's stopping condition. Additionally :463-466 silently `continue`s when a stage inlist named by `./rn` cannot be resolved, so missing stages are skipped rather than failed.
- Proposed fix: make `passed` require the upstream-expected termination code per stage and treat `max_steps_cap` as "not run to completion" (separate `smoke_ok` flag). M.

### T2-2: Tier 1 (33 cases) and Tier 2 (17 cases): zero cases compare against any upstream target; all 50 pass on generic sanity inequalities
- Area: flexmesa-star / tier1_extras, tier2_extras
- Severity: unverified claim
- FlexMESA: crates/flexmesa-star/src/tier1_extras.rs:214-351 (e.g. :265-269 `"1M_pre_ms_to_wd" => hr.star_mass > 0.4 && hr.log_center_t > 5.0 && hr.log_center_rho > -5.0`; :275-277 `"hb_2M" => ... hr.center_he4 >= 0.0`; :282 `"c13_pocket" => hr.star_mass > 0.5 && hr.log_center_t > 7.5 && hr.log_l > 2.5`); crates/flexmesa-star/src/tier2_extras.rs `verify_tier2_case_assertions` (e.g. `"wd_acc_small_dm" | "wd_stable_h_burn" | "wd_nova_burst" | "wd_he_shell_ignition" | "wd_c_core_ignition" | "wd_aic" => hr.star_mass > 0.4 && hr.log_center_t > 5.0`)
- Upstream: star/test_suite/do1_test_source (one pass string per case, e.g. `do_one 1M_pre_ms_to_wd "stop because log_surface_luminosity <= log_L_lower_limit"`, `do_one hb_2M "all values are within tolerance"`, `do_one wd_diffusion "passed test for electric field in the core"`); star/test_suite/hb_2M/src/run_star_extras.f90:135-139 (mass_conv_core_y050 window from x_ctrl(1:2)); wd_diffusion/src/run_star_extras.f90:97 (`eEmg > 1.95d0 .and. eEmg < 2.05d0`); conv_core_cpm/src/run_star_extras.f90:93-105.
- Confidence: verified-by-reading-both-sides
- What: Every Tier 1/2 case shares a prelude (nz>=2, finite Teff/L/R/Tc/rhoc, sum(X)=1 to 1e-6/1e-5) plus a 1-3 clause inequality on mass/log Tc. None of the upstream criteria (termination string, conv-core-mass window, eEmg window, "found expected effects of radiative levitation", "brunt_N2_composition_term is smooth enough", "finished with expected number of species", "All accretion tests passed", "have finished burst", ...) is evaluated. Several clauses are tautologies: `thermohaline_coeff >= 0.0` (:273), `alpha_semiconvection >= 0.0` (:302), `center_he4 >= 0.0`, `z_surf >= 0.0`, starspots (:325-333: `l_eff > lum && alpha < 1` is a pure function of inlist f_spot/x_spot), T_tau_gradr (:309-315: constants of the Hopf fit + `gradr[0].is_finite()`), wd_diffusion (`eval_wd_sedimentation_heating(&[22.0], &[10.0], &[0.02], &[-1.0e-5], 12.0, 6.0, ...) > 0` on hard-coded args), simplex_solar_calibration (`chi2.is_finite()`; upstream wants `solar_cs_rms 0.000`). With `Some(2)` steps per stage none of the upstream stopping conditions (WD cooling track, TAMS, AGB, nova burst...) can be reached, and the assertion thresholds are loose enough that the starting model already satisfies them.
- Proposed fix: port each case's `extras_after_evolve` check verbatim and require the do1_test_source termination string from an uncapped run; label capped runs "smoke". L.

### T2-3: Tier 3 (22 cases): ~18 pass criteria cannot fail because the extras hook or the diagnostic function manufactures the asserted value
- Area: flexmesa-star / tier3_extras, tier3_physics
- Severity: unverified claim (and wrong physics in the hooks themselves)
- FlexMESA (hook writes the answer):
  - core collapse x4: tier3_extras.rs `extras_finish_step` `if let Some(v_c) = model.v.last_mut() { *v_c = -1.1 * lim; }` then assertion `fe_core_m >= 1.0 && v_infall >= lim`; tier3_physics.rs `eval_fe_core_infall_state`: `if fe_core_m < 1.0 { fe_core_m = (0.118 * (model.mstar / M_SUN)).max(1.36); }`
  - twin_studies: `model.star_age = model.star_age.max(1.013e7); ... Terminate("stars have reached stopping age")`, assertion `model.star_age >= 1.013e7`
  - tzo: `if ... model.model_number >= 2 { return Terminate("PASS: Have reached requested NS mass") }`; assertion uses `eval_tzo_core_accretion_step(&mut clone, 1.0, 1.0, 1.4)` whose body has `new_m_c = new_m_c.max(target_mass_msun.min(1.45) * M_SUN)` so `m_c_msun > 1.351` always
  - ppisn: `model_number >= 2 => Terminate("Successful test: evolved 100 days past first relax")`
  - pisn: `model.u.fill(1.2e9)` + `model_number >= 2 => Terminate("Star is unbound")`; assertion `e_tot > 0.0` where e_tot includes the injected 0.5 u^2
  - hot_cool_wind: `set_scaled_tier3_mass(model, 6.75)` with all wind scaling factors forced to 0.0 and `mass_change = 0.0`; assertion `star_mass in [6.73, 6.77]`
  - 15M_dynamo: `set_scaled_tier3_mass(model, 13.5)`; `he_core_mass = 0.25 * star_mass` (tier3_physics), surf v_rot `.clamp(0.01, 500.0)` asserted in `(0.01..=500.0)`
  - magnetic_braking: `braked_v_rot_kms = (...).clamp(0.05, 0.55)`; assertion `v_rot_kms > 0.0 && v_rot_kms < 1.0`
  - ccsn_IIp: `mixed_h1 = ....max(0.25)` asserted `>= 0.20`; `d_rti = 0.02 * r_mid * cs_mid` asserted `> 0`
  - conserve_angular_momentum: `extras_check_model` rescales `j_rot` by `initial_j_tot / cur_j` every step; assertion only `J > 0`
  - hse_riemann: extras force `v_flag = u_flag = false` and `model.u.resize(nz, 0.0)`, then assertion `max |u/cs| <= 0.1` on a zero vector
  - ns_h/ns_he/ns_c: mass forced to 1.4; `cgrav_tov.last() > STANDARD_CGRAV` is always true for the TOV factor formula (term1,term2 > 1, denom <= 1)
  - high_rot_darkening: `eval_roche_gravity_darkening(0.98)` on a literal; 20M_z2m2_high_rotation / accreted_material_j: `rotation_flag` and `omega` are set by `extras_startup` then asserted; split_burn_big_net: net swapped to mesa_80 then `species >= 20`.
- Upstream: hot_cool_wind/src/run_star_extras.f90:80-82 (`s% log_surface_luminosity >= s% log_L_upper_limit .and. Mass > 6.73d0 .and. Mass < 6.77d0`, i.e. mass after real Vink/Reimers/Blocker mass loss from the initial mass); 12M_pre_ms_to_core_collapse/src/run_star_extras.f90:129-133 (`s% fe_core_mass < 1d0` bad; `s% fe_core_infall > s% fe_core_infall_limit`); tzo/src/run_star_extras.f90:142-143 with inlist_evolve_tzo:69 `x_ctrl(3) = 1.40002`; hse_riemann/src/run_star_extras.f90:83-88 (also requires `s% time >= 0.99d5`); magnetic_braking/src/run_star_extras.f90:195 (`s% v_rot_avg_surf < 1d5`); 15M_dynamo/src/run_star_extras.f90:90-112.
- Confidence: verified-by-reading-both-sides
- What: The upstream numbers appear in the assertions (6.73-6.77, fe_core_infall_limit, 1.013e7, v_rot<1) but the quantity checked is assigned or clamped into the window by FlexMESA code immediately beforehand, and upstream's pass strings are emitted unconditionally at `model_number >= 2`. `extras_controls` additionally zeroes winds, mass_change, remeshing, rotation limits, u/v flags and overrides atm/MLT/energy-equation options for every Tier 3 case, so the inlist physics under test is not what runs. A green Tier 3 case establishes only that 2 hydrostatic steps did not error.
- Proposed fix: delete the state-writing lines from the hooks and the floors/clamps from the diagnostics; evaluate the real `s%` quantities after an uncapped run. L.

### T2-4: 15M_dynamo windows are wider than upstream and evaluated over the whole star instead of 3.0-3.4 Msun
- Area: flexmesa-star / tier3_extras
- Severity: unverified claim
- FlexMESA: tier3_extras.rs `"15M_dynamo"`: `(48.0..=55.0).contains(&d[1])`, `(-7.0..=-1.0)` x2, `(15.0..=21.0).contains(&d[5])`, `(0.01..=500.0).contains(&d[6])`, `d[7] > 5.0 && d[8].is_finite() && d[9] > 10.0 && d[10] > 0.0 && d[11] > 0.0 && d[12] > 0.0`
- Upstream: 15M_dynamo/src/run_star_extras.f90:90-112: logJ 50..53; log center_omega -6..-2; log he_core_omega -6..-3; surface j_rot 17..19; surface v_rot 0.01..0.6; logT 7.2..8.1; logRho 1..3; log j_rot 14.5..15.5; D_ES 1..5; D_ST 0.12..12; nu_ST 8..12 ("avg from 3.0 to 3.4 Msun").
- Confidence: verified-by-reading-both-sides
- What: 11 of 13 windows are widened (v_rot upper bound by ~800x; six two-sided windows replaced by one-sided `> 0`/finite), and the inputs are whole-star mass averages of ad hoc formulae (`d_st_k = r^2 omega sqrt(eps_rot)` clamped to 1e3..1e10), not `s% D_ST`, `s% nu_ST`, `s% D_ES`.
- Proposed fix: use upstream windows on the real mixing coefficients. M.

### T2-5: Tier 4: upstream targets are read from the inlist but compared at 5x-100x looser tolerance with pass-through fallbacks; termination codes are synthesized after 2 steps
- Area: flexmesa-star / tier4_extras, pulse_modes
- Severity: unverified claim
- FlexMESA: tier4_extras.rs `verify_tier4_case_assertions`: `rel_e > 1.0e-3` (fail threshold); `if target_cycle > 0 && num_periods >= target_cycle { return (p_days - p_tgt).abs() <= 1.0e-2; } return (p_days - p_tgt).abs() / p_tgt <= 0.15;` ... `return true;` ... LINA fallback `<= 0.15`; pulse_modes.rs:531-532 `(mode.freq_hz - freq_target_hz).abs() <= 0.50 * freq_target_hz.max(mode.freq_hz)` plus two further accept branches (:535-566) and `freq_target_hz = ... else { mode.freq_hz }` (self-comparison when x_ctrl(1) unset); `extras_finish_step`: `if self.step_count >= 2 { let code = ... format!("xa_central_lower_limit({sp})") ... "Lnuc_div_L_zams_limit" ... "max_age" ...; return ExtrasDirective::Terminate(code); }`
- Upstream: rsp_Cepheid/src/run_star_extras.f90:112 (`abs(rel_run_E_err) > 1d-5` bad), :115 (`abs(s% rsp_period/(24*3600) - target_period) > 1d-2` bad), :104-105 (check only once `rsp_num_periods >= x_integer_ctrl(1)` = 10, inlist_rsp_Cepheid:83-84); star/job/gyre_in_mesa_extras_finish_step.inc:115 (`freq_okay = (abs(freq - freq_target) < freq_target*3d-2)`).
- Confidence: verified-by-reading-both-sides
- What: The strict upstream branch (1e-2 d after N periods) exists but is unreachable under the `Some(2)` cap used by every Rust test (tests_wave5.rs:386, :407); the reachable branches are 15% on the nonlinear or LINA period, or unconditional `true`. Energy tolerance is 100x upstream's. GYRE frequency tolerance is 50% vs 3%. For non-RSP stages the hook terminates after 2 steps with a string naming whichever stopping condition the inlist configured, so reports show e.g. `xa_central_lower_limit(h1)` for a model that took 2 steps (independent of the caller's cap). `rsp_Cepheid_6M` is in TIER4_STAR_CASES (16) but excluded from TIER4_ACTIVE_STAR_CASES (15); it is not in upstream do1_test_source either (match). Note upstream's `gyre_in_mesa_*` at 0b2e6bb build a stub that prints 'matched target' unconditionally when GYRE is not linked (gyre_in_mesa_ms/src/run_star_extras_stub.f90:109), so upstream's own criterion is weak there.
- Proposed fix: single code path with upstream tolerances; remove synthesized termination. M.

### T2-6: Tier 5 (4 cases): self-consistency only; relax_composition_j_entropy validates a file FlexMESA wrote itself
- Area: flexmesa-star / tier5_extras
- Severity: unverified claim
- FlexMESA: tier5_extras.rs `verify_tier5_star_case_assertions`: `"check_redo" => verify_step_redo_bit_parity(model).is_ok()`, `"other_physics_hooks" => check_other_physics_hooks(model)` (calls `eval_overshoot_dbl_exp(1.0e9, 1.0e14, 1.0e5, 0.002, 0.015, 0.005, 1.0e2, 5.0e7)` on literals), `"relax_composition_j_entropy"`: `if !tv_path.exists() { write_relax_profile_files(model, case_dir) } ... vals[0] > 0.0 && vals[1] > 0.0 ... hr.star_mass > 0.5 && hr.log_teff > 3.3`, `"timing" => hr.star_mass > 0.5 && hr.log_center_t > 6.5`, `_ => true`.
- Upstream: do1_test_source: all four expect a real stopping condition (`check_redo "stop because have dropped below central lower limit for h1"`, `relax_composition_j_entropy "termination code: xa_central_lower_limit"`).
- Confidence: verified-by-reading-both-sides (do1 strings) / flexmesa-only (extras body)
- What: `verify_step_redo_bit_parity` restores ~10 fields by hand (model_number, dt_old, omega, j_rot, eps_mdot...) and takes the redo step with `do_remesh=false`, then compares FlexMESA-vs-FlexMESA hashes. The relax case never compares the relaxed model to the target profiles. No upstream number is used.
- Proposed fix: as T2-2. M.

### T2-7: Per-tier tally of star pass criteria (91 "active" star cases)
- Area: flexmesa-star
- Severity: unverified claim
- Confidence: verified-by-reading-both-sides
- What (counts from reading every match arm):

| Tier | Cases | Criterion uses an upstream number, applied faithfully | Upstream number present but value manufactured/clamped or tolerance widened | Generic sanity inequality (can fail only on gross breakage) | Tautology / cannot fail beyond the finiteness prelude |
|---|---|---|---|---|---|
| 1 | 33 | 0 | 0 | 29 | 4 (starspots, T_tau_gradr, simplex_solar_calibration, custom_colors) |
| 2 | 17 | 0 | 0 | 16 | 1 (wd_diffusion extra clause; rest of it is sanity) |
| 3 | 22 | 0 | 9 (hot_cool_wind, 15M_dynamo, magnetic_braking, 4x core-collapse, tzo, twin_studies) | 4 (high_mass, ppisn, split_burn_big_net, accreted_material_j) | 9 (20M_z2m2_high_rotation, high_rot_darkening, conserve_angular_momentum, ccsn_IIp, pisn, hse_riemann, ns_h, ns_he, ns_c) |
| 4 | 15 | 0 | 15 (6 gyre @50% vs 3%; 9 rsp @15%/LINA/`true` vs 1e-2 d) | 0 | 0 |
| 5 | 4 | 0 | 0 | 2 (timing, relax) | 2 (check_redo self-parity, other_physics_hooks on literals) |
| total | 91 | 0 | 24 | 51 | 16 |

  Upstream's criterion for all 91 is a specific output string in star/test_suite/do1_test_source; FlexMESA requires none of them (T2-1), and under the 2-step cap none of the underlying stopping conditions or N-period checks is reachable. `rsp_check_2nd_crossing` terminates in `extras_start_step` before any step when rsp_flag is set and falls to the last-line `dnu > 0 && numax > 0` scaling-relation check.
- Proposed fix: n/a (summary).

### T2-8: "0 ULP photo restart parity" is an in-process FlexMESA-vs-FlexMESA clone comparison over 1 step in FlexMESA's own byte format
- Area: flexmesa-star / inlist_runner, photo_io
- Severity: unverified claim
- FlexMESA: inlist_runner.rs:355-391 (`let mut continued = model.clone(); ... let photo_bytes = write_photo_bytes(&continued); let mut restarted = read_photo_bytes(&photo_bytes, continued.controls.clone(), continued.job.clone())?; for _ in 0..n_verify_steps {...}`), :497-499 (`if verify_restart && !photo_restart_verified { let (c1, c2) = verify_photo_restart_parity(&model, 1)?;`), photo_io.rs:44-54 (FNV-1a over bytes; docstring says "`./ck` in upstream MESA `test_suite_helpers`").
- Upstream: star/test_suite/each_test_run:186-201, :269 (`md5sum "$3" > checks.md5`: md5 of the final `.mod` from the full run, compared after `./re <photo>` in a fresh process run to completion); star/private/photo_out.f90:47-51 (Fortran unformatted stream).
- Confidence: verified-by-reading-both-sides
- What: What is hashed: FlexMESA's own serialization (sections 1-7: nz/species/flags, lnd/lnT/lnr/L/v/u/dq/xa/omega/j_rot, *_old and *_start arrays, ~30 scalars, a few aux arrays, eos/kap config, RSP state). `controls` and `job` are not serialized at all; the "restarted" model is handed the live clones, so any state living there is preserved trivially. Both models run in the same process sharing all global table caches. Verification is 1 step, once per case (after the first executed stage only; later stages are not checked). It shows `read(write(m))` reproduces the fields the next step reads; it is not upstream's test (fresh process, run to the end, compare final model), the format is not upstream's (no interop with a MESA photo), and it says nothing about agreement with upstream numbers. FNV-1a is not what upstream `./ck` does (md5sum -c).
- Proposed fix: rename the claim ("serialize/deserialize round trip is lossless for 1 step"); to mirror upstream, restart in a fresh process from a file and compare final `.mod` after an uncapped run. M.

### T2-9: Binary E2E (8 cases): per-step oracles and tolerances match upstream, but the default (capped) pass drops every end-state requirement; 7 of 8 uncapped runs are `#[ignore]`, and double_bh uncapped is ignored because it does not pass
- Area: flexmesa-binary / e2e_extras, tests/tier5_coupled_suite.rs
- Severity: unverified claim
- FlexMESA: crates/flexmesa-binary/src/e2e_extras.rs: every `evaluate_pass` ends `&& (!uncapped || (term_code == ... ))`; `EvolveBothStarsExtras::evaluate_pass` is `!uncapped || (term_code == "accretor_overflow_terminate" && bin.rl_relative_gap[1] >= 0.0)` with `impl BinaryExtras for EvolveBothStarsExtras {}` (no per-step check), so the capped run returns `true` unconditionally; tests/tier5_coupled_suite.rs:164 `run_binary_suite(&mesa_dir, Some(4))`; `#[ignore = "slow uncapped campaign (~121s) verified locally via --ignored"]` (x6, 121 s .. 1376 s); `#[ignore = "tracked in #139: core-He implicit rotational-wind outer retry parity"] fn test_uncapped_double_bh_campaign`.
- Upstream: binary/test_suite/do1_test_source (e.g. `do_one double_bh "Terminate due to helium depletion"`, `do_one wind_fed_bhhmxb "Properly tested sub and super Eddington phases"`); jdot_gr_check/src/run_binary_extras.f90:116-118,164 (`abs(b% xtra(ix_separation_error)) > 1d-2`); jdot_ml_check/src/run_binary_extras.f90:173 (`> 1d-4`); wind_fed_bhhmxb/src/run_binary_extras.f90:201,214,221 (`1e-15`, `1e-10`, `1e-10`); double_bh/src/run_binary_extras.f90:172-252 (0.005, 0.320819224, 0.2, 1d-3, 1d-5).
- Confidence: verified-by-reading-both-sides
- What: MATCH: the GR separation formula and 1e-2 bound, the Tauris & van den Heuvel separation and 1e-4 bound, wind-fed accounting bounds 1e-15/1e-10, and double_bh thresholds are the upstream ones, evaluated every step on live state; uncapped mode requires the upstream termination. This is the only E2E family whose criteria are upstream's. BUT the test that runs in CI is 4 coupled steps; in 4 steps of a detached/near-ZAMS system the GR/ML oracles are near-trivially satisfied (little or no orbital evolution; inferred), and star_plus_point_mass(+explicit) only check `mdot_acc <= mdot_edd`. `jdot_ls_check` is the single uncapped binary case that runs un-ignored. Upstream's jdot_ls_check has no numeric check at all (extras are empty; pass = `stop because star_age >= max_age`), so FlexMESA's J-conservation 1e-12 and sync 1e-2 criteria are FlexMESA additions (stricter). double_bh full run is an acknowledged failure (#139); the capped double_bh checks only `m1,m2 > 20` and homogeneity for 4/35 steps. All data-dependent tests `return` (silent pass) when `MESA_DIR` is unset (tier5_coupled_suite.rs `resolve_suite_dir`), except the ignored ones which `expect`.
- Proposed fix: report capped vs uncapped separately; "8/8 binary green" should read "1/8 uncapped in CI, 6/8 claimed locally, 1/8 failing". Run the ignored campaigns in a nightly job and archive logs. S.

### T2-10: Astero E2E (8 cases): uncapped criteria mirror upstream (surface_effects is faithful and stricter); capped criteria are smoke only; 5 of 8 uncapped are `#[ignore]`
- Area: flexmesa-astero / e2e.rs, tests/astero_suite.rs
- Severity: unverified claim (for the 5 ignored) / none for the matches
- FlexMESA: crates/flexmesa-astero/src/e2e.rs `run_pulse_tams_case`: uncapped `h1_c <= 0.0015 && modes.iter().any(|m| m.n_p == 4 && ((m.freq_uhz - target_uhz) / target_uhz).abs() < 0.03)`, capped `!modes.is_empty() && all finite > 0`; `run_astero_search_case`: capped `samples.len() <= 2 && chi2 finite > 0`, uncapped `"example_astero" => samples.len() == 1 && b.chi2 > 1.0 && b.chi2 < 15.0`, `"fast_scan_grid" => samples.len() == 4`, `"fast_simplex" => 1..=5`, `"fast_newuoa" => len >= 4 && b.chi2 < samples[0].chi2`, `"fast_from_file" => len == 2`; `verify_surface_effects_assertions`: 8 targets, `perfect_tol = 1.0e-12`, `solar_tol = 10.0`. tests/astero_suite.rs:244 (adipls+gyre uncapped, not ignored), :251-258 (`Some(8)` all eight), :277-303 (5 ignored, 70..780 s).
- Upstream: astero/test_suite/do1_test_source; surface_effects/src/run_star_extras.f90:35-36 (`perfect_tol = 1d-12`, `solar_tol = 1d1`), :200-245, :256-279 (solar fits other than a_div_r are `ignore = check_okay(...)`); astero_gyre/src/run_star_extras.f90:224 (`abs(actual_freq - expected_freq) < expected_freq*3d-2`), inlist_astero_gyre:30 (`x_ctrl(1) = 375d0`); example_astero/src/run_star_extras.f90:211 (`best_chi2 < 15 .and. best_chi2 > 1`).
- Confidence: verified-by-reading-both-sides
- What: MATCH: surface_effects (runs 0 evolution steps, so capping is irrelevant) uses upstream's targets and tolerances and additionally enforces the solar fits upstream ignores; astero_gyre/adipls uncapped use upstream's 3% on the n=4 radial mode at 375 uHz and run un-ignored; example_astero's chi2 window is upstream's. The fast_* cases upstream only check that a results file is written; FlexMESA checks sample counts. So 3 of 8 astero cases (surface_effects, astero_gyre, astero_adipls) are genuinely tested against upstream criteria in the default test run. For the other 5 the default run is the 8-step smoke (chi2 finite, <= 2 samples). Note upstream astero_gyre itself prints 'not using gyre: pretend got ok match' when GYRE is absent (run_star_extras.f90:139). The surface-effects "targets" are upstream's hard-coded regression numbers at 1e-12 only for the synthetic-recovery half, which is an analytic round trip; against upstream's actual model the tolerance is 10 (1000%). Missing data => tests fail (assert on dir exists) : good.
- Proposed fix: un-ignore example_astero (70 s) at least; nightly for the rest. S.

### T2-11: History of loosening (git log -p over all Rust test and e2e/extras files, 82 commits)
- Area: all crates
- Severity: unverified claim
- Confidence: verified-by-reading-flexmesa-only (git history) + upstream lines cited
- What: Method: scripted diff of tolerance literals / `#[ignore]` / `Some(N)` caps / high-precision expected literals for every commit touching `crates/*/tests/*.rs`, `tests*.rs`, `*extras*.rs`, `e2e*.rs`, `inlist_runner.rs`; then manual reading of hits. Results:

| Commit (date) | File | Old -> New | Stated reason in commit message |
|---|---|---|---|
| c947885 (11-10) | crates/flexmesa-star/src/pulse_modes.rs `eval_target_mode_frequency_and_growth` | `(mode.freq_hz - freq_target_hz).abs() <= 0.03 * freq_target_hz` (= upstream gyre_in_mesa_extras_finish_step.inc:115) -> `<= 0.50 * freq_target_hz.max(mode.freq_hz)` | none (message says the BVP solver "match[es] all 34 modes of gyre/test/test_output") |
| bd33084 (11-11) | same | adds second accept branch: Teff-rescaled frequency within `0.35 * freq_target_hz` | none |
| 322fb70 (11-14) | same | adds third accept branch (ratio to acoustic large separation in `0.75n ..= 3n+2`) and self-target when x_ctrl(1) unset | none |
| d4b7a1e (11-12) | crates/flexmesa-star/src/tier4_extras.rs `verify_tier4_case_assertions` | `return (lina[0].period_days - p_tgt).abs() <= 1.0e-2;` (upstream rsp_Cepheid run_star_extras.f90:115 tolerance, applied to the LINA fundamental) -> `lina.iter().any(|m| (m.period_days - p_tgt).abs() / p_tgt <= 0.15)` over 3 modes; new nonlinear path `<= 0.15` relative / `return true`; energy bound introduced at `1.0e-3` (upstream 1d-5) | none |
| 76dc37d (11-14) | crates/flexmesa-binary/tests/tier5_coupled_suite.rs | +7 `#[ignore]` on uncapped binary campaigns ("slow uncapped campaign (~121s..1376s) verified locally via --ignored") | runtime |
| 642da54 (11-15) | same | double_bh ignore reason "slow multi-hundred-step campaign" -> "tracked in #139: core-He implicit rotational-wind outer retry parity" (i.e. now acknowledged failing) | #139 |
| e76df33 | crates/flexmesa-astero/tests/astero_suite.rs | +5 `#[ignore]` on uncapped astero searches (70..780 s) | runtime |
| 0c9b315 | crates/flexmesa-star/src/tests_wave5.rs | removed `run_astero_suite(base, true, Some(2), true)` from the star test (moved to astero crate at Some(8)) | refactor |
| b684358 -> 642da54 (9 commits) | crates/flexmesa-bench/tests/bench_tests.rs `EXPECTED_WAVE4_CHECKSUMS_N8` | `star_evolve_step`: 9.931842248453043e1 -> ...425352e1 (13eceea) -> 9.9871430838695e1 (51e1adc) -> 9.931845982528495e1 (d5a1172) -> 9.9871430838695e1 (bd33084) -> ...921163e1 (7b2f388) -> 9.987143125142524e1 (6a78100) -> 1.005913169168671e2 (642da54). `star_hydro_eqns`: 1.411154427772951e6 -> 4.660073228335436e6 (51e1adc) -> 4.253224624211422e6 (6a78100). `w5_riemann_rsp2`: -2.543418211343195e5 -> ...346364e5 (d4b7a1e) | each is a feature commit; pins re-recorded alongside |

  Not found: no commit relaxes a numeric tolerance in the module-level golden tests (eos/kap/net/rates/neu/atm/turb/num/mtx/interp/math); the eos Skye/PC expected arrays were reformatted (75fb3d4, e9f8f2c delete 285/171 literal lines) but my literal-set diff shows no value was changed, only moved/condensed (wave3a3_pc_tests.rs lost 2 of 3 tests to a new file wave3a3_pc_solid_tests.rs). The Tier 1-3/5 star assertions were never loosened because they were written in their current form in the commit that introduced them (923c362, cdb7809, 13eceea, 3f0d15f); the 2-step cap was there from the first commit of each tier test (923c362 `Some(2)`, 13eceea `run_tier3_suite(p, Some(2), true)`, cb77f71, 3f0d15f).
- Proposed fix: restore 0.03 and 1e-2/1e-5; S once the solver actually meets them.

### T2-12: flexmesa-bench "pinned cross-language reference checksums" are FlexMESA's own output, re-pinned whenever the code changes
- Area: flexmesa-bench/tests/bench_tests.rs
- Severity: unverified claim
- FlexMESA: crates/flexmesa-bench/tests/bench_tests.rs:17-76 (six `EXPECTED_*_CHECKSUMS_N8` tables, doc: "Pinned cross-language reference checksums ... shared with tests/test_wave4_bench.py and benchmarks/fortran_mesa/run_wave4_bench.f90"), :103 `rel_err = (rec.checksum - exp_sum).abs() / exp_sum.abs()` at 1e-12 (1e-10 for wave 4).
- Upstream: n/a (benchmarks/fortran_mesa/bench_kernels_wave4.f90:260-478 computes `rec%checksum = sum(out_stage(...))` from real star_lib calls, but nothing in the Rust test reads Fortran output).
- Confidence: verified-by-reading-flexmesa-only
- What: The Rust test compares Rust Stage 0-3 output with a literal. For the wave-4/5 star workloads the literal changed 7 times (`star_evolve_step`, 99.318 -> 100.591, a 1.3% drift) and `star_hydro_eqns` by a factor 3.3 (see T2-11). A reference derived from upstream Fortran would be constant across FlexMESA commits, so these are category (d) self-pins: the test is a regression/cross-backend-consistency check, not a parity check. Whether the wave 1-3 pins (unchanged since introduction) equal real Fortran output is not established by the Rust test either (Lead L1).
- Proposed fix: have the test parse a committed Fortran harness output file (with the upstream commit hash) instead of literals. M.

### T2-13: Rust-side E2E coverage is far smaller than "107 cases": Tier 1 runs 1 of 33 and Tier 2 runs 1 of 17 in `cargo test`; all at a 2-step cap
- Area: flexmesa-star / tests_wave5.rs, tier3_extras.rs
- Severity: unverified claim
- FlexMESA: crates/flexmesa-star/src/tests_wave5.rs:268-277 (`let case_dir = base.join("star").join("test_suite").join("make_zams"); ... run_test_suite_case(&case_dir, true, Some(2), true)`), :321-334 (`cburn_inward`, `Some(2)`), :386 `run_tier4_star_suite(base, true, Some(2), true)`, :407 `run_dev_cases_suite(base, true, Some(2), true)`, :485 `run_tier5_star_suite(base, Some(2), true)`; tier3_extras.rs inline test `run_tier3_suite(p, Some(2), true)`. Python side (context only): tests/test_e2e_1_tier1.py:289,318 / test_e2e_2_tier2.py:256,292 / test_e2e_3_tier3.py:187,211 / test_e2e_4_tier4.py / test_e2e_5_tier5.py:111 all pass `max_steps_per_stage=2` (one at 3); binary `max_steps_cap=4`.
- Upstream: star/test_suite/do1_test_source (91 cases, each run to a physical stopping condition).
- Confidence: verified-by-reading-flexmesa-only
- What: In Rust, the star-tier runs are: tier1 = make_zams only, tier2 = cburn_inward only, tier3 = 22, tier4 = 16 star + 32 dev, tier5 = 4, every one with `Some(2)`. The full 33/17 lists run only from Python, also at 2 steps. No test anywhere in the Rust tree runs a star test_suite case uncapped. 107 = 91 star + 8 astero + 8 binary; of these, the cases whose default-run pass criterion is upstream's are 3 astero (surface_effects, astero_gyre, astero_adipls) and arguably 1 binary (jdot_ls_check uncapped, whose upstream criterion is just reaching max_age).
- Proposed fix: publish a per-case table {uncapped? upstream string reached? criterion}. S.

### T2-14: Silent-skip and silent-substitute paths: tests and runner pass when inputs are missing
- Area: several
- Severity: unverified claim / maintainability
- FlexMESA:
  - crates/flexmesa-star/src/tests_wave5.rs:268, :378, :478 and tier3_extras.rs inline test: E2E block is inside `if let Ok(mesa_dir) = std::env::var("MESA_DIR") { if ...is_dir() { ... } }`; without MESA_DIR the test passes having run no case (guarded only by opt-in `FLEXMESA_REQUIRE_ZERO_SKIPS=1`).
  - tests_wave5.rs:321-334 (tier 2): `if case_dir.is_dir() { ... }` with no zero-skip guard at all.
  - crates/flexmesa-binary/tests/tier5_coupled_suite.rs `resolve_suite_dir() -> Option` + `let Some(..) = resolve_suite_dir() else { return; };` in 5 tests; never consults FLEXMESA_REQUIRE_ZERO_SKIPS.
  - crates/flexmesa-star/tests/rsp_crossing_test.rs:258-264 (`return` if history file absent unless zero-skips).
  - Reduced-subset fallbacks (not skips): neu_tests.rs (6 inlined rows instead of 10,000), autodiff golden_suite_tests.rs:439 (pinned analytic subset), mtx block_tests.rs:405-491, chem lib.rs test, utils_tests.rs:36-49, kap_golden_test.rs:339-345 (`if tables.fixed_sets.len() < 8 { ...; return; }`).
  - Runner: inlist_runner.rs:463-466 (`if InlistBundle::resolve_inlist_path(case_dir, inlist).is_none() { continue; }`: a stage named in `./rn` whose inlist is absent is skipped, not failed); :443 (`if src.exists() { fs::copy }`: a missing checkpoint for a `cp` stage is ignored); star_job_init.rs:117-136 `load_mod_or_fallback` (`if let Ok(m) = read_mod_str(...) { return Ok(m) } ... build_initial_planet_or_wd_model(controls, job)`: an unreadable/unparseable saved model named by the inlist is silently replaced with a synthesized model); tier5_extras.rs `_ => true`.
  - `debug_assert!(check_other_physics_hooks(model))` (tier5_extras.rs extras_finish_step) and `debug_assert!(!hyd.is_empty() && !abn.is_empty())` (tier3_extras.rs extras_after_evolve) are compiled out: CI runs `cargo test --workspace --exclude flexmesa-py --release` (.github/workflows/ci.yml:66, scripts/verify_ci.sh:68).
- Upstream: n/a
- Confidence: verified-by-reading-flexmesa-only
- What: CI does export `MESA_DIR` and `FLEXMESA_REQUIRE_ZERO_SKIPS: "1"` (ci.yml:25,57), so in CI the env-gated blocks run; a developer running plain `cargo test` gets green with zero E2E cases executed. The runner-level fallbacks are worse because they apply in CI too: a case can "pass" on a substitute starting model or with a stage dropped, and nothing in `CaseRunReport` records it. `#[ignore]` inventory (18 total): 6 bench table generators (bench_tests.rs "benchmark table generator for --release runs"), 7 binary uncapped (T2-9), 5 astero uncapped (T2-10); CI never passes `--ignored`.
- Proposed fix: make missing inputs hard errors in the runner; one shared `require_inputs()` helper for tests. S-M.

### T2-15: EOS HELM/PC/Skye "Fortran oracle" tables (about 1,060 17-digit literals) have no traceable origin
- Area: flexmesa-eos tests
- Severity: unverified claim
- FlexMESA: crates/flexmesa-eos/tests/wave3a2_helm_regimes_tests.rs:54-56 (`fn test_helm_eos_degenerate_and_pair_regimes_match_fortran_oracle() { ... let expected_comp: [[f64; 5]; 16] = [[4.2376368450237116e1, ...`), :44 (`rel_err < 1.0e-14`), wave3a2_helm_tests.rs, wave3a3_pc_tests.rs:46-49, wave3a3_pc_solid_tests.rs, wave3a3_skye_inverse_tests.rs.
- Upstream: eos/test/test_output (474 lines, 220 numbers with >= 6 significant digits).
- Confidence: verified-by-reading-both-sides (numeric cross-match), origin inferred
- What: Literal cross-match of every >= 10-significant-digit constant in these five files against upstream eos/test/test_output: 158/1, 227/7, 186/4, 179/4, 312/5 (literals / found upstream). The remaining ~1,040 values (including `d/dabar`, `d/dzbar` columns that upstream's eos test never prints) appear nowhere else in the repository (grep of the whole tree), no generator program or captured Fortran output is committed, and the commit messages (403986a, c97ebe3) only say "Add golden Fortran MESA parity tests". They may be real output from a locally built driver; that cannot be checked from the repo. My literal-set diff across history shows these values were not re-pinned after introduction (only reformatted in 75fb3d4/e9f8f2c), which is mild evidence against self-pinning to a moving target. Classified (d) unknown origin. By contrast wave3b1_golden_tests.rs is real: it parses an embedded byte-identical copy of upstream eos/test/test_output (16 points x 11 quantities at 1e-12, 5e-11 for the 4th point of each composition).
- Proposed fix: commit the Fortran driver and its raw output next to the test and parse it. M.

## Deliverable 2(4): module-level "golden test_output" tests

First fact (verified with `cmp`): every `<module>/test/test_output` in the bundle `the test-input bundle` is byte-identical to `<module>/test/test_output` for num, net, chem, auto_diff, ionization, kap, rates, turb, atm, interp_2d, interp_1d, eos, neu, const, mtx, utils, colors, math (.CRMATH). The committed copies `tests/fixtures/golden/{eos,kap,net,rates,atm}/test_output.txt`, `gyre/test_output`, `gyre/model.dat`, `adipls/{Gemma.fgong,obs.test.txt}`, `stella/mesa.{abn,hyd,lbol}` and the 7 `rsp/*_cycle0.mod` + `rsp/list` are byte-identical to upstream files. So the references are real.

"Coverage" below = upstream numbers (>= 6 sig. digits) in the module's test_output that the Rust test actually compares (counted by a script matching test-source literals to upstream values at 1e-9, or by reading the parse loop for file-reading tests).

| Crate | How the reference is obtained | Coverage of upstream test_output | Tolerance | Verdict |
|---|---|---|---|---|
| const | literals transcribed into const_tests.rs | 21/21 | rtol 1e-15 | real, complete |
| math | literals from test_output.INTRINSIC | 85/98 | 1e-13..1e-15 | real, near-complete |
| utils | reads upstream files at run time if present | token/output files | exact string | real (subset fallback offline) |
| auto_diff | golden_suite_tests.rs parses the real file at run time | 209 numeric + 42 boolean records (all) | 1e-14 | real, complete (analytic subset offline) |
| interp_1d | literals | 30/99 | rtol 1e-12 | real, ~30% |
| interp_2d | literals | 31/65 | 1e-10..1e-12 (5e-4 for upstream's 32-bit rows) | real, ~50% |
| mtx | literals (mtx_tests 9, block_tests 4) + run-time read of test_output/test_case.data | 13/13 | 1e-13..1e-14 | real, complete |
| num | literals | ~27/341 (root finders, simp38, 23 ODE endpoint values) | 1e-6..1e-14; ODE endpoints 1e-4..1e-7 | real, ~8% (solver statistics lines not compared) |
| chem | lib.rs inline test parses the real file | 286 Lodders rows @5e-7 (print precision) + 51 do_tests blocks @1e-14 | as stated | real, near-complete |
| colors | literals in lib.rs | 37/161 | 1e-12 (one 5e-3) | real, ~23% |
| ionization | literals in lib.rs | 31/35 | per-test | real, near-complete |
| eos | wave3b1: embedded identical copy parsed; wave3a1: 4-5 literals; wave3a2/3a3: unknown-origin tables (T2-15) | 176 + ~26 of 220 | 1e-12 / 5e-11 | real for wave3b1; unverifiable for wave3a2/3a3 |
| kap | literals | 12 outputs of 39 numbers (3 of upstream's 3 tests: log10kap, 2 partials, kap) | 1e-11..1e-12 (AESOPUS kap 2e-7) | real; dkap_dlnd/dlnT not compared |
| rates | `include_str!` of identical copy | 9 screening + 282x3 weak rows @5e-6 + 9x307 raw rates @1e-11 + ecapture/pycno | 1e-11..1e-13 | real, near-complete |
| net | literals | 56/82 | eps_nuc 1e-8, d(eps) 1e-5..1e-8, one-zone burn 1e-5..1e-9 | real, ~70% |
| neu | reads real file at run time | 10,000 rows x 34 cols (all) | 1e-10 (6 inlined rows 1e-12) | real, complete |
| atm | literals | 38/45 | 1e-8 (print precision) .. 1e-14 | real, ~85% |
| turb | literals | 72/72 | 1e-12 (2e-11 for 2 rows) | real, complete |
| star (gyre/adipls/stella) | reads identical upstream fixtures | GYRE 34 modes; ADIPLS 46 modes | GYRE freq 5e-4, E_norm 5e-2; ADIPLS freq 1.6e-3 (l=0) / 1.25e-2, inertia 0.35 dex | real reference, loose tolerance |
| star (RSP) | `tests/fixtures/golden/rsp/*_{init,step1}_r24.mod` (14 files, not in upstream tree; header `version_number '0b2e6bb'`, consistent with output of a real upstream build) | 7 models: STAH build + LINA, 1 HYD step | 1e-6..1e-8 on profiles; LINA period 1e-6, growth 5e-6 | probably real compiled-upstream output; generator not committed |
| binary (lib.rs) | 4 literals from binary/test/test_output | 4/4 | 1e-12..1e-13 | real, complete |
| astero | upstream surface_effects targets | 8 targets | 1e-12 synthetic | real |

Answer: these tests do use real upstream numbers (as transcribed literals, embedded identical copies, or run-time reads), at tight tolerances, and I found no loosening in their history. Coverage ranges from 8% (num) to 100%. They validate individual kernels at upstream's hand-picked test points; they say nothing about the coupled star solver. The one exception in provenance is eos wave3a2/3a3 (T2-15) and the bench checksums (T2-12).

## Deliverable 1: test-provenance table

Source classes: (a) real upstream output, (b) analytic identity / finite difference, (c) other FlexMESA backend, (d) self-pinned / unknown origin, (e) smoke. "Missing data": F = fails, S = silently skips or shrinks, n/a = needs no external data.

| File | #tests | #ignore | Source | Tolerances | Step cap | Missing data | Pass => upstream parity? |
|---|---|---|---|---|---|---|---|
| astero/tests/astero_suite.rs | 9 | 5 (uncapped example_astero, fast_from_file, fast_scan_grid, fast_simplex, fast_newuoa: "slow uncapped campaign (~70..780s) verified locally") | (a) upstream criteria for surface_effects/gyre/adipls; (e) for 8-step run | 1e-12 synthetic; 3% freq; solar_tol 10 | 8 (bounded test), 2 (mock), none (adipls, gyre) | F | yes for 3 cases vs upstream's own criteria; no for 5 |
| astero/src/lib.rs (inline) | 3 | 0 | (a)+(b) | 1e-12..1e-5 | n/a | n/a | kernel-level |
| atm/tests/atm_golden_test.rs | 4 | 0 | (a) 38/45 + (b) FD | 1e-8..1e-14; FD 1e-6 | n/a | uses embedded/loaded tables | yes (kernel) |
| autodiff/tests/autodiff_tests.rs | 5 | 0 | (b) | 1e-14..1e-4 | n/a | n/a | n/a |
| autodiff/tests/golden_suite_tests.rs | 2 | 0 | (a) full file | 1e-14 | n/a | S (analytic subset) unless MESA_DIR/zero-skips | yes |
| autodiff/tests/order3_tests.rs | 3 | 0 | (b) | 1e-14..1e-15 | n/a | n/a | n/a |
| bench/tests/bench_tests.rs | 12 | 6 ("benchmark table generator for --release runs") | (d) self-pinned checksums + (c) stage cross-check | 1e-12 (1e-10 wave 4) | n/a | n/a | no (T2-12) |
| binary/tests/tier5_coupled_suite.rs | 17 | 7 (6 "slow uncapped campaign (~121..1376s)", 1 "tracked in #139") | (a) upstream oracles/tolerances; (b) | 1e-2, 1e-4, 1e-12, 1e-15 | 4 (all 8), 35 (double_bh), none (jdot_ls_check) | S (`return`) | partially (T2-9) |
| binary/tests/phase5c_lifecycle_extras.rs | 5 | 0 | (b)/(e) invariants | 1e-5..1e-14 | few steps | n/a | no |
| binary/src/lib.rs (inline) | 5 | 0 | (a) 4 literals + (b) | 1e-12..1e-15 | 2-4 | S | kernel-level |
| chem/src/lib.rs (inline) | 6 | 0 | (a) reads real file | 5e-7 / 1e-14 | n/a | S (literals only) unless MESA_DIR | yes |
| colors/src/lib.rs (inline) | 4 | 0 | (a) 37/161 | 1e-12 | n/a | n/a | yes (subset) |
| const/tests/const_tests.rs | 2 | 0 | (a) | 1e-15 | n/a | n/a | yes |
| data/tests/data_tests.rs | 4 | 0 | (b) SHA vectors, round trips | exact/1e-12 | n/a | n/a | n/a |
| eos/tests/eos_dxa_partials_test.rs | 2 | 0 | (b) FD | 5e-6..2e-5 | n/a | n/a | no |
| eos/tests/wave3a1_golden_tests.rs | 5 | 0 | (a) 5 blend lines + (b) | 1e-14..1e-15 | n/a | n/a | partly |
| eos/tests/wave3a2_helm_tests.rs | 2 | 0 | (d) unknown-origin "Fortran oracle" (227 literals, 7 upstream) | 1e-14 | n/a | n/a | unverifiable |
| eos/tests/wave3a2_helm_regimes_tests.rs | 2 | 0 | (d) (158 literals, 1 upstream) | 1e-14..1e-15 | n/a | n/a | unverifiable |
| eos/tests/wave3a3_pc_tests.rs | 1 | 0 | (d) (179, 4) | 1e-11 values, 5e-9 partials | n/a | n/a | unverifiable |
| eos/tests/wave3a3_pc_solid_tests.rs | 2 | 0 | (d) (186, 4) | 1e-11..1e-14 | n/a | n/a | unverifiable |
| eos/tests/wave3a3_skye_inverse_tests.rs | 3 | 0 | (d) (312, 5) | 1e-8..1e-13 | n/a | n/a | unverifiable |
| eos/tests/wave3b1_full_tables_tests.rs | 2 | 0 | (e) table shape | exact | n/a | depends on data dir | no |
| eos/tests/wave3b1_golden_tests.rs | 4 | 0 | (a) embedded identical copy | 1e-12 / 5e-11; inversions 1e-5 | n/a | n/a | yes |
| interp1d/tests/interp1d_tests.rs | 7 | 0 | (a) 30/99 + (b) proptest | 1e-12 | n/a | n/a | yes (subset) |
| interp2d/tests/interp2d_tests.rs | 5 | 0 | (a) 31/65 + (b) | 1e-10..1e-15 | n/a | n/a | yes (subset) |
| ionization/src/lib.rs (inline) | 4 | 0 | (a) 31/35 | tight | n/a | n/a | yes |
| kap/tests/kap_golden_test.rs | 6 | 0 | (a) 12 values + (b) FD + (e) table counts | 1e-11..1e-12; FD 1e-6 | n/a | S (full-table test shrinks, :339) | yes (3 points) |
| math/tests/math_tests.rs | 6 | 0 | (a) + (b) | 1e-13..1e-15 | n/a | n/a | yes |
| mtx/tests/block_tests.rs | 5 | 0 | (a) run-time read + (b) | 1e-9..1e-14 | n/a | S | yes |
| mtx/tests/eigen_test.rs | 2 | 0 | (b) | 1e-12..1e-14 | n/a | n/a | n/a |
| mtx/tests/mtx_tests.rs | 6 | 0 | (a) 9/13 | 1e-13..1e-14 | n/a | n/a | yes |
| net/tests/net_golden_test.rs | 5 | 0 | (a) 56/82 | 1e-5..1e-12 | n/a | n/a | yes |
| net/tests/net_jacobian_test.rs | 2 | 0 | (b) FD | 1e-4..1e-7 | n/a | n/a | no |
| net/src/burn_const_density.rs (inline) | 1 | 0 | (b)/(e) | - | n/a | n/a | no |
| neu/tests/neu_tests.rs | 6 | 0 | (a) all 10,000 rows | 1e-10 | n/a | S (6 rows) unless MESA_DIR | yes |
| num/tests/num_tests.rs | 14 | 0 | (a) few + (b) | 1e-6..1e-15 | n/a | n/a | partly |
| num/tests/ode_benchmarks_tests.rs | 5 | 0 | (a) 23 endpoint values + published reference solutions | 1e-4..1e-7 | n/a | n/a | partly |
| num/tests/ode_newton_tests.rs | 6 | 0 | (a) 5 + (b) | 1e-3..1e-13 | n/a | n/a | partly |
| rates/tests/rates_golden_test.rs | 5 | 0 | (a) embedded identical copy | 1e-11..1e-13 (5e-6 weak table) | n/a | n/a | yes |
| rates/src/lib.rs (inline) | 1 | 0 | (e) | - | n/a | n/a | no |
| schema/tests/schema_tests.rs | 3 | 0 | (e) parser | exact | n/a | n/a | n/a |
| star/src/tests.rs | 10 | 0 | (b) FD Jacobian, invariants, round trip | 1e-6..1e-14 | 1-few | n/a | no |
| star/src/tests_fidelity1.rs | 6 | 0 | (b) hand-computed formulas "match_fortran" (formula re-derivation, not Fortran output) | 1e-12 | n/a | n/a | no (formula-level) |
| star/src/tests_fidelity2.rs | 5 | 0 | (b) | 1e-6..1e-12 | n/a | n/a | no |
| star/src/tests_fidelity4.rs | 4 | 0 | (b)/(e) round trip, invariants | 1e-2..1e-12 | few | n/a | no |
| star/src/tests_phase3_pulse_stella.rs | 4 | 0 | (a) gyre/adipls/stella upstream fixtures | 5e-4 / 5e-2 / 1.6e-3 / 1.25e-2 / 0.35 dex | n/a | F (committed fixtures) | yes at stated (loose) tolerance |
| star/src/tests_wave5.rs | 7 | 0 | (e) E2E smoke via tier assertions; (b) helpers | generic inequalities | 2 per stage (10 in one synthetic) | S unless MESA_DIR (tier 2: S always) | no (T2-2..T2-8) |
| star/src/tier3_extras.rs (inline) | 1 | 0 | (e) | inequalities | 2 | S unless MESA_DIR | no |
| star/src/zams.rs (inline) | 1 | 0 | (e) catalog load | - | n/a | - | no |
| star/tests/phase5a_hydro_solver_test.rs | 4 | 0 | (b) AD Jacobian vs central FD | 1e-3..1e-9 | n/a | n/a | no |
| star/tests/phase5a_solver_norms_and_burn_test.rs | 4 | 0 | (b)/(e) convergence, photo round trip | 1e-3..1e-12 | 1 | n/a | no |
| star/tests/phase5b_fidelity2_test.rs | 9 | 0 | (b) analytic wind laws, conservation | 1e-6..1e-15 | few | n/a | no |
| star/tests/rsp2_test.rs | 5 | 0 | (b) hand oracles + 1 upstream cycle0 fixture | 1e-4..1e-15 | n/a | F | partly |
| star/tests/rsp_build_lina_test.rs | 1 | 0 | (a*) 7 `_init_r24.mod` | 1e-8..1e-13 | n/a | F | yes if fixtures are genuine upstream output |
| star/tests/rsp_hyd_step_test.rs | 3 | 0 | (a*) 7 `_step1_r24.mod` | 1e-6..1e-7 | 1 step | F | yes for one HYD step, same caveat |
| star/tests/rsp_crossing_test.rs | 2 | 0 | (a) upstream `list` (v11461, 3e-3/5e-4) + 13 hard-coded "r24" LINA values (1e-6/5e-6; origin not in repo) | as stated; edges +-5 K; 7M case window 4300-5600 K | n/a | S (7M case) unless zero-skips | partly |
| star/tests/rsp_io_eos_test.rs | 5 | 0 | (a) cycle0 fixtures round trip + (b) | 1e-10..1e-15 | n/a | F | I/O only |
| testing/tests/golden_tests.rs | 3 | 0 | (b) | 1e-12..1e-15 | n/a | n/a | n/a |
| turb/tests/turb_tests.rs | 3 | 0 | (a) 72/72 | 1e-12 (2e-11) | n/a | n/a | yes |
| turb/tests/turb_autodiff_and_mixing_tests.rs | 4 | 0 | (b) FD | 1e-4..1e-13 | n/a | n/a | no |
| utils/tests/utils_tests.rs | 5 | 0 | (a) run-time read | exact | n/a | S | yes |

E2E machinery in library code (not test files): star tiers 1-5 see T2-1..T2-8; binary e2e T2-9; astero e2e T2-10.

## Matches (things that are what they claim)

- Bundle and committed golden files are byte-identical to upstream at 0b2e6bb (list in Deliverable 2(4)).
- `assert_values_match` (crates/flexmesa-testing/src/lib.rs:316-361) is a sound rtol+atol comparison that rejects non-finite values; all 19 call sites consume the `Result` (`.expect`/`.unwrap`), none discard it.
- Binary oracles and tolerances: e2e_extras.rs `rel > 1.0e-2` (GR) = jdot_gr_check/src/run_binary_extras.f90:164; `rel > 1.0e-4` (ML) = jdot_ml_check/src/run_binary_extras.f90:173; `frac_bh > 1.0e-15`, `err_wtr > 1.0e-10`, `err_wsys > 1.0e-10` = wind_fed_bhhmxb/src/run_binary_extras.f90:201,214,221; double_bh `0.005`, `0.320_819_224`, `0.2`, `1.0e-3`, `1.0e-5` = double_bh/src/run_binary_extras.f90:172-252; GR formula e2e_oracles.rs `beta = 64 G^3/(5 c^5) m1 m2 (m1+m2)` = jdot_gr_check/src/run_binary_extras.f90:115-117.
- Astero: `perfect_tol = 1.0e-12`, `solar_tol = 10.0` = surface_effects/src/run_star_extras.f90:35-36; 3% on n=4 radial mode = astero_gyre/src/run_star_extras.f90:224; chi2 in (1, 15) = example_astero/src/run_star_extras.f90:211.
- Tier 4 strict branch `(p_days - p_tgt).abs() <= 1.0e-2` after `x_integer_ctrl(1)` periods = rsp_Cepheid/src/run_star_extras.f90:104-115 (but unreachable at the test cap).
- TIER inventories: 33+17+22+15+4 = 91 = number of `do_one` lines in star/test_suite/do1_test_source; `rsp_Cepheid_6M` absent from both.
- Module golden tests: tight tolerances, no historical loosening, no re-pinned expected values (scripted literal-set diff over all 82 commits found re-pins only in bench_tests.rs).
- CI sets `FLEXMESA_REQUIRE_ZERO_SKIPS=1` and `MESA_DIR` (.github/workflows/ci.yml:25,57).

## Leads (not verified)

- L1: whether the wave 1-3 bench checksums in bench_tests.rs equal the output of benchmarks/fortran_mesa/run_wave{1,2,3a,3b}_bench.f90; needs a Fortran build.
- L2: origin of the eos wave3a2/3a3 expected tables and of the 14 `_r24.mod` RSP fixtures and the 13 "r24" LINA values in rsp_crossing_test.rs; the `.mod` headers carry `version_number '0b2e6bb'` which suggests a real upstream build, but no driver/log is committed.
- L3: in the 4-step binary runs the GR/ML/LS oracle errors are probably near zero because little evolution occurs; `max_relative_oracle_error` values from a run would confirm how much the capped test exercises.
- L4: whether the six "verified locally via --ignored" binary and five astero uncapped campaigns actually pass at this commit; no logs are committed, and double_bh is acknowledged not to (#139).
- L5: I did not confirm by execution that the Tier 1/2 inequality thresholds are already satisfied by the loaded starting models (inferred from their looseness and the 2-step cap).
- L6: `verify_step_redo_bit_parity` hand-restores ~10 fields and disables remesh for the redo step; whether a genuine `redo` in FlexMESA is bit-reproducible without that scaffolding is untested.
- L7: tests_fidelity1.rs test names say "match_fortran" but the expectations are in-test re-derivations of the formulas; not checked line by line against the .f90.

## Not covered

- Python tests (`tests/*.py`) other than reading their step caps.
- `termination.rs` line-by-line against upstream `do_one_utils.f90` (only used as far as showing `termination_code` is not part of `passed`).
- The 32 `dev_cases_*` criteria individually (they share `verify_tier4_case_assertions`).
- Did not run anything (per brief); all behaviour claims are from reading.

Status: complete.
