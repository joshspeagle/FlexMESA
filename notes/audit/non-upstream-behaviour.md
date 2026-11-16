# C1 (Task B): inventory of behaviour upstream MESA does not have

Tree under audit: `the audited tree` (PR #138 head `91eda0b`). Upstream: `the upstream checkout` at `0b2e6bb`.
Paths below are relative to those roots. Written incrementally; later sections appended.

## B.1 Initial-model construction and `&star_job` relaxation (physics-changing)

### C1-1: Any model outside 0.51..15.8 Msun, and every tier-3/4/5 case, starts from a rescaled 1 Msun ZAMS model, not from upstream's ZAMS/pre-MS/saved model
- Area: flexmesa-star / `star_job_init.rs`
- Severity: wrong physics
- FlexMESA: `crates/flexmesa-star/src/star_job_init.rs:166-167` (`let out_of_zams_range = controls.initial_mass > 15.8 || controls.initial_mass < 0.51;`), `:214-246` (branch `is_tier3_case(case_name) || (is_tier4_case..) || is_tier5_star_case(case_name) || out_of_zams_range` -> `build_initial_planet_or_wd_model`), `:94-112` (`controls.initial_mass = 1.0; let mut model = load_zams_model(controls, job)?; ... model.ln_r[k] += dln_r; model.lum[k] *= new_mstar / old_mstar;`)
- Upstream: `star/private/init_model.f90:65-135` `get_zams_model` interpolates the ZAMS file between bracketing masses for any mass in the file (0.08..100 Msun in `zams_z2m2_y28.data`); `star/private/create_initial_model.f90:70-245` `build_initial_model` integrates a real envelope (used only when `create_initial_model = .true.`). No mass-range or case-name switch exists.
- Confidence: verified-by-reading-flexmesa-only for the Rust; upstream structure verified by reading `init_model.f90` subroutine list and `create_initial_model.f90`.
- What: For the listed cases the "initial model" is the 1 Msun ZAMS profile with `ln r` shifted by `ln(M/Msun)/3` and `L` scaled by `M/Msun` (homology at fixed density and temperature). The priority of this branch is ABOVE `job.load_saved_model` (line 254) and `job.create_pre_ms_model` (line 270), so for a tier-3/4/5 case or any `initial_mass > 15.8` the saved `.mod` the inlist asks for is not loaded and the pre-MS builder is not used; only the header mass of the `.mod` is peeked (`peek_mod_header_mass`, `:54-69`, used at `:235-237`). The function name and doc comment claim correspondence with `create_initial_model.f90`; nothing of that file (envelope integration from `(M, R)` with `initial_model_relax_num_steps`, `center_logP_1st_try_for_create_initial_model` etc.) is implemented.
- Proposed fix: delete the branch; route by `&star_job` flags only, as upstream `do_star_job_controls_before`/`do_load1_star` do. L.

### C1-2: `equilibrate_tier2_scaled_model` rescales the gravitational constant and the opacity so that the mis-scaled model satisfies the structure equations
- Area: flexmesa-star / `relax.rs`
- Severity: wrong physics
- FlexMESA: `crates/flexmesa-star/src/relax.rs:24-38` (`model.cgrav[k] = STANDARD_CGRAV * alpha_m.powf(-2.0 / 3.0) * m_ratio;` and `model.controls.opacity_factor = alpha_m.powf(-2.0 / 3.0);`), called from `star_job_init.rs:469-475` and `:571`.
- Upstream: none. `cgrav(k)` is `standard_cgrav` unless `use_other_cgrav` (`star/private/star_utils.f90` `set_m_grav_and_grav`), `opacity_factor` is a user control (default 1, `star/defaults/controls.defaults`).
- Confidence: verified-by-reading-flexmesa-only (upstream absence: grep of `star/private` for any `cgrav` scaling by mass found none beyond `other_cgrav`).
- What: With `alpha_m = xmstar/Msun`, every zone's `G` is multiplied by `alpha_m^(-2/3) * (alpha_m * q*Msun / m_grav)`, i.e. chosen so that `G m / r^4` of the homologously stretched model equals that of the original 1 Msun ZAMS model, and the opacity is multiplied by `alpha_m^(-2/3)` so the radiative gradient is preserved. The result is a model that converges because gravity and opacity were changed, not because it is a star of that mass. `opacity_factor` is then explicitly preserved across later stages (`star_job_init.rs:317-323`). This applies whenever `is_tier2_or_3` is true (tier 2/3/4/5 cases, any `relax_initial_mass_to_remove_h_env`, any mass outside 0.51..15.8).
- Proposed fix: remove; implement real `relax_mass`/`create_initial_model`. L.

### C1-3: `relax.f90` is not ported; "relaxations" are algebraic rescalings with no evolution steps
- Area: flexmesa-star / `relax.rs`, `relax_comp.rs`
- Severity: wrong physics / unverified claim
- FlexMESA: `crates/flexmesa-star/src/relax.rs:280-319` (`relax_mass_to_target`: `model.ln_r[k] += dln_r; if scale_lum { model.lum[k] *= xm_ratio; }` in a loop of `n_steps = 8` that does no solves), `:385-405` (`relax_tau_factor_to_target`: loop only reassigns `model.controls.tau_factor`), `:409-441` (`relax_irradiation_to_target`: loop reassigns `irradiation_flux`), `:445-494` (`relax_from_profile_files`), `:323-355` (`relax_to_remove_h_env`)
- Upstream: `star/private/relax.f90` (4215 LOC, 126 routines). `do_relax_mass` (`relax.f90:125`), `do_relax_composition` (`:315`), `do_relax_entropy` (`:693`), `do_relax_angular_momentum` (`:946`), `do_relax_tau_factor` (`:1393`), `do_relax_irradiation` (`:1617`), `do_relax_core` (`:1873`) all call `do_internal_evolve` (`:3780`), i.e. a real sequence of `star_evolve_step` calls with a `relax_*_check_model` callback that ramps the parameter and lets the model re-equilibrate.
- Confidence: verified-by-reading-both-sides (Rust read in full; upstream routine list and `do_internal_evolve` structure).
- What: (a) Mass relaxation is a homology shift of radius by `dlnM/3` (plus optional L scaling); no solver is invoked. (b) `relax_initial_mass_to_remove_H_env` picks the first zone with `X(h1) <= 1e-4`, clamps its `q` to `[0.1, 0.98]`, or falls back to `0.85 M` (`relax.rs:342,352`), neither in upstream. (c) entropy relaxation (`relax.rs:485-492`) is `ln_t += 0.01*(s/1e8 - 1)` clamped to `[6, 22]`: a 1% nudge with an invented normalisation `1e8`, where upstream (`do_relax_entropy`, `relax.f90`) solves for `T` from the target entropy via the EOS and relaxes with `other_energy` over `max_steps_to_relax_entropy` steps. (d) angular-momentum relaxation sets `j_rot` by linear interpolation of the file directly and `omega = j/(2/3 r^2)`; upstream relaxes with a torque over many steps and uses `i_rot` including the rotational deformation. (e) The files are found by fixed names `angular_momentum.dat`/`entropy.dat` in the case dir (`relax.rs:474,485`) and applied whenever present, regardless of `relax_initial_angular_momentum`/`relax_initial_entropy` and of the inlist's `relax_angular_momentum_filename`/`relax_entropy_filename`.
- Proposed fix: port `do_internal_evolve` and the `relax_*` family. L.

### C1-4: `settle_relaxed_loaded_model`: seven pseudo-steps with a fixed shrinking dt ladder, errors discarded
- Area: flexmesa-star / `relax.rs`
- Severity: wrong numerics
- FlexMESA: `crates/flexmesa-star/src/relax.rs:534-543` (`for dt_yr in [100.0, 10.0, 1.0, 0.1, 1.0e-2, 1.0e-3, 1.0e-4] { ... let _ = crate::evolve::star_evolve_step(...)`), gate `:505-509` (`model.dt >= 10.0 * SECYER` -> skip)
- Upstream: no such ladder. The doc comment cites `do_internal_evolve` at `relax.f90:1431`; the routine is actually at `relax.f90:3780` and is a general evolve loop with a caller-supplied termination check, not a fixed dt sequence.
- Confidence: verified-by-reading-flexmesa-only + inferred for upstream line.
- What: After loading a `.mod` with `relax_tau_factor` or `relax_mass` and an initial dt below 10 yr, seven evolve steps are taken with nuclear burning, mass change and remeshing switched off, the result of each step ignored (`let _ =`), then age/model number/dt are restored. This changes the starting structure of those cases in a way upstream does not, and silently tolerates non-convergence.
- Proposed fix: remove once real relaxations exist. M.

### C1-5: Case-name branches in the library initialiser
- Area: flexmesa-star / `star_job_init.rs`
- Severity: wrong physics (for the cases) / maintainability
- FlexMESA: `crates/flexmesa-star/src/star_job_init.rs`
  - `:161-165` derives `case_name` from the directory name of the inlist.
  - `:168-187` `is_astero_case`, `is_pulse_mod_case = case_name.starts_with("gyre_in_mesa_")`, `case_name.starts_with("dev_rsp2_")`, `is_tier2_or_3` built from the four tier lists.
  - `:206-213` `case_name == "make_env"`: loads a ZAMS model with mass `x_ctrl(1).clamp(0.55, 15.0)` and excises everything hotter than `4.0e6` K.
  - `:240` `case_name.starts_with("rsp_")`.
  - `:266` `matches!(case_name, "make_he_wd" | "make_co_wd" | "make_o_ne_wd")` -> scaled-ZAMS builder.
  - `:337`, `:350` `case_name != "tzo"` skips composition relaxation and `relax_m_center`.
  - `:501-505` `use_other_cgrav && !is_tier3_case(case_name)`.
  - `:539-579` for tier-2 cases loaded from `.mod`: `matches!(case_name, "irradiated_planet" | "wd_acc_small_dm" | "wd_c_core_ignition")` unconditionally discards the loaded model and substitutes the scaled-ZAMS model; other tier-2 cases do the same when the initial residual norm exceeds `0.25`.
- Upstream: none of these exist; `star/private/init.f90` and `star/job/run_star_support.f90` branch only on `&star_job` flags.
- Confidence: verified-by-reading-flexmesa-only.
- What: The library function that every run goes through behaves differently depending on the name of the directory containing the inlist. Copying an upstream test case to a directory with another name changes which initial model is built. For `irradiated_planet`, `wd_acc_small_dm`, `wd_c_core_ignition` the upstream-provided starting `.mod` (a real planet/WD model) is thrown away in favour of a rescaled solar ZAMS model with modified `G`.
- Proposed fix: remove all `case_name` use from `star_job_init.rs`. L (because the cases then fail until real physics exists).

### C1-6: `load_mod_or_fallback` swallows `.mod` parse errors
- Area: flexmesa-star / `star_job_init.rs`
- Severity: wrong numerics / unverified claim
- FlexMESA: `crates/flexmesa-star/src/star_job_init.rs:131-136` (`if let Ok(m) = read_mod_str(...) { ... return Ok(m); } ... build_initial_planet_or_wd_model(controls, job)`)
- Upstream: `star/private/read_model.f90` `do_read_saved_model` returns `ierr /= 0` and the run stops.
- Confidence: verified-by-reading-flexmesa-only.
- What: If the saved model cannot be found or parsed, a scaled 1 Msun ZAMS model is silently used. Also `resolve_model_path` (`:43-48`) falls back to `tests/fixtures/golden/rsp/<file>` inside the repository via `env!("CARGO_MANIFEST_DIR")`, a test fixture path compiled into library code.
- Proposed fix: propagate the error. S.

### C1-7: Flags and controls "preserved" across stages regardless of the new inlist
- Area: flexmesa-star / `star_job_init.rs`
- Severity: wrong numerics
- FlexMESA: `crates/flexmesa-star/src/star_job_init.rs:292-316` (`model.controls.tau_factor, initial_mass, initial_z, initial_y = preserved_*`)
- Upstream: `tau_factor`, `v_flag` etc. are star-info state saved in the model/photo (`star_data_step_input.inc`) and are correctly not controls; `initial_mass/initial_z/initial_y` are controls re-read from each inlist.
- Confidence: verified-by-reading-flexmesa-only (upstream by knowledge of `ctrls_io.f90` namelist; `tau_factor` is in `star_data`).
- What: Preserving `v_flag/u_flag/rotation_flag/tau_factor` mirrors upstream semantics (these live in the model). Preserving `initial_mass/initial_z/initial_y` over the new inlist values does not: a later stage inlist that sets `initial_z` is ignored. Minor, but it interacts with EOS/kap `Zbase` defaults if those are derived from `initial_z`.
- Proposed fix: S.

## B.2 The `tierN_extras.rs` files: not ports of upstream `run_star_extras.f90`

Verdict: these files are compiled into the library (`crates/flexmesa-star/src/lib.rs`), selected by directory name in
`inlist_runner.rs:417-429,470-482`, and they (a) overwrite the controls the inlist set, (b) inject case-specific physics
steps and state edits, (c) force termination after a fixed small number of steps while emitting upstream's success
termination strings, and (d) define the pass criterion. None of the upstream `star/test_suite/<case>/src/run_star_extras.f90`
logic (which checks physical targets from `x_ctrl`, e.g. `tzo/src/run_star_extras.f90:143`, `ppisn/src/run_star_extras.f90:1375`,
`pisn/src/run_star_extras.f90:227`) is translated. Upstream has per-case code too, but it lives in the test case, is compiled
per case, and never alters the star library's behaviour by case name.

### C1-8: Tier 3 extras force termination after 2 steps with upstream's success strings, and fake the physical trigger
- Area: flexmesa-star / `tier3_extras.rs`
- Severity: unverified claim (the "green" status of these cases) / wrong physics
- FlexMESA: `crates/flexmesa-star/src/tier3_extras.rs`
  - `:398-412` core-collapse cases `12M_pre_ms_to_core_collapse | 20M_pre_ms_to_core_collapse | zams_to_cc_80 | make_pre_ccsn_13bvn`: `let lim = model.controls.fe_core_infall_limit.unwrap_or(1.0e7); if let Some(v_c) = model.v.last_mut() { *v_c = -1.1 * lim; }` i.e. the centre velocity is overwritten every step with 1.1x the infall limit so that the `fe_core_infall_limit` termination fires.
  - `:412-419` `pisn`: `model.u.fill(1.2e9); if model.model_number >= 2 { return Terminate("Star is unbound") }`.
  - `:420-424` `ppisn`: `model_number >= 2` -> `Terminate("Successful test: evolved 100 days past first relax")`.
  - `:425-429` `ccsn_IIp`: `model_number >= 2` -> `Terminate("extras_finish_step")`.
  - `:430-436` `twin_studies`: `model.star_age = model.star_age.max(1.013e7)` then `Terminate("stars have reached stopping age")`.
  - `:385-394` `tzo`: `x_integer_ctrl(1) == 4 && model_number >= 2` -> `Terminate("PASS: Have reached requested NS mass")`.
  - `:376-384` `conserve_angular_momentum`: after each step all `j_rot` are rescaled so total J equals the initial value; the case's assertion (`:134-137`) then checks J is conserved.
- Upstream: `star/test_suite/pisn/src/run_star_extras.f90:227` sets `'Star is unbound'` only when the star actually unbinds; `ppisn/src/run_star_extras.f90:1375`; `tzo/src/run_star_extras.f90:143`.
- Confidence: verified-by-reading-flexmesa-only for the Rust; upstream strings located by grep, surrounding conditions not read line by line.
- What: the termination strings that upstream emits when a physical condition is reached are emitted after two timesteps irrespective of the model state; state variables are overwritten to make built-in stop conditions fire. A "pass" for these cases carries no physical information.
- Proposed fix: remove; port each case's `run_star_extras.f90` into test code outside the library. L.

### C1-9: Tier 2/3/4/5 `extras_controls` overwrite the inlist's physics controls
- Area: flexmesa-star / `tier2_extras.rs`, `tier3_extras.rs`, `tier4_extras.rs`, `tier5_extras.rs`
- Severity: wrong physics
- FlexMESA:
  - `tier2_extras.rs:464-492`: `op_split_burn = true; min_logt_for_net = 8.3` (except 3 accretion cases); `mesh_delta_coeff = 0.0` (remeshing off); `atm_option = Table`; `energy_eqn_option = EpsGrav`; Blocker/Reimers winds zeroed; `dt = 1e2 yr` (or `1.0` s for `conductive_flame`); `wd_acc_small_dm`: `max_age = None`.
  - `tier3_extras.rs:236-256`: as above with `min_logt_for_net = 8.5`, plus `mlt_option = Cox; mixing_length_alpha = 2.0; use_other_alpha_mlt = false; use_other_adjust_mdot = false; use_other_torque = (case == magnetic_braking); tau_factor = 1.0; mass_change = 0.0; eps_nuc_factor = 1.0;` all four wind scaling factors zeroed; five stop limits set to `None`. `:288-301`: `v_flag = false; u_flag = false` ("Solve structure in 4-variable hydrostatic layout"), `dt = 1e2 yr`.
  - `tier3_extras.rs:258-287`: mass forcibly rescaled by homology (`set_scaled_tier3_mass`, `:215-233`) to hard-coded values: `high_mass` 80, `hot_cool_wind` 6.75, `15M_dynamo` 13.5, `accreted_material_j` 20, `pisn` 65, `ppisn` 40, `ns_*` 1.4, `tzo` 15, `twin_studies` 20 Msun.
  - `tier3_extras.rs:308-357`: rotation imposed with hard-coded rates per case (`0.98`, `0.65`, `0.20` of critical, 50/200/10 km/s) instead of the inlist's `new_omega_div_omega_crit`/`new_surface_rotation_v`; `split_burn_big_net` hard-codes `mesa_80.net`; `pisn` `u.fill(1.2e9)`.
  - `tier4_extras.rs:203-251` (non-RSP, non-astero tier-4 cases, i.e. `gyre_in_mesa_*`): same overrides plus `rotation_flag = false`, all `use_other_*` false, diffusion/premix/phase separation off, all winds zero, `dxdt_nuc_factor = 0.0; non_nuc_neu_factor = 0.0; varcontrol_target = 1e-3; max_years_for_timestep = 1e2`, `dt = 10 yr` or `1e-2 yr`.
  - `tier5_extras.rs:332-345`: `op_split_burn = true; min_logt_for_net = 8.5; mesh_delta_coeff = 0.0; atm Table; EpsGrav; Cox; alpha = 2.0; v_flag = u_flag = false; dt = 1e2 yr`.
  - `tier1_extras.rs:375-409`: `op_split_burn = true` when `species > 8` or for 11 named cases; `make_env`/`make_sdb` overrides.
- Upstream: `star/defaults/controls.defaults:9325` `op_split_burn = .false.`; there is no `min_logT_for_net` control at all (only `max_logT_for_net`, `controls.defaults:6759`); the test inlists set their own winds, MLT option, alpha, `v_flag`/`u_flag` etc.
- Confidence: verified-by-reading-flexmesa-only; upstream defaults checked.
- What: `min_logt_for_net` is a FlexMESA-invented control (`controls.rs:77`, default `6.3` at `controls_defaults.rs:43`) below which `do_net` and the split burner skip the zone (`micro_net.rs:130-131`, `struct_burn_mix.rs:135-136`). Setting it to 8.3/8.5 turns off all nuclear burning below T = 2e8..3e8 K, i.e. all hydrogen burning and core helium burning, in every tier 2-5 case. Combined with remeshing off, winds off, hydrodynamics off (`v_flag`/`u_flag` false in cases whose whole purpose is hydrodynamics: `hse_riemann`, `ccsn_IIp`, `pisn`, `ppisn`), these cases do not exercise the physics their names and inlists describe.
- Proposed fix: delete; if a case cannot run with its real inlist it is not passing. L.

### C1-10: Tier 4 non-RSP cases terminate after 2 steps with a termination code synthesised from the inlist
- Area: flexmesa-star / `tier4_extras.rs`
- Severity: unverified claim
- FlexMESA: `crates/flexmesa-star/src/tier4_extras.rs:361-377` (`if self.step_count >= 2 { let code = if let Some(sp) = ...xa_central_lower_limit_species... { format!("xa_central_lower_limit({sp})") } else if stop_near_zams { "Lnuc_div_L_zams_limit" } else if max_age.is_some() { "max_age" } else { "max_model_number" }; return Terminate(code) }`); `:299-304` `rsp_check_2nd_crossing` with `rsp_flag` terminates before the first step with `"other_rsp_build_model"`.
- Upstream: no counterpart.
- Confidence: verified-by-reading-flexmesa-only.
- What: after two steps the run stops and reports whichever termination reason the inlist would have produced had the star actually evolved to it. Logs/reports then show e.g. `xa_central_lower_limit(h1)` for a model that took two steps.
- Proposed fix: delete. S.

### C1-11: Tier 1/2 extras run physics modules from `extras_finish_step`, keyed on case name, with hard-coded parameters
- Area: flexmesa-star / `tier1_extras.rs`, `tier2_extras.rs`
- Severity: wrong physics
- FlexMESA:
  - `tier1_extras.rs:435-441`: `do_conv_premix(model, 3, false)` when `do_conv_premix` or case is `conv_core_cpm | 16M_conv_premix`.
  - `tier1_extras.rs:443-449`: `16M_predictive_mix`: `add_predictive_mixing(model, 1.0e-2, 1.0e-4)` then an extra composition diffusion solve with `compute_convection_sigmas(model, dt_mix, 1.0, 1.0e10)`.
  - `tier1_extras.rs:451-463`: `extended_convective_penetration`: `eval_castellani_penetration_vel(1.0e5, 1.0e9, 1.0e8, 0.42, 0.40, 1.3, 1.2)` with constant arguments, then `d_mix[nz/2] = max(d_mix, v_pen*1e8)` (the middle zone of the mesh) and an extra mixing solve.
  - `tier1_extras.rs:464-476`: element diffusion `do_element_diffusion(model, model.dt.min(1.0e11), DiffusionScheme::Paquette)` after the step, when `do_element_diffusion` or one of three case names; errors discarded (`let _ =`).
  - `tier1_extras.rs:478-493`: `make_metals` with `initial_z < 0.02`: every step, every zone with `X(h1) > 0.72` has `0.02` moved from h1 to the last species.
  - `tier1_extras.rs:416-424`: `starspots`: `tau_factor /= alpha.max(0.1)`.
  - `tier2_extras.rs:494-510`: irradiation heat added to `eps_mdot` capped at `1e-2 * L/M`.
  - `tier2_extras.rs:511-522` + `:134-214`: phase separation as a post-step operation with invented latent heat `0.77 kT/(abar*amu*tau_cryst)`, `tau_cryst = max(dt, 1e9 yr)`, `gamma_crit = 175/185`, and a `dx_ne = (0.015 x(1-x)).clamp(0, 0.02)` Ne swap; the latent luminosity is spread uniformly over all zones (`eps_lat = l_lat / mstar`).
  - `tier2_extras.rs:523-532`: element diffusion with `DiffusionScheme::CaplanOcp` for `wd_diffusion`.
  - `tier2_extras.rs:533-548`: `conductive_flame` terminates with `"flame_reached_target"` when `model_number >= 5` (or flame q past target); `cburn_inward` terminates `"flame_reached_center"` at `model_number >= 5`.
  - `tier2_extras.rs:216-323` `apply_accretion_composition_and_nova_wind`: accreted composition blended into the top `(nz/10).clamp(1, 8)` zones with weight clamped to `[0.01, 0.5]`; super-Eddington wind uses `kap_surf.max(0.2)`.
- Upstream: element diffusion is called inside the step (`star/private/evolve.f90` -> `do_element_diffusion`, `element_diffusion.f90`) with substepping to cover the full `dt`; conv premix inside `set_mixing_info` (`mix_info.f90` -> `conv_premix.f90`); phase separation `do_phase_separation` (`phase_separation.f90`) called from `evolve.f90` with the Blouin/Caplan phase diagrams and latent heat through `eps_phase_separation`; accretion composition in `adjust_mass.f90` (`set_xa` for added mass). None is driven by case name; none has these constants.
- Confidence: verified-by-reading-flexmesa-only (upstream call sites by routine name).
- What: several physics features are not wired into the evolve step at all (or only here), run at the wrong point in the step (after it is accepted), with time truncated (`dt.min(1e11)` s = 3170 yr), and with fabricated parameters. `do_element_diffusion`, `do_conv_premix`, `do_phase_separation` being reachable only from tier extras means a user inlist with `do_element_diffusion = .true.` outside a tier-1/2 directory gets no diffusion (see lead L1 to confirm absence of an in-step call).
- Proposed fix: wire the real modules into `evolve.rs`; delete the extras. L.

### C1-12: Pass criteria in `verify_tierN_case_assertions` are near-vacuous
- Area: flexmesa-star / tier extras
- Severity: unverified claim
- FlexMESA: `tier3_extras.rs:99` (`"high_mass" => hr.star_mass >= 50.0 && hr.log_center_t > 6.5`), `:147`, `:165` (`"ppisn" => hr.star_mass >= 20.0 && hr.log_center_t > 6.5`), `:189` (`"twin_studies" => hr.star_mass >= 10.0 && model.star_age >= 1.013e7`, with the age set by the extras at `:431`); `tier2_extras.rs:430-436` (six WD accretion/ignition cases: `hr.star_mass > 0.4 && hr.log_center_t > 5.0`); `tier5_extras.rs:406` (`"timing" => hr.star_mass > 0.5 && hr.log_center_t > 6.5`), `:401`; `tier4_extras.rs:464-465` fallback `dnu > 0 && numax > 0`; `inlist_runner.rs:516-521` twin match `|dM| < 0.5 && |dlogTeff| < 0.2`; `inlist_runner.rs:529` default `nz >= 2 && teff finite`.
- Upstream: each case's `run_star_extras.f90` `extras_after_evolve` checks targets from the inlist, and the suite compares against `test_output` goldens.
- Confidence: verified-by-reading-flexmesa-only.
- What: "107/107 green" means these predicates returned true. They test that mass and central temperature are in a very wide range.
- Proposed fix: replace with upstream's own checks. M.

## B.3 Heuristics in the core step that have no upstream counterpart

### C1-15: `atm_option = 'table'` with an empty `atm_table` uses a "legacy table proxy": the surface boundary condition is back-fitted to the model's own start-of-step surface state
- Area: flexmesa-star / `hydro_brunt_surf.rs`
- Severity: wrong physics
- FlexMESA: `crates/flexmesa-star/src/hydro_brunt_surf.rs:260-263` (`let legacy_table_proxy = model.controls.atm_option == AtmOption::Table;`), `:319-345` (`let p_gas_surf = (p_start - dp_0 - 0.5 * p_rad).max(1.0); ... let kap_eff = (tau_surf * g_s / p_gas_surf).max(1.0e-10); (kap_eff, 0.0, 0.0)`), `:395-405` (`lnt_ad.val = model.ln_t_start[0] + (teff / teff_s.max(1.0)).ln();`). Default `atm_table` is empty (`controls_defaults.rs:37`), and every tier 2/3/5 extras and the non-RSP tier-4 extras set `atm_option = AtmOption::Table` without setting `atm_table` (`tier2_extras.rs:473`, `tier3_extras.rs:241`, `tier4_extras.rs:210`, `tier5_extras.rs:336`, `tier1_extras.rs:399,404`).
- Upstream: `atm/public/atm_lib.f90` `atm_eval_table` + `star/private/atm_support.f90` `get_table`: a real table lookup in `(Teff, logg)` with blending to `T_tau` off-table; `atm_table` default is `'tau_100'` (`star/defaults/controls.defaults:4385`). No path derives the BC opacity from the model's own pressure.
- Confidence: verified-by-reading-flexmesa-only (upstream by routine names and the default).
- What: in this path an "effective opacity" is computed such that the grey `P = tau g / kappa` boundary pressure equals the first cell's start-of-step pressure minus half a cell of weight, with zero derivatives, and the boundary temperature is the start-of-step surface temperature scaled by `Teff/Teff_start`. The outer boundary condition is thereby satisfied by construction for whatever model is supplied. Together with C1-2 this is why rescaled models "converge". The tier extras force this path for most non-tier-1 cases.
- Proposed fix: remove the proxy; error on `Table` with unknown/empty table or apply upstream's default `'tau_100'`. S (but exposes failing cases).

### C1-16: conv_premix, predictive mixing, phase separation and element diffusion are placeholder algorithms with constants not found upstream
- Area: flexmesa-star / `premix.rs`, `diffusion.rs`
- Severity: wrong physics
- FlexMESA:
  - `premix.rs:94-185` `do_conv_premix`: for each convective boundary, homogenise a fixed window of up to 8 cells plus the adjacent radiative cell (`let k_end = (k_bdy + 8).min(nz - 1)`), accept if `gradr > gradl` there; at most `max_iters` (3 or 5) cells per step; `avoid_fuel_increase` tests species index 0 against `+0.05` (`:149-154`); const-pressure handled by "One damped Newton step on lnd" with `chi_rho.max(0.1)` and step clamp `[-0.1, 0.1]` (`:68-78`); `d_mix = d_mix[k_bdy].max(1.0e12)` (`:165`).
  - `premix.rs:187-249` predictive mixing: no trial mixing; "predicted" gradr is `gradr[k_adj] * max(kap_bdy/kap_adj, 1) + 0.5*max(gradr_bdy - gradr_adj, 0)` (`:228-231`), then one adjacent face is marked convective per boundary per call.
  - `premix.rs:17-41` phase diagram: 5-coefficient polynomials `[0.642856, -1.954689, 2.487018, -1.789000, 0.613815]` (C/O) and `[0.435920, -1.242958, 1.711583, -1.366634, 0.462089]` (O/Ne) in a number-fraction variable; `premix.rs:301-397` moves the excess from the 6 cells above (`let k_top = k.saturating_sub(6)`), crystallisation front located by `gam >= max(gamma, 100)` with `gamma = 175` passed from `evolve_support.rs:138`.
  - `diffusion.rs:93-102` "Paquette" collision integrals: `z_ij = 0.35 + 0.25 * w; zp_ij = 0.85 + 0.45 * w; zpp_ij = 1.35 + 0.65 * w` with `w = 0.5*(1 + tanh(0.7*(psi - 0.5)))`; `:103-116` Stanton-Murillo/Caplan likewise closed-form blends; `:421-567` driver: one backward-Euler step for the whole (clamped) `dt` with a single abundance-weighted mean diffusion coefficient for all species (`d_mean`, `:501-514`), settling transfers capped at 25% of the donor cell (`:524-531`), `dt.clamp(1.0, 1.0e13)` at the call site (`evolve_support.rs:147`).
- Upstream:
  - `star/private/conv_premix.f90` (1568 LOC, 16 routines): per-zone ingestion with fractional-cell tracking (`advance_bdy`), burn-type dependent `avoid_increase`, `conv_premix_time_factor`, full EOS/MLT re-evaluation at fixed P and T (`update_model_`).
  - `star/private/predictive_mix.f90` (974 LOC): `do_predictive_mixing` mixes the trial region and recomputes `gradr/grada` with EOS and opacity (`eval_mixing_coeffs`), with `predictive_superad_thresh`, `predictive_avoid_reversal`, `predictive_limit_ingestion`.
  - `star/private/phase_separation.f90:409-439` `blouin_delta_xo`: `a1 = -0.311540, a2 = 2.114743, a3 = -1.661095, a4 = -1.406005, a5 = 1.263897` (`:418-423`); O/Ne `blouin_delta_xne` `a1 = -0.120299 ...` (`:454-459`); 3-component Medin-Cumming tables (`:481-812`); front located by EOS phase `s% phase(k) > eos_phase_boundary = 0.9` (`:35,:94`), then `mix_outward` using the Ledoux criterion (`:351-407`).
  - `star/private/paquette_coeffs.f90` (721 LOC of tabulated spline coefficients), `diffusion_support.f90` (1800), `diffusion_procs.f90` (1746), `diffusion.f90` (2050): coupled multi-species Burgers system per class, implicit solve with adaptive substeps.
- Confidence: verified-by-reading-both-sides for the phase-diagram coefficients and front criterion; verified-by-reading-flexmesa-only plus upstream routine lists for the rest.
- What: each of these four features exists as a function with upstream's name but an invented algorithm. The phase diagram coefficients do not match upstream's Blouin et al. fits. Results from any run with `do_conv_premix`, `predictive_mix`, `do_phase_separation` or `do_element_diffusion` are not MESA results.
- Proposed fix: port the upstream modules. L each.

### C1-17: Newton solver uses a bespoke backtracking/acceptance rule and hard-coded step limits in place of upstream's `adjust_correction`
- Area: flexmesa-star / `solver.rs`
- Severity: wrong numerics
- FlexMESA: `crates/flexmesa-star/src/solver.rs:55-57` (`let max_dln = 0.5_f64; let max_dx_chem = 0.25_f64; let eps_neg_xa = 5.0e-5_f64;`), `:297-334` (up to 7 halvings; accept if `iter == 1 || next_rnorm <= 1.5 * prev_residual_norm.max(1.0e-4)`), `:336-355` (otherwise take the best trial if `best_rnorm < 1.0e6`), `:384-386` (`tight_resid = residual_norm <= 1.0e-12 && max_residual <= 1.0e-10 && pass_corr_without_coeff` as an extra convergence path), `:248-252` (`max_newton_iters.max(40)` for gold2).
- Upstream: `star/private/star_solver.f90:400-413` (`scale_max_correction`, `min_corr_coeff = s% corr_coeff_limit`, `solver_reduced_min_corr_coeff`), `:739-814` `adjust_correction` (cubic/quadratic line search on `f = 0.5*sum(equ^2)` with slope test), `:288-298` (`max_tries` from `solver_max_tries_before_reject`, `max_tries_for_retry`, `max_tries_after_5/10/20_retries`), `:455,:557` (`tiny_corr_factor`, `tiny_corr_coeff_limit`), `:576-585` (`solver_itermin`, `force_another_iteration`); `solver_support.f90` `Bdomain`/`clip_so_non_negative`.
- Confidence: verified-by-reading-both-sides (Rust loop read in full; upstream by the cited line ranges).
- What: none of `scale_max_correction`, `corr_coeff_limit`, `solver_itermin`, `tiny_corr_*`, `max_tries*` (other than a name read into `inlist_populate_limits.rs`) appear in the Rust crate (grep). The acceptance rule allows the residual to grow by 50% per iteration and always accepts the first iteration. Iteration counts, retries and therefore timestep histories cannot match upstream even where the equations do.
- Proposed fix: port `do_solver_work` and `adjust_correction`. M.

### C1-18: Mesh planner is a different algorithm from `mesh_plan.f90`
- Area: flexmesa-star / `mesh_adjust.rs`
- Severity: wrong numerics
- FlexMESA: `crates/flexmesa-star/src/mesh_adjust.rs:259-349`: a cell is split into `ceil(g).clamp(2, 4)` equal parts when its own `g > 1` (`:305-316`); two neighbours are merged when `g_k + g_{k+1} < 0.45`, same mixing type, away from the 2 outer/3 inner cells (`:326-336`); `mesh_max_allowed_ratio` is stored (`:148`) but never applied. Mesh functions: `P`, `T`, `R`, `M`, `gradT`, `omega`, `gam`, and one xa function using the maximum weight over all configured species applied to every species (`:117-127,:219-227`).
- Upstream: `star/private/mesh_plan.f90:601-` `pick_new_points` walks the old mesh choosing each new `dq` so that the largest mesh-function change equals `mesh_delta` (`pick_next_dq`), limits neighbour ratios (`next_dq_max = dq_new(k_new-1)*mesh_max_allowed_ratio`, `:636`), merges up to `max_num_merge_cells`, then `smooth_new_points` (`:525-598`); `mesh_functions.f90` adds `log_tau`, `log_kap`, per-species `xa_function_species(i)/weight(i)/param(i)` (default he4/30/1d-2, `controls.defaults:6127-6129`), and `adjust_mesh_support.f90` applies the `mesh_dlog_*_dlogP_extra`, `xtra_coef_*` and `mesh_delta_coeff_for_highT` local multipliers.
- Confidence: verified-by-reading-both-sides (Rust in full; upstream by structure and cited lines).
- What: zone counts and zone placement cannot match upstream; the `0.45` merge threshold and the 2..4 equal split are invented.
- Proposed fix: port `mesh_plan.f90`/`mesh_functions.f90`/`adjust_mesh_support.f90`. L.

### C1-19: MLT/TDC face evaluation: hard-coded parameters and silent fallbacks
- Area: flexmesa-star / `hydro_vars.rs`
- Severity: wrong physics (when the controls differ from the hard-coded values) / wrong numerics
- FlexMESA: `crates/flexmesa-star/src/hydro_vars.rs`
  - `:183-185` `if k == 0 && model.tau[0] < 2.0 / 3.0 && model.tau[0] > 0.0 { gradr_f = gradr_f * (1.0 + 2.0 / (3.0 * model.tau[0])); }`
  - `:215-223` `steps_before_use_tdc` switch between Cox and TDC by model number.
  - `:225-229` TDC only when `k > 0 && vc_old > 0.0 && dt > 0`; otherwise falls through to static MLT with the option `Tdc` passed to `set_mlt`.
  - `:245-249` `tdc_alpha_d: 1.0, tdc_alpha_pt: 0.0, tdc_alpha_c: 1.0, tdc_alpha_s: 1.0`; `:261,:300` `max_conv_vel: 1.0e99`; `:298-299` `henyey_nu: 8.0, henyey_y: 1.0 / 3.0`; `:343` thermohaline driving species fixed to `DrivingSpecies::he4()`.
  - `:287` `set_tdc(..).ok()` and `:354,:388` `if let Ok(..)` discard TDC/thermohaline/semiconvection errors.
  - `:203-213` `fixed_gradr_sticky`.
  - `:73-90` `update_sig_from_d_mix`: cap `sig <= 1.0e6 * dm_min / dt`.
- Upstream: `star/private/turb_support.f90` `do1_mlt_eval` (uses `s% TDC_alfa_D/alfap/alfat/alfam`... via `turb` module args, `max_conv_vel = s% csound_face(k)*s% max_conv_vel_div_csound`, `Henyey_MLT_nu_param`, `Henyey_MLT_y_param`, `alt_scale_height_flag`, `set_superad_reduction`, `use_other_mlt_results`); `star/private/turb_info.f90:58-59,:227` (`fixed_gradr_for_rest_of_solver_iters`, the counterpart of the sticky flag: MATCH in intent); `star/private/hydro_vars.f90:588-599` (`use_T_tau_gradr_factor`, the only upstream gradr surface factor, off by default and T-tau-relation specific); `star/private/mix_info.f90:1448` `siglim = sig_term_limit*xmstar*min(dq(k),dq(k-1))/dt` with `sig_term_limit` default `1d13`.
- Confidence: verified-by-reading-flexmesa-only for the Rust; upstream lines located by grep (inferred for exact equivalence).
- What: `set_superad_reduction`, `alt_scale_height`, `max_conv_vel_div_csound`, `okay_to_reduce_gradT_excess`/`gradT_excess_*`, `Henyey_MLT_*_param`, `use_other_mlt` have no implementation (grep for those names in `flexmesa-star/src` finds none). The unconditional surface `gradr` multiplier and the `1e6` sigma cap (vs `1e13`) are not upstream.
- Proposed fix: M.

### C1-20: Wind scheme "legacy fallback" and other small invented rules
- Area: flexmesa-star
- Severity: wrong physics (narrow)
- FlexMESA:
  - `crates/flexmesa-star/src/winds_driver.rs:156-172`: when both `hot_wind_scheme` and the cool scheme are blank, a scheme is chosen from whichever scaling factor is positive (`Dutch`, `Vink`, `Bjorklund`, `Reimers`, `Blocker`, in that order). Upstream (`star/private/winds.f90` `do_set_mdot`): blank scheme means no wind regardless of scaling factors.
  - `crates/flexmesa-star/src/zams.rs:330-338` `yrs_for_init_timestep`: `1e7` yr below 0.3 Msun, `1e5/M^2.5` up to 30 Msun, `20*(30/M)^1.5` above. Upstream `star/private/star_utils.f90:2913-2920`: `1d5` for `M <= 1`, else `1d5/M^2.5`. (0.5 Msun: 5.7e5 yr vs 1e5 yr; 60 Msun: 7.1 yr vs 3.6 yr.) verified-by-reading-both-sides.
  - `crates/flexmesa-star/src/eps_grav.rs:185-197`: lnS/standard blend weight `alfa = eos_frac_pc * w_gam` using `model.gam[k]`. Upstream `star/private/eps_grav.f90:84-99`: weight is `w_gam` alone (when `eos_frac_PC > 0`) and uses `gam_start(k)`. verified-by-reading-both-sides.
  - `crates/flexmesa-star/src/inlist_runner.rs:267-274`: `other_torque` applied explicitly before the step as `j_rot += dj*dt` floored at `1.0e10`. Upstream applies `extra_jdot` inside the angular-momentum solve.
  - `crates/flexmesa-star/src/evolve.rs:210` `dt_floor = min_timestep_limit.clamp(1.0e-20, 1.0)`; `:221-222` `retry_limit.clamp(1, 100)`, `timestep_factor_for_retries.clamp(0.05, 0.95)`; `:347` `if retries >= 4 { retry_fac.min(0.2) }`. Upstream `evolve.f90` `prepare_to_retry` uses `timestep_factor_for_retries` as given and terminates when `dt < min_timestep_limit`.
  - `crates/flexmesa-star/src/evolve_support.rs:447-457` pre-MS relax loop: at most 600 steps, a failed step silently ends the relaxation (`.is_err() { break; }`).
- Confidence: verified-by-reading-flexmesa-only unless stated.
- Proposed fix: S each.

## B.4 Counts per file

Pattern counts in non-test library code (`.clamp(`; `.max(<number>`/`.min(<number>`; `unwrap_or(`; discarded results
`let _ =`/`.is_err()`; `if let Ok`; lines mentioning `case_name`/`cname`). These are pattern counts, not verified
heuristics: many `.max(1.0e-30)` are divide-by-zero guards upstream also has in spirit. Files with a total of 12 or more:

| file (`crates/flexmesa-`) | clamp | max/min literal | unwrap_or | discarded | if let Ok | case-name lines |
| :-- | --: | --: | --: | --: | --: | --: |
| `astero/src/inlist_parser.rs` | 0 | 0 | 82 | 0 | 2 | 0 |
| `astero/src/optimize.rs` | 10 | 12 | 0 | 2 | 1 | 0 |
| `astero/src/sample_eval.rs` | 3 | 8 | 8 | 0 | 0 | 0 |
| `astero/src/scaling.rs` | 0 | 13 | 0 | 0 | 0 | 0 |
| `astero/src/search_driver.rs` | 3 | 11 | 3 | 4 | 0 | 0 |
| `astero/src/search_methods.rs` | 0 | 4 | 0 | 12 | 0 | 0 |
| `astero/src/surface_corr.rs` | 0 | 19 | 0 | 0 | 0 | 0 |
| `binary/src/e2e.rs` | 0 | 2 | 2 | 0 | 0 | 8 |
| `binary/src/e2e_extras.rs` | 1 | 14 | 5 | 0 | 0 | 2 |
| `binary/src/edd_wind.rs` | 5 | 14 | 0 | 0 | 0 | 0 |
| `binary/src/jdot_tides.rs` | 0 | 15 | 0 | 0 | 0 | 0 |
| `binary/src/mdot_implicit.rs` | 1 | 13 | 1 | 0 | 0 | 0 |
| `binary/src/roche_deformation.rs` | 4 | 11 | 0 | 1 | 0 | 0 |
| `binary/src/system_init.rs` | 0 | 9 | 4 | 0 | 0 | 0 |
| `binary/src/system_jdot.rs` | 4 | 22 | 15 | 0 | 0 | 0 |
| `binary/src/system_rlof.rs` | 3 | 21 | 7 | 0 | 0 | 0 |
| `binary/src/system_step.rs` | 5 | 18 | 3 | 5 | 1 | 0 |
| `star/src/adjust_mass.rs` | 3 | 26 | 3 | 0 | 0 | 0 |
| `star/src/diffusion.rs` | 9 | 44 | 2 | 0 | 0 | 0 |
| `star/src/eps_grav.rs` | 2 | 2 | 11 | 0 | 0 | 0 |
| `star/src/eps_mdot.rs` | 1 | 15 | 3 | 0 | 0 | 0 |
| `star/src/evolve.rs` | 4 | 8 | 1 | 1 | 1 | 0 |
| `star/src/evolve_support.rs` | 3 | 7 | 10 | 3 | 0 | 0 |
| `star/src/history.rs` | 0 | 12 | 5 | 4 | 0 | 0 |
| `star/src/hydro_brunt_surf.rs` | 2 | 23 | 4 | 0 | 1 | 0 |
| `star/src/hydro_energy.rs` | 0 | 11 | 24 | 0 | 0 | 0 |
| `star/src/hydro_eqns.rs` | 0 | 17 | 10 | 0 | 0 | 0 |
| `star/src/hydro_momentum.rs` | 0 | 10 | 19 | 0 | 0 | 0 |
| `star/src/hydro_rti.rs` | 0 | 18 | 19 | 0 | 0 | 0 |
| `star/src/hydro_state.rs` | 4 | 9 | 0 | 1 | 0 | 0 |
| `star/src/hydro_vars.rs` | 0 | 12 | 5 | 0 | 3 | 0 |
| `star/src/hydro_work.rs` | 0 | 5 | 12 | 0 | 0 | 0 |
| `star/src/inlist_runner.rs` | 0 | 1 | 3 | 0 | 0 | 21 |
| `star/src/mdot_redo.rs` | 2 | 32 | 4 | 0 | 0 | 0 |
| `star/src/mesh.rs` | 0 | 10 | 2 | 1 | 0 | 0 |
| `star/src/mesh_adjust.rs` | 2 | 10 | 1 | 0 | 0 | 0 |
| `star/src/mesh_remap.rs` | 4 | 15 | 7 | 0 | 0 | 0 |
| `star/src/model_io.rs` | 0 | 3 | 14 | 0 | 1 | 0 |
| `star/src/overshoot.rs` | 2 | 11 | 0 | 0 | 0 | 0 |
| `star/src/pre_ms.rs` | 4 | 12 | 5 | 0 | 2 | 0 |
| `star/src/premix.rs` | 11 | 23 | 4 | 3 | 0 | 0 |
| `star/src/pulse.rs` | 0 | 14 | 7 | 8 | 0 | 0 |
| `star/src/pulse_bvp_recon.rs` | 2 | 28 | 11 | 0 | 1 | 0 |
| `star/src/pulse_exporters.rs` | 0 | 17 | 6 | 12 | 0 | 0 |
| `star/src/pulse_modes.rs` | 4 | 46 | 17 | 0 | 0 | 0 |
| `star/src/pulse_utils.rs` | 3 | 35 | 11 | 0 | 0 | 0 |
| `star/src/relax.rs` | 5 | 31 | 8 | 2 | 0 | 0 |
| `star/src/relax_comp.rs` | 2 | 14 | 14 | 2 | 0 | 0 |
| `star/src/riemann.rs` | 0 | 19 | 4 | 0 | 0 | 0 |
| `star/src/riemann_eqns.rs` | 0 | 6 | 17 | 0 | 0 | 0 |
| `star/src/rotation.rs` | 15 | 36 | 1 | 1 | 0 | 0 |
| `star/src/rotation_mix.rs` | 3 | 62 | 0 | 0 | 0 | 0 |
| `star/src/rsp2.rs` | 1 | 18 | 0 | 1 | 0 | 0 |
| `star/src/rsp2_ad.rs` | 0 | 20 | 2 | 1 | 0 | 0 |
| `star/src/rsp2_eqns.rs` | 0 | 26 | 3 | 1 | 0 | 0 |
| `star/src/rsp_crossing.rs` | 0 | 5 | 14 | 10 | 0 | 0 |
| `star/src/rsp_stats.rs` | 0 | 4 | 3 | 5 | 0 | 0 |
| `star/src/star_job_init.rs` | 1 | 7 | 6 | 0 | 2 | 20 |
| `star/src/stella.rs` | 10 | 31 | 6 | 0 | 0 | 0 |
| `star/src/tier1_extras.rs` | 8 | 13 | 5 | 3 | 0 | 20 |
| `star/src/tier2_extras.rs` | 6 | 42 | 6 | 2 | 0 | 16 |
| `star/src/tier3_extras.rs` | 0 | 4 | 16 | 0 | 1 | 26 |
| `star/src/tier3_physics.rs` | 9 | 48 | 16 | 0 | 0 | 0 |
| `star/src/tier4_extras.rs` | 0 | 0 | 5 | 2 | 0 | 32 |
| `star/src/tier5_extras.rs` | 5 | 12 | 4 | 6 | 1 | 13 |
| `star/src/timestep.rs` | 0 | 12 | 0 | 0 | 0 | 0 |
| `star/src/winds_driver.rs` | 2 | 17 | 9 | 0 | 0 | 0 |

## B.5 Remaining items

### C1-21: Several tier-3 assertions evaluate closed-form functions of constants, not the model
- Area: flexmesa-star / `tier3_extras.rs`, `tier3_physics.rs`
- Severity: unverified claim
- FlexMESA: `crates/flexmesa-star/src/tier3_extras.rs:109-116` (`high_rot_darkening`: `eval_roche_gravity_darkening(0.98)` with a literal argument; the only model-dependent term is `hr.star_mass >= 2.0`), `:143-146` (`magnetic_braking`: `eval_magnetic_braking_torque(model, 5000.0)` with `0 < v_rot < 1 km/s`), `:181-188` (`tzo`: `eval_tzo_core_accretion_step(&mut tzo_state, 1.0, 1.0, 1.4)` on a clone, `m_c_msun > 1.351`), `tier2_extras.rs:418-429` (`wd_diffusion`: `eval_wd_sedimentation_heating(&[22.0], &[10.0], &[0.02], &[-1.0e-5], 12.0, 6.0, ...) > 0`), `tier5_extras.rs:287-303` (`other_physics_hooks`: three helper formulae on literal inputs).
- Upstream: e.g. `star/test_suite/high_rot_darkening/src/run_star_extras.f90` checks the evolved star's history values against targets.
- Confidence: verified-by-reading-flexmesa-only.
- What: the functions in `tier3_physics.rs` (`eval_roche_gravity_darkening`, `eval_pair_instability_gamma1_and_energy`, `eval_duffell_rti_shock_state`, `eval_tov_effective_cgrav`, `eval_tzo_core_accretion_step`, `eval_fe_core_infall_state`, ...) are called only from the assertion functions and one in-crate unit test (grep), plus `eval_tov_effective_cgrav` from `inlist_runner.rs:123` as the `other_cgrav` hook. They do not participate in the evolution; they are diagnostics that let an assertion mention the physics a case is named after.
- Proposed fix: delete with the extras. S.

### C1-22: `astero` `surface_effects` reports `max_model_number` after zero evolution steps
- Area: flexmesa-astero / `e2e.rs`
- Severity: unverified claim
- FlexMESA: `crates/flexmesa-astero/src/e2e.rs:231-240` (`steps_taken: 0, ..., termination_code: String::from("max_model_number")`), `:249` (`if total_modes != 81`).
- Upstream: the case evolves the loaded model for its `max_model_number` steps and calls ADIPLS/GYRE through `astero` (`astero/private/extras_support.f90`).
- Confidence: verified-by-reading-flexmesa-only.
- What: the termination code is a literal. Mode frequencies in all astero cases come from an in-crate BVP solver (`scan_star_bvp_eigenmodes`, `flexmesa-star/src/pulse_bvp.rs`, `pulse_modes.rs`), not from GYRE or ADIPLS (see the coverage map).
- Proposed fix: S.

## B.6 Summary

Do the "extras" files port upstream's per-case `run_star_extras.f90`? No. Upstream's per-case code (a) lives in the test
case, (b) reads targets from `x_ctrl`, (c) sets a termination string when a physical condition is met, and (d) never
changes library behaviour. FlexMESA's `tier1..5_extras.rs`:

- are part of the library and are selected by the name of the directory holding the inlist (`inlist_runner.rs:409-429`);
- overwrite inlist controls with a fixed "make it converge" set (operator-split burning with burning disabled below
  logT 8.3/8.5, remeshing off, winds off, hydrodynamics off, `EpsGrav` energy equation, Cox MLT with alpha 2, the
  tautological surface BC of C1-15, dt = 100 yr): C1-9;
- inject initial conditions (masses, rotation rates, networks, velocities) from hard-coded tables: C1-9;
- terminate after 2 (tier 3, tier 4 non-RSP) or 5 (two tier-2 cases) steps with upstream's success strings: C1-8, C1-10, C1-11;
- define pass criteria that are wide inequalities on mass and central temperature or functions of constants: C1-12, C1-21.

The exceptions that look like legitimate per-case plumbing: `tier4_extras.rs` RSP branch (`:329-359`, stops at
`x_integer_ctrl(1)` periods, checks the period against `x_ctrl(1)` at `:403-444`, which mirrors the upstream
`rsp_*` extras in intent), `tier5_extras.rs` `check_redo` (`:181-232`, a real redo comparison, although several
fields are restored by hand at `:208-219` before comparing), the binary `e2e_extras.rs` analytical oracles
(`jdot_gr_check`, `jdot_ml_check`, `jdot_ls_check`), and `case_staging.rs` (copies the case directory to a temp dir;
pure plumbing, no case-name physics).

Counts (verified items only, by category):

| file | (i) case-name branches (lines) | (ii) hard-coded mass/parameter-range algorithm switches | (iii) clamp/fallback/settling/synthetic construction (distinct items written up) | changes physics results? |
| :-- | --: | --: | --: | :-- |
| `star_job_init.rs` | 20 lines, 9 distinct branches (C1-5) | 3 (`>15.8 or <0.51`, `<=0.02` with irradiation, `relax_mass <= 0.3`) | 4 (scaled-ZAMS builder, `.mod` fallback, tier-2 residual `> 0.25` fallback, repo-fixture path) | yes, all |
| `relax.rs` | 0 | 0 | 6 (G/opacity rescale, homology mass relax, H-env fallbacks, entropy nudge, J overwrite, 7-step dt ladder) | yes, all |
| `tier1_extras.rs` | 20 | 0 | 7 (C1-11) | yes |
| `tier2_extras.rs` | 16 | 0 | 6 (C1-9, C1-11) | yes |
| `tier3_extras.rs` | 26 | 9 forced masses | 8 (C1-8, C1-9) | yes |
| `tier4_extras.rs` | 32 | 0 | 3 (C1-9, C1-10) | yes for non-RSP cases |
| `tier5_extras.rs` | 13 | 0 | 2 (C1-9; hand-restored fields in redo check) | yes |
| `tier3_physics.rs` | 0 | 1 (`target_mass_msun > 1.35`, `:432`) | diagnostics only | no (assertion-only) |
| `inlist_runner.rs` | 21 (tier dispatch, `twin_studies`) | 0 | 2 (explicit torque floor `1e10`, per-step forced `refresh_mixing_state`) | minor |
| `hydro_brunt_surf.rs` | 0 | 0 | 1 (table proxy, C1-15) | yes |
| `hydro_vars.rs` | 0 | 0 | 5 (C1-19) | yes for non-default controls; surface gradr factor always |
| `solver.rs` | 0 | 0 | 4 (C1-17) | numerics (iteration path) |
| `mesh_adjust.rs` | 0 | 0 | 2 (C1-18) | numerics (zoning) |
| `premix.rs`, `diffusion.rs` | 0 | 0 | 4 placeholder algorithms (C1-16) | yes when enabled |
| `winds_driver.rs`, `zams.rs`, `eps_grav.rs`, `evolve.rs`, `evolve_support.rs` | 0 | 2 (`zams.rs:331-337` mass breakpoints 0.3/30) | 6 (C1-20) | narrow |
| `case_staging.rs`, `model_io.rs`, `inlist_populate*.rs` | 0 | 0 | parse fallbacks only (`unwrap_or` defaults; unknown option strings, C1-13) | via silent defaults |
| `flexmesa-binary/src/e2e*.rs` | 10 (dispatch to per-case oracles) | 0 | not examined in depth | not established |
| `flexmesa-astero/src/e2e.rs` | 20 | 0 | 1 (C1-22) | reporting |

## Matches

- `crates/flexmesa-star/src/zams.rs:437-494` vs `star/private/init_model.f90` `get_zams_model`/`get1_mass`: bracketing by mass and log-mass interpolation `alfa = (ln m - ln m2)/(ln m1 - ln m2)` of `lnd, lnT, lnR, L, xa` on the first model's `dq` grid (verified-by-reading-flexmesa-only against the upstream formula quoted in the Rust comment; upstream structure checked by routine list).
- `crates/flexmesa-star/src/pre_ms.rs:278-366` vs `star/private/pre_ms_model.f90:451-556`: same outward integration in `log Pgas` steps with 2 predictor-corrector passes and 9 inner `(m, r)` iterations; root tolerances `1e-3` match upstream `epsx = epsy = 1d-3` (`pre_ms_model.f90:226-227`). Differences: fixed basic-net composition and default EOS config (`pre_ms.rs:384-391`), defaults `pre_ms_T_c`/`pre_ms_d_log10_P` differ (C1-14).
- `crates/flexmesa-star/src/evolve_support.rs:417-482` vs upstream `do_relax_to_radiative_core` (`star/private/relax.f90`): a real evolve loop that stops when a radiative core appears or `Lnuc/L > 0.1` (verified-by-reading-flexmesa-only).
- `crates/flexmesa-star/src/winds_driver.rs:174-220` vs `star/private/winds.f90` `do_set_mdot`: hot/cool scheme selection by `hot_wind_full_on_T`/`cool_wind_full_on_T` with linear blend, RGB->AGB switch on centre h1/he4, `min_wind`/`max_wind`, rotational enhancement with KH limit (structure matches; verified-by-reading-flexmesa-only).
- `crates/flexmesa-star/src/eps_grav.rs:37-145` vs `star/private/eps_grav.f90:120-193` `do_std_eps_grav`: same `-T cp ((1 - grada chiT) dlnT/dt - grada chiRho dlnd/dt)` form with AD partials (not checked term by term beyond the head).
- `crates/flexmesa-star/src/hydro_vars.rs:203-213` sticky radiative faces vs `star/private/turb_info.f90:58-59,227` `fixed_gradr_for_rest_of_solver_iters`: same idea.
- `crates/flexmesa-star/src/star_job_init.rs:281-310` preserving `v_flag/u_flag/rotation_flag/tau_factor` across stages matches upstream, where these are model state saved in the `.mod`.
- Binary component initialisation (`crates/flexmesa-binary/src/e2e.rs:47-83`) seeds each star from the ZAMS catalog and so does not go through the rescaled-1-Msun branch of C1-1.

## Leads (not verified)

- L1: `ZamsCatalog::builtin()` vs the full `zams_z2m2_y28.data` (`flexmesa-binary/src/e2e.rs:30-45` looks for the full file separately): check what mass range and `initial_z` the builtin covers, and whether `initial_z != 0.02` silently uses the Z=0.02 catalog (`zams_filename` default is empty, `controls_defaults.rs:19`).
- L2: `inlist_runner.rs:284-294` calls `set_hydro_vars` + `refresh_mixing_state` + `save_start_values` after every step in the driver; check whether this changes state relative to a plain `star_evolve` loop (two entry points with different behaviour).
- L3: `verify_photo_restart_parity` (`inlist_runner.rs:351-392`) compares a continued model with one deserialised from the bytes just written by the same code; it shows the photo captures the state, and is the meaning of "0 ULP restart parity". It runs once per case, 1 step.
- L4: `tier1_extras.rs` and `evolve_support.rs:113-156` both call `do_conv_premix`/`do_element_diffusion` for the same step in tier-1 cases with those controls on (double application?).
- L5: `flexmesa-binary/src/system_step.rs:400-423` integrates the orbit with a midpoint substep and clamps `dt_reduction_factor_for_j` to `[0.05, 0.5]`; compare with `binary/private/binary_evolve.f90` (single update per step).
- L6: `model_io.rs:480` `if let Ok(net) = get_or_load_net(..)`: a `.mod` naming an unsupported net appears to be accepted with a different net; `:364` unknown isotope names map to nuclide index 1.
- L7: `core_boundaries.rs`, `termination.rs`: `Option`-typed limits replace upstream sentinel defaults; `fe_core_infall_limit`, `log_center_temp_upper_limit = 11`, `log_center_density_upper_limit = 12` are active by default upstream but off here (C1-14).

## Not covered

- `crates/flexmesa-binary/src` beyond the case dispatch (`e2e.rs`, `e2e_extras.rs`) and pattern counts: `system_step.rs`, `system_rlof.rs`, `system_jdot.rs`, `mdot_implicit.rs` were not read for heuristics.
- `crates/flexmesa-astero/src` beyond `e2e.rs:100-260`.
- The RSP files (`rsp_*.rs`), pulse exporters, `stella.rs`, `rotation*.rs`, `adjust_mass.rs`, `mesh_remap.rs`, `timestep*.rs`, `model_io.rs`, `photo_io.rs`: only pattern counts (B.4), not read line by line.
- Upstream per-case `run_star_extras.f90` files were located by their termination strings only; the conditions around them were not read.
