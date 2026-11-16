# F1: flexmesa-star core evolution machinery vs upstream star/private (0b2e6bb)

FlexMESA paths are relative to `the audited tree` (PR #138 head), prefix
`crates/flexmesa-star/src/` omitted where obvious. Upstream paths relative to
`star/private/` unless stated.

Status: written incrementally. See "Coverage of this audit" at the end for what was
and was not read.

---

## Area 1: step driver, retry/redo control flow

### Matches (Area 1)

- Coarse order of operations inside a try: set_mdot -> adjust_mass -> conv_premix ->
  phase separation -> element diffusion -> extra heat -> struct_burn_mix -> implicit
  mdot action. `evolve.rs:226-262` + `evolve_support.rs:110-198` vs
  `evolve.f90:427,570-686`.
- Retry dt factor default path: `dt_try * timestep_factor_for_retries`
  (`evolve.rs:318,330-331`) vs `evolve.f90:2228-2229` (but see F1-3 for the extra
  0.2 factor).
- `timestep_hold = model_number + retry_hold` and `model_number_for_last_retry`
  recorded on retry: `evolve.rs:335-342` vs `evolve.f90:2269-2275` (FlexMESA takes a
  `max` with the previous hold and ignores `neg_mass_fraction_hold`).
- `do_chem = do_burn || do_mix`, `sig = 0` when mixing off, split-burn energy
  bookkeeping `non_epsnuc_energy_change_from_split_burn = sum(e*dm)_after -
  sum(e*dm)_before`: `struct_burn_mix.rs:465-481,492` vs
  `struct_burn_mix.f90:95-140`.
- Uniform extra-heat injection formula `extra*(qtop-qbot)/dq(k)` including the
  fully-inside case (algebraically identical): `evolve_support.rs:171-184` vs
  `evolve.f90:1656-1671`.
- `eval_total_energy_integrals` cell formulae (IE = dm*e; PE = dm*0.5*(phi_00 +
  phi_p1); KE v-flag = dm*0.25*(v00^2+vp1^2)): `evolve_support.rs:36-99`; spot
  checked against `star_utils.f90` (not line-by-line).

### F1-1: "energy conservation error" is the sum of solver residuals, not upstream's independent energy audit
- Area: flexmesa-star/evolve_support
- Severity: wrong numerics (diagnostic) / unverified claim
- FlexMESA: `evolve_support.rs:226-272`
  (`err_erg += dt * model.dm[k] * (d_mech_int_dt + d_eturb_dt + dl_dm - sources);`)
  and `:213-218` (`if energy_eqn_option == EpsGrav ... error_in_energy_conservation = 0.0`)
- Upstream: `evolve.f90:1343-1344`
  (`s% error_in_energy_conservation = s% total_energy_end - (s% total_energy_old + s% total_energy_sources_and_sinks)`),
  sources/sinks assembled at `:1255-1338` from phase1 (adjust_mass, premix,
  phase separation, diffusion, split burn) and phase2 (nuclear, neutrino,
  irradiation, extra heat, radiation, work) terms.
- Confidence: verified-by-reading-both-sides
- What: Upstream compares the change in the integrated total energy over the whole
  step (including everything done before the solver: mass adjustment, remesh,
  premixing, diffusion, split burning) against independently integrated sources
  and sinks. FlexMESA instead re-evaluates the cell energy-equation residual of the
  converged solution and sums `dt*dm*residual`. That quantity is by construction of
  the order of the Newton residual tolerance and is blind to every energy error
  made outside the solver (adjust_mass, remesh, operator-split diffusion/premix,
  split burn, floors/clamps on composition). With `energy_eqn_option='eps_grav'`
  it is hard-wired to exactly 0. Any reported `rel_E_err` /
  `cumulative_energy_error` parity with MESA is therefore not evidence of energy
  conservation. `total_energy_end` is computed but never used in the error. The
  upstream retry gates `limit_for_rel_error_in_energy_conservation` /
  `hard_limit_for_rel_error_in_energy_conservation` (`evolve.f90:1375-1420`)
  have nothing meaningful to act on.
- Proposed fix: implement the phase1/phase2 audit as upstream
  (`total_energy_end - (total_energy_old + sources_and_sinks)`), M.

### F1-2: retry after a remesh throws the remeshed grid away (upstream keeps it unless `restore_mesh_on_retry`)
- Area: flexmesa-star/evolve
- Severity: wrong numerics (control-flow divergence)
- FlexMESA: `evolve.rs:317-329` (`if let Some(snap) = pre_remesh_model.take() { ... *model = snap; ... num_splits = 0;`)
- Upstream: `evolve.f90:2191-2207` (`if (s% restore_mesh_on_retry .and. .not. s% RSP_flag)` ... `xh_old = prev_mesh_xh`), default `restore_mesh_on_retry = .false.` (`star/defaults/controls.defaults:6343`); otherwise `set_current_to_old` (`:2259`) returns to the post-remesh `*_old`.
- Confidence: verified-by-reading-both-sides
- What: In upstream the remesh happens once in `prepare_for_new_step` (first try)
  and `new_generation` stores the remeshed model as `*_old`; retries restart from
  the remeshed model. FlexMESA unconditionally replaces the whole `StarModel`
  with the pre-remesh clone on the first retry, i.e. it behaves as
  `restore_mesh_on_retry=.true.` with no way to turn it off, and the control is
  not consulted. The snapshot assignment `*model = snap` also rewinds every
  field of the model (controls mutated since, counters other than `num_retries`,
  `aux.step_hist`), not just the mesh.
- Proposed fix: honour `restore_mesh_on_retry` (default false) and restore only
  mesh arrays, S.

### F1-3: invented retry policy: factor 0.2 from the 4th retry, clamps on `retry_limit` and `timestep_factor_for_retries`, dt floor clamp
- Area: flexmesa-star/evolve
- Severity: wrong numerics (behaviour upstream lacks)
- FlexMESA: `evolve.rs:330` (`let fac = if retries >= 4 { retry_fac.min(0.2) } else { retry_fac };`),
  `:219-220` (`retry_limit.clamp(1, 100)`, `timestep_factor_for_retries.clamp(0.05, 0.95)`),
  `:208` (`min_timestep_limit.clamp(1.0e-20, 1.0)`), `:331` (`.max(dt_floor)`)
- Upstream: `evolve.f90:2228-2229` (`retry_factor = s% timestep_factor_for_retries; s% dt = s% dt*retry_factor`), `:2221` (`if (s% retry_limit > 0 .and. s% retry_cnt > s% retry_limit)`), `:2246` (`if (s% dt <= max(s% min_timestep_limit,0d0))` -> terminate).
- Confidence: verified-by-reading-both-sides
- What: Upstream applies exactly `timestep_factor_for_retries` on every retry,
  treats `retry_limit <= 0` as "no limit", and terminates with
  `t_min_timestep_limit` when dt drops to `min_timestep_limit`. FlexMESA cuts by
  at most 0.2 from the 4th retry on (not applied on the pre-remesh branch),
  silently clamps user values (a `retry_limit` of 0 or >100, or a retry factor
  outside [0.05,0.95], is changed without any message), clamps
  `min_timestep_limit` to at most 1 s (a user value of e.g. 1e3 s is ignored)
  and pins dt at the floor instead of stopping. Retry dt sequences will differ
  from MESA after the third retry. `max_number_retries` does not exist in the crate and
  `redo_limit` is read from the inlist (`inlist_populate_phase5b.rs:243`) but
  not consulted in `evolve.rs`.
- Proposed fix: remove the 0.2 rule and clamps; follow upstream semantics, S.

### F1-4: `mlt_vc_old` is overwritten on every `save_start_values` call, so TDC "old" convective velocity is not the previous-step value and is not restored on retry
- Area: flexmesa-star/hydro_state, mesh, evolve
- Severity: wrong physics (TDC only)
- FlexMESA: `hydro_state.rs:439` (`self.mlt_vc_old.clone_from_slice(mlt_vc);` inside `save_start`), called from `mesh.rs:379-386` (`save_start_values`), which is called in `evolve.rs:112` (restore), `evolve_support.rs:310` (after remesh), `struct_burn_mix.rs:483` (before solve); consumer `hydro_vars.rs:224-228` (`let vc_old = model.aux.mlt_vc_old...; if mlt_opt == Tdc && model.dt > 0.0 && k > 0 && vc_old > 0.0`). `restore_from_old` (`evolve.rs:76-114`) has no `mlt_vc` restore.
- Upstream: `evolve_support.f90:193` (`s% mlt_vc(k) = s% mlt_vc_old(k)` in `set_current_to_old`) and `evolve.f90:316` (`s% okay_to_set_mlt_vc = .false.  ! don't change mlt_vc until have set mlt_vc_old`); `mlt_vc_old` is set once per step in `new_generation`.
- Confidence: verified-by-reading-both-sides (data flow); effect size inferred
- What: Upstream TDC integrates `dv_c/dt` from the accepted previous-step
  velocity `mlt_vc_old`, which is fixed for the whole step and restored on retry.
  FlexMESA re-snapshots `mlt_vc_old` from the current `mlt_vc` at every
  `save_start_values`, each preceded by a `set_hydro_vars` that already advanced
  `mlt_vc` by `dt` from the previous `mlt_vc_old`. In one step the TDC velocity is
  therefore advanced several times (remesh path, do_struct_burn_mix pre-solve,
  each retry restore), and after a failed attempt the "old" velocity is
  whatever the failed iterate left behind. The additional guard `vc_old > 0.0`
  means TDC is never used at a face that was radiative at the last snapshot
  (falls through to the non-TDC branch), which upstream does not do.
- Proposed fix: keep a true per-step `mlt_vc_old` set only when a step is
  accepted, restore `mlt_vc` from it in `restore_from_old`, S/M.

### F1-5: `restore_from_old` does not restore centre boundary state, and several per-step quantities upstream restores
- Area: flexmesa-star/evolve
- Severity: wrong numerics (state not restored)
- FlexMESA: `evolve.rs:76-114`; restores `star_age, mstar, lnd, ln_t, ln_r, lum, v, u, xa, dq, et_rsp, hp_face, alpha_rti, omega, j_rot, eps_mdot, mstar_dot, total_energy, cumulative_energy_error`.
- Upstream: `evolve_support.f90:156-178` (`s% M_center = s% M_center_old; s% R_center = s% R_center_old; s% L_center = s% L_center_old; s% v_center = s% v_center_old; ... s% Teff = s% Teff_old; s% dt_limit_ratio = s% dt_limit_ratio_old; s% gradT_excess_alpha = s% gradT_excess_alpha_old; s% crystal_core_boundary_mass = ...`).
- Confidence: verified-by-reading-both-sides
- What: No `*_center_old` fields exist in the crate (grep for
  `m_center_old|r_center_old|v_center_old|l_center_old` returns nothing).
  Upstream moves the inner boundary each try (`R_center = R_center_old +
  dt*v_center`, `evolve.f90:551-557`) and rewinds it on retry; the non-RSP
  step driver has no such update (the only `r_center += dt*v_center` is in
  `rsp_hyd.rs:84`), so models with `v_center /= 0` or a moving core boundary
  (remove_center, RSP-like setups) are not handled, and anything that changes
  `m_center` mid-step is not rolled back. `gradT_excess_alpha`,
  `crystal_core_boundary_mass` and `mlt_vc` (F1-4) are likewise not rewound. On
  the other hand FlexMESA forcibly zeroes `burn_avg_epsnuc` and calls
  `conserve_shell_j_rot_and_distortion` here.
- Proposed fix: add the missing `_old` fields and the `R_center` update, S.

### F1-6: `*_start` thermodynamic arrays are taken after adjust_mass / premix / diffusion, upstream takes most of them at the same point but `xa_start`/`energy_start` handling around split burn differs
- Area: flexmesa-star/struct_burn_mix
- Severity: wrong numerics (minor; needs test)
- FlexMESA: `struct_burn_mix.rs:452-463,483-490` (start `lnd/lnT/lnR/v/csound` captured before split burn and restored after `save_start_values`), `:256-257` (`model.xa_start[...] .copy_from_slice(&out.xa_after)`).
- Upstream: `struct_burn_mix.f90:155` (`call save_start_values`) after `do_burn` (`:106`) and `set_vars_if_needed` (`:112`); `save_start_values` stores `energy_start, lnR_start, L_start, xa_start, xh_start` etc. (`:268-372`).
- Confidence: verified-by-reading-both-sides
- What: Consistent with upstream in that energy_start/xa_start/xh_start are
  post-split-burn. FlexMESA then puts back pre-burn `lnd_start, ln_t_start,
  ln_r_start, v_start, csound_start` while `xh_start` (packed by
  `save_start_values`) stays post-burn; upstream `do_burn` leaves lnd/lnT
  unchanged unless `op_split_burn` variable-T path updates T, in which case
  upstream `xh_start` is post-burn too. The two start representations
  (`xh_start` vs `ln_t_start`) can disagree in FlexMESA for zones burned with
  the variable-T solver. Listed as a divergence to test rather than a proven
  error.
- Proposed fix: single source for start-of-solve state, S.

### F1-7: operator-split pre-physics has hard-coded arguments and swallows failures
- Area: flexmesa-star/evolve_support
- Severity: wrong numerics / maintainability
- FlexMESA: `evolve_support.rs:117-121` (`let _ = do_conv_premix(model, 5, ...)`),
  `:133-134` (`do_phase_separation(model, dt.max(1.0), 175.0, is_one)`),
  `:140-144` (`do_element_diffusion(model, dt.clamp(1.0, 1.0e13), DiffusionScheme::Paquette)`),
  `:194` (`let _ = set_rotation_mixing_info(model);`)
- Upstream: `evolve.f90:593-628` (`call do_conv_premix(s, ierr); if (failed('do_conv_premix')) return` ... `call do_element_diffusion(s, dt, ierr)`), eps_pre_mix `= (e_before - e_after)/dt` at `:594-602`.
- Confidence: verified-by-reading-both-sides for the call sites; callee internals covered in Area 6
- What: (a) premix is capped at 5 iterations by a literal; (b) the phase
  separation threshold Gamma=175 is a literal (upstream uses the EOS phase
  variable and `phase_separation_option`, no Gamma constant in the driver);
  (c) the diffusion timestep is clamped to 1e13 s (3.2e5 yr): for any longer
  step only 1e13 s worth of diffusion is applied, silently; the scheme is fixed
  to Paquette regardless of `diffusion_use_*` controls; (d) every one of these
  can fail upstream and force a retry (`failed(...)` -> `do_step_part2 = retry`),
  FlexMESA discards the return values. Upstream also never calls
  `set_mixing_info` after diffusion (`okay_to_set_mixing_info = .false.`,
  `evolve.f90:614`); FlexMESA re-runs `refresh_mixing_state` inside
  `do_struct_burn_mix` both before and after the solve
  (`struct_burn_mix.rs:478,554`).
- Proposed fix: thread controls and errors through, S each.

### F1-8: remesh gating differs: extra "sharp dt drop" and "recent retry" vetoes, and `remesh_dt_limit` compared with the new dt
- Area: flexmesa-star/evolve_support
- Severity: wrong numerics (behaviour upstream lacks)
- FlexMESA: `evolve_support.rs:285-300` (`sharp_dt_drop = ... dt_try < 0.25 * model.dt_old`; `recent_retry = model.model_number <= m_ret + 1`; `dt_try >= min_remesh_dt`)
- Upstream: `evolve.f90:1972-1974` (`if (s% restore_mesh_on_retry .and. s% model_number_for_last_retry > s% model_number - s% num_steps_to_hold_mesh_after_retry) return`), `adjust_mesh.f90:99` (`if (s% dt < s% remesh_dt_limit)`, evaluated before `s% dt = s% dt_next`, i.e. on the previous dt); also `evolve.f90:1879` no remesh on `doing_first_model_of_run`.
- Confidence: verified-by-reading-both-sides
- What: Upstream skips remesh after a retry only when `restore_mesh_on_retry`
  is set (default off, hold count default 0). FlexMESA always skips the remesh
  for the step(s) after a retry and whenever the new dt is under a quarter of
  the previous dt; neither rule exists upstream. Upstream skips remesh on the
  first model of a run; FlexMESA does not. Mesh histories diverge after the
  first retry.
- Proposed fix: implement upstream gating, S.

### F1-9: step failure reporting carries no zone/equation information; a `Terminate` from the timestep controller is returned as `InvalidInput`
- Area: flexmesa-star/evolve
- Severity: maintainability
- FlexMESA: `evolve.rs:310-313` (`"star_evolve_step failed after {retries}/{max_retries} retries (dt=..., dt_floor=...)"`), `:299` (`return Err(StarError::InvalidInput(reason))`), `:292` (`let _ = restore_from_old(model);`)
- Upstream: `evolve.f90:2231-2236` (`' retry: ' // trim(s% retry_message), s% retry_message_k, s% model_number`) and termination codes (`t_min_timestep_limit`, `t_max_number_retries`).
- Confidence: verified-by-reading-both-sides
- What: The last solver error (which does carry some detail, see Area 2) is
  dropped: when retries are exhausted the user gets only counts and dt, with no
  model number, zone, variable or reason for the final failed attempt. Upstream
  prints a retry reason with zone index per retry. Normal terminations chosen by
  the timestep controller are reported through the "invalid input" error
  variant, and an error from the state restore on that path is ignored.
- Proposed fix: carry the last attempt's error and model number into the
  message; add a termination variant, S.

### F1-10: `max_age` handling: dt truncated with a 1 s floor and a 1e-12 relative stop; upstream's graded approach to max_age absent
- Area: flexmesa-star/evolve
- Severity: wrong numerics (minor)
- FlexMESA: `evolve.rs:424-438` (`if model.star_age >= max_age * (1.0 - 1.0e-12) { break; }` ... `model.dt_next = rem_sec.max(1.0);`)
- Upstream: `evolve.f90:1899-1907,2082-2083` (`s% dt_next = max(0d0, s% max_age*secyer - s% time)`), `:2090-2141` (`num_adjusted_dt_steps_before_max_age`), `max_age_in_seconds`, `max_age_in_days`.
- Confidence: verified-by-reading-both-sides
- What: The final step can overshoot max_age by up to 1 s (floor), and a run
  stops up to `1e-12*max_age` early. `max_age_in_seconds/_days`,
  `num_adjusted_dt_steps_before_max_age`, `force_timestep*`,
  `force_timestep_min*` (`evolve.f90:1909-1920`) are not handled in the driver.
  Age is accumulated as `star_age += dt/SECYER` (`evolve.rs:271`) rather than
  from `time = time_old + dt` in seconds, which loses precision in years for
  second-scale steps on old stars (upstream keeps `time` in seconds).
- Proposed fix: S.

### F1-11: the retry loop re-evaluates `do_set_mdot` from the restored state each try and a private redo loop replaces upstream `select_mdot_action`
- Area: flexmesa-star/evolve
- Severity: unverified claim (not audited line by line)
- FlexMESA: `evolve.rs:224-262` (`mdot_redo.select_action`, `note_solver_failure`, `force_accept_next`, `last_conv_mdot`)
- Upstream: `evolve.f90:747-1058` (`setup_for_implicit_mdot_loop`, `select_mdot_action`)
- Confidence: verified-by-reading-flexmesa-only (structure); `mdot_redo.rs` not compared
- What: Upstream returns `retry` when the solver fails inside the implicit
  mdot loop. FlexMESA instead has two fallbacks with no upstream counterpart: on
  solver failure it asks `note_solver_failure` for another mdot and redoes at
  the same dt, and failing that it re-runs with the last converged mdot and
  force-accepts. These change which steps are retried with smaller dt.
- Proposed fix: compare `mdot_redo.rs` against `select_mdot_action`, M.


---

## Area 2: Newton solver (`solver.rs`, `solver_norms.rs` vs `star_solver.f90`, `solver_support.f90`, `struct_burn_mix.f90`)

Overall: the tolerance *schedule* (tol1/tol2/tol3, gold, gold2 and the iteration
switch points) and its defaults are ported faithfully. Everything else in the
iteration (damping, line search, iteration budget, failure exits, norm
definitions, composition domain checks, post-convergence sanity checks) is a
different algorithm from upstream. The module docstring "Exact upstream MESA
residual (`sizequ`) and correction (`sizeB`) norm evaluators"
(`solver_norms.rs:2`) is not correct.

### Matches (Area 2)

- Three-stage residual tolerance switching by iteration, for standard / gold /
  gold2: `solver_norms.rs:285-313` vs `star_solver.f90:186-225,289-303`.
- Defaults of all `tol_residual_norm{1,2,3}`, `tol_max_residual{1,2,3}`,
  `iter_for_resid_tol{2,3}`, `gold_*`, `gold2_*`, `tol_correction_norm=3e-5`,
  `tol_max_correction=3e-3`, `correction_xa_limit=5e-3`, `xa_scale=1e-5`,
  `include_{L,v,u,w}_in_correction_limits=.true.`,
  `use_gold_tolerances=.true.`, `maxT_for_gold_tolerances=1d99`:
  `controls_defaults.rs:57-58,227-264` vs `star/defaults/controls.defaults:8560-8561,8689-8795,9193-9263`.
- Pass logic `(pass_resid && pass_corr_with_coeff) || (disabled_resid &&
  pass_corr_without_coeff)` with `disabled = tol_max_residual > 1e2 &&
  tol_residual_norm > 1e2`: `solver.rs:377-389` vs `star_solver.f90:487-499`
  (FlexMESA adds a third acceptance path, F1-15; upstream additionally requires
  `.not. disabled_resid_tests` inside `pass_resid_tests`, equivalent here).
- Gold level selection incl. the `maxT_for_gold_tolerances` gate:
  `solver_norms.rs:237-263` vs `struct_burn_mix.f90:531-553` (except the
  `init_model_number` offset, F1-17).
- `x_scale` for lnd, lnT, lnR (`max(1, |xh_start|)`) and species
  (`max(xa_scale, xa_start)`): `hydro_eqns.rs:184-186,216-221` vs
  `solver_support.f90:58-64`.
- `sizeB` exclusions: `i_Hp`, `i_alpha_RTI` always skipped; species with
  `xa_start < correction_xa_limit` skipped; `ignore_species_in_max_correction`;
  RSP2 `w` weight `1/(1e3+|w|)`, zero when `|w|<1`:
  `solver_norms.rs:158-222` vs `solver_support.f90:345-420`.

### F1-12: damping and line search are a different algorithm (always-on halving line search with invented step caps) instead of upstream `scale_correction_norm` + `Bdomain` + `adjust_correction`
- Area: flexmesa-star/solver
- Severity: wrong numerics (convergence path differs every step)
- FlexMESA: `solver.rs:55-57` (`let max_dln = 0.5_f64; let max_dx_chem = 0.25_f64; let eps_neg_xa = 5.0e-5_f64;`), `:101-102` (`coeff.min(min_chem_alpha.max(1.0e-2)); coeff.clamp(1.0e-6, 1.0)`), `:297-334` (`for _bt in 0..7 { ... if iter == 1 || next_rnorm <= 1.5 * prev_residual_norm.max(1.0e-4) { accept } ... alpha *= 0.5; }`), `:336-355` (fallback "best of the 7" if `best_rnorm < 1.0e6`).
- Upstream: `star_solver.f90:390-409` (`if (correction_norm*correction_factor > s% scale_correction_norm) correction_factor = min(correction_factor, s% scale_correction_norm/correction_norm)` ... `correction_factor = max(min_corr_coeff, correction_factor)`), `:219-223` (`if (skip_global_corr_coeff_limit) then min_corr_coeff = 1`), `:842-870` (`f_target = max(fold/2, fold + alf*coeff*slope)`, cubic backtracking on f = 1/2|equ|^2 only when `min_corr_coeff < 1`), `solver_support.f90:623-700` (`Bdomain`); defaults `scale_correction_norm = 0.1d0`, `scale_max_correction = 1d99`, `corr_coeff_limit = 1d-2` (`controls.defaults:9028-9032`).
- Confidence: verified-by-reading-both-sides
- What: Upstream on a first try takes a single trial per iteration with
  `coeff = min(1, 0.1/correction_norm)` (average scaled correction capped at
  0.1), further reduced by `Bdomain` for negative mass fractions, and performs
  no residual-based line search at all (`min_corr_coeff = 1` -> `adjust_correction`
  returns after one evaluation). A residual line search with sufficient-decrease
  test and cubic interpolation is used only after a retry, bounded below by
  `corr_coeff_limit`. FlexMESA never applies `scale_correction_norm`; it instead
  caps per-variable changes at |dlnd|,|dlnT|,|dlnR| <= 0.5 and |dX| <= 0.25
  (constants that do not exist upstream), always runs up to 7 halvings with an
  ad hoc accept rule (first iteration always accepted; later ones accepted if the
  mean residual grows by no more than 1.5x or is below 1.5e-4), and can accept a
  step that *increased* the residual up to 1e6. Consequences: iteration counts
  (which feed the `solver_iters_timestep_limit` timestep limiter), which steps
  fail, and the converged iterate all differ from MESA; each iteration can cost
  up to 8 residual+Jacobian evaluations. `corr_coeff_limit`,
  `scale_correction_norm`, `scale_max_correction`, `solver_reduced_min_corr_coeff`,
  `solver_itermin_until_reduce_min_corr_coeff`, `tiny_corr_coeff_limit` are not
  present in the crate (grep).
- Proposed fix: port `do_solver_work`/`adjust_correction`/`Bdomain` literally, M.

### F1-13: iteration budget and forced-iteration rules differ (`max_newton_iters` vs `solver_max_tries_before_reject` family; no `solver_itermin`)
- Area: flexmesa-star/solver
- Severity: wrong numerics
- FlexMESA: `solver.rs:248-252` (`if gold_level == 2 { max_newton_iters.max(40) } else { max_newton_iters.max(1) }`), `inlist_populate_limits.rs:17-22` (`get_i64("solver_max_tries_before_reject").or_else(|| ctrl.get_i64("max_tries1"))` -> `max_newton_iters`), default 25 (`controls_defaults.rs:56`).
- Upstream: `star_solver.f90:267-279` (`doing_first_model_of_run -> max_tries1`; `retry_cnt > 20/10/5 -> max_tries_after_{20,10,5}_retries`; `retry_cnt > 0 -> max_tries_for_retry`; else `solver_max_tries_before_reject`), defaults 250/75/50/35/25/25 (`controls.defaults:8897-8947`); `:581-594` + `solver_support.f90:772-796` (`if (iter < itermin) force_another_iteration = 1`, `solver_itermin = 2`, plus the accretion surface-entropy revision).
- Confidence: verified-by-reading-both-sides
- What: (a) The first model of a run gets 25 iterations instead of 250; the
  growing budgets after 5/10/20 retries are absent; gold2 gets a floor of 40
  that upstream does not have. (b) The doc comment maps `max_newton_iters` to
  `max_tries1` (`controls.rs:102`) but the inlist reader lets either name set
  it, so an inlist setting only `max_tries1 = 250` raises the budget for every
  step. (c) Upstream never accepts convergence before iteration 2
  (`solver_itermin`) and forces extra iterations for the newly accreted surface
  (`max_num_surf_revisions`); FlexMESA can accept after one iteration. Solver
  iteration counts are not comparable with MESA.
- Proposed fix: S.

### F1-14: residual and correction norms are not upstream's `sizequ`/`sizeB` definitions
- Area: flexmesa-star/solver_norms
- Severity: wrong numerics (convergence criterion values differ) / unverified claim ("exact")
- FlexMESA: `solver_norms.rs:67-69` (`return compute_vector_norms(equ);` plain mean over all `nvar*nz`), `:161-168` (`dx = (dx_scaled*x_scale).abs(); denom = (lum_surf_ref + model.lum[k].abs())`), `:185` (`acc.push(dx / (0.1 * cs_s + vel_k.abs()).max(1.0))`), `:215-231` (species: only a per-cell mean, `sum_xa_corr`/`num_xa_terms` reset each cell, individual terms not added to the sum).
- Upstream: `solver_support.f90:211-228` and `:265-277` (with `do_burn .or. do_mix`: `num_terms = nvar*nz` then `+ nvar_chem*nz`, and the chem residuals are summed in both loops, i.e. counted twice); `:331-345` (`s% correction_weight(s% i_lum,k) = 1d0/(frac*s% L_start(1) + abs(s% L(k)))`) applied at `:398` as `abs(B(j,k)*s% correction_weight(j,k))` where `B` is the *scaled* correction; `:361-362,417-427` (`num_xa_terms`, `sum_xa_corr` initialised once outside `cell_loop`, never reset; running mean added per cell) and `:428-455` (`species_loop` adds every species term with `xa_start >= x_limit` again individually); `:556-566` (`max_correction = 1d99` when the max correction grows three iterations running at the same zone and variable).
- Confidence: verified-by-reading-both-sides
- What: Four concrete differences. (1) Upstream's `equ_norm` with chemistry on
  is `(sum_hydro + 2*sum_chem)/((nvar+nvar_chem)*nz)`; FlexMESA uses
  `(sum_hydro+sum_chem)/(nvar*nz)`. (2) For L, v, u, w upstream multiplies the
  already x_scale-normalised correction by the weight (so the term is
  `dL/(x_scale*(0.1 L1 + |L|))`, numerically negligible); FlexMESA un-scales first
  and uses `dL/(0.1 L1+|L|)`, which is the physically sensible quantity but makes
  L and v corrections *binding* on convergence where upstream's are not. The L and
  v `x_scale` values also differ (`hydro_eqns.rs:178-195`:
  `max(Lsun, 1e-3 Lmax, Lmax*q, |L_start|)` and `min(r/dt, csound)`; upstream
  `max(1, |xh_start|)`, `solver_support.f90:61`), which changes the Jacobian
  column scaling. (3) The species contribution to `correction_norm` is
  different (upstream: cumulative running mean + every individual term;
  FlexMESA: per-cell mean only). (4) The "max correction growing at the same
  place" kill switch is missing. Upstream (1)-(3) look like accidents of history
  rather than intent, but they define what MESA's tolerances mean; with the same
  tolerance numbers FlexMESA converges to a different criterion.
- Proposed fix: either port literally (S) or keep and stop calling it exact; record in DECISIONS.

### F1-15: extra convergence acceptance path and legacy-tolerance override that upstream lacks
- Area: flexmesa-star/solver, solver_norms
- Severity: wrong numerics (minor)
- FlexMESA: `solver.rs:384-390` (`tight_resid = residual_norm <= 1.0e-12 && max_residual <= 1.0e-10 && pass_corr_without_coeff` -> converged), `solver_norms.rs:272-283` (`has_custom_scalar_tol` -> returns `c.tol_residual_norm, c.tol_max_residual` for all iterations, bypassing gold and the 3-stage schedule), defaults `tol_residual_norm: 1.0e-4, tol_max_residual: 1.0e-2` (`controls_defaults.rs:59-60`).
- Upstream: `star_solver.f90:497-499` (only the two-way test); upstream has no `tol_residual_norm` / `tol_max_residual` controls (only `...1/2/3`).
- Confidence: verified-by-reading-both-sides
- What: The `tight_resid` path ignores the damping coefficient on the correction
  test. The override is triggered by a floating-point comparison of control
  values against magic defaults (`(c.tol_residual_norm - 1.0e-4).abs() > 1e-15`),
  so setting a non-upstream scalar silently disables gold tolerances.
- Proposed fix: delete both, S.

### F1-16: no composition domain checks inside the solver (`min_xa_hard_limit`, `sum_xa_hard_limit`), mass fractions are clamped instead of triggering a retry
- Area: flexmesa-star/solver
- Severity: wrong numerics (behaviour upstream lacks: silent clamp)
- FlexMESA: `solver.rs:149-150` (`.clamp(-5.0e-5 - x_start, 1.0 + 5.0e-5 - x_start)`), `:162-179` (`normalize_species_fractions`: clamp to [0,1] then divide by the sum, once after the loop, for converged and non-converged results alike). `min_xa_hard_limit`, `sum_xa_hard_limit`, `xa_clip_limit` do not occur in the crate.
- Upstream: `solver_support.f90:1399-1414` (`if (s% xa(j,k) < min_xa_hard_limit) ... ierr = -1; s% why_Tlim = Tlim_neg_X`), `:1432-1445` (`if (abs(sum_xa - 1d0) > sum_xa_hard_limit) ... ierr = -1`), `:1446-1450` (renormalise every iteration when `> 1d-12`); defaults `min_xa_hard_limit = -1d-5`, `sum_xa_hard_limit = 5d-4` (`controls.defaults:9049,9085`); `Bdomain` `eps = -0.5d0*min_xa_hard_limit` (`solver_support.f90:669`).
- Confidence: verified-by-reading-both-sides
- What: In MESA a mass fraction below -1e-5 or a sum off by more than 5e-4 at
  any iteration rejects the step (retry with smaller dt, plus
  `neg_mass_fraction_hold`). FlexMESA lets mass fractions go to -5e-5 (5x the
  upstream hard limit), clips anything beyond, never retries on this, and
  renormalises at the end whatever the sum error was. Clipping and
  renormalising silently creates/destroys mass of individual species; it is also
  applied after the solve, so the returned state no longer satisfies the
  converged equations. The damping threshold (`eps_neg_xa = 5e-5`, `solver.rs:57`)
  is 10x upstream's `-0.5*min_xa_hard_limit = 5e-6`.
- Proposed fix: port `check1_chem` and the `Bdomain` chem loop, S.

### F1-17: missing solver exits and guards: `check_after_converge`, jump limits, "avg corr too large", high-T correction tolerances, `init_model_number` offset
- Area: flexmesa-star/solver, struct_burn_mix
- Severity: wrong numerics
- FlexMESA: no counterpart found (grep for `check_after_converge`, `negative cell volume`, `corr_norm_jump_limit`, `tol_correction_norm_high_t`, `corr_param_factor` in non-test sources returns nothing); `solver_norms.rs:240-243` (`mnum > c.steps_before_use_gold2_tolerances`); `solver_norms.rs:314` (`(rn, mr, c.tol_correction_norm, c.tol_max_correction)`).
- Upstream: `struct_burn_mix.f90:695-738` (`if (s% lnR(k) <= s% lnR(k+1)) ... converged = .false.`; logT>12, logT<1, logRho>12, logRho<-30 reject), `:483-500` (`set_tol_correction`: `T_max >= 1d9 -> tol_correction_norm_high_T = 3d-3, tol_max_correction_high_T = 3d-1`; `>= 6d9 -> 8d-3, 8d-1`), `:534-535` (`s% model_number > s% steps_before_use_gold2_tolerances + max(0,s% init_model_number)`), `star_solver.f90:379-385` (`correction_norm > s% corr_param_factor*s% scale_correction_norm` -> 'avg corr too large'), `:536-555` (jump limits, tiny corrections), `:563-566` ('coeff too small').
- Confidence: verified-by-reading-both-sides (upstream), verified-by-reading-flexmesa-only for absence
- What: (a) A "converged" FlexMESA model is never checked for non-monotonic
  radius or out-of-range T/rho. (b) Above 1e9 K upstream relaxes the correction
  tolerances by 100x (and ~270x above 6e9 K); FlexMESA keeps 3e-5/3e-3, so
  advanced-burning models are held to a far tighter criterion than MESA and will
  retry where MESA proceeds. (c) Upstream aborts an iteration early when the mean
  scaled correction exceeds `10*0.1 = 1` (except first model); FlexMESA iterates
  on. (d) `steps_before_use_gold*_tolerances` is measured from model 0, not from
  `init_model_number`.
- Proposed fix: S each.

### F1-18: solver error reporting has no zone / variable / model number
- Area: flexmesa-star/solver
- Severity: maintainability
- FlexMESA: `solver.rs:363-365` (`"line search failed at Newton iteration {iter}"`), `:199-203` (`"block-tridiagonal linear solve failed: {e}"`); `HenyeySolverReport` (`:27-40`) has norms only. `NormAcc` (`solver_norms.rs:12-36`) does not track the arg-max.
- Upstream: `solver_support.f90:227-231` (`j_max = j; k_max = k` for the max residual), `:404-409` (`max_zone`, `max_var`), `star_solver.f90:502-535` (prints model number, which tolerance failed, values), `evolve.f90:2231-2236` (retry message with zone).
- Confidence: verified-by-reading-both-sides
- What: When a solve does not converge, nothing identifies the zone, equation
  or variable with the largest residual/correction, nor the model number or dt.
  Non-convergence within the iteration budget is returned as `Ok(report)` with
  `converged=false` and no reason. A NaN in the residual makes the line search
  fail with the same generic message.
- Proposed fix: track arg-max in the norms and put (model, iter, zone, eqn/var name, value, tolerance) in the report and error, S.

### Not found / not applicable (Area 2)
- Jacobian reuse: neither side reuses the Jacobian between iterations (upstream
  refactors every iteration, `star_solver.f90:331`). FlexMESA re-evaluates the
  Jacobian on every line-search trial (`eval_henyey_equations_into` at
  `solver.rs:307`), so no stale-Jacobian issue.

---

## Area 3: structure equations and partials

Read on both sides: `hydro_eqns.rs` (density eqn, surface BCs, chem eqns,
assembly), `hydro_momentum.rs`, `hydro_energy.rs`, `hydro_work.rs`,
`hydro_temperature.rs`, `eps_grav.rs`, the MLT/TDC call site in
`hydro_vars.rs:150-345`; upstream `hydro_eqns.f90`, `hydro_momentum.f90`,
`hydro_energy.f90`, `hydro_temperature.f90`, `eps_grav.f90`,
`hydro_chem_eqns.f90`, relevant helpers in `star_utils.f90`. NOT read line by
line: the rest of `hydro_vars.rs` vs `hydro_vars.f90` (set_vars ordering,
Teff/atm coupling, `set_Teff_info_for_eqns`), `hydro_rti.rs`, Riemann (`u_flag`)
equations, RSP2 equations.

General assessment: the default hydrostatic/`v_flag` equation set is a
recognisable and mostly faithful transcription of the upstream residual forms,
scalings and index conventions; I found no off-by-one in face/cell indexing in
the default paths. The divergences are concentrated in optional terms
(mlt_Pturb, eps_grav form, alternative T-gradient forms, RSP L surface BC,
relax-time equation switching) and in the convection model default.

### Matches (Area 3)

- Density/volume equation `lnR(k) - ln(r(k+1)^3 + 3 dm/(4 pi rho))/3`, centre
  uses `R_center^3`: `hydro_eqns.rs:244-272` vs `hydro_eqns.f90:565-571`
  (FlexMESA adds a mass-correction factor on `dm`, see F1-25).
- Surface P BC `(lnP_bc - lnP1)/max(1, lnP1)` with
  `dP0 = G m dm/(8 pi r^4)`, zero when `use_momentum_outer_BC`; surface T BC
  `(lnT_bc - lnT1)/max(1,lnT1)` with `dT0 = dP0*gradT*T/P`:
  `hydro_eqns.rs:293-303,340-374` vs `hydro_eqns.f90:855-862,1022-1024,1054-1056`.
- Compression BC `(rho2*dlnd1 - rho1*dlnd2)/dt` and fixed-vsurf BC
  `(v - fixed_vsurf)/csound_start(1)`: `hydro_eqns.rs:305-338` vs
  `hydro_eqns.f90:1091-1096,1119`.
- Momentum equation residual
  `((grav + other + RTI)*dm_face/A - dPtot - d_mlt_Pturb)/Ptot_avg`, face mass
  `(dm(k)+dm(k-1))/2` (k=1: `dm/2`), time-centred area
  `4pi(r^2 + r r0 + r0^2)/3` and `1/(r r0)`, drag `-c v/dt`, RTI diffusion and
  kick terms, surface variant with `P_surf` in place of `P(k-1)`:
  `hydro_momentum.rs:86-105,154-170,172-222,224-290,292-316` vs
  `hydro_momentum.f90:128-141,165-186,240-266,300-345`,
  `star_utils.f90:3671-3688`.
- Momentum-equation composition partials
  `(dP00 - resid*alfa*dP00)/Pavg`, `(-dPm1 - resid*beta*dPm1)/Pavg`:
  `hydro_momentum.rs:341-388` vs `hydro_momentum.f90:268-287,385-395`.
- Radius equation `v*dt/r_start - expm1(lnR - lnR_start)` with optional time
  centring of v: `hydro_momentum.rs:426-441` vs `hydro_momentum.f90:590-594`.
- Energy equation term selection for the three branches (eps_grav form, simple
  PdV form, total-energy form) and the sign of every term:
  `hydro_energy.rs:398-490` vs `hydro_energy.f90:92-117`.
- `dL/dm` with `L_theta` time centring and `L_center` at the centre:
  `hydro_energy.rs:56-86` vs `hydro_energy.f90:164-180`.
- Sources: `eps_nuc` (with d/dlnd, d/dlnT) or `burn_avg_epsnuc` for split-burn
  zones, time-centred `0.5*(non_nuc_neu_start + non_nuc_neu)`, `extra_heat`,
  premix/phase-separation heating gates, `irradiation_heat`, `eps_mdot`, RTI
  energy diffusion: `hydro_energy.rs:125-250` vs `hydro_energy.f90:181-262`.
- Face work `A*v*P_face` with `P_face = alfa*P(k) + beta*P(k-1)` (k=1:
  alfa=1), `u_face = (r - r_start)/dt` when there is no velocity variable,
  centre work `4 pi R_c^2 P_start(nz) v_c`, and the dxa partial factors:
  `hydro_work.rs:96-126,129-246,249-271` vs `hydro_energy.f90:592-755,757-784`.
- Energy-equation scaling `min(1 or 1e-6 at k=1, frac*dedt_eqn_r_scale)*dt/e_start`:
  `hydro_energy.rs:369-389` vs `star_utils.f90:3691-3710` (denominator of the
  fraction differs, F1-25).
- Default T-gradient equation `delm*dlnPdm_qhse*gradT - (T(k-1)-T(k))/Tpoint`:
  `hydro_temperature.rs:167-190` vs `hydro_temperature.f90:270-281,405-470`.
- dPrad/dm form incl. flux limiter `lambda = (6+3R)/(6+(3+R)R)`:
  `hydro_temperature.rs:66-122` vs `hydro_temperature.f90:60-104` (Lrad
  condition differs, F1-23).
- Chemistry equation: residual `(dxdt_mix + dxdt_nuc - dxdt_actual)/eqn_scale`,
  `eqn_scale = max(min_chem_eqn_scale, x_scale/dt)`, tridiagonal mixing
  partials, burn partials; no extra floors beyond `1e-30` guards. There is no
  `mix_scale` floor: `hydro_eqns.rs:386-500` vs `hydro_chem_eqns.f90:93-160`,
  `mix_info.f90:2303-2364`.
- `sig` definition `mix_factor*D*(4 pi r^2 rho_face)^2 / (xmstar*(dq00+dqm1)/2)`
  and the `sig_term_limit*xmstar*min(dq)/dt` cap applied only when
  `max(sig) >= 1`: `mix_info.rs:359-394` vs `mix_info.f90:1415-1453`
  (the high-Tcenter `sig_min_factor` limiter at `:1455-1560` is not ported, and
  the two control values are passed as literals `1.0, 1.0e13` at
  `struct_burn_mix.rs:429` instead of reading `sig_term_limit`).

### F1-19: default convection model is Cox MLT; upstream default at 0b2e6bb is `MLT_option = 'TDC'`
- Area: flexmesa-star/controls_defaults, hydro_vars
- Severity: wrong physics (default mismatch)
- FlexMESA: `controls_defaults.rs:26` (`mlt_option: MltOption::Cox,`); only changed when the inlist names it (`inlist_populate.rs:378-384`)
- Upstream: `star/defaults/controls.defaults:2203` (`MLT_option = 'TDC'`), with `TDC_alpha_D = 1d0`, `TDC_alpha_R = 0d0`, `TDC_alpha_Pt = 0d0`, `steps_before_use_TDC = 0` (`:2218-2223`)
- Confidence: verified-by-reading-both-sides
- What: Every inlist that does not set `MLT_option` runs time-dependent
  convection in upstream at this commit and steady-state Cox MLT in FlexMESA.
  For slowly evolving models the two agree closely, but they are different
  equations (TDC carries `mlt_vc_old` and dt), so any claim of parity with
  upstream on a case relying on the default is comparing different physics.
  Several per-case hooks additionally force Cox
  (`tier1_extras.rs:398`, `tier3_extras.rs:243`, `tier4_extras.rs:211`,
  `tier5_extras.rs:338`).
- Proposed fix: set the default to TDC once F1-4/F1-20 are fixed; record in DECISIONS, S.

### F1-20: TDC call site: silent fallback to static MLT when `mlt_vc_old <= 0`, at `k = 0`, or when the TDC solve errors; TDC/Henyey parameters hard-coded
- Area: flexmesa-star/hydro_vars
- Severity: wrong physics (TDC only)
- FlexMESA: `hydro_vars.rs:225-229` (`if mlt_opt == MltOption::Tdc && model.dt > 0.0 && k > 0 && vc_old > 0.0`), `:243-249` (`tdc_alpha_d: 1.0, ... tdc_alpha_pt: 0.0, tdc_alpha_c: 1.0, tdc_alpha_s: 1.0`), `:262` (`max_conv_vel: 1.0e99, include_mlt_corr_to_tdc: true`), `:288` (`set_tdc(&tdc_params, &tdc_in).ok().map(|r| r.output)`), `:296-298` (`henyey_nu: 8.0, henyey_y: 1.0 / 3.0, max_conv_vel: 1.0e99`), `:251-261` (`scale: ... .max(1.0e20)`)
- Upstream: `turb_info.f90` `do1_mlt_2` passes `s% TDC_alpha_D, TDC_alpha_R, TDC_alpha_Pt, TDC_alpha_C, TDC_alpha_S`, `s% mlt_vc_old(k)` as `conv_vel_start` for all faces, `max_conv_vel` from `max_conv_vel_div_csound*csound_face`, and `Henyey_MLT_y_param`/`Henyey_MLT_nu_param` (controls at `controls.defaults:2218-2222,2319-2320,3363`). (Call-site argument list not quoted line by line.)
- Confidence: verified-by-reading-flexmesa-only for the fallback logic; upstream controls verified in defaults
- What: With TDC selected, a face whose previous convective velocity is zero
  is evaluated with instantaneous MLT (through `set_mlt` with
  `option: Tdc`), so convection switches on at full MLT strength in one step
  rather than growing on the TDC timescale; that is the regime TDC exists for.
  A failing TDC solve is swallowed (`.ok()`) and replaced by MLT with no
  message. `TDC_alpha_D/Pt/C/S`, `Henyey_MLT_*`, and `max_conv_vel_div_csound`
  set in an inlist have no effect here.
- Proposed fix: pass `vc_old` through unconditionally, propagate errors, read
  controls, S/M.

### F1-21: `mlt_Pturb` pressure difference uses different face velocities for the two cells (upstream uses `mlt_vc_old(k)` for both)
- Area: flexmesa-star/hydro_momentum, hydro_temperature
- Severity: wrong physics (only when `mlt_Pturb_factor > 0`; default 0)
- FlexMESA: `hydro_momentum.rs:282-288` (`let pturb_00 = eval_mlt_pturb_cell_ad(model, k, false); let pturb_m1 = eval_mlt_pturb_cell_ad(model, k - 1, true); ... (pturb_m1 - pturb_00)`), with `eval_mlt_pturb_cell_ad` (`:37-51`) using `mlt_vc_old[k]` of the cell index passed and returning 0 when it is `<= 0`; same pattern in `hydro_temperature.rs:21-24`.
- Upstream: `hydro_momentum.f90:189-194` (`d_mlt_Pturb_ad = s% mlt_Pturb_factor*pow2(s% mlt_vc_old(k))*(rho_m1 - rho_00)/3d0`), `hydro_temperature.f90:447-449` (`mlt_Ptrb00 = ...pow2(s% mlt_vc_old(k))*wrap_d_00(s,k)/3d0; mlt_Ptrbm1 = ...pow2(s% mlt_vc_old(k))*wrap_d_m1(s,k)/3d0`), gated there on `include_mlt_Pturb_in_thermodynamic_gradients`.
- Confidence: verified-by-reading-both-sides
- What: Upstream's turbulent-pressure jump across face k is
  `vc_old(k)^2 (rho(k-1) - rho(k))/3`. FlexMESA computes
  `(rho(k-1) vc_old(k-1)^2 - rho(k) vc_old(k)^2)/3`. At the top of a convective
  region (`vc_old(k-1) = 0`, `vc_old(k) > 0`) this is `-rho(k) vc_old(k)^2/3`,
  the full turbulent pressure instead of a small density-difference term: a
  spurious force at every convective boundary. The T-gradient equation adds the
  term without the upstream `include_mlt_Pturb_in_thermodynamic_gradients`
  gate. The work term (`hydro_work.rs:214-238`) does use the face form
  correctly.
- Proposed fix: use `mlt_vc_old[k]` (face k) for both sides, S.

### F1-22: RSP luminosity outer BC is a factor 4 below upstream
- Area: flexmesa-star/hydro_temperature
- Severity: wrong physics (only `use_RSP_L_eqn_outer_BC` without RSP2)
- FlexMESA: `hydro_temperature.rs:36-40` (`let sigma_sb = 0.25 * CLIGHT * CRAD; let l_rad_surf = (0.5 * FOUR_PI * sigma_sb) * (r_0 * r_0) * t_0.pow4();`) with the comment "Upstream `set_RSP_Lsurf_BC`: `L1 = 0.5 * 4 * pi * r(1)^2 * (c * a / 4) * T(1)^4`"
- Upstream: `hydro_temperature.f90:361` (`rhs_ad = s%RSP2_Lsurf_factor * area_ad * clight * (crad * pow4(T_surf))`), `star/defaults/controls_dev.defaults:68` (`RSP2_Lsurf_factor = 0.5d0`), `controls.defaults:4523` ("solves L1 - s% RSP2_Lsurf_factor*4*pi*r^2*c*a*T1^4 = 0"); scale `maxval(s% L_start(1:nz))` (`:367`).
- Confidence: verified-by-reading-both-sides
- What: Upstream sets `L1 = 0.5 * 4 pi r^2 * c a T^4` (= 2 * 4 pi r^2 sigma T^4).
  FlexMESA sets `L1 = 0.5 * 4 pi r^2 * (c a/4) T^4`, i.e. 4x smaller, and the
  comment misquotes upstream. The factor is a literal rather than
  `RSP2_Lsurf_factor`, the area is not time-centred, and the residual is scaled
  by `max(|L_start(1)|, Lsun)` instead of `max(L_start)`. The RSP2 path
  (`rsp2_ad.rs:419`, `area * t4_00 * (lsurf_factor * CLIGHT * CRAD)`) has the
  upstream form, so the two FlexMESA implementations disagree with each other.
- Proposed fix: S.

### F1-23: `eps_grav` form: lnS blending, latent-heat centring and the composition term differ from upstream
- Area: flexmesa-star/eps_grav
- Severity: wrong physics (when `energy_eqn_option = 'eps_grav'`)
- FlexMESA: `eps_grav.rs:183-194` (`alfa_lns = f_pc * w_gam` with `w_gam = ((model.gam[k] - gam_off)/gam_span).clamp(0,1)`), `:79-86` (latent heat subtracted after the 0.5/0.5 centring, current values only), `:88-118` (`eps_grav += (-e_ad * dlne_dx + p_div_rho_chirho * dlnp_dx) * dx_dt` summed over species, `p_div_rho_chirho = P*chiT/(rho*chiRho)`), `:209-229` (`d_eps_grav_dx = scale_comp*(-e_k*dlne_dx + p_div_rho_chirho*dlnp_dx)`).
- Upstream: `eps_grav.f90:66-78` (`using_PC = (s% eos_frac_PC(k) > 0)`; switch on `s% gam_start(k)`; `alfa = (Gamma - full_off)/(full_on - full_off)`), `:113-121` (`eps_grav = eps_grav - (dlnd_dt*latent_ddlnRho + dlnT_dt*latent_ddlnT)` before centring, `eps_grav_start` includes `latent_*_start`), `:176-232` (composition term `-de/dt` with `de = e - e(xa_start; rho, T)` from a second EOS call, time-centred with `e(xa; rho_start, T_start) - e_start`), `:187-190` (`d_eps_grav_dx = -energy*dlnE_dxa/dt + (Peos/Rho)*dlnPeos_dxa*dxh_lnd/dt`).
- Confidence: verified-by-reading-both-sides
- What: (1) Upstream's lnS/standard blend weight depends only on start-of-step
  Gamma once any PC fraction is present; FlexMESA multiplies by the PC fraction
  and uses the current Gamma (which adds an untracked dependence on the
  iterate). (2) Latent heat: upstream averages current and start latent
  coefficients; FlexMESA uses current at full weight. (3) The composition term
  is not upstream's: upstream evaluates the EOS at the start composition and
  takes the finite energy difference (exact for any composition change);
  FlexMESA uses a first-order expansion in dX with an additional
  `P chiT/(rho chiRho) dlnP/dX` term that does not appear in upstream's value
  at all (upstream has a different P-term only in the *partial*
  `d_eps_grav_dx`), and applies no time centring. For large composition
  changes (flashes, mixing events) the two differ at first order.
- Proposed fix: port `eval_eps_grav_composition` with the two EOS calls, M.

### F1-24: energy-equation form is not switched to eps_grav during relax (`no_dedt_form_during_relax`), and `u_flag` override missing from the selection
- Area: flexmesa-star/hydro_energy
- Severity: wrong numerics
- FlexMESA: `hydro_energy.rs:398-490` selects on `model.controls.energy_eqn_option` only; `no_dedt_form_during_relax` does not occur in the crate (grep). `inlist_populate.rs:420-422` maps any unrecognised string to `Dedt` (`_ => EnergyEqnOption::Dedt`).
- Upstream: `hydro_energy.f90:323-327` (`if (.not. eps_grav_form) then; if (s% doing_relax .and. s% no_dedt_form_during_relax) eps_grav_form = .true.`), default `no_dedt_form_during_relax = .true.` (`controls.defaults:7900`); `hydro_eqns.f90:112-118` (`call mesa_error(...,'Invalid choice for energy_eqn_option')`).
- Confidence: verified-by-reading-both-sides
- What: All upstream relaxation phases (relax mass, composition, to radiative
  core, etc.) solve the eps_grav form by default; FlexMESA's relax paths solve
  the dedt form (and use a different eps_grav, F1-23, if asked). A misspelt
  `energy_eqn_option` is silently accepted as 'dedt' where upstream stops.
- Proposed fix: S.

### F1-25: smaller structure-equation divergences (each verified-by-reading-both-sides)
- Area: flexmesa-star/hydro_*
- Severity: wrong numerics (minor) / behaviour upstream lacks
- Items:
  1. Mass corrections in the continuity equation and surface offset:
     `hydro_eqns.rs:253-258` (`dr3 = (3.0 * model.dm[k] * mc / FOUR_PI) / rho_00`),
     `:288-303` (`dp_0 = cgrav*m_grav*fp_0*dm*mc_0/(8 pi r^4)`); upstream
     `hydro_eqns.f90:565` (`dr3 = (s% dm(k)/rho)/(4d0*pi/3d0)`) and `:857`
     (`dP0 = s% cgrav(1)*s% m_grav(1)*s% dm(1)/(8*pi*pow4(r))`) have neither
     `mass_correction` nor `fp_rot`. Affects `use_mass_corrections` and rotating
     models.
  2. `do1_gradT_eqn` (`use_gradT_actual_vs_gradT_MLT_for_T_gradient_eqn`):
     FlexMESA uses `(P(k-1)-P(k))/P_face` and `(T(k-1)-T(k))/T_face`
     (`hydro_temperature.rs:51-63`); upstream uses log differences
     `lnT_m1 - lnT_00`, `lnPeos_m1 - lnPeos_00` (`hydro_temperature.f90:181-183`).
  3. dPrad/dm form: `Lrad = L*gradT/gradr` applied whenever
     `gradr > 0 && |gradr-gradT| > 1e-5|gradr|` (`hydro_temperature.rs:77-85`);
     upstream additionally requires `lnT/ln10 <= max_logT_for_mlt` and
     `mlt_mixing_type` convective or semiconvective
     (`hydro_temperature.f90:62-69`).
  4. Radius equation zero-velocity forcing: FlexMESA
     `logt < velocity_logt_lower_bound` (`hydro_momentum.rs:404-406`) lacks
     upstream's `.and. s% dt < secyer*s% max_dt_yrs_for_velocity_logT_lower_bound`
     (`hydro_momentum.f90:573-576`), uses current rather than start `csound`,
     and the momentum equation lacks the matching `accel = 2 v/dt` branch
     (`hydro_momentum.f90:239-246`).
  5. Drag heating: both faces gated on `q[k] > min_q_for_drag` and applied at
     the centre cell (`hydro_energy.rs:225-239`); upstream gates face k on
     `q(k)` and face k+1 on `q(k+1)` and skips `k == nz`
     (`hydro_energy.f90:232-250`).
  6. Diffusion heating: `diff_heat = if eps_wd == 0.0 { eps_diff } else { eps_wd }`
     (`hydro_energy.rs:176-184`) ignores `do_element_diffusion`,
     `do_WD_sedimentation_heating`, `do_diffusion_heating`; upstream
     `hydro_energy.f90:205-211`.
  7. Split-burn zones: `burn_avg_epsnuc.filter(|&v| v != 0.0).unwrap_or(model.eps_nuc[k])`
     (`hydro_energy.rs:139-146`): a legitimately zero average (burn skipped for
     infall, `max_abar_for_burning`, `min_logT_for_net`) is replaced by the
     instantaneous `eps_nuc` while the composition is held fixed; upstream uses
     `burn_avg_epsnuc(k)` unconditionally (`hydro_energy.f90:194-196`).
  8. Energy scaling denominator: `total_internal_energy_start` recomputed as
     `sum(max(e_start,0)*dm)` (`hydro_eqns.rs:520-522`,
     `hydro_energy.rs:373-384`); upstream `s% total_internal_energy_old`
     (`star_utils.f90:3706`).
  9. Time-centred geometry in the momentum/T-gradient equations requires
     `v_flag || u_flag` (`hydro_momentum.rs:58-62`); upstream keys only on
     `using_velocity_time_centering` (`star_utils.f90:3681`). Differs only if
     `steps_before_use_velocity_time_centering >= 0` without a velocity
     variable (default -1).
  10. `Pvsc` is not averaged with `Pvsc_start` in the work term when
      `include_P_in_velocity_time_centering` (`hydro_work.rs:186-196` vs
      `hydro_energy.f90:677-690`); `extra_pressure` (`use_other_pressure`) is
      absent everywhere.
  11. PE/KE time differences use plain f64 (`hydro_work.rs:13-65,68-93`);
      upstream evaluates `PE_new - PE_start` and `KE_new - KE_start` in quad
      precision (`star_utils.f90:2245-2290,2365-2382`, `real(qp)`) because of
      cancellation. With GM/r ~ 1e15 erg/g the f64 rounding is ~0.1 erg/g per
      difference, i.e. ~0.1/dt erg/g/s of noise in the total-energy form:
      irrelevant for year-scale steps, comparable to physical terms for
      second-scale hydrodynamic steps.
  12. `eval_equ` face-reconstruction paths (`use_face_reconstruction`),
      `fix_d_eos_dxa_partials`, `w_div_wc` and `j_rot` equations
      (`hydro_eqns.f90:592-720`) have no counterpart in the assembly loop
      (`hydro_eqns.rs:541-562`).
- Proposed fix: S each.

### F1-26: equation assembly cannot fail on bad numbers and reports no equation/zone
- Area: flexmesa-star/hydro_eqns
- Severity: maintainability
- FlexMESA: `hydro_eqns.rs:259-264` is the only failure path (`"non-positive inner volume at zone {k}: {}"`); momentum, energy, T-gradient and chem routines return `()`; surface BC clamps instead of failing (`:341` `.max(StarAdOrder1::constant(1.0e-30))`, `:365` `.max(StarAdOrder1::constant(1.0))`).
- Upstream: every `do1_*` sets `s% retry_message = 'error in do1_...'` (`hydro_eqns.f90:150-290`), `is_bad(residual)` checks in each equation, `mesa_error` on bad `P_bc`/`T_bc` (`:865-870`).
- Confidence: verified-by-reading-both-sides
- What: A NaN residual is only noticed later as a failed line search with a
  generic message (F1-18). Clamping a non-positive `P_surf + dP0` or
  `T_surf + dT0` to a tiny positive number hides an atmosphere failure and makes
  the BC derivative zero.
- Proposed fix: return `Result` with (equation name, zone, value), S.

---

## Area 4: timestep control (`timestep.rs`, `timestep_limits.rs`, `timestep_checks.rs` vs `timestep.f90`)

Read fully on the FlexMESA side: `timestep.rs`, `timestep_limits.rs`, first
330 lines of `timestep_checks.rs` (`check_change`, `check_integer_limit`,
`check_dx_limits`, `check_lg_x_cntr_limits`, `check_dx_nuc_drop_full`, start of
`check_structural_limits`). Upstream: `timestep_controller`,
`do_timestep_limits`, `check_integer_limit`, `check_solver_iters_timestep_limit`,
`check_dX`, `check_change`, `check_delta_mdot`, `check_delta_mstar`,
`check_rel_error_in_energy`, the two Courant checks, `check_varcontrol_limit`,
`eval_varcontrol`, `filter_dt_next`. NOT compared line by line: the
dlgP/dlgRho/dlgT/dlgE/dlgR, lgL_*, Teff, centre and HR limiters in
`timestep_checks.rs:330-571` and `check_dX_nuc_drop` internals.

### Matches (Area 4)

- `eval_varcontrol`: variables (lnd, lnT, lnR only), 5-point smoothing for
  interior, 2:1 weighted edge terms, the `k=2` 3-point term, the lnd `/3`,
  `sumscales = sum max(1,|xh_old(j,1)|)`, `nterms = 3*nz`:
  `timestep.rs:46-92` vs `timestep.f90:2315-2345`. FlexMESA also reproduces
  upstream's omission of the `k = nz-1` 3-point term, which upstream computes
  (`timestep.f90:2338-2339`) and never adds (upstream bug, shared).
- H211 low-pass controller
  `limiter(limiter(1/r)^(1/4) * limiter(1/r_old)^(1/4) * (dt/dt_old)^(-1/4))`
  with `limiter(x) = 1 + 10 atan((x-1)/10)`, first-order fallback
  `dt/dt_limit_ratio` when no history: `timestep.rs:40-44,125-153` vs
  `timestep.f90:2376-2393,2421-2425`.
- `dt_limit_ratio = max over limiters`, limiters at or below 1 contribute 0,
  varcontrol contributes its raw ratio: `timestep_limits.rs:266-284`,
  `timestep_checks.rs:13-35` vs `timestep.f90:330,766-771`.
- `check_change` (time_delta_coeff scaling, hard limit -> retry, `r <= 1 -> 0`)
  and `check_integer_limit` (`value/limit - 0.05`): `timestep_checks.rs:13-54`
  vs `timestep.f90:737-771,384-409`.
- `max_timestep_factor` / `_at_high_T` / `min_timestep_factor` clamps and the
  relabelling of the reason only when varcontrol was limiting:
  `timestep_limits.rs:286-322` vs `timestep.f90:333-356`.
- `skip_hard_limit` definition: `timestep_limits.rs:139-145` vs
  `timestep.f90:105-111`.
- `timestep_hold` veto of increases, hard maximum timestep:
  `timestep_limits.rs:349-369` vs `timestep.f90:55-68`, `evolve.f90:2056-2063`.
- `varcontrol_dt_limit_ratio_hard_max` retry and
  `min_allowed_varcontrol_target` terminate: `timestep_limits.rs:122-137,147-163`
  vs `timestep.f90:2265-2291`.
- Limiters present on both sides (names as upstream): varcontrol, solver iters,
  burn steps, diffusion steps/iters, dX and dX_div_X, dL_div_L, dlgP, dlgRho,
  dlgT, dlgE, dlgR, lgL_H/He/z/photo/nuc, dlgTeff, dlgRho_cntr, dlgT_cntr,
  dlgT_max, dlgP_cntr, lg_XH/XHe/XC_cntr, delta_mstar, delta_mdot, delta_lgL,
  delta_HR, cell collapse, min_dr_div_cs, energy error, dX_nuc_drop.
- Defaults that agree: `min_timestep_factor 0.8`, `max_timestep_factor 1.2`,
  `max_timestep_factor_at_high_T 1.1`, `timestep_factor_for_retries 0.5`,
  `retry_hold 1`, `retry_limit 100`, `use_dt_low_pass_controller`,
  `never_skip_hard_limits`, `solver_iters_timestep_limit 7 / 14 / 18`,
  `delta_lg_star_mass_limit 5d-3`, energy-error limits `1d-4 / 1d-3`,
  `min_timestep_limit 1d-6`
  (`controls_timestep_stop.rs:340-461` vs `controls.defaults:9442-9753,11236-11298`).

### F1-27: `varcontrol_target` default is 1e-4; upstream is 1e-3
- Area: flexmesa-star/controls_defaults
- Severity: wrong numerics (default mismatch)
- FlexMESA: `controls_defaults.rs:55` (`varcontrol_target: 1.0e-4,`)
- Upstream: `star/defaults/controls.defaults:9636` (`varcontrol_target = 1d-3`), `:9648` (`min_allowed_varcontrol_target = 1d-4`)
- Confidence: verified-by-reading-both-sides
- What: The primary timestep controller target is 10x tighter than upstream for
  any inlist that does not set it; with the H211 controller this means
  systematically smaller steps (roughly 10x where varcontrol is the limiter),
  so step counts, model numbers at a given age, and anything keyed on model
  number differ from MESA. One per-case hook sets 1e-3 explicitly
  (`tier4_extras.rs:248`), which suggests the mismatch was noticed for a case
  rather than fixed at the default.
- Proposed fix: S.

### F1-28: the energy-conservation timestep limiter and hard-limit retry are inert because of F1-1
- Area: flexmesa-star/timestep_limits
- Severity: wrong numerics
- FlexMESA: `timestep_limits.rs:42-54` (`check_change((sh.error_in_energy_conservation / sh.total_energy_end).abs(), ts.limit_for_rel_error_in_energy_conservation, ts.hard_limit_for_rel_error_in_energy_conservation, ...)`) fed by `evolve_support.rs:270`
- Upstream: `timestep.f90:2037-2042`; defaults on (`limit_for_rel_error_in_energy_conservation = 1d-4`, `hard_limit_... = 1d-3`, `controls.defaults:11297-11298`)
- Confidence: verified-by-reading-both-sides
- What: The check is ported correctly, but its input is the sum of converged
  solver residuals (F1-1), which is orders of magnitude below 1e-4 of the total
  energy. In upstream this limiter is active by default and both reduces dt and
  forces retries when the true energy audit is off by more than 1e-3. In
  FlexMESA it can never fire, and with `energy_eqn_option='eps_grav'` its input
  is identically zero.
- Proposed fix: follows from F1-1.

### F1-29: all limiters run on the first model of a run; upstream applies only varcontrol there
- Area: flexmesa-star/timestep_limits
- Severity: wrong numerics (minor)
- FlexMESA: `timestep_limits.rs:176-264` (no first-model guard; only `apply_delta_mdot_limits` skips its hard limit for `model_number <= 1`, `evolve.rs:373`)
- Upstream: `timestep.f90:130` (`if (.not. s% doing_first_model_of_run) then` ... all other checks)
- Confidence: verified-by-reading-both-sides
- What: On the first step after a load, `*_old` values (L, Teff, powers) are
  not yet meaningful; upstream skips every limiter except varcontrol. FlexMESA
  evaluates them, which can cut dt or force a hard-limit retry on step 1.
- Proposed fix: S.

### F1-30: controller details that differ
- Area: flexmesa-star/timestep_limits, timestep_checks, evolve
- Severity: wrong numerics (minor) / behaviour upstream lacks
- Confidence: verified-by-reading-both-sides
- Items:
  1. With `use_dt_low_pass_controller = .false.` FlexMESA applies the atan
     limiter (`timestep_limits.rs:303`, `filter_limit(1.0 / dt_limit_ratio...)`);
     upstream uses the bare first-order `dt/dt_limit_ratio`
     (`timestep.f90:2394-2396`).
  2. Solver-iteration limit selection uses the raw control flags
     (`timestep_limits.rs:177-183`: `if c.use_gold2_tolerances {...} else if c.use_gold_tolerances`);
     upstream uses `s% using_gold_tolerances` as actually applied in this solve
     (maxT gate, steps_before, `init_model_number`) and subtracts the extra
     accretion iteration (`timestep.f90:438-446`). Combined with F1-12/F1-13
     the iteration count itself is not comparable.
  3. `force_timestep_min*`: FlexMESA bumps `dt_next` to
     `min(dt*max(factor,1), force_min)` from the *previous* dt inside the
     controller (`timestep_limits.rs:334-346`); upstream applies
     `dt = min(dt*force_timestep_min_factor, force_timestep_min)` to the new dt
     in `prepare_for_new_step` (`evolve.f90:1909-1915`).
  4. High-T max timestep: FlexMESA takes a non-sticky `min` each step
     (`timestep_limits.rs:356-361`); upstream switches
     `max_years_for_timestep` permanently once `max T >= max_timestep_hi_T_limit`
     (`evolve.f90:707-714`).
  5. `delta_mdot` limiter is applied after the controller with its own
     re-filtering and a different hard-limit skip rule
     (`evolve.rs:345-431`: `skip_hard = model_number <= 1 || (retries > 0 && relax_hard_limits_after_retry && !never_skip_hard_limits)`);
     upstream treats it as one more entry in the same `maxloc` with the common
     `skip_hard_limit` (`timestep.f90:296-297,1889-1917`). The max-timestep and
     reason bookkeeping are not re-applied after this adjustment.
  6. `check_dX`: (a) upstream's species tests for `'X'` and `'Y'` are
     `if (cid /= ih1 .or. cid /= ih2 .or. cid /= ih3) cycle` and
     `if (cid /= ihe4 .or. cid /= ihe3) cycle` (`timestep.f90:548-551`), which
     are always true, so in upstream `dX_limit_species = 'X'` or `'Y'` never
     limits anything (upstream bug). FlexMESA implements the intended meaning
     (`timestep_checks.rs:119-123`: `"X" => nuc.z == 1, "Y" => nuc.z == 2`), so
     those limits bind in FlexMESA and not in MESA. (b) Upstream indexes
     `s% dX_decreases_only(j)` by species index j (`timestep.f90:557`, another
     upstream slip); FlexMESA uses the control index i
     (`timestep_checks.rs:129`). (c) The high-T variants
     `dX_div_X_at_high_T_limit*` (`timestep.f90:508-514`) and the
     `min_D_mix` cutoff (`:526-530`) are not ported (`D_mix > 0` is used,
     `timestep_checks.rs:103`). These "silent corrections" of upstream are not
     in the tolerance/discrepancy ledger as far as this audit saw (not
     searched exhaustively).
  7. Courant limiter: plain minimum of `dr/cs` over all cells
     (`timestep_limits.rs:60-73`); upstream `min_dr_div_cs` restricts by
     `min_k_for_dt_div_min_dr_div_cs_limit = 20`, `min_q/max_q = 0.005/0.995`
     and `min_abs_du_div_cs... = 0.01` (`controls.defaults:10592-10613`;
     `star_utils.f90` routine not read).
  8. `eval_varcontrol` returns `0.0` when the `*_old` arrays have a different
     length from the current model (`timestep.rs:77-84`), i.e. "no change",
     letting dt grow by `max_timestep_factor`; upstream stops
     (`timestep.f90:2313`, `mesa_error('not associated xh_old')`).
  9. Limiters absent: `check_lgL_nuc_cat_change`, `check_dlgT_max_at_high_T_change`,
     `check_dYe_highT_change`, `check_dlog_eps_nuc_change`, `check_dX_div_X_cntr`,
     `check_lg_X{Ne,O,Si}_cntr`, `check_X{H,He,C,Ne,O,Si}_cntr`,
     `check_adjust_J_q`, `other_timestep_limit` (`timestep.f90:136-320`; absence
     judged from the limiter names present in `timestep_checks.rs` /
     `timestep_limits.rs`).
- Proposed fix: S each.

### F1-31: retry/limit messages lack zone and value
- Area: flexmesa-star/timestep_checks
- Severity: maintainability
- FlexMESA: `timestep_checks.rs:27` (`format!("{msg} hard limit")`), `:132` (`format!("dX {} hard limit", nuc.name)`)
- Upstream: `timestep.f90:757-759` (`s% retry_message = trim(msg) // ' hard limit'; s% retry_message_k = max_k`), `:563-566` (`s% Tlim_dX_species = j; s% Tlim_dX_cell = k`)
- Confidence: verified-by-reading-both-sides
- What: The offending zone, the measured value and the limit are not carried
  in the retry reason; the retry reason string is then discarded by the driver
  (`evolve.rs:284-286`, `TimestepAction::Retry(_)`).
- Proposed fix: S.

---

## Area 8: defaults (done before Areas 5-7 because it is mechanical)

Method: a script matched every `name: value,` initialiser in
`controls_defaults.rs`, `controls_timestep_stop.rs`, `controls_mix_wind.rs`,
`job_controls.rs`, `rsp_config.rs` against `name = value` in upstream
`star/defaults/controls.defaults` and `controls_dev.defaults` by lower-cased
name (script: `a scratch script (not kept)
Only controls whose Rust field name equals the upstream name are covered;
controls renamed in Rust, hard-coded literals and controls that do not exist in
FlexMESA are not. `Option::None` vs upstream sentinel values (1d99, -1, 0) were
treated as equal and are not listed.

### F1-32: default-value mismatches (verified-by-reading-both-sides via script, each line re-read)
- Area: flexmesa-star/controls_*
- Severity: wrong physics / wrong numerics (per item)

| control | FlexMESA | upstream (controls.defaults line) | affects |
| :-- | :-- | :-- | :-- |
| `MLT_option` | `Cox` (`controls_defaults.rs:26`) | `'TDC'` (2203) | convection (F1-19) |
| `varcontrol_target` | `1e-4` (`controls_defaults.rs:55`) | `1d-3` (9636) | timestep (F1-27) |
| `screening_mode` | `Extended` (`controls_defaults.rs:49`) | `'chugunov'` (6657) | reaction rates |
| `do_premix_heating` | `false` (`controls_defaults.rs:183`) | `.true.` (2923) | energy eqn |
| `do_phase_separation_heating` | `false` (`controls_defaults.rs:69`) | `.true.` (7434) | energy eqn |
| `conv_premix_avoid_increase` | `true` (`controls_mix_wind.rs:358`) | `.false.` (2940) | premix |
| `delta_lgP_cntr_limit` | `0.05` (`controls_timestep_stop.rs:435`) | `-1` (10719) | timestep: a limiter upstream has off |
| `delta_lg_XH_drop_only`, `_XHe_`, `_XC_` | `true` (`controls_timestep_stop.rs:443,448,453`) | `.false.` (11209-11211) | timestep |
| `P_function_weight` | `30` (`controls_mix_wind.rs:417`) | `40` (5930) | mesh |
| `max_surface_cell_dq` | `1e-8` (`controls_mix_wind.rs:415`) | `1d-12` (5858) | mesh |
| `max_center_cell_dq` | `1e-6` (`controls_mix_wind.rs:416`) | `1d-7` (5845) | mesh |
| `min_D_mix` | `0.0` (`controls_defaults.rs:33`) | `1d3` (3585) | mixing (only with `set_min_D_mix`) |
| `atm_T_tau_errtol` | `1e-4` (`rsp_config.rs:291`) | `1d-7` (4316) | surface BC |
| `Lnuc_div_L_zams_limit` | `0.99` (`controls_defaults.rs:97`) | `0.9d0` (1796) | ZAMS stop |
| `accretion_h1`, `accretion_he4` | `-1.0` (`controls_defaults.rs:83-84`) | `0` (5521, 5524) | accretion composition (meaning of the sentinel not checked) |
| `log_center_temp_upper_limit` | none (`controls_defaults.rs:108`) | `11d0` (970) | termination |
| `log_center_density_upper_limit` | none (`controls_defaults.rs:110`) | `12d0` (957) | termination |
| `fe_core_infall_limit` | none (`controls_defaults.rs:133`) | `3d7` (1688) | termination |
| `pulse_data_format` | `"GYRE"` (`job_controls.rs:459`) | `'FGONG'` (571) | output |
| `column_depth_for_irradiation` | `0.0` (`controls_defaults.rs:72`) | `-1` (4565) | irradiation gate |

- Not mismatches although the numbers differ: `velocity_q_upper_bound 1.0` vs
  `1d99`, `velocity_tau_lower_bound 0` vs `-1d99`, `velocity_logT_lower_bound 0`
  vs `-1d99`, `max_logT_for_include_P_and_L_in_velocity_time_centering 99` vs
  `1d99` (same behaviour for physical values).
- Hard-coded where upstream has a control (found while reading Areas 1-4):
  `sig_term_limit = 1e13` and `mix_factor = 1` literals
  (`struct_burn_mix.rs:429`; equal to upstream defaults),
  `clean_mixing_glitches(model, 1.0, 0.1)` (`struct_burn_mix.rs:419`),
  premix iteration cap 5, phase-separation Gamma 175, diffusion dt cap 1e13 s
  (F1-7), TDC and Henyey parameters (F1-20), solver damping constants (F1-12).
- Solver-related controls with no FlexMESA counterpart (so their upstream
  defaults cannot be honoured): see F1-12, F1-13, F1-16, F1-17.
- Proposed fix: align defaults; add a test that diffs the Rust defaults
  against the upstream `.defaults` files, S.

---

## Area 5: mesh (partial: planner and mesh-function configuration only)

Read: `mesh_adjust.rs:1-290` (config, `eval_mesh_gval_parts`, `plan_mesh`).
Upstream: `mesh_functions.f90:94-200` (function list), `mesh_plan.f90` outline
and the lines quoted below. NOT read: `mesh.rs` (apart from
`save_start_values`/`save_old_values`), `mesh_remap.rs` (conservative remap of
xa/energy/j), upstream `mesh_adjust.f90` (2789 lines), `adjust_mesh.f90`,
`adjust_mesh_support.f90`. Conservation properties of the remap were therefore
not audited.

### Matches (Area 5)
- P and T mesh functions `weight*log10(P)`, `weight*log10(T)`; R function
  `weight*log10(1 + (r/Rsun)/param)`; M function likewise; `gam_function` tanh
  form: `mesh_adjust.rs:147-185` vs `mesh_functions.f90` (function list at
  `:125-170`; individual formulae not all re-read).
- Defaults `T_function1_weight 110`, `max_dq 1d-2`, `min_dq 1d-14`,
  `mesh_max_allowed_ratio 2.5`, `max_delta_x_for_merge 0.1`,
  `min_dq_for_xa_convective 1d-6`: `controls_mix_wind.rs:407-418` vs
  `controls.defaults:5640-5762,5943`.

### F1-33: the mesh planner is a different algorithm from upstream `do_mesh_plan`
- Area: flexmesa-star/mesh_adjust
- Severity: wrong numerics (mesh differs from MESA every remesh)
- FlexMESA: `mesh_adjust.rs:237-243` (`non_xa_split = g_non_xa[k] > 1.0 || dq_k > max_dq || surf_over || cen_over`), `:258-276` (`n_eff = (g_k.ceil() as usize).clamp(2, 4)` equal subcells), `:279-289` (`if k >= 2 && k + 3 < nz && mixing_type[k] == mixing_type[k+1] && g_k + gval[k + 1] < 0.45 && dq_k + dq[k+1] <= max_dq` -> merge the pair)
- Upstream: `mesh_plan.f90:601-950` (`pick_new_points`: sweeps from the surface choosing each new cell width with `pick_next_dq`), `:1122` (`dgnew = min(delta_gval_max(k_old), delta_gval_max(k_old+1))`), `:636` (`next_dq_max = dq_new(k_new-1)*mesh_max_allowed_ratio`), `:525-600` (`smooth_new_points` with `alfa = mesh_max_allowed_ratio`), `:724-800` (merge rules incl. `max_delta_x_for_merge`, convective/crystal boundaries), `:151-160` (centre/surface cell caps).
- Confidence: verified-by-reading-both-sides (structure); upstream internals read only at the quoted lines
- What: Upstream builds the new grid by marching in mass and choosing each new
  cell so that every mesh function changes by at most `delta_gval_max` across
  it, limited to `mesh_max_allowed_ratio` times the previous cell, then smooths
  neighbouring ratios; new cells need not align with old cell boundaries.
  FlexMESA treats each old cell independently: split into 2-4 equal parts when
  its own function jump exceeds 1, merge with the next cell when the two jumps
  sum to less than 0.45 (a constant with no upstream counterpart), with no
  neighbour-ratio enforcement in the part read. `mesh_max_allowed_ratio` and
  `max_delta_x_for_merge` are carried in the config struct but are referenced
  nowhere else in `mesh_adjust.rs`, `mesh.rs` or `mesh_remap.rs` (grep), i.e.
  they are dead controls. Zone counts and placement will not match MESA.
- Proposed fix: port `pick_new_points`/`pick_next_dq`/`smooth_new_points`, L.

### F1-34: mesh-function configuration differs: composition function applied to all species with the maximum weight; several upstream mesh functions missing
- Area: flexmesa-star/mesh_adjust
- Severity: wrong numerics
- FlexMESA: `mesh_adjust.rs:72-81,97-98` (`xa_function_weight: if max_xa_w > 0.0 { max_xa_w } else { 20.0 }`, `xa_p ... unwrap_or(1.0e-3)`), `:187-195` (loop `for j in 0..model.species`), `:40,49` (`p_function_weight: 30.0`, `min_dq: 1.0e-12, max_dq: 1.0e-1` in `Default`), `controls_mix_wind.rs:424-426` (`xa_function_species: Vec::new()` etc.), `:415-417` (F1-32 rows)
- Upstream: `mesh_functions.f90:169-174` (one gval per listed `xa_function_species(k)` with its own weight/param), `controls.defaults:6127-6129` (`xa_function_species(1) = 'he4'`, weight `30`, param `1d-2`), function list `E, P, T1, T2, R, R2, R3, M, gradT, log_tau, log_kap, gam, omega, convective_bdy` (`:125-168`)
- Confidence: verified-by-reading-both-sides
- What: By default upstream refines on helium-4 only, with weight 30 and
  offset 1e-2. FlexMESA, with an empty species list, refines on every species
  in the network with weight 20 and offset 1e-3, so trace-species gradients
  drive splits that MESA would not make; a user list selects the weight (max
  over the list) but not the species. `E_function`, `T_function2`,
  `R_function2/3`, `log_tau_function`, `log_kap_function`,
  `convective_bdy_weight`, `mesh_logX_species`/`mesh_dlogX_dlogP_extra`, the
  high-T `mesh_delta_coeff_for_highT` scaling and `use_other_mesh_functions`
  are not represented in the config struct.
- Proposed fix: M.

---

## Areas 6 and 7: mixing; mass change and winds: NOT audited line by line

I ran out of budget before a side-by-side read of `mix_info.rs`,
`overshoot.rs`, `premix.rs`, `diffusion.rs`, `omega_mix.rs`,
`adjust_mass.rs`, `winds.rs`, `winds_driver.rs`, `eps_mdot.rs` against
`mix_info.f90`, `overshoot*.f90`, `conv_premix.f90`, `predictive_mix.f90`,
`turb_info.f90`, `element_diffusion.f90`, `adjust_mass.f90`, `winds.f90`,
`eps_mdot.f90`. What was established in passing (all already cited above):

- `sig` construction and the `sig_term_limit` cap match upstream; the high-Tc
  `sig_min_factor_for_high_Tcenter` limiter (default 0.01, active above
  `Tcenter_max_for_sig_min_factor_full_off`) is absent
  (`mix_info.rs:359-394` vs `mix_info.f90:1455-1560`).
  verified-by-reading-both-sides.
- Call-site divergences for premix / phase separation / element diffusion and
  the extra `refresh_mixing_state` calls: F1-7.
- `do_adjust_mass_step` (`adjust_mass.rs:510-598`) has the recognisable shape
  of upstream `revise_q_and_dq` (k_below_const_q / k_const_mass selection by q
  and logT, the `qfrac` 0.67-1.50 / 0.50-2.00 window with the dominant-species
  test). Not compared line by line. verified-by-reading-flexmesa-only.

---

## Leads (not verified)

- `hydro_vars.rs:181-183`: `if k == 0 && model.tau[0] < 2.0/3.0 && model.tau[0] > 0.0 { gradr_f = gradr_f * (1.0 + 2.0/(3.0*model.tau[0])); }`.
  Upstream multiplies `gradr` by `s% gradr_factor(k)` (`turb_info.f90:133-141`),
  which is tied to `use_T_tau_gradr_factor` (default `.false.`,
  `controls.defaults:4606`) and to the chosen T(tau) relation, at all k.
  FlexMESA appears to apply an Eddington-specific factor at the surface face
  whenever the surface tau is below 2/3, regardless of the control. Where
  upstream sets `gradr_factor` was not read.
- Rotation gating of the gradr `ft/fp` factor uses
  `use_gravity_rotation_correction` (`hydro_vars.rs:154-165`) where upstream
  uses `mlt_use_rotation_correction` (`turb_info.f90:132`).
- TDC `scale` argument `max(|L(1)|, |L(k)|, 1e20)` (`hydro_vars.rs:251-261`)
  vs upstream's scale in `turb_info.f90` (not read).
- `mdot_redo.rs` vs `select_mdot_action` (`evolve.f90:788-1058`): not compared
  (F1-11).
- `evolve_support.rs:373-382`: `update_mlt_sticky_and_sig_after_hydro` has an
  `else if model.aux.solver_had_mlt_flip && eval_cnt >= 3` branch that freezes
  non-convective faces with `T > 1e6` after any MLT flip, with no control
  gating; upstream's sticky-gradr logic is keyed on
  `make_gradr_sticky_in_solver_iters` / `min_logT_for_...` only (defaults off,
  `controls.defaults:2335-2336`). Upstream `mlt_info`/`turb_info` sticky code
  not read to confirm there is no such unconditional branch.
- `relax_pre_ms_to_radiative_core` (`evolve_support.rs:416-476`): fixed
  600-step loop, overrides atm option to Eddington T-tau, `pextra_factor = 2`,
  `max_timestep_factor = 2`, and silently stops on the first failed step
  (`.is_err() { break; }`); upstream `relax.f90:3295-3408` not compared.
- (Checked, not a divergence) `premix_omega`: the driver passes the full dt
  (`evolve_support.rs:190-192`) and the callee halves it
  (`rotation_mix.rs:394`, `if premix_omega { 0.5 * dt }`), matching
  `do_solve_omega_mix(s, 0.5d0*dt)` (`evolve.f90:634`, `struct_burn_mix.f90:237-241`).
- `eval_total_energy_integrals` excludes `e_rot` from `e_tot`
  (`evolve_support.rs:98`); upstream includes rotational energy only when
  `include_rotation_in_total_energy` (`star_utils.f90:2419`); the integral
  routine itself was not compared.
- Star age accumulation in years (`evolve.rs:271`) vs upstream seconds
  (F1-10): effect on restart parity claims not measured.

---

## Coverage of this audit

| Area | Status |
| :-- | :-- |
| 1 step driver / retry | read both sides in full for `evolve.rs`, `evolve_support.rs`, `struct_burn_mix.rs` driver vs `evolve.f90` (parts 1/2, prepare/retry/redo, energy audit), `struct_burn_mix.f90`, `evolve_support.f90:set_current_to_old`. `mdot_redo.rs`, `finish_step`, `star_lib`/`run_star_support` loop not read. |
| 2 Newton solver | read both sides in full (`solver.rs`, `solver_norms.rs`; `star_solver.f90` main loop and `adjust_correction` start, `solver_support.f90` xscale/sizequ/sizeB/Bdomain/check1_chem). `set_vars_for_solver` not read. |
| 3 structure equations | default-path equations read both sides; `hydro_vars` only at the MLT/TDC call site; RTI, Riemann, RSP2 not read. |
| 4 timestep | controller, varcontrol, filter, generic checks, dX read both sides; ~half of the individual limiters not compared. |
| 5 mesh | planner and config only; remap/conservation not read. |
| 6 mixing | not audited (only `sig`). |
| 7 mass change / winds | not audited. |
| 8 defaults | mechanical comparison for same-named controls in five files; renamed or absent controls not covered. |

No cargo/pytest was run; every statement about behaviour is from reading code.
Effect sizes are inferred unless stated.
