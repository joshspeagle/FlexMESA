# F4: flexmesa-star mixing, mass change, rotation, mesh remap, termination/history

> **Reference note (`#141`):** Line numbers, commit diffs, and pre-rebaseline measurements in this historical audit register refer to MESA `main` at commit `0b2e6bb` (2026-11-04) and `FlexMESA` commit `91eda0b`. Active repository inputs, module goldens, `case_table.md`, and `defaults_diff_26_4_1.md` target **MESA release `26.4.1`** (tag `26.4.1`, commit `c8c2026`, Zenodo `19722306`).

Auditor scope: parts of `crates/flexmesa-star/src` not reached by F1, compared with
upstream `star/private/*.f90` at `0b2e6bb`. FlexMESA paths are relative to
`crates/flexmesa-star/src/` unless they start with `crates/`. Upstream paths are
relative to `star/private/` unless stated. No code was run (per brief); the one
data file consulted is `runs/fm/hb_2M.tsv`.

---

## Area 1: mixing coefficients

### Call graph actually executed in a normal `star_evolve_step`

`evolve.rs:232-240` -> `do_adjust_mass_step` -> (`refresh_mixing_state` only if
`mstar_dot != 0`) -> `apply_step_pre_physics` (`evolve_support.rs:113-205`:
conv_premix, phase separation, element diffusion, rotation) ->
`do_struct_burn_mix` (`struct_burn_mix.rs:501-558`) -> `set_hydro_vars`
-> `run_struct_burn_mix_core` -> `refresh_mixing_state` (`:396-437`) -> Henyey
solve with `sig` frozen -> `set_hydro_vars` -> `refresh_mixing_state` again.

`refresh_mixing_state` is FlexMESA's `set_mixing_info`. Its order is:
rotation reset, `clean_mixing_glitches(model, 1.0, 0.1)`,
`apply_predictive_mixing_from_controls`, `locate_convection_boundaries`,
`apply_overshooting`, rotation augment, `compute_convection_sigmas(model, dt, 1.0, 1.0e13)`, RTI.
Upstream `set_mixing_info` (`mix_info.f90:48-420`) order is: copy MLT results,
7 glitch routines, `do_mix_envelope`, zero `conv_vel` off convection,
`set_cz_boundary_info`, `locate_convection_boundaries`, `add_predictive_mixing`,
`locate_convection_boundaries` again, `locate_mixing_boundaries`,
`add_overshooting`, `add_RTI_turbulence`, `set_cz_bdy_mass`, the three
`set_min_D_mix*` floors, `other_D_mix`, rotation, `D_mix_zero_region`/H-He/H-C
crossover zeroing.

So predictive mixing IS reachable from a normal run (`struct_burn_mix.rs:421-423`,
gated by `do_mix && mix_factor > 0` and any `predictive_mix(i)` set). conv_premix,
phase separation and element diffusion are reachable through
`apply_step_pre_physics` (`evolve_support.rs:120,131,144`) whenever their
controls are on. All four are the placeholder algorithms of C1-16.

### F4-1: gradL composition term is smoothed with the wrong kernel, with no threshold, and leaks across the convective boundary
- Area: flexmesa-star / `hydro_state.rs`, `hydro_vars.rs`
- Severity: wrong physics
- FlexMESA: `hydro_state.rs:453-474` (`let wt = (half_width + 1 - dj.unsigned_abs()) as f64;`, loop `for k in 1..nz`), `hydro_state.rs:476-485` (`smooth_triangular_signed(&raw_b, mw.num_cells_for_smooth_gradl_composition_term)`), consumer `hydro_vars.rs:187-192` (`gradl_f = grada_f + b_term`). `threshold_for_smooth_gradL_composition_term` and `threshold_for_smooth_brunt_B` do not exist in the crate (grep over `controls*.rs`, `inlist_populate*.rs`: only the two `num_cells_*` are parsed, `inlist_populate_phase5b.rs:133-138`).
- Upstream: `hydro_vars.f90:1124-1135` (`call threshold_smoothing(s% smoothed_brunt_B, s% threshold_for_smooth_gradL_composition_term, s% nz, s% num_cells_for_smooth_gradL_composition_term, preserve_sign, work)`), `star_utils.f90:2846-2900` (`threshold_smoothing`: "only smooth contiguous regions where |dd| >= dd_thresh", each region smoothed in isolation), `star_utils.f90:2807-2841` (`weighted_smoothing`: weights built by `weight(j) = weight(j) + weight(j-1)` repeated `2*ns` times, i.e. binomial coefficients C(2ns, j); end points `i = 1` and `i = n` of each region untouched), `brunt.f90:95-104` (same for `brunt_B`).
- Confidence: verified-by-reading-both-sides
- What: three differences. (a) Kernel: upstream is binomial (for `ns = 3`: 1,6,15,20,15,6,1), FlexMESA is triangular (1,2,3,4,3,2,1), so even with default controls every `gradL` and every smoothed `brunt_B` differs from MESA. (b) Threshold: inlist values are silently dropped. (c) Region isolation: upstream never spreads B outside the contiguous run of faces where `|B| >= threshold`; FlexMESA spreads it `ns` faces to each side everywhere. With `use_Ledoux_criterion` and a composition jump at a convective-core edge, FlexMESA puts `B > 0` on up to `ns` faces inside the homogeneous core, making `gradL > grada` there, which pulls the Ledoux boundary inward and (with `alpha_semiconvection = 0`) leaves those faces unmixed. For `hb_2M` (`num_cells = 10`, thresholds `0.02`/`0.1`, Ledoux) the smear is 10 faces each side of the He-core edge.
- Proposed fix: port `weighted_smoothing` and `threshold_smoothing`, add the two threshold controls. S.

### F4-2: `hb_2M` lifetime (71 vs 136 Myr): most plausible cause is that the core never grows, because "predictive mixing" tests a Ledoux gradient that upstream predictive mixing ignores, plus F4-1
- Area: flexmesa-star / `premix.rs`, `struct_burn_mix.rs`
- Severity: wrong physics
- FlexMESA: call site `struct_burn_mix.rs:421-423`; body `premix.rs:187-249`: `let superad = (model.gradr[k_bdy] - grad_crit_bdy).max(0.0); if superad < superad_thresh.max(0.0) { continue; }` (`:212-215`), `gradr_pred = gradr[k_adj] * kap_ratio + 0.5 * (gradr[k_bdy] - gradr[k_adj]).max(0.0)` (`:230-231`), `grad_crit_adj = if use_ledoux { model.gradl[k_adj] } else { model.grada[k_adj] }; if gradr_pred > grad_crit_adj` (`:232-237`), one face marked per boundary (`:240-243`). Selection `premix.rs:271-295` passes only `ingestion_factor`, `superad_thresh`, `bdy_q_min`, `bdy_q_max`.
- Upstream: `predictive_mix.f90` `add_predictive_mixing` / `do_predictive_mixing`: mixes the trial region to homogeneity, re-evaluates EOS, opacity, `gradr`, `grada` on the mixed composition (`eval_mixing_coeffs`), and extends the boundary while `gradr/grada - 1 > predictive_superad_thresh`; composition gradients are removed by construction, so the test is Schwarzschild on the mixed state. Selection honours `predictive_zone_type`, `predictive_zone_loc`, `predictive_bdy_loc`, `predictive_avoid_reversal`, `predictive_limit_ingestion` (`predictive_mix.f90` criteria loop; controls in `controls.defaults`).
- Confidence: verified-by-reading-flexmesa-only for the mechanism (upstream predictive_mix read only at routine level, C1-16 covers it); the causal attribution to the 71 Myr result is `inferred` (no run made).
- What: `hb_2M` sets `use_Ledoux_criterion = .true.`, `predictive_mix(1) = .true.` core, `predictive_avoid_reversal(1) = 'he4'`, `alpha_semiconvection` unset (0). In real MESA the core grows 0.12 -> 0.24 Msun because predictive mixing repeatedly ingests He-rich layers until the mixed-state boundary is Schwarzschild-neutral. In FlexMESA the candidate face `k_adj` is the first face outside the core, exactly where the composition jump sits, so `gradL[k_adj] = grada + B` with B large (and widened by F4-1); the surrogate `gradr_pred` uses unmixed opacities and has no composition update, so `gradr_pred > gradL[k_adj]` is rarely or never true. When it is true only one face per parameter set per call is added, and `d_mix`/`mixing_type` are rebuilt from MLT at the next `set_hydro_vars` (`hydro_vars.rs:423-425`), so nothing accumulates. `zone_type`, `zone_loc`, `bdy_loc`, `avoid_reversal`, `limit_ingestion` are parsed (`inlist_populate_phase5b.rs:163-172`) and never read (grep: no reader outside the parser), so both `hb_2M` sets (core and surf) are applied to every boundary. Net: the He-burning core keeps roughly its Ledoux size, burns a smaller fuel reservoir, and exhausts early. The FlexMESA history is consistent with this: `center_he4 < 0.03` already at 42 Myr (`runs/fm/hb_2M.tsv` model 121), the remaining 30 Myr is a tail to the `1e-4` stop. Contributing: F4-1 (boundary pulled inward), F4-3 (radiative faces latched during the solve), F4-5 (invented glitch cleanups), and no semiconvection in this case on either side.
- Proposed fix: port `predictive_mix.f90`. L. Until then, reject inlists with `predictive_mix(i) = .true.` rather than run the surrogate.

### F4-3: invented solver-iteration rule latches every radiative face with T > 1e6 K after any face flips twice
- Area: flexmesa-star / `evolve_support.rs`
- Severity: wrong numerics
- FlexMESA: `evolve_support.rs:375-383` (`} else if model.aux.solver_had_mlt_flip && eval_cnt >= 3 { for k in 0..nz { if model.t[k] > 1.0e6 && model.mixing_type[k] != ConvectiveMixing { model.aux.fixed_gradr_sticky[k] = true; } } }`), flip counter `:360-366`.
- Upstream: `turb_info.f90:57-62` (`if (make_gradr_sticky_in_solver_iters .and. s% solver_iter > 3) then ... fixed_gradr_for_rest_of_solver_iters(k) = (s% mlt_mixing_type(k) == no_mixing)`), default `make_gradr_sticky_in_solver_iters = .false.`, `min_logT_for_make_gradr_sticky_in_solver_iters = 1d99` (`controls.defaults:2335-2336`).
- Confidence: verified-by-reading-both-sides
- What: the first branch (`sticky_ctrl && eval_cnt > 3`) matches upstream apart from counting MLT evaluations instead of `solver_iter`. The second branch has no upstream counterpart and is active with default controls: once any single face has changed convective state twice during a solve, all currently non-convective faces hotter than 1e6 K are forced radiative for the rest of that solve. This changes `gradT` at growing convective boundaries in the default path.
- Proposed fix: delete the second branch. S.

### F4-4: `min_D_mix` has inverted meaning and the `set_min_D_mix*` floors, `D_mix_zero_region*`, crossover zeroing and `do_mix_envelope` are absent
- Area: flexmesa-star / `mix_info.rs`, `struct_burn_mix.rs`
- Severity: wrong physics
- FlexMESA: `mix_info.rs:101-108` (`let d_floor = clip_d_limit.max(model.controls.min_d_mix); ... if model.d_mix[k] > 0.0 && model.d_mix[k] < d_floor { set_face_radiative(model, k); }`); only reader of `min_d_mix` (grep). No `set_min_d_mix`, `min_d_mix_below_tmax`, `min_d_mix_in_h_he`, `d_mix_zero_region`, `dq_d_mix_zero_at_h_he_crossover`, `t_mix_limit` anywhere in the crate.
- Upstream: `mix_info.f90:248-256` (`if (s% set_min_D_mix ...) ... if (s% D_mix(k) >= s% min_D_mix) cycle ... s% D_mix(k) = s% min_D_mix; s% mixing_type(k) = minimum_mixing`), `:258-282` (below-Tmax and H/He floors), `:312-367` (zero regions), `:1351-1390` (`do_mix_envelope`).
- Confidence: verified-by-reading-both-sides
- What: upstream `min_D_mix` raises D to a floor when `set_min_D_mix` is true. FlexMESA instead treats `min_D_mix` as a threshold below which mixing is removed and the face is forced radiative (including `gradT = gradr`). An inlist with `set_min_D_mix = .true.`, `min_D_mix = 1d2` gets the opposite of what it asks for.
- Proposed fix: implement the floors as upstream and stop feeding `min_d_mix` to the clip. S.

### F4-5: glitch cleanup runs with hard-coded non-default thresholds and differs routine by routine
- Area: flexmesa-star / `mix_info.rs`, `struct_burn_mix.rs`
- Severity: wrong physics
- FlexMESA: `struct_burn_mix.rs:418-420` (`clean_mixing_glitches(model, 1.0, 0.1)`); `mix_info.rs:90-182`, `:184-219`, `:221-229`.
- Upstream: `mix_info.f90:157-180` (calls), `:965-993` (`tiny = s% clip_D_limit`, `:980`), `:995-1080`, `:1115-1188` (`if (min_gap < 0) return`, `:1132`), `:1190-1245`, `:1247-1348` (`if (.not. s% remove_embedded_semiconvection) return`, `:1260`); defaults `clip_D_limit = 0`, `min_convective_gap = -1`, `min_thermohaline_gap = -1`, `min_thermohaline_dropout = -1`, `min_semiconvection_gap = -1`, `remove_embedded_semiconvection = .false.` (`controls.defaults:3492,3503,3514,3530,3542,3553`).
- Confidence: verified-by-reading-both-sides
- What: with default controls upstream does only `remove_mixing_singletons`. FlexMESA additionally and unconditionally (1) removes all mixing with `D < 1 cm^2/s` and forces `gradT = gradr`, `mlt_D = mlt_vc = 0` there (upstream clip default 0, and upstream `remove_tiny_mixing` does not touch `gradT`); (2) closes radiative gaps thinner than `0.1 Hp` between convective faces and between thermohaline faces (upstream off; upstream gap test uses `Hp = Peos/(rho*grav)` at `ktop`, FlexMESA `scale_height[k]`); (3) always converts semiconvection sandwiched between convection into convection, with `D = max(D, average of the two bounding D)` (upstream off by default, and when on uses `D_mix(ktop1)` and a different region walk); none of the user controls are read. Singleton fill: upstream averages `cdc`, divides by `(4 pi r^2 rho(k))^2` using the cell density and recomputes `conv_vel = 3 D / lambda` only for convective type; FlexMESA uses the face density and averages `conv_vel`. Upstream's surface (`k=1`) and centre (`k=nz`) singleton handling (`:1041-1078`) and `okay_to_remove_mixing_singleton` are missing; `remove_thermohaline_dropouts` (needs `max_dropout_gradL_sub_grada`, q-interpolation) is replaced by the generic gap bridge.
- Proposed fix: port the seven routines with their controls. M.

### F4-6: overshoot reference diffusivity `D_ob` has an invented floor and a different velocity interpolation
- Area: flexmesa-star / `overshoot.rs`
- Severity: wrong physics
- FlexMESA: `overshoot.rs:346-351` (`let vc_ob = ((1.0 - w) * model.mlt_vc[k_ob] + w * model.mlt_vc[k_ob_next]).max(0.0); ... let lambda_ob = model.controls.mixing_length_alpha * hp_ob; let d_ob = (vc_ob * lambda_ob / 3.0).max(model.mlt_d[bdy.conv_bdy_loc]);`).
- Upstream: `overshoot_utils.f90:319-368` (`lambda = (1-w)*s%mlt_mixing_length(k) + w*s%mlt_mixing_length(k+1)`; three-way case on which of `conv_vel(k)`, `conv_vel(k+1)` are zero, interpolating to `vc = 0` at `r_cb` when one face is non-convective; `D = vc*lambda/3._dp`).
- Confidence: verified-by-reading-both-sides
- What: `D_ob` sets the amplitude of every overshoot profile (`D = (D0 + Delta0*D_ob)*factor`). FlexMESA takes the larger of the interpolated value and the MLT D at the convective face nearest the boundary; upstream has no such floor. FlexMESA also interpolates linearly between the two faces even when one is non-convective (velocity 0), where upstream interpolates to zero at the sub-cell boundary `r_cb`. For small `f0` (overshoot point in the boundary cell) the two give different amplitudes. `lambda` uses `alpha * scale_height` rather than the stored `mlt_mixing_length` (differs when `redo_conv_for_dr_lt_mixing_length` shortened it).
- Proposed fix: port `eval_over_bdy_params` verbatim. S.

### F4-7: overshoot `Hp_cb` uses a 2-point linear P, rho estimate instead of upstream's 3-cell log interpolation
- Area: flexmesa-star / `mix_info.rs`
- Severity: wrong numerics
- FlexMESA: `mix_info.rs:336-345` (`let p_cb = (1.0 - w) * model.peos[cell_index] + w * model.peos[k_next]; let rho_cb = (1.0 - w) * model.rho[cell_index] + w * model.rho[k_next];`, fallback `model.scale_height[cell_index]`).
- Upstream: `overshoot_utils.f90:152-167` (`x0 = dq(k_o)/2; x1 = dq(k_o) + dq(k_m)/2; x2 = ...; x = dq(k_o) + cz_bdy_dq(k); call two_piece_linear_coeffs(...); P = exp(a0*lnPeos(k_o) + a1*lnPeos(k_m) + a2*lnPeos(k_i))`).
- Confidence: verified-by-reading-both-sides
- What: FlexMESA applies the face-fraction weight `w` (position between the two faces of cell k) to the cell-centre values of cells k and k+1, which places the boundary half a cell too deep and interpolates P linearly rather than ln P between k-1, k, k+1. `Hp_cb` scales both `f0` and `f`, so overshoot extents differ at the percent level per boundary, more on coarse meshes. `r_cb` (`:329-331`) and `conv_bdy_q` do match upstream (`overshoot_utils.f90:86-98`, `mix_info.f90:602,738`).
- Proposed fix: port `eval_conv_bdy_Hp`. S.

### F4-8: overshoot face loop omits upstream's post-loop zeroing, the `cz_bdy_dq` update at `D_min`, and `D_min` inside the zone; adds an invented Peclet taper and scales `f2` in the mass ramp
- Area: flexmesa-star / `overshoot.rs`
- Severity: wrong numerics
- FlexMESA: `overshoot.rs:358-410` (loops `break` only), `:450-466` (`if inside_cz { model.d_mix[k] = d_ov; ...; true } else { if d_ov < cfg.d_min { return false; } ...`), `:423-441` (Peclet block using `peclet_function_full_on/off`), `:190-202` (`if set.mass_full_on > set.mass_full_off && m_msun < set.mass_full_on { ... f2 *= fac; }`).
- Upstream: `overshoot.f90:219-255` (on `D(k) < overshoot_D_min`: `s%cz_bdy_dq(k) = find0(...)` then `exit face_loop`, tested before the inside/outside split), `:275-287` (inside: `cdc`, `D_mix` set; outside: only if `D(k) > D_mix(k)`), `:293-295` (`s%cdc(k:k_cb:dk) = 0; s%D_mix(k:k_cb:dk) = 0; s%mixing_type(k:k_cb:dk) = no_mixing`), `overshoot_exp.f90:89-101` / `overshoot_step.f90:93-105` (ramp `if (s%star_mass < s%overshoot_mass_full_on(j))`, scales `f`, `f0` only). No `peclet` string exists in upstream `star/` (grep over `overshoot*.f90`, `controls.defaults`).
- Confidence: verified-by-reading-both-sides
- What: (a) `peclet_function_full_on/off` are FlexMESA-only controls (default 0 = off, `controls_mix_wind.rs:353-354`) presented in the doc comment as if MESA controls. (b) Upstream moves `cz_bdy_dq` to where D falls below `overshoot_D_min`, which feeds `cz_top_mass`/`cz_bot_mass` and history core masses; FlexMESA leaves the Ledoux position. (c) When the loop ends while still inside the zone upstream turns the remainder radiative; FlexMESA leaves it. (d) `f2` is ramped in FlexMESA but not upstream. (e) Inside the zone FlexMESA also overwrites `conv_vel`; upstream does not touch `conv_vel` in `add_overshooting`.
- Proposed fix: port `add_overshooting` face loop; remove the Peclet controls or document them as extensions. S.

### F4-9: `cz_bdy_dq` gets an invented half-cell value when there is no sign change, and uses cell `grada` for all non-convective pairs
- Area: flexmesa-star / `mix_info.rs`
- Severity: wrong numerics
- FlexMESA: `mix_info.rs:253-271` (`} else { (model.grada[k] - model.gradr[k], model.grada[k + 1] - model.gradr[k + 1]) }; if dg0 * dg1 < 0.0 { ... } else { *bdy_dq = 0.5 * model.dq[k]; }`).
- Upstream: `mix_info.f90:445-455` (`else if (mt2 == semiconvective_mixing .or. mt1 == semiconvective_mixing) then dg0 = s% grada_face(k-1) - s% gradr(k-1) ... else cycle`; `if (dg0*dg1 >= 0) cycle`, leaving 0).
- Confidence: verified-by-reading-both-sides
- What: upstream leaves `cz_bdy_dq = 0` (boundary at the outer face) when the criterion does not change sign across the cell, which happens whenever the mixing type was changed by anything other than the local gradient test (glitch cleanup, predictive mixing, sticky faces). FlexMESA places the boundary at mid-cell. `model.grada` is the cell-centred value where upstream uses `grada_face`; index `k` therefore refers to a cell on one side and a face on the other. Affects `conv_bdy_q`, `r_cb`, overshoot extents and core-mass diagnostics.
- Proposed fix: match upstream branches and use the face `grada`. S.

### F4-10: `sig` call site hard-codes `mix_factor = 1` and `sig_term_limit = 1e13`; the high-Tcenter `sig_min_factor` limiter is missing; a second `sig` routine uses a 1e6 cap
- Area: flexmesa-star / `struct_burn_mix.rs`, `mix_info.rs`, `hydro_vars.rs`
- Severity: wrong numerics
- FlexMESA: `struct_burn_mix.rs:429` (`compute_convection_sigmas(model, dt, 1.0, 1.0e13);`), `mix_info.rs:359-397`; `hydro_vars.rs:73-90` (`sig_k = sig_k.min(1.0e6 * dm_min / model.dt);`) called from `evolve_support.rs:337-339` and `hydro_vars.rs:533,549` when `sig` is all zero.
- Upstream: `mix_info.f90:1392-1505` (`cdcterm = s% mix_factor*cdc`; `siglim = sig_term_limit*xmstar*min(dq(k),dq(k-1))/dt` with `sig_term_limit = s% sig_term_limit`, default `1d13` at `controls.defaults:11679`; then the `sig_min_factor_for_high_Tcenter` block, default `0.01`, active above `Tcenter_max_for_sig_min_factor_full_off`).
- Confidence: verified-by-reading-both-sides
- What: the core formula and the limiter form match upstream (including `max_sig < 1` early exit). `mix_factor` is applied later in the equation (`hydro_eqns.rs:396`), so the hard-coded 1.0 is not a double-count, but upstream applies `mix_factor` before the limit and FlexMESA after, which differs when the limit binds. `sig_term_limit` is not a control. The species-aware limiter that upstream applies at high central temperature to avoid negative abundances is absent (matters for advanced burning). The fallback `update_sig_from_d_mix` has a different, invented cap `1e6` and is what sets `sig` on paths where `refresh_mixing_state` has not run.
- Proposed fix: read both controls, port the `do1_region` limiter, delete the second routine. S.

### F4-11: Brunt B is clamped to [-10, 10], zeroed for tiny dlnP, and lacks the mass-correction term
- Area: flexmesa-star / `hydro_brunt_surf.rs`
- Severity: wrong numerics
- FlexMESA: `hydro_brunt_surf.rs:46-50` (`if dlnp_peos.abs() < 1.0e-14 { model.brunt_b[k] = 0.0; continue; }`), `:97-101` (`b_k.clamp(-10.0, 10.0)` else `0.0`), `:92` (EOS failure silently falls back to a gradmu form).
- Upstream: `brunt.f90` `get_brunt_B` (`delta_lnP = s% lnPeos(k-1) - s% lnPeos(k); if (delta_lnP > -1d-50) then ... dlnP_dm = -cgrav*m/(pi4*pow4(r)*Ppoint); delta_lnP = dlnP_dm*s% dm_bar(k)`; `s% brunt_B(k) = (lnP1 - lnP2)/delta_lnP/chiT_face`; `if (s% use_mass_corrections) ... - chiRho_face*delta_lnMbar/delta_lnP/chiT_face`; bad number -> `ierr = -1`).
- Confidence: verified-by-reading-both-sides
- What: the main formula (two EOS calls at face rho, T with the compositions of the two cells) matches. Divergences: invented clamp at |B| = 10; pressure inversions (`delta_lnP >= 0`) are handled upstream by substituting the hydrostatic estimate, FlexMESA keeps the inverted sign (or zeros it only if < 1e-14); `use_mass_corrections` term absent; EOS failure and non-finite B are swallowed instead of failing the step.
- Proposed fix: S.

### F4-12: MLT call site drops several upstream no-mixing gates, `remove_small_D_limit`, superadiabatic reduction, `mlt_gradT_fraction`, and hard-codes the thermohaline driving isotope
- Area: flexmesa-star / `hydro_vars.rs`
- Severity: wrong physics
- FlexMESA: `hydro_vars.rs:203-213` (only gates: `alpha_mlt <= 0.0 || sticky_no_conv`), `:337-344` (`species: DrivingSpecies::he4()`), `:363-397` (semiconvection additionally requires `b_term > 0.0` and `gradr - grada > 1e-40`). grep for `superad_reduction`, `mlt_gradt_fraction`, `max_logt_for_mlt`, `mlt_make_surface_no_mixing`, `dominant_iso`, `remove_small_d_limit`, `max_v_for_convection`, `no_mlt_below_shock` over the crate: no hits outside tests.
- Upstream: `turb_info.f90:145-217` (gates: `mlt_make_surface_no_mixing`, crystallized `phase(k) > 0.5 .and. mu(k) > 1.7`, `phase_sep_mixing_mass`, `max_logT_for_mlt`, `no_MLT_below_shock`, `max_v_for_convection`, `max_q_for_convection_with_hydro_on`, `max_v_div_cs_for_convection`, `max_abs_du_div_cs_for_convection`), `:245-250` (`adjust_gradT_fraction`), `turb_support.f90:371-421` (`use_superad_reduction`), `:424-443` (thermohaline if `gradL_composition_term < 0`, else semiconvection if `gradr > grada`), `:448-449` (`if (D%val < s% remove_small_D_limit .or. is_bad(D%val)) mixing_type = no_mixing`, default `1d-6`), `turb_support.f90:181` (`iso = s% dominant_iso_for_thermohaline(k)`, computed in `hydro_vars.f90:1101-1119`).
- Confidence: verified-by-reading-both-sides
- What: (a) Convection is not suppressed in crystallized material, below `phase_sep_mixing_mass`, behind shocks or in fast-moving zones; relevant to WD cooling and every hydro (`v_flag`/`u_flag`) case. (b) `use_superad_reduction` (used by massive-star inlists) cannot be honoured. (c) The Brown et al. / Traxler thermohaline options take the driving isotope's charge and mass; FlexMESA always passes he4. The Kippenhahn default does not use it. (d) Semiconvection gate matches in effect (`b_term > 0` is implied by Ledoux-stable and Schwarzschild-unstable). (e) `remove_small_D_limit` absent. What matches: `gradL = grada_face + composition term` only under `use_Ledoux_criterion`, thermohaline before semiconvection, `mlt_vc = 0` unless convective (`hydro_vars.rs:402-406` vs `turb_support.f90:205`), `mlt_cdc = D*(4 pi r^2 rho_face)^2`... except FlexMESA forms `cdc` from `conv_vel*lambda/3` (`:401`), not from `D`, so for thermohaline/semiconvective faces `mlt_cdc` is inconsistent with `mlt_d` (it is overwritten later by `compute_convection_sigmas`, so the solve is unaffected).
- Proposed fix: M.

### F4-13: convective-region burn classification and `prune_bad_cz` are not part of boundary location
- Area: flexmesa-star / `mix_info.rs`, `overshoot.rs`
- Severity: wrong numerics
- FlexMESA: `overshoot.rs:114-145` (`classify_cz_burning` over `k_top..=k_bot` using cell `xh_mass`, `yhe_mass`), `mix_info.rs:276-305` (no prune).
- Upstream: `mix_info.f90:671-688` (`if (dr/Hp < s% prune_bad_cz_min_Hp_height ...) ... max_eps < exp10(prune_bad_cz_min_log_eps_nuc)` -> region set radiative), `:698-735` (classification over `kk = k, min(nz, k_bot+1)`; a region flag is also set on the previous boundary `i-1`).
- Confidence: verified-by-reading-both-sides
- What: thresholds and precedence (Z, then He, then H) match. Range differs by one cell at each end. `prune_bad_cz_*` missing (default off upstream, so only inlists that set it are affected). The classification exists only for overshoot matching; predictive mixing ignores it (F4-2).
- Proposed fix: S.

### Matches (area 1)

- Sub-cell boundary interpolation when the criterion changes sign: `mix_info.rs:265-267` vs `mix_info.f90:456` (`find0(0, dg0, dq(k-1), dg1)`), and use of `gradL - gradr` when either face is convective (`mix_info.rs:256-261` vs `mix_info.f90:445-447`). verified-by-reading-both-sides.
- Scan direction and index range for boundaries (faces `nz-1 .. 2` 1-based), `conv_bdy_q = q(k) - cz_bdy_dq(k)` for both top and bottom: `mix_info.rs:276-305` vs `mix_info.f90:581-611,736-741`. FlexMESA's `conv_bdy_loc` is the adjacent convective face where upstream's is the first non-convective face (bottom: `k_bot = k+1`); the struct doc states this and consumers use `cell_index`, which equals upstream `eval_conv_bdy_k`.
- `r_cb` from linear r^3 in q: `mix_info.rs:329-331` vs `overshoot_utils.f90:86-98`.
- Overshoot rule matching: first matching non-empty set wins, `'none'` short-circuits, `is_core = (i == 1 && R_center == 0 && top)`: `overshoot.rs:252-276,88-111` vs `overshoot.f90:95-163`. `min_overshoot_q` skip and surface skip match (`overshoot.rs:236,256` vs `overshoot.f90:77-92`).
- Exponential and step(+exponential) profile factors, `D = (D0 + Delta0*D_ob)*factor`, `vc = (D0/D_ob + Delta0)*vc_ob*factor`: `overshoot.rs:68-87,443-448` vs `overshoot_exp.f90:142-151`, `overshoot_step.f90:140-152`. `dr` measured from `r_ob`: `overshoot.rs:367,388`.
- `r_ob = r_cb -/+ f0*Hp_cb` with clipping to the centre/surface and the bracketing search: `overshoot.rs:321-340` vs `overshoot_utils.f90:273-315`; r^3 weight `w`: `:341-345` vs `:319-320`.
- `limit_overshoot_Hp_using_size_of_convection_zone` cap `alpha*Hp > dr -> Hp = dr/alpha` with `overshoot_alpha` override: `overshoot.rs:297-319` vs `overshoot_utils.f90:174-223` (FlexMESA uses the global alpha, upstream `alpha_mlt(k)`).
- Brunt-B stabilisation probes `B(k + dk)`: `overshoot.rs:360-365,381-386` vs `overshoot.f90:198-214` (FlexMESA reads the smoothed `brunt_b`, upstream `unsmoothed_brunt_B`).
- Defaults: `overshoot_D_min = 1e-2`, `remove_mixing_glitches = true`, `mix_factor = 1`, `num_cells_for_smooth_gradL_composition_term = 3`, `num_cells_for_smooth_brunt_B = 2`, `use_Ledoux_criterion = false` agree with `controls.defaults:2671,3466,3353,2073,7831,2057`.
- `sig(k) = mix_factor * D * (4 pi r^2 rho_face)^2 / (xmstar*(dq(k)+dq(k-1))/2)` with mass-weighted face density, and `siglim = sig_term_limit*xmstar*min(dq)/dt` applied only if `max(sig) >= 1`: `mix_info.rs:377-396,64-86` vs `mix_info.f90:1411-1452`.
- `eval_dxdt_mix` stencil: `mix_info.rs:401-425` vs `mix_info.f90:2303-2364` (formula as documented; upstream body not re-read line by line).
- Brunt B main formula: `hydro_brunt_surf.rs:66-91` vs `brunt.f90` `get_brunt_B`.
- Mixing info is not recomputed during Newton iterations (`sig` frozen): `struct_burn_mix.rs:439-494` matches upstream, where `set_mixing_info` is skipped inside the solver.

### Not reached (area 1)

- `mixing_D_limit_for_log`, `locate_mixing_boundaries`, `set_cz_bdy_mass` consumers (`mass_conv_core`), `do_smoothing_by_mass`, `set_dPdr_dRhodr_info`, RTI mixing (`hydro_rti.rs`).
- `check_for_redo_mlt` (`hydro_vars.rs:434-473`) vs `mlt_info.f90`/`turb_info.f90` `check_for_redo_MLT`: read on the FlexMESA side only.
- The turb crate kernels themselves (`set_thermohaline`, `set_semiconvection`): F2 scope.


---

## Area 2: mass change (`winds.rs`, `winds_driver.rs`, `adjust_mass.rs`, `eps_mdot.rs`, `mdot_redo.rs`)

Summary: the individual wind formulae and `eps_mdot` are faithful ports. The
driver around them has one real bug (super-Eddington wind), one invented
fallback that turns winds on where MESA has none, several missing upstream
gates that are active by default, and an implicit-mdot loop that differs from
`select_mdot_action` in its limits and exits.

### F4-14: super-Eddington wind is a factor 2 too small and uses the wrong Eddington luminosity
- Area: flexmesa-star / `winds_driver.rs`
- Severity: wrong physics
- FlexMESA: `winds_driver.rs:120-129` (`let ledd_eff = mw.super_eddington_wind_ledd_factor.max(0.0) * l_edd; if l1 > ledd_eff { let v_esc2 = (2.0 * cgrav * m1 / r1).max(1.0); wind_mdot = c.super_eddington_scaling_factor * (l1 - ledd_eff) / v_esc2; }`), `l_edd` from `:101-102` (`kap_surf = model.opacity.first()`; `l_edd = FOUR_PI * CLIGHT * cgrav * m1 / kap_surf`).
- Upstream: `winds.f90:649-653` (`Ledd = s% prev_Ledd; Leff = L/s% super_eddington_wind_Ledd_factor; if (Leff <= Ledd) return; vesc2 = 2d0 * s% cgrav(1)*M/R; s% super_eddington_wind_mdot = s% super_eddington_scaling_factor*(Leff - Ledd)/(0.5d0 * vesc2)`); `prev_Ledd = eval_Ledd(s,ierr)` (`evolve.f90:1787`, a mass-weighted average over the outer layers from the previous step).
- Confidence: verified-by-reading-both-sides
- What: (a) upstream divides by `0.5*vesc2 = GM/R`; FlexMESA divides by `vesc2 = 2GM/R`, giving half the mass-loss rate. (b) Upstream forms `L/factor - Ledd`; FlexMESA forms `L - factor*Ledd`, which is `factor` times larger (equal only for the default factor 1). (c) Upstream's `Ledd` is `prev_Ledd`; FlexMESA uses the single surface-cell opacity, also in the wind-boost test (`:239`, upstream `winds.f90:218`). (d) Upstream `compute_delta_m` (`adjust_mass.f90:41-48`) lets the super-Eddington wind override `mstar_dot` subject to `max_wind`; no such step in FlexMESA.
- Proposed fix: S (formula); M to add `eval_Ledd`/`prev_Ledd`.

### F4-15: invented "legacy fallback" applies a wind when both scheme names are blank
- Area: flexmesa-star / `winds_driver.rs`
- Severity: wrong physics
- FlexMESA: `winds_driver.rs:156-172` (`if hot_s == WindScheme::None && cool_s == WindScheme::None { let fallback = if c.dutch_scaling_factor > 0.0 { WindScheme::Dutch } else if c.vink_scaling_factor > 0.0 { ... }; hot_s = fallback; cool_s = fallback; }`).
- Upstream: `winds.f90:320-321` `eval_wind_for_scheme` (`if ((.not. use_other) .and. len_trim(scheme) == 0) then wind = 0`).
- Confidence: verified-by-reading-both-sides
- What: in MESA a scaling factor without a scheme name does nothing. FlexMESA picks a scheme from whichever scaling factor is positive (Dutch, then Vink, Bjorklund, Reimers, Blocker) and applies it as both hot and cool wind. An upstream inlist that sets e.g. `Reimers_scaling_factor` but leaves `cool_wind_RGB_scheme` empty loses no mass in MESA and loses mass here. Also unknown scheme names: `'other'` is rejected here; upstream calls `other_wind`.
- Proposed fix: delete the fallback. S.

### F4-16: default-active upstream wind gates are missing (`max_T_center_for_*_mass_loss`, envelope limits, dt ramp, RLO wind)
- Area: flexmesa-star / `winds_driver.rs`
- Severity: wrong physics
- FlexMESA: `winds_driver.rs:92-320`; grep over the crate for `max_t_center_for`, `mass_change_full`, `wind_h_envelope`, `wind_he_layer`, `rlo_scaling`, `rlo_wind`: no hits.
- Upstream: `winds.f90:255-268` (`if (s% starting_T_center > s% max_T_center_for_any_mass_loss) then wind_mdot = 0 else if (... > s% max_T_center_for_full_mass_loss) then wind_mdot = wind_mdot*(...)`), defaults `2d9` / `1d9` (`controls.defaults:5268,5284`); `wind_H_envelope_limit`, `wind_H_He_envelope_limit`, `wind_He_layer_limit`; `mass_change_full_on_dt`/`_off_dt` ramp; `eval_rlo_wind` and `xfer_ratio` on `mass_change > 0`.
- Confidence: verified-by-reading-both-sides
- What: with default controls MESA tapers all mass loss to zero between `T_c = 1e9` and `2e9 K`. FlexMESA keeps the full wind through advanced burning. The other gates are off by default but are silently ignored when set.
- Proposed fix: S each.

### F4-17: rotational enhancement and the implicit-mdot test use different surface rotation measures; `select_mdot_action` limits and exits differ
- Area: flexmesa-star / `winds_driver.rs`, `mdot_redo.rs`, `evolve.rs`
- Severity: wrong numerics
- FlexMESA: `winds_driver.rs:279-287` (`omega_crit = eval_omega_crit(cgrav, m1, r1, l1, kap_surf, 0.999_999); w_div_wc = (model.omega[0].abs() / omega_crit).clamp(0.0, 1.0)`); `mdot_redo.rs:282` (`if self.mdot_redo_cnt == 0 || (self.mdot_redo_cnt == 1 && self.ignored_first_step)` computes `max_wind_mdot`), `:306-312` (`if (self.mdot_redo_cnt > 0 || w > self.surf_limit) && dt >= 0.1 * SECYER { return ForceRetry; }`), `:327-329` (`if w <= self.surf_limit && (self.surf_limit - w) <= tol { return Accept; }`), `:358-365` (`if dt < 0.1 * SECYER { return Accept; }`), `:332` (`mdot_revise_factor.max(1.001)`), `:264-270` (returns Accept when `!rotation_flag`); `evolve.rs:246-262` (solver-failure fallbacks, F1-11).
- Upstream: `winds.f90:438` `rotation_enhancement` (`enhancement = pow(max(1d-22, 1d0 - s% v_div_v_crit_avg_surf), -s% mdot_omega_power)`); `evolve.f90:798-822` (implicit wind `iwind_*` loop, independent of rotation), `:835-842` (`if (mdot_redo_cnt == 1 .or. ignored_first_step)`), `:850-869` (`wind_mdot >= max_wind_mdot`: retry if `mdot_redo_cnt /= 0`, retry if `w_div_w_crit > limit`, else exit), `:951-962` (`mdot_redo_cnt >= max_mdot_redo_cnt` -> retry), `:767-775` (`s% generations == 2` in the skip-first-step test).
- Confidence: verified-by-reading-both-sides
- What: (a) Enhancement: upstream uses `v_div_v_crit_avg_surf` (optical-depth-averaged equatorial v/v_crit from `set_surf_avg_rotation_info`); FlexMESA uses `omega/omega_crit` of the surface face only, with a gamma cap of 0.999999. The implicit loop does use an averaged quantity (`mdot_redo.rs:28-155`), so the two parts of FlexMESA disagree with each other as well. (b) `max_wind_mdot`: upstream leaves it at `10 Msun/yr` on the first pass and sets the KH-limited value only once `mdot_redo_cnt == 1`; FlexMESA sets it on the first pass, so the first comparison and the first revised mdot are clipped differently. (c) The `dt >= 0.1 yr` guards, the "within tol below the limit -> accept" exit and the clamps on `mdot_revise_factor`/tol are invented; upstream retries regardless of dt. (d) `max_tries_for_implicit_wind` / `iwind_tolerance` / `iwind_lambda` (implicit wind for non-rotating stars) are not implemented. The bracketing/bisection and boost sequence (`mdot_redo.rs:368-414`) does match `evolve.f90:976-1055` line for line.
- Proposed fix: use the averaged v/v_crit in the enhancement, restore upstream gating. M.

### F4-18: accreted material gets the old surface specific angular momentum, not the old surface omega; accretion-side j remap uses cell intervals for a face quantity
- Area: flexmesa-star / `adjust_mass.rs`
- Severity: wrong numerics
- FlexMESA: `adjust_mass.rs:276-280` (`let j_acc = if use_accreted_material_j { accreted_material_j } else { starting_j_rot_surf };`), `:296-344` (j_rot of new cells from overlap of `cum_old`/`cum_new`, which are cumulative cell masses, `model.j_rot[k] = jr_acc / dx`).
- Upstream: `adjust_mass.f90:1114-1230` `set_omega_adjust_mass` (`:1199` `else ! use old surface omega in all the new material: s% omega(k) = s% omega(k_below_just_added); call set1_irot(...); s% j_rot(k) = s% omega(k)*s% i_rot(k)% val`; with `use_accreted_material_j`: `j_rot = accreted_material_j` for `k <= k_below_just_added-2` and the two boundary points set so that `total J = old_j_tot + delta_m*accreted_material_j`), intervals `old_xout/new_xout` built from `dmbar`.
- Confidence: verified-by-reading-both-sides
- What: default accretion in MESA adds material co-rotating with the old surface (same omega, hence larger j at larger r). FlexMESA gives it the old surface j. With `use_accreted_material_j` upstream closes the J budget exactly; FlexMESA does not. On the accretion branch FlexMESA remaps `j_rot` on cell-mass intervals, but `j_rot` lives on faces with weights `dm_bar`, so total J (`sum j*dm_bar`) is not conserved by the remap. The mass-loss branch (`remap_j_rot_and_adjust_j_lost`, `:397-447`) does use `dm_bar` intervals as upstream `set1_omega`.
- Proposed fix: M.

### F4-19: `min_star_mass_for_loss` / `max_star_mass_for_gain` stop the wind one step late; uniform-rescale fallback when `qfrac <= 0`
- Area: flexmesa-star / `winds_driver.rs`, `adjust_mass.rs`
- Severity: wrong numerics
- FlexMESA: `winds_driver.rs:261-273` (mdot zeroed only when `mstar` is already past the limit), `adjust_mass.rs:220-221` (`let uniform_fallback = qfrac <= 0.0; let k_b_remap = if uniform_fallback { nz } else { k_b };`), `:226-231` (all cells scaled by `xm_new / sum_prev_dm`).
- Upstream: `adjust_mass.f90:247-256` (`new_mstar = Msun*s% min_star_mass_for_loss; delta_m = new_mstar - old_mstar; s% mstar_dot = delta_m/dt`), `revise_q_and_dq` (kB is moved inward until `qfrac` is acceptable or `kB == nz`; no uniform rescale).
- Confidence: verified-by-reading-both-sides
- What: upstream lands exactly on the mass limit; FlexMESA overshoots by up to one step of mass change. The uniform fallback changes the mass of every interior cell (so composition and thermal state are advected through the whole star by index) in a case upstream handles by extending the middle region to the centre.
- Proposed fix: S.

### Matches (area 2)

- Reimers `4e-13 L R / M`: `winds.rs:88-98` vs `winds.f90` (`wind = 4d-13*(L1*R1/M1)/(Lsun*Rsun/Msun)`). Blocker `4.83e-9 M^-2.1 L^2.7 * Reimers`: `:104-118` vs `eval_blocker_wind`.
- Vink: every coefficient, the 22500/27500 K limits, the jump temperature `1e3*(61.2 + 2.59*(-13.636 + 0.889*log10(Z/Zsolar)))`, `dT = 100`, `Zsolar = 0.019`, `vinf/vesc` 2.6 and 1.3 with `(Z/Zsolar)^0.13`: `winds.rs:123-172` vs `winds.f90` `eval_Vink_wind`. verified-by-reading-both-sides.
- Bjorklund (`-5.52, 2.39, 1.48, 2.12, 0.75, 1.87`, `Zbjork = 0.014`, `Meff` with 0.34): `winds.rs:179-206` vs `eval_Bjorklund_wind`.
- Nieuwenhuijzen (`-14.02, 1.24, 0.16, 0.81`), de Jager (`1.769, -1.676, -8.158`), van Loon (`-5.65, 1.05, -6.3`, 3500 K), Nugis-Lamers (`1e-11 L^1.29 Y^1.7 sqrt(Z)`): `winds.rs:212-272` vs `winds.f90`.
- Dutch: `T_low = 1e4`, `T_high = 1.1e4`, linear blend, WR switch at surface `X < 0.4`, low-T scheme selection, single multiplication by `Dutch_scaling_factor`: `winds.rs:278-313` vs `winds.f90`.
- Hot/cool blending (`cool_wind_full_on_T`, `hot_wind_full_on_T`, equal-T case, `beta`), RGB->AGB switch (`center_h1 < 0.01 && center_he4 < RGB_to_AGB_wind_switch`), `Z = Zbase` when `Zbase >= 0`, `mass_change` precedence, `no_wind_if_no_rotation`, `min_wind`/`max_wind`: `winds_driver.rs:131-266` vs `winds.f90:54-220`. `R1 = sqrt(L1/(pi crad c T1^4))`: `:99-100`.
- Rotational enhancement algebra (`(1-w)^-power`, `max_rotational_mdot_boost`, limit `min(kh_fac*M/t_KH, wind*boost_fac)`, `max_mdot_jump_for_rotation*prev`, recheck `max_wind`, trigger `min_wind > 0 || using scheme || w > 0.8`): `winds_driver.rs:275-316` vs `winds.f90` `rotation_enhancement`; `t_KH = 0.75 G M^2/(R L)` matches `eval_kh_timescale`.
- Wind defaults (all scaling factors 0, `cool/hot_wind_full_on_T` not re-checked, `RGB_to_AGB_wind_switch = 1e-4`, `mdot_omega_power = 0.43`, `max_rotational_mdot_boost = 1e4`, `max_mdot_jump_for_rotation = 2`, `rotational_mdot_boost_fac = 1e5`, `rotational_mdot_kh_fac = 0.3`, `surf_avg_tau_min = 1`, `surf_avg_tau = 100`, `mdot_revise_factor = 1.1`, `implicit_mdot_boost = 0.1`, `surf_omega_div_omega_crit_tol = 0.05`, k_below_const_q / k_const_mass q and logT limits, `eps_mdot_factor = 1`, `do_adjust_J_lost`, `adjust_J_fraction = 1`, `min_q_for_adjust_J_lost = 0.995`, `min_J_div_delta_J = 3`): `controls_mix_wind.rs:363-455` vs `controls.defaults:4005-5439`.
- Region selection `revise_q_and_dq`: `min_kA = 5` (0-based 4), the three-way q / logT tests for kA and kB, `lnTmax` clipping, `qfrac = 1 + dqacc*(1 - xqB)/(xqB - xqA)` (algebraically equal to upstream `(xqB_new - xqA)/(xqB_old - xqA)`), the `kB-kA > 10`, `0.67 < qfrac < 1.5`, `0.5 < qfrac < 2` plus dominant-species change `kB -= 5` rule: `adjust_mass.rs:536-595` vs `adjust_mass.f90:607-760`. Outer region constant q, middle stretched, inner constant mass: `adjust_mass.rs:225-241`. (After `kB -= 5` upstream keeps the old `qfrac` and renormalises; FlexMESA recomputes it.)
- Composition of new cells by exact mass overlap with old cells plus `xaccrete` for the just-added part, then renormalise: `adjust_mass.rs:296-340` vs `adjust_mass.f90` `set1_xa`.
- T and rho are not interpolated (cells keep their thermodynamic state by index) and radii are rebuilt from `dm/rho` outward from the centre: `adjust_mass.rs:359` vs `adjust_mass.f90` `update_radius`.
- `angular_momentum_removed` weights (`0.5*(dm(k-1)+dm(k))`, first point `0.5*dm(1)`) and `adjust_J_lost` (`actual = f*mass_lost*j_surf + (1-f)*J_removed`, search for `last_k`, linear-in-q rescale `1 + frac*(q - qlast)`): `adjust_mass.rs:419-495` vs `adjust_mass.f90:527-553,1476-1572`.
- `eps_mdot`: mass flux, density-weighted flux, face interpolation on the previous mesh, `total_mass_through_cell`, the three-term advected-energy form, the PdV difference, the cell total-energy change, leak fraction `factor*|L dt/((grada - gradT_mid) cp T flux_bar)|` capped at 1, and the `leak_control`/`leak` parcel bookkeeping: `eps_mdot.rs:66-306,308-592` vs `eps_mdot.f90:64-470,472-650`. Read side by side for the driver and `set_leak_frac`; `make_compressed_intersect` and `leak` were compared structurally, not line by line. Differences: kinetic energy uses `0.5*v(k)^2` where upstream `cell_specific_KE` averages both faces for `v_flag`; quad precision replaced by a rearranged double-precision difference form; the innermost `grad_r_sub_grad_a` uses `grada(nz)` instead of copying `nz-1`.
- `eval_surf_avg_w_div_w_crit` structure (tau window `surf_avg_tau_min..surf_avg_tau`, mass-weighted `Lrad/Ledd`, `gamma_factor`): `mdot_redo.rs:28-155`; compared with upstream in area 3 matches.

### Not reached (area 2)

- `get_xa_for_accretion` (`adjust_xyz.f90`) vs `build_accreted_xa` (`adjust_mass.rs:48-123`): FlexMESA side read only.
- `set_D_omega` / `am_nu_rot` remap on mass change, `smooth_outer_xa_*`, `xa_removed`: absent in FlexMESA, not assessed further.
- Order of operations: FlexMESA recomputes radii from `rho` and `dm` before the mass change when `eps_mdot` is needed (`adjust_mass.rs:187-192`); upstream does not. Effect is at solver-tolerance level; not quantified.

---

## Area 3: rotation (`rotation.rs`, `rotation_mix.rs`, `omega_mix.rs`, `mdot_redo.rs` surface averages)

Summary: omega/j_rot are operator-split from the structure solve, as in
upstream's default (`w_div_wc_flag`, `j_rot_flag` off), and the half-step
before / half-step after split under `premix_omega` matches. The fp/ft fit is
the upstream Fabry et al. fit. The moment of inertia, the definition of
`w = Omega/Omega_crit`, the DSI coefficient, the mu-gradient entering the
Spruit-Tayler dynamo, the Solberg-Hoiland coefficient, and the conservation
handling in the transport solve all differ from upstream.

There are two rotational-mixing implementations. `rotation.rs:163-208`
(`eval_rotational_mixing_zone`, simplified textbook forms such as
`d_es = k_therm/|grada-gradt| * min(om^2 r/g, 0.9)`) is called from
`set_rotation_mixing_info` (`rotation.rs:312-346`), whose result is discarded at
its only step-loop call site (`evolve_support.rs:198`: `let _ = set_rotation_mixing_info(model);`).
The coefficients that reach `D_mix` and the omega transport come from
`rotation_mix.rs:51-255` (`eval_heger_rot_diffusivities`), which is modelled on
upstream `rotation_mix_info.f90`. Findings below refer to the latter unless stated.

### F4-20: moment of inertia `i_rot(w)` is an invented rational fit, 10% low at w = 0.5 and 30% low at w = 0.89
- Area: flexmesa-star / `rotation.rs`
- Severity: wrong physics
- FlexMESA: `rotation.rs:58` (`let i_rot_over_r2 = (2.0 / 3.0) * (1.0 + 0.1325 * w2 + 0.0480 * w4) / (1.0 + 0.0450 * w2);`), used at `:237-238`.
- Upstream: `hydro_rotation.f90:181-227` `eval_i_rot` (`re = r*(1d0+w2/6d0-0.0002507d0*w4+0.06075d0*w6); B = (1d0+w2/5d0-0.2735d0*w4-0.4327d0*w6-3d0/2d0*0.5583d0*lg_one_sub_w4); A = (1d0-0.1076d0*w4-0.2336d0*w6-0.5583d0*lg_one_sub_w4); ir = two_thirds*pow2(re)*B/A`).
- Confidence: verified-by-reading-both-sides; numbers by evaluating both expressions (python3): `i/r^2` upstream / FlexMESA = 0.6811 / 0.6690 at w = 0.2, 0.7632 / 0.6831 at 0.5, 0.9495 / 0.7157 at 0.8, 1.0523 / 0.7307 at 0.89.
- What: the coefficients 0.1325, 0.0480, 0.0450 appear nowhere in upstream. Since `omega = j_rot / i_rot`, every rotating model has a different omega for the same angular momentum, and total J, rotational energy and all shear-driven coefficients inherit the error. Both agree at w = 0 (2/3 r^2).
- Proposed fix: port `eval_i_rot`. S.

### F4-21: `w = Omega/Omega_crit` is computed as `Omega / sqrt(GM/r^3)` instead of inverting the Roche relation, and fp/ft are clamped to [0.75, 1] / [0.95, 1]
- Area: flexmesa-star / `rotation.rs`
- Severity: wrong physics
- FlexMESA: `rotation.rs:235-237` (`let om_c = eval_omega_crit(model.cgrav[k], m_k, r_k, 0.0, 0.0, 0.99); let w = if om_c > 0.0 { (om_k.abs() / om_c).min(max_w) } else { 0.0 }; let (fp, ft, i_r2) = eval_rotation_fp_ft(w, 0.75, 0.95);`), clamps at `:56-57`.
- Upstream: `hydro_rotation.f90:229-245` (`s% w_div_w_crit_roche(k) = w_div_w_roche_jrot(s% r(k), s% m(k), s% j_rot(k), s% cgrav(k), s% w_div_wcrit_max, s% w_div_wcrit_max2, s% w_div_wc_flag)`), `:111-178` (bisection on `jrot/sqrt(G M rpsi) = (2/3) w C(w)/A(w)`), `:56-109` (omega form: `rpsi*|omega|^(2/3)/(GM)^(1/3) = w^(2/3)*rpsi_from_re_factor(w)`), `:753-811` `eval_fp_ft` (no clamp; no `fp_min`/`ft_min` string exists in upstream `star/`).
- Confidence: verified-by-reading-both-sides
- What: in upstream `w` is the ratio to the critical rate at the equatorial radius of the deformed isobar, obtained by bisection from `j_rot`. `Omega*sqrt(r_psi^3/GM)` equals `w*(1 - w^2/6 + ...)^1.5`, so FlexMESA's `w` is too small: 0.470 where upstream has 0.5, 0.673 for 0.8, 0.712 for 0.89 (python3). On top of that fp and ft are floored at 0.75 and 0.95 (values of an older MESA control that upstream no longer has), while the upstream fit gives fp = 0.384, ft = 0.760 at w = 0.8. The centrifugal correction to hydrostatic balance and to `gradr` is therefore much weaker than MESA's for fast rotators. The fit coefficients themselves (`0.3293, 0.4926, 0.5560`, exponent `5.626`, `2/3, 0.2133, 0.1068`, `0.07955, 0.2322`) match `hydro_rotation.f90:788-792,51` exactly. The fixed-point loop `omega = j/i_rot(w(omega))` (`rotation.rs:267-300`, up to 25 sweeps) replaces upstream's direct `j -> w` bisection; `w_div_wcrit_max2` and the sigmoid cap are absent.
- Proposed fix: port `w_div_w_roche_jrot`, drop the clamps. S.

### F4-22: the Spruit-Tayler dynamo receives `N2_mu` multiplied by `am_gradmu_factor` (0.05), uses an invented magnetic diffusivity, and omits upstream's convective-luminosity branch
- Area: flexmesa-star / `rotation_mix.rs`, `rotation.rs`
- Severity: wrong physics
- FlexMESA: `rotation_mix.rs:165-166` (`let n2_mu = (g_k / hp[k]) * delta[k] * f_mu * b_pos; let n2_t = (g_k / hp[k]) * delta[k] * (-dgrad[k]);`), passed to the dynamo at `:191-193`; `rotation.rs:83-90` (`eval_magnetic_diffusivity`: `7.0e11 * z * ln_lam * T^-1.5` clamped to `[1e-6, 1e16]`); `rotation.rs:154-157` (`nu_st = ((f_mag * sm) / (om * q * rho)).clamp(0.0, cap_d)`).
- Upstream: `rotation_mix_info.f90:431-726` `setup` (`:586-587` `N2(i) = -grav(i)*(1/gamma1(i) - dlnRho_dlnP)/scale_height(i); N2_mu(i) = -grav(i)/scale_height(i)*(1/chiRho(i) - delta(i)*dlnT_dlnP - dlnRho_dlnP)`, from the smoothed `grad_density`, `grad_temperature`); `set_ST` (`:1137-1348`; `xmagfmu = 1` at `:1168`; `xmagnmu = N2_mu(k)`; `xsig = calc_sige(abar, zbar, rho, T, Cp, kap_cond, opacity); xeta = calc_eta(xsig)`; `if ((xmagnt <= 0.D0) .AND. (xmagnmu > 0.D0) .AND. (xmagn > 0.D0)) xmagnu = sqrt(xmagnu*scale_height(k)*(one_third)*pow(grav*delta*scale_height*MAX(0.D0,L(k))/(64.D0*pi*xmagdn*Cp*xmagtn*xmagrn*xmagrn),one_third))`).
- Confidence: verified-by-reading-both-sides
- What: `f_mu = am_gradmu_factor` is the Heger et al. sensitivity factor for the hydrodynamic instabilities (it multiplies `Ri_mu` and `ve_mu` upstream). Upstream does not apply it in the dynamo. FlexMESA does, so where composition gradients dominate, `N_mu^2` is 20 times smaller; since `S_0 ~ (omega/N_mu)^4` and `eta_0 ~ (omega/N_mu)^6`, the stress is up to 400 times and the effective diffusivity up to 8000 times larger. N2 also comes from `B` and `grada - gradT` rather than from the smoothed density and temperature gradients. The magnetic diffusivity is a Spitzer-type closed form instead of upstream's conductivity routine with the conductive opacity. The upstream branch that replaces `nu` in thermally unstable, mu-stable layers is missing. The algebra of `q0, S0, eta0, q1, S1a/b, eta1a/b`, the harmonic combination and `nu = S/(omega q rho)`, `D = min(f*eta, nu)` does match `set_ST` (`rotation.rs:92-158`).
- Proposed fix: M.

### F4-23: D_DSI uses a different formula; D_SH is always zero; every coefficient is smoothed with a fixed 1/4-1/2-1/4 kernel
- Area: flexmesa-star / `rotation_mix.rs`
- Severity: wrong physics
- FlexMESA: `rotation_mix.rs:205-213` (`v_tmp[k] = cs[k].min((hp[k] / t_dyn[k]) * (1.0 - (ri_all[k] / RI_CRIT_DSI).max(0.0)).powi(2)); ... raw_dsi[k] = (inst_h.min(h_buf[k]).min(hp[k]) * v_tmp[k]).min(hp[k] * cs[k]);`), `:252` (`d_sh: empty()`), `:240-251` (`sm3`: `out[j] = 0.25 * arr[j - 1] + 0.5 * arr[j] + 0.25 * arr[j + 1]`).
- Upstream: `rotation_mix_info.f90:727-782` `set_D_DSI` (`height = min(instability_height, scale_height(k)); D = height*height/t_dyn(k); s% D_DSI(k) = min(D, scale_height(k)*csound(k))`), `:784-824` `set_D_SH` (`D = pow2(height*s% D_SH(k)*r(k)/grav(k))/t_dyn(k)`), `smooth_for_rotation(s, s% D_DSI, s% smooth_D_DSI, ...)` with `smooth_D_* = 0` by default (`controls.defaults:3923-3931`).
- Confidence: verified-by-reading-both-sides
- What: upstream D_DSI is `min(d_inst, Hp)^2 / t_dyn`. FlexMESA multiplies one `Hp/t_dyn` factor by `(1 - Ri/Ri_crit)^2` and replaces the other length by the minimum of three heights including a velocity scale height that upstream uses only for SSI/ES/GSF. `D_SH_factor` has no effect at all. The unconditional 3-point smoothing is applied where upstream applies none by default, and the `smooth_D_*` controls are not read. SSI, ES and GSF follow the upstream structure (`rotation_mix.rs:215-238` vs `rotation_mix_info.f90:826-1108`: `v_ssi = sqrt(visc/R_crit*|domega/dlnR|)`, `Ris_1 = 0.125*2500*Pr*Ri_T`, `(1 - max(Ris)/Ri_crit)^2`, `H = min(d_inst, H_v, Hp)`, cap `Hp*cs`), compared at the level of the listed expressions, not every guard.
- Proposed fix: M.

### F4-24: omega transport hides non-conservation by rescaling all omega, caps `am_sig_term_limit` at 1e10, and ignores torques
- Area: flexmesa-star / `rotation_mix.rs`
- Severity: wrong numerics
- FlexMESA: `rotation_mix.rs:318` (`let sig_cap = mw.am_sig_term_limit.clamp(1.0e4, 1.0e10);`), `:330` (second cap `sig_cap * denom_min / dt_transport`), `:372-380` (`let scale = if ... { j_tot_before / j_tot_after } else { 1.0 }; ... model.omega[k] *= scale;`), `:355-369` (one linear solve per substep, `break` on non-finite). grep `extra_jdot|extra_omegadot` in the crate: no hits.
- Upstream: `hydro_rotation.f90:437-490` `get1_am_sig` (`sig_term_limit = s% am_sig_term_limit`, default `1d13`), `solve_omega_mix.f90:62-68` (`max_steps = 20, min_steps = 4, max_iters_per_substep = 4, tol_correction_max = 1d-4`), `:190-250` (Newton iterations per substep; failure -> `retry`), `:296-302` (`if (abs(J_tot0 - J_tot1) > s% angular_momentum_error_retry*abs(J_tot0)) ... do_solve_omega_mix = retry`), `:524-660` (`rhs` includes `s% extra_jdot(k)`, `s% extra_omegadot(k)` and the `am_sig_j` i_rot-gradient terms).
- Confidence: verified-by-reading-both-sides
- What: (a) Upstream forces `omega(1) = omega(2)` after each iteration and then *checks* total J, retrying the step when the relative error exceeds `1e-6`. FlexMESA makes the same surface assignment and then multiplies every omega in the star by `J_before/J_after`, so the error introduced at the surface is redistributed over all layers and no retry can occur. (b) The stiffness cap is three orders of magnitude tighter than the default and is not controllable; with long timesteps this limits how close to solid-body a convective zone can get in one step. (c) `other_torque`/binary torques (`extra_jdot`, `extra_omegadot`) are not part of the solve, so magnetic braking and tidal torques cannot enter here. (d) `am_nu_j` (j-diffusion form) and `set_uniform_am_nu_non_rot`, `am_nu_omega_rot_factor`, `am_nu_omega_non_rot_factor` are absent (grep).
- What matches: harmonic-mean `nu`, `r_mid`, `rho(k)`, `/dm(k)` in `am_sig` (`rotation_mix.rs:322-327` vs `hydro_rotation.f90:467-472`); `c_omega = am_sig*i_rot_mid`; backward-Euler rows with `f = dt/(dmbar*i_rot)` and `dmbar(1) = 0.5*dm(1)` (`:342-350` vs `solve_omega_mix.f90:571-645`); 4 to 20 substeps from `0.5*min_mixing_timescale` (`:333-340` vs `:168-178`); `am_nu_*_factor < 0` falls back to `D_*_factor` (`:285-303` vs `mix_info.f90:1664-1670`); `nu_ST` (not `D_ST`) in the viscosity.
- Proposed fix: M.

### F4-25: rotational chemical mixing includes viscosity, lacks the high-T taper, and Spruit-Tayler time smoothing is absent
- Area: flexmesa-star / `rotation_mix.rs`
- Severity: wrong numerics
- FlexMESA: `rotation_mix.rs:264-268` (`let d_rot_k = mw.d_visc_factor.max(0.0) * diffs.d_visc[k] + mw.am_d_mix_factor.max(0.0) * (...)`).
- Upstream: `mix_info.f90:1604-1630` (`f` tapering from 1 at `D_mix_rotation_max_logT_full_on = 9.4` to 0 at `D_mix_rotation_min_logT_full_off = 9.5`; `s% D_mix_rotation(k) = f*s% am_D_mix_factor*( ! note: have dropped viscosity from mixing  D_DSI_factor*D_DSI + ...)`), `mix_info.f90:299-305` (`if (s% D_mix(k) < 1d-10) s% D_mix(k) = 0d0`), `rotation_mix_info.f90` ST block (`ST_angsml`, `ST_angsmt` limiter against the previous step's `D_ST`, `nu_ST`).
- Confidence: verified-by-reading-both-sides
- What: with `D_visc_factor > 0` FlexMESA mixes composition with the molecular/radiative viscosity, which upstream explicitly excludes. Rotational mixing is not switched off above log T = 9.4-9.5. The per-step growth limiter on the dynamo coefficients is missing, so `D_ST`/`nu_ST` can jump freely between steps.
- Proposed fix: S.

### Matches (area 3)

- fp, ft fit coefficients and `log_term_power`: `rotation.rs:52-57` vs `hydro_rotation.f90:51,788-792`. verified-by-reading-both-sides.
- Operator split of omega transport, half step before and half step after the structure solve when `premix_omega`: `evolve_support.rs:193-197`, `evolve.rs:243`, `rotation_mix.rs:393-396` (`0.5 * dt`) vs upstream `evolve.f90` (`premix_omega` half steps around `do_struct_burn_mix`; upstream call site not re-read line by line, labelled inferred for the upstream side).
- `j_rot` held fixed and `omega = j_rot/i_rot` re-derived after radius changes: `rotation.rs:267-300` vs `hydro_rotation.f90:277-287` (`set_omega`).
- Surface averages for the implicit-mdot test: tau window between `surf_avg_tau_min` and `surf_avg_tau` with partial first/last cell, mass-weighted `Lrad/Ledd`, `gamma_factor = 1 - min(avg, 0.9999)`, equatorial mid radius, `omega_crit = sqrt(gamma_factor*G*m_mid/r_mid^3)`, mass-weighted `|omega/omega_crit|`: `mdot_redo.rs:28-155` vs `hydro_rotation.f90:593-750`. The equatorial factor `1 + w^2/6 - 0.005124 w^4 + 0.06562 w^6` matches `re_from_rpsi_factor` (`:887-893`), but is evaluated with FlexMESA's `w` (F4-21). FlexMESA returns `w_div_w_crit_avg_surf`; the wind enhancement upstream uses `v_div_v_crit_avg_surf`, which is numerically the same ratio.
- Viscosity (`5.53e3*zbar*rho6^(5/6)*ctmp/abar^(1/3) + 4 a T^4/(15 c kap rho) + 1.893e6*rm23^2.5/(zbar*ctmp*xi2)`) and the `gamma`, `xi2`, `ctmp` fits: `rotation_mix.rs:108-119` vs `rotation_mix_info.f90` `setup`. FlexMESA additionally caps it at `Hp*cs`.
- Eddington-Sweet velocity pieces `ve0` (bracket term, `grada*omega^2*r^3*L/(-(gradT-grada)*delta*(G m)^2)`) and `ve_mu = (Hp/t_KH)*f_mu*B/(gradT-grada)`: `rotation_mix.rs:134-141` vs `setup`. `t_KH` omits upstream's neutrino luminosity term `L_neu`.
- `Ri_T`, `Ri_mu` with `ri0 = (rho*delta/P)*(dlnR/domega * g)^2`: `rotation_mix.rs:171-173` vs `setup`.
- Rotation control defaults listed under area 2 matches.

### Not reached (area 3)

- Remap of omega/j_rot in `mesh_remap.rs` on remesh (see area 4 for what was read).
- `set_D_omega`/`D_omega_flag`, `am_nu_rot_flag` time-dependent diffusivities: absent in FlexMESA.
- Tidal deformation hook (`rotation.rs:245-252`) vs `blend_tidal_values`: not compared.
- GSF edge handling at the centre (`rotation_mix_info.f90:1077-1100`) vs `fill_velocity_scale_heights`: not compared.

---

## Area 4: mesh remap (`mesh_remap.rs`, post-remesh pass in `hydro_state.rs`) vs `mesh_adjust.f90`

Read: `mesh_remap.rs` in full, `hydro_state.rs:505-598`, the driver
`mesh_adjust.rs:356-366`; upstream `do_mesh_adjust` (`mesh_adjust.f90:45-400`),
`do_L`, `do1_lnT`, `set_lnT_for_energy_with_tol`, `adjust_omega`. Not read:
upstream `do_lnR_and_lnd`, `do_xa`, `get1_lpp`, `do_v`, `do_u` bodies; the mesh
planner (F1-33/34 cover it).

The remap runs twice for changed cells: `remap_onto_dq` (piecewise-constant
conservative averages, `mesh_remap.rs:239-457`) and then
`smooth_post_remesh_hydro_and_rotation` (`hydro_state.rs:505-598`, called from
`evolve_support.rs:318`), which overwrites `lnd`, `lnT`, `L`, `omega` for
non-identical cells with higher-order reconstructions. The second pass is what
survives.

### F4-26: angular momentum on remesh is restored by rescaling omega in every cell of the star; upstream adds a solid-body correction in the changed region only
- Area: flexmesa-star / `hydro_state.rs`, `mesh_remap.rs`
- Severity: wrong numerics
- FlexMESA: `hydro_state.rs:559-561` (`model.omega[k] = pre.omega[kk_hint] + tf * (pre.omega[...] - pre.omega[kk_hint]);`), `:586-597` (`let sc = j_old / j_new; model.omega[..nz_new].iter_mut().for_each(|w| *w *= sc);`, two passes); first pass `mesh_remap.rs:397-441` uses `i_rot_k = (2/3) r^2` (`:437`).
- Upstream: `mesh_adjust.f90:1938-2013` `adjust_omega` (`j_error = sum(old_dqbar*old_j_rot) - sum(new_dqbar*j_rot)`; `correction_norm = sum over nzlo..nzhi of new_dqbar(k)*i_rot(k)`; `domega = j_error/correction_norm`; `s% j_rot(k) = s% j_rot(k) + domega*s% i_rot(k)% val` for `k = nzlo, nzhi`), after interpolating both `omega` and `j_rot` at the new face positions (`:234-246`).
- Confidence: verified-by-reading-both-sides
- What: both sides end with the same total J. Upstream confines the correction to the cells whose mesh changed and applies it as a uniform shift in omega, leaving untouched cells bit-identical. FlexMESA multiplies omega everywhere, so every remesh perturbs the rotation profile of the entire star, including cells flagged as exact copies, by the relative interpolation error. FlexMESA interpolates omega linearly and derives j; upstream interpolates j_rot and derives omega.
- Proposed fix: port `adjust_omega`. S.

### F4-27: remap omits upstream's degenerate-cell exception and the total-energy balance (`max_rel_delta_IE_for_mesh_total_energy_balance = 0.05`); L, mlt_vc, D_mix handled differently; EOS-inversion failures are ignored
- Area: flexmesa-star / `mesh_remap.rs`, `hydro_state.rs`
- Severity: wrong numerics
- FlexMESA: `hydro_state.rs:565-581` (energy target from limited linear reconstruction, then `let _ = crate::mesh_remap::set_lnt_for_energy(model, &e_target, &exact_flags);`), `mesh_remap.rs:119-208` (Newton on lnT with `step.clamp(-0.15, 0.15)`, `ln_t.clamp(3.0, 26.0)`, 20 iterations, no failure path), `hydro_state.rs:526` (`let cap = 1.6 * pre.energy[kk].abs() / pre.dq[kk]`), `:555-558` (L linear in q between old faces). grep `max_rel_delta_ie_for_mesh|eta_limit` in the crate: no hits.
- Upstream: `mesh_adjust.f90:1338-1520` `do1_lnT` (`:1414` `if (eta_old(k_old) >= eta_limit) then call store_lnT_in_xh(s, k, avg_lnT, xh)` with `eta_limit = -1d-6` at `:38`; `delta_energy = avg_PE + avg_KE - (new_PE + new_KE)` limited to `avg_energy*s% max_rel_delta_IE_for_mesh_total_energy_balance`, `energy_new(k) = avg_energy + delta_energy`; on inversion failure `new_lnT = avg_lnT`), default `0.05` (`controls.defaults:6310`); `do_L` (`interpolate_vector(..., interp_pm, ...)`); `do_interp_pt_val` for `mlt_vc` and `D_mix`.
- Confidence: verified-by-reading-both-sides for the listed lines; upstream integrals (`get_old_value_integral`) not read.
- What: (a) Upstream does not invert energy for T in any cell with `eta >= -1e-6` (partially or fully degenerate matter, i.e. most of a WD or an evolved core); it keeps the mass-averaged lnT. FlexMESA inverts energy everywhere. In degenerate matter `dlnE/dlnT` is tiny (FlexMESA floors it at `1e-4`, `mesh_remap.rs:162`), so the inversion is ill-conditioned there. (b) Upstream by default moves the change in potential plus kinetic energy caused by the remap into the internal energy of the new cell (up to 5% of it), conserving total energy; FlexMESA conserves internal energy only. (c) L: linear vs monotone piecewise-cubic. (d) `mlt_vc` and `D_mix` are not interpolated onto the new mesh; they are recomputed. For TDC, `mlt_vc_old` on the new mesh then has a different origin (see F1-4). (e) A failed or unconverged inversion is silently accepted.
- Proposed fix: M.

### Matches (area 4)

- Composition: per-species linear reconstruction with the smaller of the two one-sided slopes when they agree in sign, slope sum forced to zero by adjusting the extreme-slope species, all slopes zeroed if any species would leave [0, 1] at a cell edge, `xa_c0 = xa/sum`: `mesh_remap.rs:33-111` vs `mesh_adjust.f90:270-298` (driver part; `get1_lpp` body not read). Integration over the overlap then renormalisation: `mesh_remap.rs:340-367`. Species mass is conserved up to the final clamp/renormalise in `normalize_and_assign_remapped_xa` (`:12-28`), which upstream also does.
- Density from conserved volume (cumulative specific volume interpolated with a monotone Hermite, new `rho = dq / dV`), radii rebuilt from the centre: `hydro_state.rs:510-518,562-565,579` vs upstream `do_lnR_and_lnd` (by its interface and comments: r^3 interpolated, density from new volumes; body not read, labelled inferred on the upstream side).
- Unchanged cells copied verbatim: `mesh_remap.rs:304-327`, `hydro_state.rs:538-553` vs `cell_type(k) == unchanged_type` branches (`mesh_adjust.f90` `do1_lnT`).
- T from energy at fixed new density and composition (`mesh_adjust_get_T_from_E = .true.` default): `mesh_remap.rs:119-208` vs `set_lnT_for_energy_with_tol` (tolerance `1e-12` vs upstream `1e-11`).
- `dqbar` definition for face-centred conservation (`0.5*dq(1)`, `0.5*(dq(k-1)+dq(k))`, `0.5*dq(nz-1)+dq(nz)`): `mesh_remap.rs:210-226` vs `mesh_adjust.f90:137-146`.

---

## Area 5: termination, core masses, history definitions

Read: `termination.rs` (limit list and phase block), `core_boundaries.rs:1-337`,
`history.rs:55-100`; upstream `report.f90` `get_core_info`/`set_core_info`,
`star_utils.f90` `set_phase_of_evolution`, the `xa_central_*` block of
`do_one_utils.f90` `do_check_limits`, defaults. The rest of `do_check_limits`
(about 60 limits) was not compared one by one.

### F4-28: history `log_LH` is the total nuclear power, not hydrogen-burning power; `center_h1`/`center_he4` are X and Y of the centre cell
- Area: flexmesa-star / `history.rs`
- Severity: wrong numerics
- FlexMESA: `history.rs:71-72` (`let l_nuc_lsun = powers.power_nuc_burn.max(1.0e-30);`), `:85` (`log_lh: l_nuc_lsun.log10()`), `:74-75` (`center_h1 = model.xh_mass[c_idx]; center_he4 = model.yhe_mass[c_idx]`).
- Upstream: `history.f90:1689-1690` (`case(h_log_LH) val = safe_log10(s% power_h_burn)`); `report.f90:118-130` (`center_avg_x`: mass-weighted average over the innermost `center_avg_value_dq = 1d-8`).
- Confidence: verified-by-reading-both-sides for `log_LH`; `xh_mass`/`yhe_mass` semantics (X and Y totals rather than the single isotopes) verified-by-reading-flexmesa-only from their use as X and Y elsewhere (`winds_driver.rs:104-110`).
- What: the `log_LH` column cannot be compared with MESA's once helium burns. In `runs/fm/hb_2M.tsv`, `log_LH` (1.80) is close to `log_L` (1.75) throughout core He burning, as expected for total nuclear power, while MESA's `log_LH` there is the H-shell contribution only. `center_he4` includes he3; negligible in practice. The termination test reads the centre cell directly (`termination.rs:173`) with `<=` where upstream uses `center_avg_x` and `<` (`do_one_utils.f90:1279-1280`); negligible.
- Proposed fix: S.

### F4-29: `stop_at_phase_*` uses invented criteria
- Area: flexmesa-star / `termination.rs`
- Severity: wrong physics
- FlexMESA: `termination.rs:429-446` (`stop_at_phase_tp_agb && cores.co_core_mass > 0.45 && powers.power_he_burn > 1.0e3`; `stop_at_phase_c_burn && hr.center_he4 < 1.0e-2 && (hr.log_center_t >= 8.5 || powers.power_z_burn >= 1.0)`; `stop_at_phase_ne_burn && hr.log_center_t >= 9.0`; `stop_at_phase_o_burn && hr.log_center_t >= 9.15`). Only these four phases exist (grep).
- Upstream: `star_utils.f90:2923-2980` `set_phase_of_evolution` (`L_by_category(i_burn_si) > 1d2` -> Si_Burn; `(L_by_category(ioo) + L_by_category(ico)) > 1d2 .and. center_he4 < 1d-4` -> O_Burn; `L_by_category(i_burn_ne) > 1d2 .and. center_he4 < 1d-4` -> Ne_Burn; `L_by_category(icc) > 1d2 .and. center_he4 < 1d-4` -> C_Burn; `center_he4 < 1d-4 .and. he_core_mass - co_core_mass <= 0.1d0 .and. any(burn_he_conv_region)` -> TP_AGB; plus PreMS, ZAMS, IAMS, TAMS, He_Burn, ZACHeB, TACHeB, WDCS), controls `stop_at_phase_*` (`controls.defaults:1843-1855`).
- Confidence: verified-by-reading-both-sides
- What: none of the FlexMESA thresholds (0.45 Msun, 1e3 Lsun, logT 8.5 / 9.0 / 9.15, he4 < 1e-2) occur upstream; MESA defines phases by burning-category luminosities above 100 Lsun and central helium below 1e-4. Runs that stop on a phase stop at a different model. Nine of the thirteen phase stops are not implemented.
- Proposed fix: port `set_phase_of_evolution`. S.

### F4-30: default-active upstream stop conditions are off; `when_to_stop_rtol/atol` target-hitting absent
- Area: flexmesa-star / `controls_defaults.rs`, `termination.rs`
- Severity: wrong numerics
- FlexMESA: `controls_defaults.rs:108,110,133` (`log_center_temp_upper_limit: None`, `log_center_density_upper_limit: None`, `fe_core_infall_limit: None`), `:20-21` (`max_model_number: None`, `max_age: None`). grep `when_to_stop` in the crate: no hits.
- Upstream: `controls.defaults:957` (`log_center_density_upper_limit = 12d0`), `:970` (`log_center_temp_upper_limit = 11d0`), `:1688` (`fe_core_infall_limit = 3d7`), `:886` (`max_age = 1d36`), `:870-871` (`when_to_stop_rtol`, `when_to_stop_atol`, used by `compare_to_target` in `do_one_utils.f90` to redo a step so that a limit is hit within tolerance).
- Confidence: verified-by-reading-both-sides (defaults); `compare_to_target` not read beyond its call sites.
- What: confirms C1 lead L7. A core-collapse run with default controls stops in MESA at 3e7 cm/s iron-core infall and keeps going here. With the upstream defaults `1d99` the target-hitting logic is inert, so its absence only matters for inlists that set it.
- Proposed fix: S.

### Matches (area 5)

- Core-mass search order (outside in), the four boundary definitions in both modes (dominant-species mass number `A >= 47 / 20 / 16 / 4` when the fraction control is negative; otherwise `h1 <= he_core_boundary_h1_fraction && he4 >= min_boundary_fraction`, `he4 <= co_core_boundary_he4_fraction && c12+o16 >= min_x`, `he4+c12 <= one_core... && o16+ne20 >= min_x`, `si28 <= fe_core... && sum(A>=47) >= min_x`), neutron-rich core by `Ye`: `core_boundaries.rs:149-337` vs `report.f90:1054-1190`. Defaults 0.1 for all four fractions: `controls_timestep_stop.rs:324-329` vs `controls.defaults:771-828`.
- Sub-cell placement by the dominant-species crossing between cell centres with the same four special cases: `core_boundaries.rs:117-141` vs `report.f90:1229-1250`. Difference: in threshold mode FlexMESA interpolates on the threshold quantity (`:88-115`, e.g. where h1 crosses 0.1) instead of the dominant-species crossing that upstream always uses, and returns the midpoint where upstream raises `mesa_error` for same-sign differences. Sub-cell effect on `he_core_mass`.
- Limit names present in `termination.rs:144-460` correspond to upstream controls of the same name (list not checked for completeness or ordering).

---

## Leads (not verified)

- L1: `refresh_mixing_state` runs again after the solve (`struct_burn_mix.rs:554-556`), so end-of-step mixing diagnostics come from the end-of-step structure; upstream keeps start-of-step mixing info unless `recalc_mix_info_after_evolve` (default `.false.`, `controls.defaults:3565`). Affects `mass_conv_core`-type history values by up to one step of boundary motion. Call order read on both sides; consequence not measured.
- L2: upstream freezes mixing info after phase separation and before element diffusion (`evolve.f90:632`: `s% okay_to_set_mixing_info = .false.`); FlexMESA recomputes it after diffusion inside `run_struct_burn_mix_core`. Matters only with `do_element_diffusion`.
- L3: `tier1_extras.rs:435-446` calls `do_conv_premix(model, 3, false)` and `add_predictive_mixing(model, 1.0e-2, 1.0e-4)` for named cases in addition to the generic call sites (C1 lead L4). I confirmed the generic sites exist (`evolve_support.rs:120`, `struct_burn_mix.rs:421`); whether both fire in the same step for those cases was not traced.
- L4: `update_rotation_distortion_factors` sets `j_rot = i_rot*omega` (`rotation.rs:262`) and is called from `set_mlt_vars` when the j-conserving path is not taken (`hydro_vars.rs:40-44`); whether any path lets omega rather than j be the conserved quantity across a radius change was not traced.
- L5: the overshoot Brunt test reads the smoothed `brunt_b` where upstream reads `unsmoothed_brunt_B`; with F4-1's leak this can stop overshoot a few cells early. Not quantified.
- L6: `do_set_mdot` uses `model.lum[0]` and `m_grav[0]`; upstream passes `L_phot*Lsun` and `mstar` (`evolve.f90:427`). Differences only with `use_mass_corrections` or when the photosphere is inside the model.
- L7: `hb_2M` attribution (F4-2) should be tested by a run with `use_Ledoux_criterion = .false.` (Schwarzschild core, no B term): if the FlexMESA lifetime moves most of the way toward MESA's, F4-1/F4-2 are confirmed as dominant.

## Citation note

FlexMESA line numbers were checked with `grep -n` against the worktree after
writing; upstream line numbers for the quoted statements were checked the same
way against `0b2e6bb`. Ranges give the extent of the block containing the quoted
line and may be a few lines wide of the exact statement.

## What was not reached (overall)

- Area 1: `locate_mixing_boundaries`, `set_cz_bdy_mass` consumers, `mixing_D_limit_for_log`, RTI mixing, `check_for_redo_mlt` upstream side, upstream `predictive_mix.f90` body (C1-16 scope).
- Area 2: `get_xa_for_accretion`, `make_compressed_intersect`/`leak` line by line, `D_omega`/`am_nu_rot` remap on mass change.
- Area 3: tidal blending, `D_omega_flag`/`am_nu_rot_flag` paths, GSF centre edge cases, the upstream `premix_omega` call site.
- Area 4: upstream `do_lnR_and_lnd`, `do_xa`, `get1_lpp`, `do_v`, `do_u`, `get_old_value_integral` bodies; FlexMESA `v`/`u` reconstruction vs upstream kinetic-energy-conserving `do_v`/`do_u`.
- Area 5: most of `do_check_limits` (limit-by-limit comparison and ordering), `compute_power_categories` vs `set_power_info`, profile output.
- No code was executed; every behavioural consequence is argued from source. The `hb_2M` attribution in F4-2 is the main item that needs a confirming run (lead L7).
