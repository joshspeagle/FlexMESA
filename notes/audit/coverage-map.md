# C1 (Task A): upstream coverage map, MESA `0b2e6bb` vs FlexMESA PR #138 (`91eda0b`)

> **Reference note (`#141`):** Line numbers, commit diffs, and pre-rebaseline measurements in this historical audit register refer to MESA `main` at commit `0b2e6bb` (2026-11-04) and `FlexMESA` commit `91eda0b`. Active repository inputs, module goldens, `case_table.md`, and `defaults_diff_26_4_1.md` target **MESA release `26.4.1`** (tag `26.4.1`, commit `c8c2026`, Zenodo `19722306`).

Upstream root `the upstream checkout`, FlexMESA root `the audited tree`. LOC = `wc -l` of the upstream file.
Status: PORTED / PARTIAL / STUB / ABSENT. "quick" = classified from the mechanical cross-reference (grep of upstream
file name and of every upstream subroutine name of length > 5 in `crates/flexmesa-{star,binary,astero}/src`) plus a skim
of the Rust function list, not a line-by-line read of both sides. "deep" = both sides opened and compared.
Rust sizes for scale: `flexmesa-star/src` = 108 files, ~46.6k lines including ~2.2k lines of in-crate tests
(upstream `star/private` = 133.6k lines, `star/job` 3.9k, `star/public` 3.4k).

Written incrementally. Totals are at the end.

## 1. `star/private` files with no counterpart at all (ABSENT, quick; 0 upstream subroutine names found in Rust)

| upstream path | LOC | status | FlexMESA file | evidence |
| :-- | --: | :-- | :-- | :-- |
| star/private/pgstar_abundance.f90 | 403 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_color_magnitude.f90 | 936 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_colors.f90 | 206 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_ctrls_io.f90 | 6567 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_dpg_dnu.f90 | 86 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_dynamo.f90 | 316 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_equation_residuals.f90 | 263 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_full.f90 | 1762 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_grid.f90 | 1039 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_hist_track.f90 | 838 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_history_panels.f90 | 966 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_hr.f90 | 261 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_kipp.f90 | 816 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_kipp_residuals.f90 | 428 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_l_r.f90 | 89 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_l_teff.f90 | 89 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_l_v.f90 | 89 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_logg_logt.f90 | 90 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_logg_teff.f90 | 90 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_logl_r.f90 | 97 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_logl_teff.f90 | 89 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_logl_v.f90 | 94 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_mixing_ds.f90 | 554 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_mode_prop.f90 | 291 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_network.f90 | 284 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_power.f90 | 280 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_production.f90 | 351 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_profile_panels.f90 | 1008 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_r_l.f90 | 89 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_r_teff.f90 | 89 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_rti.f90 | 489 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_stub.f90 | 187 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_summary.f90 | 449 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_summary_burn.f90 | 303 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_summary_history.f90 | 283 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_summary_profile.f90 | 307 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_support.f90 | 1770 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_tmaxrho.f90 | 109 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_trho.f90 | 109 | ABSENT | none | no plotting layer in Rust; quick |
| star/private/pgstar_trho_profile.f90 | 566 | ABSENT | none | no plotting layer in Rust; quick |

pgstar subtotal: 23132 LOC in 40 files (python `src/flexmesa/plotting` is a separate matplotlib layer, not a port).

## 2. Controls coverage (namelists)

Method: names extracted from upstream `*.defaults` (first-column assignments, lowercased, de-duplicated, array
subscripts dropped). "read" = the name occurs as a word in non-comment lines of the Rust inlist-reading files
(`flexmesa-star/src/inlist*.rs`, `rsp_config.rs`; `flexmesa-binary/src/inlist.rs`). "referenced" = the same word also
occurs in non-comment lines of some other non-test, non-`controls*.rs`/`job_controls.rs`/`model.rs` file of the crate.
Both are word-match heuristics: controls the parser stores under a different field name are counted as read but not
referenced (false negative), and a word appearing in unrelated code is a false positive. Treat "referenced" as
+/- 10%.

| namelist | upstream names | read by FlexMESA parser | read and referenced elsewhere | not read |
| :-- | --: | --: | --: | --: |
| `star/defaults/controls.defaults` (`&controls`) | 1375 | 593 (43%) | ~485 (35%) | 782 (57%) |
| `star/defaults/star_job.defaults` (`&star_job`) | 419 | 104 (25%) | ~59 (14%) | 315 (75%) |
| `binary/defaults/binary_controls.defaults` | 179 | 111 (62%) | ~111 | 68 |
| `binary/defaults/binary_job.defaults` | 30 | 5 | 5 | 25 |
| `eos/defaults/eos.defaults` (`&eos`) | 102 | 5 (`use_FreeEOS`, `use_OPAL_SCVH`, `use_PC`, `use_Skye`, `use_CMS`) | 5 | 97 |
| `kap/defaults/kap.defaults` (`&kap`) | 36 | 5 (`Zbase`, `kap_file_prefix`, `kap_CO_prefix`, `kap_lowT_prefix`, `use_Type2_opacities`) | 5 | 31 |
| `star/defaults/pgstar.defaults` | 2939 | 0 | 0 | all |

(eos/kap counts: `crates/flexmesa-star/src/inlist_populate.rs:259-293`, read in full, verified-by-reading-flexmesa-only.)

### C1-13: Unknown and unsupported controls are silently ignored; unknown option strings silently map to a default
- Area: flexmesa-star / `inlist.rs`, `inlist_populate.rs`
- Severity: wrong physics (silent) / unverified claim
- FlexMESA: `crates/flexmesa-star/src/inlist.rs:34-47` (`SectionControls::insert` stores every assignment in a `HashMap`; no list of known names; grep for `unknown|unsupported|unrecognized|warn` in `inlist*.rs` finds nothing); `inlist_populate.rs:41-73` (`f64_into` etc. copy a value only if a known key is present); `inlist_populate.rs:378-390` (`mlt_option`: `_ => MltOption::Cox`), `:392-397` (`atm_option`: `_ => AtmOption::TTau`, so `'irradiated_grey'`, `'Paczynski_grey'`, `'fixed_Teff'`, `'fixed_Tsurf'`, `'fixed_Psurf'`, `'fixed_Psurf_and_Tsurf'`, `'WD_tau_25'` all become `T_tau`), `:403-409` (`atm_T_tau_relation` unknown -> Eddington), `:411-417` (`screening_mode` unknown -> Extended), `:419-423` (`energy_eqn_option` anything but `'eps_grav'` -> dedt), `:425-433` (semiconvection/thermohaline parse errors ignored).
- Upstream: a Fortran namelist read fails on an unknown name (`star/private/ctrls_io.f90` `read_controls_file`: `read(unit, nml=controls, iostat=ierr)` then "Failed while trying to read control namelist file"); unknown option strings are errors at use (e.g. `atm_support.f90` "Unknown value for atm_option").
- Confidence: verified-by-reading-flexmesa-only; upstream behaviour is standard namelist semantics, `ctrls_io.f90` not opened line by line (inferred).
- What: 57% of `&controls` and 75% of `&star_job` names have no effect, without any diagnostic. Every upstream test inlist therefore "parses", but an unknown fraction of each inlist's physics settings are dropped. A run cannot tell the user which of their settings were honoured.
- Proposed fix: keep a registry of supported names; error (or at least warn) on any assignment not in it, and on unknown option strings. M.

### C1-14: Default-value mismatches against upstream `*.defaults`
- Area: flexmesa-star / `controls_defaults.rs`, `controls_mix_wind.rs`, `controls_timestep_stop.rs`, `job_controls.rs`
- Severity: wrong physics (defaults)
- Confidence: verified-by-reading-both-sides for the rows marked V (both lines printed); the rest from the automated comparison (549 same-named scalar defaults compared, 108 flagged, most of the flags being `Option::None` standing for an upstream sentinel such as `1d99`/`-1`/`0`).
- Method: every upstream scalar default whose lowercased name equals a field name in a Rust `Default` body was compared numerically.

Real mismatches (upstream `star/defaults/controls.defaults:<line>` vs FlexMESA `crates/flexmesa-star/src/<file>:<line>`):

| control | upstream | FlexMESA | |
| :-- | :-- | :-- | :-- |
| `MLT_option` | `'TDC'` (:2203) | `MltOption::Cox` (`controls_defaults.rs:26`) | V |
| `varcontrol_target` | `1d-3` (:9636) | `1.0e-4` (`controls_defaults.rs:55`) | V |
| `min_D_mix` | `1d3` (:3585) | `0.0` (`controls_defaults.rs:33`) | V |
| `screening_mode` | `'chugunov'` (:6657) | `ScreeningMode::Extended` (`controls_defaults.rs:49`) | V |
| `Lnuc_div_L_zams_limit` | `0.9d0` (:1796) | `0.99` (`controls_defaults.rs:97`) | V |
| `do_premix_heating` | `.true.` (:2923) | `false` (`controls_defaults.rs:183`) | V |
| `do_phase_separation_heating` | `.true.` (:7434) | `false` (`controls_defaults.rs:69`) | V |
| `conv_premix_avoid_increase` | `.false.` (:2940) | `true` (`controls_mix_wind.rs:358`) | V |
| `max_center_cell_dq` | `1d-7` (:5845) | `1.0e-6` (`controls_mix_wind.rs:416`) | V |
| `max_surface_cell_dq` | `1d-12` (:5858) | `1.0e-8` (`controls_mix_wind.rs:415`) | V |
| `P_function_weight` | `40` (:5930) | `30.0` (`controls_mix_wind.rs:417`) | V |
| `delta_lgP_cntr_limit` | `-1` (off) (:10719) | `0.05` (`controls_timestep_stop.rs:435`) | V |
| `delta_lg_XH_drop_only`, `..XHe..`, `..XC..` | `.false.` (:11209-11211) | `true` (`controls_timestep_stop.rs:443,448,453`) | V (first) |
| `atm_T_tau_errtol` | `1d-7` (:4316) | `1.0e-4` (`rsp_config.rs:291`) | |
| `accretion_h1`, `accretion_he4` | `0` (:5521,:5524) | `-1.0` (`controls_defaults.rs:83-84`) | |
| `fe_core_infall_limit` | `3d7` (:1688) | `None` = off (`controls_defaults.rs:133`; `termination.rs:58` only tests when `Some`) | V |
| `log_center_temp_upper_limit` | `11d0` (:970) | `None` (`controls_defaults.rs:108`) | V |
| `log_center_density_upper_limit` | `12d0` (:957) | `None` (`controls_defaults.rs:110`) | V |
| `atm_table` | `'tau_100'` (:4385) | empty (`controls_defaults.rs:37`) | |
| `zams_filename` | `'zams_z2m2_y28.data'` (:11561) | empty (`controls_defaults.rs:19`) | |
| `pulse_data_format` | `'FGONG'` (:571) | `"GYRE"` (`job_controls.rs:459`) | |
| `RSP_mass/Teff/L/X/Z` | `-1` (:3063-3067) | `0.65/6500/45/0.75/0.0004` (`controls_defaults.rs:145-149`) | |
| `velocity_q_upper_bound`, `velocity_tau_lower_bound`, `velocity_logT_lower_bound` | `1d99/-1d99/-1d99` (:8027,:8039,:8051) | `1.0/0.0/0.0` (`controls_defaults.rs:221-223`) | equivalent in practice |

`star/defaults/star_job.defaults:<line>` vs `crates/flexmesa-star/src/job_controls.rs:<line>`:

| control | upstream | FlexMESA | |
| :-- | :-- | :-- | :-- |
| `pre_ms_T_c` | `3d5` (:396) | `9.0e5` (:438) | V |
| `pre_ms_d_log10_P` | `0` (:416) | `0.004` (:440) | V |
| `pre_ms_relax_num_steps` | `300` (:441) | `20` (:444) | V |
| `auto_extend_net` | `.true.` (:2438) | `false` (:465) | V |
| `core_avg_eps` | `1d-6` (:1822) | `0.0` (:505) | V |
| `save_photo_when_terminate` | `.true.` (:270) | `false` (:455) | |

Also not an upstream control at all: `min_logt_for_net` (default `6.3`, `controls_defaults.rs:43`), read from the inlist
at `inlist_populate.rs:313`; upstream has only `max_logT_for_net` (`controls.defaults:6759`).

Matches: the other ~440 compared scalar defaults agree exactly (e.g. `op_split_burn = .false.`, `op_split_burn_min_T = 2d9`,
`op_split_burn_eps = 1d-5`, `max_abar_for_burning = 199`, `max_logT_for_net = 10.2`).

## 3. `star/private`, all non-pgstar files (104 files)

Rust file paths are under `crates/flexmesa-star/src/` unless a crate is named. Numbers in parentheses are Rust line counts. "n/m names" = upstream routine names (length > 5) found anywhere in the three Rust crates.

| upstream path | LOC | status | FlexMESA file(s) | evidence |
| :-- | --: | :-- | :-- | :-- |
| `star/private/adjust_mass.f90` | 1624 | PARTIAL | adjust_mass.rs (598) | quick. Three-region `do_adjust_mass` and `j_rot` remap present; `set_D_omega`, `angular_momentum_removed` bookkeeping, `set_xa` for accreted composition via `get_xa_for_accretion` not found (3/18 routine names) |
| `star/private/adjust_mesh.f90` | 700 | PARTIAL | evolve_support.rs `remesh_star_before_step`, mesh_adjust.rs | quick. Driver only; `set_types_of_new_cells`, `okay_to_merge` logic, gval work arrays not present |
| `star/private/adjust_mesh_split_merge.f90` | 1461 | ABSENT | none | quick. `use_split_merge_amr` path: 0/22 routine names in Rust |
| `star/private/adjust_mesh_support.f90` | 457 | STUB | mesh_adjust.rs `eval_mesh_gval_parts` | deep (C1-18). No `mesh_dlog_*_dlogP_extra`, `xtra_coef_*`, `mesh_delta_coeff_for_highT`, `E_function`, burn-zone multipliers |
| `star/private/adjust_net.f90` | 249 | ABSENT | none | quick. `auto_extend_net` is read (job_controls.rs:465) but unused; tier1 extras hard-code `cno_extras.net` for the `adjust_net` case (tier1_extras.rs:425) |
| `star/private/adjust_xyz.f90` | 1468 | PARTIAL | relax.rs `apply_composition_relaxation`, relax_comp.rs, adjust_mass.rs `build_accreted_xa` | quick. Uniform/Y/Z rescale present (per-zone scaling of H, He, metals groups); `set_standard_composition`/zfracs choices, `set_abundance_ratios`, `change_net` remap rules, `get_xa_for_standard_metals` not found (6/31 names) |
| `star/private/alloc.f90` | 3248 | ABSENT | model.rs, hydro_state.rs (Vec-based) | infrastructure: functionally replaced by Rust ownership; no port needed. Counted under "infrastructure replaced" |
| `star/private/atm_support.f90` | 1262 | PARTIAL | hydro_brunt_surf.rs `eval_surf_bc` (:165-408) | deep. Has `T_tau` with fixed opacity (4 relations), `table` (7 tables, off-table blend), `simple_photosphere`. Missing: `irradiated_grey`, `Paczynski_grey`, `fixed_Teff/Tsurf/Psurf/Psurf_and_Tsurf`, `atm_T_tau_opacity = varying/iterated`, `build_atm` (atmosphere structure for pulse output), `T_tau_gradr_factor`. Adds a non-upstream table proxy (C1-15) |
| `star/private/auto_diff_support.f90` | 1460 | PARTIAL | star_ad.rs (185), wrappers inline in hydro_*.rs, rsp2_ad.rs | quick. 3/118 wrapper names; Rust builds stencil duals ad hoc |
| `star/private/brunt.f90` | 335 | PARTIAL | hydro_brunt_surf.rs `do_brunt_b`, `do_brunt_n2` | quick. Core formula present; smoothing implemented separately in hydro_state.rs; `brunt_B` via MHM form with EOS re-evaluation not checked |
| `star/private/conv_premix.f90` | 1568 | STUB | premix.rs `do_conv_premix` (:94-185) | deep (C1-16). Fixed 8-cell window homogenisation; no `advance_bdy` fractional-cell ingestion, no burn-type `avoid_increase`, no `conv_premix_time_factor` |
| `star/private/create_initial_model.f90` | 501 | STUB | star_job_init.rs `build_initial_planet_or_wd_model` (:74-113) | deep (C1-1). Rescales the 1 Msun ZAMS model; no envelope integration from (M, R) |
| `star/private/ctrls_io.f90` | 4195 | PARTIAL | inlist.rs, inlist_populate*.rs, flexmesa-schema | deep. 593/1375 `&controls` names read; unknown names silently ignored (C1-13); no `write_controls`/`get_control`/`set_control` |
| `star/private/diffusion.f90` | 2050 | STUB | diffusion.rs `do_element_diffusion` (:421-567) | deep (C1-16). One backward-Euler step with a single mean D; no coupled Burgers solve over cells, no substeps, no classes |
| `star/private/diffusion_procs.f90` | 1746 | STUB | diffusion.rs `eval_burgers_velocities` (:220-378) | deep-ish. Local per-zone Burgers solve; none of the 17 upstream routine names |
| `star/private/diffusion_support.f90` | 1800 | STUB | diffusion.rs `eval_coulomb_collision_factors` (:67-118) | deep. Closed-form blends for collision integrals; 0/16 names |
| `star/private/do_one_utils.f90` | 1540 | PARTIAL | termination.rs (492) | quick. `do_check_limits` stop conditions partly present with `Option` limits; terminal/log output and `do_one_check_model` flow absent |
| `star/private/element_diffusion.f90` | 582 | STUB | evolve_support.rs:144-156 | deep. Single call with hard-coded `DiffusionScheme::Paquette`; `diffusion_class_*`, `diffusion_min_*`, `diffusion_dt_limit`, rad-lev (`op_mono`) not wired |
| `star/private/eos_support.f90` | 413 | PARTIAL | micro.rs `do_eos`, pre_ms.rs (`eos_pt_get`) | quick. `get_eos` path only; `solve_eos_given_DE/PgasT/...` helpers and `eos_frac_*` reporting not found (0/8 names) |
| `star/private/eps_grav.f90` | 408 | PORTED | eps_grav.rs (235) | deep-ish. `do_std_eps_grav` and `do_lnS_eps_grav` present; blend weight differs (C1-20); composition term via partials rather than EOS re-evaluation at `xa_start` |
| `star/private/eps_mdot.f90` | 706 | PORTED | eps_mdot.rs (592) | quick. 4/6 routine names, comparable size |
| `star/private/evolve.f90` | 2404 | PARTIAL | evolve.rs (584), evolve_support.rs (482), mdot_redo.rs (417) | deep. Step/retry loop, implicit-mdot redo (`select_mdot_action`), energy-conservation accounting present. Missing: `do_step_part1` pieces (adjust_net, remove shells, `do_mesh` variants), redo (`prepare_to_redo`) as a driver action, `report_problems`, most `other_*` hooks, `set_age` details; non-upstream retry factors (C1-20) |
| `star/private/evolve_support.f90` | 305 | PARTIAL | model.rs/mesh.rs `save_old_values`, evolve.rs `restore_from_old` | quick. Generation bookkeeping re-implemented |
| `star/private/gravity_darkening.f90` | 149 | ABSENT | none (tier3_physics.rs has a closed-form diagnostic only) | quick. 0/4 names; upstream table interpolation absent |
| `star/private/history.f90` | 3510 | STUB | history.rs (227) | deep. 17 fixed columns vs upstream ~400 selectable; no `history_columns.list` |
| `star/private/history_specs.f90` | 801 | ABSENT | none | quick. grep `history_columns` in crates/flexmesa-star: none |
| `star/private/hydro_alpha_rti_eqns.f90` | 195 | PORTED | hydro_rti.rs `do1_dalpha_rti_dt_eqn` | quick |
| `star/private/hydro_chem_eqns.f90` | 224 | PORTED | hydro_eqns.rs `do1_chem_eqns` (:377-498) | quick |
| `star/private/hydro_energy.f90` | 904 | PORTED | hydro_energy.rs (574), hydro_work.rs (331) | quick. 11/17 routine names; not checked term by term (another auditor) |
| `star/private/hydro_eqns.f90` | 1164 | PARTIAL | hydro_eqns.rs (589) | quick+. `do1_density_eqn`, surface BC eqns, chem, assembly present. Missing: `do1_w_div_wc_eqn`, `do1_dj_rot_dt_eqn` (rotation variables are not solver unknowns), `set_compression_BC`, `fix_d_eos_dxa_partials` (grep) |
| `star/private/hydro_momentum.f90` | 610 | PORTED | hydro_momentum.rs (478) | quick. 6/15 names incl. `do1_momentum_eqn`, `do1_radius_eqn` |
| `star/private/hydro_riemann.f90` | 463 | PARTIAL | riemann.rs, riemann_ad.rs, riemann_eqns.rs (781) | quick. HLLC faces present; PPM/face reconstruction absent (see reconstructed_face_support) |
| `star/private/hydro_rotation.f90` | 927 | PARTIAL | rotation.rs (346) | quick. `fp/ft`, `omega_crit`, shell `j_rot` conservation; `w_div_wc` as a variable, `set_i_rot` with deformation, `set_rotation_info` flow largely absent (3/26 names) |
| `star/private/hydro_rsp2.f90` | 1255 | PARTIAL | rsp2.rs, rsp2_ad.rs, rsp2_eqns.rs (1209) | quick. 12/34 names |
| `star/private/hydro_rsp2_support.f90` | 444 | PARTIAL | rsp2_remesh.rs (290) | quick. 1/14 names |
| `star/private/hydro_temperature.f90` | 509 | PARTIAL | hydro_temperature.rs (191) | quick. `do1_gradT_eqn`, `do1_alt_dlnT_dm_eqn`, `do1_dlnT_dm_eqn` present; 191 vs 509 lines |
| `star/private/hydro_vars.f90` | 1165 | PARTIAL | hydro_vars.rs (559), micro.rs, hydro_brunt_surf.rs | deep. `set_hydro_vars` order of micro/MLT/brunt/surf present; `set_vars_if_needed`/`update_vars` skip logic, `set_Teff` variants, `set_cgrav`/`other_cgrav` flow simplified |
| `star/private/init.f90` | 1233 | STUB | star_job_init.rs (596) | deep (C1-1, C1-5). Model selection branches on case name and mass range; `do_star_init`, `do_load1_star`, `do_relax_to_radiative_core` only loosely mirrored |
| `star/private/init_model.f90` | 468 | PORTED | zams.rs (596) | deep-ish. `get_zams_model` log-mass interpolation present; bypassed for many runs by C1-1 |
| `star/private/ionization_potentials.f90` | 596 | PORTED | flexmesa-ionization/src/potentials_data.rs | quick. Data table lives in the ionization crate |
| `star/private/kap_support.f90` | 526 | PARTIAL | micro.rs `do_kap` (:409-483) | quick. Single `kap_get`; `op_mono`, `use_other_kap`, compton/`kap_frac_*` bookkeeping not found (0/5 names) |
| `star/private/magnetic_diffusion.f90` | 137 | PARTIAL | rotation.rs `eval_magnetic_diffusivity` | quick. 0/5 names; one function |
| `star/private/mass_utils.f90` | 594 | PARTIAL | core_boundaries.rs, eps_mdot.rs | quick. 2/9 names |
| `star/private/mesh_adjust.f90` | 2789 | PARTIAL | mesh_remap.rs (457), hydro_state.rs `smooth_post_remesh_hydro_and_rotation` | quick. xh/xa conservative remap with monotone slopes and `set_lnT_for_energy` present; `do_prune_mesh_surface`, `adjust_omega`, `do_v`/`do_u` momentum-conserving remaps, RSP2 `Hp_face`/`etrb` remaps not found (5/35 names) |
| `star/private/mesh_functions.f90` | 327 | STUB | mesh_adjust.rs `eval_mesh_gval_parts` (:156-235) | deep (C1-18). 7 functions; no `log_tau`, `log_kap`, per-species xa functions |
| `star/private/mesh_plan.f90` | 1302 | STUB | mesh_adjust.rs `plan_mesh` (:259-349) | deep (C1-18). Different algorithm; 0/14 routine names |
| `star/private/micro.f90` | 766 | PARTIAL | micro.rs (495) | quick. `set_micro_vars` sequence present; `set_eos_with_mask`, `shutdown_microphys`, `other_*` hooks absent |
| `star/private/mix_info.f90` | 2421 | PARTIAL | mix_info.rs (425), hydro_rti.rs (395), struct_burn_mix.rs `refresh_mixing_state` | deep-ish. Boundary location, sigmas, glitch removal, RTI mixing present. Missing or simplified: `set_cz_bdy_mass`, `locate_mixing_boundaries`, `do_mix_envelope`, `sig_min_factor_*`, per-type gap controls (`min_convective_gap`, ...; hard-coded `0.1 Hp`, struct_burn_mix.rs:419), `set_use_gradr` |
| `star/private/mod_typical_charge.f90` | 111 | PORTED | flexmesa-ionization/src/typical_charge.rs | quick |
| `star/private/net.f90` | 813 | PARTIAL | micro_net.rs (242), micro.rs | quick. `do_net` present; `set_rate_factors`/`other_net_get`, `op_mono_factors`, `net_tables`, weak-rate options, `save_rates` absent; adds non-upstream `min_logt_for_net` (C1-9) |
| `star/private/neu.f90` | 129 | PORTED | micro.rs `do_neu` (:349-408) | quick |
| `star/private/overshoot.f90` | 327 | PARTIAL | overshoot.rs `apply_overshooting` | quick. Param-set matching by zone type/loc/bdy and mass ramp present |
| `star/private/overshoot_exp.f90` | 176 | PARTIAL | overshoot.rs | quick |
| `star/private/overshoot_step.f90` | 172 | PARTIAL | overshoot.rs | quick |
| `star/private/overshoot_utils.f90` | 376 | PARTIAL | overshoot.rs | quick. 2/4 names (`eval_conv_bdy_*`) |
| `star/private/paquette_coeffs.f90` | 721 | ABSENT | none | deep. Upstream spline tables absent; diffusion.rs:93-102 uses closed-form blends |
| `star/private/phase_separation.f90` | 912 | STUB | premix.rs `do_phase_separation` (:301-397) | deep (C1-16). Phase-diagram coefficients do not match `blouin_delta_xo` (:418-423); no Medin-Cumming 3-component; no `mix_outward` |
| `star/private/photo_in.f90` | 314 | PARTIAL | photo_io.rs `read_photo_bytes` | quick. Own binary format; cannot read upstream photos; equivalent role |
| `star/private/photo_out.f90` | 179 | PARTIAL | photo_io.rs `write_photo_bytes` | quick. Own format |
| `star/private/pre_ms_model.f90` | 738 | PORTED | pre_ms.rs (511) | deep-ish. Integration scheme and tolerances match; defaults and composition differ (C1-14, Matches) |
| `star/private/predictive_mix.f90` | 974 | STUB | premix.rs `add_predictive_mixing_in_q_range` (:187-249) | deep (C1-16). No trial mixing or re-evaluation |
| `star/private/profile.f90` | 1186 | STUB | history.rs `format_profile_string` | quick. Fixed columns; no `profile_columns.list` |
| `star/private/profile_getval.f90` | 2515 | ABSENT | none | quick. 0/14 names |
| `star/private/pulse.f90` | 185 | PARTIAL | pulse_exporters.rs, pulse.rs | quick |
| `star/private/pulse_cafein.f90` | 722 | PARTIAL | pulse_exporters.rs | quick. 0/6 names; exporter present by format name |
| `star/private/pulse_fgong.f90` | 710 | PARTIAL | pulse.rs | quick |
| `star/private/pulse_gr1d.f90` | 182 | PARTIAL | pulse_exporters.rs | quick |
| `star/private/pulse_gsm.f90` | 130 | ABSENT | none (HDF5 GSM output) | quick. Format name appears only in comments |
| `star/private/pulse_gyre.f90` | 476 | PARTIAL | pulse.rs | quick |
| `star/private/pulse_osc.f90` | 608 | PARTIAL | pulse.rs | quick |
| `star/private/pulse_saio.f90` | 347 | PARTIAL | pulse_exporters.rs | quick |
| `star/private/pulse_utils.f90` | 525 | PORTED | pulse_utils.rs (568) | quick. 6/9 names |
| `star/private/read_model.f90` | 977 | PARTIAL | model_io.rs `read_mod_str` (:235-529) | quick. Reads header and columns; `read_properties` variants, `mlt_vc`/`RSP` extras partially; error handling replaced by defaults (lead L6) |
| `star/private/reconstructed_face_support.f90` | 534 | ABSENT | none | quick. 0/15 names; no PPM reconstruction |
| `star/private/relax.f90` | 4215 | STUB | relax.rs (573), relax_comp.rs (246) | deep (C1-2, C1-3, C1-4). 11/126 routine names; no `do_internal_evolve`-driven relaxations |
| `star/private/remove_shells.f90` | 1397 | ABSENT | relax.rs `excise_core_by_temperature` only | quick. 0/32 names; one of ~15 remove_* operations has an analogue |
| `star/private/report.f90` | 1419 | PARTIAL | core_boundaries.rs (443), history.rs | quick. Core masses and power categories; most of `do_report` integrals absent |
| `star/private/rotation_mix_info.f90` | 1371 | PARTIAL | rotation_mix.rs (398), rotation.rs | quick. Heger-type DSI/SH/SSI/ES/GSF and Spruit-Tayler present by name; smoothing and `D_omega`/`am_nu` time smoothing not found (2/13 names) |
| `star/private/rsp.f90` | 1030 | PORTED | rsp_driver.rs, rsp_build.rs, rsp_stats.rs | quick. 14/19 names |
| `star/private/rsp_build.f90` | 986 | PORTED | rsp_build.rs, rsp_stah.rs, rsp_grid.rs | quick |
| `star/private/rsp_def.f90` | 870 | PORTED | rsp_def.rs, rsp_grid.rs, rsp_state_io.rs | quick |
| `star/private/rsp_eval_eos_and_kap.f90` | 1047 | PORTED | rsp_eos_inv.rs, rsp_eos_kap.rs | quick. 15/16 names |
| `star/private/rsp_lina.f90` | 1269 | PORTED | rsp_lina.rs, rsp_lina_io.rs, rsp_turb_partials.rs | quick |
| `star/private/rsp_relax_env.f90` | 1335 | PORTED | rsp_relax_env.rs | quick |
| `star/private/rsp_step.f90` | 3834 | PORTED | rsp_hyd*.rs, rsp_eqns*.rs | quick. 36/53 names; Rust ~3.2k lines vs 3.8k |
| `star/private/set_flags.f90` | 792 | STUB | star_job_init.rs:379-419 (inline resize) | quick. 1/16 names; no remap of state when a flag changes (e.g. `set_v_flag` initialisation, `set_RSP2_flag` conversion beyond rsp2_remesh.rs) |
| `star/private/solve_omega_mix.f90` | 669 | PARTIAL | omega_mix.rs (103), rotation_mix.rs `solve_omega_transport` | quick. Implicit tridiagonal omega diffusion; substep control and torques simplified |
| `star/private/solver_support.f90` | 1547 | PARTIAL | solver.rs, solver_norms.rs, hydro_eqns.rs `compute_x_scale` | deep-ish. x-scale and norms present; `Bdomain`, `force_another_iteration`, `edit_lnR` absent |
| `star/private/star_bcyclic.f90` | 981 | PORTED | flexmesa-mtx/src/bcyclic.rs | quick |
| `star/private/star_history_def.f90` | 1390 | ABSENT | none | quick |
| `star/private/star_job_ctrls_io.f90` | 1682 | PARTIAL | inlist_populate.rs `populate_star_job` | deep. 104/419 names read, ~59 used |
| `star/private/star_private_def.f90` | 548 | ABSENT | controls.rs, lib.rs | infrastructure replaced |
| `star/private/star_profile_def.f90` | 1504 | ABSENT | none | quick |
| `star/private/star_solver.f90` | 2043 | PARTIAL | solver.rs (410) | deep (C1-17). Newton loop and gold tolerances present; line search and iteration control are not upstream's |
| `star/private/star_utils.f90` | 4162 | PARTIAL | mesh.rs, hydro_state.rs, evolve_support.rs, core_boundaries.rs | quick. 31/158 routine names |
| `star/private/starspots.f90` | 104 | STUB | tier1_extras.rs `eval_starspot_effective_luminosity` | quick. Exists only in a case extras; not wired to `do_starspots` control |
| `star/private/struct_burn_mix.f90` | 1227 | PARTIAL | struct_burn_mix.rs (558) | deep-ish. `do_struct_burn_mix`, op-split `burn1_zone` loop, convergence call present; `do_mix_omega`, `set_tol_correction` by T regime, `RTI_check_after_converge`, `check_after_converge` absent |
| `star/private/tdc_hydro.f90` | 592 | ABSENT | none | quick. `eq_div_w: StarAdOrder1::ZERO` passed to TDC (hydro_vars.rs:283) |
| `star/private/tdc_hydro_support.f90` | 870 | ABSENT | none | quick. 0/22 names |
| `star/private/timestep.f90` | 2437 | PARTIAL | timestep.rs, timestep_checks.rs, timestep_limits.rs (1267) | deep-ish. Controller/H211 filter and ~25 of 41 limiters present by control name; missing `check_dYe_highT_change`, `check_dlog_eps_nuc_change`, `check_adjust_J_q`, `check_dX_div_X_cntr`, `check_lgL_nuc_cat_change`, Ne/O/Si centre checks |
| `star/private/turb_info.f90` | 615 | PARTIAL | hydro_vars.rs `set_mlt_vars`, `check_for_redo_mlt` | deep (C1-19). Missing `adjust_gradT_fraction`, `adjust_gradT_excess`, `set_gradT_excess_alpha`, `switch_to_radiative/adiabatic` |
| `star/private/turb_support.f90` | 538 | PARTIAL | hydro_vars.rs `eval_face_mlt` (:95-428) | deep (C1-19). Missing `set_superad_reduction`, `check_if_must_fall_back_to_MLT` logic, `max_conv_vel_div_csound`, alt scale height, hard-coded TDC/Henyey params |
| `star/private/winds.f90` | 706 | PORTED | winds.rs (342), winds_driver.rs (320) | deep-ish. 10/14 names; all listed schemes; non-upstream scheme fallback (C1-20) |
| `star/private/write_model.f90` | 278 | PORTED | model_io.rs `write_mod_string` (:25-234) | quick |

Subtotals `star/private` excluding pgstar (110458 LOC, 104 files):

| status | files | upstream LOC | share of non-pgstar LOC | of which not marked "quick" |
| :-- | --: | --: | --: | --: |
| PORTED | 22 | 17950 | 16% | 2320 |
| PARTIAL | 50 | 53140 | 48% | 21536 |
| STUB | 17 | 23259 | 21% | 21177 |
| ABSENT | 15 | 16109 | 15% | 4517 |

pgstar: 23132 LOC ABSENT (section 1). `star/private` total 133590 LOC.

## 4. `star/public`, `star/job`, `star_data`

| upstream path | LOC | status | FlexMESA file(s) | evidence |
| :-- | --: | :-- | :-- | :-- |
| `star/job/other_extras.f90` | 207 | PARTIAL | inlist_runner.rs `StarExtras` trait (:55-118) | deep. 6 extras hooks + 6 `other_*` hooks vs upstream 10 extras procedures and 51 `use_other_*` controls (18 `use_other_*` names occur in the crate) |
| `star/job/run_star.f90` | 48 | PORTED | inlist_runner.rs `run_test_suite_case` | trivial driver |
| `star/job/run_star_support.f90` | 3669 | PARTIAL | inlist_runner.rs (588), star_job_init.rs (596) | deep. Stage loop over `rn` scripts and `&star_job` actions (25% of names). No `LOGS/history.data`/profile/photo files written by the runner (only `.mod`, inlist_runner.rs:127-141), no pgstar, no `do_star_job_controls_after` parity; case-name behaviour (C1-5) |
| `star/public/star_def.f90` | 26 | PORTED | lib.rs | trivial re-export module |
| `star/public/star_lib.f90` | 3332 | ABSENT | lib.rs re-exports; flexmesa-py | quick. 3/247 public routine names; the Rust/Python API is a different surface (no `star_set_*`/`star_get_*`, no hook registration API) |
| `star_data/public/star_data_def.f90` | 161 | PARTIAL | model.rs (584), hydro_state.rs (596), controls*.rs | quick. `StarModel` holds a subset of `star_info` |
| `star_data/public/star_data_lib.f90` | 34 | PORTED | n/a | trivial |
| `star_data/public/star_pgstar.f90` | 265 | ABSENT | none | pgstar data |

Subtotal: PORTED 108, PARTIAL 4037, STUB 0, ABSENT 3597 (total 7742 LOC).

`.inc` files in aggregate: `star_data/public/*.inc` 2825 LOC (`star_info` fields: PARTIAL, `model.rs`/`hydro_state.rs` hold a subset), `star_data/private/star_controls*.inc` + `star_job_controls*.inc` 1831 LOC (PARTIAL, 43% and 25% of names, section 2), `star_data/private/pgstar_controls.inc` 1921 LOC (ABSENT), `star/job/*.inc` 963 LOC (`standard_run_star_extras.inc`, `test_suite_extras.inc`, `gyre_in_mesa_extras_*.inc`, `multi_stars_extras.inc`: ABSENT as ports; the tier extras are not translations of them). Not listed: `star/other/*.f90` (56 hook files, 3176 LOC): 6 hooks have a trait method.

## 5. `binary`

| upstream path | LOC | status | FlexMESA file(s) | evidence |
| :-- | --: | :-- | :-- | :-- |
| `binary/job/run_binary.f90` | 44 | PORTED | e2e.rs | trivial driver |
| `binary/private/binary_ce.f90` | 411 | ABSENT | none | quick. No common-envelope code (grep `common_envelope|ce_`: none) |
| `binary/private/binary_ctrls_io.f90` | 806 | PARTIAL | flexmesa-binary/src/inlist.rs (554) | quick. 111/179 names read |
| `binary/private/binary_disk.f90` | 318 | PORTED | disk.rs (284) | quick. 4/7 names |
| `binary/private/binary_do_one_utils.f90` | 373 | ABSENT | none | quick. 0/6 names (terminal output/log) |
| `binary/private/binary_edot.f90` | 231 | PORTED | jdot_tides.rs, system_jdot.rs | quick. 4/4 names |
| `binary/private/binary_evolve.f90` | 703 | PARTIAL | system_step.rs (593), system.rs, orbit.rs | quick. 2/9 names; orbit update uses midpoint substeps (lead L5) |
| `binary/private/binary_history.f90` | 880 | ABSENT | none | quick. 0/21 names |
| `binary/private/binary_history_specs.f90` | 286 | ABSENT | none | quick |
| `binary/private/binary_irradiation.f90` | 62 | ABSENT | none | quick. grep `irradiation` in flexmesa-binary: none |
| `binary/private/binary_jdot.f90` | 330 | PORTED | jdot_tides.rs (430), system_jdot.rs (545) | quick. 7/7 names |
| `binary/private/binary_job_ctrls_io.f90` | 274 | PARTIAL | inlist.rs | quick. 5/30 names |
| `binary/private/binary_mdot.f90` | 1023 | PORTED | mdot.rs, mdot_implicit.rs, system_rlof.rs, edd_wind.rs | quick. 11/14 names |
| `binary/private/binary_photos.f90` | 324 | ABSENT | none | quick |
| `binary/private/binary_private_def.f90` | 302 | ABSENT | lib.rs | infrastructure replaced |
| `binary/private/binary_roche_deformation.f90` | 416 | PORTED | roche_deformation.rs (395) | quick. 9/11 names |
| `binary/private/binary_tides.f90` | 461 | PARTIAL | jdot_tides.rs, system_jdot.rs | quick. 3/8 names |
| `binary/private/binary_timestep.f90` | 325 | PORTED | timestep.rs (310) | quick |
| `binary/private/binary_utils.f90` | 215 | PORTED | orbit.rs (162) | quick. 6/9 names |
| `binary/private/binary_wind.f90` | 173 | PORTED | edd_wind.rs, system_rlof.rs | quick. 2/3 names |
| `binary/private/pgbinary_ctrls_io.f90` | 2814 | ABSENT | none | plotting |
| `binary/private/pgbinary_full.f90` | 1071 | ABSENT | none | plotting |
| `binary/private/pgbinary_grid.f90` | 576 | ABSENT | none | plotting |
| `binary/private/pgbinary_hist_track.f90` | 827 | ABSENT | none | plotting |
| `binary/private/pgbinary_history_panels.f90` | 915 | ABSENT | none | plotting |
| `binary/private/pgbinary_lib.f90` | 184 | ABSENT | none | plotting |
| `binary/private/pgbinary_orbit.f90` | 300 | ABSENT | none | plotting |
| `binary/private/pgbinary_star.f90` | 627 | ABSENT | none | plotting |
| `binary/private/pgbinary_stub.f90` | 225 | ABSENT | none | plotting |
| `binary/private/pgbinary_summary.f90` | 448 | ABSENT | none | plotting |
| `binary/private/pgbinary_summary_history.f90` | 286 | ABSENT | none | plotting |
| `binary/private/pgbinary_support.f90` | 711 | ABSENT | none | plotting |
| `binary/private/run_binary_support.f90` | 1016 | PARTIAL | system.rs, system_init.rs, e2e.rs | quick. 2/6 names |
| `binary/public/binary_def.f90` | 290 | PARTIAL | system.rs | quick |
| `binary/public/binary_lib.f90` | 272 | PARTIAL | lib.rs | quick. 3/19 names |
| `binary/public/binary_pgbinary.f90` | 166 | ABSENT | none | plotting |

Subtotal: PORTED 3075, PARTIAL 3822, STUB 0, ABSENT 11788 (total 18685 LOC).

`.inc`: `binary/public/binary_controls.inc` 189 + `binary_data.inc` 326 (PARTIAL), `pgbinary_controls.inc` 852 (ABSENT), `binary/private/binary_job_controls.inc` 36 (PARTIAL).

## 6. `astero`

| upstream path | LOC | status | FlexMESA file(s) | evidence |
| :-- | --: | :-- | :-- | :-- |
| `astero/private/adipls_support.f90` | 1116 | ABSENT | flexmesa-star/src/pulse_bvp*.rs, pulse_modes.rs (internal BVP solver) | deep-ish. ADIPLS is not called; frequencies come from `scan_star_bvp_eigenmodes` (flexmesa-astero/src/e2e.rs:186) |
| `astero/private/adipls_support_stub.f90` | 266 | PORTED | n/a | stub upstream |
| `astero/private/astero_run_support.f90` | 1127 | PARTIAL | search_driver.rs, optimize.rs, search_methods.rs, inlist_parser.rs | quick. 4/22 names |
| `astero/private/astero_support.f90` | 1530 | PARTIAL | surface_corr.rs (427), ratios_chi2.rs (445), scaling.rs | quick. 10/43 names |
| `astero/private/extras_support.f90` | 1238 | PARTIAL | sample_eval.rs (384), search_driver.rs | quick. 2/28 names |
| `astero/private/gyre_support.f90` | 308 | ABSENT | internal BVP solver as above | deep-ish. GYRE is not called |
| `astero/private/gyre_support_stub.f90` | 49 | PORTED | n/a | stub upstream |
| `astero/private/pgstar_astero_plots.f90` | 645 | ABSENT | none | plotting |
| `astero/private/pgstar_astero_plots_stub.f90` | 45 | PORTED | n/a | stub upstream |
| `astero/public/astero_def.f90` | 1645 | PARTIAL | inlist_astero.rs (427) | quick. 1/36 names |
| `astero/public/astero_lib.f90` | 146 | PARTIAL | lib.rs | quick. 2/7 names |

Subtotal: PORTED 360, PARTIAL 5686, STUB 0, ABSENT 2069 (total 8115 LOC).

## 7. Other upstream modules, at module level (all "quick": file lists, sizes and keyword presence; contents not compared)

Upstream LOC = `private/*.f90|*.f|*.dek` + `public/*.f90`. Rust LOC = all `src/**/*.rs` of the crate including tests.
A keyword hit shows that a feature has code, not that it is correct or reachable from `flexmesa-star`.

| upstream module | upstream LOC (private + public) | FlexMESA crate (LOC) | present (by file/keyword) | absent or flagged |
| :-- | --: | :-- | :-- | :-- |
| `eos` | 15990 + 1552 | `flexmesa-eos` (13152) | HELM (`helm/`), OPAL/SCVH and FreeEOS tables (`table_set.rs`, `blend_alfa.rs`), PC (`pc/`: `blin.rs`, `coulomb_ion.rs`, `elect.rs`), Skye (`skye/`), CMS (`cms/`), ideal (`ideal.rs`), gamma-law DE (`gamma_de.rs`), `eosPT` (`eos_pt.rs`), `ion_offset` | `eos_ctrls_io.f90` (659): only 5 of 102 `&eos` names are read from an inlist (`use_FreeEOS`, `use_OPAL_SCVH`, `use_PC`, `use_Skye`, `use_CMS`; `flexmesa-star/src/inlist_populate.rs:259-267`), so none of the blend-region limits (`logRho_min_OPAL_SCVH_limit`, `mass_fraction_limit_for_PC`, `Skye_min_gamma_for_solid`, `Z_all_HELM`, ...) can be set; no `other_eos` hooks; `create_EXCOR7/FSCRliq8_table.f90` (558) not needed. `pre_ms.rs:384` builds with `EosConfig::default()` regardless of the inlist |
| `kap` | 11364 + 1703 | `flexmesa-kap` (4194) | fixed-metal tables (`fixed.rs`), Type2 C/O (`co.rs`, `co_find.rs`), low-T incl. Freedman/AESOPUS/kapCN by keyword (`low_t.rs`, `loader*.rs`), conduction (`conduction.rs`), Compton (`compton.rs`), blends (`blend.rs`) | OP monochromatic and radiative acceleration: `op_common.f`, `op_def.f90`, `op_ev.f90`, `op_eval.f90`, `op_eval_mombarg.f90`, `op_load.f`, `op_load_master.f90`, `op_osc.f`, `op_radacc.f90` (4324 LOC) ABSENT (no `op_mono`/`mombarg`/`radacc` hits), so `1.4M_ms_op_mono` and `radiative_levitation` cannot be doing what they name. `kap_ctrls_io.f90`: 5 of 36 `&kap` names read (`Zbase`, three prefixes, `use_Type2_opacities`); no `kap_blend_*`/`use_Zbase_for_Type1`/`cubic_interpolation_in_X/Z` controls; `kap_aesopus.f90` (608) and `kapcn.f90` (441) only as loader variants (not checked) |
| `net` | 11582 + 1835 | `flexmesa-net` (10057) | `approx21` (7 files), general net derivs (`derivs*.rs`), `.net` parser (`parser.rs`, `initialize.rs`), const-density and const-P burners, screening (`screen*.rs`), Rosenbrock tableau | `net_initialize.f90` 2302 -> `initialize.rs` 602; which upstream `.net` files load was not tested (lead); `net_eval.f90` rate-factor/`other_net` hooks not found in star |
| `rates` | 14948 + 2269 | `flexmesa-rates` (3102) | REACLIB evaluation (`loader.rs`, `tfactors.rs`), weak/ecapture tables (`weak.rs`, `ecapture.rs`), screening extended + Chugunov (`screen*.rs`), pycno (keyword) | largest gap among the microphysics modules (18% by LOC): `ratelib.f90` has 183 `rate_*` routines, `ratelib.rs` has 29 functions; `raw_rates.f90` 1140 -> 217; `rates_initialize.f90` (1422), `rates_names.f90` (368), `rates_reverses.f90` (173), `rates_support.f90` (706, rate-table cache) and `suzuki_tables.f90` (553) have no same-size counterpart; `special_rate_factor`/rate-override controls absent |
| `neu` | 1716 + 117 | `flexmesa-neu` (1914) | pair+plasma, photo, brems, recombination (`pair_plas.rs`, `phot.rs`, `brem.rs`, `reco.rs`) | none flagged |
| `atm` | 3567 + 239 | `flexmesa-atm` (2091) | `T_tau` uniform and varying, 4 relations (Eddington, solar_Hopf, Krishna_Swamy, Trampedach_solar), tables incl. `WD_tau_25`/`DB_WD_tau_25`, irradiated grey, Paczynski (keyword) | in `flexmesa-star` only uniform `T_tau` with `max_iters = 0` and tables are called (`hydro_brunt_surf.rs:374-389`, `:216-258`); `irradiated_grey`, `Paczynski_grey`, `fixed_*`, varying/iterated opacity are unreachable from an inlist (`inlist_populate.rs:392-397`) |
| `turb` | 1791 + 306 | `flexmesa-turb` (2238) | MLT variants Cox, Henyey, ML1, ML2, Mihalas (`mlt.rs`); TDC (`tdc.rs`, `tdc_support.rs`, `tdc_search.rs`); semiconvection Langer_85 (`semiconvection.rs`); thermohaline Kippenhahn, Traxler, Brown_Garaud (`thermohaline.rs`) | module looks complete by file list. Star-side use is partial: default option is Cox not TDC (C1-14), TDC only where `vc_old > 0`, TDC alphas and Henyey parameters hard-coded, no `tdc_hydro` `Eq/Uq` terms (C1-19, section 3) |
| `chem` | 451 + 3371 | `flexmesa-chem` (2364) | nuclide catalog, composition info, solar scales (AG89 and others) | `chem_isos_io`, `lodders_mod` not identified; not examined |
| `ionization` | 1642 + 264 | `flexmesa-ionization` (2837) | ion tables load/eval, typical charge, Fe56-in-He4 data, potentials | `ion_table_plot.f90` n/a |
| `colors` | 3137 + 640 | `flexmesa-colors` (2406) | SED cube, filters, photometry, interpolation | `colors_ctrls_io`, `colors_history` (history-column hookup) not identified |
| `num` | 10987 + 1461 | `flexmesa-num` (4320) | DOPRI5, DOP853 (`rk.rs`), Rosenbrock family ros2/rose2/ros3p/ros3pl/rodas3/rodas4/rodasp (`rosenbrock*.rs`), Newton (`newton.rs`), Brent, safe_root/brackets (`root.rs`), simplex, NEWUOA, BOBYQA, quadrature | size ratios suggest simplification: `mod_bobyqa.f` 2243 -> `bobyqa.rs` 422 (it reuses `newuoa::fit_min_frobenius_quad`), `mod_newuoa.f` 1583 -> 428, `mod_newton.f90` 1536 -> 270; no implicit Radau/SEULEX-type solvers (`estrad.dek`, `slvrad.dek`, `slvseu.dek`: no keyword hits); `mod_solout_root`, `mod_random`, `mod_qsort` partly |
| `mtx` | 3544 + 378 | `flexmesa-mtx` (3943) | BCYCLIC, banded LU, block tridiagonal, dense LAPACK subset, plus sparse and eigen routines not in upstream `mtx` | `pre_conditioners.f90` absent; `DGBSVX` wrapper not identified |
| `interp_1d` | 3543 + 1426 | `flexmesa-interp1d` (1194) | piecewise monotonic (`pm.rs`), monotonicity-preserving (`mp.rs`), eval, remesh | `_sg` variants n/a; `_autodiff` variants (859 LOC) and most of `interp_1d_misc.f90` (712) not identified |
| `interp_2d` | 12593 + 972 | `flexmesa-interp2d` (2682) | bicubic, bi-PM, Akima 760, Renka 790 | `_sg` halves n/a |
| `auto_diff` | 14316 + 29 (generated) | `flexmesa-autodiff` (2746) | const-generic order 1 (N vars), 2-var order 2 and order 3 | `auto_diff_real_tdc_module` not identified in this crate (TDC partials are handled in `flexmesa-turb`) |
| `math` | 250 + 632 | `flexmesa-math` (468) | safe log/exp/pow wrappers over Rust `std` | upstream links CRMATH (crlibm) for bit-reproducible transcendental functions; Rust `std` `ln/exp/powf` are platform libm, so 16-digit agreement with Fortran MESA is not guaranteed by construction (lead) |
| `const` | 254 | `flexmesa-const` (359) | constants | not compared value by value here |
| `utils` | 2147 + 1397 | `flexmesa-utils` (407), `flexmesa-schema` (779), `flexmesa-data` (1745) | namelist parser, data loaders | dict/idict/NaN utilities n/a |
| `gyre` interface | 171 (`gyre/public`) | none | n/a | ABSENT. No GYRE library call; an internal adiabatic BVP solver stands in (`flexmesa-star/src/pulse_bvp.rs` 480, `pulse_bvp_recon.rs` 590, `pulse_modes.rs` 586) for `gyre_in_mesa_*`, `astero_gyre` and all astero frequency evaluations |
| `adipls` interface | 51 | none | n/a | ABSENT, same substitution (`flexmesa-astero/src/e2e.rs:174-186`, `OuterBoundKind::AdiplsIsothermal`) |

## 8. Totals

Scope: every `.f90` under `star/private`, `star/public`, `star/job`, `star_data/public`, `binary/private|public|job`,
`astero/private|public` = 199 files, 168132 LOC (`.inc` files, 9101 LOC, excluded from the percentages and described in
aggregate above).

| status | upstream LOC | share of all 168132 | share excluding plotting (134940) |
| :-- | --: | --: | --: |
| PORTED | 21493 | 13% | 16% |
| PARTIAL | 66685 | 40% | 49% |
| STUB | 23259 | 14% | 17% |
| ABSENT | 56695 | 34% | 23503 = 17% |

Plotting = `pgstar*` (23132), `pgbinary*` (8984), `binary_pgbinary` (166), `star_pgstar` (265), `pgstar_astero_plots` (645) = 33192 LOC, all ABSENT.
Of the non-plotting ABSENT lines about 4100 are infrastructure that Rust replaces by construction (`alloc.f90` 3248,
`star_private_def.f90` 548, `binary_private_def.f90` 302) and 3332 are the public `star_lib` API.

By area (non-plotting): `star/private` 110458 LOC: PORTED 16%, PARTIAL 48%, STUB 21%, ABSENT 15%. `binary` (non-plotting
8535 LOC): PORTED 36%, PARTIAL 45%, ABSENT 19%. `astero` (7470 non-plotting): PORTED 5% (upstream stubs), PARTIAL 76%,
ABSENT 19% (the ADIPLS and GYRE couplings).

Overall fraction ported, by LOC:

- Strict (PORTED only): 13% of everything, 16% excluding plotting.
- Weighted estimate (PORTED x 0.9, PARTIAL x 0.5, STUB x 0.1): about 55k of 135k non-plotting lines = 41%; about 33% of
  everything. My uncertainty on the weighted figure is roughly +/- 10 percentage points (30% to 50% excluding
  plotting): 71 of the 104 non-pgstar `star/private` rows are "quick" classifications, the PARTIAL weight is a guess, and
  the PORTED rows for the RSP files (10371 LOC, half of all PORTED lines) and the hydro equation files rest on routine-name
  matches and size, not on reading the equations.
- A size cross-check points the same way: the three Rust crates hold about 58k lines of non-test source for 135k lines
  of non-plotting Fortran, and about 4k of those Rust lines are case extras, staging and assertions with no upstream
  counterpart in the library.

What the status mix means in practice: the pieces that are close to upstream are the equation residual kernels
(momentum, energy, chem, RTI, Riemann), the RSP module, ZAMS loading, the pre-MS builder, winds, `eps_grav`/`eps_mdot`
and the binary orbital physics (jdot, mdot, Roche geometry). The pieces that are placeholders or missing are the ones
that decide which model is evolved and how it is controlled: initial-model construction and every `relax_*` operation,
the mesh planner, the Newton iteration control, element diffusion, convective premixing, predictive mixing, phase
separation, rotation as solver variables, history/profile output, 57% of `&controls` and 75% of `&star_job`.

## Matches (coverage-level)

- Routine-name coverage is high for `rsp_step.f90` (36/53), `rsp_eval_eos_and_kap.f90` (15/16), `rsp.f90` (14/19), `binary_jdot.f90` (7/7), `binary_edot.f90` (4/4), `binary_mdot.f90` (11/14), `binary_roche_deformation.f90` (9/11), `winds.f90` (10/14), `hydro_energy.f90` (11/17).
- 441 of the 549 same-named scalar defaults compared agree with upstream `controls.defaults`/`star_job.defaults` (section 2).
- All seven upstream atmosphere tables are recognised by name (`crates/flexmesa-star/src/inlist_populate.rs:445-456`), all five MLT variants plus TDC are parsed (`:378-390`), and all three thermohaline options exist in `flexmesa-turb`.

## Leads (not verified)

- Which `.net` files `flexmesa-net` can actually load and evolve (only `basic.net` and `approx21.net` are named in the crate's own docs; `mesa_80.net`/`cno_extras.net` are referenced by tier extras).
- Whether `flexmesa-kap` AESOPUS/kapCN loaders are reachable (`KapLowTOption::from_prefix`) and tested.
- Whether BOBYQA/NEWUOA in `flexmesa-num` are Powell's algorithms or simplified trust-region schemes sharing one quadratic fit.
- `flexmesa-math` vs CRMATH bit-level differences.
- Python/JAX side (`src/flexmesa/star/*.py`, 10 files) was not examined at all; this map covers the Rust crates only.
