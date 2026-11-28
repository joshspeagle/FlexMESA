# 1D Stellar Structure & Evolution (`flexmesa.star`)

!!! warning "Frozen & Unverified JAX Star Helpers (`#166`)"
    The JAX `star` modules (`jax_kernels`, `jax_winds_mix`, `jax_evolve`,
    `jax_diffusion_rot`, `jax_hydro_pulse`, `jax_rsp`) are **frozen and
    unverified** (`JAX_ENGINE_STATUS = "frozen-unverified-until-#143"`) until
    the canonical Safe Rust engine passes the upstream MESA `26.4.1` history
    harness (`#143`). No JAX stellar evolution step exists (`StarModel` wraps
    the Rust solver), and these helpers must not be modified before `#143`
    passes.

The `flexmesa.star` subpackage implements the 1D Lagrangian stellar structure and evolution engine, bridging the Safe Rust core (`crates/flexmesa-star`) with frozen (`float64`) JAX structure, wind, mixing, hydrodynamic, and pulsation formula kernels. It provides:

- **Model Initialization & I/O**: Zero-Age Main Sequence (`StarModel.from_zams`), the MESA 26.4.1 pre-main-sequence builder (`StarModel.from_pre_ms`, `build_pre_ms_model`), upstream ASCII `.mod` readers/writers (`from_mod_file`, `from_mod_str`, `write_mod_file`, `to_mod_str`), generated little-endian `serde` binary `.photo` checkpoints (`FMPHOTO2` v30 via `to_photo_bytes`, `try_write_photo_bytes`, `from_photo_bytes`, `load_photo_bytes`, `verify_photo_restart`, saving primary/start/old arrays plus `controls`, `job`, `rsp`, energy-accounting scalars, and cumulative solver counters while skipping ephemeral `MicrophysicsCache` hit/miss counters so restarts remain bit-exact against uninterrupted runs; `FlexMESA` `.photo` bytes remain internal and are not binary-compatible with Fortran MESA `photos/`), and MESA `26.4.1` `LOGS/history.data`, `LOGS/profileN.data` (`StarModel.to_profile_str`), and `LOGS/profiles.index` column evaluators (including `phase_of_evolution` and upstream energy-conservation columns), incremental writers (`StarLogsWriter`, `IncrementalHistoryWriter`, `run_inlist_stage_with_logs`, `run_test_suite_case_with_logs`), and lossless ASCII readers (`read_mesa_log_file`, `read_profiles_index_file`), with `HistoryRow.log_lh` reporting hydrogen-burning luminosity (`pp + cno`) separately from helium, metal, and net nuclear burning (`#159`, `#163`). `&controls` and `&star_job` defaults match MESA `26.4.1` (`#149`, `#151`, `#152`, `#155`, `#156`, `#159`, `#161`, `#299`), using upstream scalar sentinels (`1e36`, `11.0`, `12.0`, `99.0`, `3e7`, `0.99`, `0.0`, `-1`, `-111`), explicit `set_*` / `set_initial_*` rotation booleans, and explicit `&kap::zbase` (`-1` default without silent cross-namelist copying).
- **Coupled Henyey Solver & Time Evolution**: Staggered 1D cell/face grids, 24-slot 3-cell stencil automatic differentiation (`StarAdOrder1`), coupled block-tridiagonal Newton-Raphson relaxation (`solve_henyey`, `step_struct_burn_mix`, `evolve`) porting MESA `26.4.1`'s `do_solver_work`, `adjust_correction`, `Bdomain`, `scale_correction_norm`, retry-dependent `max_tries` schedule, `solver_itermin`, high-$T$ correction tolerance relaxation, `check_after_converge`, and `sizequ`/`sizeB` norms (`#152`, `#177`, `#251`, `#252`, `#260`), structured `StarError::{StepFailure, Terminated}` diagnostics (`StepFailureInfo` carrying zone, equation, paired variable, value, norms, and retry cause, with zero-retry fast-fail on `InvalidInput`/`IoError`/`Terminated`), in-place `PrimaryStateBuffer` rollback (`restore_mesh_on_retry`), 2-slot bit-exact `MicrophysicsCache` reuse (`micro_cache_hits`, `micro_cache_misses`), retry-proof solver/redo counters (`total_num_solver_calls`, `num_redos`, `num_retries`), optional JSONL step telemetry (`FLEXMESA_STEP_TRACE` / `FLEXMESA_STEP_TRACE_FILE`), MESA `26.4.1` end-of-step total-energy conservation accounting (`eval_total_energy_integrals` with `Quad` PE/KE, rotational, `TDC`, `RSP2`, and `RSP` terms, plus Phase 1 and Phase 2 `update_step_energy_conservation` for both `dedt` and `eps_grav`; `#151`), MESA `26.4.1` adaptive mesh planning, burning/composition gradient multipliers, conservative energy/momentum/angular-momentum remapping, and `split_merge_amr` (`do_adjust_mesh`, `do_mesh_plan`, `set_mesh_function_data`, `get_gval_info`, `remap_onto_dq`, `remesh_split_merge`, `do_prune_mesh_surface`; `#153`), MESA `26.4.1` 3-region `do_adjust_mass_step` (`compute_delta_m`, `get_xa_for_accretion`, `set_omega_adjust_mass`, `adjust_j_lost`, `smooth_outer_xa`, and `eps_mdot` mass-change heating; `#159`), `select_mdot_action` rotational-wind redo/retry control (`#139`, `#152`), MESA `26.4.1` `do_check_limits` with `when_to_stop_{rtol,atol}` target-hitting redos (`redo_limit`), `set_phase_of_evolution` across all 13 `stop_at_phase_*` phases, exact `time_sec` `max_age` / graded `num_adjusted_dt_steps_before_max_age` approach (`#159`), sub-cell `CoreMasses` and `PowerCategories` reporting, and Söderlind-Wang H211 PI timestep control (`timestep_controller`, `eval_h211_timestep_jax`) with full `timestep.f90` soft/hard limiters and `min_timestep_limit` / `max_number_retries` / `redo_limit` termination (`#151`, `#152`, `#177`, `#253`).
- **Advanced Stellar Physics** *(partial/unverified against MESA `26.4.1`; `#154`, `#157`-`#158`)*: 16-set exponential and step convective overshooting (`overshoot.rs`) with `'none'` short-circuiting, 3-cell log-$H_P$ boundary interpolation, `cospi` mass ramp, and `unsmoothed_brunt_B` cutoff, verbatim predictive mixing (`predictive_mix.rs`) and convective premixing (`conv_premix.rs`), threshold-gated binomial `brunt_B` and `gradL_composition_term` smoothing (`#155`), Reimers/Blocker/Vink/Björklund/Nugis-Lamers/Dutch/`'other'` stellar winds (`do_set_mdot`, `do_set_mdot_with_extras`), super-Eddington wind (`eval_super_eddington_wind` with `0.5 * vesc2` and start-of-step `prev_Ledd`), single-star RLO wind (`eval_rlo_wind`), central-temperature/envelope/timestep wind gates, and surface-averaged `v_div_v_crit` rotational enhancement shared with `flexmesa-binary` (`#159`), Burgers multi-fluid atomic diffusion (`do_element_diffusion_step`, `do_solve_diffusion`, `get_matrix_coeffs`, `paquette_coefficients`, supporting Paquette et al. 1986 splines, Stanton & Murillo 2016, Caplan, Bauer, & Freeman 2022 OCP blending, Thoul/Hu and CGS Burgers solvers, backward-Euler `BE_solve` and block-tridiagonal Rosenbrock `solve_with_isolve`, and `eps_diffusion` / `eps_WD_sedimentation` heating; kernel-verified against MESA `26.4.1` Fortran oracles in `#156`, while `op_mono` radiative levitation fails loudly pending C10), white-dwarf crystallization phase separation (`do_phase_separation_step` with Blouin et al. 2021a,b `'CO'` and `'ONe'` shifts, EOS `phase > 0.9` front, constant-`Pgas` Skye updates, `mix_outward`, and `eps_phase_separation`; kernel-verified against MESA `26.4.1` in `#156`), shellular Roche rotation, angular momentum mixing (`solve_omega_mix`), Spruit-Tayler magnetic dynamos (`#158`), coupled 3-cell `StarAdOrder1` spherical HLLC Riemann `u_flag` hydrodynamics (`riemann_ad`) and Duffell (2016) Rayleigh-Taylor `alpha_rti` shock mixing (`hydro_rti`), `RSP2` radial pulsation equations (`#153`), and `GYRE`/`FGONG`/`OSC` structure exporters (`to_gyre_str`, `to_fgong_str`, `to_osc_str`, plus `flexmesa_star::pulse_fgong::get_fgong_data` with `'T_tau'` `build_atm` support for `flexmesa-adipls`).
- **Inlist & `test_suite` Runners**: Recursive Fortran `inlist` bundle parsing (`parse_inlist_bundle`; full `&controls`/`&star_job` wiring tracked in `#147`), `rn` stage extraction (`parse_rn_stages`), internal linear oscillation stub and `RSP` `LINA`/envelope builders (`eval_linear_oscillation_modes`, `eval_target_mode_frequency_and_growth`, and `eval_astero_scaling_dnu_numax_uhz` pending `GYRE`/`ADIPLS` in `#170`/`#171`, `eval_rsp_lina_modes`, `build_rsp_envelope_model`), and the `star/test_suite` runner (`run_test_suite_case`, accepting optional caller-supplied test-side `extras` callbacks while canonical MESA `26.4.1` trajectory comparison lives in `flexmesa.testing.history_compare`).

For governing equations and numerical architecture, see [1D Stellar Structure, Henyey & Hydrodynamics](../concepts/stellar-structure-and-hydro.md) and [Convection, Diffusion, Rotation & Dynamos](../concepts/convection-mixing-rotation.md).

---

## Quick Usage Example

```python
import flexmesa.star as star

# Load a 1.0 Msun, Z = 0.02 Zero-Age Main Sequence model
model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
model.eval_hydro_vars()
print(f"nz={model.nz}, Teff={model.teff:.1f} K, log10(T_c)={model.ln_t[-1] / 2.302585093:.4f}")
```

---

## Package Overview (`flexmesa.star`)

::: flexmesa.star

---

## Stellar Model Driver (`flexmesa.star.model`)

::: flexmesa.star.model

---

## Structure Reports & State Containers (`flexmesa.star._arrays`)

::: flexmesa.star._arrays

---

## Differentiable Structure & Henyey Kernels (`flexmesa.star.jax_kernels`)

::: flexmesa.star.jax_kernels

---

## Winds & Convective Mixing Kernels (`flexmesa.star.jax_winds_mix`)

::: flexmesa.star.jax_winds_mix

---

## Mesh & Timestep Control Kernels (`flexmesa.star.jax_evolve`)

::: flexmesa.star.jax_evolve

---

## Diffusion, Phase Separation & Rotation Kernels (`flexmesa.star.jax_diffusion_rot`)

::: flexmesa.star.jax_diffusion_rot

---

## HLLC Hydrodynamics & Pulsation Kernels (`flexmesa.star.jax_hydro_pulse`)

::: flexmesa.star.jax_hydro_pulse

---

## `RSP` & `RSP2` Turbulent Pulsation Kernels (`flexmesa.star.jax_rsp`)

::: flexmesa.star.jax_rsp

---

## Inlist Bundle & `test_suite` Runner (`flexmesa.star.e2e`)

::: flexmesa.star.e2e

---

## Compiled Rust Extension Bridge (`flexmesa._rust`)

Low-level Safe Rust PyO3 bindings exposed via `flexmesa._rust` for 1D stellar structure, evolution, extras hooks, pulsation exports, and test-suite drivers (`_ArrF64` denotes a `float64` NumPy `ndarray`).

### `star_py`, `star_lib_py`, and `extras_py`

- **`RustStarModel`** (`class RustStarModel`; static constructors `from_zams`, `from_pre_ms`, `from_mod_str`, `from_photo_bytes`; core methods `to_mod_str`, `to_photo_bytes`, `eval_hydro_vars`, `set_evolve_controls`, `save_start_values`, `save_old_values`, `record_start_of_step_energy_integrals`, `eval_equations`, `solve_henyey`, `update_mixing`, `eval_dxdt_mix`, `solve_diffusion`, `run_omega_mix`, `step_struct_burn_mix`, `apply_wind_and_adjust_mass`, `adjust_mesh`, `pick_next_timestep`, `evolve_step`, `evolve`, `history_tuple`, `to_history_line`, `to_profile_str`, `copy`, `set_primary_state`): Python wrapper around `flexmesa_star::StarModel` exposing primary state, zone/face profiles, controls, and solver telemetry.
- **`star_interp_q`** (`(dq1: _ArrF64, v1: _ArrF64, dq2: _ArrF64) -> _ArrF64`): Interpolates profile `v1` from Lagrangian grid `dq1` onto `dq2` (`interp_q`).
- **`star_yrs_for_init_timestep`** (`(m_msun: float) -> float`): Computes the initial ZAMS timestep in years (`yrs_for_init_timestep`).
- **`StarLibModel`** (`class StarLibModel`; static constructor `new_for_build`; properties `nz`, `species`, `model_number`; methods `get`, `set`, `get_scalar`, `set_scalar`, `xa`, `get_atm_options`, `set_atm_options`, `normalize_dqs`, `set_qs`, `set_m_and_dm`, `set_dm_bar`, `change_to_xa_for_accretion`, `basic_composition_info`, `get_atm_pt`, `get_surf_pt`, `solve_eos_given_pgas_t`, `solve_eos_given_pgas_t_auto`, `do_eos_for_cell`, `do_kap_for_cell`, `set_mlt_vars`, `d_gradt_dlnt00`, `pack_xh`, `write_model`): Model handle exposing the `star_lib` subset for custom build hooks (`#453`).
- **`atm_teff`** (`(l: float, r: float) -> float`): Evaluates black-body effective temperature in $\mathrm{K}$ for $L$ in $\mathrm{erg\,s^{-1}}$ and $R$ in $\mathrm{cm}$ (`atm_Teff` in `atm/public/atm_lib.f90`).
- **`crmath_log`** (`(x: float) -> float`): Evaluates correctly rounded natural logarithm (`log` in `math_lib`).
- **`PyRspBuildContext`** (`class PyRspBuildContext`; attributes `case_dir`, `output_dir`, `work_dir`, `rsp_mass`, `rsp_x`, `rsp_z`, `rsp_alfam`, `rsp_alfap`, `rsp_alfat`, `rsp_gammar`, `x_ctrl`, `x_integer_ctrl`, `x_logical_ctrl`, `x_character_ctrl`; method `eval_lina`): Context passed into Python `other_rsp_build_model` hooks for static `RSP` envelope and `LINA` instability scans (`#148`).
- **`star_apply_extras_updates`** (`(model: RustStarModel, updates: dict[str, Any]) -> None`): Applies a MESA star extras update dictionary onto a `RustStarModel` in place (`#148`, `#231`), raising `ValueError` on unrecognized keys or invalid values.
- **`star_extras_snapshot`** (`(model: RustStarModel, case_dir: str = "", include_history: bool = True) -> dict[str, Any]`): Builds the MESA `star_info` extras snapshot dictionary for a `RustStarModel` without running a full trace (`#148`, `#231`).
- **`star_solve_gyre_mesa_modes_full`** (`(gyre_content: str, l: int = 0, omega_min: float = 0.5, omega_max: float = 12.0, n_freq: int = 400) -> list[tuple[int, int, int, int, float, list[float], list[float]]]`): Solves adiabatic `GYRE` MESA modes and returns `(l, n_p, n_g, n_pg, freq_uhz, x_grid, xi_r)` per eigenmode.

### `star_e2e_py`

- **`binary_run_test_suite_case`** (`(case_dir: str, max_steps_cap: int | None = None, allow_unsupported_controls: bool = False, extras: Any = None) -> dict[str, Any]`): Runs a coupled `binary/test_suite/<case>` directory and returns a report dictionary.
- **`star_build_rsp_envelope_model`** (`(m_msun: float, l_lsun: float, teff_k: float, x_h: float = 0.73, z_metal: float = 0.007) -> Any`): Builds a static `RSP` envelope `RustStarModel` with $(M/M_\odot, L/L_\odot, T_{\mathrm{eff}}, X, Z)$.
- **`star_copy_cleaned_case`** (`(src_dir: str, dst_dir: str, suite: str = "star") -> None`): Copies a `star`, `binary`, or `astero` case directory from `src_dir` to `dst_dir` while stripping unsupported or unused namelist keys.
- **`star_eval_astero_scaling_dnu_numax_uhz`** (`(model: Any) -> tuple[float, float]`): Evaluates global asteroseismic scaling parameters $(\Delta\nu, \nu_{\max})$ in $\mu\mathrm{Hz}$ on `model`.
- **`star_eval_linear_oscillation_modes`** (`(model: Any, l: int = 1, n_orders: list[int] = ...) -> list[dict[str, Any]]`): Evaluates adiabatic/non-adiabatic linear oscillation modes at degree `l` across radial orders `n_orders` on `model`.
- **`star_eval_rsp_lina_modes`** (`(m_msun: float, l_lsun: float, teff_k: float, x_h: float = 0.73, z_metal: float = 0.007, n_modes: int = 3) -> list[dict[str, Any]]`): Evaluates linear non-adiabatic (`LINA`) radial pulsation periods ($\mathrm{d}$) and growth rates for an `RSP` envelope with $(M/M_\odot, L/L_\odot, T_{\mathrm{eff}}, X, Z)$.
- **`star_eval_target_mode_frequency_and_growth`** (`(model: Any) -> tuple[dict[str, Any], bool]`): Evaluates the target oscillation mode requested by `x_integer_ctrl(3..4)` and `x_ctrl(1..2)` on `model`, returning `(mode_dict, matched_target)`.
- **`star_parse_inlist_bundle`** (`(inlist_path: str, allow_unsupported_controls: bool = False) -> dict[str, Any]`): Parses a MESA `inlist` file (recursively resolving `read_extra_*_inlist` includes) and returns a summary dictionary of parsed controls.
- **`star_parse_rn_stages`** (`(case_dir: str, skip_optional: bool = True) -> list[tuple[str, str, str]]`): Parses an upstream MESA `./rn` script in `case_dir` into `(stage_kind, arg1, arg2)` tuples.
- **`star_run_astero_test_suite_case`** (`(case_dir: str, skip_optional: bool = True, max_steps_per_stage: int | None = None, verify_restart: bool = True, allow_unsupported_controls: bool = False, extras: Any = None) -> dict[str, Any]`): Runs an `astero/test_suite/<case>` directory and returns a report dictionary.
- **`star_run_test_suite_case`** (`(case_dir: str, skip_optional: bool = True, max_steps_per_stage: int | None = None, verify_restart: bool = False, allow_unsupported_controls: bool = False, extras: Any = None) -> dict[str, Any]`): Runs a multi-stage upstream MESA `test_suite` case directory and returns a structured telemetry dictionary.
- **`star_verify_photo_restart`** (`(model: Any, n_steps: int = 1) -> tuple[int, int]`): Runs a `.photo` restart bit-identity check on `model` over `n_steps`.

### `star_wave5_py`

- **`star_add_predictive_mixing`** (`(model: RustStarModel, ingested_mass_limit_coeff: float = 0.1, superad_thresh: float = 0.005) -> int`): Applies predictive mixing at convective boundaries on `model`.
- **`star_do_conv_premix`** (`(model: RustStarModel, max_iters: int = 10, avoid_fuel_increase: bool = True) -> int`): Runs iterative convective premixing on `model` and returns the number of cells absorbed across convective boundaries.
- **`star_do_element_diffusion`** (`(model: RustStarModel, dt: float, scheme: str = "stanton_murillo") -> tuple[_ArrF64, _ArrF64]`): Runs multi-species element diffusion across all zones of `model` over `dt` seconds, returning `(v_gt_2d, eps_wd_sed_1d)`.
- **`star_do_phase_separation`** (`(model: RustStarModel, dt: float, gamma_crystallization: float = 175.0, is_one_mixture: bool = False) -> _ArrF64`): Executes white-dwarf crystallization phase separation on `model` and returns the gravothermal heating rate array $\epsilon_{\mathrm{phase\_sep},k}$.
- **`star_eval_burgers_velocities`** (`(a: _ArrF64, z: _ArrF64, x_mass: _ArrF64, rho: float, t: float, dlnt_dr: float, g_grav: float, g_rad: _ArrF64 | None = None, scheme: str = "stanton_murillo") -> tuple[_ArrF64, _ArrF64, float]`): Solves the single-zone multi-species Burgers resistance equations and returns `(v_gt, d_self, ee_force)`.
- **`star_eval_caplan_ocp_dstar`** (`(gamma_i: float, kappa: float) -> float`): Evaluates the Caplan, Bauer & Freeman (2022) strong-coupling OCP dimensionless self-diffusion coefficient $D^*(\Gamma, \kappa)$.
- **`star_eval_omega_crit`** (`(cgrav: float, m_grav: float, r_polar: float, lum: float, opacity: float, max_gamma_edd: float = 0.99) -> float`): Evaluates local critical Roche angular velocity $\Omega_{\mathrm{crit}}$ ($\mathrm{rad\,s^{-1}}$).
- **`star_eval_phase_separation_shift`** (`(x_heavy: float, is_one_mixture: bool = False) -> float`): Evaluates the Blouin et al. (2020, 2021) solid-liquid distillation shift $\Delta x_{\mathrm{heavy}}$.
- **`star_eval_radiative_levitation_accel`** (`(kappa_species: float, lum_face: float, r_face: float) -> float`): Evaluates radiative levitation acceleration $g_{\mathrm{rad},i}$ ($\mathrm{cm\,s^{-2}}$).
- **`star_eval_rotation_fp_ft`** (`(w_div_w_crit: float, fp_min: float = 0.75, ft_min: float = 0.95) -> tuple[float, float, float]`): Evaluates Fabry et al. (2022) / Paxton et al. (2019) Roche rotational distortion polynomials $(f_P, f_T, i_{\mathrm{rot}}/r^2)$.
- **`star_eval_rotational_mixing_zone`** (`(r: float, hp: float, grav: float, rho: float, t: float, cp: float, opacity: float, chi_t: float, chi_rho: float, grada: float, gradt: float, brunt_n2_t: float, brunt_n2_mu: float, omega: float, dln_omega_dln_r: float, abar: float, zbar: float, d_visc: float = 0.0) -> tuple[float, float, float, float, float, float, float, float, float]`): Evaluates all five rotational instabilities and the Spruit-Tayler dynamo at a single spherical shell face, returning `(d_dsi, d_sh, d_ssi, d_es, d_gsf, d_st, nu_st, b_phi, b_r)`.
- **`star_eval_wd_sedimentation_heating`** (`(a: _ArrF64, z: _ArrF64, x_mass: _ArrF64, v_gt: _ArrF64, abar: float, zbar: float, g_grav: float) -> float`): Evaluates white-dwarf gravitational sedimentation heating rate $\epsilon_{\mathrm{WD,sed}}$ ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
- **`star_set_rotation_mixing_info`** (`(model: RustStarModel) -> _ArrF64`): Runs `set_rotation_mixing_info` (`star/private/rotation_mix_info.f90`) on `model`, returning a `(nz, 9)` array `(D_DSI, D_SH, D_SSI, D_ES, D_GSF, D_ST, nu_ST, dynamo_B_phi, dynamo_B_r)` with diffusivities in $\mathrm{cm^2\,s^{-1}}$ and magnetic fields in $\mathrm{G}$, raising `ValueError` on explicit-error paths (`#503`).

### `star_wave5_2_py`

- **`star_build_pulse_points`** (`(model: RustStarModel, add_center_point: bool = True, add_double_points: bool = False, grad_mu_threshold: float = 1.0e99) -> _ArrF64`): Builds the center-to-surface pulsation structure array `(n_pts, 26)` containing physical variables and Dziembowski variables $(V, V_g, A_*, U, c_1)$.
- **`star_eval_center_parabolic`** (`(r_inner: float, r_outer: float, v_inner: float, v_outer: float) -> float`): Parabolically extrapolates $v(r)$ to $r = 0$ (`pulse_utils`).
- **`star_eval_hllc_face`** (`(p_l: float, rho_l: float, cs_l: float, u_l: float, dm_l: float, p_r: float, rho_r: float, cs_r: float, u_r: float, dm_r: float, r_face: float, m_grav_face: float, cgrav_face: float) -> tuple[float, float, float, float, float, float, float]`): Evaluates the well-balanced Lagrangian HLLC Riemann state at an interior cell interface (`hydro_riemann`).
- **`star_eval_hllc_surface_face`** (`(p_surf_cell: float, rho_surf: float, cs_surf: float, u_surf: float, dm_surf: float, r_surf: float, m_surf: float, cgrav_surf: float, p_bc: float) -> tuple[float, float, float, float, float, float, float]`): Evaluates the surface ($k = 0$) HLLC Riemann state (`hydro_riemann`).
- **`star_eval_riemann_momentum_residual`** (`(u_k: float, u_old_k: float, dt: float, dm_k: float, r_outer: float, r_inner: float, p_face_outer: float, p_face_inner: float, p_cell_k: float, g_cell_k: float) -> float`): Evaluates the spherical Lagrangian HLLC momentum residual for cell $k$ (`hydro_momentum`).
- **`star_eval_riemann_profile`** (`(model: RustStarModel, u_cell: _ArrF64, u_old_cell: _ArrF64, p_surf_bc: float = 0.0, parallel: bool = True) -> tuple[_ArrF64, _ArrF64]`): Evaluates full-profile HLLC Riemann face states `(nz, 7)` and cell momentum residuals `(nz,)` across all zones of `model`.
- **`star_eval_rsp2_face`** (`(r_face: float, m_grav_face: float, cgrav_face: float, p_face: float, rho_face: float, t_face: float, cp_face: float, grad_ad_face: float, kappa_face: float, w_inner: float, w_outer: float, t4_inner: float, t4_outer: float, lnp_inner: float, lnp_outer: float, chi_face: float, du_over_r_face: float, dm_bar: float, alfa: float = 1.5, alfac: float = 1.0, alfat: float = 0.01) -> tuple[float, float, float, float, float, float]`): Evaluates face-centered `RSP2` scale height, superadiabaticity, viscous stress, and radiative/convective/turbulent luminosities (`hydro_rsp2`).
- **`star_eval_rsp2_zone`** (`(w: float, t: float, rho: float, p: float, cp: float, grad_ad: float, delta: float, kappa: float, hp_cell: float, grad_t: float, r_outer: float, r_inner: float, u_outer: float, u_inner: float, dm: float, dlnd_dt: float = 0.0, alfa: float = 1.5, alfap: float = 1.0, alfas: float = 1.0, alfac: float = 1.0, alfad: float = 1.0, alfam: float = 0.25, alfat: float = 0.01, gammar: float = 1.0, min_w2_for_turbulence: float = 0.0, w_min_for_damping: float = 0.0) -> tuple[float, float, float, float, float, float, float]`): Evaluates a cell's `RSP2` turbulent pressure, eddy viscosity, viscous heating, buoyant source, dissipation sink, radiative cooling sink, and net turbulent energy rate (`hydro_rsp2`).
- **`star_solve_stationary_rsp2_w`** (`(t: float, rho: float, p: float, cp: float, grad_ad: float, delta: float, kappa: float, hp_cell: float, grad_t: float, alfa: float = 1.5, alfas: float = 1.0, alfad: float = 1.0, gammar: float = 1.0) -> float`): Solves stationary local `RSP2` turbulent balance for $w$ (`rsp2_solve`).
- **`star_export_cafein`** (`(model: RustStarModel, add_center_point: bool = True) -> str`): Exports `model` in `CAFEIN` ASCII format (`pulse_cafein`).
- **`star_export_fgong`** (`(model: RustStarModel, add_center_point: bool = True, keep_surface_point: bool = False, add_atmosphere: bool = False) -> str`): Exports `model` as `FGONG` text (`star_export_pulse_data` in `star/private/pulse.f90`).
- **`star_export_gr1d`** (`(model: RustStarModel, _add_center_point: bool = True) -> str`): Exports `model` in `GR1D` ASCII format (`pulse_gr1d`).
- **`star_export_gsm`** (`(model: RustStarModel, add_center_point: bool = True) -> bytes`): Exports `model` in `GYRE` `GSM` v110 format (`pulse_gsm`).
- **`star_export_gyre`** (`(model: RustStarModel, add_center_point: bool = True, keep_surface_point: bool = False, add_atmosphere: bool = False) -> str`): Exports `model` as `GYRE` text (`star_export_pulse_data` in `star/private/pulse.f90`).
- **`star_export_osc`** (`(model: RustStarModel, add_center_point: bool = True, keep_surface_point: bool = False, add_atmosphere: bool = False) -> str`): Exports `model` as `OSC` text (`star_export_pulse_data` in `star/private/pulse.f90`).
- **`star_export_pulse_by_format`** (`(model: RustStarModel, format: str, add_center_point: bool = True) -> str`): Exports `model` in any pulsation format (`"gyre"`, `"fgong"`, `"osc"`, `"gsm"`, `"saio"`, `"cafein"`, `"gr1d"`).
- **`star_export_saio`** (`(model: RustStarModel, add_center_point: bool = True) -> str`): Exports `model` in `SAIO` ASCII format (`pulse_saio`).
- **`star_export_stella_pair`** (`(model: RustStarModel) -> tuple[str, str]`): Exports `model` as a `STELLA` `(hyd_str, abn_str)` pair (`stella_io`).
- **`star_run_stella_from_hyd_abn`** (`(hyd_str: str, abn_str: str, eta_str: str, times_days: list[float]) -> tuple[list[tuple[float, float]], list[tuple[float, float, float, float]]]`): Runs `STELLA` radiation-hydrodynamics light-curve and Sobolev `Fe II` 5169 line-velocity evaluation on `.hyd`, `.abn`, and `FeII_5169_eta.dat` strings.
- **`star_solve_adipls_fgong_modes`** (`(fgong_str: str, l: int, freq_min_uhz: float, freq_max_uhz: float, n_scan: int = 500) -> list[tuple[int, int, int, int, float, float, float, float]]`): Solves the 4th-order `ADIPLS` BVP on an `FGONG` model string.
- **`star_solve_gyre_mesa_modes`** (`(mesa_str: str, l: int, omega_min: float, omega_max: float, n_scan: int = 200, _nonadiabatic: bool = True) -> list[tuple[int, int, int, int, float, float, float, float, float]]`): Solves the 4th-order BVP on a `GYRE` MESA-format model string.


## Frozen JAX Engine Status (`flexmesa.star.JAX_ENGINE_STATUS`)

- **`JAX_ENGINE_STATUS`** (`str`, `"frozen-unverified-until-#143"`): Marks the derived JAX `star` engine twins as frozen and not an upstream parity oracle until the canonical Rust engine passes the upstream-history harness (`#143`, `#166`).
