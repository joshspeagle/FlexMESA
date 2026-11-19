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

- **Model Initialization & I/O**: Zero-Age Main Sequence (`StarModel.from_zams`), homothetic Pre-Main-Sequence contraction (`StarModel.from_pre_ms`), upstream ASCII `.mod` readers/writers (`from_mod_file`, `from_mod_str`, `write_mod_file`, `to_mod_str`), generated little-endian `serde` binary `.photo` checkpoints (`FMPHOTO2` v22 via `to_photo_bytes`, `try_write_photo_bytes`, `from_photo_bytes`, `load_photo_bytes`, `verify_photo_restart`, saving primary/start/old arrays plus `controls`, `job`, `rsp`, and cumulative solver counters while skipping ephemeral `MicrophysicsCache` hit/miss counters so restarts remain bit-exact against uninterrupted runs; `FlexMESA` `.photo` bytes remain internal and are not binary-compatible with Fortran MESA `photos/`), and MESA `26.4.1` `LOGS/history.data`, `LOGS/profileN.data` (`StarModel.to_profile_str`), and `LOGS/profiles.index` column evaluators, incremental writers (`StarLogsWriter`, `IncrementalHistoryWriter`, `run_inlist_stage_with_logs`, `run_test_suite_case_with_logs`), and lossless ASCII readers (`read_mesa_log_file`, `read_profiles_index_file`), with `HistoryRow.log_lh` reporting hydrogen-burning luminosity (`pp + cno`) separately from helium, metal, and net nuclear burning. `&controls` and `&star_job` defaults match MESA `26.4.1` (`#149`), using upstream scalar sentinels (`1e36`, `11.0`, `12.0`, `99.0`, `3e7`, `0.99`, `0.0`, `-1`, `-111`), explicit `set_*` / `set_initial_*` rotation booleans, and explicit `&kap::zbase` (`-1` default without silent cross-namelist copying).
- **Coupled Henyey Solver & Time Evolution**: Staggered 1D cell/face grids, 24-slot 3-cell stencil automatic differentiation (`StarAdOrder1`), coupled block-tridiagonal Newton-Raphson relaxation (`solve_henyey`, `step_struct_burn_mix`, `evolve`) porting MESA `26.4.1`'s `do_solver_work`, `adjust_correction`, `Bdomain`, `scale_correction_norm`, retry-dependent `max_tries` schedule, `solver_itermin`, high-$T$ correction tolerance relaxation, `check_after_converge`, and `sizequ`/`sizeB` norms (`#152`, `#177`, `#251`, `#252`), structured `StarError::{StepFailure, Terminated}` diagnostics (`StepFailureInfo` carrying zone, equation, paired variable, value, norms, and retry cause, with zero-retry fast-fail on `InvalidInput`/`IoError`/`Terminated`), in-place `PrimaryStateBuffer` rollback (`restore_mesh_on_retry`), 2-slot bit-exact `MicrophysicsCache` reuse (`micro_cache_hits`, `micro_cache_misses`), retry-proof solver/redo counters (`total_num_solver_calls`, `num_redos`, `num_retries`), optional JSONL step telemetry (`FLEXMESA_STEP_TRACE` / `FLEXMESA_STEP_TRACE_FILE`), 3-pass conservative remeshing (`do_adjust_mesh`, `remap_onto_dq`; `#153`), 3-region `do_adjust_mass` with Neumaier-compensated summation, `select_mdot_action` rotational-wind redo/retry control (`#139`, `#152`), and cancellation-free `eps_mdot` mass-change heating (`#154`, `#159`), sub-cell `CoreMasses` and `PowerCategories` reporting, and Söderlind-Wang H211 PI timestep control (`timestep_controller`, `eval_h211_timestep_jax`) with full `timestep.f90` soft/hard limiters and `min_timestep_limit` / `max_number_retries` / `redo_limit` termination (`#152`, `#177`, `#253`).
- **Advanced Stellar Physics** *(partial/unverified against MESA `26.4.1`; `#153`-`#159`)*: 16-set exponential and step convective overshooting with `'none'` short-circuiting, convection-zone thickness $H_P$ cap, and Péclet damping (`OvershootConfig`), predictive mixing (`PredictiveParamSet`; `#155`), Reimers/Blocker/Vink/Björklund/Nugis-Lamers/Dutch stellar winds with rotational boost (`#154`), Burgers multi-fluid atomic diffusion (`do_element_diffusion`) with post-diffusion `[0, 1]` abundance clamping and Neumaier normalization (`#156`, `op_radacc` absent), Caplan strong-coupling OCP diffusion, white dwarf C/O crystallization phase separation (`#156`), shellular Roche rotation, angular momentum mixing (`solve_omega_mix`), Spruit-Tayler magnetic dynamos (`#158`), coupled 3-cell `StarAdOrder1` spherical HLLC Riemann `u_flag` hydrodynamics (`riemann_ad`) and Duffell (2016) Rayleigh-Taylor `alpha_rti` shock mixing (`hydro_rti`), `RSP2` radial pulsation equations (`#153`), and `GYRE`/`FGONG`/`OSC` structure exporters (`to_gyre_str`, `to_fgong_str`, `to_osc_str`).
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
