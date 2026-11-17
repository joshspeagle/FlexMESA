# 1D Stellar Structure & Evolution (`flexmesa.star`)

The `flexmesa.star` subpackage implements the 1D Lagrangian stellar structure and evolution engine, bridging the Safe Rust core (`crates/flexmesa-star`) with pure-JAX (`float64`) differentiable structure, wind, mixing, hydrodynamic, and pulsation kernels. It provides:

- **Model Initialization & I/O**: Zero-Age Main Sequence (`StarModel.from_zams`), homothetic Pre-Main-Sequence contraction (`StarModel.from_pre_ms`), upstream ASCII `.mod` readers/writers (`from_mod_file`, `from_mod_str`, `write_mod_file`, `to_mod_str`), optional post-load `doing_relax` thermal settling for relaxed `.mod` checkpoints, internal IEEE-754 binary `.photo` snapshots (`to_photo_bytes`, `from_photo_bytes`, `verify_photo_restart`, including `eps_mdot_old` and step-history scalars; Fortran MESA `.photo` binary compatibility is tracked in `#150`), and MESA `26.4.1` `LOGS/history.data`, `LOGS/profileN.data` (`StarModel.to_profile_str`), and `LOGS/profiles.index` column evaluators, incremental writers (`StarLogsWriter`, `IncrementalHistoryWriter`, `run_inlist_stage_with_logs`, `run_test_suite_case_with_logs`), and lossless ASCII readers (`read_mesa_log_file`, `read_profiles_index_file`), with `HistoryRow.log_lh` reporting hydrogen-burning luminosity (`pp + cno`) separately from helium, metal, and net nuclear burning.
- **Coupled Henyey Solver & Time Evolution**: Staggered 1D cell/face grids, 24-slot 3-cell stencil automatic differentiation (`StarAdOrder1`), coupled block-tridiagonal Newton-Raphson relaxation (`solve_henyey`, `step_struct_burn_mix`, `evolve`) with 3-stage (`gold` / `gold2` / standard) residual and `correction_xa_limit`-filtered `sizeB` convergence norms, 3-pass conservative remeshing (`do_adjust_mesh`, `remap_onto_dq`; `#153`), 3-region `do_adjust_mass` with Neumaier-compensated summation and cancellation-free `eps_mdot` mass-change heating (`#154`, `#159`, `#177`), sub-cell `CoreMasses` and `PowerCategories` reporting, and Söderlind-Wang H211 PI timestep control (`eval_h211_timestep_jax`) with `TimestepStopControls` retry and termination rollback (`#152`).
- **Advanced Stellar Physics** *(partial/unverified against MESA `26.4.1`; `#152`-`#159`)*: 16-set exponential and step convective overshooting with `'none'` short-circuiting, convection-zone thickness $H_P$ cap, and Péclet damping (`OvershootConfig`), predictive mixing (`PredictiveParamSet`; `#155`), Reimers/Blocker/Vink/Björklund/Nugis-Lamers/Dutch stellar winds with rotational boost (`#154`), Burgers multi-fluid atomic diffusion (`do_element_diffusion`) with post-diffusion `[0, 1]` abundance clamping and Neumaier normalization (`#156`, `op_radacc` absent), Caplan strong-coupling OCP diffusion, white dwarf C/O crystallization phase separation (`#156`), shellular Roche rotation, angular momentum mixing (`solve_omega_mix`), Spruit-Tayler magnetic dynamos (`#158`), coupled 3-cell `StarAdOrder1` spherical HLLC Riemann `u_flag` hydrodynamics (`riemann_ad`) and Duffell (2016) Rayleigh-Taylor `alpha_rti` shock mixing (`hydro_rti`), `RSP2` radial pulsation equations (`#153`), and `GYRE`/`FGONG`/`OSC` structure exporters (`to_gyre_str`, `to_fgong_str`, `to_osc_str`).
- **Legacy `inlist` & `test_suite` Runners**: Recursive Fortran `inlist` bundle parsing (`parse_inlist_bundle`; full `&controls`/`&star_job` wiring tracked in `#147`), `rn` stage extraction (`parse_rn_stages`), internal linear oscillation stub and `RSP` `LINA`/envelope builders (`eval_linear_oscillation_modes` pending `GYRE`/`ADIPLS` in `#170`/`#171`, `eval_rsp_lina_modes`, `build_rsp_envelope_model`), and legacy pre-audit smoke-harness runners (`TIER1_CASES`, `TIER2_CASES`, `TIER3_CASES`, `TIER4_STAR_CASES`, `TIER4_ASTERO_CASES`, `TIER4_DEV_CASES`, `run_test_suite_case`, `run_tier1_suite`, `run_tier2_suite`, `run_tier3_suite`, `run_tier4_suite`, `run_astero_suite`, `run_dev_cases_suite`; synthetic `Tier1CaseExtras` wrappers slated for removal in `#148`, canonical MESA `26.4.1` trajectory comparison lives in `flexmesa.testing.history_compare`).

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

## Legacy Pulsation, `RSP`, `astero` & `dev_cases_*` Runner (`flexmesa.star.e2e_tier45`)

*(Note: Pre-audit smoke-harness scaffolding; synthetic `Tier1CaseExtras` shortcuts are slated for removal in `#148`, and MESA `26.4.1` trajectory parity is tracked in `flexmesa.testing.history_compare` and [Declared Known Gaps](../benchmarks/known-gaps.md).)*

::: flexmesa.star.e2e_tier45
