# 1D Stellar Structure & Evolution (`flexmesa.star`)

The `flexmesa.star` subpackage implements the 1D Lagrangian stellar structure and evolution engine, bridging the Safe Rust core (`crates/flexmesa-star`) with pure-JAX (`float64`) differentiable structure, wind, mixing, hydrodynamic, and pulsation kernels. It provides:

- **Model Initialization & I/O**: Zero-Age Main Sequence (`StarModel.from_zams`), homothetic Pre-Main-Sequence contraction (`StarModel.from_pre_ms`), upstream ASCII `.mod` readers/writers (`from_mod_file`, `from_mod_str`, `write_mod_file`, `to_mod_str`), optional post-load `doing_relax` thermal settling for relaxed `.mod` checkpoints, and exact IEEE-754 binary `.photo` snapshots (`to_photo_bytes`, `from_photo_bytes`, `verify_photo_restart`, including `eps_mdot_old` and step-history scalars).
- **Coupled Henyey Solver & Time Evolution**: Staggered 1D cell/face grids, 24-slot 3-cell stencil automatic differentiation (`StarAdOrder1`), coupled block-tridiagonal Newton-Raphson relaxation (`solve_henyey`, `step_struct_burn_mix`, `evolve`) with 3-stage (`gold` / `gold2` / standard) residual and `correction_xa_limit`-filtered `sizeB` convergence norms, 3-pass conservative remeshing (`do_adjust_mesh`, `remap_onto_dq`), 3-region `do_adjust_mass` with Neumaier-compensated summation and cancellation-free `eps_mdot` mass-change heating, sub-cell `CoreMasses` and `PowerCategories` reporting, and Söderlind-Wang H211 PI timestep control (`eval_h211_timestep_jax`) with `TimestepStopControls` retry and termination rollback.
- **Advanced Stellar Physics**: 16-set exponential and step convective overshooting with `'none'` short-circuiting, convection-zone thickness $H_P$ cap, and Péclet damping (`OvershootConfig`), predictive mixing (`PredictiveParamSet`), Reimers/Blocker/Vink/Björklund/Nugis-Lamers/Dutch stellar winds with rotational boost, Burgers multi-fluid atomic diffusion (`do_element_diffusion`) with post-diffusion `[0, 1]` abundance clamping and Neumaier normalization, Caplan strong-coupling OCP diffusion, white dwarf C/O crystallization phase separation, shellular Roche rotation, angular momentum mixing (`solve_omega_mix`), Spruit-Tayler magnetic dynamos, coupled 3-cell `StarAdOrder1` spherical HLLC Riemann `u_flag` hydrodynamics (`riemann_ad`) and Duffell (2016) Rayleigh-Taylor `alpha_rti` shock mixing (`hydro_rti`), `RSP2` radial pulsation equations, and `GYRE`/`FGONG`/`OSC` structure exporters (`to_gyre_str`, `to_fgong_str`, `to_osc_str`).
- **End-to-End `inlist` & `test_suite` Execution**: Recursive Fortran `inlist` bundle parsing (`parse_inlist_bundle`), `rn` stage extraction (`parse_rn_stages`), linear oscillation mode and `RSP` `LINA`/envelope builders (`eval_linear_oscillation_modes`, `eval_rsp_lina_modes`, `build_rsp_envelope_model`), and automated upstream `test_suite` execution across Tiers 1, 2, 3, and 4 (`TIER1_CASES`, `TIER2_CASES`, `TIER3_CASES`, `TIER4_STAR_CASES`, `TIER4_ASTERO_CASES`, `TIER4_DEV_CASES`, `run_test_suite_case`, `run_tier1_suite`, `run_tier2_suite`, `run_tier3_suite`, `run_tier4_suite`, `run_astero_suite`, `run_dev_cases_suite`).

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

## Tier 4 Pulsation, `RSP`, `astero` & `dev_cases_*` Runner (`flexmesa.star.e2e_tier45`)

::: flexmesa.star.e2e_tier45
