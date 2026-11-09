# 1D Stellar Structure & Evolution (`flexmesa.star`)

The `flexmesa.star` subpackage implements the 1D Lagrangian stellar structure and evolution engine, bridging the Safe Rust core (`crates/flexmesa-star`) with pure-JAX (`float64`) differentiable structure, wind, mixing, hydrodynamic, and pulsation kernels. It provides:

- **Model Initialization & I/O**: Zero-Age Main Sequence (`StarModel.from_zams`), homothetic Pre-Main-Sequence contraction (`StarModel.from_pre_ms`), upstream ASCII `.mod` readers/writers (`from_mod_file`, `from_mod_str`, `write_mod_file`, `to_mod_str`), and exact IEEE-754 binary `.photo` snapshots (`to_photo_bytes`, `from_photo_bytes`, `verify_photo_restart`).
- **Coupled Henyey Solver & Time Evolution**: Staggered 1D cell/face grids, 3-cell stencil automatic differentiation (`StarAdOrder1`), coupled block-tridiagonal Newton-Raphson relaxation (`solve_henyey`, `step_struct_burn_mix`, `evolve`), adaptive mesh refinement (`adjust_mesh`), and Söderlind-Wang H211 PI timestep control (`eval_h211_timestep_jax`).
- **Advanced Stellar Physics**: Exponential/step convective overshooting, Reimers/Blocker/Vink/Dutch stellar winds, Burgers multi-fluid atomic diffusion, Caplan strong-coupling OCP diffusion, white dwarf C/O crystallization phase separation, shellular Roche rotation and Spruit-Tayler magnetic dynamos, well-balanced 1D spherical HLLC Riemann hydrodynamics, `RSP2` radial pulsation equations, and `GYRE`/`FGONG`/`OSC` structure exporters (`to_gyre_str`, `to_fgong_str`, `to_osc_str`).
- **End-to-End `inlist` & `test_suite` Execution**: Recursive Fortran `inlist` bundle parsing (`parse_inlist_bundle`), `rn` stage extraction (`parse_rn_stages`), and automated upstream `test_suite` execution (`run_test_suite_case`).

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

## HLLC Hydrodynamics, `RSP2` & Pulsation Kernels (`flexmesa.star.jax_hydro_pulse`)

::: flexmesa.star.jax_hydro_pulse

---

## Inlist Bundle & `test_suite` Runner (`flexmesa.star.e2e`)

::: flexmesa.star.e2e
