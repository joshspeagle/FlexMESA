# Tutorial 6: End-to-End Differentiable JAX Workflows (`float64`)

A central scientific motivation of `FlexMESA` is enabling gradient-based calibration, sensitivity analysis, and Hamiltonian Monte Carlo inference across stellar microphysics, 1D Henyey structure solvers, binary mass transfer, and asteroseismology. Because importing `flexmesa` automatically enables 64-bit IEEE-754 floating-point precision (`jax_enable_x64 = True`), every `_jax` kernel and `backend="jax"` dispatch entry point supports exact forward-mode (`jax.jacfwd`, `jax.jvp`) and reverse-mode (`jax.grad`, `jax.vjp`) automatic differentiation alongside vectorized batching (`jax.vmap`).

This tutorial demonstrates end-to-end differentiable workflows across six core modules:

1. **Equations of State (`eos.eval_helm_eos`, `eos.eos_dt_get`)**
2. **Compton & Conduction Opacities (`kap.compton_opacity_jax`, `kap.combine_rad_with_conduction_jax`)**
3. **Plasma Screening & Nuclear Reaction Networks (`rates.screen_extended_jax`, `net.eval_net_jax`)**
4. **Convective Mixing-Length Theory & `RSP2` Turbulence (`turb.set_mlt_jax`, `star.eval_rsp2_zone_jax`)**
5. **Block-Tridiagonal Henyey Linear Solves (`mtx.solve_block_tridiagonal_jax` with `pcr` and `@jax.custom_vjp` `rust_ffi`)**
6. **Binary Roche-Lobe & Asteroseismic Likelihoods (`binary.eval_rlobe_jax`, `astero.jax_kernels.eval_combined_corr_jax`, `astero.jax_kernels.eval_astero_chi2_jax`)**

---

## 1. Differentiating Equations of State, Opacity & Nuclear Networks

In traditional stellar evolution codes, thermodynamic and nuclear derivatives with respect to composition $\{X_i\}$ or temperature $T$ are either omitted or approximated by finite differences. In `FlexMESA`, we can compose `eos`, `kap`, `rates`, and `net` inside a single `@jax.jit` function and compute exact gradients or Jacobians via `jax.grad` and `jax.jacfwd`:

```python
import jax
import jax.numpy as jnp
from flexmesa import chem, eos, kap, net, rates

# 1. Differentiate HELM free-energy EOS pressure w.r.t. (T, rho, abar, zbar)
def helm_ln_pgas(params: jax.Array) -> jax.Array:
    t_in, rho_in, abar_in, zbar_in = params[0], params[1], params[2], params[3]
    out, _, _ = eos.eval_helm_eos(t_in, rho_in, abar_in, zbar_in, backend="jax")
    return out.res[0]  # ln(P_gas)

grad_helm = jax.grad(helm_ln_pgas)(
    jnp.array([5.0e7, 1.0e3, 1.30, 1.10], dtype=jnp.float64)
)
print(
    f"HELM grad(ln P_gas): d/dT={float(grad_helm[0]):.4e} K^-1, "
    f"d/drho={float(grad_helm[1]):.4e} cm^3/g, d/dAbar={float(grad_helm[2]):.4f}"
)

# 2. Differentiate Poutanen (2017) Compton opacity and conductive combination
d_kap_dt = jax.grad(
    lambda t_in: kap.compton_opacity_jax(
        rho=10.0,
        t=t_in,
        lnfree_e=-0.2,
        d_lnfree_e_dln_rho=0.0,
        d_lnfree_e_dln_t=0.0,
        eta=-2.0,
        d_eta_dln_rho=1.0,
        d_eta_dln_t=-1.5,
    )[0]
)(jnp.float64(2.0e8))
print(f"Compton opacity sensitivity d(kap_Compton)/dT = {float(d_kap_dt):.4e} cm^2/g/K")

# 3. Differentiate extended plasma screening and basic.net nuclear energy generation
d_scor_drho = jax.grad(
    lambda d_in: rates.screen_extended_jax(
        temp=jnp.float64(1.5e7),
        den=d_in,
        zbar=jnp.float64(1.1),
        abar=jnp.float64(1.3),
        z2bar=jnp.float64(1.4),
        a1=jnp.float64(1.0),
        z1=jnp.float64(1.0),
        a2=jnp.float64(14.0),
        z2=jnp.float64(7.0),
    )[0]
)(jnp.float64(100.0))

net_info = net.get_net_info("basic.net")
xa_init = jnp.array(
    [0.70, 1.0e-4, 0.28, 0.003, 0.001, 0.009, 0.002, 0.0049], dtype=jnp.float64
)
comp = chem.get_composition_info(net_info.chem_id, xa_init)

deps_dx = jax.grad(
    lambda x_vec: net.eval_net_jax(
        "basic.net",
        x_vec,
        jnp.float64(1.8e7),
        jnp.float64(80.0),
        abar=float(comp.abar),
        zbar=float(comp.zbar),
        z2bar=float(comp.z2bar),
        ye=float(comp.ye),
    )[0]
)(xa_init)
print(
    f"Extended screening d(scor)/drho = {float(d_scor_drho):.4e} cm^3/g, "
    f"basic.net d(eps_nuc)/dX(1H) = {float(deps_dx[0]):.4e} erg/g/s"
)
```

---

## 2. Vectorized `jax.vmap` and Differentiable Block-Tridiagonal Henyey Solves

Solving the coupled 1D Henyey structure equations requires inverting an $N_{\mathrm{zones}} \times N_{\mathrm{zones}}$ block-tridiagonal Jacobian $\mathbf{J}\,\delta\mathbf{x} = -\mathbf{R}$. Rather than unrolling Newton or Thomas elimination loops inside the tape, `mtx.solve_block_tridiagonal_jax` supports two differentiable modes:

- **`solver_mode="pcr"`**: Pure-JAX $\lceil \log_2 N_{\mathrm{zones}} \rceil$ Parallel Cyclic Reduction using `jax.lax.while_loop` and `jnp.linalg.solve` on GPU/CPU.
- **`solver_mode="rust_ffi"`**: Zero-copy custom VJP (`@jax.custom_vjp`) that delegates both the forward solve $\mathbf{J}\,\mathbf{x} = \mathbf{b}$ and the transposed adjoint solve $\mathbf{J}^\top \boldsymbol{\lambda} = \bar{\mathbf{x}}$ to the multi-threaded Rust `bcyclic` solver:

$$
\bar{\mathbf{b}} = \mathbf{J}^{-\top} \bar{\mathbf{x}}, \qquad \bar{\mathbf{D}}_k = -\boldsymbol{\lambda}_k \mathbf{x}_k^\top, \qquad \bar{\mathbf{L}}_k = -\boldsymbol{\lambda}_k \mathbf{x}_{k-1}^\top, \qquad \bar{\mathbf{U}}_k = -\boldsymbol{\lambda}_k \mathbf{x}_{k+1}^\top
$$

```python
import jax
import jax.numpy as jnp
from flexmesa import mtx, turb

# 1. Vectorize and differentiate Cox MLT across a batch of 64 radial zones via jax.vmap
gradr_batch = jnp.linspace(0.35, 0.55, 64, dtype=jnp.float64)
vmapped_grad_mlt = jax.jit(
    jax.vmap(
        jax.grad(
            lambda gr: turb.set_mlt_jax(
                1.0, 1.0, 2.0e8, 2.7e4, 1.5e9,
                1.0e-5, 1.0e8, 1.0e5, 1.0, gr, 0.40, 0.40,
            )[1]
        )
    )
)
dgradt_dgradr_batch = vmapped_grad_mlt(gradr_batch)
print(
    f"Radiative zone d(grad_T)/d(grad_r) = {float(dgradt_dgradr_batch[0]):.6f}, "
    f"Convective zone d(grad_T)/d(grad_r) = {float(dgradt_dgradr_batch[-1]):.6f}"
)

# 2. Differentiate through a block-tridiagonal Henyey system using both "pcr" and "rust_ffi"
nz, nb = 16, 4
eye_block = jnp.broadcast_to(4.0 * jnp.eye(nb, dtype=jnp.float64), (nz, nb, nb))
sub_block = jnp.broadcast_to(-0.5 * jnp.eye(nb, dtype=jnp.float64), (nz, nb, nb))
rhs_vec = jnp.ones((nz, nb), dtype=jnp.float64)

def loss_pcr(scale: jax.Array) -> jax.Array:
    sol = mtx.solve_block_tridiagonal_jax(
        sub_block, scale * eye_block, sub_block, rhs_vec, solver_mode="pcr"
    )
    return jnp.sum(sol * sol)

def loss_ffi(scale: jax.Array) -> jax.Array:
    sol = mtx.solve_block_tridiagonal_jax(
        sub_block, scale * eye_block, sub_block, rhs_vec, solver_mode="rust_ffi"
    )
    return jnp.sum(sol * sol)

g_pcr = jax.grad(loss_pcr)(jnp.float64(1.0))
g_ffi = jax.grad(loss_ffi)(jnp.float64(1.0))
print(
    f"Block-tridiagonal adjoint gradient parity: "
    f"grad(PCR) = {float(g_pcr):.12f}, grad(rust_ffi) = {float(g_ffi):.12f}, "
    f"rel_diff = {abs(float(g_pcr - g_ffi)) / abs(float(g_pcr)):.2e}"
)
```

---

## 3. Differentiable Binary & Asteroseismic Forward Models

Finally, `binary` and `astero.jax_kernels` allow you to differentiate binary Roche-lobe radii, implicit mass-transfer rates, and surface-corrected asteroseismic $\chi^2$ likelihoods directly with respect to physical stellar parameters $(M_1, M_2, R, T_{\mathrm{eff}})$:

```python
import jax
import jax.numpy as jnp
from flexmesa import astero, binary, constants

# 1. Jacobian of Eggleton Roche-lobe radius R_RL,1 w.r.t. (M_1, M_2, a)
drl_dparams = jax.jacfwd(
    lambda p: binary.eval_rlobe_jax(p[0], p[1], p[2])
)(
    jnp.array(
        [2.0 * constants.MSUN, 1.4 * constants.MSUN, 6.0 * constants.RSUN],
        dtype=jnp.float64,
    )
)
print(
    f"Eggleton Jacobian: dR_RL/dM1 = {float(drl_dparams[0]):.3e} cm/g, "
    f"dR_RL/dM2 = {float(drl_dparams[1]):.3e} cm/g, dR_RL/da = {float(drl_dparams[2]):.5f}"
)

# 2. Differentiate a combined Ball & Gizon (2014) + Spectroscopic chi^2 loss w.r.t. (M/Msun, R/Rsun)
obs_freqs = jnp.array([2294.2, 2429.1, 2564.0, 2699.0, 2834.1], dtype=jnp.float64)
obs_sigmas = jnp.full_like(obs_freqs, 0.20)
orders = jnp.array([16.0, 17.0, 18.0, 19.0, 20.0], dtype=jnp.float64)
inertias = jnp.array([1.25, 1.10, 1.00, 0.95, 0.92], dtype=jnp.float64)
spec_obs = jnp.array([5777.0, 0.0], dtype=jnp.float64)
spec_sig = jnp.array([50.0, 0.05], dtype=jnp.float64)

@jax.jit
def seismic_forward_chi2(theta: jax.Array) -> jax.Array:
    m_msun, r_rsun, teff_k = theta[0], theta[1], theta[2]
    dnu_mod = astero.jax_kernels.eval_delta_nu_radius_scaling_jax(
        m_msun * constants.MSUN, r_rsun * constants.RSUN
    )
    mod_freqs = dnu_mod * (orders + 1.45)
    nu_corr, _, _ = astero.jax_kernels.eval_combined_corr_jax(
        obs_freqs, obs_sigmas, mod_freqs, inertias, jnp.float64(1.0)
    )
    chi2_tot, _, _ = astero.jax_kernels.eval_astero_chi2_jax(
        obs_freqs,
        obs_sigmas,
        nu_corr,
        spec_obs,
        spec_sig,
        jnp.array([teff_k, 0.0], dtype=jnp.float64),
        jnp.float64(2.0 / 3.0),
    )
    return chi2_tot

theta0 = jnp.array([1.01, 1.005, 5790.0], dtype=jnp.float64)
val_chi2, grad_chi2 = jax.value_and_grad(seismic_forward_chi2)(theta0)
print(
    f"Forward chi^2 = {float(val_chi2):.4f}, "
    f"d(chi^2)/d(M/Msun) = {float(grad_chi2[0]):.2f}, "
    f"d(chi^2)/d(R/Rsun) = {float(grad_chi2[1]):.2f}, "
    f"d(chi^2)/dTeff = {float(grad_chi2[2]):.4f} K^-1"
)
```
