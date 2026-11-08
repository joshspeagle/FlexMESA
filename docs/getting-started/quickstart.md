# 5-Minute Quickstart

This walkthrough introduces the five core workflows in `FlexMESA`: evaluating equations of state and nuclear reaction networks across CPU (`backend="rust"`) and GPU (`backend="jax"`), differentiating microphysics with `jax.grad`, evolving a $1\,M_\odot$ Zero-Age Main Sequence (`ZAMS`) star, evolving a mass-transferring binary system, and computing asteroseismic surface corrections and publication-styled diagnostic figures.

---

## 1. Evaluating Equations of State & Nuclear Networks (`rust` vs. `jax`)

Every shell-local microphysics kernel in `FlexMESA` exposes a unified dispatcher controlled by `backend="auto" | "rust" | "jax"`. When `backend="auto"` (the default), standard Python floats and `numpy.ndarray` inputs route zero-copy to the parallel Safe Rust engine (`flexmesa._rust`), whereas `jax.Array` inputs compile into branchless `@jax.jit` kernels in IEEE-754 double precision (`jnp.float64`).

```python
import jax.numpy as jnp
import numpy as np
from flexmesa import chem, eos, net

# 1. Unified 26-slot EOS evaluation on CPU (Safe Rust) and (P_gas, T) inversion
cids = (chem.get_nuclide_index("c12"), chem.get_nuclide_index("o16"))
xa_co = np.array([0.50, 0.50], dtype=np.float64)
dt_out = eos.eos_dt_get(
    rho=1.0e6, temp=1.0e7, chem_id=cids, xa=xa_co, use_skye=True
)
pgas = float(np.exp(dt_out.res[eos.I_LN_PGAS]))
pt_out = eos.eos_pt_get(
    pgas=pgas, temp=1.0e7, chem_id=cids, xa=xa_co, use_skye=True
)
print(
    f"Skye EOS: ln(P_gas)={dt_out.res[eos.I_LN_PGAS]:.6f}, "
    f"Gamma_1={dt_out.res[eos.I_GAMMA1]:.6f}, "
    f"recovered log10(rho)={pt_out.log10_rho:.10f}"
)

# 2. Vectorized 1024-zone Skye EOS evaluation on JAX (@jax.jit + vmap)
rho_grid = jnp.logspace(4.0, 7.0, 1024, dtype=jnp.float64)
t_grid = jnp.full(1024, 1.0e7, dtype=jnp.float64)
abar_grid = jnp.full(1024, 13.714285714285714, dtype=jnp.float64)
zbar_grid = jnp.full(1024, 6.857142857142857, dtype=jnp.float64)
xa_grid = jnp.tile(jnp.asarray(xa_co), (1024, 1))
skye_jax = eos.eval_skye_eos_zones(
    rho_grid,
    t_grid,
    abar_grid,
    zbar_grid,
    xa_grid,
    z=(6, 8),
    acmi=jnp.array([12.0, 15.99491461957], dtype=jnp.float64),
    a=jnp.array([12.0, 16.0], dtype=jnp.float64),
    backend="jax",
)

# 3. Evaluate the 8-isotope basic.net nuclear reaction network
net_info = net.get_net_info("basic.net")
xa_solar = np.array(
    [0.70, 3.0e-5, 0.28, 0.0034, 0.001, 0.0096, 0.002, 0.0039702],
    dtype=np.float64,
)
comp = chem.get_composition_info(net_info.chem_id, xa_solar)
net_res = net.eval_net(
    "basic.net",
    x=xa_solar,
    temp=8.0e7,
    rho=100.0,
    abar=float(comp.abar),
    zbar=float(comp.zbar),
    z2bar=float(comp.z2bar),
    ye=float(comp.ye),
    screening_mode="extended",
)
print(
    f"basic.net: eps_nuc={net_res.eps_nuc:.6e} erg/g/s, "
    f"d(eps_nuc)/dT={net_res.d_eps_nuc_dt:.6e}"
)
```

---

## 2. Differentiating Microphysics with `jax.grad`

Because `FlexMESA` implements branchless `@jax.jit` kernels alongside `@jax.custom_jvp` and `@jax.custom_vjp` analytical Jacobian rules for tabulated opacities (`flexmesa.kap`), nuclear reaction networks (`flexmesa.net`), and block-tridiagonal solvers (`flexmesa.mtx`), you can compose any microphysics evaluator directly with `jax.grad`, `jax.jacfwd`, or `jax.hessian`:

```python
import jax
import jax.numpy as jnp
import numpy as np
from flexmesa import chem, kap, net, neu

# 1. Exact temperature and density gradients of Rosseland mean opacity
kap_fn = kap.make_differentiable_kap(z=0.02, x=0.70, zbase=0.02, zbar=1.28)
dlogkap_dlogrho, dlogkap_dlogt = jax.grad(kap_fn, argnums=(0, 1))(
    jnp.float64(-5.0), jnp.float64(6.0)
)
print(
    f"Opacity derivatives: dln(kap)/dln(rho)={float(dlogkap_dlogrho):.6f}, "
    f"dln(kap)/dln(T)={float(dlogkap_dlogt):.6f}"
)

# 2. Reverse-mode gradient of nuclear energy generation w.r.t. composition X_i
info = net.get_net_info("basic.net")
xa_j = jnp.array(
    [0.70, 3.0e-5, 0.28, 0.0034, 0.001, 0.0096, 0.002, 0.0039702],
    dtype=jnp.float64,
)
comp = chem.get_composition_info(info.chem_id, np.asarray(xa_j))
deps_dxa = jax.grad(
    lambda x: net.eval_net_jax(
        "basic.net",
        x,
        jnp.float64(8.0e7),
        jnp.float64(100.0),
        float(comp.abar),
        float(comp.zbar),
        float(comp.z2bar),
        float(comp.ye),
    )[0]
)(xa_j)
print(f"d(eps_nuc)/dX(h1) = {float(deps_dxa[0]):.6e} erg/g/s")

# 3. Second derivative (curvature) of pair + plasmon neutrino losses w.r.t. T
d2eps_dt2 = jax.grad(
    jax.grad(
        lambda t: neu.neu_pair_plas_jax(t, jnp.float64(1.0e6), 12.0, 6.0)
    )
)(jnp.float64(1.0e9))
print(f"d^2(eps_neu)/dT^2 = {float(d2eps_dt2):.6e} erg/g/s/K^2")
```

---

## 3. Building & Evolving a $1\,M_\odot$ ZAMS Stellar Model (`flexmesa.star`)

`StarModel` manages a 1D spherical star discretized into $\mathrm{nz}$ Lagrangian shells ordered from the outer photosphere ($k = 0$) to the stellar center ($k = \mathrm{nz} - 1$). Calling `model.evolve()` advances the coupled hydrostatic structure, nuclear reaction network, and convective mixing equations using the parallel Hirshman et al. (2010) `bcyclic` block-tridiagonal Henyey solver and the Söderlind & Wang (2006) H211 digital timestep controller:

```python
import numpy as np
from flexmesa import star

# Initialize a 1.0 Msun, Z = 0.02 Zero-Age Main Sequence star
model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
print(f"Initialized {model.nz} zones with species {model.iso_names}")

# Record evolutionary history across 15 coupled Henyey steps
history = [model.history_row()]
for _ in range(15):
    model.evolve_step(do_remesh=False, op_split_burn=False, use_bcyclic=True)
    history.append(model.history_row())

last = history[-1]
print(
    f"Step {last.model_number}: age={last.star_age:.3e} yr, "
    f"log10(Teff)={last.log_teff:.4f}, log10(L/Lsun)={last.log_l:.4f}, "
    f"center H1={last.center_h1:.6f}"
)

# Inspect radial profiles (k = 0 is surface, k = nz - 1 is center)
log10_t_profile = model.ln_t / np.log(10.0)
log10_rho_profile = model.lnd / np.log(10.0)
print(
    f"Surface T = {10.0**log10_t_profile[0]:.1f} K, "
    f"Central T = {10.0**log10_t_profile[-1]:.3e} K"
)
```

---

## 4. Evolving a Mass-Transferring Binary (`flexmesa.binary`)

`BinarySystem` couples a donor `StarModel` to either a compact point-mass companion (`from_point_mass`, supporting Kerr black-hole ISCO spin and Eddington-limited accretion) or an evolved secondary star (`from_two_stars`). Each call to `binary.step(dt)` updates the Keplerian orbit, evaluates Eggleton (1983) Roche-lobe radii and implicit Ritter (1988) / Kolb & Ritter (1990) Roche-lobe overflow (`RLOF`) mass-transfer rates, and integrates gravitational-wave, magnetic-braking, and tidal torques:

```python
import jax
import jax.numpy as jnp
from flexmesa import binary, constants, star

# 1. Construct a 2.0 Msun donor orbiting a 1.4 Msun compact accretor
donor = star.StarModel.from_zams(initial_mass=2.0, initial_z=0.02)
sys = binary.BinarySystem.from_point_mass(
    donor,
    companion_mass_msun=1.4,
    initial_period_days=1.2,
    eccentricity=0.0,
)
dt_next, iters = sys.step(dt=1.0e4 * constants.SECYER)
rl_d, rl_a = sys.rl
gap_d, _ = sys.rl_relative_gap
print(
    f"Orbital separation = {sys.separation / constants.RSUN:.3f} Rsun, "
    f"R_RL,1 = {rl_d / constants.RSUN:.3f} Rsun, "
    f"relative gap = {gap_d:.4f}, dM/dt = {sys.mtransfer_rate:.3e} g/s"
)

# 2. Differentiate the optically thin Ritter (1988) RLOF rate w.r.t. donor radius
dmdot_dr = jax.grad(
    lambda r_d: binary.eval_ritter_mdot_thin_jax(
        2.0 * constants.MSUN,
        1.4 * constants.MSUN,
        r_d,
        rl_d,
        2.0e-7,
        2.5e5,
        6500.0,
        0.61,
        False,
    )
)(jnp.float64(1.001 * rl_d))
print(f"Ritter RLOF sensitivity d(mdot)/dR_1 = {float(dmdot_dr):.4e} g/s/cm")
```

---

## 5. Asteroseismology (`flexmesa.astero`) & Diagnostic Plots (`flexmesa.plotting`)

`flexmesa.astero` computes cumulative acoustic transit integrals, global scaling relations ($\Delta\nu$, $\nu_{\max}$, $\nu_{\mathrm{ac}}$, $\Delta\Pi_1$), five empirical surface-effect frequency corrections, and Roxburgh & Vorontsov (2003) 5-point frequency separation ratios. `flexmesa.plotting` (commonly aliased as `plot`) generates publication-ready `matplotlib` figures that replace Fortran MESA's interactive X11 `PGSTAR` windows:

```python
import matplotlib.pyplot as plt
import numpy as np
from flexmesa import astero, constants
from flexmesa import plotting as plot

# 1. Compute asymptotic and solar-scaled global oscillation parameters
dnu_solar = astero.eval_delta_nu_radius_scaling(
    m_g=1.0 * constants.MSUN, r_phot_cm=1.0 * constants.RSUN
)
nu_max_solar = astero.eval_nu_max(
    m_g=1.0 * constants.MSUN,
    r_phot_cm=1.0 * constants.RSUN,
    teff_k=5777.0,
)
print(f"Solar-scaled Delta_nu = {dnu_solar:.2f} uHz, nu_max = {nu_max_solar:.1f} uHz")

# 2. Apply Ball & Gizon (2014) 2-term combined surface-effect correction
n_ord = np.arange(14, 26, dtype=np.float64)
nu_mod = 135.0 * (n_ord + 1.45)
inertia = 1.0e-7 * np.exp((3100.0 - nu_mod) / 1000.0)
sigma = np.full_like(nu_mod, 0.15)
nu_obs = nu_mod - 1.2 * ((nu_mod / 3090.0) ** 3) / inertia

corr = astero.eval_surface_correction(
    "combined", nu_obs, sigma, nu_mod, inertia, nu_max=3090.0
)
chi2_corr = astero.eval_astero_chi2(
    nu_obs, sigma, corr["nu_corrected"], [], [], []
)["chi2_frequencies"]
print(
    f"a_3 = {corr['p0']:.4e}, a_-1 = {corr['p1']:.4e}, "
    f"Surface-corrected chi^2 = {chi2_corr:.4e}"
)

# 3. Generate a publication-styled asteroseismic echelle diagram
fig, ax = plot.plot_echelle_diagram(
    obs_freqs=nu_obs,
    obs_sigmas=sigma,
    delta_nu=135.0,
    model_freqs_uncorr=nu_mod,
    model_freqs_corr=corr["nu_corrected"],
)
plt.close(fig)
```
