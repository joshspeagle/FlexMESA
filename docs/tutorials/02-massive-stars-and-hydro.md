# Tutorial 2: Massive Stars, Shellular Rotation, Winds & HLLC Hydrodynamics

Massive stars ($M_\star \gtrsim 8\,M_\odot$) experience physical processes that play a secondary role in low-mass dwarfs: intense radiation-driven stellar winds that strip multiple solar masses of envelope material, rapid shellular rotation and magnetic-dynamo torques that mix nuclear-processed elements across radiative zones, and supersonic hydrodynamic waves during late burning stages or envelope eruptions. This tutorial demonstrates how to model a $15\,M_\odot$ star in `flexmesa.star` with shellular rotational distortion, the five hydrodynamic rotational instabilities plus the Spruit-Tayler (2002) magnetic dynamo, composite `Dutch` stellar winds, and 1D Lagrangian Harten-Lax-van Leer-Contact (`HLLC`) Riemann hydrodynamics.

---

## 1. Initializing a $15\,M_\odot$ Star and Evaluating Roche Rotational Distortion

In a rotating star with shellular angular velocity $\Omega(m)$, centrifugal acceleration distorts isobars from spheres into Roche equipotentials. Following Paxton et al. (2019, MESA V) and Fabry et al. (2022), `FlexMESA` preserves the 1D spherical form of the hydrostatic and radiative transport equations by introducing dimensionless centrifugal distortion factors $f_P(\Omega / \Omega_{\mathrm{crit}})$ and $f_T(\Omega / \Omega_{\mathrm{crit}})$, where the Eddington-reduced critical Roche angular velocity is:

$$
\Omega_{\mathrm{crit}} = \sqrt{(1 - \Gamma_{\mathrm{Edd}})\,\frac{G m_{\mathrm{grav}}}{r^3}}, \qquad \Gamma_{\mathrm{Edd}} = \frac{\kappa L}{4\pi c G m_{\mathrm{grav}}}
$$

You can evaluate $\Omega_{\mathrm{crit}}$ and $(f_P, f_T, i_{\mathrm{rot}}/r^2)$ either in Safe Rust (`_rust.star_eval_omega_crit`, `_rust.star_eval_rotation_fp_ft`) or in differentiable JAX (`star.eval_omega_crit_jax`, `star.eval_rotation_fp_ft_jax`):

```python
import jax
import jax.numpy as jnp
from flexmesa import _rust, constants, star

# Initialize a 15 Msun ZAMS star
model_15m = star.StarModel.from_zams(initial_mass=15.0, initial_z=0.02)
r_surf = float(model_15m.r[0])
m_surf = float(model_15m.m[0])
l_surf = float(model_15m.lum[0])
kap_surf = float(model_15m.opacity[0])

# 1. Compute critical Roche angular velocity at the photosphere
omega_crit_rust = _rust.star_eval_omega_crit(
    constants.STANDARD_CGRAV, m_surf, r_surf, l_surf, kap_surf
)

# 2. Evaluate Roche distortion factors at Omega / Omega_crit = 0.5
w_ratio = 0.5
fp_rust, ft_rust, irot_rust = _rust.star_eval_rotation_fp_ft(w_ratio)

# Differentiate f_P with respect to Omega / Omega_crit via jax.grad
dfp_dw = jax.grad(lambda w: star.eval_rotation_fp_ft_jax(w)[0])(jnp.float64(w_ratio))

print(f"15 Msun photosphere: Omega_crit = {omega_crit_rust:.4e} rad/s")
print(
    f"At Omega/Omega_crit = 0.5: f_P = {fp_rust:.6f}, f_T = {ft_rust:.6f}, "
    f"i_rot/r^2 = {irot_rust:.6f}, df_P/d(Omega/Omega_crit) = {float(dfp_dw):.6f}"
)
```

---

## 2. Rotational Mixing Instabilities and the Spruit-Tayler Magnetic Dynamo

Differential rotation in radiative zones drives five hydrodynamic instabilities: Dynamical Shear (`DSI`), Solberg-Høiland (`SH`), Secular Shear (`SSI`), Eddington-Sweet meridional circulation (`ES`), and Goldreich-Schubert-Fricke (`GSF`). In addition, radial shear $q = |\partial \ln \Omega / \partial \ln r|$ winds poloidal magnetic seed fields into toroidal fields until the pinch-type Tayler instability saturates (Spruit 2002), producing an effective magnetic viscosity $\nu_{\mathrm{ST}}$ (which dominates core-envelope angular momentum coupling) and compositional diffusivity $D_{\mathrm{ST}}$.

In `FlexMESA`, `_rust.star_eval_rotational_mixing_zone` and `star.eval_rotational_mixing_zone_jax` evaluate a single shell's 9-tuple `(d_dsi, d_sh, d_ssi, d_es, d_gsf, d_st, nu_st, b_phi, b_r)`, while `model.set_rotation_mixing_info()` and `model.run_omega_mix(...)` couple rotational mixing and implicit angular momentum diffusion across the entire 1D stellar profile:

```python
import jax.numpy as jnp
import numpy as np
from flexmesa import star

model_15m = star.StarModel.from_zams(initial_mass=15.0, initial_z=0.02)

# Evaluate single-zone rotational instabilities and Spruit-Tayler dynamo in radiative envelope (zone k=300)
k = 300
d_dsi, d_sh, d_ssi, d_es, d_gsf, d_st, nu_st, b_phi, b_r = (
    star.eval_rotational_mixing_zone_jax(
        r=jnp.float64(model_15m.r[k]),
        hp=jnp.float64(model_15m.scale_height[k]),
        grav=jnp.float64(model_15m.grav[k]),
        rho=jnp.float64(model_15m.rho[k]),
        t=jnp.float64(model_15m.t[k]),
        cp=jnp.float64(model_15m.cp[k]),
        opacity=jnp.float64(model_15m.opacity[k]),
        chi_t=jnp.float64(model_15m.chi_t[k]),
        chi_rho=jnp.float64(model_15m.chi_rho[k]),
        grada=jnp.float64(model_15m.grada[k]),
        gradt=jnp.float64(model_15m.gradt[k]),
        brunt_n2_t=jnp.float64(max(float(model_15m.brunt_n2[k]), 1.0e-6)),
        brunt_n2_mu=jnp.float64(2.0e-7),
        omega=jnp.float64(5.0e-5),
        dln_omega_dln_r=jnp.float64(-0.6),
        abar=jnp.float64(1.3),
        zbar=jnp.float64(1.1),
    )
)
print(
    f"Zone {k}: D_ES={float(d_es):.3e} cm^2/s, nu_ST={float(nu_st):.3e} cm^2/s, "
    f"B_phi={float(b_phi):.2f} G, B_r={float(b_r):.4f} G"
)

# Populate full-star rotational mixing coefficients and diffuse Omega over dt = 1e3 yr
d_rot_profile = model_15m.set_rotation_mixing_info()
omega_init = np.linspace(4.0e-5, 8.0e-5, model_15m.nz)
nu_omega = np.full(model_15m.nz, 1.0e10)
model_15m.run_omega_mix(dt=1.0e3 * 3.15576e7, omega=omega_init, nu_omega=nu_omega, n_substeps=4)
print(f"Updated surface Omega = {model_15m.omega[0]:.4e} rad/s, center Omega = {model_15m.omega[-1]:.4e} rad/s")
```

---

## 3. Radiation-Driven Stellar Winds (`Dutch`, `Vink`, `Nugis & Lamers`, `de Jager`)

For hot O/B stars, Wolf-Rayet stars, and cool red supergiants, `FlexMESA` provides both the `StarModel.apply_wind_and_adjust_mass` step operator (which strips mass over a timestep $\Delta t$ and rescales shell masses) and the differentiable `star.eval_winds_jax` kernel (which evaluates Reimers 1975, Blöcker 1995, de Jager 1988, Vink et al. 2001 with its bistability jump around $T_{\mathrm{eff}} \approx 25{,}000\,\mathrm{K}$, Nugis & Lamers 2000 Wolf-Rayet winds, and the composite `Dutch` recipe):

```python
import jax
import jax.numpy as jnp
from flexmesa import star

model_15m = star.StarModel.from_zams(initial_mass=15.0, initial_z=0.02)

# Evaluate all wind prescriptions simultaneously in JAX for the 15 Msun star
winds = model_15m.eval_winds_jax(eta=1.0)
for key in ("w_vink", "w_de_jager", "w_van_loon", "w_dutch", "w_reimers", "w_blocker"):
    print(f"  {key:14s}: dM/dt = {float(winds[key]):.4e} Msun/yr")

# Differentiate the composite Dutch wind mass-loss rate with respect to surface luminosity
l_surf = float(model_15m.lum[0])
m_surf = float(model_15m.m[0])
t_surf = float(model_15m.teff)
dmdot_dl = jax.grad(
    lambda lum: star.eval_winds_jax(lum, m_surf, t_surf, eta=1.0)["w_dutch"]
)(jnp.float64(l_surf))
print(f"Sensitivity d(mdot_Dutch)/dL = {float(dmdot_dl):.4e} (Msun/yr)/(erg/s)")

# Apply the Dutch wind over a 10,000 yr step (dt in seconds, mdot returned in g/s)
mdot_g_s = model_15m.apply_wind_and_adjust_mass(
    scheme="Dutch", eta=1.0, dt=1.0e4 * 3.15576e7
)
mdot_msun_yr = mdot_g_s / 1.98840987e33 * 3.15576e7
print(f"Applied Dutch wind: dM/dt = {mdot_msun_yr:.4e} Msun/yr, new M = {model_15m.mstar / 1.98840987e33:.6f} Msun")
```

---

## 4. 1D Lagrangian HLLC Riemann Hydrodynamics

When stellar envelopes experience rapid dynamical expansion, pulsations, or shock waves, hydrostatic equilibrium ($\partial P / \partial m = -G m / 4\pi r^4$) must be replaced by the 1D Lagrangian momentum equation with an upwind Riemann solver at cell interfaces. `FlexMESA` implements the Toro (2009) and Paxton et al. (2015, MESA III) 1D Lagrangian Harten-Lax-van Leer-Contact (`HLLC`) solver across both Safe Rust (`_rust.star_eval_hllc_face`, `_rust.star_eval_hllc_surface_face`, `model.eval_riemann_profile`) and JAX (`star.eval_hllc_face_jax`, `star.eval_hllc_surface_face_jax`, `star.eval_riemann_momentum_residual_jax`).

Given left (inner, $k$) and right (outer, $k-1$) states $(P_L, \rho_L, c_{s,L}, u_L, dm_L)$ and $(P_R, \rho_R, c_{s,R}, u_R, dm_R)$, `eval_hllc_face` computes the 7-tuple `(p_l_star, p_r_star, s_l, s_r, u_face, p_face, ke_flux)` containing the wave-adjusted pressures, Lagrangian shock speeds $(S_L, S_R)$, contact wave velocity $u^\star$ (`index 4`), contact pressure $P^\star$ (`index 5`), and mechanical energy flux $A_{\mathrm{face}} P^\star u^\star$:

```python
import jax
import jax.numpy as jnp
import numpy as np
from flexmesa import _rust, constants, star

model_15m = star.StarModel.from_zams(initial_mass=15.0, initial_z=0.02)

# 1. Evaluate the full-star 1D HLLC Riemann face profile at rest (u = 0)
u_zero = np.zeros(model_15m.nz, dtype=np.float64)
face_states, mom_res = model_15m.eval_riemann_profile(
    u_cell=u_zero, u_old_cell=u_zero, p_surf_bc=float(model_15m.peos[0]), parallel=True
)
print(
    f"Full-star HLLC profile ({model_15m.nz} faces): "
    f"surface P* = {face_states[0, 5]:.4e} dyn/cm^2, center P* = {face_states[-1, 5]:.4e} dyn/cm^2"
)

# 2. Evaluate a single interior shock interface in Rust and differentiate P* w.r.t. u_L in JAX
k = 100
p_l, rho_l, cs_l, dm_l = (
    float(model_15m.peos[k]),
    float(model_15m.rho[k]),
    float(model_15m.csound[k]),
    float(model_15m.dm[k]),
)
p_r, rho_r, cs_r, dm_r = (
    float(model_15m.peos[k - 1]),
    float(model_15m.rho[k - 1]),
    float(model_15m.csound[k - 1]),
    float(model_15m.dm[k - 1]),
)
r_f, m_f = float(model_15m.r[k]), float(model_15m.m[k])

# Outward velocity jump u_L = 2e6 cm/s vs u_R = 0
hllc_rust = _rust.star_eval_hllc_face(
    p_l, rho_l, cs_l, 2.0e6, dm_l,
    p_r, rho_r, cs_r, 0.0, dm_r,
    r_f, m_f, constants.STANDARD_CGRAV,
)

# Index 4 is u_face (u*), Index 5 is p_face (P*)
dp_star_du_l = jax.grad(
    lambda u_in: star.eval_hllc_face_jax(
        jnp.float64(p_l), jnp.float64(rho_l), jnp.float64(cs_l), u_in, jnp.float64(dm_l),
        jnp.float64(p_r), jnp.float64(rho_r), jnp.float64(cs_r), jnp.float64(0.0), jnp.float64(dm_r),
        jnp.float64(r_f), jnp.float64(m_f), jnp.float64(constants.STANDARD_CGRAV),
    )[5]
)(jnp.float64(2.0e6))

# Surface boundary HLLC Riemann state
hllc_surf = star.eval_hllc_surface_face_jax(
    jnp.float64(model_15m.peos[0]),
    jnp.float64(model_15m.rho[0]),
    jnp.float64(model_15m.csound[0]),
    jnp.float64(1.0e5),
    jnp.float64(model_15m.dm[0]),
    jnp.float64(model_15m.r[0]),
    jnp.float64(model_15m.m[0]),
    jnp.float64(constants.STANDARD_CGRAV),
    jnp.float64(0.8 * model_15m.peos[0]),
)

print(
    f"Zone {k} HLLC shock: u* = {hllc_rust[4]:.4e} cm/s, P* = {hllc_rust[5]:.4e} dyn/cm^2, "
    f"dP*/du_L = {float(dp_star_du_l):.4e} (dyn/cm^2)/(cm/s), surface u* = {float(hllc_surf[4]):.4e} cm/s"
)
```
