# Tutorial 3: White Dwarf Cooling, Crystallization & Atomic Diffusion

As a carbon-oxygen (`C/O`) or oxygen-neon (`O/Ne`) white dwarf cools over gigayear timescales, its degenerate interior transitions from a strongly coupled Coulomb liquid into a body-centered cubic (`bcc`) solid lattice while trace heavy isotopes sink gravitationally toward the core. These two processes release gravitational and latent energy that delays white dwarf cooling by up to $1\text{ to }2\,\mathrm{Gyr}$. This tutorial demonstrates how to evaluate the `Skye` and Potekhin-Chabrier (`PC`) degenerate equations of state in `flexmesa.eos`, model first-order crystallization and Blouin et al. (2020) / Medin & Cumming (2010) phase separation in `flexmesa.star`, and solve the Burgers (1969) multi-fluid atomic diffusion, Caplan et al. (2022) one-component plasma (`OCP`) self-diffusion, and radiative levitation equations.

---

## 1. `Skye` and `PC` Equations of State Across the Crystallization Transition

In a dense white dwarf core ($\rho \sim 10^6\,\mathrm{g\,cm^{-3}}$, $T \sim 3 \times 10^6\text{ to }2 \times 10^7\,\mathrm{K}$), the pressure is dominated by degenerate relativistic electrons while the thermal heat capacity and latent heat are governed by the multicomponent Coulomb ion plasma. The dimensionless ion coupling parameter is:

$$
\Gamma_i = \frac{Z_i^2 e^2}{a_i k_{\mathrm{B}} T}, \qquad a_i = \left(\frac{3 Z_i}{4\pi n_e}\right)^{1/3}
$$

When the mean coupling parameter $\bar{\Gamma}$ approaches $\Gamma_{\mathrm{crit}} \approx 175$, `Skye` (Jermyn et al. 2021) determines the liquid-to-solid transition thermodynamically by minimizing the Helmholtz free energy $F_{\mathrm{liquid}}(\rho, T) - F_{\mathrm{solid}}(\rho, T)$ with linear mixing rules (`skye_solid_mixing_rule="PC"` or `"Ogata"`), whereas `PC` (Potekhin & Chabrier 2010) transitions across a prescribed interval $\Gamma \in [170, 180]$.

```python
import jax
import jax.numpy as jnp
import numpy as np
from flexmesa import _rust, chem, eos

# Define a 50/50 Carbon-12 / Oxygen-16 white dwarf core mixture
c12 = chem.get_nuclide_info("c12").chem_id
o16 = chem.get_nuclide_info("o16").chem_id
chem_ids = [c12, o16]
xa_co = np.array([0.50, 0.50], dtype=np.float64)
comp = chem.get_composition_info(chem_ids, xa_co)

rho_wd = 2.0e6  # g/cm^3
for temp_wd in (1.5e7, 6.0e6, 2.5e6):
    out_skye = eos.eval_skye_eos(
        rho=rho_wd,
        temp=temp_wd,
        abar=float(comp.abar),
        zbar=float(comp.zbar),
        chem_id=chem_ids,
        xa=xa_co,
        skye_solid_mixing_rule="PC",
    )
    out_pc = eos.eval_pc_eos(
        rho=rho_wd,
        temp=temp_wd,
        abar=float(comp.abar),
        zbar=float(comp.zbar),
        chem_id=chem_ids,
        xa=xa_co,
    )
    # Direct PyO3 Rust binding `_rust.eos_eval_skye` returns (res, d_dlnd, d_dln_t)
    res_rust, _, _ = _rust.eos_eval_skye(
        rho_wd, temp_wd, float(comp.abar), float(comp.zbar), chem_ids, xa_co
    )
    print(
        f"T = {temp_wd:.2e} K: Skye ln(Pgas) = {float(out_skye.res[0]):.6f}, "
        f"PC ln(Pgas) = {float(out_pc.res[0]):.6f}, "
        f"Gamma_1(Skye) = {float(res_rust[eos.I_GAMMA1]):.5f}, "
        f"phase(Skye) = {float(res_rust[eos.I_PHASE]):.3f}"
    )

# Differentiate Skye specific internal energy ln(E) w.r.t. temperature in JAX
d_lne_dt = jax.grad(
    lambda t_in: eos.eval_skye_eos(
        jnp.float64(rho_wd),
        t_in,
        jnp.float64(comp.abar),
        jnp.float64(comp.zbar),
        chem_ids,
        jnp.asarray(xa_co),
        backend="jax",
    ).res[1]
)(jnp.float64(5.0e6))
print(f"Skye JAX sensitivity d(ln E)/dT at T=5e6 K = {float(d_lne_dt):.4e} K^-1")
```

![EOS phase diagram and Skye Coulomb crystallization transition](../assets/figures/eos_helm_skye_phase_diagram.png)

*Figure 1: Left: The 7-component equation-of-state partition across the $(\log_{10} \rho, \log_{10} T)$ plane, showing the `Skye`/`PC` high-density degenerate plasma domain at $\log_{10} \rho \gtrsim 3$. Right: Variation of the adiabatic exponent $\Gamma_1$ across the `Skye` Coulomb crystallization boundary ($\Gamma_{\mathrm{crit}} \approx 175$) for a carbon-oxygen mixture.*

---

## 2. Crystallization Phase Separation and Sedimentation Heating

When a liquid binary mixture of ${}^{12}\mathrm{C}$ ($Z = 6$) and ${}^{16}\mathrm{O}$ ($Z = 8$) freezes, the solid crystal lattice is enriched in the higher-charge species (${}^{16}\mathrm{O}$) relative to the coexisting liquid by a composition shift $\Delta X_{\mathrm{O}}(X_{\mathrm{O}})$. Because the newly frozen solid is denser and oxygen-rich, the residual carbon-enriched liquid at the crystallization front is buoyant and mixes upward via Rayleigh-Taylor overturn (`model.do_phase_separation`).

In `FlexMESA`, `_rust.star_eval_phase_separation_shift` and `star.eval_phase_separation_shift_jax` evaluate the Blouin et al. (2020) `C/O` (`is_one_mixture=False`) and Medin & Cumming (2010) `O/Ne` (`is_one_mixture=True`) phase-diagram shifts $\Delta X_{\mathrm{heavy}}$, while `_rust.star_eval_wd_sedimentation_heating` computes the local gravitational energy release $\epsilon_{\mathrm{sed}}$ from heavy-ion sedimentation:

```python
import jax
import jax.numpy as jnp
from flexmesa import _rust, star

# 1. Evaluate C/O and O/Ne crystallization phase-diagram enrichment shifts
for x_o in (0.25, 0.50, 0.75):
    dx_co_rust = _rust.star_eval_phase_separation_shift(x_o, is_one_mixture=False)
    dx_one_rust = _rust.star_eval_phase_separation_shift(x_o, is_one_mixture=True)
    ddx_dx = jax.grad(
        lambda x: star.eval_phase_separation_shift_jax(x, is_one_mixture=False)
    )(jnp.float64(x_o))
    print(
        f"X_heavy = {x_o:.2f}: dX(C/O) = {dx_co_rust:.5f} (d(dX)/dX = {float(ddx_dx):.4f}), "
        f"dX(O/Ne) = {dx_one_rust:.5f}"
    )

# 2. Apply phase separation across a stellar model profile
model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
eps_phase = model.do_phase_separation(
    dt=1.0e5 * 3.15576e7, gamma_crystallization=175.0, is_one_mixture=False
)
print(f"Evaluated phase separation profile across {len(eps_phase)} zones.")
```

---

## 3. Burgers Multi-Fluid Atomic Diffusion, Caplan OCP & Radiative Levitation

In white dwarf envelopes and radiative stellar interiors, gravitational settling, thermal diffusion, electric fields, and selective radiative acceleration separate chemical species over diffusion timescales. `FlexMESA` solves the Burgers (1969) multi-fluid resistance equations in both Safe Rust (`_rust.star_eval_burgers_velocities`) and JAX (`star.eval_burgers_velocities_jax`), coupled with Stanton & Murillo (2016) screened collision integrals, Caplan et al. (2022) strongly coupled one-component plasma (`OCP`) self-diffusion (`star.eval_caplan_ocp_dstar_jax`), and monochromatic radiative levitation accelerations (`_rust.star_eval_radiative_levitation_accel`).

```python
import jax
import jax.numpy as jnp
import numpy as np
from flexmesa import _rust, star

# 1. Solve Burgers multi-fluid diffusion velocities for a C12 / O16 / Ne22 mixture
a_ions = np.array([12.0, 16.0, 22.0], dtype=np.float64)
z_ions = np.array([6.0, 8.0, 10.0], dtype=np.float64)
x_mass = np.array([0.49, 0.49, 0.02], dtype=np.float64)
rho_env = 1.0e4     # g/cm^3
t_env = 1.0e7       # K
dlnt_dr = -1.0e-8   # cm^-1
g_grav = 1.0e8      # cm/s^2

v_gt_rust, e_field_rust, g_eff_rust = _rust.star_eval_burgers_velocities(
    a_ions, z_ions, x_mass, rho_env, t_env, dlnt_dr, g_grav
)

for name, v in zip(["C12", "O16", "Ne22"], v_gt_rust):
    print(f"  {name}: drift velocity v_diff = {v:.4e} cm/s")

# 2. Compute gravitational sedimentation heating from 22Ne settling
eps_sed = _rust.star_eval_wd_sedimentation_heating(
    a_ions, z_ions, x_mass, v_gt_rust, 13.8, 6.9, g_grav
)
print(f"22Ne sedimentation heating rate eps_sed = {eps_sed:.4e} erg/g/s")

# 3. Evaluate Caplan et al. (2022) OCP dimensionless diffusion D* and radiative levitation g_rad
d_star = star.eval_caplan_ocp_dstar_jax(jnp.float64(120.0), jnp.float64(0.6))
ddstar_dgamma = jax.grad(
    lambda g: star.eval_caplan_ocp_dstar_jax(g, jnp.float64(0.6))
)(jnp.float64(120.0))
g_rad_fe = _rust.star_eval_radiative_levitation_accel(
    kappa_species=3.2, lum_face=1.0e33, r_face=1.0e9
)
print(
    f"Caplan OCP D*(Gamma=120, kappa=0.6) = {float(d_star):.5f} "
    f"(dD*/dGamma = {float(ddstar_dgamma):.4e}), g_rad = {g_rad_fe:.3f} cm/s^2"
)

# 4. Advance full-star element diffusion on a StarModel
model_1m = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
v_surf, e_surf = model_1m.do_element_diffusion(
    dt=1.0e6 * 3.15576e7, scheme="stanton_murillo"
)
print(f"Full-star diffusion step completed: {len(v_surf)} zones updated.")
```
