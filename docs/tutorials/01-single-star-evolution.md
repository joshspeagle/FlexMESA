# Tutorial 1: Single-Star Evolution from Pre-MS and ZAMS

This tutorial walks through initializing and evolving single-star models in `flexmesa.star` for a solar-mass star ($1\,M_\odot$, radiative core and convective envelope) and an intermediate-mass star ($4\,M_\odot$, convective core and radiative envelope). Along the way, we inspect global evolutionary tracks (`HistoryRow`), examine 1D Lagrangian radial profiles (`rho`, `t`, `peos`, `eps_nuc`, `gradr`, `grada`, `conv_vel`), and generate publication-styled Hertzsprung-Russell (`HR`), central temperature-density ($\log_{10} T_c$ versus $\log_{10} \rho_c$), and Kippenhahn convective history diagrams using `flexmesa.plotting`.

---

## 1. Initializing Pre-Main-Sequence and ZAMS Models

A 1D stellar evolution calculation begins from a hydrostatic starting model on a discrete Lagrangian mass grid $\{dq_k\}_{k=0}^{N_{\mathrm{zones}}-1}$ ordered from the photosphere ($k = 0$, $q = 1$) to the center ($k = N_{\mathrm{zones}} - 1$, $q \to 0$). `FlexMESA` provides two primary constructors on `StarModel`:

- **Pre-Main-Sequence Polytrope (`StarModel.from_pre_ms`)**: Integrates an $n = 1.5$ convective polytrope parameterized by total mass `initial_mass` ($M_\odot$), metallicity `initial_z`, and initial central temperature `t_c` ($\mathrm{K}$, typically $3 \times 10^5$ to $9 \times 10^5\,\mathrm{K}$ prior to deuterium or hydrogen ignition), and relaxes it to hydrostatic and thermal equilibrium via the coupled Henyey solver (`eval_hydro=True`).
- **Zero-Age Main Sequence Interpolation (`StarModel.from_zams`)**: Interpolates log-mass profiles across the built-in 8-mass Zero-Age Main Sequence (`ZAMS`) anchor catalog ($M_\star \in [0.51, 15.0]\,M_\odot$ at $Z = 0.02$) and converges the initial thermal structure where core hydrogen burning ($\epsilon_{\mathrm{nuc}}$) balances surface radiative luminosity ($L_{\mathrm{surf}}$).

```python
from flexmesa import constants, star

# 1. Construct a 1 Msun pre-main-sequence model at Tc = 9e5 K
pre_ms = star.StarModel.from_pre_ms(
    initial_mass=1.0,
    initial_z=0.02,
    t_c=9.0e5,
    d_log10_p=0.02,
    eval_hydro=True,
)
pre_row = pre_ms.history_row()
print(
    f"Pre-MS 1 Msun: nz={pre_ms.nz}, log10(Teff)={pre_row.log_teff:.4f}, "
    f"log10(L/Lsun)={pre_row.log_l:.4f}, log10(Tc)={pre_row.log_center_t:.4f}"
)

# 2. Construct 1 Msun and 4 Msun ZAMS models
zams_1m = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
zams_4m = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)

for label, mdl in [("1 Msun ZAMS", zams_1m), ("4 Msun ZAMS", zams_4m)]:
    row = mdl.history_row()
    print(
        f"{label}: nz={mdl.nz}, R={mdl.r[0] / constants.RSUN:.3f} Rsun, "
        f"log10(L/Lsun)={row.log_l:.4f}, log10(Tc)={row.log_center_t:.4f}"
    )
```

---

## 2. Evolving Stellar Structure and Recording Evolutionary History

Advancing a model in time with `model.evolve` (or step-by-step via `model.evolve_step`) couples four numerical operations at each timestep $\Delta t$:

1. **Adaptive Mesh Refinement (`do_remesh=True`)**: Evaluates the scalar mesh spacing gradient function $g(k)$ across pressure, temperature, radius, and nuclear burning gradients, splitting over-spaced zones and merging under-spaced adjacent zones via mass-conserving piecewise monotonic cubic remapping (`interp_q`).
2. **Convective Mixing & Overshooting (`update_mixing`)**: Evaluates Mixing-Length Theory (`MLT`) on all interior faces, locates Schwarzschild or Ledoux convective boundaries ($\nabla_{\mathrm{rad}} = \nabla_{\mathrm{ad}}$), and applies exponential diffusive overshooting beyond convective cores and envelopes.
3. **Coupled Block-Tridiagonal Henyey Solve (`use_bcyclic=True`)**: Solves the nonlinear 1D Lagrangian continuity, momentum, radius, energy, temperature, and composition equations using damped Newton-Raphson iterations and the parallel Hirshman et al. (2010) odd-even cyclic reduction (`bcyclic`) solver.
4. **Söderlind & Wang (2006) `H211` Digital Filter Timestep Control**: Regulates $\Delta t_{n+1}$ so the relative state variation `varcontrol` tracks `varcontrol_target` smoothly without step-size oscillations.

```python
from flexmesa import star

model_1m = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
model_1m.set_evolve_controls(mesh_delta_coeff=1.0, varcontrol_target=1.0e-4)

history_1m = [model_1m.history_row()]
for _ in range(12):
    report = model_1m.evolve_step(
        do_remesh=True,
        op_split_burn=False,
        use_bcyclic=True,
    )
    history_1m.append(model_1m.history_row())

last = history_1m[-1]
print(
    f"Final step {last.model_number}: age={last.star_age:.3e} yr, "
    f"center H1={last.center_h1:.6f}, He4={last.center_he4:.6f}, "
    f"Newton iters={report.iterations}"
)
```

---

## 3. Inspecting Radial Structure and Convective Core vs. Envelope Profiles

Because `1 Msun` stars burn hydrogen primarily via the $pp$-chain (which scales gently with temperature as $\epsilon_{pp} \propto T^4$), radiative diffusion easily transports the core luminosity outward, leaving a radiative core ($\nabla_{\mathrm{rad}} < \nabla_{\mathrm{ad}}$) and a steep sub-photospheric convective envelope driven by high $\mathrm{H}^-$ and partial-ionization opacities. Conversely, `4 Msun` stars burn hydrogen via the temperature-sensitive CNO bi-cycle ($\epsilon_{\mathrm{CNO}} \propto T^{16}$), concentrating nuclear energy generation in the innermost shells and driving a vigorous convective core ($\nabla_{\mathrm{rad}} > \nabla_{\mathrm{ad}}$, `conv_vel > 0`) surrounded by a radiative envelope.

All 1D profile arrays on `StarModel` follow the **Index and Memory-Order Contract**: 0-based C-contiguous `float64` arrays ordered from the surface (`k = 0`) to the center (`k = nz - 1`).

```python
import numpy as np
from flexmesa import star

zams_1m = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
zams_4m = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)

for label, mdl in [("1 Msun", zams_1m), ("4 Msun", zams_4m)]:
    q_prof = np.asarray(mdl.q)
    gradr = np.asarray(mdl.gradr)
    grada = np.asarray(mdl.grada)
    conv_vel = np.asarray(mdl.conv_vel)
    eps_nuc = np.asarray(mdl.eps_nuc)

    # Identify convective zones where gradr > grada
    conv_mask = gradr > grada
    q_conv_max = float(q_prof[conv_mask].max()) if np.any(conv_mask) else 0.0
    q_conv_min = float(q_prof[conv_mask].min()) if np.any(conv_mask) else 0.0
    print(
        f"{label}: center T={mdl.t[-1]:.3e} K, center rho={mdl.rho[-1]:.2f} g/cm^3, "
        f"center P={mdl.peos[-1]:.3e} dyn/cm^2, peak eps_nuc={eps_nuc.max():.3e} erg/g/s"
    )
    print(
        f"  Convective mass-fraction span q in [{q_conv_min:.4f}, {q_conv_max:.4f}], "
        f"max v_conv={conv_vel.max():.2e} cm/s"
    )
```

![Internal convective structure of a 4 Msun ZAMS star and 8-mass ZAMS Hertzsprung-Russell diagram](../assets/figures/wave4_star_structure_evolution.png)

*Figure 1: Left: Logarithmic temperature gradients ($\nabla_{\mathrm{rad}}$ in solid blue, $\nabla_{\mathrm{ad}}$ in dashed orange, and actual Mixing-Length Theory $\nabla_T$ in dash-dotted green) as a function of enclosed mass fraction $m/M_\star$ for a $4\,M_\odot$, $Z = 0.02$ Zero-Age Main Sequence star, showing the convective core boundary at $m/M_\star \approx 0.24$. Right: The Zero-Age Main Sequence locus across the 8-mass anchor catalog ($0.51$ to $15.0\,M_\odot$) in the Hertzsprung-Russell diagram alongside a $4\,M_\odot$ core hydrogen-burning track.*

---

## 4. Generating HR, Central $T_c$ vs. $\rho_c$, and Kippenhahn Diagrams

The `flexmesa.plotting` subpackage provides headless, publication-styled `matplotlib` routines that replace legacy `PGSTAR` X11 windows:

- `plotting.plot_hr_diagram(log_teff, log_l, color_val=...)`: Plots the evolutionary track in $(\log_{10} T_{\mathrm{eff}}, \log_{10} L/L_\odot)$ with inverted effective temperature axis and optional colorbar encoding central hydrogen mass fraction $X_c$.
- `plotting.plot_tc_rhoc_diagram(log_rhoc, log_tc, show_burn_lines=True)`: Plots the central thermodynamic trajectory $(\log_{10} \rho_c, \log_{10} T_c)$ against hydrogen, helium, and carbon ignition thresholds.
- `plotting.plot_kippenhahn(age_myr, mass_grid_msun, mix_type_grid, eps_nuc_grid=..., total_mass_msun=...)`: Renders the time evolution of convective, overshooting, semiconvective, and nuclear-burning regions in enclosed mass coordinates $m/M_\odot$.

```python
import matplotlib.pyplot as plt
import numpy as np
from flexmesa import plotting, star

model_4m = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
n_steps = 8
n_grid = 120
q_uniform = np.linspace(0.01, 1.0, n_grid)

ages_myr = []
log_teff, log_l, log_rhoc, log_tc, center_h1 = [], [], [], [], []
mass_grid = q_uniform * 4.0
mix_grid = np.zeros((n_grid, n_steps))
eps_grid = np.zeros((n_grid, n_steps))

for step_idx in range(n_steps):
    if step_idx > 0:
        model_4m.evolve_step(do_remesh=False, use_bcyclic=True)
    row = model_4m.history_row()
    ages_myr.append(row.star_age * 1.0e-6)
    log_teff.append(row.log_teff)
    log_l.append(row.log_l)
    log_rhoc.append(row.log_center_rho)
    log_tc.append(row.log_center_t)
    center_h1.append(row.center_h1)

    # Interpolate radial profiles onto a uniform mass-coordinate grid for Kippenhahn plotting
    q_descending = np.asarray(model_4m.q)[::-1]
    mix_grid[:, step_idx] = np.interp(q_uniform, q_descending, np.asarray(model_4m.mixing_type)[::-1])
    eps_grid[:, step_idx] = np.log10(
        np.maximum(np.interp(q_uniform, q_descending, np.asarray(model_4m.eps_nuc)[::-1]), 1.0)
    )

fig_hr, ax_hr = plotting.plot_hr_diagram(
    np.array(log_teff),
    np.array(log_l),
    color_val=np.array(center_h1),
)
plt.close(fig_hr)

fig_tc, ax_tc = plotting.plot_tc_rhoc_diagram(
    np.array(log_rhoc),
    np.array(log_tc),
    label="4 Msun ZAMS Track",
)
plt.close(fig_tc)

fig_kipp, ax_kipp = plotting.plot_kippenhahn(
    np.array(ages_myr),
    mass_grid,
    mix_grid,
    eps_nuc_grid=eps_grid,
    total_mass_msun=np.full(n_steps, 4.0),
)
plt.close(fig_kipp)
print(f"Generated HR, Tc-rhoc, and Kippenhahn figures across {n_steps} steps.")
```
