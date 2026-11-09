# User Guide: Publication Diagnostic Plotting (`flexmesa.plotting`)

Fortran MESA's `PGSTAR` subsystem relies on `PGPLOT` and an interactive X11 server to render diagnostic windows during a run. In `FlexMESA`, the `flexmesa.plotting` and `flexmesa.colors` subpackages provide headless, scriptable `matplotlib` routines styled for astronomical journals (Computer Modern typography, inward ticks on all four axes, frameless legends, and colorblind-safe Okabe-Ito palettes). Every plotting function returns `(fig, ax)` (or `(fig, axes)` for multi-panel layouts) so you can customize annotations or save 300 dpi PNG and vector PDF figures directly.

---

## 1. Configuring the Journal Plot Style (`use_mesa_plot_style`, `figsize`)

Calling `plotting.use_mesa_plot_style()` applies `plotting.mesa_rc_params()` globally, while `plotting.figsize(layout="single" | "double" | "grid")` returns standard journal column dimensions in inches:

```python
from flexmesa import plotting

plotting.use_mesa_plot_style()
rc = plotting.mesa_rc_params()
print(
    f"Single-column figsize = {plotting.figsize('single')}, "
    f"Double-column figsize = {plotting.figsize('double')}, "
    f"xtick.direction = {rc['xtick.direction']!r}"
)
```

---

## 2. Stellar Structure & Evolutionary Track Plots

`flexmesa.plotting` provides three core functions for single-star structure and evolutionary tracks:

1. **Hertzsprung-Russell Diagram (`plotting.plot_hr_diagram`)**: Plots $\log_{10}(L/L_\odot)$ against inverted $\log_{10}(T_{\mathrm{eff}}/\mathrm{K})$, optionally coloring points along the track by a third scalar such as central hydrogen fraction $X_c$ (`color_val`, `vmin`, `vmax`).
2. **Central Temperature-Density Diagram (`plotting.plot_tc_rhoc_diagram`)**: Plots $\log_{10}(T_c/\mathrm{K})$ against $\log_{10}(\rho_c/\mathrm{g\,cm^{-3}})$ with reference hydrogen, helium, and carbon ignition curves (`show_burn_lines=True`).
3. **Kippenhahn Convective & Burning History (`plotting.plot_kippenhahn`)**: Shades convective, overshooting, semiconvective, and thermohaline zones alongside nuclear energy generation $\log_{10}(\epsilon_{\mathrm{nuc}})$ in enclosed mass coordinates $m/M_\odot$ versus stellar age (`Myr`).

```python
import matplotlib.pyplot as plt
import numpy as np
from flexmesa import plotting, star

model = star.StarModel.from_zams(initial_mass=2.0, initial_z=0.02)
n_steps, n_grid = 5, 80
q_grid = np.linspace(0.01, 1.0, n_grid)

ages_myr, log_teff, log_l, log_rhoc, log_tc, xc = [], [], [], [], [], []
mass_grid = q_grid * 2.0
mix_grid = np.zeros((n_grid, n_steps))
eps_grid = np.zeros((n_grid, n_steps))

for i in range(n_steps):
    if i > 0:
        model.evolve_step(do_remesh=False, use_bcyclic=True)
    row = model.history_row()
    ages_myr.append(row.star_age * 1.0e-6)
    log_teff.append(row.log_teff)
    log_l.append(row.log_l)
    log_rhoc.append(row.log_center_rho)
    log_tc.append(row.log_center_t)
    xc.append(row.center_h1)

    q_inc = np.asarray(model.q)[::-1]
    mix_grid[:, i] = np.interp(q_grid, q_inc, np.asarray(model.mixing_type)[::-1])
    eps_grid[:, i] = np.log10(
        np.maximum(np.interp(q_grid, q_inc, np.asarray(model.eps_nuc)[::-1]), 1.0)
    )

fig1, _ = plotting.plot_hr_diagram(np.array(log_teff), np.array(log_l), color_val=np.array(xc))
plt.close(fig1)

fig2, _ = plotting.plot_tc_rhoc_diagram(np.array(log_rhoc), np.array(log_tc))
plt.close(fig2)

fig3, _ = plotting.plot_kippenhahn(
    np.array(ages_myr), mass_grid, mix_grid, eps_nuc_grid=eps_grid
)
plt.close(fig3)
print("Rendered HR, Tc-rhoc, and Kippenhahn diagnostic figures.")
```

---

## 3. Asteroseismic, Binary & Synthetic Photometry Figures

For asteroseismology, binary evolution, and synthetic photometry, `flexmesa.plotting` and `flexmesa.colors` provide four specialized visualization workflows:

- **Propagation Diagram (`plotting.plot_propagation_diagram`)**: Plots the Brunt-Väisälä buoyancy frequency $N(r)$ and Lamb acoustic frequencies $S_\ell(r) = \sqrt{\ell(\ell+1)}\,c_s(r)/r$ for $\ell \in \{1, 2\}$ in microhertz ($\mu\mathrm{Hz}$) against fractional radius $r/R_\star$.
- **Echelle Diagram (`plotting.plot_echelle_diagram`)**: Plots mode frequencies $\nu_{n\ell}$ against reduced frequency $\nu_{n\ell} \bmod \Delta\nu$, overlaying observed error bars, uncorrected adiabatic model frequencies, and surface-corrected model frequencies.
- **Frequency Separation Ratios (`plotting.plot_separation_ratios`)**: Plots the 5-point core-probing ratios $r_{010}(\nu)$ and $r_{02}(\nu)$ with observational error bars.
- **Binary Evolution Multi-Panel (`plotting.plot_binary_evolution`)**: Renders orbital period $P_{\mathrm{orb}}$, mass-transfer rate $\log_{10}|\dot{M}_{\mathrm{RLOF}}|$, and eccentricity $e(t)$ versus binary age.

```python
import matplotlib.pyplot as plt
import numpy as np
from flexmesa import astero, colors, plotting, star

model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
r_cm = np.asarray(model.r)
cs_cm_s = np.asarray(model.csound)

# 1. Asteroseismic propagation, echelle, and separation-ratio plots
fig_prop, _ = plotting.plot_propagation_diagram(
    r_cm / r_cm[0], np.asarray(model.brunt_n2), cs_cm_s, r_cm, nu_max_uhz=3090.0
)
plt.close(fig_prop)

orders = np.arange(17, 23, dtype=np.float64)
l0_obs = 135.0 * (orders + 1.44)
l1_obs = 135.0 * (orders + 0.50 + 1.42)
l2_obs = l0_obs - 9.0
sigs = np.full_like(l0_obs, 0.2)

fig_ech, _ = plotting.plot_echelle_diagram(
    l0_obs, sigs, delta_nu=135.0, model_freqs_uncorr=l0_obs + 3.0, model_freqs_corr=l0_obs + 0.1
)
plt.close(fig_ech)

r010 = astero.eval_ratios_r01_r10(l0_obs, sigs, l1_obs, sigs)
r02, sig_r02 = astero.eval_ratios_r02(l0_obs, sigs, l1_obs, sigs, l2_obs, sigs)
fig_rat, _ = plotting.plot_separation_ratios(
    l0_obs[1:-1],
    r010["r01"],
    r010["sig_r01"],
    r010_model=r010["r01"],
    freqs_r02=l0_obs[1:],
    r02_obs=r02,
    r02_sigmas=sig_r02,
    r02_model=r02,
)
plt.close(fig_rat)

# 2. Synthetic Kurucz SED and Johnson UBVRIJM photometry
wave_aa, flux_cgs = colors.eval_sed(
    teff=float(model.teff),
    log_g=4.44,
    metallicity=0.0,
    r=float(r_cm[0]),
)
mags_vega = colors.eval_colors(
    teff=float(model.teff),
    log_g=4.44,
    metallicity=0.0,
    r=float(r_cm[0]),
    mag_system="Vega",
)
print(
    f"Kurucz SED ({len(wave_aa)} wavelength bins): "
    f"Johnson V = {float(mags_vega[2]):.3f} mag, B - V = {float(mags_vega[1] - mags_vega[2]):.3f} mag"
)
```

![Synthetic Kurucz SED and Johnson filter curves](../assets/figures/colors_kurucz_sed_johnson.png)

*Figure 1: Kurucz (2003) synthetic spectral energy distributions across effective temperatures $T_{\mathrm{eff}} \in [3500, 10000]\,\mathrm{K}$ overlaid with the normalized Johnson-Cousins $UBVRI$ filter transmission curves in `flexmesa.colors`.*
