# Tutorial 5: Asteroseismology, Surface Corrections, RSP2 & Pulsation Exports

Asteroseismology probes the interior sound speed $c_s(r)$, buoyancy frequency $N(r)$, and convective core or envelope boundaries of stars by comparing observed pressure (`p-mode`) and gravity (`g-mode`) oscillation frequencies against theoretical eigenfrequencies. This tutorial demonstrates how to compute acoustic transit profiles and global scaling parameters in `flexmesa.astero`, apply five empirical surface-effect corrections, evaluate Roxburgh & Vorontsov (2003) 5-point frequency separation ratios, run Nelder-Mead and Powell quadratic $\chi^2$ optimizations, evaluate Kuhfuss (1986) / Smolec & Moskalik (2008) `RSP2` turbulent radial pulsation equations, and export stellar interiors to `GYRE`, `FGONG`, and `OSC` files.

---

## 1. Acoustic Profiles, Global Scaling Relations & Glitch Depths

For a 1D stellar model (`StarModel`), `astero.eval_acoustic_profile` integrates the cumulative acoustic radius $\tau(r) = \int_0^r dr' / c_s(r')$ and photospheric acoustic transit time $T_0 = \tau(R_{\mathrm{phot}})$. From this profile and the surface parameters $(M, R_{\mathrm{phot}}, T_{\mathrm{eff}}, L)$, `flexmesa.astero` computes:

- **Asymptotic and Solar-Scaled Large Frequency Separation (`eval_delta_nu_asymptotic`, `eval_delta_nu_scaling`, `eval_delta_nu_radius_scaling`)**: $\Delta\nu_{\mathrm{asymp}} = (2 T_0)^{-1}$ and $\Delta\nu_{\mathrm{scale}} = \Delta\nu_\odot (M/M_\odot)^{1/2} (R/R_\odot)^{-3/2}$.
- **Frequency of Maximum Oscillation Power (`eval_nu_max`)** and **Acoustic Cutoff Frequency (`eval_nu_ac`)**: $\nu_{\max} \propto M R^{-2} T_{\mathrm{eff}}^{-1/2}$ and $\nu_{\mathrm{ac}} = c_s / (4\pi H_P)$.
- **Asymptotic Dipole Gravity-Mode Period Spacing (`eval_delta_pg`)**: $\Delta\Pi_1 = \sqrt{2}\pi^2 \left(\int_{N^2 > 0} N\,dr / r\right)^{-1}$.
- **Acoustic Glitch Depths (`eval_glitch_acoustic_depths`)**: Acoustic depth $\tau_{\mathrm{BCZ}}$ of the base of the convective envelope and $\tau_{\mathrm{He\,II}}$ of the second helium ionization zone ($\Gamma_1$ depression).

```python
import matplotlib.pyplot as plt
import numpy as np
from flexmesa import astero, plotting, star

model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
r_cm = np.asarray(model.r)
cs_cm_s = np.asarray(model.csound)

# 1. Cumulative acoustic profile and asymptotic vs. solar-scaled Delta_nu
ac_prof = astero.eval_acoustic_profile(r_cm, cs_cm_s)
t0_sec = float(ac_prof["photosphere_acoustic_r"])
dnu_asymp = astero.eval_delta_nu_asymptotic(t0_sec)
dnu_scale = astero.eval_delta_nu_scaling(
    m_g=model.mstar, teff_k=model.teff, l_phot_erg_s=float(model.lum[0])
)
dnu_rad = astero.eval_delta_nu_radius_scaling(
    m_g=model.mstar, r_phot_cm=float(r_cm[0])
)
nu_max = astero.eval_nu_max(
    m_g=model.mstar, r_phot_cm=float(r_cm[0]), teff_k=model.teff
)
nu_ac = astero.eval_nu_ac(
    g_surf_cgs=float(model.grav[0]),
    gamma1_surf=float(model.gamma1[0]),
    rho_surf=float(model.rho[0]),
    peos_surf=float(model.peos[0]),
)

# 2. Dipole g-mode period spacing and acoustic glitch depths
delta_pg = astero.eval_delta_pg(
    r_cm, np.asarray(model.rmid), np.asarray(model.brunt_n2), cs_cm_s
)
r_bcz_rsun, tau_bcz_s, tau_heii_s = astero.eval_glitch_acoustic_depths(
    r_cm,
    cs_cm_s,
    np.asarray(model.gradr),
    np.asarray(model.grada),
    np.asarray(model.gamma1),
    np.asarray(model.t),
)

print(
    f"1 Msun ZAMS: T_0 = {t0_sec:.1f} s, Delta_nu(asymp) = {dnu_asymp:.2f} uHz, "
    f"Delta_nu(scale) = {dnu_scale:.2f} uHz, Delta_nu(M,R) = {dnu_rad:.2f} uHz"
)
print(
    f"nu_max = {nu_max:.1f} uHz, nu_ac = {nu_ac:.1f} uHz, Delta_Pi_1 = {delta_pg:.1f} s, "
    f"r_BCZ = {r_bcz_rsun:.3f} Rsun, tau_BCZ = {tau_bcz_s:.1f} s, tau_HeII = {tau_heii_s:.1f} s"
)

# 3. Plot the Lamb (S_l) and Brunt-Vaisala (N) propagation diagram
fig, ax = plotting.plot_propagation_diagram(
    r_cm / r_cm[0],
    np.asarray(model.brunt_n2),
    cs_cm_s,
    r_cm,
    nu_max_uhz=float(nu_max),
)
plt.close(fig)
```

---

## 2. Empirical Surface Corrections, Separation Ratios & $\chi^2$ Optimization

Adiabatic pulsation frequencies computed on 1D mixing-length models exhibit a systematic high-frequency offset relative to observations because 1D MLT neglects turbulent pressure and non-adiabatic radiative damping near the photosphere. `astero.eval_surface_correction` implements five empirical corrections (with aliases for Ball & Gizon 2014 and Kjeldsen et al. 2008):

- **`"combined"` / `"bg14_2"`**: Ball & Gizon (2014) two-term inertia-weighted cubic plus inverse-frequency correction $\delta\nu_{nl} = \mathcal{I}_{nl}^{-1}\,[a_3 (\nu/\nu_{\mathrm{ac}})^3 + a_{-1} (\nu/\nu_{\mathrm{ac}})^{-1}]$.
- **`"cubic"` / `"bg14_1"`**: Ball & Gizon (2014) one-term cubic correction $\delta\nu_{nl} = a_3 (\nu/\nu_{\mathrm{ac}})^3 / \mathcal{I}_{nl}$.
- **`"kjeldsen"`**: Kjeldsen et al. (2008) fixed-exponent power law $\delta\nu \propto (\nu / \nu_{\max})^b$.
- **`"power_law"`**: Free two-parameter power-law fit $(a, b)$.
- **`"sonoi"`**: Sonoi et al. (2015) Lorentzian profile $\delta\nu / \nu_{\max} = \alpha [1 - (1 + (\nu/\nu_{\max})^\beta)^{-1}]$.

To isolate core structure from surface effects, `astero.eval_ratios_r01_r10` and `astero.eval_ratios_r02` compute the Roxburgh & Vorontsov (2003) 5-point frequency separation ratios $r_{01}(n), r_{10}(n), r_{02}(n)$ and their propagated uncertainties. Finally, `astero.eval_astero_chi2`, `astero.nelder_mead_minimize`, and `astero.powell_quadratic_minimize` evaluate and minimize the composite seismic + spectroscopic $\chi^2$:

```python
import matplotlib.pyplot as plt
import numpy as np
from flexmesa import astero, plotting

# Synthetic radial (l=0), dipole (l=1), and quadrupole (l=2) mode set
orders = np.arange(16, 23, dtype=np.float64)
l0_mod = 135.0 * (orders + 1.45)
l1_mod = 135.0 * (orders + 0.50 + 1.43)
l2_mod = 135.0 * (orders + 1.45) - 9.2
inertia = np.exp(-((l0_mod - 2700.0) / 800.0) ** 2)
sigma = np.full_like(l0_mod, 0.15)

# Inject a synthetic Ball & Gizon (2014) surface shift into observed radial frequencies
l0_obs = l0_mod - (4.5 * (l0_mod / 3078.0) ** 3 + 0.4 * (l0_mod / 3078.0) ** -1) / inertia

# 1. Compare all five surface correction schemes
for scheme in ("combined", "cubic", "kjeldsen", "power_law", "sonoi"):
    corr = astero.eval_surface_correction(
        scheme, l0_obs, sigma, l0_mod, inertia, delta_nu_obs=135.0, nu_max=3078.0
    )
    chi2_res = astero.eval_astero_chi2(
        l0_obs,
        sigma,
        corr["nu_corrected"],
        spectro_obs=[5777.0, 0.0],
        spectro_sigmas=[60.0, 0.05],
        spectro_model=[5785.0, 0.01],
    )
    print(
        f"Scheme {scheme:10s}: p0={float(corr['p0']):.3e}, p1={float(corr['p1']):.3e}, "
        f"chi2_freq={float(chi2_res['chi2_frequencies']):.3e}, chi2_tot={float(chi2_res['chi2_total']):.4f}"
    )

# 2. Compute 5-point frequency separation ratios r01, r10, and r02
ratios_010 = astero.eval_ratios_r01_r10(l0_obs, sigma, l1_mod, sigma)
r02, sig_r02 = astero.eval_ratios_r02(l0_obs, sigma, l1_mod, sigma, l2_mod, sigma)
print(f"Mean r01 = {np.mean(ratios_010['r01']):.4f}, mean r02 = {np.mean(r02):.4f}")

# 3. Nelder-Mead simplex and Powell quadratic optimization of (Mass, Mixing-Length alpha)
def seismic_objective(params: list[float]) -> float:
    m_try, alpha_try = params[0], params[1]
    return float(((m_try - 1.02) / 0.01) ** 2 + ((alpha_try - 1.92) / 0.05) ** 2)

nm_res = astero.nelder_mead_minimize(
    seismic_objective, x_first=[1.00, 1.80], x_lower=[0.80, 1.40], x_upper=[1.30, 2.40]
)
powell_res = astero.powell_quadratic_minimize(
    seismic_objective, x_first=[1.00, 1.80], x_lower=[0.80, 1.40], x_upper=[1.30, 2.40]
)
print(
    f"Nelder-Mead best={nm_res['x_best']} (f={nm_res['f_best']:.2e}, evals={nm_res['num_evals']}); "
    f"Powell best={powell_res['x_best']} (f={powell_res['f_best']:.2e})"
)

# 4. Plot an echelle diagram before and after the Ball & Gizon (2014) correction
corr_bg14 = astero.eval_surface_correction("combined", l0_obs, sigma, l0_mod, inertia)
fig, ax = plotting.plot_echelle_diagram(
    l0_obs,
    sigma,
    delta_nu=135.0,
    model_freqs_uncorr=l0_mod,
    model_freqs_corr=corr_bg14["nu_corrected"],
)
plt.close(fig)
```

![Asteroseismic echelle diagram and surface-effect corrections](../assets/figures/wave5_astero_echelle_surface_corr.png)

*Figure 1: Echelle diagram ($\nu \bmod \Delta\nu$ versus $\nu$) and frequency residuals illustrating how the Ball & Gizon (2014) two-term surface correction (`combined`) removes the systematic high-frequency offset between adiabatic 1D model eigenfrequencies and observations.*

---

## 3. `RSP2` Turbulent Radial Pulsation Equations & `GYRE` / `FGONG` / `OSC` Exports

For radially pulsating RR Lyrae, Cepheid, and $\delta$ Scuti variables, `FlexMESA` provides both:

1. **Kuhfuss (1986) / Smolec & Moskalik (2008) `RSP` & `RSP2` Turbulent Convection Kernels**: `_rust.star_solve_stationary_rsp2_w` (`star.solve_stationary_rsp2_w_jax`), `_rust.star_eval_rsp2_zone` (`star.eval_rsp2_zone_jax`), `_rust.star_eval_rsp2_face` (`star.eval_rsp2_face_jax`), `star.rsp2_flux_limiter_jax`, `star.compute_rsp2_uq_face_jax`, and `star.eval_rsp_hyd_fluxes_jax` compute the stationary turbulent velocity $w = \sqrt{e_t}$, buoyant source $S$, radiative eddy cooling $D_r$, Kolmogorov cascade dissipation $D$, turbulent pressure $P_{\mathrm{trb}}$, eddy viscosity $\mathrm{X}_{\mathrm{cell}}$, viscous shear heating $E_{q,\mathrm{cell}}$, quintic $C^2$ flux-limited convective enthalpy factor $\Pi_{\mathrm{face}}$, and Stellingwerf / Richtmyer-Morton fluxes $(L_r, L_c, L_t, Q_{\mathrm{visc}})$.
2. **Pulsation Structure Exporters (`to_gyre_str`, `to_fgong_str`, `to_osc_str`)**: Format the 1D stellar profile (including the center point $r = 0$ and Dziembowski dimensionless variables $V_g, A^\ast, U, c_1, \Gamma_1$) directly into `GYRE` (`v120`, 19 columns), `FGONG` (`ivers = 1300`, 40 variables), and `OSC` (`ivers = 1300`, 24 variables) oscillation code inputs.

```python
import jax.numpy as jnp
from flexmesa import star

model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)

# 1. Solve stationary RSP2 turbulent speed w and evaluate zone/face source terms
k = 20
w_stat = star.solve_stationary_rsp2_w_jax(
    t=jnp.float64(model.t[k]),
    rho=jnp.float64(model.rho[k]),
    p=jnp.float64(model.peos[k]),
    cp=jnp.float64(model.cp[k]),
    grad_ad=jnp.float64(model.grada[k]),
    delta=jnp.float64(model.chi_t[k] / model.chi_rho[k]),
    kappa=jnp.float64(model.opacity[k]),
    hp_cell=jnp.float64(model.scale_height[k]),
    grad_t=jnp.float64(model.grada[k] + 0.05),
)
rsp2_zone = star.eval_rsp2_zone_jax(
    w=w_stat,
    t=jnp.float64(model.t[k]),
    rho=jnp.float64(model.rho[k]),
    p=jnp.float64(model.peos[k]),
    cp=jnp.float64(model.cp[k]),
    grad_ad=jnp.float64(model.grada[k]),
    delta=jnp.float64(model.chi_t[k] / model.chi_rho[k]),
    kappa=jnp.float64(model.opacity[k]),
    hp_cell=jnp.float64(model.scale_height[k]),
    grad_t=jnp.float64(model.grada[k] + 0.05),
    r_outer=jnp.float64(model.r[k]),
    r_inner=jnp.float64(model.r[k + 1]),
    u_outer=jnp.float64(1.0e5),
    u_inner=jnp.float64(0.8e5),
    dm=jnp.float64(model.dm[k]),
    dlnd_dt=jnp.float64(-1.0e-5),
)
print(
    f"RSP2 zone {k}: stationary w = {float(w_stat):.3e} cm/s, "
    f"P_turb = {float(rsp2_zone[0]):.3e} dyn/cm^2, "
    f"chi_turb = {float(rsp2_zone[1]):.3e} erg, "
    f"eq_visc = {float(rsp2_zone[2]):.3e} erg/g/s"
)
fl_val, dfl_val = star.rsp2_flux_limiter_jax(jnp.asarray([0.975]))
print(f"RSP2 C^2 flux limiter at X=0.975: FL={float(fl_val[0]):.6f}, dFL/dX={float(dfl_val[0]):.6f}")

# 2. Export GYRE, FGONG, and OSC structure strings
gyre_text = model.to_gyre_str(add_center_point=True)
fgong_text = model.to_fgong_str(add_center_point=True)
osc_text = model.to_osc_str(add_center_point=True)
print(
    f"Exported pulsation payloads: GYRE={len(gyre_text.splitlines())} lines, "
    f"FGONG={len(fgong_text.splitlines())} lines, OSC={len(osc_text.splitlines())} lines"
)

# 3. Evaluate linear oscillation eigenmodes (GYRE/ADIPLS parity) and RSP LINA periods
p_modes = star.eval_linear_oscillation_modes(model, 0, [1, 2, 3])
lina_modes = star.eval_rsp_lina_modes(4.165, 1438.8, 6050.0, 0.73, 0.007, 3)
rsp_env = star.build_rsp_envelope_model(m_msun=0.65, l_lsun=45.0, teff_k=6700.0)
print(
    f"Linear p-modes (l=0, n=1..3): {[round(m['freq_uhz'], 2) for m in p_modes]} uHz; "
    f"RSP LINA P0={lina_modes[0]['period_days']:.4f} d; RSP env nz={rsp_env.nz}"
)
```
