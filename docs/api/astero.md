# Asteroseismology & Optimization (`flexmesa.astero`)

The `flexmesa.astero` subpackage implements upstream MESA's `astero/` diagnostic, surface-correction, and seismic model-fitting suite across Safe Rust (`crates/flexmesa-astero`) and pure JAX (`float64`). It provides:

- **Acoustic Profiles & Global Scaling Relations**: Cumulative acoustic radius $\tau(r) = \int_0^r dr'/c_s(r')$, acoustic depth $\tau_0 - \tau(r)$, asymptotic and solar-calibrated $\Delta\nu$, $\nu_{\max}$, and acoustic cutoff frequency $\nu_{\mathrm{ac}}$, traditional asymptotic dipole g-mode period spacing $\Delta\Pi_1$ (with optional rotation modification), and acoustic glitch locations ($\tau_{\mathrm{He\,II}}$, $\tau_{\mathrm{BCZ}}$).
- **Empirical Surface-Effect Corrections**: Five empirical frequency-shift schemes accounting for near-surface superadiabatic convection and non-adiabatic wave physics (`"none"`, `"kjeldsen"` / `"power_law"`, `"cubic"` / `"bg14_1"`, `"combined"` / `"bg14_2"`, and `"sonoi"`).
- **Frequency-Separation Ratios & Seismic Likelihoods**: Roxburgh & Vorontsov (2003) 5-point small-to-large frequency separation ratios $(r_{01}, r_{10}, r_{02})$ insensitive to outer surface layers, mode-by-mode $\chi^2_{\mathrm{seism}}$, spectroscopic $\chi^2_{\mathrm{spec}}$, and weighted total $\chi^2_{\mathrm{tot}}$.
- **Derivative-Free & Gradient-Based Optimization**: Standalone Nelder-Mead simplex (`nelder_mead_minimize`), Powell quadratic direction-set (`powell_quadratic_minimize`), and Cartesian grid-search drivers (`scan_cartesian_grid`), alongside end-to-end differentiable JAX seismic loss functions (`jax.grad`, `jax.hessian`).

For theoretical background and tutorials, see [Asteroseismology & Pulsation Exports](../concepts/asteroseismology.md) and [Tutorial 5: Asteroseismology & Pulsations](../tutorials/05-asteroseismology-and-pulsations.md).

---

## Quick Usage Example

```python
import flexmesa.astero as astero
from flexmesa import constants

dnu = astero.eval_delta_nu_radius_scaling(
    m_g=1.0 * constants.MSUN, r_phot_cm=1.0 * constants.RSUN
)
numax = astero.eval_nu_max(
    m_g=1.0 * constants.MSUN, r_phot_cm=1.0 * constants.RSUN, teff_k=5777.0
)
print(f"Solar scaling: dnu={dnu:.2f} uHz, nu_max={numax:.1f} uHz")
```

---

## Package Overview (`flexmesa.astero`)

::: flexmesa.astero

---

## Scaling Relations & Surface-Correction Wrappers (`flexmesa.astero.scaling_corr`)

::: flexmesa.astero.scaling_corr

---

## Differentiable Acoustic & Surface-Correction Kernels (`flexmesa.astero.jax_kernels`)

::: flexmesa.astero.jax_kernels

---

## Seismic Optimization Drivers (`flexmesa.astero.drivers`)

::: flexmesa.astero.drivers
