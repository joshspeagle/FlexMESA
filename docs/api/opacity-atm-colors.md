# Opacity, Atmospheres & Photometry (`kap`, `atm`, `colors`)

This reference page covers the radiative and conductive opacity subpackage (`flexmesa.kap`), the surface boundary condition and model atmosphere subpackage (`flexmesa.atm`), and the synthetic spectral energy distribution and filter photometry subpackage (`flexmesa.colors`).

For physical derivations and diagnostic plots, see [Opacity, Atmospheres & Photometry](../concepts/opacity-and-atmospheres.md).

---

## Quick Usage Example

```python
import numpy as np
import flexmesa.atm as atm
import flexmesa.colors as colors
import flexmesa.kap as kap
from flexmesa import constants

# 1. Total Rosseland mean opacity (radiative + electron conduction)
k_res = kap.kap_get_results(
    z=0.02,
    x=0.70,
    zbase=0.02,
    log_rho=-5.0,
    log_t=5.0,
    lnfree_e=20.0,
    zbar=1.10,
)

# 2. Tabulated photospheric surface boundary (P_surf, T_surf)
a_res = atm.eval_table(
    luminosity=constants.LSUN,
    r=constants.RSUN,
    m=constants.MSUN,
    cgrav=constants.STANDARD_CGRAV,
    table_id=atm.ATM_TABLE_PHOTOSPHERE,
    z=0.02,
    skip_partials=False,
    teff=5777.0,
)

# 3. Synthetic Johnson-Cousins colors at Teff = 5777 K, log g = 4.44, [M/H] = 0.0
phot = colors.eval_colors(
    teff=5777.0, log_g=4.44, metallicity=0.0, r=constants.RSUN
)
print(
    f"kappa={k_res.kap:.4f} cm^2/g, T_surf={np.exp(a_res.ln_t):.1f} K, "
    f"M_bol={float(phot[0]):.3f} mag"
)
```

---

## Radiative, Conductive & Compton Opacities (`flexmesa.kap`)

`flexmesa.kap` combines low-temperature molecular (`Ferguson`/`AESOPUS`) and high-temperature atomic (`OPAL` Type 1/Type 2 and `Opacity Project`) Rosseland mean radiative opacities $\kappa_{\mathrm{rad}}(\log_{10} R, \log_{10} T, X, Z)$ with Poutanen (2017) relativistic Compton scattering and Cassisi et al. (2007) / Potekhin degenerate electron thermal conduction $\kappa_{\mathrm{cond}}$ via harmonic blending $\kappa^{-1} = \kappa_{\mathrm{rad}}^{-1} + \kappa_{\mathrm{cond}}^{-1}$.

::: flexmesa.kap.core

::: flexmesa.kap.submodels

::: flexmesa.kap.jax_kernels

---

## Model Atmospheres & Outer Boundary Conditions (`flexmesa.atm`)

`flexmesa.atm` evaluates outer photospheric boundary conditions $(P_{\mathrm{surf}}, T_{\mathrm{surf}})$ and their exact $(L, R, M)$ partial derivatives using grey $T(\tau)$ relations (Eddington, Krishna-Swamy, solar Hopf, Guillot irradiated planetary atmospheres), RK4 optical-depth integration, and pre-validated 2D bicubic spline atmosphere tables (`photosphere`, `tau_100`, `tau_10`, `tau_1`, `tau_1m1`, `wd_tau_25`, `db_wd_tau_25`).

::: flexmesa.atm._core

::: flexmesa.atm._jax

---

## Synthetic SEDs & Filter Photometry (`flexmesa.colors`)

`flexmesa.colors` synthesizes stellar spectral energy distributions (SEDs) across the 1199-wavelength Kurucz/Castelli `(Teff, log_g, [M/H])` grid using a 32-gather 3D cubic Hermite interpolator and integrates bolometric corrections and synthetic absolute magnitudes across Johnson-Cousins $UBVRIJHK$ filter transmission curves via non-uniform quadratic Simpson quadrature.

::: flexmesa.colors.photometry

::: flexmesa.colors.sed

::: flexmesa.colors.synthetic

::: flexmesa.colors.utils
