# Nuclear Networks, Rates & Neutrinos (`chem`, `ionization`, `neu`, `rates`, `net`)

This reference page documents the five nuclear and microphysical reaction subpackages: nuclide catalogs and solar abundance scales (`flexmesa.chem`), partial ionization and effective ionic charges (`flexmesa.ionization`), thermal neutrino cooling (`flexmesa.neu`), thermonuclear REACLIB rates and plasma screening (`flexmesa.rates`), and nuclear reaction networks and stiff 1-zone burners (`flexmesa.net`).

For governing equations and diagnostic plots, see [Nuclear Networks, Rates & Neutrinos](../concepts/nuclear-and-neutrinos.md).

---

## Quick Usage Example

```python
import numpy as np
import flexmesa.chem as chem
import flexmesa.net as net
import flexmesa.neu as neu

# 1. Solar mixture composition moments on basic.net (8 isotopes)
net_info = net.get_net_info("basic.net")
xa = np.array([0.70, 2.0e-5, 0.28, 3.0e-3, 1.0e-3, 6.0e-3, 2.0e-3, 9.98e-3])
comp = chem.get_composition_info(net_info.chem_id, xa)

# 2. Thermal neutrino loss rate (Itoh et al. 1996)
neu_out = neu.neu_get(
    t=1.0e8,
    log10_t=8.0,
    rho=1.0e4,
    log10_rho=4.0,
    abar=float(comp.abar),
    zbar=float(comp.zbar),
)

# 3. Nuclear reaction network RHS and analytical Jacobian
net_out = net.eval_net(
    "basic.net",
    x=xa,
    temp=8.0e7,
    rho=100.0,
    abar=float(comp.abar),
    zbar=float(comp.zbar),
    z2bar=float(comp.z2bar),
    ye=float(comp.ye),
    screening_mode="extended",
)
print(
    f"abar={comp.abar:.4f}, eps_neu={neu_out.loss[0]:.3e} erg/g/s, "
    f"eps_nuc={net_out.eps_nuc:.3e} erg/g/s"
)
```

---

## Nuclide Catalog & Solar Abundance Scales (`flexmesa.chem`)

`flexmesa.chem` provides 347 built-in reference nuclides (plus full `isotopes.data` parsing), 286 solar isotopic abundances, 10 standard solar abundance compilations (`AG89`, `GS98`, `L03`, `AGS05`, `AGSS09`, `L09`, `A09_Prz`, `MB22_phot`, `AAG21_phot`, `AAG21`), and composition moment reductions ($\bar{A}, \bar{Z}, \overline{Z^2}, Y_e, \mu$).

::: flexmesa.chem.catalog

::: flexmesa.chem.composition

---

## Partial Ionization & Effective Ionic Charge (`flexmesa.ionization`)

`flexmesa.ionization` evaluates effective ionic charge states $Z_{\mathrm{eff}}(T, \rho)$ for trace heavy elements (including $\mathrm{Fe}^{56}$ in $\mathrm{He}^4$ via pre-validated 2D bicubic splines and general $Z \in [1, 30]$ depression-of-continuum ionization) used in atomic diffusion and gravitational settling.

::: flexmesa.ionization.typical_charge

::: flexmesa.ionization.fe56_he4

::: flexmesa.ionization.tables

---

## Thermal Neutrino Cooling (`flexmesa.neu`)

`flexmesa.neu` ports the Itoh et al. (1996) fitting formulae across all five thermal neutrino emission channels (`pair`, `plasmon`, `photoneutrino`, `bremsstrahlung`, and `recombination`) with exact $(\rho, T, \bar{A}, \bar{Z})$ partial derivatives.

::: flexmesa.neu.driver

::: flexmesa.neu.inputs

::: flexmesa.neu.pair_plas

::: flexmesa.neu.phot

::: flexmesa.neu.brem

::: flexmesa.neu.reco

---

## Thermonuclear Rates, Weak Reactions & Screening (`flexmesa.rates`)

`flexmesa.rates` evaluates 316 built-in JINA REACLIB 7-parameter thermonuclear rates, weak electron-capture and $\beta$-decay tables, and Salpeter (1954), Dewitt, Graboske & Cooper (1973), Alastuey & Jancovici (1978), Itoh (1979) (`extended`), and Chugunov et al. (2007) plasma screening factors.

::: flexmesa.rates.core

::: flexmesa.rates.screening

::: flexmesa.rates.weak

::: flexmesa.rates.jax_kernels

::: flexmesa.rates.jax_screening

---

## Nuclear Reaction Networks & Stiff Burners (`flexmesa.net`)

`flexmesa.net` constructs arbitrary reaction topologies (`basic.net`, `pp_and_cno_extras.net`, `co_burn.net`, `approx21.net`, `mesa_80.net`), evaluates specific nuclear energy generation $\epsilon_{\mathrm{nuc}}$, reaction neutrino losses $\epsilon_{\nu,\mathrm{nuc}}$, molar abundance derivatives $dY_i/dt$, and exact analytical sparse/dense Jacobians $\partial(dY_i/dt)/\partial Y_j$ backed by thread-local `NetWorkState` buffers, and integrates stiff 1-zone prescribed-trajectory (`burn_1_zone`), constant-density variable-temperature (`burn_1_zone_const_density`), and constant-pressure (`burn_1_zone_const_p`) networks via semi-implicit Bader-Deuflhard and Rosenbrock time stepping.

::: flexmesa.net._core

::: flexmesa.net._jax
