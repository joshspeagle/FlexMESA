# Equations of State (`flexmesa.eos`)

The `flexmesa.eos` subpackage ports MESA's 7-component thermodynamic Equation of State (`eos/`) across Safe Rust (`crates/flexmesa-eos`) and JAX (`float64`). Every evaluation populates the canonical 26-slot thermodynamic state vector `res[0..26]` alongside exact analytical first and second logarithmic density and temperature partial derivatives (`d_dlnd[0..26]`, `d_dln_t[0..26]`) and optional composition partials (`d_dxa`):

1. **`Ideal` (`eval_ideal_eos`)**: Neutral ideal gas plus thermal radiation pressure for reference and outer boundary baselines.
2. **`HELM` (`eval_helm_eos`)**: Timmes & Swesty (2000) Helmholtz free-energy table with biquintic electron-positron interpolation, recursive `@jax.custom_jvp` Horner basis derivatives, and Yakovlev & Shalybkov (1989) Coulomb corrections.
3. **`PC` (`eval_pc_eos`)**: Potekhin & Chabrier (2010) analytical degenerate ionized plasma and Coulomb crystal EOS for white dwarf cores and neutron star envelopes.
4. **`Skye` (`eval_skye_eos`)**: Jermyn et al. (2021) unified third-order forward-mode automatic differentiation (`Order3` / nested `jax.jacfwd`) free-energy minimizer with smooth translational entropy blending across the liquid-to-solid Coulomb crystallization transition ($\Gamma_{\mathrm{crit}} \approx 175$).
5. **`FreeEOS`, `OPAL/SCVH` & `CMS` (`eval_free_eos`, `eval_opal_scvh_eos`, `eval_cms_eos`)**: 4D $(X, Z, \log_{10} Q, \log_{10} T)$ tabulated bicubic spline evaluators covering partially ionized stellar envelopes, solar-like interiors, and giant-planet hydrogen-helium mixtures.
6. **Polygonal $C^2$ Blending & Root-Finding Inversions (`eos_dt_get`, `eos_pt_get`)**: Smooth quintic smootherstep blending ($S_5(u) = 6u^5 - 15u^4 + 10u^3$) across component boundaries in the $(\log_{10}\rho, \log_{10}T)$ plane, and damped Newton-Raphson inversions from $(P_{\mathrm{gas}}, T)$ or $(s, T)$ to $\ln\rho$.

For physical derivations and phase diagrams, see [Equations of State (eos)](../concepts/eos.md).

---

## Quick Usage Example

```python
import numpy as np
import flexmesa.eos as eos
import flexmesa.net as net

net_info = net.get_net_info("basic.net")
xa = np.array([0.70, 2.0e-5, 0.28, 3.0e-3, 1.0e-3, 6.0e-3, 2.0e-3, 9.98e-3])
out = eos.eos_dt_get(
    rho=100.0,
    temp=1.5e7,
    chem_id=net_info.chem_id,
    xa=xa,
)
print(
    f"ln P_gas={out.res[eos.I_LN_PGAS]:.6f}, "
    f"grad_ad={out.res[eos.I_GRAD_AD]:.5f}, "
    f"Gamma_1={out.res[eos.I_GAMMA1]:.5f}"
)
```

---

## Package Overview (`flexmesa.eos`)

::: flexmesa.eos

---

## Core State Vector, Blending & 4D Tables (`flexmesa.eos.core`)

::: flexmesa.eos.core

---

## Aparicio Gauss-Fermi Integrals (`flexmesa.eos.fermi`)

::: flexmesa.eos.fermi

---

## Analytical & Free-Energy Evaluators (`ideal`, `helm`, `pc`, `skye`)

::: flexmesa.eos.ideal

::: flexmesa.eos.helm

::: flexmesa.eos.pc

::: flexmesa.eos.skye

---

## Thermodynamic Root-Finding Inversions (`flexmesa.eos.solver`)

::: flexmesa.eos.solver
