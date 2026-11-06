# Wave 1 Microphysics: Thermal Neutrinos & Turbulent Convection

This guide covers the physical equations, numerical root-finding algorithms, and automatic differentiation strategies implemented in `flexmesa.neu` (`crates/flexmesa-neu`) and `flexmesa.turb` (`crates/flexmesa-turb`).

---

## 1. Five-Channel Thermal Neutrino Energy Loss (`neu`)

In evolved stellar cores at $T \gtrsim 10^{7.5}\,\mathrm{K}$, thermal neutrino emission ($\epsilon_\nu$ in $\mathrm{erg\,g^{-1}\,s^{-1}}$) dominates energy loss over surface photon radiation. `flexmesa.neu` ports the complete Itoh et al. (1996, ApJS, 102, 411) fitting formulae from `neu/public/neu_lib.f90` and `neu/private/neu.f90` across all five physical channels:

1. **Electron-Positron Pair Annihilation ($e^- + e^+ \to \nu + \bar{\nu}$, `pair`)**: Dominant at high temperature ($T \gtrsim 10^9\,\mathrm{K}$) and moderate density where relativistic $e^\pm$ pairs are abundant.
2. **Plasmon Decay ($\gamma_{\mathrm{plas}} \to \nu + \bar{\nu}$, `plas`)**: Collective transverse and longitudinal plasma excitations decaying in degenerate electron gases ($\rho \gtrsim 10^5\,\mathrm{g\,cm^{-3}}$).
3. **Photoneutrino Process ($e^- + \gamma \to e^- + \nu + \bar{\nu}$, `phot`)**: Compton-like scattering with neutrino-pair emission, dominating at intermediate temperatures ($10^8\,\mathrm{K} \lesssim T \lesssim 10^{8.8}\,\mathrm{K}$).
4. **Bremsstrahlung ($e^- + (Z, A) \to e^- + (Z, A) + \nu + \bar{\nu}$, `brem`)**: Coulomb scattering off partially or fully crystallized ions across liquid and bcc lattice regimes.
5. **Electron-Ion Recombination (`reco`)**: Bound-free electron capture onto K-shell ions. Across the low-temperature threshold interval $\log_{10} T_{\min} < \log_{10} T \le \log_{10} T_{\mathrm{lim}}$ (with $\log_{10} T_{\min} = 7.0$ and default $\log_{10} T_{\mathrm{lim}} = 7.5$), all five channels are modulated by a smooth cosine turn-on factor:
   $$f_{\mathrm{cut}}(\log_{10} T) = \frac{1}{2}\left[1 - \cos\!\left(\pi\,\frac{\log_{10} T - \log_{10} T_{\min}}{\log_{10} T_{\mathrm{lim}} - \log_{10} T_{\min}}\right)\right].$$

![Itoh et al. (1996) thermal neutrino energy-loss channels and logarithmic temperature sensitivity](../assets/figures/neu_regimes.png)

### Exact 4-Variable Jacobian Propagation
Both Safe Rust (`neu_get` propagating exact analytical chain-rule derivatives in `NeuRate`) and Python/JAX (`neu_get` with `backend="auto" | "rust" | "jax"` and `neu_pair_plas_jax`) return the total rate $\epsilon_\nu$ and channel breakdown alongside exact analytical partial derivatives with respect to $(T, \rho, \bar{A}, \bar{Z})$:

$$\mathbf{J}_{\nu} = \left(\epsilon_\nu,\, \frac{\partial \epsilon_\nu}{\partial T},\, \frac{\partial \epsilon_\nu}{\partial \rho},\, \frac{\partial \epsilon_\nu}{\partial \bar{A}},\, \frac{\partial \epsilon_\nu}{\partial \bar{Z}}\right).$$

---

## 2. Convective and Diffusive Turbulent Transport (`turb`)

`flexmesa.turb` (`crates/flexmesa-turb`) implements four physical mixing regimes:

### 2.1 Local Mixing-Length Theory (`set_mlt`, `set_mlt_jax`)
For a convectively unstable shell ($\nabla_r > \nabla_L$), local Mixing-Length Theory solves the cubic convective efficiency equation (Cox & Giuli 1968, §14.82) for the dimensionless convective efficiency $\Gamma$:

$$a_0 \Gamma^3 + \Gamma^2 + \Gamma - a_0 B^3 = 0, \qquad B = A\,\left(\nabla_r - \nabla_L\right)^{1/2},$$

where the geometric prefactor $a_0$ and radiative-to-convective conductance ratio $A$ depend on the selected prescription (`Cox`, `TDC`, `Henyey`, `ML1`, `ML2`, or `Mihalas`). `FlexMESA` solves this depressed cubic analytically via hyperbolic/trigonometric Vieta-Cardano formulas in `DualOrder1<6>` (Safe Rust) and branchless `jnp.where` (JAX), recovering:

$$Y = \nabla_T - \nabla_L = \frac{\nabla_r - \nabla_L}{1 + a_0 \Gamma^2 (\Gamma + 1)^{-1}}, \qquad \nabla_T = \nabla_L + Y.$$

### 2.2 Kuhfuss (1986) 1D Time-Dependent Convection (`set_tdc`)
When stellar pulsation or rapid nuclear burning timescales approach the convective turnover time, steady-state MLT breaks down. `set_tdc` solves the time-integrated Kuhfuss (1986) turbulent kinetic energy equation over a step $\Delta t$ for the scaled convective speed $A_f(Y)$:

$$\frac{d A}{dt} = \Xi_0(Y) + \Xi_1 A + \Xi_2 A^2, \qquad Q(Y) = \left(L - L_0 \nabla_L\right) - L_0 Y - c_0 A_f(Y)\,Y_{\mathrm{env}}(Y) = 0.$$

To guarantee convergence across 80 orders of magnitude in $Y$, `set_tdc` searches in $Z = \ln|Y| \in [-100, 100]$:

1. **Bisection Bracketing**: Narrows $[Z_{\mathrm{lb}}, Z_{\mathrm{ub}}]$ until $Z_{\mathrm{ub}} - Z_{\mathrm{lb}} < 1.0$.
2. **Damped Newton-Raphson Refinement**: Iterates $Z_{n+1} = Z_n - Q(Z_n) / Q'(Z_n)$ with midpoint bracket clamping whenever a trial step exits $[Z_{\mathrm{lb}}, Z_{\mathrm{ub}}]$, terminating when $|Q| / \mathrm{scale} \le 10^{-8}$ or $|\Delta Z| < 10^{-13}$.
3. **Implicit Function Theorem Differentiation**: Once $Z^*$ converges in scalar `f64`, all partial derivatives of $\nabla_T, v_{\mathrm{conv}}, D$ with respect to $(\chi_T, \chi_\rho, C_P, g, H_P, \rho, P, T, \kappa, \nabla_r, \nabla_{\mathrm{ad}}, \nabla_L)$ are computed in a **single** pass at $Y^* = \exp(Z^*)$:
   $$\frac{\partial Y^*}{\partial \theta} = -\left(\frac{\partial Q}{\partial Y}\bigg|_{Y^*}\right)^{-1} \frac{\partial Q}{\partial \theta}\bigg|_{Y^*}.$$

This eliminates MESA's nested 70-component `auto_diff_real_tdc` inside the Newton loop and accelerates 4096-zone `turb_tdc` evaluation by **$54.1\times$ in 4-thread Safe Rust** and **$146.0\times$ in batched GPU JAX** relative to 4-thread OpenMP Fortran MESA.

![Steady-state Cox MLT vs. transient Kuhfuss TDC temperature gradient and Implicit Function Theorem sensitivity](../assets/figures/turb_mlt_tdc_comparison.png)

### 2.3 Semiconvection and Thermohaline Mixing (`set_semiconvection`, `set_thermohaline`)
- **Langer, El Eid, & Fricke (1985) Semiconvection**: Evaluates compositional diffusion $D_{\mathrm{sc}}$ and modified temperature gradient $\nabla_T$ in Ledoux-stable, Schwarzschild-unstable zones ($\nabla_{\mathrm{ad}} < \nabla_r < \nabla_L$).
- **Salt-Finger Thermohaline Mixing**: Implements Kippenhahn et al. (1980), Traxler, Garaud, & Stellmach (2011), and Brown, Garaud, & Stellmach (2013) double-diffusive fingering convection driven by an inverted mean molecular weight gradient ($\nabla_\mu < 0$ in $\nabla_r < \nabla_{\mathrm{ad}}$ radiative zones).
