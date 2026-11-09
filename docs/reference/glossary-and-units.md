# Glossary, Physical Units & Indexing Contract

`FlexMESA` adopts standard centimeter-gram-second (`CGS`) physical units with temperatures in kelvin ($\mathrm{K}$) and nuclear energies in mega-electronvolts ($\mathrm{MeV}$) or ergs ($\mathrm{erg}$) across all internal microphysics kernels, matching upstream Fortran MESA (`const/public/const_def.f90`), while exposing convenient solar-unit and calendar-unit attributes (`star_age`, `binary_age_yr`, `HistoryRow.star_mass`, `initial_period_days`) on high-level stellar and binary driver classes.

---

## 1. Fundamental Physical & Astronomical Constants (`flexmesa.constants`)

| Constant Name | Symbol | Numerical Value | Units | Physical Description |
| :--- | :---: | ---: | :--- | :--- |
| `constants.MSUN` | $M_\odot$ | $1.98840987 \times 10^{33}$ | $\mathrm{g}$ | Nominal IAU/MESA solar mass |
| `constants.RSUN` | $R_\odot$ | $6.957 \times 10^{10}$ | $\mathrm{cm}$ | Nominal IAU/MESA solar photospheric radius |
| `constants.LSUN` | $L_\odot$ | $3.828 \times 10^{33}$ | $\mathrm{erg\,s^{-1}}$ | Nominal IAU/MESA solar bolometric luminosity |
| `constants.SECYER` | $\mathrm{yr}$ | $3.15576 \times 10^{7}$ | $\mathrm{s}$ | Julian year ($365.25 \times 86400\,\mathrm{s}$) |
| `constants.SECDAY` | $\mathrm{d}$ | $8.64 \times 10^{4}$ | $\mathrm{s}$ | Mean solar day ($86400\,\mathrm{s}$) |
| `constants.STANDARD_CGRAV` | $G$ | $6.67430 \times 10^{-8}$ | $\mathrm{cm^3\,g^{-1}\,s^{-2}}$ | Newtonian gravitational constant |
| `constants.CLIGHT` | $c$ | $2.99792458 \times 10^{10}$ | $\mathrm{cm\,s^{-1}}$ | Speed of light in vacuum |
| `constants.KERG` | $k_{\mathrm{B}}$ | $1.380649 \times 10^{-16}$ | $\mathrm{erg\,K^{-1}}$ | Boltzmann constant |
| `constants.CRAD` | $a_{\mathrm{rad}}$ | $7.56573325 \times 10^{-15}$ | $\mathrm{erg\,cm^{-3}\,K^{-4}}$ | Radiation density constant ($4\sigma_{\mathrm{SB}}/c$) |
| `constants.AVO` | $N_{\mathrm{A}}$ | $6.02214076 \times 10^{23}$ | $\mathrm{mol^{-1}}$ | Avogadro constant ($1\,\mathrm{amu}^{-1}$) |
| `constants.PLANCK_H` | $h$ | $6.62607015 \times 10^{-27}$ | $\mathrm{erg\,s}$ | Planck constant |
| `constants.ME` | $m_e$ | $9.1093837015 \times 10^{-28}$ | $\mathrm{g}$ | Electron rest mass |
| `constants.EV2ERG` | $\mathrm{eV}$ | $1.602176634 \times 10^{-12}$ | $\mathrm{erg}$ | Electronvolt to erg conversion |

```python
from flexmesa import constants

print(
    f"1 Msun = {constants.MSUN:.8e} g, "
    f"1 Rsun = {constants.RSUN:.4e} cm, "
    f"1 yr   = {constants.SECYER:.5e} s"
)
```

---

## 2. Stellar Structure, Microphysics & Seismic Symbols

| Symbol | Code Identifier | Units | Definition & Staggered Grid Location |
| :---: | :--- | :--- | :--- |
| $m_k$ | `model.m` | $\mathrm{g}$ | Enclosed gravitational mass at outer cell face $k$ |
| $q_k$ | `model.q` | dimensionless | Relative enclosed mass coordinate $q_k = m_k / M_{\mathrm{tot}} \in (0, 1]$ at outer face $k$ |
| $dq_k$ | `model.dq` | dimensionless | Cell mass thickness $dq_k = \Delta m_k / M_{\mathrm{tot}}$ ($\sum_{k=0}^{\mathrm{nz}-1} dq_k = 1$) at cell center $k$ |
| $r_k$ | `model.r` | $\mathrm{cm}$ | Eulerian radius at outer cell face $k$ |
| $L_k$ | `model.lum` | $\mathrm{erg\,s^{-1}}$ | Net radiative plus convective luminosity at outer cell face $k$ |
| $u_k$ | `u_cell` | $\mathrm{cm\,s^{-1}}$ | Radial Lagrangian fluid velocity in cell $k$ |
| $\rho_k$ | `model.rho` | $\mathrm{g\,cm^{-3}}$ | Mass density at cell center $k$ (`lnd = ln(rho)`) |
| $T_k$ | `model.t` | $\mathrm{K}$ | Temperature at cell center $k$ (`lnT = ln(T)`) |
| $P_{\mathrm{gas}}$ | `eos.I_LN_PGAS` | $\mathrm{dyn\,cm^{-2}}$ | Gas pressure (ions + electrons + positrons + Coulomb) at cell center $k$ |
| $P_{\mathrm{rad}}$ | `model.prad` | $\mathrm{dyn\,cm^{-2}}$ | Thermal radiation pressure $\frac{1}{3} a_{\mathrm{rad}} T^4$ at cell center $k$ |
| $P$ | `model.peos` | $\mathrm{dyn\,cm^{-2}}$ | Total thermodynamic pressure $P = P_{\mathrm{gas}} + P_{\mathrm{rad}}$ at cell center $k$ |
| $\Gamma_1$ | `eos.I_GAMMA1` | dimensionless | First adiabatic exponent $(\partial\ln P / \partial\ln\rho)_s$ at cell center $k$ (`eos.I_GAMMA3` for $\Gamma_3$) |
| $\nabla_{\mathrm{ad}}$ | `model.grada` | dimensionless | Adiabatic temperature gradient $(\partial\ln T / \partial\ln P)_s$ |
| $\nabla_{\mathrm{rad}}$ | `model.gradr` | dimensionless | Radiative temperature gradient $3\kappa P L / (16\pi a_{\mathrm{rad}} c G m T^4)$ at face $k$ |
| $\nabla_T$ | `model.gradt` | dimensionless | Actual structural temperature gradient $d\ln T / d\ln P$ at face $k$ |
| $\nabla_{\mathrm{L}}$ | `model.gradl` | dimensionless | Ledoux temperature gradient $\nabla_{\mathrm{ad}} + (\chi_\mu / \chi_T)\nabla_\mu$ |
| $\eta$ | `eos.I_ETA` | dimensionless | Dimensionless electron degeneracy parameter $\mu_e / (k_{\mathrm{B}} T)$ |
| $\Gamma_{\mathrm{Coul}}$ | `pc_ion` $\Gamma$ | dimensionless | One-component or multi-component plasma Coulomb coupling parameter |
| $\phi_{\mathrm{cryst}}$ | `eos.I_PHASE` | dimensionless | Continuous `Skye` crystallization phase parameter ($\phi = 0$ liquid, $\phi = 1$ solid) |
| $\bar{A}, \bar{Z}$ | `abar`, `zbar` | $\mathrm{amu}$, $e$ | Mean atomic mass number $\bar{A} = (\sum X_i/A_i)^{-1}$ and mean ionic charge $\bar{Z} = \bar{A}\sum Z_i X_i/A_i$ |
| $Y_e$ | `ye` | $\mathrm{mol\,g^{-1}}$ | Electron-to-baryon molar fraction $Y_e = \bar{Z}/\bar{A} = \sum Z_i X_i / A_i$ |
| $\kappa$ | `model.opacity` | $\mathrm{cm^2\,g^{-1}}$ | Total Rosseland mean opacity $(\kappa_{\mathrm{rad}}^{-1} + \kappa_{\mathrm{cond}}^{-1})^{-1}$ at cell center $k$ |
| $\epsilon_{\mathrm{nuc}}$ | `model.eps_nuc` | $\mathrm{erg\,g^{-1}\,s^{-1}}$ | Specific thermonuclear energy generation rate at cell center $k$ |
| $\epsilon_\nu$ | `model.non_nuc_neu` | $\mathrm{erg\,g^{-1}\,s^{-1}}$ | Specific thermal neutrino cooling rate (Itoh 5-channel) at cell center $k$ |
| $\epsilon_{\mathrm{grav}}$ | `model.eps_grav` | $\mathrm{erg\,g^{-1}\,s^{-1}}$ | Specific gravothermal energy rate $-T\,ds/dt$ at cell center $k$ |
| $v_{\mathrm{conv}}$ | `model.conv_vel` | $\mathrm{cm\,s^{-1}}$ | Turbulent convective velocity from `MLT` or `TDC`/`RSP2` at face $k$ |
| $w$ | `w_turb` | $\mathrm{cm\,s^{-1}}$ | Convective turbulent velocity ($e_t = w^2$ in $\mathrm{erg\,g^{-1}}$) in `RSP2` |
| $N^2$ | `model.brunt_n2` | $\mathrm{rad^2\,s^{-2}}$ | Squared Brunt-Väisälä buoyancy frequency at face $k$ |
| $\Omega$ | `omega` | $\mathrm{rad\,s^{-1}}$ | Shellular rotational angular velocity at face $k$ |
| $R_{\mathrm{RL},1}$ | `sys.rl[0]` | $\mathrm{cm}$ | Eggleton (1983) volume-equivalent donor Roche-lobe radius |
| $\Delta\nu$ | `delta_nu` | $\mu\mathrm{Hz}$ | Large asteroseismic frequency separation $\langle \nu_{n,\ell} - \nu_{n-1,\ell} \rangle$ |
| $\nu_{\max}$ | `nu_max` | $\mu\mathrm{Hz}$ | Frequency of maximum acoustic oscillation power |
| $\Delta\Pi_1$ | `delta_pi1` | $\mathrm{s}$ | Asymptotic dipole ($\ell = 1$) gravity-mode period spacing |

---

## 3. Index and Memory-Order Contract

Every module in `FlexMESA` adheres to a single, explicit indexing and memory-layout contract bridging Fortran MESA (`1`-based, column-major) and Safe Rust / NumPy / JAX (`0`-based, C-contiguous row-major):

1. **Radial Direction (`k = 0` Photosphere to `k = nz - 1` Center)**:
   - Index `k = 0` is the **outermost surface (photospheric) zone** (`q[0] = 1.0`, matching Fortran MESA's `k = 1`).
   - Index `k = nz - 1` is the **innermost central core zone** (`q[nz - 1] = dq[nz - 1]`, matching Fortran MESA's `k = s% nz`).
2. **Staggered Cell vs. Face Alignment**:
   - Outer face `k` bounds the outer surface of cell `k`.
   - Intensive thermodynamic and composition quantities ($\rho_k, T_k, P_k, \kappa_k, \epsilon_{\mathrm{nuc},k}, X_{k,i}, dq_k$) live at the mass center of cell `k`.
   - Extensive and flux quantities ($m_k, q_k, r_k, L_k, \nabla_{\mathrm{rad},k}, \nabla_{T,k}, v_{\mathrm{conv},k}, N^2_k, \Omega_k$) live on outer face `k`.
3. **2D Array Memory Order (`C`-Contiguous Row-Major)**:
   - Multi-zone species and block-tridiagonal arrays have leading radial dimension `nz`: `xa` has shape `(nz, n_species)` (`xa[k, i]` is mass fraction of species `i` in zone `k`), and Henyey block-tridiagonal Jacobians $(L, D, U)$ have shape `(nz, n_var, n_var)`.
