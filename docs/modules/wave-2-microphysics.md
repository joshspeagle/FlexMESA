# Wave 2 Microphysics: Nuclear Composition, Ionization & Synthetic Photometry

This guide covers the physical foundations, numerical interpolation schemes, and Three-Tier Hybrid implementations (`Safe Rust`, `Python/PyO3`, and `JAX`) in `flexmesa.chem` (`crates/flexmesa-chem`), `flexmesa.ionization` (`crates/flexmesa-ionization`), and `flexmesa.colors` (`crates/flexmesa-colors`).

---

## 1. Atomic and Nuclear Composition (`chem`)

Stellar equations of state, radiative opacities, and nuclear reaction networks depend on moments of the nuclide mass-fraction vector $\mathbf{X} = (X_1, \dots, X_K)$ across $K$ active isotopes. `flexmesa.chem` embeds the complete upstream MESA 5,095-nuclide ground-state and isomeric catalog (`isotopes.data`, including Wigner-function 24-point temperature-dependent partition functions and mass excesses $\Delta M_i$ in $\mathrm{MeV}$) alongside 12 solar and meteoritic reference abundance mixtures (Anders & Grevesse 1989, Grevesse & Noels 1993, Grevesse & Sauval 1998, Lodders 2003, Asplund et al. 2004/2005/2009, and Caffau et al. 2011).

### 1.1 Exact Composition Moments and Partial Derivatives
Given mass fractions $X_i$, integer baryon numbers $A_i = Z_i + N_i$, atomic charges $Z_i$, and atomic weights $W_i$ ($\mathrm{g\,mol^{-1}}$), `basic_composition_info` and `get_composition_info` compute the molar abundances $Y_i = X_i / A_i$, mean atomic weight $\bar{A}$, mean ionic charge $\bar{Z}$, free-electron abundance $Y_e$, atomic-weight mass correction $\sum_i Y_i W_i$, and Coulomb coupling moments in a single $O(K)$ pass:

$$S_X = \sum_{i=1}^K X_i, \qquad S_Y = \sum_{i=1}^K \frac{X_i}{A_i}, \qquad \bar{A} = \frac{S_X}{S_Y}, \qquad \bar{Z} = \bar{A} \sum_{i=1}^K \frac{Z_i X_i}{A_i}, \qquad Y_e = \frac{\bar{Z}}{\bar{A}},$$

$$\langle Z^2 \rangle = \bar{A} \sum_{i=1}^K \frac{Z_i^2 X_i}{A_i}, \qquad \langle Z^{5/3} \rangle = \bar{A} \sum_{i=1}^K \frac{Z_i^{5/3} X_i}{A_i}, \qquad \mathrm{mass\_correction} = \sum_{i=1}^K \frac{W_i X_i}{A_i},$$

together with the exact analytical partial derivatives with respect to each species mass fraction $X_i$:

$$\frac{\partial \bar{A}}{\partial X_i} = \frac{\bar{A}\,(A_i - \bar{A})}{A_i\,S_X}, \qquad \frac{\partial \bar{Z}}{\partial X_i} = \frac{\bar{A}\,(Z_i - \bar{Z})}{A_i\,S_X}.$$

### 1.2 Nuclear Binding Energy & Q-Values
For any nuclear reaction $\sum_r \nu_r \, {}^{A_r}\mathrm{El}_r \to \sum_p \nu_p \, {}^{A_p}\mathrm{El}_p$, `reaction_qtotal` evaluates the exact relativistic mass-excess energy release $Q_{\mathrm{total}} = \sum_r \nu_r \Delta M_r - \sum_p \nu_p \Delta M_p$ ($\mathrm{MeV}$).

=== "Python & JAX (`flexmesa.chem`)"

    ```python
    import jax.numpy as jnp
    import numpy as np
    from flexmesa import chem

    # 1. Lookup nuclide properties and solar mixtures (Tier A Safe Rust catalog)
    he4 = chem.get_nuclide_info("he4")
    cno_frac = chem.solar_cno_fraction("gs98")
    print(f"He4 W = {he4.w:.6f} u, GS98 CNO/Z = {cno_frac:.6f}")

    # 2. Vectorized composition moments across 1024 zones (PyO3 or @jax.jit)
    isos = ("h1", "he4", "c12", "o16")
    x_grid = np.tile(np.array([0.70, 0.28, 0.008, 0.012]), (1024, 1))
    comp_cpu = chem.get_composition_info_zones(isos, x_grid, backend="rust")
    comp_gpu = chem.get_composition_info_zones(isos, jnp.asarray(x_grid))
    ```

=== "Safe Rust (`crates/flexmesa-chem`)"

    ```rust
    use flexmesa_chem::{
        NuclideCatalog, SolarScale, get_composition_info, solar_cno_fraction,
    };

    let cat = NuclideCatalog::builtin();
    let ids: Vec<usize> = ["h1", "he4", "c12", "o16"]
        .iter()
        .map(|s| cat.get_nuclide_index(s).unwrap())
        .collect();
    let x = [0.70, 0.28, 0.008, 0.012];
    let info = get_composition_info(&cat, &ids, &x, true).unwrap();
    let cno_agss09 = solar_cno_fraction(SolarScale::Agss09, &cat);
    ```

---

## 2. Partial Ionization & Effective Ionic Charge (`ionization`)

In dense, partially ionized stellar interiors and white dwarf envelopes, pressure ionization and plasma screening modify the effective ionic charge $Z_{\mathrm{eff}}$ of trace heavy elements and alter thermodynamic and transport properties. `flexmesa.ionization` ports all three MESA ionization subsystems:

1. **2D Bicubic Spline Evaluation of $\mathrm{Fe}^{56}$ in a $\mathrm{He}^4$ Plasma (`eval_charge_of_fe56_in_he4`)**:
   - Interpolates $Z_{\mathrm{eff}}(\log_{10} n_e, \log_{10} T)$ over a $105 \times 30$ Thomas-Fermi-Dirac-Hubbard grid spanning $\log_{10}(n_e / \mathrm{cm}^{-3}) \in [24.0, 31.0]$ and $\log_{10}(T / \mathrm{K}) \in [6.0, 8.0]$ using pre-validated 2D bicubic Hermite splines (`interp_evbicub_prevalidated_db`), returning $(Z_{\mathrm{eff}}, \partial Z_{\mathrm{eff}}/\partial \log_{10} n_e, \partial Z_{\mathrm{eff}}/\partial \log_{10} T)$.
2. **4D Composition- and Density-Temperature Ionization Tables (`eval_ionization`)**:
   - Performs bilinear interpolation across metallicity $Z \in \{0.00, 0.02, 0.04, 0.20, 1.00\}$ and hydrogen mass fraction $X \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$ over $(X, Z)$ bicubic spline subtables on a $57 \times 36$ $(\log_{10} Q, \log_{10} T)$ grid, where the rescaled density coordinate is:
     $$\log_{10} Q = \log_{10} \rho - 2\,\log_{10} T + 12,$$
     returning all 28 ionic and plasma state quantities (`logPgas`, `logpe`, `logpp_H..Fe`, `Z_H..Fe`, `fneut_H..Fe`).
3. **Continuum Depression & Diffusion Typical Charges (`chi_info`, `chi_effective`, `eval_typical_charge`)**:
   - Implements Paquette et al. (1986) and Dupuis et al. (1992) continuum depression of ionization potentials alongside Allen (1973) whole-number ionic charges for atomic diffusion.

![Four-way Wave 2.5 runtime comparison and Fe56 in He4 effective charge profiles](../assets/figures/wave2_benchmark_staircase.png)

---

## 3. Synthetic Stellar Spectra & Bolometric Corrections (`colors`)

To compare evolutionary tracks directly with observational color-magnitude diagrams, `flexmesa.colors` synthesizes stellar spectral energy distributions (SEDs), bolometric magnitudes $M_{\mathrm{bol}}$, bolometric fluxes $F_{\mathrm{bol}}$, normalized grid interpolation distances `Interp_rad`, and synthetic magnitudes across the 7 Johnson photometric passbands (`B, I, J, M, R, U, V`, 10 output columns total) in Vega, AB, and ST zero-point systems from the Kurucz (2003, `lcb98cor.dat`) stellar atmosphere atlas ($76 \times 11 \times 8$ grid over $T_{\mathrm{eff}} \in [3500, 50000]\,\mathrm{K}$, $\log g \in [0.0, 5.0]$, $[\mathrm{M/H}] \in [-5.0, 1.0]$ at $N_\lambda = 1199$ wavelengths).

### 3.1 32-Gather Deduplicated 3D Cubic Hermite SED Synthesis
Upstream MESA evaluates each stellar SED by performing a 64-point (`4 x 4 x 4`) 3D cubic Hermite interpolation at every wavelength $\lambda_m$. Because the three 1D Catmull-Rom / finite-difference tangent operators share the 8 interior cell corners $(i_x + k_x, i_y + k_y, i_z + k_z)$ for $k_x, k_y, k_z \in \{0, 1\}$, `FlexMESA` factors the interior corner value weight $w_{\mathrm{val}}$ and the three tangent weights $(w_{dx}, w_{dy}, w_{dz})$ into a single combined weight:

$$w_{\mathrm{corner}}(k_x, k_y, k_z) = h_{00}(t_x)\text{ or }h_{01}(t_x)\;\cdots\; + w_{dx} + w_{dy} + w_{dz},$$

reducing memory gathers per model from 56 to **32 distinct grid spectra** (a 43% reduction in memory traffic) while preserving 16-digit bit-level accuracy ($3.57 \times 10^{-16}$ relative error).

### 3.2 Precomputed Non-Uniform Simpson Panel Weights
For each filter transmission curve $S_k(\lambda)$ resampled onto the 1199-point Kurucz wavelength grid, the photon-counting filter flux integral and energy-counting bolometric integral are integrated via non-uniform quadratic Simpson panels over $(x_{2p}, x_{2p+1}, x_{2p+2})$ with step widths $h_0 = x_{2p+1} - x_{2p}$ and $h_1 = x_{2p+2} - x_{2p+1}$:

$$I = \sum_{p=0}^{598} \left(c_{0,p} y_{2p} + c_{1,p} y_{2p+1} + c_{2,p} y_{2p+2}\right), \qquad \begin{cases} c_{0,p} = \dfrac{h_0 + h_1}{6 h_0}(2 h_0 - h_1), \\[6pt] c_{1,p} = \dfrac{(h_0 + h_1)^3}{6 h_0 h_1}, \\[6pt] c_{2,p} = \dfrac{h_0 + h_1}{6 h_1}(2 h_1 - h_0). \end{cases}$$

By precomputing $(c_{0,p}, c_{1,p}, c_{2,p})$ and the wavelength-weighted filter vectors $\lambda_m S_k(\lambda_m)$ once at table initialization and fusing the bolometric and 7 Johnson filter integrals over a stack-allocated `[T; 1199]` buffer, `FlexMESA` accelerates 1024-model SED and 10-column photometry synthesis by **$68.9\times$ in 4-thread Safe Rust** and **$151.6\times$ in GPU JAX** relative to upstream Fortran MESA.

![Kurucz emergent SEDs with Johnson filter passbands and synthetic B-V color indices](../assets/figures/colors_kurucz_sed_johnson.png)
