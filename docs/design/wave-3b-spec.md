# Wave 3b Technical Specification: 4D Tabulated EOS, Opacity, Nuclear Rates, Reaction Networks & Model Atmospheres (`eos_4d`, `kap`, `rates`, `net`, `atm`)

## 1. Scope and Architectural Placement

Wave 3b completes the stellar microphysics layer of `FlexMESA` under the **Integrated Three-Tier Hybrid Architecture** (`DECISIONS #6` through `#8`):

1. **4D Tabulated Equations of State (`OPAL/SCVH`, `FreeEOS`, `CMS`) & Full 6-Component Cascade (`crates/flexmesa-eos` + `src/flexmesa/eos/`)**:
   - **Tier B Differentiable Microphysics**: Ports MESA's 4D $(X, Z, \log_{10} Q, \log_{10} T)$ bicubic spline interpolators (`eosDT_eval.f90`, `eosdt_load_tables.f90`, `cms.f90`, `free_eos.f90`, `opal_scvh.f90`) with 2D not-a-knot bicubic splines in $(\log_{10} Q, \log_{10} T)$ ($\log_{10} Q = \log_{10}\rho - 2\log_{10} T + 12$), quadratic/cubic Lagrange interpolation in $X$, linear interpolation in $Z$, additive `CMS` H/He volume-additivity mixing, the full 6-component `eosDT_get` priority cascade (`CMS -> PC -> Skye -> FreeEOS -> OPAL/SCVH -> HELM -> ideal`), and the analytic ideal $\gamma$-law $(\rho, E)$ inverse solver `eos_gamma_DE_get`.
2. **Radiative, Conductive & Compton Opacities (`crates/flexmesa-kap` + `src/flexmesa/kap/`)**:
   - **Tier B Differentiable Microphysics**: Ports `kap` (`kap_eval.f90`, `kap_eval_fixed.f90`, `kap_eval_co.f90`, `kap_eval_lt.f90`, `kap_cond.f90`, `compton.f90`) covering OPAL/OP Type 1 fixed-metal tables (`gn93`, `gs98`, `a09`, `op_gs98`, `op_a09_nans_removed_by_hand`), Type 2 C/O-enhanced 2D $(f_C, f_O)$ bilinear/biquadratic tables (`gn93_co`, `gs98_co`, `a09_co`), Ferguson et al. (2005), Freedman et al. (2008, 2014), Lederer & Aringer (2009) `kapCN`, and Marigo & Aringer (2009) `AESOPUS` low-$T$ molecular opacities, Potekhin (`condint.f`) and Blouin et al. (2020) H/He electron-conduction opacities, Poutanen (2017) relativistic/degenerate Compton scattering, and $C^2$ quintic smootherstep regime blending.
3. **Nuclear Reaction Rates, Plasma Screening & Weak Rates (`crates/flexmesa-rates` + `src/flexmesa/rates/`)**:
   - **Tier B Differentiable Microphysics**: Ports `rates` (`rates_def.f90`, `raw_rates.f90`, `reaclib_rates.f90`, `ratelib.f90`, `screen.f90`, `screen5.f90`, `rates_weak.f90`, `ecapture.f90`) covering the 316 built-in MESA reactions, 7-parameter REACLIB forward and detailed-balance reverse rate expansions over the 24-node `TPART` partition-function grid, Salpeter (1954), Graboske/Alastuey/Itoh (`screen5`), and Chugunov et al. (2007) plasma electrostatic screening, Langanke & Martinez-Pinedo (2000), Oda et al. (1994), Fuller, Fowler & Newman (1985), and Suzuki, Toki & Nomoto (2016) 2D bicubic weak-rate tables, and analytical/4-point Gauss-Laguerre Fermi-Dirac phase-space electron captures (`eval_ecapture`).
4. **Nuclear Reaction Networks, Analytical Jacobians & Stiff Burners (`crates/flexmesa-net` + `src/flexmesa/net/`)**:
   - **Tier C Coupled Solvers & Networks**: Ports `net` (`net_initialize.f90`, `net_derivs.f90`, `net_eval.f90`, `net_burn.f90`, `net_burn_const_p.f90`) parsing recursive MESA `.net` files (`basic.net`, `approx21.net`, `pp_and_cno_extras.net`, `cno_extras.net`, `co_burn.net`, `hot_cno.net`, etc.), assembling molar abundance derivatives $dY_i/dt$, nuclear and neutrino energy generation rates $(\epsilon_{\mathrm{nuc}}, \epsilon_{\mathrm{neu}})$, and exact analytical Jacobians $(\partial \dot{X}_i/\partial X_j, \partial \dot{X}_i/\partial T, \partial \dot{X}_i/\partial \rho)$ via 2-lane SIMD dual numbers (`F64x2`), paired with stiff Kaps-Rentrop (`ROS3P`) and Hairer-Wanner `RODAS4` Rosenbrock/Bader-Deuflhard single-zone constant-$(\rho, T)$ and constant-$P$ burners and a `jax.custom_vjp` wrapper (`eval_net_jax`).
5. **Stellar Model Atmospheres & Optical-Depth Integrations (`crates/flexmesa-atm` + `src/flexmesa/atm/`)**:
   - **Tier B Differentiable Microphysics**: Ports `atm` (`atm_lib.f90`, `t_tau_relations.f90`, `integrate_atm.f90`, `irradiated.f90`, `table_atm.f90`) implementing Eddington, solar Hopf, Krishna-Swamy (1966), and Trampedach et al. (2014) / Ball (2021) $T(\tau)$ relations, uniform-opacity fixed-point iteration (`eval_t_tau_uniform`), Cash-Karp RK45 $dP/d\tau = g/\kappa$ integration (`eval_t_tau_varying`, `build_t_tau_varying`), Guillot (2010) irradiated grey atmospheres (`eval_irradiated`) with Steffen cubic spline evaluation of $E_2(x)$, Paczynski (1969) optical-depth-diluted $\nabla_{\mathrm{rad}}$, and 2D bicubic/Steffen model atmosphere table interpolation (`eval_table`, `get_table_alfa_beta`).

---

## 2. Index and Memory-Order Contract

All Wave 3b modules translate Fortran 1-based column-major arrays into 0-based C-contiguous row-major layouts at the public API boundary while preserving exact physical channel ordering:

| Quantity | Upstream Fortran MESA | FlexMESA (Safe Rust & Python/JAX) |
| :--- | :--- | :--- |
| **4D EOS vertex spline (`f1`)** | `f(1:4, 1:nv, 1:num_logQs, 1:num_logTs)` (`nv = 16`) | `EosDtVertex { coeffs: [f64; 64] }` (`coeffs[4*k + m]` for variable `k`, moment `m`) |
| **Opacity regime fractions (`kap_fracs`)** | `kap_fracs(1:4)` (`i_frac_lowT = 1 .. i_frac_Compton = 4`) | `[f64; 4]` (`[low_t, high_t, type2, compton]`) |
| **REACLIB expansion coefficients** | `Chapter(1:N)`, `a(1:7, 1:N)`, `inv_part(1:24, 1:N)` | Row-major `(N, 7)` and `(N, 24)` `float64` arrays |
| **Network abundance Jacobian (`d_dxdt_dx`)** | `d_dxdt_dx(1:species, 1:species)` (column-major) | C-contiguous row-major `(species, species)` where entry `[i, j]` is $\partial \dot{X}_i / \partial X_j$ |
| **Atmosphere structure profile** | `atm_structure(1:num_results_for_create_atm, 1:npts)` | C-contiguous `(npts, 9)` (`[xm, delta_r, lnP, lnd, lnT, gradT, kap, gradr, grada]`) |

---

## 3. Mathematical Formulations

### 3.1 4D $(X, Z, \log_{10} Q, \log_{10} T)$ Tabulated EOS (`OPAL/SCVH`, `FreeEOS`, `CMS`)
Given $(\rho, T)$ and $\log_{10} Q = \log_{10}\rho - 2\log_{10} T + 12$:

1. **Cell Bicubic Interpolation (`do_eos_interpolations`)**:
   On each $(\log_{10} Q_i, \log_{10} T_j)$ cell of width $(h_x, h_y)$, all 16 tabulated thermodynamic channels $f_k(\log_{10} Q, \log_{10} T)$ and their grid partials $(\partial f_k/\partial \log_{10} Q, \partial f_k/\partial \log_{10} T)$ are evaluated from the 4 cell vertices and transformed to natural-logarithm $(\ln\rho, \ln T)$ partials via:
   $$\left(\frac{\partial f_k}{\partial \ln \rho}\right)_T = \frac{1}{\ln 10}\left(\frac{\partial f_k}{\partial \log_{10} Q}\right)_{\log_{10} T}, \qquad \left(\frac{\partial f_k}{\partial \ln T}\right)_\rho = \frac{1}{\ln 10}\left(\frac{\partial f_k}{\partial \log_{10} T}\right)_{\log_{10} Q} - 2\left(\frac{\partial f_k}{\partial \ln \rho}\right)_T.$$
2. **Composition $(X, Z)$ Interpolation (`get1_eosdt_results`)**:
   For each bounding metallicity plane $Z_1, Z_2$, `get1_eosdt_for_x` evaluates a 3-point quadratic or 4-point cubic Lagrange polynomial across equally spaced $X$ planes, followed by linear interpolation in $Z \in [Z_1, Z_2]$.

### 3.2 Radiative, Compton & Conductive Opacity Blending (`kap`)
1. **Temperature and Compton Blending**:
   Across $\log_{10} T \in [\log_{10} T_{\mathrm{low}}, \log_{10} T_{\mathrm{high}}]$ (`[3.80, 3.88]` by default), low-$T$ molecular opacity $\kappa_{\mathrm{low}}$ and high-$T$ OPAL/OP opacity $\kappa_{\mathrm{high}}$ are blended in $\log_{10}\kappa$ via the $C^2$ quintic smootherstep $\mathcal{S}_5(u) = u^3(10 - 15u + 6u^2)$ using `AutoDiffReal2VarOrder1` over $(\log_{10}\rho, \log_{10} T)$ so transition derivatives propagate automatically. At high $T$ or low $\log_{10} R = \log_{10}\rho - 3\log_{10} T + 18$, $\log_{10}\kappa_{\mathrm{rad}}$ blends smoothly into Poutanen (2017) Compton opacity $\kappa_{\mathrm{Compton}}(\theta, \eta)$.
2. **Harmonic Combination with Electron Conduction**:
   Total Rosseland mean opacity combines radiative opacity $\kappa_{\mathrm{rad}}$ and conductive opacity $\kappa_{\mathrm{cond}}$ harmonically:
   $$\frac{1}{\kappa} = \frac{1}{\kappa_{\mathrm{rad}}} + \frac{1}{\kappa_{\mathrm{cond}}}, \qquad \frac{\partial \ln \kappa}{\partial \ln x} = \frac{\kappa}{\kappa_{\mathrm{rad}}}\frac{\partial \ln \kappa_{\mathrm{rad}}}{\partial \ln x} + \frac{\kappa}{\kappa_{\mathrm{cond}}}\frac{\partial \ln \kappa_{\mathrm{cond}}}{\partial \ln x}.$$

### 3.3 Nuclear Reaction Rates, Plasma Screening & Stiff Networks (`rates`, `net`)
1. **REACLIB Rate Expansion**:
   Forward molar rates $\lambda(T_9)$ ($T_9 = T / 10^9\,\mathrm{K}$) sum exponential 7-term polynomials:
   $$\lambda(T_9) = \sum_m \exp\!\left(a_{m,0} + \frac{a_{m,1}}{T_9} + \frac{a_{m,2}}{T_9^{1/3}} + a_{m,3} T_9^{1/3} + a_{m,4} T_9 + a_{m,5} T_9^{5/3} + a_{m,6} \ln T_9\right),$$
   while reverse rates multiply by the nuclear partition-function ratio $Q_{\mathrm{part}}(T_9)$ log-interpolated over the 24-node `TPART` grid.
2. **Network Rate Assembly (`get_dydt`) and Analytical Jacobians**:
   For each reaction $r$ with reactant stoichiometry $(n_1, i_1), (n_2, i_2), (n_3, i_3)$ and screened molar rate $R_r(\rho, T)$, the forward-mode 2-lane SIMD dual vector `F64x2` accumulates $(r_r, \partial r_r/\partial \ln T, \partial r_r/\partial \ln\rho)$ into molar time derivatives $\dot{Y}_i$ and analytical abundance derivatives $\partial \dot{X}_i / \partial X_j = (A_i / A_j)\,\partial \dot{Y}_i / \partial Y_j$.

### 3.4 Stellar Model Atmospheres (`atm`)
For grey $T(\tau)$ relations $T^4(\tau) = \frac{3}{4} T_{\mathrm{eff}}^4 \left[\tau + q(\tau)\right]$, hydrostatic equilibrium $dP/d\tau = g / \kappa(P, T)$ is integrated either analytically under uniform opacity (`eval_t_tau_uniform` with fixed-point iteration on $\kappa(P_{\mathrm{surf}}, T_{\mathrm{surf}})$), numerically via Cash-Karp RK45 from $\tau_0 = 10^{-4}$ to $\tau_{\mathrm{base}}$ (`eval_t_tau_varying`), or interpolated from 3D $(Z, \log g, T_{\mathrm{eff}})$ precomputed atmosphere tables (`eval_table`) using 2D not-a-knot bicubic splines in $(\log g, T_{\mathrm{eff}})$ and 4-point Steffen monotonic cubic splines in $\log_{10}(Z / Z_{\odot,\mathrm{GN93}})$.

---

## 4. Verification and Wave 3b.5 Benchmark Gates

1. **Wave 3b.1 (`eos_4d`) Golden Gate**: Match `OPAL/SCVH`, `FreeEOS`, `CMS`, and full `eosDT_get` / `eosPT_get` / `eos_gamma_DE_get` golden checkpoints in `eos/test/test_output` to $\le 10^{-12}$.
2. **Wave 3b.2 (`kap`) Golden Gate**: Match `test_kap` (`fixed_metal`, `co`, `lowT`, `kapCN`, `AESOPUS`, `conduction`, `compton`) golden outputs in `kap/test/test_output` to $\le 10^{-7}$.
3. **Wave 3b.3 (`rates`) Golden Gate**: Match `test_rates`, `test_screen`, `test_weak`, and `test_ecapture` golden outputs in `rates/test/test_output` to $\le 10^{-12}$.
4. **Wave 3b.4 (`net` & `atm`) Golden Gate**: Match `net/test/test_output` and `atm/test/test_output` (`Eddington`, `solar_Hopf`, `Krishna_Swamy`, `Trampedach_solar`, `irradiated`, `table_atm`) to $\le 10^{-11}$.
5. **Wave 3b.5 Upstream Fortran MESA Gate**: Link and benchmark upstream `libeos.a`, `libkap.a`, `librates.a`, `libnet.a`, and `libatm.a` against Safe Rust, `PyO3`, and `@jax.jit`, verify 15/15 parity and performance gates pass, and sync the MkDocs Material website.
