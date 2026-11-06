# Wave 3a Technical Specification: Analytical & Single-Table EOS Core (`Ideal`, `HELM`, `PC`, `Skye` & Quintic Blending)

## 1. Scope and Architectural Placement

Wave 3a ports the analytical and single-table equation-of-state (`eos`) foundation of MESA into `FlexMESA` under the **Integrated Three-Tier Hybrid Architecture** (`DECISIONS #6` through `#8`):

1. **Two-Variable Third-Order Automatic Differentiation (`crates/flexmesa-autodiff`)**:
   - **Tier A / B Foundation**: Extends `flexmesa-autodiff` with `AutoDiffReal2VarOrder3`, a 10-component two-variable third-order hyperdual number storing $(f, \partial_1 f, \partial_2 f, \partial_{11} f, \partial_{12} f, \partial_{22} f, \partial_{111} f, \partial_{112} f, \partial_{122} f, \partial_{222} f)$ along with order-reduction operators `differentiate_1()` and `differentiate_2()`. This enables exact Maxwell-consistent thermodynamic state assembly from the specific Helmholtz free energy $F(T, \rho)$ in `Skye`.
2. **Core EOS Schema, Ideal Gas, Ion Offsets, Blending & Fermi-Dirac Integrals (`crates/flexmesa-eos` + `src/flexmesa/eos/`)**:
   - **Tier B Differentiable Microphysics**: Implements the 26-channel MESA thermodynamic state vector (`EosEvalOutput`), neutral ground-state ionization energy offsets (`ion_offset`), ideal gas + radiation thermodynamics (`ideal`), $C^2$ quintic smootherstep and 2D polygon signed-distance blending (`eos_blend`), and the Aparicio (1998) four-subinterval Gauss-Legendre / Gauss-Laguerre relativistic Fermi-Dirac integral quadrature (`gauss_fermi`).
3. **Helmholtz (`HELM`), Potekhin-Chabrier (`PC`) & `Skye` Equations of State (`crates/flexmesa-eos` + `src/flexmesa/eos/`)**:
   - **Tier B Differentiable Microphysics**: Implements the Timmes & Swesty (2000) biquintic (`h5`) and bicubic (`h3`) Hermite Helmholtz EOS (`helm`), the Potekhin & Chabrier (2000, 2010, 2013) degenerate Coulomb liquid and BCC solid EOS (`pc`), the Jermyn et al. (2021) `Skye` third-order free-energy minimization EOS with smoothed latent heat across crystallization (`skye`), and Newton/Brent $(P_{\mathrm{gas}}, T)$ and $(S, T) / (E, T)$ thermodynamic inversions (`eosPT_get`, `eosDT_get_Rho`, `eosDT_get_T`) with Implicit Function Theorem partial derivatives across Safe Rust (`flexmesa._rust`) and `@jax.jit` (`float64`).

---

## 2. Index and Memory-Order Contract

Every Wave 3a module translates upstream Fortran MESA's 1-based column-major conventions into 0-based C-contiguous row-major layouts at the public API boundary while preserving exact channel ordering for upstream golden verification:

| Quantity | Upstream Fortran MESA | FlexMESA (Safe Rust & Python/JAX) |
| :--- | :--- | :--- |
| **Basic EOS result vector (`res`)** | `res(1:26)` (`i_lnPgas = 1 .. i_frac_ideal = 26`) | `[f64; 26]` / shape `(..., 26)` (`0: lnPgas .. 25: frac_ideal`) |
| **Composition derivatives (`d_dxa`)** | `d_dxa(1:2, 1:species)` (column-major: `1=lnPgas, 2=lnE`) | C-contiguous shape `(..., species, 2)` or `[ [dlnPgas_dxa, dlnE_dxa]; species ]` |
| **Polygon vertices (`eos_blend`)** | `coords(1:n, 1:2)` (column-major) | C-contiguous slice `&[[f64; 2]]` / shape `(n, 2)` |
| **HELM electron-positron table** | `f(1:imax, 1:jmax)` over $(\log_{10}(\rho Y_e), \log_{10} T)$ | C-contiguous row-major `(imax, jmax)` with cell-local 36 `h5` and 16 `h3` weights |
| **PC screening tables (`EXCOR7`, `FSCRliq8`)** | `tbl(1:4, 1:nvals, 1:nlnRS, 1:nlnGAME)` | C-contiguous row-major `(nvals, nlnRS, nlnGAME, 4)` |

### 2.1 The 26-Element `EosBasicIndex` Vector
For any state $(\rho, T, \{X_k\})$, the 26-element thermodynamic vector `res[0..26]` (together with logarithmic partial derivatives `d_dlnd[0..26]` $= (\partial \mathrm{res}/\partial \ln\rho)_{T, X}$ and `d_dlnT[0..26]` $= (\partial \mathrm{res}/\partial \ln T)_{\rho, X}$) stores:

1. `0` (`i_lnPgas`): $\ln P_{\mathrm{gas}}$ ($\mathrm{dyn\,cm^{-2}}$)
2. `1` (`i_lnE`): $\ln E$ ($\mathrm{erg\,g^{-1}}$, total specific internal energy including radiation)
3. `2` (`i_lnS`): $\ln S$ ($\mathrm{erg\,g^{-1}\,K^{-1}}$, total specific entropy including radiation)
4. `3` (`i_mu`): Mean molecular weight $\mu$ (dimensionless)
5. `4` (`i_lnfree_e`): $\ln Y_e^{\mathrm{free}} = \ln(n_e^{\mathrm{free}} / (N_A \rho))$
6. `5` (`i_eta`): Electron degeneracy parameter $\eta_e = \mu_e / (k_{\mathrm{B}} T)$
7. `6` (`i_grad_ad`): Adiabatic temperature gradient $\nabla_{\mathrm{ad}} = (\partial \ln T / \partial \ln P)_S$
8. `7` (`i_chiRho`): Density susceptibility $\chi_\rho = (\partial \ln P / \partial \ln \rho)_{T}$
9. `8` (`i_chiT`): Temperature susceptibility $\chi_T = (\partial \ln P / \partial \ln T)_{\rho}$
10. `9` (`i_Cp`): Specific heat at constant pressure $C_P$ ($\mathrm{erg\,g^{-1}\,K^{-1}}$)
11. `10` (`i_Cv`): Specific heat at constant volume $C_V = (\partial E / \partial T)_\rho$ ($\mathrm{erg\,g^{-1}\,K^{-1}}$)
12. `11` (`i_dE_dRho`): $(\partial E / \partial \rho)_T$ ($\mathrm{erg\,cm^3\,g^{-2}}$)
13. `12` (`i_dS_dT`): $(\partial S / \partial T)_\rho$ ($\mathrm{erg\,g^{-1}\,K^{-2}}$)
14. `13` (`i_dS_dRho`): $(\partial S / \partial \rho)_T$ ($\mathrm{erg\,cm^3\,g^{-2}\,K^{-1}}$)
15. `14` (`i_gamma1`): First adiabatic exponent $\Gamma_1 = (\partial \ln P / \partial \ln \rho)_S$
16. `15` (`i_gamma3`): Third adiabatic exponent $\Gamma_3 = 1 + (\partial \ln T / \partial \ln \rho)_S$
17. `16` (`i_phase`): Crystallization phase order parameter $\phi \in [0, 1]$ ($0 =$ liquid, $1 =$ solid)
18. `17` (`i_latent_ddlnT`): Smoothed latent heat temperature derivative $T (\partial S_{\mathrm{latent}} / \partial \ln T)_\rho$ ($\mathrm{erg\,g^{-1}}$)
19. `18` (`i_latent_ddlnRho`): Smoothed latent heat density derivative $T (\partial S_{\mathrm{latent}} / \partial \ln \rho)_T$ ($\mathrm{erg\,g^{-1}}$)
20. `19..26` (`i_frac_HELM .. i_frac_ideal`): Component blend weights $(f_{\mathrm{HELM}}, f_{\mathrm{OPAL/SCVH}}, f_{\mathrm{FreeEOS}}, f_{\mathrm{PC}}, f_{\mathrm{Skye}}, f_{\mathrm{CMS}}, f_{\mathrm{ideal}})$ summing to unity.

---

## 3. Module 3a.1: Core Algebra, Ideal EOS, Blending & Fermi-Dirac Integrals

### 3.1 Two-Variable Third-Order Dual Numbers (`AutoDiffReal2VarOrder3`)
An `AutoDiffReal2VarOrder3` instance represents the truncated bivariate Taylor jet of $f(x_1, x_2)$ up to total degree 3:

$$f(x_1 + \delta_1, x_2 + \delta_2) = f_0 + f_1 \delta_1 + f_2 \delta_2 + \frac{1}{2} f_{11} \delta_1^2 + f_{12} \delta_1 \delta_2 + \frac{1}{2} f_{22} \delta_2^2 + \frac{1}{6} f_{111} \delta_1^3 + \frac{1}{2} f_{112} \delta_1^2 \delta_2 + \frac{1}{2} f_{122} \delta_1 \delta_2^2 + \frac{1}{6} f_{222} \delta_2^3.$$

For any unary composition $g(f(x_1, x_2))$ with scalar derivatives $g', g'', g'''$ evaluated at $f_0$:

$$\partial_i g = g' f_i, \qquad \partial_{ij} g = g'' f_i f_j + g' f_{ij},$$

$$\partial_{ijk} g = g''' f_i f_j f_k + g'' \left(f_{ij} f_k + f_{ik} f_j + f_{jk} f_i\right) + g' f_{ijk}.$$

The partial differentiation operators `differentiate_1(f)` and `differentiate_2(f)` shift the first-, second-, and third-order partials of an `AutoDiffReal2VarOrder3` value into a new `AutoDiffReal2VarOrder3` value (with zeroed third-order slots) representing $\partial f / \partial x_1$ and $\partial f / \partial x_2$:

$$\operatorname{differentiate\_1}(f) = \left(f_1,\, f_{11},\, f_{12},\, f_{111},\, f_{112},\, f_{122},\, 0,\, 0,\, 0,\, 0\right), \qquad \operatorname{differentiate\_2}(f) = \left(f_2,\, f_{12},\, f_{22},\, f_{112},\, f_{122},\, f_{222},\, 0,\, 0,\, 0,\, 0\right).$$

### 3.2 Neutral Ideal Gas, Radiation & Ground-State Ionization Offsets (`ideal`, `ion_offset`)
1. **Neutral Ideal Gas + Radiation (`eval_ideal_eos`, `eval_ideal_eos_from_weights`)**:
   Following upstream `eos/private/ideal.f90` (`get_ideal_for_eosdt` and `ideal_eos`), species with mass fraction $X_k > 10^{-10}$ are normalized to number fractions $Y_k = (X_k / A_k) / \sum_j (X_j / A_j)$ and evaluated via the Sackur-Tetrode ideal ion Helmholtz free energy `compute_F_ideal_ion` (`skye_ideal.f90`) with zero electron free energy (`etaele = -1e99`, `xnefer = 1e-20`), explicit Planckian radiation terms ($P_{\mathrm{rad}} = \frac{1}{3} a_{\mathrm{rad}} T^4$, $E_{\mathrm{rad}} = a_{\mathrm{rad}} T^4 / \rho$, $S_{\mathrm{rad}} = \frac{4}{3} a_{\mathrm{rad}} T^3 / \rho$), and neutral mean molecular weight `res[i_mu] = abar` (`res[i_frac_ideal] = 1.0`). All 26 channels and analytical $(\ln\rho, \ln T)$ derivatives are reduced via `pack_for_export` in `AutoDiffReal2VarOrder3`.
2. **Ground-State Ionization Energy Offset (`compute_ion_offset`)**:
   Using the 28-element NIST total atomic binding energy table $B_{\mathrm{NIST}}(Z)$ ($1 \le Z \le 28$, $\mathrm{H}$ through $\mathrm{Ni}$, in $\mathrm{eV}$):
   $$\Delta E_{\mathrm{ion}} = \frac{\mathrm{eV\_to\_ergs}}{\mathrm{amu}} \sum_{k=1}^S \frac{X_k}{A_k}\,B_{\mathrm{NIST}}\!\left(\min(Z_k, 28)\right).$$

### 3.3 $C^2$ Quintic Smootherstep & 2D Polygon Signed-Distance Blending (`eos_blend`)
1. **Quintic Smootherstep (`quintic_blend`, `blend_eos_results`)**:
   Given linear blend fraction $\alpha_0 \in [0, 1]$ (carried as an `AutoDiffReal2VarOrder1` over $\ln\rho$ and $\ln T$), the $C^2$-continuous quintic weight is:
   $$\alpha(\alpha_0) = 10\alpha_0^3 - 15\alpha_0^4 + 6\alpha_0^5 = \alpha_0^3\left(10 + \alpha_0(-15 + 6\alpha_0)\right), \qquad \frac{d\alpha}{d\alpha_0} = 30\alpha_0^2(1 - \alpha_0)^2.$$
   Blending two EOS evaluations $(\mathrm{res}_A, \mathrm{res}_B)$ propagates exact boundary transition derivatives:
   $$\mathrm{res} = \alpha\,\mathrm{res}_A + (1 - \alpha)\,\mathrm{res}_B, \qquad \frac{\partial \mathrm{res}}{\partial \ln x} = \alpha\frac{\partial \mathrm{res}_A}{\partial \ln x} + (1 - \alpha)\frac{\partial \mathrm{res}_B}{\partial \ln x} + \frac{\partial \alpha}{\partial \ln x}\left(\mathrm{res}_A - \mathrm{res}_B\right).$$
2. **2D Polygon Containment and Smoothed Distance (`is_contained`, `min_distance_to_polygon`)**:
   For a polygon with vertices $\mathbf{v}_0, \dots, \mathbf{v}_{n-1} \in \mathbb{R}^2$ and query point $\mathbf{p} \in \mathbb{R}^2$:
   - `is_contained` evaluates the winding number across all $n$ directed edges (returning `true` when the winding number is non-zero).
   - For each edge $\mathbf{e}_i = \mathbf{v}_{i+1} - \mathbf{v}_i$ with projection parameter $\lambda_i = \frac{(\mathbf{p} - \mathbf{v}_i)\cdot \mathbf{e}_i}{\|\mathbf{e}_i\|^2}$, evaluates the exact endpoint distance $\|\mathbf{p} - \mathbf{v}_i\|$ when $\lambda_i < 0$ or $\|\mathbf{p} - \mathbf{v}_{i+1}\|$ when $\lambda_i > 1$, and on the interior segment $\lambda_i \in [0, 1]$ ($\mathbf{x}_i = \mathbf{v}_i + \lambda_i \mathbf{e}_i$) adds the regularization floor $\epsilon = \mathrm{float32}(10^{-10})$ in quadrature:
     $$d_i(\mathbf{p}) = \sqrt{\|\mathbf{x}_i - \mathbf{p}\|^2 + \epsilon^2} \quad (\lambda_i \in [0, 1]), \qquad d(\mathbf{p}) = \min_{0 \le i < n} d_i(\mathbf{p}).$$

### 3.4 Aparicio (1998) Generalized Relativistic Fermi-Dirac Integrals (`gauss_fermi`)
Evaluates the generalized Fermi-Dirac integral and its $(\eta, \theta)$ partial derivatives:

$$F_k(\eta, \theta) = \int_0^\infty \frac{x^k \sqrt{1 + \frac{1}{2}\theta x}}{1 + \exp(x - \eta)}\,dx, \qquad \frac{\partial F_k}{\partial \eta} = \int_0^\infty \frac{x^k \sqrt{1 + \frac{1}{2}\theta x}}{\left(1 + \exp(x - \eta)\right)\left(1 + \exp(\eta - x)\right)}\,dx,$$

$$\frac{\partial F_k}{\partial \theta} = \frac{1}{4}\int_0^\infty \frac{x^{k+1}}{\sqrt{1 + \frac{1}{2}\theta x}\,\left(1 + \exp(x - \eta)\right)}\,dx,$$

by mapping $\eta \mapsto \xi = \sigma^{-1}\ln(1 + e^{\sigma(\eta - d)})$ ($d = 3.3609, \sigma = 0.091186$), computing rational breakpoints $s_1 < s_2 < s_3$, and applying 20-point Gauss-Legendre quadrature on $[0, \sqrt{s_1}]$ (under $x = z^2$), $[s_1, s_2]$, and $[s_2, s_3]$, followed by 20-point Gauss-Laguerre quadrature on $[s_3, \infty)$.

---

## 4. Module 3a.2: Timmes & Swesty Helmholtz EOS (`HELM`)

### 4.1 Biquintic (`h5`) and Bicubic (`h3`) Hermite Tensor Interpolation
On each cell $[d_i, d_{i+1}] \times [T_j, T_{j+1}]$ of the $\log_{10}(\rho Y_e) \times \log_{10} T$ grid:
1. **Biquintic Electron-Positron Free Energy (`h5`)**:
   Using quintic Hermite basis functions $\psi_0(z), \psi_1(z), \psi_2(z)$ on $z \in [0, 1]$ and the 9 nodal values $(f, f_d, f_t, f_{dd}, f_{tt}, f_{dt}, f_{ddt}, f_{dtt}, f_{ddtt})$ at the four cell corners ($36$ coefficients `fi[0..36]`), evaluates the specific electron-positron Helmholtz free energy $F_{ep}(\rho, T)$ and all partial derivatives up to 3rd order ($F_d, F_t, F_{dd}, F_{tt}, F_{dt}, F_{ddd}, F_{ddt}, F_{dtt}, F_{ttt}$).
2. **Bicubic Cancellation-Guard, Chemical Potential & Number Density (`h3`)**:
   Using cubic Hermite basis functions $\psi_0^{(3)}(z), \psi_1^{(3)}(z)$ and the 4 nodal values at the four cell corners ($16$ coefficients), interpolates $\partial P_{ep}/\partial\rho$ (`dpdf`), electron degeneracy $\eta_e$ (`ef`), and net free electron number density $x_{ne}$ (`xf`) with their first and second partial derivatives.

### 4.2 Sackur-Tetrode Ions, Yakovlev-Shalybkov Coulomb Corrections & Maxwell Relations
Combines Planckian radiation (`helm_radiation`), Sackur-Tetrode ideal ions with low-entropy exponential rolloff (`helm_ideal_ions`), biquintic electron-positron pairs (`helm_electron_positron`), and Yakovlev & Shalybkov (1989) strong/weak Coulomb corrections (`helm_coulomb2`, with smooth linear transitions over $\Gamma \in [0.8, 1.0]$ and $\rho Y_e / \rho_{\mathrm{deg}}$), and verifies thermodynamic consistency via the three dimensionless Maxwell residuals:

$$d_{se} = \frac{T(\partial S/\partial T)_\rho}{(\partial E/\partial T)_\rho} - 1, \qquad d_{pe} = \frac{\rho^2(\partial E/\partial \rho)_T + T(\partial P/\partial T)_\rho}{P} - 1, \qquad d_{sp} = -\frac{\rho^2(\partial S/\partial \rho)_T}{(\partial P/\partial T)_\rho} - 1.$$

---

## 5. Module 3a.3: Potekhin-Chabrier (`PC`), `Skye` & Inverse Solvers

### 5.1 Potekhin-Chabrier (`PC`) Degenerate Plasma & Crystal EOS
Evaluates degenerate multi-ion plasma thermodynamics in `AutoDiffReal2VarOrder1` over $(T, \rho)$:
1. **Electron Degeneracy & Thermodynamics (`CHEMFIT7`, `FERINV7`, `ELECT11`)**:
   Computes relativistic Fermi momentum $p_{F0} = (3\pi^2 n_e)^{1/3}(\hbar/m_e c)$, Fermi temperature $T_F = \sqrt{1 + p_{F0}^2} - 1$, degeneracy index $\theta = T/T_F$, Antia (1993) inverse non-relativistic Fermi integral $X_{1/2}(\frac{2}{3}\theta^{-3/2})$ (`FERINV7`), and Chabrier & Potekhin (1998) relativistic correction $\chi = X_{1/2} - \frac{3}{2}\ln C_T(\theta, T)$, followed by Blinnikov (`BLIN9`) and Sommerfeld (`SOMMERF`) electron gas thermodynamics blended across $\chi = 28$.
2. **Coulomb Liquid & Solid Mixtures (`MELANGE9`, `EOSFI8`)**:
   Combines OCP liquid (`FITION9` + Wigner-Kirkwood $\hbar^2$ diffraction $F_{\mathrm{WK}} = \frac{1}{24}(T_p/T)^2$), BCC harmonic and anharmonic crystal (`FHARM12`/`HLfit12` + `ANHARM8`), electron exchange-correlation (`EXCOR7`), electron-ion liquid and solid screening (`FSCRliq8`, `FSCRsol8`), and linear-mixing deviations (`CORMIX`), blended linearly over $\Gamma_i \in [\Gamma_{\mathrm{start}}, \Gamma_{\mathrm{full}}]$.

### 5.2 Jermyn et al. (2021) `Skye` EOS with Third-Order Free-Energy Minimization
Evaluates the total specific Helmholtz free energy $F(T, \rho) \in \mathrm{AutoDiffReal2VarOrder3}$:

$$F(T, \rho) = F_{\mathrm{rad}}(T, \rho) + F_{\mathrm{ideal,ion}}(T, \rho) + F_{\mathrm{ele}}(T, \rho) + F_{\mathrm{xc}}(T, \rho) - T S_{\mathrm{mix}} + F_{\mathrm{Coul}}(T, \rho),$$

where $F_{\mathrm{ele}}(T, \rho)$ is packed directly from the biquintic `HELM` interpolator into `AutoDiffReal2VarOrder3`, $F_{\mathrm{liq}}$ includes the Baiko & Yakovlev (2019) non-perturbative quantum liquid correction and Potekhin & Chabrier (2013) liquid screening, and $F_{\mathrm{sol}}$ includes the harmonic/anharmonic BCC crystal, solid screening, and Medin & Cumming (2010) multi-ion mixing (`"PC"` or `"Ogata"`). At the crystallization transition, `decide_phase` computes the smoothed phase parameter and latent heat derivatives:

$$\Delta = \frac{k_{\mathrm{B}} T}{100 \bar{A}\,\mathrm{amu}}, \qquad \phi = \frac{1}{1 + \exp\!\left(-(F_{\mathrm{liq}} - F_{\mathrm{sol}})/\Delta\right)},$$

$$F_{\mathrm{excess}} = \left((1 - \phi)F_{\mathrm{liq}} + \phi F_{\mathrm{sol}}\right) - \min(F_{\mathrm{liq}}, F_{\mathrm{sol}}), \qquad S_{\mathrm{latent}} = -\operatorname{differentiate\_1}(F_{\mathrm{excess}}),$$

$$\mathrm{latent\_ddlnT} = T^2 \operatorname{differentiate\_1}(S_{\mathrm{latent}}), \qquad \mathrm{latent\_ddlnRho} = T\rho \operatorname{differentiate\_2}(S_{\mathrm{latent}}).$$

All 26 MESA outputs and their $(\ln\rho, \ln T)$ derivatives follow directly from $S_{\mathrm{gas}} = -\operatorname{differentiate\_1}(F_{\mathrm{gas}})$ and $P_{\mathrm{gas}} = \rho^2 \operatorname{differentiate\_2}(F_{\mathrm{gas}})$, guaranteeing exact Maxwell consistency ($|d_{se}|, |d_{pe}|, |d_{sp}| \le 10^{-14}$).

### 5.3 Thermodynamic Root Inversions (`eosPT_get`, `eosDT_get_Rho`, `eosDT_get_T`)
Given target $y_{\mathrm{target}}$ (such as $\log_{10} P_{\mathrm{gas}}$, $\ln S$, $\ln E$, or $\log_{10} P_{\mathrm{total}}$), solves $y(x) - y_{\mathrm{target}} = 0$ for $x = \log_{10}\rho$ or $x = \log_{10} T$ using safeguarded Newton-Raphson with analytic derivative $\partial y / \partial x$ and Brent fallback, and applies the Implicit Function Theorem:

$$\left(\frac{\partial \ln \rho}{\partial \ln P_{\mathrm{gas}}}\right)_T = \frac{1}{\chi_{\rho,\mathrm{gas}}}, \qquad \left(\frac{\partial \ln \rho}{\partial \ln T}\right)_{P_{\mathrm{gas}}} = -\frac{\chi_{T,\mathrm{gas}}}{\chi_{\rho,\mathrm{gas}}}.$$

---

## 6. Verification and Wave 3a.5 Benchmark Gates

1. **`eos_blend` Golden Gate**: Match all 5 polygon containment and signed-distance golden points in `eos/test/test_output` to $\le 10^{-15}$.
2. **`gauss_fermi` Golden Gate**: Match `test_dirac_integrals` (`theta_e = 0.929542529701454`) and 16-digit `gfortran` oracle outputs across $\eta \in [-10, 50]$ to $\le 10^{-14}$.
3. **`HELM` Golden Gate**: Match all 4 compositions (`pure Helium`, `pure Hydrogen`, `mixed Z with H&He 3:1 ratio`, `solar`) at Regime 4 ($T = 2 \times 10^8\,\mathrm{K}, \rho = 100\,\mathrm{g\,cm^{-3}}$) in `eos/test/test_output` to $\le 10^{-14}$.
4. **`PC` & `Skye` Golden Gate**: Match upstream Fortran MESA `Get_PC_EOS_Results` and `Get_Skye_EOS_Results` across liquid, crystallizing, and solid BCC regimes to $\le 10^{-13}$ and verify `Skye` Maxwell residuals $\le 10^{-14}$.
5. **Wave 3a.5 Upstream Fortran MESA Gate**: Link and benchmark upstream `libeos.a` against Safe Rust, `PyO3`, and `@jax.jit`, and sync the MkDocs Material website.
