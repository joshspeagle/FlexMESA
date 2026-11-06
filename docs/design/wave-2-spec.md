# Wave 2 Technical Specification: Atomic, Ionization & Photometric Microphysics (`chem`, `ionization`, `colors`)

## 1. Scope and Architectural Placement

Wave 2 ports MESA's three atomic and photometric microphysics packages (`chem`, `ionization`, and `colors`) under the **Integrated Three-Tier Hybrid Architecture** (`DECISIONS #6`–`#8`):

1. **`chem` (`crates/flexmesa-chem` + `src/flexmesa/chem/`)**:
   - **Tier A / B Hybrid**: Static element, reaction-category, solar-abundance, and nuclide catalogs (`NuclideData`, `ChemCatalog`, `NuclideSet`) are implemented canonically in Safe Rust (`#![forbid(unsafe_code)]`) and exposed zero-copy via `flexmesa._rust`. Shell-local composition moment reductions (`get_composition_info`, `basic_composition_info`, `composition_info`) provide both a zero-copy Safe Rust kernel (`backend="rust"`, scalar and persistent `rayon` multi-zone batch) and a branchless `@jax.jit` `float64` kernel (`backend="jax"`, `get_composition_info_jax`) behind `backend="auto" | "rust" | "jax"`.
2. **`ionization` (`crates/flexmesa-ionization` + `src/flexmesa/ionization/`)**:
   - **Tier B Differentiable Microphysics**: Implements the 2D Thomas-Fermi Debye-Huckel $^{56}\mathrm{Fe}$-in-$^{4}\mathrm{He}$ effective charge interpolator (`eval_charge_of_Fe56_in_He4`), the 4D $(Z, X, \log Q, \log T)$ 28-quantity ionization table evaluator (`eval_ionization`), and the Paquette et al. (1986) / Dupuis et al. (1992) continuum-depression typical ionic charge (`chi_info`, `chi_effective`, `eval_typical_charge`) across Safe Rust (`flexmesa._rust`) and `@jax.jit` (`float64`).
3. **`colors` (`crates/flexmesa-colors` + `src/flexmesa/colors/`)**:
   - **Tier B Differentiable Post-Processing & Photometry**: Implements 4D incomplete cubic Hermite spectral energy distribution (SED) interpolation (`hermite_interp_vector`), log-trilinear (`trilinear_interp_vector`) and $K=4$ inverse-distance-weighted nearest-neighbor (`construct_sed_knn`) interpolation, distance dilution (`dilute_flux`), composite non-uniform Simpson + trapezoidal integration (`simpson_integration`), bolometric photometry (`calculate_bolometric`), and photon-counting synthetic filter magnitudes (`calculate_synthetic`) in the Vega, AB, and ST photometric systems across Safe Rust (`flexmesa._rust`) and `@jax.jit` (`float64`).

---

## 2. Index and Memory-Order Contract

Every Wave 2 module translates upstream Fortran MESA's 1-based column-major conventions into 0-based C-contiguous row-major layouts at the public API boundary while preserving exact 1-based MESA identifier compatibility where required by upstream golden tests:

| Quantity | Upstream Fortran MESA | FlexMESA (Safe Rust & Python/JAX) |
| :--- | :--- | :--- |
| **Element ID (`el_id` / `Z`)** | `e_h = 1 .. e_cn = 112` (`el_name(0:112)` with `'neut'` at `0`) | `ElementId` / integer `1 .. 112` matching atomic number $Z$ (`0` for neutron) |
| **Nuclide ID (`chem_id`)** | `1 .. 7856` (`nuclide_not_found = -1`) | `1 .. 7856` for MESA `chem_id` compatibility + 0-based array slices (`nuclide_not_found = -1`) |
| **Partition function grid** | `pfcn(1:24, 1:nnuclides)` (column-major) | C-contiguous shape `(nnuclides, 24)` (`Tpart` length `24` in $\mathrm{GK}$) |
| **Fe56-in-He4 2D spline** | `f(1:4, 1:105, 1:30)` on `(log_ne, logT)` | C-contiguous shape `(105, 30, 4, 4)` via `interp_mkbicub_db` |
| **4D ionization tables** | `ion_tbl(1:4, 1:28, 1:175, 1:102, 1:6, 1:5)` | C-contiguous shape `(n_z, n_x, 28, 175, 102, 4)` over $(Z, X, v, \log Q, \log T, c)$ |
| **4D SED flux cube (`colors`)** | `cube_flux(1:n_teff, 1:n_logg, 1:n_meta, 1:n_lambda)` (column-major) | C-contiguous shape `(n_teff, n_logg, n_meta, n_lambda)` with $\lambda$ contiguous in memory |

---

## 3. Module 2a: Atomic & Nuclear Catalog (`chem`)

### 3.1 Nuclide Atomic Mass, Binding Energy, and Coulomb Moments
For each nuclide $i$ with proton number $Z_i$, neutron number $N_i$, baryon number $A_i = Z_i + N_i$, and atomic mass excess $\Delta M_i$ (in $\mathrm{MeV}$):

1. **Double-Precision Atomic Weight ($W_i$ in $\mathrm{amu}$)**:
   $$W_i = A_i + \Delta M_i\,\frac{\mathrm{mev\_to\_ergs}}{\mathrm{amu}\cdot c^2},$$
   where $\mathrm{mev\_to\_ergs} = 10^6\,\mathrm{eV\_to\_ergs}$, $\mathrm{amu}$, and $c = \mathrm{clight}$ are taken directly from `flexmesa.constants` (`crates/flexmesa-const`).
2. **Nuclear Binding Energy ($B_i$ in $\mathrm{MeV}$) and Reaction $Q$-Value**:
   $$B_i = Z_i \Delta M_p + N_i \Delta M_n - \Delta M_i, \qquad Q_i = -\Delta M_i = B_i - Z_i \Delta M_p - N_i \Delta M_n,$$
   with proton mass excess $\Delta M_p = 7.288970500\,\mathrm{MeV}$ (`h1`) and neutron mass excess $\Delta M_n = 8.071317100\,\mathrm{MeV}$ (`neut`). For a reaction with $N_{\mathrm{in}}$ reactants and $N_{\mathrm{out}}$ products:
   $$Q_{\mathrm{total}} = \sum_{j=1}^{N_{\mathrm{in}}} \Delta M_{\mathrm{in}, j} - \sum_{k=1}^{N_{\mathrm{out}}} \Delta M_{\mathrm{out}, k}.$$
3. **Coulomb Power ($Z_i^{5/3}$)**:
   $$\mathrm{Z53}_i = Z_i^{5/3} = \operatorname{pow}(Z_i, 5/3).$$

### 3.2 Solar Abundance Scales and $[\mathrm{M/H}]$
1. **Anders & Grevesse (1989) 286-Isotope Reference (`init_ag_data`)**:
   - Normalizes the 286 isotopic mass fractions `solx(1:286)` by adding $1 - \sum_{j=1}^{286} \mathrm{solx}_j$ into `solx(1)` (`h1`), yielding:
     $$Z_{\odot,\mathrm{AG89}} = \sum_{Z_j \ge 3} \mathrm{solx}_j = 1.8987109631891819 \times 10^{-2}, \qquad Y_{e,\odot,\mathrm{AG89}} = \sum_{j=1}^{286} \frac{Z_j}{A_j}\,\mathrm{solx}_j = 8.5280190902027331 \times 10^{-1}.$$
2. **Ten Elemental Metal-Fraction Scales (`*_element_zfrac(1:112)`)**:
   - `AG89` (`1`) and `L09` (`7`) normalize their tabulated elemental mass fractions over $Z \ge 3$.
   - `GN93` (`2`), `GS98` (`3`), `L03` (`4`), `AGS05` (`5`), `AGSS09` (`6`), `A09_Prz` (`8`), `MB22_photospheric` (`9`), and `AAG21_photospheric` (`10`) initialize $\epsilon_Z = -20.0$ for all $Z \in \{1,\dots,112\}$, populate $\epsilon_Z = \log_{10}(N_Z / N_{\mathrm{H}}) + 12$ for $Z \in [3, 92]$, and convert to metal mass fractions using each element's dominant isotope mass $W_{\mathrm{main}, Z}$:
     $$w_Z = 10^{\epsilon_Z}\,W_{\mathrm{main}, Z}, \qquad f_Z = \frac{w_Z}{\sum_{Z'=3}^{92} w_{Z'}} \quad (3 \le Z \le 92).$$
3. **Spectroscopic Metallicity (`chem_M_div_h`)**:
   $$\left[\mathrm{M/H}\right] = \log_{10}\!\left(\frac{Z}{X}\right) - \log_{10}\!\left(\frac{Z_\odot}{1 - Z_\odot - Y_\odot}\right).$$

### 3.3 Composition Moments and Analytical Jacobians (`get_composition_info`)
Given baryon mass fractions $X_1, \dots, X_S$ and nuclide IDs $\mathrm{cid}_1, \dots, \mathrm{cid}_S$ with molar abundances $Y_i = X_i / A_i$:

$$\Sigma_X = \sum_{i=1}^S X_i, \quad \Sigma_Y = \sum_{i=1}^S Y_i, \quad \bar{A} = \frac{\Sigma_X}{\Sigma_Y}, \quad \bar{Z} = \bar{A} \sum_{i=1}^S Y_i Z_i, \quad \langle Z^2 \rangle = \bar{A} \sum_{i=1}^S Y_i Z_i^2,$$

$$\langle Z^{5/3} \rangle = \bar{A} \sum_{i=1}^S Y_i Z_i^{5/3}, \quad Y_e = \frac{\bar{Z}}{\bar{A}}, \quad \mathrm{mc} = \sum_{i=1}^S Y_i W_i,$$

$$X_{\mathrm{H}} = \sum_{Z_i = 1} X_i, \quad X_{\mathrm{He}} = \sum_{Z_i = 2} X_i, \quad X_Z = \operatorname{clip}\!\left(1 - (X_{\mathrm{H}} + X_{\mathrm{He}}),\, 0,\, 1\right),$$

with exact partial derivatives with respect to $X_i$:

$$\frac{\partial \bar{A}}{\partial X_i} = \frac{\bar{A}\left(A_i - \bar{A}\right)}{A_i\,\Sigma_X}, \qquad \frac{\partial \bar{Z}}{\partial X_i} = \frac{\bar{A}\left(Z_i - \bar{Z}\right)}{A_i\,\Sigma_X}, \qquad \frac{\partial \mathrm{mc}}{\partial X_i} = \frac{W_i}{A_i} - \mathrm{mc}.$$

---

## 4. Module 2b: Effective Ionic Charge & 4D Ionization Tables (`ionization`)

### 4.1 2D Thomas-Fermi Debye-Huckel $^{56}\mathrm{Fe}$ in $^{4}\mathrm{He}$ (`eval_charge_of_Fe56_in_He4`)
Constructs a not-a-knot (`ibc = 0`) 2D bicubic spline via `interp_mkbicub_db` over the $105 \times 30$ grid $(\log_{10} n_e \in [24.0, 31.0],\, \log_{10} T \in [6.0, 8.0])$ from `log_ne_fe56_he4.data`, `logT_fe56_he4.data`, and `z_fe56_he4.data`, and evaluates $Z_{\mathrm{Fe56}}(\log_{10} n_e, \log_{10} T)$ via `interp_evbicub_db`.

### 4.2 Three-Stage 4D $(Z, X, \log Q, \log T)$ Table Interpolation (`eval_ionization`)
Given $(Z, X, \rho, \log_{10}\rho, T, \log_{10}T)$:
1. **Stage 1 — Clamped 2D $(\log Q, \log T)$ Bicubic Evaluation at Fixed $(Z_k, X_m)$**:
   With $\log Q = \log_{10}\rho - 2\log_{10}T + 12$ on the uniform $175 \times 102$ grid ($\log Q \in [-10.00, 5.66], \Delta\log Q = 0.09$; $\log T \in [2.10, 8.16], \Delta\log T = 0.06$), out-of-range coordinates are clamped to grid bounds, cell indices are computed via $\operatorname{int}((\log Q - \log Q_{\min})/\Delta\log Q + 10^{-4})$, and all 28 quantities are evaluated via the 2D cubic Hermite/spline polynomial using NaN-sanitized bicubic coefficients.
2. **Stage 2 — 1D $X$-Interpolation at Fixed $Z_k$ ($\Delta X = 0.2$)**:
   - If $X < 10^{-6}$ or $N_X = 1$: returns the $X = 0$ table result.
   - If $X \le 0.2$ or $X \ge X_{N_X - 1}$: 3-point Lagrange quadratic interpolation over $[0, 0.2, 0.4]$ or $[X_{N_X-3}, X_{N_X-2}, X_{N_X-1}]$.
   - If $0.2 < X < X_{N_X - 1}$: 4-point Catmull-Rom / blended-quadratic cubic interpolation with $\alpha = (X - X_{i_{\mathrm{lo}}+1}) / \Delta X \in [0, 1]$:
     $$c_1 = -\frac{\alpha(\alpha - 1)^2}{2}, \quad c_2 = \frac{2 + \alpha^2(-5 + 3\alpha)}{2}, \quad c_3 = \frac{\alpha + \alpha^2(4 - 3\alpha)}{2}, \quad c_4 = \frac{\alpha^2(\alpha - 1)}{2}.$$
3. **Stage 3 — 1D $Z$-Interpolation across $Z \in \{0.00, 0.02, 0.04, 0.20, 1.00\}$**:
   - $Z < 10^{-6}$: returns $Z = 0.00$.
   - $10^{-6} \le Z \le 0.04$: 3-point Lagrange quadratic interpolation across $\{0.00, 0.02, 0.04\}$ ($\Delta Z = 0.02$).
   - $0.04 < Z \le 0.20$ or $0.20 < Z \le 1 - 10^{-6}$: cosine-smoothed 2-point interpolation:
     $$t = \frac{Z - Z_a}{Z_b - Z_a}, \qquad \alpha = \frac{1 - \cos(\pi t)}{2}, \qquad \mathrm{res} = (1 - \alpha)\,\mathrm{res}(Z_a) + \alpha\,\mathrm{res}(Z_b).$$
   - $Z > 1 - 10^{-6}$: returns $Z = 1.00$.

### 4.3 Continuum-Depression Typical Ionic Charge (`chi_info`, `chi_effective`, `eval_typical_charge`)
Implements Paquette et al. (1986) and Dupuis et al. (1992) continuum depression against Allen (1973) ionization potentials $I_{Z, i}$ ($Z \le 30$):

$$\chi = 1.987 \times 10^{-4}\,T\left(-8.392 - \log_{10}\rho + 1.5\log_{10}T - \log_{10}\frac{z_1}{a_1}\right),$$

$$\chi_{\mathrm{eff}}(z_2) = \chi + \frac{c_0}{z_2^3} + \min\!\left[c_1 z_2,\; c_2\left(\left(\frac{z_2}{z_1}\right)^{2/3} + 0.6\right)\right].$$

---

## 5. Module 2c: 4D SED Interpolation & Synthetic Photometry (`colors`)

### 5.1 Normalized Interpolation Radius and Incomplete 4D Cubic Hermite Tensor Interpolation
1. **Interpolation Radius (`compute_interp_radius`)**: Minimum Euclidean distance in min-max normalized $(T_{\mathrm{eff}}, \log g, [\mathrm{M/H}])$ space across all valid lookup table rows.
2. **4D Cubic Hermite Interpolation (`hermite_interp_vector`)**:
   Given cell $(i_x, i_y, i_z)$, normalized coordinates $(t_x, t_y, t_z) \in [0, 1]^3$ (clamped at boundaries), Hermite basis functions $h_{00}(t) = 2t^3 - 3t^2 + 1$, $h_{01}(t) = -2t^3 + 3t^2$, $h_{10}(t) = t^3 - 2t^2 + t$, $h_{11}(t) = t^3 - t^2$, and finite-difference corner derivatives $(\partial_x F, \partial_y F, \partial_z F)$ (centered in the interior, one-sided at boundaries), evaluates for every wavelength $\lambda$:
   $$F_{\mathrm{surf}}(\lambda) = \sum_{\alpha_z, \alpha_y, \alpha_x \in \{0,1\}} \left[w_{000} F + w_{100} \Delta x\,\partial_x F + w_{010} \Delta y\,\partial_y F + w_{001} \Delta z\,\partial_z F\right].$$

### 5.2 Distance Dilution, Non-Uniform Simpson Integration & Synthetic Magnitudes
1. **Bolometric Photometry (`calculate_bolometric`)**:
   Dilutes surface flux by $(R/d)^2$, zeroes negative fluxes on $i \in [0, n_\lambda - 2]$, integrates $F_{\mathrm{bol}} = \int F_{\mathrm{obs}}(\lambda)\,d\lambda$ via composite non-uniform quadratic Simpson integration (with a final trapezoidal panel when $n_\lambda$ is even), and returns $\mathrm{Mag}_{\mathrm{bol}} = -2.5\log_{10}(F_{\mathrm{bol}})$.
2. **Synthetic Filter Magnitudes (`calculate_synthetic`)**:
   Resamples filter transmission $T(\lambda)$ onto $\lambda_{\mathrm{SED}}$ with endpoint clamping ($T(\lambda \le \lambda_1^{(F)}) = T_1^{(F)}$, $T(\lambda \ge \lambda_m^{(F)}) = T_m^{(F)}$), evaluates photon-weighted band flux:
   $$F_{\mathrm{syn}} = \frac{\operatorname{Simpson}(\lambda_{\mathrm{SED}},\, F_{\mathrm{obs}}\,T_{\mathrm{SED}}\,\lambda_{\mathrm{SED}})}{\operatorname{Simpson}(\lambda_{\mathrm{SED}},\, T_{\mathrm{SED}}\,\lambda_{\mathrm{SED}})}, \qquad m_{\mathrm{filter}} = -2.5\log_{10}\!\left(\frac{F_{\mathrm{syn}}}{F_0}\right),$$
   against precomputed Vega, AB ($3631\,\mathrm{Jy}$), or ST ($3.63 \times 10^{-9}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$) zero-point fluxes $F_0$.

---

## 6. Verification and Wave 2.5 Benchmark Gates

1. **`chem` Golden Gate**: Match all 2,635 lines of `chem/test/test_output` to $\le 10^{-14}$ relative error.
2. **`ionization` Golden Gate**: Match all 43 lines of `ionization/test/test_output` (3 `eval_charge_of_Fe56_in_He4` points and 28 `eval_ionization` outputs) to $\le 10^{-14}$ relative error.
3. **`colors` Golden Gate**: Match all 234 lines of `colors/test/test_output` (3 stellar archetypes, 3 parameter sweeps, 21 solar SED points, and 4 Stefan-Boltzmann checks) to $\le 10^{-14}$ relative error.
4. **Wave 2.5 Upstream Fortran MESA Gate**: Link and benchmark upstream `libchem.a`, `libionization.a`, and `libcolors.a` against Safe Rust, `PyO3`, and `@jax.jit`, and sync the MkDocs Material website.
