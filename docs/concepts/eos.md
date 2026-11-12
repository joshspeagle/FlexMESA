# Equations of State (`flexmesa.eos`)

The equation of state (EOS) links the microscopic thermodynamics of a stellar plasma (density $\rho$, temperature $T$, and nuclide mass fractions $\mathbf{X}$) to the macroscopic pressure $P$, specific internal energy $e$, specific entropy $s$, specific heats ($c_V, c_P$), and adiabatic indices ($\Gamma_1, \Gamma_3, \nabla_{\mathrm{ad}}$) that govern hydrostatic balance and convective stability. Because stellar interiors span more than twenty orders of magnitude in density and ten orders of magnitude in temperature, no single physical approximation holds everywhere. `flexmesa.eos` (`crates/flexmesa-eos`) implements MESA's complete 7-component composite equation of state with $C^2$-continuous polygonal quintic blending, third-order free-energy automatic differentiation, and Implicit Function Theorem $(P_{\mathrm{gas}}, T)$ and $(\gamma, e)$ root-finding inversions.

---

## 1. Canonical 26-Slot Thermodynamic State & Maxwell Consistency

Every equation-of-state evaluator in `FlexMESA` populates the canonical 26-slot MESA thermodynamic state vector `res[0..26]` (including $\ln P_{\mathrm{gas}}$, $\ln e$, $\ln s$, $\mu$, $1/\mu_e$, $\eta$, $c_V$, $c_P$, $\chi_\rho$, $\chi_T$, $\Gamma_1$, $\Gamma_3 - 1$, $\nabla_{\mathrm{ad}}$, and phase indicator $\phi_{\mathrm{cryst}}$) alongside exact analytical partial derivatives with respect to $(\ln\rho, \ln T)$ (`d_dlnd`, `d_dlnT`) and species mass fractions $X_j$ (`d_dxa`, tracking $[\partial \ln P_{\mathrm{gas}}/\partial X_j, \partial \ln e/\partial X_j, \partial \ln s/\partial X_j]$ analytically across `HELM`, `OPAL/SCVH`, `FreeEOS`, `CMS`, and $C^2$ quintic blend transitions, with `Skye`, `PC`, and `Ideal` returning `0.0` matching upstream component evaluators, for coupled Henyey momentum, energy, and $\epsilon_{\mathrm{grav}}$ Jacobians), plus the active component weight vector (`eos_frac`).

When an EOS component is formulated in terms of its specific Helmholtz free energy $F(\rho, T, \mathbf{X})$ ($\mathrm{erg\,g^{-1}}$), evaluating $F$ over a two-variable third-order hyperdual number (`AutoDiffReal2VarOrder3` in Safe Rust, or nested forward-mode `jax.jacfwd` in JAX) yields the exact value, gradient, Hessian, and third-order partial derivatives of $F$ with respect to $(\rho, T)$ in a single forward pass. Because all thermodynamic quantities derive analytically from a single scalar potential $F(\rho, T)$, the Maxwell relations ($\partial s / \partial \rho|_T = -\partial (P_{\mathrm{gas}}/\rho^2) / \partial T|_\rho$ and $\partial^2 F / \partial \rho \partial T = \partial^2 F / \partial T \partial \rho$) are satisfied to machine precision:

$$P_{\mathrm{gas}} = \rho^2 \left(\frac{\partial F}{\partial \rho}\right)_T, \qquad s = -\left(\frac{\partial F}{\partial T}\right)_\rho, \qquad e = F + T s,$$

$$c_V = \left(\frac{\partial e}{\partial T}\right)_\rho = -T \left(\frac{\partial^2 F}{\partial T^2}\right)_\rho, \qquad \chi_\rho = \left(\frac{\partial \ln P}{\partial \ln \rho}\right)_T, \qquad \chi_T = \left(\frac{\partial \ln P}{\partial \ln T}\right)_\rho,$$

$$\Gamma_3 - 1 = \frac{P\,\chi_T}{\rho\,T\,c_V}, \qquad \Gamma_1 = \chi_\rho + \chi_T\,(\Gamma_3 - 1), \qquad \nabla_{\mathrm{ad}} = \frac{\Gamma_3 - 1}{\Gamma_1}, \qquad c_P = c_V\,\frac{\Gamma_1}{\chi_\rho}.$$

---

## 2. The Seven Equation-of-State Components

`flexmesa.eos` combines seven specialized thermodynamic models across the $(\log_{10}\rho, \log_{10} T)$ plane:

| Component | Entry Point | Physical Regime & Formulation |
| :--- | :--- | :--- |
| **1. `Ideal`** | `eval_ideal_eos` | Low-density, low-temperature neutral atomic gas ($\mu = \bar{A}$) plus blackbody radiation pressure $P_{\mathrm{rad}} = \frac{1}{3}a T^4$ and cumulative NIST ground-state ionization offsets $E_{\mathrm{ion}}(\mathbf{X})$. |
| **2. `HELM`** | `eval_helm_eos` | Timmes & Swesty (2000) biquintic/bicubic Hermite electron-positron plasma table (`helm_table_full.bin`, $2701 \times 1001$ grid in $\log_{10}(\rho Y_e)$ and $\log_{10}T$) with ideal ions and Yakovlev & Shalybkov (1989) Coulomb corrections. |
| **3. `PC`** | `eval_pc_eos` | Potekhin & Chabrier (2000, 2010) strongly coupled one-component plasma (`OCP`) liquid and body-centered cubic (`bcc`) solid free energies with Baiko et al. (2001) lattice phonons and linear multi-ion mixing. |
| **4. `Skye`** | `eval_skye_eos` | Jermyn et al. (2021) third-order free-energy minimization EOS combining `HELM` electrons, Tanaka & Ichimaru (`EXCOR7`) screening, Ogata et al. (1993) non-ideal solid/liquid mixing, and smoothed crystallization latent heat. |
| **5. `OPAL/SCVH`** | `eval_opal_scvh_eos` | Rogers & Nayfonov (2002) `OPAL` blended with Saumon, Chabrier & Van Horn (1995) `SCVH` 4D $(X, Z, \log_{10} Q, \log_{10} T)$ partial-ionization tables for main-sequence and cool stellar envelopes. |
| **6. `FreeEOS`** | `eval_free_eos` | Irwin (2004) free-energy minimization 4D $(X, Z, \log_{10} Q, \log_{10} T)$ bicubic spline tables covering 20 elements across solar and metal-poor stellar interiors. |
| **7. `CMS`** | `eval_cms_eos` | Chabrier, Mazevet & Soubiran (2019) pure-hydrogen ($X=1$) and pure-helium ($Y=1$) tables combined via additive volume and entropy mixing with non-ideal H-He interaction terms for giant planets and brown dwarfs. |

### 2.1 Neutral Ideal Gas & Ionization Zero-Point Offset (`ideal`)
In low-temperature neutral gas where ionization is negligible, `eval_ideal_eos` evaluates the translational Helmholtz free energy of neutral atoms with thermal de Broglie wavelength density $n_{Q,j}(T)$:

$$F_{\mathrm{ideal}}(\rho, T, \mathbf{y}) = \frac{k_{\mathrm{B}} T}{\bar{A} m_u} \sum_{j=1}^K y_j \left[\ln\!\left(\frac{n_j}{n_{Q,j}(T)}\right) - 1\right], \qquad n_{Q,j}(T) = \left(\frac{2\pi W_j m_u k_{\mathrm{B}} T}{h^2}\right)^{3/2}.$$

To link the neutral zero-point energy of `ideal` with fully ionized interior equations of state (`Skye` and `HELM`), `compute_ion_offset` and `compute_ion_offset_from_za` sum the cumulative NIST ground-state ionization energies $E_{\mathrm{ion}}(\mathbf{X})$ required to strip neutral atoms of atomic number $Z_i \in \{1, \dots, 28\}$ to bare nuclei.

### 2.2 Timmes & Swesty (2000) Helmholtz EOS (`helm`)
The `HELM` equation of state interpolates the electron-positron Helmholtz free energy $F_{e^\pm}(\rho Y_e, T)$ from 9 precomputed nodal fields ($F, F_\rho, F_T, F_{\rho\rho}, F_{TT}, F_{\rho T}, F_{\rho\rho T}, F_{\rho TT}, F_{\rho\rho TT}$) using 1D quintic Hermite basis polynomials ($\psi_0, \psi_1, \psi_2$ and their first three analytical derivatives), while electron chemical potential $\eta_{e^-}$ and pressure-ionization modifiers are interpolated via bicubic Hermite basis polynomials ($\omega_0, \omega_1, \omega_2$).

In JAX, differentiating directly through the standard algebraic form `w0 = 1 - z^2 (3 - 2 z)` differs from Fortran MESA's Horner derivative polynomial `dw0 = 6 z (z - 1)` by 1 to 2 ULPs, which is amplified by $\mathcal{O}(\Delta\rho^{-2}) \sim 10^{10}$ in second-order density curvature at low density. `FlexMESA` attaches recursive `@jax.custom_jvp` rules (`quintic_w0_coord` and `cubic_w0_coord`) that bind each derivative order to the exact Horner polynomial used by Fortran MESA and Safe Rust.

### 2.3 Potekhin & Chabrier (2010) `PC` and Jermyn et al. (2021) `Skye`
In dense white-dwarf cores and neutron-star envelopes where the ion Coulomb coupling parameter

$$\Gamma_i = \frac{Z_i^2 e^2}{a_i k_{\mathrm{B}} T}, \qquad a_i = \left(\frac{3 Z_i}{4\pi n_e}\right)^{1/3}$$

exceeds unity, non-ideal ion-ion and electron-ion correlations dominate the ionic pressure and specific heat, and the plasma freezes into a body-centered cubic (bcc) crystal near $\Gamma_i \approx 175$:

- **Aparicio (1998) 20-Point Gauss-Fermi Integrals (`dfermi`)**: Implements MESA's 4-segment Gauss-Legendre and Gauss-Laguerre quadrature for relativistic Fermi-Dirac integrals $F_k(\eta, \beta)$ ($k \in \{-1/2, 1/2, 3/2, 5/2\}$) and their analytical $(\eta, \beta)$ Jacobians.
- **`PC` (`eval_pc_eos`)**: Evaluates Potekhin & Chabrier (2000, 2010) OCP liquid and bcc solid free energies, Baiko et al. (2001) harmonic and anharmonic lattice phonons, and the linear-mixing rule across multi-species ion mixtures.
- **`Skye` (`eval_skye_eos`)**: Evaluates total liquid and solid free energies using third-order automatic differentiation (`AutoDiffReal2VarOrder3` in Safe Rust and single-pass `jax.jacfwd` in JAX), incorporates Ogata et al. (1993) and Potekhin-Chabrier non-ideal mixing corrections, determines the equilibrium crystallization fraction $\phi_{\mathrm{cryst}} \in [0, 1]$ by minimizing total free energy with thermal blurring across $\Delta F / (k_{\mathrm{B}} T)$, and injects the smoothed crystallization latent heat directly into `latent_ddlnT` and `latent_ddlnRho`.

### 2.4 4D Tabulated Partial-Ionization Tables (`OPAL/SCVH`, `FreeEOS`, `CMS`)
In partially ionized stellar envelopes ($\log_{10} T \lesssim 6.5$), `FlexMESA` evaluates four-dimensional $(X, Z, \log_{10} Q, \log_{10} T)$ bicubic spline tables with rescaled density coordinate

$$\log_{10} Q = \log_{10}\rho - 2\log_{10} T + 12.$$

Each $(X_m, Z_n)$ plane stores precomputed 2D not-a-knot bicubic spline coefficients (`[f, f_qq, f_tt, f_qqtt]`) for all 16 base thermodynamic channels (`lnPgas` through `eta`). `eval_xtable_16` evaluates all 16 bicubic splines in a single pass and maps $(\partial/\partial \log_{10} Q, \partial/\partial \log_{10} T)$ to natural-logarithm $(\ln\rho, \ln T)$ derivatives, followed by quadratic/cubic Lagrange interpolation in $X$ and linear interpolation in $Z$.

![EOS component blending phase diagram and C12/O16 white dwarf interior Gamma_1 profile](../assets/figures/eos_helm_skye_phase_diagram.png)

*Figure 1: The left panel maps the active equation-of-state regime boundaries and $C^2$ polygonal quintic transition zones across the $(\log_{10}\rho, \log_{10} T)$ plane (`HELM`, `Skye`, `PC`, `FreeEOS`, `OPAL/SCVH`, `CMS`, and low-temperature `Ideal`), overlaid with a $1\,M_\odot$ Zero-Age Main Sequence profile and a cooling white-dwarf core trajectory. The right panel shows the adiabatic exponent $\Gamma_1$ and equilibrium crystallization fraction $\phi_{\mathrm{cryst}}$ computed by `eval_skye_eos` across a $\mathrm{C}^{12}/\mathrm{O}^{16}$ white-dwarf core ($T = 10^7\,\mathrm{K}$) as the plasma freezes into a body-centered cubic lattice near $\Gamma \approx 175$.*

---

## 3. $C^2$ Quintic Smootherstep Blending & Root-Finding Inversions

### 3.1 Polygonal Distance & Quintic Smootherstep (`eos_dt_get`)
To transition smoothly between `CMS`, `PC`, `Skye`, `FreeEOS`, `OPAL/SCVH`, `HELM`, and `Ideal` across the $(\log_{10}\rho, \log_{10}T)$ plane without introducing gradient discontinuities into the Henyey Jacobian, `min_distance_to_polygon` computes the signed distance $d(\log_{10}\rho, \log_{10}T)$ to each regime polygon of transition width $\Delta$ (`0.1` dex by default) and `quintic_blend` evaluates the $C^2$ smootherstep polynomial:

$$s(x) = 6 x^5 - 15 x^4 + 10 x^3, \qquad x \in [0, 1], \qquad s'(0) = s'(1) = s''(0) = s''(1) = 0.$$

Because both the first and second derivatives of $s(x)$ vanish at the transition boundaries, the blended thermodynamic state and its $(\ln\rho, \ln T)$ Jacobian remain smooth everywhere.

### 3.2 Inverse $(P_{\mathrm{gas}}, T)$ and $(\gamma, e)$ Solvers (`eos_pt_get`, `eos_gamma_de_get`)
- **Fixed-$(P_{\mathrm{gas}}, T)$ Inversion (`eos_pt_get`)**: Solves $\ln P_{\mathrm{gas}}(\log_{10}\rho, \log_{10}T) = \ln P_{\mathrm{gas,target}}$ for $\log_{10}\rho$ at fixed $(T, \mathbf{X})$ via bracketed Newton-Raphson iteration (`eos_dt_get_rho`) and transforms all 26 thermodynamic partial derivatives from $(\rho, T)$ to $(P_{\mathrm{gas}}, T)$ coordinates via the Implicit Function Theorem:

  $$\left(\frac{\partial f}{\partial \ln P_{\mathrm{gas}}}\right)_T = \frac{1}{\chi_\rho}\left(\frac{\partial f}{\partial \ln \rho}\right)_T, \qquad \left(\frac{\partial f}{\partial \ln T}\right)_{P_{\mathrm{gas}}} = \left(\frac{\partial f}{\partial \ln T}\right)_\rho - \frac{\chi_T}{\chi_\rho}\left(\frac{\partial f}{\partial \ln \rho}\right)_T.$$

- **Ideal $\gamma$-Law + Radiation $(\rho, e) \mapsto T$ Inversion (`eos_gamma_de_get`)**: Provides an exact closed-form $(\rho, e) \mapsto T$ solver for an ideal $\gamma$-law gas plus radiation pressure by solving the depressed quartic $a T^4 + (\rho N_{\mathrm{A}} k_{\mathrm{B}} / [\bar{A}(\gamma - 1)]) T - \rho e = 0$ via Ferrari's resolvent cubic followed by two Newton-Raphson polishing steps.

=== "Python & JAX (`flexmesa.eos`)"

    ```python
    import jax.numpy as jnp
    import numpy as np
    from flexmesa import chem, eos

    # 1. Unified (rho, T) EOS evaluation and (P_gas, T) inversion on CPU
    cids = (chem.get_nuclide_index("c12"), chem.get_nuclide_index("o16"))
    xa = np.array([0.50, 0.50])
    dt_out = eos.eos_dt_get(
        rho=1.0e6, temp=1.0e7, chem_id=cids, xa=xa, use_skye=True
    )
    pgas_target = float(np.exp(dt_out.res[eos.I_LN_PGAS]))
    pt_out = eos.eos_pt_get(
        pgas=pgas_target,
        temp=1.0e7,
        chem_id=cids,
        xa=xa,
        use_skye=True,
    )
    print(f"Recovered log10(rho) = {pt_out.log10_rho:.12f}")

    # 2. Vectorized 1024-zone Skye evaluation on GPU via @jax.jit(vmap)
    t_zones = jnp.full(1024, 1.0e7, dtype=jnp.float64)
    rho_zones = jnp.logspace(4.0, 7.0, 1024, dtype=jnp.float64)
    abar_zones = jnp.full(1024, 13.714285714285714, dtype=jnp.float64)
    zbar_zones = jnp.full(1024, 6.857142857142857, dtype=jnp.float64)
    xa_zones = jnp.tile(jnp.array([0.50, 0.50], dtype=jnp.float64), (1024, 1))
    skye_gpu = eos.eval_skye_eos_zones(
        rho_zones,
        t_zones,
        abar_zones,
        zbar_zones,
        xa_zones,
        z=(6, 8),
        acmi=jnp.array([12.0, 15.99491461957], dtype=jnp.float64),
        a=jnp.array([12.0, 16.0], dtype=jnp.float64),
        backend="jax",
    )
    ```

=== "Safe Rust (`crates/flexmesa-eos`)"

    ```rust
    use flexmesa_chem::NuclideCatalog;
    use flexmesa_eos::{
        EosConfig, HelmTable, I_LN_PGAS, SkyeConfig,
        eos_dt_get, eos_pt_get, eval_skye_eos,
    };

    let cat = NuclideCatalog::builtin_ref();
    let cids = [
        cat.get_nuclide_index("c12").unwrap(),
        cat.get_nuclide_index("o16").unwrap(),
    ];
    let xa = [0.50, 0.50];
    let ht = HelmTable::builtin();
    let skye = eval_skye_eos(
        &SkyeConfig::default(),
        ht,
        1.0e6,
        1.0e7,
        13.714285714285714,
        6.857142857142857,
        &cids,
        &xa,
    ).unwrap();

    let cfg = EosConfig { use_skye: true, ..EosConfig::default() };
    let dt = eos_dt_get(ht, 1.0e6, 1.0e7, &cids, &xa, &cfg).unwrap();
    let log_pgas = dt.res[I_LN_PGAS] / std::f64::consts::LN_10;
    let pt = eos_pt_get(ht, &cids, &xa, log_pgas, 7.0, &cfg).unwrap();
    assert!((pt.log10_rho - 6.0).abs() < 1.0e-7);
    ```
