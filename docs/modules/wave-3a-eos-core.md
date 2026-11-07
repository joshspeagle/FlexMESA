# Wave 3a Equations of State: Ideal, Helmholtz, Potekhin-Chabrier & Skye

This guide covers the thermodynamic formulations, free-energy automatic differentiation reductions, biquintic/bicubic polynomial basis functions, polygonal quintic regime blending, and root-finding inversions implemented in `flexmesa.eos` (`crates/flexmesa-eos`).

---

## 1. Canonical 26-Slot Thermodynamic State & Free-Energy Reduction

Every equation-of-state kernel in `FlexMESA` populates a canonical 26-slot thermodynamic state vector `res[0..26]` together with analytical partial derivatives with respect to $(\ln \rho, \ln T)$ (`d_dlnd`, `d_dlnT`) and species mass fractions $X_k$ (`d_dxa`), plus a 6-component EOS regime weight vector (`eos_frac` across `OPAL/SCVH`, `HELM`, `Skye`, `PC`, `FreeEOS`, and `CMS`, or 7-component extended weights including `ideal`).

When an EOS component is defined via its specific Helmholtz free energy $F(\rho, T, \mathbf{X})$ ($\mathrm{erg\,g^{-1}}$), evaluating $F$ over a two-variable third-order hyperdual number (`AutoDiffReal2VarOrder3` in Safe Rust, or nested forward-mode `jax.jacfwd` in JAX) yields the exact value, gradient, Hessian, and third-order partials of $F$ with respect to $(\rho, T)$ in a single forward pass. From $F$ and its temperature/density partials, pressure, specific internal energy, specific entropy, specific heats, and adiabatic indices follow without finite-difference truncation error:

$$P_{\mathrm{gas}} = \rho^2 \left(\frac{\partial F}{\partial \rho}\right)_T, \qquad s = -\left(\frac{\partial F}{\partial T}\right)_\rho, \qquad e = F + T s,$$

$$c_V = \left(\frac{\partial e}{\partial T}\right)_\rho = -T \left(\frac{\partial^2 F}{\partial T^2}\right)_\rho, \qquad \chi_\rho = \left(\frac{\partial \ln P}{\partial \ln \rho}\right)_T, \qquad \chi_T = \left(\frac{\partial \ln P}{\partial \ln T}\right)_\rho,$$

$$\Gamma_3 - 1 = \frac{P\,\chi_T}{\rho\,T\,c_V}, \qquad \Gamma_1 = \chi_\rho + \chi_T\,(\Gamma_3 - 1), \qquad \nabla_{\mathrm{ad}} = \frac{\Gamma_3 - 1}{\Gamma_1}, \qquad c_P = c_V\,\frac{\Gamma_1}{\chi_\rho}.$$

---

## 2. Analytical Neutral Ideal Gas & Radiation (`ideal`)

In low-density, low-temperature atomic gas where ionization is negligible, `eval_ideal_eos` evaluates a multi-species neutral ideal gas of mean molecular weight $\mu = \bar{A}$ combined with blackbody radiation pressure $P_{\mathrm{rad}} = \frac{1}{3} a T^4$:

$$F_{\mathrm{ideal}}(\rho, T, \mathbf{y}) = \frac{k_{\mathrm{B}} T}{\bar{A} m_u} \sum_{j=1}^K y_j \left[\ln\!\left(\frac{n_j}{n_{Q,j}(T)}\right) - 1\right], \qquad n_{Q,j}(T) = \left(\frac{2\pi W_j m_u k_{\mathrm{B}} T}{h^2}\right)^{3/2}.$$

To link the neutral zero-point energy of `ideal` with fully ionized interior equations of state (`Skye`), `compute_ion_offset` and `compute_ion_offset_from_za` sum the cumulative NIST ground-state ionization energies $E_{\mathrm{ion}}(\mathbf{X})$ required to strip neutral atoms of charge $Z_i \in \{1, \dots, 28\}$ to bare nuclei.

---

## 3. Timmes & Swesty (2000) Helmholtz EOS (`helm`)

The `HELM` equation of state (`eval_helm_eos`) models an arbitrarily degenerate and relativistic electron-positron plasma on a $541 \times 201$ $(\log_{10}(\rho Y_e), \log_{10} T)$ table (`helm_table.safetensors`) paired with ideal ion gas, blackbody radiation, and Yakovlev & Shalybkov (1989) Coulomb corrections:

1. **Biquintic Hermite Free-Energy Interpolation**:
   - The electron-positron Helmholtz free energy $F_{e^\pm}(\rho Y_e, T)$ is interpolated from 9 precomputed nodal fields ($F, F_\rho, F_T, F_{\rho\rho}, F_{TT}, F_{\rho T}, F_{\rho\rho T}, F_{\rho TT}, F_{\rho\rho TT}$) using 1D quintic Hermite basis polynomials ($\psi_0, \psi_1, \psi_2$ and their first three analytical derivatives), guaranteeing thermodynamic consistency ($\partial^2 F / \partial \rho \partial T = \partial^2 F / \partial T \partial \rho$) across cell boundaries.
2. **Bicubic Hermite Number Density & Pressure-Ionization Tables**:
   - Free electron chemical potential $\eta_{e^-}$, electron-positron number density $n_{e^\pm}$, and MESA's pressure-ionization and Coulomb screening modifiers are interpolated from 4 nodal fields ($f, f_\rho, f_T, f_{\rho T}$) via bicubic Hermite basis functions ($\omega_0, \omega_1, \omega_2$).
3. **Custom Horner `@jax.custom_jvp` Basis Chain in JAX**:
   - Evaluating higher-order JAX autodiff directly through `w0 = 1 - z^2 (3 - 2 z)` differs from Fortran MESA's explicit Horner derivative polynomial `dw0 = 6 z (z - 1)` by 1-2 ULPs, which is amplified by $O(\Delta \rho^{-2}) \sim 10^{10}$ in second-order density curvature at low density. `FlexMESA` attaches recursive `@jax.custom_jvp` rules (`quintic_w0_coord` and `cubic_w0_coord`) that bind each derivative order to the exact Horner polynomial used by Fortran MESA and Safe Rust.

---

## 4. Potekhin & Chabrier (2010) & Jermyn et al. (2021) `Skye` EOS (`pc`, `skye`)

In dense white dwarf cores and neutron star envelopes where the ion Coulomb coupling parameter

$$\Gamma_i = \frac{Z_i^2 e^2}{a_i k_{\mathrm{B}} T}, \qquad a_i = \left(\frac{3 Z_i}{4\pi n_e}\right)^{1/3}$$

exceeds unity, non-ideal ion-ion and electron-ion correlations dominate the ionic pressure and specific heat, and the plasma freezes into a body-centered cubic (bcc) crystal near $\Gamma_i \approx 175$.

1. **Aparicio (1998) 20-Point Gauss-Fermi Integrals (`dfermi`) & Electron Backgrounds**:
   - `dfermi` (`gauss_fermi.rs` / `fermi.py`) implements MESA's standalone 20-point Aparicio (1998) 4-segment Gauss-Legendre and Gauss-Laguerre quadrature for relativistic Fermi-Dirac integrals $F_k(\eta, \beta)$ ($k \in \{-1/2, 1/2, 3/2, 5/2\}$) and their analytical $(\eta, \beta)$ Jacobians. Within the EOS evaluators, `PC` uses the analytical Potekhin-Chabrier electron chemical potential and thermodynamic fits (`chemfit` and `elect11`), whereas `Skye` evaluates the third-order biquintic/bicubic `HELM` electron-positron table (`eval_electron_positron_order3`) combined with Tanaka & Ichimaru (`EXCOR7`) exchange-correlation screening.
2. **Potekhin & Chabrier (2010) (`eval_pc_eos`)**:
   - Evaluates the Potekhin & Chabrier (2000, 2010) one-component plasma (OCP) liquid and bcc solid free energies, Baiko et al. (2001) harmonic and anharmonic lattice phonons, and linear-mixing rule across multi-species ion mixtures with a phase transition over $\Gamma \in [\Gamma_{\mathrm{melt}}, \Gamma_{\mathrm{freeze}}]$.
3. **Jermyn et al. (2021) `Skye` (`eval_skye_eos`)**:
   - Evaluates the total liquid and solid free energies using third-order automatic differentiation (`AutoDiffReal2VarOrder3` in Safe Rust and single-pass `jax.jacfwd` in JAX), incorporates Ogata et al. (1993) and Potekhin-Chabrier non-ideal liquid/solid mixing corrections, determines the equilibrium phase index $\phi_{\mathrm{cryst}} \in [0, 1]$ by minimizing total free energy with thermal blurring across $\Delta F / (k_{\mathrm{B}} T)$, and injects the smoothed crystallization latent heat directly into `latent_ddlnT` and `latent_ddlnRho`.

![EOS component blending phase diagram and C12/O16 white dwarf interior Gamma_1 profile](../assets/figures/eos_helm_skye_phase_diagram.png)

---

## 5. Polygonal Quintic Blending (`eos_dt_get`) & Root-Finding Inversions (`eos_pt_get`, `eos_gamma_de_get`)

To transition smoothly between `Skye`, `PC`, `HELM`, and `ideal` across the $(\log_{10}\rho, \log_{10}T)$ plane without introducing gradient discontinuities into the Henyey Jacobian:

- **Polygonal Distance & Quintic Smootherstep (`min_distance_to_polygon`, `quintic_blend`)**:
  - Given the signed distance $d(\log_{10}\rho, \log_{10}T)$ to a polygonal regime boundary of transition width $\Delta$ (`0.1` dex by default), `quintic_blend` evaluates $s(x) = 6 x^5 - 15 x^4 + 10 x^3$ for $x \in [0, 1]$, whose first and second derivatives vanish identically at both endpoints ($s'(0)=s'(1)=s''(0)=s''(1)=0$).
- **Unified $(\rho, T)$ Dispatcher (`eos_dt_get`) and Inverse Solvers (`eos_pt_get`, `eos_gamma_de_get`)**:
  - `eos_dt_get` blends active EOS components (`Skye`, `PC`, `HELM`, `ideal`) using the chain rule for $(\ln\rho, \ln T)$ derivatives of the blend weights.
  - `eos_pt_get` inverts $\ln P_{\mathrm{gas}}(\log_{10}\rho, \log_{10}T) = \ln P_{\mathrm{gas,target}}$ for $\log_{10}\rho$ at fixed $(T, \mathbf{X})$ via bracketed Newton-Raphson iteration (`eos_dt_get_rho`) and transforms all 26 thermodynamic partials to $(P_{\mathrm{gas}}, T)$ coordinates via the Implicit Function Theorem:
    $$\left(\frac{\partial f}{\partial \ln P_{\mathrm{gas}}}\right)_T = \frac{1}{\chi_\rho}\left(\frac{\partial f}{\partial \ln \rho}\right)_T, \qquad \left(\frac{\partial f}{\partial \ln T}\right)_{P_{\mathrm{gas}}} = \left(\frac{\partial f}{\partial \ln T}\right)_\rho - \frac{\chi_T}{\chi_\rho}\left(\frac{\partial f}{\partial \ln \rho}\right)_T.$$

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
