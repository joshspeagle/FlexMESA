# Nuclear Composition, Reaction Networks & Thermal Neutrinos (`chem`, `ionization`, `neu`, `rates`, `net`)

Nuclear burning and neutrino cooling drive the secular evolution of stars from core hydrogen ignition through advanced carbon, neon, oxygen, and silicon burning. `FlexMESA` organizes its nuclear and plasma microphysics into five tightly coupled packages: isotopic registries and composition moments (`flexmesa.chem`), partial ionization and plasma screening charges (`flexmesa.ionization`), five-channel thermal neutrino emission (`flexmesa.neu`), screened thermonuclear and weak reaction rates (`flexmesa.rates`), and stiff implicit nuclear reaction networks (`flexmesa.net`).

---

## 1. Nuclide Catalog, Solar Mixtures & Composition Moments (`flexmesa.chem`)

Stellar equations of state, radiative opacities, and nuclear reaction networks depend on moments of the nuclide mass-fraction vector $\mathbf{X} = (X_1, \dots, X_K)$ across $K$ active isotopes. `flexmesa.chem` (`crates/flexmesa-chem`) embeds 347 built-in reference nuclides (plus a full `isotopes.data` ground-state and isomeric catalog parser with Wigner-function 24-point temperature-dependent nuclear partition functions and relativistic mass excesses $\Delta M_i$ in $\mathrm{MeV}$), 286 solar isotopes, and 10 solar and meteoritic reference abundance scales (`AG89` through `AAG21`, including Anders & Grevesse 1989, Grevesse & Noels 1993, Grevesse & Sauval 1998, Lodders 2003, Asplund et al. 2004, 2005, 2009, 2021, and Caffau et al. 2011).

### 1.1 Exact Composition Moments & Species Derivatives
Given mass fractions $X_i$, integer baryon numbers $A_i = Z_i + N_i$, atomic charges $Z_i$, and atomic weights $W_i$ ($\mathrm{g\,mol^{-1}}$), `basic_composition_info` and `get_composition_info` compute the molar abundances $Y_i = X_i / A_i$, mean atomic weight $\bar{A}$, mean ionic charge $\bar{Z}$, free-electron abundance $Y_e$, atomic-weight mass correction, and Coulomb coupling moments in a single $\mathcal{O}(K)$ pass:

$$S_X = \sum_{i=1}^K X_i, \qquad S_Y = \sum_{i=1}^K \frac{X_i}{A_i}, \qquad \bar{A} = \frac{S_X}{S_Y}, \qquad \bar{Z} = \bar{A} \sum_{i=1}^K \frac{Z_i X_i}{A_i}, \qquad Y_e = \frac{\bar{Z}}{\bar{A}},$$

$$\langle Z^2 \rangle = \bar{A} \sum_{i=1}^K \frac{Z_i^2 X_i}{A_i}, \qquad \langle Z^{5/3} \rangle = \bar{A} \sum_{i=1}^K \frac{Z_i^{5/3} X_i}{A_i}, \qquad \mathrm{mass\_correction} = \sum_{i=1}^K \frac{W_i X_i}{A_i},$$

together with the exact analytical partial derivatives with respect to each species mass fraction $X_i$:

$$\frac{\partial \bar{A}}{\partial X_i} = \frac{\bar{A}\,(A_i - \bar{A})}{A_i\,S_X}, \qquad \frac{\partial \bar{Z}}{\partial X_i} = \frac{\bar{A}\,(Z_i - \bar{Z})}{A_i\,S_X}.$$

For any nuclear reaction $\sum_r \nu_r \, {}^{A_r}\mathrm{El}_r \to \sum_p \nu_p \, {}^{A_p}\mathrm{El}_p$, `reaction_qtotal` evaluates the exact relativistic mass-excess energy release $Q_{\mathrm{total}} = \sum_r \nu_r \Delta M_r - \sum_p \nu_p \Delta M_p$ ($\mathrm{MeV}$).

---

## 2. Partial Ionization & Effective Ionic Charge (`flexmesa.ionization`)

In dense, partially ionized stellar interiors and white-dwarf envelopes, pressure ionization and plasma screening modify the effective ionic charge $Z_{\mathrm{eff}}$ of trace heavy elements and alter both Burgers atomic diffusion velocities and Coulomb transport coefficients. `flexmesa.ionization` (`crates/flexmesa-ionization`) implements three ionization subsystems:

1. **2D Bicubic Spline Evaluation of $\mathrm{Fe}^{56}$ in a $\mathrm{He}^4$ Plasma (`eval_charge_of_fe56_in_he4`)**:
   Interpolates $Z_{\mathrm{eff}}(\log_{10} n_e, \log_{10} T)$ over a $105 \times 30$ Thomas-Fermi-Dirac-Hubbard grid spanning $\log_{10}(n_e / \mathrm{cm}^{-3}) \in [24.0, 31.0]$ and $\log_{10}(T / \mathrm{K}) \in [6.0, 8.0]$ using pre-validated 2D bicubic Hermite splines (`interp_evbicub_prevalidated_db`), returning $(Z_{\mathrm{eff}}, \partial Z_{\mathrm{eff}}/\partial \log_{10} n_e, \partial Z_{\mathrm{eff}}/\partial \log_{10} T)$.
2. **4D Composition- and Density-Temperature Ionization Tables (`eval_ionization`)**:
   Performs bilinear interpolation across metallicity $Z \in \{0.00, 0.02, 0.04, 0.20, 1.00\}$ and hydrogen mass fraction $X \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$ over $(X, Z)$ bicubic spline subtables on a $57 \times 36$ $(\log_{10} Q, \log_{10} T)$ grid ($\log_{10} Q = \log_{10}\rho - 2\log_{10} T + 12$), returning all 28 ionic and plasma state quantities (`logPgas`, `logpe`, `logpp_H..Fe`, `Z_H..Fe`, `fneut_H..Fe`).
3. **Continuum Depression & Diffusion Typical Charges (`chi_info`, `chi_effective`, `eval_typical_charge`)**:
   Implements Paquette et al. (1986) and Dupuis et al. (1992) continuum depression of ionization potentials alongside Allen (1973) whole-number ionic charges for multi-fluid atomic diffusion.

---

## 3. Five-Channel Thermal Neutrino Energy Loss (`flexmesa.neu`)

In evolved stellar cores at $T \gtrsim 10^{7.5}\,\mathrm{K}$, thermal neutrino emission ($\epsilon_\nu$ in $\mathrm{erg\,g^{-1}\,s^{-1}}$) escapes freely from the star and dominates energy loss over surface photon radiation. `flexmesa.neu` (`crates/flexmesa-neu`) ports the complete Itoh et al. (1996, ApJS, 102, 411) fitting formulae across all five physical channels:

1. **Electron-Positron Pair Annihilation ($e^- + e^+ \to \nu + \bar{\nu}$, `pair`)**: Dominant at high temperature ($T \gtrsim 10^9\,\mathrm{K}$) and moderate density where thermal $e^\pm$ pairs are abundant.
2. **Plasmon Decay ($\gamma_{\mathrm{plas}} \to \nu + \bar{\nu}$, `plas`)**: Collective transverse and longitudinal plasma excitations decaying in degenerate electron gases ($\rho \gtrsim 10^5\,\mathrm{g\,cm^{-3}}$).
3. **Photoneutrino Process ($e^- + \gamma \to e^- + \nu + \bar{\nu}$, `phot`)**: Compton-like scattering with neutrino-pair emission, dominating at intermediate temperatures ($10^8\,\mathrm{K} \lesssim T \lesssim 10^{8.8}\,\mathrm{K}$).
4. **Bremsstrahlung ($e^- + (Z, A) \to e^- + (Z, A) + \nu + \bar{\nu}$, `brem`)**: Coulomb scattering off partially or fully crystallized ions across liquid and body-centered cubic lattice regimes.
5. **Electron-Ion Recombination (`reco`)**: Bound-free electron capture onto K-shell ions. Across the low-temperature threshold interval $\log_{10} T_{\min} < \log_{10} T \le \log_{10} T_{\mathrm{lim}}$ (with $\log_{10} T_{\min} = 7.0$ and default $\log_{10} T_{\mathrm{lim}} = 7.5$), all five channels are modulated by a smooth cosine turn-on factor:

   $$f_{\mathrm{cut}}(\log_{10} T) = \frac{1}{2}\left[1 - \cos\!\left(\pi\,\frac{\log_{10} T - \log_{10} T_{\min}}{\log_{10} T_{\mathrm{lim}} - \log_{10} T_{\min}}\right)\right].$$

Both Safe Rust (`neu_get` propagating exact analytical chain-rule derivatives in `NeuRate`) and Python/JAX (`neu_get` with `backend="auto" | "rust" | "jax"` and `neu_pair_plas_jax`) return the total loss rate $\epsilon_\nu$ and 5-channel breakdown alongside the exact 4-variable analytical Jacobian with respect to $(T, \rho, \bar{A}, \bar{Z})$:

$$\mathbf{J}_{\nu} = \left(\epsilon_\nu,\, \frac{\partial \epsilon_\nu}{\partial T},\, \frac{\partial \epsilon_\nu}{\partial \rho},\, \frac{\partial \epsilon_\nu}{\partial \bar{A}},\, \frac{\partial \epsilon_\nu}{\partial \bar{Z}}\right).$$

![Itoh et al. (1996) thermal neutrino energy-loss channels and logarithmic temperature sensitivity](../assets/figures/neu_regimes.png)

*Figure 1: The left panel shows the total thermal neutrino energy loss rate $\log_{10}(\epsilon_\nu / \mathrm{erg\,g^{-1}\,s^{-1}})$ and individual channel contributions (`Pair`, `Plasmon`, `Photo`, and `Bremsstrahlung`) from Itoh et al. (1996) across $\log_{10}(T/\mathrm{K}) \in [7.55, 9.60]$ at fixed $\rho = 10^6\,\mathrm{g\,cm^{-3}}$, $\bar{A} = 12$, and $\bar{Z} = 6$. The right panel verifies that the exact analytical logarithmic temperature sensitivity $\partial \ln \epsilon_\nu / \partial \ln T$ propagated via `DualOrder1` matches centered finite differences across the photoneutrino-to-pair transition.*

---

## 4. Screened Thermonuclear Rates & Weak Reactions (`flexmesa.rates`)

`flexmesa.rates` (`crates/flexmesa-rates`) evaluates forward and reverse thermonuclear reaction rates across all 316 built-in MESA reactions together with electrostatic plasma screening and weak nuclear rates:

- **7-Parameter REACLIB Forward & Detailed-Balance Reverse Rates (`eval_reaclib_handle`, `eval_reaclib_jax`)**: Evaluates JINA REACLIB rate fits $\lambda(T_9) = \exp\!\left(a_0 + \sum_{k=1}^5 a_k T_9^{(2k-5)/3} + a_6 \ln T_9\right)$ and applies nuclear partition function ratios and mass-excess $Q$-values for endothermic reverse reactions.
- **Weak, Intermediate & Strong Plasma Screening (`screen_pair`, `screen_salpeter_jax`, `screen_chugunov_jax`, `screen_extended_jax`)**: In dense stellar plasmas, surrounding electron and ion clouds shield the repulsive Coulomb barrier between reacting nuclei $(Z_1, Z_2)$ and enhance the bare reaction rate by $\exp(H_{12})$. `FlexMESA` implements Salpeter (1954) weak screening, Graboske et al. (1973) / Dewitt et al. (1973) intermediate-to-strong screening (`extended`), and Chugunov, DeWitt & Yakovlev (2007) quantum one-component plasma screening with exact $(\rho, T)$ derivatives.
- **2D Tabulated Weak Rates & Phase-Space Electron Captures (`eval_weak_pair`, `eval_ecapture_pair`)**: Evaluates 2D bicubic spline weak-rate tables (`Langanke & Martínez-Pinedo 2000`, `Oda et al. 1994`, `Fuller, Fowler & Newman 1985`, `Suzuki, Toki & Nomoto 2016`) and relativistic Fermi-Dirac phase-space integrals for electron capture and $\beta$-decay.

---

## 5. Stiff Nuclear Reaction Networks (`flexmesa.net`)

A nuclear reaction network over $K$ isotopes evolves molar abundances $Y_i = X_i / A_i$ according to the stiff nonlinear ODE system

$$\frac{dY_i}{dt} = \sum_{r} \nu_{i,r}\,R_r(\rho, T, \mathbf{Y}), \qquad \epsilon_{\mathrm{nuc}} = N_{\mathrm{A}} \sum_{r} Q_r\,R_r(\rho, T, \mathbf{Y}) - \epsilon_{\mathrm{neu,weak}}.$$

`flexmesa.net` (`crates/flexmesa-net`) provides:

1. **SIMD Dual-Number RHS & Analytical Jacobians (`eval_net`, `net_get`)**:
   Assembles the molar abundance rate vector $dY_i/dt$, net nuclear energy generation $\epsilon_{\mathrm{nuc}}$, weak neutrino loss $\epsilon_{\mathrm{neu}}$, and the full analytical Jacobian $(\partial \dot{X}_i/\partial X_j, \partial \dot{X}_i/\partial T, \partial \dot{X}_i/\partial \rho)$ using 2-lane SIMD dual numbers (`F64x2`) and thread-local preallocated workspace buffers (`with_thread_local_work`) across built-in networks (`basic.net`, `approx21.net`, `cno_extras.net`, `pp_and_cno_extras.net`, `co_burn.net`).
2. **Reverse-Mode `@jax.custom_vjp` Network Wrapper (`eval_net_jax`)**:
   Wraps `eval_net` in a `@jax.custom_vjp` rule backed by the exact analytical Jacobian from Safe Rust so that `jax.grad` differentiates through nuclear energy generation and abundance rates with zero finite-difference overhead.
3. **Stiff Single-Zone Burn Integrators & Temperature-Thresholded Operator Splitting (`burn_1_zone`, `burn_1_zone_const_density`, `burn_1_zone_const_p`)**:
   Integrates stiff single-zone burning trajectories at constant density (`burn_1_zone_const_density`) or constant pressure (`burn_1_zone_const_p`) using 4th-order L-stable Kaps-Rentrop / Hairer-Wanner `RODAS4` Rosenbrock and Bader-Deuflhard semi-implicit midpoint solvers backed by thread-local `NetWorkState` buffers. During coupled stellar evolution (`do_struct_burn_mix`), setting `op_split_burn = true` restricts operator-split stiff burning to hot shells satisfying $T_{\mathrm{start},k} \ge T_{\mathrm{op\_split,min}}$ (`op_split_burn_min_t = 2.0e9 K` by default) while retaining fully coupled implicit burning and mixing in cooler zones.

=== "Python & JAX (`flexmesa.chem`, `flexmesa.neu`, `flexmesa.net`)"

    ```python
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flexmesa import chem, net, neu

    # 1. Vectorized composition moments across 1024 zones
    isos = ("h1", "he4", "c12", "o16")
    x_grid = np.tile(np.array([0.70, 0.28, 0.008, 0.012]), (1024, 1))
    comp_cpu = chem.get_composition_info_zones(isos, x_grid, backend="rust")

    # 2. Thermal neutrino energy loss and JAX temperature gradient
    res_neu = neu.neu_get(
        t=1.0e9, log10_t=9.0, rho=1.0e6, log10_rho=6.0, abar=12.0, zbar=6.0
    )
    grad_neu_t = jax.grad(
        lambda t: neu.neu_pair_plas_jax(t, jnp.float64(1.0e6), 12.0, 6.0)
    )(jnp.float64(1.0e9))

    # 3. Differentiable basic.net nuclear network evaluation via @jax.custom_vjp
    info = net.get_net_info("basic.net")
    xa = jnp.array([0.70, 3e-5, 0.28, 0.0034, 0.001, 0.0096, 0.002, 0.0039702])
    c0 = chem.get_composition_info(info.chem_id, np.asarray(xa))
    grad_eps_t = jax.grad(
        lambda t: net.eval_net_jax(
            "basic.net",
            xa,
            t,
            jnp.float64(100.0),
            float(c0.abar),
            float(c0.zbar),
            float(c0.z2bar),
            float(c0.ye),
        )[0]
    )(jnp.float64(8.0e7))
    ```

=== "Safe Rust (`crates/flexmesa-chem`, `crates/flexmesa-neu`, `crates/flexmesa-net`)"

    ```rust
    use flexmesa_chem::{NuclideCatalog, get_composition_info};
    use flexmesa_net::{NetGeneralInfo, net_get};
    use flexmesa_neu::{NeuFlags, neu_get};
    use flexmesa_rates::ScreeningMode;

    let cat = NuclideCatalog::builtin();
    let ids: Vec<usize> = ["h1", "he4", "c12", "o16"]
        .iter()
        .map(|s| cat.get_nuclide_index(s).unwrap())
        .collect();
    let info = get_composition_info(&cat, &ids, &[0.70, 0.28, 0.008, 0.012], true).unwrap();

    let neu_out = neu_get(
        Some(1.0e9), Some(9.0), Some(1.0e6), Some(6.0),
        12.0, 6.0, 7.5, NeuFlags::all(),
    ).unwrap();

    let g = NetGeneralInfo::from_name("basic.net").unwrap();
    let xa = [0.70, 3e-5, 0.28, 0.0034, 0.001, 0.0096, 0.002, 0.0039702];
    let rf = vec![1.0; g.num_reactions];
    let net_out = net_get(
        &g, false, &xa, 8.0e7, 7.90309, 100.0, 2.0,
        1.29, 1.10, 1.54, 0.85, 0.0, 0.0, 0.0,
        &rf, 1.0, ScreeningMode::Extended,
    ).unwrap();
    ```
