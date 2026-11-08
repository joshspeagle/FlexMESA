# Wave 3b Microphysics: 4D Tabulated EOS, Opacity, Nuclear Rates, Networks & Atmospheres

This guide covers the physical formulations, spline interpolation schemes, automatic differentiation wrappers, and stiff reaction network solvers implemented across `flexmesa.eos` (Wave 3b.1), `flexmesa.kap` (Wave 3b.2), `flexmesa.rates` (Wave 3b.3), `flexmesa.net` (Wave 3b.4), and `flexmesa.atm` (Wave 3b.4).

---

## 1. 4D Tabulated Equations of State (`OPAL/SCVH`, `FreeEOS`, `CMS`) & Full `eosDT_get` Cascade

In cool stellar envelopes, brown dwarfs, and giant planets ($\log_{10} T \lesssim 6.5$) where partial ionization of $\mathrm{H}$, $\mathrm{He}$, and heavy elements cannot be represented by a fully ionized ideal gas, `FlexMESA` evaluates four-dimensional $(X, Z, \log_{10} Q, \log_{10} T)$ bicubic spline tables with rescaled density coordinate $\log_{10} Q = \log_{10}\rho - 2\log_{10} T + 12$:

1. **Combined `OPAL/SCVH` (`eval_opal_scvh_eos`) and `FreeEOS` (`eval_free_eos`)**:
   - Each $(X_m, Z_n)$ plane stores precomputed 2D not-a-knot bicubic spline coefficients (`[f, f_qq, f_tt, f_qqtt]`) for all 16 base thermodynamic channels (`lnPgas` through `eta`).
   - Within each plane, `eval_xtable_16` evaluates the 16 bicubic splines in a single pass and maps $(\partial/\partial \log_{10} Q, \partial/\partial \log_{10} T)$ to natural-logarithm $(\ln\rho, \ln T)$ derivatives. Across planes, `get1_eosdt_results` applies quadratic/cubic Lagrange interpolation in $X$ and linear interpolation in $Z$.
2. **Chabrier, Mazevet & Soubiran (`CMS`) H/He EOS (`eval_cms_eos`)**:
   - Combines pure-hydrogen ($X = 1$) and pure-helium ($Y = 1$) bicubic tables via additive volume and entropy mixing at constant $(P, T)$ with non-ideal H-He interaction corrections.
3. **Full 6-Component `eosDT_get` Priority Cascade & Ideal $\gamma$-Law Inversion (`eos_gamma_de_get`)**:
   - `eos_dt_get` blends across all six MESA EOS components (`CMS`, `PC`, `Skye`, `FreeEOS`, `OPAL/SCVH`, `HELM`) and the low-$(\rho, T)$ neutral `ideal` fallback using $C^2$ quintic polygonal transition weights.
   - `eos_gamma_de_get` provides an exact closed-form $(\rho, E) \mapsto T$ solver for an ideal $\gamma$-law gas plus radiation pressure by solving the depressed quartic $a T^4 + (\rho N_{\mathrm{A}} k_{\mathrm{B}} / [\bar{A}(\gamma - 1)]) T - \rho E = 0$ via Ferrari's resolvent cubic followed by two Newton-Raphson polishing steps.

---

## 2. Radiative, Conductive & Compton Opacity (`flexmesa.kap`)

`flexmesa.kap` (`crates/flexmesa-kap`) computes the Rosseland mean opacity $\kappa(\rho, T, \mathbf{X})$ ($\mathrm{cm^2\,g^{-1}}$) and its exact logarithmic derivatives $(\partial \ln\kappa/\partial \ln\rho)_T$ and $(\partial \ln\kappa/\partial \ln T)_\rho$:

1. **Radiative Tables (Type 1 Fixed-Metal, Type 2 C/O-Enhanced & Low-$T$ Molecular)**:
   - High-temperature radiative opacities (`kap_get_fixed_metal`, `kap_get_co`) interpolate 2D bicubic splines in $(\log_{10} R, \log_{10} T)$ where $\log_{10} R = \log_{10}\rho - 3\log_{10} T + 18$, with linear or monotonic cubic (`interp_pm`) interpolation across $X$ and $Z$, plus 2D biquadratic $(f_{\mathrm{C}}, f_{\mathrm{O}})$ C/O enhancement tables for helium-burning cores.
   - Low-temperature molecular and dust opacities cover Ferguson et al. (2005) (`lowT_fa05_gs98`, `gn93`, `a09p`), Freedman et al. (2008, 2014) (`lowT_Freedman11`), Lederer & Aringer (2009) C/N-varying `kapCN`, and Marigo & Aringer (2009) C/N/O-varying `AESOPUS`.
2. **Compton Scattering, Electron Conduction & Differentiable JAX Wrapper**:
   - At high temperature ($\log_{10} T \gtrsim 8.2$) or low $\log_{10} R$, radiative opacity blends smoothly via a $C^2$ quintic smootherstep into Poutanen (2017) relativistic/degenerate Compton scattering (`compton_opacity_jax`).
   - Radiative opacity combines harmonically ($1/\kappa = 1/\kappa_{\mathrm{rad}} + 1/\kappa_{\mathrm{cond}}$) with Potekhin (`condint.f`) and Blouin et al. (2020) H/He degenerate electron conduction (`combine_rad_with_conduction_jax`), and `make_differentiable_kap` exposes a `@jax.custom_jvp` wrapper propagating analytical $(\log_{10}\rho, \log_{10} T)$ derivatives through `jax.grad`.

![Wave 3b Rosseland opacity regimes and basic.net nuclear energy generation across Safe Rust and JAX](../assets/figures/wave3b_microphysics_diagnostics.png)

---

## 3. Nuclear Reaction Rates, Plasma Screening & Reaction Networks (`flexmesa.rates`, `flexmesa.net`)

1. **REACLIB Rates, Plasma Screening & Weak Reactions (`flexmesa.rates`)**:
   - `eval_reaclib_handle` and `eval_reaclib_jax` evaluate 7-parameter REACLIB forward and detailed-balance reverse rates across all 316 built-in MESA reactions.
   - `screen_pair` and `screen_extended_jax` / `screen_salpeter_jax` / `screen_chugunov_jax` evaluate weak, intermediate, and strong plasma electrostatic screening factors $\exp(H_{12})$ and their $(\rho, T)$ derivatives.
   - `eval_weak_pair` and `eval_ecapture_pair` compute 2D bicubic weak-rate tables (`LMP`, `Oda`, `FFN`, `Suzuki`) and Fermi-Dirac phase-space electron captures (`eval_ecapture`).
2. **Stiff Reaction Networks & Analytical Jacobians (`flexmesa.net`)**:
   - `eval_net` assembles molar abundance RHS vectors $dY_i/dt$, nuclear energy generation $\epsilon_{\mathrm{nuc}}$, neutrino losses $\epsilon_{\mathrm{neu}}$, and the full analytical Jacobian $(\partial \dot{X}_i/\partial X_j, \partial \dot{X}_i/\partial T, \partial \dot{X}_i/\partial \rho)$ using 2-lane SIMD dual numbers (`F64x2`) and thread-local preallocated workspace buffers (`with_thread_local_work`).
   - `eval_net_jax` wraps `eval_net` in a `@jax.custom_vjp` rule backed by the exact analytical Jacobian from Safe Rust, while `burn_1_zone` and `burn_1_zone_const_p` integrate stiff single-zone burning trajectories via 4th-order Kaps-Rentrop / Hairer-Wanner `RODAS4` Rosenbrock and Bader-Deuflhard semi-implicit midpoint solvers.

---

## 4. Stellar Model Atmospheres (`flexmesa.atm`)

`flexmesa.atm` (`crates/flexmesa-atm`) provides surface boundary conditions $(\ln T_{\mathrm{surf}}, \ln P_{\mathrm{surf}})$ and their analytical sensitivities with respect to $(L, \ln R, \ln M, \ln \kappa)$:

- **Analytic & Integrated $T(\tau)$ Relations (`eval_t_tau_uniform`, `eval_t_tau_varying`)**: Supports Eddington, solar Hopf, Krishna-Swamy (1966), and Trampedach et al. (2014) / Ball (2021) $T(\tau)$ relations with either uniform-opacity fixed-point iteration or Cash-Karp RK45 integration of $dP/d\tau = g/\kappa$.
- **Irradiated & Tabulated Atmospheres (`eval_irradiated`, `eval_table`)**: Implements Guillot (2010) irradiated grey planetary/binary atmospheres (`eval_e2` Steffen spline for $E_2(x)$) and 2D bicubic + 1D Steffen interpolation across the 7 MESA atmosphere table sets (`ATM_TABLE_PHOTOSPHERE`, `TAU_100`, `TAU_10`, `TAU_1`, `TAU_1M1`, `WD_TAU_25`, `DB_WD_TAU_25`).

=== "Python & JAX (`flexmesa`)"

    ```python
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flexmesa import atm, chem, constants, kap, net, rates

    # 1. Differentiable opacity evaluation with @jax.custom_jvp
    kap_fn = kap.make_differentiable_kap(z=0.02, x=0.70, zbase=0.02, zbar=1.28)
    dlogkap_dlogrho, dlogkap_dlogt = jax.grad(kap_fn, argnums=(0, 1))(
        jnp.float64(-5.0), jnp.float64(6.0)
    )

    # 2. Differentiable nuclear network evaluation with @jax.custom_vjp
    info = net.get_net_info("basic.net")
    xa = jnp.array([0.70, 3e-5, 0.28, 0.0034, 0.001, 0.0096, 0.002, 0.0039702])
    comp = chem.get_composition_info(info.chem_id, np.asarray(xa))
    grad_eps_t = jax.grad(
        lambda t: net.eval_net_jax(
            "basic.net",
            xa,
            t,
            jnp.float64(100.0),
            float(comp.abar),
            float(comp.zbar),
            float(comp.z2bar),
            float(comp.ye),
        )[0]
    )(jnp.float64(8.0e7))

    # 3. Tabulated photosphere boundary condition (tau = 2/3)
    surf = atm.eval_table(
        l=constants.LSUN,
        r=constants.RSUN,
        m=constants.MSUN,
        cgrav=constants.STANDARD_CGRAV,
        id=atm.ATM_TABLE_PHOTOSPHERE,
        z=0.0179,
        skip_partials=False,
        teff=5777.0,
    )
    ```

=== "Safe Rust (`crates/`)"

    ```rust
    use flexmesa_atm::{ATM_TABLE_PHOTOSPHERE, eval_table};
    use flexmesa_const::{CLIGHT, CRAD, MSUN, PI, RSUN, STANDARD_CGRAV};
    use flexmesa_kap::{KapConfig, get_kap_results};
    use flexmesa_net::{NetGeneralInfo, net_get};
    use flexmesa_rates::ScreeningMode;

    // 1. Evaluate Type 1 + conductive + Compton opacity
    let kap_out = get_kap_results(
        &KapConfig::default(), 1.28, 0.70, 0.02, 0.004, 0.001, 0.009, 0.003,
        -5.0, 6.0, -0.16, 0.0, 0.0, -10.0, 1.0, -1.5,
    ).unwrap();

    // 2. Evaluate basic.net RHS and analytical Jacobian
    let g = NetGeneralInfo::from_name("basic.net").unwrap();
    let xa = [0.70, 3e-5, 0.28, 0.0034, 0.001, 0.0096, 0.002, 0.0039702];
    let rf = vec![1.0; g.num_reactions];
    let net_out = net_get(
        &g, false, &xa, 8.0e7, 7.90309, 100.0, 2.0,
        1.29, 1.10, 1.54, 0.85, 0.0, 0.0, 0.0,
        &rf, 1.0, ScreeningMode::Extended,
    ).unwrap();

    // 3. Evaluate photosphere table boundary condition
    let l_sun = PI * CRAD * CLIGHT * RSUN * RSUN * 5777.0_f64.powi(4);
    let atm_out = eval_table(
        l_sun, RSUN, MSUN, STANDARD_CGRAV, ATM_TABLE_PHOTOSPHERE,
        0.0179, false, 5777.0,
    ).unwrap();
    ```
