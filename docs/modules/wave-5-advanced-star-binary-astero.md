# Wave 5 Advanced Stellar Physics, Interacting Binaries & Asteroseismology

Wave 5 extends the 1D stellar evolution engine (`flexmesa.star`) with atomic diffusion, rotational distortion and magnetohydrodynamic angular-momentum transport, 1D spherical HLLC Riemann hydrodynamics, non-linear radial stellar pulsation (`RSP2`) equations, pulsation structure exporters (`FGONG`, `GYRE`, `OSC`), coupled interacting binary star evolution (`flexmesa.binary`), and asteroseismic forward modeling and optimization (`flexmesa.astero`).

---

## 1. Atomic Diffusion, Rotational Mixing & Spruit-Tayler Dynamo (`Wave 5.1`)

### Physical Formulation

1. **Burgers Multi-Fluid Atomic Diffusion & Caplan et al. (2022) Strong-Coupling OCP**:
   - Multi-species gravitational settling, thermal diffusion, and concentration diffusion follow the Burgers (1969) resistance equations coupled with Gauss-Seidel electric field neutrality ($n_e = \sum_i Z_i n_i, J_e = 0$).
   - In strongly coupled white-dwarf and neutron-star envelopes ($\Gamma \ge 1$), self-diffusion coefficients follow the Caplan, Bauer & Freeman (2022) screened one-component plasma (OCP) parameterization $D^*(\Gamma, \kappa)$.
   - At first-order liquid-solid crystallization ($\Gamma \ge \Gamma_{\mathrm{crit}} = 175.0$), C/O/Ne phase separation redistributes latent heat and shifts core composition by $\Delta X_{\mathrm{O}}(X_{\mathrm{O}})$ following Blouin et al. (2020) and Bauer (2023).
2. **Shellular Rotation, Five Hydrodynamic Instabilities & Spruit-Tayler (2002) Dynamo**:
   - On isobars distorted by angular velocity $\Omega$ relative to the Eddington-reduced critical break-up rate $\Omega_{\mathrm{crit}} = \sqrt{G M (1 - \Gamma_{\mathrm{Edd}}) / R^3}$, pressure and radiative temperature corrections $(f_P, f_T)$ follow Endal & Sofia (1976) and Paxton et al. (2019).
   - Five secular and dynamical shear instabilities are combined with the Spruit (2002) magnetic Tayler instability (`D_ST`, `nu_ST`, azimuthal field $B_\phi$, and radial field $B_r$) using thermal and compositional Brunt-Vaisala components $(N_T^2, N_\mu^2)$:
     - **Dynamical Shear Instability (`DSI`)** and **Solberg-Hoiland (`SH`)**
     - **Secular Shear Instability (`SSI`)**
     - **Eddington-Sweet Meridional Circulation (`ES`)**
     - **Goldreich-Schubert-Fricke (`GSF`)**

---

## 2. 1D Spherical HLLC Riemann Hydrodynamics, `RSP2` & Pulsation Export (`Wave 5.2`)

1. **Well-Balanced 1D Spherical Lagrangian HLLC Riemann Solver**:
   - Across cell interfaces $r_k$, pressure and acoustic wave speeds $(S_L, S_M, S_R)$ incorporate a hydrostatic gravitational jump $\Delta P_{\mathrm{grav}} = \frac{1}{2}(dm_k + dm_{k-1}) g_k / (4\pi r_k^2)$ so that a hydrostatic profile ($u = 0$) satisfies $u_{\mathrm{face}} = 0$ to machine precision (`hydro_riemann.f90`).
2. **Kuhfuss (1986) / Smolec & Moskalik (2008) `RSP2` Turbulent Convection**:
   - Evolves specific turbulent kinetic energy $e_t = w^2$ with turbulent pressure $P_t = \alpha_p \rho w^2$, convective luminosity $L_{\mathrm{conv}}$, radiative cooling $\Pi$, buoyant source $S$, turbulent dissipation $D \propto w^3 / \Lambda$, and artificial viscosity $U_q$.
3. **Dziembowski Variables & Pulsation Exporters (`FGONG`, `GYRE`, `OSC`)**:
   - Evaluates dimensionless pulsation structure variables:

$$
V = \frac{G m \rho}{P r}, \quad V_g = \frac{V}{\Gamma_1}, \quad A^* = \frac{r N^2}{g}, \quad U = \frac{4\pi r^3 \rho}{m}, \quad c_1 = \frac{(r/R_\star)^3}{m/M_\star}
$$

   - Exports stellar profiles to standard `FGONG` (`ivers = 1300`), `GYRE` (`version 101`), and `OSC` text formats with full roundtrip parsers.

---

## 3. Interacting Binary Evolution (`Wave 5.3`: `flexmesa.binary`)

![Ritter (1988) RLOF mass-transfer curve and Kerr black-hole ISCO spin evolution](../assets/figures/wave5_binary_rlof_evolution.png)

### Orbital Mechanics, RLOF Mass Transfer & Compact Accretors

1. **Eggleton (1983) Roche-Lobe Radius & Periastron Gap**:
   - For mass ratio $q = M_1 / M_2$, the volume-equivalent Roche-lobe radius is:

$$
\frac{R_{\mathrm{RL},1}}{a} = \frac{0.49\,q^{2/3}}{0.6\,q^{2/3} + \ln\!\left(1 + q^{1/3}\right)}, \quad \delta_{\mathrm{RL},1} = \frac{R_1 - R_{\mathrm{RL},1}(1 - e)}{R_{\mathrm{RL},1}(1 - e)}
$$

2. **Ritter (1988), Kolb & Ritter (1990) & Implicit Bisection-Brent RLOF Solver**:
   - Combines optically thin isothermal atmospheric flow ($\dot{M}_{\mathrm{thin}} \propto -\rho_{\mathrm{ph}} c_s^3 R_{\mathrm{RL},1}^3 / (G M_1) \exp(\delta_{\mathrm{RL},1} / (H_P / R_{\mathrm{RL},1}))$) with optically thick sub-photospheric Polytropic/Simpson integration ($\dot{M}_{\mathrm{thick}}$) and converges implicit timestep mass-transfer rates via Hybrid Bisection-Brent (`solve_implicit_rlof_mdot`) or differentiable JAX Newton-bisection (`solve_roche_lobe_mdot_jax`).
3. **Kerr Black-Hole ISCO Spin (`Bardeen 1972`), Orbital $\dot{J}$ & Hut (1981) Tides**:
   - Tracks Kerr spin $a_* \in [0, 0.998]$ and radiative efficiency $\eta_{\mathrm{ISCO}} = 1 - \sqrt{1 - 2 / (3 r_{\mathrm{ISCO}})}$ under Eddington-limited accretion (`eval_bh_spin_and_edd_jax`), Peters (1964) gravitational-wave radiation ($\dot{J}_{\mathrm{gr}}, \dot{e}_{\mathrm{gr}}$), Rappaport, Verbunt & Joss (1983) magnetic braking ($\dot{J}_{\mathrm{mb}}$), and Hut (1981) weak-friction tidal circularization and synchronization.

---

## 4. Asteroseismology & Model Fitting (`Wave 5.4`: `flexmesa.astero`)

![Asteroseismic echelle diagram with surface-effect corrections and 5-point separation ratios](../assets/figures/wave5_astero_echelle_surface_corr.png)

### Scaling Relations, Surface Corrections & Optimization Drivers

1. **Asymptotic & Solar-Scaled Seismic Observables**:
   - Computes cumulative acoustic radius $\tau(r) = \int_0^r dr'/c_s(r')$, acoustic depth, total acoustic transit time $\tau_0$, asymptotic and solar-scaled $\Delta\nu$, $\nu_{\max}$, acoustic cutoff $\nu_{\mathrm{ac}}$, traditional and Bildsten et al. (2012) dipole g-mode period spacing $\Delta\Pi_1 = \sqrt{2}\pi^2 / \int (N/r)\,dr$, and base-of-convection-zone (BCZ) and He II ionization glitch acoustic depths ($\tau_{\mathrm{BCZ}}, \tau_{\mathrm{He\,II}}$).
2. **Five Surface-Effect Frequency Correction Schemes**:
   - Implements Kjeldsen et al. (2008) fixed power-law (`kjeldsen`), Ball & Gizon (2014) 1-term cubic (`cubic`, $\delta\nu = a_3 (\nu/\nu_{\max})^3 / \mathcal{I}$) and 2-term combined (`combined`, $\delta\nu = [a_{-1}(\nu/\nu_{\max})^{-1} + a_3(\nu/\nu_{\max})^3] / \mathcal{I}$), two-parameter power-law (`power_law`), and Sonoi et al. (2015) Lorentzian (`sonoi`) corrections, alongside Roxburgh & Vorontsov (2003) 5-point frequency separation ratios $(r_{01}, r_{10}, r_{02})$ that cancel near-surface layers.
3. **Optimization Drivers**:
   - Provides Nelder-Mead simplex (`nelder_mead_minimize`), Powell quadratic trust-region interpolation (`powell_quadratic_minimize`), and Cartesian grid scanning (`scan_cartesian_grid`) over composite seismic and spectroscopic objectives $\chi^2_{\mathrm{total}} = f_{\mathrm{seismo}}\chi^2_{\mathrm{seismo}} + (1 - f_{\mathrm{seismo}})\chi^2_{\mathrm{spectro}}$.

---

## 5. Side-by-Side Usage Examples

=== "Python & JAX (`flexmesa.star`, `flexmesa.binary`, `flexmesa.astero`)"

    ```python
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flexmesa import astero, binary, constants, star

    # 1. Differentiable Ritter (1988) RLOF rate sensitivity d(mdot)/d(R_donor)
    grad_mdot_r = jax.grad(
        lambda r_d: binary.eval_ritter_mdot_thin_jax(
            2.0 * constants.MSUN,
            1.4 * constants.MSUN,
            r_d,
            2.5 * constants.RSUN,
            2.0e-7,
            2.5e5,
            6500.0,
            0.61,
            False,
        )
    )(jnp.float64(2.502 * constants.RSUN))

    # 2. Ball & Gizon (2014) 2-term surface correction on radial modes
    n_ord = np.arange(14, 26, dtype=np.float64)
    nu_mod = 135.0 * (n_ord + 1.45)
    inertia = 1.0e-7 * np.exp((3100.0 - nu_mod) / 1000.0)
    sigma = np.full_like(nu_mod, 0.15)
    nu_obs = nu_mod - 1.2 * ((nu_mod / 3090.0) ** 3) / inertia
    corr = astero.eval_surface_correction(
        "combined", nu_obs, sigma, nu_mod, inertia, nu_max=3090.0
    )
    ```

=== "Safe Rust (`flexmesa-star`, `flexmesa-binary`, `flexmesa-astero`)"

    ```rust
    use flexmesa_astero::{SurfaceCorrectionScheme, eval_surface_correction};
    use flexmesa_binary::{eval_rlobe, eval_ritter_mdot_thin};
    use flexmesa_const::{MSUN, RSUN};

    fn main() {
        let rl_donor = eval_rlobe(2.0 * MSUN, 1.4 * MSUN, 6.5 * RSUN);
        let mdot = eval_ritter_mdot_thin(
            2.0 * MSUN, 1.4 * MSUN, 1.001 * rl_donor, rl_donor,
            2.0e-7, 2.5e5, 6500.0, 0.61, false,
        );
        println!("RLOF rate = {:.6e} g/s", mdot);
    }
    ```
