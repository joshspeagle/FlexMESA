# Convection, Element Diffusion, Shellular Rotation & Magnetic Dynamos (`turb`, `star`)

Heat and chemical transport across stellar interiors depend on the interplay between buoyancy-driven turbulence (`flexmesa.turb`), boundary mixing and overshooting, microscopic multi-fluid atomic diffusion, rotational distortion, and magnetohydrodynamic angular-momentum transport (`flexmesa.star`).

---

## 1. Local Mixing-Length Theory (`MLT`) & Time-Dependent Convection (`TDC`)

### 1.1 Local Mixing-Length Theory (`set_mlt`, `set_mlt_jax`)
In a shell where the radiative temperature gradient exceeds the Ledoux or Schwarzschild stability threshold ($\nabla_r > \nabla_L$), local Mixing-Length Theory solves the cubic convective efficiency equation (Cox & Giuli 1968, §14.82) for the dimensionless convective efficiency $\Gamma$:

$$a_0 \Gamma^3 + \Gamma^2 + \Gamma - a_0 B^3 = 0, \qquad B = A\,\left(\nabla_r - \nabla_L\right)^{1/2},$$

where the geometric prefactor $a_0$ and radiative-to-convective conductance ratio $A$ depend on the selected prescription (`Cox`, `TDC`, `Henyey`, `ML1`, `ML2`, or `Mihalas`). `FlexMESA` solves this depressed cubic analytically via hyperbolic and trigonometric Vieta-Cardano formulas in `DualOrder1<6>` (Safe Rust) and branchless `jnp.where` (JAX), recovering the superadiabaticity $Y$ and actual temperature gradient $\nabla_T$:

$$Y = \nabla_T - \nabla_L = \frac{\nabla_r - \nabla_L}{1 + a_0 \Gamma^2 (\Gamma + 1)^{-1}}, \qquad \nabla_T = \nabla_L + Y.$$

### 1.2 Kuhfuss (1986) 1D Time-Dependent Convection (`set_tdc`)
When stellar pulsation or rapid nuclear burning timescales approach the convective turnover time, steady-state Mixing-Length Theory breaks down because convective eddies cannot adjust instantaneously to changes in the local buoyancy force. `set_tdc` solves the time-integrated Kuhfuss (1986) turbulent kinetic energy equation over a step $\Delta t$ for the scaled convective speed $A_f(Y)$:

$$\frac{d A}{dt} = \Xi_0(Y) + \Xi_1 A + \Xi_2 A^2, \qquad Q(Y) = \left(L - L_0 \nabla_L\right) - L_0 Y - c_0 A_f(Y)\,Y_{\mathrm{env}}(Y) = 0.$$

To guarantee convergence across 80 orders of magnitude in $Y$, `set_tdc` searches in the logarithmic coordinate $Z = \ln|Y| \in [-100, 100]$:

1. **Bisection Bracketing**: Narrows the bracket $[Z_{\mathrm{lb}}, Z_{\mathrm{ub}}]$ until $Z_{\mathrm{ub}} - Z_{\mathrm{lb}} < 1.0$.
2. **Damped Newton-Raphson Refinement**: Iterates $Z_{n+1} = Z_n - Q(Z_n) / Q'(Z_n)$ with midpoint bracket clamping whenever a trial step exits $[Z_{\mathrm{lb}}, Z_{\mathrm{ub}}]$, terminating when $|Q| / \mathrm{scale} \le 10^{-8}$ or $|\Delta Z| < 10^{-13}$.
3. **Implicit Function Theorem Differentiation**: Once $Z^*$ converges in scalar `f64`, all partial derivatives of $(\nabla_T, v_{\mathrm{conv}}, D_{\mathrm{mix}})$ with respect to the 12 shell inputs $\theta = (\chi_T, \chi_\rho, C_P, g, H_P, \rho, P, T, \kappa, \nabla_r, \nabla_{\mathrm{ad}}, \nabla_L)$ are computed in a single analytical pass at $Y^* = \exp(Z^*)$:

   $$\frac{\partial Y^*}{\partial \theta} = -\left(\frac{\partial Q}{\partial Y}\bigg|_{Y^*}\right)^{-1} \frac{\partial Q}{\partial \theta}\bigg|_{Y^*}.$$

Because `FlexMESA` applies the Implicit Function Theorem at the converged scalar root rather than propagating a 70-component dual number through every intermediate Newton iteration, 4096-zone `turb_tdc` evaluation runs **$54.1\times$ faster in 4-thread Safe Rust** and **$146.0\times$ faster in batched GPU JAX** than 4-thread OpenMP Fortran MESA.

![Steady-state Cox MLT vs. transient Kuhfuss TDC temperature gradient and Implicit Function Theorem sensitivity](../assets/figures/turb_mlt_tdc_comparison.png)

*Figure 1: The left panel compares the steady-state Cox & Giuli (1968) Mixing-Length Theory temperature gradient $\nabla_T$ against transient Kuhfuss (1986) Time-Dependent Convection (`set_tdc`) across short, intermediate, and asymptotic timestep ratios $\Delta t / \tau_{\mathrm{conv}}$ as a function of radiative gradient $\nabla_r$. The right panel shows the exact analytical Implicit Function Theorem sensitivity $\partial \nabla_T / \partial \nabla_r$ overlaid with centered finite-difference evaluations.*

---

## 2. Semiconvection, Thermohaline Mixing & Convective Boundaries

### 2.1 Semiconvection, Salt-Finger Thermohaline Mixing & Brassard Brunt-Väisälä $B_k$ (`flexmesa.turb`, `flexmesa.star`)
- **Brassard et al. (1991) Finite-Difference EOS vs. $\nabla_\mu$ Composition Gradient (`do_brunt_b`)**: When `use_brunt_gradmux_form = false` (default, matching `star/private/brunt.f90:236-306`), `do_brunt_b` evaluates `eos_dt_get` at the interpolated face state $(\rho_f, T_f)$ with the compositions $\mathbf{X}_k$ and $\mathbf{X}_{k-1}$ of the adjacent cells:

  $$B_k = \frac{\ln\!\left[P_{\mathrm{rad},f} + P_{\mathrm{gas}}(\rho_f, T_f, \mathbf{X}_k)\right] - \ln\!\left[P_{\mathrm{rad},f} + P_{\mathrm{gas}}(\rho_f, T_f, \mathbf{X}_{k-1})\right]}{(\ln P_{\mathrm{eos},k-1} - \ln P_{\mathrm{eos},k})\,\chi_{T,f}},$$

  capturing partial-ionization and Coulomb/degeneracy shifts across chemical boundaries in both gas- and radiation-dominated interiors, while `use_brunt_gradmux_form = true` evaluates $B_k = (\chi_{\rho,f}/\chi_{T,f})\,(\Delta \ln \mu / \Delta \ln P_{\mathrm{eos}})$.
- **Langer, El Eid, & Fricke (1985) Semiconvection (`set_semiconvection` in `set_mlt_vars`)**: In Ledoux-stable, Schwarzschild-unstable faces ($\nabla_{\mathrm{ad}} < \nabla_r < \nabla_L$, $B_k > 0$) with `use_ledoux_criterion = true` and `alpha_semiconvection > 0`, `set_mlt_vars` dispatches to `set_semiconvection` (`Langer85GradTIsGradr` or `Langer85`) to evaluate compositional diffusion $D_{\mathrm{sc}}$ and $\nabla_T$ while zeroing `mlt_vc` (`turb_support.f90:440`):

  $$D_{\mathrm{sc}} = \alpha_{\mathrm{sc}}\,\frac{K_{\mathrm{rad}}}{6 c_P \rho}\,\frac{\nabla_r - \nabla_{\mathrm{ad}}}{\nabla_L - \nabla_r}, \qquad K_{\mathrm{rad}} = \frac{4 a c T^3}{3 \kappa \rho}.$$

- **Salt-Finger Thermohaline Mixing (`set_thermohaline` in `set_mlt_vars`)**: In Schwarzschild-stable faces ($\nabla_r < \nabla_L$) destabilized by an inverted mean-molecular-weight gradient ($B_k < 0$) with `thermohaline_coeff > 0`, `set_mlt_vars` dispatches to `set_thermohaline` (`Kippenhahn`, `TraxlerGaraudStellmach11`, or `BrownGaraudStellmach13`) to compute double-diffusive salt-finger diffusivity $D_{\mathrm{thrm}}$.

### 2.2 Overshooting, Convective Premixing (`CPM`) & Predictive Mixing (`flexmesa.star`)
- **16-Set Exponential & Step Convective Overshooting (`OvershootConfig`, `OvershootParamSet`, `update_mixing`)**: `model.update_mixing()` (backed by `flexmesa_star::locate_convection_boundaries` and `apply_overshooting` in Safe Rust) detects sub-cell Schwarzschild or Ledoux sign crossings $(\nabla_r - \nabla_{\mathrm{crit}} = 0)$, evaluates up to 16 `overshoot_sets` in upstream MESA priority order with `'none'` short-circuiting, optionally caps $H_{P,\mathrm{cb}}$ by the convection-zone radial thickness (`limit_overshoot_Hp_using_size_of_convection_zone`), applies Péclet-number thermal damping (`peclet_function_full_on`/`off`), and extends the convective diffusion coefficient beyond the boundary from an anchor point $r_0$ at distance $f_0 H_P$ inside the convection zone:

  $$D_{\mathrm{ov}}(r) = D_0 \exp\!\left(-\frac{2\,|r - r_0|}{f\,H_P}\right).$$

- **Convective Premixing (`do_conv_premix`)**: Following Paxton et al. (2019, §5), iteratively tests whether mixing an adjacent radiative shell into a convective core or shell causes that radiative shell to become convectively unstable ($\nabla_r > \nabla_{\mathrm{ad}}$), advancing the convective boundary to self-consistent neutrality before the coupled Henyey step.
- **Multi-Set Predictive Mixing (`add_predictive_mixing`, `apply_predictive_mixing_from_controls`)**: Following Paxton et al. (2018, §2), evaluates enabled `predictive_sets` across $q$ intervals to regulate convective boundary growth during core hydrogen and helium burning by limiting single-step mass ingestion (`ingestion_factor`) and enforcing a minimum superadiabaticity threshold ($\nabla_r / \nabla_{\mathrm{ad}} - 1 > \delta_{\mathrm{superad}}$) on the convective side of the boundary.

---

## 3. Burgers Multi-Fluid Atomic Diffusion & Crystallization Phase Separation

In radiative stellar envelopes, solar interiors, subdwarfs, and white dwarfs, microscopic gravitational settling, thermal diffusion, and concentration gradients stratify chemical elements over evolutionary timescales:

1. **Burgers (1969) Multi-Fluid Resistance Equations (`do_element_diffusion`, `eval_burgers_velocities_jax`)**:
   Solves the coupled Burgers (1969) momentum and heat-flow equations across all ions and electrons simultaneously via a tridiagonal Thomas solve, enforcing zero net charge density ($n_e = \sum_i Z_i n_i$) and zero electric current ($J_e = 0$) with Stanton & Murillo (2016) or Paquette et al. (1986) screened Coulomb collision integrals and radiative levitation accelerations. After both `do_element_diffusion` and `do_phase_separation`, species mass fractions are clamped to $[0, 1]$ and renormalized per zone with Neumaier-compensated summation so near-pure hydrogen white-dwarf envelopes ($X_{\mathrm{H}} \approx 1 - 10^{-15}$) remain strictly inside $[0, 1]$ during subsequent `kap` and `eos` table lookups (`clamp_unit_roundoff` in `set_basic_vars`).
2. **Caplan, Bauer & Freeman (2022) Strong-Coupling OCP (`eval_caplan_ocp_dstar_jax`)**:
   In strongly coupled white-dwarf and neutron-star interiors ($\Gamma \ge 1$), classical binary collision integrals break down. `FlexMESA` evaluates the Caplan, Bauer & Freeman (2022) screened one-component plasma (`OCP`) dimensionless self-diffusion parameterization $D^*(\Gamma, \kappa)$ to capture liquid-state correlation suppression.
3. **White-Dwarf Crystallization Phase Separation (`do_phase_separation`, `eval_phase_separation_shift_jax`)**:
   When a $\mathrm{C}/\mathrm{O}$ or $\mathrm{O}/\mathrm{Ne}$ white-dwarf core cools below the crystallization threshold ($\Gamma \ge \Gamma_{\mathrm{crit}} = 175.0$), the solid phase is enriched in the higher-$Z$ species ($\Delta X_{\mathrm{O}}(X_{\mathrm{O}}) > 0$) relative to the coexisting liquid. `do_phase_separation` shifts the solidifying core composition following Blouin et al. (2020) and Bauer (2023) and mixes the buoyant carbon-enriched liquid upward across the overlying convective mantle.

---

## 4. Roche Shellular Rotation, Angular Momentum Mixing, Five Instabilities & Spruit-Tayler Dynamo

### 4.1 Rotational Distortion Factors $(f_P, f_T)$, `solve_omega_mix` & Wind Boost
In the Endal & Sofia (1976) and Paxton et al. (2019) shellular rotation formalism, angular velocity $\Omega(m)$ is constant on isobars of volume-equivalent radius $r_P$. Centrifugal forces distort isobaric surfaces relative to the Eddington-reduced critical break-up rate $\Omega_{\mathrm{crit}} = \sqrt{G m (1 - \Gamma_{\mathrm{Edd}}) / r_P^3}$ (`eval_omega_crit`), enhance mass-loss rates near critical rotation in `do_set_mdot` alongside Björklund, Nugis-Lamers, and Dutch wind prescriptions (with differentiable JAX `z_surf = 0` Wolf-Rayet guards in `eval_winds_jax`), diffuse internal angular momentum implicitly via `solve_omega_mix`, and modify the hydrostatic pressure gradient and radiative temperature gradient via the dimensionless correction factors $(f_P, f_T)$ (`eval_rotation_fp_ft_jax`, `set_rotation_mixing_info`):

$$\frac{\partial P}{\partial m} = -\frac{G m}{4\pi r_P^4}\,f_P, \qquad \nabla_{\mathrm{rad,rot}} = \nabla_{\mathrm{rad}}\,\frac{f_T}{f_P}.$$

### 4.2 Five Hydrodynamic Instabilities & Spruit-Tayler (2002) Magnetic Dynamo
Differential rotation ($q = |\partial \ln\Omega / \partial \ln r| > 0$) drives both hydrodynamic shear instabilities and magnetohydrodynamic winding of poloidal magnetic fields into toroidal fields ($B_\phi \gg B_r$). `eval_rotational_mixing_zone_jax` and `set_rotation_mixing_info` combine five hydrodynamic instabilities with the Spruit (2002) magnetic Tayler dynamo using the thermal and compositional Brunt-Väisälä components $(N_T^2, N_\mu^2)$:

- **Dynamical Shear Instability (`DSI`)** and **Solberg-Høiland Instability (`SH`)**: Fast dynamical instabilities triggered when rotational shear or an inward decline in specific angular momentum overcomes buoyancy stabilization (Richardson criterion $R_i < R_{i,\mathrm{crit}} = 1/4$).
- **Secular Shear Instability (`SSI`)**: Radiative thermal diffusion weakens thermal buoyancy $N_T^2$ on small spatial scales, allowing shear turbulence to develop when the Péclet-reduced Richardson number falls below $1/4$.
- **Eddington-Sweet Meridional Circulation (`ES`)**: Large-scale thermal imbalance on centrifugally distorted isobars drives meridional circulation velocities $v_{\mathrm{ES}} \propto (\Omega / \Omega_{\mathrm{crit}})^2\,(L / m g)\,(N_T^2 + N_\mu^2)^{-1}$.
- **Goldreich-Schubert-Fricke Instability (`GSF`)**: Baroclinic axisymmetric instability driven by vertical gradients in angular velocity where thermal diffusion relaxes buoyancy restoring forces.
- **Spruit-Tayler (2002) Magnetic Dynamo (`D_ST`, `nu_ST`, $B_\phi$, $B_r$)**: Differential rotation winds a weak radial seed field $B_r$ into an azimuthal field $B_\phi$ until the pinch-type Tayler instability saturates, generating an effective magnetic viscosity $\nu_{\mathrm{ST}}$ (which dominates core-envelope angular momentum coupling in red giants and white-dwarf progenitors) and compositional diffusivity $D_{\mathrm{ST}}$.

=== "Python & JAX (`flexmesa.turb`, `flexmesa.star`)"

    ```python
    import jax
    import jax.numpy as jnp
    from flexmesa import star, turb

    # 1. Differentiate Cox MLT temperature gradient w.r.t. radiative gradient
    dgradt_dgradr = jax.grad(
        lambda gr: turb.set_mlt_jax(
            1.0, 1.0, 2.0e8, 2.7e4, 1.5e9,
            1.0e-5, 1.0e8, 1.0e5, 1.0, gr, 0.40, 0.40,
        )[1]
    )(jnp.float64(0.45))

    # 2. Evaluate Roche rotational distortion (f_P, f_T, i_rot/r^2) and Spruit-Tayler dynamo
    fp, ft, i_rot_over_r2 = star.eval_rotation_fp_ft_jax(
        w_div_w_crit=jnp.float64(1.0e-4 / 3.0e-4)
    )
    rot_mix = star.eval_rotational_mixing_zone_jax(
        r=jnp.float64(2.0e10),
        hp=jnp.float64(5.0e9),
        grav=jnp.float64(1.0e5),
        rho=jnp.float64(10.0),
        t=jnp.float64(2.0e7),
        cp=jnp.float64(3.5e8),
        opacity=jnp.float64(0.4),
        chi_t=jnp.float64(1.0),
        chi_rho=jnp.float64(1.0),
        grada=jnp.float64(0.40),
        gradt=jnp.float64(0.35),
        brunt_n2_t=jnp.float64(1.0e-6),
        brunt_n2_mu=jnp.float64(2.0e-7),
        omega=jnp.float64(1.0e-4),
        dln_omega_dln_r=jnp.float64(-0.8),
        abar=jnp.float64(1.3),
        zbar=jnp.float64(1.1),
    )
    ```

=== "Safe Rust (`crates/flexmesa-turb`, `crates/flexmesa-star`)"

    ```rust
    use flexmesa_autodiff::DualOrder1;
    use flexmesa_turb::{MltInputs, MltParams, set_mlt};

    fn main() {
        let gradr = DualOrder1::<6>::independent(0.45, 5);
        let inp = MltInputs {
            chi_t: DualOrder1::constant(1.0),
            chi_rho: DualOrder1::constant(1.0),
            cp: DualOrder1::constant(2.0e8),
            grav: DualOrder1::constant(2.7e4),
            lambda: DualOrder1::constant(1.5e9),
            rho: DualOrder1::constant(1.0e-5),
            p: DualOrder1::constant(1.0e8),
            t: DualOrder1::constant(1.0e5),
            opacity: DualOrder1::constant(1.0),
            gradr,
            grada: DualOrder1::constant(0.40),
            gradl: DualOrder1::constant(0.40),
        };
        let mlt = set_mlt(&MltParams::cox(2.0), &inp).unwrap();
        println!("grad_T = {:.6}, d(grad_T)/d(grad_r) = {:.6}", mlt.grad_t.val, mlt.grad_t.d1[5]);
    }
    ```
