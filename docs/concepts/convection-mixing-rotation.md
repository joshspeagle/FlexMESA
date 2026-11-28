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
3. **Partial Derivatives of the Root**: Both the Safe Rust engine and the Python twin (`flexmesa.turb.tdc_dual.AutoDiffRealTdc`) run the Newton iteration on a nested dual number (`AutoDiffRealTdc`, upstream's `auto_diff_real_tdc`) that carries the value, its derivative with respect to $Z$, the structure partials and the mixed second partials (and the JAX benchmark kernel propagates `jax.jvp` through the same bisection-stopped Newton updates). Each update $Z + (-Q / Q')$ therefore carries structure partials forward, and the returned $Y$ has the partials of the last update, as in `turb/private/tdc.f90:253-345`. They approach the Implicit Function Theorem result

   $$\frac{\partial Y^*}{\partial \theta} = -\left(\frac{\partial Q}{\partial Y}\bigg|_{Y^*}\right)^{-1} \frac{\partial Q}{\partial \theta}\bigg|_{Y^*}$$

   as the iteration converges, and differ from it by an amount of the order of the last Newton correction.

![Steady-state Cox MLT vs. transient Kuhfuss TDC temperature gradient and Implicit Function Theorem sensitivity](../assets/figures/turb_mlt_tdc_comparison.png)

*Figure 1: The left panel compares the steady-state Cox & Giuli (1968) Mixing-Length Theory temperature gradient $\nabla_T$ against transient Kuhfuss (1986) Time-Dependent Convection (`set_tdc`) at fixed $\Delta t = 15\,\mathrm{s}$ as a function of radiative gradient $\nabla_r$. The right panel shows the analytical Implicit Function Theorem sensitivity $\partial \nabla_T / \partial \nabla_r$ for steady Cox MLT and transient TDC ($\Delta t = 15\,\mathrm{s}$).*

---

## 2. Semiconvection, Thermohaline Mixing & Convective Boundaries

### 2.1 Semiconvection, Salt-Finger Thermohaline Mixing & Brassard Brunt-Väisälä $B_k$ (`flexmesa.turb`, `flexmesa.star`)
- **Brassard et al. (1991) Finite-Difference EOS vs. $\nabla_\mu$ Composition Gradient (`do_brunt_b`)**: When `use_brunt_gradmux_form = false` (default, matching `star/private/brunt.f90:236-306`), `do_brunt_b` evaluates `eos_dt_get` at the interpolated face state $(\rho_f, T_f)$ with the compositions $\mathbf{X}_k$ and $\mathbf{X}_{k-1}$ of the adjacent cells:

  $$B_k = \frac{\ln\!\left[P_{\mathrm{rad},f} + P_{\mathrm{gas}}(\rho_f, T_f, \mathbf{X}_k)\right] - \ln\!\left[P_{\mathrm{rad},f} + P_{\mathrm{gas}}(\rho_f, T_f, \mathbf{X}_{k-1})\right]}{(\ln P_{\mathrm{eos},k-1} - \ln P_{\mathrm{eos},k})\,\chi_{T,f}},$$

  capturing partial-ionization and Coulomb/degeneracy shifts across chemical boundaries in both gas- and radiation-dominated interiors, while `use_brunt_gradmux_form = true` evaluates $B_k = (\chi_{\rho,f}/\chi_{T,f})\,(\Delta \ln \mu / \Delta \ln P_{\mathrm{eos}})$.
- **Langer, El Eid, & Fricke (1985) Semiconvection (`set_semiconvection` in `set_mlt_vars`)**: In Ledoux-stable, Schwarzschild-unstable faces ($\nabla_{\mathrm{ad}} < \nabla_r < \nabla_L$, $B_k > 0$) with `use_ledoux_criterion = true` and `alpha_semiconvection > 0`, `set_mlt_vars` dispatches to `set_semiconvection` (`Langer85GradTIsGradr` or `Langer85`) to evaluate compositional diffusion $D_{\mathrm{sc}}$ and $\nabla_T$ while zeroing `mlt_vc` (`star/private/turb_support.f90:408-430`):

  $$D_{\mathrm{sc}} = \alpha_{\mathrm{sc}}\,\frac{K_{\mathrm{rad}}}{6 c_P \rho}\,\frac{\nabla_r - \nabla_{\mathrm{ad}}}{\nabla_L - \nabla_r}, \qquad K_{\mathrm{rad}} = \frac{4 a c T^3}{3 \kappa \rho}.$$

- **Salt-Finger Thermohaline Mixing (`set_thermohaline` in `set_mlt_vars`)**: In Schwarzschild-stable faces ($\nabla_r < \nabla_L$) destabilized by an inverted mean-molecular-weight gradient ($B_k < 0$) with `thermohaline_coeff > 0`, `set_mlt_vars` dispatches to `set_thermohaline` (`Kippenhahn`, `TraxlerGaraudStellmach11`, or `BrownGaraudStellmach13`) to compute double-diffusive salt-finger diffusivity $D_{\mathrm{thrm}}$.

### 2.2 Overshooting, Convective Premixing (`CPM`) & Predictive Mixing (`flexmesa.star`)
- **16-Set Exponential & Step Convective Overshooting (`OvershootConfig`, `OvershootParamSet`, `update_mixing`)**: `model.update_mixing()` (backed by `flexmesa_star::locate_convection_boundaries` and `add_overshooting` in Safe Rust) detects sub-cell Schwarzschild or Ledoux sign crossings $(\nabla_r - \nabla_{\mathrm{crit}} = 0)$, evaluates up to 16 `overshoot_sets` in upstream MESA priority order with `'none'` short-circuiting and case-sensitive selector matching (`star/private/overshoot.f90`), caps $H_{P,\mathrm{cb}}$ by the convection-zone radial thickness (`limit_overshoot_Hp_using_size_of_convection_zone`), and extends the convective diffusion coefficient beyond the boundary from an anchor point $r_0$ at distance $f_0 H_P$ inside the convection zone:

  $$D_{\mathrm{ov}}(r) = D_0 \exp\!\left(-\frac{2\,|r - r_0|}{f\,H_P}\right).$$

- **Convective Premixing (`do_conv_premix`)** *(partial/unverified; `#155`)*: Inspired by Paxton et al. (2019, §5), iteratively tests whether mixing an adjacent radiative shell into a convective core or shell causes that shell to exceed $\nabla_r > \nabla_{\mathrm{ad}}$; how closely `conv_premix` follows upstream `26.4.1` is recorded routine by routine in `notes/audit/routine_table.md` and tracked under `#155`.
- **Multi-Set Predictive Mixing (`add_predictive_mixing`, `apply_predictive_mixing_from_controls`)** *(partial/unverified; `#155`)*: Evaluates enabled `predictive_sets` across $q$ intervals to regulate convective boundary growth (`ingestion_factor`, $\delta_{\mathrm{superad}}$); full upstream alignment is tracked in `#155`.

---

## 3. Burgers Multi-Fluid Atomic Diffusion & Crystallization Phase Separation

In radiative stellar envelopes, solar interiors, subdwarfs, and white dwarfs, microscopic gravitational settling, thermal diffusion, and concentration gradients stratify chemical elements over evolutionary timescales *(the Safe Rust diffusion and phase-separation kernels are partly ported from MESA `26.4.1` in `#156` and compared with compiled Fortran reference vectors; routine-level evidence is in `notes/audit/routine_table.md`. Monochromatic radiative levitation `op_mono` fails loudly pending C10, and end-to-end stellar-track agreement is tracked under `#143` and `notes/audit/case_table.md`)*:

1. **Burgers (1969) Multi-Fluid Resistance Equations (`do_element_diffusion_step`, `do_solve_diffusion`, `get_matrix_coeffs`)**:
   Solves the coupled multi-species Burgers (1969) momentum and heat-flow equations per diffusion class across active stellar faces via Thoul-Hu (`do1_solve_thoul_hu`) or CGS (`solve_burgers_cgs_with_thermal` / `solve_burgers_cgs_no_thermal` with linear degeneracy blending across `cgs_thermal_diffusion_eta_full_{on,off}`), using Paquette et al. (1986) spline collision integrals (`paquette_coefficients`), pure-Coulomb Debye-Huckel resistance, or Stanton & Murillo (2016) effective screening (`get_sm_coeffs`). Time integration supports adaptive backward-Euler substepping (`BE_solve`) and block-tridiagonal Rosenbrock `solve_with_isolve` across all seven MESA solvers (`ros2_solver` through `rodasp_solver`), together with `fixup`, `smooth_x`, `redistribute_mass`, and `finish_element_diffusion` (`eps_diffusion` and `eps_WD_sedimentation`).
2. **Caplan, Bauer & Freeman (2022) Strong-Coupling OCP (`get_cbf_coeffs`, `get_dstar_ocp`, `kappa_cbf`)**:
   In strongly coupled white-dwarf and neutron-star interiors ($\Gamma > 4$), `FlexMESA` blends Stanton & Murillo (2016) into the Caplan, Bauer & Freeman (2022) screened one-component plasma (`OCP`) dimensionless self-diffusion parameterization $D^*(\Gamma, \kappa)$ (with `bitsafe_erf_fit` and relativistic electron screening for $\rho \ge 10^6\,\mathrm{g\,cm^{-3}}$).
3. **White-Dwarf Crystallization Phase Separation (`do_phase_separation_step`, `blouin_delta_xo`, `blouin_delta_xne`)**:
   When a $\mathrm{C}/\mathrm{O}$ or $\mathrm{O}/\mathrm{Ne}$ white-dwarf core crosses the Skye EOS crystallization boundary (`phase > 0.9`), the solid lattice is enriched in the higher-$Z$ species relative to the coexisting liquid using the Blouin et al. (2021a,b) number-fraction fits (`'CO'` and `'ONe'`, matching MESA `26.4.1` `phase_separation.f90`, where three-component Medin-Cumming tables are absent). `do_phase_separation_step` updates thermodynamic state at constant $P_{\mathrm{gas}}$ with `Skye_use_ion_offsets = .false.`, mixes the buoyant carbon- or oxygen-enriched liquid outward via Ledoux/Brunt-B or mean-molecular-weight stability (`mix_outward`), records `eps_phase_separation`, and marks crystallized and phase-separation mixing regions (`MixingType::Crystallized`, `MixingType::PhaseSeparationMixing`).

---

## 4. Roche Shellular Rotation, Angular Momentum Mixing, Five Instabilities & Spruit-Tayler Dynamo

*(Note: Integrated stellar evolution with rotation as coupled solver variables and Spruit-Tayler transport remains unverified against MESA `26.4.1` tracks, with related cases failing or timing out under `#158`.)*

### 4.1 Rotational Distortion Factors $(f_P, f_T)$, `solve_omega_mix` & Wind Boost
In the Endal & Sofia (1976) and Paxton et al. (2019) shellular rotation formalism, angular velocity $\Omega(m)$ is constant on isobars of volume-equivalent radius $r_P$. Centrifugal forces distort isobaric surfaces relative to the Eddington-reduced critical break-up rate $\Omega_{\mathrm{crit}} = \sqrt{G m (1 - \Gamma_{\mathrm{Edd}}) / r_P^3}$ (`eval_omega_crit`), enhance mass-loss rates near critical rotation in `do_set_mdot` using the optical-depth surface-averaged `v_div_v_crit_avg_surf` shared across `flexmesa-star` and `flexmesa-binary` alongside Reimers, Blocker, Vink, Nieuwenhuijzen, de Jager, van Loon, Björklund, Nugis-Lamers, Dutch, `'other'`, super-Eddington (`0.5 * vesc2` and start-of-step `prev_Ledd`), and single-star RLO winds (`#159`; frozen JAX helpers remain in `eval_winds_jax`), diffuse internal angular momentum implicitly via `solve_omega_mix`, and modify the hydrostatic pressure gradient and radiative temperature gradient via the dimensionless correction factors $(f_P, f_T)$ (`eval_rotation_fp_ft_jax`, `set_rotation_mixing_info`):

$$\frac{\partial P}{\partial m} = -\frac{G m}{4\pi r_P^4}\,f_P, \qquad \nabla_{\mathrm{rad,rot}} = \nabla_{\mathrm{rad}}\,\frac{f_T}{f_P}.$$

### 4.2 Five Hydrodynamic Instabilities & Spruit-Tayler (2002) Magnetic Dynamo
Differential rotation ($q = |\partial \ln\Omega / \partial \ln r| > 0$) drives both hydrodynamic shear instabilities and magnetohydrodynamic winding of poloidal magnetic fields into toroidal fields ($B_\phi \gg B_r$). `eval_rotational_mixing_zone_jax` and `set_rotation_mixing_info` combine five hydrodynamic instabilities with the Spruit (2002) magnetic Tayler dynamo using the thermal and compositional Brunt-Väisälä components $(N_T^2, N_\mu^2)$:

- **Dynamical Shear Instability (`DSI`)** and **Solberg-Høiland Instability (`SH`)**: Fast dynamical instabilities triggered when rotational shear or an inward decline in specific angular momentum overcomes buoyancy stabilization (Richardson criterion $R_i < R_{i,\mathrm{crit}} = 1/4$).
- **Secular Shear Instability (`SSI`)**: Radiative thermal diffusion weakens thermal buoyancy $N_T^2$ on small spatial scales, allowing shear turbulence to develop when the Péclet-reduced Richardson number falls below $1/4$.
- **Eddington-Sweet Meridional Circulation (`ES`)**: Large-scale thermal imbalance on centrifugally distorted isobars drives meridional circulation velocities $v_{\mathrm{ES}} \propto (\Omega / \Omega_{\mathrm{crit}})^2\,(L / m g)\,(N_T^2 + N_\mu^2)^{-1}$.
- **Goldreich-Schubert-Fricke Instability (`GSF`)**: Baroclinic axisymmetric instability driven by vertical gradients in angular velocity where thermal diffusion relaxes buoyancy restoring forces.
- **Spruit-Tayler (2002) Magnetic Dynamo (`D_ST`, `nu_ST`, $B_\phi$, $B_r$)**: Differential rotation winds a weak radial seed field $B_r$ into an azimuthal field $B_\phi$ until the pinch-type Tayler instability saturates, generating an effective magnetic viscosity $\nu_{\mathrm{ST}}$ (which dominates core-envelope angular momentum coupling in red giants and white-dwarf progenitors) and compositional diffusivity $D_{\mathrm{ST}}$.

`update_rotation_mixing_info` turns these coefficients into the rotational chemical diffusivity $D_{\mathrm{mix,rot}} = f\,f_c\,\sum_i f_i D_i$ (with `am_D_mix_factor` $f_c$, the `D_*_factor` weights $f_i$ and an optical-depth switch $f$) and into the angular-momentum viscosities `am_nu_rot`, `am_nu_omega` and `am_nu_j`. With `&star_job` `change_D_omega_flag = .true.` and `new_D_omega_flag = .true.`, the weighted sum is first stored as $D_\omega$, smoothed over `smooth_D_omega` cells and diffused between faces by one implicit step at rate `D_omega_mixing_rate`. $D_\omega$ is zeroed in convective cells with $q \le$ `max_q_for_D_omega_zero_in_convection_region`, and it then replaces the sum in $D_{\mathrm{mix,rot}}$ and, unless `use_D_omega_for_am_nu_rot = .false.`, sets `am_nu_rot` $= $ `am_nu_factor` $\cdot D_\omega$. `change_am_nu_rot_flag` applies the same smoothing and mixing to `am_nu_rot` itself (`smooth_am_nu_rot`, `nu_omega_mixing_rate`). The Rust routines follow `star/private/rotation_mix_info.f90` and `star/private/mix_info.f90` of MESA `26.4.1`.

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
