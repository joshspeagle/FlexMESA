# 1D Stellar Structure, Henyey Solver, HLLC Hydrodynamics & RSP2 Pulsation (`flexmesa.star`)

`flexmesa.star` (`crates/flexmesa-star` and `src/flexmesa/star/`) implements the 1D Lagrangian hydrostatic and hydrodynamic stellar evolution engine, coupling equations of state (`eos`), radiative and conductive opacities (`kap`), screened nuclear reaction networks (`rates`, `net`), thermal neutrino cooling (`neu`), turbulent convective mixing (`turb`), and surface model atmospheres (`atm`) into a unified block-tridiagonal Newton-Raphson solver.

![Internal convective structure of a 4 Msun ZAMS star and 8-mass ZAMS HR diagram](../assets/figures/wave4_star_structure_evolution.png)

*Figure 1: The left panel displays the radiative ($\nabla_{\mathrm{rad}}$), adiabatic ($\nabla_{\mathrm{ad}}$), and actual Mixing-Length Theory ($\nabla_T$) logarithmic temperature gradients across enclosed mass $m/M_\star$ for a $4\,M_\odot$, $Z = 0.02$ Zero-Age Main Sequence model ($N_{\mathrm{zones}} = 1094$), highlighting the convective core boundary where $\nabla_{\mathrm{rad}} = \nabla_{\mathrm{ad}}$. The right panel shows the Zero-Age Main Sequence locus across the built-in 8-mass anchor catalog ($M_\star \in [0.51, 15.0]\,M_\odot$ at $Z = 0.02$) in the Hertzsprung-Russell diagram alongside a $4\,M_\odot$ core hydrogen-burning evolutionary trajectory.*

---

## 1. 1D Lagrangian Structure Equations & Staggered Grid

`StarModel` discretizes a spherical star of total mass $M_\star$ into $\mathrm{nz}$ Lagrangian mass shells ordered from the outer photosphere ($k = 0$) to the stellar center ($k = \mathrm{nz} - 1$) with relative shell mass fractions $\Delta q_k = dm_k / M_\star$ ($\sum_{k=0}^{\mathrm{nz}-1} \Delta q_k = 1$). Thermodynamic intensive variables ($\ln\rho_k, \ln T_k$) and species mass fractions $X_{j,k}$ are defined at **cell centers** (enclosed mass $m_{\mathrm{mid},k}$), whereas geometric radius $\ln r_k$, luminosity $L_k$, velocity $u_k$, and convective fluxes are defined at **outer cell faces** (enclosed mass $m_k = M_\star q_k$):

1. **Mass Continuity (`equ_lnR`)**:

   $$\frac{\partial r^3}{\partial m} = \frac{3}{4\pi\rho} \quad \Longrightarrow \quad R_{\ln r, k} = \ln r_k - \frac{1}{3}\ln\!\left(r_{k+1}^3 + \frac{3\,dm_k}{4\pi\rho_k}\right) = 0.$$

2. **Hydrostatic / Hydrodynamic Momentum Balance (`equ_lnd`)**:
   Across interior face $k$ with face-averaged mass $dm_{\mathrm{bar},k} = \frac{1}{2}(dm_{k-1} + dm_k)$, the pressure difference balances gravity (plus hydrodynamic acceleration $du_k/dt$ when active):

   $$\frac{\partial P}{\partial m} = -\frac{G m}{4\pi r^4} - \frac{1}{4\pi r^2}\frac{\partial u}{\partial t}.$$

3. **Radiative + Convective Temperature Transport (`equ_lnT`)**:

   $$\frac{\partial \ln T}{\partial \ln P} = \nabla_{T,k} \quad \Longrightarrow \quad R_{\ln T, k} = \left(\ln T_{k-1} - \ln T_k\right) - \nabla_{T,k}\,\left(\ln P_{k-1} - \ln P_k\right) = 0.$$

4. **Thermal Energy Conservation & Nuclear Burning (`equ_L`, `hydro_energy.rs`)**:

   $$\frac{\partial L}{\partial m} = \epsilon_{\mathrm{nuc}} - \bar{\epsilon}_{\nu,\mathrm{non\text{-}nuc}} + \epsilon_{\mathrm{grav}} + \epsilon_{\mathrm{extra}} + \epsilon_{\mathrm{diff}} + \epsilon_{\mathrm{premix}} + \epsilon_{\mathrm{phase}} + \epsilon_{\mathrm{irr}} + \epsilon_{\mathrm{drag}} + \epsilon_{\mathrm{RTI}} - \frac{\partial e_{\mathrm{turb}}}{\partial t},$$

   where $\bar{\epsilon}_{\nu,\mathrm{non\text{-}nuc}} = \frac{1}{2}(\epsilon_{\nu,\mathrm{start}} + \epsilon_\nu)$ time-centers non-nuclear neutrino losses whenever `non_nuc_neu_start` is populated (`hydro_energy.f90:258`), luminosity gradients use velocity time-centering $\theta_L \in [0.5, 1]$ (`l_theta_for_velocity_time_centering`), mechanical work incorporates time-centered gas pressure $\theta_P \in [0.5, 1]$ (`p_theta_for_velocity_time_centering`), artificial viscosity $P_{\mathrm{vsc}}$ (`pvsc_cq`, `pvsc_zshock`), and MLT turbulent pressure $P_{\mathrm{turb}}$, and $\epsilon_{\mathrm{grav}}$ (`eps_grav.rs`) blends the base thermodynamic $(P, \rho, T)$ form (`eval_eps_grav_std_cell`) with the degenerate Potekhin & Chabrier (2010) entropy $\ln S$ form (`eval_eps_grav_lns_cell`) across $\Gamma \in [\Gamma_{\ln S,\mathrm{off}}, \Gamma_{\ln S,\mathrm{on}}]$ weighted by `eos_frac_pc`, time-centered specific heat and adiabatic gradients (`use_time_centered_eps_grav`), phase-transition latent heat (`latent_ddlnrho`, `latent_ddlnt`), and composition work (`include_composition_in_eps_grav`).

5. **Species Evolution (`equ_chem`, `hydro_eqns.rs`)**:

   $$\frac{\partial X_{j,k}}{\partial t} = \left(\frac{\partial X_{j,k}}{\partial t}\right)_{\mathrm{burn}} + \frac{1}{dm_k}\left(\sigma_{k+1}\,\left[X_{j,k+1} - X_{j,k}\right] - \sigma_k\,\left[X_{j,k} - X_{j,k-1}\right]\right),$$

   where $\sigma_k = f_{\mathrm{mix}} (4\pi r_k^2 \rho_{\{face\},k})^2 D_{\mathrm{mix},k} / dm_{\mathrm{bar},k}$ is the Lagrangian diffusive coupling conductance scaled by `mix_factor`, and nuclear burning is gated by `do_burn` and operator-split burning (`doing_op_split_burn`).

---

## 2. Coupled Henyey Newton-Raphson Solver & 3-Cell Stencil AD

Because the residual vector $\mathbf{F}_k(\mathbf{x}_{k-1}, \mathbf{x}_k, \mathbf{x}_{k+1}) = \mathbf{0}$ of shell $k$ depends only on the primary state variables of shell $k$ and its immediate outer ($k-1$) and inner ($k+1$) neighbors, the global Jacobian $\mathbf{J} = \partial\mathbf{F}/\partial\mathbf{x}$ is **block-tridiagonal** with lower blocks $\mathbf{L}_k$, diagonal blocks $\mathbf{D}_k$, and upper blocks $\mathbf{U}_k$ of dimension $N_{\mathrm{var}} \times N_{\mathrm{var}}$:

$$\mathbf{L}_k\,\delta\tilde{\mathbf{x}}_{k-1} + \mathbf{D}_k\,\delta\tilde{\mathbf{x}}_k + \mathbf{U}_k\,\delta\tilde{\mathbf{x}}_{k+1} = -\mathbf{F}_k, \qquad k \in \{0, \dots, \mathrm{nz}-1\}.$$

- **3-Cell Stencil Automatic Differentiation (`StarAdOrder1` & `jax.jacfwd`)**: In Safe Rust, `eval_equations` populates $(\mathbf{L}_k, \mathbf{D}_k, \mathbf{U}_k)$ analytically using the 3-cell stencil dual number `StarAdOrder1` (`m1`, `00`, `p1` slots). In JAX, `eval_hydro_residuals_jax` and `solve_henyey_linear_jax` assemble and invert the block-tridiagonal system via either $\lceil \log_2 N_{\mathrm{zones}} \rceil$ Parallel Cyclic Reduction (`solver_mode="pcr"`) or Safe Rust `bcyclic` FFI with an analytical Implicit Function Theorem `@jax.custom_vjp` adjoint (`solver_mode="rust_ffi"`).
- **Damped Line Search & Hydrostatic Scale-Height Floor**: `solve_henyey` scales corrections by `x_scale`, applies a backtracking line-search factor $\alpha \in (0, 1]$ whenever trial state variables violate physical bounds or fail to reduce $\|\mathbf{F}\|_2$, and converges to machine-precision residuals within 3 to 6 iterations per evolutionary step.

---

## 3. Model Initialization, Adaptive Remeshing & H211 Timestep Control

### 3.1 Four Initialization Pathways (`StarModel`)
1. **Built-In ZAMS Catalog (`StarModel.from_zams`)**: Loads or interpolates a $Z = 0.02$ Zero-Age Main Sequence model for any stellar mass $M_\star \in [0.51, 15.0]\,M_\odot$ across the 8 built-in anchor models (`0.51, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0 Msun`).
2. **Upstream Fortran MESA `.mod` Files (`StarModel.from_mod_file`, `from_mod_str`)**: Parses and validates MESA ASCII `.mod` files (`lnd`, `lnT`, `lnR`, `L`, `dq`, and species mass fractions `xa`), with full export via `to_mod_str()` and `write_mod_file()`.
3. **Convective Pre-Main-Sequence Polytropes (`StarModel.from_pre_ms`)**: Integrates the $n = 1.5$ convective Lane-Emden polytrope and maps it onto the requested $(M_\star, T_c, Z)$ configuration.
4. **Binary Checkpoints & Custom Arrays (`from_photo_bytes`, `set_primary_state`)**: Restores binary `.photo` restart snapshots or populates primary state arrays directly from NumPy/JAX arrays.

### 3.2 Adaptive Mesh Adjustment (`adjust_mesh`, `eval_mesh_gval_jax`)
Following Paxton et al. (2011, §6.1), `eval_mesh_gval_jax` evaluates the monotonically increasing mesh spacing function $g_k$ combining pressure, temperature, radius, and composition variation weights controlled by `mesh_delta_coeff`. When $\Delta g_k$ exceeds the split threshold, `adjust_mesh` subdivides shell $k$; when adjacent shells fall below the merge threshold, it merges them and conservatively remaps mass fractions and internal energy via `interp_q`.

### 3.3 Smoothed `varcontrol` & Söderlind-Wang H211 Timestep Controller (`pick_next_timestep`)
After each converged step, `eval_varcontrol` computes the 5-point boxcar-smoothed relative state variation norm $v_c = \|\delta\mathbf{x}_{\mathrm{smooth}}\|$ and adjusts the next timestep $\Delta t_{n+1}$ using the Söderlind & Wang (2006) second-order digital filter (`H211b` with $b = 4$, implemented in `eval_h211_timestep_jax`) alongside nuclear fuel depletion limits (`dx_nuc_drop_limit`):

$$\frac{\Delta t_{n+1}}{\Delta t_n} = \left(\frac{v_{\mathrm{target}}}{v_{c,n}}\right)^{1/b} \left(\frac{v_{\mathrm{target}}}{v_{c,n-1}}\right)^{1/b} \left(\frac{\Delta t_n}{\Delta t_{n-1}}\right)^{-1/(2b)}.$$

---

## 4. Well-Balanced Spherical HLLC Riemann Hydrodynamics & `RSP2` Pulsation

### 4.1 1D Spherical Lagrangian HLLC Riemann Solver (`eval_riemann_profile`, `eval_hllc_face_jax`)
For dynamic stellar envelopes, supernova shock propagation, and pulsating variables, `FlexMESA` implements the Toro (2009) and Paxton et al. (2018) 1D spherical Lagrangian Harten-Lax-van Leer-Contact (`HLLC`) Riemann solver across cell interfaces $r_k$:

- **Well-Balanced Hydrostatic Reconstruction**: Interface left and right pressures $(P_L, P_R)$ incorporate the hydrostatic gravitational jump $\Delta P_{\mathrm{grav}} = \frac{1}{2}(dm_k + dm_{k-1}) g_k / (4\pi r_k^2)$ so that a static hydrostatic profile ($u_k = 0$) produces zero contact velocity ($u_{\mathrm{face},k} = 0$) and zero momentum residual to machine precision ($\le 10^{-16}$).
- **Acoustic Wave Speeds & Contact State**: Given left and right sound speeds $(c_L, c_R)$ and Davis wave-speed bounds $(S_L, S_R)$, `eval_hllc_face_jax` evaluates the intermediate contact wave speed $S_M = u_{\mathrm{face}}$ and contact pressure $P^*$ for the hydrodynamic momentum and work equations.

### 4.2 Non-Linear Radial Stellar Pulsation (`RSP` & `RSP2`: `rsp_driver`, `eval_rsp2_zone_jax`, `eval_rsp2_face_jax`)
To model large-amplitude radial pulsators (RR Lyrae, Cepheids, BLAPs, and $\delta$ Scuti variables), `FlexMESA` implements both the Smolec & Moskalik (2008) / Paxton et al. (2019, MESA V) standalone `RSP` subsystem (`STAH` + `RELAX_ENV` static envelope builder, nonsymmetric Schur QR `LINA` linear non-adiabatic eigensolver, and 5-equation banded Newton `HYD` cycle integrator) and the coupled `RSP2` Henyey equations (`hydro_rsp2.f90` and `hydro_rsp2_support.f90`):

- Evolves specific turbulent kinetic energy $e_{t,k} = w_k^2$ with isotropic turbulent pressure $P_{\mathrm{trb},k} = \frac{2}{3}\alpha_p \rho_k w_k^2$, Stellingwerf harmonic radiative luminosity $L_{r,k}$, Kuhfuss convective enthalpy factor $\Pi_k = x_{\alpha_s}\alpha C_{P,k} Y_k$ (with optional quintic $C^2$ enthalpy flux limiting `rsp2_flux_limiter_jax`) and convective luminosity $L_{c,k}$, non-local turbulent diffusion luminosity $L_{t,k}$, buoyant source $S_k$, Kolmogorov cascade sink $D_k = (\alpha_d x_{\mathrm{cede}}/\alpha)(w_k^3 - w_{\min}^3)/H_{p,k}$, radiative eddy-cooling sink $D_{r,k} = 4\sigma_{\mathrm{SB}}(x_{\gamma_r}\gamma_r/\alpha)^2 T_k^3 w_k^2 / (\rho_k^2 C_{P,k}\kappa_k H_{p,k}^2)$, eddy viscosity $\mathrm{X}_k$, viscous shear heating $E_{q,\mathrm{cell},k} = 4\pi \mathrm{X}_k \Delta(u/r)_k / \Delta m_k$, and face viscous acceleration $U_{q,\mathrm{face},k}$.
- `remesh_for_rsp2` maps hydrostatic `StarModel` profiles onto the two-region anchor-temperature `RSP2` grid via Steffen (1990) piecewise-monotonic cubic interpolation followed by quasi-hydrostatic `eosDT` temperature revision, and `solve_stationary_rsp2_w_jax` (`rsp2_adjust_vars_before_call_solver`) solves the local equilibrium balance $S(w) - D(w) - D_r(w) = 0$.

=== "Python & JAX (`flexmesa.star`)"

    ```python
    import jax
    import jax.numpy as jnp
    from flexmesa import star

    # 1. Load the canonical 4.0 Msun, Z = 0.02 ZAMS model and evolve 20 steps
    model = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    reports = model.evolve(
        max_steps=20,
        do_remesh=False,
        op_split_burn=False,
        use_bcyclic=True,
        max_age=1.0e5,
    )
    row = model.history_row()
    print(f"Age = {row.star_age:.2e} yr, log10(L/Lsun) = {row.log_l:.4f}")

    # 2. Differentiate a scalar objective of the Henyey Newton correction via @jax.custom_vjp
    sys = model.eval_equations(include_chem=False)
    grad_equ = jax.grad(
        lambda equ_j: jnp.sum(
            star.solve_henyey_linear_jax(
                jnp.asarray(sys.lblk),
                jnp.asarray(sys.dblk),
                jnp.asarray(sys.ublk),
                equ_j,
                solver_mode="rust_ffi",
            )
            ** 2
        )
    )(jnp.asarray(sys.equ))
    ```

=== "Safe Rust (`crates/flexmesa-star`)"

    ```rust
    use flexmesa_star::{
        StarControls, StarJobControls, extract_history_row, load_zams_model,
        star_evolve,
    };

    fn main() {
        let controls = StarControls {
            initial_mass: 4.0,
            initial_z: 0.02,
            max_age: Some(1.0e5),
            ..StarControls::default()
        };
        let mut model =
            load_zams_model(controls, StarJobControls::default()).unwrap();
        let reports =
            star_evolve(&mut model, 20, false, false, true).unwrap();
        let row = extract_history_row(&model);
        println!("Steps = {}, log10(Tc) = {:.4}", reports.len(), row.log_center_t);
    }
    ```
