# Interacting Binary Evolution, RLOF & Compact Accretors (`flexmesa.binary`)

Most massive and intermediate-mass stars exchange mass or angular momentum with a binary companion during their lifetimes, producing X-ray binaries, stripped helium stars, Type Ia supernova progenitors, and gravitational-wave merger sources. `flexmesa.binary` (`crates/flexmesa-binary` and `src/flexmesa/binary/`) couples the 1D stellar evolution engine (`flexmesa.star`) with Keplerian orbital dynamics, Eggleton (1983) Roche-lobe geometry, implicit Ritter (1988) and Kolb & Ritter (1990) Roche-lobe overflow (`RLOF`) mass-transfer solvers, Bardeen (1972) Kerr black-hole spin evolution, and gravitational-wave, magnetic-braking, and tidal orbital torques.

![Ritter (1988) RLOF mass-transfer curve and Kerr black-hole ISCO spin evolution](../assets/figures/wave5_binary_rlof_evolution.png)

*Figure 1: The left panel displays the Roche-lobe overflow mass-transfer rate $\log_{10}(|\dot{M}_{\mathrm{RLOF}}| / M_\odot\,\mathrm{yr^{-1}})$ as a function of the relative Roche-lobe overfill factor $\delta_{\mathrm{RL},1} = (R_1 - R_{\mathrm{RL},1}) / R_{\mathrm{RL},1}$ for a $2.0\,M_\odot$ donor orbiting a $1.4\,M_\odot$ compact companion, showing the exponential Ritter (1988) optically thin atmospheric regime ($\delta_{\mathrm{RL},1} < 0$) transitioning continuously into the Kolb & Ritter (1990) optically thick adiabatic regime ($\delta_{\mathrm{RL},1} > 0$). The right panel shows the Bardeen (1972) dimensionless Kerr black-hole spin $a_* \in [0, 0.998]$ and innermost stable circular orbit (`ISCO`) radiative efficiency $\eta_{\mathrm{ISCO}}$ as a function of accreted rest mass $M_{\mathrm{acc}} / M_{\mathrm{BH},0}$.*

---

## 1. Keplerian Orbital Mechanics & Eggleton (1983) Roche-Lobe Geometry

For a binary system with donor mass $M_1$, companion mass $M_2$, total mass $M_{\mathrm{tot}} = M_1 + M_2$, orbital separation $a$, and eccentricity $e \in [0, 1)$, the orbital period $P_{\mathrm{orb}}$ and orbital angular momentum $J_{\mathrm{orb}}$ (`eval_j_orb`, `separation_from_period`, `period_from_separation`, `separation_from_j_orb`) satisfy:

$$P_{\mathrm{orb}} = 2\pi \sqrt{\frac{a^3}{G\,(M_1 + M_2)}}, \qquad J_{\mathrm{orb}} = M_1 M_2 \sqrt{\frac{G\,a\,(1 - e^2)}{M_1 + M_2}}.$$

Given the mass ratio $q = M_1 / M_2$, `eval_rlobe` and `eval_rlobe_jax` evaluate the Eggleton (1983, ApJ, 268, 368) volume-equivalent Roche-lobe radius $R_{\mathrm{RL},1}$ and the periastron-adjusted relative Roche-lobe gap $\delta_{\mathrm{RL},1}$ (`eval_rl_relative_gap`):

$$\frac{R_{\mathrm{RL},1}}{a} = \frac{0.49\,q^{2/3}}{0.6\,q^{2/3} + \ln\!\left(1 + q^{1/3}\right)}, \qquad \delta_{\mathrm{RL},1} = \frac{R_1 - R_{\mathrm{RL},1}(1 - e)}{R_{\mathrm{RL},1}(1 - e)}.$$

---

## 2. Ritter (1988), Kolb & Ritter (1990) & Implicit RLOF Mass Transfer

When a donor star expands toward or beyond its Roche lobe around the inner Lagrangian point $L_1$, mass transfer proceeds through two physical regimes:

1. **Optically Thin Atmospheric Overflow (`Ritter 1988`, `eval_ritter_mdot_thin`, `eval_ritter_mdot_thin_jax`)**:
   Even when the photospheric radius remains slightly inside the Roche lobe ($R_1 \le R_{\mathrm{RL},1}$, so $\delta_{\mathrm{RL},1} \le 0$), the extended isothermal atmosphere above the photosphere spills across the $L_1$ saddle point at a rate that decays exponentially with the ratio of the Roche-lobe potential scale height $H_{P,\mathrm{RL}}$ to $R_{\mathrm{RL},1}$:

   $$\dot{M}_{\mathrm{thin}} = -\frac{2\pi}{\sqrt{e_{\mathrm{Eu}}}}\,F_1(q)\,R_{\mathrm{RL},1}^3\,\frac{\rho_{\mathrm{ph}}\,c_{s,\mathrm{ph}}^3}{G M_1}\,\exp\!\left(\frac{R_1 - R_{\mathrm{RL},1}}{H_{P,\mathrm{RL}}}\right), \qquad c_{s,\mathrm{ph}} = \sqrt{\frac{k_{\mathrm{B}} T_{\mathrm{eff}}}{\mu_{\mathrm{ph}} m_u}}.$$

2. **Optically Thick Sub-Photospheric Overflow (`Kolb & Ritter 1990`, `eval_kolb_mdot_thick`)**:
   Once the donor photosphere overfills its Roche lobe ($R_1 > R_{\mathrm{RL},1}$), the total mass-transfer rate combines the photospheric saddle flux with a Simpson-rule integral over the sub-photospheric layers between pressure $P_{\mathrm{ph}}$ and $P_{\mathrm{RL}}$:

   $$\dot{M}_{\mathrm{RLOF}} = \dot{M}_{\mathrm{thin}}(R_1 = R_{\mathrm{RL},1}) - 2\pi\,F_1(q)\,\frac{R_{\mathrm{RL},1}^3}{G M_1}\int_{P_{\mathrm{ph}}}^{P_{\mathrm{RL}}} F_3(\Gamma_1)\,\sqrt{\frac{P}{\rho}}\,dP.$$

3. **Implicit Timestep RLOF Root-Finder (`solve_implicit_rlof_mdot`, `solve_roche_lobe_mdot_jax`)**:
   Because explicit evaluation of $\dot{M}_{\mathrm{RLOF}}(R_1)$ exhibits severe numerical oscillations due to the small atmospheric scale height $H_{P,\mathrm{RL}} / R_{\mathrm{RL},1} \sim 10^{-4}$ to $10^{-2}$, `FlexMESA` solves the implicit consistency equation $f(\dot{M}) = \dot{M} - \dot{M}_{\mathrm{scheme}}(M_1 + \dot{M}\Delta t, a(\dot{M})) = 0$ over each timestep $\Delta t$ using a hybrid Bisection-Brent solver in Safe Rust and a differentiable Newton-bisection solver in JAX.
4. **Outer $L_2$ Lagrangian Disk Mass Loss & Contact Binary Equipotential Solver (`eval_l2_mass_loss_fraction`, `ImplicitRlofSolver::solve_contact_mdot`)**:
   In overcontact binaries where both stars simultaneously approach or fill their Roche lobes (`contact_scheme` in `binary_mdot.f90:113-147` with mass ratio $q = M_a / M_d$ and Marchant et al. 2016 slope $s_c = q^{-0.52}$), `ImplicitRlofSolver::solve_contact_mdot` solves the dimensionless equipotential balance objective $f_{\mathrm{contact}}(\dot{M}) = s_c\,\delta_{\mathrm{RL},d} - \delta_{\mathrm{RL},a} = 0$ (for $q < 1$, or $\delta_{\mathrm{RL},d} - \delta_{\mathrm{RL},a} / s_c = 0$ for $q \ge 1$) over $[-\dot{M}_{\max}, 0]$, while `eval_l2_mass_loss_fraction` computes the fraction of transferred mass lost through a circumbinary $L_2$ outflow alongside its specific angular momentum. When the companion's relative Roche-lobe overflow exceeds the donor's ($\delta_{\mathrm{RL},2} > \delta_{\mathrm{RL},1}$), `BinarySystem::check_donor_switch` swaps donor and accretor indices (`do_Initial_Binary_Checks` in `binary_evolve.f90`).

---

## 3. Kerr Black-Hole ISCO Spin & Eddington-Limited Accretion (`Bardeen 1972`)

When the companion is an accreting black hole (`is_black_hole=True` in `BinarySystem.from_point_mass`), `eval_bh_spin_and_edd_jax` evolves the black hole's gravitational mass $M_{\mathrm{BH}}$ and dimensionless Kerr spin parameter $a_* \in [0, 0.998]$ following Bardeen (1972) and Paxton et al. (2015, §2.5):

- **Innermost Stable Circular Orbit (`ISCO`) Radius & Radiative Efficiency**:
   Given $Z_1(a_*)$ and $Z_2(a_*)$, the dimensionless prograde ISCO radius $r_{\mathrm{ISCO}} = R_{\mathrm{ISCO}} / (G M_{\mathrm{BH}} / c^2) = 3 + Z_2 - \sqrt{(3 - Z_1)(3 + Z_1 + 2 Z_2)}$ determines the specific binding energy and radiative efficiency $\eta_{\mathrm{ISCO}}$:

  $$\eta_{\mathrm{ISCO}}(a_*) = 1 - \sqrt{1 - \frac{2}{3\,r_{\mathrm{ISCO}}(a_*)}}, \qquad \dot{M}_{\mathrm{Edd}} = \frac{4\pi G M_{\mathrm{BH}}}{c\,\kappa_{\mathrm{es}}\,\eta_{\mathrm{ISCO}}(a_*)}.$$

- **Rest-Mass vs. Gravitational-Mass Growth & System Mass-Loss Bookkeeping**:
  A rest-mass accretion step $\Delta M_0 = \min(|\dot{M}_{\mathrm{tr}}|, \dot{M}_{\mathrm{Edd}})\,\Delta t$ increases the black-hole gravitational mass by $\Delta M_{\mathrm{BH}} = (1 - \eta_{\mathrm{ISCO}})\,\Delta M_0$ (`binary_evolve.f90:296-309`), while the radiated accretion energy $\eta_{\mathrm{ISCO}}\,\Delta M_0$ is accounted for in the system mass-loss rate $\dot{M}_{\mathrm{system\_transfer}}(\text{accretor})$ so total gravitating mass-momentum bookkeeping remains exact (`BinarySystem::accretor_mass_and_sys_loss`).

---

## 4. Orbital Angular Momentum Loss ($\dot{J}$) & Hut (1981) Tidal Evolution

`flexmesa.binary` evolves orbital angular momentum $J_{\mathrm{orb}}$, eccentricity $e$, and component spin rates $\Omega_{1,2}$ under four secular torque mechanisms:

1. **Peters (1964) Gravitational-Wave Radiation & Eccentricity Decay (`eval_jdot_gr_jax`, `eval_edot_gr`)**:
   Quadrupole gravitational-wave emission extracts orbital angular momentum (`eval_jdot_gr`) and circularizes eccentric orbits (`eval_edot_gr`, coupled directly into `BinarySystem::step` matching `binary_edot.f90:96`):

   $$\frac{\dot{J}_{\mathrm{gr}}}{J_{\mathrm{orb}}} = -\frac{32}{5}\,\frac{G^3}{c^5}\,\frac{M_1 M_2 (M_1 + M_2)}{a^4\,(1 - e^2)^{5/2}}\left(1 + \frac{7}{8}e^2\right), \qquad \dot{e}_{\mathrm{gr}} = -\frac{304}{15}\,\frac{G^3 M_1 M_2 (M_1 + M_2)}{c^5\,a^4\,(1 - e^2)^{5/2}}\,e\left(1 + \frac{121}{304}e^2\right).$$

2. **Rappaport, Verbunt & Joss (1983) Magnetic Braking (`eval_jdot_mb_jax`)**:
   Magnetized stellar winds coupled to a donor with a convective envelope and radiative core extract spin and orbital angular momentum at rate $\dot{J}_{\mathrm{mb}} = -3.8 \times 10^{-30}\,M_1 R_\odot^4\,(R_1 / R_\odot)^{\gamma_{\mathrm{mb}}}\,\Omega_{\mathrm{orb}}^3\,\mathrm{dyn\,cm}$ (default $\gamma_{\mathrm{mb}} = 3.0$).
3. **Spin-Orbit Tidal Coupling & Missing Stellar Wind Torque (`eval_jdot_ls_with_winds`, `eval_jdot_missing_wind`)**:
   Evaluates spin-orbit angular momentum exchange $\dot{J}_{\mathrm{ls}} = -\sum_{i=1}^2 (\Delta S_i - \dot{M}_{\mathrm{wind},i} j_{\mathrm{surf},i}\Delta t) / \Delta t$ after subtracting stellar wind spin loss (`binary_jdot.f90:129-145`) alongside the companion's missing stellar wind angular momentum torque (`default_jdot_missing_wind`).
4. **Hut (1981) Weak-Friction Tidal Circularization & Synchronization (`eval_edot_tidal_hut_jax`)**:
   Evaluates the Hut (1981, A&A, 99, 126) fifth-order eccentricity polynomials $f_1(e^2)$ through $f_5(e^2)$ to compute the tidal circularization rate $\dot{e}_{\mathrm{tidal}}$ and spin-orbit synchronization timescales for convective and radiative envelopes, paired with monotonically wrapped $[0, 1)$ true-anomaly phase grids (`build_anomaly_grid`) for eccentric phase-dependent RLOF integration.

=== "Python & JAX (`flexmesa.binary`)"

    ```python
    import jax
    import jax.numpy as jnp
    from flexmesa import binary, constants, star

    # 1. Coupled Donor + Kerr Black Hole BinarySystem step
    donor = star.StarModel.from_zams(initial_mass=2.0, initial_z=0.02)
    sys = binary.BinarySystem.from_point_mass(
        donor,
        companion_mass_msun=5.0,
        initial_period_days=1.5,
        eccentricity=0.05,
        is_black_hole=True,
        initial_bh_spin=0.2,
    )
    dt_next, iters = sys.step(dt=1.0e4 * constants.SECYER)

    # 2. Differentiable Ritter (1988) RLOF rate sensitivity d(mdot)/d(R_donor)
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
    ```

=== "Safe Rust (`crates/flexmesa-binary`)"

    ```rust
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
