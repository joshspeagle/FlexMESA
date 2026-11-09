# Tutorial 4: Interacting Binaries, Mass Transfer & Compact Accretors

Most massive stars and the progenitors of Type Ia supernovae, X-ray binaries, and gravitational-wave mergers evolve in close binary systems where tidal forces distort stellar envelopes and drive mass transfer across the inner Lagrangian point $L_1$. This tutorial walks through simulating detached and semi-detached binaries in `flexmesa.binary`, evaluating Eggleton (1983) Roche-lobe geometry, comparing optically thin (`Ritter`) and optically thick (`Kolb`) Roche-lobe overflow (`RLOF`) mass-transfer rates, tracking Kerr black-hole spin-up and Eddington-limited accretion, computing orbital angular momentum losses ($\dot{J}_{\mathrm{gr}}, \dot{J}_{\mathrm{mb}}, \dot{J}_{\mathrm{ml}}, \dot{J}_{\mathrm{ls}}$), and modeling Hut (1981) tidal circularization and synchronization.

---

## 1. Eggleton Roche-Lobe Geometry, `Ritter`, and `Kolb` Mass Transfer

In a binary with donor mass $M_1$, accretor mass $M_2$, mass ratio $q = M_1 / M_2$, and orbital separation $a$, the volume-equivalent Roche-lobe radius $R_{\mathrm{RL},1}$ is given by the Eggleton (1983) rational approximation (`binary.eval_rlobe`), and the dimensionless overflow gap is $\Delta_{\mathrm{RL}} = (R_1 - R_{\mathrm{RL},1}) / R_{\mathrm{RL},1}$ (`binary.eval_rl_relative_gap`, including periastron scaling $(1 - e)$ for eccentric orbits).

When the donor photosphere approaches or overflows $R_{\mathrm{RL},1}$:

- **Optically Thin Atmosphere (`Ritter 1988`, `binary.eval_ritter_mdot_thin`)**: Gas flows isothermally across the $L_1$ saddle point at the local sound speed, yielding an exponential sensitivity $\dot{M}_{\mathrm{Ritter}} \propto -\exp[(R_1 - R_{\mathrm{RL},1}) / H_{P,\mathrm{RL}}]$.
- **Optically Thick Envelope (`Kolb & Ritter 1990`, `binary.eval_kolb_mdot_thick`)**: When $R_1 > R_{\mathrm{RL},1}$, `FlexMESA` adds the optically thick adiabatic pressure-integral over all interior donor shells with $r(m) \in [R_{\mathrm{RL},1}, R_1]$.

```python
import jax
import jax.numpy as jnp
from flexmesa import binary, constants, star

donor = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
m1_g = 4.0 * constants.MSUN
m2_g = 1.4 * constants.MSUN
period_s = 1.2 * 86400.0

# 1. Compute orbital separation, Roche-lobe radius, and relative overflow gap
a_cm = binary.separation_from_period(m1_g, m2_g, period_s)
rl1_cm = binary.eval_rlobe(m1_g, m2_g, a_cm)
r1_cm = float(donor.r[0])
gap = binary.eval_rl_relative_gap(r1_cm, rl1_cm, eccentricity=0.0)

# 2. Evaluate optically thin Ritter mass transfer and its exact JAX derivative d(mdot)/dR_1
rho_s, p_s, t_s, mu_s = (
    float(donor.rho[0]),
    float(donor.peos[0]),
    float(donor.t[0]),
    float(donor.mu[0]),
)
mdot_ritter = binary.eval_ritter_mdot_thin(
    m1_g, m2_g, r1_cm, rl1_cm, rho_s, p_s, t_s, mu_s
)
dmdot_dr1 = jax.grad(
    lambda r_in: binary.eval_ritter_mdot_thin_jax(
        m1_g, m2_g, r_in, rl1_cm, rho_s, p_s, t_s, mu_s
    )
)(jnp.float64(rl1_cm))

# 3. Evaluate optically thick Kolb-Ritter mass transfer when the donor overflows R_RL by 3%
rl_overflow_cm = 0.97 * r1_cm
mdot_kolb = binary.eval_kolb_mdot_thick(
    m1_g,
    m2_g,
    rl_overflow_cm,
    donor.r,
    donor.peos,
    donor.t,
    donor.mu,
    donor.gamma1,
)
print(
    f"P = 1.2 d: a = {a_cm / constants.RSUN:.3f} Rsun, R_RL,1 = {rl1_cm / constants.RSUN:.3f} Rsun, "
    f"R_1 = {r1_cm / constants.RSUN:.3f} Rsun, gap = {gap:.4f}"
)
print(
    f"Ritter mdot = {mdot_ritter / constants.MSUN * 3.15576e7:.4e} Msun/yr, "
    f"Kolb thick mdot (3% overflow) = {mdot_kolb / constants.MSUN * 3.15576e7:.4e} Msun/yr, "
    f"d(mdot_Ritter)/dR_1 at R_RL = {float(dmdot_dr1):.4e} g/s/cm"
)
```

![Binary Roche-lobe overflow and orbital evolution](../assets/figures/wave5_binary_rlof_evolution.png)

*Figure 1: Coupled donor and compact-accretor binary evolution across the onset of Roche-lobe overflow (`RLOF`), showing the exponential transition from detached evolution ($\Delta_{\mathrm{RL}} < 0$) into self-regulated mass transfer ($\Delta_{\mathrm{RL}} \approx 0$) and the corresponding orbital period evolution.*

---

## 2. Evolving `PointMass` and `TwoStars` Binary Systems

The `binary.BinarySystem` class couples either a live 1D `StarModel` donor and a compact point-mass companion (`BinarySystem.from_point_mass`, supporting white dwarfs, neutron stars, and Kerr black holes) or two simultaneously evolving 1D `StarModel` instances (`BinarySystem.from_two_stars`):

```python
import matplotlib.pyplot as plt
import numpy as np
from flexmesa import binary, constants, plotting, star

# 1. Donor + Point-Mass Black Hole Binary
donor = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
sys_bh = binary.BinarySystem.from_point_mass(
    donor=donor,
    companion_mass_msun=5.0,
    initial_period_days=1.5,
    eccentricity=0.05,
    is_black_hole=True,
    initial_bh_spin=0.2,
)

ages_myr, periods_d, lg_mdot, eccs = [], [], [], []
for _ in range(5):
    mdot_step, iters = sys_bh.step(dt=2.0e4 * 3.15576e7)
    ages_myr.append(sys_bh.binary_age_yr * 1.0e-6)
    periods_d.append(sys_bh.period / 86400.0)
    mdot_msun_yr = abs(mdot_step) / constants.MSUN * 3.15576e7
    lg_mdot.append(np.log10(max(mdot_msun_yr, 1.0e-99)))
    eccs.append(sys_bh.eccentricity)

fig, axes = plotting.plot_binary_evolution(
    np.array(ages_myr),
    np.array(periods_d),
    np.array(lg_mdot),
    eccentricity=np.array(eccs),
)
plt.close(fig)

# 2. Donor + Accretor Two-Star Binary
donor_2s = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
accretor_2s = star.StarModel.from_zams(initial_mass=2.0, initial_z=0.02)
sys_2s = binary.BinarySystem.from_two_stars(
    donor=donor_2s,
    accretor=accretor_2s,
    initial_period_days=2.5,
    eccentricity=0.0,
)
mdot_2s, _ = sys_2s.step(dt=1.0e4 * 3.15576e7)
print(
    f"Two-star binary step: P = {sys_2s.period / 86400.0:.5f} d, "
    f"R_RL = ({sys_2s.rl[0] / constants.RSUN:.3f}, {sys_2s.rl[1] / constants.RSUN:.3f}) Rsun"
)
```

---

## 3. Kerr Black-Hole Spin Evolution, Angular Momentum Losses & Hut Tides

Beyond Roche-lobe mass transfer, binary orbital evolution is governed by the conservation and loss of orbital angular momentum $J_{\mathrm{orb}}$:

- **Bardeen (1970) & Thorne (1974) Kerr Black-Hole ISCO Accretion (`binary.eval_bh_spin_and_edd_jax`)**: Returns the 4-tuple `(m_bh_zero_eq, a_spin_current, mdot_edd, eta)` containing the effective zero-spin reference mass $M_{\mathrm{BH},0}$, updated dimensionless Kerr spin $a_\star \in [0, 0.998]$, Eddington mass-accretion rate $\dot{M}_{\mathrm{Edd}} = 4\pi G M_{\mathrm{BH}} / (\kappa c \eta_{\mathrm{rad}})$, and ISCO radiative efficiency $\eta_{\mathrm{rad}} = 1 - \sqrt{1 - 2/(3 r_{\mathrm{ISCO}})}$.
- **Orbital Angular Momentum Torques ($\dot{J}_{\mathrm{orb}}$)**:
  - **Gravitational Waves (`eval_jdot_gr_jax`)**: Peters (1964) quadrupole radiation torque $\dot{J}_{\mathrm{gr}} = -\frac{32}{5}\frac{G^3 M_1 M_2 (M_1 + M_2)}{c^5 a^4}\,J_{\mathrm{orb}}$.
  - **Magnetic Braking (`eval_jdot_mb_jax`)**: Rappaport, Verbunt & Joss (1983) magnetized wind torque on stars with convective envelopes.
  - **Mass Loss & L2 Outflow (`_rust.binary_eval_jdot_ml`, `binary.eval_l2_mass_loss_fraction`)**: Specific angular momentum carried away by isotropic stellar winds, non-conservative accretion, and Lu et al. (2023) outer-Lagrangian $L_2$ outflow.
  - **Spin-Orbit Coupling (`_rust.binary_eval_jdot_ls`)**: Conservation of total spin plus orbital angular momentum.
- **Hut (1981) Weak-Friction Tidal Evolution (`_rust.binary_eval_tides_hut`, `binary.eval_edot_tidal_hut_jax`)**: High-eccentricity polynomial expansions $F_1(e)$ through $F_5(e)$ for tidal circularization ($\dot{e}_{\mathrm{tid}}$) and synchronization ($\dot{\Omega}_{\mathrm{tid}}$) via Zahn (1977) convective or radiative damping.

```python
import jax
import jax.numpy as jnp
from flexmesa import _rust, binary, constants

# 1. Kerr Black-Hole ISCO spin-up and Eddington accretion rate
m_bh_init = 5.0 * constants.MSUN
m_bh_curr = 5.5 * constants.MSUN
m_eq, a_spin, mdot_edd, eta_rad = binary.eval_bh_spin_and_edd_jax(
    m_bh_init=jnp.float64(m_bh_init),
    a_spin_init=jnp.float64(0.3),
    m_bh_current=jnp.float64(m_bh_curr),
    kappa_surf=jnp.float64(0.34),
)
print(
    f"Kerr BH (5.0 -> 5.5 Msun): M_eq = {float(m_eq) / constants.MSUN:.4f} Msun, "
    f"a* = {float(a_spin):.5f}, eta_rad = {float(eta_rad):.5f}, "
    f"Mdot_Edd = {float(mdot_edd) / constants.MSUN * 3.15576e7:.3e} Msun/yr"
)

# 2. Orbital angular momentum losses: GR, Magnetic Braking, Mass Loss, and Spin-Orbit
m1 = 1.2 * constants.MSUN
m2 = 0.8 * constants.MSUN
sep = 3.5 * constants.RSUN
per = binary.period_from_separation(m1, m2, sep)
j_orb = binary.eval_j_orb(m1, m2, sep, eccentricity=0.05)
rl1 = binary.eval_rlobe(m1, m2, sep)

jdot_gr = binary.eval_jdot_gr_jax(m1, m2, sep, j_orb)
jdot_mb = binary.eval_jdot_mb_jax(
    m_star=m1,
    r_star=1.1 * constants.RSUN,
    rl_star=rl1,
    period=per,
    gamma_mb=3.0,
    q_conv_env=0.08,
)
jdot_ml = _rust.binary_eval_jdot_ml(
    m1,
    m2,
    sep,
    float(per),
    0.05,
    -1.0e-9 * constants.MSUN / 3.15576e7,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)
jdot_ls = _rust.binary_eval_jdot_ls(1.0e50, 1.001e50, 0.0, 0.0, 1.0e10)

print(
    f"Torques (g cm^2 s^-2): jdot_GR = {float(jdot_gr):.3e}, jdot_MB = {float(jdot_mb):.3e}, "
    f"jdot_ML = {jdot_ml:.3e}, jdot_LS = {jdot_ls:.3e}"
)

# 3. Hut (1981) tidal synchronization and circularization
f2, f3, f4, f5, k_div_t, t_sync_s, edot_hut = _rust.binary_eval_tides_hut(
    m1,
    m2,
    1.1 * constants.RSUN,
    sep,
    0.10,
    float(per),
    1.5e-5,
    0.15,
    0.4,
    1.5,
    0.1,
    1.0,
    True,
)
dedot_de = jax.grad(
    lambda e_in: binary.eval_edot_tidal_hut_jax(
        k_div_t, m1, m2, 1.1 * constants.RSUN, sep, e_in, 1.5e-5, per
    )
)(jnp.float64(0.10))
print(
    f"Hut tides: t_sync = {t_sync_s / 3.15576e7:.2e} yr, de/dt = {edot_hut:.4e} s^-1, "
    f"d(de/dt)/de = {float(dedot_de):.4e} s^-1"
)
```

---

## 4. Analytical Orbital Oracles, Massive-Star Winds & Custom `other_*` Hooks

Stage `E2E-5` provides analytical physics oracles and multi-stage runners for all 8 upstream `binary/test_suite` cases (`double_bh`, `evolve_both_stars`, `jdot_gr_check`, `jdot_ls_check`, `jdot_ml_check`, `star_plus_point_mass`, `star_plus_point_mass_explicit_mdot`, `wind_fed_bhhmxb`) and the 4 Tier 5/6 `star/test_suite` hook cases (`check_redo`, `other_physics_hooks`, `relax_composition_j_entropy`, `timing`):

- **Peters (1964) GW Inspiral & Tauris & van den Heuvel (2003) Non-Conservative Oracles (`binary.eval_gr_expected_separation_rsun`, `binary.eval_tauris_vdh_expected_separation_rsun`)**: Closed-form orbital separation trajectories $a(t)$ (`jdot_gr_check`) and $a(q; \alpha, \beta, \delta, \gamma)$ (`jdot_ml_check`).
- **Brott et al. (2011) Bi-Stability Massive-Star Winds (`binary.eval_brott_wind_msun_yr`, `binary/test_suite/double_bh`)**: Combines Vink et al. (2001), Nieuwenhuijzen & de Jager (1990), and Hamann et al. (1995) Wolf-Rayet winds across the bi-stability jump for $70\,M_\odot + 65\,M_\odot$ binary black-hole progenitors (`double_bh`).
- **Custom `other_*` Physics Hooks & Step-Redo Parity (`star.eval_overshoot_dbl_exp`, `star.eval_grafener_kudritzki_wind_msun_yr`, `star.verify_step_redo_bit_parity`)**: Evaluates double-exponential convective overshooting $D_{\mathrm{ov}}(r)$, Gräfener & Kudritzki (2008) Wolf-Rayet optically thick wind mass-loss rates, and `0 ULP` `.photo` step-redo parity (`check_redo`).

```python
from flexmesa import binary, constants, star

# 1. Peters (1964) GW inspiral and Tauris & van den Heuvel (2003) non-conservative separation
a_gr = binary.eval_gr_expected_separation_rsun(
    a0_rsun=2.5, m1_msun=1.4, m2_msun=1.0, time_sec=1.0e8 * 3.15576e7
)
a_ml = binary.eval_tauris_vdh_expected_separation_rsun(
    a0_rsun=10.0, q0=2.0, q=1.2, alpha=0.2, beta=0.3, delta=0.1, gamma=1.5
)
w_brott = binary.eval_brott_wind_msun_yr(
    l_surf_lsun=6.0e5,
    m_surf_msun=65.0,
    r_surf_rsun=12.0,
    t_surf_k=48000.0,
    x_surf=0.35,
    y_surf=0.63,
    z_div_z_solar=0.1,
)
print(
    f"Oracles: a_GR = {a_gr:.4f} Rsun, a_TVdH = {a_ml:.4f} Rsun, "
    f"Mdot_Brott = {w_brott:.3e} Msun/yr"
)

# 2. Double-exponential overshoot, Grafener-Kudritzki WR wind, and step-redo bit parity
d_ov = star.eval_overshoot_dbl_exp(
    d0=1.0e14,
    delta_r=1.0e9,
    hp_bdy=5.0e9,
    f_ov1=0.015,
    f_ov2=0.005,
    d_ov_trans=1.0e12,
)
mdot_gk = star.eval_grafener_kudritzki_wind_msun_yr(
    m_surf_g=60.0 * constants.MSUN,
    l_surf_erg_s=1.0e6 * constants.LSUN,
    r_surf_cm=30.0 * constants.RSUN,
    t_surf_k=40000.0,
    x_surf=0.70,
    z_surf=0.02,
    w_scale=1.0,
)
zams_1m = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
redo_ok = star.verify_step_redo_bit_parity(zams_1m)
print(
    f"Other hooks: D_ov = {d_ov:.3e} cm^2/s, Mdot_GK = {mdot_gk:.3e} Msun/yr, "
    f"redo parity = {redo_ok}"
)
```
