# Interacting Binaries & Accretors (`flexmesa.binary`)

The `flexmesa.binary` subpackage ports upstream MESA's `binary/` module across Safe Rust (`crates/flexmesa-binary`) and pure JAX (`float64`). It models detached, semi-detached, and compact-object binary evolution by coupling one or two `StarModel` instances (or a donor star and a point-mass Neutron Star or Kerr Black Hole accretor) to orbital angular momentum and eccentricity evolution:

- **Roche-Lobe Geometry & Mass-Transfer Schemes**: Keplerian separation $a(M_1, M_2, P)$, Eggleton (1983) volume-equivalent Roche-lobe radius $R_{\mathrm{RL},1}/a$ (`eval_rlobe`), periastron overflow gap (`eval_rl_relative_gap`), explicit Ritter (1988) optically thin atmospheric mass transfer (`eval_ritter_mdot_thin`), Kolb & Ritter (1990) optically thick adiabatic overflow integration (`eval_kolb_mdot_thick`), and implicit bisection/secant $\dot{M}_{\mathrm{RLOF}}$ solvers.
- **Compact-Object Accretion & Kerr Spin Evolution**: Eddington-limited mass transfer, radiation pressure wind retention (`alpha`, `beta`, `delta`, `gamma`), and exact Bardeen (1970) / Thorne (1974) Innermost Stable Circular Orbit (ISCO) dimensionless spin $a_* \in [0, 0.998]$ and radiative efficiency $\eta_{\mathrm{ISCO}}$ (`eval_bh_spin_and_edd_jax`).
- **Orbital Angular Momentum & Hut (1981) Tides**: Peters (1964) gravitational-wave radiation (`eval_jdot_gr_jax`), Rappaport, Verbunt & Joss (1983) magnetic braking (`eval_jdot_mb_jax`), L2/circumbinary mass-loss fraction (`eval_l2_mass_loss_fraction`), and Hut (1981) weak-friction tidal eccentricity evolution (`eval_edot_tidal_hut_jax`).

For governing equations and physical derivations, see [Interacting Binaries & Accretors](../concepts/binaries.md) and [Tutorial 4: Interacting Binaries & Accretors](../tutorials/04-interacting-binaries.md).

---

## Quick Usage Example

```python
import flexmesa.binary as binary
import flexmesa.star as star
from flexmesa import constants

# Initialize a 4.0 Msun ZAMS donor + 1.4 Msun NS X-ray binary in a 1.5-day orbit
donor = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
sys_bin = binary.BinarySystem.from_point_mass(
    donor,
    companion_mass_msun=1.4,
    initial_period_days=1.5,
    eccentricity=0.05,
)
print(
    f"a={sys_bin.separation / constants.RSUN:.3f} Rsun, "
    f"R_RL1={sys_bin.rl[0] / constants.RSUN:.3f} Rsun"
)
```

---

## Package Overview (`flexmesa.binary`)

::: flexmesa.binary

---

## Coupled Binary System Driver (`flexmesa.binary.system`)

::: flexmesa.binary.system

---

## Roche-Lobe & Mass-Transfer Wrappers (`flexmesa.binary.orbit_mdot`)

::: flexmesa.binary.orbit_mdot

---

## Differentiable Roche-Lobe & Kerr BH Kernels (`flexmesa.binary.jax_orbit_mdot`)

::: flexmesa.binary.jax_orbit_mdot

---

## Differentiable $\dot{J}$ & Hut Tidal Kernels (`flexmesa.binary.jax_jdot_tides`)

::: flexmesa.binary.jax_jdot_tides
