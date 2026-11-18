# Interacting Binaries & Accretors (`flexmesa.binary`)

!!! warning "Frozen & Unverified JAX Binary Helpers (`#166`)"
    The JAX `binary` modules (`jax_kernels`, `jax_orbit_mdot`,
    `jax_jdot_tides`) are **frozen and unverified**
    (`JAX_ENGINE_STATUS = "frozen-unverified-until-#143"`) until the canonical
    Safe Rust engine passes the upstream MESA `26.4.1` history harness
    (`#143`). `BinarySystem` wraps the Rust driver, and these JAX helpers must
    not be modified before `#143` passes.

The `flexmesa.binary` subpackage ports upstream MESA's `binary/` module in canonical Safe Rust (`crates/flexmesa-binary`) alongside frozen (`float64`) JAX helper kernels. It models detached, semi-detached, and compact-object binary evolution by coupling one or two `StarModel` instances (or a donor star and a point-mass Neutron Star or Kerr Black Hole accretor) to orbital angular momentum and eccentricity evolution:

- **Roche-Lobe Geometry & Mass-Transfer Schemes**: Keplerian separation $a(M_1, M_2, P)$, Eggleton (1983) volume-equivalent Roche-lobe radius $R_{\mathrm{RL},1}/a$ (`eval_rlobe`), periastron overflow gap (`eval_rl_relative_gap`), explicit Ritter (1988) optically thin atmospheric mass transfer (`eval_ritter_mdot_thin`), Kolb & Ritter (1990) optically thick adiabatic overflow integration (`eval_kolb_mdot_thick`), and implicit bisection/secant $\dot{M}_{\mathrm{RLOF}}$ solvers.
- **Compact-Object Accretion & Kerr Spin Evolution**: Eddington-limited mass transfer, radiation pressure wind retention (`alpha`, `beta`, `delta`, `gamma`), and exact Bardeen (1970) / Thorne (1974) Innermost Stable Circular Orbit (ISCO) dimensionless spin $a_* \in [0, 0.998]$ and radiative efficiency $\eta_{\mathrm{ISCO}}$ (`eval_bh_spin_and_edd_jax`).
- **Orbital Angular Momentum & Hut (1981) Tides**: Peters (1964) gravitational-wave radiation (`eval_jdot_gr_jax`), Rappaport, Verbunt & Joss (1983) magnetic braking (`eval_jdot_mb_jax`), L2/circumbinary mass-loss fraction (`eval_l2_mass_loss_fraction`), and Hut (1981) weak-friction tidal eccentricity evolution (`eval_edot_tidal_hut_jax`).
- **Analytical Orbital Oracles, Massive-Star Winds, Checkpoints, Counters, `BinaryExtras` Hooks, `binary_history.data` Writers & `binary/test_suite` Runner**: Peters (1964) GW inspiral separation (`eval_gr_expected_separation_rsun` for `jdot_gr_check`), Tauris & van den Heuvel (2003) non-conservative mass-transfer separation (`eval_tauris_vdh_expected_separation_rsun` for `jdot_ml_check`), Brott et al. (2011) massive-star winds (`eval_brott_wind_msun_yr` for `double_bh`), full coupled binary checkpoints (`FMBINCP1` v1 via `BinarySystem.to_photo_bytes` / `from_photo_bytes` in Python and `to_checkpoint_bytes` / `from_checkpoint_bytes` / `save_checkpoint` / `load_checkpoint` in Rust) preserving orbital state, controls, and retry-proof `BinaryStepCounters` (`num_retries`, `num_implicit_trials`, `num_dt_cuts`, `num_hard_limit_retries`), custom `BinaryExtras` (`other_wind_cgs`, `finish_step`, `BinaryExtrasAction`, `DefaultBinaryExtras`) via `step_with_extras` / `try_step_with_extras` in `flexmesa-binary`, incremental MESA `26.4.1` `binary_history.data` writers (`BinaryLogsWriter`, `IncrementalBinaryHistoryWriter`, `run_binary_test_suite_case_with_logs`, with `append_to_star_history` forwarding to `LOGS1/` and `LOGS2/`; the `edot`, `edot_tidal` and `edot_enhance` columns report the terms the orbital update applied, stored in `BinarySystem::last_edot` as an `EdotBreakdown` from `eval_coupled_edot_breakdown`), and the `binary/test_suite` runner (`TIER5_BINARY_CASES`, `run_binary_test_suite_case`, `run_binary_suite`; coupled binary evolution is currently unverified against real MESA output, pending [#143](https://github.com/joshspeagle/FlexMESA/issues/143) and [#160](https://github.com/joshspeagle/FlexMESA/issues/160)).

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
