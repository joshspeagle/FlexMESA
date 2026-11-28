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
- **Checkpoints, Counters, `BinaryExtras` Hooks, `binary_history.data` Writers & `binary/test_suite` Runner**: Full coupled binary checkpoints (`FMBINCP1` v2 via `BinarySystem.to_photo_bytes` / `from_photo_bytes` in Python and `to_checkpoint_bytes` / `from_checkpoint_bytes` / `save_checkpoint` / `load_checkpoint` in Rust) preserving orbital state, controls, and retry-proof `BinaryStepCounters` (`num_retries`, `num_implicit_trials`, `num_dt_cuts`, `num_hard_limit_retries`), custom `BinaryExtras` (`other_wind_cgs`, `check_model`, `finish_step`, `extra_binary_history_columns`, `extra_star_history_columns`, `BinaryExtrasAction`, `DefaultBinaryExtras`) via `step_with_extras` / `try_step_with_extras` in `flexmesa-binary`, incremental MESA `26.4.1` `binary_history.data` writers (`BinaryLogsWriter`, `IncrementalBinaryHistoryWriter`, `run_binary_test_suite_case_with_logs`, with `append_to_star_history` forwarding to `LOGS1/` and `LOGS2/`; the `edot`, `edot_tidal` and `edot_enhance` columns report the terms the orbital update applied, stored in `BinarySystem::last_edot` as an `EdotBreakdown` from `eval_coupled_edot_breakdown`), and the `binary/test_suite` runner (`run_binary_test_suite_case`, accepting optional caller-supplied test-side `extras` callbacks; coupled binary evolution is currently unverified against real MESA output, pending [#143](https://github.com/joshspeagle/FlexMESA/issues/143) and [#160](https://github.com/joshspeagle/FlexMESA/issues/160)).

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

---

## Compiled Rust Extension Bridge (`flexmesa._rust`)

Low-level Safe Rust PyO3 bindings exposed via `flexmesa._rust` for interacting binaries (`_ArrF64` denotes a `float64` NumPy `ndarray`).

### `binary_py` and `extras_py`

- **`RustBinarySystem`** (`class RustBinarySystem`; static constructors `new_point_mass`, `new_two_stars`, `from_photo_bytes`; methods `to_photo_bytes`, `step`): Python wrapper around `flexmesa_binary::BinarySystem` exposing donor/accretor states, orbital separation, period, eccentricity, $J_{\mathrm{orb}}$, Roche radii, relative gaps, $\dot{M}$, and $\dot{J}$.
- **`binary_build_anomaly_grid`** (`(anomaly_steps: int, eccentricity: float) -> tuple[_ArrF64, _ArrF64]`): Builds the eccentric true-anomaly and normalized time grid `(theta_co, time_co)`.
- **`binary_eval_accreted_material_j`** (`(m_donor: float, m_accretor: float, r_accretor: float, separation: float) -> tuple[float, int]`): Evaluates de Mink et al. (2013) accreted specific angular momentum `(j_acc, mode)`.
- **`binary_eval_arras_mdot_thin`** (`(m_donor: float, m_accretor: float, separation: float, r_donor: float, rho_surf: float, t_surf: float, mu_surf: float) -> float`): Evaluates the Arras optically thin RLOF mass transfer rate ($\mathrm{g\,s^{-1}}$).
- **`binary_eval_bh_spin_and_edd`** (`(m_bh_init: float, a_spin_init: float, m_bh_current: float, kappa_surf: float) -> tuple[float, float, float, float]`): Computes Kerr BH equivalent initial non-rotating mass, current spin, and Eddington accretion quantities from `(m_bh_init, a_spin_init, m_bh_current, kappa_surf)`.
- **`binary_eval_j_orb`** (`(m1: float, m2: float, separation: float, eccentricity: float) -> float`): Computes orbital angular momentum $J_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-1}}$).
- **`binary_eval_jdot_gr`** (`(m_donor: float, m_accretor: float, separation: float, j_orb: float) -> float`): Evaluates gravitational wave $\dot{J}_{\mathrm{gr}}$ ($\mathrm{g\,cm^2\,s^{-2}}$).
- **`binary_eval_jdot_ls`** (`(s_donor_old: float, s_donor: float, s_accretor_old: float, s_accretor: float, dt: float) -> float`): Evaluates spin-orbit tidal torque $\dot{J}_{\mathrm{ls}}$ ($\mathrm{g\,cm^2\,s^{-2}}$).
- **`binary_eval_jdot_mb`** (`(m_star: float, r_star: float, rl_star: float, period: float, gamma_mb: float, q_conv_env: float, scale_for_low_qconv_env: bool) -> float`): Evaluates Rappaport-Verbunt-Joss (1983) magnetic braking $\dot{J}_{\mathrm{mb}}$.
- **`binary_eval_jdot_ml`** (`(m_donor: float, m_accretor: float, separation: float, period: float, eccentricity: float, mdot_sys_tr_donor: float, mdot_sys_wind_donor: float, mdot_sys_tr_accretor: float, mdot_sys_wind_accretor: float, mdot_sys_cct: float, gamma_mt: float) -> float`): Evaluates system mass-loss $\dot{J}_{\mathrm{ml}}$ ($\mathrm{g\,cm^2\,s^{-2}}$).
- **`binary_eval_kolb_mdot_thick`** (`(m_donor: float, m_accretor: float, rl_donor: float, r_profile: _ArrF64, p_profile: _ArrF64, t_profile: _ArrF64, mu_profile: _ArrF64, gamma1_profile: _ArrF64) -> float`): Evaluates the Kolb & Ritter (1990) optically thick RLOF mass transfer rate ($\mathrm{g\,s^{-1}}$).
- **`binary_eval_l2_mass_loss_fraction`** (`(donor_mass_msun: float, accretor_mass_msun: float, mass_transfer_rate_msun_yr: float, orbital_separation_rsun: float, disk_alpha: float, disk_mu: float) -> float`): Evaluates the Lu et al. (2023) thin-disk outer $L_2$ mass-loss fraction $f_{L2}$.
- **`binary_eval_l2_overflow_limit`** (`(m1: float, m2: float) -> float`): Evaluates the Marchant et al. (2016) contact binary $L_2$ overflow limit.
- **`binary_eval_ritter_mdot_thin`** (`(m_donor: float, m_accretor: float, r_donor: float, rl_donor: float, rho_surf: float, p_surf: float, t_surf: float, mu_surf: float, is_kolb_scheme: bool) -> float`): Evaluates the Ritter (1988) optically thin RLOF mass transfer rate ($\mathrm{g\,s^{-1}}$).
- **`binary_eval_rl_relative_gap`** (`(r: float, rl: float, eccentricity: float) -> float`): Evaluates periastron relative Roche-lobe gap $(R - R_L(1-e)) / (R_L(1-e))$.
- **`binary_eval_rlobe`** (`(m1: float, m2: float, a: float) -> float`): Evaluates Eggleton (1983) Roche-lobe radius $R_{L,1}(m_1, m_2, a)$ ($\mathrm{cm}$).
- **`binary_eval_tides_hut`** (`(m_star: float, m_companion: float, r_phot: float, separation: float, eccentricity: float, period: float, omega_star: float, m_env_msun: float, r_env_rsun: float, l_phot_lsun: float, r_gyr_sq: float, f_tid: float, has_convective_envelope: bool) -> tuple[float, float, float, float, float, float, float]`): Evaluates Hut (1981) eccentricity polynomials and tidal rates.
- **`binary_eval_wind_transfer`** (`(m_donor: float, m_accretor: float, r_donor: float, rl_donor: float, separation: float, eccentricity: float, b_wind: float, alpha_bh: float, beta_bh: float, max_frac: float) -> tuple[float, float]`): Evaluates Tout & Eggleton (1988) wind enhancement and Hurley et al. (2002) Bondi-Hoyle fraction.
- **`binary_period_from_separation`** (`(m1: float, m2: float, separation: float) -> float`): Computes orbital period $P$ ($\mathrm{s}$) from separation $a$ ($\mathrm{cm}$) and masses $m_1, m_2$ ($\mathrm{g}$).
- **`binary_pick_next_timestep_rl`** (`(donor_time_step_yr: float, max_timestep_sec: float, rl_gap_old_donor: float, rl_gap_donor: float, fr: float = 0.01, fr_limit: float = 0.01, fr_dt_limit: float = 0.0, dt_softening_factor: float = 0.0, time_delta_coeff: float = 1.0) -> tuple[float, int]`): Evaluates `binary_pick_next_timestep` matching upstream `binary/test/src/test_binary.f90`.
- **`binary_separation_from_j_orb`** (`(m1: float, m2: float, j_orb: float, eccentricity: float) -> float`): Computes orbital separation $a$ ($\mathrm{cm}$) from $J_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-1}}$).
- **`binary_separation_from_period`** (`(m1: float, m2: float, period: float) -> float`): Computes orbital separation $a$ ($\mathrm{cm}$) from period $P$ ($\mathrm{s}$) and masses $m_1, m_2$ ($\mathrm{g}$).
- **`binary_extras_snapshot`** (`(bin: RustBinarySystem, case_dir: str = "") -> dict[str, Any]`): Builds the MESA `binary_info` extras snapshot dictionary for a `RustBinarySystem` without running a full trace (`#148`, `#231`).
