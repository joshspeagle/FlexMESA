# SPDX-License-Identifier: LGPL-3.0-or-later
"""Binary orbital mechanics, Roche-lobe overflow, $L_2$ disk mass loss,
angular momentum loss, tidal evolution, and coupled binary star evolution
(`flexmesa.binary`).

Corresponds to the upstream Fortran MESA `binary/` module.
"""

from __future__ import annotations

from flexmesa.binary import jax_kernels, orbit_mdot, system
from flexmesa.star import e2e_tier45

BinarySystem = system.BinarySystem
TIER5_BINARY_CASES = e2e_tier45.TIER5_BINARY_CASES

eval_brott_wind_msun_yr = e2e_tier45.eval_brott_wind_msun_yr
eval_gr_expected_separation_rsun = e2e_tier45.eval_gr_expected_separation_rsun
eval_tauris_vdh_expected_separation_rsun = (
    e2e_tier45.eval_tauris_vdh_expected_separation_rsun
)
run_binary_suite = e2e_tier45.run_binary_suite
run_binary_test_suite_case = e2e_tier45.run_binary_test_suite_case

eval_rlobe = orbit_mdot.eval_rlobe
eval_rl_relative_gap = orbit_mdot.eval_rl_relative_gap
separation_from_period = orbit_mdot.separation_from_period
period_from_separation = orbit_mdot.period_from_separation
eval_j_orb = orbit_mdot.eval_j_orb
separation_from_j_orb = orbit_mdot.separation_from_j_orb
eval_ritter_mdot_thin = orbit_mdot.eval_ritter_mdot_thin
eval_kolb_mdot_thick = orbit_mdot.eval_kolb_mdot_thick
eval_l2_mass_loss_fraction = orbit_mdot.eval_l2_mass_loss_fraction
pick_next_timestep_rl = orbit_mdot.pick_next_timestep_rl

eval_rlobe_jax = jax_kernels.eval_rlobe_jax
eval_rl_relative_gap_jax = jax_kernels.eval_rl_relative_gap_jax
separation_from_period_jax = jax_kernels.separation_from_period_jax
period_from_separation_jax = jax_kernels.period_from_separation_jax
eval_j_orb_jax = jax_kernels.eval_j_orb_jax
separation_from_j_orb_jax = jax_kernels.separation_from_j_orb_jax
eval_ritter_mdot_thin_jax = jax_kernels.eval_ritter_mdot_thin_jax
solve_roche_lobe_mdot_jax = jax_kernels.solve_roche_lobe_mdot_jax
eval_bh_spin_and_edd_jax = jax_kernels.eval_bh_spin_and_edd_jax
eval_jdot_gr_jax = jax_kernels.eval_jdot_gr_jax
eval_jdot_mb_jax = jax_kernels.eval_jdot_mb_jax
eval_edot_tidal_hut_jax = jax_kernels.eval_edot_tidal_hut_jax

__all__ = [
    "TIER5_BINARY_CASES",
    "BinarySystem",
    "eval_bh_spin_and_edd_jax",
    "eval_brott_wind_msun_yr",
    "eval_edot_tidal_hut_jax",
    "eval_gr_expected_separation_rsun",
    "eval_j_orb",
    "eval_j_orb_jax",
    "eval_jdot_gr_jax",
    "eval_jdot_mb_jax",
    "eval_kolb_mdot_thick",
    "eval_l2_mass_loss_fraction",
    "eval_ritter_mdot_thin",
    "eval_ritter_mdot_thin_jax",
    "eval_rl_relative_gap",
    "eval_rl_relative_gap_jax",
    "eval_rlobe",
    "eval_rlobe_jax",
    "eval_tauris_vdh_expected_separation_rsun",
    "period_from_separation",
    "period_from_separation_jax",
    "pick_next_timestep_rl",
    "run_binary_suite",
    "run_binary_test_suite_case",
    "separation_from_j_orb",
    "separation_from_j_orb_jax",
    "separation_from_period",
    "separation_from_period_jax",
    "solve_roche_lobe_mdot_jax",
]
