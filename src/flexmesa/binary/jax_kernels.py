# SPDX-License-Identifier: LGPL-3.0-or-later
"""Re-export facade for differentiable JAX (`float64`) binary kernels
(`flexmesa.binary.jax_kernels`).
"""

from __future__ import annotations

from flexmesa.binary import jax_jdot_tides, jax_orbit_mdot

eval_bh_spin_and_edd_jax = jax_jdot_tides.eval_bh_spin_and_edd_jax
eval_edot_tidal_hut_jax = jax_jdot_tides.eval_edot_tidal_hut_jax
eval_jdot_gr_jax = jax_jdot_tides.eval_jdot_gr_jax
eval_jdot_mb_jax = jax_jdot_tides.eval_jdot_mb_jax

eval_j_orb_jax = jax_orbit_mdot.eval_j_orb_jax
eval_ritter_mdot_thin_jax = jax_orbit_mdot.eval_ritter_mdot_thin_jax
eval_rl_relative_gap_jax = jax_orbit_mdot.eval_rl_relative_gap_jax
eval_rlobe_jax = jax_orbit_mdot.eval_rlobe_jax
period_from_separation_jax = jax_orbit_mdot.period_from_separation_jax
separation_from_j_orb_jax = jax_orbit_mdot.separation_from_j_orb_jax
separation_from_period_jax = jax_orbit_mdot.separation_from_period_jax
solve_roche_lobe_mdot_jax = jax_orbit_mdot.solve_roche_lobe_mdot_jax

__all__ = [
    "eval_bh_spin_and_edd_jax",
    "eval_edot_tidal_hut_jax",
    "eval_j_orb_jax",
    "eval_jdot_gr_jax",
    "eval_jdot_mb_jax",
    "eval_ritter_mdot_thin_jax",
    "eval_rl_relative_gap_jax",
    "eval_rlobe_jax",
    "period_from_separation_jax",
    "separation_from_j_orb_jax",
    "separation_from_period_jax",
    "solve_roche_lobe_mdot_jax",
]
