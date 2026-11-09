# SPDX-License-Identifier: LGPL-3.0-or-later
"""1D stellar structure, ZAMS/pre-MS initialization, `.mod`/`.photo` I/O,
zone/face microphysics, coupled Henyey solver, convective mixing/overshooting,
stellar winds, adaptive remeshing, H211 timestep control, and the evolution
driver (`flexmesa.star`).
"""

from __future__ import annotations

from flexmesa.star import (
    e2e,
    e2e_tier45,
    jax_diffusion_rot,
    jax_evolve,
    jax_hydro_pulse,
    jax_kernels,
    jax_winds_mix,
    model,
)

EvolveStepReport = model.EvolveStepReport
HenyeySolverReport = model.HenyeySolverReport
HenyeySystemResult = model.HenyeySystemResult
HistoryRow = model.HistoryRow
JaxStarState = model.JaxStarState
StarModel = model.StarModel
ALL_ACTIVE_STAR_CASES = e2e_tier45.ALL_ACTIVE_STAR_CASES
TIER1_CASES = e2e.TIER1_CASES
TIER2_CASES = e2e.TIER2_CASES
TIER3_CASES = e2e.TIER3_CASES
TIER4_ACTIVE_STAR_CASES = e2e_tier45.TIER4_ACTIVE_STAR_CASES
TIER4_ASTERO_CASES = e2e_tier45.TIER4_ASTERO_CASES
TIER4_DEV_CASES = e2e_tier45.TIER4_DEV_CASES
TIER4_STAR_CASES = e2e_tier45.TIER4_STAR_CASES
TIER5_STAR_CASES = e2e_tier45.TIER5_STAR_CASES
build_rsp_envelope_model = e2e_tier45.build_rsp_envelope_model
compute_staggered_grid_jax = jax_kernels.compute_staggered_grid_jax
eval_astero_scaling_dnu_numax_uhz = e2e_tier45.eval_astero_scaling_dnu_numax_uhz
eval_burgers_velocities_jax = jax_diffusion_rot.eval_burgers_velocities_jax
eval_caplan_ocp_dstar_jax = jax_diffusion_rot.eval_caplan_ocp_dstar_jax
eval_convection_sigmas_jax = jax_winds_mix.eval_convection_sigmas_jax
eval_dxdt_mix_jax = jax_winds_mix.eval_dxdt_mix_jax
eval_dziembowski_variables_jax = jax_hydro_pulse.eval_dziembowski_variables_jax
eval_face_mlt_and_brunt_jax = jax_kernels.eval_face_mlt_and_brunt_jax
eval_grafener_kudritzki_wind_msun_yr = (
    e2e_tier45.eval_grafener_kudritzki_wind_msun_yr
)
eval_h211_timestep_jax = jax_evolve.eval_h211_timestep_jax
eval_hllc_face_jax = jax_hydro_pulse.eval_hllc_face_jax
eval_hllc_surface_face_jax = jax_hydro_pulse.eval_hllc_surface_face_jax
eval_hydro_residuals_jax = jax_kernels.eval_hydro_residuals_jax
eval_linear_oscillation_modes = e2e_tier45.eval_linear_oscillation_modes
eval_mesh_gval_jax = jax_evolve.eval_mesh_gval_jax
eval_omega_crit_jax = jax_diffusion_rot.eval_omega_crit_jax
eval_optical_depth_jax = jax_kernels.eval_optical_depth_jax
eval_overshoot_dbl_exp = e2e_tier45.eval_overshoot_dbl_exp
eval_phase_separation_shift_jax = (
    jax_diffusion_rot.eval_phase_separation_shift_jax
)
eval_riemann_momentum_residual_jax = (
    jax_hydro_pulse.eval_riemann_momentum_residual_jax
)
eval_rotation_fp_ft_jax = jax_diffusion_rot.eval_rotation_fp_ft_jax
eval_rotational_mixing_zone_jax = (
    jax_diffusion_rot.eval_rotational_mixing_zone_jax
)
eval_rsp2_face_jax = jax_hydro_pulse.eval_rsp2_face_jax
eval_rsp2_zone_jax = jax_hydro_pulse.eval_rsp2_zone_jax
eval_rsp_2nd_crossing_edges = e2e_tier45.eval_rsp_2nd_crossing_edges
eval_rsp_lina_modes = e2e_tier45.eval_rsp_lina_modes
eval_target_mode_frequency_and_growth = (
    e2e_tier45.eval_target_mode_frequency_and_growth
)
eval_varcontrol_jax = jax_evolve.eval_varcontrol_jax
eval_winds_jax = jax_winds_mix.eval_winds_jax
face_weights_jax = jax_kernels.face_weights_jax
interp_q = model.interp_q
parse_inlist_bundle = e2e.parse_inlist_bundle
parse_rn_stages = e2e.parse_rn_stages
run_all_active_star_suite = e2e_tier45.run_all_active_star_suite
run_astero_suite = e2e_tier45.run_astero_suite
run_astero_test_suite_case = e2e_tier45.run_astero_test_suite_case
run_dev_cases_suite = e2e_tier45.run_dev_cases_suite
run_test_suite_case = e2e.run_test_suite_case
run_tier1_suite = e2e.run_tier1_suite
run_tier2_suite = e2e.run_tier2_suite
run_tier3_suite = e2e.run_tier3_suite
run_tier4_star_suite = e2e_tier45.run_tier4_star_suite
run_tier4_suite = e2e_tier45.run_tier4_suite
run_tier5_star_suite = e2e_tier45.run_tier5_star_suite
solve_henyey_linear_jax = jax_kernels.solve_henyey_linear_jax
solve_stationary_rsp2_w_jax = jax_hydro_pulse.solve_stationary_rsp2_w_jax
verify_step_redo_bit_parity = e2e_tier45.verify_step_redo_bit_parity
yrs_for_init_timestep = model.yrs_for_init_timestep

__all__ = [
    "ALL_ACTIVE_STAR_CASES",
    "TIER1_CASES",
    "TIER2_CASES",
    "TIER3_CASES",
    "TIER4_ACTIVE_STAR_CASES",
    "TIER4_ASTERO_CASES",
    "TIER4_DEV_CASES",
    "TIER4_STAR_CASES",
    "TIER5_STAR_CASES",
    "EvolveStepReport",
    "HenyeySolverReport",
    "HenyeySystemResult",
    "HistoryRow",
    "JaxStarState",
    "StarModel",
    "build_rsp_envelope_model",
    "compute_staggered_grid_jax",
    "e2e",
    "e2e_tier45",
    "eval_astero_scaling_dnu_numax_uhz",
    "eval_burgers_velocities_jax",
    "eval_caplan_ocp_dstar_jax",
    "eval_convection_sigmas_jax",
    "eval_dxdt_mix_jax",
    "eval_dziembowski_variables_jax",
    "eval_face_mlt_and_brunt_jax",
    "eval_grafener_kudritzki_wind_msun_yr",
    "eval_h211_timestep_jax",
    "eval_hllc_face_jax",
    "eval_hllc_surface_face_jax",
    "eval_hydro_residuals_jax",
    "eval_linear_oscillation_modes",
    "eval_mesh_gval_jax",
    "eval_omega_crit_jax",
    "eval_optical_depth_jax",
    "eval_overshoot_dbl_exp",
    "eval_phase_separation_shift_jax",
    "eval_riemann_momentum_residual_jax",
    "eval_rotation_fp_ft_jax",
    "eval_rotational_mixing_zone_jax",
    "eval_rsp2_face_jax",
    "eval_rsp2_zone_jax",
    "eval_rsp_2nd_crossing_edges",
    "eval_rsp_lina_modes",
    "eval_target_mode_frequency_and_growth",
    "eval_varcontrol_jax",
    "eval_winds_jax",
    "face_weights_jax",
    "interp_q",
    "jax_diffusion_rot",
    "jax_evolve",
    "jax_hydro_pulse",
    "jax_kernels",
    "jax_winds_mix",
    "model",
    "parse_inlist_bundle",
    "parse_rn_stages",
    "run_all_active_star_suite",
    "run_astero_suite",
    "run_astero_test_suite_case",
    "run_dev_cases_suite",
    "run_test_suite_case",
    "run_tier1_suite",
    "run_tier2_suite",
    "run_tier3_suite",
    "run_tier4_star_suite",
    "run_tier4_suite",
    "run_tier5_star_suite",
    "solve_henyey_linear_jax",
    "solve_stationary_rsp2_w_jax",
    "verify_step_redo_bit_parity",
    "yrs_for_init_timestep",
]
