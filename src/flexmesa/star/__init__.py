# SPDX-License-Identifier: LGPL-3.0-or-later
"""1D stellar structure, ZAMS/pre-MS initialization, `.mod`/`.photo` I/O,
zone/face microphysics, coupled Henyey solver, convective mixing/overshooting,
stellar winds, adaptive remeshing, H211 timestep control, and the evolution
driver (`flexmesa.star`).
"""

from __future__ import annotations

from flexmesa.star import (
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
compute_staggered_grid_jax = jax_kernels.compute_staggered_grid_jax
eval_burgers_velocities_jax = jax_diffusion_rot.eval_burgers_velocities_jax
eval_caplan_ocp_dstar_jax = jax_diffusion_rot.eval_caplan_ocp_dstar_jax
eval_convection_sigmas_jax = jax_winds_mix.eval_convection_sigmas_jax
eval_dxdt_mix_jax = jax_winds_mix.eval_dxdt_mix_jax
eval_dziembowski_variables_jax = jax_hydro_pulse.eval_dziembowski_variables_jax
eval_face_mlt_and_brunt_jax = jax_kernels.eval_face_mlt_and_brunt_jax
eval_h211_timestep_jax = jax_evolve.eval_h211_timestep_jax
eval_hllc_face_jax = jax_hydro_pulse.eval_hllc_face_jax
eval_hllc_surface_face_jax = jax_hydro_pulse.eval_hllc_surface_face_jax
eval_hydro_residuals_jax = jax_kernels.eval_hydro_residuals_jax
eval_mesh_gval_jax = jax_evolve.eval_mesh_gval_jax
eval_omega_crit_jax = jax_diffusion_rot.eval_omega_crit_jax
eval_optical_depth_jax = jax_kernels.eval_optical_depth_jax
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
eval_varcontrol_jax = jax_evolve.eval_varcontrol_jax
eval_winds_jax = jax_winds_mix.eval_winds_jax
face_weights_jax = jax_kernels.face_weights_jax
interp_q = model.interp_q
solve_henyey_linear_jax = jax_kernels.solve_henyey_linear_jax
solve_stationary_rsp2_w_jax = jax_hydro_pulse.solve_stationary_rsp2_w_jax
yrs_for_init_timestep = model.yrs_for_init_timestep

__all__ = [
    "EvolveStepReport",
    "HenyeySolverReport",
    "HenyeySystemResult",
    "HistoryRow",
    "JaxStarState",
    "StarModel",
    "compute_staggered_grid_jax",
    "eval_burgers_velocities_jax",
    "eval_caplan_ocp_dstar_jax",
    "eval_convection_sigmas_jax",
    "eval_dxdt_mix_jax",
    "eval_dziembowski_variables_jax",
    "eval_face_mlt_and_brunt_jax",
    "eval_h211_timestep_jax",
    "eval_hllc_face_jax",
    "eval_hllc_surface_face_jax",
    "eval_hydro_residuals_jax",
    "eval_mesh_gval_jax",
    "eval_omega_crit_jax",
    "eval_optical_depth_jax",
    "eval_phase_separation_shift_jax",
    "eval_riemann_momentum_residual_jax",
    "eval_rotation_fp_ft_jax",
    "eval_rotational_mixing_zone_jax",
    "eval_rsp2_face_jax",
    "eval_rsp2_zone_jax",
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
    "solve_henyey_linear_jax",
    "solve_stationary_rsp2_w_jax",
    "yrs_for_init_timestep",
]
