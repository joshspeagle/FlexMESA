# SPDX-License-Identifier: GPL-3.0-or-later
"""1D stellar structure, ZAMS/pre-MS initialization, `.mod`/`.photo` I/O,
zone/face microphysics, coupled Henyey solver, convective mixing/overshooting,
stellar winds, adaptive remeshing, H211 timestep control, and the evolution
driver (`flexmesa.star`).

Note:
    The JAX `star` helper kernels (`jax_kernels`, `jax_winds_mix`, `jax_evolve`,
    `jax_diffusion_rot`, `jax_hydro_pulse`, `jax_rsp`;
    `JAX_ENGINE_STATUS == "frozen-unverified-until-#143"`) are **frozen and
    unverified** under `#166` (`DECISIONS #17`) until the canonical Rust engine
    passes the `#143` upstream MESA `26.4.1` history harness. No JAX stellar
    evolution step exists (`StarModel` wraps the Rust solver).
"""

from __future__ import annotations

from flexmesa import _dispatch
from flexmesa.star import (
    e2e,
    jax_diffusion_rot,
    jax_evolve,
    jax_hydro_pulse,
    jax_kernels,
    jax_rsp,
    jax_winds_mix,
    model,
)

JAX_ENGINE_STATUS: str = _dispatch.FROZEN_JAX_ENGINE_STATUS

EvolveStepReport = model.EvolveStepReport
HenyeySolverReport = model.HenyeySolverReport
HenyeySystemResult = model.HenyeySystemResult
HistoryRow = model.HistoryRow
JaxStarState = model.JaxStarState
StarModel = model.StarModel
build_rsp_envelope_model = e2e.build_rsp_envelope_model
compute_rsp2_uq_face_jax = jax_rsp.compute_rsp2_uq_face_jax
compute_staggered_grid_jax = jax_kernels.compute_staggered_grid_jax
copy_cleaned_case = e2e.copy_cleaned_case
eval_astero_scaling_dnu_numax_uhz = e2e.eval_astero_scaling_dnu_numax_uhz
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
eval_linear_oscillation_modes = e2e.eval_linear_oscillation_modes
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
eval_rsp2_face_jax = jax_rsp.eval_rsp2_face_jax
eval_rsp2_turbulent_eqn_residual_jax = (
    jax_rsp.eval_rsp2_turbulent_eqn_residual_jax
)
eval_rsp2_zone_jax = jax_rsp.eval_rsp2_zone_jax
eval_rsp_hyd_fluxes_jax = jax_rsp.eval_rsp_hyd_fluxes_jax
eval_rsp_lina_modes = e2e.eval_rsp_lina_modes
eval_sobolev_feii_5169_tau_jax = jax_hydro_pulse.eval_sobolev_feii_5169_tau_jax
eval_target_mode_frequency_and_growth = (
    e2e.eval_target_mode_frequency_and_growth
)
eval_varcontrol_jax = jax_evolve.eval_varcontrol_jax
eval_winds_jax = jax_winds_mix.eval_winds_jax
face_weights_jax = jax_kernels.face_weights_jax
interp_q = model.interp_q
parse_inlist_bundle = e2e.parse_inlist_bundle
parse_rn_stages = e2e.parse_rn_stages
rsp2_flux_limiter_jax = jax_rsp.rsp2_flux_limiter_jax
run_stella_from_hyd_abn = jax_hydro_pulse.run_stella_from_hyd_abn
run_test_suite_case = e2e.run_test_suite_case
solve_adipls_fgong_modes = jax_hydro_pulse.solve_adipls_fgong_modes
solve_gyre_mesa_modes = jax_hydro_pulse.solve_gyre_mesa_modes
solve_henyey_linear_jax = jax_kernels.solve_henyey_linear_jax
solve_stationary_rsp2_w_jax = jax_rsp.solve_stationary_rsp2_w_jax
yrs_for_init_timestep = model.yrs_for_init_timestep

__all__ = [
    "JAX_ENGINE_STATUS",
    "EvolveStepReport",
    "HenyeySolverReport",
    "HenyeySystemResult",
    "HistoryRow",
    "JaxStarState",
    "StarModel",
    "build_rsp_envelope_model",
    "compute_rsp2_uq_face_jax",
    "compute_staggered_grid_jax",
    "copy_cleaned_case",
    "e2e",
    "eval_astero_scaling_dnu_numax_uhz",
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
    "eval_linear_oscillation_modes",
    "eval_mesh_gval_jax",
    "eval_omega_crit_jax",
    "eval_optical_depth_jax",
    "eval_phase_separation_shift_jax",
    "eval_riemann_momentum_residual_jax",
    "eval_rotation_fp_ft_jax",
    "eval_rotational_mixing_zone_jax",
    "eval_rsp2_face_jax",
    "eval_rsp2_turbulent_eqn_residual_jax",
    "eval_rsp2_zone_jax",
    "eval_rsp_hyd_fluxes_jax",
    "eval_rsp_lina_modes",
    "eval_sobolev_feii_5169_tau_jax",
    "eval_target_mode_frequency_and_growth",
    "eval_varcontrol_jax",
    "eval_winds_jax",
    "face_weights_jax",
    "interp_q",
    "jax_diffusion_rot",
    "jax_evolve",
    "jax_hydro_pulse",
    "jax_kernels",
    "jax_rsp",
    "jax_winds_mix",
    "model",
    "parse_inlist_bundle",
    "parse_rn_stages",
    "rsp2_flux_limiter_jax",
    "run_stella_from_hyd_abn",
    "run_test_suite_case",
    "solve_adipls_fgong_modes",
    "solve_gyre_mesa_modes",
    "solve_henyey_linear_jax",
    "solve_stationary_rsp2_w_jax",
    "yrs_for_init_timestep",
]
