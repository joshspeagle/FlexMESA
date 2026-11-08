# SPDX-License-Identifier: LGPL-3.0-or-later
"""1D stellar structure, ZAMS/pre-MS initialization, `.mod`/`.photo` I/O,
zone/face microphysics, coupled Henyey solver, convective mixing/overshooting,
and stellar winds (`flexmesa.star`).
"""

from __future__ import annotations

from flexmesa.star import jax_kernels, jax_winds_mix, model

HenyeySolverReport = model.HenyeySolverReport
HenyeySystemResult = model.HenyeySystemResult
JaxStarState = model.JaxStarState
StarModel = model.StarModel
compute_staggered_grid_jax = jax_kernels.compute_staggered_grid_jax
eval_convection_sigmas_jax = jax_winds_mix.eval_convection_sigmas_jax
eval_dxdt_mix_jax = jax_winds_mix.eval_dxdt_mix_jax
eval_face_mlt_and_brunt_jax = jax_kernels.eval_face_mlt_and_brunt_jax
eval_hydro_residuals_jax = jax_kernels.eval_hydro_residuals_jax
eval_optical_depth_jax = jax_kernels.eval_optical_depth_jax
eval_winds_jax = jax_winds_mix.eval_winds_jax
face_weights_jax = jax_kernels.face_weights_jax
interp_q = model.interp_q
solve_henyey_linear_jax = jax_kernels.solve_henyey_linear_jax
yrs_for_init_timestep = model.yrs_for_init_timestep

__all__ = [
    "HenyeySolverReport",
    "HenyeySystemResult",
    "JaxStarState",
    "StarModel",
    "compute_staggered_grid_jax",
    "eval_convection_sigmas_jax",
    "eval_dxdt_mix_jax",
    "eval_face_mlt_and_brunt_jax",
    "eval_hydro_residuals_jax",
    "eval_optical_depth_jax",
    "eval_winds_jax",
    "face_weights_jax",
    "interp_q",
    "jax_kernels",
    "jax_winds_mix",
    "model",
    "solve_henyey_linear_jax",
    "yrs_for_init_timestep",
]
