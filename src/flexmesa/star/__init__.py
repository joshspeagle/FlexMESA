# SPDX-License-Identifier: LGPL-3.0-or-later
"""1D stellar structure, ZAMS/pre-MS builders, I/O, and microphysics."""

from __future__ import annotations

from flexmesa.star import jax_kernels, model

JaxStarState = model.JaxStarState
StarModel = model.StarModel
compute_staggered_grid_jax = jax_kernels.compute_staggered_grid_jax
eval_face_mlt_and_brunt_jax = jax_kernels.eval_face_mlt_and_brunt_jax
eval_optical_depth_jax = jax_kernels.eval_optical_depth_jax
face_weights_jax = jax_kernels.face_weights_jax
interp_cell_to_face_jax = jax_kernels.interp_cell_to_face_jax
interp_q = model.interp_q
yrs_for_init_timestep = model.yrs_for_init_timestep

__all__: list[str] = [
    "JaxStarState",
    "StarModel",
    "compute_staggered_grid_jax",
    "eval_face_mlt_and_brunt_jax",
    "eval_optical_depth_jax",
    "face_weights_jax",
    "interp_cell_to_face_jax",
    "interp_q",
    "jax_kernels",
    "model",
    "yrs_for_init_timestep",
]
