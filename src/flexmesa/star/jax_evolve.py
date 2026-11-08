# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differentiable JAX kernels for adaptive mesh spacing variation (`g_k`),
structural variation (`varcontrol`), and the Söderlind & Wang (2006) H211PI
digital timestep filter (`flexmesa.star`, Wave 4.4).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants

_LN10 = float(constants.LN10)
_L_SUN = float(constants.L_SUN)


@jax.jit
def eval_mesh_gval_jax(
    ln_peos: jax.Array,
    ln_t: jax.Array,
    mesh_delta_coeff: float = 1.0,
    p_function_weight: float = 30.0,
    t_function1_weight: float = 110.0,
) -> jax.Array:
    """Compute dimensionless mesh variation indicator `g_k` (`(nz,)`).

    Args:
        ln_peos: Cell logarithmic pressures of shape `(nz,)`.
        ln_t: Cell logarithmic temperatures of shape `(nz,)`.
        mesh_delta_coeff: Global mesh spacing factor (`1.0` default).
        p_function_weight: Pressure spacing weight (`30.0` default).
        t_function1_weight: Temperature spacing weight (`110.0` default).

    Returns:
        Dimensionless mesh variation array `gval` of shape `(nz,)`.
    """
    inv_ln10 = 1.0 / _LN10
    df_p = p_function_weight * jnp.abs(ln_peos[:-1] - ln_peos[1:]) * inv_ln10
    df_t = t_function1_weight * jnp.abs(ln_t[:-1] - ln_t[1:]) * inv_ln10
    g_inner = jnp.maximum(mesh_delta_coeff, 0.0) * jnp.maximum(df_p, df_t)
    return jnp.concatenate([g_inner, g_inner[-1:]])


@jax.jit
def eval_varcontrol_jax(
    lnd: jax.Array,
    ln_t: jax.Array,
    ln_r: jax.Array,
    lum: jax.Array,
    lnd_old: jax.Array,
    ln_t_old: jax.Array,
    ln_r_old: jax.Array,
    lum_old: jax.Array,
) -> jax.Array:
    """Evaluate normalized structural variation `varcontrol` (`timestep.f90`).

    Args:
        lnd: Current cell log-densities of shape `(nz,)`.
        ln_t: Current cell log-temperatures of shape `(nz,)`.
        ln_r: Current face log-radii of shape `(nz,)`.
        lum: Current face luminosities of shape `(nz,)` (`erg s^-1`).
        lnd_old: Previous step cell log-densities of shape `(nz,)`.
        ln_t_old: Previous step cell log-temperatures of shape `(nz,)`.
        ln_r_old: Previous step face log-radii of shape `(nz,)`.
        lum_old: Previous step face luminosities of shape `(nz,)` (`erg s^-1`).

    Returns:
        Scalar JAX array holding `varcontrol`.
    """
    nz = lnd.shape[0]
    l_scale = jnp.maximum(jnp.abs(lum[0]), _L_SUN)
    dlnd = jnp.abs(lnd - lnd_old) / (1.0 + jnp.abs(lnd))
    dlnt = jnp.abs(ln_t - ln_t_old) / (1.0 + jnp.abs(ln_t))
    dlnr = jnp.abs(ln_r - ln_r_old) / (1.0 + jnp.abs(ln_r))
    dlum = jnp.abs(lum - lum_old) / (l_scale + jnp.abs(lum))
    return jnp.sum(dlnd + dlnt + dlnr + dlum) / (4.0 * nz)


def _filter_limit_jax(x: jax.Array | float) -> jax.Array:
    """Smooth arctangent step-ratio limiter `1 + 10 * arctan((x - 1) / 10)`."""
    kappa = 10.0
    return 1.0 + kappa * jnp.arctan((x - 1.0) / kappa)


@jax.jit
def eval_h211_timestep_jax(
    dt: float | jax.Array,
    dt_old: float | jax.Array,
    dt_limit_ratio: float | jax.Array,
    dt_limit_ratio_old: float | jax.Array = 1.0,
    max_timestep_factor: float = 1.2,
    min_timestep_factor: float = 0.5,
) -> jax.Array:
    """Evaluate Söderlind & Wang (2006) H211PI next timestep `dt_next` (`s`).

    Args:
        dt: Current timestep in seconds (`s`).
        dt_old: Previous timestep in seconds (`s`).
        dt_limit_ratio: Current step variation ratio `w_n`.
        dt_limit_ratio_old: Previous step variation ratio `w_{n-1}`.
        max_timestep_factor: Upper multiplier cap (`1.2` default).
        min_timestep_factor: Lower multiplier floor (`0.5` default).

    Returns:
        Proposed next timestep `dt_next` in seconds (`s`).
    """
    dt_arr = jnp.asarray(dt, dtype=jnp.float64)
    dt_old_arr = jnp.asarray(dt_old, dtype=jnp.float64)
    w_curr = jnp.maximum(
        jnp.asarray(dt_limit_ratio, dtype=jnp.float64), 1.0e-10
    )
    w_prev_in = jnp.asarray(dt_limit_ratio_old, dtype=jnp.float64)
    w_prev = jnp.where(w_prev_in > 0.0, jnp.maximum(w_prev_in, 1.0e-10), w_curr)
    dt_ratio = jnp.where(
        (dt_old_arr > 0.0) & (dt_arr > 0.0),
        jnp.maximum(dt_arr / dt_old_arr, 1.0e-10),
        1.0,
    )

    r_n = jnp.maximum(_filter_limit_jax(1.0 / w_curr), 1.0e-4)
    r_nm1 = jnp.maximum(_filter_limit_jax(1.0 / w_prev), 1.0e-4)
    raw_factor = _filter_limit_jax(
        (r_n**0.25) * (r_nm1**0.25) * (dt_ratio ** (-0.25))
    )
    clamped = jnp.clip(raw_factor, min_timestep_factor, max_timestep_factor)
    return dt_arr * clamped
