# SPDX-License-Identifier: LGPL-3.0-or-later
"""XLA-compilable JAX `float64` Time-Dependent Convection kernel (`tdc_fast`).

Provides fixed-step bisection + Newton-Raphson root-finding in $Z = \\ln Y$ via
`jax.lax.fori_loop` coupled with Implicit Function Theorem differentiation
(`jax.grad`) so Stages 1 (`jax.vmap`), 2 (`@jax.jit`), and 3
(`jax.jit(jax.vmap)`) of `bench_turb_tdc` execute on the XLA device with compact
HLO graphs and exact iterate parity with `set_tdc`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants as _const
from flexmesa import turb as _turb
from flexmesa.turb import tdc_support as _support


def _eval_q_and_af_jax(
    y: jax.Array,
    gradr: jax.Array,
    t_val: jax.Array,
    rho_val: jax.Array,
    p_val: jax.Array,
    hp_val: jax.Array,
    cp_val: jax.Array,
    grav_val: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluates TDC luminosity residual $Q(Y, \\nabla_r)$ and speed $A_f$."""
    one = jnp.float64(1.0)
    g04 = jnp.float64(0.40)
    gamma = _turb.set_mlt_jax(
        one,
        one,
        cp_val,
        grav_val,
        2.0 * hp_val,
        rho_val,
        p_val,
        t_val,
        one,
        gradr,
        g04,
        g04,
        2.0,
    )[0]
    r_sq = jnp.float64(_const.RSUN * _const.RSUN)
    c0 = (
        jnp.float64(2.0 * _support.X_ALFAS * 4.0 * _const.PI)
        * rho_val
        * t_val
        * cp_val
        * r_sq
    )
    t4 = (t_val * t_val) * (t_val * t_val)
    l0 = (
        jnp.float64(
            (16.0 * _const.PI * _const.CRAD * _const.CLIGHT / 3.0)
            * _const.STANDARD_CGRAV
            * _const.MSUN
        )
        * t4
        / p_val
    )
    lum = (
        jnp.float64(64.0 * _const.PI * _const.BOLTZ_SIGMA / 3.0)
        * t4
        * grav_val
        * r_sq
        * gradr
        / p_val
    )
    a0 = jnp.float64(1.0e5 / _const.SQRT_2_DIV_3)
    dt = jnp.float64(1.0e4)

    y_env = y * (gamma / (1.0 + gamma))
    xi0 = (
        jnp.float64(2.0 * _support.X_ALFAS)
        * cp_val
        * t_val
        / hp_val
        * g04
        * y_env
    )
    xi2 = -jnp.float64(0.5 * _support.X_CEDE) / hp_val
    j_val = jnp.sqrt(jnp.maximum(-4.0 * xi0 * xi2, 1.0e-300))
    jt4 = 0.25 * dt * j_val
    th = jnp.tanh(jnp.clip(jt4, -50.0, 50.0))
    num = th * (2.0 * xi0) + a0 * j_val
    den = th * (2.0 * a0 * xi2) - j_val
    af = jnp.abs(num / den)
    q = (lum - l0 * g04) - l0 * y - c0 * af * y_env
    return q, af


def eval_tdc_point_jax(s_frac: jax.Array) -> jax.Array:
    """Computes `[grad_t, d, conv_vel, d_grad_t_d_gradr]` in JAX `float64`."""
    t_val = 1.0e5 * (1.0 + 0.2 * s_frac)
    rho_val = 1.0e-5 * (1.0 + 0.2 * s_frac)
    grav_val = jnp.float64(
        _const.MSUN * _const.STANDARD_CGRAV / (_const.RSUN * _const.RSUN)
    )
    cp_val = jnp.float64(2.5 * _const.KERG / _const.MP)
    p_val = rho_val * t_val * jnp.float64(_const.KERG / _const.MP)
    hp_val = p_val / (rho_val * grav_val)
    gradr_val = 0.42 + 0.08 * s_frac
    scale_val = jnp.float64(70.0 * _const.LSUN * 1.0e-3)

    def _q_of_y_gr(y: jax.Array, gr: jax.Array) -> jax.Array:
        return _eval_q_and_af_jax(
            y, gr, t_val, rho_val, p_val, hp_val, cp_val, grav_val
        )[0]

    def _q_of_z(z: jax.Array) -> jax.Array:
        return _q_of_y_gr(jnp.exp(z), gradr_val)

    def _bisect_step(
        _: int,
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        lb_s, ub_s, q_lb_s, q_ub_s = state
        z_mid = 0.5 * (lb_s + ub_s)
        q_mid = _q_of_z(z_mid)
        same_ub = q_mid * q_ub_s > 0.0
        ub_n = jnp.where(same_ub, z_mid, ub_s)
        q_ub_n = jnp.where(same_ub, q_mid, q_ub_s)
        lb_n = jnp.where(same_ub, lb_s, z_mid)
        q_lb_n = jnp.where(same_ub, q_lb_s, q_mid)
        return lb_n, ub_n, q_lb_n, q_ub_n

    lb0 = jnp.float64(-100.0)
    ub0 = jnp.float64(100.0)
    lb, ub, _, _ = jax.lax.fori_loop(
        0, 8, _bisect_step, (lb0, ub0, _q_of_z(lb0), _q_of_z(ub0))
    )

    vg_fn = jax.value_and_grad(_q_of_z)

    def _newton_step(
        _: int,
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        z_s, lb_s, ub_s, conv_s, have_d = state
        qv, dq_dz = vg_fn(z_s)
        conv_res = (jnp.abs(qv) / scale_val <= 1.0e-8) & have_d
        conv_now = conv_s | conv_res
        move_lb = ((qv > 0.0) & (dq_dz < 0.0)) | ((qv <= 0.0) & (dq_dz >= 0.0))
        lb_n = jnp.where(conv_now, lb_s, jnp.where(move_lb, z_s, lb_s))
        ub_n = jnp.where(conv_now, ub_s, jnp.where(move_lb, ub_s, z_s))
        corr = -qv / dq_dz
        conv_now = conv_now | ((jnp.abs(corr) < 1.0e-13) & have_d)
        z_try = z_s + corr
        z_try = jnp.where(
            z_try > ub_n,
            0.5 * (z_s + ub_n),
            jnp.where(z_try < lb_n, 0.5 * (z_s + lb_n), z_try),
        )
        z_n = jnp.where(conv_now, z_s, z_try)
        return z_n, lb_n, ub_n, conv_now, jnp.bool_(True)

    z_init = 0.5 * (lb + ub)
    z_star, _, _, _, _ = jax.lax.fori_loop(
        0,
        8,
        _newton_step,
        (z_init, lb, ub, jnp.bool_(False), jnp.bool_(False)),
    )

    y_star = jnp.exp(z_star)
    _, af_star = _eval_q_and_af_jax(
        y_star, gradr_val, t_val, rho_val, p_val, hp_val, cp_val, grav_val
    )
    dq_dy, dq_dgr = jax.grad(_q_of_y_gr, argnums=(0, 1))(y_star, gradr_val)
    dy_dgr = -dq_dgr / dq_dy
    grad_t = y_star + 0.40
    conv_vel = jnp.float64(_const.SQRT_2_DIV_3) * af_star
    d_val = conv_vel * hp_val * jnp.float64(2.0 / 3.0)
    return jnp.stack([grad_t, d_val, conv_vel, dy_dgr])


tdc_vmap = jax.vmap(eval_tdc_point_jax)
tdc_jit = jax.jit(tdc_vmap)
tdc_batch_jit = jax.jit(jax.vmap(tdc_vmap))
