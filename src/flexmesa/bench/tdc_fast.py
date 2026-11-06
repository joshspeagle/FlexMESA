# SPDX-License-Identifier: LGPL-3.0-or-later
"""XLA-compilable JAX `float64` Time-Dependent Convection kernel (`tdc_fast`).

Provides fixed-step bisection + Newton-Raphson root-finding in $Z = \\ln Y$ via
`jax.lax.fori_loop` coupled with Implicit Function Theorem differentiation
(`jax.grad`) so Stages 2 (`@jax.jit`) and 3 (`jax.jit(jax.vmap)`) of
`bench_turb_tdc` execute on the XLA device with compact HLO graphs and exact
iterate parity with `set_tdc`.
"""

from __future__ import annotations

from typing import Final

import jax
import jax.numpy as jnp

from flexmesa import constants as _const
from flexmesa import turb as _turb
from flexmesa.turb import tdc_support as _tdc_support

X_ALFAS: Final[float] = _tdc_support.X_ALFAS
X_CEDE: Final[float] = _tdc_support.X_CEDE
GRAV_SOLAR: Final[float] = (
    _const.MSUN * _const.STANDARD_CGRAV / (_const.RSUN * _const.RSUN)
)
RGAS_IDEAL: Final[float] = _const.KERG / _const.MP
CP_IDEAL: Final[float] = 2.5 * RGAS_IDEAL


def _eval_q_and_af_given_gamma(
    y: jax.Array,
    gamma: jax.Array,
    lum: jax.Array,
    c0: jax.Array,
    l0: jax.Array,
    hp_val: jax.Array,
    cp_val: jax.Array,
    t_val: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluates TDC residual $Q(Y)$ and speed $A_f$ for precomputed $\\Gamma$.

    Args:
        y: Dimensionless superadiabaticity $Y = \\nabla_T - \\nabla_L$.
        gamma: Convective efficiency $\\Gamma$ from steady-state Cox MLT.
        lum: Local luminosity $L$ in $\\mathrm{erg\\,s^{-1}}$.
        c0: Convective luminosity prefactor $c_0$ in $\\mathrm{erg\\,cm^{-1}}$.
        l0: Radiative luminosity prefactor $L_0$ in $\\mathrm{erg\\,s^{-1}}$.
        hp_val: Pressure scale height $H_P$ in $\\mathrm{cm}$.
        cp_val: Specific heat $C_P$ in $\\mathrm{erg\\,g^{-1}\\,K^{-1}}$.
        t_val: Temperature $T$ in $\\mathrm{K}$.

    Returns:
        Two-tuple `(q, af)` of luminosity residual $Q(Y)$ in
        $\\mathrm{erg\\,s^{-1}}$ and scaled convective speed $A_f$ in
        $\\mathrm{cm\\,s^{-1}}$.
    """
    g04 = jnp.float64(0.40)
    a0 = jnp.float64(1.0e5 / _const.SQRT_2_DIV_3)
    dt = jnp.float64(1.0e4)

    y_env = y * (gamma / (1.0 + gamma))
    xi0 = jnp.float64(2.0 * X_ALFAS) * cp_val * t_val / hp_val * g04 * y_env
    xi2 = -jnp.float64(0.5 * X_CEDE) / hp_val
    j_val = jnp.sqrt(jnp.maximum(-4.0 * xi0 * xi2, 1.0e-300))
    jt4 = 0.25 * dt * j_val
    th = jnp.tanh(jnp.clip(jt4, -50.0, 50.0))
    num = th * (2.0 * xi0) + a0 * j_val
    den = th * (2.0 * a0 * xi2) - j_val
    af = jnp.abs(num / den)
    q = (lum - l0 * g04) - l0 * y - c0 * af * y_env
    return q, af


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
    """Evaluates TDC luminosity residual $Q(Y, \\nabla_r)$ and speed $A_f$.

    Args:
        y: Dimensionless superadiabaticity $Y = \\nabla_T - \\nabla_L$.
        gradr: Radiative temperature gradient $\\nabla_r$ (dimensionless).
        t_val: Temperature $T$ in $\\mathrm{K}$.
        rho_val: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        p_val: Total pressure $P$ in $\\mathrm{dyn\\,cm^{-2}}$.
        hp_val: Pressure scale height $H_P$ in $\\mathrm{cm}$.
        cp_val: Specific heat $C_P$ in $\\mathrm{erg\\,g^{-1}\\,K^{-1}}$.
        grav_val: Local gravity $g$ in $\\mathrm{cm\\,s^{-2}}$.

    Returns:
        Two-tuple `(q, af)` of luminosity residual $Q(Y)$ in
        $\\mathrm{erg\\,s^{-1}}$ and final scaled convective speed $A_f$ in
        $\\mathrm{cm\\,s^{-1}}$.
    """
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
        jnp.float64(2.0 * X_ALFAS * 4.0 * _const.PI)
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
    return _eval_q_and_af_given_gamma(
        y, gamma, lum, c0, l0, hp_val, cp_val, t_val
    )


def eval_tdc_point_jax(s_frac: jax.Array) -> jax.Array:
    """Computes `[grad_t, d, conv_vel, d_grad_t_d_gradr]` in JAX `float64`.

    Narrows the logarithmic superadiabaticity bracket $Z = \\ln Y \\in [-100,
    100]$ to width $< 1.0$ via 8 bisection steps (`jax.lax.fori_loop`), refines
    $Z^*$ via 20 damped Newton-Raphson iterations with the exact MESA residual
    ($|Q| / \\mathrm{scale} \\le 10^{-8}$) and step ($|\\Delta Z| < 10^{-13}$)
    convergence latches, and propagates $\\partial \\nabla_T / \\partial
    \\nabla_r$ at the converged root $Y^* = \\exp(Z^*)$ via the Implicit
    Function Theorem:
    $$\\frac{\\partial Y^*}{\\partial \\nabla_r} =
      -\\left(\\frac{\\partial Q}{\\partial Y}\\right)^{-1}
      \\frac{\\partial Q}{\\partial \\nabla_r}.$$

    Args:
        s_frac: Normalized radial shell coordinate $s \\in [0, 1]$ (`float64`
            scalar).

    Returns:
        4-element `float64` JAX array `[grad_t, d, conv_vel, d_grad_t_d_gradr]`
        of shape `(4,)` in CGS units (dimensionless, $\\mathrm{cm^2\\,s^{-1}}$,
        $\\mathrm{cm\\,s^{-1}}$, dimensionless).
    """
    t_val = 1.0e5 * (1.0 + 0.2 * s_frac)
    rho_val = 1.0e-5 * (1.0 + 0.2 * s_frac)
    grav_val = jnp.float64(GRAV_SOLAR)
    cp_val = jnp.float64(CP_IDEAL)
    p_val = rho_val * t_val * jnp.float64(RGAS_IDEAL)
    hp_val = p_val / (rho_val * grav_val)
    gradr_val = 0.42 + 0.08 * s_frac
    scale_val = jnp.float64(70.0 * _const.LSUN * 1.0e-3)

    # Hoist loop-invariant MLT efficiency and luminosity prefactors outside
    # the bisection and Newton-Raphson root-finding loops.
    one = jnp.float64(1.0)
    g04 = jnp.float64(0.40)
    gamma0 = _turb.set_mlt_jax(
        one,
        one,
        cp_val,
        grav_val,
        2.0 * hp_val,
        rho_val,
        p_val,
        t_val,
        one,
        gradr_val,
        g04,
        g04,
        2.0,
    )[0]
    r_sq = jnp.float64(_const.RSUN * _const.RSUN)
    c0 = (
        jnp.float64(2.0 * X_ALFAS * 4.0 * _const.PI)
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
    lum0 = (
        jnp.float64(64.0 * _const.PI * _const.BOLTZ_SIGMA / 3.0)
        * t4
        * grav_val
        * r_sq
        * gradr_val
        / p_val
    )

    def _q_of_z(z: jax.Array) -> jax.Array:
        return _eval_q_and_af_given_gamma(
            jnp.exp(z), gamma0, lum0, c0, l0, hp_val, cp_val, t_val
        )[0]

    def _q_of_y_gr(y: jax.Array, gr: jax.Array) -> jax.Array:
        return _eval_q_and_af_jax(
            y, gr, t_val, rho_val, p_val, hp_val, cp_val, grav_val
        )[0]

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
        20,
        _newton_step,
        (z_init, lb, ub, jnp.bool_(False), jnp.bool_(False)),
    )

    y_star = jnp.exp(z_star)
    _, af_star = _eval_q_and_af_given_gamma(
        y_star, gamma0, lum0, c0, l0, hp_val, cp_val, t_val
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
