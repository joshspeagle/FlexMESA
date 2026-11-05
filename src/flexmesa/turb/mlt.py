# SPDX-License-Identifier: LGPL-3.0-or-later
"""Local Mixing-Length Theory (`set_mlt`, `turb/private/mlt.f90`).

Solves the cubic convective efficiency equation (Cox & Giuli 1968, §14.82)
$$a_0 \\Gamma^3 + \\Gamma^2 + \\Gamma - a_0 B^3 = 0$$
across six geometric MLT prescriptions (`Cox`, `TDC`, `Henyey`, `ML1`, `ML2`,
and `Mihalas`), supporting IEEE-754 `float`, forward-mode `DualOrder1`, and
vectorized/differentiable JAX `float64` arrays (`set_mlt_jax`).

All inputs and outputs use CGS physical units (`cm`, `g`, `s`, `erg`, `K`,
`dyn`) and dimensionless logarithmic temperature gradients $d\\ln T / d\\ln P$.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants as _const
from flexmesa import utils as _utils
from flexmesa.turb import types as _types


def set_mlt(
    params: _types.MltParams,
    inp: _types.MltInputs,
) -> _types.TurbOutput:
    """Evaluate local Mixing-Length Theory (`calc_MLT` in `mlt.f90`).

    Args:
        params: Non-differentiable MLT configuration parameters (`MltParams`).
        inp: Thermodynamic and structure inputs (`MltInputs`) with `float` or
            `DualOrder1` fields in CGS units.

    Returns:
        `TurbOutput` containing `(gamma, grad_t, y_face, conv_vel, d,
        mixing_type)`.

    Raises:
        ValueError: If an intermediate quantity (`Zeta`) is `NaN` or `Inf`.
    """
    if (
        params.mixing_length_alpha <= 0.0
        or _utils.is_bad(params.mixing_length_alpha)
        or _types.val_of(inp.t) <= 0.0
        or _types.val_of(inp.rho) <= 0.0
        or _types.val_of(inp.p) <= 0.0
        or _types.val_of(inp.cp) <= 0.0
        or _types.val_of(inp.opacity) <= 0.0
        or _utils.is_bad(_types.val_of(inp.gradr))
        or _utils.is_bad(_types.val_of(inp.gradl))
        or _utils.is_bad(_types.val_of(inp.grada))
    ):
        msg = "Invalid or non-positive thermodynamic input in set_mlt"
        raise ValueError(msg)
    proto = inp.gradr
    zero = _types.const_like(proto, 0.0)
    one = _types.const_like(proto, 1.0)

    if _types.val_of(inp.gradr) > _types.val_of(inp.gradl):
        q = inp.chi_t / inp.chi_rho
        if params.option in (_types.MltOption.COX, _types.MltOption.TDC):
            a0 = _types.const_like(proto, 9.0 / 4.0)
            conv_cond = (
                inp.cp
                * inp.grav
                * _types.s_pow2(inp.lambda_)
                * inp.rho
                * _types.s_sqrt(q * inp.rho / (2.0 * inp.p))
                / 9.0
            )
            rad_cond = (
                (4.0 / 3.0 * _const.CRAD * _const.CLIGHT)
                * _types.s_pow3(inp.t)
                / (inp.opacity * inp.rho)
            )
            a_val = conv_cond / rad_cond
        else:
            if params.option == _types.MltOption.HENYEY:
                ff1 = 1.0 / params.henyey_nu
                ff2 = 0.5
                ff3 = 8.0 / params.henyey_y
                ff4 = 1.0 / params.henyey_y
            elif params.option == _types.MltOption.ML1:
                ff1, ff2, ff3, ff4 = 0.125, 0.5, 24.0, 0.0
            elif params.option == _types.MltOption.ML2:
                ff1, ff2, ff3, ff4 = 1.0, 2.0, 16.0, 0.0
            else:
                ff1, ff2, ff3, ff4 = 0.125, 0.5, 16.0, 2.0

            omega = inp.lambda_ * inp.rho * inp.opacity
            ff4_omega2_plus_1 = ff4 / _types.s_pow2(omega) + 1.0
            a0 = ((3.0 / 16.0) * ff2 * ff3) / ff4_omega2_plus_1
            a_1 = 4.0 * inp.cp * _types.s_sqrt(ff1 * inp.p * q * inp.rho)
            a_2 = params.mixing_length_alpha * omega * ff4_omega2_plus_1
            a_num = a_1 * a_2
            a_den = (ff3 * _const.CRAD * _const.CLIGHT) * _types.s_pow3(inp.t)
            a_val = a_num / a_den

        b_cubed = (_types.s_pow2(a_val) / a0) * (inp.gradr - inp.gradl)
        delta = a0 * b_cubed
        f = -2.0 + 9.0 * a0 + 27.0 * a0 * a0 * delta
        if _types.val_of(f) > 1.0e100:
            f0 = f
        else:
            term = -1.0 + 3.0 * a0
            f0_sq = _types.s_pow2(f) + 4.0 * term * term * term
            f0 = f if _types.val_of(f0_sq) <= 0.0 else _types.s_sqrt(f0_sq)

        # Cardano depressed cubic root for a0 * Gamma^3 + Gamma^2 + Gamma
        # - a0 * B^3 = 0 (`set_MLT` in `turb/private/mlt.f90`).
        f1_arg = f + f0
        if _types.val_of(f1_arg) <= 0.0:
            return _types.TurbOutput(
                gamma=zero,
                grad_t=zero,
                y_face=zero,
                conv_vel=zero,
                d=zero,
                mixing_type=_const.MixingType.NO_MIXING,
            )
        f1 = _types.s_powf(f1_arg, _const.ONE_THIRD)
        f2 = (2.0 * _const.TWO_13) * (1.0 - 3.0 * a0) / f1
        gamma = (_const.FOUR_13 * f1 + f2 - 2.0) / (6.0 * a0)
        if _types.val_of(gamma) <= 0.0:
            return _types.TurbOutput(
                gamma=gamma,
                grad_t=zero,
                y_face=zero,
                conv_vel=zero,
                d=zero,
                mixing_type=_const.MixingType.NO_MIXING,
            )

        conv_vel = (
            params.mixing_length_alpha
            * _types.s_sqrt(q * inp.p / (8.0 * inp.rho))
            * gamma
            / a_val
        )
        if _types.val_of(conv_vel) > params.max_conv_vel:
            conv_vel = _types.with_val_like(conv_vel, params.max_conv_vel)
        d = conv_vel * inp.lambda_ / 3.0
        zeta = _types.s_exp(3.0 * _types.s_log(gamma) - _types.s_log(b_cubed))
    else:
        gamma = _types.const_like(proto, -1.0e99)
        zeta = zero
        conv_vel = zero
        d = zero

    if _utils.is_bad(_types.val_of(zeta)):
        msg = "NaN or Inf encountered in MLT Zeta"
        raise ValueError(msg)
    if _types.val_of(zeta) > 1.0:
        zeta = one

    grad_t = (1.0 - zeta) * inp.gradr + zeta * inp.gradl
    y_face = grad_t - inp.gradl
    mixing_type = (
        _const.MixingType.CONVECTIVE_MIXING
        if _types.val_of(y_face) > 0.0
        else _const.MixingType.NO_MIXING
    )
    return _types.TurbOutput(
        gamma=gamma,
        grad_t=grad_t,
        y_face=y_face,
        conv_vel=conv_vel,
        d=d,
        mixing_type=mixing_type,
    )


def set_mlt_jax(
    chi_t: jax.Array,
    chi_rho: jax.Array,
    cp: jax.Array,
    grav: jax.Array,
    lambda_: jax.Array,
    rho: jax.Array,
    p: jax.Array,
    t: jax.Array,
    opacity: jax.Array,
    gradr: jax.Array,
    grada: jax.Array,
    gradl: jax.Array,
    mixing_length_alpha: float = 2.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate Cox Mixing-Length Theory in pure JAX `float64`.

    Compatible with `jax.jit`, `jax.vmap`, and `jax.grad`. Applies safe
    `jnp.where` masking before logarithmic and power operations so gradients
    remain finite across radiative ($\\nabla_r \\le \\nabla_L$) and convective
    ($\\nabla_r > \\nabla_L$) zones.

    Args:
        chi_t: $(\\partial \\ln P / \\partial \\ln T)_\\rho$, shape `(*batch,)`.
        chi_rho: $(\\partial \\ln P / \\partial \\ln \\rho)_T$, shape
            `(*batch,)`.
        cp: Specific heat $C_P$ in $\\mathrm{erg\\,g^{-1}\\,K^{-1}}$, shape
            `(*batch,)`.
        grav: Gravitational acceleration $g$ in $\\mathrm{cm\\,s^{-2}}$, shape
            `(*batch,)`.
        lambda_: Mixing length $\\Lambda$ in $\\mathrm{cm}$, shape `(*batch,)`.
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$, shape `(*batch,)`.
        p: Total pressure $P$ in $\\mathrm{dyn\\,cm^{-2}}$, shape `(*batch,)`.
        t: Temperature $T$ in $\\mathrm{K}$, shape `(*batch,)`.
        opacity: Rosseland opacity $\\kappa$ in $\\mathrm{cm^2\\,g^{-1}}$, shape
            `(*batch,)`.
        gradr: Radiative gradient $\\nabla_r$, shape `(*batch,)`.
        grada: Adiabatic gradient $\\nabla_{\\mathrm{ad}}$, shape `(*batch,)`.
        gradl: Ledoux gradient $\\nabla_L$, shape `(*batch,)`.
        mixing_length_alpha: Mixing-length parameter $\\alpha_{\\mathrm{MLT}}$.

    Returns:
        Five-tuple `(gamma, grad_t, y_face, conv_vel, d)` of `float64` JAX
        arrays with shape `(*batch,)`.
    """
    del grada
    chi_t64 = jnp.asarray(chi_t, dtype=jnp.float64)
    chi_rho64 = jnp.asarray(chi_rho, dtype=jnp.float64)
    cp64 = jnp.asarray(cp, dtype=jnp.float64)
    grav64 = jnp.asarray(grav, dtype=jnp.float64)
    lam64 = jnp.asarray(lambda_, dtype=jnp.float64)
    rho64 = jnp.asarray(rho, dtype=jnp.float64)
    p64 = jnp.asarray(p, dtype=jnp.float64)
    t64 = jnp.asarray(t, dtype=jnp.float64)
    kap64 = jnp.asarray(opacity, dtype=jnp.float64)
    gradr64 = jnp.asarray(gradr, dtype=jnp.float64)
    gradl64 = jnp.asarray(gradl, dtype=jnp.float64)

    unstable = gradr64 > gradl64
    dgrad_safe = jnp.where(unstable, gradr64 - gradl64, 1.0e-30)

    q = chi_t64 / chi_rho64
    a0 = jnp.float64(9.0 / 4.0)
    conv_cond = (
        cp64
        * grav64
        * (lam64 * lam64)
        * rho64
        * jnp.sqrt(q * rho64 / (2.0 * p64))
        / 9.0
    )
    rad_cond = (
        (4.0 / 3.0 * _const.CRAD * _const.CLIGHT)
        * (t64 * t64 * t64)
        / (kap64 * rho64)
    )
    a_val = conv_cond / rad_cond
    b_cubed = ((a_val * a_val) / a0) * dgrad_safe
    delta = a0 * b_cubed
    f = -2.0 + 9.0 * a0 + 27.0 * a0 * a0 * delta
    term = -1.0 + 3.0 * a0
    f0_sq = f * f + 4.0 * term * term * term
    f0 = jnp.where(
        f > 1.0e100,
        f,
        jnp.where(f0_sq <= 0.0, f, jnp.sqrt(jnp.maximum(f0_sq, 0.0))),
    )
    # Cardano cubic root with safe floor on positive convective branch.
    f1_arg = jnp.maximum(f + f0, 1.0e-99)
    f1 = jnp.power(f1_arg, _const.ONE_THIRD)
    f2 = (2.0 * _const.TWO_13) * (1.0 - 3.0 * a0) / f1
    gamma_conv = (_const.FOUR_13 * f1 + f2 - 2.0) / (6.0 * a0)
    gamma_pos = jnp.maximum(gamma_conv, 1.0e-99)

    vc_conv = (
        mixing_length_alpha
        * jnp.sqrt(q * p64 / (8.0 * rho64))
        * gamma_pos
        / a_val
    )
    d_conv = vc_conv * lam64 / 3.0
    zeta_conv = jnp.minimum(
        jnp.exp(3.0 * jnp.log(gamma_pos) - jnp.log(b_cubed)),
        1.0,
    )

    gamma = jnp.where(unstable, gamma_conv, -1.0e99)
    zeta = jnp.where(unstable, zeta_conv, 0.0)
    conv_vel = jnp.where(unstable, vc_conv, 0.0)
    d = jnp.where(unstable, d_conv, 0.0)
    grad_t = (1.0 - zeta) * gradr64 + zeta * gradl64
    y_face = grad_t - gradl64
    return gamma, grad_t, y_face, conv_vel, d
