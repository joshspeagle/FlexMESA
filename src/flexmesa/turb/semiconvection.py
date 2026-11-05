# SPDX-License-Identifier: LGPL-3.0-or-later
"""Semiconvective mixing (`set_semiconvection`, `semiconvection.f90`).

Implements Langer, El Eid, & Fricke (1983, 1985) semiconvective mixing
diffusivity
$$D_{\\mathrm{sc}} = \\frac{\\alpha_{\\mathrm{sc}} K_r}{6 C_P \\rho}
\\frac{\\nabla_r - \\nabla_{\\mathrm{ad}}}{\\nabla_L - \\nabla_r}$$
and temperature-gradient prescriptions (`"Langer_85 mixing; gradT = gradr"`
and `"Langer_85"`).
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp

from flexmesa import constants as _const
from flexmesa.turb import types as _types


@dataclasses.dataclass(frozen=True)
class SemiconvectionParams:
    """Non-differentiable configuration parameters for `set_semiconvection`.

    Attributes:
        option: Semiconvection prescription (`SemiconvectionOption`).
        alpha_semiconvection: Efficiency parameter $\\alpha_{\\mathrm{sc}}$.
        cgrav: Gravitational constant $G$ in $\\mathrm{cm^3\\,g^{-1}\\,s^{-2}}$.
        gradl_composition_term: Ledoux composition term $B = \\nabla_L -
            \\nabla_{\\mathrm{ad}}$.
        m: Enclosed mass $m$ at the cell face in $\\mathrm{g}$.
    """

    option: _types.SemiconvectionOption = (
        _types.SemiconvectionOption.LANGER_85_GRADT_IS_GRADR
    )
    alpha_semiconvection: float = 0.1
    cgrav: float = _const.STANDARD_CGRAV
    gradl_composition_term: float = 0.0
    m: float = _const.MSUN


@dataclasses.dataclass(frozen=True)
class SemiconvectionInputs:
    """Differentiable thermodynamic inputs for `set_semiconvection`.

    Attributes:
        lum: Total luminosity $L$ across the face in $\\mathrm{erg\\,s^{-1}}$.
        lambda_: Mixing length $\\Lambda$ in $\\mathrm{cm}$.
        t: Temperature $T$ in $\\mathrm{K}$.
        p: Total pressure $P$ in $\\mathrm{dyn\\,cm^{-2}}$.
        pr: Radiation pressure $P_{\\mathrm{rad}}$ in $\\mathrm{dyn\\,cm^{-2}}$.
        beta: Gas-to-total pressure ratio $\\beta = P_{\\mathrm{gas}} / P$.
        opacity: Rosseland mean opacity $\\kappa$ in $\\mathrm{cm^2\\,g^{-1}}$.
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        cp: Specific heat $C_P$ in $\\mathrm{erg\\,g^{-1}\\,K^{-1}}$.
        gradr: Radiative temperature gradient $\\nabla_r$ (dimensionless).
        grada: Adiabatic temperature gradient $\\nabla_{\\mathrm{ad}}$.
        gradl: Ledoux temperature gradient $\\nabla_L$ (dimensionless).
    """

    lum: _types.ScalarOrDual
    lambda_: _types.ScalarOrDual
    t: _types.ScalarOrDual
    p: _types.ScalarOrDual
    pr: _types.ScalarOrDual
    beta: _types.ScalarOrDual
    opacity: _types.ScalarOrDual
    rho: _types.ScalarOrDual
    cp: _types.ScalarOrDual
    gradr: _types.ScalarOrDual
    grada: _types.ScalarOrDual
    gradl: _types.ScalarOrDual


def set_semiconvection(
    params: SemiconvectionParams,
    inp: SemiconvectionInputs,
) -> _types.TurbOutput:
    """Evaluate Langer (1983, 1985) semiconvective mixing.

    Args:
        params: Non-differentiable configuration (`SemiconvectionParams`).
        inp: Differentiable thermodynamic inputs (`SemiconvectionInputs`).

    Returns:
        `TurbOutput` containing `(gamma, grad_t, y_face, conv_vel, d,
        mixing_type)`.
    """
    proto = inp.gradr
    zero = _types.const_like(proto, 0.0)
    rad_cond = (
        (4.0 / 3.0 * _const.CRAD * _const.CLIGHT)
        * _types.s_pow3(inp.t)
        / (inp.opacity * inp.rho)
    )
    d = (
        params.alpha_semiconvection
        * rad_cond
        / (6.0 * inp.cp * inp.rho)
        * (inp.gradr - inp.grada)
        / (inp.gradl - inp.gradr)
    )
    d_val = _types.val_of(d)
    if math.isnan(d_val):
        msg = "NaN diffusivity in set_semiconvection"
        raise ValueError(msg)
    if d_val <= 0.0:
        return _types.TurbOutput(
            gamma=zero,
            grad_t=zero,
            y_face=zero,
            conv_vel=zero,
            d=d,
            mixing_type=_const.MixingType.NO_MIXING,
        )

    conv_vel = 3.0 * d / inp.lambda_
    if params.option == _types.SemiconvectionOption.LANGER_85_GRADT_IS_GRADR:
        grad_t = inp.gradr
        y_face = grad_t - inp.grada
        return _types.TurbOutput(
            gamma=zero,
            grad_t=grad_t,
            y_face=y_face,
            conv_vel=conv_vel,
            d=d,
            mixing_type=_const.MixingType.SEMICONVECTIVE_MIXING,
        )

    alpha = min(1.0, params.alpha_semiconvection)
    bc = 32.0 - 24.0 * inp.beta - inp.beta * inp.beta
    lg = (
        (16.0 * _const.PI * _const.CLIGHT * params.m * params.cgrav)
        * inp.pr
        / (inp.p * inp.opacity)
    )
    a0 = (alpha * params.gradl_composition_term) * lg
    a1 = -2.0 * bc * inp.lum
    a2 = (2.0 * alpha) * bc * inp.grada * lg
    a3 = -2.0 * bc * inp.gradl * lg
    b1 = 32.0 - 36.0 * inp.beta + 9.0 * inp.beta * inp.beta
    b2 = b1 * a0
    a = a1 + a2 + a3 + b2
    b3 = -2.0 * inp.gradl * inp.lum + alpha * inp.grada * inp.grada * lg
    b4 = (inp.gradl - alpha * inp.grada) * lg
    b5 = 2.0 * bc * (inp.lum + b4) - b2
    b6 = b2 * inp.grada + bc * b3
    b7 = (-4.0 * (alpha - 2.0)) * bc * lg * b6
    b = b7 + b5 * b5
    div = (2.0 * (alpha - 2.0)) * bc * lg
    bsq = _types.s_sqrt(b)
    grad_t = (a + bsq) / div
    y_face = grad_t - inp.grada
    return _types.TurbOutput(
        gamma=zero,
        grad_t=grad_t,
        y_face=y_face,
        conv_vel=conv_vel,
        d=d,
        mixing_type=_const.MixingType.SEMICONVECTIVE_MIXING,
    )


def set_semiconvection_jax(
    lambda_: jax.Array,
    t: jax.Array,
    opacity: jax.Array,
    rho: jax.Array,
    cp: jax.Array,
    gradr: jax.Array,
    grada: jax.Array,
    gradl: jax.Array,
    alpha_semiconvection: float = 0.1,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate Langer (1985) semiconvective diffusivity in pure JAX `float64`.

    Args:
        lambda_: Mixing length $\\Lambda$ in $\\mathrm{cm}$, shape `(*batch,)`.
        t: Temperature $T$ in $\\mathrm{K}$, shape `(*batch,)`.
        opacity: Rosseland opacity $\\kappa$ in $\\mathrm{cm^2\\,g^{-1}}$, shape
            `(*batch,)`.
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$, shape `(*batch,)`.
        cp: Specific heat $C_P$ in $\\mathrm{erg\\,g^{-1}\\,K^{-1}}$, shape
            `(*batch,)`.
        gradr: Radiative gradient $\\nabla_r$, shape `(*batch,)`.
        grada: Adiabatic gradient $\\nabla_{\\mathrm{ad}}$, shape `(*batch,)`.
        gradl: Ledoux gradient $\\nabla_L$, shape `(*batch,)`.
        alpha_semiconvection: Efficiency parameter $\\alpha_{\\mathrm{sc}}$.

    Returns:
        Four-tuple `(grad_t, y_face, conv_vel, d)` of `float64` JAX arrays.
    """
    lam64 = jnp.asarray(lambda_, dtype=jnp.float64)
    t64 = jnp.asarray(t, dtype=jnp.float64)
    kap64 = jnp.asarray(opacity, dtype=jnp.float64)
    rho64 = jnp.asarray(rho, dtype=jnp.float64)
    cp64 = jnp.asarray(cp, dtype=jnp.float64)
    gradr64 = jnp.asarray(gradr, dtype=jnp.float64)
    grada64 = jnp.asarray(grada, dtype=jnp.float64)
    gradl64 = jnp.asarray(gradl, dtype=jnp.float64)

    rad_cond = (
        (4.0 / 3.0 * _const.CRAD * _const.CLIGHT)
        * (t64 * t64 * t64)
        / (kap64 * rho64)
    )
    denom = jnp.where(gradl64 != gradr64, gradl64 - gradr64, 1.0e-30)
    d = (
        alpha_semiconvection
        * rad_cond
        / (6.0 * cp64 * rho64)
        * (gradr64 - grada64)
        / denom
    )
    active = d > 0.0
    grad_t = jnp.where(active, gradr64, 0.0)
    y_face = jnp.where(active, gradr64 - grada64, 0.0)
    conv_vel = jnp.where(active, 3.0 * d / lam64, 0.0)
    return grad_t, y_face, conv_vel, d
