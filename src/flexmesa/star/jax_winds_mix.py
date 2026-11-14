# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differentiable JAX kernels for stellar winds, Lagrangian mixing coupling
(`sig`), and conservative species diffusion (`flexmesa.star`, Wave 4.3).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants

_FOUR_PI = float(constants.FOUR_PI)
_L_SUN = float(constants.L_SUN)
_M_SUN = float(constants.M_SUN)
_R_SUN = float(constants.R_SUN)
_SECYER = float(constants.SECYER)
_CRAD = float(constants.CRAD)
_CLIGHT = float(constants.CLIGHT)
_PI = float(constants.PI)
_STANDARD_CGRAV = float(constants.STANDARD_CGRAV)


@jax.jit
def eval_convection_sigmas_jax(
    r: jax.Array,
    rho: jax.Array,
    dq: jax.Array,
    d_mix: jax.Array,
    xmstar: float,
    dt: float,
    mix_factor: float = 1.0,
    sig_term_limit: float = 1.0e10,
) -> jax.Array:
    """Compute Lagrangian face mixing coupling factors `sig` (`(nz,)`).

    Args:
        r: Face radii of shape `(nz,)` (`cm`).
        rho: Cell densities of shape `(nz,)` (`g cm^-3`).
        dq: Cell relative mass fractions of shape `(nz,)`.
        d_mix: Face diffusion coefficients of shape `(nz,)` (`cm^2 s^-1`).
        xmstar: Enveloped mass `M_* - M_center` in grams (`g`).
        dt: Timestep in seconds (`s`).
        mix_factor: Global diffusivity multiplier (`1.0` default).
        sig_term_limit: Stability limiter factor (`1e10` default).

    Returns:
        Face coupling factors `sig` of shape `(nz,)` (`g s^-1`, `sig[0] = 0`).
    """
    nz = r.shape[0]
    idx = jnp.arange(nz)
    dq_m1 = jnp.concatenate([dq[:1], dq[:-1]])
    rho_m1 = jnp.concatenate([rho[:1], rho[:-1]])
    denom_dq = jnp.maximum(dq_m1 + dq, 1.0e-99)
    rho_face = (dq_m1 * rho + dq * rho_m1) / denom_dq

    area_rho = _FOUR_PI * (r * r) * rho_face
    cdc = (area_rho * area_rho) * d_mix
    dm_bar = jnp.maximum(0.5 * xmstar * denom_dq, 1.0e-99)
    sig_raw = jnp.where(idx == 0, 0.0, mix_factor * cdc / dm_bar)

    max_sig = jnp.max(sig_raw)
    dq_min = jnp.minimum(dq, dq_m1)
    sig_lim = sig_term_limit * xmstar * dq_min / jnp.maximum(dt, 1.0e-30)
    sig_capped = jnp.minimum(sig_raw, sig_lim)
    return jnp.where(max_sig >= 1.0, sig_capped, sig_raw)


@jax.jit
def eval_dxdt_mix_jax(
    xa: jax.Array,
    sig: jax.Array,
    dm: jax.Array,
) -> jax.Array:
    """Evaluate explicit mass-conserving diffusive species mixing rates.

    Args:
        xa: Cell species mass fractions of shape `(nz, species)`.
        sig: Face Lagrangian coupling factors of shape `(nz,)` (`g s^-1`).
        dm: Cell masses of shape `(nz,)` (`g`).

    Returns:
        Diffusive rates `(dX/dt)_mix` of shape `(nz, species)` (`s^-1`).
    """
    nz = xa.shape[0]
    idx = jnp.arange(nz)[:, None]
    xa_m1 = jnp.concatenate([xa[:1], xa[:-1]], axis=0)
    xa_p1 = jnp.concatenate([xa[1:], xa[-1:]], axis=0)
    sig_00 = jnp.where(idx == 0, 0.0, sig[:, None])
    sig_p1 = jnp.concatenate([sig[1:], jnp.zeros((1,), dtype=sig.dtype)])[
        :, None
    ]

    flux_in = sig_00 * (xa_m1 - xa)
    flux_out = sig_p1 * (xa - xa_p1)
    return (flux_in - flux_out) / jnp.maximum(dm[:, None], 1.0e-99)


@jax.jit
def eval_winds_jax(
    l_surf: float | jax.Array,
    m_surf: float | jax.Array,
    t_surf: float | jax.Array,
    x_surf: float | jax.Array = 0.70,
    y_surf: float | jax.Array = 0.28,
    z_surf: float | jax.Array = 0.02,
    eta: float = 1.0,
) -> dict[str, jax.Array]:
    """Evaluate differentiable wind mass-loss rates (`M_sun yr^-1`, `g s^-1`).

    Args:
        l_surf: Surface luminosity in `erg s^-1`, scalar or shape `(*batch,)`.
        m_surf: Stellar mass in grams (`g`), scalar or shape `(*batch,)`.
        t_surf: Effective temperature in Kelvin (`K`), scalar or shape
            `(*batch,)`.
        x_surf: Surface hydrogen mass fraction `X`, scalar or shape
            `(*batch,)`.
        y_surf: Surface helium mass fraction `Y`, scalar or shape `(*batch,)`.
        z_surf: Surface metal mass fraction `Z`, scalar or shape `(*batch,)`.
        eta: Wind scaling multiplier (`1.0` default).

    Returns:
        Dictionary of arrays with shape `(*batch,)` (matching broadcast
        inputs) containing `w_reimers`, `w_blocker`, `w_vink`, `w_bjorklund`,
        `w_nieuwenhuijzen`, `w_de_jager`, `w_van_loon`, `w_dutch` in
        `M_sun yr^-1` and `mdot_dutch` in `g s^-1`.
    """
    l_s = jnp.asarray(l_surf, dtype=jnp.float64)
    m_s = jnp.asarray(m_surf, dtype=jnp.float64)
    t_s = jnp.asarray(t_surf, dtype=jnp.float64)
    x_s = jnp.asarray(x_surf, dtype=jnp.float64)
    y_s = jnp.asarray(y_surf, dtype=jnp.float64)
    z_s = jnp.asarray(z_surf, dtype=jnp.float64)

    r_s = jnp.sqrt(l_s / (_PI * _CRAD * _CLIGHT * (t_s**4)))
    l_lsun = l_s / _L_SUN
    m_msun = m_s / _M_SUN
    r_rsun = r_s / _R_SUN

    w_reimers_1 = 4.0e-13 * l_lsun * r_rsun / m_msun
    w_reimers = eta * w_reimers_1
    w_blocker = eta * 4.83e-9 * (m_msun ** (-2.1)) * (l_lsun**2.7) * w_reimers_1

    z_rat = jnp.maximum(z_s / 0.019, 1.0e-10)
    log_z_rat = jnp.log10(z_rat)
    t_jump = 1.0e3 * (61.2 + 2.59 * (-13.636 + 0.889 * log_z_rat))
    d_t = 100.0
    alpha_vink = jnp.clip((t_s - (t_jump - d_t)) / (2.0 * d_t), 0.0, 1.0)
    alpha_vink = jnp.where(t_s > 27_500.0, 1.0, alpha_vink)
    alpha_vink = jnp.where(t_s < 22_500.0, 0.0, alpha_vink)

    log_l_1e5 = jnp.log10(l_lsun / 1.0e5)
    log_m_30 = jnp.log10(m_msun / 30.0)
    vinf_hot = 2.6 * (z_rat**0.13)
    log_t_4e4 = jnp.log10(t_s / 4.0e4)
    log_w1 = (
        -6.697
        + 2.194 * log_l_1e5
        - 1.313 * log_m_30
        - 1.226 * jnp.log10(vinf_hot / 2.0)
        + 0.933 * log_t_4e4
        - 10.92 * (log_t_4e4**2)
        + 0.85 * log_z_rat
    )
    vinf_cool = 1.3 * (z_rat**0.13)
    log_t_2e4 = jnp.log10(t_s / 2.0e4)
    log_w2 = (
        -6.688
        + 2.210 * log_l_1e5
        - 1.339 * log_m_30
        - 1.601 * jnp.log10(vinf_cool / 2.0)
        + 1.07 * log_t_2e4
        + 0.85 * log_z_rat
    )
    w_vink_1 = alpha_vink * (10.0**log_w1) + (1.0 - alpha_vink) * (10.0**log_w2)
    w_vink = eta * w_vink_1

    w_nieuw = eta * (
        10.0
        ** (
            -14.02
            + 1.24 * jnp.log10(l_lsun)
            + 0.16 * jnp.log10(m_msun)
            + 0.81 * jnp.log10(r_rsun)
        )
    )
    w_de_jager_1 = 10.0 ** (
        1.769 * jnp.log10(l_lsun) - 1.676 * jnp.log10(t_s) - 8.158
    )
    w_de_jager = eta * w_de_jager_1
    w_van_loon = eta * (
        10.0
        ** (
            -5.65
            + 1.05 * jnp.log10(l_lsun / 1.0e4)
            - 6.3 * jnp.log10(t_s / 3500.0)
        )
    )

    z_s_safe = jnp.maximum(z_s, 1.0e-30)
    w_wr = 1.0e-11 * (l_lsun**1.29) * (y_s**1.7) * jnp.sqrt(z_s_safe)
    w_high = jnp.where(x_s < 0.4, w_wr, w_vink_1)
    alpha_dutch = jnp.clip((t_s - 10_000.0) / 1000.0, 0.0, 1.0)
    w_dutch = eta * (alpha_dutch * w_high + (1.0 - alpha_dutch) * w_de_jager_1)
    mdot_dutch = -w_dutch * _M_SUN / _SECYER

    gamma_e = 0.34 * l_s / (_FOUR_PI * _CLIGHT * _STANDARD_CGRAV * m_s)
    m_eff = jnp.maximum(m_s * (1.0 - gamma_e), 1.0e-6 * m_s)
    log_t_45k = jnp.log10(t_s / 4.5e4)
    log_z_bj = jnp.log10(jnp.maximum(z_s / 0.014, 1.0e-10))
    log_w_bjork = (
        -5.52
        + 2.39 * jnp.log10(l_lsun / 1.0e6)
        - 1.48 * jnp.log10((m_eff / _M_SUN) / 45.0)
        + 2.12 * log_t_45k
        + (0.75 - 1.87 * log_t_45k) * log_z_bj
    )
    w_bjorklund = eta * (10.0**log_w_bjork)

    return {
        "w_reimers": w_reimers,
        "w_blocker": w_blocker,
        "w_vink": w_vink,
        "w_bjorklund": w_bjorklund,
        "w_nieuwenhuijzen": w_nieuw,
        "w_de_jager": w_de_jager,
        "w_van_loon": w_van_loon,
        "w_dutch": w_dutch,
        "mdot_dutch": mdot_dutch,
    }
