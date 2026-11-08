# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differentiable JAX (`float64`) kernels for binary orbital geometry and
explicit/implicit Roche-lobe overflow (`flexmesa.binary`).

Corresponds to `binary/private/binary_utils.f90` and `binary_mdot.f90`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants as const


@jax.jit
def eval_rlobe_jax(
    m1: jax.Array | float,
    m2: jax.Array | float,
    a: jax.Array | float,
) -> jax.Array:
    """Evaluate Eggleton (1983) Roche-lobe radius $R_{L,1}$ (cm).

    Args:
        m1: Primary mass (g).
        m2: Companion mass (g).
        a: Orbital separation (cm).

    Returns:
        Roche-lobe radius $R_{L,1}$ (cm).
    """
    m1_f = jnp.asarray(m1, dtype=jnp.float64)
    m2_f = jnp.asarray(m2, dtype=jnp.float64)
    a_f = jnp.asarray(a, dtype=jnp.float64)
    q = jnp.power(m1_f / m2_f, const.ONE_THIRD)
    q2 = q * q
    return a_f * 0.49 * q2 / (0.6 * q2 + jnp.log1p(q))


@jax.jit
def eval_rl_relative_gap_jax(
    r: jax.Array | float,
    rl: jax.Array | float,
    eccentricity: jax.Array | float = 0.0,
) -> jax.Array:
    """Evaluate periastron relative Roche-lobe gap $(R - R_{L,p}) / R_{L,p}$.

    Args:
        r: Stellar photospheric radius (cm).
        rl: Circular Roche-lobe radius (cm).
        eccentricity: Orbital eccentricity $e \\in [0, 1)$.

    Returns:
        Dimensionless relative Roche-lobe gap.
    """
    r_f = jnp.asarray(r, dtype=jnp.float64)
    rl_peri = jnp.asarray(rl, dtype=jnp.float64) * (
        1.0 - jnp.asarray(eccentricity, dtype=jnp.float64)
    )
    return (r_f - rl_peri) / rl_peri


@jax.jit
def separation_from_period_jax(
    m1: jax.Array | float,
    m2: jax.Array | float,
    period: jax.Array | float,
) -> jax.Array:
    """Compute orbital separation $a$ (cm) from period $P$ (s) and masses (g).

    Args:
        m1: Primary mass (g).
        m2: Companion mass (g).
        period: Orbital period (s).

    Returns:
        Orbital separation $a$ (cm).
    """
    p_over_2pi = jnp.asarray(period, dtype=jnp.float64) / (2.0 * const.PI)
    gm = const.STANDARD_CGRAV * (
        jnp.asarray(m1, dtype=jnp.float64) + jnp.asarray(m2, dtype=jnp.float64)
    )
    return jnp.power(gm * p_over_2pi * p_over_2pi, const.ONE_THIRD)


@jax.jit
def period_from_separation_jax(
    m1: jax.Array | float,
    m2: jax.Array | float,
    separation: jax.Array | float,
) -> jax.Array:
    """Compute orbital period $P$ (s) from separation $a$ (cm) and masses (g).

    Args:
        m1: Primary mass (g).
        m2: Companion mass (g).
        separation: Orbital separation (cm).

    Returns:
        Orbital period $P$ (s).
    """
    a_f = jnp.asarray(separation, dtype=jnp.float64)
    gm = const.STANDARD_CGRAV * (
        jnp.asarray(m1, dtype=jnp.float64) + jnp.asarray(m2, dtype=jnp.float64)
    )
    return 2.0 * const.PI * jnp.sqrt((a_f * a_f * a_f) / gm)


@jax.jit
def eval_j_orb_jax(
    m1: jax.Array | float,
    m2: jax.Array | float,
    separation: jax.Array | float,
    eccentricity: jax.Array | float = 0.0,
) -> jax.Array:
    """Compute orbital angular momentum $J_{\\mathrm{orb}}$ in CGS units.

    Args:
        m1: Primary mass (g).
        m2: Companion mass (g).
        separation: Orbital separation (cm).
        eccentricity: Orbital eccentricity $e$.

    Returns:
        Orbital angular momentum $J_{\\mathrm{orb}}$ (`g cm^2 s^-1`).
    """
    m1_f = jnp.asarray(m1, dtype=jnp.float64)
    m2_f = jnp.asarray(m2, dtype=jnp.float64)
    a_f = jnp.asarray(separation, dtype=jnp.float64)
    e_f = jnp.asarray(eccentricity, dtype=jnp.float64)
    factor = const.STANDARD_CGRAV * a_f * (1.0 - e_f * e_f) / (m1_f + m2_f)
    return m1_f * m2_f * jnp.sqrt(factor)


@jax.jit
def separation_from_j_orb_jax(
    m1: jax.Array | float,
    m2: jax.Array | float,
    j_orb: jax.Array | float,
    eccentricity: jax.Array | float = 0.0,
) -> jax.Array:
    """Compute orbital separation $a$ (cm) from $J_{\\mathrm{orb}}$.

    Args:
        m1: Primary mass (g).
        m2: Companion mass (g).
        j_orb: Orbital angular momentum ($\\mathrm{g\\,cm^2\\,s^{-1}}$).
        eccentricity: Orbital eccentricity $e$.

    Returns:
        Orbital separation $a$ (cm).
    """
    m1_f = jnp.asarray(m1, dtype=jnp.float64)
    m2_f = jnp.asarray(m2, dtype=jnp.float64)
    j_f = jnp.asarray(j_orb, dtype=jnp.float64)
    e_f = jnp.asarray(eccentricity, dtype=jnp.float64)
    j_over_m1m2 = j_f / (m1_f * m2_f)
    return (
        j_over_m1m2
        * j_over_m1m2
        * (m1_f + m2_f)
        / (const.STANDARD_CGRAV * (1.0 - e_f * e_f))
    )


@jax.jit
def eval_ritter_mdot_thin_jax(
    m_donor: jax.Array | float,
    m_accretor: jax.Array | float,
    r_donor: jax.Array | float,
    rl_donor: jax.Array | float,
    rho_surf: jax.Array | float,
    p_surf: jax.Array | float,
    t_surf: jax.Array | float,
    mu_surf: jax.Array | float,
    is_kolb_scheme: bool = False,
) -> jax.Array:
    """Evaluate Ritter (1988) optically thin RLOF rate ($\\mathrm{g\\,s^{-1}}$).

    Args:
        m_donor: Donor mass (g).
        m_accretor: Accretor mass (g).
        r_donor: Donor photospheric radius (cm).
        rl_donor: Donor Roche-lobe radius (cm).
        rho_surf: Surface cell density ($\\mathrm{g\\,cm^{-3}}$).
        p_surf: Surface cell pressure ($\\mathrm{dyn\\,cm^{-2}}$).
        t_surf: Surface cell temperature (K).
        mu_surf: Surface mean molecular weight.
        is_kolb_scheme: Cap exponential factor at unity when $R_d > R_{L,d}$.

    Returns:
        Optically thin rate $\\dot{M}_{\\mathrm{thin}} \\le 0$ (`g s^-1`).
    """
    md = jnp.asarray(m_donor, dtype=jnp.float64)
    ma = jnp.asarray(m_accretor, dtype=jnp.float64)
    rd = jnp.asarray(r_donor, dtype=jnp.float64)
    rld = jnp.asarray(rl_donor, dtype=jnp.float64)
    rho = jnp.asarray(rho_surf, dtype=jnp.float64)
    p = jnp.asarray(p_surf, dtype=jnp.float64)
    t = jnp.asarray(t_surf, dtype=jnp.float64)
    mu = jnp.asarray(mu_surf, dtype=jnp.float64)

    grav = const.STANDARD_CGRAV * md / (rd * rd)
    hp = p / (grav * rho)
    v_th = jnp.sqrt(const.KERG * t / (const.MP * mu))
    q = ma / md
    f1 = 1.23 + 0.5 * (jnp.log(jnp.clip(q, 0.5, 10.0)) / const.LN10)
    mdot0 = (
        (2.0 * const.PI / jnp.exp(0.5))
        * (v_th * v_th * v_th)
        * (rld * rld * rld)
        / (const.STANDARD_CGRAV * md)
        * rho
        * f1
    )

    q2 = jnp.clip(q, 0.04, 20.0)
    log_q2 = jnp.log(q2) / const.LN10
    gamma_q = jnp.where(
        q2 < 1.0,
        0.954 + 0.025 * log_q2 - 0.038 * log_q2 * log_q2,
        0.954 + 0.039 * log_q2 + 0.114 * log_q2 * log_q2,
    )
    ritter_exp = (rd - rld) / (hp / gamma_q)
    use_cap = jnp.logical_and(is_kolb_scheme, ritter_exp > 0.0)
    return jnp.where(
        use_cap,
        -mdot0,
        -mdot0 * jnp.exp(jnp.minimum(ritter_exp, 60.0)),
    )


def _rlof_residual(
    mdot: jax.Array,
    m_donor: jax.Array,
    m_accretor: jax.Array,
    r_donor: jax.Array,
    separation: jax.Array,
    dt: jax.Array,
) -> jax.Array:
    """Implicit RLOF residual $R_d(\\dot{M}) - R_{L,d}(\\dot{M})$."""
    dm = mdot * dt
    m1_new = jnp.maximum(m_donor + dm, 0.05 * const.MSUN)
    m2_new = jnp.maximum(m_accretor - dm, 0.05 * const.MSUN)
    r_new = r_donor * jnp.sqrt(m1_new / m_donor)
    rl_new = eval_rlobe_jax(m1_new, m2_new, separation)
    return jnp.asarray(r_new - rl_new)


@jax.custom_vjp
@jax.jit
def solve_roche_lobe_mdot_jax(
    m_donor: jax.Array | float,
    m_accretor: jax.Array | float,
    r_donor: jax.Array | float,
    separation: jax.Array | float,
    dt: jax.Array | float,
) -> jax.Array:
    """Solve implicit Roche-lobe overflow $R_d(\\dot{M}) = R_{L,d}(\\dot{M})$
    with an analytical Implicit Function Theorem `custom_vjp` rule.

    Args:
        m_donor: Donor mass (g).
        m_accretor: Accretor mass (g).
        r_donor: Initial donor photospheric radius (cm).
        separation: Orbital separation (cm).
        dt: Timestep duration (s).

    Returns:
        Implicit mass transfer rate $\\dot{M}_{\\mathrm{tr}} \\le 0$ (`g s^-1`).
    """
    md = jnp.asarray(m_donor, dtype=jnp.float64)
    ma = jnp.asarray(m_accretor, dtype=jnp.float64)
    rd = jnp.asarray(r_donor, dtype=jnp.float64)
    a = jnp.asarray(separation, dtype=jnp.float64)
    dt_f = jnp.asarray(dt, dtype=jnp.float64)

    f0 = _rlof_residual(jnp.float64(0.0), md, ma, rd, a, dt_f)

    def _newton_step(_i: int, mdot_cur: jax.Array) -> jax.Array:
        res, dres = jax.value_and_grad(_rlof_residual, argnums=0)(
            mdot_cur, md, ma, rd, a, dt_f
        )
        return jnp.minimum(mdot_cur - res / dres, 0.0)

    mdot_star = jax.lax.fori_loop(0, 12, _newton_step, jnp.float64(0.0))
    return jnp.where(f0 > 0.0, mdot_star, jnp.float64(0.0))


def _solve_rlof_fwd(
    m_donor: jax.Array,
    m_accretor: jax.Array,
    r_donor: jax.Array,
    separation: jax.Array,
    dt: jax.Array,
) -> tuple[
    jax.Array,
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
]:
    md = jnp.asarray(m_donor, dtype=jnp.float64)
    ma = jnp.asarray(m_accretor, dtype=jnp.float64)
    rd = jnp.asarray(r_donor, dtype=jnp.float64)
    a = jnp.asarray(separation, dtype=jnp.float64)
    dt_f = jnp.asarray(dt, dtype=jnp.float64)
    mdot_star = solve_roche_lobe_mdot_jax(md, ma, rd, a, dt_f)
    return mdot_star, (mdot_star, md, ma, rd, a, dt_f)


def _solve_rlof_bwd(
    residuals: tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
    ],
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    mdot_star, md, ma, rd, a, dt_f = residuals
    df_dmdot, df_dmd, df_dma, df_drd, df_da, df_ddt = jax.grad(
        _rlof_residual, argnums=(0, 1, 2, 3, 4, 5)
    )(mdot_star, md, ma, rd, a, dt_f)
    active = mdot_star < 0.0
    adj = jnp.where(active, -g / df_dmdot, 0.0)
    return (
        adj * df_dmd,
        adj * df_dma,
        adj * df_drd,
        adj * df_da,
        adj * df_ddt,
    )


solve_roche_lobe_mdot_jax.defvjp(_solve_rlof_fwd, _solve_rlof_bwd)
