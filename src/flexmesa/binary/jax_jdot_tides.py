# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differentiable JAX (`float64`) kernels for Kerr black hole ISCO/Eddington
accretion, orbital angular momentum loss, and Hut (1981) tidal evolution.

Corresponds to `binary/private/binary_edot.f90`, `binary_jdot.f90`,
`binary_tides.f90`, and `binary_wind.f90`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants as const


@jax.jit
def eval_bh_spin_and_edd_jax(
    m_bh_init: jax.Array | float,
    a_spin_init: jax.Array | float,
    m_bh_current: jax.Array | float,
    kappa_surf: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate Kerr BH ISCO mass, spin, Eddington rate, and efficiency.

    Args:
        m_bh_init: Initial black hole mass (g).
        a_spin_init: Initial dimensionless Kerr spin $a \\in [0, 1]$.
        m_bh_current: Current black hole mass (g).
        kappa_surf: Donor surface opacity ($\\mathrm{cm^2\\,g^{-1}}$).

    Returns:
        Tuple `(m_bh_zero_eq, a_spin_current, mdot_edd, eta)`.
    """
    m_init = jnp.asarray(m_bh_init, dtype=jnp.float64)
    a_init = jnp.clip(jnp.asarray(a_spin_init, dtype=jnp.float64), 0.0, 1.0)
    m_cur = jnp.asarray(m_bh_current, dtype=jnp.float64)
    kap = jnp.asarray(kappa_surf, dtype=jnp.float64)

    one_m_a2 = jnp.maximum(1.0 - a_init * a_init, 1.0e-30)
    z1 = 1.0 + jnp.power(one_m_a2, const.ONE_THIRD) * (
        jnp.power(1.0 + a_init, const.ONE_THIRD)
        + jnp.power(jnp.maximum(1.0 - a_init, 1.0e-30), const.ONE_THIRD)
    )
    z2 = jnp.sqrt(3.0 * a_init * a_init + z1 * z1)
    r_isco = (
        3.0
        + z2
        - jnp.sqrt(jnp.maximum((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 1.0e-30))
    )
    m_eq = m_init * jnp.sqrt(r_isco / 6.0)

    m_eff = jnp.minimum(m_cur, jnp.sqrt(6.0) * m_eq)
    ratio = m_eq / m_eff
    a_cur = (
        jnp.sqrt(2.0 / 3.0)
        * ratio
        * (4.0 - jnp.sqrt(jnp.maximum(18.0 * ratio * ratio - 2.0, 1.0e-30)))
    )

    m_ratio = m_eff / (3.0 * m_eq)
    eta = jnp.maximum(
        1.0 - jnp.sqrt(jnp.maximum(1.0 - m_ratio * m_ratio, 1.0e-30)),
        1.0e-12,
    )
    mdot_edd = (
        4.0
        * const.PI
        * const.STANDARD_CGRAV
        * m_cur
        / (const.CLIGHT * kap * eta)
    )
    return m_eq, a_cur, mdot_edd, eta


@jax.jit
def eval_jdot_gr_jax(
    m_donor: jax.Array | float,
    m_accretor: jax.Array | float,
    separation: jax.Array | float,
    j_orb: jax.Array | float,
) -> jax.Array:
    """Evaluate gravitational wave torque $\\dot{J}_{\\mathrm{gr}}$ in CGS.

    Args:
        m_donor: Donor mass (g).
        m_accretor: Accretor mass (g).
        separation: Orbital separation (cm).
        j_orb: Orbital angular momentum ($\\mathrm{g\\,cm^2\\,s^{-1}}$).

    Returns:
        Gravitational wave torque $\\dot{J}_{\\mathrm{gr}}$ (`g cm^2 s^-2`).
    """
    md = jnp.asarray(m_donor, dtype=jnp.float64)
    ma = jnp.asarray(m_accretor, dtype=jnp.float64)
    a = jnp.asarray(separation, dtype=jnp.float64)
    j = jnp.asarray(j_orb, dtype=jnp.float64)
    a4 = (a * a) * (a * a)
    return (
        -(
            32.0
            * (const.STANDARD_CGRAV**3)
            * ma
            * md
            * (ma + md)
            / (5.0 * (const.CLIGHT**5) * a4)
        )
        * j
    )


@jax.jit
def eval_jdot_mb_jax(
    m_star: jax.Array | float,
    r_star: jax.Array | float,
    rl_star: jax.Array | float,
    period: jax.Array | float,
    gamma_mb: jax.Array | float = 3.0,
    q_conv_env: jax.Array | float = 0.05,
    scale_for_low_qconv_env: bool = True,
) -> jax.Array:
    """Evaluate Rappaport-Verbunt-Joss (1983) magnetic braking torque.

    Args:
        m_star: Stellar mass (g).
        r_star: Stellar radius (cm).
        rl_star: Roche-lobe radius (cm).
        period: Orbital period (s).
        gamma_mb: Magnetic braking index (default `3.0`).
        q_conv_env: Convective envelope mass fraction.
        scale_for_low_qconv_env: Damp when $q_{\\mathrm{conv,env}} < 0.02$.

    Returns:
        Magnetic braking torque $\\dot{J}_{\\mathrm{mb}}$ (`g cm^2 s^-2`).
    """
    m = jnp.asarray(m_star, dtype=jnp.float64)
    r_eff = jnp.minimum(
        jnp.asarray(r_star, dtype=jnp.float64),
        jnp.asarray(rl_star, dtype=jnp.float64),
    )
    omega = 2.0 * const.PI / jnp.asarray(period, dtype=jnp.float64)
    gam = jnp.asarray(gamma_mb, dtype=jnp.float64)
    q_env = jnp.asarray(q_conv_env, dtype=jnp.float64)
    base = (
        -3.8e-30
        * m
        * (const.RSUN**4)
        * jnp.power(r_eff / const.RSUN, gam)
        * (omega * omega * omega)
    )
    damped = base * jnp.exp(-0.02 / jnp.maximum(q_env, 1.0e-30) + 1.0)
    return jnp.where(
        jnp.logical_and(scale_for_low_qconv_env, q_env < 0.02), damped, base
    )


@jax.jit
def eval_edot_tidal_hut_jax(
    k_div_t: jax.Array | float,
    m_star: jax.Array | float,
    m_companion: jax.Array | float,
    r_phot: jax.Array | float,
    separation: jax.Array | float,
    eccentricity: jax.Array | float,
    omega_star: jax.Array | float,
    period: jax.Array | float,
    f_tid: jax.Array | float = 1.0,
) -> jax.Array:
    """Evaluate Hut (1981) tidal circularization rate (`s^-1`).

    Args:
        k_div_t: Tidal dissipation ratio $(k/T)$ ($\\mathrm{s}^{-1}$).
        m_star: Stellar mass (g).
        m_companion: Companion mass (g).
        r_phot: Photospheric radius (cm).
        separation: Orbital separation (cm).
        eccentricity: Orbital eccentricity $e$.
        omega_star: Surface spin angular velocity ($\\mathrm{rad\\,s^{-1}}$).
        period: Orbital period (s).
        f_tid: Tidal scaling factor (`Ftid`).

    Returns:
        Tidal eccentricity derivative $\\dot{e}_{\\mathrm{tidal}}$ (`s^-1`).
    """
    kt = jnp.asarray(k_div_t, dtype=jnp.float64)
    q = jnp.asarray(m_companion, dtype=jnp.float64) / jnp.asarray(
        m_star, dtype=jnp.float64
    )
    r_over_a = jnp.asarray(r_phot, dtype=jnp.float64) / jnp.asarray(
        separation, dtype=jnp.float64
    )
    e = jnp.asarray(eccentricity, dtype=jnp.float64)
    w = jnp.asarray(omega_star, dtype=jnp.float64)
    p = jnp.asarray(period, dtype=jnp.float64)
    ft = jnp.asarray(f_tid, dtype=jnp.float64)

    e2 = e * e
    e4 = e2 * e2
    f3 = 1.0 + 3.75 * e2 + 1.875 * e4 + (5.0 / 64.0) * e4 * e2
    f4 = 1.0 + 1.5 * e2 + 0.125 * e4
    one_m_e2 = jnp.maximum(1.0 - e2, 1.0e-12)
    omega_orb = 2.0 * const.PI / p
    bracket = f3 - (11.0 / 18.0) * (w / omega_orb) * f4 * jnp.power(
        one_m_e2, 1.5
    )
    raw = (
        -27.0
        * ft
        * kt
        * q
        * (1.0 + q)
        * jnp.power(r_over_a, 8.0)
        * e
        * jnp.power(one_m_e2, -6.5)
        * bracket
    )
    return jnp.where(e > 0.0, raw, 0.0)
