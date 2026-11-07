# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branchless JAX Coulomb liquid/crystal and screening kernels for `PC` EOS.

Implements `EXCOR7`, `FITION9`, `FHARM12`, and `ANHARM8` from
`eos/private/pc_eos.f90` using branchless `jnp.where` primitives and
forward-mode automatic differentiation (`jax.jacfwd`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _excor7_fxc(theta: jax.Array, game: jax.Array) -> jax.Array:
    """Evaluate Tanaka & Ichimaru `EXCOR7` free energy $f_{\\mathrm{xc}}$."""
    sqth = jnp.sqrt(theta)
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta3 * theta
    th_safe = jnp.maximum(theta, 0.04)
    sq_safe = jnp.sqrt(th_safe)
    hi = theta > 0.04
    t1 = jnp.where(hi, jnp.tanh(1.0 / th_safe), 1.0)
    t2 = jnp.where(hi, jnp.tanh(1.0 / sq_safe), 1.0)

    a0 = 0.75 + 3.04363 * theta2 - 0.09227 * theta3 + 1.7035 * theta4
    a1 = 1.0 + 8.31051 * theta2 + 5.1105 * theta4
    a = 0.610887 * a0 / a1 * t1

    b0 = 0.341308 + 12.070873 * theta2 + 1.148889 * theta4
    b1 = 1.0 + 10.495346 * theta2 + 1.326623 * theta4
    b = sqth * t2 * b0 / b1

    d0 = 0.614925 + 16.996055 * theta2 + 1.489056 * theta4
    d1 = 1.0 + 10.10935 * theta2 + 1.22184 * theta4
    d = sqth * t2 * d0 / d1

    e0 = 0.539409 + 2.522206 * theta2 + 0.178484 * theta4
    e1 = 1.0 + 2.555501 * theta2 + 0.146319 * theta4
    e = theta * t1 * e0 / e1

    exp1th = jnp.exp(-1.0 / jnp.maximum(theta, 1.0e-3))
    c = (0.872496 + 0.025248 * exp1th) * e
    discr = jnp.sqrt(4.0 * e - d * d)

    s1 = -c / e * game
    b2 = b - c * d / e
    sqge = jnp.sqrt(game)
    s2 = -2.0 * b2 * sqge / e
    r3 = 1.0 + d * sqge + e * game
    b3 = a - c / e
    c3 = (d / e * b2 - b3) / e
    s3 = c3 * jnp.log(r3)
    b4 = 2.0 - d * d / e
    c4 = d + 2.0 * e * sqge
    s4a = 2.0 / (e * discr)
    s4b = d * b3 + b4 * b2
    s4c = jnp.arctan(c4 / discr) - jnp.arctan(d / discr)
    s4 = s4a * s4b * s4c
    return s1 + s2 + s3 + s4


def excor7_jax(rs: jax.Array, game: jax.Array) -> tuple[jax.Array, ...]:
    """Evaluate `EXCOR7(RS, GAME)` returning `(f, u, p, s, cv, pdt, pdr)`.

    Args:
        rs: Electron Wigner-Seitz parameter $r_s$.
        game: Electron Coulomb coupling parameter $\\Gamma_e$.

    Returns:
        7-tuple `(fxc, uxc, pxc, sxc, cvxc, pdtxc, pdrxc)`.
    """
    theta = rs * 0.543 / game
    fxc = _excor7_fxc(theta, game)
    grad_fn = jax.grad(_excor7_fxc, argnums=(0, 1))
    fxcdh, fxcdg = grad_fn(theta, game)
    (fxcdhh, fxcdhg), (_, fxcdgg) = jax.jacfwd(grad_fn, argnums=(0, 1))(
        theta, game
    )
    pxc = (game * fxcdg - 2.0 * theta * fxcdh) / 3.0
    uxc = game * fxcdg - theta * fxcdh
    sxc = uxc - fxc
    cvxc = (
        2.0 * theta * (game * fxcdhg - fxcdh)
        - theta * theta * fxcdhh
        - game * game * fxcdgg
    )
    pdlh = theta * (game * fxcdhg - 2.0 * fxcdh - 2.0 * theta * fxcdhh) / 3.0
    pdlg = game * (fxcdg + game * fxcdgg - 2.0 * theta * fxcdhg) / 3.0
    pdrxc = pxc + (pdlg - 2.0 * pdlh) / 3.0
    pdtxc = game * (theta * fxcdhg - game * fxcdgg / 3.0) - theta * (
        fxcdh / 0.75 + theta * fxcdhh / 1.5
    )
    return fxc, uxc, pxc, sxc, cvxc, pdtxc, pdrxc


def fition9_jax(gami: jax.Array) -> tuple[jax.Array, ...]:
    """Evaluate `FITION9(GAMI)` returning `(f, u, p, s, cv, pdt, pdr)`.

    Args:
        gami: Ion Coulomb coupling parameter $\\Gamma_i$.

    Returns:
        7-tuple `(f, u, p, s, cv, pdt, pdr)`.
    """
    a1 = -0.907347
    a2 = 0.62849
    c1 = 0.004500
    g1 = 170.0
    c2 = -8.4e-5
    g2 = 0.0037
    a3 = -0.8660254038 - a1 / jnp.sqrt(a2)
    sq_g = jnp.sqrt(gami)
    f0 = a1 * (
        jnp.sqrt(gami * (a2 + gami))
        - a2 * jnp.log(jnp.sqrt(gami / a2) + jnp.sqrt(1.0 + gami / a2))
    ) + 2.0 * a3 * (sq_g - jnp.arctan(sq_g))
    u0 = sq_g**3 * (a1 / jnp.sqrt(a2 + gami) + a3 / (1.0 + gami))
    gami2 = gami * gami
    u = u0 + c1 * gami2 / (g1 + gami) + c2 * gami2 / (g2 + gami2)
    f = (
        f0
        + c1 * (gami - g1 * jnp.log(1.0 + gami / g1))
        + 0.5 * c2 * jnp.log(1.0 + gami2 / g2)
    )
    cv = -0.5 * sq_g**3 * (
        a1 * a2 / jnp.sqrt(a2 + gami) ** 3
        + a3 * (1.0 - gami) / (1.0 + gami) ** 2
    ) - gami2 * (
        c1 * g1 / (g1 + gami) ** 2 + c2 * (g2 - gami2) / (g2 + gami2) ** 2
    )
    p = u / 3.0
    pdr = (4.0 * u - cv) / 9.0
    pdt = cv / 3.0
    return f, u, p, u - f, cv, pdt, pdr


def _hlfit_f_mid(eta: jax.Array) -> jax.Array:
    """Evaluate intermediate-regime bcc phonon free energy `f_mid(eta)`."""
    alpha, beta, gamma = 0.265764, 0.334547, 0.932446
    a1, a2, a3, a4, a6, a8 = (
        0.1839,
        0.593586,
        0.0054814,
        5.01813e-4,
        3.9247e-7,
        5.8356e-11,
    )
    b0, b2, b4, b5, b6, b7 = (
        261.66,
        7.07997,
        0.0409484,
        0.000397355,
        5.11148e-5,
        2.19749e-6,
    )
    b9, b11 = a6 * 0.004757014, a8 * 0.0047770935
    eta2 = eta * eta
    up = (
        eta
        * (eta * (eta * (eta * (eta2 * (eta2 * a8 + a6) + a4) + a3) + a2) + a1)
        + 1.0
    )
    dn = (
        eta2
        * (
            eta2
            * (
                eta * (eta * (eta * (eta2 * (eta2 * b11 + b9) + b7) + b6) + b5)
                + b4
            )
            + b2
        )
        + b0
    )
    ea = jnp.exp(-eta * alpha)
    eb = jnp.exp(-eta * beta)
    eg = jnp.exp(-eta * gamma)
    return jnp.log(1.0 - ea) + jnp.log(1.0 - eb) + jnp.log(1.0 - eg) - up / dn


def fharm12_anharm8_jax(
    gami: jax.Array, tpt: jax.Array
) -> tuple[jax.Array, ...]:
    """Evaluate `FHARM12 + ANHARM8` returning `(f, u, p, s, cv, pdt, pdr)`.

    Args:
        gami: Ion Coulomb coupling parameter $\\Gamma_i$.
        tpt: Quantum plasma temperature ratio $T_p / T$.

    Returns:
        7-tuple `(f_tot, u_tot, p_tot, s_tot, cv_tot, pdt_i, pdr_i)`.
    """
    clm, u1, c11 = -2.49389, 0.5113875, 0.0047770935
    eta = jnp.maximum(tpt, 1.0e-99)
    eta2 = eta * eta
    f_mid = _hlfit_f_mid(eta)
    df_deta = jax.grad(_hlfit_f_mid)(eta)
    d2f_deta2 = jax.grad(jax.grad(_hlfit_f_mid))(eta)
    u_mid = -eta * df_deta
    cv_mid = -eta2 * d2f_deta2

    u_hi = 3.0 / (eta**3 * c11)
    f_lo = 3.0 * jnp.log(eta) + clm - 1.5 * u1 * eta + eta2 / 24.0
    u_lo = 3.0 - 1.5 * u1 * eta + eta2 / 12.0
    cv_lo = 3.0 - eta2 / 12.0
    f_hl = jnp.where(
        eta > 1.0e5, -u_hi / 3.0, jnp.where(eta < 1.0e-5, f_lo, f_mid)
    )
    u_hl = jnp.where(eta > 1.0e5, u_hi, jnp.where(eta < 1.0e-5, u_lo, u_mid))
    cv_th = jnp.where(
        eta > 1.0e5, 4.0 * u_hi, jnp.where(eta < 1.0e-5, cv_lo, cv_mid)
    )
    s_harm = u_hl - f_hl
    u0 = -0.895929256 * gami
    e0 = 1.5 * u1 * tpt
    u_th = u_hl + e0
    f_harm = u0 + f_hl + e0
    u_harm = u0 + u_th
    p_harm = u0 / 3.0 + 0.5 * u_th
    pdt_harm = 0.5 * cv_th
    pdr_harm = u0 / 2.25 + 0.75 * u_th - 0.25 * cv_th

    b1 = 0.12
    aa = (10.9, 247.0, 1.765e5)
    ck = b1 / aa[0]
    tpt2 = tpt * tpt
    tpt4 = tpt2 * tpt2
    tq = b1 * tpt2 / gami
    tk2 = ck * tpt2
    sup_gn = jnp.exp(-tk2)
    f_ah = jnp.zeros_like(gami)
    u_ah = jnp.zeros_like(gami)
    p_ah = jnp.zeros_like(gami)
    cv_ah = jnp.zeros_like(gami)
    pdt_ah = jnp.zeros_like(gami)
    pdr_ah = jnp.zeros_like(gami)
    for idx, acn in enumerate(aa):
        cn = float(idx + 1)
        sup_gn = sup_gn / gami
        f_ah = f_ah - sup_gn * (acn / cn)
        u_ah = u_ah + (1.0 + 2.0 / cn * tk2) * acn * sup_gn
        pn = acn / 3.0 + acn / cn * tk2
        p_ah = p_ah + pn * sup_gn
        cv_ah = (
            cv_ah
            + (
                (cn + 1.0) * acn
                + (4.0 - 2.0 / cn) * acn * tk2
                + 4.0 * acn * ck * ck / cn * tpt4
            )
            * sup_gn
        )
        pdt_ah = (
            pdt_ah
            + (pn * (cn + 1.0 + 2.0 * tk2) - 2.0 * acn / cn * tk2) * sup_gn
        )
        pdr_ah = (
            pdr_ah + (pn * (1.0 - cn / 3.0 - tk2) + acn / cn * tk2) * sup_gn
        )
    f_ah = f_ah - tq
    u_ah = u_ah - tq
    p_ah = p_ah - tq / 1.5
    pdr_ah = pdr_ah - tq / 4.5
    return (
        f_harm + f_ah,
        u_harm + u_ah,
        p_harm + p_ah,
        s_harm + u_ah - f_ah,
        cv_th + cv_ah,
        pdt_harm + pdt_ah,
        pdr_harm + pdr_ah,
    )
