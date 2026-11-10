# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branchless JAX Coulomb liquid/crystal and screening kernels for `PC` EOS.

Implements `EXCOR7`, `FITION9`, `FHARM12`, and `ANHARM8` from
`eos/private/pc_eos.f90` using branchless `jnp.where` primitives and
forward-mode automatic differentiation (`jax.jacfwd`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _excor7_fxc(
    theta: jax.Array, game: jax.Array, th_min: float = 0.04
) -> jax.Array:
    """Evaluate Tanaka & Ichimaru `EXCOR7` free energy $f_{\\mathrm{xc}}$."""
    sqth = jnp.sqrt(theta)
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta3 * theta
    th_safe = jnp.maximum(theta, th_min)
    sq_safe = jnp.sqrt(th_safe)
    hi = theta > th_min
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
    sqth = jnp.sqrt(theta)
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta3 * theta
    hi = theta > 0.04
    th_s = jnp.where(hi, theta, 1.0)
    sq_s = jnp.sqrt(th_s)
    cht1, sht1 = jnp.cosh(1.0 / th_s), jnp.sinh(1.0 / th_s)
    cht2, sht2 = jnp.cosh(1.0 / sq_s), jnp.sinh(1.0 / sq_s)
    th_c1 = th_s * cht1
    th_c2 = th_s * cht2
    z0 = jnp.zeros_like(theta)
    o1 = jnp.ones_like(theta)
    t1 = jnp.where(hi, sht1 / cht1, o1)
    t2 = jnp.where(hi, sht2 / cht2, o1)
    t1dh = jnp.where(hi, -1.0 / (th_c1 * th_c1), z0)
    t1dhh = jnp.where(
        hi, (cht1 - sht1 / th_s) * 2.0 / (th_c1 * th_c1 * th_c1), z0
    )
    t2dh = jnp.where(hi, -sq_s * 0.5 / (th_c2 * th_c2), z0)
    t2dhh = jnp.where(
        hi, (sq_s * cht2 * 0.75 - sht2 * 0.5) / (th_c2 * th_c2 * th_c2), z0
    )

    a0 = theta2 * 3.04363 - theta3 * 0.09227 + theta4 * 1.7035 + 0.75
    a0dh = theta * 6.08726 - theta2 * 0.27681 + theta3 * 6.814
    a0dhh = -theta * 0.55362 + theta2 * 20.442 + 6.08726
    a1 = theta2 * 8.31051 + theta4 * 5.1105 + 1.0
    a1dh = theta * 16.62102 + theta3 * 20.442
    a1dhh = theta2 * 61.326 + 16.62102
    a = a0 / a1 * t1 * 0.610887
    ah = a0dh / a0 - a1dh / a1 + t1dh / t1
    adh = a * ah
    adhh = adh * ah + a * (
        a0dhh / a0
        - (a0dh / a0) ** 2
        - a1dhh / a1
        + (a1dh / a1) ** 2
        + t1dhh / t1
        - (t1dh / t1) ** 2
    )

    b0 = theta2 * 12.070873 + theta4 * 1.148889 + 0.341308
    b0dh = theta * 24.141746 + theta3 * 4.595556
    b0dhh = theta2 * 13.786668 + 24.141746
    b1 = theta2 * 10.495346 + theta4 * 1.326623 + 1.0
    b1dh = theta * 20.990692 + theta3 * 5.306492
    b1dhh = theta2 * 15.919476 + 20.990692
    b = sqth * t2 * b0 / b1
    bh = 0.5 / theta + t2dh / t2 + b0dh / b0 - b1dh / b1
    bdh = b * bh
    bdhh = bdh * bh + b * (
        -0.5 / theta2
        + t2dhh / t2
        - (t2dh / t2) ** 2
        + b0dhh / b0
        - (b0dh / b0) ** 2
        - b1dhh / b1
        + (b1dh / b1) ** 2
    )

    d0 = theta2 * 16.996055 + theta4 * 1.489056 + 0.614925
    d0dh = theta * 33.99211 + theta3 * 5.956224
    d0dhh = theta2 * 17.868672 + 33.99211
    d1 = theta2 * 10.10935 + theta4 * 1.22184 + 1.0
    d1dh = theta * 20.2187 + theta3 * 4.88736
    d1dhh = theta2 * 14.66208 + 20.2187
    d = sqth * t2 * d0 / d1
    dh = 0.5 / theta + t2dh / t2 + d0dh / d0 - d1dh / d1
    ddh = d * dh
    ddhh = ddh * dh + d * (
        -0.5 / theta2
        + t2dhh / t2
        - (t2dh / t2) ** 2
        + d0dhh / d0
        - (d0dh / d0) ** 2
        - d1dhh / d1
        + (d1dh / d1) ** 2
    )

    e0 = theta2 * 2.522206 + theta4 * 0.178484 + 0.539409
    e0dh = theta * 5.044412 + theta3 * 0.713936
    e0dhh = theta2 * 2.141808 + 5.044412
    e1 = theta2 * 2.555501 + theta4 * 0.146319 + 1.0
    e1dh = theta * 5.111002 + theta3 * 0.585276
    e1dhh = theta2 * 1.755828 + 5.111002
    e = theta * t1 * e0 / e1
    eh = 1.0 / theta + t1dh / t1 + e0dh / e0 - e1dh / e1
    edh = e * eh
    edhh = edh * eh + e * (
        t1dhh / t1
        - (t1dh / t1) ** 2
        + e0dhh / e0
        - (e0dh / e0) ** 2
        - e1dhh / e1
        + (e1dh / e1) ** 2
        - 1.0 / theta2
    )

    exp1th = jnp.exp(-1.0 / jnp.maximum(theta, 1.0e-3))
    c = (exp1th * 0.025248 + 0.872496) * e
    cdh = exp1th * 0.025248 / theta2 * e + c * edh / e
    cdhh = (
        exp1th * 0.025248 / theta2 * (edh + (-theta * 2.0 + 1.0) / theta2 * e)
        + cdh * edh / e
        + c * edhh / e
        - c * (edh / e) ** 2
    )
    discr = jnp.sqrt(e * 4.0 - d * d)
    didh = (edh * 4.0 - d * ddh * 2.0) * 0.5 / discr
    didhh = (
        -(((edh * 2.0 - d * ddh) / discr) ** 2)
        + edhh * 2.0
        - ddh * ddh
        - d * ddhh
    ) / discr

    s1 = -c / e * game
    s1h = cdh / c - edh / e
    s1dh = s1 * s1h
    s1dhh = s1dh * s1h + s1 * (
        cdhh / c - (cdh / c) ** 2 - edhh / e + (edh / e) ** 2
    )
    s1dg = -c / e
    s1dhg = s1dg * s1h

    b2 = b - c * d / e
    b2dh = bdh - (cdh * d + c * ddh) / e + c * d * edh / (e * e)
    b2dhh = (
        bdhh
        - (cdhh * d + cdh * ddh * 2.0 + c * ddhh) / e
        + ((cdh * d + c * ddh - c * d * edh / e) * edh * 2.0 + c * d * edhh)
        / (e * e)
    )
    sqge = jnp.sqrt(game)
    s2 = -b2 * sqge * 2.0 / e
    s2h = b2dh / b2 - edh / e
    s2dh = s2 * s2h
    s2dhh = s2dh * s2h + s2 * (
        b2dhh / b2 - (b2dh / b2) ** 2 - edhh / e + (edh / e) ** 2
    )
    s2dg = s2 * 0.5 / game
    s2dgg = -s2dg * 0.5 / game
    s2dhg = s2dh * 0.5 / game

    r3 = e * game + d * sqge + 1.0
    r3dh = edh * game + ddh * sqge
    r3dhh = edhh * game + ddhh * sqge
    r3dg = e + d * 0.5 / sqge
    r3dgg = -d * 0.25 / (game * sqge)
    r3dhg = edh + ddh * 0.5 / sqge

    b3 = a - c / e
    b3dh = adh - cdh / e + c * edh / (e * e)
    b3dhh = (
        adhh
        - cdhh / e
        + (cdh * edh * 2.0 + c * edhh) / (e * e)
        - c * edh * edh * 2.0 / (e * e * e)
    )
    c3 = (d / e * b2 - b3) / e
    c3dh = (
        (ddh * b2 + d * b2dh + b3 * edh) / (e * e)
        - d * b2 * edh * 2.0 / (e * e * e)
        - b3dh / e
    )
    c3dhh = (
        -b3dhh
        + (
            ddhh * b2
            + ddh * b2dh * 2.0
            + d * b2dhh
            + b3dh * edh * 2.0
            + b3 * edhh
        )
        / e
        - ((ddh * b2 * 2.0 + d * b2dh * 2.0 + b3 * edh) * edh + d * b2 * edhh)
        * 2.0
        / (e * e)
        + d * b2 * edh * edh * 6.0 / (e * e * e)
    ) / e

    s3 = c3 * jnp.log(r3)
    s3dh = s3 * c3dh / c3 + c3 * r3dh / r3
    s3dhh = (
        (s3dh * c3dh + s3 * c3dhh) / c3
        - s3 * (c3dh / c3) ** 2
        + (c3dh * r3dh + c3 * r3dhh) / r3
        - c3 * (r3dh / r3) ** 2
    )
    s3dg = c3 * r3dg / r3
    s3dgg = c3 * (r3dgg / r3 - (r3dg / r3) ** 2)
    s3dhg = (c3dh * r3dg + c3 * r3dhg) / r3 - c3 * r3dg * r3dh / (r3 * r3)

    b4 = -d * d / e + 2.0
    b4dh = edh * (d / e) * (d / e) - d * ddh * 2.0 / e
    b4dhh = (
        edhh * (d / e) * (d / e)
        + edh * (d / e) * (d / e) * (ddh / d - edh / e) * 2.0
        - (ddh * ddh + d * ddhh) * 2.0 / e
        + d * ddh * edh * 2.0 / (e * e)
    )
    c4 = e * sqge * 2.0 + d
    c4dh = edh * sqge * 2.0 + ddh
    c4dhh = edhh * sqge * 2.0 + ddhh
    c4dg = e / sqge
    c4dgg = -e * 0.5 / (game * sqge)
    c4dhg = edh / sqge

    s4a = 2.0 / (e * discr)
    s4ah = edh / e + didh / discr
    s4adh = -s4a * s4ah
    s4adhh = -s4adh * s4ah - s4a * (
        edhh / e - (edh / e) ** 2 + didhh / discr - (didh / discr) ** 2
    )
    s4b = d * b3 + b4 * b2
    s4bdh = ddh * b3 + d * b3dh + b4dh * b2 + b4 * b2dh
    s4bdhh = (
        ddhh * b3
        + ddh * b3dh * 2.0
        + d * b3dhh
        + b4dhh * b2
        + b4dh * b2dh * 2.0
        + b4 * b2dhh
    )
    s4c = jnp.arctan(c4 / discr) - jnp.arctan(d / discr)
    up1 = c4dh * discr - c4 * didh
    dn1 = discr * discr + c4 * c4
    up2 = ddh * discr - d * didh
    dn2 = discr * discr + d * d
    s4cdh = up1 / dn1 - up2 / dn2
    s4cdhh = (
        (c4dhh * discr - c4 * didhh) / dn1
        - up1 * (discr * didh + c4 * c4dh) * 2.0 / (dn1 * dn1)
        - (ddhh * discr - d * didhh) / dn2
        + up2 * (discr * didh + d * ddh) * 2.0 / (dn2 * dn2)
    )
    s4cdg = c4dg * discr / dn1
    s4cdgg = c4dgg * discr / dn1 - c4 * discr * ((c4dg / dn1) ** 2) * 2.0
    s4cdhg = (
        c4dhg * discr
        + c4dg * didh
        - c4dg * discr / dn1 * (discr * didh + c4 * c4dh) * 2.0
    ) / dn1

    s4 = s4a * s4b * s4c
    s4dh = s4adh * s4b * s4c + s4a * s4bdh * s4c + s4a * s4b * s4cdh
    s4dhh = (
        s4adhh * s4b * s4c
        + s4a * s4bdhh * s4c
        + s4a * s4b * s4cdhh
        + (s4adh * s4bdh * s4c + s4adh * s4b * s4cdh + s4a * s4bdh * s4cdh)
        * 2.0
    )
    s4dg = s4a * s4b * s4cdg
    s4dgg = s4a * s4b * s4cdgg
    s4dhg = s4a * s4b * s4cdhg + s4cdg * (s4adh * s4b + s4a * s4bdh)

    fxc = s1 + s2 + s3 + s4
    fxcdh = s1dh + s2dh + s3dh + s4dh
    fxcdg = s1dg + s2dg + s3dg + s4dg
    fxcdhh = s1dhh + s2dhh + s3dhh + s4dhh
    fxcdgg = s2dgg + s3dgg + s4dgg
    fxcdhg = s1dhg + s2dhg + s3dhg + s4dhg

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
    return _hlfit_mid(eta)[0]


def _hlfit_mid(eta: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate intermediate-regime bcc phonon `(f_mid, u_mid, cv_mid)`."""
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
    f = jnp.log(1.0 - ea) + jnp.log(1.0 - eb) + jnp.log(1.0 - eg) - up / dn
    up1 = (
        eta
        * (
            eta
            * (
                eta * (eta2 * (eta2 * (8.0 * a8) + 6.0 * a6) + 4.0 * a4)
                + 3.0 * a3
            )
            + 2.0 * a2
        )
        + a1
    )
    up2 = (
        eta
        * (
            eta * (eta2 * (eta2 * (56.0 * a8) + 30.0 * a6) + 12.0 * a4)
            + 6.0 * a3
        )
        + 2.0 * a2
    )
    dn1 = eta * (
        eta2
        * (
            eta
            * (
                eta
                * (
                    eta * (eta2 * (eta2 * (11.0 * b11) + 9.0 * b9) + 7.0 * b7)
                    + 6.0 * b6
                )
                + 5.0 * b5
            )
            + 4.0 * b4
        )
        + 2.0 * b2
    )
    dn2 = (
        eta2
        * (
            eta
            * (
                eta
                * (
                    eta
                    * (eta2 * (eta2 * (110.0 * b11) + 72.0 * b9) + 42.0 * b7)
                    + 30.0 * b6
                )
                + 20.0 * b5
            )
            + 12.0 * b4
        )
        + 2.0 * b2
    )
    u_over_eta = (
        ea * alpha / (1.0 - ea)
        + eb * beta / (1.0 - eb)
        + eg * gamma / (1.0 - eg)
        - (up1 * dn - dn1 * up) / (dn * dn)
    )
    cv_over_eta2 = (
        ea * (alpha * alpha) / ((1.0 - ea) * (1.0 - ea))
        + eb * (beta * beta) / ((1.0 - eb) * (1.0 - eb))
        + eg * (gamma * gamma) / ((1.0 - eg) * (1.0 - eg))
        + ((up2 * dn - dn2 * up) * dn - (up1 * dn - dn1 * up) * dn1 * 2.0)
        / (dn * dn * dn)
    )
    return f, u_over_eta * eta, cv_over_eta2 * eta2


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
    f_mid, u_mid, cv_mid = _hlfit_mid(eta)

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
