# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branchless `@jax.jit` (`float64`) 3rd-order jet Coulomb kernels for `Skye`.

Implements the one-component plasma (OCP) liquid and solid free energies,
relativistic electron screening corrections, boundary extrapolation, `EXCOR7`
exchange-correlation, and liquid/solid mixing corrections from
`skye_coulomb.f90`, `skye_coulomb_liquid.f90`, and `skye_coulomb_solid.f90`
(Jermyn et al. 2021, ApJ, 913, 72) using `(..., 10)` `AutoDiffReal2VarOrder3`
array jets (`_jet_jax`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants
from flexmesa.eos import _jet_jax

_ME_IN_AMU: float = constants.ME / constants.AMU
COUL_PREF: float = (constants.QE * constants.QE) / (
    constants.RBOHR * constants.BOLTZM
)
_XR_PREF: float = ((9.0 * constants.PI / 4.0) ** (1.0 / 3.0)) * constants.FINE


def ocp_liq_j3(g: jax.Array, tpt: jax.Array) -> jax.Array:
    """Evaluate classical + quantum OCP liquid free energy jet `(..., 10)`."""
    sq32 = jnp.sqrt(3.0) / 2.0
    a1, a2 = -0.9070, 0.62954
    a3 = -sq32 - a1 / jnp.sqrt(a2)
    b1, b2, b3, b4 = 4.56e-3, 211.6, -1.0e-4, 4.62e-3
    sq_g = _jet_jax.j3_sqrt(g)
    g_a2 = _jet_jax.j3_scale(g, 1.0 / a2)
    log_a = _jet_jax.j3_log(
        _jet_jax.j3_sqrt(g_a2) + _jet_jax.j3_sqrt(_jet_jax.j3_add_s(g_a2, 1.0))
    )
    fa = _jet_jax.j3_scale(
        _jet_jax.j3_sqrt(_jet_jax.j3_mul(g, _jet_jax.j3_add_s(g, a2)))
        - _jet_jax.j3_scale(log_a, a2),
        a1,
    ) + _jet_jax.j3_scale(sq_g - _jet_jax.j3_atan(sq_g), 2.0 * a3)
    fb = _jet_jax.j3_scale(
        g
        - _jet_jax.j3_scale(
            _jet_jax.j3_log(
                _jet_jax.j3_add_s(_jet_jax.j3_scale(g, 1.0 / b2), 1.0)
            ),
            b2,
        ),
        b1,
    ) + _jet_jax.j3_scale(
        _jet_jax.j3_log(
            _jet_jax.j3_add_s(
                _jet_jax.j3_scale(_jet_jax.j3_pow2(g), 1.0 / b4), 1.0
            )
        ),
        0.5 * b3,
    )
    q1, q2, q4 = 5.994, 70.3, 22.7
    q3 = (0.25 - q1 / q2) * q4
    eta = _jet_jax.j3_scale(tpt, 1.0 / jnp.sqrt(3.0))
    fq = (
        _jet_jax.j3_scale(eta, q1)
        - _jet_jax.j3_scale(
            _jet_jax.j3_log(
                _jet_jax.j3_add_s(_jet_jax.j3_scale(eta, 1.0 / q2), 1.0)
            ),
            q1 * q2,
        )
        + _jet_jax.j3_scale(
            _jet_jax.j3_log(
                _jet_jax.j3_add_s(
                    _jet_jax.j3_scale(_jet_jax.j3_pow2(eta), 1.0 / q4), 1.0
                )
            ),
            0.5 * q3,
        )
    )
    return fa + fb + fq


def _hlfit_f_mid_j3(eta: jax.Array) -> jax.Array:
    """Evaluate intermediate-regime bcc phonon free-energy jet `f_mid(eta)`."""
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
    eta2 = _jet_jax.j3_mul(eta, eta)

    p_up = _jet_jax.j3_add_s(_jet_jax.j3_scale(eta2, a8), a6)
    p_up = _jet_jax.j3_add_s(_jet_jax.j3_mul(eta2, p_up), a4)
    p_up = _jet_jax.j3_add_s(_jet_jax.j3_mul(eta, p_up), a3)
    p_up = _jet_jax.j3_add_s(_jet_jax.j3_mul(eta, p_up), a2)
    p_up = _jet_jax.j3_add_s(_jet_jax.j3_mul(eta, p_up), a1)
    up = _jet_jax.j3_add_s(_jet_jax.j3_mul(eta, p_up), 1.0)

    p_dn = _jet_jax.j3_add_s(_jet_jax.j3_scale(eta2, b11), b9)
    p_dn = _jet_jax.j3_add_s(_jet_jax.j3_mul(eta2, p_dn), b7)
    p_dn = _jet_jax.j3_add_s(_jet_jax.j3_mul(eta, p_dn), b6)
    p_dn = _jet_jax.j3_add_s(_jet_jax.j3_mul(eta, p_dn), b5)
    p_dn = _jet_jax.j3_add_s(_jet_jax.j3_mul(eta, p_dn), b4)
    p_dn = _jet_jax.j3_add_s(_jet_jax.j3_mul(eta2, p_dn), b2)
    dn = _jet_jax.j3_add_s(_jet_jax.j3_mul(eta2, p_dn), b0)

    ea = _jet_jax.j3_exp(_jet_jax.j3_scale(eta, -0.265764))
    eb = _jet_jax.j3_exp(_jet_jax.j3_scale(eta, -0.334547))
    eg = _jet_jax.j3_exp(_jet_jax.j3_scale(eta, -0.932446))
    return (
        _jet_jax.j3_log(_jet_jax.j3_rsub_s(1.0, ea))
        + _jet_jax.j3_log(_jet_jax.j3_rsub_s(1.0, eb))
        + _jet_jax.j3_log(_jet_jax.j3_rsub_s(1.0, eg))
        - _jet_jax.j3_div(up, dn)
    )


def ocp_sol_j3(gami: jax.Array, tpt: jax.Array) -> jax.Array:
    """Evaluate harmonic + anharmonic OCP solid free-energy jet `(..., 10)`."""
    tpt2 = _jet_jax.j3_mul(tpt, tpt)
    g_inv = _jet_jax.j3_inv(gami)
    g2_inv = _jet_jax.j3_inv(_jet_jax.j3_pow2(gami))
    g3_inv = _jet_jax.j3_inv(_jet_jax.j3_pow3(gami))
    f_neg = -(
        _jet_jax.j3_scale(g_inv, 10.9)
        + _jet_jax.j3_scale(g2_inv, 247.0 / 2.0)
        + _jet_jax.j3_scale(g3_inv, 1.765e5 / 3.0)
    )
    f_ah = _jet_jax.j3_mul(
        f_neg, _jet_jax.j3_exp(_jet_jax.j3_scale(tpt2, -(0.12 / 10.9)))
    ) - _jet_jax.j3_div(_jet_jax.j3_scale(tpt2, 0.12), gami)

    tpt_v = tpt[..., 0]
    tpt_m = _jet_jax.j3_min_s(_jet_jax.j3_max_s(tpt, 1.0e-5), 1.0e5)
    f_mid = _hlfit_f_mid_j3(tpt_m)
    tpt_hi = _jet_jax.j3_max_s(tpt, 1.0)
    f_hi = _jet_jax.j3_rdiv_s(
        -1.0,
        _jet_jax.j3_scale(
            _jet_jax.j3_mul(_jet_jax.j3_mul(tpt_hi, tpt_hi), tpt_hi),
            0.0047770935,
        ),
    )
    tpt_lo = _jet_jax.j3_min_s(tpt, 1.0)
    f_lo = (
        _jet_jax.j3_add_s(
            _jet_jax.j3_scale(_jet_jax.j3_log(tpt_lo), 3.0), -2.49389
        )
        - _jet_jax.j3_scale(tpt_lo, 1.5 * 0.5113875)
        + _jet_jax.j3_scale(_jet_jax.j3_mul(tpt_lo, tpt_lo), 1.0 / 24.0)
    )
    f_th = _jet_jax.j3_where(
        tpt_v > 1.0e5, f_hi, _jet_jax.j3_where(tpt_v < 1.0e-5, f_lo, f_mid)
    )
    f_harm = _jet_jax.j3_add_s(
        _jet_jax.j3_scale(gami, -0.895929256)
        + f_th
        + _jet_jax.j3_scale(tpt, 1.5 * 0.5113875)
        - _jet_jax.j3_scale(_jet_jax.j3_log(tpt), 3.0)
        + _jet_jax.j3_scale(_jet_jax.j3_log(gami), 1.5),
        1.323515,
    )
    return f_harm + f_ah


def _screen_liq_with_xr_j3(
    zi: jax.Array,
    ge: jax.Array,
    rs: jax.Array,
    xr: jax.Array,
    xr2: jax.Array,
    gr: jax.Array,
) -> jax.Array:
    """Evaluate liquid screening jet given precomputed `(xr, xr2, gr)`."""
    z2 = zi * zi
    logz = jnp.log(zi)
    a = 1.11 * jnp.power(zi, 0.475)
    b = 0.2 + 0.078 * logz * logz
    nu = 1.16 + 0.08 * logz
    cdh = (zi / jnp.sqrt(3.0)) * (
        jnp.power(1.0 + zi, 1.5) - 1.0 - jnp.power(zi, 1.5)
    )
    ctf = (
        (18.0 / 175.0)
        * jnp.power(12.0 / constants.PI, 2.0 / 3.0)
        * jnp.power(zi, 7.0 / 3.0)
        * (1.0 - jnp.power(zi, -1.0 / 3.0) + 0.2 * jnp.power(zi, -0.5))
    )
    g1 = _jet_jax.j3_add_s(
        _jet_jax.j3_div(
            _jet_jax.j3_scale(
                _jet_jax.j3_sqrt(_jet_jax.j3_scale(ge, 1.0 / zi)), 0.78
            ),
            _jet_jax.j3_add_s(
                _jet_jax.j3_mul(
                    ge, _jet_jax.j3_pow3(_jet_jax.j3_rdiv_s(zi, rs))
                ),
                21.0,
            ),
        ),
        1.0,
    )
    g2 = _jet_jax.j3_add_s(
        _jet_jax.j3_scale(
            _jet_jax.j3_mul(
                _jet_jax.j3_add_s(
                    _jet_jax.j3_rdiv_s(
                        1.0,
                        _jet_jax.j3_add_s(
                            _jet_jax.j3_scale(ge, 2.0), 0.001 * z2
                        ),
                    ),
                    1.0,
                ),
                _jet_jax.j3_div(
                    _jet_jax.j3_pow3(rs),
                    _jet_jax.j3_add_s(
                        _jet_jax.j3_scale(_jet_jax.j3_pow2(rs), 6.0), 1.0
                    ),
                ),
            ),
            (zi - 1.0) / 9.0,
        ),
        1.0,
    )
    h = _jet_jax.j3_div(
        _jet_jax.j3_add_s(_jet_jax.j3_scale(xr2, 0.2), 1.0),
        _jet_jax.j3_add_s(
            _jet_jax.j3_scale(xr, 0.18 * jnp.power(zi, -0.25))
            + _jet_jax.j3_scale(xr2, 0.37 * jnp.power(zi, -0.5) + 0.2),
            1.0,
        ),
    )
    sq_ge = _jet_jax.j3_sqrt(ge)
    ge_nu = _jet_jax.j3_powf(ge, nu)
    num = -_jet_jax.j3_mul(
        ge,
        _jet_jax.j3_scale(sq_ge, cdh)
        + _jet_jax.j3_mul(
            _jet_jax.j3_mul(_jet_jax.j3_scale(ge_nu, ctf * a), g1), h
        ),
    )
    den = _jet_jax.j3_add_s(
        _jet_jax.j3_div(
            _jet_jax.j3_scale(sq_ge, b)
            + _jet_jax.j3_div(
                _jet_jax.j3_mul(_jet_jax.j3_scale(g2, a), ge_nu), rs
            ),
            gr,
        ),
        1.0,
    )
    return _jet_jax.j3_div(num, den)


def screen_liq_j3(zi: jax.Array, ge: jax.Array, rs: jax.Array) -> jax.Array:
    """Evaluate liquid electron-ion screening free-energy jet `(..., 10)`."""
    xr = _jet_jax.j3_rdiv_s(_XR_PREF, rs)
    xr2 = _jet_jax.j3_pow2(xr)
    gr = _jet_jax.j3_sqrt(_jet_jax.j3_add_s(xr2, 1.0))
    return _screen_liq_with_xr_j3(zi, ge, rs, xr, xr2, gr)


def screen_sol_j3(
    zi: jax.Array, mi: jax.Array, ge: jax.Array, rs: jax.Array
) -> jax.Array:
    """Evaluate solid electron-ion screening free-energy jet `(..., 10)`."""
    z2 = zi * zi
    z5 = z2 * z2 * zi
    z6 = z5 * zi
    logz = jnp.log(zi)
    z5_3 = jnp.power(zi, 5.0 / 3.0)
    z7_6 = jnp.power(zi, 7.0 / 6.0)
    z2_3 = jnp.power(zi, 2.0 / 3.0)
    s = 1.0 / (1.0 + 1.0e-2 * jnp.power(logz, 1.5) + 0.097 / z2)
    b1 = 1.0 - 1.1866 * jnp.power(zi, -0.267) + 0.27 / zi
    b2 = 1.0 + (2.25 * jnp.power(zi, -1.0 / 3.0)) * (
        1.0 + 0.684 * z5 + 0.222 * z6
    ) / (1.0 + 0.222 * z6)
    b3 = 41.5 / (1.0 + logz)
    b4 = 0.395 * logz + 0.347 * jnp.power(zi, -1.5)

    g = _jet_jax.j3_scale(ge, z5_3)
    cotpt = jnp.sqrt(3.0 * _ME_IN_AMU / mi) / z7_6
    tpt = _jet_jax.j3_div(_jet_jax.j3_scale(g, cotpt), _jet_jax.j3_sqrt(rs))
    tpt_s2 = _jet_jax.j3_pow2(_jet_jax.j3_scale(tpt, 0.205))
    supp = _jet_jax.j3_exp(
        _jet_jax.j3_max_s(_jet_jax.j3_min_s(-tpt_s2, 100.0), -100.0)
    )
    q = _jet_jax.j3_sqrt(
        _jet_jax.j3_div(
            tpt_s2 + _jet_jax.j3_log(_jet_jax.j3_add_s(supp, 1.0)),
            _jet_jax.j3_log(
                _jet_jax.j3_rsub_s(
                    constants.EULERNUM,
                    _jet_jax.j3_scale(supp, constants.EULERNUM - 2.0),
                )
            ),
        )
    )
    xr = _jet_jax.j3_rdiv_s(_XR_PREF, rs)
    xr2 = _jet_jax.j3_pow2(xr)
    a = _jet_jax.j3_div(
        _jet_jax.j3_add_s(_jet_jax.j3_scale(xr2, 17.9), b3),
        _jet_jax.j3_add_s(_jet_jax.j3_scale(xr2, b4), 1.0),
    )
    f_inf = _jet_jax.j3_scale(
        _jet_jax.j3_sqrt(_jet_jax.j3_add_s(_jet_jax.j3_rdiv_s(b2, xr2), 1.0)),
        0.00352 * z2_3 * b1,
    )
    f_sol = -_jet_jax.j3_mul(
        _jet_jax.j3_mul(f_inf, g),
        _jet_jax.j3_add_s(
            _jet_jax.j3_mul(a, _jet_jax.j3_powf(_jet_jax.j3_div(q, g), s)), 1.0
        ),
    )
    gr = _jet_jax.j3_sqrt(_jet_jax.j3_add_s(xr2, 1.0))
    alpha = _jet_jax.j3_scale(
        _jet_jax.j3_mul(_jet_jax.j3_div(rs, ge), gr),
        3.0 * jnp.power(4.0 / (9.0 * constants.PI), 2.0 / 3.0),
    )
    f_liq = _screen_liq_with_xr_j3(zi, ge, rs, xr, xr2, gr)
    sw = _jet_jax.j3_pow3(
        _jet_jax.j3_tanh(
            _jet_jax.j3_max_s(
                _jet_jax.j3_min_s(_jet_jax.j3_scale(alpha, 2.0), 100.0), -100.0
            )
        )
    )
    return _jet_jax.j3_mul(sw, f_liq) + _jet_jax.j3_mul(
        _jet_jax.j3_rsub_s(1.0, sw), f_sol
    )


def _extrapolate_ocp_boundary_j3(
    is_liq: bool,
    temp: jax.Array,
    rs: jax.Array,
    sq_rs: jax.Array,
    tg_prod: jax.Array,
    fake_d: jax.Array,
    z5_3: jax.Array,
    cotpt: jax.Array,
    g_bound: jax.Array,
) -> jax.Array:
    """Linearly extrapolate OCP free energy in $T$ past `g_bound`."""
    tb_v1 = _jet_jax.j3_var1(tg_prod[..., 0] / g_bound)
    g_b = _jet_jax.j3_rdiv_s(z5_3 * COUL_PREF, _jet_jax.j3_mul(tb_v1, rs))
    tp_b = _jet_jax.j3_div(_jet_jax.j3_scale(g_b, cotpt), sq_rs)
    fb = ocp_liq_j3(g_b, tp_b) if is_liq else ocp_sol_j3(g_b, tp_b)
    dfb = _jet_jax.j3_mul(_jet_jax.j3_diff1(fb), tb_v1)
    tb = _jet_jax.j3_scale(tg_prod, 1.0 / g_bound)
    return _jet_jax.j3_compose(fb, tb, fake_d) + _jet_jax.j3_mul(
        _jet_jax.j3_rsub_s(1.0, _jet_jax.j3_div(tb, temp)),
        _jet_jax.j3_compose(dfb, tb, fake_d),
    )


def species_nonideal_j3(
    temp: jax.Array,
    rs: jax.Array,
    game: jax.Array,
    zi: jax.Array,
    mi: jax.Array,
    min_g_sol: jax.Array,
    max_g_liq: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate single-species `(f_liq, f_sol)` 3rd-order jets."""
    z_s = jnp.maximum(jnp.floor(zi), 1.0)
    m_s = jnp.maximum(mi, 1.0e-30)
    z5_3 = jnp.power(z_s, 5.0 / 3.0)
    z7_6 = jnp.power(z_s, 7.0 / 6.0)
    cotpt = jnp.sqrt(3.0 * _ME_IN_AMU / m_s) / z7_6
    sq_rs = _jet_jax.j3_sqrt(rs)
    gami = _jet_jax.j3_rdiv_s(z5_3 * COUL_PREF, _jet_jax.j3_mul(temp, rs))
    tpt = _jet_jax.j3_div(_jet_jax.j3_scale(gami, cotpt), sq_rs)
    fake_d = _jet_jax.j3_var2(jnp.zeros_like(temp[..., 0]))
    tg_prod = _jet_jax.j3_mul(temp, gami)
    fl_ext = _extrapolate_ocp_boundary_j3(
        True, temp, rs, sq_rs, tg_prod, fake_d, z5_3, cotpt, max_g_liq
    )
    fs_ext = _extrapolate_ocp_boundary_j3(
        False, temp, rs, sq_rs, tg_prod, fake_d, z5_3, cotpt, min_g_sol
    )
    f_liq = _jet_jax.j3_where(
        gami[..., 0] < max_g_liq, ocp_liq_j3(gami, tpt), fl_ext
    ) + screen_liq_j3(z_s, game, rs)
    f_sol = _jet_jax.j3_where(
        gami[..., 0] > min_g_sol, ocp_sol_j3(gami, tpt), fs_ext
    ) + screen_sol_j3(z_s, m_s, game, rs)
    return f_liq, f_sol


def excor7_j3(rs: jax.Array, game: jax.Array) -> jax.Array:
    """Evaluate `excor7_order3` (`coulomb_liquid.rs:143-198`) as `(..., 10)`."""
    theta = _jet_jax.j3_div(_jet_jax.j3_scale(rs, 0.543), game)
    sqth = _jet_jax.j3_sqrt(theta)
    th2 = _jet_jax.j3_mul(theta, theta)
    th3 = _jet_jax.j3_mul(th2, theta)
    th4 = _jet_jax.j3_mul(th3, theta)
    hi = theta[..., 0] > 0.007
    one_j = _jet_jax.j3_const(jnp.ones_like(theta[..., 0]))
    th_s = _jet_jax.j3_where(hi, theta, one_j)
    sq_s = _jet_jax.j3_where(hi, sqth, one_j)
    t1 = _jet_jax.j3_where(hi, _jet_jax.j3_tanh(_jet_jax.j3_inv(th_s)), one_j)
    t2 = _jet_jax.j3_where(hi, _jet_jax.j3_tanh(_jet_jax.j3_inv(sq_s)), one_j)

    a0 = _jet_jax.j3_add_s(
        _jet_jax.j3_scale(th2, 3.04363)
        - _jet_jax.j3_scale(th3, 0.09227)
        + _jet_jax.j3_scale(th4, 1.7035),
        0.75,
    )
    a1 = _jet_jax.j3_add_s(
        _jet_jax.j3_scale(th2, 8.31051) + _jet_jax.j3_scale(th4, 5.1105), 1.0
    )
    a = _jet_jax.j3_mul(
        _jet_jax.j3_div(_jet_jax.j3_scale(a0, 0.610887), a1), t1
    )

    b0 = _jet_jax.j3_add_s(
        _jet_jax.j3_scale(th2, 12.070873) + _jet_jax.j3_scale(th4, 1.148889),
        0.341308,
    )
    b1 = _jet_jax.j3_add_s(
        _jet_jax.j3_scale(th2, 10.495346) + _jet_jax.j3_scale(th4, 1.326623),
        1.0,
    )
    b = _jet_jax.j3_div(_jet_jax.j3_mul(_jet_jax.j3_mul(sqth, t2), b0), b1)

    d0 = _jet_jax.j3_add_s(
        _jet_jax.j3_scale(th2, 16.996055) + _jet_jax.j3_scale(th4, 1.489056),
        0.614925,
    )
    d1 = _jet_jax.j3_add_s(
        _jet_jax.j3_scale(th2, 10.10935) + _jet_jax.j3_scale(th4, 1.22184), 1.0
    )
    d = _jet_jax.j3_div(_jet_jax.j3_mul(_jet_jax.j3_mul(sqth, t2), d0), d1)

    e0 = _jet_jax.j3_add_s(
        _jet_jax.j3_scale(th2, 2.522206) + _jet_jax.j3_scale(th4, 0.178484),
        0.539409,
    )
    e1 = _jet_jax.j3_add_s(
        _jet_jax.j3_scale(th2, 2.555501) + _jet_jax.j3_scale(th4, 0.146319), 1.0
    )
    e = _jet_jax.j3_div(_jet_jax.j3_mul(_jet_jax.j3_mul(theta, t1), e0), e1)

    exp1th = _jet_jax.j3_exp(-_jet_jax.j3_inv(_jet_jax.j3_max_s(theta, 1.0e-3)))
    c_over_e = _jet_jax.j3_add_s(_jet_jax.j3_scale(exp1th, 0.025248), 0.872496)
    discr = _jet_jax.j3_sqrt(_jet_jax.j3_scale(e, 4.0) - _jet_jax.j3_mul(d, d))
    sqge = _jet_jax.j3_sqrt(game)
    s1 = -_jet_jax.j3_mul(c_over_e, game)
    b2 = b - _jet_jax.j3_mul(c_over_e, d)
    s2 = -_jet_jax.j3_div(_jet_jax.j3_scale(_jet_jax.j3_mul(b2, sqge), 2.0), e)
    r3 = _jet_jax.j3_add_s(
        _jet_jax.j3_mul(e, game) + _jet_jax.j3_mul(d, sqge), 1.0
    )
    b3 = a - c_over_e
    c3 = _jet_jax.j3_div(_jet_jax.j3_mul(_jet_jax.j3_div(d, e), b2) - b3, e)
    s3 = _jet_jax.j3_mul(c3, _jet_jax.j3_log(r3))
    b4 = _jet_jax.j3_rsub_s(2.0, _jet_jax.j3_div(_jet_jax.j3_mul(d, d), e))
    c4 = _jet_jax.j3_scale(_jet_jax.j3_mul(e, sqge), 2.0) + d
    s4a = _jet_jax.j3_rdiv_s(2.0, _jet_jax.j3_mul(e, discr))
    s4b = _jet_jax.j3_mul(d, b3) + _jet_jax.j3_mul(b4, b2)
    s4c = _jet_jax.j3_atan(_jet_jax.j3_div(c4, discr)) - _jet_jax.j3_atan(
        _jet_jax.j3_div(d, discr)
    )
    return s1 + s2 + s3 + _jet_jax.j3_mul(_jet_jax.j3_mul(s4a, s4b), s4c)


def mixing_corrections_j3(
    rs: jax.Array,
    game: jax.Array,
    ya: jax.Array,
    azion: jax.Array,
    mask: jax.Array,
    use_ogata: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return `(zmean, s_mix, fmix_liq_j3, fmix_sol_j3)` for a mixture."""
    y = jnp.where(mask, ya, 0.0)
    z = jnp.where(mask, azion, 0.0)
    zmean = jnp.sum(y * z)
    z2mean = jnp.sum(y * z * z)
    z53 = jnp.sum(y * jnp.power(jnp.maximum(z, 0.0), constants.FIVE_THIRDS))
    z52 = jnp.sum(y * (jnp.sqrt(jnp.maximum(z, 0.0)) ** 5))
    z321 = jnp.sum(y * z * (jnp.sqrt(jnp.maximum(z + 1.0, 0.0)) ** 3))
    s_mix = -jnp.sum(
        jnp.where(y > 0.0, y * jnp.log(jnp.maximum(y, 1.0e-99)), 0.0)
    )

    d = z2mean / jnp.maximum(zmean * zmean, 1.0e-99)
    active_mix = (zmean > 0.0) & (jnp.abs(d - 1.0) >= 1.0e-9)
    d_safe = jnp.where(active_mix, d, 2.0)
    zm = z2mean + zmean
    dif0 = jnp.where(
        rs[..., 0] < 1.0e-9,
        z52 - jnp.sqrt(z2mean * z2mean * z2mean / jnp.maximum(zmean, 1.0e-99)),
        z321 - jnp.sqrt(zm * zm * zm / jnp.maximum(zmean, 1.0e-99)),
    )
    dif_r = dif0 / jnp.maximum(z52, 1.0e-99)
    dif_fdh = _jet_jax.j3_mul(
        _jet_jax.j3_scale(game, dif0),
        _jet_jax.j3_sqrt(_jet_jax.j3_scale(game, 1.0 / 3.0)),
    )
    p3 = jnp.power(d_safe, -0.2)
    d0 = (2.6 * dif_r + 14.0 * (dif_r**3)) / (1.0 - p3)
    gp = _jet_jax.j3_scale(
        _jet_jax.j3_powf(_jet_jax.j3_scale(game, z53), p3), d0
    )
    fmix0 = _jet_jax.j3_div(dif_fdh, _jet_jax.j3_add_s(gp, 1.0))
    gq = _jet_jax.j3_scale(gp, d_safe * d_safe * 0.0117)
    zero_j = _jet_jax.j3_const(jnp.zeros_like(rs[..., 0]))
    fmix_liq = _jet_jax.j3_where(
        active_mix,
        _jet_jax.j3_div(
            fmix0, _jet_jax.j3_powf(_jet_jax.j3_add_s(gq, 1.0), 1.5 / p3 - 1.0)
        ),
        zero_j,
    )

    same_z = (z[:, None] == z[None, :]) & mask[:, None] & mask[None, :]
    ca = jnp.sum(jnp.where(same_z, y[None, :], 0.0), axis=1)
    idx = jnp.arange(z.shape[0])
    is_u = mask & (z > 0.0) & (jnp.argmax(same_z, axis=1) == idx)
    zi, zj = z[:, None], z[None, :]
    pair_ok = is_u[:, None] & is_u[None, :] & (zj > zi)
    rz = jnp.where(pair_ok, zj / jnp.maximum(zi, 1.0), 2.0)
    c_sum = jnp.maximum(ca[:, None] + ca[None, :], 1.0e-40)
    x2 = jnp.clip(
        jnp.where(pair_ok, ca[None, :] / c_sum, 0.5), 1.0e-30, 1.0 - 1.0e-15
    )
    rm1 = rz - 1.0
    sq = jnp.sqrt(x2)
    dg_og = (
        0.05
        * (rm1 * rm1)
        / ((1.0 + 0.64 * rm1) * (1.0 + 0.5 * rm1 * rm1))
        / (
            1.0
            + (sq * (sq - 0.3) * (sq - 0.7) * (sq - 1.0))
            * 27.0
            * rm1
            / (1.0 + 0.1 * rm1)
        )
    )
    x_pc = x2 / rz + (1.0 - 1.0 / rz) * jnp.power(x2, rz)
    dg_pc = (
        0.012
        * ((x_pc * (1.0 - x_pc)) / (x2 * (1.0 - x2)))
        * (1.0 - 1.0 / (rz * rz))
        * (1.0 - x2 + x2 * jnp.power(rz, 5.0 / 3.0))
    )
    dg = jnp.where(use_ogata, dg_og, dg_pc)
    coeff_sol = jnp.sum(
        jnp.where(
            pair_ok,
            jnp.power(jnp.maximum(zi, 1.0), 5.0 / 3.0)
            * ca[:, None]
            * ca[None, :]
            * dg,
            0.0,
        )
    )
    fmix_sol = _jet_jax.j3_scale(game, coeff_sol)
    return zmean, s_mix, fmix_liq, fmix_sol
