# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branchless JAX/XLA (`float64`) plasma electrostatic screening kernels.

Provides differentiable `@jax.jit` evaluators for Salpeter (1954),
Graboske/Alastuey/Itoh (`fxt_screen5`), and Chugunov et al. (2007) screening.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants


@jax.jit
def screen_salpeter_jax(
    temp: jax.Array,
    den: jax.Array,
    zbar: jax.Array,
    abar: jax.Array,
    z2bar: jax.Array,
    z1: jax.Array,
    z2: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate Salpeter (1954) weak plasma screening in JAX (`float64`).

    Args:
        temp: Temperature $T$ in $\\mathrm{K}$ (`scalar` or `(N,)`).
        den: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        zbar: Mean ionic charge $\\bar{Z}$ (`scalar` or `(N,)`).
        abar: Mean atomic weight $\\bar{A}$ (`scalar` or `(N,)`).
        z2bar: Mean square charge $\\overline{Z^2}$ (`scalar` or `(N,)`).
        z1: Reactant 1 atomic number $Z_1$.
        z2: Reactant 2 atomic number $Z_2$.

    Returns:
        Tuple `(scor, dscor_dt, dscor_dd)` of screening enhancement factor and
        its partial derivatives with respect to $T$ ($\\mathrm{K}^{-1}$) and
        $\\rho$ ($\\mathrm{cm^3\\,g^{-1}}$).
    """
    t = jnp.asarray(temp, dtype=jnp.float64)
    d = jnp.asarray(den, dtype=jnp.float64)
    zb = jnp.asarray(zbar, dtype=jnp.float64)
    ab = jnp.asarray(abar, dtype=jnp.float64)
    z2b = jnp.asarray(z2bar, dtype=jnp.float64)
    z1_v = jnp.asarray(z1, dtype=jnp.float64)
    z2_v = jnp.asarray(z2, dtype=jnp.float64)
    uncharged = (zb <= 0.0) | (z1_v <= 0.0) | (z2_v <= 0.0)

    zeta = jnp.maximum((z2b + zb) / ab, 0.0)
    lnf = 1.88e8 * z1_v * z2_v * jnp.sqrt(d * zeta / (t * t * t))
    scor = jnp.exp(lnf)
    dscor_dt = -1.5 * lnf / t * scor
    dscor_dd = 0.5 * lnf / d * scor
    z0 = jnp.zeros_like(t)
    return (
        jnp.where(uncharged, jnp.ones_like(t), scor),
        jnp.where(uncharged, z0, dscor_dt),
        jnp.where(uncharged, z0, dscor_dd),
    )


@jax.jit
def screen_extended_jax(
    temp: jax.Array,
    den: jax.Array,
    zbar: jax.Array,
    abar: jax.Array,
    z2bar: jax.Array,
    a1: jax.Array,
    z1: jax.Array,
    a2: jax.Array,
    z2: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate Graboske/Alastuey/Itoh (`fxt_screen5`) plasma screening in JAX.

    Args:
        temp: Temperature $T$ in $\\mathrm{K}$ (`scalar` or `(N,)`).
        den: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        zbar: Mean ionic charge $\\bar{Z}$ (`scalar` or `(N,)`).
        abar: Mean atomic weight $\\bar{A}$ (`scalar` or `(N,)`).
        z2bar: Mean square charge $\\overline{Z^2}$ (`scalar` or `(N,)`).
        a1: Reactant 1 atomic mass $A_1$ ($\\mathrm{g\\,mol^{-1}}$).
        z1: Reactant 1 atomic number $Z_1$.
        a2: Reactant 2 atomic mass $A_2$ ($\\mathrm{g\\,mol^{-1}}$).
        z2: Reactant 2 atomic number $Z_2$.

    Returns:
        Tuple `(scor, dscor_dt, dscor_dd)` of screening enhancement factor and
        its partial derivatives with respect to $T$ and $\\rho$.
    """
    t = jnp.asarray(temp, dtype=jnp.float64)
    d = jnp.asarray(den, dtype=jnp.float64)
    zb_raw = jnp.asarray(zbar, dtype=jnp.float64)
    ab = jnp.asarray(abar, dtype=jnp.float64)
    z2b = jnp.asarray(z2bar, dtype=jnp.float64)
    z1_raw = jnp.asarray(z1, dtype=jnp.float64)
    z2_raw = jnp.asarray(z2, dtype=jnp.float64)
    uncharged = (zb_raw <= 0.0) | (z1_raw <= 0.0) | (z2_raw <= 0.0)

    zb = jnp.maximum(zb_raw, 1.0e-30)
    z1_s = jnp.maximum(z1_raw, 1.0e-30)
    z2_s = jnp.maximum(z2_raw, 1.0e-30)

    ytot = 1.0 / ab
    rr = d * ytot
    tempi = 1.0 / t
    dtempidt = -tempi * tempi
    deni = 1.0 / d
    pp = jnp.sqrt(rr * tempi * jnp.maximum(z2b + zb, 1.0e-30))
    dppdt = 0.5 * rr * dtempidt * (z2b + zb) / pp
    dppdd = 0.5 * ytot * tempi * (z2b + zb) / pp
    qlam0z = 1.88e8 * tempi * pp
    qlam0zdt = 1.88e8 * (dtempidt * pp + tempi * dppdt)
    qlam0zdd = 1.88e8 * tempi * dppdd
    taufac = 4.248719e3 / 3.0 * (tempi**constants.ONE_THIRD)
    taufacdt = -constants.ONE_THIRD * taufac * tempi
    xni = (rr * zb) ** constants.ONE_THIRD
    dxnidd = constants.ONE_THIRD * xni * deni
    aa = 2.27493e5 * tempi * xni
    daadt = 2.27493e5 * dtempidt * xni
    daadd = 2.27493e5 * tempi * dxnidd

    z12 = z1_s + z2_s
    zs13 = z12**constants.ONE_THIRD
    zs13inv = 1.0 / zs13
    zhat = (z12 ** (5.0 / 3.0)) - (z1_s ** (5.0 / 3.0)) - (z2_s ** (5.0 / 3.0))
    zhat2 = (
        (z12 ** (5.0 / 12.0)) - (z1_s ** (5.0 / 12.0)) - (z2_s ** (5.0 / 12.0))
    )
    lzav = (5.0 / 3.0) * jnp.log(jnp.maximum(1.0e-99, z1_s * z2_s / z12))
    aznut = (z1_s * z1_s * z2_s * z2_s * a1 * a2 / (a1 + a2)) ** (
        constants.ONE_THIRD
    )

    bb = z1_s * z2_s
    qq = (2.0**constants.ONE_THIRD) * bb * zs13inv
    gamef_raw = qq * aa
    gamefdt_raw = qq * daadt
    gamefdd_raw = qq * daadd
    tau12 = taufac * aznut
    tau12dt = taufacdt * aznut
    alf12_raw = gamef_raw / tau12
    alf12dt_raw = (gamefdt_raw - alf12_raw * tau12dt) / tau12
    alf12dd_raw = gamefdd_raw / tau12

    cap = alf12_raw > 1.6
    alf12 = jnp.where(cap, 1.6, alf12_raw)
    alf12dt = jnp.where(cap, 0.0, alf12dt_raw)
    alf12dd = jnp.where(cap, 0.0, alf12dd_raw)
    gamef = jnp.where(cap, 1.6 * tau12, gamef_raw)
    gamefdt = jnp.where(cap, 1.6 * tau12dt, gamefdt_raw)
    gamefdd = jnp.where(cap, 0.0, gamefdd_raw)
    qq_inv = zs13 / ((2.0**constants.ONE_THIRD) * bb)
    gamp = jnp.where(cap, gamef * qq_inv, aa)
    gampdt = jnp.where(cap, gamefdt * qq_inv, daadt)
    gampdd = jnp.where(cap, 0.0, daadd)

    h12w = bb * qlam0z
    dh12wdt = bb * qlam0zdt
    dh12wdd = bb * qlam0zdd

    gamp14 = gamp**0.25
    cc = (
        0.896434 * gamp * zhat
        - 3.44740 * gamp14 * zhat2
        - 0.5551 * (jnp.log(gamp) + lzav)
        - 2.996
    )
    dccdgamp = (
        0.896434 * zhat - 0.25 * 3.44740 * gamp14 * zhat2 / gamp - 0.5551 / gamp
    )
    dccdt = dccdgamp * gampdt
    dccdd = dccdgamp * gampdd

    a3 = alf12 * alf12 * alf12
    da3 = 3.0 * alf12 * alf12
    qq_s = 0.014 + 0.0128 * alf12
    rr_s = (5.0 / 32.0) - alf12 * qq_s
    drrdalf = -(qq_s + alf12 * 0.0128)
    ss = tau12 * rr_s
    dssdt = tau12dt * rr_s + tau12 * drrdalf * alf12dt
    dssdd = tau12 * drrdalf * alf12dd

    tt = -0.0098 + 0.0048 * alf12
    uu = 0.0055 + alf12 * tt
    duudalf = tt + alf12 * 0.0048
    vv = gamef * alf12 * uu
    dvvdt = (
        gamefdt * alf12 + gamef * alf12dt
    ) * uu + gamef * alf12 * duudalf * alf12dt
    dvvdd = (
        gamefdd * alf12 + gamef * alf12dd
    ) * uu + gamef * alf12 * duudalf * alf12dd

    ww = ss + vv
    dwwdt = dssdt + dvvdt
    dwwdd = dssdd + dvvdd
    xx = a3 * ww
    dxxdt = da3 * alf12dt * ww + a3 * dwwdt
    dxxdd = da3 * alf12dd * ww + a3 * dwwdd

    xlgfac = jnp.maximum(0.77, 1.0 - 0.0562 * a3)
    fac_lg = jnp.where(xlgfac > 0.77, -0.0562 * da3 / xlgfac, 0.0)
    dxlgfacdt = fac_lg * alf12dt
    dxlgfacdd = fac_lg * alf12dd

    h12s = cc - xx + jnp.log(xlgfac)
    dh12sdt = dccdt - dxxdt + dxlgfacdt
    dh12sdd = dccdd - dxxdd + dxlgfacdd

    rr_wt = 2.0 * (0.8 - gamef)
    drrdt = -2.0 * gamefdt
    drrdd = -2.0 * gamefdd
    ss_wt = 2.0 * (gamef - 0.3)
    dss_dt = 2.0 * gamefdt
    dss_dd = 2.0 * gamefdd

    h12b = h12w * rr_wt + h12s * ss_wt
    dh12bdt = dh12wdt * rr_wt + h12w * drrdt + dh12sdt * ss_wt + h12s * dss_dt
    dh12bdd = dh12wdd * rr_wt + h12w * drrdd + dh12sdd * ss_wt + h12s * dss_dd

    h12_raw = jnp.where(gamef <= 0.3, h12w, jnp.where(gamef <= 0.8, h12b, h12s))
    dh12dt = jnp.where(
        gamef <= 0.3, dh12wdt, jnp.where(gamef <= 0.8, dh12bdt, dh12sdt)
    )
    dh12dd = jnp.where(
        gamef <= 0.3, dh12wdd, jnp.where(gamef <= 0.8, dh12bdd, dh12sdd)
    )

    h12 = jnp.clip(h12_raw, 0.0, 300.0)
    scor = jnp.exp(h12)
    dscor_dt = jnp.where(h12 == 300.0, 0.0, scor * dh12dt)
    dscor_dd = jnp.where(h12 == 300.0, 0.0, scor * dh12dd)
    z0 = jnp.zeros_like(t)
    return (
        jnp.where(uncharged, jnp.ones_like(t), scor),
        jnp.where(uncharged, z0, dscor_dt),
        jnp.where(uncharged, z0, dscor_dd),
    )


@jax.jit
def screen_chugunov_jax(
    temp: jax.Array,
    den: jax.Array,
    zbar: jax.Array,
    abar: jax.Array,
    z1: jax.Array,
    z2: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate Chugunov, DeWitt & Yakovlev (2007) plasma screening in JAX.

    Args:
        temp: Temperature $T$ in $\\mathrm{K}$ (`scalar` or `(N,)`).
        den: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        zbar: Mean ionic charge $\\bar{Z}$ (`scalar` or `(N,)`).
        abar: Mean atomic weight $\\bar{A}$ (`scalar` or `(N,)`).
        z1: Reactant 1 atomic number $Z_1$.
        z2: Reactant 2 atomic number $Z_2$.

    Returns:
        Tuple `(scor, dscor_dt, dscor_dd)` of screening enhancement factor and
        its partial derivatives with respect to $T$ and $\\rho$.
    """
    t = jnp.asarray(temp, dtype=jnp.float64)
    d = jnp.asarray(den, dtype=jnp.float64)
    zb_raw = jnp.asarray(zbar, dtype=jnp.float64)
    ab = jnp.asarray(abar, dtype=jnp.float64)
    z1_raw = jnp.asarray(z1, dtype=jnp.float64)
    z2_raw = jnp.asarray(z2, dtype=jnp.float64)
    uncharged = (zb_raw <= 0.0) | (z1_raw <= 0.0) | (z2_raw <= 0.0)

    zb = jnp.maximum(zb_raw, 1.0e-30)
    z1_s = jnp.maximum(z1_raw, 1.0)
    z2_s = jnp.maximum(z2_raw, 1.0)

    mav = ab * constants.AMU
    ntot = d / mav
    ae = (3.0 / (4.0 * constants.PI * ntot * zb)) ** constants.ONE_THIRD
    r1 = ae * (jnp.floor(z1_s) ** constants.ONE_THIRD)
    r2 = ae * (jnp.floor(z2_s) ** constants.ONE_THIRD)
    rav = 0.5 * (r1 + r2)
    dravdd = -rav / (3.0 * d)

    w = jnp.sqrt(
        4.0
        * constants.PI
        * z1_s
        * z2_s
        * constants.QE
        * constants.QE
        * ntot
        / mav
    )
    dwdd = 0.5 * w / d
    tp = constants.HBAR * w / constants.BOLTZM
    dtpdd = (constants.HBAR / constants.BOLTZM) * dwdd
    tn = t / tp
    dtndt = 1.0 / tp
    dtndd = -tn * dtpdd / tp

    alfa = 0.5 * (1.0 - jnp.cos(constants.PI * (tn - 0.1) / 0.1))
    dalfadtn = (
        0.5 * constants.PI * jnp.sin(constants.PI * (tn - 0.1) / 0.1) / 0.1
    )
    dalfadt = dalfadtn * dtndt
    dalfadd = dalfadtn * dtndd
    beta = 1.0 - alfa
    tk_mid = (beta * 0.1 + alfa * 0.2) * tp
    dtkdt_mid = (-dalfadt * 0.1 + dalfadt * 0.2) * tp
    dtkdd_mid = (-dalfadd * 0.1 + dalfadd * 0.2) * tp + (
        beta * 0.1 + alfa * 0.2
    ) * dtpdd
    denom_mid = tp * (tk_mid * dtpdd - tp * dtkdd_mid) / (tk_mid**3.0)

    tk = jnp.where(tn > 0.2, t, jnp.where(tn <= 0.1, 0.1 * tp, tk_mid))
    dtkdt = jnp.where(tn > 0.2, 1.0, jnp.where(tn <= 0.1, 0.0, dtkdt_mid))
    dtkdd = jnp.where(
        tn > 0.2, 0.0, jnp.where(tn <= 0.1, 0.1 * dtpdd, dtkdd_mid)
    )
    denom = jnp.where(
        tn > 0.2,
        tp * t * dtpdd / (t**3.0),
        jnp.where(tn <= 0.1, 0.0, denom_mid),
    )

    u_zeta = 4.0 * tp * tp / (3.0 * constants.PI * constants.PI * tk * tk)
    zeta = u_zeta**constants.ONE_THIRD
    dzetadt = -2.0 * zeta * dtkdt / (3.0 * tk)
    dzetadd = (
        (zeta / (3.0 * u_zeta))
        * (8.0 / (3.0 * constants.PI * constants.PI))
        * denom
    )

    gam_raw = (
        z1_s
        * z2_s
        * constants.QE
        * constants.QE
        / (rav * constants.BOLTZM * tk)
    )
    dgamdt_raw = -gam_raw * dtkdt / tk
    dgamdd_raw = -gam_raw * (tk * dravdd + rav * dtkdd) / (rav * tk)

    alfa_g = 0.5 * (1.0 - jnp.cos(constants.PI * (gam_raw - 590.0) / 10.0))
    dalfadgam = (
        0.5
        * constants.PI
        * jnp.sin(constants.PI * (gam_raw - 590.0) / 10.0)
        / 10.0
    )
    dalfadt_g = dalfadgam * dgamdt_raw
    dalfadd_g = dalfadgam * dgamdd_raw
    beta_g = 1.0 - alfa_g
    gam_mid = beta_g * gam_raw + alfa_g * 600.0
    dgamdt_mid = -dalfadt_g * gam_raw + beta_g * dgamdt_raw + dalfadt_g * 600.0
    dgamdd_mid = -dalfadd_g * gam_raw + beta_g * dgamdd_raw + dalfadd_g * 600.0
    gam = jnp.where(
        gam_raw >= 600.0,
        600.0,
        jnp.where(gam_raw >= 590.0, gam_mid, gam_raw),
    )
    dgamdt = jnp.where(
        gam_raw >= 600.0,
        0.0,
        jnp.where(gam_raw >= 590.0, dgamdt_mid, dgamdt_raw),
    )
    dgamdd = jnp.where(
        gam_raw >= 600.0,
        0.0,
        jnp.where(gam_raw >= 590.0, dgamdd_mid, dgamdd_raw),
    )

    a_z = 0.022
    b_z = 0.41 - 0.6 / gam
    db_z_dt = 0.6 * dgamdt / (gam * gam)
    db_z_dd = 0.6 * dgamdd / (gam * gam)
    g_z = 0.06 + 2.2 / gam
    dg_z_dt = -2.2 * dgamdt / (gam * gam)
    dg_z_dd = -2.2 * dgamdd / (gam * gam)

    u = 1.0 + a_z * zeta + b_z * zeta * zeta + g_z * (zeta**3.0)
    dudt = (
        a_z * dzetadt
        + db_z_dt * zeta * zeta
        + 2.0 * b_z * zeta * dzetadt
        + dg_z_dt * (zeta**3.0)
        + 3.0 * g_z * zeta * zeta * dzetadt
    )
    dudd = (
        a_z * dzetadd
        + db_z_dd * zeta * zeta
        + 2.0 * b_z * zeta * dzetadd
        + dg_z_dd * (zeta**3.0)
        + 3.0 * g_z * zeta * zeta * dzetadd
    )
    u13 = u**constants.ONE_THIRD
    tgam = gam / u13
    dtgamdt = (dgamdt - constants.ONE_THIRD * gam * dudt / u) / u13
    dtgamdd = (dgamdd - constants.ONE_THIRD * gam * dudd / u) / u13

    ca1, ca2 = 2.7822, 98.34
    ca3 = jnp.sqrt(3.0) - ca1 / jnp.sqrt(ca2)
    cb1, cb2, cb3, cb4 = -1.7476, 66.07, 1.12, 65.0

    t1 = ca1 / jnp.sqrt(ca2 + tgam)
    dt1dt = -0.5 * ca1 * dtgamdt / ((ca2 + tgam) ** 1.5)
    dt1dd = -0.5 * ca1 * dtgamdd / ((ca2 + tgam) ** 1.5)
    t2 = ca3 / (1.0 + tgam)
    dt2dt = -ca3 * dtgamdt / ((1.0 + tgam) ** 2.0)
    dt2dd = -ca3 * dtgamdd / ((1.0 + tgam) ** 2.0)
    tg32 = tgam**1.5
    dtg32dt = 1.5 * jnp.sqrt(tgam) * dtgamdt
    dtg32dd = 1.5 * jnp.sqrt(tgam) * dtgamdd
    t3 = tg32 * (t1 + t2)
    dt3dt = dtg32dt * (t1 + t2) + tg32 * (dt1dt + dt2dt)
    dt3dd = dtg32dd * (t1 + t2) + tg32 * (dt1dd + dt2dd)
    tg2 = tgam * tgam
    dtg2dt = 2.0 * tgam * dtgamdt
    dtg2dd = 2.0 * tgam * dtgamdd
    t4 = cb1 * tg2 / (cb2 + tgam)
    dt4dt = cb1 * (dtg2dt - tg2 * dtgamdt / (cb2 + tgam)) / (cb2 + tgam)
    dt4dd = cb1 * (dtg2dd - tg2 * dtgamdd / (cb2 + tgam)) / (cb2 + tgam)
    t5 = cb3 * tg2 / (cb4 + tg2)
    dt5dt = cb3 * (dtg2dt - tg2 * dtg2dt / (cb4 + tg2)) / (cb4 + tg2)
    dt5dd = cb3 * (dtg2dd - tg2 * dtg2dd / (cb4 + tg2)) / (cb4 + tg2)

    h0 = jnp.minimum(t3 + t4 + t5, 300.0)
    scor = jnp.exp(h0)
    dscor_dt = jnp.where(h0 >= 300.0, 0.0, scor * (dt3dt + dt4dt + dt5dt))
    dscor_dd = jnp.where(h0 >= 300.0, 0.0, scor * (dt3dd + dt4dd + dt5dd))
    z0 = jnp.zeros_like(t)
    return (
        jnp.where(uncharged, jnp.ones_like(t), scor),
        jnp.where(uncharged, z0, dscor_dt),
        jnp.where(uncharged, z0, dscor_dd),
    )
