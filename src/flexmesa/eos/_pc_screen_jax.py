# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branchless JAX `float64` kernels for `PC` electron-ion screening and mixing.

Ports `FSCRliq8`, `FSCRsol8`, `CORMIX`, and `EOSFI8` (`eos/private/pc_eos.f90`
and `crates/flexmesa-eos/src/pc/{coulomb_screen.rs, eosfi.rs}`).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from flexmesa.eos import _pc_coulomb_jax


class PcContrib(NamedTuple):
    """Thermodynamic contribution tuple `(f, u, p, s, cv, pdt, pdr)`."""

    f: jax.Array
    u: jax.Array
    p: jax.Array
    s: jax.Array
    cv: jax.Array
    pdt: jax.Array
    pdr: jax.Array


def fscr_liq8_jax(
    rs_in: jax.Array, game_in: jax.Array, zion_in: jax.Array
) -> PcContrib:
    """Evaluate `FSCRliq8(RS, GAME, Zion)` (`pc_eos.f90` lines 781-960)."""
    xrs = 0.0140047
    tiny = 1.0e-19
    rs = jnp.maximum(rs_in, tiny)
    game = jnp.maximum(game_in, 1.0e-30)
    zion = jnp.maximum(zion_in, 1.0)

    sqg = jnp.sqrt(game)
    sqz1 = jnp.sqrt(1.0 + zion)
    sqz = jnp.sqrt(zion)
    cdh0 = zion / 1.73205
    cdh = cdh0 * (sqz1 * sqz1 * sqz1 - sqz * sqz * sqz - 1.0)
    zln = jnp.log(zion)
    z13 = jnp.exp(zln / 3.0)
    x = xrs / rs
    ctf = zion * zion * 0.2513 * (z13 - 1.0 + 0.2 / jnp.sqrt(z13))
    p01 = 1.11 * jnp.exp(0.475 * zln)
    p03 = 0.2 + 0.078 * zln * zln
    ptx = 1.16 + 0.08 * zln
    tx = jnp.power(game, ptx)
    txdg = tx * ptx / game
    txdgg = txdg * (ptx - 1.0) / game
    ty1 = 1.0 / (game * 2.0 + 1.0e-3 * zion * zion)
    ty1dg = -ty1 * ty1 * 2.0
    ty1dgg = -ty1 * ty1dg * 4.0
    rs2 = rs * rs
    rs3 = rs2 * rs
    ty2 = rs2 * 6.0 + 1.0
    ty2dx = -rs2 * 12.0 / x
    ty2dxx = -ty2dx * 3.0 / x
    ty = rs3 / ty2 * (ty1 + 1.0)
    tyx = 3.0 / x + ty2dx / ty2
    tydx = -ty * tyx
    tydg = rs3 * ty1dg / ty2
    p1 = (zion - 1.0) / 9.0
    cor1 = ty * p1 + 1.0
    cor1dx = tydx * p1
    cor1dg = tydg * p1
    cor1dxx = (
        ty * (3.0 / (x * x) + (ty2dx / ty2) ** 2 - ty2dxx / ty2) - tydx * tyx
    ) * p1
    cor1dgg = rs3 * ty1dgg / ty2 * p1
    cor1dxg = -tydg * tyx * p1
    u0 = jnp.sqrt(game / zion) * rs3 * 0.78
    u0dx = -u0 * 3.0 / x
    u0dg = u0 * 0.5 / game
    u0dxx = -u0dx * 4.0 / x
    u0dgg = -u0dg * 0.5 / game
    u0dxg = -u0dg * 3.0 / x
    d0dg = zion * zion * zion
    d0 = game * d0dg + rs3 * 21.0
    d0dx = -rs3 * 63.0 / x
    d0dxx = rs3 * 252.0 / (x * x)
    cor0 = u0 / d0 + 1.0
    cor0dx = (u0dx - u0 * d0dx / d0) / d0
    cor0dg = (u0dg - u0 * d0dg / d0) / d0
    cor0dxx = (
        u0dxx - (u0dx * d0dx * 2.0 + u0 * d0dxx) / d0 + ((d0dx / d0) ** 2) * 2.0
    ) / d0
    cor0dgg = (
        u0dgg - u0dg * (2.0 * d0dg) / d0 + u0 * 2.0 * (d0dg / d0) * (d0dg / d0)
    ) / d0
    cor0dxg = (
        u0dxg
        - (u0dx * d0dg + u0dg * d0dx) / d0
        + u0 * d0dx * (2.0 * d0dg) / (d0 * d0)
    ) / d0
    rele = jnp.sqrt(x * x + 1.0)
    q1 = 0.18 / jnp.sqrt(sqz)
    q2 = 0.2 + 0.37 / sqz
    h1u = x * x / 5.0 + 1.0
    h1d = x * q1 + x * x * q2 + 1.0
    h1 = h1u / h1d
    h1x = x * 0.4 / h1u - (x * (2.0 * q2) + q1) / h1d
    h1dx = h1 * h1x
    h1dxx = h1dx * h1x + h1 * (
        0.4 / h1u
        - (x * 0.4 / h1u) ** 2
        - (2.0 * q2) / h1d
        + ((x * (2.0 * q2) + q1) / h1d) ** 2
    )
    up = sqg * cdh + tx * cor0 * h1 * (p01 * ctf)
    updx = tx * (cor0dx * h1 + cor0 * h1dx) * (p01 * ctf)
    updg = (0.5 * cdh) / sqg + (txdg * cor0 + tx * cor0dg) * h1 * (p01 * ctf)
    updxx = (
        tx * (cor0dxx * h1 + cor0dx * h1dx * 2.0 + cor0 * h1dxx) * (p01 * ctf)
    )
    updgg = -(0.25 * cdh) / (sqg * game) + (
        txdgg * cor0 + txdg * cor0dg * 2.0 + tx * cor0dgg
    ) * h1 * (p01 * ctf)
    updxg = (
        txdg * (cor0dx * h1 + cor0 * h1dx) + tx * (cor0dxg * h1 + cor0dg * h1dx)
    ) * (p01 * ctf)
    dn1 = sqg * p03 + tx * cor1 * p01 / rs
    dn1dx = tx * (cor1 / xrs + cor1dx / rs) * p01
    dn1dg = (0.5 * p03) / sqg + (txdg * cor1 + tx * cor1dg) * p01 / rs
    dn1dxx = tx * (cor1dx * 2.0 + x * cor1dxx) * (p01 / xrs)
    dn1dgg = (
        -(0.25 * p03) / (game * sqg)
        + (txdgg * cor1 + txdg * cor1dg * 2.0 + tx * cor1dgg) * p01 / rs
    )
    dn1dxg = (
        txdg * (cor1 / xrs + cor1dx / rs) + tx * (cor1dg / xrs + cor1dxg / rs)
    ) * p01
    rele2 = rele * rele
    rele3 = rele2 * rele
    dn = dn1 / rele + 1.0
    dndx = dn1dx / rele - x * dn1 / rele3
    dndxx = (
        dn1dxx - (x * dn1dx * 2.0 + dn1 - x * x * dn1 * 3.0 / rele2) / rele2
    ) / rele
    dndg = dn1dg / rele
    dndgg = dn1dgg / rele
    dndxg = dn1dxg / rele - x * dn1dg / rele3
    fscr = -up / dn * game
    fx = (up * dndx / dn - updx) / dn
    fxdg = (
        (updg * dndx + updx * dndg + up * dndxg - up * dndx * dndg * 2.0 / dn)
        / dn
        - updxg
    ) / dn
    fdx = fx * game
    fg = (up * dndg / dn - updg) / dn
    fdg = fg * game - up / dn
    fdxx = (
        (
            (up * dndxx + (updx * dndx - up * dndx * dndx / dn) * 2.0) / dn
            - updxx
        )
        / dn
        * game
    )
    fdgg = (
        fg * 2.0
        + game
        * ((dndg * (updg - up * dndg / dn) * 2.0 + up * dndgg) / dn - updgg)
        / dn
    )
    fdxg = fx + game * fxdg
    uscr = game * fdg
    cvscr = -game * game * fdgg
    pscr = (x * fdx + game * fdg) / 3.0
    pdtscr = -game * game * (x * fxdg + fdgg) / 3.0
    pdrscr = (
        pscr * 12.0 + x * x * fdxx + x * game * fdxg * 2.0 + game * game * fdgg
    ) / 9.0
    sscr = uscr - fscr
    active = rs_in >= tiny
    z0 = jnp.zeros_like(rs_in)
    return PcContrib(
        f=jnp.where(active, fscr, z0),
        u=jnp.where(active, uscr, z0),
        p=jnp.where(active, pscr, z0),
        s=jnp.where(active, sscr, z0),
        cv=jnp.where(active, cvscr, z0),
        pdt=jnp.where(active, pdtscr, z0),
        pdr=jnp.where(active, pdrscr, z0),
    )


def fscr_sol8_jax(
    rs_in: jax.Array,
    gami_in: jax.Array,
    zion_in: jax.Array,
    tpt_in: jax.Array,
) -> PcContrib:
    """Evaluate `FSCRsol8(RS, GAMI, Zion, TPT)` (`pc_eos.f90` 1034-1200)."""
    ap0, ap1, ap2, ap3 = 1.1866, 0.684, 17.9, 41.5
    enat = 2.7182818285
    tiny = 1.0e-19
    px = 0.205
    rs = jnp.maximum(rs_in, tiny)
    gami = jnp.maximum(gami_in, 1.0e-30)
    zion = jnp.maximum(zion_in, 1.0)
    tpt = jnp.maximum(tpt_in, 1.0e-30)

    xsr = 0.0140047 / rs
    z13 = jnp.power(zion, 1.0 / 3.0)
    p1 = 0.00352 * (1.0 - ap0 / jnp.power(zion, 0.267) + 0.27 / zion)
    z5 = zion**5
    z6 = z5 * zion
    p2 = 1.0 + 2.25 / z13 * (1.0 + ap1 * z5 + 0.222 * z6) / (1.0 + 0.222 * z6)
    zln = jnp.log(zion)
    xsr2 = xsr * xsr
    finf = jnp.sqrt(p2 / xsr2 + 1.0) * (z13 * z13 * p1)
    finfx = -p2 / ((xsr2 + p2) * xsr)
    finfdx = finf * finfx
    finfdxx = finfdx * finfx - finfdx * (xsr2 * 3.0 + p2) / ((xsr2 + p2) * xsr)
    r1 = ap3 / (1.0 + zln)
    r2 = 0.395 * zln + 0.347 / zion / jnp.sqrt(zion)
    r3 = 1.0 / (1.0 + zln * jnp.sqrt(zln) * 0.01 + 0.097 / (zion * zion))
    q1u = xsr2 * ap2 + r1
    q1d = xsr2 * r2 + 1.0
    q1 = q1u / q1d
    q1x = xsr * 2.0 * (ap2 / q1u - r2 / q1d)
    q1xdx = q1x / xsr + xsr2 * 4.0 * ((r2 / q1d) ** 2 - (ap2 / q1u) ** 2)
    q1dx = q1 * q1x
    q1dxx = q1dx * q1x + q1 * q1xdx

    tpt_lo = jnp.minimum(tpt, 6.0 / px)
    px_tpt = tpt_lo * px
    y0 = px_tpt * px_tpt
    y0dx = y0 / xsr
    y0dg = y0 * 2.0 / gami
    y0dgg = y0dg / gami
    y0dxg = y0dg / xsr
    y1 = jnp.exp(y0)
    y1dx = y1 * y0dx
    y1dg = y1 * y0dg
    y1dxx = y1 * (y0dx * y0dx)
    y1dgg = y1 * (y0dg * y0dg + y0dgg)
    y1dxg = y1 * (y0dx * y0dg + y0dxg)
    sa = y1 + 1.0
    supa = jnp.log(sa)
    supadx = y1dx / sa
    supadg = y1dg / sa
    supadxx = (y1dxx - y1dx * y1dx / sa) / sa
    supadgg = (y1dgg - y1dg * y1dg / sa) / sa
    supadxg = (y1dxg - y1dx * y1dg / sa) / sa
    em2 = enat - 2.0
    sb = -em2 / y1 + enat
    supb = jnp.log(sb)
    em2y1 = em2 / (y1 * y1 * sb)
    supbdx = em2y1 * y1dx
    supbdg = em2y1 * y1dg
    supbdxx = em2y1 * (y1dxx - y1dx * y1dx * 2.0 / y1 - y1dx * supbdx)
    supbdgg = em2y1 * (y1dgg - y1dg * y1dg * 2.0 / y1 - y1dg * supbdg)
    supbdxg = em2y1 * (y1dxg - y1dx * y1dg * 2.0 / y1 - y1dg * supbdx)
    sup_l = jnp.sqrt(supa / supb)
    supx_l = (supadx / supa - supbdx / supb) * 0.5
    supdx_l = sup_l * supx_l
    supg_l = (supadg / supa - supbdg / supb) * 0.5
    supdg_l = sup_l * supg_l
    supdxx_l = supdx_l * supx_l + sup_l * 0.5 * (
        supadxx / supa
        - (supadx / supa) ** 2
        - supbdxx / supb
        + (supbdx / supb) ** 2
    )
    supdgg_l = supdg_l * supg_l + sup_l * 0.5 * (
        supadgg / supa
        - (supadg / supa) ** 2
        - supbdgg / supb
        + (supbdg / supb) ** 2
    )
    supdxg_l = supdx_l * supg_l + sup_l * 0.5 * (
        (supadxg - supadx * supadg / supa) / supa
        - (supbdxg - supbdx * supbdg / supb) / supb
    )

    sup_h = tpt * px
    supdx_h = tpt * (0.5 * px) / xsr
    supdg_h = tpt * px / gami
    supdxx_h = -supdx_h * 0.5 / xsr
    supdgg_h = jnp.zeros_like(rs)
    supdxg_h = supdx_h / gami

    cond_tpt = tpt < (6.0 / px)
    sup = jnp.where(cond_tpt, sup_l, sup_h)
    supdx = jnp.where(cond_tpt, supdx_l, supdx_h)
    supdg = jnp.where(cond_tpt, supdg_l, supdg_h)
    supdxx = jnp.where(cond_tpt, supdxx_l, supdxx_h)
    supdgg = jnp.where(cond_tpt, supdgg_l, supdgg_h)
    supdxg = jnp.where(cond_tpt, supdxg_l, supdxg_h)

    gr3 = jnp.power(gami / sup, r3)
    gr3x = -supdx / sup * r3
    gr3dx = gr3 * gr3x
    gr3dxx = gr3dx * gr3x - gr3 * (supdxx / sup - (supdx / sup) ** 2) * r3
    gr3g = (1.0 / gami - supdg / sup) * r3
    gr3dg = gr3 * gr3g
    gr3dgg = (
        gr3dg * gr3g
        + gr3 * ((supdg / sup) ** 2 - supdgg / sup - 1.0 / (gami * gami)) * r3
    )
    gr3dxg = (
        gr3dg * gr3x + gr3 * (supdx * supdg / (sup * sup) - supdxg / sup) * r3
    )

    gr3_2 = gr3 * gr3
    gr3_3 = gr3_2 * gr3
    w = q1 / gr3 + 1.0
    wdx = q1dx / gr3 - q1 * gr3dx / gr3_3
    wdg = -q1 * gr3dg / gr3_2
    wdxx = (
        q1dxx / gr3
        - (q1dx * gr3dx * 2.0 + q1 * (gr3dxx - gr3dx * gr3dx * 2.0 / gr3))
        / gr3_2
    )
    wdgg = q1 * (gr3dg * gr3dg * 2.0 / gr3 - gr3dgg) / gr3_2
    wdxg = -(q1dx * gr3dg + q1 * (gr3dxg - gr3dx * gr3dg * 2.0 / gr3)) / gr3_2

    fscr = -gami * finf * w
    fdx = -gami * (finfdx * w + finf * wdx)
    fdxx = -gami * (finfdxx * w + finfdx * wdx * 2.0 + finf * wdxx)
    fdg = -finf * w - gami * finf * wdg
    fdgg = -finf * wdg * 2.0 - gami * finf * wdgg
    fdxg = -finfdx * w - finf * wdx - gami * (finfdx * wdg + finf * wdxg)
    s_scr = -gami * gami * finf * wdg
    uscr = s_scr + fscr
    cvscr = -gami * gami * fdgg
    pscr = (xsr * fdx + gami * fdg) / 3.0
    pdtscr = gami * gami * (xsr * finf * (finfx * wdg + wdxg) - fdgg) / 3.0
    pdrscr = (
        pscr * 12.0
        + xsr * xsr * fdxx
        + xsr * gami * fdxg * 2.0
        + gami * gami * fdgg
    ) / 9.0
    active = rs_in >= tiny
    z0 = jnp.zeros_like(rs_in)
    return PcContrib(
        f=jnp.where(active, fscr, z0),
        u=jnp.where(active, uscr, z0),
        p=jnp.where(active, pscr, z0),
        s=jnp.where(active, s_scr, z0),
        cv=jnp.where(active, cvscr, z0),
        pdt=jnp.where(active, pdtscr, z0),
        pdr=jnp.where(active, pdrscr, z0),
    )


def eosfi8_c2_pair_jax(
    cmi: jax.Array, zion: jax.Array, rs: jax.Array, gami: jax.Array
) -> tuple[PcContrib, PcContrib]:
    """Evaluate `EOSFI8` (`pc_eos.f90` lines 484-627) for `(c2_liq, c2_sol)`."""
    aum = 1822.888
    zion_s = jnp.maximum(zion, 1.0)
    cmi_s = jnp.maximum(cmi, 1.0)
    game = gami / jnp.power(zion_s, 5.0 / 3.0)
    xc = PcContrib(*_pc_coulomb_jax.excor7_jax(rs, game))
    cotpt = jnp.sqrt(3.0 / aum / cmi_s) / jnp.power(zion_s, 7.0 / 6.0)
    tpt = gami / jnp.sqrt(rs) * cotpt
    fid_ion = jnp.log(tpt * tpt / gami) * 1.5 - 1.323515

    ii = PcContrib(*_pc_coulomb_jax.fition9_jax(gami))
    fitot_l = ii.f + fid_ion
    uitot_l = ii.u + 1.5
    pitot_l = ii.p + 1.0
    cvitot_l = ii.cv + 1.5
    scitot_l = uitot_l - fitot_l
    pdti_l = ii.pdt + 1.0
    pdri_l = ii.pdr + 1.0
    scr_l = fscr_liq8_jax(rs, game, zion_s)

    fitot_s, uitot_s, pitot_s, scitot_s, cvitot_s, pdti_s, pdri_s = (
        _pc_coulomb_jax.fharm12_anharm8_jax(gami, tpt)
    )
    scr_s = fscr_sol8_jax(rs, gami, zion_s, tpt)

    c2_liq = PcContrib(
        f=fitot_l + scr_l.f + xc.f * zion_s,
        u=uitot_l + scr_l.u + xc.u * zion_s,
        p=pitot_l + scr_l.p + xc.p * zion_s,
        s=scitot_l + scr_l.s + xc.s * zion_s,
        cv=cvitot_l + scr_l.cv + xc.cv * zion_s,
        pdt=pdti_l + scr_l.pdt + xc.pdt * zion_s,
        pdr=pdri_l + scr_l.pdr + xc.pdr * zion_s,
    )
    c2_sol = PcContrib(
        f=fitot_s + scr_s.f + xc.f * zion_s,
        u=uitot_s + scr_s.u + xc.u * zion_s,
        p=pitot_s + scr_s.p + xc.p * zion_s,
        s=scitot_s + scr_s.s + xc.s * zion_s,
        cv=cvitot_s + scr_s.cv + xc.cv * zion_s,
        pdt=pdti_s + scr_s.pdt + xc.pdt * zion_s,
        pdr=pdri_s + scr_s.pdr + xc.pdr * zion_s,
    )
    return c2_liq, c2_sol


def cormix_jax(
    rs: jax.Array,
    game: jax.Array,
    zmean: jax.Array,
    z2mean: jax.Array,
    z52: jax.Array,
    z53: jax.Array,
    z321: jax.Array,
) -> PcContrib:
    """Evaluate `CORMIX` in branchless JAX (`pc_eos.f90` lines 1414-1476)."""
    tiny = 1.0e-9
    d = z2mean / (zmean * zmean)
    active = jnp.abs(d - 1.0) >= tiny
    d_safe = jnp.where(active, d, 1.0 + 1.0e-4)
    gami_mean = game * z53
    zm = z2mean + zmean
    dif0 = jnp.where(
        rs < tiny,
        z52 - jnp.sqrt(z2mean * z2mean * z2mean / zmean),
        z321 - jnp.sqrt(zm * zm * zm / zmean),
    )
    dif_r = dif0 / z52
    dif_fdh = game * jnp.sqrt(game / 3.0) * dif0
    p3 = jnp.power(d_safe, -0.2)
    d0 = (2.6 * dif_r + 14.0 * dif_r * dif_r * dif_r) / (1.0 - p3)
    gp = jnp.power(gami_mean, p3) * d0
    fmix0 = dif_fdh / (gp + 1.0)
    q = d_safe * d_safe * 0.0117
    r = 1.5 / p3 - 1.0
    gq = gp * q
    fmix = fmix0 / jnp.power(gq + 1.0, r)
    g = -gp * p3 / (gp + 1.0) - gq * (r * p3) / (gq + 1.0) + 1.5
    umix = fmix * g
    pmix = umix / 3.0
    gdg = -(gp / ((gp + 1.0) ** 2) + gq * r / ((gq + 1.0) ** 2)) * (p3 * p3)
    udg = umix * g + fmix * gdg
    cvmix = umix - udg
    pdtmix = pmix - udg / 3.0
    pdrmix = pmix + udg / 9.0
    smix = umix - fmix
    z0 = jnp.zeros_like(rs)
    return PcContrib(
        f=jnp.where(active, fmix, z0),
        u=jnp.where(active, umix, z0),
        p=jnp.where(active, pmix, z0),
        s=jnp.where(active, smix, z0),
        cv=jnp.where(active, cvmix, z0),
        pdt=jnp.where(active, pdtmix, z0),
        pdr=jnp.where(active, pdrmix, z0),
    )
