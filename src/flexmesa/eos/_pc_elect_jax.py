# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branchless JAX `float64` kernels for `PC` ideal relativistic electrons.

Ports `FERMI10`, `FERINV7`, `CHEMFIT`, `BLIN9` (`BLIN9a`, `BLIN9b`, `BLIN9c`),
and `ELECT11` (`ELECT11a`, `ELECT11b`) from `eos/private/pc_eos.f90` and
`crates/flexmesa-eos/src/pc/{ferinv.rs, blin.rs, blin_c.rs, elect.rs}`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from flexmesa import constants

_PI26: float = constants.PI * constants.PI / 6.0

_FERINV1_A: tuple[float, ...] = (
    1.999_266_880_833e4,
    5.702_479_099_336e3,
    6.610_132_843_877e2,
    3.818_838_129_486e1,
    1.0,
)
_FERINV1_B: tuple[float, ...] = (
    1.771_804_140_488e4,
    -2.014_785_161_019e3,
    9.130_355_392_717e1,
    -1.670_718_177_489,
)
_FERINV1_C: tuple[float, ...] = (
    -1.277_060_388_085e-2,
    7.187_946_804_945e-2,
    -4.262_314_235_106e-1,
    4.997_559_426_872e-1,
    -1.285_579_118_012,
    -3.930_805_454_272e-1,
    1.0,
)
_FERINV1_D: tuple[float, ...] = (
    -9.745_794_806_288e-3,
    5.485_432_756_838e-2,
    -3.299_466_243_26e-1,
    4.077_841_975_923e-1,
    -1.145_531_476_975,
    -6.067_091_689_181e-2,
)

_BLIN9A_AC: tuple[tuple[float, ...], ...] = (
    (0.370_450_57, 0.412_584_37, 9.777_982e-2, 5.373_415_3e-3, 3.874_628_1e-5),
    (0.396_031_09, 0.694_687_95, 0.223_227_60, 1.526_293_4e-2, 1.308_193_9e-4),
    (0.769_346_19, 1.789_143_7, 0.707_549_74, 5.675_567_2e-2, 5.557_148_0e-4),
)
_BLIN9A_AU: tuple[tuple[float, ...], ...] = (
    (0.431_398_81, 1.759_753_7, 4.104_465_4, 7.746_703_8, 13.457_678),
    (0.817_631_76, 2.472_333_9, 5.116_006_1, 9.044_146_5, 15.049_882),
    (1.255_846_1, 3.207_040_6, 6.123_908_2, 10.316_126, 16.597_079),
)
_BLIN9B_AX: tuple[float, ...] = (
    7.265_351e-2,
    0.269_460_8,
    0.533_122,
    0.786_880_1,
    0.956_931_3,
)
_BLIN9B_AXI: tuple[float, ...] = (
    0.263_560_32,
    1.413_403_1,
    3.596_425_8,
    7.085_810_0,
    12.640_801,
)
_BLIN9B_AH: tuple[float, ...] = (
    3.818_735e-2,
    0.125_673_2,
    0.198_630_8,
    0.197_633_4,
    0.106_542_0,
)
_BLIN9B_AV: tuple[float, ...] = (
    0.295_058_69,
    0.320_648_56,
    7.391_557_0e-2,
    3.608_738_9e-3,
    2.336_989_4e-5,
)


class BlinSeries1(NamedTuple):
    """First-order Blinnikov series `(w, wdx, wdt)`."""

    w: jax.Array
    wdx: jax.Array
    wdt: jax.Array


class Elect11Result(NamedTuple):
    """Ideal relativistic electron gas thermodynamic quantities (`ELECT11`)."""

    dens: jax.Array
    fe_id: jax.Array
    pe_id: jax.Array
    ue_id: jax.Array
    se_id: jax.Array
    cve: jax.Array
    chite: jax.Array
    chire: jax.Array


def fermi10_jax(x: jax.Array, xmax: float) -> tuple[jax.Array, jax.Array]:
    """Evaluate `FERMI10(x, xmax)` in JAX (`pc_eos.f90` lines 1818-1841)."""
    xm = max(float(xmax), 3.0)
    x_safe = jnp.clip(x, -xm, xm)
    fp_mid = 1.0 / (jnp.exp(x_safe) + 1.0)
    fp = jnp.where(x > xm, 0.0, jnp.where(x < -xm, 1.0, fp_mid))
    return fp, 1.0 - fp


def ferinv7_n1_jax(f_in: jax.Array) -> jax.Array:
    """Evaluate `FERINV7(f, n=1)` in JAX (`pc_eos.f90` lines 2165-2304)."""
    f = jnp.maximum(f_in, 1.0e-99)
    t_lo = jnp.minimum(f, 4.0)
    up_lo = jnp.zeros_like(f)
    for ai in reversed(_FERINV1_A):
        up_lo = up_lo * t_lo + ai
    down_lo = jnp.zeros_like(f)
    for bi in reversed(_FERINV1_B):
        down_lo = down_lo * t_lo + bi
    x_lo = jnp.log(t_lo * up_lo / down_lo)

    f_hi = jnp.maximum(f, 4.0)
    t_hi = jnp.power(f_hi, -1.0 / 1.5)
    up_hi = jnp.zeros_like(f)
    for ci in reversed(_FERINV1_C):
        up_hi = up_hi * t_hi + ci
    down_hi = jnp.zeros_like(f)
    for di in reversed(_FERINV1_D):
        down_hi = down_hi * t_hi + di
    x_hi = (up_hi / down_hi) / t_hi
    return jnp.where(f < 4.0, x_lo, x_hi)


def chemfit_jax(dens: jax.Array, temp: jax.Array) -> jax.Array:
    """Evaluate `CHEMFIT(dens, temp)` (`pc_eos.f90` lines 2889-2975)."""
    denr = dens / 2.573_380_6e6
    temr = temp / 1.877_886_5e4
    pf0 = jnp.power(29.608_813_2 * denr, 1.0 / 3.0)
    tf = jnp.where(
        pf0 > 1.0e-4, jnp.sqrt(1.0 + pf0 * pf0) - 1.0, 0.5 * pf0 * pf0
    )
    theta = temr / tf
    theta32 = theta * jnp.sqrt(theta)
    q2 = 12.0 + 8.0 / theta32
    sqt = jnp.sqrt(temr)
    t1 = jnp.exp(-jnp.minimum(theta, 700.0))
    t2 = t1 * t1
    u3 = t2 + 1.612
    d3 = 6.192 * jnp.power(theta, 0.0944) * t2 + 5.535 * jnp.power(theta, 0.698)
    q3 = 1.365_568_127 - u3 / d3
    th_safe = jnp.maximum(theta, 1.0e-5)
    t1_safe = jnp.exp(-jnp.minimum(th_safe, 700.0))
    q1 = jnp.where(theta > 1.0e-5, 1.5 * t1_safe / (1.0 - t1_safe), 1.5 / theta)
    g = (1.0 + q2 * temr * q3 + q1 * sqt) * temr
    h = (1.0 + 0.5 * temr / theta) * (1.0 + q2 * temr)
    ct = 1.0 + g / h
    f = (2.0 / 3.0) / theta32
    return ferinv7_n1_jax(f) - 1.5 * jnp.log(ct)


def _blend_series1(
    fp: jax.Array, fm: jax.Array, a: BlinSeries1, b: BlinSeries1
) -> BlinSeries1:
    return BlinSeries1(
        w=a.w * fp + b.w * fm,
        wdx=a.wdx * fp + b.wdx * fm,
        wdt=a.wdt * fp + b.wdt * fm,
    )


def blin9a_jax(
    temp: jax.Array, chi: jax.Array
) -> tuple[BlinSeries1, BlinSeries1, BlinSeries1]:
    """Evaluate `BLIN9a` (`pc_eos.f90` lines 2417-2526)."""
    echi = jnp.exp(-jnp.clip(chi, -500.0, 500.0))
    series: list[BlinSeries1] = []
    for k in range(3):
        w = jnp.zeros_like(temp)
        wdx = jnp.zeros_like(temp)
        wdt = jnp.zeros_like(temp)
        for i in range(5):
            ac = _BLIN9A_AC[k][i]
            au = _BLIN9A_AU[k][i]
            aa = jnp.exp(-au)
            sq = jnp.sqrt(1.0 + 0.5 * au * temp)
            dn = echi + aa
            dn2 = dn * dn
            w = w + sq / dn * ac
            wdx = wdx + sq / dn2 * ac
            wdt = wdt + (ac * au) / (sq * dn)
        series.append(BlinSeries1(w=w, wdx=wdx * echi, wdt=wdt * 0.25))
    return series[0], series[1], series[2]


def blin9b_jax(
    temp: jax.Array, chi_in: jax.Array
) -> tuple[BlinSeries1, BlinSeries1, BlinSeries1]:
    """Evaluate `BLIN9b` (`pc_eos.f90` lines 2528-2653)."""
    chi = jnp.maximum(chi_in, 1.0e-3)
    sqchi = jnp.sqrt(chi)
    series: list[BlinSeries1] = []
    for k in range(3):
        kf = float(k)
        w = jnp.zeros_like(temp)
        wdx = jnp.zeros_like(temp)
        wdt = jnp.zeros_like(temp)
        chi_k1 = chi ** (k + 1)
        for i in range(5):
            ax = _BLIN9B_AX[i]
            axi = _BLIN9B_AXI[i]
            ah_axk = _BLIN9B_AH[i] * (ax**k)
            av = _BLIN9B_AV[i]
            ce = ax - 1.0
            echi = jnp.exp(jnp.clip(chi * ce, -500.0, 500.0))
            de = echi + 1.0
            d = chi * temp * (0.5 * ax) + 1.0
            h = chi_k1 * sqchi * jnp.sqrt(d) / de
            hx = (kf + 1.5) / chi + temp * (0.25 * ax) / d - echi * ce / de
            hdx = h * hx
            hdt = h * (chi * (0.25 * ax) / d)

            xichi = chi + axi
            dxi = xichi * temp * 0.5 + 1.0
            v = (xichi**k) * jnp.sqrt(xichi * dxi)
            vdx = v * ((kf + 0.5) / xichi + temp * 0.25 / dxi)
            vdt = v * (xichi * 0.25 / dxi)

            w = w + h * ah_axk + v * av
            wdx = wdx + hdx * ah_axk + vdx * av
            wdt = wdt + hdt * ah_axk + vdt * av
        series.append(BlinSeries1(w=w, wdx=wdx, wdt=wdt))
    return series[0], series[1], series[2]


def blin9c_jax(
    temp: jax.Array, chi_in: jax.Array
) -> tuple[BlinSeries1, BlinSeries1, BlinSeries1]:
    """Evaluate `BLIN9c` (`pc_eos.f90` lines 2655-2887)."""
    chi = jnp.maximum(chi_in, 1.0e-3)
    sqchi = jnp.sqrt(chi)
    chi2 = chi * chi
    chi3 = chi2 * chi
    chi4 = chi2 * chi2
    pi26_2_07 = 0.7 * _PI26 * _PI26

    s_small: list[BlinSeries1] = []
    for k in range(3):
        w = jnp.zeros_like(temp)
        wdx = jnp.zeros_like(temp)
        wdt = jnp.zeros_like(temp)
        c = jnp.zeros_like(temp)
        for j in range(5):
            cnu = float(k + j) + 0.5
            chinu = (chi ** (k + j)) * sqchi
            f = chinu * (
                chi / (cnu + 1.0)
                + (_PI26 * cnu) / chi
                + (pi26_2_07 * cnu * (cnu - 1.0) * (cnu - 2.0)) / chi3
            )
            fdx = chinu * (
                (_PI26 * cnu * (cnu - 1.0)) / chi2
                + (pi26_2_07 * cnu * (cnu - 1.0) * (cnu - 2.0) * (cnu - 3.0))
                / chi4
                + 1.0
            )
            if j == 0:
                w = f
                wdx = fdx
            elif j == 1:
                c = temp * 0.25
                w = w + c * f
                wdx = wdx + c * fdx
                wdt = f * 0.25
            else:
                jf = float(j)
                c = -c * temp * ((2.0 * jf - 3.0) / (4.0 * jf))
                w = w + c * f
                wdx = wdx + c * fdx
                wdt = wdt + (c * jf / temp) * f
        s_small.append(BlinSeries1(w=w, wdx=wdx, wdt=wdt))

    d = chi * temp * 0.5 + 1.0
    r = jnp.sqrt(chi * d)
    rx = 0.5 / chi + temp * 0.25 / d
    rdx = r * rx
    rdt = chi2 * 0.25 / r

    am: list[jax.Array] = []
    amdx: list[jax.Array] = []
    amdt: list[jax.Array] = []
    for k in range(3):
        kf = float(k)
        dm = chi * temp * (0.5 * (kf + 1.0)) + (kf + 0.5)
        am_k = (chi**k) * dm / r
        fmx = (
            (kf - 0.5) / chi + temp * (0.5 * (kf + 1.0)) / dm - temp * 0.25 / d
        )
        fmt = (0.5 * (kf + 1.0)) / dm * chi - chi * 0.25 / d
        am.append(am_k)
        amdx.append(am_k * fmx)
        amdt.append(am_k * fmt)

    sq2t = jnp.sqrt(temp * 2.0)
    sq2t3 = sq2t * sq2t * sq2t
    a = chi * temp + sq2t * r + 1.0
    adx = temp + sq2t * rdx
    adt = chi + r / sq2t + sq2t * rdt
    xt1 = chi + 1.0 / temp
    aln = jnp.log(a)
    asq3 = a * sq2t3
    fj0 = xt1 * r * 0.5 - aln / sq2t3
    fj0dx = (r + xt1 * rdx) * 0.5 - adx / asq3
    fj0dt = (
        (xt1 * rdt - r / (temp * temp)) * 0.5
        - adt / asq3
        + aln * 0.75 / (temp * temp * sq2t)
    )
    fj1 = (r * r * r / 1.5 - fj0) / temp
    fj1dx = (r * r * rdx * 2.0 - fj0dx) / temp
    fj1dt = (r * r * rdt * 2.0 - fj0dt - fj1) / temp
    fj2 = (chi * r * r * r * 0.5 - fj1 * 1.25) / temp
    fj2dx = (r * r * r * 0.5 + chi * r * r * rdx * 1.5 - fj1dx * 1.25) / temp
    fj2dt = (chi * r * r * rdt * 1.5 - fj1dt * 1.25 - fj2) / temp

    s_large = (
        BlinSeries1(
            w=fj0 + am[0] * _PI26,
            wdx=fj0dx + amdx[0] * _PI26,
            wdt=fj0dt + amdt[0] * _PI26,
        ),
        BlinSeries1(
            w=fj1 + am[1] * _PI26,
            wdx=fj1dx + amdx[1] * _PI26,
            wdt=fj1dt + amdt[1] * _PI26,
        ),
        BlinSeries1(
            w=fj2 + am[2] * _PI26,
            wdx=fj2dx + amdx[2] * _PI26,
            wdt=fj2dt + amdt[2] * _PI26,
        ),
    )
    cond = (chi_in * temp) < 0.1
    return tuple(  # type: ignore[return-value]
        BlinSeries1(
            w=jnp.where(cond, sm.w, lg.w),
            wdx=jnp.where(cond, sm.wdx, lg.wdx),
            wdt=jnp.where(cond, sm.wdt, lg.wdt),
        )
        for sm, lg in zip(s_small, s_large, strict=True)
    )


def blin9_jax(
    temp: jax.Array, chi: jax.Array
) -> tuple[BlinSeries1, BlinSeries1, BlinSeries1]:
    """Evaluate `BLIN9(temp, chi)` (`pc_eos.f90` lines 2306-2415)."""
    chi1 = 0.6
    chi2 = 14.0
    xmax = 30.0
    dchi1 = 0.1
    dchi2 = chi2 - chi1 - dchi1
    x1 = (chi - chi1) * (xmax / dchi1)
    x2 = (chi - chi2) * (xmax / dchi2)

    fp1, fm1 = fermi10_jax(x1, xmax)
    fp2, fm2 = fermi10_jax(x2, xmax)
    a = blin9a_jax(temp, chi)
    b = blin9b_jax(temp, chi)
    c = blin9c_jax(temp, chi)

    ab = tuple(
        _blend_series1(fp1, fm1, ak, bk) for ak, bk in zip(a, b, strict=True)
    )
    bc = tuple(
        _blend_series1(fp2, fm2, bk, ck) for bk, ck in zip(b, c, strict=True)
    )
    return tuple(  # type: ignore[return-value]
        BlinSeries1(
            w=jnp.where(
                x1 < -xmax,
                ak.w,
                jnp.where(x2 < xmax, jnp.where(x1 < xmax, abk.w, bck.w), ck.w),
            ),
            wdx=jnp.where(
                x1 < -xmax,
                ak.wdx,
                jnp.where(
                    x2 < xmax, jnp.where(x1 < xmax, abk.wdx, bck.wdx), ck.wdx
                ),
            ),
            wdt=jnp.where(
                x1 < -xmax,
                ak.wdt,
                jnp.where(
                    x2 < xmax, jnp.where(x1 < xmax, abk.wdt, bck.wdt), ck.wdt
                ),
            ),
        )
        for ak, abk, bck, ck in zip(a, ab, bc, c, strict=True)
    )


def elect11a_jax(temp: jax.Array, chi: jax.Array) -> Elect11Result:
    """Evaluate `ELECT11a(temp, chi)` (`pc_eos.f90` lines 1557-1620)."""
    bohr = 137.036
    pi2 = constants.PI * constants.PI
    bohr2 = bohr * bohr
    bohr3 = bohr2 * bohr
    temr = temp / bohr2
    w0, w1, w2 = blin9_jax(temr, chi)
    tpi = temr * jnp.sqrt(temr * 2.0) / pi2
    denr = tpi * (w1.w * temr + w0.w)
    pr = (temr * tpi / 3.0) * (w2.w * temr + w1.w * 2.0)
    u = temr * tpi * (w2.w * temr + w1.w)
    pe_id = pr / (denr * temr)
    ue_id = u / (denr * temr)
    fe_id = chi - pe_id
    dens = denr * bohr3
    se_id = ue_id - fe_id
    dndt = tpi * (w0.w * 1.5 / temr + w1.w * 2.5 + w0.wdt + temr * w1.wdt)
    dpdt = (tpi / 3.0) * (
        w1.w * 5.0
        + temr * w1.wdt * 2.0
        + temr * w2.w * 3.5
        + temr * temr * w2.wdt
    )
    dudt = tpi * (
        w1.w * 2.5 + temr * w1.wdt + temr * w2.w * 3.5 + temr * temr * w2.wdt
    )
    dndh = tpi * (w0.wdx + temr * w1.wdx)
    dpdh = (tpi / 3.0) * temr * (w1.wdx * 2.0 + temr * w2.wdx)
    dudh = tpi * temr * (w1.wdx + temr * w2.wdx)
    cve = (dudt - dudh * dndt / dndh) / denr
    chite = (temr / pr) * (dpdt - dpdh * dndt / dndh)
    chire = (denr / pr) * (dpdh / dndh)
    return Elect11Result(dens, fe_id, pe_id, ue_id, se_id, cve, chite, chire)


def elect11b_jax(temp: jax.Array, chi_in: jax.Array) -> Elect11Result:
    """Evaluate `ELECT11b(temp, chi)` (`pc_eos.f90` lines 1622-1705)."""
    bohr = 137.036
    pi2 = constants.PI * constants.PI
    bohr2 = bohr * bohr
    bohr3 = bohr2 * bohr
    temr = temp / bohr2
    chi = jnp.maximum(chi_in, 1.0e-3)
    ef0 = chi * temr
    delta_ef = (
        (temr * temr * (pi2 / 6.0))
        * (ef0 * (ef0 + 2.0) * 2.0 + 1.0)
        / (ef0 * (ef0 + 1.0) * (ef0 + 2.0))
    )
    ef = jnp.maximum(ef0 + delta_ef, 1.0e-30)
    g = ef + 1.0
    ef_hi = jnp.maximum(ef, 1.0e-5)
    g_hi = ef_hi + 1.0
    pf_hi = jnp.sqrt(g_hi * g_hi - 1.0)
    f0_hi = (
        pf_hi * (pf_hi * pf_hi * 2.0 + 1.0) * g_hi
        - (pf_hi**3) / 0.375
        - jnp.log(pf_hi + g_hi)
    ) / (8.0 * pi2)
    df_hi = -temr * temr * pf_hi * g_hi / 6.0
    p0_hi = (
        pf_hi * g_hi * (pf_hi * pf_hi / 1.5 - 1.0) + jnp.log(pf_hi + g_hi)
    ) / (8.0 * pi2)
    del_p_hi = temr * temr * pf_hi * (pf_hi * pf_hi + 2.0) / g_hi / 18.0
    cve_hi = temr * g_hi * pi2 / (pf_hi * pf_hi)

    pf_lo = jnp.sqrt(ef * 2.0)
    f0_lo = (pf_lo**5) * 0.1 / pi2
    df_lo = -temr * temr * pf_lo / 6.0
    p0_lo = f0_lo / 1.5
    del_p_lo = temr * temr * pf_lo / 9.0
    cve_lo = temr * (pi2 / 2.0) / ef

    cond = ef > 1.0e-5
    pf = jnp.where(cond, pf_hi, pf_lo)
    f0 = jnp.where(cond, f0_hi, f0_lo)
    df = jnp.where(cond, df_hi, df_lo)
    p0 = jnp.where(cond, p0_hi, p0_lo)
    del_p = jnp.where(cond, del_p_hi, del_p_lo)
    cve = jnp.where(cond, cve_hi, cve_lo)

    f = f0 + df
    p = p0 + del_p
    s = df * (-2.0)
    u = f + s
    chire = (pf**5) / (p * g * (9.0 * pi2))
    chite = del_p * 2.0 / p
    denr = (pf**3) / (3.0 * pi2)
    dens = denr * bohr3
    dt = denr * temr
    pe_id = p / dt
    ue_id = u / dt
    fe_id = f / dt
    se_id = s / dt
    d1 = delta_ef / ef
    d2 = d1 * (4.0 - (pf / g) * 2.0)
    cve = cve / (d2 + 1.0)
    se_id = se_id / (d1 + 1.0)
    chite = chite / (d2 + 1.0)
    return Elect11Result(dens, fe_id, pe_id, ue_id, se_id, cve, chite, chire)


def elect11_jax(temp: jax.Array, chi: jax.Array) -> Elect11Result:
    """Evaluate `ELECT11(temp, chi)` (`pc_eos.f90` lines 1479-1555)."""
    chi1 = 0.6
    chi2 = 28.0
    xmax = 20.0
    dchi1 = 0.1
    dchi2 = chi2 - chi1 - dchi1
    x2 = (chi - chi2) * (xmax / dchi2)
    a = elect11a_jax(temp, chi)
    b = elect11b_jax(temp, chi)
    fp, fm = fermi10_jax(x2, xmax)
    return Elect11Result(
        *(
            jnp.where(
                x2 < -xmax,
                av,
                jnp.where(x2 > xmax, bv, av * fp + bv * fm),
            )
            for av, bv in zip(a, b, strict=True)
        )
    )
