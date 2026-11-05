# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bremsstrahlung (`brem_neu`) neutrino energy loss rates and derivatives.

Ports `brem_neu_weak_degen`, `brem_neu_liquid_metal`, and `brem_neu` (Itoh et
al. 1996, ApJS, 102, 411, Section 5) from `neu/private/mod_neu.f90`.

Degeneracy Blending Strategy:
    - For $T \\ge 0.3\\,T_F$, evaluates the weakly degenerate electron gas fit
      (`brem_neu_weak_degen`, eqs. 5.1-5.9).
    - For $T \\le 0.05\\,T_F$, evaluates the strongly degenerate liquid-metal
      fit (`brem_neu_liquid_metal`, eqs. 5.16-5.26).
    - For $0.05\\,T_F < T < 0.3\\,T_F$, blends both regimes via the smooth
      cosine ramp $\\alpha = \\tfrac{1}{2}(1 - \\cos(\\pi\\,\\mathrm{tfrac}))$
      with exact chain-rule partial derivatives with respect to
      $(T, \\rho, \\bar{A}, \\bar{Z})$.
"""

from __future__ import annotations

import math

from flexmesa import constants as _const
from flexmesa.neu import inputs as _inputs

# Angular scale pi / 5 (`fac3` in `mod_neu.f90`).
FAC3: float = _const.PI / 5.0


def _combine_brem_rate(
    t85: float,
    t86: float,
    inp: _inputs.NeuInputs,
    f: tuple[float, float, float, float, float],
    g: tuple[float, float, float, float, float],
) -> _inputs.NeuRate:
    """Combine vector (`f`) and axial-vector (`g`) structure factors."""
    pref = 0.5738 * inp.zbar * inp.ye * t86 * inp.den
    prefdt = 0.5738 * inp.zbar * inp.ye * 6.0 * t85 * inp.den * 1.0e-8
    prefdd = 0.5738 * inp.zbar * inp.ye * t86
    prefda = -pref * inp.abari
    prefdz = 0.5738 * 2.0 * inp.ye * t86 * inp.den
    comb = _inputs.TFAC4 * f[0] - _inputs.TFAC5 * g[0]
    return _inputs.NeuRate(
        val=pref * comb,
        d_t=prefdt * comb
        + pref * (_inputs.TFAC4 * f[1] - _inputs.TFAC5 * g[1]),
        d_rho=prefdd * comb
        + pref * (_inputs.TFAC4 * f[2] - _inputs.TFAC5 * g[2]),
        d_abar=prefda * comb
        + pref * (_inputs.TFAC4 * f[3] - _inputs.TFAC5 * g[3]),
        d_zbar=prefdz * comb
        + pref * (_inputs.TFAC4 * f[4] - _inputs.TFAC5 * g[4]),
    )


def brem_neu_weak_degen(t8: float, inp: _inputs.NeuInputs) -> _inputs.NeuRate:
    """Evaluates weakly degenerate bremsstrahlung fit (`brem_neu_weak_degen`).

    Implements Itoh et al. (1996) eqs. 5.1-5.9 for $T \\ge 0.3\\,T_F$, combining
    weakly degenerate vector (`fbrem`) and axial-vector (`gbrem`) structure
    factors.

    Args:
        t8: Scaled temperature $T_8 = T / 10^8\\,\\mathrm{K}$.
        inp: Precomputed thermodynamic and composition invariants (`NeuInputs`).

    Returns:
        Volumetric rate `NeuRate` in $\\mathrm{erg\\,cm^{-3}\\,s^{-1}}$.
    """
    t812 = math.sqrt(t8)
    t832 = t8 * t812
    t82 = t8 * t8
    t83 = t82 * t8
    t85 = t82 * t83
    t86 = t85 * t8
    t8m1 = 1.0 / t8
    t8m2 = t8m1 * t8m1
    t8m3 = t8m2 * t8m1
    t8m5 = t8m3 * t8m2
    t8m6 = t8m5 * t8m1

    dum = 7.05e6 * t832 + 5.12e4 * t83
    dumdt = (1.5 * 7.05e6 * t812 + 3.0 * 5.12e4 * t82) * 1.0e-8
    z = 1.0 / dum
    eta = inp.rm * z
    etadt = -inp.rm * z * z * dumdt
    etadd = inp.rmdd * z
    etada = inp.rmda * z
    etadz = inp.rmdz * z
    etam1 = 1.0 / eta
    etam2 = etam1 * etam1
    etam3 = etam2 * etam1

    a0 = 23.5 + 6.83e4 * t8m2 + 7.81e8 * t8m5
    f0 = (-2.0 * 6.83e4 * t8m3 - 5.0 * 7.81e8 * t8m6) * 1.0e-8
    xnum = 1.0 / a0
    edum = 1.0 + 1.47 * etam1 + 3.29e-2 * etam2
    dz = -1.47 * etam2 - 2.0 * 3.29e-2 * etam3
    c00 = 1.26 * (1.0 + etam1)
    cz = -1.26 * etam2
    inv_edum = 1.0 / edum
    xden = c00 * inv_edum
    d_scale = (cz - xden * dz) * inv_edum
    f_tup = (
        xnum + xden,
        -xnum * xnum * f0 + d_scale * etadt,
        d_scale * etadd,
        d_scale * etada,
        d_scale * etadz,
    )

    ga0 = 230.0 + 6.7e5 * t8m2 + 7.66e9 * t8m5
    gf0 = (-2.0 * 6.7e5 * t8m3 - 5.0 * 7.66e9 * t8m6) * 1.0e-8
    gz = 1.0 + inp.rm * 1.0e-9
    gxnum = 1.0 / (ga0 * gz)
    neg_gx2 = -gxnum * gxnum
    g_scale = neg_gx2 * ga0 * 1.0e-9

    p1 = math.pow(t8, 3.85)
    gc00 = 7.75e5 * t832 + 247.0 * p1
    gdd00 = (1.5 * 7.75e5 * t812 + 3.85 * 247.0 * p1 / t8) * 1.0e-8
    p2 = math.pow(t8, 1.4)
    gc01 = 4.07 + 0.0240 * p2
    gdd01 = 1.4 * 0.0240 * (p2 / t8) * 1.0e-8
    p3 = math.pow(t8, -0.110)
    gc02 = 4.59e-5 * p3
    gdd02 = -0.11 * 4.59e-5 * (p3 / t8) * 1.0e-8

    den_p = math.pow(inp.den, 0.656)
    gd2 = gc00 * inp.rmi + gc01 + gc02 * den_p
    gd2dt = gdd00 * inp.rmi + gdd01 + gdd02 * den_p
    neg_c00_rmi2 = -gc00 * inp.rmi * inp.rmi
    gd2dd = neg_c00_rmi2 * inp.rmdd + 0.656 * gc02 * math.pow(inp.den, -0.344)
    gxden = 1.0 / gd2
    neg_gxd2 = -gxden * gxden
    g_tup = (
        gxnum + gxden,
        neg_gx2 * gf0 * gz + neg_gxd2 * gd2dt,
        g_scale * inp.rmdd + neg_gxd2 * gd2dd,
        (g_scale + neg_gxd2 * neg_c00_rmi2) * inp.rmda,
        (g_scale + neg_gxd2 * neg_c00_rmi2) * inp.rmdz,
    )
    return _combine_brem_rate(t85, t86, inp, f_tup, g_tup)


def _eval_liq_fourier(
    u: float,
    a0_u: float,
    cos_k: tuple[float, ...],
    sin_k: tuple[float, ...],
    c0: float,
    lin: float,
    const_term: float,
    a: tuple[float, ...],
    b: tuple[float, ...],
) -> tuple[float, float]:
    """Evaluates 5-harmonic liquid-metal Fourier fit and density derivative.

    Computes $(f(u), \\partial f / \\partial \\rho)$ for Itoh et al. (1996)
    Section 5.2 (`fb`, `ft`, `gb`, `gt` and their $\\rho$ derivatives `c00..c03`
    in `brem_neu_liquid_metal`) over the dimensionless Fourier angle
    $u = (\\pi / 5)(\\log_{10} \\rho - 3)$ and density chain-rule factor
    $a_{0,u} = (\\pi / (5 \\ln 10)) / \\rho$.

    Args:
        u: Scaled logarithmic density argument $u = (\\pi / 5)(\\log_{10}\\rho
            - 3)$.
        a0_u: Derivative factor $\\partial u / \\partial \\rho =
            (\\pi / 5) / (\\rho \\ln 10)$.
        cos_k: Harmonics $(\\cos u, \\cos 2u, \\cos 3u, \\cos 4u, \\cos 5u)$.
        sin_k: Harmonics $(\\sin u, \\sin 2u, \\sin 3u, \\sin 4u, \\sin 5u)$.
        c0: Zeroth cosine harmonic coefficient (multiplied by $0.5$).
        lin: Linear slope coefficient multiplying $u$.
        const_term: Additive constant offset.
        a: Cosine coefficients $(a_1, a_2, a_3, a_4, a_5)$.
        b: Sine coefficients $(b_1, b_2, b_3, b_4)$.

    Returns:
        Tuple `(val, dval_drho)`.
    """
    val = (
        0.5 * c0
        + lin * u
        + const_term
        + a[0] * cos_k[0]
        + b[0] * sin_k[0]
        + a[1] * cos_k[1]
        + b[1] * sin_k[1]
        + a[2] * cos_k[2]
        + b[2] * sin_k[2]
        + a[3] * cos_k[3]
        + b[3] * sin_k[3]
        + a[4] * cos_k[4]
    )
    dval_drho = a0_u * (
        lin
        - a[0] * sin_k[0]
        + b[0] * cos_k[0]
        - a[1] * sin_k[1] * 2.0
        + b[1] * cos_k[1] * 2.0
        - a[2] * sin_k[2] * 3.0
        + b[2] * cos_k[2] * 3.0
        - a[3] * sin_k[3] * 4.0
        + b[3] * cos_k[3] * 4.0
        - a[4] * sin_k[4] * 5.0
    )
    return val, dval_drho


def brem_neu_liquid_metal(t8: float, inp: _inputs.NeuInputs) -> _inputs.NeuRate:
    """Evaluates strongly degenerate liquid-metal fit (`brem_neu_liquid_metal`).

    Implements Itoh et al. (1996) eqs. 5.16-5.26 for $T \\le 0.05\\,T_F$, where
    ions form a strongly coupled Coulomb liquid ($1 \\le \\Gamma < 180$).

    Args:
        t8: Scaled temperature $T_8 = T / 10^8\\,\\mathrm{K}$.
        inp: Precomputed thermodynamic and composition invariants (`NeuInputs`).

    Returns:
        Volumetric rate `NeuRate` in $\\mathrm{erg\\,cm^{-3}\\,s^{-1}}$.
    """
    t82 = t8 * t8
    t83 = t82 * t8
    t85 = t82 * t83
    t86 = t85 * t8
    t8m1 = 1.0 / t8

    u = FAC3 * (inp.logden - 3.0)
    a0_u = _const.ILN10 * FAC3 * inp.deni
    cos1 = math.cos(u)
    sin1 = math.sin(u)
    sin2 = 2.0 * sin1 * cos1
    cos2 = 2.0 * cos1 * cos1 - 1.0
    sin3 = sin1 * (3.0 - 4.0 * sin1 * sin1)
    cos3 = cos1 * (4.0 * cos1 * cos1 - 3.0)
    sin4 = 2.0 * sin2 * cos2
    cos4 = 2.0 * cos2 * cos2 - 1.0
    sin5 = sin1 * (5.0 - sin1 * sin1 * (20.0 - 16.0 * sin1 * sin1))
    cos5 = cos1 * (cos1 * cos1 * (16.0 * cos1 * cos1 - 20.0) + 5.0)
    cos_k = (cos1, cos2, cos3, cos4, cos5)
    sin_k = (sin1, sin2, sin3, sin4, sin5)

    # Evaluate Fourier components (fb, ft, gb, gt) and their rho derivatives.
    fb, c00 = _eval_liq_fourier(
        u,
        a0_u,
        cos_k,
        sin_k,
        0.17946,
        0.00945,
        0.34529,
        (-0.05821, -0.01089, -0.01147, -0.00656, -0.00519),
        (-0.04969, -0.01584, -0.00504, -0.00281),
    )
    ft, c01 = _eval_liq_fourier(
        u,
        a0_u,
        cos_k,
        sin_k,
        0.06781,
        -0.02342,
        0.24819,
        (-0.00944, -0.01289, -0.00589, -0.00404, -0.00330),
        (-0.02213, -0.01136, -0.00467, -0.00131),
    )
    gb, c02 = _eval_liq_fourier(
        u,
        a0_u,
        cos_k,
        sin_k,
        0.00766,
        -0.01259,
        0.07917,
        (-0.00710, -0.00028, 0.00232, 0.00044, 0.00158),
        (0.02300, -0.01078, 0.00118, -0.00089),
    )
    gt, c03 = _eval_liq_fourier(
        u,
        a0_u,
        cos_k,
        sin_k,
        -0.00769,
        -0.00829,
        0.05211,
        (0.00356, -0.00184, 0.00146, 0.00031, 0.00069),
        (0.01052, -0.00354, -0.00014, -0.00018),
    )

    # Evaluate ion-ion Coulomb coupling parameter Gamma (`dum`) and powers.
    dum = (
        2.275e-1
        * inp.zbar
        * inp.zbar
        * t8m1
        * math.pow(inp.den6 * inp.abari, _const.ONE_THIRD)
    )
    dumdt = -dum * inp.tempi
    dumdd = _const.ONE_THIRD * dum * inp.deni
    dumda = -_const.ONE_THIRD * dum * inp.abari
    dumdz = 2.0 * dum * inp.zbari

    gm1 = 1.0 / dum
    gm2 = gm1 * gm1
    gm13 = math.pow(gm1, _const.ONE_THIRD)
    gm23 = gm13 * gm13
    gm43 = gm13 * gm1
    gm53 = gm23 * gm1

    v = -0.05483 - 0.01946 * gm13 + 1.86310 * gm23 - 0.78873 * gm1
    a0 = (
        _const.ONE_THIRD * 0.01946 * gm43
        - _const.TWO_THIRDS * 1.86310 * gm53
        + 0.78873 * gm2
    )
    w = -0.06711 + 0.06859 * gm13 + 1.74360 * gm23 - 0.74498 * gm1
    a1 = (
        -_const.ONE_THIRD * 0.06859 * gm43
        - _const.TWO_THIRDS * 1.74360 * gm53
        + 0.74498 * gm2
    )

    f_liq = (
        v * fb + (1.0 - v) * ft,
        a0 * dumdt * (fb - ft),
        a0 * dumdd * (fb - ft) + v * c00 + (1.0 - v) * c01,
        a0 * dumda * (fb - ft),
        a0 * dumdz * (fb - ft),
    )
    g_liq = (
        w * gb + (1.0 - w) * gt,
        a1 * dumdt * (gb - gt),
        a1 * dumdd * (gb - gt) + w * c02 + (1.0 - w) * c03,
        a1 * dumda * (gb - gt),
        a1 * dumdz * (gb - gt),
    )
    return _combine_brem_rate(t85, t86, inp, f_liq, g_liq)


def brem_neu(inp: _inputs.NeuInputs) -> _inputs.NeuRate:
    """Evaluates volumetric bremsstrahlung neutrino energy loss rate.

    Blends `brem_neu_weak_degen` ($T \\ge 0.3\\,T_F$) and
    `brem_neu_liquid_metal` ($T \\le 0.05\\,T_F$) across
    $0.05\\,T_F < T < 0.3\\,T_F$.

    Args:
        inp: Precomputed thermodynamic and composition invariants (`NeuInputs`).

    Returns:
        Volumetric rate `NeuRate` in $\\mathrm{erg\\,cm^{-3}\\,s^{-1}}$ before
        dividing by $\\rho$.
    """
    tflo = 0.05
    tfhi = 0.3
    a_tf = 5.9302e9

    t8 = inp.temp * 1.0e-8
    u = math.pow(inp.den6 * inp.ye, _const.TWO_THIRDS)
    tfermi = a_tf * (math.sqrt(u) - 1.0)
    if inp.temp >= tfhi * tfermi:
        return brem_neu_weak_degen(t8, inp)
    if inp.temp <= tflo * tfermi:
        return brem_neu_liquid_metal(t8, inp)

    weak = brem_neu_weak_degen(t8, inp)
    liq = brem_neu_liquid_metal(t8, inp)
    dtf = tfhi - tflo
    tfrac = (inp.temp / tfermi - tflo) / dtf
    alfa = 0.5 * (1.0 - math.cos(_const.PI * tfrac))
    beta = 1.0 - alfa

    dtfermidu = 0.5 * a_tf * math.pow(u, -0.5)
    dtfrac_dtf = -inp.temp / (tfermi * tfermi * dtf)
    dudv = _const.TWO_THIRDS * math.pow(inp.den6 * inp.ye, -1.0 / 3.0)
    dudd = dudv * 1.0e-6 * inp.ye
    duda = -dudv * inp.den6 * inp.ye * inp.abari
    dudz = dudv * inp.den6 * inp.abari

    dtfracdt = 1.0 / (tfermi * dtf)
    dtfracdd = dtfermidu * dudd * dtfrac_dtf
    dtfracda = dtfermidu * duda * dtfrac_dtf
    dtfracdz = dtfermidu * dudz * dtfrac_dtf

    d_cos = 0.5 * _const.PI * math.sin(_const.PI * tfrac)
    dalfadt = dtfracdt * d_cos
    dalfadd = dtfracdd * d_cos
    dalfada = dtfracda * d_cos
    dalfadz = dtfracdz * d_cos
    diff_val = weak.val - liq.val
    return _inputs.NeuRate(
        val=alfa * weak.val + beta * liq.val,
        d_t=alfa * weak.d_t + beta * liq.d_t + dalfadt * diff_val,
        d_rho=alfa * weak.d_rho + beta * liq.d_rho + dalfadd * diff_val,
        d_abar=alfa * weak.d_abar + beta * liq.d_abar + dalfada * diff_val,
        d_zbar=alfa * weak.d_zbar + beta * liq.d_zbar + dalfadz * diff_val,
    )
