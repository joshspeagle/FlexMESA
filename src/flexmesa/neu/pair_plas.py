# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pair-production (`pair_neu`) and plasmon-decay (`plas_neu`) neutrino losses.

Ports `pair_neu` (Itoh et al. 1996, ApJS, 102, 411, eqs. 2.1-2.8) and
`plas_neu` (eqs. 4.1, 4.5-4.11) from `neu/private/mod_neu.f90`.

Physical Regimes:
    - **Pair Production (`pair_neu`)**: Dominant at high temperature
      ($T \\gtrsim 10^9\\,\\mathrm{K}$) and low-to-moderate density where
      thermal photons create $e^+ e^-$ pairs that annihilate into
      $\\nu \\bar{\\nu}$ pairs. Separates fit coefficients at $T_9 = 10$
      ($T = 10^{10}\\,\\mathrm{K}$).
    - **Plasmon Decay (`plas_neu`)**: Dominant at high electron degeneracy where
      transverse (`ft`), longitudinal (`fl`), and mixed (`fxy`) collective
      plasma excitations decay into $\\nu_e \\bar{\\nu}_e$ pairs.
"""

from __future__ import annotations

import math

from flexmesa import constants as _const
from flexmesa.neu import inputs as _inputs


def pair_neu(inp: _inputs.NeuInputs) -> _inputs.NeuRate:
    """Evaluates volumetric pair-neutrino energy loss rate and derivatives.

    Implements Itoh et al. (1996) eqs. 2.1-2.8 (`pair_neu` in `mod_neu.f90`):
    $$Q_{\\mathrm{pair}} = \\mathrm{TFAC4}\\,(1 +
      \\mathrm{TFAC3}\\,q_{\\mathrm{pair}})\\,g(\\lambda)\\,e^{-2/\\lambda}
      \\,f_{\\mathrm{pair}}(\\lambda, \\xi)$$

    Args:
        inp: Precomputed thermodynamic and composition invariants (`NeuInputs`).

    Returns:
        Volumetric rate `NeuRate` in $\\mathrm{erg\\,cm^{-3}\\,s^{-1}}$ before
        dividing by $\\rho$.
    """
    # Evaluate g(lambda) polynomial and its temperature derivative (eq. 2.3).
    gl = (
        1.0
        - 13.04 * inp.xl2
        + 133.5 * inp.xl4
        + 1534.0 * inp.xl6
        + 918.6 * inp.xl8
    )
    gldt = inp.xldt * (
        -26.08 * inp.xl
        + 534.0 * inp.xl3
        + 9204.0 * inp.xl5
        + 7348.8 * inp.xl7
    )

    # Evaluate f_pair numerator and denominator across the T9 = 10 boundary.
    a1 = 6.002e19 + 2.084e20 * inp.zeta + 1.872e21 * inp.zeta2
    a2 = 2.084e20 + 2.0 * 1.872e21 * inp.zeta

    if inp.t9 < 10.0:
        b1 = math.exp(-5.5924 * inp.zeta)
        b2 = -b1 * 5.5924
        den_a1 = (
            9.383e-1 * inp.xlm1 - 4.141e-1 * inp.xlm2 + 5.829e-2 * inp.xlm3
        )
        den_a2 = (
            -9.383e-1 * inp.xlm2
            + 2.0 * 4.141e-1 * inp.xlm3
            - 3.0 * 5.829e-2 * inp.xlm4
        )
    else:
        b1 = math.exp(-4.9924 * inp.zeta)
        b2 = -b1 * 4.9924
        den_a1 = 1.2383 * inp.xlm1 - 8.141e-1 * inp.xlm2
        den_a2 = -1.2383 * inp.xlm2 + 2.0 * 8.141e-1 * inp.xlm3

    xnum = a1 * b1
    c = a2 * b1 + a1 * b2
    xnumdt = c * inp.zetadt
    xnumdd = c * inp.zetadd
    xnumda = c * inp.zetada
    xnumdz = c * inp.zetadz

    b1_z = 3.0 * inp.zeta2
    xden = inp.zeta3 + den_a1
    xdendt = b1_z * inp.zetadt + den_a2 * inp.xldt
    xdendd = b1_z * inp.zetadd
    xdenda = b1_z * inp.zetada
    xdendz = b1_z * inp.zetadz

    inv_xden = 1.0 / xden
    fpair = xnum * inv_xden
    fpairdt = (xnumdt - fpair * xdendt) * inv_xden
    fpairdd = (xnumdd - fpair * xdendd) * inv_xden
    fpairda = (xnumda - fpair * xdenda) * inv_xden
    fpairdz = (xnumdz - fpair * xdendz) * inv_xden

    # Evaluate q_pair(lambda, rm) ratio and its partial derivatives (eq. 2.4).
    q_a1 = 10.7480 * inp.xl2 + 0.3967 * inp.xlp5 + 1.005
    q_a2 = inp.xldt * (2.0 * 10.7480 * inp.xl + 0.5 * 0.3967 * inp.xlmp5)
    q_xnum = 1.0 / q_a1
    q_xnumdt = -q_xnum * q_xnum * q_a2

    d_a1 = 7.692e7 * inp.xl3 + 9.715e6 * inp.xlp5
    d_a2 = inp.xldt * (3.0 * 7.692e7 * inp.xl2 + 0.5 * 9.715e6 * inp.xlmp5)
    d_c = 1.0 / d_a1
    d_b1 = 1.0 + inp.rm * d_c
    q_xden = math.pow(d_b1, -0.3)
    d_fac = -0.3 * q_xden / d_b1
    q_xdendt = -d_fac * inp.rm * d_c * d_c * d_a2
    q_xdendd = d_fac * inp.rmdd * d_c
    q_xdenda = d_fac * inp.rmda * d_c
    q_xdendz = d_fac * inp.rmdz * d_c

    qpair = q_xnum * q_xden
    qpairdt = q_xnumdt * q_xden + q_xnum * q_xdendt
    qpairdd = q_xnum * q_xdendd
    qpairda = q_xnum * q_xdenda
    qpairdz = q_xnum * q_xdendz

    # Combine Boltzmann factor exp(-2 / lambda) with f_pair and g(lambda).
    e_a1 = math.exp(-2.0 * inp.xlm1)
    e_a2 = e_a1 * 2.0 * inp.xlm2 * inp.xldt

    s0 = e_a1 * fpair
    s0dt = e_a2 * fpair + e_a1 * fpairdt
    s0dd = e_a1 * fpairdd
    s0da = e_a1 * fpairda
    s0dz = e_a1 * fpairdz

    s_gl = gl * s0
    spairdt = gl * s0dt + gldt * s0
    spairdd = gl * s0dd
    spairda = gl * s0da
    spairdz = gl * s0dz

    # Scale by weak-interaction neutral/charged-current prefactors (TFAC3,
    # TFAC4) and assemble the volumetric rate and partial derivatives
    # with respect to (T, rho, abar, zbar).
    w1 = _inputs.TFAC4 * (1.0 + _inputs.TFAC3 * qpair)
    w2 = _inputs.TFAC4 * _inputs.TFAC3
    return _inputs.NeuRate(
        val=w1 * s_gl,
        d_t=w1 * spairdt + w2 * qpairdt * s_gl,
        d_rho=w1 * spairdd + w2 * qpairdd * s_gl,
        d_abar=w1 * spairda + w2 * qpairda * s_gl,
        d_zbar=w1 * spairdz + w2 * qpairdz * s_gl,
    )


def plas_neu(inp: _inputs.NeuInputs) -> _inputs.NeuRate:
    """Evaluates volumetric plasmon-decay neutrino energy loss rate.

    Implements Itoh et al. (1996) eqs. 4.1 and 4.5-4.11 (`plas_neu` in
    `mod_neu.f90`):
    $$Q_{\\mathrm{plas}} = C_V^2 \\left(3.0 \\times 10^{21}\\,\\lambda^9\\right)
      \\gamma^6 e^{-\\gamma}\\left(f_T + f_L\\right) f_{xy}(x, y)$$

    Args:
        inp: Precomputed thermodynamic and composition invariants (`NeuInputs`).

    Returns:
        Volumetric rate `NeuRate` in $\\mathrm{erg\\,cm^{-3}\\,s^{-1}}$ before
        dividing by $\\rho$.
    """
    # Evaluate dimensionless plasmon frequency squared gamma^2 (eq. 4.6).
    a1 = 1.019e-6 * inp.rm
    a2 = math.pow(a1, _const.TWO_THIRDS)
    a3 = _const.TWO_THIRDS * a2 / a1
    b1 = math.sqrt(1.0 + a2)
    b2 = 1.0 / b1
    c00 = 1.0 / (inp.temp * inp.temp * b1)

    gl2 = 1.1095e11 * inp.rm * c00
    gl2dt = -2.0 * gl2 * inp.tempi
    d_rm = inp.rm * c00 * b2 * 0.5 * b2 * a3 * 1.019e-6
    gl2dd = 1.1095e11 * (inp.rmdd * c00 - d_rm * inp.rmdd)
    gl2da = 1.1095e11 * (inp.rmda * c00 - d_rm * inp.rmda)
    gl2dz = 1.1095e11 * (inp.rmdz * c00 - d_rm * inp.rmdz)

    # Evaluate transverse (f_T) and longitudinal (f_L) plasmon factors.
    gl = math.sqrt(gl2)
    gl12 = math.sqrt(gl)
    gl32 = gl * gl12
    gl72 = gl2 * gl32
    gl6 = gl2 * gl2 * gl2
    gum = 1.0 / gl2

    ft = 2.4 + 0.6 * gl12 + 0.51 * gl + 1.25 * gl32
    dft_dgl2 = (0.25 * 0.6 * gl12 + 0.5 * 0.51 * gl + 0.75 * 1.25 * gl32) * gum
    ftdt = dft_dgl2 * gl2dt
    ftdd = dft_dgl2 * gl2dd
    ftda = dft_dgl2 * gl2da
    ftdz = dft_dgl2 * gl2dz

    fl_num = 8.6 * gl2 + 1.35 * gl72
    dfl_num = 8.6 + 1.75 * 1.35 * gl72 * gum
    fl_den = 225.0 - 17.0 * gl + gl2
    dfl_den = -0.5 * 17.0 * gl * gum + 1.0
    inv_fl_den = 1.0 / fl_den
    fl = fl_num * inv_fl_den
    dfl_dgl2 = (dfl_num - fl * dfl_den) * inv_fl_den
    fldt = dfl_dgl2 * gl2dt
    fldd = dfl_dgl2 * gl2dd
    flda = dfl_dgl2 * gl2da
    fldz = dfl_dgl2 * gl2dz

    # Evaluate 2D correction factor f_xy(x, y) across the active domain.
    cc = math.log10(2.0 * inp.rm)
    xlnt = inp.logtemp
    xnum = _const.ONE_SIXTH * (17.5 + cc - 3.0 * xlnt)
    xnumdt = -_const.ILN10 * 0.5 * inp.tempi
    d_cc = _const.ILN10 * _const.ONE_SIXTH * inp.rmi
    xnumdd = d_cc * inp.rmdd
    xnumda = d_cc * inp.rmda
    xnumdz = d_cc * inp.rmdz

    xden = _const.ONE_SIXTH * (-24.5 + cc + 3.0 * xlnt)
    xdendt = _const.ILN10 * 0.5 * inp.tempi

    if abs(xnum) > 0.7 or xden < 0.0:
        fxy, fxydt, fxydd, fxyda, fxydz = 1.0, 0.0, 0.0, 0.0, 0.0
    else:
        p_a1 = 0.39 - 1.25 * xnum - 0.35 * math.sin(4.5 * xnum)
        p_a2 = -1.25 - 4.5 * 0.35 * math.cos(4.5 * xnum)
        arg = 4.5 * xnum + 0.9
        p_b1 = 0.3 * math.exp(-math.pow(arg, 2.0))
        p_b2 = -p_b1 * 2.0 * arg * 4.5
        c_val = min(0.0, xden - 1.6 + 1.25 * xnum)
        if c_val == 0.0:
            dumdt, dumdd, dumda, dumdz = 0.0, 0.0, 0.0, 0.0
        else:
            dumdt = xdendt + 1.25 * xnumdt
            dumdd = xnumdd + 1.25 * xnumdd
            dumda = xnumda + 1.25 * xnumda
            dumdz = xnumdz + 1.25 * xnumdz
        d_val = 0.57 - 0.25 * xnum
        a3_val = c_val / d_val
        g_exp = math.exp(-a3_val * a3_val)
        f1 = -g_exp * 2.0 * a3_val / d_val
        c01 = f1 * (dumdt + a3_val * 0.25 * xnumdt)
        c02 = f1 * (dumdd + a3_val * 0.25 * xnumdd)
        c03 = f1 * (dumda + a3_val * 0.25 * xnumda)
        c04 = f1 * (dumdz + a3_val * 0.25 * xnumdz)
        diff_ab = p_a1 - p_b1
        ddiff_ab = p_a2 - p_b2
        fxy = 1.05 + diff_ab * g_exp
        fxydt = ddiff_ab * xnumdt * g_exp + diff_ab * c01
        fxydd = ddiff_ab * xnumdd * g_exp + diff_ab * c02
        fxyda = ddiff_ab * xnumda * g_exp + diff_ab * c03
        fxydz = ddiff_ab * xnumdz * g_exp + diff_ab * c04

    # Combine (f_T + f_L) * f_xy with exp(-gamma), gamma^6, and lambda^9
    # to form the volumetric plasmon-decay loss rate (eq. 4.1).
    # Propagate partial derivatives through each product stage via the
    # Leibniz rule for (T, rho, abar, zbar).
    sum_ftl = ft + fl
    s0 = sum_ftl * fxy
    s0dt = (ftdt + fldt) * fxy + sum_ftl * fxydt
    s0dd = (ftdd + fldd) * fxy + sum_ftl * fxydd
    s0da = (ftda + flda) * fxy + sum_ftl * fxyda
    s0dz = (ftdz + fldz) * fxy + sum_ftl * fxydz

    exp_mgl = math.exp(-gl)
    dexp_dgl2 = -0.5 * exp_mgl * gl * gum
    s1 = exp_mgl * s0
    s1dt = exp_mgl * s0dt + dexp_dgl2 * gl2dt * s0
    s1dd = exp_mgl * s0dd + dexp_dgl2 * gl2dd * s0
    s1da = exp_mgl * s0da + dexp_dgl2 * gl2da * s0
    s1dz = exp_mgl * s0dz + dexp_dgl2 * gl2dz * s0

    dgl6_dgl2 = 3.0 * gl6 * gum
    s2 = gl6 * s1
    s2dt = gl6 * s1dt + dgl6_dgl2 * gl2dt * s1
    s2dd = gl6 * s1dd + dgl6_dgl2 * gl2dd * s1
    s2da = gl6 * s1da + dgl6_dgl2 * gl2da * s1
    s2dz = gl6 * s1dz + dgl6_dgl2 * gl2dz * s1

    pref = _inputs.TFAC0 * 3.0e21 * inp.xl9
    dpref_dt = _inputs.TFAC0 * 3.0e21 * 9.0 * inp.xl8 * inp.xldt
    return _inputs.NeuRate(
        val=pref * s2,
        d_t=pref * s2dt + dpref_dt * s2,
        d_rho=pref * s2dd,
        d_abar=pref * s2da,
        d_zbar=pref * s2dz,
    )
