# SPDX-License-Identifier: LGPL-3.0-or-later
"""Photoneutrino (`phot_neu`) energy loss rates and analytical derivatives.

Ports `phot_neu` (Itoh et al. 1996, ApJS, 102, 411, eqs. 3.2-3.8 and Table 2)
from `neu/private/mod_neu.f90`.

Fourier Expansion Across Temperature Regimes:
    Evaluates the 5-harmonic Fourier series for $a_0(\\tau), a_1(\\tau),
    a_2(\\tau)$ across three temperature intervals:
    - $10^7 \\le T < 10^8\\,\\mathrm{K}$ ($\\tau = \\log_{10} T - 7$,
      $c = 0.5654 + \\tau$)
    - $10^8 \\le T < 10^9\\,\\mathrm{K}$ ($\\tau = \\log_{10} T - 8$,
      $c = 1.5654$)
    - $T \\ge 10^9\\,\\mathrm{K}$ ($\\tau = \\log_{10} T - 9$, $c = 1.5654$)
"""

from __future__ import annotations

import dataclasses
import math

from flexmesa import constants as _const
from flexmesa.neu import inputs as _inputs

# Angular frequency omega_1 = 5 * pi / 3 (`fac1` in `mod_neu.f90`).
FAC1: float = 5.0 * _const.PI / 3.0

# Angular frequency omega_2 = 10 * pi (`fac2` in `mod_neu.f90`).
FAC2: float = 10.0 * _const.PI


@dataclasses.dataclass(frozen=True)
class PhotBandCoeffs:
    """Table 2 Fourier coefficients `(c_j[0..7], d_j[0..5])` for $j = 0, 1, 2$.

    Attributes:
        c0: Cosine harmonic coefficients $c_{0,0 \\dots 6}$ for $a_0(\\tau)$.
        d0: Sine harmonic coefficients $d_{0,1 \\dots 5}$ for $a_0(\\tau)$.
        c1: Cosine harmonic coefficients $c_{1,0 \\dots 6}$ for $a_1(\\tau)$.
        d1: Sine harmonic coefficients $d_{1,1 \\dots 5}$ for $a_1(\\tau)$.
        c2: Cosine harmonic coefficients $c_{2,0 \\dots 6}$ for $a_2(\\tau)$.
        d2: Sine harmonic coefficients $d_{2,1 \\dots 5}$ for $a_2(\\tau)$.
    """

    c0: tuple[float, ...]
    d0: tuple[float, ...]
    c1: tuple[float, ...]
    d1: tuple[float, ...]
    c2: tuple[float, ...]
    d2: tuple[float, ...]


# Table 2 coefficients for 10^7 <= T < 10^8 K (Itoh et al. 1996).
BAND_7_8 = PhotBandCoeffs(
    c0=(1.008e11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    d0=(0.0, 0.0, 0.0, 0.0, 0.0),
    c1=(8.156e10, 9.728e8, -3.806e9, -4.384e9, -5.774e9, -5.249e9, -5.153e9),
    d1=(-1.879e10, -9.667e9, -5.602e9, -3.370e9, -1.825e9),
    c2=(1.067e11, -9.782e9, -7.193e9, -6.936e9, -6.893e9, -7.041e9, -7.193e9),
    d2=(-2.919e10, -1.185e10, -7.270e9, -4.222e9, -1.560e9),
)

# Table 2 coefficients for 10^8 <= T < 10^9 K (Itoh et al. 1996).
BAND_8_9 = PhotBandCoeffs(
    c0=(9.889e10, -4.524e8, -6.088e6, 4.269e7, 5.172e7, 4.910e7, 4.388e7),
    d0=(-1.135e8, 1.256e8, 5.149e7, 3.436e7, 1.005e7),
    c1=(1.813e11, -7.556e9, -3.304e9, -1.031e9, -1.764e9, -1.851e9, -1.928e9),
    d1=(1.652e9, -3.119e9, -1.839e9, -1.458e9, -8.956e8),
    c2=(9.750e10, 3.484e10, 5.199e9, -1.695e9, -2.865e9, -3.395e9, -3.418e9),
    d2=(-1.548e10, -9.338e9, -5.899e9, -3.035e9, -1.598e9),
)

# Table 2 coefficients for T >= 10^9 K (Itoh et al. 1996).
BAND_9_UP = PhotBandCoeffs(
    c0=(9.581e10, 4.107e8, 2.305e8, 2.236e8, 1.580e8, 2.165e8, 1.721e8),
    d0=(4.724e8, 2.976e8, 2.242e8, 7.937e7, 4.859e7),
    c1=(
        1.459e12,
        1.314e11,
        -1.169e11,
        -1.765e11,
        -1.867e11,
        -1.983e11,
        -1.896e11,
    ),
    d1=(-7.094e11, -3.697e11, -2.189e11, -1.273e11, -5.705e10),
    c2=(2.424e11, -3.669e9, -8.691e9, -7.967e9, -7.932e9, -7.987e9, -8.333e9),
    d2=(-2.254e10, -1.551e10, -7.793e9, -4.489e9, -2.185e9),
)


def _eval_fourier_row(
    c: tuple[float, ...],
    d: tuple[float, ...],
    cos_k: tuple[float, ...],
    sin_k: tuple[float, ...],
    last: float,
    xast: float,
    taudt: float,
) -> tuple[float, float]:
    """Evaluates Fourier series $(a_j, da_j/dT)$ from `c[0..7]`, `d[0..5]`.

    Implements Itoh et al. (1996) eq. 3.8 for a single polynomial coefficient
    $a_j(\\tau)$ ($j \\in \\{0, 1, 2\\}$) and its exact temperature derivative:
    $$a_j(\\tau) = \\tfrac{1}{2} c_{j,0} + \\sum_{k=1}^5 \\left(c_{j,k}
      \\cos(k\\omega_1\\tau) + d_{j,k}\\sin(k\\omega_1\\tau)\\right) +
      \\tfrac{1}{2} c_{j,6} \\cos(\\omega_2\\tau)$$

    Args:
        c: Cosine harmonic coefficients $c_{j,0 \\dots 6}$ (length 7).
        d: Sine harmonic coefficients $d_{j,1 \\dots 5}$ (length 5).
        cos_k: Harmonics $(\\cos(\\omega_1\\tau), \\dots,
            \\cos(5\\omega_1\\tau))$.
        sin_k: Harmonics $(\\sin(\\omega_1\\tau), \\dots,
            \\sin(5\\omega_1\\tau))$.
        last: High-frequency cosine $\\cos(\\omega_2\\tau)$.
        xast: High-frequency sine $\\sin(\\omega_2\\tau)$.
        taudt: Derivative $\\partial\\tau/\\partial T = 1 / (T \\ln 10)$.

    Returns:
        Tuple `(a_j, da_j_dT)`.
    """
    a = (
        0.5 * c[0]
        + c[1] * cos_k[0]
        + d[0] * sin_k[0]
        + c[2] * cos_k[1]
        + d[1] * sin_k[1]
        + c[3] * cos_k[2]
        + d[2] * sin_k[2]
        + c[4] * cos_k[3]
        + d[3] * sin_k[3]
        + c[5] * cos_k[4]
        + d[4] * sin_k[4]
        + 0.5 * c[6] * last
    )
    da_dt = (
        taudt
        * FAC1
        * (
            -c[1] * sin_k[0]
            + d[0] * cos_k[0]
            - c[2] * sin_k[1] * 2.0
            + d[1] * cos_k[1] * 2.0
            - c[3] * sin_k[2] * 3.0
            + d[2] * cos_k[2] * 3.0
            - c[4] * sin_k[3] * 4.0
            + d[3] * cos_k[3] * 4.0
            - c[5] * sin_k[4] * 5.0
            + d[4] * cos_k[4] * 5.0
        )
        - 0.5 * c[6] * xast * FAC2 * taudt
    )
    return a, da_dt


def phot_neu(inp: _inputs.NeuInputs) -> _inputs.NeuRate:
    """Evaluates volumetric photoneutrino energy loss rate and derivatives.

    Implements Itoh et al. (1996) eqs. 3.2-3.8 (`phot_neu` in `mod_neu.f90`):
    $$Q_{\\mathrm{phot}} = \\mathrm{TFAC4}\\,(1 -
      \\mathrm{TFAC3}\\,q_{\\mathrm{phot}})\\,(\\rho Y_e)\\,\\lambda^5
      f_{\\mathrm{phot}}(\\lambda, \\xi)$$

    Args:
        inp: Precomputed thermodynamic and composition invariants (`NeuInputs`).

    Returns:
        Volumetric rate `NeuRate` in $\\mathrm{erg\\,cm^{-3}\\,s^{-1}}$ before
        dividing by $\\rho$, or zero when $T \\le 10^7\\,\\mathrm{K}$.
    """
    if inp.temp <= 1.0e7:
        return _inputs.NeuRate()

    if inp.temp < 1.0e8:
        tau = inp.logtemp - 7.0
        cc = 0.5654 + tau
        dccdt = 1.0 / (_const.LN10 * inp.temp)
        band = BAND_7_8
    elif inp.temp < 1.0e9:
        tau = inp.logtemp - 8.0
        cc = 1.5654
        dccdt = 0.0
        band = BAND_8_9
    else:
        tau = inp.logtemp - 9.0
        cc = 1.5654
        dccdt = 0.0
        band = BAND_9_UP

    taudt = _const.ILN10 * inp.tempi
    cos1 = math.cos(FAC1 * tau)
    sin1 = math.sin(FAC1 * tau)
    sin2 = 2.0 * sin1 * cos1
    cos2 = 2.0 * cos1 * cos1 - 1.0
    sin3 = sin1 * (3.0 - 4.0 * sin1 * sin1)
    cos3 = cos1 * (4.0 * cos1 * cos1 - 3.0)
    sin4 = 2.0 * sin2 * cos2
    cos4 = 2.0 * cos2 * cos2 - 1.0
    sin5 = sin1 * (5.0 - sin1 * sin1 * (20.0 - 16.0 * sin1 * sin1))
    cos5 = cos1 * (cos1 * cos1 * (16.0 * cos1 * cos1 - 20.0) + 5.0)
    last = math.cos(FAC2 * tau)
    xast = math.sin(FAC2 * tau)

    cos_k = (cos1, cos2, cos3, cos4, cos5)
    sin_k = (sin1, sin2, sin3, sin4, sin5)
    a0, f0 = _eval_fourier_row(
        band.c0, band.d0, cos_k, sin_k, last, xast, taudt
    )
    a1, f1 = _eval_fourier_row(
        band.c1, band.d1, cos_k, sin_k, last, xast, taudt
    )
    a2, f2 = _eval_fourier_row(
        band.c2, band.d2, cos_k, sin_k, last, xast, taudt
    )

    dum = a0 + a1 * inp.zeta + a2 * inp.zeta2
    dumdt = (
        f0
        + f1 * inp.zeta
        + a1 * inp.zetadt
        + f2 * inp.zeta2
        + 2.0 * a2 * inp.zeta * inp.zetadt
    )
    dumdd = a1 * inp.zetadd + 2.0 * a2 * inp.zeta * inp.zetadd
    dumda = a1 * inp.zetada + 2.0 * a2 * inp.zeta * inp.zetada
    dumdz = a1 * inp.zetadz + 2.0 * a2 * inp.zeta * inp.zetadz

    z = math.exp(-cc * inp.zeta)
    xnum = dum * z
    xnumdt = dumdt * z - dum * z * (dccdt * inp.zeta + inp.zetadt * cc)
    xnumdd = dumdd * z - dum * z * cc * inp.zetadd
    xnumda = dumda * z - dum * z * cc * inp.zetada
    xnumdz = dumdz * z - dum * z * cc * inp.zetadz

    xden = (
        inp.zeta3
        + 6.290e-3 * inp.xlm1
        + 7.483e-3 * inp.xlm2
        + 3.061e-4 * inp.xlm3
    )
    dz3 = 3.0 * inp.zeta2
    xdendt = dz3 * inp.zetadt - inp.xldt * (
        6.290e-3 * inp.xlm2
        + 2.0 * 7.483e-3 * inp.xlm3
        + 3.0 * 3.061e-4 * inp.xlm4
    )
    xdendd = dz3 * inp.zetadd
    xdenda = dz3 * inp.zetada
    xdendz = dz3 * inp.zetadz

    inv_xden = 1.0 / xden
    fphot = xnum * inv_xden
    fphotdt = (xnumdt - fphot * xdendt) * inv_xden
    fphotdd = (xnumdd - fphot * xdendd) * inv_xden
    fphotda = (xnumda - fphot * xdenda) * inv_xden
    fphotdz = (xnumdz - fphot * xdendz) * inv_xden

    q_a0 = 1.0 + 2.045 * inp.xl
    q_xnum = 0.666 * math.pow(q_a0, -2.066)
    q_xnumdt = -2.066 * q_xnum / q_a0 * 2.045 * inp.xldt

    q_dum = (
        1.875e8 * inp.xl
        + 1.653e8 * inp.xl2
        + 8.499e8 * inp.xl3
        - 1.604e8 * inp.xl4
    )
    q_dumdt = inp.xldt * (
        1.875e8
        + 2.0 * 1.653e8 * inp.xl
        + 3.0 * 8.499e8 * inp.xl2
        - 4.0 * 1.604e8 * inp.xl3
    )

    inv_qdum = 1.0 / q_dum
    q_xden = 1.0 + inp.rm * inv_qdum
    q_xdendt = -inp.rm * inv_qdum * inv_qdum * q_dumdt
    q_xdendd = inp.rmdd * inv_qdum
    q_xdenda = inp.rmda * inv_qdum
    q_xdendz = inp.rmdz * inv_qdum

    inv_qxden = 1.0 / q_xden
    qphot = q_xnum * inv_qxden
    qphotdt = (q_xnumdt - qphot * q_xdendt) * inv_qxden
    neg_qp = -qphot * inv_qxden
    qphotdd = neg_qp * q_xdendd
    qphotda = neg_qp * q_xdenda
    qphotdz = neg_qp * q_xdendz

    s0 = inp.xl5 * fphot
    s0dt = 5.0 * inp.xl4 * inp.xldt * fphot + inp.xl5 * fphotdt
    s0dd = inp.xl5 * fphotdd
    s0da = inp.xl5 * fphotda
    s0dz = inp.xl5 * fphotdz

    s1 = inp.rm * s0
    s1dt = inp.rm * s0dt
    s1dd = inp.rm * s0dd + inp.rmdd * s0
    s1da = inp.rm * s0da + inp.rmda * s0
    s1dz = inp.rm * s0dz + inp.rmdz * s0

    w1 = _inputs.TFAC4 * (1.0 - _inputs.TFAC3 * qphot)
    w2 = -_inputs.TFAC4 * _inputs.TFAC3
    sphot = w1 * s1
    if sphot <= 0.0:
        return _inputs.NeuRate()
    return _inputs.NeuRate(
        val=sphot,
        d_t=w1 * s1dt + w2 * qphotdt * s1,
        d_rho=w1 * s1dd + w2 * qphotdd * s1,
        d_abar=w1 * s1da + w2 * qphotda * s1,
        d_zbar=w1 * s1dz + w2 * qphotdz * s1,
    )
