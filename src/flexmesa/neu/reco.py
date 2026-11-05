# SPDX-License-Identifier: LGPL-3.0-or-later
"""Recombination (`reco_neu`) neutrino energy loss rates and Antia (1993) fits.

Ports `ifermi12`, `zfermim12` (Antia 1993, ApJS, 84, 101), and `reco_neu` (Itoh
et al. 1996, ApJS, 102, 411, Section 6) from `neu/private/mod_neu.f90`.

Fermi-Dirac Inversion and Recombination Rate:
    - `ifermi12` evaluates the inverse Fermi-Dirac integral $F_{1/2}^{-1}(f)$
      to relative accuracy $4.19 \\times 10^{-9}$ across non-degenerate
      ($f < 4$) and degenerate ($f \\ge 4$) regimes.
    - `zfermim12` evaluates $F_{-1/2}(\\nu) = 2\\,dF_{1/2}/d\\nu$ to relative
      accuracy $1.23 \\times 10^{-12}$.
    - `reco_neu` evaluates free-bound electron capture neutrino emission over
      electron degeneracy parameter $\\nu \\in [-20, 10]$.
"""

from __future__ import annotations

import math

from flexmesa.neu import inputs as _inputs


def ifermi12(f: float) -> float:
    """Evaluates inverse Fermi-Dirac integral $F_{1/2}^{-1}(f)$ (Antia 1993).

    Uses two minimax rational approximations (`ifermi12` in `mod_neu.f90`):
    - For $f < 4$: $\\nu = \\ln(f\\,P_4(f)/Q_3(f))$
    - For $f \\ge 4$: $\\nu = f^{2/3}\\,P_6(f^{-2/3})/Q_5(f^{-2/3})$

    Args:
        f: Dimensionless Fermi-Dirac integral value $F_{1/2}(\\nu) > 0$.

    Returns:
        Dimensionless electron chemical potential $\\nu = \\mu_e / (k_B T)$.
    """
    a1 = (
        1.999266880833e4,
        5.702479099336e3,
        6.610132843877e2,
        3.818838129486e1,
        1.0,
    )
    b1 = (
        1.771804140488e4,
        -2.014785161019e3,
        9.130355392717e1,
        -1.670718177489,
    )
    a2 = (
        -1.277060388085e-2,
        7.187946804945e-2,
        -4.262314235106e-1,
        4.997559426872e-1,
        -1.285579118012,
        -3.930805454272e-1,
        1.0,
    )
    b2 = (
        -9.745794806288e-3,
        5.485432756838e-2,
        -3.299466243260e-1,
        4.077841975923e-1,
        -1.145531476975,
        -6.067091689181e-2,
    )
    if f < 4.0:
        rn = f + a1[3]
        for i in (2, 1, 0):
            rn = rn * f + a1[i]
        den = b1[3]
        for i in (2, 1, 0):
            den = den * f + b1[i]
        return math.log(f * rn / den)

    ff = 1.0 / math.pow(f, 1.0 / 1.5)
    rn = ff + a2[5]
    for i in (4, 3, 2, 1, 0):
        rn = rn * ff + a2[i]
    den = b2[5]
    for i in (4, 3, 2, 1, 0):
        den = den * ff + b2[i]
    return rn / (den * ff)


def zfermim12(x: float) -> float:
    """Evaluates Fermi-Dirac integral $F_{-1/2}(x)$ (Antia 1993).

    Uses two minimax rational approximations (`zfermim12` in `mod_neu.f90`):
    - For $x < 2$: $F_{-1/2}(x) = e^x\\,P_7(e^x)/Q_7(e^x)$
    - For $x \\ge 2$: $F_{-1/2}(x) = \\sqrt{x}\\,P_{11}(x^{-2})/Q_{11}(x^{-2})$

    Args:
        x: Dimensionless electron chemical potential $\\nu = \\mu_e / (k_B T)$.

    Returns:
        Fermi-Dirac integral value $F_{-1/2}(x) = 2\\,dF_{1/2}/dx$.
    """
    a1 = (
        1.71446374704454e7,
        3.88148302324068e7,
        3.16743385304962e7,
        1.14587609192151e7,
        1.83696370756153e6,
        1.14980998186874e5,
        1.98276889924768e3,
        1.0,
    )
    b1 = (
        9.67282587452899e6,
        2.87386436731785e7,
        3.26070130734158e7,
        1.77657027846367e7,
        4.81648022267831e6,
        6.13709569333207e5,
        3.13595854332114e4,
        4.35061725080755e2,
    )
    a2 = (
        -4.46620341924942e-15,
        -1.58654991146236e-12,
        -4.44467627042232e-10,
        -6.84738791621745e-8,
        -6.64932238528105e-6,
        -3.69976170193942e-4,
        -1.12295393687006e-2,
        -1.60926102124442e-1,
        -8.52408612877447e-1,
        -7.45519953763928e-1,
        2.98435207466372,
        1.0,
    )
    b2 = (
        -2.23310170962369e-15,
        -7.94193282071464e-13,
        -2.22564376956228e-10,
        -3.43299431079845e-8,
        -3.33919612678907e-6,
        -1.86432212187088e-4,
        -5.69764436880529e-3,
        -8.34904593067194e-2,
        -4.78770844009440e-1,
        -4.99759250374148e-1,
        1.86795964993052,
        4.16485970495288e-1,
    )
    if x < 2.0:
        xx = math.exp(x)
        rn = xx + a1[6]
        for i in (5, 4, 3, 2, 1, 0):
            rn = rn * xx + a1[i]
        den = b1[7]
        for i in (6, 5, 4, 3, 2, 1, 0):
            den = den * xx + b1[i]
        return xx * rn / den

    xx = 1.0 / (x * x)
    rn = xx + a2[10]
    for i in (9, 8, 7, 6, 5, 4, 3, 2, 1, 0):
        rn = rn * xx + a2[i]
    den = b2[11]
    for i in (10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0):
        den = den * xx + b2[i]
    return math.sqrt(x) * rn / den


def reco_neu(inp: _inputs.NeuInputs) -> _inputs.NeuRate:
    """Evaluates volumetric recombination neutrino energy loss rate.

    Implements Itoh et al. (1996) Section 6 (`reco_neu` in `mod_neu.f90`),
    evaluating the bound-free electron capture fit $J(\\zeta, \\nu)$ and its
    partial derivatives over $\\nu \\in [-20, 10]$.

    Args:
        inp: Precomputed thermodynamic and composition invariants (`NeuInputs`).

    Returns:
        Volumetric rate `NeuRate` in $\\mathrm{erg\\,cm^{-3}\\,s^{-1}}$ before
        dividing by $\\rho$, or zero when $\\nu \\notin [-20, 10]$.
    """
    xnum = 1.10520e8 * inp.den * inp.ye / (inp.temp * math.sqrt(inp.temp))
    xnumdt = -1.50 * xnum * inp.tempi
    xnumdd = xnum * inp.deni
    xnumda = -xnum * inp.abari
    xnumdz = xnum * inp.zbari

    # Evaluate electron degeneracy parameter nu = mu_e / (k_B T) via
    # Antia (1993) inverse Fermi-Dirac rational approximation.
    nu = ifermi12(xnum)
    if nu < -20.0 or nu > 10.0:
        return _inputs.NeuRate()

    a0 = 1.0 / (0.5 * zfermim12(nu))
    nudt = a0 * xnumdt
    nudd = a0 * xnumdd
    nuda = a0 * xnumda
    nudz = a0 * xnumdz
    nu2 = nu * nu
    nu3 = nu2 * nu

    if nu < 0.0:
        a1, a2, a3 = 1.51e-2, 2.42e-1, 1.21
        b, c, d = 3.71e-2, 9.06e-1, 9.28e-1
        f1, f2, f3 = 0.0, 0.0, 0.0
    else:
        a1, a2, a3 = 1.23e-2, 2.66e-1, 1.30
        b, c, d = 1.17e-1, 8.97e-1, 1.77e-1
        f1, f2, f3 = -1.20e-2, 2.29e-2, -1.04e-3

    zeta = 1.579e5 * inp.zbar * inp.zbar * inp.tempi
    zetadt = -zeta * inp.tempi
    zetadz = 2.0 * zeta * inp.zbari

    c00_p = 1.0 / (1.0 + f1 * nu + f2 * nu2 + f3 * nu3)
    c01_p = f1 + f2 * 2.0 * nu + f3 * 3.0 * nu2
    dum = zeta * c00_p
    k_nu = -c00_p * c00_p * zeta * c01_p
    dumdt = zetadt * c00_p + k_nu * nudt
    dumdd = k_nu * nudd
    dumda = k_nu * nuda
    dumdz = zetadz * c00_p + k_nu * nudz

    inv_dum = 1.0 / dum
    dd00 = math.pow(dum, -2.25)
    dd01 = math.pow(dum, -4.55)
    c00 = a1 * inv_dum + a2 * dd00 + a3 * dd01
    c01 = -(a1 * inv_dum + 2.25 * a2 * dd00 + 4.55 * a3 * dd01) * inv_dum

    z_cnu = math.exp(c * nu)
    dd00_g = b * z_cnu * (1.0 + d * dum)
    gum = 1.0 + dd00_g
    gumdt = dd00_g * c * nudt + b * z_cnu * d * dumdt
    gumdd = dd00_g * c * nudd + b * z_cnu * d * dumdd
    gumda = dd00_g * c * nuda + b * z_cnu * d * dumda
    gumdz = dd00_g * c * nudz + b * z_cnu * d * dumdz

    z_nu = math.exp(nu)
    inv_gum = 1.0 / gum
    bigj = c00 * z_nu * inv_gum
    bigjdt = (c01 * dumdt + c00 * nudt - c00 * inv_gum * gumdt) * z_nu * inv_gum
    bigjdd = (c01 * dumdd + c00 * nudd - c00 * inv_gum * gumdd) * z_nu * inv_gum
    bigjda = (c01 * dumda + c00 * nuda - c00 * inv_gum * gumda) * z_nu * inv_gum
    bigjdz = (c01 * dumdz + c00 * nudz - c00 * inv_gum * gumdz) * z_nu * inv_gum

    # Combine bound-free prefactor, J(zeta, nu), and Pauli blocking factor
    # 1 / (1 + exp(zeta + nu)) into the volumetric recombination rate.
    z_zn = math.exp(zeta + nu)
    inv_1pz = 1.0 / (1.0 + z_zn)
    inv_bigj = 1.0 / bigj
    sreco = (
        _inputs.TFAC6
        * 2.649e-18
        * inp.ye
        * math.pow(inp.zbar, 13.0)
        * inp.den
        * bigj
        * inv_1pz
    )
    return _inputs.NeuRate(
        val=sreco,
        d_t=sreco * (bigjdt * inv_bigj - z_zn * (zetadt + nudt) * inv_1pz),
        d_rho=sreco * (inp.deni + bigjdd * inv_bigj - z_zn * nudd * inv_1pz),
        d_abar=sreco * (-inp.abari + bigjda * inv_bigj - z_zn * nuda * inv_1pz),
        d_zbar=sreco
        * (
            14.0 * inp.zbari
            + bigjdz * inv_bigj
            - z_zn * (zetadz + nudz) * inv_1pz
        ),
    )
