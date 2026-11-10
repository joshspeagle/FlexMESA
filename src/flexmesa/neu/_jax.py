# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native JAX ``float64`` thermal neutrino loss kernels (all 5 Itoh channels).

Implements differentiable pure-JAX ``float64`` kernels for all five Itoh et al.
(1996, ApJS, 102, 411) thermal neutrino reaction channels (`pair`, `plas`,
`phot`, `brem`, `reco`), matching ``crates/flexmesa-neu/src/`` and supporting
``jax.jit``, ``jax.vmap``, ``jax.grad``, and ``jax.jacfwd``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants as _const
from flexmesa.neu import brem as _brem
from flexmesa.neu import inputs as _inputs
from flexmesa.neu import phot as _phot
from flexmesa.neu import reco as _reco


def _horner_jax(coeffs: tuple[float, ...], z: jax.Array) -> jax.Array:
    """Evaluates polynomial $\\sum_i c_i z^i$ via Horner's scheme in JAX."""
    acc = jnp.asarray(coeffs[-1], dtype=jnp.float64)
    for c in reversed(coeffs[:-1]):
        acc = acc * z + c
    return acc


def zfermim12_jax(x: jax.Array) -> jax.Array:
    """Evaluates Fermi-Dirac integral $F_{-1/2}(x)$ (Antia 1993) in JAX.

    Args:
        x: Dimensionless electron chemical potential $\\nu = \\mu_e / (k_B T)$
            (scalar or ``(*batch,)`` in ``float64``).

    Returns:
        Fermi-Dirac integral $F_{-1/2}(x) = 2\\,dF_{1/2}/dx$ in ``float64``.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
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
    x_low = jnp.minimum(x, 2.0)
    xx_lo = jnp.exp(x_low)
    val_lo = xx_lo * _horner_jax(a1, xx_lo) / _horner_jax(b1, xx_lo)

    x_hi = jnp.maximum(x, 2.0)
    xx_hi = 1.0 / (x_hi * x_hi)
    val_hi = jnp.sqrt(x_hi) * _horner_jax(a2, xx_hi) / _horner_jax(b2, xx_hi)
    return jnp.where(x < 2.0, val_lo, val_hi)


@jax.custom_jvp
def ifermi12_jax(f: jax.Array) -> jax.Array:
    """Evaluates inverse Fermi-Dirac integral $F_{1/2}^{-1}(f)$ in JAX.

    Uses a custom JVP matching MESA's analytical identity
    $d F_{1/2}^{-1}(f) / df = 2 / F_{-1/2}(\\nu)$ (`zfermim12`).

    Args:
        f: Dimensionless Fermi-Dirac integral value $F_{1/2}(\\nu) > 0$.

    Returns:
        Dimensionless electron chemical potential $\\nu = \\mu_e / (k_B T)$.
    """
    f = jnp.asarray(f, dtype=jnp.float64)
    f_lo = jnp.clip(f, 1.0e-300, 4.0)
    rn_lo = _horner_jax(_reco._IFERMI12_A1, f_lo)
    den_lo = _horner_jax(_reco._IFERMI12_B1, f_lo)
    val_lo = jnp.log(f_lo * rn_lo / den_lo)

    f_hi = jnp.maximum(f, 4.0)
    ff = 1.0 / jnp.power(f_hi, 1.0 / 1.5)
    rn_hi = _horner_jax(_reco._IFERMI12_A2, ff)
    den_hi = _horner_jax(_reco._IFERMI12_B2, ff)
    val_hi = rn_hi / (den_hi * ff)
    return jnp.where(f < 4.0, val_lo, val_hi)


@ifermi12_jax.defjvp
def _ifermi12_jvp(
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """Evaluates JVP of `ifermi12_jax` via $d\\nu/df = 2 / F_{-1/2}(\\nu)$."""
    (f,) = primals
    (df,) = tangents
    nu = ifermi12_jax(f)
    dnu_df = 1.0 / (0.5 * zfermim12_jax(nu))
    return nu, df * dnu_df


def _cutoff_factor_jax(t: jax.Array, log10_tlim: float = 7.5) -> jax.Array:
    """Evaluates smooth low-temperature cosine cutoff $f_{\\mathrm{cut}}(T)$."""
    logtemp = jnp.log10(jnp.maximum(t, 1.0))
    dtlim = jnp.maximum(log10_tlim - _inputs.LOG10_TMIN_NEU, 1.0e-12)
    phase = (logtemp - _inputs.LOG10_TMIN_NEU) / dtlim
    f_cos = 0.5 * (1.0 - jnp.cos(_const.PI * phase))
    return jnp.where(
        (t <= _inputs.TMIN_NEU) | (logtemp <= _inputs.LOG10_TMIN_NEU),
        0.0,
        jnp.where(logtemp <= log10_tlim, f_cos, 1.0),
    )


def _pair_vol_jax(
    t: jax.Array, rho: jax.Array, abar: jax.Array, zbar: jax.Array
) -> jax.Array:
    """Evaluates volumetric pair-neutrino loss $Q_{\\mathrm{pair}}$ in JAX."""
    ye = zbar / abar
    rm = rho * ye
    xl = (t * 1.0e-9) * _inputs.CON1
    xl2 = xl * xl
    xl3 = xl2 * xl
    xl4 = xl2 * xl2
    xl6 = xl4 * xl2
    xl8 = xl4 * xl4
    xlm1 = 1.0 / xl
    xlm2 = xlm1 * xlm1
    xlm3 = xlm1 * xlm2
    xlp5 = jnp.sqrt(xl)

    zeta = jnp.power(rm * 1.0e-9, _const.ONE_THIRD) * xlm1
    zeta2 = zeta * zeta
    zeta3 = zeta2 * zeta

    gl = 1.0 - 13.04 * xl2 + 133.5 * xl4 + 1534.0 * xl6 + 918.6 * xl8
    a1 = 6.002e19 + 2.084e20 * zeta + 1.872e21 * zeta2
    b1 = jnp.where(t < 1.0e10, jnp.exp(-5.5924 * zeta), jnp.exp(-4.9924 * zeta))
    den_a1 = jnp.where(
        t < 1.0e10,
        9.383e-1 * xlm1 - 4.141e-1 * xlm2 + 5.829e-2 * xlm3,
        1.2383 * xlm1 - 8.141e-1 * xlm2,
    )
    fpair = (a1 * b1) / (zeta3 + den_a1)
    q_xnum = 1.0 / (10.7480 * xl2 + 0.3967 * xlp5 + 1.005)
    d_a1 = 7.692e7 * xl3 + 9.715e6 * xlp5
    qpair = q_xnum * jnp.power(1.0 + rm / d_a1, -0.3)
    return (
        _inputs.TFAC4
        * (1.0 + _inputs.TFAC3 * qpair)
        * gl
        * jnp.exp(-2.0 * xlm1)
        * fpair
    )


def _plas_vol_jax(
    t: jax.Array, rho: jax.Array, abar: jax.Array, zbar: jax.Array
) -> jax.Array:
    """Evaluates volumetric plasmon-neutrino loss $Q_{\\mathrm{plas}}$."""
    ye = zbar / abar
    rm = rho * ye
    xl = (t * 1.0e-9) * _inputs.CON1
    xl9 = jnp.power(xl, 9)

    a2_p = jnp.power(1.019e-6 * rm, _const.TWO_THIRDS)
    b1_p = jnp.sqrt(1.0 + a2_p)
    gl2 = 1.1095e11 * rm / (t * t * b1_p)
    gl_p = jnp.sqrt(gl2)
    gl12 = jnp.sqrt(gl_p)
    gl32 = gl_p * gl12
    gl72 = gl2 * gl32
    gl6 = gl2 * gl2 * gl2
    ft = 2.4 + 0.6 * gl12 + 0.51 * gl_p + 1.25 * gl32
    fl = (8.6 * gl2 + 1.35 * gl72) / (225.0 - 17.0 * gl_p + gl2)

    cc = jnp.log10(2.0 * rm)
    xlnt = jnp.log10(t)
    xnum = _const.ONE_SIXTH * (17.5 + cc - 3.0 * xlnt)
    xden = _const.ONE_SIXTH * (-24.5 + cc + 3.0 * xlnt)
    xnum_s = jnp.clip(xnum, -0.7, 0.7)
    p_a1 = 0.39 - 1.25 * xnum_s - 0.35 * jnp.sin(4.5 * xnum_s)
    p_b1 = 0.3 * jnp.exp(-jnp.square(4.5 * xnum_s + 0.9))
    c_val = jnp.minimum(0.0, xden - 1.6 + 1.25 * xnum_s)
    a3_val = c_val / (0.57 - 0.25 * xnum_s)
    fxy_in = 1.05 + (p_a1 - p_b1) * jnp.exp(-a3_val * a3_val)
    fxy = jnp.where((jnp.abs(xnum) > 0.7) | (xden < 0.0), 1.0, fxy_in)
    return _inputs.TFAC0 * 3.0e21 * xl9 * gl6 * jnp.exp(-gl_p) * (ft + fl) * fxy


def _phot_vol_jax(
    t: jax.Array, rho: jax.Array, abar: jax.Array, zbar: jax.Array
) -> jax.Array:
    """Evaluates volumetric photoneutrino loss $Q_{\\mathrm{phot}}$ in JAX."""
    ye = zbar / abar
    rm = rho * ye
    logtemp = jnp.log10(t)
    xl = (t * 1.0e-9) * _inputs.CON1
    xl2 = xl * xl
    xl3 = xl2 * xl
    xl4 = xl2 * xl2
    xl5 = xl4 * xl
    xlm1 = 1.0 / xl
    xlm2 = xlm1 * xlm1
    xlm3 = xlm1 * xlm2
    zeta = jnp.power(rm * 1.0e-9, _const.ONE_THIRD) * xlm1
    zeta2 = zeta * zeta
    zeta3 = zeta2 * zeta

    tau = jnp.where(
        t < 1.0e8,
        logtemp - 7.0,
        jnp.where(t < 1.0e9, logtemp - 8.0, logtemp - 9.0),
    )
    cc = jnp.where(t < 1.0e8, 0.5654 + tau, 1.5654)

    cos1 = jnp.cos(_phot.FAC1 * tau)
    sin1 = jnp.sin(_phot.FAC1 * tau)
    sin2 = 2.0 * sin1 * cos1
    cos2 = 2.0 * cos1 * cos1 - 1.0
    sin3 = sin1 * (3.0 - 4.0 * sin1 * sin1)
    cos3 = cos1 * (4.0 * cos1 * cos1 - 3.0)
    sin4 = 2.0 * sin2 * cos2
    cos4 = 2.0 * cos2 * cos2 - 1.0
    sin5 = sin1 * (5.0 - sin1 * sin1 * (20.0 - 16.0 * sin1 * sin1))
    cos5 = cos1 * (cos1 * cos1 * (16.0 * cos1 * cos1 - 20.0) + 5.0)
    last = jnp.cos(_phot.FAC2 * tau)
    cos_k = (cos1, cos2, cos3, cos4, cos5)
    sin_k = (sin1, sin2, sin3, sin4, sin5)

    def _row(
        c78: tuple[float, ...],
        d78: tuple[float, ...],
        c89: tuple[float, ...],
        d89: tuple[float, ...],
        c9u: tuple[float, ...],
        d9u: tuple[float, ...],
    ) -> jax.Array:
        c = [
            jnp.where(t < 1.0e8, c78[k], jnp.where(t < 1.0e9, c89[k], c9u[k]))
            for k in range(7)
        ]
        d = [
            jnp.where(t < 1.0e8, d78[k], jnp.where(t < 1.0e9, d89[k], d9u[k]))
            for k in range(5)
        ]
        acc = 0.5 * c[0] + 0.5 * c[6] * last
        for k in range(5):
            acc = acc + c[k + 1] * cos_k[k] + d[k] * sin_k[k]
        return acc

    b78, b89, b9u = _phot.BAND_7_8, _phot.BAND_8_9, _phot.BAND_9_UP
    a0 = _row(b78.c0, b78.d0, b89.c0, b89.d0, b9u.c0, b9u.d0)
    a1 = _row(b78.c1, b78.d1, b89.c1, b89.d1, b9u.c1, b9u.d1)
    a2 = _row(b78.c2, b78.d2, b89.c2, b89.d2, b9u.c2, b9u.d2)

    xnum = (a0 + a1 * zeta + a2 * zeta2) * jnp.exp(-cc * zeta)
    xden = zeta3 + 6.290e-3 * xlm1 + 7.483e-3 * xlm2 + 3.061e-4 * xlm3
    fphot = xnum / xden
    q_xnum = 0.666 * jnp.power(1.0 + 2.045 * xl, -2.066)
    q_dum = 1.875e8 * xl + 1.653e8 * xl2 + 8.499e8 * xl3 - 1.604e8 * xl4
    qphot = q_xnum / (1.0 + rm / q_dum)
    sphot = _inputs.TFAC4 * (1.0 - _inputs.TFAC3 * qphot) * rm * xl5 * fphot
    return jnp.where((t <= 1.0e7) | (sphot <= 0.0), 0.0, sphot)


def _brem_vol_jax(
    t: jax.Array, rho: jax.Array, abar: jax.Array, zbar: jax.Array
) -> jax.Array:
    """Evaluates volumetric bremsstrahlung loss $Q_{\\mathrm{brem}}$ in JAX."""
    ye = zbar / abar
    rm = rho * ye
    den6 = rho * 1.0e-6
    t8 = t * 1.0e-8
    t812 = jnp.sqrt(t8)
    t832 = t8 * t812
    t82 = t8 * t8
    t83 = t82 * t8
    t86 = t83 * t83
    t8m1 = 1.0 / t8
    t8m2 = t8m1 * t8m1
    t8m5 = t8m2 * t8m2 * t8m1

    eta = rm / (7.05e6 * t832 + 5.12e4 * t83)
    etam1 = 1.0 / eta
    etam2 = etam1 * etam1
    f_wk = 1.0 / (23.5 + 6.83e4 * t8m2 + 7.81e8 * t8m5) + (
        1.26 * (1.0 + etam1) / (1.0 + 1.47 * etam1 + 3.29e-2 * etam2)
    )
    ga0 = 230.0 + 6.7e5 * t8m2 + 7.66e9 * t8m5
    gxnum = 1.0 / (ga0 * (1.0 + rm * 1.0e-9))
    gc00 = 7.75e5 * t832 + 247.0 * jnp.power(t8, 3.85)
    gc01 = 4.07 + 0.0240 * jnp.power(t8, 1.4)
    gc02 = 4.59e-5 * jnp.power(t8, -0.110)
    g_wk = gxnum + 1.0 / (gc00 / rm + gc01 + gc02 * jnp.power(rho, 0.656))
    pref = 0.5738 * zbar * ye * t86 * rho
    weak = pref * (_inputs.TFAC4 * f_wk - _inputs.TFAC5 * g_wk)

    u_ang = _brem.FAC3 * (jnp.log10(rho) - 3.0)
    c1, s1 = jnp.cos(u_ang), jnp.sin(u_ang)
    s2, c2 = 2.0 * s1 * c1, 2.0 * c1 * c1 - 1.0
    s3, c3 = s1 * (3.0 - 4.0 * s1 * s1), c1 * (4.0 * c1 * c1 - 3.0)
    s4, c4 = 2.0 * s2 * c2, 2.0 * c2 * c2 - 1.0
    c5 = c1 * (c1 * c1 * (16.0 * c1 * c1 - 20.0) + 5.0)
    ck = (c1, c2, c3, c4, c5)
    sk = (s1, s2, s3, s4)

    def _f_liq(
        c0: float,
        lin: float,
        cst: float,
        a: tuple[float, ...],
        b: tuple[float, ...],
    ) -> jax.Array:
        val = 0.5 * c0 + lin * u_ang + cst + a[4] * ck[4]
        for k in range(4):
            val = val + a[k] * ck[k] + b[k] * sk[k]
        return val

    fb = _f_liq(
        0.17946,
        0.00945,
        0.34529,
        (-0.05821, -0.01089, -0.01147, -0.00656, -0.00519),
        (-0.04969, -0.01584, -0.00504, -0.00281),
    )
    ft = _f_liq(
        0.06781,
        -0.02342,
        0.24819,
        (-0.00944, -0.01289, -0.00589, -0.00404, -0.00330),
        (-0.02213, -0.01136, -0.00467, -0.00131),
    )
    gb = _f_liq(
        0.00766,
        -0.01259,
        0.07917,
        (-0.00710, -0.00028, 0.00232, 0.00044, 0.00158),
        (0.02300, -0.01078, 0.00118, -0.00089),
    )
    gt = _f_liq(
        -0.00769,
        -0.00829,
        0.05211,
        (0.00356, -0.00184, 0.00146, 0.00031, 0.00069),
        (0.01052, -0.00354, -0.00014, -0.00018),
    )
    dum = (
        2.275e-1 * zbar * zbar * t8m1 * jnp.power(den6 / abar, _const.ONE_THIRD)
    )
    gm1 = 1.0 / dum
    gm13 = jnp.power(gm1, _const.ONE_THIRD)
    gm23 = gm13 * gm13
    v = -0.05483 - 0.01946 * gm13 + 1.86310 * gm23 - 0.78873 * gm1
    w = -0.06711 + 0.06859 * gm13 + 1.74360 * gm23 - 0.74498 * gm1
    f_l = v * fb + (1.0 - v) * ft
    g_l = w * gb + (1.0 - w) * gt
    liq = pref * (_inputs.TFAC4 * f_l - _inputs.TFAC5 * g_l)

    u = jnp.power(den6 * ye, _const.TWO_THIRDS)
    tfermi = 5.9302e9 * (jnp.sqrt(u) - 1.0)
    tf_safe = jnp.where(jnp.abs(tfermi) < 1.0, 1.0, tfermi)
    tfrac = (t / tf_safe - 0.05) / 0.25
    alfa = 0.5 * (1.0 - jnp.cos(_const.PI * tfrac))
    blend = alfa * weak + (1.0 - alfa) * liq
    return jnp.where(
        t >= 0.3 * tfermi, weak, jnp.where(t <= 0.05 * tfermi, liq, blend)
    )


def _reco_vol_jax(
    t: jax.Array, rho: jax.Array, abar: jax.Array, zbar: jax.Array
) -> jax.Array:
    """Evaluates volumetric recombination loss $Q_{\\mathrm{reco}}$ in JAX."""
    ye = zbar / abar
    xnum = 1.10520e8 * rho * ye / (t * jnp.sqrt(t))
    nu_raw = ifermi12_jax(xnum)
    in_range = (nu_raw >= -20.0) & (nu_raw <= 10.0)
    nu = jnp.where(in_range, jnp.clip(nu_raw, -20.0, 10.0), 0.0)
    nu2 = nu * nu
    nu3 = nu2 * nu

    a1 = jnp.where(nu < 0.0, 1.51e-2, 1.23e-2)
    a2 = jnp.where(nu < 0.0, 2.42e-1, 2.66e-1)
    a3 = jnp.where(nu < 0.0, 1.21, 1.30)
    b = jnp.where(nu < 0.0, 3.71e-2, 1.17e-1)
    c = jnp.where(nu < 0.0, 9.06e-1, 8.97e-1)
    d = jnp.where(nu < 0.0, 9.28e-1, 1.77e-1)
    f1 = jnp.where(nu < 0.0, 0.0, -1.20e-2)
    f2 = jnp.where(nu < 0.0, 0.0, 2.29e-2)
    f3 = jnp.where(nu < 0.0, 0.0, -1.04e-3)

    zeta = 1.579e5 * zbar * zbar / t
    dum = zeta / (1.0 + f1 * nu + f2 * nu2 + f3 * nu3)
    c00 = a1 / dum + a2 * jnp.power(dum, -2.25) + a3 * jnp.power(dum, -4.55)
    gum = 1.0 + b * jnp.exp(c * nu) * (1.0 + d * dum)
    bigj = c00 * jnp.exp(nu) / gum
    exp_arg = jnp.where(in_range, jnp.minimum(zeta + nu, 500.0), 0.0)
    q_raw = bigj / (1.0 + jnp.exp(exp_arg))
    sreco = _inputs.TFAC6 * 2.649e-18 * ye * jnp.power(zbar, 13.0) * rho * q_raw
    return jnp.where(in_range, sreco, 0.0)


def neu_channels_jax(
    t: jax.Array,
    rho: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    log10_tlim: float = 7.5,
) -> jax.Array:
    """Evaluates all 5 specific neutrino loss rates in JAX ``float64``.

    Args:
        t: Temperature $T$ in Kelvin ($\\mathrm{K}$).
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        abar: Mean atomic weight $\\bar{A}$.
        zbar: Mean ionic charge $\\bar{Z}$.
        log10_tlim: Upper bound of cosine cutoff in $\\log_{10}(T/\\mathrm{K})$.

    Returns:
        1D ``float64`` JAX array of shape ``(5,)`` containing specific rates
        ``[pair, plas, phot, brem, reco]`` in $\\mathrm{erg\\,g^{-1}\\,s^{-1}}$.
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    abar = jnp.asarray(abar, dtype=jnp.float64)
    zbar = jnp.asarray(zbar, dtype=jnp.float64)
    active = t > _inputs.TMIN_NEU
    t_s = jnp.where(active, t, _inputs.TMIN_NEU)
    f_cut = _cutoff_factor_jax(t_s, log10_tlim) / rho
    q_vols = jnp.stack(
        [
            _pair_vol_jax(t_s, rho, abar, zbar),
            _plas_vol_jax(t_s, rho, abar, zbar),
            _phot_vol_jax(t_s, rho, abar, zbar),
            _brem_vol_jax(t_s, rho, abar, zbar),
            _reco_vol_jax(t_s, rho, abar, zbar),
        ],
        axis=0,
    )
    return jnp.where(active, f_cut * q_vols, jnp.zeros_like(q_vols))


def neu_get_jax(
    t: jax.Array,
    rho: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    log10_tlim: float = 7.5,
    flags: tuple[bool, bool, bool, bool, bool] = (
        True,
        True,
        True,
        True,
        True,
    ),
) -> tuple[jax.Array, jax.Array]:
    """Evaluates total and per-channel neutrino rates and ``jacfwd`` partials.

    Args:
        t: Temperature $T$ in Kelvin ($\\mathrm{K}$).
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        abar: Mean atomic weight $\\bar{A}$.
        zbar: Mean ionic charge $\\bar{Z}$.
        log10_tlim: Smooth cosine cutoff ceiling in $\\log_{10}(T/\\mathrm{K})$.
        flags: 5-tuple of channel booleans ``(pair, plas, phot, brem, reco)``.

    Returns:
        Tuple ``(loss, sources)`` where ``loss`` has shape ``(5,)``
        ``[val, d_t, d_rho, d_abar, d_zbar]`` and ``sources`` has shape
        ``(5, 5)`` indexed as ``sources[channel, deriv]``.
    """
    t_j = jnp.asarray(t, dtype=jnp.float64)
    rho_j = jnp.asarray(rho, dtype=jnp.float64)
    abar_j = jnp.asarray(abar, dtype=jnp.float64)
    zbar_j = jnp.asarray(zbar, dtype=jnp.float64)

    def _fn(
        t_v: jax.Array, r_v: jax.Array, a_v: jax.Array, z_v: jax.Array
    ) -> jax.Array:
        return neu_channels_jax(t_v, r_v, a_v, z_v, log10_tlim)

    vals = _fn(t_j, rho_j, abar_j, zbar_j)
    dt, drho, dabar, dzbar = jax.jacfwd(_fn, argnums=(0, 1, 2, 3))(
        t_j, rho_j, abar_j, zbar_j
    )
    sources = jnp.stack([vals, dt, drho, dabar, dzbar], axis=1)
    mask = jnp.asarray(flags, dtype=jnp.float64)[:, None]
    sources_masked = jnp.where(t_j <= _inputs.TMIN_NEU, 0.0, sources * mask)
    loss = jnp.sum(sources_masked, axis=0)
    return loss, sources_masked
