# SPDX-License-Identifier: LGPL-3.0-or-later
"""Radiation, ideal-ion, and electron-positron `HELM` kernels in JAX.

Ports `helm_radiation.dek`, `helm_ideal_ions.dek`, and
`helm_electron_positron.dek` into `@jax.jit`-compatible `jnp.float64` array
operations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants
from flexmesa.eos import _helm_jax_coulomb, _helm_jax_poly

HELM_SIFAC: float = 8.6322745944370191e-45
ThermoVar = _helm_jax_coulomb.HelmThermoVarJax


def eval_radiation_jax(
    temp: jax.Array, rho: jax.Array
) -> tuple[ThermoVar, ThermoVar, ThermoVar]:
    """Evaluate `HELM` radiation `(prad, erad, srad)` (`helm_radiation.dek`)."""
    tempinv = 1.0 / temp
    deninv = 1.0 / rho
    third = 1.0 / 3.0
    z0 = jnp.zeros_like(temp)

    prad_v = (constants.CRAD * third) * temp * temp * temp * temp
    dpraddt = 4.0 * prad_v * tempinv
    dpraddtt = 4.0 * constants.CRAD * temp * temp
    erad_v = 3.0 * prad_v * deninv
    deraddd = -erad_v * deninv
    deraddt = 3.0 * dpraddt * deninv
    deradddd = -2.0 * deraddd * deninv
    deradddt = -deraddt * deninv
    deraddtt = 3.0 * dpraddtt * deninv
    srad_v = (prad_v * deninv + erad_v) * tempinv
    dsraddd = (-prad_v * deninv * deninv + deraddd) * tempinv
    dsraddt = (dpraddt * deninv + deraddt - srad_v) * tempinv

    prad = ThermoVar(
        prad_v, z0, dpraddt, z0, z0, z0, z0, z0, z0, dpraddtt, z0, z0
    )
    erad = ThermoVar(
        erad_v,
        deraddd,
        deraddt,
        z0,
        z0,
        deradddd,
        deradddt,
        z0,
        z0,
        deraddtt,
        z0,
        z0,
    )
    srad = ThermoVar(
        srad_v,
        dsraddd,
        dsraddt,
        z0,
        z0,
        (2.0 * prad_v * deninv * deninv * deninv + deradddd) * tempinv,
        (-dpraddt * deninv * deninv + deradddt - dsraddd) * tempinv,
        z0,
        z0,
        (dpraddtt * deninv + deraddtt - 2.0 * dsraddt) * tempinv,
        z0,
        z0,
    )
    return prad, erad, srad


def eval_ideal_ions_jax(
    temp: jax.Array, rho: jax.Array, abar: jax.Array
) -> tuple[ThermoVar, ThermoVar, ThermoVar, ThermoVar]:
    """Evaluate `HELM` ideal ions `(pion, eion, sion, xni)`."""
    tempinv = 1.0 / temp
    deninv = 1.0 / rho
    ytot1 = 1.0 / abar
    kt = constants.KERG * temp
    ktinv = 1.0 / kt
    kergavo = constants.KERG * constants.AVO
    z0 = jnp.zeros_like(temp)

    xni_v = constants.AVO * ytot1 * rho
    dxnidd = constants.AVO * ytot1
    dxnida = -xni_v * ytot1
    xni = ThermoVar(
        xni_v, dxnidd, z0, dxnida, z0, z0, z0, -dxnidd * ytot1, z0, z0, z0, z0
    )
    pion_v = jnp.maximum(xni_v * kt, 1.0e-99)
    dpiondd = dxnidd * kt
    dpiondt = xni_v * constants.KERG
    dpionda = -pion_v * ytot1
    dpionddt = dxnidd * constants.KERG
    dpiondda = -dxnidd * ytot1 * kt
    dpiondta = dxnida * constants.KERG
    pion = ThermoVar(
        pion_v,
        dpiondd,
        dpiondt,
        dpionda,
        z0,
        z0,
        dpionddt,
        dpiondda,
        z0,
        z0,
        dpiondta,
        z0,
    )
    eion_v = 1.5 * deninv * pion_v
    deiondt = 1.5 * deninv * dpiondt
    deionda = 1.5 * deninv * dpionda
    eion = ThermoVar(
        eion_v,
        z0,
        deiondt,
        deionda,
        z0,
        z0,
        z0,
        1.5 * deninv * dpiondda - deionda * deninv,
        z0,
        z0,
        1.5 * deninv * dpiondta,
        z0,
    )
    y_s = ktinv * ytot1
    etaion = jnp.log(xni_v * HELM_SIFAC * y_s * jnp.sqrt(y_s))
    detaidd = dxnidd / xni_v
    detaidt = -1.5 * tempinv
    detaida = dxnida / xni_v - 1.5 * ytot1
    y_k = kergavo * ytot1
    z_k = y_k * ytot1
    s_v = (eion_v + pion_v * deninv) * tempinv - etaion * y_k
    s_dd = (dpiondd - pion_v * deninv) * deninv * tempinv - detaidd * y_k
    s_dt = (
        (deiondt + dpiondt * deninv) - (eion_v + pion_v * deninv) * tempinv
    ) * tempinv - detaidt * y_k
    s_da = (deionda + dpionda * deninv) * tempinv + etaion * z_k - detaida * y_k
    s_ddd = (
        2.0 * (pion_v * deninv - dpiondd) * deninv * deninv * tempinv
        + detaidd * detaidd * y_k
    )
    s_ddt = z0
    s_dda = (
        (
            1.5 * deninv * dpiondda
            - deionda * deninv
            + (dpiondda - dpionda * deninv) * deninv
        )
        * tempinv
        + detaidd * z_k
        - (-dxnidd * ytot1 / xni_v - dxnidd * dxnida / (xni_v * xni_v)) * y_k
    )
    s_dtt = (
        -2.0 * (deiondt + dpiondt * deninv) * tempinv * tempinv
        + 2.0 * (eion_v + pion_v * deninv) * tempinv * tempinv * tempinv
        + detaidt * tempinv * y_k
    )
    s_dta = (
        1.5 * deninv * dpiondta
        + dpiondta * deninv
        - (deionda + dpionda * deninv) * tempinv
    ) * tempinv + detaidt * z_k
    neg = s_v < 0.0
    s0 = 1.0e3
    s0_sq = s0 * s0
    denom = jnp.where(neg, s0 + s_v, 1.0)
    d2 = denom * denom
    d3 = d2 * denom
    sion = ThermoVar(
        val=jnp.where(neg, s0 * s_v / denom, s_v),
        dd=jnp.where(neg, s0_sq * s_dd / d2, s_dd),
        dt=jnp.where(neg, s0_sq * s_dt / d2, s_dt),
        da=jnp.where(neg, s0_sq * s_da / d2, s_da),
        dz=z0,
        ddd=jnp.where(
            neg, s0_sq * (s_ddd / d2 - 2.0 * s_dd * s_dd / d3), s_ddd
        ),
        ddt=jnp.where(
            neg, s0_sq * (s_ddt / d2 - 2.0 * s_dd * s_dt / d3), s_ddt
        ),
        dda=jnp.where(
            neg, s0_sq * (s_dda / d2 - 2.0 * s_dd * s_da / d3), s_dda
        ),
        ddz=z0,
        dtt=jnp.where(
            neg, s0_sq * (s_dtt / d2 - 2.0 * s_dt * s_dt / d3), s_dtt
        ),
        dta=jnp.where(
            neg, s0_sq * (s_dta / d2 - 2.0 * s_dt * s_da / d3), s_dta
        ),
        dtz=z0,
    )
    return pion, eion, sion, xni


def eval_elec_pos_jax(
    temp: jax.Array,
    rho: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    t_lo: jax.Array,
    dth: jax.Array,
    d_lo: jax.Array,
    dd: jax.Array,
    fi_f: jax.Array,
    fi_dpdf: jax.Array,
    fi_ef: jax.Array,
    fi_xf: jax.Array,
) -> tuple[
    ThermoVar,
    ThermoVar,
    ThermoVar,
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
]:
    """Evaluate `HELM` electron-positron `(pele, eele, sele, eta_t, xne_t)`."""
    ytot1 = 1.0 / abar
    ye = ytot1 * zbar
    din = ye * rho
    dt2 = dth * dth
    dti = 1.0 / dth
    dt2i = 1.0 / dt2
    dt3i = dt2i * dti
    dd2 = dd * dd
    ddi = 1.0 / dd
    dd2i = 1.0 / dd2
    dd3i = dd2i * ddi
    xt = jnp.maximum((temp - t_lo) * dti, 0.0)
    xd = jnp.maximum((din - d_lo) * ddi, 0.0)
    qt0, qt1, qt2, qt3 = _helm_jax_poly.quintic_basis_1d(
        xt, dth, dt2, dti, dt2i, dt3i
    )
    qd0, qd1, qd2, qd3 = _helm_jax_poly.quintic_basis_1d(
        xd, dd, dd2, ddi, dd2i, dd3i
    )
    ct0, ct1, _ = _helm_jax_poly.cubic_basis_1d(xt, dth, dti, dt2i)
    cd0, cd1, _ = _helm_jax_poly.cubic_basis_1d(xd, dd, ddi, dd2i)

    free = _helm_jax_poly.h5(fi_f, qt0, qd0)
    df_d = _helm_jax_poly.h5(fi_f, qt0, qd1)
    df_t = _helm_jax_poly.h5(fi_f, qt1, qd0)
    df_dd = _helm_jax_poly.h5(fi_f, qt0, qd2)
    df_tt = _helm_jax_poly.h5(fi_f, qt2, qd0)
    df_dt = _helm_jax_poly.h5(fi_f, qt1, qd1)
    df_ddd = _helm_jax_poly.h5(fi_f, qt0, qd3)
    df_ddt = _helm_jax_poly.h5(fi_f, qt1, qd2)
    df_dtt = _helm_jax_poly.h5(fi_f, qt2, qd1)
    df_ttt = _helm_jax_poly.h5(fi_f, qt3, qd0)

    dpepdd_in = jnp.maximum(_helm_jax_poly.h3(fi_dpdf, ct0, cd0), 1.0e-30)
    dpepddd_in = _helm_jax_poly.h3(fi_dpdf, ct0, cd1)
    dpepddt_in = _helm_jax_poly.h3(fi_dpdf, ct1, cd0)
    etaele = _helm_jax_poly.h3(fi_ef, ct0, cd0)
    x_ef = _helm_jax_poly.h3(fi_ef, ct0, cd1)
    eta_t = (
        etaele,
        ye * x_ef,
        _helm_jax_poly.h3(fi_ef, ct1, cd0),
        -x_ef * din * ytot1,
        x_ef * rho * ytot1,
    )
    xnefer = _helm_jax_poly.h3(fi_xf, ct0, cd0)
    x_xf = jnp.maximum(_helm_jax_poly.h3(fi_xf, ct0, cd1), 1.0e-30)
    xne_t = (
        xnefer,
        ye * x_xf,
        _helm_jax_poly.h3(fi_xf, ct1, cd0),
        -x_xf * din * ytot1,
        x_xf * rho * ytot1,
    )

    dindd = ye
    dinda = -din * ytot1
    dindz = rho * ytot1
    dindda = -ye * ytot1
    dinddz = ytot1
    din2 = din * din
    pele_v = din2 * df_d
    use_h5_dp = (
        pele_v / (din * constants.KERG * temp / (abar * constants.MP)) < 1.0e6
    )
    ww_p = jnp.where(use_h5_dp, din2 * df_dd + 2.0 * din * df_d, dpepdd_in)
    zz_p = jnp.where(
        use_h5_dp, din2 * df_ddd + 4.0 * din * df_dd + 2.0 * df_d, dpepddd_in
    )
    y_p = jnp.where(use_h5_dp, din2 * df_ddt + 2.0 * din * df_dt, dpepddt_in)
    pele = ThermoVar(
        val=pele_v,
        dd=ww_p * dindd,
        dt=din2 * df_dt,
        da=ww_p * dinda,
        dz=ww_p * dindz,
        ddd=zz_p * dindd * dindd,
        ddt=y_p * dindd,
        dda=(zz_p * dindd + ww_p) * dindda,
        ddz=(zz_p * dindd + ww_p) * dinddz,
        dtt=din2 * df_dtt,
        dta=y_p * dinda,
        dtz=y_p * dindz,
    )
    ye2 = ye * ye
    y_ye = ye * ytot1
    sele = ThermoVar(
        val=-df_t * ye,
        dd=-df_dt * ye2,
        dt=-df_tt * ye,
        da=-df_dt * dinda * ye + df_t * y_ye,
        dz=-df_dt * dindz * ye - df_t * ytot1,
        ddd=-df_ddt * ye2 * ye,
        ddt=-df_dtt * ye2,
        dda=-df_ddt * dinda * ye2 + 2.0 * df_dt * ye2 * ytot1,
        ddz=-df_ddt * dindz * ye2 - 2.0 * df_dt * y_ye,
        dtt=-df_ttt * ye,
        dta=-df_dtt * dinda * ye + df_tt * y_ye,
        dtz=-df_dtt * dindz * ye - df_tt * ytot1,
    )
    eele = ThermoVar(
        val=ye * free + temp * sele.val,
        dd=ye * df_d * dindd + temp * sele.dd,
        dt=temp * sele.dt,
        da=-y_ye * free + ye * df_d * dinda + temp * sele.da,
        dz=ytot1 * free + ye * df_d * dindz + temp * sele.dz,
        ddd=ye * df_dd * dindd * dindd + temp * sele.ddd,
        ddt=ye * df_dt * dindd + sele.dd + temp * sele.ddt,
        dda=(
            -y_ye * df_d * dindd
            + ye * df_dd * dinda * dindd
            + ye * df_d * dindda
            + temp * sele.dda
        ),
        ddz=(
            ytot1 * df_d * dindd
            + ye * df_dd * dindz * dindd
            + ye * df_d * dinddz
            + temp * sele.ddz
        ),
        dtt=sele.dt + temp * sele.dtt,
        dta=temp * sele.dta,
        dtz=temp * sele.dtz,
    )
    return pele, eele, sele, eta_t, xne_t
