# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branchless JAX `float64` kernels for the `HELM` equation of state.

Ports `helm_sum_totals.dek`, `helm_gammas.dek`, and `do_convert_helm_results`
(`eos_helm_eval.f90`) into `@jax.jit`-compiled `jnp.float64` array operations.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from flexmesa import constants
from flexmesa.eos import _helm_jax_components, _helm_jax_coulomb


def _pack_helm_26(
    active_16: list[jax.Array], z0: jax.Array, frac_helm: jax.Array
) -> jax.Array:
    """Pack the 16 active `HELM` channels and 10 trailing channels (`26,`)."""
    return jnp.stack(
        [*active_16, z0, z0, z0, frac_helm, z0, z0, z0, z0, z0, z0], axis=-1
    )


def _eval_helm_cell_core(
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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate the 26-channel `HELM` EOS and derivatives on a known cell."""
    tempinv = 1.0 / temp
    deninv = 1.0 / rho
    z0 = jnp.zeros_like(temp)

    prad, erad, srad = _helm_jax_components.eval_radiation_jax(temp, rho)
    pion, eion, sion, xni = _helm_jax_components.eval_ideal_ions_jax(
        temp, rho, abar
    )
    pele, eele, sele, eta_t, xne_t = _helm_jax_components.eval_elec_pos_jax(
        temp,
        rho,
        abar,
        zbar,
        t_lo,
        dth,
        d_lo,
        dd,
        fi_f,
        fi_dpdf,
        fi_ef,
        fi_xf,
    )
    etaele, detadd, detadt, detada, detadz = eta_t
    xnefer, dxnedd, dxnedt, dxneda, dxnedz = xne_t

    pcoul, ecoul, scoul = _helm_jax_coulomb.eval_helm_coulomb_jax(
        temp, rho, abar, zbar, xni, pion
    )
    pgas = pion.add(pele).add(pcoul)
    egas = eion.add(eele).add(ecoul)
    sgas = sion.add(sele).add(scoul)
    pres = prad.add(pgas)
    ener = erad.add(egas)
    entr = srad.add(sgas)

    presinv = 1.0 / pres.val
    ww = temp * presinv
    yy = -ww * presinv * pres.dt
    chit = ww * pres.dt
    dchitdd = yy * pres.dd + ww * pres.ddt
    dchitdt = yy * pres.dt + ww * pres.dtt + pres.dt * presinv
    dchitda = yy * pres.da + ww * pres.dta
    dchitdz = yy * pres.dz + ww * pres.dtz

    ww = rho * presinv
    yy = -ww * presinv * pres.dd
    chid = ww * pres.dd
    dchiddd = yy * pres.dd + ww * pres.ddd + pres.dd * presinv
    dchiddt = yy * pres.dt + ww * pres.ddt
    dchidda = yy * pres.da + ww * pres.dda
    dchiddz = yy * pres.dz + ww * pres.ddz

    cv = ener.dt
    dcvdd = ener.ddt
    dcvdt = ener.dtt
    dcvda = ener.dta
    dcvdz = ener.dtz

    ww = pres.val * deninv
    yy = -ww * deninv
    zz = 1.0 / cv
    z_g = tempinv * zz
    xx = chit * z_g
    x_g = ww * xx
    dfk = xx * zz
    gam3 = x_g + 1.0
    dgam3dd = (pres.dd * deninv + yy) * xx + (dchitdd * z_g - dfk * dcvdd) * ww
    dgam3dt = (pres.dt * deninv - ww * tempinv) * xx + (
        dchitdt * z_g - dfk * dcvdt
    ) * ww
    dgam3da = pres.da * deninv * xx + (dchitda * z_g - dfk * dcvda) * ww
    dgam3dz = pres.dz * deninv * xx + (dchitdz * z_g - dfk * dcvdz) * ww

    gam1 = chit * x_g + chid
    dgam1dd = dchitdd * x_g + chit * dgam3dd + dchiddd
    dgam1dt = dchitdt * x_g + chit * dgam3dt + dchiddt
    dgam1da = dchitda * x_g + chit * dgam3da + dchidda
    dgam1dz = dchitdz * x_g + chit * dgam3dz + dchiddz

    z_n = 1.0 / gam1
    nabad = x_g * z_n
    dnabdd = (dgam3dd - nabad * dgam1dd) * z_n
    dnabdt = (dgam3dt - nabad * dgam1dt) * z_n
    dnabda = (dgam3da - nabad * dgam1da) * z_n
    dnabdz = (dgam3dz - nabad * dgam1dz) * z_n

    z_c = 1.0 / chid
    ww_c = gam1 * z_c
    yy_c = -ww_c * z_c * cv
    zz_c = cv * z_c
    cp = cv * ww_c
    dcpdd = dcvdd * ww_c + zz_c * dgam1dd + yy_c * dchiddd
    dcpdt = dcvdt * ww_c + zz_c * dgam1dt + yy_c * dchiddt
    dcpda = dcvda * ww_c + zz_c * dgam1da + yy_c * dchidda
    dcpdz = dcvdz * ww_c + zz_c * dgam1dz + yy_c * dchiddz

    mu = abar / (1.0 + zbar)
    free_e = jnp.maximum(xnefer, 1.0e-99) / (constants.AVO * rho)
    one = jnp.ones_like(temp)

    res = _pack_helm_26(
        [
            jnp.log(pgas.val),
            jnp.log(ener.val),
            jnp.log(entr.val),
            mu,
            jnp.log(free_e),
            etaele,
            nabad,
            chid,
            chit,
            cp,
            cv,
            ener.dd,
            entr.dt,
            entr.dd,
            gam1,
            gam3,
        ],
        z0,
        one,
    )
    d_dlnd = _pack_helm_26(
        [
            pgas.dd * rho / pgas.val,
            ener.dd * rho / ener.val,
            entr.dd * rho / entr.val,
            z0,
            dxnedd / (constants.AVO * free_e) - 1.0,
            detadd * rho,
            dnabdd * rho,
            dchiddd * rho,
            dchitdd * rho,
            dcpdd * rho,
            dcvdd * rho,
            ener.ddd * rho,
            entr.ddt * rho,
            entr.ddd * rho,
            dgam1dd * rho,
            dgam3dd * rho,
        ],
        z0,
        z0,
    )
    d_dln_t = _pack_helm_26(
        [
            pgas.dt * temp / pgas.val,
            ener.dt * temp / ener.val,
            entr.dt * temp / entr.val,
            z0,
            (dxnedt * temp / (constants.AVO * rho)) / free_e,
            detadt * temp,
            dnabdt * temp,
            dchiddt * temp,
            dchitdt * temp,
            dcpdt * temp,
            dcvdt * temp,
            ener.ddt * temp,
            entr.dtt * temp,
            entr.ddt * temp,
            dgam1dt * temp,
            dgam3dt * temp,
        ],
        z0,
        z0,
    )
    d_dabar = _pack_helm_26(
        [
            pgas.da / pgas.val,
            ener.da / ener.val,
            entr.da / entr.val,
            z0,
            (dxneda / (constants.AVO * rho)) / free_e,
            detada,
            dnabda,
            dchidda,
            dchitda,
            dcpda,
            dcvda,
            ener.dda,
            entr.dta,
            entr.dda,
            dgam1da,
            dgam3da,
        ],
        z0,
        z0,
    )
    d_dzbar = _pack_helm_26(
        [
            pgas.dz / pgas.val,
            ener.dz / ener.val,
            entr.dz / entr.val,
            z0,
            (dxnedz / (constants.AVO * rho)) / free_e,
            detadz,
            dnabdz,
            dchiddz,
            dchitdz,
            dcpdz,
            dcvdz,
            ener.ddz,
            entr.dtz,
            entr.ddz,
            dgam1dz,
            dgam3dz,
        ],
        z0,
        z0,
    )
    valid = (
        (abar > 0.0)
        & (zbar > 0.0)
        & (sele.dt >= 0.0)
        & (pgas.val >= 1.0e-20)
        & (egas.val >= 1.0e-20)
        & (sgas.val >= 1.0e-20)
        & (cv >= 0.0)
        & (gam1 >= 0.0)
    )[..., None]
    nan = jnp.full_like(res, jnp.nan)
    return (
        jnp.where(valid, res, nan),
        jnp.where(valid, d_dlnd, nan),
        jnp.where(valid, d_dln_t, nan),
        jnp.where(valid, d_dabar, nan),
        jnp.where(valid, d_dzbar, nan),
    )


_eval_helm_cell_jit = jax.jit(_eval_helm_cell_core)


def eval_helm_cell_jax(
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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate the 26-channel `HELM` EOS and derivatives on a known cell.

    Args:
        temp: Temperature in K (`(*batch,)`).
        rho: Mass density in g cm^-3 (`(*batch,)`).
        abar: Mean atomic weight (`(*batch,)`).
        zbar: Mean atomic charge (`(*batch,)`).
        t_lo: Lower cell temperature bound in K (`(*batch,)`).
        dth: Cell temperature width in K (`(*batch,)`).
        d_lo: Lower cell `Y_e * rho` bound in mol cm^-3 (`(*batch,)`).
        dd: Cell `Y_e * rho` width in mol cm^-3 (`(*batch,)`).
        fi_f: 36-element Helmholtz free-energy stencil (`(*batch, 36)`).
        fi_dpdf: 16-element pressure-derivative stencil (`(*batch, 16)`).
        fi_ef: 16-element electron degeneracy stencil (`(*batch, 16)`).
        fi_xf: 16-element electron number-density stencil (`(*batch, 16)`).

    Returns:
        Tuple `(res, d_dlnd, d_dln_t, d_dabar, d_dzbar)`, each of shape
        `(*batch, 26)`.
    """
    out = _eval_helm_cell_jit(
        temp, rho, abar, zbar, t_lo, dth, d_lo, dd, fi_f, fi_dpdf, fi_ef, fi_xf
    )
    return cast(
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], out
    )
