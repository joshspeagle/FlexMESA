# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branchless JAX Coulomb corrections (`helm_coulomb2.dek`) for `HELM` EOS.

Ports the Yakovlev & Shalybkov (1989) strong-coupling (`plasg >= 1`) and
weak-coupling (`plasg < 1`) uniform-background Coulomb free-energy fits and
low-temperature/low-density cutoff from `eos/private/helm_coulomb2.dek` into
`@jax.jit`-compatible `jnp.float64` array operations.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from flexmesa import constants

A1: float = -0.898004
B1: float = 0.96786
C1: float = 0.220703
D1: float = -0.86097
E1: float = 2.5269
A2: float = 0.29561
B2: float = 1.9885
C2: float = 0.288675
DEFAULT_COULOMB_TEMP_CUT: float = -1.0e99
DEFAULT_COULOMB_DEN_CUT: float = -1.0e99


class HelmThermoVarJax(NamedTuple):
    """Thermodynamic quantity and partials in `(rho, T, abar, zbar)`."""

    val: jax.Array
    dd: jax.Array
    dt: jax.Array
    da: jax.Array
    dz: jax.Array
    ddd: jax.Array
    ddt: jax.Array
    dda: jax.Array
    ddz: jax.Array
    dtt: jax.Array
    dta: jax.Array
    dtz: jax.Array

    def add(self, other: HelmThermoVarJax) -> HelmThermoVarJax:
        """Return the component-wise sum `self + other`."""
        return HelmThermoVarJax(
            val=self.val + other.val,
            dd=self.dd + other.dd,
            dt=self.dt + other.dt,
            da=self.da + other.da,
            dz=self.dz + other.dz,
            ddd=self.ddd + other.ddd,
            ddt=self.ddt + other.ddt,
            dda=self.dda + other.dda,
            ddz=self.ddz + other.ddz,
            dtt=self.dtt + other.dtt,
            dta=self.dta + other.dta,
            dtz=self.dtz + other.dtz,
        )


def _select_thermo_var(
    cond: jax.Array, strong: HelmThermoVarJax, weak: HelmThermoVarJax
) -> HelmThermoVarJax:
    """Select elementwise between `strong` and `weak` `HelmThermoVarJax`."""
    return HelmThermoVarJax(
        val=jnp.where(cond, strong.val, weak.val),
        dd=jnp.where(cond, strong.dd, weak.dd),
        dt=jnp.where(cond, strong.dt, weak.dt),
        da=jnp.where(cond, strong.da, weak.da),
        dz=jnp.where(cond, strong.dz, weak.dz),
        ddd=jnp.where(cond, strong.ddd, weak.ddd),
        ddt=jnp.where(cond, strong.ddt, weak.ddt),
        dda=jnp.where(cond, strong.dda, weak.dda),
        ddz=jnp.where(cond, strong.ddz, weak.ddz),
        dtt=jnp.where(cond, strong.dtt, weak.dtt),
        dta=jnp.where(cond, strong.dta, weak.dta),
        dtz=jnp.where(cond, strong.dtz, weak.dtz),
    )


def _eval_strong_coulomb(
    rho: jax.Array,
    deninv: jax.Array,
    pion: HelmThermoVarJax,
    plasg: HelmThermoVarJax,
    plasginv: jax.Array,
) -> tuple[HelmThermoVarJax, HelmThermoVarJax, jax.Array, jax.Array, jax.Array]:
    """Evaluate strong-coupling (`plasg >= 1`) Coulomb terms."""
    third = 1.0 / 3.0
    p1 = jnp.sqrt(jnp.sqrt(plasg.val))
    p2 = 1.0 / p1
    p3 = p1 * plasginv
    p4 = p2 * plasginv
    p5 = p3 * plasginv
    p6 = p4 * plasginv
    u0 = A1 * plasg.val + B1 * p1 + C1 * p2 + D1
    du0 = A1 + 0.25 * B1 * p3 - 0.25 * C1 * p4
    ddu0 = -0.1875 * B1 * p5 + 0.3125 * C1 * p6
    z_ec = pion.val * deninv
    ec_val = z_ec * u0
    x_ec = deninv * u0
    y_ec = deninv * du0
    ww_ec = z_ec * du0
    dfk = z_ec * ddu0
    ec = HelmThermoVarJax(
        val=ec_val,
        dd=pion.dd * x_ec - z_ec * x_ec + ww_ec * plasg.dd,
        dt=pion.dt * x_ec + ww_ec * plasg.dt,
        da=pion.da * x_ec + ww_ec * plasg.da,
        dz=ww_ec * plasg.dz,
        ddd=2.0 * pion.dd * (y_ec * plasg.dd - deninv * x_ec)
        + z_ec
        * (
            2.0 * deninv * x_ec
            + (ddu0 * plasg.dd - 2.0 * y_ec) * plasg.dd
            + du0 * plasg.ddd
        ),
        ddt=pion.ddt * x_ec
        + pion.dd * y_ec * plasg.dt
        - pion.dt * deninv * x_ec
        - z_ec * y_ec * plasg.dt
        + (pion.dt * y_ec + dfk * plasg.dt) * plasg.dd
        + ww_ec * plasg.ddt,
        dda=pion.dda * x_ec
        + pion.dd * y_ec * plasg.da
        - pion.da * deninv * x_ec
        - z_ec * y_ec * plasg.da
        + (pion.da * y_ec + dfk * plasg.da) * plasg.dd
        + ww_ec * plasg.dda,
        ddz=pion.dd * y_ec * plasg.dz
        - z_ec * y_ec * plasg.dz
        + dfk * plasg.dz * plasg.dd
        + ww_ec * plasg.ddz,
        dtt=(2.0 * pion.dt * y_ec + dfk * plasg.dt) * plasg.dt
        + ww_ec * plasg.dtt,
        dta=pion.dta * x_ec
        + pion.dt * y_ec * plasg.da
        + (pion.da * y_ec + dfk * plasg.da) * plasg.dt
        + ww_ec * plasg.dta,
        dtz=pion.dt * y_ec * plasg.dz
        + dfk * plasg.dz * plasg.dt
        + ww_ec * plasg.dtz,
    )
    y_pc = third * rho
    pc = HelmThermoVarJax(
        val=y_pc * ec.val,
        dd=third * ec.val + y_pc * ec.dd,
        dt=y_pc * ec.dt,
        da=y_pc * ec.da,
        dz=y_pc * ec.dz,
        ddd=2.0 * third * ec.dd + y_pc * ec.ddd,
        ddt=third * ec.dt + y_pc * ec.ddt,
        dda=third * ec.da + y_pc * ec.dda,
        ddz=third * ec.dz + y_pc * ec.ddz,
        dtt=y_pc * ec.dtt,
        dta=y_pc * ec.dta,
        dtz=y_pc * ec.dtz,
    )
    u0_s = 3.0 * B1 * p1 - 5.0 * C1 * p2 + D1 * (jnp.log(plasg.val) - 1.0) - E1
    du0_s = 0.75 * B1 * p3 + 1.25 * C1 * p4 + D1 * plasginv
    ddu0_s = -0.5625 * B1 * p5 - 1.5625 * C1 * p6 - D1 * plasginv * plasginv
    return pc, ec, u0_s, du0_s, ddu0_s


def _eval_weak_coulomb(
    deninv: jax.Array,
    pion: HelmThermoVarJax,
    plasg: HelmThermoVarJax,
    plasginv: jax.Array,
) -> tuple[HelmThermoVarJax, HelmThermoVarJax, jax.Array, jax.Array, jax.Array]:
    """Evaluate weak-coupling (`plasg < 1`) Coulomb terms."""
    third = 1.0 / 3.0
    x_wk = jnp.sqrt(plasg.val)
    p1 = plasg.val * x_wk
    p2 = jnp.power(plasg.val, B2)
    p3 = x_wk
    p4 = p2 * plasginv
    p5 = p3 * plasginv
    p6 = p4 * plasginv
    u0 = C2 * p1 - third * A2 * p2
    du0 = 1.5 * C2 * p3 - third * A2 * B2 * p4
    ddu0 = 0.75 * C2 * p5 - third * A2 * B2 * (B2 - 1.0) * p6
    pc_val = -pion.val * u0
    x_pw = pion.val * du0
    y_pw = pion.val * ddu0
    pc = HelmThermoVarJax(
        val=pc_val,
        dd=-pion.dd * u0 - x_pw * plasg.dd,
        dt=-pion.dt * u0 - x_pw * plasg.dt,
        da=-pion.da * u0 - x_pw * plasg.da,
        dz=-x_pw * plasg.dz,
        ddd=-(2.0 * pion.dd * du0 + y_pw * plasg.dd) * plasg.dd
        - x_pw * plasg.ddd,
        ddt=-pion.ddt * u0
        - pion.dd * du0 * plasg.dt
        - (pion.dt * du0 + y_pw * plasg.dt) * plasg.dd
        - x_pw * plasg.ddt,
        dda=-pion.dda * u0
        - pion.dd * du0 * plasg.da
        - (pion.da * du0 + y_pw * plasg.da) * plasg.dd
        - x_pw * plasg.dda,
        ddz=-pion.dd * du0 * plasg.dz
        - y_pw * plasg.dz * plasg.dd
        - x_pw * plasg.ddz,
        dtt=-(2.0 * pion.dt * du0 + y_pw * plasg.dt) * plasg.dt
        - x_pw * plasg.dtt,
        dta=-pion.dta * u0
        - pion.dt * du0 * plasg.da
        - (pion.da * du0 + y_pw * plasg.da) * plasg.dt
        - x_pw * plasg.dta,
        dtz=-pion.dt * du0 * plasg.dz
        - y_pw * plasg.dz * plasg.dt
        - x_pw * plasg.dtz,
    )
    z_ew = 3.0 * deninv
    x_ew = deninv * deninv
    ec_val = z_ew * pc.val
    ec = HelmThermoVarJax(
        val=ec_val,
        dd=z_ew * pc.dd - ec_val * deninv,
        dt=z_ew * pc.dt,
        da=z_ew * pc.da,
        dz=z_ew * pc.dz,
        ddd=z_ew * pc.ddd
        - z_ew * deninv * pc.dd
        + (2.0 * ec_val - 3.0 * pc.dd) * x_ew,
        ddt=z_ew * pc.ddt - 3.0 * pc.dt * x_ew,
        dda=z_ew * pc.dda - 3.0 * pc.da * x_ew,
        ddz=z_ew * pc.ddz - 3.0 * pc.dz * x_ew,
        dtt=z_ew * pc.dtt,
        dta=z_ew * pc.dta,
        dtz=z_ew * pc.dtz,
    )
    u0_s = C2 * p1 - (A2 / B2) * (B2 - 1.0) * p2
    du0_s = 1.5 * C2 * p3 - A2 * (B2 - 1.0) * p4
    ddu0_s = 0.75 * C2 * p5 - A2 * (B2 - 1.0) * (B2 - 1.0) * p6
    return pc, ec, u0_s, du0_s, ddu0_s


def eval_helm_coulomb_jax(
    temp: jax.Array,
    rho: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xni: HelmThermoVarJax,
    pion: HelmThermoVarJax,
    coulomb_temp_cut: float = DEFAULT_COULOMB_TEMP_CUT,
    coulomb_den_cut: float = DEFAULT_COULOMB_DEN_CUT,
) -> tuple[HelmThermoVarJax, HelmThermoVarJax, HelmThermoVarJax]:
    """Evaluate `HELM` Coulomb `(pcoul, ecoul, scoul)` (`helm_coulomb2.dek`)."""
    third = 1.0 / 3.0
    forth = 4.0 / 3.0
    fiveth = 5.0 / 3.0
    forthpi = forth * constants.PI
    tempinv = 1.0 / temp
    deninv = 1.0 / rho
    ytot1 = 1.0 / abar
    ktinv = 1.0 / (constants.KERG * temp)

    s_c = forthpi * zbar * xni.val
    sinv = 1.0 / s_c
    dsdd = forthpi * zbar * xni.dd
    dsda = forthpi * zbar * xni.da
    dsdz = forthpi * xni.val
    dsdda = forthpi * zbar * xni.dda
    dsddz = forthpi * xni.dd
    aele = jnp.power(sinv, third)
    aeleinv = 1.0 / aele
    z_a = -third * aele * sinv
    y_a = -forth * z_a * sinv
    daeledd = z_a * dsdd
    daeleda = z_a * dsda
    daeledz = z_a * dsdz
    daeleddd = y_a * dsdd * dsdd
    daeledda = y_a * dsda * dsdd + z_a * dsdda
    daeleddz = y_a * dsdz * dsdd + z_a * dsddz

    esqu = constants.QE * constants.QE
    eplasg = esqu * ktinv * aeleinv
    z_ep = -eplasg * aeleinv
    y_ep = -2.0 * z_ep * aeleinv
    deplasgdd = z_ep * daeledd
    deplasgdt = -eplasg * tempinv
    deplasgda = z_ep * daeleda
    deplasgdz = z_ep * daeledz
    deplasgddd = y_ep * daeledd * daeledd + z_ep * daeleddd
    deplasgddt = -deplasgdd * tempinv
    deplasgdda = y_ep * daeleda * daeledd + z_ep * daeledda
    deplasgddz = y_ep * daeledz * daeledd + z_ep * daeleddz
    deplasgdtt = 2.0 * eplasg * tempinv * tempinv
    deplasgdta = -deplasgda * tempinv
    deplasgdtz = -deplasgdz * tempinv

    x_z = jnp.power(zbar, third)
    z_z = x_z * x_z * x_z * x_z * x_z
    ww_z = fiveth * x_z * x_z
    plasg_val = z_z * eplasg
    plasginv = 1.0 / plasg_val
    plasg = HelmThermoVarJax(
        val=plasg_val,
        dd=z_z * deplasgdd,
        dt=z_z * deplasgdt,
        da=z_z * deplasgda,
        dz=z_z * deplasgdz + ww_z * eplasg,
        ddd=z_z * deplasgddd,
        ddt=z_z * deplasgddt,
        dda=z_z * deplasgdda,
        ddz=z_z * deplasgddz + ww_z * deplasgdd,
        dtt=z_z * deplasgdtt,
        dta=z_z * deplasgdta,
        dtz=z_z * deplasgdtz + ww_z * deplasgdt,
    )

    pc_st, ec_st, u0_ss, du0_ss, ddu0_ss = _eval_strong_coulomb(
        rho, deninv, pion, plasg, plasginv
    )
    pc_wk, ec_wk, u0_sw, du0_sw, ddu0_sw = _eval_weak_coulomb(
        deninv, pion, plasg, plasginv
    )
    is_st = plasg.val >= 1.0
    pcoul = _select_thermo_var(is_st, pc_st, pc_wk)
    ecoul = _select_thermo_var(is_st, ec_st, ec_wk)

    u0_s = jnp.where(is_st, u0_ss, u0_sw)
    du0_s = jnp.where(is_st, du0_ss, du0_sw)
    ddu0_s = jnp.where(is_st, ddu0_ss, ddu0_sw)
    z_sc = -constants.AVO * ytot1 * constants.KERG
    scoul_val = z_sc * u0_s
    ww_sc = z_sc * du0_s
    x_sc = z_sc * ddu0_s
    lead_sc = jnp.where(is_st, x_sc, ww_sc)
    scoul = HelmThermoVarJax(
        val=scoul_val,
        dd=ww_sc * plasg.dd,
        dt=ww_sc * plasg.dt,
        da=ww_sc * plasg.da - scoul_val * ytot1,
        dz=ww_sc * plasg.dz,
        ddd=x_sc * plasg.dd * plasg.dd + ww_sc * plasg.ddd,
        ddt=x_sc * plasg.dt * plasg.dd + ww_sc * plasg.ddt,
        dda=x_sc * plasg.da * plasg.dd
        + ww_sc * plasg.dda
        - lead_sc * ytot1 * plasg.dd,
        ddz=x_sc * plasg.dz * plasg.dd + ww_sc * plasg.ddz,
        dtt=x_sc * plasg.dt * plasg.dt + ww_sc * plasg.dtt,
        dta=x_sc * plasg.da * plasg.dt
        + ww_sc * plasg.dta
        - lead_sc * ytot1 * plasg.dt,
        dtz=x_sc * plasg.dz * plasg.dt + ww_sc * plasg.dtz,
    )

    in_cut = (temp <= coulomb_temp_cut) & (rho <= coulomb_den_cut)
    deep_cut = temp <= 0.1 * coulomb_temp_cut
    active_cut = in_cut & (~deep_cut)
    ratio = jnp.where(active_cut, coulomb_temp_cut / temp, 1.0)
    cf_exp = jnp.exp(1.0 - ratio * ratio)
    cf = jnp.where(in_cut, jnp.where(deep_cut, 0.0, cf_exp), 1.0)
    dcfdt = jnp.where(active_cut, cf_exp * 2.0 * ratio * ratio * tempinv, 0.0)

    def _apply_cut(v: HelmThermoVarJax) -> HelmThermoVarJax:
        return v._replace(
            val=v.val * cf,
            dd=v.dd * cf,
            dt=cf * v.dt + dcfdt * v.val,
            da=v.da * cf,
            dz=v.dz * cf,
        )

    return _apply_cut(pcoul), _apply_cut(ecoul), _apply_cut(scoul)
