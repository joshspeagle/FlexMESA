# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branchless `@jax.jit` (`float64`) Coulomb and mixing kernels for `Skye`.

Implements the one-component plasma (OCP) liquid and solid free energies,
relativistic electron screening corrections, linear mixing corrections, and
Ogata / Potekhin-Chabrier solid anharmonic mixing corrections from
`skye_coulomb_liquid.f90` and `skye_coulomb_solid.f90` (Jermyn et al. 2021,
ApJ, 913, 72).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants
from flexmesa.eos import _pc_coulomb_jax

_ME_IN_AMU: float = constants.ME / constants.AMU


def _ocp_liq_t(
    t_v: jax.Array, rs: jax.Array, z5_3: jax.Array, cotpt: jax.Array
) -> jax.Array:
    """Evaluate classical + quantum OCP liquid free energy at `(t_v, rs)`."""
    g = (z5_3 * constants.QE * constants.QE) / (
        (constants.RBOHR * constants.BOLTZM) * t_v * rs
    )
    tpt = g * cotpt / jnp.sqrt(rs)
    sq32 = jnp.sqrt(3.0) / 2.0
    a1, a2 = -0.9070, 0.62954
    a3 = -sq32 - a1 / jnp.sqrt(a2)
    b1, b2, b3, b4 = 4.56e-3, 211.6, -1.0e-4, 4.62e-3
    sq_g = jnp.sqrt(g)
    fa = a1 * (
        jnp.sqrt(g * (a2 + g))
        - a2 * jnp.log(jnp.sqrt(g / a2) + jnp.sqrt(1.0 + g / a2))
    ) + 2.0 * a3 * (sq_g - jnp.arctan(sq_g))
    fb = b1 * (g - b2 * jnp.log(1.0 + g / b2)) + 0.5 * b3 * jnp.log(
        1.0 + (g * g) / b4
    )
    q1, q2, q4 = 5.994, 70.3, 22.7
    q3 = (0.25 - q1 / q2) * q4
    eta = tpt / jnp.sqrt(3.0)
    fq = (
        q1 * eta
        - (q1 * q2) * jnp.log(1.0 + eta / q2)
        + (0.5 * q3) * jnp.log(1.0 + (eta * eta) / q4)
    )
    return fa + fb + fq


def _ocp_sol_t(
    t_v: jax.Array, rs: jax.Array, z5_3: jax.Array, cotpt: jax.Array
) -> jax.Array:
    """Evaluate harmonic + anharmonic OCP solid free energy at `(t_v, rs)`."""
    g = (z5_3 * constants.QE * constants.QE) / (
        (constants.RBOHR * constants.BOLTZM) * t_v * rs
    )
    tpt = g * cotpt / jnp.sqrt(rs)
    tpt2 = tpt * tpt
    f_ah = (
        -(10.9 / g + 247.0 / (2.0 * g * g) + 1.765e5 / (3.0 * g * g * g))
        * jnp.exp(-(0.12 / 10.9) * tpt2)
        - 0.12 * tpt2 / g
    )
    tpt_m = jnp.clip(tpt, 1.0e-5, 1.0e5)
    f_mid = _pc_coulomb_jax._hlfit_f_mid(tpt_m)
    f_hi = -1.0 / (0.0047770935 * tpt * tpt * tpt)
    f_lo = 3.0 * jnp.log(tpt) - 2.49389 - (1.5 * 0.5113875) * tpt + tpt2 / 24.0
    f_th = jnp.where(tpt > 1.0e5, f_hi, jnp.where(tpt < 1.0e-5, f_lo, f_mid))
    f_harm = (
        -0.895929256 * g
        + f_th
        + (1.5 * 0.5113875) * tpt
        - 3.0 * jnp.log(tpt)
        + 1.5 * jnp.log(g)
        + 1.323515
    )
    return f_harm + f_ah


def _screen_liq(zi: jax.Array, ge: jax.Array, rs: jax.Array) -> jax.Array:
    """Evaluate `ocp_liquid_screening_free_energy_correction`."""
    z2 = zi * zi
    logz = jnp.log(zi)
    a = 1.11 * jnp.power(zi, 0.475)
    b = 0.2 + 0.078 * logz * logz
    nu = 1.16 + 0.08 * logz
    cdh = (zi / jnp.sqrt(3.0)) * (
        jnp.power(1.0 + zi, 1.5) - 1.0 - jnp.power(zi, 1.5)
    )
    ctf = (
        (18.0 / 175.0)
        * jnp.power(12.0 / constants.PI, 2.0 / 3.0)
        * jnp.power(zi, 7.0 / 3.0)
        * (1.0 - jnp.power(zi, -1.0 / 3.0) + 0.2 * jnp.power(zi, -0.5))
    )
    xr = (jnp.power(9.0 * constants.PI / 4.0, 1.0 / 3.0) * constants.FINE) / rs
    xr2 = xr * xr
    gr = jnp.sqrt(1.0 + xr2)
    g1 = 1.0 + 0.78 * jnp.sqrt(ge / zi) / (
        21.0 + ge * (zi / rs) * (zi / rs) * (zi / rs)
    )
    g2 = 1.0 + ((zi - 1.0) / 9.0) * (1.0 + 1.0 / (0.001 * z2 + 2.0 * ge)) * (
        (rs * rs * rs) / (1.0 + 6.0 * rs * rs)
    )
    h = (1.0 + 0.2 * xr2) / (
        1.0
        + (0.18 * jnp.power(zi, -0.25)) * xr
        + (0.37 * jnp.power(zi, -0.5)) * xr2
        + 0.2 * xr2
    )
    sq_ge = jnp.sqrt(ge)
    ge_nu = jnp.power(ge, nu)
    return (
        -ge
        * (cdh * sq_ge + (ctf * a) * ge_nu * g1 * h)
        / (1.0 + (b * sq_ge + a * g2 * ge_nu / rs) / gr)
    )


def _screen_sol(
    zi: jax.Array, mi: jax.Array, ge: jax.Array, rs: jax.Array
) -> jax.Array:
    """Evaluate `ocp_solid_screening_free_energy_correction`."""
    z2 = zi * zi
    z5 = z2 * z2 * zi
    z6 = z5 * zi
    logz = jnp.log(zi)
    z5_3 = jnp.power(zi, 5.0 / 3.0)
    z7_6 = jnp.power(zi, 7.0 / 6.0)
    z2_3 = jnp.power(zi, 2.0 / 3.0)
    s = 1.0 / (1.0 + 1.0e-2 * jnp.power(logz, 1.5) + 0.097 / z2)
    b1 = 1.0 - 1.1866 * jnp.power(zi, -0.267) + 0.27 / zi
    b2 = 1.0 + (2.25 * jnp.power(zi, -1.0 / 3.0)) * (
        1.0 + 0.684 * z5 + 0.222 * z6
    ) / (1.0 + 0.222 * z6)
    b3 = 41.5 / (1.0 + logz)
    b4 = 0.395 * logz + 0.347 * jnp.power(zi, -1.5)
    g = ge * z5_3
    cotpt = jnp.sqrt(3.0 * _ME_IN_AMU / mi) / z7_6
    tpt = g * cotpt / jnp.sqrt(rs)
    tpt_s2 = (0.205 * tpt) * (0.205 * tpt)
    supp = jnp.exp(jnp.clip(-tpt_s2, -100.0, 100.0))
    q = jnp.sqrt(
        (tpt_s2 + jnp.log(1.0 + supp))
        / jnp.log(constants.EULERNUM - (constants.EULERNUM - 2.0) * supp)
    )
    xr = (jnp.power(9.0 * constants.PI / 4.0, 1.0 / 3.0) * constants.FINE) / rs
    xr2 = xr * xr
    a = (b3 + 17.9 * xr2) / (1.0 + b4 * xr2)
    f_inf = (0.00352 * z2_3 * b1) * jnp.sqrt(1.0 + b2 / xr2)
    f_sol = -f_inf * g * (1.0 + a * jnp.power(q / g, s))
    gr = jnp.sqrt(1.0 + xr2)
    alpha = (
        3.0 * jnp.power(4.0 / (9.0 * constants.PI), 2.0 / 3.0) * (rs / ge) * gr
    )
    f_liq = _screen_liq(zi, ge, rs)
    sw = jnp.tanh(jnp.clip(2.0 * alpha, -100.0, 100.0)) ** 3
    return sw * f_liq + (1.0 - sw) * f_sol


def species_nonideal(
    t_v: jax.Array,
    rs: jax.Array,
    game: jax.Array,
    zi: jax.Array,
    mi: jax.Array,
    min_g_sol: jax.Array,
    max_g_liq: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate single-species `(f_liq, f_sol)` non-ideal free energy."""
    z_s = jnp.maximum(jnp.floor(zi), 1.0)
    m_s = jnp.maximum(mi, 1.0e-30)
    z5_3 = jnp.power(z_s, 5.0 / 3.0)
    z7_6 = jnp.power(z_s, 7.0 / 6.0)
    cotpt = jnp.sqrt(3.0 * _ME_IN_AMU / m_s) / z7_6
    gami = (z5_3 * constants.QE * constants.QE) / (
        (constants.RBOHR * constants.BOLTZM) * t_v * rs
    )
    tb_liq = t_v * gami / max_g_liq
    fl_in = _ocp_liq_t(t_v, rs, z5_3, cotpt)
    fl_b, dfl_dlnt = jax.jvp(
        lambda t_b: _ocp_liq_t(t_b, rs, z5_3, cotpt), (tb_liq,), (tb_liq,)
    )
    fl_ext = fl_b + (1.0 - tb_liq / t_v) * dfl_dlnt
    f_liq = jnp.where(gami < max_g_liq, fl_in, fl_ext) + _screen_liq(
        z_s, game, rs
    )

    tb_sol = t_v * gami / min_g_sol
    fs_in = _ocp_sol_t(t_v, rs, z5_3, cotpt)
    fs_b, dfs_dlnt = jax.jvp(
        lambda t_b: _ocp_sol_t(t_b, rs, z5_3, cotpt), (tb_sol,), (tb_sol,)
    )
    fs_ext = fs_b + (1.0 - tb_sol / t_v) * dfs_dlnt
    f_sol = jnp.where(gami > min_g_sol, fs_in, fs_ext) + _screen_sol(
        z_s, m_s, game, rs
    )
    return f_liq, f_sol


def mixing_corrections(
    rs: jax.Array,
    game: jax.Array,
    ya: jax.Array,
    azion: jax.Array,
    mask: jax.Array,
    use_ogata: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return `(zmean, s_mix, fmix_liq, fmix_sol)` for a plasma mixture."""
    y = jnp.where(mask, ya, 0.0)
    z = jnp.where(mask, azion, 0.0)
    zmean = jnp.sum(y * z)
    z2mean = jnp.sum(y * z * z)
    z53 = jnp.sum(y * jnp.power(jnp.maximum(z, 0.0), constants.FIVE_THIRDS))
    z52 = jnp.sum(y * (jnp.sqrt(jnp.maximum(z, 0.0)) ** 5))
    z321 = jnp.sum(y * z * (jnp.sqrt(jnp.maximum(z + 1.0, 0.0)) ** 3))
    s_mix = -jnp.sum(
        jnp.where(y > 0.0, y * jnp.log(jnp.maximum(y, 1e-99)), 0.0)
    )

    d = z2mean / jnp.maximum(zmean * zmean, 1.0e-99)
    active_mix = (zmean > 0.0) & (jnp.abs(d - 1.0) >= 1.0e-9)
    d_safe = jnp.where(active_mix, d, 2.0)
    zm = z2mean + zmean
    dif0 = jnp.where(
        rs < 1.0e-9,
        z52 - jnp.sqrt(z2mean * z2mean * z2mean / jnp.maximum(zmean, 1.0e-99)),
        z321 - jnp.sqrt(zm * zm * zm / jnp.maximum(zmean, 1.0e-99)),
    )
    dif_r = dif0 / jnp.maximum(z52, 1.0e-99)
    dif_fdh = dif0 * game * jnp.sqrt(game / 3.0)
    p3 = jnp.power(d_safe, -0.2)
    d0 = (2.6 * dif_r + 14.0 * (dif_r**3)) / (1.0 - p3)
    gp = d0 * jnp.power(game * z53, p3)
    fmix0 = dif_fdh / (1.0 + gp)
    gq = (d_safe * d_safe * 0.0117) * gp
    fmix_liq = jnp.where(
        active_mix, fmix0 / jnp.power(1.0 + gq, 1.5 / p3 - 1.0), 0.0
    )

    same_z = (z[:, None] == z[None, :]) & mask[:, None] & mask[None, :]
    ca = jnp.sum(jnp.where(same_z, y[None, :], 0.0), axis=1)
    idx = jnp.arange(z.shape[0])
    is_u = mask & (z > 0.0) & (jnp.argmax(same_z, axis=1) == idx)
    zi, zj = z[:, None], z[None, :]
    pair_ok = is_u[:, None] & is_u[None, :] & (zj > zi)
    rz = jnp.where(pair_ok, zj / jnp.maximum(zi, 1.0), 2.0)
    c_sum = jnp.maximum(ca[:, None] + ca[None, :], 1e-40)
    x2_raw = jnp.where(pair_ok, ca[None, :] / c_sum, 0.5)
    x2 = jnp.clip(x2_raw, 1.0e-30, 1.0 - 1.0e-15)
    rm1 = rz - 1.0
    sq = jnp.sqrt(x2)
    dg_og = (
        0.05
        * (rm1 * rm1)
        / ((1.0 + 0.64 * rm1) * (1.0 + 0.5 * rm1 * rm1))
        / (
            1.0
            + (sq * (sq - 0.3) * (sq - 0.7) * (sq - 1.0))
            * 27.0
            * rm1
            / (1.0 + 0.1 * rm1)
        )
    )
    x_pc = x2 / rz + (1.0 - 1.0 / rz) * jnp.power(x2, rz)
    dg_pc = (
        0.012
        * ((x_pc * (1.0 - x_pc)) / (x2 * (1.0 - x2)))
        * (1.0 - 1.0 / (rz * rz))
        * (1.0 - x2 + x2 * jnp.power(rz, 5.0 / 3.0))
    )
    dg = jnp.where(use_ogata, dg_og, dg_pc)
    gami_i = game * jnp.power(jnp.maximum(zi, 1.0), 5.0 / 3.0)
    fmix_sol = jnp.sum(
        jnp.where(pair_ok, gami_i * ca[:, None] * ca[None, :] * dg, 0.0)
    )
    return zmean, s_mix, fmix_liq, fmix_sol
