# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native `@jax.jit` kernels and `custom_jvp` rules for the `PC` EOS.

Provides `eval_pc_eos_jax` in pure vectorized JAX (`jnp.float64`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants
from flexmesa.eos import (
    _jet_jax,
    _pc_coulomb_jax,
    _pc_elect_jax,
    _pc_screen_jax,
    core,
)

excor7_jax = _pc_coulomb_jax.excor7_jax
fharm12_anharm8_jax = _pc_coulomb_jax.fharm12_anharm8_jax
fition9_jax = _pc_coulomb_jax.fition9_jax

_UN_T6: float = 0.3157746
_TINY: float = 1.0e-7
_AUM: float = 1822.888
_RSIMELT: float = 140.0
_RAD: float = 2.5568570411948021e-07


def _eval_pc_res_single(
    rho: jax.Array,
    t: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xa: jax.Array,
    azion: jax.Array,
    acmi: jax.Array,
    include_radiation: jax.Array | bool,
    g_start: jax.Array | float,
    g_full: jax.Array | float,
    mf_lim: jax.Array | float,
    n_real: jax.Array,
) -> jax.Array:
    """Evaluate the 26-channel `PC` `res` vector for a single zone in JAX."""
    ay_raw = jnp.where(
        (xa >= mf_lim) & (azion >= 0.0),
        xa / jnp.maximum(acmi, 1.0e-99),
        0.0,
    )
    y_sum = jnp.sum(ay_raw)
    ay = jnp.where(
        jnp.abs(y_sum - 1.0) > _TINY,
        ay_raw / jnp.maximum(y_sum, 1.0e-99),
        ay_raw,
    )
    mask_all = (ay >= _TINY) & (azion >= 0.0)
    cmi_mean = jnp.sum(jnp.where(mask_all, ay * acmi, 0.0))
    mask = (ay >= _TINY) & (azion > 0.0)
    y = jnp.where(mask, ay, 0.0)
    z = jnp.where(mask, azion, 1.0)
    cmi_s = jnp.where(mask, acmi, 1.0)

    zmean = jnp.sum(y * z)
    z2mean = jnp.sum(y * z * z)
    z13 = jnp.power(z, 1.0 / 3.0)
    z53 = jnp.sum(y * (z13**5))
    z73 = jnp.sum(y * (z13**7))
    z52 = jnp.sum(y * (jnp.sqrt(z) ** 5))
    z321 = jnp.sum(y * z * (jnp.sqrt(z + 1.0) ** 3))

    temp_au = t * (1.0e-6 / _UN_T6)
    t_au2 = temp_au * temp_au
    uint_rad = t_au2 * t_au2 * _RAD
    press_rad = uint_rad / 3.0

    dens0 = rho / 11.20587 * (zmean / cmi_mean)
    chi = _pc_elect_jax.chemfit_jax(dens0, temp_au)
    dens, _, pe_id, ue_id, se_id, cve, chite, chire = _pc_elect_jax.elect11_jax(
        temp_au, chi
    )
    dte = dens * temp_au
    presse = pe_id * dte
    uinte = ue_id * dte

    rs = jnp.power((1.0 / dens) * (0.75 / jnp.pi), 1.0 / 3.0)
    rsi = rs * (cmi_mean * z73 * _AUM)
    game = 1.0 / (rs * temp_au)
    gami_mean = game * z53

    rsi_v = jax.lax.stop_gradient(rsi)
    gm_v = jax.lax.stop_gradient(gami_mean)
    is_liq = (rsi_v < _RSIMELT) | (gm_v < g_start)
    is_sol = (~is_liq) & (gm_v > g_full)
    a_blend = (gami_mean - g_start) / (g_full - g_start)
    alfa = jnp.where(is_liq, 0.0, jnp.where(is_sol, 1.0, a_blend))
    beta = jnp.where(is_liq, 1.0, jnp.where(is_sol, 0.0, -a_blend + 1.0))

    densi = dens / zmean
    pressi = densi * temp_au
    ctp = (1.0 / t_au2) * (4.0 * jnp.pi / _AUM)
    xc_zone = _pc_screen_jax.PcContrib(*_pc_coulomb_jax.excor7_jax(rs, game))

    def _species_contrib(
        cmi_i: jax.Array, z_i: jax.Array, y_i: jax.Array
    ) -> tuple[jax.Array, ...]:
        gami_i = game * jnp.power(z_i, 5.0 / 3.0)
        dni = densi * y_i
        pri = dni * temp_au
        c0, c1 = _pc_screen_jax.eosfi8_c2_pair_jax(
            cmi_i, z_i, rs, gami_i, xc=xc_zone
        )
        c2 = tuple(alfa * v1 + beta * v0 for v0, v1 in zip(c0, c1, strict=True))
        return (
            c2[1] * pri,
            dni * (c2[3] - jnp.log(jnp.maximum(y_i, 1.0e-99))),
            c2[2] * pri,
            dni * c2[4],
            pri * c2[5],
            pri * c2[6],
            ctp * dni * (z_i * z_i / cmi_i),
        )

    def _sp_step(
        i: jax.Array, acc: tuple[jax.Array, ...]
    ) -> tuple[jax.Array, ...]:
        m_i = mask[i]
        contrib = _species_contrib(cmi_s[i], z[i], y[i])
        return tuple(
            a_k + jnp.where(m_i, c_k, 0.0)
            for a_k, c_k in zip(acc, contrib, strict=True)
        )

    init_acc = (
        uinte,
        se_id * dens,
        presse,
        cve * dens,
        presse * chite,
        presse * chire,
        jnp.zeros_like(rho),
    )
    uint, s_tot, press, cv_tot, pdlt, pdlr, tpt2 = jax.lax.fori_loop(
        0, n_real, _sp_step, init_acc
    )

    fwk = jnp.where(is_liq, tpt2 / 24.0, 0.0)
    uwk = fwk * 2.0
    uint = uint + uwk * pressi
    s_tot = s_tot + fwk * densi
    press = press + uwk * 0.5 * pressi
    cv_tot = cv_tot - uwk * densi
    pdlt = pdlt - fwk * pressi
    pdlr = pdlr + uwk * pressi

    mix_liq = _pc_screen_jax.cormix_jax(rs, game, zmean, z2mean, z52, z53, z321)
    fmix_sol = game * _pc_screen_jax.solid_fmix_coeff_jax(ay, azion)
    z0 = jnp.zeros_like(fmix_sol)
    mix_sol = (fmix_sol, fmix_sol, fmix_sol / 3.0, z0, z0, z0, fmix_sol / 2.25)
    mix = tuple(
        jnp.where(is_liq, ml, ms)
        for ml, ms in zip(mix_liq, mix_sol, strict=True)
    )

    uint = uint + mix[1] * pressi
    s_tot = s_tot + densi * (mix[1] - mix[0])
    press = press + mix[2] * pressi
    cv_tot = cv_tot + densi * mix[4]
    pdlt = pdlt + pressi * mix[5]
    pdlr = pdlr + pressi * mix[6]

    prad_nkt = press_rad / pressi
    p_nkt0 = press / pressi
    u_nkt0 = uint / pressi
    s_nk0 = s_tot / densi
    cv_nkt0 = cv_tot / densi
    chir0 = pdlr / press
    chit0 = pdlt / press

    tnk = rho * t * (8.31447e7 / cmi_mean)
    pgas = p_nkt0 * tnk
    p_tot_nkt = p_nkt0 + prad_nkt
    chit_rad = chit0 * (p_nkt0 / p_tot_nkt) + prad_nkt * 4.0 / p_tot_nkt
    chir_rad = chir0 * (p_nkt0 / p_tot_nkt)
    inc = jnp.asarray(include_radiation)
    p_nkt = jnp.where(inc, p_tot_nkt, p_nkt0)
    u_nkt = jnp.where(inc, u_nkt0 + prad_nkt * 3.0, u_nkt0)
    s_nk = jnp.where(inc, s_nk0 + prad_nkt * 4.0, s_nk0)
    cv_nkt = jnp.where(inc, cv_nkt0 + prad_nkt * 12.0, cv_nkt0)
    chir = jnp.where(inc, chir_rad, chir0)
    chit = jnp.where(inc, chit_rad, chit0)

    p = p_nkt * tnk
    n_ion = 1.0 / (abar * constants.AMU)
    nkerg = n_ion * constants.KERG
    cv = cv_nkt * nkerg
    gamma3 = p / rho * chit / (t * cv) + 1.0
    gamma1 = chit * (gamma3 - 1.0) + chir
    grad_ad = (gamma3 - 1.0) / gamma1
    cp = cv * gamma1 / chir
    de_drho = (-chit + 1.0) * p / (rho * rho)
    ds_dt = cv / t
    ds_drho = -p * chit / (rho * rho * t)
    mu = jax.lax.stop_gradient(abar / (1.0 + zbar))
    lnfree_e = jax.lax.stop_gradient(jnp.log(zbar / abar))
    ln_pgas = jnp.log(pgas)
    e_val = u_nkt * t * nkerg
    s_val = s_nk * nkerg
    ln_e = jnp.where(
        jax.lax.stop_gradient(e_val) < 1.0e-99,
        jnp.log(1.0e-99),
        jnp.log(jnp.maximum(e_val, 1.0e-99)),
    )
    ln_s = jnp.where(
        jax.lax.stop_gradient(s_val) < 1.0e-99,
        jnp.log(1.0e-99),
        jnp.log(jnp.maximum(s_val, 1.0e-99)),
    )
    phase = jnp.where(
        gm_v < g_start,
        0.0,
        jnp.where(gm_v > g_full, 1.0, a_blend),
    )
    o = jnp.ones_like(rho)
    return jnp.stack(
        [
            ln_pgas,
            ln_e,
            ln_s,
            mu * o,
            lnfree_e * o,
            chi,
            grad_ad,
            chir,
            chit,
            cp,
            cv,
            de_drho,
            ds_dt,
            ds_drho,
            gamma1,
            gamma3,
            phase,
            z0,
            z0,
            z0,
            z0,
            z0,
            o,
            z0,
            z0,
            z0,
        ],
        axis=-1,
    )


def _pad_pc_species(
    xa: jax.Array, azion: jax.Array, acmi: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pad species arrays to bucket size `8` or `32` with `-1.0` sentinel."""
    n_sp = int(azion.shape[-1])
    bucket = _jet_jax.bucket_size(n_sp)
    return (
        jnp.int32(n_sp),
        _jet_jax.pad_trailing(xa, bucket, -1.0),
        _jet_jax.pad_trailing(azion, bucket, -1.0),
        _jet_jax.pad_trailing(acmi, bucket, 1.0),
    )


def _eval_pc_jvp_dir(
    rho: jax.Array,
    t: jax.Array,
    drho: jax.Array,
    dt: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xa: jax.Array,
    azion: jax.Array,
    acmi: jax.Array,
    include_radiation: jax.Array,
    g_start: jax.Array,
    g_full: jax.Array,
    mf_lim: jax.Array,
    n_real: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate `(res, d_res)` along `(drho, dt)` for a single zone."""
    res_p, d_res = jax.jvp(
        lambda r_in, t_in: _eval_pc_res_single(
            r_in,
            t_in,
            abar,
            zbar,
            xa,
            azion,
            acmi,
            include_radiation,
            g_start,
            g_full,
            mf_lim,
            n_real,
        ),
        (rho, t),
        (drho, dt),
    )
    return res_p, d_res


def _eval_pc_single_traced(
    rho: jax.Array,
    t: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xa: jax.Array,
    azion: jax.Array,
    acmi: jax.Array,
    include_radiation: jax.Array,
    g_start: jax.Array,
    g_full: jax.Array,
    mf_lim: jax.Array,
    n_real: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate `(res, d_dlnd, d_dln_t)` with DCE-friendly primal `res`."""
    z0 = jnp.zeros_like(rho)
    args = (
        abar,
        zbar,
        xa,
        azion,
        acmi,
        include_radiation,
        g_start,
        g_full,
        mf_lim,
        n_real,
    )
    res = _eval_pc_res_single(rho, t, *args)
    _, d_dlnd = _eval_pc_jvp_dir(rho, t, rho, z0, *args)
    _, d_dln_t = _eval_pc_jvp_dir(rho, t, z0, t, *args)
    return res, d_dlnd, d_dln_t


_eval_pc_jvp_dir_jit = jax.jit(_eval_pc_jvp_dir)
_eval_pc_zones_vmap = jax.vmap(
    _eval_pc_single_traced,
    in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None),
)
_eval_pc_zones_jvp_dir_jit = jax.jit(
    jax.vmap(
        _eval_pc_jvp_dir,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None),
    )
)


def _eval_pc_single_eager(
    r_s: jax.Array,
    t_s: jax.Array,
    *tail: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate single-zone `(res, d_dlnd, d_dln_t)` via directional JVPs."""
    z0 = jnp.zeros((), dtype=jnp.float64)
    res_s, dlnd_s = _eval_pc_jvp_dir_jit(r_s, t_s, r_s, z0, *tail)
    _, dlnt_s = _eval_pc_jvp_dir_jit(r_s, t_s, z0, t_s, *tail)
    return res_s, dlnd_s, dlnt_s


def _eval_pc_native_core(
    r_in: jax.Array,
    t_in: jax.Array,
    a_in: jax.Array,
    z_in: jax.Array,
    x_in: jax.Array,
    az_in: jax.Array,
    ac_in: jax.Array,
    inc_rad_j: jax.Array,
    g_start_j: jax.Array,
    g_full_j: jax.Array,
    mf_lim_j: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate native JAX `PC` EOS on bucketed species arrays."""
    orig_shape = r_in.shape
    n = core.NUM_EOS_BASIC_RESULTS
    if r_in.size == 0:
        empty = jnp.zeros((*orig_shape, n), dtype=jnp.float64)
        return empty, empty, empty
    n_real, x_p, az_a, ac_a = _pad_pc_species(
        x_in, az_in.reshape(-1), ac_in.reshape(-1)
    )
    j_args = (inc_rad_j, g_start_j, g_full_j, mf_lim_j, n_real)
    a_a = jnp.broadcast_to(a_in, orig_shape)
    z_a = jnp.broadcast_to(z_in, orig_shape)
    sp = az_a.shape[0]
    x_a = jnp.broadcast_to(x_p, (*orig_shape, sp))
    traced = any(
        isinstance(v, jax.core.Tracer)
        for v in (r_in, t_in, a_a, z_a, x_a, az_a, ac_a)
    )
    if not orig_shape:
        if traced:
            return _eval_pc_single_traced(
                r_in, t_in, a_a, z_a, x_a, az_a, ac_a, *j_args
            )
        return _eval_pc_single_eager(
            r_in, t_in, a_a, z_a, x_a, az_a, ac_a, *j_args
        )

    r_1d, t_1d = r_in.reshape(-1), t_in.reshape(-1)
    a_1d, z_1d = a_a.reshape(-1), z_a.reshape(-1)
    x_2d = x_a.reshape((-1, sp))
    if traced:
        res_v, dlnd_v, dlnt_v = _eval_pc_zones_vmap(
            r_1d, t_1d, a_1d, z_1d, x_2d, az_a, ac_a, *j_args
        )
    elif r_1d.shape[0] <= 16:
        rows = [
            _eval_pc_single_eager(
                r_1d[k],
                t_1d[k],
                a_1d[k],
                z_1d[k],
                x_2d[k],
                az_a,
                ac_a,
                *j_args,
            )
            for k in range(r_1d.shape[0])
        ]
        res_v = jnp.stack([row[0] for row in rows], axis=0)
        dlnd_v = jnp.stack([row[1] for row in rows], axis=0)
        dlnt_v = jnp.stack([row[2] for row in rows], axis=0)
    else:
        z0_1d = jnp.zeros_like(r_1d)
        res_v, dlnd_v = _eval_pc_zones_jvp_dir_jit(
            r_1d, t_1d, r_1d, z0_1d, a_1d, z_1d, x_2d, az_a, ac_a, *j_args
        )
        _, dlnt_v = _eval_pc_zones_jvp_dir_jit(
            r_1d, t_1d, z0_1d, t_1d, a_1d, z_1d, x_2d, az_a, ac_a, *j_args
        )
    return (
        res_v.reshape((*orig_shape, n)),
        dlnd_v.reshape((*orig_shape, n)),
        dlnt_v.reshape((*orig_shape, n)),
    )


_pc_native_jvp_fn = jax.custom_jvp(_eval_pc_native_core)


@_pc_native_jvp_fn.defjvp
def _eval_pc_native_jvp(
    primals: tuple[jax.Array, ...], tangents: tuple[jax.Array, ...]
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array],
]:
    return core.pack_dt_jvp(
        primals[0],
        primals[1],
        tangents[0],
        tangents[1],
        _pc_native_jvp_fn(*primals),
    )


def eval_pc_eos_jax(
    rho: jax.Array,
    temp: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xa: jax.Array,
    azion: jax.Array,
    acmi: jax.Array,
    *,
    include_radiation: bool = True,
    pc_gamma_start_crystal: float = 170.0,
    pc_gamma_full_crystal: float = 180.0,
    mass_fraction_limit: float = 1.0e-4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate 26-channel `PC` EOS in native `@jax.jit` with `custom_jvp`."""
    f64 = jnp.float64
    sg = jax.lax.stop_gradient
    res_o, dlnd_0, dlnt_0 = _pc_native_jvp_fn(
        jnp.asarray(rho, f64),
        jnp.asarray(temp, f64),
        sg(jnp.asarray(abar, f64)),
        sg(jnp.asarray(zbar, f64)),
        sg(jnp.asarray(xa, f64)),
        sg(jnp.asarray(azion, f64)).reshape(-1),
        sg(jnp.asarray(acmi, f64)).reshape(-1),
        sg(jnp.asarray(bool(include_radiation))),
        sg(jnp.asarray(float(pc_gamma_start_crystal), dtype=f64)),
        sg(jnp.asarray(float(pc_gamma_full_crystal), dtype=f64)),
        sg(jnp.asarray(float(mass_fraction_limit), dtype=f64)),
    )
    return res_o, dlnd_0, dlnt_0
