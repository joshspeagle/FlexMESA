# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native `@jax.jit` kernels and `custom_jvp` rules for the `PC` EOS.

Provides `get_pc_alfa_jax` and `eval_pc_eos_jax` in pure vectorized JAX
(`jnp.float64`), alongside explicit `solver_mode="rust_ffi"` host callback
support.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _rust, constants
from flexmesa.eos import (
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


def get_pc_alfa_jax(
    log_rho: jax.Array,
    log_t: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    use_gamma_limit: bool = True,
    log_rho1: float = 2.999,
    log_rho2: float = 2.8,
    log_t1: float = 7.6,
    log_t2: float = 7.7,
    log_gamma_e_all_helm: float = -1.8,
    log_gamma_e_all_pc: float = -1.7,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate `Get_PC_alfa` (`eospc_eval.f90` lines 31-96) in JAX."""
    log_rho_lo0 = jnp.asarray(log_rho2, dtype=jnp.float64)
    log_rho_hi = jnp.asarray(log_rho1, dtype=jnp.float64)

    log_ge0 = jnp.log10(
        constants.QE
        * constants.QE
        * jnp.power(
            constants.FOUR_THIRDS_PI * constants.AVO * zbar / abar,
            constants.ONE_THIRD,
        )
        / constants.KERG
    )
    log_ge = log_ge0 + log_rho / 3.0 - log_t
    log_ge_lo = jnp.asarray(log_gamma_e_all_helm, dtype=jnp.float64)
    log_ge_hi = jnp.asarray(log_gamma_e_all_pc, dtype=jnp.float64)
    log_t_lo_g = log_ge0 + log_rho_lo0 / 3.0 - log_ge_lo
    log_t_hi_g = log_ge0 + log_rho_hi / 3.0 - log_ge_hi

    span_ge = log_ge_hi - log_ge_lo
    alfa_ge = (log_ge_hi - log_ge) / span_ge
    dt_ge = jnp.ones_like(log_rho) / span_ge
    dr_ge = -(1.0 / 3.0) * jnp.ones_like(log_rho) / span_ge

    log_rho_lo_eff = jnp.where(
        log_t >= log_t_lo_g,
        (log_t_lo_g + log_ge_lo - log_ge0) * 3.0,
        log_rho_lo0,
    )
    span_r = log_rho_hi - log_rho_lo_eff
    alfa_r = (log_rho_hi - log_rho) / span_r
    dr_r = -1.0 / span_r

    out_outside_g = (log_rho <= log_rho_lo0) | (log_ge <= log_ge_lo)
    out_inside_g = (log_rho >= log_rho_hi) & (log_ge >= log_ge_hi)
    use_ge_branch = log_t >= log_t_hi_g

    alfa_g = jnp.where(
        out_outside_g,
        1.0,
        jnp.where(out_inside_g, 0.0, jnp.where(use_ge_branch, alfa_ge, alfa_r)),
    )
    dt_g = jnp.where(
        out_outside_g | out_inside_g,
        0.0,
        jnp.where(use_ge_branch, dt_ge, 0.0),
    )
    dr_g = jnp.where(
        out_outside_g | out_inside_g,
        0.0,
        jnp.where(use_ge_branch, dr_ge, dr_r),
    )

    span_t = log_t2 - log_t1
    alfa_t_br = (log_t - log_t1) / span_t
    dt_t_br = jnp.ones_like(log_rho) / span_t
    span_r0 = log_rho_hi - log_rho_lo0
    alfa_r0 = (log_rho_hi - log_rho) / span_r0
    dr_r0 = -jnp.ones_like(log_rho) / span_r0

    out_outside_t = (log_rho <= log_rho_lo0) | (log_t >= log_t2)
    out_inside_t = (log_rho >= log_rho_hi) & (log_t <= log_t1)
    use_t_branch = log_t >= log_t1

    alfa_t = jnp.where(
        out_outside_t,
        1.0,
        jnp.where(
            out_inside_t, 0.0, jnp.where(use_t_branch, alfa_t_br, alfa_r0)
        ),
    )
    dt_t = jnp.where(
        out_outside_t | out_inside_t,
        0.0,
        jnp.where(use_t_branch, dt_t_br, 0.0),
    )
    dr_t = jnp.where(
        out_outside_t | out_inside_t,
        0.0,
        jnp.where(use_t_branch, 0.0, dr_r0),
    )

    use_g = jnp.asarray(use_gamma_limit)
    return (
        jnp.where(use_g, alfa_g, alfa_t),
        jnp.where(use_g, dt_g, dt_t),
        jnp.where(use_g, dr_g, dr_t),
    )


def _solid_fmix_coeff_jax(ay: jax.Array, azion: jax.Array) -> jax.Array:
    """Compute solid pair-mixing coefficient (`melange9` lines 210-233)."""
    n = ay.shape[0]
    idx = jnp.arange(n)
    upper = idx[:, None] < idx[None, :]
    valid_i = (ay >= _TINY) & (azion > 0.0)
    pair_mask = upper & valid_i[:, None] & valid_i[None, :]

    zi = jnp.where(valid_i, azion, 1.0)[:, None]
    zj = jnp.where(valid_i, azion, 1.0)[None, :]
    yi = jnp.where(valid_i, ay, 0.0)[:, None]
    yj = jnp.where(valid_i, ay, 0.0)[None, :]

    rz = zj / zi
    ysum_ij = jnp.maximum(yi + yj, 1.0e-99)
    x2 = jnp.where(pair_mask, yj / ysum_ij, 0.5)
    x1 = jnp.maximum(1.0 - x2, 1.0e-99)
    pair_mask = pair_mask & (x2 >= _TINY) & ((1.0 - x2) > 0.0)

    x = x2 / rz + (1.0 - 1.0 / rz) * jnp.power(jnp.maximum(x2, 1.0e-99), rz)
    delta_g = (
        0.012
        * (1.0 - 1.0 / (rz * rz))
        * (x1 + x2 * jnp.power(rz, 5.0 / 3.0))
        * (x / jnp.maximum(x2, 1.0e-99))
        * jnp.maximum(1.0 - x, 0.0)
        / x1
    )
    terms = jnp.power(zi, 5.0 / 3.0) * yi * yj * delta_g
    return jnp.sum(jnp.where(pair_mask, terms, 0.0))


def _eval_pc_res_single(
    rho: jax.Array,
    t: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xa: jax.Array,
    azion: jax.Array,
    acmi: jax.Array,
    include_radiation: bool,
    g_start: float,
    g_full: float,
    mf_lim: float,
) -> jax.Array:
    """Evaluate the 26-channel `PC` `res` vector for a single zone in JAX."""
    ay_raw = jnp.where(xa >= mf_lim, xa / jnp.maximum(acmi, 1.0e-99), 0.0)
    y_sum = jnp.sum(ay_raw)
    ay = jnp.where(
        jnp.abs(y_sum - 1.0) > _TINY,
        ay_raw / jnp.maximum(y_sum, 1.0e-99),
        ay_raw,
    )
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
    cmi_mean = jnp.sum(y * cmi_s)

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

    def _species_contrib(
        cmi_i: jax.Array, z_i: jax.Array, y_i: jax.Array
    ) -> tuple[jax.Array, ...]:
        gami_i = game * jnp.power(z_i, 5.0 / 3.0)
        dni = densi * y_i
        pri = dni * temp_au
        c0, c1 = _pc_screen_jax.eosfi8_c2_pair_jax(cmi_i, z_i, rs, gami_i)
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

    du, ds, dp, dcv, dpdlt, dpdlr, dtpt2 = jax.vmap(_species_contrib)(
        cmi_s, z, y
    )
    uint = uinte + jnp.sum(jnp.where(mask, du, 0.0))
    s_tot = se_id * dens + jnp.sum(jnp.where(mask, ds, 0.0))
    press = presse + jnp.sum(jnp.where(mask, dp, 0.0))
    cv_tot = cve * dens + jnp.sum(jnp.where(mask, dcv, 0.0))
    pdlt = presse * chite + jnp.sum(jnp.where(mask, dpdlt, 0.0))
    pdlr = presse * chire + jnp.sum(jnp.where(mask, dpdlr, 0.0))
    tpt2 = jnp.sum(jnp.where(mask, dtpt2, 0.0))

    fwk = jnp.where(is_liq, tpt2 / 24.0, 0.0)
    uwk = fwk * 2.0
    uint = uint + uwk * pressi
    s_tot = s_tot + fwk * densi
    press = press + uwk * 0.5 * pressi
    cv_tot = cv_tot - uwk * densi
    pdlt = pdlt - fwk * pressi
    pdlr = pdlr + uwk * pressi

    mix_liq = _pc_screen_jax.cormix_jax(rs, game, zmean, z2mean, z52, z53, z321)
    fmix_sol = game * _solid_fmix_coeff_jax(ay, azion)
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
    if include_radiation:
        p_tot_nkt = p_nkt0 + prad_nkt
        chit = chit0 * (p_nkt0 / p_tot_nkt) + prad_nkt * 4.0 / p_tot_nkt
        chir = chir0 * (p_nkt0 / p_tot_nkt)
        p_nkt = p_tot_nkt
        u_nkt = u_nkt0 + prad_nkt * 3.0
        s_nk = s_nk0 + prad_nkt * 4.0
        cv_nkt = cv_nkt0 + prad_nkt * 12.0
    else:
        p_nkt, u_nkt, s_nk, cv_nkt, chir, chit = (
            p_nkt0,
            u_nkt0,
            s_nk0,
            cv_nkt0,
            chir0,
            chit0,
        )

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


@functools.partial(
    jax.jit,
    static_argnames=("include_radiation", "g_start", "g_full", "mf_lim"),
)
def _eval_pc_single_jit(
    rho: jax.Array,
    t: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xa: jax.Array,
    azion: jax.Array,
    acmi: jax.Array,
    include_radiation: bool,
    g_start: float,
    g_full: float,
    mf_lim: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate `(res, d_dlnd, d_dln_t)` for a single zone via forward JVP."""
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
    )
    def _fn(r_in: jax.Array, t_in: jax.Array) -> jax.Array:
        return _eval_pc_res_single(r_in, t_in, *args)

    res = _fn(rho, t)
    d_drho, d_dt = jax.jacfwd(_fn, argnums=(0, 1))(rho, t)
    return res, d_drho * rho, d_dt * t


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
    solver_mode: str = "jax",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate 26-channel `PC` EOS in native `@jax.jit` with `custom_jvp`."""
    inc_rad = bool(include_radiation)
    g_start = float(pc_gamma_start_crystal)
    g_full = float(pc_gamma_full_crystal)
    mf_lim = float(mass_fraction_limit)
    c_args = (inc_rad, g_start, g_full, mf_lim)

    def _eval_pc_core(
        r_in: jax.Array,
        t_in: jax.Array,
        a_in: jax.Array,
        z_in: jax.Array,
        x_in: jax.Array,
        az_in: jax.Array,
        ac_in: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if solver_mode == "rust_ffi":
            return _eval_pc_rust_ffi_core(
                r_in, t_in, a_in, z_in, x_in, az_in, ac_in, *c_args
            )
        r_a = jnp.asarray(r_in, dtype=jnp.float64)
        t_a = jnp.asarray(t_in, dtype=jnp.float64)
        orig_shape = r_a.shape
        a_a = jnp.broadcast_to(jnp.asarray(a_in, dtype=jnp.float64), orig_shape)
        z_a = jnp.broadcast_to(jnp.asarray(z_in, dtype=jnp.float64), orig_shape)
        az_a = jnp.asarray(az_in, dtype=jnp.float64).reshape(-1)
        ac_a = jnp.asarray(ac_in, dtype=jnp.float64).reshape(-1)
        sp = az_a.shape[0]
        x_a = jnp.broadcast_to(
            jnp.asarray(x_in, dtype=jnp.float64), (*orig_shape, sp)
        )
        if orig_shape == ():
            r_s, d_s, t_s = _eval_pc_single_jit(
                r_a, t_a, a_a, z_a, x_a, az_a, ac_a, *c_args
            )
            return r_s, d_s, t_s
        r_1d, t_1d = r_a.reshape(-1), t_a.reshape(-1)
        a_1d, z_1d = a_a.reshape(-1), z_a.reshape(-1)
        x_2d = x_a.reshape((-1, sp))
        vmapped = jax.vmap(
            lambda ri, ti, ai, zi, xi: _eval_pc_single_jit(
                ri, ti, ai, zi, xi, az_a, ac_a, *c_args
            )
        )
        res_v, dlnd_v, dlnt_v = vmapped(r_1d, t_1d, a_1d, z_1d, x_2d)
        n = core.NUM_EOS_BASIC_RESULTS
        return (
            res_v.reshape((*orig_shape, n)),
            dlnd_v.reshape((*orig_shape, n)),
            dlnt_v.reshape((*orig_shape, n)),
        )

    _eval_pc = jax.custom_jvp(_eval_pc_core)

    def _eval_pc_jvp(
        primals: tuple[jax.Array, ...], tangents: tuple[jax.Array, ...]
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array],
        tuple[jax.Array, jax.Array, jax.Array],
    ]:
        r_p, t_p, a_p, z_p, x_p, az_p, ac_p = primals
        dr, dt, _, _, _, _, _ = tangents
        res, d_dlnd, d_dln_t = _eval_pc_core(
            r_p, t_p, a_p, z_p, x_p, az_p, ac_p
        )
        dln_rho = jnp.expand_dims(dr / r_p, axis=-1)
        dln_t = jnp.expand_dims(dt / t_p, axis=-1)
        d_res = d_dlnd * dln_rho + d_dln_t * dln_t
        z = jnp.zeros_like(res)
        return (res, d_dlnd, d_dln_t), (d_res, z, z)

    _eval_pc.defjvp(_eval_pc_jvp)
    res_o, dlnd_o, dlnt_o = _eval_pc(rho, temp, abar, zbar, xa, azion, acmi)
    return res_o, dlnd_o, dlnt_o


def _eval_pc_rust_ffi_core(
    r_in: jax.Array,
    t_in: jax.Array,
    a_in: jax.Array,
    z_in: jax.Array,
    x_in: jax.Array,
    az_in: jax.Array,
    ac_in: jax.Array,
    inc_rad: bool,
    g_start: float,
    g_full: float,
    mf_lim: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Explicit `solver_mode='rust_ffi'` host callback for `PC` EOS."""

    def _host_eval_pc(
        rho_np: np.ndarray,
        temp_np: np.ndarray,
        abar_np: np.ndarray,
        zbar_np: np.ndarray,
        xa_np: np.ndarray,
        azion_np: np.ndarray,
        acmi_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        orig_shape = np.shape(rho_np)
        r_1d = np.ascontiguousarray(np.asarray(rho_np, np.float64).reshape(-1))
        t_1d = np.ascontiguousarray(np.asarray(temp_np, np.float64).reshape(-1))
        nz = r_1d.size
        a_1d = np.broadcast_to(
            np.asarray(abar_np, np.float64), orig_shape
        ).reshape(nz)
        z_1d = np.broadcast_to(
            np.asarray(zbar_np, np.float64), orig_shape
        ).reshape(nz)
        az_1d = np.ascontiguousarray(
            np.asarray(azion_np, np.float64).reshape(-1)
        )
        ac_1d = np.ascontiguousarray(
            np.asarray(acmi_np, np.float64).reshape(-1)
        )
        sp = az_1d.size
        xa_2d = np.broadcast_to(
            np.asarray(xa_np, np.float64), (*orig_shape, sp)
        ).reshape(-1)
        out = _rust.eos_eval_pc_zones(
            r_1d,
            t_1d,
            np.ascontiguousarray(a_1d),
            np.ascontiguousarray(z_1d),
            np.ascontiguousarray(xa_2d),
            az_1d,
            ac_1d,
            inc_rad,
            g_start,
            g_full,
            mf_lim,
            True,
        )
        n = core.NUM_EOS_BASIC_RESULTS
        return (
            out[:, :n].reshape((*orig_shape, n)),
            out[:, n : 2 * n].reshape((*orig_shape, n)),
            out[:, 2 * n : 3 * n].reshape((*orig_shape, n)),
        )

    shape = (*r_in.shape, core.NUM_EOS_BASIC_RESULTS)
    vc = jax.ShapeDtypeStruct(shape, jnp.float64)
    r_c, d_c, t_c = jax.pure_callback(
        _host_eval_pc,
        (vc, vc, vc),
        r_in,
        t_in,
        a_in,
        z_in,
        x_in,
        az_in,
        ac_in,
        vmap_method="expand_dims",
    )
    return r_c, d_c, t_c
