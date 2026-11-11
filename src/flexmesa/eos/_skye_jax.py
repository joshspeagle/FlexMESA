# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branchless `@jax.jit` (`float64`) 3rd-order jet kernels for `Skye` EOS.

Implements Jermyn et al. (2021, ApJ, 913, 72) third-order Helmholtz free-energy
minimization (`skye.f90`, `skye_ideal.f90`, `skye_coulomb*.f90`, and
`skye_thermodynamics.f90`) using flat `(..., 10)` `AutoDiffReal2VarOrder3`
Taylor jets (`_jet_jax`) and `jax.lax.fori_loop` species accumulation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp

from flexmesa import constants
from flexmesa.eos import (
    _helm_jax_poly,
    _jet_jax,
    _skye_coulomb_jax,
    core,
)

_SKYE_MIN_GAMMA_SOLID: float = 170.0
_SKYE_MAX_GAMMA_LIQUID: float = 200.0
_NIST_EV: jax.Array = jnp.asarray(
    core.NIST_IONIZATION_TABLE_EV, dtype=jnp.float64
)


def _pad_skye_species(
    xa: jax.Array, azion: jax.Array, acmi: jax.Array, a_in: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pad species arrays to bucket size `8` or `32` with `-1.0` sentinel."""
    n_sp = int(azion.shape[-1])
    bucket = _jet_jax.bucket_size(n_sp)
    return (
        jnp.int32(n_sp),
        _jet_jax.pad_trailing(xa, bucket, -1.0),
        _jet_jax.pad_trailing(azion, bucket, -1.0),
        _jet_jax.pad_trailing(acmi, bucket, 1.0),
        _jet_jax.pad_trailing(a_in, bucket, 1.0),
    )


def _eval_electron_positron_j3(
    temp_v: jax.Array,
    d_v: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    t_lo: jax.Array,
    dth: jax.Array,
    d_lo: jax.Array,
    dd: jax.Array,
    fi_f: jax.Array,
    fi_ef: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Build `(f_ele, eta_ele)` `(10,)` jets (`elec_pos.rs:239-278`)."""
    ye = zbar / abar
    din = ye * d_v
    dt2, dd2 = dth * dth, dd * dd
    dti, ddi = 1.0 / dth, 1.0 / dd
    dt2i, dd2i = 1.0 / dt2, 1.0 / dd2
    xt = jnp.maximum((temp_v - t_lo) * dti, 0.0)
    xd = jnp.maximum((din - d_lo) * ddi, 0.0)
    qt0, qt1, qt2, qt3 = _helm_jax_poly.quintic_basis_1d(
        xt, dth, dt2, dti, dt2i, dt2i * dti
    )
    qd0, qd1, qd2, qd3 = _helm_jax_poly.quintic_basis_1d(
        xd, dd, dd2, ddi, dd2i, dd2i * ddi
    )
    ct0, ct1, _ = _helm_jax_poly.cubic_basis_1d(xt, dth, dti, dt2i)
    cd0, cd1, _ = _helm_jax_poly.cubic_basis_1d(xd, dd, ddi, dd2i)
    ye2 = ye * ye
    ye3 = ye2 * ye
    ye4 = ye2 * ye2
    f_ele = jnp.stack(
        [
            _helm_jax_poly.h5(fi_f, qt0, qd0) * ye,
            _helm_jax_poly.h5(fi_f, qt1, qd0) * ye,
            _helm_jax_poly.h5(fi_f, qt0, qd1) * ye2,
            _helm_jax_poly.h5(fi_f, qt2, qd0) * ye,
            _helm_jax_poly.h5(fi_f, qt1, qd1) * ye2,
            _helm_jax_poly.h5(fi_f, qt0, qd2) * ye3,
            _helm_jax_poly.h5(fi_f, qt3, qd0) * ye,
            _helm_jax_poly.h5(fi_f, qt2, qd1) * ye2,
            _helm_jax_poly.h5(fi_f, qt1, qd2) * ye3,
            _helm_jax_poly.h5(fi_f, qt0, qd3) * ye4,
        ],
        axis=-1,
    )
    z0 = jnp.zeros_like(temp_v)
    eta_ele = jnp.stack(
        [
            _helm_jax_poly.h3(fi_ef, ct0, cd0),
            _helm_jax_poly.h3(fi_ef, ct1, cd0),
            _helm_jax_poly.h3(fi_ef, ct0, cd1) * ye,
            z0,
            z0,
            z0,
            z0,
            z0,
            z0,
            z0,
        ],
        axis=-1,
    )
    return f_ele, eta_ele


def _compute_f_ideal_ion_j3(
    temp_j: jax.Array,
    den_j: jax.Array,
    abar: jax.Array,
    mask: jax.Array,
    ya: jax.Array,
    acmi: jax.Array,
    ion_off: jax.Array,
) -> jax.Array:
    """Evaluate `compute_F_ideal_ion` + `ion_off` as `(10,)` jet."""
    two_pi_amu = 2.0 * constants.PI * constants.AMU
    sifac = (constants.PLANCK_H**3) / (two_pi_amu * jnp.sqrt(two_pi_amu))
    kt = _jet_jax.j3_scale(temp_j, constants.KERG)
    n_tot = _jet_jax.j3_scale(den_j, 1.0 / (constants.AMU * abar))
    n_q = _jet_jax.j3_scale(_jet_jax.j3_powf(kt, 1.5), 1.0 / sifac)
    w_s = jnp.where(mask, jnp.maximum(acmi, 1.0e-30), 1.0)
    y_s = jnp.where(mask, jnp.maximum(ya, 1.0e-99), 1.0)
    c_ion = jnp.sum(
        jnp.where(mask, ya * (jnp.log(y_s / jnp.power(w_s, 1.5)) - 1.0), 0.0)
    )
    y_sum = jnp.sum(jnp.where(mask, ya, 0.0))
    f_per_ion = _jet_jax.j3_add_s(
        _jet_jax.j3_scale(_jet_jax.j3_log(_jet_jax.j3_div(n_tot, n_q)), y_sum),
        c_ion,
    )
    f_ion = _jet_jax.j3_scale(
        _jet_jax.j3_mul(f_per_ion, kt), 1.0 / (constants.AMU * abar)
    )
    return _jet_jax.j3_add_s(f_ion, ion_off)


def _pack_for_export_j3(
    f_ion: jax.Array,
    df_coul: jax.Array,
    f_ele: jax.Array,
    temp_j: jax.Array,
    den_j: jax.Array,
    xnefer: jax.Array,
    eta_ele: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    phase: jax.Array,
    latent_ddlnt: jax.Array,
    latent_ddlnrho: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Assemble `(res, d_dlnd, d_dln_t)` (`free_energy.rs:175-250`)."""
    f_gas = f_ion + df_coul + f_ele
    sgas = -_jet_jax.j3_diff1(f_gas)
    pgas = _jet_jax.j3_mul(_jet_jax.j3_pow2(den_j), _jet_jax.j3_diff2(f_gas))
    egas = f_gas + _jet_jax.j3_mul(temp_j, sgas)

    t4 = _jet_jax.j3_pow4(temp_j)
    t3 = _jet_jax.j3_pow3(temp_j)
    prad = _jet_jax.j3_scale(t4, constants.CRAD / 3.0)
    erad = _jet_jax.j3_div(_jet_jax.j3_scale(t4, constants.CRAD), den_j)
    srad = _jet_jax.j3_div(
        _jet_jax.j3_scale(t3, 4.0 * constants.CRAD / 3.0), den_j
    )
    p = prad + pgas
    e = erad + egas
    s = srad + sgas
    valid = (s[0] >= 0.0) & (e[0] >= 0.0) & (pgas[0] >= 0.0)

    ln_pgas = _jet_jax.j3_log(_jet_jax.j3_max_s(pgas, 1.0e-300))
    ln_e = _jet_jax.j3_log(_jet_jax.j3_max_s(e, 1.0e-300))
    ln_s = _jet_jax.j3_log(_jet_jax.j3_max_s(s, 1.0e-300))
    mu_j = _jet_jax.j3_const(abar / (1.0 + zbar))
    ln_free_e = _jet_jax.j3_log(
        _jet_jax.j3_div(
            _jet_jax.j3_max_s(xnefer, 1.0e-99),
            _jet_jax.j3_scale(den_j, constants.AVO),
        )
    )
    chit = _jet_jax.j3_div(_jet_jax.j3_mul(_jet_jax.j3_diff1(p), temp_j), p)
    chid = _jet_jax.j3_div(_jet_jax.j3_mul(_jet_jax.j3_diff2(p), den_j), p)
    cv = _jet_jax.j3_diff1(e)
    gam3_m1 = _jet_jax.j3_div(
        _jet_jax.j3_mul(_jet_jax.j3_div(p, den_j), chit),
        _jet_jax.j3_mul(temp_j, cv),
    )
    gam3 = _jet_jax.j3_add_s(gam3_m1, 1.0)
    gam1 = chid + _jet_jax.j3_mul(gam3_m1, chit)
    nabad = _jet_jax.j3_div(gam3_m1, gam1)
    cp = _jet_jax.j3_div(_jet_jax.j3_mul(cv, gam1), chid)

    ch19 = jnp.stack(
        [
            ln_pgas,
            ln_e,
            ln_s,
            mu_j,
            ln_free_e,
            eta_ele,
            nabad,
            chid,
            chit,
            cp,
            cv,
            _jet_jax.j3_diff2(e),
            _jet_jax.j3_diff1(s),
            _jet_jax.j3_diff2(s),
            gam1,
            gam3,
            phase,
            latent_ddlnt,
            latent_ddlnrho,
        ],
        axis=0,
    )
    z7 = jnp.zeros(7, dtype=jnp.float64)
    frac_res = z7.at[core.I_FRAC_SKYE - 19].set(1.0)
    res = jnp.concatenate([ch19[:, 0], frac_res])
    d_dlnd = jnp.concatenate([ch19[:, 2] * den_j[0], z7])
    d_dln_t = jnp.concatenate([ch19[:, 1] * temp_j[0], z7])
    nan26 = jnp.full_like(res, jnp.nan)
    return (
        jnp.where(valid, res, nan26),
        jnp.where(valid, d_dlnd, nan26),
        jnp.where(valid, d_dln_t, nan26),
    )


def _eval_skye_cell_zone_core(
    rho: jax.Array,
    temp: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xa: jax.Array,
    azion: jax.Array,
    acmi: jax.Array,
    a_in: jax.Array,
    t_lo: jax.Array,
    dth: jax.Array,
    d_lo: jax.Array,
    dd: jax.Array,
    fi_f: jax.Array,
    fi_ef: jax.Array,
    use_ion_offsets: jax.Array,
    use_ogata: jax.Array,
    mass_fraction_limit: jax.Array,
    n_real: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate 26-channel `Skye` EOS on bucketed species arrays."""
    mask = (xa > mass_fraction_limit) & (azion >= 0.0) & (a_in > 0.0)
    sel_x = jnp.where(mask, xa, 0.0)
    norm = jnp.sum(sel_x)
    x_n = sel_x / jnp.where(norm > 0.0, norm, 1.0)
    yj = jnp.where(mask, x_n / jnp.where(mask, a_in, 1.0), 0.0)
    ya_n = jnp.sum(yj)
    ya = yj / jnp.where(ya_n > 0.0, ya_n, 1.0)

    z_idx = jnp.clip(jnp.round(azion).astype(jnp.int32) - 1, 0, 27)
    z_ok = (azion >= 1.0) & (azion <= 28.0) & (a_in > 0.0)
    ev_sum = jnp.sum(
        jnp.where(z_ok, _NIST_EV[z_idx] * xa / jnp.maximum(a_in, 1.0), 0.0)
    )
    ion_off = jnp.where(
        use_ion_offsets, ev_sum * constants.EV2ERG / constants.AMU, 0.0
    )

    temp_j = _jet_jax.j3_var1(temp)
    den_j = _jet_jax.j3_var2(rho)
    f_ele, eta_ele = _eval_electron_positron_j3(
        temp, rho, abar, zbar, t_lo, dth, d_lo, dd, fi_f, fi_ef
    )
    f_ion = _compute_f_ideal_ion_j3(
        temp_j, den_j, abar, mask, ya, acmi, ion_off
    )
    xnefer = _jet_jax.j3_scale(den_j, constants.AVO * (zbar / abar))
    dens_b = _jet_jax.j3_scale(xnefer, constants.RBOHR**3)
    rs = _jet_jax.j3_powf(
        _jet_jax.j3_rdiv_s(3.0 / (4.0 * constants.PI), dens_b), 1.0 / 3.0
    )
    game = _jet_jax.j3_rdiv_s(
        _skye_coulomb_jax.COUL_PREF, _jet_jax.j3_mul(temp_j, rs)
    )

    sp_ok = mask & (ya > 1.0e-7) & (azion > 0.0)
    z_jet = _jet_jax.j3_const(jnp.zeros_like(temp))
    min_g_s = jnp.asarray(_SKYE_MIN_GAMMA_SOLID, dtype=jnp.float64)
    max_g_l = jnp.asarray(_SKYE_MAX_GAMMA_LIQUID, dtype=jnp.float64)

    def _sp_step(
        i: jax.Array, acc: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        dl_acc, ds_acc = acc
        fl_i, fs_i = _skye_coulomb_jax.species_nonideal_j3(
            temp_j, rs, game, azion[i], acmi[i], min_g_s, max_g_l
        )
        w_i = jnp.where(sp_ok[i], ya[i], 0.0)
        return (
            dl_acc + _jet_jax.j3_scale(fl_i, w_i),
            ds_acc + _jet_jax.j3_scale(fs_i, w_i),
        )

    df_l0, df_s0 = jax.lax.fori_loop(0, n_real, _sp_step, (z_jet, z_jet))
    zmean, s_mix, fmix_l, fmix_s = _skye_coulomb_jax.mixing_corrections_j3(
        rs, game, ya, azion, mask, use_ogata
    )
    f_pi = _jet_jax.j3_add_s(
        _jet_jax.j3_scale(_skye_coulomb_jax.excor7_j3(rs, game), zmean), -s_mix
    )
    kt = _jet_jax.j3_scale(temp_j, constants.KERG / (abar * constants.AMU))
    df_liq = _jet_jax.j3_mul(df_l0 + fmix_l + f_pi, kt)
    df_sol = _jet_jax.j3_mul(df_s0 + fmix_s + f_pi, kt)

    df_coul = _jet_jax.j3_min(df_liq, df_sol)
    blur = _jet_jax.j3_scale(kt, 0.01)
    dfl_v, dfs_v, blur_v = df_liq[0], df_sol[0], blur[0]
    ex = _jet_jax.j3_exp(
        _jet_jax.j3_min_s(
            _jet_jax.j3_max_s(_jet_jax.j3_div(df_liq - df_sol, blur), -25.0),
            25.0,
        )
    )
    sig = _jet_jax.j3_div(ex, _jet_jax.j3_add_s(ex, 1.0))
    one_j = _jet_jax.j3_const(jnp.ones_like(temp))
    phase = _jet_jax.j3_where(
        dfl_v < dfs_v - 20.0 * blur_v,
        z_jet,
        _jet_jax.j3_where(dfs_v < dfl_v - 20.0 * blur_v, one_j, sig),
    )
    df_blur = (
        _jet_jax.j3_mul(df_liq, _jet_jax.j3_rsub_s(1.0, phase))
        + _jet_jax.j3_mul(df_sol, phase)
        - df_coul
    )
    latent_s = -_jet_jax.j3_diff1(df_blur)
    latent_ddlnt = _jet_jax.j3_mul(
        _jet_jax.j3_diff1(latent_s), _jet_jax.j3_pow2(temp_j)
    )
    latent_ddlnrho = _jet_jax.j3_mul(
        _jet_jax.j3_mul(temp_j, den_j), _jet_jax.j3_diff2(latent_s)
    )
    res, d_dlnd, d_dln_t = _pack_for_export_j3(
        f_ion,
        df_coul,
        f_ele,
        temp_j,
        den_j,
        xnefer,
        eta_ele,
        abar,
        zbar,
        phase,
        latent_ddlnt,
        latent_ddlnrho,
    )
    nan26 = jnp.full_like(res, jnp.nan)
    ok_comp = norm > 0.0
    return (
        jnp.where(ok_comp, res, nan26),
        jnp.where(ok_comp, d_dlnd, nan26),
        jnp.where(ok_comp, d_dln_t, nan26),
    )


_eval_skye_cell_zone_bucket_jit = jax.jit(_eval_skye_cell_zone_core)
_eval_skye_cell_zones_bucket_jit = jax.jit(
    jax.vmap(
        _eval_skye_cell_zone_core,
        in_axes=(
            0,
            0,
            0,
            0,
            0,
            None,
            None,
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
            None,
            None,
        ),
    )
)


def _run_skye_bucket(
    jit_fn: Callable[..., object],
    rho: jax.Array,
    temp: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xa: jax.Array,
    azion: jax.Array,
    acmi: jax.Array,
    a_in: jax.Array,
    *cell_and_flags: jax.Array,
    mass_fraction_limit: float | jax.Array = 1.0e-4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Promote inputs to `float64`, pad species, and run bucketed `jit_fn`."""
    n_real, xa_p, az_p, ac_p, a_p = _pad_skye_species(
        jnp.asarray(xa, dtype=jnp.float64),
        jnp.asarray(azion, dtype=jnp.float64),
        jnp.asarray(acmi, dtype=jnp.float64),
        jnp.asarray(a_in, dtype=jnp.float64),
    )
    mf_j = jnp.asarray(mass_fraction_limit, dtype=jnp.float64)
    out = jit_fn(
        jnp.asarray(rho, dtype=jnp.float64),
        jnp.asarray(temp, dtype=jnp.float64),
        jnp.asarray(abar, dtype=jnp.float64),
        jnp.asarray(zbar, dtype=jnp.float64),
        xa_p,
        az_p,
        ac_p,
        a_p,
        *cell_and_flags,
        mf_j,
        n_real,
    )
    return cast(tuple[jax.Array, jax.Array, jax.Array], out)


def eval_skye_cell_zone_jax(
    rho: jax.Array,
    temp: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xa: jax.Array,
    azion: jax.Array,
    acmi: jax.Array,
    a_in: jax.Array,
    t_lo: jax.Array,
    dth: jax.Array,
    d_lo: jax.Array,
    dd: jax.Array,
    fi_f: jax.Array,
    fi_ef: jax.Array,
    use_ion_offsets: jax.Array,
    use_ogata: jax.Array,
    mass_fraction_limit: float | jax.Array = 1.0e-4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate 26-channel `Skye` EOS at a single zone on a known `HelmCell`."""
    return _run_skye_bucket(
        _eval_skye_cell_zone_bucket_jit,
        rho,
        temp,
        abar,
        zbar,
        xa,
        azion,
        acmi,
        a_in,
        t_lo,
        dth,
        d_lo,
        dd,
        fi_f,
        fi_ef,
        use_ion_offsets,
        use_ogata,
        mass_fraction_limit=mass_fraction_limit,
    )


def eval_skye_cell_zones_jit(
    rho: jax.Array,
    temp: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xa: jax.Array,
    azion: jax.Array,
    acmi: jax.Array,
    a_in: jax.Array,
    t_lo: jax.Array,
    dth: jax.Array,
    d_lo: jax.Array,
    dd: jax.Array,
    fi_f: jax.Array,
    fi_ef: jax.Array,
    use_ion_offsets: jax.Array,
    use_ogata: jax.Array,
    mass_fraction_limit: float | jax.Array = 1.0e-4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate 26-channel `Skye` EOS across `N_zones` zones on `HelmCell`s."""
    return _run_skye_bucket(
        _eval_skye_cell_zones_bucket_jit,
        rho,
        temp,
        abar,
        zbar,
        xa,
        azion,
        acmi,
        a_in,
        t_lo,
        dth,
        d_lo,
        dd,
        fi_f,
        fi_ef,
        use_ion_offsets,
        use_ogata,
        mass_fraction_limit=mass_fraction_limit,
    )
