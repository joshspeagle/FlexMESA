# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branchless `@jax.jit` (`float64`) kernels for the `Skye` equation of state.

Implements Jermyn et al. (2021, ApJ, 913, 72) third-order Helmholtz free-energy
minimization (`skye.f90`, `skye_ideal.f90`, `skye_coulomb*.f90`, and
`skye_thermodynamics.f90`) using forward-mode automatic differentiation
(`jax.jacfwd`) over `jnp.float64` arrays.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from flexmesa import constants
from flexmesa.eos import (
    _helm_jax_poly,
    _pc_coulomb_jax,
    _skye_coulomb_jax,
    core,
    ideal,
)

_SKYE_MIN_GAMMA_SOLID: float = 170.0
_SKYE_MAX_GAMMA_LIQUID: float = 200.0
_NIST_EV: jax.Array = jnp.asarray(
    core.NIST_IONIZATION_TABLE_EV, dtype=jnp.float64
)


def _skye_fb_and_aux(
    td: jax.Array, *args: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """Return `([f_gas, df_blur], (f_gas, etaele, phase))` at `[T, rho]`."""
    (
        abar,
        zbar,
        mask,
        ya,
        azion,
        acmi,
        ion_off,
        ogata,
        t_lo,
        dth,
        d_lo,
        dd,
        fi_f,
        fi_ef,
    ) = args
    t_v, d_v = td[0], td[1]
    ye = zbar / abar
    din = ye * d_v
    dt2, dd2 = dth * dth, dd * dd
    dti, ddi = 1.0 / dth, 1.0 / dd
    dt2i, dd2i = 1.0 / dt2, 1.0 / dd2
    qt_w0 = _helm_jax_poly.quintic_w0_coord(
        t_v, t_lo, dth, dt2, dti, dt2i, dt2i * dti
    )
    qd_w0 = _helm_jax_poly.quintic_w0_coord(
        din, d_lo, dd, dd2, ddi, dd2i, dd2i * ddi
    )
    ct_w0 = _helm_jax_poly.cubic_w0_coord(t_v, t_lo, dth, dti, dt2i)
    cd_w0 = _helm_jax_poly.cubic_w0_coord(din, d_lo, dd, ddi, dd2i)
    f_ele = _helm_jax_poly.h5(fi_f, qt_w0, qd_w0) * ye
    etaele = _helm_jax_poly.h3(fi_ef, ct_w0, cd_w0)

    f_ion = ideal._ideal_free_energy(t_v, d_v, abar, mask, ya, acmi) + ion_off
    xnefer = constants.AVO * ye * d_v
    rs = jnp.power(
        3.0 / ((4.0 * constants.PI) * xnefer * (constants.RBOHR**3)), 1.0 / 3.0
    )
    game = (constants.QE * constants.QE) / (
        (constants.RBOHR * constants.BOLTZM) * t_v * rs
    )
    fl_sp, fs_sp = jax.vmap(
        _skye_coulomb_jax.species_nonideal,
        in_axes=(None, None, None, 0, 0, None, None),
    )(
        t_v,
        rs,
        game,
        azion,
        acmi,
        jnp.asarray(_SKYE_MIN_GAMMA_SOLID),
        jnp.asarray(_SKYE_MAX_GAMMA_LIQUID),
    )
    sp_ok = mask & (ya > 1.0e-7) & (azion != 0.0)
    df_l0 = jnp.sum(jnp.where(sp_ok, fl_sp * ya, 0.0))
    df_s0 = jnp.sum(jnp.where(sp_ok, fs_sp * ya, 0.0))
    zmean, s_mix, fmix_l, fmix_s = _skye_coulomb_jax.mixing_corrections(
        rs, game, ya, azion, mask, ogata
    )
    fxc = _pc_coulomb_jax._excor7_fxc(0.543 * rs / game, game, th_min=0.007)
    f_pi = fxc * zmean - s_mix
    kt = t_v * (constants.KERG / (abar * constants.AMU))
    df_liq, df_sol = (df_l0 + fmix_l + f_pi) * kt, (df_s0 + fmix_s + f_pi) * kt

    df_coul = jnp.where(df_liq <= df_sol, df_liq, df_sol)
    blur = kt / 100.0
    in_trans = jnp.abs(df_liq - df_sol) <= 20.0 * blur
    ex = jnp.exp(jnp.clip((df_liq - df_sol) / blur, -25.0, 25.0))
    phase = jnp.where(
        df_liq < df_sol - 20.0 * blur,
        0.0,
        jnp.where(df_sol < df_liq - 20.0 * blur, 1.0, ex / (ex + 1.0)),
    )
    df_blur = jnp.where(
        in_trans,
        df_liq * (1.0 - phase) + df_sol * phase - df_coul,
        0.0,
    )
    f_gas = f_ion + df_coul + f_ele
    return jnp.stack([f_gas, df_blur]), (f_gas, etaele, phase)


def _skye_grad_and_aux(
    td: jax.Array, *args: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Return `(grad_fb, (f_gas, grad_f, etaele, phase))`."""
    grad_fb, (f_gas, etaele, phase) = jax.jacfwd(
        _skye_fb_and_aux, argnums=0, has_aux=True
    )(td, *args)
    return grad_fb, (f_gas, grad_fb[0], etaele, phase)


def _skye_state_with_aux(
    t_v: jax.Array, d_v: jax.Array, *args: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Assemble the 26-channel `Skye` state at `(T, rho)`."""
    abar, zbar = args[0], args[1]
    hess_fb, (f_v, grad_f, etaele, phase) = jax.jacfwd(
        _skye_grad_and_aux, argnums=0, has_aux=True
    )(jnp.stack([t_v, d_v]), *args)
    latent_ddlnt = -hess_fb[1, 0, 0] * t_v * t_v
    latent_ddlnrho = -t_v * d_v * hess_fb[1, 0, 1]
    mu = abar / (1.0 + zbar)
    ln_free_e = jnp.log(jnp.maximum(zbar / abar, 1.0e-99))
    res = ideal.pack_free_energy_thermo_jax(
        t_v,
        d_v,
        f_v,
        grad_f[0],
        grad_f[1],
        hess_fb[0, 0, 0],
        hess_fb[0, 0, 1],
        hess_fb[0, 1, 1],
        mu,
        ln_free_e,
        etaele,
        phase,
        latent_ddlnt,
        latent_ddlnrho,
        core.I_FRAC_SKYE,
    )
    return res, res


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
    mass_fraction_limit: float = 1.0e-4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate 26-channel `Skye` EOS at a single zone on a known `HelmCell`.

    Args:
        rho: Mass density $\\rho$ in `g cm^-3` (scalar).
        temp: Temperature $T$ in `K` (scalar).
        abar: Mean atomic weight $\\bar{A}$ (scalar).
        zbar: Mean atomic charge $\\bar{Z}$ (scalar).
        xa: Species mass fractions of shape `(N_species,)`.
        azion: Species atomic charges $Z_i$ of shape `(N_species,)`.
        acmi: Species atomic weights $W_i$ in `amu` of shape `(N_species,)`.
        a_in: Species baryon numbers $A_i$ of shape `(N_species,)`.
        t_lo: Lower temperature bound of the `HELM` cell in `K`.
        dth: Temperature width $\\Delta T$ of the `HELM` cell in `K`.
        d_lo: Lower electron density bound of the `HELM` cell in `g cm^-3`.
        dd: Electron density width $\\Delta(\\rho Y_e)$ in `g cm^-3`.
        fi_f: 36-element biquintic free-energy stencil of shape `(36,)`.
        fi_ef: 16-element bicubic chemical-potential stencil of shape `(16,)`.
        use_ion_offsets: Boolean scalar toggling NIST ionization offsets.
        use_ogata: Boolean scalar (`True` for Ogata, `False` for PC solid mix).
        mass_fraction_limit: Minimum mass fraction included in `Skye`.

    Returns:
        Tuple `(res, d_dlnd, d_dln_t)` of shape `(26,)` each.
    """
    mask = xa > mass_fraction_limit
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
    (d_dt, d_dd), res = jax.jacfwd(
        _skye_state_with_aux, argnums=(0, 1), has_aux=True
    )(
        temp,
        rho,
        abar,
        zbar,
        mask,
        ya,
        azion,
        acmi,
        ion_off,
        use_ogata,
        t_lo,
        dth,
        d_lo,
        dd,
        fi_f,
        fi_ef,
    )
    return (
        cast(jax.Array, res),
        cast(jax.Array, d_dd * rho),
        cast(jax.Array, d_dt * temp),
    )


eval_skye_cell_zones_jit = jax.jit(
    jax.vmap(
        eval_skye_cell_zone_jax,
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
        ),
    )
)
