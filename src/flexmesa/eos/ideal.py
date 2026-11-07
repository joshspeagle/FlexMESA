# SPDX-License-Identifier: LGPL-3.0-or-later
"""Neutral ideal-gas plus Planckian radiation equation of state (`ideal.f90`).

Provides the Tier B dual-track (`backend="auto" | "rust" | "jax"`) Python and
JAX interfaces for `ideal_eos` (`eos/private/ideal.f90`).
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust, constants
from flexmesa.eos import core


def eval_ideal_eos(
    temp: float,
    rho: float,
    abar: float,
    zbar: float,
    xa: np.ndarray,
    chem_id: list[int] | np.ndarray,
) -> core.EosEvalOutput:
    """Evaluate the neutral ideal-gas + radiation EOS from `xa` and `chem_id`.

    Args:
        temp: Temperature $T$ in `K` (scalar).
        rho: Mass density $\\rho$ in `g cm^-3` (scalar).
        abar: Mean baryon number $\\bar{A}$ (dimensionless).
        zbar: Mean charge $\\bar{Z}$ (dimensionless).
        xa: Species mass fractions of shape `(N_species,)`.
        chem_id: 1-based MESA nuclide identifiers of shape `(N_species,)`.

    Returns:
        [`EosEvalOutput`] containing `res`, `d_dlnd`, and `d_dln_t` of shape
        `(26,)`.
    """
    xa_np = np.ascontiguousarray(xa, dtype=np.float64)
    cids = [int(v) for v in chem_id]
    res_np, dlnd_np, dlnt_np = _rust.eos_eval_ideal(
        float(temp),
        float(rho),
        float(abar),
        float(zbar),
        xa_np,
        cids,
    )
    return core.EosEvalOutput(res=res_np, d_dlnd=dlnd_np, d_dln_t=dlnt_np)


def eval_ideal_eos_from_weights(
    temp: float | jax.Array,
    rho: float | jax.Array,
    abar: float,
    zbar: float,
    xa: np.ndarray | jax.Array,
    weights: np.ndarray | jax.Array,
    a: np.ndarray | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> core.EosEvalOutput:
    """Evaluate the neutral ideal-gas + radiation EOS (`ideal.f90`).

    Args:
        temp: Temperature $T$ in `K` (scalar).
        rho: Mass density $\\rho$ in `g cm^-3` (scalar).
        abar: Mean baryon number $\\bar{A}$ (dimensionless).
        zbar: Mean charge $\\bar{Z}$ (dimensionless).
        xa: Species mass fractions of shape `(N_species,)`.
        weights: Atomic weights $W_k$ in `amu` of shape `(N_species,)`.
        a: Baryon numbers $A_k$ of shape `(N_species,)`.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        [`EosEvalOutput`] containing `res`, `d_dlnd`, and `d_dln_t` of shape
        `(26,)`.
    """
    resolved = _dispatch.resolve_backend(backend, temp, rho, xa)
    if resolved == "jax":
        res, d_dlnd, d_dln_t = _eval_ideal_eos_jax(
            jnp.asarray(temp, dtype=jnp.float64),
            jnp.asarray(rho, dtype=jnp.float64),
            jnp.asarray(abar, dtype=jnp.float64),
            jnp.asarray(zbar, dtype=jnp.float64),
            jnp.asarray(xa, dtype=jnp.float64),
            jnp.asarray(weights, dtype=jnp.float64),
            jnp.asarray(a, dtype=jnp.float64),
        )
        return core.EosEvalOutput(res=res, d_dlnd=d_dlnd, d_dln_t=d_dln_t)
    xa_np = np.ascontiguousarray(xa, dtype=np.float64)
    w_np = np.ascontiguousarray(weights, dtype=np.float64)
    a_np = np.ascontiguousarray(a, dtype=np.float64)
    res_np, dlnd_np, dlnt_np = _rust.eos_eval_ideal_from_weights(
        float(temp),
        float(rho),
        float(abar),
        float(zbar),
        xa_np,
        w_np,
        a_np,
    )
    return core.EosEvalOutput(res=res_np, d_dlnd=dlnd_np, d_dln_t=dlnt_np)


def eval_ideal_eos_zones(
    temp: np.ndarray | jax.Array,
    rho: np.ndarray | jax.Array,
    abar: float,
    zbar: float,
    xa: np.ndarray | jax.Array,
    weights: np.ndarray | jax.Array,
    a: np.ndarray | jax.Array,
    *,
    parallel: bool = True,
    backend: _dispatch.Backend = "auto",
) -> core.EosEvalOutput:
    """Evaluate the neutral ideal-gas + radiation EOS across `N_zones` zones.

    Args:
        temp: Zone temperatures $T$ in `K` of shape `(N_zones,)`.
        rho: Zone mass densities $\\rho$ in `g cm^-3` of shape `(N_zones,)`.
        abar: Mean baryon number $\\bar{A}$ (dimensionless).
        zbar: Mean charge $\\bar{Z}$ (dimensionless).
        xa: Species mass fractions of shape `(N_species,)`.
        weights: Atomic weights $W_k$ in `amu` of shape `(N_species,)`.
        a: Baryon numbers $A_k$ of shape `(N_species,)`.
        parallel: Whether to use the Rayon worker pool on the `"rust"` backend.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        [`EosEvalOutput`] with `res`, `d_dlnd`, and `d_dln_t` of shape
        `(N_zones, 26)`.
    """
    resolved = _dispatch.resolve_backend(backend, temp, rho, xa)
    if resolved == "jax":
        t_j = jnp.asarray(temp, dtype=jnp.float64)
        r_j = jnp.asarray(rho, dtype=jnp.float64)
        ab_j = jnp.asarray(abar, dtype=jnp.float64)
        zb_j = jnp.asarray(zbar, dtype=jnp.float64)
        xa_j = jnp.asarray(xa, dtype=jnp.float64)
        w_j = jnp.asarray(weights, dtype=jnp.float64)
        a_j = jnp.asarray(a, dtype=jnp.float64)
        vmapped = jax.vmap(
            _eval_ideal_eos_core,
            in_axes=(0, 0, None, None, None, None, None),
        )
        res_j, dlnd_j, dlnt_j = vmapped(t_j, r_j, ab_j, zb_j, xa_j, w_j, a_j)
        return core.EosEvalOutput(
            res=cast(jax.Array, res_j),
            d_dlnd=cast(jax.Array, dlnd_j),
            d_dln_t=cast(jax.Array, dlnt_j),
        )
    out = _rust.eos_eval_ideal_zones(
        np.ascontiguousarray(temp, dtype=np.float64),
        np.ascontiguousarray(rho, dtype=np.float64),
        float(abar),
        float(zbar),
        np.ascontiguousarray(xa, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
        np.ascontiguousarray(a, dtype=np.float64),
        bool(parallel),
    )
    n = core.NUM_EOS_BASIC_RESULTS
    return core.EosEvalOutput(
        res=out[:, :n],
        d_dlnd=out[:, n : 2 * n],
        d_dln_t=out[:, 2 * n : 3 * n],
    )


def _ideal_free_energy(
    t_in: jax.Array,
    d_in: jax.Array,
    abar: jax.Array,
    mask: jax.Array,
    ya: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Evaluate `compute_F_ideal_ion` in `erg g^-1` (`skye_ideal.f90`)."""
    two_pi_amu = 2.0 * constants.PI * constants.AMU
    sifac = (constants.PLANCK_H**3) / (two_pi_amu * jnp.sqrt(two_pi_amu))
    kt = constants.KERG * t_in
    n_tot = d_in / (constants.AMU * abar)
    n_q = jnp.power(kt, 1.5) / sifac
    n_j = jnp.where(mask, ya * n_tot, 1.0)
    n_qj = jnp.where(mask, n_q * jnp.power(weights, 1.5), 1.0)
    terms = jnp.where(mask, ya * (jnp.log(n_j / n_qj) - 1.0), 0.0)
    return jnp.sum(terms) * kt / (constants.AMU * abar)


def _ideal_free_energy_vec(
    td: jax.Array,
    abar: jax.Array,
    mask: jax.Array,
    ya: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Evaluate `compute_F_ideal_ion` at `td = [T, rho]` (`skye_ideal.f90`)."""
    return _ideal_free_energy(td[0], td[1], abar, mask, ya, weights)


def _ideal_grad_and_val(
    td: jax.Array,
    abar: jax.Array,
    mask: jax.Array,
    ya: jax.Array,
    weights: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Return `(grad_F, (grad_F, F))` at `td = [T, rho]`."""
    f_val, grad_f = jax.value_and_grad(_ideal_free_energy_vec, argnums=0)(
        td, abar, mask, ya, weights
    )
    return grad_f, (grad_f, f_val)


def pack_free_energy_thermo_jax(
    t_v: jax.Array,
    d_v: jax.Array,
    f_v: jax.Array,
    df_dt: jax.Array,
    df_dd: jax.Array,
    d2f_dt2: jax.Array,
    d2f_dtdd: jax.Array,
    d2f_dd2: jax.Array,
    mu: jax.Array,
    ln_free_e: jax.Array,
    etaele: jax.Array,
    phase: jax.Array,
    latent_ddlnt: jax.Array,
    latent_ddlnrho: jax.Array,
    frac_idx: int,
) -> jax.Array:
    """Assemble the 26-channel MESA state from free-energy partials."""
    sgas = -df_dt
    pgas = d_v * d_v * df_dd
    egas = f_v + t_v * sgas
    prad = constants.CRAD * (t_v**4) / 3.0
    erad = constants.CRAD * (t_v**4) / d_v
    srad = 4.0 * constants.CRAD * (t_v**3) / (3.0 * d_v)
    p_tot = prad + pgas
    e_tot = erad + egas
    s_tot = srad + sgas

    dp_dt = 4.0 * prad / t_v + d_v * d_v * d2f_dtdd
    dp_dd = 2.0 * d_v * df_dd + d_v * d_v * d2f_dd2
    de_dt = 4.0 * erad / t_v - t_v * d2f_dt2
    de_dd = -erad / d_v + df_dd - t_v * d2f_dtdd
    ds_dt = 3.0 * srad / t_v - d2f_dt2
    ds_dd = -srad / d_v - d2f_dtdd

    chit = dp_dt * t_v / p_tot
    chid = dp_dd * d_v / p_tot
    gam3 = 1.0 + (p_tot / d_v) * chit / (t_v * de_dt)
    gam1 = chid + (gam3 - 1.0) * chit
    nabad = (gam3 - 1.0) / gam1
    cp = de_dt * gam1 / chid
    z0 = jnp.zeros_like(t_v)
    fracs = [z0 + (1.0 if i == frac_idx else 0.0) for i in range(19, 26)]
    return jnp.stack(
        [
            jnp.log(pgas),
            jnp.log(e_tot),
            jnp.log(s_tot),
            mu + z0,
            ln_free_e + z0,
            etaele + z0,
            nabad,
            chid,
            chit,
            cp,
            de_dt,
            de_dd,
            ds_dt,
            ds_dd,
            gam1,
            gam3,
            phase + z0,
            latent_ddlnt + z0,
            latent_ddlnrho + z0,
            *fracs,
        ]
    )


def _ideal_state_from_td(
    t_v: jax.Array,
    d_v: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    mask: jax.Array,
    ya: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Assemble the 26-channel `ideal_eos` state at `(T, rho)`."""
    td = jnp.stack([t_v, d_v])
    hess_f, (grad_f, f_v) = jax.jacfwd(
        _ideal_grad_and_val, argnums=0, has_aux=True
    )(td, abar, mask, ya, weights)
    z0 = jnp.zeros_like(t_v) + zbar * 0.0
    ln_free_e = jnp.log(1.0e-20 / (constants.AVO * d_v))
    return pack_free_energy_thermo_jax(
        t_v,
        d_v,
        f_v,
        grad_f[0],
        grad_f[1],
        hess_f[0, 0],
        hess_f[0, 1],
        hess_f[1, 1],
        abar + z0,
        ln_free_e,
        z0 - 1.0e99,
        z0,
        z0,
        z0,
        core.I_FRAC_IDEAL,
    )


def _ideal_state_with_aux(
    t_v: jax.Array,
    d_v: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    mask: jax.Array,
    ya: jax.Array,
    weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return `(res, res)` for single-pass `jacfwd(..., has_aux=True)`."""
    res = _ideal_state_from_td(t_v, d_v, abar, zbar, mask, ya, weights)
    return res, res


def _eval_ideal_eos_core(
    temp: jax.Array,
    rho: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xa: jax.Array,
    weights: jax.Array,
    a: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate `ideal_eos` and its log-derivatives via JAX autodiff."""
    mask = (xa > 1.0e-10) & (a > 0.0) & (weights > 0.0)
    sel_x = jnp.where(mask, xa, 0.0)
    sum_x = jnp.sum(sel_x)
    sel_x = sel_x / jnp.where(sum_x > 0.0, sum_x, 1.0)
    a_safe = jnp.where(mask, a, 1.0)
    ya_raw = jnp.where(mask, sel_x / a_safe, 0.0)
    sum_y = jnp.sum(ya_raw)
    ya = ya_raw / jnp.where(sum_y > 0.0, sum_y, 1.0)
    w_safe = jnp.where(mask, weights, 1.0)
    (d_dt, d_dd), res = jax.jacfwd(
        _ideal_state_with_aux, argnums=(0, 1), has_aux=True
    )(temp, rho, abar, zbar, mask, ya, w_safe)
    return res, d_dd * rho, d_dt * temp


_eval_ideal_eos_jit = jax.jit(_eval_ideal_eos_core)


def _eval_ideal_eos_jax(
    temp: jax.Array,
    rho: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xa: jax.Array,
    weights: jax.Array,
    a: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Typed wrapper around JIT-compiled `_eval_ideal_eos_core`."""
    res, d_dlnd, d_dln_t = _eval_ideal_eos_jit(
        temp, rho, abar, zbar, xa, weights, a
    )
    return (
        cast(jax.Array, res),
        cast(jax.Array, d_dlnd),
        cast(jax.Array, d_dln_t),
    )
