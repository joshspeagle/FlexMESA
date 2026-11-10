# SPDX-License-Identifier: LGPL-3.0-or-later
"""Timmes & Swesty (2000) `HELM` equation of state (`eos_helm_eval.f90`).

Provides the Tier B dual-track (`backend="auto" | "rust" | "jax"`) Python and
native `@jax.jit` interfaces for the 26-channel MESA `HELM` equation of state
(`helmeos2` and `do_convert_helm_results`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust
from flexmesa.eos import _cache_io, _helm_jax, core

eval_helm_cell_jax = _helm_jax.eval_helm_cell_jax


def _lookup_helm_cell_np(
    temp_np: np.ndarray, din_np: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    t_arr, dt_sav, d_arr, dd_sav, vertices = _cache_io.get_helm_np_arrays()
    t_a = np.asarray(temp_np, dtype=np.float64)
    d_a = np.asarray(din_np, dtype=np.float64)
    batch_shape = t_a.shape
    lt = np.log10(np.maximum(t_a, 1.0e-99))
    ldin = np.log10(np.maximum(d_a, 1.0e-99))
    jat0 = np.clip(
        np.floor((lt - _cache_io.HELM_LOGTLO) * 100.0).astype(np.int32),
        0,
        _cache_io.HELM_JMAX - 2,
    )
    iat0 = np.clip(
        np.floor((ldin - _cache_io.HELM_LOGDLO) * 100.0).astype(np.int32),
        0,
        _cache_io.HELM_IMAX - 2,
    )
    corners = np.stack(
        [
            vertices[jat0, iat0],
            vertices[jat0, iat0 + 1],
            vertices[jat0 + 1, iat0],
            vertices[jat0 + 1, iat0 + 1],
        ],
        axis=-1,
    )
    fi_f = corners[..., 0:9, :].reshape((*batch_shape, 36))
    fi_dpdf = corners[..., 9:13, :].reshape((*batch_shape, 16))
    fi_ef = corners[..., 13:17, :].reshape((*batch_shape, 16))
    fi_xf = corners[..., 17:21, :].reshape((*batch_shape, 16))
    return (
        np.asarray(t_arr[jat0], dtype=np.float64),
        np.asarray(dt_sav[jat0], dtype=np.float64),
        np.asarray(d_arr[iat0], dtype=np.float64),
        np.asarray(dd_sav[iat0], dtype=np.float64),
        fi_f,
        fi_dpdf,
        fi_ef,
        fi_xf,
    )


def _lookup_helm_cell_jax(
    temp: jax.Array, din: jax.Array
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Look up `HelmCell` bounds and 4-corner Hermite stencils in JAX."""
    temp_sg = jax.lax.stop_gradient(jnp.asarray(temp, dtype=jnp.float64))
    din_sg = jax.lax.stop_gradient(jnp.asarray(din, dtype=jnp.float64))
    if not isinstance(temp_sg, jax.core.Tracer) and not isinstance(
        din_sg, jax.core.Tracer
    ):
        t_lo, dth, d_lo, dd, fi_f, fi_dpdf, fi_ef, fi_xf = _lookup_helm_cell_np(
            np.asarray(temp_sg), np.asarray(din_sg)
        )
        return (
            jnp.asarray(t_lo, dtype=jnp.float64),
            jnp.asarray(dth, dtype=jnp.float64),
            jnp.asarray(d_lo, dtype=jnp.float64),
            jnp.asarray(dd, dtype=jnp.float64),
            jnp.asarray(fi_f, dtype=jnp.float64),
            jnp.asarray(fi_dpdf, dtype=jnp.float64),
            jnp.asarray(fi_ef, dtype=jnp.float64),
            jnp.asarray(fi_xf, dtype=jnp.float64),
        )
    b_shape = temp_sg.shape
    sc = jax.ShapeDtypeStruct(b_shape, jnp.float64)
    s36 = jax.ShapeDtypeStruct((*b_shape, 36), jnp.float64)
    s16 = jax.ShapeDtypeStruct((*b_shape, 16), jnp.float64)
    t_c, dt_c, d_c, dd_c, f_c, dp_c, ef_c, xf_c = jax.pure_callback(
        _lookup_helm_cell_np,
        (sc, sc, sc, sc, s36, s16, s16, s16),
        temp_sg,
        din_sg,
        vmap_method="expand_dims",
    )
    return t_c, dt_c, d_c, dd_c, f_c, dp_c, ef_c, xf_c


def _eval_helm_jax(
    temp: jax.Array,
    rho: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate `HELM` EOS and derivatives in JAX at `(T, rho, abar, zbar)`."""
    ye = zbar / abar
    temp_c = jnp.clip(temp, 1.0e3, 1.0e13)
    din_raw = ye * rho
    din_c = jnp.clip(din_raw, 1.0e-12, 1.0e15)
    rho_c = jnp.where(din_c != din_raw, din_c / ye, rho)
    t_lo, dth, d_lo, dd, fi_f, fi_dpdf, fi_ef, fi_xf = _lookup_helm_cell_jax(
        temp_c, din_c
    )
    return eval_helm_cell_jax(
        temp_c,
        rho_c,
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


def eval_helm_eos(
    temp: float | jax.Array,
    rho: float | jax.Array,
    abar: float | jax.Array,
    zbar: float | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> tuple[core.EosEvalOutput, np.ndarray | jax.Array, np.ndarray | jax.Array]:
    """Evaluate the 26-channel `HELM` equation of state (`eos_helm_eval.f90`).

    Args:
        temp: Temperature $T$ in `K` (scalar).
        rho: Mass density $\\rho$ in `g cm^-3` (scalar).
        abar: Mean atomic weight $\\bar{A}$ (dimensionless).
        zbar: Mean atomic charge $\\bar{Z}$ (dimensionless).
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Tuple `(out, d_dabar, d_dzbar)` where `out` is an [`EosEvalOutput`]
        containing `(res, d_dlnd, d_dln_t)` of shape `(26,)` and `d_dabar`,
        `d_dzbar` are 26-element composition partial derivative arrays.
    """
    resolved = _dispatch.resolve_backend(backend, temp, rho, abar, zbar)
    if resolved == "jax":
        res, d_dlnd, d_dln_t, d_dabar, d_dzbar = _eval_helm_jax(
            jnp.asarray(temp, dtype=jnp.float64),
            jnp.asarray(rho, dtype=jnp.float64),
            jnp.asarray(abar, dtype=jnp.float64),
            jnp.asarray(zbar, dtype=jnp.float64),
        )
        return (
            core.EosEvalOutput(res=res, d_dlnd=d_dlnd, d_dln_t=d_dln_t),
            d_dabar,
            d_dzbar,
        )
    res_np, dlnd_np, dlnt_np, da_np, dz_np = _rust.eos_eval_helm(
        float(temp), float(rho), float(abar), float(zbar)
    )
    return (
        core.EosEvalOutput(res=res_np, d_dlnd=dlnd_np, d_dln_t=dlnt_np),
        da_np,
        dz_np,
    )


def eval_helm_eos_zones(
    temp: np.ndarray | jax.Array,
    rho: np.ndarray | jax.Array,
    abar: np.ndarray | jax.Array,
    zbar: np.ndarray | jax.Array,
    *,
    parallel: bool = True,
    backend: _dispatch.Backend = "auto",
) -> core.EosEvalOutput:
    """Evaluate the 26-channel `HELM` EOS across `N_zones` stellar zones.

    Args:
        temp: Zone temperatures $T$ in `K` of shape `(N_zones,)`.
        rho: Zone mass densities $\\rho$ in `g cm^-3` of shape `(N_zones,)`.
        abar: Zone mean atomic weights $\\bar{A}$ of shape `(N_zones,)`.
        zbar: Zone mean atomic charges $\\bar{Z}$ of shape `(N_zones,)`.
        parallel: Whether to use the Rayon worker pool on the `"rust"` backend.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        [`EosEvalOutput`] with `res`, `d_dlnd`, and `d_dln_t` of shape
        `(N_zones, 26)`.
    """
    resolved = _dispatch.resolve_backend(backend, temp, rho, abar, zbar)
    if resolved == "jax":
        res, d_dlnd, d_dln_t, _, _ = _eval_helm_jax(
            jnp.asarray(temp, dtype=jnp.float64),
            jnp.asarray(rho, dtype=jnp.float64),
            jnp.asarray(abar, dtype=jnp.float64),
            jnp.asarray(zbar, dtype=jnp.float64),
        )
        return core.EosEvalOutput(res=res, d_dlnd=d_dlnd, d_dln_t=d_dln_t)
    out = _rust.eos_eval_helm_zones(
        np.ascontiguousarray(temp, dtype=np.float64),
        np.ascontiguousarray(rho, dtype=np.float64),
        np.ascontiguousarray(abar, dtype=np.float64),
        np.ascontiguousarray(zbar, dtype=np.float64),
        bool(parallel),
    )
    n = core.NUM_EOS_BASIC_RESULTS
    return core.EosEvalOutput(
        res=out[:, :n],
        d_dlnd=out[:, n : 2 * n],
        d_dln_t=out[:, 2 * n : 3 * n],
    )
