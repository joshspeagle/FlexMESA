# SPDX-License-Identifier: LGPL-3.0-or-later
"""Timmes & Swesty (2000) `HELM` equation of state (`eos_helm_eval.f90`).

Provides the Tier B dual-track (`backend="auto" | "rust" | "jax"`) Python and
JAX interfaces for the 26-channel MESA `HELM` equation of state (`helmeos2` and
`do_convert_helm_results`).
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust
from flexmesa.eos import _helm_jax, core

eval_helm_cell_jax = _helm_jax.eval_helm_cell_jax


def _host_lookup_cells(
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
    """Fetch cell bounds and Hermite stencils from the Rust `HelmTable`."""
    orig_shape = np.shape(temp_np)
    lt_1d = np.ascontiguousarray(
        np.log10(np.asarray(temp_np, dtype=np.float64)).reshape(-1)
    )
    dn_1d = np.ascontiguousarray(
        np.asarray(din_np, dtype=np.float64).reshape(-1)
    )
    t_lo, dth, d_lo, dd, fi_f, fi_dpdf, fi_ef, fi_xf = (
        _rust.eos_helm_cells_batch(lt_1d, dn_1d)
    )
    return (
        t_lo.reshape(orig_shape),
        dth.reshape(orig_shape),
        d_lo.reshape(orig_shape),
        dd.reshape(orig_shape),
        fi_f.reshape((*orig_shape, 36)),
        fi_dpdf.reshape((*orig_shape, 16)),
        fi_ef.reshape((*orig_shape, 16)),
        fi_xf.reshape((*orig_shape, 16)),
    )


def _lookup_helm_cell_core(
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
    """Look up `HelmCell` bounds and stencils via host `pure_callback`."""
    shape = temp.shape
    dt = jnp.float64
    spec = (
        jax.ShapeDtypeStruct(shape, dt),
        jax.ShapeDtypeStruct(shape, dt),
        jax.ShapeDtypeStruct(shape, dt),
        jax.ShapeDtypeStruct(shape, dt),
        jax.ShapeDtypeStruct((*shape, 36), dt),
        jax.ShapeDtypeStruct((*shape, 16), dt),
        jax.ShapeDtypeStruct((*shape, 16), dt),
        jax.ShapeDtypeStruct((*shape, 16), dt),
    )
    out = jax.pure_callback(
        _host_lookup_cells, spec, temp, din, vmap_method="expand_dims"
    )
    return cast(
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        out,
    )


_lookup_helm_cell_jvp_fn = jax.custom_jvp(_lookup_helm_cell_core)


def _lookup_helm_cell_jvp(
    primals: tuple[jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array],
) -> tuple[
    tuple[jax.Array, ...],
    tuple[jax.Array, ...],
]:
    """Return zero tangents for piecewise-constant cell bounds and stencils."""
    del tangents
    primal_out = _lookup_helm_cell_core(*primals)
    zero_tan = tuple(jnp.zeros_like(arr) for arr in primal_out)
    return primal_out, zero_tan


_lookup_helm_cell_jvp_fn.defjvp(_lookup_helm_cell_jvp)


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
    """Typed wrapper around `custom_jvp` `_lookup_helm_cell_jvp_fn`."""
    out = _lookup_helm_cell_jvp_fn(temp, din)
    return cast(
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        out,
    )


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
