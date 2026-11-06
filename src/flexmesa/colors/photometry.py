# SPDX-License-Identifier: LGPL-3.0-or-later
"""10-column history photometry evaluators for ``flexmesa.colors``.

Ports upstream MESA ``colors/private/colors_history.f90`` and
``colors/public/colors_lib.f90`` into unified Safe Rust (``flexmesa._rust``)
and branchless ``@jax.jit`` (``float64``) evaluators.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _dispatch, _rust
from flexmesa.colors import sed, synthetic, utils

type ArrayLike = npt.NDArray[np.float64] | jax.Array


def _parse_mag_and_method_ids(mag_system: str, method: str) -> tuple[int, int]:
    """Map ``(mag_system, method)`` strings to static integer IDs for JAX."""
    ms = mag_system.strip().upper()
    if ms == "VEGA":
        m_id = 0
    elif ms == "AB":
        m_id = 1
    elif ms == "ST":
        m_id = 2
    else:
        msg = f"Unknown magnitude system: {mag_system!r}"
        raise ValueError(msg)

    mt = method.strip().upper()
    if mt == "HERMITE":
        t_id = 0
    elif mt == "LINEAR":
        t_id = 1
    elif mt == "KNN":
        t_id = 2
    else:
        msg = f"Unknown SED interpolation method: {method!r}"
        raise ValueError(msg)
    return m_id, t_id


def _eval_colors_with_grid(
    teff: jax.Array | float,
    log_g: jax.Array | float,
    metallicity: jax.Array | float,
    r: jax.Array | float,
    d: jax.Array | float,
    grid: utils.ColorsBuiltinGrid,
    mag_system_id: int = 0,
    method_id: int = 0,
) -> jax.Array:
    """Evaluate 10-column ``colors`` history vector given explicit ``grid``."""
    if method_id == 0:
        fluxes = sed._eval_sed_hermite_with_grid(
            teff, log_g, metallicity, r, d, grid
        )
    elif method_id == 1:
        fluxes = sed.eval_sed_linear_jax(teff, log_g, metallicity, r, d)
    else:
        fluxes = sed.eval_sed_knn_jax(teff, log_g, metallicity, r, d)

    w = grid.wavelengths
    cleaned = synthetic.clean_sed_fluxes_jax(w, fluxes)
    bol_flux = utils.simpson_integration_jax(w, cleaned)
    bol_mag = jnp.asarray(
        synthetic.flux_to_magnitude(bol_flux), dtype=jnp.float64
    )
    interp_rad = sed._compute_interp_radius_with_grid(
        grid, teff, log_g, metallicity
    )
    zps = (grid.vega_zp, grid.ab_zp, grid.st_zp)[mag_system_id]

    weighted = (cleaned * grid.filters_on_sed_grid) * w
    int_f = utils.simpson_integration_jax(w, weighted)
    syn_f = int_f / grid.filter_norm_integrals
    mags = -2.5 * jnp.log10(syn_f / zps)

    return jnp.concatenate(
        [jnp.stack([bol_mag, bol_flux, interp_rad], axis=0), mags],
        axis=0,
    )


_eval_colors_single_jit = jax.jit(
    _eval_colors_with_grid, static_argnames=("mag_system_id", "method_id")
)


def eval_colors_jax(
    teff: jax.Array | float,
    log_g: jax.Array | float,
    metallicity: jax.Array | float,
    r: jax.Array | float,
    d: jax.Array | float = utils.D_10PC_CM,
    mag_system_id: int = 0,
    method_id: int = 0,
) -> jax.Array:
    """Evaluate 10-column ``colors`` history vector in ``@jax.jit`` ``float64``.

    Args:
        teff: Effective temperature in ``K``.
        log_g: Log10 surface gravity in ``cm s^-2``.
        metallicity: Metallicity ``[M/H]`` in dex.
        r: Photospheric radius in ``cm``.
        d: Observer distance in ``cm``.
        mag_system_id: ``0`` for Vega, ``1`` for AB, ``2`` for ST.
        method_id: ``0`` for Hermite, ``1`` for Linear, ``2`` for KNN.

    Returns:
        1D array of shape ``(10,)`` containing
        ``[Mag_bol, Flux_bol, Interp_rad, B, I, J, M, R, U, V]``.
    """
    return _eval_colors_single_jit(
        teff,
        log_g,
        metallicity,
        r,
        d,
        utils.get_builtin_colors_grid(),
        mag_system_id,
        method_id,
    )


def eval_colors(
    teff: float | jax.Array,
    log_g: float | jax.Array,
    metallicity: float | jax.Array,
    r: float | jax.Array,
    d: float | jax.Array = utils.D_10PC_CM,
    *,
    mag_system: str = "Vega",
    method: str = "Hermite",
    backend: _dispatch.Backend = "auto",
) -> ArrayLike:
    """Evaluate the 10-column ``colors`` history vector for a stellar model.

    Args:
        teff: Effective temperature in ``K``.
        log_g: Log10 surface gravity in ``cm s^-2``.
        metallicity: Metallicity ``[M/H]`` in dex.
        r: Photospheric radius in ``cm``.
        d: Observer distance in ``cm`` (default ``10 pc``).
        mag_system: Zero-point system (``"Vega"``, ``"AB"``, or ``"ST"``).
        method: SED interpolator (``"Hermite"``, ``"Linear"``, or ``"KNN"``).
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        1D array of shape ``(10,)`` containing
        ``[Mag_bol, Flux_bol, Interp_rad, B, I, J, M, R, U, V]``.
    """
    resolved = _dispatch.resolve_backend(
        backend, teff, log_g, metallicity, r, d
    )
    if resolved == "rust":
        return _rust.py_colors_eval_colors(
            float(teff),
            float(log_g),
            float(metallicity),
            float(r),
            float(d),
            mag_system,
            method,
        )
    m_id, t_id = _parse_mag_and_method_ids(mag_system, method)
    return eval_colors_jax(teff, log_g, metallicity, r, d, m_id, t_id)


@functools.partial(jax.jit, static_argnames=("mag_system_id",))
def _eval_colors_dual_jax(
    teff: jax.Array | float,
    log_g: jax.Array | float,
    metallicity: jax.Array | float,
    r: jax.Array | float,
    d: jax.Array | float,
    mag_system_id: int,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled value and ``(10, 4)`` Jacobian in JAX ``float64``."""
    x_vec = jnp.stack(
        [
            jnp.asarray(teff, dtype=jnp.float64),
            jnp.asarray(log_g, dtype=jnp.float64),
            jnp.asarray(metallicity, dtype=jnp.float64),
            jnp.asarray(r, dtype=jnp.float64),
        ]
    )

    def _fn(vec: jax.Array) -> jax.Array:
        return eval_colors_jax(
            vec[0], vec[1], vec[2], vec[3], d, mag_system_id, 0
        )

    return _fn(x_vec), jax.jacfwd(_fn)(x_vec)


def eval_colors_dual(
    teff: float | jax.Array,
    log_g: float | jax.Array,
    metallicity: float | jax.Array,
    r: float | jax.Array,
    d: float | jax.Array = utils.D_10PC_CM,
    *,
    mag_system: str = "Vega",
    backend: _dispatch.Backend = "auto",
) -> tuple[ArrayLike, ArrayLike]:
    """Evaluate ``eval_colors`` and its ``(10, 4)`` Jacobian.

    Args:
        teff: Effective temperature in ``K``.
        log_g: Log10 surface gravity in ``cm s^-2``.
        metallicity: Metallicity ``[M/H]`` in dex.
        r: Photospheric radius in ``cm``.
        d: Observer distance in ``cm``.
        mag_system: Zero-point system (``"Vega"``, ``"AB"``, or ``"ST"``).
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Tuple ``(values, jacobian)`` with shapes ``(10,)`` and ``(10, 4)``.
    """
    resolved = _dispatch.resolve_backend(
        backend, teff, log_g, metallicity, r, d
    )
    if resolved == "rust":
        return _rust.py_colors_eval_colors_dual(
            float(teff),
            float(log_g),
            float(metallicity),
            float(r),
            float(d),
            mag_system,
        )
    m_id, _ = _parse_mag_and_method_ids(mag_system, "Hermite")
    return _eval_colors_dual_jax(teff, log_g, metallicity, r, d, m_id)


@functools.partial(jax.jit, static_argnames=("mag_system_id", "method_id"))
def _eval_colors_zones_with_grid_jax(
    teff: jax.Array,
    log_g: jax.Array,
    metallicity: jax.Array,
    r: jax.Array,
    d: jax.Array,
    grid: utils.ColorsBuiltinGrid,
    mag_system_id: int,
    method_id: int,
) -> jax.Array:
    """JIT-compiled ``vmap`` multi-zone evaluator with explicit ``grid``."""
    return jax.vmap(
        lambda t_i, g_i, m_i, r_i: _eval_colors_with_grid(
            t_i, g_i, m_i, r_i, d, grid, mag_system_id, method_id
        )
    )(teff, log_g, metallicity, r)


def _eval_colors_zones_jax(
    teff: jax.Array,
    log_g: jax.Array,
    metallicity: jax.Array,
    r: jax.Array,
    d: jax.Array,
    mag_system_id: int,
    method_id: int,
) -> jax.Array:
    """JIT-compiled ``vmap`` multi-zone evaluator in JAX ``float64``."""
    return _eval_colors_zones_with_grid_jax(
        teff,
        log_g,
        metallicity,
        r,
        d,
        utils.get_builtin_colors_grid(),
        mag_system_id,
        method_id,
    )


def eval_colors_zones(
    teff: ArrayLike,
    log_g: ArrayLike,
    metallicity: ArrayLike,
    r: ArrayLike,
    d: float | jax.Array = utils.D_10PC_CM,
    *,
    mag_system: str = "Vega",
    method: str = "Hermite",
    backend: _dispatch.Backend = "auto",
    parallel: bool = True,
) -> ArrayLike:
    """Evaluate 10-column ``colors`` output across ``n_zones`` stellar models.

    Args:
        teff: 1D array of temperatures in ``K`` of shape ``(n_zones,)``.
        log_g: 1D array of ``log10(g)`` in ``cm s^-2`` of shape ``(n_zones,)``.
        metallicity: 1D array of ``[M/H]`` in dex of shape ``(n_zones,)``.
        r: 1D array of radii in ``cm`` of shape ``(n_zones,)``.
        d: Observer distance in ``cm`` (default ``10 pc``).
        mag_system: Zero-point system (``"Vega"``, ``"AB"``, or ``"ST"``).
        method: SED interpolator (``"Hermite"``, ``"Linear"``, or ``"KNN"``).
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).
        parallel: Whether to use the persistent ``rayon`` thread pool in Rust.

    Returns:
        2D array of shape ``(n_zones, 10)``.
    """
    resolved = _dispatch.resolve_backend(
        backend, teff, log_g, metallicity, r, d
    )
    if resolved == "rust":
        return _rust.py_colors_eval_colors_zones(
            _dispatch.as_c_f64(teff),
            _dispatch.as_c_f64(log_g),
            _dispatch.as_c_f64(metallicity),
            _dispatch.as_c_f64(r),
            float(d),
            mag_system,
            method,
            parallel,
        )
    m_id, t_id = _parse_mag_and_method_ids(mag_system, method)
    return _eval_colors_zones_jax(
        jnp.asarray(teff, dtype=jnp.float64),
        jnp.asarray(log_g, dtype=jnp.float64),
        jnp.asarray(metallicity, dtype=jnp.float64),
        jnp.asarray(r, dtype=jnp.float64),
        jnp.asarray(d, dtype=jnp.float64),
        m_id,
        t_id,
    )
