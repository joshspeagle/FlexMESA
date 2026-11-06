# SPDX-License-Identifier: LGPL-3.0-or-later
"""Quadrature, flux dilution, and grid utilities for ``flexmesa.colors``.

Ports upstream MESA ``colors/private/colors_utils.f90`` and
``colors/private/knn_interp.f90`` into unified Safe Rust (``flexmesa._rust``)
and branchless ``@jax.jit`` (``float64``) kernels.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _dispatch, _rust

type ArrayLike = npt.NDArray[np.float64] | jax.Array

D_10PC_CM: float = 3.0857e19
Z_OVER_X_REF_GS98: float = 2.30057173972e-2

COLOR_COLUMN_NAMES: tuple[str, ...] = tuple(_rust.py_colors_column_names())
JOHNSON_FILTER_NAMES: tuple[str, ...] = tuple(_rust.py_colors_filter_names())


class ColorsBuiltinGrid(NamedTuple):
    """Preloaded ``Kurucz2003all`` + ``Johnson`` grid arrays in JAX ``float64``.

    Attributes:
        teff_grid: Effective temperature axis in ``K`` of shape ``(76,)``.
        logg_grid: Log10 surface gravity axis in CGS of shape ``(11,)``.
        meta_grid: Metallicity axis ``[M/H]`` in dex of shape ``(8,)``.
        wavelengths: Canonical wavelength grid in ``Angstrom`` of shape
            ``(1199,)``.
        slot_map: Flat ``(76 * 11 * 8,)`` index map into ``sed_table``.
        sed_table: Surface flux table of shape ``(n_slots, 1199)`` in
            ``erg s^-1 cm^-2 Angstrom^-1``.
        filters_on_sed_grid: Resampled Johnson filter curves of shape
            ``(7, 1199)``.
        filter_norm_integrals: Photon-counting denominators
            ``int S_lambda lambda dlambda`` of shape ``(7,)``.
        vega_zp: Vega zero-point fluxes of shape ``(7,)``.
        ab_zp: AB zero-point fluxes of shape ``(7,)``.
        st_zp: ST zero-point fluxes of shape ``(7,)``.
    """

    teff_grid: jax.Array
    logg_grid: jax.Array
    meta_grid: jax.Array
    wavelengths: jax.Array
    slot_map: jax.Array
    sed_table: jax.Array
    filters_on_sed_grid: jax.Array
    filter_norm_integrals: jax.Array
    vega_zp: jax.Array
    ab_zp: jax.Array
    st_zp: jax.Array


@functools.lru_cache(maxsize=1)
def get_builtin_colors_grid() -> ColorsBuiltinGrid:
    """Load the built-in ``Kurucz2003all`` + ``Johnson`` grid from Rust.

    Returns:
        Cached :class:`ColorsBuiltinGrid` containing ``jnp.float64`` and
        ``jnp.int64`` device arrays.
    """
    (
        t_g,
        g_g,
        m_g,
        wav,
        s_map,
        s_tbl,
        f_grid,
        f_norm,
        v_zp,
        a_zp,
        s_zp,
    ) = _rust.py_colors_builtin_grid()
    return ColorsBuiltinGrid(
        teff_grid=jnp.asarray(t_g, dtype=jnp.float64),
        logg_grid=jnp.asarray(g_g, dtype=jnp.float64),
        meta_grid=jnp.asarray(m_g, dtype=jnp.float64),
        wavelengths=jnp.asarray(wav, dtype=jnp.float64),
        slot_map=jnp.asarray(s_map, dtype=jnp.int64),
        sed_table=jnp.asarray(s_tbl, dtype=jnp.float64),
        filters_on_sed_grid=jnp.asarray(f_grid, dtype=jnp.float64),
        filter_norm_integrals=jnp.asarray(f_norm, dtype=jnp.float64),
        vega_zp=jnp.asarray(v_zp, dtype=jnp.float64),
        ab_zp=jnp.asarray(a_zp, dtype=jnp.float64),
        st_zp=jnp.asarray(s_zp, dtype=jnp.float64),
    )


def remove_dat(path: str) -> str:
    """Strip the file extension starting at the first ``.`` character.

    Args:
        path: Input file path or filter identifier string.

    Returns:
        Substring prior to the first ``.`` character.
    """
    trimmed = path.strip()
    dot = trimmed.find(".")
    return trimmed[:dot] if dot != -1 else trimmed


@jax.jit
def dilute_flux_jax(
    surface_flux: jax.Array,
    r: jax.Array | float,
    d: jax.Array | float,
) -> jax.Array:
    """Apply geometric dilution ``(r / d)^2`` in JAX ``float64``.

    Args:
        surface_flux: 1D surface flux array of shape ``(n_lambda,)`` in
            ``erg s^-1 cm^-2 Angstrom^-1``.
        r: Stellar photospheric radius in ``cm``.
        d: Observer distance in ``cm``.

    Returns:
        Diluted flux array of shape ``(n_lambda,)`` in
        ``erg s^-1 cm^-2 Angstrom^-1``.
    """
    ratio = jnp.asarray(r, dtype=jnp.float64) / jnp.asarray(
        d, dtype=jnp.float64
    )
    return jnp.asarray(surface_flux, dtype=jnp.float64) * (ratio * ratio)


def dilute_flux(
    surface_flux: ArrayLike,
    r: float | jax.Array,
    d: float | jax.Array = D_10PC_CM,
    *,
    backend: _dispatch.Backend = "auto",
) -> ArrayLike:
    """Apply geometric flux dilution ``(R / d)^2`` to a stellar SED.

    Args:
        surface_flux: 1D surface flux array of shape ``(n_lambda,)`` in
            ``erg s^-1 cm^-2 Angstrom^-1``.
        r: Stellar radius in ``cm``.
        d: Distance in ``cm`` (default ``3.0857e19 cm`` = ``10 pc``).
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Diluted flux array of shape ``(n_lambda,)``.
    """
    resolved = _dispatch.resolve_backend(backend, surface_flux, r, d)
    if resolved == "rust":
        arr = _dispatch.as_c_f64(surface_flux)
        return _rust.py_colors_dilute_flux(arr, float(r), float(d))
    return dilute_flux_jax(jnp.asarray(surface_flux, dtype=jnp.float64), r, d)


@jax.jit
def trapezoidal_integration_jax(x: jax.Array, y: jax.Array) -> jax.Array:
    """Evaluate composite non-uniform trapezoidal rule in JAX ``float64``.

    Args:
        x: 1D monotonic grid array of shape ``(n,)``.
        y: 1D integrand values of shape ``(n,)``.

    Returns:
        Scalar integral ``int y(x) dx``.
    """
    x_f = jnp.asarray(x, dtype=jnp.float64)
    y_f = jnp.asarray(y, dtype=jnp.float64)
    dx = x_f[1:] - x_f[:-1]
    return jnp.sum(0.5 * dx * (y_f[1:] + y_f[:-1]))


def trapezoidal_integration(
    x: ArrayLike,
    y: ArrayLike,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Evaluate composite non-uniform trapezoidal quadrature.

    Args:
        x: 1D grid coordinates of shape ``(n,)``.
        y: 1D integrand values of shape ``(n,)``.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Integrated scalar value.
    """
    resolved = _dispatch.resolve_backend(backend, x, y)
    if resolved == "rust":
        x_arr = _dispatch.as_c_f64(x)
        y_arr = _dispatch.as_c_f64(y)
        return _rust.py_colors_trapezoidal_integration(x_arr, y_arr)
    return trapezoidal_integration_jax(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(y, dtype=jnp.float64),
    )


@jax.jit
def simpson_integration_jax(x: jax.Array, y: jax.Array) -> jax.Array:
    """Evaluate composite non-uniform Simpson quadrature in JAX ``float64``.

    Supports batched ``y`` of shape ``(*batch, n)`` integrated along the last
    axis against 1D ``x`` of shape ``(n,)``. Matches upstream MESA
    ``colors_utils.f90:simpson_integration``, including the trapezoidal tail
    interval when ``x.shape[0]`` is even.

    Args:
        x: 1D grid coordinates of shape ``(n,)``.
        y: Integrand values of shape ``(*batch, n)``.

    Returns:
        Integrated value of shape ``(*batch,)``.
    """
    x_f = jnp.asarray(x, dtype=jnp.float64)
    y_f = jnp.asarray(y, dtype=jnp.float64)
    n = x_f.shape[0]
    n_pairs = (n - 1) // 2
    end = 2 * n_pairs

    x0 = x_f[0:end:2]
    x1 = x_f[1 : end + 1 : 2]
    x2 = x_f[2 : end + 2 : 2]
    y0 = y_f[..., 0:end:2]
    y1 = y_f[..., 1 : end + 1 : 2]
    y2 = y_f[..., 2 : end + 2 : 2]

    h1 = x1 - x0
    h2 = x2 - x1
    h12 = h1 + h2
    w_outer = h12 / 6.0
    c0 = w_outer * (2.0 * h1 - h2) / h1
    c1 = w_outer * (h12 * h12) / (h1 * h2)
    c2 = w_outer * (2.0 * h2 - h1) / h2
    total = jnp.sum(y0 * c0 + y1 * c1 + y2 * c2, axis=-1)

    if n % 2 == 0:
        total = total + 0.5 * (x_f[-1] - x_f[-2]) * (
            y_f[..., -1] + y_f[..., -2]
        )
    return total


def simpson_integration(
    x: ArrayLike,
    y: ArrayLike,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Evaluate composite non-uniform Simpson quadrature.

    Args:
        x: 1D grid coordinates of shape ``(n,)``.
        y: 1D integrand values of shape ``(n,)``.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Integrated scalar value.
    """
    resolved = _dispatch.resolve_backend(backend, x, y)
    if resolved == "rust":
        x_arr = _dispatch.as_c_f64(x)
        y_arr = _dispatch.as_c_f64(y)
        return _rust.py_colors_simpson_integration(x_arr, y_arr)
    return simpson_integration_jax(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(y, dtype=jnp.float64),
    )


@jax.jit
def interpolate_array_jax(
    x_in: jax.Array,
    y_in: jax.Array,
    x_out: jax.Array,
) -> jax.Array:
    """Resample 1D curve ``(x_in, y_in)`` onto ``x_out`` with flat clamping.

    Args:
        x_in: 1D source coordinates of shape ``(n_in,)``.
        y_in: 1D source values of shape ``(n_in,)``.
        x_out: 1D target coordinates of shape ``(n_out,)``.

    Returns:
        Resampled 1D array of shape ``(n_out,)``.
    """
    xi = jnp.asarray(x_in, dtype=jnp.float64)
    yi = jnp.asarray(y_in, dtype=jnp.float64)
    xo = jnp.asarray(x_out, dtype=jnp.float64)
    return jnp.interp(xo, xi, yi, left=yi[0], right=yi[-1])


def interpolate_array(
    x_in: ArrayLike,
    y_in: ArrayLike,
    x_out: ArrayLike,
    *,
    backend: _dispatch.Backend = "auto",
) -> ArrayLike:
    """Resample 1D curve ``(x_in, y_in)`` onto ``x_out`` with flat clamping.

    Args:
        x_in: 1D source coordinates of shape ``(n_in,)``.
        y_in: 1D source values of shape ``(n_in,)``.
        x_out: 1D target coordinates of shape ``(n_out,)``.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Resampled 1D array of shape ``(n_out,)``.
    """
    resolved = _dispatch.resolve_backend(backend, x_in, y_in, x_out)
    if resolved == "rust":
        xi = _dispatch.as_c_f64(x_in)
        yi = _dispatch.as_c_f64(y_in)
        xo = _dispatch.as_c_f64(x_out)
        return _rust.py_colors_interpolate_array(xi, yi, xo)
    return interpolate_array_jax(
        jnp.asarray(x_in, dtype=jnp.float64),
        jnp.asarray(y_in, dtype=jnp.float64),
        jnp.asarray(x_out, dtype=jnp.float64),
    )
