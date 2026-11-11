# SPDX-License-Identifier: LGPL-3.0-or-later
"""3D stellar atmosphere SED interpolators for ``flexmesa.colors``.

Ports upstream MESA ``colors/private/hermite_interp.f90``,
``linear_interp.f90``, and ``knn_interp.f90`` into branchless ``@jax.jit``
(``float64``) and Safe Rust (``flexmesa._rust``) evaluators over the
``Kurucz2003all`` $(T_{\\mathrm{eff}}, \\log g, [\\mathrm{M/H}], \\lambda)$
lookup cube.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _dispatch, _rust
from flexmesa.colors import utils

type ArrayLike = npt.NDArray[np.float64] | jax.Array


def h00(t: jax.Array) -> jax.Array:
    """Evaluate cubic Hermite value basis ``h00(t) = 2 t^3 - 3 t^2 + 1``."""
    return 2.0 * (t * t * t) - 3.0 * (t * t) + 1.0


def h10(t: jax.Array) -> jax.Array:
    """Evaluate cubic Hermite tangent basis ``h10(t) = t^3 - 2 t^2 + t``."""
    return (t * t * t) - 2.0 * (t * t) + t


def h01(t: jax.Array) -> jax.Array:
    """Evaluate cubic Hermite value basis ``h01(t) = -2 t^3 + 3 t^2``."""
    return -2.0 * (t * t * t) + 3.0 * (t * t)


def h11(t: jax.Array) -> jax.Array:
    """Evaluate cubic Hermite tangent basis ``h11(t) = t^3 - t^2``."""
    return (t * t * t) - (t * t)


def _find_interval_jax(
    grid: jax.Array,
    val: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Locate bracketing index ``i`` and fractional coordinate ``t`` in JAX."""
    n = grid.shape[0]
    v = jnp.asarray(val, dtype=jnp.float64)
    idx = jnp.clip(jnp.searchsorted(grid, v, side="right") - 1, 0, n - 2)
    x0 = grid[idx]
    x1 = grid[idx + 1]
    t = jnp.where(
        v < grid[0],
        0.0,
        jnp.where(v > grid[-1], 1.0, (v - x0) / (x1 - x0)),
    )
    return idx, t


def _sed_at_jax(
    grid: utils.ColorsBuiltinGrid,
    it: jax.Array,
    ig: jax.Array,
    im: jax.Array,
) -> jax.Array:
    """Fetch 1D SED row at ``(it, ig, im)`` from ``grid.sed_table``."""
    ng = grid.logg_grid.shape[0]
    nm = grid.meta_grid.shape[0]
    return grid.sed_table[grid.slot_map[(it * ng + ig) * nm + im]]


def _axis_d2(ax: jax.Array, v: jax.Array) -> jax.Array:
    """Compute squared normalized 1D distance to the nearest grid point."""
    span = jnp.maximum(ax[-1] - ax[0], 1.0e-12)
    d = jnp.min(jnp.abs(ax - v)) / span
    return d * d


def _compute_interp_radius_with_grid(
    grid: utils.ColorsBuiltinGrid,
    teff: jax.Array | float,
    log_g: jax.Array | float,
    metallicity: jax.Array | float,
) -> jax.Array:
    """Compute normalized interpolation radius given explicit ``grid``."""
    t_v = jax.lax.stop_gradient(jnp.asarray(teff, dtype=jnp.float64))
    g_v = jax.lax.stop_gradient(jnp.asarray(log_g, dtype=jnp.float64))
    m_v = jax.lax.stop_gradient(jnp.asarray(metallicity, dtype=jnp.float64))
    return jnp.sqrt(
        _axis_d2(grid.teff_grid, t_v)
        + _axis_d2(grid.logg_grid, g_v)
        + _axis_d2(grid.meta_grid, m_v)
    )


@jax.jit
def compute_interp_radius_jax(
    teff: jax.Array | float,
    log_g: jax.Array | float,
    metallicity: jax.Array | float,
) -> jax.Array:
    """Compute normalized Euclidean distance to the nearest Kurucz grid node.

    Args:
        teff: Effective temperature in ``K``.
        log_g: Log10 surface gravity in ``cm s^-2``.
        metallicity: Metallicity ``[M/H]`` in dex.

    Returns:
        Scalar normalized interpolation radius ``Interp_rad``.
    """
    return _compute_interp_radius_with_grid(
        utils.get_builtin_colors_grid(), teff, log_g, metallicity
    )


def _axis_stencil_jax(
    ax: jax.Array,
    idx0: jax.Array,
) -> tuple[
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
]:
    """Precompute ``((i0, i1), (im0, ip1), (s0, s1))`` stencil arrays."""
    n = ax.shape[0]
    dx_c = ax[idx0 + 1] - ax[idx0]
    i0 = idx0
    i1 = idx0 + 1
    ip0, im0 = jnp.minimum(i0 + 1, n - 1), jnp.maximum(i0 - 1, 0)
    ip1, im1 = jnp.minimum(i1 + 1, n - 1), jnp.maximum(i1 - 1, 0)
    den0 = ax[ip0] - ax[im0]
    den1 = ax[ip1] - ax[im1]
    s0 = jnp.where(jnp.abs(den0) > 0.0, dx_c / den0, 0.0)
    s1 = jnp.where(jnp.abs(den1) > 0.0, dx_c / den1, 0.0)
    return (i0, i1), (im0, ip1), (s0, s1)


def _eval_sed_hermite_with_grid(
    teff: jax.Array | float,
    log_g: jax.Array | float,
    metallicity: jax.Array | float,
    r: jax.Array | float,
    d: jax.Array | float,
    grid: utils.ColorsBuiltinGrid,
) -> jax.Array:
    """Evaluate 3D cubic Hermite SED using an explicit ``ColorsBuiltinGrid``."""
    nlam = grid.wavelengths.shape[0]

    i0, tx = _find_interval_jax(
        grid.teff_grid, jnp.asarray(teff, dtype=jnp.float64)
    )
    j0, ty = _find_interval_jax(
        grid.logg_grid, jnp.asarray(log_g, dtype=jnp.float64)
    )
    k0, tz = _find_interval_jax(
        grid.meta_grid, jnp.asarray(metallicity, dtype=jnp.float64)
    )

    hx, hdx = (h00(tx), h01(tx)), (h10(tx), h11(tx))
    hy, hdy = (h00(ty), h01(ty)), (h10(ty), h11(ty))
    hz, hdz = (h00(tz), h01(tz)), (h10(tz), h11(tz))

    idx_i, ext_i, sx_i = _axis_stencil_jax(grid.teff_grid, i0)
    idx_j, ext_j, sy_j = _axis_stencil_jax(grid.logg_grid, j0)
    idx_k, ext_k, sz_k = _axis_stencil_jax(grid.meta_grid, k0)

    gx = (hdx[0] * sx_i[0], hdx[1] * sx_i[1])
    gy = (hdy[0] * sy_j[0], hdy[1] * sy_j[1])
    gz = (hdz[0] * sz_k[0], hdz[1] * sz_k[1])
    sgn = (-1.0, 1.0)

    result = jnp.zeros(nlam, dtype=jnp.float64)
    for ii in range(2):
        i, i_ext = idx_i[ii], ext_i[ii]
        for jj in range(2):
            j, j_ext = idx_j[jj], ext_j[jj]
            for kk in range(2):
                k, k_ext = idx_k[kk], ext_k[kk]
                w_corner = (
                    hx[ii] * hy[jj] * hz[kk]
                    + sgn[ii] * gx[1 - ii] * hy[jj] * hz[kk]
                    + sgn[jj] * hx[ii] * gy[1 - jj] * hz[kk]
                    + sgn[kk] * hx[ii] * hy[jj] * gz[1 - kk]
                )
                w_x = sgn[ii] * gx[ii] * hy[jj] * hz[kk]
                w_y = sgn[jj] * hx[ii] * gy[jj] * hz[kk]
                w_z = sgn[kk] * hx[ii] * hy[jj] * gz[kk]

                result = (
                    result
                    + w_corner * _sed_at_jax(grid, i, j, k)
                    + w_x * _sed_at_jax(grid, i_ext, j, k)
                    + w_y * _sed_at_jax(grid, i, j_ext, k)
                    + w_z * _sed_at_jax(grid, i, j, k_ext)
                )

    return cast(jax.Array, utils.dilute_flux_jax(result, r, d))


_eval_sed_hermite_jit = jax.jit(_eval_sed_hermite_with_grid)


def eval_sed_hermite_jax(
    teff: jax.Array | float,
    log_g: jax.Array | float,
    metallicity: jax.Array | float,
    r: jax.Array | float,
    d: jax.Array | float = utils.D_10PC_CM,
) -> jax.Array:
    """Evaluate 3D cubic Hermite SED interpolation in JAX ``float64``.

    Args:
        teff: Effective temperature in ``K``.
        log_g: Log10 surface gravity in ``cm s^-2``.
        metallicity: Metallicity ``[M/H]`` in dex.
        r: Photospheric radius in ``cm``.
        d: Observer distance in ``cm``.

    Returns:
        Diluted 1D SED flux array of shape ``(1199,)`` in
        ``erg s^-1 cm^-2 Angstrom^-1``.
    """
    return cast(
        jax.Array,
        _eval_sed_hermite_jit(
            teff, log_g, metallicity, r, d, utils.get_builtin_colors_grid()
        ),
    )


@jax.jit
def eval_sed_linear_jax(
    teff: jax.Array | float,
    log_g: jax.Array | float,
    metallicity: jax.Array | float,
    r: jax.Array | float,
    d: jax.Array | float = utils.D_10PC_CM,
) -> jax.Array:
    """Evaluate 3D log-fallback trilinear SED interpolation in JAX ``float64``.

    Matches upstream MESA ``linear_interp.f90:construct_sed_linear`` by
    clamping all 8 cell corners to ``1.0e-10`` before linear and log blending.

    Args:
        teff: Effective temperature in ``K``.
        log_g: Log10 surface gravity in ``cm s^-2``.
        metallicity: Metallicity ``[M/H]`` in dex.
        r: Photospheric radius in ``cm``.
        d: Observer distance in ``cm``.

    Returns:
        Diluted 1D SED flux array of shape ``(1199,)``.
    """
    grid = utils.get_builtin_colors_grid()
    i0, tx = _find_interval_jax(
        grid.teff_grid, jnp.asarray(teff, dtype=jnp.float64)
    )
    j0, ty = _find_interval_jax(
        grid.logg_grid, jnp.asarray(log_g, dtype=jnp.float64)
    )
    k0, tz = _find_interval_jax(
        grid.meta_grid, jnp.asarray(metallicity, dtype=jnp.float64)
    )
    wx, wy, wz = (1.0 - tx, tx), (1.0 - ty, ty), (1.0 - tz, tz)

    corners = []
    weights = []
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                corners.append(_sed_at_jax(grid, i0 + ii, j0 + jj, k0 + kk))
                weights.append(wx[ii] * wy[jj] * wz[kk])

    c_floored = jnp.maximum(jnp.stack(corners, axis=0), 1.0e-10)
    w_vec = jnp.stack(weights, axis=0)[:, None]
    lin_interp = jnp.sum(w_vec * c_floored, axis=0)
    log_interp = jnp.exp(jnp.sum(w_vec * jnp.log(c_floored), axis=0))
    result = jnp.where(lin_interp > 1.0e-10, log_interp, lin_interp)
    return cast(jax.Array, utils.dilute_flux_jax(result, r, d))


@jax.jit
def eval_sed_knn_jax(
    teff: jax.Array | float,
    log_g: jax.Array | float,
    metallicity: jax.Array | float,
    r: jax.Array | float,
    d: jax.Array | float = utils.D_10PC_CM,
) -> jax.Array:
    """Evaluate 4-nearest-neighbor inverse-distance SED interpolation in JAX.

    Searches the full ``(76, 11, 8)`` normalized 3D Kurucz grid to match
    upstream MESA ``knn_interp.f90:find_k_nearest_neighbors`` and
    ``interpolate_knn_sed``.

    Args:
        teff: Effective temperature in ``K``.
        log_g: Log10 surface gravity in ``cm s^-2``.
        metallicity: Metallicity ``[M/H]`` in dex.
        r: Photospheric radius in ``cm``.
        d: Observer distance in ``cm``.

    Returns:
        Diluted 1D SED flux array of shape ``(1199,)``.
    """
    grid = utils.get_builtin_colors_grid()
    t_v = jnp.asarray(teff, dtype=jnp.float64)
    g_v = jnp.asarray(log_g, dtype=jnp.float64)
    m_v = jnp.asarray(metallicity, dtype=jnp.float64)

    t_inv = 1.0 / jnp.maximum(grid.teff_grid[-1] - grid.teff_grid[0], 1.0e-12)
    g_inv = 1.0 / jnp.maximum(grid.logg_grid[-1] - grid.logg_grid[0], 1.0e-12)
    m_inv = 1.0 / jnp.maximum(grid.meta_grid[-1] - grid.meta_grid[0], 1.0e-12)

    nt_v = (t_v - grid.teff_grid[0]) * t_inv
    ng_v = (g_v - grid.logg_grid[0]) * g_inv
    nm_v = (m_v - grid.meta_grid[0]) * m_inv

    dt2 = ((grid.teff_grid - grid.teff_grid[0]) * t_inv - nt_v) ** 2
    dg2 = ((grid.logg_grid - grid.logg_grid[0]) * g_inv - ng_v) ** 2
    dm2 = ((grid.meta_grid - grid.meta_grid[0]) * m_inv - nm_v) ** 2
    d2_flat = (
        dt2[:, None, None] + dg2[None, :, None] + dm2[None, None, :]
    ).reshape(-1)
    order = jnp.argsort(d2_flat, stable=True)[:4]
    top_d = jnp.maximum(jnp.sqrt(d2_flat[order]), 1.0e-10)
    top_s = grid.sed_table[grid.slot_map[order]]
    inv_w = 1.0 / top_d
    w_norm = (inv_w / jnp.sum(inv_w))[:, None]
    result = jnp.sum(w_norm * top_s, axis=0)
    return cast(jax.Array, utils.dilute_flux_jax(result, r, d))


def eval_sed(
    teff: float | jax.Array,
    log_g: float | jax.Array,
    metallicity: float | jax.Array,
    r: float | jax.Array,
    d: float | jax.Array = utils.D_10PC_CM,
    *,
    method: str = "Hermite",
    backend: _dispatch.Backend = "auto",
) -> tuple[ArrayLike, ArrayLike]:
    """Evaluate the diluted stellar SED ``(wavelengths, fluxes)``.

    Args:
        teff: Effective temperature in ``K``.
        log_g: Log10 surface gravity in ``cm s^-2``.
        metallicity: Metallicity ``[M/H]`` in dex.
        r: Photospheric radius in ``cm``.
        d: Observer distance in ``cm`` (default ``10 pc``).
        method: SED method (``"Hermite"``, ``"Linear"``, or ``"KNN"``).
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Tuple ``(wavelengths, fluxes)`` of 1D arrays of shape ``(1199,)``.

    Raises:
        ValueError: If ``method`` is not recognized.
    """
    resolved = _dispatch.resolve_backend(
        backend, teff, log_g, metallicity, r, d
    )
    if resolved == "rust":
        return _rust.py_colors_eval_sed(
            float(teff),
            float(log_g),
            float(metallicity),
            float(r),
            float(d),
            method,
        )
    grid = utils.get_builtin_colors_grid()
    m_upper = method.strip().upper()
    if m_upper == "HERMITE":
        flux = eval_sed_hermite_jax(teff, log_g, metallicity, r, d)
    elif m_upper == "LINEAR":
        flux = eval_sed_linear_jax(teff, log_g, metallicity, r, d)
    elif m_upper == "KNN":
        flux = eval_sed_knn_jax(teff, log_g, metallicity, r, d)
    else:
        msg = f"Unknown SED interpolation method: {method!r}"
        raise ValueError(msg)
    return grid.wavelengths, flux
