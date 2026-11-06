# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Effective charge of $\mathrm{Fe56}$ in $\mathrm{He4}$ (`mod_ionization`).

Implements the Three-Tier Hybrid (`Tier B`) 2D not-a-knot bicubic spline
interpolator (`eval_charge_of_fe56_in_he4`, `eval_charge_of_fe56_in_he4_jax`,
and `eval_charge_of_fe56_in_he4_zones`) over the $105 \times 30$
Thomas-Fermi-Dirac-Hubbard $(\log_{10} n_e, \log_{10} T)$ grid from upstream
MESA `mod_ionization.f90`.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _dispatch, _rust


@functools.lru_cache(maxsize=1)
def _fe56_he4_grid_jax() -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Load and cache the $105 \times 30$ `Fe56He4Table` spline for JAX.

    Returns:
        Tuple `(log_ne, log_t, coeffs)` where `log_ne` has shape `(105,)`,
        `log_t` has shape `(30,)`, and `coeffs` has shape `(105, 30, 4)`.
    """
    lne, lt, c = _rust.py_ion_fe56_he4_grid()
    return (
        jnp.asarray(lne, dtype=jnp.float64),
        jnp.asarray(lt, dtype=jnp.float64),
        jnp.asarray(c, dtype=jnp.float64).reshape(105, 30, 4),
    )


def _axis_basis(
    q: jax.Array, grid: jax.Array
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    r"""Compute 1D cubic spline cell basis and derivative terms along `grid`."""
    idx = jnp.clip(
        jnp.searchsorted(grid, q, side="right") - 1, 0, grid.shape[0] - 2
    )
    g0, g1 = grid[idx], grid[idx + 1]
    h = g1 - g0
    hi = 1.0 / h
    u = (q - g0) * hi
    ui = 1.0 - u
    u2, ui2 = u * u, ui * ui
    c = u * (u2 - 1.0)
    ci = ui * (ui2 - 1.0)
    cd = 3.0 * u2 - 1.0
    cdi = -3.0 * ui2 + 1.0
    return idx, h, hi, u, ui, c, ci, cd, cdi


def _eval_fe56_he4_core(
    log10_ne: jax.Array,
    log10_t: jax.Array,
    xs: jax.Array,
    ys: jax.Array,
    coeffs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Branchless 2D bicubic spline evaluator for $Z(\mathrm{Fe56})$ in JAX.

    Args:
        log10_ne: Query $\log_{10}(n_e / \mathrm{cm^{-3}})$ tensor.
        log10_t: Query $\log_{10}(T / \mathrm{K})$ tensor.
        xs: Grid $\log_{10} n_e$ of shape `(105,)`.
        ys: Grid $\log_{10} T$ of shape `(30,)`.
        coeffs: Bicubic spline array of shape `(105, 30, 4)`.

    Returns:
        Tuple `(z, dz_dlog10_ne, dz_dlog10_t)` of `float64` `jax.Array` tensors.
    """
    x = jnp.asarray(log10_ne, dtype=jnp.float64)
    y = jnp.asarray(log10_t, dtype=jnp.float64)
    ix, hx, hxi, xp, xpi, cx, cxi, cxd, cxdi = _axis_basis(x, xs)
    iy, hy, hyi, yp, ypi, cy, cyi, cyd, cydi = _axis_basis(y, ys)
    hx2, hy2 = hx * hx, hy * hy

    c00, c01 = coeffs[ix, iy], coeffs[ix, iy + 1]
    c10, c11 = coeffs[ix + 1, iy], coeffs[ix + 1, iy + 1]
    y_lin0 = ypi * c00[..., 0] + yp * c01[..., 0]
    y_lin1 = ypi * c10[..., 0] + yp * c11[..., 0]
    y_xx0 = ypi * c00[..., 1] + yp * c01[..., 1]
    y_xx1 = ypi * c10[..., 1] + yp * c11[..., 1]
    y_yy0 = cyi * c00[..., 2] + cy * c01[..., 2]
    y_yy1 = cyi * c10[..., 2] + cy * c11[..., 2]
    y_xxyy0 = cyi * c00[..., 3] + cy * c01[..., 3]
    y_xxyy1 = cyi * c10[..., 3] + cy * c11[..., 3]

    val = (
        xpi * y_lin0
        + xp * y_lin1
        + (hx2 / 6.0) * (cxi * y_xx0 + cx * y_xx1)
        + (hy2 / 6.0) * (xpi * y_yy0 + xp * y_yy1)
        + (hx2 * hy2 / 36.0) * (cxi * y_xxyy0 + cx * y_xxyy1)
    )
    df_dx = (
        hxi * (y_lin1 - y_lin0)
        + (hx / 6.0) * (cxdi * y_xx0 + cxd * y_xx1)
        + (hxi * hy2 / 6.0) * (y_yy1 - y_yy0)
        + (hx * hy2 / 36.0) * (cxdi * y_xxyy0 + cxd * y_xxyy1)
    )
    df_dy = (
        hyi
        * (xpi * (c01[..., 0] - c00[..., 0]) + xp * (c11[..., 0] - c10[..., 0]))
        + (hx2 * hyi / 6.0)
        * (cxi * (c01[..., 1] - c00[..., 1]) + cx * (c11[..., 1] - c10[..., 1]))
        + (hy / 6.0)
        * (
            xpi * (cydi * c00[..., 2] + cyd * c01[..., 2])
            + xp * (cydi * c10[..., 2] + cyd * c11[..., 2])
        )
        + (hx2 * hy / 36.0)
        * (
            cxi * (cydi * c00[..., 3] + cyd * c01[..., 3])
            + cx * (cydi * c10[..., 3] + cyd * c11[..., 3])
        )
    )
    return val, df_dx, df_dy


_eval_fe56_he4_jit = jax.jit(_eval_fe56_he4_core)


def eval_charge_of_fe56_in_he4_jax(
    log10_ne: jax.Array | float,
    log10_t: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Evaluate `(z, dz_dlog10_ne, dz_dlog10_t)` for $\mathrm{Fe56}$ in JAX.

    Args:
        log10_ne: Base-10 log electron density $\log_{10}(n_e /
            \mathrm{cm^{-3}})$ of shape `(*batch,)`.
        log10_t: Base-10 log temperature $\log_{10}(T / \mathrm{K})$ of shape
            `(*batch,)`.

    Returns:
        Tuple `(z, dz_dlog10_ne, dz_dlog10_t)` of `float64` `jax.Array` tensors.
    """
    xs, ys, coeffs = _fe56_he4_grid_jax()
    z, dz_dlne, dz_dlt = _eval_fe56_he4_jit(
        jnp.asarray(log10_ne, dtype=jnp.float64),
        jnp.asarray(log10_t, dtype=jnp.float64),
        xs,
        ys,
        coeffs,
    )
    return z, dz_dlne, dz_dlt


def eval_charge_of_fe56_in_he4(
    log10_ne: float,
    log10_t: float,
    *,
    backend: _dispatch.Backend = "auto",
) -> float:
    r"""Evaluate the effective charge $Z(\mathrm{Fe56})$ in ${}^4\mathrm{He}$.

    Args:
        log10_ne: Base-10 log electron density $\log_{10}(n_e /
            \mathrm{cm^{-3}})$.
        log10_t: Base-10 log temperature $\log_{10}(T / \mathrm{K})$.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Effective ionic charge $Z(\mathrm{Fe56})$ (dimensionless).

    Raises:
        ValueError: If `log10_ne` or `log10_t` is outside the table domain.
    """
    resolved = _dispatch.resolve_backend(backend, log10_ne, log10_t)
    lne_f, lt_f = float(log10_ne), float(log10_t)
    if resolved == "jax":
        xs, ys, _ = _fe56_he4_grid_jax()
        if not (
            float(xs[0]) <= lne_f <= float(xs[-1])
            and float(ys[0]) <= lt_f <= float(ys[-1])
        ):
            raise ValueError(
                f"Coordinates ({lne_f}, {lt_f}) out of Fe56-in-He4 range."
            )
        z_j, _, _ = eval_charge_of_fe56_in_he4_jax(lne_f, lt_f)
        return float(z_j)
    z_r, _, _ = _rust.py_ion_charge_of_fe56_in_he4(lne_f, lt_f)
    return float(z_r)


def eval_charge_of_fe56_in_he4_zones(
    log10_ne: npt.NDArray[np.float64] | jax.Array,
    log10_t: npt.NDArray[np.float64] | jax.Array,
    *,
    parallel: bool = True,
    backend: _dispatch.Backend = "auto",
) -> npt.NDArray[np.float64] | jax.Array:
    r"""Evaluate `[z, dz_dlog10_ne, dz_dlog10_t]` across `n_zones` zones.

    Args:
        log10_ne: 1D array of $\log_{10}(n_e / \mathrm{cm^{-3}})$ of shape
            `(n_zones,)`.
        log10_t: 1D array of $\log_{10}(T / \mathrm{K})$ of shape `(n_zones,)`.
        parallel: When using `"rust"`, evaluate across the persistent 4-thread
            `rayon` worker pool when `n_zones >= 64`.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        2D `float64` array of shape `(n_zones, 3)` with columns
        `[z, dz_dlog10_ne, dz_dlog10_t]`.

    Raises:
        ValueError: If `log10_ne` and `log10_t` are not 1D arrays of equal
            length.
    """
    resolved = _dispatch.resolve_backend(backend, log10_ne, log10_t)
    if resolved == "jax":
        lne_j = jnp.asarray(log10_ne, dtype=jnp.float64)
        lt_j = jnp.asarray(log10_t, dtype=jnp.float64)
        if lne_j.ndim != 1 or lt_j.shape != lne_j.shape:
            raise ValueError(
                "log10_ne and log10_t must be 1D arrays of equal shape."
            )
        z_j, dx_j, dy_j = eval_charge_of_fe56_in_he4_jax(lne_j, lt_j)
        return jnp.stack([z_j, dx_j, dy_j], axis=-1)
    lne_np = _dispatch.as_c_f64(log10_ne)
    lt_np = _dispatch.as_c_f64(log10_t)
    if lne_np.ndim != 1 or lt_np.shape != lne_np.shape:
        raise ValueError(
            "log10_ne and log10_t must be 1D arrays of equal shape."
        )
    return _rust.py_ion_charge_of_fe56_in_he4_zones(
        lne_np, lt_np, bool(parallel)
    )
