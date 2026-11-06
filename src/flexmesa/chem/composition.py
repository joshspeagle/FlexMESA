# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Differentiable composition moments and multi-zone kernels (`chem`).

Implements the Three-Tier Hybrid (`Tier B`) composition moment evaluators
(`get_composition_info`, `basic_composition_info`, `composition_info`,
`get_composition_info_jax`, and `get_composition_info_zones`), routing between
Safe Rust (`flexmesa._rust` + persistent 4-thread `rayon` worker pool) and
the branchless `float64` `@jax.jit` kernel `get_composition_info_jax`,
compatible with `jax.jit`, `jax.vmap`, `jax.grad`, and `jax.jacrev`.

Index and Memory-Order Contract:
    - Species lists (`chem_ids`) accept either 1-based MESA integer `chem_id`s
      or trimmed isotope names (such as ``"h1"``, ``"c12"``, ``"fe56"``).
    - Single-zone abundance vectors `x` have shape `(n_species,)` in `float64`.
    - Multi-zone abundance arrays `x_zones` have row-major C-contiguous shape
      `(n_zones, n_species)` in `float64`.
    - `get_composition_info_zones` returns a 2D array of shape `(n_zones, 10)`
      with columns:
      `[xh, xhe, xz, abar, zbar, z2bar, z53bar, ye, mass_correction, sumx]`.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _dispatch, _rust
from flexmesa.chem import catalog as _cat


@dataclasses.dataclass(frozen=True)
class CompositionInfo:
    r"""Single-zone or batched composition moments and analytical partials.

    Attributes:
        xh: Hydrogen mass fraction $X_\mathrm{H} = \sum_{Z_i = 1} X_i$
            (dimensionless).
        xhe: Helium mass fraction $X_\mathrm{He} = \sum_{Z_i = 2} X_i$
            (dimensionless).
        xz: Metal mass fraction $Z = \max(0, \min(1, 1 - (X_\mathrm{H} +
            X_\mathrm{He})))$ (dimensionless).
        abar: Mean nucleon number $\bar{A} = (\sum_i X_i) / (\sum_i Y_i)$
            (dimensionless).
        zbar: Mean proton number $\bar{Z} = \bar{A} \sum_i Y_i Z_i$
            (dimensionless).
        z2bar: Mean squared charge $\overline{Z^2} = \bar{A} \sum_i Y_i Z_i^2$
            (dimensionless).
        z53bar: Mean $5/3$-power charge
            $\overline{Z^{5/3}} = \bar{A} \sum_i Y_i Z_i^{5/3}$
            (dimensionless).
        ye: Electron-to-baryon ratio $Y_e = \bar{Z} / \bar{A}$ (dimensionless).
        mass_correction: Atomic mass correction $\sum_i Y_i W_i$
            (dimensionless).
        sumx: Sum of baryon mass fractions $\sum_i X_i$ (dimensionless).
        dabar_dx: Partial derivatives $\partial \bar{A} / \partial X_i$ of
            shape `(n_species,)` (or `(0,)` when `skip_partials=True`).
        dzbar_dx: Partial derivatives $\partial \bar{Z} / \partial X_i$ of
            shape `(n_species,)` (or `(0,)` when `skip_partials=True`).
        dmc_dx: Partial derivatives
            $\partial (\text{mass\_correction}) / \partial X_i$ of shape
            `(n_species,)` (or `(0,)` when `skip_partials=True`).
    """

    xh: float
    xhe: float
    xz: float
    abar: float
    zbar: float
    z2bar: float
    z53bar: float
    ye: float
    mass_correction: float
    sumx: float
    dabar_dx: npt.NDArray[np.float64]
    dzbar_dx: npt.NDArray[np.float64]
    dmc_dx: npt.NDArray[np.float64]


@functools.lru_cache(maxsize=64)
def _species_props_jax(
    cids: tuple[int, ...],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Cache `(z_arr, a_arr, w_arr, z53_arr)` JAX arrays for a species tuple.

    Args:
        cids: Tuple of 1-based integer `chem_id` identifiers.

    Returns:
        Cached tuple `(z_arr, a_arr, w_arr, z53_arr)` of 1D `float64`
        `jax.Array` tensors.
    """
    _, z_vals, a_vals, w_vals, z53_vals = _rust.py_chem_species_props(
        list(cids)
    )
    return (
        jnp.asarray(z_vals, dtype=jnp.float64),
        jnp.asarray(a_vals, dtype=jnp.float64),
        jnp.asarray(w_vals, dtype=jnp.float64),
        jnp.asarray(z53_vals, dtype=jnp.float64),
    )


def _composition_moments_only_core(
    x: jax.Array,
    z_arr: jax.Array,
    a_arr: jax.Array,
    w_arr: jax.Array,
    z53_arr: jax.Array,
) -> jax.Array:
    r"""JIT-compiled `float64` kernel for the 10 composition moments only.

    Args:
        x: Mass-fraction array of shape `(*batch, n_species)` in `float64`.
        z_arr: Proton numbers $Z_i$ of shape `(n_species,)` in `float64`.
        a_arr: Baryon numbers $A_i$ of shape `(n_species,)` in `float64`.
        w_arr: Atomic weights $W_i$ of shape `(n_species,)` in `float64`.
        z53_arr: Coulomb powers $Z_i^{5/3}$ of shape `(n_species,)` in
            `float64`.

    Returns:
        Moments tensor of shape `(*batch, 10)` with columns `[xh, xhe, xz,
        abar, zbar, z2bar, z53bar, ye, mass_correction, sumx]`.
    """
    x_f64 = jnp.asarray(x, dtype=jnp.float64)
    y = x_f64 / a_arr
    xh = jnp.sum(jnp.where(jnp.equal(z_arr, 1.0), x_f64, 0.0), axis=-1)
    xhe = jnp.sum(jnp.where(jnp.equal(z_arr, 2.0), x_f64, 0.0), axis=-1)
    xz = jnp.clip(1.0 - (xh + xhe), 0.0, 1.0)
    sumx = jnp.sum(x_f64, axis=-1)
    sumy = jnp.sum(y, axis=-1)
    abar = sumx / sumy
    zbar = jnp.sum(y * z_arr, axis=-1) * abar
    z2bar = jnp.sum(y * (z_arr * z_arr), axis=-1) * abar
    z53bar = jnp.sum(y * z53_arr, axis=-1) * abar
    ye = zbar / abar
    mc = jnp.sum(y * w_arr, axis=-1)
    return jnp.stack(
        [xh, xhe, xz, abar, zbar, z2bar, z53bar, ye, mc, sumx], axis=-1
    )


def _composition_moments_jax_core(
    x: jax.Array,
    z_arr: jax.Array,
    a_arr: jax.Array,
    w_arr: jax.Array,
    z53_arr: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""JIT-compiled `float64` kernel for composition moments and partials.

    Args:
        x: Mass-fraction array of shape `(*batch, n_species)` in `float64`.
        z_arr: Proton numbers $Z_i$ of shape `(n_species,)` in `float64`.
        a_arr: Baryon numbers $A_i$ of shape `(n_species,)` in `float64`.
        w_arr: Atomic weights $W_i$ of shape `(n_species,)` in `float64`.
        z53_arr: Coulomb powers $Z_i^{5/3}$ of shape `(n_species,)` in
            `float64`.

    Returns:
        Tuple `(moments, dabar_dx, dzbar_dx, dmc_dx)` where `moments` has shape
        `(*batch, 10)` and each derivative array has shape
        `(*batch, n_species)`.
    """
    moments = _composition_moments_only_core(x, z_arr, a_arr, w_arr, z53_arr)
    abar_exp = jnp.expand_dims(moments[..., 3], axis=-1)
    zbar_exp = jnp.expand_dims(moments[..., 4], axis=-1)
    mc_exp = jnp.expand_dims(moments[..., 8], axis=-1)
    sumx_exp = jnp.expand_dims(moments[..., 9], axis=-1)

    dabar_dx = abar_exp * (a_arr - abar_exp) / a_arr / sumx_exp
    dzbar_dx = abar_exp * (z_arr - zbar_exp) / a_arr / sumx_exp
    dmc_dx = (w_arr / a_arr) - mc_exp
    return moments, dabar_dx, dzbar_dx, dmc_dx


_composition_moments_only_jit = jax.jit(_composition_moments_only_core)
_composition_moments_jit = jax.jit(_composition_moments_jax_core)


def get_composition_info_jax(
    chem_ids: Sequence[int | str],
    x: jax.Array | npt.NDArray[np.float64],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Evaluate differentiable composition moments and partials in JAX.

    Args:
        chem_ids: Sequence of length `n_species` containing 1-based `chem_id`
            integers or isotope names (e.g. ``["c12", "fe56"]``).
        x: Mass fractions $X_i$ of shape `(n_species,)` or `(*batch, n_species)`
            in `float64` (dimensionless).

    Returns:
        Tuple `(moments, dabar_dx, dzbar_dx, dmc_dx)` of `jax.Array` tensors in
        `float64`, where `moments` has shape `(*batch, 10)` (`[xh, xhe, xz,
        abar, zbar, z2bar, z53bar, ye, mass_correction, sumx]`) and each
        partial derivative array has shape `(*batch, n_species)`.

    Raises:
        ValueError: If `chem_ids` is empty, any species is unknown, or
            `x.shape[-1] != len(chem_ids)`.
    """
    cids = _cat._resolve_chem_ids(chem_ids)
    x_f64 = jnp.asarray(x, dtype=jnp.float64)
    if x_f64.ndim < 1 or x_f64.shape[-1] != len(cids):
        raise ValueError(
            f"x last axis ({x_f64.shape}) must equal len(chem_ids) "
            f"({len(cids)})."
        )
    z_arr, a_arr, w_arr, z53_arr = _species_props_jax(cids)
    m, da, dz, dmc = _composition_moments_jit(
        x_f64, z_arr, a_arr, w_arr, z53_arr
    )
    return m, da, dz, dmc


def get_composition_info(
    chem_ids: Sequence[int | str],
    x: Sequence[float] | npt.NDArray[np.float64] | jax.Array,
    *,
    skip_partials: bool = False,
    backend: _dispatch.Backend = "auto",
) -> CompositionInfo:
    r"""Compute composition moments and optional partial derivatives.

    Args:
        chem_ids: Sequence of 1-based `chem_id` integers or isotope names.
        x: Mass-fraction vector $X_i$ of shape `(n_species,)` (dimensionless).
        skip_partials: If `True`, skip computing `dabar_dx`, `dzbar_dx`, and
            `dmc_dx` (returning empty `(0,)` arrays).
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        `CompositionInfo` dataclass with all 10 moments and 3 derivative arrays.

    Raises:
        ValueError: If `chem_ids` is empty, `x.shape != (len(chem_ids),)`, or
            any species or `backend` is invalid.
    """
    resolved = _dispatch.resolve_backend(backend, x)
    cids = _cat._resolve_chem_ids(chem_ids)
    if resolved == "jax":
        x_jax = jnp.asarray(x, dtype=jnp.float64)
        if x_jax.shape != (len(cids),):
            raise ValueError(f"x shape ({x_jax.shape}) must be ({len(cids)},).")
        z_arr, a_arr, w_arr, z53_arr = _species_props_jax(cids)
        if skip_partials:
            m_j = _composition_moments_only_jit(
                x_jax, z_arr, a_arr, w_arr, z53_arr
            )
            m_seq = np.asarray(m_j, dtype=np.float64)
            empty = np.empty(0, dtype=np.float64)
            da_arr, dz_arr, dmc_arr = empty, empty, empty
        else:
            m_j, da_j, dz_j, dmc_j = _composition_moments_jit(
                x_jax, z_arr, a_arr, w_arr, z53_arr
            )
            m_seq = np.asarray(m_j, dtype=np.float64)
            da_arr, dz_arr, dmc_arr = (
                np.asarray(da_j, dtype=np.float64),
                np.asarray(dz_j, dtype=np.float64),
                np.asarray(dmc_j, dtype=np.float64),
            )
    else:
        x_np = np.asarray(x, dtype=np.float64)
        if x_np.shape != (len(cids),):
            raise ValueError(f"x shape ({x_np.shape}) must be ({len(cids)},).")
        m_tup, da_vals, dz_vals, dmc_vals = _rust.py_chem_get_composition_info(
            list(cids), [float(v) for v in x_np], bool(skip_partials)
        )
        m_seq = np.asarray(m_tup, dtype=np.float64)
        da_arr, dz_arr, dmc_arr = (
            np.asarray(da_vals, dtype=np.float64),
            np.asarray(dz_vals, dtype=np.float64),
            np.asarray(dmc_vals, dtype=np.float64),
        )

    return CompositionInfo(
        xh=float(m_seq[0]),
        xhe=float(m_seq[1]),
        xz=float(m_seq[2]),
        abar=float(m_seq[3]),
        zbar=float(m_seq[4]),
        z2bar=float(m_seq[5]),
        z53bar=float(m_seq[6]),
        ye=float(m_seq[7]),
        mass_correction=float(m_seq[8]),
        sumx=float(m_seq[9]),
        dabar_dx=da_arr,
        dzbar_dx=dz_arr,
        dmc_dx=dmc_arr,
    )


def basic_composition_info(
    chem_ids: Sequence[int | str],
    x: Sequence[float] | npt.NDArray[np.float64] | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> CompositionInfo:
    r"""Compute composition moments without partial derivatives.

    Args:
        chem_ids: Sequence of 1-based `chem_id` integers or isotope names.
        x: Mass-fraction vector $X_i$ of shape `(n_species,)`.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        `CompositionInfo` with empty `(0,)` partial derivative arrays.

    Raises:
        ValueError: If `chem_ids` or `x` has invalid shape or species.
    """
    return get_composition_info(
        chem_ids, x, skip_partials=True, backend=backend
    )


def composition_info(
    chem_ids: Sequence[int | str],
    x: Sequence[float] | npt.NDArray[np.float64] | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> CompositionInfo:
    r"""Compute composition moments and analytical partial derivatives.

    Args:
        chem_ids: Sequence of 1-based `chem_id` integers or isotope names.
        x: Mass-fraction vector $X_i$ of shape `(n_species,)`.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        `CompositionInfo` including `dabar_dx`, `dzbar_dx`, and `dmc_dx`.

    Raises:
        ValueError: If `chem_ids` or `x` has invalid shape or species.
    """
    return get_composition_info(
        chem_ids, x, skip_partials=False, backend=backend
    )


def get_composition_info_zones(
    chem_ids: Sequence[int | str],
    x_zones: npt.NDArray[np.float64] | jax.Array,
    *,
    parallel: bool = True,
    backend: _dispatch.Backend = "auto",
) -> npt.NDArray[np.float64] | jax.Array:
    r"""Evaluate the 10 composition moments across `n_zones` stellar zones.

    Args:
        chem_ids: Sequence of length `n_species` of 1-based `chem_id`s or names.
        x_zones: 2D abundance array of shape `(n_zones, n_species)` in
            `float64`.
        parallel: When using the `"rust"` backend, evaluate across the
            persistent 4-thread `rayon` worker pool when `n_zones >= 64`.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        2D `float64` array of shape `(n_zones, 10)` containing columns
        `[xh, xhe, xz, abar, zbar, z2bar, z53bar, ye, mass_correction, sumx]`.

    Raises:
        ValueError: If `x_zones.ndim != 2` or `x_zones.shape[1] !=
            len(chem_ids)`.
    """
    resolved = _dispatch.resolve_backend(backend, x_zones)
    cids = _cat._resolve_chem_ids(chem_ids)
    if resolved == "jax":
        x_jax = jnp.asarray(x_zones, dtype=jnp.float64)
        if x_jax.ndim != 2 or x_jax.shape[1] != len(cids):
            raise ValueError(
                f"x_zones shape ({x_jax.shape}) must be (n_zones, {len(cids)})."
            )
        z_arr, a_arr, w_arr, z53_arr = _species_props_jax(cids)
        out: jax.Array = _composition_moments_only_jit(
            x_jax, z_arr, a_arr, w_arr, z53_arr
        )
        return out
    arr = np.ascontiguousarray(x_zones, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(cids):
        raise ValueError(
            f"x_zones shape ({arr.shape}) must be (n_zones, {len(cids)})."
        )
    return _rust.py_chem_get_composition_info_zones(
        list(cids), arr, bool(parallel)
    )
