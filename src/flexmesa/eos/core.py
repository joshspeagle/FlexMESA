# SPDX-License-Identifier: LGPL-3.0-or-later
"""Core EOS schema, polygonal blending, and ground-state ionization offsets.

Provides the Tier B dual-track (`backend="auto" | "rust" | "jax"`) Python and
JAX interfaces for Wave 3a.1 (`eos_def`, `eos_blend`, and `ion_offset`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust, chem, constants

NUM_EOS_BASIC_RESULTS: int = 26
NUM_EOS_D_DXA_RESULTS: int = 2
NUM_EOS_FRAC_RESULTS: int = 7

I_LN_PGAS: int = 0
I_LN_E: int = 1
I_LN_S: int = 2
I_MU: int = 3
I_LN_FREE_E: int = 4
I_ETA: int = 5
I_GRAD_AD: int = 6
I_CHI_RHO: int = 7
I_CHI_T: int = 8
I_CP: int = 9
I_CV: int = 10
I_DE_DRHO: int = 11
I_DS_DT: int = 12
I_DS_DRHO: int = 13
I_GAMMA1: int = 14
I_GAMMA3: int = 15
I_PHASE: int = 16
I_LATENT_DDLNT: int = 17
I_LATENT_DDLNRHO: int = 18
I_FRAC_HELM: int = 19
I_FRAC_OPAL_SCVH: int = 20
I_FRAC_FREE_EOS: int = 21
I_FRAC_PC: int = 22
I_FRAC_SKYE: int = 23
I_FRAC_CMS: int = 24
I_FRAC_IDEAL: int = 25

EOS_BASIC_RESULT_NAMES: tuple[str, ...] = tuple(_rust.eos_basic_result_names())

NIST_IONIZATION_TABLE_EV: jnp.ndarray = jnp.asarray(
    [
        13.598434599702,
        79.0051545,
        203.4861711,
        399.14864,
        670.9809,
        1030.1085,
        1486.058,
        2043.8428,
        2715.89,
        3511.696,
        4420.0,
        5451.06,
        6604.95,
        7888.53,
        9305.8,
        10859.7,
        12556.4,
        14400.8,
        16382.0,
        18510.0,
        20788.0,
        23221.0,
        25820.0,
        28582.0,
        31514.0,
        34619.0,
        37899.0,
        41356.0,
    ],
    dtype=jnp.float64,
)


@dataclasses.dataclass(frozen=True)
class EosEvalOutput:
    """Complete output of an EOS evaluation at a single thermodynamic state.

    Attributes:
        res: 26-channel thermodynamic state vector of shape `(26,)` (`f64`).
        d_dlnd: Partial derivatives `(d res / d ln rho)_T` of shape `(26,)`.
        d_dln_t: Partial derivatives `(d res / d ln T)_rho` of shape `(26,)`.
    """

    res: np.ndarray | jnp.ndarray
    d_dlnd: np.ndarray | jnp.ndarray
    d_dln_t: np.ndarray | jnp.ndarray


def _quintic_blend_core(alpha0: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Evaluate the $C^2$ quintic blend polynomial and derivative in JAX."""
    a = jnp.asarray(alpha0, dtype=jnp.float64)
    a2 = a * a
    a3 = a2 * a
    s_raw = -a3 * (-10.0 + a * (15.0 - 6.0 * a))
    ds_raw = 30.0 * (a - 1.0) * (a - 1.0) * a2
    inside = (a > 0.0) & (a < 1.0)
    s = jnp.where(a <= 0.0, 0.0, jnp.where(a >= 1.0, 1.0, s_raw))
    ds = jnp.where(inside, ds_raw, 0.0)
    return s, ds


_quintic_blend_jit = jax.jit(_quintic_blend_core)


def quintic_blend_jax(alpha0: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Evaluate the $C^2$ quintic blend polynomial and derivative in JAX.

    Args:
        alpha0: Raw blend coordinate (dimensionless, scalar or `(*batch,)`).

    Returns:
        Tuple `(s, ds_dalpha0)` clamped to `[0, 1]` outside `[0, 1]`.
    """
    s, ds = _quintic_blend_jit(jnp.asarray(alpha0, dtype=jnp.float64))
    return cast(jax.Array, s), cast(jax.Array, ds)


def quintic_blend(
    alpha0: float | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> tuple[float | jax.Array, float | jax.Array]:
    """Evaluate the $C^2$ quintic blend polynomial and derivative.

    Args:
        alpha0: Raw blend coordinate in `[0, 1]` (dimensionless).
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Tuple `(s, ds_dalpha0)`.
    """
    resolved = _dispatch.resolve_backend(backend, alpha0)
    if resolved == "jax":
        return quintic_blend_jax(jnp.asarray(alpha0, dtype=jnp.float64))
    return _rust.eos_quintic_blend(float(alpha0))


def blend_eos_results(
    alpha_in: float | jax.Array,
    d_alpha_dlog_t_in: float | jax.Array,
    d_alpha_dlog_rho_in: float | jax.Array,
    linear_blend: bool,
    res_1: EosEvalOutput,
    res_2: EosEvalOutput,
    *,
    backend: _dispatch.Backend = "auto",
) -> EosEvalOutput:
    """Blend two 26-channel EOS evaluations (`Do_Blend` in `eosdt_support.f90`).

    Args:
        alpha_in: Raw blend coordinate in `[0, 1]` (dimensionless).
        d_alpha_dlog_t_in: Partial derivative `d(alpha_0) / d(log10 T)`.
        d_alpha_dlog_rho_in: Partial derivative `d(alpha_0) / d(log10 rho)`.
        linear_blend: If True, blend linearly; otherwise apply $C^2$ quintic
            smoothing.
        res_1: First EOS state evaluation (weighted by `alpha`).
        res_2: Second EOS state evaluation (weighted by `1 - alpha`).
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Blended [`EosEvalOutput`] with chain-rule partial derivatives.
    """
    resolved = _dispatch.resolve_backend(
        backend, alpha_in, res_1.res, res_2.res
    )
    if resolved == "jax":
        a0 = jnp.asarray(alpha_in, dtype=jnp.float64)
        da0_dlnt = (
            jnp.asarray(d_alpha_dlog_t_in, dtype=jnp.float64) / constants.LN10
        )
        da0_dlnd = (
            jnp.asarray(d_alpha_dlog_rho_in, dtype=jnp.float64) / constants.LN10
        )
        if linear_blend:
            alpha = a0
            d_alpha_dlnt = da0_dlnt
            d_alpha_dlnd = da0_dlnd
        else:
            a2 = a0 * a0
            a3 = a2 * a0
            alpha = -a3 * (-10.0 + a0 * (15.0 - 6.0 * a0))
            big_a = 30.0 * (a0 - 1.0) * (a0 - 1.0) * a2
            d_alpha_dlnt = big_a * da0_dlnt
            d_alpha_dlnd = big_a * da0_dlnd
        beta = 1.0 - alpha
        r1 = jnp.asarray(res_1.res, dtype=jnp.float64)
        r2 = jnp.asarray(res_2.res, dtype=jnp.float64)
        dlnd1 = jnp.asarray(res_1.d_dlnd, dtype=jnp.float64)
        dlnd2 = jnp.asarray(res_2.d_dlnd, dtype=jnp.float64)
        dlnt1 = jnp.asarray(res_1.d_dln_t, dtype=jnp.float64)
        dlnt2 = jnp.asarray(res_2.d_dln_t, dtype=jnp.float64)
        return EosEvalOutput(
            res=alpha * r1 + beta * r2,
            d_dlnd=alpha * dlnd1 + beta * dlnd2 + d_alpha_dlnd * (r1 - r2),
            d_dln_t=alpha * dlnt1 + beta * dlnt2 + d_alpha_dlnt * (r1 - r2),
        )
    res_b, dlnd_b, dlnt_b = _rust.eos_blend_results(
        float(alpha_in),
        float(d_alpha_dlog_t_in),
        float(d_alpha_dlog_rho_in),
        bool(linear_blend),
        np.ascontiguousarray(res_1.res, dtype=np.float64),
        np.ascontiguousarray(res_1.d_dlnd, dtype=np.float64),
        np.ascontiguousarray(res_1.d_dln_t, dtype=np.float64),
        np.ascontiguousarray(res_2.res, dtype=np.float64),
        np.ascontiguousarray(res_2.d_dlnd, dtype=np.float64),
        np.ascontiguousarray(res_2.d_dln_t, dtype=np.float64),
    )
    return EosEvalOutput(res=res_b, d_dlnd=dlnd_b, d_dln_t=dlnt_b)


def is_contained(
    coords: np.ndarray,
    x: float,
    y: float,
) -> bool:
    """Return True if point `(x, y)` lies inside the polygon `coords`.

    Args:
        coords: Polygon vertices of shape `(N, 2)` in `(logRho, logT)` space.
        x: Horizontal coordinate (e.g., `log10(rho)` in `g cm^-3`).
        y: Vertical coordinate (e.g., `log10(T)` in `K`).

    Returns:
        True when the winding number around `(x, y)` is non-zero.
    """
    arr = np.ascontiguousarray(coords, dtype=np.float64)
    return bool(_rust.eos_is_contained(arr, float(x), float(y)))


def _min_distance_to_polygon_core(
    coords: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute the differentiable minimum distance from `p` to `coords`."""
    c = jnp.asarray(coords, dtype=jnp.float64)
    pt = jnp.asarray(p, dtype=jnp.float64)
    starts = c
    ends = jnp.roll(c, shift=-1, axis=0)
    diff_line = ends - starts
    diff_start = pt[None, :] - starts
    diff_end = pt[None, :] - ends
    len_sq = jnp.sum(diff_line * diff_line, axis=1)
    lam = jnp.sum(diff_start * diff_line, axis=1) / len_sq
    d_start = jnp.sqrt(jnp.sum(diff_start * diff_start, axis=1))
    d_end = jnp.sqrt(jnp.sum(diff_end * diff_end, axis=1))
    near = starts + lam[:, None] * diff_line - pt[None, :]
    eps = jnp.float64(np.float32(1.0e-10))
    d_mid = jnp.sqrt(jnp.sum(near * near, axis=1) + eps * eps)
    d_seg = jnp.where(lam < 0.0, d_start, jnp.where(lam > 1.0, d_end, d_mid))
    return jnp.min(d_seg)


_min_distance_to_polygon_jit = jax.jit(_min_distance_to_polygon_core)


def min_distance_to_polygon_jax(
    coords: jax.Array,
    p: jax.Array,
) -> jax.Array:
    """Compute the differentiable minimum distance from `p` to polygon `coords`.

    Args:
        coords: Polygon vertices of shape `(N, 2)` (`float64`).
        p: Query point of shape `(2,)` (`float64`).

    Returns:
        Minimum distance scalar (`float64`) regularized by `eps = 1e-10_f32` on
        interior segment projections.
    """
    return cast(jax.Array, _min_distance_to_polygon_jit(coords, p))


def min_distance_to_polygon(
    coords: np.ndarray | jax.Array,
    x: float | jax.Array,
    y: float | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> tuple[float | jax.Array, float | jax.Array, float | jax.Array]:
    """Compute the minimum distance from `(x, y)` to polygon `coords`.

    Args:
        coords: Polygon vertices of shape `(N, 2)` (`float64`).
        x: First coordinate (`float64`).
        y: Second coordinate (`float64`).
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Tuple `(d, dd_dx, dd_dy)` on `"rust"` or via `jax.value_and_grad` on
        `"jax"`.
    """
    resolved = _dispatch.resolve_backend(backend, coords, x, y)
    if resolved == "jax":
        c_j = jnp.asarray(coords, dtype=jnp.float64)
        p_j = jnp.stack(
            [
                jnp.asarray(x, dtype=jnp.float64),
                jnp.asarray(y, dtype=jnp.float64),
            ]
        )
        val, grad = jax.value_and_grad(
            lambda pt: min_distance_to_polygon_jax(c_j, pt)
        )(p_j)
        return val, grad[0], grad[1]
    c_np = np.ascontiguousarray(coords, dtype=np.float64)
    return _rust.eos_min_distance_to_polygon(c_np, float(x), float(y))


def compute_ion_offset_from_za(
    xa: np.ndarray | jax.Array,
    z: np.ndarray | list[int],
    a: np.ndarray | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Compute the ground-state ionization energy offset in `erg g^-1`.

    Args:
        xa: Species mass fractions of shape `(N_species,)` (dimensionless).
        z: Atomic numbers $Z_k$ of shape `(N_species,)` (`int`).
        a: Mass numbers $A_k$ of shape `(N_species,)` (`float64`).
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Ionization energy offset $\\Delta E_{\\mathrm{ion}}$ in `erg g^-1`.
    """
    resolved = _dispatch.resolve_backend(backend, xa, a)
    if resolved == "jax":
        xa_j = jnp.asarray(xa, dtype=jnp.float64)
        z_j = jnp.asarray(z, dtype=jnp.int32)
        a_j = jnp.asarray(a, dtype=jnp.float64)
        valid = (z_j >= 1) & (z_j <= 28) & (a_j > 0.0)
        idx = jnp.clip(z_j - 1, 0, 27)
        a_safe = jnp.maximum(a_j, 1.0)
        ev = jnp.where(
            valid, NIST_IONIZATION_TABLE_EV[idx] * xa_j / a_safe, 0.0
        )
        return jnp.sum(ev) * constants.EV2ERG / constants.AMU
    xa_np = np.ascontiguousarray(xa, dtype=np.float64)
    a_np = np.ascontiguousarray(a, dtype=np.float64)
    z_vals = [int(v) for v in z]
    return float(_rust.eos_compute_ion_offset_from_za(xa_np, z_vals, a_np))


def nuclide_arrays_jax(
    chem_id: Sequence[int],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return `(azion, acmi, a_baryon)` as `jnp.float64` arrays for `chem_id`.

    Args:
        chem_id: 1-based MESA nuclide identifiers of length `N_species`.

    Returns:
        Tuple `(azion, acmi, a_baryon)` of shape `(N_species,)` in `float64`.
    """
    infos = [chem.get_nuclide_info(int(cid)) for cid in chem_id]
    return (
        jnp.asarray([float(inf.z) for inf in infos], dtype=jnp.float64),
        jnp.asarray([float(inf.w) for inf in infos], dtype=jnp.float64),
        jnp.asarray([float(inf.a) for inf in infos], dtype=jnp.float64),
    )


def pack_dt_jvp(
    r_p: jax.Array,
    t_p: jax.Array,
    dr: jax.Array,
    dt: jax.Array,
    primal_out: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array],
]:
    """Assemble `(primal, tangent)` for `(rho, T)` `custom_jvp` EOS rules.

    Args:
        r_p: Primal mass density $\\rho$ in `g cm^-3`.
        t_p: Primal temperature $T$ in `K`.
        dr: Tangent $\\delta \\rho$ in `g cm^-3`.
        dt: Tangent $\\delta T$ in `K`.
        primal_out: Tuple `(res, d_dlnd, d_dln_t)` at `(r_p, t_p)`.

    Returns:
        `(primal_out, (dres, zero, zero))` matching logarithmic EOS partials.
    """
    res_0, dlnd_0, dlnt_0 = primal_out
    dln_r = jnp.expand_dims(dr / r_p, axis=-1)
    dln_t = jnp.expand_dims(dt / t_p, axis=-1)
    dres = dlnd_0 * dln_r + dlnt_0 * dln_t
    z = jnp.zeros_like(res_0)
    return primal_out, (dres, z, z)


def compute_ion_offset(
    xa: np.ndarray,
    chem_id: list[int] | np.ndarray,
) -> float:
    """Compute the ground-state ionization offset from `xa` and `chem_id`.

    Args:
        xa: Species mass fractions of shape `(N_species,)` (dimensionless).
        chem_id: 1-based MESA nuclide identifiers of shape `(N_species,)`.

    Returns:
        Ionization energy offset $\\Delta E_{\\mathrm{ion}}$ in `erg g^-1`.
    """
    xa_np = np.ascontiguousarray(xa, dtype=np.float64)
    chem_ids = [int(v) for v in chem_id]
    return float(_rust.eos_compute_ion_offset(xa_np, chem_ids))
