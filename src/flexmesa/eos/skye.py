# SPDX-License-Identifier: LGPL-3.0-or-later
"""Jermyn et al. (2021) `Skye` plasma/crystallization EOS (`skye.f90`).

Provides the Tier B dual-track (`backend="auto" | "rust" | "jax"`) Python and
JAX interfaces for `get_skye_alfa`, `get_skye_alfa_simple`, and `eval_skye_eos`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust, chem, constants
from flexmesa.eos import _skye_jax, core, helm

_eval_skye_cell_zone_jit = jax.jit(_skye_jax.eval_skye_cell_zone_jax)
eval_skye_cell_zones_jit = _skye_jax.eval_skye_cell_zones_jit


def _quad_jax(p: jax.Array) -> jax.Array:
    """Return the 1..4 quadrant index for 2D vectors `p` of shape `(N, 2)`."""
    return jnp.where(
        (p[:, 0] >= 0.0) & (p[:, 1] >= 0.0),
        1,
        jnp.where(
            (p[:, 0] < 0.0) & (p[:, 1] >= 0.0),
            2,
            jnp.where((p[:, 0] < 0.0) & (p[:, 1] < 0.0), 3, 4),
        ),
    )


def _is_contained_jax(coords: jax.Array, pt: jax.Array) -> jax.Array:
    """Return True if 2D point `pt` lies inside polygon `coords` (`(N, 2)`)."""
    xs = coords - pt[None, :]
    xe = jnp.roll(xs, shift=-1, axis=0)
    dist = xe - xs
    det = xs[:, 1] * dist[:, 0] - xs[:, 0] * dist[:, 1]
    qs, qe = _quad_jax(xs), _quad_jax(xe)
    inc = jnp.where(
        qs == qe,
        0,
        jnp.where(
            qe == (qs % 4) + 1,
            1,
            jnp.where(qs == (qe % 4) + 1, -1, jnp.where(det < 0.0, 2, -2)),
        ),
    )
    return jnp.sum(inc) != 0


def get_skye_alfa_jax(
    log_rho: jax.Array,
    log_t: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate `Get_Skye_alfa` (`skye.f90` lines 36-117) in JAX.

    Args:
        log_rho: Base-10 log mass density $\\log_{10} \\rho$ in `g cm^-3`.
        log_t: Base-10 log temperature $\\log_{10} T$ in `K`.
        abar: Mean atomic weight $\\bar{A}$.
        zbar: Mean atomic charge $\\bar{Z}$.

    Returns:
        Tuple `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
    """
    log_rho_f = jnp.asarray(log_rho, dtype=jnp.float64)
    log_t_f = jnp.asarray(log_t, dtype=jnp.float64)
    abar_f = jnp.asarray(abar, dtype=jnp.float64)
    zbar_f = jnp.asarray(zbar, dtype=jnp.float64)

    skye_blend_width = 0.1
    t_ion = (
        jnp.maximum(7.3, jnp.log10(1.0e5 * zbar_f * zbar_f)) + skye_blend_width
    )
    rho_ion = (
        jnp.maximum(2.3, jnp.log10(abar_f * zbar_f * zbar_f * zbar_f))
        + skye_blend_width
    )
    rad_offset = (
        jnp.log10(
            abar_f
            * constants.MP
            * constants.CRAD
            / (3.0 * constants.KERG * (zbar_f + 1.0))
        )
        - 6.0
    )

    bounds = jnp.stack(
        [
            jnp.asarray([-12.0, 8.3], dtype=jnp.float64),
            jnp.stack([jnp.asarray(-12.0, dtype=jnp.float64), t_ion]),
            jnp.stack([rho_ion, t_ion]),
            jnp.stack([rho_ion, jnp.asarray(3.0, dtype=jnp.float64)]),
            jnp.asarray([15.0, 3.0], dtype=jnp.float64),
            jnp.asarray([15.0, 13.0], dtype=jnp.float64),
            jnp.stack([39.0 + rad_offset, jnp.asarray(13.0, jnp.float64)]),
            jnp.stack([24.9 + rad_offset, jnp.asarray(8.3, jnp.float64)]),
        ]
    )

    def _alfa_fn(lr: jax.Array, lt: jax.Array) -> jax.Array:
        pt = jnp.stack([lr, lt])
        contained = _is_contained_jax(bounds, jax.lax.stop_gradient(pt))
        dist = core.min_distance_to_polygon_jax(bounds, pt)
        sdist = jnp.where(contained, -dist, dist) / skye_blend_width
        return jnp.where(sdist <= 0.0, 0.0, jnp.where(sdist >= 1.0, 1.0, sdist))

    alfa = _alfa_fn(log_rho_f, log_t_f)
    d_dlr, d_dlt = jax.grad(_alfa_fn, argnums=(0, 1))(log_rho_f, log_t_f)
    return alfa, d_dlt, d_dlr


def _ramp_01(x: jax.Array, x_lo: float, x_hi: float) -> jax.Array:
    """Evaluate a linear `[0, 1]` ramp between `x_lo` and `x_hi`."""
    return jnp.where(
        x < x_lo, 0.0, jnp.where(x <= x_hi, (x - x_lo) / (x_hi - x_lo), 1.0)
    )


def get_skye_alfa_simple_jax(
    log_rho: jax.Array,
    log_t: jax.Array,
    log_rho_min_for_any_skye: float = 2.3,
    log_rho_min_for_all_skye: float = 2.4,
    log_t_min_for_any_skye: float = 7.2,
    log_t_min_for_all_skye: float = 7.4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate `Get_Skye_alfa_simple` (`skye.f90` lines 120-173) in JAX.

    Args:
        log_rho: Base-10 log mass density $\\log_{10} \\rho$ in `g cm^-3`.
        log_t: Base-10 log temperature $\\log_{10} T$ in `K`.
        log_rho_min_for_any_skye: Lower $\\log_{10} \\rho$ threshold for `Skye`.
        log_rho_min_for_all_skye: Upper $\\log_{10} \\rho$ threshold for `Skye`.
        log_t_min_for_any_skye: Lower $\\log_{10} T$ threshold for `Skye`.
        log_t_min_for_all_skye: Upper $\\log_{10} T$ threshold for `Skye`.

    Returns:
        Tuple `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
    """
    lr = jnp.asarray(log_rho, dtype=jnp.float64)
    lt = jnp.asarray(log_t, dtype=jnp.float64)

    def _fn(x_lr: jax.Array, x_lt: jax.Array) -> jax.Array:
        b_lt = _ramp_01(x_lt, log_t_min_for_any_skye, log_t_min_for_all_skye)
        b_lr = _ramp_01(
            x_lr, log_rho_min_for_any_skye, log_rho_min_for_all_skye
        )
        return (1.0 - b_lr) * (1.0 - b_lt)

    alfa = _fn(lr, lt)
    dlr, dlt = jax.grad(_fn, argnums=(0, 1))(lr, lt)
    return alfa, dlt, dlr


def get_skye_alfa(
    log_rho: float | jax.Array,
    log_t: float | jax.Array,
    abar: float | jax.Array,
    zbar: float | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> tuple[float | jax.Array, float | jax.Array, float | jax.Array]:
    """Evaluate `Get_Skye_alfa` (`skye.f90` lines 36-117).

    Args:
        log_rho: Base-10 log mass density $\\log_{10} \\rho$ in `g cm^-3`.
        log_t: Base-10 log temperature $\\log_{10} T$ in `K`.
        abar: Mean atomic weight $\\bar{A}$.
        zbar: Mean atomic charge $\\bar{Z}$.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Tuple `(alfa, d_alfa_dlogT, d_alfa_dlogRho)` where `alfa == 0` means
        pure `Skye` and `alfa == 1` means pure `HELM`/other.
    """
    if _dispatch.resolve_backend(backend, log_rho, log_t, abar, zbar) == "jax":
        return get_skye_alfa_jax(
            jnp.asarray(log_rho, dtype=jnp.float64),
            jnp.asarray(log_t, dtype=jnp.float64),
            jnp.asarray(abar, dtype=jnp.float64),
            jnp.asarray(zbar, dtype=jnp.float64),
        )
    return _rust.eos_get_skye_alfa(
        float(log_rho), float(log_t), float(abar), float(zbar)
    )


def get_skye_alfa_simple(
    log_rho: float | jax.Array,
    log_t: float | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> tuple[float | jax.Array, float | jax.Array, float | jax.Array]:
    """Evaluate `Get_Skye_alfa_simple` (`skye.f90` lines 120-173).

    Args:
        log_rho: Base-10 log mass density $\\log_{10} \\rho$ in `g cm^-3`.
        log_t: Base-10 log temperature $\\log_{10} T$ in `K`.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Tuple `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
    """
    if _dispatch.resolve_backend(backend, log_rho, log_t) == "jax":
        return get_skye_alfa_simple_jax(
            jnp.asarray(log_rho, dtype=jnp.float64),
            jnp.asarray(log_t, dtype=jnp.float64),
        )
    return _rust.eos_get_skye_alfa_simple(float(log_rho), float(log_t))


def eval_host_dt_jvp(
    rho: jax.Array,
    temp: jax.Array,
    host_cb: Callable[
        [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate a `(rho, temp)` host EOS callback with exact `custom_jvp`."""
    vc = jax.ShapeDtypeStruct((core.NUM_EOS_BASIC_RESULTS,), jnp.float64)

    def _eval_core(
        r_in: jax.Array, t_in: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        res_v, dlnd_v, dlnt_v = jax.pure_callback(
            host_cb, (vc, vc, vc), r_in, t_in
        )
        return res_v, dlnd_v, dlnt_v

    _eval = jax.custom_jvp(_eval_core)

    def _eval_jvp(
        primals: tuple[jax.Array, jax.Array],
        tangents: tuple[jax.Array, jax.Array],
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array],
        tuple[jax.Array, jax.Array, jax.Array],
    ]:
        r_p, t_p = primals
        dr, dt = tangents
        res, d_dlnd, d_dln_t = _eval_core(r_p, t_p)
        dres = d_dlnd * (dr / r_p) + d_dln_t * (dt / t_p)
        z = jnp.zeros_like(res)
        return (res, d_dlnd, d_dln_t), (dres, z, z)

    _eval.defjvp(_eval_jvp)
    return cast(
        tuple[jax.Array, jax.Array, jax.Array],
        _eval(
            jnp.asarray(rho, dtype=jnp.float64),
            jnp.asarray(temp, dtype=jnp.float64),
        ),
    )


def eval_skye_eos_jax(
    rho: jax.Array,
    temp: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    chem_id: Sequence[int],
    xa: jax.Array,
    *,
    skye_use_ion_offsets: bool = True,
    skye_solid_mixing_rule: str = "PC",
    mass_fraction_limit: float = 1.0e-4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate the 26-channel `Skye` EOS in JAX via native forward-mode AD.

    Args:
        rho: Mass density $\\rho$ in `g cm^-3` (scalar).
        temp: Temperature $T$ in `K` (scalar).
        abar: Mean atomic weight $\\bar{A}$.
        zbar: Mean atomic charge $\\bar{Z}$.
        chem_id: 1-based MESA nuclide identifiers of length `N_species`.
        xa: Species mass fractions of shape `(N_species,)`.
        skye_use_ion_offsets: Whether to include neutral ground-state offsets.
        skye_solid_mixing_rule: Solid mixing rule (`"PC"` or `"Ogata"`).
        mass_fraction_limit: Minimum species mass fraction included in `Skye`.

    Returns:
        Tuple `(res, d_dlnd, d_dln_t)` of JAX arrays with shape `(26,)`.
    """
    infos = [chem.get_nuclide_info(int(cid)) for cid in chem_id]
    azion = jnp.asarray([float(inf.z) for inf in infos], dtype=jnp.float64)
    acmi = jnp.asarray([float(inf.w) for inf in infos], dtype=jnp.float64)
    a_in = jnp.asarray([float(inf.a) for inf in infos], jnp.float64)
    r_j, t_j = jnp.asarray(rho, jnp.float64), jnp.asarray(temp, jnp.float64)
    ab_j, zb_j = jnp.asarray(abar, jnp.float64), jnp.asarray(zbar, jnp.float64)
    t_lo, dth, d_lo, dd, fi_f, _, fi_ef, _ = helm._lookup_helm_cell_jax(
        t_j, (zb_j / ab_j) * r_j
    )
    return cast(
        tuple[jax.Array, jax.Array, jax.Array],
        _eval_skye_cell_zone_jit(
            r_j,
            t_j,
            ab_j,
            zb_j,
            jnp.asarray(xa, jnp.float64),
            azion,
            acmi,
            a_in,
            t_lo,
            dth,
            d_lo,
            dd,
            fi_f,
            fi_ef,
            jnp.asarray(bool(skye_use_ion_offsets)),
            jnp.asarray(str(skye_solid_mixing_rule).lower() == "ogata"),
            float(mass_fraction_limit),
        ),
    )


def eval_skye_eos(
    rho: float | jax.Array,
    temp: float | jax.Array,
    abar: float | jax.Array,
    zbar: float | jax.Array,
    chem_id: Sequence[int],
    xa: np.ndarray | jax.Array,
    *,
    skye_use_ion_offsets: bool = True,
    skye_solid_mixing_rule: str = "PC",
    mass_fraction_limit: float = 1.0e-4,
    backend: _dispatch.Backend = "auto",
) -> core.EosEvalOutput:
    """Evaluate the 26-channel Jermyn et al. (2021) `Skye` equation of state.

    Args:
        rho: Mass density $\\rho$ in `g cm^-3` (scalar).
        temp: Temperature $T$ in `K` (scalar).
        abar: Mean atomic weight $\\bar{A}$.
        zbar: Mean atomic charge $\\bar{Z}$.
        chem_id: 1-based MESA nuclide identifiers of length `N_species`.
        xa: Species mass fractions of shape `(N_species,)`.
        skye_use_ion_offsets: Whether to include neutral ground-state offsets.
        skye_solid_mixing_rule: Solid mixing rule (`"PC"` or `"Ogata"`).
        mass_fraction_limit: Minimum species mass fraction included in `Skye`.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        [`EosEvalOutput`] with `res`, `d_dlnd`, and `d_dln_t` of shape `(26,)`.
    """
    if _dispatch.resolve_backend(backend, rho, temp, abar, zbar, xa) == "jax":
        res, d_dlnd, d_dln_t = eval_skye_eos_jax(
            jnp.asarray(rho, jnp.float64),
            jnp.asarray(temp, jnp.float64),
            jnp.asarray(abar, jnp.float64),
            jnp.asarray(zbar, jnp.float64),
            chem_id,
            jnp.asarray(xa, jnp.float64),
            skye_use_ion_offsets=skye_use_ion_offsets,
            skye_solid_mixing_rule=skye_solid_mixing_rule,
            mass_fraction_limit=mass_fraction_limit,
        )
        return core.EosEvalOutput(res=res, d_dlnd=d_dlnd, d_dln_t=d_dln_t)
    res_np, dlnd_np, dlnt_np = _rust.eos_eval_skye(
        float(rho),
        float(temp),
        float(abar),
        float(zbar),
        [int(c) for c in chem_id],
        np.ascontiguousarray(xa, dtype=np.float64),
        bool(skye_use_ion_offsets),
        str(skye_solid_mixing_rule),
        float(mass_fraction_limit),
    )
    return core.EosEvalOutput(res=res_np, d_dlnd=dlnd_np, d_dln_t=dlnt_np)


def eval_skye_eos_zones(
    rho: np.ndarray | jax.Array,
    temp: np.ndarray | jax.Array,
    abar: np.ndarray | jax.Array,
    zbar: np.ndarray | jax.Array,
    xa: np.ndarray | jax.Array,
    z: Sequence[int] | np.ndarray | jax.Array,
    acmi: np.ndarray | jax.Array,
    a: np.ndarray | jax.Array,
    *,
    skye_use_ion_offsets: bool = True,
    skye_solid_mixing_rule: str = "PC",
    mass_fraction_limit: float = 1.0e-4,
    parallel: bool = True,
    backend: _dispatch.Backend = "auto",
) -> core.EosEvalOutput:
    """Evaluate the 26-channel `Skye` EOS across `N_zones` stellar zones.

    Args:
        rho: Zone mass densities $\\rho$ in `g cm^-3` of shape `(N_zones,)`.
        temp: Zone temperatures $T$ in `K` of shape `(N_zones,)`.
        abar: Zone mean atomic weights $\\bar{A}$ of shape `(N_zones,)`.
        zbar: Zone mean atomic charges $\\bar{Z}$ of shape `(N_zones,)`.
        xa: Species mass fractions of shape `(N_zones, N_species)`.
        z: Species integer atomic charges $Z_i$ of shape `(N_species,)`.
        acmi: Species atomic weights $W_i$ in `amu` of shape `(N_species,)`.
        a: Species baryon numbers $A_i$ of shape `(N_species,)`.
        skye_use_ion_offsets: Whether to include neutral ground-state offsets.
        skye_solid_mixing_rule: Solid mixing rule (`"PC"` or `"Ogata"`).
        mass_fraction_limit: Minimum species mass fraction included in `Skye`.
        parallel: Whether to use the Rayon worker pool on `"rust"`.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        [`EosEvalOutput`] with `res`, `d_dlnd`, and `d_dln_t` of shape
        `(N_zones, 26)`.
    """
    if _dispatch.resolve_backend(backend, rho, temp, abar, zbar, xa) == "jax":
        r_j, t_j = jnp.asarray(rho, jnp.float64), jnp.asarray(temp, jnp.float64)
        ab_j = jnp.asarray(abar, jnp.float64)
        zb_j = jnp.asarray(zbar, jnp.float64)
        t_lo, dth, d_lo, dd, fi_f, _, fi_ef, _ = helm._lookup_helm_cell_jax(
            t_j, (zb_j / ab_j) * r_j
        )
        res, d_dlnd, d_dln_t = eval_skye_cell_zones_jit(
            r_j,
            t_j,
            ab_j,
            zb_j,
            jnp.asarray(xa, jnp.float64),
            jnp.asarray(z, jnp.float64),
            jnp.asarray(acmi, jnp.float64),
            jnp.asarray(a, jnp.float64),
            t_lo,
            dth,
            d_lo,
            dd,
            fi_f,
            fi_ef,
            jnp.asarray(bool(skye_use_ion_offsets)),
            jnp.asarray(str(skye_solid_mixing_rule).lower() == "ogata"),
            float(mass_fraction_limit),
        )
        return core.EosEvalOutput(res=res, d_dlnd=d_dlnd, d_dln_t=d_dln_t)
    out = _rust.eos_eval_skye_zones(
        np.ascontiguousarray(rho, dtype=np.float64),
        np.ascontiguousarray(temp, dtype=np.float64),
        np.ascontiguousarray(abar, dtype=np.float64),
        np.ascontiguousarray(zbar, dtype=np.float64),
        np.ascontiguousarray(np.asarray(xa, dtype=np.float64).reshape(-1)),
        [int(v) for v in np.asarray(z).reshape(-1)],
        np.ascontiguousarray(acmi, dtype=np.float64),
        np.ascontiguousarray(a, dtype=np.float64),
        bool(skye_use_ion_offsets),
        str(skye_solid_mixing_rule),
        float(mass_fraction_limit),
        bool(parallel),
    )
    n = core.NUM_EOS_BASIC_RESULTS
    return core.EosEvalOutput(
        res=out[:, :n], d_dlnd=out[:, n : 2 * n], d_dln_t=out[:, 2 * n : 3 * n]
    )
