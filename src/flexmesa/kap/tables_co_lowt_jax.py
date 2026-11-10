# SPDX-License-Identifier: LGPL-3.0-or-later
"""Type 2 CO, `AESOPUS`, and `kapCN` table loaders and spline caches for JAX."""

from __future__ import annotations

import struct
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa.kap import tables_jax

_CO_META_CACHE: dict[str, dict[str, Any]] = {}
_CO_SPLINE_CACHE: dict[tuple[str, int], jax.Array] = {}
_CO_ALL_Z_CACHE: dict[str, jax.Array] = {}
_AESOPUS_SPLINE_CACHE: dict[int, jax.Array] = {}
_AESOPUS_ALL_Z_CACHE: jax.Array | None = None
_KAPCN_CACHE: dict[str, Any] | None = None


def get_co_meta_jax(prefix: str) -> dict[str, Any]:
    """Load Type 2 CO metadata (`zbase`, `xs`, `co_table_numbers`, etc.)."""
    if prefix in _CO_META_CACHE:
        return _CO_META_CACHE[prefix]
    mv = tables_jax.ensure_indexed()
    if prefix not in tables_jax.CO_INDEX:
        msg = f"Unknown CO table prefix: {prefix!r}"
        raise ValueError(msg)
    z_list = tables_jax.CO_INDEX[prefix]
    nz = len(z_list)
    nx = len(z_list[0][2])
    zs = np.array([z for z, _, _ in z_list], dtype=np.float64)
    xs = np.zeros((nz, nx), dtype=np.float64)
    bounds = np.zeros((nz, nx, 4), dtype=np.float64)
    num_co = np.zeros((nz, nx), dtype=np.int32)
    num_gt = np.zeros((nz, nx), dtype=np.int32)
    tbl_nums = np.full((nz, nx, 8, 8), -1, dtype=np.int32)
    next_dxc = np.full((nz, nx, 60), -1, dtype=np.int32)
    next_dxo = np.full((nz, nx, 60), -1, dtype=np.int32)
    dxc_arr = np.zeros((nz, nx, 60), dtype=np.float64)
    dxo_arr = np.zeros((nz, nx, 60), dtype=np.float64)
    dxc_l_arr = np.zeros((nz, nx, 60), dtype=np.float64)
    dxo_l_arr = np.zeros((nz, nx, 60), dtype=np.float64)
    lrs_ref: np.ndarray | None = None
    lts_ref: np.ndarray | None = None

    for iz, (_, _, x_offsets) in enumerate(z_list):
        for ix, start in enumerate(x_offsets):
            p = start + 1
            x_v, z_v = struct.unpack_from("<2d", mv, p)
            p += 16
            nco, ngt = struct.unpack_from("<2I", mv, p)
            p += 8
            bds = struct.unpack_from("<4d", mv, p)
            p += 32
            nlr, nlt = struct.unpack_from("<2I", mv, p)
            p += 8
            lrs = np.frombuffer(mv, dtype="<f8", count=nlr, offset=p).copy()
            p += nlr * 8
            lts = np.frombuffer(mv, dtype="<f8", count=nlt, offset=p).copy()
            p += nlt * 8
            if lrs_ref is None:
                lrs_ref, lts_ref = lrs, lts
            xs[iz, ix] = x_v
            bounds[iz, ix] = bds
            num_co[iz, ix] = nco
            num_gt[iz, ix] = ngt
            for n in range(nco):
                dc, do = struct.unpack_from("<2d", mv, p)
                p += 17 + nlr * nlt * 2
                dxc_arr[iz, ix, n] = dc
                dxo_arr[iz, ix, n] = do
                dxc_l_arr[iz, ix, n] = np.log10(z_v + 1.0e-3 + dc)
                dxo_l_arr[iz, ix, n] = np.log10(z_v + 1.0e-3 + do)
                for ixc, v in enumerate(tables_jax.KAP_CO_DXS):
                    if abs(v - dc) < 1.0e-6:
                        for ixo, vo in enumerate(tables_jax.KAP_CO_DXS):
                            if abs(vo - do) < 1.0e-6:
                                tbl_nums[iz, ix, ixc, ixo] = n + 1
            for i in range(nco):
                dc, do = dxc_arr[iz, ix, i], dxo_arr[iz, ix, i]
                if dc > do:
                    next_dxc[iz, ix, i] = i + 2
                else:
                    for j in range(nco):
                        if (
                            dxc_arr[iz, ix, j] > dc
                            and abs(dxo_arr[iz, ix, j] - do) == 0.0
                        ):
                            next_dxc[iz, ix, i] = j + 1
                            break
                if dc < do:
                    next_dxo[iz, ix, i] = i + 2
                else:
                    for j in range(nco):
                        if (
                            dxo_arr[iz, ix, j] > do
                            and abs(dxc_arr[iz, ix, j] - dc) == 0.0
                        ):
                            next_dxo[iz, ix, i] = j + 1
                            break

    assert lrs_ref is not None and lts_ref is not None
    with jax.ensure_compile_time_eval():
        res = {
            "zs": jnp.asarray(zs),
            "xs": jnp.asarray(xs),
            "bounds": jnp.asarray(bounds),
            "num_co": jnp.asarray(num_co),
            "num_gt": jnp.asarray(num_gt),
            "tbl_nums": jnp.asarray(tbl_nums),
            "next_dxc": jnp.asarray(next_dxc),
            "next_dxo": jnp.asarray(next_dxo),
            "dxc": jnp.asarray(dxc_arr),
            "dxo": jnp.asarray(dxo_arr),
            "dxc_l": jnp.asarray(dxc_l_arr),
            "dxo_l": jnp.asarray(dxo_l_arr),
            "log_rs": jnp.asarray(lrs_ref),
            "log_ts": jnp.asarray(lts_ref),
            "log_rs_np": lrs_ref,
            "log_ts_np": lts_ref,
            "ili_r": tables_jax.splinck(lrs_ref),
            "ili_t": tables_jax.splinck(lts_ref),
        }
    _CO_META_CACHE[prefix] = res
    return res


def get_co_z_coeffs_jax(prefix: str, iz: int) -> jax.Array:
    """Fit and cache `(nx=5, 60, nlr, nlt, 4)` CO splines for `(prefix, iz)`."""
    key = (prefix, int(iz))
    if key in _CO_SPLINE_CACHE:
        return _CO_SPLINE_CACHE[key]
    meta = get_co_meta_jax(prefix)
    mv = tables_jax.ensure_indexed()
    _, _, x_offsets = tables_jax.CO_INDEX[prefix][iz]
    lrs, lts = meta["log_rs_np"], meta["log_ts_np"]
    nlr, nlt = int(lrs.shape[0]), int(lts.shape[0])
    nx = len(x_offsets)
    raw = np.zeros((nx, 60, nlt, nlr), dtype=np.float64)
    for ix, start in enumerate(x_offsets):
        p = start + 17
        nco, _ = struct.unpack_from("<2I", mv, p)
        p += 48 + (nlr + nlt) * 8
        for n in range(nco):
            p += 17
            raw[ix, n] = (
                np.frombuffer(mv, dtype="<i2", count=nlr * nlt, offset=p)
                .astype(np.float64)
                .reshape(nlt, nlr)
                / 1000.0
            )
            p += nlr * nlt * 2
    coeffs, _, _ = tables_jax.fit_bicubic_batch(
        lrs, lts, raw.reshape(nx * 60, nlt, nlr)
    )
    with jax.ensure_compile_time_eval():
        arr = jnp.asarray(coeffs.reshape(nx, 60, nlr, nlt, 4))
    _CO_SPLINE_CACHE[key] = arr
    return arr


def get_co_all_z_coeffs_jax(prefix: str) -> jax.Array:
    """Return `(8, 5, 60, 37, 138, 4)` CO spline array for `prefix`."""
    if prefix in _CO_ALL_Z_CACHE:
        return _CO_ALL_Z_CACHE[prefix]
    with jax.ensure_compile_time_eval():
        arr = jnp.stack(
            [get_co_z_coeffs_jax(prefix, iz) for iz in range(8)], axis=0
        )
    _CO_ALL_Z_CACHE[prefix] = arr
    return arr


def get_aesopus_meta_jax() -> dict[str, Any]:
    """Return `AESOPUS_AGSS09` metadata."""
    _ = tables_jax.ensure_indexed()
    return tables_jax.AESOPUS_META


def get_aesopus_z_coeffs_jax(iz: int) -> jax.Array:
    """Fit and cache `(total_slices, nlr, nlt, 4)` AESOPUS splines for `iz`."""
    iz_i = int(iz)
    if iz_i in _AESOPUS_SPLINE_CACHE:
        return _AESOPUS_SPLINE_CACHE[iz_i]
    mv = tables_jax.ensure_indexed()
    meta = tables_jax.AESOPUS_META
    lrs, lts = meta["log_rs_np"], meta["log_ts_np"]
    nlr, nlt = int(lrs.shape[0]), int(lts.shape[0])
    ts = meta["ts"][iz_i]
    ns = int(ts["total_slices"])
    p = int(ts["offset"])
    raw = np.zeros((ns, nlt, nlr), dtype=np.float64)
    for s in range(ns):
        p += 1
        raw[s] = (
            np.frombuffer(mv, dtype="<f4", count=nlr * nlt, offset=p)
            .astype(np.float64)
            .reshape(nlt, nlr)
        )
        p += nlr * nlt * 4
    coeffs, _, _ = tables_jax.fit_bicubic_batch(lrs, lts, raw)
    with jax.ensure_compile_time_eval():
        arr = jnp.asarray(coeffs)
    _AESOPUS_SPLINE_CACHE[iz_i] = arr
    return arr


def get_aesopus_all_z_coeffs_jax() -> jax.Array:
    """Return `(13, 1125, 19, 42, 4)` AESOPUS spline array across all `Z`."""
    global _AESOPUS_ALL_Z_CACHE
    if _AESOPUS_ALL_Z_CACHE is not None:
        return _AESOPUS_ALL_Z_CACHE
    meta = get_aesopus_meta_jax()
    num_zs = int(meta["zs"].shape[0])
    with jax.ensure_compile_time_eval():
        _AESOPUS_ALL_Z_CACHE = jnp.stack(
            [get_aesopus_z_coeffs_jax(iz) for iz in range(num_zs)], axis=0
        )
    return _AESOPUS_ALL_Z_CACHE


def get_kapcn_jax() -> dict[str, Any]:
    """Fit and cache all `(14, 63, 17, 18, 4)` `kapCN` splines."""
    global _KAPCN_CACHE
    if _KAPCN_CACHE is not None:
        return _KAPCN_CACHE
    mv = tables_jax.ensure_indexed()
    meta = tables_jax.KAPCN_META
    lrs, lts = meta["log_rs_np"], meta["log_ts_np"]
    nlr, nlt, nz = int(lrs.shape[0]), int(lts.shape[0]), meta["num_zs"]
    p = int(meta["offset"])
    zs = np.zeros(nz, dtype=np.float64)
    fcs = np.zeros((nz, 7), dtype=np.float64)
    fns = np.zeros((nz, 3), dtype=np.float64)
    raw = np.zeros((nz, 63, nlt, nlr), dtype=np.float64)
    for iz in range(nz):
        (zs[iz],) = struct.unpack_from("<d", mv, p)
        p += 8
        fcs[iz] = np.frombuffer(mv, dtype="<f8", count=7, offset=p)
        p += 56
        fns[iz] = np.frombuffer(mv, dtype="<f8", count=3, offset=p)
        p += 24
        for s in range(63):
            raw[iz, s] = (
                np.frombuffer(mv, dtype="<i2", count=nlr * nlt, offset=p)
                .astype(np.float64)
                .reshape(nlt, nlr)
                / 1000.0
            )
            p += nlr * nlt * 2
    coeffs, ili_r, ili_t = tables_jax.fit_bicubic_batch(
        lrs, lts, raw.reshape(nz * 63, nlt, nlr)
    )
    with jax.ensure_compile_time_eval():
        _KAPCN_CACHE = {
            "log_rs": jnp.asarray(lrs),
            "log_ts": jnp.asarray(lts),
            "xs": jnp.asarray([0.5, 0.7, 0.8], dtype=jnp.float64),
            "zs": jnp.asarray(zs),
            "fcs": jnp.asarray(fcs),
            "fns": jnp.asarray(fns),
            "ili_r": ili_r,
            "ili_t": ili_t,
            "coeffs": jnp.asarray(coeffs.reshape(nz, 63, nlr, nlt, 4)),
        }
    return _KAPCN_CACHE
