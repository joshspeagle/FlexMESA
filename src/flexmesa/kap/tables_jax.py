# SPDX-License-Identifier: LGPL-3.0-or-later
"""On-demand binary opacity table loader and bicubic spline cache for JAX.

Reads the full `FMKAP001` binary payload from
`.mesa_temp_cache/bin_cache/kap_full/kap_tables_full.bin` (triggering the Rust
table builder on first access if not yet cached) and fits 2D PSPLINE bicubic
spline coefficients `[f, f_xx, f_yy, f_xxyy]` on demand for native `@jax.jit`
evaluation.
"""

from __future__ import annotations

import os
import pathlib
import struct
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _rust

KAP_CO_DXS: tuple[float, ...] = (
    0.00,
    0.01,
    0.03,
    0.10,
    0.20,
    0.40,
    0.60,
    1.00,
)

_RAW_BYTES: memoryview | None = None
FIXED_INDEX: dict[str, list[tuple[float, list[tuple[int, int]]]]] = {}
CO_INDEX: dict[
    str, list[tuple[float, tuple[float, float, float, float], list[int]]]
] = {}
AESOPUS_META: dict[str, Any] = {}
KAPCN_META: dict[str, Any] = {}

_FIXED_CACHE: dict[str, dict[str, Any]] = {}
_POTEKHIN_CACHE: dict[str, Any] | None = None


def splinck(x: np.ndarray, ztol: float = 1.0e-3) -> bool:
    """Check whether 1D grid `x` is uniformly spaced within `ztol`."""
    n = int(x.shape[0])
    if n <= 1:
        return True
    dxavg = float(x[-1] - x[0]) / float(n - 1)
    zeps = abs(ztol * dxavg)
    diffs = np.diff(x)
    return bool(np.all(np.abs(diffs - dxavg) <= zeps))


def _mkspline_axis0(
    x: np.ndarray,
    y: np.ndarray,
    bc: int = 0,
) -> np.ndarray:
    """Vectorized 1D PSPLINE (`v_spline_db`) along axis 0 (`bc=0` or `bc=3`)."""
    n = int(x.shape[0])
    m = y.shape[1:]
    y2d = y.reshape(n, -1)
    k = int(y2d.shape[1])
    h = np.diff(x)
    d = np.zeros((n, k), dtype=np.float64)
    rhs = np.zeros((n, k), dtype=np.float64)
    sec = (y2d[1:] - y2d[:-1]) / h[:, None]
    for i in range(1, n - 1):
        d[i] = 2.0 * (h[i - 1] + h[i])
        rhs[i] = sec[i] - sec[i - 1]

    if bc == 3:
        d[0] = 2.0 * h[0]
        rhs[0] = sec[0]
        d[n - 1] = 2.0 * h[n - 2]
        rhs[n - 1] = -sec[n - 2]
        for i in range(1, n):
            t = h[i - 1] / d[i - 1]
            d[i] -= t * h[i - 1]
            rhs[i] -= t * rhs[i - 1]
        rhs[n - 1] /= d[n - 1]
        for i in range(n - 2, -1, -1):
            rhs[i] = (rhs[i] - h[i] * rhs[i + 1]) / d[i]
    else:
        d[1] = h[0] + 2.0 * h[1]
        rhs[1] = rhs[1] * h[1] / (h[0] + h[1])
        d[n - 2] = 2.0 * h[n - 3] + h[n - 2]
        rhs[n - 2] = rhs[n - 2] * h[n - 3] / (h[n - 2] + h[n - 3])
        for i in range(2, n - 1):
            t = (
                (h[i - 1] - h[i]) / d[i - 1]
                if i == n - 2
                else h[i - 1] / d[i - 1]
            )
            d[i] -= t * ((h[1] - h[0]) if i == 2 else h[i - 1])
            rhs[i] -= t * rhs[i - 1]
        rhs[n - 2] /= d[n - 2]
        for i in range(n - 3, 0, -1):
            sub = (h[1] - h[0]) * rhs[2] if i == 1 else h[i] * rhs[i + 1]
            rhs[i] = (rhs[i] - sub) / d[i]
        rhs[0] = (rhs[1] * (h[0] + h[1]) - rhs[2] * h[0]) / h[1]
        rhs[n - 1] = (
            rhs[n - 2] + (rhs[n - 2] - rhs[n - 3]) * h[n - 2] / h[n - 3]
        )

    fxx = 6.0 * rhs
    fxxx_last = 6.0 * (rhs[n - 1] - rhs[n - 2]) / h[n - 2]
    fxx[n - 1] = fxx[n - 2] + h[n - 2] * fxxx_last
    return fxx.reshape((n, *m))


def fit_bicubic_batch(
    log_rs: np.ndarray,
    log_ts: np.ndarray,
    raw_lt_lr: np.ndarray,
    bc: int = 0,
) -> tuple[np.ndarray, bool, bool]:
    """Fit 2D PSPLINE `[f, f_xx, f_yy, f_xxyy]` on `(B, ny, nx)` tables."""
    nx = int(log_rs.shape[0])
    ny = int(log_ts.shape[0])
    bsz = int(raw_lt_lr.shape[0])
    f = np.swapaxes(raw_lt_lr, -2, -1).astype(np.float64, copy=True)
    f = np.where(np.isfinite(f), f, 0.0)
    f_x_in = np.transpose(f, (1, 0, 2)).reshape(nx, bsz * ny)
    f_xx = np.transpose(
        _mkspline_axis0(log_rs, f_x_in, bc=bc).reshape(nx, bsz, ny),
        (1, 0, 2),
    )
    f_y_in = np.transpose(f, (2, 0, 1)).reshape(ny, bsz * nx)
    f_yy = np.transpose(
        _mkspline_axis0(log_ts, f_y_in, bc=bc).reshape(ny, bsz, nx),
        (1, 2, 0),
    )
    f_xy_in = np.transpose(f_xx, (2, 0, 1)).reshape(ny, bsz * nx)
    f_xxyy = np.transpose(
        _mkspline_axis0(log_ts, f_xy_in, bc=bc).reshape(ny, bsz, nx),
        (1, 2, 0),
    )
    coeffs = np.stack([f, f_xx, f_yy, f_xxyy], axis=-1)
    coeffs = np.where(np.isfinite(coeffs), coeffs, 0.0)
    return coeffs, splinck(log_rs), splinck(log_ts)


def _find_kap_bin_path() -> pathlib.Path:
    """Resolve `kap_tables_full.bin`, building via Rust if needed."""
    _ = _rust.kap_get_fixed_metal_rust(0.02, 0.70, -6.0, 6.0)
    roots = [pathlib.Path.cwd(), pathlib.Path(__file__).resolve()]
    if "FLEXMESA_CACHE_DIR" in os.environ:
        cand = (
            pathlib.Path(os.environ["FLEXMESA_CACHE_DIR"])
            / "kap_full"
            / "kap_tables_full.bin"
        )
        if cand.is_file():
            return cand
    for root in roots:
        for anc in [root, *root.parents]:
            cand = (
                anc
                / ".mesa_temp_cache"
                / "bin_cache"
                / "kap_full"
                / "kap_tables_full.bin"
            )
            if cand.is_file():
                return cand
    for root in roots:
        for anc in [root, *root.parents]:
            cand = anc / "crates" / "flexmesa-kap" / "data" / "kap_tables.bin"
            if cand.is_file():
                return cand
    msg = "Unable to locate kap_tables_full.bin or kap_tables.bin"
    raise FileNotFoundError(msg)


def ensure_indexed() -> memoryview:
    """Index byte offsets across the `FMKAP001` binary file once."""
    global _RAW_BYTES
    if _RAW_BYTES is not None:
        return _RAW_BYTES
    raw = _find_kap_bin_path().read_bytes()
    if raw[:8] != b"FMKAP001":
        msg = "Invalid header in kap_tables binary"
        raise ValueError(msg)
    mv = memoryview(raw)
    pos = 8

    def u8() -> int:
        nonlocal pos
        v = int(mv[pos])
        pos += 1
        return v

    def u32() -> int:
        nonlocal pos
        (v,) = struct.unpack_from("<I", mv, pos)
        pos += 4
        return int(v)

    def f64() -> float:
        nonlocal pos
        (v,) = struct.unpack_from("<d", mv, pos)
        pos += 8
        return float(v)

    def string() -> str:
        nonlocal pos
        ln = u32()
        s = bytes(mv[pos : pos + ln]).decode("utf-8")
        pos += ln
        return s

    for _ in range(u32()):
        prefix = string()
        z_list = []
        for _ in range(u32()):
            z_val = f64()
            x_list = []
            for _ in range(u32()):
                start = pos
                present = u8() != 0
                pos += 16
                if present:
                    pos += 32
                    nlr, nlt = u32(), u32()
                    pos += (nlr + nlt) * 8 + nlr * nlt * 2
                x_list.append((start, pos))
            z_list.append((z_val, x_list))
        FIXED_INDEX[prefix] = z_list

    for _ in range(u32()):
        prefix = string()
        z_list_co = []
        for _ in range(u32()):
            zbase = f64()
            zfracs = (f64(), f64(), f64(), f64())
            x_offsets = []
            for _ in range(u32()):
                start = pos
                present = u8() != 0
                pos += 16
                if present:
                    nco, _ = u32(), u32()
                    pos += 32
                    nlr, nlt = u32(), u32()
                    pos += (nlr + nlt) * 8
                    for _ in range(nco):
                        pos += 16
                        if u8() != 0:
                            pos += nlr * nlt * 2
                x_offsets.append(start)
            z_list_co.append((zbase, zfracs, x_offsets))
        CO_INDEX[prefix] = z_list_co

    zsun, fco_ref, fc_ref, fn_ref = f64(), f64(), f64(), f64()
    nlr_a, nlt_a, nzs_a = u32(), u32(), u32()
    log_rs_a = np.frombuffer(mv, dtype="<f8", count=nlr_a, offset=pos).copy()
    pos += nlr_a * 8
    log_ts_a = np.frombuffer(mv, dtype="<f8", count=nlt_a, offset=pos).copy()
    pos += nlt_a * 8
    zs_a = np.frombuffer(mv, dtype="<f8", count=nzs_a, offset=pos).copy()
    pos += nzs_a * 8
    ts_meta = []
    with jax.ensure_compile_time_eval():
        for _ in range(nzs_a):
            nx_a, nfco_a, nfc_a, nfn_a = u32(), u32(), u32(), u32()
            xs_a = np.frombuffer(
                mv, dtype="<f8", count=nx_a, offset=pos
            ).copy()
            pos += nx_a * 8
            fcos_a = np.frombuffer(
                mv, dtype="<f8", count=nfco_a, offset=pos
            ).copy()
            pos += nfco_a * 8
            fcs_a = np.frombuffer(
                mv, dtype="<f8", count=nfc_a, offset=pos
            ).copy()
            pos += nfc_a * 8
            fns_a = np.frombuffer(
                mv, dtype="<f8", count=nfn_a, offset=pos
            ).copy()
            pos += nfn_a * 8
            z_present = u8() != 0
            total_slices = nx_a * nfco_a * nfc_a * nfn_a
            slice_start = pos
            if z_present:
                for _ in range(total_slices):
                    if u8() != 0:
                        pos += nlr_a * nlt_a * 4
            ts_meta.append(
                {
                    "xs": jnp.asarray(xs_a),
                    "fcos": jnp.asarray(fcos_a),
                    "fcs": jnp.asarray(fcs_a),
                    "fns": jnp.asarray(fns_a),
                    "total_slices": total_slices,
                    "offset": slice_start,
                }
            )
        AESOPUS_META.update(
            {
                "zsun": zsun,
                "fco_ref": fco_ref,
                "fc_ref": fc_ref,
                "fn_ref": fn_ref,
                "log_rs": jnp.asarray(log_rs_a),
                "log_ts": jnp.asarray(log_ts_a),
                "log_rs_np": log_rs_a,
                "log_ts_np": log_ts_a,
                "zs": jnp.asarray(zs_a),
                "log_zs": jnp.asarray(np.log10(zs_a)),
                "ilinx": splinck(log_rs_a),
                "iliny": splinck(log_ts_a),
                "ts": ts_meta,
                "xs_all": jnp.stack([t["xs"] for t in ts_meta], axis=0),
                "fcos_all": jnp.stack([t["fcos"] for t in ts_meta], axis=0),
                "fcs_all": jnp.stack([t["fcs"] for t in ts_meta], axis=0),
                "fns_all": jnp.stack([t["fns"] for t in ts_meta], axis=0),
            }
        )

    ncn_z, ncn_lr, ncn_lt = u32(), u32(), u32()
    cn_lrs = np.frombuffer(mv, dtype="<f8", count=ncn_lr, offset=pos).copy()
    pos += ncn_lr * 8
    cn_lts = np.frombuffer(mv, dtype="<f8", count=ncn_lt, offset=pos).copy()
    pos += ncn_lt * 8
    KAPCN_META.update(
        {
            "num_zs": ncn_z,
            "log_rs_np": cn_lrs,
            "log_ts_np": cn_lts,
            "offset": pos,
        }
    )
    _RAW_BYTES = mv
    return mv


def get_fixed_set_jax(prefix: str) -> dict[str, Any]:
    """Load and spline-fit a fixed-metal or low-T table set for JAX."""
    canon = "lowT_fa05_a09p" if prefix == "lowT_fa05_a09" else prefix
    if canon in _FIXED_CACHE:
        return _FIXED_CACHE[canon]
    mv = ensure_indexed()
    if canon not in FIXED_INDEX:
        msg = f"Unknown fixed-metal table prefix: {prefix!r}"
        raise ValueError(msg)
    z_list = FIXED_INDEX[canon]
    num_zs = len(z_list)
    max_xs = max(len(x_list) for _, x_list in z_list)
    zs = np.array([z for z, _ in z_list], dtype=np.float64)
    num_xs = np.array([len(xl) for _, xl in z_list], dtype=np.int32)
    xs_grid = np.zeros((num_zs, max_xs), dtype=np.float64)
    bounds = np.zeros((num_zs, max_xs, 4), dtype=np.float64)

    raw_grids: list[np.ndarray] = []
    log_rs_ref: np.ndarray | None = None
    log_ts_ref: np.ndarray | None = None
    grid_idx_map = np.zeros((num_zs, max_xs), dtype=np.int32)

    for iz, (_, x_list) in enumerate(z_list):
        for ix, (start, _) in enumerate(x_list):
            p = start + 1
            (x_v, _z_v, lr_min, lr_max, lt_min, lt_max) = struct.unpack_from(
                "<6d", mv, p
            )
            p += 48
            nlr, nlt = struct.unpack_from("<2I", mv, p)
            p += 8
            lrs = np.frombuffer(mv, dtype="<f8", count=nlr, offset=p).copy()
            p += nlr * 8
            lts = np.frombuffer(mv, dtype="<f8", count=nlt, offset=p).copy()
            p += nlt * 8
            vals = (
                np.frombuffer(mv, dtype="<i2", count=nlr * nlt, offset=p)
                .astype(np.float64)
                .reshape(nlt, nlr)
                / 1000.0
            )
            if log_rs_ref is None:
                log_rs_ref, log_ts_ref = lrs, lts
            xs_grid[iz, ix] = x_v
            bounds[iz, ix] = [lr_min, lr_max, lt_min, lt_max]
            grid_idx_map[iz, ix] = len(raw_grids)
            raw_grids.append(vals)
        for ix in range(len(x_list), max_xs):
            xs_grid[iz, ix] = xs_grid[iz, len(x_list) - 1]
            bounds[iz, ix] = bounds[iz, len(x_list) - 1]
            grid_idx_map[iz, ix] = grid_idx_map[iz, len(x_list) - 1]

    assert log_rs_ref is not None and log_ts_ref is not None
    coeffs_flat, ili_r, ili_t = fit_bicubic_batch(
        log_rs_ref, log_ts_ref, np.stack(raw_grids, axis=0)
    )
    coeffs = coeffs_flat[grid_idx_map]
    with jax.ensure_compile_time_eval():
        res = {
            "zs": jnp.asarray(zs),
            "num_xs": jnp.asarray(num_xs),
            "xs": jnp.asarray(xs_grid),
            "bounds": jnp.asarray(bounds),
            "log_rs": jnp.asarray(log_rs_ref),
            "log_ts": jnp.asarray(log_ts_ref),
            "ili_r": ili_r,
            "ili_t": ili_t,
            "coeffs": jnp.asarray(coeffs),
        }
    _FIXED_CACHE[canon] = res
    return res


def get_potekhin_jax() -> dict[str, Any]:
    """Parse and spline-fit `condtabl.data` with `ZeroFirstDeriv` (`bc=3`)."""
    global _POTEKHIN_CACHE
    if _POTEKHIN_CACHE is not None:
        return _POTEKHIN_CACHE
    roots = [pathlib.Path.cwd(), pathlib.Path(__file__).resolve()]
    cond_path = None
    for root in roots:
        for anc in [root, *root.parents]:
            cand = anc / "crates" / "flexmesa-kap" / "data" / "condtabl.data"
            if cand.is_file():
                cond_path = cand
                break
    assert cond_path is not None
    lines = cond_path.read_text(encoding="utf-8").splitlines()[1:]
    log_rhos = np.zeros(71, dtype=np.float64)
    log_ts = np.zeros(29, dtype=np.float64)
    log_zs = np.zeros(15, dtype=np.float64)
    raw = np.zeros((15, 29, 71), dtype=np.float64)
    pos = 0
    for iz in range(15):
        ztoks = [float(x) for x in lines[pos].split()]
        pos += 1
        log_zs[iz] = 0.0 if abs(ztoks[0] - 1.0) == 0.0 else np.log10(ztoks[0])
        log_ts[:] = ztoks[1:30]
        for ir in range(71):
            rtoks = [float(x) for x in lines[pos].split()]
            pos += 1
            log_rhos[ir] = rtoks[0]
            raw[iz, :, ir] = rtoks[1:30]
    coeffs, _, _ = fit_bicubic_batch(log_rhos, log_ts, raw, bc=3)
    with jax.ensure_compile_time_eval():
        _POTEKHIN_CACHE = {
            "log_rhos": jnp.asarray(log_rhos),
            "log_ts": jnp.asarray(log_ts),
            "log_zs": jnp.asarray(log_zs),
            "coeffs": jnp.asarray(coeffs),
        }
    return _POTEKHIN_CACHE
