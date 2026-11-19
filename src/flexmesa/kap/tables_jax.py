# SPDX-License-Identifier: LGPL-3.0-or-later
"""On-demand binary opacity table loader and bicubic spline cache for JAX.

Reads the immutable `FMKAP001` binary payload from
`$FLEXMESA_DATA_DIR/kap_full/kap_tables_full.bin` (installed by
`flexmesa data fetch` without invoking any runtime table builder) and fits 2D
PSPLINE bicubic spline coefficients `[f, f_xx, f_yy, f_xxyy]` on demand for
native `@jax.jit` evaluation.
"""

from __future__ import annotations

import mmap
import pathlib
import struct
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa.data import fetch

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

_RAW_FILE: Any = None
_RAW_MMAP: mmap.mmap | None = None
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
    if not np.all(np.isfinite(log_rs)) or not np.all(np.isfinite(log_ts)):
        msg = "Non-finite grid coordinate in JAX kap table fit"
        raise ValueError(msg)
    nx = int(log_rs.shape[0])
    ny = int(log_ts.shape[0])
    bsz = int(raw_lt_lr.shape[0])
    f = np.swapaxes(raw_lt_lr, -2, -1).astype(np.float64, copy=True)
    if not np.all(np.isfinite(f)):
        bad = tuple(int(idx) for idx in np.argwhere(~np.isfinite(f))[0])
        msg = (
            "Non-finite raw log10(kappa) in JAX kap table at "
            f"(slice, ir, it)={bad}"
        )
        raise ValueError(msg)
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
    if not np.all(np.isfinite(coeffs)):
        msg = "Non-finite bicubic spline coefficient in JAX kap table fit"
        raise ValueError(msg)
    return coeffs, splinck(log_rs), splinck(log_ts)


def _find_kap_bin_path() -> pathlib.Path:
    """Resolve immutable `kap_tables_full.bin` inside `$FLEXMESA_DATA_DIR`."""
    cand = fetch.resolve_data_dir() / "kap_full" / "kap_tables_full.bin"
    if cand.is_file():
        return cand
    msg = (
        f"FlexMESA opacity tables not found at {cand}. "
        "Run 'flexmesa data fetch' (or set FLEXMESA_DATA_DIR) to download "
        "the verified data-v1 physics table bundle."
    )
    raise FileNotFoundError(msg)


def reset_index_cache() -> None:
    """Release `FMKAP001` mmap handles and clear parsed index caches."""
    global _RAW_FILE, _RAW_MMAP, _RAW_BYTES
    if _RAW_BYTES is not None:
        _RAW_BYTES.release()
        _RAW_BYTES = None
    if _RAW_MMAP is not None:
        _RAW_MMAP.close()
        _RAW_MMAP = None
    if _RAW_FILE is not None:
        _RAW_FILE.close()
        _RAW_FILE = None
    FIXED_INDEX.clear()
    CO_INDEX.clear()
    AESOPUS_META.clear()
    KAPCN_META.clear()
    _FIXED_CACHE.clear()


def ensure_indexed() -> memoryview:
    """Index and validate byte offsets across `FMKAP001` once."""
    global _RAW_FILE, _RAW_MMAP, _RAW_BYTES
    if _RAW_BYTES is not None:
        return _RAW_BYTES
    kap_path = _find_kap_bin_path()
    fh = kap_path.open("rb")
    mm: mmap.mmap | None = None
    mv: memoryview | None = None
    try:
        fh.seek(0, 2)
        total_len = fh.tell()
        fh.seek(0)
        if total_len < 8:
            msg = f"Truncated kap_tables_full.bin ({total_len} bytes)"
            raise ValueError(msg)
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        if mm[:8] != b"FMKAP001":
            msg = "Invalid header in kap_tables binary"
            raise ValueError(msg)
        mv = memoryview(mm)
        pos = 8

        def skip_checked(n: int) -> None:
            nonlocal pos
            if n < 0 or pos + n > total_len:
                msg = "Unexpected EOF in kap_tables_full.bin"
                raise ValueError(msg)
            pos += n

        def u8() -> int:
            nonlocal pos
            if pos + 1 > total_len:
                msg = "Unexpected EOF in kap_tables_full.bin"
                raise ValueError(msg)
            v = int(mv[pos])
            pos += 1
            return v

        def u32() -> int:
            nonlocal pos
            if pos + 4 > total_len:
                msg = "Unexpected EOF in kap_tables_full.bin"
                raise ValueError(msg)
            (v,) = struct.unpack_from("<I", mv, pos)
            pos += 4
            return int(v)

        def f64(label: str) -> float:
            nonlocal pos
            if pos + 8 > total_len:
                msg = f"Unexpected EOF reading {label} in kap_tables_full.bin"
                raise ValueError(msg)
            (v,) = struct.unpack_from("<d", mv, pos)
            pos += 8
            fv = float(v)
            if not np.isfinite(fv):
                msg = (
                    f"Non-finite f64 ({fv}) in kap_tables_full.bin for {label}"
                )
                raise ValueError(msg)
            return fv

        def f64_arr(count: int, label: str) -> np.ndarray:
            nonlocal pos
            nbytes = count * 8
            if count < 0 or pos + nbytes > total_len:
                msg = f"Unexpected EOF reading {label} in kap_tables_full.bin"
                raise ValueError(msg)
            arr = np.frombuffer(mv, dtype="<f8", count=count, offset=pos).copy()
            pos += nbytes
            if not np.all(np.isfinite(arr)):
                msg = f"Non-finite f64 in kap_tables_full.bin for {label}"
                raise ValueError(msg)
            return arr

        def string() -> str:
            nonlocal pos
            ln = u32()
            if pos + ln > total_len:
                msg = "Unexpected EOF reading string in kap_tables_full.bin"
                raise ValueError(msg)
            s = bytes(mv[pos : pos + ln]).decode("utf-8")
            pos += ln
            return s

        local_fixed: dict[str, list[tuple[float, list[tuple[int, int]]]]] = {}
        for _ in range(u32()):
            prefix = string()
            z_list = []
            for _ in range(u32()):
                z_val = f64(f"{prefix}.z")
                x_list = []
                for _ in range(u32()):
                    start = pos
                    present = u8() != 0
                    _ = f64(f"{prefix}.x")
                    _ = f64(f"{prefix}.z_val")
                    if present:
                        _ = f64_arr(4, f"{prefix}.bounds")
                        nlr, nlt = u32(), u32()
                        _ = f64_arr(nlr, f"{prefix}.log_rs")
                        _ = f64_arr(nlt, f"{prefix}.log_ts")
                        skip_checked(nlr * nlt * 2)
                    x_list.append((start, pos))
                z_list.append((z_val, x_list))
            local_fixed[prefix] = z_list

        local_co: dict[
            str,
            list[tuple[float, tuple[float, float, float, float], list[int]]],
        ] = {}
        for _ in range(u32()):
            prefix = string()
            z_list_co = []
            for _ in range(u32()):
                zbase = f64(f"{prefix}.zbase")
                zfracs = (
                    f64(f"{prefix}.zfrac_c"),
                    f64(f"{prefix}.zfrac_n"),
                    f64(f"{prefix}.zfrac_o"),
                    f64(f"{prefix}.zfrac_ne"),
                )
                x_offsets = []
                for _ in range(u32()):
                    start = pos
                    present = u8() != 0
                    _ = f64(f"{prefix}.x")
                    _ = f64(f"{prefix}.z")
                    if present:
                        nco, _ = u32(), u32()
                        _ = f64_arr(4, f"{prefix}.bounds")
                        nlr, nlt = u32(), u32()
                        _ = f64_arr(nlr, f"{prefix}.log_rs")
                        _ = f64_arr(nlt, f"{prefix}.log_ts")
                        for _ in range(nco):
                            _ = f64(f"{prefix}.dxc")
                            _ = f64(f"{prefix}.dxo")
                            if u8() != 0:
                                skip_checked(nlr * nlt * 2)
                    x_offsets.append(start)
                z_list_co.append((zbase, zfracs, x_offsets))
            local_co[prefix] = z_list_co

        zsun = f64("aesopus.zsun")
        fco_ref = f64("aesopus.fco_ref")
        fc_ref = f64("aesopus.fc_ref")
        fn_ref = f64("aesopus.fn_ref")
        nlr_a, nlt_a, nzs_a = u32(), u32(), u32()
        log_rs_a = f64_arr(nlr_a, "aesopus.log_rs")
        log_ts_a = f64_arr(nlt_a, "aesopus.log_ts")
        zs_a = f64_arr(nzs_a, "aesopus.zs")
        if nzs_a > 0 and np.any(zs_a <= 0.0):
            msg = "Non-positive metallicity in aesopus.zs"
            raise ValueError(msg)
        ts_meta: list[dict[str, Any]] = []
        with jax.ensure_compile_time_eval():
            for _ in range(nzs_a):
                nx_a, nfco_a, nfc_a, nfn_a = u32(), u32(), u32(), u32()
                xs_a = f64_arr(nx_a, "aesopus.xs")
                fcos_a = f64_arr(nfco_a, "aesopus.fcos")
                fcs_a = f64_arr(nfc_a, "aesopus.fcs")
                fns_a = f64_arr(nfn_a, "aesopus.fns")
                z_present = u8() != 0
                total_slices = nx_a * nfco_a * nfc_a * nfn_a
                slice_start = pos
                if z_present:
                    for _ in range(total_slices):
                        if u8() != 0:
                            skip_checked(nlr_a * nlt_a * 4)
                ts_meta.append(
                    {
                        "xs": jnp.asarray(xs_a),
                        "fcos": jnp.asarray(fcos_a),
                        "fcs": jnp.asarray(fcs_a),
                        "fns": jnp.asarray(fns_a),
                        "total_slices": total_slices,
                        "offset": slice_start,
                        "z_present": z_present,
                    }
                )
            empty_2d = jnp.zeros((0, 0), dtype=jnp.float64)
            local_aesopus = {
                "zsun": zsun,
                "fco_ref": fco_ref,
                "fc_ref": fc_ref,
                "fn_ref": fn_ref,
                "log_rs": jnp.asarray(log_rs_a),
                "log_ts": jnp.asarray(log_ts_a),
                "log_rs_np": log_rs_a,
                "log_ts_np": log_ts_a,
                "zs": jnp.asarray(zs_a),
                "log_zs": (
                    jnp.asarray(np.log10(zs_a))
                    if nzs_a > 0
                    else jnp.zeros((0,), dtype=jnp.float64)
                ),
                "ilinx": splinck(log_rs_a),
                "iliny": splinck(log_ts_a),
                "ts": ts_meta,
                "xs_all": (
                    jnp.stack([t["xs"] for t in ts_meta], axis=0)
                    if ts_meta
                    else empty_2d
                ),
                "fcos_all": (
                    jnp.stack([t["fcos"] for t in ts_meta], axis=0)
                    if ts_meta
                    else empty_2d
                ),
                "fcs_all": (
                    jnp.stack([t["fcs"] for t in ts_meta], axis=0)
                    if ts_meta
                    else empty_2d
                ),
                "fns_all": (
                    jnp.stack([t["fns"] for t in ts_meta], axis=0)
                    if ts_meta
                    else empty_2d
                ),
            }

        ncn_z, ncn_lr, ncn_lt = u32(), u32(), u32()
        cn_lrs = f64_arr(ncn_lr, "kapcn.log_rs")
        cn_lts = f64_arr(ncn_lt, "kapcn.log_ts")
        kapcn_offset = pos
        for iz in range(ncn_z):
            _ = f64(f"kapcn.z[{iz}]")
            _ = f64_arr(7, f"kapcn.z[{iz}].fcs")
            _ = f64_arr(3, f"kapcn.z[{iz}].fns")
            skip_checked(63 * ncn_lr * ncn_lt * 2)
        if pos != total_len:
            msg = (
                f"Unexpected trailing {total_len - pos} bytes in "
                "kap_tables_full.bin"
            )
            raise ValueError(msg)
        local_kapcn = {
            "num_zs": ncn_z,
            "log_rs_np": cn_lrs,
            "log_ts_np": cn_lts,
            "offset": kapcn_offset,
        }
    except Exception as exc:
        if mv is not None:
            mv.release()
        if mm is not None:
            mm.close()
        fh.close()
        if isinstance(exc, (struct.error, IndexError, UnicodeDecodeError)):
            msg = f"Corrupt kap_tables_full.bin payload: {exc}"
            raise ValueError(msg) from exc
        raise

    FIXED_INDEX.clear()
    FIXED_INDEX.update(local_fixed)
    CO_INDEX.clear()
    CO_INDEX.update(local_co)
    AESOPUS_META.clear()
    AESOPUS_META.update(local_aesopus)
    KAPCN_META.clear()
    KAPCN_META.update(local_kapcn)
    _RAW_FILE = fh
    _RAW_MMAP = mm
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
    grid_idx_map = np.full((num_zs, max_xs), -1, dtype=np.int32)

    for iz, (_, x_list) in enumerate(z_list):
        for ix, (start, _) in enumerate(x_list):
            p = start + 1
            x_v, _z_v = struct.unpack_from("<2d", mv, p)
            xs_grid[iz, ix] = x_v
            if mv[start] == 0:
                continue
            lr_min, lr_max, lt_min, lt_max = struct.unpack_from(
                "<4d", mv, p + 16
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
            bounds[iz, ix] = [lr_min, lr_max, lt_min, lt_max]
            grid_idx_map[iz, ix] = len(raw_grids)
            raw_grids.append(vals)
        for ix in range(len(x_list), max_xs):
            xs_grid[iz, ix] = xs_grid[iz, len(x_list) - 1]
            bounds[iz, ix] = bounds[iz, len(x_list) - 1]
            grid_idx_map[iz, ix] = grid_idx_map[iz, len(x_list) - 1]

    if log_rs_ref is None or log_ts_ref is None:
        log_rs_ref = np.linspace(-8.0, 1.0, 37, dtype=np.float64)
        log_ts_ref = np.linspace(3.75, 8.70, 138, dtype=np.float64)
        raw_grids.append(np.zeros((138, 37), dtype=np.float64))
    default_bounds = [
        float(log_rs_ref[0]),
        float(log_rs_ref[-1]),
        float(log_ts_ref[0]),
        float(log_ts_ref[-1]),
    ]
    for iz in range(num_zs):
        for ix in range(max_xs):
            if grid_idx_map[iz, ix] < 0:
                bounds[iz, ix] = default_bounds
                grid_idx_map[iz, ix] = 0
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
