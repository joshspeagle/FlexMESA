# SPDX-License-Identifier: LGPL-3.0-or-later
"""Portable `.mesa_temp_cache` binary cache loaders for JAX EOS tables."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

from flexmesa import _rust

HELM_IMAX: int = 2701
HELM_JMAX: int = 1001
HELM_LOGTLO: float = 3.0
HELM_LOGTHI: float = 13.0
HELM_LOGDLO: float = -12.0
HELM_LOGDHI: float = 15.0

_PERM_21: tuple[int, ...] = (
    0,
    2,
    4,
    1,
    3,
    5,
    6,
    7,
    8,
    9,
    11,
    10,
    12,
    13,
    15,
    14,
    16,
    17,
    19,
    18,
    20,
)

_HELM_GRID_SINGLETON: (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
) = None


def _cwd_roots() -> list[Path]:
    roots: list[Path] = []
    cwd = Path.cwd()
    for anc in [cwd, *cwd.parents]:
        if anc not in roots:
            roots.append(anc)
    return roots


def _repo_roots() -> list[Path]:
    roots: list[Path] = []
    here = Path(__file__).resolve()
    for anc in [Path.cwd(), *here.parents]:
        if anc not in roots:
            roots.append(anc)
    return roots


def resolve_mesa_data_dir() -> Path | None:
    """Locate upstream MESA `data/` from environment or relative cache."""
    for env_key in ("FLEXMESA_EOS_DATA_DIR", "FLEXMESA_DATA_DIR"):
        val = os.environ.get(env_key)
        if val:
            p = Path(val)
            if (p / "eosDT_data").is_dir():
                return p
            if (p / "data" / "eosDT_data").is_dir():
                return p / "data"
    mesa_dir = os.environ.get("MESA_DIR")
    if mesa_dir:
        p = Path(mesa_dir) / "data"
        if (p / "eosDT_data").is_dir():
            return p
    for root in _cwd_roots():
        cand = root / ".mesa_temp_cache" / "mesa_data"
        if (cand / "eosDT_data").is_dir():
            return cand
    return None


def resolve_bin_cache_dir() -> Path | None:
    """Locate the binary cache directory (`.mesa_temp_cache/bin_cache`)."""
    val = os.environ.get("FLEXMESA_BIN_CACHE_DIR")
    if val:
        p = Path(val)
        p.mkdir(parents=True, exist_ok=True)
        return p
    for root in _cwd_roots():
        cand = root / ".mesa_temp_cache" / "bin_cache"
        if cand.is_dir() or cand.is_symlink():
            return cand
        parent = root / ".mesa_temp_cache"
        if parent.is_dir():
            cand.mkdir(parents=True, exist_ok=True)
            return cand
    return None


def _helm_exp10_vec(x: np.ndarray) -> np.ndarray:
    """Match Fortran MESA `exp10` on uniform `0.01` log10 grid points."""
    out = np.exp(x * np.log(10.0))
    is_int = np.isfinite(x) & (x == np.floor(x)) & (x >= -308.0) & (x <= 308.0)
    for idx in np.flatnonzero(is_int):
        ix = int(x[idx])
        out[idx] = (1.0 / (10.0 ** (-ix))) if ix < 0 else (10.0**ix)
    return out


def get_helm_np_arrays() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Load `(t_arr, dt_sav, d_arr, dd_sav, vertices)` as host NumPy arrays."""
    global _HELM_GRID_SINGLETON
    if _HELM_GRID_SINGLETON is not None:
        return _HELM_GRID_SINGLETON

    log_t = HELM_LOGTLO + np.arange(HELM_JMAX, dtype=np.float64) * 0.01
    log_d = HELM_LOGDLO + np.arange(HELM_IMAX, dtype=np.float64) * 0.01
    t_np = _helm_exp10_vec(log_t)
    d_np = _helm_exp10_vec(log_d)
    dt_np = t_np[1:] - t_np[:-1]
    dd_np = d_np[1:] - d_np[:-1]

    # Trigger Rust `HelmTable::builtin()` once so `helm_table_full.bin` exists
    _rust.eos_helm_cells_batch(
        np.array([6.0], dtype=np.float64), np.array([1.0], dtype=np.float64)
    )

    bin_dir = resolve_bin_cache_dir()
    full_bin = (
        (bin_dir / "helm_table_full.bin") if bin_dir is not None else None
    )
    vertices_np: np.ndarray | None = None

    if full_bin is not None and full_bin.is_file():
        with full_bin.open("rb") as fh:
            hdr = fh.read(16)
        if len(hdr) == 16 and hdr[:8] == b"FMHELM01":
            imax, jmax = struct.unpack("<II", hdr[8:16])
            if imax == HELM_IMAX and jmax == HELM_JMAX:
                raw = np.fromfile(
                    full_bin,
                    dtype=np.float64,
                    count=HELM_JMAX * HELM_IMAX * 21,
                    offset=16,
                )
                if raw.size == HELM_JMAX * HELM_IMAX * 21:
                    vertices_np = np.ascontiguousarray(
                        raw.reshape(HELM_JMAX, HELM_IMAX, 21)[:, :, _PERM_21]
                    )

    if vertices_np is None:
        vertices_np = np.zeros((HELM_JMAX, HELM_IMAX, 21), dtype=np.float64)
        for root in _repo_roots():
            patch_path = (
                root / "crates" / "flexmesa-eos" / "data" / "helm_patches.bin"
            )
            if patch_path.is_file():
                raw_bytes = patch_path.read_bytes()
                (n_patches,) = struct.unpack_from("<I", raw_bytes, 0)
                stride = 4 + 21 * 8
                for k in range(n_patches):
                    off = 4 + k * stride
                    i_1b, j_1b = struct.unpack_from("<HH", raw_bytes, off)
                    vals = struct.unpack_from("<21d", raw_bytes, off + 4)
                    vertices_np[j_1b - 1, i_1b - 1, :] = [
                        vals[m] for m in _PERM_21
                    ]
                break

    _HELM_GRID_SINGLETON = (t_np, dt_np, d_np, dd_np, vertices_np)
    return _HELM_GRID_SINGLETON


_DT_SLICE_CACHE: dict[
    tuple[int, int, int],
    tuple[float, float, int, float, float, int, np.ndarray],
] = {}
_CMS_SLICE_CACHE: dict[
    int, tuple[float, float, int, float, float, int, np.ndarray]
] = {}


def _read_4d_patches_bytes() -> bytes | None:
    for root in _repo_roots():
        p = (
            root
            / "crates"
            / "flexmesa-eos"
            / "data"
            / "eos_tables_4d_patches.bin"
        )
        if p.is_file():
            return p.read_bytes()
    return None


def _build_dt_slice_via_rust(kind: int, stem: str) -> None:
    """Trigger the Rust 4D EOS loader to build and cache `stem.bin`."""
    try:
        xz_tok = stem.split("_")[1]
        z_str, x_part = xz_tok.split("z")
        z_val = int(z_str) / 100.0
        x_val = int(x_part.rstrip("x")) / 100.0
        y_val = max(0.0, (1.0 - x_val - z_val) - 1.0e-15)
        xa = np.array(
            [x_val, y_val, max(0.0, 1.0 - x_val - y_val)], dtype=np.float64
        )
        fn = _rust.eos_eval_opal_scvh if kind == 0 else _rust.eos_eval_free_eos
        fn(1.0e-2, 1.0e5, [2, 7, 40], xa)
    except Exception:
        pass


def _build_cms_slice_via_rust(ix_1b: int) -> None:
    """Trigger the Rust CMS EOS loader to build and cache slice `ix_1b`."""
    try:
        x_val = min(1.0, max(0.0, (ix_1b - 1) * 0.1))
        xa = np.array([x_val, 1.0 - x_val], dtype=np.float64)
        _rust.eos_eval_cms(1.0e-2, 1.0e5, [2, 7], xa)
    except Exception:
        pass


def load_dt_slice_jax(
    kind: int, ix_1b: int, iz_1b: int, stem: str
) -> tuple[float, float, int, float, float, int, np.ndarray]:
    """Load `(lq_min, dq, n_q, lt_min, dt, n_t, coeffs)` for 4D DT tables."""
    key = (kind, ix_1b, iz_1b)
    if key in _DT_SLICE_CACHE:
        return _DT_SLICE_CACHE[key]

    bin_dir = resolve_bin_cache_dir()
    if bin_dir is not None:
        s_bin = bin_dir / "eos_4d_slices" / f"{stem}.bin"
        if not s_bin.is_file() and resolve_mesa_data_dir() is not None:
            _build_dt_slice_via_rust(kind, stem)
        if s_bin.is_file():
            with s_bin.open("rb") as fh:
                hdr = fh.read(84)
            if len(hdr) == 84 and hdr[:8] == b"FMDT4D01":
                _, _, _, n_t, lt_min, _, dt, n_q, lq_min, _, dq = struct.unpack(
                    "<iddIdddIddd", hdr[8:84]
                )
                raw = np.fromfile(
                    s_bin, dtype=np.float64, count=n_q * n_t * 64, offset=84
                )
                if raw.size == n_q * n_t * 64:
                    arr = np.nan_to_num(
                        raw.reshape(n_q, n_t, 16, 4),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    res = (
                        float(lq_min),
                        float(dq),
                        int(n_q),
                        float(lt_min),
                        float(dt),
                        int(n_t),
                        arr,
                    )
                    _DT_SLICE_CACHE[key] = res
                    return res

    data_dir = resolve_mesa_data_dir()
    if data_dir is not None:
        subdir = "eosDT_data" if kind == 0 else "eosFreeEOS_data"
        m_stem = stem if stem.startswith("mesa-") else f"mesa-{stem}"
        for c_path in (
            data_dir / "eosDT_data" / "cache" / f"{m_stem}.bin",
            data_dir / subdir / "cache" / f"{m_stem}.bin",
            data_dir / subdir / "cache" / f"{stem}.bin",
        ):
            if c_path.is_file():
                with c_path.open("rb") as fh:
                    hdr = fh.read(88)
                if len(hdr) == 88:
                    (
                        r1_len,
                        _,
                        _,
                        n_t,
                        lt_min,
                        _,
                        dt,
                        n_q,
                        lq_min,
                        _,
                        dq,
                        _,
                        r1_tail,
                        r2_len,
                    ) = struct.unpack("<IddidddidddiII", hdr)
                    if (
                        r1_len == 76
                        and r1_tail == 76
                        and r2_len == 4 * 26 * n_q * n_t * 8
                    ):
                        raw = np.fromfile(
                            c_path,
                            dtype=np.float64,
                            count=n_t * n_q * 26 * 4,
                            offset=88,
                        )
                        if raw.size == n_t * n_q * 26 * 4:
                            arr = np.ascontiguousarray(
                                np.nan_to_num(
                                    raw.reshape(n_t, n_q, 26, 4)[
                                        :, :, :16, :
                                    ].transpose(1, 0, 2, 3),
                                    nan=0.0,
                                    posinf=0.0,
                                    neginf=0.0,
                                )
                            )
                            res = (
                                float(lq_min),
                                float(dq),
                                int(n_q),
                                float(lt_min),
                                float(dt),
                                int(n_t),
                                arr,
                            )
                            _DT_SLICE_CACHE[key] = res
                            return res

    n_q, n_t = 527, 306
    lq_min, dq = -10.09, 0.03
    lt_min, dt = 2.10, 0.02
    arr = np.zeros((n_q, n_t, 16, 4), dtype=np.float64)
    raw_b = _read_4d_patches_bytes()
    if raw_b is not None and len(raw_b) >= 4:
        (cnt,) = struct.unpack_from("<I", raw_b, 0)
        stride = 8 + 64 * 8
        for k in range(cnt):
            off = 4 + k * stride
            if off + stride > len(raw_b):
                break
            k_b, ix_b, iz_b, _, iq, jt = struct.unpack_from(
                "<BBBBHH", raw_b, off
            )
            if k_b == kind and ix_b == ix_1b and iz_b == iz_1b:
                vals = struct.unpack_from("<64d", raw_b, off + 8)
                arr[iq - 1, jt - 1] = np.asarray(
                    vals, dtype=np.float64
                ).reshape(16, 4)
    res = (lq_min, dq, n_q, lt_min, dt, n_t, arr)
    _DT_SLICE_CACHE[key] = res
    return res


def load_cms_slice_jax(
    ix_1b: int, stem: str
) -> tuple[float, float, int, float, float, int, np.ndarray]:
    """Load `(lt_min, dt, n_t, lr_min, dr, n_r, coeffs)` for CMS tables."""
    if ix_1b in _CMS_SLICE_CACHE:
        return _CMS_SLICE_CACHE[ix_1b]

    bin_dir = resolve_bin_cache_dir()
    if bin_dir is not None:
        s_bin = bin_dir / "eos_4d_slices" / f"{stem}.bin"
        if not s_bin.is_file() and resolve_mesa_data_dir() is not None:
            _build_cms_slice_via_rust(ix_1b)
        if s_bin.is_file():
            with s_bin.open("rb") as fh:
                hdr = fh.read(76)
            if len(hdr) == 76 and hdr[:8] == b"FMCMS001":
                _, _, n_t, lt_min, _, dt, n_r, lr_min, _, dr = struct.unpack(
                    "<idIdddIddd", hdr[8:76]
                )
                raw = np.fromfile(
                    s_bin, dtype=np.float64, count=n_t * n_r * 64, offset=76
                )
                if raw.size == n_t * n_r * 64:
                    arr = np.nan_to_num(
                        raw.reshape(n_t, n_r, 16, 4),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    res = (
                        float(lt_min),
                        float(dt),
                        int(n_t),
                        float(lr_min),
                        float(dr),
                        int(n_r),
                        arr,
                    )
                    _CMS_SLICE_CACHE[ix_1b] = res
                    return res

    n_t, n_r = 121, 281
    lt_min, dt = 2.0, 0.05
    lr_min, dr = -8.0, 0.05
    arr = np.zeros((n_t, n_r, 16, 4), dtype=np.float64)
    raw_b = _read_4d_patches_bytes()
    if raw_b is not None and len(raw_b) >= 4:
        (dt_cnt,) = struct.unpack_from("<I", raw_b, 0)
        stride = 8 + 64 * 8
        cms_off = 4 + dt_cnt * stride
        if cms_off + 4 <= len(raw_b):
            (cms_cnt,) = struct.unpack_from("<I", raw_b, cms_off)
            for k in range(cms_cnt):
                off = cms_off + 4 + k * stride
                if off + stride > len(raw_b):
                    break
                ix_b, it, jrho, _ = struct.unpack_from("<HHHH", raw_b, off)
                if ix_b == ix_1b:
                    vals = struct.unpack_from("<64d", raw_b, off + 8)
                    arr[it - 1, jrho - 1] = np.asarray(
                        vals, dtype=np.float64
                    ).reshape(16, 4)
    res = (lt_min, dt, n_t, lr_min, dr, n_r, arr)
    _CMS_SLICE_CACHE[ix_1b] = res
    return res
