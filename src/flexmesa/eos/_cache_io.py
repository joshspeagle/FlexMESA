# SPDX-License-Identifier: LGPL-3.0-or-later
"""Canonical binary cache loaders for JAX EOS tables (`DECISIONS #3`)."""

from __future__ import annotations

import collections
import os
import pathlib
import struct

import numpy as np

from flexmesa import _rust
from flexmesa.data import fetch

HELM_IMAX: int = 2701
HELM_JMAX: int = 1001
HELM_LOGTLO: float = 3.0
HELM_LOGTHI: float = 13.0
HELM_LOGDLO: float = -12.0
HELM_LOGDHI: float = 15.0

_MAX_DT_SLICES: int = 12
_MAX_CMS_SLICES: int = 4


def resolve_bin_cache_dir() -> pathlib.Path:
    """Resolve the canonical FlexMESA binary table directory."""
    for env_key in ("FLEXMESA_DATA_DIR", "FLEXMESA_BIN_CACHE_DIR"):
        if env_val := os.environ.get(env_key, "").strip():
            p = pathlib.Path(env_val).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
            return p
    p = fetch.resolve_data_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p


_DT_SLICE_CACHE: collections.OrderedDict[
    tuple[int, int, int],
    tuple[float, float, int, float, float, int, np.ndarray],
] = collections.OrderedDict()
_CMS_SLICE_CACHE: collections.OrderedDict[
    int, tuple[float, float, int, float, float, int, np.ndarray]
] = collections.OrderedDict()


def _build_dt_slice_via_rust(kind: int, stem: str) -> None:
    """Trigger the Rust 4D EOS loader to build and cache `stem.bin`."""
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


def _build_cms_slice_via_rust(ix_1b: int) -> None:
    """Trigger the Rust CMS EOS loader to build and cache slice `ix_1b`."""
    x_val = min(1.0, max(0.0, (ix_1b - 1) * 0.1))
    xa = np.array([x_val, 1.0 - x_val], dtype=np.float64)
    _rust.eos_eval_cms(1.0e-2, 1.0e5, [2, 7], xa)


def load_dt_slice_jax(
    kind: int, ix_1b: int, iz_1b: int, stem: str
) -> tuple[float, float, int, float, float, int, np.ndarray]:
    """Load `(lq_min, dq, n_q, lt_min, dt, n_t, coeffs)` for 4D DT tables."""
    key = (kind, ix_1b, iz_1b)
    if key in _DT_SLICE_CACHE:
        _DT_SLICE_CACHE.move_to_end(key)
        return _DT_SLICE_CACHE[key]

    bin_dir = resolve_bin_cache_dir()
    s_bin = bin_dir / "eos_4d_slices" / f"{stem}.bin"
    if not s_bin.is_file():
        _build_dt_slice_via_rust(kind, stem)
    if not s_bin.is_file():
        msg = (
            f"FlexMESA 4D EOS slice '{stem}' not found at {s_bin}. "
            "Run 'flexmesa data fetch' (or set FLEXMESA_DATA_DIR) to download "
            "the verified data-v1 physics table bundle."
        )
        raise FileNotFoundError(msg)

    with s_bin.open("rb") as fh:
        hdr = fh.read(84)
    if len(hdr) != 84 or hdr[:8] != b"FMDT4D01":
        msg = f"Invalid FMDT4D01 header in {s_bin}"
        raise ValueError(msg)
    _, _, _, n_t, lt_min, _, dt, n_q, lq_min, _, dq = struct.unpack(
        "<iddIdddIddd", hdr[8:84]
    )
    raw = np.fromfile(s_bin, dtype=np.float64, count=n_q * n_t * 64, offset=84)
    if raw.size != n_q * n_t * 64:
        msg = f"Truncated FMDT4D01 payload in {s_bin}"
        raise ValueError(msg)
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
    while len(_DT_SLICE_CACHE) > _MAX_DT_SLICES:
        _DT_SLICE_CACHE.popitem(last=False)
    return res


def load_cms_slice_jax(
    ix_1b: int, stem: str
) -> tuple[float, float, int, float, float, int, np.ndarray]:
    """Load `(lt_min, dt, n_t, lr_min, dr, n_r, coeffs)` for CMS tables."""
    if ix_1b in _CMS_SLICE_CACHE:
        _CMS_SLICE_CACHE.move_to_end(ix_1b)
        return _CMS_SLICE_CACHE[ix_1b]

    bin_dir = resolve_bin_cache_dir()
    s_bin = bin_dir / "eos_4d_slices" / f"{stem}.bin"
    if not s_bin.is_file():
        _build_cms_slice_via_rust(ix_1b)
    if not s_bin.is_file():
        msg = (
            f"FlexMESA CMS EOS slice '{stem}' not found at {s_bin}. "
            "Run 'flexmesa data fetch' (or set FLEXMESA_DATA_DIR) to download "
            "the verified data-v1 physics table bundle."
        )
        raise FileNotFoundError(msg)

    with s_bin.open("rb") as fh:
        hdr = fh.read(76)
    if len(hdr) != 76 or hdr[:8] != b"FMCMS001":
        msg = f"Invalid FMCMS001 header in {s_bin}"
        raise ValueError(msg)
    _, _, n_t, lt_min, _, dt, n_r, lr_min, _, dr = struct.unpack(
        "<idIdddIddd", hdr[8:76]
    )
    raw = np.fromfile(s_bin, dtype=np.float64, count=n_t * n_r * 64, offset=76)
    if raw.size != n_t * n_r * 64:
        msg = f"Truncated FMCMS001 payload in {s_bin}"
        raise ValueError(msg)
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
    while len(_CMS_SLICE_CACHE) > _MAX_CMS_SLICES:
        _CMS_SLICE_CACHE.popitem(last=False)
    return res
