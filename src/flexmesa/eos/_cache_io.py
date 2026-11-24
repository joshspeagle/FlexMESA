# SPDX-License-Identifier: GPL-3.0-or-later
"""Verified slice loaders for JAX EOS tables via ``flexmesa._rust``."""

from __future__ import annotations

import collections

import numpy as np

from flexmesa import _rust

HELM_IMAX: int = 2701
HELM_JMAX: int = 1001
HELM_LOGTLO: float = 3.0
HELM_LOGTHI: float = 13.0
HELM_LOGDLO: float = -12.0
HELM_LOGDHI: float = 15.0

_MAX_DT_SLICES: int = 12
_MAX_CMS_SLICES: int = 4

_DT_SLICE_CACHE: collections.OrderedDict[
    tuple[int, int, int],
    tuple[float, float, int, float, float, int, np.ndarray],
] = collections.OrderedDict()
_CMS_SLICE_CACHE: collections.OrderedDict[
    int, tuple[float, float, int, float, float, int, np.ndarray]
] = collections.OrderedDict()


def load_dt_slice_jax(
    kind: int, ix_1b: int, iz_1b: int, stem: str
) -> tuple[float, float, int, float, float, int, np.ndarray]:
    """Load `(lq_min, dq, n_q, lt_min, dt, n_t, coeffs)` for 4D DT tables."""
    key = (kind, ix_1b, iz_1b)
    if key in _DT_SLICE_CACHE:
        _DT_SLICE_CACHE.move_to_end(key)
        return _DT_SLICE_CACHE[key]

    lq_min, dq, n_q, lt_min, dt, n_t, flat = _rust.eos_dt_slice_coeffs(
        int(kind), int(ix_1b), int(iz_1b)
    )
    raw = np.asarray(flat, dtype=np.float64)
    if raw.size != n_q * n_t * 64:
        msg = (
            f"Unexpected coefficient array size for {stem}: "
            f"expected {n_q * n_t * 64}, got {raw.size}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(raw)):
        msg = f"Non-finite spline coefficient in verified slice {stem}"
        raise ValueError(msg)
    arr = raw.reshape(n_q, n_t, 16, 4)
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

    lt_min, dt, n_t, lr_min, dr, n_r, flat = _rust.eos_cms_slice_coeffs(
        int(ix_1b)
    )
    raw = np.asarray(flat, dtype=np.float64)
    if raw.size != n_t * n_r * 64:
        msg = (
            f"Unexpected CMS coefficient size for {stem}: "
            f"expected {n_t * n_r * 64}, got {raw.size}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(raw)):
        msg = f"Non-finite spline coefficient in verified CMS slice {stem}"
        raise ValueError(msg)
    arr = raw.reshape(n_t, n_r, 16, 4)
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
