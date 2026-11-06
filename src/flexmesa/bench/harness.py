# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared timing harness and deterministic stellar fixtures for Wave 1F.

Provides `LatencyStats`, `BenchRecord`, `ThermoGrid`, `BicubTable`,
`measure_us`, `measure_and_capture`, and `make_block_tridiag_fixture` matching
`crates/flexmesa-bench`.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import interp_2d as _interp2d
from flexmesa import math as _fmath

STAGE3_BATCH_SIZE: int = 4


@dataclasses.dataclass(frozen=True)
class LatencyStats:
    """Summary latency statistics in microseconds for an optimization stage.

    Attributes:
        median_us: Median wall-clock execution time in microseconds ($\\mu$s).
        iqr_us: Interquartile range ($Q_{75} - Q_{25}$) in microseconds
            ($\\mu$s).
        compile_us: First-call XLA compilation latency in microseconds
            ($0.0$ for non-JIT stages).
    """

    median_us: float
    iqr_us: float
    compile_us: float = 0.0


@dataclasses.dataclass(frozen=True)
class BenchRecord:
    """Four-stage benchmark record for a single workload and zone count.

    Attributes:
        workload: Workload identifier (`"auto_diff"`, `"interp_2d"`,
            `"mtx_tridiag"`, `"neu"`, `"turb_mlt"`, or `"turb_tdc"`).
        n_zones: Number of stellar zones ($N_{\\mathrm{zones}}$).
        stage0: Stage 0 (sequential 1-thread `flexmesa._rust` PyO3) latency.
        stage1: Stage 1 (parallel 4-thread `rayon` `flexmesa._rust`) latency.
        stage2: Stage 2 (XLA `@jax.jit` fused kernel) latency.
        stage3: Stage 3 (batched `jax.vmap` ensemble, per-profile) latency.
        max_rel_err: Maximum relative discrepancy across Stages 0..3.
        checksum: Scalar `float64` checksum matching the Rust benchmark suite.
    """

    workload: str
    n_zones: int
    stage0: LatencyStats
    stage1: LatencyStats
    stage2: LatencyStats
    stage3: LatencyStats
    max_rel_err: float
    checksum: float


def measure_us(
    n_warmup: int,
    n_repeats: int,
    op: collections.abc.Callable[[], object],
    record_first_as_compile: bool = False,
) -> LatencyStats:
    """Measures median and IQR latency (in microseconds) of callable `op`.

    Args:
        n_warmup: Number of untimed warmup iterations ($\\ge 1$ when
            `record_first_as_compile` is `True`).
        n_repeats: Number of timed iterations ($\\ge 3$).
        op: Zero-argument callable to benchmark.
        record_first_as_compile: Whether to time the very first warmup call as
            `compile_us` (including XLA trace + compilation).

    Returns:
        `LatencyStats` containing `(median_us, iqr_us, compile_us)`.
    """
    compile_us = 0.0
    warm = max(n_warmup, 1 if record_first_as_compile else 0)
    for idx in range(warm):
        t0 = time.perf_counter_ns()
        res = op()
        if isinstance(res, jax.Array):
            res.block_until_ready()
        elapsed_us = (time.perf_counter_ns() - t0) * 1.0e-3
        if idx == 0 and record_first_as_compile:
            compile_us = elapsed_us

    repeats = max(n_repeats, 3)
    samples = np.empty(repeats, dtype=np.float64)
    for idx in range(repeats):
        t0 = time.perf_counter_ns()
        res = op()
        if isinstance(res, jax.Array):
            res.block_until_ready()
        samples[idx] = (time.perf_counter_ns() - t0) * 1.0e-3

    samples.sort()
    mid = repeats // 2
    if repeats % 2 == 1:
        median_us = float(samples[mid])
    else:
        median_us = float(0.5 * (samples[mid - 1] + samples[mid]))
    q25 = float(samples[repeats // 4])
    q75 = float(samples[(3 * repeats) // 4])
    return LatencyStats(
        median_us=median_us,
        iqr_us=q75 - q25,
        compile_us=compile_us,
    )


def measure_and_capture[T](
    n_warmup: int,
    n_repeats: int,
    op: collections.abc.Callable[[], T],
    record_first_as_compile: bool = False,
) -> tuple[LatencyStats, T]:
    """Measures latency of `op` and returns `(LatencyStats, last_result)`.

    Args:
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.
        op: Zero-argument callable returning a result of type `T`.
        record_first_as_compile: Whether to record first-call compile time.

    Returns:
        Tuple `(stats, result)` where `result` is the output of the final call.
    """
    box: list[T] = []

    def _wrapped() -> T:
        val = op()
        box.clear()
        box.append(val)
        return val

    stats = measure_us(
        n_warmup,
        n_repeats,
        _wrapped,
        record_first_as_compile=record_first_as_compile,
    )
    return stats, box[0]


def measure_stage3_batch[T](
    n_warmup: int,
    n_repeats: int,
    op_batch: collections.abc.Callable[[], T],
) -> tuple[LatencyStats, T]:
    """Measures a batched Stage 3 call and amortizes latency per profile.

    Args:
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.
        op_batch: Zero-argument callable executing a batch of
            `STAGE3_BATCH_SIZE` stellar profiles.

    Returns:
        Two-tuple `(amortized_stats, batch_result)` where `median_us` and
        `iqr_us` are divided by `STAGE3_BATCH_SIZE`.
    """
    raw, out = measure_and_capture(
        n_warmup,
        n_repeats,
        op_batch,
        record_first_as_compile=True,
    )
    scale = float(STAGE3_BATCH_SIZE)
    return (
        LatencyStats(
            median_us=raw.median_us / scale,
            iqr_us=raw.iqr_us / scale,
            compile_us=raw.compile_us,
        ),
        out,
    )


def max_rel_diff(
    reference: npt.ArrayLike,
    candidate: npt.ArrayLike,
) -> float:
    """Computes maximum elementwise relative difference between two arrays.

    Args:
        reference: Reference array-like input in `float64`.
        candidate: Candidate array-like input in `float64` of matching shape
            (or batched shape `(B, *reference.shape)`).

    Returns:
        Maximum relative discrepancy across all elements, or `NaN` if any
        entry is non-finite.

    Raises:
        ValueError: If `candidate` is not broadcast-compatible with
            `reference`.
    """
    a_arr = np.asarray(reference, dtype=np.float64)
    b_arr = np.asarray(candidate, dtype=np.float64)
    if b_arr.ndim == a_arr.ndim + 1 and b_arr.shape[1:] == a_arr.shape:
        a_arr = np.broadcast_to(a_arr, b_arr.shape)
    elif a_arr.shape != b_arr.shape:
        msg = f"shape mismatch in max_rel_diff: {a_arr.shape} vs {b_arr.shape}"
        raise ValueError(msg)
    if not (np.all(np.isfinite(a_arr)) and np.all(np.isfinite(b_arr))):
        return float("nan")
    denom = np.maximum(np.maximum(np.abs(a_arr), np.abs(b_arr)), 1.0e-30)
    return float(np.max(np.abs(a_arr - b_arr) / denom))


def max_err3(d1: float, d2: float, d3: float) -> float:
    """Combines three stage relative errors while propagating `NaN`.

    Args:
        d1: Stage 1 relative discrepancy.
        d2: Stage 2 relative discrepancy.
        d3: Stage 3 relative discrepancy.

    Returns:
        Maximum of `(d1, d2, d3)`, or `NaN` if any input is `NaN`.
    """
    return float(np.max([d1, d2, d3]))


@dataclasses.dataclass(frozen=True)
class ThermoGrid:
    """Deterministic stellar thermodynamic profile across $N_{\\mathrm{zones}}$.

    Attributes:
        n_zones: Number of stellar zones ($N_{\\mathrm{zones}}$).
        log10_t: Base-10 log temperature $\\log_{10}(T/\\mathrm{K})$, shape
            `(n_zones,)`.
        t: Temperature $T$ in $\\mathrm{K}$, shape `(n_zones,)`.
        log10_rho: Base-10 log density in $\\mathrm{g\\,cm^{-3}}$, shape
            `(n_zones,)`.
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$, shape
            `(n_zones,)`.
        abar: Mean atomic weight $\\bar{A}$ in $\\mathrm{amu}$, shape
            `(n_zones,)`.
        zbar: Mean ionic charge $\\bar{Z}$ in $e$, shape `(n_zones,)`.
    """

    n_zones: int
    log10_t: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]
    log10_rho: npt.NDArray[np.float64]
    rho: npt.NDArray[np.float64]
    abar: npt.NDArray[np.float64]
    zbar: npt.NDArray[np.float64]

    @classmethod
    def create(cls, n_zones: int) -> ThermoGrid:
        """Constructs the deterministic stellar profile for `n_zones` shells.

        Args:
            n_zones: Number of radial zones ($N_{\\mathrm{zones}} \\ge 1$).

        Returns:
            Initialized `ThermoGrid` instance with `float64` NumPy arrays.
        """
        denom = float(max(n_zones - 1, 1))
        s = np.arange(n_zones, dtype=np.float64) / denom
        log10_t = 7.2 + 2.6 * s
        t = np.power(10.0, log10_t)
        log10_rho = 4.0 + 5.0 * s
        rho = np.power(10.0, log10_rho)
        abar = 4.0 + 24.0 * s
        zbar = 0.5 * abar
        return cls(
            n_zones=n_zones,
            log10_t=log10_t,
            t=t,
            log10_rho=log10_rho,
            rho=rho,
            abar=abar,
            zbar=zbar,
        )


@dataclasses.dataclass(frozen=True)
class BicubTable:
    """Precomputed $64 \\times 64$ 2D bicubic spline table over $(\\rho, T)$.

    Attributes:
        x: Strictly ascending `x` grid of shape `(64,)` in $[3.5, 9.5]$.
        y: Strictly ascending `y` grid of shape `(64,)` in $[7.0, 10.0]$.
        coeffs: Bicubic spline coefficients `[f, f_xx, f_yy, f_xxyy]` of shape
            `(64, 64, 4)`.
        x_jax: Device-resident JAX `float64` `x` grid of shape `(64,)`.
        y_jax: Device-resident JAX `float64` `y` grid of shape `(64,)`.
        coeffs_jax: Device-resident JAX `float64` coefficients `(64, 64, 4)`.
        ilinx: Uniform-spacing flag along `x`.
        iliny: Uniform-spacing flag along `y`.
    """

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    coeffs: npt.NDArray[np.float64]
    x_jax: jax.Array
    y_jax: jax.Array
    coeffs_jax: jax.Array
    ilinx: bool
    iliny: bool

    @classmethod
    def create(cls) -> BicubTable:
        """Builds the deterministic $64 \\times 64$ bicubic spline table.

        Returns:
            Initialized `BicubTable` matching `crates/flexmesa-bench`.
        """
        nx, ny = 64, 64
        x = np.linspace(3.5, 9.5, nx, dtype=np.float64)
        y = np.linspace(7.0, 10.0, ny, dtype=np.float64)
        z = np.empty((nx, ny), dtype=np.float64)
        for i in range(nx):
            for j in range(ny):
                xi, yj = float(x[i]), float(y[j])
                z[i, j] = (
                    0.25 * xi * xi
                    + 0.15 * xi * yj
                    + 0.35 * yj * yj
                    + 0.1 * _fmath.sinpi(0.2 * xi) * _fmath.cospi(0.15 * yj)
                )
        coeffs_raw, ilinx, iliny = _interp2d.interp_mkbicub_db(x, y, z)
        coeffs = np.asarray(coeffs_raw, dtype=np.float64).reshape(nx, ny, 4)
        return cls(
            x=x,
            y=y,
            coeffs=coeffs,
            x_jax=jnp.asarray(x, dtype=jnp.float64),
            y_jax=jnp.asarray(y, dtype=jnp.float64),
            coeffs_jax=jnp.asarray(coeffs, dtype=jnp.float64),
            ilinx=bool(ilinx),
            iliny=bool(iliny),
        )


def make_block_tridiag_fixture(
    nz: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Constructs the $5 \\times 5$ block-tridiagonal system across `nz` shells.

    Args:
        nz: Number of stellar shells ($N_{\\mathrm{zones}}$).

    Returns:
        4-tuple `(lblk, dblk, ublk, rhs)` with block arrays of shape
        `(nz, 5, 5)` and right-hand side of shape `(nz, 5)` in `float64`.
    """
    nvar = 5
    s = np.arange(1, nz + 1, dtype=np.float64) / float(nz)
    idx_i = np.arange(nvar, dtype=np.float64)[:, None]
    idx_j = np.arange(nvar, dtype=np.float64)[None, :]

    off_diag = 0.05 / (np.abs(idx_i - idx_j) + 1.0)
    dblk = np.broadcast_to(off_diag, (nz, nvar, nvar)).copy()
    for i in range(nvar):
        dblk[:, i, i] = 4.0 + 0.5 * float(i) + 0.2 * s

    l_base = -0.12 / (idx_i + idx_j + 2.0)
    u_base = -0.10 / (idx_i + idx_j + 2.0)
    lblk = np.broadcast_to(l_base, (nz, nvar, nvar)).copy()
    lblk[0] = 0.0
    ublk = np.broadcast_to(u_base, (nz, nvar, nvar)).copy()
    ublk[-1] = 0.0

    rhs = (1.0 + 0.2 * np.arange(nvar, dtype=np.float64)[None, :]) - (
        0.1 * s[:, None]
    )
    return lblk, dblk, ublk, rhs
