# SPDX-License-Identifier: LGPL-3.0-or-later
"""Four-stage microphysics and utility kernel benchmarks (`flexmesa.bench`).

Evaluates the Three-Tier Hybrid Architecture across Stages 0..3 for:
1. `bench_autodiff`: Two-variable second-order forward-mode hyperdual via
   zero-copy `flexmesa._rust` (`parallel=False` Stage 0, `parallel=True`
   4-thread `rayon` Stage 1) vs. XLA `@jax.jit` (Stage 2) and `@jax.jit(vmap)`
   (Stage 3).
2. `bench_interp2d`: 2D bicubic spline evaluation (`interp_2d`) across
   `flexmesa._rust.interp_evbicub_batch` (1-thread Stage 0, 4-thread `rayon`
   Stage 1) and `@jax.jit` / `@jax.jit(vmap)` (Stages 2..3).
3. `bench_neu`: Thermal neutrino energy-loss rate and 4-variable Jacobian
   evaluation (`neu`) across `flexmesa._rust.neu_get_batch` (Stages 0..1) and
   `@jax.jit` / `@jax.jit(vmap)` (Stages 2..3).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust
from flexmesa import constants as _const
from flexmesa import neu as _neu
from flexmesa.bench import bicub_fast as _bicub
from flexmesa.bench import harness as _h


def _autodiff_primal_jax(xy: jax.Array) -> jax.Array:
    """Evaluates $f(x, y) = \\exp(0.1 x y) + x^{1.5} / (y + 2)$ in JAX.

    Args:
        xy: Two-element coordinate vector `[x, y]` of shape `(2,)` (`float64`).

    Returns:
        Scalar `float64` value $f(x, y)$.
    """
    x, y = xy[0], xy[1]
    return jnp.exp(0.1 * x * y) + jnp.power(x, 1.5) / (y + 2.0)


def _autodiff_point_jax(x: jax.Array, y: jax.Array) -> jax.Array:
    """Computes `[f, df_dx, df_dy, d2f_dx2, d2f_dxdy, d2f_dy2]` at `(x, y)`.

    Args:
        x: First independent variable ($\\log_{10}T$, scalar `float64`).
        y: Second independent variable ($\\log_{10}\\rho$, scalar `float64`).

    Returns:
        6-element `float64` JAX array `[f, df_dx, df_dy, d2f_dx2, d2f_dxdy,
        d2f_dy2]` of shape `(6,)`.
    """
    xy = jnp.stack([x, y])
    val, grad = jax.value_and_grad(_autodiff_primal_jax)(xy)
    hess = jax.hessian(_autodiff_primal_jax)(xy)
    return jnp.stack(
        [val, grad[0], grad[1], hess[0, 0], hess[0, 1], hess[1, 1]]
    )


_autodiff_vmap = jax.vmap(_autodiff_point_jax, in_axes=(0, 0))
_autodiff_jit = jax.jit(_autodiff_vmap)
_autodiff_batch_jit = jax.jit(jax.vmap(_autodiff_vmap, in_axes=(0, 0)))


def bench_autodiff(
    grid: _h.ThermoGrid,
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> _h.BenchRecord:
    """Benchmarks two-variable 2nd-order automatic differentiation.

    Args:
        grid: Stellar thermodynamic profile across `n_zones` shells.
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    nz = grid.n_zones
    x_c = _dispatch.as_c_f64(grid.log10_t)
    y_c = _dispatch.as_c_f64(grid.log10_rho)

    stage0, out0 = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _rust.eval_autodiff_order2_batch(x_c, y_c, False),
    )
    stage1, out1 = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _rust.eval_autodiff_order2_batch(x_c, y_c, True),
    )

    x_j = jnp.asarray(grid.log10_t, dtype=jnp.float64)
    y_j = jnp.asarray(grid.log10_rho, dtype=jnp.float64)
    stage2, out2_j = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _autodiff_jit(x_j, y_j),
        record_first_as_compile=True,
    )

    b_sz = _h.STAGE3_BATCH_SIZE
    x_b = jnp.broadcast_to(x_j[None, :], (b_sz, nz))
    y_b = jnp.broadcast_to(y_j[None, :], (b_sz, nz))
    stage3, out3_j = _h.measure_stage3_batch(
        n_warmup,
        n_repeats,
        lambda: _autodiff_batch_jit(x_b, y_b),
    )

    out2_np = np.asarray(out2_j, dtype=np.float64)
    out3_np = np.asarray(out3_j, dtype=np.float64)
    max_rel_err = _h.max_err3(
        _h.max_rel_diff(out0, out1),
        _h.max_rel_diff(out0, out2_np),
        _h.max_rel_diff(out0, out3_np),
    )
    return _h.BenchRecord(
        workload="auto_diff",
        n_zones=nz,
        stage0=stage0,
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        max_rel_err=max_rel_err,
        checksum=float(np.sum(out2_np[:, 0])),
    )


def bench_interp2d(
    grid: _h.ThermoGrid,
    table: _h.BicubTable,
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> _h.BenchRecord:
    """Benchmarks 2D bicubic spline evaluation (`interp_2d`) across Stages 0..3.

    Args:
        grid: Stellar thermodynamic profile across `n_zones` shells.
        table: Precomputed $64 \\times 64$ bicubic spline table.
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    nz = grid.n_zones
    xq_c = _dispatch.as_c_f64(grid.log10_rho)
    yq_c = _dispatch.as_c_f64(grid.log10_t)
    tx_c = _dispatch.as_c_f64(table.x)
    ty_c = _dispatch.as_c_f64(table.y)
    tc_c = _dispatch.as_c_f64(table.coeffs)

    stage0, out0 = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _rust.interp_evbicub_batch(xq_c, yq_c, tx_c, ty_c, tc_c, False),
    )
    stage1, out1 = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _rust.interp_evbicub_batch(xq_c, yq_c, tx_c, ty_c, tc_c, True),
    )

    xq_j = jnp.asarray(grid.log10_rho, dtype=jnp.float64)
    yq_j = jnp.asarray(grid.log10_t, dtype=jnp.float64)
    tx, ty, tc = table.x_jax, table.y_jax, table.coeffs_jax
    stage2, out2_j = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _bicub.bicub_jit(xq_j, yq_j, tx, ty, tc),
        record_first_as_compile=True,
    )
    b_sz = _h.STAGE3_BATCH_SIZE
    xq_b = jnp.broadcast_to(xq_j[None, :], (b_sz, nz))
    yq_b = jnp.broadcast_to(yq_j[None, :], (b_sz, nz))
    stage3, out3_j = _h.measure_stage3_batch(
        n_warmup,
        n_repeats,
        lambda: _bicub.bicub_batch_jit(xq_b, yq_b, tx, ty, tc),
    )
    out2_np = np.asarray(out2_j, dtype=np.float64)
    out3_np = np.asarray(out3_j, dtype=np.float64)
    max_rel_err = _h.max_err3(
        _h.max_rel_diff(out0, out1),
        _h.max_rel_diff(out0, out2_np),
        _h.max_rel_diff(out0, out3_np),
    )
    return _h.BenchRecord(
        workload="interp_2d",
        n_zones=nz,
        stage0=stage0,
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        max_rel_err=max_rel_err,
        checksum=float(np.sum(out2_np[:, 0])),
    )


def _neu_pair_plas_cutoff_jax(
    t: jax.Array,
    rho: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
) -> jax.Array:
    """Evaluates pair + plasmon neutrino loss with low-$T$ cosine cutoff.

    Args:
        t: Temperature $T$ in $\\mathrm{K}$ (`float64` scalar).
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$ (`float64` scalar).
        abar: Mean atomic weight $\\bar{A}$ in $\\mathrm{amu}$ (`float64`).
        zbar: Mean ionic charge $\\bar{Z}$ in $e$ (`float64`).

    Returns:
        Specific neutrino energy-loss rate $\\epsilon_\\nu$ in
        $\\mathrm{erg\\,g^{-1}\\,s^{-1}}$.
    """
    raw = _neu.neu_pair_plas_jax(t, rho, abar, zbar)
    log10_t = jnp.log10(t)
    t_min = _neu.LOG10_TMIN_NEU
    ratio = (log10_t - t_min) / 0.5
    f_cut = jnp.where(
        log10_t <= t_min,
        0.0,
        jnp.where(
            log10_t < t_min + 0.5,
            0.5 * (1.0 - jnp.cos(_const.PI * ratio)),
            1.0,
        ),
    )
    return raw * f_cut


def _neu_point_jax(
    t: jax.Array,
    rho: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
) -> jax.Array:
    """Computes `[val, d_t, d_rho, d_abar, d_zbar]` in JAX `float64`.

    Args:
        t: Temperature $T$ in $\\mathrm{K}$ (`float64` scalar).
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$ (`float64` scalar).
        abar: Mean atomic weight $\\bar{A}$ in $\\mathrm{amu}$ (`float64`).
        zbar: Mean ionic charge $\\bar{Z}$ in $e$ (`float64`).

    Returns:
        5-element `float64` JAX array `[val, d_t, d_rho, d_abar, d_zbar]`.
    """
    val, grads = jax.value_and_grad(
        _neu_pair_plas_cutoff_jax, argnums=(0, 1, 2, 3)
    )(t, rho, abar, zbar)
    return jnp.stack([val, grads[0], grads[1], grads[2], grads[3]])


_neu_vmap = jax.vmap(_neu_point_jax, in_axes=(0, 0, 0, 0))
_neu_jit = jax.jit(_neu_vmap)
_neu_batch_jit = jax.jit(jax.vmap(_neu_vmap, in_axes=(0, 0, 0, 0)))


def bench_neu(
    grid: _h.ThermoGrid,
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> _h.BenchRecord:
    """Benchmarks thermal neutrino loss (`neu`) across Stages 0..3.

    Args:
        grid: Stellar thermodynamic profile across `n_zones` shells.
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    nz = grid.n_zones
    flags_pp = (True, True, False, False, False)
    t_c = _dispatch.as_c_f64(grid.t)
    lt_c = _dispatch.as_c_f64(grid.log10_t)
    r_c = _dispatch.as_c_f64(grid.rho)
    lr_c = _dispatch.as_c_f64(grid.log10_rho)
    a_c = _dispatch.as_c_f64(grid.abar)
    z_c = _dispatch.as_c_f64(grid.zbar)

    stage0, out0 = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _rust.neu_get_batch(
            t_c, lt_c, r_c, lr_c, a_c, z_c, 7.5, flags_pp, False
        ),
    )
    stage1, out1 = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _rust.neu_get_batch(
            t_c, lt_c, r_c, lr_c, a_c, z_c, 7.5, flags_pp, True
        ),
    )

    t_j = jnp.asarray(grid.t, dtype=jnp.float64)
    rho_j = jnp.asarray(grid.rho, dtype=jnp.float64)
    abar_j = jnp.asarray(grid.abar, dtype=jnp.float64)
    zbar_j = jnp.asarray(grid.zbar, dtype=jnp.float64)
    stage2, out2_j = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _neu_jit(t_j, rho_j, abar_j, zbar_j),
        record_first_as_compile=True,
    )

    b_sz = _h.STAGE3_BATCH_SIZE
    t_b = jnp.broadcast_to(t_j[None, :], (b_sz, nz))
    rho_b = jnp.broadcast_to(rho_j[None, :], (b_sz, nz))
    abar_b = jnp.broadcast_to(abar_j[None, :], (b_sz, nz))
    zbar_b = jnp.broadcast_to(zbar_j[None, :], (b_sz, nz))
    stage3, out3_j = _h.measure_stage3_batch(
        n_warmup,
        n_repeats,
        lambda: _neu_batch_jit(t_b, rho_b, abar_b, zbar_b),
    )

    out2_np = np.asarray(out2_j, dtype=np.float64)
    out3_np = np.asarray(out3_j, dtype=np.float64)
    max_rel_err = _h.max_err3(
        _h.max_rel_diff(out0, out1),
        _h.max_rel_diff(out0, out2_np),
        _h.max_rel_diff(out0, out3_np),
    )
    return _h.BenchRecord(
        workload="neu",
        n_zones=nz,
        stage0=stage0,
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        max_rel_err=max_rel_err,
        checksum=float(np.sum(out2_np[:, 0])),
    )
