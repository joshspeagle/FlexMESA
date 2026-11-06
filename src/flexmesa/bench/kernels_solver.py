# SPDX-License-Identifier: LGPL-3.0-or-later
"""Four-stage solver and convective transport benchmarks in Python/NumPy/JAX.

Implements Stages 0..3 for:
1. `bench_mtx_block_tridiag`: $5 \\times 5$ block-tridiagonal Henyey system
   factorization and solve (`mtx`) across NumPy and JAX `lax.scan`.
2. `bench_turb_mlt`: Cox Mixing-Length Theory (`set_mlt` / `set_mlt_jax`) with
   first-order sensitivities across scalar `DualOrder1`, `jax.vmap`,
   `@jax.jit`, and batched `(B, N_zones)` stages.
3. `bench_turb_tdc`: Kuhfuss 1D Time-Dependent Convection (`set_tdc`) with
   implicit-function-theorem sensitivities across Stages 0..3.
"""

from __future__ import annotations

import collections.abc

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import auto_diff as _ad
from flexmesa import constants as _const
from flexmesa import turb as _turb
from flexmesa.bench import harness as _h
from flexmesa.bench import kernels_micro as _micro
from flexmesa.bench import tdc_fast as _tdc_fast
from flexmesa.bench import tridiag_fast as _tridiag_fast

bench_mtx_block_tridiag = _tridiag_fast.bench_mtx_block_tridiag


def _c6(val: float) -> _ad.DualOrder1:
    """Creates a 6-variable constant `DualOrder1`."""
    return _ad.DualOrder1.constant(val, num_vars=6)


def _v6(val: float, idx: int) -> _ad.DualOrder1:
    """Creates the `idx`-th 6-variable independent `DualOrder1`."""
    return _ad.DualOrder1.independent(val, idx, num_vars=6)


def _eval_mlt_zone_dual(s_frac: float) -> tuple[float, float, float, float]:
    """Evaluates Cox `set_mlt` with 6-variable `DualOrder1` at `s_frac`.

    Args:
        s_frac: Normalized radial shell coordinate $s \\in [0, 1]$.

    Returns:
        4-tuple `(grad_t, d, conv_vel, d_grad_t_d_gradr)` in CGS units.
    """
    t_val = 1.0e5 * (1.0 + 0.5 * s_frac)
    rho_val = 1.0e-5 * (1.0 + 0.5 * s_frac)
    p_val = rho_val * t_val * _tdc_fast.RGAS_IDEAL
    hp_val = p_val / (rho_val * _tdc_fast.GRAV_SOLAR)
    inp = _turb.MltInputs(
        chi_t=_c6(1.0),
        chi_rho=_c6(1.0),
        cp=_c6(_tdc_fast.CP_IDEAL),
        grav=_c6(_tdc_fast.GRAV_SOLAR),
        lambda_=_v6(2.0 * hp_val, 3),
        rho=_v6(rho_val, 5),
        p=_c6(p_val),
        t=_v6(t_val, 4),
        opacity=_c6(1.0),
        gradr=_v6(0.45 + 0.25 * s_frac, 0),
        grada=_v6(0.40, 1),
        gradl=_v6(0.40, 2),
    )
    out = _turb.set_mlt(_turb.MltParams.cox(2.0), inp)
    assert isinstance(out.grad_t, _ad.DualOrder1)
    assert isinstance(out.d, _ad.DualOrder1)
    assert isinstance(out.conv_vel, _ad.DualOrder1)
    return (
        float(out.grad_t.val),
        float(out.d.val),
        float(out.conv_vel.val),
        float(out.grad_t.d1[0]),
    )


def _mlt_point_jax(s_frac: jax.Array) -> jax.Array:
    """Computes `[grad_t, d, conv_vel, d_grad_t_d_gradr]` in JAX `float64`.

    Args:
        s_frac: Normalized radial shell coordinate $s \\in [0, 1]$ (`float64`
            scalar).

    Returns:
        4-element `float64` JAX array `[grad_t, d, conv_vel, d_grad_t_d_gradr]`
        of shape `(4,)`.
    """
    t_val = 1.0e5 * (1.0 + 0.5 * s_frac)
    rho_val = 1.0e-5 * (1.0 + 0.5 * s_frac)
    grav_val = jnp.float64(_tdc_fast.GRAV_SOLAR)
    cp_val = jnp.float64(_tdc_fast.CP_IDEAL)
    p_val = rho_val * t_val * jnp.float64(_tdc_fast.RGAS_IDEAL)
    lam_val = 2.0 * p_val / (rho_val * grav_val)
    gradr_val = 0.45 + 0.25 * s_frac
    one, g04 = jnp.float64(1.0), jnp.float64(0.40)

    def _mlt_with_aux(
        gr: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        _, gt, _, cv, dv = _turb.set_mlt_jax(
            one,
            one,
            cp_val,
            grav_val,
            lam_val,
            rho_val,
            p_val,
            t_val,
            one,
            gr,
            g04,
            g04,
            2.0,
        )
        return gt, (dv, cv)

    (grad_t, (d_val, conv_vel)), d_gt_d_gr = jax.value_and_grad(
        _mlt_with_aux, has_aux=True
    )(gradr_val)
    return jnp.stack([grad_t, d_val, conv_vel, d_gt_d_gr])


_mlt_vmap = jax.vmap(_mlt_point_jax)
_mlt_jit = jax.jit(_mlt_vmap)
_mlt_batch_jit = jax.jit(jax.vmap(_mlt_vmap))


def _make_tdc_params_and_inputs(
    s_np: npt.NDArray[np.float64],
) -> tuple[_turb.TdcParams, list[_turb.TdcInputs]]:
    """Pre-constructs `TdcParams` and per-zone 6-variable `TdcInputs`.

    Args:
        s_np: 1D array of normalized zone coordinates $s \\in [0, 1]$.

    Returns:
        Two-tuple `(params, inputs)` for `set_tdc`.
    """
    c_one, c_zero = _c6(1.0), _c6(0.0)
    c_r = _c6(_const.RSUN)
    c_cp = _c6(_tdc_fast.CP_IDEAL)
    c_grav = _c6(_tdc_fast.GRAV_SOLAR)
    g_ad, g_l = _v6(0.40, 1), _v6(0.40, 2)

    params = _turb.TdcParams(
        conv_vel_start=1.0e5,
        mixing_length_alpha=2.0,
        tdc_alpha_d=1.0,
        tdc_alpha_r=0.0,
        tdc_alpha_pt=0.0,
        tdc_alpha_c=1.0,
        tdc_alpha_s=1.0,
        dt=1.0e4,
        cgrav=_const.STANDARD_CGRAV,
        m=_const.MSUN,
        scale=70.0 * _const.LSUN * 1.0e-3,
        max_conv_vel=1.0e99,
        include_mlt_corr_to_tdc=True,
        use_tdc_enthalpy_flux_limiter=False,
        y_face_guess=0.0,
    )
    inputs: list[_turb.TdcInputs] = []
    for s_val in s_np:
        sf = float(s_val)
        t_val = 1.0e5 * (1.0 + 0.2 * sf)
        rho_val = 1.0e-5 * (1.0 + 0.2 * sf)
        p_val = rho_val * t_val * _tdc_fast.RGAS_IDEAL
        hp_val = p_val / (rho_val * _tdc_fast.GRAV_SOLAR)
        inputs.append(
            _turb.TdcInputs(
                chi_t=c_one,
                chi_rho=c_one,
                gradr=_v6(0.42 + 0.08 * sf, 0),
                r=c_r,
                p=_c6(p_val),
                t=_v6(t_val, 4),
                rho=_v6(rho_val, 5),
                dv=c_zero,
                cp=c_cp,
                opacity=c_one,
                scale_height=_v6(hp_val, 3),
                gradl=g_l,
                grada=g_ad,
                eq_div_w=c_zero,
                grav=c_grav,
                energy=c_zero,
            )
        )
    return params, inputs


def _eval_tdc_tuple(
    params: _turb.TdcParams, inp: _turb.TdcInputs
) -> tuple[float, float, float, float]:
    """Evaluates `set_tdc` and extracts `(grad_t, d, conv_vel, d_grad_t)`."""
    out = _turb.set_tdc(params, inp).output
    assert isinstance(out.grad_t, _ad.DualOrder1)
    assert isinstance(out.d, _ad.DualOrder1)
    assert isinstance(out.conv_vel, _ad.DualOrder1)
    return (
        float(out.grad_t.val),
        float(out.d.val),
        float(out.conv_vel.val),
        float(out.grad_t.d1[0]),
    )


def _bench_turb_4stage(
    workload: str,
    nz: int,
    n_warmup: int,
    n_repeats: int,
    stage1_repeats: int,
    eval_sub_stage0: collections.abc.Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64]], None
    ],
    vmap_fn: collections.abc.Callable[[jax.Array], jax.Array],
    jit_fn: collections.abc.Callable[[jax.Array], jax.Array],
    batch_jit_fn: collections.abc.Callable[[jax.Array], jax.Array],
) -> _h.BenchRecord:
    """Shared 4-stage benchmark driver for `turb_mlt` and `turb_tdc`.

    Args:
        workload: Workload identifier (`"turb_mlt"` or `"turb_tdc"`).
        nz: Number of radial shells ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations for Stages 2 and 3.
        n_repeats: Number of timed iterations for Stages 2 and 3.
        stage1_repeats: Number of timed iterations for Stage 1 `vmap`.
        eval_sub_stage0: Callable populating `out0_sub` on `s_sub`.
        vmap_fn: Un-JITted `jax.vmap` kernel for Stage 1.
        jit_fn: `@jax.jit`-compiled kernel for Stage 2.
        batch_jit_fn: `@jax.jit(jax.vmap)` batched kernel for Stage 3.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    s_np = np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    s_j = jnp.asarray(s_np, dtype=jnp.float64)
    m_sub = min(nz, 4)
    sub_idx = np.linspace(0, nz - 1, m_sub, dtype=np.int64)
    s_sub = s_np[sub_idx]
    out0_sub = np.empty((m_sub, 4), dtype=np.float64)

    stage0_sub = _h.measure_us(1, 2, lambda: eval_sub_stage0(s_sub, out0_sub))
    scale0 = float(nz) / float(m_sub)
    stage0 = _h.LatencyStats(
        median_us=stage0_sub.median_us * scale0,
        iqr_us=stage0_sub.iqr_us * scale0,
    )
    stage1, out1_j = _h.measure_and_capture(
        1, stage1_repeats, lambda: vmap_fn(s_j)
    )
    stage2, out2_j = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: jit_fn(s_j),
        record_first_as_compile=True,
    )

    b_sz = _h.STAGE3_BATCH_SIZE
    s_b = jnp.broadcast_to(s_j[None, :], (b_sz, nz))
    stage3, out3_j = _h.measure_stage3_batch(
        n_warmup,
        n_repeats,
        lambda: batch_jit_fn(s_b),
    )

    out1_np = np.asarray(out1_j, dtype=np.float64)
    out2_np = np.asarray(out2_j, dtype=np.float64)
    out3_np = np.asarray(out3_j, dtype=np.float64)
    max_rel_err = _h.max_err3(
        _h.max_rel_diff(out0_sub, out1_np[sub_idx]),
        _h.max_rel_diff(out1_np, out2_np),
        _h.max_rel_diff(out1_np, out3_np),
    )
    return _h.BenchRecord(
        workload=workload,
        n_zones=nz,
        stage0=stage0,
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        max_rel_err=max_rel_err,
        checksum=float(np.sum(out2_np[:, 0])),
    )


def _run_mlt_stage0_sub(
    s_sub: npt.NDArray[np.float64], out0_sub: npt.NDArray[np.float64]
) -> None:
    """Evaluates Stage 0 `set_mlt` across the anchor coordinates `s_sub`."""
    for i, sf in enumerate(s_sub):
        out0_sub[i] = _eval_mlt_zone_dual(float(sf))


def _run_tdc_stage0_sub(
    s_sub: npt.NDArray[np.float64], out0_sub: npt.NDArray[np.float64]
) -> None:
    """Evaluates Stage 0 `set_tdc` across the anchor coordinates `s_sub`."""
    p0, inps0 = _make_tdc_params_and_inputs(s_sub)
    for i, inp in enumerate(inps0):
        out0_sub[i] = _eval_tdc_tuple(p0, inp)


def bench_turb_mlt(
    grid: _h.ThermoGrid,
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> _h.BenchRecord:
    """Benchmarks Cox Mixing-Length Theory (`set_mlt`) across Stages 0..3.

    Args:
        grid: Stellar thermodynamic profile across `n_zones` shells.
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    return _bench_turb_4stage(
        "turb_mlt",
        grid.n_zones,
        n_warmup,
        n_repeats,
        n_repeats,
        _run_mlt_stage0_sub,
        _mlt_vmap,
        _mlt_jit,
        _mlt_batch_jit,
    )


def bench_turb_tdc(
    grid: _h.ThermoGrid,
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> _h.BenchRecord:
    """Benchmarks Kuhfuss 1D Time-Dependent Convection (`set_tdc`) Stages 0..3.

    Args:
        grid: Stellar thermodynamic profile across `n_zones` shells.
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    return _bench_turb_4stage(
        "turb_tdc",
        grid.n_zones,
        n_warmup,
        n_repeats,
        3,
        _run_tdc_stage0_sub,
        _tdc_fast.tdc_vmap,
        _tdc_fast.tdc_jit,
        _tdc_fast.tdc_batch_jit,
    )


def run_all_wave1_benchmarks(
    zone_counts: collections.abc.Sequence[int] = (128, 1024, 4096),
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> list[_h.BenchRecord]:
    """Runs all six Wave 1 benchmark workloads across `zone_counts`.

    Args:
        zone_counts: Sequence of stellar zone counts $N_{\\mathrm{zones}}$.
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        List of `BenchRecord` instances across all workloads and zone counts.
    """
    table = _h.BicubTable.create()
    records: list[_h.BenchRecord] = []
    for nz in zone_counts:
        grid = _h.ThermoGrid.create(nz)
        records.append(_micro.bench_autodiff(grid, n_warmup, n_repeats))
        records.append(_micro.bench_interp2d(grid, table, n_warmup, n_repeats))
        records.append(_micro.bench_neu(grid, n_warmup, n_repeats))
        records.append(bench_mtx_block_tridiag(nz, n_warmup, n_repeats))
        records.append(bench_turb_mlt(grid, n_warmup, n_repeats))
        records.append(bench_turb_tdc(grid, n_warmup, n_repeats))
    return records
