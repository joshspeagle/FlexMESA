# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wave 2 four-stage microphysics benchmarks (`flexmesa.bench.kernels_wave2`).

Evaluates the Three-Tier Hybrid Architecture across Stages 0..3 for the four
Wave 2 atomic and photometric microphysics workloads:
1. `bench_chem_comp`: 8-isotope stellar composition moments (`xh, xhe, xz,
   abar, zbar, z2bar, z53bar, ye, mass_correction, sumx`) across a
   core-to-surface composition gradient (`get_composition_info_zones`).
2. `bench_ion_fe56`: Potekhin ${}^{56}\\mathrm{Fe}$-in-${}^4\\mathrm{He}$
   effective charge and partial derivatives (`[z, dz_dlog10_ne, dz_dlog10_t]`)
   on the $105 \\times 30$ Thomas-Fermi-Dirac-Hubbard bicubic spline grid.
3. `bench_ion_4d`: 4D `(Z, X, log10_rho, log10_T)` trilinear-over-bicubic
   mean ionic charge and partial pressure interpolation across all 28
   `eval_ionization` channels.
4. `bench_colors_sed`: 3D `(Teff, log_g, [M/H])` `Kurucz2003all` cubic Hermite
   SED synthesis (`1199` wavelengths) plus bolometric integration and 7-band
   `Generic/Johnson` (`B, I, J, M, R, U, V`) Vega synthetic photometry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _dispatch, _rust
from flexmesa import constants as _const
from flexmesa.bench import harness as _h
from flexmesa.chem import catalog as _chem_cat
from flexmesa.chem import composition as _chem_comp
from flexmesa.colors import photometry as _colors_phot
from flexmesa.colors import utils as _colors_utils
from flexmesa.ionization import fe56_he4 as _ion_fe56
from flexmesa.ionization import tables as _ion_tbl

WAVE2_ISO_NAMES: tuple[str, ...] = (
    "h1",
    "he3",
    "he4",
    "c12",
    "n14",
    "o16",
    "ne20",
    "mg24",
)


def make_wave2_composition_profile(nz: int) -> npt.NDArray[np.float64]:
    """Build the deterministic `(nz, 8)` stellar abundance profile `xa`.

    Matches `make_wave2_composition_profile` in `flexmesa-bench` and
    `benchmarks/fortran_mesa/bench_kernels_wave2.f90`.

    Args:
        nz: Number of radial stellar shells ($N_{\\mathrm{zones}}$).

    Returns:
        2D C-contiguous `float64` array of shape `(nz, 8)`.
    """
    s = np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    z = 0.014 + 0.016 * s
    x = 0.70 * (1.0 - s) + 0.05 * s
    y = 1.0 - x - z
    xa = np.stack(
        [
            x,
            1.0e-4 * y,
            0.9999 * y,
            0.18 * z,
            0.06 * z,
            0.48 * z,
            0.14 * z,
            0.14 * z,
        ],
        axis=-1,
    )
    return np.ascontiguousarray(xa, dtype=np.float64)


def _run_four_stages(
    workload: str,
    nz: int,
    n_warmup: int,
    n_repeats: int,
    fn_s0: Callable[[], npt.NDArray[np.float64]],
    fn_s1: Callable[[], npt.NDArray[np.float64]],
    fn_s2: Callable[[], jax.Array],
    fn_s3: Callable[[], jax.Array],
    chk_col: int,
) -> _h.BenchRecord:
    """Measure Stages 0..3 and assemble a `BenchRecord`."""
    stage0, out0 = _h.measure_and_capture(n_warmup, n_repeats, fn_s0)
    stage1, out1 = _h.measure_and_capture(n_warmup, n_repeats, fn_s1)
    stage2, out2_j = _h.measure_and_capture(
        n_warmup, n_repeats, fn_s2, record_first_as_compile=True
    )
    stage3, out3_j = _h.measure_stage3_batch(n_warmup, n_repeats, fn_s3)
    out2_np = np.asarray(out2_j, dtype=np.float64)
    out3_np = np.asarray(out3_j, dtype=np.float64)
    max_rel_err = _h.max_err3(
        _h.max_rel_diff(out0, out1),
        _h.max_rel_diff(out0, out2_np),
        _h.max_rel_diff(out0, out3_np),
    )
    return _h.BenchRecord(
        workload=workload,
        n_zones=nz,
        stage0=stage0,
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        max_rel_err=max_rel_err,
        checksum=float(np.sum(out2_np[:, chk_col])),
    )


_chem_batch_jit = jax.jit(
    jax.vmap(
        _chem_comp._composition_moments_only_core,
        in_axes=(0, None, None, None, None),
    )
)


def bench_chem_comp(
    nz: int,
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> _h.BenchRecord:
    """Benchmark 8-isotope stellar composition moments (`chem_comp`).

    Args:
        nz: Number of radial zones ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    cids = list(_chem_cat._resolve_chem_ids(WAVE2_ISO_NAMES))
    xa_c = make_wave2_composition_profile(nz)
    xa_j = jnp.asarray(xa_c, dtype=jnp.float64)
    z_a, a_a, w_a, z53_a = _chem_comp._species_props_jax(tuple(cids))
    xa_b = jnp.broadcast_to(
        xa_j[None, :, :], (_h.STAGE3_BATCH_SIZE, nz, len(cids))
    )
    return _run_four_stages(
        "chem_comp",
        nz,
        n_warmup,
        n_repeats,
        lambda: _rust.py_chem_get_composition_info_zones(cids, xa_c, False),
        lambda: _rust.py_chem_get_composition_info_zones(cids, xa_c, True),
        lambda: _chem_comp._composition_moments_only_jit(
            xa_j, z_a, a_a, w_a, z53_a
        ),
        lambda: _chem_batch_jit(xa_b, z_a, a_a, w_a, z53_a),
        chk_col=3,
    )


def _fe56_zones_jax(
    lne: jax.Array,
    lt: jax.Array,
    xs: jax.Array,
    ys: jax.Array,
    coeffs: jax.Array,
) -> jax.Array:
    """Evaluate `(nz, 3)` `[z, dz_dlog10_ne, dz_dlog10_t]` in JAX `float64`."""
    z, dx, dy = _ion_fe56._eval_fe56_he4_core(lne, lt, xs, ys, coeffs)
    return jnp.stack([z, dx, dy], axis=-1)


_fe56_zones_jit = jax.jit(_fe56_zones_jax)
_fe56_batch_jit = jax.jit(
    jax.vmap(_fe56_zones_jax, in_axes=(0, 0, None, None, None))
)


def bench_ion_fe56(
    nz: int,
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> _h.BenchRecord:
    """Benchmark Fe56-in-He4 bicubic spline (`ion_fe56`).

    Args:
        nz: Number of radial zones ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    s = np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    lne_c = _dispatch.as_c_f64(24.5 + 5.5 * s)
    lt_c = _dispatch.as_c_f64(6.2 + 1.6 * s)
    lne_j = jnp.asarray(lne_c, dtype=jnp.float64)
    lt_j = jnp.asarray(lt_c, dtype=jnp.float64)
    xs, ys, coeffs = _ion_fe56._fe56_he4_grid_jax()
    b_sz = _h.STAGE3_BATCH_SIZE
    lne_b = jnp.broadcast_to(lne_j[None, :], (b_sz, nz))
    lt_b = jnp.broadcast_to(lt_j[None, :], (b_sz, nz))
    return _run_four_stages(
        "ion_fe56",
        nz,
        n_warmup,
        n_repeats,
        lambda: _rust.py_ion_charge_of_fe56_in_he4_zones(lne_c, lt_c, False),
        lambda: _rust.py_ion_charge_of_fe56_in_he4_zones(lne_c, lt_c, True),
        lambda: _fe56_zones_jit(lne_j, lt_j, xs, ys, coeffs),
        lambda: _fe56_batch_jit(lne_b, lt_b, xs, ys, coeffs),
        chk_col=0,
    )


_ion4d_batch_jit = jax.jit(
    jax.vmap(_ion_tbl._eval_ion_batch_jit, in_axes=(0, 0, 0, 0, None))
)


def bench_ion_4d(
    nz: int,
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> _h.BenchRecord:
    """Benchmark 4D `(Z, X, log10_rho, log10_T)` ionization (`ion_4d`).

    Args:
        nz: Number of radial zones ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    s = np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    z_c = _dispatch.as_c_f64(0.005 + 0.030 * s)
    x_c = _dispatch.as_c_f64(0.62 + 0.16 * s)
    lt_c = _dispatch.as_c_f64(4.505 + 0.050 * s)
    lq_c = -2.075 + 0.080 * s
    lrho_c = _dispatch.as_c_f64(lq_c + 2.0 * lt_c - 12.0)

    z_j = jnp.asarray(z_c, dtype=jnp.float64)
    x_j = jnp.asarray(x_c, dtype=jnp.float64)
    lrho_j = jnp.asarray(lrho_c, dtype=jnp.float64)
    lt_j = jnp.asarray(lt_c, dtype=jnp.float64)
    coeffs = _ion_tbl._builtin_cell_coeffs_jax()
    b_sz = _h.STAGE3_BATCH_SIZE
    z_b = jnp.broadcast_to(z_j[None, :], (b_sz, nz))
    x_b = jnp.broadcast_to(x_j[None, :], (b_sz, nz))
    lrho_b = jnp.broadcast_to(lrho_j[None, :], (b_sz, nz))
    lt_b = jnp.broadcast_to(lt_j[None, :], (b_sz, nz))
    return _run_four_stages(
        "ion_4d",
        nz,
        n_warmup,
        n_repeats,
        lambda: _rust.py_ion_eval_ionization_zones(
            z_c, x_c, lrho_c, lt_c, False
        ),
        lambda: _rust.py_ion_eval_ionization_zones(
            z_c, x_c, lrho_c, lt_c, True
        ),
        lambda: _ion_tbl._eval_ion_batch_jit(z_j, x_j, lrho_j, lt_j, coeffs),
        lambda: _ion4d_batch_jit(z_b, x_b, lrho_b, lt_b, coeffs),
        chk_col=0,
    )


_colors_batch_jit = jax.jit(
    jax.vmap(
        lambda t_b, g_b, m_b, r_b, d_s, grid: (
            _colors_phot._eval_colors_zones_with_grid_jax(
                t_b, g_b, m_b, r_b, d_s, grid, 0, 0
            )
        ),
        in_axes=(0, 0, 0, 0, None, None),
    )
)


def bench_colors_sed(
    nz: int,
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> _h.BenchRecord:
    """Benchmark 3D Kurucz Hermite SED + Johnson photometry (`colors_sed`).

    Args:
        nz: Number of stellar models ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    s = np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    teff_c = _dispatch.as_c_f64(5760.0 + 230.0 * s)
    logg_c = _dispatch.as_c_f64(4.05 + 0.40 * s)
    meta_c = _dispatch.as_c_f64(-0.45 + 0.60 * s)
    rad_c = _dispatch.as_c_f64(_const.RSUN * (0.9 + 0.4 * s))
    d_cm = _colors_utils.D_10PC_CM

    teff_j = jnp.asarray(teff_c, dtype=jnp.float64)
    logg_j = jnp.asarray(logg_c, dtype=jnp.float64)
    meta_j = jnp.asarray(meta_c, dtype=jnp.float64)
    rad_j = jnp.asarray(rad_c, dtype=jnp.float64)
    d_j = jnp.asarray(d_cm, dtype=jnp.float64)
    c_grid = _colors_utils.get_builtin_colors_grid()
    b_sz = _h.STAGE3_BATCH_SIZE
    teff_b = jnp.broadcast_to(teff_j[None, :], (b_sz, nz))
    logg_b = jnp.broadcast_to(logg_j[None, :], (b_sz, nz))
    meta_b = jnp.broadcast_to(meta_j[None, :], (b_sz, nz))
    rad_b = jnp.broadcast_to(rad_j[None, :], (b_sz, nz))
    return _run_four_stages(
        "colors_sed",
        nz,
        n_warmup,
        n_repeats,
        lambda: _rust.py_colors_eval_colors_zones(
            teff_c, logg_c, meta_c, rad_c, d_cm, "Vega", "Hermite", False
        ),
        lambda: _rust.py_colors_eval_colors_zones(
            teff_c, logg_c, meta_c, rad_c, d_cm, "Vega", "Hermite", True
        ),
        lambda: _colors_phot._eval_colors_zones_with_grid_jax(
            teff_j, logg_j, meta_j, rad_j, d_j, c_grid, 0, 0
        ),
        lambda: _colors_batch_jit(teff_b, logg_b, meta_b, rad_b, d_j, c_grid),
        chk_col=9,
    )


def run_all_wave2_benchmarks(
    zone_counts: Sequence[int] = (128, 1024, 4096),
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> list[_h.BenchRecord]:
    """Run all four Wave 2 benchmarks across `zone_counts`."""
    records: list[_h.BenchRecord] = []
    for nz in zone_counts:
        records.append(bench_chem_comp(nz, n_warmup, n_repeats))
        records.append(bench_ion_fe56(nz, n_warmup, n_repeats))
        records.append(bench_ion_4d(nz, n_warmup, n_repeats))
        records.append(bench_colors_sed(nz, n_warmup, n_repeats))
    return records
