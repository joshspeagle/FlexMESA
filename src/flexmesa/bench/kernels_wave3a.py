# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wave 3a four-stage EOS benchmarks (`flexmesa.bench.kernels_wave3a`).

Evaluates the Three-Tier Hybrid Architecture across Stages 0..3 for the four
Wave 3a stellar equation-of-state (`eos`) workloads:
1. `bench_eos_ideal`: Neutral ideal-gas plus Planckian radiation EOS
   (`eval_ideal_eos_zones`) over the 8-isotope `WAVE2_ISO_NAMES` composition
   profile at $\\log_{10}\\rho \\in [-6, -2]$ and $\\log_{10}T \\in [3.5, 4.5]$.
2. `bench_eos_helm`: Timmes & Swesty (2000) biquintic/bicubic Helmholtz
   electron-positron + Hull & Lewis (1956) Coulomb + ideal-ion + radiation EOS
   (`eval_helm_eos_zones`) for a 50/50 ${}^{12}\\mathrm{C}/{}^{16}\\mathrm{O}$
   mixture at $\\log_{10}\\rho \\in [1.0, 6.0]$ and
   $\\log_{10}T \\in [8.0, 8.008]$.
3. `bench_eos_pc`: Potekhin & Chabrier (2010, 2013) non-ideal Coulomb liquid
   and bcc crystal EOS (`eval_pc_eos_zones`) for a 50/50
   ${}^{12}\\mathrm{C}/{}^{16}\\mathrm{O}$ white dwarf interior crossing the
   liquid-to-crystal phase transition ($\\log_{10}\\rho \\in [4.5, 7.5]$,
   $\\log_{10}T \\in [6.2, 7.4]$).
4. `bench_eos_skye`: Jermyn et al. (2021) third-order Helmholtz free-energy
   minimization EOS (`eval_skye_eos_zones`) for a 50/50
   ${}^{12}\\mathrm{C}/{}^{16}\\mathrm{O}$ plasma at
   $\\log_{10}\\rho \\in [1.0, 6.0]$ and $\\log_{10}T \\in [8.0, 8.008]$.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust, chem
from flexmesa.bench import harness as _h
from flexmesa.bench import kernels_wave2 as _w2
from flexmesa.eos import _helm_jax, _pc_jax, _skye_jax, core
from flexmesa.eos import helm as _eos_helm
from flexmesa.eos import ideal as _eos_ideal

_N_RES: int = core.NUM_EOS_BASIC_RESULTS


def _c12_o16_bench_props() -> tuple[
    float,
    float,
    np.ndarray,
    list[int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return `(abar, zbar, xa, z_int, azion, acmi, a_baryon)` for 50/50 C/O."""
    ic12 = chem.get_nuclide_index("c12")
    io16 = chem.get_nuclide_index("o16")
    xa = np.array([0.5, 0.5], dtype=np.float64)
    comp = chem.get_composition_info([ic12, io16], xa, backend="rust")
    nc12 = chem.get_nuclide_info(ic12)
    no16 = chem.get_nuclide_info(io16)
    z_int = [int(nc12.z), int(no16.z)]
    azion = np.array([float(nc12.z), float(no16.z)], dtype=np.float64)
    acmi = np.array([float(nc12.w), float(no16.w)], dtype=np.float64)
    a_baryon = np.array([float(nc12.a), float(no16.a)], dtype=np.float64)
    return float(comp.abar), float(comp.zbar), xa, z_int, azion, acmi, a_baryon


_ideal_zones_jit = jax.jit(
    lambda t_z, r_z, ab, zb, xa, w, a: jax.vmap(
        _eos_ideal._eval_ideal_eos_core,
        in_axes=(0, 0, None, None, None, None, None),
    )(t_z, r_z, ab, zb, xa, w, a)[0]
)
_ideal_batch_jit = jax.jit(
    jax.vmap(_ideal_zones_jit, in_axes=(0, 0, None, None, None, None, None))
)


def bench_eos_ideal(
    nz: int, n_warmup: int = 2, n_repeats: int = 7
) -> _h.BenchRecord:
    """Benchmark neutral ideal-gas + radiation EOS (`eos_ideal`).

    Args:
        nz: Number of radial zones ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    cids = [chem.get_nuclide_index(nm) for nm in _w2.WAVE2_ISO_NAMES]
    xa_c = _w2.make_wave2_composition_profile(1)[0]
    comp = chem.get_composition_info(cids, xa_c, backend="rust")
    abar, zbar = float(comp.abar), float(comp.zbar)
    infos = [chem.get_nuclide_info(c) for c in cids]
    w_c = _dispatch.as_c_f64([float(inf.w) for inf in infos])
    a_c = _dispatch.as_c_f64([float(inf.a) for inf in infos])

    s = np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    rho_c = _dispatch.as_c_f64(10.0 ** (-6.0 + 4.0 * s))
    t_c = _dispatch.as_c_f64(10.0 ** (3.5 + 1.0 * s))

    t_j, rho_j = jnp.asarray(t_c), jnp.asarray(rho_c)
    ab_j, zb_j = jnp.asarray(abar), jnp.asarray(zbar)
    xa_j, w_j, a_j = jnp.asarray(xa_c), jnp.asarray(w_c), jnp.asarray(a_c)
    b_sz = _h.STAGE3_BATCH_SIZE
    t_b = jnp.broadcast_to(t_j[None, :], (b_sz, nz))
    rho_b = jnp.broadcast_to(rho_j[None, :], (b_sz, nz))

    return _w2._run_four_stages(
        "eos_ideal",
        nz,
        n_warmup,
        n_repeats,
        lambda: _rust.eos_eval_ideal_zones(
            t_c, rho_c, abar, zbar, xa_c, w_c, a_c, False
        )[:, :_N_RES],
        lambda: _rust.eos_eval_ideal_zones(
            t_c, rho_c, abar, zbar, xa_c, w_c, a_c, True
        )[:, :_N_RES],
        lambda: _ideal_zones_jit(t_j, rho_j, ab_j, zb_j, xa_j, w_j, a_j),
        lambda: _ideal_batch_jit(t_b, rho_b, ab_j, zb_j, xa_j, w_j, a_j),
        chk_col=core.I_LN_PGAS,
    )


_helm_zones_jit = jax.jit(
    lambda t_z, r_z, ab_z, zb_z, t_lo, dth, d_lo, dd, fi_f, fi_d, fi_e, fi_x: (
        _helm_jax.eval_helm_cell_jax(
            t_z, r_z, ab_z, zb_z, t_lo, dth, d_lo, dd, fi_f, fi_d, fi_e, fi_x
        )[0]
    )
)
_helm_batch_jit = jax.jit(
    jax.vmap(
        _helm_zones_jit,
        in_axes=(
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
)


def bench_eos_helm(
    nz: int, n_warmup: int = 2, n_repeats: int = 7
) -> _h.BenchRecord:
    """Benchmark Timmes & Swesty (2000) `HELM` EOS (`eos_helm`).

    Args:
        nz: Number of radial zones ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    abar, zbar, _, _, _, _, _ = _c12_o16_bench_props()
    s = np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    rho_c = _dispatch.as_c_f64(10.0 ** (1.0 + 5.0 * s))
    t_c = _dispatch.as_c_f64(10.0 ** (8.0 + 0.008 * s))
    ab_c = _dispatch.as_c_f64(np.full(nz, abar, dtype=np.float64))
    zb_c = _dispatch.as_c_f64(np.full(nz, zbar, dtype=np.float64))

    t_j, rho_j = jnp.asarray(t_c), jnp.asarray(rho_c)
    ab_j, zb_j = jnp.asarray(ab_c), jnp.asarray(zb_c)
    t_lo, dth, d_lo, dd, fi_f, fi_d, fi_e, fi_x = (
        _eos_helm._lookup_helm_cell_jax(t_j, (zb_j / ab_j) * rho_j)
    )
    b_sz = _h.STAGE3_BATCH_SIZE
    t_b = jnp.broadcast_to(t_j[None, :], (b_sz, nz))
    rho_b = jnp.broadcast_to(rho_j[None, :], (b_sz, nz))

    return _w2._run_four_stages(
        "eos_helm",
        nz,
        n_warmup,
        n_repeats,
        lambda: _rust.eos_eval_helm_zones(t_c, rho_c, ab_c, zb_c, False)[
            :, :_N_RES
        ],
        lambda: _rust.eos_eval_helm_zones(t_c, rho_c, ab_c, zb_c, True)[
            :, :_N_RES
        ],
        lambda: _helm_zones_jit(
            t_j, rho_j, ab_j, zb_j, t_lo, dth, d_lo, dd, fi_f, fi_d, fi_e, fi_x
        ),
        lambda: _helm_batch_jit(
            t_b, rho_b, ab_j, zb_j, t_lo, dth, d_lo, dd, fi_f, fi_d, fi_e, fi_x
        ),
        chk_col=core.I_LN_PGAS,
    )


_pc_zones_jit = jax.jit(
    lambda r_z, t_z, ab_z, zb_z, xa_z, az, ac: _pc_jax.eval_pc_eos_jax(
        r_z, t_z, ab_z, zb_z, xa_z, az, ac
    )[0]
)
_pc_batch_jit = jax.jit(
    jax.vmap(_pc_zones_jit, in_axes=(0, 0, None, None, None, None, None))
)


def bench_eos_pc(
    nz: int, n_warmup: int = 2, n_repeats: int = 7
) -> _h.BenchRecord:
    """Benchmark Potekhin & Chabrier (2010, 2013) `PC` EOS (`eos_pc`).

    Args:
        nz: Number of radial zones ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    abar, zbar, xa_1d, _, azion_c, acmi_c, _ = _c12_o16_bench_props()
    s = np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    rho_c = _dispatch.as_c_f64(10.0 ** (4.5 + 3.0 * s))
    t_c = _dispatch.as_c_f64(10.0 ** (6.2 + 1.2 * s))
    ab_c = _dispatch.as_c_f64(np.full(nz, abar, dtype=np.float64))
    zb_c = _dispatch.as_c_f64(np.full(nz, zbar, dtype=np.float64))
    xa_2d = np.ascontiguousarray(np.tile(xa_1d, (nz, 1)), dtype=np.float64)
    xa_flat = _dispatch.as_c_f64(xa_2d.reshape(-1))

    rho_j, t_j = jnp.asarray(rho_c), jnp.asarray(t_c)
    ab_j, zb_j = jnp.asarray(ab_c), jnp.asarray(zb_c)
    xa_j = jnp.asarray(xa_2d)
    az_j, ac_j = jnp.asarray(azion_c), jnp.asarray(acmi_c)
    b_sz = _h.STAGE3_BATCH_SIZE
    rho_b = jnp.broadcast_to(rho_j[None, :], (b_sz, nz))
    t_b = jnp.broadcast_to(t_j[None, :], (b_sz, nz))

    return _w2._run_four_stages(
        "eos_pc",
        nz,
        n_warmup,
        n_repeats,
        lambda: _rust.eos_eval_pc_zones(
            rho_c,
            t_c,
            ab_c,
            zb_c,
            xa_flat,
            azion_c,
            acmi_c,
            True,
            170.0,
            180.0,
            1.0e-4,
            False,
        )[:, :_N_RES],
        lambda: _rust.eos_eval_pc_zones(
            rho_c,
            t_c,
            ab_c,
            zb_c,
            xa_flat,
            azion_c,
            acmi_c,
            True,
            170.0,
            180.0,
            1.0e-4,
            True,
        )[:, :_N_RES],
        lambda: _pc_zones_jit(rho_j, t_j, ab_j, zb_j, xa_j, az_j, ac_j),
        lambda: _pc_batch_jit(rho_b, t_b, ab_j, zb_j, xa_j, az_j, ac_j),
        chk_col=core.I_LN_PGAS,
    )


def _skye_zones_core(
    r_z: jax.Array,
    t_z: jax.Array,
    ab_z: jax.Array,
    zb_z: jax.Array,
    xa_z: jax.Array,
    az: jax.Array,
    ac: jax.Array,
    a_in: jax.Array,
    t_lo: jax.Array,
    dth: jax.Array,
    d_lo: jax.Array,
    dd: jax.Array,
    fi_f: jax.Array,
    fi_e: jax.Array,
) -> jax.Array:
    """Evaluate `Skye` `res` across zones on pre-located `HELM` cells."""
    res, _, _ = _skye_jax.eval_skye_cell_zones_jit(
        r_z,
        t_z,
        ab_z,
        zb_z,
        xa_z,
        az,
        ac,
        a_in,
        t_lo,
        dth,
        d_lo,
        dd,
        fi_f,
        fi_e,
        jnp.asarray(True),
        jnp.asarray(False),
        1.0e-4,
    )
    return jnp.asarray(res, dtype=jnp.float64)


_skye_zones_jit = jax.jit(_skye_zones_core)


_skye_batch_jit = jax.jit(
    jax.vmap(
        _skye_zones_jit,
        in_axes=(
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
)


def bench_eos_skye(
    nz: int, n_warmup: int = 2, n_repeats: int = 7
) -> _h.BenchRecord:
    """Benchmark Jermyn et al. (2021) third-order `Skye` EOS (`eos_skye`).

    Args:
        nz: Number of radial zones ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    abar, zbar, xa_1d, z_int_c, azion_c, acmi_c, a_c = _c12_o16_bench_props()
    s = np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    rho_c = _dispatch.as_c_f64(10.0 ** (1.0 + 5.0 * s))
    t_c = _dispatch.as_c_f64(10.0 ** (8.0 + 0.008 * s))
    ab_c = _dispatch.as_c_f64(np.full(nz, abar, dtype=np.float64))
    zb_c = _dispatch.as_c_f64(np.full(nz, zbar, dtype=np.float64))
    xa_2d = np.ascontiguousarray(np.tile(xa_1d, (nz, 1)), dtype=np.float64)
    xa_flat = _dispatch.as_c_f64(xa_2d.reshape(-1))

    rho_j, t_j = jnp.asarray(rho_c), jnp.asarray(t_c)
    ab_j, zb_j = jnp.asarray(ab_c), jnp.asarray(zb_c)
    xa_j = jnp.asarray(xa_2d)
    az_j = jnp.asarray(azion_c)
    ac_j = jnp.asarray(acmi_c)
    a_j = jnp.asarray(a_c)
    t_lo, dth, d_lo, dd, fi_f, _, fi_e, _ = _eos_helm._lookup_helm_cell_jax(
        t_j, (zb_j / ab_j) * rho_j
    )
    b_sz = _h.STAGE3_BATCH_SIZE
    rho_b = jnp.broadcast_to(rho_j[None, :], (b_sz, nz))
    t_b = jnp.broadcast_to(t_j[None, :], (b_sz, nz))

    return _w2._run_four_stages(
        "eos_skye",
        nz,
        n_warmup,
        n_repeats,
        lambda: _rust.eos_eval_skye_zones(
            rho_c,
            t_c,
            ab_c,
            zb_c,
            xa_flat,
            z_int_c,
            acmi_c,
            a_c,
            True,
            "PC",
            1.0e-4,
            False,
        )[:, :_N_RES],
        lambda: _rust.eos_eval_skye_zones(
            rho_c,
            t_c,
            ab_c,
            zb_c,
            xa_flat,
            z_int_c,
            acmi_c,
            a_c,
            True,
            "PC",
            1.0e-4,
            True,
        )[:, :_N_RES],
        lambda: _skye_zones_jit(
            rho_j,
            t_j,
            ab_j,
            zb_j,
            xa_j,
            az_j,
            ac_j,
            a_j,
            t_lo,
            dth,
            d_lo,
            dd,
            fi_f,
            fi_e,
        ),
        lambda: _skye_batch_jit(
            rho_b,
            t_b,
            ab_j,
            zb_j,
            xa_j,
            az_j,
            ac_j,
            a_j,
            t_lo,
            dth,
            d_lo,
            dd,
            fi_f,
            fi_e,
        ),
        chk_col=core.I_LN_PGAS,
    )


def run_all_wave3a_benchmarks(
    zone_counts: Sequence[int] = (128, 1024, 4096),
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> list[_h.BenchRecord]:
    """Run all four Wave 3a EOS benchmarks across `zone_counts`.

    Args:
        zone_counts: Sequence of radial zone counts ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations per stage.
        n_repeats: Number of timed iterations per stage.

    Returns:
        List of [`BenchRecord`] objects (`4 * len(zone_counts)` total).
    """
    records: list[_h.BenchRecord] = []
    for nz in zone_counts:
        records.append(bench_eos_ideal(nz, n_warmup, n_repeats))
        records.append(bench_eos_helm(nz, n_warmup, n_repeats))
        records.append(bench_eos_pc(nz, n_warmup, n_repeats))
        records.append(bench_eos_skye(nz, n_warmup, n_repeats))
    return records
