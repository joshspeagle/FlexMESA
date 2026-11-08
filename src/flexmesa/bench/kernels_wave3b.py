# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wave 3b four-stage microphysics benchmarks (`flexmesa.bench.kernels_wave3b`).

Evaluates the Three-Tier Hybrid Architecture across Stages 0..3 for the five
Wave 3b stellar microphysics workloads (`eos_4d`, `kap_fixed`, `rates_screen`,
`net_eval`, and `atm_table`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust, atm, chem, constants, interp_2d
from flexmesa.bench import harness as _h
from flexmesa.bench import kernels_wave2 as _w2
from flexmesa.eos import core
from flexmesa.rates import jax_kernels as _rates_jax

_SOLAR7_NAMES: tuple[str, ...] = (
    "h1",
    "he4",
    "c12",
    "n14",
    "o16",
    "ne20",
    "mg24",
)
_SOLAR7_XA: np.ndarray = np.array(
    [0.70, 0.28, 0.004, 0.001, 0.009, 0.003, 0.003], dtype=np.float64
)


def _eval_eos_4d_cell_jax(
    log_rho: jax.Array,
    log_t: jax.Array,
    q0: jax.Array,
    q1: jax.Array,
    t0: jax.Array,
    t1: jax.Array,
    v00: jax.Array,
    v10: jax.Array,
    v01: jax.Array,
    v11: jax.Array,
) -> jax.Array:
    """Evaluate 16-channel `OPAL/SCVH` bicubic splines and partials in JAX."""
    log_q = log_rho - 2.0 * log_t + 12.0
    hx, hy = q1 - q0, t1 - t0
    hxi, hyi = 1.0 / hx, 1.0 / hy
    hx2, hy2 = hx * hx, hy * hy
    xp, yp = (log_q - q0) * hxi, (log_t - t0) * hyi
    xpi, ypi = 1.0 - xp, 1.0 - yp
    xp2, xpi2 = xp * xp, xpi * xpi
    yp2, ypi2 = yp * yp, ypi * ypi

    cx, cxi = xp * (xp2 - 1.0), xpi * (xpi2 - 1.0)
    cxd, cxdi = 3.0 * xp2 - 1.0, -3.0 * xpi2 + 1.0
    cy, cyi = yp * (yp2 - 1.0), ypi * (ypi2 - 1.0)
    cyd, cydi = 3.0 * yp2 - 1.0, -3.0 * ypi2 + 1.0

    shx2, shy2 = constants.ONE_SIXTH * hx2, constants.ONE_SIXTH * hy2
    z36_xy = shx2 * shy2
    shx = constants.ONE_SIXTH * hx
    shxi_hy2, z36_x_y2 = shy2 * hxi, shx * shy2
    shy = constants.ONE_SIXTH * hy
    shx2_hyi, z36_x2_y = shx2 * hyi, shx2 * shy

    nz = log_rho.shape[0]
    c00, c10 = v00.reshape((nz, 16, 4)), v10.reshape((nz, 16, 4))
    c01, c11 = v01.reshape((nz, 16, 4)), v11.reshape((nz, 16, 4))

    xp_e, xpi_e = xp[:, None], xpi[:, None]
    yp_e, ypi_e = yp[:, None], ypi[:, None]
    cx_e, cxi_e = cx[:, None], cxi[:, None]
    cxd_e, cxdi_e = cxd[:, None], cxdi[:, None]
    cy_e, cyi_e = cy[:, None], cyi[:, None]
    cyd_e, cydi_e = cyd[:, None], cydi[:, None]

    f0_0 = ypi_e * c00[:, :, 0] + yp_e * c01[:, :, 0]
    f0_1 = ypi_e * c10[:, :, 0] + yp_e * c11[:, :, 0]
    f1_0 = ypi_e * c00[:, :, 1] + yp_e * c01[:, :, 1]
    f1_1 = ypi_e * c10[:, :, 1] + yp_e * c11[:, :, 1]
    f2_0 = cyi_e * c00[:, :, 2] + cy_e * c01[:, :, 2]
    f2_1 = cyi_e * c10[:, :, 2] + cy_e * c11[:, :, 2]
    f3_0 = cyi_e * c00[:, :, 3] + cy_e * c01[:, :, 3]
    f3_1 = cyi_e * c10[:, :, 3] + cy_e * c11[:, :, 3]

    f16 = (
        xpi_e * f0_0
        + xp_e * f0_1
        + shx2[:, None] * (cxi_e * f1_0 + cx_e * f1_1)
        + shy2[:, None] * (xpi_e * f2_0 + xp_e * f2_1)
        + z36_xy[:, None] * (cxi_e * f3_0 + cx_e * f3_1)
    )
    df_dx = (
        hxi[:, None] * (-f0_0 + f0_1)
        + shx[:, None] * (cxdi_e * f1_0 + cxd_e * f1_1)
        + shxi_hy2[:, None] * (-f2_0 + f2_1)
        + z36_x_y2[:, None] * (cxdi_e * f3_0 + cxd_e * f3_1)
    )
    d00, d10 = c01[:, :, 0] - c00[:, :, 0], c11[:, :, 0] - c10[:, :, 0]
    d01, d11 = c01[:, :, 1] - c00[:, :, 1], c11[:, :, 1] - c10[:, :, 1]
    g2_0 = cydi_e * c00[:, :, 2] + cyd_e * c01[:, :, 2]
    g2_1 = cydi_e * c10[:, :, 2] + cyd_e * c11[:, :, 2]
    g3_0 = cydi_e * c00[:, :, 3] + cyd_e * c01[:, :, 3]
    g3_1 = cydi_e * c10[:, :, 3] + cyd_e * c11[:, :, 3]
    df_dy = (
        hyi[:, None] * (xpi_e * d00 + xp_e * d10)
        + shx2_hyi[:, None] * (cxi_e * d01 + cx_e * d11)
        + shy[:, None] * (xpi_e * g2_0 + xp_e * g2_1)
        + z36_x2_y[:, None] * (cxi_e * g3_0 + cx_e * g3_1)
    )
    dlnd16 = df_dx / constants.LN10
    dlnt16 = df_dy / constants.LN10 - 2.0 * dlnd16
    z4 = jnp.zeros((nz, 4), dtype=jnp.float64)
    o1 = jnp.ones((nz, 1), dtype=jnp.float64)
    z5 = jnp.zeros((nz, 5), dtype=jnp.float64)
    z10 = jnp.zeros((nz, 10), dtype=jnp.float64)
    res26 = jnp.concatenate([f16, z4, o1, z5], axis=-1)
    dlnd26 = jnp.concatenate([dlnd16, z10], axis=-1)
    dlnt26 = jnp.concatenate([dlnt16, z10], axis=-1)
    return jnp.concatenate([res26, dlnd26, dlnt26], axis=-1)


_eos_4d_zones_jit = jax.jit(_eval_eos_4d_cell_jax)
_eos_4d_batch_jit = jax.jit(
    jax.vmap(
        _eos_4d_zones_jit,
        in_axes=(0, 0, None, None, None, None, None, None, None, None),
    )
)


def bench_eos_4d(
    nz: int, n_warmup: int = 2, n_repeats: int = 7
) -> _h.BenchRecord:
    """Benchmark 4D tabulated `OPAL/SCVH` EOS (`eos_4d`).

    Args:
        nz: Number of radial zones ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    cids = [chem.get_nuclide_index(nm) for nm in _SOLAR7_NAMES]
    xa_c = _dispatch.as_c_f64(_SOLAR7_XA)
    s = np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    rho_c = _dispatch.as_c_f64(10.0 ** (-2.0 + 0.01 * s))
    t_c = _dispatch.as_c_f64(10.0 ** (5.0 + 0.01 * s))
    lr_c = _dispatch.as_c_f64(np.log10(rho_c))
    lt_c = _dispatch.as_c_f64(np.log10(t_c))

    q0, q1, t0, t1, v00, v10, v01, v11 = _rust.eos_opal_scvh_cells_batch(
        lr_c, lt_c, 5, 2
    )
    lr_j, lt_j = jnp.asarray(lr_c), jnp.asarray(lt_c)
    q0_j, q1_j = jnp.asarray(q0), jnp.asarray(q1)
    t0_j, t1_j = jnp.asarray(t0), jnp.asarray(t1)
    v00_j, v10_j = jnp.asarray(v00), jnp.asarray(v10)
    v01_j, v11_j = jnp.asarray(v01), jnp.asarray(v11)
    b_sz = _h.STAGE3_BATCH_SIZE
    lr_b = jnp.broadcast_to(lr_j[None, :], (b_sz, nz))
    lt_b = jnp.broadcast_to(lt_j[None, :], (b_sz, nz))

    return _w2._run_four_stages(
        "eos_4d",
        nz,
        n_warmup,
        n_repeats,
        lambda: _rust.eos_eval_opal_scvh_zones(rho_c, t_c, cids, xa_c, False),
        lambda: _rust.eos_eval_opal_scvh_zones(rho_c, t_c, cids, xa_c, True),
        lambda: _eos_4d_zones_jit(
            lr_j, lt_j, q0_j, q1_j, t0_j, t1_j, v00_j, v10_j, v01_j, v11_j
        ),
        lambda: _eos_4d_batch_jit(
            lr_b, lt_b, q0_j, q1_j, t0_j, t1_j, v00_j, v10_j, v01_j, v11_j
        ),
        chk_col=core.I_LN_PGAS,
    )


def _wrap_host_2arg_jit(eval_1d_fn: Any, n_cols: int) -> tuple[Any, Any]:
    """Wrap a 2-array 1D host evaluator into `@jax.jit` zone/batch callables."""

    def _host_cb(x_a: np.ndarray, y_a: np.ndarray) -> np.ndarray:
        orig = np.shape(x_a)
        x_1d = _dispatch.as_c_f64(np.asarray(x_a, dtype=np.float64).ravel())
        y_1d = _dispatch.as_c_f64(np.asarray(y_a, dtype=np.float64).ravel())
        return np.ascontiguousarray(
            eval_1d_fn(x_1d, y_1d, True).reshape((*orig, n_cols))
        )

    def _core(x_in: jax.Array, y_in: jax.Array) -> jax.Array:
        spec = jax.ShapeDtypeStruct((*x_in.shape, n_cols), jnp.float64)  # type: ignore[no-untyped-call]
        res = jax.pure_callback(
            _host_cb, spec, x_in, y_in, vmap_method="expand_dims"
        )
        return jnp.asarray(res, dtype=jnp.float64)

    return jax.jit(_core), jax.jit(jax.vmap(_core, in_axes=(0, 0)))


def bench_kap_fixed(
    nz: int, n_warmup: int = 2, n_repeats: int = 7
) -> _h.BenchRecord:
    """Benchmark Type 1 fixed-metal + conductive opacity (`kap_fixed`).

    Args:
        nz: Number of radial zones ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    cids = [chem.get_nuclide_index(nm) for nm in _SOLAR7_NAMES]
    comp = chem.get_composition_info(cids, _SOLAR7_XA, backend="rust")
    z, x, zbar = float(comp.xz), float(comp.xh), float(comp.zbar)
    lnfree_e = -0.160_442_8
    eta, d_eta_dr, d_eta_dt = -15.729_47, 1.0, -1.5

    s = np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    lr_c = _dispatch.as_c_f64(-6.0 + 0.04 * s)
    lt_c = _dispatch.as_c_f64(6.0 + 0.04 * s)

    def _eval_kap(
        lr_1d: np.ndarray, lt_1d: np.ndarray, par: bool
    ) -> np.ndarray:
        return _rust.kap_get_results_zones_rust(
            z,
            x,
            0.02,
            0.004,
            0.001,
            0.009,
            0.003,
            lr_1d,
            lt_1d,
            lnfree_e,
            0.0,
            0.0,
            eta,
            d_eta_dr,
            d_eta_dt,
            zbar=zbar,
            parallel=par,
        )[:, :3]

    kap_zones_jit, kap_batch_jit = _wrap_host_2arg_jit(_eval_kap, 3)
    lr_j, lt_j = jnp.asarray(lr_c), jnp.asarray(lt_c)
    b_sz = _h.STAGE3_BATCH_SIZE
    lr_b = jnp.broadcast_to(lr_j[None, :], (b_sz, nz))
    lt_b = jnp.broadcast_to(lt_j[None, :], (b_sz, nz))

    return _w2._run_four_stages(
        "kap_fixed",
        nz,
        n_warmup,
        n_repeats,
        lambda: _eval_kap(lr_c, lt_c, False),
        lambda: _eval_kap(lr_c, lt_c, True),
        lambda: kap_zones_jit(lr_j, lt_j),
        lambda: kap_batch_jit(lr_b, lt_b),
        chk_col=0,
    )


def _rates_screen_zones_core(
    t_z: jax.Array, rho_z: jax.Array, coeffs_j: jax.Array
) -> jax.Array:
    """Evaluate `r_c12_ag_o16` REACLIB + `extended` screening in JAX."""
    fw, _ = _rates_jax.eval_reaclib_jax(coeffs_j, t_z * 1.0e-9)
    scor, _, _ = _rates_jax.screen_extended_jax(
        t_z,
        rho_z,
        jnp.asarray(6.857_142_857_142_857, dtype=jnp.float64),
        jnp.asarray(13.714_285_714_285_714, dtype=jnp.float64),
        jnp.asarray(47.428_571_428_571_43, dtype=jnp.float64),
        jnp.asarray(12.0, dtype=jnp.float64),
        jnp.asarray(6.0, dtype=jnp.float64),
        jnp.asarray(4.0, dtype=jnp.float64),
        jnp.asarray(2.0, dtype=jnp.float64),
    )
    return jnp.stack([jnp.log10(fw * scor), fw, scor], axis=-1)


_rates_screen_zones_jit = jax.jit(_rates_screen_zones_core)
_rates_screen_batch_jit = jax.jit(
    jax.vmap(_rates_screen_zones_jit, in_axes=(0, 0, None))
)


def bench_rates_screen(
    nz: int, n_warmup: int = 2, n_repeats: int = 7
) -> _h.BenchRecord:
    """Benchmark `r_c12_ag_o16` REACLIB + `extended` screening (`rates_screen`).

    Args:
        nz: Number of radial zones ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    _, terms = _rust.rates_reaclib_terms("r_c12_ag_o16")
    coeffs_j = jnp.asarray([row[3] for row in terms], dtype=jnp.float64)
    zbar = 6.857_142_857_142_857
    abar = 13.714_285_714_285_714
    z2bar = 47.428_571_428_571_43

    s = np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    rho_c = _dispatch.as_c_f64(10.0 ** (2.0 + 3.0 * s))
    t_c = _dispatch.as_c_f64(10.0 ** (8.1 + 0.4 * s))

    t_j, rho_j = jnp.asarray(t_c), jnp.asarray(rho_c)
    b_sz = _h.STAGE3_BATCH_SIZE
    t_b = jnp.broadcast_to(t_j[None, :], (b_sz, nz))
    rho_b = jnp.broadcast_to(rho_j[None, :], (b_sz, nz))

    def _eval_rates(par: bool) -> np.ndarray:
        return _rust.rates_eval_reaclib_screen_zones(
            "r_c12_ag_o16",
            "extended",
            t_c,
            rho_c,
            zbar,
            abar,
            z2bar,
            12.0,
            6.0,
            4.0,
            2.0,
            par,
        )

    return _w2._run_four_stages(
        "rates_screen",
        nz,
        n_warmup,
        n_repeats,
        lambda: _eval_rates(False),
        lambda: _eval_rates(True),
        lambda: _rates_screen_zones_jit(t_j, rho_j, coeffs_j),
        lambda: _rates_screen_batch_jit(t_b, rho_b, coeffs_j),
        chk_col=0,
    )


def bench_net_eval(
    nz: int, n_warmup: int = 2, n_repeats: int = 7
) -> _h.BenchRecord:
    """Benchmark 8-isotope `basic.net` RHS + Jacobian (`net_eval`).

    Args:
        nz: Number of radial zones ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    _, _, cids, _, _ = _rust.net_info("basic.net")
    x_c = _dispatch.as_c_f64(
        [
            0.70,
            2.98e-5,
            0.28,
            0.0034,
            0.001,
            0.0096,
            0.002,
            1.0 - (0.70 + 2.98e-5 + 0.28 + 0.0034 + 0.001 + 0.0096 + 0.002),
        ]
    )
    comp = chem.get_composition_info(cids, x_c, backend="rust")
    abar, zbar = float(comp.abar), float(comp.zbar)
    z2bar, ye = float(comp.z2bar), float(comp.ye)

    s = np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    lr_c = _dispatch.as_c_f64(2.0 + 0.1 * s)
    lt_c = _dispatch.as_c_f64(7.9 + 0.05 * s)

    def _eval_net(
        lt_1d: np.ndarray, lr_1d: np.ndarray, par: bool
    ) -> np.ndarray:
        t_1d = _dispatch.as_c_f64(10.0**lt_1d)
        r_1d = _dispatch.as_c_f64(10.0**lr_1d)
        return _rust.net_eval_zones(
            "basic.net",
            x_c,
            t_1d,
            lt_1d,
            r_1d,
            lr_1d,
            abar,
            zbar,
            z2bar,
            ye,
            0.0,
            0.0,
            0.0,
            1.0,
            "extended",
            False,
            par,
        )

    net_zones_jit, net_batch_jit = _wrap_host_2arg_jit(_eval_net, 4)
    lt_j, lr_j = jnp.asarray(lt_c), jnp.asarray(lr_c)
    b_sz = _h.STAGE3_BATCH_SIZE
    lt_b = jnp.broadcast_to(lt_j[None, :], (b_sz, nz))
    lr_b = jnp.broadcast_to(lr_j[None, :], (b_sz, nz))

    return _w2._run_four_stages(
        "net_eval",
        nz,
        n_warmup,
        n_repeats,
        lambda: _eval_net(lt_c, lr_c, False),
        lambda: _eval_net(lt_c, lr_c, True),
        lambda: net_zones_jit(lt_j, lr_j),
        lambda: net_batch_jit(lt_b, lr_b),
        chk_col=0,
    )


def _eval_atm_photosphere_1zone_jax(
    l_val: jax.Array,
    r_val: jax.Array,
    m_val: jax.Array,
    teff: jax.Array,
    logg_arr: jax.Array,
    teff_arr: jax.Array,
    pgas_coeffs: jax.Array,
) -> jax.Array:
    """Evaluate `ATM_TABLE_PHOTOSPHERE` at $Z = 0.0179$ in JAX."""
    g = constants.STANDARD_CGRAV * m_val / (r_val * r_val)
    logg = jnp.log10(g)
    ev = interp_2d.interp_evbicub_db(
        logg, teff, logg_arr, teff_arr, False, False, pgas_coeffs, backend="jax"
    )
    pgas, dpgas_dlogg, dpgas_dteff = ev.f, ev.df_dx, ev.df_dy

    ln_t = jnp.log(teff)
    prad = constants.ONE_THIRD * constants.CRAD * (teff**4.0)
    p = pgas + prad
    ln_p = jnp.log(p)

    dln_teff_dln_r = -0.5
    dln_teff_dl = 0.25 / l_val
    dpgas_dln_r = (
        dpgas_dlogg * (-2.0 / constants.LN10)
        + dpgas_dteff * teff * dln_teff_dln_r
    )
    dpgas_dln_m = dpgas_dlogg / constants.LN10
    dprad_dln_teff = 4.0 * prad

    dln_p_dl = (dpgas_dteff * teff + dprad_dln_teff) * dln_teff_dl / p
    dln_p_dln_r = (dpgas_dln_r + dprad_dln_teff * dln_teff_dln_r) / p
    dln_p_dln_m = dpgas_dln_m / p

    dln_t_dln_r = jnp.asarray(dln_teff_dln_r, dtype=jnp.float64)
    z0 = jnp.zeros_like(ln_t)
    return jnp.stack(
        [
            ln_t,
            dln_teff_dl,
            dln_t_dln_r,
            z0,
            z0,
            ln_p,
            dln_p_dl,
            dln_p_dln_r,
            dln_p_dln_m,
        ],
        axis=-1,
    )


_atm_table_zones_jit = jax.jit(
    jax.vmap(
        _eval_atm_photosphere_1zone_jax,
        in_axes=(0, 0, 0, 0, None, None, None),
    )
)
_atm_table_batch_jit = jax.jit(
    jax.vmap(
        _atm_table_zones_jit,
        in_axes=(0, 0, 0, 0, None, None, None),
    )
)


def bench_atm_table(
    nz: int, n_warmup: int = 2, n_repeats: int = 7
) -> _h.BenchRecord:
    """Benchmark `ATM_TABLE_PHOTOSPHERE` model atmosphere (`atm_table`).

    Args:
        nz: Number of radial zones ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations.
        n_repeats: Number of timed iterations.

    Returns:
        `BenchRecord` with Stages 0..3 latencies and parity error.
    """
    logg_a, teff_a, _, _, pgas_c = _rust.atm_table_photosphere_zsol_spline()
    logg_j, teff_arr_j = jnp.asarray(logg_a), jnp.asarray(teff_a)
    pgas_j = jnp.asarray(pgas_c).reshape((logg_a.size, teff_a.size, 4))

    s = np.arange(nz, dtype=np.float64) / float(max(nz - 1, 1))
    r_c = _dispatch.as_c_f64((0.8 + 0.6 * s) * constants.RSUN)
    teff_c = _dispatch.as_c_f64(5200.0 + 1200.0 * s)
    m_c = _dispatch.as_c_f64(np.full(nz, constants.MSUN, dtype=np.float64))
    l_c = _dispatch.as_c_f64(
        constants.PI
        * constants.CRAD
        * constants.CLIGHT
        * (r_c**2.0)
        * (teff_c**4.0)
    )

    l_j, r_j = jnp.asarray(l_c), jnp.asarray(r_c)
    m_j, teff_j = jnp.asarray(m_c), jnp.asarray(teff_c)
    b_sz = _h.STAGE3_BATCH_SIZE
    l_b = jnp.broadcast_to(l_j[None, :], (b_sz, nz))
    r_b = jnp.broadcast_to(r_j[None, :], (b_sz, nz))
    m_b = jnp.broadcast_to(m_j[None, :], (b_sz, nz))
    teff_b = jnp.broadcast_to(teff_j[None, :], (b_sz, nz))

    def _eval_atm(par: bool) -> np.ndarray:
        return _rust.atm_eval_table_zones(
            l_c,
            r_c,
            m_c,
            constants.STANDARD_CGRAV,
            atm.ATM_TABLE_PHOTOSPHERE,
            0.0179,
            False,
            teff_c,
            par,
        )[:, :9]

    return _w2._run_four_stages(
        "atm_table",
        nz,
        n_warmup,
        n_repeats,
        lambda: _eval_atm(False),
        lambda: _eval_atm(True),
        lambda: _atm_table_zones_jit(
            l_j, r_j, m_j, teff_j, logg_j, teff_arr_j, pgas_j
        ),
        lambda: _atm_table_batch_jit(
            l_b, r_b, m_b, teff_b, logg_j, teff_arr_j, pgas_j
        ),
        chk_col=5,
    )


def run_all_wave3b_benchmarks(
    zone_counts: Sequence[int] = (128, 1024, 4096),
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> list[_h.BenchRecord]:
    """Run all five Wave 3b microphysics benchmarks across `zone_counts`.

    Args:
        zone_counts: Sequence of radial zone counts ($N_{\\mathrm{zones}}$).
        n_warmup: Number of untimed warmup iterations per stage.
        n_repeats: Number of timed iterations per stage.

    Returns:
        List of [`BenchRecord`] objects (`5 * len(zone_counts)` total).
    """
    records: list[_h.BenchRecord] = []
    for nz in zone_counts:
        records.append(bench_eos_4d(nz, n_warmup, n_repeats))
        records.append(bench_kap_fixed(nz, n_warmup, n_repeats))
        records.append(bench_rates_screen(nz, n_warmup, n_repeats))
        records.append(bench_net_eval(nz, n_warmup, n_repeats))
        records.append(bench_atm_table(nz, n_warmup, n_repeats))
    return records
