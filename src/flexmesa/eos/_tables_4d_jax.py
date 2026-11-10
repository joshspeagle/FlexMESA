# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native `@jax.jit` (`jnp.float64`) 4D EOS table evaluators and blends.

Ports `Do_EoS_Interpolations`, `Eval_Xtable`, `Get1_eosdt_for_X`,
`Get1_eosdt_Results`, `eval_eosCMS_fixed_X`, `Get_FreeEOS_alfa`,
`get_opal_scvh_alfa_and_partials`, and `Get_CMS_alfa` to pure JAX.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import constants
from flexmesa.eos import _cache_io, core

_OPAL_ZS: tuple[float, ...] = (0.00, 0.02, 0.04)
_OPAL_XS: tuple[tuple[float, ...], ...] = (
    (0.00, 0.20, 0.40, 0.60, 0.80, 1.00),
    (0.00, 0.20, 0.40, 0.60, 0.80),
    (0.00, 0.20, 0.40, 0.60, 0.80),
)
_FREE_ZS: tuple[float, ...] = (
    0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.20, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
)  # fmt: skip
_CMS_XVALS: tuple[float, ...] = (
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)  # fmt: skip
_CMS_XSTR: tuple[str, ...] = (
    "000", "010", "020", "030", "040", "050", "060", "070", "080", "090", "100",
)  # fmt: skip


def _free_xs_for_iz(iz_0b: int) -> tuple[float, ...]:
    z = _FREE_ZS[iz_0b]
    if iz_0b < 5:
        return (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 - z)
    if iz_0b == 5:
        return (0.0, 0.1, 0.4, 0.9)
    if iz_0b < 13:
        return (0.0, 0.1, 1.0 - z)
    if iz_0b == 13:
        return (0.0, 0.1)
    return (0.0,)


def _format_xz_str(v: float) -> str:
    if v > 0.99999:
        return "100"
    iv = math.floor(100.0 * v + 0.5)
    return f"{iv:02d}" if v > 0.09999 else f"0{iv}"


def _dt_stem(kind: int, ix_1b: int, iz_1b: int) -> str:
    zs = _OPAL_ZS if kind == 0 else _FREE_ZS
    xs = _OPAL_XS[iz_1b - 1] if kind == 0 else _free_xs_for_iz(iz_1b - 1)
    z_str = _format_xz_str(zs[iz_1b - 1])
    x_str = _format_xz_str(xs[ix_1b - 1])
    if kind == 0:
        return f"mesa-eosDT_{z_str}z{x_str}x"
    suffix = "_CO" if iz_1b >= 13 else ""
    return f"mesa-FreeEOS_{z_str}z{x_str}x{suffix}"


def do_eos_interpolations_jax(
    v00: jax.Array,
    v10: jax.Array,
    v01: jax.Array,
    v11: jax.Array,
    x0: jax.Array,
    xget: jax.Array,
    x1: jax.Array,
    y0: jax.Array,
    yget: jax.Array,
    y1: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate all 16 bicubic splines and `(x, y)` partials on a cell."""
    sixth = 1.0 / 6.0
    hx = x1 - x0
    hy = y1 - y0
    hxi = 1.0 / hx
    hyi = 1.0 / hy
    hx2 = hx * hx
    hy2 = hy * hy

    xp = (xget - x0) * hxi
    yp = (yget - y0) * hyi
    xpi = 1.0 - xp
    ypi = 1.0 - yp
    xp2 = xp * xp
    xpi2 = xpi * xpi
    yp2 = yp * yp
    ypi2 = ypi * ypi

    cx = xp * (xp2 - 1.0)
    cxi = xpi * (xpi2 - 1.0)
    cxd = 3.0 * xp2 - 1.0
    cxdi = -3.0 * xpi2 + 1.0
    cy = yp * (yp2 - 1.0)
    cyi = ypi * (ypi2 - 1.0)
    cyd = 3.0 * yp2 - 1.0
    cydi = -3.0 * ypi2 + 1.0

    sixth_hx2 = sixth * hx2
    sixth_hy2 = sixth * hy2
    z36th_hx2_hy2 = sixth_hx2 * sixth_hy2
    sixth_hx = sixth * hx
    sixth_hxi_hy2 = sixth_hy2 * hxi
    z36th_hx_hy2 = sixth_hx * sixth_hy2
    sixth_hx2_hyi = sixth_hx2 * hyi
    sixth_hy = sixth * hy
    z36th_hx2_hy = sixth_hx2 * sixth_hy

    f1_ij, f2_ij, f3_ij, f4_ij = v00[:, 0], v00[:, 1], v00[:, 2], v00[:, 3]
    f1_10, f2_10, f3_10, f4_10 = v10[:, 0], v10[:, 1], v10[:, 2], v10[:, 3]
    f1_01, f2_01, f3_01, f4_01 = v01[:, 0], v01[:, 1], v01[:, 2], v01[:, 3]
    f1_11, f2_11, f3_11, f4_11 = v11[:, 0], v11[:, 1], v11[:, 2], v11[:, 3]

    fval = (
        xpi * (ypi * f1_ij + yp * f1_01)
        + xp * (ypi * f1_10 + yp * f1_11)
        + sixth_hx2
        * (cxi * (ypi * f2_ij + yp * f2_01) + cx * (ypi * f2_10 + yp * f2_11))
        + sixth_hy2
        * (xpi * (cyi * f3_ij + cy * f3_01) + xp * (cyi * f3_10 + cy * f3_11))
        + z36th_hx2_hy2
        * (cxi * (cyi * f4_ij + cy * f4_01) + cx * (cyi * f4_10 + cy * f4_11))
    )
    df_dx = (
        hxi * (-(ypi * f1_ij + yp * f1_01) + (ypi * f1_10 + yp * f1_11))
        + sixth_hx
        * (cxdi * (ypi * f2_ij + yp * f2_01) + cxd * (ypi * f2_10 + yp * f2_11))
        + sixth_hxi_hy2
        * (-(cyi * f3_ij + cy * f3_01) + (cyi * f3_10 + cy * f3_11))
        + z36th_hx_hy2
        * (cxdi * (cyi * f4_ij + cy * f4_01) + cxd * (cyi * f4_10 + cy * f4_11))
    )
    df_dy = (
        hyi * (xpi * (-f1_ij + f1_01) + xp * (-f1_10 + f1_11))
        + sixth_hx2_hyi * (cxi * (-f2_ij + f2_01) + cx * (-f2_10 + f2_11))
        + sixth_hy
        * (
            xpi * (cydi * f3_ij + cyd * f3_01)
            + xp * (cydi * f3_10 + cyd * f3_11)
        )
        + z36th_hx2_hy
        * (
            cxi * (cydi * f4_ij + cyd * f4_01)
            + cx * (cydi * f4_10 + cyd * f4_11)
        )
    )
    return fval, df_dx, df_dy


@jax.jit
def _eval_dt_cell_16_jit(
    v00: jax.Array,
    v10: jax.Array,
    v01: jax.Array,
    v11: jax.Array,
    q0: jax.Array,
    log_q: jax.Array,
    q1: jax.Array,
    t0: jax.Array,
    log_t: jax.Array,
    t1: jax.Array,
    oob: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    fval, df_dx, df_dy = do_eos_interpolations_jax(
        v00, v10, v01, v11, q0, log_q, q1, t0, log_t, t1
    )
    dlnd = df_dx / constants.LN10
    dlnt = df_dy / constants.LN10 - 2.0 * dlnd
    bad = oob | (fval[core.I_LN_S] > constants.LN10 * 100.0)
    fval = jnp.where(bad, jnp.nan, fval)
    return fval, dlnd, dlnt


def _lookup_dt_cell_np(
    lq_min: float,
    dq: float,
    n_q: int,
    lt_min: float,
    dt: float,
    n_t: int,
    coeffs: np.ndarray,
    lq_a: np.ndarray,
    lt_a: np.ndarray,
) -> tuple[np.ndarray, ...]:
    lq_f = np.asarray(lq_a, dtype=np.float64)
    lt_f = np.asarray(lt_a, dtype=np.float64)
    iq_raw = np.floor((lq_f - lq_min) / dq + 1.0e-4).astype(np.int64)
    jt_raw = np.floor((lt_f - lt_min) / dt + 1.0e-4).astype(np.int64)
    oob = (
        (iq_raw < 0)
        | (iq_raw >= n_q - 1)
        | (jt_raw < 0)
        | (jt_raw >= n_t - 1)
    )
    iq0 = np.clip(iq_raw, 0, max(0, n_q - 2))
    jt0 = np.clip(jt_raw, 0, max(0, n_t - 2))
    q0 = np.asarray(lq_min + iq0.astype(np.float64) * dq, dtype=np.float64)
    t0 = np.asarray(lt_min + jt0.astype(np.float64) * dt, dtype=np.float64)
    return (
        np.ascontiguousarray(coeffs[iq0, jt0], dtype=np.float64),
        np.ascontiguousarray(coeffs[iq0 + 1, jt0], dtype=np.float64),
        np.ascontiguousarray(coeffs[iq0, jt0 + 1], dtype=np.float64),
        np.ascontiguousarray(coeffs[iq0 + 1, jt0 + 1], dtype=np.float64),
        q0,
        t0,
        np.asarray(oob, dtype=np.bool_),
    )


def _eval_kind_ix_iz_16(
    kind: int, ix_1b: int, iz_1b: int, log_rho: jax.Array, log_t: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    stem = _dt_stem(kind, ix_1b, iz_1b)
    lq_min, dq, n_q, lt_min, dt, n_t, coeffs = _cache_io.load_dt_slice_jax(
        kind, ix_1b, iz_1b, stem
    )
    log_q = log_rho - 2.0 * log_t + 12.0
    lq_sg = jax.lax.stop_gradient(jnp.asarray(log_q, dtype=jnp.float64))
    lt_sg = jax.lax.stop_gradient(jnp.asarray(log_t, dtype=jnp.float64))
    if not isinstance(lq_sg, jax.core.Tracer) and not isinstance(
        lt_sg, jax.core.Tracer
    ):
        v00_n, v10_n, v01_n, v11_n, q0_n, t0_n, oob_n = _lookup_dt_cell_np(
            lq_min,
            dq,
            n_q,
            lt_min,
            dt,
            n_t,
            coeffs,
            np.asarray(lq_sg),
            np.asarray(lt_sg),
        )
        v00 = jnp.asarray(v00_n, dtype=jnp.float64)
        v10 = jnp.asarray(v10_n, dtype=jnp.float64)
        v01 = jnp.asarray(v01_n, dtype=jnp.float64)
        v11 = jnp.asarray(v11_n, dtype=jnp.float64)
        q0 = jnp.asarray(q0_n, dtype=jnp.float64)
        t0 = jnp.asarray(t0_n, dtype=jnp.float64)
        oob = jnp.asarray(oob_n, dtype=jnp.bool_)
    else:
        b_shape = lq_sg.shape
        s_v = jax.ShapeDtypeStruct((*b_shape, 16, 4), jnp.float64)
        s_s = jax.ShapeDtypeStruct(b_shape, jnp.float64)
        s_b = jax.ShapeDtypeStruct(b_shape, jnp.bool_)
        v00, v10, v01, v11, q0, t0, oob = jax.pure_callback(
            lambda q_a, t_a: _lookup_dt_cell_np(
                lq_min, dq, n_q, lt_min, dt, n_t, coeffs, q_a, t_a
            ),
            (s_v, s_v, s_v, s_v, s_s, s_s, s_b),
            lq_sg,
            lt_sg,
            vmap_method="expand_dims",
        )
    f, dlnd, dlnt = _eval_dt_cell_16_jit(
        v00,
        v10,
        v01,
        v11,
        q0,
        log_q,
        q0 + dq,
        t0,
        log_t,
        t0 + dt,
        oob,
    )
    return f, dlnd, dlnt


def _eval_for_x_16(
    kind: int,
    iz_1b: int,
    x: float,
    log_rho: jax.Array,
    log_t: jax.Array,
    use_linear: bool = False,
    tiny: float = 1.0e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    xs = _OPAL_XS[iz_1b - 1] if kind == 0 else _free_xs_for_iz(iz_1b - 1)
    n_xs = len(xs)
    if x < tiny or n_xs == 1:
        return _eval_kind_ix_iz_16(kind, 1, iz_1b, log_rho, log_t)
    if x >= xs[n_xs - 1]:
        return _eval_kind_ix_iz_16(kind, n_xs, iz_1b, log_rho, log_t)

    def _linear() -> tuple[jax.Array, jax.Array, jax.Array]:
        ix_1b = 2
        for cand in range(2, n_xs + 1):
            if xs[cand - 1] >= x:
                ix_1b = cand
                break
        r1, d1, t1 = _eval_kind_ix_iz_16(kind, ix_1b - 1, iz_1b, log_rho, log_t)
        r2, d2, t2 = _eval_kind_ix_iz_16(kind, ix_1b, iz_1b, log_rho, log_t)
        x1, x2 = xs[ix_1b - 2], xs[ix_1b - 1]
        alfa = (x - x2) / (x1 - x2)
        beta = 1.0 - alfa
        return (
            alfa * r1 + beta * r2,
            alfa * d1 + beta * d2,
            alfa * t1 + beta * t2,
        )

    if use_linear or n_xs == 2:
        return _linear()

    if x <= xs[1]:
        ix_lo, ix_hi = 1, 3
    elif x >= xs[n_xs - 2]:
        ix_lo, ix_hi = n_xs - 2, n_xs
    else:
        ix_lo, ix_hi = 1, 3
        for ix in range(3, n_xs):
            if x <= xs[ix - 1]:
                ix_lo, ix_hi = ix - 2, ix + 1
                break

    dx1 = xs[ix_lo] - xs[ix_lo - 1]
    dx2 = xs[ix_lo + 1] - xs[ix_lo]
    eq_sp = (
        abs(dx1 - dx2) <= tiny
        if (ix_hi - ix_lo == 2)
        else (
            abs(dx1 - dx2) <= tiny
            and abs(dx2 - (xs[ix_hi - 1] - xs[ix_lo + 1])) <= tiny
        )
    )
    if not eq_sp:
        return _linear()

    outs = [
        _eval_kind_ix_iz_16(kind, ix, iz_1b, log_rho, log_t)
        for ix in range(ix_lo, ix_hi + 1)
    ]
    del_x = x - xs[ix_lo - 1]
    dx = dx1
    if ix_hi - ix_lo == 2:
        denom = 2.0 * dx * dx
        c0 = (2.0 * dx * dx - 3.0 * dx * del_x + del_x * del_x) / denom
        c1 = 2.0 * (2.0 * dx - del_x) * del_x / denom
        c2 = del_x * (del_x - dx) / denom
        return (
            c0 * outs[0][0] + c1 * outs[1][0] + c2 * outs[2][0],
            c0 * outs[0][1] + c1 * outs[1][1] + c2 * outs[2][1],
            c0 * outs[0][2] + c1 * outs[1][2] + c2 * outs[2][2],
        )
    coef = max(0.0, min(1.0, (x - xs[ix_lo]) / dx))
    c0 = -coef * (coef - 1.0) * (coef - 1.0) / 2.0
    c1 = (2.0 - coef * coef * (5.0 - 3.0 * coef)) / 2.0
    c2 = coef * (1.0 + coef * (4.0 - 3.0 * coef)) / 2.0
    c3 = coef * coef * (coef - 1.0) / 2.0
    return (
        c0 * outs[0][0]
        + (c1 * outs[1][0] + (c2 * outs[2][0] + c3 * outs[3][0])),
        c0 * outs[0][1]
        + (c1 * outs[1][1] + (c2 * outs[2][1] + c3 * outs[3][1])),
        c0 * outs[0][2]
        + (c1 * outs[1][2] + (c2 * outs[2][2] + c3 * outs[3][2])),
    )


def eval_eosdt_table_jax(
    kind: int, rho: jax.Array, temp: jax.Array, x: float, z: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate 26-channel `OPAL/SCVH` (`0`) or `FreeEOS` (`1`) in JAX."""
    log_rho = jnp.log10(rho)
    log_t = jnp.log10(temp)
    zs = _OPAL_ZS if kind == 0 else _FREE_ZS
    n_zs = len(zs)
    if z <= max(zs[0], 1.0e-20):
        f16, d16, t16 = _eval_for_x_16(kind, 1, x, log_rho, log_t)
    elif z >= zs[n_zs - 1]:
        f16, d16, t16 = _eval_for_x_16(kind, n_zs, x, log_rho, log_t)
    else:
        iz_hi = 2
        for iz in range(2, n_zs + 1):
            if z < zs[iz - 1]:
                iz_hi = iz
                break
        iz1, iz2 = iz_hi - 1, iz_hi
        z1, z2 = zs[iz1 - 1], zs[iz2 - 1]
        c2 = (z - z1) / (z2 - z1)
        c1 = 1.0 - c2
        r1, d1, t1 = _eval_for_x_16(kind, iz1, x, log_rho, log_t)
        r2, d2, t2 = _eval_for_x_16(kind, iz2, x, log_rho, log_t)
        f16 = c1 * r1 + c2 * r2
        d16 = c1 * d1 + c2 * d2
        t16 = c1 * t1 + c2 * t2

    pad10 = jnp.zeros(10, dtype=jnp.float64)
    frac_idx = core.I_FRAC_OPAL_SCVH if kind == 0 else core.I_FRAC_FREE_EOS
    res = jnp.concatenate([f16, pad10]).at[frac_idx].set(1.0)
    dlnd = jnp.concatenate([d16, pad10])
    dlnt = jnp.concatenate([t16, pad10])
    return res, dlnd, dlnt


@jax.jit
def _eval_cms_cell_16_jit(
    v00: jax.Array,
    v10: jax.Array,
    v01: jax.Array,
    v11: jax.Array,
    t0: jax.Array,
    my_lt: jax.Array,
    t1: jax.Array,
    r0: jax.Array,
    my_lr: jax.Array,
    r1: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    fval, df_dx, df_dy = do_eos_interpolations_jax(
        v00, v10, v01, v11, t0, my_lt, t1, r0, my_lr, r1
    )
    iln10 = 1.0 / constants.LN10
    dlnt = df_dx * iln10
    dlnd = df_dy * iln10
    cp_val = jnp.exp(fval[core.I_CP])
    cv_val = jnp.exp(fval[core.I_CV])
    fval = fval.at[core.I_CP].set(cp_val).at[core.I_CV].set(cv_val)
    dlnt = (
        dlnt.at[core.I_CP]
        .set(dlnt[core.I_CP] * cp_val)
        .at[core.I_CV]
        .set(dlnt[core.I_CV] * cv_val)
    )
    dlnd = (
        dlnd.at[core.I_CP]
        .set(dlnd[core.I_CP] * cp_val)
        .at[core.I_CV]
        .set(dlnd[core.I_CV] * cv_val)
    )
    return fval, dlnt, dlnd


def _lookup_cms_cell_np(
    lt_min: float,
    dt: float,
    n_t: int,
    lr_min: float,
    dr: float,
    n_r: int,
    coeffs: np.ndarray,
    lr_a: np.ndarray,
    lt_a: np.ndarray,
) -> tuple[np.ndarray, ...]:
    lt_f = np.asarray(lt_a, dtype=np.float64)
    lr_f = np.asarray(lr_a, dtype=np.float64)
    it0 = np.clip(
        np.floor((lt_f - lt_min) / dt + 1.0e-4).astype(np.int64),
        0,
        max(0, n_t - 2),
    )
    ir0 = np.clip(
        np.floor((lr_f - lr_min) / dr + 1.0e-4).astype(np.int64),
        0,
        max(0, n_r - 2),
    )
    t0 = np.asarray(lt_min + it0.astype(np.float64) * dt, dtype=np.float64)
    r0 = np.asarray(lr_min + ir0.astype(np.float64) * dr, dtype=np.float64)
    return (
        np.ascontiguousarray(coeffs[it0, ir0], dtype=np.float64),
        np.ascontiguousarray(coeffs[it0 + 1, ir0], dtype=np.float64),
        np.ascontiguousarray(coeffs[it0, ir0 + 1], dtype=np.float64),
        np.ascontiguousarray(coeffs[it0 + 1, ir0 + 1], dtype=np.float64),
        t0,
        r0,
    )


def _eval_cms_slice_16(
    lt_min: float,
    dt: float,
    n_t: int,
    lr_min: float,
    dr: float,
    n_r: int,
    coeffs: np.ndarray,
    log_rho: jax.Array,
    log_t: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    lr_sg = jax.lax.stop_gradient(jnp.asarray(log_rho, dtype=jnp.float64))
    lt_sg = jax.lax.stop_gradient(jnp.asarray(log_t, dtype=jnp.float64))
    if not isinstance(lr_sg, jax.core.Tracer) and not isinstance(
        lt_sg, jax.core.Tracer
    ):
        v00_n, v10_n, v01_n, v11_n, t0_n, r0_n = _lookup_cms_cell_np(
            lt_min,
            dt,
            n_t,
            lr_min,
            dr,
            n_r,
            coeffs,
            np.asarray(lr_sg),
            np.asarray(lt_sg),
        )
        v00 = jnp.asarray(v00_n, dtype=jnp.float64)
        v10 = jnp.asarray(v10_n, dtype=jnp.float64)
        v01 = jnp.asarray(v01_n, dtype=jnp.float64)
        v11 = jnp.asarray(v11_n, dtype=jnp.float64)
        t0 = jnp.asarray(t0_n, dtype=jnp.float64)
        r0 = jnp.asarray(r0_n, dtype=jnp.float64)
    else:
        b_shape = lr_sg.shape
        s_v = jax.ShapeDtypeStruct((*b_shape, 16, 4), jnp.float64)
        s_s = jax.ShapeDtypeStruct(b_shape, jnp.float64)
        v00, v10, v01, v11, t0, r0 = jax.pure_callback(
            lambda lr_a, lt_a: _lookup_cms_cell_np(
                lt_min, dt, n_t, lr_min, dr, n_r, coeffs, lr_a, lt_a
            ),
            (s_v, s_v, s_v, s_v, s_s, s_s),
            lr_sg,
            lt_sg,
            vmap_method="expand_dims",
        )
    my_lt = jnp.clip(log_t, lt_min, lt_min + (n_t - 1) * dt)
    my_lr = jnp.clip(log_rho, lr_min, lr_min + (n_r - 1) * dr)
    f, dlnt, dlnd = _eval_cms_cell_16_jit(
        v00,
        v10,
        v01,
        v11,
        t0,
        my_lt,
        t0 + dt,
        r0,
        my_lr,
        r0 + dr,
    )
    return f, dlnt, dlnd


def eval_cms_eos_jax(
    rho: jax.Array, temp: jax.Array, x: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate 26-channel `CMS` EOS in pure JAX."""
    log_rho = jnp.log10(rho)
    log_t = jnp.log10(temp)
    if x <= _CMS_XVALS[0]:
        ix = 1
    elif x >= _CMS_XVALS[-2]:
        ix = len(_CMS_XVALS) - 1
    else:
        ix = 1
        for i in range(2, len(_CMS_XVALS) + 1):
            if x < _CMS_XVALS[i - 1]:
                ix = i - 1
                break
    t1 = _cache_io.load_cms_slice_jax(ix, f"mesa-CMS_{_CMS_XSTR[ix - 1]}x")
    t2 = _cache_io.load_cms_slice_jax(ix + 1, f"mesa-CMS_{_CMS_XSTR[ix]}x")
    r1, dt1, dr1 = _eval_cms_slice_16(*t1, log_rho, log_t)
    r2, dt2, dr2 = _eval_cms_slice_16(*t2, log_rho, log_t)
    beta = (x - _CMS_XVALS[ix - 1]) / (_CMS_XVALS[ix] - _CMS_XVALS[ix - 1])
    alfa = 1.0 - beta
    f16 = alfa * r1 + beta * r2
    dlnt16 = alfa * dt1 + beta * dt2
    dlnd16 = alfa * dr1 + beta * dr2
    pad10 = jnp.zeros(10, dtype=jnp.float64)
    res = jnp.concatenate([f16, pad10]).at[core.I_FRAC_CMS].set(1.0)
    return (
        res,
        jnp.concatenate([dlnd16, pad10]),
        jnp.concatenate([dlnt16, pad10]),
    )
