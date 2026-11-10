# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pure `@jax.jit` opacity table interpolators (`flexmesa.kap.jax_interp`).

Implements 2D PSPLINE bicubic evaluation (`do_kap_interpolations`,
`interp_evbicub_prevalidated_db`), 1D Steffen (1990) piecewise-monotonic cubic
interpolation (`interp_pm`), Type 1 fixed-metal (`kap_eval_fixed.f90`), Type 2
C/O-enhanced (`kap_eval_co.f90`), `AESOPUS` (`kap_aesopus.f90`), and `kapCN`
(`kapcn.f90`) kernels in `float64` JAX.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa.kap import tables_jax

_ONE_SIXTH: float = 1.0 / 6.0
_Z36TH: float = 1.0 / 36.0
_KAP_CO_DXS_JAX = jnp.asarray(tables_jax.KAP_CO_DXS, dtype=jnp.float64)


def _locate_log_jax(
    log_min_in: jax.Array,
    log_max_in: jax.Array,
    logs: jax.Array,
    log_find: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Locate `log_find` in 1D grid `logs`, clipping to `[log_min, log_max]`."""
    n = logs.shape[0]
    log_min = jnp.maximum(log_min_in, logs[0])
    log_max = jnp.minimum(log_max_in, logs[n - 1])
    lf = jnp.clip(log_find, log_min, log_max)
    raw_idx = jnp.searchsorted(logs, lf, side="right") - 1
    i = jnp.clip(raw_idx, 0, n - 2)
    i = jnp.where(jnp.abs(lf - log_max) < 1.0e-7, n - 2, i)
    lf = jnp.where(jnp.abs(lf - log_max) < 1.0e-7, log_max, lf)
    return i, lf, logs[i], logs[i + 1]


def do_kap_interpolations_jax(
    fin: jax.Array,
    i: jax.Array,
    j: jax.Array,
    x0: jax.Array,
    xget: jax.Array,
    x1: jax.Array,
    y0: jax.Array,
    yget: jax.Array,
    y1: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate PSPLINE bicubic cell `(i, j)` and partials."""
    hx = x1 - x0
    hxi = 1.0 / hx
    hx2 = hx * hx
    xp = (xget - x0) * hxi
    xpi = 1.0 - xp
    xp2 = xp * xp
    xpi2 = xpi * xpi
    cx = xp * (xp2 - 1.0)
    cxi = xpi * (xpi2 - 1.0)
    cxd = 3.0 * xp2 - 1.0
    cxdi = -3.0 * xpi2 + 1.0

    hy = y1 - y0
    hyi = 1.0 / hy
    hy2 = hy * hy
    yp = (yget - y0) * hyi
    ypi = 1.0 - yp
    yp2 = yp * yp
    ypi2 = ypi * ypi
    cy = yp * (yp2 - 1.0)
    cyi = ypi * (ypi2 - 1.0)
    cyd = 3.0 * yp2 - 1.0
    cydi = -3.0 * ypi2 + 1.0

    c00 = fin[i, j]
    c01 = fin[i, j + 1]
    c10 = fin[i + 1, j]
    c11 = fin[i + 1, j + 1]

    y0_f = ypi * c00[0] + yp * c01[0]
    y1_f = ypi * c10[0] + yp * c11[0]
    y0_fxx = ypi * c00[1] + yp * c01[1]
    y1_fxx = ypi * c10[1] + yp * c11[1]
    cy0_fyy = cyi * c00[2] + cy * c01[2]
    cy1_fyy = cyi * c10[2] + cy * c11[2]
    cy0_fxxyy = cyi * c00[3] + cy * c01[3]
    cy1_fxxyy = cyi * c10[3] + cy * c11[3]
    cyd0_fyy = cydi * c00[2] + cyd * c01[2]
    cyd1_fyy = cydi * c10[2] + cyd * c11[2]
    cyd0_fxxyy = cydi * c00[3] + cyd * c01[3]
    cyd1_fxxyy = cydi * c10[3] + cyd * c11[3]

    fval = (
        xpi * y0_f
        + xp * y1_f
        + _ONE_SIXTH * hx2 * (cxi * y0_fxx + cx * y1_fxx)
        + _ONE_SIXTH * hy2 * (xpi * cy0_fyy + xp * cy1_fyy)
        + _Z36TH * hx2 * hy2 * (cxi * cy0_fxxyy + cx * cy1_fxxyy)
    )
    df_dx = (
        hxi * (-y0_f + y1_f)
        + _ONE_SIXTH * hx * (cxdi * y0_fxx + cxd * y1_fxx)
        + _ONE_SIXTH * hxi * hy2 * (-cy0_fyy + cy1_fyy)
        + _Z36TH * hx * hy2 * (cxdi * cy0_fxxyy + cxd * cy1_fxxyy)
    )
    df_dy = (
        hyi * (xpi * (-c00[0] + c01[0]) + xp * (-c10[0] + c11[0]))
        + _ONE_SIXTH
        * hx2
        * hyi
        * (cxi * (-c00[1] + c01[1]) + cx * (-c10[1] + c11[1]))
        + _ONE_SIXTH * hy * (xpi * cyd0_fyy + xp * cyd1_fyy)
        + _Z36TH * hx2 * hy * (cxi * cyd0_fxxyy + cx * cyd1_fxxyy)
    )
    return fval, df_dx, df_dy


def interp_evbicub_jax(
    xget: jax.Array,
    yget: jax.Array,
    x: jax.Array,
    y: jax.Array,
    coeffs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate `interp_evbicub_prevalidated_db` at `(xget, yget)` in JAX."""
    nx = x.shape[0]
    ny = y.shape[0]
    zx = jnp.clip(xget, x[0], x[nx - 1])
    zy = jnp.clip(yget, y[0], y[ny - 1])
    i = jnp.clip(jnp.searchsorted(x, zx, side="right") - 1, 0, nx - 2)
    j = jnp.clip(jnp.searchsorted(y, zy, side="right") - 1, 0, ny - 2)
    return do_kap_interpolations_jax(
        coeffs, i, j, x[i], zx, x[i + 1], y[j], zy, y[j + 1]
    )


def _fortran_sign1(val: jax.Array) -> jax.Array:
    return jnp.where(jax.lax.stop_gradient(val) < 0.0, -1.0, 1.0)


def _dual_abs(val: jax.Array) -> jax.Array:
    return jnp.where(jax.lax.stop_gradient(val) < 0.0, -val, val)


def _dual_min(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.where(jax.lax.stop_gradient(a) <= jax.lax.stop_gradient(b), a, b)


def _limit_endpoint_slope(p: jax.Array, s: jax.Array) -> jax.Array:
    ps = jax.lax.stop_gradient(p * s)
    p_abs = jax.lax.stop_gradient(_dual_abs(p))
    s_abs2 = jax.lax.stop_gradient(2.0 * _dual_abs(s))
    return jnp.where(ps <= 0.0, 0.0 * p, jnp.where(p_abs > s_abs2, 2.0 * s, p))


def interp_pm_1d_jax(
    x_old: jax.Array,
    y_old: jax.Array,
    x_new: jax.Array,
) -> jax.Array:
    """Evaluate 1D Steffen (1990) piecewise-monotonic cubic at `x_new`."""
    n = x_old.shape[0]
    h = x_old[1:] - x_old[:-1]
    s = (y_old[1:] - y_old[:-1]) / h

    p_int = (s[:-1] * h[1:] + s[1:] * h[:-1]) / (h[:-1] + h[1:])
    sign_sum = _fortran_sign1(s[:-1]) + _fortran_sign1(s[1:])
    min_s = _dual_min(
        _dual_min(_dual_abs(s[:-1]), _dual_abs(s[1:])), 0.5 * _dual_abs(p_int)
    )
    c1_int = sign_sum * min_s

    p0 = s[0] * (1.0 + h[0] / (h[0] + h[1])) - s[1] * h[0] / (h[0] + h[1])
    c1_0 = _limit_endpoint_slope(p0, s[0])
    hl, hp = h[n - 2], h[n - 3]
    pe = s[n - 2] * (1.0 + hl / (hl + hp)) - s[n - 3] * hl / (hl + hp)
    c1_n = _limit_endpoint_slope(pe, s[n - 2])

    c1 = jnp.concatenate([c1_0[None], c1_int, c1_n[None]], axis=0)
    c2 = (3.0 * s - 2.0 * c1[:-1] - c1[1:]) / h
    c3 = (c1[:-1] + c1[1:] - 2.0 * s) / (h * h)

    k = jnp.clip(jnp.searchsorted(x_old, x_new, side="right") - 1, 0, n - 2)
    dx = x_new - x_old[k]
    return y_old[k] + dx * (c1[k] + dx * (c2[k] + dx * c3[k]))


def interp_pm_4pt_with_partials_jax(
    x_old: jax.Array,
    v_old: jax.Array,
    dr_old: jax.Array,
    dt_old: jax.Array,
    x_new: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate 4-point Steffen cubic and propagate `(dr, dt)` tangents."""

    def _fn(y: jax.Array) -> jax.Array:
        return interp_pm_1d_jax(x_old, y, x_new)

    val, dr = jax.jvp(_fn, (v_old,), (dr_old,))
    _, dt = jax.jvp(_fn, (v_old,), (dt_old,))
    return val, dr, dt


def _eval_fixed_log_rho_log_t_jax(
    coeffs_zx: jax.Array,
    bounds_zx: jax.Array,
    log_rs: jax.Array,
    log_ts: jax.Array,
    log_rho: jax.Array,
    log_t_in: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate a single fixed `(Z, X)` table at `(log_rho, log_t_in)`."""
    lr_min, lr_max, lt_min, lt_max = (
        bounds_zx[0],
        bounds_zx[1],
        bounds_zx[2],
        bounds_zx[3],
    )
    log_r_in = log_rho - 3.0 * log_t_in + 18.0
    clipped_t = log_t_in < lt_min
    log_t = jnp.maximum(log_t_in, lt_min)
    clipped_r = (log_r_in < lr_min) | (log_r_in > lr_max)
    log_r = jnp.clip(log_r_in, lr_min, lr_max)

    ir, lr_loc, lr0, lr1 = _locate_log_jax(lr_min, lr_max, log_rs, log_r)
    jt, lt_loc, lt0, lt1 = _locate_log_jax(lt_min, lt_max, log_ts, log_t)
    lk, df_dx, df_dy = do_kap_interpolations_jax(
        coeffs_zx, ir, jt, lr0, lr_loc, lr1, lt0, lt_loc, lt1
    )
    df_dx = jnp.where(clipped_r, 0.0, df_dx)
    df_dy = jnp.where(clipped_t, 0.0, df_dy)
    return lk, df_dx, df_dy - 3.0 * df_dx


def _eval_fixed_for_x_jax(
    iz: jax.Array,
    x: jax.Array,
    log_rho: jax.Array,
    log_t: jax.Array,
    num_xs: jax.Array,
    xs_grid: jax.Array,
    bounds: jax.Array,
    log_rs: jax.Array,
    log_ts: jax.Array,
    coeffs: jax.Array,
    cubic_x: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate fixed-metal opacity across `X` at metallicity index `iz`."""
    nx = num_xs[iz]
    xs_z = xs_grid[iz]
    x_min = xs_z[0]
    x_max = xs_z[nx - 1]
    x_clamped = jnp.clip(x, x_min, x_max)
    ix = jnp.clip(
        jnp.searchsorted(xs_z, x_clamped, side="right") - 1, 0, nx - 2
    )

    if cubic_x:
        i1 = jnp.where(ix + 2 >= nx, nx - 3, jnp.where(ix == 0, 1, ix))
        idxs = jnp.array([i1 - 1, i1, i1 + 1, i1 + 2])

        def _eval_ix(ixx: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _eval_fixed_log_rho_log_t_jax(
                coeffs[iz, ixx],
                bounds[iz, ixx],
                log_rs,
                log_ts,
                log_rho,
                log_t,
            )

        lks, drs, dts = jax.vmap(_eval_ix)(idxs)
        lk_c, dr_c, dt_c = interp_pm_4pt_with_partials_jax(
            xs_z[idxs], lks, drs, dts, x_clamped
        )
        lk_lo, dr_lo, dt_lo = _eval_ix(jnp.asarray(0))
        lk_hi, dr_hi, dt_hi = _eval_ix(nx - 1)
        return (
            jnp.where(x <= x_min, lk_lo, jnp.where(x >= x_max, lk_hi, lk_c)),
            jnp.where(x <= x_min, dr_lo, jnp.where(x >= x_max, dr_hi, dr_c)),
            jnp.where(x <= x_min, dt_lo, jnp.where(x >= x_max, dt_hi, dt_c)),
        )

    lk0, dr0, dt0 = _eval_fixed_log_rho_log_t_jax(
        coeffs[iz, ix], bounds[iz, ix], log_rs, log_ts, log_rho, log_t
    )
    lk1, dr1, dt1 = _eval_fixed_log_rho_log_t_jax(
        coeffs[iz, ix + 1], bounds[iz, ix + 1], log_rs, log_ts, log_rho, log_t
    )
    x0, x1 = xs_z[ix], xs_z[ix + 1]
    denom = jnp.where(jnp.abs(x0 - x1) == 0.0, 1.0, x0 - x1)
    beta = jnp.where(
        x <= x_min,
        1.0,
        jnp.where(x >= x_max, 0.0, (x_clamped - x1) / denom),
    )
    alfa = 1.0 - beta
    return (
        beta * lk0 + alfa * lk1,
        beta * dr0 + alfa * dr1,
        beta * dt0 + alfa * dt1,
    )


def eval_fixed_metal_jax(
    z: jax.Array | float,
    x: jax.Array | float,
    log_rho: jax.Array | float,
    log_t: jax.Array | float,
    *,
    prefix: str = "gs98",
    cubic_x: bool = False,
    cubic_z: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate Type 1 fixed-metal or Ferguson/Freedman low-T table in JAX."""
    tbl = tables_jax.get_fixed_set_jax(prefix)
    zs = tbl["zs"]
    num_zs = int(zs.shape[0])
    z_a = jnp.asarray(z, dtype=jnp.float64)
    x_a = jnp.asarray(x, dtype=jnp.float64)
    lr_a = jnp.asarray(log_rho, dtype=jnp.float64)
    lt_a = jnp.asarray(log_t, dtype=jnp.float64)

    z_min, z_max = zs[0], zs[num_zs - 1]
    z_clamped = jnp.clip(z_a, z_min, z_max)
    iz = jnp.clip(
        jnp.searchsorted(zs, z_clamped, side="right") - 1, 0, num_zs - 2
    )

    def _eval_z(izz: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _eval_fixed_for_x_jax(
            izz,
            x_a,
            lr_a,
            lt_a,
            tbl["num_xs"],
            tbl["xs"],
            tbl["bounds"],
            tbl["log_rs"],
            tbl["log_ts"],
            tbl["coeffs"],
            cubic_x,
        )

    if cubic_z and num_zs >= 4:
        i1 = jnp.where(iz + 2 >= num_zs, num_zs - 3, jnp.where(iz == 0, 1, iz))
        idxs = jnp.array([i1 - 1, i1, i1 + 1, i1 + 2])
        lks, drs, dts = jax.vmap(_eval_z)(idxs)
        lk_c, dr_c, dt_c = interp_pm_4pt_with_partials_jax(
            zs[idxs], lks, drs, dts, z_clamped
        )
        lk_lo, dr_lo, dt_lo = _eval_z(jnp.asarray(0))
        lk_hi, dr_hi, dt_hi = _eval_z(jnp.asarray(num_zs - 1))
        return (
            jnp.where(
                z_a <= z_min, lk_lo, jnp.where(z_a >= z_max, lk_hi, lk_c)
            ),
            jnp.where(
                z_a <= z_min, dr_lo, jnp.where(z_a >= z_max, dr_hi, dr_c)
            ),
            jnp.where(
                z_a <= z_min, dt_lo, jnp.where(z_a >= z_max, dt_hi, dt_c)
            ),
        )

    lk0, dr0, dt0 = _eval_z(iz)
    lk1, dr1, dt1 = _eval_z(iz + 1)
    z0, z1 = zs[iz], zs[iz + 1]
    beta = jnp.where(
        z_a <= z_min,
        1.0,
        jnp.where(z_a >= z_max, 0.0, (z_clamped - z1) / (z0 - z1)),
    )
    alfa = 1.0 - beta
    return (
        beta * lk0 + alfa * lk1,
        beta * dr0 + alfa * dr1,
        beta * dt0 + alfa * dt1,
    )
