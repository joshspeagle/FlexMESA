# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pure `@jax.jit` evaluators for Type 2 CO, `AESOPUS`, and `kapCN` tables."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa.kap import jax_interp, tables_co_lowt_jax, tables_jax

with jax.ensure_compile_time_eval():
    _KAP_CO_DXS_JAX = jnp.asarray(tables_jax.KAP_CO_DXS, dtype=jnp.float64)

_get_co_all_z_coeffs = tables_co_lowt_jax.get_co_all_z_coeffs_jax
_get_aesopus_all_z_coeffs = tables_co_lowt_jax.get_aesopus_all_z_coeffs_jax


def _find_co_tables_jax(
    dxco_max: jax.Array, dxc: jax.Array, dxo: jax.Array,
    dxc_l: jax.Array, dxo_l: jax.Array,
    num_co: jax.Array, num_gt: jax.Array, tbl_nums: jax.Array,
    next_dxc: jax.Array, next_dxo: jax.Array,
    dxc_tab: jax.Array, dxo_tab: jax.Array,
    dxc_l_tab: jax.Array, dxo_l_tab: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Locate up to 4 bounding `(dXC, dXO)` sub-tables (`Find_CO_Tables`)."""
    tiny = 1.0e-7
    idxc = jnp.clip(
        jnp.searchsorted(_KAP_CO_DXS_JAX[1:], dxc, side="left") + 1, 1, 7
    )
    idxo = jnp.clip(
        jnp.searchsorted(_KAP_CO_DXS_JAX[1:], dxo, side="left") + 1, 1, 7
    )

    def _s(idx_1b: jax.Array) -> jax.Array:
        return jnp.clip(idx_1b - 1, 0, 59)

    def _matches(idx_1b: jax.Array) -> jax.Array:
        s = _s(idx_1b)
        return (
            (idx_1b >= 1)
            & (jnp.abs(dxc_l - dxc_l_tab[s]) < tiny)
            & (jnp.abs(dxo_l - dxo_l_tab[s]) < tiny)
        )

    def _on_mid(idx_1b: jax.Array) -> jax.Array:
        s = _s(idx_1b)
        return jnp.abs(dxc_tab[s] - dxo_tab[s]) < tiny

    def _on_diag(idx_1b: jax.Array) -> jax.Array:
        s = _s(idx_1b)
        return jnp.abs((dxc_tab[s] + dxo_tab[s]) - dxco_max) < tiny

    i1_0 = tbl_nums[idxc - 1, idxo - 1]
    c_ge_o = dxc >= dxo
    i2_0 = jnp.where(
        c_ge_o, tbl_nums[idxc, idxo - 1], tbl_nums[idxc - 1, idxo]
    )
    i3_0 = jnp.where(
        c_ge_o, tbl_nums[idxc - 1, idxo], tbl_nums[idxc, idxo - 1]
    )
    i4_0 = tbl_nums[idxc, idxo]
    a_i1 = jnp.where(
        _matches(i2_0), i2_0,
        jnp.where(_matches(i3_0), i3_0, jnp.where(_matches(i4_0), i4_0, i1_0)),
    )
    a_m = _matches(i2_0) | _matches(i3_0) | _matches(i4_0)
    a_res = (
        a_i1, jnp.where(a_m, -1, i2_0),
        jnp.where(a_m, -1, i3_0), jnp.where(a_m, -1, i4_0),
    )
    s1_0 = _s(i1_0)
    b_res = (
        i1_0,
        jnp.where(c_ge_o, next_dxc[s1_0], next_dxo[s1_0]),
        jnp.where(c_ge_o, next_dxo[s1_0], next_dxc[s1_0]),
        i4_0,
    )
    next_pri = jnp.where(c_ge_o, next_dxc, next_dxo)
    pri_l = jnp.where(c_ge_o, dxc_l, dxo_l)
    sec_l = jnp.where(c_ge_o, dxo_l, dxc_l)
    pri_l_tab = jnp.where(c_ge_o, dxc_l_tab, dxo_l_tab)
    sec_l_tab = jnp.where(c_ge_o, dxo_l_tab, dxc_l_tab)
    diag_i1 = _on_diag(i1_0)
    i3_def = jnp.where(
        c_ge_o,
        jnp.where(diag_i1, num_co, num_gt),
        jnp.where(diag_i1, num_gt, num_co),
    )
    early_1 = (i3_0 <= 0) & (i2_0 > 0) & jnp.where(c_ge_o, diag_i1, ~diag_i1)
    i3_c = jnp.where(i3_0 <= 0, i3_def, i3_0)
    diag_i3 = _on_diag(i3_c)
    i4_c = jnp.where(diag_i3, i4_0, next_pri[_s(i3_c)])
    early_2 = (~diag_i3) & (~_on_diag(i4_c))
    i2_c = jnp.where(i2_0 <= 0, next_pri[_s(i1_0)], i2_0)
    diag_i2 = _on_diag(i2_c)
    s2_c, s4_c = _s(i2_c), _s(i4_c)
    denom_s = sec_l_tab[s2_c] - sec_l_tab[s4_c]
    safe_denom_s = jnp.where(jnp.abs(denom_s) == 0.0, 1.0, denom_s)
    line_24 = pri_l_tab[s4_c] + (pri_l_tab[s2_c] - pri_l_tab[s4_c]) * (
        sec_l - sec_l_tab[s4_c]
    ) / safe_denom_s
    keep_quad = diag_i2 | (pri_l <= pri_l_tab[s4_c]) | (pri_l <= line_24)
    i1_s = i2_c
    i2_s = next_pri[_s(i1_s)]
    early_3 = ~_on_diag(i2_s)
    i3_s = jnp.where(c_ge_o, i4_c, num_co)
    i4_s = jnp.asarray(-1, dtype=jnp.int32)
    early_12 = early_1 | early_2
    c_i1 = jnp.where(early_12, i1_0, jnp.where(keep_quad, i1_0, i1_s))
    c_i2 = jnp.where(
        early_12, -1, jnp.where(keep_quad, i2_c, jnp.where(early_3, -1, i2_s))
    )
    c_i3 = jnp.where(
        early_12, -1, jnp.where(keep_quad, i3_c, jnp.where(early_3, -1, i3_s))
    )
    c_i4 = jnp.where(early_12, -1, jnp.where(keep_quad, i4_c, i4_s))
    m1 = _matches(i1_0)
    use_a = (~m1) & (i4_0 > 0)
    use_b = (~m1) & (i4_0 <= 0) & _on_mid(i1_0)
    return (
        jnp.where(
            m1, i1_0,
            jnp.where(use_a, a_res[0], jnp.where(use_b, b_res[0], c_i1)),
        ),
        jnp.where(
            m1, -1,
            jnp.where(use_a, a_res[1], jnp.where(use_b, b_res[1], c_i2)),
        ),
        jnp.where(
            m1, -1,
            jnp.where(use_a, a_res[2], jnp.where(use_b, b_res[2], c_i3)),
        ),
        jnp.where(
            m1, -1,
            jnp.where(use_a, a_res[3], jnp.where(use_b, b_res[3], c_i4)),
        ),
    )


def _lerp_co_pair_jax(
    target_l: jax.Array, pri_a_l: jax.Array, sec_a_l: jax.Array,
    lk_a: jax.Array, dr_a: jax.Array, dt_a: jax.Array,
    pri_b_l: jax.Array, sec_b_l: jax.Array,
    lk_b: jax.Array, dr_b: jax.Array, dt_b: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    diff = pri_a_l - pri_b_l
    safe_diff = jnp.where(jnp.abs(diff) == 0.0, 1.0, diff)
    alfa = jnp.where(
        jnp.abs(diff) == 0.0, 0.0, (target_l - pri_b_l) / safe_diff
    )
    beta = 1.0 - alfa
    sec_ab = sec_b_l + (sec_a_l - sec_b_l) * alfa
    return (
        alfa * lk_a + beta * lk_b,
        alfa * dr_a + beta * dr_b,
        alfa * dt_a + beta * dt_b,
        sec_ab,
    )


def _eval_co_for_dxco_jax(
    iz: jax.Array, ix: jax.Array, dxc_in: jax.Array, dxo_in: jax.Array,
    log_rho: jax.Array, log_t_in: jax.Array,
    zs: jax.Array, xs: jax.Array, bounds: jax.Array,
    num_co: jax.Array, num_gt: jax.Array, tbl_nums: jax.Array,
    next_dxc: jax.Array, next_dxo: jax.Array,
    dxc_tab: jax.Array, dxo_tab: jax.Array,
    dxc_l_tab: jax.Array, dxo_l_tab: jax.Array,
    log_rs: jax.Array, log_ts: jax.Array, coeffs_all: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate CO opacity at `(iz, ix)` across `(dXC, dXO, logRho, logT)`."""
    x_v, z_v = xs[iz, ix], zs[iz]
    dxc0 = jnp.maximum(dxc_in, 0.0)
    dxo0 = jnp.maximum(dxo_in, 0.0)
    dxco_max = 1.0 - (x_v + z_v)
    sum_co = dxc0 + dxo0
    scale = jnp.where(
        sum_co > dxco_max, dxco_max / jnp.maximum(sum_co, 1.0e-30), 1.0
    )
    dxc = dxc0 * scale
    dxo = dxo0 * scale
    dxc_l = jnp.log10(z_v + 1.0e-3 + dxc)
    dxo_l = jnp.log10(z_v + 1.0e-3 + dxo)

    i1, i2, i3, i4 = _find_co_tables_jax(
        dxco_max, dxc, dxo, dxc_l, dxo_l, num_co[iz, ix], num_gt[iz, ix],
        tbl_nums[iz, ix], next_dxc[iz, ix], next_dxo[iz, ix],
        dxc_tab[iz, ix], dxo_tab[iz, ix], dxc_l_tab[iz, ix], dxo_l_tab[iz, ix],
    )
    bds = bounds[iz, ix]
    lr_min, lr_max, lt_min, lt_max = bds[0], bds[1], bds[2], bds[3]
    lr_in = log_rho - 3.0 * log_t_in + 18.0
    clipped_r = lr_in > lr_max
    lr_c = jnp.minimum(lr_in, lr_max)
    ir, lr_loc, lr0, lr1 = jax_interp._locate_log_jax(
        lr_min, lr_max, log_rs, lr_c
    )
    jt, lt_loc, lt0, lt1 = jax_interp._locate_log_jax(
        lt_min, lt_max, log_ts, log_t_in
    )

    def _eval_sub(ico_1b: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        s = jnp.clip(ico_1b - 1, 0, 59)
        lk, df_dx, df_dy = jax_interp.do_kap_interpolations_jax(
            coeffs_all[iz, ix, s], ir, jt, lr0, lr_loc, lr1, lt0, lt_loc, lt1
        )
        df_dx = jnp.where(clipped_r, 0.0, df_dx)
        return lk, df_dx, df_dy - 3.0 * df_dx

    lk1, dr1, dt1 = _eval_sub(i1)
    lk2, dr2, dt2 = _eval_sub(jnp.where(i2 > 0, i2, i1))
    lk3, dr3, dt3 = _eval_sub(jnp.where(i3 > 0, i3, i1))
    i4_eff = jnp.where(i4 > 0, i4, jnp.where(i3 > 0, i3, i1))
    lk4, dr4, dt4 = _eval_sub(i4_eff)

    s1 = jnp.clip(i1 - 1, 0, 59)
    s2 = jnp.clip(jnp.where(i2 > 0, i2, i1) - 1, 0, 59)
    s3 = jnp.clip(jnp.where(i3 > 0, i3, i1) - 1, 0, 59)
    s4 = jnp.clip(i4_eff - 1, 0, 59)

    c_l, o_l = dxc_l_tab[iz, ix], dxo_l_tab[iz, ix]
    c_ge_o = dxc >= dxo
    t_l = jnp.where(c_ge_o, dxo_l, dxc_l)
    p2 = jnp.where(c_ge_o, o_l[s2], c_l[s2])
    q2 = jnp.where(c_ge_o, c_l[s2], o_l[s2])
    p4 = jnp.where(c_ge_o, o_l[s4], c_l[s4])
    q4 = jnp.where(c_ge_o, c_l[s4], o_l[s4])
    p1 = jnp.where(c_ge_o, o_l[s1], c_l[s1])
    q1 = jnp.where(c_ge_o, c_l[s1], o_l[s1])
    p3 = jnp.where(c_ge_o, o_l[s3], c_l[s3])
    q3 = jnp.where(c_ge_o, c_l[s3], o_l[s3])

    v24, dr24, dt24, q24 = _lerp_co_pair_jax(
        t_l, p2, q2, lk2, dr2, dt2, p4, q4, lk4, dr4, dt4
    )
    v13, dr13, dt13, q13 = _lerp_co_pair_jax(
        t_l, p1, q1, lk1, dr1, dt1, p3, q3, lk3, dr3, dt3
    )
    q_target = jnp.where(c_ge_o, dxc_l, dxo_l)
    dq = q13 - q24
    safe_dq = jnp.where(jnp.abs(dq) == 0.0, 1.0, dq)
    alfa = jnp.where(jnp.abs(dq) == 0.0, 0.0, (q_target - q24) / safe_dq)
    beta = 1.0 - alfa
    lk_b = alfa * v13 + beta * v24
    dr_b = alfa * dr13 + beta * dr24
    dt_b = alfa * dt13 + beta * dt24

    def _ex(idx_1b: jax.Array) -> jax.Array:
        s = jnp.clip(idx_1b - 1, 0, 59)
        return (
            (idx_1b >= 1)
            & (jnp.abs(dxc_l - c_l[s]) == 0.0)
            & (jnp.abs(dxo_l - o_l[s]) == 0.0)
        )

    only1 = (i1 > 0) & (i2 <= 0) & (i3 <= 0) & (i4 <= 0)
    ex2, ex3, ex4 = _ex(i2), _ex(i3), (i4 > 0) & _ex(i4)
    return (
        jnp.where(
            only1,
            lk1,
            jnp.where(
                ex2, lk2, jnp.where(ex3, lk3, jnp.where(ex4, lk4, lk_b))
            ),
        ),
        jnp.where(
            only1,
            dr1,
            jnp.where(
                ex2, dr2, jnp.where(ex3, dr3, jnp.where(ex4, dr4, dr_b))
            ),
        ),
        jnp.where(
            only1,
            dt1,
            jnp.where(
                ex2, dt2, jnp.where(ex3, dt3, jnp.where(ex4, dt4, dt_b))
            ),
        ),
    )


def eval_co_kap_jax(
    zbase: jax.Array | float,
    x: jax.Array | float,
    dxc: jax.Array | float,
    dxo: jax.Array | float,
    log_rho: jax.Array | float,
    log_t: jax.Array | float,
    *,
    kap_co_option: str = "gn93_co",
    cubic_x: bool = False,
    cubic_z: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate Type 2 C/O-enhanced opacity in JAX."""
    meta = tables_co_lowt_jax.get_co_meta_jax(kap_co_option)
    coeffs_all = _get_co_all_z_coeffs(kap_co_option)
    zs, xs = meta["zs"], meta["xs"]
    nz, nx = int(zs.shape[0]), int(xs.shape[1])
    zb_a = jnp.asarray(zbase, dtype=jnp.float64)
    x_a = jnp.asarray(x, dtype=jnp.float64)
    dc_a = jnp.asarray(dxc, dtype=jnp.float64)
    do_a = jnp.asarray(dxo, dtype=jnp.float64)
    lr_a = jnp.asarray(log_rho, dtype=jnp.float64)
    lt_a = jnp.asarray(log_t, dtype=jnp.float64)

    def _eval_zx(
        izz: jax.Array, ixx: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _eval_co_for_dxco_jax(
            izz, ixx, dc_a, do_a, lr_a, lt_a, zs, xs, meta["bounds"],
            meta["num_co"], meta["num_gt"], meta["tbl_nums"],
            meta["next_dxc"], meta["next_dxo"], meta["dxc"], meta["dxo"],
            meta["dxc_l"], meta["dxo_l"], meta["log_rs"], meta["log_ts"],
            coeffs_all,
        )

    def _eval_z(izz: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        xs_z = xs[izz]
        x_min, x_max = xs_z[0], xs_z[nx - 1]
        x_c = jnp.clip(x_a, x_min, x_max)
        ix = jnp.clip(jnp.searchsorted(xs_z, x_c, side="right") - 1, 0, nx - 2)
        if cubic_x and nx >= 4:
            i1 = jnp.where(ix + 2 >= nx, nx - 3, jnp.where(ix == 0, 1, ix))
            idxs = jnp.array([i1 - 1, i1, i1 + 1, i1 + 2])
            lks, drs, dts = jax.vmap(lambda ixx: _eval_zx(izz, ixx))(idxs)
            lk_c, dr_c, dt_c = jax_interp.interp_pm_4pt_with_partials_jax(
                xs_z[idxs], lks, drs, dts, x_c
            )
            lk_lo, dr_lo, dt_lo = _eval_zx(izz, jnp.asarray(0))
            lk_hi, dr_hi, dt_hi = _eval_zx(izz, jnp.asarray(nx - 1))
            return (
                jnp.where(
                    x_a <= x_min, lk_lo, jnp.where(x_a >= x_max, lk_hi, lk_c)
                ),
                jnp.where(
                    x_a <= x_min, dr_lo, jnp.where(x_a >= x_max, dr_hi, dr_c)
                ),
                jnp.where(
                    x_a <= x_min, dt_lo, jnp.where(x_a >= x_max, dt_hi, dt_c)
                ),
            )
        lk0, dr0, dt0 = _eval_zx(izz, ix)
        lk1, dr1, dt1 = _eval_zx(izz, ix + 1)
        beta = jnp.where(
            x_a <= x_min,
            1.0,
            jnp.where(
                x_a >= x_max,
                0.0,
                (x_c - xs_z[ix + 1]) / (xs_z[ix] - xs_z[ix + 1]),
            ),
        )
        alfa = 1.0 - beta
        return (
            beta * lk0 + alfa * lk1,
            beta * dr0 + alfa * dr1,
            beta * dt0 + alfa * dt1,
        )

    z_min, z_max = zs[0], zs[nz - 1]
    z_c = jnp.clip(zb_a, z_min, z_max)
    iz = jnp.clip(jnp.searchsorted(zs, z_c, side="right") - 1, 0, nz - 2)
    if cubic_z and nz >= 4:
        i1 = jnp.where(iz + 2 >= nz, nz - 3, jnp.where(iz == 0, 1, iz))
        idxs = jnp.array([i1 - 1, i1, i1 + 1, i1 + 2])
        lks, drs, dts = jax.vmap(_eval_z)(idxs)
        lk_c, dr_c, dt_c = jax_interp.interp_pm_4pt_with_partials_jax(
            zs[idxs], lks, drs, dts, z_c
        )
        lk_lo, dr_lo, dt_lo = _eval_z(jnp.asarray(0))
        lk_hi, dr_hi, dt_hi = _eval_z(jnp.asarray(nz - 1))
        return (
            jnp.where(
                zb_a <= z_min, lk_lo, jnp.where(zb_a >= z_max, lk_hi, lk_c)
            ),
            jnp.where(
                zb_a <= z_min, dr_lo, jnp.where(zb_a >= z_max, dr_hi, dr_c)
            ),
            jnp.where(
                zb_a <= z_min, dt_lo, jnp.where(zb_a >= z_max, dt_hi, dt_c)
            ),
        )
    lk0, dr0, dt0 = _eval_z(iz)
    lk1, dr1, dt1 = _eval_z(iz + 1)
    beta = jnp.where(
        zb_a <= z_min,
        1.0,
        jnp.where(
            zb_a >= z_max, 0.0, (z_c - zs[iz + 1]) / (zs[iz] - zs[iz + 1])
        ),
    )
    alfa = 1.0 - beta
    return (
        beta * lk0 + alfa * lk1,
        beta * dr0 + alfa * dr1,
        beta * dt0 + alfa * dt1,
    )


def _clipped_linear_weights_jax(
    val: jax.Array, vec: jax.Array
) -> tuple[jax.Array, jax.Array]:
    n = vec.shape[0]
    loc = jnp.clip(jnp.searchsorted(vec, val, side="right") - 1, 0, n - 2)
    w2 = (val - vec[loc]) / (vec[loc + 1] - vec[loc])
    w1 = 1.0 - w2
    below = val <= vec[0]
    above = val >= vec[n - 1]
    idx = jnp.where(below, 0, jnp.where(above, n - 1, loc))
    w = jnp.where(
        below | above,
        jnp.array([1.0, 0.0], dtype=jnp.float64),
        jnp.stack([w1, w2]),
    )
    return idx, w


def _eval_aesopus_fixed_zref_jax(
    iz: jax.Array,
    x: jax.Array,
    xc: jax.Array,
    xn: jax.Array,
    xo: jax.Array,
    log_r: jax.Array,
    log_t: jax.Array,
    coeffs_all: jax.Array,
) -> jax.Array:
    meta = tables_co_lowt_jax.get_aesopus_meta_jax()
    coeffs_z = coeffs_all[iz]
    zref_i = meta["zs"][iz]
    xs_z = meta["xs_all"][iz]
    fcos_z = meta["fcos_all"][iz]
    fcs_z = meta["fcs_all"][iz]
    fns_z = meta["fns_all"][iz]
    fco = jnp.log10(jnp.maximum(xc / jnp.maximum(xo, 1.0e-99), 1.0e-99)) - (
        meta["fco_ref"]
    )
    fc = jnp.log10(jnp.maximum(xc / jnp.maximum(zref_i, 1.0e-99), 1.0e-99)) - (
        meta["fc_ref"]
    )
    fn_v = jnp.log10(
        jnp.maximum(xn / jnp.maximum(zref_i, 1.0e-99), 1.0e-99)
    ) - (meta["fn_ref"])
    ix, wx = _clipped_linear_weights_jax(x, xs_z)
    ifco, wfco = _clipped_linear_weights_jax(fco, fcos_z)
    ifc, wfc = _clipped_linear_weights_jax(fc, fcs_z)
    ifn, wfn = _clipped_linear_weights_jax(fn_v, fns_z)
    n_fco, n_fc, n_fn = (
        fcos_z.shape[0],
        fcs_z.shape[0],
        fns_z.shape[0],
    )
    nx_s = xs_z.shape[0]
    lrs, lts = meta["log_rs"], meta["log_ts"]
    res = jnp.zeros(3, dtype=jnp.float64)
    for c in range(16):
        di, dj, dk, dl = (c >> 3) & 1, (c >> 2) & 1, (c >> 1) & 1, c & 1
        w = wx[di] * wfco[dj] * wfc[dk] * wfn[dl]
        ix_c = jnp.minimum(ix + di, nx_s - 1)
        ico_c = jnp.minimum(ifco + dj, n_fco - 1)
        ic_c = jnp.minimum(ifc + dk, n_fc - 1)
        in_c = jnp.minimum(ifn + dl, n_fn - 1)
        flat = ((ix_c * n_fco + ico_c) * n_fc + ic_c) * n_fn + in_c
        f_v, dx_v, dy_v = jax_interp.interp_evbicub_jax(
            log_r, log_t, lrs, lts, coeffs_z[flat]
        )
        res = res + w * jnp.stack([f_v, dx_v, dy_v])
    return res


def eval_aesopus_jax(
    zref: jax.Array | float,
    x: jax.Array | float,
    xc: jax.Array | float,
    xn: jax.Array | float,
    xo: jax.Array | float,
    log_rho: jax.Array | float,
    log_t: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate Marigo & Aringer (2009) `AESOPUS_AGSS09` in JAX."""
    meta = tables_co_lowt_jax.get_aesopus_meta_jax()
    coeffs_all = _get_aesopus_all_z_coeffs()
    zs = meta["zs"]
    log_zs = meta["log_zs"]
    num_zs = int(zs.shape[0])
    z_a = jnp.asarray(zref, dtype=jnp.float64)
    x_a = jnp.asarray(x, dtype=jnp.float64)
    xc_a = jnp.asarray(xc, dtype=jnp.float64)
    xn_a = jnp.asarray(xn, dtype=jnp.float64)
    xo_a = jnp.asarray(xo, dtype=jnp.float64)
    lr_a = jnp.asarray(log_rho, dtype=jnp.float64)
    lt_a = jnp.asarray(log_t, dtype=jnp.float64)
    log_r = lr_a - 3.0 * lt_a + 18.0

    def _eval_z(izz: jax.Array) -> jax.Array:
        return _eval_aesopus_fixed_zref_jax(
            izz, x_a, xc_a, xn_a, xo_a, log_r, lt_a, coeffs_all
        )

    iz = jnp.clip(jnp.searchsorted(zs, z_a, side="right") - 1, 1, num_zs - 2)
    idxs = jnp.stack([iz - 1, iz, iz + 1])
    raw = jax.vmap(_eval_z)(idxs)
    x_old = log_zs[idxs]
    x_new = jnp.log10(jnp.maximum(z_a, 1.0e-99))
    res_mid = jnp.stack(
        [
            jax_interp.interp_pm_1d_jax(x_old, raw[:, c], x_new)
            for c in range(3)
        ]
    )
    res = jnp.where(
        z_a <= zs[0],
        raw[0],
        jnp.where(z_a >= zs[num_zs - 1], raw[2], res_mid),
    )
    return jnp.power(10.0, res[0]), res[1], res[2] - 3.0 * res[1]


def _lagrange_3pt_jax(a: jax.Array, x: jax.Array) -> jax.Array:
    w0 = ((x - a[1]) / (a[0] - a[1])) * ((x - a[2]) / (a[0] - a[2]))
    w1 = ((x - a[0]) / (a[1] - a[0])) * ((x - a[2]) / (a[1] - a[2]))
    w2 = ((x - a[0]) / (a[2] - a[0])) * ((x - a[1]) / (a[2] - a[1]))
    return jnp.stack([w0, w1, w2])


def eval_kapcn_jax(
    z: jax.Array | float,
    x: jax.Array | float,
    fc: jax.Array | float,
    fn_val: jax.Array | float,
    log_rho: jax.Array | float,
    log_t: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate Lederer & Aringer (2009) `kapCN` in JAX."""
    kcn = tables_co_lowt_jax.get_kapcn_jax()
    zs = kcn["zs"]
    nz = int(zs.shape[0])
    z_a = jnp.asarray(z, dtype=jnp.float64)
    x_a = jnp.asarray(x, dtype=jnp.float64)
    fc_a = jnp.asarray(fc, dtype=jnp.float64)
    fn_a = jnp.asarray(fn_val, dtype=jnp.float64)
    lr_a = jnp.asarray(log_rho, dtype=jnp.float64)
    lt_a = jnp.asarray(log_t, dtype=jnp.float64)
    log_r = lr_a - 3.0 * lt_a + 18.0

    def _eval_fixed_z(izz: jax.Array) -> jax.Array:
        fcs_z = kcn["fcs"][izz]
        fns_z = kcn["fns"][izz]
        my_x = jnp.clip(x_a, kcn["xs"][0], kcn["xs"][2])
        my_fc = jnp.clip(fc_a, fcs_z[0], fcs_z[6])
        my_fn = jnp.clip(fn_a, fns_z[0], fns_z[2])
        ifc = jnp.clip(jnp.searchsorted(fcs_z, my_fc, side="right") - 1, 1, 5)
        wx = _lagrange_3pt_jax(kcn["xs"], my_x)
        wfn = _lagrange_3pt_jax(fns_z, my_fn)
        fc_sub = jnp.stack([fcs_z[ifc - 1], fcs_z[ifc], fcs_z[ifc + 1]])
        wfc = _lagrange_3pt_jax(fc_sub, my_fc)
        out = jnp.zeros(3, dtype=jnp.float64)
        for i in range(3):
            for j in range(3):
                for n in range(3):
                    t_idx = 21 * i + 7 * j + (ifc + n - 1)
                    fv, dxv, dyv = jax_interp.interp_evbicub_jax(
                        log_r,
                        lt_a,
                        kcn["log_rs"],
                        kcn["log_ts"],
                        kcn["coeffs"][izz, t_idx],
                    )
                    out = out + (wfn[i] * wx[j] * wfc[n]) * jnp.stack(
                        [fv, dxv, dyv]
                    )
        return out

    my_z = jnp.clip(z_a, zs[0], zs[nz - 1])
    iz = jnp.clip(jnp.searchsorted(zs, z_a, side="right") - 1, 1, nz - 3)
    idxs = jnp.array([iz - 1, iz, iz + 1, iz + 2])
    res_z = jax.vmap(_eval_fixed_z)(idxs)
    log_z_ary = jnp.log10(zs[idxs])
    x_new = jnp.log10(my_z)
    res_interp = jnp.stack(
        [
            jax_interp.interp_pm_1d_jax(log_z_ary, res_z[:, c], x_new)
            for c in range(3)
        ]
    )
    res = jnp.where(jnp.abs(z_a - zs[iz]) == 0.0, res_z[1], res_interp)
    kap = jnp.power(10.0, res[0])
    return kap, res[1], res[2] - 3.0 * res[1]
