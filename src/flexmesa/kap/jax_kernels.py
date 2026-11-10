# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native `@jax.jit` (`float64`) opacity kernels for `flexmesa.kap`.

Implements Poutanen (2017) relativistic/degenerate Compton scattering,
Potekhin (1999) / Cassisi et al. (2007) + Blouin et al. (2020) electron thermal
conduction (`condtabl.data` bicubic spline), low-T/high-T/Type-2/Compton
regime blending (`kap_eval.f90`), and `make_differentiable_kap` in pure JAX
with zero host callbacks.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp

from flexmesa import constants
from flexmesa.kap import jax_co_lowt, jax_interp, tables_jax

_KEV_PER_K: float = constants.KEV * 1.0e-3
_THOMSON_OPACITY_COEF: float = constants.SIGE / constants.AMU
_ILN10: float = 1.0 / 2.302585092994046
_LOG10_COND_COEF: float = float(jnp.log10(16.0 * constants.BOLTZ_SIGMA / 3.0))

_REF_ZFRACS: dict[str, tuple[float, float, float, float]] = {
    "gn93_co": (0.173285, 0.053152, 0.482273, 0.098668),
    "gs98_co": (0.165659, 0.048844, 0.453345, 0.114576),
    "a09_co": (0.176246, 0.050427, 0.428279, 0.093860),
}


def _quintic_val_and_deriv(u: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return `S_2(u) = 6u^5 - 15u^4 + 10u^3` and `S_2'(u)`."""
    u2 = u * u
    u3 = u2 * u
    val = -(u3 * ((u * (u * (-6.0) + 15.0)) - 10.0))
    dval = 30.0 * u2 * (u - 1.0) * (u - 1.0)
    return val, dval


@jax.jit
def compton_opacity_jax(
    rho: jax.Array | float,
    t: jax.Array | float,
    lnfree_e: jax.Array | float,
    d_lnfree_e_dln_rho: jax.Array | float,
    d_lnfree_e_dln_t: jax.Array | float,
    eta: jax.Array | float,
    d_eta_dln_rho: jax.Array | float,
    d_eta_dln_t: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate Poutanen (2017) relativistic/degenerate Compton opacity."""
    _ = rho
    t_a = jnp.asarray(t, dtype=jnp.float64)
    lnfe = jnp.asarray(lnfree_e, dtype=jnp.float64)
    dlnfe_dr = jnp.asarray(d_lnfree_e_dln_rho, dtype=jnp.float64)
    dlnfe_dt = jnp.asarray(d_lnfree_e_dln_t, dtype=jnp.float64)
    eta_a = jnp.asarray(eta, dtype=jnp.float64)
    deta_dr = jnp.asarray(d_eta_dln_rho, dtype=jnp.float64)
    deta_dt = jnp.asarray(d_eta_dln_t, dtype=jnp.float64)

    theta_kev = t_a * _KEV_PER_K
    c01, c02 = 0.682, -0.0454
    c11, c12 = 0.240, 0.0043
    c21, c22 = 0.050, -0.0067
    c31, c32 = -0.037, 0.0031
    t0, alpha0 = 43.3, 0.885

    zeta = jnp.exp(c01 * eta_a + c02 * eta_a * eta_a)
    dzeta_deta = zeta * (c01 + 2.0 * c02 * eta_a)
    zeta_sq = zeta * zeta

    f1 = 1.0 + c11 * zeta + c12 * zeta_sq
    df1_deta = (c11 + 2.0 * c12 * zeta) * dzeta_deta
    f2 = 1.0 + c21 * zeta + c22 * zeta_sq
    df2_deta = (c21 + 2.0 * c22 * zeta) * dzeta_deta
    f3 = 1.0 + c31 * zeta + c32 * zeta_sq
    df3_deta = (c31 + 2.0 * c32 * zeta) * dzeta_deta

    alpha = alpha0 * f3
    dalpha_deta = alpha0 * df3_deta
    tbr = t0 * f2
    dtbr_deta = t0 * df2_deta

    ratio = theta_kev / tbr
    ln_ratio = jnp.log(ratio)
    ratio_pow = jnp.exp(alpha * ln_ratio)
    one_plus_rp = 1.0 + ratio_pow
    mfp = f1 * one_plus_rp

    kap = _THOMSON_OPACITY_COEF * jnp.exp(lnfe) / mfp
    dlrp_dln_t = alpha
    dlrp_deta = dalpha_deta * ln_ratio - alpha * (dtbr_deta / tbr)
    w_rp = ratio_pow / one_plus_rp
    dln_mfp_deta = (df1_deta / f1) + w_rp * dlrp_deta
    dln_mfp_dln_t = w_rp * dlrp_dln_t + dln_mfp_deta * deta_dt
    dln_mfp_dln_rho = dln_mfp_deta * deta_dr

    return kap, dlnfe_dr - dln_mfp_dln_rho, dlnfe_dt - dln_mfp_dln_t


@jax.jit
def combine_rad_with_conduction_jax(
    kap_rad: jax.Array | float,
    dlnkap_rad_dln_rho: jax.Array | float,
    dlnkap_rad_dln_t: jax.Array | float,
    kap_cond: jax.Array | float,
    dlnkap_cond_dln_rho: jax.Array | float,
    dlnkap_cond_dln_t: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Harmonic combination of radiative and conductive opacities in JAX."""
    kr = jnp.asarray(kap_rad, dtype=jnp.float64)
    dr_r = jnp.asarray(dlnkap_rad_dln_rho, dtype=jnp.float64)
    dr_t = jnp.asarray(dlnkap_rad_dln_t, dtype=jnp.float64)
    kc = jnp.asarray(kap_cond, dtype=jnp.float64)
    dc_r = jnp.asarray(dlnkap_cond_dln_rho, dtype=jnp.float64)
    dc_t = jnp.asarray(dlnkap_cond_dln_t, dtype=jnp.float64)

    kap = 1.0 / (1.0 / kr + 1.0 / kc)
    w_rad = kap / kr
    w_cond = kap / kc
    return kap, w_rad * dr_r + w_cond * dc_r, w_rad * dr_t + w_cond * dc_t


def electron_conduction_potekhin_jax(
    zbar: jax.Array | float,
    log_rho_in: jax.Array | float,
    log_t_in: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate Potekhin (`condtabl.data`) electron conduction opacity."""
    tbl = tables_jax.get_potekhin_jax()
    lrs, lts, lzs = tbl["log_rhos"], tbl["log_ts"], tbl["log_zs"]
    coeffs = tbl["coeffs"]

    zb = jnp.asarray(zbar, dtype=jnp.float64)
    lr_in = jnp.asarray(log_rho_in, dtype=jnp.float64)
    lt_in = jnp.asarray(log_t_in, dtype=jnp.float64)

    clipped_r = (lr_in < lrs[0]) | (lr_in > lrs[70])
    lr = jnp.clip(lr_in, lrs[0], lrs[70])
    clipped_t = (lt_in < lts[0]) | (lt_in > lts[28])
    lt = jnp.clip(lt_in, lts[0], lts[28])

    zlog = jnp.clip(jnp.log10(jnp.maximum(zb, 1.0e-30)), lzs[0], lzs[14])
    ir = jnp.clip(jnp.searchsorted(lrs, lr, side="right") - 1, 0, 69)
    jt = jnp.clip(jnp.searchsorted(lts, lt, side="right") - 1, 0, 27)
    i1 = jnp.clip(jnp.searchsorted(lzs, zlog, side="right") - 1, 0, 13)
    i2 = i1 + 1

    lk1, dr1, dt1 = jax_interp.do_kap_interpolations_jax(
        coeffs[i1], ir, jt, lrs[ir], lr, lrs[ir + 1], lts[jt], lt, lts[jt + 1]
    )
    lk2, dr2, dt2 = jax_interp.do_kap_interpolations_jax(
        coeffs[i2], ir, jt, lrs[ir], lr, lrs[ir + 1], lts[jt], lt, lts[jt + 1]
    )
    alfa = jnp.where(
        zlog <= lzs[0],
        0.0,
        jnp.where(
            zlog >= lzs[14], 1.0, (zlog - lzs[i1]) / (lzs[i2] - lzs[i1])
        ),
    )
    beta = 1.0 - alfa
    log_k = alfa * lk2 + beta * lk1
    dlk_dr = jnp.where(clipped_r, 0.0, alfa * dr2 + beta * dr1)
    dlk_dt = jnp.where(clipped_t, 0.0, alfa * dt2 + beta * dt1)

    logkap = 3.0 * lt_in - lr_in - log_k + _LOG10_COND_COEF
    kap = jnp.power(10.0, logkap)
    return kap, -1.0 - dlk_dr, 3.0 - dlk_dt


def _blouin_term_jax(
    lr: jax.Array,
    lt: jax.Array,
    lr0: float,
    lt0: float,
    alpha: float,
    sig_t: float,
    sig_r: float,
    a: float,
    b: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate one Blouin et al. (2020) Gaussian-tanh correction term."""
    rs = lr - lr0
    ts = lt - lt0
    sin_a, cos_a = jnp.sin(alpha), jnp.cos(alpha)
    u1 = (ts * cos_a + rs * sin_a) / sig_t
    u2 = (ts * sin_a - rs * cos_a) / sig_r
    g = jnp.exp(-(u1 * u1) - (u2 * u2)) * a
    dg_du1 = -2.0 * u1 * g
    dg_du2 = -2.0 * u2 * g
    dg_dlr = dg_du1 * (sin_a / sig_t) - dg_du2 * (cos_a / sig_r)
    dg_dlt = dg_du1 * (cos_a / sig_t) + dg_du2 * (sin_a / sig_r)
    th = jnp.tanh((g - 0.5) * b)
    h = th * 0.5 + 0.5
    d_gh_dg = h + g * (0.5 * b * (1.0 - th * th))
    denom = g * h + 1.0
    val = -jnp.log(denom)
    dval_dg = -d_gh_dg / denom
    return val, dval_dg * dg_dlr * _ILN10, dval_dg * dg_dlt * _ILN10


def electron_conduction_jax(
    zbar: jax.Array | float,
    log_rho: jax.Array | float,
    log_t: jax.Array | float,
    *,
    use_blouin: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate Potekhin / Blouin electron thermal conduction opacity in JAX."""
    zb = jnp.asarray(zbar, dtype=jnp.float64)
    lr = jnp.asarray(log_rho, dtype=jnp.float64)
    lt = jnp.asarray(log_t, dtype=jnp.float64)
    kap, dr, dt = electron_conduction_potekhin_jax(zb, lr, lt)
    if not use_blouin:
        return kap, dr, dt

    log2 = jnp.log10(2.0)
    log3 = jnp.log10(3.0)
    lzb = jnp.log10(jnp.maximum(zb, 1.0))
    alfa_12 = lzb / log2
    alfa_23 = (lzb - log2) / (log3 - log2)
    frac_h = jnp.where(
        zb <= 1.0, 1.0, jnp.where(zb <= 2.0, 1.0 - alfa_12, 0.0)
    )
    frac_he = jnp.where(
        zb <= 1.0,
        0.0,
        jnp.where(
            zb <= 2.0, alfa_12, jnp.where(zb <= 3.0, 1.0 - alfa_23, 0.0)
        ),
    )

    ch, dch_r, dch_t = _blouin_term_jax(
        lr, lt, 5.45, 8.40, -0.52, 0.45, 5.14, 2.0, 10.0
    )
    che, dche_r, dche_t = _blouin_term_jax(
        lr, lt, 6.50, 8.57, -0.46, 0.55, 6.20, 1.25, 2.5
    )
    c_val = jnp.where(
        zb > 3.0,
        0.0,
        jnp.where(frac_h > 0.0, ch * frac_h, 0.0)
        + jnp.where(frac_he > 0.0, che * frac_he, 0.0),
    )
    dc_r = jnp.where(
        zb > 3.0,
        0.0,
        jnp.where(frac_h > 0.0, dch_r * frac_h, 0.0)
        + jnp.where(frac_he > 0.0, dche_r * frac_he, 0.0),
    )
    dc_t = jnp.where(
        zb > 3.0,
        0.0,
        jnp.where(frac_h > 0.0, dch_t * frac_h, 0.0)
        + jnp.where(frac_he > 0.0, dche_t * frac_he, 0.0),
    )
    kap_out = jnp.where(zb > 3.0, kap, jnp.exp(jnp.log(kap) + c_val))
    return kap_out, dr + dc_r, dt + dc_t


def _eval_high_t_jax(
    z: jax.Array,
    x: jax.Array,
    zbase: jax.Array,
    xc: jax.Array,
    xo: jax.Array,
    log_rho: jax.Array,
    log_t: jax.Array,
    *,
    kap_option: str,
    kap_co_option: str,
    cubic_x: bool,
    cubic_z: bool,
    use_zbase_for_type1: bool,
    use_type2_opacities: bool,
    kap_type2_full_off_x: float,
    kap_type2_full_on_x: float,
    kap_type2_full_off_dz: float,
    kap_type2_full_on_dz: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate `Get_kap_highT_Results` in JAX."""
    if not use_type2_opacities:
        lk, dr, dt = jax_interp.eval_fixed_metal_jax(
            z,
            x,
            log_rho,
            log_t,
            prefix=kap_option,
            cubic_x=cubic_x,
            cubic_z=cubic_z,
        )
        return jnp.asarray(0.0, dtype=jnp.float64), jnp.power(10.0, lk), dr, dt

    z_t1 = jnp.minimum(z, zbase) if use_zbase_for_type1 else z
    fc, _, fo, _ = _REF_ZFRACS.get(kap_co_option, _REF_ZFRACS["gn93_co"])
    dxc = jnp.maximum(0.0, xc - fc * zbase)
    dxo = jnp.maximum(0.0, xo - fo * zbase)

    fx_mid = (kap_type2_full_off_x - x) / (
        kap_type2_full_off_x - kap_type2_full_on_x
    )
    frac_x = jnp.where(
        x >= kap_type2_full_off_x,
        0.0,
        jnp.where(x <= kap_type2_full_on_x, 1.0, fx_mid),
    )
    dz = z - zbase
    a_mid = (
        frac_x
        * (dz - kap_type2_full_off_dz)
        / (kap_type2_full_on_dz - kap_type2_full_off_dz)
    )
    alfa = jnp.where(
        dz <= kap_type2_full_off_dz,
        0.0,
        jnp.where(dz >= kap_type2_full_on_dz, frac_x, a_mid),
    )
    beta = 1.0 - alfa

    lk_co, dr_co, dt_co = jax_co_lowt.eval_co_kap_jax(
        zbase,
        x,
        dxc,
        dxo,
        log_rho,
        log_t,
        kap_co_option=kap_co_option,
        cubic_x=cubic_x,
        cubic_z=cubic_z,
    )
    kap_a = jnp.power(10.0, lk_co)
    lk_t1, dr_t1, dt_t1 = jax_interp.eval_fixed_metal_jax(
        z_t1,
        x,
        log_rho,
        log_t,
        prefix=kap_option,
        cubic_x=cubic_x,
        cubic_z=cubic_z,
    )
    kap_b = jnp.power(10.0, lk_t1)

    kap = jnp.where(
        beta == 0.0,
        kap_a,
        jnp.where(alfa == 0.0, kap_b, alfa * kap_a + beta * kap_b),
    )
    dr = jnp.where(
        beta == 0.0,
        dr_co,
        jnp.where(
            alfa == 0.0,
            dr_t1,
            (alfa * kap_a * dr_co + beta * kap_b * dr_t1) / kap,
        ),
    )
    dt = jnp.where(
        beta == 0.0,
        dt_co,
        jnp.where(
            alfa == 0.0,
            dt_t1,
            (alfa * kap_a * dt_co + beta * kap_b * dt_t1) / kap,
        ),
    )
    return alfa, kap, dr, dt


def kap_get_results_jax(
    z: jax.Array | float,
    x: jax.Array | float,
    zbase: jax.Array | float,
    xc: jax.Array | float = 0.0,
    xn: jax.Array | float = 0.0,
    xo: jax.Array | float = 0.0,
    xne: jax.Array | float = 0.0,
    log_rho: jax.Array | float = 0.0,
    log_t: jax.Array | float = 6.0,
    lnfree_e: jax.Array | float = 0.0,
    d_lnfree_e_dln_rho: jax.Array | float = 0.0,
    d_lnfree_e_dln_t: jax.Array | float = 0.0,
    eta: jax.Array | float = 0.0,
    d_eta_dln_rho: jax.Array | float = 0.0,
    d_eta_dln_t: jax.Array | float = 0.0,
    zbar: jax.Array | float = 1.0,
    *,
    kap_option: str = "gs98",
    kap_co_option: str = "gs98_co",
    kap_low_t_option: str = "lowT_fa05_gs98",
    kap_blend_log_t_upper_bdy: float = 3.88,
    kap_blend_log_t_lower_bdy: float = 3.80,
    cubic_interpolation_in_x: bool = False,
    cubic_interpolation_in_z: bool = False,
    include_electron_conduction: bool = True,
    use_blouin_conductive_opacities: bool = True,
    use_zbase_for_type1: bool = False,
    use_type2_opacities: bool = False,
    kap_type2_full_off_x: float = 1.0e-3,
    kap_type2_full_on_x: float = 1.0e-6,
    kap_type2_full_off_dz: float = 0.001,
    kap_type2_full_on_dz: float = 0.01,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate `Get_kap_Results` in native JAX."""
    _ = xne
    z_a = jnp.asarray(z, dtype=jnp.float64)
    x_a = jnp.asarray(x, dtype=jnp.float64)
    zb_a = jnp.asarray(zbase, dtype=jnp.float64)
    xc_a = jnp.asarray(xc, dtype=jnp.float64)
    xn_a = jnp.asarray(xn, dtype=jnp.float64)
    xo_a = jnp.asarray(xo, dtype=jnp.float64)
    lr_in = jnp.asarray(log_rho, dtype=jnp.float64)
    lt = jnp.asarray(log_t, dtype=jnp.float64)
    zb_ion = jnp.asarray(zbar, dtype=jnp.float64)

    clipped_rho = lr_in < -40.0
    lr = jnp.maximum(lr_in, -40.0)
    low_t_max = 4.05 if kap_low_t_option == "kapCN" else 4.50
    lo_bdy = max(float(kap_blend_log_t_lower_bdy), 3.75)
    hi_bdy = min(float(kap_blend_log_t_upper_bdy), low_t_max)

    def _eval_low_t() -> tuple[jax.Array, jax.Array, jax.Array]:
        if kap_low_t_option == "kapCN":
            fc_cn = jnp.where(zb_a <= 0.0, 1.0, xc_a / (0.1644 * zb_a))
            fn_cn = jnp.where(zb_a <= 0.0, 1.0, xn_a / (0.0532 * zb_a))
            return jax_co_lowt.eval_kapcn_jax(zb_a, x_a, fc_cn, fn_cn, lr, lt)
        if kap_low_t_option == "AESOPUS":
            return jax_co_lowt.eval_aesopus_jax(
                zb_a, x_a, xc_a, xn_a, xo_a, lr, lt
            )
        lk_l, dr_l, dt_l = jax_interp.eval_fixed_metal_jax(
            z_a,
            x_a,
            lr,
            lt,
            prefix=kap_low_t_option,
            cubic_x=cubic_interpolation_in_x,
            cubic_z=cubic_interpolation_in_z,
        )
        return jnp.power(10.0, lk_l), dr_l, dt_l

    def _eval_high_t() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return _eval_high_t_jax(
            z_a, x_a, zb_a, xc_a, xo_a, lr, lt,
            kap_option=kap_option, kap_co_option=kap_co_option,
            cubic_x=cubic_interpolation_in_x, cubic_z=cubic_interpolation_in_z,
            use_zbase_for_type1=use_zbase_for_type1,
            use_type2_opacities=use_type2_opacities,
            kap_type2_full_off_x=kap_type2_full_off_x,
            kap_type2_full_on_x=kap_type2_full_on_x,
            kap_type2_full_off_dz=kap_type2_full_off_dz,
            kap_type2_full_on_dz=kap_type2_full_on_dz,
        )

    z0 = jnp.asarray(0.0, dtype=jnp.float64)
    z1 = jnp.asarray(1.0, dtype=jnp.float64)
    lt_f = float(lt) if not isinstance(lt, jax.core.Tracer) else None
    if lt_f is not None and lt_f <= lo_bdy:
        kr, dkr_dr, dkr_dt = _eval_low_t()
        f_low, f_high, f_t2 = z1, z0, z0
    elif lt_f is not None and lt_f >= hi_bdy:
        f_t2, kr, dkr_dr, dkr_dt = _eval_high_t()
        f_low, f_high = z0, z1
    else:
        f_t2_h, k1, dr1, dt1 = _eval_high_t()
        k2, dr2, dt2 = _eval_low_t()
        u_t = jnp.clip((lt - lo_bdy) / (hi_bdy - lo_bdy), 0.0, 1.0)
        alfa_v, ds_du = _quintic_val_and_deriv(u_t)
        dalfa_dt = jnp.where(
            (lt <= lo_bdy) | (lt >= hi_bdy), 0.0, ds_du / (hi_bdy - lo_bdy)
        )
        alfa = jnp.where(
            lt <= lo_bdy, 0.0, jnp.where(lt >= hi_bdy, 1.0, alfa_v)
        )
        beta = 1.0 - alfa
        lk1 = jnp.log10(jnp.maximum(k1, 1.0e-99))
        lk2 = jnp.log10(jnp.maximum(k2, 1.0e-99))
        kr = jnp.where(
            lt <= lo_bdy,
            k2,
            jnp.where(
                lt >= hi_bdy, k1, jnp.power(10.0, alfa * lk1 + beta * lk2)
            ),
        )
        dkr_dr = alfa * dr1 + beta * dr2
        dkr_dt = alfa * dt1 + beta * dt2 + dalfa_dt * (lk1 - lk2)
        f_low, f_high = beta, alfa
        f_t2 = jnp.where(lt <= lo_bdy, 0.0, f_t2_h)

    dkr_dr = jnp.where(clipped_rho, 0.0, dkr_dr)
    log_r = lr_in - 3.0 * lt + 18.0
    lt_c_lo, lt_c_hi, lr_c_lo, lr_c_hi = 8.19, 8.69, -7.99, -7.49
    bt_v, dbt_du = _quintic_val_and_deriv(
        jnp.clip((lt - lt_c_lo) / 0.5, 0.0, 1.0)
    )
    bt = jnp.where(lt <= lt_c_lo, 0.0, jnp.where(lt > lt_c_hi, 1.0, bt_v))
    dbt_dt = jnp.where((lt <= lt_c_lo) | (lt > lt_c_hi), 0.0, dbt_du / 0.5)
    br_v, dbr_du = _quintic_val_and_deriv(
        jnp.clip((lr_c_hi - log_r) / 0.5, 0.0, 1.0)
    )
    br = jnp.where(log_r <= lr_c_lo, 1.0, jnp.where(log_r > lr_c_hi, 0.0, br_v))
    dbr_dlr = jnp.where(
        (log_r <= lr_c_lo) | (log_r > lr_c_hi), 0.0, -dbr_du / 0.5
    )
    blend = 1.0 - (1.0 - bt) * (1.0 - br)
    dblend_drho = (1.0 - bt) * dbr_dlr
    dblend_dt = (1.0 - br) * dbt_dt - 3.0 * (1.0 - bt) * dbr_dlr

    kc, dkc_dr, dkc_dt = compton_opacity_jax(
        jnp.power(10.0, lr_in), jnp.power(10.0, lt),
        lnfree_e, d_lnfree_e_dln_rho, d_lnfree_e_dln_t,
        eta, d_eta_dln_rho, d_eta_dln_t,
    )
    lk_c = jnp.where(blend > 0.0, jnp.log10(jnp.maximum(kc, 1.0e-99)), 0.0)
    lk_tbl = jnp.where(blend < 1.0, jnp.log10(jnp.maximum(kr, 1.0e-99)), 0.0)
    kap_rad = jnp.power(10.0, blend * lk_c + (1.0 - blend) * lk_tbl)
    dlk_rad_dr = (
        blend * jnp.where(blend > 0.0, dkc_dr, 0.0)
        + (1.0 - blend) * jnp.where(blend < 1.0, dkr_dr, 0.0)
        + dblend_drho * (lk_c - lk_tbl)
    )
    dlk_rad_dt = (
        blend * jnp.where(blend > 0.0, dkc_dt, 0.0)
        + (1.0 - blend) * jnp.where(blend < 1.0, dkr_dt, 0.0)
        + dblend_dt * (lk_c - lk_tbl)
    )
    fracs = jnp.stack([
        jnp.where(blend < 1.0, (1.0 - blend) * f_low, 0.0),
        jnp.where(blend < 1.0, (1.0 - blend) * f_high, 0.0),
        jnp.where(blend < 1.0, f_t2, 0.0),
        blend,
    ])
    if not include_electron_conduction:
        return kap_rad, dlk_rad_dr, dlk_rad_dt, fracs
    k_ec, dec_dr, dec_dt = electron_conduction_jax(
        zb_ion, lr_in, lt, use_blouin=use_blouin_conductive_opacities
    )
    kap_tot, dtot_dr, dtot_dt = combine_rad_with_conduction_jax(
        kap_rad, dlk_rad_dr, dlk_rad_dt, k_ec, dec_dr, dec_dt
    )
    return kap_tot, dtot_dr, dtot_dt, fracs


def make_differentiable_kap(
    z: float,
    x: float,
    zbase: float,
    xc: float = 0.0,
    xn: float = 0.0,
    xo: float = 0.0,
    xne: float = 0.0,
    lnfree_e: float = 0.0,
    d_lnfree_e_dln_rho: float = 0.0,
    d_lnfree_e_dln_t: float = 0.0,
    eta: float = 0.0,
    d_eta_dln_rho: float = 0.0,
    d_eta_dln_t: float = 0.0,
    zbar: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Create a pure `@jax.jit` / `jax.custom_jvp` callable."""
    eval_fn = jax.jit(
        functools.partial(
            kap_get_results_jax,
            z, x, zbase, xc, xn, xo, xne,
            lnfree_e=lnfree_e,
            d_lnfree_e_dln_rho=d_lnfree_e_dln_rho,
            d_lnfree_e_dln_t=d_lnfree_e_dln_t,
            eta=eta, d_eta_dln_rho=d_eta_dln_rho, d_eta_dln_t=d_eta_dln_t,
            zbar=zbar, **kwargs,
        )
    )

    @jax.custom_jvp
    def _fn(log_rho: jax.Array, log_t: jax.Array) -> jax.Array:
        kap, _, _, _ = eval_fn(log_rho=log_rho, log_t=log_t)
        return jnp.log10(kap)

    @_fn.defjvp
    def _fn_jvp(
        primals: tuple[jax.Array, jax.Array],
        tangents: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        lr, lt = primals
        dlr, dlt = tangents
        kap, dln_dr, dln_dt, _ = eval_fn(log_rho=lr, log_t=lt)
        return jnp.log10(kap), dln_dr * dlr + dln_dt * dlt

    return _fn
