# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native JAX 7-level `eosDT_get` cascade and component blend fractions.

Ports `get_HELM_alfa`, `Get_FreeEOS_alfa`, `get_opal_scvh_alfa_and_partials`,
`Get_CMS_alfa`, and `Get_eosDT_Results` (`CMS -> Skye -> PC -> FreeEOS ->
OPAL/SCVH -> HELM -> ideal`) to pure JAX (`jnp.float64`).
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import chem
from flexmesa.eos import (
    _cache_io,
    _pc_jax,
    _tables_4d_jax,
    core,
    helm,
    ideal,
    skye,
)

_HELM_BOUNDS_NP = np.array(
    [
        [_cache_io.HELM_LOGDLO, _cache_io.HELM_LOGTHI],
        [_cache_io.HELM_LOGDLO, _cache_io.HELM_LOGTLO],
        [_cache_io.HELM_LOGDHI, _cache_io.HELM_LOGTLO],
        [_cache_io.HELM_LOGDHI, _cache_io.HELM_LOGTHI],
    ],
    dtype=np.float64,
)


def _trapezoid_jax(
    x: jax.Array, lo_0: float, lo_1: float, hi_1: float, hi_0: float
) -> jax.Array:
    xv = jax.lax.stop_gradient(x)
    return jnp.where(
        xv < lo_0,
        0.0,
        jnp.where(
            xv < lo_1,
            (x - lo_0) / (lo_1 - lo_0),
            jnp.where(
                xv < hi_1,
                1.0,
                jnp.where(xv < hi_0, (x - hi_0) / (hi_1 - hi_0), 0.0),
            ),
        ),
    )


def get_helm_alfa_jax(
    log_rho: jax.Array, log_t: jax.Array, helm_blend_width: float = 0.1
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute `HELM` polygon fallback fraction `(alfa, d_dlogT, d_dlogRho)`."""
    lr_f = float(jax.lax.stop_gradient(log_rho))
    lt_f = float(jax.lax.stop_gradient(log_t))
    contained = core.is_contained(_HELM_BOUNDS_NP, lr_f, lt_f)
    sign = -1.0 if contained else 1.0
    bounds_j = jnp.asarray(_HELM_BOUNDS_NP, dtype=jnp.float64)

    def _alfa_fn(lr: jax.Array, lt: jax.Array) -> jax.Array:
        dist = sign * core.min_distance_to_polygon_jax(
            bounds_j, jnp.stack([lr, lt])
        )
        scaled = dist / helm_blend_width
        sv = jax.lax.stop_gradient(scaled)
        return jnp.where(sv < 0.0, 0.0, jnp.where(sv > 1.0, 1.0, scaled))

    val, (dt, dr) = jax.value_and_grad(_alfa_fn, argnums=(1, 0))(log_rho, log_t)
    return val, dt, dr


@jax.jit
def get_free_eos_alfa_jax(
    log_rho: jax.Array, log_t: jax.Array, z: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute raw `FreeEOS` fallback fraction `(alfa, d_dlogT, d_dlogRho)`."""

    def _alfa_fn(lr: jax.Array, lt: jax.Array) -> jax.Array:
        lq = lr - 2.0 * lt + 12.0
        b_lt = _trapezoid_jax(lt, 3.0, 3.1, 8.1, 8.2)
        b_lr = _trapezoid_jax(lr, -18.1, -18.0, 10.0, 10.1)
        b_lq = _trapezoid_jax(lq, -10.0, -9.9, 5.5, 5.6)

        lrv = jax.lax.stop_gradient(lr)
        ltv = jax.lax.stop_gradient(lt)
        lqv = jax.lax.stop_gradient(lq)
        b_lr_cut = jnp.where(
            lrv < -6.5,
            1.0,
            jnp.where(lrv < -5.5, (lr - (-5.5)) / (-6.5 - (-5.5)), 0.0),
        )
        b_lt_cut = jnp.where(
            ltv < 6.4, 0.0, jnp.where(ltv < 6.5, (lt - 6.4) / (6.5 - 6.4), 1.0)
        )
        lq_hi = jnp.where(z <= 0.3, 0.5, -0.5)
        lq_lo = jnp.where(z <= 0.3, -0.5, -1.5)
        b_lq_cut = jnp.where(
            lqv < lq_lo,
            1.0,
            jnp.where(lqv < lq_hi, (lq - lq_hi) / (lq_lo - lq_hi), 0.0),
        )
        bltc_v = jax.lax.stop_gradient(b_lt_cut)
        blqc_v = jax.lax.stop_gradient(b_lq_cut)
        blrc_v = jax.lax.stop_gradient(b_lr_cut)
        max_rq = jnp.where(blrc_v >= blqc_v, b_lr_cut, b_lq_cut)
        max_tq = jnp.where(bltc_v >= blqc_v, b_lt_cut, b_lq_cut)
        b_cut = jnp.where(
            bltc_v == 0.0,
            jnp.where(
                blqc_v == 0.0, b_lr_cut, jnp.where(blqc_v == 1.0, 1.0, max_rq)
            ),
            jnp.where(
                bltc_v == 1.0,
                1.0,
                jnp.where(
                    blqc_v == 0.0,
                    b_lt_cut,
                    jnp.where(blqc_v == 1.0, 1.0, max_tq),
                ),
            ),
        )
        return 1.0 - b_cut * b_lt * b_lr * b_lq

    val, (dt, dr) = jax.value_and_grad(_alfa_fn, argnums=(1, 0))(log_rho, log_t)
    return val, dt, dr


@jax.jit
def get_cms_alfa_jax(
    log_rho: jax.Array, log_t: jax.Array, z: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute raw `CMS` fallback fraction `(alfa, d_dlogT, d_dlogRho)`."""

    def _alfa_fn(lr: jax.Array, lt: jax.Array) -> jax.Array:
        b_lt = _trapezoid_jax(lt, 2.0, 2.1, 7.9, 8.0)
        b_lr = _trapezoid_jax(lr, -8.0, -7.9, 5.9, 6.0)
        return 1.0 - b_lt * b_lr

    val, (dt, dr) = jax.value_and_grad(_alfa_fn, argnums=(1, 0))(log_rho, log_t)
    return val, dt, dr


def get_opal_scvh_alfa_jax(
    log_t: jax.Array, log_rho: jax.Array, z: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute `OPAL/SCVH` fallback fraction `(alfa, d_dlogRho, d_dlogT)`."""
    lt = float(jax.lax.stop_gradient(log_t))
    lr = float(jax.lax.stop_gradient(log_rho))
    log_q1, log_q2, log_q_max, log_q4 = 5.69, 5.68, 5.3, -8.0
    log_rho5 = -14.299
    log_rho6 = log_rho5 - 1.0e-3
    log_rho1, log_rho2 = 3.50, 3.48
    log_t8, log_t7, log_t6, log_t5, log_t2, log_t1 = (
        2.2,
        2.3,
        4.890,
        4.899,
        7.5,
        7.6,
    )
    z_all_helm, z_no_helm = 0.040, 0.035

    if lt >= log_t1:
        z0 = jnp.zeros_like(log_t)
        return jnp.ones_like(log_t), z0, z0

    log_t3 = 0.5 * (log_rho1 - log_q1 + 12.0)
    log_t4 = 0.5 * (log_rho2 - log_q2 + 12.0)
    log_rho3 = log_q1 + 2.0 * log_t7 - 12.0
    log_rho4 = log_q2 + 2.0 * log_t7 - 12.0
    lq = lr - 2.0 * lt + 12.0

    def _alfa_fn(lr_in: jax.Array, lt_in: jax.Array) -> jax.Array:
        if (
            z >= z_all_helm
            or lt >= log_t1
            or lt <= log_t8
            or lr >= log_rho1
            or lq <= log_q4
            or lq >= log_q_max
            or (lr <= log_rho6 and lt <= log_t6)
        ):
            a = jnp.ones_like(lr_in)
        elif lt >= log_t2:
            c_dy = (lt_in - log_t2) / (log_t1 - log_t2)
            if lr > log_rho2:
                c_dx = (lr_in - log_rho2) / (log_rho1 - log_rho2)
                r = jnp.sqrt(c_dx * c_dx + c_dy * c_dy)
                rv = float(jax.lax.stop_gradient(r))
                a = (
                    jnp.ones_like(r)
                    if rv >= 1.0
                    else (jnp.zeros_like(r) if rv < 1.0e-10 else r)
                )
            else:
                a = c_dy
        elif lt >= log_t3:
            a = (
                (lr_in - log_rho2) / (log_rho1 - log_rho2)
                if lr > log_rho2
                else jnp.zeros_like(lr_in)
            )
        elif lt >= log_t4:
            lr_hi = log_q1 + 2.0 * lt - 12.0
            if lr >= lr_hi:
                a = jnp.ones_like(lr_in)
            elif lr > log_rho2:
                a = (lr_in - log_rho2) / (lr_hi - log_rho2)
            else:
                a = jnp.zeros_like(lr_in)
        elif lr > log_rho4:
            if lt > log_t7:
                sa = ((log_q1 + 2.0 * log_t4 - 12.0) - log_rho3) / (
                    log_t4 - log_t7
                )
                sb = (log_rho2 - log_rho4) / (log_t4 - log_t7)
                lr_hi_v = log_rho3 + (lt - log_t7) * sa
                lr_lo_v = log_rho4 + (lt - log_t7) * sb
                if lr >= lr_hi_v:
                    a = jnp.ones_like(lr_in)
                elif lr >= lr_lo_v:
                    a = (lr_in - lr_lo_v) / (lr_hi_v - lr_lo_v)
                else:
                    a = jnp.zeros_like(lr_in)
            elif lr > log_rho3:
                a = jnp.ones_like(lr_in)
            else:
                c_dx = (lr_in - log_rho4) / (log_rho3 - log_rho4)
                c_dy = (lt_in - log_t7) / (log_t8 - log_t7)
                r = jnp.sqrt(c_dx * c_dx + c_dy * c_dy)
                rv = float(jax.lax.stop_gradient(r))
                a = (
                    jnp.ones_like(r)
                    if rv >= 1.0
                    else (jnp.zeros_like(r) if rv < 1.0e-10 else r)
                )
        elif lr >= log_rho5 or lt > log_t5:
            a = jnp.zeros_like(lr_in)
        elif lt >= log_t6:
            if lr <= log_rho6:
                a = (lt_in - log_t5) / (log_t6 - log_t5)
            else:
                c_dx = (lr_in - log_rho6) / (log_rho5 - log_rho6)
                c_dy = (lt_in - log_t6) / (log_t5 - log_t6)
                beta = jnp.sqrt(c_dx * c_dx + c_dy * c_dy)
                r = 1.0 - beta
                rv = float(jax.lax.stop_gradient(r))
                a = (
                    jnp.ones_like(r)
                    if rv >= 1.0
                    else (jnp.zeros_like(r) if rv < 1.0e-10 else r)
                )
        else:
            a = (lr_in - log_rho5) / (log_rho6 - log_rho5)

        av = float(jax.lax.stop_gradient(a))
        if z > z_no_helm and z < z_all_helm and av < 1.0:
            zf = (z - z_no_helm) / (z_all_helm - z_no_helm)
            a = zf + (1.0 - zf) * a
        return a

    val, (dr, dt) = jax.value_and_grad(_alfa_fn, argnums=(0, 1))(log_rho, log_t)
    return val, dr, dt


def _combine_jax(
    alfa: jax.Array,
    d_dlog_t: jax.Array,
    d_dlog_rho: jax.Array,
    eval_1st: object,
    eval_2nd: object,
) -> core.EosEvalOutput:
    av = float(jax.lax.stop_gradient(alfa))
    if av >= 1.0:
        out2_only: core.EosEvalOutput = eval_2nd()  # type: ignore[operator]
        return out2_only
    out1: core.EosEvalOutput = eval_1st()  # type: ignore[operator]
    lns1 = float(jax.lax.stop_gradient(out1.res[core.I_LN_S]))
    dlnd_s1 = float(jax.lax.stop_gradient(out1.d_dlnd[core.I_LN_S]))
    dlnt_s1 = float(jax.lax.stop_gradient(out1.d_dln_t[core.I_LN_S]))
    if not (
        np.isfinite(lns1) and np.isfinite(dlnd_s1) and np.isfinite(dlnt_s1)
    ):
        out2_fb: core.EosEvalOutput = eval_2nd()  # type: ignore[operator]
        return out2_fb
    if av <= 0.0:
        return out1
    out2: core.EosEvalOutput = eval_2nd()  # type: ignore[operator]
    return core.blend_eos_results(
        alfa, d_dlog_t, d_dlog_rho, False, out2, out1, backend="jax"
    )


def eos_dt_get_native_jax(
    rho: jax.Array,
    temp: jax.Array,
    chem_id: Sequence[int],
    xa: jax.Array,
    *,
    use_skye: bool = True,
    use_pc: bool = False,
    pc_use_gamma_limit: bool = True,
    skye_solid_mixing_rule: str = "PC",
    use_cms: bool = False,
    use_free_eos: bool = True,
    use_opal_scvh: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate the 7-level `eosDT_get` cascade in pure native JAX."""
    xa_np = np.asarray(jax.lax.stop_gradient(xa), dtype=np.float64)
    cids = [int(c) for c in chem_id]
    comp = chem.get_composition_info(cids, xa_np)
    x_val, z_val = float(comp.xh), float(comp.xz)
    abar = jnp.asarray(float(comp.abar), dtype=jnp.float64)
    zbar = jnp.asarray(float(comp.zbar), dtype=jnp.float64)

    infos = [chem.get_nuclide_info(c) for c in cids]
    azion = jnp.asarray([float(inf.z) for inf in infos], dtype=jnp.float64)
    acmi = jnp.asarray([float(inf.w) for inf in infos], dtype=jnp.float64)
    a_baryon = jnp.asarray([float(inf.a) for inf in infos], dtype=jnp.float64)

    log_rho = jnp.log10(rho)
    log_t = jnp.log10(temp)

    def _lvl6() -> core.EosEvalOutput:
        a, dt, dr = get_helm_alfa_jax(log_rho, log_t, 0.1)
        return _combine_jax(
            a,
            dt,
            dr,
            lambda: helm.eval_helm_eos(temp, rho, abar, zbar, backend="jax")[0],
            lambda: ideal.eval_ideal_eos_from_weights(
                temp, rho, abar, zbar, xa, acmi, a_baryon, backend="jax"
            ),
        )

    def _lvl5() -> core.EosEvalOutput:
        if use_opal_scvh:
            a, dr, dt = get_opal_scvh_alfa_jax(log_t, log_rho, z_val)
        else:
            a, dt, dr = (
                jnp.ones_like(log_t),
                jnp.zeros_like(log_t),
                jnp.zeros_like(log_t),
            )
        return _combine_jax(
            a,
            dt,
            dr,
            lambda: core.EosEvalOutput(
                *_tables_4d_jax.eval_eosdt_table_jax(0, rho, temp, x_val, z_val)
            ),
            _lvl6,
        )

    def _lvl4() -> core.EosEvalOutput:
        if use_free_eos:
            a, dt, dr = get_free_eos_alfa_jax(log_rho, log_t, z_val)
        else:
            a, dt, dr = (
                jnp.ones_like(log_t),
                jnp.zeros_like(log_t),
                jnp.zeros_like(log_t),
            )
        return _combine_jax(
            a,
            dt,
            dr,
            lambda: core.EosEvalOutput(
                *_tables_4d_jax.eval_eosdt_table_jax(1, rho, temp, x_val, z_val)
            ),
            _lvl5,
        )

    def _lvl3() -> core.EosEvalOutput:
        if use_pc:
            a, dt, dr = _pc_jax.get_pc_alfa_jax(
                log_rho, log_t, abar, zbar, use_gamma_limit=pc_use_gamma_limit
            )
        else:
            a, dt, dr = (
                jnp.ones_like(log_t),
                jnp.zeros_like(log_t),
                jnp.zeros_like(log_t),
            )
        return _combine_jax(
            a,
            dt,
            dr,
            lambda: core.EosEvalOutput(
                *_pc_jax.eval_pc_eos_jax(rho, temp, abar, zbar, xa, azion, acmi)
            ),
            _lvl4,
        )

    def _lvl2() -> core.EosEvalOutput:
        if use_skye:
            a, dt, dr = skye.get_skye_alfa_jax(log_rho, log_t, abar, zbar)
        else:
            a, dt, dr = (
                jnp.ones_like(log_t),
                jnp.zeros_like(log_t),
                jnp.zeros_like(log_t),
            )
        return _combine_jax(
            a,
            dt,
            dr,
            lambda: core.EosEvalOutput(
                *skye.eval_skye_eos_jax(
                    rho,
                    temp,
                    abar,
                    zbar,
                    chem_id,
                    xa,
                    skye_solid_mixing_rule=skye_solid_mixing_rule,
                )
            ),
            _lvl3,
        )

    if use_cms:
        a_cms, dt_cms, dr_cms = get_cms_alfa_jax(log_rho, log_t, z_val)
    else:
        a_cms, dt_cms, dr_cms = (
            jnp.ones_like(log_t),
            jnp.zeros_like(log_t),
            jnp.zeros_like(log_t),
        )
    out = _combine_jax(
        a_cms,
        dt_cms,
        dr_cms,
        lambda: core.EosEvalOutput(
            *_tables_4d_jax.eval_cms_eos_jax(rho, temp, x_val)
        ),
        _lvl2,
    )
    return (
        jnp.asarray(out.res),
        jnp.asarray(out.d_dlnd),
        jnp.asarray(out.d_dln_t),
    )
