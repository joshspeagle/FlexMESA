# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branchless JAX kernels and `custom_jvp` rules for the `PC` equation of state.

Provides `get_pc_alfa_jax` and `eval_pc_eos_jax` with analytical
`jax.custom_jvp` logarithmic derivative rules so `jax.jit`, `jax.vmap`,
`jax.jvp`, and `jax.grad` differentiate smoothly through the 26-channel `PC`
EOS.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _rust, constants
from flexmesa.eos import _pc_coulomb_jax, core

excor7_jax = _pc_coulomb_jax.excor7_jax
fharm12_anharm8_jax = _pc_coulomb_jax.fharm12_anharm8_jax
fition9_jax = _pc_coulomb_jax.fition9_jax


def get_pc_alfa_jax(
    log_rho: jax.Array,
    log_t: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    use_gamma_limit: bool = True,
    log_rho1: float = 2.999,
    log_rho2: float = 2.8,
    log_t1: float = 7.6,
    log_t2: float = 7.7,
    log_gamma_e_all_helm: float = -1.8,
    log_gamma_e_all_pc: float = -1.7,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate `Get_PC_alfa` (`eospc_eval.f90` lines 31-96) in JAX.

    Args:
        log_rho: Base-10 log mass density $\\log_{10} \\rho$ in `g cm^-3`.
        log_t: Base-10 log temperature $\\log_{10} T$ in `K`.
        abar: Mean atomic weight $\\bar{A}$.
        zbar: Mean atomic charge $\\bar{Z}$.
        use_gamma_limit: Whether to use $\\log_{10} \\Gamma_e$ thresholds.
        log_rho1: Upper $\\log_{10} \\rho$ threshold for pure `PC`.
        log_rho2: Lower $\\log_{10} \\rho$ threshold below which `PC` is zero.
        log_t1: Lower $\\log_{10} T$ threshold when `T` limits are used.
        log_t2: Upper $\\log_{10} T$ threshold when `T` limits are used.
        log_gamma_e_all_helm: $\\log_{10} \\Gamma_e$ threshold for pure `HELM`.
        log_gamma_e_all_pc: $\\log_{10} \\Gamma_e$ threshold for pure `PC`.

    Returns:
        Tuple `(alfa, d_alfa_dlogT, d_alfa_dlogRho)` where `alfa = 0` is pure
        `PC` and `alfa = 1` is zero `PC`.
    """
    log_rho_lo0 = jnp.asarray(log_rho2, dtype=jnp.float64)
    log_rho_hi = jnp.asarray(log_rho1, dtype=jnp.float64)

    log_ge0 = jnp.log10(
        constants.QE
        * constants.QE
        * jnp.power(
            constants.FOUR_THIRDS_PI * constants.AVO * zbar / abar,
            constants.ONE_THIRD,
        )
        / constants.KERG
    )
    log_ge = log_ge0 + log_rho / 3.0 - log_t
    log_ge_lo = jnp.asarray(log_gamma_e_all_helm, dtype=jnp.float64)
    log_ge_hi = jnp.asarray(log_gamma_e_all_pc, dtype=jnp.float64)
    log_t_lo_g = log_ge0 + log_rho_lo0 / 3.0 - log_ge_lo
    log_t_hi_g = log_ge0 + log_rho_hi / 3.0 - log_ge_hi

    span_ge = log_ge_hi - log_ge_lo
    alfa_ge = (log_ge_hi - log_ge) / span_ge
    dt_ge = jnp.ones_like(log_rho) / span_ge
    dr_ge = -(1.0 / 3.0) * jnp.ones_like(log_rho) / span_ge

    log_rho_lo_eff = jnp.where(
        log_t >= log_t_lo_g,
        (log_t_lo_g + log_ge_lo - log_ge0) * 3.0,
        log_rho_lo0,
    )
    span_r = log_rho_hi - log_rho_lo_eff
    alfa_r = (log_rho_hi - log_rho) / span_r
    dr_r = -1.0 / span_r

    out_outside_g = (log_rho <= log_rho_lo0) | (log_ge <= log_ge_lo)
    out_inside_g = (log_rho >= log_rho_hi) & (log_ge >= log_ge_hi)
    use_ge_branch = log_t >= log_t_hi_g

    alfa_g = jnp.where(
        out_outside_g,
        1.0,
        jnp.where(out_inside_g, 0.0, jnp.where(use_ge_branch, alfa_ge, alfa_r)),
    )
    dt_g = jnp.where(
        out_outside_g | out_inside_g,
        0.0,
        jnp.where(use_ge_branch, dt_ge, 0.0),
    )
    dr_g = jnp.where(
        out_outside_g | out_inside_g,
        0.0,
        jnp.where(use_ge_branch, dr_ge, dr_r),
    )

    span_t = log_t2 - log_t1
    alfa_t_br = (log_t - log_t1) / span_t
    dt_t_br = jnp.ones_like(log_rho) / span_t
    span_r0 = log_rho_hi - log_rho_lo0
    alfa_r0 = (log_rho_hi - log_rho) / span_r0
    dr_r0 = -jnp.ones_like(log_rho) / span_r0

    out_outside_t = (log_rho <= log_rho_lo0) | (log_t >= log_t2)
    out_inside_t = (log_rho >= log_rho_hi) & (log_t <= log_t1)
    use_t_branch = log_t >= log_t1

    alfa_t = jnp.where(
        out_outside_t,
        1.0,
        jnp.where(
            out_inside_t, 0.0, jnp.where(use_t_branch, alfa_t_br, alfa_r0)
        ),
    )
    dt_t = jnp.where(
        out_outside_t | out_inside_t,
        0.0,
        jnp.where(use_t_branch, dt_t_br, 0.0),
    )
    dr_t = jnp.where(
        out_outside_t | out_inside_t,
        0.0,
        jnp.where(use_t_branch, 0.0, dr_r0),
    )

    use_g = jnp.asarray(use_gamma_limit)
    return (
        jnp.where(use_g, alfa_g, alfa_t),
        jnp.where(use_g, dt_g, dt_t),
        jnp.where(use_g, dr_g, dr_t),
    )


def eval_pc_eos_jax(
    rho: jax.Array,
    temp: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xa: jax.Array,
    azion: jax.Array,
    acmi: jax.Array,
    *,
    include_radiation: bool = True,
    pc_gamma_start_crystal: float = 170.0,
    pc_gamma_full_crystal: float = 180.0,
    mass_fraction_limit: float = 1.0e-4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate 26-channel `PC` EOS in JAX with analytical `custom_jvp`."""
    inc_rad = bool(include_radiation)
    g_start = float(pc_gamma_start_crystal)
    g_full = float(pc_gamma_full_crystal)
    mf_lim = float(mass_fraction_limit)

    def _host_eval_pc(
        rho_np: np.ndarray,
        temp_np: np.ndarray,
        abar_np: np.ndarray,
        zbar_np: np.ndarray,
        xa_np: np.ndarray,
        azion_np: np.ndarray,
        acmi_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        orig_shape = np.shape(rho_np)
        r_1d = np.ascontiguousarray(
            np.asarray(rho_np, dtype=np.float64).reshape(-1)
        )
        t_1d = np.ascontiguousarray(
            np.asarray(temp_np, dtype=np.float64).reshape(-1)
        )
        nz = r_1d.size
        a_1d = np.broadcast_to(
            np.asarray(abar_np, dtype=np.float64), orig_shape
        ).reshape(nz)
        z_1d = np.broadcast_to(
            np.asarray(zbar_np, dtype=np.float64), orig_shape
        ).reshape(nz)
        az_1d = np.ascontiguousarray(
            np.asarray(azion_np, dtype=np.float64).reshape(-1)
        )
        ac_1d = np.ascontiguousarray(
            np.asarray(acmi_np, dtype=np.float64).reshape(-1)
        )
        sp = az_1d.size
        xa_2d = np.broadcast_to(
            np.asarray(xa_np, dtype=np.float64), (*orig_shape, sp)
        ).reshape(-1)
        out = _rust.eos_eval_pc_zones(
            np.ascontiguousarray(r_1d),
            np.ascontiguousarray(t_1d),
            np.ascontiguousarray(a_1d),
            np.ascontiguousarray(z_1d),
            np.ascontiguousarray(xa_2d),
            az_1d,
            ac_1d,
            inc_rad,
            g_start,
            g_full,
            mf_lim,
            True,
        )
        n = core.NUM_EOS_BASIC_RESULTS
        return (
            out[:, :n].reshape((*orig_shape, n)),
            out[:, n : 2 * n].reshape((*orig_shape, n)),
            out[:, 2 * n : 3 * n].reshape((*orig_shape, n)),
        )

    def _eval_pc_core(
        r_in: jax.Array,
        t_in: jax.Array,
        a_in: jax.Array,
        z_in: jax.Array,
        x_in: jax.Array,
        az_in: jax.Array,
        ac_in: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        shape = (*r_in.shape, core.NUM_EOS_BASIC_RESULTS)
        vc = jax.ShapeDtypeStruct(shape, jnp.float64)
        res_v, dlnd_v, dlnt_v = jax.pure_callback(
            _host_eval_pc,
            (vc, vc, vc),
            r_in,
            t_in,
            a_in,
            z_in,
            x_in,
            az_in,
            ac_in,
            vmap_method="expand_dims",
        )
        return res_v, dlnd_v, dlnt_v

    _eval_pc = jax.custom_jvp(_eval_pc_core)

    def _eval_pc_jvp(
        primals: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        tangents: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array],
        tuple[jax.Array, jax.Array, jax.Array],
    ]:
        r_p, t_p, a_p, z_p, x_p, az_p, ac_p = primals
        dr, dt, _, _, _, _, _ = tangents
        res, d_dlnd, d_dln_t = _eval_pc_core(
            r_p, t_p, a_p, z_p, x_p, az_p, ac_p
        )
        dln_rho = jnp.expand_dims(dr / r_p, axis=-1)
        dln_t = jnp.expand_dims(dt / t_p, axis=-1)
        d_res = d_dlnd * dln_rho + d_dln_t * dln_t
        z = jnp.zeros_like(res)
        return (res, d_dlnd, d_dln_t), (d_res, z, z)

    _eval_pc.defjvp(_eval_pc_jvp)
    res_o, dlnd_o, dlnt_o = _eval_pc(rho, temp, abar, zbar, xa, azion, acmi)
    return res_o, dlnd_o, dlnt_o
