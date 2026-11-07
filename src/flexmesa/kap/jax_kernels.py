# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differentiable JAX (`float64`) opacity kernels for `flexmesa.kap`."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _rust, constants

_KEV_PER_K: float = constants.KEV * 1.0e-3
_THOMSON_OPACITY_COEF: float = constants.SIGE / constants.AMU


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
    """Evaluate Poutanen (2017) relativistic/degenerate Compton opacity in JAX.

    Args:
        rho: Mass density in `g cm^-3`, shape `(*batch,)`.
        t: Temperature in `K`, shape `(*batch,)`.
        lnfree_e: Natural log of free electrons per nucleon, shape `(*batch,)`.
        d_lnfree_e_dln_rho: Partial `(d ln free_e / d ln rho)_T`.
        d_lnfree_e_dln_t: Partial `(d ln free_e / d ln T)_rho`.
        eta: Electron degeneracy parameter `psi = mu_e / (k_B T)`.
        d_eta_dln_rho: Partial `(d eta / d ln rho)_T`.
        d_eta_dln_t: Partial `(d eta / d ln T)_rho`.

    Returns:
        Tuple `(kap, dlnkap_dln_rho, dlnkap_dln_t)` where `kap` has units
        `cm^2 g^-1`.
    """
    _ = rho
    t_a = jnp.asarray(t, dtype=jnp.float64)
    lnfe = jnp.asarray(lnfree_e, dtype=jnp.float64)
    dlnfe_dr = jnp.asarray(d_lnfree_e_dln_rho, dtype=jnp.float64)
    dlnfe_dt = jnp.asarray(d_lnfree_e_dln_t, dtype=jnp.float64)
    eta_a = jnp.asarray(eta, dtype=jnp.float64)
    deta_dr = jnp.asarray(d_eta_dln_rho, dtype=jnp.float64)
    deta_dt = jnp.asarray(d_eta_dln_t, dtype=jnp.float64)

    theta_kev = t_a * _KEV_PER_K
    c01 = 0.682
    c02 = -0.0454
    c11 = 0.240
    c12 = 0.0043
    c21 = 0.050
    c22 = -0.0067
    c31 = -0.037
    c32 = 0.0031
    t0 = 43.3
    alpha0 = 0.885

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

    dlnkap_dln_t = dlnfe_dt - dln_mfp_dln_t
    dlnkap_dln_rho = dlnfe_dr - dln_mfp_dln_rho
    return kap, dlnkap_dln_rho, dlnkap_dln_t


@jax.jit
def combine_rad_with_conduction_jax(
    kap_rad: jax.Array | float,
    dlnkap_rad_dln_rho: jax.Array | float,
    dlnkap_rad_dln_t: jax.Array | float,
    kap_cond: jax.Array | float,
    dlnkap_cond_dln_rho: jax.Array | float,
    dlnkap_cond_dln_t: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Harmonic combination of radiative and conductive opacities in JAX.

    Args:
        kap_rad: Radiative Rosseland opacity in `cm^2 g^-1`.
        dlnkap_rad_dln_rho: Radiative density derivative.
        dlnkap_rad_dln_t: Radiative temperature derivative.
        kap_cond: Conductive opacity in `cm^2 g^-1`.
        dlnkap_cond_dln_rho: Conductive density derivative.
        dlnkap_cond_dln_t: Conductive temperature derivative.

    Returns:
        Tuple `(kap, dlnkap_dln_rho, dlnkap_dln_t)` for harmonic sum
        `1 / kap = 1 / kap_rad + 1 / kap_cond`.
    """
    kr = jnp.asarray(kap_rad, dtype=jnp.float64)
    dr_r = jnp.asarray(dlnkap_rad_dln_rho, dtype=jnp.float64)
    dr_t = jnp.asarray(dlnkap_rad_dln_t, dtype=jnp.float64)
    kc = jnp.asarray(kap_cond, dtype=jnp.float64)
    dc_r = jnp.asarray(dlnkap_cond_dln_rho, dtype=jnp.float64)
    dc_t = jnp.asarray(dlnkap_cond_dln_t, dtype=jnp.float64)

    denom = kr + kc
    kap = (kr * kc) / denom
    w_rad = kc / denom
    w_cond = kr / denom
    dlnkap_dln_rho = w_rad * dr_r + w_cond * dc_r
    dlnkap_dln_t = w_rad * dr_t + w_cond * dc_t
    return kap, dlnkap_dln_rho, dlnkap_dln_t


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
    """Create a `jax.custom_jvp` function `f(log_rho, log_t) -> log10_kap`.

    Propagates exact analytical spline and conductive/Compton derivatives
    `(d log10 kap / d log10 rho, d log10 kap / d log10 T)` through `jax.grad`
    and `jax.jvp`.

    Args:
        z: Total metal mass fraction `Z`.
        x: Hydrogen mass fraction `X`.
        zbase: Base metallicity `Zbase`.
        xc: Carbon mass fraction `XC`.
        xn: Nitrogen mass fraction `XN`.
        xo: Oxygen mass fraction `XO`.
        xne: Neon mass fraction `XNe`.
        lnfree_e: Natural log of free electrons per nucleon.
        d_lnfree_e_dln_rho: Partial `(d ln free_e / d ln rho)_T`.
        d_lnfree_e_dln_t: Partial `(d ln free_e / d ln T)_rho`.
        eta: Electron degeneracy parameter.
        d_eta_dln_rho: Partial `(d eta / d ln rho)_T`.
        d_eta_dln_t: Partial `(d eta / d ln T)_rho`.
        zbar: Mean ionic charge `Zbar`.
        **kwargs: Optional `KapConfig` overrides (`kap_option`, etc.).

    Returns:
        A JAX-differentiable callable `fn(log_rho, log_t) -> log10_kap`.
    """

    def _host_eval(
        lr_arr: np.ndarray, lt_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lr_f = float(np.asarray(lr_arr))
        lt_f = float(np.asarray(lt_arr))
        kap, dln_dr, dln_dt, _ = _rust.kap_get_results_rust(
            z,
            x,
            zbase,
            xc,
            xn,
            xo,
            xne,
            lr_f,
            lt_f,
            lnfree_e,
            d_lnfree_e_dln_rho,
            d_lnfree_e_dln_t,
            eta,
            d_eta_dln_rho,
            d_eta_dln_t,
            zbar=zbar,
            **kwargs,
        )
        return (
            np.asarray(np.log10(kap), dtype=np.float64),
            np.asarray(dln_dr, dtype=np.float64),
            np.asarray(dln_dt, dtype=np.float64),
        )

    @jax.custom_jvp
    def _fn(log_rho: jax.Array, log_t: jax.Array) -> jax.Array:
        shape_spec = jnp.zeros((), dtype=jnp.float64)
        lk, _, _ = jax.pure_callback(
            _host_eval,
            (shape_spec, shape_spec, shape_spec),
            log_rho,
            log_t,
        )
        return jnp.asarray(lk, dtype=jnp.float64)

    @_fn.defjvp
    def _fn_jvp(
        primals: tuple[jax.Array, jax.Array],
        tangents: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        lr, lt = primals
        dlr, dlt = tangents
        shape_spec = jnp.zeros((), dtype=jnp.float64)
        lk, dln_dr, dln_dt = jax.pure_callback(
            _host_eval,
            (shape_spec, shape_spec, shape_spec),
            lr,
            lt,
        )
        lk_a = jnp.asarray(lk, dtype=jnp.float64)
        dlk = (
            jnp.asarray(dln_dr, dtype=jnp.float64) * dlr
            + jnp.asarray(dln_dt, dtype=jnp.float64) * dlt
        )
        return lk_a, dlk

    return _fn
