# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branchless JAX/XLA (`float64`) kernels for nuclear reaction rates (`rates`).

Provides differentiable `@jax.jit` evaluators for:
- 7-parameter REACLIB forward and detailed-balance reverse rate expansions
  (`eval_reaclib_jax`, `eval_reaclib_terms_jax`),
- Salpeter (1954), Graboske/Alastuey/Itoh (`screen5`), and Chugunov et al.
  (2007) plasma electrostatic screening (`screen_salpeter_jax`,
  `screen_extended_jax`, `screen_chugunov_jax`), and
- Fushiki & Lamb (1987) pycnonuclear + thermally enhanced triple-alpha energy
  generation (`eval_fl_epsnuc_3alf_jax`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants
from flexmesa.rates import jax_screening

screen_chugunov_jax = jax_screening.screen_chugunov_jax
screen_extended_jax = jax_screening.screen_extended_jax
screen_salpeter_jax = jax_screening.screen_salpeter_jax

_MAX_LN_LAMBDA: float = 227.955924206411
_DEFAULT_REACLIB_MIN_T9: float = 1.0e-2
_TPART_JAX: jax.Array = jnp.asarray(
    [
        0.1,
        0.15,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
    ],
    dtype=jnp.float64,
)


@jax.jit
def eval_reaclib_jax(
    coeffs: jax.Array,
    t9: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate a 7-parameter REACLIB forward rate expansion in JAX (`float64`).

    Args:
        coeffs: Array of shape `(N_terms, 7)` or `(7,)` containing REACLIB
            polynomial coefficients $a_0 \\dots a_6$.
        t9: Temperature $T_9 = T / 10^9\\,\\mathrm{K}$ (`scalar` or `(N,)`).

    Returns:
        Tuple `(rate, drate_dln_t)` of `float64` JAX arrays with the shape of
        `t9`, in CGS molar reaction units.
    """
    c2d = jnp.atleast_2d(jnp.asarray(coeffs, dtype=jnp.float64))
    t = jnp.maximum(jnp.asarray(t9, dtype=jnp.float64), 1.0e-30)
    ln_t9 = jnp.log(t)
    basis = jnp.stack(
        [
            jnp.ones_like(t),
            t**-1.0,
            t**-constants.ONE_THIRD,
            t**constants.ONE_THIRD,
            t,
            t ** (5.0 / 3.0),
            ln_t9,
        ],
        axis=-1,
    )
    dbasis_dln_t = jnp.stack(
        [
            jnp.zeros_like(t),
            -(t**-1.0),
            -constants.ONE_THIRD * (t**-constants.ONE_THIRD),
            constants.ONE_THIRD * (t**constants.ONE_THIRD),
            t,
            (5.0 / 3.0) * (t ** (5.0 / 3.0)),
            jnp.ones_like(t),
        ],
        axis=-1,
    )
    ln_l = jnp.minimum(jnp.matmul(basis, c2d.T), _MAX_LN_LAMBDA)
    dln_l = jnp.matmul(dbasis_dln_t, c2d.T)
    lam_terms = jnp.exp(ln_l)
    rate = jnp.sum(lam_terms, axis=-1)
    drate_dln_t = jnp.sum(lam_terms * dln_l, axis=-1)
    return rate, drate_dln_t


@jax.jit
def eval_reaclib_terms_jax(
    coeffs: jax.Array,
    inv_exps: jax.Array,
    inv_c1s: jax.Array,
    inv_c2s: jax.Array,
    inv_parts: jax.Array,
    rflags: jax.Array,
    t9: jax.Array,
    check_e_flag: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate forward and detailed-balance reverse REACLIB rates in JAX.

    Args:
        coeffs: `(N_terms, 7)` REACLIB polynomial coefficients.
        inv_exps: `(N_terms,)` detailed-balance exponent $\\Delta N$.
        inv_c1s: `(N_terms,)` first reverse coefficient $c_1$.
        inv_c2s: `(N_terms,)` second reverse coefficient $c_2$.
        inv_parts: `(N_terms, 24)` partition-function ratios on `TPART`.
        rflags: `(N_terms,)` ASCII byte codes for reaction flags.
        t9: Temperature $T_9 = T / 10^9\\,\\mathrm{K}$.
        check_e_flag: Boolean flag exempting `'e'` terms from low-$T_9$ cutoff.

    Returns:
        Tuple `(lam, dlam_dln_t, rlam, drlam_dln_t)` of `float64` JAX arrays.
    """
    t_raw = jnp.asarray(t9, dtype=jnp.float64)
    t = jnp.maximum(t_raw, 1.0e-30)
    ln_t9 = jnp.log(t)
    basis = jnp.stack(
        [
            jnp.ones_like(t),
            t**-1.0,
            t**-constants.ONE_THIRD,
            t**constants.ONE_THIRD,
            t,
            t ** (5.0 / 3.0),
            ln_t9,
        ],
        axis=-1,
    )
    dbasis_dln_t = jnp.stack(
        [
            jnp.zeros_like(t),
            -(t**-1.0),
            -constants.ONE_THIRD * (t**-constants.ONE_THIRD),
            constants.ONE_THIRD * (t**constants.ONE_THIRD),
            t,
            (5.0 / 3.0) * (t ** (5.0 / 3.0)),
            jnp.ones_like(t),
        ],
        axis=-1,
    )
    ln_l = jnp.minimum(jnp.matmul(basis, coeffs.T), _MAX_LN_LAMBDA)
    dln_l = jnp.matmul(dbasis_dln_t, coeffs.T)
    lam_terms = jnp.exp(ln_l)
    lam_sum = jnp.sum(lam_terms, axis=-1)
    dlam_sum = jnp.sum(lam_terms * dln_l, axis=-1)

    t_exp = jnp.expand_dims(t, axis=-1)
    ln_t9_exp = jnp.expand_dims(ln_t9, axis=-1)
    inv_ln = jnp.minimum(
        ln_l + inv_c1s + inv_c2s / t_exp + 1.5 * inv_exps * ln_t9_exp,
        _MAX_LN_LAMBDA,
    )
    dinv_ln = dln_l - inv_c2s / t_exp + 1.5 * inv_exps
    inv_l0 = jnp.exp(inv_ln)

    idx_m = jnp.clip(jnp.searchsorted(_TPART_JAX, t, side="right") - 1, 0, 22)
    t_lo = _TPART_JAX[idx_m]
    t_hi = _TPART_JAX[idx_m + 1]
    dt = t_hi - t_lo
    tfac = jnp.expand_dims((t - t_lo) / dt, axis=-1)
    inv_parts_t = jnp.swapaxes(inv_parts, -2, -1)
    p_lo = inv_parts_t[idx_m]
    p_hi = inv_parts_t[idx_m + 1]
    ratio = jnp.maximum(p_hi / jnp.maximum(p_lo, 1.0e-30), 1.0e-30)
    qr_mid = p_lo * (ratio**tfac)
    dqr_mid = qr_mid * jnp.log(ratio) * jnp.expand_dims(t / dt, axis=-1)

    below_grid = jnp.expand_dims(t < _TPART_JAX[0], axis=-1)
    qratio = jnp.where(below_grid, inv_parts[..., 0], qr_mid)
    dqratio = jnp.where(below_grid, 0.0, dqr_mid)

    not_weak = (rflags != ord("w")) & (rflags != ord("e"))
    rl_terms = jnp.where(not_weak, qratio * inv_l0, 0.0)
    drl_terms = jnp.where(not_weak, rl_terms * dinv_ln + dqratio * inv_l0, 0.0)
    rlam_sum = jnp.sum(rl_terms, axis=-1)
    drlam_sum = jnp.sum(drl_terms, axis=-1)

    rf0 = rflags[0]
    exempt = (rf0 == ord("w")) | (jnp.asarray(check_e_flag) & (rf0 == ord("e")))
    zero_mask = (t_raw < _DEFAULT_REACLIB_MIN_T9) & (~exempt)
    z0 = jnp.zeros_like(t_raw)
    return (
        jnp.where(zero_mask, z0, lam_sum),
        jnp.where(zero_mask, z0, dlam_sum),
        jnp.where(zero_mask, z0, rlam_sum),
        jnp.where(zero_mask, z0, drlam_sum),
    )


@jax.jit
def eval_fl_epsnuc_3alf_jax(
    temp: jax.Array,
    rho: jax.Array,
    y_he4: jax.Array,
    mu_e: jax.Array,
) -> jax.Array:
    """Evaluate Fushiki & Lamb (1987) pycnonuclear 3-alpha rate in JAX.

    Args:
        temp: Temperature $T$ in $\\mathrm{K}$.
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        y_he4: Helium-4 mass fraction $Y$.
        mu_e: Electron molecular weight $\\mu_e \\approx 1 / Y_e$.

    Returns:
        Specific nuclear energy generation rate $\\epsilon_{3\\alpha}$ in
        $\\mathrm{erg\\,g^{-1}\\,s^{-1}}$.
    """
    t_v = jnp.asarray(temp, dtype=jnp.float64)
    rho_v = jnp.asarray(rho, dtype=jnp.float64)
    y_v = jnp.asarray(y_he4, dtype=jnp.float64)
    ue_v = jnp.asarray(mu_e, dtype=jnp.float64)

    r6 = 1.0e-6 * rho_v
    r6t = 2.0 * r6 / ue_v
    t6 = 1.0e-6 * t_v
    r6t13 = r6t**constants.ONE_THIRD
    t623 = t6 ** (2.0 / 3.0)
    u = 1.35 * r6t13 / t623
    t612 = jnp.sqrt(t6)
    t632 = t6**1.5
    t613 = t6**constants.ONE_THIRD
    t653 = t6 ** (5.0 / 3.0)

    a1 = ((1.0 - 4.222e-2 * t623) ** 2.0) + 2.643e-5 * t653
    b1_lo = 16.16 / (a1 * t623) * jnp.exp(-134.92 / t613) + jnp.where(
        r6t < 5.458e3, jnp.exp(-1065.1 / t6) / t632, 0.0
    )
    g1_lo = b1_lo * jnp.exp(60.492 * r6t13 / t6)
    a2 = ((1.0 - 2.807e-2 * t623) ** 2.0) + 2.704e-6 * t653
    b2_lo = (
        244.6
        / (a2 * t623)
        * ((1.0 + 3.528e-3 * t623) ** 5.0)
        * jnp.exp(-235.72 / t613)
    ) + jnp.where(r6t < 1.836e4, jnp.exp(-3336.4 / t6) / t632, 0.0)
    g2_lo = b2_lo * jnp.exp(106.35 * r6t13 / t6)

    r6t16 = r6t ** (1.0 / 6.0)
    af = (u**-1.5) + 1.0
    b1_hi = (
        1.178
        * af
        * jnp.exp(-77.554 / r6t16)
        / (t612 * (((1.0 - 5.680e-2 * r6t13) ** 2.0) + 8.815e-7 * t6 * t6))
    )
    g1_hi = b1_hi + jnp.where(
        r6t < 5.458e3, jnp.exp((60.492 * r6t13 - 1065.1) / t6) / t632, 0.0
    )
    b2_hi = (
        13.48
        * af
        * ((1.0 + 5.070e-3 * r6t13) ** 5.0)
        * jnp.exp(-135.08 / r6t16)
        / (t612 * (((1.0 - 3.791e-2 * r6t13) ** 2.0) + 5.162e-8 * t6 * t6))
    )
    g2_hi = b2_hi + jnp.where(
        r6t < 1.836e4, jnp.exp((106.35 * r6t13 - 3336.4) / t6) / t632, 0.0
    )

    g1 = jnp.where(u < 1.0, g1_lo, g1_hi)
    g2 = jnp.where(u < 1.0, g2_lo, g2_hi)
    return 5.120e29 * g1 * g2 * (y_v**3.0) * r6 * r6
