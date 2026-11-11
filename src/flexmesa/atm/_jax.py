# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differentiable JAX kernels for surface boundary conditions (`atm`)."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _rust, constants
from flexmesa.atm import _core

_BOLTZ_SIGMA: float = constants.BOLTZ_SIGMA
_CLIGHT: float = constants.CLIGHT
_CRAD: float = constants.CRAD


@functools.partial(jax.jit, static_argnames=("t_tau_id",))
def eval_t_tau_jax(t_tau_id: int, tau: jax.Array, teff: jax.Array) -> jax.Array:
    """Evaluate $\\ln T(\\tau, T_{\\mathrm{eff}})$ in pure branchless JAX.

    Args:
        t_tau_id: $T(\\tau)$ relation identifier (`1`: Eddington,
            `2`: Solar Hopf, `3`: Krishna Swamy, `4`: Trampedach solar).
        tau: Optical depth $\\tau$ (scalar or array).
        teff: Effective temperature $T_{\\mathrm{eff}}$ in $\\mathrm{K}$.

    Returns:
        Natural logarithm of temperature $\\ln(T / \\mathrm{K})$.
    """
    tau_arr = jnp.asarray(tau, dtype=jnp.float64)
    teff_arr = jnp.asarray(teff, dtype=jnp.float64)
    teff4 = teff_arr * teff_arr * teff_arr * teff_arr
    if t_tau_id == _core.ATM_T_TAU_EDDINGTON:
        q: jax.Array | float = 2.0 / 3.0
    elif t_tau_id == _core.ATM_T_TAU_SOLAR_HOPF:
        q = (
            1.0361
            - 0.3134 * jnp.exp(-2.44799995 * tau_arr)
            - 0.29589999 * jnp.exp(-30.0 * tau_arr)
        )
    elif t_tau_id == _core.ATM_T_TAU_KRISHNA_SWAMY:
        q = (
            1.39
            - 0.815 * jnp.exp(-2.54 * tau_arr)
            - 0.025 * jnp.exp(-30.0 * tau_arr)
        )
    elif t_tau_id == _core.ATM_T_TAU_TRAMPEDACH_SOLAR:
        x = jnp.log10(jnp.maximum(tau_arr, 1.0e-99))
        c0 = 0.6887302005929656
        c1 = 0.0668697860833449
        a = 0.9262126497691250
        v = 0.7657856893402466
        b = 0.1148742902769433
        q = c0 + c1 * (x - b) + v * jnp.exp((x - a) / v)
    else:
        msg = f"Unknown t_tau_id: {t_tau_id}"
        raise ValueError(msg)
    return 0.25 * jnp.log(0.75 * teff4 * (tau_arr + q))


@jax.jit
def eval_paczynski_gradr_jax(
    t: jax.Array,
    p: jax.Array,
    rho: jax.Array,
    tau: jax.Array,
    kap: jax.Array,
    luminosity: jax.Array,
    m: jax.Array,
    r: jax.Array,
    cgrav: jax.Array,
) -> jax.Array:
    """Evaluate Paczynski (1969) diluted $\\nabla_{\\mathrm{rad}}$ in JAX.

    Args:
        t: Temperature $T$ ($\\mathrm{K}$).
        p: Pressure $P$ ($\\mathrm{dyn\\,cm^{-2}}$).
        rho: Density $\\rho$ ($\\mathrm{g\\,cm^{-3}}$).
        tau: Optical depth $\\tau$.
        kap: Opacity $\\kappa$ ($\\mathrm{cm^2\\,g^{-1}}$).
        luminosity: Luminosity $L$ ($\\mathrm{erg\\,s^{-1}}$).
        m: Mass $M$ ($\\mathrm{g}$).
        r: Radius $R$ ($\\mathrm{cm}$).
        cgrav: Gravitational constant $G$ ($\\mathrm{cm^3\\,g^{-1}\\,s^{-2}}$).

    Returns:
        Diluted radiative temperature gradient $\\nabla_{\\mathrm{rad}}$.
    """
    prad = _CRAD * t**4 / 3.0
    gradr = p * kap * luminosity / (16.0 * jnp.pi * _CLIGHT * m * cgrav * prad)
    s = (
        (2.0 * _CRAD * t**3 * jnp.sqrt(r))
        / (3.0 * cgrav * m * rho)
        * (luminosity / (8.0 * jnp.pi * _BOLTZ_SIGMA)) ** 0.25
    )
    f = 1.0 - 1.5 * tau
    dilution = (
        1.0 + f * s * (4.0 * jnp.pi * cgrav * _CLIGHT * m) / (kap * luminosity)
    ) / (1.0 + f * s)
    return jnp.where(tau < 2.0 / 3.0, gradr * dilution, gradr)


def _run_atm_table_cb(
    table_id: int,
    lv: np.ndarray,
    rv: np.ndarray,
    mv: np.ndarray,
    cgv: np.ndarray,
    zv: np.ndarray,
    skip_partials: bool,
) -> tuple[np.ndarray, ...]:
    lf = float(lv)
    rf = float(rv)
    mf = float(mv)
    teff = (lf / (4.0 * np.pi * rf * rf * _BOLTZ_SIGMA)) ** 0.25
    res = _rust.atm_eval_table(
        lf,
        rf,
        mf,
        float(cgv),
        int(table_id),
        float(zv),
        skip_partials,
        teff,
    )
    if skip_partials:
        return (
            np.asarray(res[0], dtype=np.float64),
            np.asarray(res[5], dtype=np.float64),
        )
    return tuple(np.asarray(v, dtype=np.float64) for v in res)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _eval_table_jax_op(
    table_id: int,
    l_in: jax.Array,
    r_in: jax.Array,
    m_in: jax.Array,
    cgrav_in: jax.Array,
    z_in: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    out_spec = (
        jax.ShapeDtypeStruct((), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.float64),
    )

    def _cb(
        lv: np.ndarray,
        rv: np.ndarray,
        mv: np.ndarray,
        cgv: np.ndarray,
        zv: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        return _run_atm_table_cb(table_id, lv, rv, mv, cgv, zv, True)

    res = jax.pure_callback(_cb, out_spec, l_in, r_in, m_in, cgrav_in, z_in)
    return res[0], res[1]


def _eval_table_jax_fwd(
    table_id: int,
    l_in: jax.Array,
    r_in: jax.Array,
    m_in: jax.Array,
    cgrav_in: jax.Array,
    z_in: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]]:
    spec11 = tuple(jax.ShapeDtypeStruct((), jnp.float64) for _ in range(11))

    def _cb_full(
        lv: np.ndarray,
        rv: np.ndarray,
        mv: np.ndarray,
        cgv: np.ndarray,
        zv: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        return _run_atm_table_cb(table_id, lv, rv, mv, cgv, zv, False)

    vals = jax.pure_callback(_cb_full, spec11, l_in, r_in, m_in, cgrav_in, z_in)
    ln_t, dln_t_dl, dln_t_dln_r, dln_t_dln_m = (
        vals[0],
        vals[1],
        vals[2],
        vals[3],
    )
    ln_p, dln_p_dl, dln_p_dln_r, dln_p_dln_m = (
        vals[5],
        vals[6],
        vals[7],
        vals[8],
    )
    residuals = (
        r_in,
        m_in,
        dln_t_dl,
        dln_t_dln_r,
        dln_t_dln_m,
        dln_p_dl,
        dln_p_dln_r,
        dln_p_dln_m,
    )
    return (ln_t, ln_p), residuals


def _eval_table_jax_bwd(
    _table_id: int,
    residuals: tuple[jax.Array, ...],
    cotangents: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    (
        r_in,
        m_in,
        dln_t_dl,
        dln_t_dln_r,
        dln_t_dln_m,
        dln_p_dl,
        dln_p_dln_r,
        dln_p_dln_m,
    ) = residuals
    g_lnt, g_lnp = cotangents
    dl = g_lnt * dln_t_dl + g_lnp * dln_p_dl
    dr = (g_lnt * dln_t_dln_r + g_lnp * dln_p_dln_r) / r_in
    dm = (g_lnt * dln_t_dln_m + g_lnp * dln_p_dln_m) / m_in
    zero = jnp.zeros_like(dl)
    return dl, dr, dm, zero, zero


_eval_table_jax_op.defvjp(_eval_table_jax_fwd, _eval_table_jax_bwd)


def eval_table_jax(
    luminosity: jax.Array,
    r: jax.Array,
    m: jax.Array,
    cgrav: float | jax.Array = constants.STANDARD_CGRAV,
    table_id: int = _core.ATM_TABLE_PHOTOSPHERE,
    z: float | jax.Array = 0.02,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate differentiable $(\\ln T, \\ln P)$ from an atmosphere table.

    Uses a `jax.custom_vjp` rule backed by the analytical bicubic spline
    derivatives $(\\partial \\ln T / \\partial L,
    \\partial \\ln T / \\partial \\ln R, \\partial \\ln T / \\partial \\ln M)$
    and $(\\partial \\ln P / \\partial L, \\partial \\ln P / \\partial \\ln R,
    \\partial \\ln P / \\partial \\ln M)$ computed in `flexmesa-atm`.

    Args:
        luminosity: Luminosity $L$ ($\\mathrm{erg\\,s^{-1}}$).
        r: Photospheric radius $R$ ($\\mathrm{cm}$).
        m: Stellar mass $M$ ($\\mathrm{g}$).
        cgrav: Gravitational constant $G$ ($\\mathrm{cm^3\\,g^{-1}\\,s^{-2}}$).
        table_id: Atmosphere table identifier (`100`..`106`).
        z: Metallicity mass fraction $Z$.

    Returns:
        Tuple `(ln_t, ln_p)` as scalar JAX `float64` arrays.
    """
    ln_t, ln_p = _eval_table_jax_op(
        int(table_id),
        jnp.asarray(luminosity, dtype=jnp.float64),
        jnp.asarray(r, dtype=jnp.float64),
        jnp.asarray(m, dtype=jnp.float64),
        jnp.asarray(cgrav, dtype=jnp.float64),
        jnp.asarray(z, dtype=jnp.float64),
    )
    return ln_t, ln_p
