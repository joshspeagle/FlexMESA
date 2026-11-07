# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differentiable JAX wrappers for nuclear reaction networks (`net`).

Implements `eval_net_jax` with an exact `jax.custom_vjp` rule backed by the
analytical Jacobians (`d_eps_nuc_dx`, `d_eps_nuc_dt`, `d_eps_nuc_drho`,
`d_dxdt_dx`, `d_dxdt_dt`, `d_dxdt_drho`) computed in Safe Rust (`flexmesa-net`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _rust
from flexmesa.net import _core


def eval_net_jax(
    net_name: str,
    x: jax.Array,
    temp: jax.Array,
    rho: jax.Array,
    abar: float,
    zbar: float,
    z2bar: float,
    ye: float,
    eta: float = 0.0,
    d_eta_dlnt: float = 0.0,
    d_eta_dlnrho: float = 0.0,
    weak_rate_factor: float = 1.0,
    screening_mode: str = "extended",
) -> tuple[jax.Array, jax.Array]:
    """Evaluate differentiable `(eps_nuc, dxdt)` with `jax.custom_vjp`.

    Propagates reverse-mode sensitivities with respect to `(x, temp, rho)`
    using the exact analytical derivatives `d_eps_nuc_dx`, `d_eps_nuc_dt`,
    `d_eps_nuc_drho`, `d_dxdt_dx`, `d_dxdt_dt`, and `d_dxdt_drho` from
    `flexmesa._rust.net_eval`.

    Args:
        net_name: Network filename (e.g., `"basic.net"`, `"approx21.net"`).
        x: Mass fractions (`(S,)`).
        temp: Temperature $T$ ($\\mathrm{K}$).
        rho: Mass density $\\rho$ ($\\mathrm{g\\,cm^{-3}}$).
        abar: Mean atomic mass number $\\bar{A}$.
        zbar: Mean atomic charge number $\\bar{Z}$.
        z2bar: Mean squared atomic charge $\\overline{Z^2}$.
        ye: Electron-to-baryon fraction $Y_e$.
        eta: Electron degeneracy parameter $\\eta$.
        d_eta_dlnt: $(\\partial \\eta / \\partial \\ln T)_\\rho$.
        d_eta_dlnrho: $(\\partial \\eta / \\partial \\ln \\rho)_T$.
        weak_rate_factor: Global weak-rate multiplier.
        screening_mode: Plasma screening mode.

    Returns:
        Tuple `(eps_nuc, dxdt)` where `eps_nuc` is a scalar JAX `float64` array
        and `dxdt` has shape `(S,)`.
    """
    info = _core.get_net_info(net_name)
    s = info.num_isos

    def _call_net(
        xv: np.ndarray, tv: np.ndarray, rv: np.ndarray, skip_jac: bool
    ) -> tuple[np.ndarray, ...]:
        tf = float(tv)
        rf = float(rv)
        res = _rust.net_eval(
            str(net_name),
            np.ascontiguousarray(xv, dtype=np.float64),
            tf,
            float(np.log10(tf)),
            rf,
            float(np.log10(rf)),
            float(abar),
            float(zbar),
            float(z2bar),
            float(ye),
            float(eta),
            float(d_eta_dlnt),
            float(d_eta_dlnrho),
            None,
            float(weak_rate_factor),
            str(screening_mode),
            skip_jac,
        )
        if skip_jac:
            return (
                np.asarray(res[0], dtype=np.float64),
                np.asarray(res[4], dtype=np.float64),
            )
        return tuple(np.asarray(res[k], dtype=np.float64) for k in range(8))

    @jax.custom_vjp
    def _op(
        x_in: jax.Array, t_in: jax.Array, rho_in: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        out_spec = (
            jax.ShapeDtypeStruct((), jnp.float64),
            jax.ShapeDtypeStruct((s,), jnp.float64),
        )
        res = jax.pure_callback(
            lambda xv, tv, rv: _call_net(xv, tv, rv, True),
            out_spec,
            x_in,
            t_in,
            rho_in,
        )
        return res[0], res[1]

    def _fwd(
        x_in: jax.Array, t_in: jax.Array, rho_in: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]]:
        full_spec = (
            jax.ShapeDtypeStruct((), jnp.float64),
            jax.ShapeDtypeStruct((), jnp.float64),
            jax.ShapeDtypeStruct((), jnp.float64),
            jax.ShapeDtypeStruct((s,), jnp.float64),
            jax.ShapeDtypeStruct((s,), jnp.float64),
            jax.ShapeDtypeStruct((s,), jnp.float64),
            jax.ShapeDtypeStruct((s,), jnp.float64),
            jax.ShapeDtypeStruct((s, s), jnp.float64),
        )
        (
            eps_nuc,
            d_eps_drho,
            d_eps_dt,
            d_eps_dx,
            dxdt,
            d_dxdt_drho,
            d_dxdt_dt,
            d_dxdt_dx,
        ) = jax.pure_callback(
            lambda xv, tv, rv: _call_net(xv, tv, rv, False),
            full_spec,
            x_in,
            t_in,
            rho_in,
        )
        residuals = (
            d_eps_drho,
            d_eps_dt,
            d_eps_dx,
            d_dxdt_drho,
            d_dxdt_dt,
            d_dxdt_dx,
        )
        return (eps_nuc, dxdt), residuals

    def _bwd(
        residuals: tuple[jax.Array, ...],
        cotangents: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        (
            d_eps_drho,
            d_eps_dt,
            d_eps_dx,
            d_dxdt_drho,
            d_dxdt_dt,
            d_dxdt_dx,
        ) = residuals
        g_eps, g_dxdt = cotangents
        grad_x = g_eps * d_eps_dx + jnp.dot(g_dxdt, d_dxdt_dx)
        grad_t = g_eps * d_eps_dt + jnp.dot(g_dxdt, d_dxdt_dt)
        grad_rho = g_eps * d_eps_drho + jnp.dot(g_dxdt, d_dxdt_drho)
        return grad_x, grad_t, grad_rho

    _op.defvjp(_fwd, _bwd)
    return _op(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(temp, dtype=jnp.float64),
        jnp.asarray(rho, dtype=jnp.float64),
    )
