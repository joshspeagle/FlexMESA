# SPDX-License-Identifier: LGPL-3.0-or-later
"""Relativistic Fermi-Dirac integrals via Aparicio (1998) quadrature.

Provides the Tier B dual-track (`backend="auto" | "rust" | "jax"`) Python and
JAX interfaces for `dfermi` (`eos_fermi_dirac_integral` in `gauss_fermi.f90`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust

_DQLEG020_XG: jnp.ndarray = jnp.asarray(
    [
        7.6526521133497334e-2,
        2.2778585114164508e-1,
        3.7370608871541956e-1,
        5.1086700195082710e-1,
        6.3605368072651503e-1,
        7.4633190646015079e-1,
        8.3911697182221882e-1,
        9.1223442825132591e-1,
        9.6397192727791379e-1,
        9.9312859918509492e-1,
    ],
    dtype=jnp.float64,
)

_DQLEG020_WG: jnp.ndarray = jnp.asarray(
    [
        1.5275338713072585e-1,
        1.4917298647260375e-1,
        1.4209610931838205e-1,
        1.3168863844917663e-1,
        1.1819453196151842e-1,
        1.0193011981724044e-1,
        8.3276741576704749e-2,
        6.2672048334109064e-2,
        4.0601429800386941e-2,
        1.7614007139152118e-2,
    ],
    dtype=jnp.float64,
)

_DQLAG020_XG: jnp.ndarray = jnp.asarray(
    [
        7.0539889691988753e-2,
        3.7212681800161144e-1,
        9.1658210248327356e-1,
        1.7073065310283439e0,
        2.7491992553094321e0,
        4.0489253138508869e0,
        5.6151749708616165e0,
        7.4590174536710633e0,
        9.5943928695810968e0,
        1.2038802546964316e1,
        1.4814293442630740e1,
        1.7948895520519376e1,
        2.1478788240285011e1,
        2.5451702793186906e1,
        2.9932554631700612e1,
        3.5013434240479000e1,
        4.0833057056728571e1,
        4.7619994047346502e1,
        5.5810795750063899e1,
        6.6524416525615754e1,
    ],
    dtype=jnp.float64,
)

_DQLAG020_WG: jnp.ndarray = jnp.asarray(
    [
        1.8108006241898926e-1,
        4.2255676787856397e-1,
        6.6690954670184815e-1,
        9.1535237278307367e-1,
        1.1695397071955460e0,
        1.4313549859282060e0,
        1.7029811379850227e0,
        1.9870158907927472e0,
        2.2866357812534308e0,
        2.6058347275538333e0,
        2.9497837342139509e0,
        3.3253957820093196e0,
        3.7422554705898109e0,
        4.2142367102518804e0,
        4.7625184614902093e0,
        5.4217260442455743e0,
        6.2540123569324213e0,
        7.3873143890544346e0,
        9.1513287309874796e0,
        1.2893388645939997e1,
    ],
    dtype=jnp.float64,
)


def _fdfunc1_jax(
    x: jax.Array,
    dk: jax.Array,
    eta: jax.Array,
    theta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate `fdfunc1` branchlessly in JAX `float64`."""
    xdk = jnp.power(x, dk)
    xdkp1 = x * xdk
    dxst = jnp.sqrt(1.0 + 0.5 * x * theta)
    diff = x - eta
    small = diff < 100.0
    factor_s = jnp.exp(jnp.where(small, diff, 0.0))
    denom_s = factor_s + 1.0
    fd_s = xdk * dxst / denom_s
    fdeta_s = fd_s * factor_s / denom_s
    fdtheta_s = xdkp1 / (4.0 * dxst * denom_s)

    factor_l = jnp.exp(jnp.where(small, 0.0, -diff))
    fd_l = xdk * dxst * factor_l
    fdeta_l = fd_l
    fdtheta_l = (xdkp1 / (4.0 * dxst)) * factor_l
    return (
        jnp.where(small, fd_s, fd_l),
        jnp.where(small, fdeta_s, fdeta_l),
        jnp.where(small, fdtheta_s, fdtheta_l),
    )


def _fdfunc2_jax(
    x: jax.Array,
    dk: jax.Array,
    eta: jax.Array,
    theta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate `fdfunc2` ($z^2 = x$) branchlessly in JAX `float64`."""
    xsq = x * x
    xdk = jnp.power(x, 2.0 * dk + 1.0)
    xdkp1 = xsq * xdk
    dxst = jnp.sqrt(1.0 + 0.5 * xsq * theta)
    diff = xsq - eta
    small = diff < 100.0
    factor_s = jnp.exp(jnp.where(small, diff, 0.0))
    denom_s = factor_s + 1.0
    fd_s = 2.0 * xdk * dxst / denom_s
    fdeta_s = fd_s * factor_s / denom_s
    fdtheta_s = 2.0 * xdkp1 / (4.0 * dxst * denom_s)

    factor_l = jnp.exp(jnp.where(small, 0.0, -diff))
    fd_l = 2.0 * xdk * dxst * factor_l
    fdeta_l = fd_l
    fdtheta_l = (2.0 * xdkp1 / (4.0 * dxst)) * factor_l
    return (
        jnp.where(small, fd_s, fd_l),
        jnp.where(small, fdeta_s, fdeta_l),
        jnp.where(small, fdtheta_s, fdtheta_l),
    )


def _dfermi_core(
    dk: jax.Array,
    eta: jax.Array,
    theta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate Aparicio (1998) `dfermi` in branchless JAX `float64`."""
    dk_f = jnp.asarray(dk, dtype=jnp.float64)
    eta_f = jnp.asarray(eta, dtype=jnp.float64)
    theta_f = jnp.asarray(theta, dtype=jnp.float64)

    d = 3.3609
    sg = 9.1186e-2
    eta1 = sg * (eta_f - d)
    xi = jnp.where(
        eta1 <= 50.0,
        jnp.log1p(jnp.exp(jnp.minimum(eta1, 50.0))) / sg,
        eta_f - d,
    )
    xi2 = xi * xi
    x1 = (6.7774 + 1.1418 * xi + 2.9826 * xi2) / (1.0 + 2.9826 * xi)
    x2 = (3.7601 + 9.3719e-2 * xi + 2.1063e-2 * 3.1084e1 * xi2) / (
        1.0 + 1.0056 * xi + 2.1063e-2 * xi2
    )
    x3 = (7.5669 + 1.1695 * xi + 7.5416e-1 * 6.6558 * xi2) / (
        1.0 - 1.2819e-1 * xi + 7.5416e-1 * xi2
    )
    s1 = x1 - x2
    s2 = x1
    s3 = x1 + x3
    s12 = jnp.sqrt(s1)

    def _leg(
        fn: Callable[
            [jax.Array, jax.Array, jax.Array, jax.Array],
            tuple[jax.Array, jax.Array, jax.Array],
        ],
        a_b: jax.Array,
        b_b: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        c = 0.5 * (a_b + b_b)
        h = 0.5 * (b_b - a_b)
        absc1 = c[..., None] + h[..., None] * _DQLEG020_XG
        absc2 = c[..., None] - h[..., None] * _DQLEG020_XG
        f1, df1, ddf1 = fn(absc1, dk_f, eta_f[..., None], theta_f[..., None])
        f2, df2, ddf2 = fn(absc2, dk_f, eta_f[..., None], theta_f[..., None])
        return (
            h * jnp.sum((f1 + f2) * _DQLEG020_WG, axis=-1),
            h * jnp.sum((df1 + df2) * _DQLEG020_WG, axis=-1),
            h * jnp.sum((ddf1 + ddf2) * _DQLEG020_WG, axis=-1),
        )

    r1, dr1, ddr1 = _leg(_fdfunc2_jax, jnp.zeros_like(s12), s12)
    r2, dr2, ddr2 = _leg(_fdfunc1_jax, s1, s2)
    r3, dr3, ddr3 = _leg(_fdfunc1_jax, s2, s3)

    absc_lag = s3[..., None] + _DQLAG020_XG
    fl, dfl, ddfl = _fdfunc1_jax(
        absc_lag, dk_f, eta_f[..., None], theta_f[..., None]
    )
    r4 = jnp.sum(fl * _DQLAG020_WG, axis=-1)
    dr4 = jnp.sum(dfl * _DQLAG020_WG, axis=-1)
    ddr4 = jnp.sum(ddfl * _DQLAG020_WG, axis=-1)

    return r1 + r2 + r3 + r4, dr1 + dr2 + dr3 + dr4, ddr1 + ddr2 + ddr3 + ddr4


_dfermi_jit = jax.jit(_dfermi_core)


def dfermi_jax(
    dk: jax.Array,
    eta: jax.Array,
    theta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate Aparicio (1998) `dfermi` in branchless JAX `float64`.

    Args:
        dk: Fermi-Dirac order $k$ (dimensionless).
        eta: Degeneracy parameter $\\eta = \\mu_e / (k_{\\mathrm{B}} T)$.
        theta: Relativity parameter $\\theta = k_{\\mathrm{B}} T / (m_e c^2)$.

    Returns:
        Tuple `(fd, fd_deta, fd_dtheta)` of `float64` JAX arrays.
    """
    fd, fdeta, fdtheta = _dfermi_jit(dk, eta, theta)
    return (
        cast(jax.Array, fd),
        cast(jax.Array, fdeta),
        cast(jax.Array, fdtheta),
    )


def dfermi(
    dk: float,
    eta: float | np.ndarray | jax.Array,
    theta: float | np.ndarray | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> tuple[
    float | np.ndarray | jax.Array,
    float | np.ndarray | jax.Array,
    float | np.ndarray | jax.Array,
]:
    """Evaluate the generalized relativistic Fermi-Dirac integral `dfermi`.

    Args:
        dk: Fermi-Dirac order $k$ (dimensionless).
        eta: Degeneracy parameter $\\eta = \\mu_e / (k_{\\mathrm{B}} T)$ (scalar
            or 1D array of shape `(N_zones,)`).
        theta: Relativity parameter $\\theta = k_{\\mathrm{B}} T / (m_e c^2)$
            (scalar or 1D array of shape `(N_zones,)`).
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Tuple `(fd, fd_deta, fd_dtheta)`.
    """
    resolved = _dispatch.resolve_backend(backend, eta, theta)
    if resolved == "jax":
        return dfermi_jax(
            jnp.asarray(dk, dtype=jnp.float64),
            jnp.asarray(eta, dtype=jnp.float64),
            jnp.asarray(theta, dtype=jnp.float64),
        )
    if np.ndim(eta) == 0 and np.ndim(theta) == 0:
        return _rust.eos_dfermi(
            float(dk),
            float(np.asarray(eta).item()),
            float(np.asarray(theta).item()),
        )
    eta_arr = np.ascontiguousarray(eta, dtype=np.float64)
    theta_arr = np.ascontiguousarray(theta, dtype=np.float64)
    out = _rust.eos_dfermi_zones(float(dk), eta_arr, theta_arr, True)
    return (
        cast(np.ndarray, out[:, 0]),
        cast(np.ndarray, out[:, 1]),
        cast(np.ndarray, out[:, 2]),
    )
