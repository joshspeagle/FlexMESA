# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differentiable JAX (`float64`) staggered-grid and face-microphysics kernels.

Provides pure branchless `@jax.jit` kernels for staggered cell/face geometry,
mass-weighted face interpolation (`get_face_weights`), optical depth
integration (`get_tau`), Brunt-Väisälä composition and buoyancy frequency
(`do_brunt_B`, `do_brunt_N2`), and Cox & Giuli (1968) Mixing-Length Theory
(`set_mlt_vars`) across all `nz` stellar zones.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants, turb

_FOUR_PI: float = float(constants.FOUR_PI)
_CLIGHT: float = float(constants.CLIGHT)
_CRAD: float = float(constants.CRAD)
_G_CGS: float = float(constants.STANDARD_CGRAV)


@jax.jit
def compute_staggered_grid_jax(
    dq: jax.Array,
    ln_r: jax.Array,
    mstar: float | jax.Array,
    m_center: float | jax.Array = 0.0,
    r_center: float | jax.Array = 0.0,
    cgrav: float | jax.Array = _G_CGS,
) -> dict[str, jax.Array]:
    """Compute staggered Lagrangian grid quantities across `nz` zones.

    Args:
        dq: Cell relative mass fractions of shape `(nz,)` (dimensionless).
        ln_r: Face natural log radius of shape `(nz,)` (`ln(cm)`).
        mstar: Total stellar mass in grams (`g`).
        m_center: Inner core mass in grams (`g`).
        r_center: Inner core radius in centimeters (`cm`).
        cgrav: Gravitational constant in `cm^3 g^-1 s^-2`.

    Returns:
        Dictionary with keys `("dq", "q", "xq", "dm", "dm_bar", "m", "r",
        "rmid", "grav")`, each of shape `(nz,)`.
    """
    dq_norm = dq / jnp.sum(dq)
    csum = jnp.cumsum(dq_norm)
    xq = jnp.concatenate([jnp.zeros((1,), dtype=dq_norm.dtype), csum[:-1]])
    q = jnp.where(
        jnp.arange(dq_norm.shape[0]) == dq_norm.shape[0] - 1,
        dq_norm,
        jnp.maximum(1.0 - xq, 0.0),
    )
    xmstar = mstar - m_center
    dm = dq_norm * xmstar
    m = m_center + q * xmstar

    dm_m1 = jnp.concatenate([jnp.zeros((1,), dtype=dm.dtype), dm[:-1]])
    dm_bar_mid = 0.5 * (dm_m1 + dm)
    idx = jnp.arange(dm.shape[0])
    dm_bar = jnp.where(
        idx == 0,
        0.5 * dm,
        jnp.where(idx == dm.shape[0] - 1, 0.5 * dm_m1 + dm, dm_bar_mid),
    )

    r = jnp.exp(ln_r)
    r_inner = jnp.concatenate([r[1:], jnp.asarray([r_center], dtype=r.dtype)])
    rmid = jnp.cbrt(0.5 * (r**3 + r_inner**3))
    grav = cgrav * m / (r * r)

    return {
        "dq": dq_norm,
        "q": q,
        "xq": xq,
        "dm": dm,
        "dm_bar": dm_bar,
        "m": m,
        "r": r,
        "rmid": rmid,
        "grav": grav,
    }


@jax.jit
def face_weights_jax(dq: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Compute mass-weighted face interpolation coefficients `(alfa, beta)`.

    At face `k > 0`, `f_face[k] = alfa[k] * f[k] + beta[k] * f[k - 1]`, with
    `alfa[k] = dq[k - 1] / (dq[k - 1] + dq[k])` and `beta[k] = 1 - alfa[k]`.
    At surface face `k = 0`, `alfa[0] = 1.0, beta[0] = 0.0`.

    Args:
        dq: Cell relative mass fractions of shape `(nz,)` (dimensionless).

    Returns:
        Tuple `(alfa, beta)` of shape `(nz,)`.
    """
    dq_m1 = jnp.concatenate([dq[:1], dq[:-1]])
    denom = jnp.maximum(dq_m1 + dq, 1.0e-30)
    alfa = jnp.where(jnp.arange(dq.shape[0]) == 0, 1.0, dq_m1 / denom)
    beta = 1.0 - alfa
    return alfa, beta


@jax.jit
def interp_cell_to_face_jax(
    dq: jax.Array,
    cell_val: jax.Array,
) -> jax.Array:
    """Interpolate a cell-centered array of shape `(nz,)` to faces `(nz,)`.

    Args:
        dq: Cell relative mass fractions of shape `(nz,)`.
        cell_val: Cell-centered quantity of shape `(nz,)`.

    Returns:
        Face-centered array of shape `(nz,)`.
    """
    alfa, beta = face_weights_jax(dq)
    val_m1 = jnp.concatenate([cell_val[:1], cell_val[:-1]])
    return alfa * cell_val + beta * val_m1


@jax.jit
def eval_optical_depth_jax(
    dm: jax.Array,
    rmid: jax.Array,
    opacity: jax.Array,
    tau_surf: float | jax.Array = 2.0 / 3.0,
) -> jax.Array:
    """Integrate optical depth `tau` inward from the surface across `nz` zones.

    Args:
        dm: Cell masses of shape `(nz,)` (`g`).
        rmid: Cell mid-radii of shape `(nz,)` (`cm`).
        opacity: Cell Rosseland mean opacities of shape `(nz,)` (`cm^2 g^-1`).
        tau_surf: Surface optical depth `tau[0]` (dimensionless).

    Returns:
        Face optical depth `tau` of shape `(nz,)`.
    """
    dtau = dm * opacity / (_FOUR_PI * rmid * rmid)
    csum = jnp.cumsum(dtau[:-1])
    tau_0 = jnp.asarray([tau_surf], dtype=dm.dtype)
    return jnp.concatenate([tau_0, tau_0 + csum])


@jax.jit
def eval_face_mlt_and_brunt_jax(
    dq: jax.Array,
    ln_r: jax.Array,
    lum: jax.Array,
    m_grav: jax.Array,
    rho: jax.Array,
    t: jax.Array,
    peos: jax.Array,
    ln_peos: jax.Array,
    grada: jax.Array,
    cp: jax.Array,
    chi_rho: jax.Array,
    chi_t: jax.Array,
    mu: jax.Array,
    opacity: jax.Array,
    tau_0: float | jax.Array = 2.0 / 3.0,
    mixing_length_alpha: float = 2.0,
    use_ledoux_criterion: bool = False,
    cgrav: float | jax.Array = _G_CGS,
) -> dict[str, jax.Array]:
    """Evaluate face temperature gradients, Cox MLT, and Brunt-Väisälä `N^2`.

    Args:
        dq: Cell relative mass fractions of shape `(nz,)`.
        ln_r: Face natural log radius of shape `(nz,)` (`ln(cm)`).
        lum: Face luminosity of shape `(nz,)` (`erg s^-1`).
        m_grav: Face enclosed gravitational mass of shape `(nz,)` (`g`).
        rho: Cell density of shape `(nz,)` (`g cm^-3`).
        t: Cell temperature of shape `(nz,)` (`K`).
        peos: Cell total pressure of shape `(nz,)` (`dyn cm^-2`).
        ln_peos: Cell natural log pressure of shape `(nz,)`.
        grada: Cell adiabatic temperature gradient of shape `(nz,)`.
        cp: Cell specific heat at constant pressure of shape `(nz,)`.
        chi_rho: Cell `dlnP/dlnRho` at constant `T` of shape `(nz,)`.
        chi_t: Cell `dlnP/dlnT` at constant `Rho` of shape `(nz,)`.
        mu: Cell mean molecular weight of shape `(nz,)`.
        opacity: Cell opacity of shape `(nz,)` (`cm^2 g^-1`).
        tau_0: Surface optical depth `tau[0]`.
        mixing_length_alpha: Mixing-length parameter `alpha_MLT`.
        use_ledoux_criterion: Whether to include `brunt_B` in `gradL`.
        cgrav: Gravitational constant (`cm^3 g^-1 s^-2`).

    Returns:
        Dictionary containing `("scale_height", "gradr", "gradl", "gradt",
        "mlt_vc", "mlt_d", "mlt_cdc", "brunt_b", "brunt_n2")` of shape `(nz,)`.
    """
    r = jnp.exp(ln_r)
    gm = cgrav * m_grav
    grav = gm / (r * r)

    prad = (_CRAD / 3.0) * (t**4)
    rho_f = interp_cell_to_face_jax(dq, rho)
    t_f = interp_cell_to_face_jax(dq, t)
    p_f = interp_cell_to_face_jax(dq, peos)
    prad_f = interp_cell_to_face_jax(dq, prad)
    kap_f = interp_cell_to_face_jax(dq, opacity)
    grada_f = interp_cell_to_face_jax(dq, grada)
    cp_f = interp_cell_to_face_jax(dq, cp)
    chi_rho_f = jnp.maximum(interp_cell_to_face_jax(dq, chi_rho), 1.0e-10)
    chi_t_f = jnp.maximum(interp_cell_to_face_jax(dq, chi_t), 1.0e-10)

    hp = p_f / (grav * rho_f)
    lam = mixing_length_alpha * hp

    # Brunt composition term B
    lnp_m1 = jnp.concatenate([ln_peos[:1], ln_peos[:-1]])
    mu_m1 = jnp.concatenate([mu[:1], mu[:-1]])
    dlnp = lnp_m1 - ln_peos
    dlnmu = jnp.log(jnp.maximum(mu_m1, 1.0e-10)) - jnp.log(
        jnp.maximum(mu, 1.0e-10)
    )
    b_raw = (chi_rho_f / chi_t_f) * (
        dlnmu / jnp.where(jnp.abs(dlnp) > 1.0e-14, dlnp, 1.0)
    )
    brunt_b = jnp.where(
        (jnp.arange(dq.shape[0]) > 0) & (jnp.abs(dlnp) > 1.0e-14),
        jnp.clip(b_raw, -10.0, 10.0),
        0.0,
    )

    gradr_base = p_f * kap_f * lum / (_FOUR_PI * _CLIGHT * gm * prad_f)
    surf_fac = jnp.where(
        (tau_0 > 0.0) & (tau_0 < 2.0 / 3.0),
        1.0 + 2.0 / (3.0 * tau_0),
        1.0,
    )
    gradr = jnp.where(
        jnp.arange(dq.shape[0]) == 0, gradr_base * surf_fac, gradr_base
    )
    gradl = grada_f + jnp.where(use_ledoux_criterion, brunt_b, 0.0)

    _, gradt, _, vc, mlt_d = turb.set_mlt_jax(
        chi_t=chi_t_f,
        chi_rho=chi_rho_f,
        cp=cp_f,
        grav=grav,
        lambda_=lam,
        rho=rho_f,
        p=p_f,
        t=t_f,
        opacity=kap_f,
        gradr=gradr,
        grada=grada_f,
        gradl=gradl,
        mixing_length_alpha=mixing_length_alpha,
    )
    area_rho = _FOUR_PI * r * r * rho_f
    mlt_cdc = (area_rho * area_rho) * mlt_d

    brunt_n2 = jnp.where(
        jnp.arange(dq.shape[0]) > 0,
        (grav * grav * rho_f / p_f)
        * (chi_t_f / chi_rho_f)
        * (grada_f + brunt_b - gradt),
        0.0,
    )

    return {
        "scale_height": hp,
        "gradr": gradr,
        "gradl": gradl,
        "gradt": gradt,
        "mlt_vc": vc,
        "mlt_d": mlt_d,
        "mlt_cdc": mlt_cdc,
        "brunt_b": brunt_b,
        "brunt_n2": brunt_n2,
    }
