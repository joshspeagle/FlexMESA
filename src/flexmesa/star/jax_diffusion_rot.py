# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differentiable JAX kernels for Wave 5.1 stellar diffusion, shellular
rotation, Spruit-Tayler magnetic dynamo, and crystallization phase separation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

_PI = float(constants.PI)
_AMU = float(constants.AMU)
_CLIGHT = float(constants.CLIGHT)
_QE = float(constants.QE)
_KERG = float(constants.KERG)
_HBAR = float(constants.HBAR)
_CRAD = float(constants.CRAD)
_ME = float(constants.ME)
_RI_CRIT_DSI = 0.25


def _safe_abs(x: jax.Array, eps: float = 1.0e-30) -> jax.Array:
    """Computes ``|x|`` with a zero-safe subgradient for reverse-mode AD."""
    x_nz = jnp.where(x == 0.0, eps, x)
    return jnp.where(x == 0.0, 0.0, jnp.abs(x_nz))


def _eval_effective_kt_jax(n_e: jax.Array, kt: jax.Array) -> jax.Array:
    """Computes degenerate-corrected electron thermal energy ``k_B T_eff``."""
    kf = jnp.cbrt(3.0 * _PI * _PI * jnp.maximum(n_e, 1.0e-10))
    ef = jnp.square(_HBAR * kf) / (2.0 * _ME)
    return jnp.hypot(kt, (2.0 / 3.0) * ef)


@jax.jit
def eval_rotation_fp_ft_jax(
    w_div_w_crit: jax.Array,
    fp_min: float = 0.75,
    ft_min: float = 0.95,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluates Fabry et al. (2022) / Paxton et al. (2019) Roche rotational
    distortion polynomials ``(f_P, f_T, i_rot / r^2)``.

    Args:
        w_div_w_crit: Dimensionless angular velocity ``Omega / Omega_crit``,
            shape ``(*batch,)`` (dimensionless).
        fp_min: Floor on ``f_P`` (dimensionless).
        ft_min: Floor on ``f_T`` (dimensionless).

    Returns:
        Tuple ``(f_P, f_T, i_rot_over_r2)`` of shape ``(*batch,)``.
    """
    w = jnp.clip(_safe_abs(w_div_w_crit), 0.0, 0.9999)
    w2 = w * w
    w4 = w2 * w2
    w6 = w4 * w2
    one_m_w56 = jnp.maximum(1.0 - jnp.power(w + 1.0e-30, 5.626), 1.0e-12)
    a_w = jnp.maximum(
        1.0 + 0.3293 * w4 - 0.4926 * w6 - 0.5560 * jnp.log(one_m_w56),
        1.0,
    )
    fp_raw = (1.0 - (2.0 / 3.0) * w2 - 0.2133 * w4 - 0.1068 * w6) / a_w
    ft_raw = (1.0 - 0.07955 * w4 - 0.2322 * w6) / a_w
    fp = jnp.clip(fp_raw, fp_min, 1.0)
    ft = jnp.clip(ft_raw, ft_min, 1.0)
    i_rot_over_r2 = (
        (2.0 / 3.0) * (1.0 + 0.1325 * w2 + 0.0480 * w4) / (1.0 + 0.0450 * w2)
    )
    return fp, ft, i_rot_over_r2


@jax.jit
def eval_omega_crit_jax(
    cgrav: jax.Array,
    m_grav: jax.Array,
    r_polar: jax.Array,
    lum: jax.Array,
    opacity: jax.Array,
    max_gamma_edd: float = 0.99,
) -> jax.Array:
    """Evaluates the critical Roche angular velocity ``Omega_crit`` (rad/s).

    Args:
        cgrav: Local gravitational constant (cm^3 g^-1 s^-2).
        m_grav: Enclosed gravitational mass (g).
        r_polar: Polar/spherical radius (cm).
        lum: Radiative luminosity (erg/s).
        opacity: Rosseland mean opacity (cm^2/g).
        max_gamma_edd: Cap on Eddington factor ``Gamma_Edd``.

    Returns:
        Critical angular velocity ``Omega_crit`` in rad/s.
    """
    m_eff = jnp.maximum(m_grav, 1.0e20)
    r_eff = jnp.maximum(r_polar, 1.0e5)
    l_edd = 4.0 * _PI * _CLIGHT * cgrav * m_eff / jnp.maximum(opacity, 1.0e-10)
    gamma_edd = jnp.clip(
        jnp.maximum(lum, 0.0) / jnp.maximum(l_edd, 1.0),
        0.0,
        max_gamma_edd,
    )
    return jnp.sqrt(
        jnp.maximum(
            (1.0 - gamma_edd) * cgrav * m_eff / (r_eff * r_eff * r_eff),
            1.0e-30,
        )
    )


@jax.jit
def eval_rotational_mixing_zone_jax(
    r: jax.Array,
    hp: jax.Array,
    grav: jax.Array,
    rho: jax.Array,
    t: jax.Array,
    cp: jax.Array,
    opacity: jax.Array,
    chi_t: jax.Array,
    chi_rho: jax.Array,
    grada: jax.Array,
    gradt: jax.Array,
    brunt_n2_t: jax.Array,
    brunt_n2_mu: jax.Array,
    omega: jax.Array,
    dln_omega_dln_r: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Evaluates the five hydrodynamic rotational instabilities (DSI, SH, SSI,
    ES, GSF) and Spruit-Tayler (2002) magnetic dynamo in JAX.

    Args:
        r: Shell radius (cm).
        hp: Pressure scale height (cm).
        grav: Effective gravity (cm/s^2).
        rho: Mass density (g/cm^3).
        t: Temperature (K).
        cp: Isobaric specific heat (erg g^-1 K^-1).
        opacity: Rosseland mean opacity (cm^2/g).
        chi_t: ``(d ln P / d ln T)_rho`` (dimensionless).
        chi_rho: ``(d ln P / d ln rho)_T`` (dimensionless).
        grada: Adiabatic temperature gradient (dimensionless).
        gradt: Actual temperature gradient (dimensionless).
        brunt_n2_t: Thermal Brunt-Vaisala component ``N_T^2`` (s^-2).
        brunt_n2_mu: Composition Brunt-Vaisala component ``N_mu^2`` (s^-2).
        omega: Angular velocity ``Omega`` (rad/s).
        dln_omega_dln_r: Dimensionless shear ``d ln Omega / d ln r``.
        abar: Mean atomic mass number ``A_bar``.
        zbar: Mean ionic charge ``Z_bar``.

    Returns:
        Tuple ``(d_dsi, d_sh, d_ssi, d_es, d_gsf, d_st, nu_st, b_phi, b_r)``.
    """
    _ = (chi_t, chi_rho)
    om_abs = _safe_abs(omega)
    valid = (om_abs >= 1.0e-20) & (r >= 1.0e5) & (hp >= 1.0e3)
    om = jnp.maximum(om_abs, 1.0e-30)
    q_shear = _safe_abs(dln_omega_dln_r)
    q_safe = jnp.maximum(q_shear, 1.0e-30)

    denom_k = (
        3.0
        * jnp.maximum(opacity, 1.0e-10)
        * jnp.square(jnp.maximum(rho, 1.0e-20))
        * jnp.maximum(cp, 1.0)
    )
    k_therm = (
        4.0 * _CRAD * _CLIGHT * jnp.power(jnp.maximum(t, 10.0), 3) / denom_k
    )
    n2_tot = brunt_n2_t + brunt_n2_mu
    tau_dyn = jnp.maximum(
        jnp.sqrt(jnp.maximum(r, 1.0) / jnp.maximum(grav, 1.0)), 1.0e-6
    )

    shear_rate = om * q_shear
    ri = n2_tot / jnp.maximum(shear_rate * shear_rate, 1.0e-30)
    d_dsi = jnp.where(
        (shear_rate > 0.0) & (n2_tot >= 0.0) & (ri < _RI_CRIT_DSI),
        (hp * hp / tau_dyn) * jnp.square(1.0 - ri / _RI_CRIT_DSI),
        0.0,
    )

    kappa_epi2 = 4.0 * om * om * (1.0 + 0.5 * dln_omega_dln_r)
    sh_crit = n2_tot + kappa_epi2
    sh_neg = jnp.where(sh_crit < 0.0, -sh_crit, 1.0e-30)
    d_sh = jnp.where(
        sh_crit < 0.0,
        hp * hp * jnp.minimum(jnp.sqrt(sh_neg), 1.0 / tau_dyn),
        0.0,
    )

    n2_eff_ssi = jnp.maximum(brunt_n2_t / 6.0, 0.0) + jnp.maximum(
        brunt_n2_mu, 0.0
    )
    ri_sec = n2_eff_ssi / jnp.maximum(shear_rate * shear_rate, 1.0e-30)
    d_ssi = jnp.where(
        (shear_rate > 0.0) & (n2_eff_ssi > 0.0) & (ri_sec < _RI_CRIT_DSI),
        k_therm
        * (shear_rate * shear_rate / jnp.maximum(n2_eff_ssi, 1.0e-30))
        * _RI_CRIT_DSI,
        0.0,
    )

    grad_sub = jnp.maximum(_safe_abs(grada - gradt), 1.0e-4)
    f_cen = jnp.minimum(om * om * r / jnp.maximum(grav, 1.0), 0.9)
    d_es = jnp.maximum(k_therm / grad_sub * f_cen, 0.0)

    d_gsf = jnp.where(
        (q_shear > 1.0e-6) & (brunt_n2_mu <= 1.0e-6 * _safe_abs(brunt_n2_t)),
        k_therm
        * (om * om * r / jnp.maximum(grav, 1.0))
        * q_shear
        * jnp.minimum(r / jnp.maximum(hp, 1.0e3), 10.0),
        0.0,
    )

    kt = _KERG * jnp.maximum(t, 100.0)
    e2 = _QE * _QE
    z = jnp.maximum(zbar, 1.0)
    a = jnp.maximum(abar, 1.0)
    n_i = jnp.maximum(rho, 1.0e-15) / (a * _AMU)
    n_e = jnp.maximum(n_i * z, 1.0e-10)
    kt_eff = _eval_effective_kt_jax(n_e, kt)
    lam_d = jnp.sqrt(kt_eff / (4.0 * _PI * e2 * n_e * (1.0 + z)))
    x_lam = jnp.maximum(3.0 * kt_eff * lam_d / (z * e2), 1.5)
    ln_lam = jnp.maximum(jnp.log(x_lam), 0.5)
    t_eff = jnp.maximum(kt_eff / _KERG, 100.0)
    eta_m = jnp.clip(
        7.0e11 * z * ln_lam * jnp.power(t_eff, -1.5), 1.0e-6, 1.0e16
    )

    n_t2 = jnp.maximum(brunt_n2_t, 0.0)
    n_mu2 = jnp.maximum(brunt_n2_mu, 0.0)
    n_eff2 = n_mu2 + (eta_m / jnp.maximum(k_therm, eta_m)) * n_t2
    n_eff = jnp.sqrt(jnp.maximum(n_eff2, 1.0e-30))
    q_min = jnp.power(n_eff / om, 1.75) * jnp.power(
        eta_m / jnp.maximum(r * r * n_eff, 1.0e-30), 0.25
    )
    f_damp = jnp.clip(
        1.0 - jnp.sqrt(jnp.maximum(q_min, 1.0e-30) / q_safe), 0.0, 1.0
    )

    n_mu = jnp.sqrt(jnp.maximum(n_mu2, 1.0e-30))
    q_sqrt = jnp.sqrt(q_safe)
    b_phi_0 = (
        jnp.sqrt(4.0 * _PI * jnp.maximum(rho, 1.0e-30))
        * r
        * om
        * q_sqrt
        * (om / n_mu)
    )
    b_r_0 = b_phi_0 * (om / n_mu) * q_sqrt
    nu_st_0 = (r * r * om / q_safe) * jnp.power(om / n_mu, 4) * f_damp
    d_st_0 = (
        (r * r * om)
        * jnp.power(om / n_mu, 4)
        * jnp.power(jnp.maximum(eta_m / (r * r * om), 1.0e-12), 0.25)
        * f_damp
    )

    n_t = jnp.sqrt(jnp.maximum(n_t2, 1.0e-30))
    eps_k = jnp.maximum(eta_m / (r * r * n_t), 1.0e-20)
    b_phi_1 = (
        jnp.sqrt(4.0 * _PI * jnp.maximum(rho, 1.0e-30))
        * r
        * om
        * jnp.sqrt(om / n_t)
        * q_sqrt
        * jnp.power(jnp.maximum(k_therm / (r * r * n_t), 1.0e-20), 0.125)
    )
    b_r_1 = b_phi_1 * jnp.sqrt(om / n_t) * jnp.power(eps_k, 0.125)
    nu_st_1 = (
        r
        * r
        * om
        * q_sqrt
        * jnp.sqrt(om / n_t)
        * jnp.power(jnp.maximum(k_therm / (r * r * n_t), 1.0e-20), 0.25)
        * f_damp
    )
    d_st_1 = nu_st_1 * jnp.clip(
        jnp.sqrt(eta_m / jnp.maximum(k_therm, eta_m)), 1.0e-4, 1.0
    )

    use_case0 = (n_mu2 >= (eta_m / jnp.maximum(k_therm, eta_m)) * n_t2) & (
        n_mu2 > 0.0
    )
    active = (
        valid
        & (q_shear > q_min)
        & (q_shear > 1.0e-8)
        & (n_eff2 > 1.0e-20)
        & (om_abs > 1.0e-15)
    )
    d_st = jnp.where(active, jnp.where(use_case0, d_st_0, d_st_1), 0.0)
    nu_st = jnp.where(active, jnp.where(use_case0, nu_st_0, nu_st_1), 0.0)
    b_phi = jnp.where(active, jnp.where(use_case0, b_phi_0, b_phi_1), 0.0)
    b_r = jnp.where(active, jnp.where(use_case0, b_r_0, b_r_1), 0.0)

    return (
        jnp.where(valid, jnp.clip(d_dsi, 0.0, 1.0e18), 0.0),
        jnp.where(valid, jnp.clip(d_sh, 0.0, 1.0e18), 0.0),
        jnp.where(valid, jnp.clip(d_ssi, 0.0, 1.0e18), 0.0),
        jnp.where(valid, jnp.clip(d_es, 0.0, 1.0e18), 0.0),
        jnp.where(valid, jnp.clip(d_gsf, 0.0, 1.0e18), 0.0),
        jnp.where(valid, jnp.clip(d_st, 0.0, 1.0e18), 0.0),
        jnp.where(valid, jnp.clip(nu_st, 0.0, 1.0e18), 0.0),
        jnp.where(valid, jnp.clip(b_phi, 0.0, 1.0e12), 0.0),
        jnp.where(valid, jnp.clip(b_r, 0.0, 1.0e12), 0.0),
    )


@jax.jit
def eval_caplan_ocp_dstar_jax(
    gamma_i: jax.Array, kappa: jax.Array
) -> jax.Array:
    """Evaluates the Caplan, Bauer & Freeman (2022) strong-coupling OCP
    dimensionless self-diffusion coefficient ``D*(Gamma, kappa)``.

    Args:
        gamma_i: Coulomb coupling parameter ``Gamma_i`` (dimensionless).
        kappa: Screening parameter ``a_i / lambda_TF`` (dimensionless).

    Returns:
        Dimensionless self-diffusion coefficient ``D*``.
    """
    gam = jnp.maximum(gamma_i, 0.1)
    kap = jnp.maximum(kappa, 0.0)
    pref = jnp.sqrt(_PI / 3.0)
    a_k = pref * (1.22 + 0.405 * jnp.power(kap + 1.0e-30, 1.77))
    b_k = 0.562 * jnp.exp(-0.185 * jnp.power(kap + 1.0e-30, 1.31))
    c_k = 1.11 + 0.112 * jnp.power(kap + 1.0e-30, 1.32)
    term1 = pref * a_k * jnp.power(gam, -2.5)
    term2 = (b_k / gam) * jnp.exp(-c_k * jnp.cbrt(gam))
    return jnp.maximum(term1 + term2, 1.0e-12)


@jax.jit
def eval_burgers_velocities_jax(
    a: jax.Array,
    z: jax.Array,
    x_mass: jax.Array,
    rho: jax.Array,
    t: jax.Array,
    dlnt_dr: jax.Array,
    g_grav: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solves the full-rank ``(2m + 2) x (2m + 2)`` Thoul, Bahcall & Loeb (1994)
    multi-species Burgers system in JAX with Stanton & Murillo (2016) Yukawa
    collision integrals.

    Args:
        a: Atomic mass numbers ``A_i``, shape ``(nc,)``.
        z: Ionic charges ``Z_i``, shape ``(nc,)``.
        x_mass: Species mass fractions ``X_i``, shape ``(nc,)``.
        rho: Mass density (g/cm^3).
        t: Temperature (K).
        dlnt_dr: Temperature gradient ``d ln T / dr`` (cm^-1).
        g_grav: Gravitational acceleration (cm/s^2).

    Returns:
        Tuple ``(v_gt, d_self, ee_force)`` where ``v_gt`` and ``d_self`` have
        shape ``(nc,)``.
    """
    nc = a.shape[0]
    m = nc + 1
    kt = _KERG * jnp.maximum(t, 100.0)
    e2 = _QE * _QE

    ai = jnp.maximum(a, 1.0)
    zi = jnp.maximum(z, 0.1)
    xi = jnp.clip(x_mass, 1.0e-15, 1.0)
    ni = rho * xi / (ai * _AMU)
    n_ions = jnp.sum(ni)
    sum_n_z2 = jnp.sum(ni * zi * zi)
    n_e = jnp.maximum(jnp.sum(ni * zi), 1.0e-10)

    n_s = jnp.concatenate([ni, jnp.array([n_e])])
    m_s = jnp.concatenate([ai * _AMU, jnp.array([_ME])])
    z_s = jnp.concatenate([zi, jnp.array([-1.0])])

    kt_eff_e = _eval_effective_kt_jax(n_e, kt)
    inv_lam_e2 = 4.0 * _PI * e2 * n_e / kt_eff_e
    inv_lam_i2 = 4.0 * _PI * e2 * jnp.maximum(sum_n_z2, 1.0e-10) / kt
    lam_d = 1.0 / jnp.sqrt(inv_lam_e2 + inv_lam_i2)
    a_ion = jnp.cbrt(3.0 / (4.0 * _PI * jnp.maximum(n_ions, 1.0e-10)))
    lam_eff = jnp.maximum(lam_d, a_ion)

    z_prod = jnp.maximum(jnp.abs(z_s[:, None] * z_s[None, :]), 1.0e-6)
    g_mat = jnp.clip(z_prod * e2 / (lam_eff * kt), 1.0e-12, 1.0e4)
    k11_weak = -0.25 * (1.4660 + jnp.log(g_mat)) * (1.0 - 0.0462 * g_mat)
    k11_strong = 0.3804 / jnp.power(g_mat, 1.75) + 0.1656 / g_mat
    ln_lam = jnp.maximum(jnp.where(g_mat < 1.0, k11_weak, k11_strong), 0.02)

    w_g = 1.0 / (1.0 + 0.85 * jnp.power(g_mat, 0.65))
    z_mat = 0.32 + 0.28 * w_g
    zp_mat = 0.80 + 0.50 * w_g
    zpp_mat = 1.30 + 0.70 * w_g

    mu_mat = (m_s[:, None] * m_s[None, :]) / (m_s[:, None] + m_s[None, :])
    pref_k = (16.0 / 3.0) * jnp.sqrt(_PI / (2.0 * kt)) * e2 * e2 / kt
    off_mask = 1.0 - jnp.eye(m)
    k_mat = (
        pref_k
        * jnp.sqrt(mu_mat)
        * (n_s[:, None] * n_s[None, :])
        * jnp.square(z_s[:, None] * z_s[None, :])
        * ln_lam
        * off_mask
    )

    n_sys = 2 * m + 2
    lhs = jnp.zeros((n_sys, n_sys), dtype=jnp.float64)
    rhs = jnp.zeros((n_sys,), dtype=jnp.float64)

    mtot = m_s[:, None] + m_s[None, :]
    mi_f = m_s[:, None] / mtot
    mj_f = m_s[None, :] / mtot

    sum_k = jnp.sum(k_mat, axis=1)
    sum_kz = jnp.sum(k_mat * z_mat * mj_f, axis=1)
    mom_w = k_mat - jnp.diag(sum_k)
    mom_r = -k_mat * z_mat * mi_f + jnp.diag(sum_kz)
    lhs = lhs.at[:m, :m].set(mom_w)
    lhs = lhs.at[:m, m : 2 * m].set(mom_r)
    lhs = lhs.at[:m, 2 * m].set(n_s * z_s)
    lhs = lhs.at[:m, 2 * m + 1].set(n_s * m_s)

    n_tot = n_ions + n_e
    rho_tot = jnp.sum(n_s * m_s)
    rhs = rhs.at[:m].set(n_s * m_s * g_grav - (n_s / n_tot) * rho_tot * g_grav)

    w_coeff = 2.5 * k_mat * z_mat * mj_f / n_s[:, None]
    heat_w = -w_coeff + jnp.diag(jnp.sum(w_coeff, axis=1))
    diag_r_self = (
        -0.8
        * jnp.sqrt(2.0)
        * pref_k
        * jnp.sqrt(m_s)
        * n_s
        * n_s
        * jnp.power(z_s, 4)
        * 1.5
    )
    diag_r_cross = -jnp.sum(
        k_mat
        * (
            3.0 * mi_f * mi_f
            + mj_f * mj_f * zpp_mat
            + 0.8 * mi_f * mj_f * zp_mat
        )
        / n_s[:, None],
        axis=1,
    )
    heat_r_off = (
        k_mat * mi_f * mj_f * (3.0 + zpp_mat - 0.8 * zp_mat) / n_s[None, :]
    )
    heat_r = heat_r_off + jnp.diag(diag_r_self + diag_r_cross)
    lhs = lhs.at[m : 2 * m, :m].set(heat_w)
    lhs = lhs.at[m : 2 * m, m : 2 * m].set(heat_r)
    rhs = rhs.at[m : 2 * m].set(2.5 * kt * dlnt_dr)

    lhs = lhs.at[2 * m, :m].set(n_s * z_s)
    lhs = lhs.at[2 * m + 1, :m].set(n_s * m_s)

    sol = jnp.linalg.solve(lhs, rhs)
    w_ion = sol[:nc]
    rho_ion = ni * ai * _AMU
    v_cm = jnp.sum(rho_ion * w_ion) / jnp.maximum(jnp.sum(rho_ion), 1.0e-30)
    v_gt = jnp.clip(w_ion - v_cm, -1.0e2, 1.0e2)
    k_tot_ion = jnp.sum(k_mat[:nc, :], axis=1)
    d_self = jnp.clip(
        ni * kt / jnp.maximum(k_tot_ion, 1.0e-30), 1.0e-10, 1.0e12
    )
    return v_gt, d_self, sol[2 * m]


@jax.jit
def eval_phase_separation_shift_jax(
    x_heavy: jax.Array,
    is_one_mixture: bool = False,
) -> jax.Array:
    """Evaluates the Blouin et al. (2020, 2021) 5th-order polynomial
    solid-liquid distillation shift ``Delta x_heavy`` in JAX.

    Args:
        x_heavy: Mass fraction of the heavier ion species (O16 or Ne20).
        is_one_mixture: ``False`` for C/O mixture, ``True`` for O/Ne mixture.

    Returns:
        Solid-minus-liquid mass-fraction enrichment ``Delta x_heavy``.
    """
    x = jnp.clip(x_heavy, 0.0, 1.0)
    ratio_a = jnp.where(is_one_mixture, 16.0 / 20.0, 12.0 / 16.0)
    xi = (x * ratio_a) / jnp.maximum(1.0 - x * (1.0 - ratio_a), 1.0e-12)
    c_co = jnp.array([0.642856, -1.954689, 2.487018, -1.789000, 0.613815])
    c_one = jnp.array([0.435920, -1.242958, 1.711583, -1.366634, 0.462089])
    coeffs = jnp.where(is_one_mixture, c_one, c_co)
    pows = jnp.stack([xi, xi**2, xi**3, xi**4, xi**5], axis=-1)
    delta_xi = jnp.sum(pows * coeffs, axis=-1)
    xi_solid = jnp.clip(xi + delta_xi, 0.0, 1.0)
    x_solid = xi_solid / jnp.maximum(
        ratio_a + xi_solid * (1.0 - ratio_a), 1.0e-12
    )
    return jnp.maximum(x_solid - x, 0.0)
