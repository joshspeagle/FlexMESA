# SPDX-License-Identifier: LGPL-3.0-or-later
"""Differentiable JAX (`float64`) kernels for Wave 5.4 `astero`.

Provides `@jax.jit` kernels for asteroseismic scaling relations, acoustic
profiles, g-mode period spacings, Ball & Gizon (2014) surface corrections,
Roxburgh & Vorontsov (2003) frequency separation ratios, and composite
asteroseismic + spectroscopic $\\chi^2$ evaluation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants

NU_MAX_SUN: float = 3078.0
DELTA_NU_SUN: float = 134.91
TEFF_SUN_ASTERO: float = 5772.0


@jax.jit
def eval_delta_nu_asymptotic_jax(
    photosphere_acoustic_r: jax.Array,
) -> jax.Array:
    """Evaluates asymptotic $\\Delta\\nu = 10^6 / (2 \\tau_0)$ [`uHz`].

    Args:
        photosphere_acoustic_r: Photospheric acoustic radius $\\tau_0$ [`s`].

    Returns:
        Asymptotic large frequency separation $\\Delta\\nu$ [`uHz`].
    """
    tau = jnp.maximum(
        jnp.asarray(photosphere_acoustic_r, dtype=jnp.float64), 1e-30
    )
    return 1.0e6 / (2.0 * tau)


@jax.jit
def eval_delta_nu_scaling_jax(
    m_g: jax.Array, teff_k: jax.Array, l_phot_erg_s: jax.Array
) -> jax.Array:
    """Evaluates solar-scaled $\\Delta\\nu(M, T_{\\mathrm{eff}}, L)$ [`uHz`].

    Args:
        m_g: Stellar mass $M$ [`g`].
        teff_k: Effective temperature $T_{\\mathrm{eff}}$ [`K`].
        l_phot_erg_s: Photospheric luminosity $L_{\\mathrm{phot}}$ [`erg s^-1`].

    Returns:
        Solar-scaled large frequency separation $\\Delta\\nu$ [`uHz`].
    """
    m = jnp.asarray(m_g, dtype=jnp.float64) / constants.MSUN
    teff = jnp.asarray(teff_k, dtype=jnp.float64) / TEFF_SUN_ASTERO
    lum = jnp.asarray(l_phot_erg_s, dtype=jnp.float64) / constants.LSUN
    return DELTA_NU_SUN * jnp.sqrt(m) * (teff**3.0) * (lum ** (-0.75))


@jax.jit
def eval_delta_nu_radius_scaling_jax(
    m_g: jax.Array, r_phot_cm: jax.Array
) -> jax.Array:
    """Evaluates solar-scaled $\\Delta\\nu(M, R_{\\mathrm{phot}})$ [`uHz`].

    Args:
        m_g: Stellar mass $M$ [`g`].
        r_phot_cm: Photospheric radius $R_{\\mathrm{phot}}$ [`cm`].

    Returns:
        Solar-scaled large frequency separation $\\Delta\\nu$ [`uHz`].
    """
    m = jnp.asarray(m_g, dtype=jnp.float64) / constants.MSUN
    r = jnp.asarray(r_phot_cm, dtype=jnp.float64) / constants.RSUN
    return DELTA_NU_SUN * jnp.sqrt(m) * (r ** (-1.5))


@jax.jit
def eval_nu_max_jax(
    m_g: jax.Array, r_phot_cm: jax.Array, teff_k: jax.Array
) -> jax.Array:
    """Evaluates solar-scaled $\\nu_{\\max}$ [`uHz`].

    Args:
        m_g: Stellar mass $M$ [`g`].
        r_phot_cm: Photospheric radius $R_{\\mathrm{phot}}$ [`cm`].
        teff_k: Effective temperature $T_{\\mathrm{eff}}$ [`K`].

    Returns:
        Frequency of maximum oscillation power $\\nu_{\\max}$ [`uHz`].
    """
    m = jnp.asarray(m_g, dtype=jnp.float64) / constants.MSUN
    r = jnp.asarray(r_phot_cm, dtype=jnp.float64) / constants.RSUN
    teff = jnp.asarray(teff_k, dtype=jnp.float64) / TEFF_SUN_ASTERO
    return NU_MAX_SUN * m / (r * r * jnp.sqrt(teff))


@jax.jit
def eval_nu_ac_jax(
    g_surf_cgs: jax.Array,
    gamma1_surf: jax.Array,
    rho_surf: jax.Array,
    peos_surf: jax.Array,
) -> jax.Array:
    """Evaluates surface acoustic cutoff $\\nu_{\\mathrm{ac}}$ [`uHz`].

    Args:
        g_surf_cgs: Surface gravity $g_1$ [`cm s^-2`].
        gamma1_surf: Surface adiabatic exponent $\\Gamma_{1,1}$ (dimensionless).
        rho_surf: Surface mass density $\\rho_1$ [`g cm^-3`].
        peos_surf: Surface total pressure $P_1$ [`dyn cm^-2`].

    Returns:
        Surface acoustic cutoff frequency $\\nu_{\\mathrm{ac}}$ [`uHz`].
    """
    g = jnp.asarray(g_surf_cgs, dtype=jnp.float64)
    gam1 = jnp.asarray(gamma1_surf, dtype=jnp.float64)
    rho = jnp.asarray(rho_surf, dtype=jnp.float64)
    p = jnp.maximum(jnp.asarray(peos_surf, dtype=jnp.float64), 1.0e-30)
    return (0.25e6 / constants.PI) * g * jnp.sqrt(gam1 * rho / p)


@jax.jit
def eval_acoustic_profile_jax(
    radius_cm: jax.Array,
    csound: jax.Array,
    r_center: jax.Array,
    r_phot: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluates cumulative acoustic radius, depth, and $\\tau_0$ [`s`].

    Args:
        radius_cm: Outer cell radii (`(N,)`), surface-to-center [`cm`].
        csound: Adiabatic sound speed (`(N,)`) [`cm s^-1`].
        r_center: Inner core boundary radius $R_{\\mathrm{center}}$ [`cm`].
        r_phot: Photospheric radius $R_{\\mathrm{phot}}$ [`cm`].

    Returns:
        Tuple `(acoustic_radius, acoustic_depth, photosphere_acoustic_r)` [`s`].
    """
    r_arr = jnp.asarray(radius_cm, dtype=jnp.float64)
    cs_arr = jnp.maximum(jnp.asarray(csound, dtype=jnp.float64), 1.0)
    rc = jnp.asarray(r_center, dtype=jnp.float64)
    rp = jnp.asarray(r_phot, dtype=jnp.float64)
    r_next = jnp.concatenate([r_arr[1:], jnp.expand_dims(rc, 0)])
    dt = jnp.maximum(r_arr - r_next, 0.0) / cs_arr
    tau_r = jnp.cumsum(dt[::-1])[::-1]
    tau_d = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(dt)[:-1]]
    )
    k_phot, nz = jnp.sum(r_arr >= rp), r_arr.shape[0]
    k_clamped = jnp.clip(k_phot, 1, jnp.maximum(nz - 1, 1))
    tau_interior = jnp.where(
        k_phot >= nz,
        jnp.maximum(rp - rc, 0.0) / cs_arr[-1],
        tau_r[k_clamped] + (rp - r_arr[k_clamped]) / cs_arr[k_clamped - 1],
    )
    tau_0 = jnp.where(
        rp >= r_arr[0], tau_r[0] + (rp - r_arr[0]) / cs_arr[0], tau_interior
    )
    return tau_r, tau_d, tau_0


@jax.jit
def eval_delta_pg_traditional_jax(
    radius_cm: jax.Array, r_mid: jax.Array, brunt_n2: jax.Array
) -> jax.Array:
    """Evaluates traditional asymptotic dipole ($l=1$) $\\Delta\\Pi_1$ [`s`].

    Args:
        radius_cm: Outer cell radii (`(N,)`) [`cm`].
        r_mid: Cell-centered radii (`(N,)`) [`cm`].
        brunt_n2: Squared Brunt-Vaisala frequency $N^2$ (`(N,)`) [`s^-2`].

    Returns:
        Asymptotic dipole g-mode period spacing $\\Delta\\Pi_1$ [`s`].
    """
    r_arr = jnp.asarray(radius_cm, dtype=jnp.float64)
    rm_arr = jnp.asarray(r_mid, dtype=jnp.float64)
    n2_arr = jnp.asarray(brunt_n2, dtype=jnp.float64)
    dr = jnp.maximum(rm_arr[:-1] - rm_arr[1:], 0.0)
    n2_pos = jnp.where(n2_arr[1:] > 0.0, n2_arr[1:], 0.0)
    integrand = jnp.where(
        n2_arr[1:] > 0.0,
        jnp.sqrt(n2_pos) * dr / jnp.maximum(r_arr[1:], 1.0),
        0.0,
    )
    integral = jnp.sum(integrand)
    num = jnp.sqrt(2.0) * constants.PI * constants.PI
    return jnp.where(integral > 0.0, num / jnp.maximum(integral, 1e-30), 0.0)


@jax.jit
def eval_cubic_corr_jax(
    obs_freqs: jax.Array,
    obs_sigmas: jax.Array,
    model_freqs: jax.Array,
    model_inertias: jax.Array,
    correction_factor: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluates Ball & Gizon (2014) 1-term cubic surface correction (`bg14_1`).

    Args:
        obs_freqs: Observed frequencies (`(M,)`) [`uHz`].
        obs_sigmas: Observed uncertainties (`(M,)`) [`uHz`].
        model_freqs: Uncorrected model frequencies (`(M,)`) [`uHz`].
        model_inertias: Mode inertias (`(M,)`, dimensionless).
        correction_factor: Surface correction multiplier (dimensionless).

    Returns:
        Tuple `(corrected_freqs, a3)`.
    """
    f_obs = jnp.asarray(obs_freqs, dtype=jnp.float64)
    sig = jnp.maximum(jnp.asarray(obs_sigmas, dtype=jnp.float64), 1.0e-30)
    f_mod = jnp.asarray(model_freqs, dtype=jnp.float64)
    ine = jnp.maximum(jnp.asarray(model_inertias, dtype=jnp.float64), 1.0e-30)
    cf = jnp.asarray(correction_factor, dtype=jnp.float64)
    x = (f_mod * f_mod * f_mod) / (ine * sig)
    y = (f_obs - f_mod) / sig
    xtx = jnp.sum(x * x)
    a3 = jnp.where(xtx > 0.0, jnp.sum(x * y) / jnp.maximum(xtx, 1.0e-60), 0.0)
    return f_mod + cf * a3 * (f_mod * f_mod * f_mod) / ine, a3


@jax.jit
def eval_combined_corr_jax(
    obs_freqs: jax.Array,
    obs_sigmas: jax.Array,
    model_freqs: jax.Array,
    model_inertias: jax.Array,
    correction_factor: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluates Ball & Gizon (2014) 2-term combined surface correction.

    Args:
        obs_freqs: Observed frequencies (`(M,)`) [`uHz`].
        obs_sigmas: Observed uncertainties (`(M,)`) [`uHz`].
        model_freqs: Uncorrected model frequencies (`(M,)`) [`uHz`].
        model_inertias: Mode inertias (`(M,)`, dimensionless).
        correction_factor: Surface correction multiplier (dimensionless).

    Returns:
        Tuple `(corrected_freqs, a3, a_minus1)`.
    """
    f_obs = jnp.asarray(obs_freqs, dtype=jnp.float64)
    sig = jnp.maximum(jnp.asarray(obs_sigmas, dtype=jnp.float64), 1.0e-30)
    f_mod = jnp.asarray(model_freqs, dtype=jnp.float64)
    ine = jnp.maximum(jnp.asarray(model_inertias, dtype=jnp.float64), 1.0e-30)
    cf = jnp.asarray(correction_factor, dtype=jnp.float64)
    x1 = 1.0 / (f_mod * ine * sig)
    x2 = (f_mod * f_mod * f_mod) / (ine * sig)
    y = (f_obs - f_mod) / sig
    s11, s12, s22 = jnp.sum(x1 * x1), jnp.sum(x1 * x2), jnp.sum(x2 * x2)
    r1, r2 = jnp.sum(x1 * y), jnp.sum(x2 * y)
    det = s11 * s22 - s12 * s12
    ok = jnp.abs(det) > 1.0e-60
    safe_det = jnp.where(ok, det, 1.0)
    a_m1 = jnp.where(ok, (s22 * r1 - s12 * r2) / safe_det, 0.0)
    a3 = jnp.where(ok, (s11 * r2 - s12 * r1) / safe_det, 0.0)
    corr = f_mod + cf * (a_m1 / f_mod + a3 * (f_mod * f_mod * f_mod)) / ine
    return corr, a3, a_m1


@jax.jit
def eval_ratios_r01_r10_jax(
    l0_freqs: jax.Array,
    l0_sigs: jax.Array,
    l1_freqs: jax.Array,
    l1_sigs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluates Roxburgh & Vorontsov (2003) 5-point ratios $r_{01}, r_{10}$.

    Args:
        l0_freqs: Radial ($l=0$) frequencies (`(L,)`) [`uHz`].
        l0_sigs: Radial ($l=0$) uncertainties (`(L,)`) [`uHz`].
        l1_freqs: Dipole ($l=1$) frequencies (`(L,)`) [`uHz`].
        l1_sigs: Dipole ($l=1$) uncertainties (`(L,)`) [`uHz`].

    Returns:
        Tuple `(r01, sig_r01, r10, sig_r10)` of shape `(L - 2,)`.
    """
    f0 = jnp.asarray(l0_freqs, dtype=jnp.float64)
    s0 = jnp.asarray(l0_sigs, dtype=jnp.float64)
    f1 = jnp.asarray(l1_freqs, dtype=jnp.float64)
    s1 = jnp.asarray(l1_sigs, dtype=jnp.float64)
    d01 = (
        f0[:-2] - 4.0 * f1[:-2] + 6.0 * f0[1:-1] - 4.0 * f1[1:-1] + f0[2:]
    ) / 8.0
    dnu1 = jnp.maximum(f1[1:-1] - f1[:-2], 1.0e-30)
    r01 = d01 / dnu1
    s_d01_sq = (
        s0[:-2] ** 2
        + 16.0 * (s1[:-2] ** 2 + s1[1:-1] ** 2)
        + 36.0 * s0[1:-1] ** 2
        + s0[2:] ** 2
    ) / 64.0
    sig_r01 = jnp.sqrt(
        s_d01_sq / (dnu1 * dnu1)
        + (s1[1:-1] ** 2 + s1[:-2] ** 2) * (d01 * d01) / dnu1**4
    )

    d10 = (
        -(f1[:-2] - 4.0 * f0[1:-1] + 6.0 * f1[1:-1] - 4.0 * f0[2:] + f1[2:])
        / 8.0
    )
    dnu0 = jnp.maximum(f0[2:] - f0[1:-1], 1.0e-30)
    r10 = d10 / dnu0
    s_d10_sq = (
        s1[:-2] ** 2
        + 16.0 * (s0[1:-1] ** 2 + s0[2:] ** 2)
        + 36.0 * s1[1:-1] ** 2
        + s1[2:] ** 2
    ) / 64.0
    sig_r10 = jnp.sqrt(
        s_d10_sq / (dnu0 * dnu0)
        + (s0[2:] ** 2 + s0[1:-1] ** 2) * (d10 * d10) / dnu0**4
    )
    return r01, sig_r01, r10, sig_r10


@jax.jit
def eval_ratios_r02_jax(
    l0_freqs: jax.Array,
    l0_sigs: jax.Array,
    l1_freqs: jax.Array,
    l1_sigs: jax.Array,
    l2_freqs: jax.Array,
    l2_sigs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluates quadrupole-to-radial frequency separation ratios $r_{02}(n)$.

    Args:
        l0_freqs: Radial ($l=0$) frequencies (`(L,)`) [`uHz`].
        l0_sigs: Radial ($l=0$) uncertainties (`(L,)`) [`uHz`].
        l1_freqs: Dipole ($l=1$) frequencies (`(L,)`) [`uHz`].
        l1_sigs: Dipole ($l=1$) uncertainties (`(L,)`) [`uHz`].
        l2_freqs: Quadrupole ($l=2$) frequencies (`(L,)`) [`uHz`].
        l2_sigs: Quadrupole ($l=2$) uncertainties (`(L,)`) [`uHz`].

    Returns:
        Tuple `(r02, sig_r02)` of shape `(L - 1,)`.
    """
    f0 = jnp.asarray(l0_freqs, dtype=jnp.float64)
    s0 = jnp.asarray(l0_sigs, dtype=jnp.float64)
    f1 = jnp.asarray(l1_freqs, dtype=jnp.float64)
    s1 = jnp.asarray(l1_sigs, dtype=jnp.float64)
    f2 = jnp.asarray(l2_freqs, dtype=jnp.float64)
    s2 = jnp.asarray(l2_sigs, dtype=jnp.float64)
    d02 = f0[1:] - f2[1:]
    dnu1 = jnp.maximum(f1[1:] - f1[:-1], 1.0e-30)
    r02 = d02 / dnu1
    s_num = s0[1:] ** 2 + s2[1:] ** 2
    s_den = s1[1:] ** 2 + s1[:-1] ** 2
    sig_r02 = jnp.sqrt(s_num / (dnu1 * dnu1) + s_den * (d02 * d02) / dnu1**4)
    return r02, sig_r02


def _mean_chi2_jax(
    obs: jax.Array, sigmas: jax.Array, model: jax.Array
) -> jax.Array:
    """Computes reduced $\\chi^2$ safely returning `0.0` on empty arrays."""
    obs_a = jnp.asarray(obs, dtype=jnp.float64)
    if obs_a.size == 0:
        return jnp.float64(0.0)
    sig_a = jnp.maximum(jnp.asarray(sigmas, dtype=jnp.float64), 1.0e-30)
    mod_a = jnp.asarray(model, dtype=jnp.float64)
    return jnp.mean(((mod_a - obs_a) / sig_a) ** 2)


@jax.jit
def eval_astero_chi2_jax(
    obs_freqs: jax.Array,
    obs_sigmas: jax.Array,
    corr_model_freqs: jax.Array,
    spectro_obs: jax.Array,
    spectro_sigmas: jax.Array,
    spectro_model: jax.Array,
    chi2_seismo_fraction: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluates normalized composite asteroseismic and spectroscopic $\\chi^2$.

    Args:
        obs_freqs: Observed mode frequencies (`(M,)`) [`uHz`].
        obs_sigmas: Observed mode uncertainties (`(M,)`) [`uHz`].
        corr_model_freqs: Surface-corrected model frequencies (`(M,)`) [`uHz`].
        spectro_obs: Spectroscopic targets (`(S,)`).
        spectro_sigmas: Spectroscopic uncertainties (`(S,)`).
        spectro_model: Model spectroscopic observables (`(S,)`).
        chi2_seismo_fraction: Weight of $\\chi^2_{\\mathrm{seismo}}$ in
            $[0, 1]$.

    Returns:
        Tuple `(chi2_total, chi2_seismo, chi2_spectro)`.
    """
    chi2_seismo = _mean_chi2_jax(obs_freqs, obs_sigmas, corr_model_freqs)
    chi2_spectro = _mean_chi2_jax(spectro_obs, spectro_sigmas, spectro_model)
    f_seis = jnp.clip(
        jnp.asarray(chi2_seismo_fraction, dtype=jnp.float64), 0.0, 1.0
    )
    chi2_total = f_seis * chi2_seismo + (1.0 - f_seis) * chi2_spectro
    return chi2_total, chi2_seismo, chi2_spectro
