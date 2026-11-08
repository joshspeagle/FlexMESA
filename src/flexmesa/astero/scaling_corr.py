# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-dispatched (`auto` | `rust` | `jax`) asteroseismic routines.

Exposes scaling relations, acoustic profiles, g-mode period spacings, glitch
depths, surface corrections (`kjeldsen`, `cubic`, `combined`, `power_law`,
`sonoi`), and 5-point frequency separation ratios ($r_{01}, r_{10}$).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust
from flexmesa.astero import jax_kernels


def _is_jax(x: Any, backend: Any) -> bool:
    """Returns `True` when `resolve_backend(backend, x)` selects JAX."""
    return _dispatch.resolve_backend(backend, x) == "jax"


def eval_acoustic_profile(
    radius_cm: Any,
    csound: Any,
    r_center: float = 0.0,
    r_phot: float | None = None,
    *,
    backend: str = "auto",
) -> dict[str, Any]:
    """Evaluates cumulative acoustic radius, depth, and $\\tau_0$ [`s`].

    Args:
        radius_cm: Outer cell radii (`(N,)`) [`cm`].
        csound: Adiabatic sound speed (`(N,)`) [`cm s^-1`].
        r_center: Inner core boundary radius $R_{\\mathrm{center}}$ [`cm`].
        r_phot: Photospheric radius $R_{\\mathrm{phot}}$ [`cm`] (defaults to
            `radius_cm[0]`).
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Dictionary with keys `acoustic_radius`, `acoustic_depth`, and
        `photosphere_acoustic_r` [`s`].
    """
    rp = float(radius_cm[0]) if r_phot is None else float(r_phot)
    if _is_jax(radius_cm, backend):
        tau_r, tau_d, tau_0 = jax_kernels.eval_acoustic_profile_jax(
            radius_cm, csound, jnp.float64(r_center), jnp.float64(rp)
        )
        return {
            "acoustic_radius": tau_r,
            "acoustic_depth": tau_d,
            "photosphere_acoustic_r": tau_0,
        }
    return _rust.astero_eval_acoustic_profile(
        _dispatch.as_c_f64(radius_cm),
        _dispatch.as_c_f64(csound),
        float(r_center),
        rp,
    )


def eval_delta_nu_asymptotic(
    photosphere_acoustic_r: Any,
    *,
    backend: str = "auto",
) -> Any:
    """Evaluates asymptotic $\\Delta\\nu = 10^6 / (2 \\tau_0)$ [`uHz`].

    Args:
        photosphere_acoustic_r: Photospheric acoustic radius $\\tau_0$ [`s`].
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Asymptotic large frequency separation $\\Delta\\nu$ [`uHz`].
    """
    if _is_jax(photosphere_acoustic_r, backend):
        return jax_kernels.eval_delta_nu_asymptotic_jax(photosphere_acoustic_r)
    return _rust.astero_eval_delta_nu_asymptotic(float(photosphere_acoustic_r))


def eval_delta_nu_scaling(
    m_g: Any,
    teff_k: Any,
    l_phot_erg_s: Any,
    *,
    backend: str = "auto",
) -> Any:
    """Evaluates solar-scaled $\\Delta\\nu(M, T_{\\mathrm{eff}}, L)$ [`uHz`].

    Args:
        m_g: Stellar mass $M$ [`g`].
        teff_k: Effective temperature $T_{\\mathrm{eff}}$ [`K`].
        l_phot_erg_s: Photospheric luminosity $L_{\\mathrm{phot}}$ [`erg s^-1`].
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Solar-scaled large frequency separation $\\Delta\\nu$ [`uHz`].
    """
    if _is_jax(m_g, backend):
        return jax_kernels.eval_delta_nu_scaling_jax(m_g, teff_k, l_phot_erg_s)
    return _rust.astero_eval_delta_nu_scaling(
        float(m_g), float(teff_k), float(l_phot_erg_s)
    )


def eval_delta_nu_radius_scaling(
    m_g: Any,
    r_phot_cm: Any,
    *,
    backend: str = "auto",
) -> Any:
    """Evaluates solar-scaled $\\Delta\\nu(M, R_{\\mathrm{phot}})$ [`uHz`].

    Args:
        m_g: Stellar mass $M$ [`g`].
        r_phot_cm: Photospheric radius $R_{\\mathrm{phot}}$ [`cm`].
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Solar-scaled large frequency separation $\\Delta\\nu$ [`uHz`].
    """
    if _is_jax(m_g, backend):
        return jax_kernels.eval_delta_nu_radius_scaling_jax(m_g, r_phot_cm)
    return _rust.astero_eval_delta_nu_radius_scaling(
        float(m_g), float(r_phot_cm)
    )


def eval_nu_max(
    m_g: Any,
    r_phot_cm: Any,
    teff_k: Any,
    *,
    backend: str = "auto",
) -> Any:
    """Evaluates solar-scaled $\\nu_{\\max}$ [`uHz`].

    Args:
        m_g: Stellar mass $M$ [`g`].
        r_phot_cm: Photospheric radius $R_{\\mathrm{phot}}$ [`cm`].
        teff_k: Effective temperature $T_{\\mathrm{eff}}$ [`K`].
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Frequency of maximum oscillation power $\\nu_{\\max}$ [`uHz`].
    """
    if _is_jax(m_g, backend):
        return jax_kernels.eval_nu_max_jax(m_g, r_phot_cm, teff_k)
    return _rust.astero_eval_nu_max(float(m_g), float(r_phot_cm), float(teff_k))


def eval_nu_ac(
    g_surf_cgs: Any,
    gamma1_surf: Any,
    rho_surf: Any,
    peos_surf: Any,
    *,
    backend: str = "auto",
) -> Any:
    """Evaluates surface acoustic cutoff $\\nu_{\\mathrm{ac}}$ [`uHz`].

    Args:
        g_surf_cgs: Surface gravity $g_1$ [`cm s^-2`].
        gamma1_surf: Surface adiabatic exponent $\\Gamma_{1,1}$ (dimensionless).
        rho_surf: Surface mass density $\\rho_1$ [`g cm^-3`].
        peos_surf: Surface total pressure $P_1$ [`dyn cm^-2`].
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Surface acoustic cutoff frequency $\\nu_{\\mathrm{ac}}$ [`uHz`].
    """
    if _is_jax(g_surf_cgs, backend):
        return jax_kernels.eval_nu_ac_jax(
            g_surf_cgs, gamma1_surf, rho_surf, peos_surf
        )
    return _rust.astero_eval_nu_ac(
        float(g_surf_cgs),
        float(gamma1_surf),
        float(rho_surf),
        float(peos_surf),
    )


def eval_delta_pg(
    radius_cm: Any,
    r_mid: Any,
    brunt_n2: Any,
    csound_face: Any | None = None,
    mode_freq_uhz: float = 100.0,
    *,
    bildsten2012: bool = False,
    backend: str = "auto",
) -> Any:
    """Evaluates asymptotic dipole ($l=1$) g-mode spacing $\\Delta\\Pi_1$ [`s`].

    Args:
        radius_cm: Outer cell radii (`(N,)`) [`cm`].
        r_mid: Cell-centered radii (`(N,)`) [`cm`].
        brunt_n2: Squared Brunt-Vaisala frequency $N^2$ (`(N,)`) [`s^-2`].
        csound_face: Face sound speeds (`(N,)`) [`cm s^-1`] (used if
            `bildsten2012`).
        mode_freq_uhz: Reference oscillation frequency [`uHz`].
        bildsten2012: Whether to apply Bildsten et al. (2012) WKB cavity bounds.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Asymptotic dipole period spacing $\\Delta\\Pi_1$ [`s`].
    """
    if not bildsten2012 and _is_jax(radius_cm, backend):
        return jax_kernels.eval_delta_pg_traditional_jax(
            radius_cm, r_mid, brunt_n2
        )
    r_np = _dispatch.as_c_f64(radius_cm)
    rm_np = _dispatch.as_c_f64(r_mid)
    n2_np = _dispatch.as_c_f64(brunt_n2)
    cs_np = (
        np.ones_like(r_np) * 1.0e7
        if csound_face is None
        else _dispatch.as_c_f64(csound_face)
    )
    return _rust.astero_eval_delta_pg(
        r_np, rm_np, n2_np, cs_np, float(mode_freq_uhz), bool(bildsten2012)
    )


def eval_glitch_acoustic_depths(
    radius_cm: Any,
    csound: Any,
    gradr: Any,
    grada: Any,
    gamma1: Any,
    temperature_k: Any,
    r_center: float = 0.0,
) -> tuple[float, float, float]:
    """Locates BCZ and He II glitch acoustic depths (`s`).

    Args:
        radius_cm: Outer cell radii (`(N,)`) [`cm`].
        csound: Adiabatic sound speed (`(N,)`) [`cm s^-1`].
        gradr: Radiative temperature gradient (`(N,)`).
        grada: Adiabatic temperature gradient (`(N,)`).
        gamma1: First adiabatic exponent $\\Gamma_1$ (`(N,)`).
        temperature_k: Zone temperatures (`(N,)`) [`K`].
        r_center: Inner core boundary radius [`cm`].

    Returns:
        Tuple `(r_bcz_rsun, tau_bcz_s, tau_heii_s)`.
    """
    return _rust.astero_eval_glitch_acoustic_depths(
        _dispatch.as_c_f64(radius_cm),
        _dispatch.as_c_f64(csound),
        _dispatch.as_c_f64(gradr),
        _dispatch.as_c_f64(grada),
        _dispatch.as_c_f64(gamma1),
        _dispatch.as_c_f64(temperature_k),
        float(r_center),
    )


def eval_radial_delta_nu_ols(
    n_orders: Any,
    model_freqs: Any,
) -> float:
    """Evaluates unweighted OLS radial $\\Delta\\nu_{\\mathrm{model}}$ [`uHz`].

    Args:
        n_orders: Radial mode orders (`(M_0,)`).
        model_freqs: Model radial frequencies (`(M_0,)`) [`uHz`].

    Returns:
        Radial large separation $\\Delta\\nu_{\\mathrm{model}}$ [`uHz`].
    """
    return _rust.astero_eval_radial_delta_nu_ols(
        _dispatch.as_c_f64(n_orders),
        _dispatch.as_c_f64(model_freqs),
    )


def eval_surface_correction(
    scheme: str,
    obs_freqs: Any,
    obs_sigmas: Any,
    model_freqs: Any,
    model_inertias: Any,
    l0_obs_freqs: Any | None = None,
    l0_model_freqs: Any | None = None,
    l0_inertias: Any | None = None,
    l0_orders: Any | None = None,
    delta_nu_obs: float = 135.0,
    nu_max: float = 3078.0,
    b_power: float = 4.90,
    correction_factor: float = 1.0,
    *,
    backend: str = "auto",
) -> dict[str, Any]:
    """Evaluates asteroseismic surface-effect frequency corrections.

    Args:
        scheme: Correction scheme (`"kjeldsen"`, `"cubic"`, `"combined"`,
            `"power_law"`, or `"sonoi"`).
        obs_freqs: Observed frequencies (`(M,)`) [`uHz`].
        obs_sigmas: Observed uncertainties (`(M,)`) [`uHz`].
        model_freqs: Uncorrected model frequencies (`(M,)`) [`uHz`].
        model_inertias: Mode inertias (`(M,)`, dimensionless).
        l0_obs_freqs: Radial ($l=0$) observed frequencies (`(M_0,)`) [`uHz`].
        l0_model_freqs: Radial ($l=0$) model frequencies (`(M_0,)`) [`uHz`].
        l0_inertias: Radial ($l=0$) mode inertias (`(M_0,)`, dimensionless).
        l0_orders: Radial ($l=0$) mode orders (`(M_0,)`, dimensionless).
        delta_nu_obs: Observed radial $\\Delta\\nu$ [`uHz`].
        nu_max: Reference $\\nu_{\\max}$ [`uHz`].
        b_power: Power-law exponent $b$ for `"kjeldsen"`.
        correction_factor: Multiplicative scaling factor (default `1.0`).
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Dictionary with keys `nu_corrected`, `p0`, and `p1`.
    """
    if _is_jax(model_freqs, backend) and scheme in (
        "cubic",
        "bg14_1",
        "combined",
        "bg14_2",
    ):
        if scheme in ("cubic", "bg14_1"):
            corr, a3 = jax_kernels.eval_cubic_corr_jax(
                obs_freqs,
                obs_sigmas,
                model_freqs,
                model_inertias,
                jnp.float64(correction_factor),
            )
            return {"nu_corrected": corr, "p0": a3, "p1": jnp.float64(0.0)}
        corr, a3, a_m1 = jax_kernels.eval_combined_corr_jax(
            obs_freqs,
            obs_sigmas,
            model_freqs,
            model_inertias,
            jnp.float64(correction_factor),
        )
        return {"nu_corrected": corr, "p0": a3, "p1": a_m1}

    obs_f = _dispatch.as_c_f64(obs_freqs)
    obs_s = _dispatch.as_c_f64(obs_sigmas)
    mod_f = _dispatch.as_c_f64(model_freqs)
    mod_i = _dispatch.as_c_f64(model_inertias)
    l0_of = obs_f if l0_obs_freqs is None else _dispatch.as_c_f64(l0_obs_freqs)
    l0_mf = (
        mod_f if l0_model_freqs is None else _dispatch.as_c_f64(l0_model_freqs)
    )
    l0_mi = mod_i if l0_inertias is None else _dispatch.as_c_f64(l0_inertias)
    l0_or = (
        np.arange(1, l0_mf.size + 1, dtype=np.float64)
        if l0_orders is None
        else _dispatch.as_c_f64(l0_orders)
    )
    return _rust.astero_eval_surface_correction(
        scheme,
        obs_f,
        obs_s,
        mod_f,
        mod_i,
        l0_of,
        l0_mf,
        l0_mi,
        l0_or,
        float(delta_nu_obs),
        float(nu_max),
        float(b_power),
        float(correction_factor),
    )


def eval_ratios_r01_r10(
    l0_freqs: Any,
    l0_sigs: Any,
    l1_freqs: Any,
    l1_sigs: Any,
    *,
    backend: str = "auto",
) -> dict[str, Any]:
    """Evaluates Roxburgh & Vorontsov (2003) 5-point ratios $r_{01}, r_{10}$.

    Args:
        l0_freqs: Radial ($l=0$) frequencies (`(L,)`) [`uHz`].
        l0_sigs: Radial ($l=0$) uncertainties (`(L,)`) [`uHz`].
        l1_freqs: Dipole ($l=1$) frequencies (`(L,)`) [`uHz`].
        l1_sigs: Dipole ($l=1$) uncertainties (`(L,)`) [`uHz`].
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Dictionary with keys `r01`, `sig_r01`, `r10`, and `sig_r10`.
    """
    if _is_jax(l0_freqs, backend):
        r01, s01, r10, s10 = jax_kernels.eval_ratios_r01_r10_jax(
            l0_freqs, l0_sigs, l1_freqs, l1_sigs
        )
        return {"r01": r01, "sig_r01": s01, "r10": r10, "sig_r10": s10}
    return _rust.astero_eval_ratios_r01_r10(
        _dispatch.as_c_f64(l0_freqs),
        _dispatch.as_c_f64(l0_sigs),
        _dispatch.as_c_f64(l1_freqs),
        _dispatch.as_c_f64(l1_sigs),
    )
