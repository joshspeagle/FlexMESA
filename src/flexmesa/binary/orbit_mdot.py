# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-dispatched orbital mechanics, Roche-lobe overflow, $L_2$ mass loss,
and binary timestep functions (`flexmesa.binary`).
"""

from __future__ import annotations

import typing

import jax
import numpy.typing as npt

from flexmesa import _dispatch, _rust
from flexmesa.binary import jax_kernels


def eval_rlobe(
    m1: float | jax.Array,
    m2: float | jax.Array,
    a: float | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Evaluate Eggleton (1983) Roche-lobe radius $R_{L,1}$ (cm).

    Args:
        m1: Primary mass (g).
        m2: Companion mass (g).
        a: Orbital separation (cm).
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Roche-lobe radius $R_{L,1}$ (cm).
    """
    resolved = _dispatch.resolve_backend(backend, m1, m2, a)
    if resolved == "jax":
        return typing.cast(jax.Array, jax_kernels.eval_rlobe_jax(m1, m2, a))
    return _rust.binary_eval_rlobe(float(m1), float(m2), float(a))


def eval_rl_relative_gap(
    r: float | jax.Array,
    rl: float | jax.Array,
    eccentricity: float | jax.Array = 0.0,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Evaluate periastron relative Roche-lobe gap $(R - R_{L,p}) / R_{L,p}$.

    Args:
        r: Stellar radius (cm).
        rl: Circular Roche-lobe radius (cm).
        eccentricity: Orbital eccentricity $e$.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Dimensionless relative Roche-lobe gap.
    """
    resolved = _dispatch.resolve_backend(backend, r, rl, eccentricity)
    if resolved == "jax":
        return typing.cast(
            jax.Array,
            jax_kernels.eval_rl_relative_gap_jax(r, rl, eccentricity),
        )
    return _rust.binary_eval_rl_relative_gap(
        float(r), float(rl), float(eccentricity)
    )


def separation_from_period(
    m1: float | jax.Array,
    m2: float | jax.Array,
    period: float | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Compute orbital separation $a$ (cm) from period $P$ (s) and masses (g).

    Args:
        m1: Primary mass (g).
        m2: Companion mass (g).
        period: Orbital period (s).
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Orbital separation $a$ (cm).
    """
    resolved = _dispatch.resolve_backend(backend, m1, m2, period)
    if resolved == "jax":
        return typing.cast(
            jax.Array, jax_kernels.separation_from_period_jax(m1, m2, period)
        )
    return _rust.binary_separation_from_period(
        float(m1), float(m2), float(period)
    )


def period_from_separation(
    m1: float | jax.Array,
    m2: float | jax.Array,
    separation: float | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Compute orbital period $P$ (s) from separation $a$ (cm) and masses (g).

    Args:
        m1: Primary mass (g).
        m2: Companion mass (g).
        separation: Orbital separation (cm).
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Orbital period $P$ (s).
    """
    resolved = _dispatch.resolve_backend(backend, m1, m2, separation)
    if resolved == "jax":
        return typing.cast(
            jax.Array,
            jax_kernels.period_from_separation_jax(m1, m2, separation),
        )
    return _rust.binary_period_from_separation(
        float(m1), float(m2), float(separation)
    )


def eval_j_orb(
    m1: float | jax.Array,
    m2: float | jax.Array,
    separation: float | jax.Array,
    eccentricity: float | jax.Array = 0.0,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Compute orbital angular momentum $J_{\\mathrm{orb}}$ in CGS units.

    Args:
        m1: Primary mass (g).
        m2: Companion mass (g).
        separation: Orbital separation (cm).
        eccentricity: Orbital eccentricity $e$.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Orbital angular momentum $J_{\\mathrm{orb}}$ (`g cm^2 s^-1`).
    """
    resolved = _dispatch.resolve_backend(
        backend, m1, m2, separation, eccentricity
    )
    if resolved == "jax":
        return typing.cast(
            jax.Array,
            jax_kernels.eval_j_orb_jax(m1, m2, separation, eccentricity),
        )
    return _rust.binary_eval_j_orb(
        float(m1), float(m2), float(separation), float(eccentricity)
    )


def separation_from_j_orb(
    m1: float | jax.Array,
    m2: float | jax.Array,
    j_orb: float | jax.Array,
    eccentricity: float | jax.Array = 0.0,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Compute orbital separation $a$ (cm) from $J_{\\mathrm{orb}}$.

    Args:
        m1: Primary mass (g).
        m2: Companion mass (g).
        j_orb: Orbital angular momentum (`g cm^2 s^-1`).
        eccentricity: Orbital eccentricity $e$.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Orbital separation $a$ (cm).
    """
    resolved = _dispatch.resolve_backend(backend, m1, m2, j_orb, eccentricity)
    if resolved == "jax":
        return typing.cast(
            jax.Array,
            jax_kernels.separation_from_j_orb_jax(m1, m2, j_orb, eccentricity),
        )
    return _rust.binary_separation_from_j_orb(
        float(m1), float(m2), float(j_orb), float(eccentricity)
    )


def eval_ritter_mdot_thin(
    m_donor: float | jax.Array,
    m_accretor: float | jax.Array,
    r_donor: float | jax.Array,
    rl_donor: float | jax.Array,
    rho_surf: float | jax.Array,
    p_surf: float | jax.Array,
    t_surf: float | jax.Array,
    mu_surf: float | jax.Array,
    *,
    is_kolb_scheme: bool = False,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Evaluate Ritter (1988) optically thin RLOF rate (`g s^-1`).

    Args:
        m_donor: Donor mass (g).
        m_accretor: Accretor mass (g).
        r_donor: Donor photospheric radius (cm).
        rl_donor: Donor Roche-lobe radius (cm).
        rho_surf: Surface cell density (`g cm^-3`).
        p_surf: Surface cell pressure (`dyn cm^-2`).
        t_surf: Surface cell temperature (K).
        mu_surf: Surface mean molecular weight.
        is_kolb_scheme: Cap exponential factor at unity when $R_d > R_{L,d}$.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Optically thin rate $\\dot{M}_{\\mathrm{thin}} \\le 0$ (`g s^-1`).
    """
    resolved = _dispatch.resolve_backend(
        backend,
        m_donor,
        m_accretor,
        r_donor,
        rl_donor,
        rho_surf,
        p_surf,
        t_surf,
        mu_surf,
    )
    if resolved == "jax":
        return typing.cast(
            jax.Array,
            jax_kernels.eval_ritter_mdot_thin_jax(
                m_donor,
                m_accretor,
                r_donor,
                rl_donor,
                rho_surf,
                p_surf,
                t_surf,
                mu_surf,
                is_kolb_scheme,
            ),
        )
    return _rust.binary_eval_ritter_mdot_thin(
        float(m_donor),
        float(m_accretor),
        float(r_donor),
        float(rl_donor),
        float(rho_surf),
        float(p_surf),
        float(t_surf),
        float(mu_surf),
        is_kolb_scheme,
    )


def eval_kolb_mdot_thick(
    m_donor: float,
    m_accretor: float,
    rl_donor: float,
    r_profile: npt.ArrayLike,
    p_profile: npt.ArrayLike,
    t_profile: npt.ArrayLike,
    mu_profile: npt.ArrayLike,
    gamma1_profile: npt.ArrayLike,
) -> float:
    """Evaluate Kolb & Ritter (1990) optically thick RLOF rate (`g s^-1`).

    Args:
        m_donor: Donor mass (g).
        m_accretor: Accretor mass (g).
        rl_donor: Donor Roche-lobe radius (cm).
        r_profile: Outer-to-inner radius array `(nz,)` (cm).
        p_profile: Outer-to-inner pressure array `(nz,)` (`dyn cm^-2`).
        t_profile: Outer-to-inner temperature array `(nz,)` (K).
        mu_profile: Outer-to-inner mean molecular weight array `(nz,)`.
        gamma1_profile: Outer-to-inner $\\Gamma_1$ array `(nz,)`.

    Returns:
        Optically thick rate $\\dot{M}_{\\mathrm{thick}} \\le 0$ (`g s^-1`).
    """
    return _rust.binary_eval_kolb_mdot_thick(
        float(m_donor),
        float(m_accretor),
        float(rl_donor),
        _dispatch.as_c_f64(r_profile),
        _dispatch.as_c_f64(p_profile),
        _dispatch.as_c_f64(t_profile),
        _dispatch.as_c_f64(mu_profile),
        _dispatch.as_c_f64(gamma1_profile),
    )


def eval_l2_mass_loss_fraction(
    donor_mass_msun: float,
    accretor_mass_msun: float,
    mass_transfer_rate_msun_yr: float,
    orbital_separation_rsun: float,
    disk_alpha: float = 0.1,
    disk_mu: float = 0.6,
) -> float:
    """Evaluate Lu et al. (2023) thin-disk outer $L_2$ mass-loss fraction.

    Args:
        donor_mass_msun: Donor mass ($M_\\odot$).
        accretor_mass_msun: Accretor mass ($M_\\odot$).
        mass_transfer_rate_msun_yr: Mass transfer rate (`M_sun yr^-1`).
        orbital_separation_rsun: Orbital separation ($R_\\odot$).
        disk_alpha: Shakura-Sunyaev $\\alpha$ viscosity parameter.
        disk_mu: Disk mean molecular weight.

    Returns:
        Outer $L_2$ mass-loss fraction $f_{L2} \\in [0, 1]$.
    """
    return _rust.binary_eval_l2_mass_loss_fraction(
        float(donor_mass_msun),
        float(accretor_mass_msun),
        float(mass_transfer_rate_msun_yr),
        float(orbital_separation_rsun),
        float(disk_alpha),
        float(disk_mu),
    )


def pick_next_timestep_rl(
    donor_time_step_yr: float,
    max_timestep_sec: float,
    rl_gap_old_donor: float,
    rl_gap_donor: float,
    *,
    fr: float = 0.01,
    fr_limit: float = 0.01,
    fr_dt_limit: float = 0.0,
    dt_softening_factor: float = 0.0,
    time_delta_coeff: float = 1.0,
) -> tuple[float, int]:
    """Evaluate `binary_pick_next_timestep` for a relative Roche-lobe gap step.

    Args:
        donor_time_step_yr: Completed donor timestep (yr).
        max_timestep_sec: Previous maximum binary timestep (s).
        rl_gap_old_donor: Previous relative Roche-lobe gap.
        rl_gap_donor: Current relative Roche-lobe gap.
        fr: Relative Roche-lobe gap limit (`fr`).
        fr_limit: Floor on $|g_d|$ (`fr_limit`).
        fr_dt_limit: Minimum timestep floor (yr).
        dt_softening_factor: Log-space softening weight.
        time_delta_coeff: Global scaling coefficient (`time_delta_coeff`).

    Returns:
        Tuple `(max_timestep_sec, limit_reason)`.
    """
    return _rust.binary_pick_next_timestep_rl(
        float(donor_time_step_yr),
        float(max_timestep_sec),
        float(rl_gap_old_donor),
        float(rl_gap_donor),
        float(fr),
        float(fr_limit),
        float(fr_dt_limit),
        float(dt_softening_factor),
        float(time_delta_coeff),
    )
