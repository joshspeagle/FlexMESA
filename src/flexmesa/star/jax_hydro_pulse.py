# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Branchless `@jax.jit` kernels for Wave 5.2: HLLC Riemann hydrodynamics,
Dziembowski pulsation variables, `STELLA` Sobolev optical depths, and `GYRE` /
`ADIPLS` BVP wrappers (`hydro_riemann.f90`, `hydro_momentum.f90`,
`pulse_utils.f90`, and `parse_swd_vel_feii.f90`).

`RSP2` time-dependent turbulent convection kernels are defined in
`flexmesa.star.jax_rsp` and re-exported here for backwards compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import _rust, constants
from flexmesa.star import jax_rsp

X_ALFAP: float = jax_rsp.X_ALFAP
X_ALFAS: float = jax_rsp.X_ALFAS
X_ALFAC: float = jax_rsp.X_ALFAC
X_CEDE: float = jax_rsp.X_CEDE
X_GAMMAR: float = jax_rsp.X_GAMMAR
eval_rsp2_zone_jax = jax_rsp.eval_rsp2_zone_jax
eval_rsp2_face_jax = jax_rsp.eval_rsp2_face_jax
solve_stationary_rsp2_w_jax = jax_rsp.solve_stationary_rsp2_w_jax

type _Arr5 = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
type _Arr7 = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]


@jax.jit
def eval_hllc_face_jax(
    p_l: jax.Array,
    rho_l: jax.Array,
    cs_l: jax.Array,
    u_l: jax.Array,
    dm_l: jax.Array,
    p_r: jax.Array,
    rho_r: jax.Array,
    cs_r: jax.Array,
    u_r: jax.Array,
    dm_r: jax.Array,
    r_face: jax.Array,
    m_grav_face: jax.Array,
    cgrav_face: jax.Array,
) -> _Arr7:
    r"""Evaluate the well-balanced Lagrangian HLLC Riemann state at interior
    faces (`do1_Riemann_Types` in `star/private/hydro_riemann.f90`).

    Args:
        p_l, rho_l, cs_l, u_l, dm_l: Left (inner) cell pressure (`dyn cm^-2`),
            density (`g cm^-3`), sound speed (`cm s^-1`), velocity (`cm s^-1`),
            and mass (`g`), shape `(*batch,)`.
        p_r, rho_r, cs_r, u_r, dm_r: Right (outer) cell state, shape
            `(*batch,)`.
        r_face, m_grav_face, cgrav_face: Face radius (`cm`), gravitational mass
            (`g`), and effective gravitational constant (`cm^3 g^-1 s^-2`).

    Returns:
        7-tuple `(p_l_star, p_r_star, s_l, s_r, u_face, p_face, ke_flux)` of
        shape `(*batch,)`.
    """
    r2 = jnp.maximum(r_face * r_face, 1.0e-30)
    area = constants.PI4 * r2
    g_over_area = cgrav_face * m_grav_face / (r2 * area)

    p_l_star = jnp.maximum(p_l - 0.5 * dm_l * g_over_area, 1.0e-30 * p_l)
    p_r_star = jnp.maximum(p_r + 0.5 * dm_r * g_over_area, 1.0e-30 * p_r)

    c_l = jnp.maximum(cs_l, 1.0e-10)
    c_r = jnp.maximum(cs_r, 1.0e-10)
    d_l = jnp.maximum(rho_l, 1.0e-99)
    d_r = jnp.maximum(rho_r, 1.0e-99)

    s_l = jnp.minimum(u_l - c_l, u_r - c_r)
    s_r = jnp.maximum(u_r + c_r, u_l + c_l)

    m_l = d_l * (s_l - u_l)
    m_r = d_r * (s_r - u_r)
    denom = jnp.maximum(m_r - m_l, 1.0e-99)

    u_face = (m_r * u_r - m_l * u_l + p_l_star - p_r_star) / denom
    p_face = jnp.maximum(
        (m_r * (p_l_star - m_l * u_l) - m_l * (p_r_star - m_r * u_r)) / denom,
        0.0,
    )
    ke_flux = area * p_face * u_face
    return p_l_star, p_r_star, s_l, s_r, u_face, p_face, ke_flux


@jax.jit
def eval_hllc_surface_face_jax(
    p_surf_cell: jax.Array,
    rho_surf: jax.Array,
    cs_surf: jax.Array,
    u_surf: jax.Array,
    dm_surf: jax.Array,
    r_surf: jax.Array,
    m_surf: jax.Array,
    cgrav_surf: jax.Array,
    p_bc: jax.Array,
) -> _Arr7:
    r"""Evaluate the surface ($k = 0$) HLLC Riemann boundary state
    (`setup_for_surf_Riemann` in `star/private/hydro_riemann.f90`).

    Args:
        p_surf_cell, rho_surf, cs_surf, u_surf, dm_surf: Outermost cell
            pressure (`dyn cm^-2`), density (`g cm^-3`), sound speed
            (`cm s^-1`), velocity (`cm s^-1`), and mass (`g`).
        r_surf, m_surf, cgrav_surf: Surface radius (`cm`), mass (`g`), and
            gravitational constant (`cm^3 g^-1 s^-2`).
        p_bc: Exterior surface boundary pressure (`dyn cm^-2`).

    Returns:
        7-tuple `(p_l_star, p_r_star, s_l, s_r, u_face, p_face, ke_flux)` of
        shape `(*batch,)`.
    """
    r2 = jnp.maximum(r_surf * r_surf, 1.0e-30)
    area = constants.PI4 * r2
    g_over_area = cgrav_surf * m_surf / (r2 * area)

    p_l_star = jnp.maximum(
        p_surf_cell - 0.5 * dm_surf * g_over_area, 1.0e-30 * p_surf_cell
    )
    p_r_star = jnp.maximum(p_bc, 0.0)
    c_l = jnp.maximum(cs_surf, 1.0e-10)
    d_l = jnp.maximum(rho_surf, 1.0e-99)

    s_l = u_surf - c_l
    s_r = u_surf + c_l
    imp = jnp.maximum(d_l * c_l, 1.0e-99)
    u_face = u_surf + (p_l_star - p_r_star) / (2.0 * imp)
    p_face = 0.5 * (p_l_star + p_r_star)
    ke_flux = area * p_face * u_face
    return p_l_star, p_r_star, s_l, s_r, u_face, p_face, ke_flux


@jax.jit
def eval_riemann_momentum_residual_jax(
    u_k: jax.Array,
    u_old_k: jax.Array,
    dt: jax.Array,
    dm_k: jax.Array,
    r_outer: jax.Array,
    r_inner: jax.Array,
    p_face_outer: jax.Array,
    p_face_inner: jax.Array,
    p_cell_k: jax.Array,
    g_cell_k: jax.Array,
) -> jax.Array:
    r"""Evaluate the spherical Lagrangian HLLC momentum residual
    (`do1_momentum_eqn` in `star/private/hydro_momentum.f90`).

    Args:
        u_k, u_old_k: Current and old cell velocities (`cm s^-1`).
        dt: Timestep (`s`).
        dm_k: Cell mass (`g`).
        r_outer, r_inner: Outer and inner face radii (`cm`).
        p_face_outer, p_face_inner: Outer and inner HLLC contact pressures
            (`dyn cm^-2`).
        p_cell_k: Cell total pressure (`dyn cm^-2`).
        g_cell_k: Cell gravitational acceleration (`cm s^-2`).

    Returns:
        Momentum equation residual (`cm s^-2`) of shape `(*batch,)`.
    """
    a_outer = constants.PI4 * r_outer * r_outer
    a_inner = constants.PI4 * r_inner * r_inner
    dp_force = a_outer * (p_face_outer - p_cell_k) - a_inner * (
        p_face_inner - p_cell_k
    )
    valid_dt = dt > 0.0
    safe_dt = jnp.where(valid_dt, dt, 1.0)
    du_dt = jnp.where(valid_dt, (u_k - u_old_k) / safe_dt, 0.0)
    return du_dt - dp_force / jnp.maximum(dm_k, 1.0e-99) + g_cell_k


@jax.jit
def eval_dziembowski_variables_jax(
    r: jax.Array,
    m: jax.Array,
    p: jax.Array,
    rho: jax.Array,
    gamma1: jax.Array,
    n2: jax.Array,
    cgrav: jax.Array,
    r_tot: float | jax.Array,
    m_tot: float | jax.Array,
) -> _Arr5:
    r"""Evaluate the 5 dimensionless Dziembowski pulsation variables
    $(V, V_g, A_*, U, c_1)$ across a stellar profile (`pulse_utils.f90`).

    Args:
        r, m, p, rho, gamma1, n2, cgrav: Zone radius (`cm`), enclosed mass
            (`g`), pressure (`dyn cm^-2`), density (`g cm^-3`), adiabatic
            exponent $\Gamma_1$, squared Brunt-Väisälä frequency (`s^-2`), and
            gravitational constant (`cm^3 g^-1 s^-2`).
        r_tot, m_tot: Total stellar radius $R$ (`cm`) and mass $M$ (`g`).

    Returns:
        5-tuple `(v_dziem, v_g, a_star, u_dziem, c_1)` of shape `(*batch,)`.
    """
    r_s = jnp.maximum(r, 1.0e-10)
    m_s = jnp.maximum(m, 1.0e-30)
    p_s = jnp.maximum(p, 1.0e-30)
    g1_s = jnp.maximum(gamma1, 1.0)
    g = jnp.maximum(cgrav * m_s / (r_s * r_s), 1.0e-30)
    v_dziem = g * r_s * rho / p_s
    v_g = v_dziem / g1_s
    a_star = r_s * n2 / g
    u_dziem = constants.PI4 * (r_s**3) * rho / m_s
    c_1 = ((r_s / jnp.maximum(r_tot, 1.0e-30)) ** 3) / (
        m_s / jnp.maximum(m_tot, 1.0e-30)
    )
    return v_dziem, v_g, a_star, u_dziem, c_1


@jax.jit
def eval_sobolev_feii_5169_tau_jax(
    rho: jax.Array,
    x_fe: jax.Array,
    eta: jax.Array,
    t_exp_s: float | jax.Array,
) -> jax.Array:
    r"""Evaluate the `STELLA` homologous Sobolev optical depth
    $\tau_{\mathrm{Sob}}$ for `Fe II` 5169 (`parse_swd_vel_feii.f90`).

    Args:
        rho: Zone mass density (`g cm^-3`) of shape `(*batch,)`.
        x_fe: Iron mass fraction (`dimensionless`) of shape `(*batch,)`.
        eta: Population fraction $\eta(\rho, T)$ of shape `(*batch,)`.
        t_exp_s: Time since shock breakout (`s`).

    Returns:
        Sobolev optical depth $\tau_{\mathrm{Sob}}$ of shape `(*batch,)`.
    """
    n_fe = (
        jnp.maximum(rho, 0.0) * jnp.maximum(x_fe, 0.0) / (56.0 * constants.AMU)
    )
    return (
        constants.PI
        * (constants.QE * constants.QE)
        / (constants.ME * constants.CLIGHT)
        * n_fe
        * jnp.maximum(eta, 0.0)
        * 0.023
        * jnp.maximum(t_exp_s, 0.0)
        * 5169.0e-8
    )


def solve_gyre_mesa_modes(
    mesa_str: str,
    l_deg: int,
    omega_min: float,
    omega_max: float,
    n_scan: int = 200,
    nonadiabatic: bool = True,
) -> list[tuple[int, int, int, int, float, float, float, float, float]]:
    r"""Solve the 4th/6th-order `GYRE` BVP on a `GYRE` MESA model string.

    Args:
        mesa_str: `GYRE` v101/v120 structure table text.
        l_deg: Spherical harmonic degree $\ell$.
        omega_min, omega_max: Dimensionless frequency scan bounds.
        n_scan: Number of scan intervals.
        nonadiabatic: Whether to include radiative cooling perturbations.

    Returns:
        List of eigenmode tuples `(l, n_p, n_g, n_pg, omega_re, omega_im,
        freq_uhz, inertia, eta_growth)`.
    """
    return list(
        _rust.star_solve_gyre_mesa_modes(
            str(mesa_str),
            int(l_deg),
            float(omega_min),
            float(omega_max),
            int(n_scan),
            bool(nonadiabatic),
        )
    )


def solve_adipls_fgong_modes(
    fgong_str: str,
    l_deg: int,
    freq_min_uhz: float,
    freq_max_uhz: float,
    n_scan: int = 500,
) -> list[tuple[int, int, int, int, float, float, float, float]]:
    r"""Solve the 4th/6th-order `ADIPLS` BVP on an `FGONG` model string.

    Args:
        fgong_str: `FGONG` structure file text.
        l_deg: Spherical harmonic degree $\ell$.
        freq_min_uhz, freq_max_uhz: Frequency scan bounds (`uHz`).
        n_scan: Number of scan intervals.

    Returns:
        List of eigenmode tuples `(l, n_p, n_g, n_pg, sigma2, freq_uhz,
        inertia, delta_nu_share)`.
    """
    return list(
        _rust.star_solve_adipls_fgong_modes(
            str(fgong_str),
            int(l_deg),
            float(freq_min_uhz),
            float(freq_max_uhz),
            int(n_scan),
        )
    )


def run_stella_from_hyd_abn(
    hyd_str: str,
    abn_str: str,
    eta_str: str,
    times_days: list[float],
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float, float]]]:
    r"""Run the `STELLA` bolometric light-curve and `Fe II` 5169 solver.

    Args:
        hyd_str: `.hyd` hydrodynamic profile table text.
        abn_str: `.abn` abundance table text.
        eta_str: `Fe II` 5169 $\eta(\rho, T)$ lookup table text.
        times_days: Post-breakout epochs (`days`).

    Returns:
        2-tuple `(lbol_curve, feii_velocities)`.
    """
    lbol, vel = _rust.star_run_stella_from_hyd_abn(
        str(hyd_str),
        str(abn_str),
        str(eta_str),
        [float(t) for t in times_days],
    )
    return list(lbol), list(vel)
