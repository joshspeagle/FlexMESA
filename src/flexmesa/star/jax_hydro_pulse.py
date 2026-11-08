# SPDX-License-Identifier: LGPL-3.0-or-later
"""Branchless `@jax.jit` kernels for Wave 5.2: HLLC Riemann hydrodynamics,
`RSP2` time-dependent turbulent convection, and Dziembowski pulsation
variables (`hydro_riemann.f90`, `hydro_momentum.f90`, `hydro_rsp2.f90`,
`rsp2_solve.f90`, and `pulse_utils.f90`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from flexmesa import constants

X_ALFAP: float = constants.TWO_THIRDS
X_ALFAS: float = 0.5 * constants.SQRT_2_DIV_3
X_ALFAC: float = 0.5 * constants.SQRT_2_DIV_3
X_CEDE: float = (8.0 / 3.0) * constants.SQRT_2_DIV_3
X_GAMMAR: float = 3.464_101_615_137_754_4

type _Arr5 = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
type _Arr6 = tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]
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
    """Evaluate the well-balanced Lagrangian HLLC Riemann state at interior
    faces (`do1_Riemann_Types` in `star/private/hydro_riemann.f90`).

    Args:
        p_l, rho_l, cs_l, u_l, dm_l: Inner cell $k$ pressure (`dyn cm^-2`),
            density (`g cm^-3`), sound speed (`cm s^-1`), velocity (`cm s^-1`),
            and mass (`g`) of shape `(*batch,)`.
        p_r, rho_r, cs_r, u_r, dm_r: Outer cell $k-1$ state of shape
            `(*batch,)`.
        r_face: Face radius $r_k$ (`cm`) of shape `(*batch,)`.
        m_grav_face: Enclosed gravitational mass $m_{\\mathrm{grav},k}$ (`g`) of
            shape `(*batch,)`.
        cgrav_face: Effective gravitational constant $G_{\\mathrm{eff},k}$
            (`cm^3 g^-1 s^-2`) of shape `(*batch,)`.

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
    """Evaluate the surface ($k = 0$) HLLC Riemann boundary state
    (`setup_for_surf_Riemann` in `star/private/hydro_riemann.f90`).

    Args:
        p_surf_cell, rho_surf, cs_surf, u_surf, dm_surf: Outermost cell
            pressure (`dyn cm^-2`), density (`g cm^-3`), sound speed
            (`cm s^-1`), velocity (`cm s^-1`), and mass (`g`).
        r_surf, m_surf, cgrav_surf: Surface radius (`cm`), mass (`g`), and
            gravitational constant (`cm^3 g^-1 s^-2`).
        p_bc: External atmospheric boundary pressure (`dyn cm^-2`).

    Returns:
        7-tuple `(p_l_star, p_r_star, s_l, s_r, u_face, p_face, ke_flux)`.
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
    """Evaluate the spherical Lagrangian HLLC momentum residual
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


def _rsp2_rate_coeffs_jax(
    t: jax.Array,
    rho_s: jax.Array,
    p: jax.Array,
    cp: jax.Array,
    grad_ad: jax.Array,
    delta: jax.Array,
    kappa: jax.Array,
    hp_s: jax.Array,
    grad_t: jax.Array,
    alfa: float,
    alfas: float,
    alfad: float,
    gammar: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute local `RSP2` rate coefficients `(c_s, c_d, c_dr)`."""
    y_super = grad_t - grad_ad
    a_cell = t * p * delta / (rho_s * hp_s)
    c_s = X_ALFAS * alfa * alfas * a_cell * y_super
    alfa_pos = alfa > 0.0
    safe_alfa = jnp.where(alfa_pos, alfa, 1.0)
    c_d = jnp.where(alfa_pos, (X_CEDE * alfad / safe_alfa) / hp_s, 0.0)
    c_dr_pref = jnp.where(alfa_pos, (X_GAMMAR * gammar / safe_alfa) ** 2, 0.0)
    rad_cond = (
        constants.CLIGHT
        * constants.CRAD
        * (t**3)
        * delta
        / (
            rho_s
            * rho_s
            * (jnp.maximum(cp, 1.0e-30) ** 2)
            * jnp.maximum(kappa, 1.0e-30)
        )
    )
    c_dr = c_dr_pref * rad_cond / (hp_s * hp_s)
    return c_s, c_d, c_dr


@jax.jit
def eval_rsp2_zone_jax(
    w: jax.Array,
    t: jax.Array,
    rho: jax.Array,
    p: jax.Array,
    cp: jax.Array,
    grad_ad: jax.Array,
    delta: jax.Array,
    kappa: jax.Array,
    hp_cell: jax.Array,
    grad_t: jax.Array,
    r_outer: jax.Array,
    r_inner: jax.Array,
    u_outer: jax.Array,
    u_inner: jax.Array,
    dm: jax.Array,
    dlnd_dt: jax.Array,
    alfa: float = 1.5,
    alfap: float = 1.0,
    alfas: float = 1.0,
    alfad: float = 1.0,
    alfam: float = 0.25,
    gammar: float = 1.0,
    min_w2_for_turbulence: float = 1.0e-4,
) -> _Arr7:
    """Evaluate cell-centered `RSP2` turbulent pressure, eddy viscosity, and
    turbulent energy sources/sinks (`compute_cell_et_source_and_sink` in
    `star/private/hydro_rsp2.f90`).

    Args:
        w: Specific turbulent velocity $\\sqrt{e_t}$ (`cm s^-1`).
        t, rho, p, cp, grad_ad, delta, kappa: Cell thermodynamic state and
            opacity (`K`, `g cm^-3`, `dyn cm^-2`, `erg g^-1 K^-1`,
            dimensionless, dimensionless, `cm^2 g^-1`).
        hp_cell, grad_t: Cell pressure scale height (`cm`) and temperature
            gradient.
        r_outer, r_inner, u_outer, u_inner: Bounding radii (`cm`) and
            velocities (`cm s^-1`).
        dm: Cell mass (`g`).
        dlnd_dt: Logarithmic density rate $\\mathrm{d}\\ln\\rho/\\mathrm{d}t$
            (`s^-1`).
        alfa, alfap, alfas, alfad, alfam, gammar, min_w2_for_turbulence:
            Kuhfuss (1986) `RSP2` model parameters.

    Returns:
        7-tuple `(p_turb, chi_turb, eq_visc, source_s, sink_d, sink_dr,
        de_t_dt)` of shape `(*batch,)`.
    """
    w_pos = jnp.maximum(w, 0.0)
    w2 = w_pos * w_pos
    rho_s = jnp.maximum(rho, 1.0e-99)
    hp_s = jnp.maximum(hp_cell, 1.0)

    p_turb = X_ALFAP * alfap * rho_s * w2
    use_ramp = (min_w2_for_turbulence > 0.0) & (w2 < min_w2_for_turbulence)
    safe_min_w2 = jnp.where(
        min_w2_for_turbulence > 0.0, min_w2_for_turbulence, 1.0
    )
    f_eddy = jnp.where(use_ramp, jnp.clip(w2 / safe_min_w2, 0.0, 1.0), 1.0)
    vol_mid = constants.FOUR_THIRDS_PI * 0.5 * (r_outer**3 + r_inner**3)
    chi_turb = (
        f_eddy
        * constants.FOUR_THIRDS
        * alfam
        * hp_s
        * rho_s
        * rho_s
        * w_pos
        * vol_mid
        * vol_mid
    )

    u_over_r_out = jnp.where(
        r_outer > 0.0, u_outer / jnp.where(r_outer > 0.0, r_outer, 1.0), 0.0
    )
    u_over_r_in = jnp.where(
        r_inner > 0.0, u_inner / jnp.where(r_inner > 0.0, r_inner, 1.0), 0.0
    )
    du_over_r = u_over_r_out - u_over_r_in
    dm_s = jnp.maximum(dm, 1.0e-99)
    eq_visc = jnp.maximum(chi_turb * ((du_over_r / dm_s) ** 2), 0.0)

    c_s, c_d, c_dr = _rsp2_rate_coeffs_jax(
        t,
        rho_s,
        p,
        cp,
        grad_ad,
        delta,
        kappa,
        hp_s,
        grad_t,
        alfa,
        alfas,
        alfad,
        gammar,
    )
    source_s = c_s * w_pos
    sink_d = c_d * w2 * w_pos
    sink_dr = c_dr * w2
    pdv_turb = p_turb * dlnd_dt / rho_s
    de_t_dt = source_s + eq_visc - sink_d - sink_dr + pdv_turb
    return p_turb, chi_turb, eq_visc, source_s, sink_d, sink_dr, de_t_dt


@jax.jit
def eval_rsp2_face_jax(
    r_face: jax.Array,
    m_grav_face: jax.Array,
    cgrav_face: jax.Array,
    p_face: jax.Array,
    rho_face: jax.Array,
    t_face: jax.Array,
    cp_face: jax.Array,
    grad_ad_face: jax.Array,
    kappa_face: jax.Array,
    w_inner: jax.Array,
    w_outer: jax.Array,
    t4_inner: jax.Array,
    t4_outer: jax.Array,
    lnp_inner: jax.Array,
    lnp_outer: jax.Array,
    chi_face: jax.Array,
    du_over_r_face: jax.Array,
    dm_bar: jax.Array,
    alfa: float = 1.5,
    alfac: float = 1.0,
    alfat: float = 0.01,
) -> _Arr6:
    """Evaluate face-centered `RSP2` scale height, superadiabaticity, viscous
    stress, and radiative/convective/turbulent luminosities (`do1_face` and
    `do1_Lr_Lc_Lt` in `star/private/hydro_rsp2.f90`).

    Args:
        r_face, m_grav_face, cgrav_face: Face radius (`cm`), gravitational mass
            (`g`), and effective gravitational constant (`cm^3 g^-1 s^-2`).
        p_face, rho_face, t_face, cp_face, grad_ad_face, kappa_face:
            Interpolated face thermodynamic state and opacity.
        w_inner, w_outer: Adjacent cell turbulent velocities (`cm s^-1`).
        t4_inner, t4_outer: Adjacent cell $T^4$ (`K^4`).
        lnp_inner, lnp_outer: Adjacent cell $\\ln P$.
        chi_face: Face eddy viscosity $\\mathrm{X}_{\\mathrm{face},k}$
            (`g^2 s^-1`).
        du_over_r_face: Face shear difference $\\Delta(u/r)_k$ (`s^-1`).
        dm_bar: Face mass spacing $\\overline{\\Delta m}_k$ (`g`).
        alfa, alfac, alfat: `RSP2` mixing-length, convective flux, and
            turbulent kinetic flux multipliers.

    Returns:
        6-tuple `(hp_face, y_face, pi_face, l_r, l_c, l_t)` of shape
        `(*batch,)`.
    """
    r2 = jnp.maximum(r_face * r_face, 1.0e-30)
    area = constants.PI4 * r2
    rho_s = jnp.maximum(rho_face, 1.0e-99)
    gm = jnp.maximum(cgrav_face * m_grav_face, 1.0e-30)
    hp_face = jnp.maximum(r2 * p_face / (rho_s * gm), 1.0)

    dm_s = jnp.maximum(dm_bar, 1.0e-99)
    dlnp_dm = (lnp_outer - lnp_inner) / dm_s
    dt4_dm = (t4_outer - t4_inner) / dm_s
    t4_face = jnp.maximum(t_face**4, 1.0e-30)
    valid_dlnp = jnp.abs(dlnp_dm) > 1.0e-99
    safe_dlnp = jnp.where(valid_dlnp, dlnp_dm, 1.0)
    grad_face = jnp.where(
        valid_dlnp, 0.25 * (dt4_dm / t4_face) / safe_dlnp, 0.0
    )
    y_face = grad_face - grad_ad_face

    pi_face = chi_face * du_over_r_face / dm_s
    l_r = (
        -(area * area)
        * (
            constants.CLIGHT
            * constants.CRAD
            / (3.0 * jnp.maximum(kappa_face, 1.0e-30))
        )
        * dt4_dm
    )

    w_face = 0.5 * (jnp.maximum(w_inner, 0.0) + jnp.maximum(w_outer, 0.0))
    l_c = (
        area
        * rho_s
        * t_face
        * cp_face
        * w_face
        * (X_ALFAS * alfa * alfac * y_face)
    )

    dw2_dm = (w_outer * w_outer - w_inner * w_inner) / dm_s
    l_t = (
        -(area * area)
        * alfa
        * alfat
        * (rho_s * rho_s * hp_face * w_face)
        * dw2_dm
    )
    return hp_face, y_face, pi_face, l_r, l_c, l_t


@jax.jit
def solve_stationary_rsp2_w_jax(
    t: jax.Array,
    rho: jax.Array,
    p: jax.Array,
    cp: jax.Array,
    grad_ad: jax.Array,
    delta: jax.Array,
    kappa: jax.Array,
    hp_cell: jax.Array,
    grad_t: jax.Array,
    alfa: float = 1.5,
    alfas: float = 1.0,
    alfad: float = 1.0,
    gammar: float = 1.0,
) -> jax.Array:
    """Solve the stationary local `RSP2` turbulent balance
    $S(w) - D(w) - D_r(w) = 0$ for $w$ (`star/private/rsp2_solve.f90`).

    Args:
        t, rho, p, cp, grad_ad, delta, kappa: Cell thermodynamic state and
            opacity (`(*batch,)`).
        hp_cell, grad_t: Cell pressure scale height (`cm`) and temperature
            gradient (`(*batch,)`).
        alfa, alfas, alfad, gammar: `RSP2` model multipliers.

    Returns:
        Equilibrium turbulent velocity $w$ (`cm s^-1`) of shape `(*batch,)`.
    """
    rho_s = jnp.maximum(rho, 1.0e-99)
    hp_s = jnp.maximum(hp_cell, 1.0)
    c_s, c_d, c_dr = _rsp2_rate_coeffs_jax(
        t,
        rho_s,
        p,
        cp,
        grad_ad,
        delta,
        kappa,
        hp_s,
        grad_t,
        alfa,
        alfas,
        alfad,
        gammar,
    )
    active = (grad_t > grad_ad) & (alfa > 0.0) & (alfad > 0.0)
    disc_raw = c_dr * c_dr + 4.0 * c_d * c_s
    disc_pos = active & (disc_raw > 0.0)
    disc_safe = jnp.where(disc_pos, disc_raw, 1.0)
    sqrt_disc = jnp.where(disc_pos, jnp.sqrt(disc_safe), 0.0)
    w_pos = jnp.maximum(
        (-c_dr + sqrt_disc) / (2.0 * jnp.maximum(c_d, 1.0e-99)), 0.0
    )
    return jnp.where(active, w_pos, 0.0)


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
    """Evaluate the 5 dimensionless Dziembowski pulsation variables
    $(V, V_g, A_*, U, c_1)$ across a stellar profile (`pulse_utils.f90`).

    Args:
        r, m, p, rho, gamma1, n2, cgrav: Zone radius (`cm`), enclosed mass
            (`g`), pressure (`dyn cm^-2`), density (`g cm^-3`), adiabatic
            exponent $\\Gamma_1$, squared Brunt-Väisälä frequency (`s^-2`), and
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
