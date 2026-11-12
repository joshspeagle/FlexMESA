# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Branchless `@jax.jit` (`float64`) kernels for `RSP` and `RSP2` radial
stellar pulsation and time-dependent turbulent convection (`hydro_rsp2.f90`
and `rsp/eval_equations.inc`, resolving Issue #5).
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

type _Arr2 = tuple[jax.Array, jax.Array]
type _Arr4 = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
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
def rsp2_flux_limiter_jax(x: jax.Array) -> _Arr2:
    r"""Evaluate the quintic $C^2$ smootherstep enthalpy flux limiter
    (`flux_limiter_function` in `star/private/hydro_rsp2.f90:610-642`).

    Args:
        x: Dimensionless convective-to-enthalpy flux ratio $X = G/F$ of shape
            `(*batch,)`.

    Returns:
        2-tuple `(fl, dfl_dx)` of shape `(*batch,)` giving the limited ratio
        $\mathrm{FL}(X)$ and analytical derivative $\mathrm{dFL}/\mathrm{d}X$.
    """
    x0 = 0.95
    x1 = 1.0
    inv_dx = 1.0 / (x1 - x0)
    s_raw = (x - x0) * inv_dx
    s = jnp.clip(s_raw, 0.0, 1.0)
    p = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
    dp_ds = 30.0 * s * s * (1.0 - s) * (1.0 - s)
    fl_mid = x + (1.0 - x) * p
    dfl_mid = 1.0 - p + (1.0 - x) * dp_ds * inv_dx
    fl = jnp.where(x < x0, x, jnp.where(x >= x1, 1.0, fl_mid))
    dfl_dx = jnp.where(x < x0, 1.0, jnp.where(x >= x1, 0.0, dfl_mid))
    return fl, dfl_dx


@jax.jit
def compute_rsp2_uq_face_jax(
    chi_outer: jax.Array,
    chi_inner: jax.Array,
    r_face: jax.Array,
    dm_bar: jax.Array,
) -> jax.Array:
    r"""Evaluate face eddy-viscous acceleration $U_{q,\mathrm{face},k}$
    (`compute_Uq_face` in `star/private/hydro_rsp2.f90:785-817`).

    Args:
        chi_outer: Outer cell eddy viscosity $\mathrm{X}_{k-1}$ (`erg`), shape
            `(*batch,)`.
        chi_inner: Inner cell eddy viscosity $\mathrm{X}_k$ (`erg`), shape
            `(*batch,)`.
        r_face: Face radius $r_k$ (`cm`), shape `(*batch,)`.
        dm_bar: Face mass spacing $\overline{\Delta m}_k$ (`g`), shape
            `(*batch,)`.

    Returns:
        Eddy-viscous acceleration $4\pi(\mathrm{X}_{k-1}-\mathrm{X}_k) /
        (r_k \overline{\Delta m}_k)$ (`cm s^-2`) of shape `(*batch,)`.
    """
    denom = jnp.maximum(r_face, 1.0e-30) * jnp.maximum(dm_bar, 1.0e-99)
    return constants.PI4 * (chi_outer - chi_inner) / denom


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
    r"""Compute local `RSP2` rate coefficients `(c_s, c_d, c_dr)`."""
    y_super = grad_t - grad_ad
    a_cell = t * p * delta / (rho_s * hp_s)
    c_s = X_ALFAS * alfa * alfas * a_cell * y_super
    alfa_pos = alfa > 0.0
    safe_alfa = jnp.where(alfa_pos, alfa, 1.0)
    c_d = jnp.where(alfa_pos, (X_CEDE * alfad / safe_alfa) / hp_s, 0.0)
    rad_active = alfa_pos & (gammar != 0.0)
    pom = jnp.where(
        rad_active,
        4.0 * constants.BOLTZ_SIGMA * ((X_GAMMAR * gammar / safe_alfa) ** 2),
        0.0,
    )
    pom2 = (t**3) / (
        rho_s * rho_s * jnp.maximum(cp, 1.0e-30) * jnp.maximum(kappa, 1.0e-30)
    )
    c_dr = pom * pom2 / (hp_s * hp_s)
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
    min_w2_for_turbulence: float = 0.0,
    w_min_for_damping: float = 0.0,
) -> _Arr7:
    r"""Evaluate cell-centered `RSP2` turbulent pressure, eddy viscosity,
    viscous heating, and turbulent energy terms (`hydro_rsp2.f90:717-934`).

    Args:
        w: Specific turbulent velocity $\sqrt{e_t}$ (`cm s^-1`), shape
            `(*batch,)`.
        t, rho, p, cp, grad_ad, delta, kappa: Cell thermodynamic state and
            opacity (`K`, `g cm^-3`, `dyn cm^-2`, `erg g^-1 K^-1`,
            dimensionless, dimensionless, `cm^2 g^-1`), shape `(*batch,)`.
        hp_cell, grad_t: Cell pressure scale height (`cm`) and temperature
            gradient, shape `(*batch,)`.
        r_outer, r_inner, u_outer, u_inner: Bounding radii (`cm`) and
            velocities (`cm s^-1`), shape `(*batch,)`.
        dm: Cell mass (`g`), shape `(*batch,)`.
        dlnd_dt: Logarithmic density rate $\mathrm{d}\ln\rho/\mathrm{d}t$
            (`s^-1`), shape `(*batch,)`.
        alfa, alfap, alfas, alfad, alfam, gammar: Kuhfuss (1986) `RSP2`
            mixing-length and convective/viscous/radiative multipliers.
        min_w2_for_turbulence, w_min_for_damping: Minimum $w^2$ (`cm^2 s^-2`)
            and $w$ (`cm s^-1`) thresholds for eddy viscosity and damping.

    Returns:
        7-tuple `(p_turb, chi_turb, eq_visc, source_s, sink_d, sink_dr,
        de_t_dt)` of shape `(*batch,)`.
    """
    w_pos = jnp.maximum(w, 0.0)
    w2 = w_pos * w_pos
    rho_s = jnp.maximum(rho, 1.0e-99)
    hp_s = jnp.maximum(hp_cell, 1.0)
    dm_s = jnp.maximum(dm, 1.0e-99)

    p_turb = X_ALFAP * alfap * rho_s * w2
    use_ramp = (min_w2_for_turbulence > 0.0) & (w2 < min_w2_for_turbulence)
    safe_min_w2 = jnp.where(
        min_w2_for_turbulence > 0.0, min_w2_for_turbulence, 1.0
    )
    f_eddy = jnp.where(use_ramp, jnp.clip(w2 / safe_min_w2, 0.0, 1.0), 1.0)

    u_over_r_out = jnp.where(
        r_outer > 0.0, u_outer / jnp.where(r_outer > 0.0, r_outer, 1.0), 0.0
    )
    u_over_r_in = jnp.where(
        r_inner > 0.0, u_inner / jnp.where(r_inner > 0.0, r_inner, 1.0), 0.0
    )
    du_over_r = u_over_r_out - u_over_r_in
    r6_cell = 0.5 * ((r_outer**6) + (r_inner**6))
    f_chi = f_eddy * (16.0 / 3.0) * constants.PI * (alfam * alfa) / dm_s
    chi_turb = f_chi * rho_s * rho_s * r6_cell * du_over_r * hp_s * w_pos
    eq_visc = constants.PI4 * chi_turb * du_over_r / dm_s

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
    sink_d = c_d * (w2 * w_pos - w_min_for_damping**3)
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
    r"""Evaluate face-centered `RSP2` scale height, superadiabaticity,
    convective transport factor $\Pi_k$, and radiative/convective/turbulent
    luminosities (`hydro_rsp2.f90:230-285, 436-608, 1015-1172`).

    Args:
        r_face, m_grav_face, cgrav_face: Face radius (`cm`), gravitational mass
            (`g`), and effective gravitational constant (`cm^3 g^-1 s^-2`).
        p_face, rho_face, t_face, cp_face, grad_ad_face, kappa_face:
            Interpolated face thermodynamic state and opacity.
        w_inner, w_outer: Adjacent cell turbulent velocities (`cm s^-1`).
        t4_inner, t4_outer: Adjacent cell $T^4$ (`K^4`).
        lnp_inner, lnp_outer: Adjacent cell $\ln P$.
        chi_face, du_over_r_face: Unused compatibility arguments retained for
            call-signature stability.
        dm_bar: Face mass spacing $\overline{\Delta m}_k$ (`g`).
        alfa, alfac, alfat: `RSP2` mixing-length, convective flux, and
            turbulent kinetic flux multipliers.

    Returns:
        6-tuple `(hp_face, y_face, pi_face, l_r, l_c, l_t)` of shape
        `(*batch,)`.
    """
    del chi_face, du_over_r_face
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

    pi_face = X_ALFAS * alfa * cp_face * y_face
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
        area * (X_ALFAC / X_ALFAS) * alfac * (t_face * rho_s) * pi_face * w_face
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
    r"""Solve the stationary local `RSP2` turbulent balance
    $S(w) - D(w) - D_r(w) = 0$ for $w$ (`hydro_rsp2.f90:1195-1253`).

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
def eval_rsp2_turbulent_eqn_residual_jax(
    w: jax.Array,
    w_start: jax.Array,
    rho: jax.Array,
    rho_start: jax.Array,
    p_turb: jax.Array,
    lt_outer: jax.Array,
    lt_inner: jax.Array,
    coupl: jax.Array,
    eq_visc: jax.Array,
    dm: jax.Array,
    dt: jax.Array,
    cp: jax.Array,
    t: jax.Array,
) -> jax.Array:
    r"""Evaluate the dimensionless `RSP2` turbulent-energy Henyey residual
    (`do1_turbulent_energy_eqn` in `star/private/hydro_rsp2.f90:312-417`).

    Args:
        w, w_start: End-of-step and start-of-step turbulent velocities
            (`cm s^-1`), shape `(*batch,)`.
        rho, rho_start: End-of-step and start-of-step cell densities
            (`g cm^-3`), shape `(*batch,)`.
        p_turb: Cell turbulent pressure $P_{\mathrm{trb},k}$ (`dyn cm^-2`),
            shape `(*batch,)`.
        lt_outer, lt_inner: Outer and inner face turbulent luminosities
            (`erg s^-1`), shape `(*batch,)`.
        coupl: Net convective coupling $S_k - D_k - D_{r,k}$
            (`erg g^-1 s^-1`), shape `(*batch,)`.
        eq_visc: Eddy-viscous heating $E_{q,\mathrm{cell},k}$
            (`erg g^-1 s^-1`), shape `(*batch,)`.
        dm, dt: Cell mass (`g`) and timestep (`s`), shape `(*batch,)`.
        cp, t: Specific heat $C_P$ (`erg g^-1 K^-1`) and temperature (`K`),
            shape `(*batch,)`.

    Returns:
        Dimensionless turbulent-energy equation residual of shape `(*batch,)`.
    """
    et_00 = w * w
    et_start = w_start * w_start
    ptrb_dv = p_turb * (
        1.0 / jnp.maximum(rho, 1.0e-99) - 1.0 / jnp.maximum(rho_start, 1.0e-99)
    )
    dt_dlt_dm = (lt_outer - lt_inner) * dt / jnp.maximum(dm, 1.0e-99)
    esum = (et_00 - et_start) + ptrb_dv + dt_dlt_dm - dt * (coupl + eq_visc)
    e_scale = jnp.maximum(cp * t, 1.0)
    return esum / e_scale


@jax.jit
def eval_rsp_hyd_fluxes_jax(
    r_face: jax.Array,
    t_outer: jax.Array,
    t_inner: jax.Array,
    kap_outer: jax.Array,
    kap_inner: jax.Array,
    rho_face: jax.Array,
    t_face: jax.Array,
    pii_face: jax.Array,
    w_outer: jax.Array,
    w_inner: jax.Array,
    hp_face: jax.Array,
    dm_bar: jax.Array,
    du_compr: jax.Array,
    cs_cell: jax.Array,
    rho_cell: jax.Array,
    alfa: float = 1.5,
    alfac: float = 1.0,
    alfat: float = 0.01,
    cq: float = 4.0,
    zsh: float = 0.1,
) -> _Arr4:
    r"""Evaluate Stellingwerf radiative luminosity $L_r$, Kuhfuss convective
    luminosity $L_c$, turbulent kinetic luminosity $L_t$, and artificial
    viscosity pressure $Q_{\mathrm{visc}}$ (`rsp/eval_equations.inc`).

    Args:
        r_face: Face radius (`cm`), shape `(*batch,)`.
        t_outer, t_inner: Outer (`k-1`) and inner (`k`) cell temperatures (`K`),
            shape `(*batch,)`.
        kap_outer, kap_inner: Outer and inner cell opacities (`cm^2 g^-1`),
            shape `(*batch,)`.
        rho_face, t_face, pii_face, hp_face: Face density (`g cm^-3`),
            temperature (`K`), convective factor $\Pi_k$ (`erg g^-1 K^-1`),
            and pressure scale height (`cm`), shape `(*batch,)`.
        w_outer, w_inner: Outer and inner cell turbulent velocities (`cm s^-1`),
            shape `(*batch,)`.
        dm_bar: Face mass spacing (`g`), shape `(*batch,)`.
        du_compr, cs_cell, rho_cell: Cell velocity difference $u_{\mathrm{in}}
            - u_{\mathrm{out}}$ (`cm s^-1`), sound speed (`cm s^-1`), and cell
            density (`g cm^-3`), shape `(*batch,)`.
        alfa, alfac, alfat, cq, zsh: Mixing-length, convective, turbulent
            diffusion, and artificial viscosity parameters.

    Returns:
        4-tuple `(l_r, l_c, l_t, q_visc)` of shape `(*batch,)` (`erg s^-1`,
        `erg s^-1`, `erg s^-1`, `dyn cm^-2`).
    """
    area = constants.PI4 * jnp.maximum(r_face * r_face, 1.0e-30)
    t4_m1 = jnp.maximum(t_outer, 1.0) ** 4
    t4_00 = jnp.maximum(t_inner, 1.0) ** 4
    kap_m1 = jnp.maximum(kap_outer, 1.0e-30)
    kap_00 = jnp.maximum(kap_inner, 1.0e-30)
    bw = jnp.log(t4_m1 / t4_00)
    bk = jnp.log(kap_m1 / kap_00)
    safe_bw = jnp.where(jnp.abs(bw) > 1.0e-12, bw, 1.0)
    denom_st = 1.0 - bk / safe_bw
    use_st = (jnp.abs(bw) > 1.0e-12) & (jnp.abs(denom_st) > 1.0e-12)
    diff_lin = (t4_m1 - t4_00) / (0.5 * (kap_m1 + kap_00))
    diff_st = (t4_m1 / kap_m1 - t4_00 / kap_00) / jnp.where(
        use_st, denom_st, 1.0
    )
    diff = jnp.where(use_st, diff_st, diff_lin)
    dm_s = jnp.maximum(dm_bar, 1.0e-99)
    l_r = -(constants.CRAD * constants.CLIGHT / 3.0) * diff * area * area / dm_s

    w_f = 0.5 * (jnp.maximum(w_outer, 0.0) + jnp.maximum(w_inner, 0.0))
    l_c = (
        area
        * (X_ALFAC / X_ALFAS)
        * alfac
        * (t_face * rho_face)
        * pii_face
        * w_f
    )
    detrb = w_outer * w_outer - w_inner * w_inner
    l_t = (
        -(alfa * alfat)
        * area
        * area
        * (rho_face * rho_face)
        * hp_face
        * w_f
        * detrb
        / dm_s
    )

    dv_eff = jnp.maximum(du_compr - zsh * cs_cell, 0.0)
    q_visc = cq * rho_cell * dv_eff * dv_eff
    return l_r, l_c, l_t, q_visc
