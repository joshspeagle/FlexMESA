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

from flexmesa import constants, mtx, turb

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
    return jnp.asarray(alfa * cell_val + beta * val_m1)


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
def _eval_face_mlt_and_brunt_core_jax(
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
    tau_0: float | jax.Array,
    mixing_length_alpha: float,
    use_ledoux_criterion: bool,
    cgrav: float | jax.Array,
    brunt_b_override: jax.Array,
    use_brunt_b_override: bool,
) -> dict[str, jax.Array]:
    r = jnp.exp(ln_r)
    gm = cgrav * m_grav
    grav = gm / (r * r)

    prad = (_CRAD / 3.0) * (t**4)
    alfa, beta = face_weights_jax(dq)

    def _to_face(cell_val: jax.Array) -> jax.Array:
        return jnp.asarray(
            alfa * cell_val
            + beta * jnp.concatenate([cell_val[:1], cell_val[:-1]])
        )

    rho_f = _to_face(rho)
    t_f = _to_face(t)
    p_f = _to_face(peos)
    prad_f = _to_face(prad)
    kap_f = _to_face(opacity)
    grada_f = _to_face(grada)
    cp_f = _to_face(cp)
    chi_rho_f = jnp.maximum(_to_face(chi_rho), 1.0e-10)
    chi_t_f = jnp.maximum(_to_face(chi_t), 1.0e-10)

    hp_std = p_f / (grav * rho_f)
    hp_alt = jnp.sqrt(p_f / jnp.maximum(cgrav * rho_f * rho_f, 1.0e-99))
    hp = jnp.where(jnp.isfinite(hp_std) & (hp_std <= hp_alt), hp_std, hp_alt)
    lam = mixing_length_alpha * hp

    # Brunt composition term B (use log1p((mu_{k-1} - mu_k) / mu_k) to avoid
    # catastrophic cancellation when mu_{k-1} ~ mu_k)
    lnp_m1 = jnp.concatenate([ln_peos[:1], ln_peos[:-1]])
    mu_m1 = jnp.concatenate([mu[:1], mu[:-1]])
    dlnp = lnp_m1 - ln_peos
    mu_safe = jnp.maximum(mu, 1.0e-10)
    dlnmu = jnp.log1p((jnp.maximum(mu_m1, 1.0e-10) - mu_safe) / mu_safe)
    b_raw = (chi_rho_f / chi_t_f) * (
        dlnmu / jnp.where(jnp.abs(dlnp) > 1.0e-14, dlnp, 1.0)
    )
    brunt_b_gradmux = jnp.where(
        (jnp.arange(dq.shape[0]) > 0) & (jnp.abs(dlnp) > 1.0e-14),
        jnp.clip(b_raw, -10.0, 10.0),
        0.0,
    )
    brunt_b = jnp.where(use_brunt_b_override, brunt_b_override, brunt_b_gradmux)

    gradr_base = p_f * kap_f * lum / (4.0 * _FOUR_PI * _CLIGHT * gm * prad_f)
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
    brunt_b: jax.Array | None = None,
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
        brunt_b: Optional precomputed Brunt composition term of shape `(nz,)`.

    Returns:
        Dictionary containing `("scale_height", "gradr", "gradl", "gradt",
        "mlt_vc", "mlt_d", "mlt_cdc", "brunt_b", "brunt_n2")` of shape `(nz,)`.
    """
    has_override = brunt_b is not None
    b_override = (
        jnp.asarray(brunt_b) if brunt_b is not None else jnp.zeros_like(dq)
    )
    return dict(
        _eval_face_mlt_and_brunt_core_jax(
            dq,
            ln_r,
            lum,
            m_grav,
            rho,
            t,
            peos,
            ln_peos,
            grada,
            cp,
            chi_rho,
            chi_t,
            mu,
            opacity,
            tau_0,
            mixing_length_alpha,
            use_ledoux_criterion,
            cgrav,
            b_override,
            has_override,
        )
    )


@jax.jit
def eval_hydro_residuals_jax(
    dq: jax.Array,
    dm: jax.Array,
    dm_bar: jax.Array,
    m_grav: jax.Array,
    lnd: jax.Array,
    ln_t: jax.Array,
    ln_r: jax.Array,
    lum: jax.Array,
    lnd_start: jax.Array,
    ln_peos: jax.Array,
    ln_e: jax.Array,
    energy_start: jax.Array,
    gradt: jax.Array,
    eps_nuc: jax.Array,
    non_nuc_neu: jax.Array,
    eps_mdot: jax.Array,
    ln_peos_surf: float | jax.Array,
    ln_t_surf: float | jax.Array,
    dt: float | jax.Array,
    r_center: float | jax.Array = 0.0,
    l_center: float | jax.Array = 0.0,
    cgrav: float | jax.Array = _G_CGS,
    non_nuc_neu_start: jax.Array | None = None,
    dedt_eqn_r_scale: float | jax.Array = 1.0e8,
    use_momentum_outer_bc: bool | jax.Array = False,
) -> jax.Array:
    """Evaluate the 4-equation hydrostatic Henyey residual array `(nz, 4)`.

    Computes the dimensionless residual slots `(i_equ_l, i_dv_dt, i_dlnd_dt,
    i_dlne_dt)` across all `nz` zones in pure differentiable JAX (`float64`),
    matching `eval_henyey_equations` in `flexmesa-star`.

    Args:
        dq: Cell relative mass fractions of shape `(nz,)`.
        dm: Cell masses of shape `(nz,)` (`g`).
        dm_bar: Face control-volume masses of shape `(nz,)` (`g`).
        m_grav: Face enclosed gravitational masses of shape `(nz,)` (`g`).
        lnd: Cell natural log densities of shape `(nz,)` (`ln(g cm^-3)`).
        ln_t: Cell natural log temperatures of shape `(nz,)` (`ln(K)`).
        ln_r: Face natural log radii of shape `(nz,)` (`ln(cm)`).
        lum: Face luminosities of shape `(nz,)` (`erg s^-1`).
        lnd_start: Start-of-step cell `lnd` of shape `(nz,)`.
        ln_peos: Cell natural log total pressure of shape `(nz,)`.
        ln_e: Cell natural log specific internal energy of shape `(nz,)`.
        energy_start: Start-of-step specific internal energy of shape `(nz,)`.
        gradt: Face actual temperature gradient of shape `(nz,)`.
        eps_nuc: Cell nuclear energy generation rate of shape `(nz,)`.
        non_nuc_neu: Cell thermal neutrino loss rate of shape `(nz,)`.
        eps_mdot: Cell mass-adjustment energy rate of shape `(nz,)`.
        ln_peos_surf: Surface natural log pressure from atmosphere BC.
        ln_t_surf: Surface natural log temperature from atmosphere BC.
        dt: Timestep in seconds (`s`).
        r_center: Inner core radius in centimeters (`cm`).
        l_center: Inner core luminosity in `erg s^-1`.
        cgrav: Gravitational constant (`cm^3 g^-1 s^-2`).
        non_nuc_neu_start: Optional start-of-step cell thermal neutrino loss
            rate of shape `(nz,)` (`erg g^-1 s^-1`) for trapezoidal
            time-centering `0.5 * (non_nuc_neu_start + non_nuc_neu)`.
        dedt_eqn_r_scale: Energy equation relative cell-fraction scale cap
            (`1.0e8` default; `<= 0` disables cell-fraction capping).
        use_momentum_outer_bc: When `True`, zeroes the half-cell hydrostatic
            surface pressure offset `dp_0` (`hydro_eqns.f90:479`).

    Returns:
        Dimensionless residual matrix `equ` of shape `(nz, 4)`.
    """
    nz = dq.shape[0]
    idx = jnp.arange(nz)
    alfa, beta = face_weights_jax(dq)

    rho = jnp.exp(lnd)
    t = jnp.exp(ln_t)
    r = jnp.exp(ln_r)
    peos = jnp.exp(ln_peos)
    energy = jnp.exp(ln_e)

    # 1. Continuity / volume equation (`i_dlnd_dt = 2`)
    r_p1_cubed = jnp.concatenate(
        [r[1:] ** 3, jnp.asarray([r_center**3], dtype=r.dtype)]
    )
    dr3 = (3.0 * dm / _FOUR_PI) / rho
    equ_dlnd = ln_r - jnp.log(r_p1_cubed + dr3) / 3.0

    # Surface BC dp_0 (`hydro_eqns.f90:479`: zeroed for momentum outer BC)
    dp_0_hse = (cgrav * m_grav[0] * dm[0] / (2.0 * _FOUR_PI)) / (r[0] ** 4)
    dp_0 = jnp.where(use_momentum_outer_bc, 0.0, dp_0_hse)
    p_surf = jnp.exp(ln_peos_surf)
    t_surf = jnp.exp(ln_t_surf)

    # 2. Momentum / hydrostatic pressure equation (`i_dv_dt = 1`)
    ln_p_bc = jnp.log(p_surf + dp_0)
    denom_p0 = jnp.where(ln_peos[0] > 1.0, ln_peos[0], 1.0)
    resid_p_surf = (ln_p_bc - ln_peos[0]) / denom_p0

    peos_m1 = jnp.concatenate([peos[:1], peos[:-1]])
    p_avg = alfa * peos + beta * peos_m1
    dp_hse = -(cgrav * m_grav / (r * r)) * (dm_bar / (_FOUR_PI * r * r))
    resid_p_int = (dp_hse - (peos_m1 - peos)) / p_avg
    equ_dv = jnp.where(idx == 0, resid_p_surf, resid_p_int)

    # 3. Temperature transport / surface T BC equation (`i_equ_l = 0`)
    dt_0 = dp_0 * gradt[0] * (t[0] / peos[0])
    ln_t_bc = jnp.log(t_surf + dt_0)
    denom_t0 = jnp.where(ln_t[0] > 1.0, ln_t[0], 1.0)
    resid_t_surf = (ln_t_bc - ln_t[0]) / denom_t0

    dm_m1 = jnp.concatenate([dm[:1], dm[:-1]])
    dm_k = 0.5 * (dm_m1 + dm)
    dlnp_dm_qhse = -(cgrav * m_grav) / (_FOUR_PI * (r**4) * p_avg)
    dln_t_theory = dlnp_dm_qhse * gradt * dm_k
    t_m1 = jnp.concatenate([t[:1], t[:-1]])
    t_point = alfa * t + beta * t_m1
    dln_t_actual = (t_m1 - t) / t_point
    resid_t_int = dln_t_theory - dln_t_actual
    equ_l = jnp.where(idx == 0, resid_t_surf, resid_t_int)

    # 4. Energy conservation equation (`i_dlne_dt = 3`)
    lum_p1 = jnp.concatenate(
        [lum[1:], jnp.asarray([l_center], dtype=lum.dtype)]
    )
    dl_dm = (lum - lum_p1) / dm
    neu_eff = (
        non_nuc_neu
        if non_nuc_neu_start is None
        else 0.5 * (non_nuc_neu_start + non_nuc_neu)
    )
    sources = eps_nuc - neu_eff + eps_mdot
    dt_safe = jnp.maximum(dt, 1.0e-30)
    de_dt = (energy - energy_start) / dt_safe
    pdv_dt = peos * (jnp.exp(-lnd) - jnp.exp(-lnd_start)) / dt_safe
    esum = -dl_dm + sources - de_dt - pdv_dt
    total_e_start = jnp.maximum(
        jnp.sum(jnp.maximum(energy_start, 0.0) * dm), 1.0e-30
    )
    cell_e_frac = (jnp.maximum(energy_start, 0.0) * dm) / total_e_start
    surf_cap = jnp.where(idx == 0, 1.0e-6, 1.0)
    base_scal = jnp.where(
        dedt_eqn_r_scale > 0.0,
        jnp.minimum(surf_cap, cell_e_frac * dedt_eqn_r_scale),
        surf_cap,
    )
    scal = base_scal * dt_safe / jnp.maximum(energy_start, 1.0)
    equ_dlne = esum * scal

    return jnp.stack([equ_l, equ_dv, equ_dlnd, equ_dlne], axis=-1)


def solve_henyey_linear_jax(
    lblk: jax.Array,
    dblk: jax.Array,
    ublk: jax.Array,
    equ: jax.Array,
    solver_mode: str = "pcr",
) -> jax.Array:
    """Solve the scaled Henyey linear system `J dx_scaled = -equ`.

    Args:
        lblk: Sub-diagonal blocks of shape `(nz, nvar, nvar)`.
        dblk: Diagonal blocks of shape `(nz, nvar, nvar)`.
        ublk: Super-diagonal blocks of shape `(nz, nvar, nvar)`.
        equ: Residual vector of shape `(nz, nvar)`.
        solver_mode: `"pcr"` for native JAX Parallel Cyclic Reduction or
            `"rust_ffi"` for the `jax.custom_vjp` Safe Rust `bcyclic` FFI.

    Returns:
        Scaled Newton step `dx_scaled` of shape `(nz, nvar)`.

    Raises:
        ValueError: If `solver_mode` is not `"pcr"` or `"rust_ffi"`.
    """
    rhs = -equ
    if solver_mode == "pcr":
        return mtx.solve_block_tridiag_pcr(lblk, dblk, ublk, rhs)
    if solver_mode == "rust_ffi":
        return mtx.solve_block_tridiag_rust_ffi(lblk, dblk, ublk, rhs)
    msg = f"Invalid solver_mode={solver_mode!r}; expected 'pcr' or 'rust_ffi'."
    raise ValueError(msg)
