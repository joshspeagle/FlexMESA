// SPDX-License-Identifier: LGPL-3.0-or-later
//! Time-dependent Radial Stellar Pulsation (`RSP2`) turbulent convection
//! equations and stationary pre-solver (`star/private/hydro_rsp2.f90` and
//! `star/private/rsp2_solve.f90`, resolving Issue #5).
//!
//! # Physical Formulation
//! Implements the Kuhfuss (1986) and Smolec & Moskalik (2008) 1-equation
//! non-local time-dependent turbulent convection model adopted in MESA V
//! (Paxton et al. 2019). The primary turbulent variable is the specific
//! turbulent velocity $w_k = \sqrt{e_{t,k}}$ ($\mathrm{cm\,s^{-1}}$), governed
//! by:
//! $$\frac{\mathrm{d}e_{t,k}}{\mathrm{d}t} +
//!   \frac{\partial L_{t,k}}{\partial m} =
//!   S_k + E_{q,\mathrm{cell},k} - D_k - D_{r,k} +
//!   \frac{P_{\mathrm{trb},k}}{\rho_k}
//!   \frac{\mathrm{d}\ln\rho_k}{\mathrm{d}t},$$
//! where $P_{\mathrm{trb},k} = \frac{2}{3}\alpha_p \rho_k w_k^2$ is the
//! turbulent pressure, $S_k$ is buoyant convective driving, $D_k \propto w_k^3
//! / H_p$ is the Kolmogorov cascade sink, $D_{r,k} \propto w_k^2 / H_p^2$ is
//! radiative cooling of eddies, and $E_{q,\mathrm{cell},k}$ is eddy-viscous
//! shear heating.

use flexmesa_const::{
    CLIGHT, CRAD, FOUR_THIRDS, FOUR_THIRDS_PI, PI4, SQRT_2_DIV_3, TWO_THIRDS,
};

/// Geometric constant $x_{\alpha_p} = \frac{2}{3}$ for isotropic turbulent
/// pressure.
pub const X_ALFAP: f64 = TWO_THIRDS;

/// Kuhfuss (1986) buoyant source & convective flux prefactor
/// $x_{\alpha_s} = x_{\alpha_c} = \frac{1}{2}\sqrt{2/3}$.
pub const X_ALFAS: f64 = 0.5 * SQRT_2_DIV_3;

/// Kuhfuss (1986) convective enthalpy flux prefactor
/// $x_{\alpha_c} = \frac{1}{2}\sqrt{2/3}$.
pub const X_ALFAC: f64 = 0.5 * SQRT_2_DIV_3;

/// Kuhfuss (1986) Kolmogorov turbulent dissipation prefactor
/// $x_{\mathrm{cede}} = \frac{8}{3}\sqrt{2/3}$.
pub const X_CEDE: f64 = (8.0 / 3.0) * SQRT_2_DIV_3;

/// Kuhfuss (1986) radiative eddy cooling prefactor $x_{\gamma_r} = 2\sqrt{3}$.
pub const X_GAMMAR: f64 = 3.464_101_615_137_754_4;

/// Dimensionless `RSP2` turbulent convection control parameters
/// (`star/defaults/controls.defaults`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rsp2Config {
    /// Mixing-length scale parameter $\alpha$ ($\Lambda = \alpha H_p$).
    pub alfa: f64,
    /// Turbulent pressure multiplier $\alpha_p$.
    pub alfap: f64,
    /// Buoyant driving multiplier $\alpha_s$.
    pub alfas: f64,
    /// Convective enthalpy flux multiplier $\alpha_c$.
    pub alfac: f64,
    /// Turbulent dissipation sink multiplier $\alpha_d$.
    pub alfad: f64,
    /// Eddy-viscosity multiplier $\alpha_m$.
    pub alfam: f64,
    /// Non-local turbulent kinetic flux multiplier $\alpha_t$.
    pub alfat: f64,
    /// Radiative eddy-cooling multiplier $\gamma_r$.
    pub gammar: f64,
    /// Smooth turn-on threshold $w_{\min}^2$ ($\mathrm{cm^2\,s^{-2}}$) for
    /// eddy viscosity (`RSP2_min_w2_for_turbulence`).
    pub min_w2_for_turbulence: f64,
}

impl Default for Rsp2Config {
    fn default() -> Self {
        Self {
            alfa: 1.5,
            alfap: 1.0,
            alfas: 1.0,
            alfac: 1.0,
            alfad: 1.0,
            alfam: 0.25,
            alfat: 0.01,
            gammar: 1.0,
            min_w2_for_turbulence: 1.0e-4,
        }
    }
}

/// Cell-centered `RSP2` turbulent pressure, eddy viscosity, sources, and sinks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rsp2ZoneResult {
    /// Turbulent pressure
    /// $P_{\mathrm{trb},k} = \frac{2}{3}\alpha_p \rho_k w_k^2$
    /// ($\mathrm{dyn\,cm^{-2}}$).
    pub p_turb: f64,
    /// Turbulent eddy viscosity $\mathrm{X}_{\mathrm{cell},k}$
    /// ($\mathrm{g^2\,s^{-1}}$).
    pub chi_turb: f64,
    /// Viscous shear heating $E_{q,\mathrm{cell},k}$
    /// ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub eq_visc: f64,
    /// Buoyant convective source $S_k$ ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub source_s: f64,
    /// Kolmogorov cascade dissipation sink $D_k$
    /// ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub sink_d: f64,
    /// Radiative eddy-cooling sink $D_{r,k}$
    /// ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub sink_dr: f64,
    /// Net local turbulent energy rate
    /// $S_k + E_{q,\mathrm{cell},k} - D_k - D_{r,k} + P_{\mathrm{trb},k}
    /// (\mathrm{d}\ln\rho_k/\mathrm{d}t)/\rho_k$
    /// ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub de_t_dt: f64,
}

/// Face-centered `RSP2` scale height, superadiabaticity, viscous stress, and
/// radiative/convective/turbulent luminosities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rsp2FaceResult {
    /// Face pressure scale height
    /// $H_{p,\mathrm{face},k} = r_k^2 P_{\mathrm{face},k} /
    /// (\rho_{\mathrm{face},k} G m_k)$ ($\mathrm{cm}$).
    pub hp_face: f64,
    /// Face superadiabatic temperature gradient
    /// $Y_{\mathrm{face},k} = \nabla_k - \nabla_{\mathrm{ad},k}$
    /// (dimensionless).
    pub y_face: f64,
    /// Face eddy-viscous momentum stress $\Pi_{\mathrm{face},k}$
    /// ($\mathrm{g\,cm\,s^{-2}}$).
    pub pi_face: f64,
    /// Radiative luminosity $L_{r,k}$ ($\mathrm{erg\,s^{-1}}$).
    pub l_r: f64,
    /// Convective enthalpy luminosity $L_{c,k}$ ($\mathrm{erg\,s^{-1}}$).
    pub l_c: f64,
    /// Turbulent kinetic energy luminosity $L_{t,k}$ ($\mathrm{erg\,s^{-1}}$).
    pub l_t: f64,
}

fn rsp2_rate_coeffs(
    t: f64,
    rho_s: f64,
    p: f64,
    cp: f64,
    grad_ad: f64,
    delta: f64,
    kappa: f64,
    hp_s: f64,
    grad_t: f64,
    cfg: &Rsp2Config,
) -> (f64, f64, f64) {
    let y_super = grad_t - grad_ad;
    let a_cell = t * p * delta / (rho_s * hp_s);
    let c_s = X_ALFAS * cfg.alfa * cfg.alfas * a_cell * y_super;
    let c_d = if cfg.alfa > 0.0 {
        (X_CEDE * cfg.alfad / cfg.alfa) / hp_s
    } else {
        0.0
    };
    let c_dr_pref = if cfg.alfa > 0.0 {
        (X_GAMMAR * cfg.gammar / cfg.alfa).powi(2)
    } else {
        0.0
    };
    let rad_cond = CLIGHT * CRAD * t.powi(3) * delta
        / (rho_s * rho_s * cp.max(1.0e-30).powi(2) * kappa.max(1.0e-30));
    let c_dr = c_dr_pref * rad_cond / (hp_s * hp_s);
    (c_s, c_d, c_dr)
}

/// Evaluates the cell-centered `RSP2` turbulent pressure, eddy viscosity,
/// viscous heating, buoyant source $S_k$, dissipation sink $D_k$, and radiative
/// cooling sink $D_{r,k}$ (`compute_cell_et_source_and_sink` in
/// `star/private/hydro_rsp2.f90`).
#[must_use]
pub fn eval_rsp2_zone(
    w: f64,
    t: f64,
    rho: f64,
    p: f64,
    cp: f64,
    grad_ad: f64,
    delta: f64,
    kappa: f64,
    hp_cell: f64,
    grad_t: f64,
    r_outer: f64,
    r_inner: f64,
    u_outer: f64,
    u_inner: f64,
    dm: f64,
    dlnd_dt: f64,
    cfg: &Rsp2Config,
) -> Rsp2ZoneResult {
    let w_pos = w.max(0.0);
    let w2 = w_pos * w_pos;
    let rho_s = rho.max(1.0e-99);
    let hp_s = hp_cell.max(1.0);

    let p_turb = X_ALFAP * cfg.alfap * rho_s * w2;

    let f_eddy =
        if cfg.min_w2_for_turbulence > 0.0 && w2 < cfg.min_w2_for_turbulence {
            (w2 / cfg.min_w2_for_turbulence).clamp(0.0, 1.0)
        } else {
            1.0
        };
    let vol_mid = FOUR_THIRDS_PI * 0.5 * (r_outer.powi(3) + r_inner.powi(3));
    let chi_turb = f_eddy
        * FOUR_THIRDS
        * cfg.alfam
        * hp_s
        * rho_s
        * rho_s
        * w_pos
        * vol_mid
        * vol_mid;

    let u_over_r_out = if r_outer > 0.0 {
        u_outer / r_outer
    } else {
        0.0
    };
    let u_over_r_in = if r_inner > 0.0 {
        u_inner / r_inner
    } else {
        0.0
    };
    let du_over_r = u_over_r_out - u_over_r_in;
    let dm_s = dm.max(1.0e-99);
    let eq_visc = (chi_turb * (du_over_r / dm_s).powi(2)).max(0.0);

    let (c_s, c_d, c_dr) = rsp2_rate_coeffs(
        t, rho_s, p, cp, grad_ad, delta, kappa, hp_s, grad_t, cfg,
    );
    let source_s = c_s * w_pos;
    let sink_d = c_d * w2 * w_pos;
    let sink_dr = c_dr * w2;

    let pdv_turb = p_turb * dlnd_dt / rho_s;
    let de_t_dt = source_s + eq_visc - sink_d - sink_dr + pdv_turb;

    Rsp2ZoneResult {
        p_turb,
        chi_turb,
        eq_visc,
        source_s,
        sink_d,
        sink_dr,
        de_t_dt,
    }
}

/// Evaluates the face-centered `RSP2` scale height $H_{p,\mathrm{face},k}$,
/// superadiabaticity $Y_{\mathrm{face},k}$, viscous stress
/// $\Pi_{\mathrm{face},k}$, radiative luminosity $L_{r,k}$, convective
/// luminosity $L_{c,k}$, and turbulent kinetic energy flux $L_{t,k}$
/// (`do1_face` and `do1_Lr_Lc_Lt` in `star/private/hydro_rsp2.f90`).
#[must_use]
pub fn eval_rsp2_face(
    r_face: f64,
    m_grav_face: f64,
    cgrav_face: f64,
    p_face: f64,
    rho_face: f64,
    t_face: f64,
    cp_face: f64,
    grad_ad_face: f64,
    kappa_face: f64,
    w_inner: f64,
    w_outer: f64,
    t4_inner: f64,
    t4_outer: f64,
    lnp_inner: f64,
    lnp_outer: f64,
    chi_face: f64,
    du_over_r_face: f64,
    dm_bar: f64,
    cfg: &Rsp2Config,
) -> Rsp2FaceResult {
    let r2 = (r_face * r_face).max(1.0e-30);
    let area = PI4 * r2;
    let rho_s = rho_face.max(1.0e-99);
    let gm = (cgrav_face * m_grav_face).max(1.0e-30);
    let hp_face = (r2 * p_face / (rho_s * gm)).max(1.0);

    let dm_s = dm_bar.max(1.0e-99);
    let dlnp_dm = (lnp_outer - lnp_inner) / dm_s;
    let dt4_dm = (t4_outer - t4_inner) / dm_s;
    let t4_face = t_face.powi(4).max(1.0e-30);
    let grad_face = if dlnp_dm.abs() > 1.0e-99 {
        0.25 * (dt4_dm / t4_face) / dlnp_dm
    } else {
        0.0
    };
    let y_face = grad_face - grad_ad_face;

    let pi_face = chi_face * du_over_r_face / dm_s;
    let l_r = -(area * area)
        * (CLIGHT * CRAD / (3.0 * kappa_face.max(1.0e-30)))
        * dt4_dm;

    let w_face = 0.5 * (w_inner.max(0.0) + w_outer.max(0.0));
    let l_c = area
        * rho_s
        * t_face
        * cp_face
        * w_face
        * (X_ALFAS * cfg.alfa * cfg.alfac * y_face);

    let dw2_dm = (w_outer * w_outer - w_inner * w_inner) / dm_s;
    let l_t = -(area * area)
        * cfg.alfa
        * cfg.alfat
        * (rho_s * rho_s * hp_face * w_face)
        * dw2_dm;

    Rsp2FaceResult {
        hp_face,
        y_face,
        pi_face,
        l_r,
        l_c,
        l_t,
    }
}

/// Solves the stationary local `RSP2` turbulent balance
/// $S(w) - D(w) - D_r(w) = 0$ for the equilibrium turbulent velocity $w$
/// ($\mathrm{cm\,s^{-1}}$), matching the quadratic pre-solver in
/// `star/private/rsp2_solve.f90`.
///
/// For $w > 0$ in a superadiabatic zone ($Y_{\mathrm{super}} = \nabla -
/// \nabla_{\mathrm{ad}} > 0$), dividing $c_S w - c_{Dr} w^2 - c_D w^3 = 0$ by
/// $w$ yields the exact positive root:
/// $$w = \frac{-c_{Dr} + \sqrt{c_{Dr}^2 + 4 c_D c_S}}{2 c_D}.$$
#[must_use]
pub fn solve_stationary_rsp2_w(
    t: f64,
    rho: f64,
    p: f64,
    cp: f64,
    grad_ad: f64,
    delta: f64,
    kappa: f64,
    hp_cell: f64,
    grad_t: f64,
    cfg: &Rsp2Config,
) -> f64 {
    if grad_t <= grad_ad || cfg.alfa <= 0.0 || cfg.alfad <= 0.0 {
        return 0.0;
    }
    let rho_s = rho.max(1.0e-99);
    let hp_s = hp_cell.max(1.0);
    let (c_s, c_d, c_dr) = rsp2_rate_coeffs(
        t, rho_s, p, cp, grad_ad, delta, kappa, hp_s, grad_t, cfg,
    );
    let disc = (c_dr * c_dr + 4.0 * c_d * c_s).max(0.0);
    ((-c_dr + disc.sqrt()) / (2.0 * c_d.max(1.0e-99))).max(0.0)
}
