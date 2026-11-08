// SPDX-License-Identifier: LGPL-3.0-or-later
//! 1D spherical Lagrangian HLLC Riemann hydrodynamic solver and well-balanced
//! momentum equation (`star/private/hydro_riemann.f90` and
//! `star/private/hydro_momentum.f90`).
//!
//! # Physical Formulation
//! Evaluates contact velocity $u_{\mathrm{face},k} = S_*$ and contact pressure
//! $P_{\mathrm{face},k} = P^*$ across each Lagrangian cell interface $k$ using
//! the Toro, Spruce & Speares (1994) three-wave HLLC approximate Riemann solver
//! with hydrostatic gravitational pressure reconstruction:
//! $$P_L^* = P_k - \frac{1}{2}\Delta m_k
//!   \frac{G_{\mathrm{eff},k} m_{\mathrm{grav},k}}{4\pi r_k^4}, \qquad
//!   P_R^* = P_{k-1} + \frac{1}{2}\Delta m_{k-1}
//!   \frac{G_{\mathrm{eff},k} m_{\mathrm{grav},k}}{4\pi r_k^4}.$$
//! When $u_k = u_{k-1} = 0$ and the adjacent cells are in discrete hydrostatic
//! balance ($P_L^* = P_R^*$), the HLLC contact state preserves exact
//! hydrostatic equilibrium ($u_{\mathrm{face},k} = 0$ and
//! $P_{\mathrm{face},k} = P_L^*$).

use flexmesa_const::PI4;
use rayon::prelude::*;

use crate::model::StarModel;

/// HLLC approximate Riemann state at a single Lagrangian cell interface $k$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HllcFaceState {
    /// Hydrostatically reconstructed left (inner cell $k$) pressure $P_L^*$
    /// ($\mathrm{dyn\,cm^{-2}}$).
    pub p_l_star: f64,
    /// Hydrostatically reconstructed right (outer cell $k-1$) pressure $P_R^*$
    /// ($\mathrm{dyn\,cm^{-2}}$).
    pub p_r_star: f64,
    /// Left (inward) acoustic wave speed $S_L$ ($\mathrm{cm\,s^{-1}}$).
    pub s_l: f64,
    /// Right (outward) acoustic wave speed $S_R$ ($\mathrm{cm\,s^{-1}}$).
    pub s_r: f64,
    /// HLLC contact wave speed $u_{\mathrm{face},k} = S_*$
    /// ($\mathrm{cm\,s^{-1}}$).
    pub u_face: f64,
    /// HLLC contact pressure $P_{\mathrm{face},k} = P^*$
    /// ($\mathrm{dyn\,cm^{-2}}$).
    pub p_face: f64,
    /// Mechanical work flux
    /// $4\pi r_k^2 P_{\mathrm{face},k} u_{\mathrm{face},k}$
    /// ($\mathrm{erg\,s^{-1}}$).
    pub ke_flux: f64,
}

impl HllcFaceState {
    /// Unpacks the 7 face state scalars as `(p_l_star, p_r_star, s_l, s_r,
    /// u_face, p_face, ke_flux)`.
    #[must_use]
    pub const fn to_tuple(self) -> (f64, f64, f64, f64, f64, f64, f64) {
        (
            self.p_l_star,
            self.p_r_star,
            self.s_l,
            self.s_r,
            self.u_face,
            self.p_face,
            self.ke_flux,
        )
    }
}

/// Evaluates the well-balanced Lagrangian HLLC Riemann state at interior face
/// $k$ between inner cell $k$ (state $L$) and outer cell $k-1$ (state $R$),
/// matching `do1_Riemann_Types` in `star/private/hydro_riemann.f90`.
///
/// # Arguments
/// * `p_l`, `rho_l`, `cs_l`, `u_l`, `dm_l` - Pressure
///   ($\mathrm{dyn\,cm^{-2}}$), density ($\mathrm{g\,cm^{-3}}$), adiabatic
///   sound speed ($\mathrm{cm\,s^{-1}}$), velocity ($\mathrm{cm\,s^{-1}}$), and
///   mass ($\mathrm{g}$) of inner cell $k$.
/// * `p_r`, `rho_r`, `cs_r`, `u_r`, `dm_r` - Corresponding state of outer cell
///   $k-1$.
/// * `r_face` - Face radius $r_k$ ($\mathrm{cm}$).
/// * `m_grav_face` - Enclosed gravitational mass $m_{\mathrm{grav},k}$
///   ($\mathrm{g}$).
/// * `cgrav_face` - Effective gravitational constant $G_{\mathrm{eff},k}$
///   ($\mathrm{cm^3\,g^{-1}\,s^{-2}}$).
#[must_use]
pub fn eval_hllc_face(
    p_l: f64,
    rho_l: f64,
    cs_l: f64,
    u_l: f64,
    dm_l: f64,
    p_r: f64,
    rho_r: f64,
    cs_r: f64,
    u_r: f64,
    dm_r: f64,
    r_face: f64,
    m_grav_face: f64,
    cgrav_face: f64,
) -> HllcFaceState {
    let r2 = (r_face * r_face).max(1.0e-30);
    let area = PI4 * r2;
    let g_over_area = cgrav_face * m_grav_face / (r2 * area);

    let p_l_star = (p_l - 0.5 * dm_l * g_over_area).max(1.0e-30 * p_l);
    let p_r_star = (p_r + 0.5 * dm_r * g_over_area).max(1.0e-30 * p_r);

    let c_l = cs_l.max(1.0e-10);
    let c_r = cs_r.max(1.0e-10);
    let d_l = rho_l.max(1.0e-99);
    let d_r = rho_r.max(1.0e-99);

    let s_l = (u_l - c_l).min(u_r - c_r);
    let s_r = (u_r + c_r).max(u_l + c_l);

    // Mass flux impedances: m_L = rho_L * (S_L - u_L) < 0 and
    // m_R = rho_R * (S_R - u_R) > 0.
    let m_l = d_l * (s_l - u_l);
    let m_r = d_r * (s_r - u_r);
    let denom = (m_r - m_l).max(1.0e-99);

    let u_face = (m_r * u_r - m_l * u_l + p_l_star - p_r_star) / denom;
    let p_face =
        ((m_r * (p_l_star - m_l * u_l) - m_l * (p_r_star - m_r * u_r)) / denom)
            .max(0.0);
    let ke_flux = area * p_face * u_face;

    HllcFaceState {
        p_l_star,
        p_r_star,
        s_l,
        s_r,
        u_face,
        p_face,
        ke_flux,
    }
}

/// Evaluates the surface ($k = 0$) HLLC Riemann boundary state against an
/// external atmospheric boundary pressure $P_{\mathrm{bc}}$, matching
/// `setup_for_surf_Riemann` in `star/private/hydro_riemann.f90`.
///
/// Enforces equal acoustic impedance across the surface boundary
/// ($\rho_R (S_R - u_R) = -\rho_L (S_L - u_L) = \rho_L c_{s,L}$) with zero
/// ghost velocity ($u_R = u_L$), so that:
/// $$u_{\mathrm{face},0} = u_0 +
///   \frac{P_L^* - P_{\mathrm{bc}}}{2 \rho_0 c_{s,0}}, \qquad
///   P_{\mathrm{face},0} = \frac{1}{2}(P_L^* + P_{\mathrm{bc}}).$$
#[must_use]
pub fn eval_hllc_surface_face(
    p_surf_cell: f64,
    rho_surf: f64,
    cs_surf: f64,
    u_surf: f64,
    dm_surf: f64,
    r_surf: f64,
    m_surf: f64,
    cgrav_surf: f64,
    p_bc: f64,
) -> HllcFaceState {
    let r2 = (r_surf * r_surf).max(1.0e-30);
    let area = PI4 * r2;
    let g_over_area = cgrav_surf * m_surf / (r2 * area);

    let p_l_star =
        (p_surf_cell - 0.5 * dm_surf * g_over_area).max(1.0e-30 * p_surf_cell);
    let p_r_star = p_bc.max(0.0);
    let c_l = cs_surf.max(1.0e-10);
    let d_l = rho_surf.max(1.0e-99);

    let s_l = u_surf - c_l;
    let s_r = u_surf + c_l;
    let imp = (d_l * c_l).max(1.0e-99);
    let u_face = u_surf + (p_l_star - p_r_star) / (2.0 * imp);
    let p_face = 0.5 * (p_l_star + p_r_star);
    let ke_flux = area * p_face * u_face;

    HllcFaceState {
        p_l_star,
        p_r_star,
        s_l,
        s_r,
        u_face,
        p_face,
        ke_flux,
    }
}

/// Evaluates the spherical Lagrangian HLLC momentum residual for cell $k$
/// (`do1_momentum_eqn` in `star/private/hydro_momentum.f90`).
///
/// The discrete spherical momentum conservation equation in cell $k$ (bounded
/// by outer face $k$ at radius $r_{\mathrm{outer}}$ and inner face $k+1$ at
/// radius $r_{\mathrm{inner}}$) is:
/// $$\mathcal{R}_{u,k} = \frac{u_k - u_{k,\mathrm{old}}}{\Delta t} +
///   g_{\mathrm{cell},k} - \frac{1}{\Delta m_k}\left[
///   (A_k P_{\mathrm{face},k} - A_{k+1} P_{\mathrm{face},k+1}) -
///   P_{\mathrm{cell},k} (A_k - A_{k+1}) \right],$$
/// where $A_k = 4\pi r_k^2$ and $P_{\mathrm{cell},k}(A_k - A_{k+1})$ is the
/// spherical geometric pressure source term that ensures exact cancellation
/// when $P_{\mathrm{face},k} = P_{\mathrm{face},k+1} = P_{\mathrm{cell},k}$.
#[must_use]
pub fn eval_riemann_momentum_residual(
    u_k: f64,
    u_old_k: f64,
    dt: f64,
    dm_k: f64,
    r_outer: f64,
    r_inner: f64,
    p_face_outer: f64,
    p_face_inner: f64,
    p_cell_k: f64,
    g_cell_k: f64,
) -> f64 {
    let a_outer = PI4 * r_outer * r_outer;
    let a_inner = PI4 * r_inner * r_inner;
    let dp_force = a_outer * (p_face_outer - p_cell_k)
        - a_inner * (p_face_inner - p_cell_k);
    let du_dt = if dt > 0.0 { (u_k - u_old_k) / dt } else { 0.0 };
    du_dt - dp_force / dm_k.max(1.0e-99) + g_cell_k
}

/// Evaluates the full-profile HLLC Riemann face states and cell-centered
/// momentum residuals across all `nz` zones of `model`.
#[must_use]
pub fn eval_riemann_profile(
    model: &StarModel,
    u_cell: &[f64],
    u_old_cell: &[f64],
    p_surf_bc: f64,
    parallel: bool,
) -> (Vec<HllcFaceState>, Vec<f64>) {
    let nz = model.nz;
    if nz == 0 {
        return (Vec::new(), Vec::new());
    }
    let eval_face_k = |k: usize| -> HllcFaceState {
        let p_l = model.peos[k];
        let rho_l = model.rho[k];
        let cs_l = model.csound[k];
        let u_l = u_cell.get(k).copied().unwrap_or(0.0);
        let dm_l = model.dm[k];
        let r_face = model.r[k];
        let m_grav = model.m_grav[k];
        let cgrav = model.cgrav[k];
        if k == 0 {
            eval_hllc_surface_face(
                p_l, rho_l, cs_l, u_l, dm_l, r_face, m_grav, cgrav, p_surf_bc,
            )
        } else {
            let p_r = model.peos[k - 1];
            let rho_r = model.rho[k - 1];
            let cs_r = model.csound[k - 1];
            let u_r = u_cell.get(k - 1).copied().unwrap_or(0.0);
            let dm_r = model.dm[k - 1];
            eval_hllc_face(
                p_l, rho_l, cs_l, u_l, dm_l, p_r, rho_r, cs_r, u_r, dm_r,
                r_face, m_grav, cgrav,
            )
        }
    };

    let faces: Vec<HllcFaceState> = if parallel {
        (0..nz).into_par_iter().map(eval_face_k).collect()
    } else {
        (0..nz).map(eval_face_k).collect()
    };

    let dt = model.dt;
    let r_center = model.r_center;
    let eval_res_k = |k: usize| -> f64 {
        let r_outer = model.r[k];
        let r_inner = if k + 1 < nz { model.r[k + 1] } else { r_center };
        let p_cell = model.peos[k];
        let p_outer = faces[k].p_face;
        let g_outer =
            model.cgrav[k] * model.m_grav[k] / (r_outer * r_outer).max(1.0e-30);
        let p_inner = if k + 1 < nz {
            faces[k + 1].p_face
        } else {
            let a_out = (PI4 * r_outer * r_outer).max(1.0e-30);
            p_outer + 0.5 * model.dm[k] * g_outer / a_out
        };
        let g_inner = if k + 1 < nz {
            model.cgrav[k + 1] * model.m_grav[k + 1]
                / (r_inner * r_inner).max(1.0e-30)
        } else {
            0.0
        };
        let g_cell = 0.5 * (g_outer + g_inner);
        let u_k = u_cell.get(k).copied().unwrap_or(0.0);
        let u_old_k = u_old_cell.get(k).copied().unwrap_or(u_k);
        eval_riemann_momentum_residual(
            u_k,
            u_old_k,
            dt,
            model.dm[k],
            r_outer,
            r_inner,
            p_outer,
            p_inner,
            p_cell,
            g_cell,
        )
    };

    let residuals: Vec<f64> = if parallel {
        (0..nz).into_par_iter().map(eval_res_k).collect()
    } else {
        (0..nz).map(eval_res_k).collect()
    };

    (faces, residuals)
}
