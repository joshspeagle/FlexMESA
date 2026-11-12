// SPDX-License-Identifier: LGPL-3.0-or-later
//! Coupled 3-cell stencil [`StarAdOrder1`] HLLC Riemann contact solver
//! (`do_uface_and_pface`) for `u_flag` hydrodynamics
//! (`star/private/hydro_riemann.f90:323-461`).
//!
//! All face contact states `(u_face_ad[k], p_face_ad[k])` depend exclusively on
//! cells `k - 1` (`_m1`) and `k` (`_00`), so shifting `p_face_ad[k + 1]` and
//! `u_face_ad[k + 1]` via `.shift_p1()` maps `(_m1, _00) -> (_00, _p1)` and
//! stays strictly inside the 3-cell `(k - 1, k, k + 1)` block-tridiagonal
//! Henyey stencil. Cell-centered momentum and total-energy rate assembly live
//! in [`crate::riemann_eqns`] and are re-exported here for callers.

use flexmesa_const::FOUR_PI;
use flexmesa_eos::I_GAMMA1;

use crate::controls::StarError;
use crate::hydro_momentum::eval_mlt_pturb_cell_ad;
use crate::model::StarModel;
pub use crate::riemann_eqns::{
    do1_riemann_momentum_eqn, eval_u_flag_total_energy_rates_ad,
};
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LND_M1, I_LNR_00, I_V_00, I_V_M1,
};

/// Evaluates total cell pressure $P_{\mathrm{tot},k} = P_{\mathrm{eos},k} +
/// P_{\mathrm{turb,MLT},k} + P_{\mathrm{trb,RSP2},k}$ as a [`StarAdOrder1`]
/// (`calc_Ptot_ad_tw` in `star/private/star_utils.f90`).
#[must_use]
pub fn calc_ptot_cell_ad(
    model: &StarModel,
    k: usize,
    shift_m1: bool,
    skip_mlt_pturb: bool,
) -> StarAdOrder1 {
    let mut p_ad = if shift_m1 {
        model.ln_peos_ad[k].shift_m1().exp()
    } else {
        model.ln_peos_ad[k].exp()
    };
    if !skip_mlt_pturb {
        p_ad = p_ad + eval_mlt_pturb_cell_ad(model, k, shift_m1);
    }
    if model.controls.rsp2_flag && model.controls.rsp2.alfap != 0.0 {
        p_ad =
            p_ad + crate::rsp2_ad::eval_rsp2_ptrb_cell_ad(model, k, shift_m1);
    }
    p_ad
}

/// Evaluates HLLC contact velocity `u_face_ad[k]` and contact pressure
/// `p_face_ad[k]` across all faces `k = 0..nz-1` with full 3-cell stencil
/// [`StarAdOrder1`] derivatives (`do_uface_and_Pface` in
/// `star/private/hydro_riemann.f90:323-461`).
///
/// # Errors
/// Returns [`StarError::ConvergenceFailure`] if the acoustic impedance
/// denominator $\rho_R(S_R - u_R) + \rho_L(u_L - S_L)$ becomes non-positive.
pub fn do_uface_and_pface(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    if nz == 0 {
        return Ok(());
    }
    if model.aux.u_face_ad.len() != nz {
        model.aux.u_face_ad.resize(nz, StarAdOrder1::ZERO);
    }
    if model.aux.p_face_ad.len() != nz {
        model.aux.p_face_ad.resize(nz, StarAdOrder1::ZERO);
    }

    model.aux.u_face_ad[0] = StarAdOrder1::variable(model.u[0], I_V_00);
    model.aux.p_face_ad[0] = calc_ptot_cell_ad(model, 0, false, true);

    for k in 1..nz {
        let lnr_k = StarAdOrder1::variable(model.ln_r[k], I_LNR_00);
        let r_ad = lnr_k.exp();
        let r2_ad = (r_ad * r_ad).max(StarAdOrder1::ONE);
        let area_ad = r2_ad * FOUR_PI;

        let p_l_ad = calc_ptot_cell_ad(model, k, false, false);
        let p_r_ad = calc_ptot_cell_ad(model, k - 1, true, false);
        let u_l_ad = StarAdOrder1::variable(model.u[k], I_V_00);
        let u_r_ad = StarAdOrder1::variable(model.u[k - 1], I_V_M1);
        let rho_l_ad = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
        let rho_r_ad = StarAdOrder1::variable(model.lnd[k - 1], I_LND_M1).exp();

        let gamma1_l_ad = StarAdOrder1::from_dlnd_dlnt_00(
            model.gamma1[k],
            model.eos_d_dlnd[k][I_GAMMA1],
            model.eos_d_dlnt[k][I_GAMMA1],
        );
        let gamma1_r_ad = StarAdOrder1::from_dlnd_dlnt_m1(
            model.gamma1[k - 1],
            model.eos_d_dlnd[k - 1][I_GAMMA1],
            model.eos_d_dlnt[k - 1][I_GAMMA1],
        );
        let cs_l_ad = (gamma1_l_ad * p_l_ad / rho_l_ad)
            .max(StarAdOrder1::constant(1.0e-30))
            .sqrt();
        let cs_r_ad = (gamma1_r_ad * p_r_ad / rho_r_ad)
            .max(StarAdOrder1::constant(1.0e-30))
            .sqrt();

        let fp_k = if model.controls.rotation_flag
            && model.controls.use_gravity_rotation_correction
        {
            model.aux.fp_rot.get(k).copied().unwrap_or(1.0)
        } else {
            1.0
        };
        let dpdm_grav =
            -(model.cgrav[k] * fp_k * model.m_grav[k]) / (r2_ad * area_ad);
        let p_l_star = p_l_ad + dpdm_grav * (0.5 * model.dm[k]);
        let p_r_star = p_r_ad - dpdm_grav * (0.5 * model.dm[k - 1]);

        let sl_l = u_l_ad - cs_l_ad;
        let sl_r = u_r_ad - cs_r_ad;
        let s_l = if sl_l.val < sl_r.val { sl_l } else { sl_r };

        let sr_r = u_r_ad + cs_r_ad;
        let sr_l = u_l_ad + cs_l_ad;
        let s_r = if sr_r.val > sr_l.val { sr_r } else { sr_l };

        let denom = rho_r_ad * (s_r - u_r_ad) + rho_l_ad * (u_l_ad - s_l);
        if denom.val <= 0.0 || !denom.val.is_finite() {
            return Err(StarError::ConvergenceFailure(format!(
                "non-positive HLLC contact denominator at face {k}: {}",
                denom.val
            )));
        }

        let s_star = (u_r_ad * rho_r_ad * (s_r - u_r_ad)
            + u_l_ad * rho_l_ad * (u_l_ad - s_l)
            + (p_l_star - p_r_star))
            / denom;
        let p_face_l = rho_l_ad * (u_l_ad - s_l) * (u_l_ad - s_star) + p_l_star;
        let p_face_r = rho_r_ad * (u_r_ad - s_r) * (u_r_ad - s_star) + p_r_star;

        let mut u_f = s_star;
        if k + 1 < nz
            && model.controls.rti_flag
            && model.controls.dlnddt_rti_diffusion_factor > 0.0
            && model.dt > 0.0
        {
            let eta_k = model.aux.eta_rti.get(k).copied().unwrap_or(0.0);
            if eta_k > 0.0 {
                let f_rti = model.controls.dlnddt_rti_diffusion_factor * eta_k
                    / model.dm_bar[k].max(1.0e-30);
                u_f = u_f + area_ad * (rho_l_ad - rho_r_ad) * f_rti;
            }
        }
        model.aux.u_face_ad[k] = u_f;
        model.aux.p_face_ad[k] = (p_face_l + p_face_r) * 0.5;
    }

    if model.aux.p_face_start.len() != nz
        || model.aux.p_face_start.iter().any(|&p| p < 0.0)
    {
        model.aux.p_face_start =
            model.aux.p_face_ad.iter().map(|p| p.val).collect();
        model.aux.u_face_start =
            model.aux.u_face_ad.iter().map(|u| u.val).collect();
    }
    Ok(())
}
