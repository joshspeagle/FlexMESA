// SPDX-License-Identifier: GPL-3.0-or-later
//! Coupled 3-cell stencil [`StarAdOrder1`] HLLC Riemann contact solver
//! (`do_uface_and_pface`) for `u_flag` hydrodynamics
//! (`star/private/hydro_riemann.f90:289-459`).
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
use crate::hydro_energy::eval_pvsc_cell_ad;
use crate::hydro_momentum::{
    eval_cell_peos_time_centered_ad, wrap_d_00, wrap_d_m1, wrap_peos_00,
    wrap_r_00,
};
use crate::model::StarModel;
pub use crate::riemann_eqns::{
    do1_riemann_momentum_eqn, eval_u_flag_total_energy_rates_ad,
};
use crate::rsp2_ad::eval_rsp2_ptrb_cell_ad;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LND_M1, I_LNR_00, I_V_00, I_V_M1,
};

/// Evaluates total cell pressure $P_{\mathrm{tot},k} = P_{\mathrm{eos},k} +
/// P_{\mathrm{vsc},k} + P_{\mathrm{trb,RSP2},k} + P_{\mathrm{turb,MLT},k}$ with
/// optional pressure velocity time-centering as a [`StarAdOrder1`]
/// (`calc_Ptot_ad_tw` in `star/private/star_utils.f90:3371-3438`).
#[must_use]
pub fn calc_ptot_cell_ad(
    model: &StarModel,
    k: usize,
    shift_m1: bool,
    skip_mlt_pturb: bool,
) -> StarAdOrder1 {
    let (mut p_ad, p_theta) =
        eval_cell_peos_time_centered_ad(model, k, shift_m1);
    if model.controls.use_pvsc_art_visc && model.controls.v_flag {
        let mut pvsc = eval_pvsc_cell_ad(model, k);
        if shift_m1 {
            pvsc = pvsc.shift_m1();
        }
        p_ad = p_ad + pvsc;
    }
    if model.controls.rsp2_flag && model.controls.rsp2.alfap != 0.0 {
        p_ad = p_ad + eval_rsp2_ptrb_cell_ad(model, k, shift_m1);
    }
    if !skip_mlt_pturb && model.controls.mlt_pturb_factor > 0.0 && k > 0 {
        let vc_old = model.aux.mlt_vc_old.get(k).copied().unwrap_or(0.0);
        if vc_old > 0.0 {
            let vc2_fac =
                model.controls.mlt_pturb_factor * flexmesa_math::pow2(vc_old);
            let rho_00 = if model.rho[k] > 0.0 {
                wrap_d_00(model, k)
            } else {
                StarAdOrder1::variable(model.lnd[k], I_LND_00).exp()
            };
            let rho_m1 = if model.rho[k - 1] > 0.0 {
                wrap_d_m1(model, k)
            } else {
                StarAdOrder1::variable(model.lnd[k - 1], I_LND_M1).exp()
            };
            let (alfa, beta) = model.get_face_weights(k);
            let rho_face = alfa * rho_00 + beta * rho_m1;
            let mut mlt_pturb = (vc2_fac / 3.0) * rho_face;
            if p_theta < 1.0 {
                let rho_s0 = model
                    .lnd_start
                    .get(k)
                    .copied()
                    .map_or(rho_00.val, flexmesa_math::exp);
                let rho_sm1 = model
                    .lnd_start
                    .get(k - 1)
                    .copied()
                    .map_or(rho_m1.val, flexmesa_math::exp);
                let mlt_pturb_start = vc2_fac * (rho_sm1 + rho_s0) / 6.0;
                mlt_pturb =
                    p_theta * mlt_pturb + (1.0 - p_theta) * mlt_pturb_start;
            }
            if shift_m1 {
                mlt_pturb = mlt_pturb.shift_m1();
            }
            p_ad = p_ad + mlt_pturb;
        }
    }
    p_ad
}

/// Evaluates HLLC contact velocity `u_face_ad[k]` and contact pressure
/// `p_face_ad[k]` across all faces `k = 0..nz-1` with full 3-cell stencil
/// [`StarAdOrder1`] derivatives (`do_uface_and_Pface` in
/// `star/private/hydro_riemann.f90:289-459`).
///
/// # Errors
/// Returns [`StarError::ConvergenceFailure`] if the acoustic impedance
/// denominator $\rho_R(S_R - u_R) + \rho_L(u_L - S_L)$ becomes zero or
/// non-finite.
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
    if model.aux.p_face_start.len() != nz {
        model.aux.p_face_start.resize(nz, -1.0);
    }
    if model.aux.u_face_start.len() != nz {
        model.aux.u_face_start.resize(nz, 0.0);
    }

    // `hydro_riemann.f90:344-348`: surface face returns before updating
    // `u_face_start` / `P_face_start`.
    model.aux.u_face_ad[0] = StarAdOrder1::variable(model.u[0], I_V_00);
    model.aux.p_face_ad[0] = if model.peos[0] > 0.0 {
        wrap_peos_00(model, 0)
    } else {
        model.ln_peos_ad[0].exp()
    };

    for k in 1..nz {
        let r_ad = if model.r[k] > 0.0 {
            wrap_r_00(model, k)
        } else {
            StarAdOrder1::variable(model.ln_r[k], I_LNR_00).exp()
        };
        let r2_ad = r_ad * r_ad;
        let area_ad = r2_ad * FOUR_PI;

        let p_l_ad = calc_ptot_cell_ad(model, k, false, false);
        let p_r_ad = calc_ptot_cell_ad(model, k - 1, true, false);
        let u_l_ad = StarAdOrder1::variable(model.u[k], I_V_00);
        let u_r_ad = StarAdOrder1::variable(model.u[k - 1], I_V_M1);
        let rho_l_ad = if model.rho[k] > 0.0 {
            wrap_d_00(model, k)
        } else {
            StarAdOrder1::variable(model.lnd[k], I_LND_00).exp()
        };
        let rho_r_ad = if model.rho[k - 1] > 0.0 {
            wrap_d_m1(model, k)
        } else {
            StarAdOrder1::variable(model.lnd[k - 1], I_LND_M1).exp()
        };

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
        let cs_l_ad = (gamma1_l_ad * p_l_ad / rho_l_ad).sqrt();
        let cs_r_ad = (gamma1_r_ad * p_r_ad / rho_r_ad).sqrt();

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
        if denom.val == 0.0 || !denom.val.is_finite() {
            return Err(StarError::ConvergenceFailure(format!(
                "bad HLLC contact denominator at face {k}: {}",
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
                let dm_bar = if model.dm_bar[k] == 0.0 {
                    1.0e-30
                } else {
                    model.dm_bar[k]
                };
                let f_rti =
                    model.controls.dlnddt_rti_diffusion_factor * eta_k / dm_bar;
                u_f = u_f + area_ad * (rho_l_ad - rho_r_ad) * f_rti;
            }
        }
        if model.controls.rsp2_flag {
            if let Some(&uq) = model.aux.uq_ad.get(k) {
                u_f = u_f + uq;
            }
        }
        model.aux.u_face_ad[k] = u_f;
        model.aux.p_face_ad[k] = (p_face_l + p_face_r) * 0.5;

        if model.aux.p_face_start[k] < 0.0 {
            model.aux.u_face_start[k] = model.aux.u_face_ad[k].val;
            model.aux.p_face_start[k] = model.aux.p_face_ad[k].val;
        }
    }

    Ok(())
}
