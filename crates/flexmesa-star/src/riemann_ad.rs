// SPDX-License-Identifier: LGPL-3.0-or-later
//! Coupled 3-cell stencil [`StarAdOrder1`] HLLC Riemann contact solver
//! (`do_uface_and_pface`) and cell-centered Lagrangian momentum conservation
//! equation (`do1_dudt_eqn` and `do_surf_Riemann_dudt_eqn`) for `u_flag`
//! hydrodynamics (`star/private/hydro_riemann.f90:52-461`).
//!
//! All face contact states `(u_face_ad[k], p_face_ad[k])` depend exclusively on
//! cells `k - 1` (`_m1`) and `k` (`_00`), so shifting `p_face_ad[k + 1]` and
//! `u_face_ad[k + 1]` via `.shift_p1()` maps `(_m1, _00) -> (_00, _p1)` and
//! stays strictly inside the 3-cell `(k - 1, k, k + 1)` block-tridiagonal
//! Henyey stencil.

use flexmesa_const::{FOUR_PI, LN10};
use flexmesa_eos::I_GAMMA1;

use crate::controls::StarError;
use crate::hydro_energy::eval_face_area_time_centered_ad;
use crate::hydro_eqns::HenyeySystem;
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LND_M1, I_LNR_00, I_LNR_P1, I_V_00,
    I_V_M1, I_V_P1,
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
    if !skip_mlt_pturb && model.controls.mlt_pturb_factor > 0.0 {
        let vc_old = model.aux.mlt_vc_old.get(k).copied().unwrap_or(0.0);
        if vc_old > 0.0 {
            let rho_ad = if shift_m1 {
                StarAdOrder1::variable(model.lnd[k], I_LND_M1).exp()
            } else {
                StarAdOrder1::variable(model.lnd[k], I_LND_00).exp()
            };
            p_ad = p_ad
                + rho_ad
                    * ((1.0 / 3.0)
                        * model.controls.mlt_pturb_factor
                        * vc_old
                        * vc_old);
        }
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

    if model.aux.p_face_start.len() != nz {
        model.aux.p_face_start =
            model.aux.p_face_ad.iter().map(|p| p.val).collect();
        model.aux.u_face_start =
            model.aux.u_face_ad.iter().map(|u| u.val).collect();
    }
    Ok(())
}

fn eval_face_inv_r2_ad(
    model: &StarModel,
    k: usize,
    using_v_tc: bool,
) -> StarAdOrder1 {
    let lnr_k = StarAdOrder1::variable(model.ln_r[k], I_LNR_00);
    let r_k = lnr_k.exp();
    if using_v_tc {
        let lnr_s = model.ln_r_start.get(k).copied().unwrap_or(model.ln_r[k]);
        let r_s = lnr_s.exp().max(1.0);
        StarAdOrder1::ONE / (r_k * r_s).max(StarAdOrder1::ONE)
    } else {
        StarAdOrder1::ONE / (r_k * r_k).max(StarAdOrder1::ONE)
    }
}

/// Evaluates the cell-centered Lagrangian momentum residual `do1_dudt_eqn`
/// (`star/private/hydro_riemann.f90:72-290`) for zone `k` (`k == 0` uses
/// `p_surf_opt`; interior zones `k >= 1` use `p_face_ad[k]`).
pub fn do1_riemann_momentum_eqn(
    model: &StarModel,
    k: usize,
    p_surf_opt: Option<StarAdOrder1>,
    sys: &mut HenyeySystem,
) {
    if k == 0 && p_surf_opt.is_none() {
        return;
    }
    let nz = model.nz;
    let dt = model.dt.max(1.0e-30);
    let using_v_tc = model.controls.using_velocity_time_centering
        && model.dt > 0.0
        && !model.ln_r_start.is_empty();

    let lnr_00 = StarAdOrder1::variable(model.ln_r[k], I_LNR_00);
    let lnr_s_00 = model.ln_r_start.get(k).copied().unwrap_or(model.ln_r[k]);
    let area_00 = eval_face_area_time_centered_ad(lnr_00, lnr_s_00, using_v_tc);
    let inv_r2_00 = eval_face_inv_r2_ad(model, k, using_v_tc);

    let (area_p1, inv_r2_p1) = if k + 1 < nz {
        let lnr_p1_unshifted =
            StarAdOrder1::variable(model.ln_r[k + 1], I_LNR_00);
        let lnr_s_p1 = model
            .ln_r_start
            .get(k + 1)
            .copied()
            .unwrap_or(model.ln_r[k + 1]);
        (
            eval_face_area_time_centered_ad(
                lnr_p1_unshifted,
                lnr_s_p1,
                using_v_tc,
            )
            .shift_p1(),
            eval_face_inv_r2_ad(model, k + 1, using_v_tc).shift_p1(),
        )
    } else {
        (StarAdOrder1::ZERO, StarAdOrder1::ZERO)
    };

    let p_out = if k == 0 {
        p_surf_opt.unwrap_or(model.ln_peos_surf_ad.exp())
    } else {
        model
            .aux
            .p_face_ad
            .get(k)
            .copied()
            .unwrap_or_else(|| model.ln_peos_ad[k].exp())
    };
    let flux_out = p_out * area_00;
    let flux_in = if k + 1 < nz {
        let p_in = model
            .aux
            .p_face_ad
            .get(k + 1)
            .copied()
            .unwrap_or_else(|| model.ln_peos_ad[k + 1].exp())
            .shift_p1();
        p_in * area_p1
    } else {
        StarAdOrder1::ZERO
    };

    let p_cell = calc_ptot_cell_ad(model, k, false, false);
    let geometry_source = if k + 1 == nz {
        p_cell * area_00
    } else {
        p_cell * (area_00 - area_p1)
    };

    let fp_00 = if model.controls.rotation_flag
        && model.controls.use_gravity_rotation_correction
    {
        model.aux.fp_rot.get(k).copied().unwrap_or(1.0)
    } else {
        1.0
    };
    let dm_half = 0.5 * model.dm[k];
    let gs_r =
        inv_r2_00 * (-model.cgrav[k] * fp_00 * model.m_grav[k] * dm_half);
    let gs_l = if k + 1 < nz {
        let fp_p1 = if model.controls.rotation_flag
            && model.controls.use_gravity_rotation_correction
        {
            model.aux.fp_rot.get(k + 1).copied().unwrap_or(1.0)
        } else {
            1.0
        };
        inv_r2_p1
            * (-model.cgrav[k + 1] * fp_p1 * model.m_grav[k + 1] * dm_half)
    } else {
        StarAdOrder1::ZERO
    };
    let gravity_source = gs_l + gs_r;

    let u_00 = StarAdOrder1::variable(model.u[k], I_V_00);
    let diffusion_source = if model.controls.rti_flag
        && model.controls.dudt_rti_diffusion_factor > 0.0
    {
        let fac = model.controls.dudt_rti_diffusion_factor;
        let term_00 = if k > 0 {
            let sig00 = model.aux.sig_rti.get(k).copied().unwrap_or(0.0) * fac;
            if sig00 > 0.0 {
                let u_m1 = StarAdOrder1::variable(model.u[k - 1], I_V_M1);
                (u_m1 - u_00) * sig00
            } else {
                StarAdOrder1::ZERO
            }
        } else {
            StarAdOrder1::ZERO
        };
        let term_p1 = if k + 1 < nz {
            let sigp1 =
                model.aux.sig_rti.get(k + 1).copied().unwrap_or(0.0) * fac;
            if sigp1 > 0.0 {
                let u_p1 = StarAdOrder1::variable(model.u[k + 1], I_V_P1);
                (u_00 - u_p1) * sigp1
            } else {
                StarAdOrder1::ZERO
            }
        } else {
            StarAdOrder1::ZERO
        };
        term_00 - term_p1
    } else {
        StarAdOrder1::ZERO
    };

    let dm_k = model.dm[k].max(1.0e-30);
    let mut dudt_expected = (flux_in - flux_out
        + geometry_source
        + gravity_source
        + diffusion_source)
        / dm_k;
    if let Some(&eg) = model.aux.extra_grav.get(k) {
        if eg.val != 0.0 || eg.d1.iter().any(|&d| d != 0.0) {
            dudt_expected = dudt_expected + eg;
        }
    }
    if model.controls.drag_coefficient > 0.0
        && model.q[k] > model.controls.min_q_for_drag
    {
        dudt_expected =
            dudt_expected - u_00 * (model.controls.drag_coefficient / dt);
    }

    let u_start_k = model.u_start.get(k).copied().unwrap_or(model.u[k]);
    let dudt_actual = (u_00 - u_start_k) / dt;

    let e_start_k = model
        .energy_start
        .get(k)
        .copied()
        .unwrap_or(model.energy[k]);
    let ie_plus_ke = (e_start_k + 0.5 * u_start_k * u_start_k).max(1.0);
    let cs_start_k = model
        .aux
        .csound_start
        .get(k)
        .copied()
        .filter(|&c| c > 0.0)
        .unwrap_or(model.csound[k])
        .max(1.0);
    let mut scal = dt * u_start_k.abs().max(cs_start_k) / ie_plus_ke;
    if k == 0 {
        scal *= 1.0e-2;
    }

    let resid = (dudt_expected - dudt_actual) * scal;
    sys.save_eqn_residual(&model.layout, k, model.layout.i_dv_dt, &resid);
}

fn eval1_work_u_flag_ad(model: &StarModel, f_idx: usize) -> StarAdOrder1 {
    if f_idx >= model.nz {
        let p_in = model.peos_start.last().copied().unwrap_or(0.0);
        return StarAdOrder1::constant(
            FOUR_PI * model.r_center * model.r_center * p_in * model.v_center,
        );
    }
    let using_v_tc = model.controls.using_velocity_time_centering
        && model.dt > 0.0
        && !model.ln_r_start.is_empty();
    let lnr_f = StarAdOrder1::variable(model.ln_r[f_idx], I_LNR_00);
    let lnr_s_f = model
        .ln_r_start
        .get(f_idx)
        .copied()
        .unwrap_or(model.ln_r[f_idx]);
    let area_f = eval_face_area_time_centered_ad(lnr_f, lnr_s_f, using_v_tc);

    let uf_ad = model
        .aux
        .u_face_ad
        .get(f_idx)
        .copied()
        .unwrap_or_else(|| StarAdOrder1::variable(model.u[f_idx], I_V_00));
    let u_face_tc = if using_v_tc {
        let uf_s = model
            .aux
            .u_face_start
            .get(f_idx)
            .copied()
            .unwrap_or(uf_ad.val);
        (uf_ad + uf_s) * 0.5
    } else {
        uf_ad
    };

    let logt_f = model.ln_t[f_idx] / LN10;
    let p_theta = if using_v_tc
        && model.controls.include_p_in_velocity_time_centering
        && logt_f
            <= model
                .controls
                .max_logt_for_include_p_and_l_in_velocity_time_centering
    {
        model.controls.p_theta_for_velocity_time_centering
    } else {
        1.0
    };
    let pf_ad = model
        .aux
        .p_face_ad
        .get(f_idx)
        .copied()
        .unwrap_or_else(|| model.ln_peos_ad[f_idx].exp());
    let pf_s = model
        .aux
        .p_face_start
        .get(f_idx)
        .copied()
        .unwrap_or(pf_ad.val);
    let p_face_tc = pf_ad * p_theta + pf_s * (1.0 - p_theta);

    area_f * u_face_tc * p_face_tc
}

/// Evaluates the conservative total-energy rate sum
/// $dW/dm + de_K/dt + de_{\mathrm{pot}}/dt + de/dt$ for `u_flag` cell `k`
/// (`eval_dwork`, `get_dke_dt_dpe_dt`, and `setup_de_dt_and_friends` in
/// `star/private/hydro_energy.f90:582-847` and `star_utils.f90:2210-2383`).
#[must_use]
pub fn eval_u_flag_total_energy_rates_ad(
    model: &StarModel,
    k: usize,
    dt: f64,
) -> StarAdOrder1 {
    let de_dt = (model.ln_e_ad[k].exp() - model.energy_start[k]) / dt;
    let mc = if model.controls.use_mass_corrections {
        model.aux.mass_correction.get(k).copied().unwrap_or(1.0)
    } else {
        1.0
    };

    let u_00 = StarAdOrder1::variable(model.u[k], I_V_00);
    let u_st = model.u_start.get(k).copied().unwrap_or(model.u[k]);
    let dke_dt = (u_00 * u_00 - u_st * u_st) * (0.5 * mc / dt);

    let r_00 = StarAdOrder1::variable(model.ln_r[k], I_LNR_00)
        .exp()
        .max(StarAdOrder1::ONE);
    let grav00 = -(model.cgrav[k] * model.m_grav[k]) / r_00;
    let gravp1 = if k + 1 < model.nz {
        let r_p1 = StarAdOrder1::variable(model.ln_r[k + 1], I_LNR_P1)
            .exp()
            .max(StarAdOrder1::ONE);
        -(model.cgrav[k + 1] * model.m_grav[k + 1]) / r_p1
    } else if model.r_center > 0.0 {
        StarAdOrder1::constant(
            -model.cgrav[model.nz - 1] * model.m_center / model.r_center,
        )
    } else {
        StarAdOrder1::ZERO
    };
    let pe_ad = (grav00 + gravp1) * (0.5 * mc);

    let rs_00 = model
        .ln_r_start
        .get(k)
        .copied()
        .unwrap_or(model.ln_r[k])
        .exp()
        .max(1.0);
    let grav00_s = -model.cgrav[k] * model.m_grav[k] / rs_00;
    let gravp1_s = if k + 1 < model.nz {
        let rs_p1 = model
            .ln_r_start
            .get(k + 1)
            .copied()
            .unwrap_or(model.ln_r[k + 1])
            .exp()
            .max(1.0);
        -model.cgrav[k + 1] * model.m_grav[k + 1] / rs_p1
    } else if model.r_center > 0.0 {
        -model.cgrav[model.nz - 1] * model.m_center / model.r_center
    } else {
        0.0
    };
    let pe_start = 0.5 * mc * (grav00_s + gravp1_s);
    let dpe_dt = (pe_ad - pe_start) / dt;

    let work_00 = eval1_work_u_flag_ad(model, k);
    let work_p1 = if k + 1 < model.nz {
        eval1_work_u_flag_ad(model, k + 1).shift_p1()
    } else {
        eval1_work_u_flag_ad(model, model.nz)
    };
    let dwork_dm = (work_00 - work_p1) / model.dm[k].max(1.0e-30);

    dwork_dm + dke_dt + dpe_dt + de_dt
}
