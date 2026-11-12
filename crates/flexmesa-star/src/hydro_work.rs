// SPDX-License-Identifier: LGPL-3.0-or-later
//! Spherical face mechanical work (`eval_dwork`, `eval1_work`,
//! `eval_simple_PdV_work`) and cell-specific kinetic/gravitational-potential
//! energy rates (`cell_specific_KE_qp`, `cell_specific_PE_qp`) for `v_flag` and
//! `u_flag` hydrodynamics (`star/private/hydro_energy.f90:582-902` and
//! `star/private/star_utils.f90:2197-2338`).

use flexmesa_const::{FOUR_PI, LN10};

use crate::hydro_energy::{eval_face_area_time_centered_ad, eval_pvsc_cell_ad};
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LND_M1, I_LNR_00, I_LNR_P1, I_V_00,
    I_V_P1,
};

/// Evaluates the cell-specific gravitational potential energy rate $de_{\mathrm{pot}}/dt$
/// in $\mathrm{erg\,g^{-1}\,s^{-1}}$ for cell `k` (`cell_specific_PE_qp` and
/// `cell_start_specific_PE_qp` in `star/private/star_utils.f90:2245-2338`).
#[must_use]
pub fn eval_cell_dpe_dt_ad(
    model: &StarModel,
    k: usize,
    dt: f64,
) -> StarAdOrder1 {
    let mc = if model.controls.use_mass_corrections {
        model.aux.mass_correction.get(k).copied().unwrap_or(1.0)
    } else {
        1.0
    };
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
    (pe_ad - pe_start) / dt
}

/// Evaluates the face-velocity kinetic energy rate $de_K/dt$ in
/// $\mathrm{erg\,g^{-1}\,s^{-1}}$ for `v_flag` cell `k` (`cell_specific_KE_qp` in
/// `star/private/star_utils.f90:2215-2228`).
#[must_use]
pub fn eval_v_flag_dke_dt_ad(
    model: &StarModel,
    k: usize,
    dt: f64,
) -> StarAdOrder1 {
    let mc = if model.controls.use_mass_corrections {
        model.aux.mass_correction.get(k).copied().unwrap_or(1.0)
    } else {
        1.0
    };
    let v_00 = StarAdOrder1::variable(model.v[k], I_V_00);
    let v_p1 = if k + 1 < model.nz {
        StarAdOrder1::variable(model.v[k + 1], I_V_P1)
    } else {
        StarAdOrder1::constant(model.v_center)
    };
    let vs_00 = model.v_start.get(k).copied().unwrap_or(model.v[k]);
    let vs_p1 = if k + 1 < model.nz {
        model.v_start.get(k + 1).copied().unwrap_or(model.v[k + 1])
    } else {
        model.v_center
    };
    let ke_ad = (v_00 * v_00 + v_p1 * v_p1) * (0.25 * mc);
    let ke_start = 0.25 * mc * (vs_00 * vs_00 + vs_p1 * vs_p1);
    (ke_ad - ke_start) / dt
}

/// Evaluates spherical volume-flux rate $A_k v_{c,k}$ in $\mathrm{cm^3\,s^{-1}}$
/// at face `f_idx` in face `f_idx`'s local stencil (`eval1_A_times_v_face_ad`
/// in `star/private/hydro_energy.f90:816-847`).
#[must_use]
pub fn eval1_a_times_v_face_v_flag_ad(
    model: &StarModel,
    f_idx: usize,
    vel_tc: bool,
) -> StarAdOrder1 {
    if f_idx >= model.nz {
        return StarAdOrder1::constant(
            FOUR_PI * model.r_center * model.r_center * model.v_center,
        );
    }
    let lnr_f = StarAdOrder1::variable(model.ln_r[f_idx], I_LNR_00);
    let lnr_s_f = model
        .ln_r_start
        .get(f_idx)
        .copied()
        .unwrap_or(model.ln_r[f_idx]);
    let area_f = eval_face_area_time_centered_ad(lnr_f, lnr_s_f, vel_tc);
    let v_f = StarAdOrder1::variable(model.v[f_idx], I_V_00);
    let vc_f = if vel_tc {
        let vs_f = model.v_start.get(f_idx).copied().unwrap_or(model.v[f_idx]);
        (v_f + vs_f) * 0.5
    } else {
        v_f
    };
    area_f * vc_f
}

/// Evaluates face mechanical power $W_f = A_f v_{c,f} P_{\mathrm{face},f}$ in
/// $\mathrm{erg\,s^{-1}}$ and composition-derivative prefactors
/// `(fac_dxa00, fac_dxam1)` multiplying `dln_peos_dxa[f_idx, j]` and
/// `dln_peos_dxa[f_idx - 1, j]` (`eval1_work` in
/// `star/private/hydro_energy.f90:634-813`).
#[must_use]
pub fn eval1_work_v_flag_ad(
    model: &StarModel,
    f_idx: usize,
    skip_peos: bool,
    vel_tc: bool,
) -> (StarAdOrder1, f64, f64) {
    if f_idx >= model.nz {
        let p_in = model.peos_start.last().copied().unwrap_or(0.0);
        let work_center =
            FOUR_PI * model.r_center * model.r_center * p_in * model.v_center;
        return (StarAdOrder1::constant(work_center), 0.0, 0.0);
    }
    let av_ad = eval1_a_times_v_face_v_flag_ad(model, f_idx, vel_tc);
    let (alfa, beta) = if f_idx > 0 {
        model.get_face_weights(f_idx)
    } else {
        (1.0, 0.0)
    };
    let logt_f = model.ln_t[f_idx] / LN10;
    let inc_p_tc = vel_tc
        && model.controls.include_p_in_velocity_time_centering
        && logt_f
            <= model
                .controls
                .max_logt_for_include_p_and_l_in_velocity_time_centering;
    let p_theta = if inc_p_tc {
        model.controls.p_theta_for_velocity_time_centering
    } else {
        1.0
    };

    let (peos_ad, fac_dxa00, fac_dxam1) = if skip_peos {
        (StarAdOrder1::ZERO, 0.0, 0.0)
    } else {
        let p_s_l = model
            .peos_start
            .get(f_idx)
            .copied()
            .unwrap_or(model.peos[f_idx]);
        let pl_ad =
            model.ln_peos_ad[f_idx].exp() * p_theta + p_s_l * (1.0 - p_theta);
        let pr_ad = if f_idx > 0 {
            let p_s_r = model
                .peos_start
                .get(f_idx - 1)
                .copied()
                .unwrap_or(model.peos[f_idx - 1]);
            model.ln_peos_ad[f_idx - 1].shift_m1().exp() * p_theta
                + p_s_r * (1.0 - p_theta)
        } else {
            StarAdOrder1::ZERO
        };
        let av_val = av_ad.val;
        let f_00 = av_val * alfa * p_theta * model.peos[f_idx];
        let f_m1 = if f_idx > 0 {
            av_val * beta * p_theta * model.peos[f_idx - 1]
        } else {
            0.0
        };
        (pl_ad * alfa + pr_ad * beta, f_00, f_m1)
    };

    let pvsc_ad = if model.controls.use_pvsc_art_visc {
        let pvsc_l = eval_pvsc_cell_ad(model, f_idx);
        let pvsc_r = if f_idx > 0 {
            eval_pvsc_cell_ad(model, f_idx - 1).shift_m1()
        } else {
            StarAdOrder1::ZERO
        };
        pvsc_l * alfa + pvsc_r * beta
    } else {
        StarAdOrder1::ZERO
    };

    let ptrb_ad =
        if model.controls.rsp2_flag && model.controls.rsp2.alfap != 0.0 {
            let ptrb_l =
                crate::rsp2_ad::eval_rsp2_ptrb_cell_ad(model, f_idx, false);
            let ptrb_r = if f_idx > 0 {
                crate::rsp2_ad::eval_rsp2_ptrb_cell_ad(model, f_idx - 1, true)
            } else {
                StarAdOrder1::ZERO
            };
            ptrb_l * alfa + ptrb_r * beta
        } else {
            StarAdOrder1::ZERO
        };

    let mlt_pturb_ad = if model.controls.mlt_pturb_factor > 0.0 {
        let vc_old = model.aux.mlt_vc_old.get(f_idx).copied().unwrap_or(0.0);
        if vc_old > 0.0 {
            let rho_00 =
                StarAdOrder1::variable(model.lnd[f_idx], I_LND_00).exp();
            let rho_face = if f_idx == 0 {
                rho_00
            } else {
                let rho_m1 =
                    StarAdOrder1::variable(model.lnd[f_idx - 1], I_LND_M1)
                        .exp();
                rho_00 * alfa + rho_m1 * beta
            };
            rho_face
                * ((1.0 / 3.0)
                    * model.controls.mlt_pturb_factor
                    * vc_old
                    * vc_old)
        } else {
            StarAdOrder1::ZERO
        }
    } else {
        StarAdOrder1::ZERO
    };

    let p_face_ad = peos_ad + pvsc_ad + ptrb_ad + mlt_pturb_ad;
    (av_ad * p_face_ad, fac_dxa00, fac_dxam1)
}

/// Evaluates spherical mechanical work divergence
/// $dW/dm = (W_k - W_{k+1}) / \Delta m_k$ and 3-cell composition-derivative
/// prefactors `(fac_dm_dxam1, fac_dm_dxa00, fac_dm_dxap1)` (`eval_dwork` in
/// `star/private/hydro_energy.f90:582-630`).
#[must_use]
pub fn eval_v_flag_dwork_dm_ad(
    model: &StarModel,
    k: usize,
    skip_peos: bool,
    vel_tc: bool,
) -> (StarAdOrder1, f64, f64, f64) {
    let dm_k = model.dm[k].max(1.0e-30);
    let (work_00, fac_00_dxa00, fac_00_dxam1) =
        eval1_work_v_flag_ad(model, k, skip_peos, vel_tc);
    let (work_p1_raw, fac_p1_dxap1, fac_p1_dxa00) =
        eval1_work_v_flag_ad(model, k + 1, skip_peos, vel_tc);
    let work_p1 = if k + 1 < model.nz {
        work_p1_raw.shift_p1()
    } else {
        work_p1_raw
    };
    (
        (work_00 - work_p1) / dm_k,
        fac_00_dxam1 / dm_k,
        (fac_00_dxa00 - fac_p1_dxa00) / dm_k,
        -fac_p1_dxap1 / dm_k,
    )
}

/// Evaluates `eval_simple_PdV_work` divided by $\Delta m_k$ (`P_tot * (A_k v_{c,k}
/// - A_{k+1} v_{c,k+1}) / dm_k`) and diagonal composition prefactor `fac_dm_dxa00`
/// (`star/private/hydro_energy.f90:850-902`).
#[must_use]
pub fn eval_v_flag_simple_pdv_dm_ad(
    model: &StarModel,
    k: usize,
    skip_peos: bool,
    vel_tc: bool,
) -> (StarAdOrder1, f64) {
    let dm_k = model.dm[k].max(1.0e-30);
    let av_00 = eval1_a_times_v_face_v_flag_ad(model, k, vel_tc);
    let av_p1 = if k + 1 < model.nz {
        eval1_a_times_v_face_v_flag_ad(model, k + 1, vel_tc).shift_p1()
    } else {
        eval1_a_times_v_face_v_flag_ad(model, model.nz, vel_tc)
    };
    let dv_dm = (av_00 - av_p1) / dm_k;

    let (mut p_tot_ad, p_theta) = if skip_peos {
        (StarAdOrder1::ZERO, 0.0)
    } else {
        crate::hydro_momentum::eval_cell_peos_time_centered_ad(model, k, false)
    };
    if model.controls.use_pvsc_art_visc {
        p_tot_ad = p_tot_ad + eval_pvsc_cell_ad(model, k);
    }
    if model.controls.rsp2_flag && model.controls.rsp2.alfap != 0.0 {
        p_tot_ad =
            p_tot_ad + crate::rsp2_ad::eval_rsp2_ptrb_cell_ad(model, k, false);
    }
    if k > 0 && model.controls.mlt_pturb_factor > 0.0 {
        let vc_old = model.aux.mlt_vc_old.get(k).copied().unwrap_or(0.0);
        if vc_old > 0.0 {
            let rho_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
            p_tot_ad = p_tot_ad
                + rho_00
                    * ((1.0 / 3.0)
                        * model.controls.mlt_pturb_factor
                        * vc_old
                        * vc_old);
        }
    }
    let fac_dm_dxa00 = if skip_peos {
        0.0
    } else {
        p_theta * model.peos[k] * dv_dm.val
    };
    (p_tot_ad * dv_dm, fac_dm_dxa00)
}
