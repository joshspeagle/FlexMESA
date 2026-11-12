// SPDX-License-Identifier: LGPL-3.0-or-later
//! Cell-centered Lagrangian momentum conservation (`do1_dudt_eqn` and
//! `do_surf_Riemann_dudt_eqn`) and conservative HLLC mechanical work /
//! total-energy rate assembly (`eval1_work_u_flag_ad` and
//! `eval_u_flag_total_energy_rates_ad`) for `u_flag` hydrodynamics
//! (`star/private/hydro_riemann.f90:72-290` and
//! `star/private/hydro_energy.f90:582-847`).

use flexmesa_const::FOUR_PI;

use crate::hydro_energy::eval_face_area_time_centered_ad;
use crate::hydro_eqns::HenyeySystem;
use crate::hydro_momentum::eval_p_theta;
use crate::model::StarModel;
use crate::riemann_ad::calc_ptot_cell_ad;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LNR_00, I_V_00, I_V_M1, I_V_P1,
};

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
    let using_v_tc =
        model.controls.is_velocity_time_centered(model.model_number)
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
    let using_v_tc =
        model.controls.is_velocity_time_centered(model.model_number)
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

    let p_theta = if using_v_tc {
        eval_p_theta(model, f_idx)
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
    let dpe_dt = crate::hydro_work::eval_cell_dpe_dt_ad(model, k, dt);

    let work_00 = eval1_work_u_flag_ad(model, k);
    let work_p1 = if k + 1 < model.nz {
        eval1_work_u_flag_ad(model, k + 1).shift_p1()
    } else {
        eval1_work_u_flag_ad(model, model.nz)
    };
    let dwork_dm = (work_00 - work_p1) / model.dm[k].max(1.0e-30);

    dwork_dm + dke_dt + dpe_dt + de_dt
}
