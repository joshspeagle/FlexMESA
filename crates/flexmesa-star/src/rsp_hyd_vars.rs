// SPDX-License-Identifier: LGPL-3.0-or-later
//! Per-zone thermodynamic, geometric, superadiabatic, artificial-viscosity, and
//! turbulent-pressure evaluation with analytical Jacobian partials for the
//! nonlinear `RSP` `HYD` solver (`star/private/rsp_step.f90:698-1610`).

use flexmesa_const::PI;

use crate::controls::StarError;
use crate::model::StarModel;
use crate::rsp_def::RspDerivedConsts;
use crate::rsp_eos_kap::{eval_mesa_eos_and_kap, RspChemCache};
use crate::rsp_grid::vol_shell_qp;
use crate::rsp_hyd_turb::{calc_chi, calc_eq, calc_source_sink};
pub use crate::rsp_hyd_work::{zero_boundaries, RspHydWork};

/// Evaluates all zone and face microphysics, pressures, eddy viscosity, and
/// turbulent source/sink partials (`eval_vars` in `rsp_step.f90:698-741`).
pub fn eval_vars(
    model: &mut StarModel,
    chem: &RspChemCache,
    dc: &RspDerivedConsts,
    wk: &mut RspHydWork,
    iter: usize,
    i_min: usize,
    i_max: usize,
) -> Result<(), StarError> {
    for i in i_min..=i_max {
        do1_specific_volume(model, dc, wk, i)?;
        do1_eos_and_kap(model, chem, wk, i)?;
        calc_prad(model, wk, i);
    }
    for i in i_min..=i_max {
        calc_hp_face(model, wk, i);
        calc_y_face(model, dc, wk, i);
        calc_pii_face(model, dc, wk, i);
        calc_pvsc(model, dc, wk, i);
    }
    if iter == 1 && model.controls.rsp.do_check_omega {
        for i in i_min..=i_max {
            check_omega(model, dc, wk, i);
        }
    }
    for i in i_min..=i_max {
        calc_pturb(model, dc, wk, i);
        calc_chi(model, dc, wk, i);
        calc_eq(model, dc, wk, i);
        calc_source_sink(model, dc, wk, i);
    }
    zero_boundaries(model, dc, wk);
    Ok(())
}

/// Evaluates shell specific volume $V_k$ with 106-bit `vol_shell_qp` precision
/// (`do1_specific_volume` in `rsp_step.f90:757-782`).
pub fn do1_specific_volume(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &mut RspHydWork,
    i: usize,
) -> Result<(), StarError> {
    let k0 = wk.nzn - i;
    let t1 = dc.p43 / model.dm[k0];
    let vol = if i == 1 {
        wk.dvol_dr_in[i] = 0.0;
        t1 * (model.r[k0].powi(3) - model.r_center.powi(3))
    } else {
        wk.dvol_dr_in[i] = -3.0 * t1 * model.r[k0 + 1].powi(2);
        vol_shell_qp(t1, model.r[k0], model.r[k0 + 1])
    };
    if !(vol > 0.0 && vol.is_finite()) {
        return Err(StarError::ConvergenceFailure(format!(
            "RSP HYD non-positive Vol={vol} at zone i={i} (k0={k0})"
        )));
    }
    model.aux.rsp.vol_rsp[k0] = vol;
    wk.dvol_dr_00[i] = 3.0 * t1 * model.r[k0].powi(2);
    Ok(())
}

/// Evaluates `eosDT` and Rosseland opacity plus radial chain partials
/// (`do1_eos_and_kap` in `rsp_step.f90:1086-1123`).
pub fn do1_eos_and_kap(
    model: &mut StarModel,
    chem: &RspChemCache,
    wk: &mut RspHydWork,
    i: usize,
) -> Result<(), StarError> {
    let k0 = wk.nzn - i;
    let t_in = model.t[k0];
    let vol = model.aux.rsp.vol_rsp[k0];
    let out = eval_mesa_eos_and_kap(model, chem, Some(k0), t_in, vol)?;
    model.rho[k0] = model.lnd[k0].exp();
    model.t[k0] = model.ln_t[k0].exp();
    model.pgas[k0] = out.pgas;
    model.aux.rsp.egas_rsp[k0] = out.egas;
    model.csound[k0] = out.csnd;
    model.cp[k0] = out.cp;
    model.aux.rsp.qq_rsp[k0] = out.q;
    model.opacity[k0] = out.op;

    wk.d_pg_dvol[i] = out.d_pg_dv;
    wk.d_pg_dt[i] = out.d_pg_dt;
    wk.d_egas_dvol[i] = out.d_egas_dv;
    wk.d_egas_dt[i] = out.d_egas_dt;
    wk.dcp_dvol[i] = out.cpv;
    wk.dcp_dt[i] = out.cpt;
    wk.dqq_dvol[i] = out.qv;
    wk.dqq_dt[i] = out.qt;
    wk.dk_dvol[i] = out.opv;
    wk.dk_dt[i] = out.opt;

    let (dr0, dri) = (wk.dvol_dr_00[i], wk.dvol_dr_in[i]);
    wk.d_pg_dr_00[i] = out.d_pg_dv * dr0;
    wk.d_pg_dr_in[i] = out.d_pg_dv * dri;
    wk.d_egas_dr_00[i] = out.d_egas_dv * dr0;
    wk.d_egas_dr_in[i] = out.d_egas_dv * dri;
    wk.dcp_dr_00[i] = out.cpv * dr0;
    wk.dcp_dr_in[i] = out.cpv * dri;
    wk.dqq_dr_00[i] = out.qv * dr0;
    wk.dqq_dr_in[i] = out.qv * dri;
    wk.dk_dr_00[i] = out.opv * dr0;
    wk.dk_dr_in[i] = out.opv * dri;
    Ok(())
}

fn calc_prad(model: &mut StarModel, wk: &mut RspHydWork, i: usize) {
    let k0 = wk.nzn - i;
    let v = model.aux.rsp.vol_rsp[k0];
    let f_edd = model.aux.rsp.f_edd_rsp[k0];
    let prad = f_edd * model.aux.rsp.erad_rsp[k0] / v;
    model.prad[k0] = prad;
    wk.d_pr_der[i] = f_edd / v;
    wk.d_pr_dvol[i] = -prad / v;
    wk.d_pr_dr_00[i] = wk.d_pr_dvol[i] * wk.dvol_dr_00[i];
    wk.d_pr_dr_in[i] = wk.d_pr_dvol[i] * wk.dvol_dr_in[i];
}

fn calc_hp_face(model: &mut StarModel, wk: &mut RspHydWork, i: usize) {
    let k0 = wk.nzn - i;
    let r_k = model.r[k0];
    let v_k = model.aux.rsp.vol_rsp[k0];
    let p_k = model.pgas[k0] + model.prad[k0];
    if i < wk.nzn {
        let ko = k0 - 1;
        let v_o = model.aux.rsp.vol_rsp[ko];
        let p_o = model.pgas[ko] + model.prad[ko];
        let pom = r_k * r_k / (2.0 * model.cgrav[k0] * model.m[k0]);
        let hp = pom * (p_k * v_k + p_o * v_o);
        model.aux.rsp.hp_face[k0] = hp;
        wk.dhp_dvol_00[i] =
            pom * (v_k * (wk.d_pg_dvol[i] + wk.d_pr_dvol[i]) + p_k);
        wk.dhp_dvol_out[i] =
            pom * (v_o * (wk.d_pg_dvol[i + 1] + wk.d_pr_dvol[i + 1]) + p_o);
        wk.dhp_dr_in[i] = pom
            * (p_k * wk.dvol_dr_in[i]
                + (wk.d_pg_dr_in[i] + wk.d_pr_dr_in[i]) * v_k);
        wk.dhp_dr_00[i] = 2.0 * hp / r_k
            + pom
                * (p_k * wk.dvol_dr_00[i]
                    + (wk.d_pg_dr_00[i] + wk.d_pr_dr_00[i]) * v_k
                    + p_o * wk.dvol_dr_in[i + 1]
                    + (wk.d_pg_dr_in[i + 1] + wk.d_pr_dr_in[i + 1]) * v_o);
        wk.dhp_dr_out[i] = pom
            * (p_o * wk.dvol_dr_00[i + 1]
                + (wk.d_pg_dr_00[i + 1] + wk.d_pr_dr_00[i + 1]) * v_o);
        wk.dhp_dt_00[i] = pom * v_k * wk.d_pg_dt[i];
        wk.dhp_dt_out[i] = pom * v_o * wk.d_pg_dt[i + 1];
        wk.dhp_der_00[i] = pom * v_k * wk.d_pr_der[i];
        wk.dhp_der_out[i] = pom * v_o * wk.d_pr_der[i + 1];
    } else {
        let pom = r_k * r_k / (model.cgrav[k0] * model.m[k0]);
        let hp = pom * p_k * v_k;
        model.aux.rsp.hp_face[k0] = hp;
        wk.dhp_dvol_00[i] =
            pom * (v_k * (wk.d_pg_dvol[i] + wk.d_pr_dvol[i]) + p_k);
        wk.dhp_dvol_out[i] = 0.0;
        wk.dhp_dr_in[i] = pom
            * (p_k * wk.dvol_dr_in[i]
                + (wk.d_pg_dr_in[i] + wk.d_pr_dr_in[i]) * v_k);
        wk.dhp_dr_00[i] = 2.0 * hp / r_k
            + pom
                * (p_k * wk.dvol_dr_00[i]
                    + (wk.d_pg_dr_00[i] + wk.d_pr_dr_00[i]) * v_k);
        wk.dhp_dr_out[i] = 0.0;
        wk.dhp_dt_00[i] = pom * v_k * wk.d_pg_dt[i];
        wk.dhp_dt_out[i] = 0.0;
        wk.dhp_der_00[i] = pom * v_k * wk.d_pr_der[i];
        wk.dhp_der_out[i] = 0.0;
    }
}

fn calc_y_face(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &mut RspHydWork,
    i: usize,
) {
    let k0 = wk.nzn - i;
    if i < wk.nzn && dc.alfa != 0.0 {
        let ko = k0 - 1;
        let (qq_k, cp_k) = (model.aux.rsp.qq_rsp[k0], model.cp[k0]);
        let (qq_o, cp_o) = (model.aux.rsp.qq_rsp[ko], model.cp[ko]);
        let p_k = model.pgas[k0] + model.prad[k0];
        let p_o = model.pgas[ko] + model.prad[ko];
        let pom = 0.5 * (qq_k / cp_k + qq_o / cp_o);
        let pom2 = 0.5 * (p_o - p_k);
        let y1 = pom * (p_o - p_k) - (model.ln_t[ko] - model.ln_t[k0]);

        let d_y1_dvol_00 = -pom * (wk.d_pg_dvol[i] + wk.d_pr_dvol[i])
            + pom2 * (wk.dqq_dvol[i] - qq_k * wk.dcp_dvol[i] / cp_k) / cp_k;
        let d_y1_dvol_out = pom * (wk.d_pg_dvol[i + 1] + wk.d_pr_dvol[i + 1])
            + pom2 * (wk.dqq_dvol[i + 1] - qq_o * wk.dcp_dvol[i + 1] / cp_o)
                / cp_o;
        let d_y1_dr_00 = pom2
            * ((wk.dqq_dr_00[i] - qq_k / cp_k * wk.dcp_dr_00[i]) / cp_k
                + (wk.dqq_dr_in[i + 1] - qq_o / cp_o * wk.dcp_dr_in[i + 1])
                    / cp_o)
            + pom
                * (wk.d_pg_dr_in[i + 1] - wk.d_pg_dr_00[i]
                    + wk.d_pr_dr_in[i + 1]
                    - wk.d_pr_dr_00[i]);
        let d_y1_dr_in =
            pom2 * (wk.dqq_dr_in[i] - qq_k / cp_k * wk.dcp_dr_in[i]) / cp_k
                + pom * (-wk.d_pg_dr_in[i] - wk.d_pr_dr_in[i]);
        let d_y1_dr_out = pom2
            * (wk.dqq_dr_00[i + 1] - qq_o / cp_o * wk.dcp_dr_00[i + 1])
            / cp_o
            + pom * (wk.d_pg_dr_00[i + 1] + wk.d_pr_dr_00[i + 1]);
        let d_y1_dt_00 = pom2 * (wk.dqq_dt[i] - qq_k / cp_k * wk.dcp_dt[i])
            / cp_k
            - pom * wk.d_pg_dt[i]
            + 1.0 / model.t[k0];
        let d_y1_dt_out =
            pom2 * (wk.dqq_dt[i + 1] - qq_o / cp_o * wk.dcp_dt[i + 1]) / cp_o
                + pom * wk.d_pg_dt[i + 1]
                - 1.0 / model.t[ko];
        let d_y1_der_00 = -pom * wk.d_pr_der[i];
        let d_y1_der_out = pom * wk.d_pr_der[i + 1];

        let v_sum = model.aux.rsp.vol_rsp[k0] + model.aux.rsp.vol_rsp[ko];
        let hp = model.aux.rsp.hp_face[k0];
        let r_k = model.r[k0];
        let pom_v = 2.0 / v_sum;
        let pom2_v = 8.0 * PI * r_k * r_k / model.dm_bar[k0] * hp;
        let y2 = 4.0 * PI * r_k * r_k * hp * pom_v / model.dm_bar[k0];
        let v_sum2 = v_sum * v_sum;

        let d_y2_dvol_00 = y2 / hp * wk.dhp_dvol_00[i] - pom2_v / v_sum2;
        let d_y2_dvol_out = y2 / hp * wk.dhp_dvol_out[i] - pom2_v / v_sum2;
        let d_y2_dr_00 = 2.0 * y2 / r_k + y2 / hp * wk.dhp_dr_00[i]
            - pom2_v / v_sum2 * (wk.dvol_dr_00[i] + wk.dvol_dr_in[i + 1]);
        let d_y2_dr_in =
            -pom2_v / v_sum2 * wk.dvol_dr_in[i] + y2 / hp * wk.dhp_dr_in[i];
        let d_y2_dr_out = -pom2_v / v_sum2 * wk.dvol_dr_00[i + 1]
            + y2 / hp * wk.dhp_dr_out[i];
        let d_y2_dt_00 = y2 / hp * wk.dhp_dt_00[i];
        let d_y2_dt_out = y2 / hp * wk.dhp_dt_out[i];
        let d_y2_der_00 = y2 / hp * wk.dhp_der_00[i];
        let d_y2_der_out = y2 / hp * wk.dhp_der_out[i];

        model.aux.rsp.y_face_rsp[k0] = y1 * y2;
        wk.dy_dr_00[i] = y1 * d_y2_dr_00 + y2 * d_y1_dr_00;
        wk.dy_dr_in[i] = y1 * d_y2_dr_in + y2 * d_y1_dr_in;
        wk.dy_dr_out[i] = y1 * d_y2_dr_out + y2 * d_y1_dr_out;
        wk.dy_dvol_00[i] = y1 * d_y2_dvol_00 + y2 * d_y1_dvol_00;
        wk.dy_dvol_out[i] = y1 * d_y2_dvol_out + y2 * d_y1_dvol_out;
        wk.dy_dt_00[i] = y1 * d_y2_dt_00 + y2 * d_y1_dt_00;
        wk.dy_dt_out[i] = y1 * d_y2_dt_out + y2 * d_y1_dt_out;
        wk.dy_der_00[i] = y1 * d_y2_der_00 + y2 * d_y1_der_00;
        wk.dy_der_out[i] = y1 * d_y2_der_out + y2 * d_y1_der_out;
    } else {
        model.aux.rsp.y_face_rsp[k0] = 0.0;
        wk.dy_dr_00[i] = 0.0;
        wk.dy_dr_in[i] = 0.0;
        wk.dy_dr_out[i] = 0.0;
        wk.dy_dvol_00[i] = 0.0;
        wk.dy_dvol_out[i] = 0.0;
        wk.dy_dt_00[i] = 0.0;
        wk.dy_dt_out[i] = 0.0;
        wk.dy_der_00[i] = 0.0;
        wk.dy_der_out[i] = 0.0;
    }
}

fn calc_pii_face(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &mut RspHydWork,
    i: usize,
) {
    let k0 = wk.nzn - i;
    if i == wk.nzn || i == 1 || dc.alfa == 0.0 {
        model.aux.rsp.pii_rsp[k0] = 0.0;
        wk.dpii_dr_00[i] = 0.0;
        wk.dpii_dr_in[i] = 0.0;
        wk.dpii_dr_out[i] = 0.0;
        wk.dpii_dvol_00[i] = 0.0;
        wk.dpii_dvol_out[i] = 0.0;
        wk.dpii_dt_00[i] = 0.0;
        wk.dpii_dt_out[i] = 0.0;
        wk.dpii_der_00[i] = 0.0;
        wk.dpii_der_out[i] = 0.0;
    } else {
        let pom = dc.alfas * dc.alfa;
        let pom2 = 0.5 * (model.cp[k0] + model.cp[k0 - 1]);
        let yf = model.aux.rsp.y_face_rsp[k0];
        model.aux.rsp.pii_rsp[k0] = pom * pom2 * yf;
        wk.dpii_dvol_00[i] =
            pom * (pom2 * wk.dy_dvol_00[i] + yf * 0.5 * wk.dcp_dvol[i]);
        wk.dpii_dvol_out[i] =
            pom * (pom2 * wk.dy_dvol_out[i] + yf * 0.5 * wk.dcp_dvol[i + 1]);
        wk.dpii_dr_in[i] =
            pom * (pom2 * wk.dy_dr_in[i] + yf * 0.5 * wk.dcp_dr_in[i]);
        wk.dpii_dr_00[i] = pom
            * (pom2 * wk.dy_dr_00[i]
                + yf * 0.5 * (wk.dcp_dr_00[i] + wk.dcp_dr_in[i + 1]));
        wk.dpii_dr_out[i] =
            pom * (pom2 * wk.dy_dr_out[i] + yf * 0.5 * wk.dcp_dr_00[i + 1]);
        wk.dpii_dt_00[i] =
            pom * (pom2 * wk.dy_dt_00[i] + yf * 0.5 * wk.dcp_dt[i]);
        wk.dpii_dt_out[i] =
            pom * (pom2 * wk.dy_dt_out[i] + yf * 0.5 * wk.dcp_dt[i + 1]);
        wk.dpii_der_00[i] = pom * pom2 * wk.dy_der_00[i];
        wk.dpii_der_out[i] = pom * pom2 * wk.dy_der_out[i];
    }
}

fn calc_pvsc(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &mut RspHydWork,
    i: usize,
) {
    let k0 = wk.nzn - i;
    if dc.cq == 0.0 {
        model.aux.rsp.pvsc_rsp[k0] = 0.0;
        wk.d_pvsc_dvol[i] = 0.0;
        wk.d_pvsc_dt[i] = 0.0;
        wk.d_pvsc_der[i] = 0.0;
        wk.d_pvsc_dr_in[i] = 0.0;
        wk.d_pvsc_dr_00[i] = 0.0;
        return;
    }
    let dv0 = if i > 1 {
        model.v[k0 + 1] - model.v[k0]
    } else {
        model.v_center - model.v[wk.nzn - 1]
    };
    let p = model.pgas[k0] + model.prad[k0];
    let v = model.aux.rsp.vol_rsp[k0];
    let sqrt_pv = (p * v).sqrt();
    if dv0 <= dc.zsh * sqrt_pv {
        model.aux.rsp.pvsc_rsp[k0] = 0.0;
        wk.d_pvsc_dvol[i] = 0.0;
        wk.d_pvsc_dt[i] = 0.0;
        wk.d_pvsc_der[i] = 0.0;
        wk.d_pvsc_dr_in[i] = 0.0;
        wk.d_pvsc_dr_00[i] = 0.0;
        return;
    }
    let dp_dt = wk.d_pg_dt[i];
    let dp_dvol = wk.d_pg_dvol[i] + wk.d_pr_dvol[i];
    let dp_der = wk.d_pr_der[i];
    let dp_dr_in = wk.d_pg_dr_in[i] + wk.d_pr_dr_in[i];
    let dp_dr_00 = wk.d_pg_dr_00[i] + wk.d_pr_dr_00[i];

    let d_sqrt_pv_dt = 0.5 * (dp_dt * v) / sqrt_pv;
    let d_sqrt_pv_dvol = 0.5 * (p + dp_dvol * v) / sqrt_pv;
    let d_sqrt_pv_der = 0.5 * (dp_der * v) / sqrt_pv;
    let d_sqrt_pv_dr_in = 0.5 * (dp_dr_in * v + p * wk.dvol_dr_in[i]) / sqrt_pv;
    let d_sqrt_pv_dr_00 = 0.5 * (dp_dr_00 * v + p * wk.dvol_dr_00[i]) / sqrt_pv;

    let dv = dv0 - dc.zsh * sqrt_pv;
    let d_dv_dt = -dc.zsh * d_sqrt_pv_dt;
    let d_dv_dvol = -dc.zsh * d_sqrt_pv_dvol;
    let d_dv_der = -dc.zsh * d_sqrt_pv_der;
    let d_dv_dr_in = 2.0 / model.dt - dc.zsh * d_sqrt_pv_dr_in;
    let d_dv_dr_00 = -2.0 / model.dt - dc.zsh * d_sqrt_pv_dr_00;

    let pvsc = dc.cq / v * dv * dv;
    model.aux.rsp.pvsc_rsp[k0] = pvsc;
    wk.d_pvsc_dvol[i] = -pvsc / v + 2.0 * d_dv_dvol * dc.cq / v * dv;
    wk.d_pvsc_dt[i] = dc.cq / v * 2.0 * dv * d_dv_dt;
    wk.d_pvsc_der[i] = dc.cq / v * 2.0 * dv * d_dv_der;
    wk.d_pvsc_dr_in[i] = dc.cq / v * 2.0 * dv * d_dv_dr_in
        - dc.cq * dv * dv * wk.dvol_dr_in[i] / (v * v);
    wk.d_pvsc_dr_00[i] = dc.cq / v * 2.0 * dv * d_dv_dr_00
        - dc.cq * dv * dv * wk.dvol_dr_00[i] / (v * v);
}

fn check_omega(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &mut RspHydWork,
    i: usize,
) {
    let ibot = dc.ibotom.max(1);
    if i > ibot && i < wk.nzn && dc.alfa != 0.0 {
        let k0 = wk.nzn - i;
        if wk.w[k0] > dc.efl0 {
            return;
        }
        let pom = 0.5
            * (model.aux.rsp.pii_rsp[k0] / model.aux.rsp.hp_face[k0]
                + model.aux.rsp.pii_rsp[k0 + 1]
                    / model.aux.rsp.hp_face[k0 + 1]);
        let pom2 = model.t[k0]
            * (model.pgas[k0] + model.prad[k0])
            * model.aux.rsp.qq_rsp[k0]
            / model.cp[k0];
        let sours = pom * pom2;
        let damps = (dc.cede / dc.alfa)
            / (0.5
                * (model.aux.rsp.hp_face[k0] + model.aux.rsp.hp_face[k0 + 1]));
        let pom3 = (dc.gammar * dc.gammar) / (dc.alfa * dc.alfa) * 4.0 * dc.sig;
        let pom4 = model.t[k0].powi(3) * model.aux.rsp.vol_rsp[k0].powi(2)
            / (model.cp[k0] * model.opacity[k0]);
        let damprs = pom3 * pom4
            / (0.5
                * (model.aux.rsp.hp_face[k0].powi(2)
                    + model.aux.rsp.hp_face[k0 + 1].powi(2)));
        let delta = damprs * damprs + 4.0 * damps * sours;
        let mut sol = if delta >= 0.0 {
            (-damprs + delta.sqrt()) / (2.0 * damps)
        } else {
            -99.99
        };
        if sol >= 0.0 {
            sol *= sol;
        }
        if sol > 0.0 {
            wk.w[k0] = sol.sqrt();
            model.aux.rsp.et_rsp[k0] = wk.w[k0] * wk.w[k0];
        }
    }
}

fn calc_pturb(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &mut RspHydWork,
    i: usize,
) {
    let k0 = wk.nzn - i;
    if dc.alfa == 0.0 || dc.alfap == 0.0 || i <= dc.ibotom || i >= wk.nzn {
        model.aux.rsp.ptrb_rsp[k0] = 0.0;
        wk.dptrb_dvol_00[i] = 0.0;
        wk.dptrb_dw_00[i] = 0.0;
        wk.dptrb_dr_00[i] = 0.0;
        wk.dptrb_dr_in[i] = 0.0;
    } else {
        let vol = model.aux.rsp.vol_rsp[k0];
        let w_k = wk.w[k0];
        let ptrb = dc.alfap * w_k * w_k / vol;
        model.aux.rsp.ptrb_rsp[k0] = ptrb;
        wk.dptrb_dvol_00[i] = -ptrb / vol;
        wk.dptrb_dw_00[i] = 2.0 * dc.alfap * w_k / vol;
        let tem1 = -dc.alfap * w_k * w_k / (vol * vol);
        wk.dptrb_dr_00[i] = tem1 * wk.dvol_dr_00[i];
        wk.dptrb_dr_in[i] = tem1 * wk.dvol_dr_in[i];
    }
}
