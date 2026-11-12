// SPDX-License-Identifier: LGPL-3.0-or-later
//! Kuhfuss (1986) eddy-viscosity stress (`Chi`), shear dissipation (`Eq`), and
//! buoyancy/dissipation turbulent coupling (`SOURCE`, `DAMP`, `DAMPR`, `COUPL`)
//! with analytical partials (`star/private/rsp_step.f90:1613-2060`).

use flexmesa_const::PI;

use crate::model::StarModel;
use crate::rsp_def::RspDerivedConsts;
use crate::rsp_hyd_work::RspHydWork;

/// Evaluates Kuhfuss (1986) eddy viscosity `Chi(k)` and partials
/// (`calc_Chi` in `rsp_step.f90:1613-1737`).
pub fn calc_chi(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &mut RspHydWork,
    i: usize,
) {
    let k0 = wk.nzn - i;
    let ibot = dc.ibotom.max(1);
    if dc.alfa == 0.0 || i <= ibot || i >= wk.nzn {
        model.aux.rsp.chi_rsp[k0] = 0.0;
        wk.dchi_dvol_out[i] = 0.0;
        wk.dchi_dvol_00[i] = 0.0;
        wk.dchi_dvol_in[i] = 0.0;
        wk.dchi_dt_out[i] = 0.0;
        wk.dchi_dt_00[i] = 0.0;
        wk.dchi_dt_in[i] = 0.0;
        wk.dchi_der_out[i] = 0.0;
        wk.dchi_der_00[i] = 0.0;
        wk.dchi_der_in[i] = 0.0;
        wk.dchi_dw_00[i] = 0.0;
        wk.dchi_dr_out[i] = 0.0;
        wk.dchi_dr_00[i] = 0.0;
        wk.dchi_dr_in[i] = 0.0;
        wk.dchi_dr_in2[i] = 0.0;
    } else {
        let ki = k0 + 1;
        let pom = (16.0 / 3.0) * PI * dc.alfa * dc.alfam / model.dm[k0];
        let vol = model.aux.rsp.vol_rsp[k0];
        let pom1 = wk.w[k0] / (vol * vol);
        let pom2 = 0.5 * (model.r[k0].powi(6) + model.r[ki].powi(6));
        let pom4 =
            0.5 * (model.aux.rsp.hp_face[k0] + model.aux.rsp.hp_face[ki]);
        let pom3 = model.v[k0] / model.r[k0] - model.v[ki] / model.r[ki];
        let pomt3 = pom * pom1 * pom2 * pom4;
        let pomt1 = pom * pom2 * pom3 * pom4;
        let pomt2 = pom * pom1 * pom3 * pom4;
        let pomt4 = pom * pom1 * pom2 * pom3;

        let chi = pomt1 * pom1;
        model.aux.rsp.chi_rsp[k0] = chi;

        wk.dchi_dvol_out[i] = pomt4 * 0.5 * wk.dhp_dvol_out[i];
        wk.dchi_dvol_00[i] =
            pomt4 * 0.5 * (wk.dhp_dvol_00[i] + wk.dhp_dvol_out[i - 1])
                - 2.0 * chi / vol;
        wk.dchi_dvol_in[i] = pomt4 * 0.5 * wk.dhp_dvol_00[i - 1];
        wk.dchi_dt_out[i] = pomt4 * 0.5 * wk.dhp_dt_out[i];
        wk.dchi_dt_00[i] =
            pomt4 * 0.5 * (wk.dhp_dt_00[i] + wk.dhp_dt_out[i - 1]);
        wk.dchi_dt_in[i] = pomt4 * 0.5 * wk.dhp_dt_00[i - 1];
        wk.dchi_der_out[i] = pomt4 * 0.5 * wk.dhp_der_out[i];
        wk.dchi_der_00[i] =
            pomt4 * 0.5 * (wk.dhp_der_00[i] + wk.dhp_der_out[i - 1]);
        wk.dchi_der_in[i] = pomt4 * 0.5 * wk.dhp_der_00[i - 1];
        wk.dchi_dw_00[i] = pomt1 / (vol * vol);
        wk.dchi_dr_out[i] = pomt4 * 0.5 * wk.dhp_dr_out[i];
        wk.dchi_dr_00[i] = -2.0 * chi / vol * wk.dvol_dr_00[i]
            + pomt2 * 3.0 * model.r[k0].powi(5)
            + pomt3
                * (2.0 / model.dt / model.r[k0]
                    - model.v[k0] / model.r[k0].powi(2))
            + pomt4 * 0.5 * (wk.dhp_dr_00[i] + wk.dhp_dr_out[i - 1]);
        wk.dchi_dr_in[i] = -2.0 * chi / vol * wk.dvol_dr_in[i]
            + pomt2 * 3.0 * model.r[ki].powi(5)
            - pomt3
                * (2.0 / model.dt / model.r[ki]
                    - model.v[ki] / model.r[ki].powi(2))
            + pomt4 * 0.5 * (wk.dhp_dr_in[i] + wk.dhp_dr_00[i - 1]);
        wk.dchi_dr_in2[i] = pomt4 * 0.5 * wk.dhp_dr_in[i - 1];
    }
}

/// Evaluates eddy-viscosity shear dissipation `Eq(k)` and partials
/// (`calc_Eq` in `rsp_step.f90:1740-1805`).
pub fn calc_eq(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &mut RspHydWork,
    i: usize,
) {
    let k0 = wk.nzn - i;
    let ibot = dc.ibotom.max(1);
    if dc.alfa == 0.0 || i <= ibot || i >= wk.nzn {
        model.aux.rsp.eq_rsp[k0] = 0.0;
        wk.deq_dr_out[i] = 0.0;
        wk.deq_dr_00[i] = 0.0;
        wk.deq_dr_in[i] = 0.0;
        wk.deq_dr_in2[i] = 0.0;
        wk.deq_dvol_out[i] = 0.0;
        wk.deq_dvol_00[i] = 0.0;
        wk.deq_dvol_in[i] = 0.0;
        wk.deq_dt_out[i] = 0.0;
        wk.deq_dt_00[i] = 0.0;
        wk.deq_dt_in[i] = 0.0;
        wk.deq_der_out[i] = 0.0;
        wk.deq_der_00[i] = 0.0;
        wk.deq_der_in[i] = 0.0;
        wk.deq_dw_00[i] = 0.0;
    } else {
        let ki = k0 + 1;
        let rri = 0.5 * (model.r[k0] + model.aux.rsp.r_start_rsp[k0]);
        let rrm = 0.5 * (model.r[ki] + model.aux.rsp.r_start_rsp[ki]);
        let uui = 0.5 * (model.v[k0] + model.aux.rsp.v_start_rsp[k0]);
        let uum = 0.5 * (model.v[ki] + model.aux.rsp.v_start_rsp[ki]);

        let pom0 = dc.p4 / model.dm[k0] * (uui / rri - uum / rrm);
        let chi_tw = dc.thetau * model.aux.rsp.chi_rsp[k0]
            + dc.thetau1 * model.aux.rsp.chi_start[k0];
        let pom2 = dc.p4 / model.dm[k0] * chi_tw;
        model.aux.rsp.eq_rsp[k0] = pom0 * chi_tw;

        let pom = pom0 * dc.thetau;
        wk.deq_dvol_out[i] = pom * dc.thetau * wk.dchi_dvol_out[i];
        wk.deq_dvol_00[i] = pom * dc.thetau * wk.dchi_dvol_00[i];
        wk.deq_dvol_in[i] = pom * dc.thetau * wk.dchi_dvol_in[i];
        wk.deq_dr_out[i] = pom * wk.dchi_dr_out[i];
        wk.deq_dr_00[i] = pom * wk.dchi_dr_00[i]
            + pom2 * (1.0 / (rri * model.dt) - 0.5 * uui / (rri * rri));
        wk.deq_dr_in[i] = pom * wk.dchi_dr_in[i]
            - pom2 * (1.0 / (rrm * model.dt) - 0.5 * uum / (rrm * rrm));
        wk.deq_dr_in2[i] = pom * wk.dchi_dr_in2[i];
        wk.deq_dt_out[i] = pom * wk.dchi_dt_out[i];
        wk.deq_dt_00[i] = pom * wk.dchi_dt_00[i];
        wk.deq_dt_in[i] = pom * wk.dchi_dt_in[i];
        wk.deq_der_out[i] = pom * wk.dchi_der_out[i];
        wk.deq_der_00[i] = pom * wk.dchi_der_00[i];
        wk.deq_der_in[i] = pom * wk.dchi_der_in[i];
        wk.deq_dw_00[i] = pom * wk.dchi_dw_00[i];
    }
}

/// Evaluates buoyancy `SOURCE`, Kuhfuss `DAMP`, radiative `DAMPR`, and net
/// `COUPL` plus analytical partials (`calc_source_sink` in `rsp_step.f90:1808-2060`).
pub fn calc_source_sink(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &mut RspHydWork,
    i: usize,
) {
    let k0 = wk.nzn - i;
    let ibot = dc.ibotom.max(1);
    if dc.alfa == 0.0 || i <= ibot || i >= wk.nzn {
        model.aux.rsp.source_rsp[k0] = 0.0;
        model.aux.rsp.damp_rsp[k0] = 0.0;
        model.aux.rsp.dampr_rsp[k0] = 0.0;
        model.aux.rsp.coupl_rsp[k0] = 0.0;
        return;
    }
    let ki = k0 + 1;
    let (hp0, hpi) = (model.aux.rsp.hp_face[k0], model.aux.rsp.hp_face[ki]);
    let (pii0, piii) = (model.aux.rsp.pii_rsp[k0], model.aux.rsp.pii_rsp[ki]);
    let (t_k, cp_k, op_k) = (model.t[k0], model.cp[k0], model.opacity[k0]);
    let (qq_k, vol_k) = (model.aux.rsp.qq_rsp[k0], model.aux.rsp.vol_rsp[k0]);
    let p_k = model.pgas[k0] + model.prad[k0];
    let w_k = wk.w[k0];

    let pom0 = 0.5 * (pii0 / hp0 + piii / hpi);
    let qq_div_cp = qq_k / cp_k;
    let pom2_s = t_k * p_k * qq_div_cp;
    model.aux.rsp.source_rsp[k0] = pom0 * pom2_s * w_k;

    let tem1 = pom2_s * w_k * 0.5;
    let (temi, temm) = (-pii0 / (hp0 * hp0), -piii / (hpi * hpi));
    let d_pom_dvol_00 = 0.5
        * (wk.dpii_dvol_00[i] - pii0 / hp0 * wk.dhp_dvol_00[i])
        / hp0
        + 0.5 * (wk.dpii_dvol_out[i - 1] - piii / hpi * wk.dhp_dvol_out[i - 1])
            / hpi;
    let d_qq_div_cp_dvol =
        (wk.dqq_dvol[i] - wk.dcp_dvol[i] * qq_k / cp_k) / cp_k;
    let d_pom2_dvol_00 = t_k
        * ((wk.d_pg_dvol[i] + wk.d_pr_dvol[i]) * qq_div_cp
            + p_k * d_qq_div_cp_dvol);
    let dsrc_dvol_out =
        tem1 * (wk.dpii_dvol_out[i] / hp0 + temi * wk.dhp_dvol_out[i]);
    let dsrc_dvol_00 = w_k * (d_pom_dvol_00 * pom2_s + pom0 * d_pom2_dvol_00);
    let dsrc_dvol_in =
        tem1 * (wk.dpii_dvol_00[i - 1] / hpi + temm * wk.dhp_dvol_00[i - 1]);

    let dsrc_dr_out =
        tem1 * (wk.dpii_dr_out[i] / hp0 + temi * wk.dhp_dr_out[i]);
    let mut dsrc_dr_00 = tem1
        * (wk.dpii_dr_00[i] / hp0
            + wk.dpii_dr_out[i - 1] / hpi
            + temi * wk.dhp_dr_00[i]
            + temm * wk.dhp_dr_out[i - 1]);
    let mut dsrc_dr_in = tem1
        * (wk.dpii_dr_in[i] / hp0
            + wk.dpii_dr_00[i - 1] / hpi
            + temi * wk.dhp_dr_in[i]
            + temm * wk.dhp_dr_00[i - 1]);
    let dsrc_dr_in2 =
        tem1 * (wk.dpii_dr_in[i - 1] / hpi + temm * wk.dhp_dr_in[i - 1]);

    let dsrc_dt_out =
        tem1 * (wk.dpii_dt_out[i] / hp0 + temi * wk.dhp_dt_out[i]);
    let mut dsrc_dt_00 = tem1
        * (wk.dpii_dt_00[i] / hp0
            + wk.dpii_dt_out[i - 1] / hpi
            + temi * wk.dhp_dt_00[i]
            + temm * wk.dhp_dt_out[i - 1]);
    let dsrc_dt_in =
        tem1 * (wk.dpii_dt_00[i - 1] / hpi + temm * wk.dhp_dt_00[i - 1]);

    let dsrc_der_out =
        tem1 * (wk.dpii_der_out[i] / hp0 + temi * wk.dhp_der_out[i]);
    let mut dsrc_der_00 = tem1
        * (wk.dpii_der_00[i] / hp0
            + wk.dpii_der_out[i - 1] / hpi
            + temi * wk.dhp_der_00[i]
            + temm * wk.dhp_der_out[i - 1]);
    let dsrc_der_in =
        tem1 * (wk.dpii_der_00[i - 1] / hpi + temm * wk.dhp_der_00[i - 1]);

    let dsrc_dw_00 = pom0 * pom2_s;
    let pom_w = pom0 * w_k;
    dsrc_dt_00 += pom_w / cp_k
        * (p_k * qq_k + t_k * qq_k * wk.d_pg_dt[i] + t_k * p_k * wk.dqq_dt[i]
            - t_k * p_k * qq_k / cp_k * wk.dcp_dt[i]);
    dsrc_der_00 += pom_w / cp_k * t_k * qq_k * wk.d_pr_der[i];
    dsrc_dr_00 += pom_w * t_k / cp_k
        * (qq_k * (wk.d_pg_dr_00[i] + wk.d_pr_dr_00[i])
            + p_k * wk.dqq_dr_00[i]
            - p_k * qq_k / cp_k * wk.dcp_dr_00[i]);
    dsrc_dr_in += pom_w * t_k / cp_k
        * (qq_k * (wk.d_pg_dr_in[i] + wk.d_pr_dr_in[i])
            + p_k * wk.dqq_dr_in[i]
            - p_k * qq_k / cp_k * wk.dcp_dr_in[i]);

    let pom_d = (dc.cede / dc.alfa) * (w_k.powi(3) - dc.efl0.powi(3));
    let pom2_d = 0.5 * (hp0 + hpi);
    model.aux.rsp.damp_rsp[k0] = pom_d / pom2_d;
    let tem_d = -0.5 * pom_d / (pom2_d * pom2_d);
    let d_damp_dvol_out = tem_d * wk.dhp_dvol_out[i];
    let d_damp_dvol_00 = tem_d * (wk.dhp_dvol_00[i] + wk.dhp_dvol_out[i - 1]);
    let d_damp_dvol_in = tem_d * wk.dhp_dvol_00[i - 1];
    let d_damp_dr_out = tem_d * wk.dhp_dr_out[i];
    let d_damp_dr_00 = tem_d * (wk.dhp_dr_00[i] + wk.dhp_dr_out[i - 1]);
    let d_damp_dr_in = tem_d * (wk.dhp_dr_in[i] + wk.dhp_dr_00[i - 1]);
    let d_damp_dr_in2 = tem_d * wk.dhp_dr_in[i - 1];
    let d_damp_dt_00 = tem_d * (wk.dhp_dt_00[i] + wk.dhp_dt_out[i - 1]);
    let d_damp_dt_out = tem_d * wk.dhp_dt_out[i];
    let d_damp_dt_in = tem_d * wk.dhp_dt_00[i - 1];
    let d_damp_der_00 = tem_d * (wk.dhp_der_00[i] + wk.dhp_der_out[i - 1]);
    let d_damp_der_out = tem_d * wk.dhp_der_out[i];
    let d_damp_der_in = tem_d * wk.dhp_der_00[i - 1];
    let d_damp_dw_00 = 3.0 * (dc.cede / dc.alfa) / pom2_d * w_k * w_k;

    let mut dampr_p = [0.0_f64; 14];
    if dc.gammar == 0.0 {
        model.aux.rsp.dampr_rsp[k0] = 0.0;
    } else {
        let pom_r =
            (dc.gammar * dc.gammar) / (dc.alfa * dc.alfa) * 4.0 * dc.sig;
        let pom2a = t_k.powi(3) * vol_k * vol_k;
        let pom2b = 1.0 / (cp_k * op_k);
        let pom2_r = pom2a * pom2b;
        let pom3_r = w_k * w_k;
        let pom4_r = 0.5 * (hp0 * hp0 + hpi * hpi);
        let dampr = pom_r * pom2_r * pom3_r / pom4_r;
        model.aux.rsp.dampr_rsp[k0] = dampr;
        let tem_r = -dampr / pom4_r;

        let d_pom2a_dvol = 2.0 * t_k.powi(3) * vol_k;
        let d_pom2b_dvol =
            -pom2b * (wk.dcp_dvol[i] / cp_k + wk.dk_dvol[i] / op_k);
        let d_pom2_dvol = d_pom2a_dvol * pom2b + pom2a * d_pom2b_dvol;
        let d_pom4_dvol =
            hp0 * wk.dhp_dvol_00[i] + hpi * wk.dhp_dvol_out[i - 1];
        let tem_r2 = pom_r * pom3_r / pom4_r;

        dampr_p[0] = tem_r * hp0 * wk.dhp_dr_out[i];
        dampr_p[1] = tem_r
            * (hp0 * wk.dhp_dr_00[i] + hpi * wk.dhp_dr_out[i - 1])
            + tem_r2
                * t_k.powi(3)
                * (2.0 * vol_k * wk.dvol_dr_00[i]
                    - vol_k
                        * vol_k
                        * (wk.dcp_dr_00[i] / cp_k + wk.dk_dr_00[i] / op_k))
                / (cp_k * op_k);
        dampr_p[2] = tem_r
            * (hp0 * wk.dhp_dr_in[i] + hpi * wk.dhp_dr_00[i - 1])
            + tem_r2
                * t_k.powi(3)
                * (2.0 * vol_k * wk.dvol_dr_in[i]
                    - vol_k
                        * vol_k
                        * (wk.dcp_dr_in[i] / cp_k + wk.dk_dr_in[i] / op_k))
                / (cp_k * op_k);
        dampr_p[3] = tem_r * hpi * wk.dhp_dr_in[i - 1];
        dampr_p[4] = tem_r * hp0 * wk.dhp_dvol_out[i];
        dampr_p[5] =
            pom_r * pom3_r * (d_pom2_dvol - pom2_r * d_pom4_dvol / pom4_r)
                / pom4_r;
        dampr_p[6] = tem_r * hpi * wk.dhp_dvol_00[i - 1];
        dampr_p[7] = tem_r * hp0 * wk.dhp_dt_out[i];
        dampr_p[8] = tem_r
            * (hp0 * wk.dhp_dt_00[i] + hpi * wk.dhp_dt_out[i - 1])
            + tem_r2
                * vol_k
                * vol_k
                * (3.0 * t_k * t_k
                    - t_k.powi(3) * (wk.dcp_dt[i] / cp_k + wk.dk_dt[i] / op_k))
                / (cp_k * op_k);
        dampr_p[9] = tem_r * hpi * wk.dhp_dt_00[i - 1];
        dampr_p[10] = tem_r * hp0 * wk.dhp_der_out[i];
        dampr_p[11] =
            tem_r * (hp0 * wk.dhp_der_00[i] + hpi * wk.dhp_der_out[i - 1]);
        dampr_p[12] = tem_r * hpi * wk.dhp_der_00[i - 1];
        dampr_p[13] = pom_r * pom2_r / pom4_r * 2.0 * w_k;
    }

    model.aux.rsp.coupl_rsp[k0] = model.aux.rsp.source_rsp[k0]
        - model.aux.rsp.damp_rsp[k0]
        - model.aux.rsp.dampr_rsp[k0];
    wk.dc_dr_out[i] = dsrc_dr_out - d_damp_dr_out - dampr_p[0];
    wk.dc_dr_00[i] = dsrc_dr_00 - d_damp_dr_00 - dampr_p[1];
    wk.dc_dr_in[i] = dsrc_dr_in - d_damp_dr_in - dampr_p[2];
    wk.dc_dr_in2[i] = dsrc_dr_in2 - d_damp_dr_in2 - dampr_p[3];
    wk.dc_dvol_out[i] = dsrc_dvol_out - d_damp_dvol_out - dampr_p[4];
    wk.dc_dvol_00[i] = dsrc_dvol_00 - d_damp_dvol_00 - dampr_p[5];
    wk.dc_dvol_in[i] = dsrc_dvol_in - d_damp_dvol_in - dampr_p[6];
    wk.dc_dt_out[i] = dsrc_dt_out - d_damp_dt_out - dampr_p[7];
    wk.dc_dt_00[i] = dsrc_dt_00 - d_damp_dt_00 - dampr_p[8];
    wk.dc_dt_in[i] = dsrc_dt_in - d_damp_dt_in - dampr_p[9];
    wk.dc_der_out[i] = dsrc_der_out - d_damp_der_out - dampr_p[10];
    wk.dc_der_00[i] = dsrc_der_00 - d_damp_der_00 - dampr_p[11];
    wk.dc_der_in[i] = dsrc_der_in - d_damp_der_in - dampr_p[12];
    wk.dc_dw_00[i] = dsrc_dw_00 - d_damp_dw_00 - dampr_p[13];
}
