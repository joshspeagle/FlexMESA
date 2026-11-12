// SPDX-License-Identifier: LGPL-3.0-or-later
//! Cell-centered nonlinear `RSP` `HYD` conservation equations
//! (`total_energy_eqn`, `turbulent_energy_eqn`, and `T_form_of_erad_eqn` in
//! `star/private/rsp_step.f90:3122-3758`).

use flexmesa_const::{CRAD, PI};

use crate::model::StarModel;
use crate::rsp_def::{RspDerivedConsts, F_EDD_ISOTROPIC};
use crate::rsp_eqns::{RspHydBand, LD_HD, NV};
use crate::rsp_hyd_flux::{calc_lc, rsp_calc_xp, RspLtOut};
use crate::rsp_hyd_work::RspHydWork;

const I_T_DR_IN2: usize = 5;
const I_T_DT_IN: usize = 1 + NV;
const I_T_DT_00: usize = I_T_DT_IN + NV;
const I_T_DT_OUT: usize = I_T_DT_00 + NV;
const I_T_DW_IN: usize = 2 + NV;
const I_T_DW_00: usize = I_T_DW_IN + NV;
const I_T_DW_OUT: usize = I_T_DW_00 + NV;
const I_T_DER_IN: usize = 3 + NV;
const I_T_DER_00: usize = I_T_DER_IN + NV;
const I_T_DER_OUT: usize = I_T_DER_00 + NV;
const I_T_DFR_IN: usize = 4 + NV;
const I_T_DFR_00: usize = I_T_DFR_IN + NV;
const I_T_DR_IN: usize = I_T_DR_IN2 + NV;
const I_T_DR_00: usize = I_T_DR_IN + NV;
const I_T_DR_OUT: usize = I_T_DR_00 + NV;

const I_W_DW_IN2: usize = 1;
const I_W_DR_IN2: usize = 4;
const I_W_DT_IN: usize = 5;
const I_W_DW_IN: usize = I_W_DW_IN2 + NV;
const I_W_DW_00: usize = I_W_DW_IN + NV;
const I_W_DW_OUT: usize = I_W_DW_00 + NV;
const I_W_DW_OUT2: usize = I_W_DW_OUT + NV;
const I_W_DER_IN: usize = 2 + NV;
const I_W_DER_00: usize = I_W_DER_IN + NV;
const I_W_DER_OUT: usize = I_W_DER_00 + NV;
const I_W_DR_IN: usize = I_W_DR_IN2 + NV;
const I_W_DR_00: usize = I_W_DR_IN + NV;
const I_W_DR_OUT: usize = I_W_DR_00 + NV;
const I_W_DT_00: usize = I_W_DT_IN + NV;
const I_W_DT_OUT: usize = I_W_DT_00 + NV;

const I_ER_DER_00: usize = 1 + 2 * NV;
const I_ER_DR_IN: usize = 3 + NV;
const I_ER_DR_00: usize = I_ER_DR_IN + NV;
const I_ER_DT_00: usize = 4 + NV;

#[allow(clippy::too_many_arguments)]
pub(crate) fn total_energy_eqn(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &RspHydWork,
    band: &mut RspHydBand,
    i: usize,
    p_surf: f64,
    lt_00: &RspLtOut,
    lt_00_start: f64,
    lt_in: &RspLtOut,
    lt_in_start: f64,
) {
    let k0 = wk.nzn - i;
    let (it, _, _, _, _) = RspHydBand::row_indices(i);
    band.hd[it] = [0.0; LD_HD + 1];

    let lc_in = calc_lc(model, dc, wk, i - 1);
    let lc_00 = calc_lc(model, dc, wk, i);
    let lc_00_start = if i == wk.nzn {
        0.0
    } else {
        model.aux.rsp.lc_start[k0]
    };
    let lc_in_start = if i == 1 {
        0.0
    } else {
        model.aux.rsp.lc_start[k0 + 1]
    };

    let area_00 = 4.0 * PI * model.r[k0].powi(2);
    let lr_00 = model.aux.rsp.fr_rsp[k0] * area_00;
    let d_lr_00_dfr_00 = area_00;
    let d_lr_00_dr_00 = 2.0 * lr_00 / model.r[k0];
    let lr_00_start = model.aux.rsp.fr_start[k0]
        * (4.0 * PI * model.aux.rsp.r_start_rsp[k0].powi(2));

    let (lr_in, lr_in_start, d_lr_in_dfr_in, d_lr_in_dr_in) = if i == 1 {
        let l_c = if model.controls.rsp.hydro_only {
            0.0
        } else {
            model.l_center
        };
        (l_c, l_c, 0.0, 0.0)
    } else {
        let ki = k0 + 1;
        let area_in = 4.0 * PI * model.r[ki].powi(2);
        let lr_i = model.aux.rsp.fr_rsp[ki] * area_in;
        let lr_i_st = model.aux.rsp.fr_start[ki]
            * (4.0 * PI * model.aux.rsp.r_start_rsp[ki].powi(2));
        (lr_i, lr_i_st, area_in, 2.0 * lr_i / model.r[ki])
    };

    let l_00 = dc.wtr * lr_00
        + dc.wtc * lc_00.lc_00
        + dc.wtt * lt_00.lt_00
        + dc.wtr1 * lr_00_start
        + dc.wtc1 * lc_00_start
        + dc.wtt1 * lt_00_start;
    let l_in = dc.wtr * lr_in
        + dc.wtc * lc_in.lc_00
        + dc.wtt * lt_in.lt_00
        + dc.wtr1 * lr_in_start
        + dc.wtc1 * lc_in_start
        + dc.wtt1 * lt_in_start;

    let dt = model.dt;
    let dt_div_dm = dt / model.dm[k0];
    let dv = model.aux.rsp.vol_rsp[k0] - model.aux.rsp.vol_start[k0];

    let (u_div_r, d_u_dr_00, d_u_dr_in, u_factor) =
        if model.aux.rsp.f_edd_rsp[k0] == F_EDD_ISOTROPIC || k0 + 1 == wk.nzn {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let ki = k0 + 1;
            (
                0.5 * (model.v[k0] / model.r[k0] + model.v[ki] / model.r[ki]),
                0.5 * (2.0 / dt - model.v[k0] / model.r[k0]) / model.r[k0],
                0.5 * (2.0 / dt - model.v[ki] / model.r[ki]) / model.r[ki],
                dt * (1.0 - 3.0 * model.aux.rsp.f_edd_rsp[k0]),
            )
        };

    let erad = model.aux.rsp.erad_rsp[k0];
    let erad_tw = dc.thetae * erad + dc.thetae1 * model.aux.rsp.erad_start[k0];
    let xp_00 = rsp_calc_xp(model, dc, wk, p_surf, i, true);

    let residual = model.aux.rsp.egas_rsp[k0] - model.aux.rsp.egas_start[k0]
        + erad
        - model.aux.rsp.erad_start[k0]
        + wk.w[k0].powi(2)
        - model.aux.rsp.w_start[k0].powi(2)
        + dt_div_dm * (l_00 - l_in)
        + xp_00.xp * dv
        + erad_tw * u_factor * u_div_r
        - dt * model.aux.rsp.eq_rsp[k0];
    band.hr[it] = -residual;

    let h = &mut band.hd[it];
    h[I_T_DFR_IN] = -dt_div_dm * dc.wtr * d_lr_in_dfr_in;
    h[I_T_DFR_00] = dt_div_dm * dc.wtr * d_lr_00_dfr_00;

    h[I_T_DR_IN2] = -dt_div_dm * dc.wtc * lc_in.dlc_dr_in
        - dt_div_dm * dc.wtt * lt_in.dlt_dr_in
        - dt * wk.deq_dr_in2[i];
    h[I_T_DR_IN] = wk.d_egas_dr_in[i] - dt_div_dm * dc.wtr * d_lr_in_dr_in
        + dt_div_dm * dc.wtc * (lc_00.dlc_dr_in - lc_in.dlc_dr_00)
        + dt_div_dm * dc.wtt * (lt_00.dlt_dr_in - lt_in.dlt_dr_00)
        + wk.dvol_dr_in[i] * xp_00.xp
        + dv * xp_00.d_xp_dr_in
        + erad_tw * u_factor * d_u_dr_in
        - dt * wk.deq_dr_in[i];
    h[I_T_DR_00] = wk.d_egas_dr_00[i]
        + dt_div_dm * dc.wtr * d_lr_00_dr_00
        + dt_div_dm * dc.wtc * (lc_00.dlc_dr_00 - lc_in.dlc_dr_out)
        + dt_div_dm * dc.wtt * (lt_00.dlt_dr_00 - lt_in.dlt_dr_out)
        + wk.dvol_dr_00[i] * xp_00.xp
        + dv * xp_00.d_xp_dr_00
        + erad_tw * u_factor * d_u_dr_00
        - dt * wk.deq_dr_00[i];
    h[I_T_DR_OUT] = dt_div_dm * dc.wtc * lc_00.dlc_dr_out
        + dt_div_dm * dc.wtt * lt_00.dlt_dr_out
        - dt * wk.deq_dr_out[i];

    h[I_T_DT_IN] = -dt_div_dm * dc.wtc * lc_in.dlc_dt_00
        - dt_div_dm * dc.wtt * lt_in.dlt_dt_00
        - dt * wk.deq_dt_in[i];
    h[I_T_DT_00] = wk.d_egas_dt[i]
        + dv * xp_00.d_xp_dt_00
        + dt_div_dm * dc.wtc * (lc_00.dlc_dt_00 - lc_in.dlc_dt_out)
        + dt_div_dm * dc.wtt * (lt_00.dlt_dt_00 - lt_in.dlt_dt_out)
        - dt * wk.deq_dt_00[i];
    h[I_T_DT_OUT] = dt_div_dm * dc.wtc * lc_00.dlc_dt_out
        + dt_div_dm * dc.wtt * lt_00.dlt_dt_out
        - dt * wk.deq_dt_out[i];

    h[I_T_DER_IN] = -dt_div_dm * dc.wtc * lc_in.dlc_der_00
        - dt_div_dm * dc.wtt * lt_in.dlt_der_00
        - dt * wk.deq_der_in[i];
    h[I_T_DER_00] = 1.0
        + dv * xp_00.d_xp_der_00
        + dt_div_dm * dc.wtc * (lc_00.dlc_der_00 - lc_in.dlc_der_out)
        + dt_div_dm * dc.wtt * (lt_00.dlt_der_00 - lt_in.dlt_der_out)
        + dc.thetae * u_factor * u_div_r
        - dt * wk.deq_der_00[i];
    h[I_T_DER_OUT] = dt_div_dm * dc.wtc * lc_00.dlc_der_out
        + dt_div_dm * dc.wtt * lt_00.dlt_der_out
        - dt * wk.deq_der_out[i];

    h[I_T_DW_IN] = if i <= dc.ibotom + 1 {
        0.0
    } else {
        -dt_div_dm * dc.wtc * lc_in.dlc_dw_00
            - dt_div_dm * dc.wtt * lt_in.dlt_dw_00
    };
    h[I_T_DW_00] = if i <= dc.ibotom || i == wk.nzn {
        0.0
    } else {
        2.0 * wk.w[k0]
            + dt_div_dm * dc.wtc * (lc_00.dlc_dw_00 - lc_in.dlc_dw_out)
            + dt_div_dm * dc.wtt * (lt_00.dlt_dw_00 - lt_in.dlt_dw_out)
            + dv * xp_00.d_xp_dw_00
            - dt * wk.deq_dw_00[i]
    };
    h[I_T_DW_OUT] = if i < dc.ibotom || i >= wk.nzn - 1 {
        0.0
    } else {
        dt_div_dm * dc.wtt * lt_00.dlt_dw_out
            + dt_div_dm * dc.wtc * lc_00.dlc_dw_out
    };
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn turbulent_energy_eqn(
    model: &StarModel,
    dc: &RspDerivedConsts,
    wk: &RspHydWork,
    band: &mut RspHydBand,
    i: usize,
    lt_00: &RspLtOut,
    lt_00_start: f64,
    lt_in: &RspLtOut,
    lt_in_start: f64,
) {
    let k0 = wk.nzn - i;
    let (_, iw, _, _, _) = RspHydBand::row_indices(i);
    band.hd[iw] = [0.0; LD_HD + 1];

    if dc.alfa == 0.0 || i <= dc.ibotom || i == wk.nzn {
        band.hd[iw][I_W_DW_00] = 1.0;
        band.hr[iw] = 0.0;
        return;
    }

    let l_00 = dc.wtt * lt_00.lt_00 + dc.wtt1 * lt_00_start;
    let l_in = dc.wtt * lt_in.lt_00 + dc.wtt1 * lt_in_start;
    let dt = model.dt;
    let dt_div_dm = dt / model.dm[k0];
    let dv = model.aux.rsp.vol_rsp[k0] - model.aux.rsp.vol_start[k0];
    let ptrb_tw = dc.thetat * model.aux.rsp.ptrb_rsp[k0]
        + dc.thetat1 * model.aux.rsp.ptrb_start[k0];

    let residual = (wk.w[k0].powi(2) - model.aux.rsp.w_start[k0].powi(2))
        + dt_div_dm * (l_00 - l_in)
        + dv * ptrb_tw
        - dt * (dc.gam * model.aux.rsp.coupl_rsp[k0]
            + dc.gam1 * model.aux.rsp.coupl_start[k0]
            + model.aux.rsp.eq_rsp[k0]);
    band.hr[iw] = -residual;

    let h = &mut band.hd[iw];
    h[I_W_DW_IN2] = 0.0;
    h[I_W_DW_IN] = -dt_div_dm * dc.wtt * lt_in.dlt_dw_00;
    h[I_W_DW_00] =
        2.0 * wk.w[k0] - dt * dc.gam * wk.dc_dw_00[i] - dt * wk.deq_dw_00[i]
            + dt_div_dm * dc.wtt * (lt_00.dlt_dw_00 - lt_in.dlt_dw_out)
            + dv * dc.thetat * wk.dptrb_dw_00[i];
    h[I_W_DW_OUT] = dt_div_dm * dc.wtt * lt_00.dlt_dw_out;
    h[I_W_DW_OUT2] = 0.0;

    h[I_W_DR_IN2] = -dt * dc.gam * wk.dc_dr_in2[i]
        - dt_div_dm * dc.wtt * lt_in.dlt_dr_in
        - dt * wk.deq_dr_in2[i];
    h[I_W_DR_IN] = -dt * dc.gam * wk.dc_dr_in[i] - dt * wk.deq_dr_in[i]
        + dt_div_dm * dc.wtt * (lt_00.dlt_dr_in - lt_in.dlt_dr_00)
        + dv * dc.thetat * wk.dptrb_dr_in[i]
        + ptrb_tw * wk.dvol_dr_in[i];
    h[I_W_DR_00] = -dt * dc.gam * wk.dc_dr_00[i] - dt * wk.deq_dr_00[i]
        + dt_div_dm * dc.wtt * (lt_00.dlt_dr_00 - lt_in.dlt_dr_out)
        + dv * dc.thetat * wk.dptrb_dr_00[i]
        + ptrb_tw * wk.dvol_dr_00[i];
    h[I_W_DR_OUT] = -dt * dc.gam * wk.dc_dr_out[i]
        + dt_div_dm * dc.wtt * lt_00.dlt_dr_out
        - dt * wk.deq_dr_out[i];

    h[I_W_DT_IN] = if i <= dc.ibotom + 1 {
        0.0
    } else {
        -dt_div_dm * dc.wtt * lt_in.dlt_dt_00
            - dt * wk.deq_dt_in[i]
            - dt * dc.gam * wk.dc_dt_in[i]
    };
    h[I_W_DT_00] = -dt * dc.gam * wk.dc_dt_00[i] - dt * wk.deq_dt_00[i]
        + dt_div_dm * dc.wtt * (lt_00.dlt_dt_00 - lt_in.dlt_dt_out);
    h[I_W_DT_OUT] = if i < dc.ibotom || i >= wk.nzn - 1 {
        0.0
    } else {
        -dt * dc.gam * wk.dc_dt_out[i] + dt_div_dm * dc.wtt * lt_00.dlt_dt_out
            - dt * wk.deq_dt_out[i]
    };

    h[I_W_DER_IN] = if i <= dc.ibotom + 1 {
        0.0
    } else {
        -dt_div_dm * dc.wtt * lt_in.dlt_der_00
            - dt * wk.deq_der_in[i]
            - dt * dc.gam * wk.dc_der_in[i]
    };
    h[I_W_DER_00] = -dt * dc.gam * wk.dc_der_00[i] - dt * wk.deq_der_00[i]
        + dt_div_dm * dc.wtt * (lt_00.dlt_der_00 - lt_in.dlt_der_out);
    h[I_W_DER_OUT] = if i < dc.ibotom || i >= wk.nzn - 1 {
        0.0
    } else {
        -dt * dc.gam * wk.dc_der_out[i] + dt_div_dm * dc.wtt * lt_00.dlt_der_out
            - dt * wk.deq_der_out[i]
    };
}

pub(crate) fn t_form_of_erad_eqn(
    model: &StarModel,
    wk: &RspHydWork,
    band: &mut RspHydBand,
    i: usize,
) {
    let k0 = wk.nzn - i;
    let (_, _, ie, _, _) = RspHydBand::row_indices(i);
    band.hd[ie] = [0.0; LD_HD + 1];
    let t = model.t[k0];
    let v = model.aux.rsp.vol_rsp[k0];
    let erad_expected = CRAD * t.powi(4) * v;
    let residual = erad_expected - model.aux.rsp.erad_rsp[k0];
    band.hr[ie] = -residual;
    let h = &mut band.hd[ie];
    h[I_ER_DER_00] = -1.0;
    h[I_ER_DT_00] = 4.0 * CRAD * t.powi(3) * v;
    h[I_ER_DR_00] = CRAD * t.powi(4) * wk.dvol_dr_00[i];
    h[I_ER_DR_IN] = CRAD * t.powi(4) * wk.dvol_dr_in[i];
}
