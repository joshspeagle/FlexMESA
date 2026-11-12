// SPDX-License-Identifier: LGPL-3.0-or-later
//! Analytical partial-derivative workspace [`RspHydWork`] and boundary-zeroing
//! helper for the nonlinear `RSP` `HYD` solver (`star/private/rsp_def.f90:118-137`,
//! `star/private/rsp_step.f90:2063-2134`).

use crate::model::StarModel;
use crate::rsp_def::RspDerivedConsts;

/// Workspace holding turbulent speed `w` (`k0 = 0..nzn`) and 1-based Smolec
/// (`i = 1..=nzn`) partial derivatives for `HYD` (`rsp_def.f90:118-137`).
#[derive(Debug, Clone)]
pub struct RspHydWork {
    /// Envelope zone count `NZN`.
    pub nzn: usize,
    /// Turbulent speed $w_k = \sqrt{e_{t,k}}$ (`k0 = 0..nzn`, $\mathrm{cm\,s^{-1}}$).
    pub w: Vec<f64>,
    pub(crate) dvol_dr_in: Vec<f64>,
    pub(crate) dvol_dr_00: Vec<f64>,
    pub(crate) d_pg_dvol: Vec<f64>,
    pub(crate) d_pg_dt: Vec<f64>,
    pub(crate) d_pg_dr_00: Vec<f64>,
    pub(crate) d_pg_dr_in: Vec<f64>,
    pub(crate) d_egas_dvol: Vec<f64>,
    pub(crate) d_egas_dt: Vec<f64>,
    pub(crate) d_egas_dr_00: Vec<f64>,
    pub(crate) d_egas_dr_in: Vec<f64>,
    pub(crate) dcp_dvol: Vec<f64>,
    pub(crate) dcp_dt: Vec<f64>,
    pub(crate) dcp_dr_00: Vec<f64>,
    pub(crate) dcp_dr_in: Vec<f64>,
    pub(crate) dqq_dvol: Vec<f64>,
    pub(crate) dqq_dt: Vec<f64>,
    pub(crate) dqq_dr_00: Vec<f64>,
    pub(crate) dqq_dr_in: Vec<f64>,
    pub(crate) dk_dvol: Vec<f64>,
    pub(crate) dk_dt: Vec<f64>,
    pub(crate) dk_dr_00: Vec<f64>,
    pub(crate) dk_dr_in: Vec<f64>,
    pub(crate) d_pr_der: Vec<f64>,
    pub(crate) d_pr_dvol: Vec<f64>,
    pub(crate) d_pr_dr_00: Vec<f64>,
    pub(crate) d_pr_dr_in: Vec<f64>,
    pub(crate) dhp_dvol_00: Vec<f64>,
    pub(crate) dhp_dvol_out: Vec<f64>,
    pub(crate) dhp_dr_in: Vec<f64>,
    pub(crate) dhp_dr_00: Vec<f64>,
    pub(crate) dhp_dr_out: Vec<f64>,
    pub(crate) dhp_dt_00: Vec<f64>,
    pub(crate) dhp_dt_out: Vec<f64>,
    pub(crate) dhp_der_00: Vec<f64>,
    pub(crate) dhp_der_out: Vec<f64>,
    pub(crate) dy_dr_00: Vec<f64>,
    pub(crate) dy_dr_in: Vec<f64>,
    pub(crate) dy_dr_out: Vec<f64>,
    pub(crate) dy_dvol_00: Vec<f64>,
    pub(crate) dy_dvol_out: Vec<f64>,
    pub(crate) dy_dt_00: Vec<f64>,
    pub(crate) dy_dt_out: Vec<f64>,
    pub(crate) dy_der_00: Vec<f64>,
    pub(crate) dy_der_out: Vec<f64>,
    pub(crate) dpii_dr_00: Vec<f64>,
    pub(crate) dpii_dr_in: Vec<f64>,
    pub(crate) dpii_dr_out: Vec<f64>,
    pub(crate) dpii_dvol_00: Vec<f64>,
    pub(crate) dpii_dvol_out: Vec<f64>,
    pub(crate) dpii_dt_00: Vec<f64>,
    pub(crate) dpii_dt_out: Vec<f64>,
    pub(crate) dpii_der_00: Vec<f64>,
    pub(crate) dpii_der_out: Vec<f64>,
    pub(crate) d_pvsc_dvol: Vec<f64>,
    pub(crate) d_pvsc_dt: Vec<f64>,
    pub(crate) d_pvsc_der: Vec<f64>,
    pub(crate) d_pvsc_dr_in: Vec<f64>,
    pub(crate) d_pvsc_dr_00: Vec<f64>,
    pub(crate) dptrb_dvol_00: Vec<f64>,
    pub(crate) dptrb_dw_00: Vec<f64>,
    pub(crate) dptrb_dr_00: Vec<f64>,
    pub(crate) dptrb_dr_in: Vec<f64>,
    pub(crate) dchi_dvol_out: Vec<f64>,
    pub(crate) dchi_dvol_00: Vec<f64>,
    pub(crate) dchi_dvol_in: Vec<f64>,
    pub(crate) dchi_dt_out: Vec<f64>,
    pub(crate) dchi_dt_00: Vec<f64>,
    pub(crate) dchi_dt_in: Vec<f64>,
    pub(crate) dchi_der_out: Vec<f64>,
    pub(crate) dchi_der_00: Vec<f64>,
    pub(crate) dchi_der_in: Vec<f64>,
    pub(crate) dchi_dw_00: Vec<f64>,
    pub(crate) dchi_dr_out: Vec<f64>,
    pub(crate) dchi_dr_00: Vec<f64>,
    pub(crate) dchi_dr_in: Vec<f64>,
    pub(crate) dchi_dr_in2: Vec<f64>,
    pub(crate) deq_dvol_out: Vec<f64>,
    pub(crate) deq_dvol_00: Vec<f64>,
    pub(crate) deq_dvol_in: Vec<f64>,
    pub(crate) deq_dr_out: Vec<f64>,
    pub(crate) deq_dr_00: Vec<f64>,
    pub(crate) deq_dr_in: Vec<f64>,
    pub(crate) deq_dr_in2: Vec<f64>,
    pub(crate) deq_dt_out: Vec<f64>,
    pub(crate) deq_dt_00: Vec<f64>,
    pub(crate) deq_dt_in: Vec<f64>,
    pub(crate) deq_der_out: Vec<f64>,
    pub(crate) deq_der_00: Vec<f64>,
    pub(crate) deq_der_in: Vec<f64>,
    pub(crate) deq_dw_00: Vec<f64>,
    pub(crate) dc_dr_00: Vec<f64>,
    pub(crate) dc_dr_out: Vec<f64>,
    pub(crate) dc_dr_in: Vec<f64>,
    pub(crate) dc_dr_in2: Vec<f64>,
    pub(crate) dc_dvol_00: Vec<f64>,
    pub(crate) dc_dvol_out: Vec<f64>,
    pub(crate) dc_dvol_in: Vec<f64>,
    pub(crate) dc_dt_00: Vec<f64>,
    pub(crate) dc_dt_out: Vec<f64>,
    pub(crate) dc_dt_in: Vec<f64>,
    pub(crate) dc_der_00: Vec<f64>,
    pub(crate) dc_der_out: Vec<f64>,
    pub(crate) dc_der_in: Vec<f64>,
    pub(crate) dc_dw_00: Vec<f64>,
}

impl RspHydWork {
    /// Allocates [`RspHydWork`] for `model.nz` zones and extracts $w_k = \sqrt{e_{t,k}}$.
    #[must_use]
    pub fn from_model(model: &StarModel) -> Self {
        let nzn = model.nz;
        let sz = nzn + 2;
        let w = (0..nzn)
            .map(|k| model.aux.rsp.et_rsp[k].max(0.0).sqrt())
            .collect();
        Self {
            nzn,
            w,
            dvol_dr_in: vec![0.0; sz],
            dvol_dr_00: vec![0.0; sz],
            d_pg_dvol: vec![0.0; sz],
            d_pg_dt: vec![0.0; sz],
            d_pg_dr_00: vec![0.0; sz],
            d_pg_dr_in: vec![0.0; sz],
            d_egas_dvol: vec![0.0; sz],
            d_egas_dt: vec![0.0; sz],
            d_egas_dr_00: vec![0.0; sz],
            d_egas_dr_in: vec![0.0; sz],
            dcp_dvol: vec![0.0; sz],
            dcp_dt: vec![0.0; sz],
            dcp_dr_00: vec![0.0; sz],
            dcp_dr_in: vec![0.0; sz],
            dqq_dvol: vec![0.0; sz],
            dqq_dt: vec![0.0; sz],
            dqq_dr_00: vec![0.0; sz],
            dqq_dr_in: vec![0.0; sz],
            dk_dvol: vec![0.0; sz],
            dk_dt: vec![0.0; sz],
            dk_dr_00: vec![0.0; sz],
            dk_dr_in: vec![0.0; sz],
            d_pr_der: vec![0.0; sz],
            d_pr_dvol: vec![0.0; sz],
            d_pr_dr_00: vec![0.0; sz],
            d_pr_dr_in: vec![0.0; sz],
            dhp_dvol_00: vec![0.0; sz],
            dhp_dvol_out: vec![0.0; sz],
            dhp_dr_in: vec![0.0; sz],
            dhp_dr_00: vec![0.0; sz],
            dhp_dr_out: vec![0.0; sz],
            dhp_dt_00: vec![0.0; sz],
            dhp_dt_out: vec![0.0; sz],
            dhp_der_00: vec![0.0; sz],
            dhp_der_out: vec![0.0; sz],
            dy_dr_00: vec![0.0; sz],
            dy_dr_in: vec![0.0; sz],
            dy_dr_out: vec![0.0; sz],
            dy_dvol_00: vec![0.0; sz],
            dy_dvol_out: vec![0.0; sz],
            dy_dt_00: vec![0.0; sz],
            dy_dt_out: vec![0.0; sz],
            dy_der_00: vec![0.0; sz],
            dy_der_out: vec![0.0; sz],
            dpii_dr_00: vec![0.0; sz],
            dpii_dr_in: vec![0.0; sz],
            dpii_dr_out: vec![0.0; sz],
            dpii_dvol_00: vec![0.0; sz],
            dpii_dvol_out: vec![0.0; sz],
            dpii_dt_00: vec![0.0; sz],
            dpii_dt_out: vec![0.0; sz],
            dpii_der_00: vec![0.0; sz],
            dpii_der_out: vec![0.0; sz],
            d_pvsc_dvol: vec![0.0; sz],
            d_pvsc_dt: vec![0.0; sz],
            d_pvsc_der: vec![0.0; sz],
            d_pvsc_dr_in: vec![0.0; sz],
            d_pvsc_dr_00: vec![0.0; sz],
            dptrb_dvol_00: vec![0.0; sz],
            dptrb_dw_00: vec![0.0; sz],
            dptrb_dr_00: vec![0.0; sz],
            dptrb_dr_in: vec![0.0; sz],
            dchi_dvol_out: vec![0.0; sz],
            dchi_dvol_00: vec![0.0; sz],
            dchi_dvol_in: vec![0.0; sz],
            dchi_dt_out: vec![0.0; sz],
            dchi_dt_00: vec![0.0; sz],
            dchi_dt_in: vec![0.0; sz],
            dchi_der_out: vec![0.0; sz],
            dchi_der_00: vec![0.0; sz],
            dchi_der_in: vec![0.0; sz],
            dchi_dw_00: vec![0.0; sz],
            dchi_dr_out: vec![0.0; sz],
            dchi_dr_00: vec![0.0; sz],
            dchi_dr_in: vec![0.0; sz],
            dchi_dr_in2: vec![0.0; sz],
            deq_dvol_out: vec![0.0; sz],
            deq_dvol_00: vec![0.0; sz],
            deq_dvol_in: vec![0.0; sz],
            deq_dr_out: vec![0.0; sz],
            deq_dr_00: vec![0.0; sz],
            deq_dr_in: vec![0.0; sz],
            deq_dr_in2: vec![0.0; sz],
            deq_dt_out: vec![0.0; sz],
            deq_dt_00: vec![0.0; sz],
            deq_dt_in: vec![0.0; sz],
            deq_der_out: vec![0.0; sz],
            deq_der_00: vec![0.0; sz],
            deq_der_in: vec![0.0; sz],
            deq_dw_00: vec![0.0; sz],
            dc_dr_00: vec![0.0; sz],
            dc_dr_out: vec![0.0; sz],
            dc_dr_in: vec![0.0; sz],
            dc_dr_in2: vec![0.0; sz],
            dc_dvol_00: vec![0.0; sz],
            dc_dvol_out: vec![0.0; sz],
            dc_dvol_in: vec![0.0; sz],
            dc_dt_00: vec![0.0; sz],
            dc_dt_out: vec![0.0; sz],
            dc_dt_in: vec![0.0; sz],
            dc_der_00: vec![0.0; sz],
            dc_der_out: vec![0.0; sz],
            dc_der_in: vec![0.0; sz],
            dc_dw_00: vec![0.0; sz],
        }
    }
}

/// Zeros turbulent dissipation and pressure fields on radiative inner (`1..=ibotom`)
/// and outermost (`nzn`) zones (`zero_boundaries` in `rsp_step.f90:2063-2134`).
pub fn zero_boundaries(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &mut RspHydWork,
) {
    let clear_i = |model: &mut StarModel, wk: &mut RspHydWork, i: usize| {
        let k0 = wk.nzn - i;
        model.aux.rsp.eq_rsp[k0] = 0.0;
        wk.deq_dr_in2[i] = 0.0;
        wk.deq_dr_in[i] = 0.0;
        wk.deq_dr_00[i] = 0.0;
        wk.deq_dr_out[i] = 0.0;
        wk.deq_dt_in[i] = 0.0;
        wk.deq_dt_00[i] = 0.0;
        wk.deq_dt_out[i] = 0.0;
        wk.deq_dw_00[i] = 0.0;
        model.aux.rsp.chi_rsp[k0] = 0.0;
        wk.dchi_dr_in2[i] = 0.0;
        wk.dchi_dr_in[i] = 0.0;
        wk.dchi_dr_00[i] = 0.0;
        wk.dchi_dr_out[i] = 0.0;
        wk.dchi_dt_in[i] = 0.0;
        wk.dchi_dt_00[i] = 0.0;
        wk.dchi_dt_out[i] = 0.0;
        wk.dchi_dw_00[i] = 0.0;
        model.aux.rsp.coupl_rsp[k0] = 0.0;
        wk.dc_dr_00[i] = 0.0;
        wk.dc_dr_out[i] = 0.0;
        wk.dc_dr_in[i] = 0.0;
        wk.dc_dr_in2[i] = 0.0;
        wk.dc_dt_in[i] = 0.0;
        wk.dc_dt_00[i] = 0.0;
        wk.dc_dt_out[i] = 0.0;
        wk.dc_dw_00[i] = 0.0;
        model.aux.rsp.ptrb_rsp[k0] = 0.0;
        wk.dptrb_dr_00[i] = 0.0;
        wk.dptrb_dr_in[i] = 0.0;
        wk.dptrb_dw_00[i] = 0.0;
    };
    for i in 1..=dc.ibotom.min(wk.nzn) {
        clear_i(model, wk, i);
    }
    clear_i(model, wk, wk.nzn);
}
