// SPDX-License-Identifier: LGPL-3.0-or-later
//! Face convective/turbulent luminosities (`Lc`, `Lt`), Stellingwerf
//! temperature-form radiative flux (`T_form_of_calc_Fr`), and time-weighted
//! combined pressure (`rsp_calc_XP`) for the nonlinear `RSP` `HYD` solver
//! (`star/private/rsp_step.f90:2137-2582, 3690-3792`).

use flexmesa_const::{CRAD, PI};

use crate::model::StarModel;
use crate::rsp_def::RspDerivedConsts;
use crate::rsp_hyd_work::RspHydWork;

/// Face turbulent kinetic-energy luminosity $L_t$ and analytical partials.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RspLtOut {
    pub lt_00: f64,
    pub dlt_dr_00: f64,
    pub dlt_dr_in: f64,
    pub dlt_dr_out: f64,
    pub dlt_dt_00: f64,
    pub dlt_dt_out: f64,
    pub dlt_der_00: f64,
    pub dlt_der_out: f64,
    pub dlt_dw_00: f64,
    pub dlt_dw_out: f64,
}

/// Face convective luminosity $L_c$ and analytical partials.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RspLcOut {
    pub lc_00: f64,
    pub dlc_dr_in: f64,
    pub dlc_dr_00: f64,
    pub dlc_dr_out: f64,
    pub dlc_dt_00: f64,
    pub dlc_dt_out: f64,
    pub dlc_der_00: f64,
    pub dlc_der_out: f64,
    pub dlc_dw_00: f64,
    pub dlc_dw_out: f64,
}

/// Face Stellingwerf radiative flux $F_r$ (`T_form_of_calc_Fr`) and partials.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RspFrTOut {
    pub fr_00: f64,
    pub dfr_dr_out: f64,
    pub dfr_dr_in: f64,
    pub dfr_dr_00: f64,
    pub dfr_dt_out: f64,
    pub dfr_dt_00: f64,
}

/// Time-weighted combined pressure `XP` (`rsp_calc_XP`) and partials.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RspXpOut {
    pub xp: f64,
    pub d_xp_dt_00: f64,
    pub d_xp_der_00: f64,
    pub d_xp_dw_00: f64,
    pub d_xp_dr_in: f64,
    pub d_xp_dr_00: f64,
}

/// Evaluates turbulent luminosity $L_t$ at face `i` (`0..=nzn`)
/// (`calc_Lt` in `rsp_step.f90:2137-2215`).
pub(crate) fn calc_lt(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &RspHydWork,
    i: usize,
) -> RspLtOut {
    let nzn = wk.nzn;
    let k_f = nzn + 1 - i;
    if i == 0
        || i <= dc.ibotom.max(1)
        || i == nzn
        || dc.alfa == 0.0
        || dc.alfat == 0.0
        || k_f < model.aux.rsp.min_k_for_turbulent_flux
    {
        if i > 0 && i <= nzn {
            model.aux.rsp.lt_rsp[nzn - i] = 0.0;
        }
        return RspLtOut::default();
    }
    let k0 = nzn - i;
    let ko = k0 - 1;
    let pom3 = (wk.w[ko].powi(3) - wk.w[k0].powi(3)) / model.dm_bar[k0];
    let pom =
        -2.0 / 3.0 * dc.alfa * dc.alfat * (dc.p4 * model.r[k0].powi(2)).powi(2);
    let (v0, vo) = (model.aux.rsp.vol_rsp[k0], model.aux.rsp.vol_rsp[ko]);
    let rho2_face = 0.5 * (1.0 / (v0 * v0) + 1.0 / (vo * vo));
    let hp = model.aux.rsp.hp_face[k0];
    let pom2 = hp * rho2_face;
    let lt_00 = pom * pom2 * pom3;
    model.aux.rsp.lt_rsp[k0] = lt_00;

    let tem1 = lt_00 / hp;
    let tem2 = lt_00 / pom2 * hp;
    let tem_w = pom * pom2 * 3.0 / model.dm_bar[k0];

    RspLtOut {
        lt_00,
        dlt_dr_00: 4.0 * lt_00 / model.r[k0]
            - tem2 / v0.powi(3) * wk.dvol_dr_00[i]
            - tem2 / vo.powi(3) * wk.dvol_dr_in[i + 1]
            + tem1 * wk.dhp_dr_00[i],
        dlt_dr_in: -tem2 / v0.powi(3) * wk.dvol_dr_in[i]
            + tem1 * wk.dhp_dr_in[i],
        dlt_dr_out: -tem2 / vo.powi(3) * wk.dvol_dr_00[i + 1]
            + tem1 * wk.dhp_dr_out[i],
        dlt_dt_00: tem1 * wk.dhp_dt_00[i],
        dlt_dt_out: tem1 * wk.dhp_dt_out[i],
        dlt_der_00: tem1 * wk.dhp_der_00[i],
        dlt_der_out: tem1 * wk.dhp_der_out[i],
        dlt_dw_00: -tem_w * wk.w[k0].powi(2),
        dlt_dw_out: tem_w * wk.w[ko].powi(2),
    }
}

/// Evaluates convective luminosity $L_c$ at face `i` (`0..=nzn`)
/// (`calc_Lc` in `rsp_step.f90:2218-2318`).
pub(crate) fn calc_lc(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &RspHydWork,
    i: usize,
) -> RspLcOut {
    let nzn = wk.nzn;
    if i == 0 || i <= dc.ibotom.max(1) || i == nzn || dc.alfa == 0.0 {
        if i > 0 && i <= nzn {
            model.aux.rsp.lc_rsp[nzn - i] = 0.0;
        }
        return RspLcOut::default();
    }
    let k0 = nzn - i;
    if wk.w[k0] < dc.efl0 * 1.0e-8 {
        model.aux.rsp.lc_rsp[k0] = 0.0;
        return RspLcOut::default();
    }
    let ko = k0 - 1;
    let pom3 = 0.5 * (wk.w[k0] + wk.w[ko]);
    let (v0, vo) = (model.aux.rsp.vol_rsp[k0], model.aux.rsp.vol_rsp[ko]);
    let (t0, to) = (model.t[k0], model.t[ko]);
    let r_k = model.r[k0];
    let pii = model.aux.rsp.pii_rsp[k0];

    let pom0 =
        dc.p4 * r_k * r_k * (dc.alfac / dc.alfas) * 0.5 * (t0 / v0 + to / vo);
    let lc_00 = pom0 * pii * pom3;
    model.aux.rsp.lc_rsp[k0] = lc_00;

    let dlc_dw_00 = pom0 * pii * 0.5;
    let dlc_dw_out = if i >= nzn - 1 { 0.0 } else { pom0 * pii * 0.5 };
    let pom2 = dc.p4 * r_k * r_k * pii * pom3 * 0.5 * (dc.alfac / dc.alfas);
    let pom = pom0 * pom3;

    RspLcOut {
        lc_00,
        dlc_dr_00: pom * wk.dpii_dr_00[i] + 2.0 * lc_00 / r_k
            - pom2
                * (t0 / (v0 * v0) * wk.dvol_dr_00[i]
                    + to / (vo * vo) * wk.dvol_dr_in[i + 1]),
        dlc_dr_in: pom * wk.dpii_dr_in[i]
            - pom2 * t0 / (v0 * v0) * wk.dvol_dr_in[i],
        dlc_dr_out: pom * wk.dpii_dr_out[i]
            - pom2 * to / (vo * vo) * wk.dvol_dr_00[i + 1],
        dlc_dt_00: pom * wk.dpii_dt_00[i] + pom2 / v0,
        dlc_dt_out: pom * wk.dpii_dt_out[i] + pom2 / vo,
        dlc_der_00: pom * wk.dpii_der_00[i],
        dlc_der_out: pom * wk.dpii_der_out[i],
        dlc_dw_00,
        dlc_dw_out,
    }
}

/// Evaluates Stellingwerf (1975) temperature-form radiative flux $F_r$
/// (`T_form_of_calc_Fr` in `rsp_step.f90:3690-3792`).
#[must_use]
pub(crate) fn t_form_of_calc_fr(
    model: &StarModel,
    dc: &RspDerivedConsts,
    wk: &RspHydWork,
    i: usize,
) -> RspFrTOut {
    let nzn = wk.nzn;
    if i < 1 {
        let l_in = if model.controls.rsp.hydro_only {
            0.0
        } else {
            model.l_center
        };
        return RspFrTOut {
            fr_00: l_in / (4.0 * PI * model.r_center.powi(2)),
            ..RspFrTOut::default()
        };
    }
    let k0 = nzn - i;
    if i == nzn {
        let fr_00 = 2.0 * dc.sig * model.t[k0].powi(4);
        return RspFrTOut {
            fr_00,
            dfr_dt_00: 4.0 * fr_00 / model.t[k0],
            ..RspFrTOut::default()
        };
    }
    let ko = k0 - 1;
    let w = model.t[k0].powi(4);
    let wp = model.t[ko].powi(4);
    let bw = 4.0 * (model.ln_t[ko] - model.ln_t[k0]);
    if bw.abs() < 1.0e-20 {
        return RspFrTOut::default();
    }
    let (op0, opo) = (model.opacity[k0], model.opacity[ko]);
    let bk = (opo / op0).ln();
    if (1.0 - bk / bw).abs() < 1.0e-15 || (bw - bk).abs() < 1.0e-15 {
        return RspFrTOut::default();
    }
    let t1 = -dc.cl * model.r[k0].powi(2) / (4.0 * PI * model.dm_bar[k0]);
    let t2 = (wp / opo - w / op0) / (1.0 - bk / bw);
    let fr_00 = t1 * t2;
    let t3 = t1 / (bw - bk);
    let dfr_dk_00 = (t3 / op0) * (w * bw / op0 - t2);
    let dfr_dk_out = -(t3 / opo) * (wp * bw / opo - t2);

    RspFrTOut {
        fr_00,
        dfr_dr_out: dfr_dk_out * wk.dk_dr_00[i + 1],
        dfr_dr_in: dfr_dk_00 * wk.dk_dr_in[i],
        dfr_dr_00: 2.0 * fr_00 / model.r[k0]
            + dfr_dk_00 * wk.dk_dr_00[i]
            + dfr_dk_out * wk.dk_dr_in[i + 1],
        dfr_dt_out: 4.0 * (t3 / model.t[ko]) * (wp * bw / opo - t2 * bk / bw)
            + dfr_dk_out * wk.dk_dt[i + 1],
        dfr_dt_00: -4.0 * (t3 / model.t[k0]) * (w * bw / op0 - t2 * bk / bw)
            + dfr_dk_00 * wk.dk_dt[i],
    }
}

/// Evaluates time-weighted combined pressure `XP` at Smolec index `i` (`1..=nzn + 1`)
/// (`rsp_calc_XP` in `rsp_step.f90:2498-2582`).
#[must_use]
pub(crate) fn rsp_calc_xp(
    model: &StarModel,
    dc: &RspDerivedConsts,
    wk: &RspHydWork,
    p_surf: f64,
    i: usize,
    with_prad: bool,
) -> RspXpOut {
    if i > wk.nzn {
        let xp = if model.controls.rsp.use_atm_grey_with_kap_for_psurf {
            p_surf
        } else if model.controls.rsp.use_prad_for_psurf {
            CRAD * model.aux.rsp.t_start_rsp[0].powi(4) / 3.0
        } else {
            0.0
        };
        return RspXpOut {
            xp,
            ..RspXpOut::default()
        };
    }
    let k0 = wk.nzn - i;
    let pf = if with_prad { 1.0 } else { 0.0 };
    let xp = dc.theta * (model.pgas[k0] + pf * model.prad[k0])
        + dc.theta1
            * (model.aux.rsp.pgas_start[k0]
                + pf * model.aux.rsp.prad_start[k0])
        + dc.thetaq * model.aux.rsp.pvsc_rsp[k0]
        + dc.thetaq1 * model.aux.rsp.pvsc_start[k0]
        + dc.thetat * model.aux.rsp.ptrb_rsp[k0]
        + dc.thetat1 * model.aux.rsp.ptrb_start[k0];
    RspXpOut {
        xp,
        d_xp_dt_00: dc.theta * wk.d_pg_dt[i] + dc.thetaq * wk.d_pvsc_dt[i],
        d_xp_der_00: dc.theta * pf * wk.d_pr_der[i],
        d_xp_dw_00: dc.thetat * wk.dptrb_dw_00[i],
        d_xp_dr_in: dc.theta * (wk.d_pg_dr_in[i] + pf * wk.d_pr_dr_in[i])
            + dc.thetaq * wk.d_pvsc_dr_in[i]
            + dc.thetat * wk.dptrb_dr_in[i],
        d_xp_dr_00: dc.theta * (wk.d_pg_dr_00[i] + pf * wk.d_pr_dr_00[i])
            + dc.thetaq * wk.d_pvsc_dr_00[i]
            + dc.thetat * wk.dptrb_dr_00[i],
    }
}
