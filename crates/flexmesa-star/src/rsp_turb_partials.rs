// SPDX-License-Identifier: LGPL-3.0-or-later
//! Shared 1-based (`1..=nzn`) Smolec & Moskalik (2008) convective, turbulent,
//! pressure, and eddy-viscosity partial derivatives used by both `RELAX_ENV`
//! (`star/private/rsp_relax_env.f90:576-988`) and `do_LINA`
//! (`star/private/rsp_lina.f90:263-674`).

use flexmesa_const::PI;

use crate::rsp_def::RspDerivedConsts;
use crate::rsp_grid::RspEnvGrid;
use crate::rsp_zone_micro::{
    eval_hp_face_partials, LcFilterMode, RspZoneMicro,
};

/// Turbulent source/sink, convective luminosity, turbulent luminosity,
/// turbulent pressure, and eddy-viscosity partials (`1..=nzn`).
#[derive(Debug, Clone)]
pub struct RspTurbPartials {
    /// Turbulent energy generation `SOURCE`.
    pub source: Vec<f64>,
    /// Kuhfuss dissipation `DAMP`.
    pub damp: Vec<f64>,
    /// Radiative cooling `DAMPR`.
    pub dampr: Vec<f64>,
    /// $\partial C_i / \partial R_{i-2}$.
    pub dc_dr_in2: Vec<f64>,
    /// $\partial C_i / \partial R_{i-1}$.
    pub dc_dr_in: Vec<f64>,
    /// $\partial C_i / \partial R_i$.
    pub dc_dr_00: Vec<f64>,
    /// $\partial C_i / \partial R_{i+1}$.
    pub dc_dr_out: Vec<f64>,
    /// $\partial C_i / \partial T_{i-1}$.
    pub dc_dt_in: Vec<f64>,
    /// $\partial C_i / \partial T_i$.
    pub dc_dt_00: Vec<f64>,
    /// $\partial C_i / \partial T_{i+1}$.
    pub dc_dt_out: Vec<f64>,
    /// $\partial C_i / \partial e_{t,i}$.
    pub dc_dw_00: Vec<f64>,
    /// Convective luminosity $L_{c,i}$.
    pub lc: Vec<f64>,
    /// $\partial L_{c,i} / \partial R_{i-1}$.
    pub dlcxm: Vec<f64>,
    /// $\partial L_{c,i} / \partial R_i$.
    pub dlcx0: Vec<f64>,
    /// $\partial L_{c,i} / \partial R_{i+1}$.
    pub dlcxp: Vec<f64>,
    /// $\partial L_{c,i} / \partial T_i$.
    pub dlcy0: Vec<f64>,
    /// $\partial L_{c,i} / \partial T_{i+1}$.
    pub dlcyp: Vec<f64>,
    /// $\partial L_{c,i} / \partial e_{t,i}$.
    pub dlcz0: Vec<f64>,
    /// $\partial L_{c,i} / \partial e_{t,i+1}$.
    pub dlczp: Vec<f64>,
    /// Turbulent kinetic luminosity $L_{t,i}$.
    pub lt: Vec<f64>,
    /// $\partial L_{t,i} / \partial R_{i-1}$.
    pub dltxm: Vec<f64>,
    /// $\partial L_{t,i} / \partial R_i$.
    pub dltx0: Vec<f64>,
    /// $\partial L_{t,i} / \partial R_{i+1}$.
    pub dltxp: Vec<f64>,
    /// $\partial L_{t,i} / \partial T_i$.
    pub dlty0: Vec<f64>,
    /// $\partial L_{t,i} / \partial T_{i+1}$.
    pub dltyp: Vec<f64>,
    /// $\partial L_{t,i} / \partial e_{t,i}$.
    pub dltz0: Vec<f64>,
    /// $\partial L_{t,i} / \partial e_{t,i+1}$.
    pub dltzp: Vec<f64>,
    /// Turbulent pressure $P_{\mathrm{turb},i}$.
    pub pturb: Vec<f64>,
    /// $\partial P_{\mathrm{turb},i} / \partial e_{t,i}$.
    pub dptrb_dw_00: Vec<f64>,
    /// $\partial P_{\mathrm{turb},i} / \partial R_i$.
    pub dptrb_dr_00: Vec<f64>,
    /// $\partial P_{\mathrm{turb},i} / \partial R_{i-1}$.
    pub dptrb_dr_in: Vec<f64>,
    /// Linearized outer-face eddy-viscosity coefficient `EVUU0`.
    pub evuu0: Vec<f64>,
    /// Linearized inner-face eddy-viscosity coefficient `EVUUM`.
    pub evuum: Vec<f64>,
}

impl RspTurbPartials {
    /// Allocates zero-initialized buffers for `nzn` zones.
    #[must_use]
    pub fn new(nzn: usize) -> Self {
        let len = nzn + 2;
        Self {
            source: vec![0.0; len],
            damp: vec![0.0; len],
            dampr: vec![0.0; len],
            dc_dr_in2: vec![0.0; len],
            dc_dr_in: vec![0.0; len],
            dc_dr_00: vec![0.0; len],
            dc_dr_out: vec![0.0; len],
            dc_dt_in: vec![0.0; len],
            dc_dt_00: vec![0.0; len],
            dc_dt_out: vec![0.0; len],
            dc_dw_00: vec![0.0; len],
            lc: vec![0.0; len],
            dlcxm: vec![0.0; len],
            dlcx0: vec![0.0; len],
            dlcxp: vec![0.0; len],
            dlcy0: vec![0.0; len],
            dlcyp: vec![0.0; len],
            dlcz0: vec![0.0; len],
            dlczp: vec![0.0; len],
            lt: vec![0.0; len],
            dltxm: vec![0.0; len],
            dltx0: vec![0.0; len],
            dltxp: vec![0.0; len],
            dlty0: vec![0.0; len],
            dltyp: vec![0.0; len],
            dltz0: vec![0.0; len],
            dltzp: vec![0.0; len],
            pturb: vec![0.0; len],
            dptrb_dw_00: vec![0.0; len],
            dptrb_dr_00: vec![0.0; len],
            dptrb_dr_in: vec![0.0; len],
            evuu0: vec![0.0; len],
            evuum: vec![0.0; len],
        }
    }

    /// Evaluates all convective, turbulent, pressure, and eddy-viscosity terms
    /// and partials across `1..=nzn` (`rsp_relax_env.f90:568-988` and `rsp_lina.f90:257-674`).
    #[allow(clippy::too_many_lines)]
    pub fn evaluate(
        &mut self,
        dc: &RspDerivedConsts,
        alfa: f64,
        alfap: f64,
        alfat: f64,
        efl02: f64,
        grid: &mut RspEnvGrid,
        mc: &RspZoneMicro,
        mode: LcFilterMode,
    ) {
        let nzn = grid.nzn;
        for v in [
            &mut self.source,
            &mut self.damp,
            &mut self.dampr,
            &mut self.dc_dr_in2,
            &mut self.dc_dr_in,
            &mut self.dc_dr_00,
            &mut self.dc_dr_out,
            &mut self.dc_dt_in,
            &mut self.dc_dt_00,
            &mut self.dc_dt_out,
            &mut self.dc_dw_00,
            &mut self.lc,
            &mut self.dlcxm,
            &mut self.dlcx0,
            &mut self.dlcxp,
            &mut self.dlcy0,
            &mut self.dlcyp,
            &mut self.dlcz0,
            &mut self.dlczp,
            &mut self.lt,
            &mut self.dltxm,
            &mut self.dltx0,
            &mut self.dltxp,
            &mut self.dlty0,
            &mut self.dltyp,
            &mut self.dltz0,
            &mut self.dltzp,
            &mut self.pturb,
            &mut self.dptrb_dw_00,
            &mut self.dptrb_dr_00,
            &mut self.dptrb_dr_in,
            &mut self.evuu0,
            &mut self.evuum,
        ] {
            v.fill(0.0);
        }

        if alfa == 0.0 {
            grid.et[1..=nzn].fill(0.0);
            return;
        }

        grid.et[nzn] = 0.0;
        grid.et[1..=dc.ibotom.min(nzn)].fill(0.0);

        let hp = eval_hp_face_partials(dc, grid, mc);
        let len = nzn + 2;
        let mut pii = vec![0.0_f64; len];
        let mut dpii_dr_in = vec![0.0_f64; len];
        let mut dpii_dr_00 = vec![0.0_f64; len];
        let mut dpii_dr_out = vec![0.0_f64; len];
        let mut dpii_dt_00 = vec![0.0_f64; len];
        let mut dpii_dt_out = vec![0.0_f64; len];

        for i in 1..nzn {
            let qc_i = mc.qqs[i] / mc.cps[i];
            let qc_ip1 = mc.qqs[i + 1] / mc.cps[i + 1];
            let pom = 0.5 * (qc_i + qc_ip1);
            let pom2 = 0.5 * (mc.p[i + 1] - mc.p[i]);
            let igr1 = pom * (mc.p[i + 1] - mc.p[i])
                - (grid.t[i + 1].ln() - grid.t[i].ln());
            let igr1x0 = pom2
                * ((mc.dqq_dr_00[i] - qc_i * mc.dcp_dr_00[i]) / mc.cps[i]
                    + (mc.dqq_dr_in[i + 1] - qc_ip1 * mc.dcp_dr_in[i + 1])
                        / mc.cps[i + 1])
                + pom * (mc.dp_dr_in[i + 1] - mc.dp_dr_00[i]);
            let igr1xm = pom2 * (mc.dqq_dr_in[i] - qc_i * mc.dcp_dr_in[i])
                / mc.cps[i]
                - pom * mc.dp_dr_in[i];
            let igr1xp = pom2
                * (mc.dqq_dr_00[i + 1] - qc_ip1 * mc.dcp_dr_00[i + 1])
                / mc.cps[i + 1]
                + pom * mc.dp_dr_00[i + 1];
            let igr1y0 = pom2 * (mc.dqq_dt_00[i] - qc_i * mc.dcp_dt_00[i])
                / mc.cps[i]
                - pom * mc.dp_dt_00[i]
                + 1.0 / grid.t[i];
            let igr1yp = pom2
                * (mc.dqq_dt_00[i + 1] - qc_ip1 * mc.dcp_dt_00[i + 1])
                / mc.cps[i + 1]
                + pom * mc.dp_dt_00[i + 1]
                - 1.0 / grid.t[i + 1];

            let vsum = grid.vol[i] + grid.vol[i + 1];
            let pom_v = 2.0 / vsum;
            let pom2_v = 8.0 * PI * grid.r[i] * grid.r[i] / grid.dm_bar[i]
                * grid.hp_face[i];
            let igr2 =
                4.0 * PI * grid.r[i] * grid.r[i] * grid.hp_face[i] * pom_v
                    / grid.dm_bar[i];
            let igr2x0 = 2.0 * igr2 / grid.r[i]
                + igr2 / grid.hp_face[i] * hp.dr_00[i]
                - pom2_v / (vsum * vsum) * (mc.dvr[i] + mc.dvrm[i + 1]);
            let igr2xm = -pom2_v / (vsum * vsum) * mc.dvrm[i]
                + igr2 / grid.hp_face[i] * hp.dr_in[i];
            let igr2xp = -pom2_v / (vsum * vsum) * mc.dvr[i + 1]
                + igr2 / grid.hp_face[i] * hp.dr_out[i];
            let igr2y0 = igr2 / grid.hp_face[i] * hp.dt_00[i];
            let igr2yp = igr2 / grid.hp_face[i] * hp.dt_out[i];

            grid.y_face[i] = igr1 * igr2;
            let dy_dr_00 = igr1 * igr2x0 + igr2 * igr1x0;
            let dy_dr_in = igr1 * igr2xm + igr2 * igr1xm;
            let dy_dr_out = igr1 * igr2xp + igr2 * igr1xp;
            let dy_dt_00 = igr1 * igr2y0 + igr2 * igr1y0;
            let dy_dt_out = igr1 * igr2yp + igr2 * igr1yp;

            let c_scale = dc.alfas * alfa;
            let cp_avg = 0.5 * (mc.cps[i] + mc.cps[i + 1]);
            pii[i] = c_scale * cp_avg * grid.y_face[i];
            dpii_dr_in[i] = c_scale
                * (cp_avg * dy_dr_in + grid.y_face[i] * 0.5 * mc.dcp_dr_in[i]);
            dpii_dr_00[i] = c_scale
                * (cp_avg * dy_dr_00
                    + grid.y_face[i]
                        * 0.5
                        * (mc.dcp_dr_00[i] + mc.dcp_dr_in[i + 1]));
            dpii_dr_out[i] = c_scale
                * (cp_avg * dy_dr_out
                    + grid.y_face[i] * 0.5 * mc.dcp_dr_00[i + 1]);
            dpii_dt_00[i] = c_scale
                * (cp_avg * dy_dt_00 + grid.y_face[i] * 0.5 * mc.dcp_dt_00[i]);
            dpii_dt_out[i] = c_scale
                * (cp_avg * dy_dt_out
                    + grid.y_face[i] * 0.5 * mc.dcp_dt_00[i + 1]);
        }

        for i in (dc.ibotom + 1)..nzn {
            let pom = 0.5
                * (pii[i] / grid.hp_face[i] + pii[i - 1] / grid.hp_face[i - 1]);
            let pom2 = grid.t[i] * mc.p[i] * mc.qqs[i] / mc.cps[i];
            let pom3 = grid.et[i].sqrt();
            self.source[i] = pom * pom2 * pom3;
            let tem1 = 0.5 * pom2 * pom3;
            let temi = -pii[i] / (grid.hp_face[i] * grid.hp_face[i]);
            let temm =
                -pii[i - 1] / (grid.hp_face[i - 1] * grid.hp_face[i - 1]);

            let dsrc_dr_out =
                tem1 * (dpii_dr_out[i] / grid.hp_face[i] + temi * hp.dr_out[i]);
            let mut dsrc_dr_00 = tem1
                * (dpii_dr_00[i] / grid.hp_face[i]
                    + dpii_dr_out[i - 1] / grid.hp_face[i - 1]
                    + temi * hp.dr_00[i]
                    + temm * hp.dr_out[i - 1]);
            let mut dsrc_dr_in = tem1
                * (dpii_dr_in[i] / grid.hp_face[i]
                    + dpii_dr_00[i - 1] / grid.hp_face[i - 1]
                    + temi * hp.dr_in[i]
                    + temm * hp.dr_00[i - 1]);
            let dsrc_dr_in2 = tem1
                * (dpii_dr_in[i - 1] / grid.hp_face[i - 1]
                    + temm * hp.dr_in[i - 1]);
            let dsrc_dt_out =
                tem1 * (dpii_dt_out[i] / grid.hp_face[i] + temi * hp.dt_out[i]);
            let mut dsrc_dt_00 = tem1
                * (dpii_dt_00[i] / grid.hp_face[i]
                    + dpii_dt_out[i - 1] / grid.hp_face[i - 1]
                    + temi * hp.dt_00[i]
                    + temm * hp.dt_out[i - 1]);
            let dsrc_dt_in = tem1
                * (dpii_dt_00[i - 1] / grid.hp_face[i - 1]
                    + temm * hp.dt_00[i - 1]);
            let dsrc_dw_00 = if grid.et[i] == 0.0 {
                0.0
            } else {
                0.5 * self.source[i] / grid.et[i]
            };

            let tem2 = pom * pom3;
            dsrc_dr_00 += tem2
                * grid.t[i]
                * (mc.dp_dr_00[i] * mc.qqs[i] / mc.cps[i]
                    + mc.p[i]
                        * (mc.dqq_dr_00[i] / mc.cps[i]
                            - mc.qqs[i] * mc.dcp_dr_00[i]
                                / (mc.cps[i] * mc.cps[i])));
            dsrc_dr_in += tem2
                * grid.t[i]
                * (mc.dp_dr_in[i] * mc.qqs[i] / mc.cps[i]
                    + mc.p[i]
                        * (mc.dqq_dr_in[i] / mc.cps[i]
                            - mc.qqs[i] * mc.dcp_dr_in[i]
                                / (mc.cps[i] * mc.cps[i])));
            dsrc_dt_00 += tem2
                * (pom2 / grid.t[i]
                    + grid.t[i]
                        * (mc.dp_dt_00[i] * mc.qqs[i] / mc.cps[i]
                            + mc.p[i]
                                * (mc.dqq_dt_00[i] / mc.cps[i]
                                    - mc.qqs[i] * mc.dcp_dt_00[i]
                                        / (mc.cps[i] * mc.cps[i]))));

            let pom_d =
                (dc.cede / alfa) * (grid.et[i].powf(1.5) - efl02.powf(1.5));
            let pom2_d = 0.5 * (grid.hp_face[i] + grid.hp_face[i - 1]);
            self.damp[i] = pom_d / pom2_d;
            let tem3 = -0.5 * pom_d / (pom2_d * pom2_d);
            let dd_dr_out = tem3 * hp.dr_out[i];
            let dd_dr_00 = tem3 * (hp.dr_00[i] + hp.dr_out[i - 1]);
            let dd_dr_in = tem3 * (hp.dr_in[i] + hp.dr_00[i - 1]);
            let dd_dr_in2 = tem3 * hp.dr_in[i - 1];
            let dd_dt_out = tem3 * hp.dt_out[i];
            let dd_dt_00 = tem3 * (hp.dt_00[i] + hp.dt_out[i - 1]);
            let dd_dt_in = tem3 * hp.dt_00[i - 1];
            let dd_dw_00 = 1.5 * (dc.cede / alfa) / pom2_d * grid.et[i].sqrt();

            let (
                mut ddr_dr_out,
                mut ddr_dr_00,
                mut ddr_dr_in,
                mut ddr_dr_in2,
                mut ddr_dt_out,
                mut ddr_dt_00,
                mut ddr_dt_in,
                mut ddr_dw_00,
            ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            if dc.gammar != 0.0 {
                let pr = 4.0 * dc.sig * dc.gammar * dc.gammar / (alfa * alfa);
                let pr2 = grid.t[i].powi(3) * grid.vol[i].powi(2)
                    / (mc.cps[i] * mc.k[i]);
                let pr4 = 0.5
                    * (grid.hp_face[i].powi(2) + grid.hp_face[i - 1].powi(2));
                self.dampr[i] = pr * pr2 * grid.et[i] / pr4;
                let tr1 = -self.dampr[i] / pr4;
                ddr_dr_out = tr1 * (grid.hp_face[i] * hp.dr_out[i]);
                ddr_dr_00 = tr1
                    * (grid.hp_face[i] * hp.dr_00[i]
                        + grid.hp_face[i - 1] * hp.dr_out[i - 1]);
                ddr_dr_in = tr1
                    * (grid.hp_face[i] * hp.dr_in[i]
                        + grid.hp_face[i - 1] * hp.dr_00[i - 1]);
                ddr_dr_in2 = tr1 * (grid.hp_face[i - 1] * hp.dr_in[i - 1]);
                ddr_dt_out = tr1 * (grid.hp_face[i] * hp.dt_out[i]);
                ddr_dt_00 = tr1
                    * (grid.hp_face[i] * hp.dt_00[i]
                        + grid.hp_face[i - 1] * hp.dt_out[i - 1]);
                ddr_dt_in = tr1 * (grid.hp_face[i - 1] * hp.dt_00[i - 1]);
                ddr_dw_00 = pr * pr2 / pr4;

                let tr2 = pr * grid.et[i] / pr4;
                let cpk = mc.cps[i] * mc.k[i];
                ddr_dr_00 += tr2
                    * grid.t[i].powi(3)
                    * (2.0 * grid.vol[i] * mc.dvr[i]
                        - grid.vol[i]
                            * grid.vol[i]
                            * (mc.dcp_dr_00[i] / mc.cps[i]
                                + mc.dk_dv_00[i] * mc.dvr[i] / mc.k[i]))
                    / cpk;
                ddr_dr_in += tr2
                    * grid.t[i].powi(3)
                    * (2.0 * grid.vol[i] * mc.dvrm[i]
                        - grid.vol[i]
                            * grid.vol[i]
                            * (mc.dcp_dr_in[i] / mc.cps[i]
                                + mc.dk_dv_00[i] * mc.dvrm[i] / mc.k[i]))
                    / cpk;
                ddr_dt_00 += tr2
                    * grid.vol[i]
                    * grid.vol[i]
                    * (3.0 * grid.t[i] * grid.t[i]
                        - grid.t[i].powi(3)
                            * (mc.dcp_dt_00[i] / mc.cps[i]
                                + mc.dk_dt_00[i] / mc.k[i]))
                    / cpk;
            }

            self.dc_dr_00[i] = dsrc_dr_00 - dd_dr_00 - ddr_dr_00;
            self.dc_dr_out[i] = dsrc_dr_out - dd_dr_out - ddr_dr_out;
            self.dc_dr_in[i] = dsrc_dr_in - dd_dr_in - ddr_dr_in;
            self.dc_dr_in2[i] = dsrc_dr_in2 - dd_dr_in2 - ddr_dr_in2;
            self.dc_dt_in[i] = dsrc_dt_in - dd_dt_in - ddr_dt_in;
            self.dc_dt_00[i] = dsrc_dt_00 - dd_dt_00 - ddr_dt_00;
            self.dc_dt_out[i] = dsrc_dt_out - dd_dt_out - ddr_dt_out;
            self.dc_dw_00[i] = dsrc_dw_00 - dd_dw_00 - ddr_dw_00;
        }

        for i in dc.ibotom.max(1)..nzn {
            let pom = 4.0
                * PI
                * grid.r[i]
                * grid.r[i]
                * 0.5
                * (grid.t[i] / grid.vol[i] + grid.t[i + 1] / grid.vol[i + 1])
                * (dc.alfac / dc.alfas);
            let pom3 = 0.5 * (grid.et[i].sqrt() + grid.et[i + 1].sqrt());
            self.lc[i] = pom * pii[i] * pom3;
            if grid.et[i] != 0.0 {
                self.dlcz0[i] = pom * pii[i] * 0.25 / grid.et[i].sqrt();
            }
            if grid.et[i + 1] != 0.0 {
                self.dlczp[i] = pom * pii[i] * 0.25 / grid.et[i + 1].sqrt();
            }
            let pom2 = 4.0
                * PI
                * grid.r[i]
                * grid.r[i]
                * pii[i]
                * 0.5
                * (dc.alfac / dc.alfas)
                * pom3;
            let pom_w = pom * pom3;
            self.dlcx0[i] = pom_w * dpii_dr_00[i]
                + 2.0 * self.lc[i] / grid.r[i]
                - pom2
                    * (grid.t[i] / (grid.vol[i] * grid.vol[i]) * mc.dvr[i]
                        + grid.t[i + 1] / (grid.vol[i + 1] * grid.vol[i + 1])
                            * mc.dvrm[i + 1]);
            self.dlcxm[i] = pom_w * dpii_dr_in[i]
                - pom2 * grid.t[i] / (grid.vol[i] * grid.vol[i]) * mc.dvrm[i];
            self.dlcxp[i] = pom_w * dpii_dr_out[i]
                - pom2 * grid.t[i + 1] / (grid.vol[i + 1] * grid.vol[i + 1])
                    * mc.dvr[i + 1];
            self.dlcy0[i] = pom_w * dpii_dt_00[i] + pom2 / grid.vol[i];
            self.dlcyp[i] = pom_w * dpii_dt_out[i] + pom2 / grid.vol[i + 1];

            if i == dc.ibotom {
                self.dlcz0[i] = 0.0;
            }
            if i == nzn - 1 {
                self.dlczp[i] = 0.0;
            }
            match mode {
                LcFilterMode::Relax { negflu, iter_ii } => {
                    if negflu == 0 && pii[i] < 0.0 {
                        (
                            self.lc[i],
                            self.dlcx0[i],
                            self.dlcxm[i],
                            self.dlcxp[i],
                            self.dlcy0[i],
                            self.dlcyp[i],
                            self.dlcz0[i],
                            self.dlczp[i],
                        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                    } else if negflu == 1 && pii[i] < 0.0 && iter_ii > 300 {
                        self.dlcz0[i] = 0.0;
                        self.dlczp[i] = 0.0;
                    }
                }
                LcFilterMode::Lina { efl02: lim } => {
                    if grid.et[i] < lim * 1.0e-10 {
                        (
                            self.lc[i],
                            self.dlcx0[i],
                            self.dlcxm[i],
                            self.dlcxp[i],
                            self.dlcy0[i],
                            self.dlcyp[i],
                            self.dlcz0[i],
                            self.dlczp[i],
                        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                    }
                    if i <= dc.ibotom {
                        self.lc[i] = 0.0;
                    }
                }
            }

            if alfat != 0.0 {
                let pt = -(2.0 / 3.0)
                    * alfa
                    * alfat
                    * (4.0 * PI * grid.r[i].powi(2)).powi(2);
                let pt2 = grid.hp_face[i]
                    * 0.5
                    * (1.0 / grid.vol[i].powi(2)
                        + 1.0 / grid.vol[i + 1].powi(2));
                let pt3 = (grid.et[i + 1].powf(1.5) - grid.et[i].powf(1.5))
                    / grid.dm_bar[i];
                self.lt[i] = pt * pt2 * pt3;
                self.dltx0[i] = 4.0 * self.lt[i] / grid.r[i]
                    + self.lt[i] / pt2
                        * grid.hp_face[i]
                        * (-mc.dvr[i] / grid.vol[i].powi(3)
                            - mc.dvrm[i + 1] / grid.vol[i + 1].powi(3))
                    + self.lt[i] / grid.hp_face[i] * hp.dr_00[i];
                self.dltxm[i] = self.lt[i] / pt2
                    * grid.hp_face[i]
                    * (-mc.dvrm[i] / grid.vol[i].powi(3))
                    + self.lt[i] / grid.hp_face[i] * hp.dr_in[i];
                self.dltxp[i] = self.lt[i] / pt2
                    * grid.hp_face[i]
                    * (-mc.dvr[i + 1] / grid.vol[i + 1].powi(3))
                    + self.lt[i] / grid.hp_face[i] * hp.dr_out[i];
                self.dlty0[i] = self.lt[i] / grid.hp_face[i] * hp.dt_00[i];
                self.dltyp[i] = self.lt[i] / grid.hp_face[i] * hp.dt_out[i];
                self.dltz0[i] =
                    -pt * pt2 * 1.5 * grid.et[i].sqrt() / grid.dm_bar[i];
                self.dltzp[i] =
                    pt * pt2 * 1.5 * grid.et[i + 1].sqrt() / grid.dm_bar[i];
                if matches!(mode, LcFilterMode::Lina { .. }) {
                    if i <= dc.ibotom {
                        self.lt[i] = 0.0;
                    }
                    if i == nzn - 1 {
                        self.dltzp[i] = 0.0;
                    }
                }
            }
        }

        for i in (dc.ibotom + 1)..nzn {
            if alfap != 0.0 {
                self.pturb[i] = alfap * grid.et[i] / grid.vol[i];
                self.dptrb_dw_00[i] = alfap / grid.vol[i];
                let tem1 = -alfap * grid.et[i] / (grid.vol[i] * grid.vol[i]);
                self.dptrb_dr_00[i] = tem1 * mc.dvr[i];
                self.dptrb_dr_in[i] = tem1 * mc.dvrm[i];
            }
            if dc.alfam >= 0.0 {
                let pom =
                    (16.0 / 3.0) * PI * alfa * dc.alfam * grid.et[i].sqrt();
                let pom1 = 1.0 / (grid.vol[i] * grid.vol[i] * grid.dm[i]);
                let pom2 = (grid.r[i].powi(6) + grid.r[i - 1].powi(6))
                    * (grid.hp_face[i] + grid.hp_face[i - 1])
                    * 0.25;
                self.evuu0[i] = pom * pom1 * pom2 / grid.r[i];
                self.evuum[i] = -pom * pom1 * pom2 / grid.r[i - 1];
            } else {
                let pom = -(16.0 / 3.0)
                    * PI
                    * alfa
                    * dc.alfam.abs()
                    * grid.et[i].sqrt();
                let pom1 = 1.0 / (grid.vol[i] * grid.vol[i] * grid.dm[i]);
                let pom2 = (grid.r[i].powi(3) + grid.r[i - 1].powi(3))
                    * (grid.hp_face[i] + grid.hp_face[i - 1])
                    * 0.25;
                self.evuu0[i] = pom * pom1 * pom2 / grid.r[i];
                self.evuum[i] = -pom * pom1 * pom2 / grid.r[i - 1];
            }
        }
    }
}
