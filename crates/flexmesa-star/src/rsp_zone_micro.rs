// SPDX-License-Identifier: LGPL-3.0-or-later
//! Per-zone (`1..=nzn`) microphysical chain-rule derivatives (`RspZoneMicro`),
//! face pressure scale-height partials (`HpFacePartials`), and convective-flux
//! filter mode (`LcFilterMode`) shared across `RELAX_ENV` and `do_LINA`.

use crate::rsp_def::RspDerivedConsts;
use crate::rsp_grid::RspEnvGrid;

/// Per-zone (`1..=nzn`) microphysical & volumetric partials populated prior to
/// evaluating convective/turbulent face and zone derivatives.
#[derive(Debug, Clone)]
pub struct RspZoneMicro {
    /// Total pressure $P_i$.
    pub p: Vec<f64>,
    /// $\partial P_i / \partial V_i$.
    pub dpv: Vec<f64>,
    /// $\partial P_i / \partial T_i$.
    pub dp_dt_00: Vec<f64>,
    /// Total specific energy $e_i$.
    pub e: Vec<f64>,
    /// $\partial e_i / \partial V_i$.
    pub dev: Vec<f64>,
    /// $\partial e_i / \partial T_i$.
    pub de_dt_00: Vec<f64>,
    /// Specific heat $C_{P,i}$.
    pub cps: Vec<f64>,
    /// $\partial C_{P,i} / \partial V_i$.
    pub cpv: Vec<f64>,
    /// $\partial C_{P,i} / \partial T_i$.
    pub dcp_dt_00: Vec<f64>,
    /// Thermal expansion $Q_i$.
    pub qqs: Vec<f64>,
    /// $\partial Q_i / \partial V_i$.
    pub qqv: Vec<f64>,
    /// $\partial Q_i / \partial T_i$.
    pub dqq_dt_00: Vec<f64>,
    /// Opacity $\kappa_i$.
    pub k: Vec<f64>,
    /// $\partial \kappa_i / \partial V_i$.
    pub dk_dv_00: Vec<f64>,
    /// $\partial \kappa_i / \partial T_i$.
    pub dk_dt_00: Vec<f64>,
    /// $\partial V_i / \partial R_i$.
    pub dvr: Vec<f64>,
    /// $\partial V_i / \partial R_{i-1}$.
    pub dvrm: Vec<f64>,
    /// $\partial P_i / \partial R_i$.
    pub dp_dr_00: Vec<f64>,
    /// $\partial P_i / \partial R_{i-1}$.
    pub dp_dr_in: Vec<f64>,
    /// $\partial C_{P,i} / \partial R_i$.
    pub dcp_dr_00: Vec<f64>,
    /// $\partial C_{P,i} / \partial R_{i-1}$.
    pub dcp_dr_in: Vec<f64>,
    /// $\partial Q_i / \partial R_i$.
    pub dqq_dr_00: Vec<f64>,
    /// $\partial Q_i / \partial R_{i-1}$.
    pub dqq_dr_in: Vec<f64>,
}

impl RspZoneMicro {
    /// Allocates zero-initialized microphysical buffers for `nzn` zones.
    #[must_use]
    pub fn new(nzn: usize) -> Self {
        let len = nzn + 2;
        Self {
            p: vec![0.0; len],
            dpv: vec![0.0; len],
            dp_dt_00: vec![0.0; len],
            e: vec![0.0; len],
            dev: vec![0.0; len],
            de_dt_00: vec![0.0; len],
            cps: vec![0.0; len],
            cpv: vec![0.0; len],
            dcp_dt_00: vec![0.0; len],
            qqs: vec![0.0; len],
            qqv: vec![0.0; len],
            dqq_dt_00: vec![0.0; len],
            k: vec![0.0; len],
            dk_dv_00: vec![0.0; len],
            dk_dt_00: vec![0.0; len],
            dvr: vec![0.0; len],
            dvrm: vec![0.0; len],
            dp_dr_00: vec![0.0; len],
            dp_dr_in: vec![0.0; len],
            dcp_dr_00: vec![0.0; len],
            dcp_dr_in: vec![0.0; len],
            dqq_dr_00: vec![0.0; len],
            dqq_dr_in: vec![0.0; len],
        }
    }

    /// Updates radial chain-rule derivatives `(dP_dr, dCp_dr, dQQ_dr)` at zone `i`.
    pub fn update_radial_chain(&mut self, i: usize) {
        self.dp_dr_00[i] = self.dpv[i] * self.dvr[i];
        self.dp_dr_in[i] = self.dpv[i] * self.dvrm[i];
        self.dcp_dr_00[i] = self.cpv[i] * self.dvr[i];
        self.dcp_dr_in[i] = self.cpv[i] * self.dvrm[i];
        self.dqq_dr_00[i] = self.qqv[i] * self.dvr[i];
        self.dqq_dr_in[i] = self.qqv[i] * self.dvrm[i];
    }
}

/// Convective-flux filtering mode distinguishing `RELAX_ENV` continuation stages
/// from `do_LINA` (`rsp_relax_env.f90:845-860` vs. `rsp_lina.f90:519-528`).
#[derive(Debug, Clone, Copy)]
pub enum LcFilterMode {
    /// Static `RELAX_ENV` negative-flux continuation stage.
    Relax {
        /// Continuation flag (`0`, `1`, or `2`).
        negflu: i32,
        /// Inner Newton iteration counter `II`.
        iter_ii: usize,
    },
    /// Linear `LINA` analysis with flux-limiter floor `efl02`.
    Lina {
        /// Squared flux-limiter floor `EFL0^2`.
        efl02: f64,
    },
}

/// Face pressure scale-height partial derivatives (`1..=nzn`).
#[derive(Debug, Clone)]
pub struct HpFacePartials {
    /// $\partial H_{p,i} / \partial R_{i-1}$.
    pub dr_in: Vec<f64>,
    /// $\partial H_{p,i} / \partial R_i$.
    pub dr_00: Vec<f64>,
    /// $\partial H_{p,i} / \partial R_{i+1}$.
    pub dr_out: Vec<f64>,
    /// $\partial H_{p,i} / \partial T_i$.
    pub dt_00: Vec<f64>,
    /// $\partial H_{p,i} / \partial T_{i+1}$.
    pub dt_out: Vec<f64>,
}

/// Evaluates `grid.hp_face[1..=nzn]` and its `(R, T)` partial derivatives
/// (`rsp_relax_env.f90:576-604`, `rsp_lina.f90:263-290`).
pub fn eval_hp_face_partials(
    dc: &RspDerivedConsts,
    grid: &mut RspEnvGrid,
    mc: &RspZoneMicro,
) -> HpFacePartials {
    let nzn = grid.nzn;
    let len = nzn + 2;
    let mut hp = HpFacePartials {
        dr_in: vec![0.0; len],
        dr_00: vec![0.0; len],
        dr_out: vec![0.0; len],
        dt_00: vec![0.0; len],
        dt_out: vec![0.0; len],
    };
    for i in 1..nzn {
        let pom = grid.r[i] * grid.r[i] / (2.0 * dc.g * grid.m[i]);
        grid.hp_face[i] =
            pom * (mc.p[i] * grid.vol[i] + mc.p[i + 1] * grid.vol[i + 1]);
        hp.dr_in[i] =
            pom * (mc.p[i] * mc.dvrm[i] + grid.vol[i] * mc.dp_dr_in[i]);
        hp.dr_00[i] = 2.0 * grid.hp_face[i] / grid.r[i]
            + pom
                * (mc.p[i] * mc.dvr[i]
                    + grid.vol[i] * mc.dp_dr_00[i]
                    + mc.p[i + 1] * mc.dvrm[i + 1]
                    + grid.vol[i + 1] * mc.dp_dr_in[i + 1]);
        hp.dr_out[i] = pom
            * (mc.p[i + 1] * mc.dvr[i + 1]
                + grid.vol[i + 1] * mc.dp_dr_00[i + 1]);
        hp.dt_00[i] = pom * grid.vol[i] * mc.dp_dt_00[i];
        hp.dt_out[i] = pom * grid.vol[i + 1] * mc.dp_dt_00[i + 1];
    }
    let pom_n = grid.r[nzn] * grid.r[nzn] / (2.0 * dc.g * grid.m[nzn]);
    grid.hp_face[nzn] = pom_n * mc.p[nzn] * grid.vol[nzn];
    hp.dr_in[nzn] =
        pom_n * (mc.p[nzn] * mc.dvrm[nzn] + grid.vol[nzn] * mc.dp_dr_in[nzn]);
    hp.dr_00[nzn] = 2.0 * grid.hp_face[nzn] / grid.r[nzn]
        + pom_n * (mc.p[nzn] * mc.dvr[nzn] + grid.vol[nzn] * mc.dp_dr_00[nzn]);
    hp.dt_00[nzn] = pom_n * grid.vol[nzn] * mc.dp_dt_00[nzn];
    hp
}
