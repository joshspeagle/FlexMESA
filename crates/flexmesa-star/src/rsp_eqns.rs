// SPDX-License-Identifier: LGPL-3.0-or-later
//! Nonlinear `RSP` `HYD` face equations (`acceleration_eqn`, `T_form_of_Fr_eqn`)
//! and 21-diagonal banded Newton solver (`star/private/rsp_step.f90:60-196,
//! 744-942, 2585-3120, 3760-3833`).

use flexmesa_mtx::{dgbtrf, dgbtrs};

use crate::controls::StarError;
use crate::model::StarModel;
use crate::rsp_def::RspDerivedConsts;
use crate::rsp_eqns_cell::{
    t_form_of_erad_eqn, total_energy_eqn, turbulent_energy_eqn,
};
use crate::rsp_hyd_flux::{calc_lt, rsp_calc_xp, t_form_of_calc_fr};
use crate::rsp_hyd_work::RspHydWork;

/// Unknowns per `RSP` shell (`T, w, erad, Fr, R`).
pub const NV: usize = 5;
/// Jacobian band column count (`2 * 2 * NV + 1 = 21`).
pub const LD_HD: usize = 21;
/// 1-based diagonal column in `HD` (`11`).
pub const HD_DIAG: usize = 11;
/// Banded LU storage row count (`2 * kl + ku + 1 = 31` for `kl = ku = 10`).
pub const LD_ABB: usize = 31;

const I_VAR_T: usize = 2;
const I_VAR_W: usize = 3;
const I_VAR_ER: usize = 4;
const I_VAR_FR: usize = 5;
const I_VAR_R: usize = 6;

const I_R_DR_IN2: usize = 1;
const I_R_DT_IN: usize = 2;
const I_R_DW_IN: usize = 3;
const I_R_DER_IN: usize = 4;
const I_R_DR_IN: usize = I_R_DR_IN2 + NV;
const I_R_DR_00: usize = I_R_DR_IN + NV;
const I_R_DR_OUT: usize = I_R_DR_00 + NV;
const I_R_DR_OUT2: usize = I_R_DR_OUT + NV;
const I_R_DT_00: usize = I_R_DT_IN + NV;
const I_R_DT_OUT: usize = I_R_DT_00 + NV;
const I_R_DT_OUT2: usize = I_R_DT_OUT + NV;
const I_R_DW_00: usize = I_R_DW_IN + NV;
const I_R_DW_OUT: usize = I_R_DW_00 + NV;
const I_R_DW_OUT2: usize = I_R_DW_OUT + NV;
const I_R_DER_00: usize = I_R_DER_IN + NV;
const I_R_DER_OUT: usize = I_R_DER_00 + NV;
const I_R_DER_OUT2: usize = I_R_DER_OUT + NV;

const I_FR_DFR_00: usize = 1 + 2 * NV;
const I_FR_DR_IN: usize = 2 + NV;
const I_FR_DR_00: usize = I_FR_DR_IN + NV;
const I_FR_DR_OUT: usize = I_FR_DR_00 + NV;
const I_FR_DT_00: usize = 3 + NV;
const I_FR_DT_OUT: usize = I_FR_DT_00 + NV;

/// Banded Newton linear-system buffers (`HR`, `HD`, `ABB`, `IPVT`).
#[derive(Debug, Clone)]
pub struct RspHydBand {
    /// Linear dimension `N = 5 * NZN + 1`.
    pub n: usize,
    /// 1-based RHS `-residual` on input (`1..=n`), Newton step `DX` after solve.
    pub hr: Vec<f64>,
    /// 1-based banded Jacobian `HD[row][col_band]` (`row = 1..=n`, `col = 1..=21`).
    pub hd: Vec<[f64; LD_HD + 1]>,
    ab: Vec<f64>,
    ipiv: Vec<usize>,
}

impl RspHydBand {
    /// Allocates banded Newton buffers for `nzn` envelope zones.
    #[must_use]
    pub fn new(nzn: usize) -> Self {
        let n = NV * nzn + 1;
        Self {
            n,
            hr: vec![0.0; n + 1],
            hd: vec![[0.0; LD_HD + 1]; n + 1],
            ab: vec![0.0; LD_ABB * n],
            ipiv: vec![0; n],
        }
    }

    /// Returns 1-based equation indices `(it, iw, ie, il, ir)` for shell `i` (`1..=nzn`).
    #[inline]
    #[must_use]
    pub fn row_indices(i: usize) -> (usize, usize, usize, usize, usize) {
        let base = NV * (i - 1);
        (
            I_VAR_T + base,
            I_VAR_W + base,
            I_VAR_ER + base,
            I_VAR_FR + base,
            I_VAR_R + base,
        )
    }
}

/// Evaluates all `5 * NZN + 1` nonlinear `HYD` equations and Jacobian rows
/// (`eval_eqns` in `rsp_step.f90:744-754, 2585-2697`).
pub fn eval_eqns(
    model: &mut StarModel,
    dc: &RspDerivedConsts,
    wk: &RspHydWork,
    band: &mut RspHydBand,
    p_surf: f64,
) {
    for i in 1..=wk.nzn {
        acceleration_eqn(model, dc, wk, band, i, p_surf);
        fr_eqn(model, dc, wk, band, i);
        let lt_in = calc_lt(model, dc, wk, i - 1);
        let lt_00 = calc_lt(model, dc, wk, i);
        let k0 = wk.nzn - i;
        let lt_00_start = if i == wk.nzn {
            0.0
        } else {
            model.aux.rsp.lt_start[k0]
        };
        let lt_in_start = if i == 1 {
            0.0
        } else {
            model.aux.rsp.lt_start[k0 + 1]
        };
        total_energy_eqn(
            model,
            dc,
            wk,
            band,
            i,
            p_surf,
            &lt_00,
            lt_00_start,
            &lt_in,
            lt_in_start,
        );
        turbulent_energy_eqn(
            model,
            dc,
            wk,
            band,
            i,
            &lt_00,
            lt_00_start,
            &lt_in,
            lt_in_start,
        );
        t_form_of_erad_eqn(model, wk, band, i);
        if i == 1 {
            band.hr[1] = 0.0;
            band.hd[1] = [0.0; LD_HD + 1];
            band.hd[1][I_R_DR_00] = 1.0;
        }
    }
}

/// Packs `HD` into row-major LAPACK band layout and solves `HD * DX = HR`
/// (`solve_for_corrections` in `rsp_step.f90:785-942`).
pub fn solve_for_corrections(band: &mut RspHydBand) -> Result<(), StarError> {
    let n = band.n;
    let kl = 2 * NV;
    let ku = 2 * NV;
    band.ab.fill(0.0);
    for r in 1..=n {
        for j_hd in 1..=LD_HD {
            let col_1 = r as isize + j_hd as isize - HD_DIAG as isize;
            if col_1 >= 1 && col_1 <= n as isize {
                let row_0 = LD_ABB - j_hd;
                let col_0 = (col_1 - 1) as usize;
                band.ab[row_0 * n + col_0] = band.hd[r][j_hd];
            }
        }
    }
    dgbtrf(n, kl, ku, &mut band.ab, LD_ABB, &mut band.ipiv).map_err(|e| {
        StarError::ConvergenceFailure(format!("RSP HYD dgbtrf failed: {e:?}"))
    })?;
    dgbtrs(n, kl, ku, &band.ab, LD_ABB, &band.ipiv, &mut band.hr[1..=n])
        .map_err(|e| {
            StarError::ConvergenceFailure(format!(
                "RSP HYD dgbtrs failed: {e:?}"
            ))
        })
}

fn acceleration_eqn(
    model: &StarModel,
    dc: &RspDerivedConsts,
    wk: &RspHydWork,
    band: &mut RspHydBand,
    i: usize,
    p_surf: f64,
) {
    let k0 = wk.nzn - i;
    let (_, _, _, _, ir) = RspHydBand::row_indices(i);
    band.hd[ir] = [0.0; LD_HD + 1];

    let dt = model.dt;
    let xp_out = rsp_calc_xp(model, dc, wk, p_surf, i + 1, true);
    let xp_00 = rsp_calc_xp(model, dc, wk, p_surf, i, true);

    let r_k = model.r[k0];
    let r_st = model.aux.rsp.r_start_rsp[k0];
    let area = dc.p43 * (r_k * r_k + r_k * r_st + r_st * r_st);
    let d_area_dr_00 = dc.p43 * (2.0 * r_k + r_st);
    let dm_bar = model.dm_bar[k0];
    let a_dm = area / dm_bar;
    let dv_dr = 2.0 / dt;
    let dxp_dm = (xp_out.xp - xp_00.xp) / dm_bar;
    let grav = -model.cgrav[k0] * model.m[k0] / (r_k * r_st);

    let r_00 = 0.5 * (r_k + r_st);
    let uq1 = dc.p4 / (dm_bar * r_00);
    let d_uq1_dr_00 = -uq1 * 0.5 / r_00;

    let chi_00 = dc.thetau * model.aux.rsp.chi_rsp[k0]
        + dc.thetau1 * model.aux.rsp.chi_start[k0];
    let d_chi_00_dt_in = dc.thetau * wk.dchi_dt_in[i];
    let d_chi_00_dt_00 = dc.thetau * wk.dchi_dt_00[i];
    let d_chi_00_dt_out = dc.thetau * wk.dchi_dt_out[i];
    let d_chi_00_der_in = dc.thetau * wk.dchi_der_in[i];
    let d_chi_00_der_00 = dc.thetau * wk.dchi_der_00[i];
    let d_chi_00_der_out = dc.thetau * wk.dchi_der_out[i];
    let d_chi_00_dw_00 = dc.thetau * wk.dchi_dw_00[i];
    let d_chi_00_dr_in2 = dc.thetau * wk.dchi_dr_in2[i];
    let d_chi_00_dr_in = dc.thetau * wk.dchi_dr_in[i];
    let d_chi_00_dr_00 = dc.thetau * wk.dchi_dr_00[i];
    let d_chi_00_dr_out = dc.thetau * wk.dchi_dr_out[i];

    let (
        chi_out,
        d_chi_out_dt_00,
        d_chi_out_dt_out,
        d_chi_out_dt_out2,
        d_chi_out_der_00,
        d_chi_out_der_out,
        d_chi_out_der_out2,
        d_chi_out_dw_out,
        d_chi_out_dr_in,
        d_chi_out_dr_00,
        d_chi_out_dr_out,
        d_chi_out_dr_out2,
    ) = if i == wk.nzn {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    } else {
        let ko = k0 - 1;
        (
            dc.thetau * model.aux.rsp.chi_rsp[ko]
                + dc.thetau1 * model.aux.rsp.chi_start[ko],
            dc.thetau * wk.dchi_dt_in[i + 1],
            dc.thetau * wk.dchi_dt_00[i + 1],
            dc.thetau * wk.dchi_dt_out[i + 1],
            dc.thetau * wk.dchi_der_in[i + 1],
            dc.thetau * wk.dchi_der_00[i + 1],
            dc.thetau * wk.dchi_der_out[i + 1],
            dc.thetau * wk.dchi_dw_00[i + 1],
            dc.thetau * wk.dchi_dr_in2[i + 1],
            dc.thetau * wk.dchi_dr_in[i + 1],
            dc.thetau * wk.dchi_dr_00[i + 1],
            dc.thetau * wk.dchi_dr_out[i + 1],
        )
    };

    let uq = uq1 * (chi_out - chi_00);
    let d_uq_dt_in = -uq1 * d_chi_00_dt_in;
    let d_uq_dt_00 = uq1 * (d_chi_out_dt_00 - d_chi_00_dt_00);
    let d_uq_dt_out = uq1 * (d_chi_out_dt_out - d_chi_00_dt_out);
    let d_uq_dt_out2 = uq1 * d_chi_out_dt_out2;
    let d_uq_der_in = -uq1 * d_chi_00_der_in;
    let d_uq_der_00 = uq1 * (d_chi_out_der_00 - d_chi_00_der_00);
    let d_uq_der_out = uq1 * (d_chi_out_der_out - d_chi_00_der_out);
    let d_uq_der_out2 = uq1 * d_chi_out_der_out2;
    let d_uq_dw_00 = -uq1 * d_chi_00_dw_00;
    let d_uq_dw_out = uq1 * d_chi_out_dw_out;
    let d_uq_dr_in2 = -uq1 * d_chi_00_dr_in2;
    let d_uq_dr_in = uq1 * (d_chi_out_dr_in - d_chi_00_dr_in);
    let d_uq_dr_00 = uq1 * (d_chi_out_dr_00 - d_chi_00_dr_00)
        + d_uq1_dr_00 * (chi_out - chi_00);
    let d_uq_dr_out = uq1 * (d_chi_out_dr_out - d_chi_00_dr_out);
    let d_uq_dr_out2 = uq1 * d_chi_out_dr_out2;

    let residual = (model.v[k0] - model.aux.rsp.v_start_rsp[k0]) / dt
        + area * dxp_dm
        - grav
        - uq;
    band.hr[ir] = -residual;

    let h = &mut band.hd[ir];
    h[I_R_DT_IN] = -d_uq_dt_in;
    h[I_R_DT_00] = a_dm * (-xp_00.d_xp_dt_00) - d_uq_dt_00;
    h[I_R_DT_OUT] = a_dm * xp_out.d_xp_dt_00 - d_uq_dt_out;
    h[I_R_DT_OUT2] = -d_uq_dt_out2;

    h[I_R_DR_IN2] = -d_uq_dr_in2;
    h[I_R_DR_IN] = a_dm * (-xp_00.d_xp_dr_in) - d_uq_dr_in;
    h[I_R_DR_00] = dv_dr / dt
        + d_area_dr_00 * dxp_dm
        + a_dm * (xp_out.d_xp_dr_in - xp_00.d_xp_dr_00)
        + grav / r_k
        - d_uq_dr_00
        + 0.5 * uq1 / r_00 * (chi_out - chi_00);
    h[I_R_DR_OUT] = a_dm * xp_out.d_xp_dr_00 - d_uq_dr_out;
    h[I_R_DR_OUT2] = -d_uq_dr_out2;

    h[I_R_DER_IN] = -d_uq_der_in;
    h[I_R_DER_00] = a_dm * (-xp_00.d_xp_der_00) - d_uq_der_00;
    h[I_R_DER_OUT] = a_dm * xp_out.d_xp_der_00 - d_uq_der_out;
    h[I_R_DER_OUT2] = -d_uq_der_out2;

    h[I_R_DW_IN] = 0.0;
    h[I_R_DW_00] = if i <= dc.ibotom || i == wk.nzn {
        0.0
    } else {
        a_dm * (-xp_00.d_xp_dw_00) - d_uq_dw_00
    };
    h[I_R_DW_OUT] = if i < dc.ibotom || i >= wk.nzn - 1 {
        0.0
    } else {
        a_dm * xp_out.d_xp_dw_00 - d_uq_dw_out
    };
    h[I_R_DW_OUT2] = 0.0;
}

fn fr_eqn(
    model: &StarModel,
    dc: &RspDerivedConsts,
    wk: &RspHydWork,
    band: &mut RspHydBand,
    i: usize,
) {
    let k0 = wk.nzn - i;
    let (_, _, _, il, _) = RspHydBand::row_indices(i);
    band.hd[il] = [0.0; LD_HD + 1];
    if model.controls.rsp.hydro_only {
        let residual = -model.aux.rsp.fr_rsp[k0];
        band.hr[il] = -residual;
        band.hd[il][I_FR_DFR_00] = -1.0;
        return;
    }
    let fr = t_form_of_calc_fr(model, dc, wk, i);
    let residual = fr.fr_00 - model.aux.rsp.fr_rsp[k0];
    band.hr[il] = -residual;
    let h = &mut band.hd[il];
    h[I_FR_DFR_00] = -1.0;
    h[I_FR_DR_IN] = fr.dfr_dr_in;
    h[I_FR_DR_00] = fr.dfr_dr_00;
    h[I_FR_DR_OUT] = fr.dfr_dr_out;
    h[I_FR_DT_00] = fr.dfr_dt_00;
    h[I_FR_DT_OUT] = fr.dfr_dt_out;
}
