// SPDX-License-Identifier: LGPL-3.0-or-later
//! Full-profile `RSP2` face/cell turbulent variables, Stellingwerf radiative
//! luminosity, pre-solver equilibrium initialization, and coupled Henyey
//! equation residuals (`do1_rsp2_L_eqn`, `do1_rsp2_Hp_eqn`,
//! `do1_turbulent_energy_eqn`, and `RSP2_adjust_vars_before_call_solver` in
//! `star/private/hydro_rsp2.f90:1-1255`).

use flexmesa_const::{BOLTZ_SIGMA, LN10, PI4};

use crate::model::StarModel;
use crate::rsp2::{X_CEDE, X_GAMMAR};
use crate::rsp2_ad::{
    build_rsp2_ad_work, eval_rsp2_hp_face_ad, eval_rsp2_ptrb_cell_ad,
};

/// Residuals of the three `RSP2` equations across all zones `0..nz`:
/// `equ_l` (`do1_rsp2_L_eqn`), `equ_hp` (`do1_rsp2_Hp_eqn`), and `equ_etrb`
/// (`do1_turbulent_energy_eqn`).
#[derive(Debug, Clone)]
pub struct Rsp2ProfileResiduals {
    /// Dimensionless luminosity residuals $(L_{\mathrm{exp},k} - L_k) / \max L_{\mathrm{start}}$.
    pub equ_l: Vec<f64>,
    /// Dimensionless scale-height residuals $(H_{p,\mathrm{exp},k} - H_{p,k}) / \max(H_{p,\mathrm{exp},k}, H_{p,k})$.
    pub equ_hp: Vec<f64>,
    /// Dimensionless turbulent-energy residuals (`do1_turbulent_energy_eqn`).
    pub equ_etrb: Vec<f64>,
}

#[inline]
pub(crate) fn rsp2_face_weights(model: &StarModel, k: usize) -> (f64, f64) {
    if model.controls.rsp2.use_mass_interp_face_values && k > 0 {
        let denom = (model.dq[k - 1] + model.dq[k]).max(1.0e-99);
        let alfa = model.dq[k - 1] / denom;
        (alfa, 1.0 - alfa)
    } else {
        (0.5, 0.5)
    }
}

#[inline]
pub(crate) fn is_nonturbulent_cell(model: &StarModel, k: usize) -> bool {
    let nz = model.nz;
    let alpha = model.controls.mixing_length_alpha;
    let n_outer = model.controls.rsp2.num_outermost_cells_forced_nonturbulent;
    let ibotom = if model.controls.rsp2.nz_div_ibotom > 0.0 {
        ((nz as f64) / model.controls.rsp2.nz_div_ibotom) as usize
    } else {
        0
    };
    let k_1based = k + 1;
    alpha == 0.0 || k_1based <= n_outer || k_1based > nz.saturating_sub(ibotom)
}

/// Evaluates `Hp_face_for_rsp2_eqn` (`hydro_rsp2.f90:230-285`) at face `k` (`0..nz`).
#[must_use]
pub fn eval_rsp2_hp_face_zone(model: &StarModel, k: usize) -> f64 {
    let r_00 = model.r[k];
    let d_00 = model.rho[k].max(1.0e-99);
    let peos_00 = model.peos[k];
    if k > 0 && !model.controls.rsp2.assume_hse {
        let (alfa, beta) = rsp2_face_weights(model, k);
        let rho_face = alfa * d_00 + beta * model.rho[k - 1].max(1.0e-99);
        let area = PI4 * r_00 * r_00;
        let dlnp = model.ln_peos[k - 1] - model.ln_peos[k];
        if dlnp.abs() > 1.0e-30 {
            return (-model.dm_bar[k] / (area * rho_face * dlnp)).max(1.0);
        }
    }
    let peos_div_rho = if k == 0 {
        peos_00 / d_00
    } else {
        let (alfa, beta) = rsp2_face_weights(model, k);
        alfa * peos_00 / d_00
            + beta * model.peos[k - 1] / model.rho[k - 1].max(1.0e-99)
    };
    let gm = (model.cgrav[k] * model.m[k]).max(1.0e-30);
    (r_00 * r_00 * peos_div_rho / gm).max(1.0)
}

/// Populates all `RSP2` face and cell arrays on `model.aux.rsp` (`set_RSP2_vars`
/// in `hydro_rsp2.f90:67-111`), preserves 24-slot `Eq_ad` / `Uq_ad` partials,
/// and returns the coupled Henyey equation residuals.
pub fn eval_rsp2_profile_equations(
    model: &mut StarModel,
) -> Rsp2ProfileResiduals {
    let work = build_rsp2_ad_work(model);
    let nz = model.nz;
    let l_start_max = model.lum_start.iter().copied().fold(0.0_f64, f64::max);
    let l_scale = if l_start_max > 0.0 {
        l_start_max
    } else {
        model.lum.first().copied().unwrap_or(1.0).abs().max(1.0)
    };
    let k_l_start = usize::from(!model.controls.rsp2.use_l_eqn_at_surface);
    let dt = model.dt.max(1.0e-30);

    let mut equ_l = vec![0.0; nz];
    let mut equ_hp = vec![0.0; nz];
    let mut equ_etrb = vec![0.0; nz];

    for k in k_l_start..nz {
        let l_exp = work.lr[k].val + work.lc[k].val + work.lt[k].val;
        equ_l[k] = (l_exp - model.lum[k]) / l_scale;
    }
    for (k, slot) in equ_hp.iter_mut().enumerate() {
        let hp_exp = eval_rsp2_hp_face_ad(model, k).val;
        let hp_act = model.aux.rsp.hp_face[k].max(1.0);
        *slot = (hp_exp - hp_act) / hp_exp.max(hp_act);
    }
    for (k, slot) in equ_etrb.iter_mut().enumerate() {
        let w = model.aux.rsp.et_rsp[k].max(0.0).sqrt();
        if is_nonturbulent_cell(model, k) {
            let cs_s = model
                .aux
                .rsp
                .csound_start_rsp
                .get(k)
                .copied()
                .filter(|&c| c > 0.0)
                .unwrap_or_else(|| model.csound[k])
                .max(1.0);
            *slot = w / cs_s;
        } else {
            let d_et = w * w - model.aux.rsp.w_start[k].powi(2);
            let ptrb = eval_rsp2_ptrb_cell_ad(model, k, false).val;
            let rho_s = model
                .lnd_start
                .get(k)
                .copied()
                .map_or(model.rho[k], f64::exp)
                .max(1.0e-99);
            let ptrb_dv = ptrb * (1.0 / model.rho[k] - 1.0 / rho_s);
            let logt = model.ln_t[k] / LN10;
            let l_theta = if model.controls.using_velocity_time_centering
                && model.controls.include_l_in_velocity_time_centering
                && logt
                    <= model
                        .controls
                        .max_logt_for_include_p_and_l_in_velocity_time_centering
            {
                model.controls.l_theta_for_velocity_time_centering
            } else {
                1.0
            };
            let lt_00 = work.lt[k].val * l_theta
                + (1.0 - l_theta)
                    * model.aux.rsp.lt_start.get(k).copied().unwrap_or(0.0);
            let lt_p1 = if k + 1 < nz {
                work.lt[k + 1].val * l_theta
                    + (1.0 - l_theta)
                        * model
                            .aux
                            .rsp
                            .lt_start
                            .get(k + 1)
                            .copied()
                            .unwrap_or(0.0)
            } else {
                0.0
            };
            let dt_dlt_dm = (lt_00 - lt_p1) * (dt / model.dm[k].max(1.0e-99));
            let esum = d_et + ptrb_dv + dt_dlt_dm
                - dt * work.coupl[k].val
                - dt * model.aux.eq_ad[k].val;
            let e_start = model
                .energy_start
                .get(k)
                .copied()
                .filter(|&e| e > 0.0)
                .unwrap_or_else(|| model.energy[k])
                .max(1.0);
            let scal_div_dt = (if k == 0 { 1.0e-6 } else { 1.0 }) / e_start;
            *slot = esum * scal_div_dt;
        }
    }

    Rsp2ProfileResiduals {
        equ_l,
        equ_hp,
        equ_etrb,
    }
}

/// Pre-solver turbulent velocity adjustment (`RSP2_adjust_vars_before_call_solver`
/// in `star/private/hydro_rsp2.f90:1195-1253`).
pub fn rsp2_adjust_vars_before_call_solver(model: &mut StarModel) {
    let nz = model.nz;
    let alpha = model.controls.mixing_length_alpha;
    if alpha == 0.0 || nz < 2 {
        return;
    }
    let _ = eval_rsp2_profile_equations(model);
    let rsp2 = model.controls.rsp2.clone();
    // Matches upstream `hydro_rsp2.f90:1209`, which reads `s% RSP_nz_div_IBOTOM`.
    let ibotom = ((nz as f64) / model.controls.rsp.nz_div_ibotom.max(1.0))
        .floor()
        .max(1.0) as usize;
    let k_lo = rsp2.num_outermost_cells_forced_nonturbulent;
    let k_hi = nz.saturating_sub(ibotom);

    for k in k_lo..k_hi {
        let w_cur = model.aux.rsp.et_rsp[k].max(0.0).sqrt();
        if w_cur > rsp2.w_min_for_damping {
            continue;
        }
        let pii_div_hp = 0.5
            * (model.aux.rsp.pii_rsp[k] / model.aux.rsp.hp_face[k].max(1.0)
                + model.aux.rsp.pii_rsp[k + 1]
                    / model.aux.rsp.hp_face[k + 1].max(1.0));
        let qq = model.chi_t[k]
            / (model.rho[k] * model.t[k] * model.chi_rho[k]).max(1.0e-99);
        let source = pii_div_hp * model.t[k] * model.peos[k] * qq
            / model.cp[k].max(1.0e-30);
        let hp_cell = 0.5
            * (model.aux.rsp.hp_face[k].max(1.0)
                + model.aux.rsp.hp_face[k + 1].max(1.0));
        let damp = (rsp2.alfad * X_CEDE / alpha) / hp_cell;
        let pom = 4.0 * BOLTZ_SIGMA * (rsp2.alfar * X_GAMMAR / alpha).powi(2);
        let pom2 = model.t[k].powi(3)
            / (model.rho[k].powi(2)
                * model.cp[k].max(1.0e-30)
                * model.opacity[k].max(1.0e-30));
        let dampr = pom * pom2 / (hp_cell * hp_cell);
        let del = dampr * dampr + 4.0 * damp * source;
        if del >= 0.0 && damp > 0.0 {
            let soln = (-dampr + del.sqrt()) / (2.0 * damp);
            if soln > 0.0 {
                model.aux.rsp.et_rsp[k] = soln * soln;
            }
        }
    }
}
