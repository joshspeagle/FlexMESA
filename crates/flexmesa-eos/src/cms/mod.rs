// SPDX-License-Identifier: LGPL-3.0-or-later
//! Chabrier, Mazevet & Soubiran (2019) hydrogen-helium equation of state (`CMS`).
//!
//! Ports the `CMS` table loader and evaluator from upstream MESA `eoscms_load_tables.f90`
//! and `eoscms_eval.f90` (`Get_CMS_alfa`, `get_CMS_for_eosdt`, `eval_eosCMS_fixed_X`,
//! and `include_radiation`).

pub mod config;
pub mod table;

pub use config::{get_cms_alfa, CmsConfig, CMS_NUM_XS, CMS_XSTR, CMS_XVALS};
pub use table::{CmsTableSet, CmsXTable};

use crate::tables_4d::spline::{do_eos_interpolations, NUM_EOS_TABLE_VALS};
use crate::types::{
    EosError, EosEvalOutput, I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_DE_DRHO,
    I_DS_DRHO, I_DS_DT, I_ETA, I_FRAC_CMS, I_GAMMA1, I_GAMMA3, I_GRAD_AD,
    I_LN_E, I_LN_FREE_E, I_LN_PGAS, I_LN_S, I_MU, NUM_EOS_BASIC_RESULTS,
};
use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{CRAD, LN10, ONE_THIRD};
use flexmesa_interp1d::interp_4pt_pm;

fn locate_cms_axis(
    val_in: f64,
    min_val: f64,
    delta: f64,
    num_pts: usize,
) -> (usize, f64, f64, f64) {
    let idx_raw = ((val_in - min_val) / delta + 0.0001) as isize + 1;
    if idx_raw < 1 {
        let v0 = min_val;
        let v1 = v0 + delta;
        (1, v0, v0, v1)
    } else if (idx_raw as usize) >= num_pts {
        let idx = num_pts - 1;
        let v0 = min_val + ((idx - 1) as f64) * delta;
        let v1 = v0 + delta;
        (idx, v0, v1, v1)
    } else {
        let idx = idx_raw as usize;
        let v0 = min_val + ((idx - 1) as f64) * delta;
        let v1 = v0 + delta;
        (idx, v0, val_in, v1)
    }
}

/// Evaluates a single fixed-$X$ `CMS` table (`eval_eosCMS_fixed_X`).
pub fn eval_cms_fixed_x(
    c: &CmsXTable,
    log_rho: f64,
    log_t: f64,
) -> Result<
    (
        [f64; NUM_EOS_BASIC_RESULTS],
        [f64; NUM_EOS_BASIC_RESULTS],
        [f64; NUM_EOS_BASIC_RESULTS],
    ),
    EosError,
> {
    let (it, log_t0, my_log_t, log_t1) =
        locate_cms_axis(log_t, c.log_t_min, c.delta_log_t, c.num_log_ts);
    let (irho, log_rho0, my_log_rho, log_rho1) = locate_cms_axis(
        log_rho,
        c.log_rho_min,
        c.delta_log_rho,
        c.num_log_rhos,
    );

    let v_ij =
        c.get_vertex(it, irho)
            .ok_or_else(|| EosError::InvalidState {
                context: "eval_cms_fixed_x",
                detail: format!(
                    "missing CMS vertex (it={it}, irho={irho}) for X={}",
                    c.x
                ),
            })?;
    let v_ip1_j =
        c.get_vertex(it + 1, irho)
            .ok_or_else(|| EosError::InvalidState {
                context: "eval_cms_fixed_x",
                detail: format!(
                    "missing CMS vertex (it={}, irho={irho})",
                    it + 1
                ),
            })?;
    let v_i_jp1 =
        c.get_vertex(it, irho + 1)
            .ok_or_else(|| EosError::InvalidState {
                context: "eval_cms_fixed_x",
                detail: format!(
                    "missing CMS vertex (it={it}, irho={})",
                    irho + 1
                ),
            })?;
    let v_ip1_jp1 = c.get_vertex(it + 1, irho + 1).ok_or_else(|| {
        EosError::InvalidState {
            context: "eval_cms_fixed_x",
            detail: format!(
                "missing CMS vertex (it={}, irho={})",
                it + 1,
                irho + 1
            ),
        }
    })?;

    let (fval, df_dx, df_dy) = do_eos_interpolations(
        v_ij, v_ip1_j, v_i_jp1, v_ip1_jp1, log_t0, my_log_t, log_t1, log_rho0,
        my_log_rho, log_rho1,
    );

    let mut res = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut dres_dln_t = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut dres_dln_rho = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let iln10 = 1.0 / LN10;

    for k in 0..NUM_EOS_TABLE_VALS {
        res[k] = fval[k];
        dres_dln_t[k] = df_dx[k] * iln10;
        dres_dln_rho[k] = df_dy[k] * iln10;
    }

    res[I_CP] = fval[I_CP].exp();
    dres_dln_t[I_CP] *= res[I_CP];
    dres_dln_rho[I_CP] *= res[I_CP];

    res[I_CV] = fval[I_CV].exp();
    dres_dln_t[I_CV] *= res[I_CV];
    dres_dln_rho[I_CV] *= res[I_CV];

    Ok((res, dres_dln_t, dres_dln_rho))
}

/// Adds blackbody radiation to `CMS` thermodynamic quantities (`include_radiation`).
///
/// Mirrors `include_radiation` in `eos/private/eoscms_eval.f90` (lines 302..456).
pub fn include_cms_radiation(
    rho_in: f64,
    t_in: f64,
    res: &mut [f64; NUM_EOS_BASIC_RESULTS],
    d_dlnd: &mut [f64; NUM_EOS_BASIC_RESULTS],
    d_dln_t: &mut [f64; NUM_EOS_BASIC_RESULTS],
) {
    let t = AutoDiffReal2VarOrder1::var1(t_in);
    let rho = AutoDiffReal2VarOrder1::var2(rho_in);

    let unpack = |idx: usize| -> AutoDiffReal2VarOrder1 {
        AutoDiffReal2VarOrder1 {
            val: res[idx],
            d1: [d_dln_t[idx] / t_in, d_dlnd[idx] / rho_in],
        }
    };

    let ln_pgas = unpack(I_LN_PGAS);
    let mut ln_e = unpack(I_LN_E);
    let mut ln_s = unpack(I_LN_S);
    let mut chi_rho = unpack(I_CHI_RHO);
    let mut chi_t = unpack(I_CHI_T);
    let mut cv = unpack(I_CV);
    let mu = unpack(I_MU);
    let ln_free_e = unpack(I_LN_FREE_E);
    let eta = unpack(I_ETA);

    let pgas = ln_pgas.exp();
    let t2 = t * t;
    let prad = (ONE_THIRD * CRAD) * (t2 * t2);
    let p = pgas + prad;
    ln_e = (ln_e.exp() + 3.0 * prad / rho).ln();
    ln_s = (ln_s.exp() + 4.0 * prad / (rho * t)).ln();
    cv = cv + 12.0 * prad / (rho * t);
    chi_t = chi_t * pgas / p + 4.0 * prad / p;
    chi_rho = chi_rho * pgas / p;
    let gamma3 = 1.0 + (p / rho) * chi_t / (t * cv);
    let gamma1 = chi_t * (gamma3 - 1.0) + chi_rho;
    let grad_ad = (gamma3 - 1.0) / gamma1;
    let cp = cv * gamma1 / chi_rho;
    let de_drho = (1.0 - chi_t) * p / (rho * rho);
    let ds_dt = cv / t;
    let ds_drho = -p * chi_t / (rho * rho * t);

    let pack = |idx: usize,
                ad: AutoDiffReal2VarOrder1,
                res: &mut [f64; NUM_EOS_BASIC_RESULTS],
                d_dln_t: &mut [f64; NUM_EOS_BASIC_RESULTS],
                d_dlnd: &mut [f64; NUM_EOS_BASIC_RESULTS]| {
        res[idx] = ad.val;
        d_dln_t[idx] = ad.d1[0] * t_in;
        d_dlnd[idx] = ad.d1[1] * rho_in;
    };

    pack(I_LN_PGAS, ln_pgas, res, d_dln_t, d_dlnd);
    pack(I_LN_E, ln_e, res, d_dln_t, d_dlnd);
    pack(I_LN_S, ln_s, res, d_dln_t, d_dlnd);
    pack(I_GRAD_AD, grad_ad, res, d_dln_t, d_dlnd);
    pack(I_CHI_RHO, chi_rho, res, d_dln_t, d_dlnd);
    pack(I_CHI_T, chi_t, res, d_dln_t, d_dlnd);
    pack(I_CP, cp, res, d_dln_t, d_dlnd);
    pack(I_CV, cv, res, d_dln_t, d_dlnd);
    pack(I_DE_DRHO, de_drho, res, d_dln_t, d_dlnd);
    pack(I_DS_DT, ds_dt, res, d_dln_t, d_dlnd);
    pack(I_DS_DRHO, ds_drho, res, d_dln_t, d_dlnd);
    pack(I_MU, mu, res, d_dln_t, d_dlnd);
    pack(I_LN_FREE_E, ln_free_e, res, d_dln_t, d_dlnd);
    pack(I_GAMMA1, gamma1, res, d_dln_t, d_dlnd);
    pack(I_GAMMA3, gamma3, res, d_dln_t, d_dlnd);
    pack(I_ETA, eta, res, d_dln_t, d_dlnd);
}

/// Evaluates `get_CMS_for_eosdt` from precomputed nuclear charges `z_int`.
///
/// Mirrors `get_CMS_for_eosdt` in `eos/private/eoscms_eval.f90` (lines 151..299).
#[allow(clippy::too_many_arguments)]
pub fn eval_cms_eos_from_weights(
    tables: &CmsTableSet,
    cfg: &CmsConfig,
    x: f64,
    rho: f64,
    log_rho: f64,
    t: f64,
    log_t: f64,
    z_int: &[usize],
) -> Result<EosEvalOutput, EosError> {
    let mut res = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut d_dlnd = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut d_dln_t = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut d_dx = [0.0_f64; NUM_EOS_BASIC_RESULTS];

    let mut d_dxa = vec![[0.0_f64; 2]; z_int.len()];

    if cfg.use_fixed_composition {
        if !(0..=10).contains(&cfg.fixed_composition_index) {
            return Err(EosError::InvalidState {
                context: "eval_cms_eos_from_weights",
                detail: format!(
                    "invalid CMS_fixed_composition_index={}",
                    cfg.fixed_composition_index
                ),
            });
        }
        let ix = (cfg.fixed_composition_index as usize) + 1;
        let tbl = tables.get_table(ix)?;
        let (r, dt, dr) = eval_cms_fixed_x(&tbl, log_rho, log_t)?;
        res = r;
        d_dln_t = dt;
        d_dlnd = dr;
    } else {
        let ix = if x <= CMS_XVALS[0] {
            1
        } else if x >= CMS_XVALS[CMS_NUM_XS - 2] {
            CMS_NUM_XS - 1
        } else {
            let mut found = 1;
            for i in 2..CMS_NUM_XS {
                if x < CMS_XVALS[i - 1] {
                    found = i - 1;
                    break;
                }
            }
            found
        };

        if cfg.cubic_in_x && ix > 2 && ix < CMS_NUM_XS - 2 {
            let t1 = tables.get_table(ix - 1)?;
            let t2 = tables.get_table(ix)?;
            let t3 = tables.get_table(ix + 1)?;
            let t4 = tables.get_table(ix + 2)?;
            let (r1, dt1, dr1) = eval_cms_fixed_x(&t1, log_rho, log_t)?;
            let (r2, dt2, dr2) = eval_cms_fixed_x(&t2, log_rho, log_t)?;
            let (r3, dt3, dr3) = eval_cms_fixed_x(&t3, log_rho, log_t)?;
            let (r4, dt4, dr4) = eval_cms_fixed_x(&t4, log_rho, log_t)?;
            let xx = [
                CMS_XVALS[ix - 2],
                CMS_XVALS[ix - 1],
                CMS_XVALS[ix],
                CMS_XVALS[ix + 1],
            ];
            let dx = x - CMS_XVALS[ix - 1];
            for i in 0..NUM_EOS_TABLE_VALS {
                let y = [r1[i], r2[i], r3[i], r4[i]];
                let a = interp_4pt_pm(&xx, &y).unwrap_or([0.0; 3]);
                res[i] = y[1] + dx * (a[0] + dx * (a[1] + dx * a[2]));
                d_dx[i] = a[0] + dx * (2.0 * a[1] + 3.0 * dx * a[2]);

                let yt = [dt1[i], dt2[i], dt3[i], dt4[i]];
                let at = interp_4pt_pm(&xx, &yt).unwrap_or([0.0; 3]);
                d_dln_t[i] = yt[1] + dx * (at[0] + dx * (at[1] + dx * at[2]));

                let yr = [dr1[i], dr2[i], dr3[i], dr4[i]];
                let ar = interp_4pt_pm(&xx, &yr).unwrap_or([0.0; 3]);
                d_dlnd[i] = yr[1] + dx * (ar[0] + dx * (ar[1] + dx * ar[2]));
            }
        } else {
            let t1 = tables.get_table(ix)?;
            let t2 = tables.get_table(ix + 1)?;
            let (r1, dt1, dr1) = eval_cms_fixed_x(&t1, log_rho, log_t)?;
            let (r2, dt2, dr2) = eval_cms_fixed_x(&t2, log_rho, log_t)?;
            let beta =
                (x - CMS_XVALS[ix - 1]) / (CMS_XVALS[ix] - CMS_XVALS[ix - 1]);
            let alfa = 1.0 - beta;
            let dbeta_dx = 1.0 / (CMS_XVALS[ix] - CMS_XVALS[ix - 1]);
            let dalfa_dx = -dbeta_dx;
            for i in 0..NUM_EOS_TABLE_VALS {
                res[i] = alfa * r1[i] + beta * r2[i];
                d_dx[i] = dalfa_dx * r1[i] + dbeta_dx * r2[i];
                d_dln_t[i] = alfa * dt1[i] + beta * dt2[i];
                d_dlnd[i] = alfa * dr1[i] + beta * dr2[i];
            }
        }

        for (i, &zi) in z_int.iter().enumerate() {
            if zi == 1 {
                d_dxa[i] = [d_dx[I_LN_PGAS], d_dx[I_LN_E]];
            }
        }
    }

    if cfg.include_radiation {
        include_cms_radiation(rho, t, &mut res, &mut d_dlnd, &mut d_dln_t);
    }

    let mut out = EosEvalOutput {
        res,
        d_dlnd,
        d_dln_t,
        d_dxa,
    };
    out.set_single_component_fraction(I_FRAC_CMS);
    Ok(out)
}

/// Evaluates the standalone `CMS` equation of state for `(rho, T, chem_id, xa)`.
pub fn eval_cms_eos(
    tables: &CmsTableSet,
    cfg: &CmsConfig,
    rho: f64,
    t: f64,
    chem_id: &[usize],
    xa: &[f64],
) -> Result<EosEvalOutput, EosError> {
    let catalog = NuclideCatalog::builtin_ref();
    let comp =
        get_composition_info(catalog, chem_id, xa, true).map_err(|e| {
            EosError::InvalidState {
                context: "eval_cms_eos",
                detail: e.to_string(),
            }
        })?;
    let mut z_int = Vec::with_capacity(chem_id.len());
    for &cid in chem_id {
        let nuc =
            catalog
                .get_by_id(cid)
                .ok_or_else(|| EosError::InvalidState {
                    context: "eval_cms_eos",
                    detail: format!("unknown chem_id {cid}"),
                })?;
        z_int.push(nuc.z as usize);
    }
    eval_cms_eos_from_weights(
        tables,
        cfg,
        comp.xh,
        rho,
        rho.log10(),
        t,
        t.log10(),
        &z_int,
    )
}
