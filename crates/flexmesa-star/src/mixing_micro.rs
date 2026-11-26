// SPDX-License-Identifier: GPL-3.0-or-later
//! Slice-scoped microphysics helpers (`eval_uniform_xa_eos_kap_range`,
//! `eval_eos_kap_range_by_mode`) and zero-drift range snapshots used by
//! verbatim MESA `26.4.1` predictive mixing (`predictive_mix.f90`) and
//! convective premixing (`conv_premix.f90`).

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::LN10;
use flexmesa_eos::{
    eos_dt_get_rho, eos_dt_get_zones_with_logs, EosEvalOutput, I_CHI_RHO,
    I_CHI_T, I_CP, I_CV, I_GAMMA1, I_GRAD_AD, I_LN_FREE_E, I_LN_PGAS,
    NUM_EOS_BASIC_RESULTS,
};
use rayon::prelude::*;

use crate::controls::StarError;
use crate::micro::{
    builtin_helm_table, eval_kap_for_range, solve_fix_pgas_cell,
    store_eos_for_cell,
};
use crate::model::StarModel;
use crate::star_ad::StarAdOrder1;

/// Maximum root-finder iterations for inverse EOS solves
/// (`max_iter_for_solve` in `star/private/eos_support.f90:38`).
const MAX_ITER_FOR_SOLVE: usize = 100;

/// Tolerance on $\log_{10}\rho$ for `solve_eos_given_PgasT`
/// (`star/private/predictive_mix.f90:889`, `star/private/micro.f90:330`).
const LOG_RHO_TOL_PGAS_T: f64 = 1.0e-8;

/// Evaluates uniform-composition trial EOS (`solve_eos_given_PgasT`) and
/// opacity (`do_kap_for_cell`) in parallel across `k_lo..=k_hi` using `rayon`,
/// then writes outputs back in deterministic index order
/// (`eval_mixing_coeffs` / `eval_eos` in
/// `star/private/predictive_mix.f90:720-780, 862-927`).
///
/// Reproduces MESA `26.4.1` `#169 / #277` verbatim: only the 12 cell fields
/// updated by upstream `predictive_mix::eval_eos` (`lnd`, `cp`, `cv`, `gamma1`,
/// `grada`, `chi_rho`, `chi_t`, `lnfree_e`, `eos_d_dlnd`, `eos_d_dlnt`, `xa`,
/// `zbar`) are overwritten before `do_kap_for_cell` and `do1_mlt_2`, leaving
/// `rho`, `eta`, `ln_pgas`/`pgas`/`peos`, `mu`, and `t` at their pre-trial
/// values.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `use_other_kap` is active without a
/// hook, or [`StarError::MicrophysicsError`] if composition, EOS, or opacity
/// evaluation fails for any cell in `k_lo..=k_hi`.
pub(crate) fn eval_uniform_xa_eos_kap_range(
    model: &mut StarModel,
    k_lo: usize,
    k_hi: usize,
    xa_mx: &[f64],
) -> Result<(), StarError> {
    if k_lo > k_hi {
        return Ok(());
    }
    let catalog = NuclideCatalog::builtin_ref();
    let comp_raw = get_composition_info(catalog, &model.chem_id, xa_mx, true)
        .map_err(|e| {
        StarError::MicrophysicsError(format!(
            "composition error in predictive_mix: {e}"
        ))
    })?;
    let zbar = comp_raw.zbar;
    let ht = builtin_helm_table()?;
    let species = model.species;

    let chem_id = &model.chem_id;
    let eos_cfg = &model.eos_config;
    let ln_t = &model.ln_t;
    let lnd = &model.lnd;
    let ln_pgas = &model.ln_pgas;

    let eos_results: Vec<(f64, EosEvalOutput)> = (k_lo..=k_hi)
        .into_par_iter()
        .map(|k| -> Result<(f64, EosEvalOutput), StarError> {
            let inv = eos_dt_get_rho(
                ht,
                chem_id,
                xa_mx,
                ln_t[k] / LN10,
                I_LN_PGAS as isize,
                // `solve_eos_given_PgasT` receives `lnPgas/ln10` and
                // multiplies by `ln10` again
                // (`star/private/predictive_mix.f90:901-905`,
                // `star/private/eos_support.f90:343`).
                (ln_pgas[k] / LN10) * LN10,
                lnd[k] / LN10,
                LOG_RHO_TOL_PGAS_T,
                LOG_RHO_TOL_PGAS_T * LN10,
                MAX_ITER_FOR_SOLVE,
                eos_cfg,
            )
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "solve_eos_given_PgasT failed at zone {k}: {e}"
                ))
            })?;
            if inv.log10_result < -25.0 {
                return Err(StarError::MicrophysicsError(format!(
                    "predictive_mix::eval_eos failed at zone {k}: logRho ({}) < -25",
                    inv.log10_result
                )));
            }
            Ok((inv.log10_result * LN10, inv.eos))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Verbatim `star/private/predictive_mix.f90:748-760, 912-924` (`#169 / #277`):
    // update only the 12 upstream cell fields (`lnd`, `Cp`, `Cv`, `gamma1`,
    // `grada`, `chiRho`, `chiT`, `lnfree_e`, `d_eos_dlnd`, `d_eos_dlnT`, `xa`,
    // `zbar`), keeping `rho`, `eta`, `lnPgas`/`Pgas`/`Peos`, `mu`, and `T` fixed.
    for (offset, (lnd_new, out)) in eos_results.into_iter().enumerate() {
        let k = k_lo + offset;
        model.lnd[k] = lnd_new;
        model.cp[k] = out.res[I_CP];
        model.cv[k] = out.res[I_CV];
        model.gamma1[k] = out.res[I_GAMMA1];
        model.grada[k] = out.res[I_GRAD_AD];
        model.chi_rho[k] = out.res[I_CHI_RHO];
        model.chi_t[k] = out.res[I_CHI_T];
        model.lnfree_e[k] = out.res[I_LN_FREE_E];
        model.eos_d_dlnd[k] = out.d_dlnd;
        model.eos_d_dlnt[k] = out.d_dln_t;
        model.xa[k * species..(k + 1) * species].copy_from_slice(xa_mx);
        model.zbar[k] = zbar;
    }

    eval_kap_for_range(model, k_lo, k_hi)
}

/// Evaluates per-cell composition moments, EOS (`fix_pgas_by_cell[k]` selects
/// fixed `(Pgas, T)` vs fixed `(rho, T)`), and opacity across `k_lo..=k_hi` in
/// parallel via `rayon`, writing outputs back in deterministic index order
/// (`update_model_` in `star/private/conv_premix.f90:1306-1335`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `use_other_kap` is active without a
/// hook, or [`StarError::MicrophysicsError`] if composition, EOS, or opacity
/// evaluation fails.
pub(crate) fn eval_eos_kap_range_by_mode(
    model: &mut StarModel,
    k_lo: usize,
    k_hi: usize,
    fix_pgas_by_cell: &[bool],
) -> Result<(), StarError> {
    if k_lo > k_hi {
        return Ok(());
    }
    let catalog = NuclideCatalog::builtin_ref();
    let ht = builtin_helm_table()?;
    let species = model.species;

    // Refresh cell composition moments (`basic_composition_info` in
    // `star/private/micro.f90:321-324` + `get_XYZ` clamping in
    // `star/private/star_utils.f90:2029-2043`) across `k_lo..=k_hi`.
    let comp_results: Vec<(f64, f64, f64, f64, f64, f64, f64, f64, f64)> =
        (k_lo..=k_hi)
            .into_par_iter()
            .map(|k| {
                let xa_k = &model.xa[k * species..(k + 1) * species];
                let c =
                    get_composition_info(catalog, &model.chem_id, xa_k, true)
                        .map_err(|e| {
                        StarError::MicrophysicsError(format!(
                            "composition error at zone {k}: {e}"
                        ))
                    })?;
                let xh = c.xh.clamp(0.0, 1.0);
                let xhe = c.xhe.clamp(0.0, 1.0);
                let z = (1.0 - (xh + xhe)).clamp(0.0, 1.0);
                Ok((
                    c.abar,
                    c.zbar,
                    c.z2bar,
                    c.z53bar,
                    c.ye,
                    c.mass_correction,
                    xh,
                    xhe,
                    z,
                ))
            })
            .collect::<Result<Vec<_>, StarError>>()?;
    for (offset, (abar, zbar, z2bar, z53bar, ye, mc, xh, xhe, z)) in
        comp_results.into_iter().enumerate()
    {
        let k = k_lo + offset;
        model.abar[k] = abar;
        model.zbar[k] = zbar;
        model.z2bar[k] = z2bar;
        model.z53bar[k] = z53bar;
        model.ye[k] = ye;
        model.aux.mass_correction[k] = mc;
        model.xh_mass[k] = xh;
        model.yhe_mass[k] = xhe;
        model.z_mass[k] = z;
    }

    // Partition range into Pass 1 (`fix_Pgas == true`) and Pass 2
    // (`fix_Pgas == false`), matching `conv_premix.f90:1311-1325`.
    let mut pt_cells = Vec::new();
    let mut dt_cells = Vec::new();
    for k in k_lo..=k_hi {
        if fix_pgas_by_cell[k] {
            pt_cells.push(k);
        } else {
            dt_cells.push(k);
        }
    }

    if !pt_cells.is_empty() {
        let chem_id = &model.chem_id;
        let eos_cfg = &model.eos_config;
        let xa = &model.xa;
        let ln_t = &model.ln_t;
        let lnd = &model.lnd;
        let ln_pgas = &model.ln_pgas;
        let t = &model.t;
        let pt_results: Vec<(usize, f64, f64, EosEvalOutput)> = pt_cells
            .par_iter()
            .map(
                |&k| -> Result<(usize, f64, f64, EosEvalOutput), StarError> {
                    let xa_k = &xa[k * species..(k + 1) * species];
                    let (lnd_out, rho_out, eos) = solve_fix_pgas_cell(
                        ht, chem_id, xa_k, k, ln_t[k], t[k], ln_pgas[k],
                        lnd[k], eos_cfg,
                    )?;
                    Ok((k, lnd_out, rho_out, eos))
                },
            )
            .collect::<Result<Vec<_>, _>>()?;
        for (k, lnd_new, rho_new, out) in pt_results {
            model.lnd[k] = lnd_new;
            model.rho[k] = rho_new;
            store_eos_for_cell(model, k, &out)?;
        }
    }

    if !dt_cells.is_empty() {
        let n_dt = dt_cells.len();
        let mut rho_sub = Vec::with_capacity(n_dt);
        let mut log_rho_sub = Vec::with_capacity(n_dt);
        let mut t_sub = Vec::with_capacity(n_dt);
        let mut log_t_sub = Vec::with_capacity(n_dt);
        let mut xa_sub = Vec::with_capacity(n_dt * species);
        for &k in &dt_cells {
            let log_rho = model.lnd[k] / LN10;
            if log_rho < -25.0 {
                return Err(StarError::MicrophysicsError(format!(
                    "do_eos_for_cell failed at zone {k}: logRho ({log_rho}) < -25"
                )));
            }
            rho_sub.push(model.rho[k]);
            log_rho_sub.push(log_rho);
            t_sub.push(model.t[k]);
            log_t_sub.push(model.ln_t[k] / LN10);
            xa_sub.extend_from_slice(&model.xa[k * species..(k + 1) * species]);
        }
        let dt_results = eos_dt_get_zones_with_logs(
            ht,
            &rho_sub,
            &log_rho_sub,
            &t_sub,
            &log_t_sub,
            &model.chem_id,
            &xa_sub,
            &model.eos_config,
        )
        .map_err(|e| {
            StarError::MicrophysicsError(format!("do_eos_for_cell failed: {e}"))
        })?;
        for (&k, out) in dt_cells.iter().zip(dt_results.iter()) {
            store_eos_for_cell(model, k, out)?;
        }
    }

    eval_kap_for_range(model, k_lo, k_hi)
}

/// Snapshot of cell microphysics + composition + density over `k_lo..=k_hi`
/// for fast, allocation-light save/restore in `predictive_mix` and
/// `conv_premix`.
#[derive(Debug, Clone)]
pub(crate) struct CellRangeSnapshot {
    pub(crate) k_lo: usize,
    pub(crate) k_hi: usize,
    pub(crate) xa: Vec<f64>,
    pub(crate) lnd: Vec<f64>,
    pub(crate) rho: Vec<f64>,
    pub(crate) abar: Vec<f64>,
    pub(crate) zbar: Vec<f64>,
    pub(crate) z2bar: Vec<f64>,
    pub(crate) z53bar: Vec<f64>,
    pub(crate) ye: Vec<f64>,
    pub(crate) mass_correction: Vec<f64>,
    pub(crate) xh_mass: Vec<f64>,
    pub(crate) yhe_mass: Vec<f64>,
    pub(crate) z_mass: Vec<f64>,
    pub(crate) pgas: Vec<f64>,
    pub(crate) ln_pgas: Vec<f64>,
    pub(crate) prad: Vec<f64>,
    pub(crate) peos: Vec<f64>,
    pub(crate) ln_peos: Vec<f64>,
    pub(crate) energy: Vec<f64>,
    pub(crate) ln_e: Vec<f64>,
    pub(crate) entropy: Vec<f64>,
    pub(crate) ln_s: Vec<f64>,
    pub(crate) grada: Vec<f64>,
    pub(crate) cv: Vec<f64>,
    pub(crate) cp: Vec<f64>,
    pub(crate) chi_rho: Vec<f64>,
    pub(crate) chi_t: Vec<f64>,
    pub(crate) gamma1: Vec<f64>,
    pub(crate) gamma3: Vec<f64>,
    pub(crate) eta: Vec<f64>,
    pub(crate) gam: Vec<f64>,
    pub(crate) mu: Vec<f64>,
    pub(crate) lnfree_e: Vec<f64>,
    pub(crate) csound: Vec<f64>,
    pub(crate) eos_frac_pc: Vec<f64>,
    pub(crate) kap_frac: Vec<[f64; 4]>,
    pub(crate) latent_ddlnt: Vec<f64>,
    pub(crate) latent_ddlnrho: Vec<f64>,
    pub(crate) dln_e_dxa: Vec<f64>,
    pub(crate) dln_peos_dxa: Vec<f64>,
    pub(crate) ln_peos_ad: Vec<StarAdOrder1>,
    pub(crate) ln_e_ad: Vec<StarAdOrder1>,
    pub(crate) ln_s_ad: Vec<StarAdOrder1>,
    pub(crate) eos_res: Vec<[f64; NUM_EOS_BASIC_RESULTS]>,
    pub(crate) eos_d_dlnd: Vec<[f64; NUM_EOS_BASIC_RESULTS]>,
    pub(crate) eos_d_dlnt: Vec<[f64; NUM_EOS_BASIC_RESULTS]>,
    pub(crate) opacity: Vec<f64>,
    pub(crate) d_opacity_dlnd: Vec<f64>,
    pub(crate) d_opacity_dlnt: Vec<f64>,
}

impl CellRangeSnapshot {
    /// Captures all composition, density, EOS, and opacity state across
    /// `k_lo..=k_hi`.
    #[must_use]
    pub(crate) fn capture(model: &StarModel, k_lo: usize, k_hi: usize) -> Self {
        let sp = model.species;
        let sp_lo = k_lo * sp;
        let sp_hi = (k_hi + 1) * sp;
        let r = k_lo..=k_hi;
        let kap_frac = if model.aux.kap_frac.len() > k_hi {
            model.aux.kap_frac[r.clone()].to_vec()
        } else {
            vec![[0.0; 4]; k_hi - k_lo + 1]
        };
        Self {
            k_lo,
            k_hi,
            xa: model.xa[sp_lo..sp_hi].to_vec(),
            lnd: model.lnd[r.clone()].to_vec(),
            rho: model.rho[r.clone()].to_vec(),
            abar: model.abar[r.clone()].to_vec(),
            zbar: model.zbar[r.clone()].to_vec(),
            z2bar: model.z2bar[r.clone()].to_vec(),
            z53bar: model.z53bar[r.clone()].to_vec(),
            ye: model.ye[r.clone()].to_vec(),
            mass_correction: model.aux.mass_correction[r.clone()].to_vec(),
            xh_mass: model.xh_mass[r.clone()].to_vec(),
            yhe_mass: model.yhe_mass[r.clone()].to_vec(),
            z_mass: model.z_mass[r.clone()].to_vec(),
            pgas: model.pgas[r.clone()].to_vec(),
            ln_pgas: model.ln_pgas[r.clone()].to_vec(),
            prad: model.prad[r.clone()].to_vec(),
            peos: model.peos[r.clone()].to_vec(),
            ln_peos: model.ln_peos[r.clone()].to_vec(),
            energy: model.energy[r.clone()].to_vec(),
            ln_e: model.ln_e[r.clone()].to_vec(),
            entropy: model.entropy[r.clone()].to_vec(),
            ln_s: model.ln_s[r.clone()].to_vec(),
            grada: model.grada[r.clone()].to_vec(),
            cv: model.cv[r.clone()].to_vec(),
            cp: model.cp[r.clone()].to_vec(),
            chi_rho: model.chi_rho[r.clone()].to_vec(),
            chi_t: model.chi_t[r.clone()].to_vec(),
            gamma1: model.gamma1[r.clone()].to_vec(),
            gamma3: model.gamma3[r.clone()].to_vec(),
            eta: model.eta[r.clone()].to_vec(),
            gam: model.gam[r.clone()].to_vec(),
            mu: model.mu[r.clone()].to_vec(),
            lnfree_e: model.lnfree_e[r.clone()].to_vec(),
            csound: model.csound[r.clone()].to_vec(),
            eos_frac_pc: model.aux.eos_frac_pc[r.clone()].to_vec(),
            kap_frac,
            latent_ddlnt: model.aux.latent_ddlnt[r.clone()].to_vec(),
            latent_ddlnrho: model.aux.latent_ddlnrho[r.clone()].to_vec(),
            dln_e_dxa: model.aux.dln_e_dxa_for_partials[sp_lo..sp_hi].to_vec(),
            dln_peos_dxa: model.aux.dln_peos_dxa_for_partials[sp_lo..sp_hi]
                .to_vec(),
            ln_peos_ad: model.ln_peos_ad[r.clone()].to_vec(),
            ln_e_ad: model.ln_e_ad[r.clone()].to_vec(),
            ln_s_ad: model.ln_s_ad[r.clone()].to_vec(),
            eos_res: model.eos_res[r.clone()].to_vec(),
            eos_d_dlnd: model.eos_d_dlnd[r.clone()].to_vec(),
            eos_d_dlnt: model.eos_d_dlnt[r.clone()].to_vec(),
            opacity: model.opacity[r.clone()].to_vec(),
            d_opacity_dlnd: model.d_opacity_dlnd[r.clone()].to_vec(),
            d_opacity_dlnt: model.d_opacity_dlnt[r.clone()].to_vec(),
        }
    }

    /// Restores the captured range back into `model`.
    pub(crate) fn restore(&self, model: &mut StarModel) {
        let sp = model.species;
        let sp_lo = self.k_lo * sp;
        let sp_hi = (self.k_hi + 1) * sp;
        let r = self.k_lo..=self.k_hi;
        model.xa[sp_lo..sp_hi].copy_from_slice(&self.xa);
        model.lnd[r.clone()].copy_from_slice(&self.lnd);
        model.rho[r.clone()].copy_from_slice(&self.rho);
        model.abar[r.clone()].copy_from_slice(&self.abar);
        model.zbar[r.clone()].copy_from_slice(&self.zbar);
        model.z2bar[r.clone()].copy_from_slice(&self.z2bar);
        model.z53bar[r.clone()].copy_from_slice(&self.z53bar);
        model.ye[r.clone()].copy_from_slice(&self.ye);
        model.aux.mass_correction[r.clone()]
            .copy_from_slice(&self.mass_correction);
        model.xh_mass[r.clone()].copy_from_slice(&self.xh_mass);
        model.yhe_mass[r.clone()].copy_from_slice(&self.yhe_mass);
        model.z_mass[r.clone()].copy_from_slice(&self.z_mass);
        model.pgas[r.clone()].copy_from_slice(&self.pgas);
        model.ln_pgas[r.clone()].copy_from_slice(&self.ln_pgas);
        model.prad[r.clone()].copy_from_slice(&self.prad);
        model.peos[r.clone()].copy_from_slice(&self.peos);
        model.ln_peos[r.clone()].copy_from_slice(&self.ln_peos);
        model.energy[r.clone()].copy_from_slice(&self.energy);
        model.ln_e[r.clone()].copy_from_slice(&self.ln_e);
        model.entropy[r.clone()].copy_from_slice(&self.entropy);
        model.ln_s[r.clone()].copy_from_slice(&self.ln_s);
        model.grada[r.clone()].copy_from_slice(&self.grada);
        model.cv[r.clone()].copy_from_slice(&self.cv);
        model.cp[r.clone()].copy_from_slice(&self.cp);
        model.chi_rho[r.clone()].copy_from_slice(&self.chi_rho);
        model.chi_t[r.clone()].copy_from_slice(&self.chi_t);
        model.gamma1[r.clone()].copy_from_slice(&self.gamma1);
        model.gamma3[r.clone()].copy_from_slice(&self.gamma3);
        model.eta[r.clone()].copy_from_slice(&self.eta);
        model.gam[r.clone()].copy_from_slice(&self.gam);
        model.mu[r.clone()].copy_from_slice(&self.mu);
        model.lnfree_e[r.clone()].copy_from_slice(&self.lnfree_e);
        model.csound[r.clone()].copy_from_slice(&self.csound);
        model.aux.eos_frac_pc[r.clone()].copy_from_slice(&self.eos_frac_pc);
        if model.aux.kap_frac.len() != model.nz {
            model.aux.kap_frac.resize(model.nz, [0.0; 4]);
        }
        model.aux.kap_frac[r.clone()].copy_from_slice(&self.kap_frac);
        model.aux.latent_ddlnt[r.clone()].copy_from_slice(&self.latent_ddlnt);
        model.aux.latent_ddlnrho[r.clone()]
            .copy_from_slice(&self.latent_ddlnrho);
        model.aux.dln_e_dxa_for_partials[sp_lo..sp_hi]
            .copy_from_slice(&self.dln_e_dxa);
        model.aux.dln_peos_dxa_for_partials[sp_lo..sp_hi]
            .copy_from_slice(&self.dln_peos_dxa);
        model.ln_peos_ad[r.clone()].copy_from_slice(&self.ln_peos_ad);
        model.ln_e_ad[r.clone()].copy_from_slice(&self.ln_e_ad);
        model.ln_s_ad[r.clone()].copy_from_slice(&self.ln_s_ad);
        model.eos_res[r.clone()].copy_from_slice(&self.eos_res);
        model.eos_d_dlnd[r.clone()].copy_from_slice(&self.eos_d_dlnd);
        model.eos_d_dlnt[r.clone()].copy_from_slice(&self.eos_d_dlnt);
        model.opacity[r.clone()].copy_from_slice(&self.opacity);
        model.d_opacity_dlnd[r.clone()].copy_from_slice(&self.d_opacity_dlnd);
        model.d_opacity_dlnt[r].copy_from_slice(&self.d_opacity_dlnt);
    }
}

/// Snapshot of face-centered MLT variables over `k_lo..=k_hi` so trial
/// `do1_mlt_2` calls in `predictive_mix` and `conv_premix` can be rolled back
/// to 0 ULP without repeating opacity/MLT evaluations.
#[derive(Debug, Clone)]
pub(crate) struct FaceMltSnapshot {
    pub(crate) k_lo: usize,
    pub(crate) k_hi: usize,
    pub(crate) gradt: Vec<f64>,
    pub(crate) gradt_ad: Vec<StarAdOrder1>,
    pub(crate) gradr: Vec<f64>,
    pub(crate) gradr_ad: Vec<StarAdOrder1>,
    pub(crate) gradl: Vec<f64>,
    pub(crate) gradl_ad: Vec<StarAdOrder1>,
    pub(crate) scale_height: Vec<f64>,
    pub(crate) scale_height_ad: Vec<StarAdOrder1>,
    pub(crate) mlt_vc: Vec<f64>,
    pub(crate) mlt_vc_ad: Vec<StarAdOrder1>,
    pub(crate) mlt_d: Vec<f64>,
    pub(crate) mlt_d_ad: Vec<StarAdOrder1>,
    pub(crate) mlt_cdc: Vec<f64>,
    pub(crate) mlt_cdc_ad: Vec<StarAdOrder1>,
    pub(crate) mlt_gamma: Vec<f64>,
    pub(crate) mlt_mixing_type: Vec<flexmesa_const::MixingType>,
    pub(crate) mlt_mixing_length: Vec<f64>,
    pub(crate) grada_face: Vec<f64>,
    pub(crate) grada_face_ad: Vec<StarAdOrder1>,
    pub(crate) gradt_sub_grada: Vec<f64>,
    pub(crate) mlt_gradt: Vec<f64>,
    pub(crate) y_face: Vec<f64>,
    pub(crate) l_conv: Vec<f64>,
    pub(crate) superad_reduction_factor: Vec<f64>,
    pub(crate) gradt_excess_effect: Vec<f64>,
}

impl FaceMltSnapshot {
    /// Captures all face MLT outputs over `k_lo..=k_hi`.
    #[must_use]
    pub(crate) fn capture(model: &StarModel, k_lo: usize, k_hi: usize) -> Self {
        let r = k_lo..=k_hi;
        let m = &model.aux.mixing;
        Self {
            k_lo,
            k_hi,
            gradt: model.gradt[r.clone()].to_vec(),
            gradt_ad: model.gradt_ad[r.clone()].to_vec(),
            gradr: model.gradr[r.clone()].to_vec(),
            gradr_ad: model.gradr_ad[r.clone()].to_vec(),
            gradl: model.gradl[r.clone()].to_vec(),
            gradl_ad: model.gradl_ad[r.clone()].to_vec(),
            scale_height: model.scale_height[r.clone()].to_vec(),
            scale_height_ad: model.scale_height_ad[r.clone()].to_vec(),
            mlt_vc: model.mlt_vc[r.clone()].to_vec(),
            mlt_vc_ad: model.mlt_vc_ad[r.clone()].to_vec(),
            mlt_d: model.mlt_d[r.clone()].to_vec(),
            mlt_d_ad: model.mlt_d_ad[r.clone()].to_vec(),
            mlt_cdc: model.mlt_cdc[r.clone()].to_vec(),
            mlt_cdc_ad: model.mlt_cdc_ad[r.clone()].to_vec(),
            mlt_gamma: model.mlt_gamma[r.clone()].to_vec(),
            mlt_mixing_type: m.mlt_mixing_type[r.clone()].to_vec(),
            mlt_mixing_length: m.mlt_mixing_length[r.clone()].to_vec(),
            grada_face: m.grada_face[r.clone()].to_vec(),
            grada_face_ad: m.grada_face_ad[r.clone()].to_vec(),
            gradt_sub_grada: m.gradt_sub_grada[r.clone()].to_vec(),
            mlt_gradt: m.mlt_gradt[r.clone()].to_vec(),
            y_face: m.y_face[r.clone()].to_vec(),
            l_conv: m.l_conv[r.clone()].to_vec(),
            superad_reduction_factor: m.superad_reduction_factor[r.clone()]
                .to_vec(),
            gradt_excess_effect: m.gradt_excess_effect[r].to_vec(),
        }
    }

    /// Restores the captured face MLT outputs over `k_lo..=k_hi`.
    pub(crate) fn restore(&self, model: &mut StarModel) {
        let r = self.k_lo..=self.k_hi;
        model.gradt[r.clone()].copy_from_slice(&self.gradt);
        model.gradt_ad[r.clone()].copy_from_slice(&self.gradt_ad);
        model.gradr[r.clone()].copy_from_slice(&self.gradr);
        model.gradr_ad[r.clone()].copy_from_slice(&self.gradr_ad);
        model.gradl[r.clone()].copy_from_slice(&self.gradl);
        model.gradl_ad[r.clone()].copy_from_slice(&self.gradl_ad);
        model.scale_height[r.clone()].copy_from_slice(&self.scale_height);
        model.scale_height_ad[r.clone()].copy_from_slice(&self.scale_height_ad);
        model.mlt_vc[r.clone()].copy_from_slice(&self.mlt_vc);
        model.mlt_vc_ad[r.clone()].copy_from_slice(&self.mlt_vc_ad);
        model.mlt_d[r.clone()].copy_from_slice(&self.mlt_d);
        model.mlt_d_ad[r.clone()].copy_from_slice(&self.mlt_d_ad);
        model.mlt_cdc[r.clone()].copy_from_slice(&self.mlt_cdc);
        model.mlt_cdc_ad[r.clone()].copy_from_slice(&self.mlt_cdc_ad);
        model.mlt_gamma[r.clone()].copy_from_slice(&self.mlt_gamma);
        let m = &mut model.aux.mixing;
        m.mlt_mixing_type[r.clone()].copy_from_slice(&self.mlt_mixing_type);
        m.mlt_mixing_length[r.clone()].copy_from_slice(&self.mlt_mixing_length);
        m.grada_face[r.clone()].copy_from_slice(&self.grada_face);
        m.grada_face_ad[r.clone()].copy_from_slice(&self.grada_face_ad);
        m.gradt_sub_grada[r.clone()].copy_from_slice(&self.gradt_sub_grada);
        m.mlt_gradt[r.clone()].copy_from_slice(&self.mlt_gradt);
        m.y_face[r.clone()].copy_from_slice(&self.y_face);
        m.l_conv[r.clone()].copy_from_slice(&self.l_conv);
        m.superad_reduction_factor[r.clone()]
            .copy_from_slice(&self.superad_reduction_factor);
        m.gradt_excess_effect[r].copy_from_slice(&self.gradt_excess_effect);
    }
}
