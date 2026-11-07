// SPDX-License-Identifier: LGPL-3.0-or-later
//! Hierarchical `eosDT_get` equation of state evaluator and blending cascade.
//!
//! Ports the hierarchical evaluation tree from upstream MESA `eosdt_eval.f90`
//! (`Get_eosDT_Results`, `get_level2_for_eosdt`, `get_level3_for_eosdt`,
//! `get_level6_for_eosdt`, `get_HELM_alfa`, and `combine_for_eosdt`).
//!
//! At each level of the hierarchy, a raw blend parameter $\alpha \in [0, 1]$ is
//! computed along with its analytical partials $(\partial \alpha / \partial \log_{10} T,
//! \partial \alpha / \partial \log_{10} \rho)$, where $\alpha = 0$ corresponds to
//! 100% of that level's primary EOS component (`1st`) and $\alpha = 1$ corresponds
//! to 100% of the fallback cascade (`2nd`). When $0 < \alpha < 1$, `combine_for_eosdt`
//! smooths $\alpha$ via the $C^2$ quintic kernel (`quintic_blend`) and combines the
//! two components via `blend_eos_results`.

use crate::blend::{blend_eos_results, is_contained, min_distance_to_polygon};
use crate::helm::{
    do_convert_helm_results, helmeos2, HelmConfig, HelmTable, HELM_LOGDHI,
    HELM_LOGDLO, HELM_LOGTHI, HELM_LOGTLO,
};
use crate::ideal::eval_ideal_eos_from_weights;
use crate::pc::{eval_pc_eos_from_weights, get_pc_alfa, PcConfig};
use crate::skye::{
    eval_skye_eos_from_weights, get_skye_alfa, get_skye_alfa_simple, SkyeConfig,
};
use crate::types::{EosError, EosEvalOutput};
use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_chem::{get_composition_info, NuclideCatalog};
use rayon::prelude::*;

/// Unified configuration for the hierarchical `eosDT_get` evaluator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EosConfig {
    /// Configuration for the Helmholtz (`HELM`) EOS component.
    pub helm: HelmConfig,
    /// Configuration for the Potekhin & Chabrier (`PC`) EOS component.
    pub pc: PcConfig,
    /// Configuration for the `Skye` EOS component.
    pub skye: SkyeConfig,
    /// Enable the `Skye` EOS component at level 2 (default `true`).
    pub use_skye: bool,
    /// Enable the `PC` EOS component at level 3 (default `false`).
    pub use_pc: bool,
    /// Width of the boundary blend zone around the `HELM` table polygon in dex (default `0.1`).
    pub helm_blend_width: f64,
    /// Use linear instead of quintic blending between EOS components (default `false`).
    pub linear_blend: bool,
}

impl Default for EosConfig {
    fn default() -> Self {
        Self {
            helm: HelmConfig::default(),
            pc: PcConfig::default(),
            skye: SkyeConfig::default(),
            use_skye: true,
            use_pc: false,
            helm_blend_width: 0.1,
            linear_blend: false,
        }
    }
}

/// Extracts `(z_int, azion, acmi, a_baryon)` arrays from `chem_id` and `NuclideCatalog`.
pub(crate) fn extract_species_weights(
    catalog: &NuclideCatalog,
    chem_id: &[usize],
    context: &'static str,
) -> Result<(Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>), EosError> {
    let n = chem_id.len();
    let mut z_int = Vec::with_capacity(n);
    let mut azion = Vec::with_capacity(n);
    let mut acmi = Vec::with_capacity(n);
    let mut a_baryon = Vec::with_capacity(n);
    for &cid in chem_id {
        let nuc =
            catalog
                .get_by_id(cid)
                .ok_or_else(|| EosError::InvalidState {
                    context,
                    detail: format!("unknown chem_id {cid}"),
                })?;
        z_int.push(nuc.z as usize);
        azion.push(nuc.z as f64);
        acmi.push(nuc.w);
        a_baryon.push(nuc.z_plus_n as f64);
    }
    Ok((z_int, azion, acmi, a_baryon))
}

/// Computes the raw `HELM` polygon blend fraction `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
///
/// Mirrors `get_HELM_alfa` in `eos/private/eosdt_eval.f90`.
#[must_use]
pub fn get_helm_alfa(
    _ht: &HelmTable,
    log_rho: f64,
    log_t: f64,
    helm_blend_width: f64,
) -> (f64, f64, f64) {
    let bounds = [
        [HELM_LOGDLO, HELM_LOGTHI],
        [HELM_LOGDLO, HELM_LOGTLO],
        [HELM_LOGDHI, HELM_LOGTLO],
        [HELM_LOGDHI, HELM_LOGTHI],
    ];
    let p = [
        AutoDiffReal2VarOrder1::var1(log_rho),
        AutoDiffReal2VarOrder1::var2(log_t),
    ];

    let contained = is_contained(&bounds, p);
    let mut dist = min_distance_to_polygon(&bounds, p);
    if contained {
        dist = -dist;
    }

    let scaled = dist / helm_blend_width;
    let blend = scaled
        .max(AutoDiffReal2VarOrder1::zero())
        .min(AutoDiffReal2VarOrder1::constant(1.0));

    (blend.val, blend.d1[1], blend.d1[0])
}

/// Combines two evaluated EOS components (`out_1st` with weight `1 - alfa` and
/// `out_2nd` with weight `alfa`) using `blend_eos_results`.
///
/// Matches `combine_for_eosdt` in `eos/private/eosdt_eval.f90`.
#[must_use]
pub fn combine_two_eos_components(
    alfa_in: f64,
    d_alfa_dlog_t_in: f64,
    d_alfa_dlog_rho_in: f64,
    linear_blend: bool,
    out_1st: &EosEvalOutput,
    out_2nd: &EosEvalOutput,
) -> EosEvalOutput {
    blend_eos_results(
        alfa_in,
        d_alfa_dlog_t_in,
        d_alfa_dlog_rho_in,
        linear_blend,
        out_2nd,
        out_1st,
    )
}

/// Evaluates `HELM` for `eosDT_get` with the given [`HelmTable`] and [`HelmConfig`].
#[allow(clippy::too_many_arguments)]
fn eval_helm_component(
    ht: &HelmTable,
    rho: f64,
    log_rho: f64,
    t: f64,
    log_t: f64,
    abar: f64,
    zbar: f64,
    cfg: &HelmConfig,
) -> Result<EosEvalOutput, EosError> {
    let (helm_res, _off_table) = helmeos2(
        ht, t, log_t, rho, log_rho, abar, zbar, cfg,
    )
    .map_err(|_| EosError::InvalidState {
        context: "eval_helm_component",
        detail: format!("helmeos2 failed at rho={rho:e}, T={t:e}"),
    })?;
    let (out, _, _) = do_convert_helm_results(abar, zbar, rho, t, &helm_res);
    Ok(out)
}

/// Evaluates level 6 (`HELM` blended into `ideal` outside table bounds).
#[allow(clippy::too_many_arguments)]
fn eval_level6_from_weights(
    ht: &HelmTable,
    rho: f64,
    log_rho: f64,
    t: f64,
    log_t: f64,
    abar: f64,
    zbar: f64,
    xa: &[f64],
    acmi: &[f64],
    a_baryon: &[f64],
    cfg: &EosConfig,
) -> Result<EosEvalOutput, EosError> {
    let (alfa, d_alfa_dlog_t, d_alfa_dlog_rho) =
        get_helm_alfa(ht, log_rho, log_t, cfg.helm_blend_width);

    if alfa <= 0.0 {
        return eval_helm_component(
            ht, rho, log_rho, t, log_t, abar, zbar, &cfg.helm,
        );
    }
    if alfa >= 1.0 {
        return eval_ideal_eos_from_weights(
            t, rho, abar, zbar, xa, acmi, a_baryon,
        );
    }

    let out_helm =
        eval_helm_component(ht, rho, log_rho, t, log_t, abar, zbar, &cfg.helm)?;
    let out_ideal =
        eval_ideal_eos_from_weights(t, rho, abar, zbar, xa, acmi, a_baryon)?;
    Ok(combine_two_eos_components(
        alfa,
        d_alfa_dlog_t,
        d_alfa_dlog_rho,
        cfg.linear_blend,
        &out_helm,
        &out_ideal,
    ))
}

/// Evaluates level 3 (`PC` blended into level 6).
#[allow(clippy::too_many_arguments)]
fn eval_level3_from_weights(
    ht: &HelmTable,
    rho: f64,
    log_rho: f64,
    t: f64,
    log_t: f64,
    abar: f64,
    zbar: f64,
    xa: &[f64],
    azion: &[f64],
    acmi: &[f64],
    a_baryon: &[f64],
    cfg: &EosConfig,
) -> Result<EosEvalOutput, EosError> {
    let (alfa, d_alfa_dlog_t, d_alfa_dlog_rho) = if cfg.use_pc {
        get_pc_alfa(&cfg.pc, log_rho, log_t, abar, zbar)
    } else {
        (1.0, 0.0, 0.0)
    };

    if alfa <= 0.0 {
        return eval_pc_eos_from_weights(
            &cfg.pc, rho, t, abar, zbar, xa, azion, acmi,
        );
    }
    if alfa >= 1.0 {
        return eval_level6_from_weights(
            ht, rho, log_rho, t, log_t, abar, zbar, xa, acmi, a_baryon, cfg,
        );
    }

    let out_pc =
        eval_pc_eos_from_weights(&cfg.pc, rho, t, abar, zbar, xa, azion, acmi)?;
    let out_lvl6 = eval_level6_from_weights(
        ht, rho, log_rho, t, log_t, abar, zbar, xa, acmi, a_baryon, cfg,
    )?;
    Ok(combine_two_eos_components(
        alfa,
        d_alfa_dlog_t,
        d_alfa_dlog_rho,
        cfg.linear_blend,
        &out_pc,
        &out_lvl6,
    ))
}

/// Evaluates the hierarchical `eosDT_get` equation of state from precomputed atomic weights.
///
/// Follows the upstream MESA `Get_eosDT_Results` cascade (`Skye -> PC -> HELM -> ideal`).
#[allow(clippy::too_many_arguments)]
pub fn eos_dt_get_from_weights(
    ht: &HelmTable,
    rho: f64,
    log_rho: f64,
    t: f64,
    log_t: f64,
    abar: f64,
    zbar: f64,
    xa: &[f64],
    z_int: &[usize],
    azion: &[f64],
    acmi: &[f64],
    a_baryon: &[f64],
    cfg: &EosConfig,
) -> Result<EosEvalOutput, EosError> {
    let (alfa, d_alfa_dlog_t, d_alfa_dlog_rho) = if cfg.use_skye {
        if cfg.skye.use_simple_skye_blends {
            get_skye_alfa_simple(&cfg.skye, log_rho, log_t)
        } else {
            get_skye_alfa(ht, log_rho, log_t, abar, zbar)
        }
    } else {
        (1.0, 0.0, 0.0)
    };

    if alfa <= 0.0 {
        return eval_skye_eos_from_weights(
            &cfg.skye, ht, t, rho, abar, zbar, xa, z_int, acmi, a_baryon,
        );
    }
    if alfa >= 1.0 {
        return eval_level3_from_weights(
            ht, rho, log_rho, t, log_t, abar, zbar, xa, azion, acmi, a_baryon,
            cfg,
        );
    }

    let out_skye = eval_skye_eos_from_weights(
        &cfg.skye, ht, t, rho, abar, zbar, xa, z_int, acmi, a_baryon,
    )?;
    let out_lvl3 = eval_level3_from_weights(
        ht, rho, log_rho, t, log_t, abar, zbar, xa, azion, acmi, a_baryon, cfg,
    )?;
    Ok(combine_two_eos_components(
        alfa,
        d_alfa_dlog_t,
        d_alfa_dlog_rho,
        cfg.linear_blend,
        &out_skye,
        &out_lvl3,
    ))
}

/// Evaluates `eosDT_get` for a single zone given `chem_id` and `xa`.
pub fn eos_dt_get(
    ht: &HelmTable,
    rho: f64,
    t: f64,
    chem_id: &[usize],
    xa: &[f64],
    cfg: &EosConfig,
) -> Result<EosEvalOutput, EosError> {
    let catalog = NuclideCatalog::builtin_ref();
    let comp =
        get_composition_info(catalog, chem_id, xa, true).map_err(|e| {
            EosError::InvalidState {
                context: "eos_dt_get",
                detail: e.to_string(),
            }
        })?;
    let (z_int, azion, acmi, a_baryon) =
        extract_species_weights(catalog, chem_id, "eos_dt_get")?;
    let log_rho = rho.log10();
    let log_t = t.log10();
    eos_dt_get_from_weights(
        ht, rho, log_rho, t, log_t, comp.abar, comp.zbar, xa, &z_int, &azion,
        &acmi, &a_baryon, cfg,
    )
}

/// Evaluates `eosDT_get` in parallel across `N` stellar zones using `rayon`.
pub fn eos_dt_get_zones(
    ht: &HelmTable,
    rho_zones: &[f64],
    t_zones: &[f64],
    chem_id: &[usize],
    xa_zones: &[f64],
    cfg: &EosConfig,
) -> Result<Vec<EosEvalOutput>, EosError> {
    let n_zones = rho_zones.len();
    let n_spec = chem_id.len();
    if t_zones.len() != n_zones || xa_zones.len() != n_zones * n_spec {
        return Err(EosError::InvalidState {
            context: "eos_dt_get_zones",
            detail: "zone slice length mismatch".to_string(),
        });
    }

    let catalog = NuclideCatalog::builtin_ref();
    let (z_int, azion, acmi, a_baryon) =
        extract_species_weights(catalog, chem_id, "eos_dt_get_zones")?;

    (0..n_zones)
        .into_par_iter()
        .map(|iz| {
            let xa = &xa_zones[iz * n_spec..(iz + 1) * n_spec];
            let comp = get_composition_info(catalog, chem_id, xa, true)
                .map_err(|e| EosError::InvalidState {
                    context: "eos_dt_get_zones",
                    detail: e.to_string(),
                })?;
            let rho = rho_zones[iz];
            let t = t_zones[iz];
            let (lr, lt) = (rho.log10(), t.log10());
            eos_dt_get_from_weights(
                ht, rho, lr, t, lt, comp.abar, comp.zbar, xa, &z_int, &azion,
                &acmi, &a_baryon, cfg,
            )
        })
        .collect()
}
