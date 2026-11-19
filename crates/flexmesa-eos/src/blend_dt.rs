// SPDX-License-Identifier: LGPL-3.0-or-later
//! Hierarchical 7-level `eosDT_get` equation of state evaluator and blending cascade.
//!
//! Ports the hierarchical evaluation tree from upstream MESA `eosdt_eval.f90`
//! (`Get_eosDT_Results`, `get_level1_for_eosdt` through `get_level6_for_eosdt`,
//! `get_HELM_alfa`, and `combine_for_eosdt`):
//! `CMS -> Skye -> PC -> FreeEOS -> OPAL/SCVH -> HELM -> ideal`.

use crate::blend::{blend_eos_results, is_contained, min_distance_to_polygon};
use crate::cms::{
    CmsConfig, CmsTableSet, eval_cms_eos_from_weights, get_cms_alfa,
};
use crate::helm::{
    HELM_LOGDHI, HELM_LOGDLO, HELM_LOGTHI, HELM_LOGTLO, HelmConfig, HelmTable,
    do_convert_helm_results, helmeos2,
};
use crate::ideal::eval_ideal_eos_from_weights;
use crate::pc::{PcConfig, eval_pc_eos_from_weights, get_pc_alfa};
use crate::skye::{
    SkyeConfig, eval_skye_eos_from_weights, get_skye_alfa, get_skye_alfa_simple,
};
use crate::tables_4d::{
    EosDtTableKind, EosDtTableSet, FreeEosConfig, OpalScvhConfig,
    eval_eosdt_table_from_weights, get_free_eos_alfa,
    get_opal_scvh_alfa_and_partials,
};
use crate::types::{
    EosError, EosEvalOutput, I_LN_E, I_LN_PGAS, NUM_EOS_D_DXA_RESULTS,
};
use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_chem::{NuclideCatalog, get_composition_info};
use rayon::prelude::*;

/// Unified configuration for the hierarchical 7-level `eosDT_get` evaluator.
#[allow(clippy::struct_excessive_bools)]
#[derive(
    Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize,
)]
pub struct EosConfig {
    /// Configuration for the `CMS` EOS component (level 1).
    pub cms: CmsConfig,
    /// Configuration for the `Skye` EOS component (level 2).
    pub skye: SkyeConfig,
    /// Configuration for the Potekhin & Chabrier (`PC`) EOS component (level 3).
    pub pc: PcConfig,
    /// Configuration for the `FreeEOS` tabulated EOS component (level 4).
    pub free_eos: FreeEosConfig,
    /// Configuration for the `OPAL/SCVH` tabulated EOS component (level 5).
    pub opal_scvh: OpalScvhConfig,
    /// Configuration for the Helmholtz (`HELM`) EOS component (level 6).
    pub helm: HelmConfig,
    /// Enable the `CMS` EOS component at level 1 (default `false`).
    pub use_cms: bool,
    /// Enable the `Skye` EOS component at level 2 (default `true`).
    pub use_skye: bool,
    /// Enable the `PC` EOS component at level 3 (default `false`).
    pub use_pc: bool,
    /// Enable the `FreeEOS` EOS component at level 4 (default `true`).
    pub use_free_eos: bool,
    /// Enable the `OPAL/SCVH` EOS component at level 5 (default `true`).
    pub use_opal_scvh: bool,
    /// Convert component evaluation errors to `skip` and fall back to the next level (default `true`).
    pub okay_to_convert_ierr_to_skip: bool,
    /// Width of the boundary blend zone around the `HELM` table polygon in dex (default `0.1`).
    pub helm_blend_width: f64,
    /// Use linear instead of quintic blending between EOS components (default `false`).
    pub linear_blend: bool,
}

impl Default for EosConfig {
    fn default() -> Self {
        Self {
            cms: CmsConfig::default(),
            skye: SkyeConfig::default(),
            pc: PcConfig::default(),
            free_eos: FreeEosConfig::default(),
            opal_scvh: OpalScvhConfig::default(),
            helm: HelmConfig::default(),
            use_cms: false,
            use_skye: true,
            use_pc: false,
            use_free_eos: true,
            use_opal_scvh: true,
            okay_to_convert_ierr_to_skip: true,
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

/// Computes `(x, z)` mass fractions from `(z_int, xa)`.
#[must_use]
pub(crate) fn compute_xz_from_weights(
    z_int: &[usize],
    xa: &[f64],
) -> (f64, f64) {
    let mut x = 0.0;
    let mut y = 0.0;
    for (&zi, &xi) in z_int.iter().zip(xa.iter()) {
        match zi {
            1 => x += xi,
            2 => y += xi,
            _ => {}
        }
    }
    let z = (1.0 - (x + y)).clamp(0.0, 1.0);
    (x.clamp(0.0, 1.0), z)
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

/// Returns `true` when an EOS component error represents a domain/off-table
/// evaluation failure that upstream skips under `okay_to_convert_ierr_to_skip`.
///
/// Missing or corrupt table I/O errors (`EosError::TableIo`) are never skipped.
fn should_convert_to_skip(err: &EosError, cfg: &EosConfig) -> bool {
    cfg.okay_to_convert_ierr_to_skip && !matches!(err, EosError::TableIo { .. })
}

/// Executes `combine_for_eosdt` for a single level (`skip = true` unconditionally
/// when `eval_1st` returns `Ok(None)`, or on `Err` when
/// `okay_to_convert_ierr_to_skip` is enabled).
fn combine_level_skippable<F1, F2>(
    alfa: f64,
    d_alfa_dlog_t: f64,
    d_alfa_dlog_rho: f64,
    cfg: &EosConfig,
    eval_1st: F1,
    eval_2nd: F2,
) -> Result<EosEvalOutput, EosError>
where
    F1: FnOnce() -> Result<Option<EosEvalOutput>, EosError>,
    F2: FnOnce() -> Result<EosEvalOutput, EosError>,
{
    if alfa >= 1.0 {
        return eval_2nd();
    }
    let out_1st = match eval_1st() {
        Ok(Some(out1)) => out1,
        Ok(None) => return eval_2nd(),
        Err(ref e) if should_convert_to_skip(e, cfg) => return eval_2nd(),
        Err(e) => return Err(e),
    };
    if alfa <= 0.0 {
        return Ok(out_1st);
    }
    let out_2nd = eval_2nd()?;
    Ok(combine_two_eos_components(
        alfa,
        d_alfa_dlog_t,
        d_alfa_dlog_rho,
        cfg.linear_blend,
        &out_1st,
        &out_2nd,
    ))
}

/// Executes `combine_for_eosdt` for a single level with `okay_to_convert_ierr_to_skip`.
fn combine_level<F1, F2>(
    alfa: f64,
    d_alfa_dlog_t: f64,
    d_alfa_dlog_rho: f64,
    cfg: &EosConfig,
    eval_1st: F1,
    eval_2nd: F2,
) -> Result<EosEvalOutput, EosError>
where
    F1: FnOnce() -> Result<EosEvalOutput, EosError>,
    F2: FnOnce() -> Result<EosEvalOutput, EosError>,
{
    combine_level_skippable(
        alfa,
        d_alfa_dlog_t,
        d_alfa_dlog_rho,
        cfg,
        || eval_1st().map(Some),
        eval_2nd,
    )
}

/// Evaluates `HELM` for `eosDT_get` with the given [`HelmTable`] and [`HelmConfig`]
/// (`Get_HELMEOS_Results` in `eos/private/eos_helm_eval.f90:116-154`).
///
/// Returns `Ok(None)` when `helmeos2` sets `off_table` (`skip = .true.`)
/// without invoking `do_convert_helm_results`.
#[allow(clippy::too_many_arguments)]
fn eval_helm_component(
    ht: &HelmTable,
    rho: f64,
    log_rho: f64,
    t: f64,
    log_t: f64,
    abar: f64,
    zbar: f64,
    xa: &[f64],
    z_int: &[usize],
    a_baryon: &[f64],
    cfg: &HelmConfig,
) -> Result<Option<EosEvalOutput>, EosError> {
    let clipped_cfg = HelmConfig {
        clip_to_table_boundaries: true,
        ..*cfg
    };
    let (helm_res, off_table) =
        helmeos2(ht, t, log_t, rho, log_rho, abar, zbar, &clipped_cfg)
            .map_err(|code| EosError::OffTable {
                context: "eval_helm_component",
                detail: format!(
                    "helmeos2 returned ierr={code} at rho={rho:e}, T={t:e}"
                ),
            })?;
    if off_table {
        return Ok(None);
    }
    let (mut out, d_dabar, d_dzbar) =
        do_convert_helm_results(abar, zbar, rho, t, &helm_res);
    let sumx = xa.iter().copied().sum::<f64>().max(1.0e-30);
    out.d_dxa = a_baryon
        .iter()
        .zip(z_int.iter())
        .map(|(&ai, &zi)| {
            let dabar_dx = abar * (ai - abar) / (ai * sumx);
            let dzbar_dx = abar * ((zi as f64) - zbar) / (ai * sumx);
            [
                d_dabar[I_LN_PGAS] * dabar_dx + d_dzbar[I_LN_PGAS] * dzbar_dx,
                d_dabar[I_LN_E] * dabar_dx + d_dzbar[I_LN_E] * dzbar_dx,
            ]
        })
        .collect();
    Ok(Some(out))
}

/// Evaluates the full 7-level `eosDT_get` cascade with explicit table sets.
#[allow(clippy::too_many_arguments)]
pub fn eos_dt_get_from_weights_with_tables(
    ht: &HelmTable,
    dt_tables: &EosDtTableSet,
    cms_tables: &CmsTableSet,
    x: f64,
    z: f64,
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
    let eval_lvl6 = || {
        // Upstream MESA 26.4.1 `eos/private/eosdt_eval.f90:728,749-750` passes
        // `(d_alfa_dlogRho, d_alfa_dlogT)` into `get_HELM_alfa(..., alfa,
        // d_alfa_dlogT, d_alfa_dlogRho, ierr)` and forwards `(alfa,
        // d_alfa_dlogT, d_alfa_dlogRho)` to `combine_for_eosdt` (#176).
        let (a, dr, dt) =
            get_helm_alfa(ht, log_rho, log_t, cfg.helm_blend_width);
        combine_level_skippable(
            a,
            dt,
            dr,
            cfg,
            || {
                eval_helm_component(
                    ht, rho, log_rho, t, log_t, abar, zbar, xa, z_int,
                    a_baryon, &cfg.helm,
                )
            },
            || {
                eval_ideal_eos_from_weights(
                    t, rho, abar, zbar, xa, acmi, a_baryon,
                )
            },
        )
    };

    let eval_lvl5 = || {
        let (a, dr, dt) = if cfg.use_opal_scvh {
            get_opal_scvh_alfa_and_partials(&cfg.opal_scvh, log_t, log_rho, z)?
        } else {
            (1.0, 0.0, 0.0)
        };
        combine_level(
            a,
            dt,
            dr,
            cfg,
            || {
                eval_eosdt_table_from_weights(
                    dt_tables,
                    EosDtTableKind::OpalScvh,
                    z,
                    x,
                    log_rho,
                    log_t,
                    z_int,
                    cfg.opal_scvh.use_linear_interp_for_x,
                    cfg.opal_scvh.tiny_fuzz,
                )
            },
            eval_lvl6,
        )
    };

    let eval_lvl4 = || {
        let (a, dt, dr) = if cfg.use_free_eos {
            get_free_eos_alfa(&cfg.free_eos, log_rho, log_t, z)
        } else {
            (1.0, 0.0, 0.0)
        };
        combine_level(
            a,
            dt,
            dr,
            cfg,
            || {
                eval_eosdt_table_from_weights(
                    dt_tables,
                    EosDtTableKind::FreeEos,
                    z,
                    x,
                    log_rho,
                    log_t,
                    z_int,
                    cfg.free_eos.use_linear_interp_for_x,
                    cfg.free_eos.tiny_fuzz,
                )
            },
            eval_lvl5,
        )
    };

    let eval_lvl3 = || {
        let (a, dt, dr) = if cfg.use_pc {
            get_pc_alfa(&cfg.pc, log_rho, log_t, abar, zbar)
        } else {
            (1.0, 0.0, 0.0)
        };
        combine_level(
            a,
            dt,
            dr,
            cfg,
            || {
                eval_pc_eos_from_weights(
                    &cfg.pc, rho, t, abar, zbar, xa, azion, acmi,
                )
            },
            eval_lvl4,
        )
    };

    let eval_lvl2 = || {
        let (a, dt, dr) = if cfg.use_skye {
            if cfg.skye.use_simple_skye_blends {
                get_skye_alfa_simple(&cfg.skye, log_rho, log_t)
            } else {
                get_skye_alfa(ht, log_rho, log_t, abar, zbar)
            }
        } else {
            (1.0, 0.0, 0.0)
        };
        combine_level(
            a,
            dt,
            dr,
            cfg,
            || {
                eval_skye_eos_from_weights(
                    &cfg.skye, ht, t, rho, abar, zbar, xa, z_int, acmi,
                    a_baryon,
                )
            },
            eval_lvl3,
        )
    };

    let (a_cms, dt_cms, dr_cms) = if cfg.use_cms {
        get_cms_alfa(&cfg.cms, log_rho, log_t, z)
    } else {
        (1.0, 0.0, 0.0)
    };
    let mut out = combine_level(
        a_cms,
        dt_cms,
        dr_cms,
        cfg,
        || {
            eval_cms_eos_from_weights(
                cms_tables, &cfg.cms, x, rho, log_rho, t, log_t, z_int,
            )
        },
        eval_lvl2,
    )?;
    if out.d_dxa.len() < xa.len() {
        out.d_dxa.resize(xa.len(), [0.0; NUM_EOS_D_DXA_RESULTS]);
    }
    Ok(out)
}

/// Evaluates the hierarchical 7-level `eosDT_get` equation of state from precomputed atomic weights.
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
    let (x, z) = compute_xz_from_weights(z_int, xa);
    eos_dt_get_from_weights_with_tables(
        ht,
        EosDtTableSet::builtin(),
        CmsTableSet::builtin(),
        x,
        z,
        rho,
        log_rho,
        t,
        log_t,
        abar,
        zbar,
        xa,
        z_int,
        azion,
        acmi,
        a_baryon,
        cfg,
    )
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
    let log_rho = flexmesa_math::log10(rho);
    let log_t = flexmesa_math::log10(t);
    eos_dt_get_from_weights_with_tables(
        ht,
        EosDtTableSet::builtin(),
        CmsTableSet::builtin(),
        comp.xh,
        comp.xz,
        rho,
        log_rho,
        t,
        log_t,
        comp.abar,
        comp.zbar,
        xa,
        &z_int,
        &azion,
        &acmi,
        &a_baryon,
        cfg,
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
    let dt_tables = EosDtTableSet::builtin();
    let cms_tables = CmsTableSet::builtin();

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
            let (lr, lt) = (flexmesa_math::log10(rho), flexmesa_math::log10(t));
            eos_dt_get_from_weights_with_tables(
                ht, dt_tables, cms_tables, comp.xh, comp.xz, rho, lr, t, lt,
                comp.abar, comp.zbar, xa, &z_int, &azion, &acmi, &a_baryon,
                cfg,
            )
        })
        .collect()
}
