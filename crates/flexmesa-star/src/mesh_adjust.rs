// SPDX-License-Identifier: LGPL-3.0-or-later
//! Adaptive Lagrangian mesh driver (`remesh` in `star/private/adjust_mesh.f90:1-700`),
//! bridging [`crate::mesh_functions`], [`crate::adjust_mesh_support`],
//! [`crate::mesh_plan`], [`crate::mesh_remap`], and
//! [`crate::adjust_mesh_split_merge`].

use std::sync::Mutex;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::LN10;

use crate::adjust_mesh_split_merge::{
    preview_split_merge_amr, remesh_split_merge,
};
use crate::adjust_mesh_support::{check_validity, get_gval_info_with_controls};
use crate::controls::{StarControls, StarError};
use crate::controls_mix_wind::{MixWindControls, NUM_XA_FUNCTION};
use crate::mesh_functions::{
    get_net_iso_0based, num_mesh_functions_with_controls,
};
use crate::mesh_plan::do_mesh_plan;
pub use crate::mesh_remap::{
    CellType, StandardMeshPlan, classify_new_cells, compute_monotone_xa_slopes,
    do_prune_mesh_surface, remap_onto_dq, set_lnt_for_energy,
    solve_cell_lnt_for_energy,
};
use crate::mesh_remap::{
    compute_xqs, execute_adjust_mesh_from_plan, fix_tail_dq,
};
use crate::model::StarModel;

/// Cached outcome from [`plan_mesh`] consumed by [`remap_onto_dq`].
#[derive(Debug, Clone)]
pub(crate) enum PlannedMeshOutcome {
    /// Standard MESA `adjust_mesh` plan (`use_split_merge_amr = false`).
    Standard(Box<StandardMeshPlan>),
    /// Hydrodynamic split-merge AMR trigger (`use_split_merge_amr = true`).
    SplitMergeAmr {
        /// Snapshot of `model.dq` when `plan_mesh` ran.
        raw_dq_old: Vec<f64>,
    },
    /// Deferred error detected inside infallible `plan_mesh` to be surfaced
    /// immediately by `remap_onto_dq` / `do_adjust_mesh`.
    DeferredError(StarError),
}

/// Interior-mutable slot on [`crate::hydro_state::HydroAuxState`] carrying a
/// `PlannedMeshOutcome` across the `(plan_mesh, remap_onto_dq)` call boundary.
#[derive(Default)]
pub struct PlannedMeshSlot(Mutex<Option<PlannedMeshOutcome>>);

impl Clone for PlannedMeshSlot {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for PlannedMeshSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PlannedMeshSlot(..)")
    }
}

impl PlannedMeshSlot {
    pub(crate) fn store(&self, outcome: PlannedMeshOutcome) {
        if let Ok(mut guard) = self.0.lock() {
            *guard = Some(outcome);
        }
    }

    pub(crate) fn take(&self) -> Option<PlannedMeshOutcome> {
        self.0.lock().ok().and_then(|mut guard| guard.take())
    }
}

/// Weights and scale parameters for the adaptive mesh spacing functions
/// (`star/defaults/controls.defaults:5788-6278`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshFunctionConfig {
    /// Weight $w_P$ on $\log_{10} P_{\mathrm{eos}}$ (`40.0`).
    pub p_function_weight: f64,
    /// Weight $w_{T1}$ on $\log_{10} T$ (`110.0`).
    pub t_function1_weight: f64,
    /// Weight $w_R$ on $\log_{10}(1 + (r/R_\odot)/\alpha_R)$ (`0.0`).
    pub r_function_weight: f64,
    /// Scale $\alpha_R$ for $f_R$ (`1.0e-4`).
    pub r_function_param: f64,
    /// Weight $w_M$ on $\log_{10}(1 + (M_{\mathrm{env}} q / M_\odot)/\alpha_M)$ (`0.0`).
    pub m_function_weight: f64,
    /// Scale $\alpha_M$ for $f_M$ (`1.0e-6`).
    pub m_function_param: f64,
    /// Weight $w_\nabla$ on temperature gradient $\nabla_T$ (`gradT_function_weight`, `0.0`).
    pub grad_function_weight: f64,
    /// Weight $w_\Omega$ on rotational shear profile (`0.0`).
    pub omega_function_weight: f64,
    /// Weight $w_\Gamma$ on Coulomb crystallization transition (`0.0`).
    pub gam_function_weight: f64,
    /// Center parameter $\Gamma_1$ for $f_\Gamma$ (`170.0`).
    pub gam_function_param1: f64,
    /// Width parameter $\Gamma_2$ for $f_\Gamma$ (`20.0`).
    pub gam_function_param2: f64,
    /// Primary weight $w_X$ on species logarithmic profile (`30.0`).
    pub xa_function_weight: f64,
    /// Primary floor $X_0$ for species logarithmic profile (`1.0e-2`).
    pub xa_function_param: f64,
    /// Minimum allowed zone relative mass fraction $\Delta q_{\min}$ (`1e-14`).
    pub min_dq: f64,
    /// Maximum allowed zone relative mass fraction $\Delta q_{\max}$ (`1e-2`).
    pub max_dq: f64,
    /// Minimum post-split relative mass fraction (`1e-14`).
    pub min_dq_for_split: f64,
    /// Minimum cell $\Delta q$ for composition-driven splitting (`min_dq_for_xa`, `1e-14`).
    pub min_dq_for_xa: f64,
    /// Minimum cell $\Delta q$ for composition-driven splitting in convective zones (`1e-6`).
    pub min_dq_for_xa_convective: f64,
    /// Minimum $\Delta\ln R$ across a split subcell (`mesh_min_dlnR`, `1e-9`).
    pub mesh_min_dlnr: f64,
    /// Maximum allowed adjacent cell $\Delta q$ ratio (`mesh_max_allowed_ratio`, `2.5`).
    pub mesh_max_allowed_ratio: f64,
    /// Maximum species difference allowed when merging cells (`max_delta_x_for_merge`, `0.1`).
    pub max_delta_x_for_merge: f64,
    /// Maximum surface cell relative mass fraction (`1e-12`).
    pub max_surface_cell_dq: f64,
    /// Maximum center cell relative mass fraction (`1e-7`).
    pub max_center_cell_dq: f64,
}

impl Default for MeshFunctionConfig {
    fn default() -> Self {
        Self {
            p_function_weight: 40.0,
            t_function1_weight: 110.0,
            r_function_weight: 0.0,
            r_function_param: 1.0e-4,
            m_function_weight: 0.0,
            m_function_param: 1.0e-6,
            grad_function_weight: 0.0,
            omega_function_weight: 0.0,
            gam_function_weight: 0.0,
            gam_function_param1: 170.0,
            gam_function_param2: 20.0,
            xa_function_weight: 30.0,
            xa_function_param: 1.0e-2,
            min_dq: 1.0e-14,
            max_dq: 1.0e-2,
            min_dq_for_split: 1.0e-14,
            min_dq_for_xa: 1.0e-14,
            min_dq_for_xa_convective: 1.0e-6,
            mesh_min_dlnr: 1.0e-9,
            mesh_max_allowed_ratio: 2.5,
            max_delta_x_for_merge: 0.1,
            max_surface_cell_dq: 1.0e-12,
            max_center_cell_dq: 1.0e-7,
        }
    }
}

impl MeshFunctionConfig {
    /// Builds [`MeshFunctionConfig`] from [`StarControls`] (`mix_wind`).
    #[must_use]
    pub fn from_controls(c: &StarControls) -> Self {
        let mw = &c.mix_wind;
        let max_xa_w = mw
            .xa_function_weight
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        let xa_p = mw
            .xa_function_param
            .iter()
            .copied()
            .find(|&p| p > 0.0)
            .unwrap_or(1.0e-2);
        Self {
            p_function_weight: mw.p_function_weight,
            t_function1_weight: mw.t_function1_weight,
            r_function_weight: mw.r_function_weight,
            r_function_param: mw.r_function_param,
            m_function_weight: mw.m_function_weight,
            m_function_param: mw.m_function_param,
            grad_function_weight: mw.gradt_function_weight,
            omega_function_weight: mw.omega_function_weight,
            gam_function_weight: mw.gam_function_weight,
            gam_function_param1: mw.gam_function_param1,
            gam_function_param2: mw.gam_function_param2,
            xa_function_weight: max_xa_w,
            xa_function_param: xa_p,
            min_dq: mw.min_dq,
            max_dq: mw.max_dq,
            min_dq_for_split: mw.min_dq_for_split,
            min_dq_for_xa: mw.min_dq_for_xa,
            min_dq_for_xa_convective: mw.min_dq_for_xa_convective,
            mesh_min_dlnr: mw.mesh_min_dlnr,
            mesh_max_allowed_ratio: mw.mesh_max_allowed_ratio,
            max_delta_x_for_merge: mw.max_delta_x_for_merge,
            max_surface_cell_dq: mw.max_surface_cell_dq,
            max_center_cell_dq: mw.max_center_cell_dq,
        }
    }
}

fn effective_mix_wind_controls(
    model: &StarModel,
    cfg: &MeshFunctionConfig,
) -> MixWindControls {
    let mut mw = model.controls.mix_wind.clone();
    let base = MeshFunctionConfig::from_controls(&model.controls);
    if *cfg == base {
        return mw;
    }
    mw.p_function_weight = cfg.p_function_weight;
    mw.t_function1_weight = cfg.t_function1_weight;
    mw.r_function_weight = cfg.r_function_weight;
    mw.r_function_param = cfg.r_function_param;
    mw.m_function_weight = cfg.m_function_weight;
    mw.m_function_param = cfg.m_function_param;
    mw.gradt_function_weight = cfg.grad_function_weight;
    mw.omega_function_weight = cfg.omega_function_weight;
    mw.gam_function_weight = cfg.gam_function_weight;
    mw.gam_function_param1 = cfg.gam_function_param1;
    mw.gam_function_param2 = cfg.gam_function_param2;
    mw.min_dq = cfg.min_dq;
    mw.max_dq = cfg.max_dq;
    mw.min_dq_for_split = cfg.min_dq_for_split;
    mw.min_dq_for_xa = cfg.min_dq_for_xa;
    mw.min_dq_for_xa_convective = cfg.min_dq_for_xa_convective;
    mw.mesh_min_dlnr = cfg.mesh_min_dlnr;
    mw.mesh_max_allowed_ratio = cfg.mesh_max_allowed_ratio;
    mw.max_delta_x_for_merge = cfg.max_delta_x_for_merge;
    mw.max_surface_cell_dq = cfg.max_surface_cell_dq;
    mw.max_center_cell_dq = cfg.max_center_cell_dq;

    if (cfg.xa_function_weight - base.xa_function_weight).abs() > 0.0
        || (cfg.xa_function_param - base.xa_function_param).abs() > 0.0
    {
        if cfg.xa_function_weight <= 0.0 {
            mw.xa_function_weight.fill(0.0);
        } else {
            if mw.xa_function_species.is_empty()
                || mw.xa_function_species[0].trim().is_empty()
            {
                if mw.xa_function_species.is_empty() {
                    mw.xa_function_species.push(String::from("he4"));
                } else {
                    mw.xa_function_species[0] = String::from("he4");
                }
            }
            if mw.xa_function_weight.is_empty() {
                mw.xa_function_weight.push(cfg.xa_function_weight);
            } else {
                mw.xa_function_weight[0] = cfg.xa_function_weight;
            }
            if mw.xa_function_param.is_empty() {
                mw.xa_function_param.push(cfg.xa_function_param);
            } else {
                mw.xa_function_param[0] = cfg.xa_function_param;
            }
        }
    }
    mw
}

/// Computes `delta_coeff` including high-temperature blending and heaviest-isotope
/// `xa_mesh_delta_coeff` scaling (`star/private/adjust_mesh.f90:143-179`).
#[must_use]
pub fn eval_effective_mesh_delta_coeff(
    model: &StarModel,
    mw: &MixWindControls,
) -> f64 {
    let nz = model.nz;
    let base_coeff = if model.controls.mesh_delta_coeff > 0.0 {
        model.controls.mesh_delta_coeff
    } else {
        1.0
    };
    if nz == 0 {
        return base_coeff;
    }
    let center_logt = model.ln_t[nz - 1] / LN10;
    let mut delta_coeff = if center_logt
        <= mw.logt_max_for_standard_mesh_delta_coeff
    {
        base_coeff
    } else if center_logt >= mw.logt_min_for_hight_mesh_delta_coeff {
        mw.mesh_delta_coeff_for_hight
    } else {
        let alfa = (center_logt - mw.logt_max_for_standard_mesh_delta_coeff)
            / (mw.logt_min_for_hight_mesh_delta_coeff
                - mw.logt_max_for_standard_mesh_delta_coeff);
        let beta = 1.0 - alfa;
        alfa * mw.mesh_delta_coeff_for_hight + beta * base_coeff
    };

    let catalog = NuclideCatalog::builtin_ref();
    let mut a_max = 0.0_f64;
    for &cid in &model.chem_id {
        if let Some(iso) = catalog.get_by_id(cid) {
            if iso.w > a_max {
                a_max = iso.w;
            }
        }
    }

    for j in 0..NUM_XA_FUNCTION {
        let fac = mw.xa_mesh_delta_coeff.get(j).copied().unwrap_or(1.0);
        if fac == 1.0 {
            continue;
        }
        let sp = mw.xa_function_species.get(j).map_or("", |s| s.trim());
        let Some(j0) = get_net_iso_0based(model, sp) else {
            continue;
        };
        if let Some(iso) = catalog.get_by_id(model.chem_id[j0]) {
            if iso.w == a_max {
                delta_coeff *= fac;
            }
        }
    }
    delta_coeff
}

/// Evaluates the dimensionless mesh variation indicator $g_k$ across all zones
/// `0..nz` from smoothed upstream mesh functions and local `delta_gval_max`
/// (`mesh_functions.f90` and `adjust_mesh_support.f90`).
#[must_use]
pub fn eval_mesh_gval(model: &StarModel, cfg: &MeshFunctionConfig) -> Vec<f64> {
    let nz = model.nz;
    if nz < 2 {
        return vec![0.0; nz];
    }
    let mw = effective_mix_wind_controls(model, cfg);
    let Ok(num_gvals) = num_mesh_functions_with_controls(model, &mw) else {
        return vec![0.0; nz];
    };
    if num_gvals == 0 {
        return vec![0.0; nz];
    }
    let Ok(mut ginfo) = get_gval_info_with_controls(model, &mw, num_gvals)
    else {
        return vec![0.0; nz];
    };
    let delta_coeff = eval_effective_mesh_delta_coeff(model, &mw);
    for d in &mut ginfo.delta_gval_max {
        *d *= delta_coeff;
    }

    let mut out = vec![0.0_f64; nz];
    for k in 0..(nz - 1) {
        let dg = ginfo.delta_gval_max[k]
            .min(ginfo.delta_gval_max[k + 1])
            .max(1.0e-99);
        let mut max_ratio = 0.0_f64;
        for j in 0..num_gvals {
            let col = ginfo.fcns.column(j, nz);
            let ratio = (col[k] - col[k + 1]).abs() / dg;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
        out[k] = max_ratio;
    }
    out[nz - 1] = out[nz - 2];
    out
}

/// Fallible standard mesh planning driver (`remesh` planning half in
/// `star/private/adjust_mesh.f90:84-390`).
///
/// # Errors
/// Returns [`StarError`] if pre-remesh validity checks fail, zero mesh functions
/// are active, custom hooks fail, or `do_mesh_plan` exceeds `max_allowed_nz`.
pub fn try_plan_standard_mesh(
    model: &StarModel,
    cfg: &MeshFunctionConfig,
) -> Result<Option<StandardMeshPlan>, StarError> {
    let nz = model.nz;
    if nz < 4 {
        return Ok(None);
    }
    let mw = effective_mix_wind_controls(model, cfg);

    if mw.remesh_max_allowed_logt < 100.0 {
        let max_logt =
            model.ln_t.iter().copied().fold(f64::NEG_INFINITY, f64::max) / LN10;
        if max_logt > mw.remesh_max_allowed_logt {
            return Ok(None);
        }
    }

    let delta_coeff = eval_effective_mesh_delta_coeff(model, &mw);
    // Upstream `star/private/adjust_mesh.f90:155-158, 280` copies
    // `mesh_max_allowed_ratio = s% mesh_max_allowed_ratio` before clamping
    // `s% mesh_max_allowed_ratio = 2.5d0`, passing the unclamped scalar to
    // `do_mesh_plan` on the first call (`#250` under `#169`).
    let mesh_max_allowed_ratio = mw.mesh_max_allowed_ratio;

    check_validity(model)?;

    let num_gvals = num_mesh_functions_with_controls(model, &mw)?;
    if num_gvals == 0 {
        return Err(StarError::InvalidInput(String::from(
            "must have at least 1 mesh gradient function with nonzero weight",
        )));
    }

    let mut ginfo = get_gval_info_with_controls(model, &mw, num_gvals)?;

    let mut do_not_split = vec![false; nz];
    for k in 0..(nz - 1) {
        do_not_split[k] = (model.ln_r[k] - model.ln_r[k + 1]
            < 2.0 * mw.mesh_min_dlnr)
            || (model.r[k] - model.r[k + 1]
                < 2.0 * mw.mesh_min_dr_div_cs * model.csound[k]);
    }
    do_not_split[nz - 1] = model.r[nz - 1] - model.r_center
        < 2.0 * mw.mesh_min_dr_div_cs * model.csound[nz - 1];

    let raw_dq_old = model.dq.clone();
    let mut dq_old = raw_dq_old.clone();
    fix_tail_dq(&mut dq_old)?;
    let xq_old = compute_xqs(&dq_old);

    for d in &mut ginfo.delta_gval_max {
        *d *= delta_coeff;
    }

    let raw_mesh_coeff = if model.controls.mesh_delta_coeff > 0.0 {
        model.controls.mesh_delta_coeff
    } else {
        1.0
    };

    let plan_out = do_mesh_plan(
        model,
        &mw,
        &xq_old,
        &dq_old,
        mw.min_dq,
        mw.max_dq * raw_mesh_coeff,
        mw.min_dq_for_split,
        mesh_max_allowed_ratio,
        &do_not_split,
        &ginfo.fcns,
        &ginfo.delta_gval_max,
        mw.max_center_cell_dq * raw_mesh_coeff,
        mw.max_surface_cell_dq * raw_mesh_coeff,
        mw.min_surface_cell_dq * raw_mesh_coeff,
    )?;

    let mut dq_new_fixed = plan_out.dq_new.clone();
    fix_tail_dq(&mut dq_new_fixed)?;
    let xq_new_fixed = compute_xqs(&dq_new_fixed);

    let (cell_type, _unchanged, num_splits, num_merges) =
        classify_new_cells(&xq_old, &plan_out.xq_new, &plan_out.comes_from)?;

    Ok(Some(StandardMeshPlan {
        nz_old: nz,
        dq_old,
        raw_dq_old,
        xq_old,
        dq_new: dq_new_fixed,
        xq_new: xq_new_fixed,
        comes_from: plan_out.comes_from,
        which_gval: plan_out.which_gval,
        cell_type,
        num_splits,
        num_merges,
    }))
}

fn suppressed_by_evolve_guards(
    model: &StarModel,
    cfg: &MeshFunctionConfig,
) -> bool {
    let from_ctrls = *cfg == MeshFunctionConfig::from_controls(&model.controls);
    if !from_ctrls {
        return false;
    }
    let mw = &model.controls.mix_wind;
    if model.controls.rsp_flag && !mw.use_split_merge_amr {
        return true;
    }
    if mw.restore_mesh_on_retry && mw.num_steps_to_hold_mesh_after_retry > 0 {
        if let Some(m_ret) = model.aux.step_hist.model_number_for_last_retry {
            if (m_ret as isize) + mw.num_steps_to_hold_mesh_after_retry
                > (model.model_number as isize)
            {
                return true;
            }
        }
    }
    false
}

/// Plans cell splits and merges (`do_mesh_plan` in `star/private/mesh_plan.f90`
/// or `remesh_split_merge` preview when `use_split_merge_amr` is active),
/// returning `(dq_new, num_splits, num_merges)`.
#[must_use]
pub fn plan_mesh(
    model: &StarModel,
    cfg: &MeshFunctionConfig,
) -> (Vec<f64>, usize, usize) {
    let _ = model.aux.planned_mesh.take();
    if model.nz < 4 || suppressed_by_evolve_guards(model, cfg) {
        return (model.dq.clone(), 0, 0);
    }

    let from_ctrls = *cfg == MeshFunctionConfig::from_controls(&model.controls);
    if from_ctrls && model.controls.mix_wind.use_split_merge_amr {
        match preview_split_merge_amr(model) {
            Ok((0, 0)) => return (model.dq.clone(), 0, 0),
            Ok((ns, nm)) => {
                model.aux.planned_mesh.store(
                    PlannedMeshOutcome::SplitMergeAmr {
                        raw_dq_old: model.dq.clone(),
                    },
                );
                return (model.dq.clone(), ns, nm);
            }
            Err(err) => {
                model
                    .aux
                    .planned_mesh
                    .store(PlannedMeshOutcome::DeferredError(err));
                return (model.dq.clone(), 1, 0);
            }
        }
    }

    match try_plan_standard_mesh(model, cfg) {
        Ok(None) => (model.dq.clone(), 0, 0),
        Ok(Some(plan)) => {
            let (ns, nm) = (plan.num_splits, plan.num_merges);
            if ns == 0 && nm == 0 {
                return (model.dq.clone(), 0, 0);
            }
            let dq_ret = plan.dq_new.clone();
            model
                .aux
                .planned_mesh
                .store(PlannedMeshOutcome::Standard(Box::new(plan)));
            (dq_ret, ns, nm)
        }
        Err(err) => {
            model
                .aux
                .planned_mesh
                .store(PlannedMeshOutcome::DeferredError(err));
            (model.dq.clone(), 1, 0)
        }
    }
}

/// Executes adaptive mesh planning and conservative remapping (`remesh` in
/// `star/private/adjust_mesh.f90` or `remesh_split_merge` in
/// `star/private/adjust_mesh_split_merge.f90`), returning
/// `(num_splits, num_merges)`.
///
/// # Errors
/// Returns [`StarError`] if mesh planning, remapping, or microphysics fails.
pub fn do_adjust_mesh(
    model: &mut StarModel,
    cfg: &MeshFunctionConfig,
) -> Result<(usize, usize), StarError> {
    let _ = model.aux.planned_mesh.take();
    if !model.controls.okay_to_remesh || suppressed_by_evolve_guards(model, cfg)
    {
        return Ok((0, 0));
    }

    let from_ctrls = *cfg == MeshFunctionConfig::from_controls(&model.controls);
    if from_ctrls && model.controls.mix_wind.use_split_merge_amr {
        return remesh_split_merge(model);
    }

    let Some(plan) = try_plan_standard_mesh(model, cfg)? else {
        return Ok((0, 0));
    };
    if from_ctrls && model.controls.mix_wind.mesh_max_allowed_ratio < 2.5 {
        model.controls.mix_wind.mesh_max_allowed_ratio = 2.5;
    }
    let (ns, nm) = (plan.num_splits, plan.num_merges);
    if ns == 0 && nm == 0 {
        return Ok((0, 0));
    }
    execute_adjust_mesh_from_plan(model, plan)?;
    Ok((ns, nm))
}
