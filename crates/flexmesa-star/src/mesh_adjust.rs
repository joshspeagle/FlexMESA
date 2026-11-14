// SPDX-License-Identifier: LGPL-3.0-or-later
//! Adaptive Lagrangian mesh spacing functions and cell split/merge planning
//! (`mesh_functions.f90`, `mesh_plan.f90`, `adjust_mesh_split_merge.f90`, and
//! `mesh_adjust.f90`).
//!
//! # Mesh Spacing Functions
//! Following Paxton et al. (2011, §6.1), the mesh spacing vector
//! $\mathbf{f}_k = (f_{1,k}, \dots, f_{M,k})$ includes weighted logarithmic
//! thermodynamic, geometric, rotational, Coulomb, and composition profiles:
//! - Pressure: $f_{P,k} = w_P \log_{10} P_{\mathrm{eos},k}$
//! - Temperature: $f_{T1,k} = w_{T1} \log_{10} T_k$
//! - Radius: $f_{R,k} = w_R \log_{10}(1 + (r_k / R_\odot) / \alpha_R)$
//! - Mass: $f_{M,k} = w_M \log_{10}(1 + (m_k / M_\odot) / \alpha_M)$
//! - Convective gradient transition:
//!   $f_{\nabla,k} = w_\nabla \tanh((\nabla_{\mathrm{rad},k} - \nabla_{L,k}) / 0.01)$
//! - Angular velocity: $f_{\Omega,k} = w_\Omega \log_{10}(|\Omega_k| + 10^{-8})$
//! - Coulomb coupling:
//!   $f_{\Gamma,k} = w_\Gamma \tanh((\Gamma_k - \Gamma_1) / \Gamma_2)$
//! - Species abundances: $f_{X_j,k} = w_X \log_{10}(X_{j,k} + X_0)$

use flexmesa_const::{LN10, M_SUN, R_SUN};

use crate::controls::{StarControls, StarError};
pub use crate::mesh_remap::{
    compute_monotone_xa_slopes, remap_onto_dq, set_lnt_for_energy,
};
use crate::model::StarModel;

/// Weights and scale parameters for the adaptive mesh spacing functions
/// (`mesh_functions.f90` and `mesh_plan.f90`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshFunctionConfig {
    /// Weight $w_P$ on $\log_{10} P_{\mathrm{eos}}$ (`30.0`).
    pub p_function_weight: f64,
    /// Weight $w_{T1}$ on $\log_{10} T$ (`110.0`).
    pub t_function1_weight: f64,
    /// Weight $w_R$ on $\log_{10}(1 + (r/R_\odot)/\alpha_R)$ (`0.0`).
    pub r_function_weight: f64,
    /// Scale $\alpha_R$ for $f_R$ (`1.0e-4`).
    pub r_function_param: f64,
    /// Weight $w_M$ on $\log_{10}(1 + (m/M_\odot)/\alpha_M)$ (`0.0`).
    pub m_function_weight: f64,
    /// Scale $\alpha_M$ for $f_M$ (`1.0e-2`).
    pub m_function_param: f64,
    /// Weight $w_\nabla$ on convective boundary transition (`10.0`).
    pub grad_function_weight: f64,
    /// Weight $w_\Omega$ on rotational shear profile (`0.0`).
    pub omega_function_weight: f64,
    /// Weight $w_\Gamma$ on Coulomb crystallization transition (`0.0`).
    pub gam_function_weight: f64,
    /// Center parameter $\Gamma_1$ for $f_\Gamma$ (`170.0`).
    pub gam_function_param1: f64,
    /// Width parameter $\Gamma_2$ for $f_\Gamma$ (`20.0`).
    pub gam_function_param2: f64,
    /// Weight $w_X$ on species logarithmic profiles (`20.0`).
    pub xa_function_weight: f64,
    /// Floor $X_0$ for species logarithmic profiles (`1.0e-3`).
    pub xa_function_param: f64,
    /// Minimum allowed zone relative mass fraction $\Delta q_{\min}$ (`1e-12`).
    pub min_dq: f64,
    /// Maximum allowed zone relative mass fraction $\Delta q_{\max}$ (`1e-1`).
    pub max_dq: f64,
    /// Minimum post-split relative mass fraction (`1e-14`).
    pub min_dq_for_split: f64,
    /// Maximum surface cell relative mass fraction (`1.0`).
    pub max_surface_cell_dq: f64,
    /// Maximum center cell relative mass fraction (`1.0`).
    pub max_center_cell_dq: f64,
}

impl Default for MeshFunctionConfig {
    fn default() -> Self {
        Self {
            p_function_weight: 30.0,
            t_function1_weight: 110.0,
            r_function_weight: 0.0,
            r_function_param: 1.0e-4,
            m_function_weight: 0.0,
            m_function_param: 1.0e-2,
            grad_function_weight: 10.0,
            omega_function_weight: 0.0,
            gam_function_weight: 0.0,
            gam_function_param1: 170.0,
            gam_function_param2: 20.0,
            xa_function_weight: 20.0,
            xa_function_param: 1.0e-3,
            min_dq: 1.0e-12,
            max_dq: 1.0e-1,
            min_dq_for_split: 1.0e-14,
            max_surface_cell_dq: 1.0,
            max_center_cell_dq: 1.0,
        }
    }
}

impl MeshFunctionConfig {
    /// Builds [`MeshFunctionConfig`] from [`StarControls`] (`mix_wind`).
    #[must_use]
    pub fn from_controls(c: &StarControls) -> Self {
        let mw = &c.mix_wind;
        let def = Self::default();
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
            .unwrap_or(1.0e-3);
        Self {
            p_function_weight: mw.p_function_weight,
            t_function1_weight: mw.t_function1_weight,
            r_function_weight: mw.r_function_weight,
            r_function_param: def.r_function_param,
            m_function_weight: mw.m_function_weight,
            m_function_param: def.m_function_param,
            grad_function_weight: def.grad_function_weight,
            omega_function_weight: mw.omega_function_weight,
            gam_function_weight: mw.gam_function_weight,
            gam_function_param1: def.gam_function_param1,
            gam_function_param2: def.gam_function_param2,
            xa_function_weight: if max_xa_w > 0.0 { max_xa_w } else { 20.0 },
            xa_function_param: xa_p,
            min_dq: mw.min_dq,
            max_dq: mw.max_dq,
            min_dq_for_split: mw.min_dq_for_split,
            max_surface_cell_dq: mw.max_surface_cell_dq,
            max_center_cell_dq: mw.max_center_cell_dq,
        }
    }
}

/// Evaluates the dimensionless mesh variation indicator $g_k$ across all zones
/// `0..nz` (`eval_mesh_functions` in `mesh_functions.f90` and `mesh_plan.f90`).
#[must_use]
pub fn eval_mesh_gval(model: &StarModel, cfg: &MeshFunctionConfig) -> Vec<f64> {
    let nz = model.nz;
    if nz < 2 {
        return vec![0.0; nz];
    }
    let coeff = model.controls.mesh_delta_coeff.max(0.0);
    let inv_ln10 = 1.0 / LN10;
    let a_r = cfg.r_function_param.max(1.0e-30);
    let a_m = cfg.m_function_param.max(1.0e-30);
    let x0 = cfg.xa_function_param.max(1.0e-30);
    let gam_w = cfg.gam_function_param2.max(1.0);
    let mut gval = vec![0.0; nz];

    for (k, g_k) in gval.iter_mut().enumerate().take(nz - 1) {
        let k1 = k + 1;
        let mut max_df = 0.0_f64;

        if cfg.p_function_weight > 0.0 {
            let df = cfg.p_function_weight
                * (model.ln_peos[k] - model.ln_peos[k1]).abs()
                * inv_ln10;
            max_df = max_df.max(df);
        }
        if cfg.t_function1_weight > 0.0 {
            let df = cfg.t_function1_weight
                * (model.ln_t[k] - model.ln_t[k1]).abs()
                * inv_ln10;
            max_df = max_df.max(df);
        }
        if cfg.r_function_weight > 0.0 {
            let f0 = (1.0 + (model.r[k] / R_SUN) / a_r).log10();
            let f1 = (1.0 + (model.r[k1] / R_SUN) / a_r).log10();
            max_df = max_df.max(cfg.r_function_weight * (f0 - f1).abs());
        }
        if cfg.m_function_weight > 0.0 {
            let f0 = (1.0 + (model.m[k] / M_SUN) / a_m).log10();
            let f1 = (1.0 + (model.m[k1] / M_SUN) / a_m).log10();
            max_df = max_df.max(cfg.m_function_weight * (f0 - f1).abs());
        }
        if cfg.grad_function_weight > 0.0 {
            let f0 = ((model.gradr[k] - model.gradl[k]) / 0.01).tanh();
            let f1 = ((model.gradr[k1] - model.gradl[k1]) / 0.01).tanh();
            max_df = max_df.max(cfg.grad_function_weight * (f0 - f1).abs());
        }
        if cfg.omega_function_weight > 0.0 && model.controls.rotation_flag {
            let f0 = (model.omega[k].abs() + 1.0e-8).log10();
            let f1 = (model.omega[k1].abs() + 1.0e-8).log10();
            max_df = max_df.max(cfg.omega_function_weight * (f0 - f1).abs());
        }
        if cfg.gam_function_weight > 0.0 {
            let f0 = ((model.gam[k] - cfg.gam_function_param1) / gam_w).tanh();
            let f1 = ((model.gam[k1] - cfg.gam_function_param1) / gam_w).tanh();
            max_df = max_df.max(cfg.gam_function_weight * (f0 - f1).abs());
        }
        if cfg.xa_function_weight > 0.0 {
            for j in 0..model.species {
                let x_k = model.xa[k * model.species + j];
                let x_k1 = model.xa[k1 * model.species + j];
                let df = cfg.xa_function_weight
                    * ((x_k + x0).log10() - (x_k1 + x0).log10()).abs();
                max_df = max_df.max(df);
            }
        }

        *g_k = coeff * max_df;
    }
    gval[nz - 1] = gval[nz - 2];
    gval
}

/// Plans cell splits and merges from $g_k$ (`mesh_plan` in `mesh_plan.f90`),
/// returning `(dq_new, num_splits, num_merges)`.
#[must_use]
pub fn plan_mesh(
    model: &StarModel,
    cfg: &MeshFunctionConfig,
) -> (Vec<f64>, usize, usize) {
    let nz = model.nz;
    if nz < 4 {
        return (model.dq.clone(), 0, 0);
    }
    let gval = eval_mesh_gval(model, cfg);
    let min_dq = cfg.min_dq.max(cfg.min_dq_for_split).max(1.0e-15);
    let max_dq = cfg.max_dq.clamp(min_dq * 4.0, 1.0);

    let mut dq_new = Vec::with_capacity(nz + nz / 4);
    let mut num_splits = 0_usize;
    let mut num_merges = 0_usize;
    let mut k = 0_usize;

    while k < nz {
        let dq_k = model.dq[k];
        let g_k = gval[k];
        let surf_over = k == 0 && dq_k > cfg.max_surface_cell_dq;
        let cen_over = k + 1 == nz && dq_k > cfg.max_center_cell_dq;

        let need_split = g_k > 1.0 || dq_k > max_dq || surf_over || cen_over;
        if need_split && dq_k >= 2.0 * min_dq {
            let n_split = (g_k.ceil() as usize).clamp(2, 4);
            let n_eff = if dq_k / (n_split as f64) >= min_dq {
                n_split
            } else {
                2
            };
            let sub_dq = dq_k / (n_eff as f64);
            for _ in 0..n_eff {
                dq_new.push(sub_dq);
            }
            num_splits += n_eff - 1;
            k += 1;
            continue;
        }

        if k >= 2
            && k + 3 < nz
            && model.mixing_type[k] == model.mixing_type[k + 1]
            && g_k + gval[k + 1] < 0.45
            && dq_k + model.dq[k + 1] <= max_dq
        {
            dq_new.push(dq_k + model.dq[k + 1]);
            num_merges += 1;
            k += 2;
            continue;
        }

        dq_new.push(dq_k);
        k += 1;
    }

    let sum: f64 = dq_new.iter().sum();
    if sum > 0.0 {
        for dq_val in &mut dq_new {
            *dq_val /= sum;
        }
    }
    (dq_new, num_splits, num_merges)
}

/// Executes adaptive mesh planning and conservative remapping (`do_adjust_mesh`
/// in `star/private/mesh_adjust.f90`), returning `(num_splits, num_merges)`.
///
/// # Errors
/// Returns [`StarError`] if remapping or microphysics evaluation fails.
pub fn do_adjust_mesh(
    model: &mut StarModel,
    cfg: &MeshFunctionConfig,
) -> Result<(usize, usize), StarError> {
    let (dq_new, num_splits, num_merges) = plan_mesh(model, cfg);
    if num_splits == 0 && num_merges == 0 {
        return Ok((0, 0));
    }
    remap_onto_dq(model, &dq_new)?;
    Ok((num_splits, num_merges))
}
