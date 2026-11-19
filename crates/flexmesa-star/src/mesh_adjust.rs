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
    /// Minimum cell $\Delta q$ for composition-driven splitting (`min_dq_for_xa`, `1e-14`).
    pub min_dq_for_xa: f64,
    /// Minimum cell $\Delta q$ for composition-driven splitting at convective zones (`1e-6`).
    pub min_dq_for_xa_convective: f64,
    /// Minimum $\Delta\ln R$ across a split subcell (`mesh_min_dlnR`, `1e-9`).
    pub mesh_min_dlnr: f64,
    /// Maximum allowed adjacent cell $\Delta q$ ratio (`mesh_max_allowed_ratio`, `2.5`).
    pub mesh_max_allowed_ratio: f64,
    /// Maximum species difference allowed when merging cells (`max_delta_x_for_merge`, `0.1`).
    pub max_delta_x_for_merge: f64,
    /// Maximum surface cell relative mass fraction (`1.0`).
    pub max_surface_cell_dq: f64,
    /// Maximum center cell relative mass fraction (`1.0`).
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
            min_dq: 1.0e-12,
            max_dq: 1.0e-1,
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
            .unwrap_or(1.0e-2);
        Self {
            p_function_weight: mw.p_function_weight,
            t_function1_weight: mw.t_function1_weight,
            r_function_weight: mw.r_function_weight,
            r_function_param: def.r_function_param,
            m_function_weight: mw.m_function_weight,
            m_function_param: mw.m_function_param,
            grad_function_weight: mw.gradt_function_weight,
            omega_function_weight: mw.omega_function_weight,
            gam_function_weight: mw.gam_function_weight,
            gam_function_param1: def.gam_function_param1,
            gam_function_param2: def.gam_function_param2,
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

fn eval_mesh_gval_parts(
    model: &StarModel,
    cfg: &MeshFunctionConfig,
) -> (Vec<f64>, Vec<f64>) {
    let nz = model.nz;
    if nz < 2 {
        return (vec![0.0; nz], vec![0.0; nz]);
    }
    let coeff = if model.controls.mesh_delta_coeff > 0.0 {
        model.controls.mesh_delta_coeff
    } else {
        1.0
    };
    let inv_ln10 = 1.0 / LN10;
    let a_r = cfg.r_function_param.max(1.0e-30);
    let a_m = cfg.m_function_param.max(1.0e-30);
    let x0 = cfg.xa_function_param.max(1.0e-30);
    let gam_w = cfg.gam_function_param2.max(1.0);
    let per_species_xa: Option<Vec<(usize, f64, f64)>> = if *cfg
        == MeshFunctionConfig::from_controls(&model.controls)
        && !model.controls.mix_wind.xa_function_species.is_empty()
    {
        let mw = &model.controls.mix_wind;
        let items = mw
            .xa_function_species
            .iter()
            .enumerate()
            .filter_map(|(i, sp)| {
                let name = sp.trim();
                if name.is_empty() {
                    return None;
                }
                let w = mw.xa_function_weight.get(i).copied().unwrap_or(0.0);
                if w <= 0.0 {
                    return None;
                }
                let j = model
                    .iso_names
                    .iter()
                    .position(|n| n.eq_ignore_ascii_case(name))?;
                let p = mw
                    .xa_function_param
                    .get(i)
                    .copied()
                    .filter(|&v| v > 0.0)
                    .unwrap_or(1.0e-2)
                    .max(1.0e-30);
                Some((j, w, p))
            })
            .collect();
        Some(items)
    } else {
        None
    };
    let mut g_struct = vec![0.0; nz];
    let mut g_xa = vec![0.0; nz];

    for k in 0..(nz - 1) {
        let k1 = k + 1;
        let mut max_struct = 0.0_f64;
        let mut max_xa = 0.0_f64;

        if cfg.p_function_weight > 0.0 {
            let df = cfg.p_function_weight
                * (model.ln_peos[k] - model.ln_peos[k1]).abs()
                * inv_ln10;
            max_struct = max_struct.max(df);
        }
        if cfg.t_function1_weight > 0.0 {
            let df = cfg.t_function1_weight
                * (model.ln_t[k] - model.ln_t[k1]).abs()
                * inv_ln10;
            max_struct = max_struct.max(df);
        }
        if cfg.r_function_weight > 0.0 {
            let f0 = flexmesa_math::log10(1.0 + (model.r[k] / R_SUN) / a_r);
            let f1 = flexmesa_math::log10(1.0 + (model.r[k1] / R_SUN) / a_r);
            max_struct =
                max_struct.max(cfg.r_function_weight * (f0 - f1).abs());
        }
        if cfg.m_function_weight > 0.0 {
            let f0 = flexmesa_math::log10(1.0 + (model.m[k] / M_SUN) / a_m);
            let f1 = flexmesa_math::log10(1.0 + (model.m[k1] / M_SUN) / a_m);
            max_struct =
                max_struct.max(cfg.m_function_weight * (f0 - f1).abs());
        }
        if cfg.grad_function_weight > 0.0 {
            max_struct = max_struct.max(
                cfg.grad_function_weight
                    * (model.gradt[k] - model.gradt[k1]).abs(),
            );
        }
        if cfg.omega_function_weight > 0.0 && model.controls.rotation_flag {
            let f0 = flexmesa_math::log10(model.omega[k].abs() + 1.0e-8);
            let f1 = flexmesa_math::log10(model.omega[k1].abs() + 1.0e-8);
            max_struct =
                max_struct.max(cfg.omega_function_weight * (f0 - f1).abs());
        }
        if cfg.gam_function_weight > 0.0 {
            let f0 = flexmesa_math::tanh(
                (model.gam[k] - cfg.gam_function_param1) / gam_w,
            );
            let f1 = flexmesa_math::tanh(
                (model.gam[k1] - cfg.gam_function_param1) / gam_w,
            );
            max_struct =
                max_struct.max(cfg.gam_function_weight * (f0 - f1).abs());
        }
        if let Some(specs) = per_species_xa.as_deref() {
            for &(j, w, x0_sp) in specs {
                let x_k = model.xa[k * model.species + j];
                let x_k1 = model.xa[k1 * model.species + j];
                let df = w
                    * (flexmesa_math::log10(x_k + x0_sp)
                        - flexmesa_math::log10(x_k1 + x0_sp))
                    .abs();
                max_xa = max_xa.max(df);
            }
        } else if cfg.xa_function_weight > 0.0 {
            for j in 0..model.species {
                let x_k = model.xa[k * model.species + j];
                let x_k1 = model.xa[k1 * model.species + j];
                let df = cfg.xa_function_weight
                    * (flexmesa_math::log10(x_k + x0)
                        - flexmesa_math::log10(x_k1 + x0))
                    .abs();
                max_xa = max_xa.max(df);
            }
        }

        g_struct[k] = max_struct / coeff;
        g_xa[k] = max_xa / coeff;
    }
    g_struct[nz - 1] = g_struct[nz - 2];
    g_xa[nz - 1] = g_xa[nz - 2];
    (g_struct, g_xa)
}

/// Evaluates the dimensionless mesh variation indicator $g_k$ across all zones
/// `0..nz` (`eval_mesh_functions` in `mesh_functions.f90` and `mesh_plan.f90`).
#[must_use]
pub fn eval_mesh_gval(model: &StarModel, cfg: &MeshFunctionConfig) -> Vec<f64> {
    let (g_non_xa, g_xa) = eval_mesh_gval_parts(model, cfg);
    g_non_xa
        .into_iter()
        .zip(g_xa)
        .map(|(a, b)| a.max(b))
        .collect()
}

fn subcell_dlnr_ok(
    model: &StarModel,
    k: usize,
    dq_sub: f64,
    min_dlnr: f64,
) -> bool {
    if min_dlnr <= 0.0 || k + 1 >= model.nz {
        return true;
    }
    let dq_old = model.dq[k].max(1.0e-99);
    let r_l = model.r[k + 1].max(1.0);
    let dr_old = (model.r[k] - r_l).max(0.0);
    let dlnr_new = flexmesa_math::log((r_l + dr_old * (dq_sub / dq_old)) / r_l);
    dlnr_new >= 2.0 * min_dlnr
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
    let (g_non_xa, g_xa) = eval_mesh_gval_parts(model, cfg);
    let gval: Vec<f64> = g_non_xa
        .iter()
        .zip(&g_xa)
        .map(|(&a, &b)| a.max(b))
        .collect();
    let d_coeff = if model.controls.mesh_delta_coeff > 0.0 {
        model.controls.mesh_delta_coeff
    } else {
        1.0
    };
    let min_dq = cfg.min_dq.max(cfg.min_dq_for_split).max(1.0e-15);
    let max_dq = (cfg.max_dq * d_coeff).clamp(min_dq * 4.0, 1.0);
    let max_surf_dq = (cfg.max_surface_cell_dq * d_coeff).min(1.0);
    let max_cen_dq = (cfg.max_center_cell_dq * d_coeff).min(1.0);

    let mut dq_new = Vec::with_capacity(nz + nz / 4);
    let mut num_splits = 0_usize;
    let mut num_merges = 0_usize;
    let mut k = 0_usize;

    while k < nz {
        let dq_k = model.dq[k];
        let g_k = gval[k];
        let surf_over = k == 0 && dq_k > max_surf_dq;
        let cen_over = k + 1 == nz && dq_k > max_cen_dq;
        let non_xa_split =
            g_non_xa[k] > 1.0 || dq_k > max_dq || surf_over || cen_over;
        let xa_only_split = !non_xa_split && g_xa[k] > 1.0;

        let eff_min_dq = if xa_only_split {
            let xa_floor = if model.gradr[k] > model.grada[k]
                && cfg.min_dq_for_xa_convective > 0.0
            {
                cfg.min_dq_for_xa_convective
            } else {
                cfg.min_dq_for_xa
            };
            min_dq.max(xa_floor)
        } else {
            min_dq
        };

        if (non_xa_split || xa_only_split) && dq_k >= 2.0 * eff_min_dq {
            let mut n_eff = (g_k.ceil() as usize).clamp(2, 4);
            while n_eff > 2 && dq_k / (n_eff as f64) < eff_min_dq {
                n_eff -= 1;
            }
            while n_eff >= 2
                && !subcell_dlnr_ok(
                    model,
                    k,
                    dq_k / (n_eff as f64),
                    cfg.mesh_min_dlnr,
                )
            {
                n_eff -= 1;
            }
            if n_eff >= 2 && dq_k / (n_eff as f64) >= eff_min_dq {
                let sub_dq = dq_k / (n_eff as f64);
                for _ in 0..n_eff {
                    dq_new.push(sub_dq);
                }
                num_splits += n_eff - 1;
                k += 1;
                continue;
            }
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
