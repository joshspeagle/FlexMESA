// SPDX-License-Identifier: LGPL-3.0-or-later
//! Adaptive Lagrangian mesh spacing functions, cell split/merge planning, and
//! conservative abundance/energy remapping (`mesh_functions.f90`,
//! `mesh_plan.f90`, `adjust_mesh_split_merge.f90`, and `mesh_adjust.f90`).
//!
//! # Mesh Spacing Functions
//! Following Paxton et al. (2011, §6.1), the mesh spacing vector
//! $\mathbf{f}_k = (f_{1,k}, \dots, f_{M,k})$ includes weighted logarithmic
//! thermodynamic, geometric, and composition profiles:
//! - Pressure: $f_{P,k} = w_P \log_{10} P_{\mathrm{eos},k}$
//! - Temperature: $f_{T1,k} = w_{T1} \log_{10} T_k$
//! - Radius: $f_{R,k} = w_R \log_{10}(1 + (r_k / R_\odot) / \alpha_R)$
//! - Mass: $f_{M,k} = w_M \log_{10}(1 + (m_k / M_\odot) / \alpha_M)$
//! - Convective gradient transition:
//!   $f_{\nabla,k} = w_\nabla \tanh((\nabla_{\mathrm{rad},k} - \nabla_{L,k}) / 0.01)$
//! - Species abundances: $f_{X_j,k} = w_X \log_{10}(X_{j,k} + X_0)$
//!
//! The dimensionless zone variation indicator is
//! $$g_k = C_{\mathrm{mesh}} \max_{m} |f_{m,k} - f_{m,\min(k+1,\mathrm{nz}-1)}|,$$
//! where $C_{\mathrm{mesh}} = \text{mesh\_delta\_coeff}$. Zones with $g_k > 1$
//! and $\Delta q_k \ge 2 \Delta q_{\min}$ are split into
//! $n = \min(\lceil g_k \rceil, 4)$ sub-zones, while adjacent zones $(k, k+1)$
//! sharing the same mixing classification with $g_k + g_{k+1} < 0.45$ and
//! $\Delta q_k + \Delta q_{k+1} \le \Delta q_{\max}$ are merged.

use flexmesa_const::{FOUR_PI, LN10, M_SUN, R_SUN};

use crate::controls::StarError;
use crate::hydro_vars::set_hydro_vars;
use crate::model::StarModel;
use crate::zams::interp_q;

/// Weights and scale parameters for the adaptive mesh spacing functions
/// (`mesh_functions.f90`).
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
    /// Weight $w_\nabla$ on convective boundary transition (`0.0`).
    pub grad_function_weight: f64,
    /// Weight $w_X$ on species logarithmic profiles (`0.0`).
    pub xa_function_weight: f64,
    /// Floor $X_0$ for species logarithmic profiles (`1.0e-3`).
    pub xa_function_param: f64,
    /// Minimum allowed zone relative mass fraction $\Delta q_{\min}$ (`1e-12`).
    pub min_dq: f64,
    /// Maximum allowed zone relative mass fraction $\Delta q_{\max}$ (`1e-1`).
    pub max_dq: f64,
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
            grad_function_weight: 0.0,
            xa_function_weight: 0.0,
            xa_function_param: 1.0e-3,
            min_dq: 1.0e-12,
            max_dq: 1.0e-1,
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
    let min_dq = cfg.min_dq.max(1.0e-15);
    let max_dq = cfg.max_dq.clamp(min_dq * 4.0, 1.0);

    let mut dq_new = Vec::with_capacity(nz + nz / 4);
    let mut num_splits = 0_usize;
    let mut num_merges = 0_usize;
    let mut k = 0_usize;

    while k < nz {
        let dq_k = model.dq[k];
        let g_k = gval[k];

        // 1. Check whether cell `k` should be split.
        let need_split = g_k > 1.0 || dq_k > max_dq;
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

        // 2. Check whether adjacent cells `(k, k + 1)` should be merged.
        // Do not merge the outermost 2 surface cells or innermost 2 center
        // cells, and never merge across a convective/mixing boundary.
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

/// Conservatively projects `model` onto a new mass grid `dq_new` (`mesh_adjust`
/// in `star/private/mesh_adjust.f90`).
///
/// Species mass fractions `xa` and specific internal energy `energy` are
/// projected via exact piecewise-constant overlap integration in exterior mass
/// coordinate `xq` so $\sum_k \Delta m_k X_{j,k}$ is conserved to machine
/// precision ($10^{-15}$).
///
/// # Errors
/// Returns [`StarError`] if spline interpolation or microphysics fails.
pub fn remap_onto_dq(
    model: &mut StarModel,
    dq_new: &[f64],
) -> Result<(), StarError> {
    let nz_old = model.nz;
    let nz_new = dq_new.len();
    let species = model.species;
    if nz_new < 2 {
        return Err(StarError::InvalidInput(String::from(
            "dq_new must have at least 2 zones",
        )));
    }

    let dq_old = model.dq.clone();
    let xq_old = model.xq.clone();
    let xa_old = model.xa.clone();
    let e_old = model.energy.clone();

    let lnd_new = interp_q(&dq_old, &model.lnd, dq_new)?;
    let lnt_new = interp_q(&dq_old, &model.ln_t, dq_new)?;
    let lum_new = interp_q(&dq_old, &model.lum, dq_new)?;
    let v_new = interp_q(&dq_old, &model.v, dq_new)?;
    let u_new = interp_q(&dq_old, &model.u, dq_new)?;
    let omega_new = interp_q(&dq_old, &model.omega, dq_new)?;

    model.resize_zones(nz_new);
    model.dq.copy_from_slice(dq_new);
    model.normalize_dqs()?;
    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();

    model.lnd.copy_from_slice(&lnd_new);
    model.ln_t.copy_from_slice(&lnt_new);
    model.lum.copy_from_slice(&lum_new);
    model.v.copy_from_slice(&v_new);
    model.u.copy_from_slice(&u_new);
    model.omega.copy_from_slice(&omega_new);

    // O(nz) two-pointer conservative overlap projection of `xa` and `energy`.
    let mut accum = vec![0.0; species];
    let mut kk_start = 0_usize;
    for k in 0..nz_new {
        let a0 = model.xq[k];
        let a1 = if k + 1 < nz_new { model.xq[k + 1] } else { 1.0 };
        let dx = (a1 - a0).max(1.0e-99);
        accum.fill(0.0);
        let mut e_acc = 0.0;

        while kk_start + 1 < nz_old && xq_old[kk_start + 1] <= a0 {
            kk_start += 1;
        }
        for kk in kk_start..nz_old {
            let b0 = xq_old[kk];
            if b0 >= a1 {
                break;
            }
            let b1 = if kk + 1 < nz_old { xq_old[kk + 1] } else { 1.0 };
            let ov = (a1.min(b1) - a0.max(b0)).max(0.0);
            if ov > 0.0 {
                for j in 0..species {
                    accum[j] += ov * xa_old[kk * species + j];
                }
                e_acc += ov * e_old[kk];
            }
        }
        let mut sum_x = 0.0;
        for j in 0..species {
            let xj = (accum[j] / dx).clamp(0.0, 1.0);
            model.xa[k * species + j] = xj;
            sum_x += xj;
        }
        if sum_x > 0.0 {
            for j in 0..species {
                model.xa[k * species + j] /= sum_x;
            }
        }
        model.energy[k] = e_acc / dx;
    }

    // Reconstruct face radii `ln_r` from cell densities `lnd` and masses `dm`
    // so the continuity equation `dr^3 = 3 dm / (4 pi rho)` holds identically.
    let mut r3 = model.r_center.powi(3);
    for k in (0..nz_new).rev() {
        let rho_k = model.lnd[k].exp().max(1.0e-30);
        r3 += 3.0 * model.dm[k] / (FOUR_PI * rho_k);
        model.ln_r[k] = r3.max(1.0).cbrt().ln();
    }

    let (age_old, dt_old, mstar_old) =
        (model.star_age_old, model.dt_old, model.mstar_old);
    model.set_m_grav_and_grav();
    model.pack_xh();
    set_hydro_vars(model, false, false)?;
    model.save_start_values();
    model.save_old_values();
    model.star_age_old = age_old;
    model.dt_old = dt_old;
    model.mstar_old = mstar_old;
    Ok(())
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
