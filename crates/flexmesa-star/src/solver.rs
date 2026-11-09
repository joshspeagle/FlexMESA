// SPDX-License-Identifier: LGPL-3.0-or-later
//! Block-tridiagonal damped Newton-Raphson Henyey solver (`star_solver.f90`
//! and `solver_support.f90`).
//!
//! Solves the coupled 1D stellar structure (and optional nuclear composition)
//! equations across all `nz` Lagrangian zones simultaneously at each
//! evolutionary timestep $\Delta t$ using [`solve_block_tridiag_bcyclic`]
//! (Hirshman et al. 2010 odd-even cyclic reduction) or sequential block-Thomas
//! elimination ([`solve_block_tridiagonal`]).

use flexmesa_mtx::{solve_block_tridiag_bcyclic, solve_block_tridiagonal};

use crate::controls::{StarError, VarLayout};
use crate::hydro_eqns::{eval_henyey_equations, HenyeySystem};
use crate::model::StarModel;

/// Convergence and diagnostic summary returned by [`solve_henyey_step`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HenyeySolverReport {
    /// Whether both residual and scaled-correction tolerances were satisfied.
    pub converged: bool,
    /// Number of Newton-Raphson iterations executed.
    pub iterations: usize,
    /// Mean absolute equation residual $\langle |\mathrm{equ}| \rangle$.
    pub residual_norm: f64,
    /// Maximum absolute equation residual $\max |\mathrm{equ}|$.
    pub max_residual: f64,
    /// Mean absolute dimensionless correction $\langle |\delta\tilde{x}| \rangle$.
    pub correction_norm: f64,
    /// Maximum absolute dimensionless correction $\max |\delta\tilde{x}|$.
    pub max_correction: f64,
}

/// Computes `(mean_abs, max_abs)` for a residual vector (`sizequ` in
/// `star/private/solver_support.f90:230-313`).
#[must_use]
pub fn compute_vector_norms(v: &[f64]) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let mut sum_abs = 0.0;
    let mut max_abs = 0.0_f64;
    for &x in v {
        let a = x.abs();
        sum_abs += a;
        if a > max_abs || a.is_nan() {
            max_abs = a;
        }
    }
    (sum_abs / (v.len() as f64), max_abs)
}

/// Computes `(correction_norm, max_correction)` skipping `i_lum` by default
/// (`sizeB` in `star/private/solver_support.f90:316-570`).
#[must_use]
pub fn compute_correction_norms(
    layout: &VarLayout,
    nz: usize,
    nvar: usize,
    dx_scaled: &[f64],
) -> (f64, f64) {
    if nz == 0 || nvar == 0 || dx_scaled.is_empty() {
        return (0.0, 0.0);
    }
    let mut sum_abs = 0.0;
    let mut max_abs = 0.0_f64;
    let mut count = 0_usize;
    for k in 0..nz {
        let base = k * nvar;
        for j in 0..nvar {
            if j == layout.i_lum {
                continue;
            }
            let a = dx_scaled[base + j].abs();
            sum_abs += a;
            count += 1;
            if a > max_abs || a.is_nan() {
                max_abs = a;
            }
        }
    }
    if count == 0 {
        (0.0, 0.0)
    } else {
        (sum_abs / (count as f64), max_abs)
    }
}

/// Computes the step damping factor `corr_coeff` $\in (0, 1]$ limiting the
/// maximum per-iteration change in $\ln\rho, \ln T, \ln R$, surface $L_0 > 0$,
/// and $X_{j,k}$ (`Bdomain` and `adjust_correction` in
/// `star/private/solver_support.f90:573-950`).
#[must_use]
pub fn compute_corr_coeff(
    model: &StarModel,
    nz: usize,
    nvar: usize,
    dx_scaled: &[f64],
    x_scale: &[f64],
) -> f64 {
    let layout = &model.layout;
    let max_dln = 0.5_f64;
    let max_dx_chem = 0.25_f64;
    let mut coeff = 1.0_f64;

    if nz > 0 {
        let old_lum_surf = model.lum[0];
        let dlum_surf = dx_scaled[layout.i_lum] * x_scale[layout.i_lum];
        if old_lum_surf > 0.0 && old_lum_surf + dlum_surf < 0.1 * old_lum_surf {
            coeff = coeff.min(-0.9 * old_lum_surf / dlum_surf);
        }
    }

    for k in 0..nz {
        let base = k * nvar;
        for &ivar in &[layout.i_lnd, layout.i_lnt, layout.i_lnr] {
            let dval = (dx_scaled[base + ivar] * x_scale[base + ivar]).abs();
            if dval > max_dln {
                coeff = coeff.min(max_dln / dval);
            }
        }
        if nvar > layout.nvar_hydro {
            let c_base = k * layout.species;
            for j in 0..layout.species {
                let ivar = layout.chem_var_index(j);
                let dx_j = dx_scaled[base + ivar] * x_scale[base + ivar];
                let dxa = dx_j.abs();
                if dxa > max_dx_chem {
                    coeff = coeff.min(max_dx_chem / dxa);
                }
                let x_old = model.xa[c_base + j];
                if x_old > 1.0e-12 && x_old + dx_j < 0.05 * x_old {
                    coeff = coeff.min(-0.95 * x_old / dx_j);
                }
            }
        }
    }
    coeff.clamp(1.0e-12, 1.0)
}

/// Applies a damped scaled correction
/// $x_{k,j} \leftarrow x_{\mathrm{base},k,j} + \alpha\,\delta\tilde{x}_{k,j}\,\mathrm{x\_scale}_{k,j}$
/// and synchronizes `xh`, `xa`, `r`, `rmid`, and `grav`.
///
/// During Newton iterations, `xa` updates remain linear (clamped to `[0, 1]`
/// without per-step `/= sum_x` projection so the tridiagonal diffusion
/// operator remains exact); exact mass-fraction normalization is applied once
/// after Newton convergence.
fn apply_damped_correction(
    model: &mut StarModel,
    xh_base: &[f64],
    xa_base: &[f64],
    dx_scaled: &[f64],
    x_scale: &[f64],
    alpha: f64,
    include_chem: bool,
) {
    let nz = model.nz;
    let nvar_hydro = model.layout.nvar_hydro;
    let nvar = model.layout.nvar(include_chem);

    for k in 0..nz {
        let h_base = k * nvar_hydro;
        let s_base = k * nvar;
        for i in 0..nvar_hydro {
            model.xh[h_base + i] = xh_base[h_base + i]
                + alpha * dx_scaled[s_base + i] * x_scale[s_base + i];
        }
        if include_chem && model.species > 0 {
            let c_base = k * model.species;
            for j in 0..model.species {
                let ivar = model.layout.chem_var_index(j);
                let x_new = (xa_base[c_base + j]
                    + alpha
                        * dx_scaled[s_base + ivar]
                        * x_scale[s_base + ivar])
                    .clamp(0.0, 1.0);
                model.xa[c_base + j] = x_new;
            }
        }
    }
    model.unpack_xh();
    model.set_m_grav_and_grav();
}

/// Normalizes cell species mass fractions `xa` so $\sum_{j=0}^{\mathrm{species}-1} X_{j,k} = 1$
/// in every zone `k`.
fn normalize_species_fractions(model: &mut StarModel) {
    if model.species == 0 {
        return;
    }
    for k in 0..model.nz {
        let c_base = k * model.species;
        let mut sum_x = 0.0;
        for j in 0..model.species {
            let x_clamped = model.xa[c_base + j].clamp(0.0, 1.0);
            model.xa[c_base + j] = x_clamped;
            sum_x += x_clamped;
        }
        if sum_x > 0.0 {
            for j in 0..model.species {
                model.xa[c_base + j] /= sum_x;
            }
        }
    }
}

/// Solves $J\,\delta\tilde{x} = -\mathrm{equ}$ using `bcyclic` or block-Thomas.
fn solve_linear_henyey_system(
    sys: &HenyeySystem,
    use_bcyclic: bool,
) -> Result<Vec<f64>, StarError> {
    let rhs: Vec<f64> = sys.equ.iter().map(|&r| -r).collect();
    let res = if use_bcyclic && sys.nz >= 2 {
        solve_block_tridiag_bcyclic(
            sys.nvar, sys.nz, &sys.lblk, &sys.dblk, &sys.ublk, &rhs, true,
        )
    } else {
        solve_block_tridiagonal(
            sys.nvar, sys.nz, &sys.lblk, &sys.dblk, &sys.ublk, &rhs,
        )
    };
    res.map_err(|e| {
        StarError::ConvergenceFailure(format!(
            "block-tridiagonal linear solve failed: {e}"
        ))
    })
}

/// Executes a damped Newton-Raphson Henyey solve for the current timestep
/// `model.dt` (`do_star_solve` in `star/private/star_solver.f90`).
///
/// # Arguments
/// * `model` - Mutable [`StarModel`] state with `*_start` arrays already
///   populated for the current step.
/// * `include_chem` - When `true`, solves the fully coupled `(nvar_hydro + species)`
///   system; when `false`, solves the `nvar_hydro` hydrostatic/hydrodynamic
///   equations at fixed composition `xa`.
/// * `use_bcyclic` - When `true`, uses Hirshman et al. (2010) odd-even parallel
///   cyclic reduction ([`solve_block_tridiag_bcyclic`]); when `false`, uses
///   sequential block-Thomas elimination ([`solve_block_tridiagonal`]).
///
/// # Errors
/// Returns [`StarError`] if initial microphysics fails or the Jacobian is
/// singular.
pub fn solve_henyey_step(
    model: &mut StarModel,
    include_chem: bool,
    use_bcyclic: bool,
) -> Result<HenyeySolverReport, StarError> {
    model.pack_xh();
    let mut sys = eval_henyey_equations(model, include_chem)?;
    let (mut residual_norm, mut max_residual) = compute_vector_norms(&sys.equ);
    let mut correction_norm = 0.0;
    let mut max_correction = 0.0;

    let max_iters = model.controls.max_newton_iters.max(1);
    let mut converged = false;
    let mut iter_used = 0;
    let mut xh_prev = Vec::with_capacity(model.xh.len());
    let mut xa_prev = Vec::with_capacity(model.xa.len());

    for iter in 1..=max_iters {
        iter_used = iter;
        let dx_scaled = solve_linear_henyey_system(&sys, use_bcyclic)?;
        (correction_norm, max_correction) = compute_correction_norms(
            &model.layout,
            sys.nz,
            sys.nvar,
            &dx_scaled,
        );

        let base_coeff = compute_corr_coeff(
            model,
            sys.nz,
            sys.nvar,
            &dx_scaled,
            &sys.x_scale,
        );
        xh_prev.clone_from(&model.xh);
        xa_prev.clone_from(&model.xa);
        let prev_residual_norm = residual_norm;

        let mut alpha = base_coeff;
        let mut step_accepted = false;
        let mut best_rnorm = f64::INFINITY;
        let mut best_sys: Option<(f64, f64, f64, HenyeySystem)> = None;
        for _bt in 0..8 {
            apply_damped_correction(
                model,
                &xh_prev,
                &xa_prev,
                &dx_scaled,
                &sys.x_scale,
                alpha,
                include_chem,
            );
            if let Ok(next_sys) = eval_henyey_equations(model, include_chem) {
                let (next_rnorm, next_rmax) =
                    compute_vector_norms(&next_sys.equ);
                if next_rnorm.is_finite() && next_rmax.is_finite() {
                    if iter == 1
                        || next_rnorm <= 1.5 * prev_residual_norm.max(1.0e-4)
                    {
                        sys = next_sys;
                        residual_norm = next_rnorm;
                        max_residual = next_rmax;
                        step_accepted = true;
                        break;
                    }
                    if next_rnorm < best_rnorm {
                        best_rnorm = next_rnorm;
                        best_sys =
                            Some((alpha, next_rnorm, next_rmax, next_sys));
                    }
                }
            }
            alpha *= 0.5;
        }

        if !step_accepted {
            if let Some((best_alpha, best_rnorm, best_rmax, saved_sys)) =
                best_sys
            {
                if best_alpha > 0.0
                    && best_rnorm < 1.0e6
                    && (best_rnorm <= prev_residual_norm * 2.0
                        || prev_residual_norm > 1.0)
                {
                    apply_damped_correction(
                        model,
                        &xh_prev,
                        &xa_prev,
                        &dx_scaled,
                        &sys.x_scale,
                        best_alpha,
                        include_chem,
                    );
                    sys = saved_sys;
                    residual_norm = best_rnorm;
                    max_residual = best_rmax;
                    step_accepted = true;
                }
            }
        }

        if !step_accepted {
            model.xh.clone_from(&xh_prev);
            model.xa.clone_from(&xa_prev);
            model.unpack_xh();
            model.set_m_grav_and_grav();
            return Err(StarError::ConvergenceFailure(format!(
                "line search failed at Newton iteration {iter}"
            )));
        }

        model.total_num_solver_iterations += 1;

        let corr_ok = correction_norm <= model.controls.tol_correction_norm
            && max_correction <= model.controls.tol_max_correction;
        let resid_ok = residual_norm <= model.controls.tol_residual_norm
            && max_residual <= model.controls.tol_max_residual;
        let tight_resid = residual_norm <= 1.0e-10 && max_residual <= 1.0e-8;
        let mean_norm_ok = iter >= 5
            && residual_norm <= model.controls.tol_residual_norm
            && correction_norm <= model.controls.tol_correction_norm
            && max_residual <= 0.1
            && max_correction <= 0.05;

        if (corr_ok && resid_ok) || tight_resid || mean_norm_ok {
            converged = true;
            break;
        }
    }

    if include_chem {
        normalize_species_fractions(model);
    }

    Ok(HenyeySolverReport {
        converged,
        iterations: iter_used,
        residual_norm,
        max_residual,
        correction_norm,
        max_correction,
    })
}
