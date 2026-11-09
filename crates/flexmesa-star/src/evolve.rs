// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stellar evolution step driver (`star_evolve_step`), automatic retry/backup
//! cascade on solver non-convergence, and multi-step evolutionary loop
//! (`star_evolve`) corresponding to `star/private/evolve.f90`.
//!
//! # Evolutionary Step Sequence (`evolve.f90:155-560`)
//! Each evolutionary step $n \to n+1$ executes:
//! 1. **Snapshot `old` state**: Saves $(\ln\rho, \ln T, \ln R, L, X, \Delta q, M_*)$
//!    into `*_old` so the step can be rolled back if Newton-Raphson fails and
//!    so `varcontrol` can measure the structural change over the step.
//! 2. **Adaptive Remeshing (`do_adjust_mesh`)**: Evaluates mesh spacing
//!    functions $g_k$ and splits/merges zones when `do_remesh` is enabled.
//! 3. **Stellar Winds & Mass Adjustment (`set_mdot` + `do_adjust_mass`)**:
//!    Applies empirical mass loss $\Delta M = \dot{M}\Delta t$ across the
//!    3-region pseudo-Lagrangian grid when wind scaling factors are active.
//! 4. **Structure + Burn + Mix Solve with Automatic Retry Cascade**:
//!    Executes `do_struct_burn_mix(model, dt_try, ...)`. If convergence fails,
//!    restores the pre-step state from `*_old`, halves
//!    $\Delta t_{\mathrm{try}} \leftarrow 0.5\,\Delta t_{\mathrm{try}}$,
//!    increments `model.num_retries`, and retries up to `max_retries`.
//! 5. **Bookkeeping & Timestep Control (`pick_next_timestep`)**:
//!    Increments `model.model_number`, advances `model.star_age` by
//!    $\Delta t / \mathrm{SECYER}$, and computes $\Delta t_{\mathrm{next}}$ via
//!    the Söderlind & Wang (2006) H211 digital filter.

use flexmesa_const::SECYER;

use crate::controls::StarError;
use crate::hydro_vars::set_hydro_vars;
use crate::mesh_adjust::{do_adjust_mesh, MeshFunctionConfig};
use crate::model::StarModel;
use crate::struct_burn_mix::do_struct_burn_mix;
use crate::timestep::pick_next_timestep;
use crate::winds::{do_adjust_mass, set_mdot, DutchLowTScheme, WindScheme};

/// Diagnostic report returned by a single evolutionary step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvolveStepReport {
    /// Evolutionary step number after completion (`model_number`).
    pub model_number: usize,
    /// Stellar age in years after completion (`star_age`).
    pub star_age: f64,
    /// Timestep $\Delta t$ actually taken in seconds (`s`).
    pub dt: f64,
    /// Proposed next timestep $\Delta t_{\mathrm{next}}$ in seconds (`s`).
    pub dt_next: f64,
    /// Number of Henyey Newton-Raphson iterations on the successful attempt.
    pub iterations: usize,
    /// Number of step retries (`dt` halvings) required during this step.
    pub retries: usize,
    /// Number of zone splits performed during remeshing.
    pub num_splits: usize,
    /// Number of zone merges performed during remeshing.
    pub num_merges: usize,
    /// Measured normalized structural variation `varcontrol`.
    pub varcontrol: f64,
    /// Whether the step converged within tolerances.
    pub converged: bool,
}

/// Restores `model` to its pre-step `*_old` snapshot after a failed solver
/// attempt (`do_restore` / `prepare_for_retry` in `evolve.f90:1012-1120`).
///
/// # Errors
/// Returns [`StarError`] if mesh synchronization or microphysics fails.
pub fn restore_from_old(model: &mut StarModel) -> Result<(), StarError> {
    let nz_old = model.dq_old.len();
    if nz_old != model.nz && nz_old >= 2 {
        model.resize_zones(nz_old);
    }
    model.star_age = model.star_age_old;
    model.mstar = model.mstar_old;
    model.xmstar = model.mstar - model.m_center;
    model.lnd.clone_from(&model.lnd_old);
    model.ln_t.clone_from(&model.ln_t_old);
    model.ln_r.clone_from(&model.ln_r_old);
    model.lum.clone_from(&model.lum_old);
    model.xa.clone_from(&model.xa_old);
    model.dq.clone_from(&model.dq_old);
    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();
    model.set_m_grav_and_grav();
    model.pack_xh();
    set_hydro_vars(model, false, false)?;
    model.save_start_values();
    Ok(())
}

fn active_wind(model: &StarModel) -> (WindScheme, f64) {
    if model.controls.dutch_scaling_factor > 0.0 {
        (WindScheme::Dutch, model.controls.dutch_scaling_factor)
    } else if model.controls.vink_scaling_factor > 0.0 {
        (WindScheme::Vink, model.controls.vink_scaling_factor)
    } else if model.controls.reimers_scaling_factor > 0.0 {
        (WindScheme::Reimers, model.controls.reimers_scaling_factor)
    } else if model.controls.blocker_scaling_factor > 0.0 {
        (WindScheme::Blocker, model.controls.blocker_scaling_factor)
    } else {
        (WindScheme::None, 0.0)
    }
}

/// Executes a single evolutionary step (`star_evolve_step` in `evolve.f90`):
/// remeshing $\to$ stellar wind mass adjustment $\to$ structure/burn/mix solve
/// with automatic $\Delta t$ retry halving $\to$ H211 timestep control.
///
/// # Errors
/// Returns [`StarError::ConvergenceFailure`] if all retry attempts fail.
pub fn star_evolve_step(
    model: &mut StarModel,
    do_remesh: bool,
    op_split_burn: bool,
    use_bcyclic: bool,
) -> Result<EvolveStepReport, StarError> {
    const MAX_RETRIES: usize = 16;

    if model.dt_next > 0.0 {
        model.dt = model.dt_next;
    }
    let prev_dt_old = model.dt_old;

    let (num_splits, num_merges) = if do_remesh {
        set_hydro_vars(model, false, false)?;
        let dt_curr = model.dt.max(1.0);
        crate::struct_burn_mix::refresh_mixing_state(model, dt_curr, &[]);
        do_adjust_mesh(model, &MeshFunctionConfig::default())?
    } else {
        (0, 0)
    };

    // Snapshot post-remesh state into `*_old` so `varcontrol` and
    // `restore_from_old` operate on the current `nz` zone grid, preserving the
    // previous step's `dt_old` for the H211 filter.
    model.save_old_values();
    model.dt_old = prev_dt_old;

    let dt_floor = model.controls.min_timestep_limit.clamp(1.0e-20, 1.0);
    let mut dt_try = model.dt.max(dt_floor);
    let mut retries = 0_usize;
    let (wind_scheme, wind_eta) = active_wind(model);

    loop {
        model.dt = dt_try;
        let mass_ok = if wind_scheme != WindScheme::None && wind_eta > 0.0 {
            let mdot = set_mdot(
                model,
                wind_scheme,
                DutchLowTScheme::DeJager,
                wind_eta,
            );
            if mdot == 0.0 {
                Ok(())
            } else {
                do_adjust_mass(model, mdot * dt_try, None, 0.99, 0.95)
            }
        } else {
            Ok(())
        };

        let attempt = mass_ok.and_then(|()| {
            do_struct_burn_mix(model, dt_try, op_split_burn, use_bcyclic, &[])
        });
        match attempt {
            Ok(rep) if rep.converged => {
                model.model_number += 1;
                model.star_age += dt_try / SECYER;
                model.dt = dt_try;
                let ts_rep = pick_next_timestep(model, 1.2, 0.5, 1.0e-2);
                model.dt_old = dt_try;

                return Ok(EvolveStepReport {
                    model_number: model.model_number,
                    star_age: model.star_age,
                    dt: dt_try,
                    dt_next: ts_rep.dt_next,
                    iterations: rep.iterations,
                    retries,
                    num_splits,
                    num_merges,
                    varcontrol: ts_rep.varcontrol,
                    converged: true,
                });
            }
            Ok(_) | Err(_) => {
                if retries >= MAX_RETRIES {
                    restore_from_old(model)?;
                    model.dt_old = prev_dt_old;
                    return Err(StarError::ConvergenceFailure(format!(
                        "star_evolve_step failed after {MAX_RETRIES} retries \
                         (dt={dt_try:.3e} s)"
                    )));
                }
                retries += 1;
                model.num_retries += 1;
                dt_try *= if retries >= 4 { 0.2 } else { 0.5 };
                restore_from_old(model)?;
                model.dt_old = prev_dt_old;
            }
        }
    }
}

/// Evolves `model` across up to `max_steps` steps (`star_evolve` in
/// `evolve.f90`), stopping early if `model.controls.max_model_number` or
/// `model.controls.max_age` is reached.
///
/// # Errors
/// Returns [`StarError`] if any step fails to converge after retries.
pub fn star_evolve(
    model: &mut StarModel,
    max_steps: usize,
    do_remesh: bool,
    op_split_burn: bool,
    use_bcyclic: bool,
) -> Result<Vec<EvolveStepReport>, StarError> {
    let mut reports = Vec::with_capacity(max_steps);
    for _ in 0..max_steps {
        if let Some(max_mod) = model.controls.max_model_number {
            if model.model_number >= max_mod {
                break;
            }
        }
        if let Some(max_age) = model.controls.max_age {
            if model.star_age >= max_age * (1.0 - 1.0e-12) {
                break;
            }
            let dt_base = if model.dt_next > 0.0 {
                model.dt_next
            } else {
                model.dt
            };
            let rem_sec = (max_age - model.star_age) * SECYER;
            if rem_sec > 0.0 && dt_base > rem_sec {
                model.dt_next = rem_sec.max(1.0);
                model.dt = model.dt_next;
            }
        }
        let rep =
            star_evolve_step(model, do_remesh, op_split_burn, use_bcyclic)?;
        reports.push(rep);
    }
    Ok(reports)
}
