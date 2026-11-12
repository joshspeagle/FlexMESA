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

use flexmesa_const::{R_SUN, SECYER};

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
    model.v.clone_from(&model.v_old);
    model.xa.clone_from(&model.xa_old);
    model.dq.clone_from(&model.dq_old);
    model.aux.rsp.et_rsp.clone_from(&model.aux.et_rsp_old);
    model.aux.rsp.hp_face.clone_from(&model.aux.hp_face_old);
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

    if model.controls.rsp_flag {
        if model.dt_next > 0.0 {
            model.dt = model.dt_next;
        }
        let prev_dt_old = model.dt_old;
        model.save_old_values();
        model.dt_old = prev_dt_old;
        model.aux.rsp.time_old_sec = model.aux.rsp.time_sec;
        let (e_old, _) = crate::rsp_driver::rsp_total_energy_integrals(model);
        model.aux.rsp.total_energy_old = e_old;
        model.model_number += 1;

        let (iterations, retries) =
            crate::rsp_driver::rsp_one_step(model, None)?;
        model.total_num_solver_iterations += iterations;
        model.num_retries += retries;
        model.star_age = model.aux.rsp.time_sec / SECYER;
        model.dt_next = model.aux.rsp.rsp_dt;
        model.dt_old = model.dt;

        let (etot, mut total_radiation) =
            crate::rsp_driver::rsp_total_energy_integrals(model);
        if model.aux.rsp.just_set_velocities {
            model.aux.rsp.total_energy_old = etot;
            total_radiation = 0.0;
            model.aux.cumulative_energy_error = 0.0;
        }
        let err_e = etot - (model.aux.rsp.total_energy_old - total_radiation);
        model.aux.cumulative_energy_error += err_e;
        model.aux.total_energy = etot;

        return Ok(EvolveStepReport {
            model_number: model.model_number,
            star_age: model.star_age,
            dt: model.dt,
            dt_next: model.dt_next,
            iterations,
            retries,
            num_splits: 0,
            num_merges: 0,
            varcontrol: 0.0,
            converged: true,
        });
    }

    if model.dt_next > 0.0 {
        model.dt = model.dt_next;
    }
    let prev_dt_old = model.dt_old;

    let (num_splits, num_merges) = if do_remesh && !model.controls.rsp2_flag {
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
    let rsp2_e_old = if model.controls.rsp2_flag {
        crate::rsp_driver::rsp_total_energy_integrals(model).0
    } else {
        0.0
    };

    let dt_floor = model.controls.min_timestep_limit.clamp(1.0e-20, 1.0);
    let mut dt_try = model.dt.max(dt_floor);
    if let Some(max_yr) = model.controls.max_years_for_timestep {
        dt_try = dt_try.min(max_yr * SECYER);
    }
    if model.controls.rsp.max_timestep > 0.0 {
        dt_try = dt_try.min(model.controls.rsp.max_timestep);
    }
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
                let mut dt_next = ts_rep.dt_next;
                if let Some(max_yr) = model.controls.max_years_for_timestep {
                    dt_next = dt_next.min(max_yr * SECYER);
                }
                if model.controls.rsp.max_timestep > 0.0 {
                    dt_next = dt_next.min(model.controls.rsp.max_timestep);
                }
                if model.controls.rsp2_flag {
                    model.aux.rsp.time_old_sec = model.aux.rsp.time_sec;
                    model.aux.rsp.time_sec += dt_try;
                    let (etot, total_rad) =
                        crate::rsp_driver::rsp_total_energy_integrals(model);
                    model.aux.rsp.total_energy_old = rsp2_e_old;
                    model.aux.cumulative_energy_error +=
                        etot - (rsp2_e_old - total_rad);
                    model.aux.total_energy = etot;
                    update_rsp2_cycle_stats(model, dt_try, &mut dt_next);
                }
                model.dt_next = dt_next;
                model.dt_old = dt_try;

                return Ok(EvolveStepReport {
                    model_number: model.model_number,
                    star_age: model.star_age,
                    dt: dt_try,
                    dt_next,
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

fn update_rsp2_cycle_stats(
    model: &mut StarModel,
    dt_try: f64,
    dt_next: &mut f64,
) {
    if model.nz == 0 {
        return;
    }
    let r_surf = model.r[0];
    if model.aux.rsp.rmax <= 0.0 {
        model.aux.rsp.rmax = r_surf;
        model.aux.rsp.rmin = r_surf;
    } else {
        model.aux.rsp.rmax = model.aux.rsp.rmax.max(r_surf);
        model.aux.rsp.rmin = model.aux.rsp.rmin.min(r_surf);
    }
    if model.controls.v_flag {
        let v_beg = model.v_start[0];
        let v_end = model.v[0];
        if v_end <= 0.0 && v_beg >= 0.0 && (v_beg - v_end).abs() > 0.0 {
            let frac = (v_beg / (v_beg - v_end)).clamp(0.0, 1.0);
            let t_cross = model.aux.rsp.time_old_sec + frac * dt_try;
            if model.aux.rsp.tt1 > 0.0 {
                let p_new = t_cross - model.aux.rsp.tt1;
                if p_new > 0.0 {
                    model.aux.rsp.period = p_new;
                    model.aux.rsp.num_periods += 1;
                    model.aux.rsp.delta_r =
                        (model.aux.rsp.rmax - model.aux.rsp.rmin) / R_SUN;
                }
            }
            model.aux.rsp.tt1 = t_cross;
            model.aux.rsp.rmax = r_surf;
            model.aux.rsp.rmin = r_surf;
        }
    }
    if model.controls.rsp2.target_steps_per_cycle > 0
        && model.aux.rsp.period > 0.0
    {
        let dt_cap = model.aux.rsp.period
            / (model.controls.rsp2.target_steps_per_cycle as f64);
        *dt_next = (*dt_next).min(dt_cap);
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
