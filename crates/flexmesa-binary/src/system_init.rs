// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pre-run ZAMS composition, mesh, rotational-distortion, and thermal-equilibrium
//! settling helper for coupled [`crate::system::BinarySystem`] star models
//! (`relax_initial_Z` / `relax_initial_Y` in `star/private/relax.f90:1196-1260, 3830-4160`).

use flexmesa_const::{LSUN, SECYER};
use flexmesa_star::{
    EnergyEqnOption, MeshFunctionConfig, StarControls, StarError, StarModel,
    apply_step_pre_physics, apply_uniform_rotation, compute_power_categories,
    do_adjust_mesh, do_struct_burn_mix, eval_total_energy_integrals,
    remap_onto_dq, set_hydro_vars, snapshot_step_history_old,
    update_rotation_distortion_factors,
};

fn log10_lnuc_over_lsurf(star: &StarModel) -> f64 {
    let lnuc = (compute_power_categories(star).power_nuc_burn * LSUN).abs();
    let lsurf = star.lum.first().copied().unwrap_or(0.0).abs().max(1.0);
    (lnuc.max(1.0) / lsurf).log10()
}

fn regularize_zams_mesh(star: &mut StarModel) -> Result<(), StarError> {
    let mut mcfg = MeshFunctionConfig::from_controls(&star.controls);
    mcfg.max_dq = mcfg.max_dq.max(0.1);
    mcfg.max_surface_cell_dq = mcfg.max_surface_cell_dq.max(1.0);
    mcfg.max_center_cell_dq = mcfg.max_center_cell_dq.max(1.0);
    for _ in 0..8 {
        set_hydro_vars(star, false, false)?;
        let (ns, nm) = do_adjust_mesh(star, &mcfg)?;
        if ns + nm == 0 {
            break;
        }
    }
    if star.nz < 4 {
        return Ok(());
    }
    let mut clean_dq: Vec<f64> = Vec::with_capacity(star.nz);
    let mut q_rem = 1.0_f64;
    for &dq_k in &star.dq {
        let min_allowed = if q_rem >= 0.99 { 1.0e-8 } else { 5.0e-5 };
        q_rem -= dq_k;
        if let Some(last) = clean_dq.last_mut() {
            if *last < min_allowed {
                *last += dq_k;
                continue;
            }
        }
        clean_dq.push(dq_k);
    }
    if clean_dq.len() >= 2 && clean_dq.last().copied().unwrap_or(1.0) < 5.0e-5 {
        let tail = clean_dq.pop().unwrap_or(0.0);
        if let Some(last) = clean_dq.last_mut() {
            *last += tail;
        }
    }
    let dq_sum: f64 = clean_dq.iter().sum();
    if dq_sum > 0.0 && clean_dq.len() < star.nz {
        for d in &mut clean_dq {
            *d /= dq_sum;
        }
        remap_onto_dq(star, &clean_dq)?;
    }
    Ok(())
}

fn settle_constant_xa_kh_loop(
    star: &mut StarModel,
    target_abs_log_ratio: f64,
    max_steps: usize,
) -> Result<(), StarError> {
    let mut dt_yr = 50.0_f64;
    for step in 0..max_steps {
        set_hydro_vars(star, false, false)?;
        if step >= 2 && log10_lnuc_over_lsurf(star).abs() < target_abs_log_ratio {
            break;
        }
        let backup = star.clone();
        star.save_old_values();
        star.save_start_values();
        match do_struct_burn_mix(star, dt_yr * SECYER, false, true, &[]) {
            Ok(rep) if rep.converged => {
                dt_yr = (dt_yr * 1.8).min(5.0e4);
            }
            _ => {
                *star = backup;
                dt_yr = (dt_yr * 0.5).max(1.0);
            }
        }
    }
    Ok(())
}

fn settle_cno_gold_equilibrium(
    star: &mut StarModel,
    saved_controls: &StarControls,
    omega_0: f64,
) -> Result<(), StarError> {
    star.controls = saved_controls.clone();
    star.controls.use_other_wind = false;
    star.controls.ts_stop.delta_hr_limit = -1.0;
    star.controls.ts_stop.delta_hr_hard_limit = -1.0;
    star.controls.max_newton_iters = saved_controls.max_newton_iters.max(80);
    if star.controls.rotation_flag && omega_0 > 0.0 {
        apply_uniform_rotation(star, omega_0);
    }
    update_rotation_distortion_factors(star);
    set_hydro_vars(star, false, false)?;

    for (idx, &dt_yr) in [0.1_f64, 0.5, 2.0, 8.0, 25.0, 80.0, 200.0]
        .iter()
        .enumerate()
    {
        if idx < 2 {
            star.controls.use_gold_tolerances = false;
            star.controls.use_gold2_tolerances = false;
        } else {
            star.controls.use_gold_tolerances = saved_controls.use_gold_tolerances;
            star.controls.use_gold2_tolerances = saved_controls.use_gold2_tolerances;
        }
        let dt_s = dt_yr * SECYER;
        let backup = star.clone();
        star.dt = dt_s;
        if star.controls.rotation_flag && omega_0 > 0.0 {
            apply_uniform_rotation(star, omega_0);
        }
        update_rotation_distortion_factors(star);
        set_hydro_vars(star, false, false)?;
        star.save_old_values();
        star.save_start_values();
        apply_step_pre_physics(star, dt_s)?;
        match do_struct_burn_mix(star, dt_s, false, true, &[]) {
            Ok(rep) if rep.converged => {
                star.save_old_values();
            }
            _ => {
                *star = backup;
            }
        }
    }
    Ok(())
}

/// Prepares and settles an initial [`StarModel`] at `model_number = 0, star_age = 0`
/// before coupled binary evolution begins, saving and restoring all parsed
/// `StarControls` on exit (`relax.f90:3859-4159`).
pub(crate) fn prep_and_settle_binary_star(
    star: &mut StarModel,
) -> Result<(), StarError> {
    let saved_controls = star.controls.clone();
    let omega_0 = if saved_controls.rotation_flag {
        star.omega.first().copied().unwrap_or(0.0)
    } else {
        0.0
    };
    let needs_composition_settle = star.job.relax_z || star.job.relax_y;

    if needs_composition_settle {
        apply_uniform_rotation(star, 0.0);
        star.controls.rotation_flag = false;
        update_rotation_distortion_factors(star);
        regularize_zams_mesh(star)?;

        star.controls.use_gold_tolerances = false;
        star.controls.use_gold2_tolerances = false;
        star.controls.steps_before_use_gold_tolerances = -1;
        star.controls.steps_before_use_gold2_tolerances = -1;
        star.controls.use_ledoux_criterion = false;
        star.controls.use_other_wind = false;
        star.controls.energy_eqn_option = EnergyEqnOption::EpsGrav;
        star.controls.ts_stop.delta_hr_limit = -1.0;
        star.controls.ts_stop.delta_hr_hard_limit = -1.0;
        star.controls.max_newton_iters = saved_controls.max_newton_iters.max(80);

        star.controls.do_mix = false;
        star.controls.mix_factor = 0.0;
        star.controls.do_burn = true;
        star.controls.eps_nuc_factor = 1.0;
        star.controls.dxdt_nuc_factor = 0.0;
        star.controls.op_split_burn = false;
        settle_constant_xa_kh_loop(star, 0.003, 45)?;

        settle_cno_gold_equilibrium(star, &saved_controls, omega_0)?;
    }

    star.controls = saved_controls;
    if star.controls.rotation_flag && omega_0 > 0.0 {
        apply_uniform_rotation(star, omega_0);
    }
    update_rotation_distortion_factors(star);

    if needs_composition_settle {
        star.star_age = 0.0;
        star.star_age_old = 0.0;
        star.model_number = 0;
        star.num_retries = 0;
        star.total_num_solver_iterations = 0;
        star.dt = 1.0 * SECYER;
        star.dt_next = 1.0 * SECYER;
        star.dt_old = 0.0;
        star.dt_limit_ratio_old = 0.0;
    }

    star.mstar_start = star.mstar;
    star.mstar_old = star.mstar;
    set_hydro_vars(star, false, false)?;

    let etot = eval_total_energy_integrals(star).5;
    star.aux.total_energy = etot;
    star.aux.cumulative_energy_error = 0.0;
    snapshot_step_history_old(star);
    star.aux.step_hist.total_energy_old = etot;
    star.aux.step_hist.total_energy_end = etot;
    star.aux.step_hist.error_in_energy_conservation = 0.0;
    star.aux.step_hist.cumulative_energy_error_old = 0.0;
    star.save_start_values();
    star.save_old_values();
    Ok(())
}
