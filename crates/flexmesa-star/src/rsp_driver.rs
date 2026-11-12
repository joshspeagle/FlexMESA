// SPDX-License-Identifier: LGPL-3.0-or-later
//! Top-level `RSP` setup (`rsp_setup_model`, `init_rsp_model_from_bundle`) and
//! nonlinear `HYD` timestep driver (`rsp_one_step`, `do1_step`,
//! `get_LINA_info` in `star/private/rsp.f90:204-857`).

use std::fs;
use std::path::Path;

use flexmesa_const::{CLIGHT, CRAD, PI, SECYER};

use crate::controls::{StarError, StarJobControls};
use crate::inlist::InlistBundle;
use crate::inlist_runner::StarExtras;
use crate::model::StarModel;
use crate::model_io::read_mod_str;
use crate::rsp_build::{
    do_rsp_build_model, eval_rsp_2nd_crossing_with_controls,
};
use crate::rsp_def::RspDerivedConsts;
use crate::rsp_eos_kap::{get_surf_p_t_kap, RspChemCache};
use crate::rsp_eqns::RspHydBand;
use crate::rsp_grid::{
    copy_from_xh_to_rsp, copy_results, finish_after_build_model,
    set_star_vars_rsp, RspEnvGrid,
};
use crate::rsp_hyd::{calculate_energies, hyd, init_hyd, rsp_set_teff};
use crate::rsp_hyd_work::RspHydWork;
use crate::rsp_lina::do_lina;
use crate::rsp_stats::{
    calculate_work_integrals, gather_pulse_statistics, get_grpdv,
    rsp_set_gyre_linear_analysis,
};
use crate::star_job_init::resolve_model_path;

fn apply_rsp_job_overrides(model: &mut StarModel, job: &StarJobControls) {
    if job.set_initial_age {
        model.star_age = job.initial_age;
        model.star_age_old = job.initial_age;
        model.aux.rsp.time_sec = job.initial_age * SECYER;
        model.aux.rsp.time_old_sec = model.aux.rsp.time_sec;
    }
    if job.set_initial_model_number {
        model.model_number = job.initial_model_number;
    }
    if job.set_initial_cumulative_energy_error {
        model.aux.cumulative_energy_error = job.new_cumulative_energy_error;
    }
    if job.set_initial_dt {
        if job.years_for_initial_dt > 0.0 {
            model.dt = job.years_for_initial_dt * SECYER;
            model.dt_next = model.dt;
            model.aux.rsp.rsp_dt = model.dt;
        } else if job.seconds_for_initial_dt > 0.0 {
            model.dt = job.seconds_for_initial_dt;
            model.dt_next = model.dt;
            model.aux.rsp.rsp_dt = model.dt;
        }
    }
}

/// Handles `create_RSP_model`, multi-stage `RSP` continuation, and `RSP_flag`
/// `.mod` loading for [`crate::star_job_init::init_or_update_model_from_bundle`].
pub(crate) fn init_rsp_model_from_bundle(
    bundle: &InlistBundle,
    existing_model: &mut Option<StarModel>,
    extras: &mut dyn StarExtras,
) -> Result<Option<StarModel>, StarError> {
    let job = &bundle.job;
    let controls = &bundle.controls;

    if job.create_rsp_model {
        let mut model = if controls.rsp.use_other_rsp_build_model {
            let hist_name = controls
                .x_character_ctrl
                .get(1)
                .map_or("history_7M.data", String::as_str);
            let rep = eval_rsp_2nd_crossing_with_controls(
                &bundle.case_dir,
                controls,
                job,
                &bundle.eos_config,
                &bundle.kap_config,
                hist_name,
            )?;
            let mut build_ctrl = controls.clone();
            if let Some(pt) = rep
                .points
                .iter()
                .find(|p| p.growth_rate > 0.0)
                .or_else(|| rep.points.first())
            {
                build_ctrl.rsp_teff = pt.teff_k;
                build_ctrl.rsp_l = pt.lum_lsun;
            } else {
                build_ctrl.rsp_teff = 6050.0;
                build_ctrl.rsp_l = 1438.8;
            }
            build_ctrl.rsp.nmodes = 1;
            build_ctrl.rsp.use_other_rsp_build_model = false;
            let (m, _) = do_rsp_build_model(
                build_ctrl,
                job.clone(),
                bundle.eos_config,
                bundle.kap_config.clone(),
                Some(&bundle.case_dir),
            )?;
            m
        } else {
            let (m, _) = do_rsp_build_model(
                controls.clone(),
                job.clone(),
                bundle.eos_config,
                bundle.kap_config.clone(),
                Some(&bundle.case_dir),
            )?;
            m
        };
        apply_rsp_job_overrides(&mut model, job);
        model.save_old_values();
        extras.extras_controls(&mut model);
        extras.extras_startup(&mut model, false)?;
        return Ok(Some(model));
    }

    if !job.load_saved_model
        && existing_model.as_ref().is_some_and(|p| p.controls.rsp_flag)
    {
        let keep_rsp = (!controls.rsp2.change_rsp_flag
            || controls.rsp2.new_rsp_flag)
            && !(controls.rsp2.change_rsp2_flag && controls.rsp2.new_rsp2_flag);
        if keep_rsp {
            let mut prev = existing_model.take().expect("checked Some");
            let prev_tau = prev.controls.tau_factor;
            let prev_net = prev.job.net_name.clone();
            prev.job = job.clone();
            prev.job.net_name = prev_net;
            prev.controls = controls.clone();
            prev.controls.rsp_flag = true;
            prev.controls.v_flag = true;
            if !job.set_tau_factor {
                prev.controls.tau_factor = prev_tau;
            }
            prev.eos_config = bundle.eos_config;
            prev.kap_config = bundle.kap_config.clone();
            apply_rsp_job_overrides(&mut prev, job);
            extras.extras_controls(&mut prev);
            extras.extras_startup(&mut prev, false)?;
            return Ok(Some(prev));
        }
    }

    if job.load_saved_model && !job.saved_model_name.is_empty() {
        if let Some(mod_path) =
            resolve_model_path(&bundle.case_dir, &job.saved_model_name)
        {
            if let Ok(text) = fs::read_to_string(&mod_path) {
                if let Ok(mut m) =
                    read_mod_str(&text, controls.clone(), job.clone())
                {
                    if m.controls.rsp_flag {
                        m.controls.v_flag = true;
                        m.eos_config = bundle.eos_config;
                        m.kap_config = bundle.kap_config.clone();
                        apply_rsp_job_overrides(&mut m, job);
                        rsp_setup_model(&mut m, false, false)?;
                        m.save_old_values();
                        extras.extras_controls(&mut m);
                        extras.extras_startup(&mut m, false)?;
                        return Ok(Some(m));
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Completes `rsp_setup_part1`, `begin_calculation`, and `rsp_setup_part2`
/// for a freshly built, loaded `.mod`, or restarted `.photo` `RSP` model
/// (`rsp.f90:220-334, 817-857`).
pub fn rsp_setup_model(
    model: &mut StarModel,
    build_flag: bool,
    restart_flag: bool,
) -> Result<(), StarError> {
    let nz = model.nz;
    let dc = RspDerivedConsts::from_config(&model.controls.rsp, nz);
    let chem = RspChemCache::from_controls(
        &model.chem_id,
        &model.job,
        &model.controls,
    )?;

    if restart_flag {
        model.controls.tau_factor = model.aux.rsp.rsp_tau_factor;
        return Ok(());
    }

    if build_flag {
        model.aux.rsp.rsp_tau_factor = model.controls.tau_factor;
        model.dt_next = model.aux.rsp.rsp_dt;
        model.dt = model.aux.rsp.rsp_dt;
    } else {
        model.aux.rsp.iwork = 0;
        let mut dt_init = if model.dt_next > 0.0 {
            model.dt_next
        } else {
            model.dt
        };
        if model.controls.rsp.max_timestep > 0.0
            && dt_init > model.controls.rsp.max_timestep
        {
            dt_init = model.controls.rsp.max_timestep;
        }
        model.aux.rsp.rsp_dt = dt_init;
        model.aux.rsp.rsp_tau_factor = model.controls.tau_factor;
        model.aux.rsp.period =
            dt_init * (model.controls.rsp_target_steps_per_cycle as f64);
        model.aux.rsp.have_set_velocities = true;
        copy_from_xh_to_rsp(model, &dc);
        model.lum_start.fill(0.0);
        model.aux.rsp.psurf_from_atm =
            if model.controls.rsp.use_atm_grey_with_kap_for_psurf {
                let (_ts, ps, _ks, _ta) = get_surf_p_t_kap(
                    model,
                    &chem,
                    model.m[0],
                    model.r[0],
                    model.lum[0],
                    model.controls.rsp.tau_surf_for_atm_grey_with_kap,
                    1.0e-2,
                )?;
                ps
            } else if model.controls.rsp.use_prad_for_psurf {
                CRAD * model.t[0].powi(4) / 3.0
            } else {
                0.0
            };
    }

    model.aux.rsp.min_dr_div_cs = 1.0e99;
    model.aux.rsp.min_rad_diff_time = 1.0e99;

    {
        let rsp = &mut model.aux.rsp;
        rsp.first = 0;
        rsp.tt1 = 0.0;
        rsp.id_step = 0;
        rsp.inside = 1;
        rsp.ekmax = -10.0e50;
        rsp.ekmin = 10.0e50;
        rsp.ekmaxl = rsp.ekmax;
        rsp.ekdel = rsp.ekmax - rsp.ekmin;
        rsp.lmin = 10.0e10;
        rsp.lmax = -10.0e10;
        rsp.rmax = -dc.sunr;
        rsp.rmin = dc.sunr;
        rsp.vmax = 0.0;
        rsp.grekm = 0.0;
        rsp.grekm_avg_abs = 0.0;
        rsp.grpdv = 0.0;
        rsp.delta_r = 0.0;
        rsp.delta_mag = 0.0;
        rsp.e0 = 0.0;
        rsp.num_periods = 0;
    }
    set_star_vars_rsp(model, &dc);
    let mut wk = RspHydWork::from_model(model);
    let mut band = RspHydBand::new(nz);
    init_hyd(model, &chem, &dc, &mut wk, &mut band)?;
    let _rad = calculate_energies(model, &dc, &wk);
    model.aux.rsp.e0 = model.aux.rsp.ede_start;
    calculate_work_integrals(model, &dc, None);
    model.controls.tau_factor = model.aux.rsp.rsp_tau_factor;

    finish_after_build_model(model);
    copy_results(model, &dc);
    rsp_set_teff(model);
    if model.aux.rsp.period > 0.0
        && model.controls.rsp_target_steps_per_cycle > 0
    {
        model.aux.rsp.rsp_dt = model.aux.rsp.period
            / (model.controls.rsp_target_steps_per_cycle as f64);
        model.dt_next =
            model.aux.rsp.rsp_dt * model.controls.rsp.initial_dt_factor;
        model.dt = model.dt_next;
    }
    if model.controls.rsp.use_other_rsp_build_model
        && model.controls.rsp.set_psurf_to_multiple_of_initial_p1 > 0.0
    {
        let peos1 = model.pgas[0] + model.prad[0];
        model.controls.rsp.psurf =
            peos1 * model.controls.rsp.set_psurf_to_multiple_of_initial_p1;
    }
    model.aux.total_energy = model.aux.rsp.etot;
    model.aux.rsp.total_energy_old = model.aux.rsp.etot;
    Ok(())
}

/// Evaluates `(total_energy, total_radiation)` on `model` (`rsp.f90:943-960`).
#[must_use]
pub fn rsp_total_energy_integrals(model: &mut StarModel) -> (f64, f64) {
    let dc = RspDerivedConsts::from_config(&model.controls.rsp, model.nz);
    let wk = RspHydWork::from_model(model);
    let total_radiation = calculate_energies(model, &dc, &wk);
    (model.aux.rsp.etot, total_radiation)
}

/// Executes one complete `RSP` nonlinear timestep (`rsp_one_step` in `rsp.f90:405-496`).
pub fn rsp_one_step(
    model: &mut StarModel,
    log_dir: Option<&Path>,
) -> Result<(usize, usize), StarError> {
    let nz = model.nz;
    let dc = RspDerivedConsts::from_config(&model.controls.rsp, nz);
    let chem = RspChemCache::from_controls(
        &model.chem_id,
        &model.job,
        &model.controls,
    )?;

    model.aux.rsp.just_set_velocities = false;
    if !model.aux.rsp.have_set_velocities {
        model.aux.rsp.just_set_velocities = true;
        if model.controls.rsp.use_other_rsp_linear_analysis {
            rsp_set_gyre_linear_analysis(model, &chem, log_dir)?;
        } else {
            get_lina_info(model, &chem, &dc, log_dir)?;
        }
        model.aux.rsp.period_lin = model.aux.rsp.period;
        if model.controls.rsp_target_steps_per_cycle > 0 {
            model.aux.rsp.rsp_dt = model.aux.rsp.period
                / (model.controls.rsp_target_steps_per_cycle as f64);
            model.dt = model.aux.rsp.rsp_dt;
        }
        model.aux.cumulative_energy_error = 0.0;
        model.aux.rsp.time_sec = 0.0;
        model.aux.rsp.time_old_sec = 0.0;
        set_random_velocities(model);
    }

    if model.controls.rsp.max_dt > 0.0 && model.dt > model.controls.rsp.max_dt {
        model.dt = model.controls.rsp.max_dt;
    }

    let counts = do1_step(model, &chem, &dc, log_dir)?;
    copy_results(model, &dc);
    rsp_set_teff(model);
    Ok(counts)
}

fn do1_step(
    model: &mut StarModel,
    chem: &RspChemCache,
    dc: &RspDerivedConsts,
    log_dir: Option<&Path>,
) -> Result<(usize, usize), StarError> {
    let nz = model.nz;
    model.aux.rsp.id_step += 1;

    let mut target_dt = (model.aux.rsp.period
        / (model.controls.rsp_target_steps_per_cycle.max(1) as f64))
        .min(model.dt * model.controls.rsp.max_timestep_factor);
    if model.controls.rsp.max_dt > 0.0 {
        target_dt = target_dt.min(model.controls.rsp.max_dt);
    }
    if model.controls.rsp.max_timestep > 0.0 {
        target_dt = target_dt.min(model.controls.rsp.max_timestep);
    }
    model.dt = target_dt;

    let max_dt = model.aux.rsp.min_dr_div_cs
        * model.controls.rsp.max_dt_times_min_dr_div_cs;
    if model.dt > max_dt {
        model.dt = max_dt;
    }
    if model.controls.rsp.force_timestep > 0.0 {
        model.dt = model.controls.rsp.force_timestep;
    }
    if !(model.dt > 0.0 && model.dt.is_finite()) {
        return Err(StarError::InvalidInput(format!(
            "RSP do1_step non-positive dt={}",
            model.dt
        )));
    }

    let mut wk = RspHydWork::from_model(model);
    let mut band = RspHydBand::new(nz);
    let counts = hyd(model, chem, dc, &mut wk, &mut band)?;

    model.aux.rsp.time_sec = model.aux.rsp.time_old_sec + model.dt;
    model.aux.rsp.rsp_dt = model.dt;

    let mut min_dr_cs = 1.0e99_f64;
    let mut r_00 = model.r_center;
    for i in 1..=nz {
        let k0 = nz - i;
        let r_in = r_00;
        r_00 = model.r[k0];
        if model.v[k0].abs() / model.csound[k0]
            < model.controls.rsp.v_div_cs_threshold_for_dt_limit
        {
            continue;
        }
        let dr_cs = (r_00 - r_in) / model.csound[k0];
        if dr_cs < min_dr_cs {
            min_dr_cs = dr_cs;
        }
    }
    model.aux.rsp.min_dr_div_cs = min_dr_cs;

    model.aux.rsp.min_rad_diff_time = 1.0e99;
    if model.controls.rsp.max_dt_times_min_rad_diff_time > 0.0 {
        let mut min_dt = 1.0e99_f64;
        for k0 in 0..nz {
            let mfp = model.aux.rsp.vol_rsp[k0] / model.opacity[k0];
            let diff = CLIGHT * mfp / 3.0;
            let dr = if k0 + 1 < nz {
                model.r[k0] - model.r[k0 + 1]
            } else {
                model.r[k0] - model.r_center
            };
            let dt_cell =
                dr * dr / diff * (1.0 + model.pgas[k0] / model.prad[k0]);
            if dt_cell < min_dt {
                min_dt = dt_cell;
            }
        }
        model.aux.rsp.min_rad_diff_time = min_dt;
    }

    calculate_work_integrals(model, dc, log_dir);
    let _rad = calculate_energies(model, dc, &wk);
    gather_pulse_statistics(model, dc);
    if model.controls.rsp_max_num_periods.is_some_and(|lim| {
        lim >= 0 && (model.aux.rsp.num_periods as isize) >= lim
    }) {
        get_grpdv(model, dc);
    }
    Ok(counts)
}

fn get_lina_info(
    model: &mut StarModel,
    chem: &RspChemCache,
    dc: &RspDerivedConsts,
    log_dir: Option<&Path>,
) -> Result<(), StarError> {
    if model.controls.rsp.kick_vsurf_km_per_sec == 0.0 {
        return Ok(());
    }
    let nzn = model.nz;
    let mut grid = RspEnvGrid::new(nzn);
    grid.m[0] = model.m_center;
    grid.r[0] = model.r_center;
    for i in 1..=nzn {
        let k0 = nzn - i;
        grid.m[i] = model.m[k0];
        grid.dm[i] = model.dm[k0];
        grid.dm_bar[i] = model.dm_bar[k0];
        grid.r[i] = model.r[k0];
        grid.vol[i] = model.aux.rsp.vol_rsp[k0];
        grid.t[i] = model.t[k0];
        grid.et[i] = model.aux.rsp.et_rsp[k0];
        grid.lr[i] = 4.0 * PI * model.r[k0].powi(2) * model.aux.rsp.fr_rsp[k0];
    }
    let nmodes = model.controls.rsp.nmodes.max(1);
    let l_surf = model.controls.rsp_l * dc.sunl;
    let lina = do_lina(model, chem, dc, &grid, l_surf, nmodes, log_dir)?;
    for m in 0..nmodes.min(3) {
        model.aux.rsp.lina_periods[m] = lina.periods_sec[m];
        model.aux.rsp.lina_growth_rates[m] = lina.growth_rates[m];
    }
    let mode_idx = model.controls.rsp.mode_for_setting_periodlin;
    model.aux.rsp.period = lina
        .periods_sec
        .get(mode_idx)
        .copied()
        .unwrap_or(lina.periods_sec[0]);

    let amix1 = model.controls.rsp.fraction_1st_overtone;
    let amix2 = model.controls.rsp.fraction_2nd_overtone;
    let kick = 1.0e5 * model.controls.rsp.kick_vsurf_km_per_sec;
    model.v_center = 0.0;
    for i in 1..=nzn {
        let v0 = lina.vel_modes.first().map_or(0.0, |v| v[i]);
        let v1 = lina.vel_modes.get(1).map_or(0.0, |v| v[i]);
        let v2 = lina.vel_modes.get(2).map_or(0.0, |v| v[i]);
        model.v[nzn - i] =
            kick * ((1.0 - amix1 - amix2) * v0 + amix1 * v1 + amix2 * v2);
    }
    model.aux.rsp.have_set_velocities = true;
    Ok(())
}

fn set_random_velocities(model: &mut StarModel) {
    let nzn = model.nz;
    let (avel, arnd) = (model.controls.rsp.avel, model.controls.rsp.arnd);
    if avel != 0.0 || arnd != 0.0 {
        for i in 1..=nzn {
            let k0 = nzn - i;
            let rv = model.aux.rsp.rng.next_f64();
            model.v[k0] += 1.0e5 * (i as f64) / (nzn as f64)
                * (avel + arnd * (2.0 * rv - 1.0));
        }
        model.aux.rsp.have_set_velocities = true;
    }
}
