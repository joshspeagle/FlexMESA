// SPDX-License-Identifier: GPL-3.0-or-later
//! Top-level `RSP` setup (`rsp_setup_model`, `init_rsp_model_from_bundle`) and
//! nonlinear `HYD` timestep driver (`rsp_one_step`, `do1_step`,
//! `get_LINA_info` in `star/private/rsp.f90:204-857`).

use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use flexmesa_const::{CLIGHT, CRAD, PI, SECYER};

use crate::controls::{StarError, StarJobControls};
use crate::inlist::InlistBundle;
use crate::inlist_runner::{DefaultStarExtras, RspBuildOutcome, StarExtras};
use crate::mesa_profile::{
    format_rsp_map_header, format_rsp_map_rows, parse_rsp_map_columns,
    resolve_rsp_map_column_names,
};
use crate::model::StarModel;
use crate::model_io::read_mod_str_with_dir;
use crate::rsp_build::do_rsp_build_model;
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

/// Outcome from `init_rsp_model_from_bundle` (`None` falls through to
/// non-`RSP` ZAMS/pre-MS/saved-model initialization).
pub(crate) enum InitRspOutcome {
    NotHandled,
    Model(Box<StarModel>),
    Terminate(String),
}

/// Handles `create_RSP_model`, multi-stage `RSP` continuation, and `RSP_flag`
/// `.mod` loading for [`crate::star_job_init::init_or_update_model_from_bundle_outcome`].
pub(crate) fn init_rsp_model_from_bundle(
    bundle: &InlistBundle,
    existing_model: &mut Option<StarModel>,
    extras: &mut dyn StarExtras,
) -> Result<InitRspOutcome, StarError> {
    let job = &bundle.job;
    let controls = &bundle.controls;

    if job.create_rsp_model {
        let mut model = if controls.rsp.use_other_rsp_build_model {
            match extras.other_rsp_build_model(bundle)? {
                Some(RspBuildOutcome::Model(m)) => *m,
                Some(RspBuildOutcome::Terminate(reason)) => {
                    return Ok(InitRspOutcome::Terminate(reason));
                }
                None => {
                    return Err(StarError::InvalidInput(String::from(
                        "use_other_rsp_build_model = .true. requires an other_rsp_build_model hook",
                    )));
                }
            }
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
        extras.extras_controls(&mut model)?;
        extras.extras_startup(&mut model, false)?;
        return Ok(InitRspOutcome::Model(Box::new(model)));
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
            extras.extras_controls(&mut prev)?;
            extras.extras_startup(&mut prev, false)?;
            return Ok(InitRspOutcome::Model(Box::new(prev)));
        }
    }

    if job.load_saved_model {
        let trimmed = job.saved_model_name.trim();
        if trimmed.is_empty() {
            return Err(StarError::InvalidInput(String::from(
                "load_saved_model = true requires a non-empty load_model_filename",
            )));
        }
        let mod_path = resolve_model_path(&bundle.case_dir, trimmed)
            .ok_or_else(|| {
                StarError::IoError(format!(
                    "missing saved model '{trimmed}' in '{}'",
                    bundle.case_dir.display()
                ))
            })?;
        let text = fs::read_to_string(&mod_path).map_err(|e| {
            StarError::IoError(format!(
                "failed to read saved model {}: {e}",
                mod_path.display()
            ))
        })?;
        let mut m = read_mod_str_with_dir(
            &text,
            controls.clone(),
            job.clone(),
            Some(&bundle.case_dir),
        )?;
        if m.controls.rsp_flag {
            m.controls.v_flag = true;
            m.eos_config = bundle.eos_config;
            m.kap_config = bundle.kap_config.clone();
            apply_rsp_job_overrides(&mut m, job);
            rsp_setup_model(&mut m, false, false)?;
            m.save_old_values();
            extras.extras_controls(&mut m)?;
            extras.extras_startup(&mut m, false)?;
            return Ok(InitRspOutcome::Model(Box::new(m)));
        }
    }

    Ok(InitRspOutcome::NotHandled)
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
        model.aux.rsp.writing_map = false;
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
                CRAD * flexmesa_math::powi_f90(model.t[0], 4) / 3.0
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
    set_star_vars_rsp(model, &dc)?;
    let mut wk = RspHydWork::from_model(model);
    let mut band = RspHydBand::new(nz);
    init_hyd(model, &chem, &dc, &mut wk, &mut band)?;
    let _rad = calculate_energies(model, &dc, &wk);
    model.aux.rsp.e0 = model.aux.rsp.ede_start;
    calculate_work_integrals(model, &dc, None)?;
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

/// Executes one complete `RSP` nonlinear timestep with no custom hooks
/// (`rsp_one_step` in `rsp.f90:405-496`).
///
/// # Errors
/// Returns [`StarError`] if linear analysis, hydro iteration, or a required
/// `other_*` hook fails.
pub fn rsp_one_step(
    model: &mut StarModel,
    log_dir: Option<&Path>,
) -> Result<(usize, usize), StarError> {
    let mut extras = DefaultStarExtras;
    rsp_one_step_with_extras(model, log_dir, &mut extras)
}

/// Executes one complete `RSP` nonlinear timestep with [`StarExtras`] hooks
/// (`rsp_one_step` in `rsp.f90:405-496`).
///
/// # Errors
/// Returns [`StarError`] if linear analysis, hydro iteration, or a required
/// `other_rsp_linear_analysis` hook fails.
pub fn rsp_one_step_with_extras(
    model: &mut StarModel,
    log_dir: Option<&Path>,
    extras: &mut dyn StarExtras,
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
            if !extras.other_rsp_linear_analysis(model, &chem, log_dir)? {
                return Err(StarError::InvalidInput(String::from(
                    "use_other_rsp_linear_analysis = .true. requires an other_rsp_linear_analysis hook",
                )));
            }
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

    calculate_work_integrals(model, dc, log_dir)?;
    let _rad = calculate_energies(model, dc, &wk);
    gather_pulse_statistics(model, dc);
    if model.controls.rsp_max_num_periods >= 0
        && (model.aux.rsp.num_periods as isize)
            >= model.controls.rsp_max_num_periods
    {
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
        grid.lr[i] = 4.0
            * PI
            * flexmesa_math::pow2(model.r[k0])
            * model.aux.rsp.fr_rsp[k0];
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

/// Executes post-step 3D map output and history file switching when
/// `RSP_write_map` is enabled (`add_to_map` and `read_map_specs` in
/// `star/private/rsp.f90:495-637`).
///
/// # Errors
/// Returns [`StarError`] if `RSP_map_columns_filename` cannot be opened or
/// parsed, `extra_profile_columns` fails, or map file I/O fails.
pub fn rsp_after_step_map(
    model: &mut StarModel,
    case_dir: &Path,
    log_dir: Option<&Path>,
    extras: &dyn StarExtras,
) -> Result<(), StarError> {
    if !model.controls.rsp_flag || !model.controls.rsp.write_map {
        return Ok(());
    }
    let npch1 = model.controls.rsp.map_first_period;
    let npch2 = model.controls.rsp.map_last_period;
    let ip = model.aux.rsp.num_periods as isize;

    if ip + 1 >= npch1 && ip < npch2 {
        let extra_cols = extras.extra_profile_columns(model)?;
        let extra_names: Vec<&str> =
            extra_cols.iter().map(|(name, _)| name.trim()).collect();

        let col_specs = if model.aux.rsp.writing_map {
            resolve_rsp_map_column_names(
                &model.aux.rsp.map_col_names,
                &extra_names,
            )?
        } else {
            let cols_filename =
                match model.controls.rsp.map_columns_filename.trim() {
                    "" => "map_columns.list",
                    s => s,
                };
            let cols_path = if Path::new(cols_filename).is_absolute() {
                PathBuf::from(cols_filename)
            } else {
                case_dir.join(cols_filename)
            };
            let text = fs::read_to_string(&cols_path).map_err(|e| {
                StarError::IoError(format!(
                    "failed to open {}: {e}",
                    cols_path.display()
                ))
            })?;
            let (col_names, col_specs) =
                parse_rsp_map_columns(&text, &extra_names)?;
            if model.controls.rsp.map_filename.trim().is_empty() {
                model.controls.rsp.map_filename = String::from("map.data");
            }
            let hist_name = model.controls.rsp.map_history_filename.trim();
            if hist_name.is_empty() {
                return Err(StarError::InvalidInput(String::from(
                    "rsp_map_history_filename must be non-empty when rsp_write_map is active",
                )));
            }
            let hdr = format_rsp_map_header(&col_names);
            if let Some(dir) = log_dir {
                fs::create_dir_all(dir).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to create {}: {e}",
                        dir.display()
                    ))
                })?;
                let map_path = dir.join(model.controls.rsp.map_filename.trim());
                fs::write(&map_path, hdr.as_bytes()).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to write {}: {e}",
                        map_path.display()
                    ))
                })?;
            }
            model.aux.rsp.map_col_names = col_names;
            model.aux.rsp.writing_map = true;
            model.aux.rsp.done_writing_map = false;
            model.controls.output.need_to_set_history_names_etc = true;
            model.controls.output.star_history_name = hist_name.to_owned();
            model.aux.rsp.fase0 = model.aux.rsp.time_sec;
            col_specs
        };

        let phase = if model.aux.rsp.period == 0.0 {
            0.0
        } else {
            (model.aux.rsp.time_sec - model.aux.rsp.fase0)
                / model.aux.rsp.period
        };
        let ibotom =
            RspDerivedConsts::from_config(&model.controls.rsp, model.nz).ibotom;
        let rows = format_rsp_map_rows(
            model,
            &col_specs,
            &extra_cols,
            model.controls.rsp.map_zone_interval,
            ibotom,
            phase,
        );
        if !rows.is_empty() {
            if let Some(dir) = log_dir {
                let map_path = dir.join(model.controls.rsp.map_filename.trim());
                let mut f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&map_path)
                    .map_err(|e| {
                        StarError::IoError(format!(
                            "failed to open {}: {e}",
                            map_path.display()
                        ))
                    })?;
                f.write_all(rows.as_bytes()).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to write {}: {e}",
                        map_path.display()
                    ))
                })?;
                f.flush().map_err(|e| {
                    StarError::IoError(format!(
                        "failed to flush {}: {e}",
                        map_path.display()
                    ))
                })?;
            }
        }
    }

    if ip == npch2 && !model.aux.rsp.done_writing_map {
        model.aux.rsp.done_writing_map = true;
    }
    Ok(())
}
