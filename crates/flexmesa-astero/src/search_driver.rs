// SPDX-License-Identifier: LGPL-3.0-or-later
//! Multi-track asteroseismic search driver (`run_star_astero`) and adaptive
//! evolutionary timestep controller (`extras_support.f90:58-669`,
//! `astero_run_support.f90:36-850`).

use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_const::SECYER;
use flexmesa_star::controls::StarError;
use flexmesa_star::history::extract_history_row;
use flexmesa_star::inlist::{InlistBundle, InlistLoadOptions};
use flexmesa_star::inlist_runner::{
    run_inlist_stage_with_logs, ExtrasDirective, StageRunReport, StarExtras,
    StarLogsOptions,
};
use flexmesa_star::model::StarModel;
use flexmesa_star::star_job_init::init_or_update_model_from_bundle;

use crate::inlist_astero::{
    AsteroParamSet, AsteroSearchControls, SpectroConstraint,
};
use crate::sample_eval::{
    eval_convective_envelope_base_rsun, eval_model_astero_sample,
    eval_spectro_and_delta_nu_chi2, AsteroSampleResult,
};
use crate::search_methods::{
    build_scan_grid_params, run_powell_driver, run_simplex_driver,
    write_sample_file, write_search_results_table,
};

/// Complete multi-track optimization report returned by [`run_star_astero`].
#[derive(Debug, Clone)]
pub struct AsteroSearchReport {
    /// Optimization driver name (`search_type`).
    pub search_type: String,
    /// Per-track best-fitting sample summaries in evaluation order.
    pub samples: Vec<AsteroSampleResult>,
    /// 0-based index of the minimum-$\chi^2$ sample in `samples`.
    pub best_sample_idx: Option<usize>,
    /// Minimum $\chi^2$ achieved across all evaluated tracks.
    pub best_chi2: f64,
    /// Total Henyey evolutionary steps taken across all tracks.
    pub total_steps: usize,
    /// Termination code of the last evaluated track.
    pub last_termination: String,
    /// Final model of the last evaluated track (for restart parity checks).
    pub last_model: Option<StarModel>,
}

struct AsteroTrackExtras<'a> {
    ctrl: &'a AsteroSearchControls,
    params: AsteroParamSet,
    sample_number: usize,
    case_dir: &'a Path,
    best_sample: Option<AsteroSampleResult>,
    fallback_model: Option<(f64, StarModel)>,
    prev_model: Option<StarModel>,
    entered_warm: bool,
    entered_hot: bool,
    passed_lnuc_limit: bool,
    bad_count: usize,
    eval_error: Option<StarError>,
}

impl AsteroTrackExtras<'_> {
    fn set_max_years_dt(model: &mut StarModel, max_yrs: f64) {
        let yrs = max_yrs.max(1.0);
        model.controls.max_years_for_timestep = yrs;
        model.dt = model.dt.min(yrs * SECYER);
        model.dt_next = model.dt_next.min(yrs * SECYER);
    }

    fn check_too_many_bad(&mut self) -> Option<&'static str> {
        if self.best_sample.is_some() {
            self.bad_count += 1;
            if self.ctrl.limit_num_chi2_too_big > 0
                && self.bad_count > self.ctrl.limit_num_chi2_too_big
            {
                return Some("limit_num_chi2_too_big");
            }
        } else {
            self.bad_count = 0;
        }
        None
    }

    fn check_limits(
        &self,
        model: &StarModel,
        dnu_uhz: f64,
    ) -> Option<&'static str> {
        if self.ctrl.eval_chi2_at_target_age_only
            && model.star_age >= self.ctrl.age_target
        {
            return Some("age_target");
        }
        if self.ctrl.include_age_in_chi2_spectro
            && self
                .ctrl
                .max_age_for_chi2
                .is_some_and(|m| model.star_age >= m)
        {
            return Some("max_age_for_chi2");
        }
        let check1 = |val: f64, tgt: f64, sig: f64, coeff: f64| -> bool {
            if coeff == 0.0 || sig <= 0.0 {
                return false;
            }
            let lim = tgt + sig * coeff;
            (coeff > 0.0 && val > lim) || (coeff < 0.0 && val < lim)
        };
        if self.ctrl.delta_nu > 0.0
            && check1(
                dnu_uhz,
                self.ctrl.delta_nu,
                self.ctrl.delta_nu_sigma,
                self.ctrl.sigmas_coeff_for_delta_nu_limit,
            )
        {
            return Some("delta_nu_limit");
        }
        let hr = extract_history_row(model);
        let x = model.xh_mass.first().copied().unwrap_or(0.70).max(1.0e-10);
        let y = model.yhe_mass.first().copied().unwrap_or(0.28);
        let z_div_x = (1.0 - x - y).clamp(1.0e-99, 1.0) / x;
        let feh = flexmesa_math::log10(
            z_div_x / self.ctrl.z_div_x_solar.max(1.0e-30),
        );
        let r_rsun = model.surface_r_rsun().max(1.0e-30);
        let rcz = eval_convective_envelope_base_rsun(model);
        let checks: [(SpectroConstraint, f64, &'static str); 10] = [
            (self.ctrl.teff, model.teff.max(1.0), "Teff_limit"),
            (self.ctrl.logg, hr.log_g, "logg_limit"),
            (self.ctrl.log_l, hr.log_l, "logL_limit"),
            (self.ctrl.feh, feh, "FeH_limit"),
            (self.ctrl.m_h, feh, "M_H_limit"),
            (self.ctrl.log_r, hr.log_r, "logR_limit"),
            (self.ctrl.radius, r_rsun, "R_limit"),
            (self.ctrl.surface_z_div_x, z_div_x, "surface_Z_div_X_limit"),
            (self.ctrl.surface_he, y, "surface_He_limit"),
            (self.ctrl.rcz, rcz, "Rcz_limit"),
        ];
        for (c, v, name) in checks {
            if check1(v, c.target, c.sigma, c.sigmas_coeff) {
                return Some(name);
            }
        }
        None
    }
}

impl StarExtras for AsteroTrackExtras<'_> {
    fn extras_start_step(&mut self, model: &mut StarModel) -> ExtrasDirective {
        let checking_age = self.ctrl.eval_chi2_at_target_age_only
            || self.ctrl.include_age_in_chi2_spectro;
        if !checking_age && (!self.entered_warm || !self.entered_hot) {
            self.prev_model = Some(model.clone());
        }
        if checking_age && model.star_age < self.ctrl.age_target {
            let rem = (self.ctrl.age_target - model.star_age).max(1.0);
            let cur_max = if model.controls.max_years_for_timestep > 0.0 {
                model.controls.max_years_for_timestep
            } else {
                self.ctrl.max_yrs_dt_when_cold
            }
            .max(1.0);
            if rem <= cur_max {
                Self::set_max_years_dt(model, rem);
            } else if self.ctrl.num_smaller_steps_before_age_target > 0
                && self.ctrl.dt_for_smaller_steps_before_age_target > 0.0
            {
                let n_steps = self.ctrl.num_smaller_steps_before_age_target;
                if ((rem / cur_max + 1.0e-6).floor() as usize) <= n_steps {
                    let dt_sm =
                        self.ctrl.dt_for_smaller_steps_before_age_target;
                    let i = (rem / dt_sm + 1.0e-6)
                        .floor()
                        .clamp(1.0, n_steps as f64);
                    Self::set_max_years_dt(model, rem / i);
                }
            }
        }
        ExtrasDirective::KeepGoing
    }

    fn extras_check_model(&mut self, model: &mut StarModel) -> ExtrasDirective {
        let checking_age = self.ctrl.eval_chi2_at_target_age_only
            || self.ctrl.include_age_in_chi2_spectro;
        if self.ctrl.eval_chi2_at_target_age_only
            && model.star_age < self.ctrl.age_target
        {
            return ExtrasDirective::KeepGoing;
        }
        let (chi2_sp, chi2_dnu, dnu_uhz) =
            eval_spectro_and_delta_nu_chi2(model, self.ctrl);
        if !checking_age {
            let next_dt = if self.entered_hot {
                self.ctrl.max_yrs_dt_when_hot
            } else if self.entered_warm {
                self.ctrl.max_yrs_dt_when_warm
            } else {
                self.ctrl.max_yrs_dt_when_cold
            };
            Self::set_max_years_dt(model, next_dt);
        }
        let hr = extract_history_row(model);
        let l_nuc_div_l = flexmesa_math::exp10(hr.log_lh - hr.log_l);
        if l_nuc_div_l >= self.ctrl.lnuc_div_l_limit {
            self.passed_lnuc_limit = true;
        }
        if model.star_age < self.ctrl.min_age_limit
            || (!self.passed_lnuc_limit
                && l_nuc_div_l < self.ctrl.lnuc_div_l_limit)
            || self
                .ctrl
                .min_age_for_chi2
                .is_some_and(|m| model.star_age < m)
        {
            return ExtrasDirective::KeepGoing;
        }
        if let Some(reason) = self.check_limits(model, dnu_uhz) {
            return ExtrasDirective::Terminate(String::from(reason));
        }
        let pre_score = chi2_sp + chi2_dnu;
        if self
            .fallback_model
            .as_ref()
            .is_none_or(|(best_pre, _)| pre_score < *best_pre)
        {
            self.fallback_model = Some((pre_score, model.clone()));
        }
        if chi2_sp > self.ctrl.chi2_spectroscopic_limit
            || (self.ctrl.delta_nu > 0.0
                && self.ctrl.delta_nu_sigma > 0.0
                && chi2_dnu > self.ctrl.chi2_delta_nu_limit)
        {
            if let Some(reason) = self.check_too_many_bad() {
                return ExtrasDirective::Terminate(String::from(reason));
            }
            return ExtrasDirective::KeepGoing;
        }
        if !checking_age {
            if !self.entered_warm
                && model.dt > 1.05 * self.ctrl.max_yrs_dt_when_warm * SECYER
            {
                self.entered_warm = true;
                if let Some(prev) = self.prev_model.take() {
                    *model = prev;
                }
                Self::set_max_years_dt(model, self.ctrl.max_yrs_dt_when_warm);
                return ExtrasDirective::KeepGoing;
            }
            self.entered_warm = true;
            Self::set_max_years_dt(model, self.ctrl.max_yrs_dt_when_warm);
        }
        let sample = match eval_model_astero_sample(
            model,
            self.ctrl,
            self.params,
            self.sample_number,
            self.case_dir,
        ) {
            Ok(s) => s,
            Err(e) => {
                self.eval_error = Some(e);
                return ExtrasDirective::Terminate(String::from(
                    "AsteroSampleEvalError",
                ));
            }
        };
        if self.ctrl.nl[0] > 0
            && sample.chi2_radial > self.ctrl.chi2_radial_limit
        {
            if !checking_age
                && sample.chi2_radial <= 5.0 * self.ctrl.chi2_radial_limit
            {
                self.entered_hot = true;
                Self::set_max_years_dt(model, self.ctrl.max_yrs_dt_when_hot);
            }
            if let Some(reason) = self.check_too_many_bad() {
                return ExtrasDirective::Terminate(String::from(reason));
            }
            return ExtrasDirective::KeepGoing;
        }
        if !checking_age {
            if !self.entered_hot
                && model.dt > 1.05 * self.ctrl.max_yrs_dt_when_hot * SECYER
            {
                self.entered_hot = true;
                if let Some(prev) = self.prev_model.take() {
                    *model = prev;
                }
                Self::set_max_years_dt(model, self.ctrl.max_yrs_dt_when_hot);
                return ExtrasDirective::KeepGoing;
            }
            self.entered_hot = true;
            let dt_hot = if self.ctrl.chi2_limit_for_small_timesteps > 0.0
                && sample.chi2 <= self.ctrl.chi2_limit_for_small_timesteps
            {
                self.ctrl.max_yrs_dt_chi2_small_limit
            } else {
                self.ctrl.max_yrs_dt_when_hot
            };
            Self::set_max_years_dt(model, dt_hot);
        }
        if self
            .best_sample
            .as_ref()
            .is_none_or(|b| sample.chi2 < b.chi2)
        {
            self.best_sample = Some(sample.clone());
            self.bad_count = 0;
        } else if let Some(best) = &self.best_sample {
            if best.chi2 <= self.ctrl.chi2_search_limit1
                && sample.chi2 >= self.ctrl.chi2_search_limit2
            {
                return ExtrasDirective::Terminate(String::from(
                    "chi2_search_limit2",
                ));
            }
            if sample.chi2 >= self.ctrl.chi2_relative_increase_limit * best.chi2
            {
                if let Some(reason) = self.check_too_many_bad() {
                    return ExtrasDirective::Terminate(String::from(reason));
                }
            } else {
                self.bad_count = 0;
            }
        }
        if let Some(reason) = self.check_limits(model, sample.delta_nu_uhz) {
            return ExtrasDirective::Terminate(String::from(reason));
        }
        ExtrasDirective::KeepGoing
    }
}

fn apply_params_to_bundle(
    bundle: &mut InlistBundle,
    ctrl: &AsteroSearchControls,
    params: &AsteroParamSet,
) -> Result<(), StarError> {
    bundle.controls.initial_mass = params.initial_mass;
    let z_div_x = ctrl.z_div_x_solar * flexmesa_math::exp10(params.initial_feh);
    let (x, y) = if ctrl.y_depends_on_z {
        let c = 1.0 + z_div_x * (1.0 + ctrl.dydz);
        (
            (1.0 - ctrl.y0) / c,
            (ctrl.y0 + z_div_x * (ctrl.dydz + ctrl.y0)) / c,
        )
    } else {
        ((1.0 - params.initial_y) / (1.0 + z_div_x), params.initial_y)
    };
    let z = (1.0 - x - y).clamp(1.0e-4, 0.08);
    (bundle.controls.initial_y, bundle.controls.initial_z) = (y, z);
    (bundle.controls.zbase, bundle.kap_config.zbase) = (z, z);
    (bundle.job.relax_y, bundle.job.new_y) = (true, y);
    (bundle.job.relax_z, bundle.job.new_z) = (true, z);
    if params.alpha > 0.0 {
        bundle.controls.mixing_length_alpha = params.alpha;
    }
    let ov = &mut bundle.controls.mix_wind.overshoot_sets[0];
    if params.f_ov > 0.0 {
        if ctrl.f0_ov_div_f_ov <= 0.0 {
            return Err(StarError::InvalidInput(format!(
                "astero f_ov = {} > 0 requires f0_ov_div_f_ov > 0 (got {})",
                params.f_ov, ctrl.f0_ov_div_f_ov
            )));
        }
        ov.scheme = String::from("exponential");
        ov.zone_type = String::from("any");
        ov.zone_loc = String::from("any");
        ov.bdy_loc = String::from("any");
        ov.f = params.f_ov;
        ov.f0 = ctrl.f0_ov_div_f_ov * params.f_ov;
    } else {
        ov.scheme.clear();
        (ov.f, ov.f0) = (0.0, 0.0);
    }
    if ctrl.has_custom_param[0] {
        bundle.controls.alpha_semiconvection = params.my_param1.max(0.0);
    }
    if ctrl.has_custom_param[1] {
        bundle.controls.thermohaline_coeff = params.my_param2.max(0.0);
    }
    if ctrl.has_custom_param[2] {
        bundle.controls.do_element_diffusion = params.my_param3 > 0.0;
    }
    if !ctrl.eval_chi2_at_target_age_only && !ctrl.include_age_in_chi2_spectro {
        bundle.controls.max_years_for_timestep =
            ctrl.max_yrs_dt_when_cold.max(1.0);
    }
    Ok(())
}

/// Evolves one stellar track for `params` and returns its best [`AsteroSampleResult`].
pub fn run_single_astero_track(
    base_bundle: &InlistBundle,
    ctrl: &AsteroSearchControls,
    params: AsteroParamSet,
    sample_number: usize,
    max_steps_cap: Option<usize>,
    case_dir: &Path,
) -> Result<(AsteroSampleResult, StageRunReport, StarModel), StarError> {
    run_single_astero_track_with_logs(
        base_bundle,
        ctrl,
        params,
        sample_number,
        max_steps_cap,
        case_dir,
        StarLogsOptions::default(),
    )
}

/// Evolves one stellar track for `params`, optionally streaming `LOGS/history.data` (`#143`).
pub fn run_single_astero_track_with_logs(
    base_bundle: &InlistBundle,
    ctrl: &AsteroSearchControls,
    params: AsteroParamSet,
    sample_number: usize,
    max_steps_cap: Option<usize>,
    case_dir: &Path,
    logs_opts: StarLogsOptions<'_>,
) -> Result<(AsteroSampleResult, StageRunReport, StarModel), StarError> {
    let mut bundle = base_bundle.clone();
    apply_params_to_bundle(&mut bundle, ctrl, &params)?;
    if let Some(cap) = max_steps_cap {
        bundle.controls.max_model_number =
            Some(bundle.controls.max_model_number.map_or(cap, |m| m.min(cap)));
    }
    let mut extras = AsteroTrackExtras {
        ctrl,
        params,
        sample_number,
        case_dir,
        best_sample: None,
        fallback_model: None,
        prev_model: None,
        entered_warm: false,
        entered_hot: false,
        passed_lnuc_limit: false,
        bad_count: 0,
        eval_error: None,
    };
    let mut model =
        init_or_update_model_from_bundle(&bundle, None, &mut extras)?;
    let mut logs_writer = logs_opts.build_writer(&model, case_dir, None)?;
    let stage = run_inlist_stage_with_logs(
        &mut model,
        &bundle,
        &mut extras,
        max_steps_cap,
        logs_writer.as_mut(),
    )?;
    if let Some(err) = extras.eval_error.take() {
        return Err(err);
    }
    let best = if let Some(s) = extras.best_sample {
        s
    } else {
        let fb_mod = extras.fallback_model.as_ref().map_or(&model, |(_, m)| m);
        eval_model_astero_sample(fb_mod, ctrl, params, sample_number, case_dir)?
    };
    Ok((best, stage, model))
}

/// Runs the multi-track asteroseismic search driver (`astero_run_support.f90:36-191`).
pub fn run_star_astero(
    case_dir: &Path,
    entry_inlist: &str,
    max_steps_per_stage: Option<usize>,
    ctrl: &AsteroSearchControls,
) -> Result<AsteroSearchReport, StarError> {
    run_star_astero_with_logs(
        case_dir,
        entry_inlist,
        max_steps_per_stage,
        ctrl,
        StarLogsOptions::default(),
    )
}

/// Runs the multi-track asteroseismic search driver while streaming `LOGS/` and
/// search results into `logs_opts.output_root` when requested (`#143`).
pub fn run_star_astero_with_logs(
    case_dir: &Path,
    entry_inlist: &str,
    max_steps_per_stage: Option<usize>,
    ctrl: &AsteroSearchControls,
    logs_opts: StarLogsOptions<'_>,
) -> Result<AsteroSearchReport, StarError> {
    run_star_astero_with_options(
        case_dir,
        entry_inlist,
        max_steps_per_stage,
        ctrl,
        logs_opts,
        InlistLoadOptions::default(),
    )
}

/// Runs the multi-track asteroseismic search driver with explicit [`InlistLoadOptions`].
pub fn run_star_astero_with_options(
    case_dir: &Path,
    entry_inlist: &str,
    max_steps_per_stage: Option<usize>,
    ctrl: &AsteroSearchControls,
    logs_opts: StarLogsOptions<'_>,
    inlist_opts: InlistLoadOptions,
) -> Result<AsteroSearchReport, StarError> {
    let base_bundle = InlistBundle::from_file_in_case_dir_with_options(
        case_dir,
        entry_inlist,
        inlist_opts,
    )?;
    let results_base = logs_opts.output_root.unwrap_or(case_dir);
    let out_dir: PathBuf = results_base.join(&ctrl.astero_results_directory);
    fs::create_dir_all(&out_dir).map_err(|e| {
        StarError::IoError(format!(
            "failed to create astero results directory '{}': {e}",
            out_dir.display()
        ))
    })?;

    let mut smoke_ctrl;
    let active_ctrl = if max_steps_per_stage.is_some() {
        smoke_ctrl = ctrl.clone();
        (smoke_ctrl.simplex_itermax, smoke_ctrl.simplex_fcn_calls_max) = (1, 2);
        (
            smoke_ctrl.newuoa_max_fcn_calls,
            smoke_ctrl.bobyqa_max_fcn_calls,
        ) = (2, 2);
        &smoke_ctrl
    } else {
        ctrl
    };
    let track_cap = if max_steps_per_stage.is_some() {
        1
    } else {
        usize::MAX
    };
    let want_logs = logs_opts.output_root.is_some()
        || logs_opts.explicit_history_file.is_some();
    let mut best_chi2_seen = f64::INFINITY;
    let (mut samples, mut total_steps) = (Vec::new(), 0usize);
    let (mut last_termination, mut last_model) = (String::new(), None);
    let mut eval_param_set = |p: AsteroParamSet| -> Result<f64, StarError> {
        if samples.len() >= track_cap {
            return Ok(1.0e99);
        }
        let s_num = samples.len() + 1;
        let sample_hist_path =
            out_dir.join(format!("sample_{s_num:04}_history.data"));
        let trial_logs = if want_logs {
            StarLogsOptions {
                output_root: None,
                explicit_history_file: Some(sample_hist_path.as_path()),
                history_interval_override: logs_opts.history_interval_override,
                write_profiles_override: Some(false),
            }
        } else {
            StarLogsOptions::default()
        };
        let (sample, stage, m) = run_single_astero_track_with_logs(
            &base_bundle,
            active_ctrl,
            p,
            s_num,
            max_steps_per_stage,
            case_dir,
            trial_logs,
        )?;
        total_steps += stage.steps_taken;
        last_termination = stage.termination_code;
        let c2 = sample.chi2;
        if want_logs
            && (c2 < best_chi2_seen || samples.is_empty())
            && sample_hist_path.is_file()
        {
            best_chi2_seen = best_chi2_seen.min(c2);
            if let Some(exp_dst) = logs_opts.explicit_history_file {
                if let Some(parent) = exp_dst.parent() {
                    fs::create_dir_all(parent).map_err(|e| {
                        StarError::IoError(format!(
                            "failed to create history dir '{}': {e}",
                            parent.display()
                        ))
                    })?;
                }
                fs::copy(&sample_hist_path, exp_dst).map_err(|e| {
                    StarError::IoError(format!(
                        "failed copying history to '{}': {e}",
                        exp_dst.display()
                    ))
                })?;
            } else if let Some(out_root) = logs_opts.output_root {
                let dst_dir = out_root.join(&m.controls.output.log_directory);
                fs::create_dir_all(&dst_dir).map_err(|e| {
                    StarError::IoError(format!(
                        "failed creating log dir '{}': {e}",
                        dst_dir.display()
                    ))
                })?;
                let dst_hist =
                    dst_dir.join(&m.controls.output.star_history_name);
                fs::copy(&sample_hist_path, &dst_hist).map_err(|e| {
                    StarError::IoError(format!(
                        "failed copying history to '{}': {e}",
                        dst_hist.display()
                    ))
                })?;
            }
        }
        last_model = Some(m);
        write_sample_file(&out_dir, &sample)?;
        samples.push(sample);
        Ok(c2)
    };

    let st = active_ctrl.search_type.trim().to_ascii_lowercase();
    let results_fname = match st.as_str() {
        "from_file" => {
            for p in active_ctrl
                .read_parameters_from_file(case_dir)?
                .into_iter()
                .take(track_cap)
            {
                let _ = eval_param_set(p)?;
            }
            &active_ctrl.from_file_output_filename
        }
        "scan_grid" => {
            for p in build_scan_grid_params(active_ctrl)
                .into_iter()
                .take(track_cap)
            {
                let _ = eval_param_set(p)?;
            }
            &active_ctrl.scan_grid_output_filename
        }
        "simplex" => {
            run_simplex_driver(active_ctrl, &mut eval_param_set)?;
            &active_ctrl.simplex_output_filename
        }
        "newuoa" | "bobyqa" => {
            run_powell_driver(
                active_ctrl,
                st == "bobyqa",
                &mut eval_param_set,
            )?;
            if st == "bobyqa" {
                &active_ctrl.bobyqa_output_filename
            } else {
                &active_ctrl.newuoa_output_filename
            }
        }
        "use_first_values" => {
            let _ = eval_param_set(active_ctrl.first_params())?;
            &active_ctrl.scan_grid_output_filename
        }
        other => {
            return Err(StarError::InvalidInput(format!(
                "unrecognized astero search_type '{other}'"
            )));
        }
    };
    write_search_results_table(&out_dir.join(results_fname), &samples)?;
    let best_sample_idx = samples
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.chi2.total_cmp(&b.chi2))
        .map(|(i, _)| i);
    let best_chi2 = best_sample_idx.map_or(f64::INFINITY, |i| samples[i].chi2);
    Ok(AsteroSearchReport {
        search_type: ctrl.search_type.clone(),
        samples,
        best_sample_idx,
        best_chi2,
        total_steps,
        last_termination,
        last_model,
    })
}
