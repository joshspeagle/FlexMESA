// SPDX-License-Identifier: GPL-3.0-or-later
//! Multi-track asteroseismic search driver (`run_star_astero`) and adaptive
//! evolutionary timestep controller (`extras_support.f90:58-669`,
//! `astero_run_support.f90:36-850`).

use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_const::SECYER;
use flexmesa_star::controls::StarError;
use flexmesa_star::core_boundaries::compute_power_categories;
use flexmesa_star::history::extract_history_row;
use flexmesa_star::inlist::{InlistBundle, InlistLoadOptions};
use flexmesa_star::inlist_runner::{
    run_inlist_stage_with_logs, ExtrasDirective, StageRunReport, StarExtras,
    StarLogsOptions,
};
use flexmesa_star::model::StarModel;
use flexmesa_star::report_utils::eval_photosphere_info;
use flexmesa_star::star_job_init::init_or_update_model_from_bundle;

use crate::inlist_astero::{
    AsteroParamSet, AsteroSearchControls, MAX_CONSTRAINTS,
};
use crate::sample_eval::{
    eval_model_astero_sample, eval_spectro_and_delta_nu_chi2,
    AsteroSampleResult,
};
use crate::search_methods::{
    build_scan_grid_params, run_powell_driver, run_simplex_driver,
    write_sample_file, write_search_results_table,
};

/// Sentinel $\chi^2$ returned to optimization drivers when an evolutionary
/// track finishes without accepting any model (`simplex_f = 1d99` in
/// `astero/private/astero_run_support.f90:745,751`).
pub(crate) const FAILED_TRACK_CHI2: f64 = 1.0e99;

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
    /// Termination code of the track that achieved `best_chi2`, if any.
    pub best_termination: Option<String>,
    /// Termination code of the last evaluated track.
    pub last_termination: String,
    /// Final model of the last evaluated track (for restart parity checks).
    pub last_model: Option<StarModel>,
}

struct AsteroTrackExtras<'a, 'h> {
    ctrl: &'a AsteroSearchControls,
    /// The case's own `set_constraint_value` (and `set_param`).
    hooks: Option<&'h mut dyn AsteroCaseHooks>,
    params: AsteroParamSet,
    sample_number: usize,
    case_dir: &'a Path,
    best_sample: Option<AsteroSampleResult>,
    /// Averages over the best samples of earlier tracks
    /// (`set_sample_averages`).
    averages: SampleAverages,
    /// `num_chi2_too_big`.
    bad_count: usize,
    eval_error: Option<StarError>,
}

/// Averages of age and model number over the best-$\chi^2$ samples of the
/// tracks evaluated so far (`set_sample_averages`,
/// `astero/private/astero_run_support.f90:1045-1102`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct SampleAverages {
    /// `avg_age_top_samples` (yr).
    pub(crate) age: f64,
    /// `avg_age_sigma` (yr).
    pub(crate) age_sigma: f64,
    /// `avg_model_number_top_samples`.
    pub(crate) model_number: f64,
    /// `avg_model_number_sigma`.
    pub(crate) model_number_sigma: f64,
}

impl SampleAverages {
    /// Too few samples for averages: every limit is `1d99` (`:1053-1058`).
    pub(crate) const NONE: Self = Self {
        age: 1.0e99,
        age_sigma: 1.0e99,
        model_number: 1.0e99,
        model_number_sigma: 1.0e99,
    };

    /// `set_sample_averages` over `samples` in evaluation order.
    pub(crate) fn from_samples(
        samples: &[AsteroSampleResult],
        ctrl: &AsteroSearchControls,
    ) -> Self {
        let n = samples.len().min(ctrl.max_num_samples_for_avg);
        if n < ctrl.min_num_samples_for_avg.max(2) {
            return Self::NONE;
        }
        // `set_sample_index_by_chi2`: samples in increasing chi^2.
        let mut order: Vec<usize> = (0..samples.len()).collect();
        order.sort_by(|&a, &b| samples[a].chi2.total_cmp(&samples[b].chi2));
        let (mut age, mut age2, mut mn, mut mn2) = (0.0, 0.0, 0.0, 0.0);
        for &j in &order[..n] {
            let a = samples[j].age_yr;
            let m = samples[j].model_number as f64;
            age += a;
            age2 += a * a;
            mn += m;
            mn2 += m * m;
        }
        let nf = n as f64;
        let age_sigma = ((age2 - age * age / nf) / (nf - 1.0)).max(0.0).sqrt();
        let model_number_sigma =
            ((mn2 - mn * mn / nf) / (nf - 1.0)).max(0.0).sqrt();
        Self {
            age: age / nf,
            age_sigma,
            model_number: mn / nf,
            model_number_sigma,
        }
    }
}

impl AsteroTrackExtras<'_, '_> {
    fn set_max_years_dt(model: &mut StarModel, max_yrs: f64) {
        let yrs = max_yrs.max(1.0);
        model.controls.max_years_for_timestep = yrs;
        model.dt = model.dt.min(yrs * SECYER);
        model.dt_next = model.dt_next.min(yrs * SECYER);
    }

    /// Redo the step with `dt = max_yrs` and hold the timestep for 10
    /// models (`s% dt = max_yrs*secyer; s% timestep_hold = model_number +
    /// 10`, `astero/private/extras_support.f90:301-308`).
    fn redo_with_max_years(
        model: &mut StarModel,
        max_yrs: f64,
    ) -> ExtrasDirective {
        model.dt = max_yrs * SECYER;
        model.aux.step_hist.timestep_hold = model.model_number + 10;
        ExtrasDirective::Redo
    }

    /// Frequencies, ratios and chi^2 of `model`; an evaluation error
    /// terminates the track and is reported by the caller.
    fn eval_sample(
        &mut self,
        model: &StarModel,
        values: &[f64; MAX_CONSTRAINTS],
    ) -> Result<AsteroSampleResult, ExtrasDirective> {
        eval_model_astero_sample(
            model,
            self.ctrl,
            values,
            self.params,
            self.sample_number,
            self.case_dir,
        )
        .map_err(|e| {
            self.eval_error = Some(e);
            ExtrasDirective::Terminate(String::from("AsteroSampleEvalError"))
        })
    }

    /// `constraint_value(i)` for every named constraint, from the case's
    /// `set_constraint_value` (`astero/private/extras_support.f90:204-209`);
    /// unnamed slots are `NaN` and never read.
    fn eval_constraint_values(
        &mut self,
        model: &StarModel,
    ) -> Result<[f64; MAX_CONSTRAINTS], StarError> {
        let mut values = [f64::NAN; MAX_CONSTRAINTS];
        for (i, raw) in self.ctrl.constraint_names.iter().enumerate() {
            let name = raw.trim();
            if name.is_empty() {
                continue;
            }
            let hook = self.hooks.as_deref_mut().ok_or_else(|| {
                StarError::InvalidInput(format!(
                    "astero constraint '{name}' needs the case's \
                     set_constraint_value from its run_star_extras.f90 \
                     (astero/private/extras_support.f90:204-209), but no \
                     case hook was supplied"
                ))
            })?;
            values[i] = hook.set_constraint_value(model, self.ctrl, name)?;
        }
        Ok(values)
    }

    /// `check_too_many_bad` (`astero/private/extras_support.f90:492-502`).
    fn check_too_many_bad(&mut self) -> ExtrasDirective {
        if self.best_sample.as_ref().is_some_and(|b| b.chi2 > 0.0) {
            self.bad_count += 1;
            if self.bad_count > self.ctrl.limit_num_chi2_too_big {
                return ExtrasDirective::Terminate(String::from(
                    "limit_num_chi2_too_big",
                ));
            }
            return ExtrasDirective::KeepGoing;
        }
        self.bad_count = 0;
        ExtrasDirective::KeepGoing
    }

    /// `final_checks` age stops (`astero/private/extras_support.f90:506-513`),
    /// tested after the model's chi^2 has been evaluated and the best sample
    /// saved (`:491-495`).
    fn final_age_stop(&self, model: &StarModel) -> Option<&'static str> {
        if self.ctrl.include_age_in_chi2_spectro
            && self
                .ctrl
                .max_age_for_chi2
                .is_some_and(|m| model.star_age >= m)
        {
            return Some("max_age_for_chi2");
        }
        if self.ctrl.eval_chi2_at_target_age_only
            && model.star_age >= self.ctrl.age_target
        {
            return Some("age_target");
        }
        None
    }

    /// `save_best_info` followed by `final_checks`
    /// (`astero/private/extras_support.f90:481-486, 505-531`).
    fn save_best_and_final_checks(
        &mut self,
        model: &StarModel,
        sample: &AsteroSampleResult,
    ) -> ExtrasDirective {
        // `if (best_chi2 <= 0 .or. chi2 < best_chi2) call save_best_info(s)`.
        if self
            .best_sample
            .as_ref()
            .is_none_or(|b| b.chi2 <= 0.0 || sample.chi2 < b.chi2)
        {
            self.best_sample = Some(sample.clone());
        }
        // `final_checks`: the age stops set `terminate` and the chi^2 checks
        // below still run.
        let mut directive = match self.final_age_stop(model) {
            Some(reason) => ExtrasDirective::Terminate(String::from(reason)),
            None => ExtrasDirective::KeepGoing,
        };
        let Some(best_chi2) = self
            .best_sample
            .as_ref()
            .map(|b| b.chi2)
            .filter(|&c| c > 0.0)
        else {
            return directive;
        };
        if best_chi2 <= self.ctrl.chi2_search_limit1
            && sample.chi2 >= self.ctrl.chi2_search_limit2
        {
            return ExtrasDirective::Terminate(String::from(
                "chi2_search_limit2",
            ));
        }
        if sample.chi2 >= self.ctrl.chi2_relative_increase_limit * best_chi2 {
            self.bad_count += 1;
            if self.bad_count > self.ctrl.limit_num_chi2_too_big {
                directive = ExtrasDirective::Terminate(String::from(
                    "limit_num_chi2_too_big",
                ));
            }
            return directive;
        }
        self.bad_count = 0;
        directive
    }

    fn check_limits(
        &self,
        dnu_uhz: f64,
        values: &[f64; MAX_CONSTRAINTS],
    ) -> Option<String> {
        // Upstream `check_limits` tests only the `delta_nu` and constraint
        // limits (`astero/private/extras_support.f90:621-667`). The age stops
        // are in `final_checks`, after the model's chi^2 has been saved; see
        // `final_age_stop`.
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
            return Some(String::from("delta_nu_limit"));
        }
        // `do i = 1, max_constraints` (`extras_support.f90:645-663`).
        for (i, c) in self.ctrl.indexed_constraints.iter().enumerate() {
            let name = self.ctrl.constraint_names[i].trim();
            if name.is_empty() {
                continue;
            }
            if check1(values[i], c.target, c.sigma, c.sigmas_coeff) {
                return Some(format!("{name}_limit"));
            }
        }
        None
    }
}

impl StarExtras for AsteroTrackExtras<'_, '_> {
    fn extras_start_step(&mut self, model: &mut StarModel) -> ExtrasDirective {
        let checking_age = self.ctrl.eval_chi2_at_target_age_only
            || self.ctrl.include_age_in_chi2_spectro;
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
        // `do_astero_extras_check_model`
        // (`astero/private/extras_support.f90:56-669`), in upstream's order.
        let checking_age = self.ctrl.eval_chi2_at_target_age_only
            || self.ctrl.include_age_in_chi2_spectro;
        if self.ctrl.eval_chi2_at_target_age_only
            && model.star_age < self.ctrl.age_target
        {
            return ExtrasDirective::KeepGoing;
        }
        if !checking_age {
            // `:147-150`: every model starts "cold"; the limits below raise
            // the timestep class again when the model passes them.
            if self
                .ctrl
                .min_age_for_chi2
                .is_some_and(|m| model.star_age < m)
            {
                return ExtrasDirective::KeepGoing;
            }
            model.controls.max_years_for_timestep =
                self.ctrl.max_yrs_dt_when_cold;
        } else if self.ctrl.include_age_in_chi2_spectro
            && self
                .ctrl
                .min_age_for_chi2
                .is_some_and(|m| model.star_age < m)
        {
            // `:152`.
            return ExtrasDirective::KeepGoing;
        }

        // `:172-175`: `L_nuc_burn_total < L_phot*Lnuc_div_L_limit`.
        let powers = compute_power_categories(model);
        let phot = eval_photosphere_info(model);
        if powers.power_nuc_burn < phot.l_lsun * self.ctrl.lnuc_div_l_limit
            || model.star_age < self.ctrl.min_age_limit
        {
            return ExtrasDirective::KeepGoing;
        }

        if !checking_age {
            // `:177-200`: stop a track that is older, or has more models,
            // than the best samples so far by the configured margins.
            let age_limit = self.averages.age
                + self.ctrl.avg_age_sigma_limit * self.averages.age_sigma;
            if model.star_age > age_limit {
                return ExtrasDirective::Terminate(String::from(
                    "star age > limit from top samples",
                ));
            }
            let model_limit = self.averages.model_number
                + self.ctrl.avg_model_number_sigma_limit
                    * self.averages.model_number_sigma;
            if (model.model_number as f64) > model_limit {
                return ExtrasDirective::Terminate(String::from(
                    "model number > limit from top samples",
                ));
            }
        }

        // Constraint values are set before `check_limits` (`:204-212`).
        let values = match self.eval_constraint_values(model) {
            Ok(v) => v,
            Err(e) => {
                self.eval_error = Some(e);
                return ExtrasDirective::Terminate(String::from(
                    "AsteroConstraintError",
                ));
            }
        };
        let (chi2_sp, chi2_dnu, dnu_uhz) =
            eval_spectro_and_delta_nu_chi2(model, self.ctrl, &values);
        if let Some(reason) = self.check_limits(dnu_uhz, &values) {
            return ExtrasDirective::Terminate(reason);
        }
        // `:214-225`: a bad `chi2_spectro` terminates the track.
        if !chi2_sp.is_finite() {
            return ExtrasDirective::Terminate(String::from(
                "bad chi2_spectro",
            ));
        }

        // `chi2_seismo_fraction <= 0` (`:266-280`): no frequencies are
        // needed, and the model is scored and saved before, and without, the
        // spectroscopic and `delta_nu` limits below.
        if self.ctrl.weights.chi2_seismo_fraction <= 0.0 {
            let sample = match self.eval_sample(model, &values) {
                Ok(s) => s,
                Err(dir) => return dir,
            };
            let directive = self.save_best_and_final_checks(model, &sample);
            if sample.chi2 < self.ctrl.chi2_radial_limit && !checking_age {
                model.controls.max_years_for_timestep =
                    self.ctrl.max_yrs_dt_when_warm;
            }
            return directive;
        }

        // `:282-296`: a bad model counts towards `limit_num_chi2_too_big`.
        if chi2_sp > self.ctrl.chi2_spectroscopic_limit
            || chi2_dnu > self.ctrl.chi2_delta_nu_limit
        {
            return self.check_too_many_bad();
        }

        // `:299-322`: passing the spectroscopic and `delta_nu` limits makes
        // the model "warm" and then "hot"; a step longer than either limit
        // is redone with that limit.
        if !checking_age {
            for max_yrs in [
                self.ctrl.max_yrs_dt_when_warm,
                self.ctrl.max_yrs_dt_when_hot,
            ] {
                model.controls.max_years_for_timestep = max_yrs;
                if model.dt > max_yrs * SECYER {
                    return Self::redo_with_max_years(model, max_yrs);
                }
            }
        }

        // `:324-444`: frequencies, ratios and chi^2. A model whose
        // required modes cannot all be found (or whose radial chi^2 exceeds
        // `chi2_radial_limit` while non-radial modes are required,
        // `get_radial`, `:590-594`) counts as a bad model.
        let sample = match self.eval_sample(model, &values) {
            Ok(s) => s,
            Err(dir) => return dir,
        };
        if sample.chi2 >= FAILED_TRACK_CHI2 {
            return self.check_too_many_bad();
        }

        // `:445-479`: smaller timesteps as chi^2 falls.
        if !checking_age {
            for (chi2_limit, max_yrs) in [
                (
                    self.ctrl.chi2_limit_for_smallest_timesteps,
                    self.ctrl.max_yrs_dt_chi2_smallest_limit,
                ),
                (
                    self.ctrl.chi2_limit_for_smaller_timesteps,
                    self.ctrl.max_yrs_dt_chi2_smaller_limit,
                ),
                (
                    self.ctrl.chi2_limit_for_small_timesteps,
                    self.ctrl.max_yrs_dt_chi2_small_limit,
                ),
            ] {
                if sample.chi2 <= chi2_limit {
                    model.controls.max_years_for_timestep = max_yrs;
                    if model.dt > max_yrs * SECYER {
                        return Self::redo_with_max_years(model, max_yrs);
                    }
                    break;
                }
            }
        }

        self.save_best_and_final_checks(model, &sample)
    }
}

/// A case's own `set_param` and `set_constraint_value` from its
/// `run_star_extras.f90` (for example
/// `astero/test_suite/fast_newuoa/src/run_star_extras.f90:58-184`).
///
/// Upstream `astero_extras_controls` calls `set_param` for every named search
/// parameter, in index order, before the model is loaded or built
/// (`astero/private/extras_support.f90:974-988`). Both routines are case
/// code, so they are supplied by the caller (test-side extras) and never by
/// this crate.
pub trait AsteroCaseHooks {
    /// Applies parameter `name` (as written in `param_name(i)`) with value
    /// `val` to the run's `&star_job` and `&controls` in `bundle`.
    ///
    /// # Errors
    /// Returns [`StarError`] when the case's `set_param` reports `ierr /= 0`
    /// (for example an unknown `name`).
    fn set_param(
        &mut self,
        bundle: &mut InlistBundle,
        ctrl: &AsteroSearchControls,
        name: &str,
        val: f64,
    ) -> Result<(), StarError>;

    /// The case's `set_constraint_value` (for example
    /// `astero/test_suite/fast_newuoa/src/run_star_extras.f90:58-112`):
    /// the current value of constraint `name` (as written in
    /// `constraint_name(i)`) for `model`. Upstream calls it for every named
    /// constraint before `check_limits` (`extras_support.f90:204-209`).
    ///
    /// # Errors
    /// Returns [`StarError`] when the case's routine fails (`ierr /= 0`).
    fn set_constraint_value(
        &mut self,
        model: &StarModel,
        ctrl: &AsteroSearchControls,
        name: &str,
    ) -> Result<f64, StarError>;
}

/// Value of the search parameter named `name` in `params`, by the slot the
/// inlist parser assigned to that name (`inlist_parser.rs`).
fn param_value(params: &AsteroParamSet, name: &str) -> Option<f64> {
    Some(match name.to_ascii_lowercase().as_str() {
        "initial_mass" | "mass" => params.initial_mass,
        "initial_y" | "y" => params.initial_y,
        "initial_feh" | "feh" => params.initial_feh,
        "alpha" => params.alpha,
        "f_ov" => params.f_ov,
        "my_param1" | "semiconv" => params.my_param1,
        "my_param2" | "thermohal" => params.my_param2,
        "my_param3" | "diffusion" => params.my_param3,
        _ => return None,
    })
}

fn apply_params_to_bundle(
    bundle: &mut InlistBundle,
    ctrl: &AsteroSearchControls,
    params: &AsteroParamSet,
    mut set_param: Option<&mut dyn AsteroCaseHooks>,
) -> Result<(), StarError> {
    // `astero_extras_controls` calls the case's `set_param` for every named
    // parameter in index order (`astero/private/extras_support.f90:974-988`).
    for raw in &ctrl.param_names {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        let val = param_value(params, name).ok_or_else(|| {
            StarError::InvalidInput(format!(
                "astero param_name '{name}' has no parameter slot"
            ))
        })?;
        let hook = set_param.as_deref_mut().ok_or_else(|| {
            StarError::InvalidInput(format!(
                "astero search parameter '{name}' needs the case's set_param \
                 from its run_star_extras.f90 \
                 (astero/private/extras_support.f90:974-988), but no \
                 set_param hook was supplied"
            ))
        })?;
        hook.set_param(bundle, ctrl, name, val)?;
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

/// Builds a zeroed failed-track sample sentinel with `chi2 = 1.0e99`
/// matching upstream `zero_best_info` and `simplex_f = 1d99` when a track
/// finishes without accepting any model (`astero_run_support.f90:241-246,749-753`).
pub(crate) fn failed_sample_result(
    model: &StarModel,
    params: AsteroParamSet,
    sample_number: usize,
) -> AsteroSampleResult {
    let hr = extract_history_row(model);
    AsteroSampleResult {
        sample_number,
        params,
        model_number: model.model_number,
        age_yr: model.star_age,
        teff: model.teff.max(1.0),
        logg: hr.log_g,
        log_l: hr.log_l,
        log_r: hr.log_r,
        constraint_values: [f64::NAN; MAX_CONSTRAINTS],
        delta_nu_uhz: 0.0,
        nu_max_uhz: 0.0,
        a_div_r: 0.0,
        correction_r: 0.0,
        chi2: FAILED_TRACK_CHI2,
        chi2_seismo: FAILED_TRACK_CHI2,
        chi2_spectro: FAILED_TRACK_CHI2,
        chi2_radial: FAILED_TRACK_CHI2,
        chi2_r010: 0.0,
        chi2_r02: 0.0,
        model_freqs: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        model_freqs_corr: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        model_inertias: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        model_orders: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
    }
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
    run_single_astero_track_with_case_hooks(
        base_bundle,
        ctrl,
        params,
        sample_number,
        max_steps_cap,
        case_dir,
        logs_opts,
        None,
    )
}

/// [`run_single_astero_track_with_logs`] with the case's `set_param` hook
/// ([`AsteroCaseHooks`]).
///
/// # Errors
/// Returns [`StarError`] when a named search parameter has no hook, the hook
/// fails, or the track fails.
#[allow(clippy::too_many_arguments)]
pub fn run_single_astero_track_with_case_hooks(
    base_bundle: &InlistBundle,
    ctrl: &AsteroSearchControls,
    params: AsteroParamSet,
    sample_number: usize,
    max_steps_cap: Option<usize>,
    case_dir: &Path,
    logs_opts: StarLogsOptions<'_>,
    set_param: Option<&mut dyn AsteroCaseHooks>,
) -> Result<(AsteroSampleResult, StageRunReport, StarModel), StarError> {
    run_track(
        base_bundle,
        ctrl,
        params,
        sample_number,
        max_steps_cap,
        case_dir,
        logs_opts,
        set_param,
        SampleAverages::NONE,
    )
}

/// One track of a search, with the sample averages of the tracks before it
/// (`eval1`, `astero/private/astero_run_support.f90:193-256`).
#[allow(clippy::too_many_arguments)]
fn run_track(
    base_bundle: &InlistBundle,
    ctrl: &AsteroSearchControls,
    params: AsteroParamSet,
    sample_number: usize,
    max_steps_cap: Option<usize>,
    case_dir: &Path,
    logs_opts: StarLogsOptions<'_>,
    mut set_param: Option<&mut dyn AsteroCaseHooks>,
    averages: SampleAverages,
) -> Result<(AsteroSampleResult, StageRunReport, StarModel), StarError> {
    let mut bundle = base_bundle.clone();
    apply_params_to_bundle(
        &mut bundle,
        ctrl,
        &params,
        set_param
            .as_mut()
            .map(|h| &mut **h as &mut dyn AsteroCaseHooks),
    )?;
    if let Some(cap) = max_steps_cap {
        bundle.controls.max_model_number =
            Some(bundle.controls.max_model_number.map_or(cap, |m| m.min(cap)));
    }
    let mut extras = AsteroTrackExtras {
        ctrl,
        hooks: set_param,
        params,
        sample_number,
        case_dir,
        best_sample: None,
        averages,
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
    let best = extras
        .best_sample
        .unwrap_or_else(|| failed_sample_result(&model, params, sample_number));
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
    run_star_astero_with_case_hooks(
        case_dir,
        entry_inlist,
        max_steps_per_stage,
        ctrl,
        logs_opts,
        inlist_opts,
        None,
    )
}

/// [`run_star_astero_with_options`] with the case's `set_param` hook
/// ([`AsteroCaseHooks`]), called for every track.
///
/// # Errors
/// Returns [`StarError`] when a named search parameter has no hook, the hook
/// fails, or a track fails.
pub fn run_star_astero_with_case_hooks(
    case_dir: &Path,
    entry_inlist: &str,
    max_steps_per_stage: Option<usize>,
    ctrl: &AsteroSearchControls,
    logs_opts: StarLogsOptions<'_>,
    inlist_opts: InlistLoadOptions,
    mut set_param: Option<&mut dyn AsteroCaseHooks>,
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
    let mut best_termination: Option<String> = None;
    let (mut samples, mut total_steps, mut tracks_evaluated) =
        (Vec::new(), 0usize, 0usize);
    let (mut last_termination, mut last_model) = (String::new(), None);
    let mut eval_param_set = |p: AsteroParamSet| -> Result<f64, StarError> {
        if tracks_evaluated >= track_cap {
            return Ok(FAILED_TRACK_CHI2);
        }
        tracks_evaluated += 1;
        let next_sample_num = samples.len() + 1;
        let sample_hist_path =
            out_dir.join(format!("sample_{tracks_evaluated:04}_history.data"));
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
        let (mut sample, stage, m) = run_track(
            &base_bundle,
            active_ctrl,
            p,
            next_sample_num,
            max_steps_per_stage,
            case_dir,
            trial_logs,
            set_param
                .as_mut()
                .map(|h| &mut **h as &mut dyn AsteroCaseHooks),
            SampleAverages::from_samples(&samples, active_ctrl),
        )?;
        total_steps += stage.steps_taken;
        let stage_term = stage.termination_code;
        last_termination.clone_from(&stage_term);
        let c2 = sample.chi2;
        let accepted = c2.is_finite() && c2 < 1.0e90;
        // Upstream `eval1` calls `run1_star` once per track
        // (`astero_run_support.f90:224-229`), and every run reopens
        // `LOGS/history.data` without `position='append'` because
        // `doing_first_model_of_run` is set (`star/private/history.f90:313-321`).
        // The history left at the end of a search is therefore the last
        // track's, accepted or not, so every track overwrites it here (#430).
        if want_logs && sample_hist_path.is_file() {
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
        if accepted {
            if c2 < best_chi2_seen {
                best_chi2_seen = c2;
                best_termination = Some(stage_term);
            }
            sample.sample_number = next_sample_num;
            write_sample_file(&out_dir, &sample)?;
            samples.push(sample);
            Ok(c2)
        } else {
            // Upstream `eval1` / `simplex_f` (`astero_run_support.f90:241-246,749-753`):
            // unaccepted uncapped tracks return `1d99` without incrementing
            // `sample_number` or writing sample files. Bounded smoke runs still
            // record the sentinel entry so smoke diagnostics report a track.
            if max_steps_per_stage.is_some() {
                if best_termination.is_none() {
                    best_termination = Some(stage_term);
                }
                samples.push(sample);
            }
            Ok(FAILED_TRACK_CHI2)
        }
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
        best_termination,
        last_termination,
        last_model,
    })
}

#[cfg(test)]
mod tests {
    use super::{SampleAverages, FAILED_TRACK_CHI2};
    use crate::inlist_astero::{AsteroSearchControls, MAX_CONSTRAINTS};
    use crate::sample_eval::AsteroSampleResult;

    fn sample(
        chi2: f64,
        age_yr: f64,
        model_number: usize,
    ) -> AsteroSampleResult {
        AsteroSampleResult {
            sample_number: 1,
            params: AsteroSearchControls::default().first_params(),
            model_number,
            age_yr,
            teff: 5777.0,
            logg: 4.44,
            log_l: 0.0,
            log_r: 0.0,
            constraint_values: [f64::NAN; MAX_CONSTRAINTS],
            delta_nu_uhz: 0.0,
            nu_max_uhz: 0.0,
            a_div_r: 0.0,
            correction_r: 0.0,
            chi2,
            chi2_seismo: FAILED_TRACK_CHI2,
            chi2_spectro: FAILED_TRACK_CHI2,
            chi2_radial: FAILED_TRACK_CHI2,
            chi2_r010: 0.0,
            chi2_r02: 0.0,
            model_freqs: Default::default(),
            model_freqs_corr: Default::default(),
            model_inertias: Default::default(),
            model_orders: Default::default(),
        }
    }

    /// Averages over the 3 best of 4 samples by chi^2; sigma is the sample
    /// standard deviation (`astero_run_support.f90:1045-1102`).
    #[test]
    fn sample_averages_use_the_best_samples_by_chi2() {
        let ctrl = AsteroSearchControls {
            min_num_samples_for_avg: 2,
            max_num_samples_for_avg: 3,
            ..AsteroSearchControls::default()
        };
        let rows = [
            (4.0, 100.0, 10),
            (1.0, 200.0, 20),
            (2.0, 400.0, 40),
            (3.0, 600.0, 60),
        ];
        let samples: Vec<_> =
            rows.iter().map(|&(c, a, m)| sample(c, a, m)).collect();
        let avg = SampleAverages::from_samples(&samples, &ctrl);
        // Best three: ages 200, 400, 600 (model numbers 20, 40, 60).
        assert_eq!(avg.age, 400.0);
        assert_eq!(avg.age_sigma, 200.0);
        assert_eq!(avg.model_number, 40.0);
        assert_eq!(avg.model_number_sigma, 20.0);
        let few = SampleAverages::from_samples(&samples[..1], &ctrl);
        assert_eq!(few.age, 1.0e99);
        assert_eq!(few.model_number_sigma, 1.0e99);
    }
}
