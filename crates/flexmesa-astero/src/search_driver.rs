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
    eval_model_astero_sample_with_output_root, eval_spectro_and_delta_nu_chi2,
    scan_and_save_mode_at_number, AsteroSampleResult,
};
use crate::search_methods::{
    build_scan_grid_params, extract_simplex_restart, format_show_best_report,
    read_samples_from_file, run_powell_driver, run_simplex_driver,
    write_best_model_artifacts, write_best_sample_file,
    write_search_results_table,
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
    /// Per-track evolutionary stage reports in evaluation order.
    pub track_stages: Vec<StageRunReport>,
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
    after_evolve_termination: Option<String>,
    params: AsteroParamSet,
    sample_number: usize,
    case_dir: &'a Path,
    output_root: &'a Path,
    best_sample: Option<AsteroSampleResult>,
    best_model: Option<StarModel>,
    /// Averages over the best samples of earlier tracks
    /// (`set_sample_averages`).
    averages: SampleAverages,
    /// `num_chi2_too_big`.
    bad_count: usize,
    astero_using_revised_max_yr_dt: bool,
    astero_revised_max_yr_dt: f64,
    astero_max_dt_next: f64,
    /// Current `constraint_value(i)` buffer initialized to `0.0` matching
    /// `astero/private/astero_run_support.f90:165`.
    constraint_values: [f64; MAX_CONSTRAINTS],
    /// Model number already checked by `extras_check_model` (`None` if not yet
    /// checked or after `Redo`), so `extras_after_evolve` only evaluates the
    /// final model when termination preempted `extras_check_model`.
    checked_model_number: Option<usize>,
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
    fn constraint_slot_is_read(&self, idx: usize) -> bool {
        let c = self.ctrl.indexed_constraints[idx];
        c.include || (c.sigmas_coeff != 0.0 && c.sigma > 0.0)
    }

    fn is_unset_constraint_err(err: &StarError) -> bool {
        match err {
            StarError::InvalidInput(msg) => {
                msg.contains("set_constraint_value('Rcz')")
                    && msg.contains("unset")
            }
            _ => false,
        }
    }

    /// `do_astero_extras_check_model` checking-age timestep logic
    /// (`astero/private/extras_support.f90:83-146`).
    fn apply_checking_age_dt_limit(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), ExtrasDirective> {
        if self.ctrl.num_smaller_steps_before_age_target == 0
            || self.ctrl.dt_for_smaller_steps_before_age_target <= 0.0
        {
            self.eval_error = Some(StarError::InvalidInput(String::from(
                "must set positive num_smaller_steps_before_age_target and \
                 dt_for_smaller_steps_before_age_target when checking age \
                 (astero/private/extras_support.f90:84-89)",
            )));
            return Err(ExtrasDirective::Terminate(String::from(
                "AsteroAgeControlsError",
            )));
        }
        if self.ctrl.age_target > model.star_age {
            let remaining_years = self.ctrl.age_target - model.star_age;
            if self.astero_using_revised_max_yr_dt
                && self.astero_revised_max_yr_dt > 0.0
            {
                model.controls.max_years_for_timestep =
                    self.astero_revised_max_yr_dt;
            }
            let cur_max = if model.controls.max_years_for_timestep > 0.0 {
                model.controls.max_years_for_timestep
            } else if self.ctrl.max_yrs_dt_when_cold > 0.0 {
                self.ctrl.max_yrs_dt_when_cold
            } else {
                1.0e99
            };
            let j = self.ctrl.num_smaller_steps_before_age_target as f64;
            let n_rem = (remaining_years / cur_max + 1.0e-6).floor();
            if remaining_years <= cur_max {
                model.controls.max_years_for_timestep = remaining_years;
                self.astero_using_revised_max_yr_dt = true;
                self.astero_revised_max_yr_dt =
                    model.controls.max_years_for_timestep;
                self.astero_max_dt_next =
                    model.controls.max_years_for_timestep * SECYER;
            } else if n_rem <= j {
                let prev_max_years = cur_max;
                let dt_sm = self.ctrl.dt_for_smaller_steps_before_age_target;
                let i = (remaining_years / dt_sm + 1.0e-6).floor().max(0.0);
                model.controls.max_years_for_timestep =
                    if (i + 1.0e-9) * dt_sm < remaining_years {
                        remaining_years / (i + 1.0)
                    } else if i > 0.0 {
                        remaining_years / i
                    } else {
                        remaining_years
                    };
                let min_max = prev_max_years
                    * model.controls.ts_stop.reduction_factor_for_max_timestep;
                if model.controls.max_years_for_timestep < min_max {
                    model.controls.max_years_for_timestep = min_max;
                }
                self.astero_using_revised_max_yr_dt = true;
                self.astero_revised_max_yr_dt =
                    model.controls.max_years_for_timestep;
                if model.dt_next / SECYER
                    > model.controls.max_years_for_timestep
                {
                    self.astero_max_dt_next =
                        model.controls.max_years_for_timestep * SECYER;
                }
            }
        } else if self.ctrl.include_age_in_chi2_spectro {
            let dt_sm = self.ctrl.dt_for_smaller_steps_before_age_target;
            if (model.controls.max_years_for_timestep - dt_sm).abs()
                > dt_sm * 1.0e-2
            {
                self.eval_error = Some(StarError::InvalidInput(format!(
                    "bad max_years_for_timestep ({}) vs \
                     dt_for_smaller_steps_before_age_target ({dt_sm}) \
                     (astero/private/extras_support.f90:138-145)",
                    model.controls.max_years_for_timestep
                )));
                return Err(ExtrasDirective::Terminate(String::from(
                    "AsteroBadMaxYearsTimestep",
                )));
            }
        }
        Ok(())
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
        eval_model_astero_sample_with_output_root(
            model,
            self.ctrl,
            values,
            self.params,
            self.sample_number,
            self.case_dir,
            self.output_root,
        )
        .map_err(|e| {
            self.eval_error = Some(e);
            ExtrasDirective::Terminate(String::from("AsteroSampleEvalError"))
        })
    }

    /// `constraint_value(i)` for every named constraint, from the case's
    /// `set_constraint_value` (`astero/private/extras_support.f90:204-209`);
    /// unnamed slots keep the `0.0` initial `constraint_value` buffer
    /// (`astero_run_support.f90:165`).
    ///
    /// Upstream `astero/work/src/run_star_extras.f90:96-105` leaves `val`
    /// unset (with `ierr = 0`) when `'Rcz'` finds no convective-zone base in
    /// `0.25..0.9 Rsun`. Case hooks raise an unset error on that branch so any
    /// slot actually read (`include_constraint_in_chi2_spectro` or active
    /// `sigmas_coeff`) aborts loudly, whereas unused slots leave
    /// `constraint_values[i]` unchanged. Unknown column errors from
    /// `star_get_history_output`
    /// (`astero/work/src/run_star_extras.f90:108-109`) always abort.
    fn eval_constraint_values(
        &mut self,
        model: &StarModel,
    ) -> Result<[f64; MAX_CONSTRAINTS], StarError> {
        for (i, raw) in self.ctrl.constraint_names.iter().enumerate() {
            let name = raw.trim();
            if name.is_empty() {
                continue;
            }
            let slot_used = self.constraint_slot_is_read(i);
            let res = {
                let hook = self.hooks.as_deref_mut().ok_or_else(|| {
                    StarError::InvalidInput(format!(
                        "astero constraint '{name}' needs the case's \
                         set_constraint_value from its run_star_extras.f90 \
                         (astero/private/extras_support.f90:204-209), but no \
                         case hook was supplied"
                    ))
                })?;
                hook.set_constraint_value(model, self.ctrl, name)
            };
            match res {
                Ok(val) => self.constraint_values[i] = val,
                Err(err)
                    if !slot_used && Self::is_unset_constraint_err(&err) => {}
                Err(err) => return Err(err),
            }
        }
        Ok(self.constraint_values)
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
            self.best_model = Some(model.clone());
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
        let check1 =
            |val: f64, tgt: f64, sig: f64, coeff: f64| -> (bool, f64) {
                if coeff == 0.0 || sig <= 0.0 {
                    return (false, 0.0);
                }
                let lim = tgt + sig * coeff;
                (
                    (coeff > 0.0 && val > lim) || (coeff < 0.0 && val < lim),
                    lim,
                )
            };
        if self.ctrl.delta_nu > 0.0 {
            let (hit, lim) = check1(
                dnu_uhz,
                self.ctrl.delta_nu,
                self.ctrl.delta_nu_sigma,
                self.ctrl.sigmas_coeff_for_delta_nu_limit,
            );
            if hit {
                return Some(String::from("delta_nu_limit"));
            }
            if self.ctrl.trace_limits
                && self.ctrl.sigmas_coeff_for_delta_nu_limit != 0.0
                && self.ctrl.delta_nu_sigma > 0.0
            {
                println!("delta_nu_model={dnu_uhz:.6} delta_nu_limit={lim:.6}");
            }
        }
        // `do i = 1, max_constraints` (`extras_support.f90:645-665`).
        for (i, c) in self.ctrl.indexed_constraints.iter().enumerate() {
            let (hit, lim) =
                check1(values[i], c.target, c.sigma, c.sigmas_coeff);
            let raw_name = self.ctrl.constraint_names[i].trim();
            let label = if raw_name.is_empty() {
                format!("constraint_{}", i + 1)
            } else {
                String::from(raw_name)
            };
            if hit {
                return Some(format!("{label}_limit"));
            }
            if self.ctrl.trace_limits && c.sigmas_coeff != 0.0 && c.sigma > 0.0
            {
                println!(
                    "constraint_value({i})={:.6} constraint_limit={lim:.6}",
                    values[i]
                );
            }
        }
        None
    }

    fn run_check_model_core(
        &mut self,
        model: &mut StarModel,
    ) -> ExtrasDirective {
        // `do_astero_extras_check_model`
        // (`astero/private/extras_support.f90:56-669`), in upstream's order.
        self.astero_max_dt_next = 1.0e99;
        let checking_age = self.ctrl.eval_chi2_at_target_age_only
            || self.ctrl.include_age_in_chi2_spectro;
        if checking_age {
            if let Err(dir) = self.apply_checking_age_dt_limit(model) {
                return dir;
            }
        } else {
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
        }
        if self.ctrl.include_age_in_chi2_spectro
            && self
                .ctrl
                .min_age_for_chi2
                .is_some_and(|m| model.star_age < m)
        {
            // `:152`.
            return ExtrasDirective::KeepGoing;
        }
        if self.ctrl.eval_chi2_at_target_age_only
            && model.star_age < self.ctrl.age_target
        {
            // `:153`.
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
            if let Err(e) = scan_and_save_mode_at_number(
                model,
                self.ctrl,
                self.case_dir,
                self.output_root,
            ) {
                self.eval_error = Some(e);
                return ExtrasDirective::Terminate(String::from(
                    "AsteroModeSaveError",
                ));
            }
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

impl StarExtras for AsteroTrackExtras<'_, '_> {
    fn extras_check_model(&mut self, model: &mut StarModel) -> ExtrasDirective {
        let dir = self.run_check_model_core(model);
        self.checked_model_number = if matches!(dir, ExtrasDirective::Redo) {
            None
        } else {
            Some(model.model_number)
        };
        dir
    }

    fn extras_finish_step(&mut self, model: &mut StarModel) -> ExtrasDirective {
        model.dt_next = model.dt_next.min(self.astero_max_dt_next);
        ExtrasDirective::KeepGoing
    }

    fn extras_after_evolve(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        // `astero_extras_after_evolve` (`astero/private/extras_support.f90:1109-1115`)
        // evaluates `do_astero_extras_check_model` when evolution stopped on an
        // early limit (`max_age`, `max_model_number`) before `extras_check_model`
        // ran on the final model.
        let needs_final_check = (self.ctrl.eval_chi2_at_target_age_only
            && model.star_age >= self.ctrl.age_target)
            || (self.ctrl.include_age_in_chi2_spectro
                && self
                    .ctrl
                    .max_age_for_chi2
                    .is_some_and(|m| model.star_age >= m))
            || self.ctrl.save_info_for_last_model;
        if needs_final_check
            && self.checked_model_number != Some(model.model_number)
        {
            let _ = self.run_check_model_core(model);
            self.checked_model_number = Some(model.model_number);
            if let Some(err) = self.eval_error.take() {
                return Err(err);
            }
        }
        // `save_best_info` writes the `best_model_*` files during the track,
        // so they exist before `extras_after_evolve` runs
        // (`extras_support.f90:780-858,1116`). Cases such as
        // `astero/test_suite/fast_scan_grid/src/run_star_extras.f90:205-244`
        // move them to `sample_<N>.*` from that routine.
        let results_dir =
            self.output_root.join(&self.ctrl.astero_results_directory);
        if let Some(ref best_model) = self.best_model {
            write_best_model_artifacts(
                &results_dir,
                self.ctrl,
                best_model,
                self.case_dir,
            )?;
        }
        let best_chi2 = self.best_sample.as_ref().map_or(-1.0, |b| b.chi2);
        if let Some(hook) = self.hooks.as_deref_mut() {
            if let Some(code) = hook.extras_after_evolve_for_sample(
                model,
                best_chi2,
                self.ctrl,
                &results_dir,
                self.sample_number,
            )? {
                let trimmed = code.trim();
                if trimmed.is_empty() {
                    return Err(StarError::InvalidInput(String::from(
                        "astero extras_after_evolve returned empty \
                         termination code",
                    )));
                }
                self.after_evolve_termination = Some(trimmed.to_string());
            }
        }
        Ok(())
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

    /// Optional `extras_after_evolve` hook invoked at the end of a track
    /// (`astero/private/extras_support.f90:1116`), receiving the final model
    /// and the running `best_chi2` (`-1.0` when no accepted sample exists yet;
    /// `astero/private/astero_run_support.f90:220`).
    ///
    /// Returning `Ok(Some(code))` records a custom post-evolve termination code
    /// such as `"xtra1"`
    /// (`astero/test_suite/example_astero/src/run_star_extras.f90:214`),
    /// overriding the track's [`StageRunReport`] termination code and updating
    /// [`AsteroSearchReport::last_termination`] and
    /// [`AsteroSearchReport::best_termination`].
    fn extras_after_evolve(
        &mut self,
        _model: &StarModel,
        _best_chi2: f64,
    ) -> Result<Option<String>, StarError> {
        Ok(None)
    }

    /// [`Self::extras_after_evolve`] with the search state that
    /// `astero_def` module variables give a case's routine: the controls
    /// (`num_digits`, `sample_results_prefix`, `best_model_*_filename`), the
    /// resolved `astero_results_directory` and the number of the sample the
    /// track will become, `sample_number + 1` (`sample_number` is not
    /// incremented until `eval1` returns,
    /// `astero/private/astero_run_support.f90:241-254`). The default calls
    /// [`Self::extras_after_evolve`].
    ///
    /// The `best_model_*` files of the track have already been written to
    /// `results_dir` when this runs (`extras_support.f90:780-858`).
    ///
    /// # Errors
    /// Same as [`Self::extras_after_evolve`].
    fn extras_after_evolve_for_sample(
        &mut self,
        model: &StarModel,
        best_chi2: f64,
        ctrl: &AsteroSearchControls,
        results_dir: &Path,
        next_sample_number: usize,
    ) -> Result<Option<String>, StarError> {
        let _ = (ctrl, results_dir, next_sample_number);
        self.extras_after_evolve(model, best_chi2)
    }
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
        constraint_values: [0.0; MAX_CONSTRAINTS],
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
        surf_coef1: 0.0,
        surf_coef2: 0.0,
        ratios_n: 0,
        ratios_l0_first: 0,
        ratios_l1_first: 0,
        obs_r01: Vec::new(),
        obs_r01_sigmas: Vec::new(),
        model_r01: Vec::new(),
        obs_r10: Vec::new(),
        obs_r10_sigmas: Vec::new(),
        model_r10: Vec::new(),
        obs_r02: Vec::new(),
        obs_r02_sigmas: Vec::new(),
        model_r02: Vec::new(),
        model_freqs_alt_up: Default::default(),
        model_freqs_corr_alt_up: Default::default(),
        model_inertias_alt_up: Default::default(),
        model_orders_alt_up: Default::default(),
        model_freqs_alt_down: Default::default(),
        model_freqs_corr_alt_down: Default::default(),
        model_inertias_alt_down: Default::default(),
        model_orders_alt_down: Default::default(),
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
    let output_root = logs_opts.output_root.unwrap_or(case_dir);
    run_track(
        base_bundle,
        ctrl,
        params,
        sample_number,
        max_steps_cap,
        case_dir,
        output_root,
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
    output_root: &Path,
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
        after_evolve_termination: None,
        params,
        sample_number,
        case_dir,
        output_root,
        best_sample: None,
        best_model: None,
        averages,
        bad_count: 0,
        astero_using_revised_max_yr_dt: false,
        astero_revised_max_yr_dt: bundle.controls.max_years_for_timestep,
        astero_max_dt_next: 1.0e99,
        constraint_values: [0.0; MAX_CONSTRAINTS],
        checked_model_number: None,
        eval_error: None,
    };
    let mut model =
        init_or_update_model_from_bundle(&bundle, None, &mut extras)?;
    let mut logs_writer = logs_opts.build_writer(&model, case_dir, None)?;
    let mut stage = run_inlist_stage_with_logs(
        &mut model,
        &bundle,
        &mut extras,
        max_steps_cap,
        logs_writer.as_mut(),
    )?;
    if let Some(err) = extras.eval_error.take() {
        return Err(err);
    }
    if let Some(code) = extras.after_evolve_termination.take() {
        stage.termination_code = code;
    }
    let results_dir = output_root.join(&ctrl.astero_results_directory);
    if ctrl.save_info_for_last_model {
        let vals = extras.eval_constraint_values(&model)?;
        let last_sample = eval_model_astero_sample_with_output_root(
            &model,
            ctrl,
            &vals,
            params,
            sample_number,
            case_dir,
            output_root,
        )?;
        let fname = ctrl.last_model_save_info_filename.trim();
        let fname = if fname.is_empty() {
            "last_model_info.data"
        } else {
            fname
        };
        fs::create_dir_all(&results_dir).map_err(|e| {
            StarError::IoError(format!(
                "failed creating astero_results_directory '{}': {e}",
                results_dir.display()
            ))
        })?;
        let report_text = format_show_best_report(ctrl, &last_sample);
        let p = results_dir.join(fname);
        fs::write(&p, report_text).map_err(|e| {
            StarError::IoError(format!(
                "failed writing last_model_save_info_filename '{}': {e}",
                p.display()
            ))
        })?;
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
    active_ctrl.write_controls_if_requested(results_base)?;
    let st = active_ctrl.search_type.trim().to_ascii_lowercase();
    let mut best_chi2_seen = f64::INFINITY;
    let mut best_termination: Option<String> = None;
    let (mut samples, mut total_steps, mut tracks_evaluated) =
        (Vec::new(), 0usize, 0usize);
    let mut scan_grid_skip = 0usize;
    let mut scan_grid_total: Option<usize> = None;
    let mut initial_simplex: Option<(Vec<Vec<f64>>, Vec<f64>)> = None;
    if st == "scan_grid" {
        let grid_len = build_scan_grid_params(active_ctrl).len();
        scan_grid_total = Some(grid_len);
        if active_ctrl.restart_scan_grid_from_file
            && max_steps_per_stage.is_none()
        {
            let prior = read_samples_from_file(
                &out_dir.join(&active_ctrl.scan_grid_output_filename),
                active_ctrl,
            )?;
            for s in &prior {
                if s.chi2.is_finite() && s.chi2 < best_chi2_seen {
                    best_chi2_seen = s.chi2;
                }
            }
            scan_grid_skip = prior.len();
            samples = prior;
        }
    } else if st == "simplex"
        && active_ctrl.restart_simplex_from_file
        && max_steps_per_stage.is_none()
    {
        let prior = read_samples_from_file(
            &out_dir.join(&active_ctrl.simplex_output_filename),
            active_ctrl,
        )?;
        let (kept, verts, f_vals) =
            extract_simplex_restart(active_ctrl, &prior)?;
        for s in &kept {
            if s.chi2.is_finite() && s.chi2 < best_chi2_seen {
                best_chi2_seen = s.chi2;
            }
        }
        samples = kept;
        initial_simplex = Some((verts, f_vals));
    }

    let is_powell = st == "newuoa" || st == "bobyqa";
    let (mut last_termination, mut last_model) = (String::new(), None);
    let mut track_stages: Vec<StageRunReport> = Vec::new();
    let mut eval_param_set = |p: AsteroParamSet| -> Result<f64, StarError> {
        if tracks_evaluated >= track_cap {
            return Ok(FAILED_TRACK_CHI2);
        }
        tracks_evaluated += 1;
        let next_sample_num =
            samples.iter().map(|s| s.sample_number).max().unwrap_or(0) + 1;
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
            results_base,
            trial_logs,
            set_param
                .as_mut()
                .map(|h| &mut **h as &mut dyn AsteroCaseHooks),
            SampleAverages::from_samples(&samples, active_ctrl),
        )?;
        total_steps += stage.steps_taken;
        let stage_term = stage.termination_code.clone();
        track_stages.push(stage);
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
            if active_ctrl.write_best_model_data_for_each_sample {
                write_best_sample_file(&out_dir, active_ctrl, &sample)?;
            }
            samples.push(sample);
            Ok(c2)
        } else {
            if is_powell && max_steps_per_stage.is_none() && samples.is_empty()
            {
                return Err(StarError::InvalidInput(String::from(
                    "failed to find chi^2 on 1st try; must give 'first' values that yield a chi^2 result",
                )));
            }
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
                .skip(scan_grid_skip)
                .take(track_cap)
            {
                let _ = eval_param_set(p)?;
            }
            &active_ctrl.scan_grid_output_filename
        }
        "simplex" => {
            run_simplex_driver(
                active_ctrl,
                initial_simplex,
                &mut eval_param_set,
            )?;
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
    write_search_results_table(
        &out_dir.join(results_fname),
        active_ctrl,
        &samples,
        scan_grid_total,
    )?;
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
        track_stages,
        best_termination,
        last_termination,
        last_model,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use flexmesa_const::SECYER;
    use flexmesa_star::controls::{
        AtmOption, AtmTTauOpacity, AtmTTauRelation, StarControls, StarError,
        StarJobControls,
    };
    use flexmesa_star::inlist::InlistBundle;
    use flexmesa_star::inlist_runner::{ExtrasDirective, StarExtras};
    use flexmesa_star::model::StarModel;
    use flexmesa_star::{load_zams_model, set_hydro_vars};

    use super::{
        AsteroCaseHooks, AsteroTrackExtras, SampleAverages, FAILED_TRACK_CHI2,
    };
    use crate::inlist_astero::{
        AsteroSearchControls, SpectroConstraint, MAX_CONSTRAINTS,
    };
    use crate::sample_eval::AsteroSampleResult;

    struct ScriptedHooks {
        rcz_result: Result<f64, &'static str>,
        calls: usize,
    }

    impl AsteroCaseHooks for ScriptedHooks {
        fn set_param(
            &mut self,
            _bundle: &mut InlistBundle,
            _ctrl: &AsteroSearchControls,
            _name: &str,
            _val: f64,
        ) -> Result<(), StarError> {
            Ok(())
        }

        fn set_constraint_value(
            &mut self,
            model: &StarModel,
            _ctrl: &AsteroSearchControls,
            name: &str,
        ) -> Result<f64, StarError> {
            self.calls += 1;
            match name {
                "Teff" => Ok(model.teff),
                "Rcz" => self
                    .rcz_result
                    .map_err(|msg| StarError::InvalidInput(String::from(msg))),
                other => Err(StarError::InvalidInput(format!(
                    "unknown history column '{other}'"
                ))),
            }
        }
    }

    fn zams_model() -> StarModel {
        let controls = StarControls {
            zbase: 0.02,
            calculate_brunt_n2: true,
            atm_option: AtmOption::TTau,
            atm_t_tau_relation: AtmTTauRelation::Eddington,
            atm_t_tau_opacity: AtmTTauOpacity::Fixed,
            ..StarControls::default()
        };
        let mut model = load_zams_model(controls, StarJobControls::default())
            .expect("zams");
        set_hydro_vars(&mut model, false, true).expect("set_hydro_vars");
        model
    }

    fn make_extras<'a, 'h>(
        ctrl: &'a AsteroSearchControls,
        hooks: &'h mut ScriptedHooks,
        root: &'a Path,
    ) -> AsteroTrackExtras<'a, 'h> {
        AsteroTrackExtras {
            ctrl,
            hooks: Some(hooks),
            after_evolve_termination: None,
            params: ctrl.first_params(),
            sample_number: 1,
            case_dir: root,
            output_root: root,
            best_sample: None,
            best_model: None,
            averages: SampleAverages::NONE,
            bad_count: 0,
            astero_using_revised_max_yr_dt: false,
            astero_revised_max_yr_dt: 1.0e8,
            astero_max_dt_next: 1.0e99,
            constraint_values: [0.0; MAX_CONSTRAINTS],
            checked_model_number: None,
            eval_error: None,
        }
    }

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
            surf_coef1: 0.0,
            surf_coef2: 0.0,
            ratios_n: 0,
            ratios_l0_first: 0,
            ratios_l1_first: 0,
            obs_r01: Vec::new(),
            obs_r01_sigmas: Vec::new(),
            model_r01: Vec::new(),
            obs_r10: Vec::new(),
            obs_r10_sigmas: Vec::new(),
            model_r10: Vec::new(),
            obs_r02: Vec::new(),
            obs_r02_sigmas: Vec::new(),
            model_r02: Vec::new(),
            model_freqs_alt_up: Default::default(),
            model_freqs_corr_alt_up: Default::default(),
            model_inertias_alt_up: Default::default(),
            model_orders_alt_up: Default::default(),
            model_freqs_alt_down: Default::default(),
            model_freqs_corr_alt_down: Default::default(),
            model_inertias_alt_down: Default::default(),
            model_orders_alt_down: Default::default(),
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

    /// Verifies `checking_age` runs in `extras_check_model`, never clamps sub-
    /// 1-yr `remaining_years` to `1.0`, leaves `model.dt` untouched, and lets
    /// `extras_finish_step` enforce `astero_max_dt_next` (`extras_support.f90:90-102,935`).
    #[test]
    fn checking_age_subyear_remaining_not_clamped_and_finish_caps_dt_next() {
        let ctrl = AsteroSearchControls {
            eval_chi2_at_target_age_only: true,
            age_target: 100.25,
            num_smaller_steps_before_age_target: 50,
            dt_for_smaller_steps_before_age_target: 1.0,
            ..AsteroSearchControls::default()
        };

        let mut model = zams_model();
        model.model_number = 5;
        model.star_age = 100.0;
        model.controls.max_years_for_timestep = 10.0;
        model.dt = 10.0 * SECYER;
        model.dt_next = 10.0 * SECYER;

        let root = std::env::temp_dir();
        let mut hooks = ScriptedHooks {
            rcz_result: Ok(0.71),
            calls: 0,
        };
        let mut extras = make_extras(&ctrl, &mut hooks, &root);

        assert!(matches!(
            extras.extras_start_step(&mut model),
            ExtrasDirective::KeepGoing
        ));
        assert!(matches!(
            extras.extras_check_model(&mut model),
            ExtrasDirective::KeepGoing
        ));
        assert!(
            (model.controls.max_years_for_timestep - 0.25).abs() <= 1.0e-12
        );
        assert!((model.dt - 10.0 * SECYER).abs() <= 1.0);
        assert!((extras.astero_max_dt_next - 0.25 * SECYER).abs() <= 1.0);
        assert_eq!(extras.checked_model_number, Some(5));

        assert!(matches!(
            extras.extras_finish_step(&mut model),
            ExtrasDirective::KeepGoing
        ));
        assert!((model.dt_next - 0.25 * SECYER).abs() <= 1.0);
    }

    /// Verifies `checking_age` smooth step decay (`n <= j`) and `extras_after_evolve`
    /// scoring when `do_check_limits` preempts `extras_check_model`
    /// (`extras_support.f90:103-134,1109-1115`).
    #[test]
    fn checking_age_decay_and_after_evolve_when_check_model_bypassed() {
        let mut ctrl = AsteroSearchControls {
            eval_chi2_at_target_age_only: true,
            age_target: 1000.0,
            num_smaller_steps_before_age_target: 50,
            dt_for_smaller_steps_before_age_target: 2.0,
            min_age_limit: 0.0,
            lnuc_div_l_limit: 0.0,
            ..AsteroSearchControls::default()
        };
        ctrl.weights.chi2_seismo_fraction = 0.0;
        ctrl.constraint_names[0] = String::from("Teff");
        ctrl.indexed_constraints[0] = SpectroConstraint {
            include: true,
            target: 5777.0,
            sigma: 65.0,
            sigmas_coeff: 0.0,
        };

        let mut model = zams_model();
        model.model_number = 6;
        model.star_age = 900.0;
        model.controls.max_years_for_timestep = 10.0;
        model.controls.ts_stop.reduction_factor_for_max_timestep = 0.98;
        model.dt_next = 10.0 * SECYER;

        let root = std::env::temp_dir();
        let mut hooks = ScriptedHooks {
            rcz_result: Ok(0.71),
            calls: 0,
        };
        let mut extras = make_extras(&ctrl, &mut hooks, &root);

        assert!(matches!(
            extras.extras_check_model(&mut model),
            ExtrasDirective::KeepGoing
        ));
        assert!((model.controls.max_years_for_timestep - 9.8).abs() <= 1.0e-12);
        assert!((extras.astero_max_dt_next - 9.8 * SECYER).abs() <= 1.0);
        assert!(extras.best_sample.is_none());

        // Simulate `max_age` termination on model 7 bypassing `extras_check_model`.
        model.model_number = 7;
        model.star_age = 1000.0;
        extras
            .extras_after_evolve(&mut model)
            .expect("extras_after_evolve scores final target-age model");
        let best = extras.best_sample.as_ref().expect("best_sample saved");
        assert_eq!(best.model_number, 7);
        assert!((best.age_yr - 1000.0).abs() <= 1.0e-12);
        assert_eq!(extras.checked_model_number, Some(7));

        // Re-invoking `extras_after_evolve` on an already checked model is a no-op.
        extras
            .extras_after_evolve(&mut model)
            .expect("idempotent extras_after_evolve");
        assert_eq!(hooks.calls, 1);
    }

    /// Verifies `eval_constraint_values` preserves prior buffer entries only
    /// when an inactive `'Rcz'` slot leaves `val` unset, while active slots
    /// and unknown columns always abort (`astero_run_support.f90:165`,
    /// `extras_support.f90:204-209`).
    #[test]
    fn eval_constraint_values_preserves_prior_only_on_unused_unset_rcz() {
        let mut ctrl = AsteroSearchControls::default();
        ctrl.constraint_names[5] = String::from("Rcz");
        let model = zams_model();
        let root = std::env::temp_dir();

        {
            let mut hooks = ScriptedHooks {
                rcz_result: Err(
                    "set_constraint_value('Rcz'): no convective-zone base \
                     between 0.25 and 0.9 Rsun, so upstream leaves val unset",
                ),
                calls: 0,
            };
            let mut extras = make_extras(&ctrl, &mut hooks, &root);
            let vals = extras
                .eval_constraint_values(&model)
                .expect("inactive unset Rcz retains 0.0");
            assert!((vals[5] - 0.0).abs() <= 1.0e-15);

            extras.constraint_values[5] = 0.712;
            let vals2 = extras
                .eval_constraint_values(&model)
                .expect("inactive unset Rcz retains prior 0.712");
            assert!((vals2[5] - 0.712).abs() <= 1.0e-15);
        }

        // Making the slot read (include or limit) must reject the same unset error.
        ctrl.indexed_constraints[5].include = true;
        {
            let mut hooks = ScriptedHooks {
                rcz_result: Err("set_constraint_value('Rcz'): val left unset"),
                calls: 0,
            };
            let mut extras = make_extras(&ctrl, &mut hooks, &root);
            assert!(extras.eval_constraint_values(&model).is_err());
        }

        // Unknown column name must fail even when slot is inactive.
        ctrl.indexed_constraints[5].include = false;
        ctrl.constraint_names[5] = String::from("bogus_col");
        let mut hooks = ScriptedHooks {
            rcz_result: Ok(0.71),
            calls: 0,
        };
        let mut extras = make_extras(&ctrl, &mut hooks, &root);
        assert!(extras.eval_constraint_values(&model).is_err());
    }

    struct AfterEvolveCaptureHooks {
        seen_best_chi2: Vec<f64>,
        response: Result<Option<String>, StarError>,
    }

    impl AsteroCaseHooks for AfterEvolveCaptureHooks {
        fn set_param(
            &mut self,
            _bundle: &mut InlistBundle,
            _ctrl: &AsteroSearchControls,
            _name: &str,
            _val: f64,
        ) -> Result<(), StarError> {
            Ok(())
        }

        fn set_constraint_value(
            &mut self,
            _model: &StarModel,
            _ctrl: &AsteroSearchControls,
            _name: &str,
        ) -> Result<f64, StarError> {
            Ok(0.0)
        }

        fn extras_after_evolve(
            &mut self,
            _model: &StarModel,
            best_chi2: f64,
        ) -> Result<Option<String>, StarError> {
            self.seen_best_chi2.push(best_chi2);
            self.response.clone()
        }
    }

    fn make_after_evolve_test_extras<'a, 'h>(
        ctrl: &'a AsteroSearchControls,
        hooks: &'h mut AfterEvolveCaptureHooks,
        root: &'a Path,
        best_sample: Option<AsteroSampleResult>,
    ) -> AsteroTrackExtras<'a, 'h> {
        AsteroTrackExtras {
            ctrl,
            hooks: Some(hooks),
            after_evolve_termination: None,
            params: ctrl.first_params(),
            sample_number: 1,
            case_dir: root,
            output_root: root,
            best_sample,
            best_model: None,
            averages: SampleAverages::NONE,
            bad_count: 0,
            astero_using_revised_max_yr_dt: false,
            astero_revised_max_yr_dt: 1.0e8,
            astero_max_dt_next: 1.0e99,
            constraint_values: [0.0; MAX_CONSTRAINTS],
            checked_model_number: None,
            eval_error: None,
        }
    }

    #[test]
    fn test_extras_after_evolve_receives_best_chi2_and_sets_termination() {
        let ctrl = AsteroSearchControls::default();
        let root = Path::new(".");
        let mut model = StarModel::new(
            4,
            vec![1],
            flexmesa_star::controls::StarJobControls::default(),
            flexmesa_star::controls::StarControls::default(),
        );

        let mut no_sample_hooks = AfterEvolveCaptureHooks {
            seen_best_chi2: Vec::new(),
            response: Ok(None),
        };
        {
            let mut extras = make_after_evolve_test_extras(
                &ctrl,
                &mut no_sample_hooks,
                root,
                None,
            );
            StarExtras::extras_after_evolve(&mut extras, &mut model)
                .expect("Ok(None) must succeed");
            assert_eq!(extras.after_evolve_termination, None);
        }
        assert_eq!(no_sample_hooks.seen_best_chi2, vec![-1.0]);

        let mut xtra1_hooks = AfterEvolveCaptureHooks {
            seen_best_chi2: Vec::new(),
            response: Ok(Some(String::from("  xtra1  "))),
        };
        {
            let mut extras = make_after_evolve_test_extras(
                &ctrl,
                &mut xtra1_hooks,
                root,
                Some(sample(2.800_488, 4.57e9, 734)),
            );
            StarExtras::extras_after_evolve(&mut extras, &mut model)
                .expect("xtra1 must succeed");
            assert_eq!(
                extras.after_evolve_termination.as_deref(),
                Some("xtra1")
            );
        }
        assert_eq!(xtra1_hooks.seen_best_chi2, vec![2.800_488]);

        let mut empty_hooks = AfterEvolveCaptureHooks {
            seen_best_chi2: Vec::new(),
            response: Ok(Some(String::from("   "))),
        };
        let mut extras = make_after_evolve_test_extras(
            &ctrl,
            &mut empty_hooks,
            root,
            Some(sample(2.8, 4.57e9, 734)),
        );
        let err = StarExtras::extras_after_evolve(&mut extras, &mut model)
            .expect_err("blank termination code must fail");
        assert!(matches!(err, StarError::InvalidInput(_)));
    }

    /// Hook that records the search state `extras_after_evolve_for_sample`
    /// receives and moves `best_model_save_model_filename` to
    /// `sample_<NNNN>.mod` like `fast_scan_grid`'s `extras_after_evolve`
    /// (`astero/test_suite/fast_scan_grid/src/run_star_extras.f90:205-244`).
    struct MoveBestModelHooks {
        calls: Vec<(PathBuf, usize, bool)>,
    }

    impl AsteroCaseHooks for MoveBestModelHooks {
        fn set_param(
            &mut self,
            _bundle: &mut InlistBundle,
            _ctrl: &AsteroSearchControls,
            _name: &str,
            _val: f64,
        ) -> Result<(), StarError> {
            Ok(())
        }

        fn set_constraint_value(
            &mut self,
            _model: &StarModel,
            _ctrl: &AsteroSearchControls,
            _name: &str,
        ) -> Result<f64, StarError> {
            Ok(0.0)
        }

        fn extras_after_evolve_for_sample(
            &mut self,
            _model: &StarModel,
            _best_chi2: f64,
            ctrl: &AsteroSearchControls,
            results_dir: &Path,
            next_sample_number: usize,
        ) -> Result<Option<String>, StarError> {
            let src =
                results_dir.join(ctrl.best_model_save_model_filename.trim());
            let existed = src.is_file();
            self.calls.push((
                results_dir.to_path_buf(),
                next_sample_number,
                existed,
            ));
            if existed {
                let tgt = results_dir.join(format!(
                    "{}{:0width$}.mod",
                    ctrl.sample_results_prefix.trim(),
                    next_sample_number,
                    width = ctrl.num_digits
                ));
                fs::rename(&src, &tgt).map_err(|e| {
                    StarError::IoError(format!("test mv failed: {e}"))
                })?;
            }
            Ok(None)
        }
    }

    /// `save_best_info` writes the `best_model_*` files during the track, so
    /// they must be on disk, in `astero_results_directory` under the output
    /// root, when `extras_after_evolve` runs; a track without an accepted
    /// model writes none (`extras_support.f90:780-858,1116`).
    #[test]
    fn best_model_files_exist_when_case_extras_after_evolve_runs() {
        let root = std::env::temp_dir()
            .join(format!("flexmesa_162_after_evolve_{}", std::process::id()));
        fs::create_dir_all(&root).expect("create scratch root");
        let ctrl = AsteroSearchControls {
            save_model_for_best_model: true,
            best_model_save_model_filename: String::from("best.mod"),
            astero_results_directory: String::from("outputs"),
            sample_results_prefix: String::from("sample_"),
            num_digits: 4,
            ..AsteroSearchControls::default()
        };
        let mut model = zams_model();

        let mut hooks = MoveBestModelHooks { calls: Vec::new() };
        {
            let mut extras = AsteroTrackExtras {
                ctrl: &ctrl,
                hooks: Some(&mut hooks),
                after_evolve_termination: None,
                params: ctrl.first_params(),
                sample_number: 5,
                case_dir: &root,
                output_root: &root,
                best_sample: Some(sample(0.5, 1.0e9, 7)),
                best_model: Some(zams_model()),
                averages: SampleAverages::NONE,
                bad_count: 0,
                astero_using_revised_max_yr_dt: false,
                astero_revised_max_yr_dt: 1.0e8,
                astero_max_dt_next: 1.0e99,
                constraint_values: [0.0; MAX_CONSTRAINTS],
                checked_model_number: None,
                eval_error: None,
            };
            StarExtras::extras_after_evolve(&mut extras, &mut model)
                .expect("extras_after_evolve with a best model");
        }
        let out = root.join("outputs");
        assert_eq!(hooks.calls, vec![(out.clone(), 5, true)]);
        assert!(!out.join("best.mod").exists(), "hook moved best.mod");
        assert!(out.join("sample_0005.mod").is_file());

        // A track that never accepted a model has no `best_model_*` file to
        // move (upstream `mv ... skip_errors=.true.`).
        let mut hooks = MoveBestModelHooks { calls: Vec::new() };
        {
            let mut extras = AsteroTrackExtras {
                ctrl: &ctrl,
                hooks: Some(&mut hooks),
                after_evolve_termination: None,
                params: ctrl.first_params(),
                sample_number: 6,
                case_dir: &root,
                output_root: &root,
                best_sample: None,
                best_model: None,
                averages: SampleAverages::NONE,
                bad_count: 0,
                astero_using_revised_max_yr_dt: false,
                astero_revised_max_yr_dt: 1.0e8,
                astero_max_dt_next: 1.0e99,
                constraint_values: [0.0; MAX_CONSTRAINTS],
                checked_model_number: None,
                eval_error: None,
            };
            StarExtras::extras_after_evolve(&mut extras, &mut model)
                .expect("extras_after_evolve without a best model");
        }
        assert_eq!(hooks.calls, vec![(out.clone(), 6, false)]);
        assert!(!out.join("sample_0006.mod").exists());

        fs::remove_dir_all(&root).expect("remove scratch root");
    }
}
