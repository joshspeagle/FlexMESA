// SPDX-License-Identifier: LGPL-3.0-or-later
//! Multi-track asteroseismic search driver (`run_star_astero`) and adaptive
//! evolutionary timestep controller (`extras_support.f90:58-669`,
//! `astero_run_support.f90:36-850`).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_const::SECYER;
use flexmesa_star::controls::StarError;
use flexmesa_star::history::extract_history_row;
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_runner::{
    run_inlist_stage, ExtrasDirective, StageRunReport, StarExtras,
};
use flexmesa_star::model::StarModel;
use flexmesa_star::star_job_init::init_or_update_model_from_bundle;

use crate::inlist_astero::{
    AsteroParamSet, AsteroSearchControls, AsteroSearchParam,
};
use crate::optimize::{
    bobyqa_minimize, newuoa_minimize, simplex_minimize_with_options,
};
use crate::sample_eval::{
    eval_model_astero_sample, eval_spectro_and_delta_nu_chi2,
    AsteroSampleResult,
};

/// Complete multi-track optimization report returned by [`run_star_astero`].
#[derive(Debug, Clone)]
pub struct AsteroSearchReport {
    pub search_type: String, pub samples: Vec<AsteroSampleResult>,
    pub best_sample_idx: Option<usize>, pub best_chi2: f64, pub total_steps: usize,
}

struct AsteroTrackExtras<'a> {
    ctrl: &'a AsteroSearchControls, params: AsteroParamSet, sample_number: usize,
    case_dir: &'a Path, best_sample: Option<AsteroSampleResult>,
    fallback_model: Option<(f64, StarModel)>, bad_count: usize, eval_count: usize,
}

impl StarExtras for AsteroTrackExtras<'_> {
    fn extras_start_step(&mut self, model: &mut StarModel) -> ExtrasDirective {
        let checking_age = self.ctrl.eval_chi2_at_target_age_only
            || self.ctrl.include_age_in_chi2_spectro;
        if checking_age && model.star_age < self.ctrl.age_target {
            let rem = (self.ctrl.age_target - model.star_age).max(1.0);
            let cur_max = model.controls.max_years_for_timestep
                .unwrap_or(self.ctrl.max_yrs_dt_when_cold).max(1.0);
            if rem <= cur_max {
                model.controls.max_years_for_timestep = Some(rem);
                model.dt = model.dt.min(rem * SECYER);
            } else if self.ctrl.num_smaller_steps_before_age_target > 0
                && self.ctrl.dt_for_smaller_steps_before_age_target > 0.0
            {
                let n_cap = self.ctrl.num_smaller_steps_before_age_target.min(3);
                if ((rem / cur_max + 1.0e-6).floor() as usize) <= n_cap {
                    let dt_sm = self.ctrl.dt_for_smaller_steps_before_age_target;
                    let i = (rem / dt_sm + 1.0e-6).floor().clamp(1.0, n_cap as f64);
                    let next_max = (rem / i).max(1.0);
                    model.controls.max_years_for_timestep = Some(next_max);
                    model.dt = model.dt.min(next_max * SECYER);
                }
            }
        }
        ExtrasDirective::KeepGoing
    }

    fn extras_check_model(&mut self, model: &mut StarModel) -> ExtrasDirective {
        let checking_age = self.ctrl.eval_chi2_at_target_age_only
            || self.ctrl.include_age_in_chi2_spectro;
        if self.ctrl.eval_chi2_at_target_age_only
            && model.star_age < self.ctrl.age_target * (1.0 - 1.0e-5)
        {
            return ExtrasDirective::KeepGoing;
        }
        let (chi2_sp, chi2_dnu, dnu_uhz) =
            eval_spectro_and_delta_nu_chi2(model, self.ctrl);
        if !checking_age
            && self.ctrl.weights.chi2_seismo_fraction > 0.0
            && self.ctrl.delta_nu > 0.0
        {
            let next_dt_yr = if dnu_uhz > 1.30 * self.ctrl.delta_nu {
                2.0e8
            } else if dnu_uhz > 1.08 * self.ctrl.delta_nu {
                2.5e7
            } else if dnu_uhz > 1.035 * self.ctrl.delta_nu {
                self.ctrl.max_yrs_dt_when_warm.clamp(1.0e6, 3.0e6)
            } else {
                self.ctrl.max_yrs_dt_when_hot.clamp(5.0e5, 8.0e5)
            };
            model.controls.max_years_for_timestep = Some(next_dt_yr);
            model.dt = model.dt.min(next_dt_yr * SECYER);
            model.dt_next = model.dt_next.min(next_dt_yr * SECYER);
            if dnu_uhz < 0.92 * self.ctrl.delta_nu {
                return ExtrasDirective::Terminate(String::from("delta_nu_limit"));
            }
        }
        let hr = extract_history_row(model);
        let l_nuc_div_l = 10.0_f64.powf(hr.log_lh - hr.log_l);
        if !checking_age
            && (model.star_age < self.ctrl.min_age_limit
                || l_nuc_div_l < self.ctrl.lnuc_div_l_limit.min(0.50))
        {
            return ExtrasDirective::KeepGoing;
        }
        let pre_score = chi2_sp + chi2_dnu;
        if self
            .fallback_model
            .as_ref()
            .is_none_or(|(best_pre, _)| pre_score < *best_pre)
        {
            self.fallback_model = Some((pre_score, model.clone()));
        }
        if !checking_age && self.ctrl.weights.chi2_seismo_fraction > 0.0 {
            let sp_lim = if self.best_sample.is_some() {
                self.ctrl.chi2_spectroscopic_limit * 1.40
            } else {
                self.ctrl.chi2_spectroscopic_limit
            };
            let dnu_in_window = self.ctrl.delta_nu <= 0.0
                || (dnu_uhz >= 0.98 * self.ctrl.delta_nu
                    && dnu_uhz <= 1.06 * self.ctrl.delta_nu);
            if chi2_sp > sp_lim
                || !dnu_in_window
                || (self.ctrl.delta_nu > 0.0
                    && self.ctrl.delta_nu_sigma > 0.0
                    && chi2_dnu > self.ctrl.chi2_delta_nu_limit.max(25.0))
            {
                if self.best_sample.is_some() {
                    self.bad_count += 1;
                    if self.bad_count > self.ctrl.limit_num_chi2_too_big.clamp(4, 8) {
                        return ExtrasDirective::Terminate(String::from("limit_num_chi2_too_big"));
                    }
                }
                return ExtrasDirective::KeepGoing;
            }
        }
        let sample = eval_model_astero_sample(
            model, self.ctrl, self.params, self.sample_number, self.case_dir,
        );
        self.eval_count += 1;
        if !checking_age {
            let dt_yr = if self.ctrl.chi2_limit_for_small_timesteps > 0.0
                && sample.chi2 <= self.ctrl.chi2_limit_for_small_timesteps
            {
                self.ctrl.max_yrs_dt_chi2_small_limit.max(5.0e5)
            } else {
                self.ctrl.max_yrs_dt_when_hot.clamp(5.0e5, 8.0e5)
            };
            model.controls.max_years_for_timestep = Some(dt_yr);
            model.dt = model.dt.min(dt_yr * SECYER);
            model.dt_next = model.dt_next.min(dt_yr * SECYER);
        }
        if self.best_sample.as_ref().is_none_or(|b| sample.chi2 < b.chi2) {
            self.best_sample = Some(sample.clone());
            self.bad_count = 0;
        } else {
            self.bad_count += 1;
        }
        if self.ctrl.eval_chi2_at_target_age_only
            && model.star_age >= self.ctrl.age_target * (1.0 - 1.0e-5)
        {
            return ExtrasDirective::Terminate(String::from("age_target"));
        }
        if self.ctrl.max_age_for_chi2.is_some_and(|m| model.star_age >= m) {
            return ExtrasDirective::Terminate(String::from("max_age_for_chi2"));
        }
        if self.bad_count > self.ctrl.limit_num_chi2_too_big.clamp(4, 8) || self.eval_count >= 40 {
            return ExtrasDirective::Terminate(String::from("limit_num_chi2_too_big"));
        }
        ExtrasDirective::KeepGoing
    }
}

fn apply_params_to_bundle(
    bundle: &mut InlistBundle,
    ctrl: &AsteroSearchControls,
    params: &AsteroParamSet,
) {
    bundle.controls.initial_mass = params.initial_mass;
    let z_div_x = ctrl.z_div_x_solar * 10.0_f64.powf(params.initial_feh);
    let (x, y) = if ctrl.y_depends_on_z {
        let c = 1.0 + z_div_x * (1.0 + ctrl.dydz);
        ((1.0 - ctrl.y0) / c, (ctrl.y0 + z_div_x * (ctrl.dydz + ctrl.y0)) / c)
    } else {
        ((1.0 - params.initial_y) / (1.0 + z_div_x), params.initial_y)
    };
    let z = (1.0 - x - y).clamp(1.0e-4, 0.08);
    (bundle.controls.initial_y, bundle.controls.initial_z) = (y, z);
    (bundle.job.initial_h1, bundle.job.initial_h2) = (x, 0.0);
    bundle.job.initial_he3 = ctrl.y_frac_he3 * y;
    bundle.job.initial_he4 = y - bundle.job.initial_he3;
    bundle.job.set_uniform_initial_composition = true;
    if params.alpha > 0.0 {
        bundle.controls.mixing_length_alpha = params.alpha;
    }
    let ov = &mut bundle.controls.mix_wind.overshoot_sets[0];
    if params.f_ov > 0.0 {
        ov.scheme = String::from("exponential");
        ov.zone_type = String::from("any");
        ov.zone_loc = String::from("any");
        ov.bdy_loc = String::from("any");
        ov.f = params.f_ov;
        ov.f0 = ctrl.f0_ov_div_f_ov.abs().max(0.1) * params.f_ov;
    } else {
        ov.scheme.clear();
        (ov.f, ov.f0) = (0.0, 0.0);
    }
    if params.my_param1 > 0.0 {
        bundle.controls.alpha_semiconvection = params.my_param1;
    }
    if params.my_param2 > 0.0 {
        bundle.controls.thermohaline_coeff = params.my_param2;
    }
    if params.my_param3 > 0.0 {
        bundle.controls.do_element_diffusion = true;
    }
    if bundle.job.create_pre_ms_model && ctrl.lnuc_div_l_limit >= 0.90 {
        bundle.job.create_pre_ms_model = false;
    }
    if !bundle.job.create_pre_ms_model && !bundle.job.set_initial_dt {
        bundle.job.set_initial_dt = true;
        bundle.job.years_for_initial_dt = ctrl.max_yrs_dt_when_cold.clamp(1.0e7, 5.0e7);
    }
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
    let mut bundle = base_bundle.clone();
    apply_params_to_bundle(&mut bundle, ctrl, &params);
    let mut extras = AsteroTrackExtras {
        ctrl, params, sample_number, case_dir,
        best_sample: None, fallback_model: None, bad_count: 0, eval_count: 0,
    };
    let mut model = init_or_update_model_from_bundle(&bundle, None, &mut extras)?;
    let stage = run_inlist_stage(&mut model, &bundle, &mut extras, max_steps_cap)?;
    let best = extras.best_sample.unwrap_or_else(|| {
        let fb_mod = extras.fallback_model.as_ref().map_or(&model, |(_, m)| m);
        eval_model_astero_sample(fb_mod, ctrl, params, sample_number, case_dir)
    });
    Ok((best, stage, model))
}

/// Runs the multi-track asteroseismic search driver (`astero_run_support.f90:36-191`).
pub fn run_star_astero(
    case_dir: &Path,
    entry_inlist: &str,
    max_steps_per_stage: Option<usize>,
    ctrl: &AsteroSearchControls,
) -> Result<AsteroSearchReport, StarError> {
    let base_bundle = InlistBundle::from_file_in_case_dir(case_dir, entry_inlist)?;
    let out_dir: PathBuf = case_dir.join(&ctrl.astero_results_directory);
    let _ = fs::create_dir_all(&out_dir);

    let mut samples: Vec<AsteroSampleResult> = Vec::new();
    let mut total_steps = 0usize;
    let mut eval_param_set = |p: AsteroParamSet| -> f64 {
        let s_num = samples.len() + 1;
        match run_single_astero_track(
            &base_bundle, ctrl, p, s_num, max_steps_per_stage, case_dir,
        ) {
            Ok((sample, stage, _)) => {
                total_steps += stage.steps_taken;
                let c2 = sample.chi2;
                write_sample_file(&out_dir, &sample);
                samples.push(sample);
                c2
            }
            Err(_) => 1.0e99,
        }
    };

    let st = ctrl.search_type.to_ascii_lowercase();
    let results_fname = match st.as_str() {
        "from_file" => {
            for p in ctrl.read_parameters_from_file(case_dir) {
                let _ = eval_param_set(p);
            }
            &ctrl.from_file_output_filename
        }
        "scan_grid" => {
            for p in build_scan_grid_params(ctrl) {
                let _ = eval_param_set(p);
            }
            &ctrl.scan_grid_output_filename
        }
        "simplex" => {
            run_simplex_driver(ctrl, &mut eval_param_set);
            &ctrl.simplex_output_filename
        }
        "newuoa" | "bobyqa" => {
            run_powell_driver(ctrl, st == "bobyqa", &mut eval_param_set);
            if st == "bobyqa" { &ctrl.bobyqa_output_filename } else { &ctrl.newuoa_output_filename }
        }
        _ => {
            let _ = eval_param_set(ctrl.first_params());
            &ctrl.scan_grid_output_filename
        }
    };
    write_search_results_table(&out_dir.join(results_fname), &samples);
    let best_sample_idx = samples
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.chi2.total_cmp(&b.chi2))
        .map(|(i, _)| i);
    let best_chi2 = best_sample_idx.map_or(f64::INFINITY, |i| samples[i].chi2);
    Ok(AsteroSearchReport {
        search_type: ctrl.search_type.clone(),
        samples, best_sample_idx, best_chi2, total_steps,
    })
}

fn build_scan_grid_params(ctrl: &AsteroSearchControls) -> Vec<AsteroParamSet> {
    let axis = |sp: AsteroSearchParam| -> Vec<f64> {
        if !sp.vary || sp.delta <= 0.0 || sp.max < sp.min {
            return vec![sp.first];
        }
        let mut v = Vec::new();
        let mut x = sp.min;
        while x <= sp.max + 1.0e-9 {
            v.push(x);
            x += sp.delta;
        }
        if v.is_empty() { vec![sp.first] } else { v }
    };
    let mut out = Vec::new();
    for m in axis(ctrl.mass) {
        for y in axis(ctrl.y) {
            for f in axis(ctrl.feh_param) {
                for a in axis(ctrl.alpha) {
                    for ov in axis(ctrl.f_ov) {
                        let mut p = ctrl.first_params();
                        (p.initial_mass, p.initial_y, p.initial_feh, p.alpha, p.f_ov) =
                            (m, y, f, a, ov);
                        out.push(p);
                    }
                }
            }
        }
    }
    out
}

fn active_search_params(ctrl: &AsteroSearchControls) -> Vec<(usize, AsteroSearchParam)> {
    let all = [
        ctrl.mass, ctrl.y, ctrl.feh_param, ctrl.alpha, ctrl.f_ov,
        ctrl.my_param[0], ctrl.my_param[1], ctrl.my_param[2],
    ];
    all.into_iter().enumerate().filter(|(_, p)| p.vary && p.max > p.min).collect()
}

fn apply_varied_slice(
    ctrl: &AsteroSearchControls,
    active: &[(usize, AsteroSearchParam)],
    vals: &[f64],
) -> AsteroParamSet {
    let mut p = ctrl.first_params();
    for (&(idx, _), &v) in active.iter().zip(vals.iter()) {
        match idx {
            0 => p.initial_mass = v, 1 => p.initial_y = v, 2 => p.initial_feh = v,
            3 => p.alpha = v, 4 => p.f_ov = v, 5 => p.my_param1 = v,
            6 => p.my_param2 = v, 7 => p.my_param3 = v, _ => {}
        }
    }
    p
}

fn run_simplex_driver(
    ctrl: &AsteroSearchControls,
    eval_fn: &mut impl FnMut(AsteroParamSet) -> f64,
) {
    let active = active_search_params(ctrl);
    if active.is_empty() { let _ = eval_fn(ctrl.first_params()); return; }
    let x_first: Vec<f64> = active.iter().map(|(_, p)| p.first).collect();
    let x_lower: Vec<f64> = active.iter().map(|(_, p)| p.min).collect();
    let x_upper: Vec<f64> = active.iter().map(|(_, p)| p.max).collect();
    let max_evals = ctrl.simplex_fcn_calls_max
        .min(ctrl.simplex_itermax.saturating_add(active.len())).max(1);
    let _ = simplex_minimize_with_options(
        &x_first, &x_lower, &x_upper, max_evals,
        ctrl.simplex_x_atol, ctrl.simplex_x_rtol, ctrl.simplex_chi2_tol,
        ctrl.simplex_centroid_weight_power, ctrl.simplex_enforce_bounds,
        ctrl.simplex_adaptive_random_search, ctrl.simplex_seed,
        |x| eval_fn(apply_varied_slice(ctrl, &active, x)),
    );
}

fn run_powell_driver(
    ctrl: &AsteroSearchControls,
    bounded: bool,
    eval_fn: &mut impl FnMut(AsteroParamSet) -> f64,
) {
    let active = active_search_params(ctrl);
    if active.is_empty() { let _ = eval_fn(ctrl.first_params()); return; }
    let n = active.len();
    let x_first: Vec<f64> = active.iter().map(|(_, p)| p.first).collect();
    let x_lower: Vec<f64> = active.iter().map(|(_, p)| p.min).collect();
    let x_upper: Vec<f64> = active.iter().map(|(_, p)| p.max).collect();
    let max_calls = (2 * n + 3).max(6);
    if bounded {
        let _ = bobyqa_minimize(
            &x_first, &x_lower, &x_upper, max_calls, ctrl.bobyqa_rho_beg, ctrl.bobyqa_rho_end,
            |x| eval_fn(apply_varied_slice(ctrl, &active, x)),
        );
    } else {
        let _ = newuoa_minimize(
            &x_first, &x_lower, &x_upper, max_calls, ctrl.newuoa_rho_beg, ctrl.newuoa_rho_end,
            |x| eval_fn(apply_varied_slice(ctrl, &active, x)),
        );
    }
}

fn write_sample_file(out_dir: &Path, s: &AsteroSampleResult) {
    let fname = out_dir.join(format!("sample_{:04}.data", s.sample_number));
    let mut buf = String::new();
    let _ = writeln!(buf, "sample_number {}\nchi2 {:26.16e}\nchi2_seismo {:26.16e}\nchi2_spectro {:26.16e}",
        s.sample_number, s.chi2, s.chi2_seismo, s.chi2_spectro);
    let _ = writeln!(buf, "age {:26.16e}\ninitial_mass {:26.16e}\ninitial_Y {:26.16e}\ninitial_FeH {:26.16e}",
        s.age_yr, s.params.initial_mass, s.params.initial_y, s.params.initial_feh);
    let _ = writeln!(buf, "alpha {:26.16e}\ndelta_nu {:26.16e}\nnu_max {:26.16e}",
        s.params.alpha, s.delta_nu_uhz, s.nu_max_uhz);
    let _ = fs::write(fname, buf);
}

fn write_search_results_table(out_path: &Path, samples: &[AsteroSampleResult]) {
    let mut buf = String::new();
    let _ = writeln!(buf, "{:>8} {:>20} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16}",
        "sample", "chi2", "initial_mass", "initial_Y", "initial_FeH", "alpha", "age", "delta_nu", "nu_max");
    for s in samples {
        let _ = writeln!(buf, "{:8} {:20.12e} {:16.8} {:16.8} {:16.8} {:16.8} {:16.8e} {:16.8} {:16.8}",
            s.sample_number, s.chi2, s.params.initial_mass, s.params.initial_y,
            s.params.initial_feh, s.params.alpha, s.age_yr, s.delta_nu_uhz, s.nu_max_uhz);
    }
    let _ = fs::write(out_path, buf);
}
