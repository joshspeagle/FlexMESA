// SPDX-License-Identifier: LGPL-3.0-or-later
//! Multi-stage `inlist` execution driver (`run_inlist_stage`,
//! `run_test_suite_case`), `run_star_extras` hook trait ([`StarExtras`]),
//! and bit-identical photo restart verification (`verify_photo_restart_parity`).

use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::case_staging::TempCaseStaging;
use crate::controls::StarError;
use crate::evolve::{star_evolve_step, star_evolve_step_with_extras};
use crate::history::{extract_history_row, HistoryRow};
use crate::hydro_vars::set_hydro_vars;
use crate::inlist::{InlistBundle, InlistLoadOptions};
use crate::mesa_profile::StarLogsWriter;
use crate::model::StarModel;
use crate::model_io::{
    compute_photo_checksum, load_photo_bytes, write_mod_string,
    write_photo_bytes,
};
use crate::rn_parser::{parse_rn_stages, RnStage};
use crate::rsp_eos_kap::RspChemCache;
use crate::star_job_init::{
    init_or_update_model_from_bundle_outcome, InitModelOutcome,
};
use crate::termination::{do_check_limits, LimitCheckOutcome};

/// Control flow directive returned by [`StarExtras`] step hooks (`keep_going`,
/// `redo`, `retry`, `terminate` in `star/public/star_def.inc`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtrasDirective {
    /// Continue normal evolution (`keep_going`).
    KeepGoing,
    /// Re-run the current step at identical timestep (`redo`).
    Redo,
    /// Force a step retry with reduced timestep (`retry`).
    Retry,
    /// Terminate the evolutionary stage with the given reason string (`terminate`).
    Terminate(String),
    /// Abort the stage immediately with a fatal error message (`#147`).
    Error(String),
}

/// Outcome returned by [`StarExtras::other_rsp_build_model`] (`RSP` envelope
/// construction or clean pre-evolution termination as in `rsp_check_2nd_crossing`).
#[derive(Debug, Clone)]
pub enum RspBuildOutcome {
    /// Constructed static `RSP` [`StarModel`].
    Model(Box<StarModel>),
    /// Terminated cleanly during model construction with termination code (`ierr = -1`).
    Terminate(String),
}

/// Rust trait equivalent of upstream MESA's `run_star_extras.f90` hooks and
/// `other_*` physics callbacks.
pub trait StarExtras {
    /// Invoked after `inlist` controls are loaded before `&star_job` relaxations.
    ///
    /// # Errors
    /// Returns [`StarError`] if custom control setup fails.
    fn extras_controls(
        &mut self,
        _model: &mut StarModel,
    ) -> Result<(), StarError> {
        Ok(())
    }

    /// Invoked once at the start of an evolutionary stage after model loading.
    ///
    /// # Errors
    /// Returns [`StarError`] if custom startup initialization fails.
    fn extras_startup(
        &mut self,
        _m: &mut StarModel,
        _r: bool,
    ) -> Result<(), StarError> {
        Ok(())
    }
    /// Invoked at the start of each evolutionary step before remeshing.
    fn extras_start_step(&mut self, _m: &mut StarModel) -> ExtrasDirective {
        ExtrasDirective::KeepGoing
    }
    /// Invoked immediately after a step converges to check custom limits.
    fn extras_check_model(&mut self, _m: &mut StarModel) -> ExtrasDirective {
        ExtrasDirective::KeepGoing
    }
    /// Invoked at the end of a step after history extraction.
    fn extras_finish_step(&mut self, _m: &mut StarModel) -> ExtrasDirective {
        ExtrasDirective::KeepGoing
    }
    /// Invoked once after the evolutionary loop terminates.
    ///
    /// # Errors
    /// Returns [`StarError`] if post-run export or verification fails.
    fn extras_after_evolve(
        &mut self,
        _m: &mut StarModel,
    ) -> Result<(), StarError> {
        Ok(())
    }
    /// Custom effective gravitational constant profile $G_{\mathrm{eff}}(k)$.
    ///
    /// # Errors
    /// Returns [`StarError`] if evaluating or validating `model.cgrav` fails.
    fn other_cgrav(&self, _m: &mut StarModel) -> Result<(), StarError> {
        Ok(())
    }
    /// Custom mixing-length parameter profile $\alpha_{\mathrm{MLT}}(k)$.
    fn other_alpha_mlt(&self, _m: &mut StarModel) {}
    /// Custom wind mass-loss rate $\dot{M}$ ($M_\odot\,\mathrm{yr^{-1}}$).
    ///
    /// # Errors
    /// Returns [`StarError`] if evaluating custom wind fails.
    fn other_wind(
        &self,
        _m: &StarModel,
        _l: f64,
        _ms: f64,
        _r: f64,
        _t: f64,
    ) -> Result<Option<f64>, StarError> {
        Ok(None)
    }
    /// Custom adjusted $\dot{M}$ ($\mathrm{g\,s^{-1}}$) or `None` to keep `current_mdot`.
    ///
    /// # Errors
    /// Returns [`StarError`] if evaluating custom $\dot{M}$ fails.
    fn other_adjust_mdot(
        &self,
        _m: &StarModel,
        _current_mdot: f64,
    ) -> Result<Option<f64>, StarError> {
        Ok(None)
    }
    /// Custom specific angular momentum torque $\dot{j}_{\mathrm{extra}}(k)$ ($\mathrm{cm^2\,s^{-2}}$).
    fn other_torque(&self, _m: &StarModel) -> Option<Vec<f64>> {
        None
    }
    /// Custom extra specific energy generation $\epsilon_{\mathrm{extra}}(k)$ ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    fn other_energy(&self, _m: &StarModel) -> Option<Vec<f64>> {
        None
    }
    /// Optional custom initial model builder (`use_other_build_initial_model`).
    ///
    /// # Errors
    /// Returns [`StarError`] if custom model construction fails.
    fn other_build_initial_model(
        &mut self,
        _bundle: &InlistBundle,
    ) -> Result<Option<StarModel>, StarError> {
        Ok(None)
    }
    /// Optional custom `RSP` model builder (`use_other_rsp_build_model`).
    ///
    /// # Errors
    /// Returns [`StarError`] if custom `RSP` envelope construction fails.
    fn other_rsp_build_model(
        &mut self,
        _bundle: &InlistBundle,
    ) -> Result<Option<RspBuildOutcome>, StarError> {
        Ok(None)
    }
    /// Optional custom `RSP` linear analysis (`use_other_rsp_linear_analysis`).
    ///
    /// Returns `Ok(true)` when the hook populated `lina_*`, `period`, and
    /// initial velocities, or `Ok(false)` when unimplemented.
    ///
    /// # Errors
    /// Returns [`StarError`] if custom linear pulsation analysis fails.
    fn other_rsp_linear_analysis(
        &mut self,
        _model: &mut StarModel,
        _chem: &RspChemCache,
        _log_dir: Option<&Path>,
    ) -> Result<bool, StarError> {
        Ok(false)
    }
    /// Extra per-step history columns appended to `history.data` rows
    /// (`data_for_extra_history_columns`).
    fn extra_history_columns(&self, _m: &StarModel) -> Vec<(String, f64)> {
        Vec::new()
    }
    /// Receives the staged temporary case directory path before stage execution
    /// so caller-supplied extras write auxiliary outputs inside the staged copy.
    fn set_staged_case_dir(&mut self, _case_dir: &Path) {}
}

/// Default no-op [`StarExtras`] implementation (`standard_run_star_extras.inc`).
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultStarExtras;
impl StarExtras for DefaultStarExtras {}

fn write_mod_if_missing(
    model: &StarModel,
    out_path: &Path,
) -> Result<(), StarError> {
    if !out_path.exists() || model.controls.rsp_flag {
        let mod_text = write_mod_string(model)?;
        fs::write(out_path, mod_text).map_err(|e| {
            StarError::IoError(format!(
                "failed to write {}: {e}",
                out_path.display()
            ))
        })?;
    }
    Ok(())
}

/// Telemetry and final state returned by a single `inlist` stage execution.
#[derive(Debug, Clone)]
pub struct StageRunReport {
    /// Entry `inlist` name executed in this stage.
    pub inlist_name: String,
    /// Number of evolutionary steps completed in this stage.
    pub steps_taken: usize,
    /// Total Henyey Newton-Raphson iterations across all steps in this stage.
    pub total_newton_iterations: usize,
    /// Total step retries across all steps in this stage.
    pub total_retries: usize,
    /// Termination code that ended the stage (e.g. `"max_age"`,
    /// `"xa_central_lower_limit"`, `"max_model_number"`).
    pub termination_code: String,
    /// Wall-clock runtime of the stage in seconds (`s`).
    pub runtime_sec: f64,
    /// Final [`HistoryRow`] summary at the end of the stage.
    pub final_history: HistoryRow,
    /// Complete sequence of [`HistoryRow`] records across the stage.
    pub history_rows: Vec<HistoryRow>,
}

/// Telemetry and stage reports returned by a full multi-stage `test_suite` case.
#[derive(Debug, Clone)]
pub struct CaseRunReport {
    /// Case directory name (e.g. `"1M_pre_ms_to_wd"`).
    pub case_name: String,
    /// Suite tier number (`1` for Tier 1 cases, `0` otherwise).
    pub tier: usize,
    /// Whether the case passed execution, physical assertions, and restart parity.
    pub passed: bool,
    /// Final stage termination code string.
    pub termination_code: String,
    /// Per-stage execution reports in chronological order.
    pub stages: Vec<StageRunReport>,
    /// Total evolutionary steps across all stages.
    pub total_steps: usize,
    /// Total Newton-Raphson iterations across all stages.
    pub total_newton_iterations: usize,
    /// Total retries across all stages.
    pub total_retries: usize,
    /// Total wall-clock runtime in seconds (`s`).
    pub total_runtime_sec: f64,
    /// Total wall-clock runtime in milliseconds (`ms`).
    pub wall_time_ms: f64,
    /// Final effective temperature $T_{\mathrm{eff}}$ (`K`).
    pub final_teff: f64,
    /// Final $\log_{10}(L / L_\odot)$.
    pub final_log_l: f64,
    /// Final $\log_{10}(R / R_\odot)$.
    pub final_log_r: f64,
    /// Final central $\log_{10} T_c$ (`K`).
    pub final_log_tc: f64,
    /// Final central $\log_{10} \rho_c$ ($\mathrm{g\,cm^{-3}}$).
    pub final_log_rhoc: f64,
    /// Final central ${}^1\mathrm{H}$ mass fraction $X_{\mathrm{c,H1}}$.
    pub final_center_h1: f64,
    /// Final central ${}^4\mathrm{He}$ mass fraction $X_{\mathrm{c,He4}}$.
    pub final_center_he4: f64,
    /// Whether bit-identical `.photo` restart parity passed.
    pub photo_restart_verified: bool,
    /// Whether post-run physical assertions passed.
    pub assertions_passed: bool,
}

/// Configuration for incremental MESA `LOGS/` file emission (`history.data`,
/// `profileN.data`, `profiles.index`) during stage and `test_suite` execution (`#163`).
#[derive(Debug, Clone, Copy, Default)]
pub struct StarLogsOptions<'a> {
    /// Optional root directory under which `<log_directory>/` files are written.
    pub output_root: Option<&'a Path>,
    /// Optional explicit output file path for `history.data` (used by `star_case_trace`).
    pub explicit_history_file: Option<&'a Path>,
    /// Optional override for `&controls` `history_interval` (`Some(1)` writes every step).
    pub history_interval_override: Option<usize>,
    /// Optional override for `&controls` `write_profiles_flag`.
    pub write_profiles_override: Option<bool>,
}

impl<'a> StarLogsOptions<'a> {
    /// Constructs [`StarLogsOptions`] targeting either a `LOGS/` root directory
    /// (`is_logs_dir = true`) or an explicit `history.data` path (`false`).
    #[must_use]
    pub fn from_target(
        out_target: &'a Path,
        is_logs_dir: bool,
        respect_intervals: bool,
        write_profiles: bool,
    ) -> Self {
        let history_interval_override = (!respect_intervals).then_some(1);
        if is_logs_dir {
            Self {
                output_root: Some(out_target),
                explicit_history_file: None,
                history_interval_override,
                write_profiles_override: Some(write_profiles),
            }
        } else {
            Self {
                output_root: None,
                explicit_history_file: Some(out_target),
                history_interval_override,
                write_profiles_override: Some(false),
            }
        }
    }

    /// Builds the per-stage [`StarLogsWriter`] configured by `self` (if any).
    ///
    /// # Errors
    /// Returns [`StarError`] if column resolution or output directory setup fails.
    pub fn build_writer(
        &self,
        model: &StarModel,
        case_dir: &Path,
        stage_logs_dir: Option<&str>,
    ) -> Result<Option<StarLogsWriter>, StarError> {
        if let Some(exp_hist) = self.explicit_history_file {
            Ok(Some(StarLogsWriter::for_explicit_history_file(
                model,
                case_dir,
                exp_hist.to_path_buf(),
                self.history_interval_override,
            )?))
        } else if let Some(out_root) = self.output_root {
            Ok(Some(StarLogsWriter::for_stage_with_logs_dir(
                model,
                case_dir,
                out_root,
                stage_logs_dir,
                self.history_interval_override,
                self.write_profiles_override,
            )?))
        } else {
            Ok(None)
        }
    }
}

/// Executes a single `inlist` stage on `model`, stepping until either a
/// built-in `&controls` termination limit, an [`ExtrasDirective::Terminate`],
/// or `max_steps_cap` is reached.
///
/// # Errors
/// Returns [`StarError`] if a step fails to converge after all retries or if
/// saving the final `.mod` file fails.
pub fn run_inlist_stage(
    model: &mut StarModel,
    bundle: &InlistBundle,
    extras: &mut dyn StarExtras,
    max_steps_cap: Option<usize>,
) -> Result<StageRunReport, StarError> {
    run_inlist_stage_with_logs(model, bundle, extras, max_steps_cap, None)
}

/// Executes a single `inlist` stage on `model`, optionally streaming incremental
/// MESA-format `LOGS/history.data`, `LOGS/profileN.data`, and `LOGS/profiles.index`
/// through `logs_writer` (`#163`).
///
/// # Errors
/// Returns [`StarError`] if step evolution, log emission, or final `.mod` saving fails.
pub fn run_inlist_stage_with_logs(
    model: &mut StarModel,
    bundle: &InlistBundle,
    extras: &mut dyn StarExtras,
    max_steps_cap: Option<usize>,
    mut logs_writer: Option<&mut StarLogsWriter>,
) -> Result<StageRunReport, StarError> {
    let t0 = Instant::now();
    let (mut steps_taken, mut total_newton_iterations, mut total_retries) =
        (0, 0, 0);
    let mut history_rows = Vec::new();
    let mut termination_code = String::from("max_steps_cap");

    let job_step_cap = (model.job.steps_to_take_before_terminate >= 0)
        .then_some(model.job.steps_to_take_before_terminate as usize);
    let cap = max_steps_cap
        .into_iter()
        .chain(job_step_cap)
        .min()
        .unwrap_or(50_000);
    if job_step_cap.is_some_and(|b| b <= cap) {
        termination_code = String::from("steps_to_take_before_terminate");
    }
    let do_remesh =
        model.controls.okay_to_remesh && model.controls.mesh_delta_coeff > 0.0;
    let op_split_burn = model.controls.op_split_burn;

    let init_hr = extract_history_row(model);
    if cap == 0 {
        if job_step_cap != Some(0) {
            if let LimitCheckOutcome::Terminate { code, .. } =
                do_check_limits(model, None, true)?
            {
                termination_code = code;
            }
        }
        if let Some(lw) = logs_writer.as_deref_mut() {
            let extra_cols = if lw.is_history_due(model, true, true) {
                extras.extra_history_columns(model)
            } else {
                Vec::new()
            };
            lw.record_step_with_extra(model, true, true, &extra_cols)?;
        }
    } else {
        for _ in 0..cap {
            match extras.extras_start_step(model) {
                ExtrasDirective::Terminate(reason) => {
                    termination_code = reason;
                    break;
                }
                ExtrasDirective::Error(msg) => {
                    return Err(StarError::ConvergenceFailure(msg));
                }
                ExtrasDirective::Retry | ExtrasDirective::Redo => {
                    return Err(StarError::ConvergenceFailure(String::from(
                        "extras step retry/redo is not yet implemented (#152)",
                    )));
                }
                ExtrasDirective::KeepGoing => {}
            }
            if model.controls.use_other_cgrav {
                extras.other_cgrav(model)?;
            }
            if model.controls.use_other_alpha_mlt {
                extras.other_alpha_mlt(model);
            }
            if model.controls.use_other_torque {
                if let Some(jdot) = extras.other_torque(model) {
                    if jdot.len() != model.nz {
                        return Err(StarError::InvalidInput(format!(
                            "other_torque returned {} values for nz = {}",
                            jdot.len(),
                            model.nz
                        )));
                    }
                    for (j_k, &dj_k) in model.j_rot.iter_mut().zip(jdot.iter())
                    {
                        *j_k = (*j_k + dj_k * model.dt).max(1.0e10);
                    }
                }
            }

            let step_rep = match star_evolve_step_with_extras(
                model,
                do_remesh,
                op_split_burn,
                true,
                extras,
            ) {
                Ok(rep) => rep,
                Err(StarError::Terminated { reason, .. })
                    if reason == "max_age" =>
                {
                    termination_code = reason;
                    break;
                }
                Err(e) => return Err(e),
            };
            steps_taken += 1;
            total_newton_iterations += step_rep.iterations;
            total_retries += step_rep.retries;

            let check_dir = extras.extras_check_model(model);
            let finish_dir = extras.extras_finish_step(model);
            model.mstar_start = model.mstar;
            if !model.controls.rsp_flag {
                model.sync_mesh_and_pack()?;
                set_hydro_vars(model, false, false)?;
                crate::struct_burn_mix::refresh_mixing_state(
                    model,
                    model.dt,
                    &[],
                );
                model.save_start_values()?;
            }
            let hr = extract_history_row(model);
            history_rows.push(hr);

            for dir in [&check_dir, &finish_dir] {
                if let ExtrasDirective::Error(msg) = dir {
                    return Err(StarError::ConvergenceFailure(msg.clone()));
                }
            }
            let mut will_terminate = false;
            for dir in [check_dir, finish_dir] {
                match dir {
                    ExtrasDirective::Error(msg) => {
                        return Err(StarError::ConvergenceFailure(msg));
                    }
                    ExtrasDirective::Retry | ExtrasDirective::Redo => {
                        return Err(StarError::ConvergenceFailure(
                            String::from(
                                "extras step retry/redo is not yet implemented (#152)",
                            ),
                        ));
                    }
                    ExtrasDirective::Terminate(reason) => {
                        termination_code = reason;
                        will_terminate = true;
                        break;
                    }
                    ExtrasDirective::KeepGoing => {}
                }
            }
            if !will_terminate {
                if let LimitCheckOutcome::Terminate { code, .. } =
                    do_check_limits(model, None, false)?
                {
                    termination_code = code;
                    will_terminate = true;
                }
            }
            let is_last_cap_step = steps_taken == cap;
            let is_terminating = will_terminate || is_last_cap_step;
            if let Some(lw) = logs_writer.as_deref_mut() {
                let extra_cols = if lw.is_history_due(
                    model,
                    steps_taken == 1,
                    is_terminating,
                ) {
                    extras.extra_history_columns(model)
                } else {
                    Vec::new()
                };
                lw.record_step_with_extra(
                    model,
                    steps_taken == 1,
                    is_terminating,
                    &extra_cols,
                )?;
            }
            if will_terminate {
                break;
            }
        }
    }

    extras.extras_after_evolve(model)?;

    if model.job.save_model_when_terminate
        && !model.job.save_model_filename.is_empty()
        && !model
            .job
            .save_model_filename
            .eq_ignore_ascii_case("undefined")
    {
        let out_path = bundle.case_dir.join(&model.job.save_model_filename);
        write_mod_if_missing(model, &out_path)?;
    }

    let final_history = history_rows.last().copied().unwrap_or(init_hr);

    Ok(StageRunReport {
        inlist_name: bundle.entry_inlist.clone(),
        steps_taken,
        total_newton_iterations,
        total_retries,
        termination_code,
        runtime_sec: t0.elapsed().as_secs_f64(),
        final_history,
        history_rows,
    })
}

/// Verifies bit-identical `.photo` restart parity on `model`:
/// 1. Clones `model`, advances 1 warm-up step if `lnd_old` is unpopulated so
///    H211 filter memory (`dt_limit_ratio_old`) and `*_old` arrays are active,
///    then serializes `.photo` bytes and deserializes a fresh `restarted` model.
/// 2. Evolves both `continued` and `restarted` for `n_verify_steps` steps.
/// 3. Asserts `compute_photo_checksum` of both serialized states is equal.
///
/// # Errors
/// Returns [`StarError`] if `.photo` serialization, step evolution, or
/// bit-identical checksum comparison fails.
pub fn verify_photo_restart_parity(
    model: &StarModel,
    n_verify_steps: usize,
) -> Result<(u64, u64), StarError> {
    let mut continued = model.clone();
    let do_remesh = continued.controls.okay_to_remesh
        && continued.controls.mesh_delta_coeff > 0.0;
    let op_split = continued.controls.op_split_burn;
    if continued.lnd_old.is_empty() {
        star_evolve_step(&mut continued, do_remesh, op_split, true)?;
    }

    let photo_bytes = write_photo_bytes(&continued);
    let mut restarted = load_photo_bytes(&photo_bytes)?;

    for _ in 0..n_verify_steps {
        star_evolve_step(&mut continued, do_remesh, op_split, true)?;
        star_evolve_step(&mut restarted, do_remesh, op_split, true)?;
    }

    let (b_cont, b_rest) =
        (write_photo_bytes(&continued), write_photo_bytes(&restarted));
    let (c_cont, c_rest) = (
        compute_photo_checksum(&b_cont),
        compute_photo_checksum(&b_rest),
    );
    if c_cont != c_rest {
        let diff_at = b_cont
            .iter()
            .zip(&b_rest)
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| b_cont.len().min(b_rest.len()));
        return Err(StarError::IoError(format!(
            "photo restart parity mismatch at byte {diff_at}: {c_cont:#x} != {c_rest:#x}"
        )));
    }
    Ok((c_cont, c_rest))
}

/// Executes all stages of an upstream MESA `test_suite` case directory
/// according to its `./rn` script (`skip_optional` controls whether
/// `MESA_SKIP_OPTIONAL` branches are skipped).
///
/// # Errors
/// Returns [`StarError`] if `inlist` parsing or stage execution fails.
pub fn run_test_suite_case(
    orig_case_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> Result<CaseRunReport, StarError> {
    run_test_suite_case_with_logs(
        orig_case_dir,
        skip_optional,
        max_steps_per_stage,
        verify_restart,
        StarLogsOptions::default(),
    )
}

/// Executes all stages of an upstream MESA `test_suite` case directory while
/// optionally streaming incremental MESA-format `LOGS/history.data`,
/// `LOGS/profileN.data`, and `LOGS/profiles.index` files according to `logs_opts` (`#163`).
///
/// Uses default strict inlist validation (`allow_unsupported_controls = false`).
///
/// # Errors
/// Returns [`StarError`] if `inlist` parsing, stage execution, or log output fails.
pub fn run_test_suite_case_with_logs(
    orig_case_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    logs_opts: StarLogsOptions<'_>,
) -> Result<CaseRunReport, StarError> {
    run_test_suite_case_with_options(
        orig_case_dir,
        skip_optional,
        max_steps_per_stage,
        verify_restart,
        logs_opts,
        InlistLoadOptions::default(),
    )
}

/// Executes all stages of an upstream MESA `test_suite` case directory with
/// explicit `logs_opts` and `load_opts` (`#147`).
///
/// # Errors
/// Returns [`StarError`] if `inlist` parsing, stage execution, or log output fails.
pub fn run_test_suite_case_with_options(
    orig_case_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    logs_opts: StarLogsOptions<'_>,
    load_opts: InlistLoadOptions,
) -> Result<CaseRunReport, StarError> {
    let mut extras = DefaultStarExtras;
    run_test_suite_case_with_extras(
        orig_case_dir,
        skip_optional,
        max_steps_per_stage,
        verify_restart,
        logs_opts,
        load_opts,
        &mut extras,
    )
}

/// Executes all stages of an upstream MESA `test_suite` case directory with
/// explicit `logs_opts`, `load_opts`, and caller-supplied [`StarExtras`] hooks.
///
/// # Errors
/// Returns [`StarError`] if `inlist` parsing, stage execution, or log output fails.
pub fn run_test_suite_case_with_extras(
    orig_case_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    logs_opts: StarLogsOptions<'_>,
    load_opts: InlistLoadOptions,
    extras: &mut dyn StarExtras,
) -> Result<CaseRunReport, StarError> {
    let t0 = Instant::now();
    let staging = TempCaseStaging::new(orig_case_dir)?;
    let case_dir = staging.staged_dir.as_path();
    extras.set_staged_case_dir(case_dir);
    let case_name = case_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let rn_stages = parse_rn_stages(case_dir, skip_optional)?;

    let mut current_model: Option<StarModel> = None;
    let mut stage_reports = Vec::with_capacity(rn_stages.len());
    let (mut total_steps, mut total_newton, mut total_retries) = (0, 0, 0);
    let mut photo_restart_verified = false;
    let mut terminated_in_init = false;

    for stage in &rn_stages {
        match stage {
            RnStage::CopyCheckpoint { src_mod, dst_mod } => {
                current_model = None;
                let (src, dst) =
                    (case_dir.join(src_mod), case_dir.join(dst_mod));
                if !src.exists() {
                    return Err(StarError::IoError(format!(
                        "missing checkpoint '{}' for copy to '{}'",
                        src.display(),
                        dst.display()
                    )));
                }
                fs::copy(&src, &dst).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to copy checkpoint {} -> {}: {e}",
                        src.display(),
                        dst.display()
                    ))
                })?;
            }
            RnStage::DoOne {
                inlist,
                output_mod,
                logs_dir,
            } => {
                let bundle = InlistBundle::from_file_in_case_dir_with_options(
                    case_dir, inlist, load_opts,
                )?;

                let mut model = match init_or_update_model_from_bundle_outcome(
                    &bundle,
                    current_model.take(),
                    extras,
                )? {
                    InitModelOutcome::Model(m) => *m,
                    InitModelOutcome::Terminate(reason) => {
                        terminated_in_init = true;
                        stage_reports.push(StageRunReport {
                            inlist_name: bundle.entry_inlist.clone(),
                            steps_taken: 0,
                            total_newton_iterations: 0,
                            total_retries: 0,
                            termination_code: reason,
                            runtime_sec: t0.elapsed().as_secs_f64(),
                            final_history: HistoryRow::default(),
                            history_rows: Vec::new(),
                        });
                        break;
                    }
                };

                let mut stage_logs = logs_opts.build_writer(
                    &model,
                    case_dir,
                    logs_dir.as_deref(),
                )?;

                let rep = run_inlist_stage_with_logs(
                    &mut model,
                    &bundle,
                    extras,
                    max_steps_per_stage,
                    stage_logs.as_mut(),
                )?;

                if verify_restart && !photo_restart_verified {
                    let (c1, c2) = verify_photo_restart_parity(&model, 1)?;
                    photo_restart_verified = c1 == c2;
                }

                if let Some(out_name) = output_mod {
                    let out_path = case_dir.join(out_name);
                    write_mod_if_missing(&model, &out_path)?;
                }
                total_steps += rep.steps_taken;
                total_newton += rep.total_newton_iterations;
                total_retries += rep.total_retries;
                stage_reports.push(rep);
                current_model = Some(model);
            }
        }
    }

    let runtime_sec = t0.elapsed().as_secs_f64();
    let assertions_passed = terminated_in_init
        || current_model
            .as_ref()
            .is_some_and(|m| m.nz >= 2 && m.teff.is_finite() && m.teff > 0.0);
    let final_hr = stage_reports
        .last()
        .map(|s| s.final_history)
        .or_else(|| current_model.as_ref().map(extract_history_row));
    let termination_code = stage_reports.last().map_or_else(
        || String::from("no_stages_executed"),
        |s| s.termination_code.clone(),
    );
    let reached_terminal_stop = !termination_code.is_empty()
        && termination_code != "max_steps_cap"
        && termination_code != "no_stages_executed";
    let passed = assertions_passed
        && reached_terminal_stop
        && (terminated_in_init || !verify_restart || photo_restart_verified);

    Ok(CaseRunReport {
        case_name,
        tier: 0,
        passed,
        termination_code,
        stages: stage_reports,
        total_steps,
        total_newton_iterations: total_newton,
        total_retries,
        total_runtime_sec: runtime_sec,
        wall_time_ms: runtime_sec * 1000.0,
        final_teff: final_hr.map_or(0.0, |h| flexmesa_math::exp10(h.log_teff)),
        final_log_l: final_hr.map_or(0.0, |h| h.log_l),
        final_log_r: final_hr.map_or(0.0, |h| h.log_r),
        final_log_tc: final_hr.map_or(0.0, |h| h.log_center_t),
        final_log_rhoc: final_hr.map_or(0.0, |h| h.log_center_rho),
        final_center_h1: final_hr.map_or(0.0, |h| h.center_h1),
        final_center_he4: final_hr.map_or(0.0, |h| h.center_he4),
        photo_restart_verified,
        assertions_passed,
    })
}
