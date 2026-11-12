// SPDX-License-Identifier: LGPL-3.0-or-later
//! Multi-stage `inlist` execution driver (`run_inlist_stage`,
//! `run_test_suite_case`), `run_star_extras` hook trait ([`StarExtras`]),
//! and bit-identical photo restart verification (`verify_photo_restart_parity`).

use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::case_staging::TempCaseStaging;
use crate::controls::StarError;
use crate::evolve::star_evolve_step;
use crate::history::{extract_history_row, HistoryRow};
use crate::hydro_vars::set_hydro_vars;
use crate::inlist::{parse_rn_stages, InlistBundle, RnStage};
use crate::model::StarModel;
use crate::model_io::{
    compute_photo_checksum, read_photo_bytes, write_mod_string,
    write_photo_bytes,
};
use crate::star_job_init::init_or_update_model_from_bundle;
use crate::termination::check_termination_limits_with_history;
use crate::tier1_extras::{
    is_tier1_case, verify_tier1_case_assertions, Tier1CaseExtras,
};
use crate::tier2_extras::{
    is_tier2_case, verify_tier2_case_assertions, Tier2CaseExtras,
};
use crate::tier3_extras::{
    is_tier3_case, verify_tier3_case_assertions, Tier3CaseExtras,
};
use crate::tier4_extras::{
    is_tier4_case, verify_tier4_case_assertions, Tier4CaseExtras,
};
use crate::tier5_extras::{
    is_tier5_star_case, verify_tier5_star_case_assertions, Tier5StarCaseExtras,
};

/// Control flow directive returned by [`StarExtras`] step hooks (`keep_going`,
/// `retry`, `terminate` in `star/public/star_def.inc`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtrasDirective {
    /// Continue normal evolution (`keep_going`).
    KeepGoing,
    /// Force a step retry with reduced timestep (`retry`).
    Retry,
    /// Terminate the evolutionary stage with the given reason string (`terminate`).
    Terminate(String),
}

/// Rust trait equivalent of upstream MESA's `run_star_extras.f90` hooks and
/// `other_*` physics callbacks.
pub trait StarExtras {
    /// Invoked after `inlist` controls are loaded before model initialization.
    fn extras_controls(&mut self, _model: &mut StarModel) {}

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
    fn extras_after_evolve(&mut self, _m: &mut StarModel) {}
    /// Custom effective gravitational constant profile $G_{\mathrm{eff}}(k)$.
    fn other_cgrav(&self, _m: &mut StarModel) {}
    /// Custom mixing-length parameter profile $\alpha_{\mathrm{MLT}}(k)$.
    fn other_alpha_mlt(&self, _m: &mut StarModel) {}
    /// Custom wind mass-loss rate $\dot{M}$ ($M_\odot\,\mathrm{yr^{-1}}$).
    fn other_wind(
        &self,
        _m: &StarModel,
        _l: f64,
        _ms: f64,
        _r: f64,
        _t: f64,
    ) -> Option<f64> {
        None
    }
    /// Custom multiplier or adjustment on $\dot{M}$ ($\mathrm{g\,s^{-1}}$).
    fn other_adjust_mdot(&self, _m: &StarModel, current_mdot: f64) -> f64 {
        current_mdot
    }
    /// Custom specific angular momentum torque $\dot{j}_{\mathrm{extra}}(k)$ ($\mathrm{cm^2\,s^{-2}}$).
    fn other_torque(&self, _m: &StarModel) -> Option<Vec<f64>> {
        None
    }
    /// Custom extra specific energy generation $\epsilon_{\mathrm{extra}}(k)$ ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    fn other_energy(&self, _m: &StarModel) -> Option<Vec<f64>> {
        None
    }
}

/// Default no-op [`StarExtras`] implementation (`standard_run_star_extras.inc`).
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultStarExtras;
impl StarExtras for DefaultStarExtras {}

/// Tolman-Oppenheimer-Volkoff (TOV) post-Newtonian general-relativistic
/// effective gravitational constant hook (`other_cgrav` in `ns_h`, `ns_he`,
/// `tzo`, and `neutron_star_envelope`).
#[derive(Debug, Clone, Copy, Default)]
pub struct TovGrStarExtras;
impl StarExtras for TovGrStarExtras {
    fn other_cgrav(&self, model: &mut StarModel) {
        model.cgrav = crate::tier3_physics::eval_tov_effective_cgrav(model);
    }
}

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
    let t0 = Instant::now();
    let (mut steps_taken, mut total_newton_iterations, mut total_retries) =
        (0, 0, 0);
    let mut history_rows = Vec::new();
    let mut termination_code = String::from("max_steps_cap");

    let job_step_cap = model.job.steps_to_take_before_terminate;
    let cap = max_steps_cap
        .into_iter()
        .chain(job_step_cap)
        .min()
        .unwrap_or(50_000);
    if job_step_cap.is_some_and(|b| b <= cap) {
        termination_code = String::from("steps_to_take_before_terminate");
    }
    let do_remesh = model.controls.mesh_delta_coeff > 0.0;
    let op_split_burn = model.controls.op_split_burn;

    let init_hr = extract_history_row(model);
    if cap == 0 {
        if job_step_cap != Some(0) {
            if let Some(code) =
                check_termination_limits_with_history(model, Some(&init_hr))
            {
                termination_code = code;
            }
        }
    } else {
        for _ in 0..cap {
            if let ExtrasDirective::Terminate(reason) =
                extras.extras_start_step(model)
            {
                termination_code = reason;
                break;
            }
            if model.controls.use_other_cgrav {
                extras.other_cgrav(model);
            }
            if model.controls.use_other_alpha_mlt {
                extras.other_alpha_mlt(model);
            }
            if model.controls.use_other_adjust_mdot {
                model.mstar_dot =
                    extras.other_adjust_mdot(model, model.mstar_dot);
            }
            if model.controls.use_other_torque {
                if let Some(jdot) = extras.other_torque(model) {
                    for (j_k, &dj_k) in model.j_rot.iter_mut().zip(jdot.iter())
                    {
                        *j_k = (*j_k + dj_k * model.dt).max(1.0e10);
                    }
                }
            }

            let step_rep =
                star_evolve_step(model, do_remesh, op_split_burn, true)?;
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
                model.save_start_values();
            }
            let hr = extract_history_row(model);
            history_rows.push(hr);

            for dir in [check_dir, finish_dir] {
                if let ExtrasDirective::Terminate(reason) = dir {
                    termination_code = reason;
                    break;
                }
            }
            if termination_code != "max_steps_cap"
                && termination_code != "steps_to_take_before_terminate"
            {
                break;
            }
            if let Some(code) =
                check_termination_limits_with_history(model, Some(&hr))
            {
                termination_code = code;
                break;
            }
        }
    }

    extras.extras_after_evolve(model);

    if model.job.save_model_when_terminate
        && !model.job.save_model_filename.is_empty()
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
    let do_remesh = continued.controls.mesh_delta_coeff > 0.0;
    let op_split = continued.controls.op_split_burn;
    if continued.lnd_old.is_empty() {
        star_evolve_step(&mut continued, do_remesh, op_split, true)?;
    }

    let photo_bytes = write_photo_bytes(&continued);
    let mut restarted = read_photo_bytes(
        &photo_bytes,
        continued.controls.clone(),
        continued.job.clone(),
    )?;

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
    let t0 = Instant::now();
    let staging = TempCaseStaging::new(orig_case_dir)?;
    let case_dir = staging.staged_dir.as_path();
    let case_name = case_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let is_multi_star = case_dir.join("inlist_multi_stars_job").exists();
    let rn_stages = parse_rn_stages(case_dir, skip_optional)?;
    let tier = if is_tier1_case(&case_name) {
        1
    } else if is_tier2_case(&case_name) {
        2
    } else if is_tier3_case(&case_name) {
        3
    } else if is_tier4_case(&case_name) {
        4
    } else if is_tier5_star_case(&case_name) {
        5
    } else {
        0
    };

    let mut current_model: Option<StarModel> = None;
    let mut twin_star1_hr: Option<HistoryRow> = None;
    let mut stage_reports = Vec::with_capacity(rn_stages.len());
    let (mut total_steps, mut total_newton, mut total_retries) = (0, 0, 0);
    let mut photo_restart_verified = false;

    for stage in &rn_stages {
        match stage {
            RnStage::CopyCheckpoint { src_mod, dst_mod } => {
                current_model = None;
                let (src, dst) =
                    (case_dir.join(src_mod), case_dir.join(dst_mod));
                if src.exists() {
                    fs::copy(&src, &dst).map_err(|e| {
                        StarError::IoError(format!(
                            "failed to copy checkpoint {} -> {}: {e}",
                            src.display(),
                            dst.display()
                        ))
                    })?;
                }
            }
            RnStage::DoOne {
                inlist, output_mod, ..
            } => {
                if is_multi_star {
                    if let Some(m1) = current_model.take() {
                        if case_name == "twin_studies" {
                            twin_star1_hr = Some(extract_history_row(&m1));
                        }
                    }
                }
                if InlistBundle::resolve_inlist_path(case_dir, inlist).is_none()
                {
                    continue;
                }
                let bundle =
                    InlistBundle::from_file_in_case_dir(case_dir, inlist)?;
                let use_tov = bundle.controls.use_other_cgrav;
                let mut extras: Box<dyn StarExtras> = match tier {
                    1 => Box::new(Tier1CaseExtras::new(&case_name)),
                    2 => Box::new(Tier2CaseExtras::new(&case_name)),
                    3 => Box::new(Tier3CaseExtras::new(&case_name)),
                    4 => Box::new(Tier4CaseExtras::with_case_dir(
                        &case_name, case_dir,
                    )),
                    5 => Box::new(Tier5StarCaseExtras::with_case_dir(
                        &case_name, case_dir,
                    )),
                    _ if use_tov => Box::new(TovGrStarExtras),
                    _ => Box::new(DefaultStarExtras),
                };

                let mut model = init_or_update_model_from_bundle(
                    &bundle,
                    current_model.take(),
                    extras.as_mut(),
                )?;

                let rep = run_inlist_stage(
                    &mut model,
                    &bundle,
                    extras.as_mut(),
                    max_steps_per_stage,
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
    let twin_match = twin_star1_hr
        .zip(current_model.as_ref().map(extract_history_row))
        .is_none_or(|(hr1, hr2)| {
            (hr1.star_mass - hr2.star_mass).abs() < 0.5
                && (hr1.log_teff - hr2.log_teff).abs() < 0.2
        });
    let assertions_passed = twin_match
        && current_model.as_ref().is_some_and(|m| match tier {
            1 => verify_tier1_case_assertions(m, &case_name),
            2 => verify_tier2_case_assertions(m, &case_name),
            3 => verify_tier3_case_assertions(m, &case_name),
            4 => verify_tier4_case_assertions(m, &case_name),
            5 => verify_tier5_star_case_assertions(m, &case_name, case_dir),
            _ => m.nz >= 2 && m.teff.is_finite() && m.teff > 0.0,
        });
    let final_hr = stage_reports
        .last()
        .map(|s| s.final_history)
        .or_else(|| current_model.as_ref().map(extract_history_row));
    let termination_code = stage_reports.last().map_or_else(
        || String::from("completed"),
        |s| s.termination_code.clone(),
    );
    let passed =
        assertions_passed && (!verify_restart || photo_restart_verified);

    Ok(CaseRunReport {
        case_name,
        tier,
        passed,
        termination_code,
        stages: stage_reports,
        total_steps,
        total_newton_iterations: total_newton,
        total_retries,
        total_runtime_sec: runtime_sec,
        wall_time_ms: runtime_sec * 1000.0,
        final_teff: final_hr.map_or(0.0, |h| 10.0_f64.powf(h.log_teff)),
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

/// Runs a slice of `star/test_suite/<case>` directories under `mesa_dir` in
/// order and returns their [`CaseRunReport`] records.
///
/// # Errors
/// Returns [`StarError`] if any case in `cases` fails to execute.
pub fn run_case_list(
    mesa_dir: &Path,
    cases: &[&str],
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> Result<Vec<CaseRunReport>, StarError> {
    let suite_dir = mesa_dir.join("star").join("test_suite");
    let mut reports = Vec::with_capacity(cases.len());
    for &case_name in cases {
        let case_dir = suite_dir.join(case_name);
        reports.push(run_test_suite_case(
            &case_dir,
            true,
            max_steps_per_stage,
            verify_restart,
        )?);
    }
    Ok(reports)
}
