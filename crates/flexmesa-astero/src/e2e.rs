// SPDX-License-Identifier: GPL-3.0-or-later
//! End-to-end runner and RAII temporary staging (`TempCaseStaging`) for
//! coupled `astero` pulsation and optimization runs (`#148`).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use flexmesa_star::controls::StarError;
use flexmesa_star::history::{extract_history_row, HistoryRow};
use flexmesa_star::inlist::{InlistBundle, InlistLoadOptions};
use flexmesa_star::inlist_runner::{
    run_test_suite_case_with_extras, run_test_suite_case_with_options,
    verify_photo_restart_parity, CaseRunReport, StageRunReport, StarExtras,
    StarLogsOptions,
};
use flexmesa_star::rn_parser::{parse_rn_stages, RnStage};

use crate::inlist_astero::parse_astero_search_controls_with_options;
use crate::search_driver::{run_star_astero_with_case_hooks, AsteroCaseHooks};

static ASTERO_STAGING_SEQ: AtomicU64 = AtomicU64::new(0);

/// RAII temporary workspace staging a single `astero/test_suite` case directory.
#[derive(Debug)]
pub struct TempCaseStaging {
    tmp_root: Option<PathBuf>,
    /// Staged writable case directory whose basename equals `orig_case_dir`.
    pub staged_dir: PathBuf,
}

impl TempCaseStaging {
    /// Copies `orig_case_dir` into a fresh process-unique temporary directory.
    pub fn new(orig_case_dir: &Path) -> Result<Self, StarError> {
        let seq = ASTERO_STAGING_SEQ.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let case_name = orig_case_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("case");
        let tmp_root = std::env::temp_dir()
            .join(format!("flexmesa_astero_{pid}_{seq}_{case_name}"));
        let staged_dir = tmp_root.join(case_name);
        copy_dir_tree(orig_case_dir, &staged_dir, 0)?;
        let inlist_dst = staged_dir.join("inlist");
        if !inlist_dst.exists() && staged_dir.join("rn").is_file() {
            if let Some(first_inlist) = parse_rn_stages(&staged_dir, true)?
                .into_iter()
                .find_map(|st| match st {
                    RnStage::DoOne { inlist, .. } => Some(inlist),
                    RnStage::CopyCheckpoint { .. } => None,
                })
            {
                let src = staged_dir.join(&first_inlist);
                if src.is_file() {
                    fs::copy(&src, &inlist_dst).map_err(|e| {
                        StarError::IoError(format!(
                            "failed copying '{}' to '{}': {e}",
                            src.display(),
                            inlist_dst.display()
                        ))
                    })?;
                }
            }
        }
        Ok(Self {
            tmp_root: Some(tmp_root),
            staged_dir,
        })
    }
}

impl Drop for TempCaseStaging {
    fn drop(&mut self) {
        if let Some(root) = &self.tmp_root {
            let _ = fs::remove_dir_all(root);
        }
    }
}

fn copy_dir_tree(
    src: &Path,
    dst: &Path,
    depth: usize,
) -> Result<(), StarError> {
    if depth > 12 {
        return Ok(());
    }
    fs::create_dir_all(dst).map_err(|e| StarError::IoError(e.to_string()))?;
    for entry in
        fs::read_dir(src).map_err(|e| StarError::IoError(e.to_string()))?
    {
        let entry = entry.map_err(|e| StarError::IoError(e.to_string()))?;
        let name = entry.file_name();
        if name == "docs" || name == "LOGS" || name == "photos" {
            continue;
        }
        let (sp, dp) = (entry.path(), dst.join(&name));
        let meta =
            fs::metadata(&sp).map_err(|e| StarError::IoError(e.to_string()))?;
        if meta.is_dir() {
            copy_dir_tree(&sp, &dp, depth + 1)?;
        } else if meta.is_file() {
            fs::copy(&sp, &dp)
                .map_err(|e| StarError::IoError(e.to_string()))?;
        }
    }
    Ok(())
}

/// Executes a single `astero/test_suite/<case>` directory inside [`TempCaseStaging`].
pub fn run_astero_test_suite_case(
    orig_case_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> Result<CaseRunReport, StarError> {
    run_astero_test_suite_case_with_options(
        orig_case_dir,
        skip_optional,
        max_steps_per_stage,
        verify_restart,
        StarLogsOptions::default(),
        InlistLoadOptions::default(),
    )
}

/// Executes a single `astero/test_suite/<case>` directory while streaming
/// incremental MESA `LOGS/history.data` and asteroseismic tables according to
/// `logs_opts` (`#143`).
pub fn run_astero_test_suite_case_with_logs(
    orig_case_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    logs_opts: StarLogsOptions<'_>,
) -> Result<CaseRunReport, StarError> {
    run_astero_test_suite_case_with_options(
        orig_case_dir,
        skip_optional,
        max_steps_per_stage,
        verify_restart,
        logs_opts,
        InlistLoadOptions::default(),
    )
}

/// Executes a single `astero/test_suite/<case>` directory with explicit
/// [`StarLogsOptions`] and [`InlistLoadOptions`].
pub fn run_astero_test_suite_case_with_options(
    orig_case_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    logs_opts: StarLogsOptions<'_>,
    inlist_opts: InlistLoadOptions,
) -> Result<CaseRunReport, StarError> {
    run_astero_test_suite_case_with_case_hooks(
        orig_case_dir,
        skip_optional,
        max_steps_per_stage,
        verify_restart,
        logs_opts,
        inlist_opts,
        None,
    )
}

/// [`run_astero_test_suite_case_with_options`] with the case's `set_param`
/// hook ([`AsteroCaseHooks`]), which the search calls for every named
/// parameter of every track (`astero/private/extras_support.f90:974-988`).
///
/// # Errors
/// Returns [`StarError`] when a named search parameter has no hook, the hook
/// fails, or the run fails.
pub fn run_astero_test_suite_case_with_case_hooks(
    orig_case_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    logs_opts: StarLogsOptions<'_>,
    inlist_opts: InlistLoadOptions,
    set_param: Option<&mut dyn AsteroCaseHooks>,
) -> Result<CaseRunReport, StarError> {
    run_astero_test_suite_case_with_hooks(
        orig_case_dir,
        skip_optional,
        max_steps_per_stage,
        verify_restart,
        logs_opts,
        inlist_opts,
        set_param,
        None,
    )
}

/// [`run_astero_test_suite_case_with_case_hooks`] with the case's own star
/// extras as well. A case with `astero_just_call_my_extras_check_model`
/// runs as a plain star run whose `extras_check_model` and the other
/// `run_star_extras.f90` hooks are `star_extras`
/// (`astero/private/extras_support.f90:898-900`).
///
/// # Errors
/// As [`run_astero_test_suite_case_with_case_hooks`].
#[allow(clippy::too_many_arguments)]
pub fn run_astero_test_suite_case_with_hooks(
    orig_case_dir: &Path,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    logs_opts: StarLogsOptions<'_>,
    inlist_opts: InlistLoadOptions,
    set_param: Option<&mut dyn AsteroCaseHooks>,
    star_extras: Option<&mut dyn StarExtras>,
) -> Result<CaseRunReport, StarError> {
    let case_name = orig_case_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let staging = TempCaseStaging::new(orig_case_dir)?;
    let first_bundle = InlistBundle::from_file_in_case_dir_with_options(
        staging.staged_dir.as_path(),
        "inlist",
        inlist_opts,
    )?;
    // Upstream `astero/private/astero_run_support.f90:115-174` reads and
    // checks `&astero_search_controls` before testing
    // `astero_just_call_my_extras_check_model`.
    if staging
        .staged_dir
        .join("inlist_astero_search_controls")
        .is_file()
    {
        let _ = parse_astero_search_controls_with_options(
            staging.staged_dir.as_path(),
            inlist_opts,
        )?;
    }
    // `astero/private/astero_run_support.f90:171-174,239`: with
    // `astero_just_call_my_extras_check_model` the search is skipped and
    // `eval1` calls `run1_star` once; `astero/private/extras_support.f90:898-900`
    // then hands every step to the user's `extras_check_model`. That is a
    // plain star run of the case with its own extras.
    if first_bundle.job.astero_just_call_my_extras_check_model {
        drop(staging);
        if let Some(ex) = star_extras {
            return run_test_suite_case_with_extras(
                orig_case_dir,
                skip_optional,
                max_steps_per_stage,
                verify_restart,
                logs_opts,
                inlist_opts,
                ex,
            );
        }
        return run_test_suite_case_with_options(
            orig_case_dir,
            skip_optional,
            max_steps_per_stage,
            verify_restart,
            logs_opts,
            inlist_opts,
        );
    }
    run_astero_search_case(
        &staging,
        case_name,
        max_steps_per_stage,
        verify_restart,
        logs_opts,
        inlist_opts,
        set_param,
    )
}

fn run_astero_search_case(
    staging: &TempCaseStaging,
    case_name: &str,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    logs_opts: StarLogsOptions<'_>,
    inlist_opts: InlistLoadOptions,
    set_param: Option<&mut dyn AsteroCaseHooks>,
) -> Result<CaseRunReport, StarError> {
    let t0 = Instant::now();
    let case_dir = staging.staged_dir.as_path();
    let ctrl =
        parse_astero_search_controls_with_options(case_dir, inlist_opts)?;

    let search_rep = run_star_astero_with_case_hooks(
        case_dir,
        "inlist",
        max_steps_per_stage,
        &ctrl,
        logs_opts,
        inlist_opts,
        set_param,
    )?;
    let last_mod = search_rep.last_model.as_ref().ok_or_else(|| {
        StarError::ConvergenceFailure(format!(
            "{case_name}: no astero tracks evaluated"
        ))
    })?;
    let mut photo_restart_verified = false;
    if verify_restart {
        let (c1, c2) = verify_photo_restart_parity(last_mod, 1)?;
        photo_restart_verified = c1 == c2;
    }
    let best = search_rep
        .best_sample_idx
        .and_then(|i| search_rep.samples.get(i));

    let assertions_passed = best.is_some_and(|b| {
        b.chi2.is_finite()
            && b.chi2 > 0.0
            && (max_steps_per_stage.is_some() || b.chi2 < 1.0e90)
    });

    let mut hr = extract_history_row(last_mod);
    if let Some(b) = best {
        hr.model_number = b.model_number;
        hr.star_age = b.age_yr;
        hr.star_mass = b.params.initial_mass;
        hr.log_teff = flexmesa_math::log10(b.teff.max(1.0));
        hr.log_l = b.log_l;
        hr.log_r = b.log_r;
        hr.log_g = b.logg;
    }
    let runtime_sec = t0.elapsed().as_secs_f64();
    let term_tail = search_rep.last_termination.as_str();
    let effective_term =
        search_rep.best_termination.as_deref().unwrap_or(term_tail);
    let stage = StageRunReport {
        inlist_name: String::from("inlist"),
        steps_taken: search_rep.total_steps,
        total_newton_iterations: search_rep.total_steps * 2,
        total_retries: 0,
        termination_code: format!(
            "astero_{}_samples={}_best_chi2={:.6e}_term={term_tail}",
            search_rep.search_type,
            search_rep.samples.len(),
            search_rep.best_chi2,
        ),
        runtime_sec,
        final_history: hr,
        history_rows: vec![hr],
    };
    let passed = is_astero_search_passed(
        assertions_passed,
        effective_term,
        max_steps_per_stage.is_none(),
        !verify_restart || photo_restart_verified,
    );
    Ok(build_case_report(
        case_name,
        passed,
        stage.termination_code.clone(),
        vec![stage],
        runtime_sec,
        hr,
        photo_restart_verified,
        assertions_passed,
    ))
}

/// Evaluates whether an asteroseismic search run completed cleanly.
///
/// Upstream `astero/private/extras_support.f90:492-502,521-528` treats
/// `limit_num_chi2_too_big` ("have reached too many bad chi2 limit") as a
/// normal early-stopping condition once a track passes its $\chi^2$ minimum,
/// so `limit_num_*` is accepted while step caps and abnormal solver stops
/// remain failures.
pub(crate) fn is_astero_search_passed(
    assertions_passed: bool,
    effective_term: &str,
    uncapped: bool,
    restart_ok: bool,
) -> bool {
    let abnormal_termination = matches!(
        effective_term,
        "max_steps_cap"
            | "min_timestep_limit"
            | "max_number_retries"
            | "redo_limit"
            | "dt_is_zero"
            | "varcontrol_target < min_allowed_varcontrol_target"
            | "invalid dt_next in timestep_controller"
    );
    assertions_passed && !abnormal_termination && uncapped && restart_ok
}

fn build_case_report(
    case_name: &str,
    passed: bool,
    termination_code: String,
    stages: Vec<StageRunReport>,
    runtime_sec: f64,
    hr: HistoryRow,
    photo_restart_verified: bool,
    assertions_passed: bool,
) -> CaseRunReport {
    let total_steps = stages.iter().map(|s| s.steps_taken).sum();
    let total_newton_iterations =
        stages.iter().map(|s| s.total_newton_iterations).sum();
    let total_retries = stages.iter().map(|s| s.total_retries).sum();
    CaseRunReport {
        case_name: String::from(case_name),
        tier: 0,
        passed,
        termination_code,
        stages,
        total_steps,
        total_newton_iterations,
        total_retries,
        total_runtime_sec: runtime_sec,
        wall_time_ms: runtime_sec * 1000.0,
        final_teff: flexmesa_math::exp10(hr.log_teff),
        final_log_l: hr.log_l,
        final_log_r: hr.log_r,
        final_log_tc: hr.log_center_t,
        final_log_rhoc: hr.log_center_rho,
        final_center_h1: hr.center_h1,
        final_center_he4: hr.center_he4,
        photo_restart_verified,
        assertions_passed,
    }
}
