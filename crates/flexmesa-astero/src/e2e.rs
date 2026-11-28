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
    let runtime_sec = t0.elapsed().as_secs_f64();
    Ok(finalize_astero_case_report(
        case_name,
        &search_rep,
        last_mod,
        runtime_sec,
        max_steps_per_stage,
        verify_restart,
        photo_restart_verified,
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

fn finalize_astero_case_report(
    case_name: &str,
    search_rep: &crate::search_driver::AsteroSearchReport,
    last_mod: &flexmesa_star::model::StarModel,
    runtime_sec: f64,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    photo_restart_verified: bool,
) -> CaseRunReport {
    let best = search_rep
        .best_sample_idx
        .and_then(|i| search_rep.samples.get(i));

    let assertions_passed = best.is_some_and(|b| {
        b.chi2.is_finite()
            && b.chi2 > 0.0
            && (max_steps_per_stage.is_some() || b.chi2 < 1.0e90)
    });

    let mut hr = extract_history_row(last_mod);
    let end_model_number = hr.model_number;
    if let Some(b) = best {
        hr.model_number = b.model_number;
        hr.star_age = b.age_yr;
        hr.star_mass = b.params.initial_mass;
        hr.log_teff = flexmesa_math::log10(b.teff.max(1.0));
        hr.log_l = b.log_l;
        hr.log_r = b.log_r;
        hr.log_g = b.logg;
    }
    let term_tail = search_rep.last_termination.as_str();
    let effective_term =
        search_rep.best_termination.as_deref().unwrap_or(term_tail);
    let case_termination = format!(
        "astero_{}_samples={}_best_chi2={:.6e}_term={term_tail}",
        search_rep.search_type,
        search_rep.samples.len(),
        search_rep.best_chi2,
    );
    let stages = if search_rep.track_stages.is_empty() {
        let mut stage_hr = hr;
        stage_hr.model_number = end_model_number;
        vec![StageRunReport {
            inlist_name: String::from("inlist"),
            steps_taken: search_rep.total_steps,
            total_newton_iterations: search_rep.total_steps * 2,
            total_retries: 0,
            termination_code: effective_term.to_string(),
            runtime_sec,
            final_history: stage_hr,
            history_rows: vec![stage_hr],
        }]
    } else {
        search_rep.track_stages.clone()
    };
    let passed = is_astero_search_passed(
        assertions_passed,
        effective_term,
        max_steps_per_stage.is_none(),
        !verify_restart || photo_restart_verified,
    );
    build_case_report(
        case_name,
        passed,
        case_termination,
        stages,
        runtime_sec,
        hr,
        photo_restart_verified,
        assertions_passed,
    )
}

#[cfg(test)]
mod tests {
    use super::{finalize_astero_case_report, is_astero_search_passed};
    use crate::inlist_astero::{AsteroSearchControls, MAX_CONSTRAINTS};
    use crate::sample_eval::AsteroSampleResult;
    use crate::search_driver::AsteroSearchReport;
    use flexmesa_star::controls::{StarControls, StarJobControls};
    use flexmesa_star::history::extract_history_row;
    use flexmesa_star::inlist_runner::StageRunReport;
    use flexmesa_star::model::StarModel;

    fn sample_with_model(
        sample_number: usize,
        model_number: usize,
        chi2: f64,
    ) -> AsteroSampleResult {
        AsteroSampleResult {
            sample_number,
            params: AsteroSearchControls::default().first_params(),
            model_number,
            age_yr: 4.57e9,
            teff: 5777.0,
            logg: 4.44,
            log_l: 0.0,
            log_r: 0.0,
            constraint_values: [f64::NAN; MAX_CONSTRAINTS],
            delta_nu_uhz: 135.0,
            nu_max_uhz: 3090.0,
            a_div_r: 0.0,
            correction_r: 0.0,
            chi2,
            chi2_seismo: chi2,
            chi2_spectro: 0.0,
            chi2_radial: chi2,
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
            model_freqs_alt_up: [
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            model_freqs_corr_alt_up: [
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            model_inertias_alt_up: [
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            model_orders_alt_up: [
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            model_freqs_alt_down: [
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            model_freqs_corr_alt_down: [
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            model_inertias_alt_down: [
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            model_orders_alt_down: [
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
        }
    }

    #[test]
    fn test_is_astero_search_passed_accepts_xtra1_and_blocks_abnormal() {
        assert!(is_astero_search_passed(true, "xtra1", true, true));
        assert!(is_astero_search_passed(
            true,
            "chi2_search_limit2",
            true,
            true
        ));
        assert!(!is_astero_search_passed(true, "max_steps_cap", true, true));
        assert!(!is_astero_search_passed(true, "xtra1", false, true));
        assert!(!is_astero_search_passed(true, "xtra1", true, false));
    }

    #[test]
    fn test_finalize_astero_case_report_preserves_stage_model_numbers() {
        let mut last_mod = StarModel::new(
            4,
            vec![1],
            StarJobControls::default(),
            StarControls::default(),
        );
        last_mod.model_number = 737;
        let mut stage_hr = extract_history_row(&last_mod);
        stage_hr.model_number = 737;

        let track_stage = StageRunReport {
            inlist_name: String::from("inlist"),
            steps_taken: 737,
            total_newton_iterations: 1500,
            total_retries: 0,
            termination_code: String::from("xtra1"),
            runtime_sec: 1.5,
            final_history: stage_hr,
            history_rows: vec![stage_hr],
        };

        let rep = AsteroSearchReport {
            search_type: String::from("use_first_values"),
            samples: vec![sample_with_model(1, 734, 2.800_488)],
            best_sample_idx: Some(0),
            best_chi2: 2.800_488,
            total_steps: 737,
            track_stages: vec![track_stage],
            best_termination: Some(String::from("xtra1")),
            last_termination: String::from("xtra1"),
            last_model: None,
        };

        let case_rep = finalize_astero_case_report(
            "example_astero",
            &rep,
            &last_mod,
            1.5,
            None,
            false,
            false,
        );
        assert!(case_rep.passed);
        assert_eq!(case_rep.total_steps, 737);
        assert_eq!(case_rep.stages.len(), 1);
        assert_eq!(case_rep.stages[0].final_history.model_number, 737);
        assert_eq!(case_rep.stages[0].termination_code, "xtra1");
        assert!(case_rep.termination_code.ends_with("_term=xtra1"));
    }

    #[test]
    fn test_finalize_astero_case_report_preserves_multitrack_stages() {
        let mut last_mod = StarModel::new(
            4,
            vec![1],
            StarJobControls::default(),
            StarControls::default(),
        );
        last_mod.model_number = 508;
        let mut hr0 = extract_history_row(&last_mod);
        hr0.model_number = 508;
        let mut hr1 = extract_history_row(&last_mod);
        hr1.model_number = 508;

        let stages = vec![
            StageRunReport {
                inlist_name: String::from("inlist"),
                steps_taken: 508,
                total_newton_iterations: 1000,
                total_retries: 1,
                termination_code: String::from("chi2_search_limit2"),
                runtime_sec: 1.0,
                final_history: hr0,
                history_rows: vec![hr0],
            },
            StageRunReport {
                inlist_name: String::from("inlist"),
                steps_taken: 508,
                total_newton_iterations: 1020,
                total_retries: 2,
                termination_code: String::from("chi2_search_limit2"),
                runtime_sec: 1.0,
                final_history: hr1,
                history_rows: vec![hr1],
            },
        ];

        let rep = AsteroSearchReport {
            search_type: String::from("from_file"),
            samples: vec![
                sample_with_model(1, 490, 3.0),
                sample_with_model(2, 495, 2.5),
            ],
            best_sample_idx: Some(1),
            best_chi2: 2.5,
            total_steps: 1016,
            track_stages: stages,
            best_termination: Some(String::from("chi2_search_limit2")),
            last_termination: String::from("chi2_search_limit2"),
            last_model: None,
        };

        let case_rep = finalize_astero_case_report(
            "fast_from_file",
            &rep,
            &last_mod,
            2.0,
            None,
            false,
            false,
        );
        assert!(case_rep.passed);
        assert_eq!(case_rep.stages.len(), 2);
        assert_eq!(case_rep.stages[0].final_history.model_number, 508);
        assert_eq!(case_rep.stages[1].final_history.model_number, 508);
        assert_eq!(case_rep.total_steps, 1016);
        assert_eq!(case_rep.total_newton_iterations, 2020);
        assert_eq!(case_rep.total_retries, 3);
    }
}
