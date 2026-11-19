// SPDX-License-Identifier: LGPL-3.0-or-later
//! End-to-end runner and RAII temporary staging (`TempCaseStaging`) for
//! coupled `astero` pulsation and optimization runs (`#148`).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use flexmesa_star::controls::StarError;
use flexmesa_star::history::{extract_history_row, HistoryRow};
use flexmesa_star::inlist::{InlistBundle, InlistLoadOptions};
use flexmesa_star::inlist_runner::{
    run_inlist_stage_with_logs, run_test_suite_case_with_options,
    verify_photo_restart_parity, CaseRunReport, DefaultStarExtras,
    StageRunReport, StarLogsOptions,
};
use flexmesa_star::pulse_bvp::{InertiaNormKind, OuterBoundKind};
use flexmesa_star::pulse_modes::{
    scan_star_bvp_eigenmodes, PulseModeScanOptions,
};
use flexmesa_star::rn_parser::{parse_rn_stages, RnStage};
use flexmesa_star::star_job_init::init_or_update_model_from_bundle;

use crate::inlist_astero::parse_astero_search_controls_with_options;
use crate::sample_eval::{eval_model_astero_sample, parse_gyre_outer_bound};
use crate::search_driver::run_star_astero_with_options;

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
            if let Some(first_inlist) =
                parse_rn_stages(&staged_dir, true)?.into_iter().find_map(
                    |st| match st {
                        RnStage::DoOne { inlist, .. } => Some(inlist),
                        RnStage::CopyCheckpoint { .. } => None,
                    },
                )
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
    let case_name = orig_case_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let has_pulse_inlist = orig_case_dir.join("inlist_astero_adipls").exists()
        || orig_case_dir.join("inlist_astero_gyre").exists();
    let has_search_inlist = orig_case_dir
        .join("inlist_astero_search_controls")
        .is_file();
    let has_astero_targets = if has_search_inlist {
        let c = parse_astero_search_controls_with_options(
            orig_case_dir,
            inlist_opts,
        )?;
        c.nl.iter().sum::<usize>() > 0
            || c.include_age_in_chi2_spectro
            || [
                c.teff,
                c.logg,
                c.log_l,
                c.feh,
                c.log_r,
                c.radius,
                c.m_h,
                c.surface_z_div_x,
                c.surface_he,
                c.rcz,
            ]
            .iter()
            .any(|s| s.include && s.sigma > 0.0)
    } else {
        false
    };
    if !has_pulse_inlist && !has_astero_targets {
        return run_test_suite_case_with_options(
            orig_case_dir,
            skip_optional,
            max_steps_per_stage,
            verify_restart,
            logs_opts,
            inlist_opts,
        );
    }
    if has_pulse_inlist {
        return run_pulse_tams_case(
            orig_case_dir,
            case_name,
            max_steps_per_stage,
            verify_restart,
            logs_opts,
            inlist_opts,
        );
    }
    let staging = TempCaseStaging::new(orig_case_dir)?;
    let first_bundle = InlistBundle::from_file_in_case_dir_with_options(
        staging.staged_dir.as_path(),
        "inlist",
        inlist_opts,
    )?;
    if first_bundle.controls.max_model_number == Some(0) {
        let stage_inlist = if staging.staged_dir.join("rn").is_file() {
            parse_rn_stages(staging.staged_dir.as_path(), true)?
                .into_iter()
                .find_map(|st| match st {
                    RnStage::DoOne { inlist, .. } => Some(inlist),
                    RnStage::CopyCheckpoint { .. } => None,
                })
                .unwrap_or_else(|| String::from("inlist"))
        } else {
            String::from("inlist")
        };
        return run_surface_effects_case(
            &staging,
            case_name,
            &stage_inlist,
            &first_bundle,
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
    )
}

fn run_pulse_tams_case(
    orig_case_dir: &Path,
    case_name: &str,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    logs_opts: StarLogsOptions<'_>,
    inlist_opts: InlistLoadOptions,
) -> Result<CaseRunReport, StarError> {
    let t0 = Instant::now();
    let staging = TempCaseStaging::new(orig_case_dir)?;
    let case_dir = staging.staged_dir.as_path();
    let bundle = InlistBundle::from_file_in_case_dir_with_options(
        case_dir,
        "inlist",
        inlist_opts,
    )?;

    let mut extras = DefaultStarExtras;
    let mut model =
        init_or_update_model_from_bundle(&bundle, None, &mut extras)?;
    let mut photo_restart_verified = false;
    if verify_restart {
        let (c1, c2) = verify_photo_restart_parity(&model, 1)?;
        photo_restart_verified = c1 == c2;
    }

    let mut lw = logs_opts.build_writer(&model, case_dir, None)?;
    let rep = run_inlist_stage_with_logs(
        &mut model,
        &bundle,
        &mut extras,
        max_steps_per_stage,
        lw.as_mut(),
    )?;

    let scan_opts = if case_dir.join("inlist_astero_gyre").is_file() {
        PulseModeScanOptions {
            l: 0,
            freq_min_uhz: 100.0,
            freq_max_uhz: 1000.0,
            n_freq: 200,
            add_center_point: true,
            keep_surface_point: false,
            add_atmosphere: false,
            outer_bc: parse_gyre_outer_bound(case_dir, "gyre.in")?,
            inertia_norm: InertiaNormKind::Both,
        }
    } else {
        PulseModeScanOptions {
            l: 0,
            freq_min_uhz: 50.0,
            freq_max_uhz: 1000.0,
            n_freq: 200,
            add_center_point: true,
            keep_surface_point: false,
            add_atmosphere: true,
            outer_bc: OuterBoundKind::AdiplsIsothermal,
            inertia_norm: InertiaNormKind::Both,
        }
    };
    let modes = scan_star_bvp_eigenmodes(&model, &scan_opts);
    let h1_c = model.xh_mass.last().copied().unwrap_or(0.0);
    let target_uhz = model
        .controls
        .x_ctrl
        .first()
        .copied()
        .filter(|&f| f > 0.0)
        .unwrap_or(375.0);
    let assertions_passed = if max_steps_per_stage.is_none() {
        h1_c <= 0.0015
            && modes.iter().any(|m| {
                m.n_p == 4
                    && ((m.freq_uhz - target_uhz) / target_uhz).abs() < 0.03
            })
    } else {
        !modes.is_empty()
            && modes
                .iter()
                .all(|m| m.freq_uhz.is_finite() && m.freq_uhz > 0.0)
    };

    let hr = rep.final_history;
    let runtime_sec = t0.elapsed().as_secs_f64();
    let passed = assertions_passed
        && rep.termination_code != "max_steps_cap"
        && (!verify_restart || photo_restart_verified);
    Ok(build_case_report(
        case_name,
        passed,
        rep.termination_code.clone(),
        vec![rep],
        runtime_sec,
        hr,
        photo_restart_verified,
        assertions_passed,
    ))
}

fn run_surface_effects_case(
    staging: &TempCaseStaging,
    case_name: &str,
    stage_inlist: &str,
    bundle: &InlistBundle,
    verify_restart: bool,
    logs_opts: StarLogsOptions<'_>,
    inlist_opts: InlistLoadOptions,
) -> Result<CaseRunReport, StarError> {
    let t0 = Instant::now();
    let case_dir = staging.staged_dir.as_path();
    let ctrl =
        parse_astero_search_controls_with_options(case_dir, inlist_opts)?;
    let mut extras = DefaultStarExtras;
    let model = init_or_update_model_from_bundle(bundle, None, &mut extras)?;
    let mut photo_restart_verified = false;
    if verify_restart {
        let (c1, c2) = verify_photo_restart_parity(&model, 1)?;
        photo_restart_verified = c1 == c2;
    }

    if let Some(mut lw) = logs_opts.build_writer(&model, case_dir, None)? {
        lw.record_step(&model, true, true)?;
    }
    let sample = eval_model_astero_sample(
        &model,
        &ctrl,
        ctrl.first_params(),
        1,
        case_dir,
    )?;
    write_surface_effects_freqs_dat(
        &case_dir.join("freqs.dat"),
        &ctrl,
        &sample,
    )?;
    if let Some(out_root) = logs_opts.output_root {
        fs::create_dir_all(out_root).map_err(|e| {
            StarError::IoError(format!(
                "failed creating output root '{}': {e}",
                out_root.display()
            ))
        })?;
        write_surface_effects_freqs_dat(
            &out_root.join("freqs.dat"),
            &ctrl,
            &sample,
        )?;
    }

    let total_modes: usize = sample.model_freqs.iter().map(Vec::len).sum();
    let assertions_passed = total_modes > 0
        && sample
            .model_freqs
            .iter()
            .flatten()
            .all(|&f| f.is_finite() && f > 0.0);
    let hr = extract_history_row(&model);
    let runtime_sec = t0.elapsed().as_secs_f64();
    let stage = StageRunReport {
        inlist_name: String::from(stage_inlist),
        steps_taken: 0,
        total_newton_iterations: 0,
        total_retries: 0,
        termination_code: String::from("max_model_number"),
        runtime_sec,
        final_history: hr,
        history_rows: vec![hr],
    };
    let passed =
        assertions_passed && (!verify_restart || photo_restart_verified);
    Ok(build_case_report(
        case_name,
        passed,
        String::from("max_model_number"),
        vec![stage],
        runtime_sec,
        hr,
        photo_restart_verified,
        assertions_passed,
    ))
}

fn write_surface_effects_freqs_dat(
    path: &Path,
    ctrl: &crate::inlist_astero::AsteroSearchControls,
    sample: &crate::sample_eval::AsteroSampleResult,
) -> Result<(), StarError> {
    let mut buf = String::new();
    for l in 0..4 {
        for i in 0..sample.model_freqs[l].len() {
            let _ = writeln!(
                buf,
                "{:4} {:4} {:26.16e} {:26.16e} {:26.16e} {:26.16e}",
                l,
                sample.model_orders[l][i],
                sample.model_inertias[l][i],
                ctrl.freq_target[l][i],
                ctrl.freq_sigma[l][i],
                sample.model_freqs[l][i]
            );
        }
    }
    fs::write(path, buf).map_err(|e| {
        StarError::IoError(format!("failed writing '{}': {e}", path.display()))
    })
}

fn run_astero_search_case(
    staging: &TempCaseStaging,
    case_name: &str,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    logs_opts: StarLogsOptions<'_>,
    inlist_opts: InlistLoadOptions,
) -> Result<CaseRunReport, StarError> {
    let t0 = Instant::now();
    let case_dir = staging.staged_dir.as_path();
    let ctrl =
        parse_astero_search_controls_with_options(case_dir, inlist_opts)?;

    let search_rep = run_star_astero_with_options(
        case_dir,
        "inlist",
        max_steps_per_stage,
        &ctrl,
        logs_opts,
        inlist_opts,
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
    let passed = assertions_passed
        && term_tail != "max_steps_cap"
        && !term_tail.starts_with("limit_num_")
        && max_steps_per_stage.is_none()
        && (!verify_restart || photo_restart_verified);
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
