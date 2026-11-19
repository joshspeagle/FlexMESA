// SPDX-License-Identifier: LGPL-3.0-or-later
//! Coupled binary evolution case runner for `FlexMESA` (`#148`).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Instant;

use flexmesa_const::{MSUN, RSUN, SECDAY, SECYER};
use flexmesa_star::{
    DefaultStarExtras, InlistBundle, InlistLoadOptions, StarError, StarModel,
    ZamsCatalog, init_or_update_model_from_bundle,
    load_zams_model_from_catalog, relax_composition_with_henyey,
};

use crate::binary_history::BinaryLogsWriter;
use crate::extras::{BinaryExtras, BinaryExtrasAction, DefaultBinaryExtras};
pub use crate::inlist::BinaryInlistControls;
use crate::system::{BinaryCompanion, BinarySystem};

static ZAMS_CATALOG_CACHE: OnceLock<Mutex<BTreeMap<String, Arc<ZamsCatalog>>>> =
    OnceLock::new();

fn resolve_zams_catalog(
    case_dir: &Path,
    zams_filename: &str,
) -> Result<Option<Arc<ZamsCatalog>>, StarError> {
    let fname = if zams_filename.trim().is_empty() {
        "zams_z2m2_y28.data"
    } else {
        zams_filename.trim()
    };
    let candidate = Path::new(fname);
    let resolved_path = if candidate.is_absolute() {
        candidate.is_file().then(|| candidate.to_path_buf())
    } else {
        let local = case_dir.join(candidate);
        if local.is_file() {
            Some(local)
        } else {
            case_dir.ancestors().find_map(|anc| {
                let p = anc
                    .join("data")
                    .join("star_data")
                    .join("zams_models")
                    .join(fname);
                p.is_file().then_some(p)
            })
        }
    };
    let Some(zams_path) = resolved_path else {
        return Ok(None);
    };
    let cache_key = zams_path.to_string_lossy().into_owned();
    let cache = ZAMS_CATALOG_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Some(existing) = cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&cache_key)
        .cloned()
    {
        return Ok(Some(existing));
    }
    let text = fs::read_to_string(&zams_path).map_err(|e| {
        StarError::IoError(format!(
            "failed reading ZAMS catalog '{}': {e}",
            zams_path.display()
        ))
    })?;
    let cat = ZamsCatalog::parse_ascii(&text).map_err(|e| {
        StarError::IoError(format!(
            "failed parsing ZAMS catalog '{}': {e}",
            zams_path.display()
        ))
    })?;
    let arc_cat = Arc::new(cat);
    cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(cache_key, Arc::clone(&arc_cat));
    Ok(Some(arc_cat))
}

fn init_binary_star_component(
    case_dir: &Path,
    inlist_name: &str,
    mass_msun: f64,
    options: InlistLoadOptions,
) -> Result<StarModel, StarError> {
    let mut bundle = InlistBundle::from_file_in_case_dir_with_options(
        case_dir,
        inlist_name,
        options,
    )?;
    bundle.controls.initial_mass = mass_msun;
    let needs_henyey_yz_relax = bundle.job.relax_z || bundle.job.relax_y;
    let mut init_bundle = bundle.clone();
    if needs_henyey_yz_relax {
        init_bundle.job.relax_z = false;
        init_bundle.job.relax_y = false;
    }
    let zams_fname = bundle.controls.zams_filename.trim();
    let custom_zams = !zams_fname.is_empty()
        && !zams_fname.eq_ignore_ascii_case("zams_z2m2_y28.data");
    let uses_zams_catalog = !bundle.job.load_saved_model
        && !bundle.job.create_pre_ms_model
        && !bundle.job.create_initial_model
        && !bundle.job.use_other_build_initial_model
        && (custom_zams
            || ((bundle.job.relax_z
                || (bundle.controls.initial_z - 0.02).abs() <= 1.0e-4)
                && (bundle.job.relax_y
                    || bundle.controls.initial_y < 0.0
                    || (bundle.controls.initial_y - 0.28).abs() <= 1.0e-4)));
    let seed_zams = if uses_zams_catalog {
        if let Some(cat) = resolve_zams_catalog(case_dir, zams_fname)? {
            Some(load_zams_model_from_catalog(
                &cat,
                init_bundle.controls.clone(),
                init_bundle.job.clone(),
            )?)
        } else if custom_zams {
            return Err(StarError::IoError(format!(
                "configured zams_filename '{zams_fname}' not found relative to '{}' or data/star_data/zams_models",
                case_dir.display()
            )));
        } else {
            None
        }
    } else {
        None
    };
    let mut star_extras = DefaultStarExtras;
    let mut model = init_or_update_model_from_bundle(
        &init_bundle,
        seed_zams,
        &mut star_extras,
    )?;
    model.job = bundle.job.clone();
    if needs_henyey_yz_relax {
        relax_composition_with_henyey(&mut model, &bundle.job)?;
    }
    Ok(model)
}

/// Telemetry report for a coupled binary evolution run.
#[derive(Debug, Clone)]
pub struct BinaryCaseRunReport {
    /// Directory basename of the executed case.
    pub case_name: String,
    /// Whether physical evolution terminated before any external `max_steps_cap`.
    pub passed: bool,
    /// Termination code string (e.g., `"max_age"`, `"accretor_overflow_terminate"`, `"MaxStepsCap"`).
    pub termination_code: String,
    /// Number of coupled binary steps completed.
    pub total_steps: usize,
    /// Final star 1 mass $M_1$ ($M_\odot$).
    pub final_m1_msun: f64,
    /// Final star 2 mass $M_2$ ($M_\odot$).
    pub final_m2_msun: f64,
    /// Final orbital separation $a$ ($R_\odot$).
    pub final_separation_rsun: f64,
    /// Final orbital period $P$ (days).
    pub final_period_days: f64,
    /// Final mass-transfer rate $|\dot{M}_{\mathrm{tr}}|$ ($M_\odot\,\mathrm{yr}^{-1}$).
    pub final_abs_mdot_msun_yr: f64,
    /// Wall-clock runtime in milliseconds (`ms`).
    pub wall_time_ms: f64,
    /// Cumulative outer binary retries (`outer_retry > 0`).
    pub binary_retries: usize,
    /// Cumulative coupled implicit mass-transfer trials.
    pub implicit_trials: usize,
    /// Cumulative solver `CutDt` restarts.
    pub dt_cuts: usize,
    /// Cumulative hard-limit / `delta_mdot_hard_limit` retries.
    pub hard_limit_retries: usize,
    /// Cumulative outer retries accumulated by the active donor star.
    pub donor_retries: usize,
    /// Cumulative outer retries accumulated by the accretor star (`0` for point mass).
    pub accretor_retries: usize,
}

/// Configuration for incremental MESA `binary_history.data` and component
/// `LOGS1/`/`LOGS2/` output during `binary/test_suite` execution (`#163`).
#[derive(Debug, Clone, Copy, Default)]
pub struct BinaryLogsOptions<'a> {
    /// Optional root directory under which `<log_directory>/<history_name>` and
    /// component `LOGS1/` / `LOGS2/` files are written.
    pub output_root: Option<&'a Path>,
    /// Optional explicit output file path for `binary_history.data`.
    pub explicit_binary_history_file: Option<&'a Path>,
    /// Optional override for `history_interval` (`Some(1)` writes every step).
    pub history_interval_override: Option<usize>,
    /// Whether to also write component stellar `LOGS1/` / `LOGS2/` files when
    /// `output_root` is provided (`true` by default when targeting a directory).
    pub write_star_logs: bool,
}

/// Executes a single `binary/test_suite/<case>` directory through genuine
/// coupled [`BinarySystem::try_step_with_extras`] + [`StarModel`] Henyey
/// evolution and verifies its upstream `run_binary_extras.f90` /
/// `run_star_extras.f90` physical and analytical assertions.
///
/// When `max_steps_cap` is `None` (the default), evolves to physical
/// termination. When `Some(cap)` is supplied, evolves up to `cap` coupled
/// steps while enforcing every per-step analytical oracle bound.
pub fn run_binary_test_suite_case(
    case_dir: &Path,
    max_steps_cap: Option<usize>,
) -> Result<BinaryCaseRunReport, StarError> {
    run_binary_test_suite_case_with_logs(
        case_dir,
        max_steps_cap,
        BinaryLogsOptions::default(),
    )
}

/// Executes a single `binary/test_suite/<case>` directory while optionally
/// streaming incremental MESA-format `binary_history.data` (and component
/// `LOGS1/` / `LOGS2/` files) according to `logs_opts` (`#163`).
pub fn run_binary_test_suite_case_with_logs(
    case_dir: &Path,
    max_steps_cap: Option<usize>,
    logs_opts: BinaryLogsOptions<'_>,
) -> Result<BinaryCaseRunReport, StarError> {
    run_binary_test_suite_case_with_options(
        case_dir,
        max_steps_cap,
        logs_opts,
        InlistLoadOptions::default(),
    )
}

fn contextualize_binary_step_error(
    case_name: &str,
    step_num: usize,
    err: StarError,
) -> StarError {
    match err {
        StarError::InvalidInput(msg) => StarError::InvalidInput(format!(
            "[{case_name} step {step_num}] {msg}"
        )),
        StarError::IoError(msg) => {
            StarError::IoError(format!("[{case_name} step {step_num}] {msg}"))
        }
        StarError::MicrophysicsError(msg) => StarError::MicrophysicsError(
            format!("[{case_name} step {step_num}] {msg}"),
        ),
        StarError::Terminated {
            model_number,
            reason,
        } => StarError::Terminated {
            model_number,
            reason: format!("[{case_name} step {step_num}] {reason}"),
        },
        StarError::StepFailure(mut info) => {
            info.detail =
                format!("[{case_name} step {step_num}] {}", info.detail);
            StarError::StepFailure(info)
        }
        StarError::ConvergenceFailure(msg) => StarError::ConvergenceFailure(
            format!("[{case_name} step {step_num}] {msg}"),
        ),
    }
}

/// Executes a single coupled binary directory with explicit
/// [`InlistLoadOptions`] and default no-op [`DefaultBinaryExtras`].
pub fn run_binary_test_suite_case_with_options(
    case_dir: &Path,
    max_steps_cap: Option<usize>,
    logs_opts: BinaryLogsOptions<'_>,
    inlist_opts: InlistLoadOptions,
) -> Result<BinaryCaseRunReport, StarError> {
    let mut extras = DefaultBinaryExtras;
    run_binary_test_suite_case_with_extras(
        case_dir,
        max_steps_cap,
        logs_opts,
        inlist_opts,
        &mut extras,
    )
}

/// Executes a single coupled binary directory with explicit
/// [`InlistLoadOptions`] and caller-supplied [`BinaryExtras`] hooks (`#148`).
pub fn run_binary_test_suite_case_with_extras(
    case_dir: &Path,
    max_steps_cap: Option<usize>,
    logs_opts: BinaryLogsOptions<'_>,
    inlist_opts: InlistLoadOptions,
    extras: &mut dyn BinaryExtras,
) -> Result<BinaryCaseRunReport, StarError> {
    extras.set_staged_case_dir(case_dir);
    let t0 = Instant::now();
    let case_name = case_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    let bctrl = BinaryInlistControls::from_case_dir_with_options(
        case_dir,
        inlist_opts,
    )?;

    let donor = init_binary_star_component(
        case_dir,
        &bctrl.inlist_names[0],
        bctrl.m1_msun,
        inlist_opts,
    )?;
    let mut bin = if bctrl.evolve_both_stars {
        let accretor = init_binary_star_component(
            case_dir,
            &bctrl.inlist_names[1],
            bctrl.m2_msun,
            inlist_opts,
        )?;
        BinarySystem::from_controls_two_stars(donor, accretor, bctrl.clone())?
    } else {
        BinarySystem::from_controls_point_mass(donor, bctrl)?
    };

    // `run_binary_support.f90:345-350`: `extras_binary_startup` followed by
    // `do_binary_job_controls_after`.
    match extras.startup(&mut bin, false) {
        BinaryExtrasAction::KeepGoing
        | BinaryExtrasAction::Redo
        | BinaryExtrasAction::Retry => {}
        BinaryExtrasAction::Terminate(reason) => {
            bin.last_termination_reason = Some(reason);
        }
        BinaryExtrasAction::Error(msg) => {
            return Err(StarError::InvalidInput(format!(
                "[{case_name} startup] {msg}"
            )));
        }
    }
    bin.apply_binary_job_controls_after(false)?;

    let mut logs_writer = if logs_opts.output_root.is_some()
        || logs_opts.explicit_binary_history_file.is_some()
    {
        let root = logs_opts
            .output_root
            .or_else(|| {
                logs_opts
                    .explicit_binary_history_file
                    .and_then(Path::parent)
                    .filter(|p| !p.as_os_str().is_empty())
            })
            .unwrap_or_else(|| Path::new("."));
        Some(BinaryLogsWriter::for_run(
            &bin,
            case_dir,
            root,
            logs_opts.explicit_binary_history_file,
            logs_opts.history_interval_override,
            logs_opts.write_star_logs,
        )?)
    } else {
        None
    };

    let uncapped = max_steps_cap.is_none();
    let step_limit = max_steps_cap.unwrap_or(100_000).max(1);

    let mut dt_next = match &bin.companion {
        BinaryCompanion::PointMass { .. } => bin.donor.dt_next,
        BinaryCompanion::Star(s2) => bin.donor.dt_next.min(s2.dt_next),
    };
    if !dt_next.is_finite() || dt_next <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "[{case_name}] non-positive initial binary dt_next ({dt_next})"
        )));
    }

    let mut term_code = String::from("MaxStepsCap");
    if let Some(ref startup_reason) = bin.last_termination_reason {
        term_code.clone_from(startup_reason);
    }
    while bin.last_termination_reason.is_none() && bin.model_number < step_limit
    {
        let step_num = bin.model_number + 1;
        let (suggested_dt, _) =
            bin.try_step_with_extras(dt_next, extras).map_err(|e| {
                contextualize_binary_step_error(&case_name, step_num, e)
            })?;
        dt_next = suggested_dt;
        let has_term = bin.last_termination_reason.is_some();
        let is_terminating = has_term || bin.model_number >= step_limit;
        if let Some(ref mut lw) = logs_writer {
            let is_first = bin.model_number == 1;
            let (bin_due, star_due) =
                lw.extra_history_due_flags(&bin, is_first, is_terminating);
            let extra_bin_cols = if bin_due {
                extras.extra_binary_history_columns(&bin)
            } else {
                Vec::new()
            };
            let extra_star_cols = [
                if star_due[0] {
                    extras.extra_star_history_columns(&bin, 1)
                } else {
                    Vec::new()
                },
                if star_due[1] {
                    extras.extra_star_history_columns(&bin, 2)
                } else {
                    Vec::new()
                },
            ];
            lw.record_step_with_extra(
                &bin,
                is_first,
                is_terminating,
                &extra_bin_cols,
                &extra_star_cols,
            )?;
        }
        if let Some(ref reason) = bin.last_termination_reason {
            term_code.clone_from(reason);
            break;
        }
    }
    extras.after_evolve(&mut bin)?;

    let [m1_final_g, m2_final_g] = bin.star_masses_g();
    let m1_msun = m1_final_g / MSUN;
    let m2_msun = m2_final_g / MSUN;
    let passed = term_code != "MaxStepsCap" && !term_code.is_empty();
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if std::env::var("FLEXMESA_BENCH_LOG").is_ok() {
        eprintln!(
            "[binary_e2e] case={case_name} uncapped={uncapped} passed={passed} steps={} term={term_code} m1={m1_msun:.4} m2={m2_msun:.4} sep={:.4} wall_s={:.2}",
            bin.model_number,
            bin.separation / RSUN,
            wall_ms / 1000.0
        );
    }

    let accretor_retries = match &bin.companion {
        BinaryCompanion::PointMass { .. } => 0,
        BinaryCompanion::Star(s2) => s2.num_retries,
    };
    Ok(BinaryCaseRunReport {
        case_name,
        passed,
        termination_code: term_code,
        total_steps: bin.model_number,
        final_m1_msun: m1_msun,
        final_m2_msun: m2_msun,
        final_separation_rsun: bin.separation / RSUN,
        final_period_days: bin.period / SECDAY,
        final_abs_mdot_msun_yr: bin.mtransfer_rate.abs() * SECYER / MSUN,
        wall_time_ms: wall_ms,
        binary_retries: bin.counters.num_retries,
        implicit_trials: bin.counters.num_implicit_trials,
        dt_cuts: bin.counters.num_dt_cuts,
        hard_limit_retries: bin.counters.num_hard_limit_retries,
        donor_retries: bin.donor.num_retries,
        accretor_retries,
    })
}
