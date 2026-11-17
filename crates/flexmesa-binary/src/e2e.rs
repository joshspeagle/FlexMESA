// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stage `E2E-5` (`#96` / `#114` / `#137`) genuine coupled `binary/test_suite`
//! execution driver across all 8 active `binary/test_suite` cases (`double_bh`,
//! `evolve_both_stars`, `jdot_gr_check`, `jdot_ls_check`, `jdot_ml_check`,
//! `star_plus_point_mass`, `star_plus_point_mass_explicit_mdot`,
//! `wind_fed_bhhmxb`).

use std::fs;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use flexmesa_const::{MSUN, RSUN, SECDAY, SECYER};
use flexmesa_star::{
    init_or_update_model_from_bundle, load_zams_model,
    load_zams_model_from_catalog, relax_composition_with_henyey,
    DefaultStarExtras, InlistBundle, InlistLoadOptions, StarError, StarModel,
    ZamsCatalog,
};

use crate::binary_history::BinaryLogsWriter;
use crate::e2e_extras::build_test_suite_extras;
pub use crate::e2e_oracles::{
    eval_brott_wind_msun_yr, eval_gr_expected_separation_rsun,
    eval_tauris_vdh_expected_separation_rsun, is_tier5_binary_case,
    BinaryInlistControls, TIER5_BINARY_CASES,
};
use crate::system::{BinaryCompanion, BinarySystem};

static FULL_ZAMS_CATALOG: OnceLock<Result<Option<ZamsCatalog>, String>> =
    OnceLock::new();

fn resolve_full_zams_catalog(
    case_dir: &Path,
) -> Result<Option<&'static ZamsCatalog>, StarError> {
    let res = FULL_ZAMS_CATALOG.get_or_init(|| {
        let Some(zams_path) = case_dir.ancestors().find_map(|anc| {
            let p = anc
                .join("data")
                .join("star_data")
                .join("zams_models")
                .join("zams_z2m2_y28.data");
            p.is_file().then_some(p)
        }) else {
            return Ok(None);
        };
        let text = fs::read_to_string(&zams_path).map_err(|e| {
            format!(
                "failed reading ZAMS catalog '{}': {e}",
                zams_path.display()
            )
        })?;
        let cat = ZamsCatalog::parse_ascii(&text).map_err(|e| {
            format!(
                "failed parsing ZAMS catalog '{}': {e}",
                zams_path.display()
            )
        })?;
        Ok(Some(cat))
    });
    match res {
        Ok(opt) => Ok(opt.as_ref()),
        Err(msg) => Err(StarError::IoError(msg.clone())),
    }
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
    let seed_zams = if let Some(cat) = resolve_full_zams_catalog(case_dir)? {
        Some(load_zams_model_from_catalog(
            cat,
            init_bundle.controls.clone(),
            init_bundle.job.clone(),
        )?)
    } else {
        Some(load_zams_model(
            init_bundle.controls.clone(),
            init_bundle.job.clone(),
        )?)
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

/// Telemetry and verification report for a single `binary/test_suite` case.
#[derive(Debug, Clone)]
pub struct BinaryCaseRunReport {
    /// Case directory name (e.g. `"jdot_gr_check"`).
    pub case_name: String,
    /// Whether all coupled steps and physical/analytical assertions passed.
    pub passed: bool,
    /// Termination code string (`"GrCheckPass"`, `"MlCheckPass"`, `"LsCheckPass"`,
    /// `"WindFedBhHmxbPass"`, `"DoubleBhPass"`, `"AccretorOverflowTerminate"`, `"BinaryEvolutionPass"`).
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
    /// Maximum relative analytical oracle error across the run.
    pub max_relative_oracle_error: f64,
    /// Wall-clock runtime in milliseconds (`ms`).
    pub wall_time_ms: f64,
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

/// Executes a single `binary/test_suite/<case>` directory with explicit
/// [`InlistLoadOptions`].
pub fn run_binary_test_suite_case_with_options(
    case_dir: &Path,
    max_steps_cap: Option<usize>,
    logs_opts: BinaryLogsOptions<'_>,
    inlist_opts: InlistLoadOptions,
) -> Result<BinaryCaseRunReport, StarError> {
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
        BinarySystem::from_controls_point_mass(donor, bctrl.clone())?
    };

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
    let mut case_extras = build_test_suite_extras(&case_name, &bin, &bctrl);

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
    while bin.model_number < step_limit {
        let (suggested_dt, _) = bin
            .try_step_with_extras(dt_next, case_extras.as_mut())
            .map_err(|e| {
                StarError::ConvergenceFailure(format!(
                    "[{case_name} step {}] {e}",
                    bin.model_number + 1
                ))
            })?;
        dt_next = suggested_dt;
        let has_term = bin.last_termination_reason.is_some();
        let is_terminating = has_term || bin.model_number >= step_limit;
        if let Some(ref mut lw) = logs_writer {
            lw.record_step(&bin, bin.model_number == 1, is_terminating)?;
        }
        if let Some(ref reason) = bin.last_termination_reason {
            term_code.clone_from(reason);
            break;
        }
    }

    let [m1_final_g, m2_final_g] = bin.star_masses_g();
    let m1_msun = m1_final_g / MSUN;
    let m2_msun = m2_final_g / MSUN;
    let passed = case_extras.evaluate_pass(&bin, uncapped, &term_code);
    let max_rel_err = case_extras.max_relative_oracle_error();
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if std::env::var("FLEXMESA_BENCH_LOG").is_ok() {
        eprintln!(
            "[binary_e2e] case={case_name} uncapped={uncapped} passed={passed} steps={} term={term_code} m1={m1_msun:.4} m2={m2_msun:.4} sep={:.4} max_rel_err={max_rel_err:.3e} wall_s={:.2}",
            bin.model_number, bin.separation / RSUN, wall_ms / 1000.0
        );
    }

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
        max_relative_oracle_error: max_rel_err,
        wall_time_ms: wall_ms,
    })
}

/// Executes all 8 `TIER5_BINARY_CASES` under `<mesa_dir>/binary/test_suite`.
pub fn run_binary_suite(
    mesa_dir: &Path,
    max_steps_cap: Option<usize>,
) -> Result<Vec<BinaryCaseRunReport>, StarError> {
    run_binary_suite_with_options(
        mesa_dir,
        max_steps_cap,
        InlistLoadOptions::default(),
    )
}

/// Executes all 8 `TIER5_BINARY_CASES` under `<mesa_dir>/binary/test_suite`
/// with explicit [`InlistLoadOptions`].
pub fn run_binary_suite_with_options(
    mesa_dir: &Path,
    max_steps_cap: Option<usize>,
    inlist_opts: InlistLoadOptions,
) -> Result<Vec<BinaryCaseRunReport>, StarError> {
    let suite_dir = mesa_dir.join("binary").join("test_suite");
    let mut reports = Vec::with_capacity(TIER5_BINARY_CASES.len());
    for &case_name in &TIER5_BINARY_CASES {
        let case_dir = suite_dir.join(case_name);
        reports.push(run_binary_test_suite_case_with_options(
            &case_dir,
            max_steps_cap,
            BinaryLogsOptions::default(),
            inlist_opts,
        )?);
    }
    Ok(reports)
}
