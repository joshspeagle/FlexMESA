// SPDX-License-Identifier: GPL-3.0-or-later
//! Strict schema-validated namelist parser for `&astero_search_controls` and
//! headless `&astero_pgstar_controls` (`DECISIONS #25`).
//!
//! Corresponds to `astero/private/astero_run_support.f90` and
//! `astero/defaults/{astero_search,astero_pgstar}.defaults`.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use flexmesa_schema::{
    copy_cleaned_case_dir, is_headless_plotting_section, is_include_key,
    parse_namelist_file, section_include_prefix, upstream_section_keys,
    ControlValidationReport, ControlValue, NamelistAssignment, NamelistBlock,
};
use flexmesa_star::{
    supported_star_control_inventory, InlistBundle, InlistLoadOptions,
    StarError,
};

use crate::inlist_astero::AsteroSearchControls;

/// Supported `&astero_search_controls` keys in MESA `26.4.1`.
pub const ASTERO_SEARCH_CONTROLS_SUPPORTED: &[&str] = &[
    "add_atmosphere",
    "add_center_point",
    "age_sigma",
    "age_target",
    "astero_results_directory",
    "avg_age_sigma_limit",
    "avg_model_number_sigma_limit",
    "bobyqa_output_filename",
    "bobyqa_rhoend",
    "chi2_delta_nu_limit",
    "chi2_limit_for_small_timesteps",
    "chi2_limit_for_smaller_timesteps",
    "chi2_limit_for_smallest_timesteps",
    "chi2_radial_limit",
    "chi2_relative_increase_limit",
    "chi2_search_limit1",
    "chi2_search_limit2",
    "chi2_seismo_delta_nu_fraction",
    "chi2_seismo_fraction",
    "chi2_seismo_nu_max_fraction",
    "chi2_seismo_r_010_fraction",
    "chi2_seismo_r_02_fraction",
    "chi2_spectroscopic_limit",
    "constraint_name",
    "constraint_sigma",
    "constraint_target",
    "correction_b",
    "correction_factor",
    "correction_scheme",
    "delta_nu",
    "delta_nu_sigma",
    "delta_param",
    "do_redistribute_mesh",
    "dt_for_smaller_steps_before_age_target",
    "dydz",
    "eval_chi2_at_target_age_only",
    "f0_ov_div_f_ov",
    "file_column_for_param",
    "filename_for_parameters",
    "first_param",
    "freq_sigma",
    "freq_target",
    "from_file_output_filename",
    "gyre_input_file",
    "include_age_in_chi2_spectro",
    "include_constraint_in_chi2_spectro",
    "iscan_factor",
    "keep_surface_point",
    "l0_n_obs",
    "limit_num_chi2_too_big",
    "lnuc_div_l_limit",
    "max_age_for_chi2",
    "max_num_from_file",
    "max_num_samples_for_avg",
    "max_param",
    "max_yrs_dt_chi2_small_limit",
    "max_yrs_dt_chi2_smaller_limit",
    "max_yrs_dt_chi2_smallest_limit",
    "max_yrs_dt_when_cold",
    "max_yrs_dt_when_hot",
    "max_yrs_dt_when_warm",
    "min_age_for_chi2",
    "min_age_limit",
    "min_num_samples_for_avg",
    "min_param",
    "newuoa_output_filename",
    "newuoa_rhoend",
    "nl",
    "normalize_chi2_seismo_frequencies",
    "normalize_chi2_seismo_r_010",
    "normalize_chi2_seismo_r_02",
    "normalize_chi2_spectro",
    "nu_lower_factor",
    "nu_max",
    "nu_max_sigma",
    "nu_upper_factor",
    "num_smaller_steps_before_age_target",
    "oscillation_code",
    "param_name",
    "scan_grid_output_filename",
    "search_type",
    "sigmas_coeff_for_constraint_limit",
    "sigmas_coeff_for_delta_nu_limit",
    "simplex_adaptive_random_search",
    "simplex_centroid_weight_power",
    "simplex_chi2_tol",
    "simplex_enforce_bounds",
    "simplex_fcn_calls_max",
    "simplex_itermax",
    "simplex_output_filename",
    "simplex_seed",
    "simplex_x_atol",
    "simplex_x_rtol",
    "vary_param",
    "y0",
    "y_depends_on_z",
    "z_div_x_solar",
];

/// Parsed-but-unused or unimplemented `&astero_search_controls` keys (`N1-28`).
pub const ASTERO_SEARCH_CONTROLS_UNUSED: &[&str] =
    &["gyre_non_ad", "trace_limits", "y_frac_he3"];

/// Supported control-key sets for asteroseismic namelists.
#[derive(Clone, Debug)]
pub struct AsteroControlInventory {
    /// Supported `&astero_search_controls` keys.
    pub supported: HashSet<&'static str>,
    /// Unimplemented / parsed-but-unused `&astero_search_controls` keys.
    pub unused: HashSet<&'static str>,
}

static ASTERO_INVENTORY: OnceLock<AsteroControlInventory> = OnceLock::new();

/// Returns the cached [`AsteroControlInventory`] for asteroseismic namelists.
#[must_use]
pub fn supported_astero_control_inventory() -> &'static AsteroControlInventory {
    ASTERO_INVENTORY.get_or_init(|| AsteroControlInventory {
        supported: ASTERO_SEARCH_CONTROLS_SUPPORTED.iter().copied().collect(),
        unused: ASTERO_SEARCH_CONTROLS_UNUSED.iter().copied().collect(),
    })
}

/// Validates parsed asteroseismic namelist sections against MESA `26.4.1`
/// defaults with `&astero_pgstar_controls` validation gated by `pgstar_active`
/// (`astero/private/astero_run_support.f90:103-106`).
#[must_use]
pub fn validate_astero_sections_with_pgstar<S: ::std::hash::BuildHasher>(
    sections: &HashMap<String, Vec<String>, S>,
    pgstar_active: bool,
) -> ControlValidationReport {
    let inv = supported_astero_control_inventory();
    let empty: HashSet<&'static str> = HashSet::new();
    let mut report = ControlValidationReport::default();

    for (sec, keys) in sections {
        match sec.as_str() {
            "astero_pgstar_controls" => {
                report.validate_headless_plotting_section_with_active(
                    sec,
                    keys,
                    pgstar_active,
                );
            }
            "astero_search_controls" => {
                report.validate_implemented_section(
                    sec,
                    keys,
                    &inv.supported,
                    &inv.unused,
                    &empty,
                );
            }
            other => {
                if is_headless_plotting_section(other) {
                    report.validate_headless_plotting_section_with_active(
                        other,
                        keys,
                        pgstar_active,
                    );
                } else {
                    report.validate_other_section(other);
                }
            }
        }
    }
    report
}

/// Validates parsed asteroseismic namelist sections against MESA `26.4.1`
/// defaults (`pgstar_flag = false`).
#[must_use]
pub fn validate_astero_sections<S: ::std::hash::BuildHasher>(
    sections: &HashMap<String, Vec<String>, S>,
) -> ControlValidationReport {
    validate_astero_sections_with_pgstar(sections, false)
}

fn expect_bool(sec: &str, a: &NamelistAssignment) -> Result<bool, StarError> {
    if a.subscript.is_some() {
        return Err(StarError::InvalidInput(format!(
            "unexpected subscript on scalar '&{sec}::{}' at line {}",
            a.name, a.line_number
        )));
    }
    match a.value {
        ControlValue::Logical(b) => Ok(b),
        ref other => Err(StarError::InvalidInput(format!(
            "expected boolean for '&{sec}::{}' at line {}, found {other:?}",
            a.name, a.line_number
        ))),
    }
}

fn expect_f64(sec: &str, a: &NamelistAssignment) -> Result<f64, StarError> {
    match a.value {
        ControlValue::Float(x) if x.is_finite() => Ok(x),
        ControlValue::Integer(i) => Ok(i as f64),
        ref other => Err(StarError::InvalidInput(format!(
            "expected finite real number for '&{sec}::{}' at line {}, found {other:?}",
            a.name, a.line_number
        ))),
    }
}

fn expect_i64(sec: &str, a: &NamelistAssignment) -> Result<i64, StarError> {
    match a.value {
        ControlValue::Integer(i) => Ok(i),
        ref other => Err(StarError::InvalidInput(format!(
            "expected integer for '&{sec}::{}' at line {}, found {other:?}",
            a.name, a.line_number
        ))),
    }
}

fn expect_str(sec: &str, a: &NamelistAssignment) -> Result<String, StarError> {
    match &a.value {
        ControlValue::String(s) => Ok(s.trim().to_string()),
        other => Err(StarError::InvalidInput(format!(
            "expected string for '&{sec}::{}' at line {}, found {other:?}",
            a.name, a.line_number
        ))),
    }
}

fn resolve_extra_astero_includes(
    case_dir: &Path,
    blocks: &[NamelistBlock],
    visited: &mut HashSet<(String, PathBuf)>,
    out_assigns: &mut HashMap<String, Vec<NamelistAssignment>>,
    pgstar_active: bool,
) -> Result<(), StarError> {
    #[derive(Default)]
    struct Inc {
        read: Option<bool>,
        name: Option<String>,
    }

    let mut per_sec: HashMap<String, HashMap<usize, Inc>> = HashMap::new();
    for block in blocks {
        let sec = block.name.to_ascii_lowercase();
        let entry = out_assigns.entry(sec.clone()).or_default();
        let prefix = section_include_prefix(&sec);
        let read_key = format!("read_extra_{prefix}_inlist");
        let name_key = format!("extra_{prefix}_inlist_name");
        for a in &block.assignments {
            entry.push(a.clone());
            let low = a.name.to_ascii_lowercase();
            if low == read_key {
                if let Some(idx) =
                    a.subscript.as_deref().and_then(|s| s.parse::<usize>().ok())
                {
                    let b = match a.value {
                        ControlValue::Logical(v) => v,
                        ref other => {
                            return Err(StarError::InvalidInput(format!(
                                "expected boolean for '&{sec}::{low}({idx})', found {other:?}"
                            )));
                        }
                    };
                    per_sec
                        .entry(sec.clone())
                        .or_default()
                        .entry(idx)
                        .or_default()
                        .read = Some(b);
                }
            } else if low == name_key {
                if let Some(idx) =
                    a.subscript.as_deref().and_then(|s| s.parse::<usize>().ok())
                {
                    let s = expect_str(&sec, a)?;
                    per_sec
                        .entry(sec.clone())
                        .or_default()
                        .entry(idx)
                        .or_default()
                        .name = Some(s);
                }
            }
        }
    }

    let mut pending: Vec<(String, PathBuf)> = Vec::new();
    for (sec, slots) in per_sec {
        if sec == "astero_pgstar_controls" && !pgstar_active {
            continue;
        }
        let mut indices: Vec<usize> = slots.keys().copied().collect();
        indices.sort_unstable();
        for idx in indices {
            if let Some(inc) = slots.get(&idx) {
                if inc.read == Some(true) {
                    let Some(fname) =
                        inc.name.as_deref().map(str::trim).filter(|s| {
                            !s.is_empty()
                                && !s.eq_ignore_ascii_case("undefined")
                        })
                    else {
                        return Err(StarError::InvalidInput(format!(
                            "&{sec} include flag at index {idx} is .true. without a valid filename"
                        )));
                    };
                    let inc_path = case_dir.join(fname);
                    if !inc_path.is_file() {
                        return Err(StarError::IoError(format!(
                            "missing '&{sec}' extra inlist file '{}'",
                            inc_path.display()
                        )));
                    }
                    let key = (sec.clone(), inc_path.clone());
                    if visited.insert(key.clone()) {
                        pending.push(key);
                    }
                }
            }
        }
    }

    for (sec, path) in pending {
        let extra_blocks = parse_namelist_file(&path).map_err(|e| {
            StarError::IoError(format!(
                "failed parsing extra inlist '{}' for '&{sec}': {e}",
                path.display()
            ))
        })?;
        let filtered: Vec<NamelistBlock> = extra_blocks
            .into_iter()
            .filter(|b| b.name.eq_ignore_ascii_case(&sec))
            .collect();
        resolve_extra_astero_includes(
            case_dir,
            &filtered,
            visited,
            out_assigns,
            pgstar_active,
        )?;
    }
    Ok(())
}

/// Copies an asteroseismic test case directory `src_case` to `dst_case` and
/// strips unsupported sections/keys so strict `#147` loading passes cleanly.
///
/// # Errors
/// Returns [`StarError::IoError`] if copying or rewriting files fails.
pub fn copy_cleaned_astero_case(
    src_case: &Path,
    dst_case: &Path,
) -> Result<(), StarError> {
    let star_inv = supported_star_control_inventory();
    let ast_inv = supported_astero_control_inventory();
    copy_cleaned_case_dir(
        src_case,
        dst_case,
        &|sec| {
            matches!(
                sec,
                "star_job"
                    | "eos"
                    | "kap"
                    | "controls"
                    | "pgstar"
                    | "astero_search_controls"
                    | "astero_pgstar_controls"
            )
        },
        &|sec, key| match sec {
            "star_job" => {
                is_include_key(sec, key)
                    || star_inv.star_job_supported.contains(key)
                    || star_inv.star_job_headless_gui_flags.contains(key)
            }
            "eos" => {
                is_include_key(sec, key) || star_inv.eos_supported.contains(key)
            }
            "kap" => {
                is_include_key(sec, key) || star_inv.kap_supported.contains(key)
            }
            "controls" => {
                is_include_key(sec, key)
                    || star_inv.controls_supported.contains(key)
            }
            "astero_search_controls" => {
                is_include_key(sec, key) || ast_inv.supported.contains(key)
            }
            "pgstar" | "astero_pgstar_controls" => {
                is_include_key(sec, key)
                    || upstream_section_keys(sec)
                        .is_some_and(|set| set.contains(key))
            }
            _ => false,
        },
    )
    .map_err(|e| StarError::IoError(e.to_string()))
}

/// Parses `&astero_search_controls` (and validates `&astero_pgstar_controls`)
/// from `case_dir` with strict defaults (`allow_unsupported_controls = false`).
pub fn parse_astero_search_controls(
    case_dir: &Path,
) -> Result<AsteroSearchControls, StarError> {
    parse_astero_search_controls_with_options(
        case_dir,
        InlistLoadOptions::default(),
    )
}

/// Parses `&astero_search_controls` (and validates `&astero_pgstar_controls`)
/// from `case_dir` with explicit [`InlistLoadOptions`].
pub fn parse_astero_search_controls_with_options(
    case_dir: &Path,
    options: InlistLoadOptions,
) -> Result<AsteroSearchControls, StarError> {
    let ast_path = case_dir.join("inlist_astero_search_controls");
    let main_inlist = case_dir.join("inlist");
    let mut visited: HashSet<(String, PathBuf)> = HashSet::new();
    let mut section_assigns: HashMap<String, Vec<NamelistAssignment>> =
        HashMap::new();

    let pgstar_active = if main_inlist.is_file() {
        let main_sections =
            InlistBundle::load_sections_in_case_dir(case_dir, "inlist")?;
        match main_sections.get("star_job") {
            Some(s) => s
                .try_get_bool("star_job", "pgstar_flag")?
                .unwrap_or(false),
            None => false,
        }
    } else {
        false
    };

    if ast_path.is_file() {
        let blocks = parse_namelist_file(&ast_path).map_err(|e| {
            StarError::IoError(format!(
                "failed parsing '{}': {e}",
                ast_path.display()
            ))
        })?;
        resolve_extra_astero_includes(
            case_dir,
            &blocks,
            &mut visited,
            &mut section_assigns,
            pgstar_active,
        )?;
    }

    if main_inlist.is_file() {
        let main_blocks = parse_namelist_file(&main_inlist).map_err(|e| {
            StarError::IoError(format!(
                "failed parsing '{}': {e}",
                main_inlist.display()
            ))
        })?;
        let ast_blocks: Vec<NamelistBlock> = main_blocks
            .into_iter()
            .filter(|b| {
                let low = b.name.to_ascii_lowercase();
                low == "astero_search_controls"
                    || low == "astero_pgstar_controls"
            })
            .collect();
        resolve_extra_astero_includes(
            case_dir,
            &ast_blocks,
            &mut visited,
            &mut section_assigns,
            pgstar_active,
        )?;
    }

    if !ast_path.is_file()
        && !section_assigns.contains_key("astero_search_controls")
    {
        return Err(StarError::IoError(format!(
            "missing '{}' (and no &astero_search_controls in '{}')",
            ast_path.display(),
            main_inlist.display()
        )));
    }

    let key_sets: HashMap<String, Vec<String>> = section_assigns
        .iter()
        .map(|(sec, assigns)| {
            let uniq: BTreeSet<String> = assigns
                .iter()
                .map(|a| a.name.to_ascii_lowercase())
                .collect();
            (sec.clone(), uniq.into_iter().collect())
        })
        .collect();
    let report = validate_astero_sections_with_pgstar(&key_sets, pgstar_active);
    report
        .into_result(options.allow_unsupported_controls)
        .map_err(|e| StarError::InvalidInput(e.to_string()))?;

    let mut ctrl = AsteroSearchControls::default();
    if let Some(assigns) = section_assigns.get("astero_search_controls") {
        for a in assigns {
            apply_assignment(a, &mut ctrl)?;
        }
    }
    finalize_controls(&mut ctrl)?;
    Ok(ctrl)
}

fn parse_subscript_tokens(sub: &str) -> Result<Vec<Option<usize>>, StarError> {
    let mut parts = Vec::new();
    for raw in sub.split(',') {
        let t = raw.trim();
        if t == ":" {
            parts.push(None);
        } else {
            let idx = t.parse::<usize>().map_err(|e| {
                StarError::InvalidInput(format!(
                    "invalid astero namelist subscript '{sub}': {e}"
                ))
            })?;
            parts.push(Some(idx));
        }
    }
    Ok(parts)
}

fn apply_assignment(
    a: &NamelistAssignment,
    c: &mut AsteroSearchControls,
) -> Result<(), StarError> {
    let key = a.name.to_ascii_lowercase();
    let sec = "astero_search_controls";
    if is_include_key(sec, &key) {
        return Ok(());
    }
    if ASTERO_SEARCH_CONTROLS_UNUSED.contains(&key.as_str()) {
        return Ok(());
    }
    if let Some(sub) = a.subscript.as_deref() {
        let dims = parse_subscript_tokens(sub)?;
        return apply_indexed_assignment(&key, &dims, a, c);
    }

    match key.as_str() {
        "chi2_seismo_fraction" => {
            c.weights.chi2_seismo_fraction = expect_f64(sec, a)?;
        }
        "chi2_seismo_delta_nu_fraction" => {
            c.weights.chi2_seismo_delta_nu_fraction = expect_f64(sec, a)?;
        }
        "chi2_seismo_nu_max_fraction" => {
            c.weights.chi2_seismo_nu_max_fraction = expect_f64(sec, a)?;
        }
        "chi2_seismo_r_010_fraction" => {
            c.weights.chi2_seismo_r_010_fraction = expect_f64(sec, a)?;
        }
        "chi2_seismo_r_02_fraction" => {
            c.weights.chi2_seismo_r_02_fraction = expect_f64(sec, a)?;
        }
        "normalize_chi2_seismo_frequencies" => {
            c.weights.normalize_chi2_seismo_frequencies = expect_bool(sec, a)?;
        }
        "normalize_chi2_seismo_r_010" => {
            c.weights.normalize_chi2_seismo_r_010 = expect_bool(sec, a)?;
        }
        "normalize_chi2_seismo_r_02" => {
            c.weights.normalize_chi2_seismo_r_02 = expect_bool(sec, a)?;
        }
        "normalize_chi2_spectro" => {
            c.weights.normalize_chi2_spectro = expect_bool(sec, a)?;
        }
        "z_div_x_solar" => c.z_div_x_solar = expect_f64(sec, a)?,
        "delta_nu" => c.delta_nu = expect_f64(sec, a)?,
        "delta_nu_sigma" => c.delta_nu_sigma = expect_f64(sec, a)?,
        "sigmas_coeff_for_delta_nu_limit" => {
            c.sigmas_coeff_for_delta_nu_limit = expect_f64(sec, a)?;
        }
        "nu_max" => c.nu_max = expect_f64(sec, a)?,
        "nu_max_sigma" => c.nu_max_sigma = expect_f64(sec, a)?,
        "oscillation_code" => c.oscillation_code = expect_str(sec, a)?,
        "nu_lower_factor" => c.nu_lower_factor = expect_f64(sec, a)?,
        "nu_upper_factor" => c.nu_upper_factor = expect_f64(sec, a)?,
        "gyre_input_file" => c.gyre_input_file = expect_str(sec, a)?,
        "add_atmosphere" => c.add_atmosphere = expect_bool(sec, a)?,
        "keep_surface_point" => c.keep_surface_point = expect_bool(sec, a)?,
        "add_center_point" => c.add_center_point = expect_bool(sec, a)?,
        "do_redistribute_mesh" => {
            c.do_redistribute_mesh = expect_bool(sec, a)?;
        }
        "correction_scheme" => c.correction_scheme = expect_str(sec, a)?,
        "correction_b" => c.correction_b = expect_f64(sec, a)?,
        "correction_factor" => c.correction_factor = expect_f64(sec, a)?,
        "search_type" => c.search_type = expect_str(sec, a)?,
        "include_age_in_chi2_spectro" => {
            c.include_age_in_chi2_spectro = expect_bool(sec, a)?;
        }
        "eval_chi2_at_target_age_only" => {
            c.eval_chi2_at_target_age_only = expect_bool(sec, a)?;
        }
        "age_target" => c.age_target = expect_f64(sec, a)?,
        "age_sigma" => c.age_sigma = expect_f64(sec, a)?,
        "num_smaller_steps_before_age_target" => {
            let v = expect_i64(sec, a)?;
            if v < 0 {
                return Err(StarError::InvalidInput(format!(
                    "negative '&{sec}::{key}' = {v}"
                )));
            }
            c.num_smaller_steps_before_age_target = v as usize;
        }
        "dt_for_smaller_steps_before_age_target" => {
            c.dt_for_smaller_steps_before_age_target = expect_f64(sec, a)?;
        }
        "min_age_for_chi2" => {
            let x = expect_f64(sec, a)?;
            c.min_age_for_chi2 = (x > 0.0).then_some(x);
        }
        "max_age_for_chi2" => {
            let x = expect_f64(sec, a)?;
            c.max_age_for_chi2 = (x > 0.0).then_some(x);
        }
        "min_age_limit" => c.min_age_limit = expect_f64(sec, a)?,
        "lnuc_div_l_limit" => c.lnuc_div_l_limit = expect_f64(sec, a)?,
        "chi2_spectroscopic_limit" => {
            c.chi2_spectroscopic_limit = expect_f64(sec, a)?;
        }
        "chi2_delta_nu_limit" => c.chi2_delta_nu_limit = expect_f64(sec, a)?,
        "chi2_radial_limit" => c.chi2_radial_limit = expect_f64(sec, a)?,
        "chi2_search_limit1" => c.chi2_search_limit1 = expect_f64(sec, a)?,
        "chi2_search_limit2" => c.chi2_search_limit2 = expect_f64(sec, a)?,
        "chi2_relative_increase_limit" => {
            c.chi2_relative_increase_limit = expect_f64(sec, a)?;
        }
        "limit_num_chi2_too_big" => {
            let v = expect_i64(sec, a)?;
            if v < 0 {
                return Err(StarError::InvalidInput(format!(
                    "negative '&{sec}::{key}' = {v}"
                )));
            }
            c.limit_num_chi2_too_big = v as usize;
        }
        "y_depends_on_z" => c.y_depends_on_z = expect_bool(sec, a)?,
        "y0" => c.y0 = expect_f64(sec, a)?,
        "dydz" => c.dydz = expect_f64(sec, a)?,
        "f0_ov_div_f_ov" => c.f0_ov_div_f_ov = expect_f64(sec, a)?,
        "simplex_itermax" => {
            let v = expect_i64(sec, a)?;
            if v <= 0 {
                return Err(StarError::InvalidInput(format!(
                    "non-positive '&{sec}::{key}' = {v}"
                )));
            }
            c.simplex_itermax = v as usize;
        }
        "simplex_fcn_calls_max" => {
            let v = expect_i64(sec, a)?;
            if v <= 0 {
                return Err(StarError::InvalidInput(format!(
                    "non-positive '&{sec}::{key}' = {v}"
                )));
            }
            c.simplex_fcn_calls_max = v as usize;
        }
        "simplex_x_atol" => c.simplex_x_atol = expect_f64(sec, a)?,
        "simplex_x_rtol" => c.simplex_x_rtol = expect_f64(sec, a)?,
        "simplex_chi2_tol" => c.simplex_chi2_tol = expect_f64(sec, a)?,
        "simplex_centroid_weight_power" => {
            c.simplex_centroid_weight_power = expect_f64(sec, a)?;
        }
        "simplex_enforce_bounds" => {
            c.simplex_enforce_bounds = expect_bool(sec, a)?;
        }
        "simplex_adaptive_random_search" => {
            c.simplex_adaptive_random_search = expect_bool(sec, a)?;
        }
        "simplex_seed" => {
            let v = expect_i64(sec, a)?;
            c.simplex_seed = v as i32;
        }
        "simplex_output_filename" => {
            c.simplex_output_filename = expect_str(sec, a)?;
        }
        "bobyqa_rhoend" => c.bobyqa_rho_end = expect_f64(sec, a)?,
        "bobyqa_output_filename" => {
            c.bobyqa_output_filename = expect_str(sec, a)?;
        }
        "newuoa_rhoend" => c.newuoa_rho_end = expect_f64(sec, a)?,
        "newuoa_output_filename" => {
            c.newuoa_output_filename = expect_str(sec, a)?;
        }
        "astero_results_directory" => {
            c.astero_results_directory = expect_str(sec, a)?;
        }
        "scan_grid_output_filename" => {
            c.scan_grid_output_filename = expect_str(sec, a)?;
        }
        "filename_for_parameters" => {
            c.filename_for_parameters = expect_str(sec, a)?;
        }
        "max_num_from_file" => {
            c.max_num_from_file = expect_i64(sec, a)? as isize;
        }
        "from_file_output_filename" => {
            c.from_file_output_filename = expect_str(sec, a)?;
        }
        "max_yrs_dt_when_cold" => {
            c.max_yrs_dt_when_cold = expect_f64(sec, a)?;
        }
        "max_yrs_dt_when_warm" => {
            c.max_yrs_dt_when_warm = expect_f64(sec, a)?;
        }
        "max_yrs_dt_when_hot" => c.max_yrs_dt_when_hot = expect_f64(sec, a)?,
        "chi2_limit_for_small_timesteps" => {
            c.chi2_limit_for_small_timesteps = expect_f64(sec, a)?;
        }
        "max_yrs_dt_chi2_small_limit" => {
            c.max_yrs_dt_chi2_small_limit = expect_f64(sec, a)?;
        }
        "chi2_limit_for_smaller_timesteps" => {
            c.chi2_limit_for_smaller_timesteps = expect_f64(sec, a)?;
        }
        "max_yrs_dt_chi2_smaller_limit" => {
            c.max_yrs_dt_chi2_smaller_limit = expect_f64(sec, a)?;
        }
        "chi2_limit_for_smallest_timesteps" => {
            c.chi2_limit_for_smallest_timesteps = expect_f64(sec, a)?;
        }
        "max_yrs_dt_chi2_smallest_limit" => {
            c.max_yrs_dt_chi2_smallest_limit = expect_f64(sec, a)?;
        }
        "avg_age_sigma_limit" => c.avg_age_sigma_limit = expect_f64(sec, a)?,
        "avg_model_number_sigma_limit" => {
            c.avg_model_number_sigma_limit = expect_f64(sec, a)?;
        }
        "min_num_samples_for_avg" | "max_num_samples_for_avg" => {
            let v = expect_i64(sec, a)?;
            if v < 0 {
                return Err(StarError::InvalidInput(format!(
                    "negative '&{sec}::{key}' = {v}"
                )));
            }
            if key == "min_num_samples_for_avg" {
                c.min_num_samples_for_avg = v as usize;
            } else {
                c.max_num_samples_for_avg = v as usize;
            }
        }
        other => {
            // Upstream control skipped only when `allow_unsupported_controls = true`.
            if upstream_section_keys(sec).is_some_and(|s| s.contains(other)) {
                return Ok(());
            }
            return Err(StarError::InvalidInput(format!(
                "unknown scalar control '&{sec}::{other}' at line {}",
                a.line_number
            )));
        }
    }
    Ok(())
}

fn apply_indexed_assignment(
    key: &str,
    dims: &[Option<usize>],
    a: &NamelistAssignment,
    c: &mut AsteroSearchControls,
) -> Result<(), StarError> {
    let sec = "astero_search_controls";
    if key == "nl" && dims.len() == 1 {
        let Some(el) = dims[0] else {
            let v = expect_i64(sec, a)?;
            if v < 0 {
                return Err(StarError::InvalidInput(format!(
                    "negative '&{sec}::nl(:)' = {v}"
                )));
            }
            c.nl = [v as usize; 4];
            return Ok(());
        };
        if el >= 4 {
            return Err(StarError::InvalidInput(format!(
                "'&{sec}::nl({el})' degree must be in 0..=3"
            )));
        }
        let v = expect_i64(sec, a)?;
        if v < 0 {
            return Err(StarError::InvalidInput(format!(
                "negative '&{sec}::nl({el})' = {v}"
            )));
        }
        c.nl[el] = v as usize;
        return Ok(());
    }
    if key == "iscan_factor" && dims.len() == 1 {
        let v = expect_f64(sec, a)?;
        let Some(el) = dims[0] else {
            c.iscan_factor = [v; 4];
            return Ok(());
        };
        if el >= 4 {
            return Err(StarError::InvalidInput(format!(
                "'&{sec}::iscan_factor({el})' degree must be in 0..=3"
            )));
        }
        c.iscan_factor[el] = v;
        return Ok(());
    }
    if (key == "freq_target" || key == "freq_sigma") && dims.len() == 2 {
        let v = expect_f64(sec, a)?;
        let Some(el) = dims[0] else {
            return Err(StarError::InvalidInput(format!(
                "unsupported degree wildcard on '&{sec}::{key}' at line {}",
                a.line_number
            )));
        };
        if el >= 4 {
            return Err(StarError::InvalidInput(format!(
                "'&{sec}::{key}({el},...)' degree must be in 0..=3"
            )));
        }
        let vec = if key == "freq_target" {
            &mut c.freq_target[el]
        } else {
            &mut c.freq_sigma[el]
        };
        match dims[1] {
            None => {
                vec.fill(v);
            }
            Some(0) => {
                return Err(StarError::InvalidInput(format!(
                    "'&{sec}::{key}({el},0)' mode index must be >= 1"
                )));
            }
            Some(pos) => {
                if vec.len() < pos {
                    vec.resize(pos, 0.0);
                }
                vec[pos - 1] = v;
            }
        }
        return Ok(());
    }
    if key == "l0_n_obs" && dims.len() == 1 {
        let v = expect_f64(sec, a)?;
        match dims[0] {
            None => c.l0_n_obs.fill(v),
            Some(0) => {
                return Err(StarError::InvalidInput(format!(
                    "'&{sec}::l0_n_obs(0)' index must be >= 1"
                )));
            }
            Some(pos) => {
                if c.l0_n_obs.len() < pos {
                    c.l0_n_obs.resize(pos, -1.0);
                }
                c.l0_n_obs[pos - 1] = v;
            }
        }
        return Ok(());
    }
    if dims.len() == 1 {
        let slot_range: Vec<usize> = match dims[0] {
            None => (0..10).collect(),
            Some(0) => {
                return Err(StarError::InvalidInput(format!(
                    "'&{sec}::{key}(0)' index must be >= 1"
                )));
            }
            Some(idx) if idx <= 10 => vec![idx - 1],
            Some(idx) => {
                return Err(StarError::InvalidInput(format!(
                    "'&{sec}::{key}({idx})' exceeds supported slot count (1..=10)"
                )));
            }
        };
        for i in slot_range {
            match key {
                "constraint_name" => {
                    c.constraint_names[i] = expect_str(sec, a)?;
                }
                "constraint_target" => {
                    c.indexed_constraints[i].target = expect_f64(sec, a)?;
                }
                "constraint_sigma" => {
                    c.indexed_constraints[i].sigma = expect_f64(sec, a)?;
                }
                "include_constraint_in_chi2_spectro" => {
                    c.indexed_constraints[i].include = match a.value {
                        ControlValue::Logical(b) => b,
                        ref other => {
                            return Err(StarError::InvalidInput(format!(
                                "expected boolean for '&{sec}::{key}' at line {}, found {other:?}",
                                a.line_number
                            )));
                        }
                    };
                }
                "sigmas_coeff_for_constraint_limit" => {
                    c.indexed_constraints[i].sigmas_coeff = expect_f64(sec, a)?;
                }
                "param_name" => c.param_names[i] = expect_str(sec, a)?,
                "vary_param" => {
                    c.indexed_params[i].vary = match a.value {
                        ControlValue::Logical(b) => b,
                        ref other => {
                            return Err(StarError::InvalidInput(format!(
                                "expected boolean for '&{sec}::{key}' at line {}, found {other:?}",
                                a.line_number
                            )));
                        }
                    };
                }
                "first_param" => {
                    c.indexed_params[i].first = expect_f64(sec, a)?;
                }
                "min_param" => c.indexed_params[i].min = expect_f64(sec, a)?,
                "max_param" => c.indexed_params[i].max = expect_f64(sec, a)?,
                "delta_param" => {
                    c.indexed_params[i].delta = expect_f64(sec, a)?;
                }
                "file_column_for_param" => {
                    c.indexed_file_cols[i] = expect_i64(sec, a)? as isize;
                }
                other => {
                    if upstream_section_keys(sec)
                        .is_some_and(|s| s.contains(other))
                    {
                        return Ok(());
                    }
                    return Err(StarError::InvalidInput(format!(
                        "unknown indexed control '&{sec}::{other}' at line {}",
                        a.line_number
                    )));
                }
            }
        }
        return Ok(());
    }
    if upstream_section_keys(sec).is_some_and(|s| s.contains(key)) {
        return Ok(());
    }
    Err(StarError::InvalidInput(format!(
        "unsupported subscript shape for '&{sec}::{key}' at line {}",
        a.line_number
    )))
}

fn finalize_controls(c: &mut AsteroSearchControls) -> Result<(), StarError> {
    if !matches!(
        c.search_type.trim().to_ascii_lowercase().as_str(),
        "use_first_values"
            | "scan_grid"
            | "simplex"
            | "newuoa"
            | "bobyqa"
            | "from_file"
    ) {
        return Err(StarError::InvalidInput(format!(
            "unrecognized '&astero_search_controls::search_type' option '{}'",
            c.search_type
        )));
    }
    if !matches!(
        c.oscillation_code.trim().to_ascii_lowercase().as_str(),
        "adipls" | "gyre"
    ) {
        return Err(StarError::InvalidInput(format!(
            "unrecognized '&astero_search_controls::oscillation_code' option '{}'",
            c.oscillation_code
        )));
    }
    if !matches!(
        c.correction_scheme.trim().to_ascii_lowercase().as_str(),
        "" | "none" | "kjeldsen" | "cubic" | "combined" | "power_law" | "sonoi"
    ) {
        return Err(StarError::InvalidInput(format!(
            "unrecognized '&astero_search_controls::correction_scheme' option '{}'",
            c.correction_scheme
        )));
    }

    for i in 0..10 {
        // Constraint names are not mapped here: each named constraint's value
        // comes from the case's `set_constraint_value`
        // (`astero/private/extras_support.f90:204-209`), which fails loudly
        // on a name it cannot evaluate.
        let (ps, col) = (c.indexed_params[i], c.indexed_file_cols[i]);
        let pname = c.param_names[i].trim().to_ascii_lowercase();
        match pname.as_str() {
            "" => {}
            "initial_mass" | "mass" => {
                c.mass = ps;
                c.file_column_for_mass = col;
            }
            "initial_y" | "y" => {
                c.y = ps;
                c.file_column_for_y = col;
            }
            "initial_feh" | "feh" => {
                c.feh_param = ps;
                c.file_column_for_feh = col;
            }
            "alpha" => {
                c.alpha = ps;
                c.file_column_for_alpha = col;
            }
            "f_ov" => {
                c.f_ov = ps;
                c.file_column_for_f_ov = col;
            }
            "my_param1" | "semiconv" => {
                c.my_param[0] = ps;
                c.file_column_for_my_param[0] = col;
                c.has_custom_param[0] = true;
            }
            "my_param2" | "thermohal" => {
                c.my_param[1] = ps;
                c.file_column_for_my_param[1] = col;
                c.has_custom_param[1] = true;
            }
            "my_param3" | "diffusion" => {
                c.my_param[2] = ps;
                c.file_column_for_my_param[2] = col;
                c.has_custom_param[2] = true;
            }
            other => {
                return Err(StarError::InvalidInput(format!(
                    "unrecognized/unsupported '&astero_search_controls::param_name({})' = '{}'",
                    i + 1,
                    other
                )));
            }
        }
    }
    for l in 0..4 {
        let n = c.nl[l]
            .max(c.freq_target[l].len())
            .max(c.freq_sigma[l].len());
        c.freq_target[l].resize(n, 0.0);
        c.freq_sigma[l].resize(n, 0.0);
        let pos = c.freq_target[l].iter().take_while(|&&f| f > 0.0).count();
        c.nl[l] = pos.max(c.nl[l]);
        c.freq_target[l].truncate(c.nl[l]);
        c.freq_sigma[l].truncate(c.nl[l]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_astero_inventory_keys_exist_upstream_and_are_disjoint() {
        let upstream = upstream_section_keys("astero_search_controls")
            .expect("upstream astero_search_controls keys must exist");
        for key in ASTERO_SEARCH_CONTROLS_SUPPORTED {
            assert!(
                upstream.contains(key),
                "supported key '{key}' missing from MESA 26.4.1 astero_search.defaults"
            );
            assert!(
                !ASTERO_SEARCH_CONTROLS_UNUSED.contains(key),
                "key '{key}' appears in both SUPPORTED and UNUSED"
            );
        }
        for key in ASTERO_SEARCH_CONTROLS_UNUSED {
            assert!(
                upstream.contains(key),
                "unused key '{key}' missing from MESA 26.4.1 astero_search.defaults"
            );
        }
    }
}
