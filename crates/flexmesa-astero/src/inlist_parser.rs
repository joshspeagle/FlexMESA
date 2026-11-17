// SPDX-License-Identifier: LGPL-3.0-or-later
//! Namelist parser for `&astero_search_controls`, `include` directives
//! (`src/targets.inc`), and indexed observable/parameter arrays
//! (`astero/private/astero_run_support.f90`).

use std::fs;
use std::path::Path;

use crate::inlist_astero::{
    AsteroSearchControls, AsteroSearchParam, SpectroConstraint,
};

fn parse_bool(val: &str) -> bool {
    matches!(
        val.trim().trim_matches('.').to_ascii_lowercase().as_str(),
        "true" | "t" | "1"
    )
}

fn parse_f64(val: &str) -> Option<f64> {
    val.trim().replace(['d', 'D'], "e").parse::<f64>().ok()
}

fn strip_quotes(val: &str) -> String {
    val.trim()
        .trim_matches(|c| c == '\'' || c == '"')
        .to_string()
}

/// Parses `&astero_search_controls` from `case_dir` (`inlist_astero_search_controls` + `inlist`).
pub fn parse_astero_search_controls(
    case_dir: &Path,
) -> Result<AsteroSearchControls, String> {
    let mut ctrl = AsteroSearchControls::default();
    let mut files = vec![
        String::from("inlist_astero_search_controls"),
        String::from("inlist"),
    ];
    let mut idx = 0usize;
    while idx < files.len() {
        let rel = files[idx].clone();
        idx += 1;
        if let Ok(content) = fs::read_to_string(case_dir.join(&rel)) {
            parse_namelist_text(&content, case_dir, &mut ctrl, &mut files);
        }
    }
    finalize_controls(&mut ctrl);
    Ok(ctrl)
}

fn finalize_controls(c: &mut AsteroSearchControls) {
    for i in 0..10 {
        let cs = c.indexed_constraints[i];
        match c.constraint_names[i].to_ascii_lowercase().as_str() {
            "teff" => c.teff = cs,
            "log_g" | "logg" => c.logg = cs,
            "log_l" | "logl" => c.log_l = cs,
            "feh" => c.feh = cs,
            "m_h" => c.m_h = cs,
            "log_r" | "logr" => c.log_r = cs,
            "radius" | "r" => c.radius = cs,
            "surface_z_div_x" => c.surface_z_div_x = cs,
            "surface_he" => c.surface_he = cs,
            "rcz" => c.rcz = cs,
            _ => {}
        }
        let (ps, col) = (c.indexed_params[i], c.indexed_file_cols[i]);
        match c.param_names[i].to_ascii_lowercase().as_str() {
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
            _ => {}
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
}

fn parse_namelist_text(
    content: &str,
    case_dir: &Path,
    ctrl: &mut AsteroSearchControls,
    files: &mut Vec<String>,
) {
    for raw_line in content.lines() {
        let line = raw_line.split('!').next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with('&') || line == "/" {
            continue;
        }
        if let Some(rest) = line.to_ascii_lowercase().strip_prefix("include ") {
            let inc_file = strip_quotes(&line[line.len() - rest.len()..]);
            if let Ok(inc_text) = fs::read_to_string(case_dir.join(&inc_file)) {
                parse_namelist_text(&inc_text, case_dir, ctrl, files);
            }
            continue;
        }
        let Some((lhs, rhs)) = line.split_once('=') else {
            continue;
        };
        let key = lhs.trim().to_ascii_lowercase();
        let val = rhs.trim().trim_end_matches(',');
        if key == "inlist_astero_search_controls" {
            let f = strip_quotes(val);
            if !f.is_empty() && !files.contains(&f) {
                files.push(f);
            }
            continue;
        }
        apply_key_value(&key, val, ctrl);
    }
}

fn try_set_spectro(
    s: &mut SpectroConstraint,
    name: &str,
    key: &str,
    val: &str,
) -> bool {
    if key == format!("include_{name}_in_chi2_spectro") {
        s.include = parse_bool(val);
    } else if key == format!("{name}_target") {
        s.target = parse_f64(val).unwrap_or(0.0);
    } else if key == format!("{name}_sigma") {
        s.sigma = parse_f64(val).unwrap_or(0.0);
    } else if key == format!("sigmas_coeff_for_{name}_limit") {
        s.sigmas_coeff = parse_f64(val).unwrap_or(0.0);
    } else {
        return false;
    }
    true
}

fn try_set_param(
    p: &mut AsteroSearchParam,
    name: &str,
    key: &str,
    val: &str,
) -> bool {
    if key == format!("vary_{name}") {
        p.vary = parse_bool(val);
    } else if key == format!("first_{name}") {
        p.first = parse_f64(val).unwrap_or(p.first);
    } else if key == format!("min_{name}") {
        p.min = parse_f64(val).unwrap_or(p.min);
    } else if key == format!("max_{name}") {
        p.max = parse_f64(val).unwrap_or(p.max);
    } else if key == format!("delta_{name}") {
        p.delta = parse_f64(val).unwrap_or(0.0);
    } else {
        return false;
    }
    true
}

fn apply_indexed_key(
    arr_name: &str,
    parts: &[usize],
    val: &str,
    c: &mut AsteroSearchControls,
) {
    if arr_name == "nl" && parts.len() == 1 && parts[0] < 4 {
        c.nl[parts[0]] = val.parse::<usize>().unwrap_or(0);
    } else if arr_name == "iscan_factor" && parts.len() == 1 && parts[0] < 4 {
        c.iscan_factor[parts[0]] = parse_f64(val).unwrap_or(1.0);
    } else if (arr_name == "freq_target" || arr_name == "freq_sigma")
        && parts.len() == 2
        && parts[0] < 4
        && parts[1] >= 1
    {
        if let Some(v) = parse_f64(val) {
            let vec = if arr_name == "freq_target" {
                &mut c.freq_target[parts[0]]
            } else {
                &mut c.freq_sigma[parts[0]]
            };
            if vec.len() < parts[1] {
                vec.resize(parts[1], 0.0);
            }
            vec[parts[1] - 1] = v;
        }
    } else if parts.len() == 1 && parts[0] >= 1 {
        let i = parts[0] - 1;
        if arr_name == "l0_n_obs" {
            if let Some(v) = parse_f64(val) {
                if c.l0_n_obs.len() <= i {
                    c.l0_n_obs.resize(i + 1, -1.0);
                }
                c.l0_n_obs[i] = v;
            }
        } else if i < 10 {
            match arr_name {
                "constraint_name" => c.constraint_names[i] = strip_quotes(val),
                "constraint_target" => {
                    c.indexed_constraints[i].target =
                        parse_f64(val).unwrap_or(0.0);
                }
                "constraint_sigma" => {
                    c.indexed_constraints[i].sigma =
                        parse_f64(val).unwrap_or(0.0);
                }
                "include_constraint_in_chi2_spectro" => {
                    c.indexed_constraints[i].include = parse_bool(val);
                }
                "sigmas_coeff_for_constraint_limit" => {
                    c.indexed_constraints[i].sigmas_coeff =
                        parse_f64(val).unwrap_or(0.0);
                }
                "param_name" => c.param_names[i] = strip_quotes(val),
                "vary_param" => c.indexed_params[i].vary = parse_bool(val),
                "first_param" => {
                    c.indexed_params[i].first = parse_f64(val).unwrap_or(0.0);
                }
                "min_param" => {
                    c.indexed_params[i].min = parse_f64(val).unwrap_or(0.0);
                }
                "max_param" => {
                    c.indexed_params[i].max = parse_f64(val).unwrap_or(0.0);
                }
                "delta_param" => {
                    c.indexed_params[i].delta = parse_f64(val).unwrap_or(0.0);
                }
                "file_column_for_param" => {
                    c.indexed_file_cols[i] = val.parse::<isize>().unwrap_or(-1);
                }
                _ => {}
            }
        }
        let l_opt = match arr_name {
            "l0_obs" | "l0_obs_sigma" => Some((0, arr_name.ends_with("sigma"))),
            "l1_obs" | "l1_obs_sigma" => Some((1, arr_name.ends_with("sigma"))),
            "l2_obs" | "l2_obs_sigma" => Some((2, arr_name.ends_with("sigma"))),
            "l3_obs" | "l3_obs_sigma" => Some((3, arr_name.ends_with("sigma"))),
            _ => None,
        };
        if let (Some((l, is_sig)), Some(v)) = (l_opt, parse_f64(val)) {
            let vec = if is_sig {
                &mut c.freq_sigma[l]
            } else {
                &mut c.freq_target[l]
            };
            if vec.len() <= i {
                vec.resize(i + 1, 0.0);
            }
            vec[i] = v;
        }
    }
}

fn apply_key_value(key: &str, val: &str, c: &mut AsteroSearchControls) {
    if let Some((arr_name, idx_str)) =
        key.strip_suffix(')').and_then(|s| s.split_once('('))
    {
        let parts: Vec<usize> = idx_str
            .split(',')
            .filter_map(|p| p.trim().parse::<usize>().ok())
            .collect();
        apply_indexed_key(arr_name, &parts, val, c);
        return;
    }
    if try_set_spectro(&mut c.teff, "teff", key, val)
        || try_set_spectro(&mut c.logg, "logg", key, val)
        || try_set_spectro(&mut c.log_l, "logl", key, val)
        || try_set_spectro(&mut c.feh, "feh", key, val)
        || try_set_spectro(&mut c.log_r, "logr", key, val)
        || try_set_spectro(&mut c.radius, "r", key, val)
        || try_set_spectro(&mut c.m_h, "m_h", key, val)
        || try_set_spectro(&mut c.surface_z_div_x, "surface_z_div_x", key, val)
        || try_set_spectro(&mut c.surface_he, "surface_he", key, val)
        || try_set_spectro(&mut c.rcz, "rcz", key, val)
        || try_set_param(&mut c.mass, "mass", key, val)
        || try_set_param(&mut c.y, "y", key, val)
        || try_set_param(&mut c.feh_param, "feh", key, val)
        || try_set_param(&mut c.alpha, "alpha", key, val)
        || try_set_param(&mut c.f_ov, "f_ov", key, val)
    {
        return;
    }
    for k in 0..3 {
        let name = format!("my_param{}", k + 1);
        if try_set_param(&mut c.my_param[k], &name, key, val) {
            c.has_custom_param[k] = true;
            return;
        }
    }
    let f = parse_f64(val);
    let b = parse_bool(val);
    match key {
        "chi2_seismo_fraction" => {
            c.weights.chi2_seismo_fraction = f.unwrap_or(2.0 / 3.0);
        }
        "chi2_seismo_freq_fraction" => c.weights.chi2_seismo_freq_fraction = f,
        "chi2_seismo_delta_nu_fraction" => {
            c.weights.chi2_seismo_delta_nu_fraction = f.unwrap_or(0.0);
        }
        "chi2_seismo_nu_max_fraction" => {
            c.weights.chi2_seismo_nu_max_fraction = f.unwrap_or(0.0);
        }
        "chi2_seismo_r_010_fraction" => {
            c.weights.chi2_seismo_r_010_fraction = f.unwrap_or(0.0);
        }
        "chi2_seismo_r_02_fraction" => {
            c.weights.chi2_seismo_r_02_fraction = f.unwrap_or(0.0);
        }
        "normalize_chi2_seismo_frequencies" => {
            c.weights.normalize_chi2_seismo_frequencies = b;
        }
        "normalize_chi2_seismo_r_010" => {
            c.weights.normalize_chi2_seismo_r_010 = b;
        }
        "normalize_chi2_seismo_r_02" => {
            c.weights.normalize_chi2_seismo_r_02 = b;
        }
        "normalize_chi2_spectro" => c.weights.normalize_chi2_spectro = b,
        "trace_limits" => c.trace_limits = b,
        "z_div_x_solar" => c.z_div_x_solar = f.unwrap_or(0.02293),
        "nl0" | "nl1" | "nl2" | "nl3" => {
            c.nl[(key.as_bytes()[2] - b'0') as usize] =
                val.parse().unwrap_or(0);
        }
        "delta_nu" => c.delta_nu = f.unwrap_or(-1.0),
        "delta_nu_sigma" => c.delta_nu_sigma = f.unwrap_or(-1.0),
        "sigmas_coeff_for_delta_nu_limit" => {
            c.sigmas_coeff_for_delta_nu_limit = f.unwrap_or(0.0);
        }
        "nu_max" => c.nu_max = f.unwrap_or(-1.0),
        "nu_max_sigma" => c.nu_max_sigma = f.unwrap_or(-1.0),
        "oscillation_code" => c.oscillation_code = strip_quotes(val),
        "nu_lower_factor" => c.nu_lower_factor = f.unwrap_or(0.8),
        "nu_upper_factor" => c.nu_upper_factor = f.unwrap_or(1.2),
        "gyre_input_file" => c.gyre_input_file = strip_quotes(val),
        "gyre_non_ad" => c.gyre_non_ad = b,
        "add_atmosphere" => c.add_atmosphere = b,
        "keep_surface_point" => c.keep_surface_point = b,
        "add_center_point" => c.add_center_point = b,
        "correction_scheme" => c.correction_scheme = strip_quotes(val),
        "correction_b" => c.correction_b = f.unwrap_or(4.90),
        "correction_factor" => c.correction_factor = f.unwrap_or(0.0),
        "solar_nu_max" => c.solar_nu_max = f.unwrap_or(3090.0),
        "solar_delta_nu" => c.solar_delta_nu = f.unwrap_or(135.1),
        "search_type" => c.search_type = strip_quotes(val),
        "include_age_in_chi2_spectro" => c.include_age_in_chi2_spectro = b,
        "eval_chi2_at_target_age_only" => c.eval_chi2_at_target_age_only = b,
        "age_target" => c.age_target = f.unwrap_or(4.5695e9),
        "age_sigma" => c.age_sigma = f.unwrap_or(0.0065e9),
        "num_smaller_steps_before_age_target" => {
            c.num_smaller_steps_before_age_target = val.parse().unwrap_or(50);
        }
        "dt_for_smaller_steps_before_age_target" => {
            c.dt_for_smaller_steps_before_age_target = f.unwrap_or(6.5e5);
        }
        "min_age_for_chi2" => c.min_age_for_chi2 = f.filter(|&v| v > 0.0),
        "max_age_for_chi2" => c.max_age_for_chi2 = f.filter(|&v| v > 0.0),
        "min_age_limit" => c.min_age_limit = f.unwrap_or(1.0e6),
        "lnuc_div_l_limit" => c.lnuc_div_l_limit = f.unwrap_or(0.95),
        "chi2_spectroscopic_limit" => {
            c.chi2_spectroscopic_limit = f.unwrap_or(1000.0);
        }
        "chi2_delta_nu_limit" => c.chi2_delta_nu_limit = f.unwrap_or(1000.0),
        "chi2_radial_limit" => c.chi2_radial_limit = f.unwrap_or(100.0),
        "chi2_search_limit1" => c.chi2_search_limit1 = f.unwrap_or(3.0),
        "chi2_search_limit2" => c.chi2_search_limit2 = f.unwrap_or(4.0),
        "chi2_relative_increase_limit" => {
            c.chi2_relative_increase_limit = f.unwrap_or(2.0);
        }
        "limit_num_chi2_too_big" => {
            c.limit_num_chi2_too_big = val.parse().unwrap_or(20);
        }
        "y_depends_on_z" => c.y_depends_on_z = b,
        "y0" => c.y0 = f.unwrap_or(0.248),
        "dydz" => c.dydz = f.unwrap_or(1.4),
        "f0_ov_div_f_ov" => c.f0_ov_div_f_ov = f.unwrap_or(1.0),
        "y_frac_he3" => c.y_frac_he3 = f.unwrap_or(1.0e-4),
        "simplex_itermax" => c.simplex_itermax = val.parse().unwrap_or(1000),
        "simplex_fcn_calls_max" => {
            c.simplex_fcn_calls_max = val.parse().unwrap_or(1000);
        }
        "simplex_x_atol" => c.simplex_x_atol = f.unwrap_or(1.0e-10),
        "simplex_x_rtol" => c.simplex_x_rtol = f.unwrap_or(1.0e-10),
        "simplex_chi2_tol" => c.simplex_chi2_tol = f.unwrap_or(1.0e-10),
        "simplex_centroid_weight_power" => {
            c.simplex_centroid_weight_power = f.unwrap_or(0.0);
        }
        "simplex_enforce_bounds" => c.simplex_enforce_bounds = b,
        "simplex_adaptive_random_search" => {
            c.simplex_adaptive_random_search = b;
        }
        "simplex_seed" => c.simplex_seed = val.parse().unwrap_or(1_074_698_122),
        "simplex_output_filename" => {
            c.simplex_output_filename = strip_quotes(val);
        }
        "bobyqa_rho_beg" => c.bobyqa_rho_beg = f.unwrap_or(0.45),
        "bobyqa_rho_end" | "bobyqa_rhoend" => {
            c.bobyqa_rho_end = f.unwrap_or(1.0e-6);
        }
        "bobyqa_max_fcn_calls" => {
            c.bobyqa_max_fcn_calls = val.parse().unwrap_or(1000);
        }
        "bobyqa_output_filename" => {
            c.bobyqa_output_filename = strip_quotes(val);
        }
        "newuoa_rho_beg" => c.newuoa_rho_beg = f.unwrap_or(0.45),
        "newuoa_rho_end" | "newuoa_rhoend" => {
            c.newuoa_rho_end = f.unwrap_or(1.0e-6);
        }
        "newuoa_max_fcn_calls" => {
            c.newuoa_max_fcn_calls = val.parse().unwrap_or(1000);
        }
        "newuoa_output_filename" => {
            c.newuoa_output_filename = strip_quotes(val);
        }
        "astero_results_directory" => {
            c.astero_results_directory = strip_quotes(val);
        }
        "scan_grid_output_filename" => {
            c.scan_grid_output_filename = strip_quotes(val);
        }
        "filename_for_parameters" => {
            c.filename_for_parameters = strip_quotes(val);
        }
        "max_num_from_file" => c.max_num_from_file = val.parse().unwrap_or(-1),
        "file_column_for_mass" => {
            c.file_column_for_mass = val.parse().unwrap_or(3);
        }
        "file_column_for_y" => c.file_column_for_y = val.parse().unwrap_or(4),
        "file_column_for_feh" => {
            c.file_column_for_feh = val.parse().unwrap_or(5);
        }
        "file_column_for_alpha" => {
            c.file_column_for_alpha = val.parse().unwrap_or(6);
        }
        "file_column_for_f_ov" => {
            c.file_column_for_f_ov = val.parse().unwrap_or(7);
        }
        "file_column_for_my_param1" => {
            c.file_column_for_my_param[0] = val.parse().unwrap_or(8);
        }
        "file_column_for_my_param2" => {
            c.file_column_for_my_param[1] = val.parse().unwrap_or(9);
        }
        "file_column_for_my_param3" => {
            c.file_column_for_my_param[2] = val.parse().unwrap_or(10);
        }
        "from_file_output_filename" => {
            c.from_file_output_filename = strip_quotes(val);
        }
        "max_yrs_dt_when_cold" => c.max_yrs_dt_when_cold = f.unwrap_or(1.0e8),
        "max_yrs_dt_when_warm" => c.max_yrs_dt_when_warm = f.unwrap_or(1.0e7),
        "max_yrs_dt_when_hot" => c.max_yrs_dt_when_hot = f.unwrap_or(1.0e6),
        "chi2_limit_for_small_timesteps" => {
            c.chi2_limit_for_small_timesteps = f.unwrap_or(-1.0);
        }
        "max_yrs_dt_chi2_small_limit" => {
            c.max_yrs_dt_chi2_small_limit = f.unwrap_or(3.0e5);
        }
        "sigmas_coeff_for_hot_timesteps" => {
            c.sigmas_coeff_for_hot_timesteps = f.unwrap_or(2.0);
        }
        "sigmas_coeff_for_warm_timesteps" => {
            c.sigmas_coeff_for_warm_timesteps = f.unwrap_or(5.0);
        }
        _ => {}
    }
}
