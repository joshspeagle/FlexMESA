// SPDX-License-Identifier: GPL-3.0-or-later
//! Optimization drivers (`scan_grid`, `simplex`, `newuoa`, `bobyqa`),
//! output writers, surface-correction dispatch, and mode-matching helpers
//! extracted from `search_driver` and `sample_eval`.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use flexmesa_star::mesa_profile::{
    format_star_profile_text, resolve_star_profile_columns,
};
use flexmesa_star::pulse::default_fgong_header;
use flexmesa_star::pulse_bvp::BvpEigenmode;
use flexmesa_star::pulse_fgong::{get_fgong_data, PulseDataInput};
use flexmesa_star::pulse_gyre_data::star_get_pulse_data_gyre;
use flexmesa_star::pulse_write::{
    fortran_1pe, write_fgong_data, write_gyre_data,
};
use flexmesa_star::star_lib::star_write_model;
use flexmesa_star::{StarError, StarModel};

use crate::inlist_astero::{
    AsteroParamSet, AsteroSearchControls, AsteroSearchParam, MAX_CONSTRAINTS,
};
use crate::optimize::{
    try_bobyqa_minimize, try_newuoa_minimize, try_simplex_minimize_with_config,
};
use crate::sample_eval::AsteroSampleResult;
use crate::scaling::{eval_radial_delta_nu_ols, DELTA_NU_SUN, NU_MAX_SUN};
use crate::surface_corr::{
    estimate_l0_n_obs_with_numax, eval_combined_corr, eval_cubic_corr,
    eval_kjeldsen_corr, eval_power_law_corr, eval_sonoi_corr, eval_surf_coefs,
};

/// Grid points of `do_scan_grid` in upstream's visiting order
/// (`astero/private/astero_run_support.f90:366-531`).
///
/// `do_param(max_parameters)` recurses down to parameter 1, so the parameter
/// with the lowest `param_name` index varies fastest. A varied parameter
/// starts at `min_param` and steps by repeated addition of `delta_param`
/// while it is not above `max_param + 1d-6`; a fixed or unnamed one stays at
/// `first_param` for a single pass (`set_starting_values`, `do_param`).
pub(crate) fn build_scan_grid_params(
    ctrl: &AsteroSearchControls,
) -> Vec<AsteroParamSet> {
    const EPS: f64 = 1.0e-6; // `astero_run_support.f90:374`
    let n = ctrl.param_names.len();
    let named: Vec<bool> = ctrl
        .param_names
        .iter()
        .map(|s| !s.trim().is_empty())
        .collect();
    let mut param: Vec<f64> = (0..n)
        .map(|i| {
            let ps = ctrl.indexed_params[i];
            if !named[i] {
                0.0
            } else if ps.vary {
                ps.min
            } else {
                ps.first
            }
        })
        .collect();
    let mut out = Vec::new();
    scan_grid_do_param(ctrl, &named, n - 1, &mut param, &mut out, EPS);
    out
}

/// `recursive subroutine do_param(k, ierr)` with a 0-based `k`.
fn scan_grid_do_param(
    ctrl: &AsteroSearchControls,
    named: &[bool],
    k: usize,
    param: &mut [f64],
    out: &mut Vec<AsteroParamSet>,
    eps: f64,
) {
    let ps = ctrl.indexed_params[k];
    loop {
        if ps.vary && named[k] && param[k] > ps.max + eps {
            break;
        }
        if k == 0 {
            // `do1_grid`: evaluate this grid point.
            let mut p = ctrl.first_params();
            for (i, name) in ctrl.param_names.iter().enumerate() {
                if named[i] {
                    set_param_slot(&mut p, name.trim(), param[i]);
                }
            }
            out.push(p);
        } else {
            scan_grid_do_param(ctrl, named, k - 1, param, out, eps);
        }
        if ps.delta <= 0.0 || !ps.vary || !named[k] {
            break;
        }
        param[k] += ps.delta;
    }
    if ps.vary {
        param[k] = ps.min;
    }
}

/// Stores `value` in the [`AsteroParamSet`] slot that `param_name` selects
/// (the names accepted by the `&astero_search_controls` parser).
fn set_param_slot(p: &mut AsteroParamSet, name: &str, value: f64) {
    match name.to_ascii_lowercase().as_str() {
        "initial_mass" | "mass" => p.initial_mass = value,
        "initial_y" | "y" => p.initial_y = value,
        "initial_feh" | "feh" => p.initial_feh = value,
        "alpha" => p.alpha = value,
        "f_ov" => p.f_ov = value,
        "my_param1" | "semiconv" => p.my_param1 = value,
        "my_param2" | "thermohal" => p.my_param2 = value,
        "my_param3" | "diffusion" => p.my_param3 = value,
        // The parser rejects every other name.
        _ => {}
    }
}

fn active_search_params(
    ctrl: &AsteroSearchControls,
) -> Result<Vec<(String, AsteroSearchParam)>, StarError> {
    let mut out = Vec::new();
    if ctrl.param_names.iter().any(|s| !s.trim().is_empty()) {
        for (i, raw) in ctrl.param_names.iter().enumerate() {
            let name = raw.trim();
            if name.is_empty() {
                continue;
            }
            let p = ctrl.indexed_params[i];
            if p.vary {
                if p.min >= p.max {
                    return Err(StarError::InvalidInput(format!(
                        "min_param >= max_param for '{name}' ({} >= {})",
                        p.min, p.max
                    )));
                }
                out.push((String::from(name), p));
            }
        }
        return Ok(out);
    }
    let fallback = [
        ("initial_mass", ctrl.mass),
        ("initial_y", ctrl.y),
        ("initial_feh", ctrl.feh_param),
        ("alpha", ctrl.alpha),
        ("f_ov", ctrl.f_ov),
        ("my_param1", ctrl.my_param[0]),
        ("my_param2", ctrl.my_param[1]),
        ("my_param3", ctrl.my_param[2]),
    ];
    for (name, p) in fallback {
        if p.vary {
            if p.min >= p.max {
                return Err(StarError::InvalidInput(format!(
                    "min_param >= max_param for '{name}' ({} >= {})",
                    p.min, p.max
                )));
            }
            out.push((String::from(name), p));
        }
    }
    Ok(out)
}

fn apply_varied_slice(
    ctrl: &AsteroSearchControls,
    active: &[(String, AsteroSearchParam)],
    vals: &[f64],
) -> AsteroParamSet {
    let mut p = ctrl.first_params();
    for (i, raw) in ctrl.param_names.iter().enumerate() {
        let name = raw.trim();
        if !name.is_empty() {
            set_param_slot(&mut p, name, ctrl.indexed_params[i].first);
        }
    }
    for ((name, _), &v) in active.iter().zip(vals.iter()) {
        set_param_slot(&mut p, name, v);
    }
    p
}

pub(crate) fn extract_simplex_restart(
    ctrl: &AsteroSearchControls,
    samples: &[AsteroSampleResult],
) -> Result<(Vec<AsteroSampleResult>, Vec<Vec<f64>>, Vec<f64>), StarError> {
    let active = active_search_params(ctrl)?;
    let nvar = active.len();
    if nvar == 0 {
        return Err(StarError::InvalidInput(String::from(
            "restart_simplex_from_file requires at least one varied parameter",
        )));
    }
    if samples.len() < nvar + 1 {
        return Err(StarError::InvalidInput(format!(
            "restart_simplex_from_file requires at least {} samples, found {}",
            nvar + 1,
            samples.len()
        )));
    }
    let mut order: Vec<usize> = (0..samples.len()).collect();
    order.sort_by(|&a, &b| samples[a].chi2.total_cmp(&samples[b].chi2));
    let mut max_sample_num = 0usize;
    let mut vertices = Vec::with_capacity(nvar + 1);
    let mut f_vals = Vec::with_capacity(nvar + 1);
    for &idx in &order[..=nvar] {
        let s = &samples[idx];
        max_sample_num = max_sample_num.max(s.sample_number);
        f_vals.push(s.chi2);
        let row: Vec<f64> = active
            .iter()
            .map(|(name, _)| param_slot_value(&s.params, name).unwrap_or(0.0))
            .collect();
        vertices.push(row);
    }
    let kept: Vec<AsteroSampleResult> = samples
        .iter()
        .filter(|s| s.sample_number <= max_sample_num)
        .cloned()
        .collect();
    Ok((kept, vertices, f_vals))
}

pub(crate) fn run_simplex_driver(
    ctrl: &AsteroSearchControls,
    initial_simplex: Option<(Vec<Vec<f64>>, Vec<f64>)>,
    eval_fn: &mut impl FnMut(AsteroParamSet) -> Result<f64, StarError>,
) -> Result<(), StarError> {
    let active = active_search_params(ctrl)?;
    if active.is_empty() {
        let _ = eval_fn(ctrl.first_params())?;
        return Ok(());
    }
    let x_first: Vec<f64> = active.iter().map(|(_, p)| p.first).collect();
    let x_lower: Vec<f64> = active.iter().map(|(_, p)| p.min).collect();
    let x_upper: Vec<f64> = active.iter().map(|(_, p)| p.max).collect();
    let max_iters = ctrl.simplex_itermax.max(1);
    let max_evals = ctrl.simplex_fcn_calls_max.max(1);
    let _ = try_simplex_minimize_with_config(
        &x_first,
        &x_lower,
        &x_upper,
        max_iters,
        max_evals,
        ctrl.simplex_x_atol,
        ctrl.simplex_x_rtol,
        ctrl.simplex_chi2_tol,
        ctrl.simplex_centroid_weight_power,
        ctrl.simplex_enforce_bounds,
        ctrl.simplex_adaptive_random_search,
        ctrl.simplex_seed,
        ctrl.simplex_alpha,
        ctrl.simplex_beta,
        ctrl.simplex_gamma,
        ctrl.simplex_delta,
        initial_simplex,
        |x| eval_fn(apply_varied_slice(ctrl, &active, x)),
    )?;
    Ok(())
}

pub(crate) fn run_powell_driver(
    ctrl: &AsteroSearchControls,
    bounded: bool,
    eval_fn: &mut impl FnMut(AsteroParamSet) -> Result<f64, StarError>,
) -> Result<(), StarError> {
    let active = active_search_params(ctrl)?;
    if active.is_empty() {
        let _ = eval_fn(ctrl.first_params())?;
        return Ok(());
    }
    let x_first: Vec<f64> = active.iter().map(|(_, p)| p.first).collect();
    let x_lower: Vec<f64> = active.iter().map(|(_, p)| p.min).collect();
    let x_upper: Vec<f64> = active.iter().map(|(_, p)| p.max).collect();
    if bounded {
        let max_calls = ctrl.bobyqa_max_fcn_calls.max(1);
        let _ = try_bobyqa_minimize(
            &x_first,
            &x_lower,
            &x_upper,
            max_calls,
            ctrl.bobyqa_rho_beg,
            ctrl.bobyqa_rho_end,
            |x| eval_fn(apply_varied_slice(ctrl, &active, x)),
        )?;
    } else {
        let max_calls = ctrl.newuoa_max_fcn_calls.max(1);
        let _ = try_newuoa_minimize(
            &x_first,
            &x_lower,
            &x_upper,
            max_calls,
            ctrl.newuoa_rho_beg,
            ctrl.newuoa_rho_end,
            |x| eval_fn(apply_varied_slice(ctrl, &active, x)),
        )?;
    }
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn write_sample_file(
    out_dir: &Path,
    s: &AsteroSampleResult,
) -> Result<(), StarError> {
    let fname = out_dir.join(format!("sample_{:04}.data", s.sample_number));
    let mut buf = String::new();
    let _ = writeln!(
        buf,
        "sample_number {}\nchi2 {:26.16e}\nchi2_seismo {:26.16e}\nchi2_spectro {:26.16e}",
        s.sample_number, s.chi2, s.chi2_seismo, s.chi2_spectro
    );
    let _ = writeln!(
        buf,
        "age {:26.16e}\ninitial_mass {:26.16e}\ninitial_Y {:26.16e}\ninitial_FeH {:26.16e}",
        s.age_yr, s.params.initial_mass, s.params.initial_y, s.params.initial_feh
    );
    let _ = writeln!(
        buf,
        "alpha {:26.16e}\ndelta_nu {:26.16e}\nnu_max {:26.16e}",
        s.params.alpha, s.delta_nu_uhz, s.nu_max_uhz
    );
    fs::write(&fname, buf).map_err(|e| {
        StarError::IoError(format!(
            "failed writing sample file '{}': {e}",
            fname.display()
        ))
    })
}

/// Formats `show_best` diagnostic summary matching `astero/public/astero_def.f90:1338-1503`.
pub(crate) fn format_show_best_report(
    ctrl: &AsteroSearchControls,
    s: &AsteroSampleResult,
) -> String {
    let mut out = String::new();
    if ctrl.weights.chi2_seismo_fraction > 0.0 {
        for l in 0..4 {
            if ctrl.nl[l] > 0 && !s.model_freqs[l].is_empty() {
                let h1 = format!("l={l}");
                let h2 = format!("l{l}_freq");
                let h3 = format!("l{l}_corr");
                let h4 = format!("l{l}_obs");
                let h5 = format!("l{l}_sigma");
                let _ = writeln!(
                    out,
                    "\n{:>6}{:>6}{:>20}{:>20}{:>20}{:>20}{:>20}{:>20}",
                    h1, "n", "chi2term", h2, h3, h4, h5, "log E"
                );
                for i in 0..ctrl.nl[l].min(s.model_freqs[l].len()) {
                    let f_obs =
                        ctrl.freq_target[l].get(i).copied().unwrap_or(-1.0);
                    let f_sig =
                        ctrl.freq_sigma[l].get(i).copied().unwrap_or(1.0);
                    if f_obs < 0.0 || f_sig <= 0.0 {
                        continue;
                    }
                    let f_m = s.model_freqs[l][i];
                    let f_c =
                        s.model_freqs_corr[l].get(i).copied().unwrap_or(f_m);
                    let e =
                        s.model_inertias[l].get(i).copied().unwrap_or(1.0e-30);
                    let ord = s.model_orders[l].get(i).copied().unwrap_or(0);
                    let chi2term = flexmesa_math::pow2((f_c - f_obs) / f_sig);
                    let log_e = flexmesa_math::log10(e.max(1.0e-99));
                    let _ = writeln!(
                        out,
                        "      {:6}{}{:20.10}{:20.10}{:20.10}{:20.10}{:20.10}",
                        ord,
                        fortran_1pe(chi2term, 20, 10, None),
                        f_m,
                        f_c,
                        f_obs,
                        f_sig,
                        log_e
                    );
                }
            }
        }
        if ctrl.weights.chi2_seismo_r_010_fraction > 0.0 && s.ratios_n > 0 {
            let _ = writeln!(
                out,
                "\n{:>6}{:>6}{:>20}{:>20}{:>20}{:>20}{:>20}",
                "r01",
                "l=0 n",
                "chi2term",
                "r01",
                "r01_obs",
                "r01_sigma",
                "l0_obs"
            );
            for k in 0..s.ratios_n.min(s.model_r01.len()) {
                let sig = s.obs_r01_sigmas[k];
                if sig <= 0.0 {
                    continue;
                }
                let chi2term =
                    flexmesa_math::pow2((s.model_r01[k] - s.obs_r01[k]) / sig);
                let idx0 = s.ratios_l0_first + 1 + k;
                let ord0 = s.model_orders[0].get(idx0).copied().unwrap_or(0);
                let f0_obs =
                    ctrl.freq_target[0].get(idx0).copied().unwrap_or(0.0);
                let _ = writeln!(
                    out,
                    "      {:6}{}{:20.10}{:20.10}{:20.10}{:20.10}",
                    ord0,
                    fortran_1pe(chi2term, 20, 10, None),
                    s.model_r01[k],
                    s.obs_r01[k],
                    sig,
                    f0_obs
                );
            }
            let _ = writeln!(
                out,
                "\n{:>6}{:>6}{:>20}{:>20}{:>20}{:>20}{:>20}",
                "r10",
                "l=1 n",
                "chi2term",
                "r10",
                "r10_obs",
                "r10_sigma",
                "l1_obs"
            );
            for k in 0..s.ratios_n.min(s.model_r10.len()) {
                let sig = s.obs_r10_sigmas[k];
                if sig <= 0.0 {
                    continue;
                }
                let chi2term =
                    flexmesa_math::pow2((s.model_r10[k] - s.obs_r10[k]) / sig);
                let idx1 = s.ratios_l1_first + 1 + k;
                let ord1 = s.model_orders[1].get(idx1).copied().unwrap_or(0);
                let f1_obs =
                    ctrl.freq_target[1].get(idx1).copied().unwrap_or(0.0);
                let _ = writeln!(
                    out,
                    "      {:6}{}{:20.10}{:20.10}{:20.10}{:20.10}",
                    ord1,
                    fortran_1pe(chi2term, 20, 10, None),
                    s.model_r10[k],
                    s.obs_r10[k],
                    sig,
                    f1_obs
                );
            }
        }
        if ctrl.weights.chi2_seismo_r_02_fraction > 0.0
            && !s.obs_r02_sigmas.is_empty()
        {
            let _ = writeln!(
                out,
                "\n{:>6}{:>6}{:>20}{:>20}{:>20}{:>20}{:>20}",
                "r02",
                "l=0 n",
                "chi2term",
                "r02",
                "r02_obs",
                "r02_sigma",
                "l0_obs"
            );
            for k in 0..s.obs_r02_sigmas.len() {
                let sig = s.obs_r02_sigmas[k];
                if sig <= 0.0 {
                    continue;
                }
                let chi2term =
                    flexmesa_math::pow2((s.model_r02[k] - s.obs_r02[k]) / sig);
                let idx0 = s.ratios_l0_first + 1 + k;
                let ord0 = s.model_orders[0].get(idx0).copied().unwrap_or(0);
                let f0_obs =
                    ctrl.freq_target[0].get(idx0).copied().unwrap_or(0.0);
                let _ = writeln!(
                    out,
                    "      {:6}{}{:20.10}{:20.10}{:20.10}{:20.10}",
                    ord0,
                    fortran_1pe(chi2term, 20, 10, None),
                    s.model_r02[k],
                    s.obs_r02[k],
                    sig,
                    f0_obs
                );
            }
        }
    }
    let write1 = |buf: &mut String, label: &str, x: f64| {
        if x.abs() < 1.0e6 {
            let _ = writeln!(buf, "{label:>40}{x:20.10}");
        } else {
            let _ = writeln!(buf, "{label:>40}{x:20.10e}");
        }
    };
    if ctrl.age_sigma > 0.0 && ctrl.include_age_in_chi2_spectro {
        let chi2term =
            flexmesa_math::pow2((s.age_yr - ctrl.age_target) / ctrl.age_sigma);
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "{:>40}{}",
            "age",
            fortran_1pe(s.age_yr, 20, 10, None)
        );
        let _ = writeln!(
            out,
            "{:>40}{}",
            "age_target",
            fortran_1pe(ctrl.age_target, 20, 10, None)
        );
        let _ = writeln!(
            out,
            "{:>40}{}",
            "age_sigma",
            fortran_1pe(ctrl.age_sigma, 20, 10, None)
        );
        let _ = writeln!(
            out,
            "{:>40}{}",
            "age chi2term",
            fortran_1pe(chi2term, 20, 10, None)
        );
    }
    for (i, c) in ctrl.indexed_constraints.iter().enumerate() {
        let name = ctrl.constraint_names[i].trim();
        if name.is_empty() {
            continue;
        }
        let _ = writeln!(out);
        write1(&mut out, name, s.constraint_values[i]);
        if c.sigma > 0.0 && c.include {
            let chi2term = flexmesa_math::pow2(
                (s.constraint_values[i] - c.target) / c.sigma,
            );
            write1(&mut out, &format!("{name}_obs"), c.target);
            write1(&mut out, &format!("{name}_sigma"), c.sigma);
            write1(&mut out, &format!("{name} chi2term"), chi2term);
        }
    }
    let _ = writeln!(out);
    write1(&mut out, "delta_nu", s.delta_nu_uhz);
    write1(&mut out, "nu_max", s.nu_max_uhz);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{:>40}{}",
        ctrl.surf_coef1_name.trim(),
        fortran_1pe(s.surf_coef1, 20, 10, None)
    );
    let _ = writeln!(
        out,
        "{:>40}{}",
        ctrl.surf_coef2_name.trim(),
        fortran_1pe(s.surf_coef2, 20, 10, None)
    );
    let _ = writeln!(out);
    for raw in &ctrl.param_names {
        let pname = raw.trim();
        if pname.is_empty() {
            continue;
        }
        let val = param_slot_value(&s.params, pname).unwrap_or(0.0);
        write1(&mut out, pname, val);
    }
    let _ =
        writeln!(out, "{:>40}{}", "age", fortran_1pe(s.age_yr, 20, 10, None));
    let _ = writeln!(out);
    if (ctrl.weights.chi2_seismo_fraction - 1.0).abs() < 1.0e-15 {
        write1(&mut out, "chi^2 seismo", s.chi2_seismo);
    } else if ctrl.weights.chi2_seismo_fraction <= 0.0 {
        write1(&mut out, "chi^2 spectro", s.chi2_spectro);
    } else {
        write1(&mut out, "chi^2 combined", s.chi2);
        write1(
            &mut out,
            "chi2_seismo_fraction",
            ctrl.weights.chi2_seismo_fraction,
        );
        write1(&mut out, "chi^2 seismo", s.chi2_seismo);
        write1(&mut out, "chi^2 spectro", s.chi2_spectro);
    }
    let _ = writeln!(out, "\n{:>40}{:16}\n\n", "model number", s.model_number);
    out
}

pub(crate) fn write_best_sample_file(
    out_dir: &Path,
    ctrl: &AsteroSearchControls,
    s: &AsteroSampleResult,
) -> Result<(), StarError> {
    fs::create_dir_all(out_dir).map_err(|e| {
        StarError::IoError(format!(
            "failed creating astero_results_directory '{}': {e}",
            out_dir.display()
        ))
    })?;
    let num_str =
        format!("{:0width$}", s.sample_number, width = ctrl.num_digits);
    let fname = out_dir.join(format!(
        "{}{}{}",
        ctrl.sample_results_prefix.trim(),
        num_str,
        ctrl.sample_results_postfix.trim()
    ));
    let body = format_show_best_report(ctrl, s);
    fs::write(&fname, body).map_err(|e| {
        StarError::IoError(format!(
            "failed writing sample file '{}': {e}",
            fname.display()
        ))
    })
}

pub(crate) fn write_best_model_artifacts(
    out_dir: &Path,
    ctrl: &AsteroSearchControls,
    model: &StarModel,
    case_dir: &Path,
) -> Result<(), StarError> {
    if !(ctrl.save_model_for_best_model
        || ctrl.write_fgong_for_best_model
        || ctrl.write_gyre_for_best_model
        || ctrl.write_profile_for_best_model)
    {
        return Ok(());
    }
    fs::create_dir_all(out_dir).map_err(|e| {
        StarError::IoError(format!(
            "failed creating astero_results_directory '{}': {e}",
            out_dir.display()
        ))
    })?;
    if ctrl.save_model_for_best_model {
        let p = out_dir.join(ctrl.best_model_save_model_filename.trim());
        star_write_model(model, &p)?;
    }
    if ctrl.write_fgong_for_best_model {
        let pulse_in = PulseDataInput::from_model(model)
            .map_err(|e| StarError::InvalidInput(e.to_string()))?;
        let (gd, pd) = get_fgong_data(
            &pulse_in,
            ctrl.add_center_point,
            ctrl.keep_surface_point,
            ctrl.add_atmosphere,
        )
        .map_err(|e| StarError::InvalidInput(e.to_string()))?;
        let txt = write_fgong_data(
            &gd,
            &pd,
            &default_fgong_header(),
            model.controls.fgong_ivers,
        )
        .map_err(|e| StarError::InvalidInput(e.to_string()))?;
        let p = out_dir.join(ctrl.best_model_fgong_filename.trim());
        fs::write(&p, txt).map_err(|e| {
            StarError::IoError(format!(
                "failed writing best FGONG '{}': {e}",
                p.display()
            ))
        })?;
    }
    if ctrl.write_gyre_for_best_model {
        let (gd, pd) = star_get_pulse_data_gyre(
            model,
            ctrl.add_center_point,
            ctrl.keep_surface_point,
            ctrl.add_atmosphere,
        )
        .map_err(|e| StarError::InvalidInput(e.to_string()))?;
        let txt = write_gyre_data(&gd, &pd, model.controls.gyre_data_schema)
            .map_err(|e| StarError::InvalidInput(e.to_string()))?;
        let p = out_dir.join(ctrl.best_model_gyre_filename.trim());
        fs::write(&p, txt).map_err(|e| {
            StarError::IoError(format!(
                "failed writing best GYRE '{}': {e}",
                p.display()
            ))
        })?;
    }
    if ctrl.write_profile_for_best_model {
        let specs = resolve_star_profile_columns(model, Some(case_dir))?;
        let txt = format_star_profile_text(model, &specs);
        let p = out_dir.join(ctrl.best_model_profile_filename.trim());
        fs::write(&p, txt).map_err(|e| {
            StarError::IoError(format!(
                "failed writing best profile '{}': {e}",
                p.display()
            ))
        })?;
    }
    Ok(())
}

fn param_slot_value(p: &AsteroParamSet, name: &str) -> Option<f64> {
    Some(match name.trim().to_ascii_lowercase().as_str() {
        "initial_mass" | "mass" => p.initial_mass,
        "initial_y" | "y" => p.initial_y,
        "initial_feh" | "feh" => p.initial_feh,
        "alpha" => p.alpha,
        "f_ov" => p.f_ov,
        "my_param1" | "semiconv" => p.my_param1,
        "my_param2" | "thermohal" => p.my_param2,
        "my_param3" | "diffusion" => p.my_param3,
        _ => return None,
    })
}

fn parse_width_and_prec(
    fmt: &str,
    default_w: usize,
    default_d: usize,
) -> (char, usize, usize) {
    let inner = fmt
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim()
        .to_ascii_lowercase();
    let stripped = inner
        .strip_prefix("1p")
        .or_else(|| inner.strip_prefix("0p"))
        .unwrap_or(&inner);
    let (kind, rest) = if let Some(r) = stripped.strip_prefix("es") {
        ('e', r)
    } else if let Some(r) = stripped.strip_prefix("en") {
        ('e', r)
    } else if let Some(first) = stripped.chars().next() {
        (first, &stripped[first.len_utf8()..])
    } else {
        ('e', "")
    };
    let (w, d) = match rest.split_once('.') {
        Some((ws, ds)) => (
            ws.parse::<usize>().unwrap_or(default_w),
            ds.parse::<usize>().unwrap_or(default_d),
        ),
        None => (rest.parse::<usize>().unwrap_or(default_w), default_d),
    };
    (kind, w.max(1), d)
}

fn fmt_int_field(v: usize, fmt: &str) -> String {
    let (_, w, _) = parse_width_and_prec(fmt, 26, 0);
    format!("{v:>w$}")
}

fn fmt_isize_field(v: isize, fmt: &str) -> String {
    let (_, w, _) = parse_width_and_prec(fmt, 26, 0);
    format!("{v:>w$}")
}

fn fmt_txt_field(v: &str, fmt: &str) -> String {
    let (_, w, _) = parse_width_and_prec(fmt, 26, 0);
    format!("{v:>w$}")
}

fn fmt_dbl_field(v: f64, fmt: &str) -> String {
    let (kind, w, d) = parse_width_and_prec(fmt, 26, 16);
    match kind {
        'f' => format!("{v:>w$.d$}"),
        'g' => format!("{v:>w$.d$e}"),
        _ => fortran_1pe(v, w, d, None),
    }
}

/// Writes MESA-format `show_all_sample_results` search table
/// (`astero/public/astero_def.f90:1040-1335`).
pub(crate) fn write_search_results_table(
    out_path: &Path,
    ctrl: &AsteroSearchControls,
    samples: &[AsteroSampleResult],
    total_grid_points: Option<usize>,
) -> Result<(), StarError> {
    let mut order: Vec<usize> = (0..samples.len()).collect();
    order.sort_by(|&a, &b| samples[a].chi2.total_cmp(&samples[b].chi2));

    let dfmt = &ctrl.astero_results_dbl_format;
    let ifmt = &ctrl.astero_results_int_format;
    let tfmt = &ctrl.astero_results_txt_format;

    let mut meta: Vec<(&str, String, bool)> = Vec::new();
    meta.push(("samples", fmt_int_field(samples.len(), ifmt), true));
    if let Some(tot) = total_grid_points.filter(|&t| t > 0) {
        meta.push(("total", fmt_int_field(tot, ifmt), true));
    }
    meta.push(("version_number", fmt_txt_field("\"flexmesa\"", tfmt), false));
    meta.push(("compiler", fmt_txt_field("\"rustc\"", tfmt), false));
    meta.push(("build", fmt_txt_field("\"safe-rust\"", tfmt), false));
    meta.push(("MESA_SDK_version", fmt_txt_field("\"none\"", tfmt), false));
    meta.push(("math_backend", fmt_txt_field("\"flexmesa\"", tfmt), false));
    meta.push(("date", fmt_txt_field("\"26.4.1\"", tfmt), false));
    meta.push((
        "search_type",
        fmt_txt_field(&format!("\"{}\"", ctrl.search_type.trim()), tfmt),
        false,
    ));

    let mut buf = String::new();
    for line_idx in 0..3 {
        for (col_idx, (name, val, _)) in meta.iter().enumerate() {
            match line_idx {
                0 => buf.push_str(&fmt_int_field(col_idx + 1, ifmt)),
                1 => buf.push_str(&fmt_txt_field(name, tfmt)),
                _ => buf.push_str(val),
            }
        }
        buf.push('\n');
    }
    buf.push('\n');

    let mut headers: Vec<String> = Vec::new();
    headers.push(String::from("sample"));
    headers.push(String::from("chi2"));
    for raw in &ctrl.param_names {
        let p = raw.trim();
        if !p.is_empty() {
            headers.push(String::from(p));
        }
    }
    headers.push(String::from("age"));
    headers.push(String::from("model_number"));
    for raw in &ctrl.constraint_names {
        let c = raw.trim();
        if !c.is_empty() {
            headers.push(String::from(c));
        }
    }
    headers.push(String::from("delta_nu"));
    headers.push(String::from("nu_max"));
    headers.push(String::from(ctrl.surf_coef1_name.trim()));
    headers.push(String::from(ctrl.surf_coef2_name.trim()));
    headers.push(String::from("chi2_seismo"));
    headers.push(String::from("chi2_spectro"));
    headers.push(String::from("nl0"));
    headers.push(String::from("nl1"));
    headers.push(String::from("nl2"));
    headers.push(String::from("nl3"));
    headers.push(String::from("ratios_n"));
    headers.push(String::from("ratios_l0_first"));
    headers.push(String::from("ratios_l1_first"));

    let include_seismo_cols = ctrl.weights.chi2_seismo_fraction > 0.0;
    let max_ratios_n = samples.iter().map(|s| s.ratios_n).max().unwrap_or(0);
    if include_seismo_cols {
        for l in 0..4 {
            for j in 1..=ctrl.nl[l] {
                headers.push(format!("l{l}_order_{j}"));
                headers.push(format!("l{l}_obs_{j}"));
                headers.push(format!("l{l}_obs_sigma_{j}"));
                headers.push(format!("l{l}_freq_{j}"));
                headers.push(format!("l{l}_freq_corr_{j}"));
                headers.push(format!("l{l}_inertia_{j}"));
            }
        }
        for j in 1..=max_ratios_n {
            headers.push(format!("r01_obs_{j}"));
            headers.push(format!("r01_obs_sigmas_{j}"));
            headers.push(format!("r01_{j}"));
            headers.push(format!("r10_obs_{j}"));
            headers.push(format!("r10_obs_sigmas_{j}"));
            headers.push(format!("r10_{j}"));
        }
        if ctrl.weights.chi2_seismo_r_02_fraction > 0.0 {
            for j in 1..=ctrl.nl[0] {
                headers.push(format!("r02_obs_{j}"));
                headers.push(format!("r02_obs_sigmas_{j}"));
                headers.push(format!("r02_{j}"));
            }
        }
    }
    let is_simplex = ctrl.search_type.trim().eq_ignore_ascii_case("simplex");
    if is_simplex {
        headers.push(String::from("step_type"));
    }

    for i in 1..=headers.len() {
        buf.push_str(&fmt_int_field(i, ifmt));
    }
    buf.push('\n');
    for h in &headers {
        buf.push_str(&fmt_txt_field(h, tfmt));
    }
    buf.push('\n');

    for &idx in &order {
        let s = &samples[idx];
        buf.push_str(&fmt_int_field(s.sample_number, ifmt));
        buf.push_str(&fmt_dbl_field(s.chi2, dfmt));
        for raw in &ctrl.param_names {
            let pname = raw.trim();
            if !pname.is_empty() {
                let v = param_slot_value(&s.params, pname).unwrap_or(0.0);
                buf.push_str(&fmt_dbl_field(v, dfmt));
            }
        }
        buf.push_str(&fmt_dbl_field(s.age_yr, dfmt));
        buf.push_str(&fmt_int_field(s.model_number, ifmt));
        for (c_idx, raw) in ctrl.constraint_names.iter().enumerate() {
            if !raw.trim().is_empty() {
                buf.push_str(&fmt_dbl_field(s.constraint_values[c_idx], dfmt));
            }
        }
        buf.push_str(&fmt_dbl_field(s.delta_nu_uhz, dfmt));
        buf.push_str(&fmt_dbl_field(s.nu_max_uhz, dfmt));
        buf.push_str(&fmt_dbl_field(s.surf_coef1, dfmt));
        buf.push_str(&fmt_dbl_field(s.surf_coef2, dfmt));
        buf.push_str(&fmt_dbl_field(s.chi2_seismo, dfmt));
        buf.push_str(&fmt_dbl_field(s.chi2_spectro, dfmt));
        for &n in &ctrl.nl {
            buf.push_str(&fmt_int_field(n, ifmt));
        }
        buf.push_str(&fmt_int_field(s.ratios_n, ifmt));
        buf.push_str(&fmt_int_field(s.ratios_l0_first, ifmt));
        buf.push_str(&fmt_int_field(s.ratios_l1_first, ifmt));
        if include_seismo_cols {
            for l in 0..4 {
                for j in 0..ctrl.nl[l] {
                    let ord = s.model_orders[l].get(j).copied().unwrap_or(0);
                    let f_o =
                        ctrl.freq_target[l].get(j).copied().unwrap_or(0.0);
                    let f_s = ctrl.freq_sigma[l].get(j).copied().unwrap_or(0.0);
                    let f_m = s.model_freqs[l].get(j).copied().unwrap_or(0.0);
                    let f_c =
                        s.model_freqs_corr[l].get(j).copied().unwrap_or(0.0);
                    let ine =
                        s.model_inertias[l].get(j).copied().unwrap_or(0.0);
                    buf.push_str(&fmt_isize_field(ord, ifmt));
                    buf.push_str(&fmt_dbl_field(f_o, dfmt));
                    buf.push_str(&fmt_dbl_field(f_s, dfmt));
                    buf.push_str(&fmt_dbl_field(f_m, dfmt));
                    buf.push_str(&fmt_dbl_field(f_c, dfmt));
                    buf.push_str(&fmt_dbl_field(ine, dfmt));
                }
            }
            for j in 0..max_ratios_n {
                buf.push_str(&fmt_dbl_field(
                    s.obs_r01.get(j).copied().unwrap_or(0.0),
                    dfmt,
                ));
                buf.push_str(&fmt_dbl_field(
                    s.obs_r01_sigmas.get(j).copied().unwrap_or(0.0),
                    dfmt,
                ));
                buf.push_str(&fmt_dbl_field(
                    s.model_r01.get(j).copied().unwrap_or(0.0),
                    dfmt,
                ));
                buf.push_str(&fmt_dbl_field(
                    s.obs_r10.get(j).copied().unwrap_or(0.0),
                    dfmt,
                ));
                buf.push_str(&fmt_dbl_field(
                    s.obs_r10_sigmas.get(j).copied().unwrap_or(0.0),
                    dfmt,
                ));
                buf.push_str(&fmt_dbl_field(
                    s.model_r10.get(j).copied().unwrap_or(0.0),
                    dfmt,
                ));
            }
            if ctrl.weights.chi2_seismo_r_02_fraction > 0.0 {
                for j in 0..ctrl.nl[0] {
                    buf.push_str(&fmt_dbl_field(
                        s.obs_r02.get(j).copied().unwrap_or(0.0),
                        dfmt,
                    ));
                    buf.push_str(&fmt_dbl_field(
                        s.obs_r02_sigmas.get(j).copied().unwrap_or(0.0),
                        dfmt,
                    ));
                    buf.push_str(&fmt_dbl_field(
                        s.model_r02.get(j).copied().unwrap_or(0.0),
                        dfmt,
                    ));
                }
            }
        }
        if is_simplex {
            buf.push_str(&fmt_txt_field("", tfmt));
        }
        buf.push('\n');
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            StarError::IoError(format!(
                "failed creating output directory '{}': {e}",
                parent.display()
            ))
        })?;
    }
    fs::write(out_path, buf).map_err(|e| {
        StarError::IoError(format!(
            "failed writing search results '{}': {e}",
            out_path.display()
        ))
    })
}

/// Reads MESA-format sample results from file for `restart_scan_grid_from_file`
/// and `restart_simplex_from_file` (`astero/public/astero_def.f90:1506-1708`).
pub(crate) fn read_samples_from_file(
    path: &Path,
    ctrl: &AsteroSearchControls,
) -> Result<Vec<AsteroSampleResult>, StarError> {
    let text = fs::read_to_string(path).map_err(|e| {
        StarError::IoError(format!(
            "failed reading samples file '{}': {e}",
            path.display()
        ))
    })?;
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 6 {
        return Err(StarError::InvalidInput(format!(
            "samples file '{}' has fewer than 6 header lines",
            path.display()
        )));
    }
    let num_samples: usize = lines[2]
        .split_whitespace()
        .next()
        .ok_or_else(|| {
            StarError::InvalidInput(format!(
                "missing sample count on line 3 of '{}'",
                path.display()
            ))
        })?
        .parse()
        .map_err(|e| {
            StarError::InvalidInput(format!(
                "invalid sample count on line 3 of '{}': {e}",
                path.display()
            ))
        })?;

    let (_, iw, _) =
        parse_width_and_prec(&ctrl.astero_results_int_format, 26, 0);
    let (_, dw, _) =
        parse_width_and_prec(&ctrl.astero_results_dbl_format, 26, 16);

    let active_param_names: Vec<&str> = ctrl
        .param_names
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    let active_constraint_indices: Vec<usize> = ctrl
        .constraint_names
        .iter()
        .enumerate()
        .filter_map(|(i, s)| (!s.trim().is_empty()).then_some(i))
        .collect();

    let parse_f64_tok = |s: &str, col_name: &str| -> Result<f64, StarError> {
        let norm = s.trim().replace(['D', 'd'], "E");
        norm.parse::<f64>().map_err(|e| {
            StarError::InvalidInput(format!(
                "failed parsing float '{s}' for {col_name} in '{}': {e}",
                path.display()
            ))
        })
    };

    let header_toks: Vec<&str> = lines[5].split_whitespace().collect();
    let header_ratios_n = header_toks
        .iter()
        .filter(|h| {
            h.strip_prefix("r01_obs_").is_some_and(|rest| {
                !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit())
            })
        })
        .count();
    let header_r02_n = header_toks
        .iter()
        .filter(|h| {
            h.strip_prefix("r02_obs_").is_some_and(|rest| {
                !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit())
            })
        })
        .count();

    let mut samples = Vec::with_capacity(num_samples);
    for (row_idx, raw_line) in
        lines.iter().skip(6).take(num_samples).enumerate()
    {
        let toks = split_sample_row_tokens(
            raw_line,
            iw,
            dw,
            active_param_names.len(),
            active_constraint_indices.len(),
        )?;
        let mut cursor = 0usize;
        let next_tok = |c: &mut usize| -> Result<&str, StarError> {
            let t = toks.get(*c).ok_or_else(|| {
                StarError::InvalidInput(format!(
                    "truncated sample row {} in '{}'",
                    row_idx + 1,
                    path.display()
                ))
            })?;
            *c += 1;
            Ok(t.as_str())
        };
        let sample_number: usize =
            next_tok(&mut cursor)?.parse().map_err(|e| {
                StarError::InvalidInput(format!(
                    "invalid sample_number in '{}': {e}",
                    path.display()
                ))
            })?;
        let chi2 = parse_f64_tok(next_tok(&mut cursor)?, "chi2")?;
        let mut params = ctrl.first_params();
        for pname in &active_param_names {
            let val = parse_f64_tok(next_tok(&mut cursor)?, pname)?;
            set_param_slot(&mut params, pname, val);
        }
        let age_yr = parse_f64_tok(next_tok(&mut cursor)?, "age")?;
        let model_number: usize =
            next_tok(&mut cursor)?.parse().map_err(|e| {
                StarError::InvalidInput(format!(
                    "invalid model_number in '{}': {e}",
                    path.display()
                ))
            })?;
        let mut constraint_values = [0.0_f64; MAX_CONSTRAINTS];
        for &ci in &active_constraint_indices {
            constraint_values[ci] =
                parse_f64_tok(next_tok(&mut cursor)?, "constraint")?;
        }
        let delta_nu_uhz = parse_f64_tok(next_tok(&mut cursor)?, "delta_nu")?;
        let nu_max_uhz = parse_f64_tok(next_tok(&mut cursor)?, "nu_max")?;
        let surf_coef1 = parse_f64_tok(next_tok(&mut cursor)?, "surf_coef1")?;
        let surf_coef2 = parse_f64_tok(next_tok(&mut cursor)?, "surf_coef2")?;
        let chi2_seismo = parse_f64_tok(next_tok(&mut cursor)?, "chi2_seismo")?;
        let chi2_spectro =
            parse_f64_tok(next_tok(&mut cursor)?, "chi2_spectro")?;
        let mut nl_row = [0usize; 4];
        for slot in &mut nl_row {
            *slot = next_tok(&mut cursor)?.parse().unwrap_or(0);
        }
        let ratios_n: usize = next_tok(&mut cursor)?.parse().unwrap_or(0);
        let ratios_l0_first: usize =
            next_tok(&mut cursor)?.parse().unwrap_or(0);
        let ratios_l1_first: usize =
            next_tok(&mut cursor)?.parse().unwrap_or(0);

        let mut model_freqs: [Vec<f64>; 4] = Default::default();
        let mut model_freqs_corr: [Vec<f64>; 4] = Default::default();
        let mut model_inertias: [Vec<f64>; 4] = Default::default();
        let mut model_orders: [Vec<isize>; 4] = Default::default();
        let mut obs_r01 = Vec::new();
        let mut obs_r01_sigmas = Vec::new();
        let mut model_r01 = Vec::new();
        let mut obs_r10 = Vec::new();
        let mut obs_r10_sigmas = Vec::new();
        let mut model_r10 = Vec::new();
        let mut obs_r02 = Vec::new();
        let mut obs_r02_sigmas = Vec::new();
        let mut model_r02 = Vec::new();
        if ctrl.weights.chi2_seismo_fraction > 0.0
            && cursor + 6 * nl_row.iter().sum::<usize>() <= toks.len()
        {
            for l in 0..4 {
                for _ in 0..nl_row[l] {
                    let ord: isize =
                        next_tok(&mut cursor)?.parse().unwrap_or(0);
                    let _f_obs = parse_f64_tok(next_tok(&mut cursor)?, "obs")?;
                    let _f_sig = parse_f64_tok(next_tok(&mut cursor)?, "sig")?;
                    let f_m = parse_f64_tok(next_tok(&mut cursor)?, "freq")?;
                    let f_c =
                        parse_f64_tok(next_tok(&mut cursor)?, "freq_corr")?;
                    let ine = parse_f64_tok(next_tok(&mut cursor)?, "inertia")?;
                    model_orders[l].push(ord);
                    model_freqs[l].push(f_m);
                    model_freqs_corr[l].push(f_c);
                    model_inertias[l].push(ine);
                }
            }
            let ratio_cols = header_ratios_n.max(ratios_n);
            if ratio_cols > 0 && cursor + 6 * ratio_cols <= toks.len() {
                for j in 0..ratio_cols {
                    let r01o =
                        parse_f64_tok(next_tok(&mut cursor)?, "r01_obs")?;
                    let r01s =
                        parse_f64_tok(next_tok(&mut cursor)?, "r01_sig")?;
                    let r01m = parse_f64_tok(next_tok(&mut cursor)?, "r01")?;
                    let r10o =
                        parse_f64_tok(next_tok(&mut cursor)?, "r10_obs")?;
                    let r10s =
                        parse_f64_tok(next_tok(&mut cursor)?, "r10_sig")?;
                    let r10m = parse_f64_tok(next_tok(&mut cursor)?, "r10")?;
                    if j < ratios_n {
                        obs_r01.push(r01o);
                        obs_r01_sigmas.push(r01s);
                        model_r01.push(r01m);
                        obs_r10.push(r10o);
                        obs_r10_sigmas.push(r10s);
                        model_r10.push(r10m);
                    }
                }
            }
            let r02_cols = if ctrl.weights.chi2_seismo_r_02_fraction > 0.0 {
                header_r02_n.max(nl_row[0])
            } else {
                0
            };
            if r02_cols > 0 && cursor + 3 * r02_cols <= toks.len() {
                for j in 0..r02_cols {
                    let r02o =
                        parse_f64_tok(next_tok(&mut cursor)?, "r02_obs")?;
                    let r02s =
                        parse_f64_tok(next_tok(&mut cursor)?, "r02_sig")?;
                    let r02m = parse_f64_tok(next_tok(&mut cursor)?, "r02")?;
                    if j < nl_row[0] {
                        obs_r02.push(r02o);
                        obs_r02_sigmas.push(r02s);
                        model_r02.push(r02m);
                    }
                }
            }
        }
        samples.push(AsteroSampleResult {
            sample_number,
            params,
            model_number,
            age_yr,
            teff: 0.0,
            logg: 0.0,
            log_l: 0.0,
            log_r: 0.0,
            constraint_values,
            delta_nu_uhz,
            nu_max_uhz,
            a_div_r: if ctrl.correction_scheme.eq_ignore_ascii_case("kjeldsen")
            {
                surf_coef1
            } else {
                0.0
            },
            correction_r: if ctrl
                .correction_scheme
                .eq_ignore_ascii_case("kjeldsen")
            {
                surf_coef2
            } else {
                1.0
            },
            chi2,
            chi2_seismo,
            chi2_spectro,
            chi2_radial: 0.0,
            chi2_r010: 0.0,
            chi2_r02: 0.0,
            model_freqs,
            model_freqs_corr,
            model_inertias,
            model_orders,
            surf_coef1,
            surf_coef2,
            ratios_n,
            ratios_l0_first,
            ratios_l1_first,
            obs_r01,
            obs_r01_sigmas,
            model_r01,
            obs_r10,
            obs_r10_sigmas,
            model_r10,
            obs_r02,
            obs_r02_sigmas,
            model_r02,
            model_freqs_alt_up: Default::default(),
            model_freqs_corr_alt_up: Default::default(),
            model_inertias_alt_up: Default::default(),
            model_orders_alt_up: Default::default(),
            model_freqs_alt_down: Default::default(),
            model_freqs_corr_alt_down: Default::default(),
            model_inertias_alt_down: Default::default(),
            model_orders_alt_down: Default::default(),
        });
    }
    if samples.len() != num_samples {
        return Err(StarError::InvalidInput(format!(
            "expected {num_samples} samples in '{}', parsed {}",
            path.display(),
            samples.len()
        )));
    }
    Ok(samples)
}

fn split_sample_row_tokens(
    line: &str,
    iw: usize,
    dw: usize,
    n_params: usize,
    n_constraints: usize,
) -> Result<Vec<String>, StarError> {
    let ws: Vec<String> = line.split_whitespace().map(str::to_string).collect();
    let min_cols = 17 + n_params + n_constraints;
    if ws.len() >= min_cols {
        return Ok(ws);
    }
    let bytes = line.as_bytes();
    let mut out = Vec::new();
    let mut pos = 0usize;
    let take = |pos: &mut usize, w: usize, out: &mut Vec<String>| -> bool {
        if *pos + w > bytes.len() {
            return false;
        }
        let s = String::from_utf8_lossy(&bytes[*pos..*pos + w])
            .trim()
            .to_string();
        *pos += w;
        out.push(s);
        true
    };
    if !(take(&mut pos, iw, &mut out) && take(&mut pos, dw, &mut out)) {
        return Err(StarError::InvalidInput(String::from(
            "unable to parse sample results row",
        )));
    }
    for _ in 0..n_params {
        let _ = take(&mut pos, dw, &mut out);
    }
    let _ = take(&mut pos, dw, &mut out);
    let _ = take(&mut pos, iw, &mut out);
    for _ in 0..n_constraints {
        let _ = take(&mut pos, dw, &mut out);
    }
    for _ in 0..6 {
        let _ = take(&mut pos, dw, &mut out);
    }
    for _ in 0..7 {
        let _ = take(&mut pos, iw, &mut out);
    }
    Ok(out)
}

pub(crate) fn eval_radial_chi2(
    ctrl: &AsteroSearchControls,
    corr_l0: &[f64],
) -> f64 {
    let mut sum = 0.0_f64;
    let mut n = 0usize;
    for ((&obs, &sig), &mc) in ctrl.freq_target[0]
        .iter()
        .zip(ctrl.freq_sigma[0].iter())
        .zip(corr_l0.iter())
    {
        if obs >= 0.0 && sig > 0.0 {
            let d = (mc - obs) / sig;
            sum += d * d;
            n += 1;
        }
    }
    if ctrl.weights.normalize_chi2_seismo_frequencies && n > 0 {
        sum / (n as f64)
    } else {
        sum
    }
}

pub(crate) struct SurfaceCorrOutput {
    pub corrected: Vec<f64>,
    pub a_div_r: f64,
    pub correction_r: f64,
    pub surf_coef1: f64,
    pub surf_coef2: f64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_surface_corr(
    ctrl: &AsteroSearchControls,
    mod_l0: &[f64],
    in_l0: &[f64],
    all_obs: &[f64],
    all_sig: &[f64],
    all_mod: &[f64],
    all_in: &[f64],
    dnu_obs: f64,
    numax_obs: f64,
    nu_max_model: f64,
    nu_max_sun: f64,
    delta_nu_sun: f64,
    delta_nu_uhz: &mut f64,
) -> Result<SurfaceCorrOutput, StarError> {
    let nu_max_sun_eff = if nu_max_sun > 0.0 {
        nu_max_sun
    } else {
        NU_MAX_SUN
    };
    let delta_nu_sun_eff = if delta_nu_sun > 0.0 {
        delta_nu_sun
    } else {
        DELTA_NU_SUN
    };
    let cf = ctrl.correction_factor.max(0.0);
    let scheme = ctrl.correction_scheme.trim().to_ascii_lowercase();
    match scheme.as_str() {
        "" | "none" => Ok(SurfaceCorrOutput {
            corrected: all_mod.to_vec(),
            a_div_r: 0.0,
            correction_r: 1.0,
            surf_coef1: 0.0,
            surf_coef2: 0.0,
        }),
        "kjeldsen" => {
            if cf <= 0.0 || mod_l0.is_empty() {
                return Ok(SurfaceCorrOutput {
                    corrected: all_mod.to_vec(),
                    a_div_r: 0.0,
                    correction_r: 1.0,
                    surf_coef1: 0.0,
                    surf_coef2: 0.0,
                });
            }
            let n_obs = if !ctrl.l0_n_obs.is_empty() && ctrl.l0_n_obs[0] > 0.0 {
                ctrl.l0_n_obs.clone()
            } else {
                estimate_l0_n_obs_with_numax(
                    &ctrl.freq_target[0],
                    dnu_obs,
                    numax_obs,
                    delta_nu_sun_eff,
                    nu_max_sun_eff,
                )
            };
            let (mut c, adr, cr) = eval_kjeldsen_corr(
                &ctrl.freq_target[0],
                mod_l0,
                in_l0,
                &n_obs,
                dnu_obs,
                all_mod,
                all_in,
                numax_obs,
                ctrl.correction_b,
                cf,
            );
            for (ci, (&obs, &orig)) in
                c.iter_mut().zip(all_obs.iter().zip(all_mod.iter()))
            {
                if obs < 0.0 {
                    *ci = orig;
                }
            }
            let dnu_m = eval_radial_delta_nu_ols(&n_obs, mod_l0);
            if dnu_obs > 0.0 && numax_obs > 0.0 && dnu_m > 0.0 {
                *delta_nu_uhz = dnu_m;
            }
            let (sc1, sc2) =
                eval_surf_coefs(&scheme, adr, cr, nu_max_model, nu_max_sun_eff);
            Ok(SurfaceCorrOutput {
                corrected: c,
                a_div_r: adr,
                correction_r: cr,
                surf_coef1: sc1,
                surf_coef2: sc2,
            })
        }
        "cubic" => {
            let (c, a3) =
                eval_cubic_corr(all_obs, all_sig, all_mod, all_in, cf);
            let (sc1, sc2) =
                eval_surf_coefs(&scheme, a3, 0.0, nu_max_model, nu_max_sun_eff);
            Ok(SurfaceCorrOutput {
                corrected: c,
                a_div_r: a3,
                correction_r: 1.0,
                surf_coef1: sc1,
                surf_coef2: sc2,
            })
        }
        "combined" => {
            let (c, a3, a_neg1) =
                eval_combined_corr(all_obs, all_sig, all_mod, all_in, cf);
            let (sc1, sc2) = eval_surf_coefs(
                &scheme,
                a3,
                a_neg1,
                nu_max_model,
                nu_max_sun_eff,
            );
            Ok(SurfaceCorrOutput {
                corrected: c,
                a_div_r: a3,
                correction_r: a_neg1,
                surf_coef1: sc1,
                surf_coef2: sc2,
            })
        }
        "power_law" => {
            let (c, a, b) = eval_power_law_corr(
                all_obs,
                all_sig,
                all_mod,
                all_in,
                mod_l0,
                in_l0,
                nu_max_model,
                cf,
            );
            let (sc1, sc2) =
                eval_surf_coefs(&scheme, a, b, nu_max_model, nu_max_sun_eff);
            Ok(SurfaceCorrOutput {
                corrected: c,
                a_div_r: a,
                correction_r: b,
                surf_coef1: sc1,
                surf_coef2: sc2,
            })
        }
        "sonoi" => {
            let (c, a, b) = eval_sonoi_corr(
                all_obs,
                all_sig,
                all_mod,
                all_in,
                mod_l0,
                in_l0,
                nu_max_model,
                cf,
            );
            let (sc1, sc2) =
                eval_surf_coefs(&scheme, a, b, nu_max_model, nu_max_sun_eff);
            Ok(SurfaceCorrOutput {
                corrected: c,
                a_div_r: a,
                correction_r: b,
                surf_coef1: sc1,
                surf_coef2: sc2,
            })
        }
        other => Err(StarError::InvalidInput(format!(
            "unrecognized astero correction_scheme '{other}'"
        ))),
    }
}

/// Strictly monotonic mode matching (`set_to_closest` / `find_closest` in
/// `astero/private/astero_support.f90:223-292`).
#[allow(dead_code)]
pub(crate) fn match_degree_modes(
    targets: &[f64],
    modes: &[BvpEigenmode],
) -> Option<(Vec<f64>, Vec<f64>, Vec<isize>)> {
    let mut mf = Vec::with_capacity(targets.len());
    let mut mi = Vec::with_capacity(targets.len());
    let mut mo = Vec::with_capacity(targets.len());
    let mut jprev: Option<usize> = None;

    for &target_nu in targets {
        let start = jprev.map_or(0, |jp| jp + 1);
        if start >= modes.len() {
            return None;
        }
        let mut best_j = start;
        let mut best_d = (modes[start].freq_uhz - target_nu).abs();
        for (j, m) in modes.iter().enumerate().skip(start) {
            let d = (m.freq_uhz - target_nu).abs();
            if d < best_d {
                best_d = d;
                best_j = j;
            }
            if m.freq_uhz > target_nu {
                break;
            }
        }
        mf.push(modes[best_j].freq_uhz);
        mi.push(modes[best_j].e_norm.max(1.0e-30));
        mo.push(modes[best_j].n_pg as isize);
        jprev = Some(best_j);
    }
    Some((mf, mi, mo))
}

#[cfg(test)]
mod tests {
    use super::build_scan_grid_params;
    use crate::inlist_astero::{AsteroSearchControls, AsteroSearchParam};

    fn param(
        vary: bool,
        first: f64,
        min: f64,
        max: f64,
        delta: f64,
    ) -> AsteroSearchParam {
        AsteroSearchParam {
            vary,
            first,
            min,
            max,
            delta,
        }
    }

    /// `astero/test_suite/fast_scan_grid/inlist_astero_search_controls`
    /// at MESA 26.4.1: `initial_mass` (index 1) and `alpha` (index 5) vary.
    #[test]
    fn scan_grid_varies_the_lowest_param_index_fastest() {
        let mut ctrl = AsteroSearchControls::default();
        let set = [
            (0, "initial_mass", param(true, 1.0, 0.99, 1.01, 0.02)),
            (1, "initial_Y", param(false, 0.267, 0.24, 0.30, 0.01)),
            (2, "initial_FeH", param(false, 0.0, 0.0, 0.17, 0.03)),
            (4, "alpha", param(true, 1.8, 1.7, 1.9, 0.2)),
            (7, "f_ov", param(false, 0.015, 0.0, 0.03, 0.0)),
        ];
        for (i, name, p) in set {
            ctrl.param_names[i] = String::from(name);
            ctrl.indexed_params[i] = p;
        }
        let grid = build_scan_grid_params(&ctrl);
        let got: Vec<(f64, f64)> =
            grid.iter().map(|p| (p.initial_mass, p.alpha)).collect();
        // `min + delta` by repeated addition, as upstream does.
        let (m1, a1) = (0.99 + 0.02, 1.7 + 0.2);
        assert_eq!(got, vec![(0.99, 1.7), (m1, 1.7), (0.99, a1), (m1, a1)]);
        assert!(grid.iter().all(|p| p.initial_y == 0.267 && p.f_ov == 0.015));
    }

    /// `do_param` exits before the first evaluation when `min_param`
    /// exceeds `max_param + 1d-6`, so the grid is empty.
    #[test]
    fn scan_grid_with_min_above_max_has_no_points() {
        let mut ctrl = AsteroSearchControls::default();
        ctrl.param_names[0] = String::from("initial_mass");
        ctrl.indexed_params[0] = param(true, 1.0, 1.2, 1.0, 0.1);
        assert!(build_scan_grid_params(&ctrl).is_empty());
    }
}
