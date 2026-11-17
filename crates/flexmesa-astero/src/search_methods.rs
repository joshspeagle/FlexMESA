// SPDX-License-Identifier: LGPL-3.0-or-later
//! Optimization drivers (`scan_grid`, `simplex`, `newuoa`, `bobyqa`),
//! output writers, surface-correction dispatch, and mode-matching helpers
//! extracted from `search_driver` and `sample_eval`.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use flexmesa_star::pulse_bvp::BvpEigenmode;
use flexmesa_star::StarError;

use crate::inlist_astero::{
    AsteroParamSet, AsteroSearchControls, AsteroSearchParam,
};
use crate::optimize::{
    try_bobyqa_minimize, try_newuoa_minimize, try_simplex_minimize_with_options,
};
use crate::sample_eval::AsteroSampleResult;
use crate::scaling::eval_radial_delta_nu_ols;
use crate::surface_corr::{
    estimate_l0_n_obs, eval_combined_corr, eval_cubic_corr, eval_kjeldsen_corr,
    eval_power_law_corr, eval_sonoi_corr,
};

pub(crate) fn build_scan_grid_params(
    ctrl: &AsteroSearchControls,
) -> Vec<AsteroParamSet> {
    let axis = |sp: AsteroSearchParam| -> Vec<f64> {
        if !sp.vary || sp.delta <= 0.0 || sp.max < sp.min {
            return vec![sp.first];
        }
        let mut v = Vec::new();
        let mut x = sp.min;
        while x <= sp.max + 1.0e-9 {
            v.push(x);
            x += sp.delta;
        }
        if v.is_empty() {
            vec![sp.first]
        } else {
            v
        }
    };
    let mut out = Vec::new();
    for m in axis(ctrl.mass) {
        for y in axis(ctrl.y) {
            for f in axis(ctrl.feh_param) {
                for a in axis(ctrl.alpha) {
                    for ov in axis(ctrl.f_ov) {
                        for p1 in axis(ctrl.my_param[0]) {
                            for p2 in axis(ctrl.my_param[1]) {
                                for p3 in axis(ctrl.my_param[2]) {
                                    out.push(AsteroParamSet {
                                        initial_mass: m,
                                        initial_y: y,
                                        initial_feh: f,
                                        alpha: a,
                                        f_ov: ov,
                                        my_param1: p1,
                                        my_param2: p2,
                                        my_param3: p3,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn active_search_params(
    ctrl: &AsteroSearchControls,
) -> Vec<(usize, AsteroSearchParam)> {
    let all = [
        ctrl.mass,
        ctrl.y,
        ctrl.feh_param,
        ctrl.alpha,
        ctrl.f_ov,
        ctrl.my_param[0],
        ctrl.my_param[1],
        ctrl.my_param[2],
    ];
    all.into_iter()
        .enumerate()
        .filter(|(_, p)| p.vary && p.max > p.min)
        .collect()
}

fn apply_varied_slice(
    ctrl: &AsteroSearchControls,
    active: &[(usize, AsteroSearchParam)],
    vals: &[f64],
) -> AsteroParamSet {
    let mut p = ctrl.first_params();
    for (&(idx, _), &v) in active.iter().zip(vals.iter()) {
        match idx {
            0 => p.initial_mass = v,
            1 => p.initial_y = v,
            2 => p.initial_feh = v,
            3 => p.alpha = v,
            4 => p.f_ov = v,
            5 => p.my_param1 = v,
            6 => p.my_param2 = v,
            7 => p.my_param3 = v,
            _ => {}
        }
    }
    p
}

pub(crate) fn run_simplex_driver(
    ctrl: &AsteroSearchControls,
    eval_fn: &mut impl FnMut(AsteroParamSet) -> Result<f64, StarError>,
) -> Result<(), StarError> {
    let active = active_search_params(ctrl);
    if active.is_empty() {
        let _ = eval_fn(ctrl.first_params())?;
        return Ok(());
    }
    let x_first: Vec<f64> = active.iter().map(|(_, p)| p.first).collect();
    let x_lower: Vec<f64> = active.iter().map(|(_, p)| p.min).collect();
    let x_upper: Vec<f64> = active.iter().map(|(_, p)| p.max).collect();
    let max_evals = ctrl.simplex_fcn_calls_max.max(1);
    let _ = try_simplex_minimize_with_options(
        &x_first,
        &x_lower,
        &x_upper,
        max_evals,
        ctrl.simplex_x_atol,
        ctrl.simplex_x_rtol,
        ctrl.simplex_chi2_tol,
        ctrl.simplex_centroid_weight_power,
        ctrl.simplex_enforce_bounds,
        ctrl.simplex_adaptive_random_search,
        ctrl.simplex_seed,
        |x| eval_fn(apply_varied_slice(ctrl, &active, x)),
    )?;
    Ok(())
}

pub(crate) fn run_powell_driver(
    ctrl: &AsteroSearchControls,
    bounded: bool,
    eval_fn: &mut impl FnMut(AsteroParamSet) -> Result<f64, StarError>,
) -> Result<(), StarError> {
    let active = active_search_params(ctrl);
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

pub(crate) fn write_search_results_table(
    out_path: &Path,
    samples: &[AsteroSampleResult],
) -> Result<(), StarError> {
    let mut buf = String::new();
    let _ = writeln!(
        buf,
        "{:>8} {:>20} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16} {:>16}",
        "sample",
        "chi2",
        "initial_mass",
        "initial_Y",
        "initial_FeH",
        "alpha",
        "age",
        "delta_nu",
        "nu_max"
    );
    for s in samples {
        let _ = writeln!(
            buf,
            "{:8} {:20.12e} {:16.8} {:16.8} {:16.8} {:16.8} {:16.8e} {:16.8} {:16.8}",
            s.sample_number,
            s.chi2,
            s.params.initial_mass,
            s.params.initial_y,
            s.params.initial_feh,
            s.params.alpha,
            s.age_yr,
            s.delta_nu_uhz,
            s.nu_max_uhz
        );
    }
    fs::write(out_path, buf).map_err(|e| {
        StarError::IoError(format!(
            "failed writing search results '{}': {e}",
            out_path.display()
        ))
    })
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
        if obs > 0.0 && sig > 0.0 {
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
    delta_nu_uhz: &mut f64,
) -> Result<(Vec<f64>, f64, f64), StarError> {
    if ctrl.correction_factor <= 0.0 {
        return Ok((all_mod.to_vec(), 0.0, 1.0));
    }
    match ctrl.correction_scheme.trim().to_ascii_lowercase().as_str() {
        "" | "none" => Ok((all_mod.to_vec(), 0.0, 1.0)),
        "kjeldsen" => {
            if mod_l0.is_empty() {
                return Ok((all_mod.to_vec(), 0.0, 1.0));
            }
            let n_obs = if !ctrl.l0_n_obs.is_empty() && ctrl.l0_n_obs[0] > 0.0 {
                ctrl.l0_n_obs.clone()
            } else {
                estimate_l0_n_obs(&ctrl.freq_target[0], dnu_obs)
            };
            let (c, adr, cr) = eval_kjeldsen_corr(
                &ctrl.freq_target[0],
                mod_l0,
                in_l0,
                &n_obs,
                dnu_obs,
                all_mod,
                all_in,
                numax_obs,
                ctrl.correction_b,
                ctrl.correction_factor,
            );
            let dnu_m = eval_radial_delta_nu_ols(&n_obs, mod_l0);
            if dnu_m > 0.0 {
                *delta_nu_uhz = dnu_m;
            }
            Ok((c, adr, cr))
        }
        "cubic" => {
            let (c, a3) = eval_cubic_corr(
                all_obs,
                all_sig,
                all_mod,
                all_in,
                ctrl.correction_factor,
            );
            Ok((c, a3, 1.0))
        }
        "combined" => Ok(eval_combined_corr(
            all_obs,
            all_sig,
            all_mod,
            all_in,
            ctrl.correction_factor,
        )),
        "power_law" => Ok(eval_power_law_corr(
            all_obs,
            all_sig,
            all_mod,
            all_in,
            mod_l0,
            in_l0,
            numax_obs,
            ctrl.correction_factor,
        )),
        "sonoi" => Ok(eval_sonoi_corr(
            all_obs,
            all_sig,
            all_mod,
            all_in,
            mod_l0,
            in_l0,
            numax_obs,
            ctrl.correction_factor,
        )),
        other => Err(StarError::InvalidInput(format!(
            "unrecognized astero correction_scheme '{other}'"
        ))),
    }
}

/// Strictly monotonic mode matching (`set_to_closest` / `find_closest` in
/// `astero/private/astero_support.f90:223-292`).
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
