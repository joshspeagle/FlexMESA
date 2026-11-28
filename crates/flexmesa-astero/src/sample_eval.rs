// SPDX-License-Identifier: GPL-3.0-or-later
//! Per-sample spectroscopic and asteroseismic observable evaluation, pulsation
//! mode matching, surface-effect correction, and composite $\chi^2$ scoring
//! (`astero/private/extras_support.f90:155-485`,
//! `astero/private/astero_support.f90:223-760`).

use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::time::Instant;

use flexmesa_const::{MSUN, RSUN};
use flexmesa_star::controls::StarError;
use flexmesa_star::eval_photosphere_info;
use flexmesa_star::history::extract_history_row;
use flexmesa_star::model::StarModel;
use flexmesa_star::pulse_bvp::{BvpEigenmode, InertiaNormKind, OuterBoundKind};
use flexmesa_star::pulse_modes::{
    scan_bounds_from_targets, scan_star_bvp_eigenmodes, PulseModeScanOptions,
};
use flexmesa_star::pulse_write::fortran_1pe;

use crate::adipls_support::AsteroAdipls;
use crate::gyre_support::{AsteroGyre, OscResult};
use crate::inlist_astero::{
    AsteroParamSet, AsteroSearchControls, MAX_CONSTRAINTS,
};
use crate::ratios_chi2::{
    eval_astero_chi2, eval_frequency_ratios_with_trimming,
    eval_r02_model_with_indices, eval_r02_obs_with_indices,
    interpolate_ratio_r010, interpolate_ratio_r02,
};
use crate::scaling::{
    eval_acoustic_profile, eval_delta_nu_asymptotic, DELTA_NU_SUN, NU_MAX_SUN,
    TEFF_SUN_ASTERO,
};
use crate::search_methods::{
    apply_surface_corr, eval_radial_chi2, SurfaceCorrOutput,
};

/// Best-fitting stellar model summary for a single evolutionary track sample.
#[derive(Debug, Clone)]
pub struct AsteroSampleResult {
    /// 1-based sample index in the optimization run.
    pub sample_number: usize,
    /// Evaluated stellar track parameter vector.
    pub params: AsteroParamSet,
    /// Evolutionary step model number at the best-fitting model.
    pub model_number: usize,
    /// Stellar age in years (`yr`).
    pub age_yr: f64,
    /// Effective temperature $T_{\mathrm{eff}}$ (`K`).
    pub teff: f64,
    /// Surface gravity $\log g$ (cgs).
    pub logg: f64,
    /// Log photospheric luminosity $\log(L/L_\odot)$.
    pub log_l: f64,
    /// Log photospheric radius $\log(R/R_\odot)$.
    pub log_r: f64,
    /// `constraint_value(i)` from the case's `set_constraint_value`, in
    /// `constraint_name` index order (`NaN` for unnamed slots;
    /// `sample_constraint_value`, `astero/private/astero_run_support.f90:999`).
    pub constraint_values: [f64; MAX_CONSTRAINTS],
    /// Model large frequency separation $\Delta\nu_{\mathrm{model}}$ ($\mu\mathrm{Hz}$).
    pub delta_nu_uhz: f64,
    /// Model frequency of maximum power $\nu_{\max,\mathrm{model}}$ ($\mu\mathrm{Hz}$).
    pub nu_max_uhz: f64,
    /// Surface-effect amplitude parameter (`a_div_r`, `a3`, `b3`, `p0`, or `s0`).
    pub a_div_r: f64,
    /// Secondary surface-effect parameter (`r`, `b1`, `p1`, or `s1`).
    pub correction_r: f64,
    /// Total weighted $\chi^2$.
    pub chi2: f64,
    /// Seismic component $\chi^2_{\mathrm{seismo}}$.
    pub chi2_seismo: f64,
    /// Spectroscopic component $\chi^2_{\mathrm{spectro}}$.
    pub chi2_spectro: f64,
    /// Radial-only frequency component $\chi^2_{\mathrm{radial}}$.
    pub chi2_radial: f64,
    /// Five-point separation ratio component $\chi^2_{r_{010}}$.
    pub chi2_r010: f64,
    /// Quadrupole-to-radial separation ratio component $\chi^2_{r_{02}}$.
    pub chi2_r02: f64,
    /// Uncorrected matched model frequencies ($\mu\mathrm{Hz}$) for $l = 0..3$.
    pub model_freqs: [Vec<f64>; 4],
    /// Surface-corrected matched model frequencies ($\mu\mathrm{Hz}$) for $l = 0..3$.
    pub model_freqs_corr: [Vec<f64>; 4],
    /// Normalized mode inertias $E_{\mathrm{norm}}$ for matched modes ($l = 0..3$).
    pub model_inertias: [Vec<f64>; 4],
    /// Radial orders $n_{pg}$ for matched modes ($l = 0..3$).
    pub model_orders: [Vec<isize>; 4],
    /// Primary surface-correction coefficient (`surf_coef1`, `astero/private/astero_support.f90:1280-1333`).
    pub surf_coef1: f64,
    /// Secondary surface-correction coefficient (`surf_coef2`, `astero/private/astero_support.f90:1280-1333`).
    pub surf_coef2: f64,
    /// Number of trimmed ratio pairs (`ratios_n`).
    pub ratios_n: usize,
    /// Zero-based starting $l=0$ index for frequency ratios (`ratios_l0_first`).
    pub ratios_l0_first: usize,
    /// Zero-based starting $l=1$ index for frequency ratios (`ratios_l1_first`).
    pub ratios_l1_first: usize,
    /// Observed $r_{01}$ ratios (`len = ratios_n`).
    pub obs_r01: Vec<f64>,
    /// Uncertainties $\sigma_{r_{01}}$ (`len = ratios_n`).
    pub obs_r01_sigmas: Vec<f64>,
    /// Model $r_{01}$ ratios interpolated at observed frequencies (`len = ratios_n`).
    pub model_r01: Vec<f64>,
    /// Observed $r_{10}$ ratios (`len = ratios_n`).
    pub obs_r10: Vec<f64>,
    /// Uncertainties $\sigma_{r_{10}}$ (`len = ratios_n`).
    pub obs_r10_sigmas: Vec<f64>,
    /// Model $r_{10}$ ratios interpolated at observed frequencies (`len = ratios_n`).
    pub model_r10: Vec<f64>,
    /// Observed $r_{02}$ ratios padded to `nl[0]`.
    pub obs_r02: Vec<f64>,
    /// Uncertainties $\sigma_{r_{02}}$ padded to `nl[0]` (`0.0` when unpaired).
    pub obs_r02_sigmas: Vec<f64>,
    /// Model $r_{02}$ ratios padded to `nl[0]`.
    pub model_r02: Vec<f64>,
    /// Next-best higher-frequency uncorrected matches (`save_next_best_at_higher_frequency`).
    pub model_freqs_alt_up: [Vec<f64>; 4],
    /// Surface-corrected next-best higher-frequency matches (`save_next_best_at_higher_frequency`).
    pub model_freqs_corr_alt_up: [Vec<f64>; 4],
    /// Normalized mode inertias for next-best higher-frequency matches.
    pub model_inertias_alt_up: [Vec<f64>; 4],
    /// Radial orders for next-best higher-frequency matches.
    pub model_orders_alt_up: [Vec<isize>; 4],
    /// Next-best lower-frequency uncorrected matches (`save_next_best_at_lower_frequency`).
    pub model_freqs_alt_down: [Vec<f64>; 4],
    /// Surface-corrected next-best lower-frequency matches (`save_next_best_at_lower_frequency`).
    pub model_freqs_corr_alt_down: [Vec<f64>; 4],
    /// Normalized mode inertias for next-best lower-frequency matches.
    pub model_inertias_alt_down: [Vec<f64>; 4],
    /// Radial orders for next-best lower-frequency matches.
    pub model_orders_alt_down: [Vec<isize>; 4],
}

/// Spectroscopic terms of $\chi^2$ in `constraint_name` index order, with the
/// constraint values from the case's `set_constraint_value`
/// (`get_chi2_spectro`, `astero/private/extras_support.f90:671-695`).
/// Returns `(targets, sigmas, model values, names, Teff, log g, log L, log R)`.
fn eval_spectro_obs(
    model: &StarModel,
    ctrl: &AsteroSearchControls,
    constraint_values: &[f64; MAX_CONSTRAINTS],
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<String>,
    f64,
    f64,
    f64,
    f64,
) {
    let hr = extract_history_row(model);
    let teff = model.teff.max(1.0);
    let (mut obs, mut sig, mut mval, mut names) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    if ctrl.include_age_in_chi2_spectro && ctrl.age_sigma > 0.0 {
        obs.push(ctrl.age_target);
        sig.push(ctrl.age_sigma);
        mval.push(model.star_age);
        names.push(String::from("age"));
    }
    for (i, c) in ctrl.indexed_constraints.iter().enumerate() {
        if c.include && c.sigma > 0.0 {
            obs.push(c.target);
            sig.push(c.sigma);
            mval.push(constraint_values[i]);
            let cname = ctrl.constraint_names[i].trim();
            names.push(if cname.is_empty() {
                format!("constraint_{}", i + 1)
            } else {
                String::from(cname)
            });
        }
    }
    (obs, sig, mval, names, teff, hr.log_g, hr.log_l, hr.log_r)
}

/// Computes model $(\Delta\nu, \nu_{\max})$ in $\mu\mathrm{Hz}$ using the
/// star model's solar references (`star/private/report.f90:342-355`).
fn eval_model_delta_nu_and_nu_max(model: &StarModel, teff: f64) -> (f64, f64) {
    let phot = eval_photosphere_info(model);
    let r_phot_cm = phot.r_cm.max(1.0e-30);
    let o = &model.controls.output;
    let teff_sun = if o.astero_teff_sun > 0.0 {
        o.astero_teff_sun
    } else {
        TEFF_SUN_ASTERO
    };
    let nu_max_sun = if o.nu_max_sun > 0.0 {
        o.nu_max_sun
    } else {
        NU_MAX_SUN
    };
    let delta_nu_sun = if o.delta_nu_sun > 0.0 {
        o.delta_nu_sun
    } else {
        DELTA_NU_SUN
    };
    let delta_nu_uhz = if o.get_delta_nu_from_scaled_solar && phot.l_lsun > 0.0
    {
        let star_mass = (model.mstar / MSUN).max(0.0);
        delta_nu_sun * star_mass.sqrt() * flexmesa_math::pow3(teff / teff_sun)
            / flexmesa_math::pow(phot.l_lsun, 0.75)
    } else {
        let (_, _, tau_0) = eval_acoustic_profile(
            &model.r,
            &model.csound,
            model.r_center,
            r_phot_cm,
        );
        if tau_0 > 0.0 {
            eval_delta_nu_asymptotic(tau_0)
        } else {
            let r_rsun = r_phot_cm / RSUN;
            delta_nu_sun
                * (model.mstar / MSUN).sqrt()
                * flexmesa_math::pow(r_rsun, -1.5)
        }
    };
    let r_rsun = r_phot_cm / RSUN;
    let nu_max_uhz = nu_max_sun * (model.mstar / MSUN)
        / (r_rsun * r_rsun * (teff.max(0.0) / teff_sun).sqrt());
    (delta_nu_uhz, nu_max_uhz)
}

/// Fast pre-screen returning `(chi2_spectro, chi2_delta_nu, delta_nu_uhz)` without mode solving.
#[must_use]
pub fn eval_spectro_and_delta_nu_chi2(
    model: &StarModel,
    ctrl: &AsteroSearchControls,
    constraint_values: &[f64; MAX_CONSTRAINTS],
) -> (f64, f64, f64) {
    let (sp_obs, sp_sig, sp_mod, _, teff, _, _, _) =
        eval_spectro_obs(model, ctrl, constraint_values);
    let mut sum_sp = 0.0_f64;
    let mut n_sp = 0usize;
    for i in 0..sp_obs.len() {
        if sp_sig[i] > 0.0 {
            let d = (sp_mod[i] - sp_obs[i]) / sp_sig[i];
            sum_sp += d * d;
            n_sp += 1;
        }
    }
    let chi2_spectro = if ctrl.weights.normalize_chi2_spectro && n_sp > 0 {
        sum_sp / (n_sp as f64)
    } else {
        sum_sp
    };
    let (delta_nu_uhz, _) = eval_model_delta_nu_and_nu_max(model, teff);
    // `if (delta_nu_sigma > 0) chi2_delta_nu = pow2((delta_nu -
    // delta_nu_model)/delta_nu_sigma)`, else 0
    // (`astero/private/extras_support.f90:236-241`).
    let chi2_delta_nu = if ctrl.delta_nu_sigma > 0.0 {
        flexmesa_math::pow2(
            (ctrl.delta_nu - delta_nu_uhz) / ctrl.delta_nu_sigma,
        )
    } else {
        0.0
    };
    (chi2_spectro, chi2_delta_nu, delta_nu_uhz)
}

/// Parses `outer_bound` from a `GYRE` input file, stripping `!` comments first.
pub fn parse_gyre_outer_bound(
    case_dir: &Path,
    gyre_file: &str,
) -> Result<OuterBoundKind, StarError> {
    let fname = gyre_file.trim();
    if fname.is_empty() {
        return Err(StarError::InvalidInput(String::from(
            "oscillation_code = 'gyre' requires a non-empty gyre_input_file",
        )));
    }
    let path = case_dir.join(fname);
    let text = fs::read_to_string(&path).map_err(|e| {
        StarError::IoError(format!(
            "failed to read GYRE input file '{}': {e}",
            path.display()
        ))
    })?;
    for raw_line in text.lines() {
        let line = raw_line.split('!').next().unwrap_or("").trim();
        let Some((lhs, rhs)) = line.split_once('=') else {
            continue;
        };
        if lhs.trim().eq_ignore_ascii_case("outer_bound") {
            // Strip Fortran namelist separator `,` before unquoting
            // (`gyre_input_file`, `astero/private/gyre_support.f90:45-55`).
            let val = rhs
                .trim()
                .trim_end_matches(',')
                .trim()
                .trim_matches(|c| c == '\'' || c == '"');
            return match val.to_ascii_uppercase().as_str() {
                "VACUUM" | "ZERO_R" => Ok(OuterBoundKind::Vacuum),
                "JCD" | "DZIEM" | "UNNO" | "ISOTHERMAL" => {
                    Ok(OuterBoundKind::AdiplsIsothermal)
                }
                _ => Err(StarError::InvalidInput(format!(
                    "unrecognized GYRE outer_bound '{val}' in '{}'",
                    path.display()
                ))),
            };
        }
    }
    Ok(OuterBoundKind::AdiplsIsothermal)
}

/// Evaluates spectroscopic and asteroseismic $\chi^2$ on `model`
/// (`astero/private/extras_support.f90:155-485`).
pub fn eval_model_astero_sample(
    model: &StarModel,
    ctrl: &AsteroSearchControls,
    constraint_values: &[f64; MAX_CONSTRAINTS],
    params: AsteroParamSet,
    sample_number: usize,
    case_dir: &Path,
) -> Result<AsteroSampleResult, StarError> {
    eval_model_astero_sample_with_output_root(
        model,
        ctrl,
        constraint_values,
        params,
        sample_number,
        case_dir,
        case_dir,
    )
}

/// Evaluates spectroscopic and asteroseismic $\chi^2$ on `model`, directing
/// per-model FGONG/GYRE/mode outputs into `output_root`.
#[allow(clippy::too_many_lines)]
pub(crate) fn eval_model_astero_sample_with_output_root(
    model: &StarModel,
    ctrl: &AsteroSearchControls,
    constraint_values: &[f64; MAX_CONSTRAINTS],
    params: AsteroParamSet,
    sample_number: usize,
    case_dir: &Path,
    output_root: &Path,
) -> Result<AsteroSampleResult, StarError> {
    // Clamp component fractions to `[0, 1]` before computing `f_freq`
    // (`astero/private/extras_support.f90:158-170`).
    let f_r010 = ctrl.weights.chi2_seismo_r_010_fraction.clamp(0.0, 1.0);
    let f_r02 = ctrl.weights.chi2_seismo_r_02_fraction.clamp(0.0, 1.0);
    let f_dnu = ctrl.weights.chi2_seismo_delta_nu_fraction.clamp(0.0, 1.0);
    let f_numax = ctrl.weights.chi2_seismo_nu_max_fraction.clamp(0.0, 1.0);
    let f_freq = ctrl
        .weights
        .chi2_seismo_freq_fraction
        .unwrap_or(1.0 - (f_r010 + f_r02 + f_dnu + f_numax))
        .clamp(0.0, 1.0);
    // Checked before `if (chi2_seismo_fraction <= 0d0)`
    // (`astero/private/extras_support.f90:244-259, 266`).
    if f_numax > 0.0 && (ctrl.nu_max <= 0.0 || ctrl.nu_max_sigma <= 0.0) {
        return Err(StarError::InvalidInput(String::from(
            "chi2_seismo_nu_max_fraction > 0 requires positive nu_max and nu_max_sigma",
        )));
    }
    let (sp_obs, sp_sig, sp_mod, sp_names, teff, logg, log_l, log_r) =
        eval_spectro_obs(model, ctrl, constraint_values);
    let (mut delta_nu_uhz, nu_max_uhz) =
        eval_model_delta_nu_and_nu_max(model, teff);
    let dnu_obs = if ctrl.delta_nu > 0.0 {
        ctrl.delta_nu
    } else {
        delta_nu_uhz
    };
    // `nu_max` is never replaced with `nu_max_model` (`astero/private/astero_support.f90:183, 204`).
    let numax_obs = ctrl.nu_max;
    // Use the star model's solar references (`star/private/report.f90:345-355`).
    let model_nu_max_sun = if model.controls.output.nu_max_sun > 0.0 {
        model.controls.output.nu_max_sun
    } else {
        NU_MAX_SUN
    };
    let model_delta_nu_sun = if model.controls.output.delta_nu_sun > 0.0 {
        model.controls.output.delta_nu_sun
    } else {
        DELTA_NU_SUN
    };

    let mut model_freqs: [Vec<f64>; 4] = Default::default();
    let mut model_freqs_corr: [Vec<f64>; 4] = Default::default();
    let mut model_inertias: [Vec<f64>; 4] = Default::default();
    let mut model_orders: [Vec<isize>; 4] = Default::default();
    let mut model_freqs_alt_up: [Vec<f64>; 4] = Default::default();
    let mut model_freqs_corr_alt_up: [Vec<f64>; 4] = Default::default();
    let mut model_inertias_alt_up: [Vec<f64>; 4] = Default::default();
    let mut model_orders_alt_up: [Vec<isize>; 4] = Default::default();
    let mut model_freqs_alt_down: [Vec<f64>; 4] = Default::default();
    let mut model_freqs_corr_alt_down: [Vec<f64>; 4] = Default::default();
    let mut model_inertias_alt_down: [Vec<f64>; 4] = Default::default();
    let mut model_orders_alt_down: [Vec<isize>; 4] = Default::default();
    let (mut a_div_r, mut correction_r) = (0.0_f64, 1.0_f64);
    let (mut surf_coef1, mut surf_coef2) = (0.0_f64, 0.0_f64);
    let mut mode_match_failed = false;

    // Upstream skips all mode calls when `chi2_seismo_fraction <= 0d0` or when
    // `(chi2_seismo_freq_fraction + r_010 + r_02) <= 0d0`
    // (`astero/private/extras_support.f90:261-280, 326, 348`).
    let need_modes = ctrl.weights.chi2_seismo_fraction > 0.0
        && ctrl.nl.iter().sum::<usize>() > 0
        && (f_freq + f_r010 + f_r02) > 0.0;
    let mut pre_chi2_radial = 0.0_f64;
    if need_modes {
        let mut scanner =
            DegreeScanner::init(model, ctrl, case_dir, output_root)?;
        if ctrl.nl[0] > 0 {
            if let Some(m0) =
                scanner.scan_degree(model, ctrl, 0, dnu_obs, delta_nu_uhz)?
            {
                (model_freqs[0], model_inertias[0], model_orders[0]) =
                    (m0.freqs, m0.inertias, m0.orders);
                model_freqs_alt_up[0] = m0.freqs_alt_up;
                model_inertias_alt_up[0] = m0.inertias_alt_up;
                model_orders_alt_up[0] = m0.orders_alt_up;
                model_freqs_alt_down[0] = m0.freqs_alt_down;
                model_inertias_alt_down[0] = m0.inertias_alt_down;
                model_orders_alt_down[0] = m0.orders_alt_down;
                let has_all_l0 =
                    !model_freqs[0].windows(2).enumerate().any(|(idx, w)| {
                        ctrl.freq_target[0]
                            .get(idx + 1)
                            .is_some_and(|&t| t >= 0.0)
                            && (w[1] - w[0]).abs() == 0.0
                    });
                if has_all_l0 {
                    let sc0 = apply_surface_corr(
                        ctrl,
                        &model_freqs[0],
                        &model_inertias[0],
                        &ctrl.freq_target[0],
                        &ctrl.freq_sigma[0],
                        &model_freqs[0],
                        &model_inertias[0],
                        dnu_obs,
                        numax_obs,
                        nu_max_uhz,
                        model_nu_max_sun,
                        model_delta_nu_sun,
                        &mut delta_nu_uhz,
                    )?;
                    (a_div_r, correction_r) = (sc0.a_div_r, sc0.correction_r);
                    (surf_coef1, surf_coef2) = (sc0.surf_coef1, sc0.surf_coef2);
                    pre_chi2_radial = eval_radial_chi2(ctrl, &sc0.corrected);
                    model_freqs_corr[0] = sc0.corrected;
                    if pre_chi2_radial > ctrl.chi2_radial_limit
                        && ctrl.nl[1] + ctrl.nl[2] + ctrl.nl[3] > 0
                    {
                        mode_match_failed = true;
                    }
                } else {
                    model_freqs_corr[0].clone_from(&model_freqs[0]);
                    mode_match_failed = true;
                }
            } else {
                mode_match_failed = true;
            }
        }
        if !mode_match_failed {
            for l in 1..4 {
                if ctrl.nl[l] == 0 {
                    continue;
                }
                if let Some(ml) = scanner.scan_degree(
                    model,
                    ctrl,
                    l,
                    dnu_obs,
                    delta_nu_uhz,
                )? {
                    (model_freqs[l], model_inertias[l], model_orders[l]) =
                        (ml.freqs, ml.inertias, ml.orders);
                    model_freqs_alt_up[l] = ml.freqs_alt_up;
                    model_inertias_alt_up[l] = ml.inertias_alt_up;
                    model_orders_alt_up[l] = ml.orders_alt_up;
                    model_freqs_alt_down[l] = ml.freqs_alt_down;
                    model_inertias_alt_down[l] = ml.inertias_alt_down;
                    model_orders_alt_down[l] = ml.orders_alt_down;
                } else {
                    mode_match_failed = true;
                    break;
                }
            }
        }
        if !mode_match_failed && ctrl.nl[1] + ctrl.nl[2] + ctrl.nl[3] > 0 {
            let (mut all_obs, mut all_sig, mut all_mod, mut all_in) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for l in 0..4 {
                all_obs.extend_from_slice(&ctrl.freq_target[l]);
                all_sig.extend_from_slice(&ctrl.freq_sigma[l]);
                all_mod.extend_from_slice(&model_freqs[l]);
                all_in.extend_from_slice(&model_inertias[l]);
            }
            let sc_all = apply_surface_corr(
                ctrl,
                &model_freqs[0],
                &model_inertias[0],
                &all_obs,
                &all_sig,
                &all_mod,
                &all_in,
                dnu_obs,
                numax_obs,
                nu_max_uhz,
                model_nu_max_sun,
                model_delta_nu_sun,
                &mut delta_nu_uhz,
            )?;
            (a_div_r, correction_r) = (sc_all.a_div_r, sc_all.correction_r);
            (surf_coef1, surf_coef2) = (sc_all.surf_coef1, sc_all.surf_coef2);
            let mut offset = 0usize;
            for l in 0..4 {
                let n = model_freqs[l].len();
                model_freqs_corr[l] =
                    sc_all.corrected[offset..offset + n].to_vec();
                offset += n;
            }
            if ctrl.save_next_best_at_higher_frequency {
                model_freqs_corr_alt_up = eval_alt_surface_corr(
                    ctrl,
                    &model_freqs,
                    &model_inertias,
                    &model_freqs_alt_up,
                    &model_inertias_alt_up,
                    &all_obs,
                    &all_sig,
                    dnu_obs,
                    numax_obs,
                    nu_max_uhz,
                    model_nu_max_sun,
                    model_delta_nu_sun,
                )?;
            }
            if ctrl.save_next_best_at_lower_frequency {
                model_freqs_corr_alt_down = eval_alt_surface_corr(
                    ctrl,
                    &model_freqs,
                    &model_inertias,
                    &model_freqs_alt_down,
                    &model_inertias_alt_down,
                    &all_obs,
                    &all_sig,
                    dnu_obs,
                    numax_obs,
                    nu_max_uhz,
                    model_nu_max_sun,
                    model_delta_nu_sun,
                )?;
            }
        } else if !mode_match_failed {
            if ctrl.save_next_best_at_higher_frequency {
                model_freqs_corr_alt_up[0].clone_from(&model_freqs_corr[0]);
            }
            if ctrl.save_next_best_at_lower_frequency {
                model_freqs_corr_alt_down[0].clone_from(&model_freqs_corr[0]);
            }
        }
    }

    let mut ratios_n = 0usize;
    let mut ratios_l0_first = 0usize;
    let mut ratios_l1_first = 0usize;
    let (mut r010_obs, mut r010_sig, mut r010_mod) =
        (Vec::new(), Vec::new(), Vec::new());
    let (mut obs_r01, mut obs_r01_sigmas, mut model_r01) =
        (Vec::new(), Vec::new(), Vec::new());
    let (mut obs_r10, mut obs_r10_sigmas, mut model_r10) =
        (Vec::new(), Vec::new(), Vec::new());
    let (mut r02_obs_vec, mut r02_sig_vec, mut r02_mod_vec) =
        (Vec::new(), Vec::new(), Vec::new());
    let mut obs_r02 = vec![0.0_f64; ctrl.nl[0]];
    let mut obs_r02_sigmas = vec![0.0_f64; ctrl.nl[0]];
    let mut model_r02 = vec![0.0_f64; ctrl.nl[0]];

    // Match `astero/private/astero_support.f90:610-622, 1410-1478` and
    // `astero/private/extras_support.f90:396-425`: populate observed/model
    // separation ratios when modes were computed, reject mismatched `r_010`
    // sequences or empty `r_02` pairs via `mode_match_failed`, and interpolate
    // `r_02` against uncorrected `model_freqs[0]` (`astero/private/astero_support.f90:1458`).
    if need_modes && !mode_match_failed {
        if (f_r010 > 0.0 || f_r02 > 0.0) && ctrl.nl[0] > 0 && ctrl.nl[1] > 0 {
            let obs_r = eval_frequency_ratios_with_trimming(
                &ctrl.freq_target[0],
                &ctrl.freq_sigma[0],
                &ctrl.freq_target[1],
                &ctrl.freq_sigma[1],
                dnu_obs,
                true,
                0,
                0,
                0,
            );
            ratios_n = obs_r.n;
            ratios_l0_first = obs_r.l0_first;
            ratios_l1_first = obs_r.l1_first;
            if f_r010 > 0.0 {
                let mod_chk = eval_frequency_ratios_with_trimming(
                    &model_freqs_corr[0],
                    &ctrl.freq_sigma[0],
                    &model_freqs_corr[1],
                    &ctrl.freq_sigma[1],
                    dnu_obs,
                    true,
                    0,
                    0,
                    0,
                );
                if obs_r.n == 0
                    || mod_chk.n != obs_r.n
                    || mod_chk.l0_first != obs_r.l0_first
                    || mod_chk.l1_first != obs_r.l1_first
                {
                    mode_match_failed = true;
                } else {
                    let mod_r = eval_frequency_ratios_with_trimming(
                        &model_freqs_corr[0],
                        &ctrl.freq_sigma[0],
                        &model_freqs_corr[1],
                        &ctrl.freq_sigma[1],
                        dnu_obs,
                        false,
                        obs_r.n,
                        obs_r.l0_first,
                        obs_r.l1_first,
                    );
                    let l0_mod_axis = &model_freqs[0]
                        [obs_r.l0_first + 1..=obs_r.l0_first + obs_r.n];
                    let l1_mod_axis = &model_freqs[1]
                        [obs_r.l1_first + 1..=obs_r.l1_first + obs_r.n];
                    obs_r01.extend_from_slice(&obs_r.ratios_r01);
                    obs_r01_sigmas.extend_from_slice(&obs_r.sigmas_r01);
                    obs_r10.extend_from_slice(&obs_r.ratios_r10);
                    obs_r10_sigmas.extend_from_slice(&obs_r.sigmas_r10);
                    for k in 0..obs_r.n {
                        let f0_obs =
                            ctrl.freq_target[0][obs_r.l0_first + 1 + k];
                        let f1_obs =
                            ctrl.freq_target[1][obs_r.l1_first + 1 + k];
                        let mr01 = interpolate_ratio_r010(
                            f0_obs,
                            l0_mod_axis,
                            &mod_r.ratios_r01,
                        );
                        let mr10 = interpolate_ratio_r010(
                            f1_obs,
                            l1_mod_axis,
                            &mod_r.ratios_r10,
                        );
                        if ctrl.trace_chi2_seismo_ratios_info {
                            println!(
                                "r01_obs={:.6} r01_mod={:.6} r10_obs={:.6} r10_mod={:.6}",
                                obs_r.ratios_r01[k],
                                mr01,
                                obs_r.ratios_r10[k],
                                mr10
                            );
                        }
                        model_r01.push(mr01);
                        model_r10.push(mr10);
                        r010_obs.push(obs_r.ratios_r01[k]);
                        r010_obs.push(obs_r.ratios_r10[k]);
                        r010_sig.push(obs_r.sigmas_r01[k]);
                        r010_sig.push(obs_r.sigmas_r10[k]);
                        r010_mod.push(mr01);
                        r010_mod.push(mr10);
                    }
                }
            }
        }
        if !mode_match_failed && f_r02 > 0.0 && ctrl.nl[2] > 0 {
            if ctrl.nl[0] == 0 || ctrl.nl[1] == 0 || ratios_n == 0 {
                mode_match_failed = true;
            } else {
                let obs_r2 = eval_r02_obs_with_indices(
                    &ctrl.freq_target[0],
                    &ctrl.freq_sigma[0],
                    &ctrl.freq_target[1],
                    &ctrl.freq_sigma[1],
                    &ctrl.freq_target[2],
                    &ctrl.freq_sigma[2],
                    ratios_n,
                    ratios_l0_first,
                    ratios_l1_first,
                );
                let (rm, _) = eval_r02_model_with_indices(
                    &model_freqs_corr[0],
                    &model_freqs_corr[1],
                    &model_freqs_corr[2],
                    ratios_l0_first,
                    ratios_l1_first,
                    &obs_r2,
                );
                let (_, fm) = eval_r02_model_with_indices(
                    &model_freqs[0],
                    &model_freqs_corr[1],
                    &model_freqs_corr[2],
                    ratios_l0_first,
                    ratios_l1_first,
                    &obs_r2,
                );
                for k in 0..obs_r2.ratios_r02.len() {
                    let s = obs_r2.sigmas_r02[k];
                    if s > 0.0 && !rm.is_empty() {
                        let val_m = interpolate_ratio_r02(
                            obs_r2.r02_freqs[k],
                            &fm,
                            &rm,
                        );
                        if ctrl.trace_chi2_seismo_ratios_info {
                            println!(
                                "r02_obs={:.6} r02_mod={:.6} sigma={:.6}",
                                obs_r2.ratios_r02[k], val_m, s
                            );
                        }
                        r02_obs_vec.push(obs_r2.ratios_r02[k]);
                        r02_sig_vec.push(s);
                        r02_mod_vec.push(val_m);
                        if k < ctrl.nl[0] {
                            obs_r02[k] = obs_r2.ratios_r02[k];
                            obs_r02_sigmas[k] = s;
                            model_r02[k] = val_m;
                        }
                    }
                }
                if r02_obs_vec.is_empty() {
                    mode_match_failed = true;
                }
            }
        }
    }

    if mode_match_failed {
        let cr_fail = if pre_chi2_radial > 0.0 {
            pre_chi2_radial
        } else {
            1.0e99
        };
        return Ok(AsteroSampleResult {
            sample_number,
            params,
            model_number: model.model_number,
            age_yr: model.star_age,
            teff,
            logg,
            log_l,
            log_r,
            constraint_values: *constraint_values,
            delta_nu_uhz,
            nu_max_uhz,
            a_div_r,
            correction_r,
            chi2: 1.0e99,
            chi2_seismo: 1.0e99,
            chi2_spectro: 1.0e99,
            chi2_radial: cr_fail,
            chi2_r010: 1.0e99,
            chi2_r02: 1.0e99,
            model_freqs,
            model_freqs_corr,
            model_inertias,
            model_orders,
            surf_coef1,
            surf_coef2,
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
            model_freqs_alt_up,
            model_freqs_corr_alt_up,
            model_inertias_alt_up,
            model_orders_alt_up,
            model_freqs_alt_down,
            model_freqs_corr_alt_down,
            model_inertias_alt_down,
            model_orders_alt_down,
        });
    }

    let (mut all_obs, mut all_sig, mut all_corr) =
        (Vec::new(), Vec::new(), Vec::new());
    for l in 0..4 {
        for i in 0..model_freqs_corr[l].len() {
            if let (Some(&f_t), Some(&f_s)) =
                (ctrl.freq_target[l].get(i), ctrl.freq_sigma[l].get(i))
            {
                if f_t >= 0.0 && f_s > 0.0 {
                    if ctrl.trace_chi2_seismo_frequencies_info {
                        let chi2term = flexmesa_math::pow2(
                            (model_freqs_corr[l][i] - f_t) / f_s,
                        );
                        println!(
                            "{:6}{:6}{:6}{:6}{}{}{}{}{}{}",
                            model.model_number,
                            i + 1,
                            l,
                            model_orders[l].get(i).copied().unwrap_or(0),
                            fortran_1pe(chi2term, 20, 10, None),
                            fortran_1pe(model_freqs[l][i], 20, 10, None),
                            fortran_1pe(model_freqs_corr[l][i], 20, 10, None),
                            fortran_1pe(f_t, 20, 10, None),
                            fortran_1pe(f_s, 20, 10, None),
                            fortran_1pe(
                                flexmesa_math::log10(
                                    model_inertias[l]
                                        .get(i)
                                        .copied()
                                        .unwrap_or(1.0e-30)
                                        .max(1.0e-99),
                                ),
                                20,
                                10,
                                None,
                            ),
                        );
                    }
                    all_obs.push(f_t);
                    all_sig.push(f_s);
                    all_corr.push(model_freqs_corr[l][i]);
                }
            }
        }
    }

    let (chi2, chi2_seismo, chi2_spectro, mut chi2_radial, chi2_r010, chi2_r02) =
        eval_astero_chi2(
            &ctrl.weights,
            &all_obs,
            &all_sig,
            &all_corr,
            &r010_obs,
            &r010_sig,
            &r010_mod,
            &r02_obs_vec,
            &r02_sig_vec,
            &r02_mod_vec,
            ctrl.delta_nu,
            ctrl.delta_nu_sigma,
            delta_nu_uhz,
            ctrl.nu_max,
            ctrl.nu_max_sigma,
            nu_max_uhz,
            &sp_obs,
            &sp_sig,
            &sp_mod,
        );
    if !model_freqs_corr[0].is_empty() {
        chi2_radial = eval_radial_chi2(ctrl, &model_freqs_corr[0]);
    }
    if ctrl.trace_chi2_spectro_info {
        for i in 0..sp_obs.len() {
            let d = (sp_mod[i] - sp_obs[i]) / sp_sig[i];
            println!(
                "chi2_spectro_info {} val={:.6} obs={:.6} sigma={:.6} chi2term={:.6}",
                sp_names[i], sp_mod[i], sp_obs[i], sp_sig[i], d * d
            );
        }
        println!("chi2_spectro={chi2_spectro:.6}");
    }
    if ctrl.trace_chi2_seismo_delta_nu_info && ctrl.delta_nu_sigma > 0.0 {
        let chi2_dnu = flexmesa_math::pow2(
            (ctrl.delta_nu - delta_nu_uhz) / ctrl.delta_nu_sigma,
        );
        println!(
            "chi2_delta_nu={chi2_dnu:.6} delta_nu_model={delta_nu_uhz:.6} delta_nu_obs={:.6}",
            ctrl.delta_nu
        );
    }
    if ctrl.trace_chi2_seismo_nu_max_info && ctrl.nu_max_sigma > 0.0 {
        let chi2_numax =
            flexmesa_math::pow2((ctrl.nu_max - nu_max_uhz) / ctrl.nu_max_sigma);
        println!(
            "chi2_nu_max={chi2_numax:.6} nu_max_model={nu_max_uhz:.6} nu_max_obs={:.6}",
            ctrl.nu_max
        );
    }

    Ok(AsteroSampleResult {
        sample_number,
        params,
        model_number: model.model_number,
        age_yr: model.star_age,
        teff,
        logg,
        log_l,
        log_r,
        constraint_values: *constraint_values,
        delta_nu_uhz,
        nu_max_uhz,
        a_div_r,
        correction_r,
        chi2,
        chi2_seismo,
        chi2_spectro,
        chi2_radial,
        chi2_r010,
        chi2_r02,
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
        model_freqs_alt_up,
        model_freqs_corr_alt_up,
        model_inertias_alt_up,
        model_orders_alt_up,
        model_freqs_alt_down,
        model_freqs_corr_alt_down,
        model_inertias_alt_down,
        model_orders_alt_down,
    })
}

#[allow(clippy::too_many_arguments)]
fn eval_alt_surface_corr(
    ctrl: &AsteroSearchControls,
    model_freqs: &[Vec<f64>; 4],
    model_inertias: &[Vec<f64>; 4],
    alt_freqs: &[Vec<f64>; 4],
    alt_inertias: &[Vec<f64>; 4],
    all_obs: &[f64],
    all_sig: &[f64],
    dnu_obs: f64,
    numax_obs: f64,
    nu_max_uhz: f64,
    model_nu_max_sun: f64,
    model_delta_nu_sun: f64,
) -> Result<[Vec<f64>; 4], StarError> {
    let mut alt_mod = Vec::new();
    let mut alt_in = Vec::new();
    for l in 0..4 {
        for i in 0..model_freqs[l].len() {
            let af = alt_freqs[l].get(i).copied().unwrap_or(0.0);
            let ai = alt_inertias[l].get(i).copied().unwrap_or(0.0);
            if l > 0 && af > 0.0 && ai > 0.0 {
                alt_mod.push(af);
                alt_in.push(ai);
            } else {
                alt_mod.push(model_freqs[l][i]);
                alt_in.push(model_inertias[l][i]);
            }
        }
    }
    let mut dnu_scratch = 0.0;
    let sc: SurfaceCorrOutput = apply_surface_corr(
        ctrl,
        &model_freqs[0],
        &model_inertias[0],
        all_obs,
        all_sig,
        &alt_mod,
        &alt_in,
        dnu_obs,
        numax_obs,
        nu_max_uhz,
        model_nu_max_sun,
        model_delta_nu_sun,
        &mut dnu_scratch,
    )?;
    let mut out: [Vec<f64>; 4] = Default::default();
    let mut offset = 0usize;
    for l in 0..4 {
        let n = model_freqs[l].len();
        out[l] = sc.corrected[offset..offset + n].to_vec();
        offset += n;
    }
    Ok(out)
}

/// Triggers oscillation mode scanning at `save_mode_model_number` when
/// `chi2_seismo_fraction <= 0.0` skipped normal frequency evaluation
/// (`astero/private/extras_support.f90:910-912`).
pub(crate) fn scan_and_save_mode_at_number(
    model: &StarModel,
    ctrl: &AsteroSearchControls,
    case_dir: &Path,
    output_root: &Path,
) -> Result<(), StarError> {
    if ctrl.save_mode_model_number <= 0
        || model.model_number.cast_signed() != ctrl.save_mode_model_number
    {
        return Ok(());
    }
    let (delta_nu_uhz, _) =
        eval_model_delta_nu_and_nu_max(model, model.teff.max(1.0));
    let dnu_obs = if ctrl.delta_nu > 0.0 {
        ctrl.delta_nu
    } else {
        delta_nu_uhz
    };
    let mut scanner = DegreeScanner::init(model, ctrl, case_dir, output_root)?;
    for l in 0..4 {
        if ctrl.nl[l] == 0 {
            continue;
        }
        let _ = scanner.scan_degree(model, ctrl, l, dnu_obs, delta_nu_uhz)?;
    }
    Ok(())
}

struct MatchedDegreeWithAlt {
    freqs: Vec<f64>,
    inertias: Vec<f64>,
    orders: Vec<isize>,
    freqs_alt_up: Vec<f64>,
    inertias_alt_up: Vec<f64>,
    orders_alt_up: Vec<isize>,
    freqs_alt_down: Vec<f64>,
    inertias_alt_down: Vec<f64>,
    orders_alt_down: Vec<isize>,
}

enum DegreeScanner {
    Adipls(Box<AsteroAdipls>),
    Gyre {
        outer_bc: OuterBoundKind,
        gyre_driver: Option<Box<AsteroGyre>>,
        stored_for_gyre: bool,
        output_root: std::path::PathBuf,
    },
}

impl DegreeScanner {
    fn init(
        model: &StarModel,
        ctrl: &AsteroSearchControls,
        case_dir: &Path,
        output_root: &Path,
    ) -> Result<Self, StarError> {
        let osc_trimmed = ctrl.oscillation_code.trim();
        if osc_trimmed.eq_ignore_ascii_case("adipls") {
            let mut drv = AsteroAdipls::init_with_redistribute(
                case_dir,
                ctrl,
                ctrl.do_redistribute_mesh,
            )?;
            drv.set_output_root(output_root, &ctrl.astero_results_directory);
            Ok(Self::Adipls(Box::new(drv)))
        } else if osc_trimmed.eq_ignore_ascii_case("gyre") {
            let outer_bc =
                parse_gyre_outer_bound(case_dir, &ctrl.gyre_input_file)?;
            let mesa_dir =
                std::env::var("MESA_DIR").unwrap_or_else(|_| String::from("."));
            let mut gyre_driver =
                match AsteroGyre::init(case_dir, ctrl, &mesa_dir) {
                    Ok(mut drv) => {
                        drv.set_output_root(
                            output_root,
                            &ctrl.astero_results_directory,
                        );
                        drv.store_model(model)?;
                        Some(Box::new(drv))
                    }
                    Err(_) => None,
                };
            let stored_for_gyre = gyre_driver.is_some();
            if !stored_for_gyre
                && (ctrl.write_fgong_for_each_model
                    || ctrl.write_gyre_for_each_model)
            {
                write_per_model_pulse_files(model, ctrl, output_root)?;
                gyre_driver = None;
            }
            Ok(Self::Gyre {
                outer_bc,
                gyre_driver,
                stored_for_gyre,
                output_root: output_root.to_path_buf(),
            })
        } else {
            Err(StarError::InvalidInput(format!(
                "unrecognized oscillation_code '{}'",
                ctrl.oscillation_code
            )))
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn scan_degree(
        &mut self,
        model: &StarModel,
        ctrl: &AsteroSearchControls,
        l: usize,
        dnu_obs: f64,
        delta_nu_model: f64,
    ) -> Result<Option<MatchedDegreeWithAlt>, StarError> {
        let (nu_lo, nu_hi) = scan_bounds_from_targets(
            &ctrl.freq_target[l],
            ctrl.nu_lower_factor,
            ctrl.nu_upper_factor,
            dnu_obs,
        );
        match self {
            Self::Adipls(driver) => {
                let raw_iscan =
                    (ctrl.iscan_factor[l] * (ctrl.nl[l] as f64)) as i32;
                let iscan = raw_iscan.max(1);
                let store_now = driver.stored_model().is_none();
                let raw_modes = driver.get_one_el_info(
                    model,
                    l,
                    nu_lo,
                    nu_hi,
                    iscan,
                    delta_nu_model,
                    store_now,
                )?;
                if ctrl.trace_time_in_oscillation_code {
                    if let Some(dt) = driver.last_oscillation_elapsed_sec() {
                        println!(
                            "adipls time_in_oscillation_code l={l} elapsed_s={dt:.6}"
                        );
                    }
                }
                Ok(match_osc_results_with_alt(
                    &ctrl.freq_target[l],
                    &raw_modes,
                    l,
                ))
            }
            Self::Gyre {
                outer_bc,
                gyre_driver,
                stored_for_gyre,
                output_root,
            } => {
                let el = i32::try_from(l).unwrap_or(0);
                // Run GYRE unconditionally (`astero/private/gyre_support.f90:75-118`,
                // without gating on `gyre_non_ad`).
                if let Some(drv) = gyre_driver.as_mut() {
                    let store_now = !*stored_for_gyre;
                    match drv.get_modes(model, el, store_now) {
                        Ok(raw_modes) => {
                            *stored_for_gyre = true;
                            if ctrl.trace_time_in_oscillation_code {
                                if let Some(dt) =
                                    drv.last_oscillation_elapsed_sec()
                                {
                                    println!(
                                        "gyre time_in_oscillation_code l={l} elapsed_s={dt:.6}"
                                    );
                                }
                            }
                            return Ok(match_osc_results_with_alt(
                                &ctrl.freq_target[l],
                                &raw_modes,
                                l,
                            ));
                        }
                        Err(e) if e.to_string().contains("not ported") => {}
                        Err(e) => return Err(e),
                    }
                }
                let t0 = Instant::now();
                let n_scan = ((2.0
                    * (ctrl.nl[l] as f64)
                    * ctrl.iscan_factor[l].max(1.0))
                    as usize)
                    .clamp(240, 600);
                let opts = PulseModeScanOptions {
                    l,
                    freq_min_uhz: nu_lo,
                    freq_max_uhz: nu_hi,
                    n_freq: n_scan,
                    add_center_point: ctrl.add_center_point,
                    keep_surface_point: ctrl.keep_surface_point,
                    add_atmosphere: ctrl.add_atmosphere,
                    outer_bc: *outer_bc,
                    inertia_norm: InertiaNormKind::Both,
                };
                let raw_modes = scan_star_bvp_eigenmodes(model, &opts);
                if ctrl.trace_time_in_oscillation_code {
                    let dt = t0.elapsed().as_secs_f64();
                    println!(
                        "gyre time_in_oscillation_code l={l} elapsed_s={dt:.6}"
                    );
                }
                if ctrl.save_mode_model_number > 0
                    && model.model_number.cast_signed()
                        == ctrl.save_mode_model_number
                    && el == ctrl.el_to_save
                {
                    save_bvp_mode_if_requested(
                        output_root,
                        ctrl,
                        el,
                        &raw_modes,
                    )?;
                }
                Ok(match_degree_modes_with_alt(
                    &ctrl.freq_target[l],
                    &raw_modes,
                    l,
                ))
            }
        }
    }
}

fn write_per_model_pulse_files(
    model: &StarModel,
    ctrl: &AsteroSearchControls,
    output_root: &Path,
) -> Result<(), StarError> {
    let dir = output_root.join(&ctrl.astero_results_directory);
    fs::create_dir_all(&dir).map_err(|e| {
        StarError::IoError(format!(
            "failed creating astero_results_directory '{}': {e}",
            dir.display()
        ))
    })?;
    let num_str = format!(
        "{:0width$}",
        model.model_number,
        width = ctrl.model_num_digits
    );
    if ctrl.write_fgong_for_each_model {
        let pulse_in =
            flexmesa_star::pulse_fgong::PulseDataInput::from_model(model)
                .map_err(|e| StarError::InvalidInput(e.to_string()))?;
        let (fg_g, fg_p) = flexmesa_star::pulse_fgong::get_fgong_data(
            &pulse_in,
            ctrl.add_center_point,
            ctrl.keep_surface_point,
            ctrl.add_atmosphere,
        )
        .map_err(|e| StarError::InvalidInput(e.to_string()))?;
        let fg_text = flexmesa_star::pulse_write::write_fgong_data(
            &fg_g,
            &fg_p,
            &flexmesa_star::pulse::default_fgong_header(),
            model.controls.fgong_ivers,
        )
        .map_err(|e| StarError::InvalidInput(e.to_string()))?;
        let path = dir.join(format!(
            "{}{}{}",
            ctrl.fgong_prefix.trim(),
            num_str,
            ctrl.fgong_postfix.trim()
        ));
        fs::write(&path, fg_text).map_err(|e| {
            StarError::IoError(format!(
                "failed writing FGONG '{}': {e}",
                path.display()
            ))
        })?;
    }
    if ctrl.write_gyre_for_each_model {
        let (gd, pd) =
            flexmesa_star::pulse_gyre_data::star_get_pulse_data_gyre(
                model,
                ctrl.add_center_point,
                ctrl.keep_surface_point,
                ctrl.add_atmosphere,
            )
            .map_err(|e| StarError::InvalidInput(e.to_string()))?;
        let text = flexmesa_star::pulse_write::write_gyre_data(
            &gd,
            &pd,
            model.controls.gyre_data_schema,
        )
        .map_err(|e| StarError::InvalidInput(e.to_string()))?;
        let path = dir.join(format!(
            "{}{}{}",
            ctrl.gyre_prefix.trim(),
            num_str,
            ctrl.gyre_postfix.trim()
        ));
        fs::write(&path, text).map_err(|e| {
            StarError::IoError(format!(
                "failed writing GYRE '{}': {e}",
                path.display()
            ))
        })?;
    }
    Ok(())
}

fn save_bvp_mode_if_requested(
    output_root: &Path,
    ctrl: &AsteroSearchControls,
    el: i32,
    modes: &[BvpEigenmode],
) -> Result<(), StarError> {
    if let Some(m) = modes.iter().find(|m| m.n_pg == ctrl.order_to_save) {
        let rel = ctrl.save_mode_filename.trim();
        let rel = if rel.is_empty() {
            "save_mode.data"
        } else {
            rel
        };
        let path = if Path::new(rel).is_absolute() {
            std::path::PathBuf::from(rel)
        } else {
            output_root.join(rel)
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                StarError::IoError(format!(
                    "failed creating save_mode_filename dir '{}': {e}",
                    parent.display()
                ))
            })?;
        }
        let mut buf = String::new();
        let _ = writeln!(
            buf,
            "{:>8}{:>8}{:>8}{:>20}{:>20}{:>20}",
            "el", "order", "em", "freq (microHz)", "inertia", "growth rate (s)"
        );
        let _ = writeln!(
            buf,
            "{:8}{:8}{:8}{:20.10}{}{}",
            el,
            m.n_pg,
            0,
            m.freq_uhz,
            fortran_1pe(m.e_norm, 20, 10, None),
            fortran_1pe(0.0, 20, 10, None),
        );
        fs::write(&path, buf).map_err(|e| {
            StarError::IoError(format!(
                "failed writing save_mode_filename '{}': {e}",
                path.display()
            ))
        })?;
    }
    Ok(())
}

/// Sorts mode outputs by ascending frequency and greedily matches each target
/// monotonically with next-lower/higher neighbours (`qsort` and `set_to_closest`,
/// `astero/private/astero_support.f90:148-180, 223-313`).
fn match_osc_results_with_alt(
    targets: &[f64],
    modes: &[OscResult],
    l: usize,
) -> Option<MatchedDegreeWithAlt> {
    let mut order: Vec<usize> = (0..modes.len()).collect();
    order.sort_by(|&a, &b| {
        modes[a].cyclic_freq.total_cmp(&modes[b].cyclic_freq)
    });
    let mut out = MatchedDegreeWithAlt {
        freqs: Vec::with_capacity(targets.len()),
        inertias: Vec::with_capacity(targets.len()),
        orders: Vec::with_capacity(targets.len()),
        freqs_alt_up: vec![0.0; targets.len()],
        inertias_alt_up: vec![0.0; targets.len()],
        orders_alt_up: vec![0; targets.len()],
        freqs_alt_down: vec![0.0; targets.len()],
        inertias_alt_down: vec![0.0; targets.len()],
        orders_alt_down: vec![0; targets.len()],
    };
    let mut jprev: Option<usize> = None;

    for (idx, &target_nu) in targets.iter().enumerate() {
        let start = jprev.map_or(0, |jp| jp + 1);
        if start >= order.len() {
            return None;
        }
        let mut best_j = start;
        let mut best_d = (modes[order[start]].cyclic_freq - target_nu).abs();
        for (j, &m_idx) in order.iter().enumerate().skip(start) {
            let m = &modes[m_idx];
            let d = (m.cyclic_freq - target_nu).abs();
            if d < best_d {
                best_d = d;
                best_j = j;
            }
            if m.cyclic_freq > target_nu {
                break;
            }
        }
        let chosen = &modes[order[best_j]];
        out.freqs.push(chosen.cyclic_freq);
        out.inertias.push(chosen.inertia.max(1.0e-30));
        out.orders.push(isize::try_from(chosen.order).unwrap_or(0));
        if l > 0 {
            if best_j > 0 {
                let md = &modes[order[best_j - 1]];
                out.freqs_alt_down[idx] = md.cyclic_freq;
                out.inertias_alt_down[idx] = md.inertia.max(1.0e-30);
                out.orders_alt_down[idx] =
                    isize::try_from(md.order).unwrap_or(0);
            }
            if best_j + 1 < order.len() {
                let mu = &modes[order[best_j + 1]];
                out.freqs_alt_up[idx] = mu.cyclic_freq;
                out.inertias_alt_up[idx] = mu.inertia.max(1.0e-30);
                out.orders_alt_up[idx] = isize::try_from(mu.order).unwrap_or(0);
            }
        }
        jprev = Some(best_j);
    }
    Some(out)
}

fn match_degree_modes_with_alt(
    targets: &[f64],
    modes: &[BvpEigenmode],
    l: usize,
) -> Option<MatchedDegreeWithAlt> {
    let mut out = MatchedDegreeWithAlt {
        freqs: Vec::with_capacity(targets.len()),
        inertias: Vec::with_capacity(targets.len()),
        orders: Vec::with_capacity(targets.len()),
        freqs_alt_up: vec![0.0; targets.len()],
        inertias_alt_up: vec![0.0; targets.len()],
        orders_alt_up: vec![0; targets.len()],
        freqs_alt_down: vec![0.0; targets.len()],
        inertias_alt_down: vec![0.0; targets.len()],
        orders_alt_down: vec![0; targets.len()],
    };
    let mut jprev: Option<usize> = None;

    for (idx, &target_nu) in targets.iter().enumerate() {
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
        out.freqs.push(modes[best_j].freq_uhz);
        out.inertias.push(modes[best_j].e_norm.max(1.0e-30));
        out.orders
            .push(isize::try_from(modes[best_j].n_pg).unwrap_or(0));
        if l > 0 {
            if best_j > 0 {
                let md = &modes[best_j - 1];
                out.freqs_alt_down[idx] = md.freq_uhz;
                out.inertias_alt_down[idx] = md.e_norm.max(1.0e-30);
                out.orders_alt_down[idx] =
                    isize::try_from(md.n_pg).unwrap_or(0);
            }
            if best_j + 1 < modes.len() {
                let mu = &modes[best_j + 1];
                out.freqs_alt_up[idx] = mu.freq_uhz;
                out.inertias_alt_up[idx] = mu.e_norm.max(1.0e-30);
                out.orders_alt_up[idx] = isize::try_from(mu.n_pg).unwrap_or(0);
            }
        }
        jprev = Some(best_j);
    }
    Some(out)
}
