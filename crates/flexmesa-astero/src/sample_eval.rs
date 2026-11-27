// SPDX-License-Identifier: GPL-3.0-or-later
//! Per-sample spectroscopic and asteroseismic observable evaluation, pulsation
//! mode matching, surface-effect correction, and composite $\chi^2$ scoring
//! (`astero/private/extras_support.f90:155-485`, `astero_support.f90:223-760`).

use std::path::Path;

use flexmesa_const::RSUN;
use flexmesa_star::controls::StarError;
use flexmesa_star::eval_photosphere_info;
use flexmesa_star::history::extract_history_row;
use flexmesa_star::model::StarModel;
use flexmesa_star::pulse_modes::scan_bounds_from_targets;
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
    eval_acoustic_profile, eval_delta_nu_asymptotic,
    eval_delta_nu_radius_scaling, eval_nu_max,
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
    /// Primary surface-correction coefficient (`surf_coef1`, `astero_support.f90:1280-1333`).
    pub surf_coef1: f64,
    /// Secondary surface-correction coefficient (`surf_coef2`, `astero_support.f90:1280-1333`).
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
    if ctrl.include_age_in_chi2_spectro {
        obs.push(ctrl.age_target);
        sig.push(ctrl.age_sigma);
        mval.push(model.star_age);
        names.push(String::from("age"));
    }
    for (i, c) in ctrl.indexed_constraints.iter().enumerate() {
        if c.include {
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

fn eval_model_delta_nu_and_nu_max(model: &StarModel, teff: f64) -> (f64, f64) {
    let r_phot_cm = model.surface_r_rsun().max(1.0e-30) * RSUN;
    let phot = eval_photosphere_info(model);
    let o = &model.controls.output;
    let delta_nu_uhz = if o.get_delta_nu_from_scaled_solar
        && phot.l_lsun > 0.0
        && o.astero_teff_sun > 0.0
    {
        let star_mass = (model.mstar / flexmesa_const::MSUN).max(0.0);
        o.delta_nu_sun
            * star_mass.sqrt()
            * flexmesa_math::pow3(teff / o.astero_teff_sun)
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
            eval_delta_nu_radius_scaling(model.mstar, r_phot_cm)
        }
    };
    let nu_max_uhz = eval_nu_max(model.mstar, r_phot_cm, teff);
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
    let cnt_sp = sp_obs.len();
    for i in 0..cnt_sp {
        sum_sp += flexmesa_math::pow2((sp_mod[i] - sp_obs[i]) / sp_sig[i]);
    }
    let chi2_spectro = if ctrl.weights.normalize_chi2_spectro {
        sum_sp / (cnt_sp as f64)
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

/// Evaluates spectroscopic and asteroseismic $\chi^2$ on `model` (`extras_support.f90:155-485`).
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
    let (sp_obs, sp_sig, sp_mod, sp_names, teff, logg, log_l, log_r) =
        eval_spectro_obs(model, ctrl, constraint_values);
    let (mut delta_nu_uhz, nu_max_uhz) =
        eval_model_delta_nu_and_nu_max(model, teff);
    let dnu_obs = if ctrl.delta_nu > 0.0 {
        ctrl.delta_nu
    } else {
        delta_nu_uhz
    };
    let numax_obs = if ctrl.nu_max > 0.0 {
        ctrl.nu_max
    } else {
        nu_max_uhz
    };
    let model_nu_max_sun = if model.controls.output.nu_max_sun > 0.0 {
        model.controls.output.nu_max_sun
    } else {
        ctrl.solar_nu_max
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

    let need_modes = ctrl.nl.iter().sum::<usize>() > 0
        && (ctrl.weights.chi2_seismo_fraction > 0.0 || sp_obs.is_empty());
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

    if !mode_match_failed
        && ctrl.nl[0] > 0
        && ctrl.nl[1] > 0
        && ctrl.weights.chi2_seismo_r_010_fraction > 0.0
    {
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
        )?;
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
        )?;
        if obs_r.n == 0
            || mod_chk.n != obs_r.n
            || mod_chk.l0_first != obs_r.l0_first
            || mod_chk.l1_first != obs_r.l1_first
        {
            mode_match_failed = true;
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

    let mut ratios_n = 0usize;
    let mut ratios_l0_first = 0usize;
    let mut ratios_l1_first = 0usize;
    let (mut r010_obs, mut r010_sig, mut r010_mod) =
        (Vec::new(), Vec::new(), Vec::new());
    let (mut obs_r01, mut obs_r01_sigmas, mut model_r01) =
        (Vec::new(), Vec::new(), Vec::new());
    let (mut obs_r10, mut obs_r10_sigmas, mut model_r10) =
        (Vec::new(), Vec::new(), Vec::new());
    if (ctrl.weights.chi2_seismo_r_010_fraction > 0.0
        || ctrl.weights.chi2_seismo_r_02_fraction > 0.0)
        && ctrl.nl[0] > 0
        && ctrl.nl[1] > 0
    {
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
        )?;
        ratios_n = obs_r.n;
        ratios_l0_first = obs_r.l0_first;
        ratios_l1_first = obs_r.l1_first;
        if obs_r.n > 0 && ctrl.weights.chi2_seismo_r_010_fraction > 0.0 {
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
            )?;
            let l0_mod_axis =
                &model_freqs[0][obs_r.l0_first + 1..=obs_r.l0_first + obs_r.n];
            let l1_mod_axis =
                &model_freqs[1][obs_r.l1_first + 1..=obs_r.l1_first + obs_r.n];
            obs_r01.extend_from_slice(&obs_r.ratios_r01);
            obs_r01_sigmas.extend_from_slice(&obs_r.sigmas_r01);
            obs_r10.extend_from_slice(&obs_r.ratios_r10);
            obs_r10_sigmas.extend_from_slice(&obs_r.sigmas_r10);
            for k in 0..obs_r.n {
                let f0_obs = ctrl.freq_target[0][obs_r.l0_first + 1 + k];
                let f1_obs = ctrl.freq_target[1][obs_r.l1_first + 1 + k];
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
                        obs_r.ratios_r01[k], mr01, obs_r.ratios_r10[k], mr10
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

    let (mut r02_obs_vec, mut r02_sig_vec, mut r02_mod_vec) =
        (Vec::new(), Vec::new(), Vec::new());
    let mut obs_r02 = vec![0.0_f64; ctrl.nl[0]];
    let mut obs_r02_sigmas = vec![0.0_f64; ctrl.nl[0]];
    let mut model_r02 = vec![0.0_f64; ctrl.nl[0]];
    if ctrl.weights.chi2_seismo_r_02_fraction > 0.0
        && ctrl.nl[0] > 0
        && ctrl.nl[1] > 0
        && ctrl.nl[2] > 0
        && ratios_n > 0
    {
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
        let (rm, fm) = eval_r02_model_with_indices(
            &model_freqs_corr[0],
            &model_freqs_corr[1],
            &model_freqs_corr[2],
            ratios_l0_first,
            ratios_l1_first,
            &obs_r2,
        );
        for k in 0..obs_r2.ratios_r02.len() {
            let s = obs_r2.sigmas_r02[k];
            if s > 0.0 && !rm.is_empty() {
                let val_m =
                    interpolate_ratio_r02(obs_r2.r02_freqs[k], &fm, &rm);
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
            if sp_sig[i] > 0.0 {
                let d = (sp_mod[i] - sp_obs[i]) / sp_sig[i];
                println!(
                    "chi2_spectro_info {} val={:.6} obs={:.6} sigma={:.6} chi2term={:.6}",
                    sp_names[i], sp_mod[i], sp_obs[i], sp_sig[i], d * d
                );
            }
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
    Gyre(Box<AsteroGyre>),
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
            // Upstream declares `gyre_non_ad` (`astero_def.f90:256,394`,
            // `astero_search.defaults:1342`) without reading it in
            // `astero_support.f90` or `gyre_support.f90`, so accept and ignore
            // it here rather than inventing a rejection.
            if ctrl.gyre_input_file.trim().is_empty() {
                return Err(StarError::InvalidInput(String::from(
                    "oscillation_code = 'gyre' requires a non-empty \
                     gyre_input_file",
                )));
            }
            let mesa_dir = std::env::var("MESA_DIR")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| String::from("."));
            let mut drv = AsteroGyre::init(case_dir, ctrl, &mesa_dir)?;
            drv.set_output_root(output_root, &ctrl.astero_results_directory);
            drv.store_model(model)?;
            Ok(Self::Gyre(Box::new(drv)))
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
        match self {
            Self::Adipls(driver) => {
                let (nu_lo, nu_hi) = scan_bounds_from_targets(
                    &ctrl.freq_target[l],
                    ctrl.nu_lower_factor,
                    ctrl.nu_upper_factor,
                    dnu_obs,
                );
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
            Self::Gyre(driver) => {
                let el = i32::try_from(l).map_err(|_| {
                    StarError::InvalidInput(format!(
                        "degree l={l} overflows i32"
                    ))
                })?;
                let store_now = driver.gyre.model().is_none();
                let raw_modes = driver.get_modes(model, el, store_now)?;
                if ctrl.trace_time_in_oscillation_code {
                    if let Some(dt) = driver.last_oscillation_elapsed_sec() {
                        println!(
                            "gyre time_in_oscillation_code l={l} elapsed_s={dt:.6}"
                        );
                    }
                }
                Ok(match_osc_results_with_alt(
                    &ctrl.freq_target[l],
                    &raw_modes,
                    l,
                ))
            }
        }
    }
}

fn match_osc_results_with_alt(
    targets: &[f64],
    modes: &[OscResult],
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
        let mut best_d = (modes[start].cyclic_freq - target_nu).abs();
        for (j, m) in modes.iter().enumerate().skip(start) {
            let d = (m.cyclic_freq - target_nu).abs();
            if d < best_d {
                best_d = d;
                best_j = j;
            }
            if m.cyclic_freq > target_nu {
                break;
            }
        }
        out.freqs.push(modes[best_j].cyclic_freq);
        out.inertias.push(modes[best_j].inertia.max(1.0e-30));
        out.orders
            .push(isize::try_from(modes[best_j].order).unwrap_or(0));
        if l > 0 {
            if best_j > 0 {
                let md = &modes[best_j - 1];
                out.freqs_alt_down[idx] = md.cyclic_freq;
                out.inertias_alt_down[idx] = md.inertia.max(1.0e-30);
                out.orders_alt_down[idx] =
                    isize::try_from(md.order).unwrap_or(0);
            }
            if best_j + 1 < modes.len() {
                let mu = &modes[best_j + 1];
                out.freqs_alt_up[idx] = mu.cyclic_freq;
                out.inertias_alt_up[idx] = mu.inertia.max(1.0e-30);
                out.orders_alt_up[idx] = isize::try_from(mu.order).unwrap_or(0);
            }
        }
        jprev = Some(best_j);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use flexmesa_star::controls::{
        AtmOption, AtmTTauOpacity, AtmTTauRelation, StarControls, StarError,
        StarJobControls,
    };
    use flexmesa_star::model::StarModel;
    use flexmesa_star::{load_zams_model, set_hydro_vars};

    use super::{
        eval_model_astero_sample, eval_spectro_and_delta_nu_chi2,
        scan_and_save_mode_at_number,
    };
    use crate::inlist_astero::{AsteroSearchControls, MAX_CONSTRAINTS};

    static TEST_DIR_SEQ: AtomicUsize = AtomicUsize::new(1);

    struct TempDirGuard {
        path: PathBuf,
    }

    impl TempDirGuard {
        fn new(tag: &str) -> Self {
            let id = TEST_DIR_SEQ.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "flexmesa_sample_eval_{tag}_{}_{id}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }

        fn write(&self, rel: &str, text: &str) {
            fs::write(self.path.join(rel), text).expect("write fixture file");
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    const VALID_L0_GYRE_IN: &str = "\
&model
/
&mode
  l = 0
/
&osc
  outer_bound = 'VACUUM'
  variables_set = 'JCD'
/
&rot
/
&num
  diff_scheme = 'COLLOC_GL4'
/
&scan
  grid_type = 'LINEAR'
  freq_min = 1500
  freq_max = 3500
  freq_min_units = 'UHZ'
  freq_max_units = 'UHZ'
  n_freq = 40
/
&grid
/
";

    fn hydrated_solar_zams() -> StarModel {
        let controls = StarControls {
            zbase: 0.02,
            calculate_brunt_n2: true,
            atm_option: AtmOption::TTau,
            atm_t_tau_relation: AtmTTauRelation::Eddington,
            atm_t_tau_opacity: AtmTTauOpacity::Fixed,
            ..StarControls::default()
        };
        let mut model = load_zams_model(controls, StarJobControls::default())
            .expect("zams");
        set_hydro_vars(&mut model, false, true).expect("set_hydro_vars");
        model
    }

    fn base_gyre_ctrl(
        gyre_file: &str,
        gyre_non_ad: bool,
    ) -> AsteroSearchControls {
        let mut ctrl = AsteroSearchControls {
            oscillation_code: String::from("gyre"),
            gyre_input_file: String::from(gyre_file),
            gyre_non_ad,
            nl: [1, 0, 0, 0],
            ..AsteroSearchControls::default()
        };
        ctrl.freq_target[0] = vec![2000.0];
        ctrl.freq_sigma[0] = vec![1.0];
        ctrl.weights.chi2_seismo_fraction = 1.0;
        ctrl
    }

    #[test]
    fn gyre_init_failure_returns_explicit_error_without_bvp_fallback() {
        let tmp = TempDirGuard::new("gyre_init_err");
        let mut model = hydrated_solar_zams();
        model.model_number = 5;

        // 1. Missing gyre_input_file must return IoError, not BVP modes.
        let missing_ctrl = base_gyre_ctrl("missing_gyre.in", false);
        let missing_err = eval_model_astero_sample(
            &model,
            &missing_ctrl,
            &[0.0; MAX_CONSTRAINTS],
            missing_ctrl.first_params(),
            1,
            &tmp.path,
        )
        .expect_err("missing gyre.in must fail loudly");
        assert!(
            matches!(missing_err, StarError::IoError(_)),
            "expected IoError, got {missing_err:?}"
        );

        // 2. Empty gyre_input_file must return InvalidInput.
        let empty_ctrl = base_gyre_ctrl("   ", false);
        let empty_err = eval_model_astero_sample(
            &model,
            &empty_ctrl,
            &[0.0; MAX_CONSTRAINTS],
            empty_ctrl.first_params(),
            1,
            &tmp.path,
        )
        .expect_err("empty gyre_input_file must fail loudly");
        assert!(
            matches!(empty_err, StarError::InvalidInput(_)),
            "expected InvalidInput, got {empty_err:?}"
        );

        // 3. Invalid GYRE namelist (e.g. missing &model) must propagate
        //    AsteroGyre::init's InvalidInput error (even when outer_bound is
        //    present) rather than falling back to BVP.
        tmp.write("bad_gyre.in", "&osc\n  outer_bound = 'VACUUM'\n/\n");
        let bad_ctrl = base_gyre_ctrl("bad_gyre.in", false);
        let bad_err = eval_model_astero_sample(
            &model,
            &bad_ctrl,
            &[0.0; MAX_CONSTRAINTS],
            bad_ctrl.first_params(),
            1,
            &tmp.path,
        )
        .expect_err("malformed gyre.in must fail loudly");
        assert!(
            matches!(bad_err, StarError::InvalidInput(ref m) if m.contains("&model")),
            "expected GYRE &model InvalidInput, got {bad_err:?}"
        );

        let mut save_ctrl = bad_ctrl;
        save_ctrl.save_mode_model_number = 5;
        let save_err = scan_and_save_mode_at_number(
            &model, &save_ctrl, &tmp.path, &tmp.path,
        )
        .expect_err("save_mode path must also fail loudly on init");
        assert!(
            matches!(save_err, StarError::InvalidInput(ref m) if m.contains("&model")),
            "expected GYRE &model InvalidInput, got {save_err:?}"
        );
    }

    #[test]
    fn gyre_non_ad_is_accepted_and_ignored_like_upstream() {
        let tmp = TempDirGuard::new("gyre_non_ad_ignored");
        tmp.write("gyre.in", VALID_L0_GYRE_IN);
        let mut model = hydrated_solar_zams();
        model.model_number = 7;

        let ctrl_ad = base_gyre_ctrl("gyre.in", false);
        let ctrl_non_ad = base_gyre_ctrl("gyre.in", true);

        let res_ad = eval_model_astero_sample(
            &model,
            &ctrl_ad,
            &[0.0; MAX_CONSTRAINTS],
            ctrl_ad.first_params(),
            1,
            &tmp.path,
        )
        .expect("l=0 GYRE sample eval with gyre_non_ad = false succeeds");
        let res_non_ad = eval_model_astero_sample(
            &model,
            &ctrl_non_ad,
            &[0.0; MAX_CONSTRAINTS],
            ctrl_non_ad.first_params(),
            1,
            &tmp.path,
        )
        .expect("l=0 GYRE sample eval must accept and ignore gyre_non_ad");

        assert_eq!(res_non_ad.chi2.to_bits(), res_ad.chi2.to_bits());
        assert_eq!(res_non_ad.model_orders, res_ad.model_orders);
        assert_eq!(
            res_non_ad.model_freqs[0]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            res_ad.model_freqs[0]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            res_non_ad.model_inertias[0]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            res_ad.model_inertias[0]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );

        let mut save_ctrl = ctrl_non_ad;
        save_ctrl.save_mode_model_number = 7;
        save_ctrl.el_to_save = -1;
        scan_and_save_mode_at_number(&model, &save_ctrl, &tmp.path, &tmp.path)
            .expect("save_mode path must accept and ignore gyre_non_ad");
    }

    #[test]
    fn gyre_unported_get_modes_returns_explicit_error_without_bvp_fallback() {
        let tmp = TempDirGuard::new("gyre_unported_err");
        // l = 1 with variables_set = 'DZIEM' initializes AsteroGyre cleanly and
        // stores the model, then fails in AdBvp::new with GyreError::Unported.
        tmp.write(
            "gyre_l1_dziem.in",
            "\
&model
/
&mode
  l = 1
/
&osc
  outer_bound = 'VACUUM'
  variables_set = 'DZIEM'
/
&rot
/
&num
  diff_scheme = 'COLLOC_GL4'
/
&scan
  grid_type = 'LINEAR'
  freq_min = 1500
  freq_max = 3500
  freq_min_units = 'UHZ'
  freq_max_units = 'UHZ'
  n_freq = 40
/
&grid
/
",
        );
        let model = hydrated_solar_zams();
        let mut ctrl = AsteroSearchControls {
            oscillation_code: String::from("gyre"),
            gyre_input_file: String::from("gyre_l1_dziem.in"),
            gyre_non_ad: false,
            nl: [0, 1, 0, 0],
            ..AsteroSearchControls::default()
        };
        ctrl.freq_target[1] = vec![2000.0];
        ctrl.freq_sigma[1] = vec![1.0];
        ctrl.weights.chi2_seismo_fraction = 1.0;

        let err = eval_model_astero_sample(
            &model,
            &ctrl,
            &[0.0; MAX_CONSTRAINTS],
            ctrl.first_params(),
            1,
            &tmp.path,
        )
        .expect_err("unported GYRE mode path must fail loudly");
        assert!(
            matches!(
                err,
                StarError::InvalidInput(ref m) if m.contains("not ported")
            ),
            "expected GYRE not ported error, got {err:?}"
        );
    }

    #[test]
    fn chi2_spectro_reproduces_upstream_zero_neg_sigma_and_empty_cnt() {
        use crate::ratios_chi2::eval_astero_chi2;

        let model = hydrated_solar_zams();
        let values = [0.0_f64; MAX_CONSTRAINTS];

        // 1. Empty spectroscopic selection: NaN when normalized, 0.0 when unnormalized.
        let mut ctrl = AsteroSearchControls {
            include_age_in_chi2_spectro: false,
            ..AsteroSearchControls::default()
        };
        ctrl.weights.normalize_chi2_spectro = true;
        let (chi2_empty_norm, _, _) =
            eval_spectro_and_delta_nu_chi2(&model, &ctrl, &values);
        assert!(chi2_empty_norm.is_nan());

        ctrl.weights.normalize_chi2_spectro = false;
        let (chi2_empty_raw, _, _) =
            eval_spectro_and_delta_nu_chi2(&model, &ctrl, &values);
        assert_eq!(chi2_empty_raw.to_bits(), 0.0_f64.to_bits());

        // 2. Constraint with `sigma == 0`: `+Inf` if `value != target`, `NaN` if equal.
        ctrl.weights.normalize_chi2_spectro = true;
        ctrl.indexed_constraints[0].include = true;
        ctrl.indexed_constraints[0].target = 5777.0;
        ctrl.indexed_constraints[0].sigma = 0.0;
        let mut vals_diff = values;
        vals_diff[0] = 5800.0;
        let (chi2_inf, _, _) =
            eval_spectro_and_delta_nu_chi2(&model, &ctrl, &vals_diff);
        assert!(chi2_inf.is_infinite() && chi2_inf.is_sign_positive());

        let mut vals_eq = values;
        vals_eq[0] = 5777.0;
        let (chi2_zero_div_zero, _, _) =
            eval_spectro_and_delta_nu_chi2(&model, &ctrl, &vals_eq);
        assert!(chi2_zero_div_zero.is_nan());

        // 3. `include_age_in_chi2_spectro = true` with `age_sigma == 0` -> Inf.
        ctrl.indexed_constraints[0].include = false;
        ctrl.include_age_in_chi2_spectro = true;
        ctrl.age_target = model.star_age + 1.0e8;
        ctrl.age_sigma = 0.0;
        let (chi2_age_inf, _, _) =
            eval_spectro_and_delta_nu_chi2(&model, &ctrl, &values);
        assert!(chi2_age_inf.is_infinite() && chi2_age_inf.is_sign_positive());

        // 4. Negative sigma is squared and included in `cnt`, matching `get_chi2_spectro`.
        ctrl.include_age_in_chi2_spectro = false;
        ctrl.indexed_constraints[0].include = true;
        ctrl.indexed_constraints[0].target = 5777.0;
        ctrl.indexed_constraints[0].sigma = -10.0;
        let (chi2_neg_sig, _, _) =
            eval_spectro_and_delta_nu_chi2(&model, &ctrl, &vals_diff);
        let expected_term0 = flexmesa_math::pow2((5800.0_f64 - 5777.0) / -10.0);
        assert_eq!(chi2_neg_sig.to_bits(), 0x4015_28F5_C28F_5C28);
        assert_eq!(chi2_neg_sig.to_bits(), expected_term0.to_bits());

        // Adding a second constraint confirms `cnt == 2` includes the negative-sigma term.
        ctrl.indexed_constraints[1].include = true;
        ctrl.indexed_constraints[1].target = 1.0;
        ctrl.indexed_constraints[1].sigma = 2.0;
        vals_diff[1] = 3.0;
        let (chi2_two_terms, _, _) =
            eval_spectro_and_delta_nu_chi2(&model, &ctrl, &vals_diff);
        assert_eq!(
            chi2_two_terms.to_bits(),
            ((expected_term0 + 1.0) / 2.0).to_bits()
        );

        // Meanwhile `get_chi2` (`eval_astero_chi2`) still guards `sigma > 0`.
        let (_, _, chi2_get_chi2, _, _, _) = eval_astero_chi2(
            &ctrl.weights,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            &[5777.0],
            &[-10.0],
            &[5800.0],
        );
        assert_eq!(chi2_get_chi2.to_bits(), 0.0_f64.to_bits());
    }
}
