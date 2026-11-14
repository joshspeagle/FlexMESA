// SPDX-License-Identifier: LGPL-3.0-or-later
//! Per-sample spectroscopic and asteroseismic observable evaluation, pulsation
//! mode matching, surface-effect correction, and composite $\chi^2$ scoring
//! (`astero/private/extras_support.f90:155-485`, `astero_support.f90:223-760`).

use std::fs;
use std::path::Path;

use flexmesa_const::RSUN;
use flexmesa_star::history::extract_history_row;
use flexmesa_star::model::StarModel;
use flexmesa_star::pulse_bvp::{InertiaNormKind, OuterBoundKind};
use flexmesa_star::pulse_modes::{
    scan_bounds_from_targets, scan_star_bvp_eigenmodes, PulseModeScanOptions,
};

use crate::inlist_astero::{
    AsteroParamSet, AsteroSearchControls, SpectroConstraint,
};
use crate::ratios_chi2::{
    eval_astero_chi2, eval_frequency_ratios_r02_with_dnu,
    eval_frequency_ratios_with_trimming, interpolate_ratio_r010,
    interpolate_ratio_r02,
};
use crate::scaling::{
    eval_acoustic_profile, eval_delta_nu_asymptotic,
    eval_delta_nu_radius_scaling, eval_nu_max,
};
use crate::search_methods::{
    apply_surface_corr, eval_radial_chi2, match_degree_modes,
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
    /// Surface $[\mathrm{Fe/H}]$.
    pub feh: f64,
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
}

/// Evaluates the radius of the convective envelope base $R_{\mathrm{cz}} / R_\odot$
/// from the outermost convective-to-radiative boundary below the photosphere.
#[must_use]
pub fn eval_convective_envelope_base_rsun(model: &StarModel) -> f64 {
    let r_surf = model.r.first().copied().unwrap_or(RSUN).max(1.0);
    for k in 0..model.nz.saturating_sub(1) {
        if model.r[k] > 0.995 * r_surf {
            continue;
        }
        let conv_k = model.gradr.get(k).copied().unwrap_or(0.0)
            > model.grada.get(k).copied().unwrap_or(0.4);
        let conv_in = model.gradr.get(k + 1).copied().unwrap_or(0.0)
            > model.grada.get(k + 1).copied().unwrap_or(0.4);
        if conv_k && !conv_in {
            let dr_k = model.gradr[k] - model.grada[k];
            let dr_in = model.grada[k + 1] - model.gradr[k + 1];
            let w = (dr_k / (dr_k + dr_in).max(1.0e-30)).clamp(0.0, 1.0);
            let r_bcz = (1.0 - w) * model.r[k] + w * model.r[k + 1];
            return r_bcz / RSUN;
        }
    }
    0.0
}

fn eval_spectro_obs(
    model: &StarModel,
    ctrl: &AsteroSearchControls,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64, f64, f64, f64, f64, f64) {
    let hr = extract_history_row(model);
    let (teff, r_rsun) = (model.teff.max(1.0), model.surface_r_rsun().max(1.0e-30));
    let x = model.xh_mass.first().copied().unwrap_or(0.70).max(1.0e-10);
    let y = model.yhe_mass.first().copied().unwrap_or(0.28);
    let z_div_x = (1.0 - x - y).clamp(1.0e-99, 1.0) / x;
    let feh = (z_div_x / ctrl.z_div_x_solar.max(1.0e-30)).log10();
    let rcz = eval_convective_envelope_base_rsun(model);
    let pairs: [(SpectroConstraint, f64); 10] = [
        (ctrl.teff, teff), (ctrl.logg, hr.log_g), (ctrl.log_l, hr.log_l),
        (ctrl.feh, feh), (ctrl.m_h, feh), (ctrl.log_r, hr.log_r),
        (ctrl.radius, r_rsun), (ctrl.surface_z_div_x, z_div_x),
        (ctrl.surface_he, y), (ctrl.rcz, rcz),
    ];
    let (mut obs, mut sig, mut mval) = (Vec::new(), Vec::new(), Vec::new());
    if ctrl.include_age_in_chi2_spectro && ctrl.age_sigma > 0.0 {
        obs.push(ctrl.age_target);
        sig.push(ctrl.age_sigma);
        mval.push(model.star_age);
    }
    for (c, v) in pairs {
        if c.include && c.sigma > 0.0 {
            obs.push(c.target);
            sig.push(c.sigma);
            mval.push(v);
        }
    }
    (obs, sig, mval, teff, hr.log_g, hr.log_l, hr.log_r, feh, r_rsun, z_div_x, rcz)
}

/// Fast pre-screen returning `(chi2_spectro, chi2_delta_nu, delta_nu_uhz)` without mode solving.
#[must_use]
pub fn eval_spectro_and_delta_nu_chi2(
    model: &StarModel,
    ctrl: &AsteroSearchControls,
) -> (f64, f64, f64) {
    let (sp_obs, sp_sig, sp_mod, _, _, _, _, _, _, _, _) = eval_spectro_obs(model, ctrl);
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
    let r_phot_cm = model.surface_r_rsun().max(1.0e-30) * RSUN;
    let (_, _, tau_0) =
        eval_acoustic_profile(&model.r, &model.csound, model.r_center, r_phot_cm);
    let delta_nu_uhz = if tau_0 > 0.0 {
        eval_delta_nu_asymptotic(tau_0)
    } else {
        eval_delta_nu_radius_scaling(model.mstar, r_phot_cm)
    };
    let chi2_delta_nu = if ctrl.delta_nu > 0.0 && ctrl.delta_nu_sigma > 0.0 {
        let d = (ctrl.delta_nu - delta_nu_uhz) / ctrl.delta_nu_sigma;
        d * d
    } else {
        0.0
    };
    (chi2_spectro, chi2_delta_nu, delta_nu_uhz)
}

/// Parses `outer_bound` from a `GYRE` input file, stripping `!` comments first.
#[must_use]
pub fn parse_gyre_outer_bound(case_dir: &Path, gyre_file: &str) -> OuterBoundKind {
    let Ok(text) = fs::read_to_string(case_dir.join(gyre_file)) else {
        return OuterBoundKind::AdiplsIsothermal;
    };
    for raw_line in text.lines() {
        let line = raw_line.split('!').next().unwrap_or("").trim();
        let Some((lhs, rhs)) = line.split_once('=') else { continue };
        if lhs.trim().eq_ignore_ascii_case("outer_bound") {
            let val = rhs.trim().trim_matches(|c| c == '\'' || c == '"');
            if val.eq_ignore_ascii_case("VACUUM") || val.eq_ignore_ascii_case("ZERO_R") {
                return OuterBoundKind::Vacuum;
            }
            return OuterBoundKind::AdiplsIsothermal;
        }
    }
    OuterBoundKind::AdiplsIsothermal
}

/// Evaluates spectroscopic and asteroseismic $\chi^2$ on `model` (`extras_support.f90:155-485`).
#[must_use]
pub fn eval_model_astero_sample(
    model: &StarModel,
    ctrl: &AsteroSearchControls,
    params: AsteroParamSet,
    sample_number: usize,
    case_dir: &Path,
) -> AsteroSampleResult {
    let (sp_obs, sp_sig, sp_mod, teff, logg, log_l, log_r, feh, _, _, _) =
        eval_spectro_obs(model, ctrl);
    let r_phot_cm = model.surface_r_rsun().max(1.0e-30) * RSUN;
    let (_, _, tau_0) =
        eval_acoustic_profile(&model.r, &model.csound, model.r_center, r_phot_cm);
    let mut delta_nu_uhz = if tau_0 > 0.0 {
        eval_delta_nu_asymptotic(tau_0)
    } else {
        eval_delta_nu_radius_scaling(model.mstar, r_phot_cm)
    };
    let nu_max_uhz = eval_nu_max(model.mstar, r_phot_cm, teff);
    let dnu_obs = if ctrl.delta_nu > 0.0 { ctrl.delta_nu } else { delta_nu_uhz };
    let numax_obs = if ctrl.nu_max > 0.0 { ctrl.nu_max } else { nu_max_uhz };

    let mut model_freqs: [Vec<f64>; 4] = Default::default();
    let mut model_freqs_corr: [Vec<f64>; 4] = Default::default();
    let mut model_inertias: [Vec<f64>; 4] = Default::default();
    let mut model_orders: [Vec<isize>; 4] = Default::default();
    let (mut a_div_r, mut correction_r) = (0.0_f64, 1.0_f64);
    let mut mode_match_failed = false;

    let need_modes = ctrl.nl.iter().sum::<usize>() > 0
        && (ctrl.weights.chi2_seismo_fraction > 0.0 || sp_obs.is_empty());
    let mut pre_chi2_radial = 0.0_f64;
    if need_modes {
        let is_gyre = ctrl.oscillation_code.eq_ignore_ascii_case("gyre");
        let outer_bc = if is_gyre {
            parse_gyre_outer_bound(case_dir, &ctrl.gyre_input_file)
        } else {
            OuterBoundKind::AdiplsIsothermal
        };
        let scan_deg = |l: usize| -> Option<(Vec<f64>, Vec<f64>, Vec<isize>)> {
            let (nu_lo, nu_hi) = scan_bounds_from_targets(
                &ctrl.freq_target[l], ctrl.nu_lower_factor, ctrl.nu_upper_factor, dnu_obs,
            );
            let n_scan = ((2.0 * (ctrl.nl[l] as f64) * ctrl.iscan_factor[l].max(1.0)) as usize)
                .clamp(240, 600);
            let opts = PulseModeScanOptions {
                l, freq_min_uhz: nu_lo, freq_max_uhz: nu_hi, n_freq: n_scan,
                add_center_point: ctrl.add_center_point, keep_surface_point: ctrl.keep_surface_point,
                add_atmosphere: ctrl.add_atmosphere, outer_bc, inertia_norm: InertiaNormKind::Both,
            };
            let raw_modes = scan_star_bvp_eigenmodes(model, &opts);
            match_degree_modes(&ctrl.freq_target[l], &raw_modes)
        };
        if ctrl.nl[0] > 0 {
            if let Some((mf, mi, mo)) = scan_deg(0) {
                (model_freqs[0], model_inertias[0], model_orders[0]) = (mf, mi, mo);
                let (c0, adr, cr) = apply_surface_corr(
                    ctrl, &model_freqs[0], &model_inertias[0], &ctrl.freq_target[0],
                    &ctrl.freq_sigma[0], &model_freqs[0], &model_inertias[0], dnu_obs, numax_obs,
                    &mut delta_nu_uhz,
                );
                (a_div_r, correction_r) = (adr, cr);
                pre_chi2_radial = eval_radial_chi2(ctrl, &c0);
                model_freqs_corr[0] = c0;
                if pre_chi2_radial > ctrl.chi2_radial_limit
                    && ctrl.nl[1] + ctrl.nl[2] + ctrl.nl[3] > 0
                {
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
                if let Some((mf, mi, mo)) = scan_deg(l) {
                    (model_freqs[l], model_inertias[l], model_orders[l]) = (mf, mi, mo);
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
            let (corr_all, adr, cr) = apply_surface_corr(
                ctrl, &model_freqs[0], &model_inertias[0], &all_obs, &all_sig,
                &all_mod, &all_in, dnu_obs, numax_obs, &mut delta_nu_uhz,
            );
            (a_div_r, correction_r) = (adr, cr);
            let mut offset = 0usize;
            for l in 0..4 {
                let n = model_freqs[l].len();
                model_freqs_corr[l] = corr_all[offset..offset + n].to_vec();
                offset += n;
            }
        }
    }

    if mode_match_failed {
        let cr_fail = if pre_chi2_radial > 0.0 { pre_chi2_radial } else { 1.0e99 };
        return AsteroSampleResult {
            sample_number, params, model_number: model.model_number,
            age_yr: model.star_age, teff, logg, log_l, log_r, feh,
            delta_nu_uhz, nu_max_uhz, a_div_r, correction_r,
            chi2: 1.0e99, chi2_seismo: 1.0e99, chi2_spectro: 1.0e99,
            chi2_radial: cr_fail, chi2_r010: 1.0e99, chi2_r02: 1.0e99,
            model_freqs, model_freqs_corr, model_inertias, model_orders,
        };
    }

    let (mut all_obs, mut all_sig, mut all_corr) = (Vec::new(), Vec::new(), Vec::new());
    for l in 0..4 {
        all_obs.extend_from_slice(&ctrl.freq_target[l]);
        all_sig.extend_from_slice(&ctrl.freq_sigma[l]);
        all_corr.extend_from_slice(&model_freqs_corr[l]);
    }
    let (mut r010_obs, mut r010_sig, mut r010_mod) = (Vec::new(), Vec::new(), Vec::new());
    if ctrl.weights.chi2_seismo_r_010_fraction > 0.0 && ctrl.nl[0] > 0 && ctrl.nl[1] > 0 {
        let obs_r = eval_frequency_ratios_with_trimming(
            &ctrl.freq_target[0], &ctrl.freq_sigma[0], &ctrl.freq_target[1], &ctrl.freq_sigma[1],
            dnu_obs, true, 0, 0, 0,
        );
        let mod_r = eval_frequency_ratios_with_trimming(
            &model_freqs_corr[0], &ctrl.freq_sigma[0], &model_freqs_corr[1], &ctrl.freq_sigma[1],
            dnu_obs, false, obs_r.n, obs_r.l0_first, obs_r.l1_first,
        );
        r010_obs.extend_from_slice(&obs_r.ratios_r010);
        r010_sig.extend_from_slice(&obs_r.sigmas_r010);
        for &f_obs in &obs_r.r010_freqs {
            r010_mod.push(interpolate_ratio_r010(
                f_obs, &mod_r.r010_freqs, &mod_r.ratios_r010,
            ));
        }
    }
    let (mut r02_obs, mut r02_sig, mut r02_mod) = (Vec::new(), Vec::new(), Vec::new());
    if ctrl.weights.chi2_seismo_r_02_fraction > 0.0 && ctrl.nl[0] > 0 && ctrl.nl[2] > 0 {
        let (ro, so, fo) = eval_frequency_ratios_r02_with_dnu(
            &ctrl.freq_target[0], &ctrl.freq_sigma[0], &ctrl.freq_target[1], &ctrl.freq_sigma[1],
            &ctrl.freq_target[2], &ctrl.freq_sigma[2], 0, 0, dnu_obs,
        );
        let (rm, _, fm) = eval_frequency_ratios_r02_with_dnu(
            &model_freqs_corr[0], &ctrl.freq_sigma[0], &model_freqs_corr[1], &ctrl.freq_sigma[1],
            &model_freqs_corr[2], &ctrl.freq_sigma[2], 0, 0, dnu_obs,
        );
        if !rm.is_empty() {
            for i in 0..ro.len() {
                if so[i] > 0.0 {
                    r02_obs.push(ro[i]);
                    r02_sig.push(so[i]);
                    r02_mod.push(interpolate_ratio_r02(fo[i], &fm, &rm));
                }
            }
        }
    }

    let (chi2, chi2_seismo, chi2_spectro, mut chi2_radial, chi2_r010, chi2_r02) =
        eval_astero_chi2(
            &ctrl.weights, &all_obs, &all_sig, &all_corr, &r010_obs, &r010_sig,
            &r010_mod, &r02_obs, &r02_sig, &r02_mod, ctrl.delta_nu,
            ctrl.delta_nu_sigma, delta_nu_uhz, ctrl.nu_max, ctrl.nu_max_sigma,
            nu_max_uhz, &sp_obs, &sp_sig, &sp_mod,
        );
    if !model_freqs_corr[0].is_empty() {
        chi2_radial = eval_radial_chi2(ctrl, &model_freqs_corr[0]);
    }

    AsteroSampleResult {
        sample_number, params, model_number: model.model_number,
        age_yr: model.star_age, teff, logg, log_l, log_r, feh,
        delta_nu_uhz, nu_max_uhz, a_div_r, correction_r,
        chi2, chi2_seismo, chi2_spectro, chi2_radial, chi2_r010, chi2_r02,
        model_freqs, model_freqs_corr, model_inertias, model_orders,
    }
}
