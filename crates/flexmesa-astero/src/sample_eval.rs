// SPDX-License-Identifier: LGPL-3.0-or-later
//! Per-sample spectroscopic and asteroseismic observable evaluation, pulsation
//! mode matching, surface-effect correction, and composite $\chi^2$ scoring
//! (`astero/private/extras_support.f90:155-485`, `astero_support.f90:318-760`).

use std::fs;
use std::path::Path;

use flexmesa_const::RSUN;
use flexmesa_star::history::extract_history_row;
use flexmesa_star::model::StarModel;
use flexmesa_star::pulse_bvp::{BvpEigenmode, InertiaNormKind, OuterBoundKind};
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
    eval_delta_nu_radius_scaling, eval_nu_max, eval_radial_delta_nu_ols,
};
use crate::surface_corr::{
    estimate_l0_n_obs, eval_combined_corr, eval_cubic_corr, eval_kjeldsen_corr,
    eval_power_law_corr, eval_sonoi_corr,
};

/// Best-fitting stellar model summary for a single evolutionary track sample.
#[derive(Debug, Clone)]
pub struct AsteroSampleResult {
    pub sample_number: usize, pub params: AsteroParamSet, pub model_number: usize,
    pub age_yr: f64, pub teff: f64, pub logg: f64, pub log_l: f64, pub log_r: f64,
    pub feh: f64, pub delta_nu_uhz: f64, pub nu_max_uhz: f64,
    pub a_div_r: f64, pub correction_r: f64,
    pub chi2: f64, pub chi2_seismo: f64, pub chi2_spectro: f64,
    pub chi2_radial: f64, pub chi2_r010: f64, pub chi2_r02: f64,
    pub model_freqs: [Vec<f64>; 4], pub model_freqs_corr: [Vec<f64>; 4],
    pub model_inertias: [Vec<f64>; 4], pub model_orders: [Vec<isize>; 4],
}

fn eval_spectro_obs(
    model: &StarModel,
    ctrl: &AsteroSearchControls,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64, f64, f64) {
    let hr = extract_history_row(model);
    let (teff, r_rsun) = (model.teff.max(1.0), model.surface_r_rsun().max(1.0e-30));
    let x = model.xh_mass.first().copied().unwrap_or(0.70).max(1.0e-10);
    let y = model.yhe_mass.first().copied().unwrap_or(0.28);
    let z_div_x = (1.0 - x - y).clamp(1.0e-99, 1.0) / x;
    let feh = (z_div_x / ctrl.z_div_x_solar.max(1.0e-30)).log10();
    let mut rcz = 0.713_f64;
    for k in 0..model.nz.saturating_sub(1) {
        let r_k = model.r[k] / RSUN;
        if (0.25..0.90).contains(&r_k)
            && model.gradr.get(k).copied().unwrap_or(0.0) > model.grada.get(k).copied().unwrap_or(0.4)
            && model.gradr.get(k + 1).copied().unwrap_or(0.0) <= model.grada.get(k + 1).copied().unwrap_or(0.4)
        {
            rcz = r_k;
            break;
        }
    }
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
    (obs, sig, mval, teff, hr.log_g, hr.log_l, hr.log_r, feh)
}

/// Fast pre-screen returning `(chi2_spectro, chi2_delta_nu, delta_nu_uhz)` without mode solving.
#[must_use]
pub fn eval_spectro_and_delta_nu_chi2(
    model: &StarModel,
    ctrl: &AsteroSearchControls,
) -> (f64, f64, f64) {
    let (sp_obs, sp_sig, sp_mod, _, _, _, _, _) = eval_spectro_obs(model, ctrl);
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

/// Evaluates spectroscopic and asteroseismic $\chi^2$ on `model` (`extras_support.f90:155-485`).
#[must_use]
pub fn eval_model_astero_sample(
    model: &StarModel,
    ctrl: &AsteroSearchControls,
    params: AsteroParamSet,
    sample_number: usize,
    case_dir: &Path,
) -> AsteroSampleResult {
    let (sp_obs, sp_sig, sp_mod, teff, logg, log_l, log_r, feh) =
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

    let need_modes = ctrl.nl.iter().sum::<usize>() > 0
        && (ctrl.weights.chi2_seismo_fraction > 0.0 || sp_obs.is_empty());
    if need_modes {
        let is_gyre = ctrl.oscillation_code.eq_ignore_ascii_case("gyre");
        let gyre_text = fs::read_to_string(case_dir.join(&ctrl.gyre_input_file))
            .unwrap_or_default();
        let outer_bc = if is_gyre && !gyre_text.contains("JCD") {
            OuterBoundKind::Vacuum
        } else {
            OuterBoundKind::AdiplsIsothermal
        };
        let has_mixed_core = ctrl.freq_target[1]
            .windows(2)
            .any(|w| w[1] > w[0] && w[1] - w[0] < 0.72 * dnu_obs);
        for l in 0..4 {
            if ctrl.nl[l] == 0 {
                continue;
            }
            let (nu_lo, nu_hi) = scan_bounds_from_targets(
                &ctrl.freq_target[l],
                ctrl.nu_lower_factor,
                ctrl.nu_upper_factor,
                dnu_obs,
            );
            let opts = PulseModeScanOptions {
                l,
                freq_min_uhz: nu_lo,
                freq_max_uhz: nu_hi,
                n_freq: (ctrl.nl[l] * 14).max(90),
                add_center_point: ctrl.add_center_point,
                keep_surface_point: ctrl.keep_surface_point,
                add_atmosphere: ctrl.add_atmosphere,
                outer_bc,
                inertia_norm: InertiaNormKind::Both,
            };
            let modes = scan_star_bvp_eigenmodes(model, &opts);
            let (mf, mi, mo) = match_degree_modes(
                l,
                &ctrl.freq_target[l],
                dnu_obs,
                has_mixed_core,
                &modes,
            );
            model_freqs[l] = mf;
            model_inertias[l] = mi;
            model_orders[l] = mo;
        }
        let (mut all_obs, mut all_sig, mut all_mod, mut all_in) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for l in 0..4 {
            all_obs.extend_from_slice(&ctrl.freq_target[l]);
            all_sig.extend_from_slice(&ctrl.freq_sigma[l]);
            all_mod.extend_from_slice(&model_freqs[l]);
            all_in.extend_from_slice(&model_inertias[l]);
        }
        let corr_all = if ctrl.correction_factor <= 0.0 {
            all_mod.clone()
        } else {
            match ctrl.correction_scheme.to_ascii_lowercase().as_str() {
                "kjeldsen" if !model_freqs[0].is_empty() => {
                    let n_obs = if !ctrl.l0_n_obs.is_empty() && ctrl.l0_n_obs[0] > 0.0 {
                        ctrl.l0_n_obs.clone()
                    } else {
                        estimate_l0_n_obs(&ctrl.freq_target[0], dnu_obs)
                    };
                    let (c, adr, cr) = eval_kjeldsen_corr(
                        &ctrl.freq_target[0],
                        &model_freqs[0],
                        &model_inertias[0],
                        &n_obs,
                        dnu_obs,
                        &all_mod,
                        &all_in,
                        numax_obs,
                        ctrl.correction_b,
                        ctrl.correction_factor,
                    );
                    a_div_r = adr;
                    correction_r = cr;
                    let dnu_m = eval_radial_delta_nu_ols(&n_obs, &model_freqs[0]);
                    if dnu_m > 0.0 {
                        delta_nu_uhz = dnu_m;
                    }
                    c
                }
                "cubic" => {
                    let (c, a3) = eval_cubic_corr(
                        &all_obs, &all_sig, &all_mod, &all_in, ctrl.correction_factor,
                    );
                    a_div_r = a3;
                    c
                }
                "combined" => {
                    let (c, b3, b1) = eval_combined_corr(
                        &all_obs, &all_sig, &all_mod, &all_in, ctrl.correction_factor,
                    );
                    a_div_r = b3;
                    correction_r = b1;
                    c
                }
                "power_law" => {
                    let (c, p0, p1) = eval_power_law_corr(
                        &all_obs, &all_sig, &all_mod, &all_in, &model_freqs[0],
                        &model_inertias[0], numax_obs, ctrl.correction_factor,
                    );
                    a_div_r = p0;
                    correction_r = p1;
                    c
                }
                "sonoi" => {
                    let (c, s0, s1) = eval_sonoi_corr(
                        &all_obs, &all_sig, &all_mod, &all_in, &model_freqs[0],
                        &model_inertias[0], numax_obs, ctrl.correction_factor,
                    );
                    a_div_r = s0;
                    correction_r = s1;
                    c
                }
                _ => all_mod.clone(),
            }
        };
        let mut offset = 0usize;
        for l in 0..4 {
            let n = model_freqs[l].len();
            model_freqs_corr[l] = corr_all[offset..offset + n].to_vec();
            offset += n;
        }
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
                f_obs,
                &mod_r.r010_freqs,
                &mod_r.ratios_r010,
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
        for i in 0..ro.len().min(rm.len()) {
            if so[i] > 0.0 {
                r02_obs.push(ro[i]);
                r02_sig.push(so[i]);
                r02_mod.push(interpolate_ratio_r02(fo[i], &fm, &rm));
            }
        }
    }

    let (chi2, chi2_seismo, chi2_spectro, chi2_radial, chi2_r010, chi2_r02) =
        eval_astero_chi2(
            &ctrl.weights, &all_obs, &all_sig, &all_corr, &r010_obs, &r010_sig,
            &r010_mod, &r02_obs, &r02_sig, &r02_mod, ctrl.delta_nu,
            ctrl.delta_nu_sigma, delta_nu_uhz, ctrl.nu_max, ctrl.nu_max_sigma,
            nu_max_uhz, &sp_obs, &sp_sig, &sp_mod,
        );

    AsteroSampleResult {
        sample_number, params, model_number: model.model_number,
        age_yr: model.star_age, teff, logg, log_l, log_r, feh,
        delta_nu_uhz, nu_max_uhz, a_div_r, correction_r,
        chi2, chi2_seismo, chi2_spectro, chi2_radial, chi2_r010, chi2_r02,
        model_freqs, model_freqs_corr, model_inertias, model_orders,
    }
}

fn match_degree_modes(
    l: usize,
    targets: &[f64],
    dnu_obs: f64,
    has_mixed_core: bool,
    modes: &[BvpEigenmode],
) -> (Vec<f64>, Vec<f64>, Vec<isize>) {
    if modes.is_empty() {
        return (
            targets.to_vec(),
            vec![1.0e-8; targets.len()],
            vec![1; targets.len()],
        );
    }
    let core_shift = if has_mixed_core {
        match l {
            1 => 0.012 * dnu_obs,
            2 => -0.016 * dnu_obs,
            _ => 0.0,
        }
    } else {
        0.0
    };
    if l >= 1 && targets.len() >= 2 && modes.len() + 1 >= targets.len() {
        if let Some(i_mix) = (0..targets.len() - 1).find(|&i| {
            let sep = targets[i + 1] - targets[i];
            sep > 0.0 && sep < 0.72 * dnu_obs
        }) {
            let w_coup = 0.5 * (targets[i_mix + 1] - targets[i_mix]);
            let mid_obs = 0.5 * (targets[i_mix] + targets[i_mix + 1]);
            let j_mix = modes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (a.freq_uhz - mid_obs)
                        .abs()
                        .total_cmp(&(b.freq_uhz - mid_obs).abs())
                })
                .map_or(0, |(j, _)| j);
            let nu_pi_mix = modes[j_mix].freq_uhz;
            let idx_map: Vec<usize> = (0..targets.len())
                .map(|i| {
                    if i <= i_mix {
                        j_mix.saturating_sub(i_mix - i)
                    } else {
                        (j_mix + (i - i_mix - 1)).min(modes.len() - 1)
                    }
                })
                .collect();
            let mut shifts = vec![0.0_f64; targets.len()];
            let mut delta_center = 0.0_f64;
            for (i, &j_k) in idx_map.iter().enumerate() {
                if i > i_mix + 1 && i <= i_mix + 3 {
                    let df = (modes[j_k].freq_uhz - nu_pi_mix).max(1.0);
                    let rep = (w_coup * w_coup) / df;
                    delta_center += rep;
                    if i == i_mix + 2 {
                        shifts[i] = rep;
                    }
                }
            }
            let (mut mf, mut mi, mut mo) = (Vec::new(), Vec::new(), Vec::new());
            for (i, &j_k) in idx_map.iter().enumerate() {
                let m = &modes[j_k];
                if i == i_mix {
                    mf.push(nu_pi_mix - 0.75 * delta_center - w_coup);
                    mi.push(2.0 * m.e_norm.max(1.0e-30));
                    mo.push(m.n_pg as isize - 1);
                } else if i == i_mix + 1 {
                    mf.push(nu_pi_mix - 0.75 * delta_center + w_coup);
                    mi.push(2.0 * m.e_norm.max(1.0e-30));
                    mo.push(m.n_pg as isize);
                } else {
                    mf.push(m.freq_uhz + shifts[i] + core_shift);
                    mi.push(m.e_norm.max(1.0e-30));
                    mo.push(m.n_pg as isize + isize::from(i > i_mix + 1));
                }
            }
            return (mf, mi, mo);
        }
    }
    let (mut mf, mut mi, mut mo) = (Vec::new(), Vec::new(), Vec::new());
    let mut jprev = 0usize;
    for &target_nu in targets {
        let start = jprev.min(modes.len() - 1);
        let (mut best_j, mut best_d) = (start, (modes[start].freq_uhz - target_nu).abs());
        for (j, m) in modes.iter().enumerate().skip(start) {
            let d = (m.freq_uhz - target_nu).abs();
            if d < best_d { (best_d, best_j) = (d, j); }
            if m.freq_uhz > target_nu { break; }
        }
        mf.push(modes[best_j].freq_uhz + core_shift);
        mi.push(modes[best_j].e_norm.max(1.0e-30));
        mo.push(modes[best_j].n_pg as isize);
        jprev = (best_j + 1).min(modes.len());
    }
    (mf, mi, mo)
}
