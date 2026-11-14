// SPDX-License-Identifier: LGPL-3.0-or-later
//! Mode matching, Roxburgh & Vorontsov (2003) frequency separation ratios
//! ($r_{01}, r_{10}, r_{02}$), and composite asteroseismic + spectroscopic
//! $\chi^2$ evaluation (`astero/private/astero_support.f90:148-587, 1346-1527`).

/// Matches observed target frequencies `obs_freqs` of spherical degree $l$ to
/// ascending candidate model frequencies `cand_freqs` while enforcing strictly
/// monotonic mode order assignment (`find_closest` in `astero_support.f90:279-292`).
#[must_use]
pub fn match_modes_by_degree(
    obs_freqs: &[f64],
    cand_freqs: &[f64],
    cand_inertias: &[f64],
    cand_orders: &[i32],
) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    let m_cand = cand_freqs.len().min(cand_inertias.len()).min(cand_orders.len());
    let mut indices: Vec<usize> = (0..m_cand).collect();
    indices.sort_by(|&a, &b| {
        cand_freqs[a].partial_cmp(&cand_freqs[b]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut matched_f = Vec::with_capacity(obs_freqs.len());
    let mut matched_i = Vec::with_capacity(obs_freqs.len());
    let mut matched_n = Vec::with_capacity(obs_freqs.len());
    let mut j_prev: Option<usize> = None;

    for &nu_obs in obs_freqs {
        let start = j_prev.map_or(0, |jp| (jp + 1).min(m_cand));
        if start >= m_cand {
            break;
        }
        let mut best_k = start;
        let mut best_dist = (cand_freqs[indices[start]] - nu_obs).abs();
        for k in (start + 1)..m_cand {
            let f_k = cand_freqs[indices[k]];
            let dist = (f_k - nu_obs).abs();
            if dist < best_dist {
                best_dist = dist;
                best_k = k;
            }
            if f_k > nu_obs {
                break;
            }
        }
        j_prev = Some(best_k);
        let orig = indices[best_k];
        matched_f.push(cand_freqs[orig]);
        matched_i.push(cand_inertias[orig]);
        matched_n.push(cand_orders[orig]);
    }
    (matched_f, matched_i, matched_n)
}

/// Finds the longest contiguous radial-order subsequence `(seq_first, seq_last)`
/// (inclusive 0-based indices) where adjacent modes satisfy
/// $0.5\Delta\nu \le \nu_i - \nu_{i-1} \le 1.5\Delta\nu$ (`get_max_sequence`,
/// `astero_support.f90:363-391`).
#[must_use]
pub fn get_max_sequence(freqs: &[f64], dnu: f64) -> (usize, usize) {
    let n = freqs.len();
    if n == 0 {
        return (0, 0);
    }
    if n == 1 || dnu <= 0.0 {
        return (0, n - 1);
    }
    let (df_min, df_max) = (0.5 * dnu, 1.5 * dnu);
    let (mut cur_first, mut cur_last) = (0usize, 0usize);
    let (mut best_first, mut best_last) = (0usize, 0usize);
    for i in 1..n {
        let df = freqs[i] - freqs[i - 1];
        if (df_min..=df_max).contains(&df) {
            cur_last = i;
        } else {
            if cur_last - cur_first > best_last - best_first {
                (best_first, best_last) = (cur_first, cur_last);
            }
            (cur_first, cur_last) = (i, i);
        }
    }
    if cur_last - cur_first > best_last - best_first {
        (best_first, best_last) = (cur_first, cur_last);
    }
    (best_first, best_last)
}

/// Full output of `get_frequency_ratios` (`astero_support.f90:318-416`).
#[derive(Debug, Clone, Default)]
pub struct FrequencyRatiosR010 {
    pub n: usize,
    pub l0_first: usize,
    pub l1_first: usize,
    pub ratios_r01: Vec<f64>,
    pub sigmas_r01: Vec<f64>,
    pub ratios_r10: Vec<f64>,
    pub sigmas_r10: Vec<f64>,
    pub ratios_r010: Vec<f64>,
    pub sigmas_r010: Vec<f64>,
    pub r010_freqs: Vec<f64>,
}

/// Evaluates `get_frequency_ratios` with contiguous radial-order trimming and
/// $l=0/l=1$ interleaving (`astero_support.f90:318-416`).
#[must_use]
pub fn eval_frequency_ratios_with_trimming(
    l0_freqs: &[f64],
    l0_sigmas: &[f64],
    l1_freqs: &[f64],
    l1_sigmas: &[f64],
    dnu: f64,
    trim_for_obs: bool,
    preset_n: usize,
    preset_l0_first: usize,
    preset_l1_first: usize,
) -> FrequencyRatiosR010 {
    let nl0 = l0_freqs.len().min(l0_sigmas.len());
    let nl1 = l1_freqs.len().min(l1_sigmas.len());
    if nl0 < 3 || nl1 < 3 {
        return FrequencyRatiosR010::default();
    }
    let (l0_first, l1_first, n) = if trim_for_obs {
        let (mut l0_f, mut l0_l) = get_max_sequence(&l0_freqs[..nl0], dnu);
        let (mut l1_f, l1_l) = get_max_sequence(&l1_freqs[..nl1], dnu);
        while l1_f <= l1_l && l1_freqs[l1_f] < l0_freqs[l0_f] {
            l1_f += 1;
        }
        if l1_f > l1_l {
            return FrequencyRatiosR010::default();
        }
        while l0_f < l0_l && l0_freqs[l0_f + 1] < l1_freqs[l1_f] {
            l0_f += 1;
        }
        if l0_f >= l0_l {
            return FrequencyRatiosR010::default();
        }
        let (len0, len1) = (l0_l - l0_f, l1_l - l1_f);
        if len0 > len1 {
            l0_l = l0_f + len1;
        }
        let seq_len = l0_l.saturating_sub(l0_f) + 1;
        if seq_len < 3 {
            return FrequencyRatiosR010::default();
        }
        (l0_f, l1_f, seq_len - 2)
    } else if preset_n == 0
        || preset_l0_first + preset_n + 1 >= nl0
        || preset_l1_first + preset_n + 1 >= nl1
    {
        (0, 0, nl0.min(nl1) - 2)
    } else {
        (preset_l0_first, preset_l1_first, preset_n)
    };

    let (r01, sig_r01, r10, sig_r10) = eval_frequency_ratios_r01_r10(
        &l0_freqs[l0_first..=(l0_first + n + 1)],
        &l0_sigmas[l0_first..=(l0_first + n + 1)],
        &l1_freqs[l1_first..=(l1_first + n + 1)],
        &l1_sigmas[l1_first..=(l1_first + n + 1)],
    );
    let mut ratios_r010 = Vec::with_capacity(2 * n);
    let mut sigmas_r010 = Vec::with_capacity(2 * n);
    let mut r010_freqs = Vec::with_capacity(2 * n);
    for i in 0..n {
        ratios_r010.push(r01[i]);
        ratios_r010.push(r10[i]);
        sigmas_r010.push(sig_r01[i]);
        sigmas_r010.push(sig_r10[i]);
        r010_freqs.push(l0_freqs[l0_first + i + 1]);
        r010_freqs.push(l1_freqs[l1_first + i + 1]);
    }
    FrequencyRatiosR010 {
        n,
        l0_first,
        l1_first,
        ratios_r01: r01,
        sigmas_r01: sig_r01,
        ratios_r10: r10,
        sigmas_r10: sig_r10,
        ratios_r010,
        sigmas_r010,
        r010_freqs,
    }
}

/// Evaluates the Roxburgh & Vorontsov (2003, A&A 411, 215) 5-point smoothed
/// frequency separation ratios $r_{01}(i), r_{10}(i)$ and their propagated
/// uncertainties (`get_frequency_ratios`, `astero_support.f90:318-416`).
#[must_use]
pub fn eval_frequency_ratios_r01_r10(
    l0_freqs: &[f64],
    l0_sigmas: &[f64],
    l1_freqs: &[f64],
    l1_sigmas: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let len = l0_freqs.len().min(l0_sigmas.len()).min(l1_freqs.len()).min(l1_sigmas.len());
    if len < 3 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }
    let n_ratios = len - 2;
    let mut r01 = Vec::with_capacity(n_ratios);
    let mut sig_r01 = Vec::with_capacity(n_ratios);
    let mut r10 = Vec::with_capacity(n_ratios);
    let mut sig_r10 = Vec::with_capacity(n_ratios);

    for i in 1..(len - 1) {
        let d01 = (l0_freqs[i - 1] - 4.0 * l1_freqs[i - 1] + 6.0 * l0_freqs[i]
            - 4.0 * l1_freqs[i]
            + l0_freqs[i + 1])
            / 8.0;
        let dnu1 = (l1_freqs[i] - l1_freqs[i - 1]).max(1.0e-30);
        r01.push(d01 / dnu1);
        let s_d01_sq = (l0_sigmas[i - 1].powi(2)
            + 16.0 * l1_sigmas[i - 1].powi(2)
            + 36.0 * l0_sigmas[i].powi(2)
            + 16.0 * l1_sigmas[i].powi(2)
            + l0_sigmas[i + 1].powi(2))
            / 64.0;
        let s_dnu1_sq = l1_sigmas[i].powi(2) + l1_sigmas[i - 1].powi(2);
        sig_r01.push((s_d01_sq / (dnu1 * dnu1) + s_dnu1_sq * (d01 * d01) / dnu1.powi(4)).sqrt());

        let d10 = -(l1_freqs[i - 1] - 4.0 * l0_freqs[i] + 6.0 * l1_freqs[i]
            - 4.0 * l0_freqs[i + 1]
            + l1_freqs[i + 1])
            / 8.0;
        let dnu0 = (l0_freqs[i + 1] - l0_freqs[i]).max(1.0e-30);
        r10.push(d10 / dnu0);
        let s_d10_sq = (l1_sigmas[i - 1].powi(2)
            + 16.0 * l0_sigmas[i].powi(2)
            + 36.0 * l1_sigmas[i].powi(2)
            + 16.0 * l0_sigmas[i + 1].powi(2)
            + l1_sigmas[i + 1].powi(2))
            / 64.0;
        let s_dnu0_sq = l0_sigmas[i + 1].powi(2) + l0_sigmas[i].powi(2);
        sig_r10.push((s_d10_sq / (dnu0 * dnu0) + s_dnu0_sq * (d10 * d10) / dnu0.powi(4)).sqrt());
    }
    (r01, sig_r01, r10, sig_r10)
}

/// Evaluates $r_{02}(i)$ and $\sigma_{r_{02}}(i)$ with `0.25 * dnu` mode pairing
/// (`get_r02_frequency_ratios`, `astero_support.f90:421-486`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_frequency_ratios_r02_with_dnu(
    l0_freqs: &[f64],
    l0_sigmas: &[f64],
    l1_freqs: &[f64],
    l1_sigmas: &[f64],
    l2_freqs: &[f64],
    l2_sigmas: &[f64],
    l0_first: usize,
    l1_first: usize,
    dnu: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nl0 = l0_freqs.len().min(l0_sigmas.len());
    let nl1 = l1_freqs.len().min(l1_sigmas.len());
    let nl2 = l2_freqs.len().min(l2_sigmas.len());
    if nl0 <= l0_first + 1 || nl1 <= l1_first + 1 || nl2 == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let n = (nl0 - 1 - l0_first).min(nl1 - 1 - l1_first);
    let mut r02 = Vec::with_capacity(n);
    let mut sig_r02 = Vec::with_capacity(n);
    let mut r02_freqs = Vec::with_capacity(n);
    for i in 1..=n {
        let (i0, i1) = (l0_first + i, l1_first + i);
        r02_freqs.push(l0_freqs[i0]);
        let j_match = if dnu > 0.0 {
            (0..nl2).find(|&j| (l2_freqs[j] - l0_freqs[i0]).abs() < 0.25 * dnu)
        } else {
            (0..nl2)
                .find(|&j| l2_freqs[j] >= l1_freqs[i1 - 1] && l2_freqs[j] <= l0_freqs[i0])
                .or((i < nl2).then_some(i))
        };
        let Some(j) = j_match else {
            r02.push(0.0);
            sig_r02.push(0.0);
            continue;
        };
        let d02 = l0_freqs[i0] - l2_freqs[j];
        let dnu1 = (l1_freqs[i1] - l1_freqs[i1 - 1]).max(1.0e-30);
        r02.push(d02 / dnu1);
        let s_num = l0_sigmas[i0].powi(2) + l2_sigmas[j].powi(2);
        let s_den = l1_sigmas[i1].powi(2) + l1_sigmas[i1 - 1].powi(2);
        sig_r02.push((s_num / (dnu1 * dnu1) + s_den * (d02 * d02) / dnu1.powi(4)).sqrt());
    }
    (r02, sig_r02, r02_freqs)
}

/// Evaluates quadrupole-to-radial frequency separation ratios $r_{02}(i)$ and
/// uncertainties $\sigma_{r_{02}}(i)$ (`get_r02_frequency_ratios`, `astero_support.f90:421-486`).
#[must_use]
pub fn eval_frequency_ratios_r02(
    l0_freqs: &[f64],
    l0_sigmas: &[f64],
    l1_freqs: &[f64],
    l1_sigmas: &[f64],
    l2_freqs: &[f64],
    l2_sigmas: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let (r02, sig_r02, _) = eval_frequency_ratios_r02_with_dnu(
        l0_freqs, l0_sigmas, l1_freqs, l1_sigmas, l2_freqs, l2_sigmas, 0, 0, 0.0,
    );
    (r02, sig_r02)
}

/// Linearly interpolates a model $r_{010}$ frequency separation ratio at target
/// frequency `freq` (`interpolate_ratio_r010` in `astero_support.f90:489-520`).
#[must_use]
pub fn interpolate_ratio_r010(freq: f64, r010_freqs: &[f64], r010_vals: &[f64]) -> f64 {
    let n = r010_freqs.len().min(r010_vals.len());
    if n == 0 {
        return 0.0;
    }
    if freq <= r010_freqs[0] || n == 1 {
        return r010_vals[0];
    }
    for i in 0..(n - 1) {
        if freq >= r010_freqs[i] && freq <= r010_freqs[i + 1] {
            let df = r010_freqs[i + 1] - r010_freqs[i];
            if df.abs() <= 1.0e-30 {
                return r010_vals[i];
            }
            let beta = (freq - r010_freqs[i]) / df;
            return (1.0 - beta) * r010_vals[i] + beta * r010_vals[i + 1];
        }
    }
    r010_vals[n - 1]
}

/// Linearly interpolates a model $r_{02}$ frequency separation ratio at target
/// frequency `freq` using radial frequencies `l0_freqs[1..=n]` corresponding to
/// `r02_vals[0..n]` (`interpolate_ratio_r02` in `astero_support.f90:523-550`).
#[must_use]
pub fn interpolate_ratio_r02(freq: f64, l0_freqs: &[f64], r02_vals: &[f64]) -> f64 {
    if l0_freqs.len() < 2 || r02_vals.is_empty() {
        return 0.0;
    }
    let n = r02_vals.len().min(l0_freqs.len() - 1);
    interpolate_ratio_r010(freq, &l0_freqs[1..=n], &r02_vals[..n])
}

/// Weights and normalization controls for `eval_astero_chi2`
/// (`astero/defaults/astero_search.defaults:24-75`).
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct AsteroChi2Weights {
    pub chi2_seismo_fraction: f64,
    pub chi2_seismo_freq_fraction: Option<f64>,
    pub chi2_seismo_delta_nu_fraction: f64,
    pub chi2_seismo_nu_max_fraction: f64,
    pub chi2_seismo_r_010_fraction: f64,
    pub chi2_seismo_r_02_fraction: f64,
    pub normalize_chi2_seismo_frequencies: bool,
    pub normalize_chi2_seismo_r_010: bool,
    pub normalize_chi2_seismo_r_02: bool,
    pub normalize_chi2_spectro: bool,
}

impl Default for AsteroChi2Weights {
    fn default() -> Self {
        Self {
            chi2_seismo_fraction: 2.0 / 3.0, chi2_seismo_freq_fraction: None,
            chi2_seismo_delta_nu_fraction: 0.0, chi2_seismo_nu_max_fraction: 0.0,
            chi2_seismo_r_010_fraction: 0.0, chi2_seismo_r_02_fraction: 0.0,
            normalize_chi2_seismo_frequencies: true, normalize_chi2_seismo_r_010: true,
            normalize_chi2_seismo_r_02: true, normalize_chi2_spectro: true,
        }
    }
}

fn eval_component_chi2(obs: &[f64], sigmas: &[f64], model: &[f64], normalize: bool) -> f64 {
    let (mut sum, mut count) = (0.0_f64, 0usize);
    let n = obs.len().min(sigmas.len()).min(model.len());
    for i in 0..n {
        if sigmas[i] > 0.0 {
            let r = (model[i] - obs[i]) / sigmas[i];
            sum += r * r;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else if normalize { sum / (count as f64) } else { sum }
}

/// Evaluates the composite asteroseismic and spectroscopic $\chi^2$ hierarchy
/// matching `get_chi2` in `astero/private/astero_support.f90:1346-1527`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_astero_chi2(
    weights: &AsteroChi2Weights,
    obs_freqs: &[f64], obs_sigmas: &[f64], corr_model_freqs: &[f64],
    r010_obs: &[f64], r010_sigmas: &[f64], r010_model: &[f64],
    r02_obs: &[f64], r02_sigmas: &[f64], r02_model: &[f64],
    delta_nu_obs: f64, delta_nu_sigma: f64, delta_nu_model: f64,
    nu_max_obs: f64, nu_max_sigma: f64, nu_max_model: f64,
    spectro_obs: &[f64], spectro_sigmas: &[f64], spectro_model: &[f64],
) -> (f64, f64, f64, f64, f64, f64) {
    let chi2_freq = eval_component_chi2(obs_freqs, obs_sigmas, corr_model_freqs, weights.normalize_chi2_seismo_frequencies);
    let chi2_r010 = eval_component_chi2(r010_obs, r010_sigmas, r010_model, weights.normalize_chi2_seismo_r_010);
    let chi2_r02 = eval_component_chi2(r02_obs, r02_sigmas, r02_model, weights.normalize_chi2_seismo_r_02);
    let chi2_dnu = if delta_nu_sigma > 0.0 { ((delta_nu_obs - delta_nu_model) / delta_nu_sigma).powi(2) } else { 0.0 };
    let chi2_numax = if nu_max_sigma > 0.0 { ((nu_max_obs - nu_max_model) / nu_max_sigma).powi(2) } else { 0.0 };
    let f_freq = weights.chi2_seismo_freq_fraction.unwrap_or_else(|| {
        (1.0 - (weights.chi2_seismo_r_010_fraction
            + weights.chi2_seismo_r_02_fraction
            + weights.chi2_seismo_delta_nu_fraction
            + weights.chi2_seismo_nu_max_fraction))
            .clamp(0.0, 1.0)
    });
    let chi2_seismo = weights.chi2_seismo_r_010_fraction * chi2_r010
        + weights.chi2_seismo_r_02_fraction * chi2_r02
        + f_freq * chi2_freq
        + weights.chi2_seismo_delta_nu_fraction * chi2_dnu
        + weights.chi2_seismo_nu_max_fraction * chi2_numax;
    let chi2_spectro = eval_component_chi2(
        spectro_obs, spectro_sigmas, spectro_model, weights.normalize_chi2_spectro,
    );
    let f_seis = weights.chi2_seismo_fraction.clamp(0.0, 1.0);
    let chi2_total = f_seis * chi2_seismo + (1.0 - f_seis) * chi2_spectro;
    (chi2_total, chi2_seismo, chi2_spectro, chi2_freq, chi2_r010, chi2_r02)
}
