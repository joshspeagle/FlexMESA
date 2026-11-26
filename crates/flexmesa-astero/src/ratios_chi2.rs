// SPDX-License-Identifier: GPL-3.0-or-later
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
    let m_cand = cand_freqs
        .len()
        .min(cand_inertias.len())
        .min(cand_orders.len());
    let mut indices: Vec<usize> = (0..m_cand).collect();
    indices.sort_by(|&a, &b| {
        cand_freqs[a]
            .partial_cmp(&cand_freqs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
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
/// $\nu_i - \nu_{i-1} \le 1.5\Delta\nu$ (`get_max_sequence`,
/// `astero_support.f90:553-587`).
#[must_use]
pub fn get_max_sequence(freqs: &[f64], dnu: f64) -> (usize, usize) {
    let n = freqs.len();
    if n == 0 {
        return (0, 0);
    }
    if n == 1 || dnu <= 0.0 {
        return (0, n - 1);
    }
    let df_max = 1.5 * dnu;
    let (mut seq_i, mut seq_n) = (0usize, 0usize);
    let (mut max_seq_i, mut max_seq_n) = (0usize, 0usize);
    loop {
        let i = seq_i + seq_n;
        if i + 1 >= n {
            break;
        }
        seq_i = i;
        seq_n = 1;
        for j in seq_i..(n - 1) {
            if freqs[j + 1] - freqs[j] > df_max {
                if seq_n > max_seq_n {
                    max_seq_i = seq_i;
                    max_seq_n = seq_n;
                }
                break;
            }
            seq_n += 1;
        }
    }
    if seq_n > max_seq_n {
        max_seq_i = seq_i;
        max_seq_n = seq_n;
    }
    if max_seq_n == 0 {
        (0, 0)
    } else {
        (max_seq_i, max_seq_i + max_seq_n - 1)
    }
}

/// Full output of `get_frequency_ratios` (`astero_support.f90:318-416`).
#[derive(Debug, Clone, Default)]
pub struct FrequencyRatiosR010 {
    /// Number of $r_{01}$ and $r_{10}$ ratio pairs.
    pub n: usize,
    /// Starting 0-based index in `l0_freqs`.
    pub l0_first: usize,
    /// Starting 0-based index in `l1_freqs`.
    pub l1_first: usize,
    /// 5-point smoothed $r_{01}(i)$ ratios (`len = n`).
    pub ratios_r01: Vec<f64>,
    /// Propagated $1\sigma$ uncertainties $\sigma_{r_{01}}(i)$ (`len = n`).
    pub sigmas_r01: Vec<f64>,
    /// 5-point smoothed $r_{10}(i)$ ratios (`len = n`).
    pub ratios_r10: Vec<f64>,
    /// Propagated $1\sigma$ uncertainties $\sigma_{r_{10}}(i)$ (`len = n`).
    pub sigmas_r10: Vec<f64>,
    /// Interleaved $(r_{01}, r_{10})$ sequence (`len = 2 * n`).
    pub ratios_r010: Vec<f64>,
    /// Interleaved $(\sigma_{r_{01}}, \sigma_{r_{10}})$ uncertainties (`len = 2 * n`).
    pub sigmas_r010: Vec<f64>,
    /// Central frequencies $\nu$ ($\mu\mathrm{Hz}$) corresponding to `ratios_r010` (`len = 2 * n`).
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
        let (mut l1_f, mut l1_l) = get_max_sequence(&l1_freqs[..nl1], dnu);
        // Trim high end of l0 until < last l1 (`astero_support.f90:349-358`).
        while l0_freqs[l0_l] >= l1_freqs[l1_l] {
            if l0_l <= l0_f {
                return FrequencyRatiosR010::default();
            }
            l0_l -= 1;
        }
        // Trim low end of l1 until > first l0 (`astero_support.f90:361-367`).
        while l1_freqs[l1_f] <= l0_freqs[l0_f] {
            l1_f += 1;
            if l1_f > l1_l {
                return FrequencyRatiosR010::default();
            }
        }
        // Trim low end of l0 until only 1 < 1st l1 (`astero_support.f90:370-374`).
        while l0_f < l0_l && l0_freqs[l0_f + 1] < l1_freqs[l1_f] {
            l0_f += 1;
        }
        // Trim high end of l1 until only 1 > last l0 (`astero_support.f90:377-381`).
        while l1_l > l1_f && l1_freqs[l1_l - 1] > l0_freqs[l0_l] {
            l1_l -= 1;
        }
        let l0_seq_n = l0_l.saturating_sub(l0_f) + 1;
        let l1_seq_n = l1_l.saturating_sub(l1_f) + 1;
        if l0_seq_n != l1_seq_n || l0_seq_n < 3 {
            return FrequencyRatiosR010::default();
        }
        (l0_f, l1_f, l0_seq_n - 2)
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
    let len = l0_freqs
        .len()
        .min(l0_sigmas.len())
        .min(l1_freqs.len())
        .min(l1_sigmas.len());
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
        let s_d01_sq = (flexmesa_math::pow2(l0_sigmas[i - 1])
            + 16.0 * flexmesa_math::pow2(l1_sigmas[i - 1])
            + 36.0 * flexmesa_math::pow2(l0_sigmas[i])
            + 16.0 * flexmesa_math::pow2(l1_sigmas[i])
            + flexmesa_math::pow2(l0_sigmas[i + 1]))
            / 64.0;
        let s_dnu1_sq = flexmesa_math::pow2(l1_sigmas[i])
            + flexmesa_math::pow2(l1_sigmas[i - 1]);
        sig_r01.push(
            (s_d01_sq / (dnu1 * dnu1)
                + s_dnu1_sq * (d01 * d01) / flexmesa_math::pow4(dnu1))
            .sqrt(),
        );

        let d10 = -(l1_freqs[i - 1] - 4.0 * l0_freqs[i] + 6.0 * l1_freqs[i]
            - 4.0 * l0_freqs[i + 1]
            + l1_freqs[i + 1])
            / 8.0;
        let dnu0 = (l0_freqs[i + 1] - l0_freqs[i]).max(1.0e-30);
        r10.push(d10 / dnu0);
        let s_d10_sq = (flexmesa_math::pow2(l1_sigmas[i - 1])
            + 16.0 * flexmesa_math::pow2(l0_sigmas[i])
            + 36.0 * flexmesa_math::pow2(l1_sigmas[i])
            + 16.0 * flexmesa_math::pow2(l0_sigmas[i + 1])
            + flexmesa_math::pow2(l1_sigmas[i + 1]))
            / 64.0;
        let s_dnu0_sq = flexmesa_math::pow2(l0_sigmas[i + 1])
            + flexmesa_math::pow2(l0_sigmas[i]);
        sig_r10.push(
            (s_d10_sq / (dnu0 * dnu0)
                + s_dnu0_sq * (d10 * d10) / flexmesa_math::pow4(dnu0))
            .sqrt(),
        );
    }
    (r01, sig_r01, r10, sig_r10)
}

/// Matched observed $r_{02}$ ratios and fixed `i2_for_r02` pairings
/// (`get_r02_frequency_ratios`, `astero_support.f90:421-486`).
#[derive(Debug, Clone, Default)]
pub struct FrequencyRatiosR02Obs {
    /// Observed $r_{02}(i)$ ratios (`len = ratios_n`).
    pub ratios_r02: Vec<f64>,
    /// Propagated uncertainties $\sigma_{r_{02}}(i)$ (`0.0` when unpaired).
    pub sigmas_r02: Vec<f64>,
    /// Central radial mode frequencies $\nu_{0, i + \mathrm{l0\_first}}$ ($\mu\mathrm{Hz}$).
    pub r02_freqs: Vec<f64>,
    /// Fixed 0-based `l2` mode index (`None` when unpaired) reused for models.
    pub i2_for_r02: Vec<Option<usize>>,
}

/// Computes observed $r_{02}$ ratios and fixes `i2_for_r02(i)` using the
/// $0.25\Delta\nu$ window around $l=0$ target frequencies (`init = .true.` in
/// `get_r02_frequency_ratios`, `astero_support.f90:421-486`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_r02_obs_with_indices(
    l0_freqs: &[f64],
    l0_sigmas: &[f64],
    l1_freqs: &[f64],
    l1_sigmas: &[f64],
    l2_freqs: &[f64],
    l2_sigmas: &[f64],
    ratios_n: usize,
    l0_first: usize,
    l1_first: usize,
) -> FrequencyRatiosR02Obs {
    let nl0 = l0_freqs.len().min(l0_sigmas.len());
    let nl1 = l1_freqs.len().min(l1_sigmas.len());
    let nl2 = l2_freqs.len().min(l2_sigmas.len());
    if ratios_n == 0
        || nl0 <= l0_first + ratios_n
        || nl1 <= l1_first + ratios_n
        || nl2 == 0
    {
        return FrequencyRatiosR02Obs::default();
    }
    let mut ratios_r02 = Vec::with_capacity(ratios_n);
    let mut sigmas_r02 = Vec::with_capacity(ratios_n);
    let mut r02_freqs = Vec::with_capacity(ratios_n);
    let mut i2_for_r02 = Vec::with_capacity(ratios_n);
    for i in 1..=ratios_n {
        let (i0, i1) = (l0_first + i, l1_first + i);
        let dnu = l1_freqs[i1] - l1_freqs[i1 - 1];
        let f0 = l0_freqs[i0];
        let df = 0.25 * dnu;
        let (fmin, fmax) = (f0 - df, f0 + df);
        let mut dfmin = 1.0e99_f64;
        let mut jmin: Option<usize> = None;
        for j in 0..nl2 {
            let f2 = l2_freqs[j];
            if f2 <= fmax && f2 >= fmin && (f2 - f0).abs() < dfmin {
                dfmin = (f2 - f0).abs();
                jmin = Some(j);
            }
            if f2 > f0 {
                break;
            }
        }
        r02_freqs.push(f0);
        i2_for_r02.push(jmin);
        if let Some(j) = jmin {
            let d02 = f0 - l2_freqs[j];
            ratios_r02.push(d02 / dnu);
            let s_num = (flexmesa_math::pow2(l0_sigmas[i0])
                + flexmesa_math::pow2(l2_sigmas[j]))
            .sqrt();
            let s_den = (flexmesa_math::pow2(l1_sigmas[i1])
                + flexmesa_math::pow2(l1_sigmas[i1 - 1]))
            .sqrt();
            sigmas_r02.push(
                (flexmesa_math::pow2(s_num / dnu)
                    + flexmesa_math::pow2(s_den * d02 / (dnu * dnu)))
                .sqrt(),
            );
        } else {
            ratios_r02.push(f64::NAN);
            sigmas_r02.push(0.0);
        }
    }
    FrequencyRatiosR02Obs {
        ratios_r02,
        sigmas_r02,
        r02_freqs,
        i2_for_r02,
    }
}

/// Computes model $r_{02}$ ratios reusing fixed `i2_for_r02` pairings from
/// observations (`init = .false.` in `get_r02_frequency_ratios`,
/// `astero_support.f90:466-475`).
#[must_use]
pub fn eval_r02_model_with_indices(
    l0_freqs: &[f64],
    l1_freqs: &[f64],
    l2_freqs: &[f64],
    l0_first: usize,
    l1_first: usize,
    obs: &FrequencyRatiosR02Obs,
) -> (Vec<f64>, Vec<f64>) {
    let n = obs.i2_for_r02.len();
    let mut r02 = Vec::with_capacity(n);
    let mut r02_freqs = Vec::with_capacity(n);
    for (k, &j_opt) in obs.i2_for_r02.iter().enumerate() {
        let i = k + 1;
        let (i0, i1) = (l0_first + i, l1_first + i);
        if obs.sigmas_r02[k] == 0.0 {
            continue;
        }
        let Some(j) = j_opt else {
            continue;
        };
        if i0 >= l0_freqs.len() || i1 >= l1_freqs.len() || j >= l2_freqs.len() {
            continue;
        }
        let dnu = l1_freqs[i1] - l1_freqs[i1 - 1];
        r02.push((l0_freqs[i0] - l2_freqs[j]) / dnu);
        r02_freqs.push(l0_freqs[i0]);
    }
    (r02, r02_freqs)
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
                .find(|&j| {
                    l2_freqs[j] >= l1_freqs[i1 - 1]
                        && l2_freqs[j] <= l0_freqs[i0]
                })
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
        let s_num = flexmesa_math::pow2(l0_sigmas[i0])
            + flexmesa_math::pow2(l2_sigmas[j]);
        let s_den = flexmesa_math::pow2(l1_sigmas[i1])
            + flexmesa_math::pow2(l1_sigmas[i1 - 1]);
        sig_r02.push(
            (s_num / (dnu1 * dnu1)
                + s_den * (d02 * d02) / flexmesa_math::pow4(dnu1))
            .sqrt(),
        );
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
        l0_freqs, l0_sigmas, l1_freqs, l1_sigmas, l2_freqs, l2_sigmas, 0, 0,
        0.0,
    );
    (r02, sig_r02)
}

/// Linearly interpolates a model $r_{010}$ frequency separation ratio at target
/// frequency `freq` (`interpolate_ratio_r010` in `astero_support.f90:489-520`).
#[must_use]
pub fn interpolate_ratio_r010(
    freq: f64,
    r010_freqs: &[f64],
    r010_vals: &[f64],
) -> f64 {
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
/// frequency `freq` using radial frequencies `l0_freqs` corresponding to
/// `r02_vals` (`interpolate_ratio_r02` in `astero_support.f90:523-550`).
#[must_use]
pub fn interpolate_ratio_r02(
    freq: f64,
    l0_freqs: &[f64],
    r02_vals: &[f64],
) -> f64 {
    if l0_freqs.is_empty() || r02_vals.is_empty() {
        return 0.0;
    }
    if l0_freqs.len() > r02_vals.len() {
        let n = r02_vals.len().min(l0_freqs.len() - 1);
        interpolate_ratio_r010(freq, &l0_freqs[1..=n], &r02_vals[..n])
    } else {
        let n = r02_vals.len().min(l0_freqs.len());
        interpolate_ratio_r010(freq, &l0_freqs[..n], &r02_vals[..n])
    }
}

/// Weights and normalization controls for `eval_astero_chi2`
/// (`astero/defaults/astero_search.defaults:24-75`).
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct AsteroChi2Weights {
    /// Weight $f_{\mathrm{seismo}} \in [0, 1]$ on $\chi^2_{\mathrm{seismo}}$ in total $\chi^2$.
    pub chi2_seismo_fraction: f64,
    /// Optional explicit weight on $\chi^2_{\mathrm{freq}}$ inside $\chi^2_{\mathrm{seismo}}$.
    pub chi2_seismo_freq_fraction: Option<f64>,
    /// Weight on $\chi^2_{\Delta\nu}$ inside $\chi^2_{\mathrm{seismo}}$.
    pub chi2_seismo_delta_nu_fraction: f64,
    /// Weight on $\chi^2_{\nu_{\max}}$ inside $\chi^2_{\mathrm{seismo}}$.
    pub chi2_seismo_nu_max_fraction: f64,
    /// Weight on $\chi^2_{r_{010}}$ inside $\chi^2_{\mathrm{seismo}}$.
    pub chi2_seismo_r_010_fraction: f64,
    /// Weight on $\chi^2_{r_{02}}$ inside $\chi^2_{\mathrm{seismo}}$.
    pub chi2_seismo_r_02_fraction: f64,
    /// Divide $\chi^2_{\mathrm{freq}}$ by the number of matched frequencies.
    pub normalize_chi2_seismo_frequencies: bool,
    /// Divide $\chi^2_{r_{010}}$ by the number of $r_{010}$ terms.
    pub normalize_chi2_seismo_r_010: bool,
    /// Divide $\chi^2_{r_{02}}$ by the number of $r_{02}$ terms.
    pub normalize_chi2_seismo_r_02: bool,
    /// Divide $\chi^2_{\mathrm{spectro}}$ by the number of active spectroscopic constraints.
    pub normalize_chi2_spectro: bool,
}

impl Default for AsteroChi2Weights {
    fn default() -> Self {
        Self {
            // `astero/defaults/astero_search.defaults:24`: 0.667d0.
            chi2_seismo_fraction: 0.667,
            chi2_seismo_freq_fraction: None,
            chi2_seismo_delta_nu_fraction: 0.0,
            chi2_seismo_nu_max_fraction: 0.0,
            chi2_seismo_r_010_fraction: 0.0,
            chi2_seismo_r_02_fraction: 0.0,
            normalize_chi2_seismo_frequencies: true,
            normalize_chi2_seismo_r_010: true,
            normalize_chi2_seismo_r_02: true,
            normalize_chi2_spectro: true,
        }
    }
}

fn eval_component_chi2(
    obs: &[f64],
    sigmas: &[f64],
    model: &[f64],
    normalize: bool,
) -> f64 {
    let (mut sum, mut count) = (0.0_f64, 0usize);
    let n = obs.len().min(sigmas.len()).min(model.len());
    for i in 0..n {
        if sigmas[i] > 0.0 {
            let r = (model[i] - obs[i]) / sigmas[i];
            sum += r * r;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else if normalize {
        sum / (count as f64)
    } else {
        sum
    }
}

/// Evaluates the composite asteroseismic and spectroscopic $\chi^2$ hierarchy
/// matching `get_chi2` in `astero/private/astero_support.f90:1346-1527`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_astero_chi2(
    weights: &AsteroChi2Weights,
    obs_freqs: &[f64],
    obs_sigmas: &[f64],
    corr_model_freqs: &[f64],
    r010_obs: &[f64],
    r010_sigmas: &[f64],
    r010_model: &[f64],
    r02_obs: &[f64],
    r02_sigmas: &[f64],
    r02_model: &[f64],
    delta_nu_obs: f64,
    delta_nu_sigma: f64,
    delta_nu_model: f64,
    nu_max_obs: f64,
    nu_max_sigma: f64,
    nu_max_model: f64,
    spectro_obs: &[f64],
    spectro_sigmas: &[f64],
    spectro_model: &[f64],
) -> (f64, f64, f64, f64, f64, f64) {
    let chi2_freq = eval_component_chi2(
        obs_freqs,
        obs_sigmas,
        corr_model_freqs,
        weights.normalize_chi2_seismo_frequencies,
    );
    let chi2_r010 = eval_component_chi2(
        r010_obs,
        r010_sigmas,
        r010_model,
        weights.normalize_chi2_seismo_r_010,
    );
    let chi2_r02 = eval_component_chi2(
        r02_obs,
        r02_sigmas,
        r02_model,
        weights.normalize_chi2_seismo_r_02,
    );
    let chi2_dnu = if delta_nu_sigma > 0.0 {
        flexmesa_math::pow2((delta_nu_obs - delta_nu_model) / delta_nu_sigma)
    } else {
        0.0
    };
    let chi2_numax = if nu_max_sigma > 0.0 {
        flexmesa_math::pow2((nu_max_obs - nu_max_model) / nu_max_sigma)
    } else {
        0.0
    };
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
        spectro_obs,
        spectro_sigmas,
        spectro_model,
        weights.normalize_chi2_spectro,
    );
    let f_seis = weights.chi2_seismo_fraction.clamp(0.0, 1.0);
    let chi2_total = f_seis * chi2_seismo + (1.0 - f_seis) * chi2_spectro;
    (
        chi2_total,
        chi2_seismo,
        chi2_spectro,
        chi2_freq,
        chi2_r010,
        chi2_r02,
    )
}
