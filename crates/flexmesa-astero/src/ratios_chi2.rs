// SPDX-License-Identifier: GPL-3.0-or-later
//! Mode matching, Roxburgh & Vorontsov (2003) frequency separation ratios
//! ($r_{01}, r_{10}, r_{02}$), and composite asteroseismic + spectroscopic
//! $\chi^2$ evaluation (`astero/private/astero_support.f90:148-587, 1346-1527`).

use flexmesa_star::controls::StarError;

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

/// Finds the longest contiguous radial-order subsequence `(seq_first, seq_count)`
/// with 1-based start index `seq_first` and length `seq_count` (`(0, 0)` when
/// `freqs.len() <= 1`) where adjacent modes satisfy
/// $\nu_{j+1} - \nu_j \le 1.5\Delta\nu$ (`get_max_sequence`,
/// `astero/private/astero_support.f90:553-587`).
#[must_use]
pub fn get_max_sequence(freqs: &[f64], dnu: f64) -> (usize, usize) {
    let nl = freqs.len();
    let (mut max_seq_i, mut max_seq_n) = (0usize, 0usize);
    let (mut seq_i, mut seq_n) = (0usize, 0usize);
    loop {
        let i = seq_i + seq_n + 1;
        if i >= nl {
            break;
        }
        seq_i = i;
        seq_n = 1;
        for j in seq_i..nl {
            if freqs[j] - freqs[j - 1] > 1.5 * dnu {
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
    (max_seq_i, max_seq_n)
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
/// $l=0/l=1$ interleaving (`astero/private/astero_support.f90:318-416`).
///
/// # Errors
///
/// Returns `StarError::InvalidInput` when `trim_for_obs` is `true`, `l1` is
/// non-empty, and `get_max_sequence` returns an empty sequence (`count == 0`)
/// where upstream computes out-of-bounds `last = first - 1`
/// (`astero/private/astero_support.f90:341-350`; `DECISIONS #43`, `#169` / `#644`).
#[allow(clippy::too_many_arguments)]
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
) -> Result<FrequencyRatiosR010, StarError> {
    let nl0 = l0_freqs.len().min(l0_sigmas.len());
    let nl1 = l1_freqs.len().min(l1_sigmas.len());
    if nl1 == 0 {
        return Ok(FrequencyRatiosR010::default());
    }
    let (l0_first, l1_first, n) = if trim_for_obs {
        let (l0_seq_i, l0_seq_cnt) = get_max_sequence(&l0_freqs[..nl0], dnu);
        let (l1_seq_i, l1_seq_cnt) = get_max_sequence(&l1_freqs[..nl1], dnu);
        if l0_seq_cnt == 0 || l1_seq_cnt == 0 {
            return Err(StarError::InvalidInput(format!(
                "get_max_sequence returned empty sequence \
                 (l0_count={l0_seq_cnt}, l1_count={l1_seq_cnt}); upstream \
                 computes out-of-bounds last = first - 1 \
                 (astero/private/astero_support.f90:341-350; DECISIONS #43, #169 / #644)"
            )));
        }
        let mut l0_f = l0_seq_i - 1;
        let mut l0_l = l0_f + l0_seq_cnt - 1;
        let mut l1_f = l1_seq_i - 1;
        let mut l1_l = l1_f + l1_seq_cnt - 1;
        // Trim high end of l0 until < last l1 (`astero_support.f90:349-358`).
        while l0_freqs[l0_l] >= l1_freqs[l1_l] {
            if l0_l <= l0_f {
                return Ok(FrequencyRatiosR010::default());
            }
            l0_l -= 1;
        }
        // Trim low end of l1 until > first l0 (`astero_support.f90:361-367`).
        while l1_freqs[l1_f] <= l0_freqs[l0_f] {
            l1_f += 1;
            if l1_f > l1_l {
                return Ok(FrequencyRatiosR010::default());
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
            return Ok(FrequencyRatiosR010::default());
        }
        (l0_f, l1_f, l0_seq_n - 2)
    } else if nl0 < 3 || nl1 < 3 {
        return Ok(FrequencyRatiosR010::default());
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
    Ok(FrequencyRatiosR010 {
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
    })
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

/// Computes model $r_{02}$ ratios positionally aligned with the observed
/// `ratios_r02` / `sigmas_r02` (`model_ratios_r02(i)`, `init = .false.` in
/// `get_r02_frequency_ratios`, `astero_support.f90:466-475`).
///
/// Unlike [`eval_r02_model_with_indices`], which compacts the output, entry
/// `k` always corresponds to observed ratio `k` (`i = k + 1`). Unpaired
/// entries (`sigmas_r02 == 0` or no fixed `l = 2` partner) stay `NaN`,
/// matching `fill_with_NaNs(r02)` upstream, and are never read by
/// [`interpolate_ratio_r02_masked`].
#[must_use]
pub fn eval_r02_model_aligned(
    l0_freqs: &[f64],
    l1_freqs: &[f64],
    l2_freqs: &[f64],
    l0_first: usize,
    l1_first: usize,
    obs: &FrequencyRatiosR02Obs,
) -> Vec<f64> {
    let mut r02 = vec![f64::NAN; obs.i2_for_r02.len()];
    for (k, &j_opt) in obs.i2_for_r02.iter().enumerate() {
        let (i0, i1) = (l0_first + k + 1, l1_first + k + 1);
        if obs.sigmas_r02[k] == 0.0 {
            continue;
        }
        let Some(j) = j_opt else {
            continue;
        };
        if i0 >= l0_freqs.len() || i1 >= l1_freqs.len() || j >= l2_freqs.len() {
            continue;
        }
        let d02 = l0_freqs[i0] - l2_freqs[j];
        let dnu = l1_freqs[i1] - l1_freqs[i1 - 1];
        r02[k] = d02 / dnu;
    }
    r02
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
/// frequency `freq` (`interpolate_ratio_r010` in
/// `astero/private/astero_support.f90:489-517`).
#[must_use]
pub fn interpolate_ratio_r010(
    freq: f64,
    r010_freqs: &[f64],
    r010_vals: &[f64],
) -> f64 {
    let ratios_n = r010_freqs.len().min(r010_vals.len());
    if ratios_n == 0 {
        return 0.0;
    }
    if freq <= r010_freqs[0] {
        return r010_vals[0];
    }
    if freq >= r010_freqs[ratios_n - 1] {
        return r010_vals[ratios_n - 1];
    }
    for j in 1..ratios_n {
        if freq < r010_freqs[j] {
            let alfa = (freq - r010_freqs[j - 1])
                / (r010_freqs[j] - r010_freqs[j - 1]);
            let beta = 1.0 - alfa;
            return alfa * r010_vals[j] + beta * r010_vals[j - 1];
        }
    }
    0.0
}

/// Linearly interpolates a model $r_{02}$ ratio at target frequency `freq`
/// (`interpolate_ratio_r02`, `astero_support.f90:520-550`), skipping every
/// entry with `sigmas_r02(i) == 0` and treating all entries as active.
///
/// Convenience wrapper over [`interpolate_ratio_r02_masked`] with
/// `nl0 = min(model_freqs.len(), model_ratios.len())`; `model_freqs[k]` and
/// `model_ratios[k]` must be positionally aligned.
#[must_use]
pub fn interpolate_ratio_r02(
    freq: f64,
    model_freqs: &[f64],
    model_ratios: &[f64],
) -> f64 {
    let n = model_freqs.len().min(model_ratios.len());
    interpolate_ratio_r02_masked(
        freq,
        model_freqs,
        model_ratios,
        &vec![1.0; n],
        n,
    )
}

/// Bit-faithful port of MESA `interpolate_ratio_r02`
/// (`astero_support.f90:520-550`; called per active `i` from
/// `get_chi2_seismo`, `astero_support.f90:1451-1458`).
///
/// All slices are the 0-based view of upstream's 1-based callee arrays, so
/// `model_freqs[k]` is `model_freq(0, k + 1)`, **unshifted** by
/// `ratios_l0_first`; `model_ratios[k]` is `model_ratios_r02(k + 1)` and
/// `sigmas_r02[k]` is `sigmas_r02(k + 1)`. Loops run over `i = 1..=nl0`
/// (here `0..min(nl0, slice lengths)`) and skip `sigmas_r02(i) == 0`.
///
/// Upstream quirks kept on purpose: `i_lo` is fixed at the first active
/// entry and is never advanced inside the scan, so the bracket is
/// `[i_lo, i_hi]` (not neighbouring modes); the low end clamps to
/// `model_ratios(i_lo)`, and `freq` above every active entry returns
/// `model_ratios(i_hi)` of the last active entry. Returns `0` when no entry
/// is active. `alfa = (freq - f_lo) / (f_hi - f_lo)`, `beta = 1 - alfa`, and
/// `ratio = alfa * r_hi + beta * r_lo` use upstream's operand order.
#[must_use]
pub fn interpolate_ratio_r02_masked(
    freq: f64,
    model_freqs: &[f64],
    model_ratios: &[f64],
    sigmas_r02: &[f64],
    nl0: usize,
) -> f64 {
    let n = nl0
        .min(sigmas_r02.len())
        .min(model_freqs.len())
        .min(model_ratios.len());
    let Some(i_lo) = (0..n).find(|&i| sigmas_r02[i] != 0.0) else {
        return 0.0;
    };
    if freq <= model_freqs[i_lo] {
        return model_ratios[i_lo];
    }
    let mut i_hi = i_lo;
    for i in (i_lo + 1)..n {
        if sigmas_r02[i] == 0.0 {
            continue;
        }
        i_hi = i;
        if freq > model_freqs[i] {
            continue;
        }
        let alfa = (freq - model_freqs[i_lo])
            / (model_freqs[i_hi] - model_freqs[i_lo]);
        let beta = 1.0 - alfa;
        return alfa * model_ratios[i_hi] + beta * model_ratios[i_lo];
    }
    model_ratios[i_hi]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_max_sequence_matches_mesa_oracle_edge_cases() {
        let normal = [900.0, 960.0, 1060.0, 1160.0, 1260.0, 1360.0, 1520.0];
        assert_eq!(get_max_sequence(&normal, 100.0), (1, 6));
        assert_eq!(get_max_sequence(&[], 100.0), (0, 0));
        assert_eq!(get_max_sequence(&normal[..1], 100.0), (0, 0));

        // Non-positive delta_nu causes every spacing > 1.5 * dnu -> length 1.
        assert_eq!(get_max_sequence(&normal[..5], 0.0), (1, 1));
        assert_eq!(get_max_sequence(&normal[..5], -1.0), (1, 1));

        // Upstream skips one element (`seq_i + seq_n + 1`) after each gap.
        let gap = [900.0, 1100.0, 1200.0, 1300.0, 1400.0];
        assert_eq!(get_max_sequence(&gap[..4], 100.0), (3, 2));
        assert_eq!(get_max_sequence(&gap, 100.0), (3, 3));

        // Ties keep the earliest sequence (`seq_n > max_seq_n`).
        let tie = [900.0, 1000.0, 1200.0, 1300.0, 1400.0];
        assert_eq!(get_max_sequence(&tie, 100.0), (1, 2));
    }

    #[test]
    fn eval_frequency_ratios_with_trimming_handles_empty_and_single_mode() {
        let l0 = [900.0, 1000.0, 1100.0, 1200.0];
        let s0 = [0.1; 4];

        // `nl1 == 0` returns empty ratios cleanly (`astero_support.f90:339`).
        let no_l1 = eval_frequency_ratios_with_trimming(
            &l0,
            &s0,
            &[],
            &[],
            100.0,
            true,
            0,
            0,
            0,
        )
        .expect("nl1 == 0 should return Ok(default)");
        assert_eq!(no_l1.n, 0);

        // `nl0 == 1` or `nl1 == 1` returns explicit deferred #169 error.
        let l1 = [950.0, 1050.0, 1150.0, 1250.0];
        let s1 = [0.1; 4];
        let err_l0 = eval_frequency_ratios_with_trimming(
            &l0[..1],
            &s0[..1],
            &l1,
            &s1,
            100.0,
            true,
            0,
            0,
            0,
        )
        .expect_err("nl0 == 1 with trim_for_obs must error");
        assert!(
            matches!(err_l0, StarError::InvalidInput(ref msg) if msg.contains("#169")),
            "expected #169 citation, got {err_l0:?}"
        );
        let err_l1 = eval_frequency_ratios_with_trimming(
            &l0,
            &s0,
            &l1[..1],
            &s1[..1],
            100.0,
            true,
            0,
            0,
            0,
        )
        .expect_err("nl1 == 1 with trim_for_obs must error");
        assert!(
            matches!(err_l1, StarError::InvalidInput(ref msg) if msg.contains("#169")),
            "expected #169 citation, got {err_l1:?}"
        );

        // `nl0 == 2, nl1 == 2` exits with `n == 0` (`astero_support.f90:389`).
        let too_short = eval_frequency_ratios_with_trimming(
            &l0[..2],
            &s0[..2],
            &l1[..2],
            &s1[..2],
            100.0,
            true,
            0,
            0,
            0,
        )
        .expect("nl0 == 2, nl1 == 2 should return Ok(default)");
        assert_eq!(too_short.n, 0);
    }

    #[test]
    fn interpolate_ratio_r010_matches_mesa_oracle_bit_exact() {
        let freqs = [1800.0, 1935.0, 2070.0, 2205.0];
        let vals = [
            0.0123456789012345,
            -0.0234567890123456,
            0.0345678901234567,
            -0.0456789012345678,
        ];
        assert_eq!(
            interpolate_ratio_r010(1750.0, &freqs, &vals).to_bits(),
            0x3F89_48B0_F905_91BF
        );
        assert_eq!(
            interpolate_ratio_r010(1800.0, &freqs, &vals).to_bits(),
            0x3F89_48B0_F905_91BF
        );
        assert_eq!(
            interpolate_ratio_r010(1999.1234567, &freqs, &vals).to_bits(),
            0x3F70_CFA3_A7CC_C6F2
        );
        assert_eq!(
            interpolate_ratio_r010(2205.0, &freqs, &vals).to_bits(),
            0xBFA7_6339_95D6_E0A7
        );
        assert_eq!(
            interpolate_ratio_r010(2300.0, &freqs, &vals).to_bits(),
            0xBFA7_6339_95D6_E0A7
        );
        assert_eq!(
            interpolate_ratio_r010(f64::NAN, &freqs, &vals).to_bits(),
            0.0_f64.to_bits()
        );

        // `ratios_n == 0` and `ratios_n == 1` cases.
        assert_eq!(interpolate_ratio_r010(1900.0, &[], &[]).to_bits(), 0);
        assert_eq!(
            interpolate_ratio_r010(2500.0, &freqs[..1], &vals[..1]).to_bits(),
            0x3F89_48B0_F905_91BF
        );
        assert_eq!(
            interpolate_ratio_r010(f64::NAN, &freqs[..1], &vals[..1]).to_bits(),
            0
        );

        // Duplicate and non-monotonic model frequencies follow upstream scan.
        let dup_f = [100.0, 200.0, 200.0, 300.0];
        let dup_v = [10.0, 20.0, 30.0, 40.0];
        assert_eq!(
            interpolate_ratio_r010(200.0, &dup_f, &dup_v).to_bits(),
            0x403E_0000_0000_0000
        );
        let nm_f1 = [100.0, 300.0, 200.0, 400.0];
        assert_eq!(
            interpolate_ratio_r010(250.0, &nm_f1, &dup_v).to_bits(),
            0x4031_8000_0000_0000
        );
        let nm_f2 = [100.0, 200.0, 200.0, 150.0];
        assert_eq!(
            interpolate_ratio_r010(160.0, &nm_f2, &dup_v).to_bits(),
            0x4044_0000_0000_0000
        );
    }

    #[test]
    fn interpolate_ratio_r02_masked_keeps_stale_i_lo_bracket() {
        let f = [100.0, 200.0, 300.0, 400.0];
        let r = [1.0, 4.0, 9.0, 16.0];
        let s = [1.0; 4];
        // `i_lo` stays at entry 0 (`astero_support.f90:529-531`): bracket
        // is [100, 300], so 0.75 * 9 + 0.25 * 1 = 7 (neighbours give 6.5).
        let got = interpolate_ratio_r02_masked(250.0, &f, &r, &s, 4);
        assert_eq!(got.to_bits(), 7.0_f64.to_bits());
        // Low clamp (`freq <= model_freqs(i_lo)`), including equality.
        assert_eq!(interpolate_ratio_r02_masked(50.0, &f, &r, &s, 4), 1.0);
        assert_eq!(interpolate_ratio_r02_masked(100.0, &f, &r, &s, 4), 1.0);
        // `freq == model_freqs(i)` takes alfa = 1 against `i_lo`.
        assert_eq!(interpolate_ratio_r02_masked(400.0, &f, &r, &s, 4), 16.0);
        // Above all active entries returns the last active ratio.
        assert_eq!(interpolate_ratio_r02_masked(500.0, &f, &r, &s, 4), 16.0);
        // `nl0` bounds the scan (`do i = i_lo + 1, nl(0)`).
        assert_eq!(interpolate_ratio_r02_masked(250.0, &f, &r, &s, 2), 4.0);
    }

    #[test]
    fn interpolate_ratio_r02_masked_skips_zero_sigma_entries() {
        let f = [100.0, 200.0, 300.0, 400.0];
        let r = [1.0, 4.0, 9.0, 16.0];
        let s = [0.0, 1.0, 0.0, 1.0];
        // `i_lo` is the first entry with `sigmas_r02 /= 0` (index 1).
        assert_eq!(interpolate_ratio_r02_masked(150.0, &f, &r, &s, 4), 4.0);
        // Entry 2 is skipped; bracket is [200, 400]: 0.5 * 16 + 0.5 * 4.
        assert_eq!(interpolate_ratio_r02_masked(300.0, &f, &r, &s, 4), 10.0);
        assert_eq!(interpolate_ratio_r02_masked(450.0, &f, &r, &s, 4), 16.0);
        // A single active entry returns its own ratio for any frequency.
        let one = [0.0, 0.0, 1.0, 0.0];
        assert_eq!(interpolate_ratio_r02_masked(10.0, &f, &r, &one, 4), 9.0);
        assert_eq!(interpolate_ratio_r02_masked(999.0, &f, &r, &one, 4), 9.0);
        // No active entry returns zero (`if (i_lo == 0) return`).
        assert_eq!(
            interpolate_ratio_r02_masked(250.0, &f, &r, &[0.0; 4], 4),
            0.0
        );
        // `sigmas_r02` entries past `ratios_n` are zero upstream.
        assert_eq!(
            interpolate_ratio_r02_masked(250.0, &f, &r, &[0.0, 0.0, 1.0], 4),
            9.0
        );
    }

    #[test]
    fn interpolate_ratio_r02_unmasked_wrapper_matches_all_active_mask() {
        let f = [100.0, 200.0, 300.0];
        let r = [1.0, 4.0, 9.0];
        let want = interpolate_ratio_r02_masked(150.0, &f, &r, &[1.0; 3], 3);
        assert_eq!(
            interpolate_ratio_r02(150.0, &f, &r).to_bits(),
            want.to_bits()
        );
        assert_eq!(interpolate_ratio_r02(150.0, &[], &[]), 0.0);
    }

    #[test]
    fn r02_pipeline_uses_unshifted_model_axis_with_nonzero_l0_first() {
        // `ratios_l0_first = 1`, `ratios_n = 2`: observed ratios sit at
        // `l0[2], l0[3]` but upstream passes `model_freq(0, 1:)` unshifted.
        let l0 = [900.0, 1000.0, 1100.0, 1200.0, 1300.0];
        let l1 = [950.0, 1050.0, 1150.0, 1250.0, 1350.0];
        let sg = [0.1; 5];
        let l2 = [1090.0, 1195.0];
        let obs = eval_r02_obs_with_indices(
            &l0, &sg, &l1, &sg, &l2, &[0.1; 2], 2, 1, 1,
        );
        assert_eq!(obs.i2_for_r02, vec![Some(0), Some(1)]);
        assert_eq!(obs.r02_freqs, vec![1100.0, 1200.0]);

        let m0 = [905.0, 1005.0, 1105.0, 1205.0, 1305.0];
        let m1 = [955.0, 1055.0, 1155.0, 1255.0, 1355.0];
        let m2 = [1095.0, 1198.0];
        let mr = eval_r02_model_aligned(&m0, &m1, &m2, 1, 1, &obs);
        assert_eq!(mr.len(), 2);
        assert_eq!(mr[0].to_bits(), ((1105.0_f64 - 1095.0) / 100.0).to_bits());
        assert_eq!(mr[1].to_bits(), ((1205.0_f64 - 1198.0) / 100.0).to_bits());

        // Sigmas padded with zeros to `nl(0)` (`sigmas_r02 = 0d0`).
        let mut sig = obs.sigmas_r02.clone();
        sig.resize(m0.len(), 0.0);
        // freq 950 lies between model_freq(0,1) = 905 and (0,2) = 1005.
        let alfa = (950.0_f64 - 905.0) / (1005.0 - 905.0);
        let beta = 1.0 - alfa;
        let want = alfa * mr[1] + beta * mr[0];
        let got = interpolate_ratio_r02_masked(950.0, &m0, &mr, &sig, 5);
        assert_eq!(got.to_bits(), want.to_bits());
        // Target 1100 exceeds model_freq(0,2): last active ratio (index 1).
        let got = interpolate_ratio_r02_masked(1100.0, &m0, &mr, &sig, 5);
        assert_eq!(got.to_bits(), mr[1].to_bits());
        // The previous shifted axis `l0[l0_first + i]` clamped to `mr[0]`.
        let shifted = [1105.0, 1205.0];
        let old = interpolate_ratio_r02(1100.0, &shifted, &mr);
        assert_eq!(old.to_bits(), mr[0].to_bits());
        assert_ne!(old.to_bits(), got.to_bits());
    }

    #[test]
    fn r02_model_aligned_keeps_nan_at_zero_sigma_positions() {
        let l0 = [900.0, 1000.0, 1100.0, 1200.0, 1300.0];
        let l1 = [950.0, 1050.0, 1150.0, 1250.0, 1350.0];
        let obs = FrequencyRatiosR02Obs {
            ratios_r02: vec![f64::NAN, 0.05],
            sigmas_r02: vec![0.0, 0.1],
            r02_freqs: vec![1100.0, 1200.0],
            i2_for_r02: vec![None, Some(1)],
        };
        let l2 = [1090.0, 1195.0];
        let mr = eval_r02_model_aligned(&l0, &l1, &l2, 1, 1, &obs);
        assert!(mr[0].is_nan());
        assert_eq!(mr[1].to_bits(), ((1200.0_f64 - 1195.0) / 100.0).to_bits());
        // Compacted legacy output drops position 0, aligned output does not.
        let (compact, _) =
            eval_r02_model_with_indices(&l0, &l1, &l2, 1, 1, &obs);
        assert_eq!(compact, vec![mr[1]]);
        // Interpolation never reads the skipped entry.
        let mut sig = obs.sigmas_r02.clone();
        sig.resize(l0.len(), 0.0);
        let got = interpolate_ratio_r02_masked(1100.0, &l0, &mr, &sig, 5);
        assert_eq!(got.to_bits(), mr[1].to_bits());
    }
}
