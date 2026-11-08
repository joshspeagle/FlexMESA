// SPDX-License-Identifier: LGPL-3.0-or-later
//! Asteroseismic surface-effect frequency correction routines.
//!
//! Implements all five upstream Fortran MESA surface correction schemes:
//! 1. `"kjeldsen"`: Kjeldsen, Bedding & Christensen-Dalsgaard (2008).
//! 2. `"cubic"` / `"bg14_1"`: Ball & Gizon (2014) 1-term $a_3 \nu^3 / \mathcal{I}$.
//! 3. `"combined"` / `"bg14_2"`: Ball & Gizon (2014) 2-term.
//! 4. `"power_law"`: Ball & Gizon (2017) free power-law $a(\nu/\nu_{\max})^b / Q_{nl}$.
//! 5. `"sonoi"`: Sonoi et al. (2015) modified Lorentzian.
//!
//! Corresponds to `astero/private/astero_support.f90:702-1341` and tested in
//! `astero/test_suite/surface_effects/src/run_star_extras.f90`.

use flexmesa_autodiff::DualOrder1;
use flexmesa_math::pow;

use crate::scaling::eval_radial_delta_nu_ols;

/// Linearly interpolates radial ($l=0$) mode inertia $\mathcal{I}_0(\nu)$ at
/// arbitrary frequency `freq` ($\mu\mathrm{Hz}$), clamping to endpoint values
/// outside $[\nu_{0,1}, \nu_{0,N_0}]$.
///
/// Corresponds to `interpolate_l0_inertia` in `astero_support.f90:702-724`.
#[must_use]
pub fn interpolate_l0_inertia(
    freq: f64,
    l0_freqs: &[f64],
    l0_inertias: &[f64],
) -> f64 {
    let n = l0_freqs.len().min(l0_inertias.len());
    if n == 0 {
        return 1.0;
    }
    if n == 1 || freq <= l0_freqs[0] {
        return l0_inertias[0];
    }
    if freq >= l0_freqs[n - 1] {
        return l0_inertias[n - 1];
    }
    for i in 0..(n - 1) {
        if freq >= l0_freqs[i] && freq <= l0_freqs[i + 1] {
            let df = (l0_freqs[i + 1] - l0_freqs[i]).max(1.0e-30);
            let frac = (freq - l0_freqs[i]) / df;
            return (1.0 - frac) * l0_inertias[i] + frac * l0_inertias[i + 1];
        }
    }
    l0_inertias[n - 1]
}

/// Evaluates the Kjeldsen, Bedding & Christensen-Dalsgaard (2008) radial
/// surface-effect parameters `(a_div_r, r_scale)` and corrected frequencies.
///
/// Corresponds to `get_one_el_info` (`astero_support.f90:186-214`) and
/// `get_kjeldsen_radial_freq_corr` / `get_kjeldsen_nonradial_freq_corr`
/// (`astero_support.f90:727-777`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_kjeldsen_corr(
    l0_obs_freqs: &[f64],
    l0_model_freqs: &[f64],
    l0_inertias: &[f64],
    l0_orders: &[f64],
    delta_nu_obs: f64,
    all_model_freqs: &[f64],
    all_inertias: &[f64],
    nu_max: f64,
    b_power: f64,
    correction_factor: f64,
) -> (Vec<f64>, f64, f64) {
    let n0 = l0_obs_freqs.len().min(l0_model_freqs.len());
    if n0 == 0 {
        return (all_model_freqs.to_vec(), 0.0, 1.0);
    }
    let avg_obs = l0_obs_freqs[..n0].iter().sum::<f64>() / (n0 as f64);
    let avg_mod = l0_model_freqs[..n0].iter().sum::<f64>() / (n0 as f64);
    let dnu_mod = eval_radial_delta_nu_ols(l0_orders, l0_model_freqs);
    let denom_r = b_power * (avg_mod / avg_obs.max(1.0e-30))
        - dnu_mod / delta_nu_obs.max(1.0e-30);
    let r_scale = (b_power - 1.0) / denom_r;
    let sum_nu_b: f64 = l0_obs_freqs[..n0]
        .iter()
        .map(|&nu| pow(nu / nu_max.max(1.0e-30), b_power))
        .sum();
    let a_val = (avg_obs - r_scale * avg_mod).min(0.0) * (n0 as f64)
        / sum_nu_b.max(1.0e-30);
    let a_div_r = a_val / r_scale;

    let mut corr = Vec::with_capacity(all_model_freqs.len());
    for (idx, &f_mod) in all_model_freqs.iter().enumerate() {
        let q_nl = if idx < all_inertias.len() && !l0_model_freqs.is_empty() {
            let e0 = interpolate_l0_inertia(f_mod, l0_model_freqs, l0_inertias);
            (all_inertias[idx] / e0.max(1.0e-30)).max(1.0e-30)
        } else {
            1.0
        };
        let df = correction_factor
            * (a_div_r / q_nl)
            * pow(f_mod / nu_max.max(1.0e-30), b_power);
        corr.push(f_mod + df);
    }
    (corr, a_div_r, r_scale)
}

/// Evaluates the Ball & Gizon (2014) 1-term cubic surface correction
/// (`correction_scheme = "cubic"` / `"bg14_1"`):
///
/// $$\delta\nu_{li} = \frac{a_3\,(\nu_{li}^{\mathrm{model}})^3}{\mathcal{I}_{li}}$$
///
/// Returns `(corrected_freqs, a3)`.
///
/// Corresponds to `get_cubic_all_freq_corr` in `astero_support.f90:865-927`.
#[must_use]
pub fn eval_cubic_corr(
    obs_freqs: &[f64],
    obs_sigmas: &[f64],
    model_freqs: &[f64],
    model_inertias: &[f64],
    correction_factor: f64,
) -> (Vec<f64>, f64) {
    let n = obs_freqs
        .len()
        .min(obs_sigmas.len())
        .min(model_freqs.len())
        .min(model_inertias.len());
    let mut xtx = 0.0_f64;
    let mut xty = 0.0_f64;
    for i in 0..n {
        let f_m = model_freqs[i];
        let sig = obs_sigmas[i].max(1.0e-30);
        let ine = model_inertias[i].max(1.0e-30);
        let x_i = (f_m * f_m * f_m) / (ine * sig);
        let y_i = (obs_freqs[i] - f_m) / sig;
        xtx += x_i * x_i;
        xty += x_i * y_i;
    }
    let a3 = if xtx > 0.0 { xty / xtx } else { 0.0 };
    let mut corr = Vec::with_capacity(n);
    for i in 0..n {
        let f_m = model_freqs[i];
        let ine = model_inertias[i].max(1.0e-30);
        corr.push(f_m + correction_factor * a3 * (f_m * f_m * f_m) / ine);
    }
    (corr, a3)
}

/// Evaluates the Ball & Gizon (2014) 2-term inverse + cubic surface correction
/// (`correction_scheme = "combined"` / `"bg14_2"`):
///
/// $$\delta\nu_{li} = \frac{a_{-1}(\nu_{li}^{\mathrm{model}})^{-1}
///   + a_3(\nu_{li}^{\mathrm{model}})^3}{\mathcal{I}_{li}}$$
///
/// Returns `(corrected_freqs, a3, a_minus1)` matching the argument order
/// `(x0, x1) = (b3, b1)` of `astero_get_combined_all_freq_corr`
/// (`astero_support.f90:929-1007`).
#[must_use]
pub fn eval_combined_corr(
    obs_freqs: &[f64],
    obs_sigmas: &[f64],
    model_freqs: &[f64],
    model_inertias: &[f64],
    correction_factor: f64,
) -> (Vec<f64>, f64, f64) {
    let n = obs_freqs
        .len()
        .min(obs_sigmas.len())
        .min(model_freqs.len())
        .min(model_inertias.len());
    let mut s11 = 0.0_f64;
    let mut s12 = 0.0_f64;
    let mut s22 = 0.0_f64;
    let mut r1 = 0.0_f64;
    let mut r2 = 0.0_f64;
    for i in 0..n {
        let f_m = model_freqs[i];
        let sig = obs_sigmas[i].max(1.0e-30);
        let ine = model_inertias[i].max(1.0e-30);
        let x1 = 1.0 / (f_m * ine * sig);
        let x2 = (f_m * f_m * f_m) / (ine * sig);
        let y = (obs_freqs[i] - f_m) / sig;
        s11 += x1 * x1;
        s12 += x1 * x2;
        s22 += x2 * x2;
        r1 += x1 * y;
        r2 += x2 * y;
    }
    let det = s11 * s22 - s12 * s12;
    let (a_minus1, a3) = if det.abs() > 1.0e-60 {
        ((s22 * r1 - s12 * r2) / det, (s11 * r2 - s12 * r1) / det)
    } else {
        (0.0, 0.0)
    };
    let mut corr = Vec::with_capacity(n);
    for i in 0..n {
        let f_m = model_freqs[i];
        let ine = model_inertias[i].max(1.0e-30);
        let df = (a_minus1 / f_m + a3 * (f_m * f_m * f_m)) / ine;
        corr.push(f_m + correction_factor * df);
    }
    (corr, a3, a_minus1)
}

#[allow(clippy::too_many_arguments)]
fn fit_two_param_surface_corr<F>(
    obs_freqs: &[f64],
    obs_sigmas: &[f64],
    model_freqs: &[f64],
    model_inertias: &[f64],
    l0_freqs: &[f64],
    l0_inertias: &[f64],
    nu_max: f64,
    correction_factor: f64,
    a_init: f64,
    b_init: f64,
    eval_term: F,
) -> (Vec<f64>, f64, f64)
where
    F: Fn(DualOrder1<2>, DualOrder1<2>, f64, f64) -> DualOrder1<2>,
{
    let n = obs_freqs
        .len()
        .min(obs_sigmas.len())
        .min(model_freqs.len())
        .min(model_inertias.len());
    let mut a = a_init;
    let mut b = b_init;

    let mut q_arr = Vec::with_capacity(n);
    for i in 0..n {
        let e0 = interpolate_l0_inertia(model_freqs[i], l0_freqs, l0_inertias);
        q_arr.push(model_inertias[i] / e0.max(1.0e-30));
    }

    for _iter in 0..1000 {
        let a_ad = DualOrder1::<2>::independent(a, 0);
        let b_ad = DualOrder1::<2>::independent(b, 1);
        let mut s11 = 0.0_f64;
        let mut s12 = 0.0_f64;
        let mut s22 = 0.0_f64;
        let mut r1 = 0.0_f64;
        let mut r2 = 0.0_f64;

        for i in 0..n {
            let f_ad = eval_term(a_ad, b_ad, model_freqs[i], nu_max);
            let sig = obs_sigmas[i].max(1.0e-30);
            let x1 = -f_ad.d1[0] / sig;
            let x2 = -f_ad.d1[1] / sig;
            let y =
                ((obs_freqs[i] - model_freqs[i]) * q_arr[i] - f_ad.val) / sig;
            s11 += x1 * x1;
            s12 += x1 * x2;
            s22 += x2 * x2;
            r1 += x1 * y;
            r2 += x2 * y;
        }
        let det = s11 * s22 - s12 * s12;
        if !det.is_finite() || det.abs() <= 1.0e-60 {
            break;
        }
        let da = (s22 * r1 - s12 * r2) / det;
        let db = (s11 * r2 - s12 * r1) / det;
        if !da.is_finite() || !db.is_finite() {
            break;
        }
        a -= da;
        b -= db;
        if da.abs() < 1.0e-8 && db.abs() < 1.0e-6 {
            break;
        }
    }
    if b < 0.0 {
        a = 0.0;
    }

    let a_c = DualOrder1::<2>::constant(a);
    let b_c = DualOrder1::<2>::constant(b);
    let mut corr = Vec::with_capacity(n);
    for i in 0..n {
        let df = eval_term(a_c, b_c, model_freqs[i], nu_max).val
            / q_arr[i].max(1.0e-30);
        corr.push(model_freqs[i] + correction_factor * df);
    }
    (corr, a, b)
}

/// Evaluates the Ball & Gizon (2017) free power-law surface correction
/// (`correction_scheme = "power_law"`) using forward-mode `DualOrder1<2>`
/// Gauss-Newton iteration:
///
/// $$\delta\nu_{\mathrm{PL}}(\nu; a, b) = \frac{a}{Q_{nl}} \left(\frac{\nu}{\nu_{\max}}\right)^b$$
///
/// Returns `(corrected_freqs, a, b)`.
///
/// Corresponds to `get_power_law_all_freq_corr` in `astero_support.f90:1010-1138`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_power_law_corr(
    obs_freqs: &[f64],
    obs_sigmas: &[f64],
    model_freqs: &[f64],
    model_inertias: &[f64],
    l0_freqs: &[f64],
    l0_inertias: &[f64],
    nu_max: f64,
    correction_factor: f64,
) -> (Vec<f64>, f64, f64) {
    fit_two_param_surface_corr(
        obs_freqs,
        obs_sigmas,
        model_freqs,
        model_inertias,
        l0_freqs,
        l0_inertias,
        nu_max,
        correction_factor,
        -5.25,
        5.37,
        |a_ad, b_ad, freq, numax| {
            let ratio = DualOrder1::<2>::constant(freq / numax);
            a_ad * ratio.pow(b_ad)
        },
    )
}

/// Evaluates the Sonoi et al. (2015) modified Lorentzian surface correction
/// (`correction_scheme = "sonoi"`) using forward-mode `DualOrder1<2>`
/// Gauss-Newton iteration:
///
/// $$\delta\nu_{\mathrm{Sonoi}}(\nu; a, b) = \frac{a\,\nu_{\max}}{Q_{nl}}
///   \left[1 - \frac{1}{1 + (\nu / \nu_{\max})^b}\right]$$
///
/// Returns `(corrected_freqs, a, b)`.
///
/// Corresponds to `get_sonoi_all_freq_corr` in `astero_support.f90:1140-1269`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_sonoi_corr(
    obs_freqs: &[f64],
    obs_sigmas: &[f64],
    model_freqs: &[f64],
    model_inertias: &[f64],
    l0_freqs: &[f64],
    l0_inertias: &[f64],
    nu_max: f64,
    correction_factor: f64,
) -> (Vec<f64>, f64, f64) {
    fit_two_param_surface_corr(
        obs_freqs,
        obs_sigmas,
        model_freqs,
        model_inertias,
        l0_freqs,
        l0_inertias,
        nu_max,
        correction_factor,
        -3.59e-3,
        11.26,
        |a_ad, b_ad, freq, numax| {
            let ratio = DualOrder1::<2>::constant(freq / numax);
            let one = DualOrder1::<2>::constant(1.0);
            a_ad * numax * (one - one / (one + ratio.pow(b_ad)))
        },
    )
}
