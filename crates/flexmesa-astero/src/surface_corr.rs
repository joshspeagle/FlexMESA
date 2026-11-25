// SPDX-License-Identifier: GPL-3.0-or-later
//! Asteroseismic surface-effect frequency correction routines.
//!
//! Implements all five upstream Fortran MESA surface correction schemes:
//! 1. `"kjeldsen"`: Kjeldsen, Bedding & Christensen-Dalsgaard (2008).
//! 2. `"cubic"` / `"bg14_1"`: Ball & Gizon (2014) 1-term $a_3 \nu^3 / \mathcal{I}$.
//! 3. `"combined"` / `"bg14_2"`: Ball & Gizon (2014) 2-term.
//! 4. `"power_law"`: Ball & Gizon (2017) free power-law $a(\nu/\nu_{\max})^b / Q_{nl}$.
//! 5. `"sonoi"`: Sonoi et al. (2015) modified Lorentzian.
//!
//! Corresponds to `astero/private/astero_support.f90:702-1341`.

use flexmesa_autodiff::DualOrder1;
use flexmesa_math::pow;

use crate::scaling::{DELTA_NU_SUN, NU_MAX_SUN};

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

/// Estimates radial orders `l0_n_obs` using upstream MESA `init_obs_data`
/// (`astero_support.f90:655-664`), falling back to `floor(nu_0 / delta_nu)` when
/// `nu_max <= 0`.
#[must_use]
pub fn estimate_l0_n_obs_with_numax(
    l0_obs_freqs: &[f64],
    delta_nu: f64,
    nu_max: f64,
    delta_nu_sun: f64,
    nu_max_sun: f64,
) -> Vec<f64> {
    if l0_obs_freqs.is_empty() || delta_nu <= 0.0 {
        return Vec::new();
    }
    if nu_max > 0.0 && delta_nu_sun > 0.0 && nu_max_sun > 0.0 {
        let range = l0_obs_freqs[l0_obs_freqs.len() - 1] - l0_obs_freqs[0];
        let norders =
            ((range / delta_nu + 0.5).trunc() as isize + 1).max(1) as usize;
        let nmax =
            (nu_max / delta_nu) * (delta_nu_sun / nu_max_sun) * 22.6 - 1.6;
        let n1 = (nmax - (((norders - 1) / 2) as f64)).trunc() as isize;
        return l0_obs_freqs
            .iter()
            .take(norders.min(l0_obs_freqs.len()))
            .enumerate()
            .map(|(i, &f)| {
                if i == 0 {
                    n1 as f64
                } else {
                    let offset = ((f - l0_obs_freqs[0]) / delta_nu + 0.5)
                        .trunc() as isize;
                    (n1 + offset) as f64
                }
            })
            .collect();
    }
    let n_first = (l0_obs_freqs[0] / delta_nu).trunc();
    (0..l0_obs_freqs.len())
        .map(|j| n_first + (j as f64))
        .collect()
}

/// Estimates radial orders `l0_n_obs(j)` when omitted (`astero_support.f90:655-664`).
#[must_use]
pub fn estimate_l0_n_obs(l0_obs_freqs: &[f64], delta_nu: f64) -> Vec<f64> {
    estimate_l0_n_obs_with_numax(
        l0_obs_freqs,
        delta_nu,
        -1.0,
        DELTA_NU_SUN,
        NU_MAX_SUN,
    )
}

/// Maps internal fitted surface-correction coefficients to upstream exported
/// `(surf_coef1, surf_coef2)` (`astero/private/astero_support.f90:1280-1334`).
#[must_use]
pub fn eval_surf_coefs(
    scheme: &str,
    primary: f64,
    secondary: f64,
    model_nu_max: f64,
    nu_max_sun: f64,
) -> (f64, f64) {
    match scheme.trim().to_ascii_lowercase().as_str() {
        "kjeldsen" | "power_law" | "sonoi" => (primary, secondary),
        "cubic" => {
            let ratio = if nu_max_sun > 0.0 && model_nu_max > 0.0 {
                5000.0 * model_nu_max / nu_max_sun
            } else {
                5000.0
            };
            (primary * ratio * ratio * ratio, 0.0)
        }
        "combined" => {
            let ratio = if nu_max_sun > 0.0 && model_nu_max > 0.0 {
                5000.0 * model_nu_max / nu_max_sun
            } else {
                5000.0
            };
            (primary * ratio * ratio * ratio, secondary / ratio)
        }
        _ => (0.0, 0.0),
    }
}

/// Evaluates the Kjeldsen, Bedding & Christensen-Dalsgaard (2008) radial
/// surface-effect parameters `(a_div_r, r_scale)` and corrected frequencies
/// (`astero_support.f90:182-215, 727-777`).
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
    if n0 == 0
        || correction_factor <= 0.0
        || delta_nu_obs <= 0.0
        || nu_max <= 0.0
    {
        return (all_model_freqs.to_vec(), 0.0, 1.0);
    }
    let est_orders;
    let orders_slice = if l0_orders.len() >= n0 && l0_orders[0] > 0.0 {
        &l0_orders[..n0]
    } else {
        est_orders = estimate_l0_n_obs_with_numax(
            &l0_obs_freqs[..n0],
            delta_nu_obs,
            nu_max,
            DELTA_NU_SUN,
            NU_MAX_SUN,
        );
        &est_orders
    };
    let (mut cnt, mut sum_obs, mut sum_mod, mut sum_n) =
        (0usize, 0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..n0 {
        if l0_obs_freqs[i] < 0.0 {
            continue;
        }
        cnt += 1;
        sum_obs += l0_obs_freqs[i];
        sum_mod += l0_model_freqs[i];
        sum_n += orders_slice.get(i).copied().unwrap_or(i as f64);
    }
    if cnt == 0 {
        return (all_model_freqs.to_vec(), 0.0, 1.0);
    }
    let avg_obs = sum_obs / (cnt as f64);
    let avg_mod = sum_mod / (cnt as f64);
    let avg_n = sum_n / (cnt as f64);
    let (mut sum_1, mut sum_2, mut sum_nu_b) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..n0 {
        if l0_obs_freqs[i] < 0.0 {
            continue;
        }
        let n_i = orders_slice.get(i).copied().unwrap_or(i as f64);
        let dn = n_i - avg_n;
        sum_1 += (l0_model_freqs[i] - avg_mod) * dn;
        sum_2 += dn * dn;
        sum_nu_b += pow(l0_obs_freqs[i] / nu_max, b_power);
    }
    if sum_2 == 0.0 || sum_nu_b == 0.0 {
        return (all_model_freqs.to_vec(), 0.0, 1.0);
    }
    let dnu_mod = sum_1 / sum_2;
    let denom_r = b_power * (avg_mod / avg_obs) - dnu_mod / delta_nu_obs;
    let r_scale = (b_power - 1.0) / denom_r;
    if r_scale <= 0.0 || !r_scale.is_finite() {
        return (all_model_freqs.to_vec(), 0.0, 1.0);
    }
    let a_val = (avg_obs - r_scale * avg_mod).min(0.0) * (n0 as f64) / sum_nu_b;
    let a_div_r = a_val / r_scale;

    let corr = all_model_freqs
        .iter()
        .enumerate()
        .map(|(idx, &f_mod)| {
            if b_power <= 0.0 || f_mod <= 0.0 {
                return f_mod;
            }
            let is_l0 =
                idx < n0 && (f_mod - l0_model_freqs[idx]).abs() <= 1e-12;
            if is_l0 && idx < l0_obs_freqs.len() && l0_obs_freqs[idx] < 0.0 {
                return f_mod;
            }
            let q_nl = if is_l0 {
                1.0
            } else if idx < all_inertias.len() && !l0_model_freqs.is_empty() {
                let e0 =
                    interpolate_l0_inertia(f_mod, l0_model_freqs, l0_inertias);
                (all_inertias[idx] / e0.max(1.0e-30)).max(1.0e-30)
            } else {
                1.0
            };
            f_mod
                + correction_factor
                    * (a_div_r / q_nl)
                    * pow(f_mod / nu_max, b_power)
        })
        .collect();
    (corr, a_div_r, r_scale)
}

/// Evaluates the Ball & Gizon (2014) 1-term cubic surface correction (`astero_support.f90:865-927`).
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
    let (mut xtx, mut xty) = (0.0_f64, 0.0_f64);
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
    let corr = (0..n)
        .map(|i| {
            let f_m = model_freqs[i];
            f_m + correction_factor * a3 * (f_m * f_m * f_m)
                / model_inertias[i].max(1.0e-30)
        })
        .collect();
    (corr, a3)
}

/// Evaluates the Ball & Gizon (2014) 2-term inverse + cubic surface correction (`astero_support.f90:929-1007`).
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
    let (mut s11, mut s12, mut s22, mut r1, mut r2) = (0.0, 0.0, 0.0, 0.0, 0.0);
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
    let corr = (0..n)
        .map(|i| {
            let f_m = model_freqs[i];
            let df = (a_minus1 / f_m + a3 * (f_m * f_m * f_m))
                / model_inertias[i].max(1.0e-30);
            f_m + correction_factor * df
        })
        .collect();
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
    zero_a_when_b_neg: bool,
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
    let (mut a, mut b) = (a_init, b_init);
    let q_arr: Vec<f64> = (0..n)
        .map(|i| {
            let e0 =
                interpolate_l0_inertia(model_freqs[i], l0_freqs, l0_inertias);
            model_inertias[i] / e0.max(1.0e-30)
        })
        .collect();

    for _iter in 0..1000 {
        let a_ad = DualOrder1::<2>::independent(a, 0);
        let b_ad = DualOrder1::<2>::independent(b, 1);
        let (mut s11, mut s12, mut s22, mut r1, mut r2) =
            (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            let f_ad = eval_term(a_ad, b_ad, model_freqs[i], nu_max);
            let sig = obs_sigmas[i].max(1.0e-30);
            let (x1, x2) = (-f_ad.d1[0] / sig, -f_ad.d1[1] / sig);
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
    if zero_a_when_b_neg && b < 0.0 {
        a = 0.0;
    }
    let corr = apply_two_param_surface_corr(
        model_freqs,
        model_inertias,
        l0_freqs,
        l0_inertias,
        nu_max,
        correction_factor,
        a,
        b,
        eval_term,
    );
    (corr, a, b)
}

#[inline]
fn power_law_term(
    a_ad: DualOrder1<2>,
    b_ad: DualOrder1<2>,
    freq: f64,
    numax: f64,
) -> DualOrder1<2> {
    let ratio = DualOrder1::<2>::constant(freq / numax);
    a_ad * ratio.pow(b_ad)
}

#[inline]
fn sonoi_term(
    a_ad: DualOrder1<2>,
    b_ad: DualOrder1<2>,
    freq: f64,
    numax: f64,
) -> DualOrder1<2> {
    let ratio = DualOrder1::<2>::constant(freq / numax);
    let one = DualOrder1::<2>::constant(1.0);
    a_ad * numax * (one - one / (one + ratio.pow(b_ad)))
}

#[allow(clippy::too_many_arguments)]
fn apply_two_param_surface_corr<F>(
    model_freqs: &[f64],
    model_inertias: &[f64],
    l0_freqs: &[f64],
    l0_inertias: &[f64],
    nu_max: f64,
    correction_factor: f64,
    a: f64,
    b: f64,
    eval_term: F,
) -> Vec<f64>
where
    F: Fn(DualOrder1<2>, DualOrder1<2>, f64, f64) -> DualOrder1<2>,
{
    let n = model_freqs.len().min(model_inertias.len());
    let a_c = DualOrder1::<2>::constant(a);
    let b_c = DualOrder1::<2>::constant(b);
    let mut corr = Vec::with_capacity(n);
    for i in 0..n {
        let e0 = interpolate_l0_inertia(model_freqs[i], l0_freqs, l0_inertias);
        let q = (model_inertias[i] / e0.max(1.0e-30)).max(1.0e-30);
        let df = eval_term(a_c, b_c, model_freqs[i], nu_max).val / q;
        corr.push(model_freqs[i] + correction_factor * df);
    }
    corr
}

/// Evaluates the Ball & Gizon (2017) free power-law surface correction (`astero_support.f90:1010-1138`).
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
        false,
        power_law_term,
    )
}

/// Evaluates the radial-only ($l=0$) Ball & Gizon (2017) power-law surface correction (`astero_support.f90:1010-1089`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_power_law_corr_radial_only(
    l0_obs_freqs: &[f64],
    l0_obs_sigmas: &[f64],
    l0_freqs: &[f64],
    l0_inertias: &[f64],
    model_freqs: &[f64],
    model_inertias: &[f64],
    nu_max: f64,
    correction_factor: f64,
) -> (Vec<f64>, f64, f64) {
    let (_, a, b) = eval_power_law_corr(
        l0_obs_freqs,
        l0_obs_sigmas,
        l0_freqs,
        l0_inertias,
        l0_freqs,
        l0_inertias,
        nu_max,
        correction_factor,
    );
    let corr = apply_two_param_surface_corr(
        model_freqs,
        model_inertias,
        l0_freqs,
        l0_inertias,
        nu_max,
        correction_factor,
        a,
        b,
        power_law_term,
    );
    (corr, a, b)
}

/// Evaluates the Sonoi et al. (2015) modified Lorentzian surface correction (`astero_support.f90:1140-1269`).
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
        true,
        sonoi_term,
    )
}

/// Evaluates the radial-only ($l=0$) Sonoi et al. (2015) modified Lorentzian surface correction (`astero_support.f90:1140-1219`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_sonoi_corr_radial_only(
    l0_obs_freqs: &[f64],
    l0_obs_sigmas: &[f64],
    l0_freqs: &[f64],
    l0_inertias: &[f64],
    model_freqs: &[f64],
    model_inertias: &[f64],
    nu_max: f64,
    correction_factor: f64,
) -> (Vec<f64>, f64, f64) {
    let (_, a, b) = eval_sonoi_corr(
        l0_obs_freqs,
        l0_obs_sigmas,
        l0_freqs,
        l0_inertias,
        l0_freqs,
        l0_inertias,
        nu_max,
        correction_factor,
    );
    let corr = apply_two_param_surface_corr(
        model_freqs,
        model_inertias,
        l0_freqs,
        l0_inertias,
        nu_max,
        correction_factor,
        a,
        b,
        sonoi_term,
    );
    (corr, a, b)
}
