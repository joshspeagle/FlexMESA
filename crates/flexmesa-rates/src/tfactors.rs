// SPDX-License-Identifier: GPL-3.0-or-later
//! Precomputed temperature power factors (`T_Factors`), REACLIB 7-parameter
//! polynomial evaluator, and detailed-balance reverse rate coefficients
//! (`rates/private/ratelib.f90`, `rates/private/reaclib_eval.f90`, and
//! `rates/private/reaclib_support.f90:217-257`).

use flexmesa_const::{
    AVO, BOLTZM, FIVE_THIRDS, FOUR_THIRDS, HBAR, LN10, ONE_THIRD, PI,
    TWO_THIRDS,
};

use crate::types::{ReaclibTerm, MAX_LN_LAMBDA};

pub use flexmesa_chem::{get_partition_fcn_indx, T_PART as TPART};

/// Precomputed fractional powers of $T_9 = T / 10^9\,\mathrm{K}$ matching
/// `T_Factors` in `rates/public/rates_def.f90` and `tfactors` in `ratelib.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TFactors {
    /// Clamped $\log_{10}(T/\mathrm{K}) = \max(\log_{10} T, 0)$.
    pub log_t: f64,
    /// Clamped temperature $T = \max(T, 1\,\mathrm{K})$.
    pub temp: f64,
    /// Natural log of temperature $\ln(T/\mathrm{K})$.
    pub ln_t: f64,
    /// Temperature in $10^9\,\mathrm{K}$, $T_9 = 10^{-9} T$.
    pub t9: f64,
    /// Natural log of $T_9$, $\ln T_9 = \ln T - 9 \ln 10$.
    pub ln_t9: f64,
    /// $T_9^2$.
    pub t92: f64,
    /// $T_9^3$.
    pub t93: f64,
    /// $T_9^4$.
    pub t94: f64,
    /// $T_9^5$.
    pub t95: f64,
    /// $T_9^6$.
    pub t96: f64,
    /// $T_9^7$.
    pub t97: f64,
    /// $T_9^{-1}$.
    pub t9i: f64,
    /// $T_9^{-2}$.
    pub t9i2: f64,
    /// $T_9^{-3}$.
    pub t9i3: f64,
    /// $T_9^{1/2}$.
    pub t912: f64,
    /// $T_9^{3/2}$.
    pub t932: f64,
    /// $T_9^{5/2}$.
    pub t952: f64,
    /// $T_9^{7/2}$.
    pub t972: f64,
    /// $T_9^{-1/2}$.
    pub t9i12: f64,
    /// $T_9^{-3/2}$.
    pub t9i32: f64,
    /// $T_9^{-5/2}$.
    pub t9i52: f64,
    /// $T_9^{-7/2}$.
    pub t9i72: f64,
    /// $T_9^{1/3}$.
    pub t913: f64,
    /// $T_9^{2/3}$.
    pub t923: f64,
    /// $T_9^{4/3}$.
    pub t943: f64,
    /// $T_9^{5/3}$.
    pub t953: f64,
    /// $T_9^{7/3}$.
    pub t973: f64,
    /// $T_9^{11/3}$.
    pub t9113: f64,
    /// $T_9^{-1/3}$.
    pub t9i13: f64,
    /// $T_9^{-2/3}$.
    pub t9i23: f64,
    /// $T_9^{-4/3}$.
    pub t9i43: f64,
    /// $T_9^{-5/3}$.
    pub t9i53: f64,
    /// $T_9^{1/4}$.
    pub t914: f64,
    /// $T_9^{3/4}$.
    pub t934: f64,
    /// $T_9^{5/4}$.
    pub t954: f64,
    /// $T_9^{7/4}$.
    pub t974: f64,
    /// $T_9^{-1/4}$.
    pub t9i14: f64,
    /// $T_9^{-3/4}$.
    pub t9i34: f64,
    /// $T_9^{-5/4}$.
    pub t9i54: f64,
    /// $T_9^{-7/4}$.
    pub t9i74: f64,
    /// $T_9^{1/5}$.
    pub t915: f64,
    /// $T_9^{-1/5}$.
    pub t9i15: f64,
    /// $T_9^{3/5}$.
    pub t935: f64,
    /// $T_9^{4/5}$.
    pub t945: f64,
    /// $T_9^{-4/5}$.
    pub t9i45: f64,
    /// $T_9^{6/5}$.
    pub t965: f64,
    /// $T_9^{-6/5}$.
    pub t9i65: f64,
    /// $T_9^{1/6}$.
    pub t916: f64,
    /// $T_9^{-1/6}$.
    pub t9i16: f64,
    /// $T_9^{7/6}$.
    pub t976: f64,
    /// $T_9^{-7/6}$.
    pub t9i76: f64,
    /// $T_9^{1/7}$.
    pub t917: f64,
    /// $T_9^{-1/7}$.
    pub t9i17: f64,
    /// $T_9^{2/7}$.
    pub t927: f64,
    /// $T_9^{4/7}$.
    pub t947: f64,
    /// $T_9^{-4/7}$.
    pub t9i47: f64,
    /// $T_9^{1/8}$.
    pub t918: f64,
    /// $T_9^{-1/8}$.
    pub t9i18: f64,
    /// $T_9^{3/8}$.
    pub t938: f64,
    /// $T_9^{-3/8}$.
    pub t9i38: f64,
    /// $T_9^{5/8}$.
    pub t958: f64,
    /// $T_9^{-5/8}$.
    pub t9i58: f64,
}

impl TFactors {
    /// Initializes [`TFactors`] from `(log_t_in, temp_in)` using exact Fortran
    /// operation order (`ratelib.f90:tfactors`).
    #[must_use]
    pub fn from_log_t(log_t_in: f64, temp_in: f64) -> Self {
        // `rates/private/ratelib.f90:3521-3606` (`tfactors`)
        let log_t = log_t_in.max(0.0);
        let temp = temp_in.max(1.0);
        let ln_t = log_t * LN10;
        let ln_t9 = (log_t - 9.0) * LN10;
        let t9 = temp * 1.0e-9;

        let t92 = t9 * t9;
        let t93 = t92 * t9;
        let t94 = t93 * t9;
        let t95 = t94 * t9;
        let t96 = t95 * t9;
        let t97 = t96 * t9;

        let t912 = t9.sqrt();
        let t932 = t9 * t912;
        let t952 = t9 * t932;
        let t972 = t9 * t952;

        let t913 = flexmesa_math::pow(t9, ONE_THIRD);
        let t923 = t913 * t913;
        let t943 = t9 * t913;
        let t953 = t9 * t923;
        let t973 = t953 * t923;
        let t9113 = t973 * t943;

        let t914 = flexmesa_math::pow(t9, 0.25);
        let t934 = t914 * t914 * t914;
        let t954 = t9 * t914;
        let t974 = t9 * t934;

        let t915 = flexmesa_math::pow(t9, 0.2);
        let t935 = t915 * t915 * t915;
        let t945 = t915 * t935;
        let t965 = t9 * t915;

        let t916 = flexmesa_math::pow(t9, 1.0 / 6.0);
        let t976 = t9 * t916;

        let t917 = flexmesa_math::pow(t9, 1.0 / 7.0);
        let t927 = t917 * t917;
        let t947 = t927 * t927;

        let t918 = t914.sqrt();
        let t938 = t918 * t918 * t918;
        let t958 = t938 * t918 * t918;

        let t9i = 1.0 / t9;
        let t9i2 = t9i * t9i;
        let t9i3 = t9i2 * t9i;

        let t9i12 = 1.0 / t912;
        let t9i32 = t9i * t9i12;
        let t9i52 = t9i * t9i32;
        let t9i72 = t9i * t9i52;

        let t9i13 = 1.0 / t913;
        let t9i23 = t9i13 * t9i13;
        let t9i43 = t9i * t9i13;
        let t9i53 = t9i * t9i23;

        let t9i14 = 1.0 / t914;
        let t9i34 = t9i14 * t9i14 * t9i14;
        let t9i54 = t9i * t9i14;
        let t9i74 = t9i * t9i34;

        let t9i15 = 1.0 / t915;
        let t9i35 = t9i15 * t9i15 * t9i15;
        let t9i45 = t9i15 * t9i35;
        let t9i65 = t9i * t9i15;

        let t9i16 = 1.0 / t916;
        let t9i76 = 1.0 / t976;

        let t9i17 = 1.0 / t917;
        let t9i27 = t9i17 * t9i17;
        let t9i47 = t9i27 * t9i27;

        let t9i18 = 1.0 / t918;
        let t9i38 = t9i18 * t9i18 * t9i18;
        let t9i58 = t9i38 * t9i18 * t9i18;

        Self {
            log_t,
            temp,
            ln_t,
            t9,
            ln_t9,
            t92,
            t93,
            t94,
            t95,
            t96,
            t97,
            t9i,
            t9i2,
            t9i3,
            t912,
            t932,
            t952,
            t972,
            t9i12,
            t9i32,
            t9i52,
            t9i72,
            t913,
            t923,
            t943,
            t953,
            t973,
            t9113,
            t9i13,
            t9i23,
            t9i43,
            t9i53,
            t914,
            t934,
            t954,
            t974,
            t9i14,
            t9i34,
            t9i54,
            t9i74,
            t915,
            t9i15,
            t935,
            t945,
            t9i45,
            t965,
            t9i65,
            t916,
            t9i16,
            t976,
            t9i76,
            t917,
            t9i17,
            t927,
            t947,
            t9i47,
            t918,
            t9i18,
            t938,
            t9i38,
            t958,
            t9i58,
        }
    }

    /// Initializes [`TFactors`] directly from temperature $T$ ($\mathrm{K}$).
    #[must_use]
    pub fn from_temp(temp_in: f64) -> Self {
        let temp = temp_in.max(1.0);
        Self::from_log_t(flexmesa_math::log10(temp), temp)
    }
}

/// Evaluates a slice of [`ReaclibTerm`]s at temperature
/// $T_9 = T / 10^9\,\mathrm{K}$, returning
/// `(lambda, dlambda_dlnT, rlambda, drlambda_dlnT)`.
#[must_use]
pub fn eval_reaclib_terms(
    terms: &[ReaclibTerm],
    t9: f64,
    reaclib_min_t9: f64,
    check_e_flag: bool,
) -> (f64, f64, f64, f64) {
    // `rates/private/reaclib_eval.f90:86-222`
    const LAM_MAX: f64 = 1.0e99;

    if terms.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let rflag0 = terms[0].rflag;
    if t9 < reaclib_min_t9
        && rflag0 != b'w'
        && !(check_e_flag && rflag0 == b'e')
    {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let t9inv = 1.0 / t9;
    let ln_t9 = flexmesa_math::log(t9);
    let f = [
        1.0,
        t9inv,
        flexmesa_math::pow(t9inv, ONE_THIRD),
        flexmesa_math::pow(t9, ONE_THIRD),
        t9,
        flexmesa_math::pow(t9, FIVE_THIRDS),
        ln_t9,
    ];
    let dt9fac_dt9 = [
        0.0,
        -t9inv * t9inv,
        -ONE_THIRD * flexmesa_math::pow(t9inv, FOUR_THIRDS),
        ONE_THIRD * flexmesa_math::pow(t9inv, TWO_THIRDS),
        1.0,
        FIVE_THIRDS * flexmesa_math::pow(t9, TWO_THIRDS),
        t9inv,
    ];
    let df_dln_t = [
        t9 * dt9fac_dt9[0],
        t9 * dt9fac_dt9[1],
        t9 * dt9fac_dt9[2],
        t9 * dt9fac_dt9[3],
        t9 * dt9fac_dt9[4],
        t9 * dt9fac_dt9[5],
        t9 * dt9fac_dt9[6],
    ];

    let mut indx = get_partition_fcn_indx(t9);
    if indx >= 24 {
        indx = 23;
    }

    let mut lam_sum = 0.0;
    let mut dlam_sum = 0.0;
    let mut rlam_sum = 0.0;
    let mut drlam_sum = 0.0;

    for term in terms {
        let mut ln1 = 0.0;
        for k in 0..7 {
            ln1 += f[k] * term.coeffs[k];
        }
        let (ln_l, lam, dlam) = if ln1 > MAX_LN_LAMBDA {
            (MAX_LN_LAMBDA, LAM_MAX, 0.0)
        } else {
            let lam_v = flexmesa_math::exp(ln1);
            let mut dln_l = 0.0;
            for k in 0..7 {
                dln_l += df_dln_t[k] * term.coeffs[k];
            }
            (ln1, lam_v, dln_l * lam_v)
        };
        lam_sum += lam;
        dlam_sum += dlam;

        if term.rflag == b'w' || term.rflag == b'e' || term.inv_part[0] == 0.0 {
            continue;
        }

        let (qratio, dqratio_dln_t) = if indx < 1 {
            (1.0, 0.0)
        } else {
            let m = indx - 1;
            let dt = TPART[m + 1] - TPART[m];
            let tfac = (t9 - TPART[m]) / dt;
            let dtfac_dln_t = t9 / dt;
            let a = term.inv_part[m + 1] / term.inv_part[m];
            let blurp = flexmesa_math::pow(a, tfac);
            let qr = term.inv_part[m] * blurp;
            let dqr = qr * flexmesa_math::log(a) * dtfac_dln_t;
            (qr, dqr)
        };

        let inv_exp = f64::from(term.inv_exp);
        let dt9i_dln_t = -t9inv;
        let ln1_inv =
            ln_l + term.inv_c1 + term.inv_c2 * t9inv + 1.5 * inv_exp * ln_t9;
        let (fac1, dfac1_dln_t) = if ln1_inv < MAX_LN_LAMBDA {
            let f1 = flexmesa_math::exp(ln1_inv);
            let dln1 = dlam / lam.max(1.0e-99)
                + term.inv_c2 * dt9i_dln_t
                + 1.5 * inv_exp;
            (f1, dln1 * f1)
        } else {
            (LAM_MAX, 0.0)
        };

        let inv_lam = fac1 * qratio;
        let dinv_lam = if lam < 1.0e-99 {
            0.0
        } else {
            dfac1_dln_t * qratio + fac1 * dqratio_dln_t
        };
        rlam_sum += inv_lam;
        drlam_sum += dinv_lam;
    }

    (lam_sum, dlam_sum, rlam_sum, drlam_sum)
}

/// Evaluates the detailed-balance reverse-to-forward multiplier
/// $\lambda_{\mathrm{rev}} / \lambda_{\mathrm{fwd}}$ for a single REACLIB term
/// at temperature $T_9 = T / 10^9\,\mathrm{K}$ (`compute_some_inverse_lambdas`
/// with `lambda = 1, ln_lambda = 0` in `rates/private/reaclib_eval.f90:144-222`,
/// called by `eval_table_reverse` in `rates/private/raw_rates.f90:880-920`).
#[must_use]
pub fn eval_reaclib_inverse_ratio(term: &ReaclibTerm, t9: f64) -> f64 {
    const LAM_MAX: f64 = 1.0e99;

    let mut indx = get_partition_fcn_indx(t9);
    if indx >= 24 {
        indx = 23;
    }
    let qratio = if indx < 1 {
        1.0
    } else {
        let m = indx - 1;
        let p0 = term.inv_part[m];
        let p1 = term.inv_part[m + 1];
        if p0 > 0.0 && p1 > 0.0 {
            let dt = TPART[m + 1] - TPART[m];
            let tfac = (t9 - TPART[m]) / dt;
            let a = p1 / p0;
            p0 * flexmesa_math::pow(a, tfac)
        } else {
            1.0
        }
    };

    let ln_t9 = flexmesa_math::log(t9);
    let t9i = 1.0 / t9;
    let ln1 = term.inv_c1
        + term.inv_c2 * t9i
        + 1.5 * f64::from(term.inv_exp) * ln_t9;
    let fac1 = if ln1 < MAX_LN_LAMBDA {
        flexmesa_math::exp(ln1)
    } else {
        LAM_MAX
    };
    fac1 * qratio
}

/// Computes the detailed-balance first reverse coefficient
/// `inverse_coefficients(1)` from nuclear masses `w` (`AMU`) and ground-state
/// spins `spin` (`rates/private/reaclib_support.f90:217-257` and
/// `rates/test/src/test_rates.f90:305-378`).
#[must_use]
pub fn compute_rev_ratio_coeff1(
    w_in: &[f64],
    spin_in: &[f64],
    w_out: &[f64],
    spin_out: &[f64],
) -> f64 {
    let fac = flexmesa_math::pow(
        1.0e9 * BOLTZM / (2.0 * PI * HBAR * HBAR * AVO),
        1.5,
    ) / AVO;
    let g_in: f64 = spin_in.iter().map(|&s| 2.0 * s + 1.0).product();
    let g_out: f64 = spin_out.iter().map(|&s| 2.0 * s + 1.0).product();
    let m_in: f64 = w_in.iter().copied().product();
    let m_out: f64 = w_out.iter().copied().product();
    let tmp = m_in / m_out;
    let mut log_c1 =
        flexmesa_math::log(flexmesa_math::pow(tmp, 1.5) * (g_in / g_out));
    let inv_exp = (w_in.len() as i32) - (w_out.len() as i32);
    if inv_exp != 0 {
        log_c1 += f64::from(inv_exp) * flexmesa_math::log(fac);
    }
    log_c1
}
