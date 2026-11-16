// SPDX-License-Identifier: LGPL-3.0-or-later
//! Analytical physics oracles and Brott et al. (2011) massive-star wind law
//! for the 8 upstream MESA `binary/test_suite` verification cases.
//!
//! Corresponds to `binary/test_suite/jdot_gr_check/src/run_binary_extras.f90:99-120`,
//! `binary/test_suite/jdot_ml_check/src/run_binary_extras.f90:99-132`, and
//! `binary/test_suite/double_bh/src/run_star_extras.f90:55-206`.

use flexmesa_const::{CLIGHT, MSUN, RSUN, STANDARD_CGRAV};
use flexmesa_math::{exp10, log10, pow};

pub use crate::inlist::BinaryInlistControls;

/// Canonical 8 upstream MESA `binary/test_suite` cases.
pub const TIER5_BINARY_CASES: [&str; 8] = [
    "double_bh",
    "evolve_both_stars",
    "jdot_gr_check",
    "jdot_ls_check",
    "jdot_ml_check",
    "star_plus_point_mass",
    "star_plus_point_mass_explicit_mdot",
    "wind_fed_bhhmxb",
];

/// Returns `true` if `case_name` is one of [`TIER5_BINARY_CASES`].
#[must_use]
pub fn is_tier5_binary_case(case_name: &str) -> bool {
    TIER5_BINARY_CASES.contains(&case_name)
}

/// Evaluates the exact Peters (1964, PhRv 136, B1224) gravitational-wave
/// inspiral orbital separation $a(t)$ in $R_\odot$ for circular point masses:
///
/// $$a(t) = \left(a_0^4 - 4\beta t\right)^{1/4}, \qquad
///   \beta = \frac{64 G^3}{5 c^5} M_1 M_2 (M_1 + M_2)$$
///
/// Corresponds to `binary/test_suite/jdot_gr_check/src/run_binary_extras.f90:109-119`.
#[must_use]
pub fn eval_gr_expected_separation_rsun(
    a0_rsun: f64,
    m1_msun: f64,
    m2_msun: f64,
    age_sec: f64,
) -> f64 {
    let a0_cm = a0_rsun * RSUN;
    let m1 = m1_msun * MSUN;
    let m2 = m2_msun * MSUN;
    let g3 = STANDARD_CGRAV * STANDARD_CGRAV * STANDARD_CGRAV;
    let c5 = CLIGHT * CLIGHT * CLIGHT * CLIGHT * CLIGHT;
    let beta = (64.0 * g3 / (5.0 * c5)) * m1 * m2 * (m1 + m2);
    let a4 = (a0_cm.powi(4) - 4.0 * beta * age_sec).max(0.0);
    pow(a4, 0.25) / RSUN
}

/// Evaluates the exact Tauris & van den Heuvel (2006, Eq. 16.20) analytical
/// orbital separation $a(q)$ in $R_\odot$ under parameterized non-conservative
/// mass transfer $(\alpha, \beta, \delta, \gamma)$ with retained fraction
/// $\epsilon = 1 - \alpha - \beta - \delta$.
///
/// Corresponds to `binary/test_suite/jdot_ml_check/src/run_binary_extras.f90:116-130`.
#[must_use]
pub fn eval_tauris_vdh_expected_separation_rsun(
    a0_rsun: f64,
    q0: f64,
    q: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
) -> f64 {
    let eps = 1.0 - alpha - beta - delta;
    let one_m_eps = (1.0 - eps).max(1.0e-15);
    let term1 = pow(q / q0, 2.0 * (alpha + gamma * delta - 1.0));
    let term2 =
        pow((q + 1.0) / (q0 + 1.0), (-alpha - beta + delta) / one_m_eps);
    let term3 = if eps.abs() > 1.0e-12 {
        let exp3 = 3.0
            + 2.0
                * (alpha * eps * eps + beta + gamma * delta * one_m_eps.powi(2))
                / (eps * one_m_eps);
        pow((eps * q + 1.0) / (eps * q0 + 1.0), exp3)
    } else {
        let coeff = 2.0 * (beta + gamma * delta);
        (coeff * (q - q0)).exp()
    };
    a0_rsun * term1 * term2 * term3
}

/// Evaluates the Brott et al. (2011, A&A 530, A115) massive-star wind mass-loss
/// rate $\dot{M}_{\mathrm{wind}} \ge 0$ in $M_\odot\,\mathrm{yr}^{-1}$
/// blending Vink et al. (2001, Eqs. 24-25), Nieuwenhuijzen & de Jager (1990),
/// and `0.1 *` Hamann, Koesterke & Wessolowski (1995) across the bi-stability
/// jump $T_{\mathrm{eff,jump}}(Z / Z_\odot)$ ($Z_\odot = 0.017$).
///
/// Corresponds line-for-line to `binary/test_suite/double_bh/src/run_star_extras.f90:55-206`.
#[must_use]
pub fn eval_brott_wind_msun_yr(
    l_surf_lsun: f64,
    m_surf_msun: f64,
    r_surf_rsun: f64,
    t_surf_k: f64,
    x_surf: f64,
    y_surf: f64,
    z_div_z_solar: f64,
) -> f64 {
    let _ = y_surf;
    if l_surf_lsun <= 0.0
        || m_surf_msun <= 0.0
        || r_surf_rsun <= 0.0
        || t_surf_k <= 0.0
        || z_div_z_solar <= 0.0
    {
        return 0.0;
    }

    let log_z_ratio = log10(z_div_z_solar);
    let teff_jump =
        1.0e3 * (61.2 + 2.59 * (-13.636 + 0.889 * log_z_ratio));

    let vink_wind = eval_brott_vink_wind(
        l_surf_lsun,
        m_surf_msun,
        t_surf_k,
        z_div_z_solar,
        log_z_ratio,
        teff_jump,
    );
    let nieu_wind = eval_brott_nieuwenhuijzen_wind(
        l_surf_lsun,
        m_surf_msun,
        r_surf_rsun,
        log_z_ratio,
    );
    let hamann_wind = eval_brott_hamann_wind(l_surf_lsun, x_surf, log_z_ratio);

    if t_surf_k < teff_jump {
        vink_wind.max(nieu_wind)
    } else {
        let alfa = if x_surf > 0.7 {
            1.0
        } else if x_surf > 0.4 {
            (x_surf - 0.4) / 0.3
        } else {
            0.0
        };
        alfa * vink_wind + (1.0 - alfa) * hamann_wind
    }
}

fn eval_brott_vink_wind(
    l_lsun: f64,
    m_msun: f64,
    t_k: f64,
    z_div_z_solar: f64,
    log_z_ratio: f64,
    teff_jump: f64,
) -> f64 {
    let alfa = if t_k > 27_500.0 {
        1.0
    } else if t_k < 22_500.0 {
        0.0
    } else {
        let dt = 100.0;
        if t_k > teff_jump + dt {
            1.0
        } else if t_k < teff_jump - dt {
            0.0
        } else {
            (t_k - (teff_jump - dt)) / (2.0 * dt)
        }
    };

    let w1 = if alfa > 0.0 {
        let vinf_div_vesc = 2.6 * pow(z_div_z_solar, 0.13);
        let log_t_40k = log10(t_k / 4.0e4);
        let log_mdot = -6.697
            + 2.194 * log10(l_lsun / 1.0e5)
            - 1.313 * log10(m_msun / 30.0)
            - 1.226 * log10(vinf_div_vesc / 2.0)
            + 0.933 * log_t_40k
            - 10.92 * log_t_40k * log_t_40k
            + 0.85 * log_z_ratio;
        exp10(log_mdot)
    } else {
        0.0
    };

    let w2 = if alfa < 1.0 {
        let vinf_div_vesc = 1.3 * pow(z_div_z_solar, 0.13);
        let log_mdot = -6.688
            + 2.210 * log10(l_lsun / 1.0e5)
            - 1.339 * log10(m_msun / 30.0)
            - 1.601 * log10(vinf_div_vesc / 2.0)
            + 1.07 * log10(t_k / 2.0e4)
            + 0.85 * log_z_ratio;
        exp10(log_mdot)
    } else {
        0.0
    };

    alfa * w1 + (1.0 - alfa) * w2
}

fn eval_brott_nieuwenhuijzen_wind(
    l_lsun: f64,
    m_msun: f64,
    r_rsun: f64,
    log_z_ratio: f64,
) -> f64 {
    let log10w = -14.02
        + 1.24 * log10(l_lsun)
        + 0.16 * log10(m_msun)
        + 0.81 * log10(r_rsun)
        + 0.85 * log_z_ratio;
    exp10(log10w)
}

fn eval_brott_hamann_wind(l_lsun: f64, x_surf: f64, log_z_ratio: f64) -> f64 {
    let log10w = -11.95 + 1.5 * log10(l_lsun) - 2.85 * x_surf + 0.85 * log_z_ratio;
    exp10(log10w) / 10.0
}
