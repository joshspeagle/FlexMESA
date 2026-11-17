// SPDX-License-Identifier: LGPL-3.0-or-later
//! Explicit Roche-lobe overflow (RLOF) mass transfer prescriptions and re-exports
//! of the implicit RLOF root solvers.
//!
//! Implements Ritter (1988) optically thin overflow, Kolb & Ritter (1990)
//! optically thick overflow, eccentric phase-integrated Ritter/Kolb overflow,
//! and the Arras potential-expansion scheme (`binary/private/binary_mdot.f90`).

use flexmesa_const::{KERG, MP, ONE_THIRD, PI, STANDARD_CGRAV};
use flexmesa_math::{exp, log10, pow};

pub use crate::mdot_implicit::{
    update_change_factor_after_step, ImplicitMdotSolver, ImplicitRlofSolver,
    ImplicitStepAction,
};
use crate::orbit::{build_anomaly_grid, eval_rlobe};

/// Evaluates the Ritter (1988, A&A 202, 93) optically thin Roche-lobe overflow
/// mass transfer rate $\dot{M}_{\mathrm{thin}} \le 0$ ($\mathrm{g\,s^{-1}}$).
///
/// When `is_kolb_scheme` is `true` and $R_d > R_{L,d}$, caps the exponential
/// factor at unity ($\dot{M}_{\mathrm{thin}} = -\dot{M}_{\mathrm{thin},0}$) so
/// it joins continuously onto the Kolb & Ritter (1990) optically thick branch.
///
/// Corresponds to `get_info_for_ritter` in `binary/private/binary_mdot.f90:685-727`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_ritter_mdot_thin(
    m_donor: f64,
    m_accretor: f64,
    r_donor: f64,
    rl_donor: f64,
    rho_surf: f64,
    p_surf: f64,
    t_surf: f64,
    mu_surf: f64,
    is_kolb_scheme: bool,
) -> f64 {
    let grav = STANDARD_CGRAV * m_donor / (r_donor * r_donor);
    let hp = p_surf / (grav * rho_surf);
    let v_th = (KERG * t_surf / (MP * mu_surf)).sqrt();
    let q = m_accretor / m_donor;
    let q1 = q.clamp(0.5, 10.0);
    let f1 = 1.23 + 0.5 * log10(q1);
    let mdot0 = (2.0 * PI / exp(0.5))
        * (v_th * v_th * v_th)
        * (rl_donor * rl_donor * rl_donor)
        / (STANDARD_CGRAV * m_donor)
        * rho_surf
        * f1;

    let q2 = q.clamp(0.04, 20.0);
    let log_q2 = log10(q2);
    let gamma_q = if q2 < 1.0 {
        0.954 + 0.025 * log_q2 - 0.038 * log_q2 * log_q2
    } else {
        0.954 + 0.039 * log_q2 + 0.114 * log_q2 * log_q2
    };
    let hr = hp / gamma_q;
    let ritter_exponent = (r_donor - rl_donor) / hr;

    if is_kolb_scheme && ritter_exponent > 0.0 {
        -mdot0
    } else if ritter_exponent < -100.0 {
        0.0
    } else {
        -mdot0 * exp(ritter_exponent.min(60.0))
    }
}

/// Evaluates the phase-averaged Ritter (1988) optically thin mass transfer rate
/// $\dot{M}_{\mathrm{thin}} \le 0$ ($\mathrm{g\,s^{-1}}$) over an eccentric
/// orbit using trapezoidal integration on `anomaly_steps` true-anomaly points.
///
/// Corresponds to `get_info_for_ritter_eccentric` in `binary_mdot.f90:833-896`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_ritter_mdot_eccentric(
    m_donor: f64,
    m_accretor: f64,
    r_donor: f64,
    rl_donor: f64,
    eccentricity: f64,
    rho_surf: f64,
    p_surf: f64,
    t_surf: f64,
    mu_surf: f64,
    is_kolb_scheme: bool,
    anomaly_steps: usize,
) -> f64 {
    if eccentricity <= 0.0 {
        return eval_ritter_mdot_thin(
            m_donor,
            m_accretor,
            r_donor,
            rl_donor,
            rho_surf,
            p_surf,
            t_surf,
            mu_surf,
            is_kolb_scheme,
        );
    }
    let (theta_co, time_co) = build_anomaly_grid(anomaly_steps, eccentricity);
    let n = theta_co.len();
    let one_m_e2 = 1.0 - eccentricity * eccentricity;
    let mut mdot_prev = 0.0_f64;
    let mut dm = 0.0_f64;
    for i in 0..n {
        let rl_i = rl_donor * one_m_e2
            / (1.0 + eccentricity * flexmesa_math::cos(theta_co[i]));
        let mdot_i = eval_ritter_mdot_thin(
            m_donor,
            m_accretor,
            r_donor,
            rl_i,
            rho_surf,
            p_surf,
            t_surf,
            mu_surf,
            is_kolb_scheme,
        );
        if i > 0 {
            dm += 0.5 * (mdot_prev + mdot_i) * (time_co[i] - time_co[i - 1]);
        }
        mdot_prev = mdot_i;
    }
    dm
}

/// Evaluates $F_3(\Gamma_1) = \sqrt{\Gamma_1}\left(\frac{2}{\Gamma_1+1}\right)^{\frac{\Gamma_1+1}{2(\Gamma_1-1)}}$
/// used in the Kolb & Ritter (1990) optically thick overflow integrand.
#[inline]
#[must_use]
pub fn kolb_f3(gamma1: f64) -> f64 {
    let g = gamma1.max(1.0001);
    g.sqrt() * pow(2.0 / (g + 1.0), (g + 1.0) / (2.0 * (g - 1.0)))
}

/// Evaluates the Kolb & Ritter (1990, A&A 236, 385) optically thick Roche-lobe
/// overflow mass transfer rate $\dot{M}_{\mathrm{thick}} \le 0$ ($\mathrm{g\,s^{-1}}$)
/// integrated over outer donor zones where $r_k > R_{L,d}$.
///
/// Corresponds to `calculate_kolb_mdot_thick` and `get_info_for_kolb` in
/// `binary/private/binary_mdot.f90:729-788`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_kolb_mdot_thick(
    m_donor: f64,
    m_accretor: f64,
    rl_donor: f64,
    r_profile: &[f64],
    p_profile: &[f64],
    t_profile: &[f64],
    mu_profile: &[f64],
    gamma1_profile: &[f64],
) -> f64 {
    let nz = r_profile.len();
    if nz < 2 || r_profile[0] <= rl_donor {
        return 0.0;
    }
    let mut i = 0usize;
    while i < nz && r_profile[i] > rl_donor {
        i += 1;
    }
    if i == 0 || i >= nz {
        return 0.0;
    }
    let index_r = i - 1;
    let mut integral = 0.0_f64;
    for j in 0..index_r {
        let f3 = kolb_f3(gamma1_profile[j]);
        let v_th = (KERG * t_profile[j] / (MP * mu_profile[j])).sqrt();
        integral += f3 * v_th * (p_profile[j + 1] - p_profile[j]);
    }
    let f3_last = kolb_f3(gamma1_profile[index_r]);
    let v_th_last =
        (KERG * t_profile[index_r] / (MP * mu_profile[index_r])).sqrt();
    let dr_frac = (r_profile[index_r] - rl_donor)
        / (r_profile[index_r] - r_profile[index_r + 1]).max(1.0e-30);
    integral += f3_last
        * v_th_last
        * dr_frac
        * (p_profile[index_r + 1] - p_profile[index_r]);

    let q_temp = (m_accretor / m_donor).clamp(0.5, 10.0);
    let f1 = 1.23 + 0.5 * log10(q_temp);
    -2.0 * PI * f1 * (rl_donor * rl_donor * rl_donor)
        / (STANDARD_CGRAV * m_donor)
        * integral
}

/// Evaluates the phase-averaged Kolb & Ritter (1990) optically thick mass
/// transfer rate $\dot{M}_{\mathrm{thick}} \le 0$ ($\mathrm{g\,s^{-1}}$) over
/// an eccentric orbit (`get_info_for_kolb_eccentric` in `binary_mdot.f90:898-948`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_kolb_mdot_eccentric(
    m_donor: f64,
    m_accretor: f64,
    rl_donor: f64,
    eccentricity: f64,
    r_profile: &[f64],
    p_profile: &[f64],
    t_profile: &[f64],
    mu_profile: &[f64],
    gamma1_profile: &[f64],
    anomaly_steps: usize,
) -> f64 {
    if r_profile.is_empty() {
        return 0.0;
    }
    if eccentricity <= 0.0 {
        return eval_kolb_mdot_thick(
            m_donor,
            m_accretor,
            rl_donor,
            r_profile,
            p_profile,
            t_profile,
            mu_profile,
            gamma1_profile,
        );
    }
    let one_m_e2 = 1.0 - eccentricity * eccentricity;
    if r_profile[0] < rl_donor * one_m_e2 / (1.0 + eccentricity) {
        return 0.0;
    }
    let (theta_co, time_co) = build_anomaly_grid(anomaly_steps, eccentricity);
    let n = theta_co.len();
    let mut mdot_prev = 0.0_f64;
    let mut dm = 0.0_f64;
    for i in 0..n {
        let rl_i = rl_donor * one_m_e2
            / (1.0 + eccentricity * flexmesa_math::cos(theta_co[i]));
        let mdot_i = eval_kolb_mdot_thick(
            m_donor,
            m_accretor,
            rl_i,
            r_profile,
            p_profile,
            t_profile,
            mu_profile,
            gamma1_profile,
        );
        if i > 0 {
            dm += 0.5 * (mdot_prev + mdot_i) * (time_co[i] - time_co[i - 1]);
        }
        mdot_prev = mdot_i;
    }
    dm
}

/// Evaluates the Arras optically thin Roche-lobe overflow mass transfer rate
/// $\dot{M}_{\mathrm{thin}} \le 0$ ($\mathrm{g\,s^{-1}}$) using the multipole
/// expansion of the $L_1$ potential difference (`binary_mdot.f90:790-831`).
#[must_use]
pub fn eval_arras_mdot_thin(
    m_donor: f64,
    m_accretor: f64,
    separation: f64,
    r_donor: f64,
    rho_surf: f64,
    t_surf: f64,
    mu_surf: f64,
) -> f64 {
    let q = m_accretor / m_donor;
    let v_th = (KERG * t_surf / (MP * mu_surf)).sqrt();
    let omega = (STANDARD_CGRAV * (m_donor + m_accretor)
        / (separation * separation * separation))
        .sqrt();
    let q13 = pow(q, ONE_THIRD);
    let asl = 4.0 + 4.16 / (-0.96 + q13 + 1.0 / q13);
    let v_om = v_th / omega;
    let area = 2.0 * PI * v_om * v_om / (asl * (asl - 1.0)).sqrt();

    let mfac1 = 1.0 + q;
    let mfac2 = 1.0 + 5.0 * q + 13.0 * q * q;
    let eval_phi = |rv: f64| {
        let rfac = rv / separation;
        let rfac3 = rfac * rfac * rfac;
        -(STANDARD_CGRAV * m_donor / rv)
            * (1.0 + mfac1 * rfac3 / 3.0 + 4.0 * mfac2 * rfac3 * rfac3 / 45.0)
    };
    let rv_l1 = eval_rlobe(m_donor, m_accretor, separation);
    let phi_l1 = eval_phi(rv_l1);
    let phi = eval_phi(r_donor);
    let my_ritter_exponent = (-(phi_l1 - phi) / (v_th * v_th)).min(80.0);
    let rho_l1 = (rho_surf / exp(0.5)) * exp(my_ritter_exponent);
    -rho_l1 * v_th * area
}

/// Evaluates the isotropic phase-dependent mass-transfer eccentricity
/// enhancement $\dot{e}_{\mathrm{enhance}}$ ($\mathrm{s}^{-1}$) over an
/// eccentric orbit (`edot_enhancement_Isotropic` in `binary/private/binary_edot.f90:188-228`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_edot_enhancement_isotropic(
    m_donor: f64,
    m_accretor: f64,
    r_donor: f64,
    rl_donor: f64,
    eccentricity: f64,
    rho_surf: f64,
    p_surf: f64,
    t_surf: f64,
    mu_surf: f64,
    r_profile: &[f64],
    p_profile: &[f64],
    t_profile: &[f64],
    mu_profile: &[f64],
    gamma1_profile: &[f64],
    is_kolb_scheme: bool,
    anomaly_steps: usize,
) -> f64 {
    if eccentricity <= 0.0 || (m_donor + m_accretor) <= 0.0 {
        return 0.0;
    }
    let (theta_co, time_co) = build_anomaly_grid(anomaly_steps, eccentricity);
    let n = theta_co.len();
    let one_m_e2 = 1.0 - eccentricity * eccentricity;
    let m_tot = (m_donor + m_accretor).max(1.0);
    let mut edot_prev = 0.0_f64;
    let mut de = 0.0_f64;
    for i in 0..n {
        let cos_th = flexmesa_math::cos(theta_co[i]);
        let rl_i = rl_donor * one_m_e2 / (1.0 + eccentricity * cos_th);
        let mut mdot_i = eval_ritter_mdot_thin(
            m_donor,
            m_accretor,
            r_donor,
            rl_i,
            rho_surf,
            p_surf,
            t_surf,
            mu_surf,
            is_kolb_scheme,
        );
        if is_kolb_scheme && !r_profile.is_empty() {
            mdot_i += eval_kolb_mdot_thick(
                m_donor,
                m_accretor,
                rl_i,
                r_profile,
                p_profile,
                t_profile,
                mu_profile,
                gamma1_profile,
            );
        }
        let edot_i = -mdot_i / m_tot * (eccentricity + cos_th);
        if i > 0 {
            de += 0.5 * (edot_prev + edot_i) * (time_co[i] - time_co[i - 1]);
        }
        edot_prev = edot_i;
    }
    de
}
