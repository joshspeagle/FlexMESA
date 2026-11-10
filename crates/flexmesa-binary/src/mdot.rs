// SPDX-License-Identifier: LGPL-3.0-or-later
//! Explicit and implicit Roche-lobe overflow (RLOF) mass transfer prescriptions.
//!
//! Implements Ritter (1988) optically thin overflow, Kolb & Ritter (1990)
//! optically thick overflow, eccentric phase-integrated Ritter/Kolb overflow,
//! the Arras potential-expansion scheme, and the upstream MESA implicit RLOF
//! root solver (`"roche_lobe"`, `"contact"`, `"Ritter"`, `"Kolb"`).
//!
//! Corresponds to `binary/private/binary_mdot.f90`.

use flexmesa_const::{KERG, MP, MSUN, ONE_THIRD, PI, SECYER, STANDARD_CGRAV};
use flexmesa_math::{exp, log10, pow};

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
        let rl_i =
            rl_donor * one_m_e2 / (1.0 + eccentricity * theta_co[i].cos());
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
/// Input slices are ordered surface-to-center (`0..nz`), matching upstream MESA.
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
/// an eccentric orbit.
///
/// Corresponds to `get_info_for_kolb_eccentric` in `binary_mdot.f90:898-948`.
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
        let rl_i =
            rl_donor * one_m_e2 / (1.0 + eccentricity * theta_co[i].cos());
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
/// expansion of the $L_1$ potential difference.
///
/// Corresponds to `get_info_for_arras` in `binary/private/binary_mdot.f90:790-831`.
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

/// Bracketing and root-finding state for the implicit Roche-lobe overflow
/// solver (`check_implicit_rlo` & `pick_mdot_for_implicit_rlo`).
#[derive(Debug, Clone)]
pub struct ImplicitRlofSolver {
    /// Target tolerance on $|f(\dot{M}_{\mathrm{tr}})|$ (`implicit_scheme_tolerance`).
    pub tolerance: f64,
    /// Step multiplication factor (`change_factor`, in `[1.05, 1.50]`).
    pub change_factor: f64,
    /// Maximum allowed implicit $|\dot{M}_{\mathrm{tr}}|$ ($M_\odot\,\mathrm{yr}^{-1}$).
    pub max_implicit_abs_mdot_msun_yr: f64,
    /// Minimum $|\dot{M}_{\mathrm{tr}}|$ before detachment ($M_\odot\,\mathrm{yr}^{-1}$).
    pub roche_min_mdot_msun_yr: f64,
    /// Starting trial $|\dot{M}_{\mathrm{tr}}|$ ($M_\odot\,\mathrm{yr}^{-1}$).
    pub starting_mdot_msun_yr: f64,
    /// Whether to use 3-point quadratic interpolation when bracketed (`use_quadratic`).
    pub use_quadratic: bool,
}

impl Default for ImplicitRlofSolver {
    fn default() -> Self {
        Self {
            tolerance: 1.0e-4,
            change_factor: 1.25,
            max_implicit_abs_mdot_msun_yr: 1.0e-1,
            roche_min_mdot_msun_yr: 1.0e-14,
            starting_mdot_msun_yr: 1.0e-12,
            use_quadratic: true,
        }
    }
}

#[inline]
fn find0_quadratic(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    x3: f64,
    y3: f64,
) -> Option<f64> {
    let d12 = x1 - x2;
    let d23 = x2 - x3;
    let d13 = x1 - x3;
    if d12.abs() < 1.0e-40 || d23.abs() < 1.0e-40 || d13.abs() < 1.0e-40 {
        return None;
    }
    let s12 = (y1 - y2) / d12;
    let s23 = (y2 - y3) / d23;
    let a = (s12 - s23) / d13;
    let b = s12 - a * (x1 + x2);
    let c = y1 - a * x1 * x1 - b * x1;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 || a.abs() < 1.0e-40 {
        return None;
    }
    let root_sqrt = disc.sqrt();
    let r1 = (-b + root_sqrt) / (2.0 * a);
    let r2 = (-b - root_sqrt) / (2.0 * a);
    let lo = x1.min(x2).min(x3);
    let hi = x1.max(x2).max(x3);
    if r1 > lo && r1 < hi {
        Some(r1)
    } else if r2 > lo && r2 < hi {
        Some(r2)
    } else {
        None
    }
}

impl ImplicitRlofSolver {
    /// Evaluates the `"roche_lobe"` implicit objective function:
    ///
    /// $$f(\dot{M}_{\mathrm{tr}}) = 2\left(g_d + \frac{1}{2}\mathrm{tol}\right)$$
    ///
    /// Corresponds to `binary_mdot.f90:102-112`.
    #[must_use]
    pub fn roche_lobe_objective(&self, rl_relative_gap_donor: f64) -> f64 {
        2.0 * (rl_relative_gap_donor + 0.5 * self.tolerance)
    }

    /// Evaluates the `"contact"` overcontact equipotential objective function
    /// with mass ratio $q = M_a / M_d$ and Marchant et al. (2016) slope
    /// $s_c = q^{-0.52}$ (`check_implicit_rlo` in `binary_mdot.f90:113-147`):
    ///
    /// $$f_{\mathrm{contact}}(\dot{M}_{\mathrm{tr}}) = \begin{cases}
    ///   2(g_d + \tfrac{1}{2}\mathrm{tol}) & \text{if } g_a < 0 \text{ and } (g_d s_c - g_a > \mathrm{tol} \text{ or } g_d < -\mathrm{tol}), \\
    ///   g_d s_c - g_a & \text{if } q < 1, \\
    ///   g_d - g_a / s_c & \text{if } q \ge 1.
    /// \end{cases}$$
    #[must_use]
    pub fn contact_objective(
        &self,
        rl_relative_gap_donor: f64,
        rl_relative_gap_accretor: f64,
        q_accretor_over_donor: f64,
    ) -> f64 {
        let q = q_accretor_over_donor.max(1.0e-12);
        let slope_contact = pow(q, -0.52);
        if rl_relative_gap_accretor < 0.0
            && (rl_relative_gap_donor * slope_contact
                - rl_relative_gap_accretor
                > self.tolerance
                || rl_relative_gap_donor < -self.tolerance)
        {
            self.roche_lobe_objective(rl_relative_gap_donor)
        } else if q < 1.0 {
            rl_relative_gap_donor * slope_contact - rl_relative_gap_accretor
        } else {
            rl_relative_gap_donor - rl_relative_gap_accretor / slope_contact
        }
    }

    /// Solves for the implicit mass transfer rate $\dot{M}_{\mathrm{tr}} \le 0$
    /// ($\mathrm{g\,s^{-1}}$) given a callable `eval_gap(mdot_g_s) -> rl_relative_gap_donor`,
    /// following the bracketing and quadratic/secant/bisection algorithm of
    /// `check_implicit_rlo` and `pick_mdot_for_implicit_rlo`
    /// (`binary_mdot.f90:36-425`).
    ///
    /// Returns `(converged_mdot_g_s, num_iterations, final_objective)`.
    pub fn solve_roche_lobe_mdot<F>(
        &self,
        initial_mdot_g_s: f64,
        max_tries: usize,
        mut eval_gap: F,
    ) -> (f64, usize, f64)
    where
        F: FnMut(f64) -> f64,
    {
        self.solve_with_objective(initial_mdot_g_s, max_tries, |mdot| {
            self.roche_lobe_objective(eval_gap(mdot))
        })
    }

    /// Solves for the implicit contact-binary mass transfer rate
    /// $\dot{M}_{\mathrm{tr}} \le 0$ ($\mathrm{g\,s^{-1}}$) driving
    /// `contact_objective(gap_d, gap_a, q_accretor_over_donor) = 0` when both
    /// components are in overcontact (`"contact"` scheme in `binary_mdot.f90:113-147`).
    pub fn solve_contact_mdot<F>(
        &self,
        initial_mdot_g_s: f64,
        q_accretor_over_donor: f64,
        max_tries: usize,
        mut eval_gaps: F,
    ) -> (f64, usize, f64)
    where
        F: FnMut(f64) -> (f64, f64),
    {
        self.solve_with_objective(initial_mdot_g_s, max_tries, |mdot| {
            let (gap_d, gap_a) = eval_gaps(mdot);
            self.contact_objective(gap_d, gap_a, q_accretor_over_donor)
        })
    }

    fn solve_with_objective<F>(
        &self,
        initial_mdot_g_s: f64,
        max_tries: usize,
        mut eval_f: F,
    ) -> (f64, usize, f64)
    where
        F: FnMut(f64) -> f64,
    {
        let msun_per_yr = MSUN / SECYER;
        let max_abs = self.max_implicit_abs_mdot_msun_yr * msun_per_yr;
        let min_abs = self.roche_min_mdot_msun_yr * msun_per_yr;

        let mut mdot = initial_mdot_g_s.min(0.0).max(-max_abs);
        let mut have_lo = false;
        let mut have_hi = false;
        let mut mdot_lo = 0.0_f64;
        let mut mdot_hi = 0.0_f64;
        let mut f_lo = 0.0_f64;
        let mut f_hi = 0.0_f64;
        let mut mdot_prev = 0.0_f64;
        let mut f_prev = 0.0_f64;
        let mut have_prev = false;

        for iter in 1..=max_tries.max(1) {
            let f = eval_f(mdot);
            if f.abs() <= self.tolerance
                || (f < 0.0 && mdot == 0.0)
                || (f > 0.0 && (-mdot - max_abs).abs() <= 1.0e-12 * max_abs)
            {
                return (mdot, iter, f);
            }
            if f < 0.0 {
                if have_hi {
                    mdot_prev = mdot_hi;
                    f_prev = f_hi;
                    have_prev = true;
                }
                have_hi = true;
                mdot_hi = mdot;
                f_hi = f;
            } else {
                if have_lo {
                    mdot_prev = mdot_lo;
                    f_prev = f_lo;
                    have_prev = true;
                }
                have_lo = true;
                mdot_lo = mdot;
                f_lo = f;
            }

            let raw_next = if have_lo && have_hi {
                let quad_cand =
                    if self.use_quadratic && have_prev && iter % 2 == 0 {
                        find0_quadratic(
                            mdot_lo, f_lo, mdot_hi, f_hi, mdot_prev, f_prev,
                        )
                    } else {
                        None
                    };
                if let Some(q_root) = quad_cand {
                    q_root
                } else if iter % 2 == 0 && (f_hi - f_lo).abs() > 1.0e-30 {
                    let sec = (mdot_lo * f_hi - mdot_hi * f_lo) / (f_hi - f_lo);
                    if sec > mdot_hi && sec < mdot_lo {
                        sec
                    } else {
                        0.5 * (mdot_lo + mdot_hi)
                    }
                } else {
                    0.5 * (mdot_lo + mdot_hi)
                }
            } else if !have_hi {
                if mdot_lo == 0.0 {
                    -self.starting_mdot_msun_yr * msun_per_yr
                } else {
                    mdot_lo * self.change_factor
                }
            } else {
                mdot_hi / self.change_factor
            };

            let next_mdot = raw_next.max(-max_abs);
            if -next_mdot < min_abs && f < 0.0 {
                let f_zero = eval_f(0.0);
                return (0.0, iter, f_zero);
            }
            if have_lo
                && have_hi
                && (mdot_hi - mdot_lo).abs()
                    < 1.0e-6 * mdot_hi.abs().min(mdot_lo.abs())
            {
                let f_next = eval_f(next_mdot);
                return (next_mdot, iter, f_next);
            }
            mdot = next_mdot;
        }
        let final_f = eval_f(mdot);
        (mdot, max_tries, final_f)
    }
}
