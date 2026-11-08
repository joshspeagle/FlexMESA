// SPDX-License-Identifier: LGPL-3.0-or-later
//! Binary orbital mechanics, Eggleton (1983) Roche-lobe geometry, Kepler
//! relations, and eccentric true-anomaly time grids.
//!
//! Corresponds to `binary/private/binary_utils.f90` and
//! `binary/private/binary_evolve.f90`.

use flexmesa_autodiff::DualOrder1;
use flexmesa_const::{ONE_THIRD, PI, STANDARD_CGRAV};
use flexmesa_math::pow;

/// Evaluates the Roche-lobe radius $R_{L,1}$ (cm) for a primary of mass `m1`
/// (g) with companion mass `m2` (g) at orbital separation `a` (cm) using the
/// Eggleton (1983, ApJ 268, 368) approximation:
///
/// $$q^{1/3} = \left(\frac{m_1}{m_2}\right)^{1/3}, \qquad
/// R_{L,1} = a \frac{0.49\, q^{2/3}}{0.6\, q^{2/3} + \ln(1 + q^{1/3})}$$
///
/// Corresponds to `eval_rlobe` in `binary/private/binary_utils.f90:204-212`.
#[must_use]
pub fn eval_rlobe(m1: f64, m2: f64, a: f64) -> f64 {
    let q = pow(m1 / m2, ONE_THIRD);
    let q2 = q * q;
    a * 0.49 * q2 / (0.6 * q2 + q.ln_1p())
}

/// Forward-mode dual-number variant of [`eval_rlobe`] over `(m1, m2, a)`.
#[must_use]
pub fn eval_rlobe_ad<const N: usize>(
    m1: DualOrder1<N>,
    m2: DualOrder1<N>,
    a: DualOrder1<N>,
) -> DualOrder1<N> {
    let q = (m1 / m2).powf(ONE_THIRD);
    let q2 = q * q;
    a * (q2 * 0.49) / (q2 * 0.6 + (q + 1.0).ln())
}

/// Evaluates the dimensionless relative Roche-lobe gap at periastron:
///
/// $$g_i = \frac{R_i - R_{L,i}(1 - e)}{R_{L,i}(1 - e)}$$
///
/// Values $g_i < 0$ indicate detachment; $g_i \ge 0$ indicates Roche-lobe
/// overflow at periastron.
///
/// Corresponds to `set_angular_momentum_j` in `binary_utils.f90:195-198`.
#[must_use]
pub fn eval_rl_relative_gap(r: f64, rl: f64, eccentricity: f64) -> f64 {
    let rl_peri = rl * (1.0 - eccentricity);
    (r - rl_peri) / rl_peri
}

/// Computes the orbital separation $a$ (cm) from orbital period $P$ (s) and
/// component masses $m_1, m_2$ (g) via Kepler's third law:
///
/// $$a = \left[ G(m_1 + m_2) \left(\frac{P}{2\pi}\right)^2 \right]^{1/3}$$
///
/// Corresponds to `set_period_eccentricity` in `binary_utils.f90:153-155`.
#[must_use]
pub fn separation_from_period(m1: f64, m2: f64, period: f64) -> f64 {
    let p_over_2pi = period / (2.0 * PI);
    let gm = STANDARD_CGRAV * (m1 + m2);
    pow(gm * p_over_2pi * p_over_2pi, ONE_THIRD)
}

/// Computes the orbital period $P$ (s) from orbital separation $a$ (cm) and
/// component masses $m_1, m_2$ (g) via Kepler's third law:
///
/// $$P = 2\pi \sqrt{\frac{a^3}{G(m_1 + m_2)}}$$
///
/// Corresponds to `set_separation_eccentricity` in `binary_utils.f90:174-176`.
#[must_use]
pub fn period_from_separation(m1: f64, m2: f64, separation: f64) -> f64 {
    let a3 = separation * separation * separation;
    let gm = STANDARD_CGRAV * (m1 + m2);
    2.0 * PI * (a3 / gm).sqrt()
}

/// Computes the orbital angular momentum $J_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-1}}$)
/// from component masses $m_1, m_2$ (g), separation $a$ (cm), and eccentricity $e$:
///
/// $$J_{\mathrm{orb}} = m_1 m_2 \sqrt{\frac{G a (1 - e^2)}{m_1 + m_2}}$$
///
/// Corresponds to `set_angular_momentum_j` in `binary_utils.f90:190-191`.
#[must_use]
pub fn eval_j_orb(m1: f64, m2: f64, separation: f64, eccentricity: f64) -> f64 {
    let one_minus_e2 = 1.0 - eccentricity * eccentricity;
    let factor = STANDARD_CGRAV * separation * one_minus_e2 / (m1 + m2);
    m1 * m2 * factor.sqrt()
}

/// Computes the orbital separation $a$ (cm) from orbital angular momentum
/// $J_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-1}}$), masses $m_1, m_2$ (g),
/// and eccentricity $e$:
///
/// $$a = \left(\frac{J_{\mathrm{orb}}}{m_1 m_2}\right)^2
///       \frac{m_1 + m_2}{G(1 - e^2)}$$
///
/// Corresponds to `binary_evolve_step` in `binary_evolve.f90:354-355`.
#[must_use]
pub fn separation_from_j_orb(
    m1: f64,
    m2: f64,
    j_orb: f64,
    eccentricity: f64,
) -> f64 {
    let j_over_m1m2 = j_orb / (m1 * m2);
    let one_minus_e2 = 1.0 - eccentricity * eccentricity;
    j_over_m1m2 * j_over_m1m2 * (m1 + m2) / (STANDARD_CGRAV * one_minus_e2)
}

/// Evaluates the critical relative Roche-lobe overflow gap at which a contact
/// binary overflows the outer $L_2$ equipotential (Marchant et al. 2016, A&A
/// 588, A50):
///
/// $$g_{L2}(q) = 0.29858997 \arctan\!\left(1.83530121\, q^{0.39661426}\right),
/// \qquad q = \frac{\min(m_1, m_2)}{\max(m_1, m_2)}$$
///
/// Corresponds to `binary_check_model` in `binary_evolve.f90:479-484`.
#[must_use]
pub fn eval_l2_overflow_limit(m1: f64, m2: f64) -> f64 {
    let q = m1.min(m2) / m1.max(m2).max(1.0e-99);
    0.2985_8997 * (1.8353_0121 * q.powf(0.3966_1426)).atan()
}

/// Constructs the true anomaly grid $\theta_i \in [0, 2\pi)$ and normalized
/// orbital time grid $t_{\mathrm{co}}(\theta_i) \in [0, 1)$ for eccentric
/// phase-integrated mass transfer and eccentricity enhancement.
///
/// Corresponds to `binary_evolve.f90:122-135` and `343-351`.
#[must_use]
pub fn build_anomaly_grid(
    anomaly_steps: usize,
    eccentricity: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = anomaly_steps.max(2);
    let mut theta_co = Vec::with_capacity(n);
    let mut time_co = Vec::with_capacity(n);
    let e = eccentricity.clamp(0.0, 0.9999_9999);
    let sqrt_factor = ((1.0 - e) / (1.0 + e)).sqrt();
    let e_sqrt_1me2 = e * (1.0 - e * e).sqrt();

    for i in 0..n {
        let theta = 2.0 * PI * (i as f64) / (n as f64);
        theta_co.push(theta);
        let mut t_norm = (2.0 * (sqrt_factor * (0.5 * theta).tan()).atan()
            - e_sqrt_1me2 * theta.sin() / (1.0 + e * theta.cos()))
            / (2.0 * PI);
        if i > n / 2 {
            t_norm += 1.0;
        }
        time_co.push(t_norm);
    }
    (theta_co, time_co)
}
