// SPDX-License-Identifier: LGPL-3.0-or-later
//! Kerr black hole ISCO spin and Eddington accretion rate (Bardeen 1972;
//! King & Kolb 1999; Podsiadlowski et al. 2003), Tout & Eggleton (1988) tidal
//! wind enhancement, Bondi-Hoyle wind accretion (Hurley et al. 2002), and
//! accreted specific angular momentum (de Mink et al. 2013).
//!
//! Corresponds to `binary/private/binary_mdot.f90:428-622` and `binary_wind.f90`.

use flexmesa_const::{CLIGHT, ONE_THIRD, PI, STANDARD_CGRAV};
use flexmesa_math::pow;

/// Computes the equivalent initial non-rotating black hole mass
/// $M_{\mathrm{BH},0}^{\mathrm{eq}} = M_{\mathrm{BH,init}} \sqrt{r_{\mathrm{ISCO}}/6}$
/// from initial BH mass $M_{\mathrm{BH,init}}$ (g) and Kerr spin parameter
/// $a_{\mathrm{spin}} \in [0, 1]$ (Bardeen, Press & Teukolsky 1972, ApJ 178, 347).
///
/// Corresponds to `binary_evolve.f90:142-160`.
#[must_use]
pub fn bh_initial_equatorial_mass(m_bh_init: f64, a_spin_init: f64) -> f64 {
    let a = a_spin_init.clamp(0.0, 1.0);
    let one_m_a2 = (1.0 - a * a).max(0.0);
    let z1 = 1.0
        + pow(one_m_a2, ONE_THIRD)
            * (pow(1.0 + a, ONE_THIRD) + pow(1.0 - a, ONE_THIRD));
    let z2 = (3.0 * a * a + z1 * z1).sqrt();
    let r_isco =
        3.0 + z2 - ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).max(0.0).sqrt();
    m_bh_init * (r_isco / 6.0).sqrt()
}

/// Computes the instantaneous Kerr spin parameter $a_{\mathrm{spin}} \in [0, 1]$
/// of an accreting black hole of current mass `m_bh` (g) given
/// $M_{\mathrm{BH},0}^{\mathrm{eq}}$ (King & Kolb 1999, MNRAS 305, 654 Eq. 6).
///
/// Corresponds to `binary_history.f90:638-643`.
#[must_use]
pub fn eval_bh_spin(m_bh: f64, m_bh_zero_eq: f64) -> f64 {
    let m_eff = m_bh.min(6.0_f64.sqrt() * m_bh_zero_eq);
    let ratio = m_bh_zero_eq / m_eff;
    (2.0 / 3.0_f64).sqrt()
        * ratio
        * (4.0 - (18.0 * ratio * ratio - 2.0).max(0.0).sqrt())
}

/// Computes the Eddington accretion rate $\dot{M}_{\mathrm{Edd}}$ ($\mathrm{g\,s^{-1}}$)
/// and radiative efficiency $\eta$ (Podsiadlowski, Rappaport & Han 2003,
/// MNRAS 341, 385 Eqs. 6 and 9).
///
/// Corresponds to `eval_mdot_edd` in `binary/private/binary_mdot.f90:428-493`.
#[must_use]
pub fn eval_mdot_edd(
    m_accretor: f64,
    m_bh_zero_eq: f64,
    kappa_surf: f64,
) -> (f64, f64) {
    let m_ratio =
        m_accretor.min(6.0_f64.sqrt() * m_bh_zero_eq) / (3.0 * m_bh_zero_eq);
    let eta = (1.0 - (1.0 - m_ratio * m_ratio).max(0.0).sqrt()).max(1.0e-12);
    let mdot_edd =
        4.0 * PI * STANDARD_CGRAV * m_accretor / (CLIGHT * kappa_surf * eta);
    (mdot_edd, eta)
}

/// Computes the Tout & Eggleton (1988, MNRAS 231, 823 Eq. 2) tidal wind
/// enhancement factor $1 + B_{\mathrm{wind}} \min((R/R_L)^6, 0.5^6)$.
///
/// Corresponds to `Tout_enhance_wind` in `binary/private/binary_wind.f90:131-170`.
#[must_use]
pub fn eval_tout_wind_enhancement(
    r_star: f64,
    rl_star: f64,
    b_wind: f64,
) -> f64 {
    let ratio = (r_star / rl_star).min(0.5);
    let r2 = ratio * ratio;
    1.0 + b_wind * (r2 * r2 * r2)
}

/// Computes the Bondi-Hoyle wind mass transfer fraction $f_{\mathrm{wind}} \in [0, f_{\max}]$
/// (Hurley, Tout & Pols 2002, MNRAS 329, 897 Eqs. 6, 8, 9).
///
/// Corresponds to `Bondi_Hoyle_wind_transfer` in `binary/private/binary_wind.f90:83-129`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_bondi_hoyle_wind_fraction(
    m_donor: f64,
    m_accretor: f64,
    r_donor: f64,
    separation: f64,
    eccentricity: f64,
    alpha_bh: f64,
    beta_bh: f64,
    max_frac: f64,
) -> f64 {
    let v_orb2 = STANDARD_CGRAV * (m_donor + m_accretor) / separation;
    let v_wind2 = 2.0 * beta_bh * STANDARD_CGRAV * m_donor / r_donor;
    let gm_over_vw2 = STANDARD_CGRAV * m_accretor / v_wind2;
    let denom = 2.0
        * separation
        * separation
        * (1.0 - eccentricity * eccentricity).sqrt();
    let raw = (alpha_bh / denom)
        * (gm_over_vw2 * gm_over_vw2)
        * pow(1.0 + v_orb2 / v_wind2, -1.5);
    raw.min(max_frac)
}

/// Evaluates the specific angular momentum $j_{\mathrm{acc}}$ ($\mathrm{cm^2\,s^{-1}}$)
/// of accreted material and accretion mode (`1 = ballistic impact`,
/// `2 = Keplerian disk`) following Ulrich & Burger (1976) and de Mink et al.
/// (2013, ApJ 764, 166 Sec. A.3.3).
///
/// Corresponds to `eval_accreted_material_j` in `binary/private/binary_mdot.f90:950-986`.
#[must_use]
pub fn eval_accreted_material_j(
    m_donor: f64,
    m_accretor: f64,
    r_accretor: f64,
    separation: f64,
) -> (f64, i32) {
    let q = (m_accretor / m_donor).clamp(0.0667, 15.0);
    let r_min = 0.0425 * separation * pow(q + q * q, 0.25);
    if r_accretor < r_min {
        ((STANDARD_CGRAV * m_accretor * r_accretor).sqrt(), 2)
    } else {
        ((1.7 * STANDARD_CGRAV * m_accretor * r_min).sqrt(), 1)
    }
}
