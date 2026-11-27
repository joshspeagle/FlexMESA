// SPDX-License-Identifier: GPL-3.0-or-later
//! Kerr black hole ISCO spin and Eddington accretion rate (Bardeen 1972;
//! King & Kolb 1999; Podsiadlowski et al. 2003), decoupled Eddington mass
//! retention vs. radiation-corrected accretion (`binary_mdot.f90:428-622`),
//! Tout & Eggleton (1988) tidal wind enhancement, Bondi-Hoyle wind accretion
//! (Hurley et al. 2002), and accreted specific angular momentum (de Mink et al.
//! 2013).
//!
//! Corresponds to `binary/private/binary_mdot.f90:428-622` and `binary_wind.f90`.

use flexmesa_const::{CLIGHT, MSUN, ONE_THIRD, PI, SECYER, STANDARD_CGRAV};
use flexmesa_math::{cos, pow, pow2, pow6};

use crate::orbit::build_anomaly_grid;

/// Point-mass Eddington retention and system mass-loss accounting returned by
/// [`eval_point_mass_edd_and_retention`] (`adjust_mdots` in
/// `binary/private/binary_mdot.f90:577-620`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointMassEddRetention {
    /// Eddington accretion ceiling $\dot{M}_{\mathrm{Edd}}$ ($\mathrm{g\,s^{-1}}$).
    pub mdot_edd: f64,
    /// Radiative efficiency $\eta \in [0, 1]$ (`mdot_edd_eta`).
    pub mdot_edd_eta: f64,
    /// Total arriving mass rate at accretor `actual_mtransfer_rate` ($\le 0$, $\mathrm{g\,s^{-1}}$).
    pub actual_mtransfer_rate: f64,
    /// Pre-radiation-correction accreted mass rate $\dot{M}_{\mathrm{acc,raw}} \ge 0$ ($\mathrm{g\,s^{-1}}$).
    pub mdot_acc_raw: f64,
    /// Final retained point-mass growth rate `component_mdot(a_i)` ($\ge 0$, $\mathrm{g\,s^{-1}}$).
    pub component_mdot_accretor: f64,
    /// Accretion luminosity $\eta \dot{M}_{\mathrm{acc,raw}} c^2$ ($\mathrm{erg\,s^{-1}}$).
    pub accretion_luminosity: f64,
    /// Mass lost isotropically from donor vicinity `mdot_system_transfer(d_i)` ($\le 0$, $\mathrm{g\,s^{-1}}$).
    pub mdot_system_transfer_donor: f64,
    /// Mass lost isotropically from accretor vicinity `mdot_system_transfer(a_i)` ($\le 0$, $\mathrm{g\,s^{-1}}$).
    pub mdot_system_transfer_accretor: f64,
    /// Mass lost from circumbinary coplanar toroid `mdot_system_cct` ($\le 0$, $\mathrm{g\,s^{-1}}$).
    pub mdot_system_cct: f64,
}

/// Computes the equivalent initial non-rotating black hole mass
/// $M_{\mathrm{BH},0}^{\mathrm{eq}} = M_{\mathrm{BH,init}} \sqrt{r_{\mathrm{ISCO}}/6}$
/// from initial BH mass $M_{\mathrm{BH,init}}$ (g) and Kerr spin parameter
/// $a_{\mathrm{spin}}$ clamped to $[0, 1]$ (Bardeen, Press & Teukolsky 1972,
/// ApJ 178, 347).
///
/// Corresponds to `run_binary_support.f90:312-316` and `binary_evolve.f90:142-160`.
#[must_use]
pub fn bh_initial_equatorial_mass(m_bh_init: f64, a_spin_init: f64) -> f64 {
    let a = a_spin_init.clamp(0.0, 1.0);
    let one_m_a2 = (1.0 - a * a).max(0.0);
    let z1 = 1.0
        + pow(one_m_a2, ONE_THIRD)
            * (pow(1.0 + a, ONE_THIRD) + pow(1.0 - a, ONE_THIRD));
    // `binary/private/binary_evolve.f90:154`: `sqrt(3d0*pow2(a) + pow2(Z1))`.
    let z2 = (3.0 * pow2(a) + pow2(z1)).sqrt();
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
    if m_bh_zero_eq <= 0.0 || m_bh <= 0.0 {
        return 0.0;
    }
    let m_eff = m_bh.min(6.0_f64.sqrt() * m_bh_zero_eq);
    let ratio = m_bh_zero_eq / m_eff;
    ((2.0 / 3.0_f64).sqrt()
        * ratio
        // `binary/private/binary_history.f90:641`: `18d0*pow2(...)`.
        * (4.0 - (18.0 * pow2(ratio) - 2.0).max(0.0).sqrt()))
    .clamp(0.0, 1.0)
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
    eval_mdot_edd_with_overrides(
        m_accretor,
        m_bh_zero_eq,
        kappa_surf,
        -1.0,
        -1.0,
    )
}

/// Evaluates `eval_mdot_edd` (`binary_mdot.f90:428-493`) honoring optional
/// namelist overrides `use_this_for_mdot_edd_eta > 0` and
/// `use_this_for_mdot_edd > 0` (in $M_\odot\,\mathrm{yr}^{-1}$).
#[must_use]
pub fn eval_mdot_edd_with_overrides(
    m_accretor: f64,
    m_bh_zero_eq: f64,
    kappa_surf: f64,
    use_this_for_mdot_edd_eta: f64,
    use_this_for_mdot_edd_msun_yr: f64,
) -> (f64, f64) {
    let eq_mass = if m_bh_zero_eq > 0.0 {
        m_bh_zero_eq
    } else {
        m_accretor
    };
    let eta = if use_this_for_mdot_edd_eta > 0.0 {
        use_this_for_mdot_edd_eta
    } else {
        let m_ratio = m_accretor.min(6.0_f64.sqrt() * eq_mass)
            / (3.0 * eq_mass.max(1.0e-99));
        (1.0 - (1.0 - m_ratio * m_ratio).max(0.0).sqrt()).max(1.0e-12)
    };

    let mdot_edd = if use_this_for_mdot_edd_msun_yr > 0.0 {
        use_this_for_mdot_edd_msun_yr * (MSUN / SECYER)
    } else {
        4.0 * PI * STANDARD_CGRAV * m_accretor
            / (CLIGHT * kappa_surf.max(1.0e-12) * eta)
    };
    (mdot_edd, eta)
}

/// Evaluates decoupled Eddington retention (`limit_retention_by_mdot_edd`) and
/// rest-mass radiation loss (`use_radiation_corrected_transfer_rate`) for a
/// point-mass accretor, matching `eval_mdot_edd` and `adjust_mdots` in
/// `binary/private/binary_mdot.f90:428-493, 577-620`:
///
/// 1. $\dot{M}_{\mathrm{arr}} = -(\dot{M}_{\mathrm{RLOF}}\,\epsilon + \dot{M}_{\mathrm{wind,tr},d}) \ge 0$.
/// 2. If `limit_retention_by_mdot_edd`, $\dot{M}_{\mathrm{acc,raw}} = \min(\dot{M}_{\mathrm{arr}}, \dot{M}_{\mathrm{Edd}})$
///    even when `use_radiation_corrected_transfer_rate == false`.
/// 3. $L_{\mathrm{acc}} = \eta \dot{M}_{\mathrm{acc,raw}} c^2$.
/// 4. If `use_radiation_corrected_transfer_rate`, $\dot{M}_a = (1 - \eta)\dot{M}_{\mathrm{acc,raw}}$,
///    otherwise $\dot{M}_a = \dot{M}_{\mathrm{acc,raw}}$.
/// 5. Excess mass (both super-Eddington overflow and radiated rest mass) leaves
///    isotropically with the accretor's specific orbital angular momentum via
///    `mdot_system_transfer_accretor`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_point_mass_edd_and_retention(
    m_accretor: f64,
    m_bh_zero_eq: f64,
    donor_surf_h1: f64,
    donor_surf_opacity: f64,
    use_es_opacity_for_mdot_edd: bool,
    use_this_for_mdot_edd_eta: f64,
    use_this_for_mdot_edd_msun_yr: f64,
    limit_retention_by_mdot_edd: bool,
    use_radiation_corrected_transfer_rate: bool,
    mtransfer_rate: f64,
    mdot_wind_transfer_donor: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
) -> PointMassEddRetention {
    let kappa = if use_es_opacity_for_mdot_edd {
        0.2 * (1.0 + donor_surf_h1.clamp(0.0, 1.0))
    } else {
        donor_surf_opacity.max(1.0e-6)
    };
    let (mdot_edd, mdot_edd_eta) = eval_mdot_edd_with_overrides(
        m_accretor,
        m_bh_zero_eq,
        kappa,
        use_this_for_mdot_edd_eta,
        use_this_for_mdot_edd_msun_yr,
    );

    let fixed_xfer_fraction = 1.0 - alpha - beta - delta;
    let actual_mtransfer_rate =
        mtransfer_rate * fixed_xfer_fraction + mdot_wind_transfer_donor;
    let mut mdot_acc_raw = (-actual_mtransfer_rate).max(0.0);
    if limit_retention_by_mdot_edd && mdot_acc_raw > mdot_edd {
        mdot_acc_raw = mdot_edd;
    }
    let accretion_luminosity = mdot_edd_eta * mdot_acc_raw * CLIGHT * CLIGHT;
    let component_mdot_accretor = if use_radiation_corrected_transfer_rate {
        (1.0 - mdot_edd_eta) * mdot_acc_raw
    } else {
        mdot_acc_raw
    };

    let (
        mdot_system_transfer_donor,
        mdot_system_transfer_accretor,
        mdot_system_cct,
    ) = if mtransfer_rate + mdot_wind_transfer_donor >= 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        (
            mtransfer_rate * alpha,
            (actual_mtransfer_rate + component_mdot_accretor)
                + mtransfer_rate * beta,
            mtransfer_rate * delta,
        )
    };

    PointMassEddRetention {
        mdot_edd,
        mdot_edd_eta,
        actual_mtransfer_rate,
        mdot_acc_raw,
        component_mdot_accretor,
        accretion_luminosity,
        mdot_system_transfer_donor,
        mdot_system_transfer_accretor,
        mdot_system_cct,
    }
}

/// Computes the Tout & Eggleton (1988, MNRAS 231, 823 Eq. 2) circular tidal wind
/// enhancement factor $1 + B_{\mathrm{wind}} \min((R/R_L)^6, 0.5^6)$.
#[must_use]
pub fn eval_tout_wind_enhancement(
    r_star: f64,
    rl_star: f64,
    b_wind: f64,
) -> f64 {
    1.0 + b_wind * pow6(r_star / rl_star.max(1.0)).min(pow6(0.5))
}

/// Evaluates orbit-averaged Tout & Eggleton (1988, MNRAS 231, 823 Eq. 2) tidally
/// enhanced wind mass-loss rate ($\mathrm{g\,s^{-1}}$) integrated over
/// `anomaly_steps` true-anomaly points (`Tout_enhance_wind` in
/// `binary/private/binary_wind.f90:131-170`).
#[must_use]
pub fn eval_tout_wind_enhancement_orbit_averaged(
    mstar_dot: f64,
    r_star: f64,
    rl_star: f64,
    b_wind: f64,
    eccentricity: f64,
    anomaly_steps: usize,
) -> f64 {
    let (theta_co, time_co) = build_anomaly_grid(anomaly_steps, eccentricity);
    let n = theta_co.len();
    let one_m_e2 = 1.0 - pow2(eccentricity);
    let cap = pow6(0.5);
    let mut mdot_prev = 0.0_f64;
    let mut dm = 0.0_f64;
    for i in 0..n {
        let rl_d = one_m_e2 / (1.0 + eccentricity * cos(theta_co[i])) * rl_star;
        let r_rl = pow6(r_star / rl_d.max(1.0)).min(cap);
        let mdot_i = mstar_dot * (1.0 + b_wind * r_rl);
        if i > 0 {
            dm += 0.5 * (mdot_prev + mdot_i) * (time_co[i] - time_co[i - 1]);
        }
        mdot_prev = mdot_i;
    }
    dm
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
    if separation <= 0.0 || r_donor <= 0.0 || m_donor <= 0.0 || beta_bh <= 0.0 {
        return 0.0;
    }
    let v_orb = (STANDARD_CGRAV * (m_donor + m_accretor) / separation).sqrt();
    let v_wind = (2.0 * beta_bh * STANDARD_CGRAV * m_donor / r_donor).sqrt();
    let raw = alpha_bh
        / pow2(separation)
        / (2.0 * (1.0 - pow2(eccentricity)).max(1.0e-12).sqrt())
        * pow2(STANDARD_CGRAV * m_accretor / pow2(v_wind))
        * pow(1.0 + pow2(v_orb / v_wind), -1.5);
    raw.clamp(0.0, max_frac)
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
    let q = (m_accretor / m_donor.max(1.0e-99)).clamp(0.0667, 15.0);
    let r_min = 0.0425 * separation * pow(q + q * q, 0.25);
    if r_accretor < r_min {
        (
            (STANDARD_CGRAV * m_accretor * r_accretor.max(0.0)).sqrt(),
            2,
        )
    } else {
        (
            (STANDARD_CGRAV * m_accretor * 1.7 * r_min.max(0.0)).sqrt(),
            1,
        )
    }
}
