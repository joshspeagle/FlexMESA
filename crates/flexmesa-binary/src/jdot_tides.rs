// SPDX-License-Identifier: LGPL-3.0-or-later
//! Orbital angular momentum loss ($\dot{J}_{\mathrm{gr}}, \dot{J}_{\mathrm{ml}},
//! \dot{J}_{\mathrm{mb}}, \dot{J}_{\mathrm{ls}}$) and tidal synchronization
//! and circularization (Hut 1981; Zahn 1977; Hurley, Tout & Pols 2002).
//!
//! Corresponds to `binary/private/binary_jdot.f90`, `binary_tides.f90`, and
//! `binary_edot.f90`.

use flexmesa_const::{
    CLIGHT, MSUN, ONE_THIRD, PI, RSUN, SECYER, STANDARD_CGRAV,
};
use flexmesa_math::{exp, pow};

/// Evaluates gravitational wave orbital angular momentum loss $\dot{J}_{\mathrm{gr}}$
/// ($\mathrm{g\,cm^2\,s^{-2}}$) in `default_jdot_gr` (`binary/private/binary_jdot.f90:95-111`):
///
/// $$\dot{J}_{\mathrm{gr}} = -\frac{32 G^3 M_a M_d (M_a + M_d)}{5 c^5 a^4} J_{\mathrm{orb}}$$
#[must_use]
pub fn eval_jdot_gr(
    m_donor: f64,
    m_accretor: f64,
    separation: f64,
    j_orb: f64,
) -> f64 {
    let g3 = STANDARD_CGRAV * STANDARD_CGRAV * STANDARD_CGRAV;
    let c5 = CLIGHT * CLIGHT * CLIGHT * CLIGHT * CLIGHT;
    let a2 = separation * separation;
    let a4 = a2 * a2;
    -(32.0 * g3 * m_accretor * m_donor * (m_accretor + m_donor)
        / (5.0 * c5 * a4))
        * j_orb
}

/// Evaluates orbital angular momentum loss due to mass loss from the system
/// $\dot{J}_{\mathrm{ml}}$ ($\mathrm{g\,cm^2\,s^{-2}}$) following Tauris &
/// van den Heuvel (2006) (`default_jdot_ml`, `binary_jdot.f90:113-134`):
///
/// $$\dot{J}_{\mathrm{ml}} =
/// \left[
///   (\dot{M}_{\mathrm{sys,tr},d} + \dot{M}_{\mathrm{sys,wind},d})
///   \left(\frac{M_a}{M_a+M_d} a\right)^2 +
///   (\dot{M}_{\mathrm{sys,tr},a} + \dot{M}_{\mathrm{sys,wind},a})
///   \left(\frac{M_d}{M_a+M_d} a\right)^2
/// \right] \frac{2\pi}{P}\sqrt{1 - e^2} +
/// \dot{M}_{\mathrm{sys,cct}}\,\gamma_{\mathrm{mt}}\sqrt{G(M_d + M_a)a}$$
///
/// All system mass-loss rate arguments are negative ($\le 0$, $\mathrm{g\,s^{-1}}$).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_jdot_ml(
    m_donor: f64,
    m_accretor: f64,
    separation: f64,
    period: f64,
    eccentricity: f64,
    mdot_sys_tr_donor: f64,
    mdot_sys_wind_donor: f64,
    mdot_sys_tr_accretor: f64,
    mdot_sys_wind_accretor: f64,
    mdot_sys_cct: f64,
    gamma_mt: f64,
) -> f64 {
    let m_tot = m_donor + m_accretor;
    let r_d = (m_accretor / m_tot) * separation;
    let r_a = (m_donor / m_tot) * separation;
    let omega_orb = 2.0 * PI / period;
    let jdot_iso = ((mdot_sys_tr_donor + mdot_sys_wind_donor) * (r_d * r_d)
        + (mdot_sys_tr_accretor + mdot_sys_wind_accretor) * (r_a * r_a))
        * omega_orb
        * (1.0 - eccentricity * eccentricity).sqrt();
    let jdot_cct =
        mdot_sys_cct * gamma_mt * (STANDARD_CGRAV * m_tot * separation).sqrt();
    jdot_iso + jdot_cct
}

/// Evaluates magnetic braking orbital angular momentum loss $\dot{J}_{\mathrm{mb}}$
/// ($\mathrm{g\,cm^2\,s^{-2}}$) following Rappaport, Verbunt & Joss (1983, ApJ
/// 275, 713) with optional Podsiadlowski et al. (2002, ApJ 565, 1107) low
/// convective-envelope damping (`default_jdot_mb`, `binary_jdot.f90:247-328`):
///
/// $$\dot{J}_{\mathrm{mb}} = -3.8\times 10^{-30}\, M\, R_\odot^4
/// \left(\frac{\min(R, R_L)}{R_\odot}\right)^{\gamma_{\mathrm{mb}}}
/// \left(\frac{2\pi}{P}\right)^3 f_{\mathrm{scale}}(q_{\mathrm{conv,env}})$$
#[must_use]
pub fn eval_jdot_mb(
    m_star: f64,
    r_star: f64,
    rl_star: f64,
    period: f64,
    gamma_mb: f64,
    q_conv_env: f64,
    scale_for_low_qconv_env: bool,
) -> f64 {
    const Q_SCALE: f64 = 0.02;
    let r_eff = r_star.min(rl_star);
    let rsun2 = RSUN * RSUN;
    let rsun4 = rsun2 * rsun2;
    let omega = 2.0 * PI / period;
    let base = -3.8e-30
        * m_star
        * rsun4
        * pow(r_eff / RSUN, gamma_mb)
        * (omega * omega * omega);
    if scale_for_low_qconv_env && q_conv_env < Q_SCALE {
        base * exp(-Q_SCALE / q_conv_env.max(1.0e-99) + 1.0)
    } else {
        base
    }
}

/// Evaluates spin-orbit coupling torque $\dot{J}_{\mathrm{ls}}$ ($\mathrm{g\,cm^2\,s^{-2}}$)
/// conserving total system angular momentum against non-wind stellar spin changes.
///
/// Corresponds to `default_jdot_ls` in `binary/private/binary_jdot.f90:136-174`.
#[must_use]
pub fn eval_jdot_ls(
    s_donor_old: f64,
    s_donor: f64,
    s_accretor_old: f64,
    s_accretor: f64,
    dt: f64,
) -> f64 {
    eval_jdot_ls_with_winds(
        s_donor_old,
        s_donor,
        0.0,
        0.0,
        s_accretor_old,
        s_accretor,
        0.0,
        0.0,
        dt,
    )
}

/// Evaluates spin-orbit coupling torque $\dot{J}_{\mathrm{ls}}$ ($\mathrm{g\,cm^2\,s^{-2}}$)
/// subtracting stellar wind spin-down $\dot{M}_{\mathrm{wind}}\,j_{\mathrm{surf}}\,\Delta t$
/// from each component's spin angular momentum change (`default_jdot_ls` in
/// `binary/private/binary_jdot.f90:152-173`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_jdot_ls_with_winds(
    s_donor_old: f64,
    s_donor: f64,
    mdot_wind_donor: f64,
    j_surf_donor: f64,
    s_accretor_old: f64,
    s_accretor: f64,
    mdot_wind_accretor: f64,
    j_surf_accretor: f64,
    dt: f64,
) -> f64 {
    let dt_safe = dt.max(1.0e-99);
    let dj_donor =
        s_donor_old - s_donor + mdot_wind_donor * j_surf_donor * dt_safe;
    let dj_accretor = s_accretor_old - s_accretor
        + mdot_wind_accretor * j_surf_accretor * dt_safe;
    (dj_donor + dj_accretor) / dt_safe
}

/// Evaluates the missing wind spin angular momentum torque on the accretor
/// $\dot{J}_{\mathrm{missing\_wind}}$ ($\mathrm{g\,cm^2\,s^{-2}}$) when stellar
/// mass accretion partially or fully masks wind mass loss inside the accretor's
/// net $\dot{M}_a$ (`default_jdot_missing_wind` in
/// `binary/private/binary_jdot.f90:176-199`).
#[must_use]
pub fn eval_jdot_missing_wind(
    is_point_mass: bool,
    mstar_dot_accretor: f64,
    mtransfer_rate: f64,
    fixed_xfer_fraction: f64,
    mdot_system_wind_accretor: f64,
    j_surf_accretor: f64,
) -> f64 {
    if is_point_mass {
        return 0.0;
    }
    let mdot_missing = if mstar_dot_accretor < 0.0 {
        mtransfer_rate * fixed_xfer_fraction
    } else {
        mdot_system_wind_accretor
    };
    mdot_missing * j_surf_accretor
}

/// Evaluates the Peters (1964, PhRv 136, B1224) gravitational-wave orbital
/// circularization rate $\dot{e}_{\mathrm{gr}}$ ($\mathrm{s}^{-1}$)
/// (`default_edot_gr` in `binary/private/binary_edot.f90:92-116`):
///
/// $$\dot{e}_{\mathrm{gr}} = -\frac{304\,G^3 M_1 M_2 (M_1 + M_2)}{15\,c^5 a^4 (1 - e^2)^{5/2}}\,
///   e\left(1 + \frac{121}{304} e^2\right)$$
#[must_use]
pub fn eval_edot_gr(
    m_donor: f64,
    m_accretor: f64,
    separation: f64,
    eccentricity: f64,
) -> f64 {
    if eccentricity <= 0.0 || separation <= 0.0 {
        return 0.0;
    }
    let g3 = STANDARD_CGRAV * STANDARD_CGRAV * STANDARD_CGRAV;
    let c5 = CLIGHT * CLIGHT * CLIGHT * CLIGHT * CLIGHT;
    let a2 = separation * separation;
    let a4 = a2 * a2;
    let e2 = eccentricity * eccentricity;
    let one_m_e2 = (1.0 - e2).max(1.0e-12);
    -(304.0 * g3 * m_donor * m_accretor * (m_donor + m_accretor) * eccentricity
        / (15.0 * c5 * a4 * pow(one_m_e2, 2.5)))
        * (1.0 + (121.0 / 304.0) * e2)
}

/// Hut (1981, A&A 99, 126 Eq. 11) eccentricity polynomial
/// $f_2(e) = 1 + \frac{15}{2}e^2 + \frac{45}{8}e^4 + \frac{5}{16}e^6$.
#[inline]
#[must_use]
pub fn hut_f2(e: f64) -> f64 {
    let e2 = e * e;
    let e4 = e2 * e2;
    1.0 + 7.5 * e2 + 5.625 * e4 + 0.3125 * e4 * e2
}

/// Hut (1981, A&A 99, 126 Eq. 11) eccentricity polynomial
/// $f_3(e) = 1 + \frac{15}{4}e^2 + \frac{15}{8}e^4 + \frac{5}{64}e^6$.
#[inline]
#[must_use]
pub fn hut_f3(e: f64) -> f64 {
    let e2 = e * e;
    let e4 = e2 * e2;
    1.0 + 3.75 * e2 + 1.875 * e4 + (5.0 / 64.0) * e4 * e2
}

/// Hut (1981, A&A 99, 126 Eq. 11) eccentricity polynomial
/// $f_4(e) = 1 + \frac{3}{2}e^2 + \frac{1}{8}e^4$.
#[inline]
#[must_use]
pub fn hut_f4(e: f64) -> f64 {
    let e2 = e * e;
    1.0 + 1.5 * e2 + 0.125 * e2 * e2
}

/// Hut (1981, A&A 99, 126 Eq. 11) eccentricity polynomial
/// $f_5(e) = 1 + 3e^2 + \frac{3}{8}e^4$.
#[inline]
#[must_use]
pub fn hut_f5(e: f64) -> f64 {
    let e2 = e * e;
    1.0 + 3.0 * e2 + 0.375 * e2 * e2
}

/// Evaluates the convective-envelope tidal dissipation ratio $(k/T)_{\mathrm{conv}}$
/// ($\mathrm{s}^{-1}$) following Hurley, Tout & Pols (2002, MNRAS 329, 897 Eqs. 30-31).
///
/// Arguments `m_star_msun`, `m_env_msun`, `r_phot_rsun`, `r_env_rsun`, and
/// `l_phot_lsun` are in solar units; `period_sec` is in seconds and
/// `omega_surf` is in $\mathrm{rad\,s^{-1}}$.
///
/// Corresponds to `k_div_T` in `binary/private/binary_tides.f90:417-437`.
#[must_use]
pub fn eval_k_div_t_convective(
    m_star_msun: f64,
    m_env_msun: f64,
    r_phot_rsun: f64,
    r_env_rsun: f64,
    l_phot_lsun: f64,
    period_sec: f64,
    omega_surf: f64,
    tidal_reduction: f64,
) -> f64 {
    let tau_conv = 0.431
        * pow(
            m_env_msun * r_env_rsun * (r_phot_rsun - 0.5 * r_env_rsun)
                / (3.0 * l_phot_lsun.max(1.0e-30)),
            ONE_THIRD,
        )
        * SECYER;
    let freq_diff = (1.0 / period_sec - omega_surf / (2.0 * PI)).abs();
    let f_conv = if freq_diff <= 1.0e-99 {
        1.0
    } else {
        let p_tid = 1.0 / freq_diff;
        pow(p_tid / (2.0 * tau_conv), tidal_reduction).min(1.0)
    };
    (2.0 / 21.0) * (f_conv / tau_conv) * (m_env_msun / m_star_msun)
}

/// Evaluates the radiative-envelope dynamical tide coupling ratio $(k/T)_{\mathrm{rad}}$
/// ($\mathrm{s}^{-1}$) following Zahn (1975, 1977) and Hurley, Tout & Pols (2002 Eq. 42).
///
/// Masses and radii/separation are in CGS (`m_star`, `m_companion` in g;
/// `r_phot`, `separation` in cm).
///
/// Corresponds to `k_div_T` in `binary/private/binary_tides.f90:438-456`.
#[must_use]
pub fn eval_k_div_t_radiative(
    m_star: f64,
    m_companion: f64,
    r_phot: f64,
    separation: f64,
) -> f64 {
    let m_msun = m_star / MSUN;
    let q = m_companion / m_star;
    let e2 = 1.592e-9 * pow(m_msun, 2.84);
    let r2 = r_phot * r_phot;
    let a5 = separation * separation * separation * separation * separation;
    let rsun3 = RSUN * RSUN * RSUN;
    (1.9782e4 / SECYER)
        * ((m_star * r2 / a5) / (MSUN / rsun3)).sqrt()
        * pow(1.0 + q, 5.0 / 6.0)
        * e2
}

/// Evaluates the Hut (1981, Eq. 11) spin synchronization timescale $t_{\mathrm{sync}}$ (s):
///
/// $$\frac{1}{t_{\mathrm{sync}}} = 3 F_{\mathrm{tid}} \left(\frac{k}{T}\right)
/// \frac{q^2}{r_{\mathrm{gyr}}^2} \left(\frac{R_{\mathrm{phot}}}{a}\right)^6$$
///
/// Corresponds to `get_tsync` in `binary/private/binary_tides.f90:360-385`.
#[must_use]
pub fn eval_tsync(
    k_div_t: f64,
    q_companion_over_star: f64,
    r_gyr_sq: f64,
    r_phot: f64,
    separation: f64,
    f_tid: f64,
) -> f64 {
    let r_over_a = r_phot / separation;
    let r2 = r_over_a * r_over_a;
    let r6 = r2 * r2 * r2;
    let inv_tsync = 3.0
        * f_tid
        * k_div_t
        * (q_companion_over_star * q_companion_over_star / r_gyr_sq)
        * r6;
    1.0 / inv_tsync.max(1.0e-99)
}

/// Evaluates the Hut (1981, Eq. 10) / Hurley et al. (2002, Eq. 29) tidal
/// circularization rate $\dot{e}_{\mathrm{tidal},s}$ ($\mathrm{s}^{-1}$) for a
/// single stellar component:
///
/// $$\dot{e}_{\mathrm{tidal},s} = -27 F_{\mathrm{tid}} \left(\frac{k}{T}\right)_s
/// q(1 + q) \left(\frac{R_{\mathrm{phot}}}{a}\right)^8
/// \frac{e}{(1 - e^2)^{13/2}}
/// \left[ f_3(e) - \frac{11}{18}\frac{\Omega_s}{\Omega_{\mathrm{orb}}}
///        f_4(e) (1 - e^2)^{3/2} \right]$$
///
/// Corresponds to `edot_tidal_Hut` in `binary/private/binary_edot.f90:146-182`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_edot_tidal_hut(
    k_div_t: f64,
    m_star: f64,
    m_companion: f64,
    r_phot: f64,
    separation: f64,
    eccentricity: f64,
    omega_star: f64,
    period: f64,
    f_tid: f64,
) -> f64 {
    if eccentricity <= 0.0 {
        return 0.0;
    }
    let q = m_companion / m_star;
    let r_over_a = r_phot / separation;
    let r2 = r_over_a * r_over_a;
    let r4 = r2 * r2;
    let r8 = r4 * r4;
    let one_m_e2 = (1.0 - eccentricity * eccentricity).max(1.0e-12);
    let omega_orb = 2.0 * PI / period;
    let bracket = hut_f3(eccentricity)
        - (11.0 / 18.0)
            * (omega_star / omega_orb)
            * hut_f4(eccentricity)
            * pow(one_m_e2, 1.5);
    -27.0
        * f_tid
        * k_div_t
        * q
        * (1.0 + q)
        * r8
        * eccentricity
        * pow(one_m_e2, -6.5)
        * bracket
}
