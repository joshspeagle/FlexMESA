// SPDX-License-Identifier: LGPL-3.0-or-later
//! Orbital angular momentum loss ($\dot{J}_{\mathrm{gr}}, \dot{J}_{\mathrm{ml}},
//! \dot{J}_{\mathrm{mb}}, \dot{J}_{\mathrm{ls}}, \dot{J}_{\mathrm{missing\_wind}}$)
//! and tidal synchronization/circularization (Hut 1981; Zahn 1977; Hurley, Tout
//! & Pols 2002).
//!
//! Corresponds to `binary/private/binary_jdot.f90`, `binary_tides.f90`, and
//! `binary_edot.f90`.

use flexmesa_const::{
    CLIGHT, MSUN, MixingType, ONE_THIRD, PI, RSUN, SECYER, STANDARD_CGRAV,
};
use flexmesa_math::{exp, pow, pow2, pow3, pow4, pow5, pow6, pow8};
use flexmesa_star::StarModel;

use crate::inlist::BinaryInlistControls;

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
    let bs4 = pow4(separation);
    let clight5 = pow5(CLIGHT);
    let cgrav3 = STANDARD_CGRAV * STANDARD_CGRAV * STANDARD_CGRAV;
    -32.0 * cgrav3 * m_accretor * m_donor * (m_accretor + m_donor)
        / (5.0 * clight5 * bs4)
        * j_orb
}

/// Evaluates orbital angular momentum loss due to mass loss from the system
/// $\dot{J}_{\mathrm{ml}}$ ($\mathrm{g\,cm^2\,s^{-2}}$) following Tauris &
/// van den Heuvel (2006) (`default_jdot_ml`, `binary_jdot.f90:113-134`).
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
    let m_ad = m_accretor + m_donor;
    let sqrt_1me2 = (1.0 - pow2(eccentricity)).sqrt();
    let jdot_d = (mdot_sys_tr_donor + mdot_sys_wind_donor)
        * pow2(m_accretor / m_ad * separation)
        * 2.0
        * PI
        / period
        * sqrt_1me2;
    let jdot_a = (mdot_sys_tr_accretor + mdot_sys_wind_accretor)
        * pow2(m_donor / m_ad * separation)
        * 2.0
        * PI
        / period
        * sqrt_1me2;
    let jdot_cct = mdot_sys_cct
        * gamma_mt
        * (STANDARD_CGRAV * (m_donor + m_accretor) * separation).sqrt();
    jdot_d + jdot_a + jdot_cct
}

/// Evaluates `check_jdot_mb_conditions` (`binary/private/binary_jdot.f90:201-246, 266, 296`),
/// returning `(use_mb, qconv_env)` for `star` given convective core/envelope
/// mass-fraction bounds in `ctrl` and `keep_mb_on`.
#[must_use]
pub fn check_jdot_mb_conditions(
    star: &StarModel,
    ctrl: &BinaryInlistControls,
    _was_using_mb: bool,
) -> (bool, f64) {
    let nz = star.nz;
    if nz == 0 {
        return (ctrl.keep_mb_on, 0.0);
    }

    let mut qconv_core = 0.0_f64;
    for k in (0..nz).rev() {
        let is_conv = star
            .mixing_type
            .get(k)
            .copied()
            .map_or(star.gradr[k] > star.grada[k], |mt| {
                mt == MixingType::ConvectiveMixing
            });
        if star.q[k] > ctrl.jdot_mb_qlim_for_check_rad_core
            && (qconv_core == 0.0 || !is_conv)
        {
            break;
        }
        if is_conv {
            qconv_core += star.dq[k];
        }
    }

    let mut qconv_env = 0.0_f64;
    for k in 0..nz {
        let is_conv = star
            .mixing_type
            .get(k)
            .copied()
            .map_or(star.gradr[k] > star.grada[k], |mt| {
                mt == MixingType::ConvectiveMixing
            });
        if star.q[k] < ctrl.jdot_mb_qlim_for_check_conv_env
            && (qconv_env == 0.0 || !is_conv)
        {
            break;
        }
        if is_conv {
            qconv_env += star.dq[k];
        }
    }

    let apply_jdot_mb = qconv_env >= ctrl.jdot_mb_min_qconv_env
        && qconv_env <= ctrl.jdot_mb_max_qconv_env
        && qconv_core <= ctrl.jdot_mb_max_qconv_core;
    (apply_jdot_mb || ctrl.keep_mb_on, qconv_env)
}

/// Evaluates magnetic braking orbital angular momentum loss $\dot{J}_{\mathrm{mb}}$
/// ($\mathrm{g\,cm^2\,s^{-2}}$) following Rappaport, Verbunt & Joss (1983, ApJ
/// 275, 713) with optional Podsiadlowski et al. (2002, ApJ 565, 1107) low
/// convective-envelope damping (`default_jdot_mb`, `binary_jdot.f90:247-328`).
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
    eval_jdot_mb_with_scale_mass(
        m_star,
        r_star,
        rl_star,
        period,
        gamma_mb,
        q_conv_env,
        scale_for_low_qconv_env,
        0.02,
    )
}

/// Evaluates `default_jdot_mb` (`binary_jdot.f90:247-328`) honoring a configurable
/// low convective-envelope damping scale `jdot_mb_mass_frac_for_scale`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_jdot_mb_with_scale_mass(
    m_star: f64,
    r_star: f64,
    rl_star: f64,
    period: f64,
    gamma_mb: f64,
    q_conv_env: f64,
    scale_for_low_qconv_env: bool,
    mass_frac_for_scale: f64,
) -> f64 {
    let rsun4 = pow4(RSUN);
    let omega = 2.0 * PI / period;
    let two_pi_div_p3 = omega * omega * omega;
    let jdot_scale = if scale_for_low_qconv_env {
        if q_conv_env > mass_frac_for_scale {
            1.0
        } else {
            exp(-mass_frac_for_scale / q_conv_env.max(1.0e-99) + 1.0)
        }
    } else {
        1.0
    };
    -3.8e-30
        * m_star
        * rsun4
        * pow(r_star.min(rl_star) / RSUN, gamma_mb)
        * two_pi_div_p3
        * jdot_scale
}

/// Evaluates spin-orbit coupling torque $\dot{J}_{\mathrm{ls}}$ ($\mathrm{g\,cm^2\,s^{-2}}$)
/// conserving total system angular momentum against non-wind stellar spin changes
/// (`default_jdot_ls` in `binary/private/binary_jdot.f90:136-174`).
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
/// from each component's spin angular momentum change (`binary_jdot.f90:152-173`).
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

/// Evaluates `default_jdot_ls` (`binary/private/binary_jdot.f90:136-174`) using
/// exact `angular_momentum_removed` ($\ge 0$, $\mathrm{g\,cm^2\,s^{-1}}$) and
/// returning `0.0` on `doing_first_model_of_run`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_jdot_ls_with_am_removed(
    doing_first_model_of_run: bool,
    point_mass_i: usize,
    model_twins_flag: bool,
    s_donor_old: f64,
    s_donor: f64,
    mstar_dot_donor: f64,
    am_removed_donor: f64,
    mdot_system_wind_donor: f64,
    s_accretor_old: f64,
    s_accretor: f64,
    mstar_dot_accretor: f64,
    am_removed_accretor: f64,
    dt: f64,
) -> f64 {
    if doing_first_model_of_run {
        return 0.0;
    }
    let mut delta_j = s_donor_old - s_donor;
    if mstar_dot_donor < 0.0 {
        delta_j -=
            am_removed_donor * (mdot_system_wind_donor / mstar_dot_donor).abs();
    }
    let mut jdot_ls = delta_j;
    if point_mass_i == 0 {
        let mut dj_a = s_accretor_old - s_accretor;
        if mstar_dot_accretor < 0.0 {
            dj_a -= am_removed_accretor;
        }
        jdot_ls += dj_a;
    } else if model_twins_flag {
        jdot_ls += jdot_ls;
    }
    jdot_ls / dt.max(1.0e-99)
}

/// Evaluates the missing wind spin angular momentum torque on the accretor
/// $\dot{J}_{\mathrm{missing\_wind}}$ ($\mathrm{g\,cm^2\,s^{-2}}$)
/// (`default_jdot_missing_wind` in `binary/private/binary_jdot.f90:176-199`).
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

// Note (`#175`): Upstream `get_edot` (`binary/private/binary_edot.f90:33-99`)
// sums `edot_tidal + edot_enhance + extra_edot` and omits a Peters (1964)
// gravitational-wave `edot_gr` term; `eval_edot_gr` was removed under `#160`
// (`DECISIONS #36`) to reproduce MESA `26.4.1` unconditionally.

/// Hut (1981, Eq. 11) polynomial $f_2(e) = 1 + \frac{15}{2}e^2 + \frac{45}{8}e^4 + \frac{5}{16}e^6$.
#[inline]
#[must_use]
pub fn hut_f2(e: f64) -> f64 {
    1.0 + 15.0 / 2.0 * pow2(e) + 45.0 / 8.0 * pow4(e) + 5.0 / 16.0 * pow6(e)
}

/// Hut (1981, Eq. 11) polynomial $f_3(e) = 1 + \frac{15}{4}e^2 + \frac{15}{8}e^4 + \frac{5}{64}e^6$.
#[inline]
#[must_use]
pub fn hut_f3(e: f64) -> f64 {
    1.0 + 15.0 / 4.0 * pow2(e) + 15.0 / 8.0 * pow4(e) + 5.0 / 64.0 * pow6(e)
}

/// Hut (1981, Eq. 11) polynomial $f_4(e) = 1 + \frac{3}{2}e^2 + \frac{1}{8}e^4$.
#[inline]
#[must_use]
pub fn hut_f4(e: f64) -> f64 {
    1.0 + 3.0 / 2.0 * pow2(e) + 1.0 / 8.0 * pow4(e)
}

/// Hut (1981, Eq. 11) polynomial $f_5(e) = 1 + 3e^2 + \frac{3}{8}e^4$.
#[inline]
#[must_use]
pub fn hut_f5(e: f64) -> f64 {
    1.0 + 3.0 * pow2(e) + 3.0 / 8.0 * pow4(e)
}

/// Evaluates the convective-envelope tidal dissipation ratio $(k/T)_{\mathrm{conv}}$
/// ($\mathrm{s}^{-1}$) following Hurley, Tout & Pols (2002, Eqs. 30-31;
/// `k_div_T` in `binary/private/binary_tides.f90:417-437`).
#[allow(clippy::too_many_arguments)]
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
    let inner = m_env_msun * r_env_rsun * (r_phot_rsun - r_env_rsun / 2.0)
        / 3.0
        / l_phot_lsun.max(1.0e-30);
    if inner <= 0.0 {
        return 0.0;
    }
    let tau_conv = 0.431 * pow(inner, ONE_THIRD) * SECYER;
    let f_conv = if 1.0 / period_sec == omega_surf / (2.0 * PI) {
        1.0
    } else {
        let p_tid = 1.0 / (1.0 / period_sec - omega_surf / (2.0 * PI)).abs();
        1.0_f64.min(pow(p_tid / (2.0 * tau_conv), tidal_reduction))
    };
    2.0 / 21.0 * f_conv / tau_conv * m_env_msun / m_star_msun.max(1.0e-30)
}

/// Evaluates the radiative-envelope dynamical tide coupling ratio $(k/T)_{\mathrm{rad}}$
/// ($\mathrm{s}^{-1}$) following Zahn (1975, 1977) and Hurley, Tout & Pols (2002 Eq. 42;
/// `k_div_T` in `binary/private/binary_tides.f90:438-456`).
#[must_use]
pub fn eval_k_div_t_radiative(
    m_star: f64,
    m_companion: f64,
    r_phot: f64,
    separation: f64,
) -> f64 {
    let q = m_companion / m_star.max(1.0e-30);
    let mut k_div_t = 1.9782e4
        * (m_star * r_phot * r_phot / pow5(separation) / (MSUN / pow3(RSUN)))
            .sqrt();
    k_div_t *= pow(1.0 + q, 5.0 / 6.0);
    let e2 = 1.592e-9 * pow(m_star / MSUN, 2.84);
    k_div_t * e2 / SECYER
}

/// Evaluates the Hut (1981, Eq. 11) spin synchronization timescale $t_{\mathrm{sync}}$ (s)
/// (`get_tsync` in `binary/private/binary_tides.f90:360-385`).
#[must_use]
pub fn eval_tsync(
    k_div_t: f64,
    q_companion_over_star: f64,
    r_gyr_sq: f64,
    r_phot: f64,
    separation: f64,
    f_tid: f64,
) -> f64 {
    let inv_tsync = 3.0
        * k_div_t
        * (q_companion_over_star * q_companion_over_star
            / r_gyr_sq.max(1.0e-99))
        * pow6(r_phot / separation.max(1.0));
    (1.0 / inv_tsync.max(1.0e-99)) / f_tid
}

/// Evaluates the Hut (1981, Eq. 10) / Hurley et al. (2002, Eq. 29) tidal
/// circularization rate $\dot{e}_{\mathrm{tidal},s}$ ($\mathrm{s}^{-1}$)
/// (`edot_tidal_Hut` in `binary/private/binary_edot.f90:146-182`).
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
    let qratio = m_companion / m_star.max(1.0e-30);
    let omega_sync = 2.0 * PI / period;
    let one_m_e2 = 1.0 - pow2(eccentricity);
    let mut edot_tidal = -27.0
        * qratio
        * (1.0 + qratio)
        * pow8(r_phot / separation)
        * eccentricity
        * pow(one_m_e2, -6.5)
        * f_tid;
    edot_tidal *= k_div_t;
    edot_tidal
        * (hut_f3(eccentricity)
            - 11.0 / 18.0 * omega_star / omega_sync
                * hut_f4(eccentricity)
                * pow(one_m_e2, 1.5))
}
