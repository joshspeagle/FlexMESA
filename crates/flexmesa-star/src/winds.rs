// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stellar wind mass-loss prescriptions (`Reimers`, `Blocker`, `Vink`,
//! `Nieuwenhuijzen`, `DeJager`, `VanLoon`, `Bjorklund`, `Dutch`), super-Eddington
//! winds, Heger et al. (2000) rotational $\Omega/\Omega_{\mathrm{crit}}$ wind
//! enhancement, and control-driven `do_set_mdot` (`star/private/winds.f90`).

use flexmesa_const::{CLIGHT, FOUR_PI, L_SUN, M_SUN, R_SUN};

pub use crate::adjust_mass::{
    build_accreted_xa, compute_delta_m, do_adjust_mass, do_adjust_mass_step,
    do_adjust_mass_with_dt, get_xa_for_accretion,
};
pub use crate::winds_driver::{
    do_set_mdot, do_set_mdot_with_extras, eval_rlo_wind,
    eval_super_eddington_wind, set_mdot,
};

/// Stellar wind mass-loss prescription (`cool_wind_RGB_scheme`,
/// `cool_wind_AGB_scheme`, and `hot_wind_scheme` in `winds.f90`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum WindScheme {
    /// Zero wind mass loss (`""`).
    None,
    /// Reimers (1975) red-giant wind (`'Reimers'`).
    Reimers,
    /// Blöcker (1995) AGB superwind (`'Blocker'`).
    Blocker,
    /// Vink, de Koter & Lamers (2001) OB-star line-driven wind (`'Vink'`).
    Vink,
    /// Nieuwenhuijzen & de Jager (1990) empirical wind (`'Nieuwenhuijzen'`).
    Nieuwenhuijzen,
    /// de Jager, Nieuwenhuijzen & van der Hucht (1988) cool wind (`'de Jager'`).
    DeJager,
    /// van Loon et al. (2005) dust-driven supergiant/AGB wind (`'van Loon'`).
    VanLoon,
    /// Björklund et al. (2023) radiation-driven O-star wind (`'Bjorklund'`).
    Bjorklund,
    /// Composite Dutch wind (`'Dutch'`: Vink / Nugis-Lamers WR at high
    /// $T_{\mathrm{eff}}$ blended with cool wind below $10^4\,\mathrm{K}$).
    Dutch,
    /// User-supplied `other_wind` hook (`'other'`, `winds.f90:319`).
    Other,
}

impl WindScheme {
    /// Parses a MESA wind scheme name (case-insensitive).
    #[must_use]
    pub fn parse_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "" => Some(Self::None),
            "other" => Some(Self::Other),
            "reimers" => Some(Self::Reimers),
            "blocker" => Some(Self::Blocker),
            "vink" => Some(Self::Vink),
            "nieuwenhuijzen" => Some(Self::Nieuwenhuijzen),
            "de jager" | "de_jager" | "dejager" => Some(Self::DeJager),
            "van loon" | "van_loon" | "vanloon" => Some(Self::VanLoon),
            "bjorklund" => Some(Self::Bjorklund),
            "dutch" => Some(Self::Dutch),
            _ => None,
        }
    }
}

/// Low-$T_{\mathrm{eff}}$ prescription inside the composite [`WindScheme::Dutch`]
/// scheme (`Dutch_wind_lowT_scheme` in `winds.f90:576-593`).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum DutchLowTScheme {
    /// de Jager et al. (1988) cool wind (`'de Jager'`, default).
    #[default]
    DeJager,
    /// van Loon et al. (2005) dust-driven wind (`'van Loon'`).
    VanLoon,
    /// Nieuwenhuijzen & de Jager (1990) wind (`'Nieuwenhuijzen'`).
    Nieuwenhuijzen,
}

impl DutchLowTScheme {
    /// Parses `Dutch_wind_lowT_scheme` (case-insensitive).
    #[must_use]
    pub fn parse_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "de jager" | "de_jager" | "dejager" => Some(Self::DeJager),
            "van loon" | "van_loon" | "vanloon" => Some(Self::VanLoon),
            "nieuwenhuijzen" => Some(Self::Nieuwenhuijzen),
            _ => None,
        }
    }
}

/// Evaluates the Reimers (1975) wind mass-loss rate $w$ in
/// $M_\odot\,\mathrm{yr}^{-1}$ (`winds.f90:323, 363-364`).
#[must_use]
#[inline]
pub fn eval_reimers_wind(
    l_surf: f64,
    m_surf: f64,
    r_surf: f64,
    eta: f64,
) -> f64 {
    if eta <= 0.0 || l_surf <= 0.0 || m_surf <= 0.0 || r_surf <= 0.0 {
        return 0.0;
    }
    (4.0e-13 * (l_surf * r_surf / m_surf) / (L_SUN * R_SUN / M_SUN)) * eta
}

/// Evaluates the Blöcker (1995) wind mass-loss rate $w$ in
/// $M_\odot\,\mathrm{yr}^{-1}$ (`eval_blocker_wind` in `winds.f90:556-561`).
#[must_use]
#[inline]
pub fn eval_blocker_wind(
    l_surf: f64,
    m_surf: f64,
    r_surf: f64,
    eta: f64,
) -> f64 {
    if eta <= 0.0 || l_surf <= 0.0 || m_surf <= 0.0 || r_surf <= 0.0 {
        return 0.0;
    }
    let w_reimers_1 = eval_reimers_wind(l_surf, m_surf, r_surf, 1.0);
    w_reimers_1
        * eta
        * 4.83e-9
        * flexmesa_math::pow(m_surf / M_SUN, -2.1)
        * flexmesa_math::pow(l_surf / L_SUN, 2.7)
}

/// Evaluates the Vink, de Koter & Lamers (2001) line-driven hot-star wind rate
/// $w$ in $M_\odot\,\mathrm{yr}^{-1}$ (`eval_Vink_wind` in `winds.f90:497-553`).
#[must_use]
pub fn eval_vink_wind(
    l_surf: f64,
    m_surf: f64,
    t_surf: f64,
    z_surf: f64,
    eta: f64,
) -> f64 {
    if eta <= 0.0
        || l_surf <= 0.0
        || m_surf <= 0.0
        || t_surf <= 0.0
        || z_surf <= 0.0
    {
        return 0.0;
    }
    let z_solar = 0.019;
    let z_rat = (z_surf / z_solar).max(1.0e-10);
    let log_z_rat = flexmesa_math::log10(z_rat);

    let d_t = 100.0;
    let t_jump = 1.0e3 * (61.2 + 2.59 * (-13.636 + 0.889 * log_z_rat));
    let alpha = if t_surf > 27_500.0 {
        1.0
    } else if t_surf < 22_500.0 {
        0.0
    } else {
        ((t_surf - (t_jump - d_t)) / (2.0 * d_t)).clamp(0.0, 1.0)
    };

    let log_l_1e5 = flexmesa_math::log10(l_surf / L_SUN / 1.0e5);
    let log_m_30 = flexmesa_math::log10(m_surf / M_SUN / 30.0);

    let w1 = if alpha > 0.0 {
        let vinf_hot = 2.6 * flexmesa_math::pow(z_rat, 0.13);
        let log_t_4e4 = flexmesa_math::log10(t_surf / 4.0e4);
        let log_w1 = -6.697 + 2.194 * log_l_1e5
            - 1.313 * log_m_30
            - 1.226 * flexmesa_math::log10(vinf_hot / 2.0)
            + 0.933 * log_t_4e4
            - 10.92 * log_t_4e4 * log_t_4e4
            + 0.85 * log_z_rat;
        flexmesa_math::exp10(log_w1)
    } else {
        0.0
    };

    let w2 = if alpha < 1.0 {
        let vinf_cool = 1.3 * flexmesa_math::pow(z_rat, 0.13);
        let log_t_2e4 = flexmesa_math::log10(t_surf / 2.0e4);
        let log_w2 = -6.688 + 2.210 * log_l_1e5
            - 1.339 * log_m_30
            - 1.601 * flexmesa_math::log10(vinf_cool / 2.0)
            + 1.07 * log_t_2e4
            + 0.85 * log_z_rat;
        flexmesa_math::exp10(log_w2)
    } else {
        0.0
    };

    (alpha * w1 + (1.0 - alpha) * w2) * eta
}

/// Evaluates the Björklund et al. (2023) radiation-driven O-star wind rate $w$
/// in $M_\odot\,\mathrm{yr}^{-1}$ (`eval_Bjorklund_wind` in `winds.f90:479-495`).
#[must_use]
pub fn eval_bjorklund_wind(
    l_surf: f64,
    m_surf: f64,
    t_surf: f64,
    z_surf: f64,
    cgrav_surf: f64,
    eta: f64,
) -> f64 {
    if eta <= 0.0
        || l_surf <= 0.0
        || m_surf <= 0.0
        || t_surf <= 0.0
        || z_surf <= 0.0
        || cgrav_surf <= 0.0
    {
        return 0.0;
    }
    let z_bjork = 0.014;
    let gamma_e = 0.34 * l_surf / (FOUR_PI * CLIGHT * cgrav_surf * m_surf);
    let m_eff = (m_surf * (1.0 - gamma_e)).max(1.0e-6 * m_surf);
    let log_t_45k = flexmesa_math::log10(t_surf / 4.5e4);
    let log_w = -5.52 + 2.39 * flexmesa_math::log10(l_surf / (1.0e6 * L_SUN))
        - 1.48 * flexmesa_math::log10(m_eff / (45.0 * M_SUN))
        + 2.12 * log_t_45k
        + (0.75 - 1.87 * log_t_45k)
            * flexmesa_math::log10((z_surf / z_bjork).max(1.0e-10));
    eta * flexmesa_math::exp10(log_w)
}

/// Evaluates the Nieuwenhuijzen & de Jager (1990) wind rate $w$ in
/// $M_\odot\,\mathrm{yr}^{-1}$ (`eval_Nieuwenhuijzen_wind` in
/// `winds.f90:619-632`).
#[must_use]
#[inline]
pub fn eval_nieuwenhuijzen_wind(
    l_surf: f64,
    m_surf: f64,
    r_surf: f64,
    eta: f64,
) -> f64 {
    if eta <= 0.0 || l_surf <= 0.0 || m_surf <= 0.0 || r_surf <= 0.0 {
        return 0.0;
    }
    let log_w = -14.02
        + 1.24 * flexmesa_math::log10(l_surf / L_SUN)
        + 0.16 * flexmesa_math::log10(m_surf / M_SUN)
        + 0.81 * flexmesa_math::log10(r_surf / R_SUN);
    eta * flexmesa_math::exp10(log_w)
}

/// Evaluates the de Jager et al. (1988) wind rate $w$ in
/// $M_\odot\,\mathrm{yr}^{-1}$ (`eval_de_Jager_wind` in `winds.f90:596-606`).
#[must_use]
#[inline]
pub fn eval_de_jager_wind(l_surf: f64, t_surf: f64, eta: f64) -> f64 {
    if eta <= 0.0 || l_surf <= 0.0 || t_surf <= 0.0 {
        return 0.0;
    }
    let log_w = 1.769 * flexmesa_math::log10(l_surf / L_SUN)
        - 1.676 * flexmesa_math::log10(t_surf)
        - 8.158;
    eta * flexmesa_math::exp10(log_w)
}

/// Evaluates the van Loon et al. (2005) wind rate $w$ in
/// $M_\odot\,\mathrm{yr}^{-1}$ (`eval_van_Loon_wind` in `winds.f90:609-616`).
#[must_use]
#[inline]
pub fn eval_van_loon_wind(l_surf: f64, t_surf: f64, eta: f64) -> f64 {
    if eta <= 0.0 || l_surf <= 0.0 || t_surf <= 0.0 {
        return 0.0;
    }
    let log_w = -5.65 + 1.05 * flexmesa_math::log10(l_surf / (1.0e4 * L_SUN))
        - 6.3 * flexmesa_math::log10(t_surf / 3.5e3);
    eta * flexmesa_math::exp10(log_w)
}

/// Evaluates the Nugis & Lamers (2000) Wolf-Rayet wind mass-loss rate $w$ in
/// $M_\odot\,\mathrm{yr}^{-1}$ (`eval_highT_Dutch` in `winds.f90:567-570`):
/// $$w_{\mathrm{NL}} = \eta \times 10^{-11}\,(L/L_\odot)^{1.29}\,Y^{1.7}\,\sqrt{Z}.$$
#[must_use]
#[inline]
pub fn eval_nugis_lamers_wind(
    l_surf: f64,
    y_surf: f64,
    z_surf: f64,
    eta: f64,
) -> f64 {
    if eta <= 0.0 || l_surf <= 0.0 || y_surf <= 0.0 || z_surf <= 0.0 {
        return 0.0;
    }
    eta * 1.0e-11
        * flexmesa_math::pow(l_surf / L_SUN, 1.29)
        * flexmesa_math::pow(y_surf, 1.7)
        * z_surf.sqrt()
}

/// Evaluates the composite Dutch wind rate $w$ in $M_\odot\,\mathrm{yr}^{-1}$
/// (`eval_Dutch_wind` and `eval_highT_Dutch` in `winds.f90:347-363, 564-593`).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn eval_dutch_wind(
    l_surf: f64,
    m_surf: f64,
    r_surf: f64,
    t_surf: f64,
    x_surf: f64,
    y_surf: f64,
    z_surf: f64,
    low_t_scheme: DutchLowTScheme,
    eta: f64,
) -> f64 {
    if eta == 0.0 || l_surf <= 0.0 || m_surf <= 0.0 || t_surf <= 0.0 {
        return 0.0;
    }
    let eval_high = || {
        if x_surf < 0.4 {
            eval_nugis_lamers_wind(l_surf, y_surf, z_surf, 1.0)
        } else {
            eval_vink_wind(l_surf, m_surf, t_surf, z_surf, 1.0)
        }
    };
    let eval_low = || match low_t_scheme {
        DutchLowTScheme::DeJager => eval_de_jager_wind(l_surf, t_surf, 1.0),
        DutchLowTScheme::VanLoon => eval_van_loon_wind(l_surf, t_surf, 1.0),
        DutchLowTScheme::Nieuwenhuijzen => {
            eval_nieuwenhuijzen_wind(l_surf, m_surf, r_surf, 1.0)
        }
    };
    let (t_low, t_high) = (10_000.0, 11_000.0);
    let w = if t_surf >= t_high {
        eval_high()
    } else if t_surf <= t_low {
        eval_low()
    } else {
        let w_low = eval_low();
        let w_high = eval_high();
        let alfa = (t_surf - t_low) / (t_high - t_low);
        (1.0 - alfa) * w_low + alfa * w_high
    };
    eta * w
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_scheme_rate(
    scheme: WindScheme,
    dutch_low_t: DutchLowTScheme,
    l1: f64,
    m1: f64,
    r1: f64,
    t1: f64,
    x: f64,
    y: f64,
    z: f64,
    cgrav: f64,
    eta: f64,
) -> f64 {
    match scheme {
        WindScheme::None | WindScheme::Other => 0.0,
        WindScheme::Reimers => eval_reimers_wind(l1, m1, r1, eta),
        WindScheme::Blocker => eval_blocker_wind(l1, m1, r1, eta),
        WindScheme::Vink => eval_vink_wind(l1, m1, t1, z, eta),
        WindScheme::Nieuwenhuijzen => eval_nieuwenhuijzen_wind(l1, m1, r1, eta),
        WindScheme::DeJager => eval_de_jager_wind(l1, t1, eta),
        WindScheme::VanLoon => eval_van_loon_wind(l1, t1, eta),
        WindScheme::Bjorklund => eval_bjorklund_wind(l1, m1, t1, z, cgrav, eta),
        WindScheme::Dutch => {
            eval_dutch_wind(l1, m1, r1, t1, x, y, z, dutch_low_t, eta)
        }
    }
}
