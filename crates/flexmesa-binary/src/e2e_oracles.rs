// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stage `E2E-5` (`#96`) `binary/test_suite` case registry, `&binary_job` /
//! `&binary_controls` inlist parser, and exact analytical orbital-evolution
//! and wind-accretion physics oracles (`Peters 1964`, `Tauris & van den Heuvel
//! 2003`, `Brott et al. 2011`).

use std::path::Path;

use flexmesa_const::{CLIGHT, MSUN, RSUN, STANDARD_CGRAV};
use flexmesa_star::{
    eval_dutch_wind, eval_nieuwenhuijzen_wind, eval_vink_wind, InlistBundle,
    StarError,
};

/// All 8 active `binary/test_suite` cases in Stage `E2E-5` (`#96`).
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

/// Returns `true` if `case_name` belongs to `TIER5_BINARY_CASES`.
#[must_use]
pub fn is_tier5_binary_case(case_name: &str) -> bool {
    TIER5_BINARY_CASES.contains(&case_name)
}

/// Parsed `&binary_job` and `&binary_controls` parameters for a binary test case.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct BinaryInlistControls {
    /// Entry inlist filename for star 1 (`inlist_names(1)`).
    pub inlist1: String,
    /// Entry inlist filename for star 2 (`inlist_names(2)`).
    pub inlist2: String,
    /// Whether both stars are actively evolved (`evolve_both_stars`).
    pub evolve_both_stars: bool,
    /// Primary initial mass $M_1$ ($M_\odot$).
    pub m1_msun: f64,
    /// Secondary initial mass $M_2$ ($M_\odot$).
    pub m2_msun: f64,
    /// Initial orbital period $P_0$ (days).
    pub initial_period_days: f64,
    /// Initial orbital eccentricity $e_0 \in [0, 1)$.
    pub initial_eccentricity: f64,
    /// Roche-lobe overflow scheme (`"Ritter"`, `"Kolb"`, `"roche_lobe"`, `"contact"`).
    pub mdot_scheme: String,
    /// Non-conservative mass-loss fraction from donor ($\alpha_{\mathrm{mt}}$).
    pub mass_transfer_alpha: f64,
    /// Non-conservative mass-loss fraction from accretor ($\beta_{\mathrm{mt}}$).
    pub mass_transfer_beta: f64,
    /// Non-conservative mass-loss fraction from circumbinary toroid ($\delta_{\mathrm{mt}}$).
    pub mass_transfer_delta: f64,
    /// Circumbinary toroid dimensionless radius ($\gamma_{\mathrm{mt}}$).
    pub mass_transfer_gamma: f64,
    /// Whether gravitational-wave $\dot{J}_{\mathrm{gr}}$ is active (`do_jdot_gr`).
    pub do_jdot_gr: bool,
    /// Whether mass-loss $\dot{J}_{\mathrm{ml}}$ is active (`do_jdot_ml`).
    pub do_jdot_ml: bool,
    /// Whether spin-orbit $\dot{J}_{\mathrm{ls}}$ is active (`do_jdot_ls`).
    pub do_jdot_ls: bool,
    /// Whether magnetic braking $\dot{J}_{\mathrm{mb}}$ is active (`do_jdot_mb`).
    pub do_jdot_mb: bool,
    /// Whether tidal synchronization is active (`do_tidal_sync`).
    pub do_tidal_sync: bool,
    /// Whether Bondi-Hoyle wind mass transfer from star 1 is active (`do_wind_mass_transfer_1`).
    pub do_wind_mass_transfer_1: bool,
    /// Bondi-Hoyle wind accretion efficiency parameter (`wind_BH_beta_1`).
    pub wind_bh_beta_1: f64,
    /// Maximum wind mass-transfer fraction (`max_wind_transfer_fraction_1`).
    pub max_wind_transfer_fraction_1: f64,
    /// Whether accretion retention is capped by the Eddington limit (`limit_retention_by_mdot_edd`).
    pub limit_retention_by_mdot_edd: bool,
    /// Whether Kerr BH radiative efficiency corrects mass transfer (`use_radiation_corrected_transfer_rate`).
    pub use_radiation_corrected_transfer_rate: bool,
    /// Initial dimensionless Kerr BH spin parameter $a_{\mathrm{BH},0} \in [0, 1)$.
    pub initial_bh_spin: f64,
    /// Max implicit RLOF tries (`max_tries_to_achieve`, `<= 0` means explicit).
    pub max_tries_to_achieve: i32,
}

impl Default for BinaryInlistControls {
    fn default() -> Self {
        Self {
            inlist1: String::from("inlist1"),
            inlist2: String::from("inlist2"),
            evolve_both_stars: false,
            m1_msun: 1.0,
            m2_msun: 0.8,
            initial_period_days: 1.0,
            initial_eccentricity: 0.0,
            mdot_scheme: String::from("Ritter"),
            mass_transfer_alpha: 0.0,
            mass_transfer_beta: 0.0,
            mass_transfer_delta: 0.0,
            mass_transfer_gamma: 0.0,
            do_jdot_gr: true,
            do_jdot_ml: true,
            do_jdot_ls: true,
            do_jdot_mb: true,
            do_tidal_sync: false,
            do_wind_mass_transfer_1: false,
            wind_bh_beta_1: 0.125,
            max_wind_transfer_fraction_1: 0.5,
            limit_retention_by_mdot_edd: false,
            use_radiation_corrected_transfer_rate: false,
            initial_bh_spin: 0.0,
            max_tries_to_achieve: 20,
        }
    }
}

impl BinaryInlistControls {
    /// Parses `&binary_job` and `&binary_controls` from `case_dir` (following
    /// `inlist` -> `inlist_project`).
    ///
    /// # Errors
    /// Returns [`StarError`] if reading or parsing the binary namelist fails.
    pub fn from_case_dir(case_dir: &Path) -> Result<Self, StarError> {
        let mut out = Self::default();
        let entry = if case_dir.join("inlist").exists() {
            "inlist"
        } else {
            "inlist_project"
        };
        let bundle = InlistBundle::from_file_in_case_dir(case_dir, entry)?;
        if let Some(bjob) = bundle.sections.get("binary_job") {
            let names = bjob.get_indexed_strings("inlist_names");
            if let Some(n1) = names.first() {
                out.inlist1.clone_from(n1);
            }
            if let Some(n2) = names.get(1) {
                out.inlist2.clone_from(n2);
            }
            if let Some(b) = bjob.get_bool("evolve_both_stars") {
                out.evolve_both_stars = b;
            }
        }
        if let Some(bc) = bundle.sections.get("binary_controls") {
            if let Some(x) = bc.get_f64("m1") {
                out.m1_msun = x;
            }
            if let Some(x) = bc.get_f64("m2") {
                out.m2_msun = x;
            }
            if let Some(x) = bc.get_f64("initial_period_in_days") {
                out.initial_period_days = x;
            }
            if let Some(x) = bc.get_f64("initial_eccentricity") {
                out.initial_eccentricity = x;
            }
            if let Some(s) = bc.get_str("mdot_scheme") {
                out.mdot_scheme = s.to_string();
            }
            if let Some(x) = bc.get_f64("mass_transfer_alpha") {
                out.mass_transfer_alpha = x;
            }
            if let Some(x) = bc.get_f64("mass_transfer_beta") {
                out.mass_transfer_beta = x;
            }
            if let Some(x) = bc.get_f64("mass_transfer_delta") {
                out.mass_transfer_delta = x;
            }
            if let Some(x) = bc.get_f64("mass_transfer_gamma") {
                out.mass_transfer_gamma = x;
            }
            if let Some(b) = bc.get_bool("do_jdot_gr") {
                out.do_jdot_gr = b;
            }
            if let Some(b) = bc.get_bool("do_jdot_ml") {
                out.do_jdot_ml = b;
            }
            if let Some(b) = bc.get_bool("do_jdot_ls") {
                out.do_jdot_ls = b;
            }
            if let Some(b) = bc.get_bool("do_jdot_mb") {
                out.do_jdot_mb = b;
            }
            if let Some(b) = bc.get_bool("do_tidal_sync") {
                out.do_tidal_sync = b;
            }
            if let Some(b) = bc.get_bool("do_wind_mass_transfer_1") {
                out.do_wind_mass_transfer_1 = b;
            }
            if let Some(x) = bc.get_f64("wind_bh_beta_1") {
                out.wind_bh_beta_1 = x;
            }
            if let Some(x) = bc.get_f64("max_wind_transfer_fraction_1") {
                out.max_wind_transfer_fraction_1 = x;
            }
            if let Some(b) = bc.get_bool("limit_retention_by_mdot_edd") {
                out.limit_retention_by_mdot_edd = b;
            }
            if let Some(b) =
                bc.get_bool("use_radiation_corrected_transfer_rate")
            {
                out.use_radiation_corrected_transfer_rate = b;
            }
            if let Some(x) = bc.get_f64("initial_bh_spin") {
                out.initial_bh_spin = x;
            }
            if let Some(x) = bc.get_i64("max_tries_to_achieve") {
                out.max_tries_to_achieve = x as i32;
            }
        }
        Ok(out)
    }
}

/// Peters (1964) analytical gravitational-wave inspiral separation $a(t)$ ($R_\odot$)
/// for a circular binary of constant masses $M_1, M_2$ ($M_\odot$) starting at
/// $a_0$ ($R_\odot$) after elapsed time $t_{\mathrm{sec}}$ (`jdot_gr_check/src/run_binary_extras.f90:70-81`).
#[must_use]
pub fn eval_gr_expected_separation_rsun(
    a0_rsun: f64,
    m1_msun: f64,
    m2_msun: f64,
    time_sec: f64,
) -> f64 {
    let a0_cm = a0_rsun * RSUN;
    let m1_g = m1_msun * MSUN;
    let m2_g = m2_msun * MSUN;
    let beta_gw = 64.0 * STANDARD_CGRAV.powi(3) * m1_g * m2_g * (m1_g + m2_g)
        / (5.0 * CLIGHT.powi(5));
    let a4 = (a0_cm.powi(4) - 4.0 * beta_gw * time_sec).max(0.0);
    a4.powf(0.25) / RSUN
}

/// Tauris & van den Heuvel (2003, Eq. 16.20) analytical orbital separation
/// $a(q)$ ($R_\odot$) under non-conservative mass transfer with parameters
/// $(\alpha, \beta, \delta, \gamma)$ from initial mass ratio $q_0 = M_{1,0}/M_{2,0}$
/// to current $q = M_1/M_2$ (`jdot_ml_check/src/run_binary_extras.f90:72-95`).
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
    let eps = (1.0 - alpha - beta - delta).clamp(1.0e-12, 1.0 - 1.0e-12);
    let one_minus_eps = 1.0 - eps;
    let term1 = (q / q0.max(1.0e-12)).powf(2.0 * (alpha + gamma * delta - 1.0));
    let term2 =
        ((q + 1.0) / (q0 + 1.0)).powf((-alpha - beta + delta) / one_minus_eps);
    let exp3 = 3.0
        + 2.0
            * (alpha * eps * eps
                + beta
                + gamma * delta * one_minus_eps * one_minus_eps)
            / (eps * one_minus_eps);
    let term3 = ((eps * q + 1.0) / (eps * q0 + 1.0)).powf(exp3);
    a0_rsun * term1 * term2 * term3
}

/// Brott et al. (2011) massive-star wind mass-loss rate $\dot{M}$
/// ($M_\odot\,\mathrm{yr}^{-1}$) combining Vink et al. (2001), Nieuwenhuijzen &
/// de Jager (1990), and Hamann et al. (1995) Wolf-Rayet winds across the
/// bi-stability jump (`double_bh/src/run_star_extras.f90:96-175`).
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
    let z_solar = 0.017_f64;
    let z_eff = (z_div_z_solar * z_solar).max(1.0e-8);
    let l_cgs = l_surf_lsun * 3.828e33;
    let m_cgs = m_surf_msun * MSUN;
    let r_cgs = r_surf_rsun * RSUN;
    let t_high = 1.0e3 * (29.5 + 1.5 * z_eff.log10());
    let t_low = 1.0e3 * (22.5 + 0.6 * z_eff.log10());

    let w_vink = eval_vink_wind(l_cgs, m_cgs, t_surf_k, z_eff, 1.0).abs();
    let w_ndj = eval_nieuwenhuijzen_wind(l_cgs, m_cgs, r_cgs, 1.0).abs();
    let w_wr = 10.0_f64
        .powf(-11.95 + 1.5 * l_surf_lsun.max(1.0).log10() - 2.85 * x_surf)
        * (z_div_z_solar.max(1.0e-6)).powf(0.85)
        / 10.0;

    let w_hot_blend = if x_surf < 0.4 {
        w_wr
    } else if x_surf < 0.7 {
        let f = (0.7 - x_surf) / 0.3;
        f * w_wr + (1.0 - f) * w_vink
    } else {
        w_vink
    };

    if t_surf_k > t_high {
        w_hot_blend
    } else if t_surf_k > t_low {
        w_hot_blend.max(w_ndj)
    } else {
        let w_dutch = eval_dutch_wind(
            l_cgs,
            m_cgs,
            r_cgs,
            t_surf_k,
            x_surf,
            y_surf,
            z_eff,
            flexmesa_star::DutchLowTScheme::DeJager,
            1.0,
        )
        .abs();
        w_ndj.max(w_dutch)
    }
}
