// SPDX-License-Identifier: LGPL-3.0-or-later
//! Binary timestep controller (`binary_pick_next_timestep`).
//!
//! Corresponds to `binary/private/binary_timestep.f90:99-322`.

use flexmesa_const::{MSUN, RSUN, SECYER};
use flexmesa_math::{exp10, log10};

/// Timestep limit reason codes matching `binary/public/binary_def.f90:39-46`.
pub const B_TLIM_COMP: i32 = 1;
/// Roche-lobe gap change limit (`fr`).
pub const B_TLIM_ROCHE: i32 = 2;
/// Orbital angular momentum change limit (`fj`).
pub const B_TLIM_JORB: i32 = 3;
/// Donor envelope mass change limit (`fm`).
pub const B_TLIM_ENV: i32 = 4;
/// Orbital separation change limit (`fa`).
pub const B_TLIM_SEP: i32 = 5;
/// Eccentricity change limit (`fe`).
pub const B_TLIM_ECC: i32 = 6;
/// Component mass change limit (`fdm`).
pub const B_TLIM_DM: i32 = 7;

/// State and control parameters for `binary_pick_next_timestep`
/// (`binary/private/binary_timestep.f90:99-322`).
#[derive(Debug, Clone)]
pub struct BinaryTimestepControls {
    /// Completed donor step duration $\Delta t$ (yr) (`s%time_step`).
    pub donor_time_step_yr: f64,
    /// Previous maximum binary timestep (s) (`b%max_timestep`).
    pub max_timestep_sec: f64,
    /// Global timestep scaling factor (`b%time_delta_coeff`, default `1.0`).
    pub time_delta_coeff: f64,
    /// Log-space timestep softening weight (`b%dt_softening_factor`, default `0.5`).
    pub dt_softening_factor: f64,
    /// Relative Roche-lobe gap `(r - rl)/rl` for `[donor, accretor]`.
    pub rl_relative_gap: [f64; 2],
    /// Previous step relative Roche-lobe gap for `[donor, accretor]`.
    pub rl_relative_gap_old: [f64; 2],
    /// Current orbital angular momentum $J_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-1}}$).
    pub angular_momentum_j: f64,
    /// Previous orbital angular momentum $J_{\mathrm{orb,old}}$ ($\mathrm{g\,cm^2\,s^{-1}}$).
    pub angular_momentum_j_old: f64,
    /// Current orbital separation $a$ (cm).
    pub separation: f64,
    /// Previous orbital separation $a_{\mathrm{old}}$ (cm).
    pub separation_old: f64,
    /// Current eccentricity $e$.
    pub eccentricity: f64,
    /// Previous eccentricity $e_{\mathrm{old}}$.
    pub eccentricity_old: f64,
    /// Current donor envelope mass $M_{\mathrm{env},d}$ ($M_\odot$).
    pub env_donor: f64,
    /// Previous donor envelope mass $M_{\mathrm{env},d,\mathrm{old}}$ ($M_\odot$).
    pub env_donor_old: f64,
    /// Current component masses `[m_donor, m_accretor]` (g).
    pub m: [f64; 2],
    /// Previous component masses `[m_donor_old, m_accretor_old]` (g).
    pub m_old: [f64; 2],
    /// Whether the companion is a point mass (`point_mass_i != 0`).
    pub is_point_mass_companion: bool,
    /// Relative change limits (`fm, fr, fj, fa, fe, fdm`; negative disables).
    pub fm: f64,
    /// Floor on donor envelope mass in `fm` (`fm_limit`, default `1e-3`).
    pub fm_limit: f64,
    /// Relative Roche-lobe gap change limit (`fr`, default `0.10`).
    pub fr: f64,
    /// Floor on $|g_i|$ in `fr` (`fr_limit`, default `1e-2`).
    pub fr_limit: f64,
    /// Minimum timestep floor for `fr` in years (`fr_dt_limit`, default `10.0`).
    pub fr_dt_limit: f64,
    /// Relative orbital angular momentum change limit (`fj`, default `0.001`).
    pub fj: f64,
    /// Relative orbital separation change limit (`fa`, default `0.01`).
    pub fa: f64,
    /// Relative eccentricity change limit (`fe`, default `0.01`).
    pub fe: f64,
    /// Floor on $e$ in `fe` (`fe_limit`, default `0.1`).
    pub fe_limit: f64,
    /// Relative component mass change limit (`fdm`, default `0.005`).
    pub fdm: f64,
}

impl Default for BinaryTimestepControls {
    fn default() -> Self {
        Self {
            donor_time_step_yr: 10.0,
            max_timestep_sec: 1.0e99,
            time_delta_coeff: 1.0,
            dt_softening_factor: 0.5,
            rl_relative_gap: [-0.1, 0.0],
            rl_relative_gap_old: [-0.1, 0.0],
            angular_momentum_j: 1.0e52,
            angular_momentum_j_old: 1.0e52,
            separation: 10.0 * RSUN,
            separation_old: 10.0 * RSUN,
            eccentricity: 0.0,
            eccentricity_old: 0.0,
            env_donor: 0.5,
            env_donor_old: 0.5,
            m: [MSUN, MSUN],
            m_old: [MSUN, MSUN],
            is_point_mass_companion: true,
            fm: 0.01,
            fm_limit: 1.0e-3,
            fr: 0.10,
            fr_limit: 1.0e-2,
            fr_dt_limit: 10.0,
            fj: 0.001,
            fa: 0.01,
            fe: 0.01,
            fe_limit: 1.0e-1,
            fdm: 0.005,
        }
    }
}

/// Evaluates the next maximum binary timestep `max_timestep` (s) and limiting
/// reason code (`B_TLIM_*`), matching `binary_pick_next_timestep` in
/// `binary/private/binary_timestep.f90:99-322` and tested in
/// `binary/test/src/test_binary.f90:25-80`.
#[must_use]
pub fn binary_pick_next_timestep(ctrl: &BinaryTimestepControls) -> (f64, i32) {
    let dt_yr = ctrl.donor_time_step_yr;
    let c_dt = ctrl.time_delta_coeff;
    let limit_dt =
        |rel: f64, f_lim: f64| dt_yr / (rel / (f_lim * c_dt) + 1.0e-99);

    let mut dtm = 1.0e99_f64;
    let mut dtr = 1.0e99_f64;
    let mut dtj = 1.0e99_f64;
    let mut dta = 1.0e99_f64;
    let mut dte = 1.0e99_f64;
    let mut dtdm = 1.0e99_f64;

    if ctrl.fj > 0.0
        && ctrl.angular_momentum_j > 0.0
        && ctrl.angular_momentum_j_old != 0.0
    {
        let rj = (ctrl.angular_momentum_j - ctrl.angular_momentum_j_old).abs()
            / ctrl.angular_momentum_j;
        dtj = limit_dt(rj, ctrl.fj);
    }

    if ctrl.fm > 0.0 && ctrl.env_donor_old != 0.0 {
        let rm = (ctrl.env_donor - ctrl.env_donor_old).abs()
            / ctrl.env_donor.max(ctrl.fm_limit);
        dtm = limit_dt(rm, ctrl.fm);
    }

    if ctrl.fr > 0.0 {
        let rr_d = if ctrl.rl_relative_gap_old[0] == 0.0 {
            0.0
        } else {
            (ctrl.rl_relative_gap_old[0] - ctrl.rl_relative_gap[0]).abs()
                / ctrl.rl_relative_gap[0].abs().max(ctrl.fr_limit)
        };
        let dtr_d = limit_dt(rr_d, ctrl.fr);
        dtr = if ctrl.is_point_mass_companion {
            dtr_d
        } else {
            let rr_a = if ctrl.rl_relative_gap_old[1] == 0.0 {
                0.0
            } else {
                (ctrl.rl_relative_gap_old[1] - ctrl.rl_relative_gap[1]).abs()
                    / ctrl.rl_relative_gap[1].abs().max(ctrl.fr_limit)
            };
            dtr_d.min(limit_dt(rr_a, ctrl.fr))
        };
        if dtr < ctrl.fr_dt_limit {
            dtr = ctrl.fr_dt_limit;
        }
    }

    if ctrl.fa > 0.0 && ctrl.separation > 0.0 && ctrl.separation_old != 0.0 {
        let ra =
            (ctrl.separation_old - ctrl.separation).abs() / ctrl.separation;
        dta = limit_dt(ra, ctrl.fa);
    }

    if ctrl.fe > 0.0 && ctrl.eccentricity_old != 0.0 {
        let re = (ctrl.eccentricity_old - ctrl.eccentricity).abs()
            / ctrl.eccentricity.max(ctrl.fe_limit);
        dte = limit_dt(re, ctrl.fe);
    }

    if ctrl.fdm > 0.0 && ctrl.m_old[0] > 0.0 {
        let rdm_d = (ctrl.m[0] - ctrl.m_old[0]).abs() / ctrl.m_old[0];
        let dtdm_d = limit_dt(rdm_d, ctrl.fdm);
        dtdm = if ctrl.is_point_mass_companion || ctrl.m_old[1] <= 0.0 {
            dtdm_d
        } else {
            let rdm_a = (ctrl.m[1] - ctrl.m_old[1]).abs() / ctrl.m_old[1];
            dtdm_d.min(limit_dt(rdm_a, ctrl.fdm))
        };
    }

    let candidates = [
        (dtm, B_TLIM_ENV),
        (dtr, B_TLIM_ROCHE),
        (dtj, B_TLIM_JORB),
        (dta, B_TLIM_SEP),
        (dte, B_TLIM_ECC),
        (dtdm, B_TLIM_DM),
    ];
    let mut dt_set_yr = 1.0e99_f64;
    let mut reason = B_TLIM_COMP;
    for (val, code) in candidates {
        if val < dt_set_yr {
            dt_set_yr = val;
            reason = code;
        }
    }
    if dt_set_yr < 1.0e-7 {
        dt_set_yr = 1.0e-7;
    }

    let f_soft = ctrl.dt_softening_factor;
    let max_timestep_new = exp10(
        f_soft * log10(ctrl.max_timestep_sec)
            + (1.0 - f_soft) * log10(dt_set_yr * SECYER),
    );
    (max_timestep_new, reason)
}
