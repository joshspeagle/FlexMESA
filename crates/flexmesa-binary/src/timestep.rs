// SPDX-License-Identifier: GPL-3.0-or-later
//! Binary timestep controller (`binary_pick_next_timestep`) and hard-limit retry
//! checks.
//!
//! Corresponds to `binary/private/binary_timestep.f90:33-322`.

use flexmesa_const::{MSUN, RSUN, SECYER};
use flexmesa_math::{exp10, log10};
use flexmesa_star::StarError;

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
/// Highest binary timestep limit code (`b_numTlim`, `binary_def.f90:46`).
pub const B_NUM_TLIM: i32 = B_TLIM_DM;

/// Returns `binary_dt_why_str(reason)` (`binary_private_def.f90:223-231`) or
/// `" "` when out of `1..=B_NUM_TLIM` (`binary_timestep.f90:64-68`).
#[must_use]
pub fn binary_dt_why_str(reason: i32) -> &'static str {
    match reason {
        B_TLIM_COMP => "b_companion",
        B_TLIM_ROCHE => "b_RL",
        B_TLIM_JORB => "b_jorb",
        B_TLIM_ENV => "b_envelope",
        B_TLIM_SEP => "b_separation",
        B_TLIM_ECC => "b_eccentricity",
        B_TLIM_DM => "b_deltam",
        _ => " ",
    }
}

/// Specific binary hard-limit violation triggered inside
/// `binary_pick_next_timestep` (`binary/private/binary_timestep.f90:168-262`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryHardLimitViolation {
    /// Orbital angular-momentum relative change exceeded `fj_hard * time_delta_coeff`.
    FjHard,
    /// Donor envelope-mass relative change exceeded `fm_hard * time_delta_coeff`.
    FmHard,
    /// Donor Roche-lobe gap relative change exceeded `fr_hard * time_delta_coeff`.
    FrHardDonor,
    /// Accretor Roche-lobe gap relative change exceeded `fr_hard * time_delta_coeff`.
    FrHardAccretor,
    /// Orbital separation relative change exceeded `fa_hard * time_delta_coeff`.
    FaHard,
    /// Eccentricity relative change exceeded `fe_hard * time_delta_coeff`.
    FeHard,
    /// Donor mass relative change exceeded `fdm_hard * time_delta_coeff`.
    FdmHardDonor,
    /// Accretor mass relative change exceeded `fdm_hard * time_delta_coeff`.
    FdmHardAccretor,
}

impl BinaryHardLimitViolation {
    /// Whether this violation sets `have_to_reduce_timestep_due_to_j = .true.`
    /// (`binary/private/binary_timestep.f90:173`).
    #[must_use]
    pub const fn reduces_timestep_due_to_j(self) -> bool {
        matches!(self, Self::FjHard)
    }
}

/// State and control parameters for `binary_pick_next_timestep`
/// (`binary/private/binary_timestep.f90:99-322`).
#[derive(Debug, Clone)]
pub struct BinaryTimestepControls {
    /// Completed donor step duration $\Delta t$ (yr) (`s%time_step`).
    pub donor_time_step_yr: f64,
    /// Completed donor step duration $\Delta t$ (s) (`s%dt`), which replaces
    /// a negative `max_timestep` (`binary_timestep.f90:121`).
    pub donor_dt_sec: f64,
    /// Previous maximum binary timestep (s) (`b%max_timestep`).
    pub max_timestep_sec: f64,
    /// Global timestep scaling factor (`b%time_delta_coeff`, default `1.0`).
    pub time_delta_coeff: f64,
    /// Log-space timestep softening weight (`b%dt_softening_factor`, default `0.5`).
    pub dt_softening_factor: f64,
    /// Relative Roche-lobe gap `(r - rl*(1-e))/(rl*(1-e))` for `[donor, accretor]`.
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
    /// Whether this is the first evolutionary step of the run (`b% doing_first_model_of_run`).
    pub doing_first_model_of_run: bool,
    /// Whether hard limits are suppressed on this step (`b% ignore_hard_limits_this_step`,
    /// set by `set_angular_momentum_j` in `binary_utils.f90:200`).
    pub ignore_hard_limits_this_step: bool,
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
            donor_dt_sec: 10.0 * SECYER,
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
            doing_first_model_of_run: false,
            ignore_hard_limits_this_step: false,
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

fn step_changes(ctrl: &BinaryTimestepControls) -> (f64, f64, f64, f64, f64) {
    if ctrl.doing_first_model_of_run {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    } else {
        let env_change = if ctrl.env_donor_old == 0.0 {
            0.0
        } else {
            ctrl.env_donor - ctrl.env_donor_old
        };
        let rel_gap_change_d = if ctrl.rl_relative_gap_old[0] == 0.0 {
            0.0
        } else {
            ctrl.rl_relative_gap_old[0] - ctrl.rl_relative_gap[0]
        };
        let j_change = if ctrl.angular_momentum_j_old == 0.0 {
            0.0
        } else {
            ctrl.angular_momentum_j - ctrl.angular_momentum_j_old
        };
        let sep_change = if ctrl.separation_old == 0.0 {
            0.0
        } else {
            ctrl.separation - ctrl.separation_old
        };
        let e_change = if ctrl.eccentricity_old == 0.0 {
            0.0
        } else {
            ctrl.eccentricity - ctrl.eccentricity_old
        };
        (env_change, rel_gap_change_d, j_change, sep_change, e_change)
    }
}

/// Evaluates the next maximum binary timestep `max_timestep` (s) and limiting
/// reason code (`B_TLIM_*`) from the relative-change limits, matching the
/// soft-limit part of `binary_pick_next_timestep` in
/// `binary/private/binary_timestep.f90:99-283` and tested in
/// `binary/test/src/test_binary.f90:25-80`. The hard limits, the
/// `varcontrol_target` update and the reset of
/// `ignore_hard_limits_this_step` are applied by the binary step loop
/// (`BinarySystem`, `#451`), in upstream order around this function.
///
/// # Errors
///
/// Returns [`StarError::ConvergenceFailure`] when no limit equals the minimum
/// (a NaN limit), where upstream stops with `mesa_error('Something wrong in
/// binary timestep')` (`binary_timestep.f90:278-280`).
pub fn binary_pick_next_timestep(
    ctrl: &BinaryTimestepControls,
) -> Result<(f64, i32), StarError> {
    let dt_yr = ctrl.donor_time_step_yr;
    let c_dt = ctrl.time_delta_coeff;
    let limit_dt =
        |rel: f64, f_lim: f64| dt_yr / (rel / (f_lim * c_dt) + 1.0e-99);

    // `binary_timestep.f90:121`: only negative `max_timestep` is replaced by `s_donor% dt`.
    let prev_max = if ctrl.max_timestep_sec < 0.0 {
        ctrl.donor_dt_sec
    } else {
        ctrl.max_timestep_sec
    };

    let (env_change, rel_gap_change_d, j_change, sep_change, e_change) =
        step_changes(ctrl);

    let mut dtm = 1.0e99_f64;
    let mut dtr = 1.0e99_f64;
    let mut dtj = 1.0e99_f64;
    let mut dta = 1.0e99_f64;
    let mut dte = 1.0e99_f64;
    let mut dtdm = 1.0e99_f64;

    if ctrl.fj > 0.0 {
        let rj = (j_change / ctrl.angular_momentum_j).abs();
        dtj = limit_dt(rj, ctrl.fj);
    }

    if ctrl.fm > 0.0 {
        let rm = (env_change / ctrl.env_donor.max(ctrl.fm_limit)).abs();
        dtm = limit_dt(rm, ctrl.fm);
    }

    if ctrl.fr > 0.0 {
        let rr_d = (rel_gap_change_d
            / ctrl.rl_relative_gap[0].abs().max(ctrl.fr_limit))
        .abs();
        let dtr_d = limit_dt(rr_d, ctrl.fr);
        let rel_gap_change_a = if ctrl.rl_relative_gap_old[1] == 0.0 {
            0.0
        } else {
            ctrl.rl_relative_gap_old[1] - ctrl.rl_relative_gap[1]
        };
        let rr_a = (rel_gap_change_a
            / ctrl.rl_relative_gap[1].abs().max(ctrl.fr_limit))
        .abs();
        dtr = dtr_d.min(limit_dt(rr_a, ctrl.fr));
    }
    if dtr < ctrl.fr_dt_limit {
        dtr = ctrl.fr_dt_limit;
    }

    if ctrl.fa > 0.0 {
        let ra = (sep_change / ctrl.separation).abs();
        dta = limit_dt(ra, ctrl.fa);
    }

    if ctrl.fe > 0.0 {
        let re = (e_change / ctrl.eccentricity.max(ctrl.fe_limit)).abs();
        dte = limit_dt(re, ctrl.fe);
    }

    // `binary_timestep.f90:244-264` (`F3-8`): evaluates `fdm` on both donor
    // and accretor, including point-mass companions.
    if ctrl.fdm > 0.0 {
        let rdm_d = (ctrl.m[0] - ctrl.m_old[0]).abs() / ctrl.m_old[0];
        dtdm = limit_dt(rdm_d, ctrl.fdm);
        let rdm_a = (ctrl.m[1] - ctrl.m_old[1]).abs() / ctrl.m_old[1];
        dtdm = dtdm.min(limit_dt(rdm_a, ctrl.fdm));
    }

    let dt_set_raw = dtm.min(dtr).min(dtj).min(dta).min(dte).min(dtdm);
    let reason = if dt_set_raw == dtm {
        B_TLIM_ENV
    } else if dt_set_raw == dtr {
        B_TLIM_ROCHE
    } else if dt_set_raw == dtj {
        B_TLIM_JORB
    } else if dt_set_raw == dta {
        B_TLIM_SEP
    } else if dt_set_raw == dte {
        B_TLIM_ECC
    } else if dt_set_raw == dtdm {
        B_TLIM_DM
    } else {
        return Err(StarError::ConvergenceFailure(String::from(
            "Something wrong in binary timestep (binary_timestep.f90:280)",
        )));
    };
    let dt_set_yr = dt_set_raw.max(1.0e-7);

    let f_soft = ctrl.dt_softening_factor;
    let max_timestep_new = exp10(
        f_soft * log10(prev_max) + (1.0 - f_soft) * log10(dt_set_yr * SECYER),
    );
    Ok((max_timestep_new, reason))
}

/// Checks binary hard relative-change limits (`fj_hard`, `fm_hard`, `fr_hard`,
/// `fa_hard`, `fe_hard`, `fdm_hard` in `binary_timestep.f90:166-264`), returning
/// the first violated limit when `ctrl.ignore_hard_limits_this_step` is `false`.
///
/// Notice upstream checks each `*_hard` inside its corresponding `if (b% f* > 0)`
/// block (`binary_timestep.f90:166, 179, 191, 220, 232, 244`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn check_binary_hard_limits(
    ctrl: &BinaryTimestepControls,
    fm_hard: f64,
    fr_hard: f64,
    fj_hard: f64,
    fa_hard: f64,
    fe_hard: f64,
    fdm_hard: f64,
) -> Option<BinaryHardLimitViolation> {
    if ctrl.ignore_hard_limits_this_step {
        return None;
    }
    let c_dt = ctrl.time_delta_coeff;
    let (env_change, rel_gap_change_d, j_change, sep_change, e_change) =
        step_changes(ctrl);

    if ctrl.fj > 0.0 && fj_hard > 0.0 {
        let rj = (j_change / ctrl.angular_momentum_j).abs();
        if rj > fj_hard * c_dt {
            return Some(BinaryHardLimitViolation::FjHard);
        }
    }
    if ctrl.fm > 0.0 && fm_hard > 0.0 {
        let rm = (env_change / ctrl.env_donor.max(ctrl.fm_limit)).abs();
        if rm > fm_hard * c_dt {
            return Some(BinaryHardLimitViolation::FmHard);
        }
    }
    if ctrl.fr > 0.0 && fr_hard > 0.0 {
        let rr_d = (rel_gap_change_d
            / ctrl.rl_relative_gap[0].abs().max(ctrl.fr_limit))
        .abs();
        if rr_d > fr_hard * c_dt {
            return Some(BinaryHardLimitViolation::FrHardDonor);
        }
        let rel_gap_change_a = if ctrl.rl_relative_gap_old[1] == 0.0 {
            0.0
        } else {
            ctrl.rl_relative_gap_old[1] - ctrl.rl_relative_gap[1]
        };
        let rr_a = (rel_gap_change_a
            / ctrl.rl_relative_gap[1].abs().max(ctrl.fr_limit))
        .abs();
        if rr_a > fr_hard * c_dt {
            return Some(BinaryHardLimitViolation::FrHardAccretor);
        }
    }
    if ctrl.fa > 0.0 && fa_hard > 0.0 {
        let ra = (sep_change / ctrl.separation).abs();
        if ra > fa_hard * c_dt {
            return Some(BinaryHardLimitViolation::FaHard);
        }
    }
    if ctrl.fe > 0.0 && fe_hard > 0.0 {
        let re = (e_change / ctrl.eccentricity.max(ctrl.fe_limit)).abs();
        if re > fe_hard * c_dt {
            return Some(BinaryHardLimitViolation::FeHard);
        }
    }
    if ctrl.fdm > 0.0 && fdm_hard > 0.0 {
        let rdm_d = (ctrl.m[0] - ctrl.m_old[0]).abs() / ctrl.m_old[0];
        if rdm_d > fdm_hard * c_dt {
            return Some(BinaryHardLimitViolation::FdmHardDonor);
        }
        let rdm_a = (ctrl.m[1] - ctrl.m_old[1]).abs() / ctrl.m_old[1];
        if rdm_a > fdm_hard * c_dt {
            return Some(BinaryHardLimitViolation::FdmHardAccretor);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f3_8_point_mass_accretor_fdm_and_hard_limits() {
        let ctrl = BinaryTimestepControls {
            donor_time_step_yr: 100.0,
            max_timestep_sec: -1.0,
            dt_softening_factor: 0.0,
            is_point_mass_companion: true,
            m: [19.99 * MSUN, 1.42 * MSUN],
            m_old: [20.0 * MSUN, 1.40 * MSUN],
            fm: -1.0,
            fr: -1.0,
            fj: -1.0,
            fa: -1.0,
            fe: -1.0,
            fdm: 0.005,
            ..BinaryTimestepControls::default()
        };
        let (max_dt_sec, reason) =
            binary_pick_next_timestep(&ctrl).expect("finite limits");
        assert_eq!(reason, B_TLIM_DM);
        // Donor relative mass change is 0.0005, point-mass accretor is 0.02/1.40 = 1/70 > 0.005,
        // so accretor sets `dtdm = 100 / ((1/70) / 0.005) = 35 yr`.
        assert!((max_dt_sec / SECYER - 35.0).abs() < 1.0e-12);

        let viol =
            check_binary_hard_limits(&ctrl, -1.0, -1.0, -1.0, -1.0, -1.0, 0.01);
        assert_eq!(viol, Some(BinaryHardLimitViolation::FdmHardAccretor));

        let mut ignored = ctrl;
        ignored.ignore_hard_limits_this_step = true;
        assert_eq!(
            check_binary_hard_limits(
                &ignored, -1.0, -1.0, -1.0, -1.0, -1.0, 0.01
            ),
            None
        );
    }

    #[test]
    fn test_f3_8_first_model_zeros_orbital_deltas_and_fj_hard_sets_j_flag() {
        let mut ctrl = BinaryTimestepControls {
            doing_first_model_of_run: true,
            angular_momentum_j: 0.9 * 1.0e52,
            angular_momentum_j_old: 1.0e52,
            fj: 0.001,
            ..BinaryTimestepControls::default()
        };
        assert_eq!(
            check_binary_hard_limits(&ctrl, -1.0, -1.0, 0.01, -1.0, -1.0, -1.0),
            None
        );
        ctrl.doing_first_model_of_run = false;
        let viol =
            check_binary_hard_limits(&ctrl, -1.0, -1.0, 0.01, -1.0, -1.0, -1.0);
        assert_eq!(viol, Some(BinaryHardLimitViolation::FjHard));
        assert!(
            viol.is_some_and(
                BinaryHardLimitViolation::reduces_timestep_due_to_j
            )
        );
    }

    #[test]
    fn test_nan_limit_is_an_explicit_error_like_upstream_mesa_error() {
        // `binary_timestep.f90:278-280`: with a NaN step every limit is NaN,
        // no limit equals the minimum and upstream stops in `mesa_error`.
        let all_nan = BinaryTimestepControls {
            donor_time_step_yr: f64::NAN,
            ..BinaryTimestepControls::default()
        };
        assert!(matches!(
            binary_pick_next_timestep(&all_nan),
            Err(StarError::ConvergenceFailure(_))
        ));
    }
}
