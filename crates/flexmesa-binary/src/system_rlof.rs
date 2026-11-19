// SPDX-License-Identifier: LGPL-3.0-or-later
//! Coupled RLOF/wind mass-transfer accounting (`adjust_mdots`), `.photo` +
//! orbital snapshot/redo state management, and single-star Henyey step wrapper
//! for [`crate::system::BinarySystem`].
//!
//! Corresponds to `binary/private/binary_mdot.f90:428-986`,
//! `binary/private/binary_wind.f90:33-170`, and
//! `binary/private/binary_evolve.f90:257-620`.

use flexmesa_const::{CLIGHT, CRAD, MSUN, PI, RSUN, SECYER};
use flexmesa_star::{
    EvolveStepReport, PrimaryStateBuffer, StarError, StarModel, do_set_mdot,
    eval_photosphere_info, star_evolve_step,
};

use crate::edd_wind::{
    eval_accreted_material_j, eval_bondi_hoyle_wind_fraction,
    eval_point_mass_edd_and_retention,
    eval_tout_wind_enhancement_orbit_averaged,
};
use crate::extras::{BinaryExtras, DefaultBinaryExtras};
use crate::inlist::BinaryInlistControls;
use crate::mdot::{
    eval_arras_mdot_thin, eval_kolb_mdot_eccentric, eval_kolb_mdot_thick,
    eval_ritter_mdot_eccentric,
};
use crate::system_jdot::{
    EdotBreakdown, JdotBreakdown, star_photosphere_radius_cm,
};

/// Per-step wind, RLOF, Eddington, and system mass-loss accounting (`adjust_mdots`
/// in `binary/private/binary_mdot.f90:506-660`, indexed `[donor_slot, accretor_slot]`).
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct CoupledMdotAccounting {
    /// Intrinsic stellar wind rates `[mdot_wind_donor, mdot_wind_accretor]` ($\le 0$, $\mathrm{g\,s^{-1}}$).
    pub mdot_wind_intrinsic: [f64; 2],
    /// Bondi-Hoyle wind transfer fractions `[f_wind_donor, f_wind_accretor]`.
    pub wind_xfer_fraction: [f64; 2],
    /// Transferred wind rates `[mdot_wind_tr_donor, mdot_wind_tr_accretor]` ($\le 0$, $\mathrm{g\,s^{-1}}$).
    pub mdot_wind_transfer: [f64; 2],
    /// Escaping wind rates `[mdot_sys_wind_donor, mdot_sys_wind_accretor]` ($\le 0$, $\mathrm{g\,s^{-1}}$).
    pub mdot_system_wind: [f64; 2],
    /// Isotropic RLOF/Eddington/radiation ejection `[mdot_sys_tr_donor, mdot_sys_tr_accretor]` ($\le 0$, $\mathrm{g\,s^{-1}}$).
    pub mdot_system_transfer: [f64; 2],
    /// Circumbinary coplanar toroid mass-loss rate `mdot_system_cct` ($\le 0$, $\mathrm{g\,s^{-1}}$).
    pub mdot_system_cct: f64,
    /// Net stellar/point-mass rates `[mstar_dot_donor, mstar_dot_accretor]` ($\mathrm{g\,s^{-1}}$).
    pub component_mdot: [f64; 2],
    /// Retained RLOF + wind accretion rate on companion ($\ge 0$, $\mathrm{g\,s^{-1}}$).
    pub retained_accretion_rate: f64,
    /// Specific angular momentum of accreted material $j_{\mathrm{acc}}$ ($\mathrm{cm^2\,s^{-1}}$).
    pub accreted_material_j: f64,
    /// Point-mass Eddington accretion limit $\dot{M}_{\mathrm{Edd}}$ ($\mathrm{g\,s^{-1}}$).
    pub mdot_edd: f64,
    /// Point-mass radiative efficiency $\eta$.
    pub mdot_edd_eta: f64,
    /// Point-mass accretion luminosity $L_{\mathrm{acc}}$ ($\mathrm{erg\,s^{-1}}$).
    pub accretion_luminosity: f64,
    /// Arriving transfer rate at accretor `actual_mtransfer_rate` ($\le 0$, $\mathrm{g\,s^{-1}}$).
    pub actual_mtransfer_rate: f64,
}

/// Complete binary + compact in-memory stellar snapshot for implicit RLOF redo
/// iterations and timestep-cut retries (`binary_evolve.f90`).
#[derive(Debug, Clone)]
pub struct BinaryStepSnapshot {
    /// Compact primary-state snapshot of `donor`.
    pub donor_state: PrimaryStateBuffer,
    /// Compact primary-state snapshot of `Star` companion (`None` when point mass).
    pub companion_state: Option<PrimaryStateBuffer>,
    /// Point-mass state `(mass_g, m_bh_zero_eq_g, is_black_hole)` when companion is a point mass.
    pub point_mass_state: Option<(f64, f64, bool)>,
    /// Point-mass index `point_mass_i` in `{0, 1, 2}`.
    pub point_mass_i: usize,
    /// Active donor index `donor_i` in `{1, 2}`.
    pub donor_i: usize,
    /// Orbital separation $a$ (cm).
    pub separation: f64,
    /// Orbital period $P$ (s).
    pub period: f64,
    /// Orbital eccentricity $e$.
    pub eccentricity: f64,
    /// Orbital angular momentum $J_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-1}}$).
    pub angular_momentum_j: f64,
    /// Roche-lobe radii `[rl_donor, rl_accretor]` (cm).
    pub rl: [f64; 2],
    /// Periastron relative Roche-lobe gaps `[gap_donor, gap_accretor]`.
    pub rl_relative_gap: [f64; 2],
    /// Mass-transfer rate $\dot{M}_{\mathrm{tr}}$ ($\mathrm{g\,s^{-1}}$).
    pub mtransfer_rate: f64,
    /// Applied step mass-transfer rate `step_mtransfer_rate` ($\mathrm{g\,s^{-1}}$).
    pub step_mtransfer_rate: f64,
    /// Orbital torque $\dot{J}_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-2}}$).
    pub jdot: f64,
    /// Binary age (yr).
    pub binary_age_yr: f64,
    /// Binary model number.
    pub model_number: usize,
    /// Maximum binary timestep (s).
    pub max_timestep_sec: f64,
    /// Implicit solver adaptive `change_factor`.
    pub change_factor: f64,
    /// Magnetic-braking active flags `[mb_1, mb_2]`.
    pub using_jdot_mb: [bool; 2],
    /// Cumulative external angular momentum change ($\mathrm{g\,cm^2\,s^{-1}}$).
    pub cumulative_j_ext: f64,
    /// Previous mass-transfer accounting.
    pub last_mdot: CoupledMdotAccounting,
    /// Previous torque breakdown.
    pub last_jdot: JdotBreakdown,
    /// Previous eccentricity-rate breakdown.
    pub last_edot: EdotBreakdown,
}

/// Orbital and step-accounting scalar snapshot passed into [`BinaryStepSnapshot`].
#[derive(Debug, Clone, Copy)]
pub struct BinaryStepScalars {
    /// Point-mass state `(mass_g, m_bh_zero_eq_g, is_black_hole)` when companion is a point mass.
    pub point_mass_state: Option<(f64, f64, bool)>,
    /// Point-mass index `point_mass_i` in `{0, 1, 2}`.
    pub point_mass_i: usize,
    /// Active donor index `donor_i` in `{1, 2}`.
    pub donor_i: usize,
    /// Orbital separation $a$ (cm).
    pub separation: f64,
    /// Orbital period $P$ (s).
    pub period: f64,
    /// Orbital eccentricity $e$.
    pub eccentricity: f64,
    /// Orbital angular momentum $J_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-1}}$).
    pub angular_momentum_j: f64,
    /// Roche-lobe radii `[rl_donor, rl_accretor]` (cm).
    pub rl: [f64; 2],
    /// Periastron relative Roche-lobe gaps `[gap_donor, gap_accretor]`.
    pub rl_relative_gap: [f64; 2],
    /// Mass-transfer rate $\dot{M}_{\mathrm{tr}}$ ($\mathrm{g\,s^{-1}}$).
    pub mtransfer_rate: f64,
    /// Applied step mass-transfer rate `step_mtransfer_rate` ($\mathrm{g\,s^{-1}}$).
    pub step_mtransfer_rate: f64,
    /// Orbital torque $\dot{J}_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-2}}$).
    pub jdot: f64,
    /// Binary age (yr).
    pub binary_age_yr: f64,
    /// Binary model number.
    pub model_number: usize,
    /// Maximum binary timestep (s).
    pub max_timestep_sec: f64,
    /// Implicit solver adaptive `change_factor`.
    pub change_factor: f64,
    /// Magnetic-braking active flags `[mb_1, mb_2]`.
    pub using_jdot_mb: [bool; 2],
    /// Cumulative external angular momentum change ($\mathrm{g\,cm^2\,s^{-1}}$).
    pub cumulative_j_ext: f64,
    /// Previous mass-transfer accounting.
    pub last_mdot: CoupledMdotAccounting,
    /// Previous torque breakdown.
    pub last_jdot: JdotBreakdown,
    /// Previous eccentricity-rate breakdown.
    pub last_edot: EdotBreakdown,
}

impl BinaryStepSnapshot {
    /// Captures a full binary + in-memory stellar snapshot before advancing a step.
    #[must_use]
    pub fn capture(
        donor: &StarModel,
        companion_star: Option<&StarModel>,
        s: BinaryStepScalars,
    ) -> Self {
        Self {
            donor_state: PrimaryStateBuffer::from_model(donor),
            companion_state: companion_star.map(PrimaryStateBuffer::from_model),
            point_mass_state: s.point_mass_state,
            point_mass_i: s.point_mass_i,
            donor_i: s.donor_i,
            separation: s.separation,
            period: s.period,
            eccentricity: s.eccentricity,
            angular_momentum_j: s.angular_momentum_j,
            rl: s.rl,
            rl_relative_gap: s.rl_relative_gap,
            mtransfer_rate: s.mtransfer_rate,
            step_mtransfer_rate: s.step_mtransfer_rate,
            jdot: s.jdot,
            binary_age_yr: s.binary_age_yr,
            model_number: s.model_number,
            max_timestep_sec: s.max_timestep_sec,
            change_factor: s.change_factor,
            using_jdot_mb: s.using_jdot_mb,
            cumulative_j_ext: s.cumulative_j_ext,
            last_mdot: s.last_mdot,
            last_jdot: s.last_jdot,
            last_edot: s.last_edot,
        }
    }

    /// Overwrites this snapshot in place from current binary and stellar state,
    /// reusing existing heap capacities across trials.
    pub fn capture_into(
        &mut self,
        donor: &StarModel,
        companion_star: Option<&StarModel>,
        s: BinaryStepScalars,
    ) {
        self.donor_state.capture_from(donor);
        match (self.companion_state.as_mut(), companion_star) {
            (Some(dst), Some(src)) => dst.capture_from(src),
            (None, Some(src)) => {
                self.companion_state =
                    Some(PrimaryStateBuffer::from_model(src));
            }
            (_, None) => self.companion_state = None,
        }
        self.point_mass_state = s.point_mass_state;
        self.point_mass_i = s.point_mass_i;
        self.donor_i = s.donor_i;
        self.separation = s.separation;
        self.period = s.period;
        self.eccentricity = s.eccentricity;
        self.angular_momentum_j = s.angular_momentum_j;
        self.rl = s.rl;
        self.rl_relative_gap = s.rl_relative_gap;
        self.mtransfer_rate = s.mtransfer_rate;
        self.step_mtransfer_rate = s.step_mtransfer_rate;
        self.jdot = s.jdot;
        self.binary_age_yr = s.binary_age_yr;
        self.model_number = s.model_number;
        self.max_timestep_sec = s.max_timestep_sec;
        self.change_factor = s.change_factor;
        self.using_jdot_mb = s.using_jdot_mb;
        self.cumulative_j_ext = s.cumulative_j_ext;
        self.last_mdot = s.last_mdot;
        self.last_jdot = s.last_jdot;
        self.last_edot = s.last_edot;
    }

    /// Restores the saved in-memory donor and optional companion star models in
    /// place while preserving `controls`, `job`, and cumulative counters.
    pub fn restore_stars(
        &self,
        donor: &mut StarModel,
        companion_star: Option<&mut StarModel>,
    ) -> Result<(), StarError> {
        self.donor_state.restore_into(donor)?;

        if let (Some(acc), Some(comp)) =
            (companion_star, self.companion_state.as_ref())
        {
            comp.restore_into(acc)?;
        }
        Ok(())
    }
}

pub(crate) fn surface_h1_and_he4(star: &StarModel) -> (f64, f64) {
    let mut x = star.xh_mass.first().copied().unwrap_or(0.7);
    let mut y = star.yhe_mass.first().copied().unwrap_or(0.28);
    if star.nz > 0 && star.species > 0 {
        let xa0 = star.cell_xa(0);
        if let Some(j) = star
            .iso_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case("h1"))
        {
            x = xa0[j];
        }
        if let Some(j) = star
            .iso_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case("he4"))
        {
            y = xa0[j];
        }
    }
    (x.clamp(0.0, 1.0), y.clamp(0.0, 1.0))
}

/// Evaluates intrinsic stellar wind $\dot{M}_{\mathrm{wind}} \le 0$ ($\mathrm{g\,s^{-1}}$)
/// without RLOF `mass_change` using [`DefaultBinaryExtras`].
pub fn eval_star_intrinsic_wind_cgs(
    star: &mut StarModel,
    rl_cm: f64,
    do_enhance: bool,
    b_wind: f64,
    ignore_rlof: bool,
) -> Result<f64, StarError> {
    eval_star_intrinsic_wind_cgs_with_extras(
        star,
        1,
        rl_cm,
        do_enhance,
        b_wind,
        ignore_rlof,
        &mut DefaultBinaryExtras,
    )
}

/// Evaluates intrinsic stellar wind $\dot{M}_{\mathrm{wind}} \le 0$ ($\mathrm{g\,s^{-1}}$)
/// without RLOF `mass_change`, routing `use_other_wind` through [`BinaryExtras::other_wind_cgs`]
/// before applying rotational boost/caps (`star/private/winds.f90:431-475`) and
/// Tout & Eggleton (1988) tidal wind enhancement (`binary_wind.f90:131-170`).
pub fn eval_star_intrinsic_wind_cgs_with_extras(
    star: &mut StarModel,
    star_idx: usize,
    rl_cm: f64,
    do_enhance: bool,
    b_wind: f64,
    ignore_rlof: bool,
    extras: &mut dyn BinaryExtras,
) -> Result<f64, StarError> {
    let custom_raw = if star.controls.use_other_wind {
        Some(
            extras
                .other_wind_cgs(star, star_idx, ignore_rlof)?
                .ok_or_else(|| {
                    StarError::InvalidInput(String::from(
                        "use_other_wind = .true. requires an other_wind hook",
                    ))
                })?,
        )
    } else {
        None
    };
    let mut wind_cgs = if let Some(raw_wind_cgs) = custom_raw {
        let mut wind_mdot = (-raw_wind_cgs).max(0.0);
        let l1 = star.lum.first().copied().unwrap_or(1.0).max(1.0);
        let t1 = if star.teff > 0.0 {
            star.teff
        } else {
            let r0 = star.r.first().copied().unwrap_or(1.0).max(1.0);
            flexmesa_math::pow(l1 / (PI * r0 * r0 * CRAD * CLIGHT), 0.25)
        }
        .max(100.0);
        let r1 = (l1 / (PI * CRAD * CLIGHT * flexmesa_math::pow4(t1)))
            .sqrt()
            .max(1.0);
        let mw = star.controls.mix_wind.clone();
        if mw.max_wind > 0.0 {
            wind_mdot = wind_mdot.min(mw.max_wind * (MSUN / SECYER));
        }
        if star.controls.rotation_flag
            && mw.mdot_omega_power > 0.0
            && wind_mdot > 0.0
        {
            let m1 = star.mstar;
            let cgrav = star
                .cgrav
                .first()
                .copied()
                .unwrap_or(flexmesa_const::STANDARD_CGRAV);
            let t_kh = (0.75 * cgrav * m1 * m1 / (r1 * l1)).max(1.0);
            let w_lim = (mw.rotational_mdot_kh_fac.max(0.0) * m1 / t_kh)
                .min(wind_mdot * mw.rotational_mdot_boost_fac.max(1.0));
            let w_div_wc = flexmesa_star::eval_surf_avg_w_div_w_crit(star);
            let mut enh = flexmesa_math::pow(
                (1.0 - w_div_wc).max(1.0e-22),
                -mw.mdot_omega_power,
            );
            if mw.max_rotational_mdot_boost > 0.0 {
                enh = enh.min(mw.max_rotational_mdot_boost);
            }
            wind_mdot = (wind_mdot * enh).min(w_lim);
            let prev_w = -star.aux.step_hist.mstar_dot_old;
            if prev_w > 0.0 && mw.max_mdot_jump_for_rotation > 0.0 {
                wind_mdot =
                    wind_mdot.min(mw.max_mdot_jump_for_rotation * prev_w);
            }
            if mw.max_wind > 0.0 {
                wind_mdot = wind_mdot.min(mw.max_wind * (MSUN / SECYER));
            }
        }
        if wind_mdot > 0.0 { -wind_mdot } else { 0.0 }
    } else {
        let saved_mc = star.controls.mass_change;
        let saved_other = star.controls.use_other_adjust_mdot;
        let saved_other_w = star.controls.use_other_wind;
        let saved_mdot = star.mstar_dot;
        star.controls.mass_change = 0.0;
        star.controls.use_other_adjust_mdot = false;
        star.controls.use_other_wind = false;
        let res = do_set_mdot(star);
        let w = star.mstar_dot.min(0.0);
        star.controls.mass_change = saved_mc;
        star.controls.use_other_adjust_mdot = saved_other;
        star.controls.use_other_wind = saved_other_w;
        star.mstar_dot = saved_mdot;
        res?;
        w
    };
    if do_enhance && wind_cgs < 0.0 {
        wind_cgs = eval_tout_wind_enhancement_orbit_averaged(
            wind_cgs,
            star_photosphere_radius_cm(star),
            rl_cm,
            b_wind,
            0.0,
            500,
        );
    }
    Ok(wind_cgs)
}

/// Evaluates intrinsic stellar wind $\dot{M}_{\mathrm{wind}} \le 0$ ($\mathrm{g\,s^{-1}}$)
/// with orbit-averaged Tout & Eggleton (1988) tidal wind enhancement over
/// `(eccentricity, anomaly_steps)` (`binary_wind.f90:131-170`).
#[allow(clippy::too_many_arguments)]
pub fn eval_star_intrinsic_wind_cgs_eccentric(
    star: &mut StarModel,
    star_idx: usize,
    rl_cm: f64,
    do_enhance: bool,
    b_wind: f64,
    eccentricity: f64,
    anomaly_steps: usize,
    ignore_rlof: bool,
    extras: &mut dyn BinaryExtras,
) -> Result<f64, StarError> {
    let mut wind_cgs = eval_star_intrinsic_wind_cgs_with_extras(
        star,
        star_idx,
        rl_cm,
        false,
        b_wind,
        ignore_rlof,
        extras,
    )?;
    if do_enhance && wind_cgs < 0.0 {
        wind_cgs = eval_tout_wind_enhancement_orbit_averaged(
            wind_cgs,
            star_photosphere_radius_cm(star),
            rl_cm,
            b_wind,
            eccentricity,
            anomaly_steps,
        );
    }
    Ok(wind_cgs)
}

/// Evaluates coupled winds, Bondi-Hoyle wind transfer, and Eddington retention
/// (`binary_mdot.f90:428-622`) using [`DefaultBinaryExtras`].
#[allow(clippy::too_many_arguments)]
pub fn prepare_coupled_mdots(
    ctrl: &BinaryInlistControls,
    donor: &mut StarModel,
    accretor_star: Option<&mut StarModel>,
    point_mass_state: Option<(f64, f64)>,
    donor_i: usize,
    rl: [f64; 2],
    separation: f64,
    eccentricity: f64,
    trial_mtransfer_rate: f64,
    ignore_rlof: bool,
) -> Result<CoupledMdotAccounting, StarError> {
    prepare_coupled_mdots_with_extras(
        ctrl,
        donor,
        accretor_star,
        point_mass_state,
        donor_i,
        rl,
        separation,
        eccentricity,
        trial_mtransfer_rate,
        ignore_rlof,
        &mut DefaultBinaryExtras,
    )
}

/// Evaluates coupled winds, Bondi-Hoyle wind transfer, decoupled Eddington
/// retention vs. radiation-corrected accretion (`binary_mdot.f90:428-622`), and
/// configures `donor` and optional `accretor` `mass_change` and `mstar_dot`
/// before calling `star_evolve_step`.
#[allow(clippy::too_many_arguments)]
pub fn prepare_coupled_mdots_with_extras(
    ctrl: &BinaryInlistControls,
    donor: &mut StarModel,
    mut accretor_star: Option<&mut StarModel>,
    point_mass_state: Option<(f64, f64)>,
    donor_i: usize,
    rl: [f64; 2],
    separation: f64,
    eccentricity: f64,
    trial_mtransfer_rate: f64,
    ignore_rlof: bool,
    extras: &mut dyn BinaryExtras,
) -> Result<CoupledMdotAccounting, StarError> {
    let mtransfer_rate = if ignore_rlof {
        0.0
    } else {
        trial_mtransfer_rate.min(0.0)
    };
    let (
        enh_d,
        b_d,
        do_wt_d,
        abh_d,
        bbh_d,
        maxf_d,
        enh_a,
        b_a,
        do_wt_a,
        abh_a,
        bbh_a,
        maxf_a,
    ) = if donor_i == 1 {
        (
            ctrl.do_enhance_wind_1,
            ctrl.tout_b_wind_1,
            ctrl.do_wind_mass_transfer_1,
            ctrl.wind_bh_alpha_1,
            ctrl.wind_bh_beta_1,
            ctrl.max_wind_transfer_fraction_1,
            ctrl.do_enhance_wind_2,
            ctrl.tout_b_wind_2,
            ctrl.do_wind_mass_transfer_2,
            ctrl.wind_bh_alpha_2,
            ctrl.wind_bh_beta_2,
            ctrl.max_wind_transfer_fraction_2,
        )
    } else {
        (
            ctrl.do_enhance_wind_2,
            ctrl.tout_b_wind_2,
            ctrl.do_wind_mass_transfer_2,
            ctrl.wind_bh_alpha_2,
            ctrl.wind_bh_beta_2,
            ctrl.max_wind_transfer_fraction_2,
            ctrl.do_enhance_wind_1,
            ctrl.tout_b_wind_1,
            ctrl.do_wind_mass_transfer_1,
            ctrl.wind_bh_alpha_1,
            ctrl.wind_bh_beta_1,
            ctrl.max_wind_transfer_fraction_1,
        )
    };

    let m_d = donor.mstar;
    let m_a = point_mass_state
        .map(|(m, _)| m)
        .or_else(|| accretor_star.as_deref().map(|s| s.mstar))
        .unwrap_or(0.0);

    let acc_i = 3 - donor_i;
    let r_d_phot = star_photosphere_radius_cm(donor);
    let w_d = eval_star_intrinsic_wind_cgs_eccentric(
        donor,
        donor_i,
        rl[0],
        enh_d,
        b_d,
        eccentricity,
        ctrl.anomaly_steps,
        ignore_rlof,
        extras,
    )?;
    let w_a = match accretor_star.as_deref_mut() {
        Some(acc) => eval_star_intrinsic_wind_cgs_eccentric(
            acc,
            acc_i,
            rl[1],
            enh_a,
            b_a,
            eccentricity,
            ctrl.anomaly_steps,
            ignore_rlof,
            extras,
        )?,
        None => 0.0,
    };

    let f_wt_d = if do_wt_d {
        eval_bondi_hoyle_wind_fraction(
            m_d,
            m_a,
            r_d_phot,
            separation,
            eccentricity,
            abh_d,
            bbh_d,
            maxf_d,
        )
    } else {
        0.0
    };
    let f_wt_a = match accretor_star.as_deref() {
        Some(acc) if do_wt_a => eval_bondi_hoyle_wind_fraction(
            m_a,
            m_d,
            star_photosphere_radius_cm(acc),
            separation,
            eccentricity,
            abh_a,
            bbh_a,
            maxf_a,
        ),
        _ => 0.0,
    };

    let w_tr_d = w_d * f_wt_d;
    let w_tr_a = w_a * f_wt_a;
    let donor_net_mdot = mtransfer_rate + w_d - w_tr_a;
    donor.controls.mass_change = mtransfer_rate / MSUN * SECYER;
    donor.controls.use_other_adjust_mdot = true;
    if donor.aux.step_hist.mstar_dot_old == 0.0 && donor.model_number > 0 {
        donor.aux.step_hist.mstar_dot_old = donor.mstar_dot;
    }
    donor.mstar_dot = donor_net_mdot;

    let mut out = CoupledMdotAccounting {
        mdot_wind_intrinsic: [w_d, w_a],
        wind_xfer_fraction: [f_wt_d, f_wt_a],
        mdot_wind_transfer: [w_tr_d, w_tr_a],
        mdot_system_wind: [w_d - w_tr_d, w_a - w_tr_a],
        ..CoupledMdotAccounting::default()
    };

    if let Some((m_pm, m_bh_eq)) = point_mass_state {
        let (x_surf, _) = surface_h1_and_he4(donor);
        let kap_surf = donor.opacity.first().copied().unwrap_or(0.34);
        let pm = eval_point_mass_edd_and_retention(
            m_pm,
            m_bh_eq,
            x_surf,
            kap_surf,
            ctrl.use_es_opacity_for_mdot_edd,
            ctrl.use_this_for_mdot_edd_eta,
            ctrl.use_this_for_mdot_edd,
            ctrl.limit_retention_by_mdot_edd,
            ctrl.use_radiation_corrected_transfer_rate,
            mtransfer_rate,
            w_tr_d,
            ctrl.mass_transfer_alpha,
            ctrl.mass_transfer_beta,
            ctrl.mass_transfer_delta,
        );
        out.mdot_edd = pm.mdot_edd;
        out.mdot_edd_eta = pm.mdot_edd_eta;
        out.accretion_luminosity = pm.accretion_luminosity;
        out.actual_mtransfer_rate = pm.actual_mtransfer_rate;
        out.retained_accretion_rate = pm.component_mdot_accretor;
        out.component_mdot = [donor_net_mdot, pm.component_mdot_accretor];
        out.mdot_system_transfer = [
            pm.mdot_system_transfer_donor,
            pm.mdot_system_transfer_accretor,
        ];
        out.mdot_system_cct = pm.mdot_system_cct;
    } else if let Some(acc) = accretor_star {
        let fixed_xfer = (1.0
            - ctrl.mass_transfer_alpha
            - ctrl.mass_transfer_beta
            - ctrl.mass_transfer_delta)
            .clamp(0.0, 1.0);
        let actual_tr = mtransfer_rate * fixed_xfer + w_tr_d;
        let retained = (-actual_tr).max(0.0);
        let acc_net_mdot = retained + w_a;
        acc.controls.mass_change = retained / MSUN * SECYER;
        acc.controls.use_other_adjust_mdot = true;
        if acc.aux.step_hist.mstar_dot_old == 0.0 && acc.model_number > 0 {
            acc.aux.step_hist.mstar_dot_old = acc.mstar_dot;
        }
        acc.mstar_dot = acc_net_mdot;
        if retained > 0.0 && donor.nz > 0 && donor.species == acc.species {
            acc.controls.accrete_same_as_surface = false;
            acc.controls.accrete_given_mass_fractions = true;
            acc.controls
                .accretion_species_id
                .clone_from(&donor.iso_names);
            acc.controls.accretion_species_xa = donor.cell_xa(0).to_vec();
        }
        out.actual_mtransfer_rate = actual_tr;
        out.retained_accretion_rate = retained;
        out.accreted_material_j = if ctrl.do_j_accretion && retained > 0.0 {
            eval_accreted_material_j(
                m_d,
                m_a,
                star_photosphere_radius_cm(acc),
                separation,
            )
            .0
        } else {
            0.0
        };
        out.component_mdot = [donor_net_mdot, acc_net_mdot];
        out.mdot_system_transfer = [
            mtransfer_rate * ctrl.mass_transfer_alpha,
            mtransfer_rate * ctrl.mass_transfer_beta,
        ];
        out.mdot_system_cct = mtransfer_rate * ctrl.mass_transfer_delta;
    }
    Ok(out)
}

/// Evaluates the explicit mass-transfer prescription (`"Ritter"`, `"Kolb"`, or `"Arras"`)
/// on the post-step donor profile (`get_info_for_explicit_mdot` in `binary_mdot.f90:662-683`)
/// using upstream's default `anomaly_steps = 500`.
pub fn eval_scheme_explicit_mdot(
    scheme: &str,
    donor: &StarModel,
    m_accretor: f64,
    rl_donor: f64,
    separation: f64,
    eccentricity: f64,
) -> Result<f64, StarError> {
    eval_scheme_explicit_mdot_with_steps(
        scheme,
        donor,
        m_accretor,
        rl_donor,
        separation,
        eccentricity,
        500,
    )
}

/// Evaluates the explicit mass-transfer prescription (`"Ritter"`, `"Kolb"`, or `"Arras"`)
/// on the post-step donor profile (`get_info_for_explicit_mdot` in `binary_mdot.f90:662-683`),
/// integrating eccentric orbits over `anomaly_steps` (`binary_mdot.f90:850,918`)
/// and using photospheric radius `b% r(d_i)` (`binary_mdot.f90:845,870,913`).
pub fn eval_scheme_explicit_mdot_with_steps(
    scheme: &str,
    donor: &StarModel,
    m_accretor: f64,
    rl_donor: f64,
    separation: f64,
    eccentricity: f64,
    anomaly_steps: usize,
) -> Result<f64, StarError> {
    if donor.nz == 0 {
        return Ok(0.0);
    }
    let (m_d, r_d) = (donor.mstar, RSUN * eval_photosphere_info(donor).r_rsun);
    let (rho0, p0, t0, mu0) = (
        donor.rho[0].max(1.0e-30),
        donor.peos[0].max(1.0),
        donor.t[0].max(10.0),
        donor.mu[0].max(0.5),
    );
    let mdot = if scheme.eq_ignore_ascii_case("Arras") {
        eval_arras_mdot_thin(m_d, m_accretor, separation, r_d, rho0, t0, mu0)
    } else if scheme.eq_ignore_ascii_case("Kolb") {
        let thin = eval_ritter_mdot_eccentric(
            m_d,
            m_accretor,
            r_d,
            rl_donor,
            eccentricity,
            rho0,
            p0,
            t0,
            mu0,
            true,
            anomaly_steps,
        );
        let thick = if eccentricity > 0.0 {
            eval_kolb_mdot_eccentric(
                m_d,
                m_accretor,
                rl_donor,
                eccentricity,
                &donor.r,
                &donor.peos,
                &donor.t,
                &donor.mu,
                &donor.gamma1,
                anomaly_steps,
            )
        } else {
            eval_kolb_mdot_thick(
                m_d,
                m_accretor,
                rl_donor,
                &donor.r,
                &donor.peos,
                &donor.t,
                &donor.mu,
                &donor.gamma1,
            )
        };
        thin + thick
    } else if scheme.eq_ignore_ascii_case("Ritter") {
        eval_ritter_mdot_eccentric(
            m_d,
            m_accretor,
            r_d,
            rl_donor,
            eccentricity,
            rho0,
            p0,
            t0,
            mu0,
            false,
            anomaly_steps,
        )
    } else {
        return Err(StarError::InvalidInput(format!(
            "unsupported explicit mdot_scheme '{scheme}' (expected 'Ritter', 'Kolb', or 'Arras')"
        )));
    };
    Ok(mdot)
}

/// Blends explicit mass-transfer rate with the previous step's rate (`cur_mdot_frac`)
/// and caps at `max_explicit_abs_mdot` ($M_\odot\,\mathrm{yr}^{-1}$)
/// (`binary_evolve.f90:485-503`).
#[must_use]
pub fn blend_explicit_mdot(
    prev_mdot: f64,
    explicit_mdot: f64,
    cur_mdot_frac: f64,
    max_explicit_abs_mdot_msun_yr: f64,
) -> f64 {
    let w = cur_mdot_frac.clamp(0.0, 1.0);
    let blended = prev_mdot * w + explicit_mdot * (1.0 - w);
    let cap_cgs = max_explicit_abs_mdot_msun_yr.max(0.0) * (MSUN / SECYER);
    blended.min(0.0).max(-cap_cgs)
}

/// Checks `check_delta_mdot` (`star/private/timestep.f90:1889-1917`) at the
/// accepted coupled binary step (`run_binary_support.f90:546-558`).
#[must_use]
pub fn star_exceeds_delta_mdot_hard_limit(star: &StarModel) -> bool {
    let mw = &star.controls.mix_wind;
    let hard_lim = mw.delta_mdot_hard_limit
        * star.controls.ts_stop.time_delta_coeff.max(1.0e-6);
    if hard_lim <= 0.0 || star.model_number <= 1 {
        return false;
    }
    let (mdot, mdot_old) = (star.mstar_dot, star.aux.step_hist.mstar_dot_old);
    if (mdot - mdot_old).abs() == 0.0 {
        return false;
    }
    let denom = mw.delta_mdot_atol.max(0.0) * (MSUN / SECYER)
        + mw.delta_mdot_rtol.max(0.0) * mdot.abs().max(mdot_old.abs());
    denom > 0.0 && (mdot - mdot_old).abs() / denom > hard_lim
}

/// Advances a single [`StarModel`] through `star_evolve_step` at trial timestep
/// `dt_try` (s), deferring `StarControls` stopping checks and binary-coupled
/// `delta_mdot_hard_limit` retries (`run_binary_support.f90:546-558`) to the
/// outer coupled binary step.
pub fn evolve_single_star_without_term_rewind(
    star: &mut StarModel,
    dt_try: f64,
    allow_remesh: bool,
) -> Result<EvolveStepReport, StarError> {
    let saved_limits = (
        star.controls.max_model_number.take(),
        std::mem::replace(&mut star.controls.max_age, 1.0e36),
        star.controls.max_age_in_seconds.take(),
        std::mem::take(&mut star.controls.xa_central_lower_limit),
        std::mem::take(&mut star.controls.xa_central_upper_limit),
        std::mem::replace(&mut star.controls.stop_near_zams, false),
        std::mem::replace(
            &mut star.controls.mix_wind.delta_mdot_hard_limit,
            -1.0,
        ),
    );

    let saved_other_wind =
        std::mem::replace(&mut star.controls.use_other_wind, false);
    star.dt = dt_try;
    star.dt_next = dt_try;
    let do_remesh = allow_remesh
        && star.controls.okay_to_remesh
        && star.controls.mesh_delta_coeff > 0.0
        && dt_try >= star.controls.mix_wind.remesh_dt_limit;
    let op_split = star.controls.op_split_burn;
    let res = star_evolve_step(star, do_remesh, op_split, true);

    star.controls.use_other_wind = saved_other_wind;

    (
        star.controls.max_model_number,
        star.controls.max_age,
        star.controls.max_age_in_seconds,
        star.controls.xa_central_lower_limit,
        star.controls.xa_central_upper_limit,
        star.controls.stop_near_zams,
        star.controls.mix_wind.delta_mdot_hard_limit,
    ) = saved_limits;
    res
}
