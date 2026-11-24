// SPDX-License-Identifier: GPL-3.0-or-later
//! Coupled `BinarySystem` state container, constructors, component switching,
//! and `.photo` + orbital state snapshot helpers (`binary_def.f90`,
//! `run_binary_support.f90`, and `binary_evolve.f90:103-255`).

use flexmesa_const::{MSUN, RSUN, SECDAY, SECYER};
use flexmesa_star::{
    StarError, StarModel, set_hydro_vars, sync_post_henyey_rotation,
};

use crate::edd_wind::bh_initial_equatorial_mass;
use crate::inlist::BinaryInlistControls;
use crate::orbit::{
    eval_j_orb, eval_rl_relative_gap, eval_rlobe, period_from_separation,
    separation_from_period,
};
use crate::roche_deformation::BinaryTidalDeformationHook;
use crate::system_init::prep_and_settle_binary_star;
use crate::system_jdot::{
    EdotBreakdown, JdotBreakdown, apply_initial_orbit_sync, eval_star_spin_j,
    star_photosphere_radius_cm,
};
use crate::system_rlof::{
    BinaryStepScalars, BinaryStepSnapshot, CoupledMdotAccounting,
};

#[inline]
pub(crate) fn compute_rl_and_gaps(
    m_d: f64,
    m_a: f64,
    r_d: f64,
    r_a: f64,
    separation: f64,
    eccentricity: f64,
) -> ([f64; 2], [f64; 2]) {
    let rl_d = eval_rlobe(m_d, m_a, separation);
    let rl_a = eval_rlobe(m_a, m_d, separation);
    (
        [rl_d, rl_a],
        [
            eval_rl_relative_gap(r_d, rl_d, eccentricity),
            eval_rl_relative_gap(r_a, rl_a, eccentricity),
        ],
    )
}

/// Companion representation in a coupled [`BinarySystem`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BinaryCompanion {
    /// Point-mass compact object (white dwarf, neutron star, or Kerr black hole).
    PointMass {
        /// Current mass $M_{\mathrm{pm}}$ (g).
        mass_g: f64,
        /// Initial equivalent non-rotating BH mass $M_{\mathrm{BH},0}^{\mathrm{eq}}$ (g).
        m_bh_zero_eq_g: f64,
        /// Whether Eddington accretion uses relativistic Kerr BH radiative efficiency.
        is_black_hole: bool,
    },
    /// Actively evolved secondary [`StarModel`].
    Star(Box<StarModel>),
}

/// Cumulative binary retry, implicit trial, timestep cut, and hard-limit
/// rollback counters preserved across snapshot rollbacks on [`BinarySystem`].
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct BinaryStepCounters {
    /// Cumulative outer step retries (`outer_retry > 0`).
    pub num_retries: usize,
    /// Cumulative coupled binary trials executed in implicit mode.
    pub num_implicit_trials: usize,
    /// Cumulative solver `CutDt` restarts.
    pub num_dt_cuts: usize,
    /// Cumulative retries triggered by binary hard limits or `delta_mdot_hard_limit`.
    pub num_hard_limit_retries: usize,
}

/// Coupled binary star evolution state and step driver (`binary_info`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BinarySystem {
    /// Active donor star model (`s_donor`).
    pub donor: StarModel,
    /// Companion (point mass or secondary `StarModel`).
    pub companion: BinaryCompanion,
    /// Orbital separation $a$ (cm).
    pub separation: f64,
    /// Orbital period $P$ (s).
    pub period: f64,
    /// Orbital eccentricity $e \in [0, 1)$.
    pub eccentricity: f64,
    /// Orbital angular momentum $J_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-1}}$).
    pub angular_momentum_j: f64,
    /// Roche-lobe radii `[rl_donor, rl_accretor]` (cm).
    pub rl: [f64; 2],
    /// Periastron relative Roche-lobe gaps `[gap_donor, gap_accretor]`.
    pub rl_relative_gap: [f64; 2],
    /// Current mass transfer rate $\dot{M}_{\mathrm{tr}} \le 0$ ($\mathrm{g\,s^{-1}}$).
    pub mtransfer_rate: f64,
    /// Mass transfer rate actually applied during the most recent accepted step
    /// (`b% step_mtransfer_rate` in `run_binary_support.f90:415`, $\mathrm{g\,s^{-1}}$).
    pub step_mtransfer_rate: f64,
    /// Initial component masses `[initial_mass(1), initial_mass(2)]` in $M_\odot$
    /// (`binary_def.f90:32` and `binary_evolve.f90:75-87`).
    pub initial_mass_msun: [f64; 2],
    /// Initial orbital period in seconds (`initial_binary_period` in
    /// `binary_def.f90:29` and `binary_evolve.f90:173`).
    pub initial_binary_period_sec: f64,
    /// Total orbital angular momentum derivative $\dot{J}_{\mathrm{orb}}$ ($\mathrm{g\,cm^2\,s^{-2}}$).
    pub jdot: f64,
    /// Binary age (yr).
    pub binary_age_yr: f64,
    /// Completed binary step count.
    pub model_number: usize,
    /// Next maximum timestep (s).
    pub max_timestep_sec: f64,
    /// Non-conservative mass transfer fraction lost from donor ($\alpha_{\mathrm{mt}}$).
    pub mass_transfer_alpha: f64,
    /// Fraction lost from vicinity of accretor ($\beta_{\mathrm{mt}}$).
    pub mass_transfer_beta: f64,
    /// Fraction lost from circumbinary toroid ($\delta_{\mathrm{mt}}$).
    pub mass_transfer_delta: f64,
    /// Dimensionless radius of circumbinary toroid ($\gamma_{\mathrm{mt}}$).
    pub mass_transfer_gamma: f64,
    /// Whether to use the implicit Roche-lobe overflow solver.
    pub use_implicit_rlof: bool,
    /// Full `&binary_job` and `&binary_controls` settings.
    pub controls: BinaryInlistControls,
    /// Point-mass component index (`0` for two stars, `1` or `2` when one star is a point mass).
    pub point_mass_i: usize,
    /// Active donor star index (`1` or `2`).
    pub donor_i: usize,
    /// Suppress RLOF mass transfer when `true` (`ignore_rlof_flag`).
    pub ignore_rlof_flag: bool,
    /// Equivalent non-rotating BH mass $M_{\mathrm{BH},0}^{\mathrm{eq}}$ in grams (`b% eq_initial_bh_mass`).
    pub eq_initial_bh_mass: f64,
    /// Suppress binary hard limits on the active step (`b% ignore_hard_limits_this_step`).
    pub ignore_hard_limits_this_step: bool,
    /// Pending extra retry reduction factor `dt_reduction_factor_for_j` (`b% have_to_reduce_timestep_due_to_j`).
    pub have_to_reduce_timestep_due_to_j: bool,
    /// Whether the donor hit the rotational implicit wind limit during an implicit RLOF trial (`b% donor_started_implicit_wind`).
    pub donor_started_implicit_wind: bool,
    /// Tidal synchronization timescales `[t_sync_1, t_sync_2]` in seconds (`b% t_sync_1`, `b% t_sync_2`).
    pub t_sync: [f64; 2],
    /// Adaptive implicit RLOF step multiplier (`change_factor`).
    pub change_factor: f64,
    /// Latched magnetic-braking flags `[star_1, star_2]`.
    pub using_jdot_mb: [bool; 2],
    /// Cumulative non-spin-orbit angular momentum removed/added ($\mathrm{g\,cm^2\,s^{-1}}$).
    pub cumulative_j_ext: f64,
    /// Last accepted step wind/RLOF/Eddington accounting.
    pub last_mdot: CoupledMdotAccounting,
    /// Last accepted step orbital torque breakdown.
    pub last_jdot: JdotBreakdown,
    /// Eccentricity-rate terms applied by the most recent orbital update
    /// (`b% edot`, `b% edot_tidal`, `b% edot_enhance`).
    pub last_edot: EdotBreakdown,
    /// Binary or stellar termination reason triggered at the end of the latest step.
    pub last_termination_reason: Option<String>,
    /// Retry, implicit trial, dt-cut, and hard-limit counters preserved across rollbacks.
    pub counters: BinaryStepCounters,
    /// Frozen stellar structure retained after `set_point_mass_i` converts an
    /// evolved star into a compact point mass (`b% s1` / `b% s2` in `binary_utils.f90:67-82`).
    collapsed_star: Option<Box<StarModel>>,
}

impl BinarySystem {
    /// Constructs a new `BinarySystem` with a donor `StarModel` and a point-mass companion.
    #[must_use]
    pub fn new_point_mass(
        donor: StarModel,
        companion_mass_msun: f64,
        initial_period_days: f64,
        eccentricity: f64,
        is_black_hole: bool,
        initial_bh_spin: f64,
    ) -> Self {
        let ctrl = BinaryInlistControls {
            m1_msun: donor.mstar / MSUN,
            m2_msun: companion_mass_msun,
            initial_period_in_days: initial_period_days,
            initial_eccentricity: eccentricity,
            initial_bh_spin: initial_bh_spin.clamp(0.0, 1.0),
            use_radiation_corrected_transfer_rate: is_black_hole,
            limit_retention_by_mdot_edd: is_black_hole,
            mdot_scheme: String::from("roche_lobe"),
            terminate_if_initial_overflow: false,
            ..BinaryInlistControls::default()
        };
        Self::from_controls_point_mass(donor, ctrl)
            .expect("valid BinarySystem::new_point_mass")
    }

    /// Constructs a new `BinarySystem` with two evolved `StarModel` instances.
    #[must_use]
    pub fn new_two_stars(
        donor: StarModel,
        accretor: StarModel,
        initial_period_days: f64,
        eccentricity: f64,
    ) -> Self {
        let ctrl = BinaryInlistControls {
            evolve_both_stars: true,
            m1_msun: donor.mstar / MSUN,
            m2_msun: accretor.mstar / MSUN,
            initial_period_in_days: initial_period_days,
            initial_eccentricity: eccentricity,
            mdot_scheme: String::from("roche_lobe"),
            terminate_if_initial_overflow: false,
            ..BinaryInlistControls::default()
        };
        Self::from_controls_two_stars(donor, accretor, ctrl)
            .expect("valid BinarySystem::new_two_stars")
    }

    /// Constructs a `Donor + PointMass` [`BinarySystem`] from [`BinaryInlistControls`],
    /// honoring `initial_separation_in_rsuns` whenever `initial_period_in_days <= 0`.
    pub fn from_controls_point_mass(
        donor: StarModel,
        controls: BinaryInlistControls,
    ) -> Result<Self, StarError> {
        let m2 = controls.m2_msun * MSUN;
        let m_bh_eq = bh_initial_equatorial_mass(m2, controls.initial_bh_spin);
        let companion = BinaryCompanion::PointMass {
            mass_g: m2,
            m_bh_zero_eq_g: m_bh_eq,
            is_black_hole: controls.use_radiation_corrected_transfer_rate,
        };
        Self::from_controls_inner(donor, companion, 2, 1, controls)
    }

    /// Constructs a two-star [`BinarySystem`] from [`BinaryInlistControls`].
    pub fn from_controls_two_stars(
        donor: StarModel,
        accretor: StarModel,
        controls: BinaryInlistControls,
    ) -> Result<Self, StarError> {
        Self::from_controls_inner(
            donor,
            BinaryCompanion::Star(Box::new(accretor)),
            0,
            1,
            controls,
        )
    }

    fn from_controls_inner(
        mut donor: StarModel,
        mut companion: BinaryCompanion,
        point_mass_i: usize,
        donor_i: usize,
        controls: BinaryInlistControls,
    ) -> Result<Self, StarError> {
        flexmesa_math::math_init();
        let m_d = donor.mstar;
        let m_a = match &companion {
            BinaryCompanion::PointMass { mass_g, .. } => *mass_g,
            BinaryCompanion::Star(s2) => s2.mstar,
        };
        let eccentricity = controls
            .initial_eccentricity
            .clamp(controls.min_eccentricity, controls.max_eccentricity);
        let (separation, period) = if controls.initial_period_in_days > 0.0 {
            let p = controls.initial_period_in_days * SECDAY;
            (separation_from_period(m_d, m_a, p), p)
        } else {
            let a = controls.initial_separation_in_rsuns * RSUN;
            (a, period_from_separation(m_d, m_a, a))
        };

        if controls.do_initial_orbit_sync_1 && point_mass_i != 1 {
            apply_initial_orbit_sync(&mut donor, period)?;
        }
        if controls.do_initial_orbit_sync_2 && point_mass_i != 2 {
            if let BinaryCompanion::Star(ref mut s2) = companion {
                apply_initial_orbit_sync(s2, period)?;
            }
        }
        let r_a_init = match &companion {
            BinaryCompanion::PointMass { .. } => 0.0,
            BinaryCompanion::Star(s2) => s2.r[0],
        };
        let (rl_init, _) = compute_rl_and_gaps(
            m_d,
            m_a,
            donor.r[0],
            r_a_init,
            separation,
            eccentricity,
        );
        if controls.use_tidal_deformation {
            donor.aux.tidal_deformation =
                Some(BinaryTidalDeformationHook::new_handle(
                    m_d,
                    m_a,
                    separation,
                    rl_init[0],
                    period,
                    controls.f_sync_switch_from_rot_defor,
                    controls.f_sync_switch_width,
                    controls.f_sync_switch_lim,
                )?);
            if let BinaryCompanion::Star(ref mut s2) = companion {
                s2.aux.tidal_deformation =
                    Some(BinaryTidalDeformationHook::new_handle(
                        m_a,
                        m_d,
                        separation,
                        rl_init[1],
                        period,
                        controls.f_sync_switch_from_rot_defor,
                        controls.f_sync_switch_width,
                        controls.f_sync_switch_lim,
                    )?);
            }
        }
        prep_and_settle_binary_star(&mut donor)?;
        if let BinaryCompanion::Star(ref mut s2) = companion {
            prep_and_settle_binary_star(s2)?;
        }

        let j_orb = eval_j_orb(m_d, m_a, separation, eccentricity);
        let r_a = match &companion {
            BinaryCompanion::PointMass { .. } => 0.0,
            BinaryCompanion::Star(s2) => star_photosphere_radius_cm(s2),
        };
        let (rl, rl_relative_gap) = compute_rl_and_gaps(
            m_d,
            m_a,
            star_photosphere_radius_cm(&donor),
            r_a,
            separation,
            eccentricity,
        );
        let ignore_rlof = if controls.change_initial_ignore_rlof_flag
            || controls.change_ignore_rlof_flag
        {
            controls.new_ignore_rlof_flag
        } else {
            false
        };

        let eq_bh_init = match &companion {
            BinaryCompanion::PointMass { m_bh_zero_eq_g, .. } => {
                *m_bh_zero_eq_g
            }
            BinaryCompanion::Star(_) => 0.0,
        };
        let init_masses_msun = if donor_i == 1 {
            [donor.mstar / MSUN, m_a / MSUN]
        } else {
            [m_a / MSUN, donor.mstar / MSUN]
        };
        let mut sys = Self {
            donor,
            companion,
            separation,
            period,
            eccentricity,
            angular_momentum_j: j_orb,
            rl,
            rl_relative_gap,
            mtransfer_rate: 0.0,
            step_mtransfer_rate: 0.0,
            initial_mass_msun: init_masses_msun,
            initial_binary_period_sec: period,
            jdot: 0.0,
            binary_age_yr: 0.0,
            model_number: 0,
            max_timestep_sec: 1.0e99,
            mass_transfer_alpha: controls.mass_transfer_alpha,
            mass_transfer_beta: controls.mass_transfer_beta,
            mass_transfer_delta: controls.mass_transfer_delta,
            mass_transfer_gamma: controls.mass_transfer_gamma,
            use_implicit_rlof: controls.max_tries_to_achieve > 0
                && controls.implicit_scheme_tolerance > 0.0,
            change_factor: controls.initial_change_factor,
            controls,
            point_mass_i,
            donor_i,
            ignore_rlof_flag: ignore_rlof,
            eq_initial_bh_mass: eq_bh_init,
            // `binarydata_init` calls `set_{period,separation}_eccentricity` ->
            // `set_angular_momentum_j` (`binary_utils.f90:200`), which sets
            // `ignore_hard_limits_this_step = .true.` for the first step.
            ignore_hard_limits_this_step: true,
            have_to_reduce_timestep_due_to_j: false,
            donor_started_implicit_wind: false,
            t_sync: [0.0; 2],
            using_jdot_mb: [false; 2],
            cumulative_j_ext: 0.0,
            last_mdot: CoupledMdotAccounting::default(),
            last_jdot: JdotBreakdown::default(),
            last_edot: EdotBreakdown::default(),
            last_termination_reason: None,
            counters: BinaryStepCounters::default(),
            collapsed_star: None,
        };
        sys.apply_binary_job_controls_after(false)?;
        sys.refresh_tidal_deformation_hooks(false)?;
        sys.update_stage_varcontrol();
        Ok(sys)
    }

    /// Attaches or updates the `use_tidal_deformation` Roche-potential hooks on
    /// active stellar components (`run_binary_support.f90:269-277`).
    pub fn refresh_tidal_deformation_hooks(
        &mut self,
        recompute_hydro: bool,
    ) -> Result<(), StarError> {
        if !self.controls.use_tidal_deformation {
            self.donor.aux.tidal_deformation = None;
            if let BinaryCompanion::Star(ref mut s2) = self.companion {
                s2.aux.tidal_deformation = None;
            }
            return Ok(());
        }
        let m_d = self.donor.mstar;
        let m_a = self.accretor_mass_g();
        self.donor.aux.tidal_deformation =
            Some(BinaryTidalDeformationHook::new_handle(
                m_d,
                m_a,
                self.separation,
                self.rl[0],
                self.period,
                self.controls.f_sync_switch_from_rot_defor,
                self.controls.f_sync_switch_width,
                self.controls.f_sync_switch_lim,
            )?);
        sync_post_henyey_rotation(&mut self.donor, 0.0);
        if recompute_hydro {
            set_hydro_vars(&mut self.donor, false, false)?;
            self.donor.save_start_values()?;
            self.donor.save_old_values();
        }
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            s2.aux.tidal_deformation =
                Some(BinaryTidalDeformationHook::new_handle(
                    m_a,
                    m_d,
                    self.separation,
                    self.rl[1],
                    self.period,
                    self.controls.f_sync_switch_from_rot_defor,
                    self.controls.f_sync_switch_width,
                    self.controls.f_sync_switch_lim,
                )?);
            sync_post_henyey_rotation(s2, 0.0);
            if recompute_hydro {
                set_hydro_vars(s2, false, false)?;
                s2.save_start_values()?;
                s2.save_old_values();
            }
        }
        Ok(())
    }

    /// Returns the current accretor mass $M_a$ (g).
    #[must_use]
    pub fn accretor_mass_g(&self) -> f64 {
        match &self.companion {
            BinaryCompanion::PointMass { mass_g, .. } => *mass_g,
            BinaryCompanion::Star(s2) => s2.mstar,
        }
    }

    /// Returns the current accretor photospheric radius $R_a$ (cm).
    #[must_use]
    pub fn accretor_radius_cm(&self) -> f64 {
        match &self.companion {
            BinaryCompanion::PointMass { .. } => 0.0,
            BinaryCompanion::Star(s2) => star_photosphere_radius_cm(s2),
        }
    }

    /// Returns `[m(1), m(2)]` in grams ordered by star index (`1` and `2`).
    #[must_use]
    pub fn star_masses_g(&self) -> [f64; 2] {
        if self.donor_i == 1 {
            [self.donor.mstar, self.accretor_mass_g()]
        } else {
            [self.accretor_mass_g(), self.donor.mstar]
        }
    }

    /// Returns `[rl(1), rl(2)]` in cm ordered by star index (`1` and `2`).
    #[must_use]
    pub fn rl_by_star(&self) -> [f64; 2] {
        if self.donor_i == 1 {
            self.rl
        } else {
            [self.rl[1], self.rl[0]]
        }
    }

    /// Returns `[rl_relative_gap(1), rl_relative_gap(2)]` ordered by star index.
    #[must_use]
    pub fn rl_relative_gap_by_star(&self) -> [f64; 2] {
        if self.donor_i == 1 {
            self.rl_relative_gap
        } else {
            [self.rl_relative_gap[1], self.rl_relative_gap[0]]
        }
    }

    /// Returns a reference to star `idx` (`1` or `2`) when that component is an evolved star.
    #[must_use]
    pub fn star_ref(&self, idx: usize) -> Option<&StarModel> {
        if self.point_mass_i == idx {
            return None;
        }
        if self.donor_i == idx {
            Some(&self.donor)
        } else if let BinaryCompanion::Star(ref s2) = self.companion {
            Some(s2)
        } else {
            None
        }
    }

    /// Returns a reference to star slot `idx` (`1` or `2`), including a star
    /// converted into a compact point mass via [`Self::set_point_mass_i`]
    /// (`b% s1` / `b% s2` in `binary_utils.f90:67-82`).
    #[must_use]
    pub fn star_slot_ref(&self, idx: usize) -> Option<&StarModel> {
        if self.point_mass_i == idx {
            self.collapsed_star.as_deref()
        } else {
            self.star_ref(idx)
        }
    }

    /// Returns conserved spin-orbit quantity $J_{\mathrm{orb}} + S_1 + S_2 - J_{\mathrm{ext}}$.
    #[must_use]
    pub fn total_angular_momentum_conserved(&self) -> f64 {
        let s_d = eval_star_spin_j(&self.donor);
        let s_a = match &self.companion {
            BinaryCompanion::PointMass { .. } => 0.0,
            BinaryCompanion::Star(s2) => eval_star_spin_j(s2),
        };
        self.angular_momentum_j + s_d + s_a - self.cumulative_j_ext
    }

    /// Updates `ignore_rlof_flag` (`binary_set_ignore_rlof_flag`).
    pub fn set_ignore_rlof_flag(&mut self, flag: bool) {
        self.ignore_rlof_flag = flag;
        if flag {
            self.mtransfer_rate = 0.0;
        }
    }

    /// Updates `eq_initial_bh_mass` in grams (`b% eq_initial_bh_mass`), keeping
    /// any active `PointMass` companion's `m_bh_zero_eq_g` synchronized.
    pub fn set_eq_initial_bh_mass(&mut self, eq_mass_g: f64) {
        self.eq_initial_bh_mass = eq_mass_g;
        if let BinaryCompanion::PointMass {
            ref mut m_bh_zero_eq_g,
            ..
        } = self.companion
        {
            *m_bh_zero_eq_g = eq_mass_g;
        }
    }

    /// Updates orbital period `period_sec` (s) and `eccentricity`, recomputing
    /// $a$, $J_{\mathrm{orb}}$, and $R_L$, and setting `ignore_hard_limits_this_step`
    /// (`set_period_eccentricity` -> `set_angular_momentum_j` in `binary_utils.f90:146-202`).
    pub fn set_period_eccentricity(
        &mut self,
        period_sec: f64,
        eccentricity: f64,
    ) -> Result<(), StarError> {
        let (m_d, m_a) = (self.donor.mstar, self.accretor_mass_g());
        self.period = period_sec;
        self.eccentricity = eccentricity.clamp(
            self.controls.min_eccentricity,
            self.controls.max_eccentricity,
        );
        self.separation = separation_from_period(m_d, m_a, self.period);
        self.angular_momentum_j =
            eval_j_orb(m_d, m_a, self.separation, self.eccentricity);
        let (rl, gaps) = compute_rl_and_gaps(
            m_d,
            m_a,
            star_photosphere_radius_cm(&self.donor),
            self.accretor_radius_cm(),
            self.separation,
            self.eccentricity,
        );
        self.rl = rl;
        self.rl_relative_gap = gaps;
        self.ignore_hard_limits_this_step = true;
        self.refresh_tidal_deformation_hooks(false)
    }

    /// Updates orbital separation `separation_cm` (cm) and `eccentricity`,
    /// recomputing $P$, $J_{\mathrm{orb}}$, and $R_L$, and setting
    /// `ignore_hard_limits_this_step` (`set_separation_eccentricity` in `binary_utils.f90:163-202`).
    pub fn set_separation_eccentricity(
        &mut self,
        separation_cm: f64,
        eccentricity: f64,
    ) -> Result<(), StarError> {
        let (m_d, m_a) = (self.donor.mstar, self.accretor_mass_g());
        self.separation = separation_cm;
        self.eccentricity = eccentricity.clamp(
            self.controls.min_eccentricity,
            self.controls.max_eccentricity,
        );
        self.period = period_from_separation(m_d, m_a, self.separation);
        self.angular_momentum_j =
            eval_j_orb(m_d, m_a, self.separation, self.eccentricity);
        let (rl, gaps) = compute_rl_and_gaps(
            m_d,
            m_a,
            star_photosphere_radius_cm(&self.donor),
            self.accretor_radius_cm(),
            self.separation,
            self.eccentricity,
        );
        self.rl = rl;
        self.rl_relative_gap = gaps;
        self.ignore_hard_limits_this_step = true;
        self.refresh_tidal_deformation_hooks(false)
    }

    /// Updates point-mass primary mass `m1_msun` ($M_\odot$) when `point_mass_i == 1`
    /// (`set_m1` in `binary_utils.f90:105-124`).
    pub fn set_m1(&mut self, m1_msun: f64) -> Result<(), StarError> {
        if self.point_mass_i != 1 {
            return Err(StarError::InvalidInput(String::from(
                "set_m1 requires point_mass_i == 1",
            )));
        }
        if let BinaryCompanion::PointMass { ref mut mass_g, .. } =
            self.companion
        {
            *mass_g = m1_msun * MSUN;
        }
        self.set_separation_eccentricity(self.separation, self.eccentricity)
    }

    /// Updates point-mass secondary mass `m2_msun` ($M_\odot$) when `point_mass_i == 2`
    /// (`set_m2` in `binary_utils.f90:126-144`).
    pub fn set_m2(&mut self, m2_msun: f64) -> Result<(), StarError> {
        if self.point_mass_i != 2 {
            return Err(StarError::InvalidInput(String::from(
                "set_m2 requires point_mass_i == 2",
            )));
        }
        if let BinaryCompanion::PointMass { ref mut mass_g, .. } =
            self.companion
        {
            *mass_g = m2_msun * MSUN;
        }
        self.set_separation_eccentricity(self.separation, self.eccentricity)
    }

    /// Applies `&binary_job` `change_*` / `change_initial_*` overrides
    /// (`do_binary_job_controls_after` in `run_binary_support.f90:894-1014`).
    pub fn apply_binary_job_controls_after(
        &mut self,
        restart: bool,
    ) -> Result<(), StarError> {
        let c = self.controls.clone();
        if c.change_ignore_rlof_flag
            || (c.change_initial_ignore_rlof_flag && !restart)
        {
            self.set_ignore_rlof_flag(c.new_ignore_rlof_flag);
        }
        if (c.change_point_mass_i
            || (c.change_initial_point_mass_i && !restart))
            && usize::try_from(c.new_point_mass_i).unwrap_or(self.point_mass_i)
                != self.point_mass_i
        {
            let new_pm = usize::try_from(c.new_point_mass_i).map_err(|_| {
                StarError::InvalidInput(format!(
                    "invalid new_point_mass_i {}",
                    c.new_point_mass_i
                ))
            })?;
            self.set_point_mass_i(new_pm)?;
        }
        if (c.change_m1 || (c.change_initial_m1 && !restart))
            && self.star_masses_g()[0] != c.new_m1
        {
            self.set_m1(c.new_m1)?;
        }
        if (c.change_m2 || (c.change_initial_m2 && !restart))
            && self.star_masses_g()[1] != c.new_m2
        {
            self.set_m2(c.new_m2)?;
        }
        if (c.change_period_eccentricity
            || (c.change_initial_period_eccentricity && !restart))
            && (self.period != c.new_period * SECDAY
                || self.eccentricity != c.new_eccentricity)
        {
            self.set_period_eccentricity(
                c.new_period * SECDAY,
                c.new_eccentricity,
            )?;
        }
        if (c.change_separation_eccentricity
            || (c.change_initial_separation_eccentricity && !restart))
            && (self.separation != c.new_separation * RSUN
                || self.eccentricity != c.new_eccentricity)
        {
            self.set_separation_eccentricity(
                c.new_separation * RSUN,
                c.new_eccentricity,
            )?;
        }
        Ok(())
    }

    /// Dynamically converts star `new_point_mass_i` (`1` or `2`) into a compact
    /// point mass (`set_point_mass_i` in `binary_utils.f90:47-82`).
    pub fn set_point_mass_i(
        &mut self,
        new_point_mass_i: usize,
    ) -> Result<(), StarError> {
        if new_point_mass_i == self.point_mass_i {
            return Ok(());
        }
        if new_point_mass_i == 0 || new_point_mass_i > 2 {
            return Err(StarError::InvalidInput(format!(
                "set_point_mass_i requires 1 or 2, got {new_point_mass_i}"
            )));
        }
        if self.point_mass_i != 0 {
            return Err(StarError::InvalidInput(format!(
                "cannot set point_mass_i={new_point_mass_i} when star {} is already a point mass",
                self.point_mass_i
            )));
        }
        let pm_mass = self.star_masses_g()[new_point_mass_i - 1];
        let m_bh_eq =
            bh_initial_equatorial_mass(pm_mass, self.controls.initial_bh_spin);
        let is_bh = self.controls.use_radiation_corrected_transfer_rate;
        let pm = BinaryCompanion::PointMass {
            mass_g: pm_mass,
            m_bh_zero_eq_g: m_bh_eq,
            is_black_hole: is_bh,
        };
        if self.donor_i == new_point_mass_i {
            if let BinaryCompanion::Star(other_star) =
                std::mem::replace(&mut self.companion, pm)
            {
                let old_donor = std::mem::replace(&mut self.donor, *other_star);
                self.collapsed_star = Some(Box::new(old_donor));
            }
            self.donor_i = 3 - new_point_mass_i;
            self.mtransfer_rate = 0.0;
            self.step_mtransfer_rate = 0.0;
            self.change_factor = self.controls.max_change_factor;
        } else if let BinaryCompanion::Star(old_comp) =
            std::mem::replace(&mut self.companion, pm)
        {
            self.collapsed_star = Some(old_comp);
        }
        self.point_mass_i = new_point_mass_i;
        self.eq_initial_bh_mass = m_bh_eq;
        self.last_termination_reason = None;
        let (rl, gaps) = compute_rl_and_gaps(
            self.donor.mstar,
            self.accretor_mass_g(),
            star_photosphere_radius_cm(&self.donor),
            0.0,
            self.separation,
            self.eccentricity,
        );
        self.rl = rl;
        self.rl_relative_gap = gaps;
        self.refresh_tidal_deformation_hooks(false)?;
        self.update_stage_varcontrol();
        Ok(())
    }

    /// Checks whether the companion star overfills its Roche lobe more deeply
    /// than the current donor and swaps donor/accretor roles (`binary_evolve.f90:210-255`).
    pub fn check_donor_switch(
        &mut self,
        keep_donor_fixed: bool,
    ) -> Result<bool, StarError> {
        if keep_donor_fixed
            || self.point_mass_i != 0
            || self.rl_relative_gap[1] <= self.rl_relative_gap[0]
        {
            return Ok(false);
        }
        self.swap_donor_and_accretor()
    }

    pub(crate) fn swap_donor_and_accretor(
        &mut self,
    ) -> Result<bool, StarError> {
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            if self.controls.report_rlo_solver_progress {
                println!(" switching donor");
            }
            std::mem::swap(&mut self.donor, s2);
            self.donor_i = 3 - self.donor_i;
            self.rl.swap(0, 1);
            self.rl_relative_gap.swap(0, 1);
            self.refresh_tidal_deformation_hooks(false)?;
            return Ok(true);
        }
        Ok(false)
    }

    pub(crate) fn maybe_switch_donor_at_step_start(
        &mut self,
    ) -> Result<(), StarError> {
        if self.point_mass_i != 0 || self.ignore_rlof_flag {
            return Ok(());
        }
        let is_contact =
            self.controls.mdot_scheme.eq_ignore_ascii_case("contact");
        if self.controls.keep_donor_fixed && !is_contact {
            return Ok(());
        }
        if self.mtransfer_rate.abs() * SECYER / MSUN
            >= self.controls.mdot_limit_donor_switch
        {
            return Ok(());
        }
        let tol = self.controls.implicit_scheme_tolerance;
        if (self.controls.mdot_scheme.eq_ignore_ascii_case("roche_lobe")
            && self.rl_relative_gap[1] > self.rl_relative_gap[0])
            || (is_contact
                && self.rl_relative_gap[1] > self.rl_relative_gap[0]
                && self.rl_relative_gap[1] < -tol
                && self.rl_relative_gap[0] < -tol)
        {
            let _ = self.swap_donor_and_accretor()?;
        }
        Ok(())
    }

    fn snapshot_parts(&self) -> (Option<&StarModel>, BinaryStepScalars) {
        let (comp_star, point_mass_state) = match &self.companion {
            BinaryCompanion::PointMass {
                mass_g,
                m_bh_zero_eq_g,
                is_black_hole,
            } => (None, Some((*mass_g, *m_bh_zero_eq_g, *is_black_hole))),
            BinaryCompanion::Star(s2) => (Some(&**s2), None),
        };
        let scalars = BinaryStepScalars {
            point_mass_state,
            point_mass_i: self.point_mass_i,
            donor_i: self.donor_i,
            ignore_rlof_flag: self.ignore_rlof_flag,
            eq_initial_bh_mass: self.eq_initial_bh_mass,
            ignore_hard_limits_this_step: self.ignore_hard_limits_this_step,
            donor_irradiation_flux: self.donor.controls.irradiation_flux,
            donor_column_depth_for_irradiation: self
                .donor
                .controls
                .column_depth_for_irradiation,
            t_sync: self.t_sync,
            separation: self.separation,
            period: self.period,
            eccentricity: self.eccentricity,
            angular_momentum_j: self.angular_momentum_j,
            rl: self.rl,
            rl_relative_gap: self.rl_relative_gap,
            mtransfer_rate: self.mtransfer_rate,
            step_mtransfer_rate: self.step_mtransfer_rate,
            jdot: self.jdot,
            binary_age_yr: self.binary_age_yr,
            model_number: self.model_number,
            max_timestep_sec: self.max_timestep_sec,
            change_factor: self.change_factor,
            using_jdot_mb: self.using_jdot_mb,
            cumulative_j_ext: self.cumulative_j_ext,
            last_mdot: self.last_mdot,
            last_jdot: self.last_jdot,
            last_edot: self.last_edot,
        };
        (comp_star, scalars)
    }

    pub(crate) fn capture_snapshot(&self) -> BinaryStepSnapshot {
        let (comp_star, scalars) = self.snapshot_parts();
        BinaryStepSnapshot::capture(&self.donor, comp_star, scalars)
    }

    pub(crate) fn capture_snapshot_into(&self, snap: &mut BinaryStepSnapshot) {
        let (comp_star, scalars) = self.snapshot_parts();
        snap.capture_into(&self.donor, comp_star, scalars);
    }

    pub(crate) fn restore_snapshot(
        &mut self,
        snap: &BinaryStepSnapshot,
    ) -> Result<(), StarError> {
        if snap.point_mass_i == 0
            && matches!(self.companion, BinaryCompanion::PointMass { .. })
            && snap.companion_state.is_some()
        {
            let Some(restored_box) = self.collapsed_star.take() else {
                return Err(StarError::InvalidInput(String::from(
                    "cannot restore two-star snapshot after point-mass conversion without retained collapsed_star",
                )));
            };
            self.companion = BinaryCompanion::Star(restored_box);
        }
        if self.donor_i != snap.donor_i {
            if let BinaryCompanion::Star(ref mut s2) = self.companion {
                std::mem::swap(&mut self.donor, s2);
            }
        }
        self.point_mass_i = snap.point_mass_i;
        self.donor_i = snap.donor_i;
        self.ignore_rlof_flag = snap.ignore_rlof_flag;
        self.eq_initial_bh_mass = snap.eq_initial_bh_mass;
        self.ignore_hard_limits_this_step = snap.ignore_hard_limits_this_step;
        self.donor.controls.irradiation_flux = snap.donor_irradiation_flux;
        self.donor.controls.column_depth_for_irradiation =
            snap.donor_column_depth_for_irradiation;
        self.t_sync = snap.t_sync;
        self.separation = snap.separation;
        self.period = snap.period;
        self.eccentricity = snap.eccentricity;
        self.angular_momentum_j = snap.angular_momentum_j;
        self.rl = snap.rl;
        self.rl_relative_gap = snap.rl_relative_gap;
        self.mtransfer_rate = snap.mtransfer_rate;
        self.step_mtransfer_rate = snap.step_mtransfer_rate;
        self.jdot = snap.jdot;
        self.binary_age_yr = snap.binary_age_yr;
        self.model_number = snap.model_number;
        self.max_timestep_sec = snap.max_timestep_sec;
        self.change_factor = snap.change_factor;
        self.using_jdot_mb = snap.using_jdot_mb;
        self.cumulative_j_ext = snap.cumulative_j_ext;
        self.last_mdot = snap.last_mdot;
        self.last_jdot = snap.last_jdot;
        self.last_edot = snap.last_edot;
        let comp_mut = match &mut self.companion {
            BinaryCompanion::PointMass {
                mass_g,
                m_bh_zero_eq_g,
                is_black_hole,
            } => {
                if let Some((m, meq, bh)) = snap.point_mass_state {
                    *mass_g = m;
                    *m_bh_zero_eq_g = meq;
                    *is_black_hole = bh;
                }
                None
            }
            BinaryCompanion::Star(s2) => Some(&mut **s2),
        };
        snap.restore_stars(&mut self.donor, comp_mut)?;
        self.refresh_tidal_deformation_hooks(false)?;
        Ok(())
    }

    /// Returns the recorded `NumericsMode` from serialized `BinarySystem`
    /// checkpoint bytes without decoding the payload.
    ///
    /// # Errors
    /// Returns [`StarError::IoError`] if the header is invalid or truncated.
    pub fn checkpoint_numerics_mode(
        bytes: &[u8],
    ) -> Result<flexmesa_math::NumericsMode, StarError> {
        decode_binary_checkpoint_frame(bytes).map(|(mode, _)| mode)
    }

    /// Serializes the full coupled [`BinarySystem`] (both stars, controls,
    /// orbit, and retry-proof counters) into deterministic binary checkpoint bytes.
    ///
    /// # Errors
    /// Returns [`StarError::IoError`] if `serde` binary encoding fails.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, StarError> {
        let payload =
            flexmesa_utils::bin_codec::to_bytes(self).map_err(|e| {
                StarError::IoError(format!(
                    "binary checkpoint serialization failed: {e}"
                ))
            })?;
        Ok(flexmesa_utils::bin_codec::encode_framed_bytes(
            BINARY_CHECKPOINT_MAGIC,
            BINARY_CHECKPOINT_VERSION,
            flexmesa_math::numerics_mode().to_wire_tag(),
            &payload,
        ))
    }

    /// Restores a coupled [`BinarySystem`] from serialized binary checkpoint
    /// bytes and reattaches `use_tidal_deformation` hooks without re-normalizing
    /// stored `dq` arrays.
    ///
    /// # Errors
    /// Returns [`StarError::IoError`] if header validation fails,
    /// `numerics_mode` mismatches, or decoding fails.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, StarError> {
        let (saved_mode, payload) = decode_binary_checkpoint_frame(bytes)?;
        let active_mode = flexmesa_math::numerics_mode();
        if saved_mode != active_mode {
            return Err(StarError::IoError(format!(
                "binary checkpoint numerics_mode mismatch: checkpoint saved in \
                 \"{saved_mode}\" mode cannot be loaded while active mode is \
                 \"{active_mode}\""
            )));
        }
        let mut sys: Self = flexmesa_utils::bin_codec::from_bytes(payload)
            .map_err(|e| {
                StarError::IoError(format!(
                    "binary checkpoint decode failed: {e}"
                ))
            })?;
        sys.refresh_tidal_deformation_hooks(false)?;
        Ok(sys)
    }

    /// Writes a binary checkpoint file to `path` on disk.
    ///
    /// # Errors
    /// Returns [`StarError::IoError`] if encoding or writing fails.
    pub fn save_checkpoint(
        &self,
        path: &std::path::Path,
    ) -> Result<(), StarError> {
        let bytes = self.to_checkpoint_bytes()?;
        std::fs::write(path, bytes).map_err(|e| {
            StarError::IoError(format!(
                "failed to write binary checkpoint '{}': {e}",
                path.display()
            ))
        })
    }

    /// Reads and restores a [`BinarySystem`] from a checkpoint file on disk.
    ///
    /// # Errors
    /// Returns [`StarError::IoError`] if reading or decoding fails.
    pub fn load_checkpoint(path: &std::path::Path) -> Result<Self, StarError> {
        let bytes = std::fs::read(path).map_err(|e| {
            StarError::IoError(format!(
                "failed to read binary checkpoint '{}': {e}",
                path.display()
            ))
        })?;
        Self::from_checkpoint_bytes(&bytes)
    }
}

const BINARY_CHECKPOINT_MAGIC: &[u8; 8] = b"FMBINCP1";
const BINARY_CHECKPOINT_VERSION: u32 = 2;

fn decode_binary_checkpoint_frame(
    bytes: &[u8],
) -> Result<(flexmesa_math::NumericsMode, &[u8]), StarError> {
    let (mode_tag, payload) = flexmesa_utils::bin_codec::decode_framed_header(
        bytes,
        BINARY_CHECKPOINT_MAGIC,
        BINARY_CHECKPOINT_VERSION,
        "binary checkpoint",
    )
    .map_err(StarError::IoError)?;
    let mode = flexmesa_math::NumericsMode::from_wire_tag(mode_tag)
        .ok_or_else(|| {
            StarError::IoError(format!(
                "invalid binary checkpoint numerics_mode tag {mode_tag}"
            ))
        })?;
    Ok((mode, payload))
}
