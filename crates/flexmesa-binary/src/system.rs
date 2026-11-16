// SPDX-License-Identifier: LGPL-3.0-or-later
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
use crate::system_jdot::{EdotBreakdown, JdotBreakdown, apply_initial_orbit_sync, eval_star_spin_j};
use crate::system_rlof::{BinaryStepSnapshot, CoupledMdotAccounting};

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
#[derive(Debug, Clone)]
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

/// Coupled binary star evolution state and step driver (`binary_info`).
#[derive(Debug, Clone)]
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
        Self::from_controls_point_mass(donor, ctrl).expect("valid BinarySystem::new_point_mass")
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
        Self::from_controls_inner(donor, BinaryCompanion::Star(Box::new(accretor)), 0, 1, controls)
    }

    fn from_controls_inner(
        mut donor: StarModel,
        mut companion: BinaryCompanion,
        point_mass_i: usize,
        donor_i: usize,
        controls: BinaryInlistControls,
    ) -> Result<Self, StarError> {
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
            m_d, m_a, donor.r[0], r_a_init, separation, eccentricity,
        );
        if controls.use_tidal_deformation {
            donor.aux.tidal_deformation = BinaryTidalDeformationHook::new_handle(
                m_d,
                m_a,
                separation,
                rl_init[0],
                period,
                controls.f_sync_switch_from_rot_defor,
                controls.f_sync_switch_width,
                controls.f_sync_switch_lim,
            );
            if let BinaryCompanion::Star(ref mut s2) = companion {
                s2.aux.tidal_deformation = BinaryTidalDeformationHook::new_handle(
                    m_a,
                    m_d,
                    separation,
                    rl_init[1],
                    period,
                    controls.f_sync_switch_from_rot_defor,
                    controls.f_sync_switch_width,
                    controls.f_sync_switch_lim,
                );
            }
        }
        prep_and_settle_binary_star(&mut donor)?;
        if let BinaryCompanion::Star(ref mut s2) = companion {
            prep_and_settle_binary_star(s2)?;
        }

        let j_orb = eval_j_orb(m_d, m_a, separation, eccentricity);
        let r_a = match &companion {
            BinaryCompanion::PointMass { .. } => 0.0,
            BinaryCompanion::Star(s2) => s2.r[0],
        };
        let (rl, rl_relative_gap) =
            compute_rl_and_gaps(m_d, m_a, donor.r[0], r_a, separation, eccentricity);
        let ignore_rlof = if controls.change_initial_ignore_rlof_flag
            || controls.change_ignore_rlof_flag
        {
            controls.new_ignore_rlof_flag
        } else {
            false
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
            use_implicit_rlof: controls.max_tries_to_achieve > 0,
            change_factor: controls.initial_change_factor,
            controls,
            point_mass_i,
            donor_i,
            ignore_rlof_flag: ignore_rlof,
            using_jdot_mb: [false; 2],
            cumulative_j_ext: 0.0,
            last_mdot: CoupledMdotAccounting::default(),
            last_jdot: JdotBreakdown::default(),
            last_edot: EdotBreakdown::default(),
            last_termination_reason: None,
        };
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
            BinaryTidalDeformationHook::new_handle(
                m_d,
                m_a,
                self.separation,
                self.rl[0],
                self.period,
                self.controls.f_sync_switch_from_rot_defor,
                self.controls.f_sync_switch_width,
                self.controls.f_sync_switch_lim,
            );
        sync_post_henyey_rotation(&mut self.donor, 0.0);
        if recompute_hydro {
            set_hydro_vars(&mut self.donor, false, false)?;
            self.donor.save_start_values();
            self.donor.save_old_values();
        }
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            s2.aux.tidal_deformation = BinaryTidalDeformationHook::new_handle(
                m_a,
                m_d,
                self.separation,
                self.rl[1],
                self.period,
                self.controls.f_sync_switch_from_rot_defor,
                self.controls.f_sync_switch_width,
                self.controls.f_sync_switch_lim,
            );
            sync_post_henyey_rotation(s2, 0.0);
            if recompute_hydro {
                set_hydro_vars(s2, false, false)?;
                s2.save_start_values();
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
            BinaryCompanion::Star(s2) => s2.r[0],
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

    /// Updates orbital period `period_sec` (s) and `eccentricity`, recomputing $a$, $J_{\mathrm{orb}}$, and $R_L$.
    pub fn set_period_eccentricity(&mut self, period_sec: f64, eccentricity: f64) {
        let (m_d, m_a) = (self.donor.mstar, self.accretor_mass_g());
        self.period = period_sec;
        self.eccentricity = eccentricity
            .clamp(self.controls.min_eccentricity, self.controls.max_eccentricity);
        self.separation = separation_from_period(m_d, m_a, self.period);
        self.angular_momentum_j = eval_j_orb(m_d, m_a, self.separation, self.eccentricity);
        let (rl, gaps) = compute_rl_and_gaps(
            m_d, m_a, self.donor.r[0], self.accretor_radius_cm(), self.separation, self.eccentricity,
        );
        self.rl = rl;
        self.rl_relative_gap = gaps;
        let _ = self.refresh_tidal_deformation_hooks(false);
    }

    /// Dynamically converts star `new_point_mass_i` (`1` or `2`) into a compact
    /// point mass (`binary_set_point_mass_i` in `binary_evolve.f90:140-200`).
    pub fn set_point_mass_i(&mut self, new_point_mass_i: usize) {
        if new_point_mass_i == self.point_mass_i || new_point_mass_i == 0 || new_point_mass_i > 2 {
            return;
        }
        let pm_mass = self.star_masses_g()[new_point_mass_i - 1];
        let m_bh_eq = bh_initial_equatorial_mass(pm_mass, self.controls.initial_bh_spin);
        let is_bh = self.controls.use_radiation_corrected_transfer_rate;
        let pm = BinaryCompanion::PointMass {
            mass_g: pm_mass,
            m_bh_zero_eq_g: m_bh_eq,
            is_black_hole: is_bh,
        };
        if self.donor_i == new_point_mass_i {
            if let BinaryCompanion::Star(other_star) = std::mem::replace(&mut self.companion, pm) {
                self.donor = *other_star;
            }
        } else {
            self.companion = pm;
        }
        self.point_mass_i = new_point_mass_i;
        self.donor_i = 3 - new_point_mass_i;
        self.mtransfer_rate = 0.0;
        self.step_mtransfer_rate = 0.0;
        self.change_factor = self.controls.max_change_factor;
        self.last_mdot = CoupledMdotAccounting::default();
        self.last_jdot = JdotBreakdown::default();
        self.last_edot = EdotBreakdown::default();
        self.last_termination_reason = None;
        let (rl, gaps) = compute_rl_and_gaps(
            self.donor.mstar, self.accretor_mass_g(), self.donor.r[0], 0.0,
            self.separation, self.eccentricity,
        );
        self.rl = rl;
        self.rl_relative_gap = gaps;
        let _ = self.refresh_tidal_deformation_hooks(false);
        self.update_stage_varcontrol();
    }

    /// Checks whether the companion star overfills its Roche lobe more deeply
    /// than the current donor and swaps donor/accretor roles (`binary_evolve.f90:210-255`).
    pub fn check_donor_switch(&mut self, keep_donor_fixed: bool) -> bool {
        if keep_donor_fixed
            || self.point_mass_i != 0
            || self.rl_relative_gap[1] <= self.rl_relative_gap[0]
        {
            return false;
        }
        self.swap_donor_and_accretor()
    }

    pub(crate) fn swap_donor_and_accretor(&mut self) -> bool {
        if let BinaryCompanion::Star(ref mut s2) = self.companion {
            std::mem::swap(&mut self.donor, s2);
            self.donor_i = 3 - self.donor_i;
            self.rl.swap(0, 1);
            self.rl_relative_gap.swap(0, 1);
            let _ = self.refresh_tidal_deformation_hooks(false);
            return true;
        }
        false
    }

    pub(crate) fn maybe_switch_donor_at_step_start(&mut self) {
        if self.point_mass_i != 0 || self.ignore_rlof_flag {
            return;
        }
        let is_contact = self.controls.mdot_scheme.eq_ignore_ascii_case("contact");
        if self.controls.keep_donor_fixed && !is_contact {
            return;
        }
        if self.mtransfer_rate.abs() * SECYER / MSUN >= self.controls.mdot_limit_donor_switch {
            return;
        }
        let tol = self.controls.implicit_scheme_tolerance;
        if (self.controls.mdot_scheme.eq_ignore_ascii_case("roche_lobe")
            && self.rl_relative_gap[1] > self.rl_relative_gap[0])
            || (is_contact
                && self.rl_relative_gap[1] > self.rl_relative_gap[0]
                && self.rl_relative_gap[1] < -tol
                && self.rl_relative_gap[0] < -tol)
        {
            let _ = self.swap_donor_and_accretor();
        }
    }

    pub(crate) fn capture_snapshot(&self) -> BinaryStepSnapshot {
        let (comp_star, pm_state) = match &self.companion {
            BinaryCompanion::PointMass { mass_g, m_bh_zero_eq_g, is_black_hole } => {
                (None, Some((*mass_g, *m_bh_zero_eq_g, *is_black_hole)))
            }
            BinaryCompanion::Star(s2) => (Some(&**s2), None),
        };
        BinaryStepSnapshot::capture(
            &self.donor, comp_star, pm_state, self.point_mass_i, self.donor_i,
            self.separation, self.period, self.eccentricity, self.angular_momentum_j,
            self.rl, self.rl_relative_gap, self.mtransfer_rate, self.step_mtransfer_rate,
            self.jdot, self.binary_age_yr, self.model_number, self.max_timestep_sec,
            self.change_factor, self.using_jdot_mb, self.cumulative_j_ext,
            self.last_mdot, self.last_jdot, self.last_edot,
        )
    }

    pub(crate) fn restore_snapshot(&mut self, snap: &BinaryStepSnapshot) -> Result<(), StarError> {
        let comp_mut = match &mut self.companion {
            BinaryCompanion::PointMass { mass_g, m_bh_zero_eq_g, is_black_hole } => {
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
        self.point_mass_i = snap.point_mass_i;
        self.donor_i = snap.donor_i;
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
        let _ = self.refresh_tidal_deformation_hooks(false);
        Ok(())
    }
}
