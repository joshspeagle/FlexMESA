// SPDX-License-Identifier: LGPL-3.0-or-later
//! Coupled `BinarySystem` evolution driver (`Donor + PointMass` or
//! `Donor + Accretor`).
//!
//! Corresponds to `binary/private/binary_evolve.f90` and
//! `binary/private/run_binary_support.f90`.

use flexmesa_const::{MSUN, SECDAY, SECYER};
use flexmesa_star::StarModel;

use crate::edd_wind::{bh_initial_equatorial_mass, eval_mdot_edd};
use crate::jdot_tides::{
    eval_edot_gr, eval_jdot_gr, eval_jdot_mb, eval_jdot_ml,
};
use crate::mdot::{
    eval_kolb_mdot_thick, eval_ritter_mdot_thin, ImplicitRlofSolver,
};
use crate::orbit::{
    eval_j_orb, eval_rl_relative_gap, eval_rlobe, period_from_separation,
    separation_from_j_orb, separation_from_period,
};
use crate::timestep::{binary_pick_next_timestep, BinaryTimestepControls};

#[inline]
fn compute_rl_and_gaps(
    m1: f64,
    m2: f64,
    r1: f64,
    r2: f64,
    separation: f64,
    eccentricity: f64,
) -> ([f64; 2], [f64; 2]) {
    let rl1 = eval_rlobe(m1, m2, separation);
    let rl2 = eval_rlobe(m2, m1, separation);
    let gap1 = eval_rl_relative_gap(r1, rl1, eccentricity);
    let gap2 = eval_rl_relative_gap(r2, rl2, eccentricity);
    ([rl1, rl2], [gap1, gap2])
}

/// Companion representation in a coupled [`BinarySystem`].
#[derive(Debug, Clone)]
pub enum BinaryCompanion {
    /// Point-mass compact object (white dwarf, neutron star, or Kerr black hole).
    PointMass {
        /// Current mass $M_2$ (g).
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
    /// Donor star model (`s_donor`).
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
}

impl BinarySystem {
    /// Constructs a new `BinarySystem` with a donor `StarModel` and a point-mass
    /// companion of `companion_mass_msun` ($M_\odot$) at initial orbital period
    /// `initial_period_days` (days) and `eccentricity`.
    #[must_use]
    pub fn new_point_mass(
        donor: StarModel,
        companion_mass_msun: f64,
        initial_period_days: f64,
        eccentricity: f64,
        is_black_hole: bool,
        initial_bh_spin: f64,
    ) -> Self {
        let m2 = companion_mass_msun * MSUN;
        let m_bh_eq = bh_initial_equatorial_mass(m2, initial_bh_spin);
        Self::from_components(
            donor,
            BinaryCompanion::PointMass {
                mass_g: m2,
                m_bh_zero_eq_g: m_bh_eq,
                is_black_hole,
            },
            initial_period_days,
            eccentricity,
        )
    }

    /// Constructs a new `BinarySystem` with two evolved `StarModel` instances.
    #[must_use]
    pub fn new_two_stars(
        donor: StarModel,
        accretor: StarModel,
        initial_period_days: f64,
        eccentricity: f64,
    ) -> Self {
        Self::from_components(
            donor,
            BinaryCompanion::Star(Box::new(accretor)),
            initial_period_days,
            eccentricity,
        )
    }

    fn from_components(
        donor: StarModel,
        companion: BinaryCompanion,
        initial_period_days: f64,
        eccentricity: f64,
    ) -> Self {
        let m1 = donor.mstar;
        let m2 = match &companion {
            BinaryCompanion::PointMass { mass_g, .. } => *mass_g,
            BinaryCompanion::Star(s2) => s2.mstar,
        };
        let period = initial_period_days * SECDAY;
        let separation = separation_from_period(m1, m2, period);
        let j_orb = eval_j_orb(m1, m2, separation, eccentricity);
        let r1 = donor.r[0];
        let r2 = match &companion {
            BinaryCompanion::PointMass { .. } => 0.0,
            BinaryCompanion::Star(s2) => s2.r[0],
        };
        let (rl, rl_relative_gap) =
            compute_rl_and_gaps(m1, m2, r1, r2, separation, eccentricity);

        Self {
            donor,
            companion,
            separation,
            period,
            eccentricity,
            angular_momentum_j: j_orb,
            rl,
            rl_relative_gap,
            mtransfer_rate: 0.0,
            jdot: 0.0,
            binary_age_yr: 0.0,
            model_number: 0,
            max_timestep_sec: 1.0e14,
            mass_transfer_alpha: 0.0,
            mass_transfer_beta: 0.0,
            mass_transfer_delta: 0.0,
            mass_transfer_gamma: 0.0,
            use_implicit_rlof: true,
        }
    }

    /// Returns the current accretor mass $M_2$ (g).
    #[must_use]
    pub fn accretor_mass_g(&self) -> f64 {
        match &self.companion {
            BinaryCompanion::PointMass { mass_g, .. } => *mass_g,
            BinaryCompanion::Star(s2) => s2.mstar,
        }
    }

    /// Returns the current accretor photospheric radius $R_2$ (cm).
    #[must_use]
    pub fn accretor_radius_cm(&self) -> f64 {
        match &self.companion {
            BinaryCompanion::PointMass { .. } => 0.0,
            BinaryCompanion::Star(s2) => s2.r[0],
        }
    }

    fn accretor_mass_and_sys_loss(
        &self,
        mdot_tr: f64,
        dt: f64,
        xfer_frac: f64,
    ) -> (f64, f64) {
        let m2_old = self.accretor_mass_g();
        let raw_acc = -xfer_frac * mdot_tr;
        if let BinaryCompanion::PointMass {
            m_bh_zero_eq_g,
            is_black_hole: true,
            ..
        } = &self.companion
        {
            let kappa = 0.2 * (1.0 + self.donor.xh_mass[0]);
            let (mdot_edd, eta) = eval_mdot_edd(m2_old, *m_bh_zero_eq_g, kappa);
            let capped = raw_acc.min(mdot_edd);
            // Include the radiated energy fraction `eta * capped` in the
            // system mass-loss rate (`binary/private/binary_mdot.f90:582-618`).
            let sys_tr_acc = (-raw_acc + (1.0 - eta) * capped)
                + self.mass_transfer_beta * mdot_tr;
            let m2_new = m2_old + (1.0 - eta) * capped * dt;
            (m2_new, sys_tr_acc)
        } else {
            (m2_old + raw_acc * dt, self.mass_transfer_beta * mdot_tr)
        }
    }

    /// Advances the coupled binary system by one timestep `dt` (s).
    pub fn step(&mut self, dt: f64) -> (f64, i32) {
        let m1_old = self.donor.mstar;
        let m2_old = self.accretor_mass_g();
        let j_old = self.angular_momentum_j;
        let a_old = self.separation;
        let e_old = self.eccentricity;
        let gap_old = self.rl_relative_gap;

        let r1 = self.donor.r[0];
        let rho1 = self.donor.rho[0];
        let p1 = self.donor.peos[0].max(1.0);
        let t1 = self.donor.t[0];
        let mu1 = self.donor.mu[0].max(0.5);

        let xfer_frac = (1.0
            - self.mass_transfer_alpha
            - self.mass_transfer_beta
            - self.mass_transfer_delta)
            .clamp(0.0, 1.0);

        let explicit_mdot = eval_ritter_mdot_thin(
            m1_old, m2_old, r1, self.rl[0], rho1, p1, t1, mu1, true,
        ) + eval_kolb_mdot_thick(
            m1_old,
            m2_old,
            self.rl[0],
            &self.donor.r,
            &self.donor.peos,
            &self.donor.t,
            &self.donor.mu,
            &self.donor.gamma1,
        );

        let mdot_tr = if self.use_implicit_rlof
            && self.rl_relative_gap[0] > -1.0e-3
        {
            let solver = ImplicitRlofSolver::default();
            let rl0 = self.rl[0];
            let e = self.eccentricity;
            let (mdot_imp, _, _) =
                solver.solve_roche_lobe_mdot(explicit_mdot, 15, |trial_mdot| {
                    let dm = trial_mdot * dt;
                    let m1_trial = (m1_old + dm).max(0.05 * MSUN);
                    let r_trial = r1 * (m1_trial / m1_old).sqrt();
                    let (m2_trial, _) = self
                        .accretor_mass_and_sys_loss(trial_mdot, dt, xfer_frac);
                    let rl_trial = eval_rlobe(m1_trial, m2_trial, a_old);
                    eval_rl_relative_gap(r_trial, rl_trial.max(0.1 * rl0), e)
                });
            mdot_imp
        } else {
            explicit_mdot
        };
        self.mtransfer_rate = mdot_tr;

        let dm_donor = mdot_tr * dt;
        let m1_new = (m1_old + dm_donor).max(0.05 * MSUN);
        let scale = m1_new / m1_old;
        self.donor.mstar = m1_new;
        for m_k in &mut self.donor.m {
            *m_k *= scale;
        }
        for r_k in &mut self.donor.r {
            *r_k *= scale.sqrt();
        }
        for lnr_k in &mut self.donor.ln_r {
            *lnr_k += 0.5 * scale.ln();
        }

        let (m2_new, mdot_sys_tr_accretor) =
            self.accretor_mass_and_sys_loss(mdot_tr, dt, xfer_frac);
        match &mut self.companion {
            BinaryCompanion::PointMass { mass_g, .. } => {
                *mass_g = m2_new;
            }
            BinaryCompanion::Star(s2) => {
                let scale2 = m2_new / s2.mstar;
                s2.mstar = m2_new;
                for m_k in &mut s2.m {
                    *m_k *= scale2;
                }
            }
        }

        // Estimate outer convective envelope mass fraction from gradr > grada
        let mut dm_conv_env = 0.0_f64;
        for k in 0..(self.donor.nz / 2) {
            if self.donor.gradr[k] > self.donor.grada[k] {
                dm_conv_env += self.donor.dm[k];
            }
        }
        let q_conv_env = (dm_conv_env / m1_new.max(1.0)).clamp(0.0, 1.0);
        let jdot_mb = if q_conv_env > 1.0e-6 && q_conv_env < 0.99 {
            eval_jdot_mb(
                m1_new,
                self.donor.r[0],
                self.rl[0],
                self.period,
                3.0,
                q_conv_env,
                true,
            )
        } else {
            0.0
        };
        let jdot_gr = eval_jdot_gr(m1_new, m2_new, self.separation, j_old);
        let jdot_ml = eval_jdot_ml(
            m1_new,
            m2_new,
            self.separation,
            self.period,
            self.eccentricity,
            self.mass_transfer_alpha * mdot_tr,
            0.0,
            mdot_sys_tr_accretor,
            0.0,
            self.mass_transfer_delta * mdot_tr,
            self.mass_transfer_gamma,
        );
        self.jdot = jdot_gr + jdot_mb + jdot_ml;
        self.angular_momentum_j = (j_old + self.jdot * dt).max(1.0e40);
        if self.eccentricity > 0.0 {
            let edot_gr = eval_edot_gr(
                m1_new,
                m2_new,
                self.separation,
                self.eccentricity,
            );
            self.eccentricity =
                (self.eccentricity + edot_gr * dt).clamp(0.0, 0.9999_9999);
        }
        self.separation = separation_from_j_orb(
            m1_new,
            m2_new,
            self.angular_momentum_j,
            self.eccentricity,
        );
        self.period = period_from_separation(m1_new, m2_new, self.separation);
        let r1_new = self.donor.r[0];
        let r2_new = self.accretor_radius_cm();
        let (rl_new, gap_new) = compute_rl_and_gaps(
            m1_new,
            m2_new,
            r1_new,
            r2_new,
            self.separation,
            self.eccentricity,
        );
        self.rl = rl_new;
        self.rl_relative_gap = gap_new;

        self.binary_age_yr += dt / SECYER;
        self.model_number += 1;

        let ctrl = BinaryTimestepControls {
            donor_time_step_yr: dt / SECYER,
            max_timestep_sec: self.max_timestep_sec,
            rl_relative_gap: self.rl_relative_gap,
            rl_relative_gap_old: gap_old,
            angular_momentum_j: self.angular_momentum_j,
            angular_momentum_j_old: j_old,
            separation: self.separation,
            separation_old: a_old,
            eccentricity: self.eccentricity,
            eccentricity_old: e_old,
            m: [m1_new, m2_new],
            m_old: [m1_old, m2_old],
            is_point_mass_companion: matches!(
                self.companion,
                BinaryCompanion::PointMass { .. }
            ),
            ..BinaryTimestepControls::default()
        };
        let (next_dt, reason) = binary_pick_next_timestep(&ctrl);
        self.max_timestep_sec = next_dt;
        (next_dt, reason)
    }

    /// Checks whether the companion star overfills its Roche lobe more deeply
    /// than the current donor ($g_2 > g_1$) and swaps the donor and accretor
    /// roles when `keep_donor_fixed` is `false` (`binary_evolve.f90:240-290`).
    pub fn check_donor_switch(&mut self, keep_donor_fixed: bool) -> bool {
        if keep_donor_fixed
            || self.rl_relative_gap[1] <= self.rl_relative_gap[0]
        {
            return false;
        }
        if let BinaryCompanion::Star(s2) = &mut self.companion {
            std::mem::swap(&mut self.donor, s2);
            self.rl.swap(0, 1);
            self.rl_relative_gap.swap(0, 1);
            return true;
        }
        false
    }
}
