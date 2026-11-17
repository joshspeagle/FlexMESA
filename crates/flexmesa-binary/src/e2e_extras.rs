// SPDX-License-Identifier: LGPL-3.0-or-later
//! Case-specific [`crate::extras::BinaryExtras`] hooks and oracle verifiers for
//! the 8 upstream MESA `binary/test_suite` campaigns (`run_binary_extras.f90`
//! and `double_bh/src/run_star_extras.f90`).

use flexmesa_const::{CLIGHT, CRAD, LSUN, MSUN, PI, RSUN, SECDAY, SECYER};
use flexmesa_star::{StarError, StarModel};

use crate::e2e_oracles::{
    eval_brott_wind_msun_yr, eval_gr_expected_separation_rsun,
    eval_tauris_vdh_expected_separation_rsun, BinaryInlistControls,
};
use crate::extras::{BinaryExtras, BinaryExtrasAction};
use crate::system::{BinaryCompanion, BinarySystem};
use crate::system_rlof::surface_h1_and_he4;

fn species_frac(
    star: &StarModel,
    zone: usize,
    iso: &str,
    fallback: f64,
) -> f64 {
    if star.nz == 0 || star.species == 0 {
        return fallback;
    }
    let k = zone.min(star.nz - 1);
    star.iso_names
        .iter()
        .position(|n| n.eq_ignore_ascii_case(iso))
        .map_or(fallback, |j| star.cell_xa(k)[j])
}

fn center_h1(star: &StarModel) -> f64 {
    species_frac(star, star.nz.saturating_sub(1), "h1", 0.7)
}
fn center_he4(star: &StarModel) -> f64 {
    species_frac(star, star.nz.saturating_sub(1), "he4", 0.28)
}
fn surface_he4(star: &StarModel) -> f64 {
    species_frac(star, 0, "he4", 0.28)
}
fn total_lnuc_erg_s(star: &StarModel) -> f64 {
    flexmesa_star::compute_power_categories(star).power_nuc_burn * LSUN
}

pub(crate) trait TestSuiteBinaryExtras: BinaryExtras {
    fn max_relative_oracle_error(&self) -> f64 {
        0.0
    }
    fn evaluate_pass(
        &self,
        bin: &BinarySystem,
        uncapped: bool,
        term_code: &str,
    ) -> bool;
}

/// Peters (1964) gravitational-wave inspiral verifier (`jdot_gr_check`).
#[derive(Debug, Clone)]
pub struct JdotGrCheckExtras {
    /// Initial orbital separation $a_0$ ($R_\odot$).
    pub a0_rsun: f64,
    /// Initial primary mass $M_1$ ($M_\odot$).
    pub m1_msun: f64,
    /// Initial secondary mass $M_2$ ($M_\odot$).
    pub m2_msun: f64,
    /// Maximum relative separation error observed across steps.
    pub max_rel_err: f64,
    /// Whether a per-step oracle bound failed mid-run.
    pub failed_mid_run: bool,
}

impl BinaryExtras for JdotGrCheckExtras {
    fn finish_step(&mut self, bin: &mut BinarySystem) -> BinaryExtrasAction {
        if bin.donor.mstar_dot != 0.0 {
            return BinaryExtrasAction::Terminate(String::from("GrCheckPass"));
        }
        let a_exp = eval_gr_expected_separation_rsun(
            self.a0_rsun,
            self.m1_msun,
            self.m2_msun,
            bin.donor.star_age * SECYER,
        );
        let rel = ((bin.separation / RSUN) - a_exp).abs() / a_exp.max(1.0e-30);
        self.max_rel_err = self.max_rel_err.max(rel);
        if rel > 1.0e-2 {
            self.failed_mid_run = true;
            return BinaryExtrasAction::Terminate(String::from(
                "GrErrorExceeded",
            ));
        }
        BinaryExtrasAction::KeepGoing
    }
}

impl TestSuiteBinaryExtras for JdotGrCheckExtras {
    fn max_relative_oracle_error(&self) -> f64 {
        self.max_rel_err
    }
    fn evaluate_pass(
        &self,
        bin: &BinarySystem,
        uncapped: bool,
        term_code: &str,
    ) -> bool {
        !self.failed_mid_run
            && self.max_rel_err < 1.0e-2
            && (bin.separation / RSUN) < self.a0_rsun
            && (!uncapped
                || (term_code == "GrCheckPass" && bin.donor.mstar_dot < 0.0))
    }
}

/// Tauris & van den Heuvel (2006) non-conservative mass-transfer verifier (`jdot_ml_check`).
#[derive(Debug, Clone)]
pub struct JdotMlCheckExtras {
    /// Initial orbital separation $a_0$ ($R_\odot$).
    pub a0_rsun: f64,
    /// Initial mass ratio $q_0 = M_1 / M_2$.
    pub q0: f64,
    /// Initial donor mass $M_1$ ($M_\odot$).
    pub m1_init_msun: f64,
    /// Maximum relative separation error observed across steps.
    pub max_rel_err: f64,
    /// Whether a per-step oracle bound failed mid-run.
    pub failed_mid_run: bool,
}

impl BinaryExtras for JdotMlCheckExtras {
    fn finish_step(&mut self, bin: &mut BinarySystem) -> BinaryExtrasAction {
        let [m1_g, m2_g] = bin.star_masses_g();
        let q = m1_g / m2_g.max(1.0);
        let a_exp = eval_tauris_vdh_expected_separation_rsun(
            self.a0_rsun,
            self.q0,
            q,
            bin.controls.mass_transfer_alpha,
            bin.controls.mass_transfer_beta,
            bin.controls.mass_transfer_delta,
            bin.controls.mass_transfer_gamma,
        );
        let rel = ((bin.separation / RSUN) - a_exp).abs() / a_exp.max(1.0e-30);
        self.max_rel_err = self.max_rel_err.max(rel);
        if rel > 1.0e-4 {
            self.failed_mid_run = true;
            return BinaryExtrasAction::Terminate(String::from(
                "MlErrorExceeded",
            ));
        }
        BinaryExtrasAction::KeepGoing
    }
}

impl TestSuiteBinaryExtras for JdotMlCheckExtras {
    fn max_relative_oracle_error(&self) -> f64 {
        self.max_rel_err
    }
    fn evaluate_pass(
        &self,
        bin: &BinarySystem,
        uncapped: bool,
        term_code: &str,
    ) -> bool {
        let target_age = bin.donor.controls.max_age.unwrap_or(1.4e10);
        let m1_msun = bin.star_masses_g()[0] / MSUN;
        !self.failed_mid_run
            && self.max_rel_err < 1.0e-4
            && (!uncapped
                || (term_code == "max_age"
                    && bin.donor.star_age >= target_age * (1.0 - 1.0e-6)
                    && m1_msun < self.m1_init_msun - 1.0e-3))
    }
}

/// Spin-orbit angular momentum conservation verifier (`jdot_ls_check`).
#[derive(Debug, Clone)]
pub struct JdotLsCheckExtras {
    /// Initial conserved total angular momentum ($\mathrm{g\,cm^2\,s^{-1}}$).
    pub j_conserved_0: f64,
    /// Maximum fractional conservation error observed across steps.
    pub max_rel_err: f64,
    /// Whether a per-step conservation bound failed mid-run.
    pub failed_mid_run: bool,
}

impl BinaryExtras for JdotLsCheckExtras {
    fn finish_step(&mut self, bin: &mut BinarySystem) -> BinaryExtrasAction {
        let j_cur = bin.total_angular_momentum_conserved();
        let rel = (j_cur - self.j_conserved_0).abs()
            / self.j_conserved_0.abs().max(1.0e-30);
        self.max_rel_err = self.max_rel_err.max(rel);
        if rel >= 1.0e-12 {
            self.failed_mid_run = true;
            return BinaryExtrasAction::Terminate(String::from(
                "LsConservationExceeded",
            ));
        }
        BinaryExtrasAction::KeepGoing
    }
}

impl TestSuiteBinaryExtras for JdotLsCheckExtras {
    fn max_relative_oracle_error(&self) -> f64 {
        self.max_rel_err
    }
    fn evaluate_pass(
        &self,
        bin: &BinarySystem,
        uncapped: bool,
        term_code: &str,
    ) -> bool {
        let target_age = bin.donor.controls.max_age.unwrap_or(1.0e10);
        let omega_orb = 2.0 * PI / bin.period;
        let omega_surf = bin.donor.omega.first().copied().unwrap_or(0.0);
        let sync_err = (omega_surf - omega_orb).abs() / omega_orb;
        !self.failed_mid_run
            && self.max_rel_err < 1.0e-12
            && (!uncapped
                || (term_code == "max_age"
                    && bin.donor.star_age >= target_age * (1.0 - 1.0e-6)
                    && sync_err < 1.0e-2))
    }
}

/// Bondi-Hoyle wind transfer and Kerr BH Eddington verifier (`wind_fed_bhhmxb`).
#[derive(Debug, Default, Clone)]
pub struct WindFedBhHmxbExtras {
    /// Whether a sub-Eddington wind-accretion step was observed.
    pub saw_sub_edd: bool,
    /// Whether a super-Eddington wind-accretion step was observed.
    pub saw_super_edd: bool,
    /// Maximum fractional BH ejecta accounting error observed across steps.
    pub max_rel_err: f64,
    /// Whether an accounting bound failed mid-run.
    pub failed_mid_run: bool,
}

impl BinaryExtras for WindFedBhHmxbExtras {
    fn finish_step(&mut self, bin: &mut BinarySystem) -> BinaryExtrasAction {
        let acct = &bin.last_mdot;
        let max_allowed =
            acct.mdot_edd * (1.0 - acct.mdot_edd_eta) * (1.0 + 1.0e-10);
        if acct.component_mdot[1] > max_allowed {
            self.failed_mid_run = true;
            return BinaryExtrasAction::Terminate(String::from(
                "ExceededEddington",
            ));
        }
        let expected_sys = if -acct.actual_mtransfer_rate > acct.mdot_edd {
            self.saw_super_edd = true;
            acct.actual_mtransfer_rate + acct.mdot_edd
                - acct.accretion_luminosity / (CLIGHT * CLIGHT)
        } else {
            self.saw_sub_edd = true;
            -acct.accretion_luminosity / (CLIGHT * CLIGHT)
        };
        let frac_bh = (acct.mdot_system_transfer[1] - expected_sys).abs()
            / expected_sys.abs();
        let donor_wind_mdot = bin.donor.mstar_dot - bin.mtransfer_rate;
        let exp_wtr = donor_wind_mdot * acct.wind_xfer_fraction[0];
        let err_wtr = (acct.mdot_wind_transfer[0] - exp_wtr).abs()
            / acct.mdot_wind_transfer[0].abs();
        let exp_wsys = donor_wind_mdot * (1.0 - acct.wind_xfer_fraction[0]);
        let err_wsys =
            (acct.mdot_system_wind[0] - exp_wsys).abs() / exp_wsys.abs();
        if bin.model_number > 1 {
            self.max_rel_err = self.max_rel_err.max(frac_bh);
        }
        if (bin.model_number > 1 && (!frac_bh.is_finite() || frac_bh > 1.0e-15))
            || !err_wtr.is_finite()
            || err_wtr > 1.0e-10
            || !err_wsys.is_finite()
            || err_wsys > 1.0e-10
        {
            self.failed_mid_run = true;
            return BinaryExtrasAction::Terminate(String::from(
                "WindFedAccountingExceeded",
            ));
        }
        BinaryExtrasAction::KeepGoing
    }
}

impl TestSuiteBinaryExtras for WindFedBhHmxbExtras {
    fn max_relative_oracle_error(&self) -> f64 {
        self.max_rel_err
    }
    fn evaluate_pass(
        &self,
        bin: &BinarySystem,
        uncapped: bool,
        term_code: &str,
    ) -> bool {
        !self.failed_mid_run
            && self.max_rel_err <= 1.0e-15
            && (!uncapped
                || (term_code == "xa_central_lower_limit(h1)"
                    && self.saw_sub_edd
                    && self.saw_super_edd
                    && center_h1(&bin.donor) < 1.0e-2))
    }
}

/// Point-mass Eddington cap verifier (`star_plus_point_mass`, `star_plus_point_mass_explicit_mdot`).
#[derive(Debug, Clone)]
pub struct PointMassEddExtras {
    /// Initial donor mass $M_1$ ($M_\odot$).
    pub m1_init_msun: f64,
    /// Whether the point-mass Eddington limit was exceeded mid-run.
    pub failed_mid_run: bool,
}

impl BinaryExtras for PointMassEddExtras {
    fn finish_step(&mut self, bin: &mut BinarySystem) -> BinaryExtrasAction {
        let acct = &bin.last_mdot;
        if acct.mdot_edd > 0.0
            && acct.component_mdot[1] > acct.mdot_edd * (1.0 + 1.0e-12)
        {
            self.failed_mid_run = true;
            return BinaryExtrasAction::Terminate(String::from(
                "PointMassExceededEddington",
            ));
        }
        BinaryExtrasAction::KeepGoing
    }
}

impl TestSuiteBinaryExtras for PointMassEddExtras {
    fn evaluate_pass(
        &self,
        bin: &BinarySystem,
        uncapped: bool,
        term_code: &str,
    ) -> bool {
        let m1_msun = bin.star_masses_g()[0] / MSUN;
        !self.failed_mid_run
            && (!uncapped
                || (term_code == "xa_central_lower_limit(h1)"
                    && center_h1(&bin.donor) < 1.0e-2
                    && m1_msun < self.m1_init_msun))
    }
}

/// Two-star contact/overflow verifier (`evolve_both_stars`).
#[derive(Debug, Default, Clone, Copy)]
pub struct EvolveBothStarsExtras;

impl BinaryExtras for EvolveBothStarsExtras {}

impl TestSuiteBinaryExtras for EvolveBothStarsExtras {
    fn evaluate_pass(
        &self,
        bin: &BinarySystem,
        uncapped: bool,
        term_code: &str,
    ) -> bool {
        !uncapped
            || (term_code == "accretor_overflow_terminate"
                && bin.rl_relative_gap[1] >= 0.0)
    }
}

/// Brott et al. (2011) `other_wind`, ZAMS RLOF settling switch, chemically
/// homogeneous `He4` check, and core-He BH collapse hooks (`double_bh`).
#[derive(Debug, Clone)]
pub struct DoubleBhExtras {
    /// Target initial orbital period $P_0$ (days) held while `ignore_rlof_flag` settles ZAMS.
    pub initial_period_in_days: f64,
    /// Maximum $Y_{\mathrm{center}} - Y_{\mathrm{surface}}$ observed during core H burning.
    pub max_qhe_diff: f64,
    /// Whether ZAMS L2 overflow or inhomogeneous evolution failed mid-run.
    pub failed_mid_run: bool,
}

impl BinaryExtras for DoubleBhExtras {
    fn other_wind_cgs(
        &mut self,
        star: &StarModel,
        _star_idx: usize,
        ignore_rlof_flag: bool,
    ) -> Result<Option<f64>, StarError> {
        if ignore_rlof_flag {
            return Ok(Some(0.0));
        }
        let (x_surf, y_surf) = surface_h1_and_he4(star);
        let z_base = if star.controls.zbase > 0.0 {
            star.controls.zbase
        } else {
            (1.0 - x_surf - y_surf).clamp(1.0e-6, 0.1)
        };
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
        let w_msun_yr = eval_brott_wind_msun_yr(
            (l1 / LSUN).max(1.0e-10),
            (star.mstar / MSUN).max(1.0e-5),
            (r1 / RSUN).max(1.0e-5),
            t1,
            x_surf,
            y_surf,
            z_base / 0.017,
        );
        Ok(Some(-(w_msun_yr.max(0.0) * (MSUN / SECYER))))
    }

    fn finish_step(&mut self, bin: &mut BinarySystem) -> BinaryExtrasAction {
        if bin.ignore_rlof_flag {
            let zams_settled = bin.star_ref(1).map(|s1| {
                let lnuc = total_lnuc_erg_s(s1).abs();
                let lsurf = s1.lum[0].abs().max(1.0);
                s1.star_age > 100.0
                    && flexmesa_math::log10(lnuc / lsurf).abs() < 0.005
            });
            if let Some(settled) = zams_settled {
                if settled {
                    let gap1 = bin.rl_relative_gap_by_star()[0];
                    if gap1 > 0.320_819_224 {
                        self.failed_mid_run = true;
                        return BinaryExtrasAction::Terminate(String::from(
                            "L2OverflowAtZams",
                        ));
                    }
                    if gap1 > 0.0 {
                        bin.controls.max_implicit_abs_mdot = 1.0e-3;
                        bin.donor.controls.mix_wind.delta_mdot_hard_limit =
                            -1.0;
                        if let BinaryCompanion::Star(ref mut s2) = bin.companion
                        {
                            s2.controls.mix_wind.delta_mdot_hard_limit = -1.0;
                        }
                    }
                    bin.set_ignore_rlof_flag(false);
                    bin.controls.terminate_if_l2_overflow = true;
                } else if let Err(e) = bin.set_period_eccentricity(
                    self.initial_period_in_days * SECDAY,
                    0.0,
                ) {
                    self.failed_mid_run = true;
                    return BinaryExtrasAction::Terminate(format!(
                        "set_period_eccentricity failed: {e}"
                    ));
                }
            }
        }
        for s_idx in [1_usize, 2_usize] {
            if let Some(st) = bin.star_ref(s_idx) {
                if center_h1(st) > 1.0e-3 {
                    let diff = center_he4(st) - surface_he4(st);
                    self.max_qhe_diff = self.max_qhe_diff.max(diff);
                    if diff > 0.2 {
                        self.failed_mid_run = true;
                        return BinaryExtrasAction::Terminate(String::from(
                            "InhomogeneousEvolution",
                        ));
                    }
                }
            }
        }
        if bin.point_mass_i == 0 {
            let s1_he = bin.star_ref(1).map_or(1.0, center_he4);
            let s2_he = bin.star_ref(2).map_or(1.0, center_he4);
            if s1_he < 1.0e-5 {
                if let Err(e) = bin.set_point_mass_i(1) {
                    self.failed_mid_run = true;
                    return BinaryExtrasAction::Terminate(format!(
                        "set_point_mass_i(1) failed: {e}"
                    ));
                }
            } else if s2_he < 1.0e-5 {
                if let Err(e) = bin.set_point_mass_i(2) {
                    self.failed_mid_run = true;
                    return BinaryExtrasAction::Terminate(format!(
                        "set_point_mass_i(2) failed: {e}"
                    ));
                }
            }
        }
        if bin.point_mass_i != 0 {
            let rem_idx = 3 - bin.point_mass_i;
            if bin
                .star_ref(rem_idx)
                .is_some_and(|s| center_he4(s) < 1.0e-5)
            {
                return BinaryExtrasAction::Terminate(String::from(
                    "DoubleBhPass",
                ));
            }
        }
        BinaryExtrasAction::KeepGoing
    }
}

impl TestSuiteBinaryExtras for DoubleBhExtras {
    fn evaluate_pass(
        &self,
        bin: &BinarySystem,
        uncapped: bool,
        term_code: &str,
    ) -> bool {
        let [m1_g, m2_g] = bin.star_masses_g();
        let (m1_msun, m2_msun) = (m1_g / MSUN, m2_g / MSUN);
        !self.failed_mid_run
            && self.max_qhe_diff <= 0.2
            && m1_msun > 20.0
            && m2_msun > 20.0
            && (!uncapped
                || (bin.point_mass_i != 0 && term_code == "DoubleBhPass"))
    }
}

pub(crate) fn build_test_suite_extras(
    case_name: &str,
    bin: &BinarySystem,
    bctrl: &BinaryInlistControls,
) -> Box<dyn TestSuiteBinaryExtras> {
    let a0_rsun = bin.separation / RSUN;
    let q0 = bctrl.m1_msun / bctrl.m2_msun.max(1.0e-12);
    match case_name {
        "jdot_gr_check" => Box::new(JdotGrCheckExtras {
            a0_rsun,
            m1_msun: bctrl.m1_msun,
            m2_msun: bctrl.m2_msun,
            max_rel_err: 0.0,
            failed_mid_run: false,
        }),
        "jdot_ml_check" => Box::new(JdotMlCheckExtras {
            a0_rsun,
            q0,
            m1_init_msun: bctrl.m1_msun,
            max_rel_err: 0.0,
            failed_mid_run: false,
        }),
        "jdot_ls_check" => Box::new(JdotLsCheckExtras {
            j_conserved_0: bin.total_angular_momentum_conserved(),
            max_rel_err: 0.0,
            failed_mid_run: false,
        }),
        "wind_fed_bhhmxb" => Box::new(WindFedBhHmxbExtras::default()),
        "star_plus_point_mass" | "star_plus_point_mass_explicit_mdot" => {
            Box::new(PointMassEddExtras {
                m1_init_msun: bctrl.m1_msun,
                failed_mid_run: false,
            })
        }
        "double_bh" => Box::new(DoubleBhExtras {
            initial_period_in_days: bctrl.initial_period_in_days,
            max_qhe_diff: 0.0,
            failed_mid_run: false,
        }),
        _ => Box::new(EvolveBothStarsExtras),
    }
}
