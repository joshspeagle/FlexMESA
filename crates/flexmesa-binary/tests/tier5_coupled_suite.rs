// SPDX-License-Identifier: GPL-3.0-or-later
//! Stage `E2E-5` (`#114`) coupled `binary/test_suite` integration tests and
//! unit checks for decoupled Eddington retention vs. radiation-corrected
//! accretion, recursive 3-level binary inlist chaining,
//! `initial_separation_in_rsuns` orbital initialization, Brott et al. (2011)
//! winds, and dynamic `set_point_mass_i` conversion.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use flexmesa_binary::{
    BinaryCompanion, BinaryExtras, BinaryExtrasAction, BinaryInlistControls,
    BinaryLogsOptions, BinarySystem, ImplicitMdotSolver, ImplicitStepAction,
    RocheDeformationTables, bh_initial_equatorial_mass,
    copy_cleaned_binary_case, eval_edot_enhancement_isotropic,
    eval_point_mass_edd_and_retention, eval_synchronicity_switch,
    run_binary_test_suite_case, run_binary_test_suite_case_with_extras,
    separation_from_period,
};
use flexmesa_const::{
    CLIGHT, CRAD, LSUN, MSUN, PI, RSUN, SECDAY, SECYER, STANDARD_CGRAV,
};
use flexmesa_math::{exp10, log10, pow};
use flexmesa_star::{
    DefaultStarExtras, InlistBundle, InlistLoadOptions, MdotRedoState,
    SelectMdotAction, StarControls, StarError, StarJobControls, StarModel,
    check_termination_limits, init_or_update_model_from_bundle,
    load_zams_model,
};

const BINARY_CASES: [&str; 8] = [
    "double_bh",
    "evolve_both_stars",
    "jdot_gr_check",
    "jdot_ls_check",
    "jdot_ml_check",
    "star_plus_point_mass",
    "star_plus_point_mass_explicit_mdot",
    "wind_fed_bhhmxb",
];

/// Exact Peters (1964) GW separation oracle
/// (`binary/test_suite/jdot_gr_check/src/run_binary_extras.f90:99-120`).
fn eval_gr_expected_separation_rsun(
    a0_rsun: f64,
    m1_msun: f64,
    m2_msun: f64,
    age_sec: f64,
) -> f64 {
    let a0_cm = a0_rsun * RSUN;
    let m1 = m1_msun * MSUN;
    let m2 = m2_msun * MSUN;
    let g3 = STANDARD_CGRAV * STANDARD_CGRAV * STANDARD_CGRAV;
    let c5 = CLIGHT * CLIGHT * CLIGHT * CLIGHT * CLIGHT;
    let beta = (64.0 * g3 / (5.0 * c5)) * m1 * m2 * (m1 + m2);
    let a4 = (flexmesa_math::pow4(a0_cm) - 4.0 * beta * age_sec).max(0.0);
    pow(a4, 0.25) / RSUN
}

/// Exact Tauris & van den Heuvel (2006) non-conservative separation oracle
/// (`binary/test_suite/jdot_ml_check/src/run_binary_extras.f90:99-132`).
fn eval_tauris_vdh_expected_separation_rsun(
    a0_rsun: f64,
    q0: f64,
    q: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
) -> f64 {
    let eps = 1.0 - alpha - beta - delta;
    let one_m_eps = (1.0 - eps).max(1.0e-15);
    let term1 = pow(q / q0, 2.0 * (alpha + gamma * delta - 1.0));
    let term2 =
        pow((q + 1.0) / (q0 + 1.0), (-alpha - beta + delta) / one_m_eps);
    let term3 = if eps.abs() > 1.0e-12 {
        let exp3 = 3.0
            + 2.0
                * (alpha * eps * eps
                    + beta
                    + gamma * delta * flexmesa_math::pow2(one_m_eps))
                / (eps * one_m_eps);
        pow((eps * q + 1.0) / (eps * q0 + 1.0), exp3)
    } else {
        let coeff = 2.0 * (beta + gamma * delta);
        flexmesa_math::exp(coeff * (q - q0))
    };
    a0_rsun * term1 * term2 * term3
}

/// Brott et al. (2011) massive-star wind law
/// (`binary/test_suite/double_bh/src/run_star_extras.f90:55-206`).
fn eval_brott_wind_msun_yr(
    l_surf_lsun: f64,
    m_surf_msun: f64,
    r_surf_rsun: f64,
    t_surf_k: f64,
    x_surf: f64,
    z_div_z_solar: f64,
) -> f64 {
    if l_surf_lsun <= 0.0
        || m_surf_msun <= 0.0
        || r_surf_rsun <= 0.0
        || t_surf_k <= 0.0
        || z_div_z_solar <= 0.0
    {
        return 0.0;
    }
    let log_z_ratio = log10(z_div_z_solar);
    let teff_jump = 1.0e3 * (61.2 + 2.59 * (-13.636 + 0.889 * log_z_ratio));
    let vink_wind = eval_brott_vink_wind(
        l_surf_lsun,
        m_surf_msun,
        t_surf_k,
        z_div_z_solar,
        log_z_ratio,
        teff_jump,
    );
    let nieu_wind = exp10(
        -14.02
            + 1.24 * log10(l_surf_lsun)
            + 0.16 * log10(m_surf_msun)
            + 0.81 * log10(r_surf_rsun)
            + 0.85 * log_z_ratio,
    );
    let hamann_wind = exp10(
        -11.95 + 1.5 * log10(l_surf_lsun) - 2.85 * x_surf + 0.85 * log_z_ratio,
    ) / 10.0;
    if t_surf_k < teff_jump {
        vink_wind.max(nieu_wind)
    } else {
        let alfa = if x_surf > 0.7 {
            1.0
        } else if x_surf > 0.4 {
            (x_surf - 0.4) / 0.3
        } else {
            0.0
        };
        alfa * vink_wind + (1.0 - alfa) * hamann_wind
    }
}

fn eval_brott_vink_wind(
    l_lsun: f64,
    m_msun: f64,
    t_k: f64,
    z_div_z_solar: f64,
    log_z_ratio: f64,
    teff_jump: f64,
) -> f64 {
    let alfa = if t_k > 27_500.0 {
        1.0
    } else if t_k < 22_500.0 {
        0.0
    } else {
        let dt = 100.0;
        if t_k > teff_jump + dt {
            1.0
        } else if t_k < teff_jump - dt {
            0.0
        } else {
            (t_k - (teff_jump - dt)) / (2.0 * dt)
        }
    };
    let w1 = if alfa > 0.0 {
        let vinf = 2.6 * pow(z_div_z_solar, 0.13);
        let lt = log10(t_k / 4.0e4);
        exp10(
            -6.697 + 2.194 * log10(l_lsun / 1.0e5)
                - 1.313 * log10(m_msun / 30.0)
                - 1.226 * log10(vinf / 2.0)
                + 0.933 * lt
                - 10.92 * lt * lt
                + 0.85 * log_z_ratio,
        )
    } else {
        0.0
    };
    let w2 = if alfa < 1.0 {
        let vinf = 1.3 * pow(z_div_z_solar, 0.13);
        exp10(
            -6.688 + 2.210 * log10(l_lsun / 1.0e5)
                - 1.339 * log10(m_msun / 30.0)
                - 1.601 * log10(vinf / 2.0)
                + 1.07 * log10(t_k / 2.0e4)
                + 0.85 * log_z_ratio,
        )
    } else {
        0.0
    };
    alfa * w1 + (1.0 - alfa) * w2
}

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

trait TestOracleExtras: BinaryExtras {
    fn max_rel_err(&self) -> f64 {
        0.0
    }
    fn failed(&self) -> bool {
        false
    }
}

struct JdotGrCheckExtras {
    a0_rsun: f64,
    m1_msun: f64,
    m2_msun: f64,
    max_rel_err: f64,
    failed_mid_run: bool,
}

impl BinaryExtras for JdotGrCheckExtras {
    // Upstream makes this test in `extras_binary_check_model`
    // (`binary/test_suite/jdot_gr_check/src/run_binary_extras.f90:151-169`); the code is the
    // hook's upstream name, as the Python adapter reports it.
    fn check_model(&mut self, bin: &mut BinarySystem) -> BinaryExtrasAction {
        if bin.donor.mstar_dot != 0.0 {
            return BinaryExtrasAction::Terminate(String::from(
                "extras_binary_check_model",
            ));
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

impl TestOracleExtras for JdotGrCheckExtras {
    fn max_rel_err(&self) -> f64 {
        self.max_rel_err
    }
    fn failed(&self) -> bool {
        self.failed_mid_run
    }
}

struct JdotMlCheckExtras {
    a0_rsun: f64,
    q0: f64,
    max_rel_err: f64,
    failed_mid_run: bool,
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

impl TestOracleExtras for JdotMlCheckExtras {
    fn max_rel_err(&self) -> f64 {
        self.max_rel_err
    }
    fn failed(&self) -> bool {
        self.failed_mid_run
    }
}

#[derive(Default)]
struct JdotLsCheckExtras {
    j_conserved_0: Option<f64>,
    max_rel_err: f64,
    failed_mid_run: bool,
}

impl BinaryExtras for JdotLsCheckExtras {
    fn finish_step(&mut self, bin: &mut BinarySystem) -> BinaryExtrasAction {
        let j_cur = bin.total_angular_momentum_conserved();
        let j0 = *self.j_conserved_0.get_or_insert(j_cur);
        let rel = (j_cur - j0).abs() / j0.abs().max(1.0e-30);
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

impl TestOracleExtras for JdotLsCheckExtras {
    fn max_rel_err(&self) -> f64 {
        self.max_rel_err
    }
    fn failed(&self) -> bool {
        self.failed_mid_run
    }
}

#[derive(Default)]
struct WindFedBhHmxbExtras {
    max_rel_err: f64,
    failed_mid_run: bool,
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
            acct.actual_mtransfer_rate + acct.mdot_edd
                - acct.accretion_luminosity / (CLIGHT * CLIGHT)
        } else {
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

impl TestOracleExtras for WindFedBhHmxbExtras {
    fn max_rel_err(&self) -> f64 {
        self.max_rel_err
    }
    fn failed(&self) -> bool {
        self.failed_mid_run
    }
}

#[derive(Default)]
struct PointMassEddExtras {
    failed_mid_run: bool,
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

impl TestOracleExtras for PointMassEddExtras {
    fn failed(&self) -> bool {
        self.failed_mid_run
    }
}

/// Brott et al. (2011) wind and QHE/collapse hooks (`binary/test_suite/double_bh`).
struct DoubleBhExtras {
    initial_period_in_days: f64,
    max_qhe_diff: f64,
    failed_mid_run: bool,
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
        let x_surf = species_frac(star, 0, "h1", 0.7);
        let y_surf = species_frac(star, 0, "he4", 0.28);
        let z_base = if star.kap_config.zbase >= 0.0 {
            star.kap_config.zbase
        } else {
            (1.0 - x_surf - y_surf).clamp(1.0e-6, 0.1)
        };
        let l1 = star.lum.first().copied().unwrap_or(1.0).max(1.0);
        let t1 = if star.teff > 0.0 {
            star.teff
        } else {
            let r0 = star.r.first().copied().unwrap_or(1.0).max(1.0);
            pow(l1 / (PI * r0 * r0 * CRAD * CLIGHT), 0.25)
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
            z_base / 0.017,
        );
        Ok(Some(-(w_msun_yr.max(0.0) * (MSUN / SECYER))))
    }

    fn finish_step(&mut self, bin: &mut BinarySystem) -> BinaryExtrasAction {
        if bin.ignore_rlof_flag {
            let zams_settled = bin.star_ref(1).map(|s1| {
                let lnuc = flexmesa_star::compute_power_categories(s1)
                    .power_nuc_burn
                    .abs()
                    * LSUN;
                let lsurf = s1.lum[0].abs().max(1.0);
                s1.star_age > 100.0 && log10(lnuc / lsurf).abs() < 0.005
            });
            if let Some(true) = zams_settled {
                let gap1 = bin.rl_relative_gap_by_star()[0];
                if gap1 > 0.320_819_224 {
                    self.failed_mid_run = true;
                    return BinaryExtrasAction::Terminate(String::from(
                        "L2OverflowAtZams",
                    ));
                }
                if gap1 > 0.0 {
                    bin.controls.max_implicit_abs_mdot = 1.0e-3;
                    bin.donor.controls.mix_wind.delta_mdot_hard_limit = -1.0;
                    if let BinaryCompanion::Star(ref mut s2) = bin.companion {
                        s2.controls.mix_wind.delta_mdot_hard_limit = -1.0;
                    }
                }
                bin.set_ignore_rlof_flag(false);
                bin.controls.terminate_if_l2_overflow = true;
            } else if zams_settled == Some(false) {
                if let Err(e) = bin.set_period_eccentricity(
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
                if species_frac(st, st.nz.saturating_sub(1), "h1", 0.7) > 1.0e-3
                {
                    let he_c =
                        species_frac(st, st.nz.saturating_sub(1), "he4", 0.28);
                    let he_s = species_frac(st, 0, "he4", 0.28);
                    let diff = he_c - he_s;
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
            let s1_he = bin.star_ref(1).map_or(1.0, |s| {
                species_frac(s, s.nz.saturating_sub(1), "he4", 0.28)
            });
            let s2_he = bin.star_ref(2).map_or(1.0, |s| {
                species_frac(s, s.nz.saturating_sub(1), "he4", 0.28)
            });
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
            if bin.star_ref(rem_idx).is_some_and(|s| {
                species_frac(s, s.nz.saturating_sub(1), "he4", 0.28) < 1.0e-5
            }) {
                return BinaryExtrasAction::Terminate(String::from(
                    "DoubleBhPass",
                ));
            }
        }
        BinaryExtrasAction::KeepGoing
    }
}

impl TestOracleExtras for DoubleBhExtras {
    fn failed(&self) -> bool {
        self.failed_mid_run || self.max_qhe_diff > 0.2
    }
}

struct DefaultOracleExtras;
impl BinaryExtras for DefaultOracleExtras {}
impl TestOracleExtras for DefaultOracleExtras {}

fn a0_rsun_from_ctrl(bctrl: &BinaryInlistControls) -> f64 {
    if bctrl.initial_separation_in_rsuns > 0.0 {
        bctrl.initial_separation_in_rsuns
    } else {
        separation_from_period(
            bctrl.m1_msun * MSUN,
            bctrl.m2_msun * MSUN,
            bctrl.initial_period_in_days * SECDAY,
        ) / RSUN
    }
}

fn build_test_oracle_extras(
    case_name: &str,
    bctrl: &BinaryInlistControls,
) -> Box<dyn TestOracleExtras> {
    let a0_rsun = a0_rsun_from_ctrl(bctrl);
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
            max_rel_err: 0.0,
            failed_mid_run: false,
        }),
        "jdot_ls_check" => Box::new(JdotLsCheckExtras::default()),
        "wind_fed_bhhmxb" => Box::new(WindFedBhHmxbExtras::default()),
        "star_plus_point_mass" | "star_plus_point_mass_explicit_mdot" => {
            Box::new(PointMassEddExtras::default())
        }
        "double_bh" => Box::new(DoubleBhExtras {
            initial_period_in_days: bctrl.initial_period_in_days,
            max_qhe_diff: 0.0,
            failed_mid_run: false,
        }),
        _ => Box::new(DefaultOracleExtras),
    }
}

struct TempDirGuard {
    root: PathBuf,
}

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir()
            .join(format!("flexmesa_bin_t5_{tag}_{}_{id}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create cleaned binary temp root");
        Self { root }
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn resolve_suite_dir() -> Option<PathBuf> {
    flexmesa_utils::require_mesa_test_input_path("binary/test_suite")
}

#[test]
fn test_decoupled_edd_retention_limit_vs_radiation_correction() {
    let m_ns = 1.4 * MSUN;
    let m_eq = bh_initial_equatorial_mass(m_ns, 0.0);
    let mdot_tr = -1.0e-3 * MSUN / SECYER;

    let capped_no_rad = eval_point_mass_edd_and_retention(
        m_ns, m_eq, 0.70, 0.34, true, -1.0, -1.0, true, false, mdot_tr, 0.0,
        0.0, 0.0, 0.0,
    );
    assert!(capped_no_rad.mdot_edd > 0.0);
    assert!(
        (capped_no_rad.component_mdot_accretor - capped_no_rad.mdot_edd).abs()
            / capped_no_rad.mdot_edd
            < 1.0e-14
    );
    let expected_eject = mdot_tr + capped_no_rad.mdot_edd;
    assert!(
        (capped_no_rad.mdot_system_transfer_accretor - expected_eject).abs()
            / expected_eject.abs()
            < 1.0e-14
    );

    let capped_with_rad = eval_point_mass_edd_and_retention(
        m_ns, m_eq, 0.70, 0.34, true, -1.0, -1.0, true, true, mdot_tr, 0.0,
        0.0, 0.0, 0.0,
    );
    let expected_rad_acc =
        capped_with_rad.mdot_edd * (1.0 - capped_with_rad.mdot_edd_eta);
    assert!(
        (capped_with_rad.component_mdot_accretor - expected_rad_acc).abs()
            / expected_rad_acc
            < 1.0e-14
    );
    let rad_mass_loss =
        -capped_with_rad.accretion_luminosity / (CLIGHT * CLIGHT);
    assert!(
        (capped_with_rad.mdot_system_transfer_accretor
            - (expected_eject + rad_mass_loss))
            .abs()
            / expected_eject.abs()
            < 1.0e-14
    );

    let uncapped_no_rad = eval_point_mass_edd_and_retention(
        m_ns, m_eq, 0.70, 0.34, true, -1.0, -1.0, false, false, mdot_tr, 0.0,
        0.0, 0.0, 0.0,
    );
    assert!(
        (uncapped_no_rad.component_mdot_accretor - (-mdot_tr)).abs()
            / (-mdot_tr)
            < 1.0e-14
    );
}

#[test]
fn test_initial_separation_and_dynamic_point_mass_conversion() {
    let c1 = StarControls {
        initial_mass: 1.0,
        zbase: 0.02,
        ..StarControls::default()
    };
    let s1 =
        load_zams_model(c1, StarJobControls::default()).expect("1 Msun ZAMS");
    let c2 = StarControls {
        initial_mass: 0.8,
        zbase: 0.02,
        ..StarControls::default()
    };
    let s2 =
        load_zams_model(c2, StarJobControls::default()).expect("0.8 Msun ZAMS");

    let ctrl = BinaryInlistControls {
        evolve_both_stars: true,
        m1_msun: 1.0,
        m2_msun: 0.8,
        initial_period_in_days: -1.0,
        initial_separation_in_rsuns: 5.5,
        terminate_if_initial_overflow: false,
        ..BinaryInlistControls::default()
    };
    let mut bin = BinarySystem::from_controls_two_stars(s1, s2, ctrl)
        .expect("two-star system");
    assert!(((bin.separation / RSUN) - 5.5).abs() < 1.0e-12);
    assert!(bin.period > 0.0);
    assert_eq!(bin.point_mass_i, 0);

    bin.set_point_mass_i(1).expect("set_point_mass_i");
    assert_eq!(bin.point_mass_i, 1);
    assert_eq!(bin.donor_i, 2);
    assert!((bin.donor.mstar / MSUN - 0.8).abs() < 1.0e-6);
    assert!(matches!(bin.companion, BinaryCompanion::PointMass { .. }));
    assert!((bin.accretor_mass_g() / MSUN - 1.0).abs() < 1.0e-6);
}

#[test]
fn test_brott_2011_wind_regimes_are_positive_and_continuous() {
    let w_hot =
        eval_brott_wind_msun_yr(5.0e5, 60.0, 12.0, 42_000.0, 0.70, 0.02);
    let w_bistab =
        eval_brott_wind_msun_yr(5.0e5, 60.0, 35.0, 23_000.0, 0.70, 0.02);
    let w_wr = eval_brott_wind_msun_yr(6.0e5, 35.0, 3.0, 80_000.0, 0.05, 0.02);
    let w_blend =
        eval_brott_wind_msun_yr(5.5e5, 45.0, 8.0, 50_000.0, 0.55, 0.02);
    for w in [w_hot, w_bistab, w_wr, w_blend] {
        assert!(w > 0.0 && w.is_finite(), "invalid Brott wind rate {w}");
    }
    assert!(w_wr > w_hot);
}

fn copy_dir_verbatim(src: &std::path::Path, dst: &std::path::Path) {
    fs::create_dir_all(dst).expect("create verbatim copy dir");
    for entry in fs::read_dir(src).expect("read src case dir") {
        let entry = entry.expect("dir entry");
        let ty = entry.file_type().expect("file type");
        let dst_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_verbatim(&entry.path(), &dst_path);
        } else if ty.is_file() {
            fs::copy(entry.path(), &dst_path).expect("copy case file");
        }
    }
}

fn assert_non_double_bh_raw_cases_pass_strict(suite_dir: &std::path::Path) {
    // Issue #299: `extra_terminal_output_file` is classified as terminal-only
    // (`DECISIONS #26`), so all 7 non-`double_bh` binary cases pass strict
    // validation without cleaning.
    for &case_name in &BINARY_CASES {
        if case_name == "double_bh" {
            continue;
        }
        let raw_dir = suite_dir.join(case_name);
        let bctrl = BinaryInlistControls::from_case_dir(&raw_dir)
            .expect("raw binary controls");
        let active_inlists: &[String] = if bctrl.evolve_both_stars {
            &bctrl.inlist_names
        } else {
            &bctrl.inlist_names[..1]
        };
        for name in active_inlists {
            if name.trim().is_empty() {
                continue;
            }
            let bundle = InlistBundle::from_file_in_case_dir(&raw_dir, name)
                .unwrap_or_else(|e| {
                    panic!(
                        "raw {case_name}/{name} must pass strict validation: {e}"
                    )
                });
            assert_eq!(
                bundle.validation_report.unsupported_controls.len(),
                0,
                "raw {case_name}/{name} has unsupported controls: {:?}",
                bundle.validation_report.unsupported_controls
            );
        }
    }
}

#[test]
fn test_recursive_binary_inlists_and_all_8_cases_capped_4_steps() {
    let Some(suite_dir) = resolve_suite_dir() else {
        return;
    };

    for &case_name in &BINARY_CASES {
        BinaryInlistControls::from_case_dir(&suite_dir.join(case_name))
            .unwrap_or_else(|e| {
                panic!(
                    "raw $MESA_DIR {case_name} binary inlist must parse without unsupported binary controls: {e}"
                )
            });
    }
    let raw_dbh_err = run_binary_test_suite_case(
        &suite_dir.join("double_bh"),
        Some(4),
    )
    .expect_err(
        "raw $MESA_DIR double_bh must fail on unsupported star controls",
    );
    assert!(
        matches!(raw_dbh_err, StarError::InvalidInput(_))
            && raw_dbh_err.to_string().contains("unsupported control"),
        "unexpected raw double_bh error: {raw_dbh_err}"
    );
    assert_non_double_bh_raw_cases_pass_strict(&suite_dir);
    let raw_gr_tmp = TempDirGuard::new("raw_gr_strict");
    let raw_gr_dir = raw_gr_tmp.root.join("jdot_gr_check");
    copy_dir_verbatim(&suite_dir.join("jdot_gr_check"), &raw_gr_dir);
    let raw_gr_rep = run_binary_test_suite_case(&raw_gr_dir, Some(4)).expect(
        "unmodified jdot_gr_check copy must pass strict validation (#299)",
    );
    assert_eq!(raw_gr_rep.total_steps, 4);
    assert_eq!(raw_gr_rep.termination_code, "MaxStepsCap");

    let tmp = TempDirGuard::new("all8");
    let cleaned_suite = tmp.root.join("binary").join("test_suite");
    for &case_name in &BINARY_CASES {
        copy_cleaned_binary_case(
            &suite_dir.join(case_name),
            &cleaned_suite.join(case_name),
        )
        .expect("copy cleaned binary suite case");
    }

    let dbh_ctrl =
        BinaryInlistControls::from_case_dir(&cleaned_suite.join("double_bh"))
            .expect("cleaned double_bh recursive inlists");
    assert!(dbh_ctrl.evolve_both_stars);
    assert!(dbh_ctrl.new_ignore_rlof_flag);
    assert_eq!(dbh_ctrl.mdot_scheme, "contact");
    assert!((dbh_ctrl.m1_msun - 70.0).abs() < 1.0e-12);
    assert!((dbh_ctrl.m2_msun - 65.0).abs() < 1.0e-12);
    assert!((dbh_ctrl.initial_period_in_days - 1.20).abs() < 1.0e-12);

    let gr_ctrl = BinaryInlistControls::from_case_dir(
        &cleaned_suite.join("jdot_gr_check"),
    )
    .expect("cleaned jdot_gr_check inlists");
    assert!(gr_ctrl.initial_period_in_days <= 0.0);
    assert!((gr_ctrl.initial_separation_in_rsuns - 2.0).abs() < 1.0e-12);
    assert!(gr_ctrl.do_jdot_gr && !gr_ctrl.do_jdot_mb && gr_ctrl.do_jdot_ml);

    for &case_name in &BINARY_CASES {
        let case_dir = cleaned_suite.join(case_name);
        let bctrl =
            BinaryInlistControls::from_case_dir(&case_dir).expect("bctrl");
        let mut extras = build_test_oracle_extras(case_name, &bctrl);
        let rep = run_binary_test_suite_case_with_extras(
            &case_dir,
            Some(4),
            BinaryLogsOptions::default(),
            InlistLoadOptions::default(),
            &mut *extras,
        )
        .unwrap_or_else(|e| panic!("capped {case_name} errored: {e}"));
        assert!(
            !rep.passed && rep.termination_code == "MaxStepsCap",
            "capped case {} must stop at MaxStepsCap without passing: {rep:?}",
            rep.case_name
        );
        assert!(!extras.failed(), "oracle failed in {case_name}");
        assert_eq!(rep.total_steps, 4);
        match rep.case_name.as_str() {
            "jdot_gr_check" => assert!(extras.max_rel_err() < 1.0e-2),
            "jdot_ml_check" => assert!(extras.max_rel_err() < 1.0e-4),
            "jdot_ls_check" => assert!(extras.max_rel_err() < 1.0e-12),
            "wind_fed_bhhmxb" => assert!(extras.max_rel_err() <= 1.0e-15),
            _ => {}
        }
    }
}

#[test]
fn test_coupled_double_bh_35_steps_preserves_star_controls() {
    let Some(suite_dir) = resolve_suite_dir() else {
        return;
    };
    let raw_cdir = suite_dir.join("double_bh");
    let raw_err = InlistBundle::from_file_in_case_dir(&raw_cdir, "inlist1")
        .expect_err("raw double_bh inlist1 must reject unsupported controls");
    assert!(
        matches!(raw_err, StarError::InvalidInput(_))
            && raw_err.to_string().contains("unsupported control"),
        "unexpected raw inlist1 error: {raw_err}"
    );
    let tmp = TempDirGuard::new("dbh35");
    let cdir = tmp.root.join("double_bh");
    copy_cleaned_binary_case(&raw_cdir, &cdir).expect("clean double_bh");
    let b1 =
        InlistBundle::from_file_in_case_dir(&cdir, "inlist1").expect("inlist1");
    assert!(b1.controls.rotation_flag);
    assert!(b1.controls.use_gravity_rotation_correction);
    assert!(b1.controls.use_gold2_tolerances);
    assert!(b1.controls.use_ledoux_criterion);
    assert!(b1.controls.use_other_wind);

    let no_hook_err = run_binary_test_suite_case(&cdir, Some(2))
        .expect_err("use_other_wind without hook must fail loudly");
    assert!(
        no_hook_err.to_string().contains("use_other_wind"),
        "unexpected missing wind hook error: {no_hook_err}"
    );

    let mut b_probe = b1;
    b_probe.controls.initial_mass = 10.0;
    b_probe.job.relax_z = false;
    b_probe.job.relax_y = false;
    b_probe.job.relax_initial_z = false;
    b_probe.job.relax_initial_y = false;
    let seed = load_zams_model(b_probe.controls.clone(), b_probe.job.clone())
        .expect("probe ZAMS");
    let mut star_extras = DefaultStarExtras;
    let s_probe = init_or_update_model_from_bundle(
        &b_probe,
        Some(seed),
        &mut star_extras,
    )
    .expect("init probe");
    let bctrl = BinaryInlistControls::from_case_dir(&cdir).expect("bctrl");
    let mut extras = DoubleBhExtras {
        initial_period_in_days: bctrl.initial_period_in_days,
        max_qhe_diff: 0.0,
        failed_mid_run: false,
    };
    let bin_probe =
        BinarySystem::from_controls_two_stars(s_probe.clone(), s_probe, bctrl)
            .expect("two-star probe");
    assert!(bin_probe.donor.controls.use_gravity_rotation_correction);
    assert!(bin_probe.donor.controls.use_gold2_tolerances);
    assert!(bin_probe.donor.controls.use_ledoux_criterion);
    assert!(bin_probe.donor.controls.use_other_wind);

    let rep = run_binary_test_suite_case_with_extras(
        &cdir,
        Some(35),
        BinaryLogsOptions::default(),
        InlistLoadOptions::default(),
        &mut extras,
    )
    .expect("double_bh 35 steps");
    assert!(
        !rep.passed && rep.termination_code == "MaxStepsCap",
        "double_bh 35-step run should stop at MaxStepsCap without passing: {rep:?}"
    );
    assert!(!extras.failed_mid_run);
    assert_eq!(rep.total_steps, 35);
    assert!(rep.final_m1_msun > 69.0 && rep.final_m2_msun > 64.0);
}

#[test]
fn test_uncapped_jdot_ls_check_campaign() {
    let Some(suite_dir) = resolve_suite_dir() else {
        return;
    };
    let raw_cdir = suite_dir.join("jdot_ls_check");
    let raw_b1 = InlistBundle::from_file_in_case_dir(&raw_cdir, "inlist1")
        .expect("raw jdot_ls_check/inlist1 must pass strict validation (#299)");
    assert_eq!(
        raw_b1.validation_report.unsupported_controls.len(),
        0,
        "unexpected unsupported controls: {:?}",
        raw_b1.validation_report.unsupported_controls
    );
    assert!(
        raw_b1.validation_report.terminal_output_keys.contains(&(
            String::from("controls"),
            String::from("extra_terminal_output_file")
        )),
        "expected extra_terminal_output_file in terminal_output_keys: {:?}",
        raw_b1.validation_report.terminal_output_keys
    );
    let tmp = TempDirGuard::new("jdot_ls_raw");
    let staged_raw = tmp.root.join("jdot_ls_check");
    copy_dir_verbatim(&raw_cdir, &staged_raw);
    let mut extras = JdotLsCheckExtras::default();
    let rep = run_binary_test_suite_case_with_extras(
        &staged_raw,
        None,
        BinaryLogsOptions::default(),
        InlistLoadOptions::default(),
        &mut extras,
    )
    .expect("uncapped unmodified jdot_ls_check");
    assert!(rep.passed, "uncapped jdot_ls_check failed: {rep:?}");
    assert_eq!(rep.termination_code, "max_age");
    assert!(rep.total_steps >= 100);
    assert!(!extras.failed_mid_run);
    assert!(extras.max_rel_err < 1.0e-12);
}

fn assert_slow_uncapped_case(
    case_name: &str,
) -> (
    flexmesa_binary::BinaryCaseRunReport,
    Box<dyn TestOracleExtras>,
) {
    let suite_dir =
        resolve_suite_dir().expect("MESA_DIR binary/test_suite must exist");
    let tmp = TempDirGuard::new(case_name);
    let cleaned = tmp.root.join(case_name);
    copy_cleaned_binary_case(&suite_dir.join(case_name), &cleaned)
        .expect("clean uncapped binary case");
    let bctrl = BinaryInlistControls::from_case_dir(&cleaned).expect("bctrl");
    let mut extras = build_test_oracle_extras(case_name, &bctrl);
    let rep = run_binary_test_suite_case_with_extras(
        &cleaned,
        None,
        BinaryLogsOptions::default(),
        InlistLoadOptions::default(),
        &mut *extras,
    )
    .unwrap_or_else(|e| panic!("uncapped {case_name} failed: {e}"));
    assert!(rep.passed, "uncapped {case_name} report failed: {rep:?}");
    assert!(!extras.failed(), "oracle failed in uncapped {case_name}");
    (rep, extras)
}

#[test]
#[ignore = "unverified uncapped binary campaign; tracked in #143 and #160"]
fn test_uncapped_jdot_gr_check_campaign() {
    let (rep, extras) = assert_slow_uncapped_case("jdot_gr_check");
    assert_eq!(rep.termination_code, "extras_binary_check_model");
    assert!(extras.max_rel_err() < 1.0e-2);
}

#[test]
#[ignore = "unverified uncapped binary campaign; tracked in #143 and #160"]
fn test_uncapped_star_plus_point_mass_explicit_mdot_campaign() {
    let (rep, _) =
        assert_slow_uncapped_case("star_plus_point_mass_explicit_mdot");
    assert_eq!(rep.termination_code, "xa_central_lower_limit");
    assert!(rep.final_m1_msun < 5.0 && rep.final_m2_msun > 10.0);
}

#[test]
#[ignore = "unverified uncapped binary campaign; tracked in #143 and #160"]
fn test_uncapped_star_plus_point_mass_campaign() {
    let (rep, _) = assert_slow_uncapped_case("star_plus_point_mass");
    assert_eq!(rep.termination_code, "xa_central_lower_limit");
    assert!(rep.final_m1_msun < 5.0 && rep.final_m2_msun > 10.0);
}

#[test]
#[ignore = "unverified uncapped binary campaign; tracked in #143 and #160"]
fn test_uncapped_jdot_ml_check_campaign() {
    let (rep, extras) = assert_slow_uncapped_case("jdot_ml_check");
    assert_eq!(rep.termination_code, "max_age");
    assert!(extras.max_rel_err() < 1.0e-4);
}

#[test]
#[ignore = "unverified uncapped binary campaign; tracked in #143 and #160"]
fn test_uncapped_evolve_both_stars_campaign() {
    let (rep, _) = assert_slow_uncapped_case("evolve_both_stars");
    assert_eq!(rep.termination_code, "accretor_overflow_terminate");
}

#[test]
#[ignore = "unverified uncapped binary campaign; tracked in #143 and #160"]
fn test_uncapped_wind_fed_bhhmxb_campaign() {
    let (rep, extras) = assert_slow_uncapped_case("wind_fed_bhhmxb");
    assert_eq!(rep.termination_code, "xa_central_lower_limit");
    assert!(extras.max_rel_err() <= 1.0e-15);
}

#[test]
#[ignore = "unverified uncapped binary campaign; tracked in #143 and #160"]
fn test_uncapped_double_bh_campaign() {
    let (rep, _) = assert_slow_uncapped_case("double_bh");
    assert_eq!(rep.termination_code, "DoubleBhPass");
    assert!(rep.final_m1_msun > 20.0 && rep.final_m2_msun > 20.0);
}

#[test]
fn test_roche_deformation_tables_and_synchronicity_switch() {
    let Some(_) = resolve_suite_dir() else {
        return;
    };
    let tables = RocheDeformationTables::global()
        .expect("roche_data tables should load from MESA_DIR");
    let fp = tables.eval_fp(-0.5, 1.35);
    let ft = tables.eval_ft(-0.5, 1.35);
    let irot = tables.eval_irot(-0.5, 1.35);
    assert!(fp > 0.3 && fp < 1.0, "unexpected fp={fp}");
    assert!(ft > 0.3 && ft < 1.0, "unexpected ft={ft}");
    assert!((1.0..5.0).contains(&irot), "unexpected irot={irot}");
    assert_eq!(tables.eval_psi(-0.5, 0.0), -9.0e99);

    let period_unit_omega = 2.0 * std::f64::consts::PI;
    let f_sync =
        eval_synchronicity_switch(1.0, period_unit_omega, 0.8, 20.0, 0.05);
    assert!((f_sync - 1.0).abs() < 1.0e-3);
    let f_far =
        eval_synchronicity_switch(0.1, period_unit_omega, 0.8, 20.0, 0.05);
    assert!(f_far < 0.05);
}

#[test]
fn test_contact_solver_fixed_delta_mdot_crosses_zero_for_donor_switch() {
    let ctrl = BinaryInlistControls {
        mdot_scheme: String::from("contact"),
        implicit_scheme_tolerance: 1.0e-3,
        initial_change_factor: 1.5,
        ..BinaryInlistControls::default()
    };
    let mut solver =
        ImplicitMdotSolver::from_controls(&ctrl, 1.5).expect("solver");
    let m0 = -1.0e-5 * MSUN / SECYER;
    // Try 1: accretor overflows more than donor (f < 0) -> steps toward 0 using fixed_delta_mdot.
    let p1 = solver
        .evaluate_trial("contact", 1, 20, m0, m0, 0.02, 0.08, 0.9)
        .expect("p1");
    let ImplicitStepAction::TryNextMdot(m1) = p1 else {
        panic!("expected TryNextMdot on try 1, got {p1:?}");
    };
    assert!(m1 > m0 && m1 < 0.0);
    let mut m_cur = m1;
    let mut crossed_zero = false;
    for try_idx in 2..=10 {
        let prop = solver
            .evaluate_trial(
                "contact", try_idx, 20, m_cur, m_cur, 0.02, 0.08, 0.9,
            )
            .expect("prop");
        if let ImplicitStepAction::TryNextMdot(m_next) = prop {
            m_cur = m_next;
            if m_cur > 0.0 {
                crossed_zero = true;
                break;
            }
        }
    }
    assert!(
        crossed_zero,
        "contact solver must cross zero when f < 0 persists"
    );
    solver.flip_for_contact_donor_switch();
    let q_new = 1.0_f64 / 0.9;
    let gap_a_eq = 0.05 * flexmesa_math::pow(q_new, -0.52);
    let p_conv = solver
        .evaluate_trial("contact", 5, 20, -m_cur, -m_cur, 0.05, gap_a_eq, q_new)
        .expect("p_conv");
    assert_eq!(p_conv, ImplicitStepAction::Converged(-m_cur));
}

#[test]
fn test_edot_isotropic_eccentricity_enhancement_and_mdot_redo_boost() {
    let r_prof = [10.0 * RSUN, 9.5 * RSUN];
    let p_prof = [1.0e5, 2.0e5];
    let t_prof = [3.0e4, 3.5e4];
    let mu_prof = [0.62, 0.62];
    let gam_prof = [1.66, 1.66];
    let edot_enh = eval_edot_enhancement_isotropic(
        20.0 * MSUN,
        15.0 * MSUN,
        10.0 * RSUN,
        9.8 * RSUN,
        0.3,
        1.0e-8,
        1.0e5,
        3.0e4,
        0.62,
        &r_prof,
        &p_prof,
        &t_prof,
        &mu_prof,
        &gam_prof,
        false,
        200,
    );
    assert!(edot_enh.is_finite());

    let mut c1 = StarControls {
        initial_mass: 20.0,
        zbase: 0.02,
        rotation_flag: true,
        surf_omega_div_omega_crit_limit: 0.98,
        ..StarControls::default()
    };
    c1.mix_wind.surf_omega_div_omega_crit_tol = 0.02;
    c1.mix_wind.max_mdot_redo_cnt = 10;
    c1.mix_wind.mdot_revise_factor = 1.2;
    c1.mix_wind.implicit_mdot_boost = 0.1;
    let mut star =
        load_zams_model(c1, StarJobControls::default()).expect("20 Msun ZAMS");
    for k in 0..star.nz {
        star.omega[k] = 1.0e-3;
    }
    // Verify check_termination_limits does NOT terminate on surf_omega_div_omega_crit_limit.
    let term = check_termination_limits(&star)
        .expect("check_termination_limits must not error on high omega");
    assert!(
        !matches!(
            term,
            flexmesa_star::LimitCheckOutcome::Terminate { ref code, .. }
                if code == "surf_omega_div_omega_crit_limit"
        ),
        "surf_omega_div_omega_crit_limit must be handled by mdot_redo, not termination"
    );
    let mut redo = MdotRedoState::setup(&mut star);
    let act = redo.select_action(&mut star, 1.0e2 * SECYER);
    assert!(matches!(
        act,
        SelectMdotAction::Accept | SelectMdotAction::RedoWithMdot(_)
    ));
}

#[test]
fn test_double_bh_che_diffusivity_stiffness_and_bh_collapse_invariants() {
    let Some(suite_dir) = resolve_suite_dir() else {
        return;
    };
    let raw_cdir = suite_dir.join("double_bh");
    let tmp = TempDirGuard::new("dbh_che");
    let cdir = tmp.root.join("double_bh");
    copy_cleaned_binary_case(&raw_cdir, &cdir).expect("clean double_bh");
    let b1 =
        InlistBundle::from_file_in_case_dir(&cdir, "inlist1").expect("inlist1");
    let mw = &b1.controls.mix_wind;
    assert!(mw.premix_omega);
    assert!((mw.am_nu_st_factor - 1.0).abs() < 1.0e-12);
    assert!(mw.am_nu_sh_factor.abs() < 1.0e-12);
    assert!((mw.am_d_mix_factor - 1.0 / 30.0).abs() < 1.0e-5);
    assert_eq!(mw.num_cells_for_smooth_gradl_composition_term, 2);

    let mut probe = b1.clone();
    probe.controls.initial_mass = 20.0;
    probe.job.relax_z = false;
    probe.job.relax_y = false;
    probe.job.relax_initial_z = false;
    probe.job.relax_initial_y = false;
    let seed = load_zams_model(probe.controls.clone(), probe.job.clone())
        .expect("zams");
    let mut extras = DefaultStarExtras;
    let mut star =
        init_or_update_model_from_bundle(&probe, Some(seed), &mut extras)
            .expect("init rotating star");
    let bctrl = BinaryInlistControls::from_case_dir(&cdir).expect("bctrl");
    let mut bin = BinarySystem::from_controls_two_stars(
        star.clone(),
        star.clone(),
        bctrl,
    )
    .expect("two-star system");
    assert!((bin.donor.controls.varcontrol_target - 1.0e-3).abs() < 1.0e-12);

    // Stress low-shear Spruit-Tayler + extreme boundary stiffness (> 2^53 raw).
    let om_sync = 2.0 * std::f64::consts::PI / bin.period;
    let nz_f = f64::from(u32::try_from(star.nz).unwrap_or(1));
    for k in 0..star.nz {
        let k_f = f64::from(u32::try_from(k).unwrap_or(0));
        star.omega[k] = om_sync * (1.0 + 1.0e-6 * k_f / nz_f);
    }
    star.dt = 1.0e4 * SECYER;
    flexmesa_star::update_rotation_distortion_factors(&mut star);
    flexmesa_star::set_hydro_vars(&mut star, false, false)
        .expect("set_hydro_vars with CHE");
    assert!(
        star.mixing_type
            .contains(&flexmesa_const::MixingType::RotationMixing),
        "CHE radiative envelope must activate RotationMixing (D_ES > 0)"
    );
    for mlt_k in &mut star.mlt_d {
        if *mlt_k > 0.0 {
            *mlt_k = 1.0e22;
        }
    }
    let j_before: f64 = star
        .dm_bar
        .iter()
        .zip(&star.j_rot)
        .map(|(m, j)| m * j)
        .sum();
    flexmesa_star::set_rotation_mixing_info(&mut star)
        .expect("set_rotation_mixing_info");
    // `am_nu_ST_factor = 1 > 0`, so upstream runs `set_ST`
    // (`star/private/rotation_mix_info.f90:178-181`): `nu_ST` is a finite,
    // non-negative viscosity that is positive on some faces of this rotating
    // model. The removed stand-in returned `nu_ST = 0` everywhere.
    let nu_st = &star.aux.rotmix.nu_st[..star.nz];
    assert!(nu_st.iter().all(|&v| v.is_finite() && v >= 0.0));
    assert!(
        nu_st.iter().any(|&v| v > 0.0),
        "set_ST gives nu_ST = 0 on every face"
    );
    let dt_star = star.dt;
    flexmesa_star::sync_post_henyey_rotation(&mut star, dt_star)
        .expect("sync_post_henyey_rotation");
    assert!(star.omega.iter().all(|w| w.is_finite() && *w > 0.0));
    assert!(star.j_rot.iter().all(|j| j.is_finite() && *j > 0.0));
    let j_after: f64 = star
        .dm_bar
        .iter()
        .zip(&star.j_rot)
        .map(|(m, j)| m * j)
        .sum();
    assert!(
        (j_after - j_before).abs() / j_before.abs().max(1.0) < 1.0e-12,
        "solve_omega_transport must conserve J_tot: before={j_before}, after={j_after}"
    );

    // Verify dynamic BH1 collapse clears stale termination reason & preserves [m1, m2].
    bin.donor.mstar = 35.0 * MSUN;
    if let BinaryCompanion::Star(ref mut s2) = bin.companion {
        s2.mstar = 32.0 * MSUN;
    }
    bin.last_termination_reason = Some(String::from("max_model_number"));
    bin.change_factor = 3.2;
    bin.set_point_mass_i(1).expect("set_point_mass_i");
    assert!(bin.last_termination_reason.is_none());
    assert!(
        (bin.change_factor - bin.controls.max_change_factor).abs() < 1.0e-12
    );
    let [m1_g, m2_g] = bin.star_masses_g();
    assert!((m1_g / MSUN - 35.0).abs() < 1.0e-12);
    assert!((m2_g / MSUN - 32.0).abs() < 1.0e-12);
}
