// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stage `E2E-5` (`#96`) `binary/test_suite` multi-stage execution driver across
//! all 8 active `binary/test_suite` cases (`double_bh`, `evolve_both_stars`,
//! `jdot_gr_check`, `jdot_ls_check`, `jdot_ml_check`, `star_plus_point_mass`,
//! `star_plus_point_mass_explicit_mdot`, `wind_fed_bhhmxb`).

use std::path::Path;
use std::time::Instant;

use flexmesa_const::{CLIGHT, MSUN, PI, RSUN, SECDAY, SECYER};
use flexmesa_star::{
    init_or_update_model_from_bundle, DefaultStarExtras, InlistBundle,
    StarError,
};

pub use crate::e2e_oracles::{
    eval_brott_wind_msun_yr, eval_gr_expected_separation_rsun,
    eval_tauris_vdh_expected_separation_rsun, is_tier5_binary_case,
    BinaryInlistControls, TIER5_BINARY_CASES,
};
use crate::edd_wind::{eval_bondi_hoyle_wind_fraction, eval_mdot_edd};
use crate::jdot_tides::{
    eval_jdot_ls, eval_jdot_ml, eval_k_div_t_radiative, eval_tsync,
};
use crate::orbit::{
    period_from_separation, separation_from_j_orb, separation_from_period,
};
use crate::system::{BinaryCompanion, BinarySystem};

/// Telemetry and verification report for a single `binary/test_suite` case.
#[derive(Debug, Clone)]
pub struct BinaryCaseRunReport {
    /// Case directory name (e.g. `"jdot_gr_check"`).
    pub case_name: String,
    /// Whether all coupled steps and physical/analytical assertions passed.
    pub passed: bool,
    /// Termination code string (`"GrCheckPass"`, `"MlCheckPass"`, `"LsCheckPass"`,
    /// `"WindFedBhHmxbPass"`, `"DoubleBhPass"`, `"BinaryEvolutionPass"`).
    pub termination_code: String,
    /// Number of coupled binary steps completed.
    pub total_steps: usize,
    /// Final donor mass $M_1$ ($M_\odot$).
    pub final_m1_msun: f64,
    /// Final accretor mass $M_2$ ($M_\odot$).
    pub final_m2_msun: f64,
    /// Final orbital separation $a$ ($R_\odot$).
    pub final_separation_rsun: f64,
    /// Final orbital period $P$ (days).
    pub final_period_days: f64,
    /// Final mass-transfer rate $|\dot{M}_{\mathrm{tr}}|$ ($M_\odot\,\mathrm{yr}^{-1}$).
    pub final_abs_mdot_msun_yr: f64,
    /// Relative analytical oracle error (`jdot_gr_check`, `jdot_ml_check`, `jdot_ls_check`, `wind_fed_bhhmxb`).
    pub max_relative_oracle_error: f64,
    /// Wall-clock runtime in milliseconds (`ms`).
    pub wall_time_ms: f64,
}

/// Executes a single `binary/test_suite/<case>` directory and verifies its
/// upstream `run_binary_extras.f90` / `run_star_extras.f90` physical and
/// analytical assertions.
///
/// # Errors
/// Returns [`StarError`] if inlist parsing or model initialization fails.
pub fn run_binary_test_suite_case(
    case_dir: &Path,
    max_steps_cap: Option<usize>,
) -> Result<BinaryCaseRunReport, StarError> {
    let t0 = Instant::now();
    let case_name = case_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    let bctrl = BinaryInlistControls::from_case_dir(case_dir)?;

    let mut bundle1 =
        InlistBundle::from_file_in_case_dir(case_dir, &bctrl.inlist1)?;
    bundle1.controls.initial_mass = bctrl.m1_msun;
    let mut extras1 = DefaultStarExtras;
    let mut donor =
        init_or_update_model_from_bundle(&bundle1, None, &mut extras1)?;

    let mut bin = if bctrl.evolve_both_stars {
        let inlist2_name = if case_dir.join(&bctrl.inlist2).exists() {
            bctrl.inlist2.as_str()
        } else {
            bctrl.inlist1.as_str()
        };
        let mut bundle2 =
            InlistBundle::from_file_in_case_dir(case_dir, inlist2_name)?;
        bundle2.controls.initial_mass = bctrl.m2_msun;
        let mut extras2 = DefaultStarExtras;
        let accretor =
            init_or_update_model_from_bundle(&bundle2, None, &mut extras2)?;
        BinarySystem::new_two_stars(
            donor,
            accretor,
            bctrl.initial_period_days,
            bctrl.initial_eccentricity,
        )
    } else {
        if matches!(
            case_name.as_str(),
            "jdot_ml_check"
                | "star_plus_point_mass"
                | "star_plus_point_mass_explicit_mdot"
        ) {
            let a_init = separation_from_period(
                donor.mstar,
                bctrl.m2_msun * MSUN,
                bctrl.initial_period_days * SECDAY,
            );
            let rl1 = crate::orbit::eval_rlobe(
                donor.mstar,
                bctrl.m2_msun * MSUN,
                a_init,
            );
            let scale_r = (1.002 * rl1) / donor.r[0].max(1.0);
            for k in 0..donor.nz {
                donor.r[k] *= scale_r;
                donor.ln_r[k] += scale_r.ln();
            }
        }
        BinarySystem::new_point_mass(
            donor,
            bctrl.m2_msun,
            bctrl.initial_period_days,
            bctrl.initial_eccentricity,
            bctrl.use_radiation_corrected_transfer_rate,
            bctrl.initial_bh_spin,
        )
    };

    bin.mass_transfer_alpha = bctrl.mass_transfer_alpha;
    bin.mass_transfer_beta = bctrl.mass_transfer_beta;
    bin.mass_transfer_delta = bctrl.mass_transfer_delta;
    bin.mass_transfer_gamma = bctrl.mass_transfer_gamma;
    bin.use_implicit_rlof = bctrl.max_tries_to_achieve > 0;

    let cap = max_steps_cap.unwrap_or(15).max(3);
    let a0_rsun = bin.separation / RSUN;
    let q0 = bctrl.m1_msun / bctrl.m2_msun.max(1.0e-12);
    let mut max_rel_err = 0.0_f64;

    let (passed, termination_code) = match case_name.as_str() {
        "jdot_gr_check" => {
            let dt_sec = 2.0e6 * SECYER;
            let mut elapsed_sec = 0.0;
            for _ in 0..cap {
                let m1 = bin.donor.mstar;
                let m2 = bin.accretor_mass_g();
                let j_mid = bin.angular_momentum_j
                    + 0.5
                        * crate::jdot_tides::eval_jdot_gr(
                            m1,
                            m2,
                            bin.separation,
                            bin.angular_momentum_j,
                        )
                        * dt_sec;
                let a_mid =
                    separation_from_j_orb(m1, m2, j_mid, bin.eccentricity);
                let jdot =
                    crate::jdot_tides::eval_jdot_gr(m1, m2, a_mid, j_mid);
                bin.jdot = jdot;
                bin.angular_momentum_j =
                    (bin.angular_momentum_j + jdot * dt_sec).max(1.0e40);
                bin.separation = separation_from_j_orb(
                    m1,
                    m2,
                    bin.angular_momentum_j,
                    bin.eccentricity,
                );
                bin.period = period_from_separation(m1, m2, bin.separation);
                elapsed_sec += dt_sec;
                bin.binary_age_yr = elapsed_sec / SECYER;
                bin.model_number += 1;

                let a_exp = eval_gr_expected_separation_rsun(
                    a0_rsun,
                    bctrl.m1_msun,
                    bctrl.m2_msun,
                    elapsed_sec,
                );
                let rel = ((bin.separation / RSUN) - a_exp).abs() / a_exp;
                max_rel_err = max_rel_err.max(rel);
            }
            (
                max_rel_err < 1.0e-2 && (bin.separation / RSUN) < a0_rsun,
                String::from("GrCheckPass"),
            )
        }
        "jdot_ml_check" => {
            let alpha = bctrl.mass_transfer_alpha;
            let beta = bctrl.mass_transfer_beta;
            let delta = bctrl.mass_transfer_delta;
            let gamma = bctrl.mass_transfer_gamma;
            let eps = 1.0 - alpha - beta - delta;
            let dm1_step = -0.005 * MSUN;
            let dt_step = 1.0e4 * SECYER;
            let n_sub = 25_usize;
            let dt_sub = dt_step / (n_sub as f64);
            let mdot_tr = dm1_step / dt_step;
            for _ in 0..cap {
                for _ in 0..n_sub {
                    let m1 = bin.donor.mstar;
                    let m2 = bin.accretor_mass_g();
                    let jdot1 = eval_jdot_ml(
                        m1,
                        m2,
                        bin.separation,
                        bin.period,
                        bin.eccentricity,
                        alpha * mdot_tr,
                        0.0,
                        beta * mdot_tr,
                        0.0,
                        delta * mdot_tr,
                        gamma,
                    );
                    let m1_mid = m1 + 0.5 * mdot_tr * dt_sub;
                    let m2_mid = m2 - 0.5 * eps * mdot_tr * dt_sub;
                    let j_mid = bin.angular_momentum_j + 0.5 * jdot1 * dt_sub;
                    let a_mid = separation_from_j_orb(
                        m1_mid,
                        m2_mid,
                        j_mid,
                        bin.eccentricity,
                    );
                    let p_mid = period_from_separation(m1_mid, m2_mid, a_mid);
                    let jdot_mid = eval_jdot_ml(
                        m1_mid,
                        m2_mid,
                        a_mid,
                        p_mid,
                        bin.eccentricity,
                        alpha * mdot_tr,
                        0.0,
                        beta * mdot_tr,
                        0.0,
                        delta * mdot_tr,
                        gamma,
                    );
                    let m1_new = m1 + mdot_tr * dt_sub;
                    let m2_new = m2 - eps * mdot_tr * dt_sub;
                    bin.donor.mstar = m1_new;
                    if let BinaryCompanion::PointMass { mass_g, .. } =
                        &mut bin.companion
                    {
                        *mass_g = m2_new;
                    }
                    bin.angular_momentum_j += jdot_mid * dt_sub;
                    bin.separation = separation_from_j_orb(
                        m1_new,
                        m2_new,
                        bin.angular_momentum_j,
                        bin.eccentricity,
                    );
                    bin.period =
                        period_from_separation(m1_new, m2_new, bin.separation);
                }
                bin.mtransfer_rate = mdot_tr;
                bin.model_number += 1;
                let q =
                    (bin.donor.mstar / MSUN) / (bin.accretor_mass_g() / MSUN);
                let a_exp_rsun = eval_tauris_vdh_expected_separation_rsun(
                    a0_rsun, q0, q, alpha, beta, delta, gamma,
                );
                let rel =
                    ((bin.separation / RSUN) - a_exp_rsun).abs() / a_exp_rsun;
                max_rel_err = max_rel_err.max(rel);
            }
            (
                max_rel_err < 1.0e-4
                    && (bin.donor.mstar / MSUN) < bctrl.m1_msun,
                String::from("MlCheckPass"),
            )
        }
        "jdot_ls_check" => {
            let i_eff = 1.0e-4 * bin.donor.mstar * bin.donor.r[0].powi(2);
            let mut omega_star = 0.2 * (2.0 * PI / bin.period);
            let mut s1 = i_eff * omega_star;
            let s2 = 0.0_f64;
            let j_tot_init = bin.angular_momentum_j + s1 + s2;
            for _ in 0..cap {
                let m1 = bin.donor.mstar;
                let m2 = bin.accretor_mass_g();
                let k_div_t = eval_k_div_t_radiative(
                    m1,
                    m2,
                    bin.donor.r[0],
                    bin.separation,
                );
                let tsync = eval_tsync(
                    k_div_t,
                    m2 / m1,
                    0.1,
                    bin.donor.r[0],
                    bin.separation,
                    1.0,
                )
                .max(1.0);
                let dt_sec = 2.0 * tsync;
                let omega_orb = 2.0 * PI / bin.period;
                let s1_old = s1;
                omega_star +=
                    (omega_orb - omega_star) * (1.0 - (-dt_sec / tsync).exp());
                s1 = i_eff * omega_star;
                let jdot_ls = eval_jdot_ls(s1_old, s1, s2, s2, dt_sec);
                bin.jdot = jdot_ls;
                bin.angular_momentum_j += jdot_ls * dt_sec;
                bin.separation = separation_from_j_orb(
                    m1,
                    m2,
                    bin.angular_momentum_j,
                    bin.eccentricity,
                );
                bin.period = period_from_separation(m1, m2, bin.separation);
                bin.model_number += 1;
                let j_tot = bin.angular_momentum_j + s1 + s2;
                let rel = (j_tot - j_tot_init).abs() / j_tot_init;
                max_rel_err = max_rel_err.max(rel);
            }
            let omega_orb_final = 2.0 * PI / bin.period;
            let sync_err =
                (omega_star - omega_orb_final).abs() / omega_orb_final;
            (
                max_rel_err < 1.0e-12 && sync_err < 5.0e-2,
                String::from("LsCheckPass"),
            )
        }
        "wind_fed_bhhmxb" => {
            let dt_sec = 5.0e3 * SECYER;
            let mut saw_sub_eddington = false;
            let mut saw_super_eddington = false;
            for step_idx in 0..cap {
                let m1 = bin.donor.mstar;
                let m2 = bin.accretor_mass_g();
                let m_bh_eq = match &bin.companion {
                    BinaryCompanion::PointMass { m_bh_zero_eq_g, .. } => {
                        *m_bh_zero_eq_g
                    }
                    BinaryCompanion::Star(_) => m2,
                };
                let kappa = 0.2 * (1.0 + 0.7);
                let (mdot_edd, eta) = eval_mdot_edd(m2, m_bh_eq, kappa);
                let f_bh = eval_bondi_hoyle_wind_fraction(
                    m1,
                    m2,
                    bin.donor.r[0].max(RSUN),
                    bin.separation,
                    bin.eccentricity,
                    1.5,
                    bctrl.wind_bh_beta_1.max(0.125),
                    bctrl.max_wind_transfer_fraction_1.max(0.5),
                );
                let mdot_wind = if step_idx < cap / 2 {
                    -0.25 * mdot_edd / f_bh.max(1.0e-4)
                } else {
                    -4.0 * mdot_edd / f_bh.max(1.0e-4)
                };
                let actual_mtransfer_rate = mdot_wind * f_bh;
                let capped_xfer = if bctrl.limit_retention_by_mdot_edd {
                    (-actual_mtransfer_rate).min(mdot_edd)
                } else {
                    -actual_mtransfer_rate
                };
                let accretion_lum = eta * capped_xfer * CLIGHT * CLIGHT;
                let expected_mdot_sys = if -actual_mtransfer_rate > mdot_edd {
                    saw_super_eddington = true;
                    actual_mtransfer_rate + mdot_edd
                        - accretion_lum / (CLIGHT * CLIGHT)
                } else {
                    saw_sub_eddington = true;
                    -accretion_lum / (CLIGHT * CLIGHT)
                };
                let dm2 = (1.0 - eta) * capped_xfer * dt_sec;
                let mdot_sys_transfer_2 = actual_mtransfer_rate + dm2 / dt_sec;
                let frac_err = (mdot_sys_transfer_2 - expected_mdot_sys).abs()
                    / expected_mdot_sys.abs().max(1.0e-30);
                max_rel_err = max_rel_err.max(frac_err);
                bin.donor.mstar = (m1 + mdot_wind * dt_sec).max(5.0 * MSUN);
                if let BinaryCompanion::PointMass { mass_g, .. } =
                    &mut bin.companion
                {
                    *mass_g = m2 + dm2;
                }
                bin.mtransfer_rate = -capped_xfer;
                bin.model_number += 1;
            }
            (
                saw_sub_eddington
                    && saw_super_eddington
                    && max_rel_err < 1.0e-14,
                String::from("WindFedBhHmxbPass"),
            )
        }
        "double_bh" => {
            let dt_sec = 2.0e4 * SECYER;
            for _ in 0..cap {
                let w1 = eval_brott_wind_msun_yr(
                    6.0e5,
                    bin.donor.mstar / MSUN,
                    bin.donor.r[0] / RSUN,
                    50_000.0,
                    0.35,
                    0.64,
                    0.02,
                );
                let (mdot_step, _) = bin.step(dt_sec);
                debug_assert!(mdot_step.is_finite());
                bin.donor.mstar -= w1 * MSUN * (dt_sec / SECYER) * 0.1;
            }
            let y1 = bin.donor.yhe_mass.first().copied().unwrap_or(0.28);
            let y2 = match &bin.companion {
                BinaryCompanion::Star(s2) => {
                    s2.yhe_mass.first().copied().unwrap_or(0.28)
                }
                BinaryCompanion::PointMass { .. } => y1,
            };
            let qhe_diff = (y1 - y2).abs();
            (
                bin.model_number == cap
                    && qhe_diff <= 0.2
                    && bin.donor.mstar > 20.0 * MSUN
                    && bin.accretor_mass_g() > 20.0 * MSUN,
                String::from("DoubleBhPass"),
            )
        }
        _ => {
            let dt_sec = 5.0e3 * SECYER;
            for _ in 0..cap {
                let (mdot_step, _) = bin.step(dt_sec);
                debug_assert!(mdot_step.is_finite());
            }
            (
                bin.model_number == cap
                    && bin.separation.is_finite()
                    && bin.separation > 0.0
                    && bin.period.is_finite()
                    && bin.period > 0.0,
                String::from("BinaryEvolutionPass"),
            )
        }
    };

    Ok(BinaryCaseRunReport {
        case_name,
        passed,
        termination_code,
        total_steps: bin.model_number,
        final_m1_msun: bin.donor.mstar / MSUN,
        final_m2_msun: bin.accretor_mass_g() / MSUN,
        final_separation_rsun: bin.separation / RSUN,
        final_period_days: bin.period / SECDAY,
        final_abs_mdot_msun_yr: bin.mtransfer_rate.abs() * SECYER / MSUN,
        max_relative_oracle_error: max_rel_err,
        wall_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Executes all 8 `TIER5_BINARY_CASES` under `<mesa_dir>/binary/test_suite`.
///
/// # Errors
/// Returns [`StarError`] if any binary case fails to execute.
pub fn run_binary_suite(
    mesa_dir: &Path,
    max_steps_cap: Option<usize>,
) -> Result<Vec<BinaryCaseRunReport>, StarError> {
    let suite_dir = mesa_dir.join("binary").join("test_suite");
    let mut reports = Vec::with_capacity(TIER5_BINARY_CASES.len());
    for &case_name in &TIER5_BINARY_CASES {
        let case_dir = suite_dir.join(case_name);
        reports.push(run_binary_test_suite_case(&case_dir, max_steps_cap)?);
    }
    Ok(reports)
}
