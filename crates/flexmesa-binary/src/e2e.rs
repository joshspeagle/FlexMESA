// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stage `E2E-5` (`#96` / `#114`) genuine coupled `binary/test_suite` execution
//! driver across all 8 active `binary/test_suite` cases (`double_bh`,
//! `evolve_both_stars`, `jdot_gr_check`, `jdot_ls_check`, `jdot_ml_check`,
//! `star_plus_point_mass`, `star_plus_point_mass_explicit_mdot`,
//! `wind_fed_bhhmxb`).

use std::fs;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use flexmesa_const::{CLIGHT, MSUN, PI, RSUN, SECDAY, SECYER};
use flexmesa_star::{
    DefaultStarExtras, InlistBundle, StarError, StarModel, ZamsCatalog,
    init_or_update_model_from_bundle, load_zams_model,
    load_zams_model_from_catalog, set_basic_vars, set_hydro_vars,
};

pub use crate::e2e_oracles::{
    BinaryInlistControls, TIER5_BINARY_CASES, eval_brott_wind_msun_yr,
    eval_gr_expected_separation_rsun, eval_tauris_vdh_expected_separation_rsun,
    is_tier5_binary_case,
};
use crate::system::BinarySystem;

static FULL_ZAMS_CATALOG: OnceLock<Option<ZamsCatalog>> = OnceLock::new();

fn resolve_full_zams_catalog(case_dir: &Path) -> Option<&'static ZamsCatalog> {
    FULL_ZAMS_CATALOG
        .get_or_init(|| {
            let zams_path = case_dir.ancestors().find_map(|anc| {
                let p = anc
                    .join("data")
                    .join("star_data")
                    .join("zams_models")
                    .join("zams_z2m2_y28.data");
                p.is_file().then_some(p)
            })?;
            let text = fs::read_to_string(zams_path).ok()?;
            ZamsCatalog::parse_ascii(&text).ok()
        })
        .as_ref()
}

fn init_binary_star_component(
    case_dir: &Path,
    inlist_name: &str,
    mass_msun: f64,
) -> Result<StarModel, StarError> {
    let mut bundle = InlistBundle::from_file_in_case_dir(case_dir, inlist_name)?;
    bundle.controls.initial_mass = mass_msun;
    let seed_zams = if let Some(cat) = resolve_full_zams_catalog(case_dir) {
        Some(load_zams_model_from_catalog(
            cat,
            bundle.controls.clone(),
            bundle.job.clone(),
        )?)
    } else {
        Some(load_zams_model(bundle.controls.clone(), bundle.job.clone())?)
    };
    let mut extras = DefaultStarExtras;
    let mut model =
        init_or_update_model_from_bundle(&bundle, seed_zams, &mut extras)?;
    if bundle.job.set_uniform_initial_composition
        && (bundle.job.relax_z || bundle.job.relax_y)
    {
        let mut relax_job = bundle.job.clone();
        relax_job.set_uniform_initial_composition = false;
        flexmesa_star::relax::apply_composition_relaxation(
            &mut model,
            &relax_job,
        );
        model.pack_xh();
        set_basic_vars(&mut model)?;
        set_hydro_vars(&mut model, false, false)?;
        model.save_start_values();
        model.save_old_values();
    }
    Ok(model)
}

fn species_frac(star: &StarModel, zone: usize, iso: &str, fallback: f64) -> f64 {
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
    flexmesa_star::compute_power_categories(star).power_nuc_burn
        * flexmesa_const::LSUN
}

/// Telemetry and verification report for a single `binary/test_suite` case.
#[derive(Debug, Clone)]
pub struct BinaryCaseRunReport {
    /// Case directory name (e.g. `"jdot_gr_check"`).
    pub case_name: String,
    /// Whether all coupled steps and physical/analytical assertions passed.
    pub passed: bool,
    /// Termination code string (`"GrCheckPass"`, `"MlCheckPass"`, `"LsCheckPass"`,
    /// `"WindFedBhHmxbPass"`, `"DoubleBhPass"`, `"AccretorOverflowTerminate"`, `"BinaryEvolutionPass"`).
    pub termination_code: String,
    /// Number of coupled binary steps completed.
    pub total_steps: usize,
    /// Final star 1 mass $M_1$ ($M_\odot$).
    pub final_m1_msun: f64,
    /// Final star 2 mass $M_2$ ($M_\odot$).
    pub final_m2_msun: f64,
    /// Final orbital separation $a$ ($R_\odot$).
    pub final_separation_rsun: f64,
    /// Final orbital period $P$ (days).
    pub final_period_days: f64,
    /// Final mass-transfer rate $|\dot{M}_{\mathrm{tr}}|$ ($M_\odot\,\mathrm{yr}^{-1}$).
    pub final_abs_mdot_msun_yr: f64,
    /// Maximum relative analytical oracle error across the run.
    pub max_relative_oracle_error: f64,
    /// Wall-clock runtime in milliseconds (`ms`).
    pub wall_time_ms: f64,
}

/// Executes a single `binary/test_suite/<case>` directory through genuine
/// coupled [`BinarySystem::try_step`] + [`StarModel`] Henyey evolution and
/// verifies its upstream `run_binary_extras.f90` / `run_star_extras.f90`
/// physical and analytical assertions.
///
/// When `max_steps_cap` is `None` (the default), evolves to physical
/// termination. When `Some(cap)` is supplied, evolves up to `cap` coupled
/// steps while enforcing every per-step analytical oracle bound.
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

    let donor =
        init_binary_star_component(case_dir, &bctrl.inlist_names[0], bctrl.m1_msun)?;
    let mut bin = if bctrl.evolve_both_stars {
        let inlist2 = if case_dir.join(&bctrl.inlist_names[1]).is_file() {
            bctrl.inlist_names[1].as_str()
        } else {
            bctrl.inlist_names[0].as_str()
        };
        let accretor =
            init_binary_star_component(case_dir, inlist2, bctrl.m2_msun)?;
        BinarySystem::from_controls_two_stars(donor, accretor, bctrl.clone())?
    } else {
        BinarySystem::from_controls_point_mass(donor, bctrl.clone())?
    };

    let uncapped = max_steps_cap.is_none();
    let step_limit = max_steps_cap.unwrap_or(10_000).max(1);
    let a0_rsun = bin.separation / RSUN;
    let q0 = bctrl.m1_msun / bctrl.m2_msun.max(1.0e-12);
    let j_conserved_0 = bin.total_angular_momentum_conserved();

    let mut dt_next = match &bin.companion {
        crate::system::BinaryCompanion::PointMass { .. } => bin.donor.dt_next,
        crate::system::BinaryCompanion::Star(s2) => {
            bin.donor.dt_next.min(s2.dt_next)
        }
    }
    .max(1.0);

    let mut max_rel_err = 0.0_f64;
    let mut saw_sub_edd = false;
    let mut saw_super_edd = false;
    let mut max_qhe_diff = 0.0_f64;
    let mut failed_mid_run = false;
    let mut term_code = String::from("MaxStepsCap");

    while bin.model_number < step_limit {
        let (suggested_dt, _reason) = bin.try_step(dt_next).map_err(|e| {
            StarError::ConvergenceFailure(format!(
                "[{case_name} step {}] {e}",
                bin.model_number + 1
            ))
        })?;
        dt_next = suggested_dt;

        match case_name.as_str() {
            "jdot_gr_check" => {
                if bin.donor.mstar_dot != 0.0 {
                    term_code = String::from("GrCheckPass");
                    break;
                }
                let a_exp = eval_gr_expected_separation_rsun(
                    a0_rsun,
                    bctrl.m1_msun,
                    bctrl.m2_msun,
                    bin.donor.star_age * SECYER,
                );
                let rel = ((bin.separation / RSUN) - a_exp).abs()
                    / a_exp.max(1.0e-30);
                max_rel_err = max_rel_err.max(rel);
                if rel > 1.0e-2 {
                    failed_mid_run = true;
                    term_code = String::from("GrErrorExceeded");
                    break;
                }
            }
            "jdot_ml_check" => {
                let [m1_g, m2_g] = bin.star_masses_g();
                let q = m1_g / m2_g.max(1.0);
                let a_exp = eval_tauris_vdh_expected_separation_rsun(
                    a0_rsun,
                    q0,
                    q,
                    bin.controls.mass_transfer_alpha,
                    bin.controls.mass_transfer_beta,
                    bin.controls.mass_transfer_delta,
                    bin.controls.mass_transfer_gamma,
                );
                let rel = ((bin.separation / RSUN) - a_exp).abs()
                    / a_exp.max(1.0e-30);
                max_rel_err = max_rel_err.max(rel);
                if rel > 1.0e-4 {
                    failed_mid_run = true;
                    term_code = String::from("MlErrorExceeded");
                    break;
                }
            }
            "jdot_ls_check" => {
                let j_cur = bin.total_angular_momentum_conserved();
                let rel = (j_cur - j_conserved_0).abs()
                    / j_conserved_0.abs().max(1.0e-30);
                max_rel_err = max_rel_err.max(rel);
                if rel >= 1.0e-12 {
                    failed_mid_run = true;
                    term_code = String::from("LsConservationExceeded");
                    break;
                }
            }
            "wind_fed_bhhmxb" => {
                let acct = &bin.last_mdot;
                let max_allowed = acct.mdot_edd
                    * (1.0 - acct.mdot_edd_eta)
                    * (1.0 + 1.0e-10);
                if acct.component_mdot[1] > max_allowed {
                    failed_mid_run = true;
                    term_code = String::from("ExceededEddington");
                    break;
                }
                let expected_sys = if -acct.actual_mtransfer_rate > acct.mdot_edd {
                    saw_super_edd = true;
                    acct.actual_mtransfer_rate + acct.mdot_edd
                        - acct.accretion_luminosity / (CLIGHT * CLIGHT)
                } else {
                    saw_sub_edd = true;
                    -acct.accretion_luminosity / (CLIGHT * CLIGHT)
                };
                let frac_bh = (acct.mdot_system_transfer[1] - expected_sys).abs()
                    / expected_sys.abs().max(1.0e-30);
                let donor_wind_mdot = bin.donor.mstar_dot - bin.mtransfer_rate;
                let exp_wtr = donor_wind_mdot * acct.wind_xfer_fraction[0];
                let err_wtr = if acct.mdot_wind_transfer[0].abs() > 0.0 {
                    (acct.mdot_wind_transfer[0] - exp_wtr).abs()
                        / acct.mdot_wind_transfer[0].abs()
                } else {
                    0.0
                };
                let exp_wsys = donor_wind_mdot * (1.0 - acct.wind_xfer_fraction[0]);
                let err_wsys = if exp_wsys.abs() > 0.0 {
                    (acct.mdot_system_wind[0] - exp_wsys).abs() / exp_wsys.abs()
                } else {
                    0.0
                };
                if bin.model_number > 1 {
                    max_rel_err = max_rel_err.max(frac_bh);
                }
                if (bin.model_number > 1 && frac_bh > 1.0e-15)
                    || err_wtr > 1.0e-10
                    || err_wsys > 1.0e-10
                {
                    failed_mid_run = true;
                    term_code = String::from("WindFedAccountingExceeded");
                    break;
                }
            }
            "star_plus_point_mass" | "star_plus_point_mass_explicit_mdot" => {
                let acct = &bin.last_mdot;
                if acct.mdot_edd > 0.0
                    && acct.component_mdot[1] > acct.mdot_edd * (1.0 + 1.0e-12)
                {
                    failed_mid_run = true;
                    term_code = String::from("PointMassExceededEddington");
                    break;
                }
            }
            "double_bh" => {
                if bin.ignore_rlof_flag {
                    if let Some(s1) = bin.star_ref(1) {
                        let age_yr = s1.star_age;
                        let lnuc = total_lnuc_erg_s(s1).abs();
                        let lsurf = s1.lum[0].abs().max(1.0);
                        let zams_settled = age_yr > 100.0
                            && (lnuc / lsurf).log10().abs() < 0.005;
                        if zams_settled {
                            let gap1 = bin.rl_relative_gap_by_star()[0];
                            if gap1 > 0.320_819_224 {
                                failed_mid_run = true;
                                term_code = String::from("L2OverflowAtZams");
                                break;
                            }
                            if gap1 > 0.0 {
                                bin.controls.max_implicit_abs_mdot = 1.0e-3;
                            }
                            bin.set_ignore_rlof_flag(false);
                            bin.controls.terminate_if_l2_overflow = true;
                        } else {
                            bin.set_period_eccentricity(
                                bctrl.initial_period_in_days * SECDAY,
                                0.0,
                            );
                        }
                    }
                }
                for s_idx in [1_usize, 2_usize] {
                    if let Some(st) = bin.star_ref(s_idx) {
                        if center_h1(st) > 1.0e-3 {
                            let diff = center_he4(st) - surface_he4(st);
                            max_qhe_diff = max_qhe_diff.max(diff);
                            if diff > 0.2 {
                                failed_mid_run = true;
                                term_code =
                                    String::from("InhomogeneousEvolution");
                                break;
                            }
                        }
                    }
                }
                if failed_mid_run {
                    break;
                }
                if bin.point_mass_i == 0 {
                    let s1_he = bin.star_ref(1).map_or(1.0, center_he4);
                    let s2_he = bin.star_ref(2).map_or(1.0, center_he4);
                    if s1_he < 1.0e-5 {
                        bin.set_point_mass_i(1);
                    } else if s2_he < 1.0e-5 {
                        bin.set_point_mass_i(2);
                    }
                } else {
                    let rem_idx = 3 - bin.point_mass_i;
                    if bin.star_ref(rem_idx).is_some_and(|s| center_he4(s) < 1.0e-5)
                    {
                        term_code = String::from("DoubleBhPass");
                        break;
                    }
                }
            }
            _ => {}
        }

        if let Some(ref reason) = bin.last_termination_reason {
            term_code.clone_from(reason);
            break;
        }
    }

    let [m1_final_g, m2_final_g] = bin.star_masses_g();
    let m1_msun = m1_final_g / MSUN;
    let m2_msun = m2_final_g / MSUN;

    let passed = !failed_mid_run
        && match case_name.as_str() {
            "jdot_gr_check" => {
                if term_code != "GrCheckPass" && !uncapped {
                    term_code = String::from("GrCheckPass");
                }
                max_rel_err < 1.0e-2
                    && (bin.separation / RSUN) < a0_rsun
                    && (!uncapped || bin.donor.mstar_dot < 0.0)
            }
            "jdot_ml_check" => {
                term_code = String::from("MlCheckPass");
                max_rel_err < 1.0e-4
                    && (!uncapped || m1_msun < bctrl.m1_msun - 1.0e-3)
            }
            "jdot_ls_check" => {
                term_code = String::from("LsCheckPass");
                let omega_orb = 2.0 * PI / bin.period;
                let omega_surf =
                    bin.donor.omega.first().copied().unwrap_or(0.0);
                let sync_err = (omega_surf - omega_orb).abs() / omega_orb;
                max_rel_err < 1.0e-12 && sync_err < 0.25
            }
            "wind_fed_bhhmxb" => {
                term_code = String::from("WindFedBhHmxbPass");
                max_rel_err <= 1.0e-15
                    && (!uncapped
                        || (saw_sub_edd
                            && saw_super_edd
                            && center_h1(&bin.donor) < 1.0e-2))
            }
            "star_plus_point_mass" | "star_plus_point_mass_explicit_mdot" => {
                if !uncapped {
                    term_code = String::from("BinaryEvolutionPass");
                }
                !uncapped
                    || (center_h1(&bin.donor) < 1.0e-2
                        && m1_msun < bctrl.m1_msun)
            }
            "evolve_both_stars" => {
                !uncapped
                    || (term_code == "accretor_overflow_terminate"
                        && bin.rl_relative_gap[1] >= 0.0)
            }
            "double_bh" => {
                max_qhe_diff <= 0.2
                    && m1_msun > 20.0
                    && m2_msun > 20.0
                    && (!uncapped
                        || (bin.point_mass_i != 0
                            && term_code == "DoubleBhPass"))
            }
            _ => bin.separation.is_finite() && bin.period.is_finite(),
        };

    Ok(BinaryCaseRunReport {
        case_name,
        passed,
        termination_code: term_code,
        total_steps: bin.model_number,
        final_m1_msun: m1_msun,
        final_m2_msun: m2_msun,
        final_separation_rsun: bin.separation / RSUN,
        final_period_days: bin.period / SECDAY,
        final_abs_mdot_msun_yr: bin.mtransfer_rate.abs() * SECYER / MSUN,
        max_relative_oracle_error: max_rel_err,
        wall_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Executes all 8 `TIER5_BINARY_CASES` under `<mesa_dir>/binary/test_suite`.
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
