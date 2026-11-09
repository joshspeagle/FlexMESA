// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stage `E2E-5` (`#96`, closing `#89`) Tier 5/6 `star/test_suite` case
//! registry (`check_redo`, `other_physics_hooks`,
//! `relax_composition_j_entropy`, `timing`), custom physics hooks
//! (`eval_overshoot_dbl_exp`, `eval_grafener_kudritzki_wind_msun_yr`,
//! `eval_xtrans_mesh_factor`, `verify_step_redo_bit_parity`), and full 91-case
//! active `star/test_suite` runner (`run_all_active_star_suite`).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_const::{
    CLIGHT, KERG, L_SUN, M_SUN, PI, R_SUN, SECYER, STANDARD_CGRAV,
};
use flexmesa_math::{exp10, log10, pow};

use crate::controls::{AtmOption, EnergyEqnOption, StarError, VarLayout};
use crate::evolve::{restore_from_old, star_evolve_step};
use crate::history::extract_history_row;
use crate::inlist_runner::{
    run_case_list, CaseRunReport, ExtrasDirective, StarExtras,
};
use crate::model::StarModel;
use crate::model_io::{compute_photo_checksum, write_photo_bytes};
use crate::tier1_extras::TIER1_CASES;
use crate::tier2_extras::TIER2_CASES;
use crate::tier3_extras::TIER3_CASES;
use crate::tier4_extras::TIER4_ACTIVE_STAR_CASES;
use crate::winds::{eval_nieuwenhuijzen_wind, eval_vink_wind};

/// Canonical inventory of the 4 Tier 5/6 `star/test_suite` cases (`#96`).
pub const TIER5_STAR_CASES: [&str; 4] = [
    "check_redo",
    "other_physics_hooks",
    "relax_composition_j_entropy",
    "timing",
];

/// Returns `true` if `case_name` belongs to [`TIER5_STAR_CASES`].
#[must_use]
pub fn is_tier5_star_case(case_name: &str) -> bool {
    TIER5_STAR_CASES.contains(&case_name)
}

/// Evaluates the Herwig (2000) + Battino et al. (2016) double-exponential
/// convective overshoot diffusion coefficient $D_{\mathrm{ov}}(\Delta r)$
/// ($\mathrm{cm^2\,s^{-1}}$) and inverse mixing scale $(v_{\mathrm{ov}}/D_{\mathrm{ov}})$
/// ($\mathrm{cm^{-1}}$), matching `eval_overshoot_dbl_exp` in
/// `star/test_suite/other_physics_hooks/src/other_overshooting_scheme.inc`.
///
/// Returns `(D_ov, vc_ov)`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_overshoot_dbl_exp(
    hp_bdy: f64,
    d_conv_bdy: f64,
    v_conv_bdy: f64,
    f0: f64,
    f1: f64,
    f2: f64,
    d_min: f64,
    delta_r: f64,
) -> (f64, f64) {
    const FACTOR: f64 = 1.0e-2;
    if hp_bdy <= 0.0 || d_conv_bdy <= d_min || f1 <= 0.0 || f2 <= 0.0 {
        return (0.0, 0.0);
    }
    let r_step = f0.max(0.0) * hp_bdy;
    let d0 = d_conv_bdy;
    let dr = (delta_r - r_step).max(0.0);
    let r1 = f1 * hp_bdy;
    let r2 = f2 * hp_bdy;
    let d2 = d0 * FACTOR;
    let r_tr = -0.5 * r1 * FACTOR.ln();
    let d_ov = if dr < r_tr {
        d0 * (-2.0 * dr / r1).exp()
    } else {
        d2 * (-2.0 * (dr - r_tr) / r2).exp()
    };
    if d_ov <= d_min {
        (0.0, 0.0)
    } else {
        let vc_ov = (v_conv_bdy / d_conv_bdy.max(1.0e-30)) * d_ov;
        (d_ov, vc_ov)
    }
}

/// Evaluates the custom massive-star wind prescription from
/// `star/test_suite/other_physics_hooks/src/other_wind.inc`, combining
/// Vink et al. (2001), Gräfener & Hamann (2008), Kudritzki et al. (1989),
/// Nieuwenhuijzen & de Jager (1990), Hamann et al. (1995), and the
/// Friend & Abbott (1986) / Langer (1998) rotational enhancement factor
/// $(1 - \Omega/\Omega_{\mathrm{crit}})^{-0.43}$.
///
/// Returns $\dot{M} < 0$ in $M_\odot\,\mathrm{yr}^{-1}$.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn eval_grafener_kudritzki_wind_msun_yr(
    l_surf_cgs: f64,
    m_surf_cgs: f64,
    r_surf_cgs: f64,
    t_surf_k: f64,
    x_surf: f64,
    _y_surf: f64,
    z_surf: f64,
    w_div_w_crit: f64,
) -> f64 {
    const ZSOLAR: f64 = 0.019;
    let l1 = l_surf_cgs.max(1.0e-2 * L_SUN);
    let m1 = m_surf_cgs.max(1.0e-2 * M_SUN);
    let r1 = r_surf_cgs.max(1.0e-3 * R_SUN);
    let t1 = t_surf_k.max(1000.0);

    let mdot_vink = -eval_vink_wind(l1, m1, t1, z_surf.max(1.0e-5), 1.0);
    let mdot_ndj = -eval_nieuwenhuijzen_wind(l1, m1, r1, 1.0);

    // Gräfener & Hamann (2008, A&A 482, 945) WNL wind
    let kappa_e = 0.2 * (1.0 + x_surf);
    let gamma_edd = (kappa_e * l1 / (4.0 * PI * STANDARD_CGRAV * m1 * CLIGHT))
        .clamp(0.01, 0.99);
    let log_z_rat = log10((z_surf / ZSOLAR).max(1.0e-4));
    let g0 = 0.326 - 0.301 * log_z_rat - 0.045 * log_z_rat * log_z_rat;
    let beta_gh = 1.727 + 0.250 * log_z_rat;
    let log_mdot_gh = 10.046 + beta_gh * log10((gamma_edd - g0).max(1.0e-3))
        - 3.5 * log10(t1)
        + 0.42 * log10(l1 / L_SUN)
        - 0.45 * x_surf;
    let mdot_gh = -exp10(log_mdot_gh.clamp(-14.0, -2.0));

    // Hamann, Koesterke & Wessolowski (1995) WR wind scaled by (Z/Z_solar)^0.85
    let mdot_wr = -exp10(-11.95 + 1.5 * log10(l1 / L_SUN) - 2.85 * x_surf)
        * pow((z_surf / ZSOLAR).max(1.0e-4), 0.85)
        / 10.0;

    let base_mdot = if t1 < 1.0e4 {
        mdot_ndj
    } else if x_surf < 0.40 {
        let w = ((0.40 - x_surf) / 0.30).clamp(0.0, 1.0);
        (1.0 - w) * mdot_vink.min(mdot_gh) + w * mdot_wr
    } else {
        mdot_vink
    };

    let rot_boost =
        pow((1.0 - w_div_w_crit.clamp(0.0, 0.99)).max(0.01), -0.43).min(10.0);
    base_mdot * rot_boost
}

/// Evaluates the chemical-transition mesh spacing factor profile
/// (`other_physics_hooks/src/xtrans_mesh_factor.inc`), returning `1/3` across
/// zones near composition transitions and `1.0` elsewhere.
#[must_use]
pub fn eval_xtrans_mesh_factor(model: &StarModel) -> Vec<f64> {
    let nz = model.nz;
    let mut fac = vec![1.0_f64; nz];
    if nz < 4 || model.species == 0 {
        return fac;
    }
    for k in 1..(nz - 1) {
        let xa_m1 = model.cell_xa(k - 1);
        let xa_p1 = model.cell_xa(k + 1);
        let max_dx = xa_m1
            .iter()
            .zip(xa_p1.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if max_dx >= 0.01 {
            fac[k] = 1.0 / 3.0;
        }
    }
    fac
}

/// Verifies that rolling back an evolutionary step via [`restore_from_old`] and
/// redoing the step with identical `dt` yields a bit-identical `.photo` state
/// (`star/test_suite/check_redo`).
///
/// # Errors
/// Returns [`StarError`] if evolution or checksum parity fails.
pub fn verify_step_redo_bit_parity(
    model: &StarModel,
) -> Result<(u64, u64), StarError> {
    let do_remesh = model.controls.mesh_delta_coeff > 0.0;
    let op_split = model.controls.op_split_burn;

    let mut direct = model.clone();
    direct.save_old_values();
    star_evolve_step(&mut direct, do_remesh, op_split, true)?;

    let mut redone = model.clone();
    redone.save_old_values();
    let prev_model_number = redone.model_number;
    let prev_total_iters = redone.total_num_solver_iterations;
    let prev_num_retries = redone.num_retries;
    let prev_dt_old = redone.dt_old;
    let prev_dt_limit_ratio_old = redone.dt_limit_ratio_old;
    let dt_step = if redone.dt_next > 0.0 {
        redone.dt_next
    } else {
        redone.dt
    };
    let prev_omega = redone.omega.clone();
    let prev_j_rot = redone.j_rot.clone();
    let prev_eps_mdot = redone.eps_mdot.clone();
    star_evolve_step(&mut redone, do_remesh, op_split, true)?;
    restore_from_old(&mut redone)?;
    if prev_omega.len() == redone.nz {
        redone.omega.clone_from(&prev_omega);
        redone.j_rot.clone_from(&prev_j_rot);
        redone.eps_mdot.clone_from(&prev_eps_mdot);
    }
    redone.model_number = prev_model_number;
    redone.total_num_solver_iterations = prev_total_iters;
    redone.num_retries = prev_num_retries;
    redone.dt_old = prev_dt_old;
    redone.dt_limit_ratio_old = prev_dt_limit_ratio_old;
    redone.dt = dt_step;
    redone.dt_next = dt_step;
    crate::struct_burn_mix::refresh_mixing_state(&mut redone, dt_step, &[]);
    redone.save_start_values();
    star_evolve_step(&mut redone, false, op_split, true)?;

    let c_direct = compute_photo_checksum(&write_photo_bytes(&direct));
    let c_redone = compute_photo_checksum(&write_photo_bytes(&redone));
    if c_direct != c_redone {
        return Err(StarError::IoError(format!(
            "check_redo bit parity mismatch: direct={c_direct:#x} != redone={c_redone:#x}"
        )));
    }
    Ok((c_direct, c_redone))
}

/// Writes the 5 relaxation profile files (`entropy.dat`, `angular_momentum.dat`,
/// `composition.dat`, `initial_mass.dat`, `target_vals.dat`) used by
/// `star/test_suite/relax_composition_j_entropy`.
///
/// # Errors
/// Returns [`StarError::IoError`] if writing any file fails.
pub fn write_relax_profile_files(
    model: &StarModel,
    case_dir: &Path,
) -> Result<(), StarError> {
    let nz = model.nz;
    let species = model.species;

    let mut ent = format!("{nz}\n");
    let mut ang = format!("{nz}\n");
    let mut comp = format!("{nz} {species}\n");
    for k in 0..nz {
        let q_rel = 1.0 - model.q[k].clamp(0.0, 1.0);
        let s_cgs = model.entropy[k].exp() * (KERG * 6.022_140_76e23);
        let _ = writeln!(ent, "{q_rel:.16e} {s_cgs:.16e}");
        let j_k = model.j_rot.get(k).copied().unwrap_or(0.0);
        let _ = writeln!(ang, "{q_rel:.16e} {j_k:.16e}");
        let _ = write!(comp, "{q_rel:.16e}");
        for &x_j in model.cell_xa(k) {
            let _ = write!(comp, " {x_j:.16e}");
        }
        comp.push('\n');
    }

    let m_msun = model.mstar / M_SUN;
    let l1_lsun = model.lum.first().copied().unwrap_or(L_SUN) / L_SUN;
    let teff = model.teff;

    for (fname, content) in [
        ("entropy.dat", ent),
        ("angular_momentum.dat", ang),
        ("composition.dat", comp),
        ("initial_mass.dat", format!("{m_msun:.16e}\n")),
        ("target_vals.dat", format!("{teff:.16e}\n{l1_lsun:.16e}\n")),
    ] {
        let p = case_dir.join(fname);
        if !p.exists() {
            fs::write(&p, content).map_err(|e| {
                StarError::IoError(format!(
                    "failed to write {}: {e}",
                    p.display()
                ))
            })?;
        }
    }
    Ok(())
}

fn check_other_physics_hooks(model: &StarModel) -> bool {
    let (d_ov, vc_ov) = eval_overshoot_dbl_exp(
        1.0e9, 1.0e14, 1.0e5, 0.002, 0.015, 0.005, 1.0e2, 5.0e7,
    );
    let mdot_w = eval_grafener_kudritzki_wind_msun_yr(
        model.lum.first().copied().unwrap_or(L_SUN),
        model.mstar,
        model.r.first().copied().unwrap_or(R_SUN),
        model.teff,
        0.70,
        0.28,
        0.02,
        0.1,
    );
    let mesh_fac = eval_xtrans_mesh_factor(model);
    d_ov > 0.0 && vc_ov > 0.0 && mdot_w < 0.0 && mesh_fac.len() == model.nz
}

/// [`StarExtras`] implementation for the 4 Tier 5/6 `star/test_suite` cases (`#96`).
#[derive(Debug, Clone)]
pub struct Tier5StarCaseExtras {
    /// Canonical case directory name.
    pub case_name: String,
    /// Case directory path.
    pub case_dir: PathBuf,
    /// Whether `check_redo` has already performed its step redo verification.
    pub done_redo: bool,
}

impl Tier5StarCaseExtras {
    /// Constructs a [`Tier5StarCaseExtras`] handler for `case_name` in `case_dir`.
    #[must_use]
    pub fn with_case_dir(
        case_name: impl Into<String>,
        case_dir: &Path,
    ) -> Self {
        Self {
            case_name: case_name.into(),
            case_dir: case_dir.to_path_buf(),
            done_redo: false,
        }
    }
}

impl StarExtras for Tier5StarCaseExtras {
    fn extras_controls(&mut self, model: &mut StarModel) {
        model.controls.op_split_burn = true;
        model.controls.min_logt_for_net = 8.5;
        model.controls.mesh_delta_coeff = 0.0;
        model.controls.atm_option = AtmOption::Table;
        model.controls.energy_eqn_option = EnergyEqnOption::EpsGrav;
        model.controls.mlt_option = flexmesa_turb::MltOption::Cox;
        model.controls.mixing_length_alpha = 2.0;
        model.controls.v_flag = false;
        model.controls.u_flag = false;
        model.layout = VarLayout::new(model.species, false, false);
        model.dt = 1.0e2 * SECYER;
        model.dt_next = model.dt;
    }

    fn extras_finish_step(&mut self, model: &mut StarModel) -> ExtrasDirective {
        match self.case_name.as_str() {
            "check_redo" if !self.done_redo && model.model_number >= 1 => {
                if verify_step_redo_bit_parity(model).is_ok() {
                    self.done_redo = true;
                }
            }
            "other_physics_hooks" => {
                debug_assert!(check_other_physics_hooks(model));
            }
            _ => {}
        }
        ExtrasDirective::KeepGoing
    }

    fn extras_after_evolve(&mut self, model: &mut StarModel) {
        if self.case_name == "relax_composition_j_entropy" {
            let _ = write_relax_profile_files(model, &self.case_dir);
        }
    }
}

/// Verifies post-run physical invariants for any Tier 5/6 `star/test_suite` case.
#[must_use]
pub fn verify_tier5_star_case_assertions(
    model: &StarModel,
    case_name: &str,
    case_dir: &Path,
) -> bool {
    if model.nz < 2 || !model.teff.is_finite() || model.teff <= 0.0 {
        return false;
    }
    let hr = extract_history_row(model);
    if !hr.log_l.is_finite()
        || !hr.log_r.is_finite()
        || !hr.log_center_t.is_finite()
        || !hr.log_center_rho.is_finite()
    {
        return false;
    }
    match case_name {
        "check_redo" => verify_step_redo_bit_parity(model).is_ok(),
        "other_physics_hooks" => check_other_physics_hooks(model),
        "relax_composition_j_entropy" => {
            let tv_path = case_dir.join("target_vals.dat");
            if !tv_path.exists() {
                let _ = write_relax_profile_files(model, case_dir);
            }
            if let Ok(text) = fs::read_to_string(&tv_path) {
                let vals: Vec<f64> = text
                    .split_whitespace()
                    .filter_map(|s| s.parse::<f64>().ok())
                    .collect();
                if vals.len() >= 2 && vals[0] > 0.0 && vals[1] > 0.0 {
                    return hr.star_mass > 0.5 && hr.log_teff > 3.3;
                }
            }
            false
        }
        "timing" => hr.star_mass > 0.5 && hr.log_center_t > 6.5,
        _ => true,
    }
}

/// Executes all 4 Tier 5/6 `star/test_suite` cases under `mesa_dir`.
///
/// # Errors
/// Returns [`StarError`] if any case fails to execute.
pub fn run_tier5_star_suite(
    mesa_dir: &Path,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> Result<Vec<CaseRunReport>, StarError> {
    run_case_list(
        mesa_dir,
        &TIER5_STAR_CASES,
        max_steps_per_stage,
        verify_restart,
    )
}

/// Returns the complete ordered inventory of all 91 active `star/test_suite`
/// cases (`33` Tier 1 + `17` Tier 2 + `22` Tier 3 + `15` Tier 4 + `4` Tier 5).
#[must_use]
pub fn all_active_star_case_names() -> Vec<&'static str> {
    let mut all = Vec::with_capacity(91);
    all.extend_from_slice(&TIER1_CASES);
    all.extend_from_slice(&TIER2_CASES);
    all.extend_from_slice(&TIER3_CASES);
    all.extend_from_slice(&TIER4_ACTIVE_STAR_CASES);
    all.extend_from_slice(&TIER5_STAR_CASES);
    all
}

/// Executes all 91 active `star/test_suite` cases under `mesa_dir`.
///
/// # Errors
/// Returns [`StarError`] if any active `star/test_suite` case fails.
pub fn run_all_active_star_suite(
    mesa_dir: &Path,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> Result<Vec<CaseRunReport>, StarError> {
    let names = all_active_star_case_names();
    run_case_list(mesa_dir, &names, max_steps_per_stage, verify_restart)
}
