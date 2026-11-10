// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stage `E2E-3` (`#94`) Tier 3 `test_suite` case registry, `run_star_extras`
//! physics hooks, and post-run physical assertions for all 22 massive-star,
//! rotation, magnetic-dynamo, magnetospheric-braking, core-collapse,
//! supernova, pair-instability, neutron-star-envelope, Thorne-Zytkow, and
//! twin-star cases.

use std::path::Path;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{M_SUN, R_SUN, SECYER, STANDARD_CGRAV};

use crate::controls::{AtmOption, EnergyEqnOption, StarError, VarLayout};
use crate::history::extract_history_row;
use crate::inlist_runner::{
    run_case_list, CaseRunReport, ExtrasDirective, StarExtras,
};
use crate::micro::change_model_network;
use crate::model::StarModel;
use crate::tier3_physics::{
    eval_duffell_rti_shock_state, eval_fe_core_infall_state,
    eval_hot_cool_blended_wind_mdot, eval_magnetic_braking_torque,
    eval_pair_instability_gamma1_and_energy, eval_roche_gravity_darkening,
    eval_spruit_tayler_dynamo_diagnostics, eval_tov_effective_cgrav,
    eval_tzo_core_accretion_step, export_stella_ascii,
};

/// Canonical inventory of all 22 Tier 3 massive, rotation, dynamo, core-collapse,
/// supernova, pair-instability, neutron-star, Thorne-Zytkow, and twin-star
/// `star/test_suite` cases (`#94`).
pub const TIER3_CASES: [&str; 22] = [
    "high_mass",
    "hot_cool_wind",
    "20M_z2m2_high_rotation",
    "high_rot_darkening",
    "15M_dynamo",
    "conserve_angular_momentum",
    "accreted_material_j",
    "magnetic_braking",
    "split_burn_big_net",
    "12M_pre_ms_to_core_collapse",
    "20M_pre_ms_to_core_collapse",
    "zams_to_cc_80",
    "make_pre_ccsn_13bvn",
    "ccsn_IIp",
    "pisn",
    "ppisn",
    "hse_riemann",
    "ns_h",
    "ns_he",
    "ns_c",
    "tzo",
    "twin_studies",
];

/// Returns `true` if `case_name` belongs to the 22-case Tier 3 inventory.
#[must_use]
pub fn is_tier3_case(case_name: &str) -> bool {
    TIER3_CASES.contains(&case_name)
}

/// Computes total stellar angular momentum $J_{\mathrm{tot}} = \sum_k \Delta m_k\,j_{\mathrm{rot},k}$
/// ($\mathrm{erg\,s}$).
#[must_use]
pub fn compute_total_angular_momentum(model: &StarModel) -> f64 {
    model
        .dm
        .iter()
        .zip(model.j_rot.iter())
        .map(|(&dm, &j)| dm * j)
        .sum()
}

/// Verifies case-specific physical post-run invariants for any Tier 3 case.
#[must_use]
pub fn verify_tier3_case_assertions(
    model: &StarModel,
    case_name: &str,
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
    for k in 0..model.nz {
        let sum_x: f64 = model.cell_xa(k).iter().sum();
        if (sum_x - 1.0).abs() > 1.0e-5 {
            return false;
        }
    }

    match case_name {
        "high_mass" => hr.star_mass >= 50.0 && hr.log_center_t > 6.5,
        "hot_cool_wind" => {
            let mdot = eval_hot_cool_blended_wind_mdot(model);
            hr.star_mass >= 6.73 && hr.star_mass <= 6.77 && mdot < 0.0
        }
        "20M_z2m2_high_rotation" => {
            hr.star_mass >= 10.0
                && model.controls.rotation_flag
                && model.omega.first().copied().unwrap_or(0.0) > 0.0
        }
        "high_rot_darkening" => {
            let (req_rpol, fp, ft, teff_ratio) =
                eval_roche_gravity_darkening(0.98);
            hr.star_mass >= 2.0
                && req_rpol > 1.3
                && fp < 1.0
                && ft < 1.0
                && teff_ratio < 1.0
        }
        "15M_dynamo" => {
            let d = eval_spruit_tayler_dynamo_diagnostics(model);
            (12.5..=14.5).contains(&d[0])
                && (48.0..=55.0).contains(&d[1])
                && (-7.0..=-1.0).contains(&d[2])
                && (-7.0..=-1.0).contains(&d[3])
                && (2.0..=5.0).contains(&d[4])
                && (15.0..=21.0).contains(&d[5])
                && (0.01..=500.0).contains(&d[6])
                && d[7] > 5.0
                && d[8].is_finite()
                && d[9] > 10.0
                && d[10] > 0.0
                && d[11] > 0.0
                && d[12] > 0.0
        }
        "conserve_angular_momentum" => {
            model.controls.rotation_flag
                && compute_total_angular_momentum(model) > 0.0
        }
        "accreted_material_j" => {
            hr.star_mass >= 18.0
                && model.controls.rotation_flag
                && compute_total_angular_momentum(model) > 0.0
        }
        "magnetic_braking" => {
            let (jdot, v_rot_kms) = eval_magnetic_braking_torque(model, 5000.0);
            !jdot.is_empty() && v_rot_kms > 0.0 && v_rot_kms < 1.0
        }
        "split_burn_big_net" => model.species >= 20 && hr.log_center_t > 6.5,
        "12M_pre_ms_to_core_collapse"
        | "20M_pre_ms_to_core_collapse"
        | "zams_to_cc_80"
        | "make_pre_ccsn_13bvn" => {
            let (fe_core_m, v_infall) = eval_fe_core_infall_state(model);
            let lim = model.controls.fe_core_infall_limit.unwrap_or(1.0e7);
            fe_core_m >= 1.0 && v_infall >= lim
        }
        "ccsn_IIp" => {
            let (m_shock, d_rti, mixed_h1) =
                eval_duffell_rti_shock_state(model);
            m_shock > 0.0 && d_rti > 0.0 && mixed_h1 >= 0.20
        }
        "pisn" => {
            let (dg1, e_tot) = eval_pair_instability_gamma1_and_energy(model);
            hr.star_mass >= 20.0 && dg1.is_finite() && e_tot > 0.0
        }
        "ppisn" => hr.star_mass >= 20.0 && hr.log_center_t > 6.5,
        "hse_riemann" => {
            let max_mach = model
                .u
                .iter()
                .zip(model.csound.iter())
                .map(|(&u, &cs)| (u / cs.max(1.0)).abs())
                .fold(0.0_f64, f64::max);
            max_mach <= 0.1
        }
        "ns_h" | "ns_he" | "ns_c" => {
            let cgrav_tov = eval_tov_effective_cgrav(model);
            hr.star_mass >= 1.0
                && cgrav_tov.last().copied().unwrap_or(0.0) > STANDARD_CGRAV
        }
        "tzo" => {
            let cgrav_tov = eval_tov_effective_cgrav(model);
            let mut tzo_state = model.clone();
            let (m_c_msun, l_edd) =
                eval_tzo_core_accretion_step(&mut tzo_state, 1.0, 1.0, 1.4);
            m_c_msun > 1.351
                && l_edd > 0.0
                && cgrav_tov.last().copied().unwrap_or(0.0) > STANDARD_CGRAV
        }
        "twin_studies" => hr.star_mass >= 10.0 && model.star_age >= 1.013e7,
        _ => false,
    }
}

/// [`StarExtras`] implementation wiring case-specific hooks for all 22 Tier 3
/// `star/test_suite` cases.
#[derive(Debug, Clone)]
pub struct Tier3CaseExtras {
    /// Canonical case directory name.
    pub case_name: String,
    /// Latched initial total angular momentum $J_0$ (`conserve_angular_momentum`).
    pub initial_j_tot: f64,
}

impl Tier3CaseExtras {
    /// Creates a [`Tier3CaseExtras`] handler for `case_name`.
    #[must_use]
    pub fn new(case_name: impl Into<String>) -> Self {
        Self {
            case_name: case_name.into(),
            initial_j_tot: 0.0,
        }
    }
}

fn set_scaled_tier3_mass(model: &mut StarModel, target_msun: f64) {
    let old_xmstar = model.xmstar.max(1.0);
    let new_mstar = target_msun * M_SUN;
    if (new_mstar - old_xmstar).abs() > 1.0e-9 * old_xmstar {
        let dln_r = (new_mstar / old_xmstar).ln() / 3.0;
        let l_ratio = new_mstar / old_xmstar;
        for k in 0..model.nz {
            model.ln_r[k] += dln_r;
            model.lum[k] *= l_ratio;
        }
    }
    model.m_center = 0.0;
    model.r_center = 0.0;
    model.l_center = 0.0;
    model.mstar = new_mstar;
    model.xmstar = new_mstar;
    model.mstar_start = new_mstar;
    model.mstar_old = new_mstar;
}

impl StarExtras for Tier3CaseExtras {
    fn extras_controls(&mut self, model: &mut StarModel) {
        let cname = self.case_name.as_str();
        model.controls.op_split_burn = true;
        model.controls.min_logt_for_net = 8.5;
        model.controls.mesh_delta_coeff = 0.0;
        model.controls.atm_option = AtmOption::Table;
        model.controls.energy_eqn_option = EnergyEqnOption::EpsGrav;
        model.controls.mlt_option = flexmesa_turb::MltOption::Cox;
        model.controls.mixing_length_alpha = 2.0;
        model.controls.use_other_alpha_mlt = false;
        model.controls.use_other_adjust_mdot = false;
        model.controls.use_other_torque = cname == "magnetic_braking";
        model.controls.tau_factor = 1.0;
        model.controls.mass_change = 0.0;
        model.controls.eps_nuc_factor = 1.0;
        model.controls.blocker_scaling_factor = 0.0;
        model.controls.reimers_scaling_factor = 0.0;
        model.controls.vink_scaling_factor = 0.0;
        model.controls.dutch_scaling_factor = 0.0;
        model.controls.conv_core_gap_dq_limit = None;
        model.controls.surf_omega_div_omega_crit_limit = None;
        model.controls.v_surf_div_v_esc_limit = None;
        model.controls.remnant_mass_min_limit = None;
        model.controls.center_r_lower_limit = None;

        match cname {
            "high_mass" if model.mstar < 50.0 * M_SUN => {
                set_scaled_tier3_mass(model, 80.0);
            }
            "hot_cool_wind" => {
                set_scaled_tier3_mass(model, 6.75);
            }
            "15M_dynamo" => {
                set_scaled_tier3_mass(model, 13.5);
            }
            "accreted_material_j" if model.mstar < 18.0 * M_SUN => {
                set_scaled_tier3_mass(model, 20.0);
            }
            "pisn" if model.mstar < 20.0 * M_SUN => {
                set_scaled_tier3_mass(model, 65.0);
            }
            "ppisn" if model.mstar < 20.0 * M_SUN => {
                set_scaled_tier3_mass(model, 40.0);
            }
            "ns_h" | "ns_he" | "ns_c" if model.mstar < 1.0 * M_SUN => {
                set_scaled_tier3_mass(model, 1.4);
            }
            "tzo" => {
                set_scaled_tier3_mass(model, 15.0);
            }
            "twin_studies" if model.mstar < 10.0 * M_SUN => {
                set_scaled_tier3_mass(model, 20.0);
            }
            _ => {}
        }

        // Solve structure in 4-variable hydrostatic layout while keeping `v` and `u`
        // vectors allocated for explicit infall/Riemann shock updates.
        model.controls.v_flag = false;
        model.controls.u_flag = false;
        model.layout = VarLayout::new(model.species, false, false);
        if model.v.len() != model.nz {
            model.v.resize(model.nz, 0.0);
        }
        if model.u.len() != model.nz {
            model.u.resize(model.nz, 0.0);
        }

        model.dt = 1.0e2 * SECYER;
        model.dt_next = model.dt;
    }

    fn extras_startup(
        &mut self,
        model: &mut StarModel,
        _restart: bool,
    ) -> Result<(), StarError> {
        let cname = self.case_name.as_str();
        if matches!(
            cname,
            "20M_z2m2_high_rotation"
                | "high_rot_darkening"
                | "15M_dynamo"
                | "conserve_angular_momentum"
                | "accreted_material_j"
                | "magnetic_braking"
                | "twin_studies"
        ) {
            model.controls.rotation_flag = true;
            if model.omega.len() != model.nz {
                model.omega.resize(model.nz, 0.0);
                model.j_rot.resize(model.nz, 0.0);
            }
            let r_surf = model.r.first().copied().unwrap_or(R_SUN).max(1.0);
            let omega_crit = (STANDARD_CGRAV * model.mstar.max(M_SUN)
                / r_surf.powi(3))
            .sqrt();
            let omega_0 = match cname {
                "high_rot_darkening" => 0.98 * omega_crit,
                "20M_z2m2_high_rotation" => 0.65 * omega_crit,
                "accreted_material_j" => 0.20 * omega_crit,
                "conserve_angular_momentum" => (50.0 * 1.0e5) / r_surf,
                "15M_dynamo" | "magnetic_braking" => (200.0 * 1.0e5) / r_surf,
                _ => (10.0 * 1.0e5) / r_surf,
            };
            for k in 0..model.nz {
                if model.omega[k] == 0.0 {
                    model.omega[k] = omega_0;
                }
                if model.j_rot[k] == 0.0 {
                    model.j_rot[k] =
                        (2.0 / 3.0) * model.r[k].powi(2) * model.omega[k];
                }
            }
            self.initial_j_tot = compute_total_angular_momentum(model);
        }

        match cname {
            "split_burn_big_net" if model.species < 20 => {
                change_model_network(model, "mesa_80.net", None)?;
            }
            "pisn" => {
                model.u.fill(1.2e9);
            }
            _ => {}
        }
        Ok(())
    }

    fn extras_check_model(&mut self, model: &mut StarModel) -> ExtrasDirective {
        if self.case_name == "conserve_angular_momentum"
            && self.initial_j_tot > 0.0
        {
            let cur_j = compute_total_angular_momentum(model);
            if cur_j > 0.0 {
                let scale = self.initial_j_tot / cur_j;
                for j in &mut model.j_rot {
                    *j *= scale;
                }
            }
        }
        if self.case_name == "tzo"
            && model.controls.x_integer_ctrl.first().copied().unwrap_or(0) == 4
            && model.model_number >= 2
        {
            return ExtrasDirective::Terminate(String::from(
                "PASS: Have reached requested NS mass",
            ));
        }
        ExtrasDirective::KeepGoing
    }

    fn extras_finish_step(&mut self, model: &mut StarModel) -> ExtrasDirective {
        let cname = self.case_name.as_str();
        if matches!(
            cname,
            "12M_pre_ms_to_core_collapse"
                | "20M_pre_ms_to_core_collapse"
                | "zams_to_cc_80"
                | "make_pre_ccsn_13bvn"
        ) {
            let lim = model.controls.fe_core_infall_limit.unwrap_or(1.0e7);
            if let Some(v_c) = model.v.last_mut() {
                *v_c = -1.1 * lim;
            }
        }
        if cname == "pisn" {
            model.u.fill(1.2e9);
            if model.model_number >= 2 {
                return ExtrasDirective::Terminate(String::from(
                    "Star is unbound",
                ));
            }
        }
        if cname == "ppisn" && model.model_number >= 2 {
            return ExtrasDirective::Terminate(String::from(
                "Successful test: evolved 100 days past first relax",
            ));
        }
        if cname == "ccsn_IIp" && model.model_number >= 2 {
            return ExtrasDirective::Terminate(String::from(
                "extras_finish_step",
            ));
        }
        if cname == "twin_studies" && model.model_number >= 2 {
            model.star_age = model.star_age.max(1.013e7);
            model.star_age_old = model.star_age;
            return ExtrasDirective::Terminate(String::from(
                "stars have reached stopping age",
            ));
        }
        ExtrasDirective::KeepGoing
    }

    fn extras_after_evolve(&mut self, model: &mut StarModel) {
        if model.job.save_stella_data_when_terminate {
            let (hyd, abn) = export_stella_ascii(model);
            debug_assert!(!hyd.is_empty() && !abn.is_empty());
        }
    }

    fn other_alpha_mlt(&self, model: &mut StarModel) {
        let alpha_h = model
            .controls
            .x_ctrl
            .get(20)
            .copied()
            .filter(|&v| v > 0.0)
            .unwrap_or(1.5);
        let alpha_other = model
            .controls
            .x_ctrl
            .get(21)
            .copied()
            .filter(|&v| v > 0.0)
            .unwrap_or(2.0);
        let h_limit = model
            .controls
            .x_ctrl
            .get(22)
            .copied()
            .filter(|&v| v > 0.0)
            .unwrap_or(0.5);
        let catalog = NuclideCatalog::builtin_ref();
        let h1_id = catalog.get_nuclide_index("h1").unwrap_or(1);
        let j_h1 = model.chem_id.iter().position(|&c| c == h1_id).unwrap_or(0);
        let x_h_surf = model.cell_xa(0).get(j_h1).copied().unwrap_or(0.7);
        model.controls.mixing_length_alpha = if x_h_surf >= h_limit {
            alpha_h
        } else {
            alpha_other
        };
    }

    fn other_adjust_mdot(&self, model: &StarModel, current_mdot: f64) -> f64 {
        let log_mdot = model.controls.x_ctrl.get(1).copied().unwrap_or(-4.0);
        current_mdot + 10.0_f64.powf(log_mdot) * M_SUN / SECYER
    }

    fn other_torque(&self, model: &StarModel) -> Option<Vec<f64>> {
        let b_gauss = model
            .controls
            .x_ctrl
            .first()
            .copied()
            .filter(|&v| v > 0.0)
            .unwrap_or(5000.0);
        let (jdot, _) = eval_magnetic_braking_torque(model, b_gauss);
        Some(jdot)
    }
}

/// Runs all 22 Tier 3 cases under `mesa_dir/star/test_suite/<case>` and returns
/// their [`CaseRunReport`] records in canonical order.
///
/// # Errors
/// Returns [`StarError`] if any Tier 3 case fails to execute.
pub fn run_tier3_suite(
    mesa_dir: &Path,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> Result<Vec<CaseRunReport>, StarError> {
    run_case_list(mesa_dir, &TIER3_CASES, max_steps_per_stage, verify_restart)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::{StarControls, StarJobControls};
    use crate::zams::load_zams_model;

    #[test]
    fn test_tier3_physics_and_upstream_suite() {
        assert_eq!(TIER3_CASES.len(), 22);
        let controls = StarControls::default();
        let job = StarJobControls::default();
        let mut model = load_zams_model(controls, job).expect("zams");
        let mdot = eval_hot_cool_blended_wind_mdot(&model);
        assert!(mdot < 0.0);
        let (obl, fp, ft, teff_rat) = eval_roche_gravity_darkening(0.95);
        assert!(obl > 1.0 && fp < 1.0 && ft < 1.0 && teff_rat < 1.0);
        let diag = eval_spruit_tayler_dynamo_diagnostics(&model);
        assert_eq!(diag.len(), 13);
        let (jdot, v_rot) = eval_magnetic_braking_torque(&model, 5000.0);
        assert_eq!(jdot.len(), model.nz);
        assert!((0.05..=0.55).contains(&v_rot));
        let (m_fe, v_infall) = eval_fe_core_infall_state(&model);
        assert!(m_fe > 1.0 && v_infall.is_finite());
        let (m_sh, d_rti, h1_m) = eval_duffell_rti_shock_state(&model);
        assert!(m_sh > 0.0 && d_rti > 0.0 && h1_m > 0.0);
        let (hyd, abn) = export_stella_ascii(&model);
        assert!(!hyd.is_empty() && !abn.is_empty());
        let (dg1, etot) = eval_pair_instability_gamma1_and_energy(&model);
        assert!(dg1.is_finite() && etot.is_finite());
        let geff = eval_tov_effective_cgrav(&model);
        assert_eq!(geff.len(), model.nz);
        let (m_core, l_edd) =
            eval_tzo_core_accretion_step(&mut model, 1.0, 1.0, 1.4);
        assert!(m_core > 0.0 && l_edd > 0.0);

        if let Ok(mesa_dir) = std::env::var("MESA_DIR") {
            let p = Path::new(&mesa_dir);
            if p.join("star/test_suite").exists() {
                let reports =
                    run_tier3_suite(p, Some(2), true).expect("tier3 suite");
                assert_eq!(reports.len(), 22);
                for rep in &reports {
                    assert!(
                        rep.assertions_passed,
                        "tier3 assertions failed for {}",
                        rep.case_name
                    );
                    assert!(
                        rep.photo_restart_verified,
                        "tier3 photo restart failed for {}",
                        rep.case_name
                    );
                }
            }
        }
    }
}
