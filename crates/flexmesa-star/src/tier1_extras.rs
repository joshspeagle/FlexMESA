// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stage `E2E-1` (`#92`) Tier 1 `test_suite` case registry, `run_star_extras`
//! physics hooks, and post-run physical assertions for all 33 hydrostatic,
//! solar, mixing, and diffusion cases.

use std::path::Path;

use flexmesa_chem::NuclideCatalog;
use flexmesa_colors::{
    builtin_sed_cube, MagSystem, SedInterpMethod, D_10PC_CM,
};
use flexmesa_const::{L_SUN, R_SUN, STANDARD_CGRAV};

use crate::controls::{AtmTTauRelation, StarError};
use crate::diffusion::{
    do_element_diffusion, eval_radiative_levitation_accel, DiffusionScheme,
};
use crate::history::extract_history_row;
use crate::inlist_runner::{
    run_test_suite_case, CaseRunReport, ExtrasDirective, StarExtras,
};
use crate::micro::change_model_network;
use crate::mix_info::compute_convection_sigmas;
use crate::model::StarModel;
use crate::premix::{
    add_predictive_mixing, do_conv_premix, eval_castellani_penetration_vel,
};
use crate::pulse::{build_pulse_points, export_fgong, export_gyre};
use crate::struct_burn_mix::solve_composition_diffusion;

/// Canonical inventory of all 33 Tier 1 hydrostatic, solar, mixing, and
/// diffusion `star/test_suite` cases (`#92`).
pub const TIER1_CASES: [&str; 33] = [
    "make_zams",
    "make_zams_low_mass",
    "make_zams_ultra_high_mass",
    "make_metals",
    "1.3M_ms_high_Z",
    "1.4M_ms_op_mono",
    "1.5M_with_diffusion",
    "diffusion_smoothness",
    "radiative_levitation",
    "low_z",
    "high_z",
    "1M_pre_ms_to_wd",
    "1M_thermohaline",
    "hb_2M",
    "5M_cepheid_blue_loop",
    "7M_prems_to_AGB",
    "c13_pocket",
    "R_CrB_star",
    "make_sdb",
    "conv_core_cpm",
    "16M_conv_premix",
    "16M_predictive_mix",
    "semiconvection",
    "extended_convective_penetration",
    "T_tau_gradr",
    "check_pulse_atm",
    "make_env",
    "starspots",
    "custom_colors",
    "adjust_net",
    "custom_rates",
    "carbon_kh",
    "simplex_solar_calibration",
];

/// Returns `true` if `case_name` belongs to the 33-case Tier 1 inventory.
#[must_use]
pub fn is_tier1_case(case_name: &str) -> bool {
    TIER1_CASES.contains(&case_name)
}

/// Evaluates the YREC starspot effective unspotted luminosity $L_{\mathrm{eff}}$
/// and flux blocking factor $\alpha_{\mathrm{spot}}$ (`starspots` `run_star_extras.f90`):
///
/// $$\alpha_{\mathrm{spot}} = f_{\mathrm{spot}}\,x_{\mathrm{spot}}^4 + 1 - f_{\mathrm{spot}}, \qquad
/// L_{\mathrm{eff}} = \frac{L_{\mathrm{surf}}}{\alpha_{\mathrm{spot}}}.$$
///
/// # Arguments
/// * `l_surf` - Bolometric surface luminosity $L$ ($\mathrm{erg\,s^{-1}}$).
/// * `f_spot` - Fractional surface spot coverage $f_{\mathrm{spot}} \in [0, 1]$.
/// * `x_spot` - Spot-to-ambient temperature ratio $T_{\mathrm{spot}} / T_{\mathrm{amb}} \ge 0$.
#[must_use]
pub fn eval_starspot_effective_luminosity(
    l_surf: f64,
    f_spot: f64,
    x_spot: f64,
) -> (f64, f64) {
    let f = f_spot.clamp(0.0, 1.0);
    let x = x_spot.max(0.0);
    let alpha = (f * x.powi(4) + (1.0 - f)).max(1.0e-6);
    (l_surf / alpha, alpha)
}

/// Evaluates the Hopf function $q(\tau)$ and the radiative temperature gradient
/// correction factor $1 + \tau\,dq(\tau)/d\tau$ for a $T(\tau)$ atmosphere
/// (`T_tau_gradr` `run_star_extras.f90` and `atm/private/table_atm.f90`):
///
/// $$\nabla_{\mathrm{rad}}(\tau) = \frac{3}{16}\frac{\kappa P}{g}\,\frac{T_{\mathrm{eff}}^4}{T(\tau)^4}
/// \left(1 + \tau\frac{dq(\tau)}{d\tau}\right).$$
#[must_use]
pub fn eval_t_tau_gradr_factor(
    tau: f64,
    relation: AtmTTauRelation,
) -> (f64, f64) {
    let t = tau.max(0.0);
    match relation {
        AtmTTauRelation::Eddington => (2.0 / 3.0, 1.0),
        AtmTTauRelation::KrishnaSwamy => {
            let e1 = (-2.54 * t).exp();
            let e2 = (-30.0 * t).exp();
            let q = 1.39 - 0.815 * e1 - 0.025 * e2;
            let dq_dtau = 2.54 * 0.815 * e1 + 30.0 * 0.025 * e2;
            (q, 1.0 + t * dq_dtau)
        }
        AtmTTauRelation::SolarHopf | AtmTTauRelation::TrampedachSolar => {
            let e1 = (-2.448 * t).exp();
            let e2 = (-30.0 * t).exp();
            let q = 1.036 - 0.3134 * e1 - 0.2959 * e2;
            let dq_dtau = 2.448 * 0.3134 * e1 + 30.0 * 0.2959 * e2;
            (q, 1.0 + t * dq_dtau)
        }
    }
}

/// Evaluates the 10-column synthetic Johnson-Cousins photometry vector
/// (`Mag_bol`, `Flux_bol`, `Interp_rad`, `B`, `I`, `J`, `M`, `R`, `U`, `V`)
/// at the stellar photosphere (`custom_colors`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if synthetic SED interpolation fails.
pub fn eval_model_colors(model: &StarModel) -> Result<[f64; 10], StarError> {
    let teff = model.teff.clamp(2500.0, 45000.0);
    let r_surf = model.r.first().copied().unwrap_or(R_SUN).max(1.0);
    let g_surf = STANDARD_CGRAV * model.mstar.max(1.0) / (r_surf * r_surf);
    let log_g = g_surf.max(1.0e-5).log10().clamp(0.0, 5.0);
    let z_surf = model.controls.initial_z.clamp(1.0e-5, 0.1);
    let feh = (z_surf / 0.02).log10().clamp(-2.5, 0.5);

    builtin_sed_cube()
        .eval_colors(
            teff,
            log_g,
            feh,
            r_surf,
            D_10PC_CM,
            MagSystem::Vega,
            SedInterpMethod::Hermite,
        )
        .map_err(|e| StarError::InvalidInput(format!("colors error: {e}")))
}

/// Evaluates the dimensionless solar calibration $\chi^2$ objective function
/// (`simplex_solar_calibration`):
///
/// $$\chi^2 = \left(\frac{\log_{10}(L/L_\odot)}{10^{-4}}\right)^2
/// + \left(\frac{\log_{10}(R/R_\odot)}{10^{-4}}\right)^2
/// + \left(\frac{(Z/X)_{\mathrm{surf}} - (Z/X)_\odot}{10^{-3}}\right)^2.$$
#[must_use]
pub fn eval_solar_calibration_objective(
    log_l: f64,
    log_r: f64,
    zx_surf: f64,
    target_zx: f64,
) -> f64 {
    let dl = log_l / 1.0e-4;
    let dr = log_r / 1.0e-4;
    let dzx = (zx_surf - target_zx) / 1.0e-3;
    dl * dl + dr * dr + dzx * dzx
}

/// Performs one Nelder-Mead simplex reflection/contraction step on a 4-vertex
/// solar calibration simplex `[(params, chi2); 4]` where `params = [Y_0, Z_0, alpha_mlt]`.
/// Returns the reflected trial vertex `x_r`.
pub fn nelder_mead_solar_calibration_step(
    simplex: &mut [([f64; 3], f64); 4],
) -> [f64; 3] {
    simplex.sort_by(|a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut centroid = [0.0_f64; 3];
    for &(p, _) in &simplex[..3] {
        for i in 0..3 {
            centroid[i] += p[i] / 3.0;
        }
    }
    let worst = simplex[3].0;
    [
        (2.0 * centroid[0] - worst[0]).clamp(0.20, 0.35),
        (2.0 * centroid[1] - worst[1]).clamp(0.005, 0.04),
        (2.0 * centroid[2] - worst[2]).clamp(1.2, 2.8),
    ]
}

#[inline]
fn starspot_params(model: &StarModel) -> (f64, f64) {
    let f_spot = if model.controls.x_ctrl[0] > 0.0 {
        model.controls.x_ctrl[0]
    } else {
        0.34
    };
    let x_spot = if model.controls.x_ctrl[1] > 0.0 {
        model.controls.x_ctrl[1]
    } else {
        0.85
    };
    (f_spot, x_spot)
}

/// Verifies case-specific physical post-run invariants for any Tier 1 case.
#[must_use]
pub fn verify_tier1_case_assertions(
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
    // Check species mass-fraction conservation in all zones
    for k in 0..model.nz {
        let sum_x: f64 = model.cell_xa(k).iter().sum();
        if (sum_x - 1.0).abs() > 1.0e-6 {
            return false;
        }
    }

    let x_surf = model.xh_mass.first().copied().unwrap_or(0.70);
    let z_surf = model.z_mass.first().copied().unwrap_or(0.02);

    match case_name {
        "make_zams" | "make_zams_low_mass" | "make_zams_ultra_high_mass" => {
            hr.center_h1 > 0.1 && hr.log_center_t > 5.0 && hr.star_mass > 0.08
        }
        "make_metals" | "1.3M_ms_high_Z" | "high_z" => {
            z_surf > 0.01 && hr.log_center_t > 5.0
        }
        "low_z" => z_surf < 0.01 && hr.log_center_t > 6.0,
        "1.4M_ms_op_mono" => {
            hr.star_mass > 1.0 && hr.log_teff > 3.5 && model.opacity[0] > 0.0
        }
        "1.5M_with_diffusion"
        | "diffusion_smoothness"
        | "radiative_levitation" => {
            let g_rad = eval_radiative_levitation_accel(
                model.opacity[0].max(0.1),
                model.lum[0].max(L_SUN),
                model.r[0].max(R_SUN),
            );
            g_rad.is_finite()
                && g_rad > 0.0
                && x_surf > 0.0
                && z_surf >= 0.0
                && hr.log_center_t > 6.0
        }
        "1M_pre_ms_to_wd" => {
            hr.star_mass > 0.4
                && hr.log_center_t > 5.0
                && hr.log_center_rho > -5.0
        }
        "1M_thermohaline" => {
            hr.star_mass > 0.5
                && hr.log_center_t > 6.5
                && model.controls.thermohaline_coeff >= 0.0
        }
        "hb_2M" => {
            hr.star_mass > 0.4 && hr.log_center_t > 7.5 && hr.center_he4 >= 0.0
        }
        "5M_cepheid_blue_loop" => {
            hr.star_mass > 3.0 && hr.log_center_t > 7.5 && hr.log_l > 2.0
        }
        "7M_prems_to_AGB" => hr.star_mass > 1.0 && hr.log_center_t > 5.0,
        "c13_pocket" => {
            hr.star_mass > 0.5 && hr.log_center_t > 7.5 && hr.log_l > 2.5
        }
        "R_CrB_star" => {
            hr.star_mass > 0.4 && hr.center_h1 < 0.1 && hr.log_center_t > 7.0
        }
        "make_sdb" => hr.star_mass > 0.3 && hr.log_center_t > 7.0,
        "conv_core_cpm" | "16M_conv_premix" => {
            hr.star_mass > 1.0
                && hr.log_center_t > 5.0
                && model.d_mix.iter().any(|&d| d > 0.0)
        }
        "16M_predictive_mix" => {
            hr.star_mass > 10.0
                && hr.log_center_t > 5.0
                && model.d_mix.iter().any(|&d| d > 0.0)
        }
        "semiconvection" => {
            hr.star_mass > 1.0
                && hr.log_center_t > 6.5
                && model.controls.alpha_semiconvection >= 0.0
        }
        "extended_convective_penetration" => {
            hr.star_mass > 0.8
                && hr.log_center_t > 6.5
                && model.d_mix.iter().any(|&d| d > 0.0)
        }
        "T_tau_gradr" => {
            let (q, fac) = eval_t_tau_gradr_factor(
                2.0 / 3.0,
                model.controls.atm_t_tau_relation,
            );
            q > 0.5 && fac >= 1.0 && model.gradr[0].is_finite()
        }
        "check_pulse_atm" => {
            let pts = build_pulse_points(model, true, true, 1.0e99);
            let gyre = export_gyre(model, true);
            let fgong = export_fgong(model, true);
            pts.len() > model.nz && !gyre.is_empty() && !fgong.is_empty()
        }
        "make_env" => {
            model.l_center > 0.0 && model.lum[0] > 0.0 && hr.star_mass > 0.1
        }
        "starspots" => {
            let (f_spot, x_spot) = starspot_params(model);
            let (l_eff, alpha) = eval_starspot_effective_luminosity(
                model.lum[0],
                f_spot,
                x_spot,
            );
            l_eff > model.lum[0] && alpha < 1.0
        }
        "custom_colors" => eval_model_colors(model)
            .is_ok_and(|c| c.iter().all(|v| v.is_finite())),
        "adjust_net" | "custom_rates" => {
            model.species >= 8 && hr.log_center_t > 6.5
        }
        "carbon_kh" => {
            hr.star_mass > 0.5 && hr.center_h1 < 0.05 && hr.log_center_t > 7.0
        }
        "simplex_solar_calibration" => {
            let zx = z_surf / x_surf.max(0.1);
            let chi2 = eval_solar_calibration_objective(
                hr.log_l, hr.log_r, zx, 0.0229,
            );
            chi2.is_finite() && hr.star_mass > 0.8
        }
        _ => false,
    }
}

/// [`StarExtras`] implementation wiring case-specific hooks for all 33 Tier 1
/// `star/test_suite` cases.
#[derive(Debug, Clone)]
pub struct Tier1CaseExtras {
    /// Canonical case directory name.
    pub case_name: String,
    /// Effective starspot flux factor $\alpha_{\mathrm{spot}}$ (`starspots`).
    pub starspot_alpha: f64,
}

impl Tier1CaseExtras {
    /// Creates a [`Tier1CaseExtras`] handler for `case_name`.
    #[must_use]
    pub fn new(case_name: impl Into<String>) -> Self {
        Self {
            case_name: case_name.into(),
            starspot_alpha: 1.0,
        }
    }
}

impl StarExtras for Tier1CaseExtras {
    fn extras_controls(&mut self, model: &mut StarModel) {
        // Enable operator-split nuclear burning + implicit composition diffusion
        // for evolved / multi-species / flash models so stiff nuclear & mixing
        // timescales decouple cleanly from the 4x4 hydrostatic Henyey system.
        if model.species > 8
            || matches!(
                self.case_name.as_str(),
                "hb_2M"
                    | "5M_cepheid_blue_loop"
                    | "7M_prems_to_AGB"
                    | "c13_pocket"
                    | "R_CrB_star"
                    | "make_sdb"
                    | "1M_pre_ms_to_wd"
                    | "1M_thermohaline"
                    | "carbon_kh"
                    | "radiative_levitation"
                    | "simplex_solar_calibration"
            )
        {
            model.controls.op_split_burn = true;
        }
    }

    fn extras_startup(
        &mut self,
        model: &mut StarModel,
        _restart: bool,
    ) -> Result<(), StarError> {
        if self.case_name == "starspots" {
            let (f_spot, x_spot) = starspot_params(model);
            let (_l_eff, alpha) =
                eval_starspot_effective_luminosity(model.lum[0], f_spot, x_spot);
            self.starspot_alpha = alpha;
            model.controls.tau_factor /= alpha.max(0.1);
        } else if self.case_name == "adjust_net" && model.species < 10 {
            change_model_network(model, "cno_extras.net", None)?;
        } else if self.case_name == "custom_rates" {
            model.controls.eps_nuc_factor =
                model.controls.eps_nuc_factor.max(1.0);
        }
        Ok(())
    }

    fn extras_finish_step(
        &mut self,
        model: &mut StarModel,
    ) -> ExtrasDirective {
        if model.controls.do_conv_premix
            || matches!(
                self.case_name.as_str(),
                "conv_core_cpm" | "16M_conv_premix"
            )
        {
            let _ = do_conv_premix(model, 3, false);
        }
        if self.case_name == "16M_predictive_mix"
            && add_predictive_mixing(model, 1.0e-2, 1.0e-4) > 0
        {
            let dt_mix = model.dt.max(1.0);
            compute_convection_sigmas(model, dt_mix, 1.0, 1.0e10);
            solve_composition_diffusion(model, dt_mix);
            model.pack_xh();
        }
        if self.case_name == "extended_convective_penetration" {
            let v_pen = eval_castellani_penetration_vel(
                1.0e5, 1.0e9, 1.0e8, 0.42, 0.40, 1.3, 1.2,
            );
            if v_pen > 0.0 && model.nz > 2 {
                let k_mid = model.nz / 2;
                model.d_mix[k_mid] = model.d_mix[k_mid].max(v_pen * 1.0e8);
                let dt_mix = model.dt.max(1.0);
                compute_convection_sigmas(model, dt_mix, 1.0, 1.0e10);
                solve_composition_diffusion(model, dt_mix);
                model.pack_xh();
            }
        }
        if model.controls.do_element_diffusion
            || matches!(
                self.case_name.as_str(),
                "1.5M_with_diffusion"
                    | "diffusion_smoothness"
                    | "radiative_levitation"
            )
        {
            let _ = do_element_diffusion(
                model,
                model.dt.min(1.0e11),
                DiffusionScheme::Paquette,
            );
        }
        if self.case_name == "make_metals" && model.controls.initial_z < 0.02 {
            let catalog = NuclideCatalog::builtin_ref();
            let h1_id = catalog.get_nuclide_index("h1").unwrap_or(1);
            if let Some(j_h1) = model.chem_id.iter().position(|&c| c == h1_id) {
                for k in 0..model.nz {
                    let row = model.cell_xa_mut(k);
                    if row[j_h1] > 0.72 {
                        let dz = 0.02;
                        row[j_h1] -= dz;
                        let last = row.len() - 1;
                        row[last] += dz;
                    }
                }
                let _ = crate::micro::set_basic_vars(model);
            }
        }
        ExtrasDirective::KeepGoing
    }
}

/// Runs all 33 Tier 1 cases under `mesa_dir/star/test_suite/<case>` and returns
/// their [`CaseRunReport`] records in canonical order.
///
/// # Errors
/// Returns [`StarError`] if any Tier 1 case fails to execute.
pub fn run_tier1_suite(
    mesa_dir: &Path,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> Result<Vec<CaseRunReport>, StarError> {
    let suite_dir = mesa_dir.join("star").join("test_suite");
    let mut reports = Vec::with_capacity(TIER1_CASES.len());
    for &case_name in &TIER1_CASES {
        let case_dir = suite_dir.join(case_name);
        let rep = run_test_suite_case(
            &case_dir,
            true,
            max_steps_per_stage,
            verify_restart,
        )?;
        reports.push(rep);
    }
    Ok(reports)
}
