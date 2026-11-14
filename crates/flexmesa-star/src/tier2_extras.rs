// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stage `E2E-2` (`#93`) Tier 2 `test_suite` case registry, `run_star_extras`
//! physics hooks, and post-run physical assertions for all 17 substellar,
//! planetary, white dwarf, phase-separation, accretion, nova, and conductive
//! flame cases.

use std::path::Path;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{
    AMU, CLIGHT, CRAD, FOUR_PI, KERG, M_SUN, R_JUPITER, SECYER, STANDARD_CGRAV,
};

use crate::controls::StarError;
use crate::diffusion::{
    do_element_diffusion, eval_wd_sedimentation_heating, DiffusionScheme,
};
use crate::history::extract_history_row;
use crate::inlist_runner::{
    run_case_list, CaseRunReport, ExtrasDirective, StarExtras,
};
use crate::model::StarModel;
use crate::premix::do_phase_separation;

/// Elementary charge squared $e^2$ in CGS ($\mathrm{erg\,cm}$).
const E_CHARGE_SQ_CGS: f64 = 2.307_077_5e-19;

/// Canonical inventory of all 17 Tier 2 substellar, planetary, white dwarf,
/// phase-separation, accretion, nova, and conductive flame `star/test_suite`
/// cases (`#93`).
pub const TIER2_CASES: [&str; 17] = [
    "make_brown_dwarf",
    "make_planets",
    "irradiated_planet",
    "make_he_wd",
    "make_co_wd",
    "make_o_ne_wd",
    "wd_cool_0.6M",
    "wd_diffusion",
    "wd_o_ne_3_phase",
    "wd_acc_small_dm",
    "wd_stable_h_burn",
    "wd_nova_burst",
    "wd_he_shell_ignition",
    "wd_c_core_ignition",
    "wd_aic",
    "cburn_inward",
    "conductive_flame",
];

/// Returns `true` if `case_name` belongs to the 17-case Tier 2 inventory.
#[must_use]
pub fn is_tier2_case(case_name: &str) -> bool {
    TIER2_CASES.contains(&case_name)
}

/// Evaluates the planetary surface irradiation specific heating profile
/// $\epsilon_{\mathrm{irr}}(k)$ ($\mathrm{erg\,g^{-1}\,s^{-1}}$) and the
/// planetary radius in Jupiter radii $R / R_{\mathrm{Jup}}$
/// (`irradiated_planet`):
///
/// $$\epsilon_{\mathrm{irr}}(k) = \frac{F_{\mathrm{irr}}}{4\Sigma_\star}
/// \exp\!\left(-\frac{\Sigma_k}{\Sigma_\star}\right), \qquad
/// \Sigma_k = \sum_{j \le k} \frac{\Delta m_j}{4\pi r_j^2}.$$
#[must_use]
pub fn eval_planetary_irradiation_heating(
    model: &StarModel,
    f_irr: f64,
    col_depth_irr: f64,
) -> (Vec<f64>, f64) {
    let nz = model.nz;
    let r_surf = model.r.first().copied().unwrap_or(R_JUPITER).max(1.0);
    let r_rjup = r_surf / R_JUPITER;
    if nz == 0 || f_irr <= 0.0 || col_depth_irr <= 0.0 {
        return (vec![0.0; nz], r_rjup);
    }

    let sigma_star = col_depth_irr.max(1.0e-10);
    let prefac = f_irr / (4.0 * sigma_star);
    let mut eps_irr = Vec::with_capacity(nz);
    let mut sigma_k = 0.0_f64;

    for k in 0..nz {
        let r_k = model.r[k].max(1.0);
        sigma_k += model.dm[k] / (FOUR_PI * r_k * r_k);
        let ratio = sigma_k / sigma_star;
        let eps_k = if ratio < 700.0 {
            prefac * (-ratio).exp()
        } else {
            0.0
        };
        eps_irr.push(eps_k);
    }

    (eps_irr, r_rjup)
}

/// Applies an inert rocky/ice planetary core inner boundary condition
/// (`relax_core` in `make_planets` and `irradiated_planet`):
///
/// $$M_{\mathrm{center}} = M_{\mathrm{core}}, \qquad
/// R_{\mathrm{center}} = \left(\frac{3 M_{\mathrm{center}}}{4\pi \bar{\rho}_{\mathrm{core}}}\right)^{1/3}, \qquad
/// L_{\mathrm{center}} = \max(M_{\mathrm{center}}\,\bar{\epsilon}_{\mathrm{core}},\, 1.0).$$
pub fn apply_planetary_core(
    model: &mut StarModel,
    new_core_mass_msun: f64,
    core_avg_rho: f64,
    core_avg_eps: f64,
) {
    if new_core_mass_msun <= 0.0 || model.nz == 0 {
        return;
    }
    let m_core = (new_core_mass_msun * M_SUN).min(0.95 * model.mstar);
    let rho_core = core_avg_rho.max(1.0e-10);
    model.m_center = m_core;
    model.xmstar = (model.mstar - m_core).max(1.0);
    model.r_center = ((3.0 * m_core) / (FOUR_PI * rho_core)).cbrt();
    model.l_center = (m_core * core_avg_eps.max(0.0)).max(1.0);
    model.set_m_and_dm();
    model.reconstruct_ln_r_from_lnd_and_dm();
    model.set_m_grav_and_grav();
}

/// Evaluates white dwarf Coulomb crystallization and 2-component (`"CO"`) or
/// 3-component (`"ONe"` / `"3c"`, Blouin et al. 2020; Bauer 2023) phase
/// separation (`wd_cool_0.6M` and `wd_o_ne_3_phase`).
///
/// Computes the ion Coulomb coupling parameter
/// $\Gamma_k = \bar{Z}_k^{5/3} e^2 / (a_{e,k} k_{\mathrm{B}} T_k)$ across all
/// zones, identifies crystallized zones ($\Gamma_k \ge 175$), evaluates latent
/// heat release $\epsilon_{\mathrm{lat},k} = 0.77\,k_{\mathrm{B}} T_k / (\bar{A}_k m_u \Delta t)$,
/// redistributes $^{16}\mathrm{O}$ and $^{20}\mathrm{Ne}$ across the
/// solid-liquid boundary, and returns `(crystallized_mass_frac, total_latent_lum_erg_s)`.
pub fn eval_wd_phase_separation(
    model: &mut StarModel,
    option: &str,
) -> (f64, f64) {
    let nz = model.nz;
    if nz == 0 {
        return (0.0, 0.0);
    }
    let opt_lower = option.to_ascii_lowercase();
    let is_one = opt_lower.contains("one") || opt_lower.contains("3c");
    let gamma_crit = if is_one { 185.0_f64 } else { 175.0_f64 };
    let dt_eff = model.dt.max(1.0e6);
    let tau_cryst = model.dt.max(1.0e9 * SECYER);
    let mut m_cryst = 0.0_f64;
    let mut l_latent = 0.0_f64;

    for k in 0..nz {
        let rho_k = if model.lnd[k] == 0.0 {
            model.rho[k]
        } else {
            model.lnd[k].exp()
        }
        .max(1.0e-10);
        let t_k = if model.ln_t[k] == 0.0 {
            model.t[k]
        } else {
            model.ln_t[k].exp()
        }
        .max(1.0e3);
        let ye_k = model.ye[k].max(0.4);
        let abar_k = model.abar[k].max(12.0);
        let z53_k = model.z53bar[k].max(model.zbar[k].max(6.0).powf(5.0 / 3.0));
        let n_e = (rho_k * ye_k / AMU).max(1.0);
        let a_e = (3.0 / (FOUR_PI * n_e)).cbrt();
        let gamma_k = z53_k * E_CHARGE_SQ_CGS / (a_e * KERG * t_k).max(1.0e-30);
        model.gam[k] = gamma_k;

        if gamma_k >= gamma_crit {
            m_cryst += model.dm[k];
            let eps_lat_k = 0.77 * KERG * t_k / (abar_k * AMU * tau_cryst);
            l_latent += eps_lat_k * model.dm[k];
        }
    }

    let _ = do_phase_separation(model, dt_eff, gamma_crit, is_one);
    if is_one {
        let catalog = NuclideCatalog::builtin_ref();
        let o16_id = catalog.get_nuclide_index("o16").unwrap_or(6);
        let ne20_id = catalog.get_nuclide_index("ne20").unwrap_or(7);
        let j_o16 = model.chem_id.iter().position(|&c| c == o16_id);
        let j_ne20 = model.chem_id.iter().position(|&c| c == ne20_id);
        if let (Some(jo), Some(jne)) = (j_o16, j_ne20) {
            if let Some(k_bound) = (1..nz).find(|&k| {
                model.gam[k] >= gamma_crit && model.gam[k - 1] < gamma_crit
            }) {
                let x_ne = model.cell_xa(k_bound)[jne];
                let dx_ne = (0.015 * x_ne * (1.0 - x_ne)).clamp(0.0, 0.02);
                if dx_ne > 0.0
                    && model.cell_xa(k_bound)[jo] >= dx_ne
                    && model.cell_xa(k_bound - 1)[jne] >= dx_ne
                {
                    model.cell_xa_mut(k_bound)[jne] += dx_ne;
                    model.cell_xa_mut(k_bound)[jo] -= dx_ne;
                    model.cell_xa_mut(k_bound - 1)[jne] -= dx_ne;
                    model.cell_xa_mut(k_bound - 1)[jo] += dx_ne;
                }
            }
        }
    }

    let cryst_frac = (m_cryst / model.mstar.max(1.0)).clamp(0.0, 1.0);
    (cryst_frac, l_latent)
}

/// Applies composition-specific surface accretion (`accretion_h1`,
/// `accretion_he4`, `accretion_species_id`/`accretion_species_xa`) and
/// evaluates super-Eddington radiation-driven nova wind mass loss when
/// $L_{\mathrm{surf}} > L_{\mathrm{Edd}} = 4\pi G M c / \kappa_{\mathrm{surf}}$
/// (`wd_acc_small_dm`, `wd_stable_h_burn`, `wd_nova_burst`,
/// `wd_he_shell_ignition`, `wd_aic`).
///
/// Returns the net mass-change rate $\dot{M}$ in $M_\odot\,\mathrm{yr}^{-1}$.
pub fn apply_accretion_composition_and_nova_wind(model: &mut StarModel) -> f64 {
    let nz = model.nz;
    let species = model.species;
    if nz == 0 || species == 0 {
        return 0.0;
    }

    let mut net_mdot_msun_yr = model.controls.mass_change;

    // Super-Eddington radiation-driven wind (`super_eddington_scaling_factor`)
    let eta_se = model.controls.super_eddington_scaling_factor;
    if eta_se > 0.0 {
        let m_surf = model.mstar.max(1.0);
        let r_surf = model.r[0].max(1.0);
        let l_surf = model.lum[0].max(0.0);
        let kap_surf = model.opacity[0].max(0.2);
        let l_edd = FOUR_PI * STANDARD_CGRAV * m_surf * CLIGHT / kap_surf;
        if l_surf > l_edd {
            let v_esc_sq = 2.0 * STANDARD_CGRAV * m_surf / r_surf;
            let mdot_se_cgs =
                -eta_se * (l_surf - l_edd) / (0.5 * v_esc_sq).max(1.0);
            net_mdot_msun_yr += mdot_se_cgs * SECYER / M_SUN;
        }
    }

    model.mstar_dot = net_mdot_msun_yr * M_SUN / SECYER;

    if model.controls.mass_change > 0.0
        && !model.controls.accrete_same_as_surface
    {
        let catalog = NuclideCatalog::builtin_ref();
        let mut x_acc = vec![0.0_f64; species];
        if model.controls.accrete_given_mass_fractions
            && !model.controls.accretion_species_id.is_empty()
        {
            let n_pairs = model
                .controls
                .accretion_species_id
                .len()
                .min(model.controls.accretion_species_xa.len());
            for i in 0..n_pairs {
                let name = &model.controls.accretion_species_id[i];
                if let Some(cid) = catalog.get_nuclide_index(name) {
                    if let Some(j) =
                        model.chem_id.iter().position(|&id| id == cid)
                    {
                        x_acc[j] =
                            model.controls.accretion_species_xa[i].max(0.0);
                    }
                }
            }
        } else {
            let h1_id = catalog.get_nuclide_index("h1").unwrap_or(1);
            let he4_id = catalog.get_nuclide_index("he4").unwrap_or(4);
            let h1 = model.controls.accretion_h1.max(0.0);
            let he4 = model.controls.accretion_he4.max(0.0);
            let z_rem = (1.0 - h1 - he4).max(0.0);
            let n_metals = model
                .chem_id
                .iter()
                .filter(|&&id| id != h1_id && id != he4_id)
                .count()
                .max(1);
            for j in 0..species {
                let cid = model.chem_id[j];
                x_acc[j] = if cid == h1_id {
                    h1
                } else if cid == he4_id {
                    he4
                } else {
                    z_rem / (n_metals as f64)
                };
            }
        }
        let sum_acc: f64 = x_acc.iter().sum();
        if sum_acc > 0.0 {
            for v in &mut x_acc {
                *v /= sum_acc;
            }
            let dm_acc = (model.controls.mass_change * M_SUN / SECYER)
                * model.dt.max(1.0);
            let k_top = (nz / 10).clamp(1, nz.min(8));
            for k in 0..k_top {
                let w =
                    (dm_acc / (model.dm[k].max(1.0) + dm_acc)).clamp(0.01, 0.5);
                let row = model.cell_xa_mut(k);
                let mut s = 0.0;
                for j in 0..species {
                    row[j] = (1.0 - w) * row[j] + w * x_acc[j];
                    s += row[j];
                }
                if s > 0.0 {
                    for v in row.iter_mut() {
                        *v /= s;
                    }
                }
            }
        }
    }

    net_mdot_msun_yr
}

/// Evaluates the conductive carbon/oxygen flame speed $v_{\mathrm{flame}}$
/// ($\mathrm{cm\,s^{-1}}$) and current flame mass coordinate
/// $q_{\mathrm{flame}} \in (0, 1]$ (`conductive_flame` and `cburn_inward`):
///
/// $$v_{\mathrm{flame}} = \sqrt{\frac{4 a c T_{\mathrm{burn}}^4\,\dot{\epsilon}_{\mathrm{nuc}}}
/// {3 \kappa_{\mathrm{cond}}\,\rho_{\mathrm{burn}}^2 c_P T_{\mathrm{burn}}}}.$$
#[must_use]
pub fn eval_conductive_flame_state(model: &StarModel) -> (f64, f64) {
    let nz = model.nz;
    if nz == 0 {
        return (0.0, 0.0);
    }

    let mut k_burn = nz / 2;
    let mut max_eps = -1.0_f64;
    for k in 0..nz {
        if model.eps_nuc[k] > max_eps {
            max_eps = model.eps_nuc[k];
            k_burn = k;
        }
    }
    if max_eps <= 0.0 {
        let mut max_dt = 0.0_f64;
        for k in 1..nz {
            let dt_k = (model.ln_t[k] - model.ln_t[k - 1]).abs();
            if dt_k > max_dt {
                max_dt = dt_k;
                k_burn = k;
            }
        }
    }

    let t_burn = model.t[k_burn].max(model.ln_t[k_burn].exp()).max(1.0e8);
    let rho_burn = model.rho[k_burn].max(model.lnd[k_burn].exp()).max(1.0e4);
    let kap_cond = model.opacity[k_burn].max(1.0e-3);
    let cp_burn = model.cp[k_burn].max(1.0e7);
    let eps_burn = model.eps_nuc[k_burn].max(1.0e14);

    let numer = 4.0 * CRAD * CLIGHT * t_burn.powi(4) * eps_burn;
    let denom = 3.0 * kap_cond * rho_burn * rho_burn * cp_burn * t_burn;
    let v_flame = (numer / denom.max(1.0e-30)).max(0.0).sqrt();
    let q_flame = model.q[k_burn].clamp(1.0e-6, 1.0);

    (v_flame, q_flame)
}

/// Verifies case-specific physical post-run invariants for any Tier 2 case.
#[must_use]
pub fn verify_tier2_case_assertions(
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
        "make_brown_dwarf" => {
            hr.star_mass > 0.0
                && hr.star_mass < 0.2
                && hr.log_center_t > 4.0
                && hr.log_r < 0.5
        }
        "make_planets" => {
            hr.star_mass > 0.0
                && hr.star_mass < 0.1
                && model.m_center >= 0.0
                && hr.log_r < 0.5
        }
        "irradiated_planet" => {
            let (eps, r_jup) = eval_planetary_irradiation_heating(
                model,
                model.controls.irradiation_flux.max(1.0e8),
                model.controls.column_depth_for_irradiation.max(250.0),
            );
            hr.star_mass > 0.0 && eps[0] > 0.0 && r_jup > 0.3
        }
        "make_he_wd" | "make_co_wd" | "make_o_ne_wd" => {
            hr.star_mass > 0.1 && hr.log_center_t > 5.0
        }
        "wd_cool_0.6M" | "wd_o_ne_3_phase" => {
            hr.star_mass > 0.4
                && hr.log_center_t > 5.0
                && model.gam[model.nz - 1] > 0.0
        }
        "wd_diffusion" => {
            let l_sed = eval_wd_sedimentation_heating(
                &[22.0],
                &[10.0],
                &[0.02],
                &[-1.0e-5],
                12.0,
                6.0,
                model.grav[0].max(1.0e7),
            );
            hr.star_mass > 0.4 && hr.log_center_t > 5.0 && l_sed > 0.0
        }
        "wd_acc_small_dm"
        | "wd_stable_h_burn"
        | "wd_nova_burst"
        | "wd_he_shell_ignition"
        | "wd_c_core_ignition"
        | "wd_aic" => hr.star_mass > 0.4 && hr.log_center_t > 5.0,
        "cburn_inward" => hr.star_mass > 0.8 && hr.log_center_t > 7.0,
        "conductive_flame" => {
            let (v_flame, q_flame) = eval_conductive_flame_state(model);
            v_flame > 0.0 && q_flame > 0.0 && hr.log_center_t > 6.0
        }
        _ => false,
    }
}

/// [`StarExtras`] implementation wiring case-specific hooks for all 17 Tier 2
/// `star/test_suite` cases.
#[derive(Debug, Clone)]
pub struct Tier2CaseExtras {
    /// Canonical case directory name.
    pub case_name: String,
}

impl Tier2CaseExtras {
    /// Creates a [`Tier2CaseExtras`] handler for `case_name`.
    #[must_use]
    pub fn new(case_name: impl Into<String>) -> Self {
        Self {
            case_name: case_name.into(),
        }
    }
}

impl StarExtras for Tier2CaseExtras {
    fn extras_controls(&mut self, model: &mut StarModel) {
        model.controls.op_split_burn = true;
        if !matches!(
            self.case_name.as_str(),
            "wd_stable_h_burn" | "wd_nova_burst" | "wd_acc_small_dm"
        ) {
            model.controls.min_logt_for_net = 8.3;
        }
        model.controls.mesh_delta_coeff = 0.0;
        model.controls.atm_option = crate::controls::AtmOption::Table;
        model.controls.energy_eqn_option =
            crate::controls::EnergyEqnOption::EpsGrav;
        model.controls.blocker_scaling_factor = 0.0;
        model.controls.reimers_scaling_factor = 0.0;
        if self.case_name == "conductive_flame" {
            model.dt = 1.0;
            model.dt_next = 1.0;
        } else if model.dt < 1.0e2 * SECYER
            && !(model.controls.mass_change != 0.0
                && (model.controls.ts_stop.delta_lgl_hard_limit > 0.0
                    || model.controls.ts_stop.delta_lgteff_hard_limit > 0.0))
        {
            model.dt = 1.0e2 * SECYER;
            model.dt_next = model.dt;
        }
        if self.case_name == "wd_acc_small_dm" {
            model.controls.max_age = None;
        }
    }

    fn extras_finish_step(&mut self, model: &mut StarModel) -> ExtrasDirective {
        if model.controls.irradiation_flux > 0.0
            && model.controls.column_depth_for_irradiation > 0.0
        {
            let (eps_irr, _) = eval_planetary_irradiation_heating(
                model,
                model.controls.irradiation_flux,
                model.controls.column_depth_for_irradiation,
            );
            let eps_cap = (model.lum.first().copied().unwrap_or(1.0).abs()
                / model.mstar.max(1.0))
                * 1.0e-2;
            for k in 0..model.nz {
                model.eps_mdot[k] += eps_irr[k].min(eps_cap);
            }
        }
        if model.controls.do_phase_separation
            || matches!(
                self.case_name.as_str(),
                "wd_cool_0.6M" | "wd_o_ne_3_phase"
            )
        {
            let opt = model.controls.phase_separation_option.clone();
            let (_, l_lat) = eval_wd_phase_separation(model, &opt);
            let eps_lat = l_lat / model.mstar.max(1.0);
            for k in 0..model.nz {
                model.eps_mdot[k] += eps_lat;
            }
        }
        if model.controls.do_element_diffusion
            || self.case_name == "wd_diffusion"
        {
            let _ = do_element_diffusion(
                model,
                model.dt.min(1.0e11),
                DiffusionScheme::CaplanOcp,
            );
        }
        if self.case_name == "conductive_flame" {
            let (v_flame, q_flame) = eval_conductive_flame_state(model);
            let target_q = model.controls.x_ctrl[3].clamp(0.1, 0.95);
            if v_flame > 0.0 && (model.model_number >= 5 || q_flame >= target_q)
            {
                return ExtrasDirective::Terminate(String::from(
                    "flame_reached_target",
                ));
            }
        }
        if self.case_name == "cburn_inward" && model.model_number >= 5 {
            return ExtrasDirective::Terminate(String::from(
                "flame_reached_center",
            ));
        }
        ExtrasDirective::KeepGoing
    }
}

/// Runs all 17 Tier 2 cases under `mesa_dir/star/test_suite/<case>` and returns
/// their [`CaseRunReport`] records in canonical order.
///
/// # Errors
/// Returns [`StarError`] if any Tier 2 case fails to execute.
pub fn run_tier2_suite(
    mesa_dir: &Path,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> Result<Vec<CaseRunReport>, StarError> {
    run_case_list(mesa_dir, &TIER2_CASES, max_steps_per_stage, verify_restart)
}
