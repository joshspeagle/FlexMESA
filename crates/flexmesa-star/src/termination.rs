// SPDX-License-Identifier: LGPL-3.0-or-later
//! Built-in `&controls` evolutionary termination condition checker
//! (`check_termination_limits`, `check_termination_limits_with_history`) and
//! core boundary mass integration (`compute_core_masses_msun`) corresponding to
//! `star/private/do_one_utils.f90`.

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{L_SUN, M_SUN, R_SUN, SECYER, STANDARD_CGRAV};

use crate::history::{extract_history_row, HistoryRow};
use crate::model::StarModel;

/// Computes core boundary masses ($M_\odot$) from cell composition profiles:
/// - `he_core_mass`: enclosed mass where $X(\mathrm{^1H}) \le 0.01$
/// - `co_core_mass`: enclosed mass where $X(\mathrm{^1H}) \le 0.01$ and $X(\mathrm{^4He}) \le 0.01$
/// - `one_core_mass`: enclosed mass where $X(\mathrm{^{12}C}) \le 0.01$ inside the CO core
#[must_use]
pub fn compute_core_masses_msun(model: &StarModel) -> (f64, f64, f64) {
    let catalog = NuclideCatalog::builtin_ref();
    let h1_id = catalog.get_nuclide_index("h1").unwrap_or(1);
    let he4_id = catalog.get_nuclide_index("he4").unwrap_or(3);
    let c12_id = catalog.get_nuclide_index("c12").unwrap_or(4);

    let h1_j = model.chem_id.iter().position(|&id| id == h1_id);
    let he4_j = model.chem_id.iter().position(|&id| id == he4_id);
    let c12_j = model.chem_id.iter().position(|&id| id == c12_id);

    let thresh = 0.01;
    let mut he_core_m = 0.0;
    let mut co_core_m = 0.0;
    let mut one_core_m = 0.0;

    for k in (0..model.nz).rev() {
        let row = model.cell_xa(k);
        let x_h = h1_j.map_or(0.0, |j| row[j]);
        let x_he = he4_j.map_or(0.0, |j| row[j]);
        let x_c = c12_j.map_or(0.0, |j| row[j]);
        let m_msun = model.m[k] / M_SUN;

        if x_h <= thresh {
            he_core_m = m_msun;
            if x_he <= thresh {
                co_core_m = m_msun;
                if x_c <= thresh && x_h + x_he + x_c <= 0.1 {
                    one_core_m = m_msun;
                }
            }
        } else {
            break;
        }
    }

    (he_core_m, co_core_m, one_core_m)
}

fn species_col_index(
    model: &StarModel,
    catalog: &NuclideCatalog,
    iso_name: &str,
) -> Option<usize> {
    if iso_name.is_empty() {
        return None;
    }
    let cid = catalog.get_nuclide_index(iso_name)?;
    model.chem_id.iter().position(|&id| id == cid)
}

fn species_total_mass_msun(model: &StarModel, j: usize) -> f64 {
    let mut iso_mass_g = 0.0;
    for k in 0..model.nz {
        iso_mass_g += model.dm[k] * model.cell_xa(k)[j];
    }
    iso_mass_g / M_SUN
}

/// Evaluates all built-in `&controls` termination conditions on `model`
/// (`star/private/do_one_utils.f90`). Returns `Some(termination_code)` if any
/// active limit is satisfied.
#[must_use]
pub fn check_termination_limits(model: &StarModel) -> Option<String> {
    check_termination_limits_with_history(model, None)
}

/// Evaluates all built-in `&controls` termination conditions on `model`,
/// reusing `cached_hr` when provided to avoid duplicate [`extract_history_row`]
/// scans per step.
#[must_use]
pub fn check_termination_limits_with_history(
    model: &StarModel,
    cached_hr: Option<&HistoryRow>,
) -> Option<String> {
    let c = &model.controls;
    let nz = model.nz;
    if nz == 0 {
        return None;
    }

    if let Some(max_mod) = c.max_model_number {
        if model.model_number >= max_mod {
            return Some(String::from("max_model_number"));
        }
    }
    if let Some(max_age) = c.max_age {
        if model.star_age >= max_age {
            return Some(String::from("max_age"));
        }
    }
    if let Some(max_age_sec) = c.max_age_in_seconds {
        if model.star_age * SECYER >= max_age_sec {
            return Some(String::from("max_age_in_seconds"));
        }
    }
    if let Some(lim) = c.rsp_max_num_periods {
        if lim >= 0 && (model.aux.rsp.num_periods as isize) >= lim {
            return Some(String::from("RSP_max_num_periods"));
        }
    }
    if c.rsp_flag
        && c.rsp.grekm_avg_abs_limit > 0.0
        && model.aux.rsp.num_periods >= 2
        && model.aux.rsp.grekm_avg_abs < c.rsp.grekm_avg_abs_limit
    {
        return Some(String::from("RSP_GREKM_avg_abs_limit"));
    }

    let catalog = NuclideCatalog::builtin_ref();
    let center_xa = model.cell_xa(nz - 1);

    let n_low = c
        .xa_central_lower_limit_species
        .len()
        .min(c.xa_central_lower_limit.len());
    for i in 0..n_low {
        let iso_name = &c.xa_central_lower_limit_species[i];
        if let Some(j) = species_col_index(model, catalog, iso_name) {
            if center_xa[j] <= c.xa_central_lower_limit[i] {
                return Some(format!("xa_central_lower_limit({iso_name})"));
            }
        }
    }

    let n_up = c
        .xa_central_upper_limit_species
        .len()
        .min(c.xa_central_upper_limit.len());
    for i in 0..n_up {
        let iso_name = &c.xa_central_upper_limit_species[i];
        if let Some(j) = species_col_index(model, catalog, iso_name) {
            if center_xa[j] >= c.xa_central_upper_limit[i] {
                return Some(format!("xa_central_upper_limit({iso_name})"));
            }
        }
    }

    let owned_hr;
    let hr = if let Some(h) = cached_hr {
        h
    } else {
        owned_hr = extract_history_row(model);
        &owned_hr
    };

    if c.stop_near_zams && c.lnuc_div_l_zams_limit > 0.0 && hr.log_l > -20.0 {
        let ratio = 10.0_f64.powf(hr.log_lh - hr.log_l);
        if ratio >= c.lnuc_div_l_zams_limit {
            return Some(String::from("Lnuc_div_L_zams_limit"));
        }
    }

    if c.log_l_upper_limit.is_some_and(|lim| hr.log_l >= lim) {
        return Some(String::from("log_L_upper_limit"));
    }
    if c.log_l_lower_limit.is_some_and(|lim| hr.log_l <= lim) {
        return Some(String::from("log_L_lower_limit"));
    }

    let teff = 10.0_f64.powf(hr.log_teff);
    if c.teff_lower_limit.is_some_and(|lim| teff <= lim) {
        return Some(String::from("Teff_lower_limit"));
    }
    if c.teff_upper_limit.is_some_and(|lim| teff >= lim) {
        return Some(String::from("Teff_upper_limit"));
    }
    if c.log_teff_lower_limit.is_some_and(|lim| hr.log_teff <= lim) {
        return Some(String::from("log_Teff_lower_limit"));
    }
    if c.log_teff_upper_limit.is_some_and(|lim| hr.log_teff >= lim) {
        return Some(String::from("log_Teff_upper_limit"));
    }
    if c.log_g_upper_limit.is_some_and(|lim| hr.log_g >= lim) {
        return Some(String::from("log_g_upper_limit"));
    }
    if c.log_g_lower_limit.is_some_and(|lim| hr.log_g <= lim) {
        return Some(String::from("log_g_lower_limit"));
    }

    if c.log_center_temp_upper_limit
        .is_some_and(|lim| hr.log_center_t >= lim)
    {
        return Some(String::from("log_center_temp_upper_limit"));
    }
    if c.log_center_temp_lower_limit
        .is_some_and(|lim| hr.log_center_t <= lim)
    {
        return Some(String::from("log_center_temp_lower_limit"));
    }
    if c.log_center_density_upper_limit
        .is_some_and(|lim| hr.log_center_rho >= lim)
    {
        return Some(String::from("log_center_density_upper_limit"));
    }
    if c.log_center_density_lower_limit
        .is_some_and(|lim| hr.log_center_rho <= lim)
    {
        return Some(String::from("log_center_density_lower_limit"));
    }

    if let Some(lim) = c.log_max_temp_upper_limit {
        let max_ln_t =
            model.ln_t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max_ln_t / std::f64::consts::LN_10 >= lim {
            return Some(String::from("log_max_temp_upper_limit"));
        }
    }

    let need_cores = c.he_core_mass_limit.is_some()
        || c.co_core_mass_limit.is_some()
        || c.one_core_mass_limit.is_some()
        || c.power_he_burn_upper_limit.is_some()
        || c.envelope_mass_limit.is_some()
        || c.he_layer_mass_lower_limit.is_some()
        || c.stop_at_phase_tp_agb;
    let (he_core_m, co_core_m, one_core_m) = if need_cores {
        compute_core_masses_msun(model)
    } else {
        (0.0, 0.0, 0.0)
    };

    if c.he_core_mass_limit.is_some_and(|lim| he_core_m >= lim) {
        return Some(String::from("he_core_mass_limit"));
    }
    if c.co_core_mass_limit.is_some_and(|lim| co_core_m >= lim) {
        return Some(String::from("co_core_mass_limit"));
    }
    if c.one_core_mass_limit.is_some_and(|lim| one_core_m >= lim) {
        return Some(String::from("one_core_mass_limit"));
    }

    let power_nuc_lsun = 10.0_f64.powf(hr.log_lh);
    if c.power_nuc_burn_upper_limit
        .is_some_and(|lim| power_nuc_lsun >= lim)
    {
        return Some(String::from("power_nuc_burn_upper_limit"));
    }
    if c.power_nuc_burn_lower_limit
        .is_some_and(|lim| power_nuc_lsun <= lim)
    {
        return Some(String::from("power_nuc_burn_lower_limit"));
    }
    let power_he_lsun = if he_core_m > 0.0 { power_nuc_lsun } else { 0.0 };
    if c.power_he_burn_upper_limit
        .is_some_and(|lim| power_he_lsun >= lim)
    {
        return Some(String::from("power_he_burn_upper_limit"));
    }

    if c.star_mass_max_limit.is_some_and(|lim| hr.star_mass >= lim) {
        return Some(String::from("star_mass_max_limit"));
    }
    if c.star_mass_min_limit.is_some_and(|lim| hr.star_mass <= lim) {
        return Some(String::from("star_mass_min_limit"));
    }
    if c.remnant_mass_min_limit
        .is_some_and(|lim| hr.star_mass <= lim)
    {
        return Some(String::from("remnant_mass_min_limit"));
    }

    let env_mass = (hr.star_mass - he_core_m).max(0.0);
    if c.envelope_mass_limit.is_some_and(|lim| env_mass <= lim) {
        return Some(String::from("envelope_mass_limit"));
    }
    let he_layer_mass = (he_core_m - co_core_m).max(0.0);
    if c.he_layer_mass_lower_limit
        .is_some_and(|lim| he_layer_mass <= lim)
    {
        return Some(String::from("he_layer_mass_lower_limit"));
    }

    if let Some(lim) = c.star_species_mass_min_limit {
        if let Some(j) = species_col_index(
            model,
            catalog,
            &c.star_species_mass_min_limit_iso,
        ) {
            if species_total_mass_msun(model, j) <= lim {
                return Some(String::from("star_species_mass_min_limit"));
            }
        }
    }

    if let Some(lim) = c.star_species_mass_max_limit {
        if let Some(j) = species_col_index(
            model,
            catalog,
            &c.star_species_mass_max_limit_iso,
        ) {
            if species_total_mass_msun(model, j) >= lim {
                return Some(String::from("star_species_mass_max_limit"));
            }
        }
    }

    if c.center_r_lower_limit
        .is_some_and(|lim| model.r[nz - 1] / R_SUN <= lim)
    {
        return Some(String::from("center_R_lower_limit"));
    }
    if let Some(lim) = c.log_psurf_upper_limit {
        if model.peos[0].max(1.0e-30).log10() >= lim {
            return Some(String::from("log_Psurf_upper_limit"));
        }
    }
    if c.log_rsurf_upper_limit.is_some_and(|lim| hr.log_r >= lim) {
        return Some(String::from("log_Rsurf_upper_limit"));
    }

    if let Some(lim) = c.fe_core_infall_limit {
        if !model.v.is_empty() {
            let max_infall =
                model.v.iter().map(|&v| -v).fold(0.0_f64, f64::max);
            if max_infall >= lim {
                return Some(String::from("fe_core_infall_limit"));
            }
        }
    }

    if let Some(lim) = c.v_surf_div_v_kh_upper_limit {
        let v_surf = model.v.first().copied().unwrap_or(0.0).abs();
        let r_surf = model.r[0].max(1.0);
        let l_surf = model.lum[0].abs().max(1.0e-10 * L_SUN);
        let t_kh =
            STANDARD_CGRAV * model.mstar * model.mstar / (r_surf * l_surf);
        let v_kh = (r_surf / t_kh.max(1.0)).max(1.0e-30);
        if v_surf / v_kh >= lim {
            return Some(String::from("v_surf_div_v_kh_upper_limit"));
        }
    }

    if c.stop_at_phase_tp_agb && co_core_m > 0.45 && power_he_lsun > 1.0e3 {
        return Some(String::from("stop_at_phase_TP_AGB"));
    }
    if c.stop_at_phase_c_burn
        && hr.center_he4 < 1.0e-2
        && (hr.log_center_t >= 8.5 || power_nuc_lsun >= 1.0)
    {
        return Some(String::from("stop_at_phase_C_burn"));
    }
    if c.stop_at_phase_ne_burn && hr.log_center_t >= 9.0 {
        return Some(String::from("stop_at_phase_Ne_burn"));
    }
    if c.stop_at_phase_o_burn && hr.log_center_t >= 9.15 {
        return Some(String::from("stop_at_phase_O_burn"));
    }

    if let Some(lim) = c.gamma_center_limit {
        if model.gam[nz - 1] >= lim {
            return Some(String::from("gamma_center_limit"));
        }
    }
    if let Some(lim) = c.v_surf_div_v_esc_limit {
        let v_surf = model.v.first().copied().unwrap_or(0.0).abs();
        let r_surf = model.r[0].max(1.0);
        let v_esc = (2.0 * STANDARD_CGRAV * model.mstar.max(1.0) / r_surf)
            .sqrt()
            .max(1.0e-30);
        if v_surf / v_esc >= lim {
            return Some(String::from("v_surf_div_v_esc_limit"));
        }
    }
    if let Some(lim) = c.surf_omega_div_omega_crit_limit {
        let omega_surf = model.omega.first().copied().unwrap_or(0.0).abs();
        let r_surf = model.r[0].max(1.0);
        let omega_crit = (STANDARD_CGRAV * model.mstar.max(1.0)
            / r_surf.powi(3))
        .sqrt()
        .max(1.0e-30);
        if omega_surf / omega_crit >= lim {
            return Some(String::from("surf_omega_div_omega_crit_limit"));
        }
    }
    if let Some(lim) = c.conv_core_gap_dq_limit {
        let mut last_conv_q: Option<f64> = None;
        let mut in_conv = false;
        for k in (0..nz).rev() {
            let is_conv = model.d_mix[k] > 0.0;
            if is_conv {
                if !in_conv {
                    if let Some(q_prev) = last_conv_q {
                        if (model.q[k] - q_prev).abs() <= lim {
                            return Some(String::from(
                                "conv_core_gap_dq_limit",
                            ));
                        }
                    }
                }
                last_conv_q = Some(model.q[k]);
                in_conv = true;
            } else {
                in_conv = false;
            }
        }
    }

    None
}
