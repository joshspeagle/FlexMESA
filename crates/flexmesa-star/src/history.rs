// SPDX-License-Identifier: GPL-3.0-or-later
//! MESA-compatible evolutionary `history.data` summary rows and legacy string
//! snapshot helpers (`star/private/history.f90` and `star/private/profile.f90`).

use flexmesa_const::{LN10, M_SUN, SECYER};

use crate::core_boundaries::{compute_core_masses, compute_power_categories};
use crate::model::StarModel;
use crate::report_utils::{
    center_avg_x_by_name, center_value, eval_conv_and_mx_summary, safe_log10,
};

/// Scalar evolutionary summary row corresponding to a single line of MESA's
/// `LOGS/history.data` (`star/private/history.f90`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HistoryRow {
    /// Evolutionary step counter (`model_number`, `history.f90:1391-1393`).
    pub model_number: usize,
    /// Stellar age in years (`star_age`, `history.f90:1397-1398`).
    pub star_age: f64,
    /// Total stellar mass in solar masses ($M_* / M_\odot$, `star_mass`, `history.f90:1429-1430`).
    pub star_mass: f64,
    /// Base-10 logarithm of timestep in years ($\log_{10}(\Delta t/\mathrm{yr})$, `log_dt`, `history.f90:1416-1417`).
    pub log_dt: f64,
    /// Base-10 logarithm of hydrogen-burning luminosity (`log_LH = safe_log10(power_h_burn)`, `history.f90:1650-1651`).
    pub log_lh: f64,
    /// Base-10 logarithm of helium-burning luminosity (`log_LHe = safe_log10(power_he_burn)`, `history.f90:1655-1656`).
    pub log_lhe: f64,
    /// Base-10 logarithm of metal-burning luminosity (`log_LZ = safe_log10(power_z_burn)`, `history.f90:1665-1666`).
    pub log_lz: f64,
    /// Base-10 logarithm of net nuclear luminosity excluding photodisintegration (`log_Lnuc = safe_log10(power_nuc_burn - power_photo)`, `history.f90:2298-2300`).
    pub log_lnuc: f64,
    /// Base-10 logarithm of surface luminosity ($\log_{10}(L_{\mathrm{surf}}/L_\odot)$, `log_L`, `history.f90:1679-1680`).
    pub log_l: f64,
    /// Base-10 logarithm of effective temperature ($\log_{10}(T_{\mathrm{eff}}/\mathrm{K})$, `log_Teff`, `history.f90:1699-1700`).
    pub log_teff: f64,
    /// Base-10 logarithm of surface radius ($\log_{10}(R_{\mathrm{surf}}/R_\odot)$, `log_R`, `history.f90:1761-1762`).
    pub log_r: f64,
    /// Base-10 logarithm of surface gravity ($\log_{10}(g_{\mathrm{surf}}/\mathrm{cm\,s^{-2}})$, `log_g`, `history.f90:1766-1767`).
    pub log_g: f64,
    /// Base-10 logarithm of central `dq`-averaged temperature (`log_cntr_T` / `log_center_T`, `report.f90:175`, `history.f90:1773-1781`).
    pub log_center_t: f64,
    /// Base-10 logarithm of central `dq`-averaged density (`log_cntr_Rho` / `log_center_Rho`, `report.f90:173`, `history.f90:1775-1783`).
    pub log_center_rho: f64,
    /// Central `dq`-averaged $^{1}\mathrm{H}$ mass fraction (`center_h1`, `star_utils.f90:3632-3655`).
    pub center_h1: f64,
    /// Central `dq`-averaged $^{4}\mathrm{He}$ mass fraction (`center_he4`, `star_utils.f90:3632-3655`).
    pub center_he4: f64,
    /// Central `dq`-averaged $^{12}\mathrm{C}$ mass fraction (`center_c12`, `star_utils.f90:3632-3655`).
    pub center_c12: f64,
    /// Convective core mass in solar masses ($M_\odot$, `mass_conv_core`, `report.f90:456-475`).
    pub mass_conv_core: f64,
    /// Helium core mass in solar masses ($M_\odot$, `he_core_mass`, `report.f90:1029-1259`).
    pub he_core_mass: f64,
    /// Carbon-oxygen core mass in solar masses ($M_\odot$, `co_core_mass`, `report.f90:1029-1259`).
    pub co_core_mass: f64,
    /// Per-step relative global energy conservation error (`rel_E_err`, `history.f90:1912-1914`).
    pub rel_e_err: f64,
    /// Number of Lagrangian zones (`num_zones`, `history.f90:1506-1508`).
    pub num_zones: usize,
    /// Cumulative step retries (`num_retries`, `history.f90:1509-1511`).
    pub num_retries: usize,
    /// Newton solver iterations used on the current step (`num_iters` / `num_solver_iterations`, `history.f90:1586-1591`).
    pub num_iters: usize,
    /// Cumulative Newton solver iterations across the run (`total_num_solver_iterations`, `history.f90:1520-1522`).
    pub total_num_solver_iterations: usize,
}

/// Extracts a [`HistoryRow`] from the current [`StarModel`] state using
/// upstream MESA `26.4.1` column definitions (`star/private/history.f90`,
/// `star/private/report.f90`, and `star/private/star_utils.f90`).
#[must_use]
pub fn extract_history_row(model: &StarModel) -> HistoryRow {
    let nz = model.nz;
    let powers = compute_power_categories(model);
    let cores = compute_core_masses(model);
    let mix = eval_conv_and_mx_summary(model);

    let log_center_t = if nz > 0 {
        center_value(model, &model.ln_t) / LN10
    } else {
        0.0
    };
    let log_center_rho = if nz > 0 {
        center_value(model, &model.lnd) / LN10
    } else {
        -99.0
    };
    let sh = &model.aux.step_hist;
    let rel_e_err = if sh.total_energy_end == 0.0 {
        0.0
    } else {
        sh.error_in_energy_conservation / sh.total_energy_end.abs()
    };

    HistoryRow {
        model_number: model.model_number,
        star_age: model.star_age,
        star_mass: model.mstar / M_SUN,
        log_dt: safe_log10(model.dt / SECYER),
        log_lh: safe_log10(powers.power_h_burn),
        log_lhe: safe_log10(powers.power_he_burn),
        log_lz: safe_log10(powers.power_z_burn),
        log_lnuc: safe_log10(powers.power_nuc_burn - powers.power_photo),
        log_l: safe_log10(model.surface_l_lsun()),
        log_teff: safe_log10(model.teff),
        log_r: safe_log10(model.surface_r_rsun()),
        log_g: safe_log10(model.grav.first().copied().unwrap_or(0.0)),
        log_center_t,
        log_center_rho,
        center_h1: center_avg_x_by_name(model, "h1"),
        center_he4: center_avg_x_by_name(model, "he4"),
        center_c12: center_avg_x_by_name(model, "c12"),
        mass_conv_core: mix.mass_conv_core,
        he_core_mass: cores.he_core_mass,
        co_core_mass: cores.co_core_mass,
        rel_e_err,
        num_zones: nz,
        num_retries: model.num_retries,
        num_iters: sh.last_solver_iterations,
        total_num_solver_iterations: model.total_num_solver_iterations,
    }
}

/// Formats the 17-column compact summary header paired with [`format_history_row`].
///
/// For the full 60-column MESA `LOGS/history.data` 6-line ASCII header, call
/// [`crate::mesa_history::format_star_history_file_header`].
#[must_use]
pub fn format_history_header() -> String {
    format!(
        "{:>12} {:>20} {:>16} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14} \
         {:>14} {:>14} {:>14} {:>14} {:>14} {:>10} {:>10} {:>12}",
        "model_number",
        "star_age",
        "star_mass",
        "log_dt",
        "log_LH",
        "log_L",
        "log_Teff",
        "log_R",
        "log_g",
        "log_center_T",
        "log_center_Rho",
        "center_h1",
        "center_he4",
        "he_core_mass",
        "num_zones",
        "retries",
        "solver_iters"
    )
}

/// Formats a single [`HistoryRow`] as a fixed-width summary line.
#[must_use]
pub fn format_history_row(row: &HistoryRow) -> String {
    format!(
        "{:12} {:20.12e} {:16.10} {:14.8} {:14.8} {:14.8} {:14.8} {:14.8} \
         {:14.8} {:14.8} {:14.8} {:14.8} {:14.8} {:14.8} {:10} {:10} {:12}",
        row.model_number,
        row.star_age,
        row.star_mass,
        row.log_dt,
        row.log_lh,
        row.log_l,
        row.log_teff,
        row.log_r,
        row.log_g,
        row.log_center_t,
        row.log_center_rho,
        row.center_h1,
        row.center_he4,
        row.he_core_mass,
        row.num_zones,
        row.num_retries,
        row.total_num_solver_iterations,
    )
}

/// Formats a complete MESA `profileN.data` radial structure snapshot from
/// `model` using the default `star/defaults/profile_columns.list` schema
/// (`star/private/profile.f90`).
#[must_use]
pub fn format_profile_string(model: &StarModel) -> String {
    let specs = crate::mesa_profile::default_star_profile_specs();
    crate::mesa_profile::format_star_profile_text(model, &specs)
}
