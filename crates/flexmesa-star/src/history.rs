// SPDX-License-Identifier: LGPL-3.0-or-later
//! MESA-compatible evolutionary `history.data` summary rows and radial
//! `profile.data` structure snapshots (`star/private/history.f90` and
//! `star/private/profile.f90`).

use std::fmt::Write as _;

use flexmesa_const::{L_SUN, M_SUN, R_SUN, SECYER};

use crate::model::StarModel;

/// Scalar evolutionary summary row corresponding to a single line of MESA's
/// `LOGS/history.data` (`history.f90`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistoryRow {
    /// Evolutionary step counter (`model_number`).
    pub model_number: usize,
    /// Stellar age in years (`star_age`).
    pub star_age: f64,
    /// Total stellar mass in solar masses ($M_* / M_\odot$, `star_mass`).
    pub star_mass: f64,
    /// Base-10 logarithm of timestep in years ($\log_{10}(\Delta t/\mathrm{yr})$).
    pub log_dt: f64,
    /// Base-10 logarithm of nuclear luminosity ($\log_{10}(L_{\mathrm{nuc}}/L_\odot)$).
    pub log_lh: f64,
    /// Base-10 logarithm of surface luminosity ($\log_{10}(L_{\mathrm{surf}}/L_\odot)$).
    pub log_l: f64,
    /// Base-10 logarithm of effective temperature ($\log_{10}(T_{\mathrm{eff}}/\mathrm{K})$).
    pub log_teff: f64,
    /// Base-10 logarithm of surface radius ($\log_{10}(R_{\mathrm{surf}}/R_\odot)$).
    pub log_r: f64,
    /// Base-10 logarithm of surface gravity ($\log_{10}(g_{\mathrm{surf}}/\mathrm{cm\,s^{-2}})$).
    pub log_g: f64,
    /// Base-10 logarithm of central temperature ($\log_{10}(T_c/\mathrm{K})$).
    pub log_center_t: f64,
    /// Base-10 logarithm of central density ($\log_{10}(\rho_c/\mathrm{g\,cm^{-3}})$).
    pub log_center_rho: f64,
    /// Central $^{1}\mathrm{H}$ mass fraction (`center_h1`).
    pub center_h1: f64,
    /// Central $^{4}\mathrm{He}$ mass fraction (`center_he4`).
    pub center_he4: f64,
    /// Helium core mass in solar masses ($M_\odot$, `he_core_mass`).
    pub he_core_mass: f64,
    /// Number of Lagrangian zones (`num_zones`).
    pub num_zones: usize,
    /// Cumulative step retries (`num_retries`).
    pub num_retries: usize,
    /// Cumulative Newton solver iterations (`total_num_solver_iterations`).
    pub total_num_solver_iterations: usize,
}

/// Extracts a [`HistoryRow`] from the current [`StarModel`] state
/// (`history.f90`).
#[must_use]
pub fn extract_history_row(model: &StarModel) -> HistoryRow {
    let nz = model.nz;
    let c_idx = nz.saturating_sub(1);

    let l_surf = model.surface_l_lsun().max(1.0e-30);
    let r_surf = model.surface_r_rsun().max(1.0e-30);
    let teff = model.teff.max(1.0);
    let g_surf = model.grav.first().copied().unwrap_or(1.0).max(1.0e-30);
    let t_center = model.t.get(c_idx).copied().unwrap_or(1.0).max(1.0);
    let rho_center = model
        .rho
        .get(c_idx)
        .copied()
        .unwrap_or(1.0e-30)
        .max(1.0e-30);

    let powers = crate::core_boundaries::compute_power_categories(model);
    let l_nuc_lsun = powers.power_nuc_burn.max(1.0e-30);

    let center_h1 = model.xh_mass.get(c_idx).copied().unwrap_or(0.0);
    let center_he4 = model.yhe_mass.get(c_idx).copied().unwrap_or(0.0);

    let he_core_mass =
        crate::core_boundaries::compute_core_masses(model).he_core_mass;

    HistoryRow {
        model_number: model.model_number,
        star_age: model.star_age,
        star_mass: model.mstar / M_SUN,
        log_dt: (model.dt.max(1.0e-30) / SECYER).log10(),
        log_lh: l_nuc_lsun.log10(),
        log_l: l_surf.log10(),
        log_teff: teff.log10(),
        log_r: r_surf.log10(),
        log_g: g_surf.log10(),
        log_center_t: t_center.log10(),
        log_center_rho: rho_center.log10(),
        center_h1,
        center_he4,
        he_core_mass,
        num_zones: nz,
        num_retries: model.num_retries,
        total_num_solver_iterations: model.total_num_solver_iterations,
    }
}

/// Formats the MESA `history.data` column header line.
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

/// Formats a single [`HistoryRow`] as a fixed-width MESA `history.data` line.
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

/// Formats a radial `profile.data` structure snapshot from `model`
/// (`star/private/profile.f90`).
#[must_use]
pub fn format_profile_string(model: &StarModel) -> String {
    let nz = model.nz;
    let mut out = String::with_capacity(512 + nz * 180);
    let _ = writeln!(
        out,
        "{:>12} {:>12} {:>16} {:>16} {:>20} {:>20} {:>16} {:>16} {:>16}",
        "model_number",
        "num_zones",
        "initial_mass",
        "initial_z",
        "star_age",
        "time_step",
        "Teff",
        "photosphere_L",
        "photosphere_r"
    );
    let _ = writeln!(
        out,
        "{:12} {:12} {:16.8} {:16.8} {:20.12e} {:20.12e} {:16.8} {:16.8} \
         {:16.8}\n",
        model.model_number,
        nz,
        model.controls.initial_mass,
        model.controls.initial_z,
        model.star_age,
        model.dt / SECYER,
        model.teff,
        model.surface_l_lsun(),
        model.surface_r_rsun(),
    );
    let _ = writeln!(
        out,
        "{:>6} {:>16} {:>14} {:>14} {:>14} {:>14} {:>16} {:>16} {:>12} \
         {:>12} {:>12} {:>8} {:>12} {:>12}",
        "zone",
        "mass",
        "logR",
        "logT",
        "logRho",
        "logP",
        "luminosity",
        "eps_nuc",
        "gradT",
        "gradr",
        "grada",
        "mix_type",
        "x_mass_frac",
        "y_mass_frac"
    );
    for k in 0..nz {
        let _ = writeln!(
            out,
            "{:6} {:16.10} {:14.8} {:14.8} {:14.8} {:14.8} {:16.8e} \
             {:16.8e} {:12.6} {:12.6} {:12.6} {:8} {:12.6} {:12.6}",
            k + 1,
            model.m[k] / M_SUN,
            (model.r[k].max(1.0e-30) / R_SUN).log10(),
            model.t[k].max(1.0).log10(),
            model.rho[k].max(1.0e-30).log10(),
            model.peos[k].max(1.0e-30).log10(),
            model.lum[k] / L_SUN,
            model.eps_nuc[k],
            model.gradt[k],
            model.gradr[k],
            model.grada[k],
            model.mixing_type[k].code(),
            model.xh_mass[k],
            model.yhe_mass[k],
        );
    }
    out
}
