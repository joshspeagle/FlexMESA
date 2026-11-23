// SPDX-License-Identifier: LGPL-3.0-or-later
//! Upstream MESA `26.4.1` `LOGS/history.data` column parser, evaluator, and
//! incremental flushing file writer (`star/private/history_specs.f90`,
//! `star/private/history.f90`, and `star/defaults/history_columns.list`).

use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use flexmesa_chem::{NuclideCatalog, category_id, category_name};
use flexmesa_const::{
    FOUR_PI, L_SUN, LN10, M_SUN, PI, R_SUN, SECDAY, SECYER,
};
use flexmesa_schema::strip_fortran_comment;

use crate::controls::StarError;
use crate::core_boundaries::CoreBoundaryInfo;
use crate::model::StarModel;
use crate::report_utils::{
    MESA_REF_VERSION_NUMBER, STAR_LOG_COL_WIDTH, StepReportContext,
    center_avg_x_by_name, center_avg_x_slot, center_value, center_value_by,
    csound_face_at, equatorial_and_polar_r_cm, eval_astero_report_scalars,
    eval_conv_time_scales, eval_history_burning_regions,
    eval_history_mixing_regions, eval_ledd_erg_s, eval_shock_info,
    eval_surface_w_div_w_kep, eval_upstream_omega_crit, format_mesa_dbl,
    format_mesa_header_str, get_lrad_div_ledd, gravity_darkening_l_coeff,
    gravity_darkening_teff_coeff, iso_slot_for_chem_id, parse_quoted_include_arg,
    push_mesa_6line_header, resolve_iso_slot, safe_log10, star_avg_x_slot,
    surface_avg_x_by_name, surface_avg_x_slot, write_mesa_dbl_width,
    write_mesa_int_width,
};

/// Active column declarations from upstream MESA `26.4.1`
/// `star/defaults/history_columns.list` (`60` columns).
pub const DEFAULT_STAR_HISTORY_COLUMNS_TEXT: &str = "\
model_number
num_zones
star_age
log_dt
star_mass
log_xmstar
log_abs_mdot
mass_conv_core
conv_mx1_top
conv_mx1_bot
conv_mx2_top
conv_mx2_bot
mx1_top
mx1_bot
mx2_top
mx2_bot
log_power_nuc_burn
power_nuc_burn
log_LH
log_LHe
log_LZ
log_Lnuc
Lnuc
pp
cno
tri_alpha
epsnuc_M_1
epsnuc_M_2
epsnuc_M_3
epsnuc_M_4
epsnuc_M_5
epsnuc_M_6
epsnuc_M_7
epsnuc_M_8
he_core_mass
co_core_mass
one_core_mass
fe_core_mass
neutron_rich_core_mass
log_Teff
log_L
log_R
log_g
v_div_csound_surf
log_cntr_P
log_cntr_Rho
log_cntr_T
center_mu
center_ye
center_abar
center h1
center he4
center c12
center o16
surface c12
surface o16
total_mass h1
total_mass he4
num_retries
num_iters
";

/// Typed cell value written to a MESA `history.data`, `profileN.data`, or
/// `binary_history.data` table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MesaCellValue {
    /// Integer column (`(i40, 1x)` / `(i32, 1x)`).
    Int(i64),
    /// Double-precision column (`(1pes40.16e3, 1x)` / `(1pes32.16e3, 1x)`).
    Float(f64),
}

impl MesaCellValue {
    /// Returns the numeric value as `f64`.
    #[must_use]
    #[inline]
    pub const fn as_f64(self) -> f64 {
        match self {
            Self::Int(v) => v as f64,
            Self::Float(v) => v,
        }
    }
}

/// Nuclide query kind in `history_columns.list` (`star/private/history_specs.f90:35-46`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryIsoQuery {
    /// `center <iso>` (`center_xa_offset`, `history.f90:1384-1386`).
    Center,
    /// `surface <iso>` (`surface_xa_offset`, `history.f90:1381-1383`).
    Surface,
    /// `average <iso>` (`average_xa_offset`, `history.f90:1378-1380`).
    Average,
    /// `total_mass <iso>` (`total_mass_offset`, `history.f90:1372-1374`).
    TotalMass,
    /// `log_center <iso>` (`log_center_xa_offset`, `history.f90:1363-1365`).
    LogCenter,
    /// `log_surface <iso>` (`log_surface_xa_offset`, `history.f90:1360-1362`).
    LogSurface,
    /// `log_average <iso>` (`log_average_xa_offset`, `history.f90:1366-1368`).
    LogAverage,
    /// `log_total_mass <iso>` (`log_total_mass_offset`, `history.f90:1369-1371`).
    LogTotalMass,
    /// `max_eps_nuc_log_xa <iso>` (`max_eps_nuc_offset`, `history.f90:1351-1353`).
    MaxEpsNucLog,
    /// `cz_top_log_xa <iso>` (`cz_top_max_offset`, `history.f90:1354-1356`).
    CzTopLog,
    /// `cz_log_xa <iso>` (`cz_max_offset`, `history.f90:1357-1359`).
    CzLog,
}

/// Resolved column specification from `history_columns.list`
/// (`star/private/history_specs.f90` and `star/private/history.f90`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryColumnSpec {
    /// Built-in scalar history column with canonical output header name.
    Scalar(&'static str),
    /// Nuclear reaction category luminosity $\log_{10}(L_c / L_\odot)$ (`1`-based `cat_id`, `history.f90:1375-1377, 3057-3065`).
    CategoryLuminosity(usize),
    /// Central `dq`-averaged category energy rate `c_log_eps_burn_<cat>` (`1`-based `cat_id`, `history.f90:1348-1350`).
    CenterLogEpsBurn(usize),
    /// Nuclide-indexed abundance/mass query `(query, chem_id, canonical_iso_name)`.
    Nuclide(HistoryIsoQuery, usize, String),
    /// `mix_type_<ii>` from `mixing_regions <N>` (`1`-based `ii`, `history.f90:983-998, 1129-1147`).
    MixType(usize),
    /// `mix_qtop_<ii>` from `mixing_regions <N>`.
    MixQtop(usize),
    /// `mix_relr_type_<ii>` from `mix_relr_regions <N>` (`history.f90:967-982, 1109-1128`).
    MixRelrType(usize),
    /// `mix_relr_top_<ii>` from `mix_relr_regions <N>`.
    MixRelrTop(usize),
    /// `burn_type_<ii>` from `burning_regions <N>` (`history.f90:951-966, 1095-1108`).
    BurnType(usize),
    /// `burn_qtop_<ii>` from `burning_regions <N>`.
    BurnQtop(usize),
    /// `burn_relr_type_<ii>` from `burn_relr_regions <N>` (`history.f90:935-950, 1081-1094`).
    BurnRelrType(usize),
    /// `burn_relr_top_<ii>` from `burn_relr_regions <N>`.
    BurnRelrTop(usize),
}

impl HistoryColumnSpec {
    /// Returns the exact header label written on line 6 of `history.data`
    /// (`do_col_pass2` in `star/private/history.f90:927-1070`).
    #[must_use]
    pub fn column_name(&self) -> String {
        match self {
            Self::Scalar(name) => (*name).to_owned(),
            Self::CategoryLuminosity(cid) => {
                category_name(*cid).unwrap_or("unknown").to_owned()
            }
            Self::CenterLogEpsBurn(cid) => {
                format!(
                    "c_log_eps_burn_{}",
                    category_name(*cid).unwrap_or("unknown")
                )
            }
            Self::Nuclide(q, _, iso) => match q {
                HistoryIsoQuery::Center => format!("center_{iso}"),
                HistoryIsoQuery::Surface => format!("surface_{iso}"),
                HistoryIsoQuery::Average => format!("average_{iso}"),
                HistoryIsoQuery::TotalMass => format!("total_mass_{iso}"),
                HistoryIsoQuery::LogCenter => format!("log_center_{iso}"),
                HistoryIsoQuery::LogSurface => format!("log_surface_{iso}"),
                HistoryIsoQuery::LogAverage => format!("log_average_{iso}"),
                HistoryIsoQuery::LogTotalMass => {
                    format!("log_total_mass_{iso}")
                }
                HistoryIsoQuery::MaxEpsNucLog => {
                    format!("max_eps_nuc_log_{iso}")
                }
                HistoryIsoQuery::CzTopLog => format!("cz_top_log_{iso}"),
                HistoryIsoQuery::CzLog => format!("cz_log_{iso}"),
            },
            Self::MixType(ii) => format!("mix_type_{ii}"),
            Self::MixQtop(ii) => format!("mix_qtop_{ii}"),
            Self::MixRelrType(ii) => format!("mix_relr_type_{ii}"),
            Self::MixRelrTop(ii) => format!("mix_relr_top_{ii}"),
            Self::BurnType(ii) => format!("burn_type_{ii}"),
            Self::BurnQtop(ii) => format!("burn_qtop_{ii}"),
            Self::BurnRelrType(ii) => format!("burn_relr_type_{ii}"),
            Self::BurnRelrTop(ii) => format!("burn_relr_top_{ii}"),
        }
    }
}

const MAX_INCLUDE_LEVEL: usize = 20;

const SUPPORTED_SCALAR_COLUMNS: &[&str] = &[
    "model_number",
    "log_star_age",
    "star_age",
    "log_star_age_sec",
    "star_age_sec",
    "star_age_min",
    "star_age_hr",
    "star_age_day",
    "star_age_yr",
    "day",
    "time_step",
    "log_dt",
    "time_step_sec",
    "log_dt_sec",
    "time_step_days",
    "log_dt_days",
    "log_star_mass",
    "star_mass",
    "log_xmstar",
    "delta_mass",
    "star_mdot",
    "log_abs_mdot",
    "m_center",
    "r_center",
    "m_center_gm",
    "r_center_cm",
    "r_center_km",
    "L_center",
    "log_L_center_ergs_s",
    "log_L_center",
    "v_center",
    "v_center_kms",
    "infall_div_cs",
    "mdot_timescale",
    "kh_div_mdot_timescales",
    "star_gravitational_mass",
    "star_mass_grav_div_mass",
    "e_thermal",
    "total_angular_momentum",
    "log_total_angular_momentum",
    "species",
    "tau_factor",
    "tau_surface",
    "log_tau_center",
    "logT_max",
    "gamma1_min",
    "logQ_max",
    "logQ_min",
    "num_zones",
    "num_retries",
    "phase_of_evolution",
    "total_num_solver_iterations",
    "num_solver_iterations",
    "num_iters",
    "surf_c12_minus_o16",
    "surf_num_c12_div_num_o16",
    "conv_mx1_top",
    "conv_mx1_bot",
    "conv_mx2_top",
    "conv_mx2_bot",
    "mx1_top",
    "mx1_bot",
    "mx2_top",
    "mx2_bot",
    "conv_mx1_top_r",
    "conv_mx1_bot_r",
    "conv_mx2_top_r",
    "conv_mx2_bot_r",
    "mx1_top_r",
    "mx1_bot_r",
    "mx2_top_r",
    "mx2_bot_r",
    "epsnuc_M_1",
    "epsnuc_M_2",
    "epsnuc_M_3",
    "epsnuc_M_4",
    "epsnuc_M_5",
    "epsnuc_M_6",
    "epsnuc_M_7",
    "epsnuc_M_8",
    "power_h_burn",
    "log_LH",
    "power_he_burn",
    "log_LHe",
    "power_photo",
    "Lnuc_photo",
    "power_z_burn",
    "log_LZ",
    "log_Lneu",
    "log_Lneu_nuc",
    "log_Lneu_nonnuc",
    "Lsurf_m",
    "luminosity",
    "log_L",
    "luminosity_ergs_s",
    "log_L_ergs_s",
    "photosphere_cell_log_density",
    "photosphere_cell_density",
    "photosphere_cell_log_opacity",
    "photosphere_cell_opacity",
    "photosphere_cell_log_free_e",
    "photosphere_cell_free_e",
    "log_Teff",
    "Teff",
    "effective_T",
    "photosphere_cell_k",
    "photosphere_cell_log_T",
    "photosphere_cell_T",
    "photosphere_T",
    "photosphere_black_body_T",
    "photosphere_logg",
    "photosphere_m",
    "photosphere_xm",
    "photosphere_L",
    "photosphere_r",
    "photosphere_log_L",
    "photosphere_log_r",
    "photosphere_csound",
    "photosphere_opacity",
    "photosphere_column_density",
    "photosphere_log_column_density",
    "photosphere_v_km_s",
    "v_phot_km_s",
    "photosphere_v_div_cs",
    "min_opacity",
    "log_min_opacity",
    "radius_cm",
    "log_R_cm",
    "radius",
    "log_R",
    "gravity",
    "log_g",
    "log_cntr_dr_cm",
    "log_max_T",
    "log_cntr_T",
    "log_cntr_Rho",
    "log_cntr_P",
    "log_center_T",
    "log_center_Rho",
    "log_center_P",
    "max_T",
    "center_T",
    "center_Rho",
    "center_P",
    "total_entropy",
    "total_internal_energy_after_adjust_mass",
    "total_gravitational_energy_after_adjust_mass",
    "total_radial_kinetic_energy_after_adjust_mass",
    "total_rotational_kinetic_energy_after_adjust_mass",
    "total_turbulent_energy_after_adjust_mass",
    "total_energy_after_adjust_mass",
    "total_internal_energy",
    "total_gravitational_energy",
    "total_radial_kinetic_energy",
    "total_rotational_kinetic_energy",
    "total_turbulent_energy",
    "total_energy",
    "total_energy_foe",
    "log_total_internal_energy",
    "log_total_gravitational_energy",
    "log_total_radial_kinetic_energy",
    "log_total_rotational_kinetic_energy",
    "log_total_turbulent_energy",
    "log_total_energy",
    "avg_abs_v_div_cs",
    "log_avg_abs_v_div_cs",
    "max_abs_v_div_cs",
    "log_max_abs_v_div_cs",
    "avg_abs_v",
    "log_avg_abs_v",
    "max_abs_v",
    "log_max_abs_v",
    "total_eps_grav",
    "work_outward_at_surface",
    "work_inward_at_center",
    "total_nuclear_heating",
    "total_non_nuc_neu_cooling",
    "total_irradiation_heating",
    "total_WD_sedimentation_heating",
    "total_extra_heating",
    "total_energy_sources_and_sinks",
    "error_in_energy_conservation",
    "rel_error_in_energy_conservation",
    "log_rel_error_in_energy_conservation",
    "tot_E_equ_err",
    "tot_E_err",
    "rel_E_err",
    "abs_rel_E_err",
    "log_rel_E_err",
    "cumulative_energy_error",
    "rel_cumulative_energy_error",
    "log_rel_cumulative_energy_error",
    "rel_run_E_err",
    "log_rel_run_E_err",
    "u_surf_km_s",
    "u_surf",
    "u_div_csound_max",
    "u_div_csound_surf",
    "surf_escape_v",
    "v_surf_div_escape_v",
    "v_div_vesc",
    "v_surf_km_s",
    "v_surf",
    "v_surf_div_v_kh",
    "v_div_csound_max",
    "v_div_csound_surf",
    "v_div_cs",
    "log_L_div_Ledd",
    "lum_div_Ledd",
    "max_L_rad_div_Ledd",
    "i_rot_total",
    "surf_avg_j_rot",
    "surf_avg_omega",
    "surf_avg_omega_crit",
    "surf_avg_omega_div_omega_crit",
    "surf_avg_v_rot",
    "surf_avg_v_crit",
    "surf_avg_v_div_v_crit",
    "surf_avg_Lrad_div_Ledd",
    "surf_avg_opacity",
    "surf_avg_logT",
    "surf_avg_logRho",
    "v_wind_Km_per_s",
    "log_rotational_mdot_boost",
    "rotational_mdot_boost",
    "grav_dark_L_polar",
    "grav_dark_Teff_polar",
    "grav_dark_L_equatorial",
    "grav_dark_Teff_equatorial",
    "min_Pgas_div_P",
    "center_degeneracy",
    "log_center_eps_nuc",
    "center_eps_nuc",
    "d_center_eps_nuc_dlnT",
    "d_center_eps_nuc_dlnd",
    "center_eps_grav",
    "center_non_nuc_neu",
    "center_gamma",
    "center_zbar",
    "center_abar",
    "center_mu",
    "center_ye",
    "center_entropy",
    "max_entropy",
    "compactness",
    "compactness_parameter",
    "mu4",
    "m4",
    "max_infall_speed",
    "center_omega",
    "center_omega_div_omega_crit",
    "surf_r_equatorial_div_r_polar",
    "surf_r_equatorial_div_r",
    "surf_r_polar_div_r",
    "h_rich_layer_mass",
    "he_rich_layer_mass",
    "co_rich_layer_mass",
    "he_core_mass",
    "he_core_radius",
    "he_core_lgT",
    "he_core_lgRho",
    "he_core_L",
    "he_core_v",
    "he_core_k",
    "co_core_mass",
    "co_core_radius",
    "co_core_lgT",
    "co_core_lgRho",
    "co_core_L",
    "co_core_v",
    "co_core_k",
    "one_core_mass",
    "one_core_radius",
    "one_core_lgT",
    "one_core_lgRho",
    "one_core_L",
    "one_core_v",
    "one_core_k",
    "fe_core_mass",
    "fe_core_radius",
    "fe_core_lgT",
    "fe_core_lgRho",
    "fe_core_L",
    "fe_core_v",
    "fe_core_k",
    "neutron_rich_core_mass",
    "neutron_rich_core_radius",
    "neutron_rich_core_lgT",
    "neutron_rich_core_lgRho",
    "neutron_rich_core_L",
    "neutron_rich_core_v",
    "neutron_rich_core_k",
    "envelope_mass",
    "envelope_fraction_left",
    "dynamic_timescale",
    "kh_timescale",
    "nuc_timescale",
    "eps_grav_integral",
    "extra_L",
    "log_extra_L",
    "log_abs_Lgrav",
    "log_Lnuc",
    "Lnuc",
    "log_Lnuc_ergs_s",
    "log_power_nuc_burn",
    "power_nuc_burn",
    "log_Lnuc_sub_log_L",
    "mass_loc_of_max_eps_nuc",
    "mass_ext_to_max_eps_nuc",
    "total_ni_co_56",
    "surface_optical_depth",
    "log_surf_optical_depth",
    "log_surf_cell_opacity",
    "log_surf_cell_density",
    "surface_cell_temperature",
    "log_surf_cell_temperature",
    "surface_cell_entropy",
    "log_surf_cell_P",
    "log_surf_cell_pressure",
    "log_surf_cell_z",
    "log_Ledd",
    "min_dr_div_cs",
    "log_min_dr_div_cs",
    "dt_div_min_dr_div_cs",
    "log_dt_div_min_dr_div_cs",
    "dt_cell_collapse",
    "dt_div_dt_cell_collapse",
    "cz_bot_mass",
    "cz_bot_radius",
    "cz_log_xmsun",
    "cz_scale_height",
    "cz_mass",
    "cz_top_mass",
    "cz_top_radius",
    "cz_top_scale_height",
    "cz_zone",
    "cz_top_zone",
    "gradT_excess_alpha",
    "gradT_excess_min_beta",
    "gradT_excess_max_lambda",
    "k_below_const_q",
    "q_below_const_q",
    "logxq_below_const_q",
    "k_const_mass",
    "q_const_mass",
    "logxq_const_mass",
    "logxq_below_just_added",
    "mass_conv_core",
    "mass_semiconv_core",
    "max_gradT_div_grada",
    "max_gradT_sub_grada",
    "min_log_mlt_Gamma",
    "max_conv_vel_div_csound",
    "min_t_eddy",
    "delta_nu",
    "delta_Pg",
    "log_delta_Pg",
    "nu_max",
    "acoustic_cutoff",
    "acoustic_radius",
    "burn_solver_maxsteps",
    "diffusion_solver_steps",
    "diffusion_solver_iters",
    "tot_E",
    "log_tot_E",
    "tot_KE",
    "log_tot_KE",
    "tot_PE",
    "log_tot_PE",
    "tot_IE",
    "log_tot_IE",
    "tot_Et",
    "log_tot_Et",
    "rsp_DeltaR",
    "rsp_DeltaMag",
    "rsp_GREKM",
    "rsp_phase",
    "rsp_period_in_days",
    "rsp_num_periods",
    "lg_Lnuc_tot",
    "H_rich",
    "N_cntr",
    "lg_Lneu",
    "He_core",
    "O_cntr",
    "lg_Lphoto",
    "CO_core",
    "Fe_core",
    "Ne_cntr",
    "Mass",
    "H_cntr",
    "Si_cntr",
    "lg_Mdot",
    "He_cntr",
    "eta_cntr",
    "gam_cntr",
    "lg_Dsurf",
    "C_cntr",
    "zones",
    "retries",
    "h1_czb_mass",
    "virial_thm_P_avg",
    "virial_thm_rel_err",
    "max_tau_conv",
    "min_tau_conv",
    "log_max_tau_conv",
    "log_min_tau_conv",
    "dt_div_max_tau_conv",
    "dt_div_min_tau_conv",
    "fe_core_infall",
    "non_fe_core_infall",
    "shock_mass",
    "shock_q",
    "shock_radius",
    "shock_velocity",
    "shock_csound",
    "shock_lgT",
    "shock_lgRho",
    "shock_lgP",
    "shock_gamma1",
    "shock_entropy",
    "shock_tau",
    "shock_k",
    "shock_pre_lgRho",
];

fn match_scalar_column(tok: &str) -> Option<&'static str> {
    SUPPORTED_SCALAR_COLUMNS
        .iter()
        .copied()
        .find(|c| c.eq_ignore_ascii_case(tok))
}

fn push_unique_spec(
    specs: &mut Vec<HistoryColumnSpec>,
    spec: HistoryColumnSpec,
) {
    if !specs.contains(&spec) {
        specs.push(spec);
    }
}

fn parse_quoted_token(raw: &str) -> Option<&str> {
    parse_quoted_include_arg(raw)
}

fn insert_nuclide_spec(
    specs: &mut Vec<HistoryColumnSpec>,
    q: HistoryIsoQuery,
    iso_raw: &str,
    line_no: usize,
) -> Result<(), StarError> {
    let cat = NuclideCatalog::builtin_ref();
    let trimmed = iso_raw.trim();
    let Some(cid) = cat.get_nuclide_index(trimmed) else {
        return Err(StarError::InvalidInput(format!(
            "history_columns.list line {line_no}: unknown nuclide {trimmed:?}"
        )));
    };
    let canon = cat
        .get_by_id(cid)
        .map_or_else(|| trimmed.to_ascii_lowercase(), |n| n.name.clone());
    push_unique_spec(specs, HistoryColumnSpec::Nuclide(q, cid, canon));
    Ok(())
}

fn insert_all_active_abundances(
    specs: &mut Vec<HistoryColumnSpec>,
    q: HistoryIsoQuery,
    model: &StarModel,
) {
    let cat = NuclideCatalog::builtin_ref();
    for &cid in &model.chem_id {
        let canon = cat
            .get_by_id(cid)
            .map_or_else(|| format!("iso_{cid}"), |n| n.name.clone());
        push_unique_spec(specs, HistoryColumnSpec::Nuclide(q, cid, canon));
    }
}

fn parse_iso_prefix_query(prefix: &str) -> Option<HistoryIsoQuery> {
    match prefix.to_ascii_lowercase().as_str() {
        "center" => Some(HistoryIsoQuery::Center),
        "surface" => Some(HistoryIsoQuery::Surface),
        "average" => Some(HistoryIsoQuery::Average),
        "total_mass" => Some(HistoryIsoQuery::TotalMass),
        "log_center" => Some(HistoryIsoQuery::LogCenter),
        "log_surface" => Some(HistoryIsoQuery::LogSurface),
        "log_average" => Some(HistoryIsoQuery::LogAverage),
        "log_total_mass" => Some(HistoryIsoQuery::LogTotalMass),
        "max_eps_nuc_log_xa" => Some(HistoryIsoQuery::MaxEpsNucLog),
        "cz_top_log_xa" => Some(HistoryIsoQuery::CzTopLog),
        "cz_log_xa" => Some(HistoryIsoQuery::CzLog),
        _ => None,
    }
}

/// Parses `history_columns.list` text recursively (supporting `include '...'`
/// and `include ''`) into deduplicated [`HistoryColumnSpec`] entries
/// (`add_history_columns` in `star/private/history_specs.f90:72-711`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] on unknown/unported column names, bad
/// region counts, or excessive include nesting, and [`StarError::IoError`] if an
/// included file cannot be opened.
pub fn parse_star_history_columns(
    text: &str,
    base_dir: Option<&Path>,
    model: &StarModel,
) -> Result<Vec<HistoryColumnSpec>, StarError> {
    let mut specs = Vec::new();
    parse_history_columns_inner(text, base_dir, model, 1, &mut specs)?;
    if specs.is_empty() {
        return Err(StarError::InvalidInput(String::from(
            "history_columns.list produced zero active columns",
        )));
    }
    Ok(specs)
}

fn parse_history_columns_inner(
    text: &str,
    base_dir: Option<&Path>,
    model: &StarModel,
    level: usize,
    specs: &mut Vec<HistoryColumnSpec>,
) -> Result<(), StarError> {
    if level > MAX_INCLUDE_LEVEL {
        return Err(StarError::InvalidInput(format!(
            "history_columns.list exceeded max include depth ({MAX_INCLUDE_LEVEL})"
        )));
    }
    for (idx, raw_line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let (code, _) = strip_fortran_comment(raw_line);
        if code.is_empty() {
            continue;
        }
        if let Some(rest) = code.strip_prefix("include") {
            if rest.is_empty()
                || rest.starts_with(char::is_whitespace)
                || rest.starts_with(['\'', '"'])
            {
                let Some(inc_target) = parse_quoted_token(rest) else {
                    return Err(StarError::InvalidInput(format!(
                        "history_columns.list line {line_no}: malformed include directive {code:?}"
                    )));
                };
                if inc_target.trim().is_empty() {
                    parse_history_columns_inner(
                        DEFAULT_STAR_HISTORY_COLUMNS_TEXT,
                        base_dir,
                        model,
                        level + 1,
                        specs,
                    )?;
                } else {
                    let inc_path = match base_dir {
                        Some(d) => d.join(inc_target.trim()),
                        None => PathBuf::from(inc_target.trim()),
                    };
                    let inc_text =
                        fs::read_to_string(&inc_path).map_err(|e| {
                            StarError::IoError(format!(
                            "failed to read included history columns {}: {e}",
                            inc_path.display()
                        ))
                        })?;
                    parse_history_columns_inner(
                        &inc_text,
                        inc_path.parent().or(base_dir),
                        model,
                        level + 1,
                        specs,
                    )?;
                }
                continue;
            }
        }

        let parts: Vec<&str> = code.split_whitespace().collect();
        if parts.len() == 1 {
            let tok = parts[0];
            let low = tok.to_ascii_lowercase();
            match low.as_str() {
                "add_center_abundances" => {
                    insert_all_active_abundances(
                        specs,
                        HistoryIsoQuery::Center,
                        model,
                    );
                    continue;
                }
                "add_surface_abundances" => {
                    insert_all_active_abundances(
                        specs,
                        HistoryIsoQuery::Surface,
                        model,
                    );
                    continue;
                }
                "add_average_abundances" => {
                    insert_all_active_abundances(
                        specs,
                        HistoryIsoQuery::Average,
                        model,
                    );
                    continue;
                }
                "add_total_mass" => {
                    insert_all_active_abundances(
                        specs,
                        HistoryIsoQuery::TotalMass,
                        model,
                    );
                    continue;
                }
                "add_log_center_abundances" => {
                    insert_all_active_abundances(
                        specs,
                        HistoryIsoQuery::LogCenter,
                        model,
                    );
                    continue;
                }
                "add_log_surface_abundances" => {
                    insert_all_active_abundances(
                        specs,
                        HistoryIsoQuery::LogSurface,
                        model,
                    );
                    continue;
                }
                "add_log_average_abundances" => {
                    insert_all_active_abundances(
                        specs,
                        HistoryIsoQuery::LogAverage,
                        model,
                    );
                    continue;
                }
                "add_log_total_mass" => {
                    insert_all_active_abundances(
                        specs,
                        HistoryIsoQuery::LogTotalMass,
                        model,
                    );
                    continue;
                }
                // MESA 26.4.1 `do_colors`
                // (`star/private/history_specs.f90:237-271, 456-469`) has the
                // legacy BC loop commented out (`! TODO. below is old module`),
                // so these four directives are valid 0-column no-ops.
                "add_bc" | "add_abs_mag" | "add_lum_band"
                | "add_log_lum_band" => {
                    continue;
                }
                _ => {}
            }

            // 1. Exact scalar column match MUST precede `<prefix>_<iso>` check
            // (`history_specs.f90:526-538`).
            if let Some(canon_scalar) = match_scalar_column(tok) {
                push_unique_spec(
                    specs,
                    HistoryColumnSpec::Scalar(canon_scalar),
                );
                continue;
            }
            // 2. Reaction category match (`history_specs.f90:539-543`).
            let cid = category_id(&low);
            if cid > 0 {
                push_unique_spec(
                    specs,
                    HistoryColumnSpec::CategoryLuminosity(cid),
                );
                continue;
            }
            // 3. Underscore-joined `<prefix>_<iso>` or `c_log_eps_burn_<cat>`
            // (`do1` in `history_specs.f90:613-640`).
            if let Some(cat_str) = low.strip_prefix("c_log_eps_burn_") {
                let c_id = category_id(cat_str);
                if c_id > 0 {
                    push_unique_spec(
                        specs,
                        HistoryColumnSpec::CenterLogEpsBurn(c_id),
                    );
                    continue;
                }
            }
            if let Some((prefix_part, iso_part)) = low.rsplit_once('_') {
                if let Some(q) = parse_iso_prefix_query(prefix_part) {
                    insert_nuclide_spec(specs, q, iso_part, line_no)?;
                    continue;
                }
            }
            return Err(StarError::InvalidInput(format!(
                "history_columns.list line {line_no}: unsupported or unknown history column {tok:?}"
            )));
        }

        if parts.len() == 2 {
            let head = parts[0].to_ascii_lowercase();
            let arg = parts[1];
            match head.as_str() {
                "mixing_regions" | "mix_relr_regions" | "burning_regions"
                | "burn_relr_regions" => {
                    let cnt: usize = arg
                        .parse()
                        .ok()
                        .filter(|&n| (1..=1000).contains(&n))
                        .ok_or_else(|| {
                            StarError::InvalidInput(format!(
                                "history_columns.list line {line_no}: bad integer count {arg:?} for {head}"
                            ))
                        })?;
                    for ii in 1..=cnt {
                        match head.as_str() {
                            "mixing_regions" => {
                                push_unique_spec(
                                    specs,
                                    HistoryColumnSpec::MixType(ii),
                                );
                                push_unique_spec(
                                    specs,
                                    HistoryColumnSpec::MixQtop(ii),
                                );
                            }
                            "mix_relr_regions" => {
                                push_unique_spec(
                                    specs,
                                    HistoryColumnSpec::MixRelrType(ii),
                                );
                                push_unique_spec(
                                    specs,
                                    HistoryColumnSpec::MixRelrTop(ii),
                                );
                            }
                            "burning_regions" => {
                                push_unique_spec(
                                    specs,
                                    HistoryColumnSpec::BurnType(ii),
                                );
                                push_unique_spec(
                                    specs,
                                    HistoryColumnSpec::BurnQtop(ii),
                                );
                            }
                            _ => {
                                push_unique_spec(
                                    specs,
                                    HistoryColumnSpec::BurnRelrType(ii),
                                );
                                push_unique_spec(
                                    specs,
                                    HistoryColumnSpec::BurnRelrTop(ii),
                                );
                            }
                        }
                    }
                    continue;
                }
                "c_log_eps_burn" => {
                    let cid = category_id(&arg.to_ascii_lowercase());
                    if cid == 0 {
                        return Err(StarError::InvalidInput(format!(
                            "history_columns.list line {line_no}: unknown category {arg:?}"
                        )));
                    }
                    push_unique_spec(
                        specs,
                        HistoryColumnSpec::CenterLogEpsBurn(cid),
                    );
                    continue;
                }
                // Legacy individual colors columns have no filter lookup in
                // MESA 26.4.1 (`do1_color` at `star/private/history_specs.f90:675-690`,
                // where `colors/public/colors_lib.f90::get_bc_id_by_name` returns `-1`).
                "bc" | "abs_mag" | "lum_band" | "log_lum_band" => {
                    return Err(StarError::InvalidInput(format!(
                        "history_columns.list line {line_no}: legacy colors \
                         column '{head} {arg}' has no active filter table in \
                         MESA 26.4.1 (get_bc_id_by_name returns -1)"
                    )));
                }
                _ => {
                    if let Some(q) = parse_iso_prefix_query(&head) {
                        insert_nuclide_spec(specs, q, arg, line_no)?;
                        continue;
                    }
                }
            }
        }

        return Err(StarError::InvalidInput(format!(
            "history_columns.list line {line_no}: unsupported directive {code:?}"
        )));
    }
    Ok(())
}

/// Resolves the active `history_columns.list` for `model` in `case_dir`
/// (`set_history_columns` in `star/private/history_specs.f90:108-120, 714-781`).
pub fn resolve_star_history_columns(
    model: &StarModel,
    case_dir: Option<&Path>,
) -> Result<Vec<HistoryColumnSpec>, StarError> {
    let requested = model.job.history_columns_file.trim();
    if !requested.is_empty() {
        let path = match case_dir {
            Some(d) => d.join(requested),
            None => PathBuf::from(requested),
        };
        let text = fs::read_to_string(&path).map_err(|e| {
            StarError::IoError(format!(
                "failed to open history_columns_file {}: {e}",
                path.display()
            ))
        })?;
        return parse_star_history_columns(
            &text,
            path.parent().or(case_dir),
            model,
        );
    }
    if let Some(d) = case_dir {
        let local = d.join("history_columns.list");
        if local.is_file() {
            let text = fs::read_to_string(&local).map_err(|e| {
                StarError::IoError(format!(
                    "failed to open {}: {e}",
                    local.display()
                ))
            })?;
            return parse_star_history_columns(&text, Some(d), model);
        }
    }
    parse_star_history_columns(
        DEFAULT_STAR_HISTORY_COLUMNS_TEXT,
        case_dir,
        model,
    )
}

/// Returns the default 60 [`HistoryColumnSpec`] items of MESA `26.4.1`.
#[must_use]
pub fn default_star_history_specs() -> Vec<HistoryColumnSpec> {
    let dummy = StarModel::new(
        1,
        vec![1],
        crate::StarJobControls::default(),
        crate::StarControls::default(),
    );
    parse_star_history_columns(DEFAULT_STAR_HISTORY_COLUMNS_TEXT, None, &dummy)
        .expect("embedded default star history_columns.list is valid")
}

/// Returns the 60 canonical column names of `star/defaults/history_columns.list`.
#[must_use]
pub fn default_star_history_column_names() -> Vec<String> {
    default_star_history_specs()
        .iter()
        .map(HistoryColumnSpec::column_name)
        .collect()
}

fn eval_core_bdy_prop(bdy: &CoreBoundaryInfo, suffix: &str) -> MesaCellValue {
    match suffix {
        "radius" => MesaCellValue::Float(bdy.radius_rsun),
        "lgT" => MesaCellValue::Float(bdy.lg_t),
        "lgRho" => MesaCellValue::Float(bdy.lg_rho),
        "L" => MesaCellValue::Float(bdy.lum_lsun),
        "v" => MesaCellValue::Float(bdy.vel_cms),
        _ => MesaCellValue::Int(bdy.kbdy_1based as i64),
    }
}

fn eval_ejecta_mass_g(model: &StarModel) -> f64 {
    if (!model.controls.u_flag && !model.controls.v_flag) || model.nz == 0 {
        return 0.0;
    }
    let mut v_div_vesc_prev = 0.0_f64;
    for k in 0..model.nz {
        let v = if model.controls.u_flag {
            if k == 0 {
                model.u[0]
            } else {
                0.5 * (model.u[k - 1] + model.u[k])
            }
        } else {
            model.v[k]
        };
        let vesc = (2.0 * model.cgrav[k].max(0.0) * model.m[k].max(0.0)
            / model.r[k].max(1.0e-99))
        .sqrt()
        .max(1.0e-99);
        let v_div_vesc = v / vesc;
        if v_div_vesc < 1.0 {
            if k == 0 {
                return 0.0;
            }
            let dm = flexmesa_num::find0(
                0.0,
                v_div_vesc_prev - 1.0,
                model.dm[k - 1],
                v_div_vesc - 1.0,
            )
            .unwrap_or(0.0)
            .max(0.0);
            let outer_sum: f64 = if k >= 2 {
                model.dm[..=(k - 2)].iter().sum()
            } else {
                0.0
            };
            return outer_sum + dm;
        }
        v_div_vesc_prev = v_div_vesc;
    }
    0.0
}

/// Evaluates `min_dr_div_cs` and 1-based `min_k` matching `set_abs_du_div_cs` and
/// `min_dr_div_cs` (`star/private/star_utils.f90:1268-1325, 1488-1541`).
pub(crate) fn eval_min_dr_div_cs(model: &StarModel) -> (f64, i64) {
    let nz = model.nz;
    if nz == 0 {
        return (1.0e99, 0);
    }
    let mut min_k_1based = nz as i64;
    if (!model.controls.v_flag && !model.controls.u_flag) || nz < 2 {
        return (1.0e99, min_k_1based);
    }

    let remnant_mass = if model
        .controls
        .ts_stop
        .check_remnant_only_for_dt_div_min_dr_div_cs_limit
    {
        model.m[0] - eval_ejecta_mass_g(model)
    } else {
        model.m[0]
    };

    let o = &model.controls.output;
    let mut abs_du_plus_cs = vec![1.0e99_f64; nz];
    let mut abs_du_div_cs = vec![1.0e99_f64; nz];
    if model.controls.v_flag {
        let vs = if model.v_start.len() >= nz {
            &model.v_start[..nz]
        } else {
            &model.v[..nz]
        };
        for k in 1..nz {
            let abs_du = (vs[k] - vs[k - 1]).abs();
            let lo = k.saturating_sub(5);
            let hi = (k + 5).min(nz - 1);
            let cs = model.csound[lo..=hi]
                .iter()
                .copied()
                .fold(1.0e-99, f64::max);
            abs_du_plus_cs[k] = abs_du + cs;
            abs_du_div_cs[k] = abs_du / cs;
        }
        abs_du_plus_cs[0] = abs_du_plus_cs[1];
        abs_du_div_cs[0] = abs_du_div_cs[1];
    } else {
        let us = if model.u_start.len() >= nz {
            &model.u_start[..nz]
        } else {
            &model.u[..nz]
        };
        let cs_start = if model.aux.csound_start.len() >= nz {
            &model.aux.csound_start[..nz]
        } else {
            &model.csound[..nz]
        };
        for k in 1..(nz - 1) {
            let abs_du =
                (us[k] - us[k + 1]).abs().max((us[k] - us[k - 1]).abs());
            let lo = k.saturating_sub(5);
            let hi = (k + 5).min(nz - 1);
            let cs = model.csound[lo..=hi]
                .iter()
                .copied()
                .fold(1.0e-99, f64::max);
            abs_du_plus_cs[k] = abs_du + cs;
            abs_du_div_cs[k] = abs_du / cs;
        }
        let k_last = nz - 1;
        let cs_last = cs_start[k_last].max(1.0e-99);
        abs_du_plus_cs[k_last] = (us[k_last] - us[k_last - 1]).abs() + cs_last;
        abs_du_div_cs[k_last] = (us[k_last] - us[k_last - 1]).abs() / cs_last;
        if nz >= 3 {
            let cs1 = cs_start[1].max(1.0e-99);
            abs_du_plus_cs[1] = (us[1] - us[2]).abs() + cs1;
            abs_du_div_cs[1] = (us[1] - us[2]).abs() / cs1;
        }
        abs_du_plus_cs[0] = abs_du_plus_cs[1];
        abs_du_div_cs[0] = abs_du_div_cs[1];
    }
    for _ in 0..3 {
        for k in 1..(nz - 1) {
            abs_du_div_cs[k] = (abs_du_div_cs[k - 1]
                + abs_du_div_cs[k]
                + abs_du_div_cs[k + 1])
                / 3.0;
        }
    }

    let k_min_0 = o.min_k_for_dt_div_min_dr_div_cs_limit.max(1) - 1;
    let mut best = 1.0e99_f64;
    if model.controls.v_flag {
        let vs = if model.v_start.len() >= nz {
            &model.v_start[..nz]
        } else {
            &model.v[..nz]
        };
        for k in k_min_0..(nz - 1) {
            if model.m[k] > remnant_mass {
                continue;
            }
            if model.q[k] > o.max_q_for_dt_div_min_dr_div_cs_limit {
                continue;
            }
            if model.q[k] < o.min_q_for_dt_div_min_dr_div_cs_limit {
                break;
            }
            let cs = model.csound[k].max(1.0e-99);
            if vs[k].abs() / cs
                < o.min_abs_u_div_cs_for_dt_div_min_dr_div_cs_limit
            {
                continue;
            }
            if abs_du_div_cs[k]
                < o.min_abs_du_div_cs_for_dt_div_min_dr_div_cs_limit
            {
                continue;
            }
            let dt_cs = (model.r[k] - model.r[k + 1]) / cs;
            if dt_cs < best {
                best = dt_cs;
                min_k_1based = (k + 1) as i64;
            }
        }
    } else {
        let us = if model.u_start.len() >= nz {
            &model.u_start[..nz]
        } else {
            &model.u[..nz]
        };
        for k in k_min_0..(nz - 1) {
            if model.m[k] > remnant_mass {
                continue;
            }
            if model.q[k] > o.max_q_for_dt_div_min_dr_div_cs_limit {
                continue;
            }
            if model.q[k] < o.min_q_for_dt_div_min_dr_div_cs_limit {
                break;
            }
            let cs = model.csound[k].max(1.0e-99);
            if us[k].abs() / cs
                < o.min_abs_u_div_cs_for_dt_div_min_dr_div_cs_limit
            {
                continue;
            }
            if abs_du_div_cs[k]
                < o.min_abs_du_div_cs_for_dt_div_min_dr_div_cs_limit
            {
                continue;
            }
            let dt_cs =
                (model.r[k] - model.r[k + 1]) / abs_du_plus_cs[k].max(1.0e-99);
            if dt_cs < best {
                best = dt_cs;
                min_k_1based = (k + 1) as i64;
            }
        }
    }
    (best, min_k_1based)
}

pub(crate) fn eval_min_cell_collapse_time(model: &StarModel) -> f64 {
    if !model.controls.v_flag || model.nz < 2 {
        return 1.0e99;
    }
    let mut rp1 = model.r_center;
    let mut vp1 = model.v_center;
    let mut best = 1.0e99;
    for k in (1..model.nz).rev() {
        let r00 = model.r[k];
        let v00 = model.v[k];
        if r00 <= rp1 {
            return -1.0;
        }
        if vp1 > v00 {
            let t = (r00 - rp1) / (vp1 - v00);
            if t < best {
                best = t;
            }
        }
        rp1 = r00;
        vp1 = v00;
    }
    best
}

#[allow(clippy::too_many_lines)]
fn eval_scalar_column(
    name: &'static str,
    model: &StarModel,
    ctx: &StepReportContext,
    astero_cache: &mut Option<(f64, f64, f64, f64)>,
) -> MesaCellValue {
    let nz = model.nz;
    let star_mass = model.mstar / M_SUN;
    let star_mdot = model.mstar_dot / (M_SUN / SECYER);
    let time_step_yr = model.dt / SECYER;
    let l_surf_lsun = model.surface_l_lsun();
    let r_surf_cm = model.r.first().copied().unwrap_or(0.0);
    let r_surf_rsun = r_surf_cm / R_SUN;
    let g_surf = model.grav.first().copied().unwrap_or(0.0);
    let csound_surf = csound_face_at(model, 0);
    let v_surf = if model.controls.u_flag {
        model.u.first().copied().unwrap_or(0.0)
    } else if model.controls.v_flag {
        model.v.first().copied().unwrap_or(0.0)
    } else {
        0.0
    };
    let has_vel = model.controls.u_flag || model.controls.v_flag;
    let vel_slice: &[f64] = if model.controls.u_flag {
        &model.u
    } else {
        &model.v
    };
    let sh = &model.aux.step_hist;
    let e_end = ctx.energies_erg.5;

    match name {
        "model_number" => MesaCellValue::Int(model.model_number as i64),
        "num_zones" | "zones" => MesaCellValue::Int(nz as i64),
        "species" => MesaCellValue::Int(model.species as i64),
        "num_retries" | "retries" => {
            MesaCellValue::Int(model.num_retries as i64)
        }
        "total_num_solver_iterations" => {
            MesaCellValue::Int(model.total_num_solver_iterations as i64)
        }
        "num_solver_iterations" | "num_iters" => {
            MesaCellValue::Int(sh.last_solver_iterations as i64)
        }
        "burn_solver_maxsteps" => {
            MesaCellValue::Int(if model.controls.op_split_burn {
                sh.burn_max_num_substeps as i64
            } else {
                0
            })
        }
        "rotation_solver_steps" => {
            MesaCellValue::Int(sh.num_rotation_solver_steps as i64)
        }
        "diffusion_solver_steps" => {
            MesaCellValue::Int(sh.num_diffusion_solver_steps as i64)
        }
        "diffusion_solver_iters" => {
            MesaCellValue::Int(sh.num_diffusion_solver_iters as i64)
        }
        "photosphere_cell_k" => {
            MesaCellValue::Int(ctx.phot.cell_k_1based as i64)
        }
        "phase_of_evolution" => {
            MesaCellValue::Int(i64::from(sh.phase_of_evolution))
        }

        "star_age" | "star_age_yr" => MesaCellValue::Float(model.star_age),
        "log_star_age" => MesaCellValue::Float(safe_log10(model.star_age)),
        "star_age_sec" => MesaCellValue::Float(model.star_age * SECYER),
        "log_star_age_sec" => {
            MesaCellValue::Float(safe_log10(model.star_age * SECYER))
        }
        "star_age_min" => MesaCellValue::Float(model.star_age * SECYER / 60.0),
        "star_age_hr" => MesaCellValue::Float(model.star_age * SECYER / 3600.0),
        "star_age_day" | "day" => {
            MesaCellValue::Float(model.star_age * SECYER / SECDAY)
        }

        "time_step" => MesaCellValue::Float(time_step_yr),
        "log_dt" => MesaCellValue::Float(safe_log10(time_step_yr)),
        "time_step_sec" => MesaCellValue::Float(model.dt),
        "log_dt_sec" => MesaCellValue::Float(safe_log10(time_step_yr * SECYER)),
        "time_step_days" => {
            MesaCellValue::Float(time_step_yr * SECYER / SECDAY)
        }
        "log_dt_days" => {
            MesaCellValue::Float(safe_log10(time_step_yr * SECYER / SECDAY))
        }

        "star_mass" | "Mass" => MesaCellValue::Float(star_mass),
        "log_star_mass" => MesaCellValue::Float(safe_log10(star_mass)),
        "log_xmstar" => MesaCellValue::Float(safe_log10(model.xmstar)),
        "delta_mass" => {
            MesaCellValue::Float(star_mass - model.controls.initial_mass)
        }
        "star_mdot" => MesaCellValue::Float(star_mdot),
        "log_abs_mdot" | "lg_Mdot" => {
            MesaCellValue::Float(safe_log10(star_mdot.abs()))
        }

        "m_center" => MesaCellValue::Float(model.m_center / M_SUN),
        "m_center_gm" => MesaCellValue::Float(model.m_center),
        "r_center" => MesaCellValue::Float(model.r_center / R_SUN),
        "r_center_cm" => MesaCellValue::Float(model.r_center),
        "r_center_km" => MesaCellValue::Float(model.r_center * 1.0e-5),
        "L_center" => MesaCellValue::Float(model.l_center / L_SUN),
        "log_L_center_ergs_s" => {
            MesaCellValue::Float(safe_log10(model.l_center))
        }
        "log_L_center" => {
            MesaCellValue::Float(safe_log10(model.l_center / L_SUN))
        }
        "v_center" => MesaCellValue::Float(model.v_center),
        "v_center_kms" => MesaCellValue::Float(model.v_center * 1.0e-5),
        "infall_div_cs" => {
            MesaCellValue::Float(if model.v_center < 0.0 && nz > 0 {
                -model.v_center / model.csound[nz - 1].max(1.0e-99)
            } else {
                0.0
            })
        }

        "mdot_timescale" => {
            MesaCellValue::Float(star_mass / star_mdot.abs().max(1.0e-99))
        }
        "kh_div_mdot_timescales" => MesaCellValue::Float(
            ctx.kh_timescale_yr / (star_mass / star_mdot.abs().max(1.0e-99)),
        ),
        "star_gravitational_mass" => MesaCellValue::Float(
            model.m_grav.first().copied().unwrap_or(0.0) / M_SUN,
        ),
        "star_mass_grav_div_mass" => {
            MesaCellValue::Float(if nz > 0 && model.m[0] > 0.0 {
                model.m_grav[0] / model.m[0]
            } else {
                1.0
            })
        }
        "e_thermal" => MesaCellValue::Float(
            (0..nz)
                .map(|k| model.dm[k] * model.t[k] * model.cp[k])
                .sum(),
        ),
        "total_angular_momentum" => {
            MesaCellValue::Float(ctx.total_angular_momentum)
        }
        "log_total_angular_momentum" => {
            MesaCellValue::Float(safe_log10(ctx.total_angular_momentum))
        }
        "tau_factor" => MesaCellValue::Float(model.controls.tau_factor),
        "tau_surface" | "surface_optical_depth" => {
            MesaCellValue::Float(model.controls.tau_factor * model.tau_base)
        }
        "log_surf_optical_depth" => MesaCellValue::Float(safe_log10(
            model.controls.tau_factor * model.tau_base,
        )),
        "log_tau_center" => MesaCellValue::Float(if nz > 0 {
            safe_log10(model.tau[nz - 1])
        } else {
            -99.0
        }),
        "logT_max" | "log_max_T" => MesaCellValue::Float(if nz > 0 {
            model.ln_t[..nz]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                / LN10
        } else {
            0.0
        }),
        "max_T" => MesaCellValue::Float(if nz > 0 {
            flexmesa_math::exp10(
                model.ln_t[..nz]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
                    / LN10,
            )
        } else {
            0.0
        }),
        "gamma1_min" => {
            // `set_min_gamma1` in `star/private/report.f90:35-51`.
            let o = &model.controls.output;
            let mut min_g1 = 1.0e99_f64;
            for k in (0..nz).rev() {
                if model.q[k] > o.gamma1_limit_max_q {
                    break;
                }
                let vesc = if model.r[k] > 0.0 {
                    (2.0 * model.cgrav[k] * model.m[k] / model.r[k])
                        .max(0.0)
                        .sqrt()
                } else {
                    0.0
                };
                if model.controls.u_flag {
                    if model.u[k] > vesc * o.gamma1_limit_max_v_div_vesc {
                        break;
                    }
                } else if model.controls.v_flag
                    && model.v[k] > vesc * o.gamma1_limit_max_v_div_vesc
                {
                    break;
                }
                if model.gamma1[k] < min_g1 {
                    min_g1 = model.gamma1[k];
                }
            }
            MesaCellValue::Float(min_g1)
        }
        "logQ_max" => MesaCellValue::Float(if nz > 0 {
            (0..nz)
                .map(|k| {
                    model.lnd[k] / LN10 - 2.0 * model.ln_t[k] / LN10 + 12.0
                })
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            0.0
        }),
        "logQ_min" => MesaCellValue::Float(if nz > 0 {
            (0..nz)
                .map(|k| {
                    model.lnd[k] / LN10 - 2.0 * model.ln_t[k] / LN10 + 12.0
                })
                .fold(f64::INFINITY, f64::min)
        } else {
            0.0
        }),

        "surf_c12_minus_o16" => MesaCellValue::Float(
            surface_avg_x_by_name(model, "c12")
                - surface_avg_x_by_name(model, "o16"),
        ),
        "surf_num_c12_div_num_o16" => MesaCellValue::Float(
            (16.0 / 12.0) * surface_avg_x_by_name(model, "c12")
                / surface_avg_x_by_name(model, "o16").max(1.0e-99),
        ),
        "conv_mx1_top" => MesaCellValue::Float(ctx.mix.conv_mx_q[0]),
        "conv_mx1_bot" => MesaCellValue::Float(ctx.mix.conv_mx_q[1]),
        "conv_mx2_top" => MesaCellValue::Float(ctx.mix.conv_mx_q[2]),
        "conv_mx2_bot" => MesaCellValue::Float(ctx.mix.conv_mx_q[3]),
        "mx1_top" => MesaCellValue::Float(ctx.mix.mx_q[0]),
        "mx1_bot" => MesaCellValue::Float(ctx.mix.mx_q[1]),
        "mx2_top" => MesaCellValue::Float(ctx.mix.mx_q[2]),
        "mx2_bot" => MesaCellValue::Float(ctx.mix.mx_q[3]),
        "conv_mx1_top_r" => MesaCellValue::Float(ctx.mix.conv_mx_r[0]),
        "conv_mx1_bot_r" => MesaCellValue::Float(ctx.mix.conv_mx_r[1]),
        "conv_mx2_top_r" => MesaCellValue::Float(ctx.mix.conv_mx_r[2]),
        "conv_mx2_bot_r" => MesaCellValue::Float(ctx.mix.conv_mx_r[3]),
        "mx1_top_r" => MesaCellValue::Float(ctx.mix.mx_r[0]),
        "mx1_bot_r" => MesaCellValue::Float(ctx.mix.mx_r[1]),
        "mx2_top_r" => MesaCellValue::Float(ctx.mix.mx_r[2]),
        "mx2_bot_r" => MesaCellValue::Float(ctx.mix.mx_r[3]),
        "mass_conv_core" => MesaCellValue::Float(ctx.mix.mass_conv_core),
        "mass_semiconv_core" => {
            MesaCellValue::Float(ctx.mix.mass_semiconv_core)
        }

        "epsnuc_M_1" => MesaCellValue::Float(ctx.epsnuc_out[0]),
        "epsnuc_M_2" => MesaCellValue::Float(ctx.epsnuc_out[1]),
        "epsnuc_M_3" => MesaCellValue::Float(ctx.epsnuc_out[2]),
        "epsnuc_M_4" => MesaCellValue::Float(ctx.epsnuc_out[3]),
        "epsnuc_M_5" => MesaCellValue::Float(ctx.epsnuc_out[4]),
        "epsnuc_M_6" => MesaCellValue::Float(ctx.epsnuc_out[5]),
        "epsnuc_M_7" => MesaCellValue::Float(ctx.epsnuc_out[6]),
        "epsnuc_M_8" => MesaCellValue::Float(ctx.epsnuc_out[7]),

        "power_h_burn" => MesaCellValue::Float(ctx.powers.power_h_burn),
        "log_LH" => MesaCellValue::Float(safe_log10(ctx.powers.power_h_burn)),
        "power_he_burn" => MesaCellValue::Float(ctx.powers.power_he_burn),
        "log_LHe" => MesaCellValue::Float(safe_log10(ctx.powers.power_he_burn)),
        "power_photo" => MesaCellValue::Float(ctx.powers.power_photo),
        "Lnuc_photo" | "lg_Lphoto" => {
            MesaCellValue::Float(safe_log10(ctx.powers.power_photo.abs()))
        }
        "power_z_burn" => MesaCellValue::Float(ctx.powers.power_z_burn),
        "log_LZ" => MesaCellValue::Float(safe_log10(ctx.powers.power_z_burn)),
        "log_Lneu" => {
            MesaCellValue::Float(safe_log10(ctx.powers.power_neutrinos))
        }
        "lg_Lneu" => {
            MesaCellValue::Float(safe_log10(ctx.powers.power_neutrinos.abs()))
        }
        "log_Lneu_nuc" => {
            MesaCellValue::Float(safe_log10(ctx.powers.power_nuc_neutrinos))
        }
        "log_Lneu_nonnuc" => {
            MesaCellValue::Float(safe_log10(ctx.powers.power_non_nuc_neu))
        }
        "power_nuc_burn" => MesaCellValue::Float(ctx.powers.power_nuc_burn),
        "log_power_nuc_burn" | "lg_Lnuc_tot" => {
            MesaCellValue::Float(safe_log10(ctx.powers.power_nuc_burn))
        }
        "Lnuc" => MesaCellValue::Float(
            ctx.powers.power_nuc_burn - ctx.powers.power_photo,
        ),
        "log_Lnuc" => MesaCellValue::Float(safe_log10(
            ctx.powers.power_nuc_burn - ctx.powers.power_photo,
        )),
        "log_Lnuc_ergs_s" => MesaCellValue::Float(safe_log10(
            (ctx.powers.power_nuc_burn - ctx.powers.power_photo) * L_SUN,
        )),
        "log_Lnuc_sub_log_L" => MesaCellValue::Float(
            safe_log10(ctx.powers.power_nuc_burn - ctx.powers.power_photo)
                - safe_log10(l_surf_lsun),
        ),

        "Lsurf_m" => MesaCellValue::Float(
            model.m.first().copied().unwrap_or(0.0) / M_SUN,
        ),
        "luminosity" => MesaCellValue::Float(l_surf_lsun),
        "log_L" => MesaCellValue::Float(safe_log10(l_surf_lsun)),
        "luminosity_ergs_s" => MesaCellValue::Float(l_surf_lsun * L_SUN),
        "log_L_ergs_s" => MesaCellValue::Float(safe_log10(l_surf_lsun * L_SUN)),
        "log_Teff" => MesaCellValue::Float(safe_log10(model.teff)),
        "Teff" | "effective_T" => MesaCellValue::Float(model.teff),
        "radius_cm" => MesaCellValue::Float(r_surf_cm),
        "log_R_cm" => MesaCellValue::Float(safe_log10(r_surf_cm)),
        "radius" => MesaCellValue::Float(r_surf_rsun),
        "log_R" => MesaCellValue::Float(safe_log10(r_surf_rsun)),
        "gravity" => MesaCellValue::Float(g_surf),
        "log_g" => MesaCellValue::Float(safe_log10(g_surf)),

        "photosphere_cell_log_density" => {
            let k = ctx
                .phot
                .cell_k_1based
                .saturating_sub(1)
                .min(nz.saturating_sub(1));
            MesaCellValue::Float(
                model.lnd.get(k).copied().unwrap_or(0.0) / LN10,
            )
        }
        "photosphere_cell_density" => {
            let k = ctx
                .phot
                .cell_k_1based
                .saturating_sub(1)
                .min(nz.saturating_sub(1));
            MesaCellValue::Float(model.rho.get(k).copied().unwrap_or(0.0))
        }
        "photosphere_cell_log_opacity" => {
            let k = ctx
                .phot
                .cell_k_1based
                .saturating_sub(1)
                .min(nz.saturating_sub(1));
            MesaCellValue::Float(safe_log10(
                model.opacity.get(k).copied().unwrap_or(0.0),
            ))
        }
        "photosphere_cell_opacity" => {
            let k = ctx
                .phot
                .cell_k_1based
                .saturating_sub(1)
                .min(nz.saturating_sub(1));
            MesaCellValue::Float(model.opacity.get(k).copied().unwrap_or(0.0))
        }
        "photosphere_cell_log_free_e" => {
            let k = ctx
                .phot
                .cell_k_1based
                .saturating_sub(1)
                .min(nz.saturating_sub(1));
            MesaCellValue::Float(
                model.lnfree_e.get(k).copied().unwrap_or(0.0) / LN10,
            )
        }
        "photosphere_cell_free_e" => {
            let k = ctx
                .phot
                .cell_k_1based
                .saturating_sub(1)
                .min(nz.saturating_sub(1));
            MesaCellValue::Float(flexmesa_math::exp(
                model.lnfree_e.get(k).copied().unwrap_or(0.0),
            ))
        }
        "photosphere_cell_log_T" => {
            let k = ctx
                .phot
                .cell_k_1based
                .saturating_sub(1)
                .min(nz.saturating_sub(1));
            MesaCellValue::Float(
                model.ln_t.get(k).copied().unwrap_or(0.0) / LN10,
            )
        }
        "photosphere_cell_T" => {
            let k = ctx
                .phot
                .cell_k_1based
                .saturating_sub(1)
                .min(nz.saturating_sub(1));
            MesaCellValue::Float(model.t.get(k).copied().unwrap_or(0.0))
        }
        "photosphere_T" => MesaCellValue::Float(ctx.phot.t_phot),
        "photosphere_black_body_T" => {
            MesaCellValue::Float(ctx.phot.black_body_t)
        }
        "photosphere_logg" => MesaCellValue::Float(ctx.phot.logg),
        "photosphere_m" => MesaCellValue::Float(ctx.phot.m_msun),
        "photosphere_xm" => MesaCellValue::Float(star_mass - ctx.phot.m_msun),
        "photosphere_L" => MesaCellValue::Float(ctx.phot.l_lsun),
        "photosphere_r" => MesaCellValue::Float(ctx.phot.r_rsun),
        "photosphere_log_L" => {
            MesaCellValue::Float(safe_log10(ctx.phot.l_lsun))
        }
        "photosphere_log_r" => {
            MesaCellValue::Float(safe_log10(ctx.phot.r_rsun))
        }
        "photosphere_csound" => MesaCellValue::Float(ctx.phot.csound),
        "photosphere_opacity" => MesaCellValue::Float(ctx.phot.opacity),
        "photosphere_column_density" => {
            MesaCellValue::Float(ctx.phot.column_density)
        }
        "photosphere_log_column_density" => {
            MesaCellValue::Float(safe_log10(ctx.phot.column_density))
        }
        "photosphere_v_km_s" | "v_phot_km_s" => {
            MesaCellValue::Float(ctx.phot.v_cm_s * 1.0e-5)
        }
        "photosphere_v_div_cs" => {
            MesaCellValue::Float(ctx.phot.v_cm_s / ctx.phot.csound.max(1.0e-99))
        }

        "min_opacity" => MesaCellValue::Float(if nz > 0 {
            model.opacity[..nz]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
        } else {
            0.0
        }),
        "log_min_opacity" => MesaCellValue::Float(if nz > 0 {
            safe_log10(
                model.opacity[..nz]
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min),
            )
        } else {
            -99.0
        }),
        "log_cntr_dr_cm" => MesaCellValue::Float(if nz > 0 {
            safe_log10(model.r[nz - 1] - model.r_center)
        } else {
            -99.0
        }),
        "log_cntr_T" | "log_center_T" => {
            MesaCellValue::Float(center_value(model, &model.ln_t) / LN10)
        }
        "log_cntr_Rho" | "log_center_Rho" => {
            MesaCellValue::Float(center_value(model, &model.lnd) / LN10)
        }
        "log_cntr_P" | "log_center_P" => {
            MesaCellValue::Float(center_value(model, &model.ln_peos) / LN10)
        }
        "center_T" => MesaCellValue::Float(flexmesa_math::exp10(
            center_value(model, &model.ln_t) / LN10,
        )),
        "center_Rho" => MesaCellValue::Float(flexmesa_math::exp10(
            center_value(model, &model.lnd) / LN10,
        )),
        "center_P" => MesaCellValue::Float(flexmesa_math::exp10(
            center_value(model, &model.ln_peos) / LN10,
        )),

        "total_entropy" => MesaCellValue::Float(
            (0..nz).map(|k| model.dm[k] * model.entropy[k]).sum(),
        ),
        "total_internal_energy_after_adjust_mass" => {
            MesaCellValue::Float(sh.total_internal_energy_after_adjust_mass)
        }
        "total_gravitational_energy_after_adjust_mass" => MesaCellValue::Float(
            sh.total_gravitational_energy_after_adjust_mass,
        ),
        "total_radial_kinetic_energy_after_adjust_mass" => {
            MesaCellValue::Float(
                sh.total_radial_kinetic_energy_after_adjust_mass,
            )
        }
        "total_rotational_kinetic_energy_after_adjust_mass" => {
            MesaCellValue::Float(
                sh.total_rotational_kinetic_energy_after_adjust_mass,
            )
        }
        "total_turbulent_energy_after_adjust_mass" => {
            MesaCellValue::Float(sh.total_turbulent_energy_after_adjust_mass)
        }
        "total_energy_after_adjust_mass" => {
            MesaCellValue::Float(sh.total_energy_after_adjust_mass)
        }
        "total_internal_energy" | "tot_IE" => {
            MesaCellValue::Float(ctx.energies_erg.0)
        }
        "total_gravitational_energy" | "tot_PE" => {
            MesaCellValue::Float(ctx.energies_erg.1)
        }
        "total_radial_kinetic_energy" | "tot_KE" => {
            MesaCellValue::Float(ctx.energies_erg.2)
        }
        "total_rotational_kinetic_energy" => {
            MesaCellValue::Float(ctx.energies_erg.3)
        }
        "total_turbulent_energy" | "tot_Et" => {
            MesaCellValue::Float(ctx.energies_erg.4)
        }
        "total_energy" | "tot_E" => MesaCellValue::Float(e_end),
        "total_energy_foe" => MesaCellValue::Float(e_end * 1.0e-51),
        "log_total_internal_energy" | "log_tot_IE" => {
            MesaCellValue::Float(safe_log10(ctx.energies_erg.0))
        }
        "log_total_gravitational_energy" | "log_tot_PE" => {
            MesaCellValue::Float(safe_log10(ctx.energies_erg.1.abs()))
        }
        "log_total_radial_kinetic_energy" | "log_tot_KE" => {
            MesaCellValue::Float(safe_log10(ctx.energies_erg.2))
        }
        "log_total_rotational_kinetic_energy" => {
            MesaCellValue::Float(safe_log10(ctx.energies_erg.3))
        }
        "log_total_turbulent_energy" | "log_tot_Et" => {
            MesaCellValue::Float(safe_log10(ctx.energies_erg.4))
        }
        "log_total_energy" | "log_tot_E" => {
            MesaCellValue::Float(safe_log10(e_end.abs()))
        }

        "total_eps_grav" => MesaCellValue::Float(sh.total_eps_grav),
        "work_outward_at_surface" => {
            MesaCellValue::Float(sh.work_outward_at_surface)
        }
        "work_inward_at_center" => {
            MesaCellValue::Float(sh.work_inward_at_center)
        }
        "total_nuclear_heating" => {
            MesaCellValue::Float(sh.total_nuclear_heating)
        }
        "total_non_nuc_neu_cooling" => {
            MesaCellValue::Float(sh.total_non_nuc_neu_cooling)
        }
        "total_irradiation_heating" => {
            MesaCellValue::Float(sh.total_irradiation_heating)
        }
        "total_WD_sedimentation_heating" => MesaCellValue::Float(
            if model.controls.do_element_diffusion {
                sh.total_wd_sedimentation_heating
            } else {
                0.0
            },
        ),
        "total_extra_heating" => MesaCellValue::Float(sh.total_extra_heating),
        "total_energy_sources_and_sinks" => {
            MesaCellValue::Float(sh.total_energy_sources_and_sinks)
        }
        "tot_E_equ_err" => MesaCellValue::Float(sh.tot_e_equ_err),

        "error_in_energy_conservation" | "tot_E_err" => {
            MesaCellValue::Float(sh.error_in_energy_conservation)
        }
        "rel_error_in_energy_conservation" | "rel_E_err" => {
            MesaCellValue::Float(if e_end == 0.0 {
                0.0
            } else {
                sh.error_in_energy_conservation / e_end.abs()
            })
        }
        "abs_rel_E_err" => MesaCellValue::Float(if e_end == 0.0 {
            0.0
        } else {
            (sh.error_in_energy_conservation / e_end).abs()
        }),
        "log_rel_error_in_energy_conservation" | "log_rel_E_err" => {
            MesaCellValue::Float(if e_end == 0.0 {
                0.0
            } else {
                safe_log10((sh.error_in_energy_conservation / e_end).abs())
            })
        }
        "cumulative_energy_error" => {
            MesaCellValue::Float(model.aux.cumulative_energy_error)
        }
        "rel_cumulative_energy_error" => {
            MesaCellValue::Float(if e_end == 0.0 {
                0.0
            } else {
                model.aux.cumulative_energy_error / e_end.abs()
            })
        }
        "log_rel_cumulative_energy_error" | "log_rel_run_E_err" => {
            MesaCellValue::Float(if e_end == 0.0 {
                0.0
            } else {
                safe_log10((model.aux.cumulative_energy_error / e_end).abs())
            })
        }
        "rel_run_E_err" => MesaCellValue::Float(if e_end == 0.0 {
            0.0
        } else {
            model.aux.cumulative_energy_error / e_end
        }),

        "avg_abs_v_div_cs" => MesaCellValue::Float(if has_vel && nz > 0 {
            (0..nz)
                .map(|k| vel_slice[k].abs() / model.csound[k].max(1.0e-99))
                .sum::<f64>()
                / (nz as f64)
        } else {
            0.0
        }),
        "log_avg_abs_v_div_cs" => MesaCellValue::Float(if has_vel && nz > 0 {
            safe_log10(
                (0..nz)
                    .map(|k| vel_slice[k].abs() / model.csound[k].max(1.0e-99))
                    .sum::<f64>()
                    / (nz as f64),
            )
        } else {
            0.0
        }),
        "max_abs_v_div_cs" => MesaCellValue::Float(if has_vel && nz > 0 {
            (0..nz)
                .map(|k| vel_slice[k].abs() / model.csound[k].max(1.0e-99))
                .fold(0.0, f64::max)
        } else {
            0.0
        }),
        "log_max_abs_v_div_cs" => MesaCellValue::Float(if has_vel && nz > 0 {
            safe_log10(
                (0..nz)
                    .map(|k| vel_slice[k].abs() / model.csound[k].max(1.0e-99))
                    .fold(0.0, f64::max),
            )
        } else {
            0.0
        }),
        "avg_abs_v" => MesaCellValue::Float(if has_vel && nz > 0 {
            vel_slice[..nz].iter().map(|v| v.abs()).sum::<f64>() / (nz as f64)
        } else {
            0.0
        }),
        "log_avg_abs_v" => MesaCellValue::Float(if has_vel && nz > 0 {
            safe_log10(
                vel_slice[..nz].iter().map(|v| v.abs()).sum::<f64>()
                    / (nz as f64),
            )
        } else {
            0.0
        }),
        "max_abs_v" => MesaCellValue::Float(if has_vel && nz > 0 {
            vel_slice[..nz].iter().map(|v| v.abs()).fold(0.0, f64::max)
        } else {
            0.0
        }),
        "log_max_abs_v" => MesaCellValue::Float(if has_vel && nz > 0 {
            safe_log10(
                vel_slice[..nz].iter().map(|v| v.abs()).fold(0.0, f64::max),
            )
        } else {
            0.0
        }),

        "u_surf_km_s" => {
            MesaCellValue::Float(if model.controls.u_flag && nz > 0 {
                model.aux.u_face_ad[0].val * 1.0e-5
            } else {
                0.0
            })
        }
        "u_surf" => MesaCellValue::Float(if model.controls.u_flag && nz > 0 {
            model.aux.u_face_ad[0].val
        } else {
            0.0
        }),
        "u_div_csound_max" => {
            MesaCellValue::Float(if model.controls.u_flag && nz > 0 {
                (0..nz)
                    .map(|k| model.u[k].abs() / model.csound[k].max(1.0e-99))
                    .fold(0.0, f64::max)
            } else {
                0.0
            })
        }
        "u_div_csound_surf" => {
            MesaCellValue::Float(if model.controls.u_flag && nz > 0 {
                model.aux.u_face_ad[0].val / csound_surf.max(1.0e-99)
            } else {
                0.0
            })
        }
        "surf_escape_v" => MesaCellValue::Float(if nz > 0 && r_surf_cm > 0.0 {
            (2.0 * model.cgrav[0] * model.m[0] / r_surf_cm)
                .max(0.0)
                .sqrt()
        } else {
            0.0
        }),
        "v_surf_div_escape_v" | "v_div_vesc" => {
            let vesc = if nz > 0 && r_surf_cm > 0.0 {
                (2.0 * model.cgrav[0] * model.m[0] / r_surf_cm)
                    .max(0.0)
                    .sqrt()
            } else {
                0.0
            };
            MesaCellValue::Float(if vesc > 0.0 { v_surf / vesc } else { 0.0 })
        }
        "v_surf_km_s" => MesaCellValue::Float(v_surf * 1.0e-5),
        "v_surf" => MesaCellValue::Float(v_surf),
        // Deferred upstream bug (#265): `star/private/history.f90:1956-1957`
        // computes `v_surf / (R_phot_Rsun / kh_timescale_yr)` in mixed units
        // (`cm/s * yr / Rsun`); reproduced verbatim until #169 decides.
        "v_surf_div_v_kh" => MesaCellValue::Float(if ctx.phot.r_rsun > 0.0 {
            v_surf / (ctx.phot.r_rsun / ctx.kh_timescale_yr.max(1.0e-99))
        } else {
            0.0
        }),
        "v_div_csound_max" => MesaCellValue::Float(if has_vel && nz > 0 {
            (0..nz)
                .map(|k| {
                    vel_slice[k].abs() / csound_face_at(model, k).max(1.0e-99)
                })
                .fold(0.0, f64::max)
        } else {
            0.0
        }),
        "v_div_csound_surf" => MesaCellValue::Float(if csound_surf > 0.0 {
            v_surf / csound_surf
        } else {
            0.0
        }),
        "v_div_cs" => {
            MesaCellValue::Float(if nz > 0 && model.csound[0] > 0.0 {
                v_surf / model.csound[0]
            } else {
                0.0
            })
        }

        "log_Ledd" => {
            let ledd = eval_ledd_erg_s(model);
            MesaCellValue::Float(if ledd > 0.0 {
                safe_log10(ledd / L_SUN)
            } else {
                0.0
            })
        }
        "log_L_div_Ledd" => {
            let ledd = eval_ledd_erg_s(model);
            MesaCellValue::Float(if ledd > 0.0 {
                safe_log10(l_surf_lsun * L_SUN / ledd)
            } else {
                0.0
            })
        }
        "lum_div_Ledd" => {
            let ledd = eval_ledd_erg_s(model);
            MesaCellValue::Float(if ledd > 0.0 {
                l_surf_lsun * L_SUN / ledd
            } else {
                0.0
            })
        }
        "max_L_rad_div_Ledd" => MesaCellValue::Float(
            (0..nz)
                .map(|k| get_lrad_div_ledd(model, k))
                .fold(0.0, f64::max),
        ),

        "i_rot_total" => {
            MesaCellValue::Float(if model.controls.rotation_flag {
                (0..nz).map(|k| model.dm_bar[k] * model.aux.i_rot[k]).sum()
            } else {
                0.0
            })
        }
        "surf_avg_j_rot" => MesaCellValue::Float(ctx.surf_rot.j_rot),
        "surf_avg_omega" => MesaCellValue::Float(ctx.surf_rot.omega),
        "surf_avg_omega_crit" => MesaCellValue::Float(ctx.surf_rot.omega_crit),
        "surf_avg_omega_div_omega_crit" => {
            MesaCellValue::Float(ctx.surf_rot.w_div_w_crit)
        }
        "surf_avg_v_rot" => {
            MesaCellValue::Float(ctx.surf_rot.v_rot_cm_s * 1.0e-5)
        }
        "surf_avg_v_crit" => {
            MesaCellValue::Float(ctx.surf_rot.v_crit_cm_s * 1.0e-5)
        }
        "surf_avg_v_div_v_crit" => {
            MesaCellValue::Float(ctx.surf_rot.v_div_v_crit)
        }
        "surf_avg_Lrad_div_Ledd" => {
            MesaCellValue::Float(ctx.surf_rot.lrad_div_ledd)
        }
        "surf_avg_opacity" => MesaCellValue::Float(ctx.surf_rot.opacity),
        "surf_avg_logT" => MesaCellValue::Float(ctx.surf_rot.log_t),
        "surf_avg_logRho" => MesaCellValue::Float(ctx.surf_rot.log_rho),
        "rotational_mdot_boost" => {
            MesaCellValue::Float(if model.controls.rotation_flag {
                sh.rotational_mdot_boost_old
            } else {
                0.0
            })
        }
        "log_rotational_mdot_boost" => {
            MesaCellValue::Float(safe_log10(if model.controls.rotation_flag {
                sh.rotational_mdot_boost_old
            } else {
                0.0
            }))
        }
        "surf_r_equatorial_div_r_polar" => {
            MesaCellValue::Float(if model.controls.rotation_flag && nz > 0 {
                let (re, rp) = equatorial_and_polar_r_cm(model, 0);
                re / rp.max(1.0e-99)
            } else {
                1.0
            })
        }
        "surf_r_equatorial_div_r" => {
            MesaCellValue::Float(if model.controls.rotation_flag && nz > 0 {
                let (re, _) = equatorial_and_polar_r_cm(model, 0);
                re / r_surf_cm.max(1.0e-99)
            } else {
                1.0
            })
        }
        "surf_r_polar_div_r" => {
            MesaCellValue::Float(if model.controls.rotation_flag && nz > 0 {
                let (_, rp) = equatorial_and_polar_r_cm(model, 0);
                rp / r_surf_cm.max(1.0e-99)
            } else {
                1.0
            })
        }
        // `star/private/history.f90:2043-2045` (km/s).
        "v_wind_Km_per_s" => {
            let denom = FOUR_PI * ctx.phot.r_rsun * R_SUN * model.tau_base;
            MesaCellValue::Float(if nz > 0 && denom > 0.0 {
                1.0e-5 * model.opacity[0] * (-model.mstar_dot).max(0.0) / denom
            } else {
                0.0
            })
        }
        // `star/private/history.f90:2901-2928` (pole `i = 0`, equator `pi / 2`).
        "grav_dark_L_polar"
        | "grav_dark_Teff_polar"
        | "grav_dark_L_equatorial"
        | "grav_dark_Teff_equatorial" => {
            MesaCellValue::Float(if model.controls.rotation_flag {
                let w_div_w_kep = eval_surface_w_div_w_kep(model);
                match name {
                    "grav_dark_L_polar" => {
                        gravity_darkening_l_coeff(w_div_w_kep, 0.0)
                            * l_surf_lsun
                    }
                    "grav_dark_Teff_polar" => {
                        gravity_darkening_teff_coeff(w_div_w_kep, 0.0)
                            * model.teff
                    }
                    "grav_dark_L_equatorial" => {
                        gravity_darkening_l_coeff(w_div_w_kep, 0.5 * PI)
                            * l_surf_lsun
                    }
                    _ => {
                        gravity_darkening_teff_coeff(w_div_w_kep, 0.5 * PI)
                            * model.teff
                    }
                }
            } else {
                0.0
            })
        }

        "min_Pgas_div_P" => MesaCellValue::Float(if nz > 0 {
            (0..nz)
                .map(|k| model.pgas[k] / model.peos[k].max(1.0e-99))
                .fold(1.0, f64::min)
        } else {
            1.0
        }),
        "center_degeneracy" => {
            MesaCellValue::Float(center_value(model, &model.eta))
        }
        "center_eps_nuc" => {
            MesaCellValue::Float(center_value(model, &model.eps_nuc))
        }
        "log_center_eps_nuc" => MesaCellValue::Float(safe_log10(center_value(
            model,
            &model.eps_nuc,
        ))),
        "d_center_eps_nuc_dlnT" => {
            MesaCellValue::Float(center_value(model, &model.d_epsnuc_dlnt))
        }
        "d_center_eps_nuc_dlnd" => {
            MesaCellValue::Float(center_value(model, &model.d_epsnuc_dlnd))
        }
        "center_eps_grav" => MesaCellValue::Float(center_value_by(
            model,
            model.eps_grav_ad.len(),
            |k| model.eps_grav_ad[k].val,
        )),
        "center_non_nuc_neu" => {
            MesaCellValue::Float(center_value(model, &model.non_nuc_neu))
        }
        "center_gamma" => MesaCellValue::Float(center_value(model, &model.gam)),
        "center_zbar" => MesaCellValue::Float(center_value(model, &model.zbar)),
        "center_abar" => MesaCellValue::Float(center_value(model, &model.abar)),
        "center_mu" => MesaCellValue::Float(center_value(model, &model.mu)),
        "center_ye" => MesaCellValue::Float(center_value(model, &model.ye)),
        "center_entropy" => MesaCellValue::Float(ctx.center_entropy(model)),
        "max_entropy" => MesaCellValue::Float(if nz > 0 {
            flexmesa_math::exp(
                model.ln_s[..nz]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max),
            ) * flexmesa_const::AMU
                / flexmesa_const::KERG
        } else {
            0.0
        }),
        "center_omega" => {
            MesaCellValue::Float(if model.controls.rotation_flag {
                center_value(model, &model.omega)
            } else {
                0.0
            })
        }
        "center_omega_div_omega_crit" => {
            MesaCellValue::Float(if model.controls.rotation_flag && nz > 0 {
                center_value_by(model, nz, |k| {
                    let wc = eval_upstream_omega_crit(model, k);
                    if wc > 0.0 { model.omega[k] / wc } else { 0.0 }
                })
                .min(1.0)
            } else {
                0.0
            })
        }

        "compactness" | "compactness_parameter" => {
            let target_m = 2.5 * M_SUN;
            if nz >= 2 && model.m[0] > target_m {
                let mut k0 = 0_usize;
                for k in (0..(nz - 1)).rev() {
                    k0 = k;
                    if model.m[k] > target_m {
                        break;
                    }
                }
                let r = model.r[k0 + 1]
                    + (model.r[k0] - model.r[k0 + 1])
                        * (target_m - model.m[k0 + 1])
                        / model.dm[k0].max(1.0e-99);
                MesaCellValue::Float(2.5 / (r / 1.0e8).max(1.0e-99))
            } else {
                MesaCellValue::Float(0.0)
            }
        }
        "m4" => {
            if nz >= 2 && model.entropy[0] > 4.0 {
                let mut k0 = 0_usize;
                for k in (0..(nz - 1)).rev() {
                    k0 = k;
                    if model.entropy[k] > 4.0 {
                        break;
                    }
                }
                MesaCellValue::Float(model.m[k0] / M_SUN)
            } else {
                MesaCellValue::Float(0.0)
            }
        }
        "mu4" => {
            let deltam = 0.3 * M_SUN;
            if nz >= 2 && model.entropy[0] > 4.0 {
                let mut k0 = 0_usize;
                for k in (0..(nz - 1)).rev() {
                    k0 = k;
                    if model.entropy[k] > 4.0 {
                        break;
                    }
                }
                let mut k2 = 0_usize;
                for k in (0..(nz - 1)).rev() {
                    k2 = k;
                    if model.m[k] > model.m[k0] + deltam {
                        break;
                    }
                }
                let dr = (model.r[k2] - model.r[k0]) / 1.0e8;
                MesaCellValue::Float(if dr == 0.0 { 0.0 } else { 0.3 / dr })
            } else {
                MesaCellValue::Float(0.0)
            }
        }
        "max_infall_speed" => MesaCellValue::Float(if has_vel && nz > 0 {
            -vel_slice[..nz]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
                * 1.0e-5
        } else {
            0.0
        }),

        "h_rich_layer_mass" | "envelope_mass" => {
            MesaCellValue::Float(star_mass - ctx.cores.he_core_mass)
        }
        "he_rich_layer_mass" => MesaCellValue::Float(
            (ctx.cores.he_core_mass - ctx.cores.co_core_mass).max(0.0),
        ),
        "co_rich_layer_mass" => MesaCellValue::Float(
            (ctx.cores.co_core_mass - ctx.cores.he_core_mass).max(0.0),
        ),
        "envelope_fraction_left" => MesaCellValue::Float(
            if model.controls.initial_mass > ctx.cores.he_core_mass {
                (star_mass - ctx.cores.he_core_mass)
                    / (model.controls.initial_mass - ctx.cores.he_core_mass)
            } else {
                1.0
            },
        ),
        "he_core_mass" | "He_core" => {
            MesaCellValue::Float(ctx.cores.he_core_mass)
        }
        "co_core_mass" | "CO_core" => {
            MesaCellValue::Float(ctx.cores.co_core_mass)
        }
        "one_core_mass" => MesaCellValue::Float(ctx.cores.one_core_mass),
        "fe_core_mass" | "Fe_core" => {
            MesaCellValue::Float(ctx.cores.fe_core_mass)
        }
        "neutron_rich_core_mass" => {
            MesaCellValue::Float(ctx.cores.neutron_rich_core_mass)
        }
        "he_core_radius"
        | "he_core_lgT"
        | "he_core_lgRho"
        | "he_core_L"
        | "he_core_v"
        | "he_core_k"
        | "co_core_radius"
        | "co_core_lgT"
        | "co_core_lgRho"
        | "co_core_L"
        | "co_core_v"
        | "co_core_k"
        | "one_core_radius"
        | "one_core_lgT"
        | "one_core_lgRho"
        | "one_core_L"
        | "one_core_v"
        | "one_core_k"
        | "fe_core_radius"
        | "fe_core_lgT"
        | "fe_core_lgRho"
        | "fe_core_L"
        | "fe_core_v"
        | "fe_core_k"
        | "neutron_rich_core_radius"
        | "neutron_rich_core_lgT"
        | "neutron_rich_core_lgRho"
        | "neutron_rich_core_L"
        | "neutron_rich_core_v"
        | "neutron_rich_core_k" => {
            let (bdy, suffix) =
                if let Some(s) = name.strip_prefix("neutron_rich_core_") {
                    (&ctx.cores.neutron_rich_bdy, s)
                } else if let Some(s) = name.strip_prefix("he_core_") {
                    (&ctx.cores.he_bdy, s)
                } else if let Some(s) = name.strip_prefix("co_core_") {
                    (&ctx.cores.co_bdy, s)
                } else if let Some(s) = name.strip_prefix("one_core_") {
                    (&ctx.cores.one_bdy, s)
                } else {
                    (
                        &ctx.cores.fe_bdy,
                        name.strip_prefix("fe_core_").unwrap_or("k"),
                    )
                };
            eval_core_bdy_prop(bdy, suffix)
        }

        "dynamic_timescale" => MesaCellValue::Float(ctx.dynamic_timescale_s),
        "kh_timescale" => MesaCellValue::Float(ctx.kh_timescale_yr),
        "nuc_timescale" => MesaCellValue::Float(ctx.nuc_timescale_yr),
        "eps_grav_integral" => MesaCellValue::Float(
            (0..nz)
                .map(|k| model.dm[k] * model.eps_grav_ad[k].val)
                .sum::<f64>()
                / L_SUN,
        ),
        "log_abs_Lgrav" => MesaCellValue::Float(safe_log10(
            ((0..nz)
                .map(|k| model.dm[k] * model.eps_grav_ad[k].val)
                .sum::<f64>()
                / L_SUN)
                .abs(),
        )),
        "extra_L" => MesaCellValue::Float(
            (0..nz)
                .map(|k| model.dm[k] * model.aux.extra_heat[k])
                .sum::<f64>()
                / L_SUN,
        ),
        "log_extra_L" => MesaCellValue::Float(safe_log10(
            (0..nz)
                .map(|k| model.dm[k] * model.aux.extra_heat[k])
                .sum::<f64>()
                / L_SUN,
        )),
        "mass_loc_of_max_eps_nuc" => {
            if nz == 0 {
                MesaCellValue::Float(0.0)
            } else {
                let k = (0..nz)
                    .max_by(|&a, &b| {
                        model.eps_nuc[a].total_cmp(&model.eps_nuc[b])
                    })
                    .unwrap_or(0);
                MesaCellValue::Float((model.m[k] - 0.5 * model.dm[k]) / M_SUN)
            }
        }
        "mass_ext_to_max_eps_nuc" => {
            if nz == 0 {
                MesaCellValue::Float(0.0)
            } else {
                let k = (0..nz)
                    .max_by(|&a, &b| {
                        model.eps_nuc[a].total_cmp(&model.eps_nuc[b])
                    })
                    .unwrap_or(0);
                MesaCellValue::Float(
                    (1.0 - model.q[k] + 0.5 * model.dq[k]) * model.xmstar
                        / M_SUN,
                )
            }
        }
        "total_ni_co_56" => {
            let co = resolve_iso_slot(model, "co56").ok().and_then(|(_, s)| s);
            let ni = resolve_iso_slot(model, "ni56").ok().and_then(|(_, s)| s);
            if let (Some(jc), Some(jn)) = (co, ni) {
                let tot: f64 = (0..nz)
                    .map(|k| {
                        model.dm[k]
                            * (model.cell_xa(k)[jc] + model.cell_xa(k)[jn])
                    })
                    .sum();
                MesaCellValue::Float(tot / M_SUN)
            } else {
                MesaCellValue::Float(0.0)
            }
        }

        "log_surf_cell_opacity" => MesaCellValue::Float(safe_log10(
            model.opacity.first().copied().unwrap_or(0.0),
        )),
        "log_surf_cell_density" | "lg_Dsurf" => MesaCellValue::Float(
            model.lnd.first().copied().unwrap_or(0.0) / LN10,
        ),
        "surface_cell_temperature" => {
            MesaCellValue::Float(model.t.first().copied().unwrap_or(0.0))
        }
        "log_surf_cell_temperature" => MesaCellValue::Float(
            model.ln_t.first().copied().unwrap_or(0.0) / LN10,
        ),
        "surface_cell_entropy" => {
            MesaCellValue::Float(model.entropy.first().copied().unwrap_or(0.0))
        }
        "log_surf_cell_P" | "log_surf_cell_pressure" => MesaCellValue::Float(
            model.ln_peos.first().copied().unwrap_or(0.0) / LN10,
        ),
        "log_surf_cell_z" => {
            if nz == 0 {
                MesaCellValue::Float(-99.0)
            } else {
                let mut h_he = 0.0;
                for iso in ["h1", "h2", "he3", "he4"] {
                    if let Ok((_, Some(j))) = resolve_iso_slot(model, iso) {
                        h_he += model.cell_xa(0)[j];
                    }
                }
                MesaCellValue::Float(safe_log10(1.0 - h_he))
            }
        }

        "min_dr_div_cs" => MesaCellValue::Float(eval_min_dr_div_cs(model).0),
        "log_min_dr_div_cs" => {
            MesaCellValue::Float(safe_log10(eval_min_dr_div_cs(model).0))
        }
        "dt_div_min_dr_div_cs" => {
            let (mdr, mk) = eval_min_dr_div_cs(model);
            MesaCellValue::Float(if mk <= 0 { 1.0e99 } else { model.dt / mdr })
        }
        "log_dt_div_min_dr_div_cs" => {
            let (mdr, mk) = eval_min_dr_div_cs(model);
            MesaCellValue::Float(if mk <= 0 {
                1.0e99
            } else {
                safe_log10(model.dt / mdr)
            })
        }
        "dt_cell_collapse" => {
            MesaCellValue::Float(eval_min_cell_collapse_time(model))
        }
        "dt_div_dt_cell_collapse" => {
            MesaCellValue::Float(model.dt / eval_min_cell_collapse_time(model))
        }

        "cz_bot_mass" => {
            MesaCellValue::Float(match ctx.mix.largest_conv_top_bot {
                Some((_, bot)) if bot + 1 == nz => model.m_center / M_SUN,
                Some((_, bot)) => model.m[bot] / M_SUN,
                None => 0.0,
            })
        }
        // `star/private/history.f90:2545-2553` (solar radii).
        "cz_bot_radius" => {
            MesaCellValue::Float(match ctx.mix.largest_conv_top_bot {
                Some((_, bot)) if bot + 1 == nz => model.r_center / R_SUN,
                Some((_, bot)) => model.r[bot] / R_SUN,
                None => 0.0,
            })
        }
        // `star/private/history.f90:2459-2465`.
        "cz_log_xmsun" => {
            MesaCellValue::Float(match ctx.mix.largest_conv_top_bot {
                Some((_, bot)) => {
                    let sum_dq: f64 =
                        model.dq.get(..bot).map_or(0.0, |s| s.iter().sum());
                    safe_log10(model.xmstar * sum_dq / M_SUN)
                }
                None => -99.0,
            })
        }
        // `star/private/history.f90:2535-2539` (cm).
        "cz_scale_height" => {
            MesaCellValue::Float(match ctx.mix.largest_conv_top_bot {
                Some((_, bot)) => {
                    model.scale_height.get(bot).copied().unwrap_or(0.0)
                }
                None => 0.0,
            })
        }
        // `star/private/history.f90:2671-2675` (solar radii).
        "cz_top_radius" => {
            MesaCellValue::Float(match ctx.mix.largest_conv_top_bot {
                Some((top, _)) => {
                    model.r.get(top).copied().unwrap_or(0.0) / R_SUN
                }
                None => 0.0,
            })
        }
        // `star/private/history.f90:2661-2665` (cm).
        "cz_top_scale_height" => {
            MesaCellValue::Float(match ctx.mix.largest_conv_top_bot {
                Some((top, _)) => {
                    model.scale_height.get(top).copied().unwrap_or(0.0)
                }
                None => 0.0,
            })
        }
        "cz_mass" => MesaCellValue::Float(match ctx.mix.largest_conv_top_bot {
            Some((_, bot)) => model.m[bot] / M_SUN,
            None => 0.0,
        }),
        "cz_top_mass" => {
            MesaCellValue::Float(match ctx.mix.largest_conv_top_bot {
                Some((top, _)) => model.m[top] / M_SUN,
                None => 0.0,
            })
        }
        "cz_zone" => MesaCellValue::Int(
            ctx.mix
                .largest_conv_top_bot
                .map_or(0, |(_, bot)| (bot + 1) as i64),
        ),
        "cz_top_zone" => MesaCellValue::Int(
            ctx.mix
                .largest_conv_top_bot
                .map_or(0, |(top, _)| (top + 1) as i64),
        ),

        "max_gradT_div_grada" => {
            let mut v = 0.0;
            for k in 1..nz {
                let ga = model.interp_val_to_pt(&model.grada, k);
                if ga != 0.0 && model.gradt[k] / ga > v {
                    v = model.gradt[k] / ga;
                }
            }
            MesaCellValue::Float(v)
        }
        "max_gradT_sub_grada" => {
            let mut v = 0.0;
            for k in 1..nz {
                let ga = model.interp_val_to_pt(&model.grada, k);
                if model.gradt[k] - ga > v {
                    v = model.gradt[k] - ga;
                }
            }
            MesaCellValue::Float(v)
        }
        "min_log_mlt_Gamma" => {
            let mut v = 1.0e99;
            for k in 1..nz {
                if model.mlt_gamma[k] > 0.0 && model.mlt_gamma[k] < v {
                    v = model.mlt_gamma[k];
                }
            }
            MesaCellValue::Float(safe_log10(v))
        }
        "max_conv_vel_div_csound" => {
            // `h_max_conv_vel_div_csound` in `star/private/history.f90:2726-2731`.
            let max_q = model.controls.output.max_conv_vel_div_csound_maxq;
            let mut v = 0.0;
            for k in 1..nz {
                if model.q[k] > max_q || model.csound[k] == 0.0 {
                    continue;
                }
                if model.conv_vel[k] / model.csound[k] > v {
                    v = model.conv_vel[k]
                        / csound_face_at(model, k).max(1.0e-99);
                }
            }
            MesaCellValue::Float(v)
        }
        "min_t_eddy" => {
            let mut v = 1.0e99;
            for k in 1..nz {
                if model.conv_vel[k] > 0.0 {
                    let te = model.scale_height[k] / model.conv_vel[k];
                    if te < v {
                        v = te;
                    }
                }
            }
            MesaCellValue::Float(v)
        }

        "delta_nu" | "delta_Pg" | "log_delta_Pg" | "nu_max"
        | "acoustic_radius" | "acoustic_cutoff" => {
            let (ac_r, dnu, nmax, dpg) =
                *astero_cache.get_or_insert_with(|| {
                    eval_astero_report_scalars(model, &ctx.phot)
                });
            match name {
                "acoustic_radius" => MesaCellValue::Float(ac_r),
                "delta_nu" => MesaCellValue::Float(dnu),
                "nu_max" => MesaCellValue::Float(nmax),
                "delta_Pg" => MesaCellValue::Float(dpg),
                "log_delta_Pg" => {
                    MesaCellValue::Float(if model.controls.calculate_brunt_n2 {
                        safe_log10(dpg)
                    } else {
                        0.0
                    })
                }
                _ => MesaCellValue::Float(if nz > 0 && model.peos[0] > 0.0 {
                    (0.25e6 / flexmesa_const::PI)
                        * g_surf
                        * (model.gamma1[0] * model.rho[0] / model.peos[0])
                            .max(0.0)
                            .sqrt()
                } else {
                    0.0
                }),
            }
        }

        "rsp_DeltaR" => MesaCellValue::Float(if model.controls.rsp_flag {
            model.aux.rsp.delta_r
        } else {
            0.0
        }),
        "rsp_DeltaMag" => MesaCellValue::Float(if model.controls.rsp_flag {
            model.aux.rsp.delta_mag
        } else {
            0.0
        }),
        "rsp_GREKM" => MesaCellValue::Float(if model.controls.rsp_flag {
            model.aux.rsp.grekm
        } else {
            0.0
        }),
        "rsp_phase" => MesaCellValue::Float(
            if model.controls.rsp_flag && model.aux.rsp.period > 0.0 {
                (model.star_age * SECYER - model.aux.rsp.tt1)
                    / model.aux.rsp.period
            } else {
                0.0
            },
        ),
        "rsp_period_in_days" => {
            MesaCellValue::Float(if model.controls.rsp_flag {
                model.aux.rsp.period / SECDAY
            } else {
                0.0
            })
        }
        "rsp_num_periods" => MesaCellValue::Int(if model.controls.rsp_flag {
            model.aux.rsp.num_periods as i64
        } else {
            0
        }),

        "H_rich" => MesaCellValue::Float(
            star_mass - ctx.cores.he_core_mass.max(ctx.cores.co_core_mass),
        ),
        "H_cntr" => MesaCellValue::Float(center_avg_x_by_name(model, "h1")),
        "He_cntr" => MesaCellValue::Float(
            center_avg_x_by_name(model, "he3")
                + center_avg_x_by_name(model, "he4"),
        ),
        "C_cntr" => MesaCellValue::Float(center_avg_x_by_name(model, "c12")),
        "N_cntr" => MesaCellValue::Float(center_avg_x_by_name(model, "n14")),
        "O_cntr" => MesaCellValue::Float(center_avg_x_by_name(model, "o16")),
        "Ne_cntr" => MesaCellValue::Float(center_avg_x_by_name(model, "ne20")),
        "Si_cntr" => MesaCellValue::Float(center_avg_x_by_name(model, "si28")),
        "eta_cntr" => {
            MesaCellValue::Float(if nz > 0 { model.eta[nz - 1] } else { 0.0 })
        }
        "gam_cntr" => {
            MesaCellValue::Float(if nz > 0 { model.gam[nz - 1] } else { 0.0 })
        }
        // `star/private/evolve.f90:1173-1174` & `history.f90:1875-1880`.
        "virial_thm_P_avg" | "virial_thm_rel_err" => {
            let virial = if sh.virial_thm_p_avg == 0.0 {
                (0..nz)
                    .map(|k| {
                        3.0 * model.dm[k] * model.peos[k]
                            / model.rho[k].max(1.0e-99)
                    })
                    .sum()
            } else {
                sh.virial_thm_p_avg
            };
            MesaCellValue::Float(match name {
                "virial_thm_P_avg" => virial,
                _ => {
                    if virial == 0.0 {
                        0.0
                    } else {
                        (virial + ctx.energies_erg.1) / virial
                    }
                }
            })
        }
        // `set_conv_time_scales` in `star/private/star_utils.f90:3706-3751`
        // and `history.f90:2276-2287`.
        "max_tau_conv"
        | "min_tau_conv"
        | "log_max_tau_conv"
        | "log_min_tau_conv"
        | "dt_div_max_tau_conv"
        | "dt_div_min_tau_conv" => {
            let (min_tc, max_tc) = eval_conv_time_scales(model);
            match name {
                "max_tau_conv" => MesaCellValue::Float(max_tc),
                "min_tau_conv" => MesaCellValue::Float(min_tc),
                "log_max_tau_conv" => MesaCellValue::Float(safe_log10(max_tc)),
                "log_min_tau_conv" => MesaCellValue::Float(safe_log10(min_tc)),
                "dt_div_max_tau_conv" => {
                    MesaCellValue::Float(if max_tc > 0.0 {
                        model.dt / max_tc
                    } else {
                        0.0
                    })
                }
                _ => MesaCellValue::Float(if min_tc > 0.0 {
                    model.dt / min_tc
                } else {
                    0.0
                }),
            }
        }
        // `star/private/report.f90:378-441` and `history.f90:2130-2133` (km/s).
        "fe_core_infall" | "non_fe_core_infall" => {
            let (fe_infall_cms, non_fe_infall_cms) =
                eval_core_infall_speeds_cms(model, ctx, has_vel, vel_slice);
            MesaCellValue::Float(if name == "fe_core_infall" {
                fe_infall_cms * 1.0e-5
            } else {
                non_fe_infall_cms * 1.0e-5
            })
        }
        // `star/private/history.f90:1991-1996`.
        "gradT_excess_alpha" => {
            MesaCellValue::Float(model.aux.mixing.gradt_excess_alpha)
        }
        "gradT_excess_min_beta" => {
            MesaCellValue::Float(model.aux.mixing.gradt_excess_min_beta)
        }
        "gradT_excess_max_lambda" => {
            MesaCellValue::Float(model.aux.mixing.gradt_excess_max_lambda)
        }
        // `star/private/history.f90:2793-2799` (1-based `k_below_const_q`).
        "k_below_const_q" => {
            MesaCellValue::Int(model.aux.k_below_const_q as i64)
        }
        "q_below_const_q" => {
            let k1 = model.aux.k_below_const_q;
            MesaCellValue::Float(if k1 > 0 && k1 <= nz {
                model.q[k1 - 1]
            } else {
                0.0
            })
        }
        "logxq_below_const_q" => {
            let k1 = model.aux.k_below_const_q;
            MesaCellValue::Float(if k1 > 0 && nz > 0 {
                let sum_dq: f64 =
                    model.dq[..(k1 - 1).min(nz)].iter().sum();
                safe_log10(sum_dq)
            } else {
                0.0
            })
        }
        // `star/private/history.f90:2801-2807` (`aux.k_const_mass` is
        // stored 0-based for `hydro_vars`; convert to 1-based here).
        "k_const_mass" => MesaCellValue::Int(if nz > 0 {
            ((model.aux.k_const_mass + 1).min(nz)) as i64
        } else {
            0
        }),
        "q_const_mass" => MesaCellValue::Float(if nz > 0 {
            let k1 = (model.aux.k_const_mass + 1).min(nz);
            model.q[k1 - 1]
        } else {
            0.0
        }),
        "logxq_const_mass" => MesaCellValue::Float(if nz > 0 {
            let k1 = (model.aux.k_const_mass + 1).min(nz);
            let sum_dq: f64 = model.dq[..(k1 - 1)].iter().sum();
            safe_log10(sum_dq)
        } else {
            0.0
        }),
        // `star/private/history.f90:2814-2815` (1-based `k_below_just_added`).
        "logxq_below_just_added" => MesaCellValue::Float(if nz > 0 {
            let n_above = model.k_below_just_added.saturating_sub(1).min(nz);
            let sum_dq: f64 = model.dq[..n_above].iter().sum();
            safe_log10(sum_dq)
        } else {
            0.0
        }),
        // `star/private/history.f90:2336-2370`,
        // `star/private/star_utils.f90:1328-1485`, and
        // `star_data/public/star_data_step_work.inc:781`.
        "shock_mass"
        | "shock_q"
        | "shock_radius"
        | "shock_velocity"
        | "shock_csound"
        | "shock_lgT"
        | "shock_lgRho"
        | "shock_lgP"
        | "shock_gamma1"
        | "shock_entropy"
        | "shock_tau"
        | "shock_k"
        | "shock_pre_lgRho" => {
            let info = eval_shock_info(model);
            if name == "shock_k" {
                MesaCellValue::Int(info.k_1based as i64)
            } else if info.k_1based == 0 {
                MesaCellValue::Float(0.0)
            } else {
                MesaCellValue::Float(match name {
                    "shock_mass" => info.mass_msun,
                    "shock_q" => info.q,
                    "shock_radius" => info.radius_rsun,
                    "shock_velocity" => info.velocity_cms,
                    "shock_csound" => info.csound_cms,
                    "shock_lgT" => info.lg_t,
                    "shock_lgRho" => info.lg_rho,
                    "shock_lgP" => info.lg_p,
                    "shock_gamma1" => info.gamma1,
                    "shock_entropy" => info.entropy,
                    "shock_tau" => info.tau,
                    // Initialized to `0.0` in `star_data_step_work.inc:781`
                    // and never reassigned in `star/private/`.
                    _ => 0.0,
                })
            }
        }
        // `star/private/init.f90:704` and `history.f90:1593-1594`: initialized to `0.0`
        // and never reassigned anywhere in `star/private/`.
        "h1_czb_mass" => MesaCellValue::Float(0.0),
        other => unreachable!(
            "validated history scalar column {other:?} missing evaluator arm"
        ),
    }
}

/// Evaluates `(fe_core_infall, non_fe_core_infall)` in $\mathrm{cm\,s^{-1}}$
/// matching `star/private/report.f90:378-441`.
fn eval_core_infall_speeds_cms(
    model: &StarModel,
    ctx: &StepReportContext,
    has_vel: bool,
    vel_slice: &[f64],
) -> (f64, f64) {
    let nz = model.nz;
    if !has_vel || nz < 2 || vel_slice.len() < nz {
        return (0.0, 0.0);
    }
    let ts = &model.controls.ts_stop;
    let k_min = (0..nz)
        .min_by(|&a, &b| vel_slice[a].total_cmp(&vel_slice[b]))
        .unwrap_or(0);
    let mut fe_infall = 0.0;
    if ctx.cores.fe_core_mass > 0.0 {
        let mut mass_sum = 0.0;
        let mut k = nz - 1;
        while k > 0 {
            if model.m[k] > M_SUN * ctx.cores.fe_core_mass {
                break;
            }
            if -vel_slice[k] > fe_infall {
                mass_sum += model.dm[k];
            }
            k -= 1;
        }
        let k_fe = k;
        if mass_sum > ts.fe_core_infall_mass * M_SUN {
            if model.controls.output.report_max_infall_inside_fe_core {
                let min_v = if model.controls.v_flag && model.v.len() >= nz {
                    model.v[k_fe..nz]
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min)
                } else {
                    0.0
                };
                fe_infall = -min_v;
            } else {
                fe_infall = -vel_slice[k_min];
            }
        }
    }
    let mut non_fe_infall = 0.0;
    if ctx.cores.he_core_mass > 0.0 {
        let mut mass_sum = 0.0;
        for k in 0..nz {
            if model.m[k] > M_SUN * ctx.cores.he_core_mass {
                continue;
            }
            if model.m[k] < M_SUN * ctx.cores.fe_core_mass {
                break;
            }
            if -vel_slice[k] > non_fe_infall {
                mass_sum += model.dm[k];
            }
        }
        if mass_sum > ts.non_fe_core_infall_mass * M_SUN
            && model.m[k_min] <= ctx.cores.he_core_mass * M_SUN
        {
            non_fe_infall = -vel_slice[k_min];
        }
    }
    (fe_infall, non_fe_infall)
}

/// `star_get_history_output` (`star/public/star_lib.f90:3019-3047`): the
/// current value of history column `name` for `model`, integers converted to
/// `f64`.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] when `name` is not a single history
/// column (upstream returns `-huge` with `ierr = 1`).
pub fn star_get_history_output(
    model: &StarModel,
    name: &str,
) -> Result<f64, StarError> {
    let specs = parse_star_history_columns(name, None, model)?;
    if specs.len() != 1 {
        return Err(StarError::InvalidInput(format!(
            "star_get_history_output: '{name}' is not a single history column"
        )));
    }
    let row = evaluate_star_history_row(model, &specs);
    row.first().map(|(_, v)| v.as_f64()).ok_or_else(|| {
        StarError::InvalidInput(format!(
            "star_get_history_output: no value for '{name}'"
        ))
    })
}

/// Evaluates all `specs` for `model` at the current step (`star/private/history.f90`).
#[must_use]
pub fn evaluate_star_history_row(
    model: &StarModel,
    specs: &[HistoryColumnSpec],
) -> Vec<(String, MesaCellValue)> {
    let ctx = StepReportContext::compute(model);
    evaluate_star_history_row_with_ctx(model, specs, &ctx)
}

/// Evaluates all `specs` for `model` reusing a precomputed [`StepReportContext`].
#[must_use]
pub fn evaluate_star_history_row_with_ctx(
    model: &StarModel,
    specs: &[HistoryColumnSpec],
    ctx: &StepReportContext,
) -> Vec<(String, MesaCellValue)> {
    let mut max_mix = 0_usize;
    let mut max_mix_relr = 0_usize;
    let mut max_burn = 0_usize;
    let mut max_burn_relr = 0_usize;
    for sp in specs {
        match *sp {
            HistoryColumnSpec::MixType(ii) | HistoryColumnSpec::MixQtop(ii) => {
                max_mix = max_mix.max(ii);
            }
            HistoryColumnSpec::MixRelrType(ii)
            | HistoryColumnSpec::MixRelrTop(ii) => {
                max_mix_relr = max_mix_relr.max(ii);
            }
            HistoryColumnSpec::BurnType(ii)
            | HistoryColumnSpec::BurnQtop(ii) => {
                max_burn = max_burn.max(ii);
            }
            HistoryColumnSpec::BurnRelrType(ii)
            | HistoryColumnSpec::BurnRelrTop(ii) => {
                max_burn_relr = max_burn_relr.max(ii);
            }
            _ => {}
        }
    }
    let mix_regs =
        eval_history_mixing_regions(model, &ctx.mix.cz_bdy_dq, max_mix);
    let mix_relr_regs = if max_mix_relr == max_mix {
        mix_regs.clone()
    } else {
        eval_history_mixing_regions(model, &ctx.mix.cz_bdy_dq, max_mix_relr)
    };
    let max_b_all = max_burn.max(max_burn_relr);
    let burn_regs = eval_history_burning_regions(model, max_b_all);
    let mut astero_cache = None;

    let nz = model.nz;
    specs
        .iter()
        .map(|sp| {
            let val = match sp {
                HistoryColumnSpec::Scalar(name) => {
                    eval_scalar_column(name, model, ctx, &mut astero_cache)
                }
                HistoryColumnSpec::CategoryLuminosity(cid) => {
                    MesaCellValue::Float(safe_log10(ctx.cat_lum_lsun[cid - 1]))
                }
                HistoryColumnSpec::CenterLogEpsBurn(cid) => {
                    MesaCellValue::Float(safe_log10(
                        ctx.center_eps_burn[cid - 1].abs(),
                    ))
                }
                HistoryColumnSpec::Nuclide(q, cid, _) => {
                    let slot = iso_slot_for_chem_id(model, *cid);
                    let v = match q {
                        HistoryIsoQuery::Center => {
                            center_avg_x_slot(model, slot)
                        }
                        HistoryIsoQuery::Surface => {
                            surface_avg_x_slot(model, slot)
                        }
                        HistoryIsoQuery::Average => {
                            star_avg_x_slot(model, slot)
                        }
                        HistoryIsoQuery::TotalMass => {
                            star_avg_x_slot(model, slot) * model.xmstar / M_SUN
                        }
                        HistoryIsoQuery::LogCenter => {
                            safe_log10(center_avg_x_slot(model, slot))
                        }
                        HistoryIsoQuery::LogSurface => {
                            safe_log10(surface_avg_x_slot(model, slot))
                        }
                        HistoryIsoQuery::LogAverage => {
                            safe_log10(star_avg_x_slot(model, slot))
                        }
                        HistoryIsoQuery::LogTotalMass => safe_log10(
                            star_avg_x_slot(model, slot) * model.xmstar / M_SUN,
                        ),
                        HistoryIsoQuery::MaxEpsNucLog => {
                            let x = slot.filter(|_| nz > 0).map_or(0.0, |j| {
                                let k = (0..nz)
                                    .max_by(|&a, &b| {
                                        model.eps_nuc[a]
                                            .total_cmp(&model.eps_nuc[b])
                                    })
                                    .unwrap_or(0);
                                model.cell_xa(k)[j]
                            });
                            safe_log10(x)
                        }
                        HistoryIsoQuery::CzTopLog => {
                            let x = match (slot, ctx.mix.largest_conv_top_bot) {
                                (Some(j), Some((top, _))) if top > 0 => {
                                    model.cell_xa(top - 1)[j]
                                }
                                _ => 0.0,
                            };
                            safe_log10(x)
                        }
                        HistoryIsoQuery::CzLog => {
                            let x = match (slot, ctx.mix.largest_conv_top_bot) {
                                (Some(j), Some((_, bot))) if bot > 0 => {
                                    model.cell_xa(bot - 1)[j]
                                }
                                _ => 0.0,
                            };
                            safe_log10(x)
                        }
                    };
                    MesaCellValue::Float(v)
                }
                HistoryColumnSpec::MixType(ii) => {
                    MesaCellValue::Int(mix_regs[ii - 1].0)
                }
                HistoryColumnSpec::MixQtop(ii) => {
                    MesaCellValue::Float(mix_regs[ii - 1].1)
                }
                HistoryColumnSpec::MixRelrType(ii) => {
                    MesaCellValue::Int(mix_relr_regs[ii - 1].0)
                }
                HistoryColumnSpec::MixRelrTop(ii) => {
                    MesaCellValue::Float(mix_relr_regs[ii - 1].2)
                }
                HistoryColumnSpec::BurnType(ii)
                | HistoryColumnSpec::BurnRelrType(ii) => {
                    MesaCellValue::Int(burn_regs[ii - 1].0)
                }
                HistoryColumnSpec::BurnQtop(ii) => {
                    MesaCellValue::Float(burn_regs[ii - 1].1)
                }
                HistoryColumnSpec::BurnRelrTop(ii) => {
                    MesaCellValue::Float(burn_regs[ii - 1].2)
                }
            };
            (sp.column_name(), val)
        })
        .collect()
}

pub(crate) fn validate_extra_header_items(
    kind: &str,
    items: &[(String, f64)],
) -> Result<(), StarError> {
    for (idx, (name, _)) in items.iter().enumerate() {
        if name.trim().is_empty() {
            return Err(StarError::InvalidInput(format!(
                "extra {kind} header item #{idx} has an empty name"
            )));
        }
    }
    Ok(())
}

/// Formats the 6-line MESA `26.4.1` `history.data` header (`do_history_info` in
/// `star/private/history.f90:378-405, 469-484`).
#[must_use]
pub fn format_star_history_file_header(
    model: &StarModel,
    specs: &[HistoryColumnSpec],
) -> String {
    format_star_history_file_header_with_extra(model, specs, &[])
}

/// Formats the 6-line MESA `26.4.1` `history.data` header including optional
/// appended extra/binary columns (`do_binary_col` in `star/private/history.f90:548-557`).
#[must_use]
pub fn format_star_history_file_header_with_extra(
    model: &StarModel,
    specs: &[HistoryColumnSpec],
    extra_col_names: &[&str],
) -> String {
    format_star_history_file_header_with_extras(
        model,
        specs,
        &[],
        extra_col_names,
    )
}

/// Formats the 6-line MESA `26.4.1` `history.data` header including both
/// `extra_history_header_items` (`star/private/history.f90:353-398`) and
/// appended extra/binary column names (`history.f90:548-557`).
#[must_use]
pub fn format_star_history_file_header_with_extras(
    model: &StarModel,
    specs: &[HistoryColumnSpec],
    extra_header_items: &[(String, f64)],
    extra_col_names: &[&str],
) -> String {
    let o = &model.controls.output;
    let mut hdr_items: Vec<(&str, String)> =
        Vec::with_capacity(11 + extra_header_items.len());
    hdr_items.push((
        "version_number",
        format_mesa_header_str(MESA_REF_VERSION_NUMBER),
    ));
    hdr_items.push(("compiler", format_mesa_header_str("rustc")));
    hdr_items.push(("build", format_mesa_header_str("flexmesa")));
    hdr_items.push((
        "MESA_SDK_version",
        format_mesa_header_str("flexmesa-0.1"),
    ));
    hdr_items.push((
        "math_backend",
        format_mesa_header_str(flexmesa_math::numerics_mode().as_str()),
    ));
    hdr_items.push(("date", format_mesa_header_str("20260401")));
    hdr_items.push(("burn_min1", format_mesa_dbl(o.burn_min1)));
    hdr_items.push(("burn_min2", format_mesa_dbl(o.burn_min2)));
    hdr_items.push(("msun", format_mesa_dbl(M_SUN)));
    hdr_items.push(("rsun", format_mesa_dbl(R_SUN)));
    hdr_items.push(("lsun", format_mesa_dbl(L_SUN)));
    for (name, val) in extra_header_items {
        hdr_items.push((name.trim(), format_mesa_dbl(*val)));
    }
    let spec_names: Vec<String> =
        specs.iter().map(HistoryColumnSpec::column_name).collect();
    let mut col_names: Vec<&str> =
        Vec::with_capacity(spec_names.len() + extra_col_names.len());
    for n in &spec_names {
        col_names.push(n.as_str());
    }
    col_names.extend_from_slice(extra_col_names);

    let mut out = String::with_capacity(512 + col_names.len() * 84);
    push_mesa_6line_header(
        &mut out,
        &hdr_items,
        &col_names,
        STAR_LOG_COL_WIDTH,
    );
    out
}

/// Formats one evaluated `history.data` row line (`do_col_pass3` in `history.f90:1073-1155`).
#[must_use]
pub fn format_star_history_data_line(
    row: &[(String, MesaCellValue)],
) -> String {
    format_star_history_data_line_with_extra(row, &[])
}

/// Formats one evaluated `history.data` row line plus optional appended `f64`
/// columns (`do_binary_col` in `star/private/history.f90:548-557`).
#[must_use]
pub fn format_star_history_data_line_with_extra(
    row: &[(String, MesaCellValue)],
    extra_cols: &[(String, f64)],
) -> String {
    let mut line =
        String::with_capacity((row.len() + extra_cols.len()) * 41 + 1);
    for (_, val) in row {
        match *val {
            MesaCellValue::Int(i) => {
                write_mesa_int_width(&mut line, i, STAR_LOG_COL_WIDTH);
            }
            MesaCellValue::Float(x) => {
                write_mesa_dbl_width(&mut line, x, STAR_LOG_COL_WIDTH);
            }
        }
    }
    for (_, x) in extra_cols {
        write_mesa_dbl_width(&mut line, *x, STAR_LOG_COL_WIDTH);
    }
    let _ = writeln!(line);
    line
}

/// Incremental `LOGS/history.data` writer that writes the 6-line MESA header on
/// the first recorded step (or to `<log_dir>/<star_history_header_name>` when
/// configured per `star/private/history.f90:315-318, 423-431`) and flushes every
/// row immediately to disk (`#163`).
pub struct IncrementalHistoryWriter {
    path: PathBuf,
    header_path: Option<PathBuf>,
    specs: Vec<HistoryColumnSpec>,
    writer: Option<BufWriter<File>>,
    rows_written: usize,
    last_model_number: Option<usize>,
}

impl IncrementalHistoryWriter {
    /// Creates an incremental writer targeting `path` with resolved `specs`.
    #[must_use]
    pub fn new(path: PathBuf, specs: Vec<HistoryColumnSpec>) -> Self {
        Self::with_header_path(path, None, specs)
    }

    /// Creates an incremental writer targeting `path` with an optional split
    /// `header_path` (`star_history_header_name`).
    #[must_use]
    pub fn with_header_path(
        path: PathBuf,
        header_path: Option<PathBuf>,
        specs: Vec<HistoryColumnSpec>,
    ) -> Self {
        Self {
            path,
            header_path,
            specs,
            writer: None,
            rows_written: 0,
            last_model_number: None,
        }
    }

    /// Removes any pre-existing file at `self.path` (and `self.header_path`) before
    /// step 1 so a step-1 stage failure never leaves a previous stage's file intact.
    pub fn clear_stale_file(&self) -> Result<(), StarError> {
        if self.path.is_file() {
            fs::remove_file(&self.path).map_err(|e| {
                StarError::IoError(format!(
                    "failed to clear stale {}: {e}",
                    self.path.display()
                ))
            })?;
        }
        if let Some(ref hp) = self.header_path {
            if hp.is_file() {
                fs::remove_file(hp).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to clear stale {}: {e}",
                        hp.display()
                    ))
                })?;
            }
        }
        Ok(())
    }

    /// Output file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of data rows flushed to disk so far.
    #[must_use]
    pub const fn rows_written(&self) -> usize {
        self.rows_written
    }

    /// Returns whether the initial `history.data` header has already been emitted.
    #[must_use]
    pub fn header_written(&self) -> bool {
        self.writer.is_some()
    }

    /// Evaluates `model`, writes the header on the first row, appends the row,
    /// and flushes immediately (`write_history_info` in `star/private/history.f90:48-53`).
    pub fn write_step(&mut self, model: &StarModel) -> Result<(), StarError> {
        self.write_step_with_extra(model, &[])
    }

    /// Evaluates `model` plus optional extra/binary columns (`append_to_star_history`
    /// in `star/private/history.f90:475-477, 548-557`), writes the header on the
    /// first row, appends the row, and flushes immediately.
    pub fn write_step_with_extra(
        &mut self,
        model: &StarModel,
        extra_cols: &[(String, f64)],
    ) -> Result<(), StarError> {
        let ctx = StepReportContext::compute(model);
        self.write_step_with_ctx_and_extra(model, &ctx, extra_cols)
    }

    /// Evaluates `model` reusing `ctx`, writes the header on the first row, appends
    /// the row, and flushes immediately.
    pub fn write_step_with_ctx_and_extra(
        &mut self,
        model: &StarModel,
        ctx: &StepReportContext,
        extra_cols: &[(String, f64)],
    ) -> Result<(), StarError> {
        self.write_step_with_ctx_and_extras(model, ctx, &[], extra_cols)
    }

    /// Evaluates `model` reusing `ctx` with both `extra_history_header_items`
    /// (`star/private/history.f90:353-398`) and appended `extra_cols`, writing
    /// the header on the first row and flushing immediately.
    pub fn write_step_with_ctx_and_extras(
        &mut self,
        model: &StarModel,
        ctx: &StepReportContext,
        extra_header_items: &[(String, f64)],
        extra_cols: &[(String, f64)],
    ) -> Result<(), StarError> {
        if self.last_model_number == Some(model.model_number) {
            return Ok(());
        }
        if self.writer.is_none() {
            validate_extra_header_items("history", extra_header_items)?;
            if let Some(parent) =
                self.path.parent().filter(|p| !p.as_os_str().is_empty())
            {
                fs::create_dir_all(parent).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to create log directory {}: {e}",
                        parent.display()
                    ))
                })?;
            }
            let extra_names: Vec<&str> =
                extra_cols.iter().map(|(k, _)| k.as_str()).collect();
            let hdr = format_star_history_file_header_with_extras(
                model,
                &self.specs,
                extra_header_items,
                &extra_names,
            );
            if let Some(ref hp) = self.header_path {
                if let Some(hparent) =
                    hp.parent().filter(|p| !p.as_os_str().is_empty())
                {
                    fs::create_dir_all(hparent).map_err(|e| {
                        StarError::IoError(format!(
                            "failed to create header directory {}: {e}",
                            hparent.display()
                        ))
                    })?;
                }
                fs::write(hp, hdr.as_bytes()).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to write history header {}: {e}",
                        hp.display()
                    ))
                })?;
            }
            let f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&self.path)
                .map_err(|e| {
                    StarError::IoError(format!(
                        "failed to open {}: {e}",
                        self.path.display()
                    ))
                })?;
            let mut bw = BufWriter::new(f);
            if self.header_path.is_none() {
                bw.write_all(hdr.as_bytes()).map_err(|e| {
                    StarError::IoError(format!(
                        "failed to write history header {}: {e}",
                        self.path.display()
                    ))
                })?;
            }
            self.writer = Some(bw);
        }

        let row = evaluate_star_history_row_with_ctx(model, &self.specs, ctx);
        let line = format_star_history_data_line_with_extra(&row, extra_cols);
        if let Some(ref mut bw) = self.writer {
            bw.write_all(line.as_bytes()).map_err(|e| {
                StarError::IoError(format!(
                    "failed to write history row to {}: {e}",
                    self.path.display()
                ))
            })?;
            bw.flush().map_err(|e| {
                StarError::IoError(format!(
                    "failed to flush {}: {e}",
                    self.path.display()
                ))
            })?;
        }
        self.rows_written += 1;
        self.last_model_number = Some(model.model_number);
        Ok(())
    }
}
