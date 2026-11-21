// SPDX-License-Identifier: LGPL-3.0-or-later
//! 1D stellar structure, ZAMS/pre-MS initialization, `.mod`/`.photo` model I/O,
//! zone/face microphysics, coupled Henyey solver, convective boundary
//! overshooting, structure-burn-mix driver, and stellar winds
//! (`flexmesa-star`, Wave 4).
//!
//! # Architecture Overview
//! Ports upstream Fortran MESA's `star_data` and `star` modules into Safe Rust
//! (`#![forbid(unsafe_code)]`) with `rayon` zone parallelism and 3-cell stencil
//! forward-mode automatic differentiation ([`StarAdOrder1`]):
//! - **State & I/O ([`model`], [`controls`], [`mesh`], [`model_io`], [`zams`],
//!   [`pre_ms`])**: Canonical [`StarModel`] representation, staggered
//!   Lagrangian mass grid, ZAMS interpolation, $n=1.5$ pre-MS polytrope
//!   builder, and `.mod`/`.photo` serialization.
//! - **Microphysics & Face Transport ([`micro`], [`hydro_vars`],
//!   [`eps_grav`])**: Parallel zone EOS, opacity, neutrino, and nuclear
//!   reaction network calls, MLT/TDC convection, Brunt-Väisälä buoyancy, and
//!   surface boundary conditions.
//! - **Henyey Equations & Solver ([`hydro_eqns`], [`solver`])**:
//!   Coupled hydrodynamic and species conservation equations, analytical
//!   block-tridiagonal Jacobian assembly, and damped Newton-Raphson solver.
//! - **Mixing, Burning & Winds ([`mix_info`], [`overshoot`],
//!   [`struct_burn_mix`], [`winds`])**: Sub-cell convective boundary detection,
//!   step/exponential overshooting, operator-split and coupled
//!   burning/diffusion, rotational angular momentum diffusion, and stellar wind
//!   mass adjustment.
//! - **Remeshing, Timestep & Evolution ([`mesh_adjust`], [`timestep`],
//!   [`evolve`], [`history`])**: Adaptive Lagrangian mesh spacing, Söderlind &
//!   Wang (2006) H211 timestep control, evolutionary retry cascade, and
//!   `history.data`/`profile.data` output.

#![forbid(unsafe_code)]
#![allow(
    clippy::approx_constant,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::fn_params_excessive_bools,
    clippy::inline_always,
    clippy::items_after_statements,
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unreadable_literal
)]

pub mod accretion_xa;
pub mod adjust_mass;
pub mod adjust_mesh_split_merge;
pub mod adjust_mesh_support;
pub mod case_staging;
pub mod cell_energy;
pub mod controls;
mod controls_defaults;
pub mod controls_diffusion;
pub mod controls_mix_wind;
pub mod controls_output;
pub mod controls_timestep_stop;
pub mod conv_premix;
pub mod core_boundaries;
pub mod diffusion;
pub mod diffusion_procs;
pub mod diffusion_support;
pub mod element_diffusion;
pub mod eps_grav;
pub mod eps_mdot;
pub mod evolve;
pub(crate) mod evolve_support;
pub mod history;
pub mod hydro_brunt_surf;
pub mod hydro_energy;
pub mod hydro_eqns;
pub mod hydro_momentum;
pub mod hydro_rotation;
pub mod hydro_rti;
pub mod hydro_state;
pub mod hydro_temperature;
pub mod hydro_vars;
pub mod hydro_work;
pub mod inlist;
pub mod inlist_inventory;
pub mod inlist_populate;
mod inlist_populate_limits;
mod inlist_populate_micro;
mod inlist_populate_phase5b;
pub mod inlist_runner;
pub mod job_controls;
pub mod mdot_redo;
pub mod mesa_history;
pub mod mesa_log_reader;
pub mod mesa_profile;
pub mod mesh;
pub mod mesh_adjust;
pub mod mesh_functions;
pub mod mesh_plan;
pub mod mass_utils;
pub mod mesh_remap;
pub mod micro;
pub mod micro_cache;
pub mod micro_net;
pub mod mix_info;
pub(crate) mod mixing_micro;
pub mod mixing_state;
pub mod model;
pub mod model_io;
pub mod omega_mix;
pub mod overshoot;
pub mod paquette_coeffs;
pub mod phase_separation;
pub mod photo_io;
pub mod pre_ms;
pub mod predictive_mix;
pub mod premix;
pub mod primary_state;
pub mod pulse;
pub mod pulse_bvp;
pub mod pulse_bvp_recon;
pub mod pulse_exporters;
pub mod pulse_modes;
pub mod pulse_utils;
pub mod relax;
pub mod relax_comp;
pub mod report_utils;
pub mod riemann;
pub mod riemann_ad;
pub mod riemann_eqns;
pub mod rn_parser;
pub mod rotation;
pub(crate) mod rotation_mix;
pub mod rsp2;
pub mod rsp2_ad;
pub mod rsp2_eqns;
pub mod rsp2_remesh;
pub mod rsp_build;
pub mod rsp_config;
pub mod rsp_def;
pub mod rsp_driver;
pub mod rsp_eos_inv;
pub mod rsp_eos_kap;
pub mod rsp_eqns;
pub(crate) mod rsp_eqns_cell;
pub mod rsp_grid;
pub mod rsp_hyd;
pub mod rsp_hyd_flux;
pub mod rsp_hyd_turb;
pub mod rsp_hyd_vars;
pub mod rsp_hyd_work;
pub mod rsp_lina;
pub(crate) mod rsp_lina_io;
pub mod rsp_relax_env;
pub mod rsp_stah;
pub(crate) mod rsp_state_io;
pub(crate) mod rsp_stats;
pub mod rsp_turb_partials;
pub mod rsp_zone_micro;
pub mod solver;
pub mod solver_norms;
pub mod solver_support;
pub mod star_ad;
pub mod star_job_init;
pub mod stella;
pub mod step_trace;
pub mod struct_burn_mix;
pub mod termination;
pub mod timestep;
mod timestep_checks;
pub mod timestep_limits;
pub mod winds;
pub(crate) mod winds_driver;
pub mod zams;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_fidelity1;
#[cfg(test)]
mod tests_fidelity2;
#[cfg(test)]
mod tests_fidelity4;
#[cfg(test)]
mod tests_phase3_pulse_stella;

pub use adjust_mass::{
    build_accreted_xa, compute_delta_m, do_adjust_mass, do_adjust_mass_step,
    do_adjust_mass_with_dt, get_xa_for_accretion,
};
pub use adjust_mesh_split_merge::{
    get1_grad, minmod3, preview_split_merge_amr, remesh_split_merge,
};
pub use adjust_mesh_support::{
    GvalInfo, OtherMeshDeltaCoeffFactorHandle, OtherMeshDeltaCoeffFactorHook,
    check_validity, get_gval_info, get_gval_info_with_controls,
};
pub use controls::{
    AtmOption, AtmTTauRelation, EnergyEqnOption, StarControls, StarError,
    StarJobControls, StepFailureInfo, VarLayout,
};
pub use controls_diffusion::{DiffusionControls, MAX_NUM_DIFFUSION_CLASSES};
pub use controls_mix_wind::{
    MixWindControls, NUM_OVERSHOOT_PARAM_SETS, NUM_PREDICTIVE_PARAM_SETS,
    OvershootParamSet, PredictiveParamSet,
};
pub use controls_output::OutputControls;
pub use controls_timestep_stop::{MAX_DX_LIMIT_CTRLS, TimestepStopControls};
pub use core_boundaries::{
    CoreBoundaryInfo, CoreMasses, PowerCategories, compute_core_masses,
    compute_power_categories, get_info_at_q, k_for_q_1based,
    snapshot_step_history_old,
};
pub use diffusion::{
    BurgersZoneResult, DIFFUSION_MIN_NC, DiffusionScheme, IsolveSolver,
    SolveDiffusionInput, SolveDiffusionOutput, SolveDiffusionStats,
    do_element_diffusion, do_solve_diffusion, eval_burgers_velocities,
    eval_caplan_ocp_dstar, eval_coulomb_collision_factors,
    eval_radiative_levitation_accel, eval_wd_sedimentation_heating,
};
pub use element_diffusion::{
    do_element_diffusion_step, finish_element_diffusion,
};
pub use eps_grav::{
    eval_eps_grav, eval_eps_grav_lns_cell, eval_eps_grav_std_cell,
};
pub use eps_mdot::{
    AdjustMassRecord, EpsMdotSurface, calculate_eps_mdot,
    eval_total_energy_profile, total_energy_profile_before_adjust_mass,
    eval_cell_specific_total_energy, eval_specific_total_energy_profile,
    find_mass_flux,
};
pub use evolve::{
    EvolveStepReport, StarPickTimestepOutcome, StarStepAttemptOutcome,
    StarStepConverged, StarStepPart1State, restore_from_old, star_evolve,
    star_evolve_step, star_evolve_step_part1, star_evolve_step_part2,
    star_evolve_step_with_extras, star_finalize_terminated_step,
    star_pick_next_timestep_after_step,
};
pub use evolve_support::{
    apply_step_pre_physics, eval_total_energy_integrals, relax_uniform_omega,
    remesh_star_before_step, update_step_energy_conservation,
};
pub use history::{
    HistoryRow, extract_history_row, format_history_header, format_history_row,
    format_profile_string,
};
pub use hydro_energy::{
    eval_cell_pressure_for_work_ad, eval_face_area_time_centered_ad,
    eval_pvsc_cell_ad, setup_d_turbulent_energy_dt, setup_dl_dm,
    setup_sources_and_others,
};
pub use hydro_eqns::{
    HenyeySystem, compute_x_scale, do_surf_bc_eqns, do1_chem_eqns,
    do1_density_eqn, do1_energy_eqn, do1_momentum_eqn, do1_radius_eqn,
    do1_temperature_eqn, eval_henyey_equations,
};
pub use hydro_momentum::{
    eval_cell_peos_time_centered_ad, eval_face_dm_bar, expected_hse_grav_term,
    expected_non_hse_term,
};
pub use hydro_rti::{
    do1_dalpha_rti_dt_eqn, eval_alpha_rti_sources, set_dpdr_drhodr_info,
    set_rti_mixing_info,
};
pub use hydro_state::{
    HydroAuxState, TidalDeformationHandle, TidalDeformationHook,
    TidalShellDeformation,
};
pub use hydro_temperature::{interp_p_point_ad, set_rsp_lsurf_bc};
pub use hydro_vars::{
    check_for_redo_mlt, do_brunt_b, do_brunt_n2, eval_surf_bc,
    set_gradt_excess_alpha, set_hydro_vars, set_mlt_vars,
    update_sig_from_d_mix,
};
pub use mixing_state::{
    set_grads, threshold_smoothing, weighted_smoothing, MixingAuxState,
};
pub use inlist::{InlistBundle, InlistLoadOptions, SectionControls};
pub use inlist_inventory::{
    StarControlInventory, copy_cleaned_star_case,
    supported_star_control_inventory,
};
pub use inlist_runner::{
    CaseRunReport, DefaultStarExtras, ExtrasDirective, RspBuildOutcome,
    StageRunReport, StarExtras, StarLogsOptions, run_inlist_stage,
    run_inlist_stage_with_logs, run_test_suite_case,
    run_test_suite_case_with_extras, run_test_suite_case_with_logs,
    run_test_suite_case_with_options, verify_photo_restart_parity,
};
pub use mdot_redo::{
    MdotRedoState, SelectMdotAction, eval_surf_avg_w_div_w_crit,
};
pub use mesa_history::{
    DEFAULT_STAR_HISTORY_COLUMNS_TEXT, HistoryColumnSpec, HistoryIsoQuery,
    IncrementalHistoryWriter, MesaCellValue, default_star_history_specs,
    evaluate_star_history_row, format_star_history_data_line,
    format_star_history_data_line_with_extra, format_star_history_file_header,
    format_star_history_file_header_with_extra, parse_star_history_columns,
    resolve_star_history_columns,
};
pub use mesa_log_reader::{
    MesaHeaderValue, MesaLogTable, read_mesa_log_file, read_mesa_log_str,
    read_profiles_index_file, read_profiles_index_str,
};
pub use mesa_profile::{
    DEFAULT_STAR_PROFILE_COLUMNS_TEXT, ProfileColumnSpec, ProfilesIndex,
    ProfilesIndexEntry, StarLogsWriter, default_star_profile_specs,
    format_star_profile_text, parse_star_profile_columns,
    resolve_star_profile_columns,
};
pub use mesh_adjust::{
    CellType, MeshFunctionConfig, StandardMeshPlan, classify_new_cells,
    compute_monotone_xa_slopes, do_adjust_mesh, do_prune_mesh_surface,
    eval_effective_mesh_delta_coeff, eval_mesh_gval, plan_mesh, remap_onto_dq,
    solve_cell_lnt_for_energy, try_plan_standard_mesh,
};
pub use mesh_functions::{
    MAX_ALLOWED_GVALS, MeshFunctionData, OtherMeshFunctionColumn,
    OtherMeshFunctionsHandle, OtherMeshFunctionsHook, num_mesh_functions,
    set_mesh_function_data,
};
pub use mesh_plan::{MeshPlanOutput, do_mesh_plan};
pub use micro::{
    change_model_network, do_eos, do_kap, do_net, do_neu, get_or_load_net,
    get_tau, global_basic_net, set_basic_vars,
};
pub use micro_cache::MicrophysicsCache;
pub use mix_info::{
    cdc_factor, clean_mixing_glitches, compute_convection_sigmas,
    do_mix_envelope, eval_dxdt_mix, face_density, locate_convection_boundaries,
    set_cz_bdy_mass, set_cz_boundary_info, set_mixing_info,
    set_mixing_info_with_skip,
};
pub use model::StarModel;
pub use model_io::{
    PHOTO_FORMAT_VERSION, STAR_DEF_VERSION, compute_photo_checksum,
    load_photo_bytes, photo_numerics_mode, read_mod_str, read_photo_bytes,
    try_write_photo_bytes, write_mod_string, write_photo_bytes,
};
pub use overshoot::{
    add_overshooting, apply_overshoot_configs, eval_conv_bdy_hp,
    eval_conv_bdy_k, eval_conv_bdy_r, eval_over_bdy_params, OvershootConfig,
    OvershootScheme,
};
pub use paquette_coeffs::paquette_coefficients;
pub use phase_separation::{
    EOS_PHASE_BOUNDARY, apply_phase_sep_mixing_types, blouin_delta_xne,
    blouin_delta_xo, do_phase_separation, do_phase_separation_step,
    eval_phase_separation_shift,
};
pub use pre_ms::{
    PreMsShell, build_pre_ms_model, build1_pre_ms_model,
    default_basic_net_composition,
};
pub use premix::{
    add_predictive_mixing, apply_predictive_mixing_from_controls,
    do_conv_premix,
};
pub use primary_state::PrimaryStateBuffer;
pub use pulse::{
    PULSE_ISO_NAMES, PulseExportOptions, PulsePoint, build_pulse_points,
    build_pulse_points_with_options, eval_center_d2, eval_center_parabolic,
    eval_center_rho, eval_face, export_fgong, export_fgong_with_version,
    export_gyre, export_gyre_with_schema, export_osc, set_segment_indices,
};
pub use pulse_bvp::{
    BvpEigenmode, BvpModel, BvpPoint, InertiaNormKind, OuterBoundKind,
    shoot_determinant, steffen_slopes,
};
pub use pulse_bvp_recon::{
    classify_mode_orders, parse_fgong_model, parse_fortran_float,
    parse_gyre_mesa_model, reconstruct_and_analyze_mode,
};
pub use pulse_exporters::{
    GsmData, export_cafein, export_gr1d, export_gsm, export_gsm_data,
    export_pulse_by_format, export_saio,
};
pub use pulse_modes::{
    LinearOscillationMode, PulseModeScanOptions,
    build_bvp_model_from_pulse_points, eval_acoustic_large_separation_hz,
    eval_astero_scaling_dnu_numax_uhz, eval_buoyancy_period_spacing_s,
    eval_kappa_mechanism_growth, eval_linear_oscillation_modes,
    eval_target_mode_frequency_and_growth, scan_bounds_from_targets,
    scan_star_bvp_eigenmodes, solve_adipls_fgong_modes, solve_bvp_eigenmodes,
    solve_gyre_mesa_modes,
};
pub use relax::{
    apply_composition_relaxation, apply_planetary_core, apply_uniform_rotation,
    relax_core_boundary, relax_from_profile_files, relax_irradiation_to_target,
    relax_mass_to_target, relax_tau_factor_to_target, relax_to_remove_h_env,
};
pub use relax_comp::relax_composition_with_henyey;
pub use report_utils::{
    ConvAndMixSummary, MESA_REF_VERSION_NUMBER, PhotosphereInfo,
    STAR_LOG_COL_WIDTH, StepReportContext, SurfAvgRotationInfo,
    center_avg_x_by_name, center_value, center_value_by,
    eval_conv_and_mx_summary, eval_conv_time_scale_zone, eval_conv_time_scales,
    eval_photosphere_info, eval_surf_avg_rotation_info,
    eval_upstream_omega_crit, format_mesa_dbl, format_mesa_dbl_width,
    format_mesa_header_str, format_mesa_header_str_width, format_mesa_int,
    format_mesa_int_width, format_mesa_txt, format_mesa_txt_width,
    parse_quoted_include_arg, push_mesa_6line_header, safe_log10,
    surface_avg_x_by_name, write_mesa_dbl_width, write_mesa_int_width,
    write_mesa_txt_width,
};
pub use riemann::{
    HllcFaceState, eval_hllc_face, eval_hllc_surface_face,
    eval_riemann_momentum_residual, eval_riemann_profile,
};
pub use riemann_ad::{
    calc_ptot_cell_ad, do_uface_and_pface, do1_riemann_momentum_eqn,
};
pub use rn_parser::{RnStage, parse_rn_stages};
pub use rotation::{
    RI_CRIT_DSI, RotationalMixZone, eval_magnetic_diffusivity, eval_omega_crit,
    eval_rotation_fp_ft, eval_rotational_mixing_zone, eval_thermal_diffusivity,
    set_rotation_mixing_info, sync_post_henyey_rotation,
    update_rotation_distortion_factors,
};
pub use rsp_build::{
    RspLinaMode, build_rsp_envelope_model, do_rsp_build_model,
    eval_rsp_lina_modes,
};
pub use rsp_config::{Rsp2Controls, RspConfig};
pub use rsp_def::{
    F_EDD_FREE_STREAM, F_EDD_ISOTROPIC, MAX_RSP_NZN, RspDerivedConsts, RspState,
};
pub use rsp_driver::{
    rsp_one_step, rsp_one_step_with_extras, rsp_setup_model,
    rsp_total_energy_integrals,
};
pub use rsp_eos_kap::{
    RspChemCache, RspEosKapOut, compute_standard_metals_xa,
    eval_mesa_eos_and_kap, eval1_gamma_pt_get_rho, eval1_mesa_eos_and_kap,
    eval1_mesa_eos_de_and_kap, eval1_mesa_eos_degas_and_kap,
    eval1_mesa_rho_given_pt, eval1_mesa_t_given_dp, get_surf_p_t_kap,
    set_rho_for_new_pgas, set_t_for_new_egas, set_t_for_new_energy,
    set_t_for_new_pgas,
};
pub use rsp_grid::{
    MesaRanmar, RspEnvGrid, RspTotalEosKap, cleanup_for_lina, copy_results,
    eop, finish_build_rsp_model, mesa_eos_kap,
};
pub use rsp_lina::{RspLinaOutput, do_lina};
pub use rsp_relax_env::relax_env;
pub use rsp_stah::stah;
pub use rsp2::{
    Rsp2Config, Rsp2FaceResult, Rsp2ZoneResult, X_ALFAC, X_ALFAP, X_ALFAS,
    X_CEDE, X_GAMMAR, compute_rsp2_uq_face, eval_rsp2_face, eval_rsp2_zone,
    rsp2_flux_limiter, solve_stationary_rsp2_w,
};
pub use rsp2_ad::{
    eval_and_save_rsp2_eqns, eval_rsp2_hp_face_ad, eval_rsp2_ptrb_cell_ad,
    update_rsp2_vars_ad,
};
pub use rsp2_eqns::{
    Rsp2ProfileResiduals, eval_rsp2_hp_face_zone, eval_rsp2_profile_equations,
    rsp2_adjust_vars_before_call_solver,
};
pub use rsp2_remesh::{remesh_for_rsp2, seed_rsp2_w_from_mlt_vc_if_needed};
pub use solver::{
    HenyeySolverReport, active_solver_tolerances, compute_corr_coeff,
    compute_correction_norms, compute_model_correction_norms,
    compute_residual_norms, compute_vector_norms, solve_henyey_step,
};
pub use star_ad::{
    I_HP_00, I_HP_M1, I_HP_P1, I_L_00, I_L_M1, I_L_P1, I_LND_00, I_LND_M1,
    I_LND_P1, I_LNR_00, I_LNR_M1, I_LNR_P1, I_LNT_00, I_LNT_M1, I_LNT_P1,
    I_OMEGA_00, I_OMEGA_M1, I_OMEGA_P1, I_V_00, I_V_M1, I_V_P1, I_W_00, I_W_M1,
    I_W_P1, NUM_STAR_AD_VARS, StarAdExt, StarAdOrder1,
};
pub use star_job_init::init_or_update_model_from_bundle;
pub use stella::{
    FeII5169EtaTable, StellaAbnZone, StellaExportOptions, StellaHydModel,
    StellaHydZone, StellaVelFeIIRow, eval_sobolev_feii_5169_snapshot,
    export_stella_pair, parse_stella_abn, parse_stella_hyd, parse_stella_lbol,
    run_stella_from_hyd_abn,
};
pub use struct_burn_mix::{
    do_struct_burn_mix, op_split_burn_zones, solve_composition_diffusion,
    solve_omega_mix,
};
pub use termination::{
    check_termination_limits, check_termination_limits_with_history,
    compute_core_masses_msun, do_check_limits, eval_core_infall_and_rebound_cms,
    eval_min_gamma1, eval_phase_of_evolution, get_ejecta_mass,
    get_remnant_mass, get_shock_mass_msun, set_phase_of_evolution,
    LimitCheckOutcome, PHASE_C_BURN, PHASE_HE_BURN, PHASE_IAMS, PHASE_NE_BURN,
    PHASE_O_BURN, PHASE_PREMS, PHASE_RELAX, PHASE_SI_BURN, PHASE_STARTING,
    PHASE_TACHEB, PHASE_TAMS, PHASE_TP_AGB, PHASE_WDCS, PHASE_ZACHEB,
    PHASE_ZAMS,
};
pub use timestep::{
    TimestepAction, TimestepLimitReason, TimestepReport,
    eval_dx_nuc_drop_ratio, eval_h211_factor, eval_varcontrol, filter_limit,
    pick_next_timestep,
};
pub use timestep_limits::timestep_controller;
pub use winds::{
    DutchLowTScheme, WindScheme, do_set_mdot, do_set_mdot_with_extras,
    eval_bjorklund_wind, eval_blocker_wind, eval_de_jager_wind,
    eval_dutch_wind, eval_nieuwenhuijzen_wind, eval_nugis_lamers_wind,
    eval_reimers_wind, eval_rlo_wind, eval_super_eddington_wind,
    eval_van_loon_wind, eval_vink_wind, set_mdot,
};
pub use zams::{
    ZamsCatalog, ZamsEntry, interp_q, load_zams_model,
    load_zams_model_from_catalog, yrs_for_init_timestep,
};
