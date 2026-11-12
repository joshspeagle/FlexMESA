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

pub mod case_staging;
pub mod controls;
mod controls_defaults;
pub mod diffusion;
pub mod eps_grav;
pub mod evolve;
pub mod history;
pub mod hydro_brunt_surf;
pub mod hydro_energy;
pub mod hydro_eqns;
pub mod hydro_momentum;
pub mod hydro_rti;
pub mod hydro_state;
pub mod hydro_temperature;
pub mod hydro_vars;
pub mod inlist;
pub mod inlist_populate;
mod inlist_populate_limits;
pub mod inlist_runner;
pub mod job_controls;
pub mod mesh;
pub mod mesh_adjust;
pub mod micro;
pub mod micro_net;
pub mod mix_info;
pub mod model;
pub mod model_io;
pub mod overshoot;
pub mod photo_io;
pub mod pre_ms;
pub mod premix;
pub mod pulse;
pub mod pulse_bvp;
pub mod pulse_bvp_recon;
pub mod pulse_exporters;
pub mod pulse_modes;
pub mod pulse_utils;
pub mod relax;
pub mod riemann;
pub mod riemann_ad;
pub mod rotation;
pub mod rsp2;
pub mod rsp2_ad;
pub mod rsp2_eqns;
pub mod rsp2_remesh;
pub mod rsp_build;
pub mod rsp_config;
pub mod rsp_crossing;
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
pub mod star_ad;
pub mod star_job_init;
pub mod stella;
pub mod struct_burn_mix;
pub mod termination;
pub mod tier1_extras;
pub mod tier2_extras;
pub mod tier3_extras;
pub mod tier3_physics;
pub mod tier4_extras;
pub mod tier5_extras;
pub mod timestep;
pub mod winds;
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
#[cfg(test)]
mod tests_wave5;

pub use controls::{
    AtmOption, AtmTTauRelation, EnergyEqnOption, StarControls, StarError,
    StarJobControls, VarLayout,
};
pub use diffusion::{
    do_element_diffusion, eval_burgers_velocities, eval_caplan_ocp_dstar,
    eval_coulomb_collision_factors, eval_radiative_levitation_accel,
    eval_wd_sedimentation_heating, BurgersZoneResult, DiffusionScheme,
};
pub use eps_grav::{
    eval_eps_grav, eval_eps_grav_lns_cell, eval_eps_grav_std_cell,
};
pub use evolve::{
    restore_from_old, star_evolve, star_evolve_step, EvolveStepReport,
};
pub use history::{
    extract_history_row, format_history_header, format_history_row,
    format_profile_string, HistoryRow,
};
pub use hydro_energy::{
    eval_cell_pressure_for_work_ad, eval_face_area_time_centered_ad,
    eval_pvsc_cell_ad, setup_d_turbulent_energy_dt, setup_dl_dm,
    setup_sources_and_others,
};
pub use hydro_eqns::{
    compute_x_scale, do1_chem_eqns, do1_density_eqn, do1_energy_eqn,
    do1_momentum_eqn, do1_radius_eqn, do1_temperature_eqn, do_surf_bc_eqns,
    eval_henyey_equations, HenyeySystem,
};
pub use hydro_momentum::{
    eval_cell_peos_time_centered_ad, eval_face_dm_bar, expected_hse_grav_term,
    expected_non_hse_term,
};
pub use hydro_rti::{
    do1_dalpha_rti_dt_eqn, eval_alpha_rti_sources, set_dpdr_drhodr_info,
    set_rti_mixing_info,
};
pub use hydro_state::HydroAuxState;
pub use hydro_temperature::{interp_p_point_ad, set_rsp_lsurf_bc};
pub use hydro_vars::{
    check_for_redo_mlt, do_brunt_b, do_brunt_n2, eval_surf_bc, set_hydro_vars,
    set_mlt_vars, update_sig_from_d_mix,
};
pub use inlist::{parse_rn_stages, InlistBundle, RnStage, SectionControls};
pub use inlist_runner::{
    run_case_list, run_inlist_stage, run_test_suite_case,
    verify_photo_restart_parity, CaseRunReport, DefaultStarExtras,
    ExtrasDirective, StageRunReport, StarExtras, TovGrStarExtras,
};
pub use mesh_adjust::{
    do_adjust_mesh, eval_mesh_gval, plan_mesh, remap_onto_dq,
    MeshFunctionConfig,
};
pub use micro::{
    change_model_network, do_eos, do_kap, do_net, do_neu, get_or_load_net,
    get_tau, global_basic_net, set_basic_vars,
};
pub use mix_info::{
    cdc_factor, clean_mixing_glitches, compute_convection_sigmas,
    eval_dxdt_mix, face_density, locate_convection_boundaries, ConvBoundary,
};
pub use model::StarModel;
pub use model_io::{
    compute_photo_checksum, read_mod_str, read_photo_bytes, write_mod_string,
    write_photo_bytes, STAR_DEF_VERSION,
};
pub use overshoot::{apply_overshooting, OvershootConfig, OvershootScheme};
pub use pre_ms::{
    build1_pre_ms_model, build_pre_ms_model, default_basic_net_composition,
    PreMsShell,
};
pub use premix::{
    add_predictive_mixing, do_conv_premix, do_phase_separation,
    eval_castellani_penetration_vel, eval_phase_separation_shift,
};
pub use pulse::{
    build_pulse_points, build_pulse_points_with_options, eval_center_d2,
    eval_center_parabolic, eval_center_rho, eval_face, export_fgong,
    export_fgong_with_version, export_gyre, export_gyre_with_schema,
    export_osc, set_segment_indices, PulseExportOptions, PulsePoint,
    PULSE_ISO_NAMES,
};
pub use pulse_bvp::{
    shoot_determinant, steffen_slopes, BvpEigenmode, BvpModel, BvpPoint,
    InertiaNormKind, OuterBoundKind,
};
pub use pulse_bvp_recon::{
    classify_mode_orders, parse_fgong_model, parse_fortran_float,
    parse_gyre_mesa_model, reconstruct_and_analyze_mode,
};
pub use pulse_exporters::{
    export_cafein, export_gr1d, export_gsm, export_gsm_data,
    export_pulse_by_format, export_saio, GsmData,
};
pub use pulse_modes::{
    build_bvp_model_from_pulse_points, eval_acoustic_large_separation_hz,
    eval_astero_scaling_dnu_numax_uhz, eval_buoyancy_period_spacing_s,
    eval_kappa_mechanism_growth, eval_linear_oscillation_modes,
    eval_target_mode_frequency_and_growth, solve_adipls_fgong_modes,
    solve_bvp_eigenmodes, solve_gyre_mesa_modes, LinearOscillationMode,
};
pub use relax::{
    apply_composition_relaxation, apply_uniform_rotation, relax_core_boundary,
    relax_from_profile_files, relax_irradiation_to_target,
    relax_mass_to_target, relax_tau_factor_to_target, relax_to_remove_h_env,
};
pub use riemann::{
    eval_hllc_face, eval_hllc_surface_face, eval_riemann_momentum_residual,
    eval_riemann_profile, HllcFaceState,
};
pub use riemann_ad::{
    calc_ptot_cell_ad, do1_riemann_momentum_eqn, do_uface_and_pface,
};
pub use rotation::{
    eval_magnetic_diffusivity, eval_omega_crit, eval_rotation_fp_ft,
    eval_rotational_mixing_zone, eval_thermal_diffusivity,
    set_rotation_mixing_info, update_rotation_distortion_factors,
    RotationalMixZone, RI_CRIT_DSI,
};
pub use rsp2::{
    compute_rsp2_uq_face, eval_rsp2_face, eval_rsp2_zone, rsp2_flux_limiter,
    solve_stationary_rsp2_w, Rsp2Config, Rsp2FaceResult, Rsp2ZoneResult,
    X_ALFAC, X_ALFAP, X_ALFAS, X_CEDE, X_GAMMAR,
};
pub use rsp2_ad::{
    eval_and_save_rsp2_eqns, eval_rsp2_hp_face_ad, eval_rsp2_ptrb_cell_ad,
    update_rsp2_vars_ad,
};
pub use rsp2_eqns::{
    eval_rsp2_hp_face_zone, eval_rsp2_profile_equations,
    rsp2_adjust_vars_before_call_solver, Rsp2ProfileResiduals,
};
pub use rsp2_remesh::{remesh_for_rsp2, seed_rsp2_w_from_mlt_vc_if_needed};
pub use rsp_build::{
    build_rsp_envelope_model, do_rsp_build_model, eval_rsp_2nd_crossing_edges,
    eval_rsp_2nd_crossing_with_controls, eval_rsp_lina_modes,
    Rsp2ndCrossingOffset, Rsp2ndCrossingPoint, Rsp2ndCrossingReport,
    RspLinaMode,
};
pub use rsp_config::{Rsp2Controls, RspConfig};
pub use rsp_def::{
    RspDerivedConsts, RspState, F_EDD_FREE_STREAM, F_EDD_ISOTROPIC, MAX_RSP_NZN,
};
pub use rsp_driver::{
    rsp_one_step, rsp_setup_model, rsp_total_energy_integrals,
};
pub use rsp_eos_kap::{
    compute_standard_metals_xa, eval1_gamma_pt_get_rho, eval1_mesa_eos_and_kap,
    eval1_mesa_eos_de_and_kap, eval1_mesa_eos_degas_and_kap,
    eval1_mesa_rho_given_pt, eval1_mesa_t_given_dp, eval_mesa_eos_and_kap,
    get_surf_p_t_kap, set_rho_for_new_pgas, set_t_for_new_egas,
    set_t_for_new_energy, set_t_for_new_pgas, RspChemCache, RspEosKapOut,
};
pub use rsp_grid::{
    cleanup_for_lina, copy_results, eop, finish_build_rsp_model, mesa_eos_kap,
    MesaRanmar, RspEnvGrid, RspTotalEosKap,
};
pub use rsp_lina::{do_lina, RspLinaOutput};
pub use rsp_relax_env::relax_env;
pub use rsp_stah::stah;
pub use solver::{
    active_solver_tolerances, compute_corr_coeff, compute_correction_norms,
    compute_model_correction_norms, compute_residual_norms,
    compute_vector_norms, solve_henyey_step, HenyeySolverReport,
};
pub use star_ad::{
    StarAdExt, StarAdOrder1, I_HP_00, I_HP_M1, I_HP_P1, I_LND_00, I_LND_M1,
    I_LND_P1, I_LNR_00, I_LNR_M1, I_LNR_P1, I_LNT_00, I_LNT_M1, I_LNT_P1,
    I_L_00, I_L_M1, I_L_P1, I_OMEGA_00, I_OMEGA_M1, I_OMEGA_P1, I_V_00, I_V_M1,
    I_V_P1, I_W_00, I_W_M1, I_W_P1, NUM_STAR_AD_VARS,
};
pub use star_job_init::init_or_update_model_from_bundle;
pub use stella::{
    eval_sobolev_feii_5169_snapshot, export_stella_pair, parse_stella_abn,
    parse_stella_hyd, parse_stella_lbol, run_stella_from_hyd_abn,
    FeII5169EtaTable, StellaAbnZone, StellaExportOptions, StellaHydModel,
    StellaHydZone, StellaVelFeIIRow,
};
pub use struct_burn_mix::{
    do_struct_burn_mix, op_split_burn_zones, solve_composition_diffusion,
    solve_omega_mix,
};
pub use termination::{
    check_termination_limits, check_termination_limits_with_history,
    compute_core_masses_msun,
};
pub use tier1_extras::{
    eval_model_colors, eval_solar_calibration_objective,
    eval_starspot_effective_luminosity, eval_t_tau_gradr_factor, is_tier1_case,
    nelder_mead_solar_calibration_step, run_tier1_suite,
    verify_tier1_case_assertions, Tier1CaseExtras, TIER1_CASES,
};
pub use tier2_extras::{
    apply_accretion_composition_and_nova_wind, apply_planetary_core,
    eval_conductive_flame_state, eval_planetary_irradiation_heating,
    eval_wd_phase_separation, is_tier2_case, run_tier2_suite,
    verify_tier2_case_assertions, Tier2CaseExtras, TIER2_CASES,
};
pub use tier3_extras::{
    compute_total_angular_momentum, is_tier3_case, run_tier3_suite,
    verify_tier3_case_assertions, Tier3CaseExtras, TIER3_CASES,
};
pub use tier3_physics::{
    eval_duffell_rti_shock_state, eval_fe_core_infall_state,
    eval_hot_cool_blended_wind_mdot, eval_magnetic_braking_torque,
    eval_pair_instability_gamma1_and_energy, eval_roche_gravity_darkening,
    eval_spruit_tayler_dynamo_diagnostics, eval_tov_effective_cgrav,
    eval_tzo_core_accretion_step, export_stella_ascii,
};
pub use tier4_extras::{
    is_tier4_case, run_astero_suite, run_astero_test_suite_case,
    run_dev_cases_suite, run_tier4_star_suite, run_tier4_suite,
    verify_tier4_case_assertions, Tier4CaseExtras, TIER4_ACTIVE_STAR_CASES,
    TIER4_ASTERO_CASES, TIER4_DEV_CASES, TIER4_STAR_CASES,
};
pub use tier5_extras::{
    all_active_star_case_names, eval_grafener_kudritzki_wind_msun_yr,
    eval_overshoot_dbl_exp, eval_xtrans_mesh_factor, is_tier5_star_case,
    run_all_active_star_suite, run_tier5_star_suite,
    verify_step_redo_bit_parity, verify_tier5_star_case_assertions,
    write_relax_profile_files, Tier5StarCaseExtras, TIER5_STAR_CASES,
};
pub use timestep::{
    eval_dx_nuc_drop_ratio, eval_h211_factor, eval_varcontrol, filter_limit,
    pick_next_timestep, TimestepLimitReason, TimestepReport,
};
pub use winds::{
    do_adjust_mass, eval_blocker_wind, eval_de_jager_wind, eval_dutch_wind,
    eval_nieuwenhuijzen_wind, eval_nugis_lamers_wind, eval_reimers_wind,
    eval_van_loon_wind, eval_vink_wind, set_mdot, DutchLowTScheme, WindScheme,
};
pub use zams::{
    interp_q, load_zams_model, load_zams_model_from_catalog,
    yrs_for_init_timestep, ZamsCatalog, ZamsEntry,
};
