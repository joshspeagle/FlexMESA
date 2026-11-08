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
//! - **State & I/O ([`model`], [`controls`], [`mesh`], [`model_io`], [`zams`], [`pre_ms`])**:
//!   Canonical [`StarModel`] representation, staggered Lagrangian mass grid,
//!   ZAMS interpolation, $n=1.5$ pre-MS polytrope builder, and `.mod`/`.photo`
//!   serialization.
//! - **Microphysics & Face Transport ([`micro`], [`hydro_vars`], [`eps_grav`])**:
//!   Parallel zone EOS, opacity, neutrino, and nuclear reaction network calls,
//!   MLT/TDC convection, Brunt-Väisälä buoyancy, and surface boundary conditions.
//! - **Henyey Equations & Solver ([`hydro_eqns`], [`solver`])**:
//!   Coupled hydrodynamic and species conservation equations, analytical
//!   block-tridiagonal Jacobian assembly, and damped Newton-Raphson solver.
//! - **Mixing, Burning & Winds ([`mix_info`], [`overshoot`], [`struct_burn_mix`], [`winds`])**:
//!   Sub-cell convective boundary detection, step/exponential overshooting,
//!   operator-split and coupled burning/diffusion, rotational angular momentum
//!   diffusion, and stellar wind mass adjustment.

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

pub mod controls;
pub mod eps_grav;
pub mod hydro_eqns;
pub mod hydro_vars;
pub mod mesh;
pub mod micro;
pub mod mix_info;
pub mod model;
pub mod model_io;
pub mod overshoot;
pub mod pre_ms;
pub mod solver;
pub mod star_ad;
pub mod struct_burn_mix;
pub mod winds;
pub mod zams;

#[cfg(test)]
mod tests;

pub use controls::{
    AtmOption, AtmTTauRelation, EnergyEqnOption, StarControls, StarError,
    StarJobControls, VarLayout,
};
pub use eps_grav::{
    eval_eps_grav, eval_eps_grav_lns_cell, eval_eps_grav_std_cell,
};
pub use hydro_eqns::{
    compute_x_scale, do1_chem_eqns, do1_density_eqn, do1_energy_eqn,
    do1_momentum_eqn, do1_radius_eqn, do1_temperature_eqn, do_surf_bc_eqns,
    eval_henyey_equations, HenyeySystem,
};
pub use hydro_vars::{
    check_for_redo_mlt, do_brunt_b, do_brunt_n2, eval_surf_bc, set_hydro_vars,
    set_mlt_vars, update_sig_from_d_mix,
};
pub use micro::{
    do_eos, do_kap, do_net, do_neu, get_tau, global_basic_net, set_basic_vars,
};
pub use mix_info::{
    cdc_factor, clean_mixing_glitches, compute_convection_sigmas,
    eval_dxdt_mix, face_density, locate_convection_boundaries, ConvBoundary,
};
pub use model::StarModel;
pub use model_io::{
    read_mod_str, read_photo_bytes, write_mod_string, write_photo_bytes,
    STAR_DEF_VERSION,
};
pub use overshoot::{apply_overshooting, OvershootConfig, OvershootScheme};
pub use pre_ms::{
    build1_pre_ms_model, build_pre_ms_model, default_basic_net_composition,
    PreMsShell,
};
pub use solver::{
    compute_corr_coeff, compute_correction_norms, compute_vector_norms,
    solve_henyey_step, HenyeySolverReport,
};
pub use star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LND_M1, I_LND_P1, I_LNR_00, I_LNR_M1,
    I_LNR_P1, I_LNT_00, I_LNT_M1, I_LNT_P1, I_L_00, I_L_M1, I_L_P1, I_OMEGA_00,
    I_OMEGA_M1, I_OMEGA_P1, I_V_00, I_V_M1, I_V_P1, NUM_STAR_AD_VARS,
};
pub use struct_burn_mix::{
    do_struct_burn_mix, op_split_burn_zones, solve_composition_diffusion,
    solve_omega_mix,
};
pub use winds::{
    do_adjust_mass, eval_blocker_wind, eval_de_jager_wind, eval_dutch_wind,
    eval_nieuwenhuijzen_wind, eval_reimers_wind, eval_van_loon_wind,
    eval_vink_wind, set_mdot, DutchLowTScheme, WindScheme,
};
pub use zams::{
    interp_q, load_zams_model, load_zams_model_from_catalog,
    yrs_for_init_timestep, ZamsCatalog, ZamsEntry,
};
