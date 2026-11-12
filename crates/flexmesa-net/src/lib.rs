// SPDX-License-Identifier: LGPL-3.0-or-later
//! Nuclear reaction networks, analytical Jacobians, `approx21`, and stiff
//! one-zone burners (`flexmesa-net`, Wave 3b.4).
//!
//! Ports upstream MESA `net` (`net_def.f90`, `net_lib.f90`, `net_initialize.f90`,
//! `net_eval.f90`, `net_derivs.f90`, `net_derivs_support.f90`, `net_screen.f90`,
//! `net_approx21.f90`, `net_burn.f90`, `net_burn_support.f90`, and
//! `net_burn_const_p.f90`) into Safe Rust (`#![forbid(unsafe_code)]`) with
//! IEEE-754 `f64` and `F64x2` extended-precision accumulation.

#![forbid(unsafe_code)]
#![allow(
    missing_docs,
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
    clippy::if_not_else,
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
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unreadable_literal
)]

pub mod approx21;
pub mod approx21_dfdy;
pub(crate) mod approx21_dfdy_heavy;
pub mod approx21_dydt;
pub mod approx21_eps;
pub mod approx21_eval;
pub mod approx21_special;
pub mod burn;
pub mod burn_const_density;
pub mod burn_const_p;
pub(crate) mod burn_stepper;
pub mod derivs;
pub mod derivs_paired;
pub mod derivs_support;
pub mod f64x2;
pub mod initialize;
pub mod parser;
pub mod reaction_synth;
pub(crate) mod rosenbrock_tableau;
pub mod screen;
pub mod screen_combo;
pub mod types;

pub use approx21::{
    approx21_pa_pg_fractions, approx21_weak_rates, Approx21Indices,
};
pub use approx21_eval::eval_net_approx21_procs;
pub use burn::{
    burn_1_zone, Burn1ZoneInput, Burn1ZoneOutput,
    Burn1ZoneOutput as OneZoneBurnResult,
};
pub use burn_const_density::{
    burn_1_zone_const_density, BurnConstDensityInput, BurnConstDensityOutput,
    BurnConstDensityOutput as OneZoneBurnConstDensityResult,
};
pub use burn_const_p::{
    burn_1_zone_const_p, BurnConstPInput, BurnConstPOutput,
    BurnConstPOutput as OneZoneBurnConstPResult,
};
pub use derivs::{
    eval_net, eval_net_with_work, get_derivs, with_thread_local_work,
};
pub use derivs_support::NetWorkState;
pub use f64x2::F64x2;
pub use initialize::{global_logttab, global_ttab};
pub use parser::{builtin_net_files, ParsedNetDef};
pub use reaction_synth::synthesize_reaction_info;
pub use screen::{init_screening_tables, screen_net};
pub use types::{
    NetError, NetEvalResult, NetGeneralInfo, ReaclibKind, ReactionKind,
    ScreenPairParams, APPROX21_CO56_NUM_MESA_REACTIONS,
    APPROX21_CO56_NUM_REACTIONS, APPROX21_NUM_MESA_REACTIONS,
    APPROX21_NUM_REACTIONS, MAX_SAFE_LOGT_FOR_RATES, NRATTAB, NUM_RVS,
    RATTAB_TEMP_LO, RATTAB_THI, RATTAB_TLO, RATTAB_TSTP,
};

use flexmesa_rates::ScreeningMode;

/// High-level single-zone network evaluation (`net_get` in `net/public/net_lib.f90`).
///
/// Evaluates nuclear energy generation `eps_nuc`, neutrino losses `eps_neu_total`,
/// category-resolved rates `eps_nuc_categories`, mass-fraction time derivatives
/// `dxdt`, and analytical Jacobians `(d_dxdt_dRho, d_dxdt_dT, d_dxdt_dx)` at a
/// single thermodynamic and composition state.
///
/// # Errors
/// Returns [`NetError`] if composition or rate evaluation fails.
pub fn net_get(
    g: &NetGeneralInfo,
    skip_jacobian: bool,
    x: &[f64],
    temp: f64,
    log_t: f64,
    rho: f64,
    log_rho: f64,
    abar: f64,
    zbar: f64,
    z2bar: f64,
    ye: f64,
    eta: f64,
    d_eta_dlnt: f64,
    d_eta_dlnrho: f64,
    rate_factors: &[f64],
    weak_rate_factor: f64,
    screening_mode: ScreeningMode,
) -> Result<NetEvalResult, NetError> {
    eval_net(
        g,
        false,
        skip_jacobian,
        x,
        temp,
        log_t,
        rho,
        log_rho,
        abar,
        zbar,
        z2bar,
        ye,
        eta,
        d_eta_dlnt,
        d_eta_dlnrho,
        rate_factors,
        weak_rate_factor,
        screening_mode,
        false,
        false,
        false,
        false,
    )
}

/// Evaluates the symbolic sparsity pattern of the mass-fraction Jacobian
/// $\partial(\mathrm{d}X_i/\mathrm{d}t)/\partial X_j$ (`net_get_symbolic_d_dxdt_dx`
/// in `net/public/net_lib.f90`).
///
/// # Errors
/// Returns [`NetError`] if network evaluation fails.
pub fn net_get_symbolic_d_dxdt_dx(
    g: &NetGeneralInfo,
    x: &[f64],
    temp: f64,
    log_t: f64,
    rho: f64,
    log_rho: f64,
    abar: f64,
    zbar: f64,
    z2bar: f64,
    ye: f64,
    eta: f64,
    d_eta_dlnt: f64,
    d_eta_dlnrho: f64,
    rate_factors: &[f64],
    weak_rate_factor: f64,
    screening_mode: ScreeningMode,
) -> Result<Vec<f64>, NetError> {
    let res = eval_net(
        g,
        false,
        false,
        x,
        temp,
        log_t,
        rho,
        log_rho,
        abar,
        zbar,
        z2bar,
        ye,
        eta,
        d_eta_dlnt,
        d_eta_dlnrho,
        rate_factors,
        weak_rate_factor,
        screening_mode,
        false,
        false,
        false,
        true,
    )?;
    Ok(res.d_dxdt_dx)
}
