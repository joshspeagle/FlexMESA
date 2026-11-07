// SPDX-License-Identifier: LGPL-3.0-or-later
//! Nuclear reaction rates, plasma electrostatic screening, tabulated weak
//! reactions, and on-the-fly relativistic Fermi-Dirac electron captures
//! (`rates`, Wave 3b.3).
//!
//! Ports upstream MESA `rates/public/rates_def.f90`, `rates/public/rates_lib.f90`,
//! and `rates/private/{ratelib,raw_rates,reaclib_input,reaclib_eval,screen,screen5,
//! screen_chugunov,pycno,load_weak,eval_weak,suzuki_tables,coulomb,eval_psi,
//! eval_ecapture}.f90` to Safe Rust (`#![forbid(unsafe_code)]`).

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
    clippy::inline_always,
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unreadable_literal
)]

pub mod ecapture;
pub mod loader;
pub mod ratelib;
pub mod raw_rates;
pub mod screen;
pub mod screen_chugunov;
pub mod tfactors;
pub mod types;
pub mod weak;

pub use ecapture::{
    eval_ecapnuc_rate, eval_ecapture_pair, eval_mui_coulomb, eval_vs_coulomb,
};
pub use loader::{RatesCatalog, SuzukiTable, WeakTable};
pub use ratelib::eval_analytic_rate;
pub use raw_rates::{
    eval_fl_epsnuc_3alf, eval_interp_rate_table, eval_reaclib_handle,
    eval_reaction_raw, set_raw_rate, set_raw_rates,
};
pub use screen::{screen_extended, screen_pair, screen_salpeter};
pub use screen_chugunov::screen_chugunov;
pub use tfactors::{
    compute_rev_ratio_coeff1, eval_reaclib_terms, get_partition_fcn_indx,
    TFactors, TPART,
};
pub use types::{
    CoulombMuiOption, CoulombVsOption, RatesError, ReaclibTerm, ReactionInfo,
    ScreenEval, ScreeningMode, WeakRateEval, DEFAULT_REACLIB_MIN_T9,
    LOW_T9_PP_CUTOFF, MAX_LN_LAMBDA, NUM_BUNDLED_REACTIONS,
    NUM_PREDEFINED_REACTIONS,
};
pub use weak::{
    bilinear_interp_2d, eval_suzuki_one, eval_weak_reaction_pair,
    eval_weaklib_one,
};
