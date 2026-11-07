// SPDX-License-Identifier: LGPL-3.0-or-later
//! Surface boundary conditions, $T(\tau)$ relations, irradiated atmospheres, and
//! atmosphere table interpolation (`flexmesa-atm`).
//!
//! Ports upstream MESA `atm` (`atm_def.f90`, `atm_lib.f90`, `atm_t_tau_relations.f90`,
//! `atm_utils.f90`, `table_atm.f90`, `atm_table.f90`, `atm_t_tau_uniform.f90`,
//! `atm_t_tau_varying.f90`, and `atm_irradiated.f90`) into Safe Rust in `f64`
//! precision.

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
    clippy::imprecise_flops,
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

pub mod irradiated;
pub mod t_tau_relations;
pub mod t_tau_uniform;
pub mod t_tau_varying;
pub mod table;
pub mod types;
pub mod utils;

pub use irradiated::eval_irradiated;
pub use t_tau_relations::{eval_t_tau, eval_t_tau_dq_dtau, get_t_tau_base};
pub use t_tau_uniform::{build_t_tau_uniform, eval_t_tau_uniform};
pub use t_tau_varying::{build_t_tau_varying, eval_t_tau_varying};
pub use table::{
    eval_table, get_table_alfa_beta, get_table_base, get_table_values,
    AtmTableInfo, AtmTables,
};
pub use types::{
    AtmEosCallbackResult, AtmError, AtmEvalOutput, AtmIrradiatedOutput,
    AtmKapCallbackResult, AtmStructure, ATM_CHIRHO, ATM_CHIT, ATM_CP, ATM_CV,
    ATM_DELTA_R, ATM_DLNKAP_DLND, ATM_DLNKAP_DLNT, ATM_GAMMA1, ATM_GRADA,
    ATM_GRADR, ATM_GRADT, ATM_KAP, ATM_LND, ATM_LNFREE_E, ATM_LNP, ATM_LNPGAS,
    ATM_LNT, ATM_TABLE_DB_WD_TAU_25, ATM_TABLE_PHOTOSPHERE, ATM_TABLE_TAU_1,
    ATM_TABLE_TAU_10, ATM_TABLE_TAU_100, ATM_TABLE_TAU_1M1,
    ATM_TABLE_WD_TAU_25, ATM_TAU, ATM_T_TAU_EDDINGTON, ATM_T_TAU_KRISHNA_SWAMY,
    ATM_T_TAU_SOLAR_HOPF, ATM_T_TAU_TRAMPEDACH_SOLAR, ATM_XM,
    NUM_RESULTS_FOR_BUILD_ATM, TABLE_ATM_VERSION,
};
pub use utils::{
    atm_black_body_t, atm_l, atm_teff, eval_e2, eval_paczynski_gradr,
    eval_teff_g, radiation_pressure,
};
