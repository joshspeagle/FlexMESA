// SPDX-License-Identifier: LGPL-3.0-or-later
//! Safe Rust radiative, conductive, and Compton opacity (`kap`) crate for `FlexMESA`.
//!
//! Ports upstream Fortran MESA's `kap` module (`kap_def.f90`, `kap_lib.f90`,
//! `kap_eval.f90`, `kap_eval_fixed.f90`, `kap_eval_co.f90`, `kap_eval_support.f90`,
//! `condint.f90`, `kap_aesopus.f90`, `kapcn.f90`, `load_kap.f90`, and `load_co_kap.f90`):
//! * Fixed-metal Type 1 (`gs98`, `gn93`, `a09`, `OP_gs98`) and Ferguson/Freedman low-$T$
//!   (`lowT_fa05_gs98`) 2D PSPLINE bicubic interpolation in $(\log_{10} R, \log_{10} T)$
//!   with 4-point Steffen (1990) piecewise-monotonic cubic interpolation in $X$ and $Z$.
//! * C/O-enhanced Type 2 (`gn93_co`, `gs98_co`, `a09_co`) 58-table $(\delta X_{\mathrm{C}},
//!   \delta X_{\mathrm{O}})$ convex-hull interpolation and smooth Type 1/Type 2 blending.
//! * Variable C/N/O low-$T$ molecular/dust opacities via Marigo & Aringer (2009)
//!   `AESOPUS` (`AESOPUS_AGSS09.h5`) and Lederer & Aringer (2009) `kapCN` (`kR_Z*.data`).
//! * Potekhin (1999) / Cassisi et al. (2007) electron thermal conduction (`condtabl.data`)
//!   with the Blouin et al. (2020) moderate-coupling H/He analytical correction.
//! * Poutanen (2017) relativistic/degenerate Compton scattering opacity.

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
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unreadable_literal
)]

pub mod blend;
pub mod co;
pub(crate) mod co_find;
pub mod compton;
pub mod conduction;
pub mod eval;
pub mod fixed;
pub mod loader;
pub(crate) mod loader_parse;
pub mod low_t;
pub mod spline;
pub mod types;

pub use co::get1_kap_co_results;
pub use compton::compton_opacity;
pub use conduction::{
    do_electron_conduction, do_electron_conduction_blouin,
    do_electron_conduction_potekhin,
};
pub use eval::{
    combine_rad_with_conduction, get_kap_high_t_results, get_kap_low_t_results,
    get_kap_results, get_kap_results_blend_t, kap_get, kap_get_zones,
};
pub use fixed::get1_kap_fixed_metal_results;
pub use low_t::{aesopus_get, kapcn_get};
pub use types::{
    KapCoOption, KapConfig, KapError, KapEvalOutput, KapLowTOption, KapOption,
    I_FRAC_COMPTON, I_FRAC_HIGH_T, I_FRAC_LOW_T, I_FRAC_TYPE2, NUM_KAP_FRACS,
};
