// SPDX-License-Identifier: LGPL-3.0-or-later
//! Equation of State (`eos`) engine for `FlexMESA` (Wave 3a).
//!
//! Provides the canonical 26-slot thermodynamic result schema (`eos_def`),
//! ground-state nuclear binding and ionization energy offsets (`ion_offset`),
//! third-order dual Helmholtz free-energy reduction (`free_energy`),
//! analytical neutral ideal gas plus radiation (`ideal`), $C^2$-continuous
//! polygonal quintic blending (`blend`), Aparicio (1998) 20-point
//! Gauss-Legendre/Gauss-Laguerre relativistic Fermi-Dirac integrals
//! (`gauss_fermi`), Helmholtz biquintic/bicubic electron-positron EOS (`helm`),
//! Potekhin & Chabrier (`pc`) and `Skye` (`skye`) plasma/crystallization EOS,
//! hierarchical `eosDT_get` blending (`blend_dt`), and inverse `(P_gas, T)` /
//! $(\rho, E)$ solvers (`inverse`).

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
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unreadable_literal
)]

pub mod blend;
pub mod blend_dt;
pub mod free_energy;
pub mod gamma_de;
pub mod gauss_fermi;
pub mod helm;
pub mod ideal;
pub mod inverse;
pub mod ion_offset;
pub mod pc;
pub mod skye;
pub mod types;

pub use blend::{
    blend_eos_results, is_contained, is_contained_scalar,
    min_distance_from_point_to_line_segment, min_distance_to_polygon, quadrant,
    quintic_blend, winding_number,
};
pub use blend_dt::{
    combine_two_eos_components, eos_dt_get, eos_dt_get_from_weights,
    eos_dt_get_zones, get_helm_alfa, EosConfig,
};
pub use free_energy::{
    compute_derived_quantities, compute_f_ideal_ion, compute_f_rad,
    compute_xne, pack_for_export, sifac_constant,
    thermodynamics_from_free_energy, DerivedThermodynamics,
};
pub use gauss_fermi::{
    dfermi, dfermi_order1, DQLAG020_WG, DQLAG020_XG, DQLEG020_WG, DQLEG020_XG,
};
pub use helm::{
    do_convert_helm_results, eval_helm_coulomb, eval_helm_eos,
    eval_helm_eos_zones, h3, h5, helmeos2, CubicBasis1D, HelmCell, HelmConfig,
    HelmTable, HelmThermoVar, HelmVertex, QuinticBasis1D, NUM_HELM_RESULTS,
};
pub use ideal::{
    eval_ideal_eos, eval_ideal_eos_from_weights, IDEAL_MASS_FRACTION_LIMIT,
};
pub use inverse::{
    eos_dt_get_rho, eos_dt_get_t, eos_gamma_de_get, eos_pt_get,
    EosGammaDeOutput, EosInverseOutput, EosPtOutput,
};
pub use ion_offset::{
    compute_ion_offset, compute_ion_offset_from_za, NIST_IONIZATION_TABLE_EV,
};
pub use pc::{
    eval_pc_eos, eval_pc_eos_from_weights, eval_pc_eos_zones, get_pc_alfa,
    melange9, Melange9Output, PcConfig,
};
pub use skye::{
    decide_phase, eval_skye_eos, eval_skye_eos_from_weights,
    eval_skye_eos_zones, extrapolate_free_energy, get_skye_alfa,
    get_skye_alfa_simple, nonideal_corrections, ocp_free_energy, SkyeConfig,
    SkyeNonidealOutput, SkyeSolidMixingRule,
};
pub use types::{
    EosError, EosEvalOutput, EOS_BASIC_RESULT_NAMES, I_CHI_RHO, I_CHI_T, I_CP,
    I_CV, I_DE_DRHO, I_DS_DRHO, I_DS_DT, I_ETA, I_FRAC_CMS, I_FRAC_FREE_EOS,
    I_FRAC_HELM, I_FRAC_IDEAL, I_FRAC_OPAL_SCVH, I_FRAC_PC, I_FRAC_SKYE,
    I_GAMMA1, I_GAMMA3, I_GRAD_AD, I_LATENT_DDLNRHO, I_LATENT_DDLNT, I_LN_E,
    I_LN_FREE_E, I_LN_PGAS, I_LN_S, I_MU, I_PHASE, NUM_EOS_BASIC_RESULTS,
    NUM_EOS_D_DXA_RESULTS, NUM_EOS_FRAC_RESULTS,
};
