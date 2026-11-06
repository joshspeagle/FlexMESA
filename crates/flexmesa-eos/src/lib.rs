// SPDX-License-Identifier: LGPL-3.0-or-later
//! Equation of State (`eos`) engine for `FlexMESA` (Wave 3a).
//!
//! Provides the canonical 26-slot thermodynamic result schema (`eos_def`),
//! ground-state nuclear binding and ionization energy offsets (`ion_offset`),
//! third-order dual Helmholtz free-energy reduction (`free_energy`),
//! analytical neutral ideal gas plus radiation (`ideal`), $C^2$-continuous
//! polygonal quintic blending (`blend`), and Aparicio (1998) 20-point
//! Gauss-Legendre/Gauss-Laguerre relativistic Fermi-Dirac integrals
//! (`gauss_fermi`).

#![forbid(unsafe_code)]
#![allow(
    clippy::approx_constant,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::similar_names,
    clippy::unreadable_literal
)]

pub mod blend;
pub mod free_energy;
pub mod gauss_fermi;
pub mod ideal;
pub mod ion_offset;
pub mod types;

pub use blend::{
    blend_eos_results, is_contained, is_contained_scalar,
    min_distance_from_point_to_line_segment, min_distance_to_polygon, quadrant,
    quintic_blend, winding_number,
};
pub use free_energy::{
    compute_derived_quantities, compute_f_ideal_ion, compute_f_rad,
    compute_xne, pack_for_export, sifac_constant,
    thermodynamics_from_free_energy, DerivedThermodynamics,
};
pub use gauss_fermi::{
    dfermi, dfermi_order1, DQLAG020_WG, DQLAG020_XG, DQLEG020_WG, DQLEG020_XG,
};
pub use ideal::{
    eval_ideal_eos, eval_ideal_eos_from_weights, IDEAL_MASS_FRACTION_LIMIT,
};
pub use ion_offset::{
    compute_ion_offset, compute_ion_offset_from_za, NIST_IONIZATION_TABLE_EV,
};
pub use types::{
    EosError, EosEvalOutput, EOS_BASIC_RESULT_NAMES, I_CHI_RHO, I_CHI_T, I_CP,
    I_CV, I_DE_DRHO, I_DS_DRHO, I_DS_DT, I_ETA, I_FRAC_CMS, I_FRAC_FREE_EOS,
    I_FRAC_HELM, I_FRAC_IDEAL, I_FRAC_OPAL_SCVH, I_FRAC_PC, I_FRAC_SKYE,
    I_GAMMA1, I_GAMMA3, I_GRAD_AD, I_LATENT_DDLNRHO, I_LATENT_DDLNT, I_LN_E,
    I_LN_FREE_E, I_LN_PGAS, I_LN_S, I_MU, I_PHASE, NUM_EOS_BASIC_RESULTS,
    NUM_EOS_D_DXA_RESULTS, NUM_EOS_FRAC_RESULTS,
};
