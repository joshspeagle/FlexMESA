// SPDX-License-Identifier: LGPL-3.0-or-later
//! Core equation-of-state (`eos`) result indices, output containers, and error
//! types (`eos/public/eos_def.f90`).
//!
//! Translates MESA's 1-based `num_eos_basic_results = 26` vector (`i_lnPgas = 1
//! .. i_frac_ideal = 26`) into a 0-based C-contiguous `[f64; 26]` array while
//! preserving exact physical channel ordering.

use std::error::Error;
use std::fmt::{self, Display};

/// Number of basic thermodynamic and blend-fraction channels returned by
/// `eosDT_get` (`num_eos_basic_results = 26`).
pub const NUM_EOS_BASIC_RESULTS: usize = 26;

/// Number of partial derivatives with respect to species mass fractions $X_k$
/// (`i_lnPgas` and `i_lnE`, `num_eos_d_dxa_results = 2`).
pub const NUM_EOS_D_DXA_RESULTS: usize = 2;

/// Number of EOS component blend-fraction channels
/// (`num_eos_frac_results = 7`).
pub const NUM_EOS_FRAC_RESULTS: usize = 7;

/// 0-based index for $\ln P_{\mathrm{gas}}$ (`i_lnPgas = 1` in Fortran).
pub const I_LN_PGAS: usize = 0;
/// 0-based index for $\ln E$ (`i_lnE = 2` in Fortran).
pub const I_LN_E: usize = 1;
/// 0-based index for $\ln S$ (`i_lnS = 3` in Fortran).
pub const I_LN_S: usize = 2;
/// 0-based index for mean molecular weight $\mu$ (`i_mu = 4` in Fortran).
pub const I_MU: usize = 3;
/// 0-based index for $\ln Y_e^{\mathrm{free}}$ (`i_lnfree_e = 5` in Fortran).
pub const I_LN_FREE_E: usize = 4;
/// 0-based index for electron degeneracy $\eta_e = \mu_e / (k_{\mathrm{B}} T)$
/// (`i_eta = 6` in Fortran).
pub const I_ETA: usize = 5;
/// 0-based index for adiabatic gradient $\nabla_{\mathrm{ad}}$
/// (`i_grad_ad = 7`).
pub const I_GRAD_AD: usize = 6;
/// 0-based index for density susceptibility $\chi_\rho$ (`i_chiRho = 8`).
pub const I_CHI_RHO: usize = 7;
/// 0-based index for temperature susceptibility $\chi_T$ (`i_chiT = 9`).
pub const I_CHI_T: usize = 8;
/// 0-based index for isobaric specific heat $C_P$ (`i_Cp = 10`).
pub const I_CP: usize = 9;
/// 0-based index for isochoric specific heat $C_V$ (`i_Cv = 11`).
pub const I_CV: usize = 10;
/// 0-based index for $(\partial E / \partial \rho)_T$ (`i_dE_dRho = 12`).
pub const I_DE_DRHO: usize = 11;
/// 0-based index for $(\partial S / \partial T)_\rho$ (`i_dS_dT = 13`).
pub const I_DS_DT: usize = 12;
/// 0-based index for $(\partial S / \partial \rho)_T$ (`i_dS_dRho = 14`).
pub const I_DS_DRHO: usize = 13;
/// 0-based index for first adiabatic exponent $\Gamma_1$ (`i_gamma1 = 15`).
pub const I_GAMMA1: usize = 14;
/// 0-based index for third adiabatic exponent $\Gamma_3$ (`i_gamma3 = 16`).
pub const I_GAMMA3: usize = 15;
/// 0-based index for crystallization phase $\phi \in [0, 1]$ (`i_phase = 17`).
pub const I_PHASE: usize = 16;
/// 0-based index for latent heat temperature derivative
/// $T \partial S / \partial \ln T$ (`i_latent_ddlnT = 18`).
pub const I_LATENT_DDLNT: usize = 17;
/// 0-based index for latent heat density derivative $T \partial S / \partial
/// \ln \rho$ (`i_latent_ddlnRho = 19`).
pub const I_LATENT_DDLNRHO: usize = 18;
/// 0-based index for `HELM` blend fraction (`i_frac_HELM = 20`).
pub const I_FRAC_HELM: usize = 19;
/// 0-based index for `OPAL/SCVH` blend fraction (`i_frac_OPAL_SCVH = 21`).
pub const I_FRAC_OPAL_SCVH: usize = 20;
/// 0-based index for `FreeEOS` blend fraction (`i_frac_FreeEOS = 22`).
pub const I_FRAC_FREE_EOS: usize = 21;
/// 0-based index for `PC` blend fraction (`i_frac_PC = 23`).
pub const I_FRAC_PC: usize = 22;
/// 0-based index for `Skye` blend fraction (`i_frac_Skye = 24`).
pub const I_FRAC_SKYE: usize = 23;
/// 0-based index for `CMS` blend fraction (`i_frac_CMS = 25`).
pub const I_FRAC_CMS: usize = 24;
/// 0-based index for `ideal` blend fraction (`i_frac_ideal = 26`).
pub const I_FRAC_IDEAL: usize = 25;

/// Canonical MESA channel names corresponding to indices `0..26`.
pub const EOS_BASIC_RESULT_NAMES: [&str; NUM_EOS_BASIC_RESULTS] = [
    "lnPgas",
    "lnE",
    "lnS",
    "mu",
    "lnfree_e",
    "eta",
    "grad_ad",
    "chiRho",
    "chiT",
    "Cp",
    "Cv",
    "dE_dRho",
    "dS_dT",
    "dS_dRho",
    "gamma1",
    "gamma3",
    "phase",
    "latent_ddlnT",
    "latent_ddlnRho",
    "HELM",
    "OPAL/SCVH",
    "FreeEOS",
    "PC",
    "Skye",
    "CMS",
    "ideal",
];

/// Complete output of an equation-of-state evaluation at a single thermodynamic
/// state $(\rho, T, \{X_k\})$.
#[derive(Clone, Debug, PartialEq)]
pub struct EosEvalOutput {
    /// Basic 26-channel thermodynamic state vector `res[0..26]`.
    pub res: [f64; NUM_EOS_BASIC_RESULTS],
    /// Partial derivatives $(\partial\mathrm{res} / \partial\ln\rho)_{T, X}$.
    pub d_dlnd: [f64; NUM_EOS_BASIC_RESULTS],
    /// Partial derivatives $(\partial\mathrm{res} / \partial\ln T)_{\rho, X}$.
    pub d_dln_t: [f64; NUM_EOS_BASIC_RESULTS],
    /// Partial derivatives `[d_lnPgas_dxa, d_lnE_dxa]` for each species $k \in
    /// [0, \mathrm{species})$.
    pub d_dxa: Vec<[f64; NUM_EOS_D_DXA_RESULTS]>,
}

impl EosEvalOutput {
    /// Create a zero-initialized [`EosEvalOutput`] for `species` nuclear
    /// species.
    #[must_use]
    pub fn zeros(species: usize) -> Self {
        Self {
            res: [0.0; NUM_EOS_BASIC_RESULTS],
            d_dlnd: [0.0; NUM_EOS_BASIC_RESULTS],
            d_dln_t: [0.0; NUM_EOS_BASIC_RESULTS],
            d_dxa: vec![[0.0; NUM_EOS_D_DXA_RESULTS]; species],
        }
    }

    /// Clear all 7 blend-fraction channels (`19..26`) and mark `frac_index` as
    /// `1.0`.
    pub fn set_single_component_fraction(&mut self, frac_index: usize) {
        for idx in I_FRAC_HELM..=I_FRAC_IDEAL {
            self.res[idx] = 0.0;
            self.d_dlnd[idx] = 0.0;
            self.d_dln_t[idx] = 0.0;
        }
        if (I_FRAC_HELM..=I_FRAC_IDEAL).contains(&frac_index) {
            self.res[frac_index] = 1.0;
        }
    }
}

/// Errors returned by [`flexmesa_eos`](crate) routines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EosError {
    /// Invalid or non-physical input parameters (such as non-positive $T$,
    /// $\rho$, $\bar{A}$, or negative thermodynamic potentials).
    InvalidState {
        /// Routine where the failure occurred.
        context: &'static str,
        /// Diagnostic message.
        detail: String,
    },
    /// Query lies outside the tabulated domain of an EOS component.
    OffTable {
        /// Routine where the domain check triggered.
        context: &'static str,
        /// Diagnostic message.
        detail: String,
    },
    /// Root-finding inversion failed to converge within `max_iter`.
    ConvergenceFailure {
        /// Inversion routine name.
        context: &'static str,
        /// Diagnostic message.
        detail: String,
    },
}

impl Display for EosError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidState { context, detail }
            | Self::OffTable { context, detail }
            | Self::ConvergenceFailure { context, detail } => {
                write!(f, "{context}: {detail}")
            }
        }
    }
}

impl Error for EosError {}
