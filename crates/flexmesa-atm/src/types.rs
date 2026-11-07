// SPDX-License-Identifier: LGPL-3.0-or-later
//! Public constants, result structures, and callback signatures for `flexmesa-atm`.
//!
//! Mirrors `atm/public/atm_def.f90` from upstream MESA.

use flexmesa_eos::NUM_EOS_BASIC_RESULTS;
use std::fmt;

/// Eddington grey $T(\tau)$ relation (`ATM_T_TAU_EDDINGTON = 1`).
pub const ATM_T_TAU_EDDINGTON: i32 = 1;
/// Solar-calibrated Hopf $T(\tau)$ relation (`ATM_T_TAU_SOLAR_HOPF = 2`).
pub const ATM_T_TAU_SOLAR_HOPF: i32 = 2;
/// Krishna-Swamy (1966) empirical solar $T(\tau)$ relation (`ATM_T_TAU_KRISHNA_SWAMY = 3`).
pub const ATM_T_TAU_KRISHNA_SWAMY: i32 = 3;
/// Ball (2021) fit to Trampedach et al. (2014) 3D solar simulation (`ATM_T_TAU_TRAMPEDACH_SOLAR = 4`).
pub const ATM_T_TAU_TRAMPEDACH_SOLAR: i32 = 4;

/// Castelli & Kurucz (2003) / Allard et al. (2001) table at $\tau = 100$ (`ATM_TABLE_TAU_100 = 100`).
pub const ATM_TABLE_TAU_100: i32 = 100;
/// Atmosphere table at $\tau = 10$ (`ATM_TABLE_TAU_10 = 101`).
pub const ATM_TABLE_TAU_10: i32 = 101;
/// Atmosphere table at $\tau = 1$ (`ATM_TABLE_TAU_1 = 102`).
pub const ATM_TABLE_TAU_1: i32 = 102;
/// Atmosphere table at $\tau = 0.1$ (`ATM_TABLE_TAU_1M1 = 103`).
pub const ATM_TABLE_TAU_1M1: i32 = 103;
/// Photosphere table at $\tau = 2/3$ (`ATM_TABLE_PHOTOSPHERE = 104`).
pub const ATM_TABLE_PHOTOSPHERE: i32 = 104;
/// Rohrmann et al. (2011) DA white dwarf table at $\tau = 25.1188$ (`ATM_TABLE_WD_TAU_25 = 105`).
pub const ATM_TABLE_WD_TAU_25: i32 = 105;
/// DB white dwarf table at $\tau = 25$ (`ATM_TABLE_DB_WD_TAU_25 = 106`).
pub const ATM_TABLE_DB_WD_TAU_25: i32 = 106;

/// Expected version tag for `table_atm` files (`table_atm_version = 5`).
pub const TABLE_ATM_VERSION: i32 = 5;

/// 0-based index of exterior atmosphere mass $m_{\mathrm{ext}}$ ($\mathrm{g}$) (`atm_xm = 1` in Fortran).
pub const ATM_XM: usize = 0;
/// 0-based index of radial distance $\Delta r$ above envelope base ($\mathrm{cm}$) (`atm_delta_r = 2`).
pub const ATM_DELTA_R: usize = 1;
/// 0-based index of $\ln(P / \mathrm{dyn\,cm^{-2}})$ (`atm_lnP = 3`).
pub const ATM_LNP: usize = 2;
/// 0-based index of $\ln(\rho / \mathrm{g\,cm^{-3}})$ (`atm_lnd = 4`).
pub const ATM_LND: usize = 3;
/// 0-based index of $\ln(T / \mathrm{K})$ (`atm_lnT = 5`).
pub const ATM_LNT: usize = 4;
/// 0-based index of temperature gradient $\nabla_T$ (`atm_gradT = 6`).
pub const ATM_GRADT: usize = 5;
/// 0-based index of Rosseland mean opacity $\kappa$ ($\mathrm{cm^2\,g^{-1}}$) (`atm_kap = 7`).
pub const ATM_KAP: usize = 6;
/// 0-based index of adiabatic exponent $\Gamma_1$ (`atm_gamma1 = 8`).
pub const ATM_GAMMA1: usize = 7;
/// 0-based index of adiabatic gradient $\nabla_{\mathrm{ad}}$ (`atm_grada = 9`).
pub const ATM_GRADA: usize = 8;
/// 0-based index of $(\partial \ln P / \partial \ln T)_\rho$ (`atm_chiT = 10`).
pub const ATM_CHIT: usize = 9;
/// 0-based index of $(\partial \ln P / \partial \ln \rho)_T$ (`atm_chiRho = 11`).
pub const ATM_CHIRHO: usize = 10;
/// 0-based index of specific heat at constant volume $c_V$ (`atm_cv = 12`).
pub const ATM_CV: usize = 11;
/// 0-based index of specific heat at constant pressure $c_P$ (`atm_cp = 13`).
pub const ATM_CP: usize = 12;
/// 0-based index of $\ln n_{e,\mathrm{free}}$ (`atm_lnfree_e = 14`).
pub const ATM_LNFREE_E: usize = 13;
/// 0-based index of $(\partial \ln \kappa / \partial \ln T)_\rho$ (`atm_dlnkap_dlnT = 15`).
pub const ATM_DLNKAP_DLNT: usize = 14;
/// 0-based index of $(\partial \ln \kappa / \partial \ln \rho)_T$ (`atm_dlnkap_dlnd = 16`).
pub const ATM_DLNKAP_DLND: usize = 15;
/// 0-based index of $\ln(P_{\mathrm{gas}} / \mathrm{dyn\,cm^{-2}})$ (`atm_lnPgas = 17`).
pub const ATM_LNPGAS: usize = 16;
/// 0-based index of optical depth $\tau$ (`atm_tau = 18`).
pub const ATM_TAU: usize = 17;
/// 0-based index of radiative temperature gradient $\nabla_{\mathrm{rad}}$ (`atm_gradr = 19`).
pub const ATM_GRADR: usize = 18;

/// Number of structure quantities per atmosphere zone (`num_results_for_build_atm = 19`).
pub const NUM_RESULTS_FOR_BUILD_ATM: usize = 19;

/// Error variants returned by `flexmesa-atm` atmosphere evaluators.
#[derive(Debug, Clone, PartialEq)]
pub enum AtmError {
    /// Invalid input parameter or non-positive stellar $L, R, M$.
    InvalidArgument {
        /// Routine where the validation failed.
        context: &'static str,
        /// Explanation of the invalid value.
        detail: String,
    },
    /// Unknown $T(\tau)$ relation ID or atmosphere table ID.
    UnknownId {
        /// Routine where the ID was queried.
        context: &'static str,
        /// Invalid numeric ID.
        id: i32,
    },
    /// Fixed-point opacity iteration or ODE integration failed to converge.
    ConvergenceFailed {
        /// Routine where convergence failed.
        context: &'static str,
        /// Diagnostic details.
        detail: String,
    },
    /// Callback evaluation (`eos_proc` or `kap_proc`) or spline lookup failed.
    CallbackError {
        /// Routine where the error occurred.
        context: &'static str,
        /// Error message from callback or interpolator.
        detail: String,
    },
}

impl fmt::Display for AtmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument { context, detail } => {
                write!(f, "invalid atm argument in {context}: {detail}")
            }
            Self::UnknownId { context, id } => {
                write!(f, "unknown atm id {id} in {context}")
            }
            Self::ConvergenceFailed { context, detail } => {
                write!(f, "atm convergence failure in {context}: {detail}")
            }
            Self::CallbackError { context, detail } => {
                write!(
                    f,
                    "atm callback/interpolation failure in {context}: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for AtmError {}

/// Output of `atm_eos_iface` callback at $(\ln P, \ln T)$.
#[derive(Debug, Clone)]
pub struct AtmEosCallbackResult {
    /// Natural log of mass density $\ln(\rho / \mathrm{g\,cm^{-3}})$.
    pub ln_rho: f64,
    /// Basic EOS results array (`NUM_EOS_BASIC_RESULTS = 16`).
    pub res: [f64; NUM_EOS_BASIC_RESULTS],
    /// Partial derivatives $(\partial \mathrm{res} / \partial \ln \rho)_T$.
    pub dres_dln_rho: [f64; NUM_EOS_BASIC_RESULTS],
    /// Partial derivatives $(\partial \mathrm{res} / \partial \ln T)_\rho$.
    pub dres_dln_t: [f64; NUM_EOS_BASIC_RESULTS],
}

/// Output of `atm_kap_iface` callback at $(\ln \rho, \ln T)$.
#[derive(Debug, Clone, Copy)]
pub struct AtmKapCallbackResult {
    /// Rosseland mean opacity $\kappa$ ($\mathrm{cm^2\,g^{-1}}$).
    pub kap: f64,
    /// Partial derivative $(\partial \ln \kappa / \partial \ln \rho)_T$.
    pub dlnkap_dln_rho: f64,
    /// Partial derivative $(\partial \ln \kappa / \partial \ln T)_\rho$.
    pub dlnkap_dln_t: f64,
}

/// Surface $(\ln T, \ln P)$ and Jacobian partial derivatives wrt $(L, \ln R, \ln M, \ln \kappa)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmEvalOutput {
    /// Natural log of surface temperature $\ln(T / \mathrm{K})$.
    pub ln_t: f64,
    /// $(\partial \ln T / \partial L)_{R, M, \kappa}$ ($\mathrm{erg^{-1}\,s}$).
    pub dln_t_dl: f64,
    /// $(\partial \ln T / \partial \ln R)_{L, M, \kappa}$.
    pub dln_t_dln_r: f64,
    /// $(\partial \ln T / \partial \ln M)_{L, R, \kappa}$.
    pub dln_t_dln_m: f64,
    /// $(\partial \ln T / \partial \ln \kappa)_{L, R, M}$.
    pub dln_t_dln_kap: f64,
    /// Natural log of surface total pressure $\ln(P / \mathrm{dyn\,cm^{-2}})$.
    pub ln_p: f64,
    /// $(\partial \ln P / \partial L)_{R, M, \kappa}$ ($\mathrm{erg^{-1}\,s}$).
    pub dln_p_dl: f64,
    /// $(\partial \ln P / \partial \ln R)_{L, M, \kappa}$.
    pub dln_p_dln_r: f64,
    /// $(\partial \ln P / \partial \ln M)_{L, R, \kappa}$.
    pub dln_p_dln_m: f64,
    /// $(\partial \ln P / \partial \ln \kappa)_{L, R, M}$.
    pub dln_p_dln_kap: f64,
    /// Converged Rosseland mean opacity $\kappa$ ($\mathrm{cm^2\,g^{-1}}$) (`0.0` for table/varying).
    pub kap: f64,
}

/// Output of `eval_irradiated` (Guillot & Havel 2011 grey irradiated atmosphere).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmIrradiatedOutput {
    /// Converged thermal Rosseland mean opacity $\kappa$ ($\mathrm{cm^2\,g^{-1}}$).
    pub kap: f64,
    /// Optical depth $\tau = P_{\mathrm{surf}} \kappa / g$ at $P_{\mathrm{surf}}$.
    pub tau: f64,
    /// Natural log of temperature $\ln(T / \mathrm{K})$ at $P_{\mathrm{surf}}$.
    pub ln_t: f64,
    /// $(\partial \ln T / \partial L)_{R, M, \kappa}$ ($\mathrm{erg^{-1}\,s}$).
    pub dln_t_dl: f64,
    /// $(\partial \ln T / \partial \ln R)_{L, M, \kappa}$.
    pub dln_t_dln_r: f64,
    /// $(\partial \ln T / \partial \ln M)_{L, R, \kappa}$.
    pub dln_t_dln_m: f64,
    /// $(\partial \ln T / \partial \ln \kappa)_{L, R, M}$.
    pub dln_t_dln_kap: f64,
}

/// Output of `build_t_tau_uniform` and `build_t_tau_varying` outward-in atmosphere profiles.
#[derive(Debug, Clone, PartialEq)]
pub struct AtmStructure {
    /// Outward-to-inward atmosphere zones; each entry holds `NUM_RESULTS_FOR_BUILD_ATM = 19` values.
    pub points: Vec<[f64; NUM_RESULTS_FOR_BUILD_ATM]>,
}
