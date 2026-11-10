// SPDX-License-Identifier: LGPL-3.0-or-later
//! Closed-form ideal $\gamma$-law $(\rho, E)$ equation of state (`eos_gamma_DE_get`).
//!
//! Mirrors `Get_eos_gamma_DE_Results` in `eos/private/eosde_eval.f90`.

use crate::types::{
    I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_DE_DRHO, I_DS_DRHO, I_DS_DT, I_ETA,
    I_FRAC_IDEAL, I_GAMMA1, I_GAMMA3, I_GRAD_AD, I_LN_E, I_LN_FREE_E,
    I_LN_PGAS, I_LN_S, I_MU, NUM_EOS_BASIC_RESULTS,
};
use flexmesa_const::{AVO, KERG};

/// Result of the closed-form ideal $\gamma$-law $(\rho, E)$ evaluation (`eos_gamma_de_get`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EosGammaDeOutput {
    /// Temperature $T$ ($\mathrm{K}$).
    pub t: f64,
    /// $\log_{10}(T / \mathrm{K})$.
    pub log10_t: f64,
    /// $(\partial \ln T / \partial \ln E)_\rho = 1$.
    pub dln_t_dln_e_c_rho: f64,
    /// $(\partial \ln T / \partial \ln \rho)_E = 0$.
    pub dln_t_dlnd_c_e: f64,
    /// $(\partial \ln P_{\mathrm{gas}} / \partial \ln E)_\rho = 1$.
    pub dln_pgas_dln_e_c_rho: f64,
    /// $(\partial \ln P_{\mathrm{gas}} / \partial \ln \rho)_E = 1$.
    pub dln_pgas_dlnd_c_e: f64,
    /// Basic EOS result array (`NUM_EOS_BASIC_RESULTS`).
    pub res: [f64; NUM_EOS_BASIC_RESULTS],
    /// Density partials $(\partial / \partial \ln \rho)_T$.
    pub d_dlnd: [f64; NUM_EOS_BASIC_RESULTS],
    /// Temperature partials $(\partial / \partial \ln T)_\rho$.
    pub d_dln_t: [f64; NUM_EOS_BASIC_RESULTS],
}

/// Evaluates the closed-form ideal $\gamma$-law EOS from $(\rho, E)$ (`eos_gamma_DE_get`).
///
/// Mirrors `Get_eos_gamma_DE_Results` in `eos/private/eosde_eval.f90`.
#[must_use]
pub fn eos_gamma_de_get(
    abar: f64,
    _zbar: f64,
    rho: f64,
    energy: f64,
    gamma: f64,
) -> EosGammaDeOutput {
    let avo_k_div_abar = AVO * KERG / abar;
    let p = (gamma - 1.0) * energy * rho;
    let t = (gamma - 1.0) * energy / avo_k_div_abar;
    let log10_t = t.log10();

    let cv = avo_k_div_abar / (gamma - 1.0);

    let raw_s = cv * (p / rho.powf(gamma)).ln() + 1.0e9;
    let entropy = if raw_s.is_nan() || raw_s <= 0.0 {
        1.0e-99
    } else {
        raw_s
    };

    let mut res = [0.0; NUM_EOS_BASIC_RESULTS];
    let mut d_dlnd = [0.0; NUM_EOS_BASIC_RESULTS];
    let mut d_dln_t = [0.0; NUM_EOS_BASIC_RESULTS];

    res[I_LN_PGAS] = p.ln();
    d_dlnd[I_LN_PGAS] = 1.0;
    d_dln_t[I_LN_PGAS] = 1.0;

    res[I_LN_E] = energy.ln();
    d_dlnd[I_LN_E] = 0.0;
    d_dln_t[I_LN_E] = 1.0;

    // Upstream `Get_eos_gamma_DE_Results` (`eos/private/eosde_eval.f90:84-104`)
    // sets `d_dlnRho_const_T(i_lnS) = 0`, `d_dlnT_const_Rho(i_lnS) = 0`,
    // `res(i_dS_dT) = 0`, `res(i_dS_dRho) = 0`, `res(i_mu) = 1 / abar`,
    // `res(i_lnfree_e) = -1e99`, and `res(i_eta) = 0`.
    res[I_LN_S] = entropy.ln();
    d_dlnd[I_LN_S] = 0.0;
    d_dln_t[I_LN_S] = 0.0;

    res[I_GRAD_AD] = (gamma - 1.0) / gamma;
    res[I_CHI_RHO] = 1.0;
    res[I_CHI_T] = 1.0;
    res[I_CP] = gamma * cv;
    res[I_CV] = cv;
    res[I_DE_DRHO] = 0.0;
    res[I_DS_DT] = 0.0;
    res[I_DS_DRHO] = 0.0;
    d_dlnd[I_DS_DRHO] = 0.0;
    d_dln_t[I_DS_DT] = 0.0;

    res[I_MU] = 1.0 / abar;
    res[I_LN_FREE_E] = -1.0e99;
    res[I_GAMMA1] = gamma;
    res[I_GAMMA3] = gamma;
    res[I_ETA] = 0.0;
    res[I_FRAC_IDEAL] = 1.0;

    EosGammaDeOutput {
        t,
        log10_t,
        dln_t_dln_e_c_rho: 1.0,
        dln_t_dlnd_c_e: 0.0,
        dln_pgas_dln_e_c_rho: 1.0,
        dln_pgas_dlnd_c_e: 1.0,
        res,
        d_dlnd,
        d_dln_t,
    }
}
