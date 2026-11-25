// SPDX-License-Identifier: GPL-3.0-or-later
//! Poutanen (2017) relativistic/degenerate Compton-scattering opacity (`Compton_Opacity`).
//!
//! Mirrors `Compton_Opacity` in `kap/private/kap_eval.f90` (lines 793-872), implementing
//! the analytical fitting formula of Poutanen (2017, ApJ 835, 119) with exact first-order
//! forward-mode automatic differentiation (`AutoDiffReal2VarOrder1`).

use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_const::{AMU, KEV, SIGE};

/// Evaluate the Poutanen (2017) electron-scattering Compton opacity $\kappa_{\mathrm{Compton}}$
/// and its logarithmic partial derivatives with respect to $\rho$ and $T$
/// (`Compton_Opacity` in `kap/private/kap_eval.f90`).
///
/// # Arguments
/// * `rho` - Mass density $\rho$ in $\mathrm{g\,cm^{-3}}$.
/// * `t` - Temperature $T$ in $\mathrm{K}$.
/// * `lnfree_e` - Natural logarithm of free electrons per nucleon $\ln n_e$.
/// * `d_lnfree_e_dln_rho` - Partial derivative $(\partial \ln n_e / \partial \ln \rho)_T$.
/// * `d_lnfree_e_dln_t` - Partial derivative $(\partial \ln n_e / \partial \ln T)_\rho$.
/// * `eta` - Electron degeneracy parameter $\eta = \mu_e / (k_{\mathrm{B}} T)$.
/// * `d_eta_dln_rho` - Partial derivative $(\partial \eta / \partial \ln \rho)_T$.
/// * `d_eta_dln_t` - Partial derivative $(\partial \eta / \partial \ln T)_\rho$.
///
/// # Returns
/// Tuple `(kap, dlnkap_dln_rho, dlnkap_dln_t)`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn compton_opacity(
    rho: f64,
    t: f64,
    lnfree_e: f64,
    d_lnfree_e_dln_rho: f64,
    d_lnfree_e_dln_t: f64,
    eta: f64,
    d_eta_dln_rho: f64,
    d_eta_dln_t: f64,
) -> (f64, f64, f64) {
    const T0: f64 = 43.3;
    const ALPHA0: f64 = 0.885;
    const C01: f64 = 0.682;
    const C02: f64 = -0.0454;
    const C11: f64 = 0.240;
    const C12: f64 = 0.0043;
    const C21: f64 = 0.050;
    const C22: f64 = -0.0067;
    const C31: f64 = -0.037;
    const C32: f64 = 0.0031;

    // Independent variables: var1 = Rho, var2 = T
    let t_ad = AutoDiffReal2VarOrder1 {
        val: t,
        d1: [0.0, 1.0],
    };
    let eta_ad = AutoDiffReal2VarOrder1 {
        val: eta,
        d1: [d_eta_dln_rho / rho, d_eta_dln_t / t],
    };
    let free_e_val = flexmesa_math::exp(lnfree_e);
    let free_e_ad = AutoDiffReal2VarOrder1 {
        val: free_e_val,
        d1: [
            free_e_val * d_lnfree_e_dln_rho / rho,
            free_e_val * d_lnfree_e_dln_t / t,
        ],
    };

    // Match `kap/private/kap_eval.f90:855-865` exact left-to-right operation order.
    let tkev = (t_ad * KEV) * 1.0e-3;
    let zeta = (C01 * eta_ad + (C02 * eta_ad) * eta_ad).exp();

    let f1 = (1.0 + C11 * zeta) + (C12 * zeta) * zeta;
    let f2 = (1.0 + C21 * zeta) + (C22 * zeta) * zeta;
    let f3 = (1.0 + C31 * zeta) + (C32 * zeta) * zeta;

    let alpha = ALPHA0 * f3;
    let tbr = T0 * f2;

    let ratio = tkev / tbr;
    let ratio_pow = ratio.pow(alpha);
    let mfp = f1 * (1.0 + ratio_pow);

    let kap_ad = (SIGE * (free_e_ad / AMU)) / mfp;

    let kap = kap_ad.val;
    let dlnkap_dln_rho = rho * kap_ad.d1[0] / kap;
    let dlnkap_dln_t = t * kap_ad.d1[1] / kap;
    (kap, dlnkap_dln_rho, dlnkap_dln_t)
}
