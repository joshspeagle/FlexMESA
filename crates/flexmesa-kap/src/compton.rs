//! Poutanen (2017) relativistic/degenerate Compton-scattering opacity (`Compton_Opacity`).
//!
//! Mirrors `Compton_Opacity` in `kap/private/kap_eval.f90` (lines 147-218), implementing
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
    let free_e_val = lnfree_e.exp();
    let free_e_ad = AutoDiffReal2VarOrder1 {
        val: free_e_val,
        d1: [
            free_e_val * d_lnfree_e_dln_rho / rho,
            free_e_val * d_lnfree_e_dln_t / t,
        ],
    };

    let tkev = t_ad * (KEV * 1.0e-3);
    let eta_sq = eta_ad * eta_ad;
    let zeta = (eta_ad * C01 + eta_sq * C02).exp();
    let zeta_sq = zeta * zeta;

    let f1 = zeta * C11 + zeta_sq * C12 + 1.0;
    let f2 = zeta * C21 + zeta_sq * C22 + 1.0;
    let f3 = zeta * C31 + zeta_sq * C32 + 1.0;

    let alpha = f3 * ALPHA0;
    let tbr = f2 * T0;

    // (tkev / tbr)^alpha = exp(alpha * ln(tkev / tbr))
    let ratio = tkev / tbr;
    let ratio_pow = (alpha * ratio.ln()).exp();
    let mfp = f1 * (ratio_pow + 1.0);

    let kap_ad = (free_e_ad * (SIGE / AMU)) / mfp;

    let kap = kap_ad.val;
    let dlnkap_dln_rho = rho * kap_ad.d1[0] / kap;
    let dlnkap_dln_t = t * kap_ad.d1[1] / kap;
    (kap, dlnkap_dln_rho, dlnkap_dln_t)
}
