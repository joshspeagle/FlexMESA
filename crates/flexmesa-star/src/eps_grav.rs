// SPDX-License-Identifier: LGPL-3.0-or-later
//! Gravitational energy generation rate $\epsilon_{\mathrm{grav},k} = -T_k\,ds_k/dt$
//! with 3-cell stencil [`StarAdOrder1`] automatic differentiation (`eps_grav.f90`).
//!
//! Implements both the standard $(\ln\rho, \ln T)$ thermodynamic identity
//! (`do_std_eps_grav` in `star/private/eps_grav.f90:247-340`):
//! $$\epsilon_{\mathrm{grav},k} = -T_k C_{P,k}\left[(1 - \nabla_{\mathrm{ad},k}\chi_{T,k})\frac{\Delta\ln T_k}{\Delta t} - \nabla_{\mathrm{ad},k}\chi_{\rho,k}\frac{\Delta\ln\rho_k}{\Delta t}\right],$$
//! and the direct specific-entropy time derivative (`do_lns_eps_grav` in
//! `star/private/eps_grav.f90:343-402`):
//! $$\epsilon_{\mathrm{grav},k} = -T_k\,\frac{\exp(\ln s_k) - \exp(\ln s_{\mathrm{start},k})}{\Delta t}.$$

use flexmesa_eos::{I_CHI_RHO, I_CHI_T, I_CP, I_GRAD_AD};

use crate::model::StarModel;
use crate::star_ad::{StarAdExt, StarAdOrder1, I_LND_00, I_LNT_00};

/// Evaluates $\epsilon_{\mathrm{grav},k}$ for cell `k` using the standard
/// $(\ln\rho, \ln T)$ thermodynamic relation (`do_std_eps_grav` in
/// `star/private/eps_grav.f90:247-340`).
///
/// All EOS thermodynamic coefficients ($C_P, \nabla_{\mathrm{ad}}, \chi_T, \chi_\rho$)
/// carry their first-order $(\partial/\partial\ln\rho_k, \partial/\partial\ln T_k)$
/// partials in [`StarAdOrder1`].
#[must_use]
pub fn eval_eps_grav_std_cell(model: &StarModel, k: usize) -> StarAdOrder1 {
    let dt = model.dt;
    if dt <= 0.0 || !dt.is_finite() {
        return StarAdOrder1::ZERO;
    }

    let lnd_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00);
    let lnt_00 = StarAdOrder1::variable(model.ln_t[k], I_LNT_00);

    let dlnd_dt = (lnd_00 - model.lnd_start[k]) / dt;
    let dlnt_dt = (lnt_00 - model.ln_t_start[k]) / dt;
    let t_ad = lnt_00.exp();

    let eos_ad = |val: f64, slot: usize| {
        StarAdOrder1::from_dlnd_dlnt_00(
            val,
            model.eos_d_dlnd[k][slot],
            model.eos_d_dlnt[k][slot],
        )
    };
    let cp_ad = eos_ad(model.cp[k], I_CP);
    let grada_ad = eos_ad(model.grada[k], I_GRAD_AD);
    let chi_t_ad = eos_ad(model.chi_t[k], I_CHI_T);
    let chi_rho_ad = eos_ad(model.chi_rho[k], I_CHI_RHO);

    let one_m_grada_chit = StarAdOrder1::ONE - grada_ad * chi_t_ad;
    let grada_chirho = grada_ad * chi_rho_ad;

    -t_ad * cp_ad * (one_m_grada_chit * dlnt_dt - grada_chirho * dlnd_dt)
}

/// Evaluates $\epsilon_{\mathrm{grav},k} = -T_k (s_k - s_{\mathrm{start},k})/\Delta t$
/// for cell `k` directly from specific entropy (`do_lns_eps_grav` in
/// `star/private/eps_grav.f90:343-402`).
#[must_use]
pub fn eval_eps_grav_lns_cell(model: &StarModel, k: usize) -> StarAdOrder1 {
    let dt = model.dt;
    if dt <= 0.0 || !dt.is_finite() {
        return StarAdOrder1::ZERO;
    }

    let t_ad = StarAdOrder1::variable(model.ln_t[k], I_LNT_00).exp();
    let s_ad = model.ln_s_ad[k].exp();
    let s_start = model.ln_s_start[k].exp();
    -t_ad * (s_ad - s_start) / dt
}

/// Evaluates $\epsilon_{\mathrm{grav},k}$ across all cells `k = 0..nz-1` and
/// stores the result in `model.eps_grav_ad` (`eval_eps_grav` in
/// `star/private/eps_grav.f90:57-146`).
pub fn eval_eps_grav(model: &mut StarModel) {
    let nz = model.nz;
    for k in 0..nz {
        model.eps_grav_ad[k] = eval_eps_grav_std_cell(model, k);
    }
}
