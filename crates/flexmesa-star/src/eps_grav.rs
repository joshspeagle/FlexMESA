// SPDX-License-Identifier: LGPL-3.0-or-later
//! Gravitational energy generation rate $\epsilon_{\mathrm{grav},k} = -T_k\,ds_k/dt$
//! with 3-cell stencil [`StarAdOrder1`] automatic differentiation and
//! composition derivatives (`star/private/eps_grav.f90`).
//!
//! Implements full line-by-line fidelity with upstream `eps_grav.f90`:
//! - **Standard $(\ln\rho, \ln T)$ thermodynamic identity (`do_std_eps_grav`,
//!   `eps_grav.f90:145-193`)**:
//!   $$\epsilon_{\mathrm{grav},k} = -T_k C_{P,k}\left[(1 - \nabla_{\mathrm{ad},k}\chi_{T,k})\frac{\Delta\ln T_k}{\Delta t} - \nabla_{\mathrm{ad},k}\chi_{\rho,k}\frac{\Delta\ln\rho_k}{\Delta t}\right] - \left(\frac{\Delta\ln\rho_k}{\Delta t}\,\Lambda_{\rho,k} + \frac{\Delta\ln T_k}{\Delta t}\,\Lambda_{T,k}\right),$$
//!   with optional time-centering (`use_time_centered_eps_grav && !doing_relax`,
//!   blending $0.5\,\epsilon_{\mathrm{grav,now}} + 0.5\,\epsilon_{\mathrm{grav,start}}$
//!   where $\epsilon_{\mathrm{grav,start}}$ evaluates start-of-step thermodynamic
//!   prefactors multiplying the live [`StarAdOrder1`] time derivatives) and
//!   phase-transition latent heat $(\Lambda_{\rho,k}, \Lambda_{T,k}) =
//!   (\mathrm{latent\_ddlnrho}_k, \mathrm{latent\_ddlnt}_k)$.
//! - **Direct specific-entropy time derivative (`do_lnS_eps_grav`,
//!   `eps_grav.f90:196-259`)**:
//!   $$\epsilon_{\mathrm{grav},k} = -T_k\,\frac{\exp(\ln s_k) - \exp(\ln s_{\mathrm{start},k})}{\Delta t}.$$
//! - **Potekhin-Chabrier (`PC`) Coulomb coupling blend
//!   (`eval1_eps_grav_and_partials`, `eps_grav.f90:85-103`)**:
//!   Linearly blends `do_lnS_eps_grav` and `do_std_eps_grav` across
//!   $\Gamma_{\mathrm{Coul},k} \in [\Gamma_{\ln S,\mathrm{off}}, \Gamma_{\ln S,\mathrm{on}}]$
//!   weighted by `eos_frac_pc[k]`.
//! - **Composition term (`include_composition_in_eps_grav`, `eps_grav.f90:262-399`)**
//!   and multiplicative `eps_grav_factor` (`eps_grav.f90:67, 187-191`).

use flexmesa_eos::{I_CHI_RHO, I_CHI_T, I_CP, I_GRAD_AD};

use crate::model::StarModel;
use crate::star_ad::{StarAdExt, StarAdOrder1, I_LND_00, I_LNT_00};

/// Evaluates $\epsilon_{\mathrm{grav},k}$ for cell `k` using the standard
/// $(\ln\rho, \ln T)$ thermodynamic relation (`do_std_eps_grav` in
/// `star/private/eps_grav.f90:145-193`), including optional time-centering,
/// phase-transition latent heat, and composition terms.
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

    let mut eps_grav =
        -t_ad * cp_ad * (one_m_grada_chit * dlnt_dt - grada_chirho * dlnd_dt);

    // Optional time-centering (`eps_grav.f90:157-165`):
    // Evaluates start-of-step thermodynamic prefactors multiplying the live
    // `dlnt_dt` and `dlnd_dt` `StarAdOrder1` time derivatives.
    if model.controls.use_time_centered_eps_grav && !model.aux.doing_relax {
        let cp_s = model
            .aux
            .cp_start
            .get(k)
            .copied()
            .filter(|&v| v > 0.0)
            .unwrap_or(model.cp[k]);
        let grada_s = model
            .aux
            .grada_start
            .get(k)
            .copied()
            .filter(|&v| v != 0.0)
            .unwrap_or(model.grada[k]);
        let chi_t_s = model
            .aux
            .chi_t_start
            .get(k)
            .copied()
            .filter(|&v| v != 0.0)
            .unwrap_or(model.chi_t[k]);
        let chi_rho_s = model
            .aux
            .chi_rho_start
            .get(k)
            .copied()
            .filter(|&v| v != 0.0)
            .unwrap_or(model.chi_rho[k]);
        let t_s = model.ln_t_start[k].exp();
        let one_m_s = 1.0 - grada_s * chi_t_s;
        let gc_s = grada_s * chi_rho_s;
        let eps_grav_start =
            (dlnt_dt * one_m_s - dlnd_dt * gc_s) * (-t_s * cp_s);
        eps_grav = eps_grav * 0.5 + eps_grav_start * 0.5;
    }

    // Phase-transition latent heat (`eps_grav.f90:167-170`):
    let lat_rho = model.aux.latent_ddlnrho.get(k).copied().unwrap_or(0.0);
    let lat_t = model.aux.latent_ddlnt.get(k).copied().unwrap_or(0.0);
    if lat_rho != 0.0 || lat_t != 0.0 {
        eps_grav = eps_grav - (dlnd_dt * lat_rho + dlnt_dt * lat_t);
    }

    // Optional composition derivative contribution (`eps_grav.f90:176-184, 262-399`):
    if model.controls.include_composition_in_eps_grav && model.species > 0 {
        let e_ad = model.ln_e_ad[k].exp();
        let p_ad = model.ln_peos_ad[k].exp();
        let rho_ad = lnd_00.exp();
        let p_div_rho_chirho = (p_ad * chi_t_ad)
            / (rho_ad * chi_rho_ad.max(StarAdOrder1::constant(1.0e-30)));
        let base = k * model.species;
        for j in 0..model.species {
            let dx_dt = (model.xa[base + j] - model.xa_start[base + j]) / dt;
            if dx_dt != 0.0 {
                let dlne_dx = model
                    .aux
                    .dln_e_dxa_for_partials
                    .get(base + j)
                    .copied()
                    .unwrap_or(0.0);
                let dlnp_dx = model
                    .aux
                    .dln_peos_dxa_for_partials
                    .get(base + j)
                    .copied()
                    .unwrap_or(0.0);
                eps_grav = eps_grav
                    + (-e_ad * dlne_dx + p_div_rho_chirho * dlnp_dx) * dx_dt;
            }
        }
    }

    eps_grav * model.controls.eps_grav_factor
}

/// Evaluates $\epsilon_{\mathrm{grav},k} = -T_k (s_k - s_{\mathrm{start},k})/\Delta t$
/// for cell `k` directly from specific entropy (`do_lnS_eps_grav` in
/// `star/private/eps_grav.f90:196-259`), scaled by `eps_grav_factor`.
#[must_use]
pub fn eval_eps_grav_lns_cell(model: &StarModel, k: usize) -> StarAdOrder1 {
    let dt = model.dt;
    if dt <= 0.0 || !dt.is_finite() {
        return StarAdOrder1::ZERO;
    }

    let t_ad = StarAdOrder1::variable(model.ln_t[k], I_LNT_00).exp();
    let s_ad = model.ln_s_ad[k].exp();
    let s_start = model.ln_s_start[k].exp();
    (-t_ad * (s_ad - s_start) / dt) * model.controls.eps_grav_factor
}

/// Evaluates $\epsilon_{\mathrm{grav},k}$ and its composition partial
/// derivatives `d_eps_grav_dx[k * species + j]` across all cells `k = 0..nz-1`
/// (`eval_eps_grav` and `eval1_eps_grav_and_partials` in
/// `star/private/eps_grav.f90:57-142`).
pub fn eval_eps_grav(model: &mut StarModel) {
    let nz = model.nz;
    let species = model.species;
    let dt = model.dt;
    let fac = model.controls.eps_grav_factor;
    if fac == 0.0 || dt <= 0.0 || !dt.is_finite() {
        model.eps_grav_ad.fill(StarAdOrder1::ZERO);
        model.aux.d_eps_grav_dx.fill(0.0);
        return;
    }

    let gam_on = model.controls.gamma_ln_s_eps_grav_full_on;
    let gam_off = model.controls.gamma_ln_s_eps_grav_full_off;
    let gam_span = (gam_on - gam_off).max(1.0e-12);
    let inc_comp =
        model.controls.include_composition_in_eps_grav && species > 0;

    for k in 0..nz {
        let f_pc = model
            .aux
            .eos_frac_pc
            .get(k)
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let alfa_lns = if f_pc > 0.0 {
            let w_gam = ((model.gam[k] - gam_off) / gam_span).clamp(0.0, 1.0);
            f_pc * w_gam
        } else {
            0.0
        };

        model.eps_grav_ad[k] = if (alfa_lns - 1.0).abs() < 1.0e-15 {
            eval_eps_grav_lns_cell(model, k)
        } else if alfa_lns <= 0.0 {
            eval_eps_grav_std_cell(model, k)
        } else {
            let eg_lns = eval_eps_grav_lns_cell(model, k);
            let eg_std = eval_eps_grav_std_cell(model, k);
            eg_lns * alfa_lns + eg_std * (1.0 - alfa_lns)
        };

        let base = k * species;
        if inc_comp {
            let e_k = model.energy[k];
            let p_div_rho_chirho = (model.peos[k] * model.chi_t[k])
                / (model.rho[k].max(1.0e-30) * model.chi_rho[k].max(1.0e-30));
            let scale_comp = (1.0 - alfa_lns) * fac / dt;
            for j in 0..species {
                let dlne_dx = model
                    .aux
                    .dln_e_dxa_for_partials
                    .get(base + j)
                    .copied()
                    .unwrap_or(0.0);
                let dlnp_dx = model
                    .aux
                    .dln_peos_dxa_for_partials
                    .get(base + j)
                    .copied()
                    .unwrap_or(0.0);
                model.aux.d_eps_grav_dx[base + j] =
                    scale_comp * (-e_k * dlne_dx + p_div_rho_chirho * dlnp_dx);
            }
        } else if base + species <= model.aux.d_eps_grav_dx.len() {
            model.aux.d_eps_grav_dx[base..base + species].fill(0.0);
        }
    }
}
