// SPDX-License-Identifier: GPL-3.0-or-later
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

use flexmesa_const::LN10;
use flexmesa_eos::{I_CHI_RHO, I_CHI_T, I_CP, I_GRAD_AD, I_LN_E};
use flexmesa_utils::is_bad;
use rayon::prelude::*;

use crate::controls::StarError;
use crate::micro::EosZoneContext;
use crate::model::StarModel;
use crate::star_ad::{StarAdExt, StarAdOrder1, I_LND_00, I_LNT_00};

/// Computes the Potekhin-Chabrier `do_lnS_eps_grav` blend weight `alfa_lns`
/// in `[0, 1]` for cell `k`.
#[inline]
fn pc_lns_blend_weight(
    model: &StarModel,
    k: usize,
    gam_off: f64,
    gam_span: f64,
) -> f64 {
    let f_pc = model
        .aux
        .eos_frac_pc
        .get(k)
        .copied()
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    if f_pc > 0.0 {
        let w_gam = ((model.gam[k] - gam_off) / gam_span).clamp(0.0, 1.0);
        f_pc * w_gam
    } else {
        0.0
    }
}

/// Evaluates the finite-difference composition contribution
/// `eps_grav_composition_term = -de/dt` and its `(\ln\rho, \ln T)` partials
/// (`eval_eps_grav_composition` in `star/private/eps_grav.f90:262-399`).
///
/// Uses `get_eos(xa_start, rho, T)` and, when
/// `use_time_centered_eps_grav && !doing_relax`, additionally evaluates
/// `get_eos(xa, rho_start, T_start)` with `theta = 0.5`.
fn eval_eps_grav_composition_term(
    model: &StarModel,
    k: usize,
    ctx: &EosZoneContext,
) -> Result<StarAdOrder1, StarError> {
    let species = model.species;
    let dt = model.dt;
    let base = k * species;
    let xa_k = &model.xa[base..base + species];
    let xa_start_k = &model.xa_start[base..base + species];

    let time_centered =
        model.controls.use_time_centered_eps_grav && !model.aux.doing_relax;
    let theta = if time_centered { 0.5 } else { 1.0 };

    let e = model.energy[k];
    let rho = model.rho[k];
    let log_rho = model.lnd[k] / LN10;
    let t = model.t[k];
    let log_t = model.ln_t[k] / LN10;

    // `star/private/eps_grav.f90:307-324`.
    let eos_xa_start = ctx.eval_cell(
        k,
        xa_start_k,
        rho,
        log_rho,
        t,
        log_t,
        &model.eos_config,
    )?;
    let e_with_xa_start = flexmesa_math::exp(eos_xa_start.res[I_LN_E]);
    let d_e_with_xa_start_dlnd =
        eos_xa_start.d_dlnd[I_LN_E] * e_with_xa_start;
    let d_e_with_xa_start_dlnt =
        eos_xa_start.d_dln_t[I_LN_E] * e_with_xa_start;

    // `star/private/micro.f90:499-500` and `star/private/eps_grav.f90:322-324`.
    let de_drho_for_partials = model.eos_d_dlnd[k][I_LN_E] * e / rho;
    let cv_for_partials = model.eos_d_dlnt[k][I_LN_E] * e / t;
    let mut de = e - e_with_xa_start;
    let mut d_de_dlnd = de_drho_for_partials * rho - d_e_with_xa_start_dlnd;
    let mut d_de_dlnt = cv_for_partials * t - d_e_with_xa_start_dlnt;

    // `star/private/eps_grav.f90:326-344`.
    if time_centered {
        let e_start = model.energy_start[k];
        let lnd_s = model.lnd_start[k];
        let lnt_s = model.ln_t_start[k];
        let rho_s = flexmesa_math::exp(lnd_s);
        let t_s = flexmesa_math::exp(lnt_s);
        let eos_dt_start = ctx.eval_cell(
            k,
            xa_k,
            rho_s,
            lnd_s / LN10,
            t_s,
            lnt_s / LN10,
            &model.eos_config,
        )?;
        let e_with_dt_start = flexmesa_math::exp(eos_dt_start.res[I_LN_E]);
        de = theta * de + (1.0 - theta) * (e_with_dt_start - e_start);
        d_de_dlnd *= theta;
        d_de_dlnt *= theta;
    }

    // `star/private/eps_grav.f90:360-397`.
    let val = -de / dt;
    if is_bad(val) {
        return Err(StarError::MicrophysicsError(format!(
            "eval_eps_grav_composition: bad eps_grav_composition_term ({val}) \
             at zone {k}"
        )));
    }
    Ok(StarAdOrder1::from_dlnd_dlnt_00(
        val,
        -d_de_dlnd / dt,
        -d_de_dlnt / dt,
    ))
}

fn eval_eps_grav_std_cell_with_ctx(
    model: &StarModel,
    k: usize,
    ctx: Option<&EosZoneContext>,
) -> Result<StarAdOrder1, StarError> {
    let dt = model.dt;
    if dt <= 0.0 || !dt.is_finite() {
        return Ok(StarAdOrder1::ZERO);
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
        let t_s = flexmesa_math::exp(model.ln_t_start[k]);
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

    // Finite-difference composition term (`eps_grav.f90:187-191, 262-399`):
    // Upstream evaluates the composition term in every cell, also where
    // `xa == xa_start`: its partials are then the rounding residue of
    // `dE_dRho_for_partials*rho - d_e_with_xa_start_dlnd`
    // (`star/private/eps_grav.f90:187-191, 318-324`).
    if model.controls.include_composition_in_eps_grav {
        let comp_term = if let Some(shared_ctx) = ctx {
            eval_eps_grav_composition_term(model, k, shared_ctx)?
        } else {
            let local_ctx = EosZoneContext::new(&model.chem_id)?;
            eval_eps_grav_composition_term(model, k, &local_ctx)?
        };
        eps_grav = eps_grav + comp_term;
    }

    if is_bad(eps_grav.val) {
        return Err(StarError::MicrophysicsError(format!(
            "do_std_eps_grav: bad value for eps_grav ({}) at zone {k}",
            eps_grav.val
        )));
    }

    Ok(eps_grav * model.controls.eps_grav_factor)
}

/// Evaluates $\epsilon_{\mathrm{grav},k}$ for cell `k` using the standard
/// $(\ln\rho, \ln T)$ thermodynamic relation (`do_std_eps_grav` in
/// `star/private/eps_grav.f90:123-209`), including optional time-centering,
/// phase-transition latent heat, and finite-difference composition heating
/// (`eval_eps_grav_composition` in `star/private/eps_grav.f90:262-399`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `get_eos` fails inside
/// `eval_eps_grav_composition` or if the resulting $\epsilon_{\mathrm{grav},k}$
/// is non-finite.
pub fn eval_eps_grav_std_cell(
    model: &StarModel,
    k: usize,
) -> Result<StarAdOrder1, StarError> {
    eval_eps_grav_std_cell_with_ctx(model, k, None)
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
    let s_start = flexmesa_math::exp(model.ln_s_start[k]);
    (-t_ad * (s_ad - s_start) / dt) * model.controls.eps_grav_factor
}

/// Evaluates one cell of `eval1_eps_grav_and_partials` (`eps_grav.f90:72-119`),
/// returning `(eps_grav_ad, used_std_form)`.
fn eval_eps_grav_cell(
    model: &StarModel,
    k: usize,
    gam_off: f64,
    gam_span: f64,
    ctx: Option<&EosZoneContext>,
) -> Result<(StarAdOrder1, bool), StarError> {
    let alfa_lns = pc_lns_blend_weight(model, k, gam_off, gam_span);
    let (ad, used_std) = if (alfa_lns - 1.0).abs() < 1.0e-15 {
        (eval_eps_grav_lns_cell(model, k), false)
    } else if alfa_lns <= 0.0 {
        (eval_eps_grav_std_cell_with_ctx(model, k, ctx)?, true)
    } else {
        let eg_lns = eval_eps_grav_lns_cell(model, k);
        let eg_std = eval_eps_grav_std_cell_with_ctx(model, k, ctx)?;
        (eg_lns * alfa_lns + eg_std * (1.0 - alfa_lns), true)
    };
    if is_bad(ad.val) {
        return Err(StarError::MicrophysicsError(format!(
            "eval_eps_grav_and_partials: bad eps_grav ({}) at zone {k}",
            ad.val
        )));
    }
    Ok((ad, used_std))
}

/// Populates `model.aux.d_eps_grav_dx[k * species .. (k + 1) * species]` for
/// cell `k` (`star/private/eps_grav.f90:53, 302-305`).
#[inline]
fn fill_d_eps_grav_dx_cell(
    model: &mut StarModel,
    k: usize,
    used_std: bool,
    inc_comp: bool,
    dt: f64,
) {
    let species = model.species;
    let base = k * species;
    if base + species > model.aux.d_eps_grav_dx.len() {
        return;
    }
    if inc_comp && used_std {
        // `star/private/eps_grav.f90:302-305`:
        // `d_eps_grav_dx(j,k) = -energy(k) * dlnE_dxa_for_partials(j,k) / dt +`
        // `(Peos(k) / Rho(k)) * dlnPeos_dxa_for_partials(j,k) * dxh_lnd(k) / dt`
        let e_k = model.energy[k];
        let p_div_rho = model.peos[k] / model.rho[k];
        let dxh_lnd = model.lnd[k] - model.lnd_start[k];
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
                -e_k * dlne_dx / dt + p_div_rho * dlnp_dx * dxh_lnd / dt;
        }
    } else {
        model.aux.d_eps_grav_dx[base..base + species].fill(0.0);
    }
}

/// Evaluates $\epsilon_{\mathrm{grav},k}$ and its composition partial
/// derivatives `d_eps_grav_dx[k * species + j]` across all cells `k = 0..nz-1`
/// (`eval_eps_grav_and_partials` and `eval1_eps_grav_and_partials` in
/// `star/private/eps_grav.f90:35-119`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `get_eos` fails inside
/// `eval_eps_grav_composition` or if any cell produces a non-finite
/// $\epsilon_{\mathrm{grav},k}$.
pub fn eval_eps_grav(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    let species = model.species;
    let dt = model.dt;
    let fac = model.controls.eps_grav_factor;
    if model.aux.d_eps_grav_dx.len() != nz * species {
        model.aux.d_eps_grav_dx.resize(nz * species, 0.0);
    }
    if fac == 0.0 || dt <= 0.0 || !dt.is_finite() {
        model.eps_grav_ad.fill(StarAdOrder1::ZERO);
        model.aux.d_eps_grav_dx.fill(0.0);
        return Ok(());
    }

    let gam_on = model.controls.gamma_ln_s_eps_grav_full_on;
    let gam_off = model.controls.gamma_ln_s_eps_grav_full_off;
    let gam_span = (gam_on - gam_off).max(1.0e-12);
    let inc_comp =
        model.controls.include_composition_in_eps_grav && species > 0;

    let need_ctx = inc_comp
        && (0..nz).any(|k| {
            let alfa_lns = pc_lns_blend_weight(model, k, gam_off, gam_span);
            (alfa_lns - 1.0).abs() >= 1.0e-15
        });

    if need_ctx {
        let ctx = EosZoneContext::new(&model.chem_id)?;
        let cell_results: Result<Vec<(StarAdOrder1, bool)>, StarError> = (0..nz)
            .into_par_iter()
            .map(|k| eval_eps_grav_cell(model, k, gam_off, gam_span, Some(&ctx)))
            .collect();
        for (k, (ad, used_std)) in cell_results?.into_iter().enumerate() {
            model.eps_grav_ad[k] = ad;
            fill_d_eps_grav_dx_cell(model, k, used_std, inc_comp, dt);
        }
    } else {
        for k in 0..nz {
            let (ad, used_std) =
                eval_eps_grav_cell(model, k, gam_off, gam_span, None)?;
            model.eps_grav_ad[k] = ad;
            fill_d_eps_grav_dx_cell(model, k, used_std, inc_comp, dt);
        }
    }
    Ok(())
}
