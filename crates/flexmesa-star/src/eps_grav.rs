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
//!   `eps_grav.f90:212-259`)**:
//!   $$\epsilon_{\mathrm{grav},k} = -T_k\,\frac{s_k - \exp(\ln s_{\mathrm{start},k})}{\Delta t}.$$
//! - **Potekhin-Chabrier (`PC`) Coulomb coupling blend
//!   (`eval1_eps_grav_and_partials`, `eps_grav.f90:85-103`)**:
//!   Uses boolean `using_PC = (eos_frac_PC[k] > 0.0)` and start-of-step
//!   `gam_start[k]` (`struct_burn_mix.f90:320`) across
//!   $[\Gamma_{\ln S,\mathrm{off}}, \Gamma_{\ln S,\mathrm{on}}]$ to select or
//!   linearly blend `do_lnS_eps_grav` and `do_std_eps_grav`.
//! - **Composition term (`include_composition_in_eps_grav`, `eps_grav.f90:262-399`)**
//!   and multiplicative `eps_grav_factor` applied once at the end (`eps_grav.f90:67`).

use flexmesa_const::LN10;
use flexmesa_eos::{
    I_CHI_RHO, I_CHI_T, I_CP, I_GRAD_AD, I_LATENT_DDLNRHO, I_LATENT_DDLNT,
    I_LN_E, I_LN_S,
};
use flexmesa_utils::is_bad;
use rayon::prelude::*;

use crate::controls::StarError;
use crate::micro::EosZoneContext;
use crate::model::StarModel;
use crate::star_ad::{StarAdExt, StarAdOrder1, I_LND_00, I_LNT_00};

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
    let d_e_with_xa_start_dlnd = eos_xa_start.d_dlnd[I_LN_E] * e_with_xa_start;
    let d_e_with_xa_start_dlnt = eos_xa_start.d_dln_t[I_LN_E] * e_with_xa_start;

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

    // `wrap_dxh_lnd(s,k)/s% dt` and `wrap_dxh_lnT(s,k)/s% dt`
    // (`star/private/eps_grav.f90:152-153`): the accumulated Newton
    // corrections, which round differently from `lnd - lnd_start`.
    let dlnd_dt = StarAdOrder1::variable(model.aux.dxh_lnd[k], I_LND_00) / dt;
    let dlnt_dt = StarAdOrder1::variable(model.aux.dxh_lnt[k], I_LNT_00) / dt;
    // `wrap_T_00(s, k)` (`star/private/auto_diff_support.f90:197-204`).
    let t_ad = StarAdOrder1::from_dlnd_dlnt_00(model.t[k], 0.0, model.t[k]);

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

    // MESA I, Equation (12) (`star/private/eps_grav.f90:162-163`).
    let mut eps_grav =
        -t_ad * cp_ad * (one_m_grada_chit * dlnt_dt - grada_chirho * dlnd_dt);

    // Phase-transition latent heat, `wrap_latent_ddlnRho_00` and
    // `wrap_latent_ddlnT_00` with their EOS partials
    // (`star/private/eps_grav.f90:164-169`,
    // `star/private/auto_diff_support.f90:853-862`), before the time
    // centering.
    let latent_ddlnrho = eos_ad(model.aux.latent_ddlnrho[k], I_LATENT_DDLNRHO);
    let latent_ddlnt = eos_ad(model.aux.latent_ddlnt[k], I_LATENT_DDLNT);
    eps_grav = eps_grav - (dlnd_dt * latent_ddlnrho + dlnt_dt * latent_ddlnt);

    // Optional time-centering (`eps_grav.f90:145-149, 173-184`):
    // Evaluates start-of-step thermodynamic prefactors multiplying the live
    // `dlnt_dt` and `dlnd_dt` `StarAdOrder1` time derivatives.
    if model.controls.use_time_centered_eps_grav && !model.aux.doing_relax {
        let theta = 0.5;
        let cp_s = model.aux.cp_start[k];
        let grada_s = model.aux.grada_start[k];
        let chi_t_s = model.aux.chi_t_start[k];
        let chi_rho_s = model.aux.chi_rho_start[k];
        let t_s = flexmesa_math::exp(model.ln_t_start[k]);
        let mut eps_grav_start = (-t_s * cp_s)
            * ((1.0 - grada_s * chi_t_s) * dlnt_dt
                - (grada_s * chi_rho_s) * dlnd_dt);
        // Start-of-step latent heat (`eps_grav.f90:179-180`).
        eps_grav_start = eps_grav_start
            - (dlnd_dt * model.aux.latent_ddlnrho_start[k]
                + dlnt_dt * model.aux.latent_ddlnt_start[k]);
        eps_grav = theta * eps_grav + (1.0 - theta) * eps_grav_start;
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

    Ok(eps_grav)
}

/// Evaluates $\epsilon_{\mathrm{grav},k}$ for cell `k` using the standard
/// $(\ln\rho, \ln T)$ thermodynamic relation (`do_std_eps_grav` in
/// `star/private/eps_grav.f90:123-209`), including optional time-centering,
/// phase-transition latent heat, and finite-difference composition heating
/// (`eval_eps_grav_composition` in `star/private/eps_grav.f90:262-399`).
///
/// Returns the unscaled `do_std_eps_grav` quantity; `eps_grav_factor` is
/// applied once at the end of [`eval_eps_grav`].
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `get_eos` fails inside
/// `eval_eps_grav_composition`, or if the resulting
/// $\epsilon_{\mathrm{grav},k}$ is non-finite.
pub fn eval_eps_grav_std_cell(
    model: &StarModel,
    k: usize,
) -> Result<StarAdOrder1, StarError> {
    eval_eps_grav_std_cell_with_ctx(model, k, None)
}

/// Evaluates $\epsilon_{\mathrm{grav},k} = -T_k (s_k - s_{\mathrm{start},k})/\Delta t$
/// for cell `k` directly from specific entropy (`do_lnS_eps_grav` in
/// `star/private/eps_grav.f90:212-259`), unscaled by `eps_grav_factor`.
///
/// Uses upstream `wrap_T_00` (`star/private/auto_diff_support.f90:197-204`)
/// and `wrap_s_00` (`auto_diff_support.f90:494-502`, with
/// `dS_dRho_for_partials` and `dS_dT_for_partials` per `micro.f90:501-502`).
#[must_use]
pub fn eval_eps_grav_lns_cell(model: &StarModel, k: usize) -> StarAdOrder1 {
    let dt = model.dt;
    let t_ad = StarAdOrder1::from_dlnd_dlnt_00(model.t[k], 0.0, model.t[k]);
    let entropy = model.entropy[k];
    let ds_drho_for_partials =
        model.eos_d_dlnd[k][I_LN_S] * entropy / model.rho[k];
    let ds_dt_for_partials = model.eos_d_dlnt[k][I_LN_S] * entropy / model.t[k];
    let s_ad = StarAdOrder1::from_dlnd_dlnt_00(
        entropy,
        ds_drho_for_partials * model.rho[k],
        ds_dt_for_partials * model.t[k],
    );
    let s_start = flexmesa_math::exp(model.ln_s_start[k]);
    -t_ad * (s_ad - s_start) / dt
}

fn do_ln_s_eps_grav(
    model: &StarModel,
    k: usize,
) -> Result<StarAdOrder1, StarError> {
    let eps_grav = eval_eps_grav_lns_cell(model, k);
    if is_bad(eps_grav.val) {
        return Err(StarError::MicrophysicsError(format!(
            "do_lnS_eps_grav: bad value for eps_grav ({}) at zone {k}",
            eps_grav.val
        )));
    }
    Ok(eps_grav)
}

/// Evaluates one cell of `eval1_eps_grav_and_partials` (`eps_grav.f90:72-119`),
/// returning `(eps_grav_ad, used_std_form)`.
fn eval_eps_grav_cell(
    model: &StarModel,
    k: usize,
    gam_on: f64,
    gam_off: f64,
    ctx: Option<&EosZoneContext>,
) -> Result<(StarAdOrder1, bool), StarError> {
    let using_pc = model.aux.eos_frac_pc[k] > 0.0;
    let gam_start = model.gam_start[k];
    let (ad, used_std) = if using_pc && gam_start >= gam_on {
        (do_ln_s_eps_grav(model, k)?, false)
    } else if using_pc && gam_start > gam_off {
        let alfa = (gam_start - gam_off) / (gam_on - gam_off);
        let eg_lns = do_ln_s_eps_grav(model, k)?;
        let eg_std = eval_eps_grav_std_cell_with_ctx(model, k, ctx)?;
        (alfa * eg_lns + (1.0 - alfa) * eg_std, true)
    } else {
        (eval_eps_grav_std_cell_with_ctx(model, k, ctx)?, true)
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
        let dxh_lnd = model.aux.dxh_lnd[k];
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
/// Returns [`StarError::MicrophysicsError`] if `dt <= 0.0`, if
/// `get_eos` fails inside `eval_eps_grav_composition`, or if any cell produces
/// a non-finite $\epsilon_{\mathrm{grav},k}$; returns [`StarError::InvalidInput`]
/// if `use_other_eps_grav` is enabled (unported hook).
pub fn eval_eps_grav(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    let species = model.species;
    let dt = model.dt;
    let fac = model.controls.eps_grav_factor;
    if dt <= 0.0 {
        return Err(StarError::MicrophysicsError(format!(
            "eval_eps_grav_and_partials: non-positive dt ({dt})"
        )));
    }
    if model.aux.d_eps_grav_dx.len() != nz * species {
        model.aux.d_eps_grav_dx.resize(nz * species, 0.0);
    }
    model.eps_grav_ad.fill(StarAdOrder1::ZERO);
    model.aux.d_eps_grav_dx.fill(0.0);

    let gam_on = model.controls.gamma_ln_s_eps_grav_full_on;
    let gam_off = model.controls.gamma_ln_s_eps_grav_full_off;
    let inc_comp =
        model.controls.include_composition_in_eps_grav && species > 0;

    let need_ctx = inc_comp
        && (0..nz).any(|k| {
            let using_pc = model.aux.eos_frac_pc[k] > 0.0;
            !(using_pc && model.gam_start[k] >= gam_on)
        });

    if need_ctx {
        let ctx = EosZoneContext::new(&model.chem_id)?;
        let cell_results: Result<Vec<(StarAdOrder1, bool)>, StarError> = (0
            ..nz)
            .into_par_iter()
            .map(|k| eval_eps_grav_cell(model, k, gam_on, gam_off, Some(&ctx)))
            .collect();
        for (k, (ad, used_std)) in cell_results?.into_iter().enumerate() {
            model.eps_grav_ad[k] = ad;
            fill_d_eps_grav_dx_cell(model, k, used_std, inc_comp, dt);
        }
    } else {
        for k in 0..nz {
            let (ad, used_std) =
                eval_eps_grav_cell(model, k, gam_on, gam_off, None)?;
            model.eps_grav_ad[k] = ad;
            fill_d_eps_grav_dx_cell(model, k, used_std, inc_comp, dt);
        }
    }

    if nz > 0 && model.controls.use_other_eps_grav {
        return Err(StarError::InvalidInput(String::from(
            "use_other_eps_grav = .true. requires an other_eps_grav hook (unported)",
        )));
    }

    for k in 0..nz {
        model.eps_grav_ad[k] = fac * model.eps_grav_ad[k];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::{StarControls, StarJobControls};
    use crate::zams::load_zams_model;

    fn test_solar_model() -> StarModel {
        let mut ctrl = StarControls {
            zbase: 0.02,
            ..StarControls::default()
        };
        ctrl.initial_mass = 1.0;
        let mut model = load_zams_model(ctrl, StarJobControls::default())
            .expect("valid 1 Msun ZAMS model");
        model.dt = 1.0e10;
        crate::micro::set_basic_vars(&mut model).expect("set_basic_vars init");
        crate::micro::do_eos(&mut model).expect("do_eos init");
        model.save_start_values().expect("save_start_values init");
        let nz = model.nz;
        for k in 0..nz {
            model.lnd[k] += 1.0e-3;
            model.ln_t[k] += 0.8e-3;
        }
        crate::micro::set_basic_vars(&mut model).expect("set_basic_vars");
        crate::micro::do_eos(&mut model).expect("do_eos");
        for k in 0..nz {
            model.aux.dxh_lnd[k] = model.lnd[k] - model.lnd_start[k];
            model.aux.dxh_lnt[k] = model.ln_t[k] - model.ln_t_start[k];
        }
        model
    }

    #[test]
    fn test_pc_blend_uses_boolean_eos_frac_pc_and_gam_start() {
        let mut model = test_solar_model();
        let k = model.nz / 2;
        model.gam[k] = 175.0;
        model.save_start_values().expect("save_start_values");
        assert_eq!(model.gam_start[k], 175.0);
        model.lnd[k] += 2.0e-3;
        model.ln_t[k] += 1.5e-3;
        crate::micro::set_basic_vars(&mut model).expect("set_basic_vars pert");
        crate::micro::do_eos(&mut model).expect("do_eos pert");
        model.aux.dxh_lnd[k] = model.lnd[k] - model.lnd_start[k];
        model.aux.dxh_lnt[k] = model.ln_t[k] - model.ln_t_start[k];

        // Moving live `gam` below full_off does not alter blend when `gam_start >= gam_on`.
        model.gam[k] = 50.0;
        model.aux.eos_frac_pc[k] = 0.25;
        model.controls.include_composition_in_eps_grav = true;
        let base = k * model.species;
        model.aux.dln_e_dxa_for_partials[base] = 0.5;
        model.aux.dln_peos_dxa_for_partials[base] = -0.3;

        eval_eps_grav(&mut model).expect("eval_eps_grav full lnS");
        let expected_lns = eval_eps_grav_lns_cell(&model, k);
        assert!(expected_lns.val.abs() > 0.0);
        assert_eq!(
            model.eps_grav_ad[k].val.to_bits(),
            expected_lns.val.to_bits()
        );
        assert!(model.aux.d_eps_grav_dx[base..base + model.species]
            .iter()
            .all(|&v| v == 0.0));

        // When `eos_frac_pc == 0.0`, high `gam_start` still takes `do_std_eps_grav`.
        model.aux.eos_frac_pc[k] = 0.0;
        model.gam_start[k] = 200.0;
        eval_eps_grav(&mut model)
            .expect("eval_eps_grav std when eos_frac_pc=0");
        let expected_std =
            eval_eps_grav_std_cell(&model, k).expect("eval_eps_grav_std_cell");
        assert!(expected_std.val.abs() > 0.0);
        assert_eq!(
            model.eps_grav_ad[k].val.to_bits(),
            expected_std.val.to_bits()
        );
        assert!(model.aux.d_eps_grav_dx[base].abs() > 0.0);

        // Intermediate `gam_start` with `eos_frac_pc > 0` blends using `alfa`,
        // then applies `eps_grav_factor` once at the end.
        model.aux.eos_frac_pc[k] = 0.25;
        model.gam_start[k] = 135.0;
        model.gam[k] = 10.0;
        model.controls.eps_grav_factor = 0.75;
        eval_eps_grav(&mut model).expect("eval_eps_grav blended");
        let eg_lns = eval_eps_grav_lns_cell(&model, k);
        let eg_std =
            eval_eps_grav_std_cell(&model, k).expect("eval_eps_grav_std_cell");
        let expected_blend = 0.75 * (0.5 * eg_lns + 0.5 * eg_std);
        assert_eq!(
            model.eps_grav_ad[k].val.to_bits(),
            expected_blend.val.to_bits()
        );
    }

    #[test]
    fn test_eps_grav_factor_zero_still_runs_d_eps_grav_dx_and_bad_checks() {
        let mut model = test_solar_model();
        let k = model.nz / 2;
        model.controls.eps_grav_factor = 0.0;
        model.controls.include_composition_in_eps_grav = true;
        let base = k * model.species;
        model.aux.dln_e_dxa_for_partials[base] = 0.5;
        model.aux.dln_peos_dxa_for_partials[base] = -0.3;

        eval_eps_grav(&mut model).expect("eval_eps_grav factor=0");
        assert!(model.eps_grav_ad.iter().all(|a| a.val == 0.0));
        assert!(model.eps_grav_ad.iter().all(|a| a.d1 == [0.0; 24]));
        assert!(model.aux.d_eps_grav_dx[base].abs() > 0.0);

        model.cp[k] = f64::NAN;
        assert!(matches!(
            eval_eps_grav(&mut model),
            Err(StarError::MicrophysicsError(_))
        ));
    }

    #[test]
    fn test_eps_grav_nonpositive_dt_and_use_other_eps_grav_error() {
        let mut model = test_solar_model();
        model.dt = 0.0;
        assert!(matches!(
            eval_eps_grav(&mut model),
            Err(StarError::MicrophysicsError(_))
        ));
        model.dt = -1.0;
        assert!(matches!(
            eval_eps_grav(&mut model),
            Err(StarError::MicrophysicsError(_))
        ));

        let mut model = test_solar_model();
        model.controls.use_other_eps_grav = true;
        assert!(matches!(
            eval_eps_grav(&mut model),
            Err(StarError::InvalidInput(_))
        ));
    }
}
