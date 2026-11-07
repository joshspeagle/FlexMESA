// SPDX-License-Identifier: LGPL-3.0-or-later
//! Grey and solar-calibrated $T(\tau)$ atmosphere with depth-varying opacity (`atm_t_tau_varying.f90`).
//!
//! Ports `eval_T_tau_varying` (inward `dopri5` integration of
//! $d\ln P / d\ln\tau = \tau g / (\kappa P)$ from $\tau_{\mathrm{outer}} = 10^{-5} \tau_{\mathrm{surf}}$
//! with up to 3 retry scalings $\times 10$, plus central finite differencing
//! $\Delta \ln T_{\mathrm{eff}} = 10^{-4}$ when `skip_partials == false`) and
//! `build_T_tau_varying`.

use crate::t_tau_relations::eval_t_tau;
use crate::types::{
    AtmEosCallbackResult, AtmError, AtmEvalOutput, AtmKapCallbackResult,
    AtmStructure, ATM_CHIRHO, ATM_CHIT, ATM_CP, ATM_CV, ATM_DELTA_R,
    ATM_DLNKAP_DLND, ATM_DLNKAP_DLNT, ATM_GAMMA1, ATM_GRADA, ATM_GRADR,
    ATM_GRADT, ATM_KAP, ATM_LND, ATM_LNFREE_E, ATM_LNP, ATM_LNPGAS, ATM_LNT,
    ATM_TAU, ATM_XM, NUM_RESULTS_FOR_BUILD_ATM,
};
use crate::utils::{eval_paczynski_gradr, radiation_pressure};
use flexmesa_const::{CGAS, LN10};
use flexmesa_eos::{
    I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_GAMMA1, I_GRAD_AD, I_LN_FREE_E, I_LN_PGAS,
};
use flexmesa_num::{dopri5, NumError};

#[allow(clippy::too_many_arguments)]
fn eval_varying_data_try<E, K>(
    tau_surf: f64,
    teff: f64,
    g: f64,
    tau_outer: f64,
    t_tau_id: i32,
    eos_proc: &mut E,
    kap_proc: &mut K,
    errtol: f64,
    max_steps: usize,
) -> Result<(f64, f64), AtmError>
where
    E: FnMut(f64, f64) -> Result<AtmEosCallbackResult, AtmError>,
    K: FnMut(
        f64,
        f64,
        &AtmEosCallbackResult,
    ) -> Result<AtmKapCallbackResult, AtmError>,
{
    const RHO_OUTER: f64 = 1.0e-10;
    const DLNTAU_SURF: f64 = 1.0e-3;
    const DLNTAU_MAX: f64 = 0.0;

    let ln_t_outer = eval_t_tau(t_tau_id, tau_outer, teff)?;
    let t_outer = ln_t_outer.exp();
    let pgas_outer = CGAS * RHO_OUTER * t_outer;
    let p_outer = pgas_outer + radiation_pressure(t_outer);
    let ln_p_outer = p_outer.ln();

    let lntau_outer = tau_outer.ln();
    let lntau_surf = tau_surf.ln();

    let res = dopri5(
        lntau_outer,
        lntau_surf,
        &[ln_p_outer],
        DLNTAU_SURF,
        DLNTAU_MAX,
        errtol,
        errtol,
        max_steps,
        |x, y| {
            let tau = x.exp();
            let ln_p = y[0];
            let ln_t = eval_t_tau(t_tau_id, tau, teff).map_err(|_| {
                NumError::NonFiniteValue("eval_t_tau failed in dopri5")
            })?;
            let eos_res = eos_proc(ln_p, ln_t).map_err(|_| {
                NumError::NonFiniteValue("eos_proc failed in dopri5")
            })?;
            let kap_res =
                kap_proc(eos_res.ln_rho, ln_t, &eos_res).map_err(|_| {
                    NumError::NonFiniteValue("kap_proc failed in dopri5")
                })?;
            Ok(vec![tau * g / (kap_res.kap * ln_p.exp())])
        },
    )
    .map_err(|e| AtmError::ConvergenceFailed {
        context: "eval_varying_data_try",
        detail: e.to_string(),
    })?;

    let ln_p = res.y[0];
    let ln_t = eval_t_tau(t_tau_id, tau_surf, teff)?;
    Ok((ln_t, ln_p))
}

#[allow(clippy::too_many_arguments)]
fn eval_varying_data<E, K>(
    tau_surf: f64,
    teff: f64,
    g: f64,
    t_tau_id: i32,
    eos_proc: &mut E,
    kap_proc: &mut K,
    errtol: f64,
    max_steps: usize,
) -> Result<(f64, f64), AtmError>
where
    E: FnMut(f64, f64) -> Result<AtmEosCallbackResult, AtmError>,
    K: FnMut(
        f64,
        f64,
        &AtmEosCallbackResult,
    ) -> Result<AtmKapCallbackResult, AtmError>,
{
    const TAU_OUTER_FACTOR: f64 = 1.0e-5;
    const MAX_TRIES: usize = 3;
    const TRY_SCALE: f64 = 10.0;

    let mut tau_outer_curr = tau_surf * TAU_OUTER_FACTOR;
    let mut errtol_curr = errtol;
    let mut last_err = None;

    for _ in 0..MAX_TRIES {
        match eval_varying_data_try(
            tau_surf,
            teff,
            g,
            tau_outer_curr,
            t_tau_id,
            eos_proc,
            kap_proc,
            errtol_curr,
            max_steps,
        ) {
            Ok(pair) => return Ok(pair),
            Err(e) => {
                last_err = Some(e);
                tau_outer_curr *= TRY_SCALE;
                errtol_curr = errtol * TRY_SCALE;
            }
        }
    }

    Err(last_err.unwrap())
}

/// Evaluates surface $(\ln T, \ln P)$ by integrating hydrostatic equilibrium
/// $d\ln P / d\ln\tau = \tau g / (\kappa P)$ with depth-varying opacity (`eval_T_tau_varying` in `atm_t_tau_varying.f90`).
///
/// # Errors
/// Returns [`AtmError`] if $L, R, M \le 0$ or the 3-attempt `dopri5` integration fails.
#[allow(clippy::too_many_arguments)]
pub fn eval_t_tau_varying<E, K>(
    tau_surf: f64,
    l: f64,
    r: f64,
    m: f64,
    cgrav: f64,
    t_tau_id: i32,
    mut eos_proc: E,
    mut kap_proc: K,
    errtol: f64,
    max_steps: usize,
    skip_partials: bool,
    teff: f64,
) -> Result<AtmEvalOutput, AtmError>
where
    E: FnMut(f64, f64) -> Result<AtmEosCallbackResult, AtmError>,
    K: FnMut(
        f64,
        f64,
        &AtmEosCallbackResult,
    ) -> Result<AtmKapCallbackResult, AtmError>,
{
    if l <= 0.0 || r <= 0.0 || m <= 0.0 {
        return Err(AtmError::InvalidArgument {
            context: "eval_t_tau_varying",
            detail: "require L, R, M > 0".to_string(),
        });
    }
    let g = cgrav * m / (r * r);

    if skip_partials {
        let (ln_t, ln_p) = eval_varying_data(
            tau_surf,
            teff,
            g,
            t_tau_id,
            &mut eos_proc,
            &mut kap_proc,
            errtol,
            max_steps,
        )?;
        return Ok(AtmEvalOutput {
            ln_t,
            dln_t_dl: 0.0,
            dln_t_dln_r: 0.0,
            dln_t_dln_m: 0.0,
            dln_t_dln_kap: 0.0,
            ln_p,
            dln_p_dl: 0.0,
            dln_p_dln_r: 0.0,
            dln_p_dln_m: 0.0,
            dln_p_dln_kap: 0.0,
            kap: 0.0,
        });
    }

    const DLNTEFF: f64 = 1.0e-4;
    let ln_teff = teff.ln();
    let ln_teff_p = ln_teff + DLNTEFF;
    let ln_teff_m = ln_teff - DLNTEFF;

    let (ln_t, ln_p) = eval_varying_data(
        tau_surf,
        ln_teff.exp(),
        g,
        t_tau_id,
        &mut eos_proc,
        &mut kap_proc,
        errtol,
        max_steps,
    )?;
    let (ln_t_p, ln_p_p) = eval_varying_data(
        tau_surf,
        ln_teff_p.exp(),
        g,
        t_tau_id,
        &mut eos_proc,
        &mut kap_proc,
        errtol,
        max_steps,
    )?;
    let (ln_t_m, ln_p_m) = eval_varying_data(
        tau_surf,
        ln_teff_m.exp(),
        g,
        t_tau_id,
        &mut eos_proc,
        &mut kap_proc,
        errtol,
        max_steps,
    )?;

    let dln_teff_dln_r = -0.5;
    let dln_teff_dl = 0.25 / l;

    let dln_t_dln_teff = (ln_t_p - ln_t_m) / (ln_teff_p - ln_teff_m);
    let dln_t_dl = dln_t_dln_teff * dln_teff_dl;
    let dln_t_dln_r = dln_t_dln_teff * dln_teff_dln_r;

    let dln_p_dln_teff = (ln_p_p - ln_p_m) / (ln_teff_p - ln_teff_m);
    let dln_p_dl = dln_p_dln_teff * dln_teff_dl;
    let dln_p_dln_r = dln_p_dln_teff * dln_teff_dln_r;

    Ok(AtmEvalOutput {
        ln_t,
        dln_t_dl,
        dln_t_dln_r,
        dln_t_dln_m: 0.0,
        dln_t_dln_kap: 0.0,
        ln_p,
        dln_p_dl,
        dln_p_dln_r,
        dln_p_dln_m: 0.0,
        dln_p_dln_kap: 0.0,
        kap: 0.0,
    })
}

/// Builds the outward-to-inward 19-column atmosphere structure profile under
/// depth-varying opacity (`build_T_tau_varying` in `atm_t_tau_varying.f90`).
///
/// # Errors
/// Returns [`AtmError`] if $L, R, M \le 0$, `dlogtau <= 0`, or `dopri5` fails.
#[allow(clippy::too_many_arguments)]
pub fn build_t_tau_varying<E, K>(
    tau_surf: f64,
    l: f64,
    r: f64,
    teff: f64,
    m: f64,
    cgrav: f64,
    ln_p_surf: f64,
    tau_outer: f64,
    t_tau_id: i32,
    mut eos_proc: E,
    mut kap_proc: K,
    errtol: f64,
    dlogtau: f64,
) -> Result<AtmStructure, AtmError>
where
    E: FnMut(f64, f64) -> Result<AtmEosCallbackResult, AtmError>,
    K: FnMut(
        f64,
        f64,
        &AtmEosCallbackResult,
    ) -> Result<AtmKapCallbackResult, AtmError>,
{
    if l <= 0.0 || r <= 0.0 || m <= 0.0 || dlogtau <= 0.0 {
        return Err(AtmError::InvalidArgument {
            context: "build_t_tau_varying",
            detail: "require L, R, M, dlogtau > 0".to_string(),
        });
    }
    if tau_outer > tau_surf {
        return Ok(AtmStructure { points: Vec::new() });
    }
    let g = cgrav * m / (r * r);
    let lntau_outer = tau_outer.ln();
    let lntau_surf = tau_surf.ln();
    let dlntau_max = dlogtau * LN10;
    let dlntau = -0.5 * dlntau_max;

    let mut build_single =
        |lntau: f64,
         delta_r: f64,
         ln_p: f64|
         -> Result<[f64; NUM_RESULTS_FOR_BUILD_ATM], AtmError> {
            let tau = lntau.exp();
            let ln_t = eval_t_tau(t_tau_id, tau, teff)?;
            let eos_res = eos_proc(ln_p, ln_t)?;
            let kap_res = kap_proc(eos_res.ln_rho, ln_t, &eos_res)?;
            let gradr = eval_paczynski_gradr(
                ln_t.exp(),
                ln_p.exp(),
                eos_res.ln_rho.exp(),
                tau,
                kap_res.kap,
                l,
                m,
                r,
                cgrav,
            );
            let mut sgl = [0.0_f64; NUM_RESULTS_FOR_BUILD_ATM];
            sgl[ATM_XM] = 0.0;
            sgl[ATM_DELTA_R] = delta_r;
            sgl[ATM_LNP] = ln_p;
            sgl[ATM_LND] = eos_res.ln_rho;
            sgl[ATM_LNT] = ln_t;
            sgl[ATM_GRADT] = gradr;
            sgl[ATM_KAP] = kap_res.kap;
            sgl[ATM_GAMMA1] = eos_res.res[I_GAMMA1];
            sgl[ATM_GRADA] = eos_res.res[I_GRAD_AD];
            sgl[ATM_CHIT] = eos_res.res[I_CHI_T];
            sgl[ATM_CHIRHO] = eos_res.res[I_CHI_RHO];
            sgl[ATM_CP] = eos_res.res[I_CP];
            sgl[ATM_CV] = eos_res.res[I_CV];
            sgl[ATM_TAU] = tau;
            sgl[ATM_LNFREE_E] = eos_res.res[I_LN_FREE_E];
            sgl[ATM_DLNKAP_DLNT] = kap_res.dlnkap_dln_t;
            sgl[ATM_DLNKAP_DLND] = kap_res.dlnkap_dln_rho;
            sgl[ATM_LNPGAS] = eos_res.res[I_LN_PGAS];
            sgl[ATM_GRADR] = gradr;
            Ok(sgl)
        };

    let dopri = dopri5(
        lntau_surf,
        lntau_outer,
        &[0.0, ln_p_surf],
        dlntau,
        dlntau_max,
        errtol,
        errtol,
        0,
        |x, y| {
            let sgl = build_single(x, y[0], y[1]).map_err(|_| {
                NumError::NonFiniteValue(
                    "callback failed in build_t_tau_varying",
                )
            })?;
            let tau = x.exp();
            let p = y[1].exp();
            let rho = sgl[ATM_LND].exp();
            let kap = sgl[ATM_KAP];
            Ok(vec![-tau / (kap * rho), tau * g / (kap * p)])
        },
    )
    .map_err(|e| AtmError::ConvergenceFailed {
        context: "build_t_tau_varying",
        detail: e.to_string(),
    })?;

    let mut points = Vec::with_capacity(dopri.history.len() + 1);
    points.push(build_single(lntau_surf, 0.0, ln_p_surf)?);
    for (x_old, h, rc) in &dopri.history {
        let x_new = x_old + h;
        let dr_new = rc[0][0] + rc[1][0];
        let lnp_new = rc[0][1] + rc[1][1];
        points.push(build_single(x_new, dr_new, lnp_new)?);
    }
    points.reverse();
    Ok(AtmStructure { points })
}
