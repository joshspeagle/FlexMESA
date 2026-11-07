// SPDX-License-Identifier: LGPL-3.0-or-later
//! Grey $T(\tau)$ atmosphere with uniform opacity (`atm_t_tau_uniform.f90`).
//!
//! Ports `eval_T_tau_uniform` (0.5-underrelaxed fixed-point opacity iteration
//! with implicit partial derivative fixup) and `build_T_tau_uniform` (outward
//! `dopri5` integration of $dr / d\ln\tau = -\tau / (\kappa \rho)$).

use crate::t_tau_relations::eval_t_tau;
use crate::types::{
    AtmEosCallbackResult, AtmError, AtmEvalOutput, AtmKapCallbackResult,
    AtmStructure, ATM_CHIRHO, ATM_CHIT, ATM_CP, ATM_CV, ATM_DELTA_R,
    ATM_DLNKAP_DLND, ATM_DLNKAP_DLNT, ATM_GAMMA1, ATM_GRADA, ATM_GRADR,
    ATM_GRADT, ATM_KAP, ATM_LND, ATM_LNFREE_E, ATM_LNP, ATM_LNPGAS, ATM_LNT,
    ATM_TAU, ATM_XM, NUM_RESULTS_FOR_BUILD_ATM,
};
use crate::utils::eval_paczynski_gradr;
use flexmesa_const::{CLIGHT, LN10, PI};
use flexmesa_eos::{
    I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_GAMMA1, I_GRAD_AD, I_LN_FREE_E, I_LN_PGAS,
};
use flexmesa_num::{dopri5, NumError};
use flexmesa_utils::is_bad;

#[allow(clippy::too_many_arguments)]
fn eval_uniform_data(
    tau: f64,
    teff: f64,
    g: f64,
    l: f64,
    m: f64,
    cgrav: f64,
    kap: f64,
    pextra_factor: f64,
    t_tau_id: i32,
    skip_partials: bool,
) -> Result<AtmEvalOutput, AtmError> {
    let p0 = tau * g / kap;
    let pextra =
        pextra_factor * (kap / tau) * (l / m) / (6.0 * PI * CLIGHT * cgrav);
    let pfactor = 1.0 + pextra;
    let p = p0 * pfactor;
    let ln_p = p.ln();
    if is_bad(ln_p) {
        return Err(AtmError::CallbackError {
            context: "eval_uniform_data",
            detail: format!("bad lnP={ln_p}"),
        });
    }
    let ln_t = eval_t_tau(t_tau_id, tau, teff)?;
    if is_bad(ln_t) {
        return Err(AtmError::CallbackError {
            context: "eval_uniform_data",
            detail: format!("bad lnT={ln_t}"),
        });
    }

    if skip_partials {
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
            kap,
        });
    }

    let dlogg_dln_r = -2.0;
    let dlogg_dln_m = 1.0;

    let dp0_dl = 0.0;
    let dp0_dln_r = dlogg_dln_r * p0;
    let dp0_dln_m = dlogg_dln_m * p0;
    let dp0_dln_kap = -p0;

    let dpfactor_dl = pextra / l;
    let dpfactor_dln_r = 0.0;
    let dpfactor_dln_m = -pextra;
    let dpfactor_dln_kap = pextra;

    let dln_p_dl = (dp0_dl * pfactor + p0 * dpfactor_dl) / p;
    let dln_p_dln_r = (dp0_dln_r * pfactor + p0 * dpfactor_dln_r) / p;
    let dln_p_dln_m = (dp0_dln_m * pfactor + p0 * dpfactor_dln_m) / p;
    let dln_p_dln_kap = (dp0_dln_kap * pfactor + p0 * dpfactor_dln_kap) / p;

    let dln_teff_dl = 0.25 / l;
    let dln_teff_dln_r = -0.5;
    let dln_t_dln_teff = 1.0;

    let dln_t_dl = dln_teff_dl * dln_t_dln_teff;
    let dln_t_dln_r = dln_teff_dln_r * dln_t_dln_teff;

    Ok(AtmEvalOutput {
        ln_t,
        dln_t_dl,
        dln_t_dln_r,
        dln_t_dln_m: 0.0,
        dln_t_dln_kap: 0.0,
        ln_p,
        dln_p_dl,
        dln_p_dln_r,
        dln_p_dln_m,
        dln_p_dln_kap,
        kap,
    })
}

/// Evaluates surface $(\ln T, \ln P, \kappa)$ from a $T(\tau)$ relation with uniform
/// opacity (`eval_T_tau_uniform` in `atm_t_tau_uniform.f90`).
///
/// When `max_iters == 0`, uses `kap_guess` directly without calling `eos_proc`
/// or `kap_proc`. When `max_iters > 0`, iterates $\kappa$ with 0.5 underrelaxation
/// until $|\kappa_{\mathrm{prev}} - \kappa| / (\mathrm{errtol} + \mathrm{errtol}\,\kappa) < 1$
/// and applies the implicit partial derivative correction.
///
/// # Errors
/// Returns [`AtmError`] on invalid arguments, callback errors, or non-convergence.
#[allow(clippy::too_many_arguments)]
pub fn eval_t_tau_uniform<E, K>(
    tau_surf: f64,
    l: f64,
    r: f64,
    m: f64,
    cgrav: f64,
    kap_guess: f64,
    pextra_factor: f64,
    t_tau_id: i32,
    mut eos_proc: E,
    mut kap_proc: K,
    errtol: f64,
    max_iters: usize,
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
            context: "eval_t_tau_uniform",
            detail: "require L, R, M > 0".to_string(),
        });
    }
    let g = cgrav * m / (r * r);
    let mut kap = kap_guess;
    let mut out = eval_uniform_data(
        tau_surf,
        teff,
        g,
        l,
        m,
        cgrav,
        kap,
        pextra_factor,
        t_tau_id,
        skip_partials,
    )?;

    if max_iters == 0 {
        return Ok(out);
    }

    let mut converged = false;
    let mut last_eos = None;
    let mut last_kap = None;

    for _ in 1..=max_iters {
        let eos_res = eos_proc(out.ln_p, out.ln_t)?;
        let kap_prev = kap;
        let kap_res = kap_proc(eos_res.ln_rho, out.ln_t, &eos_res)?;
        kap = kap_res.kap;

        let err = (kap_prev - kap).abs() / (errtol + errtol * kap);
        last_eos = Some(eos_res);
        last_kap = Some(kap_res);

        if err < 1.0 {
            converged = true;
            out.kap = kap;
            break;
        }

        kap = kap_prev + 0.5 * (kap - kap_prev);
        out = eval_uniform_data(
            tau_surf,
            teff,
            g,
            l,
            m,
            cgrav,
            kap,
            pextra_factor,
            t_tau_id,
            skip_partials,
        )?;
    }

    if !converged {
        return Err(AtmError::ConvergenceFailed {
            context: "eval_t_tau_uniform",
            detail: format!("exceeded max_iters={max_iters}"),
        });
    }

    if !skip_partials {
        let eos_res = last_eos.as_ref().unwrap();
        let kap_res = last_kap.as_ref().unwrap();
        let chi_rho = eos_res.res[I_CHI_RHO];
        let chi_t = eos_res.res[I_CHI_T];

        let dlnkap_dlnp_t = kap_res.dlnkap_dln_rho / chi_rho;
        let dlnkap_dlnt_p =
            kap_res.dlnkap_dln_t - kap_res.dlnkap_dln_rho * chi_t / chi_rho;

        let denom = 1.0
            - dlnkap_dlnp_t * out.dln_p_dln_kap
            - dlnkap_dlnt_p * out.dln_t_dln_kap;

        let dln_p_dl = (out.dln_p_dl
            + dlnkap_dlnt_p
                * (out.dln_p_dln_kap * out.dln_t_dl
                    - out.dln_p_dl * out.dln_t_dln_kap))
            / denom;
        let dln_t_dl = (out.dln_t_dl
            + dlnkap_dlnp_t
                * (out.dln_t_dln_kap * out.dln_p_dl
                    - out.dln_t_dl * out.dln_p_dln_kap))
            / denom;

        let dln_p_dln_r = (out.dln_p_dln_r
            + dlnkap_dlnt_p
                * (out.dln_p_dln_kap * out.dln_t_dln_r
                    - out.dln_p_dln_r * out.dln_t_dln_kap))
            / denom;
        let dln_t_dln_r = (out.dln_t_dln_r
            + dlnkap_dlnp_t
                * (out.dln_t_dln_kap * out.dln_p_dln_r
                    - out.dln_t_dln_r * out.dln_p_dln_kap))
            / denom;

        let dln_p_dln_m = (out.dln_p_dln_m
            + dlnkap_dlnt_p
                * (out.dln_p_dln_kap * out.dln_t_dln_m
                    - out.dln_p_dln_m * out.dln_t_dln_kap))
            / denom;
        let dln_t_dln_m = (out.dln_t_dln_m
            + dlnkap_dlnp_t
                * (out.dln_t_dln_kap * out.dln_p_dln_m
                    - out.dln_t_dln_m * out.dln_p_dln_kap))
            / denom;

        out.dln_p_dl = dln_p_dl;
        out.dln_t_dl = dln_t_dl;
        out.dln_p_dln_r = dln_p_dln_r;
        out.dln_t_dln_r = dln_t_dln_r;
        out.dln_p_dln_m = dln_p_dln_m;
        out.dln_t_dln_m = dln_t_dln_m;
        out.dln_p_dln_kap = 0.0;
        out.dln_t_dln_kap = 0.0;
    }

    Ok(out)
}

/// Builds the outward-to-inward 19-column atmosphere structure profile using
/// `dopri5` under uniform opacity (`build_T_tau_uniform` in `atm_t_tau_uniform.f90`).
///
/// # Errors
/// Returns [`AtmError`] if `dlogtau <= 0` or ODE integration fails.
#[allow(clippy::too_many_arguments)]
pub fn build_t_tau_uniform<E>(
    tau_surf: f64,
    l: f64,
    r: f64,
    teff: f64,
    m: f64,
    cgrav: f64,
    kap: f64,
    pextra_factor: f64,
    tau_outer: f64,
    t_tau_id: i32,
    mut eos_proc: E,
    errtol: f64,
    dlogtau: f64,
) -> Result<AtmStructure, AtmError>
where
    E: FnMut(f64, f64) -> Result<AtmEosCallbackResult, AtmError>,
{
    if dlogtau <= 0.0 {
        return Err(AtmError::InvalidArgument {
            context: "build_t_tau_uniform",
            detail: format!("require dlogtau > 0, got {dlogtau}"),
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
         delta_r: f64|
         -> Result<[f64; NUM_RESULTS_FOR_BUILD_ATM], AtmError> {
            let tau = lntau.exp();
            let ev = eval_uniform_data(
                tau,
                teff,
                g,
                l,
                m,
                cgrav,
                kap,
                pextra_factor,
                t_tau_id,
                true,
            )?;
            let eos_res = eos_proc(ev.ln_p, ev.ln_t)?;
            let gradr = eval_paczynski_gradr(
                ev.ln_t.exp(),
                ev.ln_p.exp(),
                eos_res.ln_rho.exp(),
                tau,
                kap,
                l,
                m,
                r,
                cgrav,
            );
            let mut sgl = [0.0_f64; NUM_RESULTS_FOR_BUILD_ATM];
            sgl[ATM_XM] = 0.0;
            sgl[ATM_DELTA_R] = delta_r;
            sgl[ATM_LNP] = ev.ln_p;
            sgl[ATM_LND] = eos_res.ln_rho;
            sgl[ATM_LNT] = ev.ln_t;
            sgl[ATM_GRADT] = gradr;
            sgl[ATM_KAP] = kap;
            sgl[ATM_GAMMA1] = eos_res.res[I_GAMMA1];
            sgl[ATM_GRADA] = eos_res.res[I_GRAD_AD];
            sgl[ATM_CHIT] = eos_res.res[I_CHI_T];
            sgl[ATM_CHIRHO] = eos_res.res[I_CHI_RHO];
            sgl[ATM_CP] = eos_res.res[I_CP];
            sgl[ATM_CV] = eos_res.res[I_CV];
            sgl[ATM_TAU] = tau;
            sgl[ATM_LNFREE_E] = eos_res.res[I_LN_FREE_E];
            sgl[ATM_DLNKAP_DLNT] = 0.0;
            sgl[ATM_DLNKAP_DLND] = 0.0;
            sgl[ATM_LNPGAS] = eos_res.res[I_LN_PGAS];
            sgl[ATM_GRADR] = gradr;
            Ok(sgl)
        };

    let dopri = dopri5(
        lntau_surf,
        lntau_outer,
        &[0.0],
        dlntau,
        dlntau_max,
        errtol,
        errtol,
        0,
        |x, y| {
            let sgl = build_single(x, y[0]).map_err(|_| {
                NumError::NonFiniteValue(
                    "callback failed in build_t_tau_uniform",
                )
            })?;
            let tau = x.exp();
            let rho = sgl[ATM_LND].exp();
            Ok(vec![-tau / (kap * rho)])
        },
    )
    .map_err(|e| AtmError::ConvergenceFailed {
        context: "build_t_tau_uniform",
        detail: e.to_string(),
    })?;

    let mut points = Vec::with_capacity(dopri.history.len() + 1);
    points.push(build_single(lntau_surf, 0.0)?);
    for (x_old, h, rc) in &dopri.history {
        let x_new = x_old + h;
        let y_new = rc[0][0] + rc[1][0];
        points.push(build_single(x_new, y_new)?);
    }
    points.reverse();
    Ok(AtmStructure { points })
}
