// SPDX-License-Identifier: LGPL-3.0-or-later
//! Grey irradiated atmosphere boundary condition (`atm_irradiated.f90`).
//!
//! Ports `eval_irradiated` from upstream MESA `atm/private/atm_irradiated.f90`,
//! implementing Eq. 6 of Guillot & Havel (2011, A&A 527, A20) with $E_2(\gamma\tau)$
//! exponential integral interpolation, 0.5-underrelaxed thermal opacity iteration,
//! and implicit temperature derivative correction.

use crate::types::{
    AtmEosCallbackResult, AtmError, AtmIrradiatedOutput, AtmKapCallbackResult,
};
use crate::utils::{eval_e2, eval_teff_g};
use flexmesa_eos::{I_CHI_RHO, I_CHI_T};

#[allow(clippy::too_many_arguments)]
fn eval_irradiated_data(
    t_int: f64,
    g: f64,
    l: f64,
    t_eq: f64,
    p_surf: f64,
    kap: f64,
    kap_v: f64,
    gamma: f64,
    skip_partials: bool,
) -> Result<AtmIrradiatedOutput, AtmError> {
    let tau = p_surf * kap / g;
    let gamma_eff = if gamma > 0.0 { gamma } else { kap_v / kap };
    let x = gamma_eff * tau;

    let (e2, de2_dx) = eval_e2(x)?;

    let f_1 = 2.0 * (1.0 + (x / 2.0 - 1.0) * (-x).exp()) * tau / (3.0 * x);
    let f_2 = 2.0 * x * (1.0 - tau * tau / 2.0) * e2 / (3.0 * tau);

    let t4_int = t_int * t_int * t_int * t_int;
    let t4_eq = t_eq * t_eq * t_eq * t_eq;
    let t4 =
        0.75 * (t4_int * (tau + 2.0 / 3.0) + t4_eq * (f_1 + f_2 + 2.0 / 3.0));
    let ln_t = 0.25 * t4.ln();

    if skip_partials {
        return Ok(AtmIrradiatedOutput {
            kap,
            tau,
            ln_t,
            dln_t_dl: 0.0,
            dln_t_dln_r: 0.0,
            dln_t_dln_m: 0.0,
            dln_t_dln_kap: 0.0,
        });
    }

    let df_1_dtau = (2.0 + (x - 2.0) * (-x).exp()) / (3.0 * x);
    let df_2_dtau = -x * (2.0 + tau * tau) * e2 / (3.0 * tau * tau);

    let df_1_dx =
        -(2.0 - (2.0 + (2.0 - x) * x) * (-x).exp()) * tau / (3.0 * x * x);
    let df_2_dx = -(tau * tau - 2.0) * (e2 + x * de2_dx) / (3.0 * tau);

    let dln_t_dln_t_int = 0.75 * (tau + 2.0 / 3.0) * t4_int / t4;
    let dln_t_dln_tau =
        0.75 * (t4_int + t4_eq * (df_1_dtau + df_2_dtau)) * tau / (4.0 * t4);
    let dln_t_dln_x = 0.75 * t4_eq * (df_1_dx + df_2_dx) * x / (4.0 * t4);

    let dln_t_int_dln_l = 0.25;
    let dln_t_int_dln_r = -0.5;
    let dln_t_int_dln_m = 0.0;
    let dln_t_int_dln_kap = 0.0;

    let dln_tau_dln_l = 0.0;
    let dln_tau_dln_r = 2.0;
    let dln_tau_dln_m = -1.0;
    let dln_tau_dln_kap = 1.0;

    let dln_x_dln_l = 0.0;
    let dln_x_dln_r = 2.0;
    let dln_x_dln_m = -1.0;
    let dln_x_dln_kap = if gamma > 0.0 { 1.0 } else { 0.0 };

    let dln_t_dl = (dln_t_dln_t_int * dln_t_int_dln_l
        + dln_t_dln_tau * dln_tau_dln_l
        + dln_t_dln_x * dln_x_dln_l)
        / l;
    let dln_t_dln_r = dln_t_dln_t_int * dln_t_int_dln_r
        + dln_t_dln_tau * dln_tau_dln_r
        + dln_t_dln_x * dln_x_dln_r;
    let dln_t_dln_m = dln_t_dln_t_int * dln_t_int_dln_m
        + dln_t_dln_tau * dln_tau_dln_m
        + dln_t_dln_x * dln_x_dln_m;
    let dln_t_dln_kap = dln_t_dln_t_int * dln_t_int_dln_kap
        + dln_t_dln_tau * dln_tau_dln_kap
        + dln_t_dln_x * dln_x_dln_kap;

    Ok(AtmIrradiatedOutput {
        kap,
        tau,
        ln_t,
        dln_t_dl,
        dln_t_dln_r,
        dln_t_dln_m,
        dln_t_dln_kap,
    })
}

/// Evaluates an irradiated grey atmosphere at fixed surface pressure $P_{\mathrm{surf}}$
/// (`eval_irradiated` in `atm_irradiated.f90`).
///
/// # Errors
/// Returns [`AtmError`] if $L, R, M \le 0$, callbacks fail, or opacity iteration does not converge.
#[allow(clippy::too_many_arguments)]
pub fn eval_irradiated<E, K>(
    l: f64,
    r: f64,
    m: f64,
    cgrav: f64,
    t_eq: f64,
    p_surf: f64,
    kap_guess: f64,
    kap_v: f64,
    gamma: f64,
    mut eos_proc: E,
    mut kap_proc: K,
    errtol: f64,
    max_iters: usize,
    skip_partials: bool,
) -> Result<AtmIrradiatedOutput, AtmError>
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
            context: "eval_irradiated",
            detail: "require L, R, M > 0".to_string(),
        });
    }
    let (t_int, g) = eval_teff_g(l, r, m, cgrav);
    let mut kap = kap_guess;
    let mut out = eval_irradiated_data(
        t_int,
        g,
        l,
        t_eq,
        p_surf,
        kap,
        kap_v,
        gamma,
        skip_partials,
    )?;

    if max_iters == 0 {
        return Ok(out);
    }

    let ln_p = p_surf.ln();
    let mut converged = false;
    let mut last_eos = None;
    let mut last_kap = None;

    for _ in 1..=max_iters {
        let eos_res = eos_proc(ln_p, out.ln_t)?;
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
        out = eval_irradiated_data(
            t_int,
            g,
            l,
            t_eq,
            p_surf,
            kap,
            kap_v,
            gamma,
            skip_partials,
        )?;
    }

    if !converged {
        return Err(AtmError::ConvergenceFailed {
            context: "eval_irradiated",
            detail: format!("exceeded max_iters={max_iters}"),
        });
    }

    if !skip_partials {
        let eos_res = last_eos.as_ref().unwrap();
        let kap_res = last_kap.as_ref().unwrap();
        let chi_rho = eos_res.res[I_CHI_RHO];
        let chi_t = eos_res.res[I_CHI_T];
        let dlnkap_dlnt_p =
            kap_res.dlnkap_dln_t - kap_res.dlnkap_dln_rho * chi_t / chi_rho;
        let denom = 1.0 - dlnkap_dlnt_p * out.dln_t_dln_kap;
        out.dln_t_dl /= denom;
        out.dln_t_dln_r /= denom;
        out.dln_t_dln_m /= denom;
        out.dln_t_dln_kap = 0.0;
    }

    Ok(out)
}
