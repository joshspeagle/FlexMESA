// SPDX-License-Identifier: LGPL-3.0-or-later
//! Grey and solar-calibrated $T(\tau)$ relations (`atm_t_tau_relations.f90`).
//!
//! Ports `get_T_tau_base`, `eval_T_tau`, and `eval_T_tau_dq_dtau` across all four
//! MESA $T(\tau)$ relations:
//! - [`ATM_T_TAU_EDDINGTON`] ($q(\tau) = 2/3$)
//! - [`ATM_T_TAU_SOLAR_HOPF`] (Sonoi et al. 2019, A&A 621, A84 / MESA II Sec. A.5)
//! - [`ATM_T_TAU_KRISHNA_SWAMY`] (Krishna-Swamy 1966, ApJ 145, 174)
//! - [`ATM_T_TAU_TRAMPEDACH_SOLAR`] (Ball 2021, RNAAS 5, 7 fit to Trampedach et al. 2014)

use crate::types::{
    AtmError, ATM_T_TAU_EDDINGTON, ATM_T_TAU_KRISHNA_SWAMY,
    ATM_T_TAU_SOLAR_HOPF, ATM_T_TAU_TRAMPEDACH_SOLAR,
};
use flexmesa_const::{ILN10, TWO_THIRDS};

/// Returns the base optical depth $\tau_{\mathrm{base}}$ where $T(\tau_{\mathrm{base}}) = T_{\mathrm{eff}}$ (`get_T_tau_base`).
///
/// # Errors
/// Returns [`AtmError::UnknownId`] if `id` is not one of the four $T(\tau)$ constants.
pub fn get_t_tau_base(id: i32) -> Result<f64, AtmError> {
    match id {
        ATM_T_TAU_EDDINGTON => Ok(2.0 / 3.0),
        ATM_T_TAU_SOLAR_HOPF => Ok(0.411_643_350_2),
        ATM_T_TAU_KRISHNA_SWAMY => Ok(0.312_156_3),
        ATM_T_TAU_TRAMPEDACH_SOLAR => Ok(0.514_792_905_805_714_7),
        _ => Err(AtmError::UnknownId {
            context: "get_t_tau_base",
            id,
        }),
    }
}

/// Evaluates $\ln(T / \mathrm{K})$ at optical depth $\tau$ and effective temperature $T_{\mathrm{eff}}$ (`eval_T_tau`).
///
/// For any relation $(T / T_{\mathrm{eff}})^4 = \tfrac{3}{4} [\tau + q(\tau)]$, computes
/// $\ln T = \tfrac{1}{4} \ln(T^4)$ in exact upstream MESA operation order.
///
/// # Errors
/// Returns [`AtmError::UnknownId`] if `id` is not one of the four $T(\tau)$ constants.
pub fn eval_t_tau(id: i32, tau: f64, teff: f64) -> Result<f64, AtmError> {
    let teff4 = teff * teff * teff * teff;
    let t4 = match id {
        ATM_T_TAU_EDDINGTON => 0.75 * teff4 * (tau + TWO_THIRDS),
        ATM_T_TAU_SOLAR_HOPF => {
            const Q1: f64 = 1.0361;
            const Q2: f64 = -0.3134;
            const Q3: f64 = 2.447_999_95;
            const Q4: f64 = -0.295_899_99;
            const Q5: f64 = 30.0;
            let e1 = (-Q3 * tau).exp();
            let e2 = (-Q5 * tau).exp();
            0.75 * teff4 * (tau + Q1 + Q2 * e1 + Q4 * e2)
        }
        ATM_T_TAU_KRISHNA_SWAMY => {
            const Q1: f64 = 1.39;
            const Q2: f64 = -0.815;
            const Q3: f64 = 2.54;
            const Q4: f64 = -0.025;
            const Q5: f64 = 30.0;
            let e1 = (-Q3 * tau).exp();
            let e2 = (-Q5 * tau).exp();
            0.75 * teff4 * (tau + Q1 + Q2 * e1 + Q4 * e2)
        }
        ATM_T_TAU_TRAMPEDACH_SOLAR => {
            const C0: f64 = 0.688_730_200_592_965_6;
            const C1: f64 = 0.066_869_786_083_344_9;
            const A: f64 = 0.926_212_649_769_125;
            const V: f64 = 0.765_785_689_340_246_6;
            const B: f64 = 0.114_874_290_276_943_3;
            let x = tau.log10();
            0.75 * teff4 * (tau + C0 + C1 * (x - B) + V * ((x - A) / V).exp())
        }
        _ => {
            return Err(AtmError::UnknownId {
                context: "eval_t_tau",
                id,
            });
        }
    };
    Ok(t4.ln() * 0.25)
}

/// Evaluates the Hopf function derivative $q'(\tau) = dq/d\tau$ at optical depth $\tau$ (`eval_T_tau_dq_dtau`).
///
/// # Errors
/// Returns [`AtmError::UnknownId`] if `id` is not one of the four $T(\tau)$ constants.
pub fn eval_t_tau_dq_dtau(id: i32, tau: f64) -> Result<f64, AtmError> {
    match id {
        ATM_T_TAU_EDDINGTON => Ok(0.0),
        ATM_T_TAU_SOLAR_HOPF => {
            const Q2: f64 = -0.3134;
            const Q3: f64 = 2.447_999_95;
            const Q4: f64 = -0.295_899_99;
            const Q5: f64 = 30.0;
            let e1 = (-Q3 * tau).exp();
            let e2 = (-Q5 * tau).exp();
            Ok(-Q2 * Q3 * e1 - Q4 * Q5 * e2)
        }
        ATM_T_TAU_KRISHNA_SWAMY => {
            const Q2: f64 = -0.815;
            const Q3: f64 = 2.54;
            const Q4: f64 = -0.025;
            const Q5: f64 = 30.0;
            let e1 = (-Q3 * tau).exp();
            let e2 = (-Q5 * tau).exp();
            Ok(-Q2 * Q3 * e1 - Q4 * Q5 * e2)
        }
        ATM_T_TAU_TRAMPEDACH_SOLAR => {
            const C1: f64 = 0.066_869_786_083_344_9;
            const A: f64 = 0.926_212_649_769_125;
            const V: f64 = 0.765_785_689_340_246_6;
            const B: f64 = 0.114_874_290_276_943_3;
            const W: f64 = 0.051_499_904_716_986_9;
            let x = tau.log10();
            Ok(
                (C1 + ((x - A) / V).exp()) / (1.0 + ((x - B) / W).exp()) / tau
                    * ILN10,
            )
        }
        _ => Err(AtmError::UnknownId {
            context: "eval_t_tau_dq_dtau",
            id,
        }),
    }
}
