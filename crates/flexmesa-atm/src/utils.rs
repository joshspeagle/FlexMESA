// SPDX-License-Identifier: LGPL-3.0-or-later
//! Blackbody relations, Paczynski radiative gradient dilution, and $E_2(x)$
//! exponential integral interpolation (`atm_utils.f90` and `atm_lib.f90`).

use crate::table::AtmTables;
use crate::types::AtmError;
use flexmesa_const::{BOLTZ_SIGMA, CLIGHT, CRAD, LN10, PI};
use flexmesa_interp1d::interp_value_and_slope;
use flexmesa_math::{exp10, pow};

/// Computes the blackbody effective temperature $T_{\mathrm{eff}} = (L / (4 \pi R^2 \sigma_{\mathrm{SB}}))^{1/4}$ (`atm_black_body_T`).
#[must_use]
pub fn atm_black_body_t(l: f64, r: f64) -> f64 {
    pow(l / (4.0 * PI * r * r * BOLTZ_SIGMA), 0.25)
}

/// Alias for [`atm_black_body_t`] (`atm_Teff` in `atm_lib.f90`).
#[must_use]
pub fn atm_teff(l: f64, r: f64) -> f64 {
    atm_black_body_t(l, r)
}

/// Computes the emergent luminosity $L = 4 \pi R^2 \sigma_{\mathrm{SB}} T_{\mathrm{eff}}^4$ (`atm_L` in `atm_lib.f90`).
#[must_use]
pub fn atm_l(teff: f64, r: f64) -> f64 {
    4.0 * PI * r * r * BOLTZ_SIGMA * teff * teff * teff * teff
}

/// Computes $(T_{\mathrm{eff}}, g)$ from $(L, R, M, G)$ (`eval_Teff_g` in `atm_utils.f90`).
#[must_use]
pub fn eval_teff_g(l: f64, r: f64, m: f64, cgrav: f64) -> (f64, f64) {
    let teff = atm_black_body_t(l, r);
    let g = cgrav * m / (r * r);
    (teff, g)
}

/// Computes radiation pressure $P_{\mathrm{rad}} = \tfrac{1}{3} a T^4$ ($\mathrm{dyn\,cm^{-2}}$).
#[must_use]
pub fn radiation_pressure(t: f64) -> f64 {
    CRAD * t * t * t * t / 3.0
}

/// Evaluates the radiative temperature gradient $\nabla_{\mathrm{rad}}$ including the
/// Paczynski (1969, Acta Astron. 19, 1) optical-depth dilution factor for $\tau < 2/3$
/// (`eval_Paczynski_gradr` in `atm_utils.f90`).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn eval_paczynski_gradr(
    t: f64,
    p: f64,
    rho: f64,
    tau: f64,
    kap: f64,
    l: f64,
    m: f64,
    r: f64,
    cgrav: f64,
) -> f64 {
    let prad = radiation_pressure(t);
    let mut gradr = p * kap * l / (16.0 * PI * CLIGHT * m * cgrav * prad);
    if tau < 2.0 / 3.0 {
        let s = (2.0 * CRAD * t * t * t * r.sqrt()) / (3.0 * cgrav * m * rho)
            * pow(l / (8.0 * PI * BOLTZ_SIGMA), 0.25);
        let f = 1.0 - 1.5 * tau;
        let dilution_factor = (1.0
            + f * s * (4.0 * PI * cgrav * CLIGHT * m) / (kap * l))
            / (1.0 + f * s);
        gradr *= dilution_factor;
    }
    gradr
}

/// Evaluates the second exponential integral $E_2(x)$ and its derivative $dE_2/dx$
/// using the 571-point Steffen monotonic cubic spline of $\log_{10} E_2(x)$
/// (`eval_E2` in `atm_utils.f90`).
///
/// # Errors
/// Returns [`AtmError::CallbackError`] if `x` lies outside `[0, 10]` or spline evaluation fails.
pub fn eval_e2(x: f64) -> Result<(f64, f64), AtmError> {
    let tables = AtmTables::builtin();
    let (val, slope) = interp_value_and_slope(&tables.e2_x, &tables.e2_f, x)
        .map_err(|e| AtmError::CallbackError {
            context: "eval_e2",
            detail: e.to_string(),
        })?;
    let e2 = exp10(val);
    let de2_dx = slope * LN10 * e2;
    Ok((e2, de2_dx))
}
