// SPDX-License-Identifier: LGPL-3.0-or-later
//! Thermohaline (salt-finger) mixing (`set_thermohaline`, `thermohaline.f90`).
//!
//! Computes compositional diffusivity $D_{\mathrm{thrm}}$ when the thermal
//! gradient is Schwarzschild stable ($\nabla_r < \nabla_{\mathrm{ad}}$)
//! but destabilized by an inverted molecular-weight gradient
//! ($\nabla_L - \nabla_{\mathrm{ad}} < 0$):
//! - [`ThermohalineOption::Kippenhahn`]: Kippenhahn, Ruschenplatt, & Thomas
//!   (1980, A&A, 91, 175).
//! - [`ThermohalineOption::TraxlerGaraudStellmach11`]: Traxler, Garaud, &
//!   Stellmach (2011, ApJL, 728, L29, eq. 24).
//! - [`ThermohalineOption::BrownGaraudStellmach13`]: Brown, Garaud, &
//!   Stellmach (2013, ApJ, 768, 34, eqs. 19, 20, 33) solved via 2D
//!   Newton-Raphson relaxation with perturbation retries.
//!
//! # Decoupling from `chem`
//! Upstream `thermohaline.f90` queries `chem_isos%Z(iso)` and
//! `chem_isos%Z_plus_N(iso)` for the isotope driving the mean molecular weight
//! inversion. [`DrivingSpecies`] encapsulates $(Z, A)$ directly so `turb`
//! requires no compile-time dependency on the `chem` module.

use std::str::FromStr;

use flexmesa_const::{
    MixingType, AMU, BOLTZM, CLIGHT, CRAD, ONE_THIRD, PI, QE,
};
use flexmesa_math::{exp, log, pow, pow2, pow3, pow4};

use crate::{TurbError, TurbOutput, TurbResult, TurbScalar};

/// Thermohaline mixing prescription (`thermohaline_option` in
/// `thermohaline.f90`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThermohalineOption {
    /// Kippenhahn, Ruschenplatt, & Thomas (1980) linear salt-finger formula.
    Kippenhahn,
    /// Traxler, Garaud, & Stellmach (2011) empirical 3D DNS fit.
    TraxlerGaraudStellmach11,
    /// Brown, Garaud, & Stellmach (2013) quasi-linear salt-finger model.
    BrownGaraudStellmach13,
}

impl FromStr for ThermohalineOption {
    type Err = TurbError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "Kippenhahn" => Ok(Self::Kippenhahn),
            "Traxler_Garaud_Stellmach_11" => Ok(Self::TraxlerGaraudStellmach11),
            "Brown_Garaud_Stellmach_13" => Ok(Self::BrownGaraudStellmach13),
            other => {
                Err(TurbError::UnknownThermohalineOption(other.to_string()))
            }
        }
    }
}

/// Atomic charge $Z$ and mass number $A$ of the species driving thermohaline
/// mixing (`chem_isos%Z(iso)` and `chem_isos%Z_plus_N(iso)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrivingSpecies {
    /// Atomic charge number $Z$.
    pub chem_z: f64,
    /// Mass number $A = Z + N$.
    pub chem_a: f64,
}

impl DrivingSpecies {
    /// Constructs a driving species with atomic number `chem_z` and mass number
    /// `chem_a`.
    #[must_use]
    pub const fn new(chem_z: f64, chem_a: f64) -> Self {
        Self { chem_z, chem_a }
    }

    /// Standard ${}^4\mathrm{He}$ driving species ($Z = 2, A = 4$).
    #[must_use]
    pub const fn he4() -> Self {
        Self::new(2.0, 4.0)
    }

    /// Standard ${}^3\mathrm{He}$ driving species ($Z = 2, A = 3$).
    #[must_use]
    pub const fn he3() -> Self {
        Self::new(2.0, 3.0)
    }

    /// Carbon-12 driving species ($Z = 6, A = 12$).
    #[must_use]
    pub const fn c12() -> Self {
        Self::new(6.0, 12.0)
    }
}

/// Non-differentiable parameters for [`set_thermohaline`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermohalineParams {
    /// Thermohaline prescription ([`ThermohalineOption`]).
    pub option: ThermohalineOption,
    /// Composition gradient term $\nabla_L - \nabla_{\mathrm{ad}}$.
    pub gradl_composition_term: f64,
    /// Hydrogen mass fraction $X({}^1\mathrm{H})$.
    pub xh1: f64,
    /// Multiplicative efficiency factor `thermohaline_coeff`.
    pub thermohaline_coeff: f64,
    /// Driving isotope charge and mass number ([`DrivingSpecies`]).
    pub species: DrivingSpecies,
}

/// Differentiable thermodynamic and structure inputs for [`set_thermohaline`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermohalineInputs<T> {
    /// Mixing length $\Lambda$ in $\mathrm{cm}$.
    pub lambda: T,
    /// Adiabatic temperature gradient $\nabla_{\mathrm{ad}}$.
    pub grada: T,
    /// Radiative temperature gradient $\nabla_{\mathrm{rad}}$.
    pub gradr: T,
    /// Temperature $T$ in $\mathrm{K}$.
    pub t: T,
    /// Rosseland mean opacity $\kappa$ in $\mathrm{cm^2\,g^{-1}}$.
    pub opacity: T,
    /// Mass density $\rho$ in $\mathrm{g\,cm^{-3}}$.
    pub rho: T,
    /// Specific heat at constant pressure $C_P$ in
    /// $\mathrm{erg\,g^{-1}\,K^{-1}}$.
    pub cp: T,
}

/// Computes scalar thermohaline diffusivity $D_{\mathrm{thrm}}$ in
/// $\mathrm{cm^2\,s^{-1}}$ (`get_D_thermohaline` in `thermohaline.f90`).
#[must_use]
pub fn get_d_thermohaline(
    option: ThermohalineOption,
    grada: f64,
    gradr: f64,
    t: f64,
    opacity: f64,
    rho: f64,
    cp: f64,
    gradl_composition_term: f64,
    species: DrivingSpecies,
    xh1: f64,
    thermohaline_coeff: f64,
) -> f64 {
    let dgrad = (grada - gradr).max(1.0e-40);
    let k_therm = 4.0 * CRAD * CLIGHT * pow3(t) / (3.0 * opacity * rho);
    let d_raw = match option {
        ThermohalineOption::Kippenhahn => {
            -3.0 * k_therm / (2.0 * rho * cp) * gradl_composition_term / dgrad
        }
        ThermohalineOption::TraxlerGaraudStellmach11
        | ThermohalineOption::BrownGaraudStellmach13 => {
            let (k_t, k_mu, nu) =
                get_diff_coeffs(k_therm, cp, rho, t, opacity, species, xh1);
            let r0 = (gradr - grada) / gradl_composition_term;
            let pr = nu / k_t;
            let tau = k_mu / k_t;
            let r_th = (r0 - 1.0) / (1.0 / tau - 1.0);
            if r_th >= 1.0 || pr < 0.0 {
                0.0
            } else if option == ThermohalineOption::TraxlerGaraudStellmach11 {
                101.0
                    * (k_mu * nu).sqrt()
                    * exp(-3.6 * r_th)
                    * pow(1.0 - r_th, 1.1)
            } else {
                k_mu * (numu(r0, r_th, pr, tau) - 1.0)
            }
        }
    };
    thermohaline_coeff * d_raw
}

/// Evaluates thermohaline mixing outputs (`set_thermohaline` in `turb.f90`).
///
/// Note that in upstream MESA (`turb.f90` lines 66-75), `D_thrm` is evaluated
/// from the scalar values of `(grada, gradr, T, opacity, rho, Cp)`, assigned as
/// a constant `D = D_thrm`, and propagated into `gradT = gradr`, `Y_face =
/// gradT - grada`, and `conv_vel = 3 * D / Lambda`.
///
/// # Errors
/// Returns [`TurbError`] only if configuration parameters are invalid.
pub fn set_thermohaline<T: TurbScalar>(
    params: &ThermohalineParams,
    inp: &ThermohalineInputs<T>,
) -> TurbResult<TurbOutput<T>> {
    let d_thrm = get_d_thermohaline(
        params.option,
        inp.grada.val(),
        inp.gradr.val(),
        inp.t.val(),
        inp.opacity.val(),
        inp.rho.val(),
        inp.cp.val(),
        params.gradl_composition_term,
        params.species,
        params.xh1,
        params.thermohaline_coeff,
    );
    let d = T::from_f64(d_thrm);
    Ok(TurbOutput {
        gamma: T::from_f64(0.0),
        grad_t: inp.gradr,
        y_face: inp.gradr - inp.grada,
        conv_vel: T::from_f64(3.0) * d / inp.lambda,
        d,
        mixing_type: MixingType::ThermohalineMixing,
    })
}

/// Computes thermal diffusivity `kt`, molecular diffusivity `kmu`, and total
/// viscosity `vis` (`get_diff_coeffs` in `thermohaline.f90`, Proffitt & Michaud
/// 1993).
fn get_diff_coeffs(
    k_therm: f64,
    cp: f64,
    rho: f64,
    t: f64,
    opacity: f64,
    species: DrivingSpecies,
    xh1: f64,
) -> (f64, f64, f64) {
    let kt = k_therm / (cp * rho);
    let qe4 = pow4(QE);
    let loglambdah_base = -19.26 - 0.5 * log(rho) + 1.5 * log(t)
        - 0.5 * log(1.0 + 0.5 * (1.0 + xh1));
    let nu_rad = 4.0 * CRAD * pow4(t) / (15.0 * CLIGHT * opacity * pow2(rho));
    let nu_mol = 0.406 * AMU.sqrt() * pow(BOLTZM * t, 2.5)
        / (qe4 * loglambdah_base * rho);
    let vis = nu_mol + nu_rad;
    let b_coeff =
        (15.0 / 16.0) * (2.0 * AMU / (5.0 * PI)).sqrt() * pow(BOLTZM, 2.5)
            / qe4;

    let kmu = if species.chem_z > 2.0 {
        let loglambdacx = loglambdah_base - log(species.chem_z);
        let loglambdacy = loglambdah_base - log(2.0 * species.chem_z);
        let ccx = log(exp(1.2 * loglambdacx) + 1.0) / 1.2;
        let ccy = log(exp(1.2 * loglambdacy) + 1.0) / 1.2;
        let acx = species.chem_a / (1.0 + species.chem_a);
        let acy = 4.0 * species.chem_a / (4.0 + species.chem_a);
        let sqrt5 = 5.0_f64.sqrt();
        2.0 * b_coeff * pow(t, 2.5)
            / (sqrt5 * rho * species.chem_z * species.chem_z)
            / (xh1 * acx.sqrt() * ccx + (1.0 - xh1) * acy.sqrt() * ccy)
    } else {
        let loglambdah = loglambdah_base - log(2.0);
        let ccy = log(exp(1.2 * loglambdah) + 1.0) / 1.2;
        (b_coeff * pow(t, 2.5) / (rho * ccy)) * (3.0 + xh1)
            / ((1.0 + xh1) * (3.0 + 5.0 * xh1) * (0.7 + 0.3 * xh1))
    };
    (kt, kmu, vis)
}

/// Evaluates salt-finger Nusselt number $\mathrm{Nu}_\mu$ (Brown et al. 2013,
/// eq. 33).
///
/// Starts from [`analytical_estimate_th`] and refines $(l_{\max},
/// \lambda_{\max})$ via [`nr_solve`]. If the initial solve fails or returns
/// $\lambda_{\max} < 0$, retries up to 200 times with perturbed $(\mathrm{Pr},
/// R_0)$ before relaxing back to the target parameters.
fn numu(r0: f64, r_th: f64, prandtl: f64, diffratio: f64) -> f64 {
    let (maxl_est, lambda_est) =
        analytical_estimate_th(r_th, prandtl, diffratio);
    let mut vars = [maxl_est, lambda_est];
    let mut ok = nr_solve(&mut vars, prandtl, diffratio, r0);

    let mut iter = 1_usize;
    while iter <= 200 && (vars[1] < 0.0 || !ok) {
        vars = [maxl_est, lambda_est];
        let scale = 1.0 + (iter as f64) * 1.0e-2;
        if nr_solve(&mut vars, prandtl * scale, diffratio, r0 / scale) {
            ok = nr_solve(&mut vars, prandtl, diffratio, r0);
        }
        iter += 1;
    }

    let (maxl2, lambdamax) = if vars[1] < 0.0 || !ok {
        (maxl_est * maxl_est, lambda_est)
    } else {
        (vars[0] * vars[0], vars[1])
    };
    1.0 + 49.0 * lambdamax * lambdamax
        / (diffratio * maxl2 * (lambdamax + diffratio * maxl2))
}

/// Analytical starting guess $(l_{\max}, \lambda_{\max})$ from Brown et al.
/// (2013) Appendix B (`analytical_estimate_th` in `thermohaline.f90`).
fn analytical_estimate_th(
    r_th: f64,
    prandtl: f64,
    diffratio: f64,
) -> (f64, f64) {
    let phi = diffratio / prandtl;
    let (mut maxl, mut lambdamax) = if r_th < 0.5 && r_th <= prandtl {
        let l = ((1.0 / (1.0 + phi)).sqrt()
            - prandtl.sqrt() * (1.0 + phi / pow2(1.0 + phi)))
        .sqrt();
        let lam = prandtl.sqrt() - prandtl * (1.0 + phi).sqrt();
        (l, lam)
    } else {
        let l = if r_th < 0.5 {
            pow(
                (1.0 / (1.0 + phi))
                    - 2.0 * (r_th * phi).sqrt() / pow(1.0 + phi, 2.5),
                0.25,
            )
        } else {
            let omr = 1.0 - r_th;
            pow(ONE_THIRD * omr + omr * omr * (5.0 - 4.0 * phi) / 27.0, 0.25)
        };
        let l4 = pow4(l);
        let lam = 2.0 * prandtl * phi * l4 * l * l / (1.0 - (1.0 + phi) * l4);
        (l, lam)
    };
    if lambdamax < 0.0 {
        maxl = 0.5;
        lambdamax = 0.5;
    }
    (maxl, lambdamax)
}

/// 2D Newton-Raphson solver for Brown et al. (2013) eqs. 19-20 (`NR` in
/// `thermohaline.f90`).
///
/// Iterates up to 20 steps to Euclidean step norm $\le 10^{-13}$, aborting and
/// restoring the initial guess if the step norm grows after iteration 5.
fn nr_solve(xrk: &mut [f64; 2], prandtl: f64, diffratio: f64, r0: f64) -> bool {
    const ACY: f64 = 1.0e-13;
    const NITER: usize = 20;
    let sav = *xrk;
    let mut err = 1.0e99;
    let mut errold = 1.0e99;
    let mut iter = 0_usize;

    while err > ACY && iter < NITER {
        let (f, j) = thermohaline_rhs(*xrk, prandtl, diffratio, r0);
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        if det == 0.0 || !det.is_finite() {
            return false;
        }
        let dx0 = (j[1][1] * f[0] - j[0][1] * f[1]) / det;
        let dx1 = (-j[1][0] * f[0] + j[0][0] * f[1]) / det;
        iter += 1;
        err = (dx0 * dx0 + dx1 * dx1).sqrt();
        if iter > 5 && err > errold {
            *xrk = sav;
            return false;
        }
        xrk[0] -= dx0;
        xrk[1] -= dx1;
        errold = err;
    }
    err <= ACY
}

/// Evaluates residuals `f(2)` and $2 \times 2$ Jacobian `j(2, 2)` for Brown et
/// al. (2013) eqs. 19 and 20 (`thermohaline_rhs` in `thermohaline.f90`).
///
/// Here `myx[0]` is the horizontal finger wavenumber $l$ and `myx[1]` is the
/// fastest-growing mode growth rate $\lambda$.
fn thermohaline_rhs(
    myx: [f64; 2],
    prandtl: f64,
    diffratio: f64,
    r0: f64,
) -> ([f64; 2], [[f64; 2]; 2]) {
    let b_2 = 1.0 + prandtl + diffratio;
    let myx1_2 = myx[0] * myx[0];
    let myx1_3 = myx1_2 * myx[0];
    let myx1_4 = myx1_3 * myx[0];
    let a_2 = myx1_2 * b_2;
    let myterm = diffratio * prandtl + prandtl + diffratio;
    let b_1 = 2.0 * myx1_2 * myterm;
    let a_1 = myx1_4 * myterm + prandtl * (1.0 - (1.0 / r0));
    let b_0 =
        3.0 * myx1_4 * diffratio * prandtl + prandtl * (diffratio - (1.0 / r0));
    let a_0 = myx1_4 * myx1_2 * diffratio * prandtl
        + myx1_2 * prandtl * (diffratio - (1.0 / r0));

    let f0 = ((myx[1] + a_2) * myx[1] + a_1) * myx[1] + a_0;
    let f1 = b_2 * myx[1] * myx[1] + b_1 * myx[1] + b_0;

    let j00 = 2.0 * myx[0] * b_2 * myx[1] * myx[1]
        + 4.0 * myx1_3 * myterm * myx[1]
        + 6.0 * myx1_4 * myx[0] * diffratio * prandtl
        + 2.0 * myx[0] * prandtl * (diffratio - (1.0 / r0));
    let j01 = 3.0 * myx[1] * myx[1] + 2.0 * a_2 * myx[1] + a_1;
    let j10 =
        4.0 * myx[0] * myterm * myx[1] + 12.0 * myx1_3 * diffratio * prandtl;
    let j11 = 2.0 * b_2 * myx[1] + b_1;

    ([f0, f1], [[j00, j01], [j10, j11]])
}
