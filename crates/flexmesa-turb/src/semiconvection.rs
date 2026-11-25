// SPDX-License-Identifier: GPL-3.0-or-later
//! Semiconvective mixing (`set_semiconvection`, `semiconvection.f90`).
//!
//! Implements Langer, El Eid, & Fricke (1983, 1985) semiconvective mixing
//! diffusivity and temperature-gradient prescriptions:
//! - [`SemiconvectionOption::Langer85GradTIsGradr`]
//!   (`"Langer_85 mixing; gradT = gradr"`): Sets $\nabla_T = \nabla_r$
//!   while using the Langer (1985) chemical diffusivity
//!   $$D_{\mathrm{sc}} = \frac{\alpha_{\mathrm{sc}} K_r}{6 C_P \rho}
//!   \frac{\nabla_r - \nabla_{\mathrm{ad}}}{\nabla_L - \nabla_r}.$$
//! - [`SemiconvectionOption::Langer85`] (`"Langer_85"`): Solves the coupled
//!   radiative + semiconvective luminosity quadratic equation for $\nabla_T$.

use std::str::FromStr;

use flexmesa_const::{MixingType, CLIGHT, CRAD, PI};

use crate::{TurbError, TurbOutput, TurbResult, TurbScalar};

/// Semiconvective mixing prescription (`semiconvection_option`).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum SemiconvectionOption {
    /// Langer (1985) diffusivity with $\nabla_T = \nabla_{\mathrm{rad}}$
    /// (`"Langer_85 mixing; gradT = gradr"`).
    Langer85GradTIsGradr,
    /// Full Langer (1985) quadratic solution for $\nabla_T$ (`"Langer_85"`).
    Langer85,
}

impl FromStr for SemiconvectionOption {
    type Err = TurbError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "Langer_85 mixing; gradT = gradr" => Ok(Self::Langer85GradTIsGradr),
            "Langer_85" => Ok(Self::Langer85),
            other => {
                Err(TurbError::UnknownSemiconvectionOption(other.to_string()))
            }
        }
    }
}

/// Non-differentiable configuration parameters for [`set_semiconvection`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemiconvectionParams {
    /// Semiconvective prescription ([`SemiconvectionOption`]).
    pub option: SemiconvectionOption,
    /// Semiconvective efficiency parameter $\alpha_{\mathrm{sc}}$
    /// (dimensionless).
    pub alpha_semiconvection: f64,
    /// Gravitational constant $G$ in $\mathrm{cm^3\,g^{-1}\,s^{-2}}$.
    pub cgrav: f64,
    /// Composition gradient contribution $B = (\phi/\delta)\nabla_\mu$ to
    /// $\nabla_L = \nabla_{\mathrm{ad}} + B$.
    pub gradl_composition_term: f64,
    /// Enclosed mass $m$ at the cell face in $\mathrm{g}$.
    pub m: f64,
}

/// Differentiable thermodynamic and structure inputs for
/// [`set_semiconvection`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemiconvectionInputs<T> {
    /// Total luminosity $L$ across the face in $\mathrm{erg\,s^{-1}}$.
    pub l: T,
    /// Mixing length $\Lambda$ in $\mathrm{cm}$.
    pub lambda: T,
    /// Temperature $T$ in $\mathrm{K}$.
    pub t: T,
    /// Total pressure $P$ in $\mathrm{dyn\,cm^{-2}}$.
    pub p: T,
    /// Radiation pressure $P_{\mathrm{rad}}$ in $\mathrm{dyn\,cm^{-2}}$.
    pub pr: T,
    /// Gas-to-total pressure ratio $\beta = P_{\mathrm{gas}} / P$.
    pub beta: T,
    /// Rosseland mean opacity $\kappa$ in $\mathrm{cm^2\,g^{-1}}$.
    pub opacity: T,
    /// Mass density $\rho$ in $\mathrm{g\,cm^{-3}}$.
    pub rho: T,
    /// Specific heat at constant pressure $C_P$ in
    /// $\mathrm{erg\,g^{-1}\,K^{-1}}$.
    pub cp: T,
    /// Radiative temperature gradient $\nabla_{\mathrm{rad}}$.
    pub gradr: T,
    /// Adiabatic temperature gradient $\nabla_{\mathrm{ad}}$.
    pub grada: T,
    /// Ledoux temperature gradient $\nabla_L$.
    pub gradl: T,
}

/// Evaluates Langer (1983, 1985) semiconvective mixing (`calc_semiconvection`).
///
/// Supports both scalar `f64` and `DualOrder1<N>` inputs via [`TurbScalar`].
///
/// # Errors
/// Returns [`TurbError`] only if configuration parameters are invalid.
pub fn set_semiconvection<T: TurbScalar>(
    params: &SemiconvectionParams,
    inp: &SemiconvectionInputs<T>,
) -> TurbResult<TurbOutput<T>> {
    let zero = T::from_f64(0.0);
    let rad_cond = (T::from_f64(4.0 / 3.0 * CRAD * CLIGHT) * inp.t.pow3())
        .div_ad(inp.opacity * inp.rho);
    let d = ((T::from_f64(params.alpha_semiconvection) * rad_cond)
        .div_ad(T::from_f64(6.0) * inp.cp * inp.rho)
        * (inp.gradr - inp.grada))
        .div_ad(inp.gradl - inp.gradr);

    // `if (D%val <= 0) return` (`turb/private/semiconvection.f90:81`):
    // `D` is `intent(out)`, so the caller receives `d` even on early return,
    // and `turb_support.f90:422` treats `NaN` or `D < remove_small_D_limit`
    // identically by falling back to `no_mixing`.
    if d.val().is_nan() || d.val() <= 0.0 {
        return Ok(TurbOutput {
            gamma: zero,
            grad_t: zero,
            y_face: zero,
            conv_vel: zero,
            d,
            mixing_type: MixingType::NoMixing,
        });
    }

    let conv_vel = (T::from_f64(3.0) * d).div_ad(inp.lambda);
    match params.option {
        SemiconvectionOption::Langer85GradTIsGradr => {
            let grad_t = inp.gradr;
            let y_face = grad_t - inp.grada;
            Ok(TurbOutput {
                gamma: zero,
                grad_t,
                y_face,
                conv_vel,
                d,
                mixing_type: MixingType::SemiconvectiveMixing,
            })
        }
        SemiconvectionOption::Langer85 => {
            let alpha = params.alpha_semiconvection.min(1.0);
            let alpha_t = T::from_f64(alpha);
            let bc = T::from_f64(32.0)
                - T::from_f64(24.0) * inp.beta
                - inp.beta * inp.beta;
            let lg =
                (T::from_f64(16.0 * PI * CLIGHT * params.m * params.cgrav)
                    * inp.pr)
                    .div_ad(inp.p * inp.opacity);
            let a0 = T::from_f64(alpha * params.gradl_composition_term) * lg;
            let a1 = T::from_f64(-2.0) * bc * inp.l;
            let a2 = T::from_f64(2.0 * alpha) * bc * inp.grada * lg;
            let a3 = T::from_f64(-2.0) * bc * inp.gradl * lg;
            let a4 = T::from_f64(32.0) * a0;
            let a5 = T::from_f64(-36.0) * inp.beta * a0;
            let a6 = T::from_f64(9.0) * inp.beta * inp.beta * a0;
            let a = a1 + a2 + a3 + a4 + a5 + a6;
            let b1 = T::from_f64(32.0) - T::from_f64(36.0) * inp.beta
                + T::from_f64(9.0) * inp.beta * inp.beta;
            let b2 = b1 * a0;
            let b3 = T::from_f64(-2.0) * inp.gradl * inp.l
                + alpha_t * inp.grada * inp.grada * lg;
            let b4 = (-(alpha_t * inp.grada) + inp.gradl) * lg;
            let b5 = -b2 + T::from_f64(2.0) * bc * (inp.l + b4);
            let b6 = b2 * inp.grada + bc * b3;
            let b7 = T::from_f64(-4.0 * (alpha - 2.0)) * bc * lg * b6;
            let b = b7 + b5 * b5;
            let div = T::from_f64(2.0 * (alpha - 2.0)) * bc * lg;
            let bsq = b.sqrt();
            let grad_t = (a + bsq).div_ad(div);
            let y_face = grad_t - inp.grada;
            Ok(TurbOutput {
                gamma: zero,
                grad_t,
                y_face,
                conv_vel,
                d,
                mixing_type: MixingType::SemiconvectiveMixing,
            })
        }
    }
}
