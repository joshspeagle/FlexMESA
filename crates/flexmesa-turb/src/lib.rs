// SPDX-License-Identifier: LGPL-3.0-or-later
//! Convective and diffusive turbulent mixing (`turb`) for `FlexMESA`.
//!
//! Ports upstream MESA's `turb` module (`turb/public/turb.f90` and
//! `turb/private/{mlt,tdc,tdc_support,semiconvection,thermohaline}.f90`) to
//! Safe Rust, supporting both IEEE-754 double precision (`f64`) and
//! const-generic first-order forward-mode dual numbers ([`DualOrder1<N>`]).
//!
//! # Physical Scope and Units
//! All physical quantities use CGS units (`cm`, `g`, `s`, `erg`, `K`, `dyn`):
//! - Temperature gradients ($\nabla_{\mathrm{rad}}, \nabla_{\mathrm{ad}},
//!   \nabla_L, \nabla_T$) and superadiabaticity $Y_{\mathrm{face}}$ are
//!   dimensionless logarithmic derivatives $d\ln T / d\ln P$.
//! - Convective speeds $v_{\mathrm{conv}}$ are in $\mathrm{cm\,s^{-1}}$.
//! - Chemical diffusion coefficients $D$ are in $\mathrm{cm^2\,s^{-1}}$.
//!
//! # Provided Submodules
//! - [`mlt`]: Local Mixing-Length Theory (`Cox`, `TDC`, `Henyey`, `ML1`, `ML2`,
//!   `Mihalas`) solving the cubic convective efficiency equation
//!   $a_0 \Gamma^3 + \Gamma^2 + \Gamma - a_0 B^3 = 0$.
//! - [`tdc`]: Kuhfuss (1986) / Smolec time-dependent convection (`set_tdc`)
//!   with bisection-bracketed Newton root-finding over $Z = \ln|Y|$ and
//!   Implicit Function Theorem derivative propagation.
//! - [`semiconvection`]: Langer, El Eid, & Fricke (1985) semiconvective
//!   diffusivity and quadratic radiative/semiconvective flux partitioning.
//! - [`thermohaline`]: Kippenhahn et al. (1980), Traxler et al. (2011), and
//!   Brown et al. (2013) salt-finger thermohaline mixing diffusivities.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::wrong_self_convention
)]

pub mod mlt;
pub mod semiconvection;
pub mod tdc;
pub(crate) mod tdc_search;
pub(crate) mod tdc_support;
pub mod thermohaline;

use std::error::Error;
use std::fmt::{self, Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

use flexmesa_autodiff::DualOrder1;
pub use flexmesa_const::MixingType;

pub use mlt::{set_mlt, MltInputs, MltOption, MltParams};
pub use semiconvection::{
    set_semiconvection, SemiconvectionInputs, SemiconvectionOption,
    SemiconvectionParams,
};
pub use tdc::{set_tdc, TdcInputs, TdcOutput, TdcParams};
pub use thermohaline::{
    get_d_thermohaline, set_thermohaline, DrivingSpecies, ThermohalineInputs,
    ThermohalineOption, ThermohalineParams,
};

/// Errors returned by [`flexmesa_turb`](crate) routines.
#[derive(Debug, Clone, PartialEq)]
pub enum TurbError {
    /// Unrecognized MLT prescription name.
    UnknownMltOption(String),

    /// Unrecognized semiconvection prescription name.
    UnknownSemiconvectionOption(String),

    /// Unrecognized thermohaline prescription name.
    UnknownThermohalineOption(String),

    /// Invalid input parameter (`dt <= 0`, `mixing_length_alpha == 0`, or
    /// `NaN`).
    InvalidParameter(&'static str),

    /// Non-convergence or invalid bracket in the TDC root solver.
    TdcConvergenceFailure(&'static str),
}

impl Display for TurbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownMltOption(s) => write!(f, "unknown MLT option: {s}"),
            Self::UnknownSemiconvectionOption(s) => {
                write!(f, "unknown semiconvection option: {s}")
            }
            Self::UnknownThermohalineOption(s) => {
                write!(f, "unknown thermohaline option: {s}")
            }
            Self::InvalidParameter(msg) => {
                write!(f, "invalid turb input parameter: {msg}")
            }
            Self::TdcConvergenceFailure(msg) => {
                write!(f, "TDC solver failed to converge: {msg}")
            }
        }
    }
}

impl Error for TurbError {}

/// Convenience result alias for [`flexmesa_turb`](crate).
pub type TurbResult<T> = Result<T, TurbError>;

/// Common output structure for convective and diffusive mixing prescriptions.
///
/// Matches the standard output tuple `(Gamma, gradT, Y_face, conv_vel, D,
/// mixing_type)` returned by `set_MLT`, `set_TDC`, `set_semiconvection`, and
/// `set_thermohaline` in `turb/public/turb.f90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurbOutput<T> {
    /// Convective efficiency parameter $\Gamma$ (dimensionless, `-1e99` in
    /// radiative zones for `set_mlt`, `0.0` for `semiconvection` and
    /// `thermohaline`).
    pub gamma: T,
    /// Actual temperature gradient $\nabla_T = d\ln T / d\ln P$
    /// (dimensionless).
    pub grad_t: T,
    /// Superadiabaticity $Y_{\mathrm{face}}$ ($\nabla_T - \nabla_L$ for MLT and
    /// TDC; $\nabla_T - \nabla_{\mathrm{ad}}$ for semiconvection and
    /// thermohaline).
    pub y_face: T,
    /// Convective or effective mixing speed $v_{\mathrm{conv}}$ in
    /// $\mathrm{cm\,s^{-1}}$.
    pub conv_vel: T,
    /// Chemical compositional diffusion coefficient $D$ in
    /// $\mathrm{cm^2\,s^{-1}}$.
    pub d: T,
    /// Active mixing classification ([`MixingType`]).
    pub mixing_type: MixingType,
}

/// Trait abstracting over primal IEEE-754 scalars (`f64`) and first-order
/// forward-mode dual numbers ([`DualOrder1<N>`]).
///
/// Enables a single generic implementation of `set_mlt`, `set_tdc`,
/// `set_semiconvection`, and `set_thermohaline` that evaluates either scalar
/// values or exact analytical partial derivatives without code duplication.
pub trait TurbScalar:
    Copy
    + Clone
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Embeds a constant `f64` scalar with zero derivatives.
    #[must_use]
    fn from_f64(v: f64) -> Self;

    /// Extracts the primal `f64` scalar value.
    #[must_use]
    fn val(self) -> f64;

    /// Replaces the primal `f64` scalar value while preserving derivatives.
    ///
    /// Matches upstream Fortran's `conv_vel%val = max_conv_vel` in `mlt.f90`.
    #[must_use]
    fn with_val(self, new_val: f64) -> Self;

    /// Evaluates $|x|$.
    #[must_use]
    fn abs(self) -> Self;

    /// Evaluates $\sqrt{x}$.
    #[must_use]
    fn sqrt(self) -> Self;

    /// Evaluates $\exp(x)$.
    #[must_use]
    fn exp(self) -> Self;

    /// Evaluates $\ln(x)$.
    #[must_use]
    fn log(self) -> Self;

    /// Evaluates $x^p$ for a constant `f64` exponent $p$.
    #[must_use]
    fn powf(self, p: f64) -> Self;

    /// Evaluates $x^2$.
    #[must_use]
    fn pow2(self) -> Self;

    /// Evaluates $x^3$.
    #[must_use]
    fn pow3(self) -> Self;

    /// Evaluates $x^4$.
    #[must_use]
    fn pow4(self) -> Self;

    /// Evaluates $\tanh(x)$.
    #[must_use]
    fn tanh(self) -> Self;

    /// Evaluates $\tan(x)$.
    #[must_use]
    fn tan(self) -> Self;

    /// Evaluates $\arctan(x)$.
    #[must_use]
    fn atan(self) -> Self;
}

impl TurbScalar for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }

    #[inline]
    fn val(self) -> f64 {
        self
    }

    #[inline]
    fn with_val(self, new_val: f64) -> Self {
        new_val
    }

    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    #[inline]
    fn exp(self) -> Self {
        flexmesa_math::exp(self)
    }

    #[inline]
    fn log(self) -> Self {
        flexmesa_math::log(self)
    }

    #[inline]
    fn powf(self, p: f64) -> Self {
        flexmesa_math::pow(self, p)
    }

    #[inline]
    fn pow2(self) -> Self {
        flexmesa_math::pow2(self)
    }

    #[inline]
    fn pow3(self) -> Self {
        flexmesa_math::pow3(self)
    }

    #[inline]
    fn pow4(self) -> Self {
        flexmesa_math::pow4(self)
    }

    #[inline]
    fn tanh(self) -> Self {
        flexmesa_math::tanh(self)
    }

    #[inline]
    fn tan(self) -> Self {
        flexmesa_math::tan(self)
    }

    #[inline]
    fn atan(self) -> Self {
        flexmesa_math::atan(self)
    }
}

impl<const N: usize> TurbScalar for DualOrder1<N> {
    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(v)
    }

    #[inline]
    fn val(self) -> f64 {
        self.val
    }

    #[inline]
    fn with_val(mut self, new_val: f64) -> Self {
        self.val = new_val;
        self
    }

    #[inline]
    fn abs(self) -> Self {
        Self::abs(self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }

    #[inline]
    fn exp(self) -> Self {
        Self::exp(self)
    }

    #[inline]
    fn log(self) -> Self {
        Self::log(self)
    }

    #[inline]
    fn powf(self, p: f64) -> Self {
        Self::powf(self, p)
    }

    #[inline]
    fn pow2(self) -> Self {
        Self::pow2(self)
    }

    #[inline]
    fn pow3(self) -> Self {
        Self::pow3(self)
    }

    #[inline]
    fn pow4(self) -> Self {
        Self::pow4(self)
    }

    #[inline]
    fn tanh(self) -> Self {
        Self::tanh(self)
    }

    #[inline]
    fn tan(self) -> Self {
        Self::tan(self)
    }

    #[inline]
    fn atan(self) -> Self {
        Self::atan(self)
    }
}
