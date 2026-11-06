// SPDX-License-Identifier: LGPL-3.0-or-later
//! Scalar trait, physical constants, error types, and enumerations for
//! `flexmesa-colors` (`colors/public/colors_def.f90` and
//! `colors/defaults/colors.defaults`).

use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Sub};

use flexmesa_autodiff::DualOrder1;

/// Trait abstracting over primal IEEE-754 scalars (`f64`) and first-order
/// forward-mode dual numbers ([`DualOrder1<N>`]) for `colors` kernels.
pub trait RealScalar:
    Copy
    + Clone
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    /// Additive identity $0$.
    #[must_use]
    fn zero() -> Self;

    /// Multiplicative identity $1$.
    #[must_use]
    fn one() -> Self;

    /// Embeds a constant `f64` scalar with zero derivatives.
    #[must_use]
    fn from_f64(v: f64) -> Self;

    /// Extracts the primal `f64` value.
    #[must_use]
    fn val(self) -> f64;

    /// Base-10 logarithm $\log_{10}(x)$.
    #[must_use]
    fn log10(self) -> Self;
}

impl RealScalar for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn val(self) -> f64 {
        self
    }
    #[inline]
    fn log10(self) -> Self {
        flexmesa_math::log10(self)
    }
}

impl<const N: usize> RealScalar for DualOrder1<N> {
    #[inline]
    fn zero() -> Self {
        Self::constant(0.0)
    }
    #[inline]
    fn one() -> Self {
        Self::constant(1.0)
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(v)
    }
    #[inline]
    fn val(self) -> f64 {
        self.val
    }
    #[inline]
    fn log10(self) -> Self {
        Self::log10(self)
    }
}

/// Default reference distance in $\mathrm{cm}$ ($10\,\mathrm{pc}$) for
/// absolute magnitudes (`colors/defaults/colors.defaults`).
pub const D_10PC_CM: f64 = 3.0857e19;

/// Default solar metal-to-hydrogen reference ratio $(Z/X)_\odot$ for the GS98
/// mixture used by `Kurucz2003all` (`colors/defaults/colors.defaults`).
pub const Z_OVER_X_REF_GS98: f64 = 2.30057173972e-2;

/// Minimum stellar model count for parallel `rayon` zone evaluation.
pub const PARALLEL_ZONE_THRESHOLD: usize = 16;

/// Error type for `flexmesa-colors` routines.
#[derive(Debug, Clone, PartialEq)]
pub enum ColorsError {
    /// Input arrays have mismatched or insufficient lengths.
    InvalidArrayLength {
        /// Context string describing the failed routine.
        context: &'static str,
        /// Provided array length.
        got: usize,
        /// Minimum or expected array length.
        expected: usize,
    },
    /// Integrated bolometric or synthetic flux is non-positive.
    NonPositiveFlux {
        /// Context string describing the failed routine.
        context: &'static str,
        /// Evaluated flux value.
        flux: f64,
    },
    /// Unknown photometric magnitude system name.
    UnknownMagSystem(String),
    /// Unknown SED interpolation method name.
    UnknownInterpMethod(String),
    /// Malformed binary SED cube data.
    BinaryParseError(&'static str),
}

impl core::fmt::Display for ColorsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidArrayLength {
                context,
                got,
                expected,
            } => write!(
                f,
                "{context}: invalid array length \
                 (got {got}, expected {expected})"
            ),
            Self::NonPositiveFlux { context, flux } => {
                write!(f, "{context}: non-positive flux ({flux:e})")
            }
            Self::UnknownMagSystem(s) => {
                write!(f, "unknown magnitude system: {s}")
            }
            Self::UnknownInterpMethod(s) => {
                write!(f, "unknown SED interpolation method: {s}")
            }
            Self::BinaryParseError(msg) => {
                write!(f, "binary SED cube parse error: {msg}")
            }
        }
    }
}

impl std::error::Error for ColorsError {}

/// Photometric zero-point system (`colors/defaults/colors.defaults`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagSystem {
    /// Vega magnitude system ($m = 0$ for $\alpha\,\mathrm{Lyr}$).
    Vega,
    /// AB magnitude system ($f_\nu = 3631\,\mathrm{Jy}$).
    Ab,
    /// ST magnitude system
    /// ($f_\lambda = 3.63 \times 10^{-9}\,$
    /// $\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$).
    St,
}

impl MagSystem {
    /// Parses a magnitude system identifier case-insensitively.
    ///
    /// # Errors
    /// Returns [`ColorsError::UnknownMagSystem`] if `name` is not `"Vega"`,
    /// `"AB"`, or `"ST"`.
    pub fn parse(name: &str) -> Result<Self, ColorsError> {
        match name.trim().to_ascii_uppercase().as_str() {
            "VEGA" => Ok(Self::Vega),
            "AB" => Ok(Self::Ab),
            "ST" => Ok(Self::St),
            _ => Err(ColorsError::UnknownMagSystem(name.to_string())),
        }
    }
}

/// 3D stellar atmosphere SED interpolation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SedInterpMethod {
    /// Cubic Hermite tensor interpolation with finite-difference slopes.
    Hermite,
    /// Log-fallback trilinear interpolation over the $2 \times 2 \times 2$
    /// cell.
    Linear,
    /// 4-nearest-neighbor inverse-distance-weighted interpolation.
    Knn,
}

impl SedInterpMethod {
    /// Parses an interpolation method identifier case-insensitively.
    ///
    /// # Errors
    /// Returns [`ColorsError::UnknownInterpMethod`] if `name` is not
    /// `"Hermite"`, `"Linear"`, or `"KNN"`.
    pub fn parse(name: &str) -> Result<Self, ColorsError> {
        match name.trim().to_ascii_uppercase().as_str() {
            "HERMITE" => Ok(Self::Hermite),
            "LINEAR" => Ok(Self::Linear),
            "KNN" => Ok(Self::Knn),
            _ => Err(ColorsError::UnknownInterpMethod(name.to_string())),
        }
    }
}
