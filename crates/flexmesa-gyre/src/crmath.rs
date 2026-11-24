// SPDX-License-Identifier: GPL-3.0-only
//! CRMATH 1.2 composites that GYRE 8.1 and the SDK LAPACK call, built only
//! from `flexmesa-math` primitives.
//!
//! Sources: `crmath_r_dp.f90` and `crmath_c_dp.f90` of CRMATH 1.2 (git commit
//! `3820ff0532fa88137086d56c383cb23a5cef14b2`, the version vendored by the MESA
//! SDK `26.6.1`). Each routine follows the Fortran operation order; the
//! complex products use the gfortran expansions of [`crate::fcomplex`].
//! The real elementary functions route through `flexmesa-math`, whose
//! strict mode calls the same `crlibm` entry points as CRMATH (`log_rz`,
//! `exp_rd`, `cos_rz`, `sin_rz`, `atan_rz`, `acos_rd`).

use crate::fcomplex::Cplx;
use crate::GyreNumError;

/// CRMATH `HALFPI_DP` (`crmath.f90:30`, `1.5707963267948966192_DP`).
pub const HALFPI_DP: f64 = f64::from_bits(0x3FF9_21FB_5444_2D18);
/// CRMATH `PI_DP` (`crmath.f90:28`, `3.1415926535897932385_DP`).
pub const PI_DP: f64 = f64::from_bits(0x4009_21FB_5444_2D18);

/// CRMATH `hypot_r_r_dp_` (`crmath_r_dp.f90:187-202`).
#[must_use]
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    flexmesa_math::hypot(x, y)
}

/// CRMATH `atan2_r_r_dp_` (`crmath_r_dp.f90:296-311`).
///
/// This is CRMATH's own octant formula over `atan_rz`, which differs from
/// the Fortran intrinsic `atan2` that `flexmesa_math::atan2` reproduces.
///
/// # Errors
///
/// [`GyreNumError::Atan2Undefined`] when `x = y = 0` (or either is NaN):
/// upstream falls through every branch and returns an unset result
/// (`DECISIONS #43`).
#[allow(clippy::float_cmp)]
pub fn atan2(y: f64, x: f64) -> Result<f64, GyreNumError> {
    if x > 0.0 {
        Ok(flexmesa_math::atan(y / x))
    } else if y > 0.0 {
        Ok(HALFPI_DP - flexmesa_math::atan(x / y))
    } else if y < 0.0 {
        Ok(-HALFPI_DP - flexmesa_math::atan(x / y))
    } else if x < 0.0 {
        Ok(flexmesa_math::atan(y / x) + PI_DP)
    } else {
        Err(GyreNumError::Atan2Undefined { y, x })
    }
}

/// CRMATH `abs_c_dp_` (`crmath_c_dp.f90:111-117`): `hypot(re, im)`.
#[must_use]
#[inline]
pub fn abs_c(z: Cplx) -> f64 {
    hypot(z.re, z.im)
}

/// CRMATH `sqrt_c_dp_` (`crmath_c_dp.f90:65-107`).
#[must_use]
#[allow(clippy::float_cmp)]
pub fn sqrt_c(z: Cplx) -> Cplx {
    if z.im == 0.0 {
        if z.re < 0.0 {
            Cplx::new(0.0, (-z.re).sqrt().copysign(z.im))
        } else {
            Cplx::new(z.re.sqrt(), 0.0_f64.copysign(z.im))
        }
    } else if z.re == 0.0 {
        let r = (0.5 * z.im.abs()).sqrt();
        Cplx::new(r, r.copysign(z.im))
    } else {
        let d = hypot(z.re, z.im);
        let (r, s) = if z.re > 0.0 {
            let r = (0.5 * d + 0.5 * z.re).sqrt();
            let s = 0.5 * z.im / r;
            (r, s)
        } else {
            let s = (0.5 * d - 0.5 * z.re).sqrt();
            let r = (0.5 * z.im / s).abs();
            (r, s)
        };
        Cplx::new(r, s.copysign(z.im))
    }
}

/// CRMATH `exp_c_dp_` (`crmath_c_dp.f90:45-51`):
/// `exp(re) * CMPLX(cos(im), sin(im))`.
#[must_use]
#[inline]
pub fn exp_c(z: Cplx) -> Cplx {
    let e = flexmesa_math::exp(z.re);
    Cplx::real_mul(
        e,
        Cplx::new(flexmesa_math::cos(z.im), flexmesa_math::sin(z.im)),
    )
}

/// CRMATH `log_c_dp_` (`crmath_c_dp.f90:21-27`):
/// `CMPLX(log(abs(z)), atan2(im, re))`.
///
/// # Errors
///
/// [`GyreNumError::Atan2Undefined`] for `z = 0`.
pub fn log_c(z: Cplx) -> Result<Cplx, GyreNumError> {
    let lr = flexmesa_math::log(abs_c(z));
    Ok(Cplx::new(lr, atan2(z.im, z.re)?))
}
