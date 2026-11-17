// SPDX-License-Identifier: LGPL-3.0-or-later
//! Trigonometric, half-turn (`pi`-scaled), inverse trigonometric, and
//! hyperbolic functions for `flexmesa-math` (`#165`).

use flexmesa_const::PI;

use crate::fast_ops;
use crate::mode::is_strict;

/// `PI_DP` exact IEEE-754 bit pattern (`0x4009_21fb_5444_2d18`).
const PI_HI: f64 = f64::from_bits(0x4009_21fb_5444_2d18);
/// Low 53-bit double-double tail of $\pi$ (`0x3ca1_a626_3314_5c07`).
const PI_LO: f64 = f64::from_bits(0x3ca1_a626_3314_5c07);
/// `HALFPI_DP` exact IEEE-754 bit pattern (`0x3ff9_21fb_5444_2d18`).
const HALFPI_HI: f64 = f64::from_bits(0x3ff9_21fb_5444_2d18);
/// Low 53-bit double-double tail of $\pi / 2$ (`0x3c91_a626_3314_5c07`).
const HALFPI_LO: f64 = f64::from_bits(0x3c91_a626_3314_5c07);
/// Nearest-even IEEE-754 $\pi / 4$ (`0x3fe9_21fb_5444_2d18`).
const QUARTER_PI: f64 = f64::from_bits(0x3fe9_21fb_5444_2d18);
/// Nearest-even IEEE-754 $3\pi / 4$ (`0x4002_d97c_7f33_21d2`).
const THREE_QUARTER_PI: f64 = f64::from_bits(0x4002_d97c_7f33_21d2);

#[inline]
fn two_sum(lhs: f64, rhs: f64) -> (f64, f64) {
    let sum = lhs + rhs;
    let virt = sum - lhs;
    let err = (lhs - (sum - virt)) + (rhs - virt);
    (sum, err)
}

/// Compute $\cos(x)$ for $x$ in radians (`math/public/math_lib_crmath.f90:83`,
/// `crmath_r_dp.f90:206`).
#[must_use]
#[inline]
pub fn cos(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::cos_rz(x)
    } else {
        fast_ops::cos(x)
    }
}

/// Compute $\sin(x)$ for $x$ in radians (`math/public/math_lib_crmath.f90:84`,
/// `crmath_r_dp.f90:216`).
#[must_use]
#[inline]
pub fn sin(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::sin_rz(x)
    } else {
        fast_ops::sin(x)
    }
}

/// Compute $(\sin(x), \cos(x))$ in radians.
#[must_use]
#[inline]
pub fn sin_cos(x: f64) -> (f64, f64) {
    if is_strict() {
        (
            flexmesa_crlibm_sys::sin_rz(x),
            flexmesa_crlibm_sys::cos_rz(x),
        )
    } else {
        fast_ops::sin_cos(x)
    }
}

/// Compute $\tan(x)$ for $x$ in radians (`math/public/math_lib_crmath.f90:85`,
/// `crmath_r_dp.f90:226`).
#[must_use]
#[inline]
pub fn tan(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::tan_rz(x)
    } else {
        fast_ops::tan(x)
    }
}

/// Compute $\cos(\pi x)$ for half-turn argument $x$
/// (`math/public/math_lib_crmath.f90:86`, `crmath_r_dp.f90:236`).
#[must_use]
#[inline]
pub fn cospi(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::cospi_rz(x)
    } else {
        fast_ops::cos(x * PI)
    }
}

/// Compute $\sin(\pi x)$ for half-turn argument $x$
/// (`math/public/math_lib_crmath.f90:87`, `crmath_r_dp.f90:246`).
#[must_use]
#[inline]
pub fn sinpi(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::sinpi_rz(x)
    } else {
        fast_ops::sin(x * PI)
    }
}

/// Compute $\tan(\pi x)$ for half-turn argument $x$
/// (`math/public/math_lib_crmath.f90:88`, `crmath_r_dp.f90:256`).
#[must_use]
#[inline]
pub fn tanpi(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::tanpi_rz(x)
    } else {
        fast_ops::tan(x * PI)
    }
}

/// Compute $\arccos(x)$ in radians (`math/public/math_lib_crmath.f90:89`,
/// `crmath_r_dp.f90:266`).
#[must_use]
#[inline]
pub fn acos(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::acos_rd(x)
    } else {
        fast_ops::acos(x)
    }
}

/// Compute $\arcsin(x)$ in radians (`math/public/math_lib_crmath.f90:90`,
/// `crmath_r_dp.f90:276`).
#[must_use]
#[inline]
pub fn asin(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::asin_rz(x)
    } else {
        fast_ops::asin(x)
    }
}

/// Compute $\arctan(x)$ in radians (`math/public/math_lib_crmath.f90:91`,
/// `crmath_r_dp.f90:286`).
#[must_use]
#[inline]
pub fn atan(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::atan_rz(x)
    } else {
        fast_ops::atan(x)
    }
}

/// Compute the four-quadrant inverse tangent $\operatorname{atan2}(y, x)$.
///
/// MESA `26.4.1` never re-exports `CRMATH`'s `atan2` from
/// `math/public/math_lib_crmath.f90:65-100`, so upstream `atan2` and `datan2`
/// calls bind to Fortran's platform intrinsic (`#196`, `#198`). In
/// [`crate::NumericsMode::Strict`], this routine evaluates hardware-independent
/// octant reduction over `crlibm::atan_rn` with 106-bit $\pi$ and $\pi/2$
/// constants (`93.42%`/`93.60%` bit-exact match and `<= 1` ULP across `20,242`
/// MESA `26.4.1` `atan2`/`datan2` vectors, vs `69.27%`/`69.01%` and `<= 2` ULP
/// for `atan_rz`). In [`crate::NumericsMode::Fast`], it calls `f64::atan2`.
#[must_use]
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    if !is_strict() {
        return fast_ops::atan2(y, x);
    }
    if y.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    let y_neg = y.is_sign_negative();
    let x_neg = x.is_sign_negative();
    if y == 0.0_f64 {
        return match (x_neg, y_neg) {
            (false, true) => -0.0_f64,
            (false, false) => 0.0_f64,
            (true, true) => -PI_HI,
            (true, false) => PI_HI,
        };
    }
    if x == 0.0_f64 {
        return if y_neg { -HALFPI_HI } else { HALFPI_HI };
    }
    if y.is_infinite() && x.is_infinite() {
        let mag = if x_neg { THREE_QUARTER_PI } else { QUARTER_PI };
        return if y_neg { -mag } else { mag };
    }
    if x.is_infinite() {
        return match (x_neg, y_neg) {
            (false, true) => -0.0_f64,
            (false, false) => 0.0_f64,
            (true, true) => -PI_HI,
            (true, false) => PI_HI,
        };
    }
    if y.is_infinite() {
        return if y_neg { -HALFPI_HI } else { HALFPI_HI };
    }
    let ay = y.abs();
    let ax = x.abs();
    let mag_ang = if ax >= ay {
        let ang = flexmesa_crlibm_sys::atan_rn(ay / ax);
        if x_neg {
            let (sum, err) = two_sum(PI_HI, -ang);
            sum + (err + PI_LO)
        } else {
            ang
        }
    } else {
        let ang = flexmesa_crlibm_sys::atan_rn(ax / ay);
        let (sum, err) = if x_neg {
            two_sum(HALFPI_HI, ang)
        } else {
            two_sum(HALFPI_HI, -ang)
        };
        sum + (err + HALFPI_LO)
    };
    if y_neg {
        -mag_ang
    } else {
        mag_ang
    }
}

/// Compute $\arccos(x) / \pi$ in half-turns
/// (`math/public/math_lib_crmath.f90:92`, `crmath_r_dp.f90:315`).
#[must_use]
#[inline]
pub fn acospi(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::acospi_rd(x)
    } else {
        fast_ops::acos(x) / PI
    }
}

/// Compute $\arcsin(x) / \pi$ in half-turns
/// (`math/public/math_lib_crmath.f90:93`, `crmath_r_dp.f90:325`).
#[must_use]
#[inline]
pub fn asinpi(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::asinpi_rz(x)
    } else {
        fast_ops::asin(x) / PI
    }
}

/// Compute $\arctan(x) / \pi$ in half-turns
/// (`math/public/math_lib_crmath.f90:94`, `crmath_r_dp.f90:335`).
#[must_use]
#[inline]
pub fn atanpi(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::atanpi_rz(x)
    } else {
        fast_ops::atan(x) / PI
    }
}

/// Compute $\cosh(x)$ (`math/public/math_lib_crmath.f90:95`,
/// `crmath_r_dp.f90:345`).
#[must_use]
#[inline]
pub fn cosh(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::cosh_rz(x)
    } else {
        fast_ops::cosh(x)
    }
}

/// Compute $\sinh(x)$ (`math/public/math_lib_crmath.f90:96`,
/// `crmath_r_dp.f90:355`).
#[must_use]
#[inline]
pub fn sinh(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::sinh_rz(x)
    } else {
        fast_ops::sinh(x)
    }
}

/// Compute $\tanh(x)$ (`math/public/math_lib_crmath.f90:97`,
/// `crmath_r_dp.f90:365-371`).
///
/// Note: `CRMATH 1.2` evaluates `tanh(x) = sinh_rz(x) / cosh_rz(x)`. Because
/// `sinh_rz` and `cosh_rz` clamp finite overflow to `±f64::MAX`, finite
/// `|x| > 710.48` yields `±1.0`, whereas `x = ±Inf` evaluates `±Inf / +Inf = NaN`.
/// `Strict` mode reproduces this upstream behaviour verbatim (`#197`).
#[must_use]
#[inline]
pub fn tanh(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::sinh_rz(x) / flexmesa_crlibm_sys::cosh_rz(x)
    } else {
        fast_ops::tanh(x)
    }
}

/// Compute $\operatorname{arcosh}(x)$ (`math/public/math_lib_crmath.f90:98`,
/// `crmath_r_dp.f90:375-381`).
#[must_use]
#[inline]
pub fn acosh(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::log_rz(
            x + (x - 1.0_f64).sqrt() * (x + 1.0_f64).sqrt(),
        )
    } else {
        fast_ops::acosh(x)
    }
}

/// Compute $\operatorname{arsinh}(x)$ (`math/public/math_lib_crmath.f90:99`,
/// `crmath_r_dp.f90:385-391`).
#[must_use]
#[inline]
pub fn asinh(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::log_rz(x + (1.0_f64 + x * x).sqrt())
    } else {
        fast_ops::asinh(x)
    }
}

/// Compute $\operatorname{artanh}(x)$ (`math/public/math_lib_crmath.f90:100`,
/// `crmath_r_dp.f90:395-401`).
#[must_use]
#[inline]
pub fn atanh(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::log_rz((1.0_f64 + x) / (1.0_f64 - x)) / 2.0_f64
    } else {
        fast_ops::atanh(x)
    }
}
