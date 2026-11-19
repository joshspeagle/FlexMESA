// SPDX-License-Identifier: LGPL-3.0-or-later
//! Canonical `f64` mathematical intrinsics and numerics-mode dispatch (`math`,
//! `#165`).
//!
//! # Contract of the `Strict` and `Fast` Backends
//!
//! All `FlexMESA` physics crates call elementary and power functions exclusively
//! through this crate from a single Rust source (`DECISIONS #23`):
//!
//! - **[`NumericsMode::Strict`] (default)**: *Correctly rounded toward zero as
//!   `CRMATH` does it.* Specifically, `strict` mode binds the vendored INRIA
//!   `crlibm 1.0beta4` routines called by MESA `26.4.1`'s `CRMATH 1.2` module
//!   (`math/public/math_lib_crmath.f90`, `crmath_r_dp.f90`), using `*_rz`
//!   entry points everywhere except `exp`, `acos`, and `acospi` (which call
//!   `*_rd`), and evaluates every `CRMATH` and `math_lib` composite (`pow`,
//!   `powi`, `exp10`, `safe_log`, `safe_log10`, `safe_sqrt`, `powm1`, `pow2`
//!   through `pow8`, `tanh`, `acosh`, `asinh`, `atanh`, `atan2`, and `hypot`) in
//!   upstream's exact IEEE-754 operation order with no fused multiply-add
//!   contraction (`fma(a, b, c) = a * b + c`). For the 7 upstream MESA `26.4.1`
//!   Fortran files that omit `use math_lib` and invoke compiler intrinsics
//!   directly (`#196`, `#198`, `DECISIONS #23`), [`atan2`] and
//!   [`intrinsic_log`], [`intrinsic_log10`], [`intrinsic_exp`], and
//!   [`intrinsic_pow`] route through `crlibm`'s round-to-nearest (`*_rn`) entry
//!   points. This contract is specified at the bit level so a future pure-Rust
//!   correctly rounded backend can replace `flexmesa-crlibm-sys` and be
//!   verified bit for bit against it.
//! - **[`NumericsMode::Fast`]**: Evaluates Rust `std` `f64` intrinsics and
//!   permits hardware fused multiply-add via [`fma`] (`a.mul_add(b, c)`). `Fast`
//!   mode is hardware-dependent and is validated against `Strict` mode, never
//!   against MESA.

#![forbid(unsafe_code)]

mod fast_ops;
pub mod io;
pub mod mode;
pub mod pre_z;
pub mod trig;
pub mod vector_eval;

use flexmesa_const::LN10;

use mode::is_strict;

pub use io::{double_to_str, heaviside, sgn, str_to_double, str_to_vector};
pub use mode::{
    numerics_mode, set_numerics_mode, with_numerics_mode, NumericsMode,
    ParseNumericsModeError,
};
pub use pre_z::{pre_z, PrecompInt, MAX_PRECOMP_INTS};
pub use trig::{
    acos, acosh, acospi, asin, asinh, asinpi, atan, atan2, atanh, atanpi, cos,
    cosh, cospi, sin, sin_cos, sinh, sinpi, tan, tanh, tanpi,
};

/// Upstream MESA `MATH_BACKEND` constant (`math/public/math_lib_crmath.f90:34`).
pub const MATH_BACKEND: &str = "CRMATH";

/// Minimum positive floor applied by [`safe_log`] and [`safe_log10`]
/// (`math/public/math_lib_crmath.f90:139,157`).
pub const SAFE_LOG_FLOOR: f64 = 1.0e-99;

/// Fallback value returned by [`safe_log`] and [`safe_log10`] on non-finite
/// input (`math/public/math_lib_crmath.f90:135,153`).
pub const SAFE_LOG_BAD_RETURN: f64 = -99.0;

/// Exact IEEE-754 bit pattern of `ln10_m = log_rz(10.0_dp)` initialized by
/// `math_init` (`math/public/math_lib_crmath.f90:111`).
const LN10_CRMATH: f64 = f64::from_bits(0x4002_6bb1_bbb5_5515);

/// Initialize the math subsystem matching MESA's `math_init`
/// (`math/public/math_lib_crmath.f90:107-115`).
///
/// Calls `crlibm_init()`, ensures [`numerics_mode`] is initialized, and
/// populates the integer power lookup table [`pre_z()`].
pub fn math_init() {
    let _ = flexmesa_crlibm_sys::crlibm_init();
    let _ = numerics_mode();
    let _ = pre_z(1);
}

/// Evaluate `a * b + c` as two separate roundings in [`NumericsMode::Strict`]
/// and via hardware/software `f64::mul_add` in [`NumericsMode::Fast`] (`#165`).
#[must_use]
#[inline]
pub fn fma(a: f64, b: f64, c: f64) -> f64 {
    if is_strict() {
        a * b + c
    } else {
        fast_ops::mul_add(a, b, c)
    }
}

/// Compute the domain-safe square root $\sqrt{\max(x, 0)}$ matching
/// `math/public/math_lib_crmath.f90:118-125`.
#[must_use]
#[inline]
pub fn safe_sqrt(x: f64) -> f64 {
    (if x < 0.0 { 0.0 } else { x }).sqrt()
}

/// Compute the domain-safe natural logarithm $\ln(\max(10^{-99}, x))$ matching
/// `math/public/math_lib_crmath.f90:128-143`.
#[must_use]
#[inline]
pub fn safe_log(x: f64) -> f64 {
    if !x.is_finite() {
        return SAFE_LOG_BAD_RETURN;
    }
    log(if x < SAFE_LOG_FLOOR {
        SAFE_LOG_FLOOR
    } else {
        x
    })
}

/// Compute the domain-safe base-10 logarithm $\log_{10}(\max(10^{-99}, x))$
/// matching `math/public/math_lib_crmath.f90:146-161`.
///
/// # Examples
/// ```
/// use flexmesa_math::safe_log10;
///
/// assert_eq!(safe_log10(10.0), 1.0);
/// assert_eq!(safe_log10(0.0).to_bits(), 0xc058_bfff_ffff_ffff);
/// assert_eq!(safe_log10(f64::NAN), -99.0);
/// ```
#[must_use]
#[inline]
pub fn safe_log10(x: f64) -> f64 {
    if !x.is_finite() {
        return SAFE_LOG_BAD_RETURN;
    }
    log10(if x < SAFE_LOG_FLOOR {
        SAFE_LOG_FLOOR
    } else {
        x
    })
}

/// Evaluate `log_rz` unconditionally on the `Strict` (`CRMATH`) backend
/// (`math/public/math_lib_crmath.f90:65`, `crmath_r_dp.f90:127`) without
/// reading or mutating the process-wide [`NumericsMode`].
#[must_use]
#[inline]
pub fn log_strict(x: f64) -> f64 {
    flexmesa_crlibm_sys::log_rz(x)
}

/// Compute the natural logarithm $\ln(x)$ (`math/public/math_lib_crmath.f90:65`,
/// `crmath_r_dp.f90:127`).
#[must_use]
#[inline]
pub fn log(x: f64) -> f64 {
    if is_strict() {
        log_strict(x)
    } else {
        fast_ops::ln(x)
    }
}

/// Compute the base-10 logarithm $\log_{10}(x)$
/// (`math/public/math_lib_crmath.f90:67`, `crmath_r_dp.f90:157`).
#[must_use]
#[inline]
pub fn log10(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::log10_rz(x)
    } else {
        fast_ops::log10(x)
    }
}

/// Compute the shifted natural logarithm $\ln(1 + x)$
/// (`math/public/math_lib_crmath.f90:69`, `crmath_r_dp.f90:137`).
#[must_use]
#[inline]
pub fn log1p(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::log1p_rz(x)
    } else {
        fast_ops::ln_1p(x)
    }
}

/// Compute the base-2 logarithm $\log_{2}(x)$
/// (`math/public/math_lib_crmath.f90:70`, `crmath_r_dp.f90:147`).
#[must_use]
#[inline]
pub fn log2(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::log2_rz(x)
    } else {
        fast_ops::log2(x)
    }
}

/// Compute the natural exponential $e^{x}$
/// (`math/public/math_lib_crmath.f90:71`, `crmath_r_dp.f90:167`).
#[must_use]
#[inline]
pub fn exp(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::exp_rd(x)
    } else {
        fast_ops::exp(x)
    }
}

/// Compute the shifted exponential $e^{x} - 1$
/// (`math/public/math_lib_crmath.f90:73`, `crmath_r_dp.f90:177`).
#[must_use]
#[inline]
pub fn expm1(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::expm1_rz(x)
    } else {
        fast_ops::exp_m1(x)
    }
}

/// Compute the base-10 exponential $10^{x}$ matching MESA's `exp10_`
/// (`math/public/math_lib_crmath.f90:164-190`).
#[must_use]
#[inline]
#[allow(clippy::float_cmp)]
pub fn exp10(x: f64) -> f64 {
    if is_strict() {
        if x.is_finite()
            && x == x.floor()
            && (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&x)
        {
            #[allow(clippy::cast_possible_truncation)]
            let ix = x as i32;
            let steps = ix.unsigned_abs().min(400);
            let mut exp10_x = 1.0_f64;
            for _ in 0..steps {
                exp10_x *= 10.0_f64;
            }
            if ix < 0 {
                1.0_f64 / exp10_x
            } else {
                exp10_x
            }
        } else {
            flexmesa_crlibm_sys::exp_rd(x * LN10_CRMATH)
        }
    } else {
        fast_ops::exp(x * LN10)
    }
}

/// Compute the Euclidean hypotenuse $\sqrt{x^{2} + y^{2}}$ matching `CRMATH`'s
/// `hypot_r_r_dp_` (`crmath_r_dp.f90:187-202`) in [`NumericsMode::Strict`] and
/// `f64::hypot` in [`NumericsMode::Fast`].
#[must_use]
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    if is_strict() {
        let ax = x.abs();
        let ay = y.abs();
        if ax > ay {
            let r = y / x;
            ax * (1.0_f64 + r * r).sqrt()
        } else if ay == 0.0_f64 {
            0.0_f64
        } else {
            let r = x / y;
            ay * (1.0_f64 + r * r).sqrt()
        }
    } else {
        fast_ops::hypot(x, y)
    }
}

/// Evaluate `pow_r_` unconditionally on the `Strict` (`CRMATH`) backend
/// (`math/public/math_lib_crmath.f90:220-255`) without reading or mutating the
/// process-wide [`NumericsMode`].
#[must_use]
#[inline]
#[allow(clippy::float_cmp)]
pub fn pow_strict(x: f64, y: f64) -> f64 {
    if x == 0.0_f64 {
        0.0_f64
    } else if y.is_finite() && y == y.floor() && y.abs() < 100.0_f64 {
        #[allow(clippy::cast_possible_truncation)]
        let iy = y as i32;
        let mut pow_x = 1.0_f64;
        for _ in 0..iy.unsigned_abs() {
            pow_x *= x;
        }
        if iy < 0 {
            1.0_f64 / pow_x
        } else {
            pow_x
        }
    } else {
        flexmesa_crlibm_sys::exp_rd(flexmesa_crlibm_sys::log_rz(x) * y)
    }
}

/// Raise `x` to a floating-point power $x^{y}$ matching MESA's `pow_r_`
/// (`math/public/math_lib_crmath.f90:220-255`) in [`NumericsMode::Strict`].
#[must_use]
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    if is_strict() {
        pow_strict(x, y)
    } else {
        fast_ops::powf(x, y)
    }
}

/// Evaluate `pow_i_` unconditionally on the `Strict` (`CRMATH`) backend
/// (`math/public/math_lib_crmath.f90:193-217`; `powi_strict(0.0, 0) == 0.0`)
/// without reading or mutating the process-wide [`NumericsMode`].
#[must_use]
#[inline]
#[allow(clippy::float_cmp)]
pub fn powi_strict(x: f64, iy: i32) -> f64 {
    if x == 0.0_f64 {
        return 0.0_f64;
    }
    let n = iy.unsigned_abs();
    let mut pow_x = 1.0_f64;
    if n <= 2200 {
        for _ in 0..n {
            pow_x *= x;
        }
    } else if x == 1.0_f64 {
        pow_x = 1.0_f64;
    } else if x == -1.0_f64 {
        pow_x = if (n & 1) == 0 { 1.0_f64 } else { -1.0_f64 };
    } else {
        let steps = 2200 + (n & 1);
        for _ in 0..steps {
            pow_x *= x;
        }
    }
    if iy < 0 {
        1.0_f64 / pow_x
    } else {
        pow_x
    }
}

/// Raise `x` to an integer power $x^{n}$ matching MESA's `pow_i_`
/// (`math/public/math_lib_crmath.f90:193-217`; note `powi(0.0, 0) == 0.0`) in
/// [`NumericsMode::Strict`].
#[must_use]
#[inline]
pub fn powi(x: f64, iy: i32) -> f64 {
    if is_strict() {
        powi_strict(x, iy)
    } else {
        fast_ops::powi(x, iy)
    }
}

/// Raise `x` to a Fortran `**` integer power (`x ** m`, GCC `__powidf2` binary
/// squaring where `0.0 ** 0 == 1.0`, `#165` Item 10b).
#[must_use]
#[inline]
pub fn powi_f90(mut x: f64, m: i32) -> f64 {
    if is_strict() {
        let mut n: u32 = m.unsigned_abs();
        let mut y: f64 = if (n & 1) != 0 { x } else { 1.0_f64 };
        n >>= 1;
        while n != 0 {
            x *= x;
            if (n & 1) != 0 {
                y *= x;
            }
            n >>= 1;
        }
        if m < 0 {
            1.0_f64 / y
        } else {
            y
        }
    } else {
        fast_ops::powi(x, m)
    }
}

/// Compute the reciprocal $x^{-1} = 1 / x$ (`math/private/math_pown.f90:64-71`).
#[must_use]
#[inline]
pub fn powm1(x: f64) -> f64 {
    1.0_f64 / x
}

/// Compute $x^{2}$ via explicit multiplication (`math/private/math_pown.f90:74-81`).
#[must_use]
#[inline]
pub fn pow2(x: f64) -> f64 {
    x * x
}

/// Compute $x^{3}$ via explicit multiplication (`math/private/math_pown.f90:84-91`).
#[must_use]
#[inline]
pub fn pow3(x: f64) -> f64 {
    x * x * x
}

/// Compute $x^{4}$ via explicit multiplication (`math/private/math_pown.f90:94-101`).
#[must_use]
#[inline]
pub fn pow4(x: f64) -> f64 {
    x * x * x * x
}

/// Compute $x^{5}$ via explicit multiplication (`math/private/math_pown.f90:104-111`).
#[must_use]
#[inline]
pub fn pow5(x: f64) -> f64 {
    x * x * x * x * x
}

/// Compute $x^{6}$ via explicit multiplication (`math/private/math_pown.f90:114-121`).
#[must_use]
#[inline]
pub fn pow6(x: f64) -> f64 {
    x * x * x * x * x * x
}

/// Compute $x^{7}$ via explicit multiplication (`math/private/math_pown.f90:124-131`).
#[must_use]
#[inline]
pub fn pow7(x: f64) -> f64 {
    x * x * x * x * x * x * x
}

/// Compute $x^{8}$ via explicit multiplication (`math/private/math_pown.f90:134-141`).
#[must_use]
#[inline]
pub fn pow8(x: f64) -> f64 {
    x * x * x * x * x * x * x * x
}

/// Compute $\ln(x)$ via `crlibm::log_rn` (round-to-nearest) in
/// [`NumericsMode::Strict`] for upstream MESA `26.4.1` call sites that bypass
/// `math_lib` into Fortran's `dlog`/`log` intrinsic (`#198`; `100.00%` bit
/// match on `10,067` vectors vs `61.92%` for `log_rz`).
#[must_use]
#[inline]
pub fn intrinsic_log(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::log_rn(x)
    } else {
        fast_ops::ln(x)
    }
}

/// Compute $\log_{10}(x)$ via `crlibm::log10_rn` (round-to-nearest) in
/// [`NumericsMode::Strict`] for upstream MESA `26.4.1` call sites that bypass
/// `math_lib` into Fortran's `dlog10`/`log10` intrinsic (`#198`; `99.84%` /
/// `99.76%` bit match with `max_ulp = 1` vs `62.84%` / `61.45%` for `log10_rz`).
#[must_use]
#[inline]
pub fn intrinsic_log10(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::log10_rn(x)
    } else {
        fast_ops::log10(x)
    }
}

/// Compute $\exp(x)$ via `crlibm::exp_rn` (round-to-nearest) in
/// [`NumericsMode::Strict`] for upstream MESA `26.4.1` call sites that bypass
/// `math_lib` into Fortran's `exp` intrinsic (`#198`; `100.00%` bit match on
/// `10,067` vectors vs `50.57%` for `exp_rd`).
#[must_use]
#[inline]
pub fn intrinsic_exp(x: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::exp_rn(x)
    } else {
        fast_ops::exp(x)
    }
}

/// Compute $x^{y}$ via `crlibm::pow_rn` (round-to-nearest) in
/// [`NumericsMode::Strict`] for upstream MESA `26.4.1` call sites that bypass
/// `math_lib` into Fortran's `real(dp) ** real(dp)` operator (`#198`; `99.97%`
/// bit match with `max_ulp = 1` on `10,294` vectors vs `41.27%` and
/// `max_ulp = 1,493` for `pow_r_`).
#[must_use]
#[inline]
pub fn intrinsic_pow(x: f64, y: f64) -> f64 {
    if is_strict() {
        flexmesa_crlibm_sys::pow_rn(x, y)
    } else {
        fast_ops::powf(x, y)
    }
}
