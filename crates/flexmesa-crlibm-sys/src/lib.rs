// SPDX-License-Identifier: LGPL-2.1-or-later
//! Safe Rust bindings to the vendored INRIA `crlibm 1.0beta4` C routines called
//! by MESA `26.4.1`'s `CRMATH 1.2` library (`#165`).
//!
//! # Safety & Reentrancy Architecture
//!
//! This crate is the **sole** crate in the `FlexMESA` Cargo workspace permitted
//! to contain `unsafe` blocks, strictly confined to the private `ffi` module
//! below (`DECISIONS #21`). Every public function exposed by this crate is safe:
//!
//! 1. **32-bit `x86` compile-time guard**: 32-bit `x86` (`x87` 80-bit register
//!    double rounding) is rejected via [`compile_error!`]; `x86_64` builds
//!    enforce `-msse2 -mfpmath=sse`, and all targets enforce `-std=c99
//!    -ffp-contract=off -fno-fast-math` with no `-march=native`.
//! 2. **Default IEEE-754 floating-point environment**: `CRLIBM_HAS_FPU_CONTROL`
//!    is unset in every committed target header under `config/`, so all 20
//!    routines execute pure IEEE-754 `double` (triple-double and SCS) arithmetic
//!    under the default round-to-nearest-even hardware environment and never
//!    read or modify FPU control registers (`crlibm_init` returns `0`).
//! 3. **Stateless multi-thread reentrancy**: `EVAL_PERF` is set to `0` in
//!    `csrc/crlibm_private.h` and `csrc/exp-td.c`, removing the non-atomic
//!    `crlibm_second_step_taken` counter. All remaining static tables in `csrc/`
//!    are `const`.
//! 4. **Rounding modes matching `CRMATH 1.2` (`crmath_r_dp.f90`)**: 17 entry
//!    points use directed round-toward-zero (`*_rz`), and 3 (`exp_rd`,
//!    `acos_rd`, `acospi_rd`) use directed round-toward-minus-infinity (`*_rd`),
//!    matching `CRMATH 1.2` symbol for symbol.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
compile_error!(
    "flexmesa-crlibm-sys does not support 32-bit x86 (x87 extended-precision \
     double rounding breaks strict IEEE-754 bit-reproducibility); build for \
     x86_64 or aarch64."
);

mod ffi {
    use core::ffi::c_ulonglong;

    extern "C" {
        pub(super) fn crlibm_init() -> c_ulonglong;

        pub(super) fn log_rz(x: f64) -> f64;
        pub(super) fn log1p_rz(x: f64) -> f64;
        pub(super) fn log2_rz(x: f64) -> f64;
        pub(super) fn log10_rz(x: f64) -> f64;

        pub(super) fn exp_rd(x: f64) -> f64;
        pub(super) fn expm1_rz(x: f64) -> f64;

        pub(super) fn cos_rz(x: f64) -> f64;
        pub(super) fn sin_rz(x: f64) -> f64;
        pub(super) fn tan_rz(x: f64) -> f64;

        pub(super) fn cospi_rz(x: f64) -> f64;
        pub(super) fn sinpi_rz(x: f64) -> f64;
        pub(super) fn tanpi_rz(x: f64) -> f64;

        pub(super) fn acos_rd(x: f64) -> f64;
        pub(super) fn asin_rz(x: f64) -> f64;
        pub(super) fn atan_rz(x: f64) -> f64;

        pub(super) fn acospi_rd(x: f64) -> f64;
        pub(super) fn asinpi_rz(x: f64) -> f64;
        pub(super) fn atanpi_rz(x: f64) -> f64;

        pub(super) fn cosh_rz(x: f64) -> f64;
        pub(super) fn sinh_rz(x: f64) -> f64;

        pub(super) fn log_rn(x: f64) -> f64;
        pub(super) fn log10_rn(x: f64) -> f64;
        pub(super) fn exp_rn(x: f64) -> f64;
        pub(super) fn atan_rn(x: f64) -> f64;
        pub(super) fn pow_rn(x: f64, y: f64) -> f64;
    }
}

/// Invoke `crlibm_init()` matching `CRMATH`'s `crmath_init()` (`crmath.f90:36`,
/// called by `math/public/math_lib_crmath.f90:109`).
///
/// With `CRLIBM_HAS_FPU_CONTROL` unset in the committed per-target headers,
/// `crlibm_init()` performs no FPU control-word mutation and returns `0`.
#[must_use]
#[inline]
pub fn crlibm_init() -> u64 {
    // SAFETY: `ffi::crlibm_init` takes no arguments, dereferences no pointers,
    // mutates no global state or FPU control registers under our committed
    // target config headers, and returns a plain integer value (`0`).
    unsafe { ffi::crlibm_init() }
}

/// Correctly rounded natural logarithm toward zero (`crlibm::log_rz`,
/// `crmath_r_dp.f90:131`).
#[must_use]
#[inline]
pub fn log_rz(x: f64) -> f64 {
    // SAFETY: `ffi::log_rz` accepts an IEEE-754 `f64` (`double`) by value,
    // reads only `static const` polynomial/reduction tables, mutates no global
    // or FPU state (`EVAL_PERF = 0`), and is pure and thread-reentrant for all
    // 64-bit IEEE-754 bit patterns (including `±0`, subnormals, `±Inf`, `NaN`).
    unsafe { ffi::log_rz(x) }
}

/// Correctly rounded `ln(1 + x)` toward zero (`crlibm::log1p_rz`,
/// `crmath_r_dp.f90:141`).
#[must_use]
#[inline]
pub fn log1p_rz(x: f64) -> f64 {
    // SAFETY: `ffi::log1p_rz` takes `f64` by value, reads only `static const`
    // tables, mutates no global or FPU state, and is pure and reentrant for all
    // `f64` inputs.
    unsafe { ffi::log1p_rz(x) }
}

/// Correctly rounded base-2 logarithm toward zero (`crlibm::log2_rz`,
/// `crmath_r_dp.f90:151`).
#[must_use]
#[inline]
pub fn log2_rz(x: f64) -> f64 {
    // SAFETY: `ffi::log2_rz` takes `f64` by value, reads only `static const`
    // tables, mutates no global or FPU state, and is pure and reentrant for all
    // `f64` inputs.
    unsafe { ffi::log2_rz(x) }
}

/// Correctly rounded base-10 logarithm toward zero (`crlibm::log10_rz`,
/// `crmath_r_dp.f90:161`).
#[must_use]
#[inline]
pub fn log10_rz(x: f64) -> f64 {
    // SAFETY: `ffi::log10_rz` takes `f64` by value, reads only `static const`
    // tables, mutates no global or FPU state, and is pure and reentrant for all
    // `f64` inputs.
    unsafe { ffi::log10_rz(x) }
}

/// Correctly rounded natural exponential toward `−∞` (`crlibm::exp_rd`,
/// identical to toward-zero since `exp(x) >= 0`; `crmath_r_dp.f90:171`).
#[must_use]
#[inline]
pub fn exp_rd(x: f64) -> f64 {
    // SAFETY: `ffi::exp_rd` takes `f64` by value, reads only `static const`
    // tables, mutates no global or FPU state, and is pure and reentrant for all
    // `f64` inputs.
    unsafe { ffi::exp_rd(x) }
}

/// Correctly rounded `exp(x) - 1` toward zero (`crlibm::expm1_rz`,
/// `crmath_r_dp.f90:181`).
#[must_use]
#[inline]
pub fn expm1_rz(x: f64) -> f64 {
    // SAFETY: `ffi::expm1_rz` takes `f64` by value, reads only `static const`
    // tables, mutates no global or FPU state, and is pure and reentrant for all
    // `f64` inputs.
    unsafe { ffi::expm1_rz(x) }
}

/// Correctly rounded cosine toward zero (`crlibm::cos_rz`,
/// `crmath_r_dp.f90:210`).
#[must_use]
#[inline]
pub fn cos_rz(x: f64) -> f64 {
    // SAFETY: `ffi::cos_rz` takes `f64` by value, uses stack-local `scs`
    // buffers and `static const` tables, mutates no global or FPU state, and is
    // pure and reentrant for all `f64` inputs.
    unsafe { ffi::cos_rz(x) }
}

/// Correctly rounded sine toward zero (`crlibm::sin_rz`,
/// `crmath_r_dp.f90:220`).
#[must_use]
#[inline]
pub fn sin_rz(x: f64) -> f64 {
    // SAFETY: `ffi::sin_rz` takes `f64` by value, uses stack-local `scs`
    // buffers and `static const` tables, mutates no global or FPU state, and is
    // pure and reentrant for all `f64` inputs.
    unsafe { ffi::sin_rz(x) }
}

/// Correctly rounded tangent toward zero (`crlibm::tan_rz`,
/// `crmath_r_dp.f90:230`).
#[must_use]
#[inline]
pub fn tan_rz(x: f64) -> f64 {
    // SAFETY: `ffi::tan_rz` takes `f64` by value, uses stack-local `scs`
    // buffers and `static const` tables, mutates no global or FPU state, and is
    // pure and reentrant for all `f64` inputs.
    unsafe { ffi::tan_rz(x) }
}

/// Correctly rounded `cos(pi * x)` toward zero (`crlibm::cospi_rz`,
/// `crmath_r_dp.f90:240`).
#[must_use]
#[inline]
pub fn cospi_rz(x: f64) -> f64 {
    // SAFETY: `ffi::cospi_rz` takes `f64` by value, uses stack-local `scs`
    // buffers and `static const` tables, mutates no global or FPU state, and is
    // pure and reentrant for all `f64` inputs.
    unsafe { ffi::cospi_rz(x) }
}

/// Correctly rounded `sin(pi * x)` toward zero (`crlibm::sinpi_rz`,
/// `crmath_r_dp.f90:250`).
#[must_use]
#[inline]
pub fn sinpi_rz(x: f64) -> f64 {
    // SAFETY: `ffi::sinpi_rz` takes `f64` by value, uses stack-local `scs`
    // buffers and `static const` tables, mutates no global or FPU state, and is
    // pure and reentrant for all `f64` inputs.
    unsafe { ffi::sinpi_rz(x) }
}

/// Correctly rounded `tan(pi * x)` toward zero (`crlibm::tanpi_rz`,
/// `crmath_r_dp.f90:260`).
#[must_use]
#[inline]
pub fn tanpi_rz(x: f64) -> f64 {
    // SAFETY: `ffi::tanpi_rz` takes `f64` by value, uses stack-local `scs`
    // buffers and `static const` tables, mutates no global or FPU state, and is
    // pure and reentrant for all `f64` inputs.
    unsafe { ffi::tanpi_rz(x) }
}

/// Correctly rounded `arccos(x)` toward `−∞` (`crlibm::acos_rd`, identical to
/// toward-zero since `acos(x) >= 0`; `crmath_r_dp.f90:270`).
#[must_use]
#[inline]
pub fn acos_rd(x: f64) -> f64 {
    // SAFETY: `ffi::acos_rd` takes `f64` by value, reads only `static const`
    // tables, mutates no global or FPU state, and is pure and reentrant for all
    // `f64` inputs.
    unsafe { ffi::acos_rd(x) }
}

/// Correctly rounded `arcsin(x)` toward zero (`crlibm::asin_rz`,
/// `crmath_r_dp.f90:280`).
#[must_use]
#[inline]
pub fn asin_rz(x: f64) -> f64 {
    // SAFETY: `ffi::asin_rz` takes `f64` by value, reads only `static const`
    // tables, mutates no global or FPU state, and is pure and reentrant for all
    // `f64` inputs.
    unsafe { ffi::asin_rz(x) }
}

/// Correctly rounded `arctan(x)` toward zero (`crlibm::atan_rz`,
/// `crmath_r_dp.f90:290`).
#[must_use]
#[inline]
pub fn atan_rz(x: f64) -> f64 {
    // SAFETY: `ffi::atan_rz` takes `f64` by value, uses stack-local `scs`
    // buffers and `static const` tables, mutates no global or FPU state, and is
    // pure and reentrant for all `f64` inputs.
    unsafe { ffi::atan_rz(x) }
}

/// Correctly rounded `arccos(x) / pi` toward `−∞` (`crlibm::acospi_rd`,
/// identical to toward-zero since `acospi(x) >= 0`; `crmath_r_dp.f90:319`).
#[must_use]
#[inline]
pub fn acospi_rd(x: f64) -> f64 {
    // SAFETY: `ffi::acospi_rd` takes `f64` by value, reads only `static const`
    // tables, mutates no global or FPU state, and is pure and reentrant for all
    // `f64` inputs.
    unsafe { ffi::acospi_rd(x) }
}

/// Correctly rounded `arcsin(x) / pi` toward zero (`crlibm::asinpi_rz`,
/// `crmath_r_dp.f90:329`).
#[must_use]
#[inline]
pub fn asinpi_rz(x: f64) -> f64 {
    // SAFETY: `ffi::asinpi_rz` takes `f64` by value, reads only `static const`
    // tables, mutates no global or FPU state, and is pure and reentrant for all
    // `f64` inputs.
    unsafe { ffi::asinpi_rz(x) }
}

/// Correctly rounded `arctan(x) / pi` toward zero (`crlibm::atanpi_rz`,
/// `crmath_r_dp.f90:339`).
#[must_use]
#[inline]
pub fn atanpi_rz(x: f64) -> f64 {
    // SAFETY: `ffi::atanpi_rz` takes `f64` by value, uses stack-local `scs`
    // buffers and `static const` tables, mutates no global or FPU state, and is
    // pure and reentrant for all `f64` inputs.
    unsafe { ffi::atanpi_rz(x) }
}

/// Correctly rounded `cosh(x)` toward zero (`crlibm::cosh_rz`,
/// `crmath_r_dp.f90:349`).
#[must_use]
#[inline]
pub fn cosh_rz(x: f64) -> f64 {
    // SAFETY: `ffi::cosh_rz` takes `f64` by value, reads only `static const`
    // tables, mutates no global or FPU state, and is pure and reentrant for all
    // `f64` inputs.
    unsafe { ffi::cosh_rz(x) }
}

/// Correctly rounded `sinh(x)` toward zero (`crlibm::sinh_rz`,
/// `crmath_r_dp.f90:359`).
#[must_use]
#[inline]
pub fn sinh_rz(x: f64) -> f64 {
    // SAFETY: `ffi::sinh_rz` takes `f64` by value, reads only `static const`
    // tables, mutates no global or FPU state, and is pure and reentrant for all
    // `f64` inputs.
    unsafe { ffi::sinh_rz(x) }
}

/// Correctly rounded natural logarithm to nearest-even (`crlibm::log_rn`,
/// used for upstream MESA Fortran `dlog`/`log` intrinsic bypasses; `#198`).
#[must_use]
#[inline]
pub fn log_rn(x: f64) -> f64 {
    // SAFETY: `ffi::log_rn` takes `f64` by value, reads only `static const`
    // tables, mutates no global or FPU state, and is pure and reentrant for all
    // `f64` inputs.
    unsafe { ffi::log_rn(x) }
}

/// Correctly rounded base-10 logarithm to nearest-even (`crlibm::log10_rn`,
/// used for upstream MESA Fortran `dlog10`/`log10` intrinsic bypasses; `#198`).
#[must_use]
#[inline]
pub fn log10_rn(x: f64) -> f64 {
    // SAFETY: `ffi::log10_rn` takes `f64` by value, reads only `static const`
    // tables, mutates no global or FPU state, and is pure and reentrant for all
    // `f64` inputs.
    unsafe { ffi::log10_rn(x) }
}

/// Correctly rounded natural exponential to nearest-even (`crlibm::exp_rn`,
/// used for upstream MESA Fortran `exp` intrinsic bypasses; `#198`).
#[must_use]
#[inline]
pub fn exp_rn(x: f64) -> f64 {
    // SAFETY: `ffi::exp_rn` takes `f64` by value, reads only `static const`
    // tables, mutates no global or FPU state (`EVAL_PERF = 0`), and is pure and
    // reentrant for all `f64` inputs.
    unsafe { ffi::exp_rn(x) }
}

/// Correctly rounded `arctan(x)` to nearest-even (`crlibm::atan_rn`, used by
/// octant-reduced `atan2` for upstream MESA `atan2`/`datan2`; `#196`, `#198`).
#[must_use]
#[inline]
pub fn atan_rn(x: f64) -> f64 {
    // SAFETY: `ffi::atan_rn` takes `f64` by value, uses stack-local `scs`
    // buffers and `static const` tables, mutates no global or FPU state, and is
    // pure and reentrant for all `f64` inputs.
    unsafe { ffi::atan_rn(x) }
}

/// Triple-double round-to-nearest `pow(x, y)` (`crlibm::pow_rn`, used for
/// upstream MESA Fortran `real(dp) ** real(dp)` intrinsic bypasses; `#198`).
#[must_use]
#[inline]
pub fn pow_rn(x: f64, y: f64) -> f64 {
    // SAFETY: `ffi::pow_rn` takes two `f64` values, reads only `static const`
    // tables in `pow.h`, mutates no global or FPU state (`USE_BUILTINS = 0`),
    // and is pure and reentrant for all `f64` input pairs.
    unsafe { ffi::pow_rn(x, y) }
}
