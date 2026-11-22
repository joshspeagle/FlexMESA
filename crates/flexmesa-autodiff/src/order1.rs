// SPDX-License-Identifier: LGPL-3.0-or-later
//! Const-generic first-order forward-mode dual numbers (`DualOrder1<N>`).
//!
//! `DualOrder1<N>` unifies MESA's `auto_diff_real_1var_order1` ($N = 1$),
//! `auto_diff_real_2var_order1` ($N = 2$), `auto_diff_real_4var_order1`
//! ($N = 4$), and `auto_diff_real_star_order1` ($N = 33$) into a single
//! zero-allocation stack-allocated type.

use flexmesa_const::{LN10, PI};
use flexmesa_math::{
    acos, acosh, asin, asinh, atan, atanh, cos, cosh, exp, expm1, log, log1p,
    pow, pow2, pow3, pow4, pow5, pow6, pow7, pow8, powm1, safe_log, sin, sinh,
    tan, tanh,
};

/// First-order forward-mode dual number with `N` partial derivatives.
///
/// Stores the primal value `val` ($f(\mathbf{x})$) and the `N`-component
/// gradient array `d1` ($\nabla f(\mathbf{x})$) in IEEE-754 `f64`.
///
/// # Examples
/// ```
/// use flexmesa_autodiff::AutoDiffReal2VarOrder1;
///
/// let x = AutoDiffReal2VarOrder1::independent(3.0, 0);
/// let y = AutoDiffReal2VarOrder1::independent(2.0, 1);
/// let z = x * y + x.pow2();
/// assert_eq!(z.val, 15.0);
/// assert_eq!(z.d1, [8.0, 3.0]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DualOrder1<const N: usize> {
    /// Primal function value $f(\mathbf{x})$.
    pub val: f64,
    /// First-order partial derivatives $[\partial f / \partial x_{0}, \dots]$.
    pub d1: [f64; N],
}

impl<const N: usize> DualOrder1<N> {
    /// Construct a dual number from an explicit value and partial-derivative
    /// array.
    #[must_use]
    #[inline]
    pub const fn new(val: f64, d1: [f64; N]) -> Self {
        Self { val, d1 }
    }

    /// Construct a constant dual number with zero partial derivatives.
    #[must_use]
    #[inline]
    pub const fn constant(val: f64) -> Self {
        Self { val, d1: [0.0; N] }
    }

    /// Construct the additive identity (`0.0` with zero partial derivatives).
    #[must_use]
    #[inline]
    pub const fn zero() -> Self {
        Self::constant(0.0)
    }

    /// Natural logarithm alias (`self.log()`).
    #[must_use]
    #[inline]
    pub fn ln(self) -> Self {
        self.log()
    }

    /// Reciprocal alias (`self.powm1()`).
    #[must_use]
    #[inline]
    pub fn recip(self) -> Self {
        self.powm1()
    }

    /// Construct the `index`-th independent variable
    /// ($\partial x_{i} / \partial x_{i} = 1$).
    ///
    /// # Panics
    /// Panics if `index >= N`.
    #[must_use]
    #[inline]
    pub fn independent(val: f64, index: usize) -> Self {
        let mut d1 = [0.0; N];
        d1[index] = 1.0;
        Self { val, d1 }
    }

    /// Apply the first-order unary chain rule $z = f(x)$ given $f(x)$ and
    /// $f'(x)$.
    #[must_use]
    #[inline]
    pub fn make_unary_operator(self, z_val: f64, z_d1x: f64) -> Self {
        let mut d1 = [0.0; N];
        for (slot, &dx) in d1.iter_mut().zip(&self.d1) {
            *slot = dx * z_d1x;
        }
        Self { val: z_val, d1 }
    }

    /// Apply the first-order binary chain rule $z = g(x, y)$ given partials.
    #[must_use]
    #[inline]
    #[allow(clippy::similar_names)]
    pub fn make_binary_operator(
        self,
        other: Self,
        z_val: f64,
        z_d1x: f64,
        z_d1y: f64,
    ) -> Self {
        let mut d1 = [0.0; N];
        for ((slot, &dx), &dy) in d1.iter_mut().zip(&self.d1).zip(&other.d1) {
            *slot = dx * z_d1x + dy * z_d1y;
        }
        Self { val: z_val, d1 }
    }

    /// Propagate a scalar derivative kernel `(f(x), f'(x), f''(x))` through
    /// `self`.
    #[must_use]
    #[inline]
    pub fn apply_unary(self, kernel: fn(f64) -> (f64, f64, f64)) -> Self {
        let (z_val, z_d1x, _) = kernel(self.val);
        self.make_unary_operator(z_val, z_d1x)
    }

    /// Whether `N` selects one of upstream's types whose generated code
    /// factors common terms out of the derivative
    /// (`auto_diff_real_2var_order1` and `auto_diff_real_4var_order1`).
    ///
    /// The one-variable type and the array types (`auto_diff_real_star_order1`,
    /// `auto_diff_real_15var_order1`) write each derivative as one expression
    /// evaluated left to right. The two forms round differently.
    const FACTORED: bool = N == 2 || N == 4;

    /// Derivative of the form upstream writes as `a*dx*b`.
    ///
    /// The array and one-variable types evaluate `(a*dx)*b`; the two- and
    /// four-variable types evaluate `q = a*b` once and then `q*dx`.
    #[inline]
    fn scaled(self, val: f64, a: f64, b: f64) -> Self {
        let mut d1 = [0.0; N];
        if Self::FACTORED {
            let q = a * b;
            for (slot, &dx) in d1.iter_mut().zip(&self.d1) {
                *slot = q * dx;
            }
        } else {
            for (slot, &dx) in d1.iter_mut().zip(&self.d1) {
                *slot = a * dx * b;
            }
        }
        Self { val, d1 }
    }

    /// `sign_self`: $\operatorname{sgn}(x)$ with zero derivatives.
    #[must_use]
    #[inline]
    pub fn sign(self) -> Self {
        Self::constant(sgn(self.val))
    }

    /// `abs_self`: $|x|$ with derivative `dx*sgn(x)`.
    #[must_use]
    #[inline]
    pub fn abs(self) -> Self {
        self.make_unary_operator(self.val.abs(), sgn(self.val))
    }

    /// `sqrt_self`: `q0 = sqrt(x)`, derivative `0.5*dx*powm1(q0)`.
    #[must_use]
    #[inline]
    pub fn sqrt(self) -> Self {
        let q0 = self.val.sqrt();
        self.scaled(q0, 0.5, powm1(q0))
    }

    /// `safe_sqrt_self`: `q0 = sqrt(x*Heaviside(x))`, derivative
    /// `0.5*q0*dx*powm1(x)`.
    ///
    /// At `x == 0` upstream's expression is `0*dx*Inf`, which is `NaN`; that
    /// is reproduced.
    #[must_use]
    #[inline]
    pub fn safe_sqrt(self) -> Self {
        let q0 = (self.val * heaviside(self.val)).sqrt();
        self.scaled(q0, 0.5 * q0, powm1(self.val))
    }

    /// `exp_self`: `q0 = exp(x)`, derivative `q0*dx`.
    #[must_use]
    #[inline]
    pub fn exp(self) -> Self {
        let q0 = exp(self.val);
        self.make_unary_operator(q0, q0)
    }

    /// `expm1_self`: value `expm1(x)`, derivative `dx*exp(x)`.
    #[must_use]
    #[inline]
    pub fn expm1(self) -> Self {
        self.make_unary_operator(expm1(self.val), exp(self.val))
    }

    /// `exp10_self`: `q0 = pow(10, x)`, derivative `q0*dx*ln10`.
    #[must_use]
    #[inline]
    pub fn exp10(self) -> Self {
        let q0 = pow(10.0, self.val);
        self.scaled(q0, q0, LN10)
    }

    /// `powm1_self`: value `powm1(x)`, derivative `-dx*powm1(pow2(x))`.
    #[must_use]
    #[inline]
    pub fn powm1(self) -> Self {
        self.make_unary_operator(powm1(self.val), -powm1(pow2(self.val)))
    }

    /// `log_self`: value `log(x)`, derivative `dx*powm1(x)`.
    #[must_use]
    #[inline]
    pub fn log(self) -> Self {
        self.make_unary_operator(log(self.val), powm1(self.val))
    }

    /// `safe_log_self`: value `safe_log(x)`, derivative `dx*powm1(x)` with no
    /// guard, as upstream.
    #[must_use]
    #[inline]
    pub fn safe_log(self) -> Self {
        self.make_unary_operator(safe_log(self.val), powm1(self.val))
    }

    /// `log1p_self`: value `log1p(x)`, derivative `dx*powm1(x + 1)`.
    #[must_use]
    #[inline]
    pub fn log1p(self) -> Self {
        self.make_unary_operator(log1p(self.val), powm1(self.val + 1.0))
    }

    /// `log10_self`: `q0 = powm1(ln10)`, value `q0*log(x)`, derivative
    /// `q0*dx*powm1(x)`.
    ///
    /// The value is a product, not the correctly rounded base-10 logarithm.
    #[must_use]
    #[inline]
    pub fn log10(self) -> Self {
        let q0 = powm1(LN10);
        self.scaled(q0 * log(self.val), q0, powm1(self.val))
    }

    /// `safe_log10_self`: `q0 = powm1(ln10)`, value `q0*safe_log(x)`,
    /// derivative `q0*dx*powm1(x)` with no guard, as upstream.
    #[must_use]
    #[inline]
    pub fn safe_log10(self) -> Self {
        let q0 = powm1(LN10);
        self.scaled(q0 * safe_log(self.val), q0, powm1(self.val))
    }

    /// `log2_self`: `q0 = powm1(log(2))`, value `q0*log(x)`, derivative
    /// `q0*dx*powm1(x)`.
    #[must_use]
    #[inline]
    pub fn log2(self) -> Self {
        let q0 = powm1(log(2.0));
        self.scaled(q0 * log(self.val), q0, powm1(self.val))
    }

    /// `sin_self`: value `sin(x)`, derivative `dx*cos(x)`.
    #[must_use]
    #[inline]
    pub fn sin(self) -> Self {
        self.make_unary_operator(sin(self.val), cos(self.val))
    }

    /// `cos_self`: value `cos(x)`, derivative `-dx*sin(x)`.
    #[must_use]
    #[inline]
    pub fn cos(self) -> Self {
        self.make_unary_operator(cos(self.val), -sin(self.val))
    }

    /// `tan_self`: `q0 = tan(x)`, derivative `dx*(pow2(q0) + 1)`.
    #[must_use]
    #[inline]
    pub fn tan(self) -> Self {
        let q0 = tan(self.val);
        self.make_unary_operator(q0, pow2(q0) + 1.0)
    }

    /// `sinpi_self`: `q0 = pi*x`, value `sin(q0)`, derivative
    /// `pi*dx*cos(q0)`.
    #[must_use]
    #[inline]
    pub fn sinpi(self) -> Self {
        let q0 = PI * self.val;
        self.scaled(sin(q0), PI, cos(q0))
    }

    /// `cospi_self`: `q0 = pi*x`, value `cos(q0)`, derivative
    /// `-pi*dx*sin(q0)`.
    #[must_use]
    #[inline]
    pub fn cospi(self) -> Self {
        let q0 = PI * self.val;
        self.scaled(cos(q0), -PI, sin(q0))
    }

    /// `tanpi_self`: `q0 = tan(pi*x)`, derivative `pi*dx*(pow2(q0) + 1)`.
    #[must_use]
    #[inline]
    pub fn tanpi(self) -> Self {
        let q0 = tan(PI * self.val);
        self.scaled(q0, PI, pow2(q0) + 1.0)
    }

    /// `sinh_self`: value `sinh(x)`, derivative `dx*cosh(x)`.
    #[must_use]
    #[inline]
    pub fn sinh(self) -> Self {
        self.make_unary_operator(sinh(self.val), cosh(self.val))
    }

    /// `cosh_self`: value `cosh(x)`, derivative `dx*sinh(x)`.
    #[must_use]
    #[inline]
    pub fn cosh(self) -> Self {
        self.make_unary_operator(cosh(self.val), sinh(self.val))
    }

    /// `tanh_self`: `q0 = tanh(x)`, derivative `-dx*(pow2(q0) - 1)`.
    #[must_use]
    #[inline]
    pub fn tanh(self) -> Self {
        let q0 = tanh(self.val);
        self.make_unary_operator(q0, -(pow2(q0) - 1.0))
    }

    /// `asin_self`: derivative `dx*powm1(sqrt(1 - pow2(x)))`.
    #[must_use]
    #[inline]
    pub fn asin(self) -> Self {
        let q0 = powm1((1.0 - pow2(self.val)).sqrt());
        self.make_unary_operator(asin(self.val), q0)
    }

    /// `acos_self`: derivative `-dx*powm1(sqrt(1 - pow2(x)))`.
    #[must_use]
    #[inline]
    pub fn acos(self) -> Self {
        let q0 = powm1((1.0 - pow2(self.val)).sqrt());
        self.make_unary_operator(acos(self.val), -q0)
    }

    /// `atan_self`: derivative `dx*powm1(pow2(x) + 1)`.
    #[must_use]
    #[inline]
    pub fn atan(self) -> Self {
        self.make_unary_operator(atan(self.val), powm1(pow2(self.val) + 1.0))
    }

    /// `asinpi_self`: `q0 = powm1(pi)`, value `q0*asin(x)`, derivative
    /// `q0*dx*powm1(sqrt(1 - pow2(x)))`.
    #[must_use]
    #[inline]
    pub fn asinpi(self) -> Self {
        let q0 = powm1(PI);
        let b = powm1((1.0 - pow2(self.val)).sqrt());
        self.scaled(q0 * asin(self.val), q0, b)
    }

    /// `acospi_self`: `q0 = powm1(pi)`, value `q0*acos(x)`, derivative
    /// `-q0*dx*powm1(sqrt(1 - pow2(x)))`.
    #[must_use]
    #[inline]
    pub fn acospi(self) -> Self {
        let q0 = powm1(PI);
        let b = powm1((1.0 - pow2(self.val)).sqrt());
        self.scaled(q0 * acos(self.val), -q0, b)
    }

    /// `atanpi_self`: value `powm1(pi)*atan(x)`, derivative
    /// `dx*powm1(pi*pow2(x) + pi)`.
    #[must_use]
    #[inline]
    pub fn atanpi(self) -> Self {
        self.make_unary_operator(
            powm1(PI) * atan(self.val),
            powm1(PI * pow2(self.val) + PI),
        )
    }

    /// `asinh_self`: derivative `dx*powm1(sqrt(pow2(x) + 1))`.
    #[must_use]
    #[inline]
    pub fn asinh(self) -> Self {
        let q0 = powm1((pow2(self.val) + 1.0).sqrt());
        self.make_unary_operator(asinh(self.val), q0)
    }

    /// `acosh_self`: derivative `dx*powm1(sqrt(pow2(x) - 1))`.
    #[must_use]
    #[inline]
    pub fn acosh(self) -> Self {
        let q0 = powm1((pow2(self.val) - 1.0).sqrt());
        self.make_unary_operator(acosh(self.val), q0)
    }

    /// `atanh_self`: derivative `-dx*powm1(pow2(x) - 1)`.
    #[must_use]
    #[inline]
    pub fn atanh(self) -> Self {
        self.make_unary_operator(atanh(self.val), -powm1(pow2(self.val) - 1.0))
    }

    /// `pow2_self`: derivative `2*dx*x`.
    #[must_use]
    #[inline]
    pub fn pow2(self) -> Self {
        self.scaled(pow2(self.val), 2.0, self.val)
    }

    /// `pow3_self`: derivative `3*dx*pow2(x)`.
    #[must_use]
    #[inline]
    pub fn pow3(self) -> Self {
        self.scaled(pow3(self.val), 3.0, pow2(self.val))
    }

    /// `pow4_self`: derivative `4*dx*pow3(x)`.
    #[must_use]
    #[inline]
    pub fn pow4(self) -> Self {
        self.scaled(pow4(self.val), 4.0, pow3(self.val))
    }

    /// `pow5_self`: derivative `5*dx*pow4(x)`.
    #[must_use]
    #[inline]
    pub fn pow5(self) -> Self {
        self.scaled(pow5(self.val), 5.0, pow4(self.val))
    }

    /// `pow6_self`: derivative `6*dx*pow5(x)`.
    #[must_use]
    #[inline]
    pub fn pow6(self) -> Self {
        self.scaled(pow6(self.val), 6.0, pow5(self.val))
    }

    /// `pow7_self`: derivative `7*dx*pow6(x)`.
    #[must_use]
    #[inline]
    pub fn pow7(self) -> Self {
        self.scaled(pow7(self.val), 7.0, pow6(self.val))
    }

    /// `pow8_self`: derivative `8*dx*pow7(x)`.
    #[must_use]
    #[inline]
    pub fn pow8(self) -> Self {
        self.scaled(pow8(self.val), 8.0, pow7(self.val))
    }

    /// `pow_self`: $x^{y}$ for a dual exponent.
    ///
    /// The array and one-variable types evaluate
    /// `(dx*y + x*dy*log(x))*pow(x, y - 1)`; the two- and four-variable types
    /// evaluate `q0*(q1*dy + dx*y)` with `q0 = pow(x, y - 1)` and
    /// `q1 = x*log(x)`.
    #[must_use]
    #[inline]
    pub fn pow(self, other: Self) -> Self {
        let x = self.val;
        let y = other.val;
        let q0 = pow(x, y - 1.0);
        let lnx = log(x);
        let mut d1 = [0.0; N];
        if Self::FACTORED {
            let q1 = x * lnx;
            for ((slot, &dx), &dy) in
                d1.iter_mut().zip(&self.d1).zip(&other.d1)
            {
                *slot = q0 * (q1 * dy + dx * y);
            }
        } else {
            for ((slot, &dx), &dy) in
                d1.iter_mut().zip(&self.d1).zip(&other.d1)
            {
                *slot = (dx * y + x * dy * lnx) * q0;
            }
        }
        Self { val: pow(x, y), d1 }
    }

    /// `pow_self_real`: value `pow(x, y)`, derivative `dx*y*pow(x, y - 1)`.
    #[must_use]
    #[inline]
    pub fn powf(self, p: f64) -> Self {
        self.scaled(pow(self.val, p), p, pow(self.val, p - 1.0))
    }

    /// `pow_self_int`: the integer exponent is converted to `real(dp)` and
    /// the expression is that of `pow_self_real`.
    #[must_use]
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        self.powf(f64::from(n))
    }

    /// `pow_real_self`: `q0 = pow(z, x)`, derivative `q0*dx*log(z)`.
    #[must_use]
    #[inline]
    pub fn rpow(self, base: f64) -> Self {
        let q0 = pow(base, self.val);
        self.scaled(q0, q0, log(base))
    }

    /// `max_self`: derivative `dx*Heaviside(x - y) + dy*Heaviside(-x + y)`.
    #[must_use]
    #[inline]
    pub fn max(self, other: Self) -> Self {
        let q0 = heaviside(self.val - other.val);
        let q1 = heaviside(-self.val + other.val);
        self.make_binary_operator(other, self.val.max(other.val), q0, q1)
    }

    /// `max_self_real`: derivative `dx*Heaviside(x - y)`.
    #[must_use]
    #[inline]
    pub fn max_scalar(self, c: f64) -> Self {
        self.make_unary_operator(self.val.max(c), heaviside(self.val - c))
    }

    /// `min_self`: derivative `dx*Heaviside(-x + y) + dy*Heaviside(x - y)`.
    #[must_use]
    #[inline]
    pub fn min(self, other: Self) -> Self {
        let q0 = heaviside(-self.val + other.val);
        let q1 = heaviside(self.val - other.val);
        self.make_binary_operator(other, self.val.min(other.val), q0, q1)
    }

    /// `min_self_real`: derivative `dx*Heaviside(-x + y)`.
    #[must_use]
    #[inline]
    pub fn min_scalar(self, c: f64) -> Self {
        self.make_unary_operator(self.val.min(c), heaviside(-self.val + c))
    }

    /// `dim_self`: `q0 = x - y`, value `-0.5*y + 0.5*x + 0.5*Abs(q0)`,
    /// derivative `-0.5*dy + 0.5*dx + 0.5*(dx - dy)*sgn(q0)`.
    #[must_use]
    #[inline]
    pub fn dim(self, other: Self) -> Self {
        let q0 = self.val - other.val;
        let s = sgn(q0);
        let mut d1 = [0.0; N];
        for ((slot, &dx), &dy) in d1.iter_mut().zip(&self.d1).zip(&other.d1) {
            *slot = -0.5 * dy + 0.5 * dx + 0.5 * (dx - dy) * s;
        }
        Self {
            val: -0.5 * other.val + 0.5 * self.val + 0.5 * q0.abs(),
            d1,
        }
    }

    /// `dim_self_real`: `q0 = x - y`, value `-0.5*y + 0.5*x + 0.5*Abs(q0)`,
    /// derivative `0.5*dx*(sgn(q0) + 1)`.
    #[must_use]
    #[inline]
    pub fn dim_scalar(self, c: f64) -> Self {
        let q0 = self.val - c;
        self.scaled(
            -0.5 * c + 0.5 * self.val + 0.5 * q0.abs(),
            0.5,
            sgn(q0) + 1.0,
        )
    }
}

/// `sgn` in `auto_diff/private/support_functions.f90:48-57`.
#[inline]
fn sgn(x: f64) -> f64 {
    if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        1.0
    }
}

/// `Heaviside` in `auto_diff/private/support_functions.f90:59-68`.
#[inline]
fn heaviside(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x == 0.0 {
        0.5
    } else {
        1.0
    }
}

impl DualOrder1<2> {
    /// Seed the first independent variable (`d1 = [1.0, 0.0]`).
    #[must_use]
    #[inline]
    pub const fn var1(val: f64) -> Self {
        Self {
            val,
            d1: [1.0, 0.0],
        }
    }

    /// Seed the second independent variable (`d1 = [0.0, 1.0]`).
    #[must_use]
    #[inline]
    pub const fn var2(val: f64) -> Self {
        Self {
            val,
            d1: [0.0, 1.0],
        }
    }

    /// First partial derivative (`d1[0]`).
    #[must_use]
    #[inline]
    pub const fn d1val1(self) -> f64 {
        self.d1[0]
    }

    /// Second partial derivative (`d1[1]`).
    #[must_use]
    #[inline]
    pub const fn d1val2(self) -> f64 {
        self.d1[1]
    }
}

impl<const N: usize> serde::Serialize for DualOrder1<N> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeTuple;
        let mut tup = serializer.serialize_tuple(1 + N)?;
        tup.serialize_element(&self.val)?;
        for &d in &self.d1 {
            tup.serialize_element(&d)?;
        }
        tup.end()
    }
}

impl<'de, const N: usize> serde::Deserialize<'de> for DualOrder1<N> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        struct DualVisitor<const M: usize>;

        impl<'de, const M: usize> serde::de::Visitor<'de> for DualVisitor<M> {
            type Value = DualOrder1<M>;

            fn expecting(
                &self,
                formatter: &mut std::fmt::Formatter<'_>,
            ) -> std::fmt::Result {
                write!(formatter, "DualOrder1<{M}> tuple of {} f64s", 1 + M)
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let val = seq.next_element()?.ok_or_else(|| {
                    serde::de::Error::invalid_length(0, &self)
                })?;
                let mut d1 = [0.0_f64; M];
                for (i, slot) in d1.iter_mut().enumerate() {
                    *slot = seq.next_element()?.ok_or_else(|| {
                        serde::de::Error::invalid_length(1 + i, &self)
                    })?;
                }
                Ok(DualOrder1 { val, d1 })
            }
        }

        deserializer.deserialize_tuple(1 + N, DualVisitor::<N>)
    }
}
