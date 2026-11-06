// SPDX-License-Identifier: LGPL-3.0-or-later
//! Two-variable third-order hyperdual numbers (`AutoDiffReal2VarOrder3`).
//!
//! `AutoDiffReal2VarOrder3` ports MESA's `auto_diff_real_2var_order3` type used
//! by the `Skye` equation of state (`eos/private/skye*.f90`) to propagate exact
//! first, second, and third partial derivatives of the specific Helmholtz free
//! energy $F(T, \rho)$ in a single forward pass.

use flexmesa_math::{pow2, pow3};

#[allow(clippy::wildcard_imports)]
use crate::kernels3::*;
use crate::order1::DualOrder1;
use crate::order2::AutoDiffReal2VarOrder2;

/// Two-variable third-order forward-mode hyperdual number.
///
/// Stores the primal value `val`, first partial derivatives (`d1val1`,
/// `d1val2`), second partial derivatives (`d2val1`, `d1val1_d1val2`, `d2val2`),
/// and third partial derivatives (`d3val1`, `d2val1_d1val2`, `d1val1_d2val2`,
/// `d3val2`).
#[derive(Clone, Copy, Debug, Default)]
pub struct AutoDiffReal2VarOrder3 {
    /// Primal function value $f(x_{1}, x_{2})$.
    pub val: f64,
    /// First partial derivative $\partial f / \partial x_{1}$.
    pub d1val1: f64,
    /// First partial derivative $\partial f / \partial x_{2}$.
    pub d1val2: f64,
    /// Second partial derivative $\partial^{2} f / \partial x_{1}^{2}$.
    pub d2val1: f64,
    /// Mixed second partial derivative
    /// $\partial^{2} f / (\partial x_{1} \partial x_{2})$.
    pub d1val1_d1val2: f64,
    /// Second partial derivative $\partial^{2} f / \partial x_{2}^{2}$.
    pub d2val2: f64,
    /// Third partial derivative $\partial^{3} f / \partial x_{1}^{3}$.
    pub d3val1: f64,
    /// Mixed third partial derivative
    /// $\partial^{3} f / (\partial x_{1}^{2} \partial x_{2})$.
    pub d2val1_d1val2: f64,
    /// Mixed third partial derivative
    /// $\partial^{3} f / (\partial x_{1} \partial x_{2}^{2})$.
    pub d1val1_d2val2: f64,
    /// Third partial derivative $\partial^{3} f / \partial x_{2}^{3}$.
    pub d3val2: f64,
}

macro_rules! unary3_method {
    ($name:ident, $kernel:path, $doc:expr) => {
        #[doc = $doc]
        #[must_use]
        #[inline]
        pub fn $name(self) -> Self {
            self.apply_unary($kernel)
        }
    };
}

impl AutoDiffReal2VarOrder3 {
    /// Construct a third-order hyperdual number from all 10 components.
    #[must_use]
    #[inline]
    #[allow(clippy::too_many_arguments, clippy::similar_names)]
    pub const fn new(
        val: f64,
        d1val1: f64,
        d1val2: f64,
        d2val1: f64,
        d1val1_d1val2: f64,
        d2val2: f64,
        d3val1: f64,
        d2val1_d1val2: f64,
        d1val1_d2val2: f64,
        d3val2: f64,
    ) -> Self {
        Self {
            val,
            d1val1,
            d1val2,
            d2val1,
            d1val1_d1val2,
            d2val2,
            d3val1,
            d2val1_d1val2,
            d1val1_d2val2,
            d3val2,
        }
    }

    /// Construct a constant hyperdual number with all partial derivatives zero.
    #[must_use]
    #[inline]
    pub const fn constant(val: f64) -> Self {
        Self::new(val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Seed the first independent variable $x_{1}$ (`d1val1 = 1.0`).
    #[must_use]
    #[inline]
    pub const fn var1(val: f64) -> Self {
        Self::new(val, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Seed the second independent variable $x_{2}$ (`d1val2 = 1.0`).
    #[must_use]
    #[inline]
    pub const fn var2(val: f64) -> Self {
        Self::new(val, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Apply the third-order unary chain rule matching MESA's
    /// `make_unary_operator`.
    #[must_use]
    #[inline]
    pub fn make_unary_operator(
        self,
        z_val: f64,
        z_d1x: f64,
        z_d2x: f64,
        z_d3x: f64,
    ) -> Self {
        let q0 = pow2(self.d1val1);
        let q1 = self.d1val2 * z_d2x;
        let q2 = pow2(self.d1val2);
        let q3 = self.d1val1 * z_d2x;
        let q4 = 2.0 * self.d1val1_d1val2;
        Self {
            val: z_val,
            d1val1: self.d1val1 * z_d1x,
            d1val2: self.d1val2 * z_d1x,
            d2val1: q0 * z_d2x + self.d2val1 * z_d1x,
            d1val1_d1val2: q1 * self.d1val1 + self.d1val1_d1val2 * z_d1x,
            d2val2: q2 * z_d2x + self.d2val2 * z_d1x,
            d3val1: 3.0 * q3 * self.d2val1
                + self.d3val1 * z_d1x
                + z_d3x * pow3(self.d1val1),
            d2val1_d1val2: q0 * self.d1val2 * z_d3x
                + q1 * self.d2val1
                + q3 * q4
                + self.d2val1_d1val2 * z_d1x,
            d1val1_d2val2: q1 * q4
                + q2 * self.d1val1 * z_d3x
                + q3 * self.d2val2
                + self.d1val1_d2val2 * z_d1x,
            d3val2: 3.0 * q1 * self.d2val2
                + self.d3val2 * z_d1x
                + z_d3x * pow3(self.d1val2),
        }
    }

    /// Apply the third-order binary chain rule matching MESA's
    /// `make_binary_operator`.
    #[must_use]
    #[inline]
    #[allow(clippy::too_many_arguments, clippy::similar_names)]
    pub fn make_binary_operator(
        self,
        other: Self,
        z_val: f64,
        z_d1x: f64,
        z_d1y: f64,
        z_d2x: f64,
        z_d1x_d1y: f64,
        z_d2y: f64,
        z_d3x: f64,
        z_d2x_d1y: f64,
        z_d1x_d2y: f64,
        z_d3y: f64,
    ) -> Self {
        let q0 = pow2(self.d1val1);
        let q1 = pow2(other.d1val1);
        let q2 = self.d1val1 * z_d1x_d1y;
        let q3 = 2.0 * q2;
        let q4 = self.d1val2 * z_d2x;
        let q5 = other.d1val2 * z_d1x_d1y;
        let q6 = self.d1val2 * z_d1x_d1y;
        let q7 = other.d1val2 * z_d2y;
        let q8 = pow2(self.d1val2);
        let q9 = pow2(other.d1val2);
        let q10 = 2.0 * q5;
        let q11 = self.d1val1 * z_d2x;
        let q12 = 3.0 * self.d2val1;
        let q13 = 3.0 * other.d2val1;
        let q14 = other.d1val1 * z_d1x_d1y;
        let q15 = other.d1val1 * z_d2y;
        let q16 = q1 * z_d1x_d2y;
        let q17 = q0 * z_d2x_d1y;
        let q18 = 2.0 * self.d1val1_d1val2;
        let q19 = 2.0 * other.d1val1_d1val2;
        let q20 = 2.0 * self.d1val1 * other.d1val1;
        let q21 = self.d1val2 * z_d2x_d1y;
        let q22 = other.d1val2 * z_d1x_d2y;
        let q23 = q9 * z_d1x_d2y;
        let q24 = q8 * z_d2x_d1y;
        let q25 = 3.0 * self.d2val2;
        let q26 = 3.0 * other.d2val2;
        Self {
            val: z_val,
            d1val1: self.d1val1 * z_d1x + other.d1val1 * z_d1y,
            d1val2: self.d1val2 * z_d1x + other.d1val2 * z_d1y,
            d2val1: q0 * z_d2x
                + q1 * z_d2y
                + q3 * other.d1val1
                + self.d2val1 * z_d1x
                + other.d2val1 * z_d1y,
            d1val1_d1val2: (q4 + q5) * self.d1val1
                + (q6 + q7) * other.d1val1
                + self.d1val1_d1val2 * z_d1x
                + other.d1val1_d1val2 * z_d1y,
            d2val2: q10 * self.d1val2
                + q8 * z_d2x
                + q9 * z_d2y
                + self.d2val2 * z_d1x
                + other.d2val2 * z_d1y,
            d3val1: 3.0 * q16 * self.d1val1
                + 3.0 * q17 * other.d1val1
                + q12 * (q11 + q14)
                + q13 * (q15 + q2)
                + self.d3val1 * z_d1x
                + other.d3val1 * z_d1y
                + z_d3x * pow3(self.d1val1)
                + z_d3y * pow3(other.d1val1),
            d2val1_d1val2: q0 * self.d1val2 * z_d3x
                + q1 * other.d1val2 * z_d3y
                + (q11 + q14) * q18
                + q15 * q19
                + q16 * self.d1val2
                + q17 * other.d1val2
                + q20 * (q21 + q22)
                + q3 * other.d1val1_d1val2
                + (q4 + q5) * self.d2val1
                + (q6 + q7) * other.d2val1
                + self.d2val1_d1val2 * z_d1x
                + other.d2val1_d1val2 * z_d1y,
            d1val1_d2val2: 2.0 * q21 * self.d1val1 * other.d1val2
                + 2.0 * q22 * self.d1val2 * other.d1val1
                + q10 * self.d1val1_d1val2
                + (q11 + q14) * self.d2val2
                + (q15 + q2) * other.d2val2
                + q18 * q4
                + q19 * (q6 + q7)
                + q23 * self.d1val1
                + q24 * other.d1val1
                + q8 * self.d1val1 * z_d3x
                + q9 * other.d1val1 * z_d3y
                + self.d1val1_d2val2 * z_d1x
                + other.d1val1_d2val2 * z_d1y,
            d3val2: 3.0 * q23 * self.d1val2
                + 3.0 * q24 * other.d1val2
                + q25 * (q4 + q5)
                + q26 * (q6 + q7)
                + self.d3val2 * z_d1x
                + other.d3val2 * z_d1y
                + z_d3x * pow3(self.d1val2)
                + z_d3y * pow3(other.d1val2),
        }
    }

    /// Substitute `(x, y)` into the bivariate Taylor jet `self` (`make_binop`).
    #[must_use]
    #[inline]
    pub fn compose_binary(self, x: Self, y: Self) -> Self {
        x.make_binary_operator(
            y,
            self.val,
            self.d1val1,
            self.d1val2,
            self.d2val1,
            self.d1val1_d1val2,
            self.d2val2,
            self.d3val1,
            self.d2val1_d1val2,
            self.d1val1_d2val2,
            self.d3val2,
        )
    }

    /// Propagate a scalar derivative kernel `(f, f', f'', f''')` through
    /// `self`.
    #[must_use]
    #[inline]
    pub fn apply_unary(self, kernel: fn(f64) -> (f64, f64, f64, f64)) -> Self {
        let (v, d1, d2, d3) = kernel(self.val);
        self.make_unary_operator(v, d1, d2, d3)
    }

    /// Extract $\partial f / \partial x_{1}$ as an [`AutoDiffReal2VarOrder3`]
    /// with zero third-order terms.
    #[must_use]
    #[inline]
    pub const fn differentiate_1(self) -> Self {
        Self::new(
            self.d1val1,
            self.d2val1,
            self.d1val1_d1val2,
            self.d3val1,
            self.d2val1_d1val2,
            self.d1val1_d2val2,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    }

    /// Extract $\partial f / \partial x_{2}$ as an [`AutoDiffReal2VarOrder3`]
    /// with zero third-order terms.
    #[must_use]
    #[inline]
    pub const fn differentiate_2(self) -> Self {
        Self::new(
            self.d1val2,
            self.d1val1_d1val2,
            self.d2val2,
            self.d2val1_d1val2,
            self.d1val1_d2val2,
            self.d3val2,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    }

    /// Project onto the second-order hyperdual [`AutoDiffReal2VarOrder2`].
    #[must_use]
    #[inline]
    pub const fn to_order2(self) -> AutoDiffReal2VarOrder2 {
        AutoDiffReal2VarOrder2::new(
            self.val,
            self.d1val1,
            self.d1val2,
            self.d2val1,
            self.d1val1_d1val2,
            self.d2val2,
        )
    }

    /// Project onto the first-order dual [`DualOrder1<2>`].
    #[must_use]
    #[inline]
    pub const fn to_order1(self) -> DualOrder1<2> {
        DualOrder1::new(self.val, [self.d1val1, self.d1val2])
    }

    unary3_method!(sign, unary3_sign, "Compute $\\operatorname{sgn}(x)$.");
    unary3_method!(abs, unary3_abs, "Compute $|x|$.");
    unary3_method!(sqrt, unary3_sqrt, "Compute $\\sqrt{x}$.");
    unary3_method!(
        safe_sqrt,
        unary3_safe_sqrt,
        "Compute $\\sqrt{\\max(x, 0)}$."
    );
    unary3_method!(exp, unary3_exp, "Compute $\\exp(x)$.");
    unary3_method!(expm1, unary3_expm1, "Compute $e^{x} - 1$.");
    unary3_method!(exp10, unary3_exp10, "Compute $10^{x}$.");
    unary3_method!(powm1, unary3_powm1, "Compute $x^{-1}$.");
    unary3_method!(log, unary3_log, "Compute $\\ln(x)$.");
    unary3_method!(
        safe_log,
        unary3_safe_log,
        "Compute $\\operatorname{safe\\_log}(x)$."
    );
    unary3_method!(log1p, unary3_log1p, "Compute $\\ln(1 + x)$.");
    unary3_method!(log10, unary3_log10, "Compute $\\log_{10}(x)$.");
    unary3_method!(
        safe_log10,
        unary3_safe_log10,
        "Compute $\\operatorname{safe\\_log10}(x)$."
    );
    unary3_method!(log2, unary3_log2, "Compute $\\log_{2}(x)$.");
    unary3_method!(sin, unary3_sin, "Compute $\\sin(x)$.");
    unary3_method!(cos, unary3_cos, "Compute $\\cos(x)$.");
    unary3_method!(tan, unary3_tan, "Compute $\\tan(x)$.");
    unary3_method!(sinpi, unary3_sinpi, "Compute $\\sin(\\pi x)$.");
    unary3_method!(cospi, unary3_cospi, "Compute $\\cos(\\pi x)$.");
    unary3_method!(tanpi, unary3_tanpi, "Compute $\\tan(\\pi x)$.");
    unary3_method!(sinh, unary3_sinh, "Compute $\\sinh(x)$.");
    unary3_method!(cosh, unary3_cosh, "Compute $\\cosh(x)$.");
    unary3_method!(tanh, unary3_tanh, "Compute $\\tanh(x)$.");
    unary3_method!(asin, unary3_asin, "Compute $\\arcsin(x)$.");
    unary3_method!(acos, unary3_acos, "Compute $\\arccos(x)$.");
    unary3_method!(atan, unary3_atan, "Compute $\\arctan(x)$.");
    unary3_method!(asinpi, unary3_asinpi, "Compute $\\arcsin(x) / \\pi$.");
    unary3_method!(acospi, unary3_acospi, "Compute $\\arccos(x) / \\pi$.");
    unary3_method!(atanpi, unary3_atanpi, "Compute $\\arctan(x) / \\pi$.");
    unary3_method!(asinh, unary3_asinh, "Compute $\\operatorname{arsinh}(x)$.");
    unary3_method!(acosh, unary3_acosh, "Compute $\\operatorname{arcosh}(x)$.");
    unary3_method!(atanh, unary3_atanh, "Compute $\\operatorname{artanh}(x)$.");
}
