// SPDX-License-Identifier: LGPL-3.0-or-later
//! Two-variable third-order hyperdual numbers (`AutoDiffReal2VarOrder3`).
//!
//! `AutoDiffReal2VarOrder3` ports MESA's `auto_diff_real_2var_order3` type used
//! by the `Skye` equation of state (`eos/private/skye*.f90`) to propagate exact
//! first, second, and third partial derivatives of the specific Helmholtz free
//! energy $F(T, \rho)$ in a single forward pass.


#[allow(clippy::wildcard_imports)]
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
    ($name:ident, $kernel:ident, $doc:expr) => {
        #[doc = $doc]
        #[must_use]
        #[inline]
        pub fn $name(self) -> Self {
            crate::order3_gen::$kernel(self)
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
        crate::order3_gen::make_unary_operator(self, z_val, z_d1x, z_d2x, z_d3x)
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
        crate::order3_gen::make_binary_operator(self, other, z_val, z_d1x, z_d1y, z_d2x, z_d1x_d1y, z_d2y, z_d3x, z_d2x_d1y, z_d1x_d2y, z_d3y)
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

    unary3_method!(sign, sign_self, "Compute $\\operatorname{sgn}(x)$.");
    unary3_method!(abs, abs_self, "Compute $|x|$.");
    unary3_method!(sqrt, sqrt_self, "Compute $\\sqrt{x}$.");
    unary3_method!(safe_sqrt, safe_sqrt_self,
        "Compute $\\sqrt{\\max(x, 0)}$."
    );
    unary3_method!(exp, exp_self, "Compute $\\exp(x)$.");
    unary3_method!(expm1, expm1_self, "Compute $e^{x} - 1$.");
    unary3_method!(exp10, exp10_self, "Compute $10^{x}$.");
    unary3_method!(powm1, powm1_self, "Compute $x^{-1}$.");
    unary3_method!(log, log_self, "Compute $\\ln(x)$.");
    unary3_method!(safe_log, safe_log_self,
        "Compute $\\operatorname{safe\\_log}(x)$."
    );
    unary3_method!(log1p, log1p_self, "Compute $\\ln(1 + x)$.");
    unary3_method!(log10, log10_self, "Compute $\\log_{10}(x)$.");
    unary3_method!(safe_log10, safe_log10_self,
        "Compute $\\operatorname{safe\\_log10}(x)$."
    );
    unary3_method!(log2, log2_self, "Compute $\\log_{2}(x)$.");
    unary3_method!(sin, sin_self, "Compute $\\sin(x)$.");
    unary3_method!(cos, cos_self, "Compute $\\cos(x)$.");
    unary3_method!(tan, tan_self, "Compute $\\tan(x)$.");
    unary3_method!(sinpi, sinpi_self, "Compute $\\sin(\\pi x)$.");
    unary3_method!(cospi, cospi_self, "Compute $\\cos(\\pi x)$.");
    unary3_method!(tanpi, tanpi_self, "Compute $\\tan(\\pi x)$.");
    unary3_method!(sinh, sinh_self, "Compute $\\sinh(x)$.");
    unary3_method!(cosh, cosh_self, "Compute $\\cosh(x)$.");
    unary3_method!(tanh, tanh_self, "Compute $\\tanh(x)$.");
    unary3_method!(asin, asin_self, "Compute $\\arcsin(x)$.");
    unary3_method!(acos, acos_self, "Compute $\\arccos(x)$.");
    unary3_method!(atan, atan_self, "Compute $\\arctan(x)$.");
    unary3_method!(asinpi, asinpi_self, "Compute $\\arcsin(x) / \\pi$.");
    unary3_method!(acospi, acospi_self, "Compute $\\arccos(x) / \\pi$.");
    unary3_method!(atanpi, atanpi_self, "Compute $\\arctan(x) / \\pi$.");
    unary3_method!(asinh, asinh_self, "Compute $\\operatorname{arsinh}(x)$.");
    unary3_method!(acosh, acosh_self, "Compute $\\operatorname{arcosh}(x)$.");
    unary3_method!(atanh, atanh_self, "Compute $\\operatorname{artanh}(x)$.");
}
