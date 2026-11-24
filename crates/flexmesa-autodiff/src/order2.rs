// SPDX-License-Identifier: GPL-3.0-or-later
//! Two-variable second-order hyperdual numbers (`AutoDiffReal2VarOrder2`).
//!
//! `AutoDiffReal2VarOrder2` (aliased as `AutoDiffReal2ndOrder`) ports MESA's
//! `auto_diff_real_2var_order2` type used by equation-of-state, opacity, and
//! nuclear reaction tables to propagate exact first and second partial
//! derivatives $(\partial_{1}, \partial_{2}, \partial_{11}, \partial_{12},
//! \partial_{22})$ in a single forward pass.



/// Two-variable second-order forward-mode hyperdual number.
///
/// Stores the primal value `val`, first partial derivatives `d1val1` and
/// `d1val2`, and second partial derivatives `d2val1`
/// ($\partial^{2} / \partial x_{1}^{2}$), `d1val1_d1val2`
/// ($\partial^{2} / (\partial x_{1} \partial x_{2})$), and `d2val2`
/// ($\partial^{2} / \partial x_{2}^{2}$).
///
/// # Examples
/// ```
/// use flexmesa_autodiff::AutoDiffReal2VarOrder2;
///
/// let x = AutoDiffReal2VarOrder2::var1(2.0);
/// let y = AutoDiffReal2VarOrder2::var2(3.0);
/// let z = x * y + x.pow2();
/// assert_eq!(z.val, 10.0);
/// assert_eq!(z.d1val1, 7.0);
/// assert_eq!(z.d1val2, 2.0);
/// assert_eq!(z.d2val1, 2.0);
/// assert_eq!(z.d1val1_d1val2, 1.0);
/// assert_eq!(z.d2val2, 0.0);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct AutoDiffReal2VarOrder2 {
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
}

macro_rules! unary_method {
    ($name:ident, $kernel:ident, $doc:expr) => {
        #[doc = $doc]
        #[must_use]
        #[inline]
        pub fn $name(self) -> Self {
            crate::order2_gen::$kernel(self)
        }
    };
}

impl AutoDiffReal2VarOrder2 {
    /// Construct a hyperdual number from all six components.
    #[must_use]
    #[inline]
    pub const fn new(
        val: f64,
        d1val1: f64,
        d1val2: f64,
        d2val1: f64,
        d1val1_d1val2: f64,
        d2val2: f64,
    ) -> Self {
        Self {
            val,
            d1val1,
            d1val2,
            d2val1,
            d1val1_d1val2,
            d2val2,
        }
    }

    /// Construct a constant hyperdual number with all derivatives set to zero.
    #[must_use]
    #[inline]
    pub const fn constant(val: f64) -> Self {
        Self::new(val, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Seed the first independent variable $x_{1}$ (`d1val1 = 1.0`).
    #[must_use]
    #[inline]
    pub const fn var1(val: f64) -> Self {
        Self::new(val, 1.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Seed the second independent variable $x_{2}$ (`d1val2 = 1.0`).
    #[must_use]
    #[inline]
    pub const fn var2(val: f64) -> Self {
        Self::new(val, 0.0, 1.0, 0.0, 0.0, 0.0)
    }

    /// Apply the second-order unary chain rule matching MESA's
    /// `make_unary_operator`.
    #[must_use]
    #[inline]
    pub fn make_unary_operator(
        self,
        z_val: f64,
        z_d1x: f64,
        z_d2x: f64,
    ) -> Self {
        crate::order2_gen::make_unary_operator(self, z_val, z_d1x, z_d2x)
    }

    /// Apply the second-order binary chain rule matching MESA's
    /// `make_binary_operator`.
    #[must_use]
    #[inline]
    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    pub fn make_binary_operator(
        self,
        other: Self,
        z_val: f64,
        z_d1x: f64,
        z_d1y: f64,
        z_d2x: f64,
        z_d1x_d1y: f64,
        z_d2y: f64,
    ) -> Self {
        crate::order2_gen::make_binary_operator(self, other, z_val, z_d1x, z_d1y, z_d2x, z_d1x_d1y, z_d2y)
    }

    /// Propagate a scalar derivative kernel `(f(x), f'(x), f''(x))` through
    /// `self`.
    #[must_use]
    #[inline]
    pub fn apply_unary(self, kernel: fn(f64) -> (f64, f64, f64)) -> Self {
        let (z_val, z_d1x, z_d2x) = kernel(self.val);
        self.make_unary_operator(z_val, z_d1x, z_d2x)
    }

    /// Extract the first partial derivative with respect to variable 1 as a
    /// hyperdual.
    #[must_use]
    #[inline]
    pub const fn differentiate_1(self) -> Self {
        Self::new(self.d1val1, self.d2val1, self.d1val1_d1val2, 0.0, 0.0, 0.0)
    }

    /// Extract the first partial derivative with respect to variable 2 as a
    /// hyperdual.
    #[must_use]
    #[inline]
    pub const fn differentiate_2(self) -> Self {
        Self::new(self.d1val2, self.d1val1_d1val2, self.d2val2, 0.0, 0.0, 0.0)
    }

    unary_method!(sign, sign_self,
        "Compute $\\operatorname{sgn}(x)$."
    );
    unary_method!(abs, abs_self, "Compute $|x|$.");
    unary_method!(sqrt, sqrt_self, "Compute $\\sqrt{x}$.");
    unary_method!(safe_sqrt, safe_sqrt_self,
        "Compute $\\sqrt{\\max(x, 0)}$."
    );
    unary_method!(exp, exp_self, "Compute $\\exp(x)$.");
    unary_method!(expm1, expm1_self, "Compute $e^{x} - 1$.");
    unary_method!(exp10, exp10_self, "Compute $10^{x}$.");
    unary_method!(powm1, powm1_self, "Compute $x^{-1}$.");
    unary_method!(log, log_self, "Compute $\\ln(x)$.");
    unary_method!(safe_log, safe_log_self,
        "Compute $\\operatorname{safe\\_log}(x)$."
    );
    unary_method!(log1p, log1p_self, "Compute $\\ln(1 + x)$.");
    unary_method!(log10, log10_self, "Compute $\\log_{10}(x)$.");
    unary_method!(safe_log10, safe_log10_self,
        "Compute $\\operatorname{safe\\_log10}(x)$."
    );
    unary_method!(log2, log2_self, "Compute $\\log_{2}(x)$.");
    unary_method!(sin, sin_self, "Compute $\\sin(x)$.");
    unary_method!(cos, cos_self, "Compute $\\cos(x)$.");
    unary_method!(tan, tan_self, "Compute $\\tan(x)$.");
    unary_method!(sinpi, sinpi_self, "Compute $\\sin(\\pi x)$.");
    unary_method!(cospi, cospi_self, "Compute $\\cos(\\pi x)$.");
    unary_method!(tanpi, tanpi_self, "Compute $\\tan(\\pi x)$.");
    unary_method!(sinh, sinh_self, "Compute $\\sinh(x)$.");
    unary_method!(cosh, cosh_self, "Compute $\\cosh(x)$.");
    unary_method!(tanh, tanh_self, "Compute $\\tanh(x)$.");
    unary_method!(asin, asin_self, "Compute $\\arcsin(x)$.");
    unary_method!(acos, acos_self, "Compute $\\arccos(x)$.");
    unary_method!(atan, atan_self, "Compute $\\arctan(x)$.");
    unary_method!(asinpi, asinpi_self,
        "Compute $\\arcsin(x) / \\pi$."
    );
    unary_method!(acospi, acospi_self,
        "Compute $\\arccos(x) / \\pi$."
    );
    unary_method!(atanpi, atanpi_self,
        "Compute $\\arctan(x) / \\pi$."
    );
    unary_method!(asinh, asinh_self,
        "Compute $\\operatorname{arsinh}(x)$."
    );
    unary_method!(acosh, acosh_self,
        "Compute $\\operatorname{arcosh}(x)$."
    );
    unary_method!(atanh, atanh_self,
        "Compute $\\operatorname{artanh}(x)$."
    );
}
