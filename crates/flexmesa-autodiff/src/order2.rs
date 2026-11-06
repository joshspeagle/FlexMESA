// SPDX-License-Identifier: LGPL-3.0-or-later
//! Two-variable second-order hyperdual numbers (`AutoDiffReal2VarOrder2`).
//!
//! `AutoDiffReal2VarOrder2` (aliased as `AutoDiffReal2ndOrder`) ports MESA's
//! `auto_diff_real_2var_order2` type used by equation-of-state, opacity, and
//! nuclear reaction tables to propagate exact first and second partial
//! derivatives $(\partial_{1}, \partial_{2}, \partial_{11}, \partial_{12},
//! \partial_{22})$ in a single forward pass.

use flexmesa_math::{heaviside, pow2};

use crate::kernels;

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
    ($name:ident, $kernel:path, $doc:expr) => {
        #[doc = $doc]
        #[must_use]
        #[inline]
        pub fn $name(self) -> Self {
            self.apply_unary($kernel)
        }
    };
    ($name:ident, pown $n:literal, $doc:expr) => {
        #[doc = $doc]
        #[must_use]
        #[inline]
        pub fn $name(self) -> Self {
            let (v, d1, d2) = kernels::unary_pown(self.val, $n);
            self.make_unary_operator(v, d1, d2)
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
        Self {
            val: z_val,
            d1val1: self.d1val1 * z_d1x,
            d1val2: self.d1val2 * z_d1x,
            d2val1: self.d2val1 * z_d1x + z_d2x * pow2(self.d1val1),
            d1val1_d1val2: self.d1val1 * self.d1val2 * z_d2x
                + self.d1val1_d1val2 * z_d1x,
            d2val2: self.d2val2 * z_d1x + z_d2x * pow2(self.d1val2),
        }
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
        let q0 = self.d1val1 * z_d1x_d1y;
        let q1 = self.d1val2 * z_d1x_d1y;
        Self {
            val: z_val,
            d1val1: self.d1val1 * z_d1x + other.d1val1 * z_d1y,
            d1val2: self.d1val2 * z_d1x + other.d1val2 * z_d1y,
            d2val1: 2.0 * q0 * other.d1val1
                + self.d2val1 * z_d1x
                + other.d2val1 * z_d1y
                + z_d2x * pow2(self.d1val1)
                + z_d2y * pow2(other.d1val1),
            d1val1_d1val2: q0 * other.d1val2
                + q1 * other.d1val1
                + self.d1val1 * self.d1val2 * z_d2x
                + self.d1val1_d1val2 * z_d1x
                + other.d1val1 * other.d1val2 * z_d2y
                + other.d1val1_d1val2 * z_d1y,
            d2val2: 2.0 * q1 * other.d1val2
                + self.d2val2 * z_d1x
                + other.d2val2 * z_d1y
                + z_d2x * pow2(self.d1val2)
                + z_d2y * pow2(other.d1val2),
        }
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

    unary_method!(
        sign,
        kernels::unary_sign,
        "Compute $\\operatorname{sgn}(x)$."
    );
    unary_method!(abs, kernels::unary_abs, "Compute $|x|$.");
    unary_method!(sqrt, kernels::unary_sqrt, "Compute $\\sqrt{x}$.");
    unary_method!(
        safe_sqrt,
        kernels::unary_safe_sqrt,
        "Compute $\\sqrt{\\max(x, 0)}$."
    );
    unary_method!(exp, kernels::unary_exp, "Compute $\\exp(x)$.");
    unary_method!(expm1, kernels::unary_expm1, "Compute $e^{x} - 1$.");
    unary_method!(exp10, kernels::unary_exp10, "Compute $10^{x}$.");
    unary_method!(powm1, kernels::unary_powm1, "Compute $x^{-1}$.");
    unary_method!(log, kernels::unary_log, "Compute $\\ln(x)$.");
    unary_method!(
        safe_log,
        kernels::unary_safe_log,
        "Compute $\\operatorname{safe\\_log}(x)$."
    );
    unary_method!(log1p, kernels::unary_log1p, "Compute $\\ln(1 + x)$.");
    unary_method!(log10, kernels::unary_log10, "Compute $\\log_{10}(x)$.");
    unary_method!(
        safe_log10,
        kernels::unary_safe_log10,
        "Compute $\\operatorname{safe\\_log10}(x)$."
    );
    unary_method!(log2, kernels::unary_log2, "Compute $\\log_{2}(x)$.");
    unary_method!(sin, kernels::unary_sin, "Compute $\\sin(x)$.");
    unary_method!(cos, kernels::unary_cos, "Compute $\\cos(x)$.");
    unary_method!(tan, kernels::unary_tan, "Compute $\\tan(x)$.");
    unary_method!(sinpi, kernels::unary_sinpi, "Compute $\\sin(\\pi x)$.");
    unary_method!(cospi, kernels::unary_cospi, "Compute $\\cos(\\pi x)$.");
    unary_method!(tanpi, kernels::unary_tanpi, "Compute $\\tan(\\pi x)$.");
    unary_method!(sinh, kernels::unary_sinh, "Compute $\\sinh(x)$.");
    unary_method!(cosh, kernels::unary_cosh, "Compute $\\cosh(x)$.");
    unary_method!(tanh, kernels::unary_tanh, "Compute $\\tanh(x)$.");
    unary_method!(asin, kernels::unary_asin, "Compute $\\arcsin(x)$.");
    unary_method!(acos, kernels::unary_acos, "Compute $\\arccos(x)$.");
    unary_method!(atan, kernels::unary_atan, "Compute $\\arctan(x)$.");
    unary_method!(
        asinpi,
        kernels::unary_asinpi,
        "Compute $\\arcsin(x) / \\pi$."
    );
    unary_method!(
        acospi,
        kernels::unary_acospi,
        "Compute $\\arccos(x) / \\pi$."
    );
    unary_method!(
        atanpi,
        kernels::unary_atanpi,
        "Compute $\\arctan(x) / \\pi$."
    );
    unary_method!(
        asinh,
        kernels::unary_asinh,
        "Compute $\\operatorname{arsinh}(x)$."
    );
    unary_method!(
        acosh,
        kernels::unary_acosh,
        "Compute $\\operatorname{arcosh}(x)$."
    );
    unary_method!(
        atanh,
        kernels::unary_atanh,
        "Compute $\\operatorname{artanh}(x)$."
    );
    unary_method!(pow2, pown 2, "Compute $x^{2}$.");
    unary_method!(pow3, pown 3, "Compute $x^{3}$.");
    unary_method!(pow4, pown 4, "Compute $x^{4}$.");
    unary_method!(pow5, pown 5, "Compute $x^{5}$.");
    unary_method!(pow6, pown 6, "Compute $x^{6}$.");
    unary_method!(pow7, pown 7, "Compute $x^{7}$.");
    unary_method!(pow8, pown 8, "Compute $x^{8}$.");

    /// Raise `self` to a hyperdual power `other` ($x^{y}$).
    #[must_use]
    #[inline]
    pub fn pow(self, other: Self) -> Self {
        let (v, d1x, d1y, d2x, d1x_d1y, d2y) =
            kernels::binary_pow(self.val, other.val);
        self.make_binary_operator(other, v, d1x, d1y, d2x, d1x_d1y, d2y)
    }

    /// Raise `self` to a constant `f64` exponent $p$.
    #[must_use]
    #[inline]
    pub fn powf(self, p: f64) -> Self {
        let (v, d1, d2) = kernels::unary_pow_scalar(self.val, p);
        self.make_unary_operator(v, d1, d2)
    }

    /// Raise `self` to a constant integer exponent $n$.
    #[must_use]
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        self.powf(f64::from(n))
    }

    /// Raise a constant positive scalar `base` to the hyperdual exponent
    /// `self` ($b^{x}$).
    #[must_use]
    #[inline]
    pub fn rpow(self, base: f64) -> Self {
        let (v, d1, d2) = kernels::unary_rpow_scalar(base, self.val);
        self.make_unary_operator(v, d1, d2)
    }

    /// Compute $\max(x, y)$ using MESA's symmetric Heaviside subgradient.
    #[must_use]
    #[inline]
    pub fn max(self, other: Self) -> Self {
        let q0 = heaviside(self.val - other.val);
        let q1 = heaviside(other.val - self.val);
        self.make_binary_operator(
            other,
            self.val.max(other.val),
            q0,
            q1,
            0.0,
            0.0,
            0.0,
        )
    }

    /// Compute $\max(x, c)$ with a scalar constant $c$.
    #[must_use]
    #[inline]
    pub fn max_scalar(self, c: f64) -> Self {
        self.make_unary_operator(self.val.max(c), heaviside(self.val - c), 0.0)
    }

    /// Compute $\min(x, y)$ using MESA's symmetric Heaviside subgradient.
    #[must_use]
    #[inline]
    pub fn min(self, other: Self) -> Self {
        let q0 = heaviside(other.val - self.val);
        let q1 = heaviside(self.val - other.val);
        self.make_binary_operator(
            other,
            self.val.min(other.val),
            q0,
            q1,
            0.0,
            0.0,
            0.0,
        )
    }

    /// Compute $\min(x, c)$ with a scalar constant $c$.
    #[must_use]
    #[inline]
    pub fn min_scalar(self, c: f64) -> Self {
        self.make_unary_operator(self.val.min(c), heaviside(c - self.val), 0.0)
    }

    /// Compute positive difference $\operatorname{dim}(x, y) = \max(x - y, 0)$.
    #[must_use]
    #[inline]
    pub fn dim(self, other: Self) -> Self {
        (self - other).max_scalar(0.0)
    }

    /// Compute positive difference $\operatorname{dim}(x, c) = \max(x - c, 0)$.
    #[must_use]
    #[inline]
    pub fn dim_scalar(self, c: f64) -> Self {
        (self - c).max_scalar(0.0)
    }
}
