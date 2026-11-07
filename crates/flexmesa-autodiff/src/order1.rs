// SPDX-License-Identifier: LGPL-3.0-or-later
//! Const-generic first-order forward-mode dual numbers (`DualOrder1<N>`).
//!
//! `DualOrder1<N>` unifies MESA's `auto_diff_real_1var_order1` ($N = 1$),
//! `auto_diff_real_2var_order1` ($N = 2$), `auto_diff_real_4var_order1`
//! ($N = 4$), and `auto_diff_real_star_order1` ($N = 33$) into a single
//! zero-allocation stack-allocated type.

use flexmesa_math::heaviside;

use crate::kernels;

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
            let (v, d1, _) = kernels::unary_pown(self.val, $n);
            self.make_unary_operator(v, d1)
        }
    };
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

    /// Raise `self` to a dual-number power `other` ($x^{y}$).
    #[must_use]
    #[inline]
    pub fn pow(self, other: Self) -> Self {
        let (v, d1x, d1y, _, _, _) = kernels::binary_pow(self.val, other.val);
        self.make_binary_operator(other, v, d1x, d1y)
    }

    /// Raise `self` to a constant `f64` exponent $p$.
    #[must_use]
    #[inline]
    pub fn powf(self, p: f64) -> Self {
        let (v, d1, _) = kernels::unary_pow_scalar(self.val, p);
        self.make_unary_operator(v, d1)
    }

    /// Raise `self` to a constant integer exponent $n$.
    #[must_use]
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        self.powf(f64::from(n))
    }

    /// Raise a constant positive scalar `base` to the dual exponent `self`
    /// ($b^{x}$).
    #[must_use]
    #[inline]
    pub fn rpow(self, base: f64) -> Self {
        let (v, d1, _) = kernels::unary_rpow_scalar(base, self.val);
        self.make_unary_operator(v, d1)
    }

    /// Compute $\max(x, y)$ using MESA's symmetric Heaviside subgradient.
    #[must_use]
    #[inline]
    pub fn max(self, other: Self) -> Self {
        let q0 = heaviside(self.val - other.val);
        let q1 = heaviside(other.val - self.val);
        self.make_binary_operator(other, self.val.max(other.val), q0, q1)
    }

    /// Compute $\max(x, c)$ with a scalar constant $c$.
    #[must_use]
    #[inline]
    pub fn max_scalar(self, c: f64) -> Self {
        self.make_unary_operator(self.val.max(c), heaviside(self.val - c))
    }

    /// Compute $\min(x, y)$ using MESA's symmetric Heaviside subgradient.
    #[must_use]
    #[inline]
    pub fn min(self, other: Self) -> Self {
        let q0 = heaviside(other.val - self.val);
        let q1 = heaviside(self.val - other.val);
        self.make_binary_operator(other, self.val.min(other.val), q0, q1)
    }

    /// Compute $\min(x, c)$ with a scalar constant $c$.
    #[must_use]
    #[inline]
    pub fn min_scalar(self, c: f64) -> Self {
        self.make_unary_operator(self.val.min(c), heaviside(c - self.val))
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
