// SPDX-License-Identifier: GPL-3.0-or-later
//! Dual number of the TDC Newton solver (`auto_diff_real_tdc` in
//! `auto_diff/private/auto_diff_real_tdc_module.f90`).
//!
//! The TDC solver iterates on $Z = \ln|Y|$ and has to return a root that still
//! carries the partial derivatives the stellar-structure solver needs.
//! Upstream does this with a number that holds a value, its derivative with
//! respect to $Z$ (`d1val1`), its partials with respect to the structure
//! variables (`d1Array`), and the mixed second partials (`d1val1_d1Array`).
//! The Newton update `Z + (-Q / dQdZ)` then carries structure partials forward
//! from one iterate to the next (`turb/private/tdc.f90:253-345`).
//!
//! Each operation below evaluates the formula of the upstream routine of the
//! same name. Where upstream has a separate routine for a real operand
//! (`mul_self_real`, `div_self_real`, ...), generic code here promotes the real
//! with [`TurbScalar::from_f64`] and uses the routine for two dual operands, so
//! the value is the same and a partial can differ from upstream's in the last
//! place.

use std::array;
use std::ops::{Add, Div, Mul, Neg, Sub};

use flexmesa_autodiff::DualOrder1;
use flexmesa_math::{
    atan, cos, cosh, exp, log, pow, pow2, pow3, pow4, tan, tanh,
};

use crate::tdc_support::TdcInfo;
use crate::TurbScalar;

/// Value, derivative with respect to the Newton variable, `N` structure
/// partials, and the `N` mixed partials (`auto_diff_real_tdc`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoDiffRealTdc<const N: usize> {
    /// Value (`val`).
    pub val: f64,
    /// Derivative with respect to the Newton variable $Z$ (`d1val1`).
    pub d1val1: f64,
    /// Partials with respect to the structure variables (`d1Array`).
    pub d1_array: [f64; N],
    /// Mixed second partials (`d1val1_d1Array`).
    pub d1val1_d1_array: [f64; N],
}

impl<const N: usize> AutoDiffRealTdc<N> {
    /// Sets `d1val1` and leaves every other component as it is, as the
    /// assignment `Z%d1val1 = 1d0` does (`turb/private/tdc.f90:272`, `:343`).
    #[must_use]
    pub const fn with_d1val1(mut self, d1val1: f64) -> Self {
        self.d1val1 = d1val1;
        self
    }

    /// Derivative with respect to the Newton variable as a number of the same
    /// type (`differentiate_auto_diff_real_tdc_1`).
    #[must_use]
    pub const fn differentiate_1(self) -> Self {
        Self {
            val: self.d1val1,
            d1val1: 0.0,
            d1_array: self.d1val1_d1_array,
            d1val1_d1_array: [0.0; N],
        }
    }
}

/// Scalars the TDC solver can lift into [`AutoDiffRealTdc`] and bring back
/// (`convert` and `unconvert` in `turb/private/tdc_support.f90:424-447`).
pub trait TdcLift: TurbScalar {
    /// The lifted type.
    type Tdc: TurbScalar;

    /// Copies the value and structure partials and zeroes the rest (`convert`).
    #[must_use]
    fn convert(self) -> Self::Tdc;

    /// Keeps the value and structure partials and drops the rest
    /// (`unconvert`).
    #[must_use]
    fn unconvert(lifted: Self::Tdc) -> Self;

    /// See [`AutoDiffRealTdc::with_d1val1`].
    #[must_use]
    fn with_d1val1(lifted: Self::Tdc, d1val1: f64) -> Self::Tdc;

    /// See [`AutoDiffRealTdc::differentiate_1`].
    #[must_use]
    fn differentiate_1(lifted: Self::Tdc) -> Self::Tdc;
}

impl TdcLift for f64 {
    type Tdc = AutoDiffRealTdc<0>;

    fn convert(self) -> Self::Tdc {
        AutoDiffRealTdc {
            val: self,
            d1val1: 0.0,
            d1_array: [],
            d1val1_d1_array: [],
        }
    }

    fn unconvert(lifted: Self::Tdc) -> Self {
        lifted.val
    }

    fn with_d1val1(lifted: Self::Tdc, d1val1: f64) -> Self::Tdc {
        lifted.with_d1val1(d1val1)
    }

    fn differentiate_1(lifted: Self::Tdc) -> Self::Tdc {
        lifted.differentiate_1()
    }
}

impl<const N: usize> TdcLift for DualOrder1<N> {
    type Tdc = AutoDiffRealTdc<N>;

    fn convert(self) -> Self::Tdc {
        AutoDiffRealTdc {
            val: self.val,
            d1val1: 0.0,
            d1_array: self.d1,
            d1val1_d1_array: [0.0; N],
        }
    }

    fn unconvert(lifted: Self::Tdc) -> Self {
        Self {
            val: lifted.val,
            d1: lifted.d1_array,
        }
    }

    fn with_d1val1(lifted: Self::Tdc, d1val1: f64) -> Self::Tdc {
        lifted.with_d1val1(d1val1)
    }

    fn differentiate_1(lifted: Self::Tdc) -> Self::Tdc {
        lifted.differentiate_1()
    }
}

/// `powm1` of `math_lib`: the reciprocal.
#[inline]
fn powm1(x: f64) -> f64 {
    1.0 / x
}

impl<const N: usize> Add for AutoDiffRealTdc<N> {
    type Output = Self;

    /// `add_self`.
    fn add(self, y: Self) -> Self {
        Self {
            val: self.val + y.val,
            d1val1: self.d1val1 + y.d1val1,
            d1_array: array::from_fn(|i| self.d1_array[i] + y.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| {
                self.d1val1_d1_array[i] + y.d1val1_d1_array[i]
            }),
        }
    }
}

impl<const N: usize> Sub for AutoDiffRealTdc<N> {
    type Output = Self;

    /// `sub_self`.
    fn sub(self, y: Self) -> Self {
        Self {
            val: self.val - y.val,
            d1val1: self.d1val1 - y.d1val1,
            d1_array: array::from_fn(|i| self.d1_array[i] - y.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| {
                self.d1val1_d1_array[i] - y.d1val1_d1_array[i]
            }),
        }
    }
}

impl<const N: usize> Mul for AutoDiffRealTdc<N> {
    type Output = Self;

    /// `mul_self`.
    fn mul(self, y: Self) -> Self {
        let x = self;
        Self {
            val: x.val * y.val,
            d1val1: x.d1val1 * y.val + x.val * y.d1val1,
            d1_array: array::from_fn(|i| {
                x.d1_array[i] * y.val + x.val * y.d1_array[i]
            }),
            d1val1_d1_array: array::from_fn(|i| {
                x.d1_array[i] * y.d1val1
                    + x.d1val1 * y.d1_array[i]
                    + x.d1val1_d1_array[i] * y.val
                    + x.val * y.d1val1_d1_array[i]
            }),
        }
    }
}

impl<const N: usize> Div for AutoDiffRealTdc<N> {
    type Output = Self;

    /// `div_self`.
    fn div(self, y: Self) -> Self {
        let x = self;
        let q0 = pow2(y.val);
        let q1 = powm1(q0);
        let q2 = x.val * y.d1val1;
        let inv_y3 = powm1(pow3(y.val));
        Self {
            val: x.val * powm1(y.val),
            d1val1: q1 * (-q2 + x.d1val1 * y.val),
            d1_array: array::from_fn(|i| {
                q1 * (x.d1_array[i] * y.val - x.val * y.d1_array[i])
            }),
            d1val1_d1_array: array::from_fn(|i| {
                (2.0 * q2 * y.d1_array[i] + q0 * x.d1val1_d1_array[i]
                    - y.val
                        * (x.d1_array[i] * y.d1val1
                            + x.d1val1 * y.d1_array[i]
                            + x.val * y.d1val1_d1_array[i]))
                    * inv_y3
            }),
        }
    }
}

impl<const N: usize> Neg for AutoDiffRealTdc<N> {
    type Output = Self;

    /// `unary_minus_self`.
    fn neg(self) -> Self {
        Self {
            val: -self.val,
            d1val1: -self.d1val1,
            d1_array: array::from_fn(|i| -self.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| -self.d1val1_d1_array[i]),
        }
    }
}

impl<const N: usize> TurbScalar for AutoDiffRealTdc<N> {
    fn div_ad(self, rhs: Self) -> Self {
        self / rhs
    }

    /// `div_self_real` (`auto_diff/private/auto_diff_real_tdc_module.f90:1342-1352`).
    fn div_real(self, c: f64) -> Self {
        let q0 = powm1(c);
        Self {
            val: q0 * self.val,
            d1val1: q0 * self.d1val1,
            d1_array: array::from_fn(|i| q0 * self.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| q0 * self.d1val1_d1_array[i]),
        }
    }

    /// Evaluates the structure-type subexpressions of `eval_xis`
    /// (`turb/private/tdc_support.f90:537-557`) with the structure type's
    /// own rules, then lifts them with `convert`. Every structure quantity
    /// in `info` was lifted from that type, so only its value and `d1_array`
    /// are nonzero.
    fn star_xi_terms(
        info: &TdcInfo<Self>,
    ) -> Option<crate::tdc_support::StarXiTerms<Self>> {
        use crate::tdc_support::{StarXiTerms, X_ALFAS, X_CEDE};
        use flexmesa_const::BOLTZ_SIGMA;
        let star = |x: Self| DualOrder1::<N> {
            val: x.val,
            d1: x.d1_array,
        };
        let (t, rho, cp, kap, hp, h) = (
            star(info.t),
            star(info.rho),
            star(info.cp),
            star(info.kap),
            star(info.hp),
            star(info.h),
        );
        let x_gammar = 2.0 * 3.0_f64.sqrt();
        let alpha_hp = info.mixing_length_alpha * hp;
        let gammar_div_alfa = (info.tdc_alpha_r * x_gammar) / alpha_hp;
        let s0_pref =
            (info.tdc_alpha_s * X_ALFAS * info.mixing_length_alpha * cp * t)
                / hp;
        let dr0 = (4.0 * BOLTZ_SIGMA * gammar_div_alfa.pow2() * t.pow3())
            / (rho.pow2() * cp * kap);
        let d0 = (info.tdc_alpha_d * X_CEDE) / alpha_hp;
        Some(StarXiTerms {
            cp_t_div_h: (cp * t / h).convert(),
            s0_pref: s0_pref.convert(),
            d0: d0.convert(),
            dr0: dr0.convert(),
        })
    }

    fn from_f64(v: f64) -> Self {
        Self {
            val: v,
            d1val1: 0.0,
            d1_array: [0.0; N],
            d1val1_d1_array: [0.0; N],
        }
    }

    fn val(self) -> f64 {
        self.val
    }

    fn with_val(mut self, new_val: f64) -> Self {
        self.val = new_val;
        self
    }

    /// `abs_self`.
    fn abs(self) -> Self {
        let x = self;
        // `sgn` of `auto_diff/private/support_functions.f90:48-57`.
        let q0 = if x.val < 0.0 {
            -1.0
        } else if x.val == 0.0 {
            0.0
        } else {
            1.0
        };
        Self {
            val: x.val.abs(),
            d1val1: q0 * x.d1val1,
            d1_array: array::from_fn(|i| q0 * x.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| q0 * x.d1val1_d1_array[i]),
        }
    }

    /// `sqrt_self`.
    fn sqrt(self) -> Self {
        let x = self;
        let q0 = x.val.sqrt();
        let q1 = 0.5 * powm1(q0);
        let inv = powm1(pow3(x.val.sqrt()));
        Self {
            val: q0,
            d1val1: q1 * x.d1val1,
            d1_array: array::from_fn(|i| q1 * x.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| {
                0.25 * (2.0 * x.d1val1_d1_array[i] * x.val
                    - x.d1_array[i] * x.d1val1)
                    * inv
            }),
        }
    }

    /// `exp_self`.
    fn exp(self) -> Self {
        let x = self;
        let q0 = exp(x.val);
        Self {
            val: q0,
            d1val1: q0 * x.d1val1,
            d1_array: array::from_fn(|i| q0 * x.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| {
                q0 * (x.d1_array[i] * x.d1val1 + x.d1val1_d1_array[i])
            }),
        }
    }

    /// `log_self`.
    fn log(self) -> Self {
        let x = self;
        let q0 = powm1(x.val);
        let inv = powm1(pow2(x.val));
        Self {
            val: log(x.val),
            d1val1: q0 * x.d1val1,
            d1_array: array::from_fn(|i| q0 * x.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| {
                (-x.d1_array[i] * x.d1val1 + x.d1val1_d1_array[i] * x.val) * inv
            }),
        }
    }

    /// `pow_self_real`.
    fn powf(self, y: f64) -> Self {
        let x = self;
        let q0 = y * pow(x.val, y - 1.0);
        let tail = pow(x.val, -2.0 + y);
        Self {
            val: pow(x.val, y),
            d1val1: q0 * x.d1val1,
            d1_array: array::from_fn(|i| q0 * x.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| {
                let q1 = x.d1_array[i] * x.d1val1;
                y * (q1 * y - q1 + x.d1val1_d1_array[i] * x.val) * tail
            }),
        }
    }

    /// `pow2_self`.
    fn pow2(self) -> Self {
        let x = self;
        let q0 = 2.0 * x.val;
        Self {
            val: pow2(x.val),
            d1val1: q0 * x.d1val1,
            d1_array: array::from_fn(|i| q0 * x.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| {
                2.0 * x.d1_array[i] * x.d1val1 + q0 * x.d1val1_d1_array[i]
            }),
        }
    }

    /// `pow3_self`.
    fn pow3(self) -> Self {
        let x = self;
        let q0 = 3.0 * pow2(x.val);
        Self {
            val: pow3(x.val),
            d1val1: q0 * x.d1val1,
            d1_array: array::from_fn(|i| q0 * x.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| {
                3.0 * x.val
                    * (2.0 * x.d1_array[i] * x.d1val1
                        + x.d1val1_d1_array[i] * x.val)
            }),
        }
    }

    /// `pow4_self`.
    fn pow4(self) -> Self {
        let x = self;
        let q0 = 4.0 * pow3(x.val);
        let sq = pow2(x.val);
        Self {
            val: pow4(x.val),
            d1val1: q0 * x.d1val1,
            d1_array: array::from_fn(|i| q0 * x.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| {
                4.0 * (3.0 * x.d1_array[i] * x.d1val1
                    + x.d1val1_d1_array[i] * x.val)
                    * sq
            }),
        }
    }

    /// `tanh_self`.
    fn tanh(self) -> Self {
        let x = self;
        let q0 = tanh(x.val);
        let q1 = pow2(q0) - 1.0;
        let inv = powm1(pow2(cosh(x.val)));
        Self {
            val: q0,
            d1val1: -q1 * x.d1val1,
            d1_array: array::from_fn(|i| -q1 * x.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| {
                -(2.0 * q0 * x.d1_array[i] * x.d1val1 - x.d1val1_d1_array[i])
                    * inv
            }),
        }
    }

    /// `tan_self`.
    fn tan(self) -> Self {
        let x = self;
        let q0 = tan(x.val);
        let q1 = pow2(q0) + 1.0;
        let inv = powm1(pow2(cos(x.val)));
        Self {
            val: q0,
            d1val1: q1 * x.d1val1,
            d1_array: array::from_fn(|i| q1 * x.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| {
                (2.0 * q0 * x.d1_array[i] * x.d1val1 + x.d1val1_d1_array[i])
                    * inv
            }),
        }
    }

    /// `atan_self`.
    fn atan(self) -> Self {
        let x = self;
        let q0 = pow2(x.val) + 1.0;
        let q1 = powm1(q0);
        let inv = powm1(pow2(q0));
        Self {
            val: atan(x.val),
            d1val1: q1 * x.d1val1,
            d1_array: array::from_fn(|i| q1 * x.d1_array[i]),
            d1val1_d1_array: array::from_fn(|i| {
                (-2.0 * x.d1_array[i] * x.d1val1 * x.val
                    + q0 * x.d1val1_d1_array[i])
                    * inv
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tdc_support::X_CEDE;

    type D = AutoDiffRealTdc<1>;

    /// `D0` and `S0` prefactor use the structure type's `div_real_self` and
    /// `div_self` rules (`d1_array = -dx*z*powm1(pow2(x))`), not the lifted
    /// type's, and carry no Newton-variable partials.
    #[test]
    fn star_xi_terms_use_structure_type_division_rules() {
        let lift = |v: f64, d: f64| D {
            val: v,
            d1val1: 0.0,
            d1_array: [d],
            d1val1_d1_array: [0.0],
        };
        let info = TdcInfo {
            include_mlt_corr_to_tdc: false,
            use_tdc_enthalpy_flux_limiter: true,
            mixing_length_alpha: 1.7,
            tdc_alpha_c: 0.2,
            tdc_alpha_s: 0.9,
            tdc_alpha_d: 0.7,
            tdc_alpha_r: 0.3,
            tdc_alpha_pt: 0.4,
            dt: 1.0e5,
            a0: lift(1.0, 0.0),
            c0: lift(1.0, 0.0),
            l: lift(1.0, 0.0),
            l0: lift(1.0, 0.0),
            gradl: lift(0.3, 0.0),
            grada: lift(0.4, 0.0),
            t: lift(3.1e6, 0.37),
            rho: lift(0.83, 0.11),
            dv: lift(1.0e-3, 0.0),
            cp: lift(2.9e8, 0.57),
            kap: lift(1.3, 0.23),
            hp: lift(7.7e9, 0.61),
            gamma: lift(1.0, 0.0),
            eq_div_w: lift(0.0, 0.0),
            h: lift(5.5e15, 0.19),
        };
        let st = D::star_xi_terms(&info).expect("lifted scalar overrides");
        let alpha_hp = info.mixing_length_alpha * info.hp.val;
        let z = info.tdc_alpha_d * X_CEDE;
        let dalpha_hp = info.mixing_length_alpha * info.hp.d1_array[0];
        let want = -dalpha_hp * z * (1.0 / (alpha_hp * alpha_hp));
        assert_eq!(st.d0.d1_array[0].to_bits(), want.to_bits());
        assert_eq!(st.d0.val.to_bits(), (z * (1.0 / alpha_hp)).to_bits());
        assert_eq!(st.d0.d1val1, 0.0);
        assert_eq!(st.d0.d1val1_d1_array[0], 0.0);
    }

    /// `x = a + b z + c p + d z p` at `z = p = 0`.
    fn var(a: f64, b: f64, c: f64, d: f64) -> D {
        D {
            val: a,
            d1val1: b,
            d1_array: [c],
            d1val1_d1_array: [d],
        }
    }

    fn close(got: f64, want: f64) {
        let scale = want.abs().max(1.0);
        assert!(
            (got - want).abs() <= 2.0e-6 * scale,
            "got {got:e}, want {want:e}"
        );
    }

    /// Checks a unary operation against the chain rule, with the first and
    /// second derivatives of the scalar function taken by central
    /// differences: `d/dz = g' b`, `d/dp = g' c`, mixed `= g'' b c + g' d`.
    fn check_unary(op: fn(D) -> D, g: fn(f64) -> f64, a: f64) {
        let (b, c, d) = (0.7, -1.3, 0.45);
        let h = 1.0e-4 * a.abs().max(1.0);
        let g1 = (g(a + h) - g(a - h)) / (2.0 * h);
        let g2 = (g(a + h) - 2.0 * g(a) + g(a - h)) / (h * h);
        let out = op(var(a, b, c, d));
        close(out.val, g(a));
        close(out.d1val1, g1 * b);
        close(out.d1_array[0], g1 * c);
        close(out.d1val1_d1_array[0], g2 * b * c + g1 * d);
    }

    #[test]
    fn unary_operations_follow_the_chain_rule_to_second_order() {
        check_unary(TurbScalar::exp, exp, 0.3);
        check_unary(TurbScalar::log, log, 1.7);
        check_unary(TurbScalar::sqrt, f64::sqrt, 2.3);
        check_unary(TurbScalar::pow2, pow2, -1.4);
        check_unary(TurbScalar::pow3, pow3, 1.2);
        check_unary(TurbScalar::pow4, pow4, 0.9);
        check_unary(TurbScalar::tanh, tanh, 0.6);
        check_unary(TurbScalar::tan, tan, 0.4);
        check_unary(TurbScalar::atan, atan, -0.8);
        check_unary(|x| x.powf(1.0 / 3.0), |x| pow(x, 1.0 / 3.0), 2.9);
        check_unary(TurbScalar::abs, f64::abs, -1.1);
        check_unary(|x| -x, |x| -x, 0.5);
    }

    #[test]
    fn products_and_quotients_carry_the_mixed_partial() {
        // x = 2 + 3 z + 5 p + 7 z p, y = 11 + 13 z + 17 p + 19 z p.
        let x = var(2.0, 3.0, 5.0, 7.0);
        let y = var(11.0, 13.0, 17.0, 19.0);
        let prod = x * y;
        assert_eq!(prod.val, 22.0);
        assert_eq!(prod.d1val1, 3.0 * 11.0 + 2.0 * 13.0);
        assert_eq!(prod.d1_array[0], 5.0 * 11.0 + 2.0 * 17.0);
        // d2(xy)/dz dp = x_zp y + x_z y_p + x_p y_z + x y_zp.
        assert_eq!(
            prod.d1val1_d1_array[0],
            7.0 * 11.0 + 3.0 * 17.0 + 5.0 * 13.0 + 2.0 * 19.0
        );
        // (x / y) * y returns x in every component.
        let back = (x / y) * y;
        close(back.val, x.val);
        close(back.d1val1, x.d1val1);
        close(back.d1_array[0], x.d1_array[0]);
        close(back.d1val1_d1_array[0], x.d1val1_d1_array[0]);
        let sum = x + y - y;
        assert_eq!(sum, x);
    }

    #[test]
    fn differentiate_1_moves_the_newton_derivative_into_the_value() {
        let x = var(2.0, 3.0, 5.0, 7.0);
        let dx = x.differentiate_1();
        assert_eq!(dx, var(3.0, 0.0, 7.0, 0.0));
        assert_eq!(x.with_d1val1(1.0), var(2.0, 1.0, 5.0, 7.0));
    }

    #[test]
    fn convert_and_unconvert_keep_value_and_structure_partials() {
        let s = DualOrder1::<2> {
            val: 4.0,
            d1: [0.5, -0.25],
        };
        let lifted = s.convert();
        assert_eq!(lifted.d1val1, 0.0);
        assert_eq!(lifted.d1val1_d1_array, [0.0, 0.0]);
        assert_eq!(DualOrder1::<2>::unconvert(lifted), s);
        assert_eq!(f64::unconvert(3.5_f64.convert()), 3.5);
    }
}
