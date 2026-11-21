// SPDX-License-Identifier: LGPL-3.0-or-later
//! Neumaier compensated summation in `real(qp)` (`accurate_real` and
//! `neumaier_sum` in `num/public/accurate_sum.f90`).
//!
//! Upstream uses this type where sums of quantities of very different size
//! must not lose the small terms, for example the mass bookkeeping in
//! `star/private/mass_utils.f90` and `star/private/eps_mdot.f90`. An
//! [`AccurateReal`] holds a running sum and a compensator, both `real(qp)`
//! ([`flexmesa_math::Quad`], `DECISIONS #46`); its value is their sum.
//!
//! Each operator below is a line-for-line translation of the upstream
//! procedure named in its documentation, including the order in which the
//! two parts are folded in. Upstream's overloads take `real(qp)` scalars;
//! a `real(dp)` operand is converted with `Quad::from` at the call site, as
//! Fortran does on assignment.

use std::ops::{Add, Div, Mul, Sub};

use flexmesa_math::Quad;

/// `type accurate_real` (`num/public/accurate_sum.f90:13-21`).
#[derive(Clone, Copy, Debug, Default)]
pub struct AccurateReal {
    /// Running sum.
    pub sum: Quad,
    /// Accumulated rounding error of the running sum.
    pub compensator: Quad,
}

/// One step of Neumaier's algorithm (`neumaier_sum`,
/// `num/public/accurate_sum.f90:80-95`).
#[inline]
pub fn neumaier_sum_qp(sum: &mut Quad, compensator: &mut Quad, summand: Quad) {
    // Upstream's left-to-right order adds the compensator before the error
    // term, which loses it when large terms cancel. Reproduced as written;
    // see deferred issue #314.
    let provisional = *sum + summand;
    if sum.abs() >= summand.abs() {
        *compensator = *compensator + (*sum - provisional) + summand;
    } else {
        *compensator = *compensator + (summand - provisional) + *sum;
    }
    *sum = provisional;
}

impl AccurateReal {
    /// Zero sum and zero compensator.
    pub const ZERO: Self = Self {
        sum: Quad::ZERO,
        compensator: Quad::ZERO,
    };

    /// `value` (`accurate_sum.f90:63-69`): `sum + compensator`.
    #[inline]
    #[must_use]
    pub fn value(self) -> Quad {
        self.sum + self.compensator
    }

    /// `set_accurate_real_num` (`accurate_sum.f90`): assignment from a
    /// `real(qp)` number, with a zero compensator.
    #[inline]
    #[must_use]
    pub const fn from_quad(new: Quad) -> Self {
        Self {
            sum: new,
            compensator: Quad::ZERO,
        }
    }

    /// `sub_accurate_real_num_rev` (`accurate_sum.f90`): `op1 - self` for a
    /// `real(qp)` `op1`.
    #[inline]
    #[must_use]
    pub fn rsub(self, op1: Quad) -> Self {
        let mut ret = Self {
            sum: op1,
            compensator: -self.compensator,
        };
        neumaier_sum_qp(&mut ret.sum, &mut ret.compensator, -self.sum);
        ret
    }

    /// `div_real_acc` (`accurate_sum.f90`): `op1 / self` for a `real(qp)`
    /// `op1`; the result has a zero compensator.
    #[inline]
    #[must_use]
    pub fn rdiv(self, op1: Quad) -> Self {
        Self::from_quad(op1 / self.value())
    }

    /// `acc_less_than_num` (`accurate_sum.f90`): `self%value() < num`.
    #[inline]
    #[must_use]
    pub fn lt(self, num: Quad) -> bool {
        self.value() < num
    }

    /// `acc_greater_than_num` (`accurate_sum.f90`): `self%value() > num`.
    #[inline]
    #[must_use]
    pub fn gt(self, num: Quad) -> bool {
        self.value() > num
    }
}

impl From<Quad> for AccurateReal {
    #[inline]
    fn from(new: Quad) -> Self {
        Self::from_quad(new)
    }
}

/// `add_accurate_real`: fold in `op2%sum`, then `op2%compensator`.
impl Add for AccurateReal {
    type Output = Self;
    #[inline]
    fn add(self, op2: Self) -> Self {
        let mut ret = self;
        neumaier_sum_qp(&mut ret.sum, &mut ret.compensator, op2.sum);
        neumaier_sum_qp(&mut ret.sum, &mut ret.compensator, op2.compensator);
        ret
    }
}

/// `add_accurate_real_num`; `add_accurate_real_num_rev` (number first) gives
/// the same result, since it also starts from the `accurate_real` operand.
impl Add<Quad> for AccurateReal {
    type Output = Self;
    #[inline]
    fn add(self, op2: Quad) -> Self {
        let mut ret = self;
        neumaier_sum_qp(&mut ret.sum, &mut ret.compensator, op2);
        ret
    }
}

/// `sub_accurate_real`: fold in `-op2%sum`, then `-op2%compensator`.
impl Sub for AccurateReal {
    type Output = Self;
    #[inline]
    fn sub(self, op2: Self) -> Self {
        let mut ret = self;
        neumaier_sum_qp(&mut ret.sum, &mut ret.compensator, -op2.sum);
        neumaier_sum_qp(&mut ret.sum, &mut ret.compensator, -op2.compensator);
        ret
    }
}

/// `sub_accurate_real_num`.
impl Sub<Quad> for AccurateReal {
    type Output = Self;
    #[inline]
    fn sub(self, op2: Quad) -> Self {
        let mut ret = self;
        neumaier_sum_qp(&mut ret.sum, &mut ret.compensator, -op2);
        ret
    }
}

/// `mult_acc_acc`.
impl Mul for AccurateReal {
    type Output = Self;
    #[inline]
    fn mul(self, op2: Self) -> Self {
        let mut ret = Self {
            sum: self.sum * op2.sum,
            compensator: self.compensator * op2.sum,
        };
        neumaier_sum_qp(
            &mut ret.sum,
            &mut ret.compensator,
            self.sum * op2.compensator,
        );
        neumaier_sum_qp(
            &mut ret.sum,
            &mut ret.compensator,
            self.compensator * op2.compensator,
        );
        ret
    }
}

/// `mult_acc_real`; `mult_real_acc` (number first) multiplies the same two
/// parts by the same number.
impl Mul<Quad> for AccurateReal {
    type Output = Self;
    #[inline]
    fn mul(self, op2: Quad) -> Self {
        Self {
            sum: self.sum * op2,
            compensator: self.compensator * op2,
        }
    }
}

/// `div_acc_acc`: the quotient of the two values, with a zero compensator.
impl Div for AccurateReal {
    type Output = Self;
    #[inline]
    fn div(self, op2: Self) -> Self {
        Self::from_quad(self.value() / op2.value())
    }
}

/// `div_acc_real`: both parts divided by the number.
impl Div<Quad> for AccurateReal {
    type Output = Self;
    #[inline]
    fn div(self, op2: Quad) -> Self {
        Self {
            sum: self.sum / op2,
            compensator: self.compensator / op2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(x: f64) -> Quad {
        Quad::from(x)
    }

    #[test]
    fn test_neumaier_compensator_holds_terms_the_sum_drops() {
        // 1e40 needs 133 bits, so adding 1 to it in binary128 (113 bits)
        // changes nothing. The compensator keeps what the sum dropped.
        let mut acc = AccurateReal::ZERO;
        for t in [q(1.0e40), q(1.0), q(1.0)] {
            acc = acc + t;
        }
        assert_eq!(acc.sum, q(1.0e40));
        assert_eq!(acc.compensator, q(2.0));
        // The free function used by `mass_utils.f90` does the same.
        let (mut sum, mut comp) = (Quad::ZERO, Quad::ZERO);
        for t in [q(1.0e40), q(1.0), q(1.0)] {
            neumaier_sum_qp(&mut sum, &mut comp, t);
        }
        assert_eq!((sum, comp), (q(1.0e40), q(2.0)));
    }

    #[test]
    fn test_neumaier_reproduces_upstream_evaluation_order() {
        // Upstream writes `compensator + (sum - provisional) + summand`,
        // which Fortran evaluates left to right. When the large terms cancel,
        // the compensator is first added to the large `sum - provisional` and
        // is lost, where Neumaier's own order `(sum - provisional) + summand`
        // would keep it. This port follows upstream, so 1e40 + 1 + 1 - 1e40
        // gives 0 here and not 2.
        let mut acc = AccurateReal::ZERO;
        for t in [q(1.0e40), q(1.0), q(1.0), q(-1.0e40)] {
            acc = acc + t;
        }
        assert_eq!(acc.value(), Quad::ZERO);
    }

    #[test]
    fn test_neumaier_branch_on_magnitude_matches_upstream_formula() {
        // |sum| >= |summand|: compensator += (sum - provisional) + summand.
        let (mut s, mut c) = (q(1.0e40), q(0.5));
        neumaier_sum_qp(&mut s, &mut c, q(3.0));
        let prov = q(1.0e40) + q(3.0);
        assert_eq!(s.to_bits(), prov.to_bits());
        assert_eq!(
            c.to_bits(),
            (q(0.5) + (q(1.0e40) - prov) + q(3.0)).to_bits()
        );
        // |sum| < |summand|: compensator += (summand - provisional) + sum.
        let (mut s, mut c) = (q(3.0), q(0.5));
        neumaier_sum_qp(&mut s, &mut c, q(1.0e40));
        let prov = q(3.0) + q(1.0e40);
        assert_eq!(
            c.to_bits(),
            (q(0.5) + (q(1.0e40) - prov) + q(3.0)).to_bits()
        );
        assert_eq!(s.to_bits(), prov.to_bits());
    }

    #[test]
    fn test_operators_follow_the_upstream_procedures() {
        let a = AccurateReal {
            sum: q(1.0e30),
            compensator: q(-7.0),
        };
        let b = AccurateReal {
            sum: q(3.0),
            compensator: q(1.0e-20),
        };
        let x = q(1.25);

        // Subtraction is addition of the negated parts, in the same order.
        let neg_b = AccurateReal {
            sum: -b.sum,
            compensator: -b.compensator,
        };
        let (d1, d2) = (a - b, a + neg_b);
        assert_eq!(d1.sum.to_bits(), d2.sum.to_bits());
        assert_eq!(d1.compensator.to_bits(), d2.compensator.to_bits());
        assert_eq!((a - x).value(), (a + (-x)).value());

        // `sub_accurate_real_num_rev`: sum = op1, compensator = -c, then
        // fold in -sum.
        let r = b.rsub(x);
        let (mut s, mut c) = (x, -b.compensator);
        neumaier_sum_qp(&mut s, &mut c, -b.sum);
        assert_eq!(r.sum.to_bits(), s.to_bits());
        assert_eq!(r.compensator.to_bits(), c.to_bits());

        // Scalar product and quotient act on both parts.
        let m = a * x;
        assert_eq!(m.sum.to_bits(), (a.sum * x).to_bits());
        assert_eq!(m.compensator.to_bits(), (a.compensator * x).to_bits());
        let d = a / x;
        assert_eq!(d.sum.to_bits(), (a.sum / x).to_bits());
        assert_eq!(d.compensator.to_bits(), (a.compensator / x).to_bits());

        // Quotients involving an `accurate_real` divisor drop the
        // compensator.
        let qq = a / b;
        assert_eq!(qq.sum.to_bits(), (a.value() / b.value()).to_bits());
        assert_eq!(qq.compensator, Quad::ZERO);
        let rq = b.rdiv(x);
        assert_eq!(rq.sum.to_bits(), (x / b.value()).to_bits());
        assert_eq!(rq.compensator, Quad::ZERO);

        // `mult_acc_acc` keeps all four partial products.
        let p = a * b;
        let (mut s, mut c) = (a.sum * b.sum, a.compensator * b.sum);
        neumaier_sum_qp(&mut s, &mut c, a.sum * b.compensator);
        neumaier_sum_qp(&mut s, &mut c, a.compensator * b.compensator);
        assert_eq!(p.sum.to_bits(), s.to_bits());
        assert_eq!(p.compensator.to_bits(), c.to_bits());

        // Comparisons use the value.
        assert!(b.gt(q(3.0)) && !b.lt(q(3.0)));
        assert!(AccurateReal::from(q(2.0)).lt(q(2.5)));
    }
}
