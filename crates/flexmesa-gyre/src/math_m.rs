// SPDX-License-Identifier: GPL-3.0-only
//! GYRE `math_m` in its CRMATH variant (`gyre/gyre/src/math/crmath/math_m.fypp`).

use crate::crmath::exp_c;
use crate::fcomplex::Cplx;

/// The constants `init_math` sets (`math_m.fypp:93-106`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MathConstants {
    /// `PI = acos(-1._RD)` (CRMATH `acos_rd`).
    pub pi: f64,
    /// `TWOPI = 2._RD*PI`.
    pub twopi: f64,
    /// `HALFPI = 0.5_RD*PI`.
    pub halfpi: f64,
    /// `GOLDEN_R = 0.5_RD*(1._RD + sqrt(5._RD))`.
    pub golden_r: f64,
    /// `DEG_TO_RAD = PI/180._RD`.
    pub deg_to_rad: f64,
    /// `RAD_TO_DEG = 1._RD/DEG_TO_RAD`.
    pub rad_to_deg: f64,
}

impl MathConstants {
    /// Evaluate the constants in `init_math`'s order.
    #[must_use]
    pub fn init() -> Self {
        let pi = flexmesa_math::acos(-1.0);
        let twopi = 2.0 * pi;
        let halfpi = 0.5 * pi;
        let golden_r = 0.5 * (1.0 + 5.0_f64.sqrt());
        let deg_to_rad = pi / 180.0;
        let rad_to_deg = 1.0 / deg_to_rad;
        Self {
            pi,
            twopi,
            halfpi,
            golden_r,
            deg_to_rad,
            rad_to_deg,
        }
    }
}

/// `pow(x, y)` for real `x`, `y` (`pow_r_r_`, `math_m.fypp:134-148`):
/// `0` if `x == 0`, else `exp(log(x)*y)`.
#[must_use]
#[inline]
pub fn pow_rr(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        flexmesa_math::exp(flexmesa_math::log(x) * y)
    }
}

/// `pow(x, y)` for real `x` and complex `y` (`pow_r_c_`): `0` if `x == 0`,
/// else `exp(log(x)*y)` with the real-times-complex expansion.
#[must_use]
#[inline]
pub fn pow_rc(x: f64, y: Cplx) -> Cplx {
    if x == 0.0 {
        Cplx::new(0.0, 0.0)
    } else {
        exp_c(Cplx::real_mul(flexmesa_math::log(x), y))
    }
}
