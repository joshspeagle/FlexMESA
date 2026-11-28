// SPDX-License-Identifier: GPL-3.0-or-later
//! `DRSCL`: divide a vector by a scalar without overflow.

use crate::lapack::blas::dscal;
use crate::lapack::dlamch;

/// `DRSCL` (`SRC/drscl.f:83-168`): multiply the `n`-vector `sx` (stride
/// `incx`) by `1/sa` without over- or underflow.
///
/// `SMLNUM = DLAMCH('S')` (`drscl.f:126`), `BIGNUM = ONE/SMLNUM`
/// (`drscl.f:127`); the loop at `drscl.f:134-164` pre-multiplies by
/// `SMLNUM` or `BIGNUM` until `CNUM/CDEN` is representable.
pub fn drscl(n: usize, sa: f64, sx: &mut [f64], incx: usize) {
    // drscl.f:121-122 (quick return)
    if n == 0 {
        return;
    }
    let smlnum = dlamch::SFMIN;
    let bignum = 1.0 / smlnum;
    // drscl.f:131-132
    let mut cden = sa;
    let mut cnum = 1.0_f64;
    loop {
        // drscl.f:135-136
        let cden1 = cden * smlnum;
        let cnum1 = cnum / bignum;
        let mul;
        let done;
        if cden1.abs() > cnum.abs() && cnum != 0.0 {
            // drscl.f:137-143
            mul = smlnum;
            done = false;
            cden = cden1;
        } else if cnum1.abs() > cden.abs() {
            // drscl.f:144-150
            mul = bignum;
            done = false;
            cnum = cnum1;
        } else {
            // drscl.f:151-156
            mul = cnum / cden;
            done = true;
        }
        // drscl.f:161, 163-164
        dscal(n, mul, sx, incx);
        if done {
            break;
        }
    }
}
