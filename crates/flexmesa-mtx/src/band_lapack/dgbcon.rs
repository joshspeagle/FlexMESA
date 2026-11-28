// SPDX-License-Identifier: GPL-3.0-or-later
//! `DGBCON`: reciprocal condition number of a general band matrix.

use super::dlatbs::dlatbs;
use super::drscl::drscl;
use crate::lapack::aux::{dlacn2, Lacn2State};
use crate::lapack::blas::{daxpy, ddot, idamax};
use crate::lapack::{dlamch, ix, lsame, LapackError};

fn bad(info: i32) -> LapackError {
    LapackError::IllegalArgument {
        routine: "DGBCON",
        info,
    }
}

/// `DGBCON` (`SRC/dgbcon.f:144-304`): estimate the reciprocal condition
/// number `1 / (norm(A) * norm(inv(A)))` of an `n`-by-`n` band matrix from
/// the `DGBTRF` factors `ab` (`AB(LDAB,*)`, `ldab >= 2*kl+ku+1`) and the
/// 1-based pivot vector `ipiv`; `anorm` is `norm(A)` in the same norm
/// (`norm` is `'1'`/`'O'` or `'I'`).
///
/// `WORK(1:N)`, `WORK(N+1:2N)` and `WORK(2N+1:3N)` of the Fortran are the
/// local vectors `x`, `v` and `cnorm`; `IWORK` and `ISAVE` live in
/// [`Lacn2State`]. `DLACN2` is [`crate::lapack::aux::dlacn2`].
///
/// When a rescaled solve would overflow, upstream leaves `RCOND = 0`
/// (`dgbcon.f:291-292`, then `GO TO 40`); this returns `0.0` then too.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] with upstream's `INFO`
/// (`dgbcon.f:199-212`), and any error from [`dlatbs`].
pub fn dgbcon(
    norm: u8,
    n: usize,
    kl: usize,
    ku: usize,
    ab: &[f64],
    ldab: usize,
    ipiv: &[usize],
    anorm: f64,
) -> Result<f64, LapackError> {
    // Test the input parameters (dgbcon.f:199-212).
    let onenrm = norm == b'1' || lsame(norm, b'O');
    if !onenrm && !lsame(norm, b'I') {
        return Err(bad(-1));
    } else if ldab < 2 * kl + ku + 1 {
        return Err(bad(-6));
    } else if anorm < 0.0 {
        return Err(bad(-8));
    }
    // Quick return if possible (dgbcon.f:214-220).
    let mut rcond = 0.0_f64;
    if n == 0 {
        rcond = 1.0;
        return Ok(rcond);
    } else if anorm == 0.0 {
        return Ok(rcond);
    }
    // dgbcon.f:222
    let smlnum = dlamch::SFMIN;
    // Estimate the norm of inv(A) (dgbcon.f:224-236).
    let mut ainvnm = 0.0_f64;
    let mut normin = b'N';
    let kase1 = if onenrm { 1 } else { 2 };
    let kd = kl + ku + 1;
    let lnoti = kl > 0;
    let mut xw = vec![0.0_f64; n]; // WORK(1:N)
    let mut vw = vec![0.0_f64; n]; // WORK(N+1:2N)
    let mut cnorm = vec![0.0_f64; n]; // WORK(2N+1:3N)
    let mut st = Lacn2State::default(); // KASE, ISAVE, IWORK
    loop {
        // dgbcon.f:237
        dlacn2(n, &mut vw, &mut xw, &mut ainvnm, &mut st);
        if st.kase == 0 {
            break;
        }
        let scale;
        if st.kase == kase1 {
            // Multiply by inv(L) (dgbcon.f:241-255).
            if lnoti {
                for j in 1..n {
                    let lm = kl.min(n - j);
                    let jp = ipiv[j - 1];
                    let t = xw[jp - 1];
                    if jp != j {
                        xw[jp - 1] = xw[j - 1];
                        xw[j - 1] = t;
                    }
                    daxpy(
                        lm,
                        -t,
                        &ab[ix(kd + 1, j, ldab)..],
                        1,
                        &mut xw[j..],
                        1,
                    );
                }
            }
            // Multiply by inv(U) (dgbcon.f:257-260).
            scale = dlatbs(
                b'U',
                b'N',
                b'N',
                normin,
                n,
                kl + ku,
                ab,
                ldab,
                &mut xw,
                &mut cnorm,
            )?;
        } else {
            // Multiply by inv(U**T) (dgbcon.f:262-267).
            scale = dlatbs(
                b'U',
                b'T',
                b'N',
                normin,
                n,
                kl + ku,
                ab,
                ldab,
                &mut xw,
                &mut cnorm,
            )?;
            // Multiply by inv(L**T) (dgbcon.f:269-283).
            if lnoti {
                for j in (1..n).rev() {
                    let lm = kl.min(n - j);
                    xw[j - 1] -=
                        ddot(lm, &ab[ix(kd + 1, j, ldab)..], 1, &xw[j..], 1);
                    let jp = ipiv[j - 1];
                    if jp != j {
                        let t = xw[jp - 1];
                        xw[jp - 1] = xw[j - 1];
                        xw[j - 1] = t;
                    }
                }
            }
        }
        // Divide X by 1/SCALE if doing so will not cause overflow
        // (dgbcon.f:285-295).
        normin = b'Y';
        if scale != 1.0 {
            let ixm = idamax(n, &xw, 1);
            if scale < xw[ixm - 1].abs() * smlnum || scale == 0.0 {
                // GO TO 40: RCOND stays 0 (dgbcon.f:291-292, 303).
                return Ok(rcond);
            }
            drscl(n, scale, &mut xw, 1);
        }
    }
    // Compute the estimate of the reciprocal condition number
    // (dgbcon.f:297-301).
    if ainvnm != 0.0 {
        rcond = (1.0 / ainvnm) / anorm;
    }
    Ok(rcond)
}
