// SPDX-License-Identifier: GPL-3.0-or-later
//! `DLATBS`: scaled solve of a banded triangular system.

use super::dtbsv::dtbsv;
use crate::lapack::aux::{fmax, fmin};
use crate::lapack::blas::{dasum, daxpy, ddot, dscal, idamax};
use crate::lapack::{dlamch, ix, lsame, LapackError};

fn bad(info: i32) -> LapackError {
    LapackError::IllegalArgument {
        routine: "DLATBS",
        info,
    }
}

/// `J = JFIRST, JLAST, JINC` as a vector of 1-based indices.
fn jseq(first: usize, last: usize, inc: isize) -> Vec<usize> {
    if inc > 0 {
        (first..=last).collect()
    } else {
        (last..=first).rev().collect()
    }
}

/// `DLATBS` (`SRC/dlatbs.f:240-807`): solve `A*x = s*b` or `A**T*x = s*b`
/// for an `n`-by-`n` triangular band matrix with `kd` off-diagonals, with
/// the scale factor `s` chosen to avoid overflow. `A(i,j)` is stored at
/// `AB(kd+1+i-j,j)` (upper) or `AB(1+i-j,j)` (lower), `ldab >= kd+1`.
///
/// Returns `scale`. `x` holds `b` on entry and the solution on exit;
/// `cnorm` (length `n`) is the off-diagonal column 1-norm, computed when
/// `normin = 'N'` (`dlatbs.f:322-349`) and reused when `'Y'`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] with upstream's `INFO`
/// (`dlatbs.f:289-309`).
pub fn dlatbs(
    uplo: u8,
    trans: u8,
    diag: u8,
    normin: u8,
    n: usize,
    kd: usize,
    ab: &[f64],
    ldab: usize,
    x: &mut [f64],
    cnorm: &mut [f64],
) -> Result<f64, LapackError> {
    let upper = lsame(uplo, b'U');
    let notran = lsame(trans, b'N');
    let nounit = lsame(diag, b'N');
    // dlatbs.f:289-309
    if !upper && !lsame(uplo, b'L') {
        return Err(bad(-1));
    } else if !notran && !lsame(trans, b'T') && !lsame(trans, b'C') {
        return Err(bad(-2));
    } else if !nounit && !lsame(diag, b'U') {
        return Err(bad(-3));
    } else if !lsame(normin, b'Y') && !lsame(normin, b'N') {
        return Err(bad(-4));
    } else if ldab < kd + 1 {
        return Err(bad(-8));
    }
    // dlatbs.f:313-315
    let mut scale = 1.0_f64;
    if n == 0 {
        return Ok(scale);
    }
    // dlatbs.f:319-320
    let smlnum = dlamch::SFMIN / dlamch::PREC;
    let bignum = 1.0 / smlnum;

    if lsame(normin, b'N') {
        // Compute the 1-norm of each column, not including the diagonal
        // (dlatbs.f:322-349).
        if upper {
            // dlatbs.f:330-333
            for j in 1..=n {
                let jlen = kd.min(j - 1);
                cnorm[j - 1] =
                    dasum(jlen, &ab[ix(kd + 1 - jlen, j, ldab)..], 1);
            }
        } else {
            // dlatbs.f:338-346
            for j in 1..=n {
                let jlen = kd.min(n - j);
                if jlen > 0 {
                    cnorm[j - 1] = dasum(jlen, &ab[ix(2, j, ldab)..], 1);
                } else {
                    cnorm[j - 1] = 0.0;
                }
            }
        }
    }

    // Scale the column norms by TSCAL if the maximum element in CNORM is
    // greater than BIGNUM (dlatbs.f:352-359).
    let imax = idamax(n, cnorm, 1);
    let tmax = cnorm[imax - 1];
    let tscal;
    if tmax <= bignum {
        tscal = 1.0_f64;
    } else {
        tscal = 1.0 / (smlnum * tmax);
        dscal(n, tscal, cnorm, 1);
    }

    // Bound on the computed solution vector (dlatbs.f:363-366).
    let j = idamax(n, x, 1);
    let mut xmax = x[j - 1].abs();
    let mut xbnd = xmax;
    let mut grow;
    let (jfirst, jlast, jinc, maind);
    if notran {
        // Growth in A * x = b (dlatbs.f:367-441).
        if upper {
            // dlatbs.f:371-375
            jfirst = n;
            jlast = 1;
            jinc = -1_isize;
            maind = kd + 1;
        } else {
            // dlatbs.f:376-381
            jfirst = 1;
            jlast = n;
            jinc = 1_isize;
            maind = 1;
        }
        let js = jseq(jfirst, jlast, jinc);
        if tscal != 1.0 {
            // dlatbs.f:383-386
            grow = 0.0;
        } else if nounit {
            // dlatbs.f:388-420
            grow = 1.0 / fmax(xbnd, smlnum);
            xbnd = grow;
            let mut exited = false;
            for &j in &js {
                // dlatbs.f:401-402
                if grow <= smlnum {
                    exited = true;
                    break;
                }
                // dlatbs.f:405-417
                let tjj = ab[ix(maind, j, ldab)].abs();
                xbnd = fmin(xbnd, fmin(1.0, tjj) * grow);
                if tjj + cnorm[j - 1] >= smlnum {
                    grow *= tjj / (tjj + cnorm[j - 1]);
                } else {
                    grow = 0.0;
                }
            }
            if !exited {
                // dlatbs.f:420
                grow = xbnd;
            }
        } else {
            // dlatbs.f:421-441
            grow = fmin(1.0, 1.0 / fmax(xbnd, smlnum));
            for &j in &js {
                // dlatbs.f:432-433
                if grow <= smlnum {
                    break;
                }
                // dlatbs.f:437
                grow *= 1.0 / (1.0 + cnorm[j - 1]);
            }
        }
    } else {
        // Growth in A**T * x = b (dlatbs.f:442-512).
        if upper {
            // dlatbs.f:446-450
            jfirst = 1;
            jlast = n;
            jinc = 1_isize;
            maind = kd + 1;
        } else {
            // dlatbs.f:451-456
            jfirst = n;
            jlast = 1;
            jinc = -1_isize;
            maind = 1;
        }
        let js = jseq(jfirst, jlast, jinc);
        if tscal != 1.0 {
            // dlatbs.f:458-461
            grow = 0.0;
        } else if nounit {
            // dlatbs.f:463-490
            grow = 1.0 / fmax(xbnd, smlnum);
            xbnd = grow;
            let mut exited = false;
            for &j in &js {
                // dlatbs.f:476-477
                if grow <= smlnum {
                    exited = true;
                    break;
                }
                // dlatbs.f:481-482
                let xj = 1.0 + cnorm[j - 1];
                grow = fmin(grow, xbnd / xj);
                // dlatbs.f:486-488
                let tjj = ab[ix(maind, j, ldab)].abs();
                if xj > tjj {
                    xbnd *= tjj / xj;
                }
            }
            if !exited {
                // dlatbs.f:490
                grow = fmin(grow, xbnd);
            }
        } else {
            // dlatbs.f:491-511
            grow = fmin(1.0, 1.0 / fmax(xbnd, smlnum));
            for &j in &js {
                // dlatbs.f:502-503
                if grow <= smlnum {
                    break;
                }
                // dlatbs.f:507-508
                let xj = 1.0 + cnorm[j - 1];
                grow /= xj;
            }
        }
    }

    if grow * tscal > smlnum {
        // Level 2 BLAS solve (dlatbs.f:514-519).
        dtbsv(uplo, trans, diag, n, kd, ab, ldab, x, 1)?;
    } else {
        // Level 1 BLAS solve, scaling intermediate results
        // (dlatbs.f:520-795).
        if xmax > bignum {
            // dlatbs.f:524-532
            scale = bignum / xmax;
            dscal(n, scale, x, 1);
            xmax = bignum;
        }
        let js = jseq(jfirst, jlast, jinc);
        if notran {
            // Solve A * x = b (dlatbs.f:534-654).
            for &j in &js {
                // dlatbs.f:541
                let mut xj = x[j - 1].abs();
                // dlatbs.f:543-548: skipped for a unit diagonal with
                // TSCAL = 1 (GO TO 100).
                if nounit || tscal != 1.0 {
                    let tjjs = if nounit {
                        ab[ix(maind, j, ldab)] * tscal
                    } else {
                        tscal
                    };
                    let tjj = tjjs.abs();
                    if tjj > smlnum {
                        // dlatbs.f:551-567
                        if tjj < 1.0 && xj > tjj * bignum {
                            let rec = 1.0 / xj;
                            dscal(n, rec, x, 1);
                            scale *= rec;
                            xmax *= rec;
                        }
                        x[j - 1] /= tjjs;
                        xj = x[j - 1].abs();
                    } else if tjj > 0.0 {
                        // dlatbs.f:568-590
                        if xj > tjj * bignum {
                            let mut rec = (tjj * bignum) / xj;
                            if cnorm[j - 1] > 1.0 {
                                rec /= cnorm[j - 1];
                            }
                            dscal(n, rec, x, 1);
                            scale *= rec;
                            xmax *= rec;
                        }
                        x[j - 1] /= tjjs;
                        xj = x[j - 1].abs();
                    } else {
                        // dlatbs.f:591-603
                        for xi in x.iter_mut().take(n) {
                            *xi = 0.0;
                        }
                        x[j - 1] = 1.0;
                        xj = 1.0;
                        scale = 0.0;
                        xmax = 0.0;
                    }
                }
                // dlatbs.f:604-625
                if xj > 1.0 {
                    let mut rec = 1.0 / xj;
                    if cnorm[j - 1] > (bignum - xmax) * rec {
                        rec *= 0.5;
                        dscal(n, rec, x, 1);
                        scale *= rec;
                    }
                } else if xj * cnorm[j - 1] > (bignum - xmax) {
                    dscal(n, 0.5, x, 1);
                    scale *= 0.5;
                }
                if upper {
                    // dlatbs.f:627-639
                    if j > 1 {
                        let jlen = kd.min(j - 1);
                        let alpha = -x[j - 1] * tscal;
                        let (head, _) = x.split_at_mut(j - 1);
                        daxpy(
                            jlen,
                            alpha,
                            &ab[ix(kd + 1 - jlen, j, ldab)..],
                            1,
                            &mut head[j - jlen - 1..],
                            1,
                        );
                        let i = idamax(j - 1, x, 1);
                        xmax = x[i - 1].abs();
                    }
                } else if j < n {
                    // dlatbs.f:640-652
                    let jlen = kd.min(n - j);
                    if jlen > 0 {
                        let alpha = -x[j - 1] * tscal;
                        let (_, tail) = x.split_at_mut(j);
                        daxpy(jlen, alpha, &ab[ix(2, j, ldab)..], 1, tail, 1);
                    }
                    let i = j + idamax(n - j, &x[j..], 1);
                    xmax = x[i - 1].abs();
                }
            }
        } else {
            // Solve A**T * x = b (dlatbs.f:655-794).
            for &j in &js {
                // dlatbs.f:662-668
                let mut xj = x[j - 1].abs();
                let mut uscal = tscal;
                let mut rec = 1.0 / fmax(xmax, 1.0);
                if cnorm[j - 1] > (bignum - xj) * rec {
                    // dlatbs.f:669-691
                    rec *= 0.5;
                    let tjjs = if nounit {
                        ab[ix(maind, j, ldab)] * tscal
                    } else {
                        tscal
                    };
                    let tjj = tjjs.abs();
                    if tjj > 1.0 {
                        rec = fmin(1.0, rec * tjj);
                        uscal /= tjjs;
                    }
                    if rec < 1.0 {
                        dscal(n, rec, x, 1);
                        scale *= rec;
                        xmax *= rec;
                    }
                }
                // dlatbs.f:693-723
                let mut sumj = 0.0_f64;
                if uscal == 1.0 {
                    if upper {
                        let jlen = kd.min(j - 1);
                        sumj = ddot(
                            jlen,
                            &ab[ix(kd + 1 - jlen, j, ldab)..],
                            1,
                            &x[j - jlen - 1..],
                            1,
                        );
                    } else {
                        let jlen = kd.min(n - j);
                        if jlen > 0 {
                            sumj = ddot(
                                jlen,
                                &ab[ix(2, j, ldab)..],
                                1,
                                &x[j..],
                                1,
                            );
                        }
                    }
                } else if upper {
                    let jlen = kd.min(j - 1);
                    for i in 1..=jlen {
                        sumj += (ab[ix(kd + i - jlen, j, ldab)] * uscal)
                            * x[j - jlen - 1 + i - 1];
                    }
                } else {
                    let jlen = kd.min(n - j);
                    for i in 1..=jlen {
                        sumj += (ab[ix(i + 1, j, ldab)] * uscal) * x[j + i - 1];
                    }
                }
                if uscal == tscal {
                    // dlatbs.f:725-785
                    x[j - 1] -= sumj;
                    xj = x[j - 1].abs();
                    // dlatbs.f:732-741: skipped for a unit diagonal with
                    // TSCAL = 1 (GO TO 150).
                    if nounit || tscal != 1.0 {
                        let tjjs = if nounit {
                            ab[ix(maind, j, ldab)] * tscal
                        } else {
                            tscal
                        };
                        let tjj = tjjs.abs();
                        if tjj > smlnum {
                            // dlatbs.f:743-758
                            if tjj < 1.0 && xj > tjj * bignum {
                                let rec = 1.0 / xj;
                                dscal(n, rec, x, 1);
                                scale *= rec;
                                xmax *= rec;
                            }
                            x[j - 1] /= tjjs;
                        } else if tjj > 0.0 {
                            // dlatbs.f:759-772
                            if xj > tjj * bignum {
                                let rec = (tjj * bignum) / xj;
                                dscal(n, rec, x, 1);
                                scale *= rec;
                                xmax *= rec;
                            }
                            x[j - 1] /= tjjs;
                        } else {
                            // dlatbs.f:773-784
                            for xi in x.iter_mut().take(n) {
                                *xi = 0.0;
                            }
                            x[j - 1] = 1.0;
                            scale = 0.0;
                            xmax = 0.0;
                        }
                    }
                } else {
                    // dlatbs.f:786-790
                    let tjjs = if nounit {
                        ab[ix(maind, j, ldab)] * tscal
                    } else {
                        tscal
                    };
                    x[j - 1] = x[j - 1] / tjjs - sumj;
                }
                // dlatbs.f:791-793
                xmax = fmax(xmax, x[j - 1].abs());
            }
        }
        // dlatbs.f:796
        scale /= tscal;
    }

    // Scale the column norms by 1/TSCAL for return (dlatbs.f:799-803).
    if tscal != 1.0 {
        dscal(n, 1.0 / tscal, cnorm, 1);
    }
    Ok(scale)
}
