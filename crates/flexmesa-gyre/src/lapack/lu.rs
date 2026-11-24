// SPDX-License-Identifier: GPL-3.0-only
//! Reference LAPACK 3.12.0 LU routines (`SRC/dgetrf.f`, `dgetrf2.f`,
//! `dgetrs.f`, `dgecon.f`, `dgerfs.f`, `dgesvx.f`), double precision, as the
//! MESA SDK `26.6.1` builds them (`gfortran -O2 -fPIC`).

use super::aux::{
    dgeequ, dlacn2, dlacpy, dlange, dlantr, dlaqge, dlaswp, dlatrs, fmax,
    Lacn2State,
};
use super::blas::{daxpy, dcopy, dgemm, dgemv, dscal, dtrsm, idamax};
use super::{dlamch, ilaenv_nb, ix, lsame, LapackError};

fn unported(routine: &'static str, lines: &'static str) -> LapackError {
    LapackError::UnportedPath { routine, lines }
}

/// `DGETRF2` (`SRC/dgetrf2.f`): recursive LU factorisation with partial
/// pivoting of the `m x n` matrix `A`. `ipiv` (length `min(m, n)`) receives
/// 1-based pivot rows. Returns `INFO` (`> 0`: `U(info, info)` is exactly 0).
///
/// # Errors
///
/// Propagates BLAS errors (none for valid sizes).
pub fn dgetrf2(
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    ipiv: &mut [usize],
) -> Result<usize, LapackError> {
    let mut info = 0;
    if m == 0 || n == 0 {
        return Ok(0);
    }
    if m == 1 {
        // Use unblocked code for one row case (dgetrf2.f:169-176)
        ipiv[0] = 1;
        if a[0] == 0.0 {
            info = 1;
        }
    } else if n == 1 {
        // One column case (dgetrf2.f:178-213)
        let sfmin = dlamch::SFMIN;
        let i = idamax(m, a, 1);
        ipiv[0] = i;
        if a[i - 1] == 0.0 {
            info = 1;
        } else {
            if i != 1 {
                a.swap(0, i - 1);
            }
            if a[0].abs() >= sfmin {
                let r = 1.0 / a[0];
                dscal(m - 1, r, &mut a[1..], 1);
            } else {
                for k in 1..m {
                    a[k] /= a[0];
                }
            }
        }
    } else {
        // Recursive case (dgetrf2.f:219-262)
        let n1 = m.min(n) / 2;
        let n2 = n - n1;
        let iinfo = dgetrf2(m, n1, a, lda, ipiv)?;
        if info == 0 && iinfo > 0 {
            info = iinfo;
        }
        // [A12; A22] row interchanges
        dlaswp(n2, &mut a[ix(1, n1 + 1, lda)..], lda, 1, n1, ipiv, 1);
        // A12 = L11 \ A12 (L11 copied: it is read only and lies in A)
        let l11 = a[..ix(n1, n1, lda) + 1].to_vec();
        dtrsm(
            b'L',
            b'L',
            b'N',
            b'U',
            n1,
            n2,
            1.0,
            &l11,
            lda,
            &mut a[ix(1, n1 + 1, lda)..],
            lda,
        )?;
        // A22 = A22 - A21*A12 (A21, A12 copied)
        let a21 = a[ix(n1 + 1, 1, lda)..].to_vec();
        let a12 = a[ix(1, n1 + 1, lda)..].to_vec();
        dgemm(
            b'N',
            b'N',
            m - n1,
            n2,
            n1,
            -1.0,
            &a21,
            lda,
            &a12,
            lda,
            1.0,
            &mut a[ix(n1 + 1, n1 + 1, lda)..],
            lda,
        )?;
        // Factor A22
        let iinfo = dgetrf2(
            m - n1,
            n2,
            &mut a[ix(n1 + 1, n1 + 1, lda)..],
            lda,
            &mut ipiv[n1..],
        )?;
        if info == 0 && iinfo > 0 {
            info = iinfo + n1;
        }
        for p in &mut ipiv[n1..m.min(n)] {
            *p += n1;
        }
        // Apply interchanges to A21
        dlaswp(n1, a, lda, n1 + 1, m.min(n), ipiv, 1);
    }
    Ok(info)
}

/// `DGETRF` (`SRC/dgetrf.f`): LU factorisation; unblocked (`DGETRF2`) when
/// `NB = 64 >= min(m, n)`.
///
/// # Errors
///
/// [`LapackError::UnportedPath`] for the blocked path (`min(m, n) > 64`).
pub fn dgetrf(
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    ipiv: &mut [usize],
) -> Result<usize, LapackError> {
    if lda < m.max(1) {
        return Err(LapackError::IllegalArgument {
            routine: "DGETRF",
            info: -4,
        });
    }
    if m == 0 || n == 0 {
        return Ok(0);
    }
    let nb = usize::try_from(ilaenv_nb("DGETRF")).unwrap_or(1);
    if nb <= 1 || nb >= m.min(n) {
        dgetrf2(m, n, a, lda, ipiv)
    } else {
        Err(unported("DGETRF", "174-216"))
    }
}

/// `DGETRS` (`SRC/dgetrs.f`): solve `op(A)*X = B` with the LU factors.
///
/// # Errors
///
/// Propagates BLAS errors.
#[allow(clippy::too_many_arguments)]
pub fn dgetrs(
    trans: u8,
    n: usize,
    nrhs: usize,
    a: &[f64],
    lda: usize,
    ipiv: &[usize],
    b: &mut [f64],
    ldb: usize,
) -> Result<(), LapackError> {
    let notran = lsame(trans, b'N');
    if !notran && !lsame(trans, b'T') && !lsame(trans, b'C') {
        return Err(LapackError::IllegalArgument {
            routine: "DGETRS",
            info: -1,
        });
    }
    if n == 0 || nrhs == 0 {
        return Ok(());
    }
    if notran {
        dlaswp(nrhs, b, ldb, 1, n, ipiv, 1);
        dtrsm(b'L', b'L', b'N', b'U', n, nrhs, 1.0, a, lda, b, ldb)?;
        dtrsm(b'L', b'U', b'N', b'N', n, nrhs, 1.0, a, lda, b, ldb)?;
    } else {
        dtrsm(b'L', b'U', b'T', b'N', n, nrhs, 1.0, a, lda, b, ldb)?;
        dtrsm(b'L', b'L', b'T', b'U', n, nrhs, 1.0, a, lda, b, ldb)?;
        dlaswp(nrhs, b, ldb, 1, n, ipiv, -1);
    }
    Ok(())
}

/// `DGECON` (`SRC/dgecon.f`): reciprocal condition number estimate from
/// the LU factors. Returns `(rcond, info)`.
///
/// # Errors
///
/// [`LapackError::UnportedPath`] when the solves need rescaling
/// (`SCALE != 1`, lines 256-261, which calls `DRSCL`).
pub fn dgecon(
    norm: u8,
    n: usize,
    a: &[f64],
    lda: usize,
    anorm: f64,
) -> Result<(f64, usize), LapackError> {
    let hugeval = dlamch::RMAX;
    let onenrm = norm == b'1' || lsame(norm, b'O');
    if !onenrm && !lsame(norm, b'I') {
        return Err(LapackError::IllegalArgument {
            routine: "DGECON",
            info: -1,
        });
    }
    if anorm < 0.0 {
        return Err(LapackError::IllegalArgument {
            routine: "DGECON",
            info: -5,
        });
    }
    if n == 0 {
        return Ok((1.0, 0));
    } else if anorm == 0.0 {
        return Ok((0.0, 0));
    } else if anorm.is_nan() || anorm > hugeval {
        // dgecon.f: INFO = -5 and RETURN without XERBLA; reported as an error
        // here (fail loudly on a NaN or overflowing norm).
        return Err(LapackError::IllegalArgument {
            routine: "DGECON",
            info: -5,
        });
    }
    let mut ainvnm = 0.0_f64;
    let mut normin = b'N';
    let kase1 = if onenrm { 1 } else { 2 };
    // WORK(1:N) = X, WORK(N+1:2N) = V, WORK(2N+1:3N) = CNORM(L),
    // WORK(3N+1:4N) = CNORM(U)
    let mut xw = vec![0.0_f64; n];
    let mut vw = vec![0.0_f64; n];
    let mut cl = vec![0.0_f64; n];
    let mut cu = vec![0.0_f64; n];
    let mut st = Lacn2State::default();
    loop {
        dlacn2(n, &mut vw, &mut xw, &mut ainvnm, &mut st);
        if st.kase == 0 {
            break;
        }
        let (sl, su);
        if st.kase == kase1 {
            sl = dlatrs(b'L', b'N', b'U', normin, n, a, lda, &mut xw, &mut cl)?;
            su = dlatrs(b'U', b'N', b'N', normin, n, a, lda, &mut xw, &mut cu)?;
        } else {
            su = dlatrs(b'U', b'T', b'N', normin, n, a, lda, &mut xw, &mut cu)?;
            sl = dlatrs(b'L', b'T', b'U', normin, n, a, lda, &mut xw, &mut cl)?;
        }
        let scale = sl * su;
        normin = b'Y';
        if scale != 1.0 {
            return Err(unported("DGECON", "256-261"));
        }
    }
    let mut info = 0;
    if ainvnm == 0.0 {
        return Ok((0.0, 1));
    }
    let rcond = (1.0 / ainvnm) / anorm;
    if rcond.is_nan() || rcond > hugeval {
        info = 1;
    }
    Ok((rcond, info))
}

/// Output of [`dgerfs`].
#[derive(Debug, Clone, PartialEq)]
pub struct Refined {
    /// Forward error bounds, one per right-hand side.
    pub ferr: Vec<f64>,
    /// Componentwise backward errors.
    pub berr: Vec<f64>,
}

/// `DGERFS` (`SRC/dgerfs.f`): iterative refinement and error bounds.
///
/// # Errors
///
/// Propagates BLAS and `DGETRS` errors.
#[allow(
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    clippy::cast_precision_loss
)]
pub fn dgerfs(
    trans: u8,
    n: usize,
    nrhs: usize,
    a: &[f64],
    lda: usize,
    af: &[f64],
    ldaf: usize,
    ipiv: &[usize],
    b: &[f64],
    ldb: usize,
    x: &mut [f64],
    ldx: usize,
) -> Result<Refined, LapackError> {
    const ITMAX: usize = 5;
    let notran = lsame(trans, b'N');
    let mut out = Refined {
        ferr: vec![0.0; nrhs],
        berr: vec![0.0; nrhs],
    };
    if n == 0 || nrhs == 0 {
        return Ok(out);
    }
    let transt = if notran { b'T' } else { b'N' };
    let nz = n + 1;
    let eps = dlamch::EPS;
    let safmin = dlamch::SFMIN;
    let safe1 = nz as f64 * safmin;
    let safe2 = safe1 / eps;
    // WORK(1:N), WORK(N+1:2N), WORK(2N+1:3N)
    let mut w1 = vec![0.0_f64; n];
    let mut w2 = vec![0.0_f64; n];
    let mut w3 = vec![0.0_f64; n];
    for j in 1..=nrhs {
        let mut count = 1;
        let mut lstres = 3.0_f64;
        loop {
            // Residual R = B - op(A)*X (dgerfs.f:298-300)
            dcopy(n, &b[ix(1, j, ldb)..], 1, &mut w2, 1);
            dgemv(
                trans,
                n,
                n,
                -1.0,
                a,
                lda,
                &x[ix(1, j, ldx)..],
                1,
                1.0,
                &mut w2,
                1,
            )?;
            for i in 1..=n {
                w1[i - 1] = b[ix(i, j, ldb)].abs();
            }
            if notran {
                for k in 1..=n {
                    let xk = x[ix(k, j, ldx)].abs();
                    for i in 1..=n {
                        w1[i - 1] += a[ix(i, k, lda)].abs() * xk;
                    }
                }
            } else {
                for k in 1..=n {
                    let mut s = 0.0_f64;
                    for i in 1..=n {
                        s += a[ix(i, k, lda)].abs() * x[ix(i, j, ldx)].abs();
                    }
                    w1[k - 1] += s;
                }
            }
            let mut s = 0.0_f64;
            for i in 0..n {
                if w1[i] > safe2 {
                    s = fmax(s, w2[i].abs() / w1[i]);
                } else {
                    s = fmax(s, (w2[i].abs() + safe1) / (w1[i] + safe1));
                }
            }
            out.berr[j - 1] = s;
            if out.berr[j - 1] > eps
                && 2.0 * out.berr[j - 1] <= lstres
                && count <= ITMAX
            {
                dgetrs(trans, n, 1, af, ldaf, ipiv, &mut w2, n)?;
                daxpy(n, 1.0, &w2, 1, &mut x[ix(1, j, ldx)..], 1);
                lstres = out.berr[j - 1];
                count += 1;
                continue;
            }
            break;
        }
        // Error bound (dgerfs.f:385-427)
        for i in 0..n {
            if w1[i] > safe2 {
                w1[i] = w2[i].abs() + nz as f64 * eps * w1[i];
            } else {
                w1[i] = w2[i].abs() + nz as f64 * eps * w1[i] + safe1;
            }
        }
        let mut st = Lacn2State::default();
        let mut ferr = 0.0_f64;
        loop {
            dlacn2(n, &mut w3, &mut w2, &mut ferr, &mut st);
            if st.kase == 0 {
                break;
            }
            if st.kase == 1 {
                dgetrs(transt, n, 1, af, ldaf, ipiv, &mut w2, n)?;
                for i in 0..n {
                    w2[i] *= w1[i];
                }
            } else {
                for i in 0..n {
                    w2[i] *= w1[i];
                }
                dgetrs(trans, n, 1, af, ldaf, ipiv, &mut w2, n)?;
            }
        }
        let mut lst = 0.0_f64;
        for i in 1..=n {
            lst = fmax(lst, x[ix(i, j, ldx)].abs());
        }
        if lst != 0.0 {
            ferr /= lst;
        }
        out.ferr[j - 1] = ferr;
    }
    Ok(out)
}

/// Output of [`dgesvx`].
#[derive(Debug, Clone, PartialEq)]
pub struct Gesvx {
    /// `EQUED`: `b'N'`, `b'R'`, `b'C'` or `b'B'`.
    pub equed: u8,
    /// Row scale factors `R` (unset entries are 0 when not computed).
    pub r: Vec<f64>,
    /// Column scale factors `C`.
    pub c: Vec<f64>,
    /// Reciprocal condition number.
    pub rcond: f64,
    /// Forward error bounds.
    pub ferr: Vec<f64>,
    /// Backward errors.
    pub berr: Vec<f64>,
    /// `WORK(1)`: reciprocal pivot growth factor.
    pub rpvgrw: f64,
    /// `INFO` (`> 0`: singular `U(info,info)`, or `n + 1` when
    /// `rcond < eps`).
    pub info: usize,
}

/// `DGESVX` (`SRC/dgesvx.f`) with `FACT = 'N'` or `'E'`: solve `A*X = B`
/// (`trans = 'N'`) or `A**T*X = B`, with optional equilibration,
/// condition estimation and refinement. `A` and `B` are overwritten by
/// their equilibrated forms, `af`/`ipiv` receive the factors and `x` the
/// solution (all column-major, leading dimension `n`).
///
/// # Errors
///
/// [`LapackError::UnportedPath`] for `FACT = 'F'`; propagates errors of the
/// routines it calls.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn dgesvx(
    fact: u8,
    trans: u8,
    n: usize,
    nrhs: usize,
    a: &mut [f64],
    af: &mut [f64],
    ipiv: &mut [usize],
    b: &mut [f64],
    x: &mut [f64],
) -> Result<Gesvx, LapackError> {
    let nofact = lsame(fact, b'N');
    let equil = lsame(fact, b'E');
    let notran = lsame(trans, b'N');
    if !nofact && !equil {
        return Err(unported("DGESVX", "401-405, 428-457"));
    }
    if !notran && !lsame(trans, b'T') && !lsame(trans, b'C') {
        return Err(LapackError::IllegalArgument {
            routine: "DGESVX",
            info: -2,
        });
    }
    let (lda, ldaf, ldb, ldx) = (n.max(1), n.max(1), n.max(1), n.max(1));
    let mut out = Gesvx {
        equed: b'N',
        r: vec![0.0; n],
        c: vec![0.0; n],
        rcond: 0.0,
        ferr: vec![0.0; nrhs],
        berr: vec![0.0; nrhs],
        rpvgrw: 0.0,
        info: 0,
    };
    let mut rowequ = false;
    let mut colequ = false;
    let mut colcnd = 0.0;
    if equil {
        // Equilibrate (dgesvx.f:472-486)
        let eq = dgeequ(n, n, a, lda);
        out.r.clone_from(&eq.r);
        out.c.clone_from(&eq.c);
        colcnd = eq.colcnd;
        if eq.info == 0 {
            out.equed = dlaqge(
                n, n, a, lda, &eq.r, &eq.c, eq.rowcnd, eq.colcnd, eq.amax,
            );
            rowequ = lsame(out.equed, b'R') || lsame(out.equed, b'B');
            colequ = lsame(out.equed, b'C') || lsame(out.equed, b'B');
        }
    }
    // Scale the right-hand side (dgesvx.f:490-504)
    if notran {
        if rowequ {
            for j in 1..=nrhs {
                for i in 1..=n {
                    b[ix(i, j, ldb)] = out.r[i - 1] * b[ix(i, j, ldb)];
                }
            }
        }
    } else if colequ {
        for j in 1..=nrhs {
            for i in 1..=n {
                b[ix(i, j, ldb)] = out.c[i - 1] * b[ix(i, j, ldb)];
            }
        }
    }
    // Factor (dgesvx.f:506-531)
    dlacpy(b'F', n, n, a, lda, af, ldaf);
    out.info = dgetrf(n, n, af, ldaf, ipiv)?;
    if out.info > 0 {
        let info = out.info;
        let mut rpvgrw = dlantr(b'M', b'U', b'N', info, info, af, ldaf)?;
        if rpvgrw == 0.0 {
            rpvgrw = 1.0;
        } else {
            rpvgrw = dlange(b'M', n, info, a, lda)? / rpvgrw;
        }
        out.rpvgrw = rpvgrw;
        out.rcond = 0.0;
        return Ok(out);
    }
    let norm = if notran { b'1' } else { b'I' };
    let anorm = dlange(norm, n, n, a, lda)?;
    let mut rpvgrw = dlantr(b'M', b'U', b'N', n, n, af, ldaf)?;
    if rpvgrw == 0.0 {
        rpvgrw = 1.0;
    } else {
        rpvgrw = dlange(b'M', n, n, a, lda)? / rpvgrw;
    }
    let (rcond, _) = dgecon(norm, n, af, ldaf, anorm)?;
    out.rcond = rcond;
    // Solve and refine (dgesvx.f:555-562)
    dlacpy(b'F', n, nrhs, b, ldb, x, ldx);
    dgetrs(trans, n, nrhs, af, ldaf, ipiv, x, ldx)?;
    let rf = dgerfs(trans, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx)?;
    out.ferr = rf.ferr;
    out.berr = rf.berr;
    // Transform the solution back (dgesvx.f:567-587)
    if notran {
        if colequ {
            for j in 1..=nrhs {
                for i in 1..=n {
                    x[ix(i, j, ldx)] = out.c[i - 1] * x[ix(i, j, ldx)];
                }
            }
            for f in &mut out.ferr {
                *f /= colcnd;
            }
        }
    } else if rowequ {
        return Err(unported("DGESVX", "578-586"));
    }
    out.rpvgrw = rpvgrw;
    if out.rcond < dlamch::EPS {
        out.info = n + 1;
    }
    Ok(out)
}
