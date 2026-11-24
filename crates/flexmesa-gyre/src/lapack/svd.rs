// SPDX-License-Identifier: GPL-3.0-only
//! Reference LAPACK 3.12.0 SVD path GYRE uses (`DGESVD('A','A')` on square
//! matrices): `SRC/dgesvd.f`, `dgebrd.f`, `dgebd2.f`, `dorgbr.f`,
//! `dorgqr.f`, `dorg2r.f`, `dorglq.f`, `dorgl2.f`, `dbdsqr.f`, double
//! precision, as the MESA SDK `26.6.1` builds them (`gfortran -O2 -fPIC`;
//! `dbdsqr` with the SDK's CRMATH tolerance, see [`dbdsqr`]).

use super::aux::{
    dlacpy, dlange, dlarf, dlarfg, dlartg, dlas2, dlasr, dlasv2, fmax, fmin,
    sign,
};
use super::blas::{drot_in, dscal, dswap_in};
use super::{dlamch, ilaenv_mnthr, ilaenv_nb, ix, lsame, LapackError};

fn unported(routine: &'static str, lines: &'static str) -> LapackError {
    LapackError::UnportedPath { routine, lines }
}

fn nb(name: &str) -> usize {
    usize::try_from(ilaenv_nb(name)).unwrap_or(1)
}

/// `DGEBD2` (`SRC/dgebd2.f`), `m >= n`: unblocked reduction to upper
/// bidiagonal form. `work` needs `max(m, n)` entries.
///
/// # Errors
///
/// [`LapackError::UnportedPath`] for `m < n` (lines 273-311).
#[allow(clippy::too_many_arguments)]
pub fn dgebd2(
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    d: &mut [f64],
    e: &mut [f64],
    tauq: &mut [f64],
    taup: &mut [f64],
    work: &mut [f64],
) -> Result<(), LapackError> {
    if m < n {
        return Err(unported("DGEBD2", "273-311"));
    }
    for i in 1..=n {
        // H(i) annihilates A(i+1:m, i) (dgebd2.f:242-252)
        let mut alpha = a[ix(i, i, lda)];
        let xs = ix(m.min(i + 1), i, lda);
        tauq[i - 1] = dlarfg(m - i + 1, &mut alpha, &mut a[xs..], 1);
        a[ix(i, i, lda)] = alpha;
        d[i - 1] = a[ix(i, i, lda)];
        a[ix(i, i, lda)] = 1.0;
        if i < n {
            let v: Vec<f64> = a[ix(i, i, lda)..ix(m, i, lda) + 1].to_vec();
            dlarf(
                b'L',
                m - i + 1,
                n - i,
                &v,
                1,
                tauq[i - 1],
                &mut a[ix(i, i + 1, lda)..],
                lda,
                work,
            )?;
        }
        a[ix(i, i, lda)] = d[i - 1];
        if i < n {
            // G(i) annihilates A(i, i+2:n) (dgebd2.f:254-271)
            let mut alpha = a[ix(i, i + 1, lda)];
            let xs = ix(i, n.min(i + 2), lda);
            taup[i - 1] = dlarfg(n - i, &mut alpha, &mut a[xs..], lda);
            a[ix(i, i + 1, lda)] = alpha;
            e[i - 1] = a[ix(i, i + 1, lda)];
            a[ix(i, i + 1, lda)] = 1.0;
            let v: Vec<f64> = (i + 1..=n).map(|j| a[ix(i, j, lda)]).collect();
            if m > i {
                dlarf(
                    b'R',
                    m - i,
                    n - i,
                    &v,
                    1,
                    taup[i - 1],
                    &mut a[ix(i + 1, i + 1, lda)..],
                    lda,
                    work,
                )?;
            }
            a[ix(i, i + 1, lda)] = e[i - 1];
        } else {
            taup[i - 1] = 0.0;
        }
    }
    Ok(())
}

/// `DGEBRD` (`SRC/dgebrd.f`): reduction to bidiagonal form; unblocked
/// (`DGEBD2`) when `NB = 32 >= min(m, n)`.
///
/// # Errors
///
/// [`LapackError::UnportedPath`] for the blocked path (`min(m, n) > 32`).
#[allow(clippy::too_many_arguments)]
pub fn dgebrd(
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    d: &mut [f64],
    e: &mut [f64],
    tauq: &mut [f64],
    taup: &mut [f64],
    lwork: usize,
) -> Result<(), LapackError> {
    if lwork < m.max(n).max(1) {
        return Err(LapackError::IllegalArgument {
            routine: "DGEBRD",
            info: -10,
        });
    }
    let minmn = m.min(n);
    if minmn == 0 {
        return Ok(());
    }
    let nbv = nb("DGEBRD").max(1);
    if nbv > 1 && nbv < minmn {
        return Err(unported("DGEBRD", "276-338"));
    }
    let mut work = vec![0.0_f64; m.max(n)];
    dgebd2(m, n, a, lda, d, e, tauq, taup, &mut work)
}

/// `DORG2R` (`SRC/dorg2r.f`): form the `m x n` `Q` with orthonormal columns
/// from `k` reflectors. `work` needs `n` entries.
///
/// # Errors
///
/// Propagates `DLARF` errors.
pub fn dorg2r(
    m: usize,
    n: usize,
    k: usize,
    a: &mut [f64],
    lda: usize,
    tau: &[f64],
    work: &mut [f64],
) -> Result<(), LapackError> {
    if n == 0 {
        return Ok(());
    }
    for j in k + 1..=n {
        for l in 1..=m {
            a[ix(l, j, lda)] = 0.0;
        }
        a[ix(j, j, lda)] = 1.0;
    }
    for i in (1..=k).rev() {
        if i < n {
            a[ix(i, i, lda)] = 1.0;
            let v: Vec<f64> = a[ix(i, i, lda)..ix(m, i, lda) + 1].to_vec();
            dlarf(
                b'L',
                m - i + 1,
                n - i,
                &v,
                1,
                tau[i - 1],
                &mut a[ix(i, i + 1, lda)..],
                lda,
                work,
            )?;
        }
        if i < m {
            dscal(m - i, -tau[i - 1], &mut a[ix(i + 1, i, lda)..], 1);
        }
        a[ix(i, i, lda)] = 1.0 - tau[i - 1];
        for l in 1..i {
            a[ix(l, i, lda)] = 0.0;
        }
    }
    Ok(())
}

/// `DORGQR` (`SRC/dorgqr.f`); unblocked (`DORG2R`) when `NB = 32 >= k`.
///
/// # Errors
///
/// [`LapackError::UnportedPath`] for the blocked path.
pub fn dorgqr(
    m: usize,
    n: usize,
    k: usize,
    a: &mut [f64],
    lda: usize,
    tau: &[f64],
) -> Result<(), LapackError> {
    if n == 0 {
        return Ok(());
    }
    let nbv = nb("DORGQR");
    if nbv > 1 && nbv < k {
        return Err(unported("DORGQR", "198-280"));
    }
    let mut work = vec![0.0_f64; n.max(1)];
    dorg2r(m, n, k, a, lda, tau, &mut work)
}

/// `DORGL2` (`SRC/dorgl2.f`): form the `m x n` `Q` with orthonormal rows
/// from `k` reflectors. `work` needs `m` entries.
///
/// # Errors
///
/// Propagates `DLARF` errors.
pub fn dorgl2(
    m: usize,
    n: usize,
    k: usize,
    a: &mut [f64],
    lda: usize,
    tau: &[f64],
    work: &mut [f64],
) -> Result<(), LapackError> {
    if m == 0 {
        return Ok(());
    }
    if k < m {
        for j in 1..=n {
            for l in k + 1..=m {
                a[ix(l, j, lda)] = 0.0;
            }
            if j > k && j <= m {
                a[ix(j, j, lda)] = 1.0;
            }
        }
    }
    for i in (1..=k).rev() {
        if i < n {
            if i < m {
                a[ix(i, i, lda)] = 1.0;
                let v: Vec<f64> = (i..=n).map(|j| a[ix(i, j, lda)]).collect();
                dlarf(
                    b'R',
                    m - i,
                    n - i + 1,
                    &v,
                    1,
                    tau[i - 1],
                    &mut a[ix(i + 1, i, lda)..],
                    lda,
                    work,
                )?;
            }
            dscal(n - i, -tau[i - 1], &mut a[ix(i, i + 1, lda)..], lda);
        }
        a[ix(i, i, lda)] = 1.0 - tau[i - 1];
        for l in 1..i {
            a[ix(i, l, lda)] = 0.0;
        }
    }
    Ok(())
}

/// `DORGLQ` (`SRC/dorglq.f`); unblocked (`DORGL2`) when `NB = 32 >= k`.
///
/// # Errors
///
/// [`LapackError::UnportedPath`] for the blocked path.
pub fn dorglq(
    m: usize,
    n: usize,
    k: usize,
    a: &mut [f64],
    lda: usize,
    tau: &[f64],
) -> Result<(), LapackError> {
    if m == 0 {
        return Ok(());
    }
    let nbv = nb("DORGLQ");
    if nbv > 1 && nbv < k {
        return Err(unported("DORGLQ", "195-260"));
    }
    let mut work = vec![0.0_f64; m.max(1)];
    dorgl2(m, n, k, a, lda, tau, &mut work)
}

/// `DORGBR` (`SRC/dorgbr.f`): generate `Q` (`vect = 'Q'`) or `P**T`
/// (`'P'`) from `DGEBRD`'s reflectors.
///
/// # Errors
///
/// [`LapackError::UnportedPath`] for `Q` with `m < k` and `P**T` with
/// `k < n` (not reached by square `DGESVD`).
#[allow(clippy::too_many_arguments)]
pub fn dorgbr(
    vect: u8,
    m: usize,
    n: usize,
    k: usize,
    a: &mut [f64],
    lda: usize,
    tau: &[f64],
) -> Result<(), LapackError> {
    let wantq = lsame(vect, b'Q');
    if m == 0 || n == 0 {
        return Ok(());
    }
    if wantq {
        if m >= k {
            dorgqr(m, n, k, a, lda, tau)
        } else {
            Err(unported("DORGBR", "265-290"))
        }
    } else if k < n {
        Err(unported("DORGBR", "296-301"))
    } else {
        // Shift the reflector vectors one row down, first row/col of P**T
        // to the unit vector (dorgbr.f:310-319)
        a[ix(1, 1, lda)] = 1.0;
        for i in 2..=n {
            a[ix(i, 1, lda)] = 0.0;
        }
        for j in 2..=n {
            for i in (2..j).rev() {
                a[ix(i, j, lda)] = a[ix(i - 1, j, lda)];
            }
            a[ix(1, j, lda)] = 0.0;
        }
        if n > 1 {
            let off = ix(2, 2, lda);
            dorglq(n - 1, n - 1, n - 1, &mut a[off..], lda, tau)?;
        }
        Ok(())
    }
}

/// `DBDSQR` (`SRC/dbdsqr.f`), `UPLO = 'U'` with singular vectors
/// (`ncvt > 0` or `nru > 0`), `ncc = 0`: SVD of an upper bidiagonal matrix
/// by implicit zero-shift / shifted QR. `d` (length `n`) receives the
/// singular values in decreasing order, `e` (length `n-1`) is destroyed,
/// `VT` (`n x ncvt`) and `U` (`nru x n`) are updated. Returns `INFO`
/// (`> 0`: number of unconverged superdiagonals).
///
/// **SDK difference.** The MESA SDK `crmath`-slot object evaluates
/// `TOLMUL = MAX(TEN, MIN(HNDRD, EPS**MEIGTH))` (`dbdsqr.f:382`) as
/// `MAX(TEN, MIN(HNDRD, exp(0*log(EPS))))`: the multiplier of `log(EPS)`
/// is a zero register (`pxor`) in the disassembly, so `TOLMUL = 10`, not
/// netlib's `EPS**(-1/8)`. This port reproduces the SDK.
///
/// # Errors
///
/// [`LapackError::UnportedPath`] for `UPLO = 'L'` (lines 358-376) and for
/// no singular vectors (`DLASQ1`, lines 336-343).
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::similar_names
)]
pub fn dbdsqr(
    uplo: u8,
    n: usize,
    ncvt: usize,
    nru: usize,
    d: &mut [f64],
    e: &mut [f64],
    vt: &mut [f64],
    ldvt: usize,
    u: &mut [f64],
    ldu: usize,
) -> Result<usize, LapackError> {
    dbdsqr_tol(uplo, n, ncvt, nru, d, e, vt, ldvt, u, ldu, None)
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::similar_names
)]
fn dbdsqr_tol(
    uplo: u8,
    n: usize,
    ncvt: usize,
    nru: usize,
    d: &mut [f64],
    e: &mut [f64],
    vt: &mut [f64],
    ldvt: usize,
    u: &mut [f64],
    ldu: usize,
    tolmul_override: Option<f64>,
) -> Result<usize, LapackError> {
    const HNDRTH: f64 = 0.01;
    const TEN: f64 = 10.0;
    const HNDRD: f64 = 100.0;
    const MAXITR: usize = 6;
    if !lsame(uplo, b'U') {
        return Err(unported("DBDSQR", "358-376"));
    }
    if n == 0 {
        return Ok(0);
    }
    let mut info = 0;
    if n > 1 {
        let rotate = ncvt > 0 || nru > 0;
        if !rotate {
            return Err(unported("DBDSQR", "336-343"));
        }
        let nm1 = n - 1;
        let nm12 = nm1 + nm1;
        let nm13 = nm12 + nm1;
        let mut work = vec![0.0_f64; 4 * nm1];
        let mut idir = 0;
        let eps = dlamch::EPS;
        let unfl = dlamch::SFMIN;
        // SDK: EPS**MEIGTH evaluated as exp(0*log(EPS)) (see doc comment)
        let tolmul = tolmul_override.unwrap_or_else(|| {
            fmax(
                TEN,
                fmin(HNDRD, flexmesa_math::exp(0.0 * flexmesa_math::log(eps))),
            )
        });
        let tol = tolmul * eps;
        let mut smax = 0.0_f64;
        for &di in d.iter().take(n) {
            smax = fmax(smax, di.abs());
        }
        for &ei in e.iter().take(n - 1) {
            smax = fmax(smax, ei.abs());
        }
        let mut smin;
        let thresh;
        if tol >= 0.0 {
            let mut sminoa = d[0].abs();
            if sminoa != 0.0 {
                let mut mu = sminoa;
                for i in 2..=n {
                    mu = d[i - 1].abs() * (mu / (mu + e[i - 2].abs()));
                    sminoa = fmin(sminoa, mu);
                    if sminoa == 0.0 {
                        break;
                    }
                }
            }
            sminoa /= (n as f64).sqrt();
            thresh = fmax(
                tol * sminoa,
                MAXITR as f64 * (n as f64 * (n as f64 * unfl)),
            );
        } else {
            thresh = fmax(
                tol.abs() * smax,
                MAXITR as f64 * (n as f64 * (n as f64 * unfl)),
            );
        }
        let maxitdivn = MAXITR * n;
        let mut iterdivn = 0;
        let mut iter: isize = -1;
        let mut oldll: isize = -1;
        let mut oldm: isize = -1;
        let mut m = n;
        let ni = n as isize;
        // Main loop (label 60)
        'l60: loop {
            if m <= 1 {
                break 'l60;
            }
            if iter >= ni {
                iter -= ni;
                iterdivn += 1;
                if iterdivn >= maxitdivn {
                    // Failure to converge (label 200)
                    info = 0;
                    for &ei in e.iter().take(n - 1) {
                        if ei != 0.0 {
                            info += 1;
                        }
                    }
                    return Ok(info);
                }
            }
            if tol < 0.0 && d[m - 1].abs() <= thresh {
                d[m - 1] = 0.0;
            }
            smax = d[m - 1].abs();
            let mut ll = 0;
            let mut split = false;
            for lll in 1..m {
                ll = m - lll;
                let abss = d[ll - 1].abs();
                let abse = e[ll - 1].abs();
                if tol < 0.0 && abss <= thresh {
                    d[ll - 1] = 0.0;
                }
                if abse <= thresh {
                    split = true;
                    break;
                }
                smax = fmax(fmax(smax, abss), abse);
            }
            if split {
                // label 80
                e[ll - 1] = 0.0;
                if ll == m - 1 {
                    m -= 1;
                    continue 'l60;
                }
            } else {
                ll = 0;
            }
            // label 90
            ll += 1;
            if ll == m - 1 {
                // 2x2 block (dbdsqr.f:487-504)
                let r = dlasv2(d[m - 2], e[m - 2], d[m - 1]);
                d[m - 2] = r.ssmax;
                e[m - 2] = 0.0;
                d[m - 1] = r.ssmin;
                if ncvt > 0 {
                    drot_in(
                        ncvt,
                        vt,
                        ix(m - 1, 1, ldvt),
                        ldvt,
                        ix(m, 1, ldvt),
                        ldvt,
                        r.csr,
                        r.snr,
                    );
                }
                if nru > 0 {
                    drot_in(
                        nru,
                        u,
                        ix(1, m - 1, ldu),
                        1,
                        ix(1, m, ldu),
                        1,
                        r.csl,
                        r.snl,
                    );
                }
                m -= 2;
                continue 'l60;
            }
            let (lli, mi) = (ll as isize, m as isize);
            if lli > oldm || mi < oldll {
                idir = if d[ll - 1].abs() >= d[m - 1].abs() {
                    1
                } else {
                    2
                };
            }
            // Convergence tests (dbdsqr.f:526-581)
            smin = 0.0;
            if idir == 1 {
                if e[m - 2].abs() <= tol.abs() * d[m - 1].abs()
                    || (tol < 0.0 && e[m - 2].abs() <= thresh)
                {
                    e[m - 2] = 0.0;
                    continue 'l60;
                }
                if tol >= 0.0 {
                    let mut mu = d[ll - 1].abs();
                    smin = mu;
                    for lll in ll..m {
                        if e[lll - 1].abs() <= tol * mu {
                            e[lll - 1] = 0.0;
                            continue 'l60;
                        }
                        mu = d[lll].abs() * (mu / (mu + e[lll - 1].abs()));
                        smin = fmin(smin, mu);
                    }
                }
            } else {
                if e[ll - 1].abs() <= tol.abs() * d[ll - 1].abs()
                    || (tol < 0.0 && e[ll - 1].abs() <= thresh)
                {
                    e[ll - 1] = 0.0;
                    continue 'l60;
                }
                if tol >= 0.0 {
                    let mut mu = d[m - 1].abs();
                    smin = mu;
                    for lll in (ll..m).rev() {
                        if e[lll - 1].abs() <= tol * mu {
                            e[lll - 1] = 0.0;
                            continue 'l60;
                        }
                        mu = d[lll - 1].abs() * (mu / (mu + e[lll - 1].abs()));
                        smin = fmin(smin, mu);
                    }
                }
            }
            oldll = lli;
            oldm = mi;
            // Shift (dbdsqr.f:588-612)
            let mut shift;
            if tol >= 0.0
                && n as f64 * tol * (smin / smax) <= fmax(eps, HNDRTH * tol)
            {
                shift = 0.0;
            } else {
                let sll;
                if idir == 1 {
                    sll = d[ll - 1].abs();
                    shift = dlas2(d[m - 2], e[m - 2], d[m - 1]).0;
                } else {
                    sll = d[m - 1].abs();
                    shift = dlas2(d[ll - 1], e[ll - 1], d[ll]).0;
                }
                if sll > 0.0 {
                    let q = shift / sll;
                    if q * q < eps {
                        shift = 0.0;
                    }
                }
            }
            iter += mi - lli;
            let w = |k: usize, off: usize| off + k - 1;
            if shift == 0.0 {
                if idir == 1 {
                    // Zero-shift QR, top to bottom (dbdsqr.f:626-657)
                    let mut cs = 1.0;
                    let mut oldcs = 1.0;
                    let mut oldsn = 0.0;
                    for i in ll..m {
                        let (c1, s1, r) = dlartg(d[i - 1] * cs, e[i - 1]);
                        cs = c1;
                        let sn = s1;
                        if i > ll {
                            e[i - 2] = oldsn * r;
                        }
                        let (c2, s2, r2) = dlartg(oldcs * r, d[i] * sn);
                        oldcs = c2;
                        oldsn = s2;
                        d[i - 1] = r2;
                        work[w(i - ll + 1, 0)] = cs;
                        work[w(i - ll + 1, nm1)] = sn;
                        work[w(i - ll + 1, nm12)] = oldcs;
                        work[w(i - ll + 1, nm13)] = oldsn;
                    }
                    let h = d[m - 1] * cs;
                    d[m - 1] = h * oldcs;
                    e[m - 2] = h * oldsn;
                    apply_rotations(
                        true,
                        m - ll + 1,
                        ncvt,
                        nru,
                        &work,
                        nm1,
                        nm12,
                        nm13,
                        vt,
                        ldvt,
                        u,
                        ldu,
                        ll,
                    )?;
                    if e[m - 2].abs() <= thresh {
                        e[m - 2] = 0.0;
                    }
                } else {
                    // Zero-shift QR, bottom to top (dbdsqr.f:664-695)
                    let mut cs = 1.0;
                    let mut oldcs = 1.0;
                    let mut oldsn = 0.0;
                    for i in (ll + 1..=m).rev() {
                        let (c1, s1, r) = dlartg(d[i - 1] * cs, e[i - 2]);
                        cs = c1;
                        let sn = s1;
                        if i < m {
                            e[i - 1] = oldsn * r;
                        }
                        let (c2, s2, r2) = dlartg(oldcs * r, d[i - 2] * sn);
                        oldcs = c2;
                        oldsn = s2;
                        d[i - 1] = r2;
                        work[w(i - ll, 0)] = cs;
                        work[w(i - ll, nm1)] = -sn;
                        work[w(i - ll, nm12)] = oldcs;
                        work[w(i - ll, nm13)] = -oldsn;
                    }
                    let h = d[ll - 1] * cs;
                    d[ll - 1] = h * oldcs;
                    e[ll - 1] = h * oldsn;
                    apply_rotations(
                        false,
                        m - ll + 1,
                        ncvt,
                        nru,
                        &work,
                        nm1,
                        nm12,
                        nm13,
                        vt,
                        ldvt,
                        u,
                        ldu,
                        ll,
                    )?;
                    if e[ll - 1].abs() <= thresh {
                        e[ll - 1] = 0.0;
                    }
                }
            } else if idir == 1 {
                // Shifted QR, top to bottom (dbdsqr.f:706-747)
                let mut f = (d[ll - 1].abs() - shift)
                    * (sign(1.0, d[ll - 1]) + shift / d[ll - 1]);
                let mut g = e[ll - 1];
                for i in ll..m {
                    let (cosr, sinr, r) = dlartg(f, g);
                    if i > ll {
                        e[i - 2] = r;
                    }
                    f = cosr * d[i - 1] + sinr * e[i - 1];
                    e[i - 1] = cosr * e[i - 1] - sinr * d[i - 1];
                    g = sinr * d[i];
                    d[i] *= cosr;
                    let (cosl, sinl, r) = dlartg(f, g);
                    d[i - 1] = r;
                    f = cosl * e[i - 1] + sinl * d[i];
                    d[i] = cosl * d[i] - sinl * e[i - 1];
                    if i < m - 1 {
                        g = sinl * e[i];
                        e[i] *= cosl;
                    }
                    work[w(i - ll + 1, 0)] = cosr;
                    work[w(i - ll + 1, nm1)] = sinr;
                    work[w(i - ll + 1, nm12)] = cosl;
                    work[w(i - ll + 1, nm13)] = sinl;
                }
                e[m - 2] = f;
                apply_rotations(
                    true,
                    m - ll + 1,
                    ncvt,
                    nru,
                    &work,
                    nm1,
                    nm12,
                    nm13,
                    vt,
                    ldvt,
                    u,
                    ldu,
                    ll,
                )?;
                if e[m - 2].abs() <= thresh {
                    e[m - 2] = 0.0;
                }
            } else {
                // Shifted QR, bottom to top (dbdsqr.f:754-796)
                let mut f = (d[m - 1].abs() - shift)
                    * (sign(1.0, d[m - 1]) + shift / d[m - 1]);
                let mut g = e[m - 2];
                for i in (ll + 1..=m).rev() {
                    let (cosr, sinr, r) = dlartg(f, g);
                    if i < m {
                        e[i - 1] = r;
                    }
                    f = cosr * d[i - 1] + sinr * e[i - 2];
                    e[i - 2] = cosr * e[i - 2] - sinr * d[i - 1];
                    g = sinr * d[i - 2];
                    d[i - 2] *= cosr;
                    let (cosl, sinl, r) = dlartg(f, g);
                    d[i - 1] = r;
                    f = cosl * e[i - 2] + sinl * d[i - 2];
                    d[i - 2] = cosl * d[i - 2] - sinl * e[i - 2];
                    if i > ll + 1 {
                        g = sinl * e[i - 3];
                        e[i - 3] *= cosl;
                    }
                    work[w(i - ll, 0)] = cosr;
                    work[w(i - ll, nm1)] = -sinr;
                    work[w(i - ll, nm12)] = cosl;
                    work[w(i - ll, nm13)] = -sinl;
                }
                e[ll - 1] = f;
                if e[ll - 1].abs() <= thresh {
                    e[ll - 1] = 0.0;
                }
                apply_rotations(
                    false,
                    m - ll + 1,
                    ncvt,
                    nru,
                    &work,
                    nm1,
                    nm12,
                    nm13,
                    vt,
                    ldvt,
                    u,
                    ldu,
                    ll,
                )?;
            }
        }
    }
    // label 160: make singular values positive, then sort (dbdsqr.f:806-846)
    for i in 1..=n {
        if d[i - 1] < 0.0 {
            d[i - 1] = -d[i - 1];
            if ncvt > 0 {
                dscal(ncvt, -1.0, &mut vt[ix(i, 1, ldvt)..], ldvt);
            }
        }
    }
    for i in 1..n {
        let mut isub = 1;
        let mut smin = d[0];
        for j in 2..=n + 1 - i {
            if d[j - 1] <= smin {
                isub = j;
                smin = d[j - 1];
            }
        }
        if isub != n + 1 - i {
            d[isub - 1] = d[n - i];
            d[n - i] = smin;
            if ncvt > 0 {
                dswap_in(
                    ncvt,
                    vt,
                    ix(isub, 1, ldvt),
                    ldvt,
                    ix(n + 1 - i, 1, ldvt),
                    ldvt,
                );
            }
            if nru > 0 {
                dswap_in(nru, u, ix(1, isub, ldu), 1, ix(1, n + 1 - i, ldu), 1);
            }
        }
    }
    Ok(info)
}

/// The `DLASR` updates of `VT` and `U` after a QR sweep (forward: rows of
/// `VT` with `WORK(1)`, `WORK(N)`; columns of `U` with `WORK(NM12+1)`,
/// `WORK(NM13+1)`; backward: the roles swap).
#[allow(clippy::too_many_arguments)]
fn apply_rotations(
    forward: bool,
    len: usize,
    ncvt: usize,
    nru: usize,
    work: &[f64],
    nm1: usize,
    nm12: usize,
    nm13: usize,
    vt: &mut [f64],
    ldvt: usize,
    u: &mut [f64],
    ldu: usize,
    ll: usize,
) -> Result<(), LapackError> {
    // WORK(N) is WORK(NM1+1)
    let (vc, vs, uc, us, dir) = if forward {
        (0, nm1, nm12, nm13, b'F')
    } else {
        (nm12, nm13, 0, nm1, b'B')
    };
    if ncvt > 0 {
        dlasr(
            b'L',
            b'V',
            dir,
            len,
            ncvt,
            &work[vc..],
            &work[vs..],
            &mut vt[ix(ll, 1, ldvt)..],
            ldvt,
        )?;
    }
    if nru > 0 {
        dlasr(
            b'R',
            b'V',
            dir,
            nru,
            len,
            &work[uc..],
            &work[us..],
            &mut u[ix(1, ll, ldu)..],
            ldu,
        )?;
    }
    Ok(())
}

/// Output of [`dgesvd`].
#[derive(Debug, Clone, PartialEq)]
pub struct Gesvd {
    /// Singular values, decreasing.
    pub s: Vec<f64>,
    /// `WORK(1)` on exit (`MAXWRK`).
    pub work1: f64,
    /// `INFO` from `DBDSQR`.
    pub info: usize,
}

/// `DGESVD` (`SRC/dgesvd.f`), `JOBU = JOBVT = 'A'`, `m >= n` with
/// `m < MNTHR` (path 10, the square case GYRE uses): `A = U*S*VT`. `A` is
/// destroyed (it holds `DGEBRD`'s output), `U` is `m x m`, `VT` is
/// `n x n` (leading dimensions `m` and `n`). `lwork` is the workspace size
/// the caller declares (GYRE passes `5*n`); it only enters the size checks
/// and `WORK(1)`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] when `lwork` is below the minimum;
/// [`LapackError::UnportedPath`] for other job options and shapes, for the
/// rescaling of matrices with norms outside `[SMLNUM, BIGNUM]`
/// (`DLASCL`, lines 660-666) and for the `INFO > 0` workspace shuffle.
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
pub fn dgesvd(
    m: usize,
    n: usize,
    a: &mut [f64],
    u: &mut [f64],
    vt: &mut [f64],
    lwork: usize,
) -> Result<Gesvd, LapackError> {
    dgesvd_tol(m, n, a, u, vt, lwork, None)
}

/// [`dgesvd`] with `DBDSQR`'s `TOLMUL` replaced. Only for the negative
/// control that shows the SDK's `TOLMUL = 10` is observable in the
/// vectors; not part of the port.
///
/// # Errors
///
/// As [`dgesvd`].
#[doc(hidden)]
pub fn dgesvd_with_tolmul(
    m: usize,
    n: usize,
    a: &mut [f64],
    u: &mut [f64],
    vt: &mut [f64],
    lwork: usize,
    tolmul: f64,
) -> Result<Gesvd, LapackError> {
    dgesvd_tol(m, n, a, u, vt, lwork, Some(tolmul))
}

#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
fn dgesvd_tol(
    m: usize,
    n: usize,
    a: &mut [f64],
    u: &mut [f64],
    vt: &mut [f64],
    lwork: usize,
    tolmul: Option<f64>,
) -> Result<Gesvd, LapackError> {
    let lda = m.max(1);
    let ldu = m.max(1);
    let ldvt = n.max(1);
    let minmn = m.min(n);
    if !(m >= n && minmn > 0) {
        return Err(unported("DGESVD", "466-628"));
    }
    let mnthr = ilaenv_mnthr(m, n);
    if m >= mnthr {
        return Err(unported("DGESVD", "674-1965"));
    }
    // Workspace bookkeeping (dgesvd.f:306-334, 440-466, 629-636)
    let nb_gebrd = nb("DGEBRD").max(1);
    let nb_org = nb("DORGQR");
    let nb_orl = nb("DORGLQ");
    let bdspac = 5 * n;
    let lwork_dorgbr_p =
        (if n > 1 { (n - 1).max(1) * nb_orl } else { 1 }).max(n);
    let lwork_dgebrd = (m + n) * nb_gebrd;
    let mut maxwrk = 3 * n + lwork_dgebrd;
    let lwork_dorgbr_q = (m.max(1) * nb_org).max(m);
    maxwrk = maxwrk.max(3 * n + lwork_dorgbr_q);
    maxwrk = maxwrk.max(3 * n + lwork_dorgbr_p);
    maxwrk = maxwrk.max(bdspac);
    let minwrk = (3 * n + m).max(bdspac);
    maxwrk = maxwrk.max(minwrk);
    if lwork < minwrk {
        return Err(LapackError::IllegalArgument {
            routine: "DGESVD",
            info: -13,
        });
    }
    // Machine constants and scaling test (dgesvd.f:652-666)
    let eps = dlamch::PREC;
    let smlnum = dlamch::SFMIN.sqrt() / eps;
    let bignum = 1.0 / smlnum;
    let anrm = dlange(b'M', m, n, a, lda)?;
    if (anrm > 0.0 && anrm < smlnum) || anrm > bignum {
        return Err(unported("DGESVD", "660-666, 3478-3491"));
    }
    // Path 10 (dgesvd.f:1972-2042): bidiagonalise, form Q and P**T, QR
    let mut s = vec![0.0_f64; n];
    let mut e = vec![0.0_f64; n];
    let mut tauq = vec![0.0_f64; n];
    let mut taup = vec![0.0_f64; n];
    let iwork = 3 * n + 1;
    dgebrd(
        m,
        n,
        a,
        lda,
        &mut s,
        &mut e,
        &mut tauq,
        &mut taup,
        lwork + 1 - iwork,
    )?;
    dlacpy(b'L', m, n, a, lda, u, ldu);
    dorgbr(b'Q', m, m, n, u, ldu, &tauq)?;
    dlacpy(b'U', n, n, a, lda, vt, ldvt);
    dorgbr(b'P', n, n, n, vt, ldvt, &taup)?;
    let info =
        dbdsqr_tol(b'U', n, n, m, &mut s, &mut e, vt, ldvt, u, ldu, tolmul)?;
    if info != 0 {
        return Err(unported("DGESVD", "3463-3474"));
    }
    #[allow(clippy::cast_precision_loss)]
    let work1 = maxwrk as f64;
    Ok(Gesvd { s, work1, info })
}
