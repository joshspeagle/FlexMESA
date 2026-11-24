// SPDX-License-Identifier: GPL-3.0-only
//! Reference LAPACK 3.12.0 complex SVD path GYRE uses (`ZGESVD('A','A')`
//! on square matrices): `SRC/zgesvd.f`, `zgebrd.f`, `zgebd2.f`,
//! `zgeqrf.f` (workspace query only), `zungbr.f`, `zungqr.f`, `zung2r.f`,
//! `zunglq.f`, `zungl2.f`, `zbdsqr.f`, double precision, as the MESA SDK
//! `26.6.1` builds them (`gfortran -O2 -fPIC`; `zbdsqr` with the SDK's
//! CRMATH tolerance, see [`zbdsqr`]).

use super::aux::{dlartg, dlas2, dlasv2, fmax, fmin, sign};
use super::zaux::{zlacgv, zlacpy, zlange, zlarf, zlarfg, zlasr};
use super::zblas::{zdrot_in, zdscal, zscal, zswap_in};
use super::{dlamch, ilaenv_mnthr, ix, lsame, LapackError};
use crate::fcomplex::Cplx;

const ZERO: Cplx = Cplx::new(0.0, 0.0);
const ONE: Cplx = Cplx::new(1.0, 0.0);

/// `ILAENV(1, name, ...)` for `ZGEQRF`, `ZGEBRD`, `ZUNGQR`, `ZUNGLQ`
/// (`SRC/ilaenv.f:287-293`, `:400-415`).
const NB32: usize = 32;
/// `ILAENV(3, name, ...)` crossover for the same routines
/// (`SRC/ilaenv.f:614-660`).
const NX128: usize = 128;

fn unported(routine: &'static str, lines: &'static str) -> LapackError {
    LapackError::UnportedPath { routine, lines }
}

fn bad(routine: &'static str, info: i32) -> LapackError {
    LapackError::IllegalArgument { routine, info }
}

/// `ZGEBD2` (`SRC/zgebd2.f`), `m >= n`: unblocked reduction of the complex
/// `m x n` `A` to real upper bidiagonal form `Q**H*A*P = B`. `work` needs
/// `max(m, n)` entries.
///
/// # Errors
///
/// [`LapackError::UnportedPath`] for `m < n` (lines 279-322).
#[allow(clippy::too_many_arguments)]
pub fn zgebd2(
    m: usize,
    n: usize,
    a: &mut [Cplx],
    lda: usize,
    d: &mut [f64],
    e: &mut [f64],
    tauq: &mut [Cplx],
    taup: &mut [Cplx],
    work: &mut [Cplx],
) -> Result<(), LapackError> {
    if lda < m.max(1) {
        return Err(bad("ZGEBD2", -4));
    }
    if m < n {
        return Err(unported("ZGEBD2", "279-322"));
    }
    for i in 1..=n {
        // H(i) annihilates A(i+1:m, i) (zgebd2.f:244-255)
        let mut alpha = a[ix(i, i, lda)];
        let xs = ix(m.min(i + 1), i, lda);
        tauq[i - 1] = zlarfg(m - i + 1, &mut alpha, &mut a[xs..], 1);
        d[i - 1] = alpha.re;
        a[ix(i, i, lda)] = ONE;
        if i < n {
            let v: Vec<Cplx> = a[ix(i, i, lda)..=ix(m, i, lda)].to_vec();
            zlarf(
                b'L',
                m - i + 1,
                n - i,
                &v,
                1,
                tauq[i - 1].conj(),
                &mut a[ix(i, i + 1, lda)..],
                lda,
                work,
            )?;
        }
        a[ix(i, i, lda)] = Cplx::from_real(d[i - 1]);
        if i < n {
            // G(i) annihilates A(i, i+2:n) (zgebd2.f:262-274)
            zlacgv(n - i, &mut a[ix(i, i + 1, lda)..], lda);
            let mut alpha = a[ix(i, i + 1, lda)];
            let xs = ix(i, n.min(i + 2), lda);
            taup[i - 1] = zlarfg(n - i, &mut alpha, &mut a[xs..], lda);
            e[i - 1] = alpha.re;
            a[ix(i, i + 1, lda)] = ONE;
            let v: Vec<Cplx> = (i + 1..=n).map(|j| a[ix(i, j, lda)]).collect();
            zlarf(
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
            zlacgv(n - i, &mut a[ix(i, i + 1, lda)..], lda);
            a[ix(i, i + 1, lda)] = Cplx::from_real(e[i - 1]);
        } else {
            taup[i - 1] = ZERO;
        }
    }
    Ok(())
}

/// `ZGEBRD` (`SRC/zgebrd.f`): reduction to bidiagonal form. With
/// `NB = 32` and crossover `NX = 128` the blocked loop (lines 304-338) runs
/// only for `min(m, n) > 128`; otherwise the whole matrix goes to
/// [`zgebd2`].
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] for `lwork < max(1, m, n)`;
/// [`LapackError::UnportedPath`] for the blocked path.
#[allow(clippy::too_many_arguments)]
pub fn zgebrd(
    m: usize,
    n: usize,
    a: &mut [Cplx],
    lda: usize,
    d: &mut [f64],
    e: &mut [f64],
    tauq: &mut [Cplx],
    taup: &mut [Cplx],
    lwork: usize,
) -> Result<(), LapackError> {
    if lda < m.max(1) {
        return Err(bad("ZGEBRD", -4));
    } else if lwork < m.max(n).max(1) {
        return Err(bad("ZGEBRD", -10));
    }
    let minmn = m.min(n);
    if minmn == 0 {
        return Ok(());
    }
    // zgebrd.f:276-302: NX = MAX(NB, ILAENV(3)) = 128 when NB < MINMN
    let nx = if NB32 > 1 && NB32 < minmn {
        NB32.max(NX128)
    } else {
        minmn
    };
    if nx < minmn {
        return Err(unported("ZGEBRD", "284-338"));
    }
    let mut work = vec![ZERO; m.max(n)];
    zgebd2(m, n, a, lda, d, e, tauq, taup, &mut work)
}

/// `ZGEQRF(m, n, ..., LWORK = -1)` (`SRC/zgeqrf.f:180-203`): the optimal
/// workspace `WORK(1)` of the workspace query, `N*NB` (1 when
/// `min(m, n) = 0`). The factorisation itself is not ported (GYRE reaches
/// `ZGEQRF` only through `ZGESVD`'s workspace query).
#[must_use]
pub fn zgeqrf_lwkopt(m: usize, n: usize) -> usize {
    if m.min(n) == 0 {
        1
    } else {
        n * NB32
    }
}

/// `ZUNGQR(m, n, k, ..., LWORK = -1)` (`SRC/zungqr.f:166-168`):
/// `MAX(1, N)*NB`.
#[must_use]
pub fn zungqr_lwkopt(n: usize) -> usize {
    n.max(1) * NB32
}

/// `ZUNGLQ(m, n, k, ..., LWORK = -1)` (`SRC/zunglq.f:165-167`):
/// `MAX(1, M)*NB`.
#[must_use]
pub fn zunglq_lwkopt(m: usize) -> usize {
    m.max(1) * NB32
}

/// `ZUNGBR(vect, m, n, k, ..., LWORK = -1)` (`SRC/zungbr.f:215-237`).
#[must_use]
pub fn zungbr_lwkopt(vect: u8, m: usize, n: usize, k: usize) -> usize {
    let mut work1 = 1;
    if lsame(vect, b'Q') {
        if m >= k {
            work1 = zungqr_lwkopt(n);
        } else if m > 1 {
            work1 = zungqr_lwkopt(m - 1);
        }
    } else if k < n {
        work1 = zunglq_lwkopt(m);
    } else if n > 1 {
        work1 = zunglq_lwkopt(n - 1);
    }
    work1.max(m.min(n))
}

/// `ZUNG2R` (`SRC/zung2r.f`): form the `m x n` `Q` with orthonormal columns
/// from `k` reflectors. `work` needs `n` entries.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] as upstream's argument checks;
/// propagates `ZLARF` errors.
pub fn zung2r(
    m: usize,
    n: usize,
    k: usize,
    a: &mut [Cplx],
    lda: usize,
    tau: &[Cplx],
    work: &mut [Cplx],
) -> Result<(), LapackError> {
    if n > m {
        return Err(bad("ZUNG2R", -2));
    } else if k > n {
        return Err(bad("ZUNG2R", -3));
    } else if lda < m.max(1) {
        return Err(bad("ZUNG2R", -5));
    }
    if n == 0 {
        return Ok(());
    }
    // Columns k+1:n to columns of the unit matrix (zung2r.f:168-173)
    for j in k + 1..=n {
        for l in 1..=m {
            a[ix(l, j, lda)] = ZERO;
        }
        a[ix(j, j, lda)] = ONE;
    }
    // zung2r.f:175-193
    for i in (1..=k).rev() {
        if i < n {
            a[ix(i, i, lda)] = ONE;
            let v: Vec<Cplx> = a[ix(i, i, lda)..=ix(m, i, lda)].to_vec();
            zlarf(
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
            zscal(m - i, -tau[i - 1], &mut a[ix(i + 1, i, lda)..], 1);
        }
        a[ix(i, i, lda)] = ONE - tau[i - 1];
        for l in 1..i {
            a[ix(l, i, lda)] = ZERO;
        }
    }
    Ok(())
}

/// `ZUNGQR` (`SRC/zungqr.f`): unblocked ([`zung2r`]) unless
/// `k > NX = 128`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] for `lwork < max(1, n)` and as
/// [`zung2r`]; [`LapackError::UnportedPath`] for the blocked path
/// (`ZLARFT`/`ZLARFB`, lines 220-280).
#[allow(clippy::too_many_arguments)]
pub fn zungqr(
    m: usize,
    n: usize,
    k: usize,
    a: &mut [Cplx],
    lda: usize,
    tau: &[Cplx],
    lwork: usize,
) -> Result<(), LapackError> {
    if n > m {
        return Err(bad("ZUNGQR", -2));
    } else if k > n {
        return Err(bad("ZUNGQR", -3));
    } else if lda < m.max(1) {
        return Err(bad("ZUNGQR", -5));
    } else if lwork < n.max(1) {
        return Err(bad("ZUNGQR", -8));
    }
    if n == 0 {
        return Ok(());
    }
    // zungqr.f:195-218: NX = ILAENV(3) = 128 when NB < K
    if NB32 > 1 && NB32 < k && NX128 < k {
        return Err(unported("ZUNGQR", "220-280"));
    }
    let mut work = vec![ZERO; n];
    zung2r(m, n, k, a, lda, tau, &mut work)
}

/// `ZUNGL2` (`SRC/zungl2.f`): form the `m x n` `Q` with orthonormal rows
/// from `k` reflectors. `work` needs `m` entries.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] as upstream's argument checks;
/// propagates `ZLARF` errors.
pub fn zungl2(
    m: usize,
    n: usize,
    k: usize,
    a: &mut [Cplx],
    lda: usize,
    tau: &[Cplx],
    work: &mut [Cplx],
) -> Result<(), LapackError> {
    if n < m {
        return Err(bad("ZUNGL2", -2));
    } else if k > m {
        return Err(bad("ZUNGL2", -3));
    } else if lda < m.max(1) {
        return Err(bad("ZUNGL2", -5));
    }
    if m == 0 {
        return Ok(());
    }
    if k < m {
        // Rows k+1:m to rows of the unit matrix (zungl2.f:165-176)
        for j in 1..=n {
            for l in k + 1..=m {
                a[ix(l, j, lda)] = ZERO;
            }
            if j > k && j <= m {
                a[ix(j, j, lda)] = ONE;
            }
        }
    }
    // zungl2.f:178-199
    for i in (1..=k).rev() {
        if i < n {
            zlacgv(n - i, &mut a[ix(i, i + 1, lda)..], lda);
            if i < m {
                a[ix(i, i, lda)] = ONE;
                let v: Vec<Cplx> = (i..=n).map(|j| a[ix(i, j, lda)]).collect();
                zlarf(
                    b'R',
                    m - i,
                    n - i + 1,
                    &v,
                    1,
                    tau[i - 1].conj(),
                    &mut a[ix(i + 1, i, lda)..],
                    lda,
                    work,
                )?;
            }
            zscal(n - i, -tau[i - 1], &mut a[ix(i, i + 1, lda)..], lda);
            zlacgv(n - i, &mut a[ix(i, i + 1, lda)..], lda);
        }
        a[ix(i, i, lda)] = ONE - tau[i - 1].conj();
        for l in 1..i {
            a[ix(i, l, lda)] = ZERO;
        }
    }
    Ok(())
}

/// `ZUNGLQ` (`SRC/zunglq.f`): unblocked ([`zungl2`]) unless
/// `k > NX = 128`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] for `lwork < max(1, m)` and as
/// [`zungl2`]; [`LapackError::UnportedPath`] for the blocked path
/// (`ZLARFT`/`ZLARFB`, lines 219-279).
#[allow(clippy::too_many_arguments)]
pub fn zunglq(
    m: usize,
    n: usize,
    k: usize,
    a: &mut [Cplx],
    lda: usize,
    tau: &[Cplx],
    lwork: usize,
) -> Result<(), LapackError> {
    if n < m {
        return Err(bad("ZUNGLQ", -2));
    } else if k > m {
        return Err(bad("ZUNGLQ", -3));
    } else if lda < m.max(1) {
        return Err(bad("ZUNGLQ", -5));
    } else if lwork < m.max(1) {
        return Err(bad("ZUNGLQ", -8));
    }
    if m == 0 {
        return Ok(());
    }
    // zunglq.f:194-217: NX = ILAENV(3) = 128 when NB < K
    if NB32 > 1 && NB32 < k && NX128 < k {
        return Err(unported("ZUNGLQ", "219-279"));
    }
    let mut work = vec![ZERO; m];
    zungl2(m, n, k, a, lda, tau, &mut work)
}

/// `ZUNGBR` (`SRC/zungbr.f`): generate `Q` (`vect = 'Q'`) or `P**H`
/// (`'P'`) from `ZGEBRD`'s reflectors.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] as upstream's argument checks;
/// propagates [`zungqr`]/[`zunglq`] errors.
#[allow(clippy::too_many_arguments)]
pub fn zungbr(
    vect: u8,
    m: usize,
    n: usize,
    k: usize,
    a: &mut [Cplx],
    lda: usize,
    tau: &[Cplx],
    lwork: usize,
) -> Result<(), LapackError> {
    let wantq = lsame(vect, b'Q');
    let mn = m.min(n);
    if !wantq && !lsame(vect, b'P') {
        return Err(bad("ZUNGBR", -1));
    } else if (wantq && (n > m || n < m.min(k)))
        || (!wantq && (m > n || m < n.min(k)))
    {
        return Err(bad("ZUNGBR", -3));
    } else if lda < m.max(1) {
        return Err(bad("ZUNGBR", -6));
    } else if lwork < mn.max(1) {
        return Err(bad("ZUNGBR", -9));
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    if wantq {
        if m >= k {
            // zungbr.f:260-264
            zungqr(m, n, k, a, lda, tau, lwork)
        } else {
            // Shift the reflector vectors one column right, first row and
            // column of Q to the unit vector (zungbr.f:274-290)
            for j in (2..=m).rev() {
                a[ix(1, j, lda)] = ZERO;
                for i in j + 1..=m {
                    a[ix(i, j, lda)] = a[ix(i, j - 1, lda)];
                }
            }
            a[ix(1, 1, lda)] = ONE;
            for i in 2..=m {
                a[ix(i, 1, lda)] = ZERO;
            }
            if m > 1 {
                let off = ix(2, 2, lda);
                zungqr(m - 1, m - 1, m - 1, &mut a[off..], lda, tau, lwork)?;
            }
            Ok(())
        }
    } else if k < n {
        // zungbr.f:297-301
        zunglq(m, n, k, a, lda, tau, lwork)
    } else {
        // Shift the reflector vectors one row down, first row and column
        // of P**H to the unit vector (zungbr.f:311-327)
        a[ix(1, 1, lda)] = ONE;
        for i in 2..=n {
            a[ix(i, 1, lda)] = ZERO;
        }
        for j in 2..=n {
            for i in (2..j).rev() {
                a[ix(i, j, lda)] = a[ix(i - 1, j, lda)];
            }
            a[ix(1, j, lda)] = ZERO;
        }
        if n > 1 {
            let off = ix(2, 2, lda);
            zunglq(n - 1, n - 1, n - 1, &mut a[off..], lda, tau, lwork)?;
        }
        Ok(())
    }
}

/// `ZBDSQR` (`SRC/zbdsqr.f`) with singular vectors (`ncvt > 0` or
/// `nru > 0`) and `ncc = 0`: SVD of a real bidiagonal matrix by implicit
/// zero-shift / shifted QR, applying the rotations to the complex `VT`
/// (`n x ncvt`) and `U` (`nru x n`). `uplo = 'U'` or `'L'`. `d`
/// (length `n`) receives the singular values in decreasing order, `e`
/// (length `n-1`) is destroyed. Returns `INFO` (`> 0`: number of
/// unconverged superdiagonals).
///
/// **SDK difference.** As for `DBDSQR` ([`super::svd::dbdsqr`]), the MESA
/// SDK `crmath`-slot object evaluates
/// `TOLMUL = MAX(TEN, MIN(HNDRD, EPS**MEIGTH))` (`zbdsqr.f:374`) as
/// `MAX(TEN, MIN(HNDRD, exp(0*log(EPS))))`, so `TOLMUL = 10`, not netlib's
/// `EPS**(-1/8)`. This port reproduces the SDK.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] for an invalid `uplo`;
/// [`LapackError::UnportedPath`] for no singular vectors (`DLASQ1`, lines
/// 328-335).
#[allow(clippy::too_many_arguments)]
pub fn zbdsqr(
    uplo: u8,
    n: usize,
    ncvt: usize,
    nru: usize,
    d: &mut [f64],
    e: &mut [f64],
    vt: &mut [Cplx],
    ldvt: usize,
    u: &mut [Cplx],
    ldu: usize,
) -> Result<usize, LapackError> {
    zbdsqr_tol(uplo, n, ncvt, nru, d, e, vt, ldvt, u, ldu, None)
}

#[allow(
    clippy::too_many_arguments,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]
fn zbdsqr_tol(
    uplo: u8,
    n: usize,
    ncvt: usize,
    nru: usize,
    d: &mut [f64],
    e: &mut [f64],
    vt: &mut [Cplx],
    ldvt: usize,
    u: &mut [Cplx],
    ldu: usize,
    tolmul_override: Option<f64>,
) -> Result<usize, LapackError> {
    const HNDRTH: f64 = 0.01;
    const TEN: f64 = 10.0;
    const HNDRD: f64 = 100.0;
    const MAXITR: usize = 6;
    let lower = lsame(uplo, b'L');
    if !lsame(uplo, b'U') && !lower {
        return Err(bad("ZBDSQR", -1));
    }
    if n == 0 {
        return Ok(0);
    }
    let mut info = 0;
    if n > 1 {
        let rotate = ncvt > 0 || nru > 0;
        if !rotate {
            return Err(unported("ZBDSQR", "328-335"));
        }
        let nm1 = n - 1;
        let nm12 = nm1 + nm1;
        let nm13 = nm12 + nm1;
        let mut rwork = vec![0.0_f64; 4 * nm1];
        let mut idir = 0;
        let eps = dlamch::EPS;
        let unfl = dlamch::SFMIN;
        if lower {
            // Rotate to upper bidiagonal (zbdsqr.f:350-368)
            for i in 1..n {
                let (cs, sn, r) = dlartg(d[i - 1], e[i - 1]);
                d[i - 1] = r;
                e[i - 1] = sn * d[i];
                d[i] *= cs;
                rwork[i - 1] = cs;
                rwork[nm1 + i - 1] = sn;
            }
            if nru > 0 {
                zlasr(b'R', b'V', b'F', nru, n, &rwork, &rwork[nm1..], u, ldu)?;
            }
        }
        // SDK: EPS**MEIGTH evaluated as exp(0*log(EPS)) (see doc comment)
        let tolmul = tolmul_override.unwrap_or_else(|| {
            fmax(
                TEN,
                fmin(HNDRD, flexmesa_math::exp(0.0 * flexmesa_math::log(eps))),
            )
        });
        let tol = tolmul * eps;
        // zbdsqr.f:379-409
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
        // Main loop (label 60, zbdsqr.f:427)
        'l60: loop {
            if m <= 1 {
                break 'l60;
            }
            if iter >= ni {
                iter -= ni;
                iterdivn += 1;
                if iterdivn >= maxitdivn {
                    // Failure to converge (label 200, zbdsqr.f:841-846)
                    info = 0;
                    for &ei in e.iter().take(n - 1) {
                        if ei != 0.0 {
                            info += 1;
                        }
                    }
                    return Ok(info);
                }
            }
            // Find diagonal block (zbdsqr.f:442-468)
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
                // 2x2 block (zbdsqr.f:474-495)
                let r = dlasv2(d[m - 2], e[m - 2], d[m - 1]);
                d[m - 2] = r.ssmax;
                e[m - 2] = 0.0;
                d[m - 1] = r.ssmin;
                if ncvt > 0 {
                    zdrot_in(
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
                    zdrot_in(
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
            // Convergence tests (zbdsqr.f:517-572)
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
            // Shift (zbdsqr.f:579-603)
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
                    // Zero-shift QR, top to bottom (zbdsqr.f:617-649)
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
                        rwork[w(i - ll + 1, 0)] = cs;
                        rwork[w(i - ll + 1, nm1)] = sn;
                        rwork[w(i - ll + 1, nm12)] = oldcs;
                        rwork[w(i - ll + 1, nm13)] = oldsn;
                    }
                    let h = d[m - 1] * cs;
                    d[m - 1] = h * oldcs;
                    e[m - 2] = h * oldsn;
                    apply_rotations(
                        true,
                        m - ll + 1,
                        ncvt,
                        nru,
                        &rwork,
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
                    // Zero-shift QR, bottom to top (zbdsqr.f:655-687)
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
                        rwork[w(i - ll, 0)] = cs;
                        rwork[w(i - ll, nm1)] = -sn;
                        rwork[w(i - ll, nm12)] = oldcs;
                        rwork[w(i - ll, nm13)] = -oldsn;
                    }
                    let h = d[ll - 1] * cs;
                    d[ll - 1] = h * oldcs;
                    e[ll - 1] = h * oldsn;
                    apply_rotations(
                        false,
                        m - ll + 1,
                        ncvt,
                        nru,
                        &rwork,
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
                // Shifted QR, top to bottom (zbdsqr.f:697-739)
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
                    rwork[w(i - ll + 1, 0)] = cosr;
                    rwork[w(i - ll + 1, nm1)] = sinr;
                    rwork[w(i - ll + 1, nm12)] = cosl;
                    rwork[w(i - ll + 1, nm13)] = sinl;
                }
                e[m - 2] = f;
                apply_rotations(
                    true,
                    m - ll + 1,
                    ncvt,
                    nru,
                    &rwork,
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
                // Shifted QR, bottom to top (zbdsqr.f:745-788)
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
                    rwork[w(i - ll, 0)] = cosr;
                    rwork[w(i - ll, nm1)] = -sinr;
                    rwork[w(i - ll, nm12)] = cosl;
                    rwork[w(i - ll, nm13)] = -sinl;
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
                    &rwork,
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
    // label 160: make singular values positive, then sort (zbdsqr.f:796-836)
    for i in 1..=n {
        if d[i - 1] < 0.0 {
            d[i - 1] = -d[i - 1];
            if ncvt > 0 {
                zdscal(ncvt, -1.0, &mut vt[ix(i, 1, ldvt)..], ldvt);
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
                zswap_in(
                    ncvt,
                    vt,
                    ix(isub, 1, ldvt),
                    ldvt,
                    ix(n + 1 - i, 1, ldvt),
                    ldvt,
                );
            }
            if nru > 0 {
                zswap_in(nru, u, ix(1, isub, ldu), 1, ix(1, n + 1 - i, ldu), 1);
            }
        }
    }
    Ok(info)
}

/// The `ZLASR` updates of `VT` and `U` after a QR sweep (forward: rows of
/// `VT` with `RWORK(1)`, `RWORK(N)`; columns of `U` with `RWORK(NM12+1)`,
/// `RWORK(NM13+1)`; backward: the roles swap).
#[allow(clippy::too_many_arguments)]
fn apply_rotations(
    forward: bool,
    len: usize,
    ncvt: usize,
    nru: usize,
    rwork: &[f64],
    nm1: usize,
    nm12: usize,
    nm13: usize,
    vt: &mut [Cplx],
    ldvt: usize,
    u: &mut [Cplx],
    ldu: usize,
    ll: usize,
) -> Result<(), LapackError> {
    // RWORK(N) is RWORK(NM1+1)
    let (vc, vs, uc, us, dir) = if forward {
        (0, nm1, nm12, nm13, b'F')
    } else {
        (nm12, nm13, 0, nm1, b'B')
    };
    if ncvt > 0 {
        zlasr(
            b'L',
            b'V',
            dir,
            len,
            ncvt,
            &rwork[vc..],
            &rwork[vs..],
            &mut vt[ix(ll, 1, ldvt)..],
            ldvt,
        )?;
    }
    if nru > 0 {
        zlasr(
            b'R',
            b'V',
            dir,
            nru,
            len,
            &rwork[uc..],
            &rwork[us..],
            &mut u[ix(1, ll, ldu)..],
            ldu,
        )?;
    }
    Ok(())
}

/// Output of [`zgesvd`].
#[derive(Debug, Clone, PartialEq)]
pub struct Zgesvd {
    /// Singular values, decreasing.
    pub s: Vec<f64>,
    /// `WORK(1)` on exit (`MAXWRK`, imaginary part 0).
    pub work1: f64,
    /// `INFO` from `ZBDSQR` (`> 0`: unconverged superdiagonals).
    pub info: usize,
}

/// `ZGESVD` (`SRC/zgesvd.f`), `JOBU = JOBVT = 'A'`, `m >= n` with
/// `m < MNTHR` (path 10, the square case GYRE uses): `A = U*S*VT`. `A` is
/// destroyed (it holds `ZGEBRD`'s output), `U` is `m x m`, `VT` is
/// `n x n` (leading dimensions `m` and `n`). `lwork` is the complex
/// workspace size the caller declares (GYRE passes `3*n`); it enters the
/// size checks and the workspace passed down. The real workspace
/// `RWORK(5*min(m, n))` is allocated here.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] when `lwork` is below the minimum
/// `2*n + m`; [`LapackError::UnportedPath`] for other shapes (`m < n`,
/// lines 2167-3677; `m >= MNTHR`, lines 661-2055) and for the rescaling of
/// matrices with norms outside `[SMLNUM, BIGNUM]` (`ZLASCL`/`DLASCL`, lines
/// 646-653, 3679-3692).
pub fn zgesvd(
    m: usize,
    n: usize,
    a: &mut [Cplx],
    u: &mut [Cplx],
    vt: &mut [Cplx],
    lwork: usize,
) -> Result<Zgesvd, LapackError> {
    zgesvd_tol(m, n, a, u, vt, lwork, None)
}

/// [`zgesvd`] with `ZBDSQR`'s `TOLMUL` replaced. Only for the negative
/// control that shows the SDK's `TOLMUL = 10` is observable in the
/// vectors; not part of the port.
///
/// # Errors
///
/// As [`zgesvd`].
#[doc(hidden)]
pub fn zgesvd_with_tolmul(
    m: usize,
    n: usize,
    a: &mut [Cplx],
    u: &mut [Cplx],
    vt: &mut [Cplx],
    lwork: usize,
    tolmul: f64,
) -> Result<Zgesvd, LapackError> {
    zgesvd_tol(m, n, a, u, vt, lwork, Some(tolmul))
}

/// [`zbdsqr`] with `TOLMUL` replaced (negative control only).
///
/// # Errors
///
/// As [`zbdsqr`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn zbdsqr_with_tolmul(
    uplo: u8,
    n: usize,
    ncvt: usize,
    nru: usize,
    d: &mut [f64],
    e: &mut [f64],
    vt: &mut [Cplx],
    ldvt: usize,
    u: &mut [Cplx],
    ldu: usize,
    tolmul: f64,
) -> Result<usize, LapackError> {
    zbdsqr_tol(uplo, n, ncvt, nru, d, e, vt, ldvt, u, ldu, Some(tolmul))
}

#[allow(clippy::many_single_char_names)]
fn zgesvd_tol(
    m: usize,
    n: usize,
    a: &mut [Cplx],
    u: &mut [Cplx],
    vt: &mut [Cplx],
    lwork: usize,
    tolmul: Option<f64>,
) -> Result<Zgesvd, LapackError> {
    let lda = m.max(1);
    let ldu = m.max(1);
    let ldvt = n.max(1);
    let minmn = m.min(n);
    if minmn == 0 {
        // zgesvd.f:625-627 (WORK(1) = MAXWRK = MINWRK = 1)
        return Ok(Zgesvd {
            s: Vec::new(),
            work1: 1.0,
            info: 0,
        });
    }
    if m < n {
        return Err(unported("ZGESVD", "2167-3677"));
    }
    let mnthr = ilaenv_mnthr(m, n);
    if m >= mnthr {
        return Err(unported("ZGESVD", "661-2055"));
    }
    // Workspace bookkeeping, path 10 (zgesvd.f:313-343, 441-465, 615-616)
    let lwork_zgebrd = (m + n) * NB32;
    let mut maxwrk = 2 * n + lwork_zgebrd;
    let lwork_zungbr_q = zungbr_lwkopt(b'Q', m, m, n);
    maxwrk = maxwrk.max(2 * n + lwork_zungbr_q);
    let lwork_zungbr_p = zungbr_lwkopt(b'P', n, n, n);
    maxwrk = maxwrk.max(2 * n + lwork_zungbr_p);
    let minwrk = 2 * n + m;
    maxwrk = maxwrk.max(minwrk);
    if lwork < minwrk {
        return Err(bad("ZGESVD", -13));
    }
    // Machine constants and scaling test (zgesvd.f:640-653)
    let eps = dlamch::PREC;
    let smlnum = dlamch::SFMIN.sqrt() / eps;
    let bignum = 1.0 / smlnum;
    let anrm = zlange(b'M', m, n, a, lda)?;
    if (anrm > 0.0 && anrm < smlnum) || anrm > bignum {
        return Err(unported("ZGESVD", "646-653, 3679-3692"));
    }
    // Path 10 (zgesvd.f:2056-2134): bidiagonalise, form Q and P**H, QR
    let itauq = 1;
    let itaup = itauq + n;
    let iwork = itaup + n;
    let lw = lwork + 1 - iwork;
    let mut s = vec![0.0_f64; n];
    // RWORK(IE), IE = 1: the superdiagonal of B
    let mut e = vec![0.0_f64; n];
    let mut tauq = vec![ZERO; n];
    let mut taup = vec![ZERO; n];
    zgebrd(m, n, a, lda, &mut s, &mut e, &mut tauq, &mut taup, lw)?;
    zlacpy(b'L', m, n, a, lda, u, ldu);
    zungbr(b'Q', m, m, n, u, ldu, &tauq, lw)?;
    zlacpy(b'U', n, n, a, lda, vt, ldvt);
    zungbr(b'P', n, n, n, vt, ldvt, &taup, lw)?;
    let info =
        zbdsqr_tol(b'U', n, n, m, &mut s, &mut e, vt, ldvt, u, ldu, tolmul)?;
    #[allow(clippy::cast_precision_loss)]
    let work1 = maxwrk as f64;
    Ok(Zgesvd { s, work1, info })
}
