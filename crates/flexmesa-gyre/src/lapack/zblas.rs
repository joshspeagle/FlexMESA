// SPDX-License-Identifier: GPL-3.0-only
//! Reference BLAS 3.12.0 complex routines (`BLAS/SRC/*.f`, `dznrm2.f90`),
//! double precision, as the MESA SDK `26.6.1` builds them
//! (`gfortran -O2 -fPIC`). Every routine here is instruction-identical to
//! that build of the netlib source except `ZDSCAL`, whose `DCMPLX` is
//! CRMATH's `dcmplx` (a plain constructor, so the arithmetic is unchanged).
//!
//! Complex arithmetic follows gfortran's inline expansion
//! ([`crate::fcomplex`]): a real operand of a complex product is widened to
//! `(r, +0)` and the full product is kept ([`Cplx::real_mul`]).
//!
//! Vectors are slices starting at `X(1)` with a positive increment.

use super::{ix, lsame, LapackError};
use crate::fcomplex::Cplx;

const ZERO: Cplx = Cplx::new(0.0, 0.0);
const ONE: Cplx = Cplx::new(1.0, 0.0);

fn bad(routine: &'static str, info: i32) -> LapackError {
    LapackError::IllegalArgument { routine, info }
}

fn unported(routine: &'static str, lines: &'static str) -> LapackError {
    LapackError::UnportedPath { routine, lines }
}

/// `DCABS1` (`BLAS/SRC/dcabs1.f:61`): `|Re z| + |Im z|`.
#[must_use]
#[inline]
pub fn dcabs1(z: Cplx) -> f64 {
    z.re.abs() + z.im.abs()
}

/// `DZASUM` (`BLAS/SRC/dzasum.f`): sum of `|Re z| + |Im z|`.
#[must_use]
pub fn dzasum(n: usize, zx: &[Cplx], incx: usize) -> f64 {
    if n == 0 || incx == 0 {
        return 0.0;
    }
    let mut stemp = 0.0_f64;
    for i in 0..n {
        stemp += dcabs1(zx[i * incx]);
    }
    stemp
}

/// `IZAMAX` (`BLAS/SRC/izamax.f`): 1-based index of the first maximal
/// [`dcabs1`]; 0 for `n == 0` or `incx == 0`.
#[must_use]
pub fn izamax(n: usize, zx: &[Cplx], incx: usize) -> usize {
    if n == 0 || incx == 0 {
        return 0;
    }
    let mut imax = 1;
    if n == 1 {
        return 1;
    }
    // izamax.f:101-107 (incx = 1) and :112-121 visit the same elements
    let mut dmax = dcabs1(zx[0]);
    for i in 2..=n {
        let v = dcabs1(zx[(i - 1) * incx]);
        if v > dmax {
            imax = i;
            dmax = v;
        }
    }
    imax
}

/// `DZNRM2` (`BLAS/SRC/dznrm2.f90`, Blue's algorithm, Anderson 2017):
/// Euclidean norm of a complex vector, the real and imaginary parts of each
/// element accumulated in that order.
#[must_use]
pub fn dznrm2(n: usize, x: &[Cplx], incx: usize) -> f64 {
    // Blue's scaling constants for binary64 (dznrm2.f90:95-102)
    const TSML: f64 = f64::from_bits(0x2000_0000_0000_0000); // 2**-511
    const TBIG: f64 = f64::from_bits(0x5E50_0000_0000_0000); // 2**486
    const SSML: f64 = f64::from_bits(0x6180_0000_0000_0000); // 2**537
    const SBIG: f64 = f64::from_bits(0x1E50_0000_0000_0000); // 2**-538
    const MAXN: f64 = f64::MAX;
    if n == 0 {
        return 0.0;
    }
    let mut notbig = true;
    let mut asml = 0.0_f64;
    let mut amed = 0.0_f64;
    let mut abig = 0.0_f64;
    let mut acc = |ax: f64| {
        if ax > TBIG {
            let t = ax * SBIG;
            abig += t * t;
            notbig = false;
        } else if ax < TSML {
            if notbig {
                let t = ax * SSML;
                asml += t * t;
            }
        } else {
            amed += ax * ax;
        }
    };
    // dznrm2.f90:144-165
    for i in 0..n {
        let z = x[i * incx];
        acc(z.re.abs());
        acc(z.im.abs());
    }
    let (scl, sumsq);
    // dznrm2.f90:169-201
    if abig > 0.0 {
        #[allow(clippy::eq_op)]
        if amed > 0.0 || amed > MAXN || amed != amed {
            abig += (amed * SBIG) * SBIG;
        }
        scl = 1.0 / SBIG;
        sumsq = abig;
    } else if asml > 0.0 {
        #[allow(clippy::eq_op)]
        if amed > 0.0 || amed > MAXN || amed != amed {
            let amed2 = amed.sqrt();
            let asml2 = asml.sqrt() / SSML;
            let (ymin, ymax) = if asml2 > amed2 {
                (amed2, asml2)
            } else {
                (asml2, amed2)
            };
            scl = 1.0;
            let r = ymin / ymax;
            sumsq = (ymax * ymax) * (1.0 + r * r);
        } else {
            scl = 1.0 / SSML;
            sumsq = asml;
        }
    } else {
        scl = 1.0;
        sumsq = amed;
    }
    scl * sumsq.sqrt()
}

/// `ZSCAL` (`BLAS/SRC/zscal.f`): `x = za*x` (returns early when
/// `za == (1, 0)`).
pub fn zscal(n: usize, za: Cplx, zx: &mut [Cplx], incx: usize) {
    if n == 0 || incx == 0 || za.feq(ONE) {
        return;
    }
    for i in 0..n {
        zx[i * incx] = za * zx[i * incx];
    }
}

/// `ZDSCAL` (`BLAS/SRC/zdscal.f`, SDK CRMATH build): `x = da*x` with real
/// `da`, `DCMPLX(da*Re x, da*Im x)` (zdscal.f:108); returns early when
/// `da == 1`.
pub fn zdscal(n: usize, da: f64, zx: &mut [Cplx], incx: usize) {
    if n == 0 || incx == 0 || da == 1.0 {
        return;
    }
    for i in 0..n {
        let z = zx[i * incx];
        zx[i * incx] = Cplx::new(da * z.re, da * z.im);
    }
}

/// `ZDROT` (`BLAS/SRC/zdrot.f`) on two vectors of one array (`x` at
/// offset `ox`, `y` at offset `oy`): `x' = c*x + s*y`, `y' = c*y - s*x`
/// with real `c`, `s` (zdrot.f:126-128).
pub fn zdrot_in(
    n: usize,
    a: &mut [Cplx],
    ox: usize,
    incx: usize,
    oy: usize,
    incy: usize,
    c: f64,
    s: f64,
) {
    for i in 0..n {
        let px = ox + i * incx;
        let py = oy + i * incy;
        let ctemp = Cplx::real_mul(c, a[px]) + Cplx::real_mul(s, a[py]);
        a[py] = Cplx::real_mul(c, a[py]) - Cplx::real_mul(s, a[px]);
        a[px] = ctemp;
    }
}

/// `ZSWAP` (`BLAS/SRC/zswap.f`) on two vectors of one array.
pub fn zswap_in(
    n: usize,
    a: &mut [Cplx],
    ox: usize,
    incx: usize,
    oy: usize,
    incy: usize,
) {
    for i in 0..n {
        a.swap(ox + i * incx, oy + i * incy);
    }
}

/// `ZCOPY` (`BLAS/SRC/zcopy.f`).
pub fn zcopy(
    n: usize,
    zx: &[Cplx],
    incx: usize,
    zy: &mut [Cplx],
    incy: usize,
) {
    for i in 0..n {
        zy[i * incy] = zx[i * incx];
    }
}

/// `ZAXPY` (`BLAS/SRC/zaxpy.f`): `y = y + za*x`.
pub fn zaxpy(
    n: usize,
    za: Cplx,
    zx: &[Cplx],
    incx: usize,
    zy: &mut [Cplx],
    incy: usize,
) {
    if n == 0 || dcabs1(za) == 0.0 {
        return;
    }
    for i in 0..n {
        zy[i * incy] = zy[i * incy] + za * zx[i * incx];
    }
}

/// `ZDOTC` (`BLAS/SRC/zdotc.f`): `conj(x)^T * y`.
#[must_use]
pub fn zdotc(
    n: usize,
    zx: &[Cplx],
    incx: usize,
    zy: &[Cplx],
    incy: usize,
) -> Cplx {
    let mut ztemp = ZERO;
    for i in 0..n {
        ztemp = ztemp + zx[i * incx].conj() * zy[i * incy];
    }
    ztemp
}

/// `ZDOTU` (`BLAS/SRC/zdotu.f`): `x^T * y`.
#[must_use]
pub fn zdotu(
    n: usize,
    zx: &[Cplx],
    incx: usize,
    zy: &[Cplx],
    incy: usize,
) -> Cplx {
    let mut ztemp = ZERO;
    for i in 0..n {
        ztemp = ztemp + zx[i * incx] * zy[i * incy];
    }
    ztemp
}

/// `ZGEMV` (`BLAS/SRC/zgemv.f`): `y = alpha*op(A)*x + beta*y`,
/// `op(A) = A`, `A**T` or `A**H`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] as upstream's argument checks.
#[allow(clippy::many_single_char_names)]
pub fn zgemv(
    trans: u8,
    m: usize,
    n: usize,
    alpha: Cplx,
    a: &[Cplx],
    lda: usize,
    x: &[Cplx],
    incx: usize,
    beta: Cplx,
    y: &mut [Cplx],
    incy: usize,
) -> Result<(), LapackError> {
    if !lsame(trans, b'N') && !lsame(trans, b'T') && !lsame(trans, b'C') {
        return Err(bad("ZGEMV", 1));
    } else if lda < m.max(1) {
        return Err(bad("ZGEMV", 6));
    } else if incx == 0 {
        return Err(bad("ZGEMV", 8));
    } else if incy == 0 {
        return Err(bad("ZGEMV", 11));
    }
    if m == 0 || n == 0 || (alpha.is_zero() && beta.feq(ONE)) {
        return Ok(());
    }
    let noconj = lsame(trans, b'T');
    let notrans = lsame(trans, b'N');
    let leny = if notrans { m } else { n };
    // y := beta*y (zgemv.f:253-278)
    if !beta.feq(ONE) {
        if beta.is_zero() {
            for i in 0..leny {
                y[i * incy] = ZERO;
            }
        } else {
            for i in 0..leny {
                y[i * incy] = beta * y[i * incy];
            }
        }
    }
    if alpha.is_zero() {
        return Ok(());
    }
    if notrans {
        // y := alpha*A*x + y (zgemv.f:284-303)
        for j in 1..=n {
            let temp = alpha * x[(j - 1) * incx];
            for i in 1..=m {
                y[(i - 1) * incy] = y[(i - 1) * incy] + temp * a[ix(i, j, lda)];
            }
        }
    } else {
        // y := alpha*A**T*x + y or alpha*A**H*x + y (zgemv.f:308-342)
        for j in 1..=n {
            let mut temp = ZERO;
            if noconj {
                for i in 1..=m {
                    temp = temp + a[ix(i, j, lda)] * x[(i - 1) * incx];
                }
            } else {
                for i in 1..=m {
                    temp = temp + a[ix(i, j, lda)].conj() * x[(i - 1) * incx];
                }
            }
            y[(j - 1) * incy] = y[(j - 1) * incy] + alpha * temp;
        }
    }
    Ok(())
}

/// `ZGERC` (`BLAS/SRC/zgerc.f`): `A = alpha*x*y**H + A`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] as upstream's argument checks.
pub fn zgerc(
    m: usize,
    n: usize,
    alpha: Cplx,
    x: &[Cplx],
    incx: usize,
    y: &[Cplx],
    incy: usize,
    a: &mut [Cplx],
    lda: usize,
) -> Result<(), LapackError> {
    if incx == 0 {
        return Err(bad("ZGERC", 5));
    } else if incy == 0 {
        return Err(bad("ZGERC", 7));
    } else if lda < m.max(1) {
        return Err(bad("ZGERC", 9));
    }
    if m == 0 || n == 0 || alpha.is_zero() {
        return Ok(());
    }
    // zgerc.f:191-218
    for j in 1..=n {
        let yj = y[(j - 1) * incy];
        if !yj.is_zero() {
            let temp = alpha * yj.conj();
            for i in 1..=m {
                a[ix(i, j, lda)] = a[ix(i, j, lda)] + x[(i - 1) * incx] * temp;
            }
        }
    }
    Ok(())
}

/// `ZGEMM` (`BLAS/SRC/zgemm.f`), `TRANSA = TRANSB = 'N'` only:
/// `C = alpha*A*B + beta*C`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] as upstream; [`LapackError::UnportedPath`]
/// for transposed or conjugated operands (lines 321-472).
#[allow(clippy::many_single_char_names)]
pub fn zgemm(
    transa: u8,
    transb: u8,
    m: usize,
    n: usize,
    k: usize,
    alpha: Cplx,
    a: &[Cplx],
    lda: usize,
    b: &[Cplx],
    ldb: usize,
    beta: Cplx,
    c: &mut [Cplx],
    ldc: usize,
) -> Result<(), LapackError> {
    let nota = lsame(transa, b'N');
    let notb = lsame(transb, b'N');
    let conja = lsame(transa, b'C');
    let conjb = lsame(transb, b'C');
    if !nota && !conja && !lsame(transa, b'T') {
        return Err(bad("ZGEMM", 1));
    } else if !notb && !conjb && !lsame(transb, b'T') {
        return Err(bad("ZGEMM", 2));
    }
    if !(nota && notb) {
        return Err(unported("ZGEMM", "321-472"));
    }
    if lda < m.max(1) {
        return Err(bad("ZGEMM", 8));
    } else if ldb < k.max(1) {
        return Err(bad("ZGEMM", 10));
    } else if ldc < m.max(1) {
        return Err(bad("ZGEMM", 13));
    }
    if m == 0 || n == 0 || ((alpha.is_zero() || k == 0) && beta.feq(ONE)) {
        return Ok(());
    }
    if alpha.is_zero() {
        // zgemm.f:280-295
        for j in 1..=n {
            for i in 1..=m {
                c[ix(i, j, ldc)] = if beta.is_zero() {
                    ZERO
                } else {
                    beta * c[ix(i, j, ldc)]
                };
            }
        }
        return Ok(());
    }
    // C := alpha*A*B + beta*C (zgemm.f:304-320)
    for j in 1..=n {
        if beta.is_zero() {
            for i in 1..=m {
                c[ix(i, j, ldc)] = ZERO;
            }
        } else if !beta.feq(ONE) {
            for i in 1..=m {
                c[ix(i, j, ldc)] = beta * c[ix(i, j, ldc)];
            }
        }
        for l in 1..=k {
            let temp = alpha * b[ix(l, j, ldb)];
            for i in 1..=m {
                c[ix(i, j, ldc)] = c[ix(i, j, ldc)] + temp * a[ix(i, l, lda)];
            }
        }
    }
    Ok(())
}

/// `ZTRSM` (`BLAS/SRC/ztrsm.f`), `SIDE = 'L'` only: solve
/// `op(A)*X = alpha*B` (`op(A) = A`, `A**T` or `A**H`), overwriting `B`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] as upstream; [`LapackError::UnportedPath`]
/// for `SIDE = 'R'` (lines 354-468).
#[allow(clippy::many_single_char_names)]
pub fn ztrsm(
    side: u8,
    uplo: u8,
    transa: u8,
    diag: u8,
    m: usize,
    n: usize,
    alpha: Cplx,
    a: &[Cplx],
    lda: usize,
    b: &mut [Cplx],
    ldb: usize,
) -> Result<(), LapackError> {
    let lside = lsame(side, b'L');
    let nrowa = if lside { m } else { n };
    let noconj = lsame(transa, b'T');
    let nounit = lsame(diag, b'N');
    let upper = lsame(uplo, b'U');
    if !lside && !lsame(side, b'R') {
        return Err(bad("ZTRSM", 1));
    } else if !upper && !lsame(uplo, b'L') {
        return Err(bad("ZTRSM", 2));
    } else if !lsame(transa, b'N')
        && !lsame(transa, b'T')
        && !lsame(transa, b'C')
    {
        return Err(bad("ZTRSM", 3));
    } else if !lsame(diag, b'U') && !lsame(diag, b'N') {
        return Err(bad("ZTRSM", 4));
    } else if lda < nrowa.max(1) {
        return Err(bad("ZTRSM", 9));
    } else if ldb < m.max(1) {
        return Err(bad("ZTRSM", 11));
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    if alpha.is_zero() {
        for j in 1..=n {
            for i in 1..=m {
                b[ix(i, j, ldb)] = ZERO;
            }
        }
        return Ok(());
    }
    if !lside {
        return Err(unported("ZTRSM", "354-468"));
    }
    if lsame(transa, b'N') {
        // B := alpha*inv(A)*B (ztrsm.f:278-310)
        for j in 1..=n {
            if !alpha.feq(ONE) {
                for i in 1..=m {
                    b[ix(i, j, ldb)] = alpha * b[ix(i, j, ldb)];
                }
            }
            if upper {
                for k in (1..=m).rev() {
                    if !b[ix(k, j, ldb)].is_zero() {
                        if nounit {
                            b[ix(k, j, ldb)] =
                                b[ix(k, j, ldb)] / a[ix(k, k, lda)];
                        }
                        let bkj = b[ix(k, j, ldb)];
                        for i in 1..k {
                            b[ix(i, j, ldb)] =
                                b[ix(i, j, ldb)] - bkj * a[ix(i, k, lda)];
                        }
                    }
                }
            } else {
                for k in 1..=m {
                    if !b[ix(k, j, ldb)].is_zero() {
                        if nounit {
                            b[ix(k, j, ldb)] =
                                b[ix(k, j, ldb)] / a[ix(k, k, lda)];
                        }
                        let bkj = b[ix(k, j, ldb)];
                        for i in k + 1..=m {
                            b[ix(i, j, ldb)] =
                                b[ix(i, j, ldb)] - bkj * a[ix(i, k, lda)];
                        }
                    }
                }
            }
        }
    } else if upper {
        // B := alpha*inv(A**T)*B or alpha*inv(A**H)*B (ztrsm.f:316-333)
        for j in 1..=n {
            for i in 1..=m {
                let mut temp = alpha * b[ix(i, j, ldb)];
                if noconj {
                    for k in 1..i {
                        temp = temp - a[ix(k, i, lda)] * b[ix(k, j, ldb)];
                    }
                    if nounit {
                        temp = temp / a[ix(i, i, lda)];
                    }
                } else {
                    for k in 1..i {
                        temp =
                            temp - a[ix(k, i, lda)].conj() * b[ix(k, j, ldb)];
                    }
                    if nounit {
                        temp = temp / a[ix(i, i, lda)].conj();
                    }
                }
                b[ix(i, j, ldb)] = temp;
            }
        }
    } else {
        // ztrsm.f:335-351
        for j in 1..=n {
            for i in (1..=m).rev() {
                let mut temp = alpha * b[ix(i, j, ldb)];
                if noconj {
                    for k in i + 1..=m {
                        temp = temp - a[ix(k, i, lda)] * b[ix(k, j, ldb)];
                    }
                    if nounit {
                        temp = temp / a[ix(i, i, lda)];
                    }
                } else {
                    for k in i + 1..=m {
                        temp =
                            temp - a[ix(k, i, lda)].conj() * b[ix(k, j, ldb)];
                    }
                    if nounit {
                        temp = temp / a[ix(i, i, lda)].conj();
                    }
                }
                b[ix(i, j, ldb)] = temp;
            }
        }
    }
    Ok(())
}

/// `ZTRSV` (`BLAS/SRC/ztrsv.f`): solve `op(A)*x = b`, `op(A) = A`,
/// `A**T` or `A**H`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] as upstream's argument checks.
#[allow(clippy::many_single_char_names)]
pub fn ztrsv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: usize,
    a: &[Cplx],
    lda: usize,
    x: &mut [Cplx],
    incx: usize,
) -> Result<(), LapackError> {
    let upper = lsame(uplo, b'U');
    let notran = lsame(trans, b'N');
    let noconj = lsame(trans, b'T');
    let nounit = lsame(diag, b'N');
    if !upper && !lsame(uplo, b'L') {
        return Err(bad("ZTRSV ", 1));
    }
    if !notran && !noconj && !lsame(trans, b'C') {
        return Err(bad("ZTRSV ", 2));
    }
    if !nounit && !lsame(diag, b'U') {
        return Err(bad("ZTRSV ", 3));
    }
    if lda < n.max(1) {
        return Err(bad("ZTRSV ", 6));
    }
    if incx == 0 {
        return Err(bad("ZTRSV ", 8));
    }
    if n == 0 {
        return Ok(());
    }
    if notran {
        if upper {
            for j in (1..=n).rev() {
                let jx = (j - 1) * incx;
                if !x[jx].feq(ZERO) {
                    if nounit {
                        x[jx] = x[jx] / a[ix(j, j, lda)];
                    }
                    let temp = x[jx];
                    for i in (1..j).rev() {
                        let ii = (i - 1) * incx;
                        x[ii] = x[ii] - temp * a[ix(i, j, lda)];
                    }
                }
            }
        } else {
            for j in 1..=n {
                let jx = (j - 1) * incx;
                if !x[jx].feq(ZERO) {
                    if nounit {
                        x[jx] = x[jx] / a[ix(j, j, lda)];
                    }
                    let temp = x[jx];
                    for i in j + 1..=n {
                        let ii = (i - 1) * incx;
                        x[ii] = x[ii] - temp * a[ix(i, j, lda)];
                    }
                }
            }
        }
    } else if upper {
        for j in 1..=n {
            let jx = (j - 1) * incx;
            let mut temp = x[jx];
            if noconj {
                for i in 1..j {
                    temp = temp - a[ix(i, j, lda)] * x[(i - 1) * incx];
                }
                if nounit {
                    temp = temp / a[ix(j, j, lda)];
                }
            } else {
                for i in 1..j {
                    temp = temp - a[ix(i, j, lda)].conj() * x[(i - 1) * incx];
                }
                if nounit {
                    temp = temp / a[ix(j, j, lda)].conj();
                }
            }
            x[jx] = temp;
        }
    } else {
        for j in (1..=n).rev() {
            let jx = (j - 1) * incx;
            let mut temp = x[jx];
            if noconj {
                for i in (j + 1..=n).rev() {
                    temp = temp - a[ix(i, j, lda)] * x[(i - 1) * incx];
                }
                if nounit {
                    temp = temp / a[ix(j, j, lda)];
                }
            } else {
                for i in (j + 1..=n).rev() {
                    temp = temp - a[ix(i, j, lda)].conj() * x[(i - 1) * incx];
                }
                if nounit {
                    temp = temp / a[ix(j, j, lda)].conj();
                }
            }
            x[jx] = temp;
        }
    }
    Ok(())
}
