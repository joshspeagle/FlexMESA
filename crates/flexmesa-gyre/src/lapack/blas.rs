// SPDX-License-Identifier: GPL-3.0-only
//! Reference BLAS 3.12.0 routines (`BLAS/SRC/*.f`, `*.f90`), double
//! precision, as the MESA SDK `26.6.1` builds them (`gfortran -O2 -fPIC`).
//!
//! Vectors are slices starting at `X(1)` with a positive increment.
//! Unrolled loops are kept as upstream writes them: the unrolling does not
//! change any single operation, but keeping it makes the transcription
//! line-checkable.

use super::{ix, lsame, LapackError};

fn bad(routine: &'static str, info: i32) -> LapackError {
    LapackError::IllegalArgument { routine, info }
}

fn unported(routine: &'static str, lines: &'static str) -> LapackError {
    LapackError::UnportedPath { routine, lines }
}

/// `DASUM` (`BLAS/SRC/dasum.f`): sum of absolute values.
#[must_use]
pub fn dasum(n: usize, dx: &[f64], incx: usize) -> f64 {
    let mut dtemp = 0.0_f64;
    if n == 0 || incx == 0 {
        return 0.0;
    }
    if incx == 1 {
        let m = n % 6;
        if m != 0 {
            for x in &dx[..m] {
                dtemp += x.abs();
            }
            if n < 6 {
                return dtemp;
            }
        }
        let mut i = m;
        while i < n {
            dtemp = dtemp
                + dx[i].abs()
                + dx[i + 1].abs()
                + dx[i + 2].abs()
                + dx[i + 3].abs()
                + dx[i + 4].abs()
                + dx[i + 5].abs();
            i += 6;
        }
    } else {
        let mut i = 0;
        while i < n * incx {
            dtemp += dx[i].abs();
            i += incx;
        }
    }
    dtemp
}

/// `DAXPY` (`BLAS/SRC/daxpy.f`): `y = y + da*x`.
pub fn daxpy(
    n: usize,
    da: f64,
    dx: &[f64],
    incx: usize,
    dy: &mut [f64],
    incy: usize,
) {
    if n == 0 || da == 0.0 {
        return;
    }
    if incx == 1 && incy == 1 {
        let m = n % 4;
        for i in 0..m {
            dy[i] += da * dx[i];
        }
        if n < 4 {
            return;
        }
        let mut i = m;
        while i < n {
            dy[i] += da * dx[i];
            dy[i + 1] += da * dx[i + 1];
            dy[i + 2] += da * dx[i + 2];
            dy[i + 3] += da * dx[i + 3];
            i += 4;
        }
    } else {
        for i in 0..n {
            dy[i * incy] += da * dx[i * incx];
        }
    }
}

/// `DCOPY` (`BLAS/SRC/dcopy.f`).
pub fn dcopy(n: usize, dx: &[f64], incx: usize, dy: &mut [f64], incy: usize) {
    for i in 0..n {
        dy[i * incy] = dx[i * incx];
    }
}

/// `DDOT` (`BLAS/SRC/ddot.f`): dot product `dx . dy` with the 5-way unroll
/// order of the netlib source.
#[must_use]
pub fn ddot(n: usize, dx: &[f64], incx: usize, dy: &[f64], incy: usize) -> f64 {
    let mut dtemp = 0.0_f64;
    if n == 0 {
        return 0.0;
    }
    if incx == 1 && incy == 1 {
        let m = n % 5;
        if m != 0 {
            for i in 0..m {
                dtemp += dx[i] * dy[i];
            }
            if n < 5 {
                return dtemp;
            }
        }
        let mut i = m;
        while i < n {
            dtemp = dtemp
                + dx[i] * dy[i]
                + dx[i + 1] * dy[i + 1]
                + dx[i + 2] * dy[i + 2]
                + dx[i + 3] * dy[i + 3]
                + dx[i + 4] * dy[i + 4];
            i += 5;
        }
    } else {
        for i in 0..n {
            dtemp += dx[i * incx] * dy[i * incy];
        }
    }
    dtemp
}

/// `DSCAL` (`BLAS/SRC/dscal.f`): `x = da*x` (returns early when `da == 1`).
pub fn dscal(n: usize, da: f64, dx: &mut [f64], incx: usize) {
    if n == 0 || incx == 0 || da == 1.0 {
        return;
    }
    for i in 0..n {
        dx[i * incx] *= da;
    }
}

/// `DSWAP` (`BLAS/SRC/dswap.f`) on two vectors of one array (`x` at
/// offset `ox`, `y` at offset `oy`), as `DBDSQR` calls it on rows/columns
/// of the same matrix.
pub fn dswap_in(
    n: usize,
    a: &mut [f64],
    ox: usize,
    incx: usize,
    oy: usize,
    incy: usize,
) {
    for i in 0..n {
        a.swap(ox + i * incx, oy + i * incy);
    }
}

/// `DROT` (`BLAS/SRC/drot.f`) on two vectors of one array:
/// `x' = c*x + s*y`, `y' = c*y - s*x`.
pub fn drot_in(
    n: usize,
    a: &mut [f64],
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
        let dtemp = c * a[px] + s * a[py];
        a[py] = c * a[py] - s * a[px];
        a[px] = dtemp;
    }
}

/// `IDAMAX` (`BLAS/SRC/idamax.f`): 1-based index of the first maximal
/// `|x(i)|`; 0 for `n == 0`.
#[must_use]
pub fn idamax(n: usize, dx: &[f64], incx: usize) -> usize {
    if n == 0 || incx == 0 {
        return 0;
    }
    let mut imax = 1;
    if n == 1 {
        return 1;
    }
    let mut dmax = dx[0].abs();
    for i in 2..=n {
        let v = dx[(i - 1) * incx].abs();
        if v > dmax {
            imax = i;
            dmax = v;
        }
    }
    imax
}

/// `DNRM2` (`BLAS/SRC/dnrm2.f90`, Blue's algorithm, Anderson 2017).
#[must_use]
pub fn dnrm2(n: usize, x: &[f64], incx: usize) -> f64 {
    // Blue's scaling constants for binary64 (dnrm2.f90:103-110)
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
    for i in 0..n {
        let ax = x[i * incx].abs();
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
    }
    let (scl, sumsq);
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

/// `DGEMV` (`BLAS/SRC/dgemv.f`): `y = alpha*op(A)*x + beta*y`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] as upstream's argument checks.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
pub fn dgemv(
    trans: u8,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
    incy: usize,
) -> Result<(), LapackError> {
    let notrans = lsame(trans, b'N');
    if !notrans && !lsame(trans, b'T') && !lsame(trans, b'C') {
        return Err(bad("DGEMV", 1));
    } else if lda < m.max(1) {
        return Err(bad("DGEMV", 6));
    } else if incx == 0 {
        return Err(bad("DGEMV", 8));
    } else if incy == 0 {
        return Err(bad("DGEMV", 11));
    }
    if m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0) {
        return Ok(());
    }
    let leny = if notrans { m } else { n };
    if beta != 1.0 {
        if beta == 0.0 {
            for i in 0..leny {
                y[i * incy] = 0.0;
            }
        } else {
            for i in 0..leny {
                y[i * incy] *= beta;
            }
        }
    }
    if alpha == 0.0 {
        return Ok(());
    }
    if notrans {
        // y := alpha*A*x + y (dgemv.f:277-296)
        for j in 1..=n {
            let temp = alpha * x[(j - 1) * incx];
            for i in 1..=m {
                y[(i - 1) * incy] += temp * a[ix(i, j, lda)];
            }
        }
    } else {
        // y := alpha*A**T*x + y (dgemv.f:301-322)
        for j in 1..=n {
            let mut temp = 0.0_f64;
            for i in 1..=m {
                temp += a[ix(i, j, lda)] * x[(i - 1) * incx];
            }
            y[(j - 1) * incy] += alpha * temp;
        }
    }
    Ok(())
}

/// `DGER` (`BLAS/SRC/dger.f`): `A = alpha*x*y**T + A`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] as upstream's argument checks.
#[allow(clippy::too_many_arguments)]
pub fn dger(
    m: usize,
    n: usize,
    alpha: f64,
    x: &[f64],
    incx: usize,
    y: &[f64],
    incy: usize,
    a: &mut [f64],
    lda: usize,
) -> Result<(), LapackError> {
    if incx == 0 {
        return Err(bad("DGER", 5));
    } else if incy == 0 {
        return Err(bad("DGER", 7));
    } else if lda < m.max(1) {
        return Err(bad("DGER", 9));
    }
    if m == 0 || n == 0 || alpha == 0.0 {
        return Ok(());
    }
    for j in 1..=n {
        let yj = y[(j - 1) * incy];
        if yj != 0.0 {
            let temp = alpha * yj;
            for i in 1..=m {
                a[ix(i, j, lda)] += x[(i - 1) * incx] * temp;
            }
        }
    }
    Ok(())
}

/// `DTRSV` (`BLAS/SRC/dtrsv.f`): solve `op(A)*x = b`, unit-stride `x`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] as upstream; [`LapackError::UnportedPath`]
/// for `incx != 1`.
pub fn dtrsv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: usize,
    a: &[f64],
    lda: usize,
    x: &mut [f64],
    incx: usize,
) -> Result<(), LapackError> {
    let upper = lsame(uplo, b'U');
    if !upper && !lsame(uplo, b'L') {
        return Err(bad("DTRSV", 1));
    } else if !lsame(trans, b'N') && !lsame(trans, b'T') && !lsame(trans, b'C')
    {
        return Err(bad("DTRSV", 2));
    } else if !lsame(diag, b'U') && !lsame(diag, b'N') {
        return Err(bad("DTRSV", 3));
    } else if lda < n.max(1) {
        return Err(bad("DTRSV", 6));
    } else if incx == 0 {
        return Err(bad("DTRSV", 8));
    }
    if n == 0 {
        return Ok(());
    }
    if incx != 1 {
        return Err(unported("DTRSV", "235-249, 261-275, 291-304, 315-329"));
    }
    let nounit = lsame(diag, b'N');
    if lsame(trans, b'N') {
        if upper {
            for j in (1..=n).rev() {
                if x[j - 1] != 0.0 {
                    if nounit {
                        x[j - 1] /= a[ix(j, j, lda)];
                    }
                    let temp = x[j - 1];
                    for i in (1..j).rev() {
                        x[i - 1] -= temp * a[ix(i, j, lda)];
                    }
                }
            }
        } else {
            for j in 1..=n {
                if x[j - 1] != 0.0 {
                    if nounit {
                        x[j - 1] /= a[ix(j, j, lda)];
                    }
                    let temp = x[j - 1];
                    for i in j + 1..=n {
                        x[i - 1] -= temp * a[ix(i, j, lda)];
                    }
                }
            }
        }
    } else if upper {
        for j in 1..=n {
            let mut temp = x[j - 1];
            for i in 1..j {
                temp -= a[ix(i, j, lda)] * x[i - 1];
            }
            if nounit {
                temp /= a[ix(j, j, lda)];
            }
            x[j - 1] = temp;
        }
    } else {
        for j in (1..=n).rev() {
            let mut temp = x[j - 1];
            for i in (j + 1..=n).rev() {
                temp -= a[ix(i, j, lda)] * x[i - 1];
            }
            if nounit {
                temp /= a[ix(j, j, lda)];
            }
            x[j - 1] = temp;
        }
    }
    Ok(())
}

/// `DGEMM` (`BLAS/SRC/dgemm.f`), `TRANSA = TRANSB = 'N'` only:
/// `C = alpha*A*B + beta*C`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] as upstream; [`LapackError::UnportedPath`]
/// for transposed operands.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
pub fn dgemm(
    transa: u8,
    transb: u8,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) -> Result<(), LapackError> {
    let nota = lsame(transa, b'N');
    let notb = lsame(transb, b'N');
    if !nota && !lsame(transa, b'C') && !lsame(transa, b'T') {
        return Err(bad("DGEMM", 1));
    } else if !notb && !lsame(transb, b'C') && !lsame(transb, b'T') {
        return Err(bad("DGEMM", 2));
    }
    if !(nota && notb) {
        return Err(unported("DGEMM", "316-373"));
    }
    if lda < m.max(1) {
        return Err(bad("DGEMM", 8));
    } else if ldb < k.max(1) {
        return Err(bad("DGEMM", 10));
    } else if ldc < m.max(1) {
        return Err(bad("DGEMM", 13));
    }
    if m == 0 || n == 0 || ((alpha == 0.0 || k == 0) && beta == 1.0) {
        return Ok(());
    }
    if alpha == 0.0 {
        for j in 1..=n {
            for i in 1..=m {
                c[ix(i, j, ldc)] = if beta == 0.0 {
                    0.0
                } else {
                    beta * c[ix(i, j, ldc)]
                };
            }
        }
        return Ok(());
    }
    // C := alpha*A*B + beta*C (dgemm.f:299-315)
    for j in 1..=n {
        if beta == 0.0 {
            for i in 1..=m {
                c[ix(i, j, ldc)] = 0.0;
            }
        } else if beta != 1.0 {
            for i in 1..=m {
                c[ix(i, j, ldc)] *= beta;
            }
        }
        for l in 1..=k {
            let temp = alpha * b[ix(l, j, ldb)];
            for i in 1..=m {
                c[ix(i, j, ldc)] += temp * a[ix(i, l, lda)];
            }
        }
    }
    Ok(())
}

/// `DTRSM` (`BLAS/SRC/dtrsm.f`), `SIDE = 'L'` only: solve
/// `op(A)*X = alpha*B`, overwriting `B`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] as upstream; [`LapackError::UnportedPath`]
/// for `SIDE = 'R'`.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
pub fn dtrsm(
    side: u8,
    uplo: u8,
    transa: u8,
    diag: u8,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &mut [f64],
    ldb: usize,
) -> Result<(), LapackError> {
    let lside = lsame(side, b'L');
    let nrowa = if lside { m } else { n };
    let nounit = lsame(diag, b'N');
    let upper = lsame(uplo, b'U');
    if !lside && !lsame(side, b'R') {
        return Err(bad("DTRSM", 1));
    } else if !upper && !lsame(uplo, b'L') {
        return Err(bad("DTRSM", 2));
    } else if !lsame(transa, b'N')
        && !lsame(transa, b'T')
        && !lsame(transa, b'C')
    {
        return Err(bad("DTRSM", 3));
    } else if !lsame(diag, b'U') && !lsame(diag, b'N') {
        return Err(bad("DTRSM", 4));
    } else if lda < nrowa.max(1) {
        return Err(bad("DTRSM", 9));
    } else if ldb < m.max(1) {
        return Err(bad("DTRSM", 11));
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    if alpha == 0.0 {
        for j in 1..=n {
            for i in 1..=m {
                b[ix(i, j, ldb)] = 0.0;
            }
        }
        return Ok(());
    }
    if !lside {
        return Err(unported("DTRSM", "337-435"));
    }
    if lsame(transa, b'N') {
        // B := alpha*inv(A)*B (dtrsm.f:276-307)
        for j in 1..=n {
            if alpha != 1.0 {
                for i in 1..=m {
                    b[ix(i, j, ldb)] *= alpha;
                }
            }
            if upper {
                for k in (1..=m).rev() {
                    if b[ix(k, j, ldb)] != 0.0 {
                        if nounit {
                            b[ix(k, j, ldb)] /= a[ix(k, k, lda)];
                        }
                        let bkj = b[ix(k, j, ldb)];
                        for i in 1..k {
                            b[ix(i, j, ldb)] -= bkj * a[ix(i, k, lda)];
                        }
                    }
                }
            } else {
                for k in 1..=m {
                    if b[ix(k, j, ldb)] != 0.0 {
                        if nounit {
                            b[ix(k, j, ldb)] /= a[ix(k, k, lda)];
                        }
                        let bkj = b[ix(k, j, ldb)];
                        for i in k + 1..=m {
                            b[ix(i, j, ldb)] -= bkj * a[ix(i, k, lda)];
                        }
                    }
                }
            }
        }
    } else if upper {
        // B := alpha*inv(A**T)*B (dtrsm.f:313-323)
        for j in 1..=n {
            for i in 1..=m {
                let mut temp = alpha * b[ix(i, j, ldb)];
                for k in 1..i {
                    temp -= a[ix(k, i, lda)] * b[ix(k, j, ldb)];
                }
                if nounit {
                    temp /= a[ix(i, i, lda)];
                }
                b[ix(i, j, ldb)] = temp;
            }
        }
    } else {
        for j in 1..=n {
            for i in (1..=m).rev() {
                let mut temp = alpha * b[ix(i, j, ldb)];
                for k in i + 1..=m {
                    temp -= a[ix(k, i, lda)] * b[ix(k, j, ldb)];
                }
                if nounit {
                    temp /= a[ix(i, i, lda)];
                }
                b[ix(i, j, ldb)] = temp;
            }
        }
    }
    Ok(())
}
