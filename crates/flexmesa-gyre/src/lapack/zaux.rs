// SPDX-License-Identifier: GPL-3.0-only
//! Reference LAPACK 3.12.0 complex auxiliary routines (`SRC/*.f`), double
//! precision, as the MESA SDK `26.6.1` builds them (`gfortran -O2 -fPIC`).
//!
//! Three objects here are the SDK's CRMATH-patched builds (netlib plus
//! `USE CRMATH`, CRMATH generic names replacing the intrinsics):
//!
//! - `zlange`: complex `ABS(z)` is CRMATH `abs_c` ([`crmath::abs_c`],
//!   `hypot(Re z, Im z)`), not the libm `cabs` of the netlib build.
//! - `zladiv`, `zlarfg`: `DCMPLX` is CRMATH `dcmplx`, a plain constructor
//!   (the bits are unchanged, including signed zeros and NaN payloads).
//!
//! The others are instruction-identical to the netlib build.

use super::aux::{fmax, sign};
use super::zblas::{dznrm2, zdscal, zgemv, zgerc, zscal};
use super::{dlamch, ix, lsame, LapackError};
use crate::crmath;
use crate::fcomplex::Cplx;

const ZERO: Cplx = Cplx::new(0.0, 0.0);
const ONE: Cplx = Cplx::new(1.0, 0.0);

fn unported(routine: &'static str, lines: &'static str) -> LapackError {
    LapackError::UnportedPath { routine, lines }
}

/// `DLADIV2` (`SRC/dladiv.f:215-251`).
#[allow(clippy::many_single_char_names)]
fn dladiv2(a: f64, b: f64, c: f64, d: f64, r: f64, t: f64) -> f64 {
    if r != 0.0 {
        let br = b * r;
        if br != 0.0 {
            (a + br) * t
        } else {
            a * t + (b * t) * r
        }
    } else {
        (a + d * (b / c)) * t
    }
}

/// `DLADIV1` (`SRC/dladiv.f:176-211`). Returns `(p, q)`.
#[allow(clippy::many_single_char_names)]
fn dladiv1(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let r = d / c;
    let t = 1.0 / (c + d * r);
    let p = dladiv2(a, b, c, d, r, t);
    let q = dladiv2(b, -a, c, d, r, t);
    (p, q)
}

/// `DLADIV` (`SRC/dladiv.f:91-171`): robust `(a + i b)/(c + i d)`
/// (Baudin and Smith 2012). Returns `(p, q)`.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn dladiv(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    const BS: f64 = 2.0;
    const HALF: f64 = 0.5;
    const TWO: f64 = 2.0;
    let mut aa = a;
    let mut bb = b;
    let mut cc = c;
    let mut dd = d;
    let ab = fmax(a.abs(), b.abs());
    let cd = fmax(c.abs(), d.abs());
    let mut s = 1.0_f64;
    let ov = dlamch::RMAX;
    let un = dlamch::SFMIN;
    let eps = dlamch::EPS;
    let be = BS / (eps * eps);
    // dladiv.f:138-157
    if ab >= HALF * ov {
        aa *= HALF;
        bb *= HALF;
        s *= TWO;
    }
    if cd >= HALF * ov {
        cc *= HALF;
        dd *= HALF;
        s *= HALF;
    }
    if ab <= un * BS / eps {
        aa *= be;
        bb *= be;
        s /= be;
    }
    if cd <= un * BS / eps {
        cc *= be;
        dd *= be;
        s *= be;
    }
    // dladiv.f:158-165
    let (p, mut q);
    if d.abs() <= c.abs() {
        (p, q) = dladiv1(aa, bb, cc, dd);
    } else {
        (p, q) = dladiv1(bb, aa, dd, cc);
        q = -q;
    }
    (p * s, q * s)
}

/// `ZLADIV` (`SRC/zladiv.f`, SDK CRMATH build): `x/y` through [`dladiv`].
#[must_use]
pub fn zladiv(x: Cplx, y: Cplx) -> Cplx {
    let (zr, zi) = dladiv(x.re, x.im, y.re, y.im);
    Cplx::new(zr, zi)
}

/// `DLAPY3` (`SRC/dlapy3.f`): `sqrt(x**2 + y**2 + z**2)` avoiding
/// overflow.
#[must_use]
pub fn dlapy3(x: f64, y: f64, z: f64) -> f64 {
    let hugeval = dlamch::RMAX;
    let xabs = x.abs();
    let yabs = y.abs();
    let zabs = z.abs();
    // MAX(XABS, YABS, ZABS) is MAX_EXPR <xabs, MAX_EXPR <yabs, zabs>>
    let w = fmax(xabs, fmax(yabs, zabs));
    if w == 0.0 || w > hugeval {
        // dlapy3.f:103
        xabs + yabs + zabs
    } else {
        let a = xabs / w;
        let b = yabs / w;
        let c = zabs / w;
        w * (a * a + b * b + c * c).sqrt()
    }
}

/// `ILAZLC` (`SRC/ilazlc.f`): last non-zero column of the `m x n` `A`
/// (0 if none). `m >= 1`.
#[must_use]
pub fn ilazlc(m: usize, n: usize, a: &[Cplx], lda: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if !a[ix(1, n, lda)].is_zero() || !a[ix(m, n, lda)].is_zero() {
        return n;
    }
    for j in (1..=n).rev() {
        for i in 1..=m {
            if !a[ix(i, j, lda)].is_zero() {
                return j;
            }
        }
    }
    0
}

/// `ILAZLR` (`SRC/ilazlr.f`): last non-zero row of the `m x n` `A`
/// (0 if none). `n >= 1`.
#[must_use]
pub fn ilazlr(m: usize, n: usize, a: &[Cplx], lda: usize) -> usize {
    if m == 0 {
        return 0;
    }
    if !a[ix(m, 1, lda)].is_zero() || !a[ix(m, n, lda)].is_zero() {
        return m;
    }
    let mut res = 0;
    for j in 1..=n {
        let mut i = m;
        while i >= 1 && a[ix(i, j, lda)].is_zero() {
            i -= 1;
        }
        res = res.max(i);
    }
    res
}

/// `ZLACGV` (`SRC/zlacgv.f`): conjugate the `n` elements of `x`.
pub fn zlacgv(n: usize, x: &mut [Cplx], incx: usize) {
    for i in 0..n {
        x[i * incx] = x[i * incx].conj();
    }
}

/// `ZLACPY` (`SRC/zlacpy.f`): copy all (`uplo` other than `U`/`L`), the
/// upper (`U`) or the lower (`L`) triangle of `A` to `B`.
pub fn zlacpy(
    uplo: u8,
    m: usize,
    n: usize,
    a: &[Cplx],
    lda: usize,
    b: &mut [Cplx],
    ldb: usize,
) {
    for j in 1..=n {
        let (lo, hi) = if lsame(uplo, b'U') {
            (1, j.min(m))
        } else if lsame(uplo, b'L') {
            (j, m)
        } else {
            (1, m)
        };
        for i in lo..=hi {
            b[ix(i, j, ldb)] = a[ix(i, j, lda)];
        }
    }
}

/// `ZLANGE` (`SRC/zlange.f`, SDK CRMATH build), norms `'M'` (max abs),
/// `'1'`/`'O'` (max column sum) and `'I'` (max row sum), with
/// `|z| = hypot(Re z, Im z)` ([`crmath::abs_c`]).
///
/// # Errors
///
/// [`LapackError::UnportedPath`] for `'F'`/`'E'` (`ZLASSQ`, lines
/// 193-202); [`LapackError::IllegalArgument`] for any other `norm`, for
/// which upstream returns an unset value.
pub fn zlange(
    norm: u8,
    m: usize,
    n: usize,
    a: &[Cplx],
    lda: usize,
) -> Result<f64, LapackError> {
    let mut value = 0.0_f64;
    if m.min(n) == 0 {
        return Ok(0.0);
    }
    if lsame(norm, b'M') {
        // zlange.f:157-163
        for j in 1..=n {
            for i in 1..=m {
                let temp = crmath::abs_c(a[ix(i, j, lda)]);
                if value < temp || temp.is_nan() {
                    value = temp;
                }
            }
        }
    } else if lsame(norm, b'O') || norm == b'1' {
        // zlange.f:168-175
        for j in 1..=n {
            let mut sum = 0.0_f64;
            for i in 1..=m {
                sum += crmath::abs_c(a[ix(i, j, lda)]);
            }
            if value < sum || sum.is_nan() {
                value = sum;
            }
        }
    } else if lsame(norm, b'I') {
        // zlange.f:180-192
        let mut work = vec![0.0_f64; m];
        for j in 1..=n {
            for i in 1..=m {
                work[i - 1] += crmath::abs_c(a[ix(i, j, lda)]);
            }
        }
        for &temp in &work {
            if value < temp || temp.is_nan() {
                value = temp;
            }
        }
    } else if lsame(norm, b'F') || lsame(norm, b'E') {
        return Err(unported("ZLANGE", "193-202"));
    } else {
        return Err(LapackError::IllegalArgument {
            routine: "ZLANGE",
            info: -1,
        });
    }
    Ok(value)
}

/// `ZLARFG` (`SRC/zlarfg.f`, SDK CRMATH build): elementary reflector `H`
/// with `H**H*(alpha; x) = (beta; 0)`, `beta` real. Updates `alpha` to
/// `beta` and `x` to `v`; returns `tau`.
pub fn zlarfg(n: usize, alpha: &mut Cplx, x: &mut [Cplx], incx: usize) -> Cplx {
    if n == 0 {
        return ZERO;
    }
    let mut xnorm = dznrm2(n - 1, x, incx);
    let mut alphr = alpha.re;
    let mut alphi = alpha.im;
    if xnorm == 0.0 && alphi == 0.0 {
        // H = I (zlarfg.f:151-155)
        return ZERO;
    }
    // zlarfg.f:160-193
    let mut beta = -sign(dlapy3(alphr, alphi, xnorm), alphr);
    let safmin = dlamch::SFMIN / dlamch::EPS;
    let rsafmn = 1.0 / safmin;
    let mut knt = 0;
    if beta.abs() < safmin {
        loop {
            knt += 1;
            zdscal(n - 1, rsafmn, x, incx);
            beta *= rsafmn;
            alphi *= rsafmn;
            alphr *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        xnorm = dznrm2(n - 1, x, incx);
        *alpha = Cplx::new(alphr, alphi);
        beta = -sign(dlapy3(alphr, alphi, xnorm), alphr);
    }
    let tau = Cplx::new((beta - alphr) / beta, -alphi / beta);
    // ALPHA - BETA: the real BETA is widened to (beta, 0), Im - 0 folds
    let scale = zladiv(ONE, Cplx::new(alpha.re - beta, alpha.im));
    zscal(n - 1, scale, x, incx);
    for _ in 0..knt {
        beta *= safmin;
    }
    *alpha = Cplx::from_real(beta);
    tau
}

/// `ZLARF` (`SRC/zlarf.f`): apply `H = I - tau*v*v**H` to `C` from the left
/// (`side = 'L'`) or right. `v` is passed as its own (copied) vector.
/// `work` needs `n` (left) or `m` (right) entries.
///
/// # Errors
///
/// Propagates [`zgemv`]/[`zgerc`] errors.
pub fn zlarf(
    side: u8,
    m: usize,
    n: usize,
    v: &[Cplx],
    incv: usize,
    tau: Cplx,
    c: &mut [Cplx],
    ldc: usize,
    work: &mut [Cplx],
) -> Result<(), LapackError> {
    let applyleft = lsame(side, b'L');
    let mut lastv = 0;
    let mut lastc = 0;
    if !tau.is_zero() {
        // zlarf.f:169-190
        lastv = if applyleft { m } else { n };
        let mut i = 1 + lastv.saturating_sub(1) * incv;
        while lastv > 0 && v[i - 1].is_zero() {
            lastv -= 1;
            if lastv > 0 {
                i -= incv;
            }
        }
        // ILAZLC/ILAZLR only matter when lastv > 0 (zlarf.f:198, 213)
        if lastv > 0 {
            lastc = if applyleft {
                ilazlc(lastv, n, c, ldc)
            } else {
                ilazlr(m, lastv, c, ldc)
            };
        }
    }
    if lastv > 0 {
        if applyleft {
            // zlarf.f:202-207
            zgemv(b'C', lastv, lastc, ONE, c, ldc, v, incv, ZERO, work, 1)?;
            zgerc(lastv, lastc, -tau, v, incv, work, 1, c, ldc)?;
        } else {
            // zlarf.f:217-222
            zgemv(b'N', lastc, lastv, ONE, c, ldc, v, incv, ZERO, work, 1)?;
            zgerc(lastc, lastv, -tau, work, 1, v, incv, c, ldc)?;
        }
    }
    Ok(())
}

/// `ZLASR` (`SRC/zlasr.f`), `PIVOT = 'V'`: apply the real plane rotations
/// `(c(j), s(j))` to the complex `A` from the left (`side = 'L'`) or right
/// (`'R'`), forward (`direct = 'F'`) or backward (`'B'`).
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] for invalid options;
/// [`LapackError::UnportedPath`] for `PIVOT = 'T'` or `'B'`.
#[allow(clippy::many_single_char_names)]
pub fn zlasr(
    side: u8,
    pivot: u8,
    direct: u8,
    m: usize,
    n: usize,
    c: &[f64],
    s: &[f64],
    a: &mut [Cplx],
    lda: usize,
) -> Result<(), LapackError> {
    let left = lsame(side, b'L');
    if !left && !lsame(side, b'R') {
        return Err(LapackError::IllegalArgument {
            routine: "ZLASR",
            info: 1,
        });
    }
    if !lsame(pivot, b'V') {
        if lsame(pivot, b'T') || lsame(pivot, b'B') {
            return Err(unported("ZLASR", "294-344, 377-427"));
        }
        return Err(LapackError::IllegalArgument {
            routine: "ZLASR",
            info: 2,
        });
    }
    let fwd = lsame(direct, b'F');
    if !fwd && !lsame(direct, b'B') {
        return Err(LapackError::IllegalArgument {
            routine: "ZLASR",
            info: 3,
        });
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    // A(J+1) = CTEMP*TEMP - STEMP*A(J); A(J) = STEMP*TEMP + CTEMP*A(J)
    // (zlasr.f:275-277, 287-289, 358-360, 370-372)
    let rot = |a: &mut [Cplx], j: usize, p_jp1: usize, p_j: usize| {
        let ctemp = c[j - 1];
        let stemp = s[j - 1];
        if ctemp != 1.0 || stemp != 0.0 {
            let temp = a[p_jp1];
            a[p_jp1] =
                Cplx::real_mul(ctemp, temp) - Cplx::real_mul(stemp, a[p_j]);
            a[p_j] =
                Cplx::real_mul(stemp, temp) + Cplx::real_mul(ctemp, a[p_j]);
        }
    };
    let (len, inner) = if left { (m, n) } else { (n, m) };
    let js: Vec<usize> = if fwd {
        (1..len).collect()
    } else {
        (1..len).rev().collect()
    };
    for j in js {
        for i in 1..=inner {
            if left {
                rot(a, j, ix(j + 1, i, lda), ix(j, i, lda));
            } else {
                rot(a, j, ix(i, j + 1, lda), ix(i, j, lda));
            }
        }
    }
    Ok(())
}

/// `ZLASWP` (`SRC/zlaswp.f`): row interchanges `k1..k2` of `ipiv`
/// (1-based) on the `n` columns of `A`, forward (`incx = 1`) or backward
/// (`incx = -1`). The interchanges are pure moves, so upstream's 32-column
/// blocking does not affect the result.
pub fn zlaswp(
    n: usize,
    a: &mut [Cplx],
    lda: usize,
    k1: usize,
    k2: usize,
    ipiv: &[usize],
    incx: i32,
) {
    let rows: Vec<usize> = match incx.cmp(&0) {
        std::cmp::Ordering::Greater => (k1..=k2).collect(),
        std::cmp::Ordering::Less => (k1..=k2).rev().collect(),
        std::cmp::Ordering::Equal => return,
    };
    for i in rows {
        let ip = ipiv[i - 1];
        if ip != i {
            for k in 1..=n {
                a.swap(ix(i, k, lda), ix(ip, k, lda));
            }
        }
    }
}
