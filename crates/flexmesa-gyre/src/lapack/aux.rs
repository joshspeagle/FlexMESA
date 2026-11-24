// SPDX-License-Identifier: GPL-3.0-only
//! Reference LAPACK 3.12.0 auxiliary routines (`SRC/*.f`, `*.f90`), double
//! precision, as the MESA SDK `26.6.1` builds them (`gfortran -O2 -fPIC`).

use super::blas::{dasum, dcopy, dgemv, dger, dnrm2, dscal, dtrsv, idamax};
use super::{dlamch, ix, lsame, LapackError};

fn unported(routine: &'static str, lines: &'static str) -> LapackError {
    LapackError::UnportedPath { routine, lines }
}

/// gfortran `MAX(a, b)` for non-NaN reals.
#[inline]
pub(crate) fn fmax(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}

/// gfortran `MIN(a, b)` for non-NaN reals.
#[inline]
pub(crate) fn fmin(a: f64, b: f64) -> f64 {
    if a < b {
        a
    } else {
        b
    }
}

/// Fortran `SIGN(a, b)` (`-fsign-zero`, the default): `|a|` with the sign
/// bit of `b`.
#[inline]
pub(crate) fn sign(a: f64, b: f64) -> f64 {
    a.abs().copysign(b)
}

/// `DLAPY2` (`SRC/dlapy2.f`): `sqrt(x**2 + y**2)` avoiding overflow.
#[must_use]
pub fn dlapy2(x: f64, y: f64) -> f64 {
    let x_is_nan = x.is_nan();
    let y_is_nan = y.is_nan();
    let mut value = 0.0;
    if x_is_nan {
        value = x;
    }
    if y_is_nan {
        value = y;
    }
    let hugeval = dlamch::RMAX;
    if !(x_is_nan || y_is_nan) {
        let xabs = x.abs();
        let yabs = y.abs();
        let w = fmax(xabs, yabs);
        let z = fmin(xabs, yabs);
        if z == 0.0 || w > hugeval {
            value = w;
        } else {
            let r = z / w;
            value = w * (1.0 + r * r).sqrt();
        }
    }
    value
}

/// `DLARTG` (`SRC/dlartg.f90`): plane rotation with `c*f + s*g = r`,
/// `-s*f + c*g = 0`. Returns `(c, s, r)`.
#[must_use]
pub fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
    // la_constants: safmin = 2**-1022, safmax = 1/safmin
    let safmin = dlamch::SFMIN;
    let safmax = 1.0 / safmin;
    let rtmin = safmin.sqrt();
    let rtmax = (safmax / 2.0).sqrt();
    let f1 = f.abs();
    let g1 = g.abs();
    if g == 0.0 {
        (1.0, 0.0, f)
    } else if f == 0.0 {
        (0.0, sign(1.0, g), g1)
    } else if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        let d = (f * f + g * g).sqrt();
        let c = f1 / d;
        let r = sign(d, f);
        (c, g / r, r)
    } else {
        let u = fmin(safmax, fmax(fmax(safmin, f1), g1));
        let fs = f / u;
        let gs = g / u;
        let d = (fs * fs + gs * gs).sqrt();
        let c = fs.abs() / d;
        let r = sign(d, f);
        let s = gs / r;
        (c, s, r * u)
    }
}

/// `DLAS2` (`SRC/dlas2.f`): singular values of `[[f, g], [0, h]]`.
/// Returns `(ssmin, ssmax)`.
#[must_use]
pub fn dlas2(f: f64, g: f64, h: f64) -> (f64, f64) {
    let fa = f.abs();
    let ga = g.abs();
    let ha = h.abs();
    let fhmn = fmin(fa, ha);
    let fhmx = fmax(fa, ha);
    if fhmn == 0.0 {
        let ssmax = if fhmx == 0.0 {
            ga
        } else {
            let r = fmin(fhmx, ga) / fmax(fhmx, ga);
            fmax(fhmx, ga) * (1.0 + r * r).sqrt()
        };
        (0.0, ssmax)
    } else if ga < fhmx {
        let as_ = 1.0 + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        let t = ga / fhmx;
        let au = t * t;
        let c = 2.0 / ((as_ * as_ + au).sqrt() + (at * at + au).sqrt());
        (fhmn * c, fhmx / c)
    } else {
        let au = fhmx / ga;
        if au == 0.0 {
            ((fhmn * fhmx) / ga, ga)
        } else {
            let as_ = 1.0 + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let p = as_ * au;
            let q = at * au;
            let c = 1.0 / ((1.0 + p * p).sqrt() + (1.0 + q * q).sqrt());
            let mut ssmin = (fhmn * c) * au;
            ssmin += ssmin;
            (ssmin, ga / (c + c))
        }
    }
}

/// Output of [`dlasv2`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Svd2 {
    /// Smaller singular value (signed).
    pub ssmin: f64,
    /// Larger singular value (signed).
    pub ssmax: f64,
    /// Right rotation sine.
    pub snr: f64,
    /// Right rotation cosine.
    pub csr: f64,
    /// Left rotation sine.
    pub snl: f64,
    /// Left rotation cosine.
    pub csl: f64,
}

/// `DLASV2` (`SRC/dlasv2.f`): SVD of the 2x2 upper triangle
/// `[[f, g], [0, h]]`.
#[must_use]
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub fn dlasv2(f: f64, g: f64, h: f64) -> Svd2 {
    let mut ft = f;
    let mut fa = ft.abs();
    let mut ht = h;
    let mut ha = h.abs();
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        std::mem::swap(&mut ft, &mut ht);
        std::mem::swap(&mut fa, &mut ha);
    }
    let gt = g;
    let ga = gt.abs();
    let (mut ssmin, mut ssmax, clt, crt, slt, srt);
    if ga == 0.0 {
        ssmin = ha;
        ssmax = fa;
        clt = 1.0;
        crt = 1.0;
        slt = 0.0;
        srt = 0.0;
    } else {
        let mut gasmal = true;
        let (mut c_lt, mut c_rt, mut s_lt, mut s_rt) = (0.0, 0.0, 0.0, 0.0);
        ssmin = 0.0;
        ssmax = 0.0;
        if ga > fa {
            pmax = 2;
            if (fa / ga) < dlamch::EPS {
                gasmal = false;
                ssmax = ga;
                ssmin = if ha > 1.0 {
                    fa / (ga / ha)
                } else {
                    (fa / ga) * ha
                };
                c_lt = 1.0;
                s_lt = ht / gt;
                s_rt = 1.0;
                c_rt = ft / gt;
            }
        }
        if gasmal {
            let d = fa - ha;
            let mut l = if d == fa { 1.0 } else { d / fa };
            let m = gt / ft;
            let mut t = 2.0 - l;
            let mm = m * m;
            let tt = t * t;
            let s = (tt + mm).sqrt();
            let r = if l == 0.0 {
                m.abs()
            } else {
                (l * l + mm).sqrt()
            };
            let a = 0.5 * (s + r);
            ssmin = ha / a;
            ssmax = fa * a;
            if mm == 0.0 {
                t = if l == 0.0 {
                    sign(2.0, ft) * sign(1.0, gt)
                } else {
                    gt / sign(d, ft) + m / t
                };
            } else {
                t = (m / (s + t) + m / (r + l)) * (1.0 + a);
            }
            l = (t * t + 4.0).sqrt();
            c_rt = 2.0 / l;
            s_rt = t / l;
            c_lt = (c_rt + s_rt * m) / a;
            s_lt = (ht / ft) * s_rt / a;
        }
        clt = c_lt;
        crt = c_rt;
        slt = s_lt;
        srt = s_rt;
    }
    let (csl, snl, csr, snr) = if swap {
        (srt, crt, slt, clt)
    } else {
        (clt, slt, crt, srt)
    };
    let tsign = match pmax {
        1 => sign(1.0, csr) * sign(1.0, csl) * sign(1.0, f),
        2 => sign(1.0, snr) * sign(1.0, csl) * sign(1.0, g),
        _ => sign(1.0, snr) * sign(1.0, snl) * sign(1.0, h),
    };
    ssmax = sign(ssmax, tsign);
    ssmin = sign(ssmin, tsign * sign(1.0, f) * sign(1.0, h));
    Svd2 {
        ssmin,
        ssmax,
        snr,
        csr,
        snl,
        csl,
    }
}

/// `DLASR` (`SRC/dlasr.f`), `PIVOT = 'V'`: apply the plane rotations
/// `(c(j), s(j))` from the left (`side = 'L'`) or right (`'R'`), forward
/// (`direct = 'F'`) or backward (`'B'`).
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] for invalid options;
/// [`LapackError::UnportedPath`] for `PIVOT = 'T'` or `'B'`.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
pub fn dlasr(
    side: u8,
    pivot: u8,
    direct: u8,
    m: usize,
    n: usize,
    c: &[f64],
    s: &[f64],
    a: &mut [f64],
    lda: usize,
) -> Result<(), LapackError> {
    let left = lsame(side, b'L');
    if !left && !lsame(side, b'R') {
        return Err(LapackError::IllegalArgument {
            routine: "DLASR",
            info: 1,
        });
    }
    if !lsame(pivot, b'V') {
        return Err(unported("DLASR", "278-341, 361-427"));
    }
    let fwd = lsame(direct, b'F');
    if !fwd && !lsame(direct, b'B') {
        return Err(LapackError::IllegalArgument {
            routine: "DLASR",
            info: 3,
        });
    }
    if m == 0 || n == 0 {
        return Ok(());
    }
    let rot = |a: &mut [f64], j: usize, p_jp1: usize, p_j: usize| {
        let ctemp = c[j - 1];
        let stemp = s[j - 1];
        if ctemp != 1.0 || stemp != 0.0 {
            let temp = a[p_jp1];
            a[p_jp1] = ctemp * temp - stemp * a[p_j];
            a[p_j] = stemp * temp + ctemp * a[p_j];
        }
    };
    if left {
        let js: Vec<usize> = if fwd {
            (1..m).collect()
        } else {
            (1..m).rev().collect()
        };
        for j in js {
            for i in 1..=n {
                rot(a, j, ix(j + 1, i, lda), ix(j, i, lda));
            }
        }
    } else {
        let js: Vec<usize> = if fwd {
            (1..n).collect()
        } else {
            (1..n).rev().collect()
        };
        for j in js {
            for i in 1..=m {
                rot(a, j, ix(i, j + 1, lda), ix(i, j, lda));
            }
        }
    }
    Ok(())
}

/// `DLASWP` (`SRC/dlaswp.f`): row interchanges `k1..k2` of `ipiv`
/// (1-based) on the `n` columns of `A`, forward (`incx = 1`) or backward
/// (`incx = -1`).
pub fn dlaswp(
    n: usize,
    a: &mut [f64],
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

/// `DLACPY` (`SRC/dlacpy.f`): copy all (`uplo` other than `U`/`L`), the
/// upper (`U`) or the lower (`L`) triangle of `A` to `B`.
pub fn dlacpy(
    uplo: u8,
    m: usize,
    n: usize,
    a: &[f64],
    lda: usize,
    b: &mut [f64],
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

/// `DLANGE` (`SRC/dlange.f`), norms `'M'` (max abs) and `'1'`/`'O'`.
///
/// # Errors
///
/// [`LapackError::UnportedPath`] for `'I'` and `'F'`.
pub fn dlange(
    norm: u8,
    m: usize,
    n: usize,
    a: &[f64],
    lda: usize,
) -> Result<f64, LapackError> {
    let mut value = 0.0_f64;
    if m.min(n) == 0 {
        return Ok(0.0);
    }
    if lsame(norm, b'M') {
        for j in 1..=n {
            for i in 1..=m {
                let temp = a[ix(i, j, lda)].abs();
                if value < temp || temp.is_nan() {
                    value = temp;
                }
            }
        }
    } else if lsame(norm, b'O') || norm == b'1' {
        for j in 1..=n {
            let mut sum = 0.0_f64;
            for i in 1..=m {
                sum += a[ix(i, j, lda)].abs();
            }
            if value < sum || sum.is_nan() {
                value = sum;
            }
        }
    } else {
        return Err(unported("DLANGE", "174-200"));
    }
    Ok(value)
}

/// `DLANTR` (`SRC/dlantr.f`), `NORM = 'M'`, `UPLO = 'U'`, `DIAG = 'N'`.
///
/// # Errors
///
/// [`LapackError::UnportedPath`] for other options.
pub fn dlantr(
    norm: u8,
    uplo: u8,
    diag: u8,
    m: usize,
    n: usize,
    a: &[f64],
    lda: usize,
) -> Result<f64, LapackError> {
    if m.min(n) == 0 {
        return Ok(0.0);
    }
    if !(lsame(norm, b'M') && lsame(uplo, b'U') && lsame(diag, b'N')) {
        return Err(unported("DLANTR", "183-199, 209-216, 218-342"));
    }
    let mut value = 0.0_f64;
    for j in 1..=n {
        for i in 1..=m.min(j) {
            let sum = a[ix(i, j, lda)].abs();
            if value < sum || sum.is_nan() {
                value = sum;
            }
        }
    }
    Ok(value)
}

/// `ILADLC` (`SRC/iladlc.f`): last non-zero column of `A` (0 if none).
#[must_use]
pub fn iladlc(m: usize, n: usize, a: &[f64], lda: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if a[ix(1, n, lda)] != 0.0 || a[ix(m, n, lda)] != 0.0 {
        return n;
    }
    for j in (1..=n).rev() {
        for i in 1..=m {
            if a[ix(i, j, lda)] != 0.0 {
                return j;
            }
        }
    }
    0
}

/// `ILADLR` (`SRC/iladlr.f`): last non-zero row of `A` (0 if none).
#[must_use]
pub fn iladlr(m: usize, n: usize, a: &[f64], lda: usize) -> usize {
    if m == 0 {
        return 0;
    }
    if a[ix(m, 1, lda)] != 0.0 || a[ix(m, n, lda)] != 0.0 {
        return m;
    }
    let mut res = 0;
    for j in 1..=n {
        let mut i = m;
        while i >= 1 && a[ix(i.max(1), j, lda)] == 0.0 {
            i -= 1;
        }
        res = res.max(i);
    }
    res
}

/// `DLARFG` (`SRC/dlarfg.f`): elementary reflector `H` with
/// `H*(alpha; x) = (beta; 0)`. Updates `alpha` to `beta` and `x` to `v`;
/// returns `tau`.
pub fn dlarfg(n: usize, alpha: &mut f64, x: &mut [f64], incx: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let mut xnorm = dnrm2(n - 1, x, incx);
    if xnorm == 0.0 {
        return 0.0;
    }
    let mut beta = -sign(dlapy2(*alpha, xnorm), *alpha);
    let safmin = dlamch::SFMIN / dlamch::EPS;
    let mut knt = 0;
    if beta.abs() < safmin {
        let rsafmn = 1.0 / safmin;
        loop {
            knt += 1;
            dscal(n - 1, rsafmn, x, incx);
            beta *= rsafmn;
            *alpha *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        xnorm = dnrm2(n - 1, x, incx);
        beta = -sign(dlapy2(*alpha, xnorm), *alpha);
    }
    let tau = (beta - *alpha) / beta;
    dscal(n - 1, 1.0 / (*alpha - beta), x, incx);
    for _ in 0..knt {
        beta *= safmin;
    }
    *alpha = beta;
    tau
}

/// `DLARF` (`SRC/dlarf.f`): apply `H = I - tau*v*v**T` to `C` from the left
/// (`side = 'L'`) or right. `v` is passed as its own (copied) vector.
///
/// # Errors
///
/// Propagates [`dgemv`]/[`dger`] errors.
#[allow(clippy::too_many_arguments)]
pub fn dlarf(
    side: u8,
    m: usize,
    n: usize,
    v: &[f64],
    incv: usize,
    tau: f64,
    c: &mut [f64],
    ldc: usize,
    work: &mut [f64],
) -> Result<(), LapackError> {
    let applyleft = lsame(side, b'L');
    let mut lastv = 0;
    let mut lastc = 0;
    if tau != 0.0 {
        lastv = if applyleft { m } else { n };
        let mut i = 1 + (lastv - 1) * incv;
        while lastv > 0 && v[i - 1] == 0.0 {
            lastv -= 1;
            if lastv > 0 {
                i -= incv;
            }
        }
        lastc = if applyleft {
            iladlc(lastv, n, c, ldc)
        } else {
            iladlr(m, lastv, c, ldc)
        };
    }
    if lastv > 0 {
        if applyleft {
            dgemv(b'T', lastv, lastc, 1.0, c, ldc, v, incv, 0.0, work, 1)?;
            dger(lastv, lastc, -tau, v, incv, work, 1, c, ldc)?;
        } else {
            dgemv(b'N', lastc, lastv, 1.0, c, ldc, v, incv, 0.0, work, 1)?;
            dger(lastc, lastv, -tau, work, 1, v, incv, c, ldc)?;
        }
    }
    Ok(())
}

/// Output of [`dgeequ`].
#[derive(Debug, Clone, PartialEq)]
pub struct Equil {
    /// Row scale factors.
    pub r: Vec<f64>,
    /// Column scale factors.
    pub c: Vec<f64>,
    /// Ratio of smallest to largest row scale.
    pub rowcnd: f64,
    /// Ratio of smallest to largest column scale.
    pub colcnd: f64,
    /// Largest absolute element.
    pub amax: f64,
    /// `INFO` (`> 0`: an exactly zero row or column).
    pub info: usize,
}

/// `DGEEQU` (`SRC/dgeequ.f`): row and column equilibration factors.
#[must_use]
pub fn dgeequ(m: usize, n: usize, a: &[f64], lda: usize) -> Equil {
    let mut out = Equil {
        r: vec![0.0; m],
        c: vec![0.0; n],
        rowcnd: 0.0,
        colcnd: 0.0,
        amax: 0.0,
        info: 0,
    };
    if m == 0 || n == 0 {
        out.rowcnd = 1.0;
        out.colcnd = 1.0;
        return out;
    }
    let smlnum = dlamch::SFMIN;
    let bignum = 1.0 / smlnum;
    let r = &mut out.r;
    for j in 1..=n {
        for i in 1..=m {
            r[i - 1] = fmax(r[i - 1], a[ix(i, j, lda)].abs());
        }
    }
    let mut rcmin = bignum;
    let mut rcmax = 0.0_f64;
    for &ri in r.iter() {
        rcmax = fmax(rcmax, ri);
        rcmin = fmin(rcmin, ri);
    }
    out.amax = rcmax;
    if rcmin == 0.0 {
        out.info = r.iter().position(|&v| v == 0.0).map_or(0, |p| p + 1);
        return out;
    }
    for ri in r.iter_mut() {
        *ri = 1.0 / fmin(fmax(*ri, smlnum), bignum);
    }
    out.rowcnd = fmax(rcmin, smlnum) / fmin(rcmax, bignum);
    let c = &mut out.c;
    for j in 1..=n {
        for i in 1..=m {
            c[j - 1] = fmax(c[j - 1], a[ix(i, j, lda)].abs() * out.r[i - 1]);
        }
    }
    let mut rcmin = bignum;
    let mut rcmax = 0.0_f64;
    for &cj in c.iter() {
        rcmin = fmin(rcmin, cj);
        rcmax = fmax(rcmax, cj);
    }
    if rcmin == 0.0 {
        out.info = m + c.iter().position(|&v| v == 0.0).map_or(0, |p| p + 1);
        return out;
    }
    for cj in c.iter_mut() {
        *cj = 1.0 / fmin(fmax(*cj, smlnum), bignum);
    }
    out.colcnd = fmax(rcmin, smlnum) / fmin(rcmax, bignum);
    out
}

/// `DLAQGE` (`SRC/dlaqge.f`): apply the equilibration; returns `EQUED`
/// (`b'N'`, `b'R'`, `b'C'` or `b'B'`).
#[allow(clippy::too_many_arguments)]
pub fn dlaqge(
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    r: &[f64],
    c: &[f64],
    rowcnd: f64,
    colcnd: f64,
    amax: f64,
) -> u8 {
    const THRESH: f64 = 0.1;
    if m == 0 || n == 0 {
        return b'N';
    }
    let small = dlamch::SFMIN / dlamch::PREC;
    let large = 1.0 / small;
    if rowcnd >= THRESH && amax >= small && amax <= large {
        if colcnd >= THRESH {
            b'N'
        } else {
            for j in 1..=n {
                let cj = c[j - 1];
                for i in 1..=m {
                    a[ix(i, j, lda)] = cj * a[ix(i, j, lda)];
                }
            }
            b'C'
        }
    } else if colcnd >= THRESH {
        for j in 1..=n {
            for i in 1..=m {
                a[ix(i, j, lda)] = r[i - 1] * a[ix(i, j, lda)];
            }
        }
        b'R'
    } else {
        for j in 1..=n {
            let cj = c[j - 1];
            for i in 1..=m {
                a[ix(i, j, lda)] = cj * r[i - 1] * a[ix(i, j, lda)];
            }
        }
        b'B'
    }
}

/// Reverse-communication state of [`dlacn2`] (`KASE`, `ISAVE`, `ISGN`).
#[derive(Debug, Clone, Default)]
pub struct Lacn2State {
    /// `KASE`: 0 on entry and on completion; 1 or 2 asks for `A*x` or
    /// `A**T*x`.
    pub kase: i32,
    isave: [usize; 3],
    isgn: Vec<i32>,
}

/// Fortran `NINT` of +-1.
#[allow(clippy::cast_possible_truncation)]
fn nint(x: f64) -> i32 {
    x.round() as i32
}

/// `DLACN2` (`SRC/dlacn2.f`): one-norm estimation by reverse
/// communication. `v` and `x` have length `n`; `est` is updated.
#[allow(clippy::cast_precision_loss)]
pub fn dlacn2(
    n: usize,
    v: &mut [f64],
    x: &mut [f64],
    est: &mut f64,
    st: &mut Lacn2State,
) {
    const ITMAX: usize = 5;
    if st.kase == 0 {
        for xi in x.iter_mut().take(n) {
            *xi = 1.0 / n as f64;
        }
        st.kase = 1;
        st.isave[0] = 1;
        st.isgn = vec![0; n];
        return;
    }
    // GO TO (20, 40, 70, 110, 140) ISAVE(1)
    let mut label = match st.isave[0] {
        1 => 20,
        2 => 40,
        3 => 70,
        4 => 110,
        _ => 140,
    };
    loop {
        match label {
            20 => {
                if n == 1 {
                    v[0] = x[0];
                    *est = v[0].abs();
                    label = 150;
                    continue;
                }
                *est = dasum(n, x, 1);
                for i in 0..n {
                    x[i] = if x[i] >= 0.0 { 1.0 } else { -1.0 };
                    st.isgn[i] = nint(x[i]);
                }
                st.kase = 2;
                st.isave[0] = 2;
                return;
            }
            40 => {
                st.isave[1] = idamax(n, x, 1);
                st.isave[2] = 2;
                label = 50;
            }
            50 => {
                for xi in x.iter_mut().take(n) {
                    *xi = 0.0;
                }
                x[st.isave[1] - 1] = 1.0;
                st.kase = 1;
                st.isave[0] = 3;
                return;
            }
            70 => {
                dcopy(n, x, 1, v, 1);
                let estold = *est;
                *est = dasum(n, v, 1);
                let mut differ = false;
                for i in 0..n {
                    let xs = if x[i] >= 0.0 { 1.0 } else { -1.0 };
                    if nint(xs) != st.isgn[i] {
                        differ = true;
                        break;
                    }
                }
                if !differ || *est <= estold {
                    label = 120;
                    continue;
                }
                for i in 0..n {
                    x[i] = if x[i] >= 0.0 { 1.0 } else { -1.0 };
                    st.isgn[i] = nint(x[i]);
                }
                st.kase = 2;
                st.isave[0] = 4;
                return;
            }
            110 => {
                let jlast = st.isave[1];
                st.isave[1] = idamax(n, x, 1);
                if x[jlast - 1] != x[st.isave[1] - 1].abs()
                    && st.isave[2] < ITMAX
                {
                    st.isave[2] += 1;
                    label = 50;
                    continue;
                }
                label = 120;
            }
            120 => {
                let mut altsgn = 1.0_f64;
                for i in 1..=n {
                    x[i - 1] = altsgn * (1.0 + (i - 1) as f64 / (n - 1) as f64);
                    altsgn = -altsgn;
                }
                st.kase = 1;
                st.isave[0] = 5;
                return;
            }
            140 => {
                let temp = 2.0 * (dasum(n, x, 1) / (3 * n) as f64);
                if temp > *est {
                    dcopy(n, x, 1, v, 1);
                    *est = temp;
                }
                label = 150;
            }
            _ => {
                st.kase = 0;
                return;
            }
        }
    }
}

/// `DLATRS` (`SRC/dlatrs.f`): solve a triangular system with scaling to
/// prevent overflow. Returns `scale`; `cnorm` is computed when
/// `normin = 'N'` and reused otherwise.
///
/// # Errors
///
/// [`LapackError::UnportedPath`] for the scaled paths GYRE never reaches
/// (`TMAX > BIGNUM`, lines 348-409; the careful solve when the growth
/// bound is below `SMLNUM`, lines 567-833).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn dlatrs(
    uplo: u8,
    trans: u8,
    diag: u8,
    normin: u8,
    n: usize,
    a: &[f64],
    lda: usize,
    x: &mut [f64],
    cnorm: &mut [f64],
) -> Result<f64, LapackError> {
    let upper = lsame(uplo, b'U');
    let notran = lsame(trans, b'N');
    let nounit = lsame(diag, b'N');
    let scale = 1.0;
    if n == 0 {
        return Ok(scale);
    }
    let smlnum = dlamch::SFMIN / dlamch::PREC;
    let bignum = 1.0 / smlnum;
    if lsame(normin, b'N') {
        if upper {
            for j in 1..=n {
                cnorm[j - 1] = dasum(j - 1, &a[ix(1, j, lda)..], 1);
            }
        } else {
            for j in 1..n {
                cnorm[j - 1] = dasum(n - j, &a[ix(j + 1, j, lda)..], 1);
            }
            cnorm[n - 1] = 0.0;
        }
    }
    let imax = idamax(n, cnorm, 1);
    let tmax = cnorm[imax - 1];
    if tmax > bignum {
        return Err(unported("DLATRS", "348-409"));
    }
    let tscal = 1.0;
    let j = idamax(n, x, 1);
    let xmax = x[j - 1].abs();
    let mut xbnd = xmax;
    let mut grow;
    let order: Vec<usize> = if notran == upper {
        (1..=n).rev().collect()
    } else {
        (1..=n).collect()
    };
    if notran {
        // Bound on the growth of x for A*x = b (dlatrs.f:437-488)
        if nounit {
            grow = 1.0 / fmax(xbnd, smlnum);
            xbnd = grow;
            let mut exited = false;
            for &j in &order {
                if grow <= smlnum {
                    exited = true;
                    break;
                }
                let tjj = a[ix(j, j, lda)].abs();
                xbnd = fmin(xbnd, fmin(1.0, tjj) * grow);
                if tjj + cnorm[j - 1] >= smlnum {
                    grow *= tjj / (tjj + cnorm[j - 1]);
                } else {
                    grow = 0.0;
                }
            }
            if !exited {
                grow = xbnd;
            }
        } else {
            grow = fmin(1.0, 1.0 / fmax(xbnd, smlnum));
            for &j in &order {
                if grow <= smlnum {
                    break;
                }
                grow *= 1.0 / (1.0 + cnorm[j - 1]);
            }
        }
    } else {
        // Bound on the growth of x for A**T*x = b (dlatrs.f:510-557)
        if nounit {
            grow = 1.0 / fmax(xbnd, smlnum);
            xbnd = grow;
            let mut exited = false;
            for &j in &order {
                if grow <= smlnum {
                    exited = true;
                    break;
                }
                let xj = 1.0 + cnorm[j - 1];
                grow = fmin(grow, xbnd / xj);
                let tjj = a[ix(j, j, lda)].abs();
                if xj > tjj {
                    xbnd *= tjj / xj;
                }
            }
            if !exited {
                grow = fmin(grow, xbnd);
            }
        } else {
            grow = fmin(1.0, 1.0 / fmax(xbnd, smlnum));
            for &j in &order {
                if grow <= smlnum {
                    break;
                }
                let xj = 1.0 + cnorm[j - 1];
                grow /= xj;
            }
        }
    }
    if grow * tscal > smlnum {
        dtrsv(uplo, trans, diag, n, a, lda, x, 1)?;
    } else {
        return Err(unported("DLATRS", "567-833"));
    }
    Ok(scale)
}
