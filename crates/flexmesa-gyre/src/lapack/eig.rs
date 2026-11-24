// SPDX-License-Identifier: GPL-3.0-only
//! Reference LAPACK 3.12.0 real nonsymmetric eigensolver (`SRC/dgeev.f` and
//! callees: `dgebal.f`, `dgebak.f`, `dgehrd.f`, `dgehd2.f`, `dorghr.f`,
//! `dhseqr.f`, `dlahqr.f`, `dlanv2.f`, `dlaln2.f`, `dlaset.f`, `dtrevc3.f`,
//! `dlascl.f` `'G'`), double precision, as the MESA SDK `26.6.1` builds them
//! (`gfortran -O2 -fPIC`; `dlanv2` with CRMATH `log_r_dp`).

use super::aux::{
    dlacpy, dlange, dlarf, dlarfg, dlartg, fmax, fmin, sign,
};
use super::blas::{daxpy, dcopy, ddot, dgemm, dgemv, dnrm2, drot_in, dscal, dswap_in, idamax};
use super::svd::dorgqr;
use super::zaux::dladiv;
use super::{dlamch, ix, lsame, LapackError};

const NB32: usize = 32;
const NX128: usize = 128;
const NB64: usize = 64;
const NMIN75: usize = 75;

fn bad(routine: &'static str, info: i32) -> LapackError {
    LapackError::IllegalArgument { routine, info }
}

fn unported(routine: &'static str, lines: &'static str) -> LapackError {
    LapackError::UnportedPath { routine, lines }
}

fn pow2_i32(exp: i32) -> f64 {
    #[allow(clippy::cast_sign_loss)]
    let biased = u64::from((1023 + exp) as u32);
    f64::from_bits(biased << 52)
}

/// Output of [`dgeev`].
#[derive(Debug, Clone, PartialEq)]
pub struct Dgeev {
    /// Real parts of the `n` eigenvalues.
    pub wr: Vec<f64>,
    /// Imaginary parts of the `n` eigenvalues.
    pub wi: Vec<f64>,
    /// `WORK(1)` on exit (`MAXWRK`).
    pub work1: f64,
    /// `INFO` from `DHSEQR` (`0` on convergence).
    pub info: usize,
}

/// `DLASET` (`SRC/dlaset.f`): initialize an `m x n` matrix with `alpha`
/// off-diagonal (`'U'` upper, `'L'` lower, otherwise full) and `beta`
/// on the diagonal.
pub fn dlaset(
    uplo: u8,
    m: usize,
    n: usize,
    alpha: f64,
    beta: f64,
    a: &mut [f64],
    lda: usize,
) {
    if lsame(uplo, b'U') {
        for j in 2..=n {
            for i in 1..=m.min(j - 1) {
                a[ix(i, j, lda)] = alpha;
            }
        }
    } else if lsame(uplo, b'L') {
        for j in 1..=m.min(n) {
            for i in j + 1..=m {
                a[ix(i, j, lda)] = alpha;
            }
        }
    } else {
        for j in 1..=n {
            for i in 1..=m {
                a[ix(i, j, lda)] = alpha;
            }
        }
    }
    for i in 1..=m.min(n) {
        a[ix(i, i, lda)] = beta;
    }
}

/// `DLASCL` (`SRC/dlascl.f`), `TYPE = 'G'`: multiply `m x n` `A` by
/// `cto/cfrom` without overflow/underflow.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] on invalid dimensions or `NaN`/`0`.
#[allow(clippy::float_cmp)]
pub fn dlascl_g(
    cfrom: f64,
    cto: f64,
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
) -> Result<(), LapackError> {
    if cfrom == 0.0 || cfrom.is_nan() {
        return Err(bad("DLASCL", -4));
    }
    if cto.is_nan() {
        return Err(bad("DLASCL", -5));
    }
    if lda < m.max(1) {
        return Err(bad("DLASCL", -9));
    }
    if n == 0 || m == 0 {
        return Ok(());
    }
    let smlnum = dlamch::SFMIN;
    let bignum = 1.0 / smlnum;
    let mut cfromc = cfrom;
    let mut ctoc = cto;
    loop {
        let cfrom1 = cfromc * smlnum;
        let (mul, done);
        if cfrom1 == cfromc {
            mul = ctoc / cfromc;
            done = true;
        } else {
            let cto1 = ctoc / bignum;
            if cto1 == ctoc {
                mul = ctoc;
                done = true;
                cfromc = 1.0;
            } else if cfrom1.abs() > ctoc.abs() && ctoc != 0.0 {
                mul = smlnum;
                done = false;
                cfromc = cfrom1;
            } else if cto1.abs() > cfromc.abs() {
                mul = bignum;
                done = false;
                ctoc = cto1;
            } else {
                mul = ctoc / cfromc;
                done = true;
                if mul == 1.0 {
                    return Ok(());
                }
            }
        }
        for j in 1..=n {
            for i in 1..=m {
                a[ix(i, j, lda)] *= mul;
            }
        }
        if done {
            break;
        }
    }
    Ok(())
}

/// `DGEBAL` (`SRC/dgebal.f`): balance `n x n` matrix `A`. Returns `(ilo, ihi)`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] on invalid `job`/`lda` or `NaN`.
#[allow(clippy::cast_precision_loss, clippy::float_cmp)]
pub fn dgebal(
    job: u8,
    n: usize,
    a: &mut [f64],
    lda: usize,
    scale: &mut [f64],
) -> Result<(usize, usize), LapackError> {
    const SCLFAC: f64 = 2.0;
    const FACTOR: f64 = 0.95;

    let want_n = lsame(job, b'N');
    let want_p = lsame(job, b'P');
    let want_s = lsame(job, b'S');
    let want_b = lsame(job, b'B');
    if !want_n && !want_p && !want_s && !want_b {
        return Err(bad("DGEBAL", -1));
    }
    if lda < n.max(1) {
        return Err(bad("DGEBAL", -4));
    }
    if n == 0 {
        return Ok((1, 0));
    }
    if want_n {
        for s in &mut scale[..n] {
            *s = 1.0;
        }
        return Ok((1, n));
    }

    let mut k = 1_usize;
    let mut l = n;

    if !want_s {
        // Row and column exchange (dgebal.f:246-308)
        let mut noconv = true;
        while noconv {
            noconv = false;
            for i in (1..=l).rev() {
                let mut canswap = true;
                for j in 1..=l {
                    if i != j && a[ix(i, j, lda)] != 0.0 {
                        canswap = false;
                        break;
                    }
                }
                if canswap {
                    scale[l - 1] = i as f64;
                    if i != l {
                        dswap_in(l, a, ix(1, i, lda), 1, ix(1, l, lda), 1);
                        dswap_in(
                            n - k + 1,
                            a,
                            ix(i, k, lda),
                            lda,
                            ix(l, k, lda),
                            lda,
                        );
                    }
                    noconv = true;
                    if l == 1 {
                        return Ok((1, 1));
                    }
                    l -= 1;
                }
            }
        }

        noconv = true;
        while noconv {
            noconv = false;
            let k_start = k;
            for j in k_start..=l {
                let mut canswap = true;
                for i in k..=l {
                    if i != j && a[ix(i, j, lda)] != 0.0 {
                        canswap = false;
                        break;
                    }
                }
                if canswap {
                    scale[k - 1] = j as f64;
                    if j != k {
                        dswap_in(l, a, ix(1, j, lda), 1, ix(1, k, lda), 1);
                        dswap_in(
                            n - k + 1,
                            a,
                            ix(j, k, lda),
                            lda,
                            ix(k, k, lda),
                            lda,
                        );
                    }
                    noconv = true;
                    k += 1;
                }
            }
        }
    }

    for s in &mut scale[k - 1..l] {
        *s = 1.0;
    }

    if want_p {
        return Ok((k, l));
    }

    let sfmin1 = dlamch::SFMIN / dlamch::PREC;
    let sfmax1 = 1.0 / sfmin1;
    let sfmin2 = sfmin1 * SCLFAC;
    let sfmax2 = 1.0 / sfmin2;

    let mut noconv = true;
    while noconv {
        noconv = false;
        for i in k..=l {
            let mut c = dnrm2(l - k + 1, &a[ix(k, i, lda)..], 1);
            let mut r = dnrm2(l - k + 1, &a[ix(i, k, lda)..], lda);
            let ica = idamax(l, &a[ix(1, i, lda)..], 1);
            let mut ca = a[ix(ica, i, lda)].abs();
            let ira = idamax(n - k + 1, &a[ix(i, k, lda)..], lda);
            let mut ra = a[ix(i, ira + k - 1, lda)].abs();
            if c == 0.0 || r == 0.0 {
                continue;
            }
            if (c + ca + r + ra).is_nan() {
                return Err(bad("DGEBAL", -3));
            }
            let mut g = r / SCLFAC;
            let mut f = 1.0_f64;
            let s = c + r;
            while c < g
                && fmax(fmax(f, c), ca) < sfmax2
                && fmin(fmin(r, g), ra) > sfmin2
            {
                f *= SCLFAC;
                c *= SCLFAC;
                ca *= SCLFAC;
                r /= SCLFAC;
                g /= SCLFAC;
                ra /= SCLFAC;
            }
            g = c / SCLFAC;
            while g >= r
                && fmax(r, ra) < sfmax2
                && fmin(fmin(fmin(f, c), g), ca) > sfmin2
            {
                f /= SCLFAC;
                c /= SCLFAC;
                g /= SCLFAC;
                ca /= SCLFAC;
                r *= SCLFAC;
                ra *= SCLFAC;
            }
            if (c + r) >= FACTOR * s {
                continue;
            }
            if f < 1.0 && scale[i - 1] < 1.0 && f * scale[i - 1] <= sfmin1 {
                continue;
            }
            if f > 1.0 && scale[i - 1] > 1.0 && scale[i - 1] >= sfmax1 / f {
                continue;
            }
            g = 1.0 / f;
            scale[i - 1] *= f;
            noconv = true;
            dscal(n - k + 1, g, &mut a[ix(i, k, lda)..], lda);
            dscal(l, f, &mut a[ix(1, i, lda)..], 1);
        }
    }
    Ok((k, l))
}

/// `DGEBAK` (`SRC/dgebak.f`): back-transform eigenvectors from `DGEBAL`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] on invalid arguments.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn dgebak(
    job: u8,
    side: u8,
    n: usize,
    ilo: usize,
    ihi: usize,
    scale: &[f64],
    m: usize,
    v: &mut [f64],
    ldv: usize,
) -> Result<(), LapackError> {
    let rightv = lsame(side, b'R');
    let leftv = lsame(side, b'L');
    if !lsame(job, b'N')
        && !lsame(job, b'P')
        && !lsame(job, b'S')
        && !lsame(job, b'B')
    {
        return Err(bad("DGEBAK", -1));
    }
    if !rightv && !leftv {
        return Err(bad("DGEBAK", -2));
    }
    if ilo < 1 || ilo > n.max(1) {
        return Err(bad("DGEBAK", -4));
    }
    if ihi < ilo.min(n) || ihi > n {
        return Err(bad("DGEBAK", -5));
    }
    if ldv < n.max(1) {
        return Err(bad("DGEBAK", -9));
    }
    if n == 0 || m == 0 || lsame(job, b'N') {
        return Ok(());
    }
    if ilo != ihi && (lsame(job, b'S') || lsame(job, b'B')) {
        if rightv {
            for i in ilo..=ihi {
                let s = scale[i - 1];
                dscal(m, s, &mut v[ix(i, 1, ldv)..], ldv);
            }
        }
        if leftv {
            for i in ilo..=ihi {
                let s = 1.0 / scale[i - 1];
                dscal(m, s, &mut v[ix(i, 1, ldv)..], ldv);
            }
        }
    }
    if lsame(job, b'P') || lsame(job, b'B') {
        for ii in 1..=n {
            let mut i = ii;
            if i >= ilo && i <= ihi {
                continue;
            }
            if i < ilo {
                i = ilo - ii;
            }
            let k = scale[i - 1] as usize;
            if k == i {
                continue;
            }
            dswap_in(m, v, ix(i, 1, ldv), ldv, ix(k, 1, ldv), ldv);
        }
    }
    Ok(())
}

/// `DGEHD2` (`SRC/dgehd2.f`): unblocked reduction to upper Hessenberg form.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] or [`dlarf`] errors.
pub fn dgehd2(
    n: usize,
    ilo: usize,
    ihi: usize,
    a: &mut [f64],
    lda: usize,
    tau: &mut [f64],
    work: &mut [f64],
) -> Result<(), LapackError> {
    if ilo < 1 || ilo > n.max(1) {
        return Err(bad("DGEHD2", -2));
    }
    if ihi < ilo.min(n) || ihi > n {
        return Err(bad("DGEHD2", -3));
    }
    if lda < n.max(1) {
        return Err(bad("DGEHD2", -5));
    }
    if ihi <= ilo {
        return Ok(());
    }
    for i in ilo..ihi {
        let mut alpha = a[ix(i + 1, i, lda)];
        let xs = ix((i + 2).min(n), i, lda);
        let mut v = a[xs..].to_vec();
        tau[i - 1] = dlarfg(ihi - i, &mut alpha, &mut v, 1);
        a[xs..].copy_from_slice(&v);
        a[ix(i + 1, i, lda)] = 1.0;

        let hv = a[ix(i + 1, i, lda)..ix(ihi, i, lda) + 1].to_vec();
        dlarf(
            b'R',
            ihi,
            ihi - i,
            &hv,
            1,
            tau[i - 1],
            &mut a[ix(1, i + 1, lda)..],
            lda,
            work,
        )?;
        dlarf(
            b'L',
            ihi - i,
            n - i,
            &hv,
            1,
            tau[i - 1],
            &mut a[ix(i + 1, i + 1, lda)..],
            lda,
            work,
        )?;
        a[ix(i + 1, i, lda)] = alpha;
    }
    Ok(())
}

/// `DGEHRD` (`SRC/dgehrd.f`): reduction to upper Hessenberg form
/// (unblocked for `nh <= 128`). Returns `WORK(1)` (`LWKOPT`).
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] or [`LapackError::UnportedPath`] for
/// blocked `nh > 128`.
#[allow(clippy::cast_precision_loss)]
pub fn dgehrd(
    n: usize,
    ilo: usize,
    ihi: usize,
    a: &mut [f64],
    lda: usize,
    tau: &mut [f64],
    lwork: isize,
) -> Result<f64, LapackError> {
    const TSIZE: usize = (64 + 1) * 64;
    let lquery = lwork == -1;
    if ilo < 1 || ilo > n.max(1) {
        return Err(bad("DGEHRD", -2));
    }
    if ihi < ilo.min(n) || ihi > n {
        return Err(bad("DGEHRD", -3));
    }
    if lda < n.max(1) {
        return Err(bad("DGEHRD", -5));
    }
    if !lquery && lwork < (n.max(1) as isize) {
        return Err(bad("DGEHRD", -8));
    }
    let lwkopt = (n * NB32 + TSIZE) as f64;
    if lquery {
        return Ok(lwkopt);
    }
    for i in 1..ilo {
        tau[i - 1] = 0.0;
    }
    for i in ihi.max(1)..n {
        tau[i - 1] = 0.0;
    }
    let nh = ihi + 1 - ilo;
    if nh <= 1 {
        return Ok(1.0);
    }
    if nh > NX128 {
        return Err(unported("DGEHRD", "297-342"));
    }
    let mut work = vec![0.0_f64; n.max(1)];
    dgehd2(n, ilo, ihi, a, lda, tau, &mut work)?;
    Ok(lwkopt)
}

/// `DORGHR` (`SRC/dorghr.f`): generate orthogonal `Q` from `DGEHRD`.
/// Returns `WORK(1)` (`LWKOPT`).
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] or [`dorgqr`] errors.
#[allow(clippy::cast_precision_loss)]
pub fn dorghr(
    n: usize,
    ilo: usize,
    ihi: usize,
    a: &mut [f64],
    lda: usize,
    tau: &[f64],
    lwork: isize,
) -> Result<f64, LapackError> {
    let lquery = lwork == -1;
    if ilo < 1 || ilo > n.max(1) {
        return Err(bad("DORGHR", -2));
    }
    if ihi < ilo.min(n) || ihi > n {
        return Err(bad("DORGHR", -3));
    }
    let nh = ihi - ilo;
    if lda < n.max(1) {
        return Err(bad("DORGHR", -5));
    }
    if !lquery && lwork < (nh.max(1) as isize) {
        return Err(bad("DORGHR", -8));
    }
    let lwkopt = (nh.max(1) * NB32) as f64;
    if lquery {
        return Ok(lwkopt);
    }
    if n == 0 {
        return Ok(1.0);
    }
    for j in (ilo + 1..=ihi).rev() {
        for i in 1..j {
            a[ix(i, j, lda)] = 0.0;
        }
        for i in j + 1..=ihi {
            a[ix(i, j, lda)] = a[ix(i, j - 1, lda)];
        }
        for i in ihi + 1..=n {
            a[ix(i, j, lda)] = 0.0;
        }
    }
    for j in 1..=ilo {
        for i in 1..=n {
            a[ix(i, j, lda)] = 0.0;
        }
        a[ix(j, j, lda)] = 1.0;
    }
    for j in ihi + 1..=n {
        for i in 1..=n {
            a[ix(i, j, lda)] = 0.0;
        }
        a[ix(j, j, lda)] = 1.0;
    }
    if nh > 0 {
        dorgqr(nh, nh, nh, &mut a[ix(ilo + 1, ilo + 1, lda)..], lda, &tau[ilo - 1..])?;
    }
    Ok(lwkopt)
}

/// Output of [`dlanv2`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dlanv2 {
    /// Updated `(1, 1)` entry.
    pub a: f64,
    /// Updated `(1, 2)` entry.
    pub b: f64,
    /// Updated `(2, 1)` entry.
    pub c: f64,
    /// Updated `(2, 2)` entry.
    pub d: f64,
    /// First eigenvalue real part.
    pub rt1r: f64,
    /// First eigenvalue imaginary part.
    pub rt1i: f64,
    /// Second eigenvalue real part.
    pub rt2r: f64,
    /// Second eigenvalue imaginary part.
    pub rt2i: f64,
    /// Rotation cosine.
    pub cs: f64,
    /// Rotation sine.
    pub sn: f64,
}

/// `DLANV2` (`SRC/dlanv2.f`, SDK CRMATH build using `log_r_dp`):
/// compute Schur factorization of a real `2 x 2` nonsymmetric matrix.
#[must_use]
#[allow(clippy::float_cmp, clippy::cast_possible_truncation)]
pub fn dlanv2(
    mut a: f64,
    mut b: f64,
    mut c: f64,
    mut d: f64,
) -> Dlanv2 {
    const MULTPL: f64 = 4.0;
    let safmin = dlamch::SFMIN;
    let eps = dlamch::PREC;
    let exp2 = (flexmesa_math::log(safmin / eps)
        / flexmesa_math::log(dlamch::BASE)
        / 2.0) as i32;
    let safmn2 = pow2_i32(exp2);
    let safmx2 = 1.0 / safmn2;
    let (mut cs, mut sn);

    if c == 0.0 {
        cs = 1.0;
        sn = 0.0;
    } else if b == 0.0 {
        cs = 0.0;
        sn = 1.0;
        std::mem::swap(&mut d, &mut a);
        b = -c;
        c = 0.0;
    } else if (a - d) == 0.0 && sign(1.0, b) != sign(1.0, c) {
        cs = 1.0;
        sn = 0.0;
    } else {
        let mut temp = a - d;
        let mut p = 0.5 * temp;
        let bcmax = fmax(b.abs(), c.abs());
        let bcmis = fmin(b.abs(), c.abs()) * sign(1.0, b) * sign(1.0, c);
        let mut scale = fmax(p.abs(), bcmax);
        let z = (p / scale) * p + (bcmax / scale) * bcmis;
        if z >= MULTPL * eps {
            let z = p + sign(scale.sqrt() * z.sqrt(), p);
            a = d + z;
            d -= (bcmax / z) * bcmis;
            let tau = super::aux::dlapy2(c, z);
            cs = z / tau;
            sn = c / tau;
            b -= c;
            c = 0.0;
        } else {
            let mut sigma = b + c;
            let mut count = 0_i32;
            loop {
                count += 1;
                scale = fmax(temp.abs(), sigma.abs());
                if scale >= safmx2 {
                    sigma *= safmn2;
                    temp *= safmn2;
                    if count <= 20 {
                        continue;
                    }
                }
                if scale <= safmn2 {
                    sigma *= safmx2;
                    temp *= safmx2;
                    if count <= 20 {
                        continue;
                    }
                }
                break;
            }
            p = 0.5 * temp;
            let mut tau = super::aux::dlapy2(sigma, temp);
            cs = (0.5 * (1.0 + sigma.abs() / tau)).sqrt();
            sn = -(p / (tau * cs)) * sign(1.0, sigma);
            let aa = a * cs + b * sn;
            let bb = -a * sn + b * cs;
            let cc = c * cs + d * sn;
            let dd = -c * sn + d * cs;
            a = aa * cs + cc * sn;
            b = bb * cs + dd * sn;
            c = -aa * sn + cc * cs;
            d = -bb * sn + dd * cs;
            temp = 0.5 * (a + d);
            a = temp;
            d = temp;
            if c != 0.0 {
                if b != 0.0 {
                    if sign(1.0, b) == sign(1.0, c) {
                        let sab = b.abs().sqrt();
                        let sac = c.abs().sqrt();
                        p = sign(sab * sac, c);
                        tau = 1.0 / (b + c).abs().sqrt();
                        a = temp + p;
                        d = temp - p;
                        b -= c;
                        c = 0.0;
                        let cs1 = sab * tau;
                        let sn1 = sac * tau;
                        temp = cs * cs1 - sn * sn1;
                        sn = cs * sn1 + sn * cs1;
                        cs = temp;
                    }
                } else {
                    b = -c;
                    c = 0.0;
                    temp = cs;
                    cs = -sn;
                    sn = temp;
                }
            }
        }
    }
    let rt1r = a;
    let rt2r = d;
    let (rt1i, rt2i) = if c == 0.0 {
        (0.0, 0.0)
    } else {
        let im = b.abs().sqrt() * c.abs().sqrt();
        (im, -im)
    };
    Dlanv2 { a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn }
}

/// `DLALN2` (`SRC/dlaln2.f`): solve `(ca*op(A) - (wr + i*wi)*D)*X = scale*B`
/// for a `1x1` or `2x2` real matrix `A`. Returns `(scale, xnorm, info)`.
#[allow(clippy::too_many_lines, clippy::float_cmp)]
pub fn dlaln2(
    ltrans: bool,
    na: usize,
    nw: usize,
    smin: f64,
    ca: f64,
    a: &[f64],
    lda: usize,
    d1: f64,
    d2: f64,
    b: &[f64],
    ldb: usize,
    wr: f64,
    wi: f64,
    x: &mut [f64],
    ldx: usize,
) -> (f64, f64, usize) {
    const RSWAP: [bool; 4] = [false, true, false, true];
    const ZSWAP: [bool; 4] = [false, false, true, true];
    const IPIVOT: [[usize; 4]; 4] = [
        [1, 2, 3, 4],
        [2, 1, 4, 3],
        [3, 4, 1, 2],
        [4, 3, 2, 1],
    ];

    let smlnum = 2.0 * dlamch::SFMIN;
    let bignum = 1.0 / smlnum;
    let smini = fmax(smin, smlnum);
    let mut info = 0_usize;
    let mut scale = 1.0_f64;
    let mut xnorm;

    if na == 1 {
        if nw == 1 {
            let mut csr = ca * a[0] - wr * d1;
            let mut cnorm = csr.abs();
            if cnorm < smini {
                csr = smini;
                cnorm = smini;
                info = 1;
            }
            let bnorm = b[0].abs();
            if cnorm < 1.0 && bnorm > 1.0 && bnorm > bignum * cnorm {
                scale = 1.0 / bnorm;
            }
            x[0] = (b[0] * scale) / csr;
            xnorm = x[0].abs();
        } else {
            let mut csr = ca * a[0] - wr * d1;
            let mut csi = -wi * d1;
            let mut cnorm = csr.abs() + csi.abs();
            if cnorm < smini {
                csr = smini;
                csi = 0.0;
                cnorm = smini;
                info = 1;
            }
            let bnorm = b[ix(1, 1, ldb)].abs() + b[ix(1, 2, ldb)].abs();
            if cnorm < 1.0 && bnorm > 1.0 && bnorm > bignum * cnorm {
                scale = 1.0 / bnorm;
            }
            let (xr, xi) =
                dladiv(scale * b[ix(1, 1, ldb)], scale * b[ix(1, 2, ldb)], csr, csi);
            x[ix(1, 1, ldx)] = xr;
            x[ix(1, 2, ldx)] = xi;
            xnorm = xr.abs() + xi.abs();
        }
        return (scale, xnorm, info);
    }

    // 2x2 system: col-major CR[0..4] = [cr(1,1), cr(2,1), cr(1,2), cr(2,2)]
    let cr11 = ca * a[ix(1, 1, lda)] - wr * d1;
    let cr22 = ca * a[ix(2, 2, lda)] - wr * d2;
    let (cr12, cr21) = if ltrans {
        (ca * a[ix(2, 1, lda)], ca * a[ix(1, 2, lda)])
    } else {
        (ca * a[ix(1, 2, lda)], ca * a[ix(2, 1, lda)])
    };
    let crv = [cr11, cr21, cr12, cr22];

    if nw == 1 {
        let mut cmax = 0.0_f64;
        let mut icmax = 1_usize;
        for (j, &v) in crv.iter().enumerate() {
            if v.abs() > cmax {
                cmax = v.abs();
                icmax = j + 1;
            }
        }
        if cmax < smini {
            let bnorm = fmax(b[ix(1, 1, ldb)].abs(), b[ix(2, 1, ldb)].abs());
            if smini < 1.0 && bnorm > 1.0 && bnorm > bignum * smini {
                scale = 1.0 / bnorm;
            }
            let temp = scale / smini;
            x[ix(1, 1, ldx)] = temp * b[ix(1, 1, ldb)];
            x[ix(2, 1, ldx)] = temp * b[ix(2, 1, ldb)];
            return (scale, temp * bnorm, 1);
        }
        let piv = &IPIVOT[icmax - 1];
        let ur11 = crv[icmax - 1];
        let cr21p = crv[piv[1] - 1];
        let ur12 = crv[piv[2] - 1];
        let cr22p = crv[piv[3] - 1];
        let ur11r = 1.0 / ur11;
        let lr21 = ur11r * cr21p;
        let mut ur22 = cr22p - ur12 * lr21;
        if ur22.abs() < smini {
            ur22 = smini;
            info = 1;
        }
        let (br1, mut br2) = if RSWAP[icmax - 1] {
            (b[ix(2, 1, ldb)], b[ix(1, 1, ldb)])
        } else {
            (b[ix(1, 1, ldb)], b[ix(2, 1, ldb)])
        };
        br2 -= lr21 * br1;
        let bbnd = fmax((br1 * (ur22 * ur11r)).abs(), br2.abs());
        if bbnd > 1.0 && ur22.abs() < 1.0 && bbnd >= bignum * ur22.abs() {
            scale = 1.0 / bbnd;
        }
        let xr2 = (br2 * scale) / ur22;
        let xr1 = (scale * br1) * ur11r - xr2 * (ur11r * ur12);
        if ZSWAP[icmax - 1] {
            x[ix(1, 1, ldx)] = xr2;
            x[ix(2, 1, ldx)] = xr1;
        } else {
            x[ix(1, 1, ldx)] = xr1;
            x[ix(2, 1, ldx)] = xr2;
        }
        xnorm = fmax(xr1.abs(), xr2.abs());
        if xnorm > 1.0 && cmax > 1.0 && xnorm > bignum / cmax {
            let temp = cmax / bignum;
            x[ix(1, 1, ldx)] *= temp;
            x[ix(2, 1, ldx)] *= temp;
            xnorm *= temp;
            scale *= temp;
        }
    } else {
        let civ = [-wi * d1, 0.0, 0.0, -wi * d2];
        let mut cmax = 0.0_f64;
        let mut icmax = 1_usize;
        for j in 0..4 {
            let m = crv[j].abs() + civ[j].abs();
            if m > cmax {
                cmax = m;
                icmax = j + 1;
            }
        }
        if cmax < smini {
            let bnorm = fmax(
                b[ix(1, 1, ldb)].abs() + b[ix(1, 2, ldb)].abs(),
                b[ix(2, 1, ldb)].abs() + b[ix(2, 2, ldb)].abs(),
            );
            if smini < 1.0 && bnorm > 1.0 && bnorm > bignum * smini {
                scale = 1.0 / bnorm;
            }
            let temp = scale / smini;
            x[ix(1, 1, ldx)] = temp * b[ix(1, 1, ldb)];
            x[ix(2, 1, ldx)] = temp * b[ix(2, 1, ldb)];
            x[ix(1, 2, ldx)] = temp * b[ix(1, 2, ldb)];
            x[ix(2, 2, ldx)] = temp * b[ix(2, 2, ldb)];
            return (scale, temp * bnorm, 1);
        }
        let piv = &IPIVOT[icmax - 1];
        let ur11 = crv[icmax - 1];
        let ui11 = civ[icmax - 1];
        let cr21p = crv[piv[1] - 1];
        let ci21p = civ[piv[1] - 1];
        let ur12 = crv[piv[2] - 1];
        let ui12 = civ[piv[2] - 1];
        let cr22p = crv[piv[3] - 1];
        let ci22p = civ[piv[3] - 1];
        let (ur11r, ui11r, lr21, li21, ur12s, ui12s, mut ur22, mut ui22);
        if icmax == 1 || icmax == 4 {
            if ur11.abs() > ui11.abs() {
                let temp = ui11 / ur11;
                ur11r = 1.0 / (ur11 * (1.0 + temp * temp));
                ui11r = -temp * ur11r;
            } else {
                let temp = ur11 / ui11;
                ui11r = -1.0 / (ui11 * (1.0 + temp * temp));
                ur11r = -temp * ui11r;
            }
            lr21 = cr21p * ur11r;
            li21 = cr21p * ui11r;
            ur12s = ur12 * ur11r;
            ui12s = ur12 * ui11r;
            ur22 = cr22p - ur12 * lr21;
            ui22 = ci22p - ur12 * li21;
        } else {
            ur11r = 1.0 / ur11;
            ui11r = 0.0;
            lr21 = cr21p * ur11r;
            li21 = ci21p * ur11r;
            ur12s = ur12 * ur11r;
            ui12s = ui12 * ur11r;
            ur22 = cr22p - ur12 * lr21 + ui12 * li21;
            ui22 = -ur12 * li21 - ui12 * lr21;
        }
        let u22abs = ur22.abs() + ui22.abs();
        if u22abs < smini {
            ur22 = smini;
            ui22 = 0.0;
            info = 1;
        }
        let (mut br1, mut bi1, mut br2, mut bi2) = if RSWAP[icmax - 1] {
            (
                b[ix(2, 1, ldb)],
                b[ix(2, 2, ldb)],
                b[ix(1, 1, ldb)],
                b[ix(1, 2, ldb)],
            )
        } else {
            (
                b[ix(1, 1, ldb)],
                b[ix(1, 2, ldb)],
                b[ix(2, 1, ldb)],
                b[ix(2, 2, ldb)],
            )
        };
        br2 = br2 - lr21 * br1 + li21 * bi1;
        bi2 = bi2 - li21 * br1 - lr21 * bi1;
        let bbnd = fmax(
            (br1.abs() + bi1.abs()) * (u22abs * (ur11r.abs() + ui11r.abs())),
            br2.abs() + bi2.abs(),
        );
        if bbnd > 1.0 && u22abs < 1.0 && bbnd >= bignum * u22abs {
            scale = 1.0 / bbnd;
            br1 *= scale;
            bi1 *= scale;
            br2 *= scale;
            bi2 *= scale;
        }
        let (xr2, xi2) = dladiv(br2, bi2, ur22, ui22);
        let xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2;
        let xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2;
        if ZSWAP[icmax - 1] {
            x[ix(1, 1, ldx)] = xr2;
            x[ix(2, 1, ldx)] = xr1;
            x[ix(1, 2, ldx)] = xi2;
            x[ix(2, 2, ldx)] = xi1;
        } else {
            x[ix(1, 1, ldx)] = xr1;
            x[ix(2, 1, ldx)] = xr2;
            x[ix(1, 2, ldx)] = xi1;
            x[ix(2, 2, ldx)] = xi2;
        }
        xnorm = fmax(xr1.abs() + xi1.abs(), xr2.abs() + xi2.abs());
        if xnorm > 1.0 && cmax > 1.0 && xnorm > bignum / cmax {
            let temp = cmax / bignum;
            x[ix(1, 1, ldx)] *= temp;
            x[ix(2, 1, ldx)] *= temp;
            x[ix(1, 2, ldx)] *= temp;
            x[ix(2, 2, ldx)] *= temp;
            xnorm *= temp;
            scale *= temp;
        }
    }
    (scale, xnorm, info)
}

/// `DLAHQR` (`SRC/dlahqr.f`): real double-shift QR algorithm on an upper
/// Hessenberg matrix `H`. Returns `INFO`.
#[allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines
)]
pub fn dlahqr(
    wantt: bool,
    wantz: bool,
    n: usize,
    ilo: usize,
    ihi: usize,
    h: &mut [f64],
    ldh: usize,
    wr: &mut [f64],
    wi: &mut [f64],
    iloz: usize,
    ihiz: usize,
    z: &mut [f64],
    ldz: usize,
) -> usize {
    const DAT1: f64 = 3.0 / 4.0;
    const DAT2: f64 = -0.4375;
    const KEXSH: usize = 10;

    if n == 0 {
        return 0;
    }
    if ilo == ihi {
        wr[ilo - 1] = h[ix(ilo, ilo, ldh)];
        wi[ilo - 1] = 0.0;
        return 0;
    }
    for j in ilo..=ihi.saturating_sub(3) {
        h[ix(j + 2, j, ldh)] = 0.0;
        h[ix(j + 3, j, ldh)] = 0.0;
    }
    if ilo + 2 <= ihi {
        h[ix(ihi, ihi - 2, ldh)] = 0.0;
    }

    let nh = ihi - ilo + 1;
    let nz = ihiz - iloz + 1;
    let safmin = dlamch::SFMIN;
    let ulp = dlamch::PREC;
    let smlnum = safmin * ((nh as f64) / ulp);

    let mut i1 = 1_usize;
    let mut i2 = n;
    let itmax = 30 * 10.max(nh);
    let mut kdefl = 0_usize;
    let mut v = [0.0_f64; 3];
    let mut i = ihi;

    while i >= ilo {
        let mut l = ilo;
        let mut split = false;
        for _its in 0..=itmax {
            let mut k = i;
            while k > l {
                if h[ix(k, k - 1, ldh)].abs() <= smlnum {
                    break;
                }
                let mut tst =
                    h[ix(k - 1, k - 1, ldh)].abs() + h[ix(k, k, ldh)].abs();
                if tst == 0.0 {
                    if k >= ilo + 2 {
                        tst += h[ix(k - 1, k - 2, ldh)].abs();
                    }
                    if k < ihi {
                        tst += h[ix(k + 1, k, ldh)].abs();
                    }
                }
                if h[ix(k, k - 1, ldh)].abs() <= ulp * tst {
                    let ab = fmax(
                        h[ix(k, k - 1, ldh)].abs(),
                        h[ix(k - 1, k, ldh)].abs(),
                    );
                    let ba = fmin(
                        h[ix(k, k - 1, ldh)].abs(),
                        h[ix(k - 1, k, ldh)].abs(),
                    );
                    let aa = fmax(
                        h[ix(k, k, ldh)].abs(),
                        (h[ix(k - 1, k - 1, ldh)] - h[ix(k, k, ldh)]).abs(),
                    );
                    let bb = fmin(
                        h[ix(k, k, ldh)].abs(),
                        (h[ix(k - 1, k - 1, ldh)] - h[ix(k, k, ldh)]).abs(),
                    );
                    let s = aa + ab;
                    if ba * (ab / s) <= fmax(smlnum, ulp * (bb * (aa / s))) {
                        break;
                    }
                }
                k -= 1;
            }
            l = k;
            if l > ilo {
                h[ix(l, l - 1, ldh)] = 0.0;
            }
            if l + 1 >= i {
                split = true;
                break;
            }
            kdefl += 1;
            if !wantt {
                i1 = l;
                i2 = i;
            }
            let (mut h11, mut h12, mut h21, mut h22);
            if kdefl.is_multiple_of(2 * KEXSH) {
                let s =
                    h[ix(i, i - 1, ldh)].abs() + h[ix(i - 1, i - 2, ldh)].abs();
                h11 = DAT1 * s + h[ix(i, i, ldh)];
                h12 = DAT2 * s;
                h21 = s;
                h22 = h11;
            } else if kdefl.is_multiple_of(KEXSH) {
                let s =
                    h[ix(l + 1, l, ldh)].abs() + h[ix(l + 2, l + 1, ldh)].abs();
                h11 = DAT1 * s + h[ix(l, l, ldh)];
                h12 = DAT2 * s;
                h21 = s;
                h22 = h11;
            } else {
                h11 = h[ix(i - 1, i - 1, ldh)];
                h21 = h[ix(i, i - 1, ldh)];
                h12 = h[ix(i - 1, i, ldh)];
                h22 = h[ix(i, i, ldh)];
            }
            let s = h11.abs() + h12.abs() + h21.abs() + h22.abs();
            let (mut rt1r, rt1i, mut rt2r, rt2i);
            if s == 0.0 {
                rt1r = 0.0;
                rt1i = 0.0;
                rt2r = 0.0;
                rt2i = 0.0;
            } else {
                h11 /= s;
                h21 /= s;
                h12 /= s;
                h22 /= s;
                let tr = (h11 + h22) / 2.0;
                let det = (h11 - tr) * (h22 - tr) - h12 * h21;
                let rtdisc = det.abs().sqrt();
                if det >= 0.0 {
                    rt1r = tr * s;
                    rt2r = rt1r;
                    rt1i = rtdisc * s;
                    rt2i = -rt1i;
                } else {
                    rt1r = tr + rtdisc;
                    rt2r = tr - rtdisc;
                    if (rt1r - h22).abs() <= (rt2r - h22).abs() {
                        rt1r *= s;
                        rt2r = rt1r;
                    } else {
                        rt2r *= s;
                        rt1r = rt2r;
                    }
                    rt1i = 0.0;
                    rt2i = 0.0;
                }
            }
            let mut m = i - 2;
            loop {
                let mut h21s = h[ix(m + 1, m, ldh)];
                let sv = (h[ix(m, m, ldh)] - rt2r).abs()
                    + rt2i.abs()
                    + h21s.abs();
                h21s = h[ix(m + 1, m, ldh)] / sv;
                v[0] = h21s * h[ix(m, m + 1, ldh)]
                    + (h[ix(m, m, ldh)] - rt1r) * ((h[ix(m, m, ldh)] - rt2r) / sv)
                    - rt1i * (rt2i / sv);
                v[1] = h21s
                    * (h[ix(m, m, ldh)] + h[ix(m + 1, m + 1, ldh)] - rt1r - rt2r);
                v[2] = h21s * h[ix(m + 2, m + 1, ldh)];
                let sv2 = v[0].abs() + v[1].abs() + v[2].abs();
                v[0] /= sv2;
                v[1] /= sv2;
                v[2] /= sv2;
                if m == l {
                    break;
                }
                let lhs = h[ix(m, m - 1, ldh)].abs() * (v[1].abs() + v[2].abs());
                let rhs = ulp
                    * v[0].abs()
                    * (h[ix(m - 1, m - 1, ldh)].abs()
                        + h[ix(m, m, ldh)].abs()
                        + h[ix(m + 1, m + 1, ldh)].abs());
                if lhs <= rhs {
                    break;
                }
                m -= 1;
            }

            for k in m..i {
                let nr = 3.min(i - k + 1);
                if k > m {
                    dcopy(nr, &h[ix(k, k - 1, ldh)..], 1, &mut v, 1);
                }
                let (mut v0, mut vrest) = (v[0], [v[1], v[2]]);
                let t1 = dlarfg(nr, &mut v0, &mut vrest, 1);
                v = [v0, vrest[0], vrest[1]];
                if k > m {
                    h[ix(k, k - 1, ldh)] = v[0];
                    h[ix(k + 1, k - 1, ldh)] = 0.0;
                    if k + 1 < i {
                        h[ix(k + 2, k - 1, ldh)] = 0.0;
                    }
                } else if m > l {
                    h[ix(k, k - 1, ldh)] *= 1.0 - t1;
                }
                let v2 = v[1];
                let t2 = t1 * v2;
                if nr == 3 {
                    let v3 = v[2];
                    let t3 = t1 * v3;
                    for j in k..=i2 {
                        let sum = h[ix(k, j, ldh)]
                            + v2 * h[ix(k + 1, j, ldh)]
                            + v3 * h[ix(k + 2, j, ldh)];
                        h[ix(k, j, ldh)] -= sum * t1;
                        h[ix(k + 1, j, ldh)] -= sum * t2;
                        h[ix(k + 2, j, ldh)] -= sum * t3;
                    }
                    for j in i1..=i.min(k + 3) {
                        let sum = h[ix(j, k, ldh)]
                            + v2 * h[ix(j, k + 1, ldh)]
                            + v3 * h[ix(j, k + 2, ldh)];
                        h[ix(j, k, ldh)] -= sum * t1;
                        h[ix(j, k + 1, ldh)] -= sum * t2;
                        h[ix(j, k + 2, ldh)] -= sum * t3;
                    }
                    if wantz {
                        for j in iloz..=ihiz {
                            let sum = z[ix(j, k, ldz)]
                                + v2 * z[ix(j, k + 1, ldz)]
                                + v3 * z[ix(j, k + 2, ldz)];
                            z[ix(j, k, ldz)] -= sum * t1;
                            z[ix(j, k + 1, ldz)] -= sum * t2;
                            z[ix(j, k + 2, ldz)] -= sum * t3;
                        }
                    }
                } else if nr == 2 {
                    for j in k..=i2 {
                        let sum = h[ix(k, j, ldh)] + v2 * h[ix(k + 1, j, ldh)];
                        h[ix(k, j, ldh)] -= sum * t1;
                        h[ix(k + 1, j, ldh)] -= sum * t2;
                    }
                    for j in i1..=i {
                        let sum = h[ix(j, k, ldh)] + v2 * h[ix(j, k + 1, ldh)];
                        h[ix(j, k, ldh)] -= sum * t1;
                        h[ix(j, k + 1, ldh)] -= sum * t2;
                    }
                    if wantz {
                        for j in iloz..=ihiz {
                            let sum =
                                z[ix(j, k, ldz)] + v2 * z[ix(j, k + 1, ldz)];
                            z[ix(j, k, ldz)] -= sum * t1;
                            z[ix(j, k + 1, ldz)] -= sum * t2;
                        }
                    }
                }
            }
        }
        if !split {
            return i;
        }
        if l == i {
            wr[i - 1] = h[ix(i, i, ldh)];
            wi[i - 1] = 0.0;
        } else if l + 1 == i {
            let res = dlanv2(
                h[ix(i - 1, i - 1, ldh)],
                h[ix(i - 1, i, ldh)],
                h[ix(i, i - 1, ldh)],
                h[ix(i, i, ldh)],
            );
            h[ix(i - 1, i - 1, ldh)] = res.a;
            h[ix(i - 1, i, ldh)] = res.b;
            h[ix(i, i - 1, ldh)] = res.c;
            h[ix(i, i, ldh)] = res.d;
            wr[i - 2] = res.rt1r;
            wi[i - 2] = res.rt1i;
            wr[i - 1] = res.rt2r;
            wi[i - 1] = res.rt2i;
            if wantt {
                if i2 > i {
                    drot_in(
                        i2 - i,
                        h,
                        ix(i - 1, i + 1, ldh),
                        ldh,
                        ix(i, i + 1, ldh),
                        ldh,
                        res.cs,
                        res.sn,
                    );
                }
                drot_in(
                    i - i1 - 1,
                    h,
                    ix(i1, i - 1, ldh),
                    1,
                    ix(i1, i, ldh),
                    1,
                    res.cs,
                    res.sn,
                );
            }
            if wantz {
                drot_in(
                    nz,
                    z,
                    ix(iloz, i - 1, ldz),
                    1,
                    ix(iloz, i, ldz),
                    1,
                    res.cs,
                    res.sn,
                );
            }
        }
        kdefl = 0;
        if l == 1 {
            break;
        }
        i = l - 1;
    }
    0
}

/// `DHSEQR` (`SRC/dhseqr.f`): compute eigenvalues and optionally Schur
/// form and Schur vectors (`n <= 75` uses `DLAHQR`). Returns `(work1, info)`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] or [`LapackError::UnportedPath`] for
/// `n > 75` (`DLAQR0`).
#[allow(clippy::cast_precision_loss)]
pub fn dhseqr(
    job: u8,
    compz: u8,
    n: usize,
    ilo: usize,
    ihi: usize,
    h: &mut [f64],
    ldh: usize,
    wr: &mut [f64],
    wi: &mut [f64],
    z: &mut [f64],
    ldz: usize,
    lwork: isize,
) -> Result<(f64, usize), LapackError> {
    let wantt = lsame(job, b'S');
    let initz = lsame(compz, b'I');
    let wantz = initz || lsame(compz, b'V');
    let lquery = lwork == -1;
    if !wantt && !lsame(job, b'E') {
        return Err(bad("DHSEQR", -1));
    }
    if !wantz && !lsame(compz, b'N') {
        return Err(bad("DHSEQR", -2));
    }
    if ilo < 1 || ilo > n.max(1) {
        return Err(bad("DHSEQR", -4));
    }
    if ihi < ilo.min(n) || ihi > n {
        return Err(bad("DHSEQR", -5));
    }
    if ldh < n.max(1) {
        return Err(bad("DHSEQR", -7));
    }
    if ldz < 1 || (wantz && ldz < n) {
        return Err(bad("DHSEQR", -11));
    }
    if !lquery && lwork < (n.max(1) as isize) {
        return Err(bad("DHSEQR", -13));
    }
    if n > NMIN75 {
        return Err(unported("DHSEQR", "456-498 (DLAQR0)"));
    }
    let work1 = (n.max(1)) as f64;
    if lquery || n == 0 {
        return Ok((work1, 0));
    }
    for i in 1..ilo {
        wr[i - 1] = h[ix(i, i, ldh)];
        wi[i - 1] = 0.0;
    }
    for i in ihi + 1..=n {
        wr[i - 1] = h[ix(i, i, ldh)];
        wi[i - 1] = 0.0;
    }
    if initz {
        dlaset(b'A', n, n, 0.0, 1.0, z, ldz);
    }
    if ilo == ihi {
        wr[ilo - 1] = h[ix(ilo, ilo, ldh)];
        wi[ilo - 1] = 0.0;
        return Ok((work1, 0));
    }
    let info =
        dlahqr(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, ilo, ihi, z, ldz);
    if info > 0 {
        return Err(unported("DHSEQR", "466-497 (DLAQR0 fallback)"));
    }
    if wantt && n > 2 {
        dlaset(b'L', n - 2, n - 2, 0.0, 0.0, &mut h[ix(3, 1, ldh)..], ldh);
    }
    Ok((work1, info))
}

/// `DTREVC3` (`SRC/dtrevc3.f`): compute eigenvectors of an upper
/// quasi-triangular matrix `T` in real Schur form. Returns `(nout, work1)`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] on invalid options or sizes.
#[allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap
)]
pub fn dtrevc3(
    side: u8,
    howmny: u8,
    select: &mut [bool],
    n: usize,
    t: &[f64],
    ldt: usize,
    vl: &mut [f64],
    ldvl: usize,
    vr: &mut [f64],
    ldvr: usize,
    mm: usize,
    lwork: isize,
) -> Result<(usize, f64), LapackError> {
    const NBMIN: usize = 8;
    let bothv = lsame(side, b'B');
    let rightv = lsame(side, b'R') || bothv;
    let leftv = lsame(side, b'L') || bothv;
    let allv = lsame(howmny, b'A');
    let over = lsame(howmny, b'B');
    let somev = lsame(howmny, b'S');
    if !rightv && !leftv {
        return Err(bad("DTREVC3", -1));
    }
    if !allv && !over && !somev {
        return Err(bad("DTREVC3", -2));
    }
    if ldt < n.max(1) {
        return Err(bad("DTREVC3", -6));
    }
    if ldvl < 1 || (leftv && ldvl < n) {
        return Err(bad("DTREVC3", -8));
    }
    if ldvr < 1 || (rightv && ldvr < n) {
        return Err(bad("DTREVC3", -10));
    }
    let lquery = lwork == -1;
    if !lquery && lwork < ((3 * n).max(1) as isize) {
        return Err(bad("DTREVC3", -14));
    }
    let m = if somev {
        let mut cnt = 0_usize;
        let mut pair = false;
        for j in 1..=n {
            if pair {
                pair = false;
                select[j - 1] = false;
            } else if j < n {
                if t[ix(j + 1, j, ldt)] == 0.0 {
                    if select[j - 1] {
                        cnt += 1;
                    }
                } else {
                    pair = true;
                    if select[j - 1] || select[j] {
                        select[j - 1] = true;
                        cnt += 2;
                    }
                }
            } else if select[n - 1] {
                cnt += 1;
            }
        }
        cnt
    } else {
        n
    };
    if mm < m {
        return Err(bad("DTREVC3", -11));
    }
    let maxwrk = (1.max(n + 2 * n * NB64)) as f64;
    if lquery || n == 0 {
        return Ok((m, maxwrk));
    }

    let lwork_u = lwork.cast_unsigned();
    let nb = if over && lwork_u >= n + 2 * n * NBMIN {
        ((lwork_u - n) / (2 * n)).min(128)
    } else {
        1
    };
    let mut work = vec![0.0_f64; n * (1 + 2 * nb.max(2))];
    let mut iscomplex = vec![0_i32; nb.max(2) + 1];

    let unfl = dlamch::SFMIN;
    let ulp = dlamch::PREC;
    let smlnum = unfl * ((n as f64) / ulp);
    let bignum = (1.0 - ulp) / smlnum;

    work[0] = 0.0;
    for j in 2..=n {
        let mut acc = 0.0;
        for i in 1..j {
            acc += t[ix(i, j, ldt)].abs();
        }
        work[j - 1] = acc;
    }

    let mut x = [0.0_f64; 4];

    if rightv {
        let mut iv = if nb > 2 { nb } else { 2 };
        let mut ip = 0_i32;
        let mut is_col = m;
        for ki in (1..=n).rev() {
            if ip == -1 {
                ip = 1;
                continue;
            } else if ki == 1 || t[ix(ki, ki - 1, ldt)] == 0.0 {
                ip = 0;
            } else {
                ip = -1;
            }
            if somev {
                let idx_sel = if ip == 0 { ki } else { ki - 1 };
                if !select[idx_sel - 1] {
                    continue;
                }
            }
            let wr = t[ix(ki, ki, ldt)];
            let wi = if ip != 0 {
                t[ix(ki, ki - 1, ldt)].abs().sqrt()
                    * t[ix(ki - 1, ki, ldt)].abs().sqrt()
            } else {
                0.0
            };
            let smin = fmax(ulp * (wr.abs() + wi.abs()), smlnum);

            if ip == 0 {
                work[ki - 1 + iv * n] = 1.0;
                for k in 1..ki {
                    work[k - 1 + iv * n] = -t[ix(k, ki, ldt)];
                }
                let mut jnxt = ki - 1;
                for j in (1..ki).rev() {
                    if j > jnxt {
                        continue;
                    }
                    let mut j1 = j;
                    let j2 = j;
                    jnxt = j - 1;
                    if j > 1 && t[ix(j, j - 1, ldt)] != 0.0 {
                        j1 = j - 1;
                        jnxt = j - 2;
                    }
                    if j1 == j2 {
                        let (mut scale, xnorm, _) = dlaln2(
                            false,
                            1,
                            1,
                            smin,
                            1.0,
                            &t[ix(j, j, ldt)..],
                            ldt,
                            1.0,
                            1.0,
                            &work[j - 1 + iv * n..],
                            n,
                            wr,
                            0.0,
                            &mut x,
                            2,
                        );
                        if xnorm > 1.0 && work[j - 1] > bignum / xnorm {
                            x[0] /= xnorm;
                            scale /= xnorm;
                        }
                        if scale != 1.0 {
                            dscal(ki, scale, &mut work[iv * n..], 1);
                        }
                        work[j - 1 + iv * n] = x[0];
                        daxpy(
                            j - 1,
                            -x[0],
                            &t[ix(1, j, ldt)..],
                            1,
                            &mut work[iv * n..],
                            1,
                        );
                    } else {
                        let (mut scale, xnorm, _) = dlaln2(
                            false,
                            2,
                            1,
                            smin,
                            1.0,
                            &t[ix(j - 1, j - 1, ldt)..],
                            ldt,
                            1.0,
                            1.0,
                            &work[j - 2 + iv * n..],
                            n,
                            wr,
                            0.0,
                            &mut x,
                            2,
                        );
                        if xnorm > 1.0 {
                            let beta = fmax(work[j - 2], work[j - 1]);
                            if beta > bignum / xnorm {
                                x[0] /= xnorm;
                                x[1] /= xnorm;
                                scale /= xnorm;
                            }
                        }
                        if scale != 1.0 {
                            dscal(ki, scale, &mut work[iv * n..], 1);
                        }
                        work[j - 2 + iv * n] = x[0];
                        work[j - 1 + iv * n] = x[1];
                        daxpy(
                            j - 2,
                            -x[0],
                            &t[ix(1, j - 1, ldt)..],
                            1,
                            &mut work[iv * n..],
                            1,
                        );
                        daxpy(
                            j - 2,
                            -x[1],
                            &t[ix(1, j, ldt)..],
                            1,
                            &mut work[iv * n..],
                            1,
                        );
                    }
                }
                if !over {
                    dcopy(ki, &work[iv * n..], 1, &mut vr[ix(1, is_col, ldvr)..], 1);
                    let ii = idamax(ki, &vr[ix(1, is_col, ldvr)..], 1);
                    let remax = 1.0 / vr[ix(ii, is_col, ldvr)].abs();
                    dscal(ki, remax, &mut vr[ix(1, is_col, ldvr)..], 1);
                    for k in ki + 1..=n {
                        vr[ix(k, is_col, ldvr)] = 0.0;
                    }
                } else if nb == 1 {
                    if ki > 1 {
                        let c_off = ix(1, ki, ldvr);
                        let (vr_head, vr_col) = vr.split_at_mut(c_off);
                        dgemv(
                            b'N',
                            n,
                            ki - 1,
                            1.0,
                            vr_head,
                            ldvr,
                            &work[iv * n..],
                            1,
                            work[ki - 1 + iv * n],
                            vr_col,
                            1,
                        )?;
                    }
                    let ii = idamax(n, &vr[ix(1, ki, ldvr)..], 1);
                    let remax = 1.0 / vr[ix(ii, ki, ldvr)].abs();
                    dscal(n, remax, &mut vr[ix(1, ki, ldvr)..], 1);
                } else {
                    for k in ki + 1..=n {
                        work[k - 1 + iv * n] = 0.0;
                    }
                    iscomplex[iv] = ip;
                }
            } else {
                if t[ix(ki - 1, ki, ldt)].abs() >= t[ix(ki, ki - 1, ldt)].abs() {
                    work[ki - 2 + (iv - 1) * n] = 1.0;
                    work[ki - 1 + iv * n] = wi / t[ix(ki - 1, ki, ldt)];
                } else {
                    work[ki - 2 + (iv - 1) * n] = -wi / t[ix(ki, ki - 1, ldt)];
                    work[ki - 1 + iv * n] = 1.0;
                }
                work[ki - 1 + (iv - 1) * n] = 0.0;
                work[ki - 2 + iv * n] = 0.0;
                for k in 1..ki - 1 {
                    work[k - 1 + (iv - 1) * n] =
                        -work[ki - 2 + (iv - 1) * n] * t[ix(k, ki - 1, ldt)];
                    work[k - 1 + iv * n] =
                        -work[ki - 1 + iv * n] * t[ix(k, ki, ldt)];
                }
                let mut jnxt = ki - 2;
                for j in (1..ki - 1).rev() {
                    if j > jnxt {
                        continue;
                    }
                    let mut j1 = j;
                    let j2 = j;
                    jnxt = j - 1;
                    if j > 1 && t[ix(j, j - 1, ldt)] != 0.0 {
                        j1 = j - 1;
                        jnxt = j - 2;
                    }
                    if j1 == j2 {
                        let (mut scale, xnorm, _) = dlaln2(
                            false,
                            1,
                            2,
                            smin,
                            1.0,
                            &t[ix(j, j, ldt)..],
                            ldt,
                            1.0,
                            1.0,
                            &work[j - 1 + (iv - 1) * n..],
                            n,
                            wr,
                            wi,
                            &mut x,
                            2,
                        );
                        if xnorm > 1.0 && work[j - 1] > bignum / xnorm {
                            x[0] /= xnorm;
                            x[2] /= xnorm;
                            scale /= xnorm;
                        }
                        if scale != 1.0 {
                            dscal(ki, scale, &mut work[(iv - 1) * n..], 1);
                            dscal(ki, scale, &mut work[iv * n..], 1);
                        }
                        work[j - 1 + (iv - 1) * n] = x[0];
                        work[j - 1 + iv * n] = x[2];
                        daxpy(
                            j - 1,
                            -x[0],
                            &t[ix(1, j, ldt)..],
                            1,
                            &mut work[(iv - 1) * n..],
                            1,
                        );
                        daxpy(
                            j - 1,
                            -x[2],
                            &t[ix(1, j, ldt)..],
                            1,
                            &mut work[iv * n..],
                            1,
                        );
                    } else {
                        let (mut scale, xnorm, _) = dlaln2(
                            false,
                            2,
                            2,
                            smin,
                            1.0,
                            &t[ix(j - 1, j - 1, ldt)..],
                            ldt,
                            1.0,
                            1.0,
                            &work[j - 2 + (iv - 1) * n..],
                            n,
                            wr,
                            wi,
                            &mut x,
                            2,
                        );
                        if xnorm > 1.0 {
                            let beta = fmax(work[j - 2], work[j - 1]);
                            if beta > bignum / xnorm {
                                let rec = 1.0 / xnorm;
                                x[0] *= rec;
                                x[2] *= rec;
                                x[1] *= rec;
                                x[3] *= rec;
                                scale *= rec;
                            }
                        }
                        if scale != 1.0 {
                            dscal(ki, scale, &mut work[(iv - 1) * n..], 1);
                            dscal(ki, scale, &mut work[iv * n..], 1);
                        }
                        work[j - 2 + (iv - 1) * n] = x[0];
                        work[j - 1 + (iv - 1) * n] = x[1];
                        work[j - 2 + iv * n] = x[2];
                        work[j - 1 + iv * n] = x[3];
                        daxpy(
                            j - 2,
                            -x[0],
                            &t[ix(1, j - 1, ldt)..],
                            1,
                            &mut work[(iv - 1) * n..],
                            1,
                        );
                        daxpy(
                            j - 2,
                            -x[1],
                            &t[ix(1, j, ldt)..],
                            1,
                            &mut work[(iv - 1) * n..],
                            1,
                        );
                        daxpy(
                            j - 2,
                            -x[2],
                            &t[ix(1, j - 1, ldt)..],
                            1,
                            &mut work[iv * n..],
                            1,
                        );
                        daxpy(
                            j - 2,
                            -x[3],
                            &t[ix(1, j, ldt)..],
                            1,
                            &mut work[iv * n..],
                            1,
                        );
                    }
                }
                if !over {
                    dcopy(
                        ki,
                        &work[(iv - 1) * n..],
                        1,
                        &mut vr[ix(1, is_col - 1, ldvr)..],
                        1,
                    );
                    dcopy(ki, &work[iv * n..], 1, &mut vr[ix(1, is_col, ldvr)..], 1);
                    let mut emax = 0.0_f64;
                    for k in 1..=ki {
                        emax = fmax(
                            emax,
                            vr[ix(k, is_col - 1, ldvr)].abs()
                                + vr[ix(k, is_col, ldvr)].abs(),
                        );
                    }
                    let remax = 1.0 / emax;
                    dscal(ki, remax, &mut vr[ix(1, is_col - 1, ldvr)..], 1);
                    dscal(ki, remax, &mut vr[ix(1, is_col, ldvr)..], 1);
                    for k in ki + 1..=n {
                        vr[ix(k, is_col - 1, ldvr)] = 0.0;
                        vr[ix(k, is_col, ldvr)] = 0.0;
                    }
                } else if nb == 1 {
                    if ki > 2 {
                        let c1_off = ix(1, ki - 1, ldvr);
                        let (vr_head, vr_tail) = vr.split_at_mut(c1_off);
                        let (vr_km1, vr_k) = vr_tail.split_at_mut(ldvr);
                        dgemv(
                            b'N',
                            n,
                            ki - 2,
                            1.0,
                            vr_head,
                            ldvr,
                            &work[(iv - 1) * n..],
                            1,
                            work[ki - 2 + (iv - 1) * n],
                            vr_km1,
                            1,
                        )?;
                        dgemv(
                            b'N',
                            n,
                            ki - 2,
                            1.0,
                            vr_head,
                            ldvr,
                            &work[iv * n..],
                            1,
                            work[ki - 1 + iv * n],
                            vr_k,
                            1,
                        )?;
                    } else {
                        dscal(
                            n,
                            work[ki - 2 + (iv - 1) * n],
                            &mut vr[ix(1, ki - 1, ldvr)..],
                            1,
                        );
                        dscal(
                            n,
                            work[ki - 1 + iv * n],
                            &mut vr[ix(1, ki, ldvr)..],
                            1,
                        );
                    }
                    let mut emax = 0.0_f64;
                    for k in 1..=n {
                        emax = fmax(
                            emax,
                            vr[ix(k, ki - 1, ldvr)].abs()
                                + vr[ix(k, ki, ldvr)].abs(),
                        );
                    }
                    let remax = 1.0 / emax;
                    dscal(n, remax, &mut vr[ix(1, ki - 1, ldvr)..], 1);
                    dscal(n, remax, &mut vr[ix(1, ki, ldvr)..], 1);
                } else {
                    for k in ki + 1..=n {
                        work[k - 1 + (iv - 1) * n] = 0.0;
                        work[k - 1 + iv * n] = 0.0;
                    }
                    iscomplex[iv - 1] = -ip;
                    iscomplex[iv] = ip;
                    iv -= 1;
                }
            }
            if nb > 1 {
                let ki2 = if ip == 0 { ki } else { ki - 1 };
                if iv <= 2 || ki2 == 1 {
                    let (w_in, w_out) = work.split_at_mut((nb + iv) * n);
                    dgemm(
                        b'N',
                        b'N',
                        n,
                        nb - iv + 1,
                        ki2 + nb - iv,
                        1.0,
                        vr,
                        ldvr,
                        &w_in[iv * n..],
                        n,
                        0.0,
                        w_out,
                        n,
                    )?;
                    let mut remax = 1.0_f64;
                    for k in iv..=nb {
                        if iscomplex[k] == 0 {
                            let ii = idamax(n, &work[(nb + k) * n..], 1);
                            remax = 1.0 / work[ii - 1 + (nb + k) * n].abs();
                        } else if iscomplex[k] == 1 {
                            let mut emax = 0.0_f64;
                            for ii in 1..=n {
                                emax = fmax(
                                    emax,
                                    work[ii - 1 + (nb + k) * n].abs()
                                        + work[ii - 1 + (nb + k + 1) * n].abs(),
                                );
                            }
                            remax = 1.0 / emax;
                        }
                        dscal(n, remax, &mut work[(nb + k) * n..], 1);
                    }
                    dlacpy(
                        b'F',
                        n,
                        nb - iv + 1,
                        &work[(nb + iv) * n..],
                        n,
                        &mut vr[ix(1, ki2, ldvr)..],
                        ldvr,
                    );
                    iv = nb;
                } else {
                    iv -= 1;
                }
            }
            is_col = is_col.saturating_sub(1);
            if ip != 0 {
                is_col = is_col.saturating_sub(1);
            }
        }
    }

    if leftv {
        let mut iv = 1_usize;
        let mut ip = 0_i32;
        let mut is_col = 1_usize;
        for ki in 1..=n {
            if ip == 1 {
                ip = -1;
                continue;
            } else if ki == n || t[ix(ki + 1, ki, ldt)] == 0.0 {
                ip = 0;
            } else {
                ip = 1;
            }
            if somev && !select[ki - 1] {
                continue;
            }
            let wr = t[ix(ki, ki, ldt)];
            let wi = if ip != 0 {
                t[ix(ki, ki + 1, ldt)].abs().sqrt()
                    * t[ix(ki + 1, ki, ldt)].abs().sqrt()
            } else {
                0.0
            };
            let smin = fmax(ulp * (wr.abs() + wi.abs()), smlnum);

            if ip == 0 {
                work[ki - 1 + iv * n] = 1.0;
                for k in ki + 1..=n {
                    work[k - 1 + iv * n] = -t[ix(ki, k, ldt)];
                }
                let mut vmax = 1.0_f64;
                let mut vcrit = bignum;
                let mut jnxt = ki + 1;
                for j in ki + 1..=n {
                    if j < jnxt {
                        continue;
                    }
                    let j1 = j;
                    let mut j2 = j;
                    jnxt = j + 1;
                    if j < n && t[ix(j + 1, j, ldt)] != 0.0 {
                        j2 = j + 1;
                        jnxt = j + 2;
                    }
                    if j1 == j2 {
                        if work[j - 1] > vcrit {
                            let rec = 1.0 / vmax;
                            dscal(
                                n - ki + 1,
                                rec,
                                &mut work[ki - 1 + iv * n..],
                                1,
                            );
                            vmax = 1.0;
                        }
                        work[j - 1 + iv * n] -= ddot(
                            j - ki - 1,
                            &t[ix(ki + 1, j, ldt)..],
                            1,
                            &work[ki + iv * n..],
                            1,
                        );
                        let (scale, _, _) = dlaln2(
                            false,
                            1,
                            1,
                            smin,
                            1.0,
                            &t[ix(j, j, ldt)..],
                            ldt,
                            1.0,
                            1.0,
                            &work[j - 1 + iv * n..],
                            n,
                            wr,
                            0.0,
                            &mut x,
                            2,
                        );
                        if scale != 1.0 {
                            dscal(
                                n - ki + 1,
                                scale,
                                &mut work[ki - 1 + iv * n..],
                                1,
                            );
                        }
                        work[j - 1 + iv * n] = x[0];
                        vmax = fmax(work[j - 1 + iv * n].abs(), vmax);
                        vcrit = bignum / vmax;
                    } else {
                        let beta = fmax(work[j - 1], work[j]);
                        if beta > vcrit {
                            let rec = 1.0 / vmax;
                            dscal(
                                n - ki + 1,
                                rec,
                                &mut work[ki - 1 + iv * n..],
                                1,
                            );
                            vmax = 1.0;
                        }
                        work[j - 1 + iv * n] -= ddot(
                            j - ki - 1,
                            &t[ix(ki + 1, j, ldt)..],
                            1,
                            &work[ki + iv * n..],
                            1,
                        );
                        work[j + iv * n] -= ddot(
                            j - ki - 1,
                            &t[ix(ki + 1, j + 1, ldt)..],
                            1,
                            &work[ki + iv * n..],
                            1,
                        );
                        let (scale, _, _) = dlaln2(
                            true,
                            2,
                            1,
                            smin,
                            1.0,
                            &t[ix(j, j, ldt)..],
                            ldt,
                            1.0,
                            1.0,
                            &work[j - 1 + iv * n..],
                            n,
                            wr,
                            0.0,
                            &mut x,
                            2,
                        );
                        if scale != 1.0 {
                            dscal(
                                n - ki + 1,
                                scale,
                                &mut work[ki - 1 + iv * n..],
                                1,
                            );
                        }
                        work[j - 1 + iv * n] = x[0];
                        work[j + iv * n] = x[1];
                        vmax = fmax(
                            fmax(
                                work[j - 1 + iv * n].abs(),
                                work[j + iv * n].abs(),
                            ),
                            vmax,
                        );
                        vcrit = bignum / vmax;
                    }
                }
                if !over {
                    dcopy(
                        n - ki + 1,
                        &work[ki - 1 + iv * n..],
                        1,
                        &mut vl[ix(ki, is_col, ldvl)..],
                        1,
                    );
                    let ii =
                        idamax(n - ki + 1, &vl[ix(ki, is_col, ldvl)..], 1)
                            + ki
                            - 1;
                    let remax = 1.0 / vl[ix(ii, is_col, ldvl)].abs();
                    dscal(n - ki + 1, remax, &mut vl[ix(ki, is_col, ldvl)..], 1);
                    for k in 1..ki {
                        vl[ix(k, is_col, ldvl)] = 0.0;
                    }
                } else if nb == 1 {
                    if ki < n {
                        let c_off = ix(1, ki, ldvl);
                        let (vl_col, vl_tail) = vl[c_off..].split_at_mut(ldvl);
                        dgemv(
                            b'N',
                            n,
                            n - ki,
                            1.0,
                            vl_tail,
                            ldvl,
                            &work[ki + iv * n..],
                            1,
                            work[ki - 1 + iv * n],
                            vl_col,
                            1,
                        )?;
                    }
                    let ii = idamax(n, &vl[ix(1, ki, ldvl)..], 1);
                    let remax = 1.0 / vl[ix(ii, ki, ldvl)].abs();
                    dscal(n, remax, &mut vl[ix(1, ki, ldvl)..], 1);
                } else {
                    for k in 1..ki {
                        work[k - 1 + iv * n] = 0.0;
                    }
                    iscomplex[iv] = ip;
                }
            } else {
                if t[ix(ki, ki + 1, ldt)].abs() >= t[ix(ki + 1, ki, ldt)].abs() {
                    work[ki - 1 + iv * n] = wi / t[ix(ki, ki + 1, ldt)];
                    work[ki + (iv + 1) * n] = 1.0;
                } else {
                    work[ki - 1 + iv * n] = 1.0;
                    work[ki + (iv + 1) * n] = -wi / t[ix(ki + 1, ki, ldt)];
                }
                work[ki + iv * n] = 0.0;
                work[ki - 1 + (iv + 1) * n] = 0.0;
                for k in ki + 2..=n {
                    work[k - 1 + iv * n] =
                        -work[ki - 1 + iv * n] * t[ix(ki, k, ldt)];
                    work[k - 1 + (iv + 1) * n] =
                        -work[ki + (iv + 1) * n] * t[ix(ki + 1, k, ldt)];
                }
                let mut vmax = 1.0_f64;
                let mut vcrit = bignum;
                let mut jnxt = ki + 2;
                for j in ki + 2..=n {
                    if j < jnxt {
                        continue;
                    }
                    let j1 = j;
                    let mut j2 = j;
                    jnxt = j + 1;
                    if j < n && t[ix(j + 1, j, ldt)] != 0.0 {
                        j2 = j + 1;
                        jnxt = j + 2;
                    }
                    if j1 == j2 {
                        if work[j - 1] > vcrit {
                            let rec = 1.0 / vmax;
                            dscal(
                                n - ki + 1,
                                rec,
                                &mut work[ki - 1 + iv * n..],
                                1,
                            );
                            dscal(
                                n - ki + 1,
                                rec,
                                &mut work[ki - 1 + (iv + 1) * n..],
                                1,
                            );
                            vmax = 1.0;
                        }
                        work[j - 1 + iv * n] -= ddot(
                            j - ki - 2,
                            &t[ix(ki + 2, j, ldt)..],
                            1,
                            &work[ki + 1 + iv * n..],
                            1,
                        );
                        work[j - 1 + (iv + 1) * n] -= ddot(
                            j - ki - 2,
                            &t[ix(ki + 2, j, ldt)..],
                            1,
                            &work[ki + 1 + (iv + 1) * n..],
                            1,
                        );
                        let (scale, _, _) = dlaln2(
                            false,
                            1,
                            2,
                            smin,
                            1.0,
                            &t[ix(j, j, ldt)..],
                            ldt,
                            1.0,
                            1.0,
                            &work[j - 1 + iv * n..],
                            n,
                            wr,
                            -wi,
                            &mut x,
                            2,
                        );
                        if scale != 1.0 {
                            dscal(
                                n - ki + 1,
                                scale,
                                &mut work[ki - 1 + iv * n..],
                                1,
                            );
                            dscal(
                                n - ki + 1,
                                scale,
                                &mut work[ki - 1 + (iv + 1) * n..],
                                1,
                            );
                        }
                        work[j - 1 + iv * n] = x[0];
                        work[j - 1 + (iv + 1) * n] = x[2];
                        vmax = fmax(
                            fmax(
                                work[j - 1 + iv * n].abs(),
                                work[j - 1 + (iv + 1) * n].abs(),
                            ),
                            vmax,
                        );
                        vcrit = bignum / vmax;
                    } else {
                        let beta = fmax(work[j - 1], work[j]);
                        if beta > vcrit {
                            let rec = 1.0 / vmax;
                            dscal(
                                n - ki + 1,
                                rec,
                                &mut work[ki - 1 + iv * n..],
                                1,
                            );
                            dscal(
                                n - ki + 1,
                                rec,
                                &mut work[ki - 1 + (iv + 1) * n..],
                                1,
                            );
                            vmax = 1.0;
                        }
                        work[j - 1 + iv * n] -= ddot(
                            j - ki - 2,
                            &t[ix(ki + 2, j, ldt)..],
                            1,
                            &work[ki + 1 + iv * n..],
                            1,
                        );
                        work[j - 1 + (iv + 1) * n] -= ddot(
                            j - ki - 2,
                            &t[ix(ki + 2, j, ldt)..],
                            1,
                            &work[ki + 1 + (iv + 1) * n..],
                            1,
                        );
                        work[j + iv * n] -= ddot(
                            j - ki - 2,
                            &t[ix(ki + 2, j + 1, ldt)..],
                            1,
                            &work[ki + 1 + iv * n..],
                            1,
                        );
                        work[j + (iv + 1) * n] -= ddot(
                            j - ki - 2,
                            &t[ix(ki + 2, j + 1, ldt)..],
                            1,
                            &work[ki + 1 + (iv + 1) * n..],
                            1,
                        );
                        let (scale, _, _) = dlaln2(
                            true,
                            2,
                            2,
                            smin,
                            1.0,
                            &t[ix(j, j, ldt)..],
                            ldt,
                            1.0,
                            1.0,
                            &work[j - 1 + iv * n..],
                            n,
                            wr,
                            -wi,
                            &mut x,
                            2,
                        );
                        if scale != 1.0 {
                            dscal(
                                n - ki + 1,
                                scale,
                                &mut work[ki - 1 + iv * n..],
                                1,
                            );
                            dscal(
                                n - ki + 1,
                                scale,
                                &mut work[ki - 1 + (iv + 1) * n..],
                                1,
                            );
                        }
                        work[j - 1 + iv * n] = x[0];
                        work[j - 1 + (iv + 1) * n] = x[2];
                        work[j + iv * n] = x[1];
                        work[j + (iv + 1) * n] = x[3];
                        vmax = fmax(
                            fmax(fmax(x[0].abs(), x[2].abs()), fmax(x[1].abs(), x[3].abs())),
                            vmax,
                        );
                        vcrit = bignum / vmax;
                    }
                }
                if !over {
                    dcopy(
                        n - ki + 1,
                        &work[ki - 1 + iv * n..],
                        1,
                        &mut vl[ix(ki, is_col, ldvl)..],
                        1,
                    );
                    dcopy(
                        n - ki + 1,
                        &work[ki - 1 + (iv + 1) * n..],
                        1,
                        &mut vl[ix(ki, is_col + 1, ldvl)..],
                        1,
                    );
                    let mut emax = 0.0_f64;
                    for k in ki..=n {
                        emax = fmax(
                            emax,
                            vl[ix(k, is_col, ldvl)].abs()
                                + vl[ix(k, is_col + 1, ldvl)].abs(),
                        );
                    }
                    let remax = 1.0 / emax;
                    dscal(n - ki + 1, remax, &mut vl[ix(ki, is_col, ldvl)..], 1);
                    dscal(
                        n - ki + 1,
                        remax,
                        &mut vl[ix(ki, is_col + 1, ldvl)..],
                        1,
                    );
                    for k in 1..ki {
                        vl[ix(k, is_col, ldvl)] = 0.0;
                        vl[ix(k, is_col + 1, ldvl)] = 0.0;
                    }
                } else if nb == 1 {
                    if ki + 1 < n {
                        let c1_off = ix(1, ki, ldvl);
                        let (vl_col1, vl_rest) = vl[c1_off..].split_at_mut(ldvl);
                        let (vl_col2, vl_tail) = vl_rest.split_at_mut(ldvl);
                        dgemv(
                            b'N',
                            n,
                            n - ki - 1,
                            1.0,
                            vl_tail,
                            ldvl,
                            &work[ki + 1 + iv * n..],
                            1,
                            work[ki - 1 + iv * n],
                            vl_col1,
                            1,
                        )?;
                        dgemv(
                            b'N',
                            n,
                            n - ki - 1,
                            1.0,
                            vl_tail,
                            ldvl,
                            &work[ki + 1 + (iv + 1) * n..],
                            1,
                            work[ki + (iv + 1) * n],
                            vl_col2,
                            1,
                        )?;
                    } else {
                        dscal(
                            n,
                            work[ki - 1 + iv * n],
                            &mut vl[ix(1, ki, ldvl)..],
                            1,
                        );
                        dscal(
                            n,
                            work[ki + (iv + 1) * n],
                            &mut vl[ix(1, ki + 1, ldvl)..],
                            1,
                        );
                    }
                    let mut emax = 0.0_f64;
                    for k in 1..=n {
                        emax = fmax(
                            emax,
                            vl[ix(k, ki, ldvl)].abs()
                                + vl[ix(k, ki + 1, ldvl)].abs(),
                        );
                    }
                    let remax = 1.0 / emax;
                    dscal(n, remax, &mut vl[ix(1, ki, ldvl)..], 1);
                    dscal(n, remax, &mut vl[ix(1, ki + 1, ldvl)..], 1);
                } else {
                    for k in 1..ki {
                        work[k - 1 + iv * n] = 0.0;
                        work[k - 1 + (iv + 1) * n] = 0.0;
                    }
                    iscomplex[iv] = ip;
                    iscomplex[iv + 1] = -ip;
                    iv += 1;
                }
            }
            if nb > 1 {
                let ki2 = if ip == 0 { ki } else { ki + 1 };
                if iv >= nb - 1 || ki2 == n {
                    let (w_in, w_out) = work.split_at_mut((nb + 1) * n);
                    dgemm(
                        b'N',
                        b'N',
                        n,
                        iv,
                        n - ki2 + iv,
                        1.0,
                        &vl[ix(1, ki2 - iv + 1, ldvl)..],
                        ldvl,
                        &w_in[ki2 - iv + n - 1..],
                        n,
                        0.0,
                        w_out,
                        n,
                    )?;
                    let mut remax = 1.0_f64;
                    for k in 1..=iv {
                        if iscomplex[k] == 0 {
                            let ii = idamax(n, &work[(nb + k) * n..], 1);
                            remax = 1.0 / work[ii - 1 + (nb + k) * n].abs();
                        } else if iscomplex[k] == 1 {
                            let mut emax = 0.0_f64;
                            for ii in 1..=n {
                                emax = fmax(
                                    emax,
                                    work[ii - 1 + (nb + k) * n].abs()
                                        + work[ii - 1 + (nb + k + 1) * n].abs(),
                                );
                            }
                            remax = 1.0 / emax;
                        }
                        dscal(n, remax, &mut work[(nb + k) * n..], 1);
                    }
                    dlacpy(
                        b'F',
                        n,
                        iv,
                        &work[(nb + 1) * n..],
                        n,
                        &mut vl[ix(1, ki2 - iv + 1, ldvl)..],
                        ldvl,
                    );
                    iv = 1;
                } else {
                    iv += 1;
                }
            }
            is_col += 1;
            if ip != 0 {
                is_col += 1;
            }
        }
    }
    Ok((m, maxwrk))
}

fn normalize_eigenvectors(
    n: usize,
    wi: &[f64],
    v: &mut [f64],
    ldv: usize,
    work: &mut [f64],
) {
    for i in 1..=n {
        if wi[i - 1] == 0.0 {
            let scl = 1.0 / dnrm2(n, &v[ix(1, i, ldv)..], 1);
            dscal(n, scl, &mut v[ix(1, i, ldv)..], 1);
        } else if wi[i - 1] > 0.0 {
            let scl = 1.0
                / super::aux::dlapy2(
                    dnrm2(n, &v[ix(1, i, ldv)..], 1),
                    dnrm2(n, &v[ix(1, i + 1, ldv)..], 1),
                );
            dscal(n, scl, &mut v[ix(1, i, ldv)..], 1);
            dscal(n, scl, &mut v[ix(1, i + 1, ldv)..], 1);
            for k in 1..=n {
                let re = v[ix(k, i, ldv)];
                let im = v[ix(k, i + 1, ldv)];
                work[k - 1] = re * re + im * im;
            }
            let k = idamax(n, work, 1);
            let (cs, sn, _r) = dlartg(v[ix(k, i, ldv)], v[ix(k, i + 1, ldv)]);
            drot_in(n, v, ix(1, i, ldv), 1, ix(1, i + 1, ldv), 1, cs, sn);
            v[ix(k, i + 1, ldv)] = 0.0;
        }
    }
}

/// `DGEEV` (`SRC/dgeev.f`): compute eigenvalues and (optionally) left/right
/// eigenvectors of a real `n x n` matrix `A`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] or [`LapackError::UnportedPath`].
#[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]
pub fn dgeev(
    jobvl: u8,
    jobvr: u8,
    n: usize,
    a: &mut [f64],
    lda: usize,
    vl: &mut [f64],
    ldvl: usize,
    vr: &mut [f64],
    ldvr: usize,
    lwork: usize,
) -> Result<Dgeev, LapackError> {
    let wantvl = lsame(jobvl, b'V');
    let wantvr = lsame(jobvr, b'V');
    if !wantvl && !lsame(jobvl, b'N') {
        return Err(bad("DGEEV ", -1));
    }
    if !wantvr && !lsame(jobvr, b'N') {
        return Err(bad("DGEEV ", -2));
    }
    if lda < n.max(1) {
        return Err(bad("DGEEV ", -5));
    }
    if ldvl < 1 || (wantvl && ldvl < n) {
        return Err(bad("DGEEV ", -9));
    }
    if ldvr < 1 || (wantvr && ldvr < n) {
        return Err(bad("DGEEV ", -11));
    }
    let minwrk;
    let mut maxwrk;
    if n == 0 {
        minwrk = 1;
        maxwrk = 1;
    } else {
        maxwrk = 2 * n + n * NB32;
        if wantvl || wantvr {
            minwrk = 4 * n;
            maxwrk = maxwrk.max(2 * n + (n - 1) * NB32);
            maxwrk = maxwrk.max(n + 1).max(2 * n);
            maxwrk = maxwrk.max(n + 1.max(n + 2 * n * NB64));
            maxwrk = maxwrk.max(4 * n);
        } else {
            minwrk = 3 * n;
            maxwrk = maxwrk.max(n + 1).max(2 * n);
        }
        maxwrk = maxwrk.max(minwrk);
    }
    if lwork < minwrk {
        return Err(bad("DGEEV ", -13));
    }
    let work1 = maxwrk as f64;
    if n == 0 {
        return Ok(Dgeev {
            wr: Vec::new(),
            wi: Vec::new(),
            work1,
            info: 0,
        });
    }

    let eps = dlamch::PREC;
    let mut smlnum = dlamch::SFMIN;
    smlnum = smlnum.sqrt() / eps;
    let bignum = 1.0 / smlnum;

    let anrm = dlange(b'M', n, n, a, lda)?;
    let mut scalea = false;
    let mut cscale = 0.0_f64;
    if anrm > 0.0 && anrm < smlnum {
        scalea = true;
        cscale = smlnum;
    } else if anrm > bignum {
        scalea = true;
        cscale = bignum;
    }
    if scalea {
        dlascl_g(anrm, cscale, n, n, a, lda)?;
    }

    let mut scale_bal = vec![0.0_f64; n];
    let (ilo, ihi) = dgebal(b'B', n, a, lda, &mut scale_bal)?;
    let mut tau = vec![0.0_f64; n];
    let lw_hrd = (lwork - 2 * n) as isize;
    dgehrd(n, ilo, ihi, a, lda, &mut tau, lw_hrd)?;

    let mut wr = vec![0.0_f64; n];
    let mut wi = vec![0.0_f64; n];
    let lw_hseqr = (lwork - n) as isize;
    let side;
    let info;
    if wantvl {
        side = b'L';
        dlacpy(b'L', n, n, a, lda, vl, ldvl);
        dorghr(n, ilo, ihi, vl, ldvl, &tau, lw_hrd)?;
        (_, info) = dhseqr(
            b'S', b'V', n, ilo, ihi, a, lda, &mut wr, &mut wi, vl, ldvl,
            lw_hseqr,
        )?;
        if wantvr {
            dlacpy(b'F', n, n, vl, ldvl, vr, ldvr);
        }
    } else if wantvr {
        side = b'R';
        dlacpy(b'L', n, n, a, lda, vr, ldvr);
        dorghr(n, ilo, ihi, vr, ldvr, &tau, lw_hrd)?;
        (_, info) = dhseqr(
            b'S', b'V', n, ilo, ihi, a, lda, &mut wr, &mut wi, vr, ldvr,
            lw_hseqr,
        )?;
    } else {
        side = b'N';
        (_, info) = dhseqr(
            b'E', b'N', n, ilo, ihi, a, lda, &mut wr, &mut wi, vr, ldvr,
            lw_hseqr,
        )?;
    }
    let eff_side = if wantvl && wantvr { b'B' } else { side };

    if info == 0 {
        if wantvl || wantvr {
            let mut sel = vec![false; n];
            dtrevc3(
                eff_side, b'B', &mut sel, n, a, lda, vl, ldvl, vr, ldvr, n,
                lw_hseqr,
            )?;
        }
        let mut work_norm = vec![0.0_f64; n];
        if wantvl {
            dgebak(b'B', b'L', n, ilo, ihi, &scale_bal, n, vl, ldvl)?;
            normalize_eigenvectors(n, &wi, vl, ldvl, &mut work_norm);
        }
        if wantvr {
            dgebak(b'B', b'R', n, ilo, ihi, &scale_bal, n, vr, ldvr)?;
            normalize_eigenvectors(n, &wi, vr, ldvr, &mut work_norm);
        }
    }

    if scalea {
        dlascl_g(
            cscale,
            anrm,
            n - info,
            1,
            &mut wr[info..],
            (n - info).max(1),
        )?;
        dlascl_g(
            cscale,
            anrm,
            n - info,
            1,
            &mut wi[info..],
            (n - info).max(1),
        )?;
        if info > 0 && ilo > 1 {
            dlascl_g(cscale, anrm, ilo - 1, 1, &mut wr, n)?;
            dlascl_g(cscale, anrm, ilo - 1, 1, &mut wi, n)?;
        }
    }
    Ok(Dgeev { wr, wi, work1, info })
}
