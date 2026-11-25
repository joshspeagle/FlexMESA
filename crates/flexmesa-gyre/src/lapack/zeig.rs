// SPDX-License-Identifier: GPL-3.0-only
//! Reference LAPACK 3.12.0 complex nonsymmetric eigensolver (`SRC/zgeev.f`
//! and callees: `zgebal.f`, `zgebak.f`, `zgehrd.f`, `zgehd2.f`, `zunghr.f`,
//! `zhseqr.f`, `zlahqr.f`, `zlaset.f`, `zlatrs.f`, `ztrevc3.f`,
//! `zlascl.f` `'G'`), double precision, as the MESA SDK `26.6.1` builds them
//! (`gfortran -O2 -fPIC`, with CRMATH `abs_c_dp`, `sqrt_c_dp`, and `dcmplx`).

use super::aux::{fmax, fmin};
use super::blas::{dscal, idamax};
use super::zaux::{zlacpy, zladiv, zlange, zlarf, zlarfg};
use super::zblas::{
    dcabs1, dzasum, dznrm2, izamax, zaxpy, zcopy, zdotc, zdotu, zdscal, zgemm,
    zgemv, zscal, zswap_in, ztrsv,
};
use super::zsvd::zungqr;
use super::{dlamch, ix, lsame, LapackError};
use crate::crmath;
use crate::fcomplex::Cplx;

const ZERO: Cplx = Cplx::new(0.0, 0.0);
const ONE: Cplx = Cplx::new(1.0, 0.0);
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

#[inline]
fn cabs2(z: Cplx) -> f64 {
    (z.re / 2.0).abs() + (z.im / 2.0).abs()
}

/// Output of [`zgeev`].
#[derive(Debug, Clone, PartialEq)]
pub struct Zgeev {
    /// Complex eigenvalues of length `n`.
    pub w: Vec<Cplx>,
    /// `WORK(1)` on exit (`MAXWRK`).
    pub work1: Cplx,
    /// `INFO` from `ZHSEQR` (`0` on convergence).
    pub info: usize,
}

/// `ZLASET` (`SRC/zlaset.f`): initialize an `m x n` complex matrix with
/// `alpha` off-diagonal and `beta` on the diagonal.
pub fn zlaset(
    uplo: u8,
    m: usize,
    n: usize,
    alpha: Cplx,
    beta: Cplx,
    a: &mut [Cplx],
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

/// `ZLASCL` (`SRC/zlascl.f`), `TYPE = 'G'`: multiply `m x n` `A` by
/// real ratio `cto/cfrom`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] on invalid dimensions or `NaN`/`0`.
#[allow(clippy::float_cmp)]
pub fn zlascl_g(
    cfrom: f64,
    cto: f64,
    m: usize,
    n: usize,
    a: &mut [Cplx],
    lda: usize,
) -> Result<(), LapackError> {
    if cfrom == 0.0 || cfrom.is_nan() {
        return Err(bad("ZLASCL", -4));
    }
    if cto.is_nan() {
        return Err(bad("ZLASCL", -5));
    }
    if lda < m.max(1) {
        return Err(bad("ZLASCL", -9));
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
                a[ix(i, j, lda)] = Cplx::real_mul(mul, a[ix(i, j, lda)]);
            }
        }
        if done {
            break;
        }
    }
    Ok(())
}

/// `ZGEBAL` (`SRC/zgebal.f`, SDK CRMATH build using `abs_c_dp`): balance
/// complex `n x n` matrix `A`. Returns `(ilo, ihi)`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] on invalid `job`/`lda` or `NaN`.
#[allow(clippy::cast_precision_loss, clippy::float_cmp)]
pub fn zgebal(
    job: u8,
    n: usize,
    a: &mut [Cplx],
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
        return Err(bad("ZGEBAL", -1));
    }
    if lda < n.max(1) {
        return Err(bad("ZGEBAL", -4));
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
        let mut noconv = true;
        while noconv {
            noconv = false;
            for i in (1..=l).rev() {
                let mut canswap = true;
                for j in 1..=l {
                    let e = a[ix(i, j, lda)];
                    if i != j && (e.re != 0.0 || e.im != 0.0) {
                        canswap = false;
                        break;
                    }
                }
                if canswap {
                    scale[l - 1] = i as f64;
                    if i != l {
                        zswap_in(l, a, ix(1, i, lda), 1, ix(1, l, lda), 1);
                        zswap_in(
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
                    let e = a[ix(i, j, lda)];
                    if i != j && (e.re != 0.0 || e.im != 0.0) {
                        canswap = false;
                        break;
                    }
                }
                if canswap {
                    scale[k - 1] = j as f64;
                    if j != k {
                        zswap_in(l, a, ix(1, j, lda), 1, ix(1, k, lda), 1);
                        zswap_in(
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
            let mut c = dznrm2(l - k + 1, &a[ix(k, i, lda)..], 1);
            let mut r = dznrm2(l - k + 1, &a[ix(i, k, lda)..], lda);
            let ica = izamax(l, &a[ix(1, i, lda)..], 1);
            let mut ca = crmath::abs_c(a[ix(ica, i, lda)]);
            let ira = izamax(n - k + 1, &a[ix(i, k, lda)..], lda);
            let mut ra = crmath::abs_c(a[ix(i, ira + k - 1, lda)]);
            if c == 0.0 || r == 0.0 {
                continue;
            }
            if (c + ca + r + ra).is_nan() {
                return Err(bad("ZGEBAL", -3));
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
            zdscal(n - k + 1, g, &mut a[ix(i, k, lda)..], lda);
            zdscal(l, f, &mut a[ix(1, i, lda)..], 1);
        }
    }
    Ok((k, l))
}

/// `ZGEBAK` (`SRC/zgebak.f`): back-transform eigenvectors from `ZGEBAL`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] on invalid arguments.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn zgebak(
    job: u8,
    side: u8,
    n: usize,
    ilo: usize,
    ihi: usize,
    scale: &[f64],
    m: usize,
    v: &mut [Cplx],
    ldv: usize,
) -> Result<(), LapackError> {
    let rightv = lsame(side, b'R');
    let leftv = lsame(side, b'L');
    if !lsame(job, b'N')
        && !lsame(job, b'P')
        && !lsame(job, b'S')
        && !lsame(job, b'B')
    {
        return Err(bad("ZGEBAK", -1));
    }
    if !rightv && !leftv {
        return Err(bad("ZGEBAK", -2));
    }
    if ilo < 1 || ilo > n.max(1) {
        return Err(bad("ZGEBAK", -4));
    }
    if ihi < ilo.min(n) || ihi > n {
        return Err(bad("ZGEBAK", -5));
    }
    if ldv < n.max(1) {
        return Err(bad("ZGEBAK", -9));
    }
    if n == 0 || m == 0 || lsame(job, b'N') {
        return Ok(());
    }
    if ilo != ihi && (lsame(job, b'S') || lsame(job, b'B')) {
        if rightv {
            for i in ilo..=ihi {
                let s = scale[i - 1];
                zdscal(m, s, &mut v[ix(i, 1, ldv)..], ldv);
            }
        }
        if leftv {
            for i in ilo..=ihi {
                let s = 1.0 / scale[i - 1];
                zdscal(m, s, &mut v[ix(i, 1, ldv)..], ldv);
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
            zswap_in(m, v, ix(i, 1, ldv), ldv, ix(k, 1, ldv), ldv);
        }
    }
    Ok(())
}

/// `ZGEHD2` (`SRC/zgehd2.f`): unblocked reduction to upper Hessenberg form.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] or [`zlarf`] errors.
pub fn zgehd2(
    n: usize,
    ilo: usize,
    ihi: usize,
    a: &mut [Cplx],
    lda: usize,
    tau: &mut [Cplx],
    work: &mut [Cplx],
) -> Result<(), LapackError> {
    if ilo < 1 || ilo > n.max(1) {
        return Err(bad("ZGEHD2", -2));
    }
    if ihi < ilo.min(n) || ihi > n {
        return Err(bad("ZGEHD2", -3));
    }
    if lda < n.max(1) {
        return Err(bad("ZGEHD2", -5));
    }
    if ihi <= ilo {
        return Ok(());
    }
    for i in ilo..ihi {
        let mut alpha = a[ix(i + 1, i, lda)];
        let xs = ix((i + 2).min(n), i, lda);
        let mut v = a[xs..].to_vec();
        tau[i - 1] = zlarfg(ihi - i, &mut alpha, &mut v, 1);
        a[xs..].copy_from_slice(&v);
        a[ix(i + 1, i, lda)] = ONE;

        let hv = a[ix(i + 1, i, lda)..ix(ihi, i, lda) + 1].to_vec();
        zlarf(
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
        zlarf(
            b'L',
            ihi - i,
            n - i,
            &hv,
            1,
            tau[i - 1].conj(),
            &mut a[ix(i + 1, i + 1, lda)..],
            lda,
            work,
        )?;
        a[ix(i + 1, i, lda)] = alpha;
    }
    Ok(())
}

/// `ZGEHRD` (`SRC/zgehrd.f`): reduction to upper Hessenberg form
/// (unblocked for `nh <= 128`). Returns `WORK(1)` (`LWKOPT`).
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] or [`LapackError::UnportedPath`].
#[allow(clippy::cast_precision_loss)]
pub fn zgehrd(
    n: usize,
    ilo: usize,
    ihi: usize,
    a: &mut [Cplx],
    lda: usize,
    tau: &mut [Cplx],
    lwork: isize,
) -> Result<Cplx, LapackError> {
    const TSIZE: usize = (64 + 1) * 64;
    let lquery = lwork == -1;
    if ilo < 1 || ilo > n.max(1) {
        return Err(bad("ZGEHRD", -2));
    }
    if ihi < ilo.min(n) || ihi > n {
        return Err(bad("ZGEHRD", -3));
    }
    if lda < n.max(1) {
        return Err(bad("ZGEHRD", -5));
    }
    if !lquery && lwork < (n.max(1) as isize) {
        return Err(bad("ZGEHRD", -8));
    }
    let lwkopt = Cplx::from_real((n * NB32 + TSIZE) as f64);
    if lquery {
        return Ok(lwkopt);
    }
    for i in 1..ilo {
        tau[i - 1] = ZERO;
    }
    for i in ihi.max(1)..n {
        tau[i - 1] = ZERO;
    }
    let nh = ihi + 1 - ilo;
    if nh <= 1 {
        return Ok(ONE);
    }
    if nh > NX128 {
        return Err(unported("ZGEHRD", "297-342"));
    }
    let mut work = vec![ZERO; n.max(1)];
    zgehd2(n, ilo, ihi, a, lda, tau, &mut work)?;
    Ok(lwkopt)
}

/// `ZUNGHR` (`SRC/zunghr.f`): generate unitary `Q` from `ZGEHRD`.
/// Returns `WORK(1)` (`LWKOPT`).
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] or [`zungqr`] errors.
#[allow(clippy::cast_precision_loss)]
pub fn zunghr(
    n: usize,
    ilo: usize,
    ihi: usize,
    a: &mut [Cplx],
    lda: usize,
    tau: &[Cplx],
    lwork: isize,
) -> Result<Cplx, LapackError> {
    let lquery = lwork == -1;
    if ilo < 1 || ilo > n.max(1) {
        return Err(bad("ZUNGHR", -2));
    }
    if ihi < ilo.min(n) || ihi > n {
        return Err(bad("ZUNGHR", -3));
    }
    let nh = ihi - ilo;
    if lda < n.max(1) {
        return Err(bad("ZUNGHR", -5));
    }
    if !lquery && lwork < (nh.max(1) as isize) {
        return Err(bad("ZUNGHR", -8));
    }
    let lwkopt = Cplx::from_real((nh.max(1) * NB32) as f64);
    if lquery {
        return Ok(lwkopt);
    }
    if n == 0 {
        return Ok(ONE);
    }
    for j in (ilo + 1..=ihi).rev() {
        for i in 1..j {
            a[ix(i, j, lda)] = ZERO;
        }
        for i in j + 1..=ihi {
            a[ix(i, j, lda)] = a[ix(i, j - 1, lda)];
        }
        for i in ihi + 1..=n {
            a[ix(i, j, lda)] = ZERO;
        }
    }
    for j in 1..=ilo {
        for i in 1..=n {
            a[ix(i, j, lda)] = ZERO;
        }
        a[ix(j, j, lda)] = ONE;
    }
    for j in ihi + 1..=n {
        for i in 1..=n {
            a[ix(i, j, lda)] = ZERO;
        }
        a[ix(j, j, lda)] = ONE;
    }
    if nh > 0 {
        #[allow(clippy::cast_sign_loss)]
        zungqr(
            nh,
            nh,
            nh,
            &mut a[ix(ilo + 1, ilo + 1, lda)..],
            lda,
            &tau[ilo - 1..],
            lwork as usize,
        )?;
    }
    Ok(lwkopt)
}

/// `ZLAHQR` (`SRC/zlahqr.f`, SDK CRMATH build with `abs_c_dp` and
/// `sqrt_c_dp`): complex single-shift QR algorithm on upper Hessenberg `H`.
/// Returns `INFO`.
#[allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines
)]
pub fn zlahqr(
    wantt: bool,
    wantz: bool,
    n: usize,
    ilo: usize,
    ihi: usize,
    h: &mut [Cplx],
    ldh: usize,
    w: &mut [Cplx],
    iloz: usize,
    ihiz: usize,
    z: &mut [Cplx],
    ldz: usize,
) -> usize {
    const DAT1: f64 = 3.0 / 4.0;
    const KEXSH: usize = 10;

    if n == 0 {
        return 0;
    }
    if ilo == ihi {
        w[ilo - 1] = h[ix(ilo, ilo, ldh)];
        return 0;
    }
    for j in ilo..=ihi.saturating_sub(3) {
        h[ix(j + 2, j, ldh)] = ZERO;
        h[ix(j + 3, j, ldh)] = ZERO;
    }
    if ilo + 2 <= ihi {
        h[ix(ihi, ihi - 2, ldh)] = ZERO;
    }
    let (jlo, jhi) = if wantt { (1, n) } else { (ilo, ihi) };
    for i in ilo + 1..=ihi {
        if h[ix(i, i - 1, ldh)].im != 0.0 {
            let mut sc =
                h[ix(i, i - 1, ldh)].div_real(dcabs1(h[ix(i, i - 1, ldh)]));
            sc = sc.conj().div_real(crmath::abs_c(sc));
            h[ix(i, i - 1, ldh)] =
                Cplx::from_real(crmath::abs_c(h[ix(i, i - 1, ldh)]));
            zscal(jhi - i + 1, sc, &mut h[ix(i, i, ldh)..], ldh);
            zscal(
                jhi.min(i + 1) - jlo + 1,
                sc.conj(),
                &mut h[ix(jlo, i, ldh)..],
                1,
            );
            if wantz {
                zscal(
                    ihiz - iloz + 1,
                    sc.conj(),
                    &mut z[ix(iloz, i, ldz)..],
                    1,
                );
            }
        }
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
    let mut v = [ZERO; 2];
    let mut i = ihi;

    while i >= ilo {
        let mut l = ilo;
        let mut split = false;
        for _its in 0..=itmax {
            let mut k = i;
            while k > l {
                if dcabs1(h[ix(k, k - 1, ldh)]) <= smlnum {
                    break;
                }
                let mut tst =
                    dcabs1(h[ix(k - 1, k - 1, ldh)]) + dcabs1(h[ix(k, k, ldh)]);
                if tst == 0.0 {
                    if k >= ilo + 2 {
                        tst += h[ix(k - 1, k - 2, ldh)].re.abs();
                    }
                    if k < ihi {
                        tst += h[ix(k + 1, k, ldh)].re.abs();
                    }
                }
                if h[ix(k, k - 1, ldh)].re.abs() <= ulp * tst {
                    let ab = fmax(
                        dcabs1(h[ix(k, k - 1, ldh)]),
                        dcabs1(h[ix(k - 1, k, ldh)]),
                    );
                    let ba = fmin(
                        dcabs1(h[ix(k, k - 1, ldh)]),
                        dcabs1(h[ix(k - 1, k, ldh)]),
                    );
                    let aa = fmax(
                        dcabs1(h[ix(k, k, ldh)]),
                        dcabs1(h[ix(k - 1, k - 1, ldh)] - h[ix(k, k, ldh)]),
                    );
                    let bb = fmin(
                        dcabs1(h[ix(k, k, ldh)]),
                        dcabs1(h[ix(k - 1, k - 1, ldh)] - h[ix(k, k, ldh)]),
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
                h[ix(l, l - 1, ldh)] = ZERO;
            }
            if l >= i {
                split = true;
                break;
            }
            kdefl += 1;
            if !wantt {
                i1 = l;
                i2 = i;
            }
            let mut t;
            if kdefl.is_multiple_of(2 * KEXSH) {
                let s = DAT1 * h[ix(i, i - 1, ldh)].re.abs();
                t = Cplx::from_real(s) + h[ix(i, i, ldh)];
            } else if kdefl.is_multiple_of(KEXSH) {
                let s = DAT1 * h[ix(l + 1, l, ldh)].re.abs();
                t = Cplx::from_real(s) + h[ix(l, l, ldh)];
            } else {
                t = h[ix(i, i, ldh)];
                let u = crmath::sqrt_c(h[ix(i - 1, i, ldh)])
                    * crmath::sqrt_c(h[ix(i, i - 1, ldh)]);
                let mut s = dcabs1(u);
                if s != 0.0 {
                    let x = Cplx::real_mul(0.5, h[ix(i - 1, i - 1, ldh)] - t);
                    let sx = dcabs1(x);
                    s = fmax(s, sx);
                    let xs = x.div_real(s);
                    let us = u.div_real(s);
                    let mut y = Cplx::real_mul(s, crmath::sqrt_c(xs * xs + us * us));
                    if sx > 0.0 {
                        let xsx = x.div_real(sx);
                        if xsx.re * y.re + xsx.im * y.im < 0.0 {
                            y = -y;
                        }
                    }
                    t = t - u * zladiv(u, x + y);
                }
            }

            let mut m = i - 1;
            let mut m_hit = false;
            while m > l {
                let h11 = h[ix(m, m, ldh)];
                let h22 = h[ix(m + 1, m + 1, ldh)];
                let h11s = h11 - t;
                let h21 = h[ix(m + 1, m, ldh)].re;
                let s = dcabs1(h11s) + h21.abs();
                let h11s = h11s.div_real(s);
                let h21 = h21 / s;
                v[0] = h11s;
                v[1] = Cplx::from_real(h21);
                let h10 = h[ix(m, m - 1, ldh)].re;
                if h10.abs() * h21.abs()
                    <= ulp * (dcabs1(h11s) * (dcabs1(h11) + dcabs1(h22)))
                {
                    m_hit = true;
                    break;
                }
                m -= 1;
            }
            if !m_hit {
                let h11 = h[ix(l, l, ldh)];
                let h11s = h11 - t;
                let h21 = h[ix(l + 1, l, ldh)].re;
                let s = dcabs1(h11s) + h21.abs();
                v[0] = h11s.div_real(s);
                v[1] = Cplx::from_real(h21 / s);
                m = l;
            }

            for k in m..i {
                if k > m {
                    zcopy(2, &h[ix(k, k - 1, ldh)..], 1, &mut v, 1);
                }
                let (mut v0, mut v1) = (v[0], [v[1]]);
                let t1 = zlarfg(2, &mut v0, &mut v1, 1);
                v = [v0, v1[0]];
                if k > m {
                    h[ix(k, k - 1, ldh)] = v[0];
                    h[ix(k + 1, k - 1, ldh)] = ZERO;
                }
                let v2 = v[1];
                let t2 = (t1 * v2).re;
                for j in k..=i2 {
                    let sum = t1.conj() * h[ix(k, j, ldh)]
                        + Cplx::real_mul(t2, h[ix(k + 1, j, ldh)]);
                    h[ix(k, j, ldh)] = h[ix(k, j, ldh)] - sum;
                    h[ix(k + 1, j, ldh)] = h[ix(k + 1, j, ldh)] - sum * v2;
                }
                for j in i1..=i.min(k + 2) {
                    let sum = t1 * h[ix(j, k, ldh)]
                        + Cplx::real_mul(t2, h[ix(j, k + 1, ldh)]);
                    h[ix(j, k, ldh)] = h[ix(j, k, ldh)] - sum;
                    h[ix(j, k + 1, ldh)] = h[ix(j, k + 1, ldh)] - sum * v2.conj();
                }
                if wantz {
                    for j in iloz..=ihiz {
                        let sum = t1 * z[ix(j, k, ldz)]
                            + Cplx::real_mul(t2, z[ix(j, k + 1, ldz)]);
                        z[ix(j, k, ldz)] = z[ix(j, k, ldz)] - sum;
                        z[ix(j, k + 1, ldz)] =
                            z[ix(j, k + 1, ldz)] - sum * v2.conj();
                    }
                }
                if k == m && m > l {
                    let mut temp = ONE - t1;
                    temp = temp.div_real(crmath::abs_c(temp));
                    h[ix(m + 1, m, ldh)] = h[ix(m + 1, m, ldh)] * temp.conj();
                    if m + 2 <= i {
                        h[ix(m + 2, m + 1, ldh)] = h[ix(m + 2, m + 1, ldh)] * temp;
                    }
                    for j in m..=i {
                        if j != m + 1 {
                            if i2 > j {
                                zscal(
                                    i2 - j,
                                    temp,
                                    &mut h[ix(j, j + 1, ldh)..],
                                    ldh,
                                );
                            }
                            zscal(
                                j - i1,
                                temp.conj(),
                                &mut h[ix(i1, j, ldh)..],
                                1,
                            );
                            if wantz {
                                zscal(
                                    nz,
                                    temp.conj(),
                                    &mut z[ix(iloz, j, ldz)..],
                                    1,
                                );
                            }
                        }
                    }
                }
            }
            let temp = h[ix(i, i - 1, ldh)];
            if temp.im != 0.0 {
                let rtemp = crmath::abs_c(temp);
                h[ix(i, i - 1, ldh)] = Cplx::from_real(rtemp);
                let temp_n = temp.div_real(rtemp);
                if i2 > i {
                    zscal(
                        i2 - i,
                        temp_n.conj(),
                        &mut h[ix(i, i + 1, ldh)..],
                        ldh,
                    );
                }
                zscal(i - i1, temp_n, &mut h[ix(i1, i, ldh)..], 1);
                if wantz {
                    zscal(nz, temp_n, &mut z[ix(iloz, i, ldz)..], 1);
                }
            }
        }
        if !split {
            return i;
        }
        w[i - 1] = h[ix(i, i, ldh)];
        kdefl = 0;
        if l == 1 {
            break;
        }
        i = l - 1;
    }
    0
}

/// `ZHSEQR` (`SRC/zhseqr.f`): compute eigenvalues and optionally Schur
/// vectors (`n <= 75` uses `ZLAHQR`). Returns `(work1, info)`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] or [`LapackError::UnportedPath`].
#[allow(clippy::cast_precision_loss)]
pub fn zhseqr(
    job: u8,
    compz: u8,
    n: usize,
    ilo: usize,
    ihi: usize,
    h: &mut [Cplx],
    ldh: usize,
    w: &mut [Cplx],
    z: &mut [Cplx],
    ldz: usize,
    lwork: isize,
) -> Result<(Cplx, usize), LapackError> {
    let wantt = lsame(job, b'S');
    let initz = lsame(compz, b'I');
    let wantz = initz || lsame(compz, b'V');
    let lquery = lwork == -1;
    if !wantt && !lsame(job, b'E') {
        return Err(bad("ZHSEQR", -1));
    }
    if !wantz && !lsame(compz, b'N') {
        return Err(bad("ZHSEQR", -2));
    }
    if ilo < 1 || ilo > n.max(1) {
        return Err(bad("ZHSEQR", -4));
    }
    if ihi < ilo.min(n) || ihi > n {
        return Err(bad("ZHSEQR", -5));
    }
    if ldh < n.max(1) {
        return Err(bad("ZHSEQR", -7));
    }
    if ldz < 1 || (wantz && ldz < n) {
        return Err(bad("ZHSEQR", -11));
    }
    if !lquery && lwork < (n.max(1) as isize) {
        return Err(bad("ZHSEQR", -13));
    }
    if n > NMIN75 {
        return Err(unported("ZHSEQR", "437-478 (ZLAQR0)"));
    }
    let work1 = Cplx::from_real((n.max(1)) as f64);
    if lquery || n == 0 {
        return Ok((work1, 0));
    }
    if ilo > 1 {
        zcopy(ilo - 1, h, ldh + 1, w, 1);
    }
    if ihi < n {
        zcopy(n - ihi, &h[ix(ihi + 1, ihi + 1, ldh)..], ldh + 1, &mut w[ihi..], 1);
    }
    if initz {
        zlaset(b'A', n, n, ZERO, ONE, z, ldz);
    }
    if ilo == ihi {
        w[ilo - 1] = h[ix(ilo, ilo, ldh)];
        return Ok((work1, 0));
    }
    let info = zlahqr(wantt, wantz, n, ilo, ihi, h, ldh, w, ilo, ihi, z, ldz);
    if info > 0 {
        return Err(unported("ZHSEQR", "447-477 (ZLAQR0 fallback)"));
    }
    if wantt && n > 2 {
        zlaset(b'L', n - 2, n - 2, ZERO, ZERO, &mut h[ix(3, 1, ldh)..], ldh);
    }
    Ok((work1, info))
}

/// `ZLATRS` (`SRC/zlatrs.f`): solve triangular system `op(A)*x = scale*b`
/// avoiding overflow. Returns `scale`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] on invalid options or dimensions.
#[allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
pub fn zlatrs(
    uplo: u8,
    trans: u8,
    diag: u8,
    normin: u8,
    n: usize,
    a: &[Cplx],
    lda: usize,
    x: &mut [Cplx],
    cnorm: &mut [f64],
) -> Result<f64, LapackError> {
    let upper = lsame(uplo, b'U');
    let notran = lsame(trans, b'N');
    let tran_t = lsame(trans, b'T');
    let nounit = lsame(diag, b'N');
    let normin_y = lsame(normin, b'Y');
    let normin_n = lsame(normin, b'N');
    if !upper && !lsame(uplo, b'L') {
        return Err(bad("ZLATRS", -1));
    }
    if !notran && !tran_t && !lsame(trans, b'C') {
        return Err(bad("ZLATRS", -2));
    }
    if !nounit && !lsame(diag, b'U') {
        return Err(bad("ZLATRS", -3));
    }
    if !normin_y && !normin_n {
        return Err(bad("ZLATRS", -4));
    }
    if lda < n.max(1) {
        return Err(bad("ZLATRS", -7));
    }
    let mut scale = 1.0_f64;
    if n == 0 {
        return Ok(scale);
    }

    let smlnum = dlamch::SFMIN / dlamch::PREC;
    let bignum = 1.0 / smlnum;

    if normin_n {
        if upper {
            for j in 1..=n {
                cnorm[j - 1] = dzasum(j - 1, &a[ix(1, j, lda)..], 1);
            }
        } else {
            for j in 1..n {
                cnorm[j - 1] = dzasum(n - j, &a[ix(j + 1, j, lda)..], 1);
            }
            cnorm[n - 1] = 0.0;
        }
    }

    let imax = idamax(n, cnorm, 1);
    let mut tmax = cnorm[imax - 1];
    let mut tscal;
    if tmax <= bignum * 0.5 {
        tscal = 1.0;
    } else if tmax <= dlamch::RMAX {
        tscal = 0.5 / (smlnum * tmax);
        dscal(n, tscal, cnorm, 1);
    } else {
        tmax = 0.0;
        if upper {
            for j in 2..=n {
                for i in 1..j {
                    tmax = fmax(
                        tmax,
                        fmax(a[ix(i, j, lda)].re.abs(), a[ix(i, j, lda)].im.abs()),
                    );
                }
            }
        } else {
            for j in 1..n {
                for i in j + 1..=n {
                    tmax = fmax(
                        tmax,
                        fmax(a[ix(i, j, lda)].re.abs(), a[ix(i, j, lda)].im.abs()),
                    );
                }
            }
        }
        if tmax <= dlamch::RMAX {
            tscal = 1.0 / (smlnum * tmax);
            for j in 1..=n {
                if cnorm[j - 1] <= dlamch::RMAX {
                    cnorm[j - 1] *= tscal;
                } else {
                    tscal *= 2.0;
                    cnorm[j - 1] = 0.0;
                    if upper {
                        for i in 1..j {
                            cnorm[j - 1] += tscal * cabs2(a[ix(i, j, lda)]);
                        }
                    } else {
                        for i in j + 1..=n {
                            cnorm[j - 1] += tscal * cabs2(a[ix(i, j, lda)]);
                        }
                    }
                    tscal *= 0.5;
                }
            }
        } else {
            ztrsv(uplo, trans, diag, n, a, lda, x, 1)?;
            return Ok(scale);
        }
    }

    let mut xmax = 0.0_f64;
    for v in &x[..n] {
        xmax = fmax(xmax, cabs2(*v));
    }
    let mut xbnd = xmax;

    let jfirst: usize;
    let jlast: usize;
    let jstep: isize;
    if notran {
        if upper {
            jfirst = n;
            jlast = 1;
            jstep = -1;
        } else {
            jfirst = 1;
            jlast = n;
            jstep = 1;
        }
    } else if upper {
        jfirst = 1;
        jlast = n;
        jstep = 1;
    } else {
        jfirst = n;
        jlast = 1;
        jstep = -1;
    }

    let mut grow;
    if tscal != 1.0 {
        grow = 0.0;
    } else if notran {
        if nounit {
            grow = 0.5 / fmax(xbnd, smlnum);
            xbnd = grow;
            let mut j = jfirst as isize;
            let end = jlast as isize;
            while (jstep > 0 && j <= end) || (jstep < 0 && j >= end) {
                if grow <= smlnum {
                    break;
                }
                let ju = j.cast_unsigned();
                let tjj = dcabs1(a[ix(ju, ju, lda)]);
                if tjj >= smlnum {
                    xbnd = fmin(xbnd, fmin(1.0, tjj) * grow);
                } else {
                    xbnd = 0.0;
                }
                if tjj + cnorm[ju - 1] >= smlnum {
                    grow *= tjj / (tjj + cnorm[ju - 1]);
                } else {
                    grow = 0.0;
                }
                j += jstep;
            }
            if grow > smlnum {
                grow = xbnd;
            }
        } else {
            grow = fmin(1.0, 0.5 / fmax(xbnd, smlnum));
            let mut j = jfirst as isize;
            let end = jlast as isize;
            while (jstep > 0 && j <= end) || (jstep < 0 && j >= end) {
                if grow <= smlnum {
                    break;
                }
                let ju = j.cast_unsigned();
                grow *= 1.0 / (1.0 + cnorm[ju - 1]);
                j += jstep;
            }
        }
    } else if nounit {
        grow = 0.5 / fmax(xbnd, smlnum);
        xbnd = grow;
        let mut j = jfirst as isize;
        let end = jlast as isize;
        while (jstep > 0 && j <= end) || (jstep < 0 && j >= end) {
            if grow <= smlnum {
                break;
            }
            let ju = j.cast_unsigned();
            let xj = 1.0 + cnorm[ju - 1];
            grow = fmin(grow, xbnd / xj);
            let tjj = dcabs1(a[ix(ju, ju, lda)]);
            if tjj >= smlnum {
                if xj > tjj {
                    xbnd *= tjj / xj;
                }
            } else {
                xbnd = 0.0;
            }
            j += jstep;
        }
        grow = fmin(grow, xbnd);
    } else {
        grow = fmin(1.0, 0.5 / fmax(xbnd, smlnum));
        let mut j = jfirst as isize;
        let end = jlast as isize;
        while (jstep > 0 && j <= end) || (jstep < 0 && j >= end) {
            if grow <= smlnum {
                break;
            }
            let ju = j.cast_unsigned();
            grow /= 1.0 + cnorm[ju - 1];
            j += jstep;
        }
    }

    if grow * tscal > smlnum {
        ztrsv(uplo, trans, diag, n, a, lda, x, 1)?;
    } else {
        if xmax > bignum * 0.5 {
            scale = (bignum * 0.5) / xmax;
            zdscal(n, scale, x, 1);
            xmax = bignum;
        } else {
            xmax *= 2.0;
        }
        let mut j = jfirst as isize;
        let end = jlast as isize;
        if notran {
            while (jstep > 0 && j <= end) || (jstep < 0 && j >= end) {
                let ju = j.cast_unsigned();
                let mut xj = dcabs1(x[ju - 1]);
                let mut skip_diag = false;
                let tjjs;
                if nounit {
                    tjjs = Cplx::real_mul(tscal, a[ix(ju, ju, lda)]);
                } else {
                    tjjs = Cplx::from_real(tscal);
                    if tscal == 1.0 {
                        skip_diag = true;
                    }
                }
                if !skip_diag {
                    let tjj = dcabs1(tjjs);
                    if tjj > smlnum {
                        if tjj < 1.0 && xj > tjj * bignum {
                            let rec = 1.0 / xj;
                            zdscal(n, rec, x, 1);
                            scale *= rec;
                            xmax *= rec;
                        }
                        x[ju - 1] = zladiv(x[ju - 1], tjjs);
                        xj = dcabs1(x[ju - 1]);
                    } else if tjj > 0.0 {
                        if xj > tjj * bignum {
                            let mut rec = (tjj * bignum) / xj;
                            if cnorm[ju - 1] > 1.0 {
                                rec /= cnorm[ju - 1];
                            }
                            zdscal(n, rec, x, 1);
                            scale *= rec;
                            xmax *= rec;
                        }
                        x[ju - 1] = zladiv(x[ju - 1], tjjs);
                        xj = dcabs1(x[ju - 1]);
                    } else {
                        for i in 0..n {
                            x[i] = ZERO;
                        }
                        x[ju - 1] = ONE;
                        xj = 1.0;
                        scale = 0.0;
                        xmax = 0.0;
                    }
                }
                if xj > 1.0 {
                    let mut rec = 1.0 / xj;
                    if cnorm[ju - 1] > (bignum - xmax) * rec {
                        rec *= 0.5;
                        zdscal(n, rec, x, 1);
                        scale *= rec;
                    }
                } else if xj * cnorm[ju - 1] > (bignum - xmax) {
                    zdscal(n, 0.5, x, 1);
                    scale *= 0.5;
                }
                if upper {
                    if ju > 1 {
                        let za = -Cplx::real_mul(tscal, x[ju - 1]);
                        let col = a[ix(1, ju, lda)..ix(ju - 1, ju, lda) + 1].to_vec();
                        zaxpy(ju - 1, za, &col, 1, x, 1);
                        let i = izamax(ju - 1, x, 1);
                        xmax = dcabs1(x[i - 1]);
                    }
                } else if ju < n {
                    let za = -Cplx::real_mul(tscal, x[ju - 1]);
                    zaxpy(
                        n - ju,
                        za,
                        &a[ix(ju + 1, ju, lda)..],
                        1,
                        &mut x[ju..],
                        1,
                    );
                    let i = ju + izamax(n - ju, &x[ju..], 1);
                    xmax = dcabs1(x[i - 1]);
                }
                j += jstep;
            }
        } else {
            while (jstep > 0 && j <= end) || (jstep < 0 && j >= end) {
                let ju = j.cast_unsigned();
                let mut xj = dcabs1(x[ju - 1]);
                let mut uscal = Cplx::from_real(tscal);
                let mut rec = 1.0 / fmax(xmax, 1.0);
                let mut tjjs = if nounit {
                    let d = if tran_t {
                        a[ix(ju, ju, lda)]
                    } else {
                        a[ix(ju, ju, lda)].conj()
                    };
                    Cplx::real_mul(tscal, d)
                } else {
                    Cplx::from_real(tscal)
                };
                if cnorm[ju - 1] > (bignum - xj) * rec {
                    rec *= 0.5;
                    let tjj = dcabs1(tjjs);
                    if tjj > 1.0 {
                        rec = fmin(1.0, rec * tjj);
                        uscal = zladiv(uscal, tjjs);
                    }
                    if rec < 1.0 {
                        zdscal(n, rec, x, 1);
                        scale *= rec;
                        xmax *= rec;
                    }
                }
                let mut csumj = ZERO;
                if uscal.feq(ONE) {
                    if upper {
                        csumj = if tran_t {
                            zdotu(ju - 1, &a[ix(1, ju, lda)..], 1, x, 1)
                        } else {
                            zdotc(ju - 1, &a[ix(1, ju, lda)..], 1, x, 1)
                        };
                    } else if ju < n {
                        csumj = if tran_t {
                            zdotu(
                                n - ju,
                                &a[ix(ju + 1, ju, lda)..],
                                1,
                                &x[ju..],
                                1,
                            )
                        } else {
                            zdotc(
                                n - ju,
                                &a[ix(ju + 1, ju, lda)..],
                                1,
                                &x[ju..],
                                1,
                            )
                        };
                    }
                } else if upper {
                    for i in 1..ju {
                        let aij = if tran_t {
                            a[ix(i, ju, lda)]
                        } else {
                            a[ix(i, ju, lda)].conj()
                        };
                        csumj = csumj + (aij * uscal) * x[i - 1];
                    }
                } else if ju < n {
                    for i in ju + 1..=n {
                        let aij = if tran_t {
                            a[ix(i, ju, lda)]
                        } else {
                            a[ix(i, ju, lda)].conj()
                        };
                        csumj = csumj + (aij * uscal) * x[i - 1];
                    }
                }
                if uscal.feq(Cplx::from_real(tscal)) {
                    x[ju - 1] = x[ju - 1] - csumj;
                    xj = dcabs1(x[ju - 1]);
                    let mut skip = false;
                    if nounit {
                        let d = if tran_t {
                            a[ix(ju, ju, lda)]
                        } else {
                            a[ix(ju, ju, lda)].conj()
                        };
                        tjjs = Cplx::real_mul(tscal, d);
                    } else {
                        tjjs = Cplx::from_real(tscal);
                        if tscal == 1.0 {
                            skip = true;
                        }
                    }
                    if !skip {
                        let tjj = dcabs1(tjjs);
                        if tjj > smlnum {
                            if tjj < 1.0 && xj > tjj * bignum {
                                let rec = 1.0 / xj;
                                zdscal(n, rec, x, 1);
                                scale *= rec;
                                xmax *= rec;
                            }
                            x[ju - 1] = zladiv(x[ju - 1], tjjs);
                        } else if tjj > 0.0 {
                            if xj > tjj * bignum {
                                let rec = (tjj * bignum) / xj;
                                zdscal(n, rec, x, 1);
                                scale *= rec;
                                xmax *= rec;
                            }
                            x[ju - 1] = zladiv(x[ju - 1], tjjs);
                        } else {
                            for i in 0..n {
                                x[i] = ZERO;
                            }
                            x[ju - 1] = ONE;
                            scale = 0.0;
                            xmax = 0.0;
                        }
                    }
                } else {
                    x[ju - 1] = zladiv(x[ju - 1], tjjs) - csumj;
                }
                xmax = fmax(xmax, dcabs1(x[ju - 1]));
                j += jstep;
            }
        }
        scale /= tscal;
    }

    if tscal != 1.0 {
        dscal(n, 1.0 / tscal, cnorm, 1);
    }
    Ok(scale)
}

/// `ZTREVC3` (`SRC/ztrevc3.f`, SDK CRMATH build with `dcmplx`): compute
/// eigenvectors of complex upper triangular matrix `T`. Returns `(nout, work1)`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] on invalid options or sizes.
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap
)]
pub fn ztrevc3(
    side: u8,
    howmny: u8,
    select: &[bool],
    n: usize,
    t: &mut [Cplx],
    ldt: usize,
    vl: &mut [Cplx],
    ldvl: usize,
    vr: &mut [Cplx],
    ldvr: usize,
    mm: usize,
    lwork: isize,
) -> Result<(usize, Cplx), LapackError> {
    const NBMIN: usize = 8;
    let bothv = lsame(side, b'B');
    let rightv = lsame(side, b'R') || bothv;
    let leftv = lsame(side, b'L') || bothv;
    let allv = lsame(howmny, b'A');
    let over = lsame(howmny, b'B');
    let somev = lsame(howmny, b'S');
    if !rightv && !leftv {
        return Err(bad("ZTREVC3", -1));
    }
    if !allv && !over && !somev {
        return Err(bad("ZTREVC3", -2));
    }
    if ldt < n.max(1) {
        return Err(bad("ZTREVC3", -6));
    }
    if ldvl < 1 || (leftv && ldvl < n) {
        return Err(bad("ZTREVC3", -8));
    }
    if ldvr < 1 || (rightv && ldvr < n) {
        return Err(bad("ZTREVC3", -10));
    }
    let lquery = lwork == -1;
    if !lquery && lwork < ((2 * n).max(1) as isize) {
        return Err(bad("ZTREVC3", -14));
    }
    let m = if somev {
        select[..n].iter().filter(|&&b| b).count()
    } else {
        n
    };
    if mm < m {
        return Err(bad("ZTREVC3", -11));
    }
    let maxwrk = Cplx::from_real(1.max(n + 2 * n * NB64) as f64);
    if lquery || n == 0 {
        return Ok((m, maxwrk));
    }

    let lwork_u = lwork.cast_unsigned();
    let nb = if over && lwork_u >= n + 2 * n * NBMIN {
        ((lwork_u - n) / (2 * n)).min(128)
    } else {
        1
    };
    let mut work = vec![ZERO; n * (1 + 2 * nb.max(1))];
    let mut rwork = vec![0.0_f64; n];

    let unfl = dlamch::SFMIN;
    let ulp = dlamch::PREC;
    let smlnum = unfl * ((n as f64) / ulp);

    for i in 1..=n {
        work[i - 1] = t[ix(i, i, ldt)];
    }
    rwork[0] = 0.0;
    for j in 2..=n {
        rwork[j - 1] = dzasum(j - 1, &t[ix(1, j, ldt)..], 1);
    }

    if rightv {
        let mut iv = nb;
        let mut is_col = m;
        for ki in (1..=n).rev() {
            if somev && !select[ki - 1] {
                continue;
            }
            let smin = fmax(ulp * dcabs1(t[ix(ki, ki, ldt)]), smlnum);
            work[ki - 1 + iv * n] = ONE;
            for k in 1..ki {
                work[k - 1 + iv * n] = -t[ix(k, ki, ldt)];
            }
            for k in 1..ki {
                t[ix(k, k, ldt)] = t[ix(k, k, ldt)] - t[ix(ki, ki, ldt)];
                if dcabs1(t[ix(k, k, ldt)]) < smin {
                    t[ix(k, k, ldt)] = Cplx::from_real(smin);
                }
            }
            if ki > 1 {
                let scale = zlatrs(
                    b'U',
                    b'N',
                    b'N',
                    b'Y',
                    ki - 1,
                    t,
                    ldt,
                    &mut work[iv * n..],
                    &mut rwork,
                )?;
                work[ki - 1 + iv * n] = Cplx::from_real(scale);
            }
            if !over {
                zcopy(
                    ki,
                    &work[iv * n..],
                    1,
                    &mut vr[ix(1, is_col, ldvr)..],
                    1,
                );
                let ii = izamax(ki, &vr[ix(1, is_col, ldvr)..], 1);
                let remax = 1.0 / dcabs1(vr[ix(ii, is_col, ldvr)]);
                zdscal(ki, remax, &mut vr[ix(1, is_col, ldvr)..], 1);
                for k in ki + 1..=n {
                    vr[ix(k, is_col, ldvr)] = ZERO;
                }
            } else if nb == 1 {
                if ki > 1 {
                    let c_off = ix(1, ki, ldvr);
                    let (vr_head, vr_col) = vr.split_at_mut(c_off);
                    zgemv(
                        b'N',
                        n,
                        ki - 1,
                        ONE,
                        vr_head,
                        ldvr,
                        &work[iv * n..],
                        1,
                        work[ki - 1 + iv * n],
                        vr_col,
                        1,
                    )?;
                }
                let ii = izamax(n, &vr[ix(1, ki, ldvr)..], 1);
                let remax = 1.0 / dcabs1(vr[ix(ii, ki, ldvr)]);
                zdscal(n, remax, &mut vr[ix(1, ki, ldvr)..], 1);
            } else {
                for k in ki + 1..=n {
                    work[k - 1 + iv * n] = ZERO;
                }
                if iv == 1 || ki == 1 {
                    let (w_in, w_out) = work.split_at_mut((nb + iv) * n);
                    zgemm(
                        b'N',
                        b'N',
                        n,
                        nb - iv + 1,
                        ki + nb - iv,
                        ONE,
                        vr,
                        ldvr,
                        &w_in[iv * n..],
                        n,
                        ZERO,
                        w_out,
                        n,
                    )?;
                    for k in iv..=nb {
                        let ii = izamax(n, &work[(nb + k) * n..], 1);
                        let remax = 1.0 / dcabs1(work[ii - 1 + (nb + k) * n]);
                        zdscal(n, remax, &mut work[(nb + k) * n..], 1);
                    }
                    zlacpy(
                        b'F',
                        n,
                        nb - iv + 1,
                        &work[(nb + iv) * n..],
                        n,
                        &mut vr[ix(1, ki, ldvr)..],
                        ldvr,
                    );
                    iv = nb;
                } else {
                    iv -= 1;
                }
            }
            for k in 1..ki {
                t[ix(k, k, ldt)] = work[k - 1];
            }
            is_col = is_col.saturating_sub(1);
        }
    }

    if leftv {
        let mut iv = 1_usize;
        let mut is_col = 1_usize;
        for ki in 1..=n {
            if somev && !select[ki - 1] {
                continue;
            }
            let smin = fmax(ulp * dcabs1(t[ix(ki, ki, ldt)]), smlnum);
            work[ki - 1 + iv * n] = ONE;
            for k in ki + 1..=n {
                work[k - 1 + iv * n] = -t[ix(ki, k, ldt)].conj();
            }
            for k in ki + 1..=n {
                t[ix(k, k, ldt)] = t[ix(k, k, ldt)] - t[ix(ki, ki, ldt)];
                if dcabs1(t[ix(k, k, ldt)]) < smin {
                    t[ix(k, k, ldt)] = Cplx::from_real(smin);
                }
            }
            if ki < n {
                let scale = zlatrs(
                    b'U',
                    b'C',
                    b'N',
                    b'Y',
                    n - ki,
                    &t[ix(ki + 1, ki + 1, ldt)..],
                    ldt,
                    &mut work[ki + iv * n..],
                    &mut rwork,
                )?;
                work[ki - 1 + iv * n] = Cplx::from_real(scale);
            }
            if !over {
                zcopy(
                    n - ki + 1,
                    &work[ki - 1 + iv * n..],
                    1,
                    &mut vl[ix(ki, is_col, ldvl)..],
                    1,
                );
                let ii =
                    izamax(n - ki + 1, &vl[ix(ki, is_col, ldvl)..], 1) + ki - 1;
                let remax = 1.0 / dcabs1(vl[ix(ii, is_col, ldvl)]);
                zdscal(n - ki + 1, remax, &mut vl[ix(ki, is_col, ldvl)..], 1);
                for k in 1..ki {
                    vl[ix(k, is_col, ldvl)] = ZERO;
                }
            } else if nb == 1 {
                if ki < n {
                    let c_off = ix(1, ki, ldvl);
                    let (vl_col, vl_tail) = vl[c_off..].split_at_mut(ldvl);
                    zgemv(
                        b'N',
                        n,
                        n - ki,
                        ONE,
                        vl_tail,
                        ldvl,
                        &work[ki + iv * n..],
                        1,
                        work[ki - 1 + iv * n],
                        vl_col,
                        1,
                    )?;
                }
                let ii = izamax(n, &vl[ix(1, ki, ldvl)..], 1);
                let remax = 1.0 / dcabs1(vl[ix(ii, ki, ldvl)]);
                zdscal(n, remax, &mut vl[ix(1, ki, ldvl)..], 1);
            } else {
                for k in 1..ki {
                    work[k - 1 + iv * n] = ZERO;
                }
                if iv == nb || ki == n {
                    let (w_in, w_out) = work.split_at_mut((nb + 1) * n);
                    zgemm(
                        b'N',
                        b'N',
                        n,
                        iv,
                        n - ki + iv,
                        ONE,
                        &vl[ix(1, ki - iv + 1, ldvl)..],
                        ldvl,
                        &w_in[ki - iv + n..],
                        n,
                        ZERO,
                        w_out,
                        n,
                    )?;
                    for k in 1..=iv {
                        let ii = izamax(n, &work[(nb + k) * n..], 1);
                        let remax = 1.0 / dcabs1(work[ii - 1 + (nb + k) * n]);
                        zdscal(n, remax, &mut work[(nb + k) * n..], 1);
                    }
                    zlacpy(
                        b'F',
                        n,
                        iv,
                        &work[(nb + 1) * n..],
                        n,
                        &mut vl[ix(1, ki - iv + 1, ldvl)..],
                        ldvl,
                    );
                    iv = 1;
                } else {
                    iv += 1;
                }
            }
            for k in ki + 1..=n {
                t[ix(k, k, ldt)] = work[k - 1];
            }
            is_col += 1;
        }
    }
    Ok((m, maxwrk))
}

fn normalize_c_eigenvectors(
    n: usize,
    v: &mut [Cplx],
    ldv: usize,
    rwork: &mut [f64],
) {
    for i in 1..=n {
        let scl = 1.0 / dznrm2(n, &v[ix(1, i, ldv)..], 1);
        zdscal(n, scl, &mut v[ix(1, i, ldv)..], 1);
        for k in 1..=n {
            let z = v[ix(k, i, ldv)];
            rwork[k - 1] = z.re * z.re + z.im * z.im;
        }
        let k = idamax(n, rwork, 1);
        let tmp = v[ix(k, i, ldv)].conj().div_real(rwork[k - 1].sqrt());
        zscal(n, tmp, &mut v[ix(1, i, ldv)..], 1);
        v[ix(k, i, ldv)] = Cplx::new(v[ix(k, i, ldv)].re, 0.0);
    }
}

/// `ZGEEV` (`SRC/zgeev.f`): compute eigenvalues and (optionally) left/right
/// eigenvectors of a complex `n x n` matrix `A`.
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] or [`LapackError::UnportedPath`].
#[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]
pub fn zgeev(
    jobvl: u8,
    jobvr: u8,
    n: usize,
    a: &mut [Cplx],
    lda: usize,
    vl: &mut [Cplx],
    ldvl: usize,
    vr: &mut [Cplx],
    ldvr: usize,
    lwork: usize,
) -> Result<Zgeev, LapackError> {
    let wantvl = lsame(jobvl, b'V');
    let wantvr = lsame(jobvr, b'V');
    if !wantvl && !lsame(jobvl, b'N') {
        return Err(bad("ZGEEV ", -1));
    }
    if !wantvr && !lsame(jobvr, b'N') {
        return Err(bad("ZGEEV ", -2));
    }
    if lda < n.max(1) {
        return Err(bad("ZGEEV ", -5));
    }
    if ldvl < 1 || (wantvl && ldvl < n) {
        return Err(bad("ZGEEV ", -8));
    }
    if ldvr < 1 || (wantvr && ldvr < n) {
        return Err(bad("ZGEEV ", -10));
    }
    let minwrk;
    let mut maxwrk;
    if n == 0 {
        minwrk = 1;
        maxwrk = 1;
    } else {
        maxwrk = n + n * NB32;
        minwrk = 2 * n;
        if wantvl || wantvr {
            maxwrk = maxwrk.max(n + (n - 1) * NB32);
            maxwrk = maxwrk.max(n + 1.max(n + 2 * n * NB64));
        }
        maxwrk = maxwrk.max(n.max(1)).max(minwrk);
    }
    if lwork < minwrk {
        return Err(bad("ZGEEV ", -12));
    }
    let work1 = Cplx::from_real(maxwrk as f64);
    if n == 0 {
        return Ok(Zgeev {
            w: Vec::new(),
            work1,
            info: 0,
        });
    }

    let eps = dlamch::PREC;
    let mut smlnum = dlamch::SFMIN;
    smlnum = smlnum.sqrt() / eps;
    let bignum = 1.0 / smlnum;

    let anrm = zlange(b'M', n, n, a, lda)?;
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
        zlascl_g(anrm, cscale, n, n, a, lda)?;
    }

    let mut scale_bal = vec![0.0_f64; n];
    let (ilo, ihi) = zgebal(b'B', n, a, lda, &mut scale_bal)?;
    let mut tau = vec![ZERO; n];
    let lw_hrd = (lwork - n) as isize;
    zgehrd(n, ilo, ihi, a, lda, &mut tau, lw_hrd)?;

    let mut w = vec![ZERO; n];
    let lw_hseqr = lwork as isize;
    let side;
    let info;
    if wantvl {
        side = b'L';
        zlacpy(b'L', n, n, a, lda, vl, ldvl);
        zunghr(n, ilo, ihi, vl, ldvl, &tau, lw_hrd)?;
        (_, info) = zhseqr(b'S', b'V', n, ilo, ihi, a, lda, &mut w, vl, ldvl, lw_hseqr)?;
        if wantvr {
            zlacpy(b'F', n, n, vl, ldvl, vr, ldvr);
        }
    } else if wantvr {
        side = b'R';
        zlacpy(b'L', n, n, a, lda, vr, ldvr);
        zunghr(n, ilo, ihi, vr, ldvr, &tau, lw_hrd)?;
        (_, info) = zhseqr(b'S', b'V', n, ilo, ihi, a, lda, &mut w, vr, ldvr, lw_hseqr)?;
    } else {
        side = b'N';
        (_, info) = zhseqr(b'E', b'N', n, ilo, ihi, a, lda, &mut w, vr, ldvr, lw_hseqr)?;
    }
    let eff_side = if wantvl && wantvr { b'B' } else { side };

    if info == 0 {
        if wantvl || wantvr {
            let sel = vec![false; n];
            ztrevc3(
                eff_side, b'B', &sel, n, a, lda, vl, ldvl, vr, ldvr, n,
                lw_hseqr,
            )?;
        }
        let mut rwork_norm = vec![0.0_f64; n];
        if wantvl {
            zgebak(b'B', b'L', n, ilo, ihi, &scale_bal, n, vl, ldvl)?;
            normalize_c_eigenvectors(n, vl, ldvl, &mut rwork_norm);
        }
        if wantvr {
            zgebak(b'B', b'R', n, ilo, ihi, &scale_bal, n, vr, ldvr)?;
            normalize_c_eigenvectors(n, vr, ldvr, &mut rwork_norm);
        }
    }

    if scalea {
        zlascl_g(
            cscale,
            anrm,
            n - info,
            1,
            &mut w[info..],
            (n - info).max(1),
        )?;
        if info > 0 && ilo > 1 {
            zlascl_g(cscale, anrm, ilo - 1, 1, &mut w, n)?;
        }
    }
    Ok(Zgeev { w, work1, info })
}
