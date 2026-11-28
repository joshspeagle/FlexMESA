//! Powell's `NEWUOA` (2006) unconstrained derivative-free trust-region
//! minimizer, transcribed line by line from MESA `26.4.1`
//! `num/public/num_newuoa.dek:22-34` and `num/private/mod_newuoa.f:7-1581`
//! (`do_newuoa`, `NEWUOB`, `BIGDEN`, `BIGLAG`, `TRSAPP`, `UPDATE`).
//!
//! The method minimizes $f: \mathbb{R}^n \to \mathbb{R}$ with quadratic
//! models that interpolate $f$ at $m = \mathrm{npt}$ points, $n + 2 \le m \le
//! (n + 1)(n + 2) / 2$. The freedom left by the interpolation conditions is
//! taken up by minimizing the Frobenius norm of the change to the model
//! Hessian. The inverse of the interpolation (KKT) matrix is kept in the
//! factored form `BMAT`, `ZMAT` and `IDZ` and updated by `UPDATE` each time an
//! interpolation point moves.
//!
//! # Storage
//!
//! Upstream partitions one workspace array `W` between all of its arrays
//! (`mod_newuoa.f:52-65`) and passes overlapping pieces of it to the
//! subroutines. This transcription keeps that storage association: every
//! array lives in one `&mut [f64]` and is addressed through 0-based offsets
//! with Fortran's column-major layout, so each read and write touches the same
//! element as upstream. Fortran subscripts are kept 1-based in the code.
//!
//! # Undefined upstream behaviour (`DECISIONS #43`)
//!
//! Two paths have no defined result upstream and return an error instead:
//!
//! - `W` shorter than the elements a routine touches (upstream writes past the
//!   end of `W`). MESA's astero driver allocates the `BOBYQA` length, which is
//!   too short for `BIGDEN` (`astero_run_support.f90:671`, Deferred bug
//!   #754); see [`newuoa_with_workspace`].
//! - `MAXFUN < NPT`: upstream returns from the initialization loop and reads
//!   `XOPT` before it is set (`mod_newuoa.f:432-436` then `:689-692`,
//!   Deferred bug #755).
//!
//! Deferred issues for both are pending (package G2 report).

use crate::NumError;

/// Result returned by [`newuoa`] and [`newuoa_with_workspace`].
#[derive(Debug, Clone)]
pub struct NewuoaResult {
    /// Returned vector of variables `X(1:N)` (`mod_newuoa.f:689-694`).
    pub x_min: Vec<f64>,
    /// Value of `F` at the return (`mod_newuoa.f:689-694`): the least
    /// calculated value, unless the final short step found a smaller one, in
    /// which case it is that step's value and [`Self::x_min`] is that step.
    pub f_min: f64,
    /// Number of `CALFUN` calls (`NF` at the return).
    pub num_evals: usize,
}

/// Length of `W` documented by upstream: `(NPT+13)*(NPT+N)+3*N*(N+3)/2`
/// (`mod_newuoa.f:41-42`).
#[must_use]
pub fn newuoa_workspace_len(n: usize, npt: usize) -> usize {
    (npt + 13) * (npt + n) + 3 * n * (n + 3) / 2
}

/// Minimizes `calfun` with `NEWUOA` (`num_newuoa.dek:22-34`), using a
/// workspace of the length upstream documents ([`newuoa_workspace_len`]).
///
/// `IPRINT` is fixed at `0` (no printing), the value every MESA caller uses.
///
/// # Arguments
///
/// - `n`: number of variables `N`.
/// - `npt`: number of interpolation conditions `NPT`, in `[N+2, (N+1)(N+2)/2]`.
/// - `x`: `X(1:N)`, the initial variables on entry and the returned variables
///   on exit (length at least `n`).
/// - `rhobeg`, `rhoend`: initial and final trust-region radii. Upstream does
///   not check them and neither does this routine.
/// - `maxfun`: upper bound `MAXFUN` on the number of `CALFUN` calls.
/// - `max_valid_value`: `F` above this at any interpolation point forces the
///   least-Frobenius-norm model replacement (`mod_newuoa.f:590-599`).
/// - `calfun`: objective `CALFUN(N, X, F)`, called with `X(1:N)`.
///
/// # Errors
///
/// - [`NumError::InvalidParameter`] if `x.len() < n`, if `npt` is outside
///   `[N+2, (N+1)(N+2)/2]` (upstream prints a message and returns,
///   `mod_newuoa.f:47-51`), or on the undefined upstream paths in the module
///   documentation.
/// - Any error returned by `calfun`, unchanged.
#[allow(clippy::missing_panics_doc)]
pub fn newuoa<F>(
    n: usize,
    npt: usize,
    x: &mut [f64],
    rhobeg: f64,
    rhoend: f64,
    maxfun: usize,
    max_valid_value: f64,
    calfun: F,
) -> Result<NewuoaResult, NumError>
where
    F: FnMut(&[f64]) -> Result<f64, NumError>,
{
    let mut w = vec![0.0_f64; newuoa_workspace_len(n, npt)];
    newuoa_with_workspace(
        n,
        npt,
        x,
        rhobeg,
        rhoend,
        maxfun,
        &mut w,
        max_valid_value,
        calfun,
    )
}

/// `do_newuoa` (`mod_newuoa.f:7-78`) with a caller-supplied workspace `W`.
///
/// `W` is partitioned exactly as upstream. Its length only matters where
/// upstream would touch an element past its end; that path returns an error
/// (`DECISIONS #43`). MESA's astero driver passes
/// `(npt+5)*(npt+nvar)+3*nvar*(nvar+5)/2` elements
/// (`astero_run_support.f90:671`), enough for every routine except `BIGDEN`.
///
/// See [`newuoa`] for the other arguments.
///
/// # Errors
///
/// As for [`newuoa`], plus [`NumError::InvalidParameter`] when upstream
/// would write past the end of `w`.
pub fn newuoa_with_workspace<F>(
    n: usize,
    npt: usize,
    x: &mut [f64],
    rhobeg: f64,
    rhoend: f64,
    maxfun: usize,
    w: &mut [f64],
    max_valid_value: f64,
    mut calfun: F,
) -> Result<NewuoaResult, NumError>
where
    F: FnMut(&[f64]) -> Result<f64, NumError>,
{
    if x.len() < n {
        return Err(NumError::InvalidParameter(
            "newuoa: X must hold at least N elements",
        ));
    }
    // mod_newuoa.f:45-51
    let np = n + 1;
    if npt < n + 2 || npt > ((n + 2) * np) / 2 {
        return Err(NumError::InvalidParameter(
            "Return from NEWUOA because NPT is not in the required interval",
        ));
    }
    let nptm = npt - np;
    // mod_newuoa.f:52-65 (1-based starts, converted to 0-based offsets).
    let ndim = npt + n;
    let ixb = 1;
    let ixo = ixb + n;
    let ixn = ixo + n;
    let ixp = ixn + n;
    let ifv = ixp + n * npt;
    let igq = ifv + npt;
    let ihq = igq + n;
    let ipq = ihq + (n * np) / 2;
    let ibmat = ipq + npt;
    let izmat = ibmat + ndim * n;
    let id = izmat + npt * nptm;
    let ivl = id + n;
    let iw = ivl + ndim;
    let lay = Layout {
        xbase: ixb - 1,
        xopt: ixo - 1,
        xnew: ixn - 1,
        xpt: ixp - 1,
        fval: ifv - 1,
        gq: igq - 1,
        hq: ihq - 1,
        pq: ipq - 1,
        bmat: ibmat - 1,
        zmat: izmat - 1,
        d: id - 1,
        vlag: ivl - 1,
        w: iw - 1,
    };
    let (f, nf) = newuob(
        n,
        npt,
        x,
        rhobeg,
        rhoend,
        maxfun,
        w,
        lay,
        ndim,
        &mut calfun,
        max_valid_value,
    )?;
    Ok(NewuoaResult {
        x_min: x[..n].to_vec(),
        f_min: f,
        num_evals: nf,
    })
}

/// 0-based offsets into `W` of the arrays `do_newuoa` passes to `NEWUOB`.
#[derive(Clone, Copy)]
struct Layout {
    xbase: usize,
    xopt: usize,
    xnew: usize,
    xpt: usize,
    fval: usize,
    gq: usize,
    hq: usize,
    pq: usize,
    bmat: usize,
    zmat: usize,
    d: usize,
    vlag: usize,
    w: usize,
}

/// Returns an error when upstream would touch `W` past element `end`
/// (0-based exclusive bound), which has no defined result (`DECISIONS #43`).
fn need_w(w: &[f64], end: usize) -> Result<(), NumError> {
    if end > w.len() {
        // DECISIONS #43: upstream writes past the end of W here (Deferred
        // bug #754 for astero's W length).
        return Err(NumError::InvalidParameter(
            "NEWUOA workspace W is too short: upstream writes past its end \
             here (Deferred bug #754, DECISIONS #43)",
        ));
    }
    Ok(())
}

/// Fortran `DMAX1(A, B)` as gfortran evaluates it.
fn dmax1(a: f64, b: f64) -> f64 {
    if b > a || a.is_nan() {
        b
    } else {
        a
    }
}

/// Fortran `DMIN1(A, B)` as gfortran evaluates it.
fn dmin1(a: f64, b: f64) -> f64 {
    if b < a || a.is_nan() {
        b
    } else {
        a
    }
}

/// Statement labels of `NEWUOB` that are targets of `GOTO`.
#[derive(Clone, Copy)]
enum Lb {
    L50,
    L70,
    L90,
    L100,
    L120,
    L290,
    L310,
    L410,
    L460,
    L490,
    L530,
}

/// `NEWUOB` (`mod_newuoa.f:81-701`). Returns `(F, NF)` at the return.
fn newuob<F>(
    n: usize,
    npt: usize,
    x: &mut [f64],
    rhobeg: f64,
    rhoend: f64,
    maxfun: usize,
    w: &mut [f64],
    o: Layout,
    ndim: usize,
    calfun: &mut F,
    max_valid_value: f64,
) -> Result<(f64, usize), NumError>
where
    F: FnMut(&[f64]) -> Result<f64, NumError>,
{
    let xbase = move |j: usize| o.xbase + j - 1;
    let xopt = move |j: usize| o.xopt + j - 1;
    let xnew = move |j: usize| o.xnew + j - 1;
    let xpt = move |k: usize, j: usize| o.xpt + (k - 1) + (j - 1) * npt;
    let fval = move |k: usize| o.fval + k - 1;
    let gq = move |i: usize| o.gq + i - 1;
    let hq = move |i: usize| o.hq + i - 1;
    let pq = move |k: usize| o.pq + k - 1;
    let bmat = move |i: usize, j: usize| o.bmat + (i - 1) + (j - 1) * ndim;
    let zmat = move |k: usize, j: usize| o.zmat + (k - 1) + (j - 1) * npt;
    let d = move |i: usize| o.d + i - 1;
    let vlag = move |i: usize| o.vlag + i - 1;
    let wk = move |i: usize| o.w + i - 1;

    // mod_newuoa.f:120-127
    let half = 0.5_f64;
    let one = 1.0_f64;
    let tenth = 0.1_f64;
    let zero = 0.0_f64;
    let np = n + 1;
    let nh = (n * np) / 2;
    let nptm = npt - np;
    let nftest = maxfun.max(1);

    // mod_newuoa.f:131-148
    need_w(w, o.zmat + npt * nptm)?;
    for j in 1..=n {
        w[xbase(j)] = x[j - 1];
        for k in 1..=npt {
            w[xpt(k, j)] = zero;
        }
        for i in 1..=ndim {
            w[bmat(i, j)] = zero;
        }
    }
    for ih in 1..=nh {
        w[hq(ih)] = zero;
    }
    for k in 1..=npt {
        w[pq(k)] = zero;
        for j in 1..=nptm {
            w[zmat(k, j)] = zero;
        }
    }

    // mod_newuoa.f:154-157
    let mut rhosq = rhobeg * rhobeg;
    let recip = one / rhosq;
    let reciq = half.sqrt() / rhosq;
    let mut nf: usize = 0;

    // Scalars of NEWUOB. Every one is assigned before it is read on every
    // path, except DIFFC and RATIO, whose first assignment depends on the
    // path taken and which are therefore tracked as `Option`.
    let mut nfm: usize = 0;
    let mut ipt: usize = 0;
    let mut jpt: usize = 0;
    let mut xipt = zero;
    let mut xjpt = zero;
    let mut f = zero;
    let mut fbeg = zero;
    let mut fopt = zero;
    let mut kopt: usize = 0;
    let mut rho = zero;
    let mut delta = zero;
    let mut idz: usize = 1;
    let mut diffa = zero;
    let mut diffb = zero;
    let mut diffc: Option<f64> = None;
    let mut itest: usize = 0;
    let mut xoptsq = zero;
    let mut xopt_set = false;
    let mut nfsav: usize = 0;
    let mut knew: isize = 0;
    let mut crvmin;
    let mut dsq = zero;
    let mut dnorm = zero;
    let mut ratio: Option<f64> = None;
    let mut dstep = zero;
    let mut alpha = zero;
    let mut beta = zero;
    let mut vquad = zero;
    let mut diff = zero;
    let mut fsave = zero;
    let mut ksave: isize = 0;
    let mut temp;
    let mut sum;
    let mut ih: usize;

    let ratio_value = |r: Option<f64>| -> Result<f64, NumError> {
        r.ok_or(NumError::InvalidParameter(
            "NEWUOA read RATIO before setting it (DECISIONS #43)",
        ))
    };

    let mut lbl = Lb::L50;
    loop {
        match lbl {
            Lb::L50 => {
                // mod_newuoa.f:158-191
                nfm = nf;
                nf += 1;
                if nfm <= 2 * n {
                    if nfm >= 1 && nfm <= n {
                        w[xpt(nf, nfm)] = rhobeg;
                    } else if nfm > n {
                        let nfmm = nfm - n;
                        w[xpt(nf, nfmm)] = -rhobeg;
                    }
                } else {
                    let nfmm = nfm - n;
                    let mut itemp = (nfmm - 1) / n;
                    jpt = nfm - itemp * n - n;
                    ipt = jpt + itemp;
                    if ipt > n {
                        itemp = jpt;
                        jpt = ipt - n;
                        ipt = itemp;
                    }
                    xipt = rhobeg;
                    if w[fval(ipt + np)] < w[fval(ipt + 1)] {
                        xipt = -xipt;
                    }
                    xjpt = rhobeg;
                    if w[fval(jpt + np)] < w[fval(jpt + 1)] {
                        xjpt = -xjpt;
                    }
                    w[xpt(nf, ipt)] = xipt;
                    w[xpt(nf, jpt)] = xjpt;
                }
                for j in 1..=n {
                    x[j - 1] = w[xpt(nf, j)] + w[xbase(j)];
                }
                lbl = Lb::L310;
            }
            Lb::L70 => {
                // mod_newuoa.f:192-238
                w[fval(nf)] = f;
                if nf == 1 {
                    fbeg = f;
                    fopt = f;
                    kopt = 1;
                } else if f < fopt {
                    fopt = f;
                    kopt = nf;
                }
                if nfm <= 2 * n {
                    if nfm >= 1 && nfm <= n {
                        w[gq(nfm)] = (f - fbeg) / rhobeg;
                        if npt < nf + n {
                            w[bmat(1, nfm)] = -one / rhobeg;
                            w[bmat(nf, nfm)] = one / rhobeg;
                            w[bmat(npt + nfm, nfm)] = -half * rhosq;
                        }
                    } else if nfm > n {
                        let nfmm = nfm - n;
                        w[bmat(nf - n, nfmm)] = half / rhobeg;
                        w[bmat(nf, nfmm)] = -half / rhobeg;
                        w[zmat(1, nfmm)] = -reciq - reciq;
                        w[zmat(nf - n, nfmm)] = reciq;
                        w[zmat(nf, nfmm)] = reciq;
                        ih = (nfmm * (nfmm + 1)) / 2;
                        temp = (fbeg - f) / rhobeg;
                        w[hq(ih)] = (w[gq(nfmm)] - temp) / rhobeg;
                        w[gq(nfmm)] = half * (w[gq(nfmm)] + temp);
                    }
                } else {
                    let nfmm = nfm - n;
                    ih = (ipt * (ipt - 1)) / 2 + jpt;
                    if xipt < zero {
                        ipt += n;
                    }
                    if xjpt < zero {
                        jpt += n;
                    }
                    w[zmat(1, nfmm)] = recip;
                    w[zmat(nf, nfmm)] = recip;
                    w[zmat(ipt + 1, nfmm)] = -recip;
                    w[zmat(jpt + 1, nfmm)] = -recip;
                    w[hq(ih)] = (fbeg - w[fval(ipt + 1)] - w[fval(jpt + 1)]
                        + f)
                        / (xipt * xjpt);
                }
                if nf < npt {
                    lbl = Lb::L50;
                    continue;
                }
                // mod_newuoa.f:242-252
                rho = rhobeg;
                delta = rho;
                idz = 1;
                diffa = zero;
                diffb = zero;
                itest = 0;
                xoptsq = zero;
                for i in 1..=n {
                    w[xopt(i)] = w[xpt(kopt, i)];
                    xoptsq += w[xopt(i)] * w[xopt(i)];
                }
                xopt_set = true;
                lbl = Lb::L90;
            }
            Lb::L90 => {
                nfsav = nf;
                lbl = Lb::L100;
            }
            Lb::L100 => {
                // mod_newuoa.f:258-274
                knew = 0;
                // TRSAPP uses W(1:4*N) (D, G, HD, HS).
                need_w(w, o.w + 4 * n)?;
                crvmin = trsapp(
                    n,
                    npt,
                    w,
                    o.xopt,
                    o.xpt,
                    o.gq,
                    o.hq,
                    o.pq,
                    delta,
                    o.d,
                    o.w,
                    o.w + np - 1,
                    o.w + np + n - 1,
                    o.w + np + 2 * n - 1,
                );
                dsq = zero;
                for i in 1..=n {
                    dsq += w[d(i)] * w[d(i)];
                }
                dnorm = dmin1(delta, dsq.sqrt());
                if dnorm < half * rho {
                    knew = -1;
                    delta *= tenth;
                    ratio = Some(-1.0);
                    if delta <= 1.5 * rho {
                        delta = rho;
                    }
                    if nf <= nfsav + 2 {
                        lbl = Lb::L460;
                        continue;
                    }
                    temp = 0.125 * crvmin * rho * rho;
                    let diffc_v = diffc.ok_or(NumError::InvalidParameter(
                        "NEWUOA read DIFFC before setting it (DECISIONS #43)",
                    ))?;
                    if temp <= dmax1(dmax1(diffa, diffb), diffc_v) {
                        lbl = Lb::L460;
                        continue;
                    }
                    lbl = Lb::L490;
                    continue;
                }
                lbl = Lb::L120;
            }
            Lb::L120 => {
                // mod_newuoa.f:279-353
                if dsq <= 1.0e-3 * xoptsq {
                    need_w(w, o.w + 2 * npt)?;
                    let tempq = 0.25 * xoptsq;
                    for k in 1..=npt {
                        sum = zero;
                        for i in 1..=n {
                            sum += w[xpt(k, i)] * w[xopt(i)];
                        }
                        temp = w[pq(k)] * sum;
                        sum -= half * xoptsq;
                        w[wk(npt + k)] = sum;
                        for i in 1..=n {
                            w[gq(i)] += temp * w[xpt(k, i)];
                            w[xpt(k, i)] -= half * w[xopt(i)];
                            w[vlag(i)] = w[bmat(k, i)];
                            w[wk(i)] = sum * w[xpt(k, i)] + tempq * w[xopt(i)];
                            let ip = npt + i;
                            for j in 1..=i {
                                w[bmat(ip, j)] = w[bmat(ip, j)]
                                    + w[vlag(i)] * w[wk(j)]
                                    + w[wk(i)] * w[vlag(j)];
                            }
                        }
                    }
                    for k in 1..=nptm {
                        let mut sumz = zero;
                        for i in 1..=npt {
                            sumz += w[zmat(i, k)];
                            w[wk(i)] = w[wk(npt + i)] * w[zmat(i, k)];
                        }
                        for j in 1..=n {
                            sum = tempq * sumz * w[xopt(j)];
                            for i in 1..=npt {
                                sum += w[wk(i)] * w[xpt(i, j)];
                            }
                            w[vlag(j)] = sum;
                            if k < idz {
                                sum = -sum;
                            }
                            for i in 1..=npt {
                                w[bmat(i, j)] += sum * w[zmat(i, k)];
                            }
                        }
                        for i in 1..=n {
                            let ip = i + npt;
                            temp = w[vlag(i)];
                            if k < idz {
                                temp = -temp;
                            }
                            for j in 1..=i {
                                w[bmat(ip, j)] += temp * w[vlag(j)];
                            }
                        }
                    }
                    ih = 0;
                    for j in 1..=n {
                        w[wk(j)] = zero;
                        for k in 1..=npt {
                            w[wk(j)] += w[pq(k)] * w[xpt(k, j)];
                            w[xpt(k, j)] -= half * w[xopt(j)];
                        }
                        for i in 1..=j {
                            ih += 1;
                            if i < j {
                                w[gq(j)] += w[hq(ih)] * w[xopt(i)];
                            }
                            w[gq(i)] += w[hq(ih)] * w[xopt(j)];
                            w[hq(ih)] = w[hq(ih)]
                                + w[wk(i)] * w[xopt(j)]
                                + w[xopt(i)] * w[wk(j)];
                            w[bmat(npt + i, j)] = w[bmat(npt + j, i)];
                        }
                    }
                    for j in 1..=n {
                        w[xbase(j)] += w[xopt(j)];
                        w[xopt(j)] = zero;
                    }
                    xoptsq = zero;
                }
                // mod_newuoa.f:359-361
                if knew > 0 {
                    alpha = biglag(
                        n,
                        npt,
                        w,
                        o.xopt,
                        o.xpt,
                        o.bmat,
                        o.zmat,
                        idz,
                        ndim,
                        knew as usize,
                        dstep,
                        o.d,
                        o.vlag,
                        o.vlag + npt,
                        o.w,
                        o.w + np - 1,
                        o.w + np + n - 1,
                    );
                }
                // mod_newuoa.f:366-411
                need_w(w, o.w + npt)?;
                for k in 1..=npt {
                    let mut suma = zero;
                    let mut sumb = zero;
                    sum = zero;
                    for j in 1..=n {
                        suma += w[xpt(k, j)] * w[d(j)];
                        sumb += w[xpt(k, j)] * w[xopt(j)];
                        sum += w[bmat(k, j)] * w[d(j)];
                    }
                    w[wk(k)] = suma * (half * suma + sumb);
                    w[vlag(k)] = sum;
                }
                beta = zero;
                for k in 1..=nptm {
                    sum = zero;
                    for i in 1..=npt {
                        sum += w[zmat(i, k)] * w[wk(i)];
                    }
                    if k < idz {
                        beta += sum * sum;
                        sum = -sum;
                    } else {
                        beta -= sum * sum;
                    }
                    for i in 1..=npt {
                        w[vlag(i)] += sum * w[zmat(i, k)];
                    }
                }
                let mut bsum = zero;
                let mut dx = zero;
                for j in 1..=n {
                    sum = zero;
                    for i in 1..=npt {
                        sum += w[wk(i)] * w[bmat(i, j)];
                    }
                    bsum += sum * w[d(j)];
                    let jp = npt + j;
                    for k in 1..=n {
                        sum += w[bmat(jp, k)] * w[d(k)];
                    }
                    w[vlag(jp)] = sum;
                    bsum += sum * w[d(j)];
                    dx += w[d(j)] * w[xopt(j)];
                }
                beta = dx * dx + dsq * (xoptsq + dx + dx + half * dsq) + beta
                    - bsum;
                w[vlag(kopt)] += one;
                // mod_newuoa.f:417-423
                if knew > 0 {
                    let vk = w[vlag(knew as usize)];
                    temp = one + alpha * beta / (vk * vk);
                    if temp.abs() <= 0.8 {
                        // BIGDEN uses W(1:11*NDIM) (W, WVEC, PROD).
                        need_w(w, o.w + 11 * ndim)?;
                        beta = bigden(
                            n,
                            npt,
                            w,
                            o.xopt,
                            o.xpt,
                            o.bmat,
                            o.zmat,
                            idz,
                            ndim,
                            kopt,
                            knew as usize,
                            o.d,
                            o.w,
                            o.vlag,
                            o.xnew,
                            o.w + ndim,
                            o.w + 6 * ndim,
                        );
                    }
                }
                lbl = Lb::L290;
            }
            Lb::L290 => {
                // mod_newuoa.f:427-431
                for i in 1..=n {
                    w[xnew(i)] = w[xopt(i)] + w[d(i)];
                    x[i - 1] = w[xbase(i)] + w[xnew(i)];
                }
                nf += 1;
                lbl = Lb::L310;
            }
            Lb::L310 => {
                // mod_newuoa.f:432-444
                if nf > nftest {
                    nf -= 1;
                    lbl = Lb::L530;
                    continue;
                }
                f = calfun(&x[..n])?;
                if nf <= npt {
                    lbl = Lb::L70;
                    continue;
                }
                if knew == -1 {
                    lbl = Lb::L530;
                    continue;
                }
                // mod_newuoa.f:449-467
                vquad = zero;
                ih = 0;
                for j in 1..=n {
                    vquad += w[d(j)] * w[gq(j)];
                    for i in 1..=j {
                        ih += 1;
                        temp = w[d(i)] * w[xnew(j)] + w[d(j)] * w[xopt(i)];
                        if i == j {
                            temp *= half;
                        }
                        vquad += temp * w[hq(ih)];
                    }
                }
                for k in 1..=npt {
                    vquad += w[pq(k)] * w[wk(k)];
                }
                diff = f - fopt - vquad;
                diffc = Some(diffb);
                diffb = diffa;
                diffa = diff.abs();
                if dnorm > rho {
                    nfsav = nf;
                }
                // mod_newuoa.f:473-483
                fsave = fopt;
                if f < fopt {
                    fopt = f;
                    xoptsq = zero;
                    for i in 1..=n {
                        w[xopt(i)] = w[xnew(i)];
                        xoptsq += w[xopt(i)] * w[xopt(i)];
                    }
                }
                ksave = knew;
                if knew > 0 {
                    lbl = Lb::L410;
                    continue;
                }
                // mod_newuoa.f:487-500
                if vquad >= zero {
                    lbl = Lb::L530;
                    continue;
                }
                let r = (f - fsave) / vquad;
                ratio = Some(r);
                if r <= tenth {
                    delta = half * dnorm;
                } else if r <= 0.7 {
                    delta = dmax1(half * delta, dnorm);
                } else {
                    delta = dmax1(half * delta, dnorm + dnorm);
                }
                if delta <= 1.5 * rho {
                    delta = rho;
                }
                // mod_newuoa.f:504-529
                let rq = dmax1(tenth * delta, rho);
                rhosq = rq * rq;
                let mut ktemp: usize = 0;
                let mut detrat = zero;
                if f >= fsave {
                    ktemp = kopt;
                    detrat = one;
                }
                for k in 1..=npt {
                    let mut hdiag = zero;
                    for j in 1..=nptm {
                        temp = one;
                        if j < idz {
                            temp = -one;
                        }
                        let z = w[zmat(k, j)];
                        hdiag += temp * (z * z);
                    }
                    let vk = w[vlag(k)];
                    temp = (beta * hdiag + vk * vk).abs();
                    let mut distsq = zero;
                    for j in 1..=n {
                        let dj = w[xpt(k, j)] - w[xopt(j)];
                        distsq += dj * dj;
                    }
                    if distsq > rhosq {
                        temp = temp
                            * (distsq / rhosq)
                            * (distsq / rhosq)
                            * (distsq / rhosq);
                    }
                    if temp > detrat && k != ktemp {
                        detrat = temp;
                        knew = k as isize;
                    }
                }
                if knew == 0 {
                    lbl = Lb::L460;
                    continue;
                }
                lbl = Lb::L410;
            }
            Lb::L410 => {
                // mod_newuoa.f:535-545
                need_w(w, o.w + ndim)?;
                let kn = knew as usize;
                update(
                    n, npt, w, o.bmat, o.zmat, &mut idz, ndim, o.vlag, beta,
                    kn, o.w,
                );
                w[fval(kn)] = f;
                ih = 0;
                for i in 1..=n {
                    temp = w[pq(kn)] * w[xpt(kn, i)];
                    for j in 1..=i {
                        ih += 1;
                        w[hq(ih)] += temp * w[xpt(kn, j)];
                    }
                }
                w[pq(kn)] = zero;
                // mod_newuoa.f:550-562
                for j in 1..=nptm {
                    temp = diff * w[zmat(kn, j)];
                    if j < idz {
                        temp = -temp;
                    }
                    for k in 1..=npt {
                        w[pq(k)] += temp * w[zmat(k, j)];
                    }
                }
                let mut gqsq = zero;
                for i in 1..=n {
                    w[gq(i)] += diff * w[bmat(kn, i)];
                    gqsq += w[gq(i)] * w[gq(i)];
                    w[xpt(kn, i)] = w[xnew(i)];
                }
                // mod_newuoa.f:568-623
                if ksave == 0 && delta == rho {
                    if ratio_value(ratio)?.abs() > 1.0e-2 {
                        itest = 0;
                    } else {
                        for k in 1..=npt {
                            w[vlag(k)] = w[fval(k)] - w[fval(kopt)];
                        }
                        let mut gisq = zero;
                        for i in 1..=n {
                            sum = zero;
                            for k in 1..=npt {
                                sum += w[bmat(k, i)] * w[vlag(k)];
                            }
                            gisq += sum * sum;
                            w[wk(i)] = sum;
                        }
                        itest += 1;
                        if gqsq < 1.0e2 * gisq {
                            itest = 0;
                        }
                        let mut do_replace = itest >= 3;
                        if !do_replace {
                            for k in 1..=npt {
                                if w[fval(k)] > max_valid_value {
                                    do_replace = true;
                                    break;
                                }
                            }
                        }
                        if do_replace {
                            for i in 1..=n {
                                w[gq(i)] = w[wk(i)];
                            }
                            for ihh in 1..=nh {
                                w[hq(ihh)] = zero;
                            }
                            for j in 1..=nptm {
                                w[wk(j)] = zero;
                                for k in 1..=npt {
                                    w[wk(j)] += w[vlag(k)] * w[zmat(k, j)];
                                }
                                if j < idz {
                                    w[wk(j)] = -w[wk(j)];
                                }
                            }
                            for k in 1..=npt {
                                w[pq(k)] = zero;
                                for j in 1..=nptm {
                                    w[pq(k)] += w[zmat(k, j)] * w[wk(j)];
                                }
                            }
                            itest = 0;
                        }
                    }
                }
                if f < fsave {
                    kopt = kn;
                }
                // mod_newuoa.f:630-636
                if f <= fsave + tenth * vquad {
                    lbl = Lb::L100;
                    continue;
                }
                if ksave > 0 {
                    lbl = Lb::L100;
                    continue;
                }
                knew = 0;
                lbl = Lb::L460;
            }
            Lb::L460 => {
                // mod_newuoa.f:637-658
                let mut distsq = 4.0 * delta * delta;
                for k in 1..=npt {
                    sum = zero;
                    for j in 1..=n {
                        let dj = w[xpt(k, j)] - w[xopt(j)];
                        sum += dj * dj;
                    }
                    if sum > distsq {
                        knew = k as isize;
                        distsq = sum;
                    }
                }
                if knew > 0 {
                    dstep =
                        dmax1(dmin1(tenth * distsq.sqrt(), half * delta), rho);
                    dsq = dstep * dstep;
                    lbl = Lb::L120;
                    continue;
                }
                if ratio_value(ratio)? > zero {
                    lbl = Lb::L100;
                    continue;
                }
                if dmax1(delta, dnorm) > rho {
                    lbl = Lb::L100;
                    continue;
                }
                lbl = Lb::L490;
            }
            Lb::L490 => {
                // mod_newuoa.f:663-688
                if rho > rhoend {
                    delta = half * rho;
                    let r = rho / rhoend;
                    ratio = Some(r);
                    if r <= 16.0 {
                        rho = rhoend;
                    } else if r <= 250.0 {
                        rho = r.sqrt() * rhoend;
                    } else {
                        rho *= tenth;
                    }
                    delta = dmax1(delta, rho);
                    lbl = Lb::L90;
                    continue;
                }
                if knew == -1 {
                    lbl = Lb::L290;
                    continue;
                }
                lbl = Lb::L530;
            }
            Lb::L530 => {
                // mod_newuoa.f:689-694
                if fopt <= f {
                    if !xopt_set {
                        // DECISIONS #43: MAXFUN < NPT returns from the
                        // initialization loop and reads XOPT before
                        // mod_newuoa.f:250 sets it (Deferred bug #755).
                        return Err(NumError::InvalidParameter(
                            "NEWUOA returned before XOPT was set (MAXFUN < \
                             NPT reads an unset XOPT, Deferred bug #755, \
                             DECISIONS #43)",
                        ));
                    }
                    for i in 1..=n {
                        x[i - 1] = w[xbase(i)] + w[xopt(i)];
                    }
                    f = fopt;
                }
                return Ok((f, nf));
            }
        }
    }
}

/// `BIGDEN` (`mod_newuoa.f:703-1056`): chooses the step `D` from `XOPT` that
/// gives the denominator of the `UPDATE` formula a large modulus when the
/// `KNEW`-th interpolation point moves to `XOPT + D`, and returns the new
/// `BETA`.
///
/// All arrays live in `w` at the given 0-based offsets, column-major as
/// upstream: `XOPT(N)`, `XPT(NPT,N)`, `BMAT(NDIM,N)`, `ZMAT(NPT,NPT-N-1)`,
/// `D(N)`, `W(NDIM)` (`wv`), `VLAG(NDIM)`, `S(N)`, `WVEC(NDIM,5)` and
/// `PROD(NDIM,5)`. `kopt` and `knew` are 1-based. On return `D`, `W`, `VLAG`,
/// `S`, `WVEC` and `PROD` hold upstream's final values.
///
/// # Panics
///
/// Panics if an offset addresses an element outside `w`.
pub fn bigden(
    n: usize,
    npt: usize,
    w: &mut [f64],
    xopt: usize,
    xpt: usize,
    bmat: usize,
    zmat: usize,
    idz: usize,
    ndim: usize,
    kopt: usize,
    knew: usize,
    d: usize,
    wv: usize,
    vlag: usize,
    s: usize,
    wvec: usize,
    prod: usize,
) -> f64 {
    let xopt = move |i: usize| xopt + i - 1;
    let xpt = move |k: usize, j: usize| xpt + (k - 1) + (j - 1) * npt;
    let bmat = move |i: usize, j: usize| bmat + (i - 1) + (j - 1) * ndim;
    let zmat = move |k: usize, j: usize| zmat + (k - 1) + (j - 1) * npt;
    let d = move |i: usize| d + i - 1;
    let wv = move |i: usize| wv + i - 1;
    let vlag = move |i: usize| vlag + i - 1;
    let s = move |i: usize| s + i - 1;
    let wvec = move |k: usize, j: usize| wvec + (k - 1) + (j - 1) * ndim;
    let prod = move |k: usize, j: usize| prod + (k - 1) + (j - 1) * ndim;

    // mod_newuoa.f:734-740
    let half = 0.5_f64;
    let one = 1.0_f64;
    let quart = 0.25_f64;
    let two = 2.0_f64;
    let zero = 0.0_f64;
    let twopi = 8.0 * flexmesa_math::atan(one);
    let nptm = npt - n - 1;
    // DEN, DENEX and PAR are 1-based: element 0 is unused.
    let mut den = [zero; 10];
    let mut denex = [zero; 10];
    let mut par = [zero; 10];

    // mod_newuoa.f:745-755
    for k in 1..=npt {
        w[wv(n + k)] = zero;
    }
    for j in 1..=nptm {
        let mut temp = w[zmat(knew, j)];
        if j < idz {
            temp = -temp;
        }
        for k in 1..=npt {
            w[wv(n + k)] += temp * w[zmat(k, j)];
        }
    }
    let alpha = w[wv(n + knew)];

    // mod_newuoa.f:762-799
    let mut dd = zero;
    let mut ds = zero;
    let mut ss = zero;
    let mut xoptsq = zero;
    for i in 1..=n {
        dd += w[d(i)] * w[d(i)];
        w[s(i)] = w[xpt(knew, i)] - w[xopt(i)];
        ds += w[d(i)] * w[s(i)];
        ss += w[s(i)] * w[s(i)];
        xoptsq += w[xopt(i)] * w[xopt(i)];
    }
    if ds * ds > 0.99 * dd * ss {
        let mut ksav = knew;
        let mut dtest = ds * ds / ss;
        for k in 1..=npt {
            if k != kopt {
                let mut dstemp = zero;
                let mut sstemp = zero;
                for i in 1..=n {
                    let diff = w[xpt(k, i)] - w[xopt(i)];
                    dstemp += w[d(i)] * diff;
                    sstemp += diff * diff;
                }
                if dstemp * dstemp / sstemp < dtest {
                    ksav = k;
                    dtest = dstemp * dstemp / sstemp;
                    ds = dstemp;
                    ss = sstemp;
                }
            }
        }
        for i in 1..=n {
            w[s(i)] = w[xpt(ksav, i)] - w[xopt(i)];
        }
    }
    let mut ssden = dd * ss - ds * ds;
    let mut iterc: usize = 0;
    let mut densav = zero;
    let mut beta;

    loop {
        // mod_newuoa.f:804-812 (label 70)
        iterc += 1;
        let mut temp = one / ssden.sqrt();
        let mut xoptd = zero;
        let mut xopts = zero;
        for i in 1..=n {
            w[s(i)] = temp * (dd * w[s(i)] - ds * w[d(i)]);
            xoptd += w[xopt(i)] * w[d(i)];
            xopts += w[xopt(i)] * w[s(i)];
        }
        // mod_newuoa.f:816-825
        let mut tempa = half * xoptd * xoptd;
        let mut tempb = half * xopts * xopts;
        den[1] = dd * (xoptsq + half * dd) + tempa + tempb;
        den[2] = two * xoptd * dd;
        den[3] = two * xopts * dd;
        den[4] = tempa - tempb;
        den[5] = xoptd * xopts;
        for v in den.iter_mut().skip(6) {
            *v = zero;
        }
        // mod_newuoa.f:829-851
        let mut tempc;
        for k in 1..=npt {
            tempa = zero;
            tempb = zero;
            tempc = zero;
            for i in 1..=n {
                tempa += w[xpt(k, i)] * w[d(i)];
                tempb += w[xpt(k, i)] * w[s(i)];
                tempc += w[xpt(k, i)] * w[xopt(i)];
            }
            w[wvec(k, 1)] = quart * (tempa * tempa + tempb * tempb);
            w[wvec(k, 2)] = tempa * tempc;
            w[wvec(k, 3)] = tempb * tempc;
            w[wvec(k, 4)] = quart * (tempa * tempa - tempb * tempb);
            w[wvec(k, 5)] = half * tempa * tempb;
        }
        for i in 1..=n {
            let ip = i + npt;
            w[wvec(ip, 1)] = zero;
            w[wvec(ip, 2)] = w[d(i)];
            w[wvec(ip, 3)] = w[s(i)];
            w[wvec(ip, 4)] = zero;
            w[wvec(ip, 5)] = zero;
        }
        // mod_newuoa.f:855-887
        for jc in 1..=5 {
            let mut nw = npt;
            if jc == 2 || jc == 3 {
                nw = ndim;
            }
            for k in 1..=npt {
                w[prod(k, jc)] = zero;
            }
            for j in 1..=nptm {
                let mut sum = zero;
                for k in 1..=npt {
                    sum += w[zmat(k, j)] * w[wvec(k, jc)];
                }
                if j < idz {
                    sum = -sum;
                }
                for k in 1..=npt {
                    w[prod(k, jc)] += sum * w[zmat(k, j)];
                }
            }
            if nw == ndim {
                for k in 1..=npt {
                    let mut sum = zero;
                    for j in 1..=n {
                        sum += w[bmat(k, j)] * w[wvec(npt + j, jc)];
                    }
                    w[prod(k, jc)] += sum;
                }
            }
            for j in 1..=n {
                let mut sum = zero;
                for i in 1..=nw {
                    sum += w[bmat(i, j)] * w[wvec(i, jc)];
                }
                w[prod(npt + j, jc)] = sum;
            }
        }
        // mod_newuoa.f:891-916
        for k in 1..=ndim {
            let mut sum = zero;
            for i in 1..=5 {
                par[i] = half * w[prod(k, i)] * w[wvec(k, i)];
                sum += par[i];
            }
            den[1] = den[1] - par[1] - sum;
            tempa =
                w[prod(k, 1)] * w[wvec(k, 2)] + w[prod(k, 2)] * w[wvec(k, 1)];
            tempb =
                w[prod(k, 2)] * w[wvec(k, 4)] + w[prod(k, 4)] * w[wvec(k, 2)];
            tempc =
                w[prod(k, 3)] * w[wvec(k, 5)] + w[prod(k, 5)] * w[wvec(k, 3)];
            den[2] = den[2] - tempa - half * (tempb + tempc);
            den[6] -= half * (tempb - tempc);
            tempa =
                w[prod(k, 1)] * w[wvec(k, 3)] + w[prod(k, 3)] * w[wvec(k, 1)];
            tempb =
                w[prod(k, 2)] * w[wvec(k, 5)] + w[prod(k, 5)] * w[wvec(k, 2)];
            tempc =
                w[prod(k, 3)] * w[wvec(k, 4)] + w[prod(k, 4)] * w[wvec(k, 3)];
            den[3] = den[3] - tempa - half * (tempb - tempc);
            den[7] -= half * (tempb + tempc);
            tempa =
                w[prod(k, 1)] * w[wvec(k, 4)] + w[prod(k, 4)] * w[wvec(k, 1)];
            den[4] = den[4] - tempa - par[2] + par[3];
            tempa =
                w[prod(k, 1)] * w[wvec(k, 5)] + w[prod(k, 5)] * w[wvec(k, 1)];
            tempb =
                w[prod(k, 2)] * w[wvec(k, 3)] + w[prod(k, 3)] * w[wvec(k, 2)];
            den[5] = den[5] - tempa - half * tempb;
            den[8] = den[8] - par[4] + par[5];
            tempa =
                w[prod(k, 4)] * w[wvec(k, 5)] + w[prod(k, 5)] * w[wvec(k, 4)];
            den[9] -= half * tempa;
        }
        // mod_newuoa.f:920-941
        let mut sum = zero;
        for i in 1..=5 {
            let p = w[prod(knew, i)];
            par[i] = half * (p * p);
            sum += par[i];
        }
        let pk = |w: &[f64], i: usize| w[prod(knew, i)];
        denex[1] = alpha * den[1] + par[1] + sum;
        tempa = two * pk(w, 1) * pk(w, 2);
        tempb = pk(w, 2) * pk(w, 4);
        tempc = pk(w, 3) * pk(w, 5);
        denex[2] = alpha * den[2] + tempa + tempb + tempc;
        denex[6] = alpha * den[6] + tempb - tempc;
        tempa = two * pk(w, 1) * pk(w, 3);
        tempb = pk(w, 2) * pk(w, 5);
        tempc = pk(w, 3) * pk(w, 4);
        denex[3] = alpha * den[3] + tempa + tempb - tempc;
        denex[7] = alpha * den[7] + tempb + tempc;
        tempa = two * pk(w, 1) * pk(w, 4);
        denex[4] = alpha * den[4] + tempa + par[2] - par[3];
        tempa = two * pk(w, 1) * pk(w, 5);
        denex[5] = alpha * den[5] + tempa + pk(w, 2) * pk(w, 3);
        denex[8] = alpha * den[8] + par[4] - par[5];
        denex[9] = alpha * den[9] + pk(w, 4) * pk(w, 5);
        // mod_newuoa.f:945-981
        sum = denex[1] + denex[2] + denex[4] + denex[6] + denex[8];
        let denold = sum;
        let mut denmax = sum;
        let mut isave: usize = 0;
        let iu: usize = 49;
        temp = twopi / ((iu + 1) as f64);
        par[1] = one;
        for i in 1..=iu {
            let angle = (i as f64) * temp;
            par[2] = flexmesa_math::cos(angle);
            par[3] = flexmesa_math::sin(angle);
            for j in [4_usize, 6, 8] {
                par[j] = par[2] * par[j - 2] - par[3] * par[j - 1];
                par[j + 1] = par[2] * par[j - 1] + par[3] * par[j - 2];
            }
            let sumold = sum;
            sum = zero;
            for j in 1..=9 {
                sum += denex[j] * par[j];
            }
            if sum.abs() > denmax.abs() {
                denmax = sum;
                isave = i;
                tempa = sumold;
            } else if i == isave + 1 {
                tempb = sum;
            }
        }
        if isave == 0 {
            tempa = sum;
        }
        if isave == iu {
            tempb = denold;
        }
        let mut step = zero;
        if tempa != tempb {
            tempa -= denmax;
            tempb -= denmax;
            step = half * (tempa - tempb) / (tempa + tempb);
        }
        let angle = temp * ((isave as f64) + step);
        // mod_newuoa.f:986-1014
        par[2] = flexmesa_math::cos(angle);
        par[3] = flexmesa_math::sin(angle);
        for j in [4_usize, 6, 8] {
            par[j] = par[2] * par[j - 2] - par[3] * par[j - 1];
            par[j + 1] = par[2] * par[j - 1] + par[3] * par[j - 2];
        }
        beta = zero;
        denmax = zero;
        for j in 1..=9 {
            beta += den[j] * par[j];
            denmax += denex[j] * par[j];
        }
        for k in 1..=ndim {
            w[vlag(k)] = zero;
            for j in 1..=5 {
                w[vlag(k)] += w[prod(k, j)] * par[j];
            }
        }
        let tau = w[vlag(knew)];
        dd = zero;
        tempa = zero;
        tempb = zero;
        for i in 1..=n {
            w[d(i)] = par[2] * w[d(i)] + par[3] * w[s(i)];
            w[wv(i)] = w[xopt(i)] + w[d(i)];
            dd += w[d(i)] * w[d(i)];
            tempa += w[d(i)] * w[wv(i)];
            tempb += w[wv(i)] * w[wv(i)];
        }
        // mod_newuoa.f:1015-1018
        if iterc >= n {
            break;
        }
        if iterc > 1 {
            densav = dmax1(densav, denold);
        }
        if denmax.abs() <= 1.1 * densav.abs() {
            break;
        }
        densav = denmax;
        // mod_newuoa.f:1023-1044
        for i in 1..=n {
            temp = tempa * w[xopt(i)] + tempb * w[d(i)] - w[vlag(npt + i)];
            w[s(i)] = tau * w[bmat(knew, i)] + alpha * temp;
        }
        for k in 1..=npt {
            sum = zero;
            for j in 1..=n {
                sum += w[xpt(k, j)] * w[wv(j)];
            }
            temp = (tau * w[wv(n + k)] - alpha * w[vlag(k)]) * sum;
            for i in 1..=n {
                w[s(i)] += temp * w[xpt(k, i)];
            }
        }
        ss = zero;
        ds = zero;
        for i in 1..=n {
            ss += w[s(i)] * w[s(i)];
            ds += w[d(i)] * w[s(i)];
        }
        ssden = dd * ss - ds * ds;
        if ssden < 1.0e-8 * dd * ss {
            break;
        }
    }
    // mod_newuoa.f:1048-1054 (label 340)
    for k in 1..=ndim {
        w[wv(k)] = zero;
        for j in 1..=5 {
            w[wv(k)] += w[wvec(k, j)] * par[j];
        }
    }
    w[vlag(kopt)] += one;
    beta
}

/// `BIGLAG` (`mod_newuoa.f:1059-1247`): step `D` of length `DELTA` from
/// `XOPT` that attempts to maximize the modulus of the `KNEW`-th Lagrange
/// function. Returns `ALPHA`, the `KNEW`-th diagonal element of `H`.
fn biglag(
    n: usize,
    npt: usize,
    w: &mut [f64],
    xopt: usize,
    xpt: usize,
    bmat: usize,
    zmat: usize,
    idz: usize,
    ndim: usize,
    knew: usize,
    delta: f64,
    d: usize,
    hcol: usize,
    gc: usize,
    gd: usize,
    s: usize,
    wv: usize,
) -> f64 {
    let xopt = move |i: usize| xopt + i - 1;
    let xpt = move |k: usize, j: usize| xpt + (k - 1) + (j - 1) * npt;
    let bmat = move |i: usize, j: usize| bmat + (i - 1) + (j - 1) * ndim;
    let zmat = move |k: usize, j: usize| zmat + (k - 1) + (j - 1) * npt;
    let d = move |i: usize| d + i - 1;
    let hcol = move |i: usize| hcol + i - 1;
    let gc = move |i: usize| gc + i - 1;
    let gd = move |i: usize| gd + i - 1;
    let s = move |i: usize| s + i - 1;
    let wv = move |i: usize| wv + i - 1;

    // mod_newuoa.f:1082-1103
    let half = 0.5_f64;
    let one = 1.0_f64;
    let zero = 0.0_f64;
    let twopi = 8.0 * flexmesa_math::atan(one);
    let delsq = delta * delta;
    let nptm = npt - n - 1;
    let mut iterc: usize = 0;
    for k in 1..=npt {
        w[hcol(k)] = zero;
    }
    for j in 1..=nptm {
        let mut temp = w[zmat(knew, j)];
        if j < idz {
            temp = -temp;
        }
        for k in 1..=npt {
            w[hcol(k)] += temp * w[zmat(k, j)];
        }
    }
    let alpha = w[hcol(knew)];

    // mod_newuoa.f:1108-1128
    let mut dd = zero;
    for i in 1..=n {
        w[d(i)] = w[xpt(knew, i)] - w[xopt(i)];
        w[gc(i)] = w[bmat(knew, i)];
        w[gd(i)] = zero;
        dd += w[d(i)] * w[d(i)];
    }
    for k in 1..=npt {
        let mut temp = zero;
        let mut sum = zero;
        for j in 1..=n {
            temp += w[xpt(k, j)] * w[xopt(j)];
            sum += w[xpt(k, j)] * w[d(j)];
        }
        temp *= w[hcol(k)];
        sum *= w[hcol(k)];
        for i in 1..=n {
            w[gc(i)] += temp * w[xpt(k, i)];
            w[gd(i)] += sum * w[xpt(k, i)];
        }
    }

    // mod_newuoa.f:1133-1151
    let mut gg = zero;
    let mut sp = zero;
    let mut dhd = zero;
    for i in 1..=n {
        gg += w[gc(i)] * w[gc(i)];
        sp += w[d(i)] * w[gc(i)];
        dhd += w[d(i)] * w[gd(i)];
    }
    let mut scale = delta / dd.sqrt();
    if sp * dhd < zero {
        scale = -scale;
    }
    let mut temp = zero;
    if sp * sp > 0.99 * dd * gg {
        temp = one;
    }
    let mut tau = scale * (sp.abs() + half * scale * dhd.abs());
    if gg * delsq < 0.01 * tau * tau {
        temp = one;
    }
    for i in 1..=n {
        w[d(i)] *= scale;
        w[gd(i)] *= scale;
        w[s(i)] = w[gc(i)] + temp * w[gd(i)];
    }

    loop {
        // mod_newuoa.f:1157-1172 (label 80)
        iterc += 1;
        dd = zero;
        sp = zero;
        let mut ss = zero;
        for i in 1..=n {
            dd += w[d(i)] * w[d(i)];
            sp += w[d(i)] * w[s(i)];
            ss += w[s(i)] * w[s(i)];
        }
        temp = dd * ss - sp * sp;
        if temp <= 1.0e-8 * dd * ss {
            return alpha;
        }
        let denom = temp.sqrt();
        for i in 1..=n {
            w[s(i)] = (dd * w[s(i)] - sp * w[d(i)]) / denom;
            w[wv(i)] = zero;
        }
        // mod_newuoa.f:1177-1200
        for k in 1..=npt {
            let mut sum = zero;
            for j in 1..=n {
                sum += w[xpt(k, j)] * w[s(j)];
            }
            sum *= w[hcol(k)];
            for i in 1..=n {
                w[wv(i)] += sum * w[xpt(k, i)];
            }
        }
        let mut cf1 = zero;
        let mut cf2 = zero;
        let mut cf3 = zero;
        let mut cf4 = zero;
        let mut cf5 = zero;
        for i in 1..=n {
            cf1 += w[s(i)] * w[wv(i)];
            cf2 += w[d(i)] * w[gc(i)];
            cf3 += w[s(i)] * w[gc(i)];
            cf4 += w[d(i)] * w[gd(i)];
            cf5 += w[s(i)] * w[gd(i)];
        }
        cf1 *= half;
        cf4 = half * cf4 - cf1;
        // mod_newuoa.f:1204-1232
        let taubeg = cf1 + cf2 + cf4;
        let mut taumax = taubeg;
        let mut tauold = taubeg;
        let mut isave: usize = 0;
        let iu: usize = 49;
        temp = twopi / ((iu + 1) as f64);
        let mut tempa = zero;
        let mut tempb = zero;
        for i in 1..=iu {
            let angle = (i as f64) * temp;
            let cth = flexmesa_math::cos(angle);
            let sth = flexmesa_math::sin(angle);
            tau = cf1 + (cf2 + cf4 * cth) * cth + (cf3 + cf5 * cth) * sth;
            if tau.abs() > taumax.abs() {
                taumax = tau;
                isave = i;
                tempa = tauold;
            } else if i == isave + 1 {
                tempb = tau;
            }
            tauold = tau;
        }
        if isave == 0 {
            tempa = tau;
        }
        if isave == iu {
            tempb = taubeg;
        }
        let mut step = zero;
        if tempa != tempb {
            tempa -= taumax;
            tempb -= taumax;
            step = half * (tempa - tempb) / (tempa + tempb);
        }
        let angle = temp * ((isave as f64) + step);
        // mod_newuoa.f:1236-1245
        let cth = flexmesa_math::cos(angle);
        let sth = flexmesa_math::sin(angle);
        tau = cf1 + (cf2 + cf4 * cth) * cth + (cf3 + cf5 * cth) * sth;
        for i in 1..=n {
            w[d(i)] = cth * w[d(i)] + sth * w[s(i)];
            w[gd(i)] = cth * w[gd(i)] + sth * w[wv(i)];
            w[s(i)] = w[gc(i)] + w[gd(i)];
        }
        if tau.abs() <= 1.1 * taubeg.abs() {
            return alpha;
        }
        if iterc >= n {
            return alpha;
        }
    }
}

/// Statement labels of `TRSAPP` that are targets of `GOTO`.
#[derive(Clone, Copy)]
enum Lt {
    L20,
    L40,
    L50,
    L90,
    L120,
    L170,
}

/// `TRSAPP` (`mod_newuoa.f:1249-1467`): truncated conjugate-gradient trial
/// step `STEP` within the trust region, followed by 2D refinements on the
/// boundary. Returns `CRVMIN`.
fn trsapp(
    n: usize,
    npt: usize,
    w: &mut [f64],
    xopt: usize,
    xpt: usize,
    gq: usize,
    hq: usize,
    pq: usize,
    delta: f64,
    step: usize,
    d: usize,
    g: usize,
    hd: usize,
    hs: usize,
) -> f64 {
    let xopt = move |i: usize| xopt + i - 1;
    let xpt = move |k: usize, j: usize| xpt + (k - 1) + (j - 1) * npt;
    let gq = move |i: usize| gq + i - 1;
    let hq = move |i: usize| hq + i - 1;
    let pq = move |k: usize| pq + k - 1;
    let step = move |i: usize| step + i - 1;
    let d = move |i: usize| d + i - 1;
    let g = move |i: usize| g + i - 1;
    let hd = move |i: usize| hd + i - 1;
    let hs = move |i: usize| hs + i - 1;

    // mod_newuoa.f:1271-1281
    let half = 0.5_f64;
    let zero = 0.0_f64;
    let twopi = 8.0 * flexmesa_math::atan(1.0);
    let delsq = delta * delta;
    let mut iterc: usize = 0;
    let itermax = n;
    let mut itersw = itermax;
    for i in 1..=n {
        w[d(i)] = w[xopt(i)];
    }

    // Scalars assigned before every read on every path.
    let mut qred = zero;
    let mut dd = zero;
    let mut ds = zero;
    let mut ss = zero;
    let mut gg = zero;
    let mut ggbeg = zero;
    let mut bstep = zero;
    let mut crvmin = zero;
    let mut sg = zero;
    let mut shs = zero;
    let mut temp;

    let mut lbl = Lt::L170;
    loop {
        match lbl {
            Lt::L20 => {
                // mod_newuoa.f:1285-1299
                qred = zero;
                dd = zero;
                for i in 1..=n {
                    w[step(i)] = zero;
                    w[hs(i)] = zero;
                    w[g(i)] = w[gq(i)] + w[hd(i)];
                    w[d(i)] = -w[g(i)];
                    dd += w[d(i)] * w[d(i)];
                }
                crvmin = zero;
                if dd == zero {
                    return crvmin;
                }
                ds = zero;
                ss = zero;
                gg = dd;
                ggbeg = gg;
                lbl = Lt::L40;
            }
            Lt::L40 => {
                // mod_newuoa.f:1303-1306
                iterc += 1;
                temp = delsq - ss;
                bstep = temp / (ds + (ds * ds + dd * temp).sqrt());
                lbl = Lt::L170;
            }
            Lt::L50 => {
                // mod_newuoa.f:1307-1354
                let mut dhd = zero;
                for j in 1..=n {
                    dhd += w[d(j)] * w[hd(j)];
                }
                let mut alpha = bstep;
                if dhd > zero {
                    temp = dhd / dd;
                    if iterc == 1 {
                        crvmin = temp;
                    }
                    crvmin = dmin1(crvmin, temp);
                    alpha = dmin1(alpha, gg / dhd);
                }
                let qadd = alpha * (gg - half * alpha * dhd);
                qred += qadd;
                let ggsav = gg;
                gg = zero;
                for i in 1..=n {
                    w[step(i)] += alpha * w[d(i)];
                    w[hs(i)] += alpha * w[hd(i)];
                    let gi = w[g(i)] + w[hs(i)];
                    gg += gi * gi;
                }
                if alpha < bstep {
                    if qadd <= 0.01 * qred {
                        return crvmin;
                    }
                    if gg <= 1.0e-4 * ggbeg {
                        return crvmin;
                    }
                    if iterc == itermax {
                        return crvmin;
                    }
                    temp = gg / ggsav;
                    dd = zero;
                    ds = zero;
                    ss = zero;
                    for i in 1..=n {
                        w[d(i)] = temp * w[d(i)] - w[g(i)] - w[hs(i)];
                        dd += w[d(i)] * w[d(i)];
                        ds += w[d(i)] * w[step(i)];
                        ss += w[step(i)] * w[step(i)];
                    }
                    if ds <= zero {
                        return crvmin;
                    }
                    if ss < delsq {
                        lbl = Lt::L40;
                        continue;
                    }
                }
                crvmin = zero;
                itersw = iterc;
                lbl = Lt::L90;
            }
            Lt::L90 => {
                // mod_newuoa.f:1358-1379
                if gg <= 1.0e-4 * ggbeg {
                    return crvmin;
                }
                sg = zero;
                shs = zero;
                for i in 1..=n {
                    sg += w[step(i)] * w[g(i)];
                    shs += w[step(i)] * w[hs(i)];
                }
                let sgk = sg + shs;
                let angtest = sgk / (gg * delsq).sqrt();
                if angtest <= -0.99 {
                    return crvmin;
                }
                iterc += 1;
                temp = (delsq * gg - sgk * sgk).sqrt();
                let tempa = delsq / temp;
                let tempb = sgk / temp;
                for i in 1..=n {
                    w[d(i)] = tempa * (w[g(i)] + w[hs(i)]) - tempb * w[step(i)];
                }
                lbl = Lt::L170;
            }
            Lt::L120 => {
                // mod_newuoa.f:1380-1420
                let mut dg = zero;
                let mut dhd = zero;
                let mut dhs = zero;
                for i in 1..=n {
                    dg += w[d(i)] * w[g(i)];
                    dhd += w[hd(i)] * w[d(i)];
                    dhs += w[hd(i)] * w[step(i)];
                }
                let cf = half * (shs - dhd);
                let qbeg = sg + cf;
                let mut qsav = qbeg;
                let mut qmin = qbeg;
                let mut isave: usize = 0;
                let iu: usize = 49;
                temp = twopi / ((iu + 1) as f64);
                let mut qnew = zero;
                let mut tempa = zero;
                let mut tempb = zero;
                for i in 1..=iu {
                    let angle = (i as f64) * temp;
                    let cth = flexmesa_math::cos(angle);
                    let sth = flexmesa_math::sin(angle);
                    qnew = (sg + cf * cth) * cth + (dg + dhs * cth) * sth;
                    if qnew < qmin {
                        qmin = qnew;
                        isave = i;
                        tempa = qsav;
                    } else if i == isave + 1 {
                        tempb = qnew;
                    }
                    qsav = qnew;
                }
                if isave == 0 {
                    tempa = qnew;
                }
                if isave == iu {
                    tempb = qbeg;
                }
                let mut angle = zero;
                if tempa != tempb {
                    tempa -= qmin;
                    tempb -= qmin;
                    angle = half * (tempa - tempb) / (tempa + tempb);
                }
                angle = temp * ((isave as f64) + angle);
                // mod_newuoa.f:1424-1435
                let cth = flexmesa_math::cos(angle);
                let sth = flexmesa_math::sin(angle);
                let reduc =
                    qbeg - (sg + cf * cth) * cth - (dg + dhs * cth) * sth;
                gg = zero;
                for i in 1..=n {
                    w[step(i)] = cth * w[step(i)] + sth * w[d(i)];
                    w[hs(i)] = cth * w[hs(i)] + sth * w[hd(i)];
                    let gi = w[g(i)] + w[hs(i)];
                    gg += gi * gi;
                }
                qred += reduc;
                let ratio = reduc / qred;
                if iterc < itermax && ratio > 0.01 {
                    lbl = Lt::L90;
                    continue;
                }
                return crvmin;
            }
            Lt::L170 => {
                // mod_newuoa.f:1443-1466
                for i in 1..=n {
                    w[hd(i)] = zero;
                }
                for k in 1..=npt {
                    temp = zero;
                    for j in 1..=n {
                        temp += w[xpt(k, j)] * w[d(j)];
                    }
                    temp *= w[pq(k)];
                    for i in 1..=n {
                        w[hd(i)] += temp * w[xpt(k, i)];
                    }
                }
                let mut ih = 0;
                for j in 1..=n {
                    for i in 1..=j {
                        ih += 1;
                        if i < j {
                            w[hd(j)] += w[hq(ih)] * w[d(i)];
                        }
                        w[hd(i)] += w[hq(ih)] * w[d(j)];
                    }
                }
                lbl = if iterc == 0 {
                    Lt::L20
                } else if iterc <= itersw {
                    Lt::L50
                } else {
                    Lt::L120
                };
            }
        }
    }
}

/// `UPDATE` (`mod_newuoa.f:1469-1581`): updates `BMAT`, `ZMAT` and `IDZ` so
/// that the `KNEW`-th interpolation point can be moved. `VLAG` holds
/// `Theta*Wcheck + e_b` on entry; `W(1:NDIM)` is working space.
fn update(
    n: usize,
    npt: usize,
    w: &mut [f64],
    bmat: usize,
    zmat: usize,
    idz: &mut usize,
    ndim: usize,
    vlag: usize,
    beta: f64,
    knew: usize,
    wv: usize,
) {
    let bmat = move |i: usize, j: usize| bmat + (i - 1) + (j - 1) * ndim;
    let zmat = move |k: usize, j: usize| zmat + (k - 1) + (j - 1) * npt;
    let vlag = move |i: usize| vlag + i - 1;
    let wv = move |i: usize| wv + i - 1;

    // mod_newuoa.f:1481-1502
    let one = 1.0_f64;
    let zero = 0.0_f64;
    let nptm = npt - n - 1;
    let mut jl: usize = 1;
    let mut temp;
    let mut tempa;
    let mut tempb = zero;
    for j in 2..=nptm {
        if j == *idz {
            jl = *idz;
        } else if w[zmat(knew, j)] != zero {
            let a = w[zmat(knew, jl)];
            let b = w[zmat(knew, j)];
            temp = (a * a + b * b).sqrt();
            tempa = w[zmat(knew, jl)] / temp;
            tempb = w[zmat(knew, j)] / temp;
            for i in 1..=npt {
                temp = tempa * w[zmat(i, jl)] + tempb * w[zmat(i, j)];
                w[zmat(i, j)] = tempa * w[zmat(i, j)] - tempb * w[zmat(i, jl)];
                w[zmat(i, jl)] = temp;
            }
            w[zmat(knew, j)] = zero;
        }
    }
    // mod_newuoa.f:1507-1518
    tempa = w[zmat(knew, 1)];
    if *idz >= 2 {
        tempa = -tempa;
    }
    if jl > 1 {
        tempb = w[zmat(knew, jl)];
    }
    for i in 1..=npt {
        w[wv(i)] = tempa * w[zmat(i, 1)];
        if jl > 1 {
            w[wv(i)] += tempb * w[zmat(i, jl)];
        }
    }
    let alpha = w[wv(knew)];
    let tau = w[vlag(knew)];
    let tausq = tau * tau;
    let denom = alpha * beta + tausq;
    w[vlag(knew)] -= one;
    // mod_newuoa.f:1524-1555
    let mut iflag = 0;
    if jl == 1 {
        temp = denom.abs().sqrt();
        tempb = tempa / temp;
        tempa = tau / temp;
        for i in 1..=npt {
            w[zmat(i, 1)] = tempa * w[zmat(i, 1)] - tempb * w[vlag(i)];
        }
        if *idz == 1 && temp < zero {
            *idz = 2;
        }
        if *idz >= 2 && temp >= zero {
            iflag = 1;
        }
    } else {
        let mut ja = 1;
        if beta >= zero {
            ja = jl;
        }
        let jb = jl + 1 - ja;
        temp = w[zmat(knew, jb)] / denom;
        tempa = temp * beta;
        tempb = temp * tau;
        temp = w[zmat(knew, ja)];
        let scala = one / (beta.abs() * temp * temp + tausq).sqrt();
        let scalb = scala * denom.abs().sqrt();
        for i in 1..=npt {
            w[zmat(i, ja)] = scala * (tau * w[zmat(i, ja)] - temp * w[vlag(i)]);
            w[zmat(i, jb)] = scalb
                * (w[zmat(i, jb)] - tempa * w[wv(i)] - tempb * w[vlag(i)]);
        }
        if denom <= zero {
            if beta < zero {
                *idz += 1;
            }
            if beta >= zero {
                iflag = 1;
            }
        }
    }
    // mod_newuoa.f:1560-1567
    if iflag == 1 {
        *idz -= 1;
        for i in 1..=npt {
            temp = w[zmat(i, 1)];
            w[zmat(i, 1)] = w[zmat(i, *idz)];
            w[zmat(i, *idz)] = temp;
        }
    }
    // mod_newuoa.f:1571-1580
    for j in 1..=n {
        let jp = npt + j;
        w[wv(jp)] = w[bmat(knew, j)];
        tempa = (alpha * w[vlag(jp)] - tau * w[wv(jp)]) / denom;
        tempb = (-beta * w[wv(jp)] - tau * w[vlag(jp)]) / denom;
        for i in 1..=jp {
            w[bmat(i, j)] =
                w[bmat(i, j)] + tempa * w[vlag(i)] + tempb * w[wv(i)];
            if i > npt {
                w[bmat(jp, i - npt)] = w[bmat(i, j)];
            }
        }
    }
}
